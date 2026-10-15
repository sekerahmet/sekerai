"""Dikkat anatomisi: eğitilmiş modellerde dikkat nasıl kullanılıyor (kullanıcı, 15 Ekim: "ne kadar detaylı ölçüm yaparsak
o kadar çok fikir verir").

    python scripts/attn_anatomy.py --runs ss_d512_tf_3000 ss_d512_tf_onem ss_d512_tf_odak ss_d512_tf_kapi ss_d512_tf_hepsi

Aynı SS doğrulama pencerelerinde (--windows × 512 token), katman ve head başına:
    önem payı   puanın sorgudan bağımsız (anahtar başına) kısmının softmax'a giden değişimdeki payı.  Satır ortalaması
                softmax'ta yok olduğu için önce çıkarılır; kalan s'_ij'de anahtar bileşeni c_j = ortalama_i s'_ij.
                pay = Σ c_j² / Σ s'_ij² (geçerli i >= j çiftleri).  "qk" yalnız q·k kısmı, "toplam" açık önem dahil.
    çukur       ilk token'a giden dikkat payı (i >= 16 sorguları)
    entropi     -Σ α ln α (nats), top1 = en büyük pay
    önem / odak / kapı (açıksa): dağılım + token türüne göre ortalama + en uç token'lar
Model ağırlıkları runs/<koşu>/checkpoint.pt (yoksa model.pt).  Sonuç runs/_anatomy.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.system import setup_process  # noqa: E402
from rgn.config import from_dict  # noqa: E402
from rgn.data.registry import get_spec  # noqa: E402
from rgn.data.loader import _open  # noqa: E402
from rgn.model import RGNModel  # noqa: E402
from rgn.paths import drive_root, runs_dir  # noqa: E402

FUNC = set("the a an and or but to of in on at for with from by as is was were be been are it he she they we i you "
           "his her their its my your our them him me us that this then so not no".split())


def load(run, device):
    d = runs_dir() / run
    p = d / "checkpoint.pt" if (d / "checkpoint.pt").exists() else d / "model.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = from_dict(ck["config"])
    m = RGNModel(cfg.model)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck.get("step")


def windows(n, T=512):
    spec = get_spec("ss")
    a = np.asarray(_open(drive_root() / spec.valid[0], spec.fmt)[:3_000_000])
    starts = np.linspace(0, len(a) - T - 1, n).astype(int)
    return torch.tensor(np.stack([a[s:s + T].astype(np.int64) for s in starts]))


def kind(tok):
    t = tok.strip()
    if not t:
        return "boşluk/satır"
    if not any(ch.isalnum() for ch in t):
        return "noktalama"
    if t.lower() in FUNC:
        return "işlev kelimesi"
    if t[0].isupper():
        return "büyük harfli"
    return "diğer kelime"


@torch.no_grad()
def analyse(model, idx, enc, device):
    B, T = idx.shape
    idx = idx.to(device)
    rope = model.rope(T)
    x = model.embed(idx)
    rel = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :]
    valid = rel >= 0
    layers = []
    per_tok = defaultdict(lambda: defaultdict(list))                 # ölçüt -> token türü -> değerler
    tok_val = defaultdict(lambda: defaultdict(list))                 # ölçüt -> token -> değerler
    toks = [[enc.decode([t]) for t in row] for row in idx.tolist()] if enc else None
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        u = blk.norm1(x)
        L = {}
        alpha = a.weights(u, rope).float()                           # (B, H, T, T), önem / odak dahil
        ent = -(alpha.clamp_min(1e-12).log() * alpha).sum(-1)       # (B, H, T)
        L["entropi"] = ent[..., 16:].mean().item()
        L["top1"] = alpha.max(-1).values[..., 16:].mean().item()
        L["çukur"] = alpha[..., 16:, 0].mean().item()
        if a.cfg.attn == "softmax":
            q, k, v = a.project(u, rope)
            aux = a._aux(u)
            if "focus" in aux:
                q = q * aux["focus"].unsqueeze(-1)
            if a.Hkv != a.H:
                k = k.repeat_interleave(a.H // a.Hkv, 1)
            s_qk = (q.float() @ k.float().transpose(-2, -1)) / a.dh ** 0.5
            shares = {}
            for name, s in (("qk", s_qk), ("toplam", s_qk + (aux["imp"].repeat_interleave(a.H // a.Hkv, 1).unsqueeze(-2)
                                                            if "imp" in aux else 0))):
                s = s.masked_fill(~valid, 0.0)
                cnt = valid.sum(-1).clamp_min(1)                     # satır başına geçerli j
                s = s - (s.sum(-1, keepdim=True) / cnt.unsqueeze(-1))
                s = s.masked_fill(~valid, 0.0)
                c = s.sum(-2, keepdim=True) / valid.sum(-2).clamp_min(1)     # anahtar bileşeni (B, H, 1, T)
                c = c.expand_as(s).masked_fill(~valid, 0.0)
                shares[name] = ((c ** 2).sum((-1, -2)) / (s ** 2).sum((-1, -2)).clamp_min(1e-12)).mean(0)   # (H,)
            L["önem payı qk"] = shares["qk"].tolist()
            L["önem payı toplam"] = shares["toplam"].tolist()
            for key, nm in (("imp", "önem"), ("focus", "odak"), ("gate", "kapı")):
                if key not in aux:
                    continue
                val = aux[key].float()
                if key == "gate":
                    val = val.mean(-1)                                   # (B, H, T)
                L[nm] = dict(ort=val.mean().item(), std=val.std().item(), min=val.min().item(), max=val.max().item(),
                             head=val.mean((0, 2)).tolist())
                if key == "gate":
                    L[nm]["kapalı<0.5"] = (val < 0.5).float().mean().item()
                if toks:
                    for b in range(B):
                        for t, pv in zip(toks[b], val[b].mean(0).tolist()):     # head ortalaması, token başına
                            per_tok[nm][kind(t)].append(pv)
                            tok_val[nm][t].append(pv)
        layers.append(L)
        x = blk(x, rope)
    summary = {nm: {kd: (float(np.mean(v)), len(v)) for kd, v in d.items()} for nm, d in per_tok.items()}
    extremes = {}
    for nm, d in tok_val.items():
        avg = sorted(((float(np.mean(v)), t, len(v)) for t, v in d.items() if len(v) >= 3 * len(model.blocks)))
        extremes[nm] = dict(en_düşük=[(t, round(m, 3)) for m, t, _ in avg[:8]], en_yüksek=[(t, round(m, 3)) for m, t, _ in avg[-8:][::-1]])
    return layers, summary, extremes


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="dikkat anatomisi")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=16)
    a = ap.parse_args()
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = windows(a.windows)
    res = {}
    for run in a.runs:
        m, step = load(run, device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        layers, summary, extremes = [], {}, {}
        for i in range(0, len(idx), 4):                              # 4'er pencere: N×N matrisler bellekte
            with ctx:
                ly, sm, ex = analyse(m, idx[i:i + 4], enc if i == 0 else None, device)
            layers.append(ly)
            if i == 0:
                summary, extremes = sm, ex
        merged = []
        for li in range(len(layers[0])):
            L = {}
            for key in layers[0][li]:
                vals = [ly[li][key] for ly in layers]
                if isinstance(vals[0], float):
                    L[key] = float(np.mean(vals))
                elif isinstance(vals[0], list):
                    L[key] = np.mean(np.array(vals), 0).tolist()
                else:
                    L[key] = vals[0]
            merged.append(L)
        res[run] = dict(step=step, layers=merged, by_kind=summary, extremes=extremes)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report(res)
    out = runs_dir() / "_anatomy.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nkaydedildi:", out)


def report(res):
    runs = list(res)
    short = {r: r.replace("ss_d512_", "") for r in runs}
    nl = len(next(iter(res.values()))["layers"])
    for key, fmt in (("önem payı qk", "%.3f"), ("önem payı toplam", "%.3f"), ("çukur", "%.3f"), ("entropi", "%.2f"),
                     ("top1", "%.3f")):
        if not any(key in res[r]["layers"][0] for r in runs):
            continue
        print("\n== %s (katman başına%s)" % (key, ", head ortalaması" if "pay" in key else ""))
        print("%-22s" % "koşu" + "".join("%8s" % ("K%d" % (i + 1)) for i in range(nl)) + "%9s" % "ort")
        for r in runs:
            vals = []
            for L in res[r]["layers"]:
                v = L.get(key)
                vals.append(float(np.mean(v)) if isinstance(v, list) else v)
            if vals[0] is None:
                continue
            print("%-22s" % short[r][:22] + "".join("%8s" % (fmt % v) for v in vals) + "%9s" % (fmt % np.mean(vals)))
    for nm in ("önem", "odak", "kapı"):
        rows = [r for r in runs if nm in res[r]["layers"][0]]
        for r in rows:
            print("\n== %s — %s" % (nm, short[r]))
            print("  katman ort: " + "  ".join("%.2f" % L[nm]["ort"] for L in res[r]["layers"]) +
                  "   | aralık %.2f..%.2f" % (min(L[nm]["min"] for L in res[r]["layers"]),
                                               max(L[nm]["max"] for L in res[r]["layers"])))
            if nm == "kapı":
                print("  kapalı (<0,5) oranı: " + "  ".join("%.2f" % L[nm]["kapalı<0.5"] for L in res[r]["layers"]))
            bk = res[r]["by_kind"].get(nm, {})
            print("  token türüne göre: " + "  ".join("%s %.2f (n %d)" % (k, v, n) for k, (v, n) in sorted(bk.items())))
            ex = res[r]["extremes"].get(nm)
            if ex:
                print("  en yüksek: " + ", ".join("%r %.2f" % (t, v) for t, v in ex["en_yüksek"]))
                print("  en düşük : " + ", ".join("%r %.2f" % (t, v) for t, v in ex["en_düşük"]))


if __name__ == "__main__":
    main()
