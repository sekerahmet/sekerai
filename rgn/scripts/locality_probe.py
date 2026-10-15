"""Konum özeti sınaması — nihai tasarımın öngörüleri (kullanıcı, 15 Ekim: "üreteceği ve öğrendikçe çökmeyeceğini nasıl
anlarız").

    python scripts/locality_probe.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_konum

L_τ,i = Σ_j pay_ij e^(−(i−j)/τ)  (yerel τ 2/8/32, genel 4/32/256), MLP kapı girdisine E · (2L − 1).
    dağılım    katman ve τ başına L'nin ortalaması / std / %5 / %95 (sabit kalan girdi = boşa harcanan)
    öngörü 1   E'nin en büyük sütunları hangi (head, τ)'da?  Uzak bakan head'lerin küçük τ'larında olmalı
    öngörü 2   L ortalamayla değiştirilince (konum bilgisi silinir) kayıp artışı: bütün token'lar, zamirler, isimler
    öngörü 3   konum/kapı oranı eğitim teşhis satırında (metrics.jsonl "diag")
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.system import setup_process  # noqa: E402
from rgn.model.attention import locality  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_anatomy import load, windows  # noqa: E402

PRON = {" he", " she", " it", " they", " him", " her", " his", " them", " their", "He", "She", "It", "They",
        " He", " She", " It", " They"}


@torch.no_grad()
def run_model(m, idx, override=None):
    T = idx.shape[1]
    rope = m.rope(T)
    x = m.embed(idx)
    locs = []
    for li, blk in enumerate(m.blocks):
        u = blk.norm1(x)
        a = blk.attn(u, rope)
        if blk.loc_proj is not None:
            a, loc = a
        else:
            at = blk.attn
            q, k, _ = at.project(u, rope)
            taus = at.cfg.locality_tau_local if at.window else at.cfg.locality_tau_global
            loc = locality(q, k, at.window, at.dh ** -0.5, taus)
        locs.append(loc.float())
        extra = None
        if blk.loc_proj is not None:
            L = override[li] if override is not None else loc
            extra = blk.loc_proj((2.0 * L - 1.0).to(x.dtype))
        s = x + a
        x = s + blk.mlp(blk.norm2(s), extra)
    logits = m.lm_head(m.norm_f(x)).float()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1), reduction="none")
    return loss.view(idx.shape[0], -1), locs


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="konum özeti sınaması")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=16)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_all = windows(a.windows)
    nxt = idx_all[:, 1:].flatten().tolist()                         # tahmin edilen token (kayıpla hizalı)
    toks = [enc.decode([t]) for t in nxt]
    prev = [enc.decode([t]) for t in idx_all[:, :-1].flatten().tolist()]
    is_pron = np.array([t in PRON for t in toks])
    is_name = np.array([t.startswith(" ") and t[1:2].isupper() and p.strip() not in {".", "!", "?", '"', ""}
                        for t, p in zip(toks, prev)])               # cümle başı olmayan büyük harfli kelime
    for run in a.runs:
        m, step = load(run, device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        losses, locs = [], None
        for i in range(0, len(idx_all), 4):
            with ctx:
                lo, lc = run_model(m, idx_all[i:i + 4].to(device))
            losses.append(lo.cpu())
            lc = [x[:, :-1].cpu() for x in lc]
            locs = lc if locs is None else [torch.cat([p, q]) for p, q in zip(locs, lc)]
        L0 = torch.cat(losses).flatten().numpy()
        kinds = m.cfg.layer_kinds()
        H = m.cfg.n_heads
        print("\n== %s (adım %s) — ortalama kayıp %.4f  (zamir n %d, isim n %d)" % (run, step, L0.mean(), is_pron.sum(),
                                                                                    is_name.sum()))
        print("  dağılım: katman  τ: ort / std / %5 / %95")
        for li, lc in enumerate(locs):
            taus = m.cfg.locality_tau_local if kinds[li] == "Y" else m.cfg.locality_tau_global
            parts = []
            for ti, t in enumerate(taus):
                A = lc[..., ti * H:(ti + 1) * H].flatten().numpy()
                parts.append("τ%g %.2f/%.2f/%.2f/%.2f" % (t, A.mean(), A.std(), *np.quantile(A, [0.05, 0.95])))
            print("    K%d %s  %s" % (li + 1, kinds[li], "   ".join(parts)))
        if not any(b.loc_proj is not None for b in m.blocks):
            continue
        print("  öngörü 1 — E sütun normları (katman: τ başına head ortalaması; en büyük 3 (head, τ)):")
        for li, b in enumerate(m.blocks):
            taus = m.cfg.locality_tau_local if kinds[li] == "Y" else m.cfg.locality_tau_global
            cn = b.loc_proj.weight.norm(dim=0).view(len(taus), H)
            top = torch.topk(cn.flatten(), 3)
            tops = ", ".join("h%d τ%g %.2f" % (int(i) % H, taus[int(i) // H], v) for v, i in zip(top.values.tolist(),
                                                                                              top.indices.tolist()))
            print("    K%d %s  τ ort: %s   en büyük: %s" % (li + 1, kinds[li],
                                                        " ".join("%.2f" % x for x in cn.mean(1).tolist()), tops))
        means = [lc.mean((0, 1)).to(device) for lc in locs]
        losses2 = []
        for i in range(0, len(idx_all), 4):
            x = idx_all[i:i + 4].to(device)
            ov = [mu.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1) for mu in means]
            with ctx:
                lo, _ = run_model(m, x, override=ov)
            losses2.append(lo.cpu())
        L2 = torch.cat(losses2).flatten().numpy()
        d = L2 - L0
        print("  öngörü 2 — konum bilgisi silinince kayıp artışı: bütün %+.4f | zamir %+.4f | isim %+.4f | diğer %+.4f" % (
            d.mean(), d[is_pron].mean(), d[is_name].mean(), d[~is_pron & ~is_name].mean()))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
