"""Kanıt miktarı sınaması (kullanıcı, 15 Ekim: "kanıt işe yarıyor mu, doğru kurduk mu, nasıl anlarız").

    python scripts/evidence_probe.py --runs ss_d512_tf_yg_epok ss_d512_tf_yg_kanit

kanıt (head başına iki sayı): güç = ln((1/n_i) Σ_j e^puan_ij), odaklanma = güç − ortalama puan ≥ 0
    1. sinyal  kanıt, sonraki-token kaybıyla ilişkili mi?  Katman başına Pearson / Spearman korelasyonu ve kanıt
               çeyreklerine göre ortalama kayıp (en düşük çeyrek -> en yüksek çeyrek).  Kanıtı kullanmayan modelde de
               ölçülür: sinyal eğitimden önce var mı?
    2. kullanım kanıtlı modelde kanıt head ortalamasıyla değiştirilince (bilgi silinir) kayıp ne kadar artıyor?
               E (kanıt -> MLP) ağırlıklarının büyüklüğü.
Aynı doğrulama pencereleri (--windows × 512).  Model ağırlıkları runs/<koşu>/checkpoint.pt (yoksa model.pt).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.system import setup_process  # noqa: E402
from rgn.model.attention import evidence, softmax_lse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_anatomy import load, windows  # noqa: E402


def _r(z):
    return z * torch.rsqrt(1.0 + z * z)


@torch.no_grad()
def run_model(m, idx, override=None):
    """-> (token kaybı (B, T-1), [katman başına kanıt (B, T, H)])."""
    T = idx.shape[1]
    rope = m.rope(T)
    x = m.embed(idx)
    evs = []
    for li, blk in enumerate(m.blocks):
        u = blk.norm1(x)
        a = blk.attn(u, rope)
        if blk.ev_proj is not None:
            a, ev = a
        else:                                                    # kanıtı kullanmayan model: yalnız ölç
            at = blk.attn
            q, k, v = at.project(u, rope)
            _, lse = softmax_lse(q, k, v, at.window, at.dh ** -0.5)
            ev = evidence(q, k, lse, at.window, at.dh ** -0.5)
        evs.append(ev.float())
        extra = None
        if blk.ev_proj is not None:
            e = override[li] if override is not None else ev
            extra = blk.ev_proj(_r(e * blk.ev_scale).to(x.dtype))
        s = x + a
        x = s + blk.mlp(blk.norm2(s), extra)
    logits = m.lm_head(m.norm_f(x)).float()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1), reduction="none")
    return loss.view(idx.shape[0], -1), evs


def _rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="kanıt miktarı sınaması")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=16)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_all = windows(a.windows)
    for run in a.runs:
        m, step = load(run, device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        losses, evs = [], None
        for i in range(0, len(idx_all), 4):
            with ctx:
                lo, ev = run_model(m, idx_all[i:i + 4].to(device))
            losses.append(lo.cpu())
            ev = [e[:, :-1].cpu() for e in ev]                    # kayıpla hizalı (i -> i+1 tahmini)
            evs = ev if evs is None else [torch.cat([p, q]) for p, q in zip(evs, ev)]
        L = torch.cat(losses).flatten().numpy()
        print("\n== %s (adım %s) — ortalama kayıp %.4f" % (run, step, L.mean()))
        H = evs[0].shape[-1] // 2
        for part, sl in (("GÜÇ", slice(0, H)), ("ODAKLANMA", slice(H, 2 * H))):
            print("  %s: katman  ort   Pearson  Spearman   kayıp: en düşük çeyrek -> en yüksek çeyrek" % part)
            for li, e in enumerate(evs):
                E = e[..., sl].mean(-1).flatten().numpy()          # head ortalaması
                p = np.corrcoef(E, L)[0, 1]
                s = np.corrcoef(_rank(E), _rank(L))[0, 1]
                qs = np.quantile(E, [0.25, 0.5, 0.75])
                bins = np.digitize(E, qs)
                ql = [L[bins == b].mean() for b in range(4)]
                print("    K%d %8.3f %9.3f %9.3f    %s" % (li + 1, E.mean(), p, s, "  ".join("%.3f" % x for x in ql)))
        if any(b.ev_proj is not None for b in m.blocks):
            means = [e.mean((0, 1)).to(device) for e in evs]       # head başına ortalama kanıt
            losses2 = []
            for i in range(0, len(idx_all), 4):
                x = idx_all[i:i + 4].to(device)
                ov = [mu.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1) for mu in means]
                with ctx:
                    lo, _ = run_model(m, x, override=ov)
                losses2.append(lo.cpu())
            L2 = torch.cat(losses2).flatten().numpy()
            print("  kullanım: kanıt ortalamayla değiştirilince kayıp %.4f -> %.4f (%+.4f)" % (L.mean(), L2.mean(),
                                                                                           L2.mean() - L.mean()))
            print("  E ağırlık normu (katman başına): " + "  ".join("%.3f" % b.ev_proj.weight.norm().item()
                                                                  for b in m.blocks))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
