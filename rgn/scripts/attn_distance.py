"""Dikkat ağırlığının uzaklık bantlarına dağılımı: her katman (Y / G) ve head için, sorgunun ne kadar uzağa baktığı.

Uzaklık = sorgu konumu − anahtar konumu.  Bantlar: 0 (kendisi), 1–15, 16–63, 64–127, 128+ (yalnız G).  Sorgular
--qmin'den sonraki konumlar (her bant erişilebilir olsun diye).  Ağırlık: Attention.weights (açık softmax, kapı hariç).

    python scripts/attn_distance.py --run fw_s000_d1024L9_yari_epok
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

BANDS = [(0, 1), (1, 16), (16, 64), (64, 128), (128, 1 << 30)]


def _name(lo, hi):
    return "%d" % lo if hi == lo + 1 else ("%d+" % lo if hi >= 1 << 30 else "%d–%d" % (lo, hi - 1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--qmin", type=int, default=256, help="bu konumdan önceki sorgular sayılmaz")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load_run(a.run, dev)
    m.float()
    N = a.seq
    toks = valid_tokens(a.dataset, a.windows * N)
    idx = torch.from_numpy(np.asarray(toks, dtype=np.int64)).view(a.windows, N)
    dist = torch.arange(N)[:, None] - torch.arange(N)[None, :]                 # (N, N) sorgu − anahtar
    band_mask = [((dist >= lo) & (dist < hi)).float().to(dev) for lo, hi in BANDS]
    qsel = torch.arange(N, device=dev) >= a.qmin
    L, H = len(m.blocks), m.blocks[0].attn.H
    mass = torch.zeros(L, H, len(BANDS), device=dev)
    mean_d = torch.zeros(L, H, device=dev)
    dist_d = dist.clamp_min(0).float().to(dev)
    for w in range(a.windows):
        x_idx = idx[w:w + 1].to(dev)
        x = m.embed(x_idx)
        rope = m.rope(N)
        rows = m.priv_map[x_idx] if m.priv_map is not None else None
        ple = m.ple_context(x)
        for l, b in enumerate(m.blocks):
            p = b.attn.weights(b.norm1(x), rope)[0]                              # (H, N, N)
            pq = p[:, qsel]                                                       # (H, Nq, N)
            for k, bm in enumerate(band_mask):
                mass[l, :, k] += (pq * bm[qsel]).sum(-1).mean(-1)
            mean_d[l] += (pq * dist_d[qsel]).sum(-1).mean(-1)
            x = b(x, rope, None, rows, None, None, None if ple is None else ple[:, :, l])
    mass /= a.windows
    mean_d /= a.windows
    kinds = [b.kind for b in m.blocks]
    print("%s (adım %s) — %d pencere × %d token, sorgular konum ≥ %d; dikkat payı (%%), head ortalaması" % (
        a.run, step, a.windows, N, a.qmin))
    print("%-7s %-4s %s %10s   %s" % ("katman", "tür", " ".join("%8s" % _name(*bd) for bd in BANDS), "ort. uzaklık",
                                       "head'ler: 16+ pay (en az / en çok)"))
    for l in range(L):
        mb = 100 * mass[l].mean(0)
        far = 100 * mass[l, :, 2:].sum(-1)
        print("%-7d %-4s %s %10.1f   %5.1f / %5.1f" % (l, kinds[l], " ".join("%8.1f" % v for v in mb.tolist()),
                                                    mean_d[l].mean().item(), far.min().item(), far.max().item()))
    print("\nhead başına 16+ pay (%%), katman satırı:")
    for l in range(L):
        far = (100 * mass[l, :, 2:].sum(-1)).tolist()
        print("%-3d %s  %s" % (l, kinds[l], " ".join("%4.0f" % v for v in far)))


if __name__ == "__main__":
    main()
