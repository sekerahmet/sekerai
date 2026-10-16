"""Özel nöron tabloları ne kadar boyut kullanıyor?  (ortak tabanlı / düşük rank'lı özel nöron fikrinin ön sınaması)

1. Etkin boyut: her katmanda g, u, w tablolarının (satır × d) sıklık ağırlıklı Gram matrisinin özdeğerleri; enerjinin
   %50 / 80 / 90 / 95 / 99'u için gereken boyut (ağırlıksız %90 da).
2. İşlevsel sınama: g, u, w satırları katman başına ortak r boyutlu tabana izdüşürülür (sıklık ağırlıklı PCA, yeniden
   eğitim YOK) → doğrulama kaybındaki artış.  Yeniden eğitimsiz kesme üst sınırdır (eğitilen düşük rank daha iyi olur).

    python scripts/private_rank.py --run fw_s000_d512_ozel_3epok
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.paths import ROOT  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

FRACS = (0.5, 0.8, 0.9, 0.95, 0.99)
PARTS = "guw"


def _basis(X, f):
    """Sıklık ağırlıklı Gram'ın özvektörleri (azalan) ve özdeğerleri: X (R, D), f (R,)."""
    C = (X * f[:, None]).T @ X
    e, V = torch.linalg.eigh(C.double())
    return e.flip(0).clamp_min(0), V.flip(1).to(X.dtype)


def _dims(e):
    c = (e.cumsum(0) / e.sum()).tolist()
    return [next(i + 1 for i, v in enumerate(c) if v >= fr) for fr in FRACS]


@torch.no_grad()
def _loss(m, win, batch=4):
    tot = 0.0
    for i in range(0, len(win), batch):
        w = win[i:i + batch]
        tot += m(w[:, :-1], w[:, 1:]).item() * len(w)
    return tot / len(win)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--counts", default="assets/fw_s000_vocab_used.json")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    m, step = load_run(a.run, "cpu")
    m.float()
    assert m.cfg.private_neurons, "özel nöronsuz model"
    D, L = m.cfg.d_model, len(m.blocks)
    c = json.loads((ROOT / a.counts).read_text(encoding="utf-8"))
    R = m.blocks[0].private.shape[0]
    f = torch.zeros(R)
    f[m.priv_map[torch.tensor(c["tokens"])]] = torch.tensor(c.get("counts") or [1] * len(c["tokens"]), dtype=torch.float)
    f = f / f.sum()
    tabs = [b.private.view(R, 3, D) for b in m.blocks]                            # görünüm: yerinde değişir
    orig = [t.clone() for t in tabs]
    bases = [[_basis(t[:, p], f) for p in range(3)] for t in tabs]
    ones = torch.ones(R) / R

    print("%s (adım %s) — %d satır × %d katman × 3 × %d; enerji için gereken boyut (sıklık ağırlıklı; son sütun "
          "ağırlıksız %%90)" % (a.run, step, R, L, D))
    print("%-7s %-4s %s %10s" % ("katman", "", " ".join("%6s" % ("%%%d" % (100 * fr)) for fr in FRACS), "ağırlıksız"))
    for l in range(L):
        for p in range(3):
            e_u = _basis(tabs[l][:, p], ones)[0]
            print("%-7s %-4s %s %10d" % (l if p == 0 else "", PARTS[p], " ".join("%6d" % v for v in _dims(bases[l][p][0])),
                                         _dims(e_u)[2]))

    a_tok = valid_tokens(a.dataset, a.windows * (a.seq + 1))
    win = torch.from_numpy(a_tok.astype(np.int64)).view(a.windows, a.seq + 1)
    base = _loss(m, win)
    print("\nişlevsel: satırlar ortak r boyutlu tabana izdüşürülür (yeniden eğitimsiz); doğrulama %d token, kayıp %.4f"
          % (win[:, 1:].numel(), base))

    def project(r, parts):
        for l in range(L):
            for p in parts:
                V = bases[l][p][1][:, :r]
                tabs[l][:, p] = (orig[l][:, p] @ V) @ V.T

    def restore():
        for t, o in zip(tabs, orig):
            t.copy_(o)

    zero = None
    for l in range(L):
        tabs[l].zero_()
    zero = _loss(m, win)
    restore()
    print("%-26s %9s %9s %12s" % ("düzen", "kayıp", "Δ", "kazancın %'si"))
    print("%-26s %9.4f %+9.4f %12s" % ("özel nöron kapalı (sıfır)", zero, zero - base, "0"))
    for r in a.ranks:
        project(r, range(3))
        x = _loss(m, win)
        restore()
        print("%-26s %9.4f %+9.4f %11.0f%%  (tablo %d / %d sayı)" % (
            "r = %d (g, u, w)" % r, x, x - base, 100 * (zero - x) / (zero - base), 3 * r, 3 * D))
    r = 64 if 64 in a.ranks else a.ranks[len(a.ranks) // 2]
    for p in range(3):
        project(r, [p])
        x = _loss(m, win)
        restore()
        print("%-26s %9.4f %+9.4f %11.0f%%" % ("r = %d yalnız %s" % (r, PARTS[p]), x, x - base,
                                                  100 * (zero - x) / (zero - base)))


if __name__ == "__main__":
    main()
