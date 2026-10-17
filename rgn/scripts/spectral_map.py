"""Spektral harita: her katmanın konum boyunca karıştırıcısı (dikkat ya da evrişim) ne yapıyor, ve ucuz bir sabit
filtre onun yerine geçebilir mi?  (eğitilmiş modelde; yeniden eğitim yok — koşudan önce öngörü)

Katman l'nin karıştırıcı girdisi x = norm1(akış), çıktısı a (akışa eklenen).  Üç ölçü:
  1 frekans   : a'nın konum ekseninde güç spektrumu (kanal ve pencere ortalaması, DC hariç), periyot bantlarına:
                ≥128 token (konu / yavaş bağlam), 32–128, 8–32, <8 (yerel söz dizimi); girdi x'inki ile yan yana.
  2 sabit filtre (LTI) R²: a ≈ Σ_j W_j · (f_j * x),  f_j nedensel sabit filtreler: gecikme 0, 1, 2, 3 ve bant ortalamaları
                4–7, 8–15, 16–31, 32–63, 64–127, 128–255, 256–511, 512+ (FNO / evrişim ailesi: konuma bağlı, içeriğe bağlı
                değil); W_j sırt regresyonuyla, kestirim pencerelerinde; R² ayrı ölçüm pencerelerinde.  "yerel" yalnız
                0–63 filtreleri, "hepsi" bütün filtreler.  Yüksek R²: iş büyük ölçüde sabit tarif; düşük: içeriğe bağlı.
  3 takas     : katmanın çıktısı yerinde sabit filtre tahminiyle değiştirilince kayıp artışı (yalnız o katman).
Okuma: yerel + yüksek R² → kısa evrişim yeter; küresel bantlar R²'yi artırıyor → FNO tarzı küresel filtre; düşük R² →
dikkat gerekli.  Sınır: bugünkü kullanımı ölçer; takas, birlikte uyum yüzünden kaybı abartabilir.

    python scripts/spectral_map.py --run fw_s000_d1024L9_yari_epok
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

DELAYS = (0, 1, 2, 3)
BANDS = ((4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1 << 30))
N_LOCAL = len(DELAYS) + 4                                     # 0–63: gecikmeler + 4–7 … 32–63
PERIODS = ((128, "≥128"), (32, "32–128"), (8, "8–32"), (0, "<8"))


def features(x):
    """x (B, N, d) -> (B, N, F, d): nedensel sabit filtreler (gecikmeler + bant ortalamaları; erken konumda eksik bant 0)."""
    B, N, d = x.shape
    cs = F.pad(x.cumsum(1), (0, 0, 1, 0))                     # cs[:, t] = Σ_{s<t} x_s
    t = torch.arange(N, device=x.device)
    out = []
    for j in DELAYS:
        out.append(F.pad(x, (0, 0, j, 0))[:, :N])
    for lo, hi in BANDS:                                       # ortalama x_{t-hi+1 .. t-lo}
        a = (t - lo + 1).clamp(min=0)                          # üst sınır (dışlayıcı): t - lo + 1
        b = (t - hi + 1).clamp(min=0)
        cnt = (a - b).clamp(min=1).to(x.dtype)
        out.append((cs[:, a] - cs[:, b]) / cnt[None, :, None])
    return torch.stack(out, 2)


def mixers(m):
    return [(b.attn if b.attn is not None else b.conv) if hasattr(b, "conv") else b.attn for b in m.blocks]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--fit", type=int, default=96, help="kestirim penceresi")
    ap.add_argument("--eval", type=int, default=16, help="ölçüm penceresi")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--qmin", type=int, default=256)
    ap.add_argument("--ridge", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0],
                    help="sırt cezası adayları (iz ortalamasına göre); katman başına ölçümde en iyisi")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load_run(a.run, dev)
    m.float()
    L, N, d = len(m.blocks), a.seq, m.cfg.d_model
    toks = valid_tokens(a.dataset, (a.fit + a.eval) * (N + 1))
    wins = torch.from_numpy(np.asarray(toks, dtype=np.int64)).view(-1, N + 1)
    mix = mixers(m)
    kinds = ["%s%s" % (b.kind, "c" if getattr(b, "conv", None) is not None else "") for b in m.blocks]
    cap = {}
    hooks = [mx.register_forward_hook(lambda mod, inp, out, l=l: cap.__setitem__(l, (inp[0], out)))
             for l, mx in enumerate(mix)]
    nF = len(DELAYS) + len(BANDS)
    sel = slice(a.qmin, N)

    def run(w):
        cap.clear()
        m.hidden(w[None, :N].to(dev))

    # 1 kestirim: normal denklemler (katman başına; yerel ve hepsi aynı birikimden alt blokla)
    XtX = [torch.zeros(nF * d, nF * d, device=dev) for _ in range(L)]
    XtY = [torch.zeros(nF * d, d, device=dev) for _ in range(L)]
    for w in wins[:a.fit]:
        run(w)
        for l in range(L):
            x, y = cap[l]
            X = features(x)[0, sel].reshape(-1, nF * d)
            XtX[l] += X.T @ X
            XtY[l] += X.T @ y[0, sel]
    R = len(a.ridge)
    Wall, Wloc = [], []                                                       # katman -> [ceza başına W]
    for l in range(L):
        A, k = XtX[l], N_LOCAL * d
        md = A.diagonal().mean()
        I = torch.eye(A.shape[0], device=dev)
        Wall.append([torch.linalg.solve(A + r * md * I, XtY[l]) for r in a.ridge])
        Wloc.append([torch.linalg.solve(A[:k, :k] + r * md * I[:k, :k], XtY[l][:k]) for r in a.ridge])
        XtX[l] = None
    # 2 ölçüm: R² ve spektrum
    sse_a, sse_l, sst = torch.zeros(L, R), torch.zeros(L, R), torch.zeros(L)
    nb = N // 2 + 1
    per = N / torch.arange(1, nb).float()                                      # periyot (token), DC hariç
    pow_o, pow_i = torch.zeros(L, nb - 1), torch.zeros(L, nb - 1)
    for w in wins[a.fit:]:
        run(w)
        for l in range(L):
            x, y = cap[l]
            X = features(x)[0, sel].reshape(-1, nF * d)
            yt = y[0, sel]
            for r in range(R):
                sse_a[l, r] += ((X @ Wall[l][r] - yt) ** 2).sum().cpu()
                sse_l[l, r] += ((X[:, :N_LOCAL * d] @ Wloc[l][r] - yt) ** 2).sum().cpu()
            sst[l] += ((yt - yt.mean(0)) ** 2).sum().cpu()
            for src, acc in ((y[0], pow_o), (x[0], pow_i)):
                f = torch.fft.rfft(src - src.mean(0), dim=0)                    # konum ekseni
                acc[l] += (f.abs() ** 2).mean(1)[1:].cpu()
    for h in hooks:
        h.remove()
    best = sse_a.argmin(1)                                                    # katman başına en iyi ceza (hepsi)
    r2_all = 1 - sse_a.min(1).values / sst
    r2_loc = 1 - sse_l.min(1).values / sst

    def bands(p):
        out = []
        for i, (lo, _) in enumerate(PERIODS):
            hi = PERIODS[i - 1][0] if i else 1e9
            out.append(float(p[(per >= lo) & (per < hi)].sum() / p.sum()))
        return out

    # 3 takas: katmanın çıktısı sabit filtre tahminiyle değiştirilince kayıp
    X_ev = wins[a.fit:].to(dev)

    def loss():
        return float(sum(F.cross_entropy(m(w[None, :N]).float()[0], w[1:N + 1]) for w in X_ev) / len(X_ev))
    base = loss()
    swap = []
    for l in range(L):
        def repl(mod, inp, out, l=l):
            x = inp[0]
            return (features(x).reshape(x.shape[0], x.shape[1], -1) @ Wall[l][int(best[l])]).to(out.dtype)
        hk = mix[l].register_forward_hook(repl)
        swap.append(loss() - base)
        hk.remove()

    print("%s (adım %s) — kestirim %d, ölçüm %d pencere × %d (konum ≥ %d); taban kayıp %.4f" % (
        a.run, step, a.fit, a.eval, N, a.qmin, base))
    print("\n%-6s %-3s | %s | %s | %8s %8s | %9s" % (
        "katman", "tür", " ".join("%6s" % n for _, n in PERIODS), " ".join("%6s" % n for _, n in PERIODS),
        "R² yerel", "R² hepsi", "takas Δ"))
    print("%-10s | %-27s | %-27s |" % ("", "çıktı enerjisi (periyot, token)", "girdi enerjisi"))
    for l in range(L):
        print("%-6d %-3s | %s | %s | %8.3f %8.3f | %+9.4f" % (
            l, kinds[l], " ".join("%6.2f" % v for v in bands(pow_o[l])), " ".join("%6.2f" % v for v in bands(pow_i[l])),
            r2_loc[l], r2_all[l], swap[l]))
    print("\nseçilen sırt cezası (hepsi): %s" % " ".join("%g" % a.ridge[int(b)] for b in best))
    print("\nokuma: R² yüksek ve yerel ≈ hepsi → kısa evrişim; hepsi > yerel → küresel (FNO tarzı) bileşen; R² düşük → "
          "içeriğe bağlı (dikkat).  Takas Δ küçük → yeniden eğitimsiz bile yerine geçiyor.")


if __name__ == "__main__":
    main()
