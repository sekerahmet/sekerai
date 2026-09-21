"""oku_yol adaylarini ORNEK uzerinde hesapla.  Egitim yok -- dogrudan cozum."""
import torch
from sehir_15 import AD, IX, N, dizi
from model_15 import Yol

torch.set_printoptions(precision=3, sci_mode=False)
m = Yol(N, boyut=2, durum=6, tohum=0)
E = m.E.detach()

YOL = [(["Istanbul", "Ankara", "Mersin"], "Sivas"),
       (["Izmir",    "Bursa",  "Mersin"], "Adana"),
       (["Konya",    "Ankara", "Mersin"], "Sivas")]

V = torch.stack([E[dizi(y)].reshape(-1) for y, _ in YOL])      # (3, 6) girdi
T = torch.stack([E[IX[c]] for _, c in YOL])                    # (3, 2) hedef

print("GIRDI  v  (6 sayi)                                  HEDEF")
for (y, c), v, t in zip(YOL, V, T):
    print(f"  {' '.join(y):<24s} {v.tolist()}")
    print(f"  {'':24s}   -> {c} {t.tolist()}")

print()
print("=" * 62)
print("ADAY 1  SABIT PENCERE   cikti = v @ W      W: 6x2 = 12 sayi")
print("=" * 62)
W = torch.linalg.lstsq(V, T).solution
print("W =")
print(W)
print()
print("  Istanbul Ankara Mersin icin ACIK hesap:")
v = V[0]
for j in range(2):
    ter = "  ".join(f"{v[i]:+.3f}*{W[i,j]:+.3f}" for i in range(6))
    print(f"    cikti[{j}] = {ter}")
    print(f"             = {(v * W[:, j]).sum():+.3f}")
print()
C = V @ W
for (y, c), o, t in zip(YOL, C, T):
    ok = AD[int(torch.cdist(o[None], E).argmin())]
    print(f"  {' '.join(y):<24s} cikti ({o[0]:+.3f} {o[1]:+.3f})"
          f"  hedef ({t[0]:+.3f} {t[1]:+.3f})  okunan {ok:<8s} {'DOGRU' if ok == c else 'YANLIS'}")

print()
print("=" * 62)
print("ADAY 2  YUVA AGIRLIGI   cikti = (a1*E1 + a2*E2 + a3*E3) @ W2")
print("        a: 3 sayi + W2: 2x2 = 4 sayi   -> 7 sayi")
print("=" * 62)
P = torch.stack([E[dizi(y)] for y, _ in YOL])                  # (3 yol, 3 yuva, 2)
en, ea = 9e9, None
for a1 in torch.linspace(-2, 2, 81):
    for a2 in torch.linspace(-2, 2, 81):
        for a3 in torch.linspace(-2, 2, 81):
            a = torch.tensor([a1, a2, a3])
            U = (P * a[None, :, None]).sum(1)                  # (3, 2)
            W2 = torch.linalg.lstsq(U, T).solution
            h = ((U @ W2 - T) ** 2).sum().item()
            if h < en:
                en, ea = h, (a.clone(), W2.clone())
a, W2 = ea
print(f"  en iyi a = {a.tolist()}   kalan hata {en:.4f}")
U = (P * a[None, :, None]).sum(1)
for (y, c), o, t in zip(YOL, U @ W2, T):
    ok = AD[int(torch.cdist(o[None], E).argmin())]
    print(f"  {' '.join(y):<24s} cikti ({o[0]:+.3f} {o[1]:+.3f})"
          f"  hedef ({t[0]:+.3f} {t[1]:+.3f})  okunan {ok:<8s} {'DOGRU' if ok == c else 'YANLIS'}")
