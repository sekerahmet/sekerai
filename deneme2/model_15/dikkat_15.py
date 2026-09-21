"""Wq, Wk, Wv ne yapiyor -- ornek uzerinde adim adim."""
import torch
from sehir_15 import AD, dizi, N
from model_15 import Yol

torch.manual_seed(0)
m = Yol(N, boyut=2, durum=6, tohum=0)
E = m.E.detach()

Wq = torch.randn(6, 2) * 0.4      # durum(6) -> soru(2)
Wk = torch.randn(2, 2) * 0.7
Wv = torch.randn(2, 2) * 0.7

onek = ["Istanbul", "Ankara", "Mersin"]
P = E[dizi(onek)]                      # (3, 2) yuvalarin konumlari

print("YOL", " ".join(onek))
for a, p in zip(onek, P):
    print(f"  {a:<10s} konum ({p[0]:+.3f} {p[1]:+.3f})")

print()
print("1) SORU  -- YOLUN TAMAMI soruyu kurar:   q = s @ Wq")
sd = m.ileri(dizi(onek))[-1]
print(f"   s (yolun durumu, 6 sayi) = [{' '.join(f'{x:+.2f}' for x in sd)}]")
q = sd @ Wq
print(f"   s @ Wq  ->  q = ({q[0]:+.3f} {q[1]:+.3f})")

print()
print("2) ANAHTAR -- her yuva kendini tanitir:   k_i = konum_i @ Wk")
print(f"   Wk = {Wk.tolist()}")
K = P @ Wk
for a, k in zip(onek, K):
    print(f"   {a:<10s} -> k = ({k[0]:+.3f} {k[1]:+.3f})")

print()
print("3) PUAN  -- soru ile anahtar ne kadar ortusuyor:   p_i = q . k_i")
puan = K @ q
for i, (a, s) in enumerate(zip(onek, puan)):
    print(f"   {a:<10s} ({q[0]:+.3f})({K[i,0]:+.3f}) + "
          f"({q[1]:+.3f})({K[i,1]:+.3f}) = {s:+.3f}")
print()
print("4) AGIRLIK -- puanlar softmax ile paya cevriliyor")
ag = puan.softmax(0)
for a, w in zip(onek, ag):
    print(f"   {a:<10s} {w:.3f}   {'#' * int(w * 40)}")
print(f"   toplam {ag.sum():.3f}")

print()
print("5) DEGER -- her yuva ne katkı verecek:   v_i = konum_i @ Wv")
print(f"   Wv = {Wv.tolist()}")
V = P @ Wv
for a, v in zip(onek, V):
    print(f"   {a:<10s} -> v = ({v[0]:+.3f} {v[1]:+.3f})")

print()
print("6) CIKTI = agirlikli toplam")
o = ag @ V
ter = "  +  ".join(f"{w:.3f}*({v[0]:+.3f} {v[1]:+.3f})" for w, v in zip(ag, V))
print(f"   {ter}")
print(f"   = ({o[0]:+.3f} {o[1]:+.3f})")
print(f"   en yakin token: {AD[int(torch.cdist(o[None], E).argmin())]}")

print()
print("UZUNLUK SERBEST: yol 10 sehir olsa 3. adimda 10 puan cikar,")
print("matrisler AYNI kalir (Wq, Wk, Wv hep 2x2).")
