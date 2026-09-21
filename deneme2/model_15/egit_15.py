"""Iki ornek uzerinde egitim.  Ayni uc sehir, ters sira, farkli cevap."""
import math
import torch
from sehir_15 import AD, IX, N, DUR, dizi
from model_15 import Yol

ORNEK = [(["Istanbul", "Ankara", "Mersin"], "Sivas"),
         (["Mersin", "Ankara", "Istanbul"], "Izmir")]

m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=0)
opt = torch.optim.Adam(m.parameters(), lr=0.02)

W = [(dizi(y), IX[h]) for y, h in ORNEK]


def kayip():
    t = 0.0
    for w, h in W:
        o, _ = m.dikkat(w)
        puan = -((m.E - o) ** 2).sum(-1)
        t = t + torch.nn.functional.cross_entropy(puan[None], torch.tensor([h]))
    return t / len(W)


print(f"sans  {math.log(N):.4f}")
print("adim     kayip")
for i in range(601):
    opt.zero_grad()
    k = kayip()
    k.backward()
    opt.step()
    if i % 100 == 0:
        print(f"{i:4d}   {k.item():.4f}")

print()
with torch.no_grad():
    for (yol, hedef), (w, h) in zip(ORNEK, W):
        o, ag = m.dikkat(w)
        p = (-((m.E - o) ** 2).sum(-1)).softmax(0)
        sec = int(p.argmax())
        print(f"{' '.join(yol):<26s} -> {AD[sec]:<10s} hedef {hedef:<8s}"
              f" olasilik {p[h]:.4f}  {'DOGRU' if sec == h else 'YANLIS'}")
        print(f"   agirlik  " + "  ".join(f"{a}:{float(x):.3f}"
                                          for a, x in zip(yol, ag)))
