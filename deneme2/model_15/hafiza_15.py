"""HAFIZA: verdigimizi hatirliyor mu.

Butun ornekler egitimde, hicbiri saklanmiyor.  Ayni ornekler geri soruluyor.
Ceza YOK -- ceza hafizayi bozar (deney_15: 0,922 -> 0,681).
  TAVAN 1.000   hepsi ogretildi
  TABAN 0.192   hep en sik cevabi soyleyen model
"""
import statistics
from collections import Counter
import torch
import torch.nn.functional as F
from sehir_15 import IX, N, DUR
from model_15 import Yol
from veri_15 import ORNEK

ADIM, TOHUM = 500, 2

d = {}
for o, h in ORNEK:
    d.setdefault(len(o), ([], []))
    d[len(o)][0].append(o); d[len(o)][1].append(h)
O = [(torch.tensor(w), torch.tensor(h)) for w, h in d.values()]
TABAN = Counter(h for _, h in ORNEK).most_common(1)[0][1] / len(ORNEK)


def gec(m, egit_op=None):
    kay, dog, say = 0.0, 0, 0
    for w, h in O:
        o, _ = m.dikkat(w)
        puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
        kay = kay + F.cross_entropy(puan, h, reduction="sum")
        dog += int((puan.argmax(-1) == h).sum()); say += len(h)
    k = kay / say
    if egit_op is not None:
        egit_op.zero_grad(); k.backward(); egit_op.step()
    return k.item(), dog / say


def kos(durum, norm, tohum):
    m = Yol(N, boyut=2, durum=durum, dur=IX[DUR], tohum=tohum, norm=norm)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(ADIM):
        gec(m, opt)
    with torch.no_grad():
        _, d = gec(m)
    return d, sum(p.numel() for p in m.parameters())


print(f"ornek {len(ORNEK)}   TABAN {TABAN:.3f}   TAVAN 1.000")
print()
print("  durum   norm    parametre   HAFIZA")
for durum in (6, 16, 32):
    for norm in (False, True):
        r = [kos(durum, norm, t) for t in range(TOHUM)]
        h = [x[0] for x in r]
        print(f"  {durum:5d}   {'ACIK ' if norm else 'KAPALI'}  {r[0][1]:8d}"
              f"   {statistics.mean(h):.3f} +- {statistics.stdev(h):.3f}")
