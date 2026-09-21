"""Toplamada HAFIZA: 1009 ornegin hepsiyle egit, hepsini geri sor.
TABAN 0.037 (en sik cevap)   TAVAN 1.000"""
import statistics
from collections import Counter
import torch
import torch.nn.functional as F
from model_15 import Yol
from toplama_15 import AD, N, ORNEK

ADIM, TOHUM = 500, 2

d = {}
for o, h in ORNEK:
    d.setdefault(len(o), ([], []))
    d[len(o)][0].append(o); d[len(o)][1].append(h)
O = [(torch.tensor(w), torch.tensor(h)) for w, h in d.values()]
TABAN = Counter(h for _, h in ORNEK).most_common(1)[0][1] / len(ORNEK)


def gec(m, opt=None):
    kay, dog, say = 0.0, 0, 0
    for w, h in O:
        o, _ = m.dikkat(w)
        puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
        kay = kay + F.cross_entropy(puan, h, reduction="sum")
        dog += int((puan.argmax(-1) == h).sum()); say += len(h)
    k = kay / say
    if opt is not None:
        opt.zero_grad(); k.backward(); opt.step()
    return k.item(), dog / say


def kos(durum, boyut, tohum):
    m = Yol(N, boyut=boyut, durum=durum, tohum=tohum, norm=True)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(ADIM):
        gec(m, opt)
    with torch.no_grad():
        _, d = gec(m)
    return m, d, sum(p.numel() for p in m.parameters())


print(f"ornek {len(ORNEK)}   TABAN {TABAN:.3f}   TAVAN 1.000   (norm ACIK)")
print()
print("  durum  boyut  parametre   HAFIZA")
enson = None
for durum in (16, 32):
    for boyut in (2, 8):
        r = [kos(durum, boyut, t) for t in range(TOHUM)]
        h = [x[1] for x in r]
        print(f"  {durum:5d}  {boyut:5d}  {r[0][2]:9d}   "
              f"{statistics.mean(h):.3f} +- {statistics.stdev(h):.3f}")
        enson = r[0][0]

print()
print("EN SON MODELIN CEVAPLARI")
with torch.no_grad():
    for o, h in ORNEK[:8]:
        w = torch.tensor(o)
        c = int((-((enson.E - enson.dikkat(w)[0]) ** 2).sum(-1)).argmax())
        print(f"  {' '.join(AD[i] for i in o):<24s} {AD[c]:>4s}"
              f"   dogru {AD[h]:>4s}  {'' if c == h else '<- YANLIS'}")
