"""Duzeltilmis olcum: (1) buyuk veri, (2) ROTA bazinda bolme, (3) cok tohum."""
import statistics
import torch
import torch.nn.functional as F
from sehir_15 import IX, N, DUR
from model_15 import Yol
from veri_15 import ROTA

TOHUM = 5
ADIM = 400


def bol(t):
    """ROTA bazinda 80/20 -- bir rotanin BUTUN onekleri ayni tarafta."""
    g = torch.Generator().manual_seed(t)
    k = torch.randperm(len(ROTA), generator=g)
    b = int(len(ROTA) * 0.8)
    yap = lambda ix: [(r[:i], r[i]) for j in ix for r in [ROTA[j]]
                      for i in range(2, len(r))]
    return yap(k[:b].tolist()), yap(k[b:].tolist())


def obek(S):
    d = {}
    for o, h in S:
        d.setdefault(len(o), ([], []))
        d[len(o)][0].append(o); d[len(o)][1].append(h)
    return [(torch.tensor(w), torch.tensor(h)) for w, h in d.values()]


def tavan(EG, TU):
    """Kurali EZBERLEYEN bir modelin tutulanda alabilecegi en iyi sonuc."""
    c = {tuple(o[-2:]): h for o, h in EG}
    return sum(1 for o, h in TU if c.get(tuple(o[-2:])) == h) / len(TU)


def gec(m, O, egit_op=None):
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


def kos(norm, wd, tohum):
    EG, TU = bol(tohum)
    OE, OT = obek(EG), obek(TU)
    m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=tohum, norm=norm)
    opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=wd)
    for _ in range(ADIM):
        gec(m, OE, opt)
    with torch.no_grad():
        _, de = gec(m, OE)
        _, dt = gec(m, OT)
    return de, dt, tavan(EG, TU)


eg0, tu0 = bol(0)
print(f"ROTA {len(ROTA)}   egitim ornegi {len(eg0)}   tutulan {len(tu0)}")
print(f"TAVAN (5 tohum ort) {statistics.mean(tavan(*bol(t)) for t in range(TOHUM)):.3f}")
print()
print("                      egitim            tutulan")
for norm in (False, True):
    for wd in (0.0, 0.03):
        e, t = [], []
        for th in range(TOHUM):
            a, b, _ = kos(norm, wd, th)
            e.append(a); t.append(b)
        ad = f"norm {'ACIK ' if norm else 'KAPALI'} wd {wd}"
        print(f"  {ad:<20s} {statistics.mean(e):.3f} +- {statistics.stdev(e):.3f}"
              f"   {statistics.mean(t):.3f} +- {statistics.stdev(t):.3f}")
