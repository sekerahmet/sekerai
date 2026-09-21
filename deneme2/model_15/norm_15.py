"""Normalizasyon acik / kapali yan yana.  Ayni veri, ayni tohumlar."""
import torch
import torch.nn.functional as F
from sehir_15 import AD, IX, N, DUR
from model_15 import Yol
from veri_15 import ORNEK

g = torch.Generator().manual_seed(1)
k = torch.randperm(len(ORNEK), generator=g)
b = int(len(ORNEK) * 0.8)
EGIT = [ORNEK[i] for i in k[:b]]
TUT  = [ORNEK[i] for i in k[b:]]


def obek(S):
    d = {}
    for o, h in S:
        d.setdefault(len(o), ([], []))
        d[len(o)][0].append(o); d[len(o)][1].append(h)
    return [(torch.tensor(w), torch.tensor(h)) for w, h in d.values()]


OE, OT = obek(EGIT), obek(TUT)


def gec(m, O):
    kay, dog, say, sn = 0.0, 0, 0, 0.0
    for w, h in O:
        o, _ = m.dikkat(w)
        puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
        kay = kay + F.cross_entropy(puan, h, reduction="sum")
        dog += int((puan.argmax(-1) == h).sum()); say += len(h)
        sn = max(sn, float(m.gez(w)[:, -1].norm(dim=-1).max().detach()))
    return kay / say, dog / say, sn


def kos(norm, wd, adim=600, tohum=0):
    m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=tohum, norm=norm)
    opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=wd)
    sic = 0
    onceki = None
    for _ in range(adim):
        kp, _d, _s = gec(m, OE)
        opt.zero_grad(); kp.backward(); opt.step()
        if onceki is not None and kp.item() > onceki * 1.5:
            sic += 1                                  # kayip aniden %50 artti
        onceki = kp.item()
    with torch.no_grad():
        _, de, sn = gec(m, OE)
        _, dt, _ = gec(m, OT)
    return de, dt, sn, sic


print("            egitim  tutulan   en buyuk |s|   kayip sicramasi")
for norm in (False, True):
    for wd in (0.0, 0.03):
        e = t = s = c = 0.0
        for th in range(3):
            a, b2, sn, sic = kos(norm, wd, tohum=th)
            e += a; t += b2; s += sn; c += sic
        ad = f"norm {'ACIK ' if norm else 'KAPALI'} wd {wd}"
        print(f"  {ad:<22s} {e/3:.3f}   {t/3:.3f}     {s/3:8.2f}        {c/3:.1f}")
print()
print("  TAVAN 0.714")
