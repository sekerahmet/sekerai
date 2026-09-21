"""Iki ariza: (1) egitim ortasinda bozulma, (2) ezber.  Toplu hesap."""
import math
import torch
import torch.nn.functional as F
from sehir_15 import IX, N, DUR
from model_15 import Yol
from veri_15 import ORNEK

g = torch.Generator().manual_seed(1)
k = torch.randperm(len(ORNEK), generator=g)
b = int(len(ORNEK) * 0.8)
EGIT = [ORNEK[i] for i in k[:b]]
TUT  = [ORNEK[i] for i in k[b:]]


def obek(S):
    """Ayni uzunluktakileri tek yigina topla."""
    d = {}
    for o, h in S:
        d.setdefault(len(o), ([], []))
        d[len(o)][0].append(o); d[len(o)][1].append(h)
    return [(torch.tensor(w), torch.tensor(h)) for w, h in d.values()]


OE, OT = obek(EGIT), obek(TUT)


def gec(m, O):
    kay, dog, say, sn = 0.0, 0, 0, 0.0
    for w, h in O:
        S = m.gez(w)
        o, _ag = m.dikkat(w)
        puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)      # (B, N)
        kay = kay + F.cross_entropy(puan, h, reduction="sum")
        dog += int((puan.argmax(-1) == h).sum()); say += len(h)
        sn = max(sn, float(S[:, -1].norm(dim=-1).max()))
    return kay / say, dog / say, sn


def kos(lr, wd, adim, iz=False, tohum=0):
    m = Yol(N, boyut=2, durum=6, dur=IX[DUR], tohum=tohum)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    kayit = []
    for i in range(adim + 1):
        kp, dg, sn = gec(m, OE)
        opt.zero_grad(); kp.backward()
        gn = math.sqrt(sum(float((p.grad ** 2).sum()) for p in m.parameters()))
        opt.step()
        if iz and i % 25 == 0:
            with torch.no_grad():
                kt, dt, _ = gec(m, OT)
            kayit.append((i, kp.item(), dg, float(kt), dt, sn, gn))
    with torch.no_grad():
        _, de, _ = gec(m, OE); _, dt, _ = gec(m, OT)
    return de, dt, kayit


print("1) BOZULMA NEREDE   lr 0.02  wd 0")
_, _, iz = kos(0.02, 0.0, 500, iz=True)
print("  adim   EG kayip  EG dog   TUT kayip  TUT dog   en buyuk |s|   gradyan")
for i, kp, dg, kt, dt, sn, gn in iz:
    im = "  <<<" if gn > 30 or sn > 30 else ""
    print(f"  {i:4d}   {kp:7.4f}   {dg:.3f}    {kt:8.3f}    {dt:.3f}   {sn:9.2f}   {gn:7.2f}{im}")

print()
print("2) AYAR TARAMASI   600 adim, 2 tohum ortalamasi")
print("     lr      wd     egitim   tutulan")
for lr in (0.02, 0.005):
    for wd in (0.0, 0.03):
        e = t = 0.0
        for s in range(2):
            a, b2, _ = kos(lr, wd, 600, tohum=s)
            e += a; t += b2
        print(f"   {lr:<7} {wd:<6} {e/2:7.3f}   {t/2:7.3f}")
print()
print("  TAVAN 0.714  -- tutulanin 6/21'inde son ikili egitimde HIC gecmiyor")
