"""SADECE IKI TERIM.  a + b = c,  a,b in 0..20.

Butun ikililer: 21 x 21 = 441.  Bir kismiyla egit, GORULMEMISLERI sor.
  TAVAN 1.000   toplama cikarilabilir
  TABAN         en sik cevap
"""
import statistics
from collections import Counter
import torch
import torch.nn.functional as F
from model_15 import Yol

ENB = 20
SAYI = 2 * ENB + 1          # 0..40  cevaplar bu araliga siginiyor
ARTI, ESIT = SAYI, SAYI + 1
N = SAYI + 2
AD = [str(i) for i in range(SAYI)] + ["+", "="]

HEPSI = [(a, b) for a in range(ENB + 1) for b in range(ENB + 1)]
soru = lambda a, b: [a, ARTI, b, ESIT]

ADIM = 400


def bol(pay, tohum):
    g = torch.Generator().manual_seed(tohum)
    k = torch.randperm(len(HEPSI), generator=g)
    b = int(len(HEPSI) * pay)
    yap = lambda I: [(soru(*HEPSI[j]), sum(HEPSI[j])) for j in I.tolist()]
    return yap(k[:b]), yap(k[b:])


def yig(S):
    return (torch.tensor([o for o, _ in S]), torch.tensor([h for _, h in S]))


def gec(m, W, opt=None):
    w, h = W
    o, _ = m.dikkat(w)
    puan = -((m.E[None] - o[:, None]) ** 2).sum(-1)
    k = F.cross_entropy(puan, h)
    if opt is not None:
        opt.zero_grad(); k.backward(); opt.step()
    return k.item(), float((puan.argmax(-1) == h).float().mean())


def kos(pay, durum, tohum, iz=False):
    EG, TU = bol(pay, tohum)
    WE, WT = yig(EG), yig(TU)
    m = Yol(N, boyut=8, durum=durum, tohum=tohum, norm=True)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for i in range(ADIM):
        gec(m, WE, opt)
        if iz and i % 100 == 0:
            with torch.no_grad():
                _, de = gec(m, WE); _, dt = gec(m, WT)
            print(f"      adim {i:4d}  egitim {de:.3f}  tutulan {dt:.3f}", flush=True)
    with torch.no_grad():
        _, de = gec(m, WE); _, dt = gec(m, WT)
    return m, de, dt, TU


EG0, TU0 = bol(0.8, 0)
print(f"ikili {len(HEPSI)}   egitim {len(EG0)}   tutulan {len(TU0)}")
print(f"TABAN {Counter(h for _, h in TU0).most_common(1)[0][1]/len(TU0):.3f}   TAVAN 1.000")
print()
print("  egitim payi  durum   HAFIZA   GENELLEME")
son = None
for pay in (0.8, 0.5):
    for durum in (16,):
        r = [kos(pay, durum, t) for t in (0, 1)]
        e = [x[1] for x in r]; u = [x[2] for x in r]
        print(f"  {pay:^11}  {durum:5d}   {statistics.mean(e):.3f}"
              f"    {statistics.mean(u):.3f} +- {statistics.stdev(u):.3f}", flush=True)
        if pay == 0.8:
            son = r[0]

m, _, _, TU = son
print()
print("GORULMEMIS TOPLAMLAR  (egitim payi 0.8)")
with torch.no_grad():
    w, h = yig(TU)
    o, _ = m.dikkat(w)
    c = (-((m.E[None] - o[:, None]) ** 2).sum(-1)).argmax(-1)
    for i in range(14):
        print(f"  {AD[w[i,0]]:>2s} + {AD[w[i,2]]:>2s} =  {AD[c[i]]:>3s}"
              f"   dogru {AD[h[i]]:>3s}  {'DOGRU' if c[i] == h[i] else 'YANLIS'}")
    d = (c - h).float()
    print()
    print(f"  dogru {float((c==h).float().mean()):.3f}"
          f"   ort hata {d.mean():+.2f}   |hata| {d.abs().mean():.2f}")
    print(f"  1 fark icinde {float((d.abs()<=1).float().mean()):.3f}"
          f"   3 fark icinde {float((d.abs()<=3).float().mean()):.3f}")
