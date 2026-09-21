"""SADECE IKI TERIM.  a + b = c,  a,b in 0..50.

Butun ikililer: 51 x 51 = 2601.  Bir kismiyla egit, GORULMEMISLERI sor.
Cevap araligi 0..100, sozlugun tamami kullaniliyor.
  TAVAN 1.000   toplama cikarilabilir
  TABAN         en sik cevap
"""
import statistics
from collections import Counter
import torch
import torch.nn.functional as F
from model_15 import Yol, LR, WD, lr_ver

ENB = 50                    # toplananlar 0..50  ->  0+0=0 ... 50+50=100
SAYI = 101                  # 0..100 -- SOZLUK DEGISMEZ
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


def kos(pay, durum=None, tohum=0, iz=False, norm=None, softmax=None,
        wd=WD, boyut=None, cosine=True):
    EG, TU = bol(pay, tohum)
    WE, WT = yig(EG), yig(TU)
    a = {k: v for k, v in dict(boyut=boyut, durum=durum, norm=norm,
                               pay=softmax).items() if v is not None}
    m = Yol(N, tohum=tohum, **a)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=wd)
    for i in range(ADIM):
        for gr in opt.param_groups:
            gr["lr"] = lr_ver(i, ADIM, cosine=cosine)
        gec(m, WE, opt)
        if iz and i % 100 == 0:
            with torch.no_grad():
                _, de = gec(m, WE); _, dt = gec(m, WT)
            print(f"      adim {i:4d}  egitim {de:.3f}  tutulan {dt:.3f}", flush=True)
    with torch.no_grad():
        _, de = gec(m, WE); _, dt = gec(m, WT)
    return m, de, dt, TU


EG0, TU0 = bol(0.5, 0)
print(f"ikili {len(HEPSI)}   egitim {len(EG0)}   tutulan {len(TU0)}")
print(f"TABAN {Counter(h for _, h in TU0).most_common(1)[0][1]/len(TU0):.3f}   TAVAN 1.000")
print()
def dizi_puan(E):
    P = E[:101] - E[:101].mean(0)
    U, S, _ = torch.linalg.svd(P, full_matrices=False)
    a = U[:, 0] * S[0]
    n = 101
    t = sum(1 for i in range(n) for j in range(i + 1, n) if a[i] > a[j])
    return t / (n * (n - 1) // 2)


print("  lr         HAFIZA  GENELLEME   |hata|   1 fark icinde")
son = None
for cos in (False, True):
    r = [kos(0.5, tohum=t, cosine=cos) for t in (0, 1)]
    e = statistics.mean(x[1] for x in r); u = statistics.mean(x[2] for x in r)
    with torch.no_grad():
        m0, TU0 = r[0][0], r[0][3]
        w, h = yig(TU0)
        c = (-((m0.E[None] - m0.dikkat(w)[0][:, None]) ** 2).sum(-1)).argmax(-1)
        d = (c - h).float()
    ad = "cosine -> lr/10" if cos else "SABIT 0.02    "
    print(f"  {ad}  {e:.3f}    {u:.3f}      {float(d.abs().mean()):5.2f}"
          f"     {float((d.abs() <= 1).float().mean()):.3f}", flush=True)
    if cos:
        son = r[0]

m, _, _, TU = son
print()
print("GORULMEMIS TOPLAMLAR  (egitim payi 0.5)")
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
