"""VERI + EGITIM + OLCUM.  Iki terimli toplama.  a + b = c,  a,b in 0..50.

Butun ikililer: 501 x 501 = 251.001.  Yarisiyla egit, otekini sor.
Cevap araligi 0..1000, sozlugun tamami kullaniliyor.

Dosya: `python veri_15.py --yaz` -> veri_15.pt  (Drive'a konacak)
  TAVAN 1.000   toplama cikarilabilir
  TABAN         en sik cevap
"""
import statistics
from collections import Counter
import torch
import torch.nn.functional as F
from model_15 import Yol, LR, WD, lr_ver

ENB = 500                   # toplananlar 0..500  ->  0+0=0 ... 500+500=1000
SAYI = 1001                 # sozluk 0..1000
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
        wd=WD, boyut=None, cosine=False, lr=LR):
    EG, TU = bol(pay, tohum)
    WE, WT = yig(EG), yig(TU)
    a = {k: v for k, v in dict(boyut=boyut, durum=durum, norm=norm,
                               pay=softmax).items() if v is not None}
    m = Yol(N, tohum=tohum, **a)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=wd)
    for i in range(ADIM):
        for gr in opt.param_groups:
            gr["lr"] = lr_ver(i, ADIM, lr=lr, cosine=cosine)
        gec(m, WE, opt)
        if iz and i % 100 == 0:
            with torch.no_grad():
                _, de = gec(m, WE); _, dt = gec(m, WT)
            print(f"      adim {i:4d}  egitim {de:.3f}  tutulan {dt:.3f}", flush=True)
    with torch.no_grad():
        _, de = gec(m, WE); _, dt = gec(m, WT)
    return m, de, dt, TU


def yaz(yol="veri_15.pt", pay=0.5, tohum=0):
    """Egitim/tutulan bolmesini DOSYAYA yaz.  Colab bunu Drive'dan okur;
    boylece her kosuda yeniden uretilmez ve bolme SABIT kalir."""
    EG, TU = bol(pay, tohum)
    d = dict(ENB=ENB, SAYI=SAYI, N=N, ARTI=ARTI, ESIT=ESIT,
             pay=pay, tohum=tohum,
             eg_w=torch.tensor([o for o, _ in EG]),
             eg_h=torch.tensor([h for _, h in EG]),
             tu_w=torch.tensor([o for o, _ in TU]),
             tu_h=torch.tensor([h for _, h in TU]))
    torch.save(d, yol)
    return d


if __name__ == "__main__":
    import sys
    if "--yaz" in sys.argv:
        import hashlib, os
        d = yaz()
        iz = hashlib.sha256(d["eg_w"].numpy().tobytes()
                            + d["tu_w"].numpy().tobytes()).hexdigest()[:16]
        print(f"veri_15.pt yazildi   {os.path.getsize('veri_15.pt')/1e6:.1f} MB")
        print(f"  sozluk {d['N']} token   egitim {len(d['eg_h'])}   tutulan {len(d['tu_h'])}")
        print(f"  iz {iz}")
        sys.exit()

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


    TOH = 5
    print(f"{TOH} tohum.  'BULDU' = tutulan > 0,50 (dagilim iki tepeli).")
    print()
    print("  wd      lr      BULDU   ortalama   tohumlar")
    son = None
    for wd in (0.01, 0.03):
        for lr in (0.02, 0.04):
            r = [kos(0.5, tohum=t, lr=lr, wd=wd) for t in range(TOH)]
            u = [x[2] for x in r]
            buldu = sum(1 for x in u if x > 0.5)
            print(f"  {wd:<7} {lr:<7} {buldu}/{TOH}    {statistics.mean(u):.3f}"
                  f"      " + " ".join(f"{x:.2f}" for x in u), flush=True)
            if wd == 0.03 and lr == 0.04:
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
