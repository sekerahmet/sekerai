"""VERI + EGITIM + OLCUM.  RAKAM TOKENLI toplama.

Sozluk    0..9  +  "+"  +  "="                 12 token
Sayilar   SABIT GENISLIK, sifir dolgulu:
            toplanan 3 hane  (000..500)
            toplam   4 hane  (0000..1000)
Girdi     4 7 2 + 1 8 2 =        8 token, HEP AYNI
Hedef     cevabin siradaki RAKAMI -- her ciftten 4 ornek:

  4 7 2 + 1 8 2 =            -> 0
  4 7 2 + 1 8 2 = 0          -> 6
  4 7 2 + 1 8 2 = 0 6        -> 5
  4 7 2 + 1 8 2 = 0 6 5      -> 4

DUR yok: genislik sabit oldugu icin durma zaten belli.

RAKAM SIRASI bir KARAR ve olcum bekliyor:
  TERS=False  654 -> once 6 (yuzler).  Elde SAGDAN gelir, model
              ilk rakami soylerken henuz bilmedigi bir eldeyi
              hesaba katmali.
  TERS=True   456 -> once 4 (birler).  Elde SOLA akar, adim adim.
"""
import statistics
import torch
import torch.nn.functional as F
from model_15 import Yol, LR, WD, lr_ver

ENB = 500                   # toplananlar 0..500
HA, HC = 3, 4               # toplanan 3 hane, cevap 4 hane
TERS = False                # rakam sirasi -- KARAR, olculmedi

ARTI, ESIT = 10, 11
N = 12
AD = [str(i) for i in range(10)] + ["+", "="]

ADIM = 400
HEPSI = [(a, b) for a in range(ENB + 1) for b in range(ENB + 1)]


def rak(x, hane):
    d = [int(c) for c in str(x).zfill(hane)]
    return d[::-1] if TERS else d


def soru(a, b):
    return rak(a, HA) + [ARTI] + rak(b, HA) + [ESIT]


def ornekler(ciftler):
    """Her cift -> HC ornek.  Cevap rakam rakam uretiliyor.

    Girdi uzunlugu cevabin kacinci rakaminda oldugumuza gore degisiyor
    (8..8+HC-1), o yuzden UZUNLUGA GORE gruplanip yiginlaniyor."""
    g = {i: ([], []) for i in range(HC)}
    for a, b in ciftler:
        q, c = soru(a, b), rak(a + b, HC)
        for i in range(HC):
            g[i][0].append(q + c[:i]); g[i][1].append(c[i])
    return [(torch.tensor(w), torch.tensor(h)) for w, h in g.values()]


def bol(pay=0.5, tohum=0):
    """CIFT bazinda bolme -- bir ciftin butun rakamlari ayni tarafta."""
    g = torch.Generator().manual_seed(tohum)
    k = torch.randperm(len(HEPSI), generator=g)
    b = int(len(HEPSI) * pay)
    return ([HEPSI[j] for j in k[:b].tolist()],
            [HEPSI[j] for j in k[b:].tolist()])


def yaz(yol="veri_15.pt", pay=0.5, tohum=0):
    """Bolmeyi DOSYAYA yaz.  Colab bunu Drive'dan okur, uretmez."""
    EG, TU = bol(pay, tohum)
    d = dict(ENB=ENB, HA=HA, HC=HC, TERS=TERS, N=N, ARTI=ARTI, ESIT=ESIT,
             AD=AD, pay=pay, tohum=tohum, cift_eg=EG, cift_tu=TU,
             eg=ornekler(EG), tu=ornekler(TU))
    torch.save(d, yol)
    return d


if __name__ == "__main__":
    import sys
    if "--yaz" in sys.argv:
        import hashlib, os
        d = yaz()
        hh = hashlib.sha256()
        for w, _ in d["eg"] + d["tu"]:
            hh.update(w.numpy().tobytes())
        iz = hh.hexdigest()[:16]
        print(f"veri_15.pt yazildi   {os.path.getsize('veri_15.pt')/1e6:.1f} MB")
        print(f"  sozluk {d['N']} token   ters {TERS}")
        ne = sum(len(h) for _, h in d["eg"]); nt = sum(len(h) for _, h in d["tu"])
        print(f"  cift {len(HEPSI)}   egitim {ne}   tutulan {nt}")
        print("  obekler: " + "  ".join(f"{w.shape[1]}token:{len(h)}" for w, h in d["eg"]))
        print(f"  iz {iz}")
        sys.exit()

    EG, TU = bol()
    print(f"sozluk {N}   ters {TERS}")
    for w, h in ornekler(EG[:2]):
        for i in range(len(h)):
            print("  " + " ".join(AD[j] for j in w[i].tolist())
                  + "   ->  " + AD[h[i]])
