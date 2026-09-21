"""TOPLAMA -- rastgele tablonun yerine YAPILI kural.

Sozluk    1..100  +  DUR                       101 token
Girdi     en fazla 5 sayi, her biri 1..20      toplam <= 100
Hedef     onekin TOPLAMI

  [7, 13]        -> 20
  [7, 13, 4]     -> 24
  [7, 13, 4, 19] -> 43

DUR yok: is "bu sayilari topla", durma karari degil.  (DUR olsaydi
orneklerin %28'i DUR olurdu ve taban sisirdi.)

Rastgele tabloda gorulmemis ikili BILINEMEZDI.  Burada her ikili
BILINEBILIR -- kural cikarilabilir.  TAVAN tam 1.000.
"""
import torch

SAYI = 100          # 1..100
ENB = 20            # toplanan en fazla 20
ADET = 5            # en fazla 5 toplanan
DI = SAYI           # DUR'un indeksi
N = SAYI + 1

# token i  ->  sayi i+1   (DUR haric)
AD = [str(i + 1) for i in range(SAYI)] + ["DUR"]
ix = lambda s: s - 1            # sayi -> token indeksi


def uret(kac=400, tohum=3):
    g = torch.Generator().manual_seed(tohum)
    d, gor = [], set()
    while len(d) < kac:
        k = int(torch.randint(2, ADET + 1, (1,), generator=g))
        a = torch.randint(1, ENB + 1, (k,), generator=g).tolist()
        t = tuple(a)
        if t in gor:
            continue
        gor.add(t)
        d.append(a)
    return d


DIZI = uret()
# onek uzunlugu 2'den basliyor -- tek sayinin "toplami" kendisidir, is degil.
ORNEK = []
for a in DIZI:
    for i in range(2, len(a) + 1):
        ORNEK.append(([ix(x) for x in a[:i]], ix(sum(a[:i]))))



def _rapor():
    from collections import Counter
    print(f"SOZLUK {N} token (1..{SAYI} + DUR)   DIZI {len(DIZI)}   ORNEK {len(ORNEK)}")
    uz = Counter(len(o) for o, _ in ORNEK)
    print("  girdi uzunlugu: " + "  ".join(f"{k}:{v}" for k, v in sorted(uz.items())))
    c = Counter(h for _, h in ORNEK)
    print(f"  ayri cevap {len(c)}   TABAN (en sik cevap) {c.most_common(1)[0][1]/len(ORNEK):.3f}")

    ce = Counter(tuple(o[-2:]) for o, _ in ORNEK)
    print(f"  son-ikili {len(ce)} ayri   tek kez gecen "
          f"{sum(1 for v in ce.values() if v == 1)}")
    print("  !! son ikili YETMEZ -- toplam BUTUN onege bagli")
    print()
    print("ORNEK")
    for o, h in ORNEK[:7]:
        print("  " + " + ".join(AD[i] for i in o) + "   ->  " + AD[h])


if __name__ == "__main__":
    _rapor()
