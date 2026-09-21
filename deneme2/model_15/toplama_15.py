"""TOPLAMA -- rastgele tablonun yerine YAPILI kural.

Sozluk    0..100  +  "+"  +  "="                103 token
Girdi     islemin KENDISI, token token
Hedef     esitligin sagi

  7 + 13 =           -> 20
  7 + 13 + 4 =       -> 24
  7 + 13 + 4 + 19 =  -> 43

"=" cevabin nerede basladigini isaretliyor -- K2'deki CEVAP isareti.

DUR yok: is "bu sayilari topla", durma karari degil.  (DUR olsaydi
orneklerin %28'i DUR olurdu ve taban sisirdi.)

Rastgele tabloda gorulmemis ikili BILINEMEZDI.  Burada her ikili
BILINEBILIR -- kural cikarilabilir.  TAVAN tam 1.000.
"""
import torch

SAYI = 101          # 0..100  -- 0 da SART: "sifir eklemek degistirmez"
ENB = 20            # toplanan 0..20
ADET = 5            # en fazla 5 toplanan
ARTI = SAYI         # "+"  token indeksi
ESIT = SAYI + 1     # "="  token indeksi
N = SAYI + 2

AD = [str(i) for i in range(SAYI)] + ["+", "="]
ix = lambda s: s               # sayi -> token indeksi  (0 -> 0)


def islem(a):
    """[7, 13, 4]  ->  token dizisi  7 + 13 + 4 ="""
    d = [ix(a[0])]
    for x in a[1:]:
        d += [ARTI, ix(x)]
    return d + [ESIT]


def uret(kac=400, tohum=3):
    g = torch.Generator().manual_seed(tohum)
    d, gor = [], set()
    while len(d) < kac:
        k = int(torch.randint(2, ADET + 1, (1,), generator=g))
        a = torch.randint(0, ENB + 1, (k,), generator=g).tolist()
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
        ORNEK.append((islem(a[:i]), ix(sum(a[:i]))))



def _rapor():
    from collections import Counter
    print(f"SOZLUK {N} token (0..{SAYI-1} + '+' + '=')"
          f"   DIZI {len(DIZI)}   ORNEK {len(ORNEK)}")
    uz = Counter(len(o) for o, _ in ORNEK)
    print("  girdi uzunlugu: " + "  ".join(f"{k}:{v}" for k, v in sorted(uz.items())))
    c = Counter(h for _, h in ORNEK)
    print(f"  ayri cevap {len(c)}   TABAN (en sik cevap) {c.most_common(1)[0][1]/len(ORNEK):.3f}")

    ce = Counter(tuple(o[-2:]) for o, _ in ORNEK)
    print(f"  son-ikili {len(ce)} ayri   tek kez gecen "
          f"{sum(1 for v in ce.values() if v == 1)}")
    print("  !! son ikili YETMEZ -- toplam BUTUN onege bagli")
    print("     (ve son token her zaman '=' -- tek basina hicbir sey soylemiyor)")
    print()
    print("ORNEK")
    for o, h in ORNEK[:7]:
        print("  " + " ".join(AD[i] for i in o) + "  " + AD[h])


if __name__ == "__main__":
    _rapor()
