"""Cesitli uzunlukta guzergahlar uret, ve gorevin ZORLUGUNU olc.

Kural IKINCI DERECEDEN: sonraki sehir son IKI sehre bagli.
Boylece son sehir TEK BASINA yetmiyor -- geriye bakmak sart.
"""
import torch
from sehir_15 import AD, IX, N, DUR, SEHIR

S = len(SEHIR)                       # 8 sehir (DUR haric)
DI = IX[DUR]

g = torch.Generator().manual_seed(7)
# T[a][b] = a'dan sonra b'ye gelindiyse SIRADAKI
T = torch.randint(0, S, (S, S), generator=g)
# bazi ciftler yolu bitirir
BIT = torch.rand(S, S, generator=g) < 0.22


def uret_rota(bas, ikinci, en_fazla=8):
    """BIT tetiklenmeden en_fazla'ya ulasirsa rota ATILIR -- yoksa
    ayni (a,b) cifti bir yerde devam, baska yerde DUR verirdi."""
    r = [bas, ikinci]
    while len(r) < en_fazla:
        a, b = r[-2], r[-1]
        if BIT[a, b]:
            return r + [DI]
        r.append(int(T[a, b]))
    return None


ROTA = []
gor = set()
for a in range(S):
    for b in range(S):
        if a == b:
            continue
        r = uret_rota(a, b)
        if r is None:
            continue
        r = tuple(r)
        if r not in gor:
            gor.add(r)
            ROTA.append(list(r))

# i=2'den basliyor: ilk iki sehir VERILI, ikinci sehir kuralla belirlenmiyor.
ORNEK = [(r[:i], r[i]) for r in ROTA for i in range(2, len(r))]


def _rapor():
    print(f"ROTA  {len(ROTA)}   ORNEK {len(ORNEK)}")
    uz = {}
    for r in ROTA:
        uz[len(r)] = uz.get(len(r), 0) + 1
    print("  rota uzunlugu:  " + "  ".join(f"{k}:{v}" for k, v in sorted(uz.items())))
    uz = {}
    for o, _ in ORNEK:
        uz[len(o)] = uz.get(len(o), 0) + 1
    print("  girdi uzunlugu: " + "  ".join(f"{k}:{v}" for k, v in sorted(uz.items())))

    print()
    print("GOREV NE KADAR ZOR -- son k sehre bakinca kac ayri devam cikiyor")
    print("  k    ayri baglam   BELIRSIZ olan   en kotu")
    for k in (1, 2, 3, 99):
        d = {}
        for o, h in ORNEK:
            d.setdefault(tuple(o[-k:]), set()).add(h)
        bel = sum(1 for v in d.values() if len(v) > 1)
        enk = max(len(v) for v in d.values())
        ad = "TAMAMI" if k == 99 else str(k)
        print(f"  {ad:<4s} {len(d):8d}   {bel:10d}   {enk:8d}")
    print()
    print("  BELIRSIZ = o kadar geriye bakmak YETMIYOR")


if __name__ == "__main__":
    _rapor()
    print()
    print("ORNEK ROTALAR")
    for r in ROTA[:6]:
        print("  " + " -> ".join(AD[i] for i in r))
