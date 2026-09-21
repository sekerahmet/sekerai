"""OKUMA kurali adaylari: uzunluk ve aci ne kadar oneme sahip.  ADIM 1b.

Uc aday, ayni sehir tablosu.  Sorulan tek sey:
  bir sehir HIC okunabilir mi -- yani onu kazandiran bir durum var mi.
"""
import torch
from sehir_15 import AD, KONUM, N

ACI = torch.linspace(0, 6.2831853, 3601)[:-1]
YON = torch.stack([ACI.cos(), ACI.sin()], 1)          # (3600, 2) birim yon

g = torch.linspace(-3.0, 3.0, 601)
IZGARA = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def _pay(kazanan, toplam):
    say = torch.bincount(kazanan, minlength=N).float() / toplam
    return say


def rapor():
    k = KONUM
    b = k / k.norm(dim=-1, keepdim=True)

    # A) IC CARPIM   argmax <s, x_c>  -- yalniz s'nin YONUNE bagli
    a = _pay((YON @ k.t()).argmax(1), len(YON))

    # B) KOSINUS     argmax <s, x_c/|x_c|>  -- uzunluk ATILIYOR
    c = _pay((YON @ b.t()).argmax(1), len(YON))

    # C) EN YAKIN    argmin |s - x_c|  -- uzunluk ve aci BIRLIKTE
    y = _pay(torch.cdist(IZGARA, k).argmin(1), len(IZGARA))

    print("  Her sehrin KAZANDIGI bolgenin payi  (0,000 = HIC okunamaz)\n")
    print("  sehir        A ic carpim   B kosinus   C en yakin")
    for i, ad in enumerate(AD):
        im = lambda v: "  OKUNAMAZ" if v < 1e-9 else f"{v:9.3f}"
        print(f"  {ad:<10s} {im(a[i])}  {im(c[i])}  {im(y[i])}")

    print()
    for ad_k, p in (("A ic carpim", a), ("B kosinus", c), ("C en yakin", y)):
        yok = [AD[i] for i in range(N) if p[i] < 1e-9]
        print(f"  {ad_k:<12s} okunamayan {len(yok)}: {', '.join(yok) or '-'}")


if __name__ == "__main__":
    rapor()
