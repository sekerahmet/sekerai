"""model_15 -- YOL TUTULMAZ, HESAPLANIR.  ADIM 2.

Girdi degisken uzunlukta bir token dizisi:
  [Istanbul]  ya da  [Istanbul, Ankara]  ya da  [Istanbul, Ankara, Mersin]
Her biri KENDI durumunu uretir.  Hicbir yerde "yol" diye bir sey saklanmaz.
"""
import torch
import torch.nn as nn


class Yol(nn.Module):
    def __init__(self, n, boyut=2, tohum=0):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut = n, boyut

        # SOZLUK: her token'a RASTGELE bir konum.
        # Hem girdi hem okumanin hedefi -- tek tablo.
        self.E = nn.Parameter(r(n, boyut))

        # GUNCELLEME: her token'a bir matris.  Donme DEGIL:
        # 2B'de donmeler degismeli, sirayi goremezdi (TASARIM ADIM 1).
        self.M = nn.Parameter(torch.eye(boyut).repeat(n, 1, 1) + 0.1 * r(n, boyut, boyut))

        self.s0 = nn.Parameter(torch.zeros(boyut))

    def ileri(self, w):
        """w: (T,) token dizisi.  Doner: (T+1, boyut) -- her adimdaki durum."""
        s = self.s0
        iz = [s]
        for t in w:
            s = self.M[t] @ s + self.E[t]
            iz.append(s)
        return torch.stack(iz)

    def oku(self, s):
        """argmin_c |s - E[c]|   (okuma kurali: TASARIM ADIM 1b)."""
        return torch.cdist(s.reshape(-1, self.boyut), self.E).argmin(-1)


def _goster():
    from sehir_15 import AD, IX, N, dizi
    m = Yol(N, boyut=2, tohum=0)
    print(f"parametre  E {m.E.numel()}  M {m.M.numel()}  s0 {m.s0.numel()}"
          f"   toplam {sum(p.numel() for p in m.parameters())}\n")

    print("SOZLUK  (rastgele, egitilecek)")
    for i, a in enumerate(AD):
        print(f"  {a:<10s} {m.E[i,0]:7.3f} {m.E[i,1]:7.3f}")

    print("\nAYNI YOLUN UC ONEKI -- her biri AYRI hesap, EGITILMEMIS")
    for onek in (["Istanbul"], ["Istanbul", "Ankara"], ["Istanbul", "Ankara", "Mersin"]):
        iz = m.ileri(dizi(onek))
        s = iz[-1]
        print(f"  {' '.join(onek):<28s} s = ({s[0]:6.3f} {s[1]:6.3f})"
              f"   -> {AD[m.oku(s).item()]}")


if __name__ == "__main__":
    with torch.no_grad():
        _goster()
