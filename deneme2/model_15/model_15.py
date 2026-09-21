"""model_15 -- YOL TUTULMAZ, HESAPLANIR.  ADIM 2.

Girdi degisken uzunlukta bir token dizisi:
  [Istanbul]  ya da  [Istanbul, Ankara]  ya da  [Istanbul, Ankara, Mersin]
Her biri KENDI durumunu uretir.  Hicbir yerde "yol" diye bir sey saklanmaz.

DURUM ile SOZLUK ayri boyutlarda:  token bir konum (2 sayi), durum ise
bir yolu tasimak zorunda -- ikisi ayni sey degil.
"""
import torch
import torch.nn as nn


class Yol(nn.Module):
    def __init__(self, n, boyut=2, durum=6, tohum=0):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum = n, boyut, durum

        # SOZLUK: her token'a RASTGELE bir konum.  Okumanin HEDEFI.
        self.E = nn.Parameter(r(n, boyut))

        # GIRIS: token duruma nasil girer.  (boyut -> durum)
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)

        # GUNCELLEME: her token'a bir matris.  Donme DEGIL:
        # 2B'de donmeler degismeli, sirayi goremezdi (TASARIM ADIM 1).
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)
                              + 0.1 * r(n, durum, durum))

        # OKUMA IZDUSUMU: durum -> sozluk uzayi.  (durum -> boyut)
        self.Q = nn.Parameter(r(durum, boyut) / durum ** 0.5)

        self.s0 = nn.Parameter(torch.zeros(durum))

    def yol(self, w):
        """w: (T,) token dizisi  ->  (T*2,) ugranan KONUMLARIN birlesimi.

        Istanbul U Ankara U Mersin  =  [x_Ist y_Ist x_Ank y_Ank x_Mer y_Mer]
        Uzunluk yolla birlikte buyur.  Hicbir sey sikismaz, hicbir sey atilmaz.
        """
        return self.E[w].reshape(-1)


    def ileri(self, w):
        """w: (T,) token dizisi.  Doner: (T+1, durum) -- her adimdaki durum."""
        s = self.s0
        iz = [s]
        for t in w:
            s = self.M[t] @ s + self.b[t]
            iz.append(s)
        return torch.stack(iz)

    def golge(self, s):
        """durum -> sozluk uzayi."""
        return s.reshape(-1, self.durum) @ self.Q

    def oku(self, s):
        """argmin_c |golge(s) - E[c]|   (okuma kurali: TASARIM ADIM 1b)."""
        return torch.cdist(self.golge(s), self.E).argmin(-1)


def _goster_yol():
    from sehir_15 import AD, N, dizi
    m = Yol(N, boyut=2, durum=6, tohum=0)
    print("YOL = ugranan konumlarin birlesimi")
    print()
    for onek in (["Istanbul"], ["Istanbul", "Ankara"],
                 ["Istanbul", "Ankara", "Mersin"],
                 ["Izmir", "Bursa", "Mersin"],
                 ["Konya", "Ankara", "Mersin"]):
        v = m.yol(dizi(onek))
        d = " ".join(f"{x:6.2f}" for x in v)
        print(f"  {' '.join(onek):<24s} {len(v)} sayi   [{d}]")


def _goster():
    from sehir_15 import AD, N, dizi
    m = Yol(N, boyut=2, durum=6, tohum=0)
    print(f"sozluk {m.boyut}   durum {m.durum}   token {N}")
    for ad, p in m.named_parameters():
        print(f"  {ad:<4s} {tuple(p.shape)}  {p.numel():4d}")
    print(f"  toplam {sum(p.numel() for p in m.parameters())}\n")

    print("SOZLUK  (rastgele, egitilecek)")
    for i, a in enumerate(AD):
        print(f"  {a:<10s} {m.E[i,0]:7.3f} {m.E[i,1]:7.3f}")

    print("\nAYNI YOLUN UC ONEKI -- her biri AYRI hesap, EGITILMEMIS")
    for onek in (["Istanbul"], ["Istanbul", "Ankara"], ["Istanbul", "Ankara", "Mersin"]):
        s = m.ileri(dizi(onek))[-1]
        gl = m.golge(s)[0]
        d = "  ".join(f"{v:6.2f}" for v in s)
        print(f"  {' '.join(onek):<24s} s = [{d}]"
              f"   golge ({gl[0]:6.2f} {gl[1]:6.2f})  -> {AD[m.oku(s).item()]}")


if __name__ == "__main__":
    with torch.no_grad():
        _goster()
