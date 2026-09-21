"""model_15 -- YOL TUTULMAZ, HESAPLANIR.

  yol       ugranan konumlarin birlesimi; uzunluk yolla buyur
  soru      YOLUN TAMAMI sorar:  s = gez(yol),  q = s @ Wq
  cevap     her yuva Wk ile puanlanir, Wv ile katki verir
  cikti     agirlikli toplam -> sozluk uzayinda bir nokta -> en yakin token
  durma     DUR token'i
"""
import torch
import torch.nn as nn


class Yol(nn.Module):
    def __init__(self, n, boyut=2, durum=6, dur=None, tohum=0):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum, self.dur = n, boyut, durum, dur

        self.E = nn.Parameter(r(n, boyut))                  # sozluk
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)   # token -> duruma giris
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)
                              + 0.1 * r(n, durum, durum))   # guncelleme
        self.s0 = nn.Parameter(torch.zeros(durum))

        self.Wq = nn.Parameter(r(durum, boyut) * 0.4)       # yol -> soru
        self.Wk = nn.Parameter(r(durum, boyut) * 0.4)       # YUVANIN DURUMU -> anahtar
        self.Wv = nn.Parameter(r(durum, boyut) * 0.4)       # YUVANIN DURUMU -> deger

    def gez(self, w):
        """Yolu bastan gez, her adimdaki durumu dondur.  (T+1, durum)"""
        s, iz = self.s0, [self.s0]
        for t in w:
            s = self.M[t] @ s + self.b[t]
            iz.append(s)
        return torch.stack(iz)

    def dikkat(self, w):
        """Yol sorar, yuvalar cevaplar.  Doner: (boyut,) ve agirliklar (T,).

        Anahtar ve deger yuvanin KONUMUNDAN degil, o ana kadarki DURUMUNDAN
        uretiliyor -- boylece ayni sehir iki farkli yerde ayni sey demiyor.
        """
        S = self.gez(w)[1:]                    # (T, durum) her yuvanin durumu
        q = S[-1] @ self.Wq                    # soru: yolun tamami
        ag = (S @ self.Wk @ q).softmax(0)      # agirlik -- UZUNLUK SERBEST
        return ag @ (S @ self.Wv), ag

    def oku(self, o):
        """Sozluk uzayindaki noktaya en yakin token."""
        return int(torch.cdist(o.reshape(1, -1), self.E).argmin())

    def uret(self, w, en_fazla=8):
        """Cikti girdiye eklenir; DUR gorunce durur."""
        w, cikan = list(w), []
        for _ in range(en_fazla):
            o, _ag = self.dikkat(torch.tensor(w))
            c = self.oku(o)
            cikan.append(c)
            w.append(c)                        # <- guzergah buyudu
            if c == self.dur:
                break
        return cikan, w
