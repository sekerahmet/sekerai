"""model_15 -- YOL TUTULMAZ, HESAPLANIR.

  yol       ugranan konumlarin birlesimi; uzunluk yolla buyur
  soru      YOLUN TAMAMI sorar:  s = gez(yol),  q = s @ Wq
  cevap     her yuva Wk ile puanlanir, Wv ile katki verir
  cikti     agirlikli toplam -> sozluk uzayinda bir nokta -> en yakin token
  durma     DUR token'i
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Yol(nn.Module):
    def __init__(self, n, boyut=2, durum=6, dur=None, tohum=0, norm=True):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum, self.dur = n, boyut, durum, dur
        self.norm = norm       # |s|=1 -- yoksa 7 adimda 49 kat siser (tani_15)

        self.E = nn.Parameter(r(n, boyut))                  # sozluk
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)   # token -> duruma giris
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)
                              + 0.1 * r(n, durum, durum))   # guncelleme
        self.s0 = nn.Parameter(torch.zeros(durum))

        self.Wq = nn.Parameter(r(durum, boyut) * 0.4)       # yol -> soru
        self.Wk = nn.Parameter(r(durum, boyut) * 0.4)       # YUVANIN DURUMU -> anahtar
        self.Wv = nn.Parameter(r(durum, boyut) * 0.4)       # YUVANIN DURUMU -> deger

    def gez(self, w):
        """w: (T,) ya da (B,T).  Doner: (B,T+1,durum) -- her adimdaki durum."""
        tek = w.dim() == 1
        w = w[None] if tek else w
        B = w.shape[0]
        s = self.s0.expand(B, self.durum)
        iz = [s]
        for t in range(w.shape[1]):
            wt = w[:, t]
            s = torch.bmm(self.M[wt], s.unsqueeze(-1)).squeeze(-1) + self.b[wt]
            if self.norm:
                s = F.normalize(s, dim=-1)
            iz.append(s)
        S = torch.stack(iz, 1)
        return S[0] if tek else S

    def dikkat(self, w):
        """Yol sorar, yuvalar cevaplar.  Doner: cikti ve agirliklar.

        Anahtar ve deger yuvanin KONUMUNDAN degil, o ana kadarki DURUMUNDAN
        uretiliyor -- boylece ayni sehir iki farkli yerde ayni sey demiyor.
        """
        tek = w.dim() == 1
        S = self.gez(w)[..., 1:, :]            # (B,T,durum) her yuvanin durumu
        S = S[None] if tek else S
        q = S[:, -1] @ self.Wq                 # (B,boyut) soru: yolun tamami
        ag = ((S @ self.Wk) @ q.unsqueeze(-1)).squeeze(-1).softmax(-1)
        o = (ag.unsqueeze(-1) * (S @ self.Wv)).sum(1)
        return (o[0], ag[0]) if tek else (o, ag)

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
