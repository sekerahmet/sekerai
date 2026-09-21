"""model_15 -- YOL TUTULMAZ, HESAPLANIR.

  yol       token dizisi; uzunluk serbest
  gez       s_t = normalize(M[token] @ s_{t-1} + b[token])
  soru      YOLUN TAMAMI sorar:  q = s_son @ Wq
  cevap     her yuva Wk ile puanlanir, Wv ile katki verir
  agirlik   RELU -- softmax DEGIL.  1'e toplanmak zorunda olsaydi
            cikti hep ORTALAMA olurdu ve toplam TASINAMAZDI.
  cikti     agirlikli TOPLAM -> sozluk uzayinda bir nokta -> en yakin token
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- OLCULMUS AYARLAR  (iki_15, 2601 ikili, %50 egitim, taban 0,028)
#   boyut   2:0,059   4:0,094   8:0,782
#   norm    ACIK 0,782   KAPALI 0,059
#   pay     softmax 0,289   relu 0,782
BOYUT, DURUM, NORM, PAY = 8, 16, True, False

# --- OLCULMEMIS  (tasindi, gerekcesi yok)
#   DURUM 16   sehir doneminde 6'ydi
#   wd 0,03    sehir verisinde olculdu, toplamada olculmedi
#   lr 0,02    hic olculmedi


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, durum=DURUM, dur=None, tohum=0,
                 norm=NORM, pay=PAY):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum, self.dur = n, boyut, durum, dur
        self.norm, self.pay = norm, pay

        self.E = nn.Parameter(r(n, boyut))                  # sozluk: token -> konum
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)   # token -> duruma giris
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)
                              + 0.1 * r(n, durum, durum))   # guncelleme
        self.s0 = nn.Parameter(torch.zeros(durum))

        self.Wq = nn.Parameter(r(durum, boyut) * 0.4)       # yol -> soru
        self.Wk = nn.Parameter(r(durum, boyut) * 0.4)       # yuva -> anahtar
        self.Wv = nn.Parameter(r(durum, boyut) * 0.4)       # yuva -> deger

    def gez(self, w):
        """w: (T,) ya da (B,T).  Doner: (B,T+1,durum) -- her adimdaki durum."""
        tek = w.dim() == 1
        w = w[None] if tek else w
        s = self.s0.expand(w.shape[0], self.durum)
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

        Anahtar ve deger yuvanin TOKEN'INDAN degil, o ana kadarki DURUMUNDAN
        uretiliyor -- boylece ayni token iki farkli yerde ayni sey demiyor.
        """
        tek = w.dim() == 1
        S = self.gez(w)[..., 1:, :]
        S = S[None] if tek else S
        q = S[:, -1] @ self.Wq
        pu = ((S @ self.Wk) @ q.unsqueeze(-1)).squeeze(-1)
        ag = pu.softmax(-1) if self.pay else pu.relu()
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
            w.append(c)
            if c == self.dur:
                break
        return cikan, w
