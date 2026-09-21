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

# BELIRLENIMCI KOSU.  Cok iplikli toplama sirasi her kosuda degisiyordu;
# ondalik farklar 400 adimda buyuyup sonucu ~0,29 / ~0,99 arasinda
# ziplatiyordu.  Ayni tohum ayni sonucu vermeliydi, vermiyordu.
# GPU'da bazi cekirdekler belirlenimci degil; orada ZORLAMIYORUZ,
# tekrarlanabilirlik CPU kosularinda gecerli.
if not torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

# --- OLCULMUS AYARLAR  (veri_15, 2601 ikili, %50 egitim, taban 0,028)
#   boyut   2:0,051  4:0,088  8:0,841  16:0,901  32:0,878
#   norm    ACIK 0,782   KAPALI 0,059
#   pay     softmax 0,289   relu 0,782
#   lr      0,005:0,008  0,01:0,025  0,02:0,810  0,04:0,930  0,08:0,146
#           sabit 0,897   cosine->lr/10 0,080
#   wd x lr  5 tohum, "kac tohumda tutulan > 0,50":
#           0,01/0,02 1/5   0,01/0,04 5/5
#           0,03/0,02 4/5   0,03/0,04 5/5 (ort 0,929, en kotu tohum 0,83)
BOYUT = 16           # token kac sayiyla tarif ediliyor
NORM = True          # |s| = 1
PAY = False          # False -> relu   True -> softmax
LR = 0.04            # SABIT.  En keskin ayar: 0,01'de genelleme 0,025,
                     #   0,02'de 0,810 -- iki kat lr, 32 kat fark.
                     #   Dusuk lr ilk buldugu cozume (EZBER) yerlesiyor;
                     #   0,01'de hafiza 1,000 ama kural YOK.
                     #   0,08'de egitim cokuyor (hafiza 0,380).
COSINE = False

# --- OLCULMEMIS  -- dikkat cokmesine karsi uc aday (21 Eylul)
#   Olculen ariza: rakam uretimi ilerledikce dikkat operandlardan
#   kopuyor.  adim 0'da ilk sayinin agirligi 2,31 -- adim 2'de 0,004.
#   Toplam agirlik 4,66 -> 1,11, cikti buyuklugu 4,51 -> 0,83,
#   dogruluk 1,00 -> 0,10.  Uc aday, ucu de relu'dan BAGIMSIZ:
OLCEK = False        # puan / sqrt(durum) -- transformerdaki gibi
ESIK = False         # relu(puan + hb), hb OGRENILEN esik
KAFA = 1             # kac dikkat kafasi
KATMAN = 1           # kac dikkat BLOGU.  Blok yuvalari GUNCELLER (artik
                     #   baglanti), havuzlanmis ciktiyi degil -- boylece
                     #   yuva i, yuva j'nin ZATEN HESAP YAPMIS halini gorur.
                     #   Derinlik 1'de bilesik hesap yapilamiyor:
                     #   onlar basamagi butun veri bicimlerinde 0,17-0,31.

WD = 0.03            # ceza.  lr ile birlikte calisiyor: biri zayifsa
                     #   oteki telafi ediyor, ikisi guclu olunca 5/5.

# --- LR OLCEGE BAGLI  (21 Eylul, Colab)
#   2.601 ikili   ->  lr 0,04     tutulan 0,930
#   251.001 ikili ->  lr 0,004    tutulan 0,980   (0,04 ile 0,086'da takiliyor)
#   veri 96,5 kat buyudu, lr 10 kat kuculdu.  sqrt(96,5) = 9,82.
#   TAHMIN:  lr = 0,04 * sqrt(2601 / veri)
#   SINIR: yigin da degisti (1.300 tam -> 25.000), tek degiskenli yasa DEGIL.
#   wd 0,03 ayni geciste OLDU (kayip tam ln(1003)'te dondu); 0,01 calisiyor.

# --- OLCULMEMIS  -- tasindi, gerekcesi YOK
DURUM = 16           # s kac sayi.  Sehir doneminde 6'ydi.


def lr_ver(i, adim, lr=LR, cosine=COSINE):
    """Cosine inis, tabani lr/10.  cosine=False -> sabit."""
    if not cosine:
        return lr
    import math
    return lr / 10 + (lr - lr / 10) * 0.5 * (1 + math.cos(math.pi * i / adim))


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, durum=DURUM, dur=None, tohum=0,
                 norm=NORM, pay=PAY, olcek=OLCEK, esik=ESIK, kafa=KAFA,
                 katman=KATMAN):
        """Ayarlar dosyanin basinda -- ayri bir ayar dosyasi YOK."""
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum, self.dur = n, boyut, durum, dur
        self.norm, self.pay = norm, pay
        self.olcek = durum ** -0.5 if olcek else 1.0
        self.kafa, self.katman = kafa, katman

        self.E = nn.Parameter(r(n, boyut))                  # sozluk: token -> konum
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)   # token -> duruma giris
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)
                              + 0.1 * r(n, durum, durum))   # guncelleme
        self.s0 = nn.Parameter(torch.zeros(durum))

        L = katman
        self.Wq = nn.Parameter(r(L, kafa, durum, boyut) * 0.4)  # yuva -> soru
        self.Wk = nn.Parameter(r(L, kafa, durum, boyut) * 0.4)  # yuva -> anahtar
        self.Wv = nn.Parameter(r(L, kafa, durum, boyut) * 0.4)  # yuva -> deger
        self.Wo = nn.Parameter(r(L, kafa, boyut, durum) * 0.4)  # geri duruma
        self.hb = nn.Parameter(torch.zeros(L, kafa)) if esik else None
        self.Wson = nn.Parameter(r(durum, boyut) * 0.4)         # son -> sozluk

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

    def blok(self, S, L):
        """Bir dikkat blogu.  Yuvalari GUNCELLER, havuzlamaz.
        Nedensel: yuva i yalniz j <= i'ye bakar (uretim sirasiyla ayni)."""
        q = torch.einsum("btd,hdc->bhtc", S, self.Wq[L])
        k = torch.einsum("btd,hdc->bhtc", S, self.Wk[L])
        v = torch.einsum("btd,hdc->bhtc", S, self.Wv[L])
        pu = torch.einsum("bhic,bhjc->bhij", q, k) * self.olcek
        if self.hb is not None:
            pu = pu + self.hb[L][None, :, None, None]
        T = S.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=S.device).tril()
        pu = pu.masked_fill(~mask, -1e9 if self.pay else -float("inf"))
        ag = pu.softmax(-1) if self.pay else pu.clamp(min=0)
        u = torch.einsum("bhij,bhjc->bhic", ag, v)
        return torch.einsum("bhic,hcd->bid", u, self.Wo[L]), ag

    def dikkat(self, w):
        """Yol sorar, yuvalar cevaplar.  katman kadar blok ust uste.

        Anahtar ve deger yuvanin TOKEN'INDAN degil DURUMUNDAN uretiliyor.
        """
        tek = w.dim() == 1
        S = self.gez(w)[..., 1:, :]
        S = S[None] if tek else S
        ag = None
        for L in range(self.katman):
            u, ag = self.blok(S, L)
            S = S + u
            if self.norm:
                S = F.normalize(S, dim=-1)
        o = S[:, -1] @ self.Wson
        ag = ag[:, :, -1, :]                     # son yuvanin agirliklari
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
