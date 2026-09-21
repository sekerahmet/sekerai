"""model_15 -- YOL TUTULMAZ, HESAPLANIR.

  gez       s_t = normalize(M[token] @ s_{t-1} + b[token])
            ozyineleme -- yuvalar soldan saga kuruluyor
  blok      HER yuva sorar, j <= i'ye bakar, sonuc yuvaya EKLENIR
            katman kadar tekrarlanir -> bilesik hesap mumkun
  agirlik   RELU -- softmax DEGIL.  1'e toplanmak zorunda olsaydi
            cikti hep ORTALAMA olurdu, TOPLAM tasinamazdi.
  cikti     son yuvanin durumu @ Wson -> en yakin E[token]
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

# ============================================================
# AYARLAR.  Ayri bir ayar dosyasi YOK.
# Iki ayri gorevde olculdu, karistirilmasin:
#   GOREV A  cevap TEK token   sozluk 1003
#   GOREV B  cevap RAKAM RAKAM sozluk 13, 251.001 cift, olcut SAYI
# ============================================================

# --- OLCULDU, GOREV A  (2.601 ikili, olcut: tutulan)
#   boyut   2:0,051  4:0,088  8:0,841  16:0,901  32:0,878
#   norm    ACIK 0,782   KAPALI 0,059
#   pay     softmax 0,289   relu 0,782
#   lr      0,005:0,008  0,01:0,025  0,02:0,810  0,04:0,930  0,08:0,146
#           sabit 0,897   cosine->lr/10 0,080   (cosine ZARARLI, cizelge YOK)
#   lr OLCEGE BAGLI:  2.601 ikili -> 0,04 (0,930)
#                     251.001     -> 0,004 (0,980); 0,04 ile 0,086'da takiliyor
#                     veri 96,5 kat, lr 10 kat kucuk.  sqrt(96,5)=9,82.
#                     TAHMIN lr = 0,04 * sqrt(2601/veri)
#                     SINIR: yigin da degisti, tek degiskenli yasa DEGIL.
#   wd      251.001 ikilide:  0,03 OLU (kayip tam ln(1003)'te dondu)
#                             0,01 calisiyor,  0,001 ezbere kayiyor

# --- OLCULDU, GOREV B  (rakam tokenli, 4000 adim, olcut SAYI)
#   durum   8:0,0148   16:0,0322   32:0,0342      dirsek 16
#   esik    kapali 0,0224 -> acik 0,1517  (dolgu "0")
#           kapali 0,0258 -> acik 0,0356  (dolgu PAD)     EN BUYUK KAZANC
#   kafa    1:0,0224   4:0,0329
#   olcek   kapali 0,0224   acik 0,0284
BOYUT = 16           # token kac sayiyla tarif ediliyor
DURUM = 16           # s kac sayi
NORM = True          # |s| = 1
PAY = False          # False -> relu   True -> softmax
ESIK = True          # relu(puan + hb), hb OGRENILEN esik
OLCEK = False        # puan / sqrt(durum).  Kucuk fayda, ESIK ile birlikte
                     #   olculmedi -- acmadan once olcmek gerek.
KAFA = 1             # kac dikkat kafasi.  4 kucuk fayda verdi, ayni not.
LR = 0.002           # SABIT.  Rakam tokenli veride 0,001/0,002/0,004
                     #   ayirt edilemedi (hepsi ~0,015).
WD = 0.01            # 0,03 buyuk veride OLDURUYOR, 0,001 ezbere kaydiriyor.

# --- OLCULMEDI
KATMAN = 1           # kac dikkat BLOGU.  Blok yuvalari GUNCELLER (artik
                     #   baglanti, nedensel maske), havuzlanmis ciktiyi degil:
                     #   yuva i, yuva j'nin ZATEN HESAP YAPMIS halini gorur.
                     #   Neden aday: onlar basamagi BUTUN veri bicimlerinde
                     #   0,17-0,31'de takili, ve ogretmenli gecmis verilse
                     #   bile duzelmiyor -- bilgi eksikligi, birikme degil.


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, durum=DURUM, tohum=0,
                 norm=NORM, pay=PAY, olcek=OLCEK, esik=ESIK, kafa=KAFA,
                 katman=KATMAN):
        """Ayarlar dosyanin basinda -- ayri bir ayar dosyasi YOK."""
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum = n, boyut, durum
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

    def uret(self, w, adim):
        """Cikti girdiye eklenir.  Uzunluk SABIT oldugu icin durma kosulu yok."""
        w, cikan = list(w), []
        for _ in range(adim):
            c = self.oku(self.dikkat(torch.tensor(w))[0])
            cikan.append(c)
            w.append(c)
        return cikan, w
