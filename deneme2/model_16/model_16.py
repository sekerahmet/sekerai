"""model_16 -- YOL TUTULMAZ, HESAPLANIR.

  gez       s_t = normalize(M[token] @ s_{t-1} + b[token])
            ozyineleme -- yuvalar soldan saga kuruluyor
  soru      SON yuva sorar:  q = s_son @ Wq
  cevap     her yuva Wk ile puanlanir, Wv ile katki verir
  agirlik   relu(puan + hb) -- softmax DEGIL.  1'e toplanmak zorunda
            olsaydi cikti hep ORTALAMA olurdu, TOPLAM tasinamazdi.
  cikti     agirlikli TOPLAM -> sozluk uzayinda nokta -> en yakin E[token]
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
#           kapali 0,0258 -> acik 0,0356  (dolgu PAD)
#
#   YUVA BASINA DOGRULUK -- dort tasarim, ayni veri, ayni olcut.
#   Uc alternatif DENENDI ve bu surum geri getirildi (22 Eylul):
#     M/b ozyineleme   1,0000  0,9416  0,4250  0,1207   8000 adim  4.529 par
#     E+P, ozet YOK    1,0000  0,1810  0,1024  0,0987   4000 adim  1.361 par
#     E+P+A ozet       1,0000  0,8968  0,1107  0,1002   4000 adim  1.617 par
#     M+E (b silinmis) 1,0000  0,8702  0,1443  0,1052   4000 adim  4.305 par
#   yuva3'u yalniz bu surum kaldirabildi (0,4250; digerlerinde ~0,10-0,14,
#   sans 0,10).  Alt satirlar YARI butce -- kiyas o yuzden kesin degil.
#
#   BILINEN KUSUR, SURUYOR:  token iceri b[w] ile giriyor, disari E[w] ile
#   okunuyor ve bu ikisi AYRI parametre.  Olculdu: cos(b,E) = 0,174,
#   rastgele 16 boyutta beklenen 0,199 -- yani bagsizlar.  Birlestirmek
#   DENENDI (yukarida M+E) ve yuva2'yi 0,9416'dan 0,8702'ye dusurdu.
BOYUT = 16           # token kac sayiyla tarif ediliyor
DURUM = 16           # s kac sayi
NORM = True          # |s| = 1
PAY = False          # False -> relu   True -> softmax
LR = 0.002           # SABIT.  Rakam tokenli veride 0,001/0,002/0,004
                     #   ayirt edilemedi (hepsi ~0,015).
WD = 0.01            # 0,03 buyuk veride OLDURUYOR, 0,001 ezbere kaydiriyor.

# YANLILIK (hb) ANAHTAR DEGIL, HER ZAMAN VAR.  Ispat, kosu gerekmez:
#   ag_j = relu(q.k_j) ise bir yuva ancak q.k_j > 0 iken agirlik alir.
#   Bu sinir ORIJINDEN GECEN bir duzlem -- secim homojen yarim uzaya
#   hapsolur.  Anahtarlar k_j = s_j @ Wk ve s_j normalize; anahtar bulutunun
#   orijini icermesi icin sebep yok.  Bulut tek taraftaysa secim
#   "HEPSI" ya da "HICBIRI"ne duser, arasi yok.
#   Gozlenen (hb eklenmeden ONCE): adim 0'da toplam agirlik 4,66;
#   adim 2'de 10 yuvanin 9'u sifir, toplam 1,11.  "Hicbiri" hali.
#   hb ile sinir q.x = -hb olur, genel yarim uzay.  Maliyet: 1 sayi.
#
# CIKARILDI (21 Eylul): KATMAN/blok, nedensel maske, artik baglanti,
#   Wson okuma yolu, OLCEK, KAFA.  Hicbiri olculmedi, hicbirinin kosusuz
#   gerekcesi yoktu.  Tek ornekte katman 2 kurulur kurulmaz OLUYDU
#   (butun agirliklar 0) ve |u|=5,8 / |s|=1, yani "artik baglanti"
#   durumu inceltmiyor UZERINE YAZIYORDU.
#   Once sorulacak soru: TOPLAMI hangi parca yapacak?


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, durum=DURUM, tohum=0,
                 norm=NORM, pay=PAY):
        """Ayarlar dosyanin basinda -- ayri bir ayar dosyasi YOK."""
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.durum = n, boyut, durum
        self.norm, self.pay = norm, pay

        self.E = nn.Parameter(r(n, boyut))                  # sozluk: token -> konum
        self.b = nn.Parameter(r(n, durum) / durum ** 0.5)   # token -> duruma giris
        self.M = nn.Parameter(torch.eye(durum).repeat(n, 1, 1)
                              + 0.1 * r(n, durum, durum))   # guncelleme
        self.s0 = nn.Parameter(torch.zeros(durum))

        # BASLANGIC OLCEGI  o = 1/sqrt(boyut).  Turetilisi:
        #   W'nin girisleri bagimsiz, ortalama 0, standart sapma o ise
        #   (x @ W)_j = toplam_i x_i W_ij  ->  varyansi  |x|^2 * o^2
        #   ve j uzerinden boyut tane terim var:
        #     |x @ W| ~ |x| * o * sqrt(boyut)        <- CIKTI boyutu
        #   Cikti boyu girdi boyuyla AYNI kalsin istiyoruz:
        #     o * sqrt(boyut) = 1   ->   o = 1/sqrt(boyut) = 0,25   (boyut 16)
        #   Boylece |q| ~ |s| = 1 ve |k| ~ 1; puan = q.k ne patlar ne soner.
        #   ONCEKI DEGER 0,4 idi ve GEREKCESI YOKTU -- elle yazilmisti.
        #   (b zaten 1/sqrt(durum) ile basliyordu; simdi ikisi tutarli.)
        #   NOT: 0,9416/0,4250 sayilari 0,4 ile olculdu.  Bu degisiklikten
        #   sonraki kosu o sayilarla BIT DUZEYINDE ayni olmaz.
        o = boyut ** -0.5
        self.Wq = nn.Parameter(r(durum, boyut) * o)     # son yuva -> soru
        self.Wk = nn.Parameter(r(durum, boyut) * o)     # yuva -> anahtar
        self.Wv = nn.Parameter(r(durum, boyut) * o)     # yuva -> deger
        self.hb = nn.Parameter(torch.zeros(1))          # YANLILIK -- hep var

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
        """Son yuva sorar, butun yuvalar cevaplar.

        Anahtar ve deger yuvanin TOKEN'INDAN degil DURUMUNDAN uretiliyor --
        boylece ayni token iki farkli yerde ayni sey demiyor.
        """
        tek = w.dim() == 1
        S = self.gez(w)[..., 1:, :]
        S = S[None] if tek else S
        q = S[:, -1] @ self.Wq
        pu = (S @ self.Wk) @ q.unsqueeze(-1)
        pu = pu.squeeze(-1) + self.hb
        ag = pu.softmax(-1) if self.pay else pu.clamp(min=0)
        o = (ag.unsqueeze(-1) * (S @ self.Wv)).sum(1)
        return (o[0], ag[0]) if tek else (o, ag)

    def dizi(self, w):
        """HER yuva kendi sorusunu sorar.  w (B,T) -> puan (B,T,n).

        `dikkat`in COGUL hali.  Mekanizma AYNI -- relu dikkat, soru
        durumdan, cevap durumlardan, okuma en yakin E.  Degisen tek sey
        KIMIN sordugu: son yuva yerine hepsi.

        NEDEN GEREKLI.  Aritmetikte bir soruyu AYRI SATIRLARA boluyorduk:
          4 7 2 + 1 8 2 =        -> 6      8 token
          4 7 2 + 1 8 2 = 6      -> 5      9 token   ...
        yani "her onek icin bir ileri gecis".  Onek 4 taneydi ve kisaydi:
        8+9+10+11 = 38 token-adimi, tek satirda 11 olurdu -- 3,5 kat.
        512'lik pencerede ayni yontem 1+2+...+511 = 130.816 token-adimi
        demek; tek satirda 512.  Oran T/2 = 256.  Ayni hesap, 256 kat is.

        NEDENSEL MASKE.  Ayri satir yonteminde `= 6` satiri `5`i
        GOREMIYORDU, cunku satirda yoktu.  Tek satira gecince hepsi ayni
        dizide; maske tam olarak o goremezligi geri koyuyor.  `dikkat`te
        gerekmiyordu (yalniz son yuva soruyor, gelecegi yok) ve o yuzden
        21 Eylul'de "gerekcesiz" diye cikarilmisti -- dogru karardi.

        relu ONCE maskelenir: puan gelecege bakmasin, sonra relu.
        Maskelenen yere -inf DEGIL, dogrudan agirlik 0 verilir; softmax
        olsaydi -inf gerekirdi ama burada pay YOK (relu).
        """
        assert w.dim() == 2, "dizi() (B,T) bekler"
        S = self.gez(w)[:, 1:]                      # (B,T,durum)
        Q, K, V = S @ self.Wq, S @ self.Wk, S @ self.Wv
        P = Q @ K.transpose(1, 2) + self.hb         # (B,T,T)
        gec = torch.ones(w.shape[1], w.shape[1], dtype=torch.bool,
                         device=w.device).tril()    # j yalniz <= j'ye bakar
        A = (P.masked_fill(~gec, -torch.inf).softmax(-1) if self.pay
             else P.clamp(min=0) * gec)
        O = A @ V                                   # (B,T,boyut)  bmm --
        #   ara (B,T,T,boyut) tensor URETMEZ; carpim olarak yazilmasi sart,
        #   yayilimla yazilsaydi B=64'te bile 1 GB olurdu.
        return -torch.cdist(O, self.E.expand(w.shape[0], -1, -1)) ** 2

    def uret_dizi(self, w, adim):
        """w (B,L) -> uretilen (B,adim).  DURUM TASINIR.

        `dizi`nin SON satirini adim adim yurutur.  Naif yol her yeni
        karakter icin ozyinelemeyi bastan calistirirdi: L uzunlugunda bir
        onekten k karakter uretmek k*L adim eder (L=60, k=30 -> 1.800).
        Burada durum tasindigi icin k adim yeter -- 60 kat.
        """
        S = self.gez(w)[:, 1:]                       # (B,L,durum)
        s = S[:, -1]
        cikan = []
        for _ in range(adim):
            q = s @ self.Wq
            pu = (S @ self.Wk) @ q.unsqueeze(-1)
            pu = pu.squeeze(-1) + self.hb
            ag = pu.softmax(-1) if self.pay else pu.clamp(min=0)
            o = (ag.unsqueeze(-1) * (S @ self.Wv)).sum(1)
            t = torch.cdist(o[:, None], self.E.expand(w.shape[0], -1, -1))
            t = t.squeeze(1).argmin(-1)              # (B,)
            cikan.append(t)
            s = torch.bmm(self.M[t], s.unsqueeze(-1)).squeeze(-1) + self.b[t]
            if self.norm:
                s = F.normalize(s, dim=-1)
            S = torch.cat([S, s[:, None]], 1)
        return torch.stack(cikan, 1)

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
