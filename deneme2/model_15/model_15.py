"""model_15 -- YOL, UGRANAN KONUMLARIN LISTESIDIR.

Kullanici, 21 Eylul:  "yol su degil mi? istanbul(x,y) U ankara U mersin.
bu sayede hangi konumdan hangi konuma gittigimi bilirim."
ve 22 Eylul: "E1 + 6 = E2 oluyor.  ana tasarimin kalbi bu."

  sozluk    E[token] -- her token'in KONUMU.  Girdi de bu, okumanin hedefi de.
  yer       P[j]     -- j. yuvaya ait konum.  Yol bir LISTE, ve listedeki
            YER bilgi tasiyor:  "_ 3 4 + _ 5 6 =" ile "_ _ 3 + 4 5 6 ="
            ayni token TORBASINI verir ama farkli sayilardir (34+56=90,
            3+456=459).  P olmadan model ikisini ayirt edemiyordu.
  yol       yuva_j = E[w_j] + P[j]            konum + yer
  ozet      s_j = normalize(A @ s_{j-1} + yuva_j)
            Her yuva hem KENDINI hem ONEKINI tasir.  A PAYLASIMLI tek
            matris -- token diziye kendi E'siyle giriyor, ayri bir
            b[token] yok.  (Eski surumde M[token]/b[token] vardi ve E
            girdiye hic girmiyordu: cos(b,E)=0,174, rastgeleyle ayni.)
  soru      son yuva sorar:  q = E[w_son] @ Wq
  cevap     her yuva Wk ile puanlanir, Wv ile katki verir
  agirlik   relu(puan + hb) -- softmax DEGIL.  1'e toplanmak zorunda olsaydi
            cikti hep ORTALAMA olurdu, TOPLAM tasinamazdi.
  cikti     agirlikli TOPLAM -> en yakin E[token]
  ozyineleme  uretilen token yola EKLENIR, dongu yeni yolla tekrarlanir
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# BELIRLENIMCI KOSU.  Cok iplikli toplama sirasi her kosuda degisiyordu;
# ondalik farklar buyuyup sonucu ziplatiyordu.  GPU'da bazi cekirdekler
# belirlenimci degil; orada zorlanmiyor.
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
#   pay     softmax 0,289   relu 0,782
#   lr      0,01:0,025  0,02:0,810  0,04:0,930  0,08:0,146
#           sabit 0,897   cosine->lr/10 0,080   (cosine ZARARLI)
#   lr OLCEGE BAGLI:  2.601 -> 0,04 ;  251.001 -> 0,004
#                     veri 96,5 kat, lr 10 kat kucuk.  sqrt(96,5)=9,82.
#   wd      251.001'de:  0,03 OLU   0,01 calisiyor   0,001 ezbere kayiyor

# --- OLCULDU, GOREV B  (rakam tokenli, 4000 adim, olcut SAYI)
#   durum   8:0,0148   16:0,0322   32:0,0342      dirsek 16
#     NOT: "durum" ESKI tasarimin parametresiydi (M, b ile ozyineleme).
#     Bu surumde ozyineleme YOK -- olcum yalniz tarihsel kayit.
BOYUT = 16           # token kac sayiyla tarif ediliyor.  YOL bunlardan olusur.
ENUZUN = 24          # kac yuvaya kadar yer kodu tutulur.  3+3->4 hanede
                     #   yol en fazla 8+4 = 12 yuva; 24 rahat pay birakir.
PAY = False          # False -> relu   True -> softmax
LR = 0.002           # SABIT.  Cosine olculdu ve zararli.
WD = 0.01

# YANLILIK (hb) ANAHTAR DEGIL, HER ZAMAN VAR.  Ispat, kosu gerekmez:
#   ag_j = relu(q.k_j) ise bir yuva ancak q.k_j > 0 iken agirlik alir.
#   Bu sinir ORIJINDEN GECEN bir duzlem -- secim homojen yarim uzaya
#   hapsolur.  Anahtar bulutunun orijini icermesi icin sebep yok; bulut
#   tek taraftaysa secim "HEPSI" ya da "HICBIRI"ne duser, arasi yok.
#   Gozlenen (hb eklenmeden ONCE): adim 0'da toplam agirlik 4,66;
#   adim 2'de 10 yuvanin 9'u sifir, toplam 1,11.  "Hicbiri" hali.
#   hb ile sinir q.x = -hb olur, genel yarim uzay.  Maliyet: 1 sayi.
#
# CIKARILDI (22 Eylul): M ve b ile OZYINELEME.
#   Tasarim en bastan "yol = ugranan konumlarin listesi" idi; kodda
#   yol yerine M[token]/b[token] ile bir durum yurutuluyordu ve E
#   girdiye HIC girmiyordu.  Olculdu: ayni tokenin b'si ile E'si
#   arasinda cos = 0,174 -- rastgele 16 boyutta beklenen 0,199.
#   Yani giren temsil ile cikan temsil birbirinden habersizdi.
#
# OZET (A) NEDEN VAR -- OLCULDU, 22 Eylul, 4000 adim, ayni veri:
#   yalniz E+P, ozet YOK :  yuva 1,0000  0,1810  0,1024  0,0987
#   eski M/b ozyineleme  :  yuva 1,0000  0,9416  0,4250  0,1207  (8000 adim)
#   Yani "her yuva kendi ONEGINI tasisin" kismi belirleyici.  Sadece
#   yer kodu vermek yetmiyor: sira gorunur oluyor ama tek dikkat turu
#   onu kullanamiyor.  A bu ozeti PAYLASIMLI tek matrisle geri getiriyor.
#
# ACIK SORUN (22 Eylul, olculdu -- denetim_15):
#   Yol yalniz KONUM LISTESI oldugundan dikkat onu bir TORBA gibi okuyor.
#     53+65 / 65+53   fark 0,000002   cevap AYNI olmali    -> DOGRU
#     53+65 / 35+65   fark 0,000000   cevap FARKLI olmali  -> HATA
#   Operand sirasinin onemsiz olmasi toplamada DOGRU.  Ama bir sayinin
#   ICINDEKI rakam sirasi bilgi tasiyor ve o kayboluyor: iki girdi ayni
#   token TORBASINI veriyor, hicbir parametre ayari ayiramaz.
#   Ayrica q = E[son token] @ Wq -- soru yolun TAMAMINA degil TEK tokene
#   bakiyor; ayni tokenle biten iki farkli yol ayni soruyu soruyor.


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, tohum=0, pay=PAY, enuzun=ENUZUN):
        """Ayarlar dosyanin basinda -- ayri bir ayar dosyasi YOK."""
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.boyut, self.pay = n, boyut, pay

        # BASLANGIC OLCEGI  o = 1/sqrt(boyut).  Turetilisi:
        #   W'nin girisleri bagimsiz, ortalama 0, standart sapma o ise
        #   (x @ W)_j = toplam_i x_i W_ij  ->  varyansi  |x|^2 * o^2
        #   yani  |x @ W| ~ |x| * o * sqrt(boyut).
        #   Cikti boyu girdi boyuyla AYNI kalsin istiyoruz:
        #     o * sqrt(boyut) = 1   ->   o = 1/sqrt(boyut) = 0,25   (boyut 16)
        #   Boylece |q| ~ |E| ve |k| ~ |E| olur; puan = q.k ne patlar ne soner.
        #   ONCEKI DEGER 0,4 idi ve GEREKCESI YOKTU -- elle yazilmisti.
        o = boyut ** -0.5
        self.E = nn.Parameter(r(n, boyut))              # token -> KONUM
        self.P = nn.Parameter(r(enuzun, boyut) * o)     # yuva -> YER
        self.A = nn.Parameter(torch.eye(boyut) + 0.1 * r(boyut, boyut))
        #   A: ozet matrisi, PAYLASIMLI (token basina degil).  Birim
        #   matrise yakin basliyor -> baslangicta ozet ~ birikimli toplam.
        #   P, E ile AYNI uzayda ve ona EKLENIYOR; o yuzden E'yi bastirmasin
        #   diye o=1/sqrt(boyut) ile olcekli basliyor (|P| ~ 1, |E| ~ 4).
        self.Wq = nn.Parameter(r(boyut, boyut) * o)     # son yuva -> soru
        self.Wk = nn.Parameter(r(boyut, boyut) * o)     # yuva -> anahtar
        self.Wv = nn.Parameter(r(boyut, boyut) * o)     # yuva -> deger
        self.hb = nn.Parameter(torch.zeros(1))          # YANLILIK -- hep var

    def yol(self, w):
        """w: (T,) ya da (B,T)  ->  ugranan konum + YER.  (B,T,boyut)"""
        T = w.shape[-1]
        assert T <= self.P.shape[0], f"yol {T} yuva, ENUZUN {self.P.shape[0]}"
        return self.E[w] + self.P[:T]

    def ozet(self, Y):
        """Y: (B,T,boyut) -> (B,T,boyut).  s_j = normalize(A s_{j-1} + yuva_j).
        Her yuva kendi ONEGINI tasir; dikkat isterse hami isterse ozeti okur."""
        B, T, C = Y.shape
        s = Y.new_zeros(B, C)
        iz = []
        for j in range(T):
            s = F.normalize(s @ self.A.T + Y[:, j], dim=-1)
            iz.append(s)
        return torch.stack(iz, 1)

    def dikkat(self, w):
        """Son yuva sorar, butun yuvalar cevaplar."""
        tek = w.dim() == 1
        Y = self.ozet(self.yol(w[None] if tek else w))  # (B,T,boyut)
        q = Y[:, -1] @ self.Wq
        pu = (Y @ self.Wk) @ q.unsqueeze(-1)
        pu = pu.squeeze(-1) + self.hb
        ag = pu.softmax(-1) if self.pay else pu.clamp(min=0)
        o = (ag.unsqueeze(-1) * (Y @ self.Wv)).sum(1)
        return (o[0], ag[0]) if tek else (o, ag)

    def oku(self, o):
        """Sozluk uzayindaki noktaya en yakin token."""
        return int(torch.cdist(o.reshape(1, -1), self.E).argmin())

    def uret(self, w, adim):
        """Cikti YOLA eklenir ve dongu yeni yolla tekrarlanir."""
        w, cikan = list(w), []
        for _ in range(adim):
            c = self.oku(self.dikkat(torch.tensor(w))[0])
            cikan.append(c)
            w.append(c)
        return cikan, w
