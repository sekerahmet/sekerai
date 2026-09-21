"""model_15 -- YOL, UGRANAN KONUMLARIN LISTESIDIR.

Kullanici, 21 Eylul:  "yol su degil mi? istanbul(x,y) U ankara U mersin.
bu sayede hangi konumdan hangi konuma gittigimi bilirim."
ve 22 Eylul: "E1 + 6 = E2 oluyor.  ana tasarimin kalbi bu."

  sozluk    E[token] -- her token'in KONUMU.  Girdi de bu, okumanin hedefi de.
  gecis     M[token] -- o konuma UGRAMAK ozeti nasil dondurur.
  ozet      s_j = normalize( M[w_j] @ s_{j-1} + E[w_j] )
            Yola yeni bir konum eklenince ozet o konumun matrisiyle
            donuyor ve uzerine konumun KENDISI biniyor.
  soru      son yuva sorar:  q = s_son @ Wq
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

# --- OLCULDU, GOREV B  (rakam tokenli, olcut SAYI, ayni veri)
#   durum   8:0,0148   16:0,0322   32:0,0342      dirsek 16
#
#   YUVA BASINA DOGRULUK -- uc tasarim yan yana:
#     M/b ozyineleme   1,0000  0,9416  0,4250  0,1207   8000 adim  4.529 par
#     E+P, ozet YOK    1,0000  0,1810  0,1024  0,0987   4000 adim  1.361 par
#     E+P+A ozet       1,0000  0,8968  0,1107  0,1002   4000 adim  1.617 par
#
#   OKUNAN:  OZYINELEME belirleyici.  Yalniz yer kodu (P) verince sira
#   GORUNUR oluyor ama tek dikkat turu onu kullanamiyor (yuva2 0,18).
#   Paylasimli tek matris (A) yuva2'yi geri getiriyor (0,90) ama yuva3'u
#   getirmiyor (0,11 -- 10 rakam icin sans 0,10, ve adim 0'dan 4000'e
#   hic trend yok).  TOKEN BASINA gecis matrisi olan eski surum yuva3'te
#   0,4250 yapiyordu.  Bu yuzden ozyinelemeye DONULDU.

BOYUT = 16           # token kac sayiyla tarif ediliyor.  YOL bunlardan olusur.
                     #   Ozet de bu uzayda: E dogrudan s'ye ekleniyor, yani
                     #   "durum boyutu" AYRI bir ayar DEGIL, boyut'a esit.
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
# CIKARILDI (22 Eylul): b[token] -- girisin kendi ayri parametresi.
#   Eski surumde token iceri b[w] diye giriyor, disari E[w] diye
#   okunuyordu ve bu ikisi ALAKASIZ iki parametre blogu idi.
#   Olculdu:  cos( b[token], E[token] ) = 0,174
#             rastgele 16 boyutta beklenen = 0,199
#   Yani modelin bir tokeni koydugu yer ile onu okudugu yer habersizdi.
#   Simdi giren de cikan da E -- cos = 1, tanim geregi.  -208 parametre.
#
# CIKARILDI (22 Eylul): P (yer kodu) ve A (paylasimli ozet matrisi).
#   Ikisi de M'nin YOKLUGUNDA siranin nasil tasinacagi sorusuna verilmis
#   cevaplardi.  M geri geldi ve sira zaten M'de: matris carpimi yer
#   degistirmez, M[5]M[3]s != M[3]M[5]s.  Ustteki tablo ikisinin de
#   M kadarini yapamadigini gosteriyor.
#
# ACIK SORUN (22 Eylul, olculmedi -- E ve b birlesince DOGAN sorun):
#   |E| ~ 4 (olceksiz baslatiliyor, cunku okuma hedefi o) ama |s| = 1
#   (normalize) ve |M @ s| ~ 1.  Yani toplamda yeni token eski ozeti
#   ~4'e 1 bastiriyor.  Eski surumde b AYRI parametreydi ve kendi
#   olcegini bulmakta serbestti; artik degil.  Egitim M'yi buyuterek
#   dengeleyebilir de, dengeleyemeyebilir de -- OLCULMEDI.


class Yol(nn.Module):
    def __init__(self, n, boyut=BOYUT, tohum=0, pay=PAY):
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
        self.M = nn.Parameter(torch.eye(boyut).repeat(n, 1, 1)
                              + 0.1 * r(n, boyut, boyut))
        #   M: token basina gecis matrisi.  Birim matrise yakin basliyor ->
        #   baslangicta ozet ~ ugranilan konumlarin birikimli toplami; egitim
        #   her tokene kendi DONUSUNU ogretiyor.  Sira buradan geliyor.
        self.Wq = nn.Parameter(r(boyut, boyut) * o)     # son yuva -> soru
        self.Wk = nn.Parameter(r(boyut, boyut) * o)     # yuva -> anahtar
        self.Wv = nn.Parameter(r(boyut, boyut) * o)     # yuva -> deger
        self.hb = nn.Parameter(torch.zeros(1))          # YANLILIK -- hep var

    def yol(self, w):
        """w: (T,) ya da (B,T)  ->  ugranan konumlar E[w].  (B,T,boyut)"""
        return self.E[w]

    def ozet(self, w):
        """w: (B,T) -> (B,T,boyut).  s_j = normalize(M[w_j] s_{j-1} + E[w_j]).
        Her yuva hem KENDINI hem ONEKINI tasir."""
        B, T = w.shape
        s = w.new_zeros(B, self.boyut, dtype=self.E.dtype)
        iz = []
        for j in range(T):
            Mj = self.M[w[:, j]]                        # (B,boyut,boyut)
            s = F.normalize((Mj @ s.unsqueeze(-1)).squeeze(-1)
                            + self.E[w[:, j]], dim=-1)
            iz.append(s)
        return torch.stack(iz, 1)

    def dikkat(self, w):
        """Son yuva sorar, butun yuvalar cevaplar."""
        tek = w.dim() == 1
        Y = self.ozet(w[None] if tek else w)            # (B,T,boyut)
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
