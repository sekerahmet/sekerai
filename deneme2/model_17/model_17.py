"""model_17 -- YOL TUTULMAZ, HESAPLANIR.

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
# --- HESAPLANDI, 22 Eylul.  TinyStories icin 16 -> 32.
#   ETKIN OKUMA BOYUTU = min(durum, boyut).  Wv (durum, boyut) ve
#   o = toplam(ag_j * s_j @ Wv); rank <= min(durum, boyut), yani cikti
#   o alt uzaydan CIKAMIYOR.  Olculdu (rastgele dizilerde o'nun ranki):
#     boyut 16 durum 16 -> 16     boyut 32 durum 16 -> 16   (boyut BOSA)
#     boyut 32 durum 32 -> 32     boyut 64 durum 16 -> 16
#   boyut > durum'un tek getirisi E_c'nin dik bileseni: sinif basina
#   SABIT ekler, yani ogrenilebilir yanlilik -- ayirt etmez.
#
#   4.000 kelime etkin boyuta yerlestirilince en yakin komsu ne kadar
#   uzakta (rastgele E, okuma payi = komsu uzakligi / 2|E_c|):
#     16 -> 0,342     32 -> 0,443     64 -> 0,517
#   16'da cikti hedefin %34'u icinde kalmak zorunda.  Kiyas: yayinlanan
#   TinyStories-1M gizli boyut 64, 33M ise 768 kullaniyor.
#   Bedel: durum 16 -> 32, jeton basi 288 -> 1.088 sayi (M durum^2).
#
#   A / B2 / C 16/16 ile OLCULDU; onlarin yedekleri kendi degerlerini
#   tasiyor (Yol(n, boyut=p['boyut'], ...)), etkilenmiyorlar.  Ama
#   yeniden kosulurlarsa 32/32 olur -- ayni ad, baska kol.
BOYUT = 32           # token kac sayiyla tarif ediliyor
DURUM = 32           # s kac sayi
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
    mimari = "kelime_matris"      # TAM1/TAM2; hedef mimari asagida (DT)

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
        """SON yuva sorar, butun yuvalar cevaplar.  w (B,T) -> o (B,boyut).

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

    def dizi(self, w, maske=None):
        """HER yuva kendi sorusunu sorar.  w (B,T) -> puan (B,T,n).

        `dikkat`in cogul hali; mekanizma AYNI, degisen tek sey kimin
        sordugu.  Sonraki-jeton egitimi konum basina okuma ister; her
        onek icin ayri ileri gecis T kat is ederdi.

        NEDENSEL MASKE ZORUNLU: butun yuvalar ayni dizide soruyor, maske
        gelecegi gormemeyi geri koyuyor.  relu yolunda agirlik dogrudan
        0'lanir (pay yok); softmax yolunda -inf ile.

        maske (B,T) bool: gercek jetonlarda True.  HER HIKAYE BIR PENCERE
        oldugu icin hikaye bitince kalan yer <dolgu>; o yuvalar cevap
        VEREMEZ.  Maskelenmezse dolgu yuvalari toplama girer ve her
        ornege BASKA sayida sahte katki biner.
        """
        assert w.dim() == 2, "dizi() (B,T) bekler"
        S = self.gez(w)[:, 1:]                       # (B,T,durum)
        Q, K, V = S @ self.Wq, S @ self.Wk, S @ self.Wv
        P = Q @ K.transpose(1, 2) + self.hb          # (B,T,T)
        gec = torch.ones(w.shape[1], w.shape[1], dtype=torch.bool,
                         device=w.device).tril()     # j yalniz <= j'ye bakar
        if maske is not None:
            gec = gec & maske[:, None, :]            # dolgu CEVAP VEREMEZ
        A = (P.masked_fill(~gec, -torch.inf).softmax(-1) if self.pay
             else P.clamp(min=0) * gec)
        # A @ V carpim olarak yazilmali: yayilimla (B,T,T,boyut) ara tensor
        # cikar ve B=2048, T=128'de tek basina GB'lara gider.
        return self.puan(A @ V)

    def ic(self, w, maske=None):
        """tani() icin ic okuma: [(durum, P, A, izin)] -- tek blok.  Egitimi
        etkilemez; dizi() ile ayni hesap."""
        S = self.gez(w)[:, 1:]
        P = (S @ self.Wq) @ (S @ self.Wk).transpose(1, 2) + self.hb
        gec = torch.ones(w.shape[1], w.shape[1], dtype=torch.bool,
                         device=w.device).tril()
        if maske is not None:
            gec = gec & maske[:, None, :]
        A = (P.masked_fill(~gec, -torch.inf).softmax(-1) if self.pay
             else P.clamp(min=0) * gec)
        return [(S, P, A, gec)]

    def puan(self, O):
        """Cikti noktasindan SOZLUK PUANI.  Buyuk = yakin.

        Tek yerde durmasi sart: dizi/oku/uret ucu de bunu cagirir, yoksa
        egitim bir kuralla, uretim baskasiyla calisir."""
        E = self.E.expand(O.shape[0], -1, -1) if O.dim() == 3 else self.E
        return -torch.cdist(O, E) ** 2

    def kayip(self, w, maske=None):
        """SONRAKI JETON, her konumda.  w (B,T) -> SKALER.

        Konum j, j+1'i tahmin eder; son konumun hedefi yok.

        maske verilirse HEDEFI DOLGU olan konumlar hem paydan hem
        PAYDADAN duser -- yani kayip, dolgusuz haliyle ayni olcekte
        kalir.  Dolgu sayilsaydi gradyanin bir kismi "dolgu tahmin et"
        ogretirdi ve hikaye basina farkli sayida dolgu oldugu icin
        ornekler esit agirlikta olmazdi."""
        p = self.dizi(w, maske)[:, :-1].reshape(-1, self.n)
        h = w[:, 1:].reshape(-1)
        if maske is None:
            return F.cross_entropy(p, h)
        a = maske[:, 1:].reshape(-1)
        k = F.cross_entropy(p, h, reduction="none")
        return (k * a).sum() / a.sum()

    def oku(self, o):
        """Sozluk uzayindaki noktaya en yakin token."""
        return int(self.puan(o.reshape(1, -1)).argmax())

    def uret(self, w, adim):
        """Cikti girdiye eklenir.  Uzunluk SABIT oldugu icin durma kosulu yok."""
        w, cikan = list(w), []
        for _ in range(adim):
            c = self.oku(self.dikkat(torch.tensor(w))[0])
            cikan.append(c)
            w.append(c)
        return cikan, w


# ============================================================
# HEDEF MIMARI -- DT (durum takibi + attention + bellek).  TASARIM.md.
# Kullanici, 23 Eylul: "derinlik olsun, küçük bütçeyle başla, DT1'i yaz".
# ============================================================
DT_GENISLIK = 256    # artik akis
DT_DURUM = 64        # S: 64 x 64 anahtar-deger yuvasi
DT_BLOK = 2          # kopru DERINLIKLE: cozulebilen adim <= blok sayisi
DT_BELLEK = 1024     # bellek yuvasi (MLP gizli boyu)
DT_YANSIMA = 1       # token basina gecis; 3'lu donme icin 2 gerekir


def durum_gecisi(k, v, be, q):
    """S_t = S_{t-1} (I - be k k^T) + be v k^T,  okuma h_t = S_t q_t.

    k (B,T,Y,d) birim, v (B,T,Y,d), be (B,T,Y), q (B,T,d) -> h (B,T,d), HAM.
    Y: token basina gecis sayisi.  S sifirdan baslar, normalize EDILMEZ:
    be (0,2)'de gecisin ozdegerleri [-1,1].  SIRALI; parcali paralel hali
    (Yang 2024) sonraki adim, bu fonksiyon onun kapisi olacak."""
    B, T, Y, d = k.shape
    S = k.new_zeros(B, d, d)
    iz = []
    for t in range(T):
        for i in range(Y):
            kt = k[:, t, i]
            Sk = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            S = S - ((be[:, t, i, None] * (Sk - v[:, t, i])).unsqueeze(-1)
                     * kt.unsqueeze(1))
        iz.append(torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1))
    return torch.stack(iz, 1)


class RMS(nn.Module):
    """Konum basina olcek.  Zamanda bir sey karistirmaz, paralelligi bozmaz."""

    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class Blok(nn.Module):
    """durum -> attention -> bellek, artik akisa ekleyerek."""

    def __init__(self, d, durum, bellek, yansima, pay, r):
        super().__init__()
        o = d ** -0.5                  # birim RMS girdi -> birim RMS cikti
        self.n1, self.n2 = RMS(d), RMS(d)
        self.Gk = nn.Parameter(r(yansima, d, durum) * o)  # yazma anahtari
        self.Gv = nn.Parameter(r(yansima, d, durum) * o)  # yazma degeri
        self.gb = nn.Parameter(torch.zeros(yansima, d))   # beta egimi
        self.gb0 = nn.Parameter(torch.zeros(yansima))     # beta 1'den baslar
        self.Gq = nn.Parameter(r(d, durum) * o)           # okuma sorusu
        a = durum ** -0.5                  # birim boylu h -> birim boylu q, k
        self.Aq = nn.Parameter(r(durum, durum) * a)       # attention sorusu
        self.Ak = nn.Parameter(r(durum, durum) * a)       # attention anahtari
        self.Av = nn.Parameter(r(durum, d) * a)           # attention degeri
        self.hb = nn.Parameter(torch.zeros(1))            # relu esigi
        self.W1 = nn.Parameter(r(d, bellek) * o)          # bellek anahtarlari
        self.b1 = nn.Parameter(torch.zeros(bellek))
        self.W2 = nn.Parameter(r(bellek, d) * bellek ** -0.5)  # bellek degerleri
        self.b2 = nn.Parameter(torch.zeros(d))
        self.pay = pay

    def ic(self, r_, maske=None):
        """(durum okumasi H, P, A, izin) -- forward ve tani() ayni hesabi okur."""
        x = self.n1(r_)
        k = F.normalize(torch.einsum("btd,yde->btye", x, self.Gk), dim=-1)
        v = torch.einsum("btd,yde->btye", x, self.Gv)
        be = 2 * torch.sigmoid(torch.einsum("btd,yd->bty", x, self.gb)
                               + self.gb0)
        q = F.normalize(x @ self.Gq, dim=-1)
        # eps 1e-3: bos yuvadan okunan kucuk gurultu birim boya sisirilmesin.
        H = F.normalize(durum_gecisi(k, v, be, q), dim=-1, eps=1e-3)
        P = (H @ self.Aq) @ (H @ self.Ak).transpose(1, 2) + self.hb
        T = r_.shape[1]
        gec = torch.ones(T, T, dtype=torch.bool, device=r_.device).tril()
        if maske is not None:
            gec = gec & maske[:, None, :]
        A = (P.masked_fill(~gec, -torch.inf).softmax(-1) if self.pay
             else P.clamp(min=0) * gec)
        return H, P, A, gec

    def forward(self, r_, maske=None):
        H, _, A, _ = self.ic(r_, maske)
        r_ = r_ + A @ (H @ self.Av)
        return r_ + F.relu(self.n2(r_) @ self.W1 + self.b1) @ self.W2 + self.b2


class DT(nn.Module):
    """HEDEF MIMARI.  Baglam DURUMDA (her hikayede sifirdan), bilgi BELLEKTE
    (kalici, benzerlikle eslesir).  Kelime basina matris YOK."""
    mimari = "dt"

    def __init__(self, n, genislik=DT_GENISLIK, durum=DT_DURUM, blok=DT_BLOK,
                 bellek=DT_BELLEK, yansima=DT_YANSIMA, tohum=0, pay=PAY):
        super().__init__()
        g = torch.Generator().manual_seed(tohum)
        r = lambda *s: torch.randn(*s, generator=g)
        self.n, self.genislik, self.durum = n, genislik, durum
        self.blok, self.bellek, self.yansima, self.pay = blok, bellek, yansima, pay
        self.X = nn.Parameter(r(n, genislik))                 # token -> akis
        self.bloklar = nn.ModuleList(
            Blok(genislik, durum, bellek, yansima, pay, r) for _ in range(blok))
        self.ns = RMS(genislik)
        # |E| ~ 1: baslangicta puanlar O(1), ilk kayip patlamasin.
        self.E = nn.Parameter(r(n, genislik) * genislik ** -0.5)

    def dizi(self, w, maske=None):
        """w (B,T) -> puan (B,T,n).  Nedensel: durum soldan saga, attention
        yalniz j <= t."""
        assert w.dim() == 2, "dizi() (B,T) bekler"
        r_ = self.X[w]
        for b in self.bloklar:
            r_ = b(r_, maske)
        return self.puan(self.ns(r_))

    def ic(self, w, maske=None):
        """tani() icin blok basina (durum, P, A, izin)."""
        r_, cik = self.X[w], []
        for b in self.bloklar:
            cik.append(b.ic(r_, maske))
            r_ = b(r_, maske)
        return cik

    def puan(self, O):
        """En yakin E: -|o-E|^2 = 2 o.E - |E|^2 - |o|^2, son terim satir sabiti."""
        return 2 * O @ self.E.T - (self.E * self.E).sum(-1)

    kayip = Yol.kayip          # ayni sonraki-jeton kaybi, ayni maske kurali
