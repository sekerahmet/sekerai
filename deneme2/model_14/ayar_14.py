# -*- coding: utf-8 -*-
"""ayar_14 -- bu kolun BUTUN dugmeleri, her birinin gerekcesiyle.

CLAUDE.md: bir ayar VARSAYILAN OLDUGU ICIN secilmez. Buradaki her
sayinin yaninda ya bir HESAP, ya bir OLCUM, ya da "OLCULMEDI" yazar.
Ucu de yoksa sayi buraya girmez.

IKI BOLUM, iki ayri sozlesme:

  VERI    model_13/model_11 ile BIREBIR ayni olmak ZORUNDA. Sinav
          sabit kalsin diye: bir deger kayarsa `one 0,9353` ile
          `one 0,0204` artik ayni tabloda okunamaz.
          Kapisi: graf 3cd9a2575e47, olcme 44e6262e37f3.

  MODEL   Bu kolun kendisi. Hicbiri model_13'ten gelmiyor; hepsi
          `deneme2/DENKLEM.md`den.
"""
from __future__ import annotations

from taban_14 import Ayar

# =====================================================================
# VERI -- DEGISMEZ.  Degerler model_13 ile birebir.
# =====================================================================
AYAR = Ayar(
    ad="model_14",
    veri_ad="veri_14",
    veri_tohum=0,
    tohum=0,
    jeton_ad="tam",      # varlik = jeton dizisi, ad kac kelimeyse o kadar
    ek_kip="tr2",        # iliski kelimenin kendisi, gercek ek allomorflari

    t_len=512,           # paketleme penceresi (KARAKTER).
    #                      !! model_14 karakter gormuyor ama bu deger
    #                      cumlelerin hangi dilime dustugunu, yani
    #                      akista kimin kimin yanina geldigini
    #                      belirliyor -- kayan pencere onu goruyor.
    kopya=40,            # her varlik icin 40 belge
    tetik=16,            # 40 belgenin 16'si biyografi onegi (oran 2/5)
    zincir_pay=0.20,     # sayfa cumlelerinin ~%20'si zincir
    n3=10000,            # uc adimli zincir orneklemi
    ret_pay=0.05,        # soru cumlelerinin ~%5'i reddetme
    ret_tut=0.20,        # reddetmenin %20'si SINAVA ayrilir --
    #                      gormedigini reddetmek GENELLEMEdir
    tam_kayip=True,

    ent_pay=0.20,
    comp_pay=0.10,
    arama_pay=0.25,      # ENT'in bu kismi ARAMA'ya; HUKUMDEN AYRIK
    ood_pay=0.05,
    kati_pay=0.0,
    belge_pay=0.0,
    n_olcum_max=3000,
)

# Kayarsa sinav ayrisir. `test_14` bunlari izlerle birlikte denetler.
SABIT = ("veri_tohum", "tohum", "jeton_ad", "ek_kip", "t_len", "kopya",
         "tetik", "zincir_pay", "n3", "ret_pay", "ret_tut", "tam_kayip",
         "ent_pay", "comp_pay", "arama_pay", "ood_pay", "kati_pay",
         "belge_pay", "n_olcum_max")
IZ_GRAF = "3cd9a2575e47"
IZ_OLCME = "44e6262e37f3"

# =====================================================================
# MODEL -- DENKLEM.md.  Hicbiri model_13'ten gelmiyor.
# =====================================================================

# --- GEOMETRI ---------------------------------------------------------
D_DURUM = 32
#  HESAP (DENKLEM §4.1): D = d olsaydi ortak sonek bir izometri olur,
#  butun ikili mesafeleri korur ve "her iliskide AYNI cevap-geometrisi"
#  dayatirdi. Cemberde sayildi: gereken hata <0,38 derece, en iyi
#  uzlasmada cikan ~100 derece -- 250 kat.
#  OLCULMEDI: 16/32/64 arasinda hangisi (A1).

D_OKUMA = 16
#  ONCE 8 IDI.  8 -> 16, gerekce OLCULDU (21 Eylul, t0, kapali form,
#  egitim YOK): sinav oneginin SONUNDAKI durumdan OZNENIN KIM OLDUGU
#  dogrusal probla okundu. 480 ozne, sans 240,5, %70/%30 ayri dilim,
#  ve okuma yalnizca durumun ILK d koordinatini goruyor:
#      d      ort sira   1.sira    kazanc/boyut
#      8       50,18      %3,2       -5,4
#      12      23,08     %10,6       -6,8   <- boyut basina en buyuk
#      16      15,37     %16,9       -1,9
#      24       8,80     %26,0       -0,55
#      32       6,09     %35,1       -0,24
#  KIMLIK DURUMDA VAR (32 boyutta 6,09/480, %35,1 tam isabet) ama
#  OKUMA GOREMIYOR (8 boyutta 50,18, %3,2). Sizintinin bedeli sirada
#  8,2 kat. d = 16 kazancin cogunu aliyor, sonrasi sonuyor.
#  MALIYET SIFIR: `p` bir buffer, `Pi` bir dilim -- parametre artmaz.
#  §4.1'in "D > d" sarti korunuyor (32 > 16).
#  !! Egri d=8 ile EGITILMIS modelden okundu, yani ALT SINIR.
#
#  ELENEN ALTERNATIF: kayba |Pz| terimi. Terim ancak "|Pz| buyuk olan
#  ornekler daha iyi okunuyor" ise ise yarar; korelasyon olculdu,
#  |Pz|~kimlik -0,068, |Pz|~cevap -0,014. SIFIR. Sorun okunabilir
#  kisimdaki KUTLE degil, HANGI YONLERIN orada oldugu.
#
#  KALABALIK NOTR: d buyudukce noktalar UZAKLASIYOR (komsu acisi
#  24,7 -> 41,8 derece), ama baslangic donme olcegi zaten
#  `aralik = n^(-1/(d-1))`, yani komsu araliginin kendisi -- oran
#  korunuyor.  (Eski not "475 nokta S^7'de 22,5 derece" bu yuzden
#  d'ye karsi bir argüman degildi.)

# --- OPERATOR ---------------------------------------------------------
K_TAM = None      # None = HEPSI tam SO(D)
#  Kac birim TAM SO(D) alacak; kalani TEK DUZLEM.  None ise AYRIM YOK.
#  OLCULDU (test_14 §3): tek duzlemli donme, kendi duzleminin disinda
#  ozdesliktir ve rastgele bir farkin ancak 2/D'sine dokunur --
#  ongorulen 0,0625, olculen 0,0629.
#
#  ONCE 80 IDI ve "!! 80 DOGRULANMADI" diye yaziliydi. OLCULDU
#  (20 Eylul, t0): 451 birimin 328'i VARLIK birimi (ad parcasi) ve
#  bunlarin yalniz 24'u (%7,3) frekansla ilk 80'e giriyordu. Yani
#  OLGUYU TASIMASI GEREKEN 304 birim, farkin %6,2'sine dokunan
#  operatore mahkumdu. Modelin yazdiginda birebir gorundu:
#     Mersin -> Mersin(0,98)   Bartin -> Bartin(0,94)
#     Isikli -> Isikli(0,91)   Sanliurfa x4
#  yani R[ad] ~ I, durum kimildamiyor ve ozne degisince cevap
#  degismiyor. DOGRULUK BILGI tam = 0,0000 idi.
#
#  §4.3'un "cozum dilsel" ayrimi (kapali sinif tam, acik sinif tek
#  duzlem) FREKANSA dayaniyordu; olcum frekans ayriminin olguyu
#  yanlis tarafa koydugunu gosterdi.
#
#  HESAP: 444 x 496 + 2048 x 32 = 285.760 parametre (onceki 129.331).
#  `donme()` artik 444 matrix_exp: 1,5 ms -> ~5 ms, epok ~27 sn.
#
#  DEGER 451 IDI, `None` YAPILDI (21 Eylul).  451 sozluk boyunu
#  (444) ASIYORDU, yani "hepsi" demek istiyor ama bunu TESADUFEN
#  soyluyordu: sozluk 451'i gecse frekans ayrimi KENDILIGINDEN ve
#  KEYFI bir kesimle geri gelirdi, hicbir kapi da soylemezdi
#  (kapi 31 SINIFI siniyordu, degerin kendisini degil).
#  `None` niyeti dogrudan yaziyor: AYRIM YOK.  Davranis degismedi.
#  Kapi 31 artik "int ise sozlukten KUCUK olmali" diye yasakliyor.
#
#  !! AYRIM OLCEK MESELESI, KALITE DEGIL. §4.3 V=50k D=256'da
#  1,63 MILYAR parametreyi gerekce gosteriyor; bu olcekte (444 birim,
#  D=32) tamami 220 bin, yani ayrima GEREK YOK. Olcek buyuyunce
#  yeniden acilir -- ama o zaman kesim FREKANSA degil ROLE gore
#  kurulur (kapi 31'in dersi).

SAAT = False
#  HESAP (§6): tekrar ayrimini D>d boslugu ve farkli capalar zaten
#  yapiyor. Acmanin bedeli: ayni olgu bildirimde ve soruda FARKLI adim
#  sayisinda gelir, kisitlar ~3 KAT olur. Kapasite zaten sinirda.

# --- KOD DEFTERI ------------------------------------------------------
K_KOD = 2048
#  Capa hedefleri. Bu bir VARLIK SAYISI DEGIL -- model neyin kod
#  olacagina kendi karar veriyor; kodlarin varliklarla ortusmesi
#  SINANACAK sey (S1), varsayim degil.
#  OLCULDU: baskin maliyet dugmesi. Is yukunun %92'si kod aramada,
#  ve arama B*K*D ile buyuyor (B=1024, ileri+geri):
#     K= 512  26 ms/adim      K=2048  40 ms
#     K=1024  31 ms           K=4096  75 ms

# --- ESIKLER: kayipta YOK, TUTULAN bolmede aranir ---------------------
R_CAPA = 0.25
#  Capa yaricapi. Kodlar arasi mesafenin YARISINDAN kucuk olmali,
#  yoksa durum iki koda birden yakin dusar.  OLCULMEDI (A2).
DELTA = 0.4
#  Itme esigi (L_dis menteşesi).  OLCULMEDI (A3).

# --- KAYIP AGIRLIKLARI: HICBIRI OLCULMEDI -----------------------------
A1_DIS = 1.0      # itme.  Gerekcesi var, degeri yok: saf cekme kaybi
#                   model_13'te coktu (bit 8,32 > unigram 6,58).
A2_CAPA = 1.0     # VQ
A3_DUZEN = 1e-4   # ezber <-> genelleme dugmesi
BETA = 0.25       # VQ baglilik agirligi -- VQ-VAE'nin standart degeri

# --- EGITIM -----------------------------------------------------------
PENCERE = 24
#  Zincirin uzunlugu -- modelin OGRENEBILECEGINI belirler. Semantik
#  bolme YOK: pencere keyfi yerden baslar ve bastaki cop durum
#  SILINMIYOR, puanlaniyor (DENKLEM.md §5.1/G, acik A3).
#  OLCULDU (20 Eylul): sinavin sordugu sey SORU + CEVAP, ve ikisinin
#  ayni zincirde olmasi gerekiyor -- cevap sorunun OZNESINDEN
#  uretiliyor. Birim akisinda 648.281 soru-cevap cifti, ortalama 14,7
#  birim:  L=16 -> %77,2   L=20 -> %97,3   L=24 -> %99,9.
#  Ilk deger 16 idi ve OLCULMEDEN yazilmisti; ciftlerin %22,8'inde
#  zincir cevaba varmadan kesiliyordu.
ISINMA = 4
#  Kayip ilk ISINMA konumu PUANLAMAZ -- pencere keyfi yerden basliyor,
#  bastaki onek cop.  OLCULDU: ayni onege dayatilan AYRI hedef sayisi
#  ve bunun belirlenimci tavani
#      konum 1: 9,20 hedef %45,1    konum 2: 3,34 %52,9
#      konum 3: 1,91  %63,2         konum 6: 1,10 %90,8
#  HESAP: bir gecis j = i mod ATLA kalinti sinifinda kalir, yani
#  L=24/ATLA=4'te 5-6 ayri konumda puanlanir ve EN COK BIRI 1..3'te.
#      W:  1 -> 5 6 6 6 (23)   4 -> 5 5 5 5 (20)   5 -> 4 5 5 5 (19)
#  W=4 TEK deger: tavani olculen uc dusuk konumu atar VE dort sinifi
#  da esitler; hicbir gecis egitimden dusmez.  Alternatif (capayi
#  adim 1'de tetiklemek) ELENDI: r 0,25 -> 0,95 gerekirdi, o yaricapta
#  2048 baslik kureyi order-1 kapliyor = sonlu otomat.  (§5.2, kapi 32)
ATLA = 4
#  Pencerelerin kesme araligi. Modelin ogrenebilecegini DEGISTIRMEZ,
#  yalniz ayni gecisin epok icinde kac kez gradyan verdigini belirler:
#  atla=1'de her gecis L-1 = 23 pencerede, atla=4'te ~6 pencerede.
#  HESAP: is ~ N(L-1)/atla. atla 1 -> 4 is 4 kata duser, ve her gecis
#  yine 6 FARKLI ofsetten gorulur (atla=L-1 olsaydi hep tek ofset
#  olurdu ve pencere basi cop durumu sistematiklesirdi).
#  Kac katkinin YETTIGI OLCULMEDI.
LR = 3e-3
BATCH = 8192
EPOK = 5
#  OLCULDU (20 Eylul, L4): ilk kosu epok 1'i 220 sn'de bitiremedi --
#  yani >117 ms/adim. Kagit uzerindeki "saniyeler" tahmini YANLISTI:
#  darbogaz FLOP degil bellek trafigi (bkz. `model_14.yol` yorumlari).
#  Gercek epok suresi P PROFIL hucresinde olculuyor.

# =====================================================================
# KAPILAR -- yalniz BU KOLUN dogruladigi seyler
# =====================================================================
assert AYAR.veri_ad == "veri_14", "kol KENDI veri modulunu okur"
assert AYAR.ek_kip == "tr2" and AYAR.jeton_ad == "tam", (
    "korpus yuzeyi degismemeli -- sinav izi buna bagli")
assert AYAR.tetik * 5 == AYAR.kopya * 2, "tetik/kopya ORANI 2/5"
assert AYAR.zincir_pay == 0.20 and AYAR.n3 == 10000, (
    "ZINCIR AZINLIK -- CLAUDE.md kural 5")
assert 0 < AYAR.ret_tut < 0.5, (
    "reddetmenin bir kismi SINAVA ayrilir: gormedigini reddetmek GENELLEME")
assert all(hasattr(AYAR, a) for a in SABIT)

assert D_OKUMA < D_DURUM, "D > d ZORUNLU -- DENKLEM §4.1 izometri celiskisi"
assert K_TAM is None or K_TAM >= 1, (
    "hicbiri TAM degilse §4.3'e gore mimari zayif kalir")
assert 0 < R_CAPA and 0 < DELTA and 0 < BETA
assert PENCERE >= 2 and 1 <= ATLA < PENCERE
assert 1 <= ISINMA < PENCERE
assert ISINMA % ATLA == 0, (
    "ISINMA ATLA'nin kati OLMALI: degilse kalinti siniflari esitlenmez "
    "ve bir gecis sinifi otekilerden bir konum az puanlanir (§5.2)")
