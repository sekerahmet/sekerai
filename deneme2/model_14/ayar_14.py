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

# --- OLGU HAFIZASI  (DENKLEM §12c) ---------------------------------
HAFIZA = True
#  C ANAHTAR, V DEGER, okuma EKLEMELI ve kapi ELEMAN BAZINDA:
#      g = ReLU(<zp, C> + hb);    z <- norm(z + g @ V)
#  TASARIM 4 (§12c).  Once softmax(top-n) idi ve DUSTU: yuvalar
#  yarisiyordu, 8.192'nin 10'u kaliyordu (§5.1/U).  Transformer'in
#  FFN'i de bir anahtar-deger hafizasi ve bizim yapimizin aynisi;
#  tek fark sigma'nin ELEMAN BAZINDA olmasi.  Kopya oradan.
#  ZINCIR, hepsi olculdu:
#      cevap hicbir blokta YOK, sansta                    §5.1/R
#      arama R_r'nin ICINDE OLAMAZ: Pi R_r rank<=d,
#        376 serbestlik vs 308x15 = 4.620 kisit  12,3 kat  §5.1/S
#        ustune R_r IZOMETRI -- keyfi tabloyu yapamaz
#      ozne okunabilirligi tam R[iliski]'de cokuyor        §5.1/S
#      fiyat mesele DEGIL: uye'nin %49'u varlik
#        konumlarinda ALINMAMIS duruyor                    §5.1/O
#  !! HAFIZA_N ve HAFIZA_TAU KALDIRILDI -- top-n ve softmax yok.
#     Kapasite sarti da degisti: 'olgu basina bir yuva' softmax
#     top-1 varsayimindan geliyordu.  ReLU'da cikti bir ALT KUMENIN
#     toplami, baglayici kisit yine PARAMETRE sayimi: M >= 1.730.
#     M = 8192 ARTIK GEREKMIYOR ama ilk kosuda degismiyor (bir kosu
#     bir karar, ve fazla kapasite guvenli taraf).

SAAT = False
#  HESAP (§6): tekrar ayrimini D>d boslugu ve farkli capalar zaten
#  yapiyor. Acmanin bedeli: ayni olgu bildirimde ve soruda FARKLI adim
#  sayisinda gelir, kisitlar ~3 KAT olur. Kapasite zaten sinirda.

# --- ANAHTAR DEFTERI  (eski adiyla KOD DEFTERI) -----------------------
K_KOD = 8192
#  §12c: capa kalkinca C artik "capa hedefi" degil, HAFIZANIN ANAHTARI.
#  OLCULDU (§5.1/T): yuva tavani M ile neredeyse DOGRUSAL --
#      M=2048  0,282     M=4096  0,556     M=7381  1,000
#  SIKISTIRMA YOK: cevap adresin keyfi fonksiyonu, olgu basina bir
#  yuva gerekiyor. 7.381 olgu -> M >= 7.381.  8192 secildi.
#  BEDELI: model 285.760 -> 744.512 (x2,61). Kucuk bir ekleme DEGIL.
#  Onceki gerekce (arama maliyeti K ile buyuyor: K=2048 40 ms,
#  K=4096 75 ms) duruyor -- BATCH bu yuzden dustu, asagi bak.

# --- ESIKLER: kayipta YOK, TUTULAN bolmede aranir ---------------------
R_CAPA = 0.0
#  0 = CAPA KAPALI (§12c).  `Yol.esik(0)` 2,0 doner, `s > esik` hic
#  tutmaz.  OLCULDU (§5.1/T): capa adresin TEK bozucusu --
#      capa ACIK   adreslerin %41,8'i cakisik (cos>0,999), tavan 0,588
#      capa KAPALI               %0,0                      tavan 1,000
#  Donmeler izometri oldugu icin adresi HIC bozmuyor; `d` de onemsiz
#  (capa kapaliyken d=8 ve d=16 AYNI: 1,000).
#  BEDELI: §3'un "durum bir koda oturur -> bilesim iner" iddiasi
#  mimaride KALMIYOR. Zaten olculmustu ki inmiyor (§3.1b). Ve §3.2
#  (reddetme = kod uyeligi) dayanaksiz kalir -- hic uygulanmamisti.
DELTA = 0.4
#  Itme esigi (L_dis menteşesi).  OLCULMEDI (A3).

# --- KAYIP AGIRLIKLARI: HICBIRI OLCULMEDI -----------------------------
A1_DIS = 1.0      # itme.  Gerekcesi var, degeri yok: saf cekme kaybi
#                   model_13'te coktu (bit 8,32 > unigram 6,58).
A2_CAPA = 0.0     # VQ KAPALI (§12c).  `kod` terimi C'yi k-ortalamaya
#                   zorluyordu ve OLCULDU (§3.1b) ki o zorlama C'ye
#                   OLGUYU degil ILISKIYI kodlatiyor (+1,680 bit vs
#                   +0,318). Hafizanin adresini bozan sey tam buydu:
#                   C anahtar olacaksa hicbir sey onu niceleyiciye
#                   itmemeli.  a2 = 0 iken terim HIC HESAPLANMAZ.
A3_DUZEN = 1e-4   # ezber <-> genelleme dugmesi
BETA = 0.25       # VQ baglilik agirligi -- a2 = 0 iken ETKISIZ
A4_HAF = 2.0      # HAFIZA BUTCESI agirligi (§12c).  HESAPLANDI:
#                   ikamenin YENI kayiptaki degeri 0,6387 - 0,2469
#                   (kod) - 0,0024 (bag) = 0,3894.  Mentese ile ikame
#                   a4 x (1-B)^2 = a4 x 0,49 odiyor -> a4 > 0,795.
#                   Ust sinir YOK: butcenin altinda maliyet 0.
#                   2,0 = tabanin 2,5 kati. Yukari hata "hafiza hic
#                   kullanilmaz", asagi hata IKAME -- ikincisini bir
#                   kez gorduk (§12b).
#                   !! DUZ L1 SINANDI, ARALIGI BOS: engellemek icin
#                   a4 > 0,639, kullanimi birakmak icin a4 < 0,410.
HAF_BUTCE = 0.30  # ort |m| bu esigin ALTINDA bedava.
#                   OLCULEN varlik konumu payindan turedi: %26,43
#                   (§5.1/R).  Amaclanan kullanim varlik konumlarinda
#                   atesler, yani ~0,26; ikame ~1,0.  SECILMEDI.
#                   !! Ortalama BUTUN konumlardan. Once `[:, isin:]`
#                   idi ve pencerenin %17'sine yazmak BEDAVAYDI;
#                   izde |m| j=1'de 4,79 cikiyordu (§5.1/U).
HAF_B0 = -0.29    # HAFIZA KAPISININ sapma baslangici (Tasarim 4).
#                   Kapi artik ReLU(<z,K> + hb), softmax DEGIL.
#                   OLCULDU (§5.1/U): softmaxta yuvalar YARISIYOR ve
#                   bir yuva gradyan almak icin 8.184 rakibi yenmeli;
#                   kazanan daha cok kazaniyor -- kendini besleyen
#                   dongu, 8.192 yuvanin 10'u kaliyor.  ReLU'da i'nin
#                   ateslemesi j'yi BASTIRMIYOR, dongu YOK.
#                   HESAP: <z,k> birim vektorlerde std ~ 1/sqrt(D) =
#                   0,177.  %p atesleme icin hb0 = -z_p / sqrt(D):
#                      %20 -> -0,149    %5 -> -0,291    %1 -> -0,411
#                   -0,29 = ~%5 (transformer FFN'lerinin tipik
#                   seyrekligi).  `hb` OGRENILIR -- sabitlenirse
#                   seyreklik bir VARSAYIM olarak kalirdi.
#                   !! Denge seyrekligi kagitta ONGORULEMEZ, olculecek.

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
BATCH = 2048
#  HESAP, varsayilan degil: hafiza aramasinin skor matrisi (B*L, M).
#  Onceki kosu B=8192, M=2048 ->  8192*24*2048*4 = 1,61 GB.
#  M 8192'ye ciktigi icin AYNI bellek butcesi B=2048 demek (1,61 GB).
#  Adim/epok 486 -> 1.943; adim basina is 4 kat kucuk, toplam benzer.
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
assert 0 <= R_CAPA and 0 < DELTA and 0 < BETA
assert R_CAPA > 0 or A2_CAPA == 0, (
    "capa KAPALI ama VQ kaybi ACIK: `bag` hic tetiklenmez, `kod` ise "
    "C'yi niceleyiciye zorlar -- §12c tam bunu kaldiriyor")
assert not HAFIZA or K_KOD >= 7381, (
    "hafiza ACIK ama yuva sayisi olgu sayisindan az -- §5.1/T: "
    "sikistirma YOK, tavan M ile dogrusal")
assert A4_HAF == 0 or A4_HAF > 0.795, (
    "butce agirligi IKAME tabaninin altinda -- §12c hesabi")
assert 0 < HAF_BUTCE < 1
assert -1.0 < HAF_B0 <= 0.0, (
    "kapi sapmasi: pozitif olursa konumlarin YARISINDAN fazlasi "
    "atesler, cok negatif olursa hicbiri")
assert PENCERE >= 2 and 1 <= ATLA < PENCERE
assert 1 <= ISINMA < PENCERE
assert ISINMA % ATLA == 0, (
    "ISINMA ATLA'nin kati OLMALI: degilse kalinti siniflari esitlenmez "
    "ve bir gecis sinifi otekilerden bir konum az puanlanir (§5.2)")
