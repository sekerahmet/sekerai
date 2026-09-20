# -*- coding: utf-8 -*-
"""ayar_13 — BU KOLUN KENDI ayari. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026:
    *"bunlarin hepsi model_09 folderi altinda olmali. model_09 diger
    hicbir model ile ayni seyi kullanmamali."*

Onceki surum `from model_b15 import AYAR as TABAN` diyordu; yani
model_09'in ayari model_b15 -> model_b14 -> model_b13 -> ... zincirinden
DEVRALINIYORDU ve zincirin herhangi bir halkasi degisince sessizce
kayardi. Artik oyle degil: her alan ASAGIDA, `Ayar()` varsayilaninin
uzerine, TEK TEK ve gerekcesiyle yaziliyor.

Bunun bir yan faydasi var ve ilk kurulumda tam da bu kacmisti: devralma
GIZLIYORDU, acik yazim GOSTERIYOR.

==========================================================================
IKI GRUP, IKI GEREKCE

  GOREV alanlari    -> model_b15'in gordugu sinavla AYNI KALMASI
                       BEKLENEN alanlar. model_03'e kadar hepsi
                       birebirdi. model_09 IKISINI BILEREK degistiriyor
                       ve ikisi de BU KOLUN TANIMI:
                         veri_ad  yeni veri (GOREV_ALAN'da zaten yok)
                         ek_kip   "tr" -> "tr2"
                       `test_11.py` bunlari BILDIRILMIS AYRISMA diye
                       listeler; LISTEDE OLMAYAN bir alan kayarsa test
                       yine duser.

  MIMARI + OPTIMIZASYON -> STANDART TARIF, referanslariyla.

==========================================================================
STANDART TARIFE GORE: NE DEVRALINMADI

1) LOOKAHEAD ORTALAMASI -- `ort_bas` 0 (YANI KAPALI)

   model_b ailesi 10.000. adimdan sonra "yavas agirlik" tutup her 2000
   adimda bir onunla karistiriyor (kosu logunda "ORTALAMA ACILDI adim
   10000  alfa 0.5  her 2000 adim"). Bu bir OPTIMIZER SARMALAYICISI
   (Lookahead, Zhang ve ark. 2019) ve nanoGPT'de, Llama'da, Pythia'da,
   GPT-2/GPT-3 tarifinde YOK.

   `model_09` standart bir transformerin ne yaptigini olcecekse,
   standart olmayan bir optimizasyon numarasiyla kosamaz. KAPALI.
   (`Ayar()` varsayilani zaten 0; model_b15 onu 10000 yapiyordu.)

2) betas -- (0.9, 0.95), (0.9, 0.999) DEGIL

   0.999 PyTorch'un VARSAYILANI, dil modeli tarifi degil:
       nanoGPT 0.95    GPT-3 0.95    Llama 0.95    Pythia 0.95

   beta2 gradyan BUYUKLUGUNUN hafiza suresi; etkin pencere ~1/(1-beta2):
       0.999 -> ~1000 adim      0.95 -> ~20 adim
   20.000 adimlik bir kosuda 1000 adim, egitimin %5'i.

   !! DURUSTCE: bizim veri rejiminde etkisi KUCUK beklenir. beta2'nin
   asil onemli oldugu durum bir jetonun binlerce adimda bir gorunmesi;
   bizde EN SEYREK jeton bile ~2 adimda bir geciyor (olculdu: varlik
   jetonu sikliklari 632.296 .. 256, batch 512'de en seyrek ~0,44
   kez/batch). Gerekce "tarif boyle", olculmus bir kazanc DEGIL.

3) Phi DARBOGAZI -- `dar_alfa` 0.0, `dar_kapi` False

   DiscoLoop projeye ozgu; standart tarifte yok. Kurulmuyor bile.

==========================================================================
ZATEN STANDART OLANLAR (varsayilandan gelenler de dahil)

    wd = 0.1              nanoGPT / Pythia / Qwen2.5 SFT -- CLAUDE.md kural 4
    cosine -> lr/10       nanoGPT / Pythia (min_lr = lr/10) -- kural 4
                          (`sabit_lr=False` bunu aciyor)
    isinma = 2000         nanoGPT warmup_iters=2000, Llama warmup=2000
                          MUTLAK deger olarak BIREBIR ayni. (Oran farkli
                          cikiyor cunku onlarin kosusu 600k adim, bizimki
                          20k -- ama tarifin yazdigi sayi 2000.)
    lr = 1e-3             Pythia-70m 1e-3. Bizim model 6,5M; bu mertebede
                          dogru yon KUCUK model -> BUYUK lr.
    grad clip 1.0         hepsi (`egit` yapiyor)
    tam_kayip = True      butun pozisyonlarda next-token: standart LM kaybi
    d 256 / nh 4          head_dim 64 -- bu olcekte GQA/MQA anlamsiz
    batch 512             olcum hattiyla paylasilir
    adim 20000            CLAUDE.md kural 1 (ILK SINIR)
"""
from __future__ import annotations

from taban_13 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari. `test_11.py` bunlari model_b15
# ile karsilastirir: biri kayarsa sinav/egitim havuzu ayrisir ve sayilar
# ayni tabloda okunamaz.
#
# !! `veri_ad` bu listede YOK ve olmamali -- ama SEBEBI DEGISTI.
# model_03'e kadar ad farkliydi, ICERIK ayniydi ve kilit icerigi
# siniyordu. model_09'te ICERIK DE FARKLI: bu kolun DUGMESI veri.
# Dolayisiyla `test_09` artik "veri_11 == veri_okul4" demiyor;
# `veri_13`in KENDI IDDIALARINI siniyor (sema, soy agaci, zincir
# siniflari, notrluk, cografya, IZ).
#
# `ek_kip` LISTEDE ve BILEREK ayrisiyor ("tr" -> "tr2"): iliski
# jetonu kelimenin kendisi, ekler gercek allomorf, soru sozcugu var,
# <YOK> dolgusu yok. `test_09` bunu bildirilmis ayrisma diye isler.
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
              "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
              "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = Ayar(
    ad="model_13",

    # --- GOREV: model_b15'in gordugu SINAVIN AYNISI --------------------
    veri_ad="veri_13",   # KENDI veri modulu -- ve BU KOLUN DUGMESI.
    #                      ICERIK model_03'tekinden FARKLI: universite
    #                      hiyerarsisi + gercek soy agaci + gercek
    #                      cografya, 24 iliski / 8 tip / 1608 varlik.
    jeton_ad="tam",      # varlik = JETON DIZISI; ad kac kelimeyse o kadar
    #                      (1-3). Dolgu YOK -- sinirI kesme isareti tasir.
    ek_kip="tr2",        # iliski KELIMENIN KENDISI (fakultesi), ek jetonu
    #                      '  <NIN>  <DIR>  + kim/neresi/hangisi. <SI> YOK.
    # !! BES ALAN DUSTU: bicim / fim_kat / soru_kat / kisayol_kat /
    # ident_frac. Hepsini `egitim_havuzu`nun SATIR TABLOSU okuyordu ve o
    # tablo SILINDI (18 Eylul). Korpus yolunda karsiliklari:
    #
    #   bicim=3       -> metin_11.BICIM, DORT yuzey, cumle basina RASTGELE
    #   fim_kat=2     -> YOK. Kullanici karari, 18 Eylul: "FIM ciksin."
    #   soru_kat=1    -> her belgenin SORU ikizi (biyografi_soru, zincir_soru)
    #   kisayol_kat=1 -> YOK. Iyelik kisa yolu bir SATIR TIPIYDI.
    #   ident_frac    -> kimlik BELGESI (korpus_11.kimlik_belgeleri)
    #
    # Deger olarak birakmak ayar_t0.json'a YALAN yazardi: "fim_kat 2"
    # diyen bir kayit, FIM kosmamis bir kosuyu anlatirdi.
    bicim=1, fim_kat=0, soru_kat=0, kisayol_kat=0, ident_frac=0.0,

    # --- KORPUS -------------------------------------------------------
    t_len=512,           # egitim penceresi (karakter). Physics 3.1 Ek C
    #                      de 512 kullaniyor. Bir pencere 7..11 cumle.
    kopya=40,            # bioS `multiM`: her varlik icin 40 belge, her
    #                      birinde CUMLE SIRASI ve YUZEY BAGIMSIZ secilir.
    #
    # !! 5 -> 40, model_11 KARARI (19 Eylul). IKI olculmus sayi ayni
    # dugmeyi isaret etti:
    #   phi 0.86   Wang 2405.15071'in taradigi 3.6-18 araliginin BIR
    #              MERTEBE ALTINDA. Ve bu, BUDAMA SONRASI DURUST deger;
    #              onceki 4.00 kayitliydi ama model 0.76 goruyordu.
    #   epok 35.3  Muennighoff 2305.16264: 4 epok bedava, 44 acikca
    #              basarisiz rejim. 35,3 o sinira yakin.
    # `kopya` IKISINI BIRDEN duzeltir ve DOGALLIGI BOZMAZ: sayfa ICI
    # yogunluk aynen kalir (zincir_pay 0.20), sadece ayni varlik
    # hakkinda daha cok BELGE olur -- gercek korpusun yapisi bu.
    # `zincir_pay`i yukseltmek ayni phi'yi verirdi ama cumlelerin
    # %40'ini zincir yapardi: model_09'un curutulmus %75'ine dogru.
    #
    # !! 40 SECILDI, 30 DEGIL -- CLAUDE.md kural 6. Olcum egrisi:
    #      kopya 30  wang_phi 4,80  epok 5,9  55M
    #      kopya 40  wang_phi 4,92  epok 4,4  73M   <- SECILEN
    #    Claude once 30'u onerdi ("egrinin dirsegi, otesi bosa") ama o
    #    bir VERIMLILIK argumaniydi. Kullanici, 19 Eylul: *"az veriyle
    #    dogru dil modeli olusturmak gibi bir hedefimiz yok."*  40, hedef
    #    OLAN iki eksende de daha iyi: wang_phi TAVANDA ve epok
    #    Muennighoff'un "4 epok bedava" bandinda.
    tetik=16,            # BIYOGRAFI SORUSU: 40 belgenin 16'si
    #                      (ORAN KORUNDU: 2/5 = 16/40 = %40)
    #                      "X hakkinda ne biliyoruz?" onegiyle gelir,
    #                      3'u onekSIZ. Kullanici karari, 18 Eylul:
    #                      *"bu bir dil modeli, dil modelinin basarili
    #                      olmasi lazim... biyografi ayri soruyla
    #                      isteyebiliriz"*.
    #
    #   ONEK KORPUSU BUYUTMEZ -- mevcut belgelerin bir kismina
    #   konuyor, yenisi eklenmiyor. Yani maruziyet denklemi ve
    #   model_08 kiyasi BOZULMUYOR.
    #
    #   2/5 SECILDI: her iki davranis da veride olsun diye. Hepsi
    #   tetiklenseydi duz biyografi (oneksiz akan metin) HIC
    #   gorulmezdi; hicbiri tetiklenmeseydi istenen davranis YOK.

    ood_pay=0.05,        # dagitim disi bolme
    tam_kayip=True,      # butun pozisyonlarda next-token -- TEK kayip

    # --- BUTCE: adim TAHMIN DEGIL, IKI OLCUMDEN TURETILDI -------------
    # (1) OLGU BASINA CUMLE / korpus gecisi -- SAYILDI, varsayilmadi:
    #     min 10, maks 10, ortalama 10,00  (kopya 5 x [bildirim + soru])
    #
    #     !! ONKAYIT §6 "epok basina 4 cumle" diyordu ve 110 epok
    #     istiyordu. O sayi yanlis: 110 x 10 = 1.100 maruziyet olurdu,
    #     model_08'in 442'sinin 2,5 KATI ve 2,8 kat hesap. Onkayit
    #     duzeltildi (18 Eylul, olcumle).
    #
    # (2) VERIM -- onkayitta L4'te OLCULDU, ve batch BUYUDUKCE DUSUYOR:
    #       batch  32   226.610 yuva/s   <- EN IYI
    #       batch  64   190.844
    #       batch 128   159.261
    #       batch 256   151.354
    #     Darbogaz hesap degil, cekirdek baslatma ve bellek bandi.
    #
    # Ikisi birlikte:
    #     44,0 gecis x 22.320.526 jeton / (32 x 512) = 59.943 -> 60.000
    #     60.000 x 32 x 512 / 22.320.526 = 44,0 gecis -> MARUZIYET 440
    #                                                   (model_08: 442)
    #
    # !! BIR ARA batch 96 / adim 20.000 SECILMISTI (18 Eylul) -- tek
    # gerekcesi adim sayisinin 20.000'e oturmasiydi, yani BIR SAYIYA
    # yaslanmak. Olculdu: batch 96 ayni jeton butcesini %23 DAHA YAVAS
    # kosuyor (94 dk vs 72 dk, L4) ve adim basina yuvayi model_08'in
    # 5,6 katina cikariyor (32'de 1,9 kat). Geri alindi.
    #
    # !! 60.000 GERI ALINDI (18 Eylul). Iki AYRI gerekce birden cokdu:
    #
    # 1) DAYANDIGI SORU KAPANDI. 60.000, model_08'e ESIT MARUZIYETIN
    #    cozumuydu. Kullanici: *"onceki model ile karsilastirma
    #    derdimiz kalmadi... model 8 alakasi yok karsilastirma bitti."*
    #    (CLAUDE.md kural 5.) Esitlenecek bir kol kalmadi.
    #
    # 2) ARITMETIGI DE COKTU. Denklem `22.320.526` jetonluk model_09
    #    korpusuna gore kuruluydu. model_10 korpusu 9.133.886 jeton
    #    (sayfa basina cumle daha az, zincir akisi ayri degil). Ayni
    #    60.000 adim artik 44,0 degil 107,6 EPOK demek:
    #
    #        adim 20.000  ->   35,9 epok   maruziyet  359
    #        adim 40.000  ->   71,8 epok   maruziyet  718
    #        adim 60.000  ->  107,6 epok   maruziyet 1076
    #
    #    Muennighoff (2305.16264): 4 epok bedava, R_D* ~ 15 (yarilanma),
    #    ve 44 epok ACIKCA basarisiz rejim diye aniliyor. 107,6 oranin
    #    cok otesinde.
    #
    # CLAUDE.md kural 1: ILK KOSU 20.000'de durur, uzatmak KULLANICI
    # KARARIDIR. Kullanici, 18 Eylul: *"direkt 60.000 olsun demedim
    # kural ilk sefer 20.000 neyse bitsin bakalim."*  Uzatma
    # SURDURMEDIR: `kos_11.py --adim 40000 --surdur`.
    # --- KORPUSUN BILESIMI -- hiperparametre DEGIL, METNIN ozelligi
    # !! model_09'da korpusun %75'i iki katli tamlama zinciriydi
    # (`X'in kardesinin bolumu`). Kullanici: *"cok sik olsun diye
    # demedim, sadece dilde var olan birsey ve kullaniyoruz ama %70
    # olmasina gerek yok"*  (CLAUDE.md kural 5).
    zincir_pay=0.20,     # sayfa cumlelerinin ~%20'si zincir cumlesi
    # UC ADIMLI zincir: *"hatta 3 lu zincir bile olur"*. Grafta 262.403
    # uclu yol var; 10.000 ORNEKLENIR. Iki adimli havuz 32.902, yani
    # zincirlerin ~%23'u uclu -> korpusun ~%4,6'si. Dogal metinde de
    # derinlestikce SEYRELIR: basit > ikili > uclu.
    n3=10000,
    # REDDETME: soru cumlelerinin ~%5'i. SECILDI, turetilmedi -- gerekce
    # onkayit §3d-2 (R-Tuning'in "pay olculur" bulgusu BELIRSIZLIK
    # rejimine ait; bizimkiler NESNEL OLARAK cevapsiz sorular).
    ret_pay=0.05,
    ret_tut=0.20,        # reddetme verisinin %20'si SINAVA ayrilir

    # --- HIZ: OLCULDU, 19 Eylul, L4, 4.000 adim x 3 yol
    #   fp16 + GradScaler     75.7 ms/adim   1.00x   (mevcut yol)
    #   bf16, scaler YOK      73.7 ms/adim   1.03x
    #   bf16 + torch.compile  41.3 ms/adim   1.83x
    # bpc@4000: 0.246 / 0.243 / 0.247 -- YORUNGE AYRISMIYOR.
    #
    # !! bf16 KAPALI. Onun icin one surdugum IKI gerekcenin IKISI de
    # olcumde SIFIR cikti:
    #   "adim basina GPU-CPU senkronu"  -> kazanc %3, olculemez
    #   "olcek tasinca adim ATLANIR"    -> ATLANAN adim 0, hic olmadi
    # Senkron kaynakta GERCEKTEN var (torch 2.14, _maybe_opt_step)
    # ama MALIYETI yokmus. Dugme duruyor, varsayilan KAPALI.
    bf16=False,
    derle=True,          # torch.compile -- 1.83x, YALNIZ egitim ileri gecisi

    batch=32,
    adim=20000,          # CLAUDE.md kural 1 -- ILK SINIR, tavan DEGIL

    # --- MIMARI: kolun TANIMI -----------------------------------------
    l=8,                 # 8 AYRI katman
    dongu=1,             # DONGU YOK (model_b15: l=4, dongu=2)
    dff=704,             # SwiGLU, 8/3 * 256 = 682,7 -> 64'un kati
    # dar_alfa / dar_kapi VARSAYILANDA (0.0 / False) -> Phi darbogazi YOK

    # --- OPTIMIZASYON: STANDART TARIF ---------------------------------
    betas=(0.9, 0.95),   # nanoGPT / GPT-3 / Llama / Pythia

    # --- BU KOLUN DUGMESI: model_a3'UN RECETESI ------------------------
    # Kullanici karari, 17 Eylul: "a3'teki bir bulgu var ve onu
    # kullanacagiz ama model_a3 ile kiyaslama yapmayacagiz."
    #
    # GEREKCE OLCULDU. Butun kollari `ent` ile dizince tablo tek bir
    # yere isaret ediyor -- ent, ozneye ozgu istatistikten arinmis
    # olcu, ve dort kat butce ona HICBIR SEY vermedi:
    #
    #   model_00 pencere    comp            ent
    #     12-20k           0.1303         0.0417
    #     72-80k           0.2760         0.0407     <- ent YERINDE SAYDI
    #
    #   kol              ent      ne yapildi
    #   model_01 wd0.5  0.0040    2-hop YOK + agirlik cezasi
    #   model_01        0.0117    2-hop YOK
    #   model_b15       0.0290    dongu + Phi
    #   model_02        0.0337    yumusak kopru dongusu
    #   model_00        0.0407    klasik, DORT KAT butce
    #   ------------------------------------------------------
    #   model_b8        0.8650    KOPRU DENETIMI
    #   model_a3        0.8743    wd 0.1 -> 0.5   (A ailesi)
    #
    # Hepsi 0.03-0.04'te cakili; SADECE IKI sey kirdi. Kopru denetimi
    # (b8) egitimde koprunun etiketini ister. Digeri bu recete.
    #
    # !! IKI ALAN BIRLIKTE DEGISIYOR ve bu BILEREK. `model_a3`in kazanci
    # `model_a`ya gore tek dugmeydi (wd), ama model_a'nin arka planinda
    # ZATEN sabit_lr=True vardi. wd 0.5'i cosine'in ustune koymak a3'un
    # kosulunu kurmak DEGILDIR: cosine sonlara dogru LR'yi onda birine
    # indirir, yani agirlik cezasinin ezberi eritmesini tam etkili
    # olacagi yerde durdurur. Recete ikisi birlikte alinir.
    # ATFETME YAPILAMAZ -- CLAUDE.md: "iki dugme birden -> yazilir, kosulur".
    wd=0.5,              # model_00: 0.1   -- CLAUDE.md kural 4 DELINIYOR
    sabit_lr=True,       # model_00: False -- cosine KAPALI, LR SABIT

    # --- !! BU KOLUN DUGMESI: GERI BESLEMELI AGIRLIK ORTALAMASI -------
    # Kullanici, 18 Eylul: *"agirlikli ortalama alarak daha iyi bir
    # model ortaya ciktigi net ama bunun icin surekli 60.000 adim
    # yurutup surekli agirlikli ortalama penceresi almak gerekecek.
    # bu modelde bastan agirlikli ortalama alarak bakalim ki daha erken
    # bir adimda diger modellerin olgunluk seviyesine kac adimda
    # geliyor."*
    #
    # 6000'e kadar normal egitim; ondan sonra HER 2000 ADIMDA
    #     phi   <- 0.5*phi + 0.5*theta          (bir oncekiyle YARI YARIYA)
    #     theta <- phi                          (egitim ORTALANMISTAN devam)
    # Ilk esikte (6000) yalniz phi kurulur; ILK GERCEK ORTALAMA 8000'de.
    #
    # !! MEKANIZMA YENI DEGIL -- Lookahead (1907.08610), 15 Eylul'de
    # eklendi ve `model_a5`te kosuldu. YENI OLAN SORU: a5 TAVANI sordu
    # ("sonradan ortalamakla ayni yere mi geliyor" -> evet, 0.8643 vs
    # 0.8820, ayirt edilemedi). Bu kol HIZI soruyor.
    # !! ESIKLER ADIM DEGIL ORAN olarak tasindi. model_08: isinma
    # kosunun %10'u, ort_bas %30'u, olc_her ile 10 olcum noktasi.
    # Ayni oranlar 60.000'de: 6.000 / 18.000 / 6.000.
    # Mutlak 2000 tasinsaydi isinma kosunun %3,3'u olurdu ve LR
    # cizelgesinin SEKLI model_08'inkinden ayrilirdi -- kiyas
    # zemini oydu.
    # !! UCU DE ORAN, MUTLAK SAYI DEGIL -- `adim` 60.000'den 20.000'e
    # inince elle guncellenmeleri gerekti ve bir kapi bunu yakaladi.
    # Oran olarak yazilinca uzatma (`--surdur`) da bozmaz.
    isinma=2000,
    olc_her=2000,        # 10 olcum noktasi (model_08 ile AYNI sayi)
    ort_bas=6000,       # kosunun %30'u -- model_08'de 6000/20000

    ort_her=2000,        # olc_her ile AYNI -- her olcum noktasi bir esik.
    ort_alfa=0.5,        # "bir onceki ile ortalama" (kullanici, 18 Eylul).
    #   !! TEK ADIMDA esit ortalama, BIRIKMISTE degil:
    #        8000   .50 t6000  .50 t8000
    #       10000   .25 .25 .50
    #       12000   .125 .125 .25 .50
    #   Kosu sonrasi aldigimiz pencere ise BES anlik goruntuyu ESIT
    #   payla ortaliyor (her biri 1/5). Ort gecikme: alfa 0.5 -> 2.000
    #   adim, esit-5 pencere -> 4.000 adim. Yani BU ORTALAMA PENCEREDEN
    #   KISA BELLEKLI. 1/3 eslesirdi; SECILMEDI, cunku (a) kullanicinin
    #   tarifi "bir onceki ile", (b) tek dugme kurali, (c) a5'te 0.5'in
    #   yettigi olculdu -- egri/pencere orani 1,82x -> 1,07x dustu,
    #   yani geri besleme ortalanacak gurultu birakmadi.
)

# Yukarida ACIKCA yazilmayan ama TASINAN degerler burada sabitleniyor:
# bir gun `Ayar()` varsayilani degisirse bu assert'ler once duser.
# --- BU KOLUN DUGMESI ---------------------------------------------
assert AYAR.wd == 0.5, "model_a3 RECETESI -- kullanici karari, 17 Eylul"
assert AYAR.sabit_lr is True, "LR SABIT -- recetenin ikinci yarisi"
assert AYAR.lr == 1e-3, "Pythia-70m ile ayni mertebe"
assert AYAR.isinma == AYAR.adim // 10, (
    "ISINMA kosunun %10'u -- model_08 2000/20000. MUTLAK 2000 "
    "tasinsaydi oran %3,3 olur, LR cizelgesinin SEKLI ayrilirdi.")
assert AYAR.d == 256 and AYAR.nh == 4, "head_dim 64"
assert AYAR.derle is True, (
    "torch.compile OLCULDU: 1.83x (L4, 4.000 adim). Kapatmak bir "
    "SECIM olur ve gerekcesi yazilmali -- sessizce kapanmasin.")
assert AYAR.bf16 is False, (
    "bf16 icin one surulen iki gerekce de OLCUMDE SIFIR cikti "
    "(hiz %3, atlanan adim 0). Acmak bir SECIM olur ve YORUNGEYI "
    "degistirir -- gerekcesi yazilmali.")
assert AYAR.batch == 32, (
    "batch 32 OLCULMUS OPTIMUM -- onkayit §6, L4: 226.610 yuva/s. "
    "Buyutmek verimi DUSURUYOR (128'de 159.261).")
# KORPUS JETONU: OLCULDU, elle yazilmadi. `korpus_11` degisirse bu
# sayi da degisir ve asagidaki kapi bunu SOYLER -- sessizce kaymaz.
KORPUS_JETON = 73_328_521  # OLCULDU, kopya 40 + zincir butcesi +
#   uydurma ad suzgeci ile.  model_10: 9.282.631 (kopya 5, butce YOK).
#   tr2 29.499 = tasarim TAVANI (zincir uzayinin %76'si)
#   phi 4,00   wang_phi 4,92 = TAVAN     EPOK 4,47
#   tensor 160.518 x 512 = 657 MB, kurulum 272 sn
#   TEKIL BELGE 74.390/74.712 (%99,57) -- kopya 40'ta bile cumle sirasi
#   ve yuzey cesitliligi TUKENMIYOR; carpisma yalniz 171 kez reddedildi.
#   model_10: 9.282.631 (kopya 5, butce YOK). Buyumenin ikisi de
#   model_11 karari: kopya 5->20 ve `zincir_butcesi` (tr2 artik TAMAMEN
#   yaziliyor, orneklenmiyor).  tr2 22.832  phi 3,09  wang_phi 3,81
#   Kurulum 73 sn, tensor 337 MB (82.261 x 512).
assert AYAR.adim == 20000, (
    "CLAUDE.md kural 1: ILK KOSU 20.000'de durur. Uzatmak kullanici "
    "karari ve SURDURMEDIR (--surdur), sifirdan kosu degil.")
EPOK = AYAR.batch * AYAR.t_len * AYAR.adim / KORPUS_JETON
# !! ARALIK DEGISTI: 30-40 -> 5-15. Kapi GEVSEMEDI, HEDEFI degisti.
# model_10 35,3 epok kosuyordu ve bu Muennighoff 2305.16264'un
# "4 epok bedava, 44 acikca basarisiz" araliginin UST ucuna yakindi --
# yani veriyi tekrar tekrar okuyup ezberleyen bir rejim. model_11'de
# `kopya` 5 -> 20 oldu; AYNI hesap butcesi (20.000 adim) artik DORT
# KAT daha cok BENZERSIZ metin uzerine yayiliyor. Olgu basina toplam
# maruziyet neredeyse AYNI kaliyor (14,9 cumle x 35,3 epok ~= 60 cumle
# x 8,8 epok); degisen sey TEKRARIN yerini CESITLILIGIN almasi.
assert 4 < EPOK < 15, (
    f"ilk kosu {EPOK:.1f} epok. Muennighoff 2305.16264: 4 epok bedava, "
    f"R_D* ~ 15 yarilanma, 44 epok ACIKCA basarisiz rejim. Hedef "
    f"ARALIK 5-15: 4'un biraz ustu, R_D* yarilanmasinin altinda. "
    f"Uzatilirsa (--surdur) epok da katlanir ve gerekcesi "
    f"OLCULENLER §2'ye yazilir.")
assert AYAR.t_len == 512 and AYAR.kopya == 40, "korpus karari"
assert AYAR.tetik * 5 == AYAR.kopya * 2, (
    "tetik/kopya ORANI korunmali: 2/5. Biyografi onegi tasiyan "
    "belge payi kopya degisince kaymamali.")
assert AYAR.zincir_pay == 0.20 and AYAR.n3 == 10000, (
    "ZINCIR AZINLIK: model_09'da korpusun %75'i zincirdi, dogal metinde "
    "bilesik yapi basit olgudan cok olmaz (CLAUDE.md kural 5).")
assert AYAR.ret_pay == 0.05, "reddetme payi -- kullanici, 18 Eylul: '%5 uygun'"
assert 0 < AYAR.ret_tut < 0.5, (
    "reddetme verisinin bir kismi SINAVA ayrilmali: gordugunu reddetmek "
    "EZBER, gormedigini reddetmek GENELLEME. Olcmek istedigimiz ikincisi.")
assert 0 < AYAR.tetik < AYAR.kopya, (
    "tetik 0 olursa biyografi sorusu HIC ogrenilmez; kopya'ya esit "
    "olursa oneksiz biyografi HIC gorulmez. Ikisi de veride olmali.")
assert AYAR.adim // AYAR.olc_her == 10, "10 olcum noktasi (model_08 ile AYNI)"
assert AYAR.n_olcum_max == 3000, "olcum hatti"
assert AYAR.veri_tohum == 0 and AYAR.tohum == 0

# --- BU KOLUN DUGMESI: geri beslemeli ortalama ACIK ----------------
assert AYAR.ort_bas == 3 * AYAR.adim // 10, (
    "kosunun %30'una kadar NORMAL egitim -- model_08: 6000/20000")
assert AYAR.ort_her == AYAR.olc_her, "her olcum noktasi bir esik"
assert AYAR.ort_alfa == 0.5, "bir onceki ile YARI YARIYA"
assert (AYAR.fim_kat, AYAR.soru_kat, AYAR.kisayol_kat, AYAR.ident_frac,
        AYAR.bicim) == (0, 0, 0, 0.0, 1), (
    "SATIR TABLOSU alanlari KAPALI olmali -- korpus yolunda karsiliklari "
    "var ama bu alanlar artik HICBIR YERDE okunmuyor")
assert AYAR.ort_bas % AYAR.ort_her == 0, (
    "ort_bas esige oturmali; oturmazsa ilk esik ort_bas'ta DEGIL, ondan "
    "sonraki ilk katta olurdu ve defterdeki 6000 YALAN SOYLERDI")
assert AYAR.isinma < AYAR.ort_bas, (
    "isinma BITMEDEN ortalama baslamamali -- LR hala tirmanirken alinan "
    "phi yorungenin kendisini degil, isinmayi ortalar")

# STANDART TARIFIN KENDISI -- projeye ozgu BASKA ne varsa KAPALI olmali.
assert AYAR.betas == (0.9, 0.95), "dil modeli tarifi"
assert AYAR.dar_alfa == 0.0 and not AYAR.dar_kapi, "Phi DARBOGAZI YOK"
assert AYAR.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK"
assert AYAR.mask_poz is None and not AYAR.mask_blok, "MASKE YOK"
assert AYAR.dongu == 1 and AYAR.l == 8, "8 AYRI katman, dongu YOK"
assert AYAR.veri_ad == "veri_13", "BU KOL KENDI veri modulunu okur"
assert "veri_ad" not in GOREV_ALAN, "ad DEGIL, ICERIK sinanir"
