# -*- coding: utf-8 -*-
"""ayar_08 — BU KOLUN KENDI ayari. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026:
    *"bunlarin hepsi model_05 folderi altinda olmali. model_05 diger
    hicbir model ile ayni seyi kullanmamali."*

Onceki surum `from model_b15 import AYAR as TABAN` diyordu; yani
model_05'in ayari model_b15 -> model_b14 -> model_b13 -> ... zincirinden
DEVRALINIYORDU ve zincirin herhangi bir halkasi degisince sessizce
kayardi. Artik oyle degil: her alan ASAGIDA, `Ayar()` varsayilaninin
uzerine, TEK TEK ve gerekcesiyle yaziliyor.

Bunun bir yan faydasi var ve ilk kurulumda tam da bu kacmisti: devralma
GIZLIYORDU, acik yazim GOSTERIYOR.

==========================================================================
IKI GRUP, IKI GEREKCE

  GOREV alanlari    -> model_b15'in gordugu sinavla AYNI KALMASI
                       BEKLENEN alanlar. model_03'e kadar hepsi
                       birebirdi. model_05 IKISINI BILEREK degistiriyor
                       ve ikisi de BU KOLUN TANIMI:
                         veri_ad  yeni veri (GOREV_ALAN'da zaten yok)
                         ek_kip   "tr" -> "tr2"
                       `test_05.py` bunlari BILDIRILMIS AYRISMA diye
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

   `model_05` standart bir transformerin ne yaptigini olcecekse,
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

from taban_08 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari. `test_05.py` bunlari model_b15
# ile karsilastirir: biri kayarsa sinav/egitim havuzu ayrisir ve sayilar
# ayni tabloda okunamaz.
#
# !! `veri_ad` bu listede YOK ve olmamali -- ama SEBEBI DEGISTI.
# model_03'e kadar ad farkliydi, ICERIK ayniydi ve kilit icerigi
# siniyordu. model_05'te ICERIK DE FARKLI: bu kolun DUGMESI veri.
# Dolayisiyla `test_05` artik "veri_05 == veri_okul4" demiyor;
# `veri_05`in KENDI IDDIALARINI siniyor (sema, soy agaci, zincir
# siniflari, notrluk, cografya, IZ).
#
# `ek_kip` LISTEDE ve BILEREK ayrisiyor ("tr" -> "tr2"): iliski
# jetonu kelimenin kendisi, ekler gercek allomorf, soru sozcugu var,
# <YOK> dolgusu yok. `test_05` bunu bildirilmis ayrisma diye isler.
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
              "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
              "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = Ayar(
    ad="model_08",

    # --- GOREV: model_b15'in gordugu SINAVIN AYNISI --------------------
    veri_ad="veri_08",   # KENDI veri modulu -- ve BU KOLUN DUGMESI.
    #                      ICERIK model_03'tekinden FARKLI: universite
    #                      hiyerarsisi + gercek soy agaci + gercek
    #                      cografya, 24 iliski / 8 tip / 1608 varlik.
    jeton_ad="tam",      # varlik = JETON DIZISI; ad kac kelimeyse o kadar
    #                      (1-3). Dolgu YOK -- sinirI kesme isareti tasir.
    ek_kip="tr2",        # iliski KELIMENIN KENDISI (fakultesi), ek jetonu
    #                      '  <NIN>  <DIR>  + kim/neresi/hangisi. <SI> YOK.
    bicim=3,             # UC TURKCE SIRA: kanonik / yuklem basta /
    #                      iliski basta. Hepsi BILDIRIM.
    kisayol_kat=1,       # IYELIK KISA YOLU: kisi basina 1 tur.
    #                      "Ayse Sahin'in dekani kimdir? Melek Aydin'dir."
    #                      DORT iliski (fakultesi, konusu, dekani,
    #                      universitesi) x 480 kisi = 1.920; bunun 644'u
    #                      TUTULAN veriye dokundugu icin URETILMIYOR
    #                      (bas 384, kenar 198, zincir 62 -- bkz.
    #                      `taban_08._kisayol_kur`) -> 1.276 satir,
    #                      havuzun %0,3'u. SAYIYI KOD BASIYOR; bu yorum
    #                      bir daha kayarsa havuz kutugune bak.
    #                      Kullanici, 18 Eylul: "cok olmayacak sorular
    #                      ekleyelim" -- pay ONCEDEN hesaplandi.
    #                      (`fim_kat=2` dersi: pay SECILIR, turemez --
    #                      OLCULENLER §1g.)
    fim_kat=2,           # bildirim satiri basina 2 bosluk varyanti
    soru_kat=1,          # !! model_07'NIN EKLENTISI; model_08 DEVRALDI: her zincir icin
    #                      BIR soru satiri.
    #                      "... arkadasi kimdir? Derya Yilmaz'dir."
    #
    #   1 SECILDI, TUREMEDI. `fim_kat=2` dersi (OLCULENLER §1g): oradaki
    #   2, FIM'i havuzun %53,3'u yapti ve kimse bu payi secmemisti.
    #   Burada pay ONCEDEN hesaplandi:
    #
    #     duz BILDIRIM (3 bicim)   93.648   %24,0
    #     BOSLUK DOLDURMA (FIM)   187.296   %48,0
    #     SORU (YENI)              31.216   % 8,0
    #     KIMLIK                   78.264   %20,0
    #     TOPLAM                  390.424
    #
    #   (KIMLIK havuzla birlikte buyur: `ident_frac=0.2` bir SAYI degil
    #   PAY. model_06'da 70.655'ti, havuz buyudugu icin 78.264 oldu --
    #   payi %20'de kaldi. SORU eklemek KIMLIGI seyreltmiyor.)
    #
    #   %8,2 az gorunuyor ama `kimdir`in anlamini KIMLIK satirlarindan
    #   ayirmak icin PAY degil BAGLAM gerekiyor: soru sozcugunden ONCEKI
    #   jeton ikisini zaten ayiriyor (varlik -> kimlik, iliski -> soru).
    #   Buyutmek ayri bir dugme olur; once BU olculur.
    ood_pay=0.05,        # dagitim disi bolme
    ident_frac=0.2,      # kimlik koprusu
    ident_kip="q1",
    tam_kayip=True,      # butun pozisyonlarda next-token

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
    ort_bas=6000,        # 0 = KAPALI idi. Bu adimdan SONRA ortala.
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
assert AYAR.lr == 1e-3 and AYAR.isinma == 2000, "nanoGPT/Llama MUTLAK 2000"
assert AYAR.d == 256 and AYAR.nh == 4, "head_dim 64"
assert AYAR.batch == 512 and AYAR.adim == 20000, "kural 1: ILK SINIR"
assert AYAR.olc_her == 2000 and AYAR.n_olcum_max == 3000, "olcum hatti"
assert AYAR.veri_tohum == 0 and AYAR.tohum == 0

# --- BU KOLUN DUGMESI: geri beslemeli ortalama ACIK ----------------
assert AYAR.ort_bas == 6000, "6000'e kadar NORMAL egitim (kullanici, 18 Eylul)"
assert AYAR.ort_her == 2000 == AYAR.olc_her, "her olcum noktasi bir esik"
assert AYAR.ort_alfa == 0.5, "bir onceki ile YARI YARIYA"
assert AYAR.kisayol_kat == 1, "IYELIK KISA YOLU: kisi basina 1x3 satir"
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
assert AYAR.veri_ad == "veri_08", "BU KOL KENDI veri modulunu okur"
assert "veri_ad" not in GOREV_ALAN, "ad DEGIL, ICERIK sinanir"
