# -*- coding: utf-8 -*-
"""ayar_03 — model_03'in KENDI ayari. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026:
    *"bunlarin hepsi model_03 folderi altinda olmali. model_03 diger
    hicbir model ile ayni seyi kullanmamali."*

Onceki surum `from model_b15 import AYAR as TABAN` diyordu; yani
model_03'in ayari model_b15 -> model_b14 -> model_b13 -> ... zincirinden
DEVRALINIYORDU ve zincirin herhangi bir halkasi degisince sessizce
kayardi. Artik oyle degil: her alan ASAGIDA, `Ayar()` varsayilaninin
uzerine, TEK TEK ve gerekcesiyle yaziliyor.

Bunun bir yan faydasi var ve ilk kurulumda tam da bu kacmisti: devralma
GIZLIYORDU, acik yazim GOSTERIYOR.

==========================================================================
IKI GRUP, IKI GEREKCE

  GOREV alanlari    -> model_b15'in gordugu SINAVIN AYNISI olmali.
                       Ayni veriyi gormezse kol hicbir sey olcmez;
                       sinav ve egitim havuzu BIT DUZEYINDE ayni kalmali.
                       `test_03.py` bunu her kosuda siniyor.

  MIMARI + OPTIMIZASYON -> STANDART TARIF, referanslariyla.

==========================================================================
STANDART TARIFE GORE: NE DEVRALINMADI

1) LOOKAHEAD ORTALAMASI -- `ort_bas` 0 (YANI KAPALI)

   model_b ailesi 10.000. adimdan sonra "yavas agirlik" tutup her 2000
   adimda bir onunla karistiriyor (kosu logunda "ORTALAMA ACILDI adim
   10000  alfa 0.5  her 2000 adim"). Bu bir OPTIMIZER SARMALAYICISI
   (Lookahead, Zhang ve ark. 2019) ve nanoGPT'de, Llama'da, Pythia'da,
   GPT-2/GPT-3 tarifinde YOK.

   `model_03` standart bir transformerin ne yaptigini olcecekse,
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

from taban_03 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari. `test_03.py` bunlari model_b15
# ile karsilastirir: biri kayarsa sinav/egitim havuzu ayrisir ve sayilar
# ayni tabloda okunamaz.
#
# !! `veri_ad` bu listede YOK ve olmamali: model_03 "veri_03" diyor,
# model_b15 "veri_okul4". Ad farkli, ICERIK ayni -- ve kilit adi degil
# ICERIGI siniyor (graf derin karsilastirma + egitim havuzu + olcme izi).
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
              "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
              "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = Ayar(
    ad="model_03",

    # --- GOREV: model_b15'in gordugu SINAVIN AYNISI --------------------
    veri_ad="veri_03",   # KENDI veri modulu; ICERIK veri_okul4 ile AYNI
    jeton_ad="tam",      # varlik = JETON DIZISI (3 yuva), tek jeton DEGIL
    ek_kip="tr",         # Turkce ek jetonlari:  '  <NIN>  <SI>  <DIR>
    bicim=3,             # 3 yuzey bicimi (ek_kip olmadan ANLAMSIZ)
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
    # ort_bas VARSAYILANDA (0) -> LOOKAHEAD KAPALI
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

# STANDART TARIFIN KENDISI -- projeye ozgu ne varsa KAPALI olmali.
assert AYAR.ort_bas == 0, "LOOKAHEAD KAPALI"
assert AYAR.betas == (0.9, 0.95), "dil modeli tarifi"
assert AYAR.dar_alfa == 0.0 and not AYAR.dar_kapi, "Phi DARBOGAZI YOK"
assert AYAR.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK"
assert AYAR.mask_poz is None and not AYAR.mask_blok, "MASKE YOK"
assert AYAR.dongu == 1 and AYAR.l == 8, "8 AYRI katman, dongu YOK"
assert AYAR.veri_ad == "veri_03", "model_03 KENDI veri modulunu okur"
assert "veri_ad" not in GOREV_ALAN, "ad DEGIL, ICERIK sinanir"
