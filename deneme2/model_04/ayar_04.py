# -*- coding: utf-8 -*-
"""ayar_04 — model_04'in KENDI ayari. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026:
    *"bunlarin hepsi model_04 folderi altinda olmali. model_04 diger
    hicbir model ile ayni seyi kullanmamali."*

Onceki surum `from model_b15 import AYAR as TABAN` diyordu; yani
model_04'in ayari model_b15 -> model_b14 -> model_b13 -> ... zincirinden
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
                       `test_04.py` bunu her kosuda siniyor.

  MIMARI + OPTIMIZASYON -> STANDART TARIF, referanslariyla.

==========================================================================
STANDART TARIFE GORE: NE DEVRALINMADI

1) LOOKAHEAD ORTALAMASI -- `ort_bas` 0 (YANI KAPALI)

   model_b ailesi 10.000. adimdan sonra "yavas agirlik" tutup her 2000
   adimda bir onunla karistiriyor (kosu logunda "ORTALAMA ACILDI adim
   10000  alfa 0.5  her 2000 adim"). Bu bir OPTIMIZER SARMALAYICISI
   (Lookahead, Zhang ve ark. 2019) ve nanoGPT'de, Llama'da, Pythia'da,
   GPT-2/GPT-3 tarifinde YOK.

   `model_04` standart bir transformerin ne yaptigini olcecekse,
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

from taban_04 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari. `test_04.py` bunlari model_b15
# ile karsilastirir: biri kayarsa sinav/egitim havuzu ayrisir ve sayilar
# ayni tabloda okunamaz.
#
# !! `veri_ad` bu listede YOK ve olmamali: model_04 "veri_04" diyor,
# model_b15 "veri_okul4". Ad farkli, ICERIK ayni -- ve kilit adi degil
# ICERIGI siniyor (graf derin karsilastirma + egitim havuzu + olcme izi).
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
              "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
              "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = Ayar(
    ad="model_04",

    # --- GOREV: model_b15'in gordugu SINAVIN AYNISI --------------------
    veri_ad="veri_04",   # KENDI veri modulu; ICERIK veri_okul4 ile AYNI
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

    # !! ODUL ASAMASININ LR'i -- model_03'te 1e-3 (Ayar varsayilani),
    # burada 1e-5 ve ACIKCA yaziliyor. Bu bir TERCIH DEGIL, olculmus bir
    # ZORUNLULUK (17 Eylul duman testi, 30 odul adimi, model_03'un
    # agirliklarindan baslayarak):
    #       lr        one      seen      comp
    #   (baslangic)  0.9267   0.9133    0.7867
    #     1e-3       0.0067   0.0067    0.0067   <- TAM COKME
    #     1e-4       0.9233   0.8667    0.7267
    #     1e-5       0.9233   0.9133    0.7867   <- SECILEN (kullanici)
    #     1e-6       0.9267   0.9133    0.7867
    # 1e-3 SIFIRDAN egitimin LR'i; odul bir INCE AYAR asamasi ve RLVR
    # literaturu 1e-6..1e-5 kullaniyor.
    #
    # !! ATFETME: model_03'e gore IKI alan degisiyor (odul VE lr) ve
    # AYRILAMAZ. Ama lr serbest bir dugme DEGIL: 1e-3'te kosu modeli 30
    # adimda siliyor, yani "odul + lr 1e-3" diye bir kol YOK. Ayirmak
    # isteyen kol: ODULSUZ, lr 1e-5 ile model_03'u surdurmek
    # (onkayit model_04.md 8 -- KOSULMADI).
    lr=1e-5,

    # ===== BU KOLUN TANIMI: ODUL ======================================
    # Kullanici karari, 17 Eylul 2026. model_03 ile TEK fark bu blok;
    # mimari, veri, havuz, sinav, wd, LR -- hepsi AYNI.
    odul_ac=True,
    odul_bolme="ent_arama",   # `ent` HUKUM bolmesi, uzerinde EGITILMEZ
    odul_batch=64,            # odul adiminin soru sayisi (batch DEGIL)
    odul_g=8,                 # grup boyutu; olculdu: G=8'de sorularin
    #                           %66'si gradyan uretiyor (ikili odulde %5,4)
    odul_sicaklik=1.0,
    odul_denetimli=False,     # denetimli kayip KAPALI -- ODUL TEK OGRETMEN
    odul_kl=0.0,              # KL cezasi KAPALI (acilirsa atfetme bozulur)
    # --- MERDIVEN. Pozitif basamaklar UYDURULMADI: aramanin daralmasindan
    #     turetildi (onkayit model_04.md 3).
    #       E 574 aday -> 1.23 bit -> 0.12
    #       F 104 aday -> 3.38 bit -> 0.34
    #       G  11 aday -> 6.78 bit -> 0.67
    #       H   1 aday -> 10.05 bit -> 1.00
    odul_zemin=-0.50,         # KAPI 1/2/3 dustu -- TURETMESI YOK, ACIK DUGME
    odul_kisayol=-0.30,       # 1 ADIMDA ulasilan cevap -- ACIK DUGME
    odul_e=0.12,
    odul_f=0.34,
    odul_g_aile=0.67,
    odul_h=1.00,
)

# Yukarida ACIKCA yazilmayan ama TASINAN degerler burada sabitleniyor:
# bir gun `Ayar()` varsayilani degisirse bu assert'ler once duser.
# --- BU KOLUN DUGMESI ---------------------------------------------
# --- ODUL: bu kolun TANIMI ----------------------------------------
assert AYAR.odul_ac is True, "model_04'un TANIMI ODUL -- kapaliysa bu model_03"
assert AYAR.odul_bolme == "ent_arama", (
    "odul `ent` uzerinde KOSAMAZ: o bir HUKUM bolmesi. ent_arama'nin "
    "zincir-basi varliklari ent'inkilerle KESISMIYOR (olculdu, 0).")
assert AYAR.odul_denetimli is False, (
    "denetimli kayip KAPALI -- kullanici, 17 Eylul: 'bu egitimde dogru "
    "cevabi vermeyecegiz'")
assert AYAR.odul_kl == 0.0, "KL KAPALI -- acilirsa 'odul mu KL mi' ayrilamaz"
assert AYAR.odul_g >= 2, "grup boyutu en az 2 olmali, yoksa varyans YOK"
_M = (AYAR.odul_zemin, AYAR.odul_kisayol, AYAR.odul_e,
      AYAR.odul_f, AYAR.odul_g_aile, AYAR.odul_h)
assert all(x < y for x, y in zip(_M, _M[1:])), (
    f"MERDIVEN SIRASI BOZUK: {_M}")
assert AYAR.odul_h == 1.0, "H olcegi sabitler"
assert AYAR.odul_zemin < AYAR.odul_kisayol, (
    "ZEMIN kisayolun ALTINDA olmali. Aksi halde model belirsizlikte "
    "SACMALAMAYI ogrenir: bozuk cevap, kopruyu atlamaktan karli olur.")

# --- model_03'ten DEVRALINAN RECETE (bu kolda DEGISMEDI) ----------
assert AYAR.wd == 0.5, "model_a3 RECETESI -- kullanici karari, 17 Eylul"
assert AYAR.sabit_lr is True, "LR SABIT -- recetenin ikinci yarisi"
assert AYAR.lr == 1e-5, (
    "ODUL ASAMASININ LR'i -- 1e-3 modeli 30 adimda siliyor (olculdu)")
assert AYAR.isinma == 2000, "nanoGPT/Llama MUTLAK 2000"
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
assert AYAR.veri_ad == "veri_04", "model_04 KENDI veri modulunu okur"
assert "veri_ad" not in GOREV_ALAN, "ad DEGIL, ICERIK sinanir"
