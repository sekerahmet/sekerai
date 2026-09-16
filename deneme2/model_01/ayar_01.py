# -*- coding: utf-8 -*-
"""ayar_01 — model_01'in KENDI ayari. TEK BASINA DURUR.

Kullanici karari, 17 Eylul 2026: *"model_00 dedigim aslinda DENEY. Ben
her deneyin kendi klasorunde olmasini istiyorum. Ortak bir seyler
kullaninca karisabiliyor, hata yapabiliyoruz."*

==========================================================================
BU KOL NE SORUYOR

`model_00` ile TEK FARK: `hop2_pay = 0.0`. Yani model EGITIMDE tek bir
2-hop ornegi GORMEZ. Veri AYNI, graf AYNI, SINAV AYNI (olcme izi
d6751004648c) -- degisen yalnizca modelin GORDUGU egitim satirlari.

HIPOTEZ (16-17 Eylul, olcumlerden turedi):

  Egitimde 2-hop ornekleri oldugu icin model, OZNE BASINA ILISKI-CIFTI
  TABLOSU ogrenip egitim hedefini TAMAMEN karsiliyor. Bilesim devresine
  hic basinc dogmuyor.

Dayanak (hepsi OLCULDU):

  parcalar tek tek       1.0000 / 1.0000     bilgi EKSIK DEGIL
  ikisi birden           0.0550 (b15) .. 0.2550 (model_00)
  kopru dogrusal sondada YOK    0.0482 vs taban 0.0361
  h(e1,r1) ~ h(e2)       YOK    +0.0053, ortanca sira 132/500
  yanlislarin %99,7'si   dogru TIPTE, %91,5'i r2 MENZILINDE
  ozne zincir basi OLMUS -> comp 0.26 ; OLMAMIS -> ent 0.04
  ama KAC KEZ oldugu     r = -0.11  (esik var, egim YOK)
  model_b8 kopruyu ZORLAYINCA  comp 0.9747  -> gorev COZULEBILIR

Literatur ayni yonu gosteriyor: 2505.17923 ve 2509.24653 egitimde 2-hop
KULLANMIYOR ("training set is restricted to contain only single-hop
data, ensuring that all two-hop data are out-of-distribution") ve
ikisinde de standart GPT-2 bilesimi COZUYOR.

!! Hipotez SONRADAN kova analizleriyle sinanmaya calisildi, UC KEZ
BASARISIZ olundu: hangi eksende kovalanirsa kovalansin "kolay ozne"
etkisiyle karisiyor. Bu veride sonradan ayirt edilemiyor -- ayirt edici
sey bir ANALIZ degil, bu EGITIM MUDAHALESI.

==========================================================================
!! `seen` KAPISI BU KOLDA TANIMSIZ

2-hop egitimden cikinca `seen` bolmesi EGITILMEMIS zincirleri olcer.
`SAGLIK-EZBER (seen >= 0.95)` bu kolda GEVSETILMEDI -- ANLAMSIZ hale
geldi ve oyle raporlanir. `SAGLIK-1HOP (one >= 0.98)` yerinde kalir ve
ON KOSUL olmaya devam eder: model olgulari ogrenememisse `comp`
yorumlanmaz.

Yan fayda: `seen` artik ikinci bir `comp` gibi davranir (ayni ozneler,
egitilmemis zincirler) -- IC KONTROL olarak okunur.

==========================================================================
DIGER HER SEY `model_00` ILE AYNI

Mimari (l=8, dongu=1, dff=704, RoPE, SwiGLU, bagli gomme, bias yok),
optimizasyon (wd 0.1, cosine->lr/10, isinma 2000, lr 1e-3,
betas (0.9,0.95), LOOKAHEAD KAPALI), veri (`veri_01`, icerik
`veri_00` ile AYNI, IZ 3f6751c4ccd4), sinav bolmeleri.
"""
from __future__ import annotations

from taban_01 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari. `test_01.py` bunlari model_b15
# ile karsilastirir: biri kayarsa sinav/egitim havuzu ayrisir ve sayilar
# ayni tabloda okunamaz.
#
# !! `veri_ad` bu listede YOK ve olmamali: model_01 "veri_01" diyor,
# model_b15 "veri_okul4". Ad farkli, ICERIK ayni -- ve kilit adi degil
# ICERIGI siniyor (graf derin karsilastirma + egitim havuzu + olcme izi).
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
              "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
              "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = Ayar(
    ad="model_01",

    # --- BU KOLUN TEK DUGMESI ------------------------------------------
    hop2_pay=0.0,        # EGITIMDE 2-HOP YOK (model_00: 1.0)

    # --- GOREV: model_b15'in gordugu SINAVIN AYNISI --------------------
    veri_ad="veri_01",   # KENDI veri modulu; ICERIK veri_okul4 ile AYNI
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

    # --- IKINCI KOSUNUN DUGMESI: AGIRLIK CEZASI -----------------------
    # Kullanici karari, 17 Eylul: "simdi kisayolu cezalandiracagiz ama
    # ayri model gerek yok. model_01'de degerini 5 katina, yani 0,5'e
    # cikarip ayni defterde tekrar calistiralim."
    #
    # GEREKCE ILK KOSUDA OLCULDU: hop2_pay=0.0 kolunda kisayol
    # COKMEDI, BUYUDU -- ent_kisayol 0.0760 (model_00) -> 0.4573, ALTI
    # KAT. Model egitimde TEK BIR 2-hop dizisi gormedigi halde r2'yi
    # dogrudan ozneye uyguluyor. Yani kisayol 2-hop orneklerinden
    # OGRENILMIYOR; 1-hop'tan gelip 2-hop sorusuna TASIYOR.
    #
    # !! BU CLAUDE.md KURAL 4'U DELIYOR ("wd 0.1 ARTIK ARANMAZ").
    # Kural kullanicinindi, delme karari da kullanicinin -- 17 Eylul.
    wd=0.5,              # 0.1'in BES KATI

    # --- OPTIMIZASYON: STANDART TARIF ---------------------------------
    betas=(0.9, 0.95),   # nanoGPT / GPT-3 / Llama / Pythia
    sabit_lr=False,      # cosine -> lr/10 (CLAUDE.md kural 4)
    # ort_bas VARSAYILANDA (0) -> LOOKAHEAD KAPALI
)

# Yukarida ACIKCA yazilmayan ama TASINAN degerler burada sabitleniyor:
# bir gun `Ayar()` varsayilani degisirse bu assert'ler once duser.
assert AYAR.wd == 0.5, "IKINCI KOSUNUN DUGMESI -- kullanici karari, 17 Eylul"
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
assert AYAR.veri_ad == "veri_01", "model_01 KENDI veri modulunu okur"
assert AYAR.hop2_pay == 0.0, "BU KOLUN TEK DUGMESI: egitimde 2-hop YOK"
assert "veri_ad" not in GOREV_ALAN, "ad DEGIL, ICERIK sinanir"
