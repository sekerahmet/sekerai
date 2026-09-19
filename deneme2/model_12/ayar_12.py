# -*- coding: utf-8 -*-
"""ayar_12 — bu kolun ayari. TEK BASINA DURUR, devralma YOK.

Her alan `Ayar()` varsayilaninin uzerine ACIKCA yazilir; devralma
gizler, acik yazim gosterir. Asagidaki `assert`ler kapidir: bir
varsayilan kayarsa once onlar duser.

model_12 = 8 blok hibrit (Gated DeltaNet-2 x6 + tam dikkat x2), Muon,
bf16. VERI VE SINAV model_11 ile BIT AYNI (olcme izi 44e6262e37f3) --
degisen her sey MODEL ve EGITIM tarafinda. Gerekceler ve karar kurali:
`belge/onkayit/model_12.md`.
"""
from __future__ import annotations

from taban_12 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari; `test_12` bunlari kiyaslar.
# `veri_ad` LISTEDE DEGIL ve olmamali: ad degil ICERIK sinanir
# (veri_12 kendi semasini, soy agacini, cografyasini ve IZ'ini tutar).
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
              "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
              "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = Ayar(
    ad="model_12",

    # --- VERI VE SINAV: model_11 ile BIT AYNI. DOKUNULMAZ. ------------
    # Kiyasi mumkun kilan tek sey bu. Sinav degisirse elde
    # kiyaslanabilir iki sayi degil, ILGISIZ iki sayi kalir.
    veri_ad="veri_12",   # 1608 varlik / 24 iliski / 8 tip / 7381 olgu
    jeton_ad="tam",      # varlik = kelime dizisi (1-3), dolgu YOK
    ek_kip="tr2",        # gercek Turkce allomorflar + soru sozcukleri
    bicim=1, fim_kat=0, soru_kat=0, kisayol_kat=0, ident_frac=0.0,
    #   ^ satir tablosu KAPALI; korpus yolunda karsiliklari var
    #     (metin_12.BICIM, belge ikizleri) ama bu alanlar okunmuyor.
    #     Deger birakmak ayar_t<N>.json'a yalan yazardi.
    t_len=512,           # egitim penceresi, karakter (Physics 3.1 Ek C)
    kopya=40,            # varlik basina belge (bioS multiM)
    tetik=16,            # kaci biyografi-soru onegiyle (oran 2/5)
    ood_pay=0.05,
    tam_kayip=True,      # butun pozisyonlarda next-token
    zincir_pay=0.20,     # zincir cumlesi AZINLIK (CLAUDE.md kural 5)
    n3=10000,            # uc adimli zincir havuzu
    ret_pay=0.05,        # reddetme payi
    ret_tut=0.2,         # reddetmenin bu kismi SINAVA ayrilir

    # --- EGITIM -------------------------------------------------------
    # bf16: gerekce MIMARI, hiz degil. GDN-2'nin durumu ve birikimleri
    # fp32 (2605.22791 D.3) ve chunk icinde gamma^-1 buyuyor; fp16'nin
    # dar ustel araligi burada gercek risk. Hiz farki olculmustu ve
    # yoktu (73.7 vs 75.7 ms, yorunge ayrismiyor).
    bf16=True,
    derle=True,          # torch.compile -- olculdu 1.83x
    # Muon: 2026 uretim tarifi (GLM-5, Kimi K2, DeepSeek-V4), PyTorch'ta
    # native. Gizli 2B matrisler Muon'a, gomme/1B/3B AdamW'ye
    # (`taban_12._optim_muon`). adjust_lr_fn AdamW RMS'ine esitledigi
    # icin lr=1e-3 tasinabiliyor.
    optim="muon",
    batch=32,            # OLCULMUS OPTIMUM: 226.610 yuva/s (L4)
    adim=20000,          # CLAUDE.md kural 1, ILK SINIR. Sonra
    #                      `kos_12.py --adim 60000 --surdur`.

    # --- MIMARI: kolun TANIMI -----------------------------------------
    l=8, dongu=1, dff=704,       # 8 blok, dongu yok, SwiGLU 8/3*d
    # 3:1 hibrit, dikkat 4. ve 8. blokta.
    #   ORAN   2507.06457 (72 model, 340M+1.3B): "recall rises steadily
    #          as full-attention layers are added and saturates around
    #          3:1". 2026 uretimi ayni yerde (Qwen3.5/3.6, GLM-5.3).
    #   YER    2406.07887 / 2504.03624: "evenly dispersed".
    karisim="GGGAGGGA",
    gdn_v_kat=2,         # Hv = 2*nh, Qwen3-Next ile ayni

    # --- OPTIMIZASYON -------------------------------------------------
    betas=(0.9, 0.95),   # nanoGPT / GPT-3 / Llama / Pythia (0.999 DEGIL)
    wd=0.5,              # model_a3 recetesi -- kullanici karari
    sabit_lr=True,       # cosine KAPALI, isinmadan sonra LR SABIT
    isinma=2000,
    olc_her=2000,        # 10 olcum noktasi
    # LOOKAHEAD KAPALI (model_11'de ACIKTI). Kullanici karari, 19 Eylul.
    #   (1) hicbir 2026 tarifinde yok -- optimizer sarmalayicisi
    #   (2) Muon ile etkilesimi OLCULMEMIS; iki bilinmeyen ust uste
    #   (3) egri okumasini geri beslemeli yapiyordu
    ort_bas=0,           # 0 = KAPALI
    ort_her=2000,        # ort_bas=0 iken OKUNMUYOR
    ort_alfa=0.5,        # ort_bas=0 iken OKUNMUYOR
)

# KORPUS JETONU: OLCULDU, elle yazilmadi. `korpus_12` degisirse bu sayi
# da degisir ve asagidaki EPOK kapisi bunu soyler -- sessizce kaymaz.
#   tr2 29.499 (tasarim tavani)  phi 4,00  wang_phi 4,92 (TAVAN)
#   tensor 160.518 x 512 = 657 MB, kurulum 272 sn
#   tekil belge 74.390/74.712 (%99,57)
KORPUS_JETON = 73_328_521
EPOK = AYAR.batch * AYAR.t_len * AYAR.adim / KORPUS_JETON

# ======================= KAPILAR ========================================
# --- MIMARI
assert AYAR.dongu == 1 and AYAR.l == 8, "8 AYRI blok, dongu YOK"
assert AYAR.d == 256 and AYAR.nh == 4, "head_dim 64"
assert AYAR.karisim == "GGGAGGGA" and len(AYAR.karisim) == AYAR.l, (
    "3:1 hibrit, dikkat 4. ve 8. blokta -- 2507.06457 + 2406.07887")
assert AYAR.gdn_v_kat == 2, "Hv = 2*nh, Qwen3-Next ile ayni"
assert AYAR.dar_alfa == 0.0 and not AYAR.dar_kapi, "Phi DARBOGAZI YOK"
assert AYAR.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK"
assert AYAR.mask_poz is None and not AYAR.mask_blok, "MASKE YOK"

# --- EGITIM
assert AYAR.optim == "muon", "bu kolun ikinci degiskeni -- atfetme YOK"
assert AYAR.bf16 is True, "GDN-2 fp32 durumu; fp16'nin ustel araligi dar"
assert AYAR.derle is True, "torch.compile OLCULDU 1.83x -- sessizce kapanmasin"
assert AYAR.wd == 0.5 and AYAR.sabit_lr is True, "model_a3 recetesi, TAM"
assert AYAR.lr == 1e-3, "Pythia-70m ile ayni mertebe"
assert AYAR.betas == (0.9, 0.95), "dil modeli tarifi, torch varsayilani DEGIL"
assert AYAR.batch == 32, "OLCULMUS OPTIMUM -- 128'de verim dusuyor"
assert AYAR.adim == 20000, "CLAUDE.md kural 1: ILK KOSU 20.000'de durur"
assert AYAR.isinma == AYAR.adim // 10, "isinma kosunun %10'u (model_11 ile AYNI)"
assert AYAR.adim // AYAR.olc_her == 10, "10 olcum noktasi"
assert AYAR.ort_bas == 0, (
    "LOOKAHEAD KAPALI -- 2026 tarifinde yok, Muon ile etkilesimi "
    "olculmemis, ve egri okumasini geri beslemeli yapiyordu.")

# --- VERI VE SINAV (model_11 ile AYNI KALMALI)
assert AYAR.veri_ad == "veri_12", "BU KOL KENDI veri modulunu okur"
assert "veri_ad" not in GOREV_ALAN, "ad DEGIL, ICERIK sinanir"
assert AYAR.t_len == 512 and AYAR.kopya == 40, "korpus karari"
assert AYAR.tetik * 5 == AYAR.kopya * 2, "biyografi onegi orani 2/5"
assert 0 < AYAR.tetik < AYAR.kopya, "tetik kopyanin ICINDE bir alt kume"
assert AYAR.zincir_pay == 0.20 and AYAR.n3 == 10000, "zincir AZINLIK"
assert AYAR.ret_pay == 0.05, "reddetme payi -- kullanici, 18 Eylul"
assert 0 < AYAR.ret_tut < 0.5, "reddetmenin bir kismi SINAVA ayrilir"
assert AYAR.n_olcum_max == 3000, "olcum hatti"
assert AYAR.veri_tohum == 0 and AYAR.tohum == 0
assert (AYAR.fim_kat, AYAR.soru_kat, AYAR.kisayol_kat, AYAR.ident_frac,
        AYAR.bicim) == (0, 0, 0, 0.0, 1), "SATIR TABLOSU alanlari KAPALI"
assert 4 < EPOK < 15, (
    f"EPOK {EPOK:.2f} -- Muennighoff 2305.16264: 4 epok bedava, "
    f"R_D* ~15, 44 ACIKCA basarisiz. Bant disina cikiliyorsa gerekce yaz.")
