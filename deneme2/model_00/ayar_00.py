# -*- coding: utf-8 -*-
"""ayar_00 — model_00'in KENDI ayarlari. Paylasilan tercihlere ESIR DEGIL.

Kullanici, 16 Eylul 2026: *"standart seyi kurmamizdaki sikinti ne?
Ayar dosyasi ise onu ayar00 diye bir dosya yap, ordan okusun."*

Hakli soru. `model_00`un iddiasi "STANDART tarif"; o iddia, projenin
birikmis optimizasyon tercihlerini sessizce devralarak korunamaz. Bu
dosya, hangi alanin NEREDEN geldigini ve NEDEN o degerde oldugunu
tek tek yaziyor.

==========================================================================
IKI GRUP, IKI KAYNAK

  VERI alanlari      -> model_b15'ten AYNEN devralinir.
                        Ayni veriyi gormezse kol hicbir sey olcmez;
                        sinav ve egitim havuzu BIT DUZEYINDE ayni kalmali.

  MIMARI + OPTIMIZASYON -> STANDART TARIF. Referanslarla birlikte
                        asagida tek tek gerekcelendirildi.

==========================================================================
DEVRALINMAYAN IKI AYAR -- ve neden

1) ort_bas 10000 -> 0     LOOKAHEAD ORTALAMASI KAPATILDI

   Paylasilan ayar 10.000. adimdan sonra "yavas agirlik" tutup her 2000
   adimda bir onunla karistiriyor (kosu logunda "ORTALAMA ACILDI adim
   10000  alfa 0.5  her 2000 adim"). Bu bir OPTIMIZER SARMALAYICISI
   (Lookahead, Zhang ve ark. 2019) ve nanoGPT'de, Llama'da, Pythia'da,
   GPT-2/GPT-3 tarifinde YOK.

   `model_00` standart bir transformerin ne yaptigini olcecekse, standart
   olmayan bir optimizasyon numarasiyla kosamaz. KAPALI.

   !! Bu, model_b15 ile arasinda EK bir fark demek. Kolun tanimi zaten
   "referans", ablasyon degil (onkayit model_00.md 3), ve CLAUDE.md
   "KIYAS ARTIK ARKA PLANDA" diyor.

2) betas (0.9, 0.999) -> (0.9, 0.95)

   0.999 PyTorch'un VARSAYILANI, dil modeli tarifi degil. Dort referans
   da 0.95 kullaniyor:
       nanoGPT   beta2 = 0.95
       GPT-3     beta2 = 0.95
       Llama     beta2 = 0.95
       Pythia    beta2 = 0.95

   beta2 gradyan BUYUKLUGUNUN hafiza suresi; etkin pencere ~1/(1-beta2):
       0.999 -> ~1000 adim      0.95 -> ~20 adim
   20.000 adimlik bir kosuda 1000 adim, egitimin %5'i.

   !! DURUSTCE: bizim veri rejiminde etkisi KUCUK beklenir. beta2'nin
   asil onemli oldugu durum bir jetonun binlerce adimda bir gorunmesi;
   bizde EN SEYREK jeton bile ~2 adimda bir geciyor (olculdu: varlik
   jetonu sikliklari 632.296 .. 256, batch 512'de en seyrek ~0,44
   kez/batch). Yani gerekce "tarif boyle", olculmus bir kazanc DEGIL.

==========================================================================
DEVRALINAN AMA ZATEN STANDART OLANLAR

    wd = 0.1              nanoGPT / Pythia / Qwen2.5 SFT -- CLAUDE.md kural 4
    cosine -> lr/10       nanoGPT / Pythia (min_lr = lr/10) -- kural 4
    isinma = 2000         nanoGPT warmup_iters=2000, Llama warmup=2000
                          MUTLAK deger olarak BIREBIR ayni. (Oran farkli
                          cikiyor cunku onlarin kosusu 600k adim, bizimki
                          20k -- ama tarifin yazdigi sayi 2000.)
    lr = 1e-3             Pythia-70m 1e-3. Bizim model 6,5M; bu mertebede
                          dogru yon KUCUK model -> BUYUK lr.
    grad clip 1.0         hepsi
    batch = 512           olcum hattiyla paylasilir, degistirilmedi
    adim = 20000          CLAUDE.md kural 1 (ILK SINIR)
    olc_her / n_olcum_max OLCUM alanlari -- kiyas icin AYNI kalmali
    tam_kayip = True      butun pozisyonlarda next-token: standart LM kaybi

==========================================================================
"""
from __future__ import annotations

import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_B), "model_a"),
           os.path.join(os.path.dirname(_B), "model_b"),
           os.path.dirname(_B)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_b15 import AYAR as TABAN                          # noqa: E402

# model_b15 ile BIREBIR AYNI kalmasi GEREKEN alanlar. `test_00.py` bunu
# her kosuda sinar: biri kayarsa sinav/egitim havuzu ayrisir ve kol
# kiyaslanamaz hale gelir.
VERI_ALAN = ("veri_ad", "veri_tohum", "ent_pay", "comp_pay", "arama_pay",
             "ood_pay", "kati_pay", "jeton_ad", "ek_kip", "bicim",
             "ident_frac", "ident_kip", "belge_pay", "tam_kayip",
             "batch", "adim", "tohum", "olc_her", "n_olcum_max")

AYAR = TABAN.degistir(
    ad="model_00",
    # --- MIMARI: kolun tanimi -----------------------------------------
    l=8,                 # 8 AYRI katman
    dongu=1,             # DONGU YOK
    dff=704,             # SwiGLU, 8/3 * 256 = 682,7 -> 64'un kati
    dar_alfa=0.0,        # Phi darbogazi YOK
    dar_kapi=False,      # ogrenilen gecit YOK
    # --- OPTIMIZASYON: STANDART TARIF ---------------------------------
    betas=(0.9, 0.95),   # nanoGPT / GPT-3 / Llama / Pythia
    ort_bas=0,           # LOOKAHEAD KAPALI -- hicbir standart tarifte YOK
)

# Devraldigimiz seylerin GERCEKTEN devralindigini burada da sabitliyoruz:
# bir gun model_b15 degisirse bu assert'ler once duser.
assert AYAR.wd == 0.1 and AYAR.sabit_lr is False, "CLAUDE.md kural 4"
assert AYAR.isinma == 2000, "nanoGPT/Llama MUTLAK 2000"
assert AYAR.ort_bas == 0, "Lookahead KAPALI olmali"
assert AYAR.betas == (0.9, 0.95), "dil modeli tarifi"
assert AYAR.dongu == 1 and AYAR.l == 8, "8 AYRI katman, dongu YOK"
