# -*- coding: utf-8 -*-
"""ayar_13 -- model_13'un ayari. VERI kismi kopya, EGITIM kismi kolun kendisi.

Kullanici karari, 20 Eylul 2026: *"tabiki kirp artik yeni bir sayfadayiz"*.

Ilk halinde `ayar_11` aynen kopyalanmisti: 446 satir, 33 alan, 32 assert.
OLCULDU: veri yolu o 33 alanin yalniz 18'ini okuyor; kalan 21'i
transformer egitim dugmesi (adim, batch, wd, sabit_lr, isinma, ort_*,
derle, bf16, l, dongu, dff, betas ...) ve bu kol transformer
egitmiyor. Assert'ler de model_11'in gerekcelerini tasiyordu --
"model_a3 RECETESI", "torch.compile 1.83x", "batch 32 OLCULMUS
OPTIMUM" -- bu kolda karsiligi olmayan cumleler. CLAUDE.md kural 7.

IKI GRUP:
  VERI      `taban_13.veri_kur` / `egitim_havuzu` okur. Degerler
            `ayar_11`den AYNEN -- korpus ve sinav degismesin diye.
            Kapisi izler: graf 3cd9a2575e47, olcme 44e6262e37f3.
  MODEL     Bu kolun KENDI dugmeleri. `Ayar`da karsiligi yok, cunku
            `Ayar` transformer icin yazilmis.
"""
from __future__ import annotations

import dataclasses as dc

from taban_13 import Ayar                                    # noqa: E402

# Sinavin AYNI kalmasi GEREKEN alanlari -- biri kayarsa korpus/sinav
# ayrisir ve sayilar model_11'in tablosuyla ayni yerde okunamaz.
GOREV_ALAN = ("veri_tohum", "ent_pay", "comp_pay", "arama_pay", "ood_pay",
              "kati_pay", "jeton_ad", "ek_kip", "belge_pay", "tam_kayip",
              "tohum", "n_olcum_max")

AYAR = Ayar(
    ad="model_13",

    # --- VERI: model_11 ile BIREBIR -----------------------------------
    veri_ad="veri_13",   # kolun KENDI kopyasi; icerik veri_11 ile ayni
    veri_tohum=0,
    tohum=0,
    jeton_ad="tam",      # varlik = jeton dizisi, ad kac kelimeyse o kadar
    ek_kip="tr2",        # iliski kelimenin kendisi, gercek ek allomorflari

    # --- KORPUS -------------------------------------------------------
    t_len=512,           # egitim penceresi (karakter)
    kopya=40,            # her varlik icin 40 belge
    tetik=16,            # 40 belgenin 16'si biyografi onegi tasir (2/5)
    zincir_pay=0.20,     # sayfa cumlelerinin ~%20'si zincir
    n3=10000,            # uc adimli zincir orneklemi
    ret_pay=0.05,        # soru cumlelerinin ~%5'i reddetme
    ret_tut=0.20,        # reddetme verisinin %20'si SINAVA ayrilir
    tam_kayip=True,      # `egitim_havuzu` bunu assert ediyor

    # --- BOLME --------------------------------------------------------
    ent_pay=0.20,
    comp_pay=0.10,
    arama_pay=0.25,
    ood_pay=0.05,
    kati_pay=0.0,
    belge_pay=0.0,
    n_olcum_max=3000,    # olcme listesi tavani
)

# --- MODEL: BU KOLUN dugmeleri.  `Ayar` transformer icin yazilmis,
#     bunlarin orada karsiligi yok.
D_BOYUT = 9          # birim basina koordinat boyutu.
#                      Belge D=3,8,16,32 olctu ve BUYUDUKCE KOTULESIYOR:
#                      uzaklikla puanlamada yuksek boyutta butun noktalar
#                      birbirine esit uzaklasiyor. Tek yuvada DD=D.
K_PENCERE = 6        # son K birim. Sozlukte kelime basina ~2 birim var,
#                      yani 6 birim ~ 3 kelime -- belgenin k=3'u.
M_BILESEN = 4        # hedef nokta sayisi. Belge 8 ve 16'yi denemis,
#                      fayda etmemis. Baglam iki yone acikken tek nokta
#                      ikisinin ortasina duser.
H_GIZLI = 96         # yonlendirme aginin tek gizli katmani
LAMBDA = 0.5         # kapanma cezasi. Belgede olmadan iki adim %43.
LR = 3e-3            # Adam
BATCH = 8192         # GPU. Belgede 256 (numpy/CPU).
EPOK = 20

# --- KAPILAR: yalniz BU KOLUN dogruladigi seyler ----------------------
assert AYAR.veri_ad == "veri_13", "kol KENDI veri modulunu okur"
assert AYAR.ek_kip == "tr2" and AYAR.jeton_ad == "tam", (
    "korpus yuzeyi degismemeli -- sinav izi buna bagli")
assert AYAR.t_len == 512 and AYAR.kopya == 40, "korpus karari (model_11)"
assert AYAR.tetik * 5 == AYAR.kopya * 2, (
    "tetik/kopya ORANI 2/5 korunmali")
assert AYAR.zincir_pay == 0.20 and AYAR.n3 == 10000, (
    "ZINCIR AZINLIK -- CLAUDE.md kural 5")
assert AYAR.ret_pay == 0.05 and 0 < AYAR.ret_tut < 0.5, (
    "reddetmenin bir kismi SINAVA ayrilir: gormedigini reddetmek GENELLEME")
assert AYAR.n_olcum_max == 3000, "olcme listesi tavani"
assert AYAR.veri_tohum == 0 and AYAR.tohum == 0

assert D_BOYUT in (3, 6, 9, 12), (
    "uzaklikla puanlama YUKSEK BOYUTTA bozulur -- belge D=16/32'yi olctu "
    "ve ikisi de daha kotu. Buyutmek bir SECIM olur ve gerekcesi yazilir.")
assert M_BILESEN >= 1 and H_GIZLI > 0 and 0.0 <= LAMBDA
assert K_PENCERE >= 2, "MLP en az iki birim gormeli"

# Kirpma sirasinda alan DUSMEDIGININ kapisi: veri yolunun okudugu 18
# alanin hepsi hala ACIKCA yaziliyor mu?
_VERI_ALAN = ("veri_ad", "veri_tohum", "tohum", "jeton_ad", "ek_kip",
              "t_len", "kopya", "tetik", "zincir_pay", "n3", "ret_pay",
              "ret_tut", "tam_kayip", "ent_pay", "comp_pay", "arama_pay",
              "ood_pay", "kati_pay", "belge_pay", "n_olcum_max")
assert all(hasattr(AYAR, a) for a in _VERI_ALAN)
