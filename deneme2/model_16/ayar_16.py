# -*- coding: utf-8 -*-
"""ayar_16 -- bu kolun BUTUN dugmeleri, her birinin gerekcesiyle.

CLAUDE.md: bir ayar VARSAYILAN OLDUGU ICIN secilmez. Buradaki her
sayinin yaninda ya bir HESAP, ya bir OLCUM, ya da "OLCULMEDI" yazar.
Ucu de yoksa sayi buraya girmez.

IKI BOLUM, iki ayri sozlesme:

  VERI    model_13/model_11 ile BIREBIR ayni olmak ZORUNDA. Sinav
          sabit kalsin diye: bir deger kayarsa `ezber_olgu 0,9353` ile
          `ezber_olgu 0,0204` artik ayni tabloda okunamaz.
          Kapisi: graf 3cd9a2575e47, olcme 44e6262e37f3.

  MODEL   BU DOSYADA YOK.  model_16'nin mimarisi model_15'ten geliyor
          ve butun ayarlari model_16.py'nin basinda duruyor.
          model_14'un mimari sabitleri (D_DURUM, K_KOD, R_CAPA, DELTA,
          BETA, HAFIZA, SAAT, PENCERE, ISINMA, ATLA, LR, BATCH, EPOK...)
          BURADAN CIKARILDI -- 254 satir.  Hicbiri okunmuyordu:
          veri yolu Ayar'in yalniz 19 alanini kullaniyor.
"""
from __future__ import annotations

from taban_16 import Ayar

# =====================================================================
# VERI -- DEGISMEZ.  Degerler model_13 ile birebir.
# =====================================================================
AYAR = Ayar(
    ad="model_16",
    veri_ad="veri_16",
    veri_tohum=0,
    tohum=0,
    jeton_ad="tam",      # varlik = jeton dizisi, ad kac kelimeyse o kadar
    ek_kip="ezber_zincir",        # iliski kelimenin kendisi, gercek ek allomorflari

    t_len=512,           # paketleme penceresi (KARAKTER).
    #                      !! model_16 karakter gormuyor ama bu deger
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

    yabanci_pay=0.20,
    cikarim_pay=0.10,
    arama_pay=0.25,      # ENT'in bu kismi ARAMA'ya; HUKUMDEN AYRIK
    ayrik_ood_pay=0.05,
    kati_pay=0.0,
    belge_pay=0.0,
    n_olcum_max=3000,
)

# Kayarsa sinav ayrisir. `test_14` bunlari izlerle birlikte denetler.
SABIT = ("veri_tohum", "tohum", "jeton_ad", "ek_kip", "t_len", "kopya",
         "tetik", "zincir_pay", "n3", "ret_pay", "ret_tut", "tam_kayip",
         "yabanci_pay", "cikarim_pay", "arama_pay", "ayrik_ood_pay", "kati_pay",
         "belge_pay", "n_olcum_max")
IZ_GRAF = "3cd9a2575e47"
IZ_OLCME = "44e6262e37f3"
