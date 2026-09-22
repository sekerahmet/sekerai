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

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler   <-- BU DOSYA
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  train_16    egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
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
    #                      Birim akisinda ARTIK PAKETLEME YOK; akis
    #                      satir satir kuruluyor (`birim_16.belgeler`).
    #                      Bu deger yalniz IKI yerde kaldi:
    #                        - hazirla_16.kur()  KARAKTER korpusu
    #                        - korpus_16.sayfalar() TETIKLENMIS dalda
    #                          belgeyi boluyor -- hala KARAKTER sayiyor,
    #                          acik kalem.
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

# =====================================================================
# BIRIM PENCERESI.  VERI ayari, mimari DEGIL.
# `t_len` KARAKTER paketleme penceresi; bunlar BIRIM akisininki.
# model_14'un mimari sabitlerini cikarirken bu ikisini de atmistim --
# yanlis siniflandirma, geri konuldu.
# =====================================================================
PENCERE = 32         # bir egitim penceresi kac BIRIM.
#                      OLCULDU: en uzun soru+cevap parcasi TAM 32 birim.
#                      Daha kisa pencere o cifti boler -- model soruyu
#                      gorur, cevabi gormez.  24'te 3.485 cift boluyordu
#                      ve HEPSI soru cumlesiydi.  32'de 0.
#                        L  kesilen parca (hepsi sorulu)
#                       24           3.485
#                       28              21
#                       31               3
#                       32               0
#                      32'nin ustu kapsama kazandirmiyor, yalniz baglam;
#                      ve bu mimaride pencere = OZYINELEME DERINLIGI.
ATLA = 4             # kayan pencerenin adimi.  1 her konumdan bir
                     #   pencere demek; buyutmek ortusmeyi azaltir ama
                     #   ayni olguyu daha az kez gosterir.

# Kayarsa sinav ayrisir. `test_14` bunlari izlerle birlikte denetler.
SABIT = ("veri_tohum", "tohum", "jeton_ad", "ek_kip", "t_len", "kopya",
         "tetik", "zincir_pay", "n3", "ret_pay", "ret_tut", "tam_kayip",
         "yabanci_pay", "cikarim_pay", "arama_pay", "ayrik_ood_pay", "kati_pay",
         "belge_pay", "n_olcum_max")
IZ_GRAF = "3cd9a2575e47"
IZ_OLCME = "44e6262e37f3"
