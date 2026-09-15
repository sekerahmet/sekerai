# -*- coding: utf-8 -*-
"""model_a9 — WANG'IN GRAFINDA paylasimli kol.  TEK FARK: veri_wang.

Onceden kayit: belge/onkayit/model_a9_a10.md  (CIFT KOL: a9 + a10)

    model_a8   veri_okul2   ood_pay=0.05
    model_a9   veri_wang    ood_pay=0.05
               ^ TEK FARK

NEDEN: KENDI HAKEMLIGIMIZIN R3'U
--------------------------------
`model_a8` Wang'in bolme tanimini birebir kurdu ve 52000-60000'de
`ood` 0.3584 olctu. Ama o sayi Wang'in %0'inin karsisina KONAMAZ:

    OOD zincirinin (r1,r2) ciftini model egitimde kac kez gordu
      veri_okul2   ortalama 879   (120 farkli cift)
      Wang         ~8,6           (40.000 olasi cift)
      ORAN         103 KAT

Artı `veri_okul` TIPLI (`vali` yalniz SEHIR'den cikar); tip onseli sansi
0.00047 -> 0.0025, BES KAT yukseltiyor. `veri_wang` iki farki da kapatir:

    cift basina zincir  10,0  (Wang ~8,6)      tip YOK -> sans 0,0010
    phi (Wang tanimi)  16,14  (Ek E.2: 12,6)   OOD havuzu 781 (SE 0,017)

ASIL DENEY TEK KOL DEGIL, CIFT
------------------------------
    model_a9    l=4  dongu=2   PAYLASIMLI    8 katman-esdegeri
    model_a10   l=8  dongu=1   PAYLASIMSIZ   8 katman-esdegeri

HESAP ESIT. `dongu=1, l=4` olsaydi yari hesap olurdu ve "paylasim mi,
hesap mi" AYRILAMAZDI. Wang'in Ek E.2'si de esit derinlikte kiyasliyor.

Karar kurali onkayit §4'te, BAGLAYICI. Ozeti:
    a10 < 0.05 ve a9 >= 0.20  ->  PAYLASIM ACIYOR
    ikisi de < 0.05           ->  ikisi de yapamiyor
    ikisi de >= 0.20          ->  paylasim GEREKLI DEGIL (makaleye karsi bulgu)

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a8`ten import eder.
"""
from __future__ import annotations

from model_a import egit, fark_bas                           # noqa: F401
from model_a8 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a9", veri_ad="veri_wang")
#      ^ SADECE FARK. ood_pay=0.05, wd=0.5, ort_bas=10000, ort_alfa=0.5
#        hepsi model_a8/model_a5'ten gelir.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
