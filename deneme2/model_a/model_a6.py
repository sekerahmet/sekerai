# -*- coding: utf-8 -*-
"""model_a6 — LOOKAHEAD'in `k`si.  TEK FARK: ort_her 2000 -> 200.

Onceden kayit: belge/onkayit/model_a6.md

    model_a5   ort_bas=10000  ort_her=2000  ort_alfa=0.5
    model_a6   ort_bas=10000  ort_her= 200  ort_alfa=0.5
                              ^ TEK FARK

TABAN `model_a5` -- `wd=0.5`, `veri_okul2`, `ort_bas`, `ort_alfa` hepsi
oradan gelir.

--------------------------------------------------------------------------
NEDEN

`model_a5` geri beslemeli ortalamayi olctu ve AYIRT EDILEMEDI cikti
(52000-60000: 0.8643 vs model_a4 0.8820). Ama o kosu k=2000 ile
yapildi ve KENDI ONKAYITI bunu zayiflik olarak KOSUDAN ONCE yazmisti:

    "k=2000 literaturun bolgesi degil. Lookahead k=5..10'da taranmis.
     Ayiran kosu: ort_her kucultulur (orn. 200). Hazir duruyor."
                                        -- onkayit/model_a5.md §8.1

Ayrilan soru: model_a5'in sonucu FIKRIN mi sonucuydu, `k`NIN mi?

--------------------------------------------------------------------------
`k`yi 10 KAT KUCULTMEK

alfa=0.5 ile her k adimda ortalama, phi'yi theta'nin ~2k adimlik ustel
ortalamasi yapar:

    model_a5  k=2000   ufuk ~4000 adim    10.000-60.000 arasi  25 olay
    model_a6  k= 200   ufuk ~ 400 adim    10.000-60.000 arasi 250 olay

IKI ZIT ETKI AYNI ANDA:
    UFUK KISALIYOR        -> daha AZ duzeltme, ham yorungeye yakin
    GERI CEKME SIKLASIYOR -> 10 KAT daha sik yarim yoldan geri cekme

Hangisi baskin, OLCULMEDI. Bu kosu onu olcuyor.

RISK: sonumleme riski model_a5'tekinden BUYUK -- orada 25 kez geri
cekildi, burada 250 kez. Sonumleme cikarsa hukum `k` aleyhine yazilir,
FIKIR aleyhine degil; ters yonde ayiran kol `ort_alfa` (0.5 -> 0.1).

CAPA YOK, ve bu bilerek: model_a5'te k=2000 secimi 30 anlik goruntuden
olculmus bir capaya dayaniyordu (yorunge 2000'lik olcekte ilerlediginden
cok DOLANIYOR). 200 adimlik olcekte ayni olcum YAPILAMAZ -- anlik
goruntuler 2000 adimda bir aliniyor. Bu kol capasiz kosuyor ve
`k`=200'un DOGRU k oldugunu IDDIA ETMIYOR; yalniz `k`nin ONEMLI OLUP
OLMADIGINI soruyor.

--------------------------------------------------------------------------
KOD DEGISMEDI. `ort_her` alani model_a5 icin zaten eklenmisti ve
ortalama blogu EGITIM ADIMI seviyesinde, olcum dalinin DISINDA:

    model_a.py:1144   _oh = ayar.ort_her or ayar.olc_her
                      if ayar.ort_bas and adim >= ayar.ort_bas \
                         and adim % _oh == 0:

`2000 % 200 == 0` oldugu icin olcum adimlarinda "once ortala, sonra olc"
sirasi bozulmuyor.

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a5`ten import eder, yalniz
`AYAR`in tek alanini degistirir.
"""
from __future__ import annotations

from model_a import egit, fark_bas                           # noqa: F401
from model_a5 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a6", ort_her=200)
#      ^ SADECE FARK. ort_bas=10000, ort_alfa=0.5, wd=0.5 ve veri_okul2
#        model_a5'ten gelir.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
