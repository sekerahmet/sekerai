# -*- coding: utf-8 -*-
"""model_a4 — IKI KATI VERI.  TEK FARK: veri_ad "veri_okul" -> "veri_okul2".

Onceden kayit: belge/onkayit/model_a4.md
Kullanici karari, 15 Eylul 2026.

SORU: `wd=0.5` calisirken OLCEK ne yapiyor?

    model_a3   wd=0.5   veri_okul    1060 varlik,  8400 olgu
    model_a4   wd=0.5   veri_okul2   2120 varlik, 16800 olgu
               ^ TEK FARK. wd, mimari, tohum, lr, adim AYNI.

BU KOL `model_a`DAN DEGIL `model_a3`TEN TUREMISTIR -- taban odur, cunku
sorulan sey "wd calisirken olcek ne yapiyor". `fark_bas` her kosuda
`model_a3 vs model_a4` basar ve tek alan gostermek zorundadir.

--------------------------------------------------------------------------
NEDEN YENIDEN ACIK BIR SORU

CLAUDE.md "Olcek `ent`i cozmuyor" diyor ve gerekcesi olculmus:
veri x4 -> A4 ENT 0.0113; phi 3.03->5.06 -> A 0.0603 -> A5 0.0090.
Ayrica `model_a2` parametre eksenini de kapatti (x2 parametre -> -0.0727).

AMA O KOSULARIN HEPSI `wd=0.1` ILE YAPILDI. Bugun olctuk ki 0.1 yanlis
taraftaydi: `model_a3` ayni mimariyle ent'i 0.4240'tan 0.5980'e cikardi
ve kisayolu 86 kat dusurdu. "Olcek ise yaramiyor" hukmu, kisayolun
sokulmedigi bir zeminde verilmisti.

Iki sonuc da anlamli:
    a4 > a3   ->  olcek YARIYOR, ama ancak kisayol sokulduktan sonra.
                  Eski "olcek cozmuyor" hukmu KOSULLU imis.
    a4 ~ a3   ->  olcek gercekten notr; eski hukum wd'den BAGIMSIZ ayakta.
    a4 < a3   ->  olcek ZARARLI ve bu wd'den bagimsiz -- arsivle tutarli.

--------------------------------------------------------------------------
DIKKAT: MODEL DE BUYUYOR, BU KACINILMAZ

Sozluk 1081 -> 2141. Gomme ve cikis basi sozlukle olcekleniyor:

    model_a3   3.427.840 parametre
    model_a4   ~3.699.000 parametre   (+%8, tamami gomme + bas)

Yani bu TAM anlamiyla "ayni model, daha cok veri" degil. Iki kat varligi
olan bir grafi daha kucuk bir gomme tablosuna sigdirmak mumkun degil;
bedel kacinilmaz ve %8'de kaliyor. Onkayitta zayiflik olarak yazili.

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_a3`ten import eder, yalniz
`AYAR`in tek alanini degistirir. `pencere_a.py` bu kolu da olcer --
veri modulunu de `ayar_t<N>.json`dan okuyor, VARSAYMIYOR.
"""
from __future__ import annotations

from model_a import egit, fark_bas                          # noqa: F401
from model_a3 import AYAR as TABAN

AYAR = TABAN.degistir(ad="model_a4", veri_ad="veri_okul2")
#      ^ SADECE FARK. wd=0.5 model_a3'ten gelir, TEKRAR YAZILMAZ.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
