# -*- coding: utf-8 -*-
"""model_a1 — DÖNGÜSÜZ kol.  TEK FARK: dongu 2 -> 1.

Onceden kayit: belge/onkayit/model_a1.md
Kullanici karari, 15 Eylul 2026.

SORU: dongu bir sey katiyor mu?

    model_a    l=4  dongu=2   3.427.840 parametre   8 katman-esdegeri
    model_a1   l=4  dongu=1   3.427.840 parametre   4 katman-esdegeri
               ^ AYNI PARAMETRE, YARI HESAP

Bu, `model_a` onkayitinin 3(c) maddesindeki "tek fark dongu" iddiasinin
GERCEK karsiligi. O iddia YANLISTI: arsivdeki duz kosuyla kiyasliyordu ve
orada `isinma` (6000 -> 2000), `betas` ((0.9,0.95) -> (0.9,0.999)) ve
`adim` (120.000 -> 20.000) da farkliydi. Burada gercekten tek alan
degisiyor ve bunu `fark_bas` her kosuda EKRANA basiyor.

2604.07822:24 birebir: "The model with R=1 is equivalent to a 4-layer
vanilla transformer." Yani bu kol, ayni eksenin DUZ kosesi.

--------------------------------------------------------------------------
BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ

`model_a`dan IMPORT eder ve `AYAR`in tek alanini degistirir. `Model`,
`veri_kur`, `egit`, olcme -- hepsi ayni koddan. `pencere_a.py` de bu kolu
olcer: o `model_a`yi import ediyor ve mimariyi `ayar_t<N>.json`dan
okuyor, VARSAYMIYOR.
"""
from __future__ import annotations

from model_a import AYAR as TABAN, egit, fark_bas           # noqa: F401

AYAR = TABAN.degistir(ad="model_a1", dongu=1)
#      ^ SADECE FARK. Mimari/veri/tohum/lr/wd/adim TEKRAR YAZILMAZ.

if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
