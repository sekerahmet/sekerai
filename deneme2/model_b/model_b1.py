# -*- coding: utf-8 -*-
"""model_b1 — DARBOGAZ, BIZIM OKUL GRAFINDA.  TEK FARK: veri_wang -> veri_okul2.

Onceden kayit: belge/onkayit/model_b1.md
Kullanici karari 16 Eylul: "bu bizim yeni modelimiz olacak, model_b1
okul verisiyle test etmek gerekiyor."

    model_b    veri_wang    ood_pay=0.05  dar_alfa=0.5
    model_b1   veri_okul2   ood_pay=0.05  dar_alfa=0.5
               ^ TEK FARK

--------------------------------------------------------------------------
BU KOLUN DEGERI: model_a8 ILE ARASINDAKI TEK FARK DARBOGAZ

    model_a8   veri_okul2  ood_pay=0.05  dar_alfa=0.0
    model_b1   veri_okul2  ood_pay=0.05  dar_alfa=0.5
                                         ^ TEK FARK

Ayni graf, ayni bolme, ayni tohum, ayni olcme izi (57cf60a5e9af), ayni
wd, ayni ortalama. Kiyas DOGRUDAN.

NEDEN: model_b darbogazin WANG GRAFINDA ne yaptigini olctu

    model_a9 (darbogaz YOK)  @60000   ent 0.5060  ood 0.0013  comp 0.9980
    model_b  (dar_alfa=0.5)  @28000   ent 0.9933  ood 0.0000  comp 0.9997

`ent` COZULDU ve beste bir butceyle; `ood` KIPIRDAMADI.
Soru: ayni sey bizim grafimizda da olur mu? Orada `ent` ZATEN yuksekti
(0.8740) ve `ood` sifir DEGILDI (0.3584).

KIYAS TABANI -- model_a8, 52000-60000:
    one 1.0000  seen 1.0000  comp 0.9770
    ood 0.3584  ent 0.8740  ent_yok 0.5895  ent_ksy 0.0100

EN ZAYIF YER: OOD havuzu bu grafta 226 (Wang'da 781) -> binom SE 0.032,
IKI KAT gurultulu. Karar bandi bu yuzden genis (onkayit §4).

ASIL IDDIA VERIMLILIK: model_b, model_a9'un 60.000'deki ent'ini (0.5060)
8.000 adimda gecti -> 7,5x az adim, adim maliyeti +%5. Bu kol o iddianin
IKINCI GRAFTA tutup tutmadigini olcer.

BU DOSYA MIMARIYI YENIDEN TANIMLAMAZ: `model_b`den import eder.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_a as M                                          # noqa: E402
from model_b import ModelB, AYAR as TABAN                     # noqa: E402

fark_bas = M.fark_bas

AYAR = TABAN.degistir(ad="model_b1", veri_ad="veri_okul2")
#      ^ SADECE FARK. dar_alfa=0.5, ood_pay=0.05, wd=0.5, ort_bas=10000
#        hepsi model_b'den gelir.


def egit(ayar=None, **kw):
    """model_a.egit, ama modeli ModelB kurar -- model_b ile AYNI sinif."""
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    fark_bas(TABAN, AYAR)
    egit(AYAR)
