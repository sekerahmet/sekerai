# -*- coding: utf-8 -*-
"""model_b11 — BELGE.  Taban model_b10, TEK FARK: belge_pay.

Onceden kayit: belge/onkayit/model_b11.md

    model_b10   belge_pay=0.0    satir basina TEK olgu
    model_b11   belge_pay=0.5    havuzun YARISI BELGE: zincirlenen IKI
                                 atomik olgu AYNI DIZIDE
                                 ^ TEK FARK

--------------------------------------------------------------------------
NE DEGISIYOR -- gercek bir satir

    simdi   [S1] Ayse Yilmaz <YOK> cocuk  ? Fatma Yilmaz <YOK> <EOS>
            [S1] Fatma Yilmaz <YOK> kardes ? Emre Yilmaz <YOK> <EOS>
            ^ IKI AYRI SATIR, asla ayni baglamda gorulmuyor

    b11     [S1] Ayse Yilmaz <YOK> cocuk ? Fatma Yilmaz <YOK> <EOS>
            [S1] Fatma Yilmaz <YOK> kardes ? Emre Yilmaz <YOK> <EOS>
            ------------------- TEK DIZI, 20 jeton -------------------

YENI OLGU YOK. Iki olgu da `one` icinde ZATEN ayri ayri var. Eklenen
tek sey BITISIKLIK.

--------------------------------------------------------------------------
NEDEN -- olculdu

Gercek metinde "Ayse'nin cocugu Fatma. Fatma'nin kardesi Emre." ayni
paragraftadir; LLM 2-hop'u KURALDAN degil BIRLIKTE GORMEKTEN ogrenir.
Bizim veride:

    koprunun AYIRT EDICI jetonu 2-hop satirlarinda geciyor: %3,28
    (ve o da isim cakismasi -- butun jetonlari geciyor: %1,43)
    t_len 11 -> iki olgu yan yana SIGMIYOR

Yani "Ahmet->Meryem" ile "Meryem->Mehmet" ASLA ayni baglamda degil.

--------------------------------------------------------------------------
BU KURAL DAYATMASI DEGIL

    kopru_kayip (b8)  "kopruyu buraya koy"        <- BIZIM kuralimiz
    belge_pay   (b11) "iki olgu ayni paragrafta"  <- GERCEK METNIN
                                                     zaten sahip oldugu

--------------------------------------------------------------------------
SIZINTI -- denetleniyor

Belgeler YALNIZ `tr2`den kurulur. comp/ent/ent_yok/ood/ent_arama
zincirlerinden kurulsaydi o zincirin koprusu BAGLAMA yazilmis olurdu,
yani cevabi elden vermis olurduk. `egitim_havuzu` her kosuda assert
ediyor ve "sizinti denetimi GECTI" basiyor.

--------------------------------------------------------------------------
BEDELI -- ve neden kacinilmaz

t_len 11 -> 20. Butun satirlar 20'ye dolgulaniyor, attention T^2:
121 -> 400, yani ~3,3 kat hesap. 20.000 adim ~14 dk -> ~45 dk.

Kacinilmaz cunku iki olgu 11 jetona SIGMIYOR (2-hop satiri zaten
11/11 dolu, bos yer 0).

MIMARI: Model bir satir bile degismedi. TEK ek parametre pozisyon
tablosunda: 11x256 -> 20x256, yani +2.304.
"""
from __future__ import annotations

import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_A = os.path.join(os.path.dirname(_B), "model_a")
for _p in (_A, _B):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                          # noqa: E402
assert hasattr(M, "egit"), (
    f"model_a MODUL degil PAKET olarak yuklendi: {getattr(M,'__file__',None)}")

from model_b import ModelB                                   # noqa: E402
from model_b10 import AYAR as TABAN                          # noqa: E402

AYAR = TABAN.degistir(ad="model_b11", belge_pay=0.5)
#      ^ SADECE FARK. tam_kayip=True ve dar_kapi=True model_b10'dan,
#        jeton_ad="tam" / dar_kafa=1 / kopru_kayip=0.0 / ood_pay=0.05 /
#        wd=0.5 / ort_bas=10000 devralinir.
#        NOT: belge_pay, tam_kayip GEREKTIRIR (egitim_havuzu assert eder):
#        kayip yalniz cevap yuvalarinda olsaydi belgenin ORTASI --
#        yani kopru -- hic ogrenilmezdi.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
