# -*- coding: utf-8 -*-
"""model_b9 — LEARNABLE GATE, TEK BASINA.  Taban model_b6, TEK FARK.

Onceden kayit: belge/onkayit/model_b9.md

    model_b6   jeton_ad="tam"  dar_kafa=1  dar_kapi=False   FIXED gate
    model_b9   jeton_ad="tam"  dar_kafa=1  dar_kapi=True    LEARNABLE
                                           ^ TEK FARK

--------------------------------------------------------------------------
NE DEGISIYOR

    fixed      h = h + 0.5 * RMSNorm(Phi(h))       her pozisyona AYNI
    learnable  h = h + a(h) * RMSNorm(Phi(h))      POZISYON BASINA
               a(h) = sigmoid(<w_a, Phi(h)> + b_a)     sekil (B, T, 1)

`d+1` = 257 ek parametre (Denk. 6). Model, enjeksiyonun ne kadarini
gecireceğine HER POZISYONDA kendisi karar verir.

w_a = b_a = 0 ile baslar -> sigmoid(0) = 0.5 = model_b6'nin dar_alfa'si.
Yani 0. adimda model_b6 ile BIT AYNI; fark yalniz OGRENME ile acilir.

--------------------------------------------------------------------------
NEDEN TEK BASINA

`model_c` uc head ekliyor. Ikisini AYNI kolda degistirirsek sonuc
cikarsa hangisinden geldigini AYIRAMAYIZ. Bu kol gate'i yalniz basina
olcer; ikisi birlikte ANCAK ikisi de tek tek sinandiktan sonra denenir.

    model_b6  ->  model_b9   TEK FARK: gate
    model_b6  ->  model_c    TEK FARK: head sayisi
    (ikisi birlikte: SONRA, ve ancak gerekiyorsa)

--------------------------------------------------------------------------
NEDEN -- iki gerekce

1. MAKALE DIL AYARINDA BUNU KULLANIYOR. arXiv 2607.00341, birebir:
     5.1 sembolik  "we use a fixed gate alpha* = 1 and temperature tau = 1"
     5.3 uc-hop    "We use a fixed gating alpha = 1 for DiscoLoop"
     5.2 DIL       "We use a learnable gate alpha ... on both
                    synthetic-language datasets"
   Makale saf sembolik token'lardan uzaklastigi anda fixed gate'i
   birakiyor. Bizim veri onlarin dil ayarindan DAHA ILERIDE (entity 2-3
   token) ve alti kolda da fixed gate kostuk.

2. COK TOKEN'LI REJIMDE ENJEKSIYON HER POZISYONDA AYNI OLMAMALI.
   `Ozlem | Yilmaz | <YOK>` dizisinin ORTASINDAKI pozisyonda "temiz
   entity embedding'i enjekte et" demek anlamsiz olabilir -- orada tek
   bir entity yok, bir PARCA var. Fixed gate bunu kapatamaz.

--------------------------------------------------------------------------
KAYDA GECEN KUSUR: bu kol GECIKTI

Kullanici 16 Eylul'de "kesinlikle test edilmesi gereken bu" dedi; ben
"siradaki model_b8 bu olsun" diye yazdim, sonra kopru teshisini kurarken
model_b8 adini ONA verdim ve bu kol KAYBOLDU. Kayit hatasi, gerekce
degil. Kullanici hatirlatti.

MIMARI: ModelB bir satir bile degismedi -- `dar_kapi` zaten forward'da
var, hic kosulmamisti.
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
from model_b6 import AYAR as TABAN                           # noqa: E402

AYAR = TABAN.degistir(ad="model_b9", dar_kapi=True)
#      ^ SADECE FARK. jeton_ad="tam", dar_kafa=1, dar_alfa=0.5,
#        ood_pay=0.05, wd=0.5, ort_bas=10000 hepsi model_b6'dan.
#        NOT: dar_kapi=True iken `dar_alfa` FORWARD'DA KULLANILMIYOR --
#        alan duruyor ki fark listesi TEK dugme gostersin.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
