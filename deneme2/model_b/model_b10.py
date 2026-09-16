# -*- coding: utf-8 -*-
"""model_b10 — DIL MODELI KAYBI.  Taban model_b6, TEK FARK: tam_kayip.

Onceden kayit: belge/onkayit/model_b10.md

    model_b6    tam_kayip=False   kayip YALNIZ cevap yuvalarinda
    model_b10   tam_kayip=True    kayip HER pozisyonda (next-token)
                                  ^ TEK FARK

--------------------------------------------------------------------------
NEDEN -- kullanici, 16 Eylul

"biz veriyi duzeltmeliyiz sanki. normal dil egitimindeki veri mimarisini
kullanmaliyiz."

Ve hakliydi. Kodun soyledigi sey su:

    lg_tam = model(xb)          # BUTUN pozisyonlarin logit'i
    lg = lg_tam[_ar, pb]        # sadece CEVAP yuvalari alindi
    kayip = cross_entropy(lg, tb)     # kalan %70 ATILDI

Yani alti koldur egittigimiz sey bir SORU-CEVAP basligi, dil modeli
DEGIL. Model 11 pozisyonun hepsi icin tahmin uretiyor, biz 3'unu
aliyoruz.

--------------------------------------------------------------------------
BUNUN OLCULEN SONUCU

Model 'Ahmet' -> 'Yilmaz' gecisini HIC tahmin etmiyor. Varligin
jetonlarini BIRIM olarak baglamasi icin tek baski, cevabi uretirken.

Ve varlik TEK BIR PARCASINDAN taninmiyor (olculdu, veri tarafi):

    yuva 0 (ad)      419 farkli jeton   varligi tek basina belirler:   7,5%
    yuva 1 (soyad)    29 farkli jeton                                  0,3%
    yuva 0 + 1      2120 farkli cift                                 100,0%

    'Fatih' -> 17 ayri varlik    'Yilmaz' -> 100 ayri varlik

Kimlik CIFTTE. Ve `model_b8` tam olarak o cifti (kopru_hedefi = yuva
0+1) zorla temsile koyunca comp 0.0657 -> 0.9747 oldu.

--------------------------------------------------------------------------
BU BIR KURAL DAYATMASI DEGIL

`kopru_kayip` (model_b8) modele "kopruyu buraya koy" diyor -- bizim
analizimizden cikan bir kurali ELLE veriyor, ve zaten "teshis, mimari
onerisi degil" diye kayitli. `ident_kip` de ayni aileden olurdu.

tam_kayip BASKA BIR SEY: modele hicbir kural verilmiyor, yalnizca
GERCEK METNIN ZATEN SAHIP OLDUGU bir ozellik veriliyor -- her token
tahmin edilir. Kurallari model kendi cikarir.

--------------------------------------------------------------------------
MESRU MU -- EVET, model NEDENSEL MASKELI

Cevap jetonlari dizinin ICINDE (`[Q2] e1 e2 r1 r2 ? a1 a2 <EOS>`).
Maske olmasaydi next-token kaybi KOPYALAMAYLA cozulurdu. Olculdu:
`F.scaled_dot_product_attention(..., is_causal=True)` -- pozisyon t
yalnizca <= t'yi goruyor. Mevcut cevap kaybi da zaten next-token: QM
pozisyonunda a1, a1 pozisyonunda a2 tahmin ediliyor. Yani tam_kayip
ayni sozlesmeyi BUTUN pozisyonlara yayiyor, yeni bir sozlesme
getirmiyor.

--------------------------------------------------------------------------
BILINEN CONFOUND -- SEYRELME, ve nasil okunuyor

Cevap yuvalari artik ~9 kayip teriminin 3'u. Ayni lr ve ayni adimda
cevap gorevine dusen gradyan ~3 kat SEYRELIYOR. Bu yuzden:

    `kayip`      = optimize edilen DIL MODELI kaybi
    `kayip_ana`  = yalniz cevap yuvalari -- model_b6'nin `kayip`
                   sutunuyla AYNI SEY, tek kiyaslanabilir sayi

Ikisi de egriye yaziliyor. (model_b8'de `kayip` sutunu kirlenmis ve b6
ile kiyaslanamaz hale gelmisti; ayni hataya dusmuyoruz.)

MIMARI: Model bir satir bile degismedi. Ek parametre YOK. Hiz: logit'ler
zaten hesaplaniyordu, degisen yalniz kac tanesinin kullanildigi.
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

AYAR = TABAN.degistir(ad="model_b10", tam_kayip=True)
#      ^ SADECE FARK. jeton_ad="tam", dar_kafa=1, dar_alfa=0.5,
#        dar_kapi=False, kopru_kayip=0.0, ood_pay=0.05, wd=0.5,
#        ort_bas=10000 hepsi model_b6'dan DEVRALINIR.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
