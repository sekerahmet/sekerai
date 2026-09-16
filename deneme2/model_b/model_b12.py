# -*- coding: utf-8 -*-
"""model_b12 — DIL MODELI OPTIMIZASYONU.  Taban model_b11, IKI DUGME.

Onceden kayit: belge/onkayit/model_b12.md

    model_b11   wd=0.5   sabit_lr=True    grokking rejimi
    model_b12   wd=0.1   sabit_lr=False   standart LM rejimi
                ^^^^^^^^^^^^^^^^^^^^^^^   IKI DUGME BIRDEN

--------------------------------------------------------------------------
!! BU BIR PAKET KOL -- ATFETME YAPILAMAZ

Kullanici karari, 16 Eylul: "LR ve wd = 0,1 ayni anda degissin."
Itiraz ettim (iki dugme birden doner, hangisinden geldigi ayrilamaz);
kullanici karari tekrarladi ve GEREKCESI VAR: `wd=0.5` + sabit LR
BIRLIKTE bir GROKKING REJIMI. Yalniz birini cevirmek iki rejimin
ARASINDA kalmak olurdu.

Sonuc OLUMLUYSA bisect kollari kosulur:
    model_b12a   yalniz wd=0.1        (sabit_lr=True kalir)
    model_b12b   yalniz sabit_lr=False (wd=0.5 kalir)

--------------------------------------------------------------------------
NEDEN -- referanslardan OLCULDU (config dosyalarindan, ozetten DEGIL)

    weight_decay
      nanoGPT train.py:60        1e-1     (GPT-2 tarifi)
      Pythia-70m.yml:62          0.1
      Pythia-160m.yml:62         0.1
      Qwen2.5 SFT                0.1      "we apply a weight decay of 0.1"
      Qwen2.5 on-egitim          YAYIMLANMAMIS (scaling laws)
      BIZ (b6..b11)              0.5      <- 5 KAT

    LR programi
      nanoGPT                    cosine -> lr/10
      Pythia                     "lr-decay-style": "cosine", min_lr lr/10
      Qwen2.5 SFT                7e-6 -> 7e-7  (kademeli AZALAN)
      BIZ (b6..b11)              SABIT         <- "grokking icin LR SONMEMELI"

--------------------------------------------------------------------------
BILINEN SAPMA -- cosine SIFIRA iniyor

model_a.py:1622'deki program 20.000'de lr'yi TAM SIFIR yapiyor;
referanslarin ucu de `lr/10`da duruyor. Duzeltmek `min_lr` diye YENI
BIR ALAN gerektirirdi -- UCUNCU dugme olurdu, eklenmedi.

    adim 10000  lr 0.000587      adim 19000  lr 0.000008
    adim 15000  lr 0.000179      adim 20000  lr 0.000000

ETKILESIM: `ort_bas=10000`, yani agirlik ortalamasi LR zaten sonerken
basliyor. Son pencerede agirliklar neredeyse DONMUS olacak.

--------------------------------------------------------------------------
MIMARI: bir satir bile degismedi. Ek parametre YOK. `model_a.py`ye
DOKUNULMADI -- `wd` ve `sabit_lr` ikisi de ZATEN Ayar alani, cosine
ZATEN kurulu. Yani bu kol kilit kirmiyor.
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
from model_b11 import AYAR as TABAN                          # noqa: E402

AYAR = TABAN.degistir(ad="model_b12", wd=0.1, sabit_lr=False)
#      ^ IKI FARK, bilerek. belge_pay=0.5, tam_kayip=True, dar_kapi=True
#        model_b11'den DEVRALINIR; jeton_ad="tam", dar_kafa=1,
#        kopru_kayip=0.0, ood_pay=0.05, ort_bas=10000 de oyle.

fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
