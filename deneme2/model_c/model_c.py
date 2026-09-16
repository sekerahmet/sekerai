# -*- coding: utf-8 -*-
"""model_c — ÇOK BAŞLI Φ.  Darboğaz jeton düzeyinden VARLIK düzeyine.

Taban `model_b6`, TEK FARK: `dar_kafa` 1 -> 3.
Önceden kayıt: `belge/onkayit/model_c.md`.

--------------------------------------------------------------------------
NEDEN -- OLCULDU (16 Eylul, `asama1`), tahmin degil

    ASAMA-1: ilk donguden sonra kopru varlik cozulebiliyor mu?

                     kopru yuva0 (EN IYI poz)   asama-2 cevap
      model_b1 ood   0.9558  (poz 3)            0.8451
      model_b6 ood   0.0442  (poz 5)            0.0354
      model_b6 seen  0.0398  (poz 5)            0.9867
                     ^ 11 pozisyonun HICBIRINDE cozulemiyor

`model_b6`da Phi'nin enjekte edecek temiz bir gommesi YOK. Uc olcum bunu
kapasite aciklamasindan ayiriyor:

    1. f_theta AYNI        b1 ve b6 bloklari birebir 3.147.776 parametre
    2. BILGI VAR           one = 0.9717 (e --r1--> b DOGRUDAN biliniyor)
    3. Phi POZISYON-YEREL  tek pozisyonda tek tek jeton gommelerinin
                           disbukey birlesimi -- uc jetonlu varligin
                           TEK gomme satiri YOK

(3) bir kusur degil, TANIMIN SONUCU. Bu kol tanimi genelliyor.

--------------------------------------------------------------------------
NE DEGISIYOR

    model_b6   Phi(h)   = softmax(W nf(h) / tau) @ W
    model_c    Phi_m(h) = (1/m) SUM_j softmax(W A_j nf(h) / tau) @ W

Her bas bir YUVA cozer; enjekte edilen sey cozulen parcalarin temiz
gommelerinin ortalamasi -- "varligin temiz gommesi"nin cok jetonlu
karsiligi.

YENI BIR MIMARI DEGIL: `model_b`de OLCULDU ki Phi zaten dikkattir
(Q=nf(h), K=V=W, olcek 1/tau; sayisal fark 4.7e-07). Tek basli dikkati
COK BASLI yapmak standart adim. Burada bas sayisi `yuva`ya baglaniyor.

    ek parametre   m * d^2 = 3 * 256^2 = 196.608   (toplamin ~%6'si)
                   A_0 da OGRENILIR; birim matris yalniz BASLANGIC.
    A_0 = I        0. adimda Phi_m'in 0. basi = model_b6'nin Phi'si
    dar_kafa = 1   HICBIR ek parametre YOK, ModelB ile BIT AYNI

Son kosul `test_sabit_c.py`de her kosuda sinaniyor.

--------------------------------------------------------------------------
SIMETRI NEDEN KIRILIYOR

Butun A_j birim matris olsaydi butun baslar AYNI jetonu cozer, gradyanlar
da AYNI olurdu -- baslar sonsuza kadar ayni kalirdi. O yuzden A_0 = I
TAM, digerleri I + kucuk gurultu. Baslangicta Phi_m ~= Phi (kol b6'nin
durdugu yerden baslar), ama baslar ayrisabilir.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_C = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_C), "model_a"),
           os.path.join(os.path.dirname(_C), "model_b"),
           os.path.dirname(_C)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_a as M                                           # noqa: E402
import model_b6                                               # noqa: E402
from model_b import ModelB                                    # noqa: E402


# A_j'nin baslangic gurultusu. d'ye gore olceklenir ki birim matrisin
# yaninda kucuk kalsin: satir normu ~1 iken bozulma ~0.02.
_GURULTU = 0.02


class ModelC(ModelB):
    """`dar_kafa=1` iken ModelB ile BIT DUZEYINDE AYNI -- ek parametre
    bile kurulmaz. `test_sabit_c.py` bunu her kosuda dogruluyor."""

    def __init__(self, ayar: M.Ayar, vocab: int):
        super().__init__(ayar, vocab)
        m = int(ayar.dar_kafa)
        assert m >= 1, f"dar_kafa >= 1 olmali: {m}"
        self.cok_bas = self.dar_acik and m > 1
        if not self.cok_bas:
            return                       # HICBIR ek parametre YOK
        assert not (ayar.dar_sert or ayar.dar_sdpa), (
            "cok basli Phi SADE yolla kosulur: sert yolda softmax YOK, "
            "SDPA yolu tek sorgu icin yazildi.")
        d = ayar.d
        A = torch.eye(d).repeat(m, 1, 1)
        # A_0 = I TAM. Digerleri I + kucuk gurultu -> SIMETRI KIRILIR.
        g = torch.Generator().manual_seed(ayar.tohum * 1000 + 7)
        A[1:] += torch.randn(m - 1, d, d, generator=g) * (_GURULTU / d ** 0.5)
        self.kafa = nn.Parameter(A)      # (m, d, d)

    def _phi(self, h):
        if not self.cok_bas:
            return super()._phi(h)
        # `head` bias'siz ve gommeye BAGLI -> head(x) == x @ emb.weight.T
        q = self.nf(h)                                  # (..., d)
        W = self.emb.weight                             # (V, d)
        t = self.ayar.dar_tau
        toplam = None
        for j in range(self.kafa.shape[0]):
            p = torch.softmax((q @ self.kafa[j].T) @ W.T / t, dim=-1)
            g = p @ W
            toplam = g if toplam is None else toplam + g
        return toplam / self.kafa.shape[0]


TABAN = model_b6.AYAR
AYAR = TABAN.degistir(ad="model_c", dar_kafa=3)
#      ^ TEK FARK. veri_okul2, jeton_ad="tam", dar_alfa=0.5, ort_bas=10000,
#        wd=0.5, l=4, d=256, dongu=2, ood_pay=0.05 -- hepsi model_b6'dan.
#        dar_kafa=3 cunku `yuva` 3: her bas bir yuva cozsun.

# `kos.py` modulden UC sey istiyor: AYAR, egit, fark_bas.
fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    """model_a.egit, ama modeli ModelC kurar. `model_b`deki desenin
    AYNISI: model_a'nin `egit`i BURADA GOLGELENIR, yoksa cok basli Phi
    OLMAYAN bir model kosar ve kunye yine 'model_c' derdi."""
    return M.egit(ayar or AYAR, model_kur=ModelC, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
