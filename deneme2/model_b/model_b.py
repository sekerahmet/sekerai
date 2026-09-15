# -*- coding: utf-8 -*-
"""model_b — YENI AILE.  Dongu turlari arasinda GOMME DARBOGAZI.

Onceden kayit: belge/onkayit/model_b.md
Kullanici karari 16 Eylul: "daha iyi bir model tasarlamaya calisiyoruz,
ama bu artik model_b olsun."

    model_a9   veri_wang  ood_pay=0.05  dar_alfa=0.0
    model_b    veri_wang  ood_pay=0.05  dar_alfa=0.5     <- TEK FARK

--------------------------------------------------------------------------
NEREDEN GELDI -- KENDI OLCUMLERIMIZ (model_a9, 60.000 adim)

    seen 1.0000  comp 1.0000  ent 0.6157  ood 0.0026   (sans 0.0010)

    kopru 1. tur sonunda top-1:  ood 0.3137   comp 1.0000
    -> KOPRU %31 DOGRU GETIRILIYOR ama cevaba donusmuyor.
       Kopru dogruyken bile 2. adim %0,8 basariyor.

    gizli durum normu 6.116   varlik gommesi normu 0.477   -> 13 KAT

Yani 2. tur, 1. turun urettigi DAGINIK vektoru tuketiyor; o vektor gomme
uzayiyla hizali degil. Egitim onu ID kenarlari icin hizalamis, OOD icin
HIC hizalamamis. 1-hop ise HER kenarda calisiyor (one 0.9727, OOD dahil)
-- cunku orada girdi zaten bir GOMME.

DARBOGAZ, 2. turun girdisini de gomme bicimine cevirir. Boylece 2. adim
ayri bir yetenek olmaktan cikip 1. adimin BICIMINE indirgenir.

--------------------------------------------------------------------------
FIKIR BIZIM DEGIL -- DiscoLoop (arXiv 2607.00341, Princeton)

Makale indirilip okundu. Teshis birebir ayni:
    "after the first loop, the bridge entity is often decodable from the
     continuous hidden vector, yet this vector remains noisy and
     geometrically misaligned with the clean discrete embedding of the
     same entity that the next loop would ideally consume."

Ve ayni makale BIZIM DENEYIMIZIN de eszamanli yapildigini yaziyor:
    "Concurrent to our work, Kohli et al. (2026) ... Their test split
     holds out 5% of atomic facts from training compositions within a
     single knowledge graph."
-> model_a8/model_a9'un tanimi. BAGIMSIZ VARDIK, ILK DEGILIZ.

BU KOL BIR TEKRARDIR. Degeri: kendi grafimizda, kendi phi'mizde calisip
calismadigini olcmek. Rejim farki buyuk (onkayit §4):
    DiscoLoop  phi 1,0   3.000 epoch   ID 71,1%   OOD 8,3%
    model_a9   phi 12,9    110 epoch   ID 99,9%   OOD 0,13%

--------------------------------------------------------------------------
MIMARI -- DiscoLoop Denklem 4-6, birebir

    H~(0) = W[x] + pos
    H(k+1)  = f_theta( H~(k) )
    H~(k+1) = H(k+1) + alfa(k) * RMSNorm( Phi(H(k+1)) )
    Phi(h)  = softmax(W h / tau) @ W
    alfa(K-1) = 0   -> son turun ciktisi DOGRUDAN kafaya gider

`Phi` HER KONUMDA ve HER TURDA uygulanir (makalenin §3.2'deki egitimsiz
mudahalesi tek konumdaydi; §4 bunu mimariye yukseltiyor).

EGITIMSIZ TAVAN ONCEDEN OLCULDU (model_a9'un 60.000 agirligina, ayni
bicimde mudahale):  ood 0.0013 -> 0.0883 (alfa=0.5), comp BOZULMADAN.
Yon dogru; egitimli hali ancak kosu soyler.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import torch
import torch.nn as nn

import model_a as M                                          # noqa: E402
from model_a9 import AYAR as TABAN                           # noqa: E402


class ModelB(M.Model):
    """DiscoLoop dongusu. `dar_alfa=0` iken model_a.Model ile BIT DUZEYINDE
    AYNI -- ek modul bile kurulmaz, parametre sayisi degismez.
    `test_sabit_b.py` bunu her kosuda dogruluyor."""

    def __init__(self, ayar: M.Ayar, vocab: int):
        super().__init__(ayar, vocab)
        self.dar_acik = ayar.dar_alfa > 0 or ayar.dar_kapi
        if not self.dar_acik:
            return                       # HICBIR ek parametre YOK
        self.dar_norm = M.RMSNorm(ayar.d)
        if ayar.dar_kapi:                # d+1 ek parametre (Denk. 6)
            self.kapi_w = nn.Parameter(torch.zeros(ayar.d))
            self.kapi_b = nn.Parameter(torch.zeros(1))
        # Yeni moduller `_ilk` uygulanmadan kurulduğu icin varsayilan
        # baslatmayla kalir: RMSNorm.g = 1, kapi = 0 -> sigmoid(0) = 0.5.

    def _phi(self, h):
        """Denk. 5. `head.weight` ZATEN `emb.weight` (bagli gomme), yani
        bu tam olarak softmax(W h / tau) @ W. Logitler final norm'dan
        geciriliyor -- modelin KENDI okuma yolu, logit lens ile ayni."""
        p = torch.softmax(self.head(self.nf(h)) / self.ayar.dar_tau, dim=-1)
        return p @ self.emb.weight

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        son = self.ayar.dongu - 1
        for k in range(self.ayar.dongu):
            for blk in self.bloklar:
                h = blk(h)
            if self.dar_acik and k < son:        # alfa(K-1) = 0
                g = self._phi(h)
                a = (torch.sigmoid((g * self.kapi_w).sum(-1, keepdim=True)
                                   + self.kapi_b)
                     if self.ayar.dar_kapi else self.ayar.dar_alfa)
                h = h + a * self.dar_norm(g)
        return self.head(self.nf(h))


AYAR = TABAN.degistir(ad="model_b", dar_alfa=0.5)
#      ^ TEK FARK. veri_wang, ood_pay=0.05, wd=0.5, l=4, dongu=2,
#        ort_bas=10000 hepsi model_a9'dan gelir.
#        dar_tau=1.0, dar_kapi=False -> once EN BASIT hali (sabit gecit).


# `kos.py` modulden UC sey istiyor: AYAR, egit, fark_bas. Ucu de burada
# olmali -- `fark_bas` disarida kalirsa kosu AttributeError ile duser ve
# bu ancak COLAB'da, egitim baslarken gorunurdu.
fark_bas = M.fark_bas


def egit(ayar=None, **kw):
    """model_a.egit, ama modeli ModelB kurar. `kos.py` bunu cagirir;
    model_a'nin `egit`i BURADA GOLGELENIR, yoksa darbogazsiz model
    kosardi ve kunye yine 'model_b' derdi -- sessiz ve olumcul."""
    return M.egit(ayar or AYAR, model_kur=ModelB, **kw)


if __name__ == "__main__":
    M.fark_bas(TABAN, AYAR)
    egit(AYAR)
