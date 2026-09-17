# -*- coding: utf-8 -*-
"""egitim_dok_05 — VERININ KENDISI, duz metin, ACIKLAMA YOK.

Kullanici karari, 17 Eylul 2026:
    *"ben tum egitim verisini aciklama olmadan bu egitim verisidir
    (hop 1 ve hop 2) gormek istiyorum. her satirdaki aciklamaya gerek
    yok."*

!! YERLESIM, kullanici UC KEZ yanlis dosyayi actiktan sonra degisti.
`analiz_05`in dokumu DENETIM icin: her satirin yaninda graf oku,
kisayol ve jeton dizisi var. Ikisi ayni klasorde durunca hep once o
aciliyordu -- ayrimi ADLANDIRMA degil YERLESIM tasiyor:

    veri/model_05/            VERININ KENDISI -- yalniz cumleler
        EGITIM_1hop.txt  EGITIM_2hop.txt
        SINAV_comp.txt  SINAV_ent.txt  SINAV_ent_yok.txt  SINAV_ood.txt
    veri/model_05/denetim/    `analiz_05`in dokumu (aciklamali)

Cumleler MODELIN GORDUGU DIZIDEN okunuyor (`analiz_05.Dok.oku`),
yeniden uretilmiyor -- burada ne goruyorsan havuzda o var.

EGITIM dosyalari UC yuzey bicimini de tasir (ayar.bicim=3).
SINAV dosyalari YALNIZ kanonik bicim -- olcme hep bicim 0 ile yapilir
(onkayit), yoksa "cesitlilik ogretti mi" ile "cesitlilikle mi sinandi"
birbirine karisirdi.

    python egitim_dok_05.py [--klasor <yol>]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import taban_05 as M                                          # noqa: E402
import analiz_05 as AZ                                        # noqa: E402

NL = chr(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", default=None)
    a = ap.parse_args()
    kl = a.klasor or os.path.join(_K, "veri", "model_05")
    os.makedirs(kl, exist_ok=True)
    d = AZ.Dok("model_05")

    isler = [("EGITIM_1hop.txt", d.v.one, M.kodla_1hop, True),
             ("EGITIM_2hop.txt", d.v.tr2, M.kodla_2hop, True)]
    # SINAV dosyalari BURADA URETILMIYOR: `analiz_05` onlari kopru ve
    # kisayolla birlikte doker (06-09). Duz halini de yazmak AYNI
    # icerigin ikinci kopyasi olurdu.

    for ad, lst, kodla, hepsi in isler:
        yol = os.path.join(kl, ad)
        n = max(1, d.ayar.bicim) if hepsi else 1
        with io.open(yol, "w", encoding="utf-8") as f:
            for x in lst:
                for i in range(n):
                    f.write(d.oku(kodla(d.v, [x], i)[0][0]) + NL)
        print(f"yazildi: {ad:<20} {len(lst) * n:>7,} satir"
              f"   {os.path.getsize(yol)/1024/1024:>5.1f} MB"
              + ("" if hepsi else "   (yalniz kanonik bicim)"))


if __name__ == "__main__":
    main()
