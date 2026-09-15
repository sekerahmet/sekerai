# -*- coding: utf-8 -*-
"""pencere_b — model_b ailesinin BIRINCIL OKUMASI.

`pencere_a`nin AYNISI, tek farki modeli `ModelB` ile kurmasi. Olcum
mantigi TEK KAYNAK olarak `pencere_a`da kalir; burada kopyalanmaz.
Kopyalansaydi iki dosya ayri ayri degisir ve sessizce farkli sayi
uretirdi -- bu projede tam boyle bir hata yasandi (arsivde olcme mantigi
iki dosyada AYRI duruyordu).

    python pencere_b.py <klasor> [--genislik 5]
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import pencere_a                                             # noqa: E402
from model_b import ModelB                                   # noqa: E402

pencere_a.MODEL_SINIFI = ModelB       # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(pencere_a.main())
