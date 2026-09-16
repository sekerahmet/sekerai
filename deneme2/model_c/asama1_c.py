# -*- coding: utf-8 -*-
"""asama1_c — model_c ailesinin ASAMA-1 TESHISI.

`asama1`in AYNISI, tek farki modeli `ModelC` ile kurmasi. Mantik TEK
KAYNAK olarak `asama1`de kalir; burada kopyalanmaz. Kopyalansaydi iki
dosya ayri ayri degisir ve sessizce farkli sayi uretirdi -- bu projede
tam boyle bir hata yasandi. (`pencere_b` / `tani_b` ile ayni desen.)

    python asama1_c.py <klasor> [--genislik 5]
"""
from __future__ import annotations

import os
import sys

_C = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_C), "model_a"),
           os.path.join(os.path.dirname(_C), "model_b"), _C):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asama1                                              # noqa: E402
from model_c import ModelC                                    # noqa: E402

asama1.MODEL_SINIFI = ModelC        # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(asama1.main())
