# -*- coding: utf-8 -*-
"""asama1_b — model_b ailesinin ASAMA-1 teshisi.

`asama1`in AYNISI, tek farki modeli `ModelB` ile kurmasi. Teshis mantigi
TEK KAYNAK olarak `asama1`de kalir; burada kopyalanmaz.
(`pencere_b` ve `tani_b` ile ayni desen.)

    python asama1_b.py <klasor> [--genislik 5] [--tara]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import asama1                                                 # noqa: E402
from model_b import ModelB                                    # noqa: E402

asama1.MODEL_SINIFI = ModelB          # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(asama1.main())
