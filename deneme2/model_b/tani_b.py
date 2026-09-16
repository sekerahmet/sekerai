# -*- coding: utf-8 -*-
"""tani_b — model_b ailesinin AYRISTIRMASI.

`tani_a`nin AYNISI, tek farki modeli `ModelB` ile kurmasi. Ayristirma
mantigi TEK KAYNAK olarak `tani_a`da kalir; burada kopyalanmaz.
(`pencere_b` ile ayni desen.)

    python tani_b.py <klasor> [--genislik 5]
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import tani_a                                                # noqa: E402
from model_b import ModelB                                   # noqa: E402

tani_a.MODEL_SINIFI = ModelB          # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(tani_a.main())
