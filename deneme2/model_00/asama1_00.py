# -*- coding: utf-8 -*-
"""asama1_00 — model_00 ailesinin ASAMA-1 TESHISI.

`asama1`in AYNISI, tek farki modeli `ModelSade` ile kurmasi.
(`pencere_00` ile ayni desen.)

    python asama1_00.py <klasor> [--genislik 5] [--sonda] [--birim]
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import asama1                                                # noqa: E402
from model_00 import ModelSade                               # noqa: E402

asama1.MODEL_SINIFI = ModelSade       # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(asama1.main())
