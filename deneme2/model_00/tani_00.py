# -*- coding: utf-8 -*-
"""tani_00 — model_00 ailesinin AYRISTIRMASI.

`tani_a`nin AYNISI, tek farki modeli `ModelSade` ile kurmasi.

    python tani_00.py <klasor> [--genislik 5]
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import tani_a                                                # noqa: E402
from model_00 import ModelSade                               # noqa: E402

tani_a.MODEL_SINIFI = ModelSade       # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(tani_a.main())
