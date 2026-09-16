# -*- coding: utf-8 -*-
"""sor_00 — model_00 ailesi icin ELLE SORU.

`model_b/sor.py`nin AYNISI, tek farki modeli `ModelSade` ile kurmasi.

    python sor_00.py <klasor> --soru "Ayse Yilmaz'in annesi"
"""
from __future__ import annotations

import sys, os
_K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_K, "model_a"))
sys.path.insert(0, os.path.join(_K, "model_b"))

import sor                                                   # noqa: E402
from model_00 import ModelSade                               # noqa: E402

sor.MODEL_SINIFI = ModelSade          # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(sor.main())
