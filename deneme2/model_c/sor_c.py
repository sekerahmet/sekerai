# -*- coding: utf-8 -*-
"""sor_c — model_c ailesinde KENDI SORUNU SOR.

`sor`un AYNISI, tek farki modeli `ModelC` ile kurmasi.
(`pencere_c` / `tani_c` / `asama1_c` ile ayni desen.)

    python sor_c.py <klasor> [--genislik 5] [--soru "Ahmet Kilic baba"]

!! BU BIR OLCU DEGIL -- elle sorulan sorular SECILMIS sorulardir.
"""
from __future__ import annotations

import os
import sys

_C = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_C), "model_a"),
           os.path.join(os.path.dirname(_C), "model_b"), _C):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sor                                                    # noqa: E402
from model_c import ModelC                                    # noqa: E402

sor.MODEL_SINIFI = ModelC             # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(sor.main())
