# -*- coding: utf-8 -*-
"""pencere_00 — model_00 ailesinin BIRINCIL OKUMASI.

`pencere_a`nin AYNISI, tek farki modeli `ModelSade` ile kurmasi. Olcum
mantigi TEK KAYNAK olarak `pencere_a`da kalir; burada kopyalanmaz.
(`pencere_b` ile ayni desen.)

    python pencere_00.py <klasor> [--genislik 5]
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "model_a"))

import pencere_a                                             # noqa: E402
from model_00 import ModelSade                               # noqa: E402

pencere_a.MODEL_SINIFI = ModelSade    # <- TEK FARK

if __name__ == "__main__":
    raise SystemExit(pencere_a.main())
