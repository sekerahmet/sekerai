"""model -- Model S Parca 4: baslangic kosulu (embedding), cozucu (adimlar), okuma (norm + bagli embedding).

    u(0) = E[token];  u <- S_n(u), n = 1..N (lie adimi, genislik w_n);  logit = P(u) E^T

widths (w_1, ..., w_N): adim basina genislik; w = k birlesik hibrit asama (k adimin `wide` formulu, integrator.Step) --
toplam FLOP sum(w) katmanli transformer'la ayni, sirali derinlik N.  Hepsi 1: bugunku transformer (baseline ile ayni agirlikla
bit duzeyinde; tests_model_s model).  Uretim, sinav, toplu uretim BaselineTransformer'dan (blok arayuzu _qkv / _finish).
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
from baseline import BaselineTransformer  # noqa: E402
from data import VOCAB  # noqa: E402
from integrator import Step  # noqa: E402
from operators import Context  # noqa: E402


class SBlock(Step):
    """lie adimi, baseline.Block arayuzuyle (forward(x, pos, attn), _qkv, _finish; heads / kv_heads): KV cache ve StaticCache
    uretimi degismeden calisir."""

    def __init__(self, d, heads, hidden, kv_heads=None, width=1):
        super().__init__(d, heads, hidden, kv_heads, "lie", 1.0, width)
        self.heads, self.kv_heads = self.A.heads, self.A.kv_heads

    def forward(self, x, pos, attn):
        return super().forward(x, Context(pos, attn))

    def _qkv(self, x, pos):
        return self.A.qkv(x, pos)

    def _finish(self, x, a):
        x = x + self.A.out(a)
        return x + self.F(x)


class ModelS(BaselineTransformer):
    def __init__(self, d, widths, heads, kv_heads=None, vocab_rows=VOCAB):
        super().__init__(d, 1, heads, kv_heads, vocab_rows)            # E, norm, tarif; bloklar asagida
        self.widths = tuple(int(w) for w in widths)
        assert self.widths and min(self.widths) >= 1, "widths >= 1"
        hidden = -(-int(8 * d / 3) // 8) * 8
        self.blocks = torch.nn.ModuleList(SBlock(d, heads, hidden, kv_heads, w) for w in self.widths)
        n = sum(self.widths)                                             # artis olcegi: sum(w) katmanli transformer gibi
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * n) if name.endswith(("w_o.weight", "w_down.weight")) else 0.02
                with torch.no_grad():
                    p.normal_(0.0, std)
