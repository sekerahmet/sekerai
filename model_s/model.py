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

    def __init__(self, d, heads, hidden, kv_heads=None, width=1, scheme="lie", values="swiglu", qk_dim=None):
        super().__init__(d, heads, hidden, kv_heads, scheme, 1.0, width, values, qk_dim)
        op = self.M if scheme == "mix" else self.A
        self.heads, self.kv_heads = op.heads, op.kv_heads

    def forward(self, x, pos, attn):
        return super().forward(x, Context(pos, attn))

    def _qkv(self, x, pos):
        if self.scheme == "mix":                        # uretim yolu (_qkv sonra _finish ayni x ile): yerel ozellik saklanir
            q, k, v, self._loc = self.M.qkv_local(x, pos)
            return q, k, v
        return self.A.qkv(x, pos)

    def _finish(self, x, a):
        if self.scheme == "mix":
            loc, self._loc = self._loc, None
            return x + self.M.out(a, loc)
        x = x + self.A.out(a)
        return x + self.F(x)


class ModelS(BaselineTransformer):
    def __init__(self, d, widths, heads, kv_heads=None, vocab_rows=VOCAB, op="split", qk_dim=None):
        """op: split (A sonra F, bugunku blok), parallel (MixOp linear values), mix (MixOp swiglu values; esit matris parametresi);
        tek ad ya da adim basina virgullu liste (orn. "split,mix,mix,split": ilk / son adimda F attention'i gorur).  qk_dim:
        head basina q / k boyu (Parca 8; varsayilan head boyu; kazanc FLOP'a, yerel genislik degismez)."""
        super().__init__(d, 1, heads, kv_heads, vocab_rows)            # E, norm, tarif; bloklar asagida
        self.widths, self.op = tuple(int(w) for w in widths), op
        ops = op.split(",") if "," in op else [op] * len(self.widths)
        assert len(ops) == len(self.widths) and set(ops) <= {"split", "parallel", "mix"}, "op: split | parallel | mix (adim basina)"
        assert self.widths and min(self.widths) >= 1, "widths >= 1"
        hidden = -(-int(8 * d / 3) // 8) * 8
        sv = {"split": ("lie", "swiglu"), "parallel": ("mix", "linear"), "mix": ("mix", "swiglu")}
        self.blocks = torch.nn.ModuleList(SBlock(d, heads, hidden, kv_heads, w, *sv[o], qk_dim=qk_dim) for w, o in zip(self.widths, ops))
        n = sum(self.widths)                                             # artis olcegi: sum(w) katmanli transformer gibi
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * n) if name.endswith(("w_o.weight", "w_down.weight", "w_out.weight")) else 0.02
                with torch.no_grad():
                    p.normal_(0.0, std)
