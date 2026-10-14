"""RGNModel — decoder-only dil modeli (doküman 1.3 bileşen haritası).

    token -> embedding -> [ön-norm -> çekim dikkati -> residual -> ön-norm -> rasyonel MLP -> residual] × L
          -> final norm -> LM head (paylaşımlı embedding) -> sonraki token
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rgn.model.attention import Attention
from rgn.model.mlp import build_mlp
from rgn.model.norm import RMSNorm
from rgn.model.position import rope_tables


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = build_mlp(cfg)

    def forward(self, x, rope=None):
        s = x + self.attn(self.norm1(x), rope)          # S_i = X_i + Attn(RMSNorm(X))_i
        return s + self.mlp(self.norm2(s))              # H_i = S_i + MLP(RMSNorm(S))_i


def _ce_sum(h, weight, target):
    return F.cross_entropy((h @ weight.T).float(), target, reduction="sum")


def lm_loss(h, weight, target, chunk=8192, use_checkpoint=True):
    """Ortalama cross-entropy; logit'ler parça parça hesaplanır.  use_checkpoint: logit'ler geri geçişte yeniden hesaplanır
    (V × N logit matrisi bellekte tutulmaz; bedeli LM head çarpımının bir kez daha yapılması)."""
    h, target = h.reshape(-1, h.shape[-1]), target.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, target.numel(), chunk):
        args = (h[i:i + chunk], weight, target[i:i + chunk])
        use = use_checkpoint and torch.is_grad_enabled()
        total = total + (checkpoint(_ce_sum, *args, use_reentrant=False) if use else _ce_sum(*args))
    return total / target.numel()


class RGNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # W_vocab = W_embᵀ
        self.use_rope = cfg.pos in ("rope", "rope+alibi")
        self.loss_checkpoint = True                     # trainer TrainConfig.loss_checkpoint ile ayarlar
        if self.use_rope:
            cos, sin = rope_tables(cfg.max_len, cfg.d_model // cfg.n_heads, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        """N(0, std); residual akışına yazan W_O ve W_2 için std / sqrt(2L); bias'lar sıfır."""
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if p.dim() == 2:
                s = std / math.sqrt(2 * self.cfg.n_layers) if name.endswith(("attn.w_o.weight", "mlp.c.weight")) else std
                nn.init.normal_(p, 0.0, s)
            elif name.endswith(".bias"):
                nn.init.zeros_(p)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def rope(self, n):
        if not self.use_rope:
            return None
        assert n <= self.cfg.max_len, "dizi %d > max_len %d (RoPE tablosu)" % (n, self.cfg.max_len)
        return self.rope_cos[:n], self.rope_sin[:n]

    def hidden(self, idx):
        """idx (B, N) -> son normalize edilmiş durum (B, N, d)."""
        x = self.embed(idx)
        rope = self.rope(idx.shape[1])
        for block in self.blocks:
            x = block(x, rope)
        return self.norm_f(x)

    def forward(self, idx, targets=None):
        """targets yoksa logit'ler (B, N, V); varsa ortalama kayıp (skaler)."""
        h = self.hidden(idx)
        if targets is None:
            return self.lm_head(h)
        return lm_loss(h, self.lm_head.weight, targets, use_checkpoint=self.loss_checkpoint)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, eos_id=None, generator=None):
        """Önbelleksiz başvuru üretimi.  temperature 0: argmax (üstel işlem yok); > 0: örnekleme."""
        limit = self.cfg.max_len if self.use_rope else None
        for _ in range(max_new_tokens):
            ctx = idx[:, -limit:] if limit else idx
            logits = self.lm_head(self.hidden(ctx)[:, -1]).float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx
