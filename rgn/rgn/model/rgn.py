"""RGNModel — decoder-only dil modeli.

    token -> gömme -> [ön-norm -> dikkat (Y: yerel pencere | G: genel) -> residual
                       -> ön-norm -> SwiGLU MLP (+ kelimeye özel nöron; shared_mlp False: yalnız özel nöron) -> residual] × L -> final norm -> LM head (paylaşımlı)

Kelimeye özel nöron: her katmanda her kelimenin kendi satırı [g | u | w] (d'şer); o kelimenin konumunda
a = SiLU(g·x)·(u·x), akışa a·w eklenir (ortak MLP'ye paralel, aynı x = RMSNorm(S)).  Adres token kimliğinden belli
(priv_map: token -> satır; 0 = ortak "diğer").  Eğitimde yalnız adımda geçen satırların kopyası kullanılır (trainer,
rgn/model/private_kernel.py); değerlendirme / üretimde tam tablo.
"""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rgn.model.attention import Attention
from rgn.model.mlp import SwiGLU
from rgn.model.norm import RMSNorm
from rgn.model.position import rope_tables


def private_vocab(cfg):
    """Özel satırı olan token'lar (liste; satır = sıra + 1)."""
    from rgn.paths import ROOT
    return json.loads((ROOT / cfg.private_vocab).read_text(encoding="utf-8"))["tokens"]


class Block(nn.Module):
    """kind 'Y': yerel (son local_window token) | 'G': genel (tam causal).  private_rows > 0: kelimeye özel nöron tablosu."""

    def __init__(self, cfg, kind="G", private_rows=0):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model) if (cfg.shared_mlp or private_rows) else None   # yalnız dikkat: ikinci norm yok
        self.attn = Attention(cfg, window=cfg.local_window if kind == "Y" else 0)
        self.mlp = SwiGLU(cfg.d_model, cfg.ff_dim(), cfg.mlp_bias) if cfg.shared_mlp else None
        self.private = nn.Parameter(torch.empty(private_rows, 3 * cfg.d_model)) if private_rows else None

    def private_out(self, xn, rows=None, priv=None):
        """xn (B, N, d) -> (B, N, d).  rows (B, N): tam tablodan; priv = (loc (B, N), sub (R, 3d), plan): eğitimde
        adımda geçen satırların kopyası, birleşik çekirdekle (ara (B, N, 3d) tensör yok)."""
        B, N, D = xn.shape
        if priv is not None:
            from rgn.model.private_kernel import private_neuron
            loc, sub, plan = priv
            return private_neuron(xn.reshape(B * N, D), loc.reshape(-1), sub, plan).view(B, N, D)
        w = F.embedding(rows, self.private).view(B, N, 3, D).to(xn.dtype)
        a = F.silu((w[..., 0, :] * xn).sum(-1)) * (w[..., 1, :] * xn).sum(-1)
        return a.unsqueeze(-1) * w[..., 2, :]

    def forward(self, x, rope, mask=None, rows=None, priv=None, parts=None):
        """H = S + MLP(RMSNorm(S)) [+ özel],  S = X + Attn(RMSNorm(X)).  parts (dict): bileşenler buraya yazılır
        (teşhis: dikkat, mlp, özel — akışa eklenen değerler; olmayan bileşen None)."""
        a = self.attn(self.norm1(x), rope, mask)
        s = x + a
        xn = self.norm2(s) if self.norm2 is not None else None
        m = None
        out = s
        if self.mlp is not None:
            m = self.mlp(xn)
            out = out + m
        p = None
        if self.private is not None:
            p = self.private_out(xn, rows, priv)
            out = out + p
        if parts is not None:
            parts.update(attn=a, mlp=m, private=p)
        return out


def _ce_sum(h, weight, target):
    return F.cross_entropy((h @ weight.T).float(), target, reduction="sum")


def lm_loss(h, weight, target, chunk=8192, use_checkpoint=True):
    """Ortalama cross-entropy (hedef -100 sayılmaz); logit'ler parça parça.  use_checkpoint: logit'ler geri geçişte
    yeniden hesaplanır (V × N logit matrisi bellekte tutulmaz)."""
    h, target = h.reshape(-1, h.shape[-1]), target.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, target.numel(), chunk):
        args = (h[i:i + chunk], weight, target[i:i + chunk])
        use = use_checkpoint and torch.is_grad_enabled()
        total = total + (checkpoint(_ce_sum, *args, use_reentrant=False) if use else _ce_sum(*args))
    return total / (target != -100).sum().clamp_min(1)


class RGNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        rows = 0
        if cfg.private_neurons:                                        # token -> özel satır (0: ortak "diğer")
            used = private_vocab(cfg)
            rows = len(used) + 1
            pm = torch.zeros(cfg.vocab_size, dtype=torch.long)
            pm[torch.tensor(used)] = torch.arange(1, rows)
            self.register_buffer("priv_map", pm)
        else:
            self.priv_map = None
        self.blocks = nn.ModuleList(Block(cfg, kind, rows) for kind in cfg.layer_kinds())
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight                       # W_vocab = W_embᵀ
        self.loss_checkpoint = True                                    # trainer TrainConfig.loss_checkpoint ile ayarlar
        cos, sin = rope_tables(cfg.max_len, cfg.d_model // cfg.n_heads, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        """N(0, std); residual akışına yazan W_O ve W_2 için std / sqrt(2L); bias 0; kapı 0 (kapı = 1); özel satırda
        g, u ~ N(0, std), w = 0 (başlangıçta özel nöronsuz modelle aynı)."""
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if name.endswith("attn.w_aux.weight"):
                nn.init.zeros_(p)
            elif name.endswith(".private"):
                nn.init.normal_(p, 0.0, std)
                p.data.view(p.shape[0], 3, -1)[:, 2].zero_()
            elif p.dim() == 2:
                s = std / math.sqrt(2 * self.cfg.n_layers) if name.endswith(("attn.w_o.weight", "mlp.c.weight")) else std
                nn.init.normal_(p, 0.0, s)
            elif name.endswith(".bias"):
                nn.init.zeros_(p)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def private_blocks(self):
        """Kelimeye özel nöronu olan bloklar (sırayla; özel nöron açıksa hepsi)."""
        return [b for b in self.blocks if b.private is not None]

    def rope(self, n):
        assert n <= self.cfg.max_len, "dizi %d > max_len %d (RoPE tablosu)" % (n, self.cfg.max_len)
        return self.rope_cos[:n], self.rope_sin[:n]

    def hidden(self, idx, masks=None, priv=None, parts=None):
        """idx (B, N) -> son normalize edilmiş durum (B, N, d).
        masks: story_masks (hikâye dizilimi) | priv: (loc, katman başına satır kopyası, plan) — eğitim yolu
        parts: liste verilirse katman başına bileşen sözlüğü (+ "embed")."""
        x = self.embed(idx)
        if parts is not None:
            parts.append({"embed": x})
        rope = self.rope(idx.shape[1])
        rows = self.priv_map[idx] if (self.priv_map is not None and priv is None) else None
        for i, block in enumerate(self.blocks):
            bp = None if priv is None else (priv[0], priv[1][i], priv[2])
            d = {} if parts is not None else None
            x = block(x, rope, None if masks is None else masks[block.kind], rows, bp, d)
            if parts is not None:
                parts.append(d)
        return self.norm_f(x)

    def forward(self, idx, targets=None, masks=None, priv=None):
        """targets yoksa logit'ler (B, N, V); varsa ortalama kayıp (skaler)."""
        h = self.hidden(idx, masks=masks, priv=priv)
        if targets is None:
            return self.lm_head(h)
        return lm_loss(h, self.lm_head.weight, targets, use_checkpoint=self.loss_checkpoint)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, eos_id=None, generator=None):
        """Önbelleksiz başvuru üretimi.  temperature 0: argmax; > 0: örnekleme."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_len:]
            logits = self.lm_head(self.hidden(ctx)[:, -1]).float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx
