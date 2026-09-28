# -*- coding: utf-8 -*-
"""model_20_transformer -- model_20 ile AYNI veri, sinav ve egitim tarifiyle kiyaslanan standart transformer (GPT-2 tipi, kucuk).

    token -> embedding E (n x D); cikista ayni matris (tied)
    LAYERS katman, her biri (pre-LN; akis normalize edilmez, yalniz dallarin girdisi):
        h <- h + W_out . attention(norm_attention(h))          W_query, W_key, W_value ayri; nedensel; q ve k'ya RoPE
                                                                (rope=False: konum bilgisi yok, model_20 gibi)
        h <- h + W_mlp_out . GELU(W_mlp_in . norm_mlp(h))
    logits = norm_final(h) . E^T
Dropout yok (model_20'de de yok, full batch).  Genislik D ve MLP birimi FACT_UNITS model_20'den; adlar kullanici onayiyla.
"""
import math

import torch
import torch.nn.functional as F

from model_20 import D, FACT_UNITS, POINTS_SEED, apply_rope   # RoPE model_20 ile ayni hesap

LAYERS = 2           # model_20'nin 2 turu gibi; ama her katmanin kendi agirligi (transformer standardi)
HEADS = 1            # model_20 gibi tek head (kullanici: "bizim modele benzeyen"); parametre sayisi head sayisindan bagimsiz


class TransformerLayer(torch.nn.Module):
    """Bir katman: attention dali ve MLP dali, ikisi de akisa ekler."""

    def __init__(self, d=D, heads=HEADS, units=FACT_UNITS, value_matrix=True, rope=True):
        super().__init__()
        assert d % heads == 0
        self.heads, self.rope = heads, rope
        self.norm_attention = torch.nn.LayerNorm(d)
        self.W_query = torch.nn.Linear(d, d)
        self.W_key = torch.nn.Linear(d, d)
        # value_matrix=False: V yok, bakilan yerin normlanmis durumu dogrudan getirilir (tek head'de V.O tek matrise esit)
        self.W_value = torch.nn.Linear(d, d) if value_matrix else None
        self.W_out = torch.nn.Linear(d, d)
        self.norm_mlp = torch.nn.LayerNorm(d)
        self.W_mlp_in = torch.nn.Linear(d, units)
        self.W_mlp_out = torch.nn.Linear(units, d)

    def forward(self, h):
        B, T, d = h.shape
        x = self.norm_attention(h)
        value = self.W_value(x) if self.W_value is not None else x
        q, k, v = (y.view(B, T, self.heads, d // self.heads).transpose(1, 2) for y in (self.W_query(x), self.W_key(x), value))
        if self.rope:
            q, k = apply_rope(q), apply_rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # olcek 1/√d_head
        h = h + self.W_out(a.transpose(1, 2).reshape(B, T, d))
        return h + self.W_mlp_out(F.gelu(self.W_mlp_in(self.norm_mlp(h))))


class TransformerModel(torch.nn.Module):
    """model_20.BlockModel ile ayni arayuz: logits(ids), loss(ids, mask) -> (toplam, nll)."""

    def __init__(self, n, d=D, layers=LAYERS, heads=HEADS, units=FACT_UNITS, seed=POINTS_SEED, value_matrix=True, rope=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(n, d)
        self.layers = torch.nn.ModuleList(TransformerLayer(d, heads, units, value_matrix, rope) for _ in range(layers))
        self.norm_final = torch.nn.LayerNorm(d)
        # GPT-2 baslangici: matrisler N(0, 0,02), akisa yazan W_out ve W_mlp_out / √(2 LAYERS); bias 0; LayerNorm 1 ve 0
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if "norm" in name:
                    continue
                if p.dim() == 1:
                    p.zero_()
                else:
                    std = 0.02 / math.sqrt(2 * layers) if name.split(".")[-2] in ("W_out", "W_mlp_out") else 0.02
                    p.copy_(torch.randn(p.shape, generator=g) * std)

    def logits(self, ids, positions=None):
        """positions (B, T) bool: yalniz o konumlar cikis katmanindan gecer -> (N, n)."""
        h = self.embedding(ids)
        for layer in self.layers:
            h = layer(h)
        if positions is not None:
            h = h[positions]
        return self.norm_final(h) @ self.embedding.weight.T

    def loss(self, ids, mask):
        valid = mask[:, 1:]
        nll = F.cross_entropy(self.logits(ids[:, :-1], valid), ids[:, 1:][valid])
        return nll, nll                                          # capa yok: toplam = nll
