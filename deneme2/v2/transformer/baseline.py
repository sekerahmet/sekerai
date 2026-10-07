"""baseline (V2) -- BaselineTransformer, GPT-2 token, tam baglam (hikaye).  V1 deneme2/transformer_baseline/baseline.py'nin
V2 hali (import yok); tasarim belge/model_z_temel/20, arayuz 21 (PackedBatch, loss_per_target / generate sozlesmesi).
Model Z ile ayni tarif ve ayni hedefler (common/data.build_batch layout="transformer"):
    girdi   BOS(=EOS_ID) t_11 .. t_1L END t_21 .. END ... END
    hedef   t_11 ..      END      t_21 ...          ... EOS
    konum   hikaye ici sira (hikaye basinda 0)
Maske: ayni hikaye ve kv <= q (dolgu kendi dolgusunu gorur).  Tarif: d 512, 8 katman, 8 head, pre-norm RMSNorm, RoPE,
QK-norm fp32, SwiGLU, bias yok, tied embedding, sozluk VOCAB (GPT-2 + END).  Uretim KV cache ile (tek hikaye).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID, EOS_ID, VOCAB  # noqa: E402


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> RoPE uygulanmis x (yarim boyut ciftleri)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(torch.nn.Module):
    """Model Z V2 Block ile ayni tarif (kopya)."""

    def __init__(self, d, heads, hidden):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = torch.nn.RMSNorm(d), torch.nn.RMSNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.q_norm, self.k_norm = torch.nn.RMSNorm(d // heads), torch.nn.RMSNorm(d // heads)
        self.proj = torch.nn.Linear(d, d, bias=False)
        self.gate_up = torch.nn.Linear(d, 2 * hidden, bias=False)
        self.down = torch.nn.Linear(hidden, d, bias=False)

    def _qkv(self, x, pos):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(x.device.type, enabled=False):                  # QK-norm fp32 (belge 20 s4)
            q, k = self.q_norm(q.float()).to(v.dtype), self.k_norm(k.float()).to(v.dtype)
        return rope(q, pos), rope(k, pos), v

    def _finish(self, x, a):
        B, T, d = x.shape
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        g, u = self.gate_up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)

    def forward(self, x, pos, attn):
        """attn: None (duz causal), bool (B, T, T) (dense) ya da FlexAttention BlockMask."""
        q, k, v = self._qkv(x, pos)
        if attn is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif torch.is_tensor(attn):
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn[:, None])
        else:
            from torch.nn.attention.flex_attention import flex_attention
            a = flex_attention(q, k, v, block_mask=attn)
        return self._finish(x, a)


class BaselineTransformer(torch.nn.Module):
    def __init__(self, d=512, layers=8, heads=8):
        super().__init__()
        self.END, self.EOS = END_ID, EOS_ID
        hidden = -(-int(8 * d / 3) // 8) * 8
        self.E = torch.nn.Embedding(VOCAB, d)
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                torch.nn.init.normal_(p, std=std)

    def hidden(self, tokens, pos, attn):
        """tokens, pos (B, T); attn: None / dense / BlockMask -> h (B, T, d)."""
        x = self.E(tokens)
        for block in self.blocks:
            x = block(x, pos, attn)
        return self.norm(x)

    def _batch_hidden(self, batch, attn=None):
        """PackedBatch (tokens, pos, doc) -> h; attn yoksa dense maske: ayni hikaye ve kv <= q (dolgu doc -1 kendi
        dolgusunu gorur, bos satir olmaz)."""
        if attn is None:
            T = batch.doc.shape[1]
            causal = torch.ones(T, T, dtype=torch.bool, device=batch.doc.device).tril()
            attn = (batch.doc[:, :, None] == batch.doc[:, None, :]) & causal
        return self.hidden(batch.tokens, batch.pos, attn)

    def loss_per_target(self, batch, attn=None, chunk=4096):
        """-> nll (K,), pred (K,), target_kind (K,) (hedefli konumlar, satir sirasiyla; belge 21 s7 sozlesmesi)."""
        h = self._batch_hidden(batch, attn)
        keep = batch.target >= 0
        hk, tgt = h[keep], batch.target[keep]
        nll = torch.empty(len(tgt), device=h.device)
        pred = torch.empty(len(tgt), dtype=torch.long, device=h.device)
        for r in range(0, len(tgt), chunk):
            lg = (hk[r:r + chunk] @ self.E.weight.T).float()
            nll[r:r + chunk] = F.cross_entropy(lg, tgt[r:r + chunk], reduction="none")
            pred[r:r + chunk] = lg.argmax(-1)
        return nll, pred, batch.target_kind[keep]

    def _step(self, tokens, cache, n):
        """KV cache: yeni token'lar (1, t) konum n.. -> son konumun logit'i; cache = dict(k, v: katman basina (1, H, Tmax,
        hd), ilk cagrida k'nin dtype'iyla ayrilir (autocast'te bf16), T: Tmax) yerinde dolar.  Yeni token'lar kendi
        aralarinda causal, onbellegin tamamini gorur."""
        t = tokens.shape[1]
        pos = torch.arange(n, n + t, device=tokens.device)[None]
        x = self.E(tokens)
        allowed = torch.ones(t, n + t, dtype=torch.bool, device=tokens.device).tril(n)
        for l, block in enumerate(self.blocks):
            q, k, v = block._qkv(x, pos)
            if cache["k"][l] is None:
                cache["k"][l] = k.new_zeros(1, k.shape[1], cache["T"], k.shape[3])
                cache["v"][l] = v.new_zeros(1, v.shape[1], cache["T"], v.shape[3])
            cache["k"][l][:, :, n:n + t], cache["v"][l][:, :, n:n + t] = k, v
            a = F.scaled_dot_product_attention(q, cache["k"][l][:, :, :n + t], cache["v"][l][:, :, :n + t],
                                               attn_mask=allowed)
            x = block._finish(x, a)
        return (self.norm(x[:, -1]) @ self.E.weight.T)[0]

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None, open_last=False):
        """prompts: hikaye basina istem cumleleri (token listeleri, END yok) -> her istem icin (uretilen cumleler, END ile
        bitti mi listesi, eos).  generator None: acgozlu, yoksa ornekleme (sicaklik 1).  Model Z generate ile ayni kural:
        cumle basinda EOS hikayeyi bitirir, cumle icinde END ya da EOS cumleyi bitirir, max_tokens'ta kesilen cumle de
        kapanir (girdiye END).  open_last (belge 48): son istem cumlesine END eklenmez, ilk uretilen cumle onun devami."""
        dev = self.E.weight.device
        out = []
        for sents in prompts:
            opened = bool(open_last and sents)
            seq = [EOS_ID] + [t for s in sents for t in list(s) + [END_ID]]
            seq = seq[:-1] if opened else seq
            cache = dict(k=[None] * len(self.blocks), v=[None] * len(self.blocks),
                         T=len(seq) + max_sentences * (max_tokens + 1))
            logits = self._step(torch.tensor([seq], device=dev), cache, 0)
            n = len(seq)
            gen, ended, eos = [], [], False
            while len(gen) < max_sentences:
                cur, done = [], False
                while len(cur) < max_tokens:
                    p = logits.float()
                    w = int(p.argmax()) if generator is None else int(torch.multinomial(
                        torch.softmax(p, -1).cpu(), 1, generator=generator))
                    if w == self.EOS and not cur and not opened:
                        eos = True
                        break
                    if w in (self.END, self.EOS):
                        done = True
                        break
                    cur.append(w)
                    logits = self._step(torch.tensor([[w]], device=dev), cache, n)
                    n += 1
                if eos:
                    break
                gen.append(cur)
                ended.append(done)
                opened = False
                logits = self._step(torch.tensor([[END_ID]], device=dev), cache, n)
                n += 1
            out.append((gen, ended, eos))
        return out
