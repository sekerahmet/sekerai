"""baseline -- BaselineTransformer: Model Z'nin (deneme2/model_z/core/sentence/sentence.py) transformer tabani.  Kullanici,
6 Ekim: "Uygun transformer ayrı klasör olsun ve ayrı bir ajan çok güzel temiz kursun"; adlar onayli.  Tek fark baglam:
hikaye tek dizi, butun onceki kelimeler gorulur:
    [BOS] w_11 .. w_1L END w_21 .. END ... w_n1 .. END  ->  her konum sonraki token'i, son END'den sonra EOS
Tarif Model Z ile ayni (rope / output_loss / Block kopya, tests_baseline T4 ayni ciktiyi sinar): pre-norm RMSNorm, RoPE,
QK-norm, SwiGLU, bias yok, giris / cikis embedding ortak.  Plan: belge/model_z_temel/17_transformer_taban_tasarimi.md.
"""
import math

import torch
import torch.nn.functional as F


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> RoPE uygulanmis x (yarim boyut ciftleri)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def output_loss(h, W, target):
    """-log P ortalamasi: h (N, d) . W^T -> logit -> cross-entropy (GPU'da derlenir, train_baseline)."""
    return F.cross_entropy((h @ W.T).float(), target)


class Block(torch.nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = torch.nn.RMSNorm(d), torch.nn.RMSNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.q_norm, self.k_norm = torch.nn.RMSNorm(d // heads), torch.nn.RMSNorm(d // heads)
        self.proj = torch.nn.Linear(d, d, bias=False)
        self.gate_up = torch.nn.Linear(d, 2 * hidden, bias=False)
        self.down = torch.nn.Linear(hidden, d, bias=False)

    def forward(self, x, pos, allowed):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        # q, k autocast'te bf16: norm fp32'de (Model Z Block ile ayni), sonra geri
        q, k = rope(self.q_norm(q.float()).to(v.dtype), pos), rope(self.k_norm(k.float()).to(v.dtype), pos)
        if allowed is None:                                      # duz causal: hizli yol (GPU'da flash)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        g, u = self.gate_up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class BaselineTransformer(torch.nn.Module):
    def __init__(self, vocab, d=256, layers=4, heads=4):
        super().__init__()
        self.vocab = vocab
        self.END = len(vocab)                                    # cumle sonu: cikis ve girdi (ayrac)
        self.EOS = len(vocab) + 1                                # hikaye sonu: yalniz cikis
        hidden = -(-int(8 * d / 3) // 8) * 8
        self.E = torch.nn.Embedding(len(vocab) + 2, d)
        self.bos = torch.nn.Parameter(torch.zeros(d))            # BOS sozlukte degil: softmax Model Z ile ayni siniflar
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        self.loss_fn = output_loss
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                torch.nn.init.normal_(p, std=std)
        torch.nn.init.normal_(self.bos, std=0.02)

    def hidden(self, tok):
        """tok (B, T): konum 0 BOS (degeri okunmaz), dolgu sagda -> h (B, T, d); duz causal, konum = sira."""
        B, T = tok.shape
        x = self.E(tok[:, 1:])
        x = torch.cat([self.bos.to(x.dtype).expand(B, 1, -1), x], 1)
        pos = torch.arange(T, device=tok.device).expand(B, -1)
        for block in self.blocks:
            x = block(x, pos, None)
        return self.norm(x)

    def forward(self, tok, target):
        """Egitim -> ortalama -log P(dogru sonraki token); logit yalniz hedefi olan satirlarda."""
        h = self.hidden(tok)
        keep = target >= 0
        return self.loss_fn(h[keep], self.E.weight, target[keep])
