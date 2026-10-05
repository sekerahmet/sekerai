"""sentence -- SentenceTransformer (kullanici, 5 Ekim: "Sentence transformer çok dikkatli kur en iyi transformer modeli
gibi"; "Safece 1 cümle sonrasına bakacak şekilde"; ad onayli).  Egitim ve olcum: train_sentence.py.

Gecmis cumleler kelime olarak degil, cumle basina tek z (sentence_z.encode_z) olarak gorulur:
    model:  [BOS] [z_1] ... [z_{k-1}]  w_1 ... w_t  ->  w_{t+1}   (cumle END ile biter)
    hikaye sonu: [BOS] [z_1] ... [z_n]  ->  EOS   (son cumleden sonra yeni cumle yerine; normal transformer'in <eos>'u)
Egitimde her cumle kendi kisa dizisi (duz causal); konum (RoPE) modelin gordugu siradir (z_j konumu j, k. cumlenin i.
kelimesi k + i).  Hikayenin tek dizi oldugu maskeli yol da var (group); iki yol ayni logit'i verir (sinandi, 3e-7).
Tarif (Llama sinifi): pre-norm RMSNorm, RoPE, QK-norm, SwiGLU, bias yok, giris / cikis embedding ortak.
"""
import math

import torch
import torch.nn.functional as F

PAD, BOS, WORD, ZTOK = 0, 1, 2, 3          # token turu


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> RoPE uygulanmis x (yarim boyut ciftleri)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


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
        q, k = rope(self.q_norm(q), pos), rope(self.k_norm(k), pos)
        if allowed is None:                                      # duz causal: hizli yol (GPU'da flash)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        g, u = self.gate_up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class SentenceTransformer(torch.nn.Module):
    def __init__(self, vocab, z_dim, d=64, layers=4, heads=4):
        super().__init__()
        self.vocab = vocab
        self.END = len(vocab)                                    # yalniz cikis: cumle sonu
        self.EOS = len(vocab) + 1                                # yalniz cikis: hikaye sonu
        hidden = -(-int(8 * d / 3) // 8) * 8
        self.E = torch.nn.Embedding(len(vocab) + 2, d)
        self.bos = torch.nn.Parameter(torch.zeros(d))
        self.z_norm = torch.nn.RMSNorm(z_dim)
        self.z_in = torch.nn.Linear(z_dim, d, bias=False)
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                torch.nn.init.normal_(p, std=std)
        torch.nn.init.normal_(self.bos, std=0.02)

    def hidden(self, kind, tok, zvec, pos, group):
        """kind, tok, pos (B, T); zvec (B, T, z_dim) (yalniz ZTOK konumlarinda okunur) -> h (B, T, d).
        group None: tek cumlelik kisa dizi [BOS][z..] kelimeler, duz causal.  group (B, T): hikaye dizisi; kelime goren
        BOS, onceki z'ler ve kendi cumlesinin onceki kelimeleri, z goren BOS ve onceki z'ler (iki yol esdeger, sinandi)."""
        T = kind.shape[1]
        x = self.E(tok)
        zpos = kind == ZTOK                                      # z yolu (512 genislik) yalniz z tokenlarinda
        x = x.index_put((zpos,), self.z_in(self.z_norm(zvec[zpos])).to(x.dtype))
        x = torch.where((kind == BOS)[..., None], self.bos.to(x.dtype).expand_as(x), x)
        allowed = None
        if group is not None:
            word, summary = kind == WORD, (kind == BOS) | (kind == ZTOK)
            causal = torch.ones(T, T, dtype=torch.bool, device=kind.device).tril()
            same = word[:, :, None] & word[:, None, :] & (group[:, :, None] == group[:, None, :])
            allowed = causal & (summary[:, None, :] | same)
            allowed = allowed | torch.eye(T, dtype=torch.bool, device=kind.device)      # dolgu satiri bos kalmasin
        for block in self.blocks:
            x = block(x, pos, allowed)
        return self.norm(x)

    def forward(self, kind, tok, zvec, pos, group, target):
        """Egitim -> ortalama -log P(dogru sonraki kelime ya da END); logit yalniz hedefi olan satirlarda."""
        h = self.hidden(kind, tok, zvec, pos, group)
        keep = target >= 0
        logits = h[keep] @ self.E.weight.T
        return F.cross_entropy(logits.float(), target[keep])

    @torch.no_grad()
    def generate(self, zs, max_words):
        """zs (B, k, z_dim): her ornegin onceki cumlelerinin z'leri (k ayni) -> sonraki cumle (acgozlu, END'e kadar);
        ilk kelime EOS ise [EOS] (hikaye bitti)."""
        B, k, _ = zs.shape
        dev = zs.device
        words = torch.zeros(B, 0, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_words + 1):
            t = words.shape[1]
            kind = torch.cat([torch.full((B, 1), BOS), torch.full((B, k), ZTOK), torch.full((B, t), WORD)], 1).to(dev)
            tok = torch.cat([torch.zeros(B, 1 + k, dtype=torch.long, device=dev), words], 1)
            zvec = torch.cat([zs.new_zeros(B, 1, zs.shape[2]), zs, zs.new_zeros(B, t, zs.shape[2])], 1)
            pos = torch.arange(1 + k + t, device=dev).expand(B, -1)
            h = self.hidden(kind, tok, zvec, pos, None)
            w = (h[:, -1] @ self.E.weight.T).argmax(-1)
            if t == 0:                                           # hikaye sonu yalniz cumle basinda
                ended = w == self.EOS
            else:
                w = torch.where(w == self.EOS, torch.full_like(w, self.END), w)
            w = torch.where(done, torch.full_like(w, self.END), w)
            done |= (w == self.END) | (w == self.EOS)
            words = torch.cat([words, w[:, None]], 1)
            if done.all():
                break
        out = []
        for row, e in zip(words.tolist(), ended.tolist()):
            out.append([self.EOS] if e else (row[:row.index(self.END)] if self.END in row else row))
        return out
