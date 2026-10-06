"""sentence (V2) -- SentenceTransformer, GPT-2 token.  V1 deneme2/model_z/core/sentence/sentence.py'nin kopyasi
+ degisiklik (import yok); tarif belge/model_z_temel/22_V2_model_z_tarifi.md, alanlar belge 21 (PackedBatch).

Hikaye tek dizi; cumle k'nin END'inin yerinde Z_k (girdi z_in(z_norm(z_k))), dizi boyu transformer akisiyla ayni,
hedefler konum konum ayni:
    girdi   BOS  t_11 .. t_1L  [Z_1]  t_21 .. t_2L  [Z_2] ...  [Z_n]
    hedef   t_11 t_12 .. END   t_21   t_22 .. END   t_31  ...  EOS
    konum   0    1    .. L     1      2    ..       2     ...  n     (z_slot_positions: Z_k k; cumle k, token i: k-1+i)
Maske (model_z_mask, tek mask_mod -> dense ya da FlexAttention BlockMask): ayni hikaye, kv <= q, ve kv BOS/ZTOK ya da
(q, kv ayni cumlenin token'i).  Uretim: SummaryCache (ozet onbellegi BOS + Z'ler kalici, cumle onbellegi cumle bitince
silinir).  Tarif (Llama sinifi, V1): pre-norm RMSNorm, RoPE, QK-norm (fp32), SwiGLU, bias yok, tied embedding.
"""
import math

import torch
import torch.nn.functional as F

from sentence_z import encode_z
from data import END_ID, EOS_ID, VOCAB, Kind  # noqa: E402  (sentence_z common/'u yola ekledi)

BOS, TOKEN, END, ZTOK, PAD = Kind.BOS, Kind.TOKEN, Kind.END, Kind.ZTOK, Kind.PAD
_ENCODE_Z_CUDA = None


def _encode_z(keys, ids, mask):
    """encode_z; CUDA'da torch.compile(dynamic=True), surec basina bir kez (belge 24 s5 C; olculdu 5,51 -> 0,47 ms, fark
    1,2e-7).  CPU'da eager (Windows'ta inductor icin MSVC yok; testler bu yolda)."""
    global _ENCODE_Z_CUDA
    if not ids.is_cuda:
        return encode_z(keys, ids, mask)
    if _ENCODE_Z_CUDA is None:
        _ENCODE_Z_CUDA = torch.compile(encode_z, dynamic=True)
    return _ENCODE_Z_CUDA(keys, ids, mask)


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> RoPE uygulanmis x (yarim boyut ciftleri)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def z_slot_positions(kind, doc):
    """kind, doc (B, T) -> RoPE konumu (B, T): BOS 0; Z_k k; cumle k'nin i. token'i (k-1)+i (k: hikayede bu cumleden
    onceki Z sayisi + 1).  Dolgu 0."""
    B, T = kind.shape
    idx = torch.arange(T, device=kind.device).expand(B, -1)
    new_doc = torch.ones_like(kind, dtype=torch.bool)
    new_doc[:, 1:] = doc[:, 1:] != doc[:, :-1]
    doc_start = torch.cummax(torch.where(new_doc, idx, torch.zeros_like(idx)), 1).values
    z = (kind == ZTOK).long()
    zc = z.cumsum(1)
    z_before_doc = torch.gather(zc - z, 1, doc_start)                       # hikaye basina kadar Z sayisi
    nz = zc - z_before_doc                                                  # hikayede bu konuma kadar (dahil) Z sayisi
    summary = (kind == BOS) | (kind == ZTOK)
    last_summary = torch.cummax(torch.where(summary, idx, torch.zeros_like(idx)), 1).values
    pos = torch.where(kind == ZTOK, nz, nz + (idx - last_summary))           # token: Z sayisi + cumle ici sira (1'den)
    pos = torch.where(kind == BOS, torch.zeros_like(pos), pos)
    return torch.where(kind == PAD, torch.zeros_like(pos), pos)


def model_z_mask(kind, doc, sent):
    """-> mask_mod(b, h, q, kv) (FlexAttention imzasi; tensor indeksiyle dense de kurulur).  Ayni hikaye, kv <= q, ve kv
    ozet (BOS / ZTOK) ya da q ile kv ayni cumlenin token'i.  Dolgu (doc -1) kendi dolgusunu gorur: bos satir olmaz."""
    def mask_mod(b, h, q, kv):
        kq, kk = kind[b, q], kind[b, kv]
        summary = (kk == BOS) | (kk == ZTOK)
        word = ((kq == TOKEN) | (kq == END)) & ((kk == TOKEN) | (kk == END)) & (sent[b, q] == sent[b, kv])
        pad = (kq == PAD) & (kk == PAD)                                    # dolgu satiri bos kalmasin (SDPA NaN)
        return (doc[b, q] == doc[b, kv]) & (kv <= q) & (summary | word | pad)
    return mask_mod


def _dense(mask_mod, B, T, device):
    """mask_mod -> bool (B, T, T) (SDPA yolu; CPU egitimi ve testler)."""
    b = torch.arange(B, device=device)[:, None, None]
    q = torch.arange(T, device=device)[None, :, None]
    kv = torch.arange(T, device=device)[None, None, :]
    return mask_mod(b, 0, q, kv)


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

    def _qkv(self, x, pos):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(x.device.type, enabled=False):                  # QK-norm fp32 (V2 tarifi, belge 20 s4)
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


class SentenceTransformer(torch.nn.Module):
    def __init__(self, keys, d=512, layers=8, heads=8):
        super().__init__()
        self.keys = keys                                         # z anahtarlari (parametre degil, state_dict'te yok)
        self.END, self.EOS = END_ID, EOS_ID
        hidden = -(-int(8 * d / 3) // 8) * 8
        self.E = torch.nn.Embedding(VOCAB, d)
        self.z_norm = torch.nn.RMSNorm(keys["z"])
        self.z_in = torch.nn.Linear(keys["z"], d, bias=False)
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                torch.nn.init.normal_(p, std=std)

    def _z(self, sents, device):
        """Token listeleri (END yok) -> z (N, z_dim) fp32, cihazda."""
        if not sents:
            return torch.zeros(0, self.keys["z"], device=device)
        L = max(1, max(len(s) for s in sents))
        ids = torch.zeros(len(sents), L, dtype=torch.long)
        mask = torch.zeros(len(sents), L, dtype=torch.bool)
        for i, s in enumerate(sents):
            ids[i, :len(s)] = torch.as_tensor(list(s), dtype=torch.long)
            mask[i, :len(s)] = True
        return encode_z(self.keys, ids.to(device), mask.to(device))

    def _embed(self, tokens, kind, zvec):
        x = self.E(tokens)
        zpos = kind == ZTOK
        if zpos.any():
            x = x.index_put((zpos,), self.z_in(self.z_norm(zvec[zpos])).to(x.dtype))
        return x

    def hidden(self, tokens, kind, pos, zvec, attn):
        """tokens, kind, pos (B, T); zvec (B, T, z_dim) (yalniz ZTOK'ta okunur); attn: None / dense / BlockMask."""
        x = self._embed(tokens, kind, zvec)
        for block in self.blocks:
            x = block(x, pos, attn)
        return self.norm(x)

    def _batch_hidden(self, batch, attn=None):
        """PackedBatch (belge 21: tokens, kind, pos, doc, sent, z_slots, z_sentences) -> h; attn yoksa dense maske.  z
        dogrudan Z_k satirlarina yazilir (zvec / boolean indeks / .any() yok: GPU senkronu yok; belge 24 s5 B).  hidden()
        ile ayni satirlar ayni sirayla (z_slots satir sirasiyla = kind == ZTOK sirasi)."""
        B, T = batch.tokens.shape
        dev = batch.tokens.device
        x = self.E(batch.tokens)
        ids, mask = batch.z_sentences                                       # (n_z, Lmax), Z_k'nin cumlesi (belge 21)
        if len(ids):                                                        # CPU tarafi uzunluk
            z = _encode_z(self.keys, ids.to(dev), mask.to(dev))
            x = x.index_put(batch.z_slots, self.z_in(self.z_norm(z)).to(x.dtype))
        if attn is None:
            attn = _dense(model_z_mask(batch.kind, batch.doc, batch.sent), B, T, dev)
        for block in self.blocks:
            x = block(x, batch.pos, attn)
        return self.norm(x)

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

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None):
        """prompts: hikaye basina istem cumleleri (token listeleri, END yok) -> her istem icin (uretilen cumleler,
        END ile bitti mi listesi, eos).  generator None: acgozlu, yoksa ornekleme.  SummaryCache ile, istem basina."""
        dev = self.E.weight.device
        out = []
        for sents in prompts:
            cache = SummaryCache(self)
            logits = cache.logits
            for s in sents:
                for t in s:
                    logits = cache.append_token(t)
                logits = cache.close_sentence(self._z([s], dev)[0])
            gen, ended, eos = [], [], False
            while len(gen) < max_sentences:
                cur, done = [], False
                while len(cur) < max_tokens:
                    p = logits.float()
                    w = int(p.argmax()) if generator is None else int(torch.multinomial(
                        torch.softmax(p, -1).cpu(), 1, generator=generator))
                    if w == self.EOS and not cur:
                        eos = True
                        break
                    if w in (self.END, self.EOS):
                        done = True
                        break
                    cur.append(w)
                    logits = cache.append_token(w)
                if eos:
                    break
                gen.append(cur)
                ended.append(done)
                logits = cache.close_sentence(self._z([cur], dev)[0])
            out.append((gen, ended, eos))
        return out


class SummaryCache:
    """Tek hikayenin KV onbellegi (22 s4): ozet (BOS + Z_1..Z_k; kelime gormedikleri icin kalici) ve cumle (simdiki
    cumlenin token'lari; cumle bitince silinir).  Konumlar kendiliginden: token k+i, Z_k k.  Her cagri sonraki token'in
    logit'ini dondurur (self.logits)."""

    def __init__(self, model):
        self.m = model
        self.dev = model.E.weight.device
        L = len(model.blocks)
        self.sum_k, self.sum_v = [None] * L, [None] * L
        self.sen_k, self.sen_v = [None] * L, [None] * L
        self.n_z, self.i = 0, 0
        x = model.E(torch.tensor([[EOS_ID]], device=self.dev))              # BOS = EOS token'i (belge 21 s1)
        self.logits = self._step(x, 0, summary=True)

    def _step(self, x, pos, summary):
        p = torch.tensor([[pos]], device=self.dev)
        for l, block in enumerate(self.m.blocks):
            q, k, v = block._qkv(x, p)
            ks = [c for c in (self.sum_k[l], None if summary else self.sen_k[l]) if c is not None] + [k]
            vs = [c for c in (self.sum_v[l], None if summary else self.sen_v[l]) if c is not None] + [v]
            a = F.scaled_dot_product_attention(q, torch.cat(ks, 2), torch.cat(vs, 2))
            if summary:
                self.sum_k[l] = k if self.sum_k[l] is None else torch.cat([self.sum_k[l], k], 2)
                self.sum_v[l] = v if self.sum_v[l] is None else torch.cat([self.sum_v[l], v], 2)
            else:
                self.sen_k[l] = k if self.sen_k[l] is None else torch.cat([self.sen_k[l], k], 2)
                self.sen_v[l] = v if self.sen_v[l] is None else torch.cat([self.sen_v[l], v], 2)
            x = block._finish(x, a)
        return (self.m.norm(x) @ self.m.E.weight.T)[0, -1]

    @torch.no_grad()
    def append_token(self, token):
        """Simdiki cumleye token -> sonraki token'in logit'i."""
        self.i += 1
        x = self.m.E(torch.tensor([[token]], device=self.dev))
        self.logits = self._step(x, self.n_z + self.i, summary=False)
        return self.logits

    @torch.no_grad()
    def close_sentence(self, z):
        """Cumle bitti: z (z_dim,) -> Z_k ozet onbellegine, cumle onbellegi silinir -> sonraki cumlenin ilk token'i (ya
        da EOS) logit'i."""
        self.n_z += 1
        self.i = 0
        self.sen_k = [None] * len(self.sen_k)
        self.sen_v = [None] * len(self.sen_v)
        x = self.m.z_in(self.m.z_norm(z.float()[None, None])).to(self.m.E.weight.dtype)
        self.logits = self._step(x, self.n_z, summary=True)
        return self.logits
