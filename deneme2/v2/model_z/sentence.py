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
shared_vocab=True (ortak sozluk, belge 29; adlar onayli): E = MeaningEmbedding [m_gain * m_hat ; e], z ayni E'den
(f = [m_hat ; e_hat] / sqrt(2), encode_z_rows, tam gradyan); keys build_keys(..., identity=False).
own_vocab=True (kontrol; kullanici 6 Ekim, ad onayli): meaning yok, E tamamen ogrenilen nn.Embedding; z ayni E'den, ortak
sozlukle ayni bicim: f = [birim(E[:h]) ; birim(E[h:])] / sqrt(2), yari ici R_t, tam gradyan; keys build_keys(None, ...).
open_z=W (belge 31; adlar onayli): son W cumlenin z'si acilir.  Her token konumu j icin bir sanal kv: u = R_t^T z_pos
(kendi cumlesinin z'si, t cumle ici sira), x = A u + b_v (ZUnbinder), her katmanda blogun k / v'si, RoPE = token'in
konumu.  KV = [gercek T ; sanal T] (sanal j gercek j ile hizali, KV_LEN = 2T); kurali model_z_open_mask(W): kelime
sorgusu cumle s-W..s-1'in, Z_s sorgusu s-W+1..s'nin sanal token'larini gorur.  Saklanan yalniz z.
"""
import math

import torch
import torch.nn.functional as F

from sentence_z import encode_z, encode_z_rows, unbind_z
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


class MeaningEmbedding(torch.nn.Module):
    """Ortak sozluk E(v) = [m_gain * m_hat(v) ; e(v)] (belge 29; ad onayli).  m_hat sabit (meaning, birim; buffer,
    state_dict'te yok: kaynak meaning dosyasi), e ogrenilir (nn.Embedding: param_groups decay'siz tanir), m_gain tek
    ogrenilen skaler (baslangic 1).  nn.Embedding arayuzu: forward(ids), weight (tied cikis).  Model bagimsiz
    (transformer da kullanabilir).  f_table(): z icin [m_hat ; e_hat] / sqrt(2) (yarilar ayri birim)."""

    def __init__(self, m_hat, e_dim):
        super().__init__()
        self.register_buffer("m_hat", m_hat.float().clone(), persistent=False)
        self.e = torch.nn.Embedding(len(m_hat), e_dim)
        self.m_gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, ids):
        return torch.cat([self.m_gain * F.embedding(ids, self.m_hat), self.e(ids)], -1)

    @property
    def weight(self):
        return torch.cat([self.m_gain * self.m_hat, self.e.weight], 1)

    def f_table(self):
        e = self.e.weight.float()
        return torch.cat([self.m_hat, e / e.norm(dim=1, keepdim=True).clamp_min(1e-9)], 1) / 2 ** 0.5


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


def model_z_open_mask(W):
    """open_z (belge 31; ad onayli): -> mask_fn(kind, doc, sent) (model_z_mask imzasi); mask_mod KV = [gercek T ; sanal T]
    uzerinde (kv >= T: sanal j = kv - T, gercek j ile hizali).  Sanal j gorunur: j token, ayni hikaye, ve q kelime
    (cumle s) iken sent[j] in [s-W, s-1], q Z_s iken sent[j] in [s-W+1, s].  mask_fn.kv_factor = 2 (KV_LEN = 2T)."""
    def mask_fn(kind, doc, sent):
        T = kind.shape[1]
        base = model_z_mask(kind, doc, sent)

        def mask_mod(b, h, q, kv):
            virt = kv >= T
            j = torch.where(virt, kv - T, kv)
            real = base(b, h, q, j) & ~virt
            kq, sq, sj = kind[b, q], sent[b, q], sent[b, j]
            word = (kq == TOKEN) & (sj >= sq - W) & (sj <= sq - 1)
            zq = (kq == ZTOK) & (sj >= sq - W + 1) & (sj <= sq)
            vis = virt & (kind[b, j] == TOKEN) & (doc[b, j] == doc[b, q]) & (doc[b, q] >= 0) & (word | zq)
            return real | vis
        return mask_mod
    mask_fn.kv_factor = 2
    return mask_fn


def _dense(mask_mod, B, T, device, kv_len=None):
    """mask_mod -> bool (B, T, kv_len) (SDPA yolu; CPU egitimi ve testler); kv_len varsayilan T."""
    b = torch.arange(B, device=device)[:, None, None]
    q = torch.arange(T, device=device)[None, :, None]
    kv = torch.arange(T if kv_len is None else kv_len, device=device)[None, None, :]
    return mask_mod(b, 0, q, kv)


class ZUnbinder(torch.nn.Module):
    """open_z (belge 31; ad onayli): acilmis z satirlari u (N, z_pos) -> sanal token girdisi x = A u + b_v (N, d).  A
    ogrenilen dogrusal harita, b_v 'sanal' tur vektoru (ayni RoPE konumundaki gercek token'dan ayirir)."""

    def __init__(self, z_pos, d):
        super().__init__()
        self.A = torch.nn.Linear(z_pos, d, bias=False)
        self.b_v = torch.nn.Parameter(torch.zeros(d))

    def forward(self, u):
        return self.A(u) + self.b_v


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

    def _kv(self, x, pos):
        """Yalniz k, v (sanal token'lar; open_z): blogun kendi agirliklari, k_norm, RoPE."""
        B, T, d = x.shape
        k, v = F.linear(self.n1(x), self.qkv.weight[d:]).view(B, T, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(x.device.type, enabled=False):
            k = self.k_norm(k.float()).to(v.dtype)
        return rope(k, pos), v

    def forward(self, x, pos, attn, virt=None):
        """attn: None (duz causal), bool (B, T, T | 2T) (dense) ya da FlexAttention BlockMask.  virt: (x_v, pos_v) sanal
        token'lar (open_z; KV = [gercek ; sanal])."""
        q, k, v = self._qkv(x, pos)
        if virt is not None:
            kv_, vv_ = self._kv(*virt)
            k, v = torch.cat([k, kv_], 2), torch.cat([v, vv_], 2)
        if attn is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif torch.is_tensor(attn):
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn[:, None])
        else:
            from torch.nn.attention.flex_attention import flex_attention
            a = flex_attention(q, k, v, block_mask=attn)
        return self._finish(x, a)


class SentenceTransformer(torch.nn.Module):
    def __init__(self, keys, d=512, layers=8, heads=8, shared_vocab=False, own_vocab=False, open_z=0):
        super().__init__()
        assert not (shared_vocab and own_vocab), "shared_vocab ve own_vocab birlikte olmaz"
        self.keys = keys                                         # z anahtarlari (parametre degil, state_dict'te yok)
        self.END, self.EOS = END_ID, EOS_ID
        self.shared_vocab, self.own_vocab, self.open_z = shared_vocab, own_vocab, int(open_z)
        hidden = -(-int(8 * d / 3) // 8) * 8
        if shared_vocab:                                         # ortak sozluk (belge 29): z ve model ayni E
            assert keys.get("identity") is False, "shared_vocab: build_keys(..., identity=False)"
            assert keys["z_pos"] == d, "d (%d) = m yarisi + e yarisi (%d) olmali" % (d, keys["z_pos"])
            self.E = MeaningEmbedding(keys["m_hat"], d - keys["m_dim"])
        elif own_vocab:                                          # kontrol: meaning yok, z modelin kendi E'sinden
            assert keys.get("identity") is False and "m_hat" not in keys, "own_vocab: build_keys(None, ..., identity=False)"
            assert keys["z_pos"] == d, "d (%d) = iki yari (%d) olmali" % (d, keys["z_pos"])
            self.E = torch.nn.Embedding(VOCAB, d)
        else:
            self.E = torch.nn.Embedding(VOCAB, d)
        self.z_norm = torch.nn.RMSNorm(keys["z"])
        self.z_in = torch.nn.Linear(keys["z"], d, bias=False)
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        if self.open_z:                                          # belge 31: z'yi acan dikkat
            self.unbinder = ZUnbinder(keys["z_pos"], d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                torch.nn.init.normal_(p, std=std)
        if self.open_z:
            torch.nn.init.normal_(self.unbinder.b_v, std=0.02)

    @property
    def mask_fn(self):
        """Bu modelin maske kurali (recipe.block_mask / dense_mask'a verilir; kv_factor ile KV_LEN = kv_factor * T)."""
        return model_z_open_mask(self.open_z) if self.open_z else model_z_mask

    @property
    def m_gain(self):
        """Ortak sozlukte m yarisinin ogrenilen olcegi (skaler tensor; gunluk icin), yoksa None."""
        return self.E.m_gain if self.shared_vocab else None

    def f_table(self):
        """z'nin token vektorleri (VOCAB, z_pos), guncel E'den: ortak sozlukte [m_hat ; e_hat]/sqrt2, own_vocab'da
        [birim(E[:h]) ; birim(E[h:])]/sqrt2; sabit F'li yolda None."""
        if self.shared_vocab:
            return self.E.f_table()
        if self.own_vocab:
            h = self.keys["m_dim"]
            E = self.E.weight.float()
            n = lambda x: x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)  # noqa: E731
            return torch.cat([n(E[:, :h]), n(E[:, h:])], 1) / 2 ** 0.5
        return None

    def _zrows(self, ids, mask, flat=None):
        """z (n, z_dim): ortak sozlukte / own_vocab'da guncel E'den (gradyanli), yoksa sabit F'den.  flat: (satir, sira)
        gercek token indeksleri CPU'da hazirsa (batch.z_flat, oneri) encode_z_rows senkronsuz."""
        if self.shared_vocab or self.own_vocab:
            return encode_z_rows(self.keys, self.f_table(), ids, mask, flat)  # eager: az cekirdek (belge 29 s3)
        return _encode_z(self.keys, ids, mask)

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
        if self.shared_vocab or self.own_vocab:
            return encode_z_rows(self.keys, self.f_table(), ids.to(device), mask.to(device))
        return encode_z(self.keys, ids.to(device), mask.to(device))

    def _embed(self, tokens, kind, zvec):
        x = self.E(tokens)
        zpos = kind == ZTOK
        if zpos.any():
            x = x.index_put((zpos,), self.z_in(self.z_norm(zvec[zpos])).to(x.dtype))
        return x

    def hidden(self, tokens, kind, pos, zvec, attn):
        """tokens, kind, pos (B, T); zvec (B, T, z_dim) (yalniz ZTOK'ta okunur); attn: None / dense / BlockMask."""
        assert not self.open_z, "hidden() open_z'yi desteklemez: _batch_hidden"
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
        virt = None
        if len(ids):                                                        # CPU tarafi uzunluk
            z = self._zrows(ids.to(dev), mask.to(dev), getattr(batch, "z_flat", None))
            x = x.index_put(batch.z_slots, self.z_in(self.z_norm(z)).to(x.dtype))
            if self.open_z:
                virt = (self._virtual(batch, z, x.dtype), batch.pos)
        if attn is None:
            attn = _dense(self.mask_fn(batch.kind, batch.doc, batch.sent), B, T, dev, (2 if self.open_z else 1) * T)
        for block in self.blocks:
            x = block(x, batch.pos, attn, virt) if virt is not None else block(x, batch.pos, attn)
        return self.norm(x)

    def _virtual(self, batch, z, dtype):
        """open_z: sanal token girdisi (B, T, d), gercek token j ile hizali (token olmayan konumlar 0; maske gizler).
        Token'in z'si: duz satir sirasinda ondan onceki ZTOK sayisi (z_slots satir sirasiyla = kind == ZTOK sirasi);
        cumle ici sira t = pos - sent - 1 (model_z duzeni)."""
        B, T = batch.kind.shape
        flat = batch.kind.reshape(-1)
        isz = (flat == ZTOK).long()
        zid = (isz.cumsum(0) - isz).clamp(max=len(z) - 1)                    # sabit sekil: GPU senkronu yok
        t = (batch.pos - batch.sent - 1).reshape(-1).clamp(0, len(self.keys["signs"]) - 1)
        u = unbind_z(self.keys, z[:, :self.keys["z_pos"]][zid], t)            # (B*T, z_pos), token olmayanlar atilir
        xv = torch.where((flat == TOKEN)[:, None], self.unbinder(u), 0.0).to(dtype)
        return xv.view(B, T, -1)

    def loss_per_target(self, batch, attn=None, chunk=4096):
        """-> nll (K,), pred (K,), target_kind (K,) (hedefli konumlar, satir sirasiyla; belge 21 s7 sozlesmesi)."""
        h = self._batch_hidden(batch, attn)
        keep = batch.target >= 0
        hk, tgt = h[keep], batch.target[keep]
        nll = torch.empty(len(tgt), device=h.device)
        pred = torch.empty(len(tgt), dtype=torch.long, device=h.device)
        W = self.E.weight                                                   # ortak sozlukte bir kez kurulur
        for r in range(0, len(tgt), chunk):
            lg = (hk[r:r + chunk] @ W.T).float()
            nll[r:r + chunk] = F.cross_entropy(lg, tgt[r:r + chunk], reduction="none")
            pred[r:r + chunk] = lg.argmax(-1)
        return nll, pred, batch.target_kind[keep]

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None):
        """prompts: hikaye basina istem cumleleri (token listeleri, END yok) -> her istem icin (uretilen cumleler,
        END ile bitti mi listesi, eos).  generator None: acgozlu, yoksa ornekleme.  SummaryCache ile, istem basina.
        Cumle en cok min(max_tokens, longest) token: z'nin konum anahtari longest'e kadar (egitimin en uzun cumlesi);
        kesilen cumle max_tokens kesimi gibi ended False ile doner (belge 26 B2)."""
        dev = self.E.weight.device
        max_tokens = min(max_tokens, len(self.keys["signs"]) - 1)
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
        self.virt = []                                                      # open_z: son W cumlenin sanal kv'si
        self.n_z, self.i = 0, 0
        x = model.E(torch.tensor([[EOS_ID]], device=self.dev))              # BOS = EOS token'i (belge 21 s1)
        self.logits = self._step(x, 0, summary=True)

    def _step(self, x, pos, summary):
        p = torch.tensor([[pos]], device=self.dev)
        for l, block in enumerate(self.m.blocks):
            q, k, v = block._qkv(x, p)
            vk = [e[0][l] for e in self.virt if pos > 0]                     # BOS sanal gormez
            vv = [e[1][l] for e in self.virt if pos > 0]
            ks = [c for c in (self.sum_k[l], None if summary else self.sen_k[l]) if c is not None] + vk + [k]
            vs = [c for c in (self.sum_v[l], None if summary else self.sen_v[l]) if c is not None] + vv + [v]
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
        if self.m.open_z and self.i:                                        # bu cumlenin sanal kv'si (Z_s de gorur)
            n = self.i
            u = unbind_z(self.m.keys, z.float()[None, :self.m.keys["z_pos"]].expand(n, -1), torch.arange(n, device=self.dev))
            xv = self.m.unbinder(u).to(self.m.E.weight.dtype)[None]
            pv = (self.n_z + 1 + torch.arange(n, device=self.dev))[None]   # token konumlari (k-1)+i, i = 1..n
            kv = [block._kv(xv, pv) for block in self.m.blocks]
            self.virt = (self.virt + [([a for a, _ in kv], [b for _, b in kv])])[-self.m.open_z:]
        self.n_z += 1
        self.i = 0
        self.sen_k = [None] * len(self.sen_k)
        self.sen_v = [None] * len(self.sen_v)
        x = self.m.z_in(self.m.z_norm(z.float()[None, None])).to(self.m.E.weight.dtype)
        self.logits = self._step(x, self.n_z, summary=True)
        return self.logits
