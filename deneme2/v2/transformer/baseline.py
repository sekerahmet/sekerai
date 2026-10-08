"""baseline (V2) -- BaselineTransformer, GPT-2 token, tam baglam (hikaye).  V1 deneme2/transformer_baseline/baseline.py'nin
V2 hali (import yok); tasarim belge/model_z_temel/20, arayuz 21 (PackedBatch, loss_per_target / generate sozlesmesi).
Model Z ile ayni tarif ve ayni hedefler (common/data.build_batch layout="transformer"):
    girdi   BOS(=EOS_ID) t_11 .. t_1L END t_21 .. END ... END
    hedef   t_11 ..      END      t_21 ...          ... EOS
    konum   hikaye ici sira (hikaye basinda 0)
Maske: ayni hikaye ve kv <= q (dolgu kendi dolgusunu gorur).  Tarif: d 512, 8 katman, 8 head, pre-norm RMSNorm, RoPE,
QK-norm fp32, SwiGLU, bias yok, tied embedding, sozluk VOCAB (GPT-2 + END; vocab_rows > VOCAB: dolgu satirlari sifir,
cikis VOCAB'a kesilir, belge 89).  Uretim KV cache ile (tek hikaye).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID, EOS_ID, ROW_LEN, VOCAB  # noqa: E402


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> RoPE uygulanmis x (yarim boyut ciftleri)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def gqa_sdpa(q, k, v, mask=None):
    """Model Z sentence.gqa_sdpa ile ayni (kopya; belge 74): GQA'da maskeli SDPA math'a dusmesin.  Esit head dogrudan SDPA,
    Tq 1 grup Tq'ya katlanir, Tq > 1 k / v head boyunca gecici genisletilir."""
    H, Hkv = q.shape[1], k.shape[1]
    if H == Hkv:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    B, _, Tq, hd = q.shape
    if Tq == 1 and (mask is None or mask.shape[-2] == 1 and (mask.dim() < 3 or mask.shape[-3] == 1)):
        return F.scaled_dot_product_attention(q.reshape(B, Hkv, H // Hkv, hd), k, v, attn_mask=mask).reshape(B, H, 1, hd)
    return F.scaled_dot_product_attention(q, k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1),
                                          attn_mask=mask)


class Block(torch.nn.Module):
    """Model Z V2 Block ile ayni tarif (kopya)."""

    def __init__(self, d, heads, hidden, kv_heads=None):
        """kv_heads (GQA; varsayilan heads): k / v head sayisi, heads'in boleni; qkv d -> d + 2 d kv / heads."""
        super().__init__()
        self.heads = heads
        self.kv_heads = int(kv_heads or heads)
        assert self.kv_heads > 0 and heads % self.kv_heads == 0, "kv_heads heads'in boleni olmali"
        self.n1, self.n2 = torch.nn.RMSNorm(d), torch.nn.RMSNorm(d)
        self.qkv = torch.nn.Linear(d, d + 2 * d * self.kv_heads // heads, bias=False)
        self.q_norm, self.k_norm = torch.nn.RMSNorm(d // heads), torch.nn.RMSNorm(d // heads)
        self.proj = torch.nn.Linear(d, d, bias=False)
        self.gate_up = torch.nn.Linear(d, 2 * hidden, bias=False)
        self.down = torch.nn.Linear(hidden, d, bias=False)

    def _qkv(self, x, pos):
        B, T, d = x.shape
        hd = d // self.heads
        if self.kv_heads == self.heads:
            q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        else:                                                                # GQA: k / v kv_heads
            q, k, v = self.qkv(self.n1(x)).split([d, self.kv_heads * hd, self.kv_heads * hd], -1)
            q = q.view(B, T, self.heads, hd).transpose(1, 2)
            k, v = (t.view(B, T, self.kv_heads, hd).transpose(1, 2) for t in (k, v))
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
        gqa = dict(enable_gqa=True) if self.kv_heads != self.heads else {}
        if attn is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, **gqa)
        elif torch.is_tensor(attn):
            a = gqa_sdpa(q, k, v, attn[:, None])
        else:
            from torch.nn.attention.flex_attention import flex_attention
            bs = attn.BLOCK_SIZE[0]                     # 128 disinda varsayilan cekirdek hata veriyor (belge 37)
            a = flex_attention(q, k, v, block_mask=attn, kernel_options=None if bs == 128 else dict.fromkeys(
                ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"), bs), **gqa)
        return self._finish(x, a)


class BaselineTransformer(torch.nn.Module):
    def __init__(self, d=512, layers=8, heads=8, kv_heads=None, vocab_rows=VOCAB):
        """kv_heads: her katmanda k / v head sayisi (GQA kiyasi; varsayilan heads).  vocab_rows: E satir sayisi (>= VOCAB;
        dolgu satirlari sifir, ilk agirlik VOCAB'li modelle ayni)."""
        super().__init__()
        self.END, self.EOS, self.row_len = END_ID, EOS_ID, ROW_LEN
        hidden = -(-int(8 * d / 3) // 8) * 8
        assert vocab_rows >= VOCAB, "vocab_rows >= VOCAB"
        self.E = torch.nn.Embedding(VOCAB, d)
        if vocab_rows > VOCAB:                                   # kurulum RNG'si VOCAB'li modelle ayni; dolgu sifir
            self.E = torch.nn.Embedding(vocab_rows, d, _weight=torch.zeros(vocab_rows, d))
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden, kv_heads) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                with torch.no_grad():
                    (p[:VOCAB] if name == "E.weight" else p).normal_(0.0, std)   # dolgu satiri rastgele sayi cekmez
        with torch.no_grad():
            self.E.weight[VOCAB:].zero_()

    def max_positions(self):
        """Uretimde hikaye basina konum siniri: egitim satiri row_len (belge 89b)."""
        return self.row_len

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

    def _logits(self, h):
        """h (..., d) -> cikis h @ E^T (bagli embedding)."""
        return (h @ self.E.weight.T)[..., :VOCAB]

    def loss_per_target(self, batch, attn=None, chunk=4096):
        """-> nll (K,), pred (K,), target_kind (K,) (hedefli konumlar, satir sirasiyla; belge 21 s7 sozlesmesi)."""
        h = self._batch_hidden(batch, attn)
        keep = batch.target >= 0
        hk, tgt = h[keep], batch.target[keep]
        nll = torch.empty(len(tgt), device=h.device)
        pred = torch.empty(len(tgt), dtype=torch.long, device=h.device)
        for r in range(0, len(tgt), chunk):
            lg = self._logits(hk[r:r + chunk]).float()
            nll[r:r + chunk] = F.cross_entropy(lg, tgt[r:r + chunk], reduction="none")
            pred[r:r + chunk] = lg.argmax(-1)
        return nll, pred, batch.target_kind[keep]

    def _step(self, tokens, cache, n):
        """KV cache: yeni token'lar (1, t) konum n.. -> son konumun normlu durumu (1, d; logit _logits ile); cache = dict(k, v: katman basina (1, H, Tmax,
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
            a = gqa_sdpa(q, cache["k"][l][:, :, :n + t], cache["v"][l][:, :, :n + t], allowed)
            x = block._finish(x, a)
        return self.norm(x[:, -1])

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None, open_last=False, on_token=None,
                 stop_when=None):
        """prompts: hikaye basina istem cumleleri (token listeleri, END yok) -> her istem icin (uretilen cumleler, END ile
        bitti mi listesi, eos).  generator None: acgozlu, yoksa ornekleme (sicaklik 1).  Model Z generate ile ayni kural:
        cumle basinda EOS hikayeyi bitirir, cumle icinde END ya da EOS cumleyi bitirir, max_tokens'ta kesilen cumle de
        kapanir (girdiye END).  open_last (belge 48): son istem cumlesine END eklenmez, ilk uretilen cumle onun devami.
        on_token(w): her uretilen token'dan sonra, on_token(None): cumle kapaninca (akan yazim; cikti degismez).  stop_when(gen): her
        kapanan cumleden sonra, True ise o istemin uretimi biter (cikti = tam uretimin oneki).  Onbellek StaticCache (belge
        84; STATIC_DECODE), yoksa _step + dict onbellek; fp32'de token token ayni.  Konum siniri (belge 89b): hikaye
        row_len'i asmaz; sinira gelince o anki cumle kesik (ended False) doner, uretim biter."""
        dev = self.E.weight.device
        out, limit = [], self.max_positions()
        for sents in prompts:
            opened = bool(open_last and sents)
            seq = [EOS_ID] + [t for s in sents for t in list(s) + [END_ID]]
            seq = seq[:-1] if opened else seq
            used = len(seq)
            if STATIC_DECODE:
                sc = StaticCache(self, max(1, min(max_sentences * (max_tokens + 1) + 1, limit - used)))
                logits, step = sc.prefill(seq), sc.append
            else:
                cache = dict(k=[None] * len(self.blocks), v=[None] * len(self.blocks),
                             T=len(seq) + max_sentences * (max_tokens + 1))
                logits, n = self._logits(self._step(torch.tensor([seq], device=dev), cache, 0))[0], [len(seq)]

                def step(w, cache=cache, n=n):
                    n[0] += 1
                    return self._logits(self._step(torch.tensor([[w]], device=dev), cache, n[0] - 1))[0]
            gen, ended, eos, full = [], [], False, False
            while len(gen) < max_sentences:
                cur, done = [], False
                while len(cur) < max_tokens:
                    if used + 2 > limit:                         # sonraki token + kapanis egitim boyunu asar (belge 89b)
                        full = True
                        break
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
                    if on_token is not None:
                        on_token(w)
                    logits = step(w)
                    used += 1
                if eos or (full and not cur):
                    break
                gen.append(cur)
                ended.append(done)
                if on_token is not None:
                    on_token(None)
                if full or (stop_when is not None and stop_when(gen)):
                    break
                opened = False
                logits = step(END_ID)
                used += 1
            out.append((gen, ended, eos))
        return out


STATIC_DECODE = True       # generate: StaticCache; False: _step + dict onbellek (eski yol, kiyas icin)
_COMPILED = {}             # derlenmis adim (surec basina bir kez; derleme hata verirse derlemesiz graph)


def decode_sdpa(q, k, v, mask, f32_scores=False):
    """Model Z sentence.decode_sdpa ile ayni (kopya; belge 84): Tq 1, grup katlama, skor fp32, maske (B, S), @ V."""
    B, H, _, hd = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    qf = (q * hd ** -0.5).reshape(B * Hkv, H // Hkv, hd)
    kt = k.reshape(B * Hkv, S, hd).transpose(1, 2)
    s = torch.bmm(qf, kt, out_dtype=torch.float32) if f32_scores else torch.bmm(qf, kt).float()
    p = torch.softmax(s.view(B, Hkv, -1, S).masked_fill(~mask[:, None, None], float("-inf")), -1).to(v.dtype)
    return torch.matmul(p, v).reshape(B, H, 1, hd)


def _bmm_f32(t):
    """bf16 / fp16 skorunu fp32 cikaran bmm (out_dtype) var mi (bir kez denenir)."""
    if "bmm_f32" not in _COMPILED:
        try:
            a = t.new_zeros(1, 1, 8)
            torch.bmm(a, a.transpose(1, 2), out_dtype=torch.float32)
            _COMPILED["bmm_f32"] = True
        except (RuntimeError, TypeError):
            _COMPILED["bmm_f32"] = False
    return _COMPILED["bmm_f32"]


def _static_step(m, f32, w, n, K, V, out):
    """StaticCache'in tek adimi (yerinde, dallanmasiz): token w konum n'de, onbellek [0, n] -> out logit, n += 1."""
    x = m.E(w)[:, None]
    ar = torch.arange(len(w), device=w.device)
    mask = torch.arange(K[0].shape[2], device=w.device)[None] <= n[:, None]
    for l, block in enumerate(m.blocks):
        q, k, v = block._qkv(x, n[:, None])
        K[l][ar, :, n] = k[:, :, 0]
        V[l][ar, :, n] = v[:, :, 0]
        x = block._finish(x, decode_sdpa(q, K[l], V[l], mask, f32))
    out.copy_(m._logits(m.norm(x[:, 0])))
    n.add_(1)


class StaticCache:
    """Sabit tamponlu KV onbellegi (belge 84; Model Z sentence.StaticCache'in transformer esi): istem _step ile tek
    geciste (tampona dogrudan), sonra adim basina _static_step; CUDA'da ilk adimda derlenip CUDA graph'a yakalanir.
    positions: istemden sonra en cok eklenecek konum (asilirsa DURUR)."""
    BUCKET = 256         # sentence.StaticCache ile ayni kademeler: BUCKET x {1, 1,5} x 2^k

    def __init__(self, model, positions):
        self.m, self.dev, self.positions, self.run = model, model.E.weight.device, positions, None

    @torch.no_grad()
    def prefill(self, seq):
        """seq (token listesi, BOS dahil) -> son konumun logit'i."""
        T = len(seq)
        b = self.BUCKET
        self.limit = next(t for k in range(40) for t in (b << k, (3 * b // 2) << k) if t >= T + self.positions)
        cache = dict(k=[None] * len(self.m.blocks), v=[None] * len(self.m.blocks), T=self.limit)
        hn = self.m._step(torch.tensor([seq], device=self.dev), cache, 0)
        self.K, self.V, self.t = cache["k"], cache["v"], T
        self.logits = self.m._logits(hn)[0]
        self.n, self.w = torch.tensor([T], device=self.dev), torch.zeros(1, dtype=torch.long, device=self.dev)
        self.out = self.logits.new_zeros(1, self.logits.shape[-1])
        self.f32 = self.out.dtype != torch.float32 and _bmm_f32(self.out)
        return self.logits

    def _step(self):
        _COMPILED.get("step", _static_step)(self.m, self.f32, self.w, self.n, self.K, self.V, self.out)

    def _capture(self):
        """sentence.StaticCache._capture ile ayni: derleme (bir kez; hata -> derlemesiz) + CUDA graph."""
        if self.dev.type != "cuda":
            return self._step
        if "step" not in _COMPILED:
            _COMPILED["step"] = torch.compile(_static_step, dynamic=False)
            for k in ("recompile_limit", "cache_size_limit"):                 # kademe x dtype x model sekilleri
                if getattr(torch._dynamo.config, k, 64) < 64:
                    setattr(torch._dynamo.config, k, 64)
        saved = self.n.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                try:
                    self._step()
                except Exception as e:  # noqa: BLE001
                    if _COMPILED["step"] is _static_step:
                        raise
                    print("StaticCache: torch.compile olmadi, derlemesiz graph: %s" % str(e).splitlines()[0][:150])
                    _COMPILED["step"] = _static_step
                    self._step()
                self.n.copy_(saved)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        return self.graph.replay

    @torch.no_grad()
    def append(self, token):
        """Token (cumle token'i ya da END) -> sonraki token'in logit'i."""
        self.t += 1
        assert self.t <= self.limit, "StaticCache: tampon asildi (%d > %d)" % (self.t, self.limit)
        self.w.fill_(int(token))
        if self.run is None:
            self.run = self._capture()
        self.run()
        self.logits = self.out[0]
        return self.logits
