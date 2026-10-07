"""sentence (V2) -- SentenceTransformer (Model Z), GPT-2 token.  Tarif belge/model_z_temel/22 (duzen, maske, onbellek),
35 (ogrenilen z), 40 / 43 (global_layers); alanlar belge 21 (PackedBatch).  Formullu z (sentence_z, z_in / z_norm,
--learned_z 0) kaldirildi (kullanici, 7 Ekim: "bence temizlik başlasın"; belge 44): git etiketi
v2-before-formula-cleanup-20261007.  Daha eski secenekler (meaning, ortak sozluk, open_z): v2-before-cleanup-20261006.

Hikaye tek dizi; cumle k'nin END'inin yerinde Z_k (girdi E(END), transformer'in END konumuyla ayni girdi), dizi boyu
transformer akisiyla ayni, hedefler konum konum ayni:
    girdi   BOS  t_11 .. t_1L  [Z_1]  t_21 .. t_2L  [Z_2] ...  [Z_n]
    hedef   t_11 t_12 .. END   t_21   t_22 .. END   t_31  ...  EOS
    konum   0    1    .. L     1      2    ..       2     ...  n     (Z_k k; cumle k, token i: k-1+i)
Maske (model_z_read_mask, tek mask_mod -> dense ya da FlexAttention BlockMask): ayni hikaye, kv <= q, ve kv BOS/ZTOK ya
da (q, kv ayni cumlenin token'i; Z_k kendi cumlesinin token'larini da gorur).  Z_k'nin her katmandaki hali sonraki
cumlelerin K/V'si (belge 35 (b)).  model_z_mask: okumasiz hali (teshis: z_ablate read_off).  Uretim: SummaryCache (ozet
onbellegi BOS + Z'ler kalici, cumle onbellegi cumle bitince silinir).  Tarif (Llama sinifi): pre-norm RMSNorm, RoPE,
QK-norm (fp32), SwiGLU, bias yok, tied embedding.

global_layers N (belge 40 s6.2 Deney G; train.py varsayilani 1, belge 43 s9): son N blok tam causal (model_z_global_mask:
ayni hikayenin butun onceki konumlari, kelime ve Z, gercek sirayla); konum GERCEK hikaye konumu (story_positions =
transformer duzeninin pos'u; mantiksal konumda farkli cumlelerin token'lari ayni konumu paylasir, belge 40 Gorus 4).
Ilk 8 - N blok bugunku gibi.  attn: (yerel, global) ikilisi; onbellek global bloklarda butun gecmisin K/V'sini tutar.

Torba (--bag_k; recipe.Bag): sinav recipe.bag_loss_per_target, uretimde SummaryCache torbayi ozette (BOS / Z_k) secer.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID, EOS_ID, VOCAB, Kind  # noqa: E402

BOS, TOKEN, END, ZTOK, PAD = Kind.BOS, Kind.TOKEN, Kind.END, Kind.ZTOK, Kind.PAD


def rope(x, pos, base=10000.0):
    """x (B, H, T, hd), pos (B, T) -> RoPE uygulanmis x (yarim boyut ciftleri)."""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float) / half)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


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


def model_z_read_mask(kind, doc, sent):
    """Ogrenilen z maskesi: model_z_mask + ZTOK sorgusu kendi cumlesinin token'larini gorur (build_batch Z_k'nin sent'ine
    kendi cumle numarasini yazar).  Z_k cumlenin sonunda: gordugu her token ondan once (belge 35 s1)."""
    def mask_mod(b, h, q, kv):
        kq, kk = kind[b, q], kind[b, kv]
        summary = (kk == BOS) | (kk == ZTOK)
        word = ((kq == TOKEN) | (kq == END) | (kq == ZTOK)) & ((kk == TOKEN) | (kk == END)) & (sent[b, q] == sent[b, kv])
        pad = (kq == PAD) & (kk == PAD)
        return (doc[b, q] == doc[b, kv]) & (kv <= q) & (summary | word | pad)
    return mask_mod


def model_z_global_mask(kind, doc, sent):
    """global_layers bloklari: ayni hikaye ve kv <= q (kelime, BOS, Z hepsi).  Dolgu (doc -1) yalniz dolguyu gorur."""
    def mask_mod(b, h, q, kv):
        return (doc[b, q] == doc[b, kv]) & (kv <= q)
    return mask_mod


def story_positions(kind):
    """(B, T) gercek hikaye ici konum: sutun - hikayenin BOS sutunu (= transformer duzeninin pos'u; Z_k END'in yerinde).
    Dolgu son BOS'tan sayar (dolgu yalniz dolguyu gorur)."""
    col = torch.arange(kind.shape[1], device=kind.device).expand_as(kind)
    return col - torch.where(kind == BOS, col, torch.zeros_like(col)).cummax(1).values


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
            bs = attn.BLOCK_SIZE[0]                     # 128 disinda varsayilan cekirdek hata veriyor (belge 37)
            a = flex_attention(q, k, v, block_mask=attn, kernel_options=None if bs == 128 else dict.fromkeys(
                ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"), bs))
        return self._finish(x, a)

class SentenceTransformer(torch.nn.Module):
    def __init__(self, d=512, layers=8, heads=8, global_layers=0):
        super().__init__()
        self.global_layers = int(global_layers)
        assert 0 <= self.global_layers <= layers, "global_layers 0..layers"
        self.mask_fn = (model_z_read_mask, model_z_global_mask) if self.global_layers else model_z_read_mask
        self.END, self.EOS = END_ID, EOS_ID
        hidden = -(-int(8 * d / 3) // 8) * 8
        self.E = torch.nn.Embedding(VOCAB, d)                    # kurma sirasi E, blocks, norm (ilk agirlik bunu izler)
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                torch.nn.init.normal_(p, std=std)

    def _batch_hidden(self, batch, attn=None):
        """PackedBatch (belge 21: tokens, kind, pos, doc, sent) -> h; attn yoksa dense maske.  Z_k girdisi E(END), okuma
        maskeden.  global_layers: attn (yerel, global) ikilisi; tek maske DURUR (sessiz yanlis yok)."""
        B, T = batch.tokens.shape
        dev = batch.tokens.device
        x = self.E(torch.where(batch.kind == ZTOK, torch.full_like(batch.tokens, END_ID), batch.tokens))
        if not self.global_layers:
            if attn is None:
                attn = _dense(self.mask_fn(batch.kind, batch.doc, batch.sent), B, T, dev)
            assert not isinstance(attn, tuple), "global_layers 0: tek maske"
            for block in self.blocks:
                x = block(x, batch.pos, attn)
            return self.norm(x)
        if attn is None:
            attn = tuple(_dense(f(batch.kind, batch.doc, batch.sent), B, T, dev) for f in self.mask_fn)
        assert isinstance(attn, tuple) and len(attn) == 2, "global_layers: attn (yerel, global) ikilisi olmali"
        real, first = story_positions(batch.kind), len(self.blocks) - self.global_layers
        for l, block in enumerate(self.blocks):
            x = block(x, real, attn[1]) if l >= first else block(x, batch.pos, attn[0])
        return self.norm(x)

    def _logits(self, h, inbag=None):
        """h (..., d) -> cikis: torbasiz h @ E^T; torbali iki asamali log p (inbag: torba, recipe.two_stage_logprobs)."""
        lg = h @ self.E.weight.T
        if not hasattr(self, "bag"):
            return lg
        from recipe import two_stage_logprobs
        return two_stage_logprobs(lg, inbag, h @ self.bag.other)

    def loss_per_target(self, batch, attn=None, chunk=4096):
        """-> nll (K,), pred (K,), target_kind (K,) (hedefli konumlar, satir sirasiyla; belge 21 s7 sozlesmesi)."""
        h = self._batch_hidden(batch, attn)
        keep = batch.target >= 0
        if hasattr(self, "bag"):
            from recipe import bag_loss_per_target
            return (*bag_loss_per_target(self, batch, h, chunk), batch.target_kind[keep])
        hk, tgt = h[keep], batch.target[keep]
        nll = torch.empty(len(tgt), device=h.device)
        pred = torch.empty(len(tgt), dtype=torch.long, device=h.device)
        for r in range(0, len(tgt), chunk):
            lg = (hk[r:r + chunk] @ self.E.weight.T).float()
            nll[r:r + chunk] = F.cross_entropy(lg, tgt[r:r + chunk], reduction="none")
            pred[r:r + chunk] = lg.argmax(-1)
        return nll, pred, batch.target_kind[keep]

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None, open_last=False, on_token=None):
        """prompts: hikaye basina istem cumleleri (token listeleri, END yok) -> her istem icin (uretilen cumleler,
        END ile bitti mi listesi, eos).  generator None: acgozlu, yoksa ornekleme.  SummaryCache ile, istem basina.
        Cumle en cok max_tokens token; kesilen cumle ended False ile doner (belge 26 B2).  Istem tek ileri geciste
        (SummaryCache.prefill, belge 46), uretim token token.  open_last (belge 48): son istem cumlesi kapanmaz, ilk
        uretilen cumle onun devami (yalniz devam token'lari; max_tokens onlara).  on_token(w): her uretilen
        token'dan sonra, on_token(None): cumle kapaninca (akan yazim; cikti degismez)."""
        out = []
        for sents in prompts:
            cache = SummaryCache(self)
            opened = bool(open_last and sents)
            logits = cache.prefill(sents[:-1] if opened else sents)
            for t in (sents[-1] if opened else ()):
                logits = cache.append_token(t)
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
                    if on_token is not None:
                        on_token(w)
                    logits = cache.append_token(w)
                if eos:
                    break
                gen.append(cur)
                ended.append(done)
                if on_token is not None:
                    on_token(None)
                opened = False
                logits = cache.close_sentence()
            out.append((gen, ended, eos))
        return out


class SummaryCache:
    """Tek hikayenin KV onbellegi (22 s4, 35 s4): ozet (BOS + Z_1..Z_k; kalici) ve cumle (simdiki cumlenin token'lari).
    Konumlar kendiliginden: token k+i, Z_k k.  Her cagri sonraki token'in logit'ini dondurur (self.logits).  Z_k ozet +
    kendi cumlesinin onbellegine bakar, K/V'si ozete yazilir, cumle onbellegi ANCAK sonra silinir.  global_layers
    bloklari: butun gecmisin K/V'si (all_k / all_v), gercek konumla (self.t: BOS 0, her token ve Z +1).  prefill: istem
    tek ileri geciste (egitimin maskeleri ve konumlari), onbellek token token yolla ayni duruma gelir.  Torbali model:
    torba ozette (BOS, Z_k) secilir, P = kapanmis cumlelerin token'lari (past)."""

    def __init__(self, model):
        self.m = model
        self.dev = model.E.weight.device
        L = len(model.blocks)
        self.sum_k, self.sum_v = [None] * L, [None] * L
        self.sen_k, self.sen_v = [None] * L, [None] * L
        self.all_k, self.all_v = [None] * L, [None] * L
        self.first_global = L - model.global_layers
        self.n_z, self.i, self.t = 0, 0, 0
        self.past, self.cur, self.inbag = [], [], None
        x = model.E(torch.tensor([[EOS_ID]], device=self.dev))              # BOS = EOS token'i (belge 21 s1)
        self.logits = self._out(self._step(x, 0, read_sentence=False, summary=True), summary=True)

    def _out(self, hn, summary):
        """Son konumun normlu durumu -> logit (torbali: ozette torba yeniden secilir)."""
        if summary and hasattr(self.m, "bag"):
            self.inbag = self.m.bag.select_one(hn[0, -1], self.m.E.weight, self.past)
        return self.m._logits(hn, self.inbag)[0, -1]

    def _step(self, x, pos, read_sentence, summary):
        """read_sentence: cumle onbellegini de gor; summary: K/V ozete (yoksa cumle onbellegine) yazilir.  Global
        bloklar: gercek konum self.t, butun gecmis."""
        p = torch.tensor([[pos]], device=self.dev)
        for l, block in enumerate(self.m.blocks):
            if l >= self.first_global:
                q, k, v = block._qkv(x, torch.tensor([[self.t]], device=self.dev))
                self.all_k[l] = k if self.all_k[l] is None else torch.cat([self.all_k[l], k], 2)
                self.all_v[l] = v if self.all_v[l] is None else torch.cat([self.all_v[l], v], 2)
                x = block._finish(x, F.scaled_dot_product_attention(q, self.all_k[l], self.all_v[l]))
                continue
            q, k, v = block._qkv(x, p)
            ks = [c for c in (self.sum_k[l], self.sen_k[l] if read_sentence else None) if c is not None] + [k]
            vs = [c for c in (self.sum_v[l], self.sen_v[l] if read_sentence else None) if c is not None] + [v]
            a = F.scaled_dot_product_attention(q, torch.cat(ks, 2), torch.cat(vs, 2))
            if summary:
                self.sum_k[l] = k if self.sum_k[l] is None else torch.cat([self.sum_k[l], k], 2)
                self.sum_v[l] = v if self.sum_v[l] is None else torch.cat([self.sum_v[l], v], 2)
            else:
                self.sen_k[l] = k if self.sen_k[l] is None else torch.cat([self.sen_k[l], k], 2)
                self.sen_v[l] = v if self.sen_v[l] is None else torch.cat([self.sen_v[l], v], 2)
            x = block._finish(x, a)
        return self.m.norm(x)

    @torch.no_grad()
    def prefill(self, sents):
        """Yeni onbellek: BOS + istem cumleleri (her biri Z_k ile kapanir) tek ileri geciste -> sonraki token'in logit'i.
        Yerel bloklar model_z_read_mask + mantiksal konum, global bloklar tam causal + gercek konum (egitimle ayni dense
        maske); ozet = BOS + Z'lerin K/V'si, global = butun konumlar; cumle onbellegi bos (son Z_k'den sonra)."""
        assert self.t == 0 and self.n_z == 0 and self.i == 0, "prefill yalniz yeni onbellekte"
        tok, kind, pos, sent = [EOS_ID], [BOS], [0], [-1]
        for k, x in enumerate(sents):
            tok += list(x) + [END_ID]                                       # Z_k girdisi E(END)
            kind += [TOKEN] * len(x) + [ZTOK]
            pos += [k + 1 + i for i in range(len(x))] + [k + 1]
            sent += [k] * (len(x) + 1)
        T = len(tok)
        t = lambda v: torch.tensor([v], device=self.dev)  # noqa: E731
        kind, sent, doc = t(kind), t(sent), torch.zeros(1, T, dtype=torch.long, device=self.dev)
        local = _dense(model_z_read_mask(kind, doc, sent), 1, T, self.dev)[:, None]
        causal = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
        summ = ((kind == BOS) | (kind == ZTOK))[0]
        x, lpos, real = self.m.E(t(tok)), t(pos), torch.arange(T, device=self.dev)[None]
        for l, block in enumerate(self.m.blocks):
            g = l >= self.first_global
            q, k, v = block._qkv(x, real if g else lpos)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal if g else local)
            if g:
                self.all_k[l], self.all_v[l] = k, v
            else:
                self.sum_k[l], self.sum_v[l] = k[:, :, summ], v[:, :, summ]
            x = block._finish(x, a)
        self.n_z, self.i, self.t = len(sents), 0, T - 1
        self.sen_k = [None] * len(self.sen_k)
        self.sen_v = [None] * len(self.sen_v)
        self.past = [t_ for x_ in sents for t_ in x_]
        self.logits = self._out(self.m.norm(x), summary=True)
        return self.logits

    @torch.no_grad()
    def append_token(self, token):
        """Simdiki cumleye token -> sonraki token'in logit'i."""
        self.i += 1
        self.t += 1
        self.cur.append(token)
        x = self.m.E(torch.tensor([[token]], device=self.dev))
        self.logits = self._out(self._step(x, self.n_z + self.i, read_sentence=True, summary=False), summary=False)
        return self.logits

    @torch.no_grad()
    def close_sentence(self):
        """Cumle bitti: Z_k (girdi E(END), cumleyi okur) ozet onbellegine, cumle onbellegi silinir -> sonraki cumlenin ilk
        token'i (ya da EOS) logit'i."""
        self.n_z += 1
        self.i = 0
        self.t += 1
        x = self.m.E(torch.tensor([[END_ID]], device=self.dev))
        self.past, self.cur = self.past + self.cur, []
        self.logits = self._out(self._step(x, self.n_z, read_sentence=True, summary=True), summary=True)
        self.sen_k = [None] * len(self.sen_k)                               # Z_k'den SONRA
        self.sen_v = [None] * len(self.sen_v)
        return self.logits

