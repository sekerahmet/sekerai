"""sentence (V2) -- SentenceTransformer (Model Z), GPT-2 token.  Tarif belge/model_z_temel/22 (duzen, maske, onbellek),
35 (ogrenilen z), 40 / 43 (global_layers); alanlar belge 21 (PackedBatch).  Kaldirilan secenekler ve eski kodun git
etiketleri: belge 44, 77 (train._archived iletisi).

Hikaye tek dizi; cumle k'nin END'inin yerinde Z_k (girdi E(END), transformer'in END konumuyla ayni girdi), dizi boyu
transformer akisiyla ayni, hedefler konum konum ayni:
    girdi   BOS  t_11 .. t_1L  [Z_1]  t_21 .. t_2L  [Z_2] ...  [Z_n]
    hedef   t_11 t_12 .. END   t_21   t_22 .. END   t_31  ...  EOS
    konum   0    1    .. L     1      2    ..       2     ...  n     (Z_k k; cumle k, token i: k-1+i)
Maske (model_z_read_mask, tek mask_mod -> dense ya da FlexAttention BlockMask): ayni hikaye, kv <= q, ve kv BOS/ZTOK ya
da (q, kv ayni cumlenin token'i; Z_k kendi cumlesinin token'larini da gorur); aralik bicimi model_z_read_bounds ile
(belge 65 (a)).  Z_k'nin her katmandaki hali sonraki
cumlelerin K/V'si (belge 35 (b)).  model_z_mask: okumasiz hali (teshis: z_ablate read_off).  Uretim: SummaryCache (ozet
onbellegi BOS + Z'ler kalici, cumle onbellegi cumle bitince silinir).  Tarif (Llama sinifi): pre-norm RMSNorm, RoPE,
QK-norm (fp32), SwiGLU, bias yok, tied embedding.

global_layers N (belge 40 s6.2 Deney G; train.py varsayilani auto = round(L / 3)): son N blok tam causal
(model_z_global_mask: ayni hikayenin butun onceki konumlari, kelime ve Z, gercek sirayla); konum GERCEK hikaye konumu
(story_positions = transformer duzeninin pos'u; mantiksal konumda farkli cumlelerin token'lari ayni konumu paylasir, belge 40
Gorus 4).  Ilk L - N blok yerel.  attn: (yerel, global) ikilisi; onbellek global bloklarda butun gecmisin K/V'sini tutar.

summaries_last (belge 66; kullanici, 8 Ekim: "Fikrine onay verdim"): satir bellekte [token'lar | ozetler | dolgu], her grupta
eski sira; model ayni (attention disi konum konum, RoPE pos'tan, maske iliskiden); maske sorgu basina iki aralik
(model_z_summaries_last_ranges).  Batch real_pos tasiyorsa bu duzen; egitim ve sinav hep bu duzende, Z'ler arada duzen
teshis / prefill yolunda (ayni hesap).
Sozluk dolgusu (belge 89; kullanici, 8 Ekim: "sözlük dolgusu ok, ekle"): vocab_rows > VOCAB ise E'nin dolgu satirlari sifir,
cikis (_logits) VOCAB'a kesilir: dolgu token'i hedef olmaz, uretilmez.
"""
import dataclasses
import functools
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID, EOS_ID, MAX_SENTENCE_TOKENS, ROW_LEN, VOCAB, Kind  # noqa: E402

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


def model_z_read_bounds(kind):
    """kind (B, T) -> (lo, ds, summ): sorgu basina aralik (belge 65 (a)).  ds = hikayenin (dolguda dolgu kosusunun) ilk
    sutunu, lo = parcanin ilk sutunu (son ozetten sonraki; BOS'ta kendisi, Z_k'da cumlesinin ilk token'i), summ = BOS |
    ZTOK.  build_batch duzenine dayanir: hikaye BOS ile baslar, hikayeler bitisik, dolgu yalniz satir sonunda tek kosu,
    cumle siniri Z_k (tests_model_z 'mask' gercek SS / FineWeb satirlariyla eski formule esitligi sinar)."""
    B, T = kind.shape
    col = torch.arange(T, device=kind.device).expand(B, T)
    summ = (kind == BOS) | (kind == ZTOK)
    pad = kind == PAD
    first_pad = pad & ~torch.cat([torch.zeros_like(pad[:, :1]), pad[:, :-1]], 1)
    after_summ = torch.cat([torch.zeros_like(summ[:, :1]), summ[:, :-1]], 1) & ~pad
    zero = torch.zeros_like(col)
    ds = torch.where((kind == BOS) | first_pad, col, zero).cummax(1).values
    lo = torch.where((kind == BOS) | first_pad | after_summ, col, zero).cummax(1).values
    return lo.int(), ds.int(), summ


def model_z_read_mask(kind, doc, sent):
    """Ogrenilen z maskesi: model_z_mask + ZTOK sorgusu kendi cumlesinin token'larini gorur.  Aralik bicimi (belge 65
    (a); sorgu basina lo / ds, kv tarafinda yalniz ozet biti): kv <= q ve (kv >= lo[q] ya da kv ozet ve kv >= ds[q]).
    Dolgu kuralini icerir (includes_padding): dolgu kendi kosusunu gorur, gercek konum dolguyu gormez.  doc, sent imza
    icin (model_z_read_bounds yalniz kind'den)."""
    lo, ds, summ = model_z_read_bounds(kind)

    def mask_mod(b, h, q, kv):
        return (kv <= q) & ((kv >= lo[b, q]) | (summ[b, kv] & (kv >= ds[b, q])))
    return mask_mod


def model_z_global_mask(kind, doc, sent):
    """global_layers bloklari: ayni hikaye ve kv <= q (kelime, BOS, Z hepsi); aralik bicimi kv >= ds[q] (belge 65 (a)).
    Dolgu yalniz kendi kosusunu gorur (includes_padding)."""
    _, ds, _ = model_z_read_bounds(kind)

    def mask_mod(b, h, q, kv):
        return (kv <= q) & (kv >= ds[b, q])
    return mask_mod


model_z_read_mask.includes_padding = model_z_global_mask.includes_padding = True   # recipe sarmaz (belge 65 olcumu)


def story_positions(kind):
    """(B, T) gercek hikaye ici konum: sutun - hikayenin BOS sutunu (= transformer duzeninin pos'u; Z_k END'in yerinde).
    Dolgu son BOS'tan sayar (dolgu yalniz dolguyu gorur)."""
    col = torch.arange(kind.shape[1], device=kind.device).expand_as(kind)
    return col - torch.where(kind == BOS, col, torch.zeros_like(col)).cummax(1).values


def summaries_last(batch):
    """PackedBatch (build_batch duzeni) -> (ayni batch [token'lar | ozetler | dolgu] sirasinda, perm (B, T): yeni sutundaki
    eski sutun).  Konum basina butun alanlar (tokens, kind, pos, doc, sent, target, target_kind) birlikte tasinir; gercek
    hikaye konumu once hesaplanip real_pos olarak eklenir (glob katmanlari; carry batch'i getirdiyse onunki).  carry
    bellegi: mem_cols kaynak satirin yeni sutununa."""
    kind = batch.kind
    B, T = kind.shape
    group = torch.where(kind == PAD, 2, torch.where((kind == BOS) | (kind == ZTOK), 1, 0))
    perm = torch.argsort(group * T + torch.arange(T, device=kind.device), dim=1)
    g = lambda t: t.gather(1, perm)  # noqa: E731
    mem = {}
    if batch.mem_rows is not None:                                       # eski sutun -> yeni sutun (kaynak satirda)
        inv, r = torch.argsort(perm, 1), batch.mem_rows.clamp_min(0)
        mem = dict(mem_cols=torch.where(batch.mem_rows >= 0, inv[r, batch.mem_cols.clamp_min(0)], -1))
    return dataclasses.replace(batch, tokens=g(batch.tokens), kind=g(kind), pos=g(batch.pos), doc=g(batch.doc),
                               sent=g(batch.sent), target=g(batch.target), target_kind=g(batch.target_kind),
                               real_pos=g(story_positions(kind) if batch.real_pos is None else batch.real_pos), **mem), perm


def model_z_summaries_last_ranges(kind, doc, sent, glob=False, n_mem=None):
    """summaries_last duzeninde (kind, doc, sent permute) yerel (glob False) ya da global maske -> mask_mod: sorgu basina
    iki aralik, token [a0, a1] ve ozet [b0, b1] (dolgu: kendi kosusu).  Token'lar (hikaye, cumle), ozetler (hikaye, cumle +
    1) sirasinda (BOS 0, Z_k k + 1): sinirlar searchsorted ile.  Yerel: token kendi cumlesinin basindan kendisine, Z_k kendi
    cumlesinin token'lari, BOS hic; global: hikayenin ilk token'indan.  Ozet: hikayenin BOS'undan, token'da Z_(k-1)'e, Z_k'da
    kendisine.  n_mem (B,) (carry bellegi, belge 83):
    satirin ilk hikayesinin sorgulari ucuncu araligi [T, T + n_mem) gorur (glob'da BOS yuvasi haric, [T + 1, ...)); anahtar
    uzunlugu T + M.  Dolgu kuralini icerir (includes_padding)."""
    B, T = kind.shape
    big, large = T + 2, 1 << 40
    tok, summ, pad = kind == TOKEN, (kind == BOS) | (kind == ZTOK), kind == PAD
    col = torch.arange(T, device=kind.device).expand(B, T)
    d, s = doc.long() * big, sent.long()
    ktok = torch.where(tok, d + s, large)                                   # artan: token'lar, sonra buyuk
    ksum = torch.where(summ, d + s + 1, torch.where(tok, -1, large))        # artan: -1, ozetler, buyuk
    a0 = torch.searchsorted(ktok, d if glob else d + s)
    a1 = torch.where(tok, col, torch.searchsorted(ktok, d + s, right=True) - 1)
    b0 = torch.searchsorted(ksum, d)
    b1 = torch.searchsorted(ksum, d + s + summ.long(), right=True) - 1
    a0 = torch.where(pad, (~pad).sum(1, keepdim=True).expand(B, T), a0)
    a1 = torch.where(pad, col, a1)
    b0, b1 = torch.where(pad, 1, b0), torch.where(pad, 0, b1)
    a0, a1, b0, b1 = (t.int() for t in (a0, a1, b0, b1))

    if n_mem is None:
        def mask_mod(b, h, q, kv):
            return ((kv >= a0[b, q]) & (kv <= a1[b, q])) | ((kv >= b0[b, q]) & (kv <= b1[b, q]))
        return mask_mod
    has = (doc == 0) & ~pad & (n_mem[:, None] > 0)
    c0 = torch.where(has, T + int(glob), T + 1).int()
    c1 = torch.where(has, T + n_mem[:, None].expand(B, T) - 1, T).int()

    def mask_mod(b, h, q, kv):
        return ((kv >= a0[b, q]) & (kv <= a1[b, q])) | ((kv >= b0[b, q]) & (kv <= b1[b, q])) | (
            (kv >= c0[b, q]) & (kv <= c1[b, q]))
    return mask_mod


_LAST_GLOB = functools.partial(model_z_summaries_last_ranges, glob=True)
model_z_summaries_last_ranges.includes_padding = _LAST_GLOB.includes_padding = True


def _dense(mask_mod, B, T, device):
    """mask_mod -> bool (B, T, T) (SDPA yolu; CPU egitimi ve testler)."""
    b = torch.arange(B, device=device)[:, None, None]
    q = torch.arange(T, device=device)[None, :, None]
    kv = torch.arange(T, device=device)[None, None, :]
    return mask_mod(b, 0, q, kv)


def gqa_sdpa(q, k, v, mask=None):
    """SDPA; GQA'da (k / v daha az head) fused cekirdege giden yol (belge 74: enable_gqa + maske math'a dusuyordu, fp32
    B x H x Tq x S).  Esit head: dogrudan SDPA (bit ayni).  Tq 1: grup Tq'ya katlanir (q (B, Hkv, G, hd), maske satirca
    yayinlanir; K / V bir kez okunur); Tq > 1: k / v head boyunca gecici genisletilir (onbellek kucuk kalir)."""
    H, Hkv = q.shape[1], k.shape[1]
    if H == Hkv:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    B, _, Tq, hd = q.shape
    if Tq == 1 and (mask is None or mask.shape[-2] == 1 and (mask.dim() < 3 or mask.shape[-3] == 1)):
        return F.scaled_dot_product_attention(q.reshape(B, Hkv, H // Hkv, hd), k, v, attn_mask=mask).reshape(B, H, 1, hd)
    return F.scaled_dot_product_attention(q, k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1),
                                          attn_mask=mask)


class Block(torch.nn.Module):
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
        with torch.autocast(x.device.type, enabled=False):                  # QK-norm fp32 (V2 tarifi, belge 20 s4)
            q, k = self.q_norm(q.float()).to(v.dtype), self.k_norm(k.float()).to(v.dtype)
        return rope(q, pos), rope(k, pos), v

    def _finish(self, x, a):
        B, T, d = x.shape
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        g, u = self.gate_up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)

    def forward(self, x, pos, attn):
        """attn: None (duz causal), bool (B, T, T) (dense) ya da FlexAttention BlockMask; carry: (maske, mem_rows,
        mem_cols) -> K / V = [satir || ayni katmanda kaynak satirlarin bellek sutunlari] (anahtar T + M)."""
        mem = None
        if isinstance(attn, tuple):
            attn, mem = attn[0], attn[1:]
        q, k, v = self._qkv(x, pos)
        if mem is not None:
            r, c = mem[0].clamp_min(0), mem[1].clamp_min(0)                 # bos yuva maskeyle kapali
            k = torch.cat([k, k[r, :, c].permute(0, 2, 1, 3)], 2)
            v = torch.cat([v, v[r, :, c].permute(0, 2, 1, 3)], 2)
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

class SentenceTransformer(torch.nn.Module):
    def __init__(self, d=512, layers=8, heads=8, global_layers=0, glob_kv_heads=None, carry_group=0, vocab_rows=VOCAB):
        """Bloklar: yerel (model_z_read_mask) x (layers - global_layers), sonda glob (tam causal) x global_layers.
        glob_kv_heads: glob bloklarinda k / v head sayisi (GQA; uretimde buyuk onbellek yalniz glob'ta), yerel bloklar tam
        head.  carry_group G (belge 83; agirlik degismez): uretimde (SummaryCache) parca row_len'e
        dolunca glob onbelleginde yalniz Z'ler kalir, G parcada sifirlanir.  vocab_rows: E satir sayisi (>= VOCAB; dolgu
        satirlari sifir, ilk agirlik VOCAB'li modelle ayni)."""
        super().__init__()
        self.carry_group, self.row_len = int(carry_group), ROW_LEN
        self.global_layers = int(global_layers)
        assert 0 <= self.global_layers <= layers, "global_layers 0..layers"
        self.mask_fn = (model_z_read_mask, model_z_global_mask) if self.global_layers else model_z_read_mask
        self.END, self.EOS = END_ID, EOS_ID
        hidden = -(-int(8 * d / 3) // 8) * 8
        assert vocab_rows >= VOCAB, "vocab_rows >= VOCAB"
        self.E = torch.nn.Embedding(VOCAB, d)               # kurma sirasi E, blocks, norm (ilk agirlik bunu izler)
        if vocab_rows > VOCAB:                                   # kurulum RNG'si VOCAB'li modelle ayni; dolgu sifir
            self.E = torch.nn.Embedding(vocab_rows, d, _weight=torch.zeros(vocab_rows, d))
        kinds = ["loc"] * (layers - self.global_layers) + ["glob"] * self.global_layers
        self.blocks = torch.nn.ModuleList(Block(d, heads, hidden, glob_kv_heads if k == "glob" else None) for k in kinds)
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "down.weight")) else 0.02
                with torch.no_grad():
                    (p[:VOCAB] if name == "E.weight" else p).normal_(0.0, std)   # dolgu satiri rastgele sayi cekmez
        with torch.no_grad():
            self.E.weight[VOCAB:].zero_()

    def max_positions(self):
        """Uretimde hikaye basina konum siniri: egitim satiri row_len; carry belleginde carry_group parca (belge 89b)."""
        return self.row_len * max(self.carry_group, 1)

    def _masks(self, last=False):
        """Maske fonksiyonlari: bugunku duzen (mask_fn) ya da summaries_last duzeninde ayni yapida (tek / ikili)."""
        if not last:
            return self.mask_fn
        return (model_z_summaries_last_ranges, _LAST_GLOB) if isinstance(self.mask_fn, tuple) else \
            model_z_summaries_last_ranges

    def _batch_hidden(self, batch, attn=None):
        """PackedBatch (belge 21: tokens, kind, pos, doc, sent) -> h; attn yoksa dense maske.  Z_k girdisi E(END), okuma
        maskeden.  global_layers: attn (yerel, global) ikilisi; tek maske DURUR (sessiz yanlis yok).  batch.real_pos varsa
        summaries_last duzeni: maskeler _masks(True), glob konumu real_pos."""
        B, T = batch.tokens.shape
        dev = batch.tokens.device
        last = getattr(batch, "real_pos", None) is not None
        mfn = self._masks(last)
        x = self.E(torch.where(batch.kind == ZTOK, torch.full_like(batch.tokens, END_ID), batch.tokens))
        mem = getattr(batch, "mem_rows", None) is not None              # carry bellegi (belge 83)
        if mem and attn is None:
            import recipe as R
            attn = tuple(R.dense_mask(batch, f) for f in mfn) if isinstance(mfn, tuple) else R.dense_mask(batch, mfn)
        w = (lambda a: (a, batch.mem_rows, batch.mem_cols)) if mem else (lambda a: a)  # noqa: E731
        if not self.global_layers:
            if attn is None:
                attn = _dense(mfn(batch.kind, batch.doc, batch.sent), B, T, dev)
            assert not isinstance(attn, tuple), "global_layers 0: tek maske"
            for block in self.blocks:
                x = block(x, batch.pos, w(attn))
            return self.norm(x)
        if attn is None:
            attn = tuple(_dense(f(batch.kind, batch.doc, batch.sent), B, T, dev) for f in mfn)
        assert isinstance(attn, tuple) and len(attn) == 2, "global_layers: attn (yerel, global) ikilisi olmali"
        real = batch.real_pos if last else story_positions(batch.kind)
        first = len(self.blocks) - self.global_layers
        for l, block in enumerate(self.blocks):
            x = block(x, real, w(attn[1])) if l >= first else block(x, batch.pos, w(attn[0]))
        return self.norm(x)

    def _logits(self, h):
        """h (..., d) -> cikis h @ E^T (bagli embedding), VOCAB sutun (sozluk dolgusu kesilir)."""
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

    def _row_rule(self, w, opened, used, limit, max_sentences, max_tokens, stop_when):
        """generate'in tek istem kurali (birebir), adim adim: w ilk argmax; ('tok' token, False) / (END, True) verir ve
        sonraki argmax'i alir; bitince (cumleler, ended, eos) dondurur (toplu uretim, belge 91)."""
        gen, ended, eos, full = [], [], False, False
        while len(gen) < max_sentences:
            cur, done = [], False
            while len(cur) < max_tokens:
                if used + 2 > limit:
                    full = True
                    break
                if w == self.EOS and not cur and not opened:
                    eos = True
                    break
                if w in (self.END, self.EOS):
                    done = True
                    break
                cur.append(w)
                w = yield w, False
                used += 1
            if eos or (full and not cur):
                break
            gen.append(cur)
            ended.append(done)
            if full or (stop_when is not None and stop_when(gen)):
                break
            opened = False
            w = yield END_ID, True
            used += 1
        return gen, ended, eos

    def _generate_rows(self, prompts, max_sentences, max_tokens, open_last, stop_when):
        """Acgozlu toplu uretim (belge 91): istemler tek StaticCache'te satir satir, adim basina tek replay; satir kurali
        _row_rule (= tek istem generate); biten satir canli degil.  -> generate ciktisi (istem sirasiyla)."""
        limit = self.max_positions()
        opened = [bool(open_last and s) for s in prompts]
        pre = [s[:-1] if o else s for s, o in zip(prompts, opened)]
        opens = [list(s[-1]) if o else [] for s, o in zip(prompts, opened)]
        n_open = max(len(o) for o in opens)
        cache = StaticCache(self, max(n_open + 1, min(n_open + max_sentences * (max_tokens + 1) + 1, limit - min(
            1 + sum(len(x) + 1 for x in p) for p in pre))), max_sentences + 1, n_open + max_tokens + 1)
        am = cache.prefill_rows(pre, opens).float().argmax(-1).tolist()
        rules, act, res = [], [], [None] * len(prompts)
        for r, s in enumerate(prompts):
            g = self._row_rule(am[r], opened[r], 1 + sum(len(x) + 1 for x in s) - opened[r], limit, max_sentences,
                               max_tokens, stop_when)
            rules.append(g)
            try:
                act.append(next(g))
            except StopIteration as e:
                act.append(None)
                res[r] = e.value
        while any(a is not None for a in act):
            live = [a is not None for a in act]
            am = cache.step_rows([a[0] if a else 0 for a in act], [bool(a and a[1]) for a in act], live)
            am = am.float().argmax(-1).tolist()
            for r, g in enumerate(rules):
                if act[r] is None:
                    continue
                try:
                    act[r] = g.send(am[r])
                except StopIteration as e:
                    act[r] = None
                    res[r] = e.value
        return res

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None, open_last=False, on_token=None,
                 stop_when=None, batch_size=32):
        """prompts: hikaye basina istem cumleleri (token listeleri, END yok) -> her istem icin (uretilen cumleler,
        END ile bitti mi listesi, eos).  generator None: acgozlu, yoksa ornekleme.  SummaryCache ile, istem basina.
        Cumle en cok max_tokens token; kesilen cumle ended False ile doner (belge 26 B2).  Istem tek ileri geciste
        (SummaryCache.prefill, belge 46), uretim token token.  open_last (belge 48): son istem cumlesi kapanmaz, ilk
        uretilen cumle onun devami (yalniz devam token'lari; max_tokens onlara).  on_token(w): her uretilen
        token'dan sonra, on_token(None): cumle kapaninca (akan yazim; cikti degismez).  stop_when(gen): her
        kapanan cumleden sonra, True ise o istemin uretimi biter (cikti = tam uretimin oneki).  Onbellek StaticCache (belge
        84; STATIC_DECODE, carry'siz model), yoksa SummaryCache; fp32'de token token ayni.  Konum siniri (belge 89b): hikaye
        max_positions()'i (egitim satiri) asmaz; sinira gelince o anki cumle kesik (ended False) doner, uretim biter.
        batch_size (belge 91): acgozlu, on_token'siz ve StaticCache'li uretimde istemler bu boyda toplu (_generate_rows);
        cikti tek tek uretimle ayni (fp32 token token)."""
        out, limit = [], self.max_positions()
        static = STATIC_DECODE and StaticCache.supports(self)
        if static and generator is None and on_token is None and batch_size > 1 and len(prompts) > 1:
            for i in range(0, len(prompts), batch_size):
                out += self._generate_rows(prompts[i:i + batch_size], max_sentences, max_tokens, open_last, stop_when)
            return out
        for sents in prompts:
            opened = bool(open_last and sents)
            n_open = len(sents[-1]) if opened else 0
            t0 = 1 + sum(len(s) + 1 for s in (sents[:-1] if opened else sents))   # prefill sonrasi konum
            cache = StaticCache(self, max(n_open + 1, min(n_open + max_sentences * (max_tokens + 1) + 1, limit - t0)),
                                max_sentences + 1, n_open + max_tokens + 1) if static else SummaryCache(self)
            logits = cache.prefill(sents[:-1] if opened else sents)
            for t in (sents[-1] if opened else ()):
                logits = cache.append_token(t)
            used = 1 + sum(len(s) + 1 for s in sents) - opened                 # BOS + token'lar + Z'ler (acik cumlede Z yok)
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
                    logits = cache.append_token(w)
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
                logits = cache.close_sentence()
                used += 1
            out.append((gen, ended, eos))
        return out


class SummaryCache:
    """Tek hikayenin KV onbellegi (22 s4, 35 s4): ozet (BOS + Z_1..Z_k; kalici) ve cumle (simdiki cumlenin token'lari).
    Konumlar kendiliginden: token k+i, Z_k k.  Her cagri sonraki token'in logit'ini dondurur (self.logits).  Z_k ozet +
    kendi cumlesinin onbellegine bakar, K/V'si ozete yazilir, cumle onbellegi ANCAK sonra silinir.  global_layers
    bloklari: butun gecmisin K/V'si (all_k / all_v), gercek konumla (self.t: BOS 0, her token ve Z +1).  prefill: istem
    tek ileri geciste (egitimin maskeleri ve konumlari), onbellek token token yolla ayni duruma gelir.  carry (model.carry_group, belge 83): cumle
    kapaninca parca dolu (used > row_len - MAX_SENTENCE_TOKENS - 1) ise yeni parca: glob onbelleginde yalniz Z'ler kalir
    (BOS ve token'lar atilir), konumlar surer; carry_group'uncu parcada onbellek sifirlanir (yeni hikaye, BOS)."""

    def __init__(self, model):
        self.m = model
        self.dev = model.E.weight.device
        L = len(model.blocks)
        self.sum_k, self.sum_v = [None] * L, [None] * L
        self.sen_k, self.sen_v = [None] * L, [None] * L
        self.all_k, self.all_v = [None] * L, [None] * L
        self.glob = [l >= L - model.global_layers for l in range(L)]
        self.n_z, self.i, self.t = 0, 0, 0
        self.piece, self.used, self.gkind = 0, 1, []                       # carry: parca no, parcadaki konum, glob tur
        x = model.E(torch.tensor([[EOS_ID]], device=self.dev))              # BOS = EOS token'i (belge 21 s1)
        self.logits = self._out(self._step(x, 0, read_sentence=False, summary=True), summary=True)

    def _out(self, hn, summary):
        """Son konumun normlu durumu -> logit."""
        return self.m._logits(hn)[0, -1]

    def _step(self, x, pos, read_sentence, summary):
        """read_sentence: cumle onbellegini de gor; summary: K/V ozete (yoksa cumle onbellegine) yazilir.  Global
        bloklar: gercek konum self.t, butun gecmis."""
        p = torch.tensor([[pos]], device=self.dev)
        self.gkind.append(ZTOK if summary and read_sentence else BOS if summary else TOKEN)
        for l, block in enumerate(self.m.blocks):
            if self.glob[l]:
                q, k, v = block._qkv(x, torch.tensor([[self.t]], device=self.dev))
                self.all_k[l] = k if self.all_k[l] is None else torch.cat([self.all_k[l], k], 2)
                self.all_v[l] = v if self.all_v[l] is None else torch.cat([self.all_v[l], v], 2)
                x = block._finish(x, gqa_sdpa(q, self.all_k[l], self.all_v[l]))
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
        if self.m.carry_group:                                           # carry: parca sinirlari token token yolda
            for x_ in sents:
                for t_ in x_:
                    self.append_token(t_)
                self.close_sentence()
            return self.logits
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
            g = self.glob[l]
            q, k, v = block._qkv(x, real if g else lpos)
            a = gqa_sdpa(q, k, v, causal if g else local)
            if g:
                self.all_k[l], self.all_v[l] = k, v
            else:
                self.sum_k[l], self.sum_v[l] = k[:, :, summ], v[:, :, summ]
            x = block._finish(x, a)
        self.n_z, self.i, self.t = len(sents), 0, T - 1
        self.sen_k = [None] * len(self.sen_k)
        self.sen_v = [None] * len(self.sen_v)
        self.logits = self._out(self.m.norm(x), summary=True)
        return self.logits

    @torch.no_grad()
    def append_token(self, token):
        """Simdiki cumleye token -> sonraki token'in logit'i."""
        self.i += 1
        self.t += 1
        x = self.m.E(torch.tensor([[token]], device=self.dev))
        self.logits = self._out(self._step(x, self.n_z + self.i, read_sentence=True, summary=False), summary=False)
        return self.logits

    @torch.no_grad()
    def close_sentence(self):
        """Cumle bitti: Z_k (girdi E(END), cumleyi okur) ozet onbellegine, cumle onbellegi silinir -> sonraki cumlenin ilk
        token'i (ya da EOS) logit'i."""
        n = self.i                                                          # cumlenin token sayisi
        self.n_z += 1
        self.i = 0
        self.t += 1
        x = self.m.E(torch.tensor([[END_ID]], device=self.dev))
        self.logits = self._out(self._step(x, self.n_z, read_sentence=True, summary=True), summary=True)
        self.sen_k = [None] * len(self.sen_k)                               # Z_k'den SONRA
        self.sen_v = [None] * len(self.sen_v)
        if self.m.carry_group:
            self.used += n + 1
            if self.used > self.m.row_len - MAX_SENTENCE_TOKENS - 1:        # sonraki cumle sigmayabilir: yeni parca
                self.piece += 1
                if self.piece == self.m.carry_group:                        # grup bitti: yeni hikaye (BOS, bellek yok)
                    self.__init__(self.m)
                else:                                                       # glob: yalniz Z'ler (BOS, token'lar atilir)
                    keep = torch.tensor([k == ZTOK for k in self.gkind], device=self.dev)
                    for l, g in enumerate(self.glob):
                        if g:
                            self.all_k[l], self.all_v[l] = self.all_k[l][:, :, keep], self.all_v[l][:, :, keep]
                    self.gkind, self.used = [ZTOK] * int(keep.sum()), 0
        return self.logits


STATIC_DECODE = True       # generate: StaticCache (destekleyen modelde); False: SummaryCache (eski yol, kiyas icin)
_COMPILED = {}             # derlenmis adim (surec basina bir kez; derleme hata verirse derlemesiz graph)


_SPLIT = 1024              # decode_sdpa: anahtar boyu >= 4 x bu ise parcali (split-KV, belge 92); parca boyu


def decode_sdpa(q, k, v, mask, f32_scores=False):
    """Tek sorgu attention (Tq 1; belge 76 R1, belge 84): q (B, H, 1, hd), k / v (B, Hkv, S, hd), mask (B, S) bool (True =
    gorulur) -> (B, H, 1, hd).  Grup katlama (head h -> kv h // G, enable_gqa eslemesi), skor fp32 (f32_scores: bf16'da
    bmm out_dtype), maske, fp32 softmax, @ V.  Maskeli SDPA decode'da split-KV'siz efficient'a dusuyordu (belge 74 s7).
    S >= 4 x _SPLIT ve kati: parcali (belge 92): parca basina skor max / exp toplami / kismi cikti (bmm'ler parca sayisi
    kadar paralel; uzun baglamda G = 4 satirlik bmm GPU'yu dolduramiyordu), sonra birlestirme; ayni matematik (fp32)."""
    B, H, _, hd = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    if S < 4 * _SPLIT or S % _SPLIT:
        qf = (q * hd ** -0.5).reshape(B * Hkv, H // Hkv, hd)
        kt = k.reshape(B * Hkv, S, hd).transpose(1, 2)
        s = torch.bmm(qf, kt, out_dtype=torch.float32) if f32_scores else torch.bmm(qf, kt).float()
        p = torch.softmax(s.view(B, Hkv, -1, S).masked_fill(~mask[:, None, None], float("-inf")), -1).to(v.dtype)
        return torch.matmul(p, v).reshape(B, H, 1, hd)
    G, C = H // Hkv, _SPLIT
    nC = S // C
    N = B * Hkv * nC
    qf = (q * hd ** -0.5).reshape(B, Hkv, 1, G, hd).expand(B, Hkv, nC, G, hd).reshape(N, G, hd)
    kt = k.reshape(N, C, hd).transpose(1, 2)
    s = torch.bmm(qf, kt, out_dtype=torch.float32) if f32_scores else torch.bmm(qf, kt).float()
    s = s.view(B, Hkv, nC, G, C).masked_fill(~mask.view(B, 1, nC, 1, C), float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))           # tamamen maskeli parca: katkisi 0
    e = torch.exp(s - m)
    o = torch.bmm(e.to(v.dtype).view(N, G, C), v.reshape(N, C, hd)).float().view(B, Hkv, nC, G, hd)
    wgt = torch.exp(m - m.amax(2, keepdim=True))
    out = (o * wgt).sum(2) / (e.sum(-1, keepdim=True) * wgt).sum(2)
    return out.to(v.dtype).reshape(B, H, 1, hd)


def _bmm_f32(t):
    """bf16 / fp16 skorunu fp32 cikaran bmm (out_dtype) bu surumde / cihazda var mi (bir kez denenir)."""
    if "bmm_f32" not in _COMPILED:
        try:
            a = t.new_zeros(1, 1, 8)
            torch.bmm(a, a.transpose(1, 2), out_dtype=torch.float32)
            _COMPILED["bmm_f32"] = True
        except (RuntimeError, TypeError):
            _COMPILED["bmm_f32"] = False
    return _COMPILED["bmm_f32"]


def _static_step(m, glob, f32, smax, w, z, live, sum_len, sen_len, tt, K, V, out):
    """StaticCache'in tek adimi, satir basina (yerinde; Python dallanmasi yok -> torch.compile / CUDA graph).  z: Z_k adimi
    (girdi E(END), ozete yazilir, cumleyi okur), degilse token w (cumleye yazilir).  live False satirin sayaclari ilerlemez
    (yazdigi bos yuva maskede; toplu uretimde biten satir).  Konum: token ozet + cumle sayaci, Z ozet sayaci (=
    SummaryCache'in n_z + i / n_z'si), glob gercek konum tt."""
    x = m.E(torch.where(z, torch.full_like(w, END_ID), w))[:, None]
    pos = torch.where(z, sum_len, sum_len + sen_len)[:, None]
    slot = torch.where(z, sum_len, smax + sen_len)
    ar = torch.arange(len(w), device=w.device)
    loc = None
    for l, block in enumerate(m.blocks):
        g = glob[l]
        j = torch.arange(K[l].shape[2], device=w.device)[None]
        if g:
            mask = j <= tt[:, None]
        else:
            if loc is None:                                              # ozet [0, sum_len (+ Z)), cumle [smax, ..]
                loc = (j < (sum_len + z.long())[:, None]) | ((j >= smax) & (j < (smax + sen_len + (~z).long())[:, None]))
            mask = loc
        q, k, v = block._qkv(x, tt[:, None] if g else pos)
        at = tt if g else slot
        K[l][ar, :, at] = k[:, :, 0]
        V[l][ar, :, at] = v[:, :, 0]
        x = block._finish(x, decode_sdpa(q, K[l], V[l], mask, f32))
    out.copy_(m._logits(m.norm(x[:, 0])))
    sen_len.copy_(torch.where(live, torch.where(z, torch.zeros_like(sen_len), sen_len + 1), sen_len))
    sum_len.add_((z & live).long())
    tt.add_(live.long())


class StaticCache:
    """SummaryCache'in sabit tamponlu esi (belge 84): ayni arayuz (prefill, append_token, close_sentence -> sonraki
    token'in logit'i) ve ayni matematik.  Istem SummaryCache.prefill ile islenir, K/V onceden ayrilmis tamponlara alinir
    (yerel [ozet | cumle], glob gercek konumla); adim _static_step.  CUDA'da ilk adimda derlenir (torch.compile, surec
    basina bir kez) ve CUDA graph'a yakalanir: adim basina tek replay.  positions / sentences / sentence_tokens: istemden
    sonra en cok eklenecek konum, Z ve cumle token'i (asilirsa DURUR).  Toplu (belge 91): prefill_rows birden cok istemi
    satir satir yukler (satir basina SummaryCache.prefill + acik cumle), step_rows satir basina token / Z / canli adimi.
    carry (parca siniri): supports False, SummaryCache."""
    BUCKET = 256         # tampon boylari kademeli: BUCKET x {1, 1,5} x 2^k (sekil basina bir derleme; dec84: istem basina
                         # degisen boy her istemde yeniden derletiyordu, 8'den sonra derlemesiz)

    @staticmethod
    def supports(model):
        return not model.carry_group

    def __init__(self, model, positions, sentences, sentence_tokens):
        assert StaticCache.supports(model), "StaticCache: carry desteklenmez (SummaryCache)"
        self.m, self.dev = model, model.E.weight.device
        self.budget = (positions, sentences, sentence_tokens)
        self.run = None

    @torch.no_grad()
    def prefill(self, sents):
        """Tek istem -> sonraki token'in logit'i (prefill_rows'un tek satiri)."""
        return self.prefill_rows([sents])[0]

    @torch.no_grad()
    def prefill_rows(self, prompts, opens=None):
        """prompts: satir basina istem cumleleri, opens: satir basina acik son cumlenin token'lari (yoksa bos) -> (B, V)
        sonraki token'in logit'i.  Butun satirlar tek ileri geciste (belge 92; once satir basina SummaryCache.prefill +
        acik token'lar adim adim): satir [BOS, cumleler (Z_k ile), acik token'lar], sagdan dolgu; yerel bloklar
        model_z_read_mask + mantiksal konum, glob bloklar causal (is_causal) + gercek konum.  Ozet (BOS + Z) ve acik cumle
        K/V'si yerel tampona, butun konumlar glob tampona (boy satirlarin en buyugune gore kademe)."""
        opens = opens or [()] * len(prompts)
        L, b, B, dev = len(self.m.blocks), self.BUCKET, len(prompts), self.dev
        up = lambda n: next(t for k in range(40) for t in (b << k, (3 * b // 2) << k) if t >= n)  # noqa: E731
        rows = []
        for sents, op in zip(prompts, opens):
            tok, kind, pos, sent = [EOS_ID], [BOS], [0], [-1]
            for k_, x in enumerate(sents):
                tok += list(x) + [END_ID]                                   # Z_k girdisi E(END)
                kind += [TOKEN] * len(x) + [ZTOK]
                pos += [k_ + 1 + i for i in range(len(x))] + [k_ + 1]
                sent += [k_] * (len(x) + 1)
            tok += list(op)                                                 # acik cumle: SummaryCache.append_token gibi
            kind += [TOKEN] * len(op)
            pos += [len(sents) + 1 + i for i in range(len(op))]
            sent += [len(sents)] * len(op)
            rows.append((tok, kind, pos, sent, len(sents), len(op)))
        T = [len(r[0]) for r in rows]
        Tp = max(T)
        pad = lambda r, i, v: r[i] + [v] * (Tp - len(r[i]))  # noqa: E731
        lt = lambda x: torch.tensor(x, device=dev)  # noqa: E731
        tok, kind = lt([pad(r, 0, 0) for r in rows]), lt([pad(r, 1, PAD) for r in rows])
        lpos, sent = lt([pad(r, 2, 0) for r in rows]), lt([pad(r, 3, -1) for r in rows])
        doc = torch.where(kind == PAD, -1, 0)
        local = _dense(model_z_read_mask(kind, doc, sent), B, Tp, dev)[:, None]
        real = torch.arange(Tp, device=dev).expand(B, Tp)
        x = self.m.E(torch.where(kind == ZTOK, torch.full_like(tok, END_ID), tok))
        self.glob = [l >= L - self.m.global_layers for l in range(L)]
        ks, vs = [], []
        for l, block in enumerate(self.m.blocks):
            g = self.glob[l]
            q, k, v = block._qkv(x, real if g else lpos)
            if g:                                                       # tek hikaye, sagdan dolgu: causal = glob maskesi
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=block.kv_heads != block.heads)
            else:
                a = gqa_sdpa(q, k, v, local)
            ks.append(k)
            vs.append(v)
            x = block._finish(x, a)
        ar = torch.arange(B, device=dev)
        self.logits = self.m._logits(self.m.norm(x[ar, lt(T) - 1]))
        summ = (kind == BOS) | (kind == ZTOK)
        opn = (kind == TOKEN) & (sent == lt([r[4] for r in rows])[:, None])
        nsum, sen = [r[4] + 1 for r in rows], [r[5] for r in rows]
        self.smax = up(max(nsum) + self.budget[1] + 1)
        self.limits = (self.smax, up(self.budget[2]), up(Tp + self.budget[0]))  # ozet, cumle, glob konum sinirlari
        self.n = [list(x_) for x_ in zip(nsum, sen, T)]                 # satir basina host sayaclari (sinir denetimi)
        self.K, self.V = [], []
        for l in range(L):
            kk, vv = ks[l], vs[l]
            kb = kk.new_zeros(B, kk.shape[1], self.limits[2] if self.glob[l] else self.smax + self.limits[1], kk.shape[3])
            vb = torch.zeros_like(kb)
            for r in range(B):
                if self.glob[l]:
                    kb[r, :, :T[r]], vb[r, :, :T[r]] = kk[r, :, :T[r]], vv[r, :, :T[r]]
                    continue
                i_s, i_o = summ[r].nonzero()[:, 0], opn[r].nonzero()[:, 0]
                kb[r, :, :len(i_s)], vb[r, :, :len(i_s)] = kk[r][:, i_s], vv[r][:, i_s]
                kb[r, :, self.smax:self.smax + len(i_o)] = kk[r][:, i_o]
                vb[r, :, self.smax:self.smax + len(i_o)] = vv[r][:, i_o]
            self.K.append(kb)
            self.V.append(vb)
        self.state = (lt(nsum), lt(sen), lt(T))                          # sum_len, sen_len, tt
        self.w, self.z = lt([0] * B), torch.zeros(B, dtype=torch.bool, device=dev)
        self.live = torch.ones(B, dtype=torch.bool, device=dev)
        self.out = torch.zeros_like(self.logits)
        self.f32 = self.out.dtype != torch.float32 and _bmm_f32(self.out)
        return self.logits

    def _step(self):
        _COMPILED.get("step", _static_step)(self.m, self.glob, self.f32, self.smax, self.w, self.z, self.live, *self.state,
                                            self.K, self.V, self.out)

    def _capture(self):
        """Ilk adimda (CUDA): derleme + CUDA graph.  Isinma adimlarinin sayac ilerlemesi geri alinir; tamponlara
        yazdiklari ya gecerli yuvalarin otesinde (maskeli) ya da replay'de ayni yuvaya yeniden yazilir."""
        if self.dev.type != "cuda":
            return self._step
        if "step" not in _COMPILED:
            _COMPILED["step"] = torch.compile(_static_step, dynamic=False)
            for k in ("recompile_limit", "cache_size_limit"):                 # kademe x dtype x model sekilleri
                if getattr(torch._dynamo.config, k, 64) < 64:
                    setattr(torch._dynamo.config, k, 64)
        saved = [t.clone() for t in self.state]
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
                for t, v in zip(self.state, saved):
                    t.copy_(v)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        return self.graph.replay

    @torch.no_grad()
    def step_rows(self, w, z, live):
        """Satir basina token w, Z bayragi z, canli live (listeler) -> (B, V) sonraki logit.  Canli olmayan satir ilerlemez."""
        for n, z_, l_ in zip(self.n, z, live):
            if l_:
                n[:] = [n[0] + z_, 0 if z_ else n[1] + 1, n[2] + 1]
                assert all(a <= b for a, b in zip(n, self.limits)), \
                    "StaticCache: tampon asildi %s > %s (positions / sentences / sentence_tokens)" % (n, self.limits)
        if len(w) == 1:                                                  # tek satir: kopyasiz doldurma
            self.w.fill_(int(w[0]))
            self.z.fill_(bool(z[0]))
            self.live.fill_(bool(live[0]))
        else:
            self.w.copy_(torch.tensor(w))
            self.z.copy_(torch.tensor(z))
            self.live.copy_(torch.tensor(live))
        if self.run is None:
            self.run = self._capture()
        self.run()
        self.logits = self.out
        return self.out

    @torch.no_grad()
    def append_token(self, token):
        """Simdiki cumleye token -> sonraki token'in logit'i (tek satir)."""
        return self.step_rows([token], [False], [True])[0]

    @torch.no_grad()
    def close_sentence(self):
        """Cumle bitti: Z_k ozete -> sonraki cumlenin ilk token'i (ya da EOS) logit'i (tek satir)."""
        return self.step_rows([END_ID], [True], [True])[0]
