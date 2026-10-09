"""beta (V2) -- DeepSeek-V4 dikkatinin kucuk yogun tabani ve Model Beta (belge 99).

Kullanici, 9 Ekim: "beta bence deepseek flash üzerine kurabiliriz model Z yerine"; "tamam onları da dışarıda bırakalım"
(MoE, mHC, Engram, CED, FP8 / FP4 yok).  Kaynak: DeepSeek-V4.1-Flash inference/model.py (HF 2cba9e4) ve DeepSeek-V4-Pro
inference/model.py (HF b5968e9), MIT; satir satir eslesme belge 99'da.

Katman (V4 Attention): gizli sorgu (wq_a -> RMSNorm -> wq_b, head basina agirliksiz RMS), tek KV head'i K = V (MQA;
wkv -> RMSNorm), son ROPE_DIM boyutta RoPE (bitisik ciftler), cikista ters RoPE, attention sink, gruplu cikis izdusumu
(wo_a grup basina, wo_b).  ratio m > 0 katmani ek olarak sikistirilmis girdileri gorur: Compressor (V4.1, ortusmesiz, ape
yok) m token'i boyut basina softmax kapisiyla tek girdiye toplar, RMSNorm, RoPE grubun ilk token konumunda.  Seyrek secim
(indexer) yok: T 2048'de girdi sayisi <= T / 4 = 512 = V4'un top-k'si (belge 99 HESAP).

Iki duzen (sentence):
    False (taban, V4): ham pencere son WINDOW token; grup = hikayede m'lik ardisik token, tamamlaninca gorunur.
    True  (Model Beta): ham pencere = BOS + kendi cumlesi + onceki raw_sentences (K) cumle; grup = cumle icinde m'lik
          parca (cumle sonu son parcayi kapatir), ratio SENTENCE = cumle basina tek girdi; onceki BUTUN cumlelerin girdileri
          gorunur (K > 0'da ham kisimla ortusur, V4'te de pencere ile girdiler ortusur).
Egitim: anahtar [ham T | girdi yuvalari T] (yuva = grubun sirasi), maske katman turu basina (mask_fn ikilisi gibi demet).
Uretim: her adimda tam ileri (onbelleksiz, v1).
"""
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID, EOS_ID, ROW_LEN, VOCAB, Kind  # noqa: E402

BOS, PAD, END = Kind.BOS, Kind.PAD, Kind.END
WINDOW = 128                # V4 / V4.1 window_size
CSA, HCA = 4, 128           # V4 compress_ratios (CSA m, HCA m')
SENTENCE = -1               # Beta: cumle basina tek girdi (HCA'nin karsiligi)
HEAD_DIM, ROPE_DIM = 128, 16     # c, rd (V4: 512 / 64 = 8)
NORM_EPS = 1e-6             # V4-Pro RMSNorm
ROPE_THETA, COMPRESS_ROPE_THETA = 10000.0, 160000.0     # V4.1 config: rope_theta / compress_rope_theta
O_GROUPS = 4
Q_HEAD_NORM = True          # q'ya head basina agirliksiz RMS (V4-Pro :498; V4.1 kodunda yok)


def layer_ratios(layers, sentence=False):
    """V4 dizilimi kucultulmus: ilk katman yalniz pencere, sonra CSA / HCA sirayla (belge 99)."""
    hca = SENTENCE if sentence else HCA
    return (0,) + tuple(CSA if i % 2 == 0 else hca for i in range(layers - 1))


def rope(x, pos, theta, inverse=False):
    """Son ROPE_DIM boyuta RoPE, bitisik ciftler (V4 apply_rotary_emb).  x (B, H, T, c) ya da (B, T, c), pos (B, T)."""
    rd = ROPE_DIM
    freq = 1.0 / theta ** (torch.arange(0, rd, 2, device=x.device, dtype=torch.float) / rd)
    ang = pos.float()[..., None] * freq                                   # (B, T, rd / 2)
    if x.dim() == 4:
        ang = ang[:, None]
    cos, sin = ang.cos(), ang.sin() * (-1 if inverse else 1)
    r = x[..., -rd:].float().unflatten(-1, (-1, 2))
    a, b = r[..., 0], r[..., 1]
    r = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).to(x.dtype)
    return torch.cat([x[..., :-rd], r], -1)


def groups(kind, doc, sent, m, sentence):
    """Satir (B, T) -> (gid, slot_end, slot_pos): token'in girdi yuvasi (girdisi yoksa T), yuvanin son token sutunu (bos -1),
    yuvanin RoPE konumu (grubun ilk token'inin hikaye ici konumu).  Taban: hikaye ici konum p ile m'lik gruplar, yalniz
    tamamlanan; Beta: BOS ve dolgu disi token'lar cumle icinde m'lik (SENTENCE: butun cumle)."""
    B, T = kind.shape
    j = torch.arange(T, device=kind.device).expand(B, T)
    real = kind != PAD
    first = torch.ones_like(real)
    first[:, 1:] = doc[:, 1:] != doc[:, :-1]
    p = j - torch.where(first, j, torch.zeros_like(j)).cummax(1).values            # hikaye ici konum (BOS 0)
    if not sentence:
        end = j + (m - 1 - p % m)
        ok = real & (end < T)
        ok &= doc.gather(1, end.clamp(max=T - 1)) == doc
    else:
        ok = real & (kind != BOS)
        e = torch.where(kind == END, j, torch.full_like(j, T)).flip(1).cummin(1).values.flip(1)   # cumlenin END'i
        if m == SENTENCE:
            end = e
        else:
            edge = torch.where((kind == END) | (kind == BOS), j, torch.full_like(j, -1))
            start = torch.cat([torch.full_like(j[:, :1], -1), edge[:, :-1]], 1).cummax(1).values + 1
            end = torch.minimum(j + (m - 1 - (j - start) % m), e)
        ok &= end < T
    is_end = ok & (end == j)
    rank = is_end.long().cumsum(1) - 1
    gid = torch.where(ok, rank.gather(1, end.clamp(max=T - 1)), torch.full_like(j, T))
    slot_end = torch.full((B, T + 1), -1, dtype=torch.long, device=kind.device)
    slot_end.scatter_(1, torch.where(is_end, rank, torch.full_like(j, T)), j)
    slot_pos = torch.full((B, T + 1), T, dtype=torch.long, device=kind.device)
    slot_pos.scatter_reduce_(1, gid, p, "amin")
    return gid, slot_end[:, :T], slot_pos[:, :T].clamp(max=T - 1)


def make_mask(m, sentence, raw_sentences=0):
    """Katman turu -> mask_fn(kind, doc, sent) -> mask_mod(b, h, q, kv); kv < T ham anahtar, kv >= T girdi yuvasi.
    raw_sentences K (Beta): ham = BOS + 0 <= sent[q] - sent[j] <= K."""
    def mask_fn(kind, doc, sent):
        T = kind.shape[1]
        slot_end = groups(kind, doc, sent, m, sentence)[1] if m else None

        def mask_mod(b, h, q, kv):
            j = kv.clamp(max=T - 1)
            if sentence:
                d = sent[b, q] - sent[b, j]
                raw = (j <= q) & (((d >= 0) & (d <= raw_sentences)) | (kind[b, j] == BOS))
            else:
                raw = (j <= q) & (q - j < WINDOW)
            raw = (kv < T) & raw & (doc[b, j] == doc[b, q])
            if not m:
                return raw
            se = slot_end[b, (kv - T).clamp(0, T - 1)]
            s = se.clamp(min=0)
            comp = (kv >= T) & (se >= 0) & (s <= q) & (doc[b, s] == doc[b, q])
            if sentence:
                comp = comp & (sent[b, s] < sent[b, q])                  # yerinde degil: flex alt grafigi copy_ derleyemez
            return raw | comp
        return mask_mod
    mask_fn.includes_padding = True                                         # recipe sarmaz
    mask_fn.kv_len = (lambda T: 2 * T) if m else (lambda T: T)             # recipe: anahtar boyu
    return mask_fn


def dense(mask_fn, kind, doc, sent):
    """mask_fn -> bool (B, T, S) (CPU yolu ve testler; recipe.dense_mask ile ayni)."""
    B, T = kind.shape
    dev = kind.device
    return mask_fn(kind, doc, sent)(torch.arange(B, device=dev)[:, None, None], 0, torch.arange(T, device=dev)[None, :, None],
                                    torch.arange(mask_fn.kv_len(T), device=dev)[None, None, :])


def rms(x, eps=NORM_EPS):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class Compressor(torch.nn.Module):
    """V4.1 Compressor (ortusmesiz, ape yok): grup icinde boyut basina softmax(wgate h) agirlikli wkv h toplami, RMSNorm,
    RoPE grubun ilk konumunda.  fp32 (V4.1 :444-446)."""

    def __init__(self, d, c):
        super().__init__()
        self.wkv, self.wgate = torch.nn.Linear(d, c, bias=False), torch.nn.Linear(d, c, bias=False)
        self.norm = torch.nn.RMSNorm(c, eps=NORM_EPS)

    def forward(self, h, gid, slot_pos, theta):
        B, T, _ = h.shape
        with torch.autocast(h.device.type, enabled=False):
            kv, sc = self.wkv(h.float()), self.wgate(h.float())
            idx = gid[..., None].expand_as(sc)
            mx = sc.new_full((B, T + 1, sc.shape[-1]), -torch.inf).scatter_reduce(1, idx, sc.detach(), "amax")
            e = (sc - mx.gather(1, idx)).exp()
            den = sc.new_zeros(B, T + 1, sc.shape[-1]).scatter_add(1, idx, e)
            num = sc.new_zeros(B, T + 1, sc.shape[-1]).scatter_add(1, idx, e * kv)
            out = self.norm((num / den.clamp_min(1e-30))[:, :T])
        return rope(out.to(h.dtype), slot_pos, theta)


class V4Block(torch.nn.Module):
    def __init__(self, d, heads, hidden, ratio):
        super().__init__()
        c, self.heads, self.ratio = HEAD_DIM, heads, ratio
        self.groups = math.gcd(heads, O_GROUPS)                             # kucuk testlerde heads < O_GROUPS
        self.theta = COMPRESS_ROPE_THETA if ratio else ROPE_THETA           # V4.1 :680-687
        q_rank, o_rank = d // 4, -(-int(1.6 * d / self.groups) // 64) * 64     # V4.1: q_lora 1280 / 5120, 8 x 1024 / 5120
        self.n1, self.n2 = torch.nn.RMSNorm(d), torch.nn.RMSNorm(d)
        self.wq_a, self.wq_b = torch.nn.Linear(d, q_rank, bias=False), torch.nn.Linear(q_rank, heads * c, bias=False)
        self.q_norm = torch.nn.RMSNorm(q_rank, eps=NORM_EPS)
        self.wkv = torch.nn.Linear(d, c, bias=False)
        self.kv_norm = torch.nn.RMSNorm(c, eps=NORM_EPS)
        self.compressor = Compressor(d, c) if ratio else None
        self.attn_sink = torch.nn.Parameter(torch.zeros(heads))
        self.wo_a = torch.nn.ModuleList(torch.nn.Linear(heads // self.groups * c, o_rank, bias=False)
                                        for _ in range(self.groups))                # grup basina ayri matris (Muon)
        self.wo_b = torch.nn.Linear(self.groups * o_rank, d, bias=False)
        self.gate_up = torch.nn.Linear(d, 2 * hidden, bias=False)
        self.down = torch.nn.Linear(hidden, d, bias=False)

    def attend(self, x, pos, meta, attn):
        """-> cikis izdusumunden once o (B, H, T, c) (ters RoPE'li); meta (gid, slot_pos) ya da None."""
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq_b(self.q_norm(self.wq_a(h))).view(B, T, self.heads, -1).transpose(1, 2)
        q = rope(rms(q) if Q_HEAD_NORM else q, pos, self.theta)              # V4-Pro :498-499
        kv = rope(self.kv_norm(self.wkv(h)), pos, self.theta)
        if self.ratio:
            kv = torch.cat([kv, self.compressor(h, meta[0], meta[1], self.theta)], 1)
        kv = kv[:, None].to(q.dtype)                                         # autocast: norm fp32, q bf16
        sink = self.attn_sink.float()[None, :, None]
        if torch.is_tensor(attn):
            s = (q.float() @ kv.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
            s = s.masked_fill(~attn[:, None], -torch.inf)
            lse = torch.logaddexp(s.logsumexp(-1), sink)                    # sink paydada (V4 denklem 27)
            o = ((s - lse[..., None]).exp() @ kv.float()).to(q.dtype)
        else:
            from torch.nn.attention.flex_attention import flex_attention
            bs = attn.BLOCK_SIZE[0]
            o, lse = flex_attention(q, kv, kv, block_mask=attn, enable_gqa=True, return_lse=True,
                                    kernel_options=None if bs == 128 else dict.fromkeys(
                                        ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"), bs))
            o = o * torch.sigmoid(lse.float() - sink)[..., None].to(o.dtype)
        return rope(o, pos, self.theta, inverse=True)                        # V4.1 :781

    def forward(self, x, pos, meta, attn):
        B, T, d = x.shape
        o = self.attend(x, pos, meta, attn).transpose(1, 2).reshape(B, T, self.groups, -1)
        wa = torch.stack([w.weight for w in self.wo_a])                      # (g, o_rank, H / g * c)
        x = x + self.wo_b(torch.einsum("btgd,grd->btgr", o, wa).flatten(2))
        g, u = self.gate_up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class V4Small(torch.nn.Module):
    def __init__(self, d=512, layers=8, heads=8, sentence=False, ratios=None, vocab_rows=VOCAB, raw_sentences=0):
        """sentence: False taban (V4 pencere), True Model Beta (cumle siniri).  ratios: katman basina m (0 yalniz pencere);
        yoksa layer_ratios.  vocab_rows: E satir sayisi (baseline ile ayni sozluk dolgusu).  raw_sentences: Beta'da ham
        kisma giren onceki cumle sayisi (taban 0)."""
        super().__init__()
        self.END, self.EOS, self.row_len, self.sentence = END_ID, EOS_ID, ROW_LEN, sentence
        self.ratios = tuple(ratios) if ratios is not None else layer_ratios(layers, sentence)
        assert len(self.ratios) == layers
        self.kinds = sorted(set(self.ratios), key=self.ratios.index)                 # katman turleri (maske basina)
        assert raw_sentences >= 0 and (sentence or not raw_sentences), "raw_sentences yalniz Beta, >= 0"
        self.raw_sentences = raw_sentences
        self.mask_fn = tuple(make_mask(m, sentence, raw_sentences) for m in self.kinds)
        hidden = -(-int(8 * d / 3) // 8) * 8
        assert vocab_rows >= VOCAB, "vocab_rows >= VOCAB"
        self.E = torch.nn.Embedding(VOCAB, d)
        if vocab_rows > VOCAB:
            self.E = torch.nn.Embedding(vocab_rows, d, _weight=torch.zeros(vocab_rows, d))
        self.blocks = torch.nn.ModuleList(V4Block(d, heads, hidden, m) for m in self.ratios)
        self.norm = torch.nn.RMSNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("wo_b.weight", "down.weight")) else 0.02
                with torch.no_grad():
                    (p[:VOCAB] if name == "E.weight" else p).normal_(0.0, std)
        with torch.no_grad():
            self.E.weight[VOCAB:].zero_()

    def config(self):
        """Modul sabitleri (kimlikte saklanir; yuklemede ayni olmali)."""
        return dict(ratios=list(self.ratios), sentence=self.sentence, raw_sentences=self.raw_sentences, window=WINDOW, head_dim=HEAD_DIM, rope_dim=ROPE_DIM,
                    o_groups=O_GROUPS, q_head_norm=Q_HEAD_NORM, norm_eps=NORM_EPS,
                    rope_theta=[ROPE_THETA, COMPRESS_ROPE_THETA])

    def max_positions(self):
        return self.row_len

    def _batch_hidden(self, batch, attn=None):
        """PackedBatch (transformer duzeni) -> h; attn: katman turu basina maske demeti (train._attn), yoksa dense."""
        if attn is None:
            attn = tuple(dense(f, batch.kind, batch.doc, batch.sent) for f in self.mask_fn)
        meta = [groups(batch.kind, batch.doc, batch.sent, m, self.sentence)[0::2] if m else None for m in self.kinds]
        x = self.E(batch.tokens)
        for block, m in zip(self.blocks, self.ratios):
            k = self.kinds.index(m)
            x = block(x, batch.pos, meta[k], attn[k])
        return self.norm(x)

    def _logits(self, h):
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

    def row(self, seq):
        """Uretim dizisi (ilk token BOS = EOS_ID) -> tek satirlik batch (build_batch transformer duzeniyle ayni alanlar)."""
        dev = self.E.weight.device
        t = torch.tensor([seq], device=dev)
        is_end = (t == self.END) & (torch.arange(len(seq), device=dev) > 0)
        kind = torch.where(is_end, END, Kind.TOKEN)
        kind[:, 0] = BOS
        sent = torch.cat([torch.zeros_like(t[:, :1]), is_end[:, :-1].long().cumsum(1)], 1)
        sent[:, 0] = -1
        return SimpleNamespace(tokens=t, kind=kind, pos=torch.arange(len(seq), device=dev)[None], doc=torch.zeros_like(t),
                               sent=sent)

    @torch.no_grad()
    def next_logits(self, seq):
        return self._logits(self._batch_hidden(self.row(seq))[0, -1])

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None, open_last=False, on_token=None,
                 stop_when=None, batch_size=1):
        """baseline.generate kurali (birebir; tek istem, her adim tam ileri): cumle basinda EOS hikayeyi bitirir, cumle icinde
        END ya da EOS cumleyi bitirir, max_tokens'ta kesilen cumle kapanir; konum siniri row_len.  batch_size yok sayilir."""
        out, limit = [], self.max_positions()
        for sents in prompts:
            opened = bool(open_last and sents)
            seq = [self.EOS] + [t for s in sents for t in list(s) + [self.END]]
            seq = seq[:-1] if opened else seq
            gen, ended, eos, full = [], [], False, False
            while len(gen) < max_sentences:
                cur, done = [], False
                while len(cur) < max_tokens:
                    if len(seq) + 2 > limit:
                        full = True
                        break
                    p = self.next_logits(seq).float()
                    w = int(p.argmax()) if generator is None else int(torch.multinomial(
                        torch.softmax(p, -1).cpu(), 1, generator=generator))
                    if w == self.EOS and not cur and not opened:
                        eos = True
                        break
                    if w in (self.END, self.EOS):
                        done = True
                        break
                    cur.append(w)
                    seq.append(w)
                    if on_token is not None:
                        on_token(w)
                if eos or (full and not cur):
                    break
                gen.append(cur)
                ended.append(done)
                if on_token is not None:
                    on_token(None)
                if full or (stop_when is not None and stop_when(gen)):
                    break
                opened = False
                seq.append(self.END)
            out.append((gen, ended, eos))
        return out
