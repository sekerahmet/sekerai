"""recipe -- V2 ortak egitim tarifi (iki model ayni kod; belge 20 §4, 21 §6; adlar onayli, kullanici 6 Ekim).

    maske       mask_fn(kind, doc, sent) -> mask_mod(b, h, q, kv) (FlexAttention imzasi).  Model kendi fonksiyonunu verir
                (Model Z: model_z_mask); transformer icin document_mask.  Ortak dolgu kurali: dolgu, ayni satirdaki
                onceki dolguyu gorur (tamamen maskeli satir SDPA'da NaN).  block_mask (FlexAttention) ve dense_mask (SDPA,
                CPU egitimi ve testler; FlexAttention CPU'da geri yayilim yapmiyor) ayni mask_mod'dan.
    wsd_lr      warmup %1, sabit, son %20 dogrusal sifira; inisin ilk adimi = total - round(decay * total) (checkpoint).
    param_groups  AdamW: 2-B agirliklar decay'li; embedding, norm, bias, 1-B decay'siz.
    muon_params / MuonAdamW  --optimizer muon: bloklarin 2-B matrisleri Muon'a, geri kalan AdamW'ye; iki optimizer tek
                arayuzde (lr takvimi, checkpoint).
    Checkpoint  model + optimizer + adim + plan + gecmis + args + RNG; .part'tan atomik; kesilip surdurulen = kesintisiz.
    SpeedWindow isinma sonrasi pencere: basta ve sonda synchronize; gercek (dolgusuz) token / sn ve duvar saati.
    output_loss egitim kaybi: tam CE, CUDA'da derlenmis (belge 24 §5 A); sinav loss_per_target'la kalir.
    Torba C0 (--bag_stage factor; belge 53 s4 / s9, 54 s1-2, 55 K3 / O7 / O8; adlar onayli 7 Ekim): B = C sabit cekirdek
                (core_ids), iki asamali olasilik.  Dagilimin TEK tanimi two_stage_logprobs (sinav, uretim, teshis onu
                cagirir); egitimin hizli yolu two_stage_loss (C tek matmul, tam sozluk yalniz kacan + tam softmax payi
                satirlarinda, sabit dilim) ona esitligiyle sinanir (tests_v2 recipe).
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import END_ID, EOS_ID, VOCAB, Kind

BAG_CHUNK = 2048            # tam sozluk satir dilimi (sabit sekil: compile tek grafik; dolgu en cok BAG_CHUNK - 1 satir)


def document_mask(kind, doc, sent):
    """Transformer: ayni hikaye ve kv <= q (sent kullanilmaz; imza Model Z ile ayni)."""
    def mask_mod(b, h, q, kv):
        return (doc[b, q] == doc[b, kv]) & (kv <= q)
    return mask_mod


def _with_padding(mask_mod, kind, doc):
    """Ortak dolgu kurali: dolgu (doc -1) ayni satirdaki onceki dolguyu gorur; gercek konum dolguyu gormez."""
    def mod(b, h, q, kv):
        pad = (kind[b, q] == Kind.PAD) & (kind[b, kv] == Kind.PAD) & (kv <= q)
        real = (kind[b, q] != Kind.PAD) & (kind[b, kv] != Kind.PAD)
        return (mask_mod(b, h, q, kv) & real) | pad
    return mod


def dense_mask(batch, mask_fn):
    """-> bool (B, T, T) (SDPA attn_mask: True = gorulur)."""
    B, T = batch.kind.shape
    mod = _with_padding(mask_fn(batch.kind, batch.doc, batch.sent), batch.kind, batch.doc)
    dev = batch.kind.device
    return mod(torch.arange(B, device=dev)[:, None, None], 0, torch.arange(T, device=dev)[None, :, None],
               torch.arange(T, device=dev)[None, None, :])


def block_mask(batch, mask_fn):
    """-> FlexAttention BlockMask (GPU'da derlenerek kurulur)."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, T = batch.kind.shape
    mod = _with_padding(mask_fn(batch.kind, batch.doc, batch.sent), batch.kind, batch.doc)
    dev = batch.kind.device
    return create_block_mask(mod, B, None, T, T, device=dev, _compile=dev.type == "cuda")


def _output_loss(h, weight, target):
    return F.cross_entropy((h @ weight.T).float(), target, ignore_index=-100)


_COMPILED = {}


def output_loss(h, weight, target):
    """h (N, d), weight (V, d), target (N,) (-100 hedefsiz) -> hedefli konumlarda ortalama CE (logit fp32).
    CUDA'da derlenmis (bir kez sarilir), CPU'da eager (Windows'ta inductor icin MSVC yok)."""
    if not h.is_cuda:
        return _output_loss(h, weight, target)
    if "output_loss" not in _COMPILED:
        _COMPILED["output_loss"] = torch.compile(_output_loss, dynamic=False)
    return _COMPILED["output_loss"](h, weight, target)


def core_ids(counts, n):
    """Train sayimi (data.token_counts; END yok) -> sabit cekirdek C (sirali int64): en sik n token + END + EOS (belge 55
    O1)."""
    order = np.argsort(-np.asarray(counts), kind="stable")
    return np.unique(np.r_[order[:n], END_ID, EOS_ID]).astype(np.int64)


def attach_bag(model, k):
    """Modele torba: bag_core_ids (k,) tampon (agent.pt kendine yeter; degeri cagiran ya da state_dict yazar) ve bag_other
    (d,) DIGER vektoru (sifir; s_O = h . bag_other).  Ilk agirliklardan sonra: RNG tuketmez (belge 55 K2)."""
    w = model.E.weight
    model.register_buffer("bag_core_ids", torch.zeros(k, dtype=torch.long, device=w.device))
    model.bag_other = torch.nn.Parameter(torch.zeros(w.shape[1], device=w.device))
    return model


def bag_mask(ids, n=VOCAB):
    return torch.zeros(n, dtype=torch.bool, device=ids.device).index_fill_(0, ids, True)


def two_stage_logprobs(logits, in_bag, other):
    """Iki asamali dagilimin TEK tanimi (belge 54 s1.1): logits (..., V), in_bag (V,) bool, other (...) DIGER logit'i ->
    log p (..., V), toplam 1.  v in B: l_v - LSE(l_B, s_O); v disinda: s_O - LSE(l_B, s_O) + l_v - LSE_disari(l).
    Disari bossa DIGER kapali (= log_softmax; belge 55 O7).  Tumleyen LSE maskeyle dogrudan (cikarma yok, belge 55 O8)."""
    logits, other = logits.float(), other.float()
    if bool(in_bag.all()):
        return torch.log_softmax(logits, -1)
    lse_in = torch.logaddexp(logits.masked_fill(~in_bag, float("-inf")).logsumexp(-1), other)
    lse_out = logits.masked_fill(in_bag, float("-inf")).logsumexp(-1)
    return torch.where(in_bag, logits - lse_in[..., None], (other - lse_in - lse_out)[..., None] + logits)


def bag_rows(target, in_core, frac, rng):
    """CPU hazirligi (veriden; GPU senkronu yok) -> (satirlar, sayilar).  Tam sozluk satirlari = kacan hedefler (hedef C
    disi: sabit C'de yalniz veriye bagli) u tam softmax payi (gecerli hedeflerden round(frac * n) tanesi; rng adim
    tohumlu, surdurmede ayni).  BAG_CHUNK'a dolgu: idx / miss / full / valid (n_dilim * BAG_CHUNK,)."""
    chunk = BAG_CHUNK
    t = np.asarray(target).reshape(-1)
    valid = t >= 0
    miss = valid & ~in_core[np.where(valid, t, 0)]
    vi = np.flatnonzero(valid)
    full = np.zeros(len(t), bool)
    full[rng.choice(vi, int(round(frac * len(vi))), replace=False)] = True
    idx = np.flatnonzero(miss | full)
    pad = -len(idx) % chunk
    rows = dict(idx=np.r_[idx, np.zeros(pad, np.int64)], miss=np.r_[miss[idx], np.zeros(pad, bool)],
                full=np.r_[full[idx], np.zeros(pad, bool)], valid=np.r_[np.ones(len(idx), bool), np.zeros(pad, bool)])
    return {k: torch.as_tensor(v) for k, v in rows.items()}, dict(miss=int(miss.sum()), valid=len(vi), full=int(full.sum()))


def _bag_in(h, w_core, other, tpos):
    """Torba ici (N, K) tek matmul -> (konum basina -log p: hedef B'de l_t ile, kacan s_O ile; p(DIGER))."""
    lg, so = (h @ w_core.T).float(), (h @ other).float()
    lse = torch.logaddexp(lg.logsumexp(-1), so)
    tl = torch.where(tpos >= 0, lg.gather(1, tpos.clamp_min(0)[:, None])[:, 0], so)
    return lse - tl, (so - lse).exp()


def _bag_full(h, weight, target, in_core, miss, full):
    """Tam sozluk satirlari (dilim, V) -> (kacanlarin -log softmax_disari[t] toplami, tam softmax CE toplami)."""
    lg = (h @ weight.T).float()
    t = lg.gather(1, target[:, None])[:, 0]
    out = lg.masked_fill(in_core, float("-inf")).logsumexp(-1) - t
    return torch.where(miss, out, 0.0).sum(), torch.where(full, lg.logsumexp(-1) - t, 0.0).sum()


def _compiled(name, fn, h):
    """CUDA'da derlenmis (bir kez sarilir, dynamic=False), CPU'da eager (output_loss ile ayni kural)."""
    if not h.is_cuda:
        return fn
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(fn, dynamic=False)
    return _COMPILED[name]


def two_stage_loss(h, weight, other, target, ids, rows, counts):
    """Egitimin hizli yolu.  h (N, d), weight E (V, d), other bag_other, target (N,) (-100 hedefsiz), ids bag_core_ids,
    rows / counts bag_rows'tan -> (amac, nll, ek).  nll = iki asamali dagilimin ortalama NLL'i (two_stage_logprobs ile
    ayni; sinav tanimi); amac = nll + secilen satirlarin tam softmax CE toplami / gecerli hedef (belge 54 s2.3 yol a,
    konum basina ayni agirlik); ek: p(DIGER) toplami, tam CE toplami (cihazda; senkron yok)."""
    V = weight.shape[0]
    in_core = bag_mask(ids, V)
    where = torch.full((V,), -1, dtype=torch.long, device=ids.device)
    where[ids] = torch.arange(len(ids), device=ids.device)
    valid = target >= 0
    term, p_other = _compiled("bag_in", _bag_in, h)(h, weight[ids], other, where[target.clamp_min(0)])
    nll_sum = torch.where(valid, term, 0.0).sum()
    full_sum = torch.zeros((), device=h.device)
    for r in range(0, len(rows["idx"]), BAG_CHUNK):
        i, ok = rows["idx"][r:r + BAG_CHUNK], rows["valid"][r:r + BAG_CHUNK]
        o, f = _compiled("bag_full", _bag_full, h)(h[i], weight, target[i].clamp_min(0), in_core,
                                                    rows["miss"][r:r + BAG_CHUNK] & ok, rows["full"][r:r + BAG_CHUNK] & ok)
        nll_sum, full_sum = nll_sum + o, full_sum + f
    n = max(counts["valid"], 1)
    nll = nll_sum / n
    return nll + full_sum / n, nll, dict(p_other=torch.where(valid, p_other, 0.0).sum().detach(), full=full_sum.detach())


def wsd_lr(step, total, peak, warmup=0.01, decay=0.2):
    """Adim (0'dan) -> lr: dogrusal isinma (max 1 adim), sabit tepe, son decay payi dogrusal sifira (son adim > 0)."""
    warm = max(1, round(warmup * total))
    down = total - round(decay * total)
    if step < warm:
        return peak * (step + 1) / warm
    if step < down:
        return peak
    return peak * (total - step) / max(1, total - down)


def param_groups(model, weight_decay=0.1, skip=()):
    """-> AdamW gruplari: 2-B agirliklar (embedding haric) decay'li; embedding, norm, bias ve 1-B decay'siz.  skip:
    baska optimizer'in parametreleri (Muon)."""
    emb = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    decay, no_decay, seen = [], [], {id(p) for p in skip}
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 and id(p) not in emb else no_decay).append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


def muon_params(model):
    """-> [(ad, p)] Muon'a gidenler: bloklarin (model.blocks) 2-B agirliklari (attention qkv / proj, MLP gate_up / down).
    E (tied: giris + cikis), norm kazanclari, bloklarin disindaki giris katmanlari ve 1-B her sey AdamW'de (belge 20 s4)."""
    emb = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    return [(n, p) for n, p in model.named_parameters()
            if n.startswith("blocks.") and p.dim() == 2 and p.requires_grad and id(p) not in emb]


class MuonAdamW:
    """Muon + AdamW tek optimizer gibi: param_groups ikisinin gruplari (lr takvimi hepsine), step / zero_grad ikisine,
    state_dict ikisini birden (checkpoint)."""

    def __init__(self, muon, adamw):
        self.muon, self.adamw = muon, adamw

    @property
    def param_groups(self):
        return self.adamw.param_groups + self.muon.param_groups

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none)
        self.muon.zero_grad(set_to_none)

    def step(self):
        self.adamw.step()
        self.muon.step()

    def state_dict(self):
        return dict(adamw=self.adamw.state_dict(), muon=self.muon.state_dict())

    def load_state_dict(self, state):
        self.adamw.load_state_dict(state["adamw"])
        self.muon.load_state_dict(state["muon"])


class Checkpoint:
    """Surdurme paketi: <dir>/checkpoint.pt (yarim dosya kalmaz)."""

    @staticmethod
    def save(dir_, model, opt, step, plan_meta, history, args):
        os.makedirs(dir_, exist_ok=True)
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        path = os.path.join(dir_, "checkpoint.pt")
        torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), step=step, plan=plan_meta, history=history,
                        args=args, rng=rng), path + ".part")
        os.replace(path + ".part", path)

    @staticmethod
    def load(dir_, model, opt, device="cpu"):
        """-> dict(step, plan, history, args); model ve optimizer yuklenir, RNG geri konur."""
        pack = torch.load(os.path.join(dir_, "checkpoint.pt"), map_location=device, weights_only=False)
        model.load_state_dict(pack["state"])
        opt.load_state_dict(pack["opt"])
        torch.set_rng_state(pack["rng"]["cpu"].cpu())                    # RNG durumu CPU ByteTensor olmali
        if pack["rng"]["cuda"] is not None and torch.cuda.is_available():  # (map_location="cuda" onu tasir)
            torch.cuda.set_rng_state_all([s.cpu() for s in pack["rng"]["cuda"]])
        return dict(step=pack["step"], plan=pack["plan"], history=pack["history"], args=pack["args"])


class SpeedWindow:
    """Hiz penceresi: start(step) isinmadan sonra bir kez, stop(step, real_tokens) pencerenin sonunda (real_tokens:
    pencerede islenen gercek, dolgusuz token).  Pencere disinda (checkpoint, sinav, uretim) durdurulmali."""

    def __init__(self):
        self.t0 = self.step0 = None

    def start(self, step):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0, self.step0 = time.perf_counter(), step

    def stop(self, step, real_tokens):
        """-> dict(steps, tokens, seconds, tokens_per_sec)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec = time.perf_counter() - self.t0
        return dict(steps=step - self.step0, tokens=int(real_tokens), seconds=round(sec, 3),
                    tokens_per_sec=round(real_tokens / sec, 1) if sec > 0 else math.nan)
