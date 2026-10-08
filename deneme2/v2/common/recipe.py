"""recipe -- V2 ortak egitim tarifi (iki model ayni kod; belge 20 §4, 21 §6; adlar onayli, kullanici 6 Ekim).

    maske       mask_fn(kind, doc, sent) -> mask_mod(b, h, q, kv) (FlexAttention imzasi).  Model kendi fonksiyonunu verir
                (Model Z: model._masks(True), summaries_last aralik maskeleri); transformer icin document_mask.  Ortak
                dolgu kurali: dolgu, ayni satirdaki onceki dolguyu gorur (tamamen maskeli satir SDPA'da NaN).  block_mask (FlexAttention) ve dense_mask (SDPA,
                CPU egitimi ve testler; FlexAttention CPU'da geri yayilim yapmiyor) ayni mask_mod'dan.
    wsd_lr      warmup %1, sabit, son %20 dogrusal sifira; inisin ilk adimi = total - round(decay * total) (checkpoint).
    param_groups  AdamW: 2-B agirliklar decay'li; embedding, norm, bias, 1-B decay'siz.
    muon_params / MuonAdamW  --optimizer muon: bloklarin 2-B matrisleri Muon'a, geri kalan AdamW'ye; iki optimizer tek
                arayuzde (lr takvimi, checkpoint).
    Checkpoint  model + optimizer + adim + plan + gecmis + args + RNG; .part'tan atomik; kesilip surdurulen = kesintisiz.
    SpeedWindow isinma sonrasi pencere: basta ve sonda synchronize; gercek (dolgusuz) token / sn ve duvar saati.
    output_loss egitim kaybi: tam CE, CUDA'da derlenmis (belge 24 §5 A); sinav loss_per_target'la kalir.
    output_loss_mtp / mtp_weights  --mtp (deneme/mtp, belge 90c): ayni-logit MTP kaybi ve agirlik takvimi; sinav dokunulmaz.
    Torba (--bag_k) kaldirildi (kullanici, 8 Ekim: "Torbada gereksiz gibi"; belge 77): git etiketi v2-before-cleanup-20261008.
"""
import math
import os
import time

import torch
import torch.nn.functional as F

from data import VOCAB, Kind

NGRAM_LR_MULT = 5.0               # bigram tablosu lr carpani (Engram x5, belge 88b s2.1; deneme)
ATTN_BLOCK = 64                   # FlexAttention blok boyu (SS d512 attention -%14, belge 37; 5w -2,7 ms/adim; bf16 esdeger)


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


def _padded(batch, mask_fn):
    """mask_fn'in mask_mod'u, dolgu kuraliyla: mask_fn.includes_padding ise kendisi (kural icinde; sarma kv tarafina
    kind yuklemesi ekler), yoksa _with_padding.  carry bellegi (batch.mem_rows): mask_fn'e satir basina bellek boyu n_mem."""
    if getattr(batch, "mem_rows", None) is not None:
        return mask_fn(batch.kind, batch.doc, batch.sent, n_mem=(batch.mem_rows >= 0).sum(1))
    mod = mask_fn(batch.kind, batch.doc, batch.sent)
    return mod if getattr(mask_fn, "includes_padding", False) else _with_padding(mod, batch.kind, batch.doc)


def dense_mask(batch, mask_fn):
    """-> bool (B, T, T + M) (SDPA attn_mask: True = gorulur; M carry bellek yuvasi, yoksa 0)."""
    B, T = batch.kind.shape
    mod = _padded(batch, mask_fn)
    dev = batch.kind.device
    S = T + (batch.mem_rows.shape[1] if getattr(batch, "mem_rows", None) is not None else 0)
    return mod(torch.arange(B, device=dev)[:, None, None], 0, torch.arange(T, device=dev)[None, :, None],
               torch.arange(S, device=dev)[None, None, :])


def block_mask(batch, mask_fn):
    """-> FlexAttention BlockMask (GPU'da derlenerek kurulur)."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, T = batch.kind.shape
    mod = _padded(batch, mask_fn)
    dev = batch.kind.device
    S = T + (batch.mem_rows.shape[1] if getattr(batch, "mem_rows", None) is not None else 0)   # carry: anahtar T + M
    return create_block_mask(mod, B, None, T, S, device=dev, BLOCK_SIZE=ATTN_BLOCK, _compile=dev.type == "cuda")


def _output_loss(h, weight, target):
    lg = (h @ weight.T).float()
    if lg.shape[1] > VOCAB:                                              # sozluk dolgusu: dolgu sutunu -inf (belge 89)
        lg = lg.masked_fill(torch.arange(lg.shape[1], device=lg.device) >= VOCAB, float("-inf"))
    return F.cross_entropy(lg, target, ignore_index=-100)


_COMPILED = {}


def output_loss(h, weight, target):
    """h (N, d), weight (V, d), target (N,) (-100 hedefsiz) -> hedefli konumlarda ortalama CE (logit fp32).
    CUDA'da derlenmis (bir kez sarilir), CPU'da eager (Windows'ta inductor icin MSVC yok)."""
    if not h.is_cuda:
        return _output_loss(h, weight, target)
    if "output_loss" not in _COMPILED:
        _COMPILED["output_loss"] = torch.compile(_output_loss, dynamic=False)
    return _COMPILED["output_loss"](h, weight, target)


def _output_loss_mtp(h, weight, target, extra, w):
    """Ana CE output_loss'unkiyle ayni (bit); her ek hedef ayri F.cross_entropy (sum) ayni logit'ten: her cagri tek
    tuketicili, derleyici tabandaki CE gibi birlestirir (belge 90c s11, ce_prof: taban +0,71 ms; log_softmax tablosu
    +14,3 ms / +13 GB, where karsilastirmasi +17,3 ms / +19,6 GB)."""
    lg = (h @ weight.T).float()
    if lg.shape[1] > VOCAB:
        lg = lg.masked_fill(torch.arange(lg.shape[1], device=lg.device) >= VOCAB, float("-inf"))
    main = F.cross_entropy(lg, target, ignore_index=-100)
    ce = torch.stack([F.cross_entropy(lg, extra[:, k], ignore_index=-100, reduction="sum") for k in range(extra.shape[1])])
    return main + (w * ce).sum() / (target >= 0).sum(), main, ce / (extra >= 0).sum(0).clamp_min(1)


def output_loss_mtp(h, weight, target, extra, w):
    """Ayni-logit MTP (modded-nanogpt kayit 53, ek parametresiz; belge 88c, 90c): h (N, d), target (N,), extra (N, K)
    k+1 sonraki hedefler (-100 hedefsiz), w (K,) agirlik tensoru -> (kayip, ana CE, ek CE (K,) ortalamalari).  kayip =
    [sum CE_0 + sum_k w_k sum CE_k] / ana hedef sayisi (resmi kodun sum'i, ana terim output_loss ile ayni olcek); gecersiz
    ek hedef 0 katki (payda degismez, resmi kod gibi).  CUDA'da derlenmis, CPU'da eager."""
    if not h.is_cuda:
        return _output_loss_mtp(h, weight, target, extra, w)
    if "output_loss_mtp" not in _COMPILED:
        _COMPILED["output_loss_mtp"] = torch.compile(_output_loss_mtp, dynamic=False)
    return _COMPILED["output_loss_mtp"](h, weight, target, extra, w)


def mtp_weights(step, total, n):
    """Adim (0'dan) -> ek hedef agirliklari [w_1 .. w_n].  Resmi takvim (modded-nanogpt kayit 53, x = adim / toplam; n 2:
    [1, 0,5, 0,25->0] -> [1, 0,5->0] -> [1]) n'ye genellenir: n + 1 esit evre; evre p < n'de k <= n - p ek hedef 0,5^k,
    en uzagi evre icinde dogrusal 0'a; son evre yalniz ana hedef (WSD inisi bu evrede: sinav saf NTP basini olcer)."""
    x = step / total
    p = sum(x >= j / (n + 1) for j in range(1, n + 1))                  # resmi kodun x < 1/3, x < 2/3 karsilastirmasi
    w = [0.5 ** k if k <= n - p else 0.0 for k in range(1, n + 1)]
    if p < n:
        w[n - p - 1] *= 1 - ((n + 1) * x - p)
    return w


def output_logprobs(model, batch, h, pos):
    """Konumlar (duz indeks) -> log p (n, VOCAB) = log_softmax(model._logits(h)) (teshis ve basvuru testleri)."""
    return torch.log_softmax(model._logits(h.flatten(0, 1)[pos]).float(), -1)


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
    baska optimizer'in parametreleri (Muon).  Bigram tablosu (model.ngram) ayri grup: wd 0, lr_mult NGRAM_LR_MULT
    (egitim dongusu lr x lr_mult yazar); model.ngram_sparse ise tablo hic yok (NgramRowAdam'da)."""
    emb = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    ng = getattr(model, "ngram", None)
    ng = {id(ng.weight)} if ng is not None else set()
    decay, no_decay, table, seen = [], [], [], {id(p) for p in skip}
    if getattr(model, "ngram_sparse", False):
        seen |= ng
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (table if id(p) in ng else decay if p.dim() >= 2 and id(p) not in emb else no_decay).append(p)
    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
    return groups + ([dict(params=table, weight_decay=0.0, lr_mult=NGRAM_LR_MULT)] if table else [])


class NgramRowAdam:
    """Bigram tablosunun seyrek satir Adam'i (--ngram_sparse 1; belge 90b ek): beta1 0, wd 0, eleman basina v.  Adimda yalniz
    model.ngram_seen satirlari (o adimda okunanlar; yaprak gradyani kirpmaya dahil) guncellenir; dokunulmayan satirin
    v'sindeki beta2 sonumu dokunuldugu adimda beta2^aralik ile topluca uygulanir.  Dokunulmayan satir AdamW(beta1 0)'da da
    hareket etmez -> yogun AdamW(betas (0, beta2), wd 0) ile ayni matematik (fp sirasi haric).  Ana optimizer'i sarar:
    param_groups (tablo grubu lr_mult'lu) / step / zero_grad / state_dict ikisini birden."""

    def __init__(self, base, model, lr, beta2, eps=1e-8):
        self.base, self.m = base, model
        w = model.ngram.weight
        self.group = dict(params=[w], lr=lr, lr_mult=NGRAM_LR_MULT, beta2=beta2, eps=eps, weight_decay=0.0)
        self.v = torch.zeros_like(w)
        self.last = torch.zeros(w.shape[0], dtype=torch.long, device=w.device)
        self.t = 0

    @property
    def param_groups(self):
        return self.base.param_groups + [self.group]

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none)

    @torch.no_grad()
    def step(self):
        self.base.step()
        self.t += 1
        seen, self.m.ngram_seen = self.m.ngram_seen, None
        if seen is None or seen[1].grad is None:
            return
        uniq, g = seen[0], seen[1].grad.float()
        b2, t = self.group["beta2"], self.t
        v = self.v[uniq] * (b2 ** (t - self.last[uniq]).double()).float()[:, None] + (1 - b2) * g * g
        self.v[uniq], self.last[uniq] = v, t
        w = self.m.ngram.weight
        w[uniq] = w[uniq] - self.group["lr"] * g / (v.sqrt() / math.sqrt(1 - b2 ** t) + self.group["eps"])

    def state_dict(self):
        return dict(base=self.base.state_dict(), ngram=dict(v=self.v, last=self.last, t=self.t, lr=self.group["lr"]))

    def load_state_dict(self, state):
        self.base.load_state_dict(state["base"])
        n = state["ngram"]
        self.v.copy_(n["v"])
        self.last.copy_(n["last"])
        self.t, self.group["lr"] = n["t"], n["lr"]


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


def _ns_batched(G, coef, steps, eps):
    """torch.optim._muon._zeropower_via_newtonschulz'un yigin hali: G (n, A, B) ayni bicimli n matris, bf16."""
    a, b, c = coef
    X = G.bfloat16()
    tall = G.size(1) > G.size(2)
    if tall:
        X = X.transpose(1, 2)
    X = X / X.flatten(1).norm(dim=1).clamp(min=eps)[:, None, None]
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        X = torch.baddbmm(X, torch.baddbmm(A, A, A, beta=b, alpha=c), X, beta=a)
    return X.transpose(1, 2) if tall else X


class BatchedMuon(torch.optim.Muon if hasattr(torch.optim, "Muon") else torch.optim.Optimizer):
    """torch.optim.Muon ile ayni matematik ve ayni durum (momentum_buffer); Newton-Schulz ayni bicimli matrislerde tek
    bmm yiginiyla (parametre basina ayri matmul yerine).  CPU'da torch Muon ile bit esit olculdu (d 512 sekilleri, tests_v2
    recipe); GPU'da olculmedi, bmm toplama sirasi addmm'den farkli olabilir."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            ps, grads, bufs = [], [], []
            self._init_group(g, ps, grads, bufs)
            if not ps:
                continue
            torch._foreach_lerp_(bufs, grads, 1 - g["momentum"])
            ups = torch._foreach_lerp(grads, bufs, g["momentum"]) if g["nesterov"] else bufs
            torch._foreach_mul_(ps, 1 - g["lr"] * g["weight_decay"])
            shapes = {}
            for i, p in enumerate(ps):
                shapes.setdefault(tuple(p.shape), []).append(i)
            for shape, idx in shapes.items():
                O = _ns_batched(torch.stack([ups[i] for i in idx]), g["ns_coefficients"], g["ns_steps"], g["eps"])
                self._apply(g, shape, [ps[i] for i in idx], O)
        return loss

    def _apply(self, g, shape, ps, O):
        """Ayni bicimli matrislerin NS ciktisi O (k, m, n) -> W -= alr O (torch Muon'un _adjust_lr olcegi)."""
        from torch.optim._muon import _adjust_lr
        alr = _adjust_lr(g["lr"], g["adjust_lr_fn"], torch.Size(shape))
        for j, p in enumerate(ps):
            p.add_(O[j], alpha=-alr)


NORMUON_BETA2 = 0.95      # li2025_normuon s4 deney ayari (beta1, beta2) = (0,95, 0,95); resmi kod varsayilani da 0,95
NORMUON_EPS = 1e-10       # makalede deger yok; resmi kod (github.com/zichongli5/NorMuon, normuon.py) sqrt(v) + 1e-10


def _momentum_stack(grads, bufs, momentum, nesterov):
    """Ayni bicimli grup: momentum (bufs yerinde) + nesterov -> NS girdisi yigin (k, m, n) bf16."""
    torch._foreach_lerp_(bufs, grads, 1 - momentum)
    ups = torch._foreach_lerp(grads, bufs, momentum) if nesterov else bufs
    return torch.stack(ups).bfloat16()


def _normuon_post(O, v, beta2, scale):
    """NorMuon satir 7-10, yigin: O (k, m, n), v (k, m, 1) -> (guncelleme O^ eta^ (k, m, n) fp32, yeni v)."""
    O = O.float()
    v = v.lerp(O.square().mean(-1, keepdim=True), 1 - beta2)
    Oh = O / (v.sqrt() + NORMUON_EPS)
    eta = scale / Oh.flatten(1).norm(dim=1).clamp_min(NORMUON_EPS)                 # O 0: NaN yok
    return Oh * eta[:, None, None], v


def _opt_fn(fn, cuda):
    """CUDA'da derlenmis (fonksiyon basina bir kez sarilir; eleman isleri birlesir, belge 94 s10.3), CPU'da eager."""
    if not cuda:
        return fn
    if fn.__name__ not in _COMPILED:
        _COMPILED[fn.__name__] = torch.compile(fn, dynamic=False)
    return _COMPILED[fn.__name__]


class NorMuon(BatchedMuon):
    """NorMuon (li2025_normuon, Algorithm 1): momentum ve NS BatchedMuon'la ayni (nesterov dahil, resmi kod gibi), sonra
    satir (cikti noronu) basina ikinci moment v = b2 v + (1 - b2) mean_sutun(O * O) (satir 7), O^ = O / (sqrt(v) + eps)
    (satir 9), eta^ = 0,2 lr sqrt(mn) / ||O^||_F (satir 10: guncelleme RMS'i 0,2 lr, AdamW'ninki), W -= eta^ O^ (satir 11;
    wd once).  adjust_lr_fn kullanilmaz.  Durum: momentum_buffer + second_momentum_buffer (m, 1) fp32.  Ayni bicimli grup
    basina: momentum + yigin ve satir 7-10 CUDA'da derlenmis tek fonksiyon (_momentum_stack, _normuon_post); CPU'da ayni
    islemler eager (8 Ekim kodu ile bit ayni, tests_v2 normuon)."""

    def __init__(self, params, beta2=NORMUON_BETA2, **kw):
        super().__init__(params, **kw)
        for g in self.param_groups:
            g.setdefault("beta2", beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            ps, grads, bufs = [], [], []
            self._init_group(g, ps, grads, bufs)
            if not ps:
                continue
            cuda = ps[0].is_cuda
            shapes = {}
            for i, p in enumerate(ps):
                shapes.setdefault(tuple(p.shape), []).append(i)
            for (m, n), idx in shapes.items():
                pp = [ps[i] for i in idx]
                for p in pp:
                    if "second_momentum_buffer" not in self.state[p]:
                        self.state[p]["second_momentum_buffer"] = torch.zeros(m, 1, dtype=torch.float, device=p.device)
                vs = [self.state[p]["second_momentum_buffer"] for p in pp]
                G = _opt_fn(_momentum_stack, cuda)([grads[i] for i in idx], [bufs[i] for i in idx], g["momentum"],
                                                   g["nesterov"])
                torch._foreach_mul_(pp, 1 - g["lr"] * g["weight_decay"])
                O = _ns_batched(G, g["ns_coefficients"], g["ns_steps"], g["eps"])
                scale = 0.2 * g["lr"] * math.sqrt(m * n)
                U, v = _opt_fn(_normuon_post, cuda)(O, torch.stack(vs), g["beta2"], torch.tensor(scale, dtype=torch.float64)
                                                    if cuda else scale)        # CUDA: lr her adim degisir, yeniden derleme yok
                torch._foreach_copy_(vs, list(v.unbind(0)))
                torch._foreach_sub_(pp, list(U.unbind(0)))
        return loss


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
