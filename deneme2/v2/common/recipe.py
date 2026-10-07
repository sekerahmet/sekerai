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
    Torba (--bag_k; belge 53-55, adlar onayli 7 Ekim): B_k = C u P_k u L_k, iki asamali olasilik.  Dagilimin TEK tanimi
                two_stage_logprobs (sinav, uretim, teshis output_logprobs ile); egitimin hizli yolu bag_train_loss ona
                esitligiyle sinanir (tests_v2 recipe).
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import END_ID, EOS_ID, VOCAB, Kind

BAG_GROUP = 512                   # grup basina torba (ardisik; grubun aday birlesimine tek matmul)
BAG_CHUNK = 2048                  # tam sozluk satir dilimi


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
    """Train sayimi (data.token_counts; END yok) -> cekirdek C (sirali int64): en sik n token + END + EOS (belge 55 O1)."""
    order = np.argsort(-np.asarray(counts), kind="stable")
    return np.unique(np.r_[order[:n], END_ID, EOS_ID]).astype(np.int64)


def bag_mask(ids, n=VOCAB):
    return torch.zeros(n, dtype=torch.bool, device=ids.device).index_fill_(0, ids, True)


def two_stage_logprobs(logits, in_bag, other):
    """Iki asamali dagilimin TEK tanimi (belge 54 s1.1): logits (..., V), in_bag (V,) ya da (..., V) bool, other (...) DIGER
    logit'i -> log p (..., V), toplam 1.  v in B: l_v - LSE(l_B, s_O); v disinda: s_O - LSE(l_B, s_O) + l_v - LSE_disari(l).
    Disari bossa DIGER kapali (= log_softmax; belge 55 O7).  Tumleyen LSE maskeyle dogrudan (cikarma yok, belge 55 O8)."""
    logits, other = logits.float(), other.float()
    if bool(in_bag.all()):
        return torch.log_softmax(logits, -1)
    lse_in = torch.logaddexp(logits.masked_fill(~in_bag, float("-inf")).logsumexp(-1), other)
    lse_out = logits.masked_fill(in_bag, float("-inf")).logsumexp(-1)
    return torch.where(in_bag, logits - lse_in[..., None], (other - lse_in - lse_out)[..., None] + logits)


def bag_index(batch):
    """-> (ids (B, T) konumun torbasi, rows, cols): torba = ozet sutunu (BOS; Model Z'de ZTOK, transformer'da END), satir
    sirasiyla numarali.  Konumun torbasi ondan once (dahil) son ozet; hedefleri o torbadan."""
    kind = batch.kind
    summ = (kind == Kind.BOS) | (kind == Kind.ZTOK) | (kind == Kind.END)
    assert bool((kind[:, 0] == Kind.BOS).all()), "her satir BOS ile baslar"
    ids = summ.flatten().long().cumsum(0).view_as(summ) - 1
    rows, cols = summ.nonzero(as_tuple=True)
    return ids, rows, cols


def bag_copy(batch, ids, rows, cols, core):
    """-> (torba, kelime) ciftleri, tekil: ayni hikayede torbanin sutunundan once gecen TOKEN'lar, core (VOCAB bool) disi.
    Sutun c'deki kelime c'nin torbasinin hedefidir; P'ye ancak bir SONRAKI torbadan girer (sizinti yok, belge 55 O2)."""
    T, n = batch.kind.shape[1], len(rows)
    bag_key = rows * T + batch.doc[rows, cols].long()                     # satir ve hikaye; torba sirasinda artan
    r, c = (batch.kind == Kind.TOKEN).nonzero(as_tuple=True)
    v = batch.tokens[r, c]
    keep = ~core[v]
    r, c, v = r[keep], c[keep], v[keep]
    word_key, inv = torch.unique((r * T + batch.doc[r, c].long()) * VOCAB + v, return_inverse=True)
    first = torch.full_like(word_key, n).scatter_reduce(0, inv, ids[r, c] + 1, "amin")    # ilk gorulmeden sonraki torba
    last = torch.searchsorted(bag_key, word_key // VOCAB, right=True) - 1                  # hikayenin son torbasi
    lens = last - first + 1
    off = torch.arange(int(lens.sum()), device=lens.device) - torch.repeat_interleave(lens.cumsum(0) - lens, lens)
    return torch.repeat_interleave(first, lens) + off, torch.repeat_interleave(word_key % VOCAB, lens)


class Bag(torch.nn.Module):
    """Ogrenen torba (belge 53-55; kullanici, 7 Ekim): B_k = C u P_k u L_k, |P_k u L_k| <= R = k - |C| (P_k fazlasi kelime
    sirasiyla kesilir).  Secici: puan(v) = q(h_k) . sg(e_v) + b_v, bf16 matmul (egitim, sinav, uretim ayni yol; belge 55
    O5); L_k = C u P_k disi ve train'de gorulmus kelimelerden puani en yuksek R - |P_k|.  other: DIGER (s_O = h . other)."""

    def __init__(self, d, k, n_core):
        super().__init__()
        self.k = int(k)
        self.register_buffer("core", torch.zeros(n_core, dtype=torch.long))
        self.register_buffer("seen", torch.zeros(VOCAB, dtype=torch.bool))
        self.q = torch.nn.Linear(d, d, bias=False)
        torch.nn.init.normal_(self.q.weight, std=0.02)
        self.b_v = torch.nn.Parameter(torch.zeros(VOCAB))
        self.other = torch.nn.Parameter(torch.zeros(d))

    @torch.no_grad()
    def fill(self, core, counts):
        """Yeni kosu: C, train'de gorulen token'lar, b_v = log (siklik + 1) (secici siklik sirasindan baslar)."""
        cnt = torch.as_tensor(np.r_[counts, 0][:VOCAB], dtype=torch.float, device=self.core.device)
        self.core.copy_(torch.as_tensor(core))
        self.seen.copy_(cnt > 0)
        self.b_v.copy_((cnt + 1).log())

    def selectable(self):
        """-> (sidx, spos): secilebilir kelimeler (train'de gorulmus, C disi; artan V indeksi) ve V -> sira (disinda -1).
        Secici yalniz bunlari puanlar (oteki kelimeler zaten secilemez; SS'te ~%56 V)."""
        ok = self.seen & ~bag_mask(self.core)
        sidx = ok.nonzero()[:, 0]
        spos = torch.full((VOCAB,), -1, dtype=torch.long, device=ok.device)
        spos[sidx] = torch.arange(len(sidx), device=ok.device)
        return sidx, spos

    def scores(self, h, E, sidx):
        """h (n, d) -> puan (n, |sidx|) fp32; matmul bf16, autocast'ten bagimsiz."""
        with torch.autocast(h.device.type, enabled=False):
            return (self.q(h.float()).bfloat16() @ E.detach()[sidx].bfloat16().T).float() + self.b_v[sidx]

    @staticmethod
    def full_score(sel):
        """select ciktisi -> puan (n, VOCAB), secilemeyenler -inf (teshis / test icin; egitim yolu kullanmaz)."""
        out = sel["score"].new_full((len(sel["score"]), VOCAB), float("-inf"))
        out[:, sel["sidx"]] = sel["score"]
        return out

    def select(self, h, E, pairs):
        """h (n, d) ozet durumlari, pairs bag_copy -> dict: inbag (n, VOCAB), pbit (P_k, kesilmis), cand_r / ok_r (n, R) P_k
        ve L_k kelimeleri (hizli yol), score / allowed (secici kaybi: C u P_k disi, gorulmus), p_over (kesilen P)."""
        n, R, dev = len(h), self.k - len(self.core), h.device
        key = (pairs[0] * VOCAB + pairs[1]).sort().values
        bag, word = key // VOCAB, key % VOCAB
        rank = torch.arange(len(key), device=dev) - torch.searchsorted(bag, bag)
        keep = rank < R
        plist = torch.full((n, R), -1, dtype=torch.long, device=dev)
        plist[bag[keep], rank[keep]] = word[keep]
        pbit = torch.zeros(n, VOCAB, dtype=torch.bool, device=dev)
        pbit[bag[keep], word[keep]] = True
        n_p = torch.bincount(bag[keep], minlength=n)
        allowed = self.seen & ~bag_mask(self.core) & ~pbit
        sidx, spos = self.selectable()
        with torch.no_grad():                                                 # secim gradyansiz; kayip selector_loss'ta
            score = self.scores(h, E, sidx).masked_fill(pbit[:, sidx], float("-inf"))   # (n, |sidx|)
        kk = min(R, len(sidx))                                                # secilebilir < R: kalan yerler bos (-1)
        top = score.topk(kk, 1)
        j = torch.arange(kk, device=dev)[None]
        lw = torch.where(torch.isfinite(top.values) & (j < (R - n_p)[:, None]), sidx[top.indices], -1)
        lw = torch.cat([lw, lw.new_full((n, R - kk), -1)], 1)
        j = torch.arange(R, device=dev)[None]
        cand_r = torch.where(j < n_p[:, None], plist, lw.gather(1, (j - n_p[:, None]).clamp_min(0)))
        ok_r = cand_r >= 0
        inbag = pbit.clone()
        inbag[:, self.core] = True
        rows_, cols_ = ok_r.nonzero(as_tuple=True)
        inbag[rows_, cand_r[rows_, cols_]] = True
        return dict(inbag=inbag, pbit=pbit, cand_r=cand_r, ok_r=ok_r, score=score, sidx=sidx, spos=spos, allowed=allowed,
                    p_over=(~keep).sum(), h=h, E=E)

    def batch_select(self, batch, h, E):
        """PackedBatch, h (B, T, d) -> select'in ciktisi + ids (B, T), rows, cols."""
        ids, rows, cols = bag_index(batch)
        sel = self.select(h[rows, cols], E, bag_copy(batch, ids, rows, cols, bag_mask(self.core)))
        return dict(sel, ids=ids, rows=rows, cols=cols)

    def select_one(self, h, E, past):
        """Uretim: h (d,) ozet durumu, past kapanmis cumlelerin token'lari -> inbag (VOCAB,)."""
        w = torch.as_tensor(sorted(set(past)), dtype=torch.long, device=h.device)
        w = w[~bag_mask(self.core)[w]]
        return self.select(h[None], E, (torch.zeros_like(w), w))["inbag"][0]

    def selector_loss(self, sel, bag, y, keep=None):
        """Secici kaybi (belge 54 s4.1, token basina havuz): izinli (C u P_k disi, gorulmus) hedeflerde -log softmax_izinli
        (puan)[y] -> (toplam, sayi).  keep (n,) bool: kayip yalniz bu torbalarda (--bag_sel_frac; secim butun torbalarda).
        Puan gradyanla yalniz kayba giren torbalar icin yeniden hesaplanir (secimin buyuk puan tensoru geri yayilmaz)."""
        n, dev = len(sel["h"]), bag.device
        if keep is None:
            rows, hs, pb = torch.arange(n, device=dev), sel["h"], sel["pbit"][:, sel["sidx"]]
        else:
            rows = keep.nonzero()[:, 0]
            hs, pb = sel["h"][rows], sel["pbit"][rows][:, sel["sidx"]]
        rmap = torch.full((n,), -1, dtype=torch.long, device=dev)
        rmap[rows] = torch.arange(len(rows), device=dev)
        m = sel["allowed"][bag, y] & (rmap[bag] >= 0)
        b, t = rmap[bag[m]], sel["spos"][y[m]]                                # izinli => secilebilir (spos >= 0)
        s = self.scores(hs, sel["E"], sel["sidx"]).masked_fill(pb, float("-inf"))
        return (s.logsumexp(1)[b] - s[b, t]).sum(), m.sum()


def attach_bag(model, k, n_core):
    """Modele model.bag (Bag) ilk agirliklardan SONRA (ana modelin ilk degerleri degismez, belge 55 K2)."""
    model.bag = Bag(model.E.weight.shape[1], k, n_core).to(model.E.weight.device)
    return model


def bag_full_mask(target, frac, rng, sel_frac=1.0):
    """CPU, veriden (rng adim tohumlu) -> (B * T,) uint8: bit 0 tam softmax payi (gecerli hedeflerden round(frac * n)),
    bit 1 secici kaybi ornegi (konum sel_frac olasilikla; torba, ozet sutunu isaretliyse kayba girer).  sel_frac 1'de rng
    yalniz pay icin kullanilir (eski kosularla ayni sira)."""
    t = np.asarray(target).reshape(-1)
    vi = np.flatnonzero(t >= 0)
    full = np.zeros(len(t), bool)
    full[rng.choice(vi, int(round(frac * len(vi))), replace=False)] = True
    sel = rng.random(len(t)) < sel_frac if sel_frac < 1 else np.ones(len(t), bool)
    return torch.as_tensor(full.astype(np.uint8) | (sel.astype(np.uint8) << 1))


def _bag_group(hg, eu, mem, u, so, valid):
    """Torba grubu (ardisik torbalar = duz konum dilimi): hg (P, d), eu (|U|, d) grubun aday birlesimi U, mem (P, |U|)
    konumun kendi torbasinda mi, u (P,) hedefin U sirasi (-1: U disi), so (P,) DIGER logit'i -> (-log p toplami, p(DIGER)
    toplami).  Tek matmul; kopya yok (7 Ekim profili: torba basina aday toplama + geri dagitim adimin ~%40'i)."""
    lg = (hg @ eu.T).float()
    lse = torch.logaddexp(lg.masked_fill(~mem, float("-inf")).logsumexp(-1), so)
    uc = u.clamp_min(0)[:, None]
    inside = (u >= 0) & mem.gather(1, uc)[:, 0]
    t = torch.where(inside, lg.gather(1, uc)[:, 0], so)
    return torch.where(valid, lse - t, 0.0).sum(), torch.where(valid, (so - lse).exp(), 0.0).sum()


def _bag_full(h, weight, target, inbag, miss, full):
    """Tam sozluk satirlari (dilim, V) -> (kacanlarin -log softmax_disari[t] toplami, tam softmax CE toplami)."""
    lg = (h @ weight.T).float()
    t = lg.gather(1, target[:, None])[:, 0]
    out = lg.masked_fill(inbag, float("-inf")).logsumexp(-1) - t
    return torch.where(miss, out, 0.0).sum(), torch.where(full, lg.logsumexp(-1) - t, 0.0).sum()


def _compiled(name, fn, h, dynamic=False):
    """CUDA'da derlenmis (bir kez sarilir), CPU'da eager (output_loss ile ayni kural).  dynamic: boyu adimdan adima degisen
    girdiler (torba gruplari) icin tek grafik."""
    if not h.is_cuda:
        return fn
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(fn, dynamic=dynamic)
    return _COMPILED[name]


def bag_train_loss(model, batch, h, full, weight, timer=None):
    """Egitimin hizli yolu (torbali model).  h (B, T, d), full (B * T,) tam softmax payi -> (amac, nll, ek).  nll = iki
    asamali dagilimin ortalama NLL'i (output_logprobs ile ayni; sinav tanimi): BAG_GROUP ardisik torbanin aday birlesimi U'ya
    tek matmul + konum maskesi (E'den U'lar tek toplamayla), tam sozluk yalniz kacan + pay satirlarinda (BAG_CHUNK).  amac = nll + pay
    satirlarinin tam CE toplami / gecerli hedef + weight x secici kaybi.  Tek GPU senkronu (dilim sayilari).  timer: secici
    suresi icin iki CUDA olayi."""
    bag, E, dev = model.bag, model.E.weight, h.device
    flags = full.to(torch.uint8) | 2 if full.dtype == torch.bool else full   # bool: yalniz tam pay (secici kaybi her torbada)
    full = (flags & 1).bool()
    T = batch.kind.shape[1]
    if timer is not None:
        timer[0].record()
    sel = bag.batch_select(batch, h, E)
    if timer is not None:
        timer[1].record()
    hf, tgt, idf = h.flatten(0, 1), batch.target.flatten(), sel["ids"].flatten()
    valid = tgt >= 0
    y = tgt.clamp_min(0)
    inb = sel["inbag"][idf, y] & valid
    miss, full = valid & ~inb, full & valid
    n = len(sel["rows"])
    core = bag.core
    inbag = sel["inbag"]
    # Torbalar duz konum sirasinda ardisik: grup g = torbalar [b0, b1) = konumlar [start[b0], start[b1]).
    start = torch.cat([sel["rows"] * T + sel["cols"], torch.tensor([len(hf)], device=dev)])
    groups = list(range(0, n, BAG_GROUP)) + [n]
    ng = len(groups) - 1
    nz = torch.stack([inbag[b0:b1].any(0) for b0, b1 in zip(groups[:-1], groups[1:])]).nonzero()   # (grup, kelime)
    host = torch.cat([start[groups], torch.bincount(nz[:, 0], minlength=ng)]).tolist()  # tek senkron: sinirlar + |U_g|
    bounds, sizes = host[:ng + 1], host[ng + 1:]
    Us = torch.split(nz[:, 1], sizes)
    # Parcalar split ile (geri: tek birlestirme).  Dilim / indeks geri yayilimi her parca icin tam boy sifir gradyan acar
    # (7 Ekim kisa profil: SliceBackward + add_ + fill_ ~24 ms/adim).
    psz = [bounds[g + 1] - bounds[g] for g in range(ng)]
    h_parts = torch.split(hf, psz)
    e_parts = torch.split(E[nz[:, 1]], sizes)                            # TEK toplama: geride tek dagitim
    so_parts = torch.split((hf @ bag.other).float(), psz)
    upos = torch.full((VOCAB,), -1, dtype=torch.long, device=dev)
    term = p_other = torch.zeros((), device=dev)
    for g, (b0, b1) in enumerate(zip(groups[:-1], groups[1:])):
        s0, s1, U = bounds[g], bounds[g + 1], Us[g]
        upos.fill_(-1)
        upos[U] = torch.arange(len(U), device=dev)
        mem = inbag[b0:b1][:, U][idf[s0:s1] - b0]
        a, b = _compiled("bag_group", _bag_group, h, True)(h_parts[g], e_parts[g], mem, upos[y[s0:s1]], so_parts[g],
                                                            valid[s0:s1])
        term, p_other = term + a, p_other + b
    rows = (miss | full).nonzero()[:, 0]
    pad = -len(rows) % BAG_CHUNK
    ok = torch.cat([torch.ones_like(rows, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool, device=dev)])
    rows = torch.cat([rows, rows.new_zeros(pad)])
    hr = torch.split(hf[rows], BAG_CHUNK)                                # tek indeks (geri tek dagitim)
    full_sum = torch.zeros((), device=dev)
    for c, r in enumerate(range(0, len(rows), BAG_CHUNK)):
        i, k = rows[r:r + BAG_CHUNK], ok[r:r + BAG_CHUNK]
        o, f = _compiled("bag_full", _bag_full, h)(hr[c], E, y[i], sel["inbag"][idf[i]], miss[i] & k, full[i] & k)
        term, full_sum = term + o, full_sum + f
    ls, n_s = bag.selector_loss(sel, idf[valid], y[valid], ((flags & 2) > 0)[start[:-1]])
    n_valid = valid.sum().clamp_min(1)
    nll = term / n_valid
    goal = nll + full_sum / n_valid + weight * ls / n_s.clamp_min(1)
    src_c = valid & bag_mask(core)[y]
    src_p = valid & sel["pbit"][idf, y]
    extra = dict(valid=n_valid.detach(), c=src_c.sum(), p=src_p.sum(), l=(inb & ~src_c & ~src_p).sum(), miss=miss.sum(),
                 p_other=p_other.detach(), full=full_sum.detach(), n_full=full.sum(), loss_selector=ls.detach(), n_selector=n_s,
                 p_over=sel["p_over"])
    return goal, nll.detach(), extra


def output_logprobs(model, batch, h, pos, sel=None):
    """Konumlar (duz indeks) -> log p (n, V): torbasiz log_softmax(h E^T); torbali two_stage_logprobs, konumun kendi B_k'si
    (sel: Bag.batch_select; yoksa kurulur)."""
    hp = h.flatten(0, 1)[pos]
    lg = (hp @ model.E.weight.T).float()
    if not hasattr(model, "bag"):
        return torch.log_softmax(lg, -1)
    sel = model.bag.batch_select(batch, h, model.E.weight) if sel is None else sel
    return two_stage_logprobs(lg, sel["inbag"][sel["ids"].flatten()[pos]], hp @ model.bag.other)


def bag_loss_per_target(model, batch, h, chunk):
    """Torbali modelin sinavi (loss_per_target sozlesmesi): -> (nll, pred) hedefli konumlarda, iki asamali dagilim."""
    sel = model.bag.batch_select(batch, h, model.E.weight)
    pos = (batch.target.flatten() >= 0).nonzero()[:, 0]
    tgt = batch.target.flatten()[pos]
    nll, pred = torch.empty(len(pos), device=h.device), torch.empty(len(pos), dtype=torch.long, device=h.device)
    for r in range(0, len(pos), chunk):
        lp = output_logprobs(model, batch, h, pos[r:r + chunk], sel)
        nll[r:r + chunk] = -lp.gather(1, tgt[r:r + chunk, None])[:, 0]
        pred[r:r + chunk] = lp.argmax(-1)
    return nll, pred


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
