# -*- coding: utf-8 -*-
"""internals_y -- egitilmis Model Y'nin ici: kalici analiz araci (kullanici, 29 Eylul: "sürekli kullanılan birşey").
Modeli DEGISTIRMEZ.  Elle yazilmis ileri hesap (_run: Block.forward'in aynisi; test: skor = model.logits) uzerinde kayit
ve mudahale.

    trace             token basina noktanin yolu: h0 = PL, her turda attention (A) ve FactUnits (F) sonrasi durum, alt
                      adimlar arasi aci, her durumda logit lens (dogru sonraki token'in sirasi ve olasiligi)
    point_drift       PF -> PL acisi token sikligina gore (bantlar), en yakin komsu kosinusu (PF'ninki rastgele duzey)
    attention_stats   tur ve head basina entropi, son 4 konumun kutlesi, ortalama uzaklik, konum 0, head ortusmesi
    unit_usage        tur basina FactUnits birimlerinin enerji payi, olu birim, birbirine cok benzeyen birimler
    ablate            parca kapatma / alpha degistirme -> Δnll, Δacc, hikaye duzeyinde eslesmis standart hata.  BAGIMLILIK
                      olcer; "o parca olmadan egitilseydi" DEGIL
    over_checkpoints  ayni olcum kosunun checkpoint_tNNNNNN.pt yedekleri boyunca (EMA optimizer durumundan kurulur)

Hikayeler: id listeleri.  Girdi ids[:-1], hedef ids[1:]; exclude_last: son hedef (SimpleStories'te hikayeyi kapatan <eos>)
sayilmaz.  Turlar kodda 0'dan, metinde 1'den (tur 3 = indeks 2); head'ler 0'dan.

    python internals_y.py <kosu adi | klasoru> <olcum> [secenekler]        (python internals_y.py -h)
Cikti <kosu klasoru>/internals/<olcum>_<agirlik>[_tNNNNNN]_<zaman>.txt + .json.  Yerel CPU, 4 is parcacigi.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model_y import BlockModel, deviation  # noqa: E402

_RUNS_ROOT = "G:/Drive'ım/model_y"                  # kosu klasorleri (Drive, bu bilgisayarda)
_SIMPLESTORIES_ROOT = "G:/Drive'ım/simplestories"    # <tag>/tokenizer.json, <tag>/valid.npy, exam_stories.npy
# train_y.train_seq'teki Muon listesi: yedekteki optimizer parametre sirasi buna bagli (once bunlar)
_MUON_HIDDEN = ("W_context", "W_value", "W_fact_in", "W_fact_up", "W_fact_out", "W_value.weight", "W_out.weight",
                "W_mlp_in.weight", "W_mlp_out.weight")
_PLAN_KEYS = ("skip_attention", "skip_facts", "heads", "canon", "alpha_attention", "alpha_facts")
_DEPENDENCE_NOTE = ("Kapatma BAGIMLILIK olcer: egitilmis model o parcaya ne kadar dayaniyor.  'O parca olmadan "
                    "egitilseydi' sorusunu CEVAPLAMAZ; o, parcasiz egitim kosusuyla olculur.")
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _unit(v):
    return F.normalize(v, dim=-1)


def _angle(a, b):
    """Iki durumun yonleri arasi aci (derece), son eksende.  2 asin(|a - b| / 2): 1 yakininda acos'tan kararli."""
    return torch.rad2deg(2 * torch.asin(((_unit(a) - _unit(b)).norm(dim=-1) / 2).clamp(max=1)))


def _work(t):
    return t.to(torch.promote_types(t.dtype, torch.float32))


# ---- elle ileri hesap

def _turn_facts(model, t):
    """Tur t'nin FactUnits'i: SHARED_FACTS=False'ta tur >= layers kendi takimini (extra_facts) kullanir."""
    if not model.shared_facts and t >= model.layers:
        return model.extra_facts[t - model.layers]
    return model.turn_blocks()[t].facts


def _turn_label(model, t):
    """'3C': tur 3, blok C.  Ayri FactUnits takimi varsa not dusulur."""
    own = "" if (model.shared_facts or t < model.layers) else ", ayri FactUnits"
    return "%d%s%s" % (t + 1, _LETTERS[t % len(model.blocks)], own)


def _units(f, v):
    """FactUnits birim etkinlikleri u (.., units); cikti u @ W_fact_out.T (FactUnits.forward ile ayni)."""
    if f.activation == "swiglu":
        return F.silu(v @ f.W_fact_in.T) * (v @ f.W_fact_up.T)
    u = torch.relu(v @ f.W_fact_in.T - f.fact_threshold)
    return u * (v @ f.W_fact_up.T) if f.activation == "reglu" else u


def _plan(model, case=None):
    """Mudahale sozlugu -> tam plan.  Anahtarlar (tur 0'dan):
        skip_attention, skip_facts   tur listesi: alt blogun katkisi sifir (normalized_update'te alpha 0 ile ayni)
        heads      {(tur, head): (dh,) vektor | None}: head'in ciktisi c_h bu vektorle degisir; None = ortalama (ablate doldurur)
        canon      {tur: (4,) carpan}: Canon agirliklari w_k bu carpanla (0 = kapali)
        alpha_attention, alpha_facts   {tur: sayi | (d,) vektor}: o turun alpha'si ELLE (yalniz normalized_update)
    plan["start"]: mudahalenin ilk turu (onceki turlar temiz hesapla ayni)."""
    case = dict(case or {})
    bad = set(case) - set(_PLAN_KEYS)
    assert not bad, "bilinmeyen mudahale %s (gecerli: %s)" % (sorted(bad), _PLAN_KEYS)
    plan = dict(skip_attention=set(case.get("skip_attention", ())), skip_facts=set(case.get("skip_facts", ())),
                heads=dict(case.get("heads") or {}), canon=dict(case.get("canon") or {}))
    touched = list(plan["skip_attention"] | plan["skip_facts"]) + [t for t, _ in plan["heads"]] + list(plan["canon"])
    for name in ("alpha_attention", "alpha_facts"):
        given = case.get(name) or {}
        plan[name] = None
        if given:
            assert model.normalized_update, "alpha yalniz normalized_update'te var"
            a = getattr(model, name).detach().clone()
            for t, v in given.items():
                a[t] = torch.as_tensor(v, dtype=a.dtype)
            plan[name] = a
            touched += list(given)
    H = model.blocks[0].attention.heads
    assert all(0 <= t < model.turns for t in touched), "tur 0..%d disinda" % (model.turns - 1)
    assert all(0 <= h < H for _, h in plan["heads"]), "head 0..%d disinda" % (H - 1)
    plan["start"] = min(touched, default=model.turns)
    return plan


def _run(model, h, plan, taps=None, start=0):
    """Turlar start..TURNS-1; h tur start'in girdisi (B, T, d) -> son durum.  Block.forward'in aynisi (paylasilan / ayri
    blok, SHARED_FACTS, Canon, head'ler, normalized_update, stream_norm, LayerNorm) + mudahale (plan) + kayit:
    taps[ad](tur, deger), ad: input, canon (ek, girdi), attention (B, H, T, T), heads (B, H, T, dh), attention_out, units,
    facts_out.  attention kaydi istenirse softmax acik hesaplanir (CausalAttention.weights), degilse SDPA."""
    taps = taps or {}
    blocks = model.turn_blocks()
    for t in range(start, model.turns):
        blk = blocks[t]
        at = blk.attention
        norm_a, norm_f = (blk.norm_attention, blk.norm_facts) if blk.layer_norm else (_unit, _unit)
        if "input" in taps:
            taps["input"](t, h)
        x = h if blk.stream_norm else norm_a(h)
        if blk.canon:                                  # x_t + sum_k w_k x_(t-k), baslangictan once 0
            w = blk.canon_weights
            if t in plan["canon"]:
                w = w * torch.as_tensor(plan["canon"][t], dtype=w.dtype)[:, None]
            T = x.shape[-2]
            full = F.pad(x, (0, 0, 3, 0))
            mix = sum(w[k] * full[..., 3 - k:3 - k + T, :] for k in range(4))
            if "canon" in taps:
                taps["canon"](t, (mix, x))
            x = x + mix
        added = None
        if t not in plan["skip_attention"]:
            H = at.heads
            q, k = at.queries_keys(x)
            v = x if H == 1 else (x @ at.W_value.T).unflatten(-1, (H, -1)).transpose(-3, -2)
            if "attention" in taps:
                T = x.shape[-2]
                s = at.scale * q @ k.transpose(-1, -2)
                s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
                a = torch.softmax(s, -1)
                taps["attention"](t, a if H > 1 else a[:, None])
                c = a @ v
            else:
                c = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scale)
            if H == 1:
                c = c[:, None]                         # (B, 1, T, d): tek head
            for (tt, hh), vec in plan["heads"].items():
                if tt == t:
                    assert vec is not None, "head ortalamasi doldurulmadi (ablate doldurur)"
                    c[:, hh] = torch.as_tensor(vec, dtype=c.dtype)
            if "heads" in taps:
                taps["heads"](t, c)
            added = c.transpose(-3, -2).flatten(-2) @ at.W_context.T
        if blk.normalized_update:                      # h = norm(h + α_A ⊙ (norm(W_context c) - h))
            if added is not None:
                a_att = model.alpha_attention[t] if plan["alpha_attention"] is None else plan["alpha_attention"][t]
                h = _unit(h + a_att * (_unit(added) - h))
        elif blk.stream_norm:
            h = norm_a(h if added is None else h + added)
        elif added is not None:
            h = h + added
        if "attention_out" in taps:
            taps["attention_out"](t, h)
        if t not in plan["skip_facts"]:
            f = _turn_facts(model, t)
            v = h if blk.stream_norm else norm_f(h)
            if blk.sphere_weights:
                v = v * v.shape[-1] ** 0.5
            u = _units(f, v)
            if "units" in taps:
                taps["units"](t, u)
            out = u @ f.W_fact_out.T
            if blk.normalized_update:                  # h = norm(h + α_F ⊙ (norm(olgu) - h))
                a_f = model.alpha_facts[t] if plan["alpha_facts"] is None else plan["alpha_facts"][t]
                h = _unit(h + a_f * (_unit(out) - h))
            elif blk.stream_norm:
                h = norm_f(h + out)
            else:
                h = h + out
        elif blk.stream_norm and not blk.normalized_update:
            h = norm_f(h)
        if "facts_out" in taps:
            taps["facts_out"](t, h)
    return h


def _scores(model, h, P=None):
    """Cikis basligi (BlockModel.logits ile ayni): e^tau phi(<h, PL>); ara durumlara uygulaninca logit lens."""
    P = model.tokens.points() if P is None else P
    if model.layer_norm:
        return model.norm_final(h) @ P.T
    if not model.stream_norm:
        h = _unit(h)
    scale = (model.scale * torch.exp(model.log_output_scale - math.log(model.scale)) if model.learn_output_scale
             else model.scale)
    if model.output_link:
        c = h @ P.T
        q, u = model.link_q, model.link_u
        return scale * (c * (1 + c * (q + c * (q * q / 3 + u))))
    return scale * h @ P.T


@torch.no_grad()
def _logits(model, ids, case=None, taps=None):
    """Elle ileri hesabin skoru (mudahaleli olabilir); mudahalesiz = model.logits (test)."""
    P = model.tokens.points()
    return _scores(model, _run(model, P[ids], _plan(model, case), taps), P)


# ---- hikayeler ve puanlama

def _as_stories(ids):
    """Bir id listesi, id listeleri ya da tensor (1B: bir hikaye; 2B: her satir bir hikaye) -> id listeleri."""
    if torch.is_tensor(ids):
        ids = ids[None] if ids.dim() == 1 else ids
        return [row.tolist() for row in ids]
    ids = list(ids)
    if ids and isinstance(ids[0], (int, np.integer)):
        ids = [ids]
    stories = [[int(x) for x in s] for s in ids]
    assert stories and min(len(s) for s in stories) >= 2, "her hikayede en az 2 token (girdi + hedef)"
    return stories


def _batches(stories, batch, exclude_last):
    """Boya gore sirali batch'ler: (x (B, L), valid (B, L - 1), idx).  Dolgu sagda: nedensel attention'da oncekileri
    etkilemez.  valid: sayilan hedefler (exclude_last: son hedef haric)."""
    order = sorted(range(len(stories)), key=lambda i: len(stories[i]))
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        L = max(len(stories[j]) for j in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        valid = torch.zeros(len(idx), L - 1, dtype=torch.bool)
        for r, j in enumerate(idx):
            x[r, :len(stories[j])] = torch.tensor(stories[j])
            valid[r, :len(stories[j]) - 1 - int(exclude_last)] = True
        yield x, valid, idx


def _score_rows(z, y, valid):
    """-> satir basina (nll toplami, dogru sayisi, hedef sayisi), float64 numpy."""
    z = _work(z)
    nll = -torch.log_softmax(z, -1).gather(-1, y[..., None])[..., 0]
    hit = z.argmax(-1) == y
    zero = torch.zeros((), dtype=z.dtype)
    return (torch.where(valid, nll, zero).sum(-1).double().numpy(), (hit & valid).sum(-1).double().numpy(),
            valid.sum(-1).double().numpy())


def _paired(base, other):
    """Hikaye duzeyinde eslesmis fark: Δ = sum_i d_i / N, se = sqrt(sum_i (d_i - Δ k_i)^2) / N (oran kestiricisi);
    d_i hikaye i'nin toplam farki, k_i hedef sayisi, N = sum k_i.  Δacc oran (puan icin x100)."""
    k = base["count"]
    N = k.sum()
    out = {}
    for key, a, b in (("nll", other["nll_sum"], base["nll_sum"]), ("acc", other["hit_sum"], base["hit_sum"])):
        d = a - b
        mean = d.sum() / N
        out["d_" + key] = float(mean)
        out["se_" + key] = float(math.sqrt(((d - mean * k) ** 2).sum()) / N)
    return out


# ---- olcumler

@torch.no_grad()
def trace(model, ids, exclude_last=True, batch=16, keep_logits=False):
    """Token basina noktanin yolu.  ids: id listesi, id listeleri ya da tensor.
    Durumlar: h0 = PL[token]; her tur t: A<t> attention alt adimindan, F<t> FactUnits'ten sonra (= tur ciktisi).  Her
    durumda: bir onceki duruma aci (derece), yon kosinusu <h, PL(girdi)> ve <h, PL(hedef)>, logit lens -- ayni cikis
    basligiyla dogru sonraki token'in sirasi (1 = en yuksek skor), olasiligi ve en yuksek skorlu token.
    -> dict(states, blocks, summary {durum: gecerli hedeflerde ortalama}, stories [{ids, valid, rank, p, top, angle}],
    n, check_logits: son durumun skoru ile model.logits arasi en buyuk fark; keep_logits: logits [(L - 1, V)])."""
    stories = _as_stories(ids)
    names = ["h0"] + ["%s%d" % (k, t + 1) for t in range(model.turns) for k in ("A", "F")]
    P = model.tokens.points()
    sums = {n: dict(angle=0.0, nll=0.0, hit=0.0, p=0.0, cos_input=0.0, cos_target=0.0) for n in names}
    ranks = {n: [] for n in names}
    rows, logits = [None] * len(stories), [None] * len(stories)
    count, check = 0, 0.0
    plan = _plan(model)
    for x, valid, idx in _batches(stories, batch, exclude_last):
        inp, y = x[:, :-1], x[:, 1:]
        states = [P[inp]]
        keep = lambda t, h: states.append(h)
        _run(model, states[0], plan, dict(attention_out=keep, facts_out=keep))
        check = max(check, float((model.logits(inp) - _scores(model, states[-1], P)).abs().max()))
        for r, i in enumerate(idx):
            rows[i] = dict(ids=stories[i], valid=valid[r, :len(stories[i]) - 1].tolist(), rank={}, p={}, top={}, angle={})
        count += int(valid.sum())
        for j, (name, s) in enumerate(zip(names, states)):
            z = _scores(model, s, P)
            zt = z.gather(-1, y[..., None])
            rank = 1 + (z > zt).sum(-1)
            lp = torch.log_softmax(_work(z), -1).gather(-1, y[..., None])[..., 0]
            top = z.argmax(-1)
            angle = _angle(s, states[j - 1]) if j else torch.zeros_like(lp)
            cos_in, cos_tg = (_unit(s) * P[inp]).sum(-1), (_unit(s) * P[y]).sum(-1)
            acc = sums[name]
            for key, val in (("angle", angle), ("nll", -lp), ("hit", (top == y).double()), ("p", lp.exp()),
                             ("cos_input", cos_in), ("cos_target", cos_tg)):
                acc[key] += float(val[valid].double().sum())
            ranks[name] += rank[valid].tolist()
            for r, i in enumerate(idx):
                L = len(stories[i]) - 1
                row = rows[i]
                row["rank"][name] = rank[r, :L].tolist()
                row["p"][name] = [round(float(v), 6) for v in lp[r, :L].exp()]
                row["top"][name] = top[r, :L].tolist()
                row["angle"][name] = [round(float(v), 3) for v in angle[r, :L]]
                if keep_logits and j == len(names) - 1:
                    logits[i] = z[r, :L].clone()
    summary = {}
    for n in names:
        a = sums[n]
        summary[n] = dict(angle=a["angle"] / count if n != "h0" else None, nll=a["nll"] / count, acc=a["hit"] / count,
                          median_rank=float(np.median(ranks[n])), p=a["p"] / count, cos_input=a["cos_input"] / count,
                          cos_target=a["cos_target"] / count)
    out = dict(states=names, blocks=["-"] + [_LETTERS[t % len(model.blocks)] for t in range(model.turns) for _ in "AF"],
               summary=summary, stories=rows, n=count, check_logits=check)
    if keep_logits:
        out["logits"] = logits
    return out


def _nearest(P, chunk=2048):
    """Her satir icin kosinusu en yuksek BASKA satir: (kosinus, indeks); parca parca (buyuk sozlukte n x n tablo yok)."""
    best, where = [], []
    for i in range(0, P.shape[0], chunk):
        c = P[i:i + chunk] @ P.T
        r = torch.arange(c.shape[0])
        c[r, r + i] = -2
        v, j = c.max(-1)
        best.append(v)
        where.append(j)
    return torch.cat(best), torch.cat(where)


def _band_edges(V):
    """Siklik sirasi bantlari: 16, 64, 256, 1024, 4096 ve sozlugun ceyrekleri (V = 4096'da 0-16 ... 3072-4096)."""
    edges = {0, V} | {e for e in (16, 64, 256, 1024, 4096) if e < V} | {V * k // 4 for k in (1, 2, 3)}
    return sorted(edges)


@torch.no_grad()
def point_drift(model, counts=None, vocab=None, bands=None, examples=(5, 50, 500, -2048, -50)):
    """Token noktalari: PF -> PL acisi (deviation, derece), |shift|, en yakin komsu kosinusu (PL'de; PF'deki rastgele
    duzey karsilastirma icin).  counts (V,): token sayimi (ornek: train akisi) -> siklik sirasi bantlari (bands: kenarlar;
    varsayilan _band_edges), hic gorulmeyen token'lar, corr(log(1 + sayim), aci).  vocab: ornek token'larin (siklik
    sirasi examples; eksi = sondan) en yakin 3 komsusu."""
    tok = model.tokens
    P, PF = tok.points(), tok.fixed_points
    V = P.shape[0]
    dev = deviation(model).double()
    shift = tok.shift.norm(dim=-1).double()
    nn_pl, nn_idx = _nearest(P)
    nn_pf, _ = _nearest(PF)
    med = lambda v: float(np.median(v.double().numpy()))              # cift sayida ortadaki ikisinin ortalamasi
    out = dict(tokens=V, d=P.shape[1], learn=bool(tok.learn), anchor=float(tok.anchor or 0.0),
               anchor_loss=float(tok.anchor_loss()),
               overall=dict(angle_median=med(dev), angle_mean=float(dev.mean()), angle_max=float(dev.max()),
                            shift_median=med(shift), nn_cos_pl_median=med(nn_pl), nn_cos_pf_median=med(nn_pf)),
               angle=dev.float().numpy(), nn_cos=nn_pl.float().numpy(), nn_index=nn_idx.numpy())
    order = np.arange(V)
    if counts is not None:
        counts = np.asarray(counts, dtype=np.float64)
        assert counts.shape == (V,), "counts sozluk boyunda olmali (%d)" % V
        order = np.argsort(-counts, kind="stable")
        edges = list(bands) if bands is not None else _band_edges(V)
        d, s, nn = dev.numpy(), shift.numpy(), nn_pl.double().numpy()
        out["bands"] = [dict(lo=lo, hi=hi, count_median=float(np.median(counts[order[lo:hi]])),
                             angle_median=float(np.median(d[order[lo:hi]])), angle_min=float(d[order[lo:hi]].min()),
                             angle_max=float(d[order[lo:hi]].max()), shift_median=float(np.median(s[order[lo:hi]])),
                             nn_cos_median=float(np.median(nn[order[lo:hi]])))
                        for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
        zero = counts == 0
        out["unseen"] = dict(n=int(zero.sum()), angle_median=float(np.median(d[zero])) if zero.any() else None,
                             angle_max=float(d[zero].max()) if zero.any() else None)
        out["corr_log_count_angle"] = float(np.corrcoef(np.log1p(counts), d)[0, 1])
        out["count"] = counts
    if vocab is not None:
        rows = []
        for r in examples:
            if -V <= r < V:
                i = int(order[r])
                c, j = (P[i] @ P.T).index_fill_(0, torch.tensor([i]), -2).topk(3)
                rows.append(dict(token=vocab[i], rank=int(r % V), count=None if counts is None else float(counts[i]),
                                 angle=float(dev[i]), neighbors=[(vocab[int(a)], float(b)) for a, b in zip(j, c)]))
        out["examples"] = rows
    return out


@torch.no_grad()
def attention_stats(model, ids, exclude_last=True, batch=8):
    """Tur ve head basina, gecerli sorgu konumlarinda (hedefi sayilan konumlar) ortalama:
      entropy       -sum_j a_tj ln a_tj (nat);  entropy_norm: / ln(t + 1), t >= 1 (1 = duzgun dagilim)
      last4         son 4 konuma (t-3..t) dusen kutle;  self: t'nin kendisi;  first: konum 0 (hikaye basi)
      distance      sum_j a_tj (t - j): ortalama geriye bakis;  max: en buyuk agirlik
      head_norm     |c_h|: head ciktisinin boyu (W_context'ten once)
    Tur basina: canon |Canon eki| / |attention girdisi|; overlap[h1][h2] sum_j min(a_h1, a_h2) (1 = ayni desen).
    reuse: ayni Block'u kullanan iki turda ayni head'in ortusmesi."""
    stories = _as_stories(ids)
    TN, H = model.turns, model.blocks[0].attention.heads
    keys = ("entropy", "entropy_norm", "last4", "self", "first", "distance", "max", "head_norm")
    acc = {k: torch.zeros(TN, H, dtype=torch.float64) for k in keys}
    overlap = torch.zeros(TN, H, H, dtype=torch.float64)
    canon = torch.zeros(TN, dtype=torch.float64)
    nb = len(model.blocks)
    pairs = [(t, t2) for t in range(TN) for t2 in range(t + 1, TN) if t % nb == t2 % nb]
    reuse = {p: torch.zeros(H, dtype=torch.float64) for p in pairs}
    reused = {t for p in pairs for t in p}
    plan = _plan(model)
    P = model.tokens.points()
    n = n_norm = 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        inp = x[:, :-1]
        T = inp.shape[1]
        pos = torch.arange(T, dtype=torch.float64)
        wq = valid.double()[:, None, :]                          # (B, 1, T): sorgu konumu sayiliyor mu
        wn = wq * (pos >= 1)
        back = (pos[:, None] - pos[None, :]).clamp(min=0)        # t - j
        kept = {}

        def on_attention(t, a):
            if t in reused:
                kept[t] = a
            a = a.double()                                       # (B, H, T, T)
            ent = -(a * torch.log(a.clamp_min(1e-300))).sum(-1)
            local = sum(F.pad(a.diagonal(-k, -2, -1), (k, 0)) for k in range(min(4, T)))
            for key, val, w in (("entropy", ent, wq), ("entropy_norm", ent / torch.log(pos + 1).clamp_min(1e-12), wn),
                                ("last4", local, wq), ("self", a.diagonal(0, -2, -1), wq), ("first", a[..., 0], wq),
                                ("distance", (a * back).sum(-1), wq), ("max", a.max(-1).values, wq)):
                acc[key][t] += (val * w).sum((0, 2))
            for h1 in range(H):
                for h2 in range(h1 + 1, H):
                    overlap[t, h1, h2] += (torch.minimum(a[:, h1], a[:, h2]).sum(-1) * wq[:, 0]).sum()

        def on_heads(t, c):
            acc["head_norm"][t] += (c.norm(dim=-1).double() * wq).sum((0, 2))

        def on_canon(t, mix_x):
            mix, xin = mix_x
            canon[t] += ((mix.norm(dim=-1) / xin.norm(dim=-1).clamp_min(1e-12)).double() * wq[:, 0]).sum()

        _run(model, P[inp], plan, dict(attention=on_attention, heads=on_heads, canon=on_canon))
        for t1, t2 in pairs:
            reuse[(t1, t2)] += (torch.minimum(kept[t1], kept[t2]).sum(-1).double() * wq).sum((0, 2))
        n += int(valid.sum())
        n_norm += int(wn.sum())
    res = {k: (v / (n_norm if k == "entropy_norm" else n)).numpy() for k, v in acc.items()}
    overlap = overlap / n
    overlap = overlap + overlap.transpose(1, 2) + torch.eye(H, dtype=torch.float64)
    return dict(turns=[_turn_label(model, t) for t in range(TN)], heads=H, n=n, **res,
                canon=(canon / n).numpy() if model.canon else None, overlap=overlap.numpy(),
                reuse=[dict(turns=(t1, t2), overlap=(reuse[(t1, t2)] / n).numpy()) for t1, t2 in pairs])


@torch.no_grad()
def unit_usage(model, ids, exclude_last=True, batch=16, sample=4096, seed=0):
    """Tur basina FactUnits birimleri, gecerli konumlarda:
      energy        E|u|^2 token basina;  out_rms: |W_fact_out u|'nun karesel ortalamasi (norm oncesi yazim boyu)
      share         birim basina enerji payi E u_i^2 / sum_k E u_k^2;  cover: payin %50 / %90 / %99'unu tasiyan en az birim
      dead          payi < 1e-5 olan birim;  rare: |u_i| > 0,1 oldugu konum orani %1'in altinda kalan birim
      zero          (relu / reglu) u_i = 0 oldugu konum orani, ortalama
      corr_pairs    ornek konumlarda (en fazla sample; tohum seed) aktivasyon korelasyonu |r| > 0,9 / > 0,7 birim cifti
    FactUnits takimi basina (paylasilan blokta bir takim birden cok turda): agirlik benzerligi |cos| > 0,8 olan cift
    (W_fact_in / W_fact_up satirlari, W_fact_out sutunlari), takimi kullanan turlar arasi enerji payi korelasyonu,
    birinde 10 kat fazla etkin birim sayisi."""
    stories = _as_stories(ids)
    TN = model.turns
    facts = [_turn_facts(model, t) for t in range(TN)]
    U = facts[0].W_fact_out.shape[1]
    e2 = torch.zeros(TN, U, dtype=torch.float64)
    active = torch.zeros(TN, U, dtype=torch.float64)
    zeros = torch.zeros(TN, dtype=torch.float64)
    out2 = torch.zeros(TN, dtype=torch.float64)
    total = sum(len(s) - 1 - int(exclude_last) for s in stories)
    pick = torch.sort(torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:sample]).values
    samples = [[] for _ in range(TN)]
    plan = _plan(model)
    P = model.tokens.points()
    offset = n = 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        M = int(valid.sum())
        here = pick[(pick >= offset) & (pick < offset + M)] - offset

        def on_units(t, u):
            uu = u[valid].double()                               # (M, U)
            e2[t] += (uu ** 2).sum(0)
            active[t] += (uu.abs() > 0.1).double().sum(0)
            zeros[t] += (uu == 0).double().mean(1).sum()
            out2[t] += ((uu @ facts[t].W_fact_out.T.double()) ** 2).sum()
            samples[t].append(uu[here].float())

        _run(model, P[x[:, :-1]], plan, dict(units=on_units))
        offset += M
        n += M
    turns = []
    for t in range(TN):
        share = e2[t] / e2[t].sum()
        cum = torch.cumsum(torch.sort(share, descending=True).values, 0)
        S = torch.cat(samples[t]).double()
        S = (S - S.mean(0)) / S.std(0).clamp_min(1e-12)
        C = (S.T @ S / max(len(S) - 1, 1)).abs().triu(1)
        turns.append(dict(turn=_turn_label(model, t), energy=float(e2[t].sum() / n), out_rms=float((out2[t] / n) ** 0.5),
                          cover=[int((cum < q).sum()) + 1 for q in (0.5, 0.9, 0.99)], dead=int((share < 1e-5).sum()),
                          rare=int((active[t] / n < 0.01).sum()), zero=float(zeros[t] / n),
                          corr_pairs=[int((C > 0.9).sum()), int((C > 0.7).sum())], sample=len(S), share=share.numpy()))
    teams = []
    for f in dict.fromkeys(facts):                                # sirali, tekrarsiz
        used = [t for t in range(TN) if facts[t] is f]
        sim = {}
        for name, W in (("in", f.W_fact_in), ("up", getattr(f, "W_fact_up", None)), ("out", f.W_fact_out.T)):
            if W is not None:
                Wn = _unit(W.double())
                c = (Wn @ Wn.T).abs().triu(1)
                sim[name] = dict(pairs=int((c > 0.8).sum()), max=float(c.max()))
        between = []
        for i, t1 in enumerate(used):
            for t2 in used[i + 1:]:
                a, b = turns[t1]["share"], turns[t2]["share"]
                r = np.log10((a + 1e-9) / (b + 1e-9))
                between.append(dict(turns=(t1, t2), corr=float(np.corrcoef(a, b)[0, 1]), more_first=int((r > 1).sum()),
                                    more_second=int((r < -1).sum()), dead_both=int(((a < 1e-5) & (b < 1e-5)).sum())))
        teams.append(dict(turns=used, similar=sim, between=between))
    return dict(units=U, n=n, activation=facts[0].activation, turns=turns, teams=teams)


def _head_means(model, stories, exclude_last, batch):
    """Head ciktilarinin (c_h, W_context'ten once) gecerli konumlarda ortalamasi: {(tur, head): (dh,)}."""
    P = model.tokens.points()
    sums, n = {}, 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        def on_heads(t, c):
            s = c.permute(0, 2, 1, 3)[valid].double().sum(0)     # (H, dh)
            sums[t] = s if t not in sums else sums[t] + s
        _run(model, P[x[:, :-1]], _plan(model), dict(heads=on_heads))
        n += int(valid.sum())
    dtype = P.dtype
    return {(t, h): (s[h] / n).to(dtype) for t, s in sums.items() for h in range(s.shape[0])}


@torch.no_grad()
def ablate(model, ids_or_stories, cases, reference=None, exclude_last=True, batch=16, log=None):
    """Parca kapatma / alpha degistirme.  cases: [(ad, mudahale)] ya da {ad: mudahale}; mudahale _plan anahtarlariyla
    (skip_attention, skip_facts, heads, canon, alpha_attention, alpha_facts; tur 0'dan).  heads'te None: head'in ciktisi
    reference hikayelerindeki ortalamasiyla (verilmezse olculen hikayelerdeki) degisir: ortalama ablasyonu.
    Temiz ileri hesap bir kez, her turun girdisi saklanir: mudahale tur t'den basliyorsa hesap t'den surer (sonuc ayni).
    -> dict(base {nll, acc, n}, rows [{case, nll, acc, d_nll, se_nll, d_acc, se_acc, start (None: yalniz cikis)}], note,
    check_logits).
    Δ = mudahale - taban, hikaye duzeyinde eslesmis (_paired).  BAGIMLILIK olcer: _DEPENDENCE_NOTE."""
    stories = _as_stories(ids_or_stories)
    cases = list(cases.items()) if isinstance(cases, dict) else list(cases)
    P = model.tokens.points()
    clean = _plan(model)
    cache = []
    for x, valid, idx in _batches(stories, batch, exclude_last):
        hs = []
        h = _run(model, P[x[:, :-1]], clean, dict(input=lambda t, v: hs.append(v)))
        cache.append((x, valid, idx, hs + [h]))
    check = float((model.logits(cache[0][0][:, :-1]) - _scores(model, cache[0][3][-1], P)).abs().max())

    def collect(plan):
        S = len(stories)
        out = dict(nll_sum=np.zeros(S), hit_sum=np.zeros(S), count=np.zeros(S))
        for x, valid, idx, hs in cache:
            h = _run(model, hs[plan["start"]], plan, start=plan["start"])
            nll, hit, cnt = _score_rows(_scores(model, h, P), x[:, 1:], valid)
            out["nll_sum"][idx], out["hit_sum"][idx], out["count"][idx] = nll, hit, cnt
        N = out["count"].sum()
        return dict(out, nll=float(out["nll_sum"].sum() / N), acc=float(out["hit_sum"].sum() / N))

    base = collect(clean)
    ref = stories if reference is None else _as_stories(reference)
    means, rows = None, []
    for name, case in cases:
        plan = _plan(model, case)
        if any(v is None for v in plan["heads"].values()):
            if means is None:
                means = _head_means(model, ref, exclude_last, batch)
            plan["heads"] = {k: means[k] if v is None else v for k, v in plan["heads"].items()}
        r = collect(plan)
        rows.append(dict(case=name, nll=r["nll"], acc=r["acc"], start=plan["start"] if plan["start"] < model.turns else None,
                         **_paired(base, r)))
        if log:
            log(_ablate_row(rows[-1]))
    return dict(base=dict(nll=base["nll"], acc=base["acc"], n=int(base["count"].sum()), stories=len(stories)),
                rows=rows, reference=None if means is None else (
                    "olculen hikayelerden" if reference is None else "ayri %d hikayeden" % len(ref)),
                note=_DEPENDENCE_NOTE, check_logits=check)


# ---- kosu klasoru: ayarlar, agirliklar, yedekler

def _config(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _checkpoints(run_dir):
    """{adim: yol}, sirali."""
    found = {}
    for f in os.listdir(run_dir):
        m = re.fullmatch(r"checkpoint_t(\d+)\.pt", f)
        if m:
            found[int(m.group(1))] = os.path.join(run_dir, f)
    return dict(sorted(found.items()))


def _build(config):
    """config.json -> bos BlockModel (agirliklar sonra yuklenir).  output_link / shared_facts yazilmamis eski config'lerde
    yoktu (colab_simplestories'in surdurmesi gibi)."""
    setting = config.get("setting", "shared")
    assert setting in ("shared", "separate"), "yalniz BlockModel (setting shared / separate), bu kosu: %s" % setting
    kw = dict(dict(output_link=False, shared_facts=True), **config.get("model_kw", {}))
    return BlockModel(config["vocab"], seed=config.get("seed", 0), stream_norm=config["stream_norm"],
                      layer_norm=config["layer_norm"], rope=config["rope"], shared=setting == "shared",
                      normalized_update=config["normalized_update"], sphere_weights=config["sphere_weights"],
                      canon=config["canon"], **kw)


def _load_weight_ema(model, optimizer_state, config):
    """Yedekteki agirlik ortalamasi (train_seq: optimizer durumunda state[i]["weight_ema"]) -> model.  Parametre sirasi
    train_seq'in gruplariyla: muon -> once _MUON_HIDDEN matrisleri; adam + weight_decay -> once W_ matrisleri; adam ->
    named_parameters sirasi.  Ortalamasi olmayan parametre (hic gradyan almamis) agirlikta kalir: train_seq'in
    surdurmesi de oyle kurar."""
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    if config.get("optimizer", "muon") == "muon":
        first = [k for k, _ in named if k.endswith(_MUON_HIDDEN)]
    elif config.get("weight_decay"):
        first = [k for k, _ in named if k.split(".")[-1].startswith("W_")]
    else:
        first = [k for k, _ in named]
    order = first + [k for k, _ in named if k not in first]
    groups = optimizer_state["param_groups"]
    ids = [i for g in groups for i in g["params"]]
    assert len(ids) == len(order) and (len(groups) == 1 or len(groups[0]["params"]) == len(first)), \
        "yedekteki optimizer gruplari modelle tutmuyor (%d / %d parametre)" % (len(ids), len(order))
    params = dict(model.named_parameters())
    found = 0
    with torch.no_grad():
        for i, k in zip(ids, order):
            w = optimizer_state["state"].get(i, {}).get("weight_ema")
            if w is not None:
                assert w.shape == params[k].shape, k
                params[k].copy_(w)
                found += 1
    assert found, "yedekte weight_ema yok (kosu weight_ema=None): --weights last"


def _load_model(run_dir, weights="ema", step=None, config=None):
    """Kosunun modeli, CPU, eval.  step None: son agirlik (weights ema: model_weight_ema.pt, last: model.pt); step:
    checkpoint_t<step>.pt (ema: optimizer durumundaki ortalama)."""
    assert weights in ("ema", "last"), weights
    config = config or _config(run_dir)
    model = _build(config)
    if step is None:
        path = os.path.join(run_dir, "model_weight_ema.pt" if weights == "ema" else "model.pt")
        assert os.path.exists(path), "%s yok (kosu bitmemis olabilir: --checkpoint last)" % path
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        pack = torch.load(_checkpoints(run_dir)[step], map_location="cpu", weights_only=True)
        assert pack["step"] == step, (pack["step"], step)
        model.load_state_dict(pack["model"])
        if weights == "ema":
            _load_weight_ema(model, pack["optimizer"], config)
    return model.eval().requires_grad_(False)


def over_checkpoints(run_dir, measure, steps=None, weights="ema", log=None):
    """Ayni olcum kosunun yedekleri (checkpoint_tNNNNNN.pt) boyunca.  measure(model) -> sonuc (ornek: lambda m: trace(m,
    hikayeler)); steps: adim listesi (None = hepsi); weights "ema": optimizer durumundaki agirlik ortalamasi (train_seq'in
    model.weight_ema'si), "last": o adimin agirligi.  Yedekler sirayla, tek tek yuklenir.  -> [dict(step, result)]"""
    packs = _checkpoints(run_dir)
    assert packs, "%s: checkpoint_t*.pt yok" % run_dir
    chosen = list(packs) if steps is None else [int(s) for s in steps]
    missing = [s for s in chosen if s not in packs]
    assert not missing, "yedek yok: %s (var: %d..%d)" % (missing, min(packs), max(packs))
    config = _config(run_dir)
    out = []
    for s in chosen:
        t0 = time.time()
        out.append(dict(step=s, result=measure(_load_model(run_dir, weights, s, config))))
        if log:
            log("adim %d bitti (%.0f sn)" % (s, time.time() - t0))
    return out


# ---- veri (CLI): config'teki tag

def _simplestories(config, root):
    """SimpleStories sinav hikayeleri: stories(count, offset) -> (valid hikaye siralari, [[eos] + hikaye + [eos]]).  Secim
    exam_simplestories / alpha_probe ile ayni: sinav kumesinin tohum 0 permutasyonu, [offset, offset + count) dilimi,
    valid sirasiyla sirali."""
    sys.path.insert(0, os.path.join(HERE, "train_simplestories"))
    import data_simplestories as DS
    tag = config["tag"]
    _, vocab = DS._load(os.path.join(root, tag, "tokenizer.json"), DS.TOKENIZERS[tag]["eos"])
    assert len(vocab) == config["vocab"], "sozluk %d, kosu %d" % (len(vocab), config["vocab"])
    eos = vocab.index(DS.EOS_TOKEN)
    valid = np.load(os.path.join(root, tag, "valid.npy"))
    starts, lengths = DS._stories(valid, eos)
    exam = np.load(os.path.join(root, "exam_stories.npy"))
    perm = np.random.default_rng(0).permutation(len(exam))

    def stories(count, offset=0):
        rows = np.sort(exam[perm[offset:offset + count]])
        return rows, [[eos] + valid[starts[r]:starts[r] + lengths[r]].tolist() + [eos] for r in rows]

    return dict(tag=tag, vocab=vocab, eos=eos, stories=stories, encode=lambda s: DS.encode(s, vocab),
                train=os.path.join(root, tag, "train.npy"))


_DATA = {"ss4096": (_simplestories, _SIMPLESTORIES_ROOT), "gpt2": (_simplestories, _SIMPLESTORIES_ROOT)}


def _token_counts(data, config, run_dir, path=None, log=print):
    """Train akisinin token sayimi: path verilirse oradan; yoksa <kosu>/internals/token_counts_<tag>_<iz>.npy (bir kez
    sayilir, sonra okunur -- train.npy buyuk)."""
    if path:
        return np.load(path)
    cache = os.path.join(run_dir, "internals", "token_counts_%s_%s.npy" % (data["tag"], config.get("fingerprint", "x")))
    if os.path.exists(cache):
        return np.load(cache)
    log("train sayimi: %s (bir kez; sonra %s)" % (data["train"], cache))
    a = np.load(data["train"], mmap_mode="r")
    counts = np.zeros(len(data["vocab"]), np.int64)
    for i in range(0, len(a), 50_000_000):
        counts += np.bincount(np.asarray(a[i:i + 50_000_000]), minlength=len(counts))[:len(counts)]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, counts)
    return counts


# ---- CLI: mudahale yazimi

def _parse_case(shape, text):
    """CLI mudahale yazimi -> (ad, aciklama, kur(model) -> mudahale).  Turlar 1'den, head'ler 0'dan; '+' ile birlesir:
        none                 mudahale yok (taban ile ayni cikmali)
        A3 / F3              tur 3'un attention / FactUnits alt adimi yok
        H3 / H3.1            tur 3'un butun head'leri / head 1'i, referans ortalamasiyla (ortalama ablasyonu)
        C3 / C               tur 3'te / butun turlarda Canon kapali
        aA3=0.5 / aF6*0.5    tur 3'un alpha_attention'i 0,5 (butun boyutlar) / tur 6'nin alpha_facts'i x 0,5
    shape: (tur sayisi, head sayisi, Block sayisi)."""
    turns, heads, nb = shape
    parts, words = [], []
    for part in text.split("+"):
        m = re.fullmatch(r"none|([AF])(\d+)|H(\d+)(?:\.(\d+))?|C(\d+)?|a([AF])(\d+)([=*])(-?[\d.]+(?:e-?\d+)?)", part)
        assert m, "mudahale okunamadi: %r (python internals_y.py -h)" % part
        turn = next((int(g) - 1 for g in (m.group(2), m.group(3), m.group(5), m.group(7)) if g), None)
        assert turn is None or 0 <= turn < turns, "%s: tur 1..%d" % (part, turns)
        where = "" if turn is None else "tur %d%s " % (turn + 1, _LETTERS[turn % nb])
        if part == "none":
            words.append("mudahale yok")
        elif m.group(1):
            key = "skip_attention" if m.group(1) == "A" else "skip_facts"
            parts.append((key, turn, None))
            words.append(where + ("attention" if m.group(1) == "A" else "FactUnits") + " yok")
        elif m.group(3):
            hs = [int(m.group(4))] if m.group(4) else list(range(heads))
            assert all(0 <= h < heads for h in hs), "%s: head 0..%d" % (part, heads - 1)
            parts += [("heads", (turn, h), None) for h in hs]
            words.append(where + ("head %d" % hs[0] if m.group(4) else "butun head'ler") + " ortalamayla")
        elif part.startswith("C"):
            ts = [turn] if turn is not None else list(range(turns))
            parts += [("canon", t, None) for t in ts]
            words.append(where + "Canon kapali" if turn is not None else "butun turlarda Canon kapali")
        else:
            key = "alpha_attention" if m.group(6) == "A" else "alpha_facts"
            parts.append((key, turn, (m.group(8), float(m.group(9)))))
            words.append(where + "%s %s %s" % (key, "=" if m.group(8) == "=" else "x", m.group(9)))

    def build(model):
        case = {}
        for key, where, arg in parts:
            if key in ("skip_attention", "skip_facts"):
                case.setdefault(key, []).append(where)
            elif key == "heads":
                case.setdefault(key, {})[where] = None
            elif key == "canon":
                case.setdefault(key, {})[where] = torch.zeros(4)
            else:
                op, val = arg
                learned = getattr(model, key)[where].detach()
                case.setdefault(key, {})[where] = learned * val if op == "*" else torch.full_like(learned, val)
        return case

    return text, ", ".join(words), build


def _standard_cases(turns, heads):
    """Varsayilan ablate listesi: none, her tur A / F / H, her head, her tur C, butun Canon."""
    cases = ["none"] + ["A%d" % t for t in range(1, turns + 1)] + ["F%d" % t for t in range(1, turns + 1)]
    cases += ["H%d" % t for t in range(1, turns + 1)]
    if heads > 1:
        cases += ["H%d.%d" % (t, h) for t in range(1, turns + 1) for h in range(heads)]
    return cases + ["C%d" % t for t in range(1, turns + 1)] + ["C"]


# ---- CLI: metin

def _fmt(v, f="%.4f"):
    return "-" if v is None else f % v


def _ablate_row(r):
    return "%-44s nll %.4f (%+.4f ± %.4f)  acc %.4f (%+.2f ± %.2f puan)  [%s]" % (
        r["case"][:44], r["nll"], r["d_nll"], r["se_nll"], r["acc"], 100 * r["d_acc"], 100 * r["se_acc"],
        "yalniz cikis" if r["start"] is None else "tur %d'den" % (r["start"] + 1))


def _text_trace(res, vocab, positions):
    L = ["## logit lens ve alt adim acilari: %d gecerli hedefte ortalama; elle ileri hesap - model.logits en buyuk fark %.1e"
         % (res["n"], res["check_logits"]),
         "durum | blok | onceki duruma aci | lens nll | lens acc | medyan sira | p(hedef) | cos(h, PL girdi) | cos(h, PL hedef)"]
    for name, b in zip(res["states"], res["blocks"]):
        s = res["summary"][name]
        L.append("%-5s | %s | %6s | %7.3f | %.4f | %7.1f | %.4f | %6.3f | %6.3f" % (
            name, b, _fmt(s["angle"], "%.1f"), s["nll"], s["acc"], s["median_rank"], s["p"], s["cos_input"], s["cos_target"]))
    tok = lambda i: (vocab[i] if vocab is not None else str(i))[:10]
    for k, row in enumerate(res["stories"]):
        ids = row["ids"]
        L += ["", "## hikaye %d (%d token): dogru sonraki token'in sirasi her durumda (1 = en yuksek skor); ilk %d hedef"
              % (k + 1, len(ids), min(positions, len(ids) - 1)),
              "%4s %-10s -> %-10s | %s | p(son) | son tahmin" % ("konum", "girdi", "hedef", " ".join(
                  "%5s" % n for n in res["states"]))]
        last = res["states"][-1]
        for t in range(min(positions, len(ids) - 1)):
            L.append("%4d%s %-10s -> %-10s | %s | %.3f | %s" % (
                t, " " if row["valid"][t] else "*", tok(ids[t]), tok(ids[t + 1]),
                " ".join("%5d" % row["rank"][n][t] for n in res["states"]), row["p"][last][t], tok(row["top"][last][t])))
    L.append("(* = sayilmayan hedef)")
    return L


def _text_point_drift(res, vocab):
    o = res["overall"]
    L = ["## token noktalari: %d token, d %d, ogrenilen %s, capa %.1e (amacta %.4f)" % (
        res["tokens"], res["d"], res["learn"], res["anchor"], res["anchor_loss"]),
         "PF -> PL acisi: medyan %.2f, ortalama %.2f, en buyuk %.2f derece; |shift| medyan %.4f" % (
             o["angle_median"], o["angle_mean"], o["angle_max"], o["shift_median"]),
         "en yakin komsu kosinusu medyan: PL %.3f | PF (rastgele duzey) %.3f" % (o["nn_cos_pl_median"], o["nn_cos_pf_median"])]
    if "bands" in res:
        L += ["", "siklik sirasi | sayim medyan | aci medyan | min | max | |shift| medyan | komsu kos. medyan"]
        L += ["%5d-%-5d | %12.0f | %6.2f | %6.2f | %6.2f | %.4f | %.3f" % (
            b["lo"], b["hi"], b["count_median"], b["angle_median"], b["angle_min"], b["angle_max"], b["shift_median"],
            b["nn_cos_median"]) for b in res["bands"]]
        u = res["unseen"]
        L.append("hic gorulmeyen token: %d, aci medyan %s, en buyuk %s; corr(log(1 + sayim), aci) = %.3f" % (
            u["n"], _fmt(u["angle_median"], "%.2f"), _fmt(u["angle_max"], "%.2f"), res["corr_log_count_angle"]))
    for e in res.get("examples", ()):
        L.append("  %-12s sira %5d sayim %s aci %5.2f -> %s" % (e["token"], e["rank"], _fmt(e["count"], "%.0f"), e["angle"],
                                                               ", ".join("%s %.2f" % nb for nb in e["neighbors"])))
    return L


def _text_attention(res):
    H = res["heads"]
    L = ["## attention (tur x head): %d gecerli sorgu konumunda ortalama" % res["n"],
         "tur   h | entropi | ent/ln(t+1) | son 4 | kendisi | konum 0 | ort. uzaklik | en buyuk | |c_h|"]
    for t, name in enumerate(res["turns"]):
        for h in range(H):
            L.append("%-5s %d | %6.2f | %5.2f | %5.3f | %5.3f | %5.3f | %6.1f | %5.3f | %5.3f" % (
                name, h, *(float(res[k][t][h]) for k in ("entropy", "entropy_norm", "last4", "self", "first", "distance",
                                                         "max", "head_norm"))))
    L += ["", "## tur basina: |Canon eki| / |girdi|; head ortusmesi sum_j min(a1, a2) (1 = ayni desen)"]
    for t, name in enumerate(res["turns"]):
        ov = res["overlap"][t]
        L.append("%-5s canon %s | %s" % (name, _fmt(None if res["canon"] is None else float(res["canon"][t]), "%.3f"),
                                         "  ".join("%d-%d %.2f" % (i, j, ov[i][j]) for i in range(H) for j in range(i + 1, H))))
    for r in res["reuse"]:
        t1, t2 = r["turns"]
        L.append("ayni Block, tur %d ile %d, ayni head ortusmesi: %s" % (t1 + 1, t2 + 1, " ".join("%.2f" % v for v in r["overlap"])))
    return L


def _text_units(res):
    L = ["## FactUnits birimleri (%d birim, %s): %d gecerli konum" % (res["units"], res["activation"], res["n"]),
         "tur | E|u|^2 | |W_out u| rms | enerji %50/%90/%99 birim | olu (<1e-5) | seyrek (<%1) | u=0 orani | |r|>0,9 / >0,7 cift"]
    for r in res["turns"]:
        L.append("%-5s | %8.3f | %7.3f | %4d / %4d / %4d | %4d | %4d | %.3f | %d / %d (%d ornek)" % (
            r["turn"], r["energy"], r["out_rms"], *r["cover"], r["dead"], r["rare"], r["zero"], *r["corr_pairs"], r["sample"]))
    L += ["", "## FactUnits takimlari: agirlik |cos| > 0,8 cift (en buyuk); takimi kullanan turlar arasi"]
    for tm in res["teams"]:
        L.append("turlar %s: %s" % ("+".join(str(t + 1) for t in tm["turns"]), "  ".join(
            "%s %d (%.2f)" % (k, v["pairs"], v["max"]) for k, v in tm["similar"].items())))
        for b in tm["between"]:
            t1, t2 = b["turns"]
            L.append("  tur %d / %d: enerji payi korelasyonu %.3f; tur %d'de 10 kat fazla %d birim, tur %d'de %d; ikisinde de "
                     "olu %d" % (t1 + 1, t2 + 1, b["corr"], t1 + 1, b["more_first"], t2 + 1, b["more_second"], b["dead_both"]))
    return L


def _text_ablate(res):
    b = res["base"]
    L = ["## ablate.  " + res["note"],
         "taban: nll %.4f  acc %.4f  (%d hedef, %d hikaye; elle ileri hesap - model.logits en buyuk fark %.1e)%s" % (
             b["nll"], b["acc"], b["n"], b["stories"], res["check_logits"],
             "" if res["reference"] is None else "; head ortalamalari %s" % res["reference"]),
         "Δ hikaye duzeyinde eslesmis, ± standart hata"]
    return L + [_ablate_row(r) for r in res["rows"]]


def _summary(measure, res):
    """over_checkpoints tablosu icin olcumun ozet sayilari."""
    if measure == "trace":
        return {"%s %s" % (k, n): res["summary"][n][k] for n in res["states"] for k in ("nll", "acc")}
    if measure == "point_drift":
        out = {"aci medyan": res["overall"]["angle_median"], "komsu kos. PL": res["overall"]["nn_cos_pl_median"]}
        out.update({"aci %d-%d" % (b["lo"], b["hi"]): b["angle_median"] for b in res.get("bands", ())})
        return out
    if measure == "attention_stats":
        out = {}
        for t, name in enumerate(res["turns"]):
            out.update({"%s entropi" % name: float(np.mean(res["entropy"][t])), "%s son4" % name: float(np.mean(res["last4"][t])),
                        "%s uzaklik" % name: float(np.mean(res["distance"][t]))})
        return out
    if measure == "unit_usage":
        out = {}
        for r in res["turns"]:
            out.update({"%s %%90 birim" % r["turn"]: r["cover"][1], "%s olu" % r["turn"]: r["dead"]})
        return out
    return {r["case"].split()[0]: r["d_nll"] for r in res["rows"]}


def _jsonable(v):
    if isinstance(v, dict):
        return {(k if isinstance(k, str) else ".".join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(x)
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if torch.is_tensor(v) or isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (float, np.floating)):
        return float("%.6g" % v) if math.isfinite(v) else None
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


_MEASURES = ("trace", "point_drift", "attention_stats", "unit_usage", "ablate")
_DEFAULT_STORIES = dict(trace=4, point_drift=0, attention_stats=128, unit_usage=128, ablate=128)


def _main(argv=None):
    ap = argparse.ArgumentParser(
        prog="internals_y.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Egitilmis Model Y'nin ici (internals_y.py'nin basindaki aciklama).",
        epilog="ablate mudahale yazimi (--cases):" + _parse_case.__doc__.split("'+' ile birlesir:")[1].split("    shape:")[0])
    ap.add_argument("run", help="kosu adi (%s/<ad>) ya da klasoru" % _RUNS_ROOT)
    ap.add_argument("measure", choices=_MEASURES)
    ap.add_argument("--weights", choices=("ema", "last"), default="ema", help="agirlik ortalamasi (varsayilan) ya da son agirlik")
    ap.add_argument("--checkpoint", help="tek yedek: adim ya da 'last' (varsayilan: kosu sonunun agirligi)")
    ap.add_argument("--checkpoints", help="yedekler boyunca: 'all', 'every:K' ya da '5000,10000,...'")
    ap.add_argument("--stories", type=int, help="sinav hikayesi sayisi (varsayilan: trace 4, digerleri 128)")
    ap.add_argument("--offset", type=int, default=0, help="sinav permutasyonunda baslangic (farkli hikayeler)")
    ap.add_argument("--text", action="append", help="trace: istem metni (<eos> ile baslatilir; butun hedefler sayilir)")
    ap.add_argument("--positions", type=int, default=60, help="trace: hikaye basina tablodaki hedef sayisi")
    ap.add_argument("--cases", nargs="+", help="ablate mudahaleleri (varsayilan: none, her tur A/F/H/C, her head, C)")
    ap.add_argument("--reference", type=int, default=48, help="ablate: head ortalamasi icin ayri hikaye sayisi")
    ap.add_argument("--counts", help="point_drift: token sayimi .npy (varsayilan: train akisi, bir kez sayilir)")
    ap.add_argument("--batch", type=int, help="batch (varsayilan 16; attention_stats 8)")
    ap.add_argument("--data", help="veri koku (varsayilan: tag'e gore %s)" % _SIMPLESTORIES_ROOT)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):              # Windows konsolu (cp1254) Δ yazamiyor; dosyalar utf-8
        sys.stdout.reconfigure(errors="replace")
    torch.set_num_threads(4)
    t0 = time.time()
    say = lambda s: print(s, flush=True)
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(_RUNS_ROOT, args.run)
    assert os.path.isdir(run_dir), "kosu klasoru yok: %s" % run_dir
    config = _config(run_dir)
    loader = _DATA.get(config.get("tag"))
    if loader is None:
        raise SystemExit("veri yukleyicisi yok: tag %r (yalniz SimpleStories: ss4096, gpt2); olcum fonksiyonlari id "
                         "listeleriyle dogrudan cagrilabilir" % config.get("tag"))
    data = loader[0](config, args.data or loader[1])
    vocab = data["vocab"]
    count = _DEFAULT_STORIES[args.measure] if args.stories is None else args.stories
    if args.measure == "trace" and args.text and args.stories is None:
        count = 0
    rows, stories = data["stories"](count, args.offset) if count else (np.array([], dtype=np.int64), [])
    batch = args.batch or (8 if args.measure == "attention_stats" else 16)
    skeleton = _build(config)
    shape = (skeleton.turns, skeleton.blocks[0].attention.heads, len(skeleton.blocks))
    lines = []

    if args.measure == "trace":
        prompts = [[data["eos"]] + data["encode"](s) for s in (args.text or ())]
        assert stories or prompts, "trace: --stories > 0 ya da --text gerekli"

        def measure(m):
            out = {}
            if stories:
                out["stories"] = trace(m, stories, batch=batch)
            if prompts:
                out["texts"] = trace(m, prompts, exclude_last=False, batch=batch)
            return out

        def text(res):
            heads = dict(stories="# SINAV HIKAYELERI (son <eos> hedefi sayilmaz)", texts="# ISTEM METINLERI (--text)")
            return sum(([heads[k]] + _text_trace(res[k], vocab, args.positions) + [""] for k in heads if k in res), [])

        summarize = lambda res: _summary("trace", res.get("stories") or res["texts"])
    elif args.measure == "point_drift":
        counts = _token_counts(data, config, run_dir, args.counts, say)
        measure = lambda m: point_drift(m, counts, vocab)
        text = lambda res: _text_point_drift(res, vocab)
        summarize = lambda res: _summary("point_drift", res)
    elif args.measure == "attention_stats":
        measure = lambda m: attention_stats(m, stories, batch=batch)
        text, summarize = _text_attention, lambda res: _summary("attention_stats", res)
    elif args.measure == "unit_usage":
        measure = lambda m: unit_usage(m, stories, batch=batch)
        text, summarize = _text_units, lambda res: _summary("unit_usage", res)
    else:
        cases = [_parse_case(shape, c) for c in (args.cases or _standard_cases(shape[0], shape[1]))]
        needs_ref = any("H" in c[0] for c in cases)
        ref = data["stories"](args.reference, args.offset + count)[1] if needs_ref and args.reference else None
        measure = lambda m: ablate(m, stories, [("%-8s %s" % (name, words), build(m)) for name, words, build in cases],
                                   reference=ref, batch=batch, log=say)
        text, summarize = _text_ablate, lambda res: _summary("ablate", res)
        if ref is not None:
            lines.append("head ortalamalari icin ayri %d hikaye: sinav permutasyonu %d..%d" % (
                len(ref), args.offset + count, args.offset + count + len(ref) - 1))

    packs = _checkpoints(run_dir)
    if args.checkpoints:
        spec = args.checkpoints
        steps = (list(packs) if spec == "all" else [s for s in packs if s % int(spec[6:]) == 0] if spec.startswith("every:")
                 else [int(s) for s in spec.split(",")])
        results = over_checkpoints(run_dir, measure, steps, args.weights, log=say)
        source, tag = "yedekler %s" % ",".join(str(s) for s in steps), "checkpoints"
        table = [summarize(r["result"]) for r in results]
        lines += ["## %d yedek boyunca ozet (%s)" % (len(results), args.weights),
                  "%-28s %s" % ("", " ".join("%10d" % r["step"] for r in results))]
        lines += ["%-28s %s" % (k[:28], " ".join("%10.4g" % row.get(k, float("nan")) for row in table)) for k in table[0]]
        for r in results:
            lines += ["", "=" * 100, "ADIM %d" % r["step"]] + text(r["result"])
        result = results
    else:
        step = None if args.checkpoint is None else (max(packs) if args.checkpoint == "last" else int(args.checkpoint))
        model = _load_model(run_dir, args.weights, step, config)
        source = ("checkpoint_t%06d.pt" % step) if step is not None else (
            "model_weight_ema.pt" if args.weights == "ema" else "model.pt")
        tag = "" if step is None else "t%06d" % step
        result = measure(model)
        lines += text(result)
    header = ["internals_y %s | kosu %s | agirlik %s (%s) | %s | sure %.0f sn" % (
        args.measure, config.get("name", os.path.basename(run_dir)), args.weights, source, time.strftime("%Y-%m-%d %H:%M"),
        time.time() - t0)]
    if count:
        header.append("sinav hikayeleri: %d (tohum 0 permutasyonu %d..%d; valid sirasi %d..%d)" % (
            count, args.offset, args.offset + count - 1, int(rows.min()), int(rows.max())))
    out_dir = os.path.join(run_dir, "internals")
    os.makedirs(out_dir, exist_ok=True)
    name = "_".join(p for p in (args.measure, args.weights, tag, time.strftime("%Y%m%d_%H%M%S")) if p)
    body = "\n".join(header + [""] + lines) + "\n"
    with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(body)
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(measure=args.measure, run=config.get("name"), weights=args.weights, source=source,
                                 stories=rows.tolist(), offset=args.offset, texts=args.text, result=result)),
                  f, ensure_ascii=False)
    print(body)
    say("yazildi: %s.txt / .json" % os.path.join(out_dir, name))


if __name__ == "__main__":
    _main()
