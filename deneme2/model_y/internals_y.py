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
    tuned_lens        her durumda ogrenilen afin cevirici + modelin cikis basligi (Belrose 2023): tahmin nerede olusuyor;
                      logit lens ile yan yana (nll, acc, son dagilima KL, sira).  Cevirici ayri hikayelerde ogrenilir
    over_checkpoints  ayni olcum kosunun checkpoint_tNNNNNN.pt yedekleri boyunca (EMA optimizer durumundan kurulur)

Hikayeler: id listeleri.  Girdi ids[:-1], hedef ids[1:]; exclude_last: son hedef (SimpleStories'te hikayeyi kapatan <eos>)
sayilmaz.  Turlar kodda 0'dan, metinde 1'den (tur 3 = indeks 2); head'ler 0'dan.
Veri (CLI): SimpleStories (config'teki tag) ya da FineWeb-Edu (kosu dataset'i fineweb-edu ya da --data FineWeb klasoru):
valid belgeleri TEK TEK, en cok 2.048 token.  Paketli batch (document_positions) desteklenmez.

    python internals_y.py <kosu adi | klasoru> <olcum> [secenekler]        (python internals_y.py -h)
Cikti <kosu klasoru>/internals/<olcum>_<agirlik>[_tNNNNNN]_<zaman>.txt + .json.  Yerel CPU, 4 is parcacigi.
"""
import argparse
import contextlib
import inspect
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model_y import BlockModel, deviation  # noqa: E402

_RUNS_ROOT = "G:/Drive'ım/model_y"                  # kosu klasorleri (Drive, bu bilgisayarda)
_SIMPLESTORIES_ROOT = "G:/Drive'ım/simplestories"    # <tag>/tokenizer.json, <tag>/valid.npy, exam_stories.npy
_FINEWEB_ROOT = "G:/Drive'ım/fineweb"                # gpt2/tokenizer.json, gpt2/shard_NNN.* (data_fineweb)
_FINEWEB_TOKENS = 2048                               # FineWeb belgesi en cok bu kadar girdiyle (tek belge, paketsiz)
# train_y.train_seq'teki Muon listesi: yedekteki optimizer parametre sirasi buna bagli (once bunlar)
_MUON_HIDDEN = ("W_context", "W_value", "W_fact_in", "W_fact_up", "W_fact_out")
_PLAN_KEYS = ("skip_attention", "skip_facts", "heads", "canon", "canon_mean", "alpha_attention", "alpha_facts")
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
    """Tur t'nin FactUnits'i: SHARED_FACTS=False'ta tur >= layers kendi takimini (extra_facts) kullanir; FIRST_TURN_FACTS=False'ta
    tur 0'da None (alt adim yok)."""
    if t == 0 and not getattr(model, "first_turn_facts", True):
        return None
    if not model.shared_facts and t >= model.layers:
        return model.extra_facts[t - model.layers]
    return model.turn_blocks()[t].facts


def _turn_label(model, t):
    """'3C': tur 3, blok C.  Ayri FactUnits takimi varsa not dusulur."""
    own = "" if (model.shared_facts or t < model.layers) else ", ayri FactUnits"
    return "%d%s%s" % (t + 1, _LETTERS[t % len(model.blocks)], own)


def _units(f, v):
    """FactUnits birim etkinlikleri u (.., units) = SiLU(W_fact_in v) * (W_fact_up v); cikti u @ W_fact_out.T
    (FactUnits.forward ile ayni)."""
    return F.silu(v @ f.W_fact_in.T) * (v @ f.W_fact_up.T)


def _plan(model, case=None):
    """Mudahale sozlugu -> tam plan.  Anahtarlar (tur 0'dan):
        skip_attention, skip_facts   tur listesi: alt blogun katkisi sifir (alpha 0 ile ayni)
        heads      {(tur, head): (dh,) vektor | None}: head'in ciktisi c_h bu vektorle degisir; None = ortalama (ablate doldurur)
        canon      {tur: (4,) carpan}: Canon agirliklari w_k bu carpanla (0 = kapali)
        canon_mean {tur: (d,) vektor | None}: Canon eki (sum_k w_k x_(t-k)) bu vektorle degisir; None = ortalama
                   (ablate doldurur)
        alpha_attention, alpha_facts   {tur: sayi | (d,) vektor}: o turun alpha'si ELLE
    plan["start"]: mudahalenin ilk turu (onceki turlar temiz hesapla ayni)."""
    case = dict(case or {})
    bad = set(case) - set(_PLAN_KEYS)
    assert not bad, "bilinmeyen mudahale %s (gecerli: %s)" % (sorted(bad), _PLAN_KEYS)
    plan = dict(skip_attention=set(case.get("skip_attention", ())), skip_facts=set(case.get("skip_facts", ())),
                heads=dict(case.get("heads") or {}), canon=dict(case.get("canon") or {}),
                canon_mean=dict(case.get("canon_mean") or {}))
    touched = list(plan["skip_attention"] | plan["skip_facts"]) + [t for t, _ in plan["heads"]] + list(plan["canon"])
    touched += list(plan["canon_mean"])
    for name in ("alpha_attention", "alpha_facts"):
        given = case.get(name) or {}
        plan[name] = None
        if given:
            a = getattr(model, name).detach().clone()
            for t, v in given.items():
                a[t] = torch.as_tensor(v, dtype=a.dtype, device=a.device)
            plan[name] = a
            touched += list(given)
    H = model.blocks[0].attention.heads
    assert all(0 <= t < model.turns for t in touched), "tur 0..%d disinda" % (model.turns - 1)
    assert getattr(model, "first_turn_facts", True) or (0 not in plan["skip_facts"] and 0 not in (case.get("alpha_facts") or {})), \
        "tur 1'de FactUnits yok (FIRST_TURN_FACTS=False): mudahale sessizce 0 verirdi"
    assert all(0 <= h < H for _, h in plan["heads"]), "head 0..%d disinda" % (H - 1)
    plan["start"] = min(touched, default=model.turns)
    return plan


def _run(model, h, plan, taps=None, start=0):
    """Turlar start..TURNS-1; h tur start'in girdisi (B, T, d) -> son durum.  Block.forward'in aynisi (LAYERS Block,
    SHARED_FACTS, Canon, head'ler, normalized update) + mudahale (plan) + kayit:
    taps[ad](tur, deger), ad: input, canon (ek, girdi), attention (B, H, T, T), heads (B, H, T, dh), attention_out, units,
    facts_out.  attention kaydi istenirse softmax acik hesaplanir (CausalAttention.weights), degilse SDPA."""
    taps = taps or {}
    blocks = model.turn_blocks()
    for t in range(start, model.turns):
        blk = blocks[t]
        at = blk.attention
        if "input" in taps:
            taps["input"](t, h)
        w = blk.canon_weights                          # x_t = h_t + sum_k w_k h_(t-k), baslangictan once 0
        if t in plan["canon"]:
            w = w * torch.as_tensor(plan["canon"][t], dtype=w.dtype, device=w.device)[:, None]
        T = h.shape[-2]
        full = F.pad(h, (0, 0, 3, 0))
        mix = sum(w[k] * full[..., 3 - k:3 - k + T, :] for k in range(4))
        if t in plan["canon_mean"]:
            vec = plan["canon_mean"][t]
            assert vec is not None, "Canon eki ortalamasi doldurulmadi (ablate doldurur)"
            mix = torch.as_tensor(vec, dtype=h.dtype, device=h.device).expand_as(h)
        if "canon" in taps:
            taps["canon"](t, (mix, h))
        x = h + mix
        if t not in plan["skip_attention"]:
            H = at.heads
            q, k = at.queries_keys(x)
            v = (x @ at.W_value.T).unflatten(-1, (H, -1)).transpose(-3, -2)
            if "attention" in taps:
                s = at.scale * q @ k.transpose(-1, -2)
                s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
                a = torch.softmax(s, -1)
                taps["attention"](t, a)
                c = a @ v
            else:
                c = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scale)
            for (tt, hh), vec in plan["heads"].items():
                if tt == t:
                    assert vec is not None, "head ortalamasi doldurulmadi (ablate doldurur)"
                    c[:, hh] = torch.as_tensor(vec, dtype=c.dtype, device=c.device)
            if "heads" in taps:
                taps["heads"](t, c)
            added = c.transpose(-3, -2).flatten(-2) @ at.W_context.T   # h = norm(h + α_A ⊙ (norm(W_context c) - h))
            a_att = model.alpha_attention[t] if plan["alpha_attention"] is None else plan["alpha_attention"][t]
            h = _unit(h + a_att * (_unit(added) - h))
        if "attention_out" in taps:
            taps["attention_out"](t, h)
        f = _turn_facts(model, t)
        if t not in plan["skip_facts"] and f is not None:
            u = _units(f, h * h.shape[-1] ** 0.5)     # kure agirliklari: girdi sqrt(d) x kosinus
            if "units" in taps:
                taps["units"](t, u)
            out = u @ f.W_fact_out.T                   # h = norm(h + α_F ⊙ (norm(olgu) - h))
            a_f = model.alpha_facts[t] if plan["alpha_facts"] is None else plan["alpha_facts"][t]
            h = _unit(h + a_f * (_unit(out) - h))
        if "facts_out" in taps:
            taps["facts_out"](t, h)
    return h


def _scores(model, h, P=None):
    """Cikis basligi (BlockModel.logits ile ayni): e^tau phi(<h, PL>); ara durumlara uygulaninca logit lens."""
    P = model.tokens.points() if P is None else P
    scale = model.scale * torch.exp(model.log_output_scale - math.log(model.scale))
    if model.output_link:
        c = h @ P.T
        q, u = model.link_q, model.link_u
        return scale * (c * (1 + c * (q + c * (q * q / 3 + u))))
    return scale * h @ P.T


@torch.no_grad()
def _logits(model, ids, case=None, taps=None):
    """Elle ileri hesabin skoru (mudahaleli olabilir); mudahalesiz = model.logits (test)."""
    P = model.tokens.points()
    return _scores(model, _run(model, model.input_states(ids), _plan(model, case), taps), P)


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
    zero = torch.zeros((), dtype=z.dtype, device=z.device)
    return (torch.where(valid, nll, zero).sum(-1).double().cpu().numpy(), (hit & valid).sum(-1).double().cpu().numpy(),
            valid.sum(-1).double().cpu().numpy())


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

def _state_names(model):
    """Durum adlari: h0, sonra her tur A<t> (attention sonrasi), F<t> (FactUnits sonrasi); bloklari."""
    names = ["h0"] + ["%s%d" % (k, t + 1) for t in range(model.turns) for k in ("A", "F")]
    return names, ["-"] + [_LETTERS[t % len(model.blocks)] for t in range(model.turns) for _ in "AF"]


def _states(model, inp, plan, P):
    """_state_names sirasiyla durumlar, her biri (B, T, d)."""
    states = [model.input_states(inp)]
    keep = lambda t, h: states.append(h)
    _run(model, states[0], plan, dict(attention_out=keep, facts_out=keep))
    return states


@torch.no_grad()
def trace(model, ids, exclude_last=True, batch=16, keep_logits=False):
    """Token basina noktanin yolu.  ids: id listesi, id listeleri ya da tensor.
    Durumlar: h0 = PL[token]; her tur t: A<t> attention alt adimindan, F<t> FactUnits'ten sonra (= tur ciktisi).  Her
    durumda: bir onceki duruma aci (derece), yon kosinusu <h, PL(girdi)> ve <h, PL(hedef)>, logit lens -- ayni cikis
    basligiyla dogru sonraki token'in sirasi (1 = en yuksek skor), olasiligi ve en yuksek skorlu token.
    -> dict(states, blocks, summary {durum: gecerli hedeflerde ortalama}, stories [{ids, valid, rank, p, top, angle}],
    n, check_logits: son durumun skoru ile model.logits arasi en buyuk fark; keep_logits: logits [(L - 1, V)])."""
    stories = _as_stories(ids)
    names, blocks = _state_names(model)
    P = model.tokens.points()
    sums = {n: dict(angle=0.0, nll=0.0, hit=0.0, p=0.0, cos_input=0.0, cos_target=0.0) for n in names}
    ranks = {n: [] for n in names}
    rows, logits = [None] * len(stories), [None] * len(stories)
    count, check = 0, 0.0
    plan = _plan(model)
    for x, valid, idx in _batches(stories, batch, exclude_last):
        x, valid = x.to(P.device), valid.to(P.device)             # modelin cihazinda (GPU'da da)
        inp, y = x[:, :-1], x[:, 1:]
        states = _states(model, inp, plan, P)
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
                row["p"][name] = [round(v, 6) for v in lp[r, :L].exp().tolist()]   # tek senkron (GPU)
                row["top"][name] = top[r, :L].tolist()
                row["angle"][name] = [round(v, 3) for v in angle[r, :L].tolist()]
                if keep_logits and j == len(names) - 1:
                    logits[i] = z[r, :L].cpu().clone()
    summary = {}
    for n in names:
        a = sums[n]
        summary[n] = dict(angle=a["angle"] / count if n != "h0" else None, nll=a["nll"] / count, acc=a["hit"] / count,
                          median_rank=float(np.median(ranks[n])), p=a["p"] / count, cos_input=a["cos_input"] / count,
                          cos_target=a["cos_target"] / count)
    out = dict(states=names, blocks=blocks, summary=summary, stories=rows, n=count, check_logits=check)
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
    P, PF = tok.points().cpu(), tok.fixed_points.cpu()          # model GPU'da olsa da: numpy ciktilari
    V = P.shape[0]
    dev = deviation(model).double().cpu()
    shift = tok.shift.norm(dim=-1).double().cpu()
    nn_pl, nn_idx = _nearest(P)
    nn_pf, _ = _nearest(PF)
    med = lambda v: float(np.median(v.double().numpy()))              # cift sayida ortadaki ikisinin ortalamasi
    out = dict(tokens=V, d=P.shape[1], anchor=float(tok.anchor or 0.0),
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
                c, j = (P[i] @ P.T).index_fill_(0, torch.tensor([i], device=P.device), -2).topk(3)
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
    dv = model.tokens.fixed_points.device                     # sayaclar modelin cihazinda
    acc = {k: torch.zeros(TN, H, dtype=torch.float64, device=dv) for k in keys}
    overlap = torch.zeros(TN, H, H, dtype=torch.float64, device=dv)
    canon = torch.zeros(TN, dtype=torch.float64, device=dv)
    nb = len(model.blocks)
    pairs = [(t, t2) for t in range(TN) for t2 in range(t + 1, TN) if t % nb == t2 % nb]
    reuse = {p: torch.zeros(H, dtype=torch.float64, device=dv) for p in pairs}
    reused = {t for p in pairs for t in p}
    plan = _plan(model)
    P = model.tokens.points()
    n = n_norm = 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        x, valid = x.to(dv), valid.to(dv)
        inp = x[:, :-1]
        T = inp.shape[1]
        pos = torch.arange(T, dtype=torch.float64, device=dv)
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

        _run(model, model.input_states(inp), plan, dict(attention=on_attention, heads=on_heads, canon=on_canon))
        for t1, t2 in pairs:
            reuse[(t1, t2)] += (torch.minimum(kept[t1], kept[t2]).sum(-1).double() * wq).sum((0, 2))
        n += int(valid.sum())
        n_norm += int(wn.sum())
    res = {k: (v / (n_norm if k == "entropy_norm" else n)).cpu().numpy() for k, v in acc.items()}
    overlap = overlap.cpu() / n
    overlap = overlap + overlap.transpose(1, 2) + torch.eye(H, dtype=torch.float64)
    return dict(turns=[_turn_label(model, t) for t in range(TN)], heads=H, n=n, **res,
                canon=(canon / n).cpu().numpy() if model.canon else None, overlap=overlap.numpy(),
                reuse=[dict(turns=(t1, t2), overlap=(reuse[(t1, t2)] / n).cpu().numpy()) for t1, t2 in pairs])


@torch.no_grad()
def unit_usage(model, ids, exclude_last=True, batch=16, sample=4096, seed=0):
    """Tur basina FactUnits birimleri, gecerli konumlarda:
      energy        E|u|^2 token basina;  out_rms: |W_fact_out u|'nun karesel ortalamasi (norm oncesi yazim boyu)
      share         birim basina enerji payi E u_i^2 / sum_k E u_k^2;  cover: payin %50 / %90 / %99'unu tasiyan en az birim
      dead          payi < 1e-5 olan birim;  rare: |u_i| > 0,1 oldugu konum orani %1'in altinda kalan birim
      zero          u_i = 0 oldugu konum orani, ortalama
      corr_pairs    ornek konumlarda (en fazla sample; tohum seed) aktivasyon korelasyonu |r| > 0,9 / > 0,7 birim cifti
    FactUnits takimi basina (paylasilan blokta bir takim birden cok turda): agirlik benzerligi |cos| > 0,8 olan cift
    (W_fact_in / W_fact_up satirlari, W_fact_out sutunlari), takimi kullanan turlar arasi enerji payi korelasyonu,
    birinde 10 kat fazla etkin birim sayisi."""
    stories = _as_stories(ids)
    TN = model.turns
    facts = [_turn_facts(model, t) for t in range(TN)]
    with_facts = [t for t in range(TN) if facts[t] is not None]   # FIRST_TURN_FACTS=False'ta tur 0 yok
    f0 = facts[with_facts[0]]
    U = f0.W_fact_out.shape[1]
    dv = f0.W_fact_out.device                                # sayaclar modelin cihazinda
    e2 = torch.zeros(TN, U, dtype=torch.float64, device=dv)
    active = torch.zeros(TN, U, dtype=torch.float64, device=dv)
    zeros = torch.zeros(TN, dtype=torch.float64, device=dv)
    out2 = torch.zeros(TN, dtype=torch.float64, device=dv)
    total = sum(len(s) - 1 - int(exclude_last) for s in stories)
    pick = torch.sort(torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:sample]).values
    samples = [[] for _ in range(TN)]
    plan = _plan(model)
    P = model.tokens.points()
    offset = n = 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        x, valid = x.to(dv), valid.to(dv)
        M = int(valid.sum())
        here = (pick[(pick >= offset) & (pick < offset + M)] - offset).to(dv)

        def on_units(t, u):
            uu = u[valid].double()                               # (M, U)
            e2[t] += (uu ** 2).sum(0)
            active[t] += (uu.abs() > 0.1).double().sum(0)
            zeros[t] += (uu == 0).double().mean(1).sum()
            out2[t] += ((uu @ facts[t].W_fact_out.T.double()) ** 2).sum()
            samples[t].append(uu[here].float())

        _run(model, model.input_states(x[:, :-1]), plan, dict(units=on_units))
        offset += M
        n += M
    turns, by_turn = [], {}
    for t in with_facts:
        share = e2[t] / e2[t].sum()
        cum = torch.cumsum(torch.sort(share, descending=True).values, 0)
        S = torch.cat(samples[t]).double()
        S = (S - S.mean(0)) / S.std(0).clamp_min(1e-12)
        C = (S.T @ S / max(len(S) - 1, 1)).abs().triu(1)
        turns.append(dict(turn=_turn_label(model, t), energy=float(e2[t].sum() / n), out_rms=float((out2[t] / n) ** 0.5),
                          cover=[int((cum < q).sum()) + 1 for q in (0.5, 0.9, 0.99)], dead=int((share < 1e-5).sum()),
                          rare=int((active[t] / n < 0.01).sum()), zero=float(zeros[t] / n),
                          corr_pairs=[int((C > 0.9).sum()), int((C > 0.7).sum())], sample=len(S), share=share.cpu().numpy()))
        by_turn[t] = turns[-1]
    teams = []
    for f in dict.fromkeys(f for f in facts if f is not None):    # sirali, tekrarsiz
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
                a, b = by_turn[t1]["share"], by_turn[t2]["share"]
                r = np.log10((a + 1e-9) / (b + 1e-9))
                between.append(dict(turns=(t1, t2), corr=float(np.corrcoef(a, b)[0, 1]), more_first=int((r > 1).sum()),
                                    more_second=int((r < -1).sum()), dead_both=int(((a < 1e-5) & (b < 1e-5)).sum())))
        teams.append(dict(turns=used, similar=sim, between=between))
    return dict(units=U, n=n, turns=turns, teams=teams)


def _head_means(model, stories, exclude_last, batch):
    """Head ciktilarinin (c_h, W_context'ten once) gecerli konumlarda ortalamasi: {(tur, head): (dh,)}."""
    P = model.tokens.points()
    sums, n = {}, 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        x, valid = x.to(P.device), valid.to(P.device)

        def on_heads(t, c):
            s = c.permute(0, 2, 1, 3)[valid].double().sum(0)     # (H, dh)
            sums[t] = s if t not in sums else sums[t] + s
        _run(model, model.input_states(x[:, :-1]), _plan(model), dict(heads=on_heads))
        n += int(valid.sum())
    dtype = P.dtype
    return {(t, h): (s[h] / n).to(dtype) for t, s in sums.items() for h in range(s.shape[0])}


def _canon_means(model, stories, exclude_last, batch):
    """Canon ekinin (sum_k w_k x_(t-k), attention girdisine eklenen) gecerli konumlarda ortalamasi: {tur: (d,)}."""
    P = model.tokens.points()
    sums, n = {}, 0
    for x, valid, idx in _batches(stories, batch, exclude_last):
        x, valid = x.to(P.device), valid.to(P.device)

        def on_canon(t, mx):
            s = mx[0][valid].double().sum(0)                     # (d,)
            sums[t] = s if t not in sums else sums[t] + s
        _run(model, model.input_states(x[:, :-1]), _plan(model), dict(canon=on_canon))
        n += int(valid.sum())
    return {t: (s / n).to(P.dtype) for t, s in sums.items()}


@torch.no_grad()
def ablate(model, ids_or_stories, cases, reference=None, exclude_last=True, batch=16, log=None):
    """Parca kapatma / alpha degistirme.  cases: [(ad, mudahale)] ya da {ad: mudahale}; mudahale _plan anahtarlariyla
    (skip_attention, skip_facts, heads, canon, canon_mean, alpha_attention, alpha_facts; tur 0'dan).  heads / canon_mean'de
    None: head'in ciktisi / Canon eki reference hikayelerindeki ortalamasiyla (verilmezse olculen hikayelerdeki) degisir:
    ortalama ablasyonu.
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
        x, valid = x.to(P.device), valid.to(P.device)
        hs = []
        h = _run(model, model.input_states(x[:, :-1]), clean, dict(input=lambda t, v: hs.append(v)))
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
    means, cmeans, rows = None, None, []
    for name, case in cases:
        plan = _plan(model, case)
        if any(v is None for v in plan["heads"].values()):
            if means is None:
                means = _head_means(model, ref, exclude_last, batch)
            plan["heads"] = {k: means[k] if v is None else v for k, v in plan["heads"].items()}
        if any(v is None for v in plan["canon_mean"].values()):
            if cmeans is None:
                cmeans = _canon_means(model, ref, exclude_last, batch)
            plan["canon_mean"] = {k: cmeans[k] if v is None else v for k, v in plan["canon_mean"].items()}
        r = collect(plan)
        rows.append(dict(case=name, nll=r["nll"], acc=r["acc"], start=plan["start"] if plan["start"] < model.turns else None,
                         **_paired(base, r)))
        if log:
            log(_ablate_row(rows[-1]))
    return dict(base=dict(nll=base["nll"], acc=base["acc"], n=int(base["count"].sum()), stories=len(stories)),
                rows=rows, reference=None if means is None and cmeans is None else (
                    "olculen hikayelerden" if reference is None else "ayri %d hikayeden" % len(ref)),
                note=_DEPENDENCE_NOTE, check_logits=check)


def _lens(model, h, P, W=None, b=None):
    """Lens skoru: cevirici h + W h + b (W, b yoksa birim), sonra modelin cikis basligi.  Girdi kureye indirilir (son
    durum da kurede; tuned lens'teki son LayerNorm'un karsiligi)."""
    if W is not None:
        h = h + h @ W.T + b
    return _scores(model, _unit(h), P)


@contextlib.contextmanager
def _frozen(model):
    """Parametreler gecici olarak requires_grad=False: cevirici egitimi modele gradyan yazmasin."""
    flags = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(model.parameters(), flags):
            p.requires_grad_(f)


def tuned_lens(model, ids, fit_ids, steps=200, lr=1e-3, batch=8, exclude_last=True, seed=0, fit_source=None, log=None):
    """Tuned lens (Belrose 2023): her durum (trace'teki h0, A<t>, F<t>) icin afin cevirici h + W h + b, W ve b 0'dan
    (birimden baslar); ciktisi modelin kendi cikis basligindan (_lens).  Amac son dagilima KL(p_son || q), p_son =
    _lens(son durum) (= model.logits, yuvarlama icinde).  Model donuk.
    Ogrenme fit_ids hikayelerinde (Adam, lr dogrusal olarak 0'a iner, adim basina batch hikaye, epok sirasi tohum seed);
    degerlendirme ids'de (ayri hikayeler).  steps=0: birim cevirici (= logit lens).
    Her durumda logit lens ve tuned lens yan yana: nll, acc, kl (nat), medyan sira (dogru token'in), agree (en yuksek
    skorlu token son durumunkiyle ayni).  Son durumda birim zaten en iyi (KL 0): ogrenilince 0 kalmali (saglik).
    -> dict(states, blocks, summary {durum: {logit, tuned}}, fit {source, stories, tokens, steps, batch, lr, kl_first,
    kl_last: durum basina ilk / son epok ortalamasi (egitim batch'lerinde)}, n, example {ids, valid, top: durum -> tuned
    lens'in en yuksek skorlu token'lari; ilk hikaye}, translators {W (S, d, d), b (S, d)})."""
    assert steps == 0 or fit_ids is not None, "ogrenme icin fit_ids (degerlendirmeden ayri hikayeler) gerekli"
    stories, fit = _as_stories(ids), _as_stories(fit_ids) if steps else []
    names, blocks = _state_names(model)
    S, plan = len(names), _plan(model)
    with torch.no_grad():
        P = model.tokens.points()
    dev = P.device
    W = torch.zeros(S, P.shape[1], P.shape[1], dtype=P.dtype, device=dev, requires_grad=True)
    b = torch.zeros(S, P.shape[1], dtype=P.dtype, device=dev, requires_grad=True)
    fit_batches = [(x.to(dev), v.to(dev), i) for x, v, i in _batches(fit, batch, exclude_last)]
    epoch = min(len(fit_batches), max(steps, 1))
    opt = torch.optim.Adam([W, b], lr=lr)
    rng, order, curve = np.random.default_rng(seed), [], []

    def teacher(states, valid):
        hs = [s[valid] for s in states]
        logp = torch.log_softmax(_work(_lens(model, hs[-1], P)), -1)
        return hs, logp, logp.exp()

    with _frozen(model):
        for step in range(steps):
            if not order:
                order = list(rng.permutation(len(fit_batches)))
            x, valid, _ = fit_batches[order.pop()]
            with torch.no_grad():
                hs, logp, p = teacher(_states(model, x[:, :-1], plan, P), valid)
                ent = float(-(p * logp).sum(-1).mean())
            kl = []
            for s in range(S):
                loss = -(p * torch.log_softmax(_work(_lens(model, hs[s], P, W[s], b[s])), -1)).sum(-1).mean()
                loss.backward()
                kl.append(float(loss.detach()) - ent)
            for g in opt.param_groups:
                g["lr"] = lr * (1 - step / steps)
            opt.step()
            opt.zero_grad()
            curve.append(kl)
            if log and (step + 1) % max(steps // 5, 1) == 0:
                log("tuned lens adim %d/%d: egitim KL (durumlar ortalamasi) %.4f" % (step + 1, steps, float(np.mean(kl))))

        kinds = ("logit", "tuned")
        sums = {n: {k: dict(nll=0.0, hit=0.0, kl=0.0, agree=0.0) for k in kinds} for n in names}
        ranks = {n: {k: [] for k in kinds} for n in names}
        example, count = None, 0
        with torch.no_grad():
            for x, valid, idx in _batches(stories, batch, exclude_last):
                x, valid = x.to(dev), valid.to(dev)
                states = _states(model, x[:, :-1], plan, P)
                hs, logp, p = teacher(states, valid)
                y = x[:, 1:][valid]
                final = logp.argmax(-1)
                count += len(y)
                for s, name in enumerate(names):
                    for kind, z in (("logit", _lens(model, hs[s], P)), ("tuned", _lens(model, hs[s], P, W[s], b[s]))):
                        lq = torch.log_softmax(_work(z), -1)
                        top = z.argmax(-1)
                        acc = sums[name][kind]
                        acc["nll"] -= float(lq.gather(-1, y[:, None]).sum())
                        acc["hit"] += float((top == y).sum())
                        acc["kl"] += float((p * (logp - lq)).sum())
                        acc["agree"] += float((top == final).sum())
                        ranks[name][kind] += (1 + (z > z.gather(-1, y[:, None])).sum(-1)).tolist()
                if 0 in idx:                                     # ilk hikaye: tuned lens'in tahmini, durum durum
                    r, L = idx.index(0), len(stories[0]) - 1
                    example = dict(ids=stories[0], valid=valid[r, :L].tolist(),
                                   top={n: _lens(model, states[s][r, :L], P, W[s], b[s]).argmax(-1).tolist()
                                        for s, n in enumerate(names)})
    first, last = np.mean(curve[:epoch], 0) if curve else None, np.mean(curve[-epoch:], 0) if curve else None
    summary = {n: {k: dict(nll=v["nll"] / count, acc=v["hit"] / count, kl=v["kl"] / count, agree=v["agree"] / count,
                           median_rank=float(np.median(ranks[n][k]))) for k, v in sums[n].items()} for n in names}
    return dict(states=names, blocks=blocks, summary=summary, n=count, stories=len(stories), example=example,
                fit=dict(source=fit_source, stories=len(fit), tokens=sum(int(v.sum()) for _, v, _ in fit_batches),
                         steps=steps, batch=batch, lr=lr, epoch_steps=epoch,
                         kl_first=None if first is None else first.tolist(), kl_last=None if last is None else last.tolist()),
                translators=dict(W=W.detach(), b=b.detach()))


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


def _model_kw(config):
    """config'teki model_kw: eski config'lerde yazilmamis anahtarlar o gunun degeriyle; kaldirilan secenegin anahtari
    tek kalan davranisin degerini tasiyorsa atilir, baska degerdeyse kosu kurulamaz (sessizce farkli model kurulmasin)."""
    kw = dict(dict(output_link=False, shared_facts=True, input_embedding=False, first_turn_facts=True,
                   input_embedding_sphere=False, rope_base=10000.0, attention_log_scale=False), **config.get("model_kw", {}))
    for key, only in (("packed_attention", "flex"), ("fact_activation", "swiglu"), ("learn_output_scale", True),
                      ("input_bigrams", 0)):
        value = kw.pop(key, only)
        assert value == only, "%s=%r kaldirildi (yalniz %r): bu kosu bugunku kodla kurulamaz" % (key, value, only)
    return kw


def _build(config):
    """config.json -> bos BlockModel (agirliklar sonra yuklenir)."""
    setting = config.get("setting", "shared")
    assert setting == "shared", "yalniz BlockModel (setting shared), bu kosu: %s" % setting
    assert (config["stream_norm"] and not config["layer_norm"] and config["normalized_update"] and config["sphere_weights"]
            and config["canon"]), "normsuz akis / LayerNorm / normalized_update, sphere_weights, canon kapali kaldirildi: bu kosu " \
        "kurulamaz"
    return BlockModel(config["vocab"], seed=config.get("seed", 0), rope=config["rope"], **_model_kw(config))


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


# ---- egitim adiminin kurulumu (analysis/analyze_speed)

def _train_kwargs(config):
    """config -> train_seq ayarlari: yalniz train_seq'in parametreleri (adim, kayit, cihaz, compile ve veri haric)."""
    import train_y as TR
    skip = {"setting", "ids", "mask", "n", "steps", "log_at", "every", "callback", "save_every", "save", "checkpoint",
            "batches", "device", "compile"}
    kw = {k: config[k] for k in set(inspect.signature(TR.train_seq).parameters) - skip if k in config}
    kw["model_kw"] = _model_kw(config)
    return kw


def _profile_setup(config, cached, device):
    """train_seq'in kendi kurulumu, 1 adim: model, optimizer (Muon / Adam gruplari, kure eksenleri), EMA onun kodundan.
    -> adimi parca parca tekrarlayanin ihtiyaci (ctx; analysis/analyze_speed)."""
    import train_y as TR
    kw = _train_kwargs(config)
    default = inspect.signature(TR.train_seq).parameters
    get = lambda k: kw[k] if k in kw else default[k].default
    got = {}
    TR.train_seq(config.get("setting", "shared"), None, None, config["vocab"], steps=1, device=device, compile=False,
                 log_at=(), save_every=1, save=lambda s, m, o: got.update(model=m, opt=o),
                 batches=lambda s: cached[s % len(cached)], **kw)
    model = got["model"]
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    rows, cols = ("W_query", "W_key", "W_fact_in", "W_fact_up", "W_value"), ("W_context", "W_fact_out")
    cuda = torch.device(device).type == "cuda"
    compile = cuda and config.get("compile", True)       # train_seq gibi: compile yalniz GPU'da
    return dict(model=model, opt=got["opt"], named=named, params=[p for _, p in named], device=device, cuda=cuda,
                unit_axis={k: int(k.split(".")[-1] in rows) for k, _ in named if k.split(".")[-1] in rows + cols},
                split=get("schedule") == "coherence", grad_clip=get("grad_clip"), ema=getattr(model, "weight_ema", None),
                compile=compile, loss_fn=torch.compile(model.loss) if compile else model.loss,
                bf16=cuda and get("matmul_precision") == "bf16", kernels=[SDPBackend.MATH])      # train_seq gibi


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


def _fineweb(config, root):
    """FineWeb-Edu valid belgeleri (data_fineweb.load_valid; egitim akisi okunmaz): stories(count, offset) -> (valid
    siralari, [[eot] + metin + [eot]] en cok _FINEWEB_TOKENS + 1 token).  Sira: sinav alt kumesi once, sonra valid'in geri
    kalani (tohum 0 permutasyonu); [offset, offset + count) dilimi, valid sirasiyla sirali.  Tek belge: paketleme yok."""
    sys.path.insert(0, os.path.join(HERE, "train_fineweb"))
    import data_fineweb as DF
    v = DF.load_valid(root, log=lambda s: None)
    assert len(v["vocab"]) == config["vocab"], "sozluk %d, kosu %d" % (len(v["vocab"]), config["vocab"])
    perm = np.random.default_rng(0).permutation(len(v["valid_starts"]))
    order = np.concatenate([v["exam"], perm[~np.isin(perm, v["exam"])]])

    def stories(count, offset=0):
        rows = np.sort(order[offset:offset + count])
        return rows, [(DF.valid_doc(v, int(r)) + [v["eot"]])[:_FINEWEB_TOKENS + 1] for r in rows]

    shards = config.get("shards") or [i for i in v["complete"] if i != DF.VALID_SHARD]
    return dict(tag="fineweb_" + DF.TAG, vocab=v["vocab"], eos=v["eot"], stories=stories,
                encode=lambda s: DF.DS.encode(s, v["vocab"]),
                train=[os.path.join(root, DF.TAG, "shard_%03d.bin" % i) for i in shards])


def _is_fineweb(root):
    return bool(root) and os.path.exists(os.path.join(root, "gpt2", "tokenizer.json")) and any(
        f.startswith("shard_") and f.endswith(".json") for f in os.listdir(os.path.join(root, "gpt2")))


def _token_counts(data, config, run_dir, path=None, log=print):
    """Train akisinin token sayimi: path verilirse oradan; yoksa <kosu>/internals/token_counts_<tag>_<iz>.npy (bir kez
    sayilir, sonra okunur -- train.npy buyuk)."""
    if path:
        return np.load(path)
    cache = os.path.join(run_dir, "internals", "token_counts_%s_%s.npy" % (data["tag"], config.get("fingerprint", "x")))
    if os.path.exists(cache):
        return np.load(cache)
    log("train sayimi: %s (bir kez; sonra %s)" % (data["train"], cache))
    paths = data["train"] if isinstance(data["train"], (list, tuple)) else [data["train"]]   # FineWeb: parca .bin'leri
    counts = np.zeros(len(data["vocab"]), np.int64)
    for path in paths:
        a = np.memmap(path, dtype=np.uint16, mode="r") if path.endswith(".bin") else np.load(path, mmap_mode="r")
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
        CM3 / CM             tur 3'te / butun turlarda Canon eki (sum_k w_k x_(t-k)) referans ortalamasiyla
        aA3=0.5 / aF6*0.5    tur 3'un alpha_attention'i 0,5 (butun boyutlar) / tur 6'nin alpha_facts'i x 0,5
    shape: (tur sayisi, head sayisi, Block sayisi)."""
    turns, heads, nb = shape
    parts, words = [], []
    for part in text.split("+"):
        m = re.fullmatch(r"none|([AF])(\d+)|H(\d+)(?:\.(\d+))?|C(\d+)?|a([AF])(\d+)([=*])(-?[\d.]+(?:e-?\d+)?)|CM(\d+)?",
                         part)
        assert m, "mudahale okunamadi: %r (python internals_y.py -h)" % part
        turn = next((int(g) - 1 for g in (m.group(2), m.group(3), m.group(5), m.group(7), m.group(10)) if g), None)
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
        elif part.startswith("CM"):
            ts = [turn] if turn is not None else list(range(turns))
            parts += [("canon_mean", t, None) for t in ts]
            words.append(where + "Canon eki ortalamayla" if turn is not None else "butun turlarda Canon eki ortalamayla")
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
            elif key in ("heads", "canon_mean"):
                case.setdefault(key, {})[where] = None
            elif key == "canon":
                case.setdefault(key, {})[where] = torch.zeros(4)
            else:
                op, val = arg
                learned = getattr(model, key)[where].detach()
                case.setdefault(key, {})[where] = learned * val if op == "*" else torch.full_like(learned, val)
        return case

    return text, ", ".join(words), build


def _standard_cases(turns, heads, first_turn_facts=True):
    """Varsayilan ablate listesi: none, her tur A / F / H, her head, her tur C ve CM, butun Canon (C, CM); F1 yalniz tur 1'de
    FactUnits varsa."""
    cases = ["none"] + ["A%d" % t for t in range(1, turns + 1)]
    cases += ["F%d" % t for t in range(1 if first_turn_facts else 2, turns + 1)]
    cases += ["H%d" % t for t in range(1, turns + 1)]
    if heads > 1:
        cases += ["H%d.%d" % (t, h) for t in range(1, turns + 1) for h in range(heads)]
    return cases + ["C%d" % t for t in range(1, turns + 1)] + ["C"] + ["CM%d" % t for t in range(1, turns + 1)] + ["CM"]


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
    L = ["## token noktalari: %d token, d %d, capa %.1e (amacta %.4f)" % (
        res["tokens"], res["d"], res["anchor"], res["anchor_loss"]),
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
    L = ["## FactUnits birimleri (%d birim): %d gecerli konum" % (res["units"], res["n"]),
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


def _text_tuned_lens(res, vocab, positions):
    f, ex = res["fit"], res["example"]
    L = ["## tuned lens (Belrose 2023): durum basina afin cevirici h + W h + b (birimden baslar), modelin cikis basligiyla; "
         "amac son dagilima KL",
         "cevirici: %s -- %d hikaye, %d token; %d adim x %d hikaye, Adam lr %.0e dogrusal inis" % (
             f["source"] or "verilen hikayeler", f["stories"], f["tokens"], f["steps"], f["batch"], f["lr"]),
         "degerlendirme: %d hikaye, %d hedef (cevirici verisinden ayri).  KL nat, son dagilima; sira: dogru token'in medyan "
         "sirasi; agree: en yuksek skorlu token son durumunkiyle ayni; egitim KL: ilk / son epok (%d adim) ortalamasi" % (
             res["stories"], res["n"], f["epoch_steps"]),
         "durum | blok | logit lens: nll    acc      KL   sira  agree | tuned lens: nll    acc      KL   sira  agree | "
         "egitim KL ilk -> son"]
    for i, (name, blk) in enumerate(zip(res["states"], res["blocks"])):
        row = []
        for k in ("logit", "tuned"):
            s = res["summary"][name][k]
            row += [s["nll"], s["acc"], s["kl"], s["median_rank"], s["agree"]]
        L.append("%-5s | %s |          %7.3f %.4f %7.3f %6.0f %.3f |          %7.3f %.4f %7.3f %6.0f %.3f | %s" % (
            name, blk, *row, "-" if f["kl_first"] is None else "%.3f -> %.3f" % (f["kl_first"][i], f["kl_last"][i])))
    if ex:
        tok = lambda i: (vocab[i] if vocab is not None else str(i))[:8]
        n = min(positions, len(ex["ids"]) - 1)
        L += ["", "## ilk hikaye: tuned lens'in en yuksek skorlu token'i, durum durum; ilk %d hedef" % n,
              "%4s  %-8s -> %-8s | %s" % ("konum", "girdi", "hedef", " ".join("%-8s" % s for s in res["states"]))]
        for t in range(n):
            L.append("%4d%s %-8s -> %-8s | %s" % (t, " " if ex["valid"][t] else "*", tok(ex["ids"][t]), tok(ex["ids"][t + 1]),
                                                 " ".join("%-8s" % tok(ex["top"][s][t]) for s in res["states"])))
        L.append("(* = sayilmayan hedef)")
    return L


def _write(out_dir, name, lines, payload):
    """<out_dir>/<name>.txt (satirlar) + .json (payload) -> yol (uzantisiz)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False)
    return path


def _text_ablate(res):
    b = res["base"]
    L = ["## ablate.  " + res["note"],
         "taban: nll %.4f  acc %.4f  (%d hedef, %d hikaye; elle ileri hesap - model.logits en buyuk fark %.1e)%s" % (
             b["nll"], b["acc"], b["n"], b["stories"], res["check_logits"],
             "" if res["reference"] is None else "; ortalamalar (head, Canon eki) %s" % res["reference"]),
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
    if measure == "tuned_lens":
        return {"%s %s nll" % (n, k): res["summary"][n][k]["nll"] for n in res["states"] for k in ("logit", "tuned")}
    return {r["case"].split()[0]: r["d_nll"] for r in res["rows"]}


def _override(config, pairs):
    """anahtar=deger listesi (deger JSON, olmazsa metin): BlockModel parametresi model_kw'ye, gerisi config'e (analysis)."""
    config = dict(config, model_kw=_model_kw(config))
    model_keys = set(inspect.signature(BlockModel).parameters)
    for pair in pairs or ():
        key, _, value = pair.partition("=")
        try:
            value = json.loads(value)
        except ValueError:
            pass
        (config["model_kw"] if key in model_keys else config)[key] = value
    return config


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


_MEASURES = ("trace", "point_drift", "attention_stats", "unit_usage", "ablate", "tuned_lens")
_DEFAULT_STORIES = dict(trace=4, point_drift=0, attention_stats=128, unit_usage=128, ablate=128, tuned_lens=64)


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
    ap.add_argument("--cases", nargs="+", help="ablate mudahaleleri (varsayilan: none, her tur A/F/H/C/CM, her head, C, CM)")
    ap.add_argument("--reference", type=int, default=48, help="ablate: head / Canon eki ortalamasi icin ayri hikaye sayisi")
    ap.add_argument("--counts", help="point_drift: token sayimi .npy (varsayilan: train akisi, bir kez sayilir)")
    ap.add_argument("--batch", type=int, help="batch (varsayilan 16; attention_stats, tuned_lens 8)")
    ap.add_argument("--data", help="veri koku (varsayilan: tag'e gore %s; FineWeb kosusunda %s); FineWeb klasoru "
                                   "verilirse valid belgeleri (tek belge, en cok %d token)" % (
                                       _SIMPLESTORIES_ROOT, _FINEWEB_ROOT, _FINEWEB_TOKENS))
    ap.add_argument("--fit-stories", type=int, default=256, help="tuned_lens: cevirici hikaye sayisi")
    ap.add_argument("--fit-offset", type=int, default=10000, help="tuned_lens: cevirici hikayeleri sinav permutasyonunda "
                                                                  "buradan (degerlendirmeyle ortusmez)")
    ap.add_argument("--lens-steps", type=int, default=200, help="tuned_lens: cevirici adimi (0: birim = logit lens)")
    ap.add_argument("--lens-lr", type=float, default=1e-3, help="tuned_lens: Adam lr (dogrusal iner)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda: model bu cihazda (butun olcumler; point_drift CPU'da)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):              # Windows konsolu (cp1254) Δ yazamiyor; dosyalar utf-8
        sys.stdout.reconfigure(errors="replace")
    torch.set_num_threads(4)
    t0 = time.time()
    say = lambda s: print(s, flush=True)
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(_RUNS_ROOT, args.run)
    assert os.path.isdir(run_dir), "kosu klasoru yok: %s" % run_dir
    config = _config(run_dir)
    # FineWeb kosusu ya da --data bir FineWeb klasoru: valid belgeleri tek tek (tag gpt2 SimpleStories'e gitmesin)
    fineweb = config.get("dataset") == "fineweb-edu" or _is_fineweb(args.data)
    loader = (_fineweb, _FINEWEB_ROOT) if fineweb else _DATA.get(config.get("tag"))
    if loader is None:
        raise SystemExit("veri yukleyicisi yok: tag %r (yalniz SimpleStories: ss4096, gpt2); olcum fonksiyonlari id "
                         "listeleriyle dogrudan cagrilabilir" % config.get("tag"))
    data = loader[0](config, args.data or loader[1])
    vocab = data["vocab"]
    count = _DEFAULT_STORIES[args.measure] if args.stories is None else args.stories
    batch = args.batch or (8 if args.measure in ("attention_stats", "tuned_lens") else 16)
    if args.measure == "trace" and args.text and args.stories is None:
        count = 0
    rows, stories = data["stories"](count, args.offset) if count else (np.array([], dtype=np.int64), [])
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
    elif args.measure == "tuned_lens":
        fit_end = args.fit_offset + args.fit_stories
        assert fit_end <= args.offset or args.fit_offset >= args.offset + count, "cevirici ve degerlendirme hikayeleri ortusuyor"
        fit_rows, fit = data["stories"](args.fit_stories, args.fit_offset) if args.lens_steps else ([], None)
        where = ("sinav permutasyonu %d..%d (valid; degerlendirmeden ayri)" % (args.fit_offset, fit_end - 1)
                 if args.lens_steps else "yok (birim)")
        measure = lambda m: {k: v for k, v in tuned_lens(m, stories, fit, steps=args.lens_steps, lr=args.lens_lr, batch=batch,
                                                         fit_source=where, log=say).items() if k != "translators"}
        text = lambda res: _text_tuned_lens(res, vocab, args.positions)
        summarize = lambda res: _summary("tuned_lens", res)
    else:
        cases = [_parse_case(shape, c) for c in (args.cases or _standard_cases(shape[0], shape[1],
                                                                              skeleton.first_turn_facts))]
        needs_ref = any("H" in c[0] or "CM" in c[0] for c in cases)
        ref = data["stories"](args.reference, args.offset + count)[1] if needs_ref and args.reference else None
        measure = lambda m: ablate(m, stories, [("%-8s %s" % (name, words), build(m)) for name, words, build in cases],
                                   reference=ref, batch=batch, log=say)
        text, summarize = _text_ablate, lambda res: _summary("ablate", res)
        if ref is not None:
            lines.append("head / Canon eki ortalamalari icin ayri %d hikaye: sinav permutasyonu %d..%d" % (
                len(ref), args.offset + count, args.offset + count + len(ref) - 1))

    packs = _checkpoints(run_dir)
    if args.checkpoints:
        spec = args.checkpoints
        steps = (list(packs) if spec == "all" else [s for s in packs if s % int(spec[6:]) == 0] if spec.startswith("every:")
                 else [int(s) for s in spec.split(",")])
        results = over_checkpoints(run_dir, lambda m: measure(m.to(args.device)), steps, args.weights, log=say)
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
        model = model.to(args.device)                         # GPU'da tuned_lens ~25 sn (CPU'da ~1 saat)
        source = ("checkpoint_t%06d.pt" % step) if step is not None else (
            "model_weight_ema.pt" if args.weights == "ema" else "model.pt")
        tag = "" if step is None else "t%06d" % step
        result = measure(model)
        lines += text(result)
    header = ["internals_y %s | kosu %s | %s | %s | sure %.0f sn" % (
        args.measure, config.get("name", os.path.basename(run_dir)), "agirlik %s (%s)" % (args.weights, source),
        time.strftime("%Y-%m-%d %H:%M"), time.time() - t0)]
    if count:
        header.append("sinav hikayeleri: %d (tohum 0 permutasyonu %d..%d; valid sirasi %d..%d)" % (
            count, args.offset, args.offset + count - 1, int(rows.min()), int(rows.max())))
    weights = args.weights
    name = "_".join(p for p in (args.measure, weights, tag, time.strftime("%Y%m%d_%H%M%S")) if p)
    path = _write(os.path.join(run_dir, "internals"), name, header + [""] + lines,
                  dict(measure=args.measure, run=config.get("name"), weights=weights, source=source,
                       stories=np.asarray(rows).tolist(), offset=args.offset, texts=args.text, result=result))
    print("\n".join(header + [""] + lines))
    say("yazildi: %s.txt / .json" % path)


if __name__ == "__main__":
    _main()
