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
    profile_step      egitim adiminin parca parca suresi ve bellegi (ileri, kayip, geri, Muon / Adam, EMA, coherence ...);
                      model train_seq'in kurulumundan, egitilmis agirlik kullanilmaz.  Gercek olcum GPU'da
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
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model_y import BlockModel, deviation  # noqa: E402

_RUNS_ROOT = "G:/Drive'ım/model_y"                  # kosu klasorleri (Drive, bu bilgisayarda)
_SIMPLESTORIES_ROOT = "G:/Drive'ım/simplestories"    # <tag>/tokenizer.json, <tag>/valid.npy, exam_stories.npy
_FINEWEB_ROOT = "G:/Drive'ım/fineweb"                # gpt2/tokenizer.json, gpt2/shard_NNN.* (data_fineweb)
_FINEWEB_TOKENS = 2048                               # FineWeb belgesi en cok bu kadar girdiyle (tek belge, paketsiz)
# train_y.train_seq'teki Muon listesi: yedekteki optimizer parametre sirasi buna bagli (once bunlar)
_MUON_HIDDEN = ("W_context", "W_value", "W_fact_in", "W_fact_up", "W_fact_out", "W_value.weight", "W_out.weight",
                "W_mlp_in.weight", "W_mlp_out.weight")
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
        canon_mean {tur: (d,) vektor | None}: Canon eki (sum_k w_k x_(t-k)) bu vektorle degisir; None = ortalama
                   (ablate doldurur)
        alpha_attention, alpha_facts   {tur: sayi | (d,) vektor}: o turun alpha'si ELLE (yalniz normalized_update)
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
            assert model.normalized_update, "alpha yalniz normalized_update'te var"
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
                w = w * torch.as_tensor(plan["canon"][t], dtype=w.dtype, device=w.device)[:, None]
            T = x.shape[-2]
            full = F.pad(x, (0, 0, 3, 0))
            mix = sum(w[k] * full[..., 3 - k:3 - k + T, :] for k in range(4))
            if t in plan["canon_mean"]:
                vec = plan["canon_mean"][t]
                assert vec is not None, "Canon eki ortalamasi doldurulmadi (ablate doldurur)"
                mix = torch.as_tensor(vec, dtype=x.dtype, device=x.device).expand_as(x)
            if "canon" in taps:
                taps["canon"](t, (mix, x))
            x = x + mix
        else:
            assert t not in plan["canon_mean"], "tur %d: Canon yok" % (t + 1)
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
                    c[:, hh] = torch.as_tensor(vec, dtype=c.dtype, device=c.device)
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
        f = _turn_facts(model, t)
        if t not in plan["skip_facts"] and f is not None:
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
        elif f is not None and blk.stream_norm and not blk.normalized_update:   # mudahale: katki 0; model atlamasi normsuz
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
      zero          (relu / reglu) u_i = 0 oldugu konum orani, ortalama
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
    return dict(units=U, n=n, activation=f0.activation, turns=turns, teams=teams)


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
    """Lens skoru: cevirici h + W h + b (W, b yoksa birim), sonra modelin cikis basligi.  Akis normunda girdi kureye
    indirilir (son durum da kurede; tuned lens'teki son LayerNorm'un karsiligi)."""
    if W is not None:
        h = h + h @ W.T + b
    if model.stream_norm and not model.layer_norm:
        h = _unit(h)
    return _scores(model, h, P)


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
    """config'teki model_kw; output_link / shared_facts yazilmamis eski config'lerde yoktu (colab_simplestories'in
    surdurmesi gibi)."""
    return dict(dict(output_link=False, shared_facts=True, input_embedding=False, input_bigrams=0, first_turn_facts=True,
                     input_embedding_sphere=False), **config.get("model_kw", {}))


def _build(config):
    """config.json -> bos BlockModel (agirliklar sonra yuklenir)."""
    setting = config.get("setting", "shared")
    assert setting in ("shared", "separate"), "yalniz BlockModel (setting shared / separate), bu kosu: %s" % setting
    kw = _model_kw(config)
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


# ---- hiz: egitim adiminin parcalari

class _Clock:
    """Parca sureleri: host saati ve GPU'da CUDA event'leri (akis sirasiyla: GPU CPU'yu beklese de sayilir); GPU'da ust
    duzey parcanin tepe bellegi (giristeki ayrilmis bellegin ustu, MB).  Ic ice parca (Newton-Schulz) yalniz sure.
    mark(ad): nokta; onceki noktadan buraya kadarki parca ad."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.rows, self.marks, self.depth = [], [], 0

    def now(self):
        e = None
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        return time.perf_counter(), e

    def ms(self, a, b):
        """(host ms, cihaz ms) iki an arasi; cihaz CPU'da host ile ayni.  GPU'da once read (senkron)."""
        host = 1e3 * (b[0] - a[0])
        return host, (a[1].elapsed_time(b[1]) if self.cuda else host)

    @contextlib.contextmanager
    def part(self, name):
        top, i = self.depth == 0, len(self.rows)
        self.rows.append(None)                           # yer: ic ice parca ustunden sonra listelenmesin
        if self.cuda and top:
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        a = self.now()
        self.depth += 1
        try:
            with torch.profiler.record_function(name):
                yield
        finally:
            self.depth -= 1
        b = self.now()
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 if self.cuda and top else None
        self.rows[i] = (name, a, b, peak, top)

    def mark(self, name):
        self.marks.append((name, self.now()))

    def read(self):
        """-> ({ad: [host ms, cihaz ms, tepe MB, ust duzey]}, ayni ad toplanir; [(ad, host ms, cihaz ms)] noktalar arasi);
        temizler."""
        if self.cuda:
            torch.cuda.synchronize()
        parts = {}
        for name, a, b, peak, top in self.rows:
            r = parts.setdefault(name, [0.0, 0.0, None, top])
            host, dev = self.ms(a, b)
            r[0], r[1] = r[0] + host, r[1] + dev
            if peak is not None:
                r[2] = max(r[2] or 0.0, peak)
        segments = [(name, *self.ms(a, b)) for (_, a), (name, b) in zip(self.marks, self.marks[1:])]
        self.rows, self.marks = [], []
        return parts, segments


def _train_kwargs(config):
    """config -> train_seq ayarlari: yalniz train_seq'in parametreleri (adim, kayit, cihaz, compile ve veri haric)."""
    import train_y as TR
    skip = {"setting", "ids", "mask", "n", "steps", "log_at", "every", "callback", "save_every", "save", "checkpoint",
            "batches", "device", "compile"}
    kw = {k: config[k] for k in set(inspect.signature(TR.train_seq).parameters) - skip if k in config}
    if config.get("setting", "shared") in TR.STEP3:
        kw["model_kw"] = _model_kw(config)
    return kw


def _forward_context(ctx):
    """train_seq'in ileri hesap baglami: compile kilidi, SDPA cekirdegi (attention_kernel), bf16 autocast (GPU)."""
    stack = contextlib.ExitStack()
    if ctx["compile"]:
        stack.enter_context(ctx["TR"].COMPILE_LOCK)
    stack.enter_context(sdpa_kernel(ctx["kernels"]))
    if ctx["bf16"]:
        stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
    return stack


def _profile_setup(config, cached, device):
    """train_seq'in kendi kurulumu, 1 adim: model, optimizer (Muon / Adam gruplari, kure eksenleri), EMA onun kodundan.
    -> adimin parcalarinin ihtiyaci (ctx)."""
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
    sphere = getattr(model, "sphere_weights", False)
    cuda = torch.device(device).type == "cuda"
    compile = cuda and config.get("compile", True)       # train_seq gibi: compile yalniz GPU'da
    return dict(TR=TR, model=model, opt=got["opt"], named=named, params=[p for _, p in named], device=device, cuda=cuda,
                unit_axis={k: int(k.split(".")[-1] in rows) for k, _ in named if sphere and k.split(".")[-1] in rows + cols},
                split=get("schedule") == "coherence", grad_clip=get("grad_clip"), ema=getattr(model, "weight_ema", None),
                compile=compile, loss_fn=torch.compile(model.loss) if compile else model.loss,
                bf16=cuda and get("matmul_precision") == "bf16", tf32=cuda and get("matmul_precision") == "tf32",
                kernels=([SDPBackend.MATH] if not cuda or get("attention_kernel") == "math" else     # train_seq gibi
                         [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))


def _step_parts(ctx, ids, mask, clock):
    """train_seq'in bir adimi, ayni sirayla parca parca (lr takvimi ve coherence'in skaler ortalamasi haric: ikisi de
    birkac sayi).  Muon ve Adam gruplari ayri step'lerle (ayni hesap); Newton-Schulz Muon'un icinde ayrica."""
    TR, model, opt = ctx["TR"], ctx["model"], ctx["opt"]
    with clock.part("batch -> cihaz"):
        ids, mask = ids.to(ctx["device"]), mask.to(ctx["device"])
    with clock.part("ileri + kayip"), _forward_context(ctx):
        if ctx["split"]:
            halves = [ctx["loss_fn"](ids[h::2].contiguous(), mask[h::2].contiguous()) for h in (0, 1)]
            weights = [mask[h::2, 1:].sum() for h in (0, 1)]
            sum(w * hn for w, (_, hn) in zip(weights, halves)) / sum(weights)     # train_seq'in nll'i (kullanilmaz)
        else:
            total, _ = ctx["loss_fn"](ids, mask)
    with clock.part("zero_grad"):
        opt.zero_grad()
    if ctx["split"]:                                     # coherence: iki yarinin gradyani, g1 kopyasi, birlestirme
        grads = None
        for half_total, _ in halves:
            with clock.part("geri yayilim"), sdpa_kernel(ctx["kernels"]):
                half_total.backward()
            if grads is None:
                with clock.part("coherence: g1 kopyasi"):
                    grads = [None if p.grad is None else p.grad.detach().clone() for p in ctx["params"]]
        with clock.part("coherence: birlestirme, teget, carpimlar"):
            sums = []                                    # train_seq gibi: parametre basina carpimlar, sonda tek senkron
            w0, w1 = (float(w) for w in weights)
            for (k, p), a in zip(ctx["named"], grads):
                if a is None:
                    p.grad = None
                    continue
                b = p.grad - a
                p.grad = (w0 * a + w1 * b) / (w0 + w1)
                if k in ctx["unit_axis"]:
                    a, b = (v - (v * p.detach()).sum(ctx["unit_axis"][k], keepdim=True) * p.detach() for v in (a, b))
                sums.append(torch.stack([(a * b).sum(), (a * a).sum(), (b * b).sum()]))
            torch.stack(sums).tolist()
    else:
        with clock.part("geri yayilim"), sdpa_kernel(ctx["kernels"]):
            total.backward()
    if ctx["grad_clip"] is not None:
        with clock.part("clip"):
            torch.nn.utils.clip_grad_norm_(ctx["params"], ctx["grad_clip"])
    if isinstance(opt, TR.Muon):
        def orthogonalize(G, steps=5, precision="fp32"):
            with clock.part("  Newton-Schulz"):
                return TR.Muon.orthogonalize(G, steps, precision)

        groups = opt.param_groups
        opt.orthogonalize = orthogonalize                # ornek ozelligi sinifin staticmethod'unu golgeler
        try:
            for name, muon in (("Muon (Newton-Schulz dahil)", True), ("Adam", False)):
                opt.param_groups = [g for g in groups if g["use_muon"] == muon]
                with clock.part(name):
                    opt.step()
        finally:
            opt.param_groups = groups
            del opt.orthogonalize
    else:
        with clock.part("Adam"):
            opt.step()
    sphere = getattr(model, "sphere_weights", False)
    if sphere:
        with clock.part("normalize_weights"):
            model.normalize_weights()
    if getattr(model, "output_link", False):
        with clock.part("link_u kirpma"), torch.no_grad():
            model.link_u.clamp_(min=0)
    if ctx["ema"] is not None:
        with clock.part("EMA"), torch.no_grad():
            ema, decay = ctx["ema"]["model"], ctx["ema"]["decay"]
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
            if sphere:
                ema.normalize_weights()


def _breakdown_parts(ctx, ids, mask, clock):
    """Ileri hesap dokumu, eager: gomme; tur basina Canon, attention (W_context ve guncelleme dahil), FactUnits (guncelleme
    dahil) _run'in noktalarindan; kayip basligi modelin kendi loss'u (hidden yerine _run'in son durumu); geri yayilim iki
    parca: kayip basligi (son durumun gradyanina kadar) ve govde."""
    model = ctx["model"]
    ids, mask = ids.to(ctx["device"]), mask.to(ctx["device"])
    at = lambda name: (lambda t, _: clock.mark(name % (t + 1)))
    with _forward_context(dict(ctx, compile=False)):
        clock.mark("basla")
        h0 = model.input_states(ids[:, :-1])
        clock.mark("gomme")
        h = _run(model, h0, _plan(model), dict(canon=at("tur %d Canon"), attention_out=at("tur %d attention"),
                                               facts_out=at("tur %d FactUnits")))
        model.hidden = lambda ids_, caches=None, document_positions=None: [h]   # kayip basligi: modelin loss'u, govde _run'dan
        try:
            total, _ = model.loss(ids, mask)
        finally:
            del model.hidden
        clock.mark("kayip basligi ileri")
    h.register_hook(lambda g: clock.mark("kayip basligi geri"))
    with sdpa_kernel(ctx["kernels"]):
        total.backward()
    clock.mark("govde geri (turlar, gomme)")
    model.zero_grad(set_to_none=True)


def _real_step_ms(config, cached, device, compile):
    """train_seq'in kendisi, ayni batch'lerle 2 K adim (ilk K isinma); ikinci K adimin adim basina suresi.  Olcum
    noktalari geri cagrida (train_seq nll.item() ile esitler)."""
    import train_y as TR
    K, marks = len(cached), []
    TR.train_seq(config.get("setting", "shared"), None, None, config["vocab"], steps=2 * K, device=device, compile=compile,
                 log_at=(), every=K, callback=lambda s, m, nll: marks.append(time.perf_counter()),
                 batches=lambda s: cached[s % K], **_train_kwargs(config))
    return 1e3 * (marks[2] - marks[1]) / K


def _step_flops(model, ids):
    """Adim basina model FLOP'u (tahmin): 3 x ileri.  Ileri, konum basina: tur basina 2 d (d x attention matrisi sayisi +
    birim x FactUnits matrisi sayisi) + SDPA math'in tam tablosu 4 d T; cikis 2 d V.  Dolgu konumlari da hesaplanir."""
    if not isinstance(model, BlockModel):
        return None
    V, d = model.tokens.fixed_points.shape
    B, T = ids.shape[0], ids.shape[1] - 1
    per = 2 * d * V
    for t, blk in enumerate(model.turn_blocks()):
        f = _turn_facts(model, t)
        per += 2 * d * (d * (3 + (blk.attention.heads > 1))
                        + (f.W_fact_out.shape[1] * (2 if f.activation == "relu" else 3) if f is not None else 0))
        per += 4 * d * T
    return 3 * B * T * per


def profile_step(config, batches, steps=5, device="cpu", real=True, ops=0, log=None, out_dir=None):
    """Bir egitim adiminin parca parca suresi (ms) ve bellegi.
    config: kosunun config.json'u (_config) ya da ayni anahtarlarla sozluk (setting, vocab, model_kw, schedule,
    weight_ema, matmul_precision, compile ...; olmayan anahtar train_seq varsayilani).  batches: adimin fonksiyonu
    step -> (ids, mask) (data_simplestories.batches) ya da (ids, mask) listesi; ilk `steps` batch bir tur isinma (compile,
    her sekil bir kez), bir tur olcum.
    Yol (b), parcalar ayri zamanlanir: train_seq'in adimi tek parca bir dongu, ara parcalari (EMA, coherence, clip)
    train_y degismeden disaridan zamanlanamaz.  Model, optimizer ve EMA train_seq'in kendi 1 adimlik kurulumundan (yeni
    nesneler; kullanicinin modeline dokunulmaz), adim train_seq'in sirasiyla parca parca tekrarlanir (_step_parts).
    Tekrarin sadakati: real=True ise train_seq'in kendisi ayni batch'lerle kosar, adim suresi yan yana yazilir.
    Ayrica eager ileri hesap dokumu (_breakdown_parts: tur basina Canon / attention / FactUnits, kayip basligi, geri
    yayilim iki parca) ve ops > 0 ise bir adimin torch.profiler tablosu (en pahali ops islem).
    GPU'da ayni anda baska bir kosu varsa sureler bozulur.  CPU sureleri GPU'yu temsil etmez.
    -> dict(setup, parts [{part, host_ms, device_ms, peak_mb, nested}], total, unmeasured_ms, per_step, breakdown,
    breakdown_total, real_step_ms, flops, ops).  out_dir: <out_dir>/profile_step_<cihaz>_<zaman>.txt + .json."""
    say = log or (lambda s: None)
    t_start = time.time()
    cached = [batches(i) if callable(batches) else batches[i] for i in range(steps)]
    assert all(len(b) == 2 for b in cached), "paketli batch (document_positions) desteklenmiyor: adim parcalari (ids, mask) ile"
    precision, grad = torch.get_float32_matmul_precision(), torch.is_grad_enabled()
    rows, whole, breakdown, op_lines, real_ms = [], [], [], None, None
    torch.set_grad_enabled(True)
    try:
        ctx = _profile_setup(config, cached, device)
        model, clock = ctx["model"], _Clock(device)
        if ctx["tf32"]:
            torch.set_float32_matmul_precision("high")
        for i in range(2 * steps):                       # ilk tur isinma
            a = clock.now()
            _step_parts(ctx, *cached[i % steps], clock)
            b = clock.now()
            parts, _ = clock.read()
            if i >= steps:
                rows.append(parts)
                whole.append(clock.ms(a, b))
        say("profile_step: %d adim olculdu (%.0f sn)" % (steps, time.time() - t_start))
        if ops:                                          # dokumden once: model.hidden yamasi compile'a dokunmasin
            acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if ctx["cuda"] else [])
            with torch.profiler.profile(activities=acts) as prof:
                _step_parts(ctx, *cached[0], _Clock(device))
                if ctx["cuda"]:
                    torch.cuda.synchronize()
            key = "self_device_time_total" if ctx["cuda"] else "self_cpu_time_total"
            op_lines = prof.key_averages().table(sort_by=key, row_limit=ops).splitlines()
        if isinstance(model, BlockModel):
            for i in range(steps + 1):                   # ilki isinma
                _breakdown_parts(ctx, *cached[i % steps], clock)
                if i:
                    breakdown.append(clock.read()[1])
                else:
                    clock.read()
        flops = [_step_flops(model, ids) for ids, _ in cached]
        kw = _train_kwargs(config)
        setup = dict(run=config.get("name"), device=str(device), gpu=torch.cuda.get_device_name(0) if ctx["cuda"] else None,
                     compile=ctx["compile"], precision="bf16" if ctx["bf16"] else "tf32" if ctx["tf32"] else "fp32",
                     split=ctx["split"], optimizer=type(ctx["opt"]).__name__,
                     weight_ema=None if ctx["ema"] is None else ctx["ema"]["decay"],
                     params=sum(p.numel() for p in ctx["params"]), batches=[list(ids.shape) for ids, _ in cached],
                     targets=[int(m[:, 1:].sum()) for _, m in cached], model_kw=kw.get("model_kw"),
                     schedule=kw.get("schedule"), grad_clip=ctx["grad_clip"])
        compile, cuda = ctx["compile"], ctx["cuda"]
        del ctx, model                                   # gercek kosudan once bellek bosalsin
        if cuda:
            torch.cuda.empty_cache()
        if real:
            say("profile_step: train_seq'in kendisi, %d adim" % (2 * steps))
            real_ms = _real_step_ms(config, cached, device, compile)
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.set_grad_enabled(grad)
    med = lambda v: float(np.median(v))
    names = list(rows[0])
    top = [n for n in names if rows[0][n][3]]
    parts = [dict(part=n, host_ms=med([r[n][0] for r in rows]), device_ms=med([r[n][1] for r in rows]),
                  peak_mb=max(r[n][2] for r in rows) if cuda and rows[0][n][3] else None, nested=not rows[0][n][3])
             for n in names]
    per_step = [dict(total_ms=w[1], parts_ms=sum(r[n][1] for n in top)) for r, w in zip(rows, whole)]
    total = dict(host_ms=med([w[0] for w in whole]), device_ms=med([w[1] for w in whole]))
    res = dict(setup=setup, parts=parts, total=total, per_step=per_step,
               unmeasured_ms=med([s["total_ms"] - s["parts_ms"] for s in per_step]),
               breakdown=[dict(part=n, host_ms=med([b[j][1] for b in breakdown]), device_ms=med([b[j][2] for b in breakdown]))
                          for j, (n, _, _) in enumerate(breakdown[0])] if breakdown else None,
               real_step_ms=real_ms, ops=op_lines,
               flops=None if flops[0] is None else dict(per_step=float(np.mean(flops)), tflops=float(np.mean(flops)) / (
                   total["device_ms"] * 1e9), tflops_real=None if real_ms is None else float(np.mean(flops)) / (real_ms * 1e9)))
    if res["breakdown"]:
        res["breakdown_total"] = dict(host_ms=sum(b["host_ms"] for b in res["breakdown"]),
                                      device_ms=sum(b["device_ms"] for b in res["breakdown"]))
    if out_dir:
        lines = ["internals_y profile_step | %s | %s | sure %.0f sn" % (
            config.get("name", "-"), time.strftime("%Y-%m-%d %H:%M"), time.time() - t_start), ""] + _text_profile(res)
        path = _write(out_dir, "profile_step_%s_%s" % (torch.device(device).type, time.strftime("%Y%m%d_%H%M%S")), lines,
                      dict(measure="profile_step", run=config.get("name"), result=res))
        say("\n".join(lines) + "\nyazildi: %s.txt / .json" % path)
    return res


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


def _text_profile(res):
    c, t = res["setup"], res["total"]
    L = ["## egitim adimi parca parca: %s%s | %s, compile %s, %s | %d batch %s, %d adimin medyani (once bir tur isinma)" % (
             c["device"], " (%s)" % c["gpu"] if c["gpu"] else "", c["optimizer"], c["compile"], c["precision"],
             len(c["batches"]), " ".join("%dx%d" % tuple(b) for b in c["batches"]), len(res["per_step"])),
         "yol (b): model, optimizer ve EMA train_seq'in 1 adimlik kurulumundan (yeni nesneler); adim train_seq'in sirasiyla "
         "parca parca (lr takvimi ve coherence'in skaler ortalamasi haric)",
         "ayarlar: schedule %s (iki yari: %s), weight_ema %s, grad_clip %s, %d parametre, model_kw %s" % (
             c["schedule"], c["split"], c["weight_ema"], c["grad_clip"], c["params"], json.dumps(c["model_kw"])),
         "cihaz ms: %s" % ("CUDA event'leri (akis sirasiyla; GPU CPU'yu beklerken de sayar); host ms: CPU'nun parcada kaldigi "
                          "sure; tepe MB: parcanin girisindeki bellegin ustu" if c["gpu"] else
                          "CPU'da host ile ayni.  CPU sureleri GPU'yu TEMSIL ETMEZ"),
         "%-42s | %9s | %9s | %8s | %5s" % ("parca", "host ms", "cihaz ms", "tepe MB", "pay")]
    for p in res["parts"]:
        L.append("%-42s | %9.2f | %9.2f | %8s | %4.1f%%" % (p["part"], p["host_ms"], p["device_ms"], _fmt(p["peak_mb"], "%.0f"),
                                                          100 * p["device_ms"] / t["device_ms"]))
    top = sum(p["device_ms"] for p in res["parts"] if not p["nested"])
    L += ["%-42s | %9s | %9.2f |" % ("parcalarin toplami (ust duzey medyanlari)", "", top),
          "%-42s | %9.2f | %9.2f |" % ("olculen adim (butun)", t["host_ms"], t["device_ms"]),
          "%-42s | %9s | %9.2f |" % ("olculmeyen (adim - parcalar, medyan)", "", res["unmeasured_ms"])]
    if res["real_step_ms"] is not None:
        L.append("train_seq'in kendisi, ayni batch'lerle (compile %s): %.2f ms/adim  (tekrar / gercek %.3f)" % (
            c["compile"], res["real_step_ms"], t["device_ms"] / res["real_step_ms"]))
    fl = res["flops"]
    if fl:
        L.append("model FLOP'u (tahmin, 3 x ileri, dolgu dahil) %.3g / adim -> %.3g TFLOP/s (tekrar)%s.  MFU = bu / cihazin "
                 "tepe FLOP'u" % (fl["per_step"], fl["tflops"], "" if fl["tflops_real"] is None else
                                  ", %.3g TFLOP/s (train_seq)" % fl["tflops_real"]))
    if res["breakdown"]:
        bt = res["breakdown_total"]
        L += ["", "## ileri hesap dokumu, eager (compile yok): tur basina Canon | attention (W_context ve guncelleme dahil) | "
                  "FactUnits (guncelleme dahil); kayip basligi modelin kendi loss'u; geri yayilim iki parca",
              "%-42s | %9s | %9s | %5s" % ("parca", "host ms", "cihaz ms", "pay")]
        L += ["%-42s | %9.2f | %9.2f | %4.1f%%" % (b["part"], b["host_ms"], b["device_ms"], 100 * b["device_ms"] / bt["device_ms"])
              for b in res["breakdown"]]
        L.append("%-42s | %9.2f | %9.2f |" % ("toplam", bt["host_ms"], bt["device_ms"]))
    if res["ops"]:
        L += ["", "## torch.profiler: bir adim, en pahali islemler (parca adlari record_function)"] + res["ops"]
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


def _padded(rows, pad, multiple=8):
    """Id listeleri -> (ids, mask): sagdan pad ile dolgu, genislik multiple'in katina (bucket batch'leri gibi)."""
    T = -(-max(len(r) for r in rows) // multiple) * multiple
    ids = torch.full((len(rows), T), pad, dtype=torch.long)
    mask = torch.zeros(len(rows), T, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)], mask[i, :len(r)] = torch.tensor(r), True
    return ids, mask


def _override(config, pairs):
    """--set anahtar=deger (deger JSON, olmazsa metin): BlockModel parametresi model_kw'ye, gerisi config'e."""
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


_MEASURES = ("trace", "point_drift", "attention_stats", "unit_usage", "ablate", "tuned_lens", "profile_step")
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
    ap.add_argument("--batch", type=int, help="batch (varsayilan 16; attention_stats, tuned_lens 8; profile_step config'in "
                                              "batch_size'i)")
    ap.add_argument("--data", help="veri koku (varsayilan: tag'e gore %s; FineWeb kosusunda %s); FineWeb klasoru "
                                   "verilirse valid belgeleri (tek belge, en cok %d token)" % (
                                       _SIMPLESTORIES_ROOT, _FINEWEB_ROOT, _FINEWEB_TOKENS))
    ap.add_argument("--fit-stories", type=int, default=256, help="tuned_lens: cevirici hikaye sayisi")
    ap.add_argument("--fit-offset", type=int, default=10000, help="tuned_lens: cevirici hikayeleri sinav permutasyonunda "
                                                                  "buradan (degerlendirmeyle ortusmez)")
    ap.add_argument("--lens-steps", type=int, default=200, help="tuned_lens: cevirici adimi (0: birim = logit lens)")
    ap.add_argument("--lens-lr", type=float, default=1e-3, help="tuned_lens: Adam lr (dogrusal iner)")
    ap.add_argument("--profile-steps", type=int, default=5, help="profile_step: olculen adim (once bir tur isinma)")
    ap.add_argument("--width", type=int, help="profile_step: hikayeler bu token'da kesilir (kucuk ayar)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda: model bu cihazda (butun olcumler; point_drift CPU'da)")
    ap.add_argument("--no-real", action="store_true", help="profile_step: train_seq'in kendisini kosma")
    ap.add_argument("--ops", type=int, default=0, help="profile_step: torch.profiler tablosunda islem sayisi (0: yok)")
    ap.add_argument("--set", nargs="+", help="profile_step: ayar degistir, anahtar=deger (ornek shared_facts=false)")
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
    if args.measure == "profile_step":
        config = _override(config, args.set)
        batch = args.batch or config.get("batch_size", 64)
        count = args.profile_steps * batch
    else:
        assert not args.set, "--set yalniz profile_step"
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
    elif args.measure == "profile_step":
        assert not args.checkpoints and args.checkpoint is None, "profile_step agirlik kullanmaz (--checkpoint(s) yok)"
        cut = sorted((s[:args.width] if args.width else s for s in stories), key=len)   # benzer boylar bir batch'te
        cached = [_padded(cut[i:i + batch], data["eos"]) for i in range(0, len(cut), batch)]
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
    if args.measure == "profile_step":
        result = profile_step(config, cached, steps=len(cached), device=args.device, real=not args.no_real, ops=args.ops,
                              log=say)
        lines += _text_profile(result)
        source, tag = "agirlik kullanilmaz; batch'ler sinav hikayelerinden%s" % (
            ", %d token'da kesilmis" % args.width if args.width else ""), torch.device(args.device).type
    elif args.checkpoints:
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
        args.measure, config.get("name", os.path.basename(run_dir)),
        source if args.measure == "profile_step" else "agirlik %s (%s)" % (args.weights, source),
        time.strftime("%Y-%m-%d %H:%M"), time.time() - t0)]
    if count:
        header.append("sinav hikayeleri: %d (tohum 0 permutasyonu %d..%d; valid sirasi %d..%d)" % (
            count, args.offset, args.offset + count - 1, int(rows.min()), int(rows.max())))
    weights = None if args.measure == "profile_step" else args.weights
    name = "_".join(p for p in (args.measure, weights, tag, time.strftime("%Y%m%d_%H%M%S")) if p)
    path = _write(os.path.join(run_dir, "internals"), name, header + [""] + lines,
                  dict(measure=args.measure, run=config.get("name"), weights=weights, source=source,
                       stories=np.asarray(rows).tolist(), offset=args.offset, texts=args.text, result=result))
    print("\n".join(header + [""] + lines))
    say("yazildi: %s.txt / .json" % path)


if __name__ == "__main__":
    _main()
