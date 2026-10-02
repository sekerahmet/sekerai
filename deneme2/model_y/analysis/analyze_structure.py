# -*- coding: utf-8 -*-
"""analyze_structure -- Model Y ic yapi analizi (structure-analyst).  Modeli DEGISTIRMEZ; mudahaleler BAGIMLILIK olcer.

Olcumler (python analysis/analyze_structure.py <kosu> <olcum> --data <FineWeb koku> --device cuda):
    probes            her durumda (h0, A<t>, F<t>) ogrenilen dogrusal okuma: o anki / onceki 1-2 / sonraki 1-2 token (softmax,
                      butun sozluk) ve belge ici konum (ridge, log(1+t)); fit belgelerinde ogrenilir, sinav belgelerinde
                      olculur; yaninda logit lens.  --checkpoint, --states, --targets, --probe-wd, --probe-early-stop
    attention_passes  ayni Block'un ilk ve ikinci gecisteki attention'i head head: entropi, uzaklik, konum 0, kendisi,
                      onceki token, ayni token, induction, uzak kutle; iki gecis arasi q, k, v, c ve durum kosinusu
    units             tur ve alt blok basina guncelleme yonunun sabit / token payi; odak turun FactUnits birimleri
                      (baglamlar, token'a baglilik, |r| > 0,9 kumeleri); alpha isaretleri ve alpha mudahaleleri
    pass_swap         ikinci gecis attention'inin key / value'su eski bir durumdan (query guncel)
    extras            head ciktisinin konuma gore degisimi, sabit attention yonu, q.k yayilimi, Block E boyutlari;
                      ortalama / yeniden ornekleme / Canon k ayrimi / ikili mudahaleler
    repair            ayni alt bloklarda atla / ortalama / optimal sabit (Li & Janson 2024); sink head'lerin belge dagilimi
    points            cikis noktalari ve girdi tablosu (nokta acisi, |PF+D|, kappa, cos(E, PF)), s phi(1), seyrek token tabani
    loop              hedef kosinusu c_y siniflara gore ve dL/du payi; tekrarda durum yakinsamasi; kopyanin c_y'si gecis basina
    answers           son turun attention'i sonrasi / FactUnits sonrasi ilk 5 adayin c'si; element sembolleri ve metal yonu
    selfgen           m = 1 ve ayni l kovasinda acgozlu kendi uretimi / gercek devam ayrimi (tur <= 12, konum esitlenmis)

    python analysis/analyze_structure.py --selftest        (kucuk sahte model, CPU)
Cikti: $KUYRUK_SONUC (yoksa <kosu>/analysis/) structure_<olcum>_<agirlik>[_tNNNNNN]_<zaman>.txt + .json; metin stdout'a da.
Turlar metinde 1'den, kodda 0'dan; head'ler 0'dan.
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # deneme2/model_y
for _p in (os.path.join(HERE, "train_simplestories"), os.path.join(HERE, "train_fineweb"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402  (pyarrow'dan once: Windows DLL sirasi)
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import internals_y as I  # noqa: E402

TARGETS = ("cur", "prev1", "prev2", "next1", "next2")
MEASURES = ("probes", "attention_passes", "units", "pass_swap", "extras", "repair", "points", "loop", "answers", "selfgen")
_unit = I._unit


# ---- ortak

def _stories_batches(stories, batch, device):
    for x, valid, idx in I._batches(stories, batch, exclude_last=True):
        yield x.to(device), valid.to(device), idx


def _targets(x, valid, lengths):
    """Gecerli konumlarda (B, T) -> duz vektorler: cur x_t, prev1 x_(t-1), prev2 x_(t-2), next1 x_(t+1), next2 x_(t+2),
    pos t; olmayan -1."""
    B, L = x.shape
    T = L - 1
    cur = x[:, :-1]
    t = torch.arange(T, device=x.device)
    none = torch.full((B, 2), -1, dtype=x.dtype, device=x.device)
    prev = torch.cat([none, cur], 1)
    next2 = torch.cat([x[:, 2:], none[:, :1]], 1)
    lens = torch.as_tensor(lengths, device=x.device)[:, None]
    out = dict(cur=cur, prev1=prev[:, 1:1 + T], prev2=prev[:, :T], next1=x[:, 1:],
               next2=torch.where(t[None] + 2 < lens, next2, -1), pos=t.expand(B, T))
    return {k: v[valid] for k, v in out.items()}


def _second_pass(model):
    """(Block sayisi nb, ikinci gecisin turlari)."""
    nb = len(model.blocks)
    return nb, list(range(nb, model.turns))


def _names(model):
    return I._state_names(model)[0]


def _context(ids, pos, decode, before=10, after=3):
    """Konum pos'taki token isaretli kisa metin: '...once [[token]] sonra...'."""
    s = lambda a: decode(a).replace("\n", "⏎")
    return "%s[[%s]]%s" % (s(ids[max(0, pos - before):pos]), s(ids[pos:pos + 1]), s(ids[pos + 1:pos + 1 + after]))


def _r2(sse, sst):
    return float(1 - sse / sst) if sst > 0 else float("nan")


def _output_parts(model):
    """(s, q, u): z = s phi(c), phi(c) = c + q c^2 + (q^2/3 + u) c^3 (OUTPUT_LINK kapaliysa q = u = 0)."""
    q = float(model.link_q) if model.output_link else 0.0
    u = float(model.link_u) if model.output_link else 0.0
    return float(model.log_output_scale.exp()), q, u


# ---- elle ileri hesap (Block.forward'in aynisi) + mudahale + kayit

def _canon_terms(blk, h):
    """Canon-A eki iki parca: k = 0 oz-terim w_0 h_t ve k >= 1 onceki token'lar sum_k w_k h_(t-k)."""
    T = h.shape[-2]
    full = F.pad(h, (0, 0, 3, 0))
    w = blk.canon_weights
    return w[0] * h, sum(w[k] * full[..., 3 - k:3 - k + T, :] for k in range(1, 4))


def _turn_mod(model, t, h, mod=None, ref=None, donor_ok=None, rec=None):
    """Tur t.  mod (mudahale): skip_A; kv = key / value'nun okundugu durum (Canon bu turun Block'uyla); heads {h: 'mean' |
    'resample'}; canon_self / canon_prev 'mean' | 'zero', canon_all 'mean'; A_const / F_const (d,) alt blok ciktisi (norm
    oncesi) sabit; facts 'skip' | 'mean' | 'resample'; alpha_A / alpha_F (d,).  'resample': ayni batch'te bir sonraki
    belgenin (roll 1) ayni konumu, o konum yoksa ortalama.  rec (kayit, ad -> fonksiyon): canon, att_in, attention (acik
    softmax), c, out_A, upd_A, units, out_F, upd_F, state (t, 'A' | 'F', h)."""
    mod, rec = mod or {}, rec or {}
    blk = model.turn_blocks()[t]
    at = blk.attention
    s_term, p_term = _canon_terms(blk, h)
    if "canon" in rec:
        rec["canon"](t, s_term, p_term)
    if mod.get("canon_all") == "mean":
        mod = dict(mod, canon_self="mean", canon_prev="mean")
    terms = dict(self=s_term, prev=p_term)
    for name in terms:
        how = mod.get("canon_" + name)
        if how == "zero":
            terms[name] = torch.zeros_like(h)
        elif how == "mean":
            terms[name] = ref["canon_" + name][t].expand_as(h)
    x = h + terms["self"] + terms["prev"]
    if "att_in" in rec:
        rec["att_in"](t, x)
    if not mod.get("skip_A"):
        src = x if mod.get("kv") is None else mod["kv"] + sum(_canon_terms(blk, mod["kv"]))
        q, k = at.queries_keys(x)
        if mod.get("kv") is not None:
            k = at.queries_keys(src)[1]
        v = (src @ at.W_value.T).unflatten(-1, (at.heads, -1)).transpose(-3, -2)
        if "attention" in rec:
            T = x.shape[-2]
            s = (at.scale * q @ k.transpose(-1, -2)).masked_fill(
                torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
            a = torch.softmax(s, -1)
            rec["attention"](t, a)
            c = a @ v
        else:
            c = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=at.scale)
        if "c" in rec:
            rec["c"](t, c)
        for hh, how in (mod.get("heads") or {}).items():
            m = ref["c"][t][hh]
            c[:, hh] = m if how == "mean" or c.shape[0] == 1 else torch.where(donor_ok[..., None], c[:, hh].roll(1, 0), m)
        added = c.transpose(-3, -2).flatten(-2) @ at.W_context.T
        if "A_const" in mod:
            added = mod["A_const"].expand_as(added)
        if "out_A" in rec:
            rec["out_A"](t, added)
        added = _unit(added)
        if "upd_A" in rec:
            rec["upd_A"](t, added, h)
        h = _unit(h + mod.get("alpha_A", model.alpha_attention[t]) * (added - h))
    if "state" in rec:
        rec["state"](t, "A", h)
    f = I._turn_facts(model, t)
    if f is not None and mod.get("facts") != "skip":
        u = I._units(f, h * h.shape[-1] ** 0.5)
        if "units" in rec:
            rec["units"](t, u)
        out = u @ f.W_fact_out.T
        how = mod.get("facts")
        if how == "mean" or (how == "resample" and out.shape[0] == 1):
            out = ref["out_F"][t].expand_as(out)
        elif how == "resample":
            out = torch.where(donor_ok[..., None], out.roll(1, 0), ref["out_F"][t])
        if "F_const" in mod:
            out = mod["F_const"].expand_as(out)
        if "out_F" in rec:
            rec["out_F"](t, out)
        out = _unit(out)
        if "upd_F" in rec:
            rec["upd_F"](t, out)
        h = _unit(h + mod.get("alpha_F", model.alpha_facts[t]) * (out - h))
    if "state" in rec:
        rec["state"](t, "F", h)
    return h


def _forward(model, inp, rec=None):
    """Temiz ileri hesap -> turlarin girdileri [h0, h1, ..., h_TURNS] (son = cikis durumu)."""
    hs = [model.input_states(inp)]
    for t in range(model.turns):
        hs.append(_turn_mod(model, t, hs[-1], rec=rec))
    return hs


def _states(model, inp):
    """I._state_names sirasiyla durumlar: h0, A1, F1, ... (FactUnits'siz turda F = A)."""
    out = [model.input_states(inp)]
    _forward(model, inp, dict(state=lambda t, kind, h: out.append(h)))
    return out


@torch.no_grad()
def _reference(model, stories, batch):
    """Gecerli konum ortalamalari: Canon k = 0 / k >= 1 terimi, head ciktisi c_h, alt blok ciktilari (norm oncesi)."""
    P = model.tokens.points()
    sums, n = {}, 0

    def add(key, t, v):
        sums.setdefault(key, {})[t] = sums.get(key, {}).get(t, 0) + v

    for x, valid, idx in _stories_batches(stories, batch, P.device):
        _forward(model, x[:, :-1], dict(
            canon=lambda t, s, p: (add("canon_self", t, s[valid].double().sum(0)), add("canon_prev", t, p[valid].double().sum(0))),
            c=lambda t, c: add("c", t, c.permute(0, 2, 1, 3)[valid].double().sum(0)),
            out_A=lambda t, o: add("out_A", t, o[valid].double().sum(0)),
            out_F=lambda t, o: add("out_F", t, o[valid].double().sum(0))))
        n += int(valid.sum())
    return {k: {t: (v / n).to(P.dtype) for t, v in d.items()} for k, d in sums.items()}


@torch.no_grad()
def _ablate_mod(model, test, cases, ref=None, batch=8, on_clean=None):
    """cases: [(ad, aciklama, {tur: mod})]; mod['kv'] fonksiyon olabilir: kv(hs), hs temiz hesabin tur girdileri.
    on_clean(hs, valid): temiz hesap uzerinde ek olcum.  -> taban, satirlar (hikaye duzeyinde eslesmis Δ, I._paired),
    elle ileri hesap ile model.logits farki."""
    P = model.tokens.points()
    S = len(test)
    sums = {k: dict(nll_sum=np.zeros(S), hit_sum=np.zeros(S), count=np.zeros(S)) for k in ["base"] + [c[0] for c in cases]}
    check = 0.0
    for x, valid, idx in _stories_batches(test, batch, P.device):
        inp, y = x[:, :-1], x[:, 1:]
        lens = torch.as_tensor([len(test[i]) - 1 for i in idx], device=P.device)
        donor_ok = (torch.arange(inp.shape[1], device=P.device)[None] < lens[:, None]).roll(1, 0)
        hs = _forward(model, inp)
        check = max(check, float((model.logits(inp) - I._scores(model, hs[-1], P)).abs().max()))
        if on_clean:
            on_clean(hs, valid)

        def put(name, h):
            a, b, c = I._score_rows(I._scores(model, h, P), y, valid)
            sums[name]["nll_sum"][idx], sums[name]["hit_sum"][idx], sums[name]["count"][idx] = a, b, c
        put("base", hs[-1])
        for name, _, mods in cases:
            start = min(mods)
            h = hs[start]
            for t in range(start, model.turns):
                m = mods.get(t)
                if m is not None and callable(m.get("kv")):
                    m = dict(m, kv=m["kv"](hs))
                h = _turn_mod(model, t, h, m, ref, donor_ok)
            put(name, h)
    base = sums["base"]
    N = base["count"].sum()
    rows = [dict(case=name, words=words, nll=float(sums[name]["nll_sum"].sum() / N),
                 acc=float(sums[name]["hit_sum"].sum() / N), **I._paired(base, sums[name])) for name, words, _ in cases]
    return dict(base=dict(nll=float(base["nll_sum"].sum() / N), acc=float(base["hit_sum"].sum() / N), n=int(N)), rows=rows,
                check_logits=check, note=I._DEPENDENCE_NOTE)


def _text_ablation(a, width=52):
    L = ["taban nll %.4f acc %.4f (%d hedef); elle ileri hesap - model.logits en buyuk fark %.1e  (%s)" % (
        a["base"]["nll"], a["base"]["acc"], a["base"]["n"], a["check_logits"], a["note"]),
         "%-16s | %-*s | Δnll ± se          | Δacc (puan)" % ("ad", width, "aciklama")]
    L += ["%-16s | %-*s | %+.4f ± %.4f | %+.2f ± %.2f" % (r["case"], width, r["words"][:width], r["d_nll"], r["se_nll"],
                                                         100 * r["d_acc"], 100 * r["se_acc"]) for r in a["rows"]]
    return L


# ---- 1. probes

@torch.no_grad()
def _collect(model, stories, batch, keep, lens_on=True):
    """keep: durum indeksleri (I._state_names sirasi) -> ([N x d fp16, sqrt(d) olcekli], hedefler, lens: [(nll, hit)])."""
    P = model.tokens.points()
    d = P.shape[1]
    feats, targ, lens = [[] for _ in keep], {}, [[0.0, 0.0] for _ in keep]
    for x, valid, idx in _stories_batches(stories, batch, P.device):
        states = _states(model, x[:, :-1])
        y = x[:, 1:][valid]
        for j, s in enumerate(keep):
            hs = states[s][valid]
            feats[j].append((hs * d ** 0.5).half())
            for c in range(0, len(hs) if lens_on else 0, 4096):    # logit lens, parca parca (V genis)
                z = I._work(I._scores(model, hs[c:c + 4096], P))
                lens[j][0] -= float(torch.log_softmax(z, -1).gather(-1, y[c:c + 4096, None]).sum())
                lens[j][1] += float((z.argmax(-1) == y[c:c + 4096]).sum())
        for k, v in _targets(x, valid, [len(stories[i]) for i in idx]).items():
            targ.setdefault(k, []).append(v)
    return [torch.cat(f) for f in feats], {k: torch.cat(v) for k, v in targ.items()}, lens


@torch.no_grad()
def _probe_score(W, b, X, y):
    """-> (acc, top5, nll)."""
    nll = hit = hit5 = 0.0
    for c in range(0, len(y), 8192):
        z = X[c:c + 8192].float() @ W.T + b
        yc = y[c:c + 8192]
        nll -= float(torch.log_softmax(z, -1).gather(-1, yc[:, None]).sum())
        top = z.topk(5, -1).indices
        hit += float((top[:, 0] == yc).sum())
        hit5 += float((top == yc[:, None]).any(-1).sum())
    return hit / len(y), hit5 / len(y), nll / len(y)


def _softmax_probe(Xf, yf, Xt, yt, V, epochs, lr, bs, seed=0, wd=0.0, early_stop=False):
    """Dogrusal softmax probe z = W x + b (W 0'dan, b fit'teki log siklik); AdamW (wd yalniz W'de), lr dogrusal 0'a.
    early_stop: fit'in son %10'u ayrilir, en iyi epokun W'si kullanilir.  -> test acc, top5, nll, egitim acc (fit'in ilk 50k
    ornegi), secilen epok."""
    mf, mt = yf >= 0, yt >= 0
    Xf, yf, Xt, yt = Xf[mf], yf[mf], Xt[mt], yt[mt]
    Xh = yh = None
    if early_stop:
        cut = int(len(yf) * 0.9)
        Xf, yf, Xh, yh = Xf[:cut], yf[:cut], Xf[cut:], yf[cut:]
    dev, d = Xf.device, Xf.shape[1]
    W = torch.zeros(V, d, device=dev, requires_grad=True)
    prior = torch.bincount(yf, minlength=V).float() + 0.1
    b = (prior / prior.sum()).log().clone().requires_grad_(True)
    opt = torch.optim.AdamW([dict(params=[W], weight_decay=wd), dict(params=[b], weight_decay=0.0)], lr=lr)
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(yf)
    steps, step, best = epochs * math.ceil(n / bs), 0, None
    with torch.enable_grad():
        for epoch in range(epochs):
            perm = torch.randperm(n, device=dev, generator=g)
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    z = Xf[j].float() @ W.T
                loss = F.cross_entropy(z.float() + b, yf[j])
                for gr in opt.param_groups:
                    gr["lr"] = lr * (1 - step / steps)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
            if early_stop:
                ha = _probe_score(W, b, Xh, yh)[0]
                if best is None or ha > best[0]:
                    best = (ha, W.detach().clone(), b.detach().clone(), epoch + 1)
    if best is not None:
        with torch.no_grad():
            W.copy_(best[1])
            b.copy_(best[2])
    acc, top5, nll = _probe_score(W, b, Xt, yt)
    return dict(acc=acc, top5=top5, nll=nll, train_acc=_probe_score(W, b, Xf[:50000], yf[:50000])[0], n_fit=n,
                n_test=len(yt), wd=wd, best_epoch=None if best is None else best[3],
                majority=float((yt == prior.argmax()).double().mean()))


def _ridge(Xf, yf, Xt, yt, lam=1e-3):
    """Ridge (ortalamasi cikarilmis, lam x ortalama kosegen) -> test R^2, tahmin."""
    dev, D = Xf.device, Xf.shape[1]
    mu = sum(Xf[c:c + 65536].double().sum(0) for c in range(0, len(Xf), 65536)) / len(Xf)
    A = torch.zeros(D, D, dtype=torch.float64, device=dev)
    r = torch.zeros(D, dtype=torch.float64, device=dev)
    ym = yf.double().mean()
    for c in range(0, len(Xf), 65536):
        Xc = Xf[c:c + 65536].double() - mu
        A += Xc.T @ Xc
        r += Xc.T @ (yf[c:c + 65536].double() - ym)
    A += lam * A.diagonal().mean() * torch.eye(D, dtype=A.dtype, device=dev)
    w = torch.linalg.solve(A, r)
    pred = torch.cat([(Xt[c:c + 65536].double() - mu) @ w for c in range(0, len(Xt), 65536)]) + ym
    yt = yt.double()
    return _r2(float(((yt - pred) ** 2).sum()), float(((yt - yt.mean()) ** 2).sum())), pred


def probes(model, test, fit, batch=8, epochs=4, lr=5e-3, bs=2048, states=None, targets=TARGETS, wd=0.0, early_stop=False,
           log=print):
    """Her durumda dogrusal okuma.  states: durum adlari (None = hepsi); targets: TARGETS'in alt kumesi."""
    names, blocks = I._state_names(model)
    keep = [i for i, n in enumerate(names) if not (n == "F1" and not model.first_turn_facts) and (states is None or n in states)]
    V = model.tokens.fixed_points.shape[0]
    t0 = time.time()
    Xf, tf, _ = _collect(model, fit, batch, keep, lens_on=False)
    Xt, tt, lens = _collect(model, test, batch, keep)
    n = len(tt["next1"])
    log("durumlar toplandi: fit %d, test %d konum, %d durum (%.0f sn)" % (len(tf["cur"]), n, len(keep), time.time() - t0))
    rows = []
    for j, s in enumerate(keep):
        row = dict(state=names[s], block=blocks[s], lens_acc=lens[j][1] / n, lens_nll=lens[j][0] / n)
        for k in targets:
            row[k] = _softmax_probe(Xf[j], tf[k], Xt[j], tt[k], V, epochs, lr, bs, wd=wd, early_stop=early_stop)
        r2, pred = _ridge(Xf[j], torch.log1p(tf["pos"].double()), Xt[j], torch.log1p(tt["pos"].double()))
        row["pos"] = dict(r2_log=r2, median_abs_err=float((torch.expm1(pred) - tt["pos"].double()).abs().median()))
        rows.append(row)
        log("%-4s %s (lens %.3f) pos R2 %.3f  (%.0f sn)" % (names[s], " ".join(
            "%s %.3f/%.3f" % (k, row[k]["acc"], row[k]["train_acc"]) for k in targets), row["lens_acc"], r2, time.time() - t0))
        Xf[j] = Xt[j] = None                                     # bellek
    return dict(rows=rows, n_test=n, n_fit=len(tf["cur"]), epochs=epochs, lr=lr, batch=bs, targets=list(targets), wd=wd,
                early_stop=early_stop, pos_test_median=float(tt["pos"].double().median()))


def _text_probes(res):
    tg = res["targets"]
    L = ["## dogrusal probe (test acc / egitim acc; probe %d fit konumunda, %d epok, AdamW lr %.0e wd %g%s, batch %d; test %d "
         "konum)" % (res["n_fit"], res["epochs"], res["lr"], res["wd"],
                     ", erken durdurma (fit'in %10'u)" if res["early_stop"] else "", res["batch"], res["n_test"]),
         "durum blok | " + " | ".join("%-11s" % k for k in tg) + " | lens next1 | next1-top5 | pos R2(log) med|err|"]
    for r in res["rows"]:
        L.append("%-5s %s   | %s | %.3f      | %s      | %.3f  %6.0f" % (
            r["state"], r["block"], " | ".join("%.3f/%.3f" % (r[k]["acc"], r[k]["train_acc"]) for k in tg), r["lens_acc"],
            "%.3f" % r["next1"]["top5"] if "next1" in r else "  -  ", r["pos"]["r2_log"], r["pos"]["median_abs_err"]))
    r = res["rows"][0]
    L.append("en sik token tahmini (cogunluk) test acc: " + "  ".join("%s %.3f" % (k, r[k]["majority"]) for k in tg))
    L.append("probe nll (nat): " + " | ".join("%s %s" % (r["state"], " ".join("%.2f" % r[k]["nll"] for k in tg))
                                             for r in res["rows"]))
    L.append("test konum medyani %.0f; pos: ridge log(1+t) -> R^2 ve |exp(tahmin)-1 - t| medyani" % res["pos_test_median"])
    return L


# ---- 2. attention_passes

@torch.no_grad()
def attention_passes(model, stories, batch=4):
    TN, H = model.turns, model.blocks[0].attention.heads
    nb, second = _second_pass(model)
    dv = model.tokens.fixed_points.device
    keys = ("entropy", "distance", "first", "self", "prev", "far", "same", "same_uniform", "induction", "induction_uniform")
    acc = {k: torch.zeros(TN, H, dtype=torch.float64, device=dv) for k in keys}
    cross = {k: torch.zeros(len(second), H, dtype=torch.float64, device=dv) for k in ("q", "k", "v", "c")}
    state_cos = torch.zeros(len(second), 2, dtype=torch.float64, device=dv)   # attention girdisi, tur ciktisi
    n = n_same = n_ind = 0
    for x, valid, idx in _stories_batches(stories, batch, dv):
        inp = x[:, :-1]
        T = inp.shape[1]
        t = torch.arange(T, device=dv)
        lower = t[None, :] < t[:, None]                           # j < t
        same = (inp[:, :, None] == inp[:, None, :]) & lower
        prevtok = torch.cat([torch.full_like(inp[:, :1], -1), inp[:, :-1]], 1)   # anahtar j'nin onceki token'i
        ind = (prevtok[:, None, :] == inp[:, :, None]) & lower
        has_same, has_ind = same.any(-1) & valid, ind.any(-1) & valid
        wq = valid.double()[:, None]
        back = (t[:, None] - t[None, :]).clamp(min=0).float()
        far = (back > 256).float()
        u_same = same.sum(-1).double() / (t + 1).double()         # duzgun dagilimda beklenen kutle
        u_ind = ind.sum(-1).double() / (t + 1).double()
        att_in, out_h, heads_c = {}, {}, {}

        def on_attention(tt, a):
            a = a.float()
            vals = dict(entropy=-(a * torch.log(a.clamp_min(1e-30))).sum(-1), distance=(a * back).sum(-1), first=a[..., 0],
                        self=a.diagonal(0, -2, -1), prev=F.pad(a.diagonal(-1, -2, -1), (1, 0)), far=(a * far).sum(-1))
            for k, v in vals.items():
                acc[k][tt] += (v.double() * wq).sum((0, 2))
            acc["same"][tt] += ((a * same[:, None].float()).sum(-1).double() * has_same[:, None]).sum((0, 2))
            acc["same_uniform"][tt] += (u_same * has_same).sum()
            acc["induction"][tt] += ((a * ind[:, None].float()).sum(-1).double() * has_ind[:, None]).sum((0, 2))
            acc["induction_uniform"][tt] += (u_ind * has_ind).sum()

        _forward(model, inp, dict(att_in=lambda tt, xin: att_in.__setitem__(tt, xin), attention=on_attention,
                                  c=lambda tt, c: heads_c.__setitem__(tt, c),
                                  state=lambda tt, kind, h: out_h.__setitem__(tt, h) if kind == "F" else None))
        for i, t2 in enumerate(second):
            t1 = t2 - nb
            at = model.turn_blocks()[t2].attention
            q1, k1 = at.queries_keys(att_in[t1])
            q2, k2 = at.queries_keys(att_in[t2])
            v1, v2 = ((z @ at.W_value.T).unflatten(-1, (H, -1)).transpose(-3, -2) for z in (att_in[t1], att_in[t2]))
            for name, a, b in (("q", q1, q2), ("k", k1, k2), ("v", v1, v2), ("c", heads_c[t1], heads_c[t2])):
                cross[name][i] += (F.cosine_similarity(a.float(), b.float(), dim=-1).double() * wq).sum((0, 2))
            for j, (a, b) in enumerate(((att_in[t1], att_in[t2]), (out_h[t1], out_h[t2]))):
                state_cos[i, j] += (F.cosine_similarity(a.float(), b.float(), dim=-1).double() * valid).sum()
        n += int(valid.sum())
        n_same += int(has_same.sum())
        n_ind += int(has_ind.sum())
    res = {k: (v / max(n_same if k.startswith("same") else n_ind if k.startswith("induction") else n, 1)).cpu().numpy()
           for k, v in acc.items()}
    return dict(turns=[I._turn_label(model, t) for t in range(TN)], heads=H, nb=nb, second=second, n=n, n_same=n_same,
                n_ind=n_ind, cross={k: (v / n).cpu().numpy() for k, v in cross.items()},
                state_cos=(state_cos / n).cpu().numpy(), **res)


def _head_ablation(run_dir):
    """internals/ablate_*.json'dan (varsa, en yenisi) 'H<tur>.<head>' -> d_nll."""
    import glob
    import json
    files = sorted(glob.glob(os.path.join(run_dir, "internals", "ablate_*.json")))
    if not files:
        return {}, None
    with open(files[-1], encoding="utf-8") as f:
        rows = json.load(f)["result"]["rows"]
    return {r["case"].split()[0]: r["d_nll"] for r in rows}, os.path.basename(files[-1])


def _text_attention_passes(res, ablation=None, source=None):
    H, nb = res["heads"], res["nb"]
    ablation = ablation or {}
    blk = lambda t: res["turns"][t].split(",")[0][-1]
    L = ["## ayni Block iki gecis: ilk gecis tur i | ikinci gecis tur i+%d (%d sorgu; ayni token adayi olan %d, induction "
         "adayi olan %d)" % (nb, res["n"], res["n_same"], res["n_ind"]),
         "same: j < t ve x_j = x_t olan konumlara kutle (adayi olan sorgularda); induction: x_(j-1) = x_t olan j'ye kutle; "
         "(u ...) = duzgun dagilimda beklenen.  far: t - j > 256.  cos q/k/v/c: ayni konumda iki gecis arasi (head icinde).",
         "ablate Δnll: head ortalamayla (%s)" % (source or "yok")]
    hdr = "blok h | entropi    | uzaklik      | konum0      | kendisi     | onceki      | far>256     | same (u)              " \
          "| induction (u)         | cos q  k  v  c       | ablate Δnll"
    for i, t2 in enumerate(res["second"]):
        t1 = t2 - nb
        L += ["", "### Block %s: tur %d | tur %d   (durum kosinusu: attention girdisi %.3f, tur ciktisi %.3f)" % (
            blk(t1), t1 + 1, t2 + 1, res["state_cos"][i][0], res["state_cos"][i][1]), hdr]
        for h in range(H):
            f = lambda k, fmt="%.3f": (fmt + "|" + fmt) % (res[k][t1][h], res[k][t2][h])
            L.append("%s %d  | %s | %s | %s | %s | %s | %s | %s (%.3f|%.3f) | %s (%.3f|%.3f) | %.2f %.2f %.2f %.2f | %s|%s" % (
                blk(t1), h, f("entropy", "%.2f"), f("distance", "%5.1f"), f("first"), f("self"), f("prev"), f("far"),
                f("same"), res["same_uniform"][t1][h], res["same_uniform"][t2][h], f("induction"),
                res["induction_uniform"][t1][h], res["induction_uniform"][t2][h], res["cross"]["q"][i][h],
                res["cross"]["k"][i][h], res["cross"]["v"][i][h], res["cross"]["c"][i][h],
                I._fmt(ablation.get("H%d.%d" % (t1 + 1, h)), "%.3f"), I._fmt(ablation.get("H%d.%d" % (t2 + 1, h)), "%.3f")))
    L += ["", "## tur ortalamalari (head'lerin ortalamasi)",
          "tur  | entropi uzaklik konum0 kendisi onceki far   same  induction"]
    for t in range(len(res["turns"])):
        L.append("%-4s | %.2f  %6.1f  %.3f  %.3f   %.3f  %.3f %.3f %.3f" % (
            res["turns"][t].split(",")[0], res["entropy"][t].mean(), res["distance"][t].mean(), res["first"][t].mean(),
            res["self"][t].mean(), res["prev"][t].mean(), res["far"][t].mean(), res["same"][t].mean(),
            res["induction"][t].mean()))
    return L


# ---- 3. units

@torch.no_grad()
def _update_dirs(model, x, valid, unit_turns):
    """Temiz ileri hesap: guncelleme yonleri {('A'|'F', tur): (M, d)} ve unit_turns'un birimleri {tur: (M, U)}."""
    dirs, units = {}, {}
    _forward(model, x[:, :-1], dict(
        upd_A=lambda t, a, h: dirs.__setitem__(("A", t), a[valid]),
        upd_F=lambda t, o: dirs.__setitem__(("F", t), o[valid]),
        units=lambda t, u: units.__setitem__(t, u[valid]) if t in unit_turns else None))
    return dirs, units


@torch.no_grad()
def units(model, test, fit, decode, batch=8, focus=1, unit_turns=(1, 2, 8, 15), top=12, contexts=8, sample=4096,
          log=print):
    """Guncelleme yonlerinin sabit / token payi (butun turlar), birimlerin token'a bagliligi (unit_turns), odak turun
    birimleri (baglamlar, kumeler)."""
    P = model.tokens.points()
    dv, V, d = P.device, P.shape[0], P.shape[1]
    unit_turns = [t for t in unit_turns if t < model.turns and I._turn_facts(model, t) is not None]
    if focus not in unit_turns:
        unit_turns = [focus] + unit_turns
    U = I._turn_facts(model, focus).W_fact_out.shape[1]
    t0 = time.time()
    # fit: sabit ve token basina ortalamalar
    dsum, tsum, cnt = {}, {}, torch.zeros(V, dtype=torch.float64, device=dv)
    usum = {t: torch.zeros(V, U, dtype=torch.float32, device=dv) for t in unit_turns}
    upsum = {t: torch.zeros(V, U, dtype=torch.float32, device=dv) for t in unit_turns}   # onceki token'a gore
    pcnt = torch.zeros(V, dtype=torch.float64, device=dv)
    ugsum = {t: torch.zeros(U, dtype=torch.float64, device=dv) for t in unit_turns}
    nfit = 0
    for x, valid, idx in _stories_batches(fit, batch, dv):
        tg = _targets(x, valid, [len(fit[i]) for i in idx])
        cur, prev = tg["cur"], tg["prev1"]
        dirs, uu = _update_dirs(model, x, valid, set(unit_turns))
        for k, v in dirs.items():
            dsum[k] = dsum.get(k, 0) + v.double().sum(0)
            if k not in tsum:
                tsum[k] = torch.zeros(V, d, dtype=torch.float32, device=dv)
            tsum[k].index_add_(0, cur, v.float())
        cnt += torch.bincount(cur, minlength=V).double()
        okp = prev >= 0
        pcnt += torch.bincount(prev[okp], minlength=V).double()
        for t, u in uu.items():
            usum[t].index_add_(0, cur, u.float())
            upsum[t].index_add_(0, prev[okp], u[okp].float())
            ugsum[t] += u.double().sum(0)
        nfit += len(cur)
    log("fit: %d konum (%.0f sn)" % (nfit, time.time() - t0))
    dmean = {k: (v / nfit).float() for k, v in dsum.items()}
    seen = cnt > 0
    tmean = {k: torch.where(seen[:, None], v / cnt.clamp(min=1)[:, None].float(), dmean[k]) for k, v in tsum.items()}
    del tsum
    ug = {t: (v / nfit).float() for t, v in ugsum.items()}
    um = {t: torch.where(seen[:, None], usum[t] / cnt.clamp(min=1)[:, None].float(), ug[t]) for t in unit_turns}
    pseen = pcnt > 0
    upm = {t: torch.where(pseen[:, None], upsum[t] / pcnt.clamp(min=1)[:, None].float(), ug[t]) for t in unit_turns}
    del usum, upsum
    # test
    sse = {k: [0.0, 0.0] for k in dmean}                          # (token ortalamasi, sabit) hatasi
    usse = {t: torch.zeros(3, U, dtype=torch.float64, device=dv) for t in unit_turns}   # token, onceki token, sabit
    energy = torch.zeros(U, dtype=torch.float64, device=dv)
    keep_vals, keep_where, samples = [], [], []                   # odak turun birimleri, konumlar (hikaye, t)
    total = sum(len(s) - 2 for s in test)
    pick = torch.sort(torch.randperm(total, generator=torch.Generator().manual_seed(0))[:sample]).values.to(dv)
    offset = 0
    for x, valid, idx in _stories_batches(test, batch, dv):
        tg = _targets(x, valid, [len(test[i]) for i in idx])
        cur, prev = tg["cur"], tg["prev1"]
        dirs, uu = _update_dirs(model, x, valid, set(unit_turns))
        for k, v in dirs.items():
            sse[k][0] += float(((v - tmean[k][cur]) ** 2).sum())
            sse[k][1] += float(((v - dmean[k]) ** 2).sum())
        okp = prev >= 0
        for t, u in uu.items():
            u = u.float()
            usse[t][0] += ((u - um[t][cur]) ** 2).double().sum(0)
            usse[t][1] += ((u[okp] - upm[t][prev[okp]]) ** 2).double().sum(0)
            usse[t][2] += ((u - ug[t]) ** 2).double().sum(0)
        u = uu[focus]
        energy += (u.double() ** 2).sum(0)
        keep_vals.append(u.half())
        keep_where.append(torch.stack([torch.as_tensor(idx, device=dv)[:, None].expand(valid.shape)[valid], tg["pos"]], 1))
        M = len(cur)
        samples.append(u[pick[(pick >= offset) & (pick < offset + M)] - offset].float())
        offset += M
    log("test: %d konum (%.0f sn)" % (offset, time.time() - t0))
    upd = [dict(kind=k[0], turn=k[1], label=I._turn_label(model, k[1]), const=float((dmean[k].double() ** 2).sum()),
                token_r2=_r2(sse[k][0], sse[k][1])) for k in sorted(dmean, key=lambda k: (k[1], k[0]))]
    unit_dep = []
    for t in unit_turns:
        s = usse[t]
        e = s[2]
        unit_dep.append(dict(turn=t, label=I._turn_label(model, t), token_r2=_r2(float(s[0].sum()), float(e.sum())),
                             prev_r2=_r2(float(s[1].sum()), float(e.sum())),
                             unit_token_r2=(1 - s[0] / e.clamp_min(1e-30)).cpu().numpy(),
                             unit_prev_r2=(1 - s[1] / e.clamp_min(1e-30)).cpu().numpy()))
    dep = {r["turn"]: r for r in unit_dep}[focus]
    share = energy / energy.sum()
    order = torch.argsort(share, descending=True)
    vals = torch.cat(keep_vals)
    where = torch.cat(keep_where).cpu().numpy()
    f = I._turn_facts(model, focus)
    frequent = lambda col, fill: torch.where(cnt >= 20, col, torch.full_like(col, fill))   # en az 20 gecen token'lar
    top_units = []
    for i in order[:top].tolist():
        col = vals[:, i].float()
        hi = torch.topk(col, contexts).indices.cpu().numpy()
        lo = torch.topk(-col, max(contexts // 2, 1)).indices.cpu().numpy()
        mt = um[focus][:, i]
        top_units.append(dict(
            unit=i, share=float(share[i]), mean=float(col.mean()), std=float(col.std()),
            pos_frac=float((col > 0).float().mean()), token_r2=float(dep["unit_token_r2"][i]),
            prev_r2=float(dep["unit_prev_r2"][i]),
            high=[dict(value=float(col[j]), text=_context(test[where[j][0]], int(where[j][1]), decode)) for j in hi],
            low=[dict(value=float(col[j]), text=_context(test[where[j][0]], int(where[j][1]), decode)) for j in lo],
            tokens_high=[(decode([k]), float(mt[k]), int(cnt[k])) for k in torch.topk(frequent(mt, -1e9), 6).indices.tolist()],
            tokens_low=[(decode([k]), float(mt[k]), int(cnt[k])) for k in torch.topk(frequent(-mt, -1e9), 4).indices.tolist()]))
    # |r| > 0,9 kumeleri (birlesim-bul)
    S = torch.cat(samples).double()
    S = (S - S.mean(0)) / S.std(0).clamp_min(1e-12)
    C = (S.T @ S / max(len(S) - 1, 1)).abs()
    C.fill_diagonal_(0)
    adj = (C > 0.9).cpu().numpy()
    parent = list(range(U))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(*np.nonzero(np.triu(adj, 1))):
        parent[root(a)] = root(b)
    groups = {}
    for a in range(U):
        if adj[a].any():
            groups.setdefault(root(a), []).append(a)
    Wo, Wi = _unit(f.W_fact_out.T.double()), _unit(f.W_fact_in.double())
    clusters = []
    for g in sorted(groups.values(), key=len, reverse=True):
        gi = torch.as_tensor(g, device=dv)
        off = ~torch.eye(len(g), dtype=torch.bool, device=dv)
        lead = max(g, key=lambda a: float(share[a]))
        clusters.append(dict(units=g, share=float(share[gi].sum()), out_cos=float((Wo[gi] @ Wo[gi].T)[off].mean()),
                             in_cos=float((Wi[gi] @ Wi[gi].T)[off].mean()), token_r2=float(np.mean(dep["unit_token_r2"][g])),
                             lead=lead, lead_tokens=[decode([k]) for k in
                                                     torch.topk(frequent(um[focus][:, lead], -1e9), 6).indices.tolist()]))
    dead = (share < 1e-5).nonzero()[:, 0].tolist()
    cover50 = int((torch.cumsum(share[order], 0) < 0.5).sum()) + 1
    top_idx, rest_idx = order[:cover50].cpu().numpy(), order[cover50:].cpu().numpy()
    log("birimler bitti (%.0f sn)" % (time.time() - t0))
    return dict(updates=upd, unit_dep=[{k: v for k, v in r.items() if not k.startswith("unit_")} for r in unit_dep],
                focus=focus, focus_label=I._turn_label(model, focus), units=U, n=offset, n_fit=nfit, top_units=top_units,
                clusters=clusters, dead=dead, cover50=cover50, top_r2=float(np.mean(dep["unit_token_r2"][top_idx])),
                rest_r2=float(np.median(dep["unit_token_r2"][rest_idx])))


def alpha_report(model, test, batch=8, focus=1):
    """alpha isaretleri (her tur) ve mudahaleler: negatif alpha 0 (tur basina, A ve F), odak turda isaret cevirme ve seyrek
    alpha_F (en buyuk k boyut)."""
    aA, aF = model.alpha_attention.detach().float(), model.alpha_facts.detach().float()
    stats = []
    for t in range(model.turns):
        for kind, a in (("A", aA[t]), ("F", aF[t])):
            if kind == "F" and I._turn_facts(model, t) is None:
                continue
            stats.append(dict(kind=kind, turn=t, label=I._turn_label(model, t), neg=float((a < 0).float().mean()),
                              median=float(a.median()), min=float(a.min()), max=float(a.max()),
                              big=[int((a.abs() > c).sum()) for c in (0.1, 0.5, 1.0)],
                              neg_mass=float(a.clamp(max=0).abs().sum() / a.abs().sum())))
    fa, aa = aF[focus], aA[focus]
    topF = torch.argsort(fa.abs(), descending=True)
    overlap = {k: dict(neg_attention=float((aa[topF[:k]] < 0).float().mean()), mean_alpha_attention=float(aa[topF[:k]].mean()))
               for k in (16, 64, 256)}
    cases = []
    for t in range(model.turns):
        cases.append(("negA%d" % (t + 1), "alpha_attention < 0 -> 0", {t: dict(alpha_A=aA[t].clamp(min=0))}))
        if I._turn_facts(model, t) is not None:
            cases.append(("negF%d" % (t + 1), "alpha_facts < 0 -> 0", {t: dict(alpha_F=aF[t].clamp(min=0))}))
    cases += [("absA%d" % (focus + 1), "alpha_attention -> |alpha|", {focus: dict(alpha_A=aa.abs())}),
              ("posA%d" % (focus + 1), "alpha_attention > 0 -> 0", {focus: dict(alpha_A=aa.clamp(max=0))})]
    for k in (8, 32, 128, 512):
        keep = torch.zeros_like(fa)
        keep[topF[:k]] = fa[topF[:k]]
        cases.append(("topF%d.%d" % (focus + 1, k), "alpha_facts yalniz en buyuk %d boyut" % k, {focus: dict(alpha_F=keep)}))
    return dict(stats=stats, overlap=overlap, ablate=_ablate_mod(model, test, cases, batch=batch), focus=focus)


def _text_units(res, al):
    L = ["## guncelleme yonu norm(u): sabit pay |E norm(u)|^2 ve token'in acikladigi pay (test R^2, ortalamalar %d fit "
         "konumundan; %d test konumu)" % (res["n_fit"], res["n"]),
         "tur                 | A sabit  A token-R2 | F sabit  F token-R2"]
    by = {(r["kind"], r["turn"]): r for r in res["updates"]}
    for t in sorted({r["turn"] for r in res["updates"]}):
        a, f = by.get(("A", t)), by.get(("F", t))
        L.append("%-19s | %.3f    %6.3f    | %s" % (a["label"], a["const"], a["token_r2"],
                                                    "%.3f    %6.3f" % (f["const"], f["token_r2"]) if f else "-"))
    L += ["", "## FactUnits birimleri token'a bagli mi: birim vektoru u'nun test R^2'si (o anki token / onceki token "
          "ortalamasiyla)"]
    L += ["%-19s token R2 %.3f   onceki token R2 %.3f" % (r["label"], r["token_r2"], r["prev_r2"]) for r in res["unit_dep"]]
    L += ["", "## odak: %s FactUnits, %d birim; enerjinin %%50'si %d birimde; olu (pay < 1e-5) %d birim; token R2: bu %d "
          "birimin ortalamasi %.3f, geri kalanlarin medyani %.3f" % (res["focus_label"], res["units"], res["cover50"],
                                                                     len(res["dead"]), res["cover50"], res["top_r2"],
                                                                     res["rest_r2"]),
          "olu birimler: %s" % res["dead"]]
    for u in res["top_units"]:
        L += ["", "### birim %d: enerji payi %.4f, ortalama %.2f, std %.2f, >0 orani %.2f, token R2 %.2f, onceki token R2 %.2f"
              % (u["unit"], u["share"], u["mean"], u["std"], u["pos_frac"], u["token_r2"], u["prev_r2"]),
              "  en cok acan token'lar (ort., sayi): " + "  ".join("%r %.1f (%d)" % t for t in u["tokens_high"]),
              "  en negatif token'lar: " + "  ".join("%r %.1f (%d)" % t for t in u["tokens_low"])]
        L += ["  + %7.2f  %s" % (c["value"], c["text"]) for c in u["high"]]
        L += ["  - %7.2f  %s" % (c["value"], c["text"]) for c in u["low"]]
    L += ["", "## |r| > 0,9 kumeleri (%d kume)" % len(res["clusters"]),
          "boy | enerji payi | W_out cos | W_in cos | token R2 | onde gelen birim: en cok acan token'lar | birimler"]
    for c in res["clusters"][:25]:
        L.append("%3d | %.4f | %.2f | %.2f | %.2f | %d: %s | %s" % (len(c["units"]), c["share"], c["out_cos"], c["in_cos"],
                                                                 c["token_r2"], c["lead"], " ".join(repr(t) for t in c["lead_tokens"]),
                                                                 c["units"][:20]))
    L += ["", "## alpha (tur ve alt blok basina d sayi)",
          "tur                 alt | negatif orani  medyan    min     max  | |a|>0,1 / 0,5 / 1 | negatif kutle payi"]
    for s in al["stats"]:
        L.append("%-19s %s   | %.3f        %6.3f  %6.2f  %6.2f | %4d %4d %4d       | %.3f" % (
            s["label"], s["kind"], s["neg"], s["median"], s["min"], s["max"], s["big"][0], s["big"][1], s["big"][2],
            s["neg_mass"]))
    L.append("odak turda alpha_F'in en buyuk k boyutunda alpha_A: " + "  ".join(
        "k=%d: negatif %.2f, ort %.3f" % (k, v["neg_attention"], v["mean_alpha_attention"]) for k, v in al["overlap"].items()))
    return L + ["", "## alpha mudahaleleri"] + _text_ablation(al["ablate"])


# ---- 4. pass_swap

def pass_swap(model, test, batch=8):
    """Ikinci gecis turlarinda key / value eski durumdan (hs[t] = tur t'nin girdisi, temiz hesap)."""
    nb, second = _second_pass(model)
    first = list(range(1, nb))
    cases = []
    for L in sorted({1, 2, nb // 2, nb} - {0}):
        cases.append(("lag%d_all2" % L, "ikinci gecisin butun turlari: K,V %d tur onceki durumdan" % L,
                      {t: dict(kv=lambda hs, t=t, L=L: hs[t - L]) for t in second}))
    cases.append(("end1_all2", "ikinci gecisin butun turlari: K,V ilk gecisin sonundan (tur %d ciktisi)" % nb,
                  {t: dict(kv=lambda hs: hs[nb]) for t in second}))
    cases.append(("input_all2", "ikinci gecisin butun turlari: K,V girdiden (h0)", {t: dict(kv=lambda hs: hs[0]) for t in second}))
    cases.append(("input_all1", "ilk gecisin turlari 2..%d: K,V girdiden (h0)" % nb, {t: dict(kv=lambda hs: hs[0]) for t in first}))
    cases += [("lag%d_t%d" % (nb, t + 1), "yalniz tur %d: K,V ayni Block'un ilk gecisinden (tur %d girdisi)" % (t + 1, t + 1 - nb),
               {t: dict(kv=lambda hs, t=t: hs[t - nb])}) for t in second]
    cases += [("lag1_t%d" % (t + 1), "yalniz tur %d: K,V bir onceki turun girdisinden" % (t + 1),
               {t: dict(kv=lambda hs, t=t: hs[t - 1])}) for t in second]
    cos, n = np.zeros(model.turns), [0]

    def on_clean(hs, valid):
        for t in second:
            cos[t] += float((F.cosine_similarity(hs[t + 1], hs[t + 1 - nb], dim=-1) * valid).sum())
        n[0] += int(valid.sum())
    res = _ablate_mod(model, test, cases, batch=batch, on_clean=on_clean)
    return dict(res, nb=nb, pass_cos={I._turn_label(model, t): cos[t] / n[0] for t in second})


def _text_pass_swap(res):
    return ["## ikinci gecis attention'i neyi okuyor: K,V eski durumdan, Q guncel",
            "ayni konumda tur ciktisi kosinusu, tur t ile t-%d: " % res["nb"] + "  ".join(
                "%s %.3f" % (k.split(",")[0], v) for k, v in res["pass_cos"].items())] + _text_ablation(res, 72)


# ---- 5. extras

def _extra_cases(model, top_heads):
    """Tur numaralari metinde 1'den.  top_heads {tur: [h1, h2]}: ikili head mudahalesi."""
    TN, H = model.turns, model.blocks[0].attention.heads
    allh = lambda how: {h: how for h in range(H)}
    cases = []
    for t in (1, 5, 7, 8, 9, 12, 15, 16):
        cases += [("Hm%d" % t, "tur %d butun head'ler ortalamayla" % t, {t - 1: dict(heads=allh("mean"))}),
                  ("Hr%d" % t, "tur %d butun head'ler baska belgeden (resample)" % t, {t - 1: dict(heads=allh("resample"))})]
    for t in (2, 3, 8, 13, 15, 16):
        cases += [("Fm%d" % t, "tur %d FactUnits ciktisi ortalamayla" % t, {t - 1: dict(facts="mean")}),
                  ("Fr%d" % t, "tur %d FactUnits ciktisi baska belgeden" % t, {t - 1: dict(facts="resample")})]
    for t in range(1, TN + 1):
        cases += [("CPm%d" % t, "tur %d Canon k>=1 (onceki token'lar) ortalamayla" % t, {t - 1: dict(canon_prev="mean")}),
                  ("CSm%d" % t, "tur %d Canon k=0 (oz-terim) ortalamayla" % t, {t - 1: dict(canon_self="mean")})]
    every = lambda m: {t: dict(m) for t in range(TN)}
    cases += [("CPm", "butun turlarda Canon k>=1 ortalamayla", every(dict(canon_prev="mean"))),
              ("CSm", "butun turlarda Canon k=0 ortalamayla", every(dict(canon_self="mean"))),
              ("CAm", "butun turlarda Canon eki ortalamayla (internals CM)", every(dict(canon_all="mean"))),
              ("CPz", "butun turlarda Canon k>=1 sifir", every(dict(canon_prev="zero"))),
              ("CSz", "butun turlarda Canon k=0 sifir", every(dict(canon_self="zero")))]
    pairs = [("A5+A13", {4: dict(skip_A=True), 12: dict(skip_A=True)}),
             ("Hm5+Hm13", {4: dict(heads=allh("mean")), 12: dict(heads=allh("mean"))}),
             ("Hm5+Hm7", {4: dict(heads=allh("mean")), 6: dict(heads=allh("mean"))}),
             ("CAm5+CAm10", {4: dict(canon_all="mean"), 9: dict(canon_all="mean")}),
             ("A15+A16", {14: dict(skip_A=True), 15: dict(skip_A=True)}),
             ("Hm15+Hm16", {14: dict(heads=allh("mean")), 15: dict(heads=allh("mean"))}),
             ("F15+F16", {14: dict(facts="skip"), 15: dict(facts="skip")}),
             ("Fm15+Fm16", {14: dict(facts="mean"), 15: dict(facts="mean")})]
    for t, hs in top_heads.items():
        pairs.append(("H%d.%d+H%d.%d" % (t, hs[0], t, hs[1]), {t - 1: dict(heads={hs[0]: "mean", hs[1]: "mean"})}))
        pairs += [("H%d.%d" % (t, h), {t - 1: dict(heads={h: "mean"})}) for h in hs]
    cases += [(n, "ikili / tekil (yedeklilik)", m) for n, m in pairs]
    return [c for c in cases if max(c[2]) < TN]                  # kucuk modelde (selftest) olmayan turlar duser


@torch.no_grad()
def extras(model, test, reference, batch=8, focus=1, dims=(382, 616, 10, 758, 132), top_heads=None, log=print):
    """Head ciktisinin konuma gore degisimi (E6), attention guncellemesinin sabit yonu m_t, q.k skor yayilimi, odak turun
    birimlerinin sabit payi, Block E'nin ozel boyutlarini okuyan head'ler (agirlik); ortalama / yeniden ornekleme / ikili
    mudahaleler."""
    P = model.tokens.points()
    dv, TN, H = P.device, model.turns, model.blocks[0].attention.heads
    t0 = time.time()
    ref = _reference(model, reference, batch)
    log("referans ortalamalari (%d belge, %.0f sn)" % (len(reference), time.time() - t0))
    c2 = torch.zeros(TN, H, dtype=torch.float64, device=dv)
    cvec, msum, hsum = {}, {}, {}
    spread = {k: torch.zeros(TN, H, dtype=torch.float64, device=dv) for k in ("std", "gap")}
    U = I._turn_facts(model, focus).W_fact_out.shape[1]
    u1 = torch.zeros(U, dtype=torch.float64, device=dv)
    u2 = torch.zeros(U, dtype=torch.float64, device=dv)
    n = 0
    for x, valid, idx in _stories_batches(test, max(1, batch // 2), dv):
        inp = x[:, :-1]
        T = inp.shape[1]
        wq = valid.double()[:, None]
        causal = torch.ones(T, T, dtype=torch.bool, device=dv).tril()
        cnt = causal.sum(-1).double()

        def on_c(t, c):
            c2[t] += ((c.double() ** 2).sum(-1) * wq).sum((0, 2))
            cvec[t] = cvec.get(t, 0) + c.permute(0, 2, 1, 3)[valid].double().sum(0)

        def on_upd(t, a, h):
            msum[t] = msum.get(t, 0) + a[valid].double().sum(0)
            hsum[t] = hsum.get(t, 0) + h[valid].double().sum(0)

        def on_att_in(t, xin):
            at = model.turn_blocks()[t].attention
            q, k = at.queries_keys(xin)
            s = (at.scale * q @ k.transpose(-1, -2)).float()
            mean = s.masked_fill(~causal, 0).sum(-1) / cnt
            var = ((s - mean[..., None]) ** 2).masked_fill(~causal, 0).sum(-1) / cnt
            mx = s.masked_fill(~causal, float("-inf")).max(-1).values
            spread["std"][t] += (var.sqrt().double() * wq).sum((0, 2))
            spread["gap"][t] += ((mx - mean).double() * wq).sum((0, 2))

        def on_units(t, u):
            if t == focus:
                u1.add_(u[valid].double().sum(0))
                u2.add_((u[valid].double() ** 2).sum(0))

        _forward(model, inp, dict(c=on_c, upd_A=on_upd, att_in=on_att_in, units=on_units))
        n += int(valid.sum())
    log("istatistik gecisi (%.0f sn)" % (time.time() - t0))
    E6 = torch.stack([(1 - ((cvec[t] / n) ** 2).sum(-1) / (c2[t] / n).clamp_min(1e-30)).clamp(min=0).sqrt()
                      for t in range(TN)]).cpu()
    m = torch.stack([msum[t] / n for t in range(TN)])            # (TN, d) ortalama guncelleme yonu
    hm = torch.stack([hsum[t] / n for t in range(TN)])           # (TN, d) attention'a giren ortalama durum
    mn = _unit(m)
    m_tokens = []
    for t in range(TN):
        z = I._scores(model, mn[t].to(P.dtype)[None], P)[0]
        m_tokens.append(dict(len=float(m[t].norm()), cos_mean_state=float((mn[t] * _unit(hm[t])).sum()),
                             top=z.topk(8).indices.tolist(), bottom=(-z).topk(5).indices.tolist()))
    eu, mean2 = u2 / n, (u1 / n) ** 2
    order = torch.argsort(eu, descending=True)
    cover = int((torch.cumsum(eu[order] / eu.sum(), 0) < 0.5).sum()) + 1
    top = order[:cover]
    unit_const = dict(cover50=cover, const_share_top=float(mean2[top].sum() / eu[top].sum()),
                      const_share_all=float(mean2.sum() / eu.sum()),
                      const_share_unit_median_top=float((mean2[top] / eu[top]).median()))
    # Block E (indeks 4): head basina W_query / W_key / W_value sutun boyu ve head icindeki yuzdeligi
    blkE = model.blocks[4 % len(model.blocks)].attention
    dim_rows = []
    for name, W in (("query", blkE.W_query), ("key", blkE.W_key), ("value", blkE.W_value)):
        norms = W.detach().double().unflatten(0, (H, -1)).norm(dim=1)   # (H, d)
        dim_rows += [dict(W=name, dim=dd, norm=norms[:, dd].cpu().numpy(),
                          pct=(norms <= norms[:, dd:dd + 1]).double().mean(1).cpu().numpy()) for dd in dims if dd < norms.shape[1]]
    cases = _extra_cases(model, top_heads or {})
    log("%d mudahale" % len(cases))
    abl = _ablate_mod(model, test, cases, ref, batch)
    log("mudahaleler (%.0f sn)" % (time.time() - t0))
    return dict(turns=[I._turn_label(model, t) for t in range(TN)], heads=H, n=n, E6=E6.numpy(),
                m_cos=(mn @ mn.T).cpu().numpy(), m=m_tokens, spread={k: (v / n).cpu().numpy() for k, v in spread.items()},
                unit_const=unit_const, focus_label=I._turn_label(model, focus), dims=dim_rows, ablate=abl,
                reference=len(reference))


def _text_extras(res, decode):
    TN, H = len(res["turns"]), res["heads"]
    lab = lambda t: res["turns"][t].split(",")[0]
    L = ["## E6: head ciktisinin konuma / belgeye gore degisimi sqrt(E|c_h - ort|^2 / E|c_h|^2) (0 = sabit; %d konum)" % res["n"],
         "tur  | " + " ".join("h%d   " % h for h in range(H))]
    L += ["%-4s | %s" % (lab(t), " ".join("%.3f" % v for v in res["E6"][t])) for t in range(TN)]
    L += ["", "## attention guncellemesinin ortalama yonu m_t = E[norm(W_context c)]: |m| (1 = tamamen sabit), ortalama "
          "girdi durumuyla kosinus, lens'te en yuksek / en dusuk token'lar"]
    L += ["%-4s |m| %.3f  cos(m, E h) %+.3f  ust: %s  alt: %s" % (
        lab(t), r["len"], r["cos_mean_state"], " ".join(repr(decode([i])) for i in r["top"]),
        " ".join(repr(decode([i])) for i in r["bottom"])) for t, r in enumerate(res["m"])]
    L += ["", "m_t'ler arasi kosinus (satir/sutun tur 1..%d)" % TN]
    L += ["%-4s %s" % (lab(t), " ".join("%+.2f" % v for v in res["m_cos"][t])) for t in range(TN)]
    L += ["", "## q.k skor yayilimi (scale dahil, sorgu basina izinli anahtarlar uzerinde): std | max - ortalama",
          "tur  | " + " ".join("h%d std/gap   " % h for h in range(H))]
    L += ["%-4s | %s" % (lab(t), " ".join("%5.2f/%5.2f  " % (res["spread"]["std"][t][h], res["spread"]["gap"][t][h])
                                          for h in range(H))) for t in range(TN)]
    u = res["unit_const"]
    L += ["", "## %s FactUnits: enerjinin %%50'sini tasiyan %d birimde sabit pay sum(E u)^2 / sum E u^2 = %.3f (birim "
          "medyani %.3f); butun birimlerde %.3f" % (res["focus_label"], u["cover50"], u["const_share_top"],
                                                   u["const_share_unit_median_top"], u["const_share_all"]),
          "", "## Block E ozel boyutlari: head basina W sutun boyu (o head'in sutunlari icinde yuzdelik)"]
    L += ["W_%-5s boyut %4d | %s" % (r["W"], r["dim"], " ".join("h%d %.2f(%.2f)" % (h, r["norm"][h], r["pct"][h])
                                                                for h in range(H))) for r in res["dims"]]
    return L + ["", "## mudahaleler (ortalamalar ayri %d belgeden)" % res["reference"]] + _text_ablation(res["ablate"])


# ---- 6. repair: atla / ortalama / optimal sabit (Li & Janson 2024); sink head'lerin belge dagilimi (Guo 2024)

REPAIR_PARTS = (("A", 0), ("A", 4), ("A", 7), ("F", 1), ("F", 15), ("A", 14), ("A", 15))


def _const_key(kind):
    return "A_const" if kind == "A" else "F_const"


def _fit_constant(model, stories, part, init, steps=150, batch=4, lr=0.003, seed=0, log=print):
    """Optimal ablation: alt blok ciktisi yerine kaybi en aza indiren sabit (model donuk), ortalamadan baslar.  Belgelerin
    1/8'i dogrulama: basta (= ortalama) ve her steps/10 adimda olculur; en dusuk dogrulama kayipli sabit doner."""
    kind, t = part
    P = model.tokens.points().detach()
    held = max(1, len(stories) // 8)
    data = list(_stories_batches(stories[held:], batch, P.device))
    check = list(_stories_batches(stories[:held], batch, P.device))
    const = lambda z, tt: {_const_key(kind): z} if tt == t else None

    @torch.no_grad()
    def held_loss(zc):
        tot = cnt = 0.0
        for x, valid, _ in check:
            h = model.input_states(x[:, :-1])
            for tt in range(model.turns):
                h = _turn_mod(model, tt, h, const(zc, tt))
            tot += float(F.cross_entropy(I._scores(model, h, P)[valid].float(), x[:, 1:][valid], reduction="sum"))
            cnt += int(valid.sum())
        return tot / cnt
    z = init.detach().clone().float().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr * float(init.norm()))
    rng = np.random.default_rng(seed)
    best = (held_loss(init.float()), init.detach().clone().float(), 0)
    start_loss = best[0]
    for step in range(steps + 1):
        if step % max(1, steps // 10) == 0 or step == steps:
            hl = held_loss(z.detach())
            if hl < best[0]:
                best = (hl, z.detach().clone(), step)
        if step == steps:
            break
        x, valid, _ = data[rng.integers(len(data))]
        with torch.no_grad():
            h = model.input_states(x[:, :-1])
            for tt in range(t):
                h = _turn_mod(model, tt, h)
        with torch.enable_grad():
            for tt in range(t, model.turns):
                h = _turn_mod(model, tt, h, const(z, tt))
            loss = F.cross_entropy(I._scores(model, h, P)[valid].float(), x[:, 1:][valid])
        for g in opt.param_groups:
            g["lr"] = lr * float(init.norm()) * (1 - step / steps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    log("optimal sabit %s%d: dogrulama kaybi ortalamada %.4f, secilen (adim %d) %.4f" % (
        kind, t + 1, start_loss, best[2], best[0]))
    return best[1], dict(held_mean=start_loss, held_best=best[0], best_step=best[2])


@torch.no_grad()
def _sink_by_document(model, stories, heads, batch=4):
    """Head basina belge basina konum 0 kutlesi (gecerli sorgularda ortalama)."""
    P = model.tokens.points()
    out = {h: np.zeros(len(stories)) for h in heads}
    for x, valid, idx in _stories_batches(stories, batch, P.device):
        def on_attention(t, a):
            for (tt, hh) in heads:
                if tt == t:
                    out[(tt, hh)][idx] = ((a[:, hh, :, 0].double() * valid).sum(-1) / valid.sum(-1).clamp(min=1)).cpu().numpy()
        _forward(model, x[:, :-1], dict(attention=on_attention))
    return out


def repair(model, test, fit, reference, decode, batch=8, steps=150, parts=REPAIR_PARTS, sink_heads=((14, 3), (15, 2)),
           lr=0.003, log=print):
    """Ayni alt bloklarda uc mudahale: atla, ortalama, optimal sabit; sink head'lerin belge basina konum 0 kutlesi."""
    t0 = time.time()
    H = model.blocks[0].attention.heads
    parts = [(k, t) for k, t in parts if t < model.turns and (k == "A" or I._turn_facts(model, t) is not None)]
    sink_heads = tuple((t, h) for t, h in sink_heads if t < model.turns and h < H)
    ref = _reference(model, reference, batch)
    log("ortalamalar (%.0f sn)" % (time.time() - t0))
    cases, fits = [], {}
    for kind, t in parts:
        lab = "%s%d" % (kind, t + 1)
        mean = ref["out_" + kind][t]
        z, fit_info = _fit_constant(model, fit, (kind, t), mean, steps=steps, lr=lr, log=log)
        fits[lab] = dict(fit_info, cos_mean=float(F.cosine_similarity(z, mean.float(), dim=0)))
        cases += [(lab + " atla", "alt blok yok", {t: dict(skip_A=True) if kind == "A" else dict(facts="skip")}),
                  (lab + " ortalama", "cikti ortalama vektor (norm oncesi)", {t: {_const_key(kind): mean}}),
                  (lab + " optimal", "cikti ogrenilen sabit (Li & Janson 2024)", {t: {_const_key(kind): z}})]
    log("sabitler (%.0f sn)" % (time.time() - t0))
    res = _ablate_mod(model, test, cases, batch=batch)
    sink = _sink_by_document(model, test, sink_heads)
    lens = np.array([len(s) - 2 for s in test])
    sink_rows = []
    for (t, h), v in sink.items():
        order = np.argsort(v)
        sink_rows.append(dict(head="%d.%d" % (t + 1, h), quantiles=np.quantile(v, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist(),
                              corr_length=float(np.corrcoef(v, np.log(lens))[0, 1]),
                              low=[(float(v[i]), decode(test[i][1:25])) for i in order[:3]],
                              high=[(float(v[i]), decode(test[i][1:25])) for i in order[-3:]]))
    log("bitti (%.0f sn)" % (time.time() - t0))
    return dict(ablate=res, fits=fits, sink=sink_rows, steps=steps, fit_docs=len(fit), reference=len(reference))


def _text_repair(res):
    L = ["## mudahale turleri yan yana (optimal sabit %d belgede %d adim; ortalama %d ayri belgeden)" % (
        res["fit_docs"], res["steps"], res["reference"])] + _text_ablation(res["ablate"], 48)
    L += ["", "optimal sabit: dogrulama kaybi ortalamada -> secilen (adim), ortalamayla kosinus: " + "  ".join(
        "%s %.4f->%.4f (%d) cos %.2f" % (k, v["held_mean"], v["held_best"], v["best_step"], v["cos_mean"])
        for k, v in res["fits"].items())]
    L += ["", "## sink head'ler: belge basina konum 0 kutlesi (Guo 2024 active-dormant sorusu)"]
    for r in res["sink"]:
        L.append("head %s: min/%%10/%%25/medyan/%%75/%%90/max %s; log(boy) ile korelasyon %.2f" % (
            r["head"], " ".join("%.3f" % q for q in r["quantiles"]), r["corr_length"]))
        L += ["   en dusuk %.3f: %r" % (v, s.replace("\n", " ")[:90]) for v, s in r["low"]]
        L += ["   en yuksek %.3f: %r" % (v, s.replace("\n", " ")[:90]) for v, s in r["high"]]
    return L


# ---- 7. points: cikis noktalari, girdi tablosu, cikis olcegi, arka plan tabani (12_matematik O2 / O3)

FREQ_BANDS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1.0)


@torch.no_grad()
def points(model, counts, test, batch=8):
    """O2: aci(P_j, PF_j), |PF_j + Δ_j|, siklik bantlari; s, q, u, s phi(1); seyrek kume R (f < 1e-6) icin log Σ_R e^z ile
    ln|R|.  O3: girdi tablosu E'de kappa_j = mean_k |E_jk| sqrt(d) / |E_j| ve cos(E_j, PF_j)."""
    tok = model.tokens
    PF, D = tok.fixed_points.double(), tok.shift.double()
    raw = PF + D
    P = F.normalize(raw, dim=-1)
    ang = torch.rad2deg(2 * torch.asin(((P - PF).norm(dim=-1) / 2).clamp(max=1))).cpu().numpy()
    rawn = raw.norm(dim=-1).cpu().numpy()
    E = model.input_embedding.detach().double()
    kappa = (E.abs().mean(-1) * E.shape[1] ** 0.5 / E.norm(dim=-1)).cpu().numpy()   # isaret vektoru 1, Gauss ~0,80
    ecos = F.cosine_similarity(E, PF, dim=-1).cpu().numpy()
    f = counts / counts.sum()
    rows = []
    for lo, hi in zip(FREQ_BANDS[:-1], FREQ_BANDS[1:]):
        sel = (f == 0) if lo == 0 else (f >= lo) & (f < hi)
        if sel.sum():
            rows.append(dict(band="gorulmemis (f = 0)" if lo == 0 else "%.0e <= f < %.0e" % (lo, hi), n=int(sel.sum()),
                             angle=np.quantile(ang[sel], [0.5, 0.9, 1]).tolist(),
                             raw_norm=np.quantile(rawn[sel], [0.1, 0.5, 0.9]).tolist(),
                             kappa=np.quantile(kappa[sel], [0.1, 0.5, 0.9]).tolist(),
                             ecos=np.quantile(ecos[sel], [0.1, 0.5, 0.9]).tolist()))
    sel = f > 0
    corr = dict(angle_logf=float(np.corrcoef(ang[sel], np.log(f[sel]))[0, 1]),
                rawnorm_logf=float(np.corrcoef(rawn[sel], np.log(f[sel]))[0, 1]))
    s, q, u = _output_parts(model)
    dev = tok.shift.device
    R = torch.as_tensor(f < 1e-6, device=dev)
    R_seen = torch.as_tensor((f < 1e-6) & (f > 0), device=dev)
    acc = dict(lse_R=0.0, lse_Rseen=0.0, lse_all=0.0, z_target=0.0, p_R=0.0)
    n = 0
    for x, valid, idx in _stories_batches(test, batch, dev):
        z = I._work(model.logits(x[:, :-1]))[valid]
        y = x[:, 1:][valid]
        lse, lR = torch.logsumexp(z, -1), torch.logsumexp(z[:, R], -1)
        acc["lse_R"] += float(lR.sum())
        acc["lse_Rseen"] += float(torch.logsumexp(z[:, R_seen], -1).sum())
        acc["lse_all"] += float(lse.sum())
        acc["z_target"] += float(z.gather(-1, y[:, None]).sum())
        acc["p_R"] += float((lR - lse).exp().sum())
        n += len(y)
    rival = {k: v / n for k, v in acc.items()}
    rival.update(ln_R=math.log(int(R.sum())), ln_Rseen=math.log(max(int(R_seen.sum()), 1)), R=int(R.sum()),
                 R_seen=int(R_seen.sum()), n=n)
    return dict(bands=rows, corr=corr, s=s, q=q, u=u, s_phi1=s * (1 + q + q * q / 3 + u), s_init=model.scale, rival=rival,
                tokens_counted=int(counts.sum()))


def _text_points(res):
    L = ["## O2 / O3: token bantlari (siklik %d egitim token'indan)" % res["tokens_counted"],
         "bant                 |     n | aci(P,PF) med/%90/max | |PF+Δ| %10/med/%90 | E: κ %10/med/%90 | E: cos(E,PF) %10/med/%90"]
    L += ["%-20s | %5d | %5.2f %5.2f %6.2f     | %.3f %.3f %.3f     | %.3f %.3f %.3f  | %+.3f %+.3f %+.3f" % (
        r["band"], r["n"], *r["angle"], *r["raw_norm"], *r["kappa"], *r["ecos"]) for r in res["bands"]]
    c, r = res["corr"], res["rival"]
    return L + ["korelasyon (gorulen token'lar): aci ~ log f %.3f, |PF+Δ| ~ log f %.3f" % (c["angle_logf"], c["rawnorm_logf"]),
                "", "## cikis olcegi: s = e^tau %.3f (baslangic %.3f), q %.4f, u %.4f, s·φ(1) %.3f" % (
                    res["s"], res["s_init"], res["q"], res["u"], res["s_phi1"]),
                "## arka plan tabani (%d test konumu): R = f < 1e-6 (%d token; %d'i gorulmus)" % (r["n"], r["R"], r["R_seen"]),
                "ortalama log Σ_R e^z %.3f (ln|R| %.3f) | gorulmus R'de %.3f (ln %.3f) | log Σ_hepsi e^z %.3f | hedef z %.3f | "
                "p(R) ortalama %.4f" % (r["lse_R"], r["ln_R"], r["lse_Rseen"], r["ln_Rseen"], r["lse_all"], r["z_target"],
                                        r["p_R"])]


# ---- 8. loop (O5 O6 O8 O9) ve answers (O10 O11): 13_dongu_faz0_matematik §5

PLATEAU = (0.30, 0.45)


def _induction_pred(seq):
    """Konum t icin: x_t'nin en son onceki gecisinin devami (yoksa -1)."""
    last, out = {}, []
    for t, x in enumerate(seq):
        out.append(seq[last[x] + 1] if x in last else -1)
        last[x] = t
    return out


@torch.no_grad()
def _target_geometry(model, stories, batch):
    """Gecerli konumlarda: c_y, c_top, dogru mu, induction sinifi (0 yok, 1 tahmin = hedef, 2 tahmin != hedef) ve dL/du'nun
    konum payi s (E_p c^3 - c_y^3)."""
    P = model.tokens.points()
    s = _output_parts(model)[0]
    out = {k: [] for k in ("cy", "ctop", "hit", "ind", "gu")}
    for x, valid, idx in _stories_batches(stories, batch, P.device):
        inp, y = x[:, :-1], x[:, 1:]
        T = inp.shape[1]
        h = model.hidden(inp)[-1][valid]
        yv = y[valid]
        pred = torch.tensor([(_induction_pred(stories[i][:T]) + [-1] * T)[:T] for i in idx], device=P.device)[valid]
        for c0 in range(0, len(h), 2048):
            hh, yy, pp = h[c0:c0 + 2048], yv[c0:c0 + 2048], pred[c0:c0 + 2048]
            c = (hh @ P.T).float()
            z = I._work(I._scores(model, hh, P))
            p = torch.softmax(z, -1)
            cy = c.gather(-1, yy[:, None])[:, 0]
            top = z.argmax(-1)
            out["cy"].append(cy.cpu())
            out["ctop"].append(c.gather(-1, top[:, None])[:, 0].cpu())
            out["hit"].append((top == yy).cpu())
            out["ind"].append(torch.where(pp < 0, 0, torch.where(pp == yy, 1, 2)).cpu())
            out["gu"].append((s * ((p * c ** 3).sum(-1) - cy ** 3)).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def _hist(v, lo=-0.2, hi=1.0, w=0.02):
    return np.histogram(np.clip(v, lo, hi - 1e-9), np.arange(lo, hi + 1e-9, w))[0].tolist()


def _cos_classes(g):
    """O5 / O6 siniflari: induction (yok / tahmin = hedef / tahmin != hedef) x top-1 (dogru / yanlis)."""
    names = {0: "induction yok", 1: "induction = hedef", 2: "induction != hedef"}
    in_plateau = lambda v: float(((v >= PLATEAU[0]) & (v <= PLATEAU[1])).mean())
    gu_total = float(g["gu"].sum())
    rows = []
    for k in (0, 1, 2):
        for hit in (True, False):
            m = (g["ind"] == k) & (g["hit"] == hit)
            if not m.any():
                continue
            cy, ct, gu = g["cy"][m], g["ctop"][m], g["gu"][m]
            rows.append(dict(cls=names[k], top1="dogru" if hit else "yanlis", n=int(m.sum()),
                             cy_q=np.quantile(cy, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(), plateau=in_plateau(cy),
                             above055=float((cy > 0.55).mean()), ctop_median=float(np.median(ct)), ctop_plateau=in_plateau(ct),
                             gu_sum=float(gu.sum()), gu_mean=float(gu.mean()), gu_se=float(gu.std() / math.sqrt(m.sum())),
                             hist=_hist(cy)))
    return rows, gu_total


def _c_from_p(p, s, q, u, rival_z=25.7):
    """'Tek etkin rakip, plato duzeyinde z = rival_z' varsayimiyla p'den c (13 §2d): s phi(c) = rival_z + logit(p)."""
    target = rival_z + math.log(p / max(1 - p, 1e-12))
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if s * (mid + q * mid ** 2 + (q * q / 3 + u) * mid ** 3) < target else (lo, mid)
    return (lo + hi) / 2


@torch.no_grad()
def _repeat_geometry(model, seqs, period):
    """seqs: [eot] + parca x k.  O8: her durumda gecis k ile k+1 arasi ayni konumda kosinus; O9: gecis basina hedefin c_y'si,
    log p_y, en guclu rakibin c'si, rakiplerin log toplami (z_y haric)."""
    P = model.tokens.points()
    s, q, u = _output_parts(model)
    names = _names(model)
    K = (len(seqs[0]) - 1) // period
    cosk = np.zeros((len(names), K - 1))
    o9 = {k: dict(cy=[], lp=[], crival=[], lse_rival=[], zy=[]) for k in range(K)}
    for seq in seqs:
        x = torch.tensor([seq], device=P.device)
        states = _states(model, x[:, :-1])
        for j, st in enumerate(states):
            h = st[0]
            for k in range(K - 1):
                a, b = h[1 + k * period:1 + (k + 1) * period], h[1 + (k + 1) * period:1 + (k + 2) * period]
                m = min(len(a), len(b))                         # son gecis girdide bir token kisa
                cosk[j, k] += float(F.cosine_similarity(a[:m], b[:m], dim=-1).mean())
        h, y = states[-1][0], x[0, 1:]
        c = (h @ P.T).float()
        z = I._work(I._scores(model, h, P))
        lp = torch.log_softmax(z, -1).gather(-1, y[:, None])[:, 0]
        cy, zy = c.gather(-1, y[:, None])[:, 0], z.gather(-1, y[:, None])[:, 0]
        zr, cr = z.scatter(-1, y[:, None], float("-inf")), c.scatter(-1, y[:, None], float("-inf"))
        for k in range(K):
            sl = slice(k * period + 1, (k + 1) * period)       # gecisin ilk hedefi (parca siniri) haric
            for key, v in (("cy", cy), ("lp", lp), ("crival", cr.max(-1).values), ("lse_rival", torch.logsumexp(zr, -1)),
                           ("zy", zy)):
                o9[k][key].append(v[sl].cpu().numpy())
    cosk /= len(seqs)
    o9r = []
    for k in range(K):
        d = {key: np.concatenate(v) for key, v in o9[k].items()}
        p = np.exp(d["lp"])
        o9r.append(dict(pass_=k + 1, n=len(p), p_mean=float(p.mean()), cy_median=float(np.median(d["cy"])),
                        cy_mean=float(d["cy"].mean()), cy_se=float(d["cy"].std() / math.sqrt(len(p))),
                        crival_median=float(np.median(d["crival"])), lse_rival_median=float(np.median(d["lse_rival"])),
                        zy_median=float(np.median(d["zy"])), cy_pred_single_rival=_c_from_p(float(np.median(p)), s, q, u)))
    ratio = (1 - cosk[:, 1:]) / np.clip(1 - cosk[:, :-1], 1e-9, None)
    return dict(states=names, cos=cosk, ratio=ratio, o9=o9r, sequences=len(seqs), period=period, passes=K)


def loop_geometry(model, test, sentences, randoms, batch=8, log=print):
    t0 = time.time()
    s, q, u = _output_parts(model)
    g = _target_geometry(model, test, batch)
    rows, gu_total = _cos_classes(g)
    log("O5/O6 (%d konum, %.0f sn)" % (len(g["cy"]), time.time() - t0))
    rep = {name: _repeat_geometry(model, seqs, period)
           for name, (seqs, period) in (("cumle", sentences), ("rastgele", randoms)) if seqs}
    log("O8/O9 (%.0f sn)" % (time.time() - t0))
    return dict(s=s, q=q, u=u, dead_point=(-1 / q if q < 0 and u == 0 else None), n=len(g["cy"]), classes=rows,
                gu_total=gu_total, gu_mean=float(g["gu"].mean()), gu_se=float(g["gu"].std() / math.sqrt(len(g["gu"]))),
                hist_all=_hist(g["cy"]), repeat=rep)


def _text_loop(res):
    total = res["gu_total"] or 1.0
    L = ["## O5 / O6: hedef kosinusu c_y = <h_son, P_y> (%d test konumu); s %.1f q %.3f u %.4f; phi'nin olu noktasi -1/q (u = 0) "
         "%.3f; plato %.2f-%.2f" % (res["n"], res["s"], res["q"], res["u"], res["dead_point"] or float("nan"), *PLATEAU),
         "sinif                 top1   |      n | c_y %10 %25 med %75 %90       | platoda | >0,55 | c_top med (platoda) | "
         "∂L/∂u payi (ort ± se)"]
    L += ["%-21s %-6s | %6d | %s | %.3f   | %.3f | %.3f (%.3f)       | %+.3f (%+.3f ± %.3f)" % (
        r["cls"], r["top1"], r["n"], " ".join("%.3f" % v for v in r["cy_q"]), r["plateau"], r["above055"],
        r["ctop_median"], r["ctop_plateau"], r["gu_sum"] / total, r["gu_mean"], r["gu_se"]) for r in res["classes"]]
    L.append("∂L/∂u toplam (konum ortalamasi) %+.4f ± %.4f; pozitif = tahmin kutlesi hedeften yuksek c^3'te (u'yu sifira "
             "iter)" % (res["gu_mean"], res["gu_se"]))
    L.append("c_y histogrami (butun konumlar, -0,2..1,0, 0,02'lik): " + " ".join(str(v) for v in res["hist_all"]))
    for name, r in res["repeat"].items():
        L += ["", "## O8: tekrar (%s, %d dizi, parca %d token, %d gecis): ayni konumda gecis k ile k+1 arasi kosinus; "
              "buzulme orani (1-cos_{k+1})/(1-cos_k)" % (name, r["sequences"], r["period"], r["passes"]),
              "durum | " + " ".join("cos k%d-%d" % (k + 1, k + 2) for k in range(r["passes"] - 1)) + " | "
              + " ".join("oran%d" % (k + 1) for k in range(r["passes"] - 2))]
        last_a = "A%d" % ((len(r["states"]) - 1) // 2)
        L += ["%-5s | %s | %s" % (nm, " ".join("%8.4f" % v for v in r["cos"][j]), " ".join("%6.3f" % v for v in r["ratio"][j]))
              for j, nm in enumerate(r["states"]) if not (nm in ("h0", "F1") or nm.startswith("A") and nm not in ("A1", last_a))]
        L += ["", "## O9: gecis basina kopyalanan token (%s); tek rakip varsayimiyla p'den tahmin edilen c ile yan yana" % name,
              "gecis |    n | p ort | c_y med (ort ± se) | tek-rakip c tahmini | en guclu rakip c med | z_y med | "
              "log Σ rakip e^z med"]
        L += ["%5d | %4d | %.3f | %.3f (%.3f ± %.3f) | %.3f               | %.3f                | %6.2f  | %6.2f" % (
            o["pass_"], o["n"], o["p_mean"], o["cy_median"], o["cy_mean"], o["cy_se"], o["cy_pred_single_rival"],
            o["crival_median"], o["zy_median"], o["lse_rival_median"]) for o in r["o9"]]
    return L


@torch.no_grad()
def _last_states(model, prefixes, wanted, batch=32):
    """Istemlerin son konumunda istenen durumlar -> {ad: (N, d)}."""
    P = model.tokens.points()
    names = _names(model)
    out = {w: [] for w in wanted}
    for i in range(0, len(prefixes), batch):
        part = prefixes[i:i + batch]
        x = torch.zeros(len(part), max(len(p) for p in part), dtype=torch.long, device=P.device)
        for r, pr in enumerate(part):
            x[r, :len(pr)] = torch.tensor(pr, device=P.device)
        states = _states(model, x)
        rows = torch.arange(len(part), device=P.device)
        last = torch.tensor([len(pr) - 1 for pr in part], device=P.device)
        for w in wanted:
            out[w].append(states[names.index(w)][rows, last])
    return {w: torch.cat(v) for w, v in out.items()}


@torch.no_grad()
def answer_geometry(model, items, symbol_items, symbol_tokens, metal_tokens, nonmetal_tokens, log=print):
    """O10: istem sonunda son turun attention'i sonrasi ve FactUnits sonrasi ilk 5 adayin c'si, dogru cevabin c'si ve sirasi.
    O11: element istemlerinde c(Au, Cu, Ag, Fe) ve dogru sembolun c'si; cos(P_Cu, P_Au), metal ortalama yonuyle kosinus."""
    P = model.tokens.points()
    pair = tuple(_names(model)[-2:])
    res = dict(groups={}, symbols={}, pair=pair)
    for group, its in items.items():
        st = _last_states(model, [it["prefix"] for it in its], pair)
        rows = []
        for i, it in enumerate(its):
            r = dict(label=it["label"], right=it["right"])
            for w in pair:
                c = (st[w][i] @ P.T).float()
                top = c.topk(5)
                r[w] = dict(top_c=top.values.tolist(), top_id=top.indices.tolist(), right_c=float(c[it["right"]]),
                            right_rank=int((c > c[it["right"]]).sum()) + 1)
            rows.append(r)
        agg = {}
        for w in pair:
            tc = np.array([r[w]["top_c"] for r in rows])
            agg[w] = dict(top1_median=float(np.median(tc[:, 0])), top5_median=float(np.median(tc[:, 4])),
                          spread_median=float(np.median(tc[:, 0] - tc[:, 4])),
                          top_in_plateau=float(((tc >= PLATEAU[0]) & (tc <= PLATEAU[1])).mean()),
                          right_c_median=float(np.median([r[w]["right_c"] for r in rows])),
                          right_top1=float(np.mean([r[w]["right_rank"] == 1 for r in rows])),
                          right_rank_median=float(np.median([r[w]["right_rank"] for r in rows])))
        res["groups"][group] = dict(n=len(rows), agg=agg, rows=rows)
        log("O10 %s: %d istem" % (group, len(rows)))
    sym = list(symbol_tokens.items())                                # [(' Au', id), ...]
    c_all = (_last_states(model, [it["prefix"] for it in symbol_items], pair[1:])[pair[1]] @ P.T).float()
    by = {}
    for i, it in enumerate(symbol_items):
        d = by.setdefault((it["template"], "altin" if it["name"] == "gold" else "metal" if it["metal"] else "ametal"),
                          dict(right=[], **{k: [] for k, _ in sym}))
        d["right"].append(float(c_all[i, it["right"]]))
        for k, tid in sym:
            d[k].append(float(c_all[i, tid]))
    res["symbols"]["by"] = [dict(template=t, group=g, n=len(d["right"]), **{k: float(np.median(v)) for k, v in d.items()})
                            for (t, g), d in by.items()]
    Pm = P.float()
    mdir, ndir = F.normalize(Pm[metal_tokens].mean(0), dim=0), F.normalize(Pm[nonmetal_tokens].mean(0), dim=0)
    res["symbols"]["geometry"] = dict(
        pair={"%s-%s" % (a, b): float(Pm[ia] @ Pm[ib]) for i, (a, ia) in enumerate(sym) for b, ib in sym[i + 1:]},
        to_metal_mean={k: float(Pm[t] @ mdir) for k, t in sym}, to_nonmetal_mean={k: float(Pm[t] @ ndir) for k, t in sym},
        metal_vs_nonmetal=float(mdir @ ndir), n_metal=len(metal_tokens), n_nonmetal=len(nonmetal_tokens))
    return res


def _text_answers(res, decode):
    A, Fs = res["pair"]
    top3 = lambda r: " ".join("%r %.3f" % (decode([t]), c) for t, c in zip(r["top_id"][:3], r["top_c"][:3]))
    L = ["## O10: istem sonunda ilk 5 adayin c'si, %s (son FactUnits oncesi) ve %s; plato %.2f-%.2f" % (A, Fs, *PLATEAU),
         "grup        n | durum | top1 c med | top5 c med | top1-top5 med | ilk 5'te platoda | dogru c med | dogru top1 | "
         "dogru sira med"]
    L += ["%-10s %3d | %-5s | %.3f      | %.3f      | %.3f         | %.3f            | %.3f       | %.3f      | %.0f" % (
        g, d["n"], w, a["top1_median"], a["top5_median"], a["spread_median"], a["top_in_plateau"], a["right_c_median"],
        a["right_top1"], a["right_rank_median"]) for g, d in res["groups"].items() for w, a in ((w, d["agg"][w]) for w in (A, Fs))]
    if "questions" in res["groups"]:
        L += ["", "### sorular, istem istem (%s -> %s): ilk 3 aday (c) ve dogru cevap" % (A, Fs)]
        L += ["%-48s | %s %s | %s %s | dogru %r %.3f -> %.3f (sira %d -> %d)" % (
            r["label"][:48], A, top3(r[A]), Fs, top3(r[Fs]), decode([r["right"]]), r[A]["right_c"], r[Fs]["right_c"],
            r[A]["right_rank"], r[Fs]["right_rank"]) for r in res["groups"]["questions"]["rows"]]
    s = res["symbols"]
    keys = [k for k in s["by"][0] if k.startswith(" ")] if s["by"] else []
    L += ["", "## O11: element istemleri, %s'da c medyani (sablon x grup)" % Fs,
          "sablon                                              grup    |  n | dogru | " + " | ".join("%5s" % k for k in keys)]
    L += ["%-50s %-7s | %2d | %.3f | %s" % (r["template"][-50:], r["group"], r["n"], r["right"],
                                            " | ".join("%.3f" % r[k] for k in keys)) for r in s["by"]]
    g = s["geometry"]
    return L + ["cikis noktalari: " + "  ".join("cos(%s) %.3f" % (k, v) for k, v in g["pair"].items()),
                "metal ortalama yonu (%d sembol) ile: %s; ametal (%d) ile: %s; metal-ametal yonleri %.3f" % (
                    g["n_metal"], " ".join("%s %.3f" % kv for kv in g["to_metal_mean"].items()), g["n_nonmetal"],
                    " ".join("%s %.3f" % kv for kv in g["to_nonmetal_mean"].items()), g["metal_vs_nonmetal"])]


NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr", "Te", "I", "Xe",
             "Rn", "At", "Ge", "Sb"}


def _answer_inputs(vocab, eot):
    """O10 / O11 istemleri: analyze_errors.QUESTIONS + PROBES, analyze_capacity.freq_facts(), SYMBOL_TEMPLATES x ELEMENTS."""
    import analyze_capacity as AC
    import analyze_errors as AE
    enc = lambda t: AE.encode(t, vocab)
    items = dict(questions=[dict(label=q, prefix=[eot] + enc(q), right=enc(c[0])[0]) for q, c, n, k in AE.QUESTIONS],
                 probes=[dict(label=q, prefix=[eot] + enc(q), right=enc(c[0])[0]) for g, q, c, n in AE.PROBES],
                 facts=[dict(label=f["prompt"], prefix=[eot] + enc(f["prompt"]), right=enc(f["answer"])[0])
                        for f in AC.freq_facts()])
    sym_items = [dict(template=tpl, name=name, metal=sym not in NONMETALS, right=enc(" " + sym)[0],
                      prefix=[eot] + enc(tpl.format(name=name, Name=name[0].upper() + name[1:])))
                 for tpl in AE.SYMBOL_TEMPLATES for name, sym in AC.ELEMENTS]
    symbol_tokens = {k: enc(k)[0] for k in (" Au", " Cu", " Ag", " Fe")}
    metal = sorted({enc(" " + s)[0] for _, s in AC.ELEMENTS if s not in NONMETALS})
    nonmetal = sorted({enc(" " + s)[0] for _, s in AC.ELEMENTS if s in NONMETALS})
    return items, sym_items, symbol_tokens, metal, nonmetal


def _loop_inputs(fw_root, vocab, eot, count=64, repeats=6, length=20, seed=0):
    """Tekrar dizileri: valid cumleleri ilk 20 token'a kirpilir (parca boyu sabit), rastgele token dizileri 20 token.
    [eot] + parca x repeats."""
    import analyze_errors as AE
    sents = [s[:length] for s in AE._sentences(fw_root, vocab, count * 2)[0] if len(s) >= length][:count]
    rng = np.random.default_rng(seed)
    rand = [rng.integers(256, 50000, length).tolist() for _ in range(count)]
    return ([[eot] + s * repeats for s in sents], length), ([[eot] + r * repeats for r in rand], length)


# ---- 9. selfgen (O16, 18_ditto_x_self_matematik §7): kendi uretimi mi gercek metin mi

ELL_BINS = ((1, 1), (2, 2), (3, 3), (4, 7), (8, 15))


def _selfgen_docs(stories, eot, region=256, max_prompt=1024):
    """(belge, istem, gercek devam): istem = belgenin ilk yarisi (<= max_prompt), devam = sonraki region token."""
    out = []
    for i, s in enumerate(stories):
        body = [x for x in s if x != eot]
        P = min(len(body) // 2, max_prompt)
        if P >= 16 and len(body) >= P + region:
            out.append((i, body[:P], body[P:P + region]))
    return out


@torch.no_grad()
def _selfgen_features(model, docs, gens, eot, turns, region, batch=8):
    """Bolgedeki karar konumlari (m_t = 1, l_t bir ELL_BINS kovasinda): istenen durumlar (fp16, sqrt(d) olcekli), sinif
    (0 gercek, 1 kendi), belge, bolge ici konum."""
    import analyze_errors as AE
    P_ = model.tokens.points()
    names = _names(model)
    pick = {t: names.index("A1" if t == 1 else "h0" if t == 0 else "F%d" % t) for t in turns}
    d = P_.shape[1]
    out = {b: dict(X={t: [] for t in turns}, y=[], doc=[], pos=[]) for b in ELL_BINS}
    seqs = [(k, lab, [eot] + prompt + list(cont)) for k, ((i, prompt, real), gen) in enumerate(zip(docs, gens))
            for lab, cont in ((0, real), (1, gen))]
    for c0 in range(0, len(seqs), batch):
        part = seqs[c0:c0 + batch]
        x = torch.full((len(part), max(len(s) for _, _, s in part)), eot, dtype=torch.long, device=P_.device)
        for r, (_, _, s) in enumerate(part):
            x[r, :len(s)] = torch.tensor(s, device=P_.device)
        states = _states(model, x)
        for r, (k, lab, s) in enumerate(part):
            ell, mm, _ = AE._match_trace(np.asarray(s))
            P = len(s) - region                                  # ilk devam token'inin indeksi
            for b in ELL_BINS:
                ts = [t for t in range(P - 1, len(s) - 1) if mm[t] == 1 and b[0] <= ell[t] <= b[1]]
                if ts:
                    idx = torch.tensor(ts, device=P_.device)
                    for t, j in pick.items():
                        out[b]["X"][t].append((states[j][r, idx] * d ** 0.5).half())
                    out[b]["y"] += [lab] * len(ts)
                    out[b]["doc"] += [k] * len(ts)
                    out[b]["pos"] += [t - P for t in ts]
    return out


def _logistic(Xtr, ytr, Xte, yte, steps=300, lr=0.01, wd=1e-3):
    """Ikili lojistik probe (standartlastirilmis girdi, AdamW, tam batch) -> test acc, egitim acc."""
    mu, sd = Xtr.float().mean(0), Xtr.float().std(0).clamp_min(1e-6)
    A, B = (Xtr.float() - mu) / sd, (Xte.float() - mu) / sd
    w = torch.zeros(A.shape[1], device=A.device, requires_grad=True)
    b = torch.zeros((), device=A.device, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=wd)
    with torch.enable_grad():
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(A @ w + b, ytr.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float((((B @ w + b) > 0).long() == yte).float().mean()), float((((A @ w + b) > 0).long() == ytr).float().mean())


def _balanced(idx, y, gen, strata):
    """Siniflari her katmanda (bolge ici konum kovasi) esitle: buyuk siniftan rastgele (tohumlu) alt ornek."""
    out = []
    for g in torch.unique(strata[idx]):
        gi = idx[strata[idx] == g]
        a, b = gi[y[gi] == 0], gi[y[gi] == 1]
        n = min(len(a), len(b))
        out += [a[torch.randperm(len(a), generator=gen)[:n]], b[torch.randperm(len(b), generator=gen)[:n]]]
    return torch.cat(out)


def selfgen_probe(model, stories, eot, turns=tuple(range(0, 13)), region=256, gen_batch=32, train_share=0.7, log=print):
    """O16: ayni l kovasinda (m = 1) acgozlu kendi uretimi ile gercek devam konumlarini tur <= 12 durumlarindan ayirma.
    Belge duzeyinde egitim / test (%70 / %30); siniflar 16'lik bolge konumu kovalarinda esitlenir.  Kontroller: h0 (token
    kimligi) ve yalniz bolge ici konum."""
    import analyze_errors as AE
    t0 = time.time()
    docs = _selfgen_docs(stories, eot, region)
    gens = []
    for c0 in range(0, len(docs), gen_batch):
        gens += AE.generate_batch(model, [[eot] + p for _, p, _ in docs[c0:c0 + gen_batch]], region, eot)[0]
    log("O16: %d belge, acgozlu %d token (%.0f sn)" % (len(docs), region, time.time() - t0))
    feats = _selfgen_features(model, docs, gens, eot, turns, region)
    log("O16: durumlar (%.0f sn)" % (time.time() - t0))
    gen = torch.Generator().manual_seed(0)
    cut = int(len(docs) * train_share)
    rows = []
    for b in ELL_BINS:
        f = feats[b]
        if not f["y"]:
            continue
        y, doc, pos = torch.tensor(f["y"]), torch.tensor(f["doc"]), torch.tensor(f["pos"])
        tr = _balanced(torch.nonzero(doc < cut)[:, 0], y, gen, pos // 16)
        te = _balanced(torch.nonzero(doc >= cut)[:, 0], y, gen, pos // 16)
        row = dict(bin="%d-%d" % b if b[0] != b[1] else "%d" % b[0], n_real=int((y == 0).sum()), n_self=int((y == 1).sum()),
                   n_train=len(tr), n_test=len(te), acc={}, train_acc={})
        if len(tr) >= 20 and len(te) >= 20:
            for t in turns:
                X = torch.cat(f["X"][t])
                dv = X.device
                row["acc"][t], row["train_acc"][t] = _logistic(X[tr.to(dv)], y[tr].to(dv), X[te.to(dv)], y[te].to(dv))
            pf = pos.float()[:, None] / 256
            pf = torch.cat([pf, pf ** 2], 1)
            row["acc_position"] = _logistic(pf[tr], y[tr], pf[te], y[te])[0]
        rows.append(row)
        log("O16 l %s: gercek %d / kendi %d konum; test acc %s" % (row["bin"], row["n_real"], row["n_self"], " ".join(
            "%d:%.3f" % (t, a) for t, a in row["acc"].items())))
    return dict(rows=rows, docs=len(docs), region=region, turns=list(turns), train_docs=cut, test_docs=len(docs) - cut)


def _text_selfgen(res):
    T = res["turns"]
    L = ["## O16: kendi acgozlu uretimi (1) mi gercek devam (0) mi -- m = 1, ayni l kovasi; tur <= 12 durumlarindan "
         "dogrusal (lojistik) probe; belgeler %d egitim / %d test (belge duzeyinde); siniflar 16'lik bolge konumu kovalarinda "
         "esit (sans 0,5)" % (res["train_docs"], res["test_docs"]),
         "durum 0 = h0 (girdi embedding, token kimligi kontrolu); t = tur t ciktisi (1: A1)",
         "l      | gercek / kendi konum | egitim / test | yalniz konum | " + " ".join("t%-4d" % t for t in T)]
    L += ["%-6s | %6d / %6d        | %5d / %5d  | %s        | %s" % (
        r["bin"], r["n_real"], r["n_self"], r["n_train"], r["n_test"],
        "%.3f" % r["acc_position"] if "acc_position" in r else "  -  ",
        " ".join("%.3f" % r["acc"][t] if t in r["acc"] else "  -  " for t in T)) for r in res["rows"]]
    L.append("egitim acc (asiri uyum kontrolu): " + " | ".join("l %s: %s" % (r["bin"], " ".join(
        "%.2f" % r["train_acc"][t] for t in T if t in r["train_acc"])) for r in res["rows"]))
    return L


# ---- CLI

def _decoder(vocab):
    import data_simplestories as DS
    return lambda ids: DS.decode([int(i) for i in ids], vocab)


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="analyze_structure.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="kosu klasoru ya da adi")
    ap.add_argument("measure", nargs="?", choices=MEASURES)
    ap.add_argument("--data", help="FineWeb koku (gpt2/ altinda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", choices=("ema", "last"), default="last")
    ap.add_argument("--checkpoint", type=int, help="checkpoint_tNNNNNN.pt adimi (varsayilan: kosu sonu)")
    ap.add_argument("--stories", type=int, default=128, help="test belgesi (sinav permutasyonu --offset'ten)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--fit-stories", type=int, default=256, help="probes / units / repair: fit belgeleri")
    ap.add_argument("--fit-offset", type=int, default=1000)
    ap.add_argument("--reference", type=int, default=48, help="extras / repair: ortalamalar icin ayri belge (testten sonra)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--states", help="probes: virgullu durum adlari (ornek h0,F2,F8,F16; varsayilan hepsi)")
    ap.add_argument("--targets", help="probes: virgullu hedefler (%s)" % ",".join(TARGETS))
    ap.add_argument("--probe-epochs", type=int, default=4)
    ap.add_argument("--probe-lr", type=float, default=5e-3)
    ap.add_argument("--probe-batch", type=int, default=2048)
    ap.add_argument("--probe-wd", type=float, default=0.0, help="probes: W'de AdamW weight decay")
    ap.add_argument("--probe-early-stop", action="store_true", help="probes: fit'in %%10'unda en iyi epok")
    ap.add_argument("--focus", type=int, default=2, help="units / extras: odak tur (1'den)")
    ap.add_argument("--fit-steps", type=int, default=150, help="repair: optimal sabitin adim sayisi")
    ap.add_argument("--fit-lr", type=float, default=0.003, help="repair: optimal sabitin lr'si (x |ortalama|)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):              # Windows konsolu (cp1254) Δ yazamiyor
        sys.stdout.reconfigure(errors="replace")
    if args.selftest:
        return _selftest()
    assert args.run and args.measure, "kosu ve olcum gerekli"
    say = lambda s: print(s, flush=True)
    t0 = time.time()
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(I._RUNS_ROOT, args.run)
    config = I._config(run_dir)
    data = I._fineweb(config, args.data or I._FINEWEB_ROOT)
    decode = _decoder(data["vocab"])
    rows, test = data["stories"](args.stories, args.offset)
    fit_rows, fit = [], []
    if args.measure in ("probes", "units", "repair"):
        assert args.fit_offset >= args.offset + args.stories or args.fit_offset + args.fit_stories <= args.offset, "ortusme"
        fit_rows, fit = data["stories"](args.fit_stories, args.fit_offset)
    ref = data["stories"](args.reference, args.offset + args.stories)[1] if args.measure in ("extras", "repair") else None
    model = I._load_model(run_dir, args.weights, args.checkpoint).to(args.device)
    ck = "" if args.checkpoint is None else "t%06d" % args.checkpoint
    say("model %s %s yuklendi (%.0f sn); test %d belge, fit %d belge" % (args.weights, ck, time.time() - t0, len(test), len(fit)))
    if args.measure == "probes":
        res = probes(model, test, fit, args.batch, args.probe_epochs, args.probe_lr, args.probe_batch,
                     states=args.states.split(",") if args.states else None,
                     targets=tuple(args.targets.split(",")) if args.targets else TARGETS, wd=args.probe_wd,
                     early_stop=args.probe_early_stop, log=say)
        lines = _text_probes(res)
    elif args.measure == "attention_passes":
        res = attention_passes(model, test, batch=max(1, args.batch // 2))
        lines = _text_attention_passes(res, *_head_ablation(run_dir))
    elif args.measure == "units":
        res = dict(units=units(model, test, fit, decode, batch=args.batch, focus=args.focus - 1, log=say),
                   alpha=alpha_report(model, test, batch=args.batch, focus=args.focus - 1))
        lines = _text_units(res["units"], res["alpha"])
    elif args.measure == "pass_swap":
        res = pass_swap(model, test, batch=args.batch)
        lines = _text_pass_swap(res)
    elif args.measure == "extras":
        res = extras(model, test, ref, batch=args.batch, focus=args.focus - 1, top_heads={5: [7, 5], 7: [5, 7]}, log=say)
        lines = _text_extras(res, decode)
    elif args.measure == "repair":
        res = repair(model, test, fit, ref, decode, batch=args.batch, steps=args.fit_steps, lr=args.fit_lr, log=say)
        lines = _text_repair(res)
    elif args.measure == "points":
        res = points(model, I._token_counts(data, config, run_dir, log=say), test, batch=args.batch)
        lines = _text_points(res)
    elif args.measure == "loop":
        res = loop_geometry(model, test, *_loop_inputs(args.data or I._FINEWEB_ROOT, data["vocab"], data["eos"]),
                            batch=args.batch, log=say)
        lines = _text_loop(res)
    elif args.measure == "answers":
        res = answer_geometry(model, *_answer_inputs(data["vocab"], data["eos"]), log=say)
        lines = _text_answers(res, decode)
    else:
        res = selfgen_probe(model, test, data["eos"], log=say)
        lines = _text_selfgen(res)
    header = ["analyze_structure %s | kosu %s | agirlik %s %s | %s | sure %.0f sn" % (
        args.measure, config.get("name"), args.weights, ck or "(kosu sonu)", time.strftime("%Y-%m-%d %H:%M"), time.time() - t0),
        "test: %d belge (sinav permutasyonu %d..%d); fit: %d belge (%d..)" % (
            len(test), args.offset, args.offset + len(test) - 1, len(fit), args.fit_offset)]
    out_dir = os.environ.get("KUYRUK_SONUC") or os.path.join(run_dir, "analysis")
    name = "_".join(p for p in ("structure", args.measure, args.weights, ck, time.strftime("%Y%m%d_%H%M%S")) if p)
    path = I._write(out_dir, name, header + [""] + lines, dict(measure=args.measure, run=config.get("name"),
                                                               weights=args.weights, checkpoint=args.checkpoint,
                                                               test=np.asarray(rows).tolist(),
                                                               fit=np.asarray(fit_rows).tolist(), result=res))
    print("\n".join(header + [""] + lines))
    say("yazildi: %s.txt / .json" % path)


def _selftest():
    """Kucuk sahte model (bu kosunun ayarlari, kucuk boyut) + rastgele belgeler: butun olcumler hatasiz kosuyor mu, elle
    ileri hesap (SDPA ve acik softmax yolu) modelle ayni mi."""
    from model_y import BlockModel
    torch.manual_seed(0)
    V = 40
    model = BlockModel(V, d=32, turns=4, layers=2, heads=2, units=24, t_max=64, first_turn_facts=False, shared_facts=False,
                       input_embedding=True, rope=True, output_link=True, attention_log_scale=False,
                       rope_base=10000.0).eval().requires_grad_(False)
    with torch.no_grad():                                         # alpha'lar karisik isaretli, Canon acik olsun
        model.alpha_attention.copy_(torch.randn_like(model.alpha_attention) * 0.3)
        model.alpha_facts.copy_(torch.randn_like(model.alpha_facts) * 0.3)
        for b in model.blocks:
            b.canon_weights.copy_(torch.randn_like(b.canon_weights) * 0.1)
    rng = np.random.default_rng(0)
    mk = lambda k: [[0] + rng.integers(1, V, rng.integers(20, 60)).tolist() + [0] for _ in range(k)]
    test, fit = mk(6), mk(10)
    vocab = ["<%d>" % i for i in range(V)]
    decode = lambda ids: "".join(vocab[int(i)] for i in ids)
    show = lambda L, k=6: print("\n".join(L[:k]))
    x = torch.tensor([test[0]])
    ref_logits = model.logits(x)
    for rec in (None, dict(attention=lambda t, a: None)):        # SDPA ve acik softmax yolu
        assert float((I._scores(model, _forward(model, x, rec)[-1]) - ref_logits).abs().max()) < 1e-4
    assert len(_states(model, x)) == len(_names(model))
    r = probes(model, test, fit, batch=4, epochs=8, lr=2e-2, bs=64, log=lambda s: None)
    show(_text_probes(r))
    assert r["rows"][0]["cur"]["acc"] > 0.5, "h0'dan o anki token okunmali"
    r2 = probes(model, test, fit, batch=4, epochs=3, lr=2e-2, bs=64, states=["h0", "F2"], targets=("next1",), wd=0.1,
                early_stop=True, log=lambda s: None)
    assert [r["state"] for r in r2["rows"]] == ["h0", "F2"] and r2["rows"][0]["next1"]["best_epoch"] in (1, 2, 3)
    show(_text_attention_passes(attention_passes(model, test, batch=3)))
    u = units(model, test, fit, decode, batch=4, focus=1, unit_turns=(1, 2), top=2, contexts=2, sample=50, log=lambda s: None)
    show(_text_units(u, alpha_report(model, test, batch=4, focus=1)), 12)
    p = pass_swap(model, test, batch=4)
    show(_text_pass_swap(p), 8)
    assert p["check_logits"] < 1e-4, p["check_logits"]
    e = extras(model, test, fit[:4], batch=4, focus=1, dims=(3, 5), top_heads={2: [0, 1]}, log=lambda s: None)
    show(_text_extras(e, decode)[-6:])
    assert e["ablate"]["check_logits"] < 1e-4
    rp = repair(model, test, fit, fit[:4], decode, batch=4, steps=5, parts=(("A", 0), ("F", 1), ("A", 3)),
                sink_heads=((3, 1),), log=lambda s: None)
    show(_text_repair(rp), 12)
    rows = {r["case"]: r for r in rp["ablate"]["rows"]}
    assert "A1 optimal" in rows and abs(rows["A4 atla"]["d_nll"]) < 10
    assert all(v["held_best"] <= v["held_mean"] + 1e-9 for v in rp["fits"].values())
    cnt = np.zeros(V)
    cnt[:30] = rng.integers(1, 1000, 30)
    show(_text_points(points(model, cnt, test, batch=4)))
    lg = loop_geometry(model, test, ([[0] + [5, 6, 7, 8, 9] * 4], 5), ([[0] + [11, 3, 17, 2, 30] * 4], 5), batch=4,
                       log=lambda s: None)
    show(_text_loop(lg), 12)
    assert lg["repeat"]["cumle"]["cos"].shape == (len(_names(model)), 3)
    its = [dict(label="t%d" % i, prefix=[0] + rng.integers(1, V, 6).tolist(), right=int(rng.integers(1, V))) for i in range(5)]
    sym = [dict(template="T {name}", name=n, metal=m, right=r, prefix=[0, 3, 4 + r]) for n, m, r in
           (("gold", True, 7), ("iron", True, 8), ("neon", False, 9))]
    show(_text_answers(answer_geometry(model, dict(questions=its), sym, {" Au": 7, " Cu": 10}, [7, 8, 10], [9],
                                       log=lambda s: None), decode))
    sg = selfgen_probe(model, mk(60) + [[0] + rng.integers(1, V, 50).tolist() for _ in range(20)], 0, turns=(0, 1, 2),
                       region=8, gen_batch=16, log=lambda s: None)
    show(_text_selfgen(sg))
    print("selftest TAMAM")


if __name__ == "__main__":
    _main()
