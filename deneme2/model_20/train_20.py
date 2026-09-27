# -*- coding: utf-8 -*-
"""train_20 -- model_20 kosulari, CPU, butun veri tek batch, Adam.
Adim 1: ardisik token ciftleri; uc ayar yan yana (sabit, serbest, capali).
Adim 2: butun cumleler dizi olarak; Adim 1 (geriye bakmaz) ile Adim 2 (attention) yan yana.

    python train_20.py            adim 1, secili token'lar     (--all: butun token'lar)
    python train_20.py --step2    adim 2
    python train_20.py --step3    adim 3 (step2 ile yan yana: shared, separate)
"""
import math
import sys
import time

import torch

import data_20
from model_20 import (CONFIDENCE, COPY_PATH, STREAM_NORM, BigramModel, BlockModel, SequenceModel, deviation, neighbors,
                      transitions)
from model_20_transformer import TransformerModel

SETTINGS = {                      # capa lr/wd gibi deneme sayisi; 1e-2 fazla sertti (kayip 1,284 > 1,270)
    "fixed": dict(learn_points=False),
    "free": dict(learn_points=True, anchor=0.0),
    "anchored": dict(learn_points=True, anchor=1e-3),
}
STEPS, LR = 4000, 0.01  # full batch: 1 adim = 1 epoch.  32 aile icin ilk deger (kullanici, 27 Eylul: "ilk olarak 4.000");
                         # 8 ailede 1000 idi (ezber 600'de tam), ondan once 2000.  Veri buyurse yeniden belirlenir.
LOG_AT = (0, 10, 50, 200, 500, 1000)
LR_FLOOR = 0.1       # cosine decay: lr sonda LR x LR_FLOOR (taban lr/10); train_seq standardi
GRAD_CLIP = 1.0      # gradient clipping: adimdaki gradient'in boyu bunu gecerse buna indirilir; train_seq standardi
WEIGHT_DECAY = 0.0   # weight decay (AdamW), yalniz W_ matrislerine; 0 = kapali (standart).  Deger olculuyor
SHOW = ("Alice", "Tom", "Smith", "Hill", "'s", "father", "aunt", "is", "Who", "?", ".")


def token_pairs(data):
    """Her cumle <eos> ... <eos>; ardisik (girdi, hedef) ciftleri."""
    inputs, targets = [], []
    for s in data["train"]:
        ids = data_20.encode(s, data["vocab"])
        inputs += ids[:-1]
        targets += ids[1:]
    return torch.tensor(inputs), torch.tensor(targets)


def bigram_floor(inputs, targets, n):
    """Yalniz son token'a bakan bir modelin ulasabilecegi en dusuk kayip (sayimlardan)."""
    C = torch.zeros(n, n, dtype=torch.float64)
    C.index_put_((inputs, targets), torch.ones(len(inputs), dtype=torch.float64), accumulate=True)
    p = C / C.sum(1, keepdim=True).clamp_min(1)
    nz = C > 0
    return float(-(C[nz] * p[nz].log()).sum() / C.sum())


def train(setting, inputs, targets, n, steps=STEPS, lr=LR, log_at=LOG_AT):
    model = BigramModel(n, **SETTINGS[setting])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    for step in range(steps + 1):
        total, nll = model.loss(inputs, targets)
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), W_next=model.next.W_next.norm().item(), dev=deviation(model).clone()))
        if step == steps:
            break
        opt.zero_grad()
        total.backward()
        opt.step()
    return model, curve


def report(results, vocab, floor, n_pairs, fingerprint, show):
    ix = {w: i for i, w in enumerate(vocab)}
    rows = [ix[w] for w in show]
    scale = next(iter(results.values()))[0].scale
    out = ["ADIM 1  veri iz %s  cift %d  sozluk %d  scale %.3f (guven %.2f)  esit olasilik %.3f  bigram tavani %.3f" % (
        fingerprint, n_pairs, len(vocab), scale, CONFIDENCE, torch.log(torch.tensor(float(len(vocab)))), floor), ""]
    out.append("ayar       " + "  ".join("%11s" % ("adim %d" % c["step"]) for c in results["fixed"][1]))
    for name, (model, curve) in results.items():
        out.append("%-9s  " % name + "  ".join("%11s" % ("%.3f" % c["nll"]) for c in curve) + "   nll")
        out.append("%-9s  " % "" + "  ".join("%11s" % ("%.1f" % c["W_next"]) for c in curve) + "   |W_next|")
        out.append("%-9s  " % "" + "  ".join("%11s" % ("%.1f°" % float(c["dev"].mean())) for c in curve)
                   + "   ort. sapma")
    for name, (model, curve) in results.items():
        if not model.tokens.learn:
            continue
        out += ["", "%s: NEREDEN NEREYE (sapma = PF ile PL arasi aci)" % name,
                "  %-8s %6s   %-36s %-36s %s" % ("token", "sapma", "PF'de en yakin 3", "PL'de en yakin 3", "egri (adim:aci)")]
        nf, af = neighbors(model.tokens.fixed_points)
        nl, al = neighbors(model.tokens.points())
        for r in rows:
            near = lambda idx, ang: ", ".join("%s %.0f°" % (vocab[j], a) for j, a in zip(idx[r].tolist(), ang[r].tolist()))
            out.append("  %-8s %5.1f°   %-36s %-36s %s" % (
                vocab[r], float(curve[-1]["dev"][r]), near(nf, af), near(nl, al),
                " ".join("%d:%.0f" % (c["step"], float(c["dev"][r])) for c in curve)))
    out += ["", "GECIS (sonraki token, en olasi 5)"]
    for r in rows:
        for name, (model, _) in results.items():
            i, v = transitions(model)
            out.append("  %-8s %-9s %s" % (vocab[r] if name == "fixed" else "", name,
                                           "  ".join("%s %.2f" % (vocab[j], p) for j, p in zip(i[r].tolist(), v[r].tolist()))))
    return "\n".join(out)


# ---- adim 2

STEP2 = {"step1": dict(attention=False), "step2": dict(attention=True)}
STEP3 = {"shared": dict(shared=True), "separate": dict(shared=False)}
PROMPT_1R = {"<eos>": [0], "Who/is": [1, 2], "ozne adi": [3], "soyadi": [4], "'s": [5], "iliski": [6], "? (kendisi)": [7]}


def pad(rows, vocab):
    """Id listeleri -> (ids, mask), sagdan <pad>; nedensel attention'da sagdaki dolgu oncekileri etkilemez."""
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), vocab.index(data_20.PAD), dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        mask[i, :len(r)] = True
    return ids, mask


def sequences(data):
    return pad([data_20.encode(s, data["vocab"]) for s in data["train"]], data["vocab"])


def train_seq(setting, ids, mask, n, steps=STEPS, lr=LR, log_at=LOG_AT, seed=0, lr_floor=LR_FLOOR, grad_clip=GRAD_CLIP,
              weight_decay=WEIGHT_DECAY, device="cpu", every=None, callback=None, compile=False, copy_path=COPY_PATH,
              save_every=None, save=None, checkpoint=None, stream_norm=STREAM_NORM):
    """Standart tarif: cosine decay (LR -> LR x lr_floor) + gradient clipping.  lr_floor=None, grad_clip=None: eski tarif
    (sabit lr); 27 Eylul oncesi kayitli Adim 2-3 sonuclari onunla uretildi.  weight_decay > 0: AdamW, yalniz W_ matrisleri.
    callback(step, model, nll): her `every` adimda, o adimin guncellemesinden ONCE (sinav, kayit, durdurma).
    compile: yalniz kayip hesabi torch.compile ile (model_19'daki gibi); full batch'te sekil sabit, bir kez derlenir.
    copy_path: Oneri A (kopya yolu ve kapisi), yalniz Adim 3 modelinde.  setting "transformer": kiyas modeli
    (model_20_transformer), ayni tarif.
    save(step, model, opt): her save_every adimda, callback'ten sonra, guncellemeden ONCE -- adim s paketi s guncelleme
    gormus modeli ve optimizer'i tasir.  checkpoint {step, model, optimizer}: o adimdan surdurur (ayni steps ve tarifle
    kesintisiz kosuyla bit duzeyinde ayni); o adimin callback'i ve kaydi tekrarlanmaz."""
    assert setting in STEP3 or not copy_path, "copy_path yalniz Adim 3 (BlockModel) icin"
    assert setting in STEP3 or stream_norm, "stream_norm=False yalniz Adim 3 (BlockModel) icin"
    assert not setting.startswith("transformer") or not weight_decay, "transformer icin weight decay gruplari tanimli degil"
    if setting in ("transformer", "transformer_novalue"):   # novalue: V matrisi yok (tek head'de V.O tek matris)
        model = TransformerModel(n, seed=seed, value_matrix=setting == "transformer")
    elif setting in STEP3:
        model = BlockModel(n, seed=seed, copy_path=copy_path, stream_norm=stream_norm, **STEP3[setting])
    else:
        model = SequenceModel(n, seed=seed, **STEP2[setting])
    model = model.to(device)
    ids, mask = ids.to(device), mask.to(device)
    # ogrenilenler: Δ (shift), W_query, W_key, W_context; Adim 1-2: + W_next; Adim 3: + FactUnits (W_next yok)
    # (PF buffer, listede yok)
    params = [p for p in model.parameters() if p.requires_grad]
    if weight_decay:
        # her adimda once W <- W - lr · weight_decay · W (kaybin desteklemedigi agirlik soner), sonra Adam adimi.
        # shift'in capasi var, fact_threshold bir esik: ikisine uygulanmaz
        named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([dict(params=[p for k, p in named if k.split(".")[-1].startswith("W_")], weight_decay=weight_decay),
                                 dict(params=[p for k, p in named if not k.split(".")[-1].startswith("W_")], weight_decay=0.0)],
                                lr=lr)
    else:
        opt = torch.optim.Adam(params, lr=lr)
    first = 0
    if checkpoint is not None:                             # surdurme: agirlik + Adam momentleri + adim
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        first = checkpoint["step"]
    loss_fn = torch.compile(model.loss) if compile else model.loss
    curve = []
    for step in range(first, steps + 1):
        resumed_here = checkpoint is not None and step == first
        if lr_floor is not None:                           # lr_t = lr · (floor + (1 - floor) · (1 + cos(π t / T)) / 2)
            for group in opt.param_groups:
                group["lr"] = lr * (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1))))
        total, nll = loss_fn(ids, mask)                    # butun cumleler, butun konumlar
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), dev=deviation(model).clone() if hasattr(model, "tokens") else None,
                              W_context=(sum(b.attention.W_context.norm().item() for b in model.blocks)
                                         if isinstance(model, BlockModel) else
                                         model.attention.W_context.norm().item() if getattr(model, "attention", None) is not None
                                         else 0.0)))
        if callback is not None and every and step % every == 0 and not resumed_here:
            callback(step, model, nll.item())
        if save is not None and save_every and step > 0 and step % save_every == 0 and not resumed_here:
            save(step, model, opt)
        if step == steps:
            break
        opt.zero_grad()                                    # onceki adimin gradyanlarini sil
        total.backward()                                   # her ogrenilen sayi x icin ∂kayip/∂x (zincir kurali, otomatik)
        if grad_clip is not None:                          # butun gradyanlarin toplam boyu > grad_clip ise olcekle indir
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()                                         # Adam: x <- x - lr · m / (√v + eps); m, v gradyanin
                                                           # yuruyen ortalamasi ve karesininki (her sayi kendi adimini atar)
    return model, curve


def questions(data, cls):
    """(istem id'leri: <eos> + soru, dogru cevaplarin ilk token'lari, sinav satiri)."""
    ix = {w: i for i, w in enumerate(data["vocab"])}
    return [([ix[data_20.EOS]] + [ix[t] for t in e["prompt"]], {ix[a.split()[0]] for a in e["answers"]}, e)
            for e in data["exam"] if e["cls"] == cls]


@torch.no_grad()
def read_questions(model, qs, vocab):
    """'?' konumunda: tahmin edilen ilk token, attention agirliklari ve getirilen c (attention yoksa None)."""
    device = next(model.parameters()).device
    ids, mask = pad([q[0] for q in qs], vocab)
    ids, mask = ids.to(device), mask.to(device)
    last = mask.sum(1) - 1
    rows = torch.arange(len(qs), device=device)
    pred = model.logits(ids)[rows, last].argmax(-1)
    if getattr(model, "attention", None) is None:
        return pred, None, None
    x = model.tokens.points()[ids]
    return pred, model.attention.weights(x)[rows, last], model.attention(x)[rows, last]


@torch.no_grad()
def answer(model, prompt, vocab, k=2):
    """Acgozlu k token."""
    ids = list(prompt)
    for _ in range(k):
        ids.append(int(model.logits(torch.tensor([ids], device=next(model.parameters()).device))[0, -1].argmax()))
    return " ".join(vocab[i] for i in ids[len(prompt):])


# ---- adim 4: ara adimli cevaplar ("<steps>"); cumle 8 token: X1 X2 's r is Y1 Y2 .

@torch.no_grad()
def generate(model, prompts, n):
    """Acgozlu uretim: ayni uzunluktaki istemler birlikte, her birine n token.  -> id listeleri (yalniz uretilen)."""
    device = next(model.parameters()).device
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for idx in groups.values():
        ids = torch.tensor([prompts[i] for i in idx], device=device)
        for _ in range(n):
            ids = torch.cat([ids, model.logits(ids)[:, -1].argmax(-1, keepdim=True)], 1)
        for row, i in zip(ids[:, ids.shape[1] - n:].tolist(), idx):
            out[i] = row
    return out


def score_steps(said, entry):
    """Ara adimli cevap (token listesi, 8 x adim).  Adlar kullanici onayiyla (27 Eylul):
    SC subject correct (cevap sorudaki kisiyle basliyor), BC bridge correct (ilk cumlenin nesnesi = kopru),
    AC answer correct (son cumlenin nesnesi = cevap; ANA OLCU), EX exact (tamami birebir),
    FC format correct (her cumle 'A B 's r_i is C D .', r_i sorudaki)."""
    want, path = entry["steps"], entry["path"]
    form = all(said[8 * i + 2] == "'s" and said[8 * i + 3] == r and said[8 * i + 4] == "is" and said[8 * i + 7] == "."
               for i, r in enumerate(path))
    return dict(SC=said[0:2] == want[0:2], BC=said[5:7] == want[5:7], AC=said[-3:-1] == want[-3:-1], EX=said == want,
                FC=form)


def exam_steps(model, data, cls, given=0):
    """<steps> sinifi: model cevabi yazar; given > 0 ise dogru cevabin ilk given token'i verilir (2: ozne, 8: ilk cumle).
    -> (sayimlar, [(soru, modelin yazdigi, dogrusu)])."""
    vocab = data["vocab"]
    ix = {w: i for i, w in enumerate(vocab)}
    counts = dict(SC=0, BC=0, AC=0, EX=0, FC=0)
    rows = []
    by_hops = {}
    for q in questions(data, cls):
        by_hops.setdefault(len(q[2]["path"]), []).append(q)
    for hops, group in by_hops.items():
        gens = generate(model, [q[0] + [ix[t] for t in q[2]["steps"][:given]] for q in group], 8 * hops - given)
        for q, g in zip(group, gens):
            said = q[2]["steps"][:given] + [vocab[t] for t in g]
            for k, v in score_steps(said, q[2]).items():
                counts[k] += v
            rows.append((data_20.detokenize(q[2]["prompt"]), data_20.detokenize(said), data_20.detokenize(q[2]["steps"])))
    return counts, rows


def report_step2(results, data):
    vocab = data["vocab"]
    ix = {w: i for i, w in enumerate(vocab)}
    first = next(iter(results.values()))[0]
    out = ["ADIM 2  veri iz %s  cumle %d  sozluk %d  scale %.3f  attention olcegi %.3f (T_MAX)" % (
        data["fingerprint"], len(data["train"]), len(vocab), first.scale, results["step2"][0].attention.scale), ""]
    out.append("ayar     " + "  ".join("%10s" % ("adim %d" % c["step"]) for c in results["step1"][1]))
    for name, (model, curve) in results.items():
        out.append("%-7s  " % name + "  ".join("%10s" % ("%.3f" % c["nll"]) for c in curve) + "   nll")
        out.append("%-7s  " % "" + "  ".join("%10s" % ("%.1f°" % float(c["dev"].mean())) for c in curve) + "   ort. sapma")
        if model.attention is not None:
            out.append("%-7s  " % "" + "  ".join("%10s" % ("%.1f" % c["W_context"]) for c in curve) + "   |W_context|")

    out += ["", "ILK TOKEN DOGRULUGU ('?'ten sonra, dogru cevaplardan birinin adi)"]
    out.append("  %-15s %5s   %s" % ("sinif", "soru", "   ".join("%7s" % k for k in results)))
    for cls in ("memory_base", "memory_derived", "chain2", "named2", "chain3", "named3"):
        qs = questions(data, cls)
        accs = []
        for model, _ in results.values():
            pred = read_questions(model, qs, vocab)[0]
            accs.append(sum(int(p) in q[1] for p, q in zip(pred.tolist(), qs)) / len(qs))
        out.append("  %-15s %5d   %s" % (cls, len(qs), "   ".join("%7.3f" % a for a in accs)))

    qs = questions(data, "memory_base")
    out += ["", "1R ILISKIYE GORE (memory_base)"]
    for rel in data_20.BASE:
        sub = [q for q in qs if q[2]["path"] == (rel,)]
        accs = []
        for model, _ in results.values():
            pred = read_questions(model, sub, vocab)[0]
            accs.append(sum(int(p) in q[1] for p, q in zip(pred.tolist(), sub)) / len(sub))
        out.append("  %-9s %4d   %s" % (rel, len(sub), "   ".join("%7.3f" % a for a in accs)))

    model = results["step2"][0]
    pred, a, c = read_questions(model, qs, vocab)
    out += ["", "step2: '?' KONUMUNDA ATTENTION (1R, %d soru ortalamasi)" % len(qs)]
    out.append("  " + "   ".join("%s %.2f" % (k, float(a[:, v].sum(1).mean())) for k, v in PROMPT_1R.items()))
    P = model.tokens.points()
    near = (torch.nn.functional.normalize(c, dim=-1) @ P.T).topk(2, dim=-1).indices
    subj = torch.tensor([q[0][3] for q in qs])
    rel = torch.tensor([q[0][6] for q in qs])
    out.append("  getirilen c'ye en yakin token: ozne adi %.2f, iliski %.2f;  en yakin 2'de ozne adi %.2f, iliski %.2f" % (
        float((near[:, 0] == subj).float().mean()), float((near[:, 0] == rel).float().mean()),
        float((near == subj[:, None]).any(1).float().mean()), float((near == rel[:, None]).any(1).float().mean())))

    out += ["", "NITEL (step2, 1R ve tutulan zincirlerden ornekler; attention'in en cok baktigi 3 konum)"]
    samples = [q for q in qs if q[2]["subject"] in ("Alice Smith", "Owen Evans", "John Smith", "Mia Hill")]
    samples += questions(data, "chain2")[:3] + questions(data, "named2")[:2]
    for prompt, _, e in samples:
        x = model.tokens.points()[torch.tensor([prompt])]
        w = model.attention.weights(x)[0, -1]
        top = w.topk(3)
        out.append("  %-52s -> %-14s dogru: %-28s bakti: %s" % (
            data_20.detokenize(e["prompt"]), answer(model, prompt, vocab), " | ".join(e["answers"]),
            ", ".join("%s %.2f" % (vocab[prompt[j]], float(v)) for j, v in zip(top.indices.tolist(), top.values.tolist()))))
    return "\n".join(out)


def report_step3(results, data):
    vocab = data["vocab"]
    ix = {w: i for i, w in enumerate(vocab)}
    people, rel = data["people"], data["rel"]
    firsts = torch.tensor(sorted({ix[p["first"]] for p in people.values()}))
    m0 = results["shared"][0]
    # kopru okumasi tur 1'in ara durumunu elle kuruyor; kopya yolu katkisini icermez
    assert not any(getattr(m, "copy_path", False) or not getattr(m, "stream_norm", True) for m, _ in results.values()), \
        "report_step3 kopya yolu kapali, akisi normalize edilen modeller icin"
    out = ["ADIM 3  veri iz %s  cumle %d  TURNS %d  FACT_UNITS %d  scale %.3f" % (
        data["fingerprint"], len(data["train"]), m0.turns, m0.blocks[0].facts.W_fact_in.shape[0], m0.scale), ""]
    out.append("ayar           " + "  ".join("%10s" % ("adim %d" % c["step"]) for c in results["step2"][1]))
    for name, (model, curve) in results.items():
        out.append("%-13s  " % name + "  ".join("%10s" % ("%.3f" % c["nll"]) for c in curve) + "   nll")
        out.append("%-13s  " % "" + "  ".join("%10s" % ("%.1f°" % float(c["dev"].mean())) for c in curve) + "   ort. sapma")
        out.append("%-13s  " % "" + "  ".join("%10s" % ("%.1f" % c["W_context"]) for c in curve) + "   |W_context|")

    out += ["", "ILK TOKEN DOGRULUGU ('?'ten sonra, dogru cevaplardan birinin adi)"]
    out.append("  %-15s %5s   %s" % ("sinif", "soru", "  ".join("%13s" % k for k in results)))
    for cls in ("memory_base", "memory_derived", "chain2", "named2", "chain3", "named3"):
        qs = questions(data, cls)
        accs = []
        for model, _ in results.values():
            pred = read_questions(model, qs, vocab)[0]
            accs.append(sum(int(p) in q[1] for p, q in zip(pred.tolist(), qs)) / len(qs))
        out.append("  %-15s %5d   %s" % (cls, len(qs), "  ".join("%13.3f" % a for a in accs)))

    qs1 = questions(data, "memory_base")
    out += ["", "1R ILISKIYE GORE (memory_base)"]
    for r in data_20.BASE:
        sub = [q for q in qs1 if q[2]["path"] == (r,)]
        accs = []
        for model, _ in results.values():
            pred = read_questions(model, sub, vocab)[0]
            accs.append(sum(int(p) in q[1] for p, q in zip(pred.tolist(), sub)) / len(sub))
        out.append("  %-9s %4d   %s" % (r, len(sub), "  ".join("%13.3f" % a for a in accs)))

    # kopru: 2R istemi <eos> Who is X1 X2 's r1 's r2 ?  ->  ozne adi 3, r1 konumu 6, r2 konumu 8, '?' 9
    out += ["", "KOPRU OKUMASI (2R zincir sorulari; kopru = oznenin r1'i; 48 ilk ad icinde en yakin)",
            "  %-13s %-15s %5s  %19s  %19s   %s" % ("ayar", "sinif", "soru", "tur1 durum = kopru", "tur1 olgu = kopru",
                                                 "'?' tur2 bakisi: ozne / r1 / r2")]
    with torch.no_grad():
        for cls in ("memory_derived", "chain2"):
            qs = [q for q in questions(data, cls) if len(q[2]["path"]) == 2]
            ids, _ = pad([q[0] for q in qs], vocab)
            bridge = torch.tensor([ix[people[data_20.follow(rel, q[2]["subject"], q[2]["path"][:1])[0]]["first"]] for q in qs])
            for name, (model, _) in results.items():
                if not isinstance(model, BlockModel):
                    continue
                P = model.tokens.points()
                hs = model.hidden(ids)
                blocks = model.turn_blocks()
                mid = torch.nn.functional.normalize(hs[0] + blocks[0].attention(hs[0]) @ blocks[0].attention.W_context.T, dim=-1)
                fact = torch.nn.functional.normalize(blocks[0].facts(mid)[:, 6], dim=-1)
                state = hs[1][:, 6]
                top_state = firsts[(state @ P[firsts].T).argmax(-1)]
                top_fact = firsts[(fact @ P[firsts].T).argmax(-1)]
                w = blocks[1].attention.weights(hs[1])[:, 9]
                out.append("  %-13s %-15s %5d  %19.2f  %19.2f   %.2f / %.2f / %.2f" % (
                    name, cls, len(qs), float((top_state == bridge).float().mean()), float((top_fact == bridge).float().mean()),
                    float(w[:, 3].mean()), float(w[:, 6].mean()), float(w[:, 8].mean())))

    out += ["", "NITEL (acgozlu 2 token)"]
    samples = [q for q in questions(data, "memory_base") if q[2]["subject"] == "Owen Evans"][:2]
    samples += [q for q in questions(data, "chain2") if q[2]["subject"] == "Owen Evans"][:4]
    samples += [q for q in questions(data, "memory_derived") if q[2]["subject"] == "Alice Smith" and len(q[2]["path"]) == 2][:2]
    for prompt, _, e in samples:
        answers = "  ".join("%s: %-14s" % (name, answer(model, prompt, vocab)) for name, (model, _) in results.items())
        out.append("  %-44s dogru: %-14s %s" % (data_20.detokenize(e["prompt"]), " | ".join(e["answers"]), answers))
    return "\n".join(out)


if __name__ == "__main__":
    torch.set_num_threads(1)
    data = data_20.build()
    data_20.audit(data)
    n = len(data["vocab"])
    if "--step2" in sys.argv:
        ids, mask = sequences(data)
        results = {}
        for name in STEP2:
            t0 = time.time()
            results[name] = train_seq(name, ids, mask, n)
            print("%s %.1f sn" % (name, time.time() - t0), flush=True)
        print()
        print(report_step2(results, data))
        sys.exit(0)
    if "--step3" in sys.argv:
        ids, mask = sequences(data)
        results = {}
        for name in ["step2"] + list(STEP3):
            t0 = time.time()
            results[name] = train_seq(name, ids, mask, n)
            print("%s %.1f sn" % (name, time.time() - t0), flush=True)
        print()
        print(report_step3(results, data))
        sys.exit(0)
    inputs, targets = token_pairs(data)
    results = {}
    for name in SETTINGS:
        t0 = time.time()
        results[name] = train(name, inputs, targets, n)
        print("%s %.1f sn" % (name, time.time() - t0), flush=True)
    show = data["vocab"][2:] if "--all" in sys.argv else SHOW
    print()
    print(report(results, data["vocab"], bigram_floor(inputs, targets, n), len(inputs), data["fingerprint"], show))
