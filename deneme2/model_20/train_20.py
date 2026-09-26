# -*- coding: utf-8 -*-
"""train_20 -- model_20 kosulari, CPU, butun veri tek batch, Adam.
Adim 1: ardisik token ciftleri; uc ayar yan yana (sabit, serbest, capali).
Adim 2: butun cumleler dizi olarak; Adim 1 (geriye bakmaz) ile Adim 2 (attention) yan yana.

    python train_20.py            adim 1, secili token'lar     (--all: butun token'lar)
    python train_20.py --step2    adim 2
"""
import sys
import time

import torch

import data_20
from model_20 import CONFIDENCE, BigramModel, SequenceModel, deviation, neighbors, transitions

SETTINGS = {                      # capa lr/wd gibi deneme sayisi; 1e-2 fazla sertti (kayip 1,284 > 1,270)
    "fixed": dict(learn_points=False),
    "free": dict(learn_points=True, anchor=0.0),
    "anchored": dict(learn_points=True, anchor=1e-3),
}
STEPS, LR = 2000, 0.01
LOG_AT = (0, 10, 50, 200, 500, 1000, 2000)
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


def train_seq(setting, ids, mask, n, steps=STEPS, lr=LR, log_at=LOG_AT):
    model = SequenceModel(n, **STEP2[setting])
    # ogrenilenler: Δ (shift), W_next, W_query, W_key, W_context  (PF buffer, listede yok)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    for step in range(steps + 1):
        total, nll = model.loss(ids, mask)                 # butun cumleler, butun konumlar
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), dev=deviation(model).clone(),
                              W_context=model.attention.W_context.norm().item() if model.attention is not None else 0.0))
        if step == steps:
            break
        opt.zero_grad()                                    # onceki adimin gradyanlarini sil
        total.backward()                                   # her ogrenilen sayi x icin ∂kayip/∂x (zincir kurali, otomatik)
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
    ids, mask = pad([q[0] for q in qs], vocab)
    last = mask.sum(1) - 1
    rows = torch.arange(len(qs))
    pred = model.logits(ids)[rows, last].argmax(-1)
    if model.attention is None:
        return pred, None, None
    x = model.tokens.points()[ids]
    return pred, model.attention.weights(x)[rows, last], model.attention(x)[rows, last]


@torch.no_grad()
def answer(model, prompt, vocab, k=2):
    """Acgozlu k token."""
    ids = list(prompt)
    for _ in range(k):
        ids.append(int(model.logits(torch.tensor([ids]))[0, -1].argmax()))
    return " ".join(vocab[i] for i in ids[len(prompt):])


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
    inputs, targets = token_pairs(data)
    results = {}
    for name in SETTINGS:
        t0 = time.time()
        results[name] = train(name, inputs, targets, n)
        print("%s %.1f sn" % (name, time.time() - t0), flush=True)
    show = data["vocab"][2:] if "--all" in sys.argv else SHOW
    print()
    print(report(results, data["vocab"], bigram_floor(inputs, targets, n), len(inputs), data["fingerprint"], show))
