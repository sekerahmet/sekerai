# -*- coding: utf-8 -*-
"""exam_kinship -- akrabalik egitiminin veriye ozgu kismi: dizi haline getirme, sinav (1R_T, 2R_T, 2R_UT; SC BC AC EX FC)
ve Adim 1-3 raporlari.  Model ve genel egitim bir ust klasorde (model_y.py, train_y.py).

    python exam_kinship.py            adim 1, secili token'lar     (--all: butun token'lar)
    python exam_kinship.py --step2    adim 2
    python exam_kinship.py --step3    adim 3 (step2 ile yan yana: shared, separate)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_y: model ve genel egitim

import data_y  # noqa: E402
from model_y import CONFIDENCE, BlockModel, neighbors, transitions  # noqa: E402
from train_y import SETTINGS, STEP2, STEP3, bigram_floor, generate, pad, train, train_seq  # noqa: E402

SHOW = ("Alice", "Tom", "Smith", "Hill", "'s", "father", "aunt", "is", "Who", "?", ".")
PROMPT_1R = {"<eos>": [0], "Who/is": [1, 2], "ozne adi": [3], "soyadi": [4], "'s": [5], "iliski": [6], "? (kendisi)": [7]}


def token_pairs(data):
    """Her cumle <eos> ... <eos>; ardisik (girdi, hedef) ciftleri."""
    inputs, targets = [], []
    for s in data["train"]:
        ids = data_y.encode(s, data["vocab"])
        inputs += ids[:-1]
        targets += ids[1:]
    return torch.tensor(inputs), torch.tensor(targets)


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


def sequences(data):
    return pad([data_y.encode(s, data["vocab"]) for s in data["train"]], data["vocab"])


def questions(data, cls):
    """(istem id'leri: <eos> + soru, dogru cevaplarin ilk token'lari, sinav satiri)."""
    ix = {w: i for i, w in enumerate(data["vocab"])}
    return [([ix[data_y.EOS]] + [ix[t] for t in e["prompt"]], {ix[a.split()[0]] for a in e["answers"]}, e)
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

def score_steps(said, entry):
    """Ara adimli cevap (token listesi: 8 x adim + bir sonraki token).  Adlar kullanici onayiyla (27 Eylul):
    SC subject correct (cevap sorudaki kisiyle basliyor), BC bridge correct (ilk cumlenin nesnesi = kopru),
    AC answer correct (son cumlenin nesnesi = cevap; ANA OLCU), EX exact (tamami birebir VE ardindan <eos>: model
    durdu -- kullanici, 27 Eylul: "ex düzeltelim"), FC format correct (her cumle 'A B 's r_i is C D .', r_i sorudaki)."""
    want, path = entry["steps"], entry["path"]
    form = all(said[8 * i + 2] == "'s" and said[8 * i + 3] == r and said[8 * i + 4] == "is" and said[8 * i + 7] == "."
               for i, r in enumerate(path))
    return dict(SC=said[0:2] == want[0:2], BC=said[5:7] == want[5:7], AC=said[len(want) - 3:len(want) - 1] == want[-3:-1],
                EX=said == want + [data_y.EOS], FC=form)


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
        gens = generate(model, [q[0] + [ix[t] for t in q[2]["steps"][:given]] for q in group], 8 * hops - given + 1)
        for q, g in zip(group, gens):
            said = q[2]["steps"][:given] + [vocab[t] for t in g]
            for k, v in score_steps(said, q[2]).items():
                counts[k] += v
            rows.append((data_y.detokenize(q[2]["prompt"]), data_y.detokenize(said), data_y.detokenize(q[2]["steps"])))
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
    for rel in data_y.BASE:
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
            data_y.detokenize(e["prompt"]), answer(model, prompt, vocab), " | ".join(e["answers"]),
            ", ".join("%s %.2f" % (vocab[prompt[j]], float(v)) for j, v in zip(top.indices.tolist(), top.values.tolist()))))
    return "\n".join(out)


def report_step3(results, data):
    vocab = data["vocab"]
    ix = {w: i for i, w in enumerate(vocab)}
    people, rel = data["people"], data["rel"]
    firsts = torch.tensor(sorted({ix[p["first"]] for p in people.values()}))
    m0 = results["shared"][0]
    # kopru okumasi tur 1'in ara durumunu elle kuruyor: yalniz 28 Eylul oncesi tasarimda gecerli
    assert not any(not getattr(m, "stream_norm", True) or getattr(m, "layer_norm", False)
                   or getattr(m, "canon", False) or getattr(m, "normalized_update", False) or getattr(m, "heads", 1) > 1
                   or getattr(m, "learn_output_scale", False) for m, _ in results.values()), \
        "report_step3 kopru okumasi 28 Eylul oncesi tasarim icin (Canon, alpha, cok head, ogrenilen olcek yok)"
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
    for r in data_y.BASE:
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
            bridge = torch.tensor([ix[people[data_y.follow(rel, q[2]["subject"], q[2]["path"][:1])[0]]["first"]] for q in qs])
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
        out.append("  %-44s dogru: %-14s %s" % (data_y.detokenize(e["prompt"]), " | ".join(e["answers"]), answers))
    return "\n".join(out)


if __name__ == "__main__":
    torch.set_num_threads(1)
    data = data_y.build()
    data_y.audit(data)
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
