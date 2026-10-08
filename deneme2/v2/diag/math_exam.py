"""math_exam -- Mathematics Dataset sinavi (belge 96; Saxton ve ark. 2019 s2.4): kayitli kosu (agent.pt) + make_math
verisi.  Bolum basina (valid = interpolate, extrapolate) modul basina en cok --per_module belge (tohum 0):
    exact      acgozlu uretilen cevap (istem = soru cumleleri, ilk uretilen cumle, END ile bitmeli) karakter karakter ayni;
               modul basina oran, genel = modul ortalamasi (makale), micro = belge ortalamasi
    answer     cevap hedefleri (data answer_only: sorunun son END / Z'si -> cevabin ilk karakteri, cevap karakterleri, cevap
               END'i; EOS ayri): nll ortalamasi, bpb = nll (EOS haric) / ln 2 / cevap bayti
    baseline   modulun train'deki en sik cevabi (math_meta.json) ayni belgelerde tam eslesme
    clean      math_meta.json sinav temizligi (soru / cift train'de)
Cikti <out>/math_exam.json ve math_exam.md (varsayilan kosu klasoru).

    python math_exam.py --run <kosu> --data <make_math cikti> [--splits valid,extrapolate] [--per_module 1000]
                        [--device cuda] [--out <klasor>]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402  (load, setup)
import train as TR  # noqa: E402

MAX_ANSWER = 31             # makale: cevap <= 30 karakter; 31. karakter uretilirse yanlis


def pick_docs(module_of, per_module, seed=0):
    """Modul basina en cok per_module belge (0: hepsi), tohumlu, sirali."""
    rng = np.random.default_rng(seed)
    out = [rng.permutation(np.flatnonzero(module_of == k))[:per_module or None] for k in np.unique(module_of)]
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


@torch.no_grad()
def answer_loss(model, layout, stories, docs, cuda):
    """Belgeler -> [nll toplami (EOS haric), hedef sayisi, EOS nll toplami, EOS sayisi]; egitimin maske yolu (train._Exam),
    answer_only hedefleri."""
    stories.answer_only = True
    try:
        ro, rs = D.pack_plan(stories.lengths()[docs], D.ROW_LEN, None)
        rows = [docs[rs[ro[r]:ro[r + 1]]].tolist() for r in range(len(ro) - 1)]
        if layout == "model_z":
            from model import summaries_last
            ex = TR._Exam(model, model._masks(True), cuda, summaries_last)
        else:
            import recipe as R
            ex = TR._Exam(model, R.document_mask, cuda)
        s = np.zeros(4)
        dev = model.E.weight.device
        for c in range(0, len(rows), TR.BATCH_ROWS):
            b = D.build_batch(stories, rows[c:c + TR.BATCH_ROWS], layout, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                nll, _, tk = ex.loss_per_target(b)
            nll, eos = nll.double().cpu(), (tk == D.TargetKind.EOS).cpu()
            s += [float(nll[~eos].sum()), int((~eos).sum()), float(nll[eos].sum()), int(eos.sum())]
        return s
    finally:
        stories.answer_only = False


def run_split(model, layout, stories, data_dir, meta, split, per_module, tok, cuda, batch_size):
    """Bir bolum -> dict(modules {ad: olculer}, overall, micro, answer_loss, answer_bpb, baseline_overall)."""
    mod = np.load(os.path.join(data_dir, split + "_doc_module.npy"))
    modules = meta["modules"]
    docs = pick_docs(mod, per_module)
    sents = [stories.sentences(int(i)) for i in docs]
    prompts = [[s.tolist() for s in x[:-1]] for x in sents]
    want = [tok.decode(x[-1].tolist()) for x in sents]
    out = model.generate(prompts, 1, MAX_ANSWER, None, batch_size=batch_size)
    got = [(tok.decode(g[0]) if g and e[0] else None) for g, e, _ in out]
    ok = np.array([g == w for g, w in zip(got, want)])
    top = meta["train_most_frequent_answer"]
    per = {}
    for k in np.unique(mod[docs]).tolist():
        name = modules[k]
        m = mod[docs] == k
        base = meta["cleanliness"][split][name].get("train_module")
        ta = top.get(base, {}).get("answer") if base else None
        per[name] = dict(n=int(m.sum()), exact=round(float(ok[m].mean()), 4),
                         baseline_exact=round(float(np.mean([w == ta for w, mm in zip(want, m) if mm])), 4) if ta
                         is not None else None, examples=[dict(q=tok.decode([t for s in prompts[i] for t in s]),
                                                               want=want[i], got=got[i])
                                                          for i in np.flatnonzero(m)[:3].tolist()],
                         **{k_: meta["cleanliness"][split][name][k_] for k_ in ("question_in_train", "pair_in_train")})
    s = answer_loss(model, layout, stories, docs, cuda)
    abytes = sum(len(w) for w in want)
    vals = [v["exact"] for v in per.values()]
    bvals = [v["baseline_exact"] for v in per.values() if v["baseline_exact"] is not None]
    return dict(docs=len(docs), modules=per, overall=round(float(np.mean(vals)), 4) if vals else None,
                micro=round(float(ok.mean()), 4) if len(ok) else None,
                answer_loss=round(s[0] / s[1], 4) if s[1] else None, answer_eos_loss=round(s[2] / s[3], 4) if s[3] else None,
                answer_bpb=round(s[0] / math.log(2) / abytes, 4) if abytes else None,
                baseline_overall=round(float(np.mean(bvals)), 4) if bvals else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, help="make_math ciktisi (vocab.json, math_meta.json)")
    ap.add_argument("--stream", default=None, help="akis koku (varsayilan --data)")
    ap.add_argument("--splits", default="valid,extrapolate")
    ap.add_argument("--per_module", type=int, default=1000, help="modul basina belge (0: hepsi)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup(args.device, log)
    model, _, layout, idt = G.load(args.run, args.data, dev)
    assert idt.get("vocab") == "ascii", "math_exam: ascii sozluklu kosu bekleniyor (%s)" % idt.get("vocab")
    meta = json.load(open(os.path.join(args.data, "math_meta.json"), encoding="utf-8"))
    stream = args.stream or args.data
    tok = D.load_tokenizer(stream, D.ASCII)
    res = dict(run=os.path.basename(os.path.normpath(args.run)), identity=idt, per_module=args.per_module, splits={})
    for split in [s for s in args.splits.split(",") if s]:
        if not os.path.exists(os.path.join(args.data, split + "_boundaries.json")):
            log("%s yok, atlandi" % split)
            continue
        st = D.TokenStories(stream, args.data, split)
        r = run_split(model, layout, st, args.data, meta, split, args.per_module, tok, dev.type == "cuda", args.batch_size)
        res["splits"][split] = r
        log("%s: %d belge, exact %s (micro %s), taban %s, cevap kaybi %s bpb %s" % (
            split, r["docs"], r["overall"], r["micro"], r["baseline_overall"], r["answer_loss"], r["answer_bpb"]))
    out = args.out or args.run
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "math_exam.json"), "w", encoding="utf-8"), indent=1)
    md = ["# math_exam: %s" % res["run"], ""]
    for split, r in res["splits"].items():
        md += ["## %s: exact %s (micro %s), en sik cevap tabani %s, cevap kaybi %s, bpb %s (%d belge)" % (
            split, r["overall"], r["micro"], r["baseline_overall"], r["answer_loss"], r["answer_bpb"], r["docs"]), "",
               "| modul | n | exact | taban | soru train'de | cift train'de |", "|---|---|---|---|---|---|"]
        md += ["| %s | %d | %s | %s | %d | %d |" % (k, v["n"], v["exact"], v["baseline_exact"], v["question_in_train"],
                                                   v["pair_in_train"]) for k, v in r["modules"].items()]
        md.append("")
    open(os.path.join(out, "math_exam.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    log("BITTI: %s" % out)
    return res


if __name__ == "__main__":
    main()
