"""order_probe -- sira sinamasi (belge 31; ad onayli): bir model sirayi okuyor mu.  diag/'a tasindi (belge 33 adim 5):
kosular generate_readings.load_run ile (global_layers kimlikten), islev ayrimi train token sayimindan (gap_v2.token_counts).

1. Kim kime: "One day, A <eylem> B." -> ikinci cumlenin ilk token'inda A mi B mi.  8 dogal kalip (5 etken, cevap B; 3
   edilgen, cevap A), 10 ad, sirali ciftler (720 ornek).  (A, B) ve (B, A) ayni kelime torbasi.
   sira etkisi = ort |m(A,B) + m(B,A)| (m = log P(dogru) - log P(oteki); torba okuyan model icin 0), sira duyarliligi =
   cift yer degisince tercih edilen adin degisme orani, dogruluk.
2. Gecmis karistirma (belge 30 / 31 on kayit): sinav hikayelerinden N hikaye x 3 rastgele k (>= 1); hedef cumle aynen,
   gecmis: C0 asil, C1 butun onceki cumlelerin ici karisik, C1p yalniz bir onceki cumlenin ici karisik.  Hedef cumlenin
   yeni icerik kaybi ve artislari.

    python order_probe.py --data <v2/simplestories_gpt2> --stream <simplestories> --runs <kosu klasoru> [...]
                          [--counts <train_token_counts.npy>] [--stories 40] [--out <json>]
CPU, eager, dense maske (model._batch_hidden(batch, None): modelin kendi maske kurali).
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402  (load, target_features, token_counts, function_tokens)
from gap_v2 import target_features  # noqa: E402

NAMES = [" Mia", " Leo", " Lily", " Max", " Sam", " Tom", " Ben", " Anna", " Zoe", " Jack"]
TEMPLATES = [("One day,{A} threw the ball to{B}.", "B"), ("One day,{A} asked{B} for help.", "B"),
             ("One day,{A} called out to{B}.", "B"), ("One day,{A} handed{B} a book.", "B"),
             ("One day,{A} waved at{B}.", "B"), ("One day,{A} was pushed by{B}.", "A"),
             ("One day,{A} was chased by{B}.", "A"), ("One day,{A} was hugged by{B}.", "A")]


def _stories(groups):
    """Hikaye -> cumle listeleri -> build_batch'in okudugu nesne (stream, sent, story, sentences)."""
    stream, sent, story = [], [], [0]
    for st in groups:
        for x in st:
            sent.append((len(stream), len(stream) + len(x)))
            stream += [int(t) for t in x]
        story.append(len(sent))
        stream.append(D.EOS_ID)
    out = SimpleNamespace(stream=np.array(stream, np.int64), sent=np.array(sent, np.int64), story=np.array(story))
    out.sentences = lambda i: [out.stream[a:b] for a, b in out.sent[out.story[i]:out.story[i + 1]]]
    return out


def _rows(groups, row_len=D.ROW_LEN):
    rows, cur, used = [], [], 0
    for i, st in enumerate(groups):
        L = 1 + sum(len(x) + 1 for x in st)
        if cur and used + L > row_len:
            rows.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += L
    return rows + [cur]


@torch.no_grad()
def _nll(model, layout, st, rows):
    """-> (nll, tgt, kind) hedef basina, satir sirasiyla (dense maske: modelin kendi kurali, global_layers dahil)."""
    out = [], [], []
    for r in rows:
        b = D.build_batch(st, [r], layout, "cpu")
        h = model._batch_hidden(b, None)
        keep = b.target >= 0
        lg = torch.log_softmax(model._logits(h[keep]).float(), -1)         # torbali modelde iki asamali (belge 55 K3)
        out[0].append(-lg.gather(1, b.target[keep][:, None])[:, 0].numpy())
        out[1].append(b.target[keep].numpy())
        out[2].append(b.target_kind[keep].numpy())
    return tuple(np.concatenate(x) for x in out)


@torch.no_grad()
def who_did_what(model, layout, enc):
    """-> dict(dogruluk, sira_etkisi, sira_duyarliligi, kalip_marj)."""
    items, groups = [], []
    for (pat, who), (a, b) in itertools.product(TEMPLATES, itertools.permutations(NAMES, 2)):
        s1 = enc(pat.format(A=a, B=b))
        ans, oth = (enc(a)[0], enc(b)[0]) if who == "A" else (enc(b)[0], enc(a)[0])
        items.append((ans, oth))
        groups.append([s1, [ans] + enc(" smiled.")])
    st, rows = _stories(groups), _rows(groups)
    m = []
    for r in rows:
        b = D.build_batch(st, [r], layout, "cpu")
        h = model._batch_hidden(b, None)[0]
        tokm = (b.kind[0] == D.Kind.TOKEN) & (b.sent[0] == 1)
        for j, i in enumerate(r):
            first = int(torch.nonzero(tokm & (b.doc[0] == j)).min())
            lg = torch.log_softmax(model._logits(h[first - 1]).float(), -1)
            m.append(float(lg[items[i][0]] - lg[items[i][1]]))
    m = np.array(m)
    pairs = list(itertools.permutations(range(len(NAMES)), 2))
    pidx = {p: i for i, p in enumerate(pairs)}
    n = len(pairs)
    flips, eff = [], []
    for t, (_, who) in enumerate(TEMPLATES):
        for x, y in pairs:
            if x < y:
                m1, m2 = m[t * n + pidx[(x, y)]], m[t * n + pidx[(y, x)]]
                flips.append(((m1 > 0) == (who == "A")) != ((m2 > 0) == (who == "B")))
                eff.append(abs(m1 + m2))
    return dict(dogruluk=round(float(np.mean(m > 0)), 4), sira_etkisi=round(float(np.mean(eff)), 4),
                sira_duyarliligi=round(float(np.mean(flips)), 4),
                kalip_marj=[round(float(m[t * n:(t + 1) * n].mean()), 3) for t in range(len(TEMPLATES))])


@torch.no_grad()
def history_shuffle(model, layout, valid, story_ids, func, seed=0):
    """Gecmis karistirma: -> {kosul: dict(hepsi, yeni_icerik, artis_yeni)} (hedef cumle FIRST + MID)."""
    rng = np.random.default_rng(seed)
    variants, flags = {c: [] for c in ("C0", "C1", "C1p")}, []
    for sid in story_ids:
        S = [x.tolist() for x in valid.sentences(int(sid))]
        if len(S) < 2:
            continue
        for k in rng.choice(np.arange(1, len(S)), min(3, len(S) - 1), replace=False):
            hist, cur = S[:k], S[k]
            seen = {t for x in hist for t in x}
            here, fl = set(), []
            for t in cur:
                fl.append((not func[t]) and t not in seen and t not in here)
                here.add(t)
            flags.append(np.array(fl))
            variants["C0"].append(hist + [cur])
            variants["C1"].append([list(rng.permutation(x)) for x in hist] + [cur])
            variants["C1p"].append(hist[:-1] + [list(rng.permutation(hist[-1]))] + [cur])
    fl = np.concatenate(flags)
    out = {}
    for c, groups in variants.items():
        st, rows = _stories(groups), _rows(groups)
        nll, _, kind = _nll(model, layout, st, rows)
        f = target_features(st, rows)
        m = ((f["kind"] == D.TargetKind.FIRST) | (f["kind"] == D.TargetKind.MID)) & (f["k"] == f["n_sent"] - 1)
        assert m.sum() == len(fl)
        x = nll[m]
        out[c] = dict(hepsi=round(float(x.mean()), 4), yeni_icerik=round(float(x[fl].mean()), 4) if fl.any() else None,
                      n=int(m.sum()), n_yeni=int(fl.sum()))
    for c in ("C1", "C1p"):
        new = (out[c]["yeni_icerik"], out["C0"]["yeni_icerik"])
        out[c]["artis_yeni"] = round(new[0] - new[1], 4) if None not in new else None
        out[c]["artis_hepsi"] = round(out[c]["hepsi"] - out["C0"]["hepsi"], 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="kosu klasorleri (agent.pt; model turu kimlikten)")
    ap.add_argument("--counts", default=None, help="train_token_counts.npy (yoksa <data>/ ya da akistan)")
    ap.add_argument("--stories", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup("cpu", log)
    tok, text = G.vocab_text(args.stream)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    valid = D.TokenStories(args.stream, args.data, "valid")
    func = G.function_tokens(text, G.token_counts(args, log))
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    ids = np.unique(ep["row_stories"])
    pick = np.random.default_rng(0).choice(ids, min(args.stories, len(ids)), replace=False)
    res = {}
    for run in args.runs:
        model, _, layout, idt = G.load(run, args.data, dev)
        name = os.path.basename(os.path.normpath(run))
        r = dict(kim_kime=who_did_what(model, layout, enc), gecmis_karistirma=history_shuffle(model, layout, valid, pick,
                                                                                             func),
                 identity={k: idt.get(k) for k in ("model", "d", "layers", "heads", "seed", "global_layers")})
        res[name] = r
        k, g = r["kim_kime"], r["gecmis_karistirma"]
        kk = "dogruluk %.3f, sira etkisi %.3f, duyarlilik %.3f" % (k["dogruluk"], k["sira_etkisi"], k["sira_duyarliligi"])
        log("%-46s kim kime: %s | gecmis karistirma, yeni icerik: C0 %s, C1 %s, C1p %s" % (
            name, kk, g["C0"]["yeni_icerik"], g["C1"]["artis_yeni"], g["C1p"]["artis_yeni"]))
        del model
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    main()
