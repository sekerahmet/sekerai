"""order_probe -- sira sinamasi (belge 31; ad onayli): bir model sirayi okuyor mu.  Iki olcu, gap_v2 gibi kayitli kosular:

1. Kim kime: "One day, A <eylem> B." -> ikinci cumlenin ilk token'inda A mi B mi.  8 dogal kalip (5 etken, cevap B; 3
   edilgen, cevap A), 10 ad, sirali ciftler (720 ornek).  (A, B) ve (B, A) ayni kelime torbasi.
   sira etkisi = ort |m(A,B) + m(B,A)| (m = log P(dogru) - log P(oteki); torba okuyan model icin 0), sira duyarliligi =
   cift yer degisince tercih edilen adin degisme orani, dogruluk.
2. Gecmis karistirma (belge 30 / 31 on kayit): sinav hikayelerinden N hikaye x 3 rastgele k (>= 1); hedef cumle aynen,
   gecmis: C0 asil, C1 butun onceki cumlelerin ici karisik, C1p yalniz bir onceki cumlenin ici karisik.  Hedef cumlenin
   yeni icerik kaybi ve artislari.

    python order_probe.py --data <v2/simplestories_gpt2> --stream <simplestories> [--tf <agent.pt>] [--mz <agent.pt> ...]
                          [--meaning <meaning agent.pt>] [--stories 40] [--out <json>]
CPU, eager, dense maske (model._batch_hidden(batch, None); open_z dahil).
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
import train as T  # noqa: E402
from gap_v2 import _load, target_features  # noqa: E402

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
    """-> (nll, tgt, kind) hedef basina, satir sirasiyla (dense maske; open_z dahil)."""
    out = [], [], []
    W = model.E.weight
    for r in rows:
        b = D.build_batch(st, [r], layout, "cpu")
        h = model._batch_hidden(b, None)
        keep = b.target >= 0
        lg = torch.log_softmax((h[keep] @ W.T).float(), -1)
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
    W = model.E.weight
    m = []
    for r in rows:
        b = D.build_batch(st, [r], layout, "cpu")
        h = model._batch_hidden(b, None)[0]
        tokm = (b.kind[0] == D.Kind.TOKEN) & (b.sent[0] == 1)
        for j, i in enumerate(r):
            first = int(torch.nonzero(tokm & (b.doc[0] == j)).min())
            lg = torch.log_softmax((h[first - 1] @ W.T).float(), -1)
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
        out[c] = dict(hepsi=round(float(x.mean()), 4), yeni_icerik=round(float(x[fl].mean()), 4), n=int(m.sum()),
                      n_yeni=int(fl.sum()))
    for c in ("C1", "C1p"):
        out[c]["artis_yeni"] = round(out[c]["yeni_icerik"] - out["C0"]["yeni_icerik"], 4)
        out[c]["artis_hepsi"] = round(out[c]["hepsi"] - out["C0"]["hepsi"], 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--tf", default=None, help="transformer agent.pt")
    ap.add_argument("--mz", nargs="*", default=[], help="Model Z agent.pt (birden cok olabilir)")
    ap.add_argument("--meaning", default=None, help="meaning agent.pt (ortak sozluk / bugunku Model Z icin)")
    ap.add_argument("--stories", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    if os.name == "nt":
        print("guc kisitlamasi (EcoQoS) kapali:", T._no_power_throttling(), flush=True)
    torch.set_num_threads(8)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.stream, "gpt2", "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    text = [tok.decode([i]) for i in range(tok.get_vocab_size())]
    valid = D.TokenStories(args.stream, args.data, "valid")
    meta = json.load(open(os.path.join(args.data, "train_boundaries.json"), encoding="utf-8"))
    longest = meta.get("max_sentence_tokens_all", meta["max_sentence_tokens"])
    count = torch.load(args.meaning, map_location="cpu", weights_only=False).get("count") if args.meaning else None
    count = np.bincount(np.asarray(valid.stream), minlength=len(text))[:len(text)] if count is None else count.numpy()
    func = np.zeros(len(text) + 1, bool)
    func[np.argsort(-count)[:50]] = True
    func[:len(text)] |= np.array([not any(c.isalnum() for c in s) for s in text])
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    ids = np.unique(ep["row_stories"])
    pick = np.random.default_rng(0).choice(ids, min(args.stories, len(ids)), replace=False)
    runs = ([("transformer", args.tf)] if args.tf else []) + [("model_z", p) for p in args.mz]
    res = {}
    for kind, path in runs:
        model, _, layout = _load(kind, path, args.meaning, longest, torch.device("cpu"), 0)
        name = os.path.basename(os.path.dirname(path)) or path
        r = dict(kim_kime=who_did_what(model, layout, enc), gecmis_karistirma=history_shuffle(model, layout, valid, pick,
                                                                                             func))
        res[name] = r
        k, g = r["kim_kime"], r["gecmis_karistirma"]
        print("%-46s kim kime: dogruluk %.3f, sira etkisi %.3f, duyarlilik %.3f | gecmis karistirma, yeni icerik: C0 %.3f, "
              "C1 %+.3f, C1p %+.3f  (%.0f sn)" % (name, k["dogruluk"], k["sira_etkisi"], k["sira_duyarliligi"],
                                                 g["C0"]["yeni_icerik"], g["C1"]["artis_yeni"], g["C1p"]["artis_yeni"],
                                                 time.time() - t0), flush=True)
        del model
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    main()
