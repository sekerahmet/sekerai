"""Strata-x guven / kalibrasyon: Strata (Thinker acik / kapali) ve transformer'in entropisi, en olasi token olasiligi, yanildiginda guveni.
Kullanim: python model_s/sx_confidence.py --run <strata> --tf <transformer> [--rows 6]"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("deneme2/v2/common", "deneme2/v2/transformer", "deneme2/v2/diag", "model_s"):
    sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np  # noqa: E402
import data as D  # noqa: E402
from generate_readings import load_run  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402


def stats(logp, tgt):
    p = logp.exp()
    H = -(p * logp).sum(-1)
    top, arg = p.max(-1)
    nll = -logp.gather(1, tgt[:, None])[:, 0]
    return H, top, arg == tgt, nll


def dist(h_fn, keep, tgt, chunk=2048):
    hs = h_fn()[keep]
    out = [stats(lp, tgt[i:i + chunk]) for i, lp in
           ((i, h_lp(hs[i:i + chunk])) for i in range(0, hs.shape[0], chunk))]
    return [torch.cat(x) for x in zip(*out)]


def main(argv=None):
    global h_lp
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--data", default=r"G:/Drive'ım/v2/simplestories_gpt2")
    ap.add_argument("--stream", default=r"G:/Drive'ım/simplestories")
    a = ap.parse_args(argv)
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    m, _ = load_run(a.run, a.data, "cpu")
    tf, _ = load_run(a.tf, a.data, "cpu")
    m.eval()
    tf.eval()
    valid = D.TokenStories(a.stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    ro, rs = ep["row_offsets"], ep["row_stories"]
    b = D.build_batch(valid, [rs[ro[r]:ro[r + 1]].tolist() for r in range(a.rows)], "transformer", "cpu")
    keep = b.target >= 0
    tgt = b.target[keep]
    with torch.no_grad():
        h_lp = lambda h: m._logits(h).float().log_softmax(-1)  # noqa: E731
        S = dist(lambda: m.forward_states(b.tokens, b.pos, b.doc)[0], keep, tgt)
        real = m.down.forward
        m.down.forward = lambda x: m.g0.expand(*x.shape[:-1], -1).clone()
        S0 = dist(lambda: m.forward_states(b.tokens, b.pos, b.doc)[0], keep, tgt)
        m.down.forward = real
        h_lp = lambda h: tf._logits(h).float().log_softmax(-1)  # noqa: E731
        T = dist(lambda: tf._batch_hidden(b), keep, tgt)
    Htf = T[0]
    print("hedef %d" % int(keep.sum()))
    print("%-22s %-16s %6s %6s %6s %6s %13s" % ("tur", "model", "nll", "H", "top-p", "acc", "yanlista top-p"))
    for nm, mm in (("hepsi", torch.ones_like(Htf, dtype=torch.bool)), ("H_tf > 2", Htf >= 2), ("H_tf 0,5-2", (Htf >= 0.5) & (Htf < 2))):
        for lab, (H, top, ok, nll) in (("Strata", S), ("Strata Thinker kapali", S0), ("transformer", T)):
            w = mm & ~ok
            print("%-22s %-16s %6.3f %6.3f %6.3f %6.3f %13.3f" % (nm, lab[:16], float(nll[mm].mean()), float(H[mm].mean()), float(top[mm].mean()),
                  float(ok[mm].float().mean()), float(top[w].mean())))
    # kalibrasyon: top-p kovasina gore gercek dogruluk
    print("kalibrasyon (top-p kovasi: n / ortalama top-p / gercek acc)")
    for lab, (H, top, ok, nll) in (("Strata", S), ("transformer", T)):
        row = []
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            mm = (top >= lo) & (top < hi)
            row.append("%d/%.2f/%.2f" % (int(mm.sum()), float(top[mm].mean()), float(ok[mm].float().mean())))
        print("  %-12s %s" % (lab, "  ".join(row)))


if __name__ == "__main__":
    main()
