"""Strata-x token izi: tek bir sinav hikayesinde token token Reader surprizi, birim sonlari, Strata / transformer kaybi, Thinker kapali
(ctx = g0) kaybi (kullanici, 11 Ekim: "bir örnek üzerinden düşünme deneyi de kesin yapardım").
Kullanim: python model_s/sx_trace.py --run <strata kosu> --tf <transformer kosu> [--story 0] [--tokens 160]
"""
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
from sx_voice_read import _no_power_throttling, first_seen  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--story", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--data", default=r"G:/Drive'ım/v2/simplestories_gpt2")
    ap.add_argument("--stream", default=r"G:/Drive'ım/simplestories")
    a = ap.parse_args(argv)
    _no_power_throttling()
    m, _ = load_run(a.run, a.data, "cpu")
    tf, _ = load_run(a.tf, a.data, "cpu")
    m.eval()
    tf.eval()
    valid = D.TokenStories(a.stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    story = int(ep["row_stories"][a.story])
    b = D.build_batch(valid, [[story]], "transformer", "cpu")
    keep = b.target >= 0
    with torch.no_grad():
        h, _, ends, info = m.forward_states(b.tokens, b.pos, b.doc)
        s = torch.zeros_like(b.target, dtype=torch.float)
        s[keep] = m._nll(h[keep], b.target[keep])
        real_down = m.down.forward
        m.down.forward = lambda x: m.g0.expand(*x.shape[:-1], -1).clone()       # Thinker kapali: her token g0
        h0 = m.forward_states(b.tokens, b.pos, b.doc)[0]
        m.down.forward = real_down
        s0 = torch.zeros_like(s)
        s0[keep] = m._nll(h0[keep], b.target[keep])
        t = torch.zeros_like(s)
        t[keep] = tf.loss_per_target(b)[0]
    first = first_seen(b.tokens, b.target, b.doc)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.stream, "gpt2", "tokenizer.json"))
    dec = lambda i: "<s>" if int(i) >= 50257 else tok.decode([int(i)])  # noqa: E731    50257: cumle siniri
    n = min(a.tokens, int(keep.sum()))
    print("hikaye %d | birim / token %.3f | Strata %.3f  transformer %.3f  Thinker kapali %.3f (ilk %d hedef)" % (
        story, float(ends[0, :n].float().mean()), float(s[0, :n].mean()), float(t[0, :n].mean()), float(s0[0, :n].mean()), n))
    print("%4s %-14s %5s %3s | %-14s %3s %6s %6s %6s %6s" % ("t", "token t", "surp", "son", "hedef t+1", "ilk", "strata", "tf", "fark", "Th.off"))
    for i in range(n):
        print("%4d %-14r %5.2f %3s | %-14r %3s %6.2f %6.2f %+6.2f %+6.2f" % (
            i, dec(b.tokens[0, i]), float(info[0, i]), "|" if ends[0, i] else "", dec(b.target[0, i]), "*" if first[0, i] else "",
            float(s[0, i]), float(t[0, i]), float(s[0, i] - t[0, i]), float(s0[0, i] - s[0, i])))


if __name__ == "__main__":
    main()
