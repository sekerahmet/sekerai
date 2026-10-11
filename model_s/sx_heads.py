"""Attention ne yapiyor (TERSINE s4): transformer checkpoint'inde head basina attention kutlesinin turleri ve head'i kapatinca kayip.
Turler (j = okunan konum, t = sorgu): kendi (j = t), onceki (j = t - 1), induksiyon (x_{j-1} = x_t, j < t: bagladaki eski esin ardili),
bas (j = belge basi), uzak icerik (geri kalan; j <= t - 2, induksiyon degil). Kapatma: head'in cikisi 0 (kapatma olcusu, egitim degil).
Kullanim: python model_s/sx_heads.py --tf <transformer> [--rows 2]"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("deneme2/v2/common", "deneme2/v2/transformer", "deneme2/v2/diag", "model_s"):
    sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np  # noqa: E402
import data as D  # noqa: E402
from generate_readings import load_run  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402


def run(tf, b, mask, off=None, collect=None):
    x = tf.E(b.tokens)
    for li, blk in enumerate(tf.blocks):
        q, k, v = blk._qkv(x, b.pos)
        s = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
        s = s.masked_fill(~mask[:, None], float("-inf"))
        w = s.softmax(-1)
        if collect is not None:
            collect.append(w)
        a = w @ v
        if off is not None and off[0] == li:
            a = a.clone()
            a[:, off[1]] = 0
        x = blk._finish(x, a)
    lp = tf._logits(tf.norm(x)).float().log_softmax(-1)
    return -lp.gather(-1, b.target.clamp(min=0)[..., None])[..., 0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--no_ablate", action="store_true", help="head kapatmayi atla (yalniz kutle)")
    ap.add_argument("--data", default=r"G:/Drive'ım/v2/simplestories_gpt2")
    ap.add_argument("--stream", default=r"G:/Drive'ım/simplestories")
    a = ap.parse_args(argv)
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    tf, _ = load_run(a.tf, a.data, "cpu")
    tf.eval()
    valid = D.TokenStories(a.stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    ro, rs = ep["row_offsets"], ep["row_stories"]
    b = D.build_batch(valid, [rs[ro[r]:ro[r + 1]].tolist() for r in range(a.rows)], "transformer", "cpu")
    keep = b.target >= 0
    B, T = b.tokens.shape
    mask = (b.doc[:, :, None] == b.doc[:, None, :]) & torch.ones(T, T, dtype=torch.bool).tril()
    i = torch.arange(T)
    first = torch.ones(B, T, dtype=torch.bool)
    first[:, 1:] = b.doc[:, 1:] != b.doc[:, :-1]
    start = torch.cummax(torch.where(first, i[None], torch.zeros_like(i)[None]), 1).values      # belge basi konumu
    prev_tok = torch.full((B, T), -1, dtype=torch.long)
    prev_tok[:, 1:] = b.tokens[:, :-1]
    ind = (prev_tok[:, None, :] == b.tokens[:, :, None]) & mask & (i[None, None] < i[None, :, None])     # (B, t, j): x_{j-1} = x_t
    types = {"kendi": (i[:, None] == i[None]).expand(B, T, T), "onceki": (i[:, None] - 1 == i[None]).expand(B, T, T) & mask,
             "induksiyon": ind, "bas": (i[None, None] == start[:, :, None]) & mask}
    rest = mask.clone()
    for m in types.values():
        rest &= ~m
    types["uzak icerik"] = rest & (i[:, None] - i[None] >= 2)
    with torch.no_grad():
        W = []
        base = run(tf, b, mask, collect=W)
        L0 = float(base[keep].mean())
        q = keep.float()
        print("hedef %d, kayip %.4f; induksiyon adayi olan sorgu payi %.3f" % (int(keep.sum()), L0,
              float((ind.any(-1) & keep).float().sum() / keep.sum())))
        print("%-5s %s | %8s  baskin" % ("head", " ".join("%10s" % t for t in types), "kapat +"))
        rows = []
        for li, w in enumerate(W):
            for h in range(w.shape[1]):
                mass = {t: float(((w[:, h] * m).sum(-1) * q).sum() / q.sum()) for t, m in types.items()}
                d = 0.0 if a.no_ablate else float(run(tf, b, mask, off=(li, h))[keep].mean()) - L0
                dom = max(mass, key=mass.get)
                rows.append((li, h, mass, d, dom))
                print("L%dH%d %s | %+8.4f  %s" % (li, h, " ".join("%10.3f" % mass[t] for t in types), d, dom), flush=True)
    print("baskin ture gore: head sayisi, kapatma toplami")
    for t in types:
        r = [x for x in rows if x[4] == t]
        print("  %-12s %2d head, kapatma toplami %+.4f" % (t, len(r), sum(x[3] for x in r)))
    print("toplam kutle ortalamasi: " + "  ".join("%s %.3f" % (t, np.mean([x[2][t] for x in rows])) for t in types))


if __name__ == "__main__":
    main()
