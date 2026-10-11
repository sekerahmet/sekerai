"""Matematikcinin tek bilinmeyeni (danisman/04): head basina ilk-k disinda kalan attention kutlesi m_dis(k), k 8 / 32; ve uzak kutlenin
(j < t - W) kesin eslesen induksiyon konumlarina (x_{j-1} = x_t) dusen payi r_b, W 64 / 256.
Kullanim: python model_s/sx_topk.py --tf <transformer> [--rows 2]"""
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rows", type=int, default=2)
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
    i = torch.arange(T)
    mask = (b.doc[:, :, None] == b.doc[:, None, :]) & torch.ones(T, T, dtype=torch.bool).tril()
    prev_tok = torch.full((B, T), -1, dtype=torch.long)
    prev_tok[:, 1:] = b.tokens[:, :-1]
    ind = (prev_tok[:, None, :] == b.tokens[:, :, None]) & mask & (i[None, None] < i[None, :, None])
    dist = i[:, None] - i[None]
    q = keep.float()
    print("%-5s %8s %8s | %10s %8s %10s %8s" % ("head", "m_dis8", "m_dis32", "uzak>64", "r_b64", "uzak>256", "r_b256"))
    agg = []
    with torch.no_grad():
        x = tf.E(b.tokens)
        for li, blk in enumerate(tf.blocks):
            qq, k, v = blk._qkv(x, b.pos)
            w = ((qq @ k.transpose(-1, -2)) / qq.shape[-1] ** 0.5).masked_fill(~mask[:, None], float("-inf")).softmax(-1)
            srt = w.sort(-1, descending=True).values
            for h in range(w.shape[1]):
                wh = w[:, h]
                m8 = 1 - srt[:, h, :, :8].sum(-1)
                m32 = 1 - srt[:, h, :, :32].sum(-1)
                row = [float((m8 * q).sum() / q.sum()), float((m32 * q).sum() / q.sum())]
                for W in (64, 256):
                    far = wh * (dist[None] > W)
                    fm = far.sum(-1)
                    rb = (far * ind).sum(-1)
                    row += [float((fm * q).sum() / q.sum()), float((rb * q).sum() / (fm * q).sum().clamp(min=1e-12))]
                agg.append(row)
                print("L%dH%d %8.3f %8.3f | %10.3f %8.3f %10.3f %8.3f" % (li, h, *row), flush=True)
            x = blk._finish(x, w @ v)
    A = np.array(agg)
    print("ortalama  m_dis8 %.3f  m_dis32 %.3f | uzak>64 %.3f r_b64 %.3f | uzak>256 %.3f r_b256 %.3f" % tuple(A.mean(0)))
    print("m_dis32 < 0,05 olan head %d / %d; r_b64 > 0,5 olan head %d" % ((A[:, 1] < 0.05).sum(), len(A), (A[:, 3] > 0.5).sum()))


if __name__ == "__main__":
    main()
