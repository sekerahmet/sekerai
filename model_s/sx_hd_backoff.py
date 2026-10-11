"""Tanimli parcalarin kaliteye ust siniri (matematikci): egitilmis FineWeb transformer'i p_0; belge ici n-gram sayimlari (m 1..M, son m
token'in onceki gecislerinin ardillari) hiyerarsik Dirichlet geri cekilmesiyle: p_m(y) = (c_m(y) + a_m p_{m-1}(y)) / (n_m + a_m); esi
olmayan m'de p_m = p_{m-1}. a_m sabit (ogrenilen a(h) bundan iyi olur), belgelerin yarisinda izgarayla secilir, oteki yarida olculur.
Kullanim: python model_s/sx_hd_backoff.py [--docs 200] [--M 4]"""
import argparse
import math
import os
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "transformer"))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from baseline import BaselineTransformer  # noqa: E402
from sx_code_recall import RUN  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402
from fw_docs import fw_docs  # noqa: E402

GRID = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)


def doc_events(ids, tgt, M):
    """konum t (hedef tgt[t]) icin m = 1..M: (c_m(y), n_m) -- yalniz t'den onceki gecisler; hedefsiz konum atlanir."""
    tabs = [defaultdict(Counter) for _ in range(M + 1)]
    ev = []
    for t in range(len(ids)):
        y = tgt[t]
        if y < 0:
            continue
        row = []
        for m in range(1, M + 1):
            if t - m + 1 < 0:
                row.append((0, 0))
                continue
            c = tabs[m].get(tuple(ids[t - m + 1:t + 1]))
            row.append((c[y], sum(c.values())) if c else (0, 0))
        ev.append(row)
        for m in range(1, M + 1):                                      # x_{t-m+1..t} -> ardil y artik "goruldu"
            if t - m + 1 >= 0:
                tabs[m][tuple(ids[t - m + 1:t + 1])][y] += 1
    return ev


def hd_loss(p0, ev, alpha):
    tot = 0.0
    for p, row in zip(p0, ev):
        for (c, n), a in zip(row, alpha):
            if n:
                p = (c + a * p) / (n + a)
        tot -= math.log(p)
    return tot / len(p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=1024)
    a = ap.parse_args()
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.eval()
    docs = fw_docs(a.docs, start=3000, maxlen=a.maxlen)                  # V2 bicimi; kod / parca olculerinden ayri belgeler
    P0, EV = [], []
    with torch.no_grad():
        for d, tg in docs:
            x = torch.tensor([d])
            lp = tf._logits(tf.hidden(x, torch.arange(len(d))[None], None)).float().log_softmax(-1)[0]
            t = torch.tensor(tg)
            ok = t >= 0
            P0.append(lp[ok].gather(1, t[ok][:, None])[:, 0].exp().tolist())
            EV.append(doc_events(d, tg, a.M))
    half = len(docs) // 2
    p0A = [p for d in P0[:half] for p in d]
    evA = [e for d in EV[:half] for e in d]
    p0B = [p for d in P0[half:] for p in d]
    evB = [e for d in EV[half:] for e in d]
    alpha = [GRID[-1]] * a.M
    for _ in range(2):                                                  # koordinat koordinat izgara
        for m in range(a.M):
            best = min(GRID, key=lambda g: hd_loss(p0A, evA, alpha[:m] + [g] + alpha[m + 1:]))
            alpha[m] = best
    base = -np.mean(np.log(p0B))
    hd = hd_loss(p0B, evB, alpha)
    print("belge %d (secim %d / olcu %d), olcu token %d | a_m %s" % (len(docs), half, len(docs) - half, len(p0B), alpha))
    print("transformer p_0: %.4f nat | + baglam ici n-gram geri cekilme (M %d): %.4f | kazanc %.4f nat (%.2f%%)" % (
        base, a.M, hd, base - hd, 100 * (base - hd) / base))
    for M in range(1, a.M + 1):
        print("  yalniz m <= %d: %.4f" % (M, hd_loss(p0B, [e[:M] for e in evB], alpha[:M])))


if __name__ == "__main__":
    main()
