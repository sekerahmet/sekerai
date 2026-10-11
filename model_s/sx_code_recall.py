"""Kod sonek bellegi (TARTISMA_attention Soz 5 / 7): FineWeb transformer'inin katman --layer durumlari k-means ile ayrik kodlara (b 8 / 12,
16 = iki yarinin 8'er bitlik carpimi); kod dizisinde belge ici en uzun onceki sonek esi, o konumun ardil TOKEN'i sonraki token'i dogru
veriyor mu. Token kimlikleriyle ayni olcu (kiyas). Kodlar ayri belgelerde ogrenilir (--fit), olcu baska belgelerde (--eval).
Kullanim: python model_s/sx_code_recall.py [--layer 5] [--fit 150] [--eval 300] [--maxlen 1024]"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "transformer"))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from baseline import BaselineTransformer  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402

RUN = r"G:/Drive'ım/v2/runs/v2_tf_fw_d768_l10_muon_lr2e-3_20261007_150554"


def states(tf, ids, layer):
    x = tf.E(torch.tensor([ids]))
    pos = torch.arange(len(ids))[None]
    for blk in tf.blocks[:layer]:
        x = blk(x, pos, None)
    return torch.nn.functional.rms_norm(x[0], (x.shape[-1],))


def kmeans(X, K, iters=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:K]].clone()
    for _ in range(iters):
        a = assign(X, C)
        S = torch.zeros_like(C).index_add_(0, a, X)
        n = torch.bincount(a, minlength=K).float()
        empty = n == 0
        C = torch.where(empty[:, None], X[torch.randint(len(X), (K,), generator=g)], S / n.clamp(min=1)[:, None])
    return C


def assign(X, C, chunk=8192):
    out = []
    cn = (C * C).sum(-1)
    for i in range(0, len(X), chunk):
        out.append((cn[None] - 2 * X[i:i + chunk] @ C.T).argmin(-1))
    return torch.cat(out)


def suffix(sym_docs, tok_docs, maxn=16):
    hit = np.zeros((maxn + 1, 2))
    total = 0
    for s, d in zip(sym_docs, tok_docs):
        last = {}
        for t in range(len(d) - 1):
            total += 1
            for n in range(min(maxn, t + 1), 0, -1):
                j = last.get(tuple(s[t - n + 1:t + 1]))
                if j is not None:
                    hit[n, 0] += 1
                    hit[n, 1] += d[j + 1] == d[t + 1]
                    break
            for n in range(1, min(maxn, t + 1) + 1):
                last[tuple(s[t - n + 1:t + 1])] = t
    return hit, total


def report(name, hit, total):
    m, c = hit[:, 0], hit[:, 1]
    row = []
    for lo in (1, 2, 4, 8):
        row.append("m>=%d kapsama %.3f dogru %.3f (%%%.0f)" % (lo, m[lo:].sum() / total, c[lo:].sum() / total,
                                                            100 * c[lo:].sum() / max(m[lo:].sum(), 1)))
    print("%-22s %s" % (name, " | ".join(row)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--fit", type=int, default=150)
    ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--maxlen", type=int, default=1024)
    a = ap.parse_args()
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.eval()
    from fw_docs import fw_docs
    docs = [d for d, _ in fw_docs(a.fit + a.eval, start=0, maxlen=a.maxlen, minlen=16)]   # V2 bicimi (END token'lari)
    fit, ev = docs[:a.fit], docs[a.fit:a.fit + a.eval]
    with torch.no_grad():
        Hf = torch.cat([states(tf, d, a.layer) for d in fit])
        He = [states(tf, d, a.layer) for d in ev]
    print("katman %d | kod ogrenme %d belge (%d durum), olcu %d belge (%d token)" % (
        a.layer, len(fit), len(Hf), len(ev), sum(len(d) for d in ev)), flush=True)
    report("token kimligi", *suffix(ev, ev))
    for b in (8, 12):
        C = kmeans(Hf, 2 ** b)
        report("kod b%d (k-means %d)" % (b, 2 ** b), *suffix([assign(h, C).tolist() for h in He], ev))
    h = Hf.shape[1] // 2                                                    # b16: iki yarinin 8'er bitlik carpimi
    C1, C2 = kmeans(Hf[:, :h], 256, seed=1), kmeans(Hf[:, h:], 256, seed=2)
    codes = [(assign(x[:, :h], C1) * 256 + assign(x[:, h:], C2)).tolist() for x in He]
    report("kod b16 (PQ 2x256)", *suffix(codes, ev))


if __name__ == "__main__":
    main()
