"""Paylasilan parca bellegi on sinamasi (TARTISMA_attention Soz 14): parcalar adres olarak calisir mi. FineWeb transformer'inin --layer
katmaninda head basina anahtarlar K parcaya (k-means, ayri belgelerde); anahtar ve sorgu ic carpimla top-s parcaya atanir; sorgunun aday
kumesi = parca kesisimi olan onceki konumlar. Olcu: gercek attention kutlesinin adaylarda kalan payi, aday orani; taban ayni sayida rastgele.
Kullanim: python model_s/sx_part_address.py [--layer 5] [--K 1024] [--fit 60] [--eval 60]"""
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
from sx_code_recall import RUN, kmeans  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402


def qk(tf, ids, layer):
    x = tf.E(torch.tensor([ids]))
    pos = torch.arange(len(ids))[None]
    for blk in tf.blocks[:layer]:
        x = blk(x, pos, None)
    q, k, _ = tf.blocks[layer]._qkv(x, pos)
    return q[0], k[0]                                                      # (H, T, d_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--fit", type=int, default=60)
    ap.add_argument("--eval", type=int, default=60)
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
    docs = [d for d, _ in fw_docs(a.fit + a.eval, start=1000, maxlen=a.maxlen, minlen=256)]   # V2 bicimi
    with torch.no_grad():
        fitk = [qk(tf, d, a.layer)[1] for d in docs[:a.fit]]
        H = fitk[0].shape[0]
        C = [kmeans(torch.cat([k[h] for k in fitk]), a.K, iters=10, seed=h) for h in range(H)]
        S_LIST = (1, 4, 16)
        acc = {s: np.zeros((H, 4)) for s in S_LIST}                         # yakalanan kutle, aday orani, rastgele taban, sorgu sayisi
        ent = np.zeros(H)
        g = torch.Generator().manual_seed(0)
        for d in docs[a.fit:]:
            q, k = qk(tf, d, a.layer)
            T = q.shape[1]
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            for h in range(H):
                w = ((q[h] @ k[h].T) / q.shape[-1] ** 0.5).masked_fill(~causal, float("-inf")).softmax(-1)
                ent[h] += float(-(w * w.clamp(min=1e-12).log()).sum(-1).mean())
                sk, sq = k[h] @ C[h].T, q[h] @ C[h].T
                for s in S_LIST:
                    mk = torch.zeros(T, a.K)
                    mk.scatter_(1, sk.topk(s, -1).indices, 1.0)
                    mq = torch.zeros(T, a.K)
                    mq.scatter_(1, sq.topk(s, -1).indices, 1.0)
                    cand = ((mq @ mk.T) > 0) & causal
                    frac = cand.float().sum(-1) / causal.float().sum(-1)
                    rnd = (torch.rand(T, T, generator=g) < frac[:, None]) & causal
                    acc[s][h] += [float((w * cand).sum(-1).sum()), float(frac.sum()), float((w * rnd).sum(-1).sum()), T]
        ent /= len(docs) - a.fit
    print("katman %d, K %d, olcu %d belge | head: entropi (nat), s'ye gore [yakalanan kutle / aday orani / rastgele taban]" % (
        a.layer, a.K, len(docs) - a.fit))
    for h in range(H):
        print("H%-2d ent %4.2f  " % (h, ent[h]) + "  ".join("s%-2d %.3f / %.3f / %.3f" % (
            s, acc[s][h, 0] / acc[s][h, 3], acc[s][h, 1] / acc[s][h, 3], acc[s][h, 2] / acc[s][h, 3]) for s in S_LIST), flush=True)
    for s in S_LIST:
        A = acc[s]
        print("ortalama s%-2d: yakalanan kutle %.3f, aday orani %.3f, rastgele %.3f" % (
            s, (A[:, 0] / A[:, 3]).mean(), (A[:, 1] / A[:, 3]).mean(), (A[:, 2] / A[:, 3]).mean()))


if __name__ == "__main__":
    main()
