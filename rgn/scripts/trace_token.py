"""Bir kelimeyi model içinde izle (yalnız gözlem; model değişmez).

    python scripts/trace_token.py --run ss_d512_tf_yg_ozel --word " cat"

Doğrulama metninde kelimenin geçtiği N yerde (önünde 200 token bağlam, ardında 56), katman katman:
    katkı          dikkat, ortak MLP ve kelimeye özel nöronun akışa yazdığının büyüklüğü / akışın büyüklüğü
    MLP adresi     o konumda en çok çalışan nöronlar (|SiLU(G)·U| ortalaması), adresin sabitliği (her geçişteki ilk 20
                   nöronla ortak ilk 20'nin Jaccard ortalaması; 1 = her bağlamda aynı nöronlar), en güçlü nöronun
                   yazdığı yönün çıkışta öne çıkardığı kelimeler
    okur / okunur  kelime hangi head'lerde neye bakıyor; sonraki token'lar hangi head'lerde ona bakıyor
    mantık merceği o katmandan sonra akış doğrudan çıkışa verilse "sonraki kelime" tahmini
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="bir kelimeyi model içinde izle")
    ap.add_argument("--run", default="ss_d512_tf_yg_epok")
    ap.add_argument("--word", default=" cat")
    ap.add_argument("--dataset", default="ss")
    ap.add_argument("--n", type=int, default=48, help="kelimenin geçtiği yer sayısı")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(a.word)
    assert len(ids) == 1, "kelime tek token olmalı: %r -> %s" % (a.word, ids)
    tid = ids[0]
    dec = lambda t: enc.decode([int(t)])  # noqa: E731
    m, step = load_run(a.run)
    m.float()
    stream = valid_tokens(a.dataset, 6_000_000)
    pos = np.flatnonzero(stream == tid)
    pos = pos[(pos >= 200) & (pos < len(stream) - 57)]
    pos = pos[np.linspace(0, len(pos) - 1, min(a.n, len(pos))).astype(int)]
    W = torch.tensor(np.stack([stream[p - 200:p + 56] for p in pos]).astype(np.int64))
    c = 200                                                                  # kelimenin konumu
    print("== %s (adım %s) | %r (token %d) doğrulama metninde %d yerde, önünde 200 token bağlam" % (
        a.run, step, a.word, tid, len(pos)))
    rope = m.rope(W.shape[1])
    E = m.lm_head.weight
    with torch.no_grad():
        parts = []
        m.hidden(W, parts=parts)
        x = parts[0]["embed"]
        for li, (blk, d) in enumerate(zip(m.blocks, parts[1:])):
            att = blk.attn.weights(blk.norm1(x), rope)                       # (B, H, T, T)
            s = x + d["attn"]
            r = x[:, c].norm(dim=-1).mean()
            share = lambda v: 0.0 if v is None else (v[:, c].norm(dim=-1).mean() / r).item()  # noqa: E731
            priv = "" if d["private"] is None else "  özel %.2f" % share(d["private"])
            if blk.mlp is None:                                              # ortak MLP yok (shared_mlp False)
                print("\nK%d %s | katkı: dikkat %.2f%s (akış büyüklüğüne oran)" % (li + 1, blk.kind, share(d["attn"]), priv))
            else:
                g, up = blk.mlp.a(blk.norm2(s)).chunk(2, dim=-1)
                hc = (F.silu(g) * up)[:, c].abs()                                # nöron çıktıları (B, ff)
                mean = hc.mean(0)
                top = mean.topk(a.top).indices
                glob20 = set(mean.topk(20).indices.tolist())
                jac = np.mean([len(glob20 & set(r.topk(20).indices.tolist())) / len(glob20 | set(r.topk(20).indices.tolist()))
                               for r in hc])
                n0 = int(top[0])
                promo = (E @ blk.mlp.c.weight[:, n0]).topk(5).indices
                print("\nK%d %s | katkı: dikkat %.2f  MLP %.2f%s (akış büyüklüğüne oran) | MLP adresi sabitliği %.2f" % (
                    li + 1, blk.kind, share(d["attn"]), share(d["mlp"]), priv, jac))
                print("   MLP adresi (en çok çalışan nöronlar): %s" % "  ".join(
                    "%d:%.2f" % (int(i), mean[i].item()) for i in top))
                print("   en güçlü nöron %d akışa yazınca öne çıkan: %s" % (n0, " | ".join(repr(dec(t)) for t in promo)))
            row = att[:, :, c, :c + 1]                                       # kelime neye bakıyor
            col = att[:, :, c + 1:, c].mean(-1)                              # sonrakiler kelimeye ne kadar bakıyor
            reads = []
            for hh in range(row.shape[1]):
                tops = row[:, hh].argmax(-1)
                cnt = Counter(dec(W[b, j]) if j != c else "(kendisi)" for b, j in enumerate(tops.tolist()))
                w, n = cnt.most_common(1)[0]
                reads.append("h%d→%r %d/%d" % (hh, w, n, len(tops)))
            print("   okur (head başına en çok baktığı): %s" % "  ".join(reads))
            hm = col.mean(0)
            print("   okunur (sonraki 55 token'ın ona ortalama bakışı, head): %s" % "  ".join(
                "h%d %.3f" % (hh, hm[hh].item()) for hh in hm.argsort(descending=True)[:4]))
            x = s + (0 if d["mlp"] is None else d["mlp"]) + (0 if d["private"] is None else d["private"])
            p = torch.softmax(m.lm_head(m.norm_f(x[:, c])), -1).mean(0)
            pt = p.topk(4)
            print("   mantık merceği (sonraki kelime tahmini): %s" % "  ".join(
                "%r %.2f" % (dec(i), v.item()) for v, i in zip(pt.values, pt.indices)))
    nxt = Counter(dec(W[b, c + 1]) for b in range(len(W)))
    print("\ngerçekte sonra gelen: %s" % "  ".join("%r %d" % kv for kv in nxt.most_common(6)))


if __name__ == "__main__":
    main()
