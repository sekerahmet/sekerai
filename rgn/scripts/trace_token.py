"""Bir kelimeyi model içinde izle (kullanıcı, 16 Ekim: "cat'i ben sistemde izleyeyim").

    python scripts/trace_token.py --run ss_d512_tf_yg_epok --word " cat"

Model ve eğitim değişmez; yalnız gözlem.  Doğrulama metninde kelimenin geçtiği N yerde (önünde 200 token bağlam, ardında
56) katman katman:
    MLP adresi   : o konumda en çok çalışan nöronlar (|f(G)·U| ortalaması), adresin sabitliği (her geçişteki ilk 20 nöron
                   ile ortak ilk 20'nin Jaccard ortalaması; 1 = her bağlamda aynı nöronlar), en güçlü nöronun akışa yazdığı
                   yönün çıkış katmanında öne çıkardığı kelimeler
    dikkat (okur): kelime hangi head'lerde neye bakıyor (en çok bakılan önceki token'lar)
    dikkat (okunur): sonraki token'lar hangi head'lerde bu kelimeye bakıyor
    mantık merceği: o katmandan sonra akış doğrudan çıkışa verilse modelin "sonraki kelime" tahmini
    katkı        : dikkat ve MLP'nin akışa yazdığının büyüklüğü / akışın büyüklüğü
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.data.loader import _open  # noqa: E402
from rgn.data.registry import get_spec  # noqa: E402
from rgn.paths import drive_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="bir kelimeyi model içinde izle")
    ap.add_argument("--run", default="ss_d512_tf_yg_epok")
    ap.add_argument("--word", default=" cat")
    ap.add_argument("--n", type=int, default=48, help="kelimenin geçtiği yer sayısı")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    if "RGN_RUNS" not in os.environ and not Path("/content").is_dir():
        os.environ["RGN_RUNS"] = str(drive_root() / "RGN" / "runs")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attn_anatomy import load
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(a.word)
    assert len(ids) == 1, "kelime tek token olmalı: %r -> %s" % (a.word, ids)
    tid = ids[0]
    dec = lambda t: enc.decode([int(t)])  # noqa: E731
    m, step = load(a.run, torch.device("cpu"))
    m.float()
    spec = get_spec("ss")
    stream = np.asarray(_open(drive_root() / spec.valid[0], spec.fmt)[:6_000_000])
    pos = np.flatnonzero(stream == tid)
    pos = pos[(pos >= 200) & (pos < len(stream) - 57)]
    pos = pos[np.linspace(0, len(pos) - 1, min(a.n, len(pos))).astype(int)]
    W = torch.tensor(np.stack([stream[p - 200:p + 56] for p in pos]).astype(np.int64))
    c = 200                                                                  # kelimenin konumu
    print("== %s (adım %s) | %r (token %d) doğrulama metninde %d yerde, önünde 200 token bağlam" % (
        a.run, step, a.word, tid, len(pos)))
    T = W.shape[1]
    rope = m.rope(T)
    E = m.lm_head.weight
    with torch.no_grad():
        x = m.embed(W)
        for li, blk in enumerate(m.blocks):
            u = blk.norm1(x)
            att = blk.attn.weights(u, rope)                                  # (B, H, T, T)
            ao = blk.attn(u, rope)
            s = x + ao
            g, up = blk.mlp.a(blk.norm2(s)).chunk(2, dim=-1)
            h = F.silu(g) * up                                               # nöron çıktıları (B, T, ff)
            mo = blk.mlp.c(h)
            xn = s + mo
            hc = h[:, c].abs()                                               # (B, ff)
            mean = hc.mean(0)
            top = mean.topk(a.top).indices
            glob20 = set(mean.topk(20).indices.tolist())
            jac = np.mean([len(glob20 & set(r.topk(20).indices.tolist())) / len(glob20 | set(r.topk(20).indices.tolist()))
                           for r in hc])
            n0 = int(top[0])
            promo = (E @ blk.mlp.c.weight[:, n0]).topk(5).indices
            r = x[:, c].norm(dim=-1).mean()
            print("\nK%d %s | katkı: dikkat %.2f  MLP %.2f (akış büyüklüğüne oran) | MLP adresi sabitliği %.2f" % (
                li + 1, blk.kind, (ao[:, c].norm(dim=-1).mean() / r).item(), (mo[:, c].norm(dim=-1).mean() / r).item(), jac))
            print("   MLP adresi (en çok çalışan nöronlar): %s" % "  ".join(
                "%d:%.2f" % (int(i), mean[i].item()) for i in top))
            print("   en güçlü nöron %d akışa yazınca öne çıkan: %s" % (n0, " | ".join(repr(dec(t)) for t in promo)))
            row = att[:, :, c, :c + 1]                                       # kelime neye bakıyor (B, H, c+1)
            col = att[:, :, c + 1:, c].mean(-1)                              # sonrakiler kelimeye ne kadar bakıyor (B, H)
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
            p = torch.softmax(m.lm_head(m.norm_f(xn[:, c])), -1).mean(0)
            pt = p.topk(4)
            print("   mantık merceği (sonraki kelime tahmini): %s" % "  ".join(
                "%r %.2f" % (dec(i), v.item()) for v, i in zip(pt.values, pt.indices)))
            x = xn
    nxt = Counter(dec(W[b, c + 1]) for b in range(len(W)))
    print("\ngerçekte sonra gelen: %s" % "  ".join("%r %d" % kv for kv in nxt.most_common(6)))


if __name__ == "__main__":
    main()
