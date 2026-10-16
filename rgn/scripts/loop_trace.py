"""Döngü teşhisi: tekrar eden devamı hangi bileşen üretiyor?

    python scripts/loop_trace.py --run fw_s000_d512_tf_yg_ozel_rms --prompt "The little dog"

Açgözlü üretim; döngüye girdiyse (aynı 4-gram >= 3 kez) döngü içindeki konumlarda seçilen token'ın logit'i bileşenlere
ayrılır (doğrudan logit katkısı, son RMSNorm ölçeği sabit): gömme, katman başına dikkat, ortak MLP ve kelimeye özel
nöron (o konumdaki token'ın kendi satırı).  Katkı = (bileşen / son_rms) · g_norm · W_U[seçilen]; ikinci adaya göre fark
"üstünlük" olarak verilir.  Ayrıca döngüyü en çok besleyen özel satırlar.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rgn.train.checkpoint import load_run  # noqa: E402


def loop_start(ids, n=4, k=3):
    """İlk n-gram'ın >= k kez geçtiği döngünün başı (yoksa None)."""
    seen = defaultdict(list)
    for i in range(len(ids) - n + 1):
        seen[tuple(ids[i:i + n])].append(i)
    rep = [v for v in seen.values() if len(v) >= k]
    return min(v[0] for v in rep) if rep else None


@torch.no_grad()
def components(m, idx):
    """-> (ad -> (T, d) akışa yazılan, son RMS (T,), logit'ler (T, V))."""
    parts = []
    h = m.hidden(idx, parts=parts)
    comp = {"gömme": parts[0]["embed"][0]}
    for li, (blk, d) in enumerate(zip(m.blocks, parts[1:])):
        comp["K%d %s dikkat" % (li + 1, blk.kind)] = d["attn"][0]
        comp["K%d %s MLP" % (li + 1, blk.kind)] = d["mlp"][0]
        if d["private"] is not None:
            comp["K%d %s özel" % (li + 1, blk.kind)] = d["private"][0]
    x = sum(comp.values())
    rms = x.pow(2).mean(-1).add(m.norm_f.eps).sqrt()
    return comp, rms, m.lm_head(h)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fw_s000_d512_tf_yg_ozel_rms")
    ap.add_argument("--prompt", default="The little dog")
    ap.add_argument("--new", type=int, default=64)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load_run(a.run, dev)
    m.float()
    p = enc.encode_ordinary(a.prompt)
    full = m.generate(torch.tensor([p], device=dev), a.new)[0].tolist()
    print("== %s (adım %s)\n%r" % (a.run, step, enc.decode(full)))
    st = loop_start(full[len(p):])
    if st is None:
        print("döngü yok")
        return
    comp, rms, logits = components(m, torch.tensor([full], device=dev))
    g, wu = m.norm_f.weight, m.lm_head.weight
    pos = list(range(len(p) + st, len(full) - 1))                      # döngü içi konumlar (tahmin: t -> t + 1)
    names = list(comp)
    tot, margin = torch.zeros(len(names)), torch.zeros(len(names))
    by_tok = Counter()
    for t in pos:
        chosen = full[t + 1]
        top2 = logits[t].topk(2).indices.tolist()
        second = top2[1] if top2[0] == chosen else top2[0]
        for j, nme in enumerate(names):
            v = comp[nme][t] / rms[t] * g
            c1, c2 = float(v @ wu[chosen]), float(v @ wu[second])
            tot[j] += c1
            margin[j] += c1 - c2
            if "özel" in nme:
                by_tok[enc.decode([full[t]])] += c1 - c2
    n = len(pos)
    print("\ndöngü: üretimin %d. token'ından, %d konum; seçilen token'ın logit'i ve ikinci adaya üstünlüğü (ortalama)" % (st, n))
    print("%-16s %12s %18s" % ("bileşen", "logit katkısı", "ikinciye üstünlük"))
    for j in margin.argsort(descending=True).tolist():
        print("%-16s %12.2f %18.2f" % (names[j], tot[j] / n, margin[j] / n))
    grp = Counter()
    for j, nme in enumerate(names):
        grp[next(k for k in ("özel", "dikkat", "MLP", "gömme") if k in nme)] += float(margin[j] / n)
    print("\ntoplam üstünlük payı: %s" % "  ".join("%s %+.2f" % kv for kv in grp.most_common()))
    if by_tok:
        print("döngüyü en çok besleyen özel satırlar: %s" % "  ".join("%r %+.1f" % kv for kv in by_tok.most_common(6)))


if __name__ == "__main__":
    main()
