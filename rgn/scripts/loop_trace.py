"""Döngü teşhisi: tekrar eden devamı hangi bileşen üretiyor? (kullanıcı, 16 Ekim: "döngüyü kırmak kolay çünkü adres belli")

    python scripts/loop_trace.py --run fw_s000_d512_tf_yg_ozel_rms --prompt "The little dog"

Açgözlü üretim; döngüye girdiyse (aynı n-gram >= 3 kez) döngü içindeki konumlarda modelin seçtiği token'ın logit'i
bileşenlere ayrılır (doğrudan logit katkısı: son RMSNorm'un ölçeği sabit tutulur): gömme, katman başına dikkat, ortak MLP
ve kelimeye özel nöron (o konumdaki token'ın kendi satırı).  Katkı = (bileşen / son_rms) · g_norm · W_U[seçilen]
— ikinci aday token'a göre fark da verilir (döngüyü seçtiren üstünlük).  Ayrıca hangi kelimelerin özel satırının
döngüyü en çok beslediği.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rgn.paths import drive_root  # noqa: E402


def loop_span(ids, n=4, k=3):
    """İlk n-gram'ın >= k kez geçtiği döngünün konumları (yoksa None)."""
    seen = defaultdict(list)
    for i in range(len(ids) - n + 1):
        seen[tuple(ids[i:i + n])].append(i)
    rep = [v for v in seen.values() if len(v) >= k]
    if not rep:
        return None
    start = min(v[0] for v in rep)
    return start


@torch.no_grad()
def decompose(m, idx):
    """-> bileşenler sözlüğü: ad -> (T, d) akışa yazılan, son rms (T,)."""
    T = idx.shape[1]
    rope = m.rope(T)
    x = m.embed(idx)
    comp = {"gömme": x[0].clone()}
    rows = m.priv_map[idx] if m.priv_map is not None else None
    for li, blk in enumerate(m.blocks):
        a = blk.attn(blk.norm1(x), rope)
        s = x + a
        xn = blk.norm2(s)
        mo = blk.mlp(xn)
        comp["K%d %s dikkat" % (li + 1, blk.kind)] = a[0]
        comp["K%d %s MLP" % (li + 1, blk.kind)] = mo[0]
        out = s + mo
        if blk.private is not None:
            po = blk._private(xn, rows)
            comp["K%d %s özel" % (li + 1, blk.kind)] = po[0]
            out = out + po
        x = out
    rms = x[0].pow(2).mean(-1).add(m.norm_f.eps if hasattr(m.norm_f, "eps") else 1e-6).sqrt()
    return comp, rms, m.lm_head(m.norm_f(x))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fw_s000_d512_tf_yg_ozel_rms")
    ap.add_argument("--prompt", default="The little dog")
    ap.add_argument("--new", type=int, default=64)
    a = ap.parse_args()
    if "RGN_RUNS" not in os.environ and not Path("/content").is_dir():
        os.environ["RGN_RUNS"] = str(drive_root() / "RGN" / "runs")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from attn_anatomy import load
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load(a.run, dev)
    m.float()
    p = enc.encode_ordinary(a.prompt)
    full = m.generate(torch.tensor([p], device=dev), a.new)[0].tolist()
    gen = full[len(p):]
    print("== %s (adım %s)\n%r" % (a.run, step, enc.decode(full)))
    st = loop_span(gen)
    if st is None:
        print("döngü yok")
        return
    idx = torch.tensor([full], device=dev)
    comp, rms, logits = decompose(m, idx)
    g = m.norm_f.weight
    pos = list(range(len(p) + st, len(full) - 1))                          # döngü içi konumlar (tahmin: pos -> pos+1)
    names = list(comp)
    tot = torch.zeros(len(names))
    margin = torch.zeros(len(names))
    priv_by_tok = Counter()
    for t in pos:
        chosen = full[t + 1]
        second = int(logits[t].topk(2).indices[1]) if int(logits[t].argmax()) == chosen else int(logits[t].argmax())
        wu = m.lm_head.weight
        for j, nme in enumerate(names):
            v = comp[nme][t] / rms[t] * g
            c1, c2 = float(v @ wu[chosen]), float(v @ wu[second])
            tot[j] += c1
            margin[j] += c1 - c2
            if "özel" in nme:
                priv_by_tok[enc.decode([full[t]])] += c1 - c2
    n = len(pos)
    print("\ndöngü: üretimin %d. token'ından itibaren, %d konum; seçilen token'ın logit'i ve ikinci adaya üstünlüğü (ortalama)" % (
        st, n))
    print("%-16s %12s %18s" % ("bileşen", "logit katkısı", "ikinciye üstünlük"))
    order = margin.argsort(descending=True)
    for j in order.tolist():
        print("%-16s %12.2f %18.2f" % (names[j], tot[j] / n, margin[j] / n))
    grp = Counter()
    for j, nme in enumerate(names):
        key = "özel" if "özel" in nme else "dikkat" if "dikkat" in nme else "MLP" if "MLP" in nme else "gömme"
        grp[key] += float(margin[j] / n)
    print("\ntoplam üstünlük payı: %s" % "  ".join("%s %+.2f" % kv for kv in grp.most_common()))
    if priv_by_tok:
        print("döngüyü en çok besleyen özel satırlar (kelime: toplam üstünlük): %s" % "  ".join(
            "%r %+.1f" % kv for kv in priv_by_tok.most_common(6)))


if __name__ == "__main__":
    main()
