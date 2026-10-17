"""Cümle uzunluğu dağılımı (token cinsinden) — yerel pencere boyu için ölçü.

Cümle sonu: çözülmüş metni (sondaki tırnak / parantez atılınca) '.', '!' ya da '?' ile biten token, satır sonu içeren
token, belge sonu (EOS).  Kısaltmalar ("Mr.", "U.S.") ve ondalık noktalar fazladan cümle sonu sayılır → uzunluklar
biraz kısa çıkar (alt sınır gibi okunmalı).

Ölçüler: cümle uzunluğu dağılımı; token başına "cümle içindeki konumu" (cümle başına uzaklık) — pencere w bunu
kapsıyorsa o token kendi cümlesinin başını görür; ayrıca önceki cümlenin başına uzaklık (iki cümle).

    python scripts/sentence_lengths.py --dataset fineweb_s000 --tokens 5000000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.data.registry import EOS_ID  # noqa: E402

WINDOWS = (16, 32, 64, 128, 256)


def end_mask(toks, enc):
    """Cümle sonu token'ları (bool dizi)."""
    uniq = np.unique(toks)
    ends = set()
    for t in uniq.tolist():
        if t == EOS_ID:
            ends.add(t)
            continue
        s = enc.decode([t])
        if "\n" in s or s.rstrip(" \"'”’)]").endswith((".", "!", "?")):
            ends.add(t)
    return np.isin(toks, np.array(sorted(ends), dtype=toks.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--tokens", type=int, default=5_000_000)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    toks = np.asarray(valid_tokens(a.dataset, a.tokens)).astype(np.int64)
    report(toks, enc, a.dataset)


def report(toks, enc, name=""):
    end = end_mask(toks, enc)
    idx = np.flatnonzero(end)
    starts = np.concatenate([[0], idx[:-1] + 1])                       # cümle başları (son cümle sonuna kadar)
    lens = idx - starts + 1
    lens = lens[lens > 1]                                              # art arda satır sonları vb. tek token'lık "cümleler"
    pct = np.percentile(lens, [10, 25, 50, 75, 90, 95, 99])
    print("%s: %d token, %d cümle (2+ token)" % (name, len(toks), len(lens)))
    print("cümle uzunluğu (token): ort %.1f | %s" % (lens.mean(), "  ".join(
        "%%%d %d" % (p, v) for p, v in zip((10, 25, 50, 75, 90, 95, 99), pct))))
    print("token ağırlıklı ortalama (bir token'ın içinde bulunduğu cümlenin boyu): %.1f" % ((lens ** 2).sum() / lens.sum()))
    # token başına cümle başına uzaklık ve önceki cümlenin başına uzaklık
    sid = np.concatenate([[0], np.cumsum(end[:-1])])                  # token'ın cümle sırası
    sstart = np.concatenate([[0], idx + 1])                            # cümle sırası -> başlangıç konumu
    pos = np.arange(len(toks))
    d1 = pos - sstart[sid]
    d2 = pos - sstart[np.maximum(sid - 1, 0)]
    print("\npencere w: kendi cümlesinin başını gören token %% | önceki cümlenin başını da gören %%")
    for w in WINDOWS:
        print("  w %4d   %5.1f%%   %5.1f%%" % (w, 100 * (d1 < w).mean(), 100 * (d2 < w).mean()))


if __name__ == "__main__":
    main()
