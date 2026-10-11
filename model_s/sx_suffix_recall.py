"""Baglam ici kesin sonek eslesmesi (TARTISMA_attention Soz 4): her konum t'de x_<=t'nin belge icinde daha once gecen en uzun soneki
(uzunluk n, en cok 16) ve o eski gecisin ardili x_{t+1}'i dogru veriyor mu.  Veri yalniz (model yok).
Kullanim: python model_s/sx_suffix_recall.py [--docs 2000]"""
import argparse
import os

import torch  # noqa: F401  (Windows: pyarrow'dan once)
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stats(docs, maxn=16):
    """docs: token listeleri -> {n: [eslesen konum sayisi, dogru ardil sayisi]}, toplam konum."""
    hit = np.zeros((maxn + 1, 2))
    total = 0
    for d in docs:
        last = {}                                              # n-gram -> son bitis konumu (yalniz < t)
        for t in range(len(d) - 1):
            total += 1
            best = 0
            for n in range(min(maxn, t + 1), 0, -1):
                key = tuple(d[t - n + 1:t + 1])
                if key in last:
                    best = n
                    j = last[key]
                    break
            if best:
                hit[best, 0] += 1
                hit[best, 1] += d[j + 1] == d[t + 1]
            for n in range(1, min(maxn, t + 1) + 1):           # x_<=t'nin sonekleri artik "once gecti"
                last[tuple(d[t - n + 1:t + 1])] = t
    return hit, total


def report(name, hit, total):
    print("== %s: %d konum" % (name, total))
    m, c = hit[:, 0], hit[:, 1]
    for lo in (1, 2, 3, 4, 8):
        mm, cc = m[lo:].sum(), c[lo:].sum()
        print("  en uzun sonek >= %2d: konumlarin %%%.1f'i, ardil dogru %%%.1f -> tum konumlarin %%%.1f'i kesin eslesmeyle dogru" % (
            lo, 100 * mm / total, 100 * cc / max(mm, 1), 100 * cc / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    a = ap.parse_args()
    tok = Tokenizer.from_file(r"G:/Drive'ım/simplestories/gpt2/tokenizer.json")
    texts = pq.read_table(os.path.join(ROOT, "deneme2/onbellek/fineweb/sample.parquet")).column("text").to_pylist()[:a.docs]
    fw = [e.ids[:4096] for e in tok.encode_batch(texts)]
    report("FineWeb-edu (ilk %d belge, en cok 4.096 token)" % len(fw), *stats(fw))
    ss = np.load(r"G:/Drive'ım/simplestories/gpt2/valid.npy", mmap_mode="r")
    so = np.load(r"G:/Drive'ım/v2/simplestories_gpt2/valid_story_offsets.npy")
    se = np.load(r"G:/Drive'ım/v2/simplestories_gpt2/valid_sentence_offsets.npy")     # (cumle, [bas, son]); hikaye ofseti cumle sirasi
    sdocs = [ss[se[so[i], 0]:se[so[i + 1] - 1, 1]].tolist() for i in range(min(a.docs, len(so) - 1))]
    report("SS valid (ilk %d hikaye)" % len(sdocs), *stats(sdocs))


if __name__ == "__main__":
    main()
