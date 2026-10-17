"""Olgu sorularını bir referans veride (ör. s000) olgunun geçip geçmediğine göre ikiye ayırır.

mine_facts --exclude yalnız aynı kalıbı (ör. "The capital of X is Y") atar; olgu başka biçimde ("Y, the capital of X")
geçiyorsa soru kalır.  Bu betik her soru için referans metinde özne ile cevabın yakın geçtiğini (aradaki uzaklık
≤ --window karakter, iki yönde) arar:
  saf   : hiç birlikte geçmiyor → referansla eğitilen model olguyu görmedi (yeni bilgi)
  biçim : başka biçimde birlikte geçiyor → model olguyu gördü ama soru kalıbını görmedi (biçimden bağımsız öğrenme)
Yakınlık kaba bir ölçü: birlikte geçmek olguyu söylemek demek değil (ör. "Thailand ... Bangkok" komşu cümlelerde);
"biçim" grubu bu yüzden üst sınır.

Çıktı: her girdi dosyası için <ad>_saf.txt ve <ad>_bicim.txt (aynı biçim; eval_all --facts ile okunur).

    python scripts/fact_split.py --facts /content/drive/MyDrive/RGN/olgu/olgu_s001_yeni.txt \
        /content/drive/MyDrive/RGN/olgu/olgu_s002_yeni.txt
"""
import argparse
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402

from mine_facts import RELATIONS  # noqa: E402

TEMPLATES = sorted({p for _, p, _, _ in RELATIONS.values()}, key=len, reverse=True)


def subject(prompt):
    """İstemden özne: şablonlardan ("{x} was born in") geri çıkarılır."""
    for t in TEMPLATES:
        m = re.fullmatch(re.escape(t).replace(re.escape("{x}"), "(.+)"), prompt)
        if m:
            return m.group(1)
    raise ValueError("şablon bulunamadı: %r" % prompt)


def read(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [l for l in lines[1:] if len(l.split("\t")) >= 5]
    return lines[0], rows


def cooccur(texts, pairs, window):
    """texts: metin parçaları (yineleyici); pairs: {(x, y)} -> birlikte geçenlerin kümesi."""
    by_x = {}
    for x, y in pairs:
        by_x.setdefault(x, set()).add(y)
    rx = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(x) for x in sorted(by_x, key=len, reverse=True)))
    found = set()
    for text in texts:
        for m in rx.finditer(text):
            x = m.group(0)
            ys = by_x.get(x, set())
            if not ys:
                continue
            ctx = text[max(0, m.start() - window):m.end() + window]
            for y in ys:
                if (x, y) not in found and re.search(r"\b%s\b" % re.escape(y), ctx):
                    found.add((x, y))
    return found


def chunks(path, chunk, max_tokens=0):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    arr = np.load(path, mmap_mode="r")
    n = max_tokens or len(arr)
    for i in range(0, n, chunk):
        yield enc.decode(np.asarray(arr[max(0, i - 200):min(n, i + chunk)]).astype(np.int64).tolist())  # 200 token örtüşme
        print("%s: tarandı %dM / %dM token" % (path, min(n, i + chunk) // 1_000_000, n // 1_000_000), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", nargs="+", required=True, help="mine_facts çıktıları")
    ap.add_argument("--data", default="v2/fineweb_edu_s000/gpt2/train.npy", help="referans token dosyası (Drive köküne göre "
                                                                              "ya da mutlak)")
    ap.add_argument("--window", type=int, default=300, help="özne ile cevap arası en çok karakter (~64 token)")
    ap.add_argument("--chunk", type=int, default=20_000_000)
    ap.add_argument("--max_tokens", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data)
    if not data.is_absolute():
        from rgn.paths import drive_root
        data = drive_root() / data
    files = {f: read(f) for f in a.facts}
    pairs = {(subject(r.split("\t")[2]), r.split("\t")[3]) for _, rows in files.values() for r in rows}
    found = cooccur(chunks(data, a.chunk, a.max_tokens), pairs, a.window)
    for f, (head, rows) in files.items():
        p = Path(f)
        saf = [r for r in rows if (subject(r.split("\t")[2]), r.split("\t")[3]) not in found]
        bic = [r for r in rows if (subject(r.split("\t")[2]), r.split("\t")[3]) in found]
        for tag, rr in (("saf", saf), ("bicim", bic)):
            (p.parent / ("%s_%s%s" % (p.stem, tag, p.suffix))).write_text(head + "\n" + "\n".join(rr) + "\n", encoding="utf-8")
        print("%s: %d soru -> saf %d, biçim %d (%s içinde ≤ %d karakter birlikte)" % (
            p.name, len(rows), len(saf), len(bic), data.name, a.window))


if __name__ == "__main__":
    main()
