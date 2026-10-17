"""s000'dan olgu sorusu çıkarır: eğitim metninde olgu kalıplarını (regex) bulur, çelişkili / belirsiz olanları atar, her
olgunun s000'da kaç kez geçtiğini sayar, aynı ilişkiden aynı token uzunluğunda 3 yanlış seçenekle soru yazar.

    python scripts/mine_facts.py --out docs/olgu_s000.txt
    python scripts/mine_facts.py --data v2/fineweb_edu_parts/s001/gpt2/train.npy \
        --exclude v2/fineweb_edu_s000/gpt2/train.npy --out docs/olgu_s001_yeni.txt   # yalnız s001'de olan olgular

Çıktı (sekmeyle ayrılmış, gözle kontrol için): ilişki, geçme sayısı, istem, doğru, yanlış1 | yanlış2 | yanlış3, s000'dan
örnek cümle.  Sayı yalnız bu kalıbın geçişi (olgu başka biçimde de geçebilir).
"""
import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from rgn.data.registry import get_spec  # noqa: E402
from rgn.paths import drive_root  # noqa: E402

NAME = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"                       # büyük harfle başlayan kelime
PERSON = r"(?:%s )(?:[A-Z]\. )?(?:%s )?%s" % (NAME, NAME, NAME)   # 2–3 kelimelik ad (orta harf olabilir)
PLACE = r"%s(?: %s){0,2}" % (NAME, NAME)
TITLE = r"(?:The |A )?%s(?: (?:of|the|and|in|on|a|to|for|%s)){0,6}" % (NAME, NAME)
YEAR = r"(1[0-9]\d\d|20[0-2]\d)"
# ilişki -> (regex, istem kalıbı, özne grubu, cevap grubu)
RELATIONS = {
    "başkent": (re.compile(r"\bThe capital of (%s) is (%s)\b" % (PLACE, PLACE)), "The capital of {x} is", 1, 2),
    "başkent2": (re.compile(r"\b(%s) is the capital (?:city )?of (%s)\b" % (PLACE, PLACE)), "The capital of {x} is", 2, 1),
    "doğum yılı": (re.compile(r"\b(%s) was born in %s\b" % (PERSON, YEAR)), "{x} was born in", 1, 2),
    "ölüm yılı": (re.compile(r"\b(%s) died in %s\b" % (PERSON, YEAR)), "{x} died in", 1, 2),
    "yazar": (re.compile(r"\b(%s) was written by (%s)\b" % (TITLE, PERSON)), "{x} was written by", 1, 2),
    "kuruluş yılı": (re.compile(r"\b(%s) was founded in %s\b" % (TITLE, YEAR)), "{x} was founded in", 1, 2),
    "mucit": (re.compile(r"\b(?:The )?(%s) was invented by (%s)\b" % (TITLE, PERSON)), "The {x} was invented by", 1, 2),
    "konum": (re.compile(r"\b(%s) is located in (%s)\b" % (PLACE, PLACE)), "{x} is located in", 1, 2),
}
MERGE = {"başkent2": "başkent"}
STOP = {"The", "This", "That", "These", "Those", "He", "She", "It", "They", "We", "You", "I", "In", "On", "At", "When",
        "After", "Before", "His", "Her", "Its", "Their", "Our", "My", "Your", "There", "Here", "Then", "But", "And", "Or",
        "If", "As", "So", "While", "Since", "Although", "Who", "What", "Which", "Where", "Why", "How", "One", "A", "An",
        "Born", "Today", "Now", "Also", "Mr", "Mrs", "Ms", "Dr", "Saint", "St", "Lord", "King", "Queen", "Sir", "Chapter",
        "Figure", "Table", "Section", "Book", "Volume", "Part", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December", "Monday", "Sunday", "God", "Jesus", "Christ"}
BUCKETS = [(1, 1), (2, 4), (5, 20), (21, 10 ** 9)]


def ok_entity(s):
    w = s.split()
    return w[0] not in STOP and w[-1] not in STOP and len(s) <= 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/olgu_s000.txt")
    ap.add_argument("--chunk", type=int, default=20_000_000)
    ap.add_argument("--max_tokens", type=int, default=0, help="0: tamamı")
    ap.add_argument("--per_bucket", type=int, default=40, help="ilişki × kova başına en çok soru")
    ap.add_argument("--dominance", type=float, default=0.8, help="öznenin en sık cevabının payı en az")
    ap.add_argument("--data", default=get_spec("fineweb_s000").train[0], help="taranan token dosyası (Drive köküne göre)")
    ap.add_argument("--exclude", nargs="*", default=[], help="bu dosyalarda aynı kalıpla geçen (özne, cevap) çiftleri atılır")
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    def scan(path, pairs, example=None):
        arr = np.load(drive_root() / path, mmap_mode="r")
        n = a.max_tokens or len(arr)
        for i in range(0, n, a.chunk):
            text = enc.decode(np.asarray(arr[i:min(n, i + a.chunk)]).astype(np.int64).tolist())
            for rel, (rx, _, gx, gy) in RELATIONS.items():
                for mt in rx.finditer(text):
                    x, y = mt.group(gx).strip(), mt.group(gy).strip()
                    if not (ok_entity(x) and (y.isdigit() or ok_entity(y))) or x == y:
                        continue
                    key = (MERGE.get(rel, rel), x, y)
                    pairs[rel][(x, y)] += 1
                    if example is not None and key not in example:
                        s0, s1 = max(0, mt.start() - 80), min(len(text), mt.end() + 40)
                        example[key] = " ".join(text[s0:s1].split())
            print("%s: tarandı %dM / %dM token" % (path, min(n, i + a.chunk) // 1_000_000, n // 1_000_000), flush=True)

    pairs = {r: Counter() for r in RELATIONS}
    example = {}
    scan(a.data, pairs, example)
    seen = set()                                                    # --exclude dosyalarında geçen (ilişki, özne, cevap)
    for path in a.exclude:
        ex_pairs = {r: Counter() for r in RELATIONS}
        scan(path, ex_pairs)
        seen |= {(MERGE.get(r, r), x, y) for r, c in ex_pairs.items() for (x, y) in c}
    facts = defaultdict(dict)                                       # ilişki -> özne -> (cevap, sayı)
    for rel, c in pairs.items():
        r = MERGE.get(rel, rel)
        by_x = defaultdict(Counter)
        for (x, y), k in c.items():
            by_x[x][y] += k
        for x, ys in by_x.items():
            prev = facts[r].get(x)
            if prev:
                ys = ys + Counter({prev[0]: prev[1]})
            y, k = ys.most_common(1)[0]
            if k / sum(ys.values()) >= a.dominance:
                facts[r][x] = (y, k)
            else:
                facts[r].pop(x, None)
    rnd = random.Random(0)
    lines = []
    for rel, fx in facts.items():
        prompt = next(p for rr, (_, p, _, _) in RELATIONS.items() if MERGE.get(rr, rr) == rel)
        answers = sorted({y for y, _ in fx.values()})
        for lo, hi in BUCKETS:
            items = sorted((x, y, k) for x, (y, k) in fx.items() if lo <= k <= hi)
            rnd.shuffle(items)
            items = [it for it in items if (rel, it[0], it[1]) not in seen]       # --exclude'da geçenler çıkar
            for x, y, k in items[:a.per_bucket]:
                ny = len(enc.encode(" " + y))
                pool = [z for z in answers if z != y and len(enc.encode(" " + z)) == ny]
                if len(pool) < 3:
                    continue
                wrong = rnd.sample(pool, 3)
                ex = example.get((rel, x, y)) or example.get((rel, y, x)) or ""
                lines.append("\t".join([rel, str(k), prompt.format(x=x), y, " | ".join(wrong), ex]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head = "ilişki\tgeçme\tistem\tdoğru\tyanlış seçenekler\törnek (%s)" % a.data
    out.write_text(head + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    cnt = Counter((l.split("\t")[0], next(b for b in BUCKETS if b[0] <= int(l.split("\t")[1]) <= b[1])) for l in lines)
    print("\n%d soru -> %s%s" % (len(lines), out, "  (--exclude'da geçen %d çift atıldı)" % len(seen) if a.exclude else ""))
    for rel in facts:
        print("  %-14s %s" % (rel, "  ".join("%d–%s: %d" % (lo, hi if hi < 10 ** 9 else "∞", cnt[(rel, (lo, hi))])
                                          for lo, hi in BUCKETS)))


if __name__ == "__main__":
    main()
