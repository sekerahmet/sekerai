"""make_ss_sentences -- gramer ajani icin SimpleStories cumleleri, kelime duzeyinde (kullanici, 3 Ekim: "ss içinde çok
sayıda örnek ile deneme yapabiliriz"; "İsimler ok"; "orda etiket vardı şimdi sadece cümle ayırmak için gerekli").

Cumle siniri: model_y SS'de sinanmis kural (f4d2a63, data_simplestories._sentence_starts; 5.000 hikayede degismezler ve
gozle: bos cumle 0, diyalog '"Why?" Alice asked' tek cumle, 'The dog said, "Woof!"' yeni cumle, 'Mr. Luis' ve ondalik
sinir degil, paragraf basi tirnak cumlesinde).  Yalniz SINIR alinir, etiket yok.  Girdi Drive'daki hazir gpt2 akislari
(<root>/gpt2/train.npy, valid.npy = SS train / test); cumle metni token'lardan cozulur, sonra kelimelere ayrilir.
Kelime: harf dizisi (kesme isareti dahil: don't, Mia's), sayi ya da tek noktalama; buyuk / kucuk harf korunur.
Egitim: train akisinin ilk hikayelerinden TRAIN cumle (tohumlu ornek).  Sinav: test hikayelerinden 'unseen' EXAM cumle
(egitimde gecmemis, butun kelimeleri egitim sozlugunde); 'seen' egitimden SEEN cumle.  Hazir veri: bir kez uretilir.

    python make_ss_sentences.py <simplestories koku> <cikti klasoru>
"""
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np

TRAIN = 200_000
EXAM = 5_000
SEEN = 1_000
WORDS_MIN, WORDS_MAX = 3, 40
TRAIN_TOKENS = 30_000_000   # train akisinin ilk bu kadar token'i (~4 x TRAIN cumle)
SEED = 0
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")

# --- f4d2a63 data_simplestories: cumle siniri (aynen; etiket kismi alinmadi)
_SPEECH = frozenset("""said asked whispered shouted replied cried exclaimed called yelled answered added thought wondered
muttered murmured screamed declared announced explained insisted begged pleaded shouts says asks replies cries calls
whispers continued agreed admitted suggested""".split())
_TITLES = frozenset(("Mr", "Mrs", "Ms", "Dr", "St", "Prof", "Sr", "Jr"))
_ATTRIBUTION = re.compile(r"^\s*(?:The\s+(?:\w+\s+)?\w+|[A-Z][\w']*(?:\s+[A-Z][\w']*)?)\s+(\w+)\b(?!\s*,\s*[\"“])")
_FLAG = dict(end=1, dot=2, digit=4, title=8, space=16, closer=32, lines=64, lower=128, quote=256, speech=512)


def _tables(tok, n):
    text = [tok.decode([i]) for i in range(n)]
    speech = np.zeros(n, dtype=bool)
    speech[[tok.encode(p + w, add_special_tokens=False).ids[0] for w in _SPEECH for p in (" ", "")]] = True
    closers = "\"')]}”’"
    tab = dict(end=np.array([("\n" in s) or s.rstrip().rstrip(closers).endswith((".", "!", "?")) for s in text])
               & ~np.array([s.strip().endswith(".") and s.strip()[:-1] in _TITLES for s in text]),
               dot=np.array([s.strip() == "." for s in text]), digit=np.array([s[:1].isdigit() for s in text]),
               lines=np.array(["\n" in s for s in text]), lower=np.array([s.lstrip()[:1].islower() for s in text]),
               closer=np.array([bool(s) and s == s.lstrip() and not s.strip(closers) for s in text]),
               space=np.array([not s.strip() for s in text]),
               quote=np.array([any(q in s for q in '"“”') for s in text]),
               title=np.array([s.strip() in _TITLES for s in text]), speech=speech)
    return sum(tab[k].astype(np.uint16) * v for k, v in _FLAG.items()).astype(np.uint16)


def sentence_starts(x, flags, tok, eos):
    """Hikayeler (eos ile biten akis) -> cumle baslangiclari (f4d2a63 _sentence_starts)."""
    f = flags[x]
    on = lambda a, name: (a & _FLAG[name]) != 0
    fn, fp = np.append(f[1:], flags[eos]), np.insert(f[:-1], 0, flags[eos])
    not_eos = x != eos
    e = on(f, "end") & ~(on(f, "dot") & (on(fn, "digit") | on(fp, "title")))
    grow = (on(f, "space") | (on(f, "closer") & ~on(fp, "space") & ~on(fp, "lines"))) & not_eos
    while True:
        more = grow[1:] & e[:-1] & ~e[1:]
        if not more.any():
            break
        e[1:] |= more
    cut = np.flatnonzero(e & ~np.append(e[1:], False) & ~on(fn, "lower") & np.append(not_eos[1:], False) & not_eos)
    first = np.flatnonzero(e & ~np.r_[False, e[:-1]])
    run0 = first[np.searchsorted(first, cut, "right") - 1]
    quoted, broken = np.zeros(len(cut), dtype=bool), np.zeros(len(cut), dtype=bool)
    for d in range(int((cut - run0).max()) + 1 if len(cut) else 0):
        inside = run0 + d <= cut
        g = f[np.minimum(run0 + d, cut)]
        quoted |= inside & on(g, "quote")
        broken |= inside & on(g, "lines")
    talk = cut[quoted & ~broken]
    pad = np.append(f, [flags[eos]] * 9)
    talk = talk[np.any([on(pad[talk + k], "speech") for k in range(1, 9)], axis=0)] if len(talk) else talk
    merged = []
    for i in talk.tolist():
        w = x[i + 1:i + 13].tolist()
        w = w[:w.index(eos)] if eos in w else w
        m = _ATTRIBUTION.match(tok.decode(w))
        if m and m.group(1).lower() in _SPEECH:
            merged.append(i)
    heads = np.flatnonzero(np.r_[True, ~not_eos[:-1]] & not_eos)
    return np.union1d(heads, np.setdiff1d(cut, merged) + 1)


def stream_sentences(a, flags, tok, eos):
    """Akis -> cumle metinleri (hikaye sirasiyla)."""
    a = np.asarray(a).astype(np.int64)
    S = sentence_starts(a, flags, tok, eos)
    ends = np.flatnonzero(a == eos)
    stop = np.minimum(np.r_[S[1:], len(a)], ends[np.searchsorted(ends, S)])   # cumle sonu: sonraki cumle ya da eos
    texts = tok.decode_batch([a[s:t].tolist() for s, t in zip(S.tolist(), stop.tolist())])
    return [t.strip() for t in texts if t.strip()]


def main(root, out):
    from tokenizers import Tokenizer
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(root, "gpt2", "tokenizer.json"))
    eos = tok.token_to_id("<|endoftext|>")
    flags = _tables(tok, tok.get_vocab_size())
    rng = random.Random(SEED)
    train_a = np.load(os.path.join(root, "gpt2", "train.npy"), mmap_mode="r")
    cut = int(np.flatnonzero(np.asarray(train_a[:TRAIN_TOKENS]) == eos)[-1]) + 1
    rows, seen = [], set()
    for s in stream_sentences(train_a[:cut], flags, tok, eos):
        w = _WORD.findall(s)
        if WORDS_MIN <= len(w) <= WORDS_MAX and s not in seen:
            seen.add(s)
            rows.append(dict(sentence=s, words=w))
    train = rng.sample(rows, min(TRAIN, len(rows)))
    vocab = Counter(w for r in train for w in r["words"])
    train_text = {r["sentence"] for r in train}
    test = stream_sentences(np.load(os.path.join(root, "gpt2", "valid.npy")), flags, tok, eos)
    rng.shuffle(test)
    exam = []
    for s in test:
        w = _WORD.findall(s)
        if WORDS_MIN <= len(w) <= WORDS_MAX and s not in train_text and all(x in vocab for x in w):
            exam.append(dict(sentence=s, words=w, split="unseen"))
            train_text.add(s)
        if len(exam) >= EXAM:
            break
    exam += [dict(r, split="seen") for r in rng.sample(train, SEEN)]
    for name, rs in (("ss_train", [dict(r, split="train") for r in train]), ("ss_exam", exam)):
        with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = Counter(min(len(r["words"]) // 10, 4) for r in train)
    print("aday cumle %d | egitim %d, sozluk %d kelime, boy (10'luk: 0 = 3-9 ...) %s | sinav %s" % (
        len(rows), len(train), len(vocab), dict(sorted(lens.items())), dict(Counter(r["split"] for r in exam))))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
