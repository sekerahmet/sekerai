"""make_ss_sentences -- gramer ajani icin SimpleStories cumleleri, kelime duzeyinde (kullanici, 3 Ekim: "ss içinde çok
sayıda örnek ile deneme yapabiliriz"; "İsimler ok"; "orda etiket vardı şimdi sadece cümle ayırmak için gerekli";
"bence ss hepsini eğitim yapalım").

Cumle siniri: model_y SS'de sinanmis kural (f4d2a63, data_simplestories._sentence_starts; 5.000 hikayede degismezler ve
gozle: bos cumle 0, diyalog '"Why?" Alice asked' tek cumle, 'The dog said, "Woof!"' yeni cumle, 'Mr. Luis' ve ondalik
sinir degil, paragraf basi tirnak cumlesinde).  Ek: acik tirnagin icinde kesilmez (senior-developer S6: '"Help! Help!"'
ortadan bolunuyordu); tirnak sayaci satir sonunda ve hikaye sonunda sifirlanir.  Yalniz SINIR alinir, etiket yok.
Girdi Drive'daki hazir gpt2 akislari (<root>/gpt2/train.npy, valid.npy = SS train / test); akis parca parca (hikaye
sinirinda) islenir.  Kelime: harf dizisi (kesme isareti dahil: don't, Mia's), sayi ya da tek noktalama; buyuk / kucuk
harf korunur.

Egitim: train akisinin TAMAMI (tekrar eden cumle bir kez).  Sinav: test hikayelerinden 'unseen' EXAM cumle (egitimde
gecmemis, butun kelimeleri egitim sozlugunde); 'seen' egitimden SEEN cumle.  Her sinav cumlesine 'valid': ayni torbadan
egitimde ya da sinavda gecen butun siralar.  Hazir veri: bir kez uretilir.
Cikti: ss_vocab.json (indeks 0 '<unk>'), ss_train_ids.npy (int32, cumleler uc uca), ss_train_offsets.npy (int64, N+1),
ss_exam.jsonl.

    python make_ss_sentences.py <simplestories koku> <cikti klasoru> [--tokens N  (yalniz ilk N token; sinama)]
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

EXAM = 5_000
SEEN = 1_000
WORDS_MIN, WORDS_MAX = 3, 40
CHUNK = 20_000_000      # akis bu kadar token'lik parcalarla (hikaye sinirinda) islenir
WORKERS = 6             # paralel surec (bellek: parca basina ~1-2 GB)
SEED = 0
UNK = "<unk>"
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")

# --- f4d2a63 data_simplestories: cumle siniri (aynen; etiket kismi alinmadi)
_SPEECH = frozenset("""said asked whispered shouted replied cried exclaimed called yelled answered added thought wondered
muttered murmured screamed declared announced explained insisted begged pleaded shouts says asks replies cries calls
whispers continued agreed admitted suggested""".split())
_TITLES = frozenset(("Mr", "Mrs", "Ms", "Dr", "St", "Prof", "Sr", "Jr"))
_ATTRIBUTION = re.compile(r"^\s*(?:The\s+(?:\w+\s+)?\w+|[A-Z][\w']*(?:\s+[A-Z][\w']*)?)\s+(\w+)\b(?!\s*,\s*[\"“])")
_FLAG = dict(end=1, dot=2, digit=4, title=8, space=16, closer=32, lines=64, lower=128, quote=256, speech=512)


def _tables(tok, n):
    """-> (bayraklar, token basina tirnak sayisi)."""
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
    flags = sum(tab[k].astype(np.uint16) * v for k, v in _FLAG.items()).astype(np.uint16)
    quotes = np.array([sum(s.count(q) for q in '"“”') for s in text], dtype=np.int64)
    return flags, quotes


def sentence_starts(x, flags, quotes, tok, eos):
    """Hikayeler (eos ile biten akis) -> cumle baslangiclari (f4d2a63 _sentence_starts + acik tirnakta kesme yok)."""
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
    count = np.cumsum(quotes[x])                       # acik tirnak: satir / hikaye basindan beri tek sayida tirnak
    last = np.maximum.accumulate(np.where(on(f, "lines") | ~not_eos, np.arange(len(x)), -1))
    base = np.where(last >= 0, count[np.maximum(last, 0)], 0)
    cut = cut[(count[cut] - base[cut]) % 2 == 0]
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


def stream_sentences(a, flags, quotes, tok, eos):
    """Akis -> cumle metinleri (hikaye sirasiyla)."""
    a = np.asarray(a).astype(np.int64)
    S = sentence_starts(a, flags, quotes, tok, eos)
    ends = np.flatnonzero(a == eos)
    stop = np.minimum(np.r_[S[1:], len(a)], ends[np.searchsorted(ends, S)])   # cumle sonu: sonraki cumle ya da eos
    texts = tok.decode_batch([a[s:t].tolist() for s, t in zip(S.tolist(), stop.tolist())])
    return [t.strip() for t in texts if t.strip()]


def chunks(a, eos, size):
    """Akis -> hikaye sinirinda kesilmis parcalarin (bas, son) sinirlari."""
    start = 0
    while start < len(a):
        stop = min(start + size, len(a))
        if stop < len(a):
            stop = start + int(np.flatnonzero(np.asarray(a[start:stop]) == eos)[-1]) + 1
        yield start, stop
        start = stop


_WORKER = {}


def _chunk(job):
    """Bir parca (ayri surecte) -> (parcanin sozlugu ilk gecis sirasiyla, yerel kimlikler, boylar, cumle imzalari, aday)."""
    root, start, stop = job
    if not _WORKER:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(root, "gpt2", "tokenizer.json"))
        _WORKER.update(tok=tok, eos=tok.token_to_id("<|endoftext|>"), tables=_tables(tok, tok.get_vocab_size()),
                       stream=np.load(os.path.join(root, "gpt2", "train.npy"), mmap_mode="r"))
    w_ = _WORKER
    local, ids, lengths, keys, n = {}, [], [], [], 0
    for s in stream_sentences(w_["stream"][start:stop], *w_["tables"], w_["tok"], w_["eos"]):
        w = _WORD.findall(s)
        n += 1
        if WORDS_MIN <= len(w) <= WORDS_MAX:
            ids.extend(local.setdefault(x, len(local)) for x in w)
            lengths.append(len(w))
            keys.append(_key(s))
    return (list(local), np.array(ids, dtype=np.int32), np.array(lengths, dtype=np.int64),
            np.array(keys, dtype=np.int64), n)


def _key(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def main(root, out, tokens=None):
    from tokenizers import Tokenizer
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(root, "gpt2", "tokenizer.json"))
    eos = tok.token_to_id("<|endoftext|>")
    flags, quotes = _tables(tok, tok.get_vocab_size())
    train_a = np.load(os.path.join(root, "gpt2", "train.npy"), mmap_mode="r")
    if tokens:
        train_a = train_a[:int(np.flatnonzero(np.asarray(train_a[:tokens]) == eos)[-1]) + 1]
    index = {UNK: 0}
    ids, lengths, keys, n_seen = [], [], [], 0
    jobs = [(root, s, t) for s, t in chunks(train_a, eos, CHUNK)]
    with Pool(min(WORKERS, len(jobs))) as pool:             # parcalar paralel; birlestirme sirayla (sozluk sirasi seri ile ayni)
        for c, (words, part_ids, part_len, part_keys, n) in enumerate(pool.imap(_chunk, jobs)):
            remap = np.array([index.setdefault(w, len(index)) for w in words], dtype=np.int32)
            ids.append(remap[part_ids] if len(part_ids) else part_ids)
            lengths.append(part_len)
            keys.append(part_keys)
            n_seen += n
            print("parca %d/%d: %d cumle (%d aday), sozluk %d, %.0f sn" % (
                c + 1, len(jobs), sum(map(len, lengths)), n_seen, len(index), time.time() - t0), flush=True)
    keys = np.concatenate(keys)
    _, first = np.unique(keys, return_index=True)               # tekrar eden cumle bir kez (ilk gecisi)
    keep = np.zeros(len(keys), dtype=bool)
    keep[first] = True
    lengths = np.concatenate(lengths)
    flat = np.concatenate(ids)[np.repeat(keep, lengths)]
    lengths = lengths[keep]
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    vocab = [w for w, _ in sorted(index.items(), key=lambda x: x[1])]
    used = np.bincount(flat, minlength=len(vocab)) > 0             # yalniz tekrar olarak gecen kelime sozlukte kalir
    train_keys = set(keys.tolist())
    del ids, keys

    rng = random.Random(SEED)
    test = stream_sentences(np.load(os.path.join(root, "gpt2", "valid.npy")), flags, quotes, tok, eos)
    rng.shuffle(test)
    exam = []
    for s in test:
        w = _WORD.findall(s)
        k = _key(s)
        if WORDS_MIN <= len(w) <= WORDS_MAX and k not in train_keys and all(x in index and used[index[x]] for x in w):
            exam.append(dict(sentence=s, words=w, split="unseen"))
            train_keys.add(k)
        if len(exam) >= EXAM:
            break
    for i in rng.sample(range(len(lengths)), SEEN):
        w = [vocab[j] for j in flat[offsets[i]:offsets[i + 1]].tolist()]
        exam.append(dict(sentence=" ".join(w), words=w, split="seen"))

    # gecerli siralar: sinav torbasiyla ayni torbadaki butun egitim / sinav cumleleri.  Once ucuz imza (boy, toplam,
    # kareler toplami), sonra tam torba.
    sig = lambda a: (len(a), int(a.sum()), int((a.astype(np.int64) ** 2).sum()))
    exam_ids = [np.array([index[x] for x in r["words"]], dtype=np.int64) for r in exam]
    bags = {}
    for r, a in zip(exam, exam_ids):
        bags.setdefault(sig(a), {}).setdefault(tuple(sorted(a.tolist())), set()).add(tuple(r["words"]))
    f64 = flat.astype(np.int64)
    tot, sq = np.add.reduceat(f64, offsets[:-1]), np.add.reduceat(f64 ** 2, offsets[:-1])
    for i in np.flatnonzero([(int(n), int(t), int(q)) in bags for n, t, q in zip(lengths, tot, sq)]).tolist():
        a = f64[offsets[i]:offsets[i + 1]]
        bag = bags[sig(a)].get(tuple(sorted(a.tolist())))
        if bag is not None:
            bag.add(tuple(vocab[j] for j in a.tolist()))
    for r, a in zip(exam, exam_ids):
        r["valid"] = sorted(list(o) for o in bags[sig(a)][tuple(sorted(a.tolist()))])

    json.dump(vocab, open(os.path.join(out, "ss_vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    np.save(os.path.join(out, "ss_train_ids.npy"), flat)
    np.save(os.path.join(out, "ss_train_offsets.npy"), offsets)
    with open(os.path.join(out, "ss_exam.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in exam:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    bands = Counter(min(int(n) // 10, 4) for n in lengths)
    print("aday %d, egitim %d cumle (tekrarsiz), %d kelime, sozluk %d | boy (10'luk) %s | sinav %s, birden fazla gecerli "
          "sira %d | %.0f sn" % (n_seen, len(lengths), len(flat), len(vocab), dict(sorted(bands.items())),
                                dict(Counter(r["split"] for r in exam)), sum(len(r["valid"]) > 1 for r in exam),
                                time.time() - t0), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--tokens") + 1]) if "--tokens" in args else None
    main(args[0], args[1], n)
