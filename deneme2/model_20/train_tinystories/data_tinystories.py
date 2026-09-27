# -*- coding: utf-8 -*-
"""data_tinystories -- TinyStories verisi.  model_18'in Drive onbellegi (v4, TAG) OKUNUR, yeniden uretilmez (kural 9).

    <eos> w1 ... wL <eos> <dolgu> ...     her hikaye bir pencere (SEQ_LEN); L + 2 > SEQ_LEN olan hikaye ATILIR, bolunmez
    egitim: akis_train; sinav: akis_valid (TinyStories'in kendi validation dosyasi).  train'de birebir gecen valid
    hikayesi valid_in_train'e yazilir, sinava girmez.

Drive <root>/onbellek/:  sozluk_<TAG>.npy (object dizisi, 8.004 token; basi SPECIAL), akis_train_<TAG>.npy ve
akis_valid_<TAG>.npy (int16, her hikayeden sonra <eos>).  Uretici: model_18 data_stories.build_cache (kod 9f40c54),
ayrinti belge/bulgu/model_18_veri.md §7; tokenizer ve decode buraya kopyalandi.
Eldan & Li, arXiv 2305.07759; lisans cdla-sharing-1.0.
"""
import hashlib
import os
import re
import unicodedata

import numpy as np
import torch

TAG = "tam_n8000_v4"
PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, NL_TOKEN = "<dolgu>", "<eos>", "<bilinmeyen>", "<nl>"
SPECIAL = (PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, NL_TOKEN)     # v4 sozlugunun basi, bu sirayla (id 0-3)
SEQ_LEN = 512      # pencere = Model X'in T_MAX'i; v4'te hikayelerin %98,4'u sigar (256'da %89,6)
WIDTH_MULTIPLE = 8 # bucket'ta batch genisligi bunun katina yuvarlanir: sekil cesidi azalir (512 / 8 = 64 genislik)

# --- tokenizer (model_18 data_stories v4'ten kopya): kelime duzeyi, kisaltma tek token, noktalama ayri
TOKEN_RE = re.compile(r"[A-Za-z]+'[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")
QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                           "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "´": "'",
                           "\xa0": " ", " ": " ", "\t": " ", "​": "", "…": "..."})
NO_SPACE_BEFORE = set(".,!?;:)]}'\"")
NO_SPACE_AFTER = set("([{")


def normalize(s):
    """Tirnak, bosluk ve aksan duzeltmesi; satir sonlari korunur."""
    s = s.translate(QUOTE_MAP)
    if not s.isascii():
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s


def tokenize(s):
    """Metin -> token'lar: satirlar TOKEN_RE ile, aralarinda tek <nl> (bos satirlar birlesir)."""
    out = []
    for line in s.split("\n"):
        t = TOKEN_RE.findall(line)
        if t:
            if out:
                out.append(NL_TOKEN)
            out += t
    return out


def encode(text, vocab):
    """Metin -> id listesi (ozel token eklemez); sozlukte olmayan kelime <bilinmeyen>."""
    ix = {w: i for i, w in enumerate(vocab)}
    return [ix.get(t, ix[UNK_TOKEN]) for t in tokenize(normalize(text))]


def decode(ids, vocab, in_quote=False):
    """id'ler -> okunur metin.  <eos> hikaye siniri ('---'), <nl> satir sonu.  in_quote: metin acik bir tirnagin
    icinde basliyor (istemin devami)."""
    s, open_ = "", False
    for x in ids:
        t = vocab[int(x)]
        if t == EOS_TOKEN:
            s += "\n\n---\n\n"
            in_quote = False
            continue
        if t == NL_TOKEN:                                  # satir basi: tirnak durumu sifirlanir
            s += "\n"
            in_quote = open_ = False
            continue
        if t == '"':                                       # tirnak hem acar hem kapar: sirayla
            attached, opens = in_quote, not in_quote
            in_quote = not in_quote
        else:
            attached, opens = t in NO_SPACE_BEFORE, t in NO_SPACE_AFTER
        s += t if (not s or s.endswith(("\n", " ")) or attached or open_) else " " + t
        open_ = opens
    return s.strip()


# --- Drive onbellegi

def fingerprint(vocab, train, valid):
    """(birlesik iz, parca izleri): sozluk metni ve iki akisin baytlari, sha256 ilk 12 hane."""
    parts = {"vocab": hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()[:12]}
    for name, a in (("train", train), ("valid", valid)):
        parts[name] = hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:12]
    joined = "|".join("%s:%s" % kv for kv in sorted(parts.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:12], parts


def build(root, tag=TAG, seq_len=SEQ_LEN, log=print):
    """Drive onbellegi -> veri.  root: TinyStories klasoru (icinde onbellek/).  Uretmez; dosya yoksa durur.
    -> vocab, train / valid (akis), <split>_start / <split>_length (pencereye sigan hikayeler), valid_in_train,
    counts, seq_len, fingerprint, fingerprints."""
    cache = os.path.join(root, "onbellek")
    files = [os.path.join(cache, "%s_%s.npy" % (k, tag)) for k in ("sozluk", "akis_train", "akis_valid")]
    missing = [f for f in files if not os.path.exists(f)]
    assert not missing, "onbellek Drive'da YOK: %s -- Colab uretmez (kural 9)" % missing
    vocab = [str(a) for a in np.load(files[0], allow_pickle=True)]
    eos = vocab.index(EOS_TOKEN)
    data = dict(vocab=vocab, seq_len=seq_len, tag=tag, counts={})
    for split, f in zip(("train", "valid"), files[1:]):
        a = np.load(f)
        ends = np.flatnonzero(a == eos)
        starts = np.concatenate([[0], ends[:-1] + 1])
        lengths = ends - starts                                  # <eos> haric
        keep = (lengths > 0) & (lengths + 2 <= seq_len)          # + bastaki ve sondaki <eos>
        data[split] = a
        data[split + "_start"], data[split + "_length"] = starts[keep], lengths[keep]
        data["counts"].update({split + "_tokens": len(a), split + "_stories": len(ends), split + "_kept": int(keep.sum()),
                               split + "_empty": int((lengths == 0).sum())})
    # sizinti: valid hikayesi train'de birebir geciyorsa sinava girmez
    valid = {data["valid"][s:s + L].tobytes(): i
             for i, (s, L) in enumerate(zip(data["valid_start"].tolist(), data["valid_length"].tolist()))}
    lengths = set(data["valid_length"].tolist())
    hits, a = set(), data["train"]
    for s, L in zip(data["train_start"].tolist(), data["train_length"].tolist()):
        if L in lengths:
            i = valid.get(a[s:s + L].tobytes())
            if i is not None:
                hits.add(i)
    data["valid_in_train"] = np.array(sorted(hits), dtype=np.int64)
    data["fingerprint"], data["fingerprints"] = fingerprint(vocab, data["train"], data["valid"])
    c = data["counts"]
    log("sozluk %d | train %d token, %d hikaye, %d pencere (%%%.2f) | valid %d hikaye, %d pencere, %d'i train'de birebir"
        % (len(vocab), c["train_tokens"], c["train_stories"], c["train_kept"], 100 * c["train_kept"] / c["train_stories"],
           c["valid_stories"], c["valid_kept"], len(hits)))
    log("iz %s  (%s)  SEQ_LEN %d  dolgu %%%.1f" % (
        data["fingerprint"], " ".join("%s %s" % kv for kv in sorted(data["fingerprints"].items())), seq_len,
        100 * (1 - (data["train_length"] + 2).sum() / (seq_len * max(c["train_kept"], 1)))))
    return data


def audit(data):
    """Sozluk v4 basi ve benzersiz; akislar int16, [1, n) araliginda (<dolgu> yok), <eos> ile biter, bos hikaye yok;
    pencereler <eos> ile baslar ve biter, arada <eos> yok.  Bozuksa DURUR.  -> sayimlar."""
    vocab = data["vocab"]
    assert tuple(vocab[:4]) == SPECIAL, vocab[:4]
    assert len(set(vocab)) == len(vocab), "sozlukte tekrar"
    eos = vocab.index(EOS_TOKEN)
    for split in ("train", "valid"):
        a, L = data[split], data[split + "_length"]
        assert a.dtype == np.int16 and len(a) and a[-1] == eos, "%s: int16 ve <eos> ile bitmeli" % split
        assert a.min() >= 1 and a.max() < len(vocab), (split, int(a.min()), int(a.max()))
        assert data["counts"][split + "_empty"] == 0, "%s: bos hikaye" % split
        assert len(L) and L.min() > 0 and L.max() + 2 <= data["seq_len"], split
        ids, mask = sequences(data, split, np.arange(min(64, len(L))))
        last = mask.sum(1) - 1
        assert (ids[:, 0] == eos).all() and (ids[torch.arange(len(last)), last] == eos).all(), split
        assert (((ids == eos) & mask).sum(1) == 2).all(), "%s: pencere icinde <eos>" % split
    return dict(data["counts"], valid_in_train=len(data["valid_in_train"]))


def sequences(data, split, rows, width=None):
    """Hikaye pencereleri -> (ids, mask), (len(rows), width): <eos> hikaye <eos>, sagdan <dolgu>; mask gercek token.
    width: varsayilan seq_len; daha dar verilirse en uzun pencere sigmali."""
    vocab, T = data["vocab"], data["seq_len"] if width is None else width
    eos = vocab.index(EOS_TOKEN)
    rows = np.asarray(rows, dtype=np.int64)
    starts, lengths = data[split + "_start"][rows], data[split + "_length"][rows]
    assert len(rows) == 0 or int(lengths.max()) + 2 <= T, "pencere genislige sigmiyor"
    ids = np.full((len(rows), T), vocab.index(PAD_TOKEN), dtype=np.int64)
    ids[:, 0] = eos
    a = data[split]
    for i, (s, L) in enumerate(zip(starts.tolist(), lengths.tolist())):
        ids[i, 1:L + 1] = a[s:s + L]
        ids[i, L + 1] = eos
    mask = np.arange(T)[None, :] < (lengths + 2)[:, None]
    return torch.from_numpy(ids), torch.from_numpy(mask)


def bucket_plan(data, batch_size, perm, bucket, seed, epoch):
    """Epok permutasyonunun ilk per_epoch x batch_size'i -> per_epoch batch (satir listeleri).  bucket x batch_size'lik
    parcalar boya gore siralanir, her parca hedef sayisi (L + 1) esit bucket batch'e kesilir; batch sirasi (seed, epoch, 1)
    tohumuyla karistirilir.  Ayni hikayeler, ayni adim sayisi; batch'in satir sayisi degisir."""
    per_epoch = len(perm) // batch_size
    lengths = data["train_length"]
    out = []
    for s in range(0, per_epoch * batch_size, bucket * batch_size):
        chunk = perm[s:min(s + bucket * batch_size, per_epoch * batch_size)]
        k = len(chunk) // batch_size
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        cum = np.cumsum(lengths[chunk] + 1)                            # hedef: pencere - 1 = L + 1
        middle = cum - (lengths[chunk] + 1) / 2                        # hikaye, orta noktasinin dustugu batch'e
        out += np.split(chunk, np.searchsorted(middle, cum[-1] * np.arange(1, k) / k))
    assert len(out) == per_epoch and all(len(b) for b in out), "bos batch"
    order = np.random.default_rng([seed, epoch, 1]).permutation(per_epoch)
    return [out[j] for j in order]


def batches(data, batch_size, seed=0, bucket=None):
    """Adimin fonksiyonu step -> (ids, mask).  Epok e: train pencerelerinin (seed, e) tohumlu permutasyonu sirayla
    bolunur; artan son parca o epokta kullanilmaz.  Ayni step her zaman ayni parca (surdurme).
    bucket=None: sekil hep (batch_size, seq_len).  bucket=K: bucket_plan -- ayni hikayeler ve adim sayisi, ama batch'i
    birlikte olusturan hikayeler DEGISIR (tarif degisikligi); genislik batch'in en uzun penceresi, WIDTH_MULTIPLE'in
    katina yuvarlanir."""
    n = len(data["train_start"])
    per_epoch = n // batch_size
    assert per_epoch > 0, "batch_size (%d) > train penceresi (%d)" % (batch_size, n)
    memo = {}

    def batch(step):
        epoch, i = divmod(step, per_epoch)
        if epoch not in memo:                              # epok basina bir permutasyon; yalniz hiz icin saklanir
            memo.clear()
            perm = np.random.default_rng([seed, epoch]).permutation(n)
            memo[epoch] = perm if bucket is None else bucket_plan(data, batch_size, perm, bucket, seed, epoch)
        if bucket is None:
            return sequences(data, "train", memo[epoch][i * batch_size:(i + 1) * batch_size])
        rows = memo[epoch][i]
        width = int(data["train_length"][rows].max()) + 2
        return sequences(data, "train", rows, min(-(-width // WIDTH_MULTIPLE) * WIDTH_MULTIPLE, data["seq_len"]))

    return batch
