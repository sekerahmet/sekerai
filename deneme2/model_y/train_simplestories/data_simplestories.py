# -*- coding: utf-8 -*-
"""data_simplestories -- SimpleStories verisi, iki tokenizer: ss4096 | gpt2.  Drive'daki hazir token dizileri OKUNUR; iz
yeniden hesaplanip fingerprint.json ile karsilastirilir, tutmazsa DURUR (kural 9).  Uretim tek seferlik: build_cache.

    train: SimpleStories train.  Pencere <eos> hikaye <eos>; SEQ_LEN'e sigmayan hikaye BOLUNUR (kullanici, 28 Eylul: "uzun
        hikayeler pencerelere bolunsun"; SimpleStories'in kendi egitimi boyle): ilk pencere <eos> + ilk SEQ_LEN - 1
        token, devam penceresi bir oncekinin son token'iyla baslar (1 token ortusme), son pencere <eos> ile biter --
        her token tam bir kez hedef.  train_head: pencere hikayenin basi mi (<eos> ile baslar).
    valid: SimpleStories test; bolunmez, sigmayan atilir.  train'de birebir (token dizisi) gecen valid hikayesi
        valid_in_train'e yazilir.  exam: sinav kumesi -- iki tokenizer'da da SEQ_LEN'e sigan, train'de birebir gecmeyen
        valid hikayeleri (exam_stories.npy; sabit, izli).
    dolgu eos id'si, mask disinda.
    bits_per_byte: toplam nll / ln 2 / toplam UTF-8 bayt -- tokenizer'dan bagimsiz; valid_bytes pencere basina.

Drive <root>/ (Colab: /content/drive/MyDrive/simplestories):
    raw/data/*.parquet                     DATASET (Hugging Face, sabit surum, sha256 dogrulanir)
    <tag>/tokenizer.json                   TOKENIZERS[tag] (sabit surum, sha256 dogrulanir)
    <tag>/train.npy, valid.npy             uint16, her hikayeden sonra eos
    <tag>/train_bytes.npy, valid_bytes.npy int32, hikaye basina UTF-8 bayt
    <tag>/fingerprint.json                 kaynak izi ve parca izleri, kaynaklar, sayimlar
    <tag>/text_reference.json              train'in kelime kumesi ve 'X and X' sayilari (count_text_errors), kaynak izi,
                                           icerik sha256'si ve kurali (build_text_reference; kullanici, 29 Eylul: "Evet")
    exam_stories.npy, exam_stories.json    sinav kumesi (valid hikaye sirasi), sha256 ve kurali
    log.txt                                build_cache / build_text_reference gunlugu
Iz (fingerprint): kaynak parcalari (sozluk, tokenizer, akislar, baytlar) fingerprint.json ile karsilastirilir; birlesik iz
bunlara pencere tablosunu (windows) ve sinav kumesini (exam) de katar -- bolme kurali ya da SEQ_LEN degisirse iz degisir.
text_reference.json birlesik ize girmez (eski kosularin izi degismesin); build kaynak izini, icerik izini ve kurali dogrular.

ss4096: SimpleStories'in kendi WordPiece'i (4.096; kucuk harf, bosluk ve satir sonu kaybolur; [UNK] 0, [EOS] 1).
gpt2: GPT-2 byte-level BPE (50.257, kayipsiz; <|endoftext|> 50256).  Sozlukte eos'un yazimi EOS_TOKEN: egitim ve sinav
kodu eos'u data_tinystories'teki adla bulur.
Finke ve digerleri, arXiv 2504.09184; lisans MIT.

    python data_simplestories.py build_cache <root> [tag ...]      tek seferlik; ag + CPU, arka planda
    python data_simplestories.py build_text_reference <root> [tag ...]   tek seferlik; CPU, tag basina ~2 dk
"""
import collections
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.request

import torch  # pyarrow'dan ONCE yuklenmeli: Windows'ta ters sirada torch'un c10.dll'i yuklenmiyor
import numpy as np

TAGS = ("ss4096", "gpt2")
SEQ_LEN = 512      # pencere = model_y T_MAX
WIDTH_MULTIPLE = 8 # bucket'ta batch genisligi bunun katina yuvarlanir
EOS_TOKEN = "<eos>"
PAD_TOKEN = EOS_TOKEN   # dolgu eos id'si: mask disinda kalir, hedef olmaz
UNK_TOKEN = "[UNK]"     # yalniz ss4096'da
NO_SPACE_BEFORE = set(".,!?;:)]}")
NO_SPACE_AFTER = set("([{")

DATASET = dict(
    repo="datasets/SimpleStories/SimpleStories", revision="e63b8adc3b1a1bdc7cac5b500d150b71346b0628", column="story",
    files=dict(valid=["data/test-00000-of-00001.parquet"],
               train=["data/train-%05d-of-00007.parquet" % i for i in range(7)]),
    sha256={  # Hugging Face LFS izleri
        "data/test-00000-of-00001.parquet": "24a513732fd6c1158e33a6130e3977749ae13d583948e5a2198ed53235e29bec",
        "data/train-00000-of-00007.parquet": "ca33531b99f3bebb4125e82f56017c223a4900331b55bcf7cde1b0f750d88fd4",
        "data/train-00001-of-00007.parquet": "eac50d3053e1b68345061edbe5aebd90062b5ff4a77d7002ca4c5580360c7664",
        "data/train-00002-of-00007.parquet": "71ada93c7f128743c4a94de81f55cc48319bde0636aeb35889aded5d52841427",
        "data/train-00003-of-00007.parquet": "cb299fafbef9cb030eebdac229494b313757db746f238084b23131e39b77d71e",
        "data/train-00004-of-00007.parquet": "6370179dae77b578e69f0e0882171b6e1586e84fe8e74ce71d8ab3e4246f96e1",
        "data/train-00005-of-00007.parquet": "2ca55fb42b239ce8e649f297b11895bc6611af8e5017eb7c46e0ccc05e2a8220",
        "data/train-00006-of-00007.parquet": "fe8aa328bfbcbf05f6953c6e9086635a840fe703bf36709a3a4d17806de89998"})
TOKENIZERS = dict(  # ss4096 = simple_stories_train tokenizer/simplestories-4096.json ile bayt bayt ayni (V1 modelleri)
    ss4096=dict(repo="SimpleStories/SimpleStories-1.25M", revision="d7932a858aea8f58a911c79d611d964483c21f47",
                file="tokenizer.json", eos="[EOS]",
                sha256="db7cdb1b1d23e2788cb816f3d336124ee00cc3a0d2f21fa9e934bbfb7d26c293"),
    gpt2=dict(repo="openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
              file="tokenizer.json", eos="<|endoftext|>",
              sha256="8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"))
_LOADED = {}  # id(vocab) -> (vocab, Tokenizer, eos id); encode / decode tokenizer'i sozlukten bulur
_UNIT = re.compile(r"\d|[^\W\d_]+|[^\w\s]")   # birim: kelime, tek rakam, tek noktalama (ss4096 on-bolucusu gibi)
_XWORD = re.compile(r"[a-z']+")
_XAX_WORDS = 7           # 'X and X'te X en cok bu kadar birim: (?:[a-z']+ ){0,6}[a-z']+
_P = np.uint64(0x9E3779B97F4A7C15)            # metin ozeti carpani (64 bit, tasma sarar)
_CHUNK = 1 << 24         # train taramasi parca boyu (token)
_TABLES = {}             # id(vocab) -> (vocab, birim tablosu)
_TEXT_REFERENCE = "text_reference.json"
_TEXT_REFERENCE_RULE = (
    "birim: kucuk harf kelime ('##' / bosluksuz harf parcalari birlesik), tek rakam, tek noktalama; words: train'de gecen "
    "alfabetik birimler; xax: her 'and' / 'or' birimi icin hemen once ve hemen sonra ayni en uzun X (1-%d birim, [a-z']+), "
    "ortusmeli sayim" % _XAX_WORDS)


def _load(path, eos):
    """tokenizer.json -> (Tokenizer, vocab): vocab id sirasiyla, eos'un yazimi EOS_TOKEN."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    vocab = [tok.id_to_token(i) for i in range(tok.get_vocab_size(with_added_tokens=True))]
    e = tok.token_to_id(eos)
    assert None not in vocab and e is not None, "%s: id'ler ardisik degil ya da %s yok" % (path, eos)
    vocab[e] = EOS_TOKEN
    _LOADED[id(vocab)] = (vocab, tok, e)
    return tok, vocab


def _tokenizer(vocab):
    """vocab -> (Tokenizer, eos id); ayni icerikli kopya da bulunur."""
    hit = _LOADED.get(id(vocab))
    if hit is None or hit[0] is not vocab:
        hit = next((h for h in _LOADED.values() if h[0] == vocab), None)
        assert hit is not None, "bu sozlugun tokenizer'i yuklu degil: once build(root, tag)"
        _LOADED[id(vocab)] = (vocab,) + hit[1:]
    return hit[1], hit[2]


def encode(text, vocab):
    """Metin -> id listesi (ozel token eklemez).  ss4096 kucuk harfe cevirir, bosluk ve satir sonunu atar; gpt2 kayipsiz."""
    return _tokenizer(vocab)[0].encode(text, add_special_tokens=False).ids


def decode(ids, vocab, in_quote=False):
    """id'ler -> okunur metin; eos hikaye siniri ('---').  gpt2: tokenizer'in kendi decode'u, kirpilmaz (kayipsiz:
    decode(encode(s)) == s).  WordPiece (ss4096): '##' parca birlesir, noktalama bosluksuz yapisir, uclar kirpilir --
    yalniz bosluk degisir, encode(decode(ids)) == ids.  in_quote (yalniz WordPiece): metin acik bir tirnagin icinde
    basliyor (istemin devami)."""
    from tokenizers.models import WordPiece
    tok, eos = _tokenizer(vocab)
    ids = [int(x) for x in ids]
    if not isinstance(tok.model, WordPiece):
        pieces = [[]]
        for x in ids:
            if x == eos:
                pieces.append([])
            else:
                pieces[-1].append(x)
        return "\n\n---\n\n".join(tok.decode(p, skip_special_tokens=False) for p in pieces)
    s, open_, digit = "", False, False
    for x in ids:
        t = vocab[x]
        if x == eos:
            s += "\n\n---\n\n"
            in_quote = open_ = digit = False
            continue
        if t.startswith("##"):
            s += t[2:]
            open_ = digit = False
            continue
        if t == '"':                                       # tirnak hem acar hem kapar: sirayla
            attached, opens = in_quote, not in_quote
            in_quote = not in_quote
        elif t in ("'", "-"):                              # kisaltma (don't) ve birlesik kelime (well-known)
            attached, opens = True, True
        else:
            attached, opens = t in NO_SPACE_BEFORE or (digit and t.isdigit()), t in NO_SPACE_AFTER
        s += t if (not s or s.endswith(("\n", " ")) or attached or open_) else " " + t
        open_, digit = opens, t.isdigit()
    return s.strip()


def bits_per_byte(nll_sum, n_bytes):
    """Toplam nll (nat) ve toplam UTF-8 bayt -> bit / bayt.  Tokenizer'dan bagimsiz; eos hedefinin ve baytin nasil
    sayildigi cagiranin tanimi."""
    return nll_sum / math.log(2) / n_bytes


# --- Drive

def fingerprint(vocab, train, valid, **extra):
    """(birlesik iz, parca izleri): sozluk metni ve dizilerin baytlari; extra: tokenizer dosyasi (bytes), bayt dizileri,
    pencere tablosu.  sha256 ilk 12 hane; str verilen parca hazir iz sayilir."""
    parts = {"vocab": hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()[:12]}
    for name, a in dict(extra, train=train, valid=valid).items():
        parts[name] = a if isinstance(a, str) else hashlib.sha256(
            a if isinstance(a, bytes) else np.ascontiguousarray(a)).hexdigest()[:12]
    joined = "|".join("%s:%s" % kv for kv in sorted(parts.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:12], parts


def _stories(a, eos):
    """Akis -> hikaye basina (baslangic, uzunluk); uzunluk eos haric."""
    ends = np.flatnonzero(a == eos)
    starts = np.concatenate([[0], ends[:-1] + 1]).astype(np.int64)
    return starts, ends - starts


def _found(a, starts, lengths, b, b_starts, b_lengths):
    """b'nin hikayelerinden a'nin hikayeleri arasinda birebir (token dizisi) gecenlerin sirasi."""
    want = {}
    for i, (s, L) in enumerate(zip(b_starts.tolist(), b_lengths.tolist())):
        want.setdefault(b[s:s + L].tobytes(), []).append(i)
    sizes, hits = set(b_lengths.tolist()), set()
    for s, L in zip(starts.tolist(), lengths.tolist()):
        if L in sizes:
            hits.update(want.get(a[s:s + L].tobytes(), ()))
    return np.array(sorted(hits), dtype=np.int64)


def _windows(starts, lengths, seq_len):
    """Hikayeler -> pencereler (start, length, head).  Hikaye X = <eos> + L token + <eos>; pencere j = X[j (seq_len - 1):
    ... + seq_len] (1 token ortusme: her hedef tam bir kez).  length = pencere - 2; head: j == 0, pencere <eos> +
    a[start:start + length + 1]; degilse a[start:start + length + 2]."""
    total = lengths + 2
    k = -(-(lengths + 1) // (seq_len - 1))                       # hikaye basina pencere: hedef L + 1, pencere basina seq_len - 1
    story = np.repeat(np.arange(len(lengths)), k)
    j = np.arange(len(story)) - np.repeat(np.cumsum(k) - k, k)
    off = j * (seq_len - 1)
    n = np.minimum(seq_len, total[story] - off)
    return starts[story] + off - (j > 0), n - 2, j == 0


def build(root, tag, seq_len=SEQ_LEN, log=print):
    """Drive -> veri.  root: simplestories klasoru, tag: TAGS'tan biri.  Uretmez: dosya yoksa, kaynak izi fingerprint.json
    ile, sinav kumesi exam_stories.json ile ya da text_reference.json (icerik izi, kaynak izi, kural) tutmazsa DURUR.
    -> vocab, train / valid (akis, uint16), train_start / _length / _head (pencereler, uzun hikaye bolunmus), train_bytes
    (hikaye basina, bos dahil; exam_train), valid_start / _length / _bytes (sigan hikayeler), valid_in_train, exam (sinav
    kumesinin valid pencereleri), text_reference (words, xax, sha256), counts, seq_len, tag, fingerprint, fingerprints."""
    return _build(root, tag, seq_len, log, reference=True)


def _build(root, tag, seq_len, log, reference):
    """build; reference=False: text_reference.json aranmaz (build_text_reference onu uretirken)."""
    d = os.path.join(root, tag)
    names = [os.path.join(tag, n) for n in ("tokenizer.json", "train.npy", "valid.npy", "train_bytes.npy",
                                            "valid_bytes.npy", "fingerprint.json")] + ["exam_stories.npy",
                                                                                       "exam_stories.json"]
    missing = [n for n in names if not os.path.exists(os.path.join(root, n))]
    assert not missing, "%s: Drive'da YOK %s -- build_cache bir kez uretir, Colab uretmez (kural 9)" % (root, missing)
    assert not reference or os.path.exists(os.path.join(d, _TEXT_REFERENCE)), \
        "%s: Drive'da YOK %s -- build_text_reference(root, %r) bir kez uretir, Colab uretmez (kural 9)" % (
            d, _TEXT_REFERENCE, tag)
    saved = json.load(open(os.path.join(d, "fingerprint.json"), encoding="utf-8"))
    _, vocab = _load(os.path.join(d, "tokenizer.json"), saved["eos"])
    eos = vocab.index(EOS_TOKEN)
    data = dict(vocab=vocab, seq_len=seq_len, tag=tag, counts={})
    whole, stories = {}, {}
    for split in ("train", "valid"):
        a = data[split] = np.load(os.path.join(d, split + ".npy"))
        b = whole[split + "_bytes"] = np.load(os.path.join(d, split + "_bytes.npy"))
        starts, lengths = stories[split] = _stories(a, eos)
        assert len(starts) == len(b), "%s: %d eos, %d bayt satiri -- hikaye siniri tutmuyor" % (split, len(starts), len(b))
        data["counts"].update({split + "_tokens": len(a), split + "_stories": len(starts),
                               split + "_empty": int((lengths == 0).sum()), split + "_bytes": int(b.sum())})
        if UNK_TOKEN in vocab:
            data["counts"][split + "_unk"] = int((a == vocab.index(UNK_TOKEN)).sum())
    starts, lengths = stories["train"]
    keep = lengths > 0
    data["train_start"], data["train_length"], data["train_head"] = _windows(starts[keep], lengths[keep], seq_len)
    data["train_bytes"] = whole["train_bytes"]
    starts, lengths = stories["valid"]
    keep = (lengths > 0) & (lengths + 2 <= seq_len)                   # + bastaki ve sondaki eos
    data["valid_start"], data["valid_length"], data["valid_bytes"] = starts[keep], lengths[keep], whole["valid_bytes"][keep]
    c = data["counts"]
    c.update(train_kept=len(data["train_start"]), valid_kept=int(keep.sum()),
             train_split=int((stories["train"][1] + 2 > seq_len).sum()),
             train_targets=int((data["train_length"] + 1).sum()))
    # sizinti: valid hikayesi train'de birebir (token dizisi) geciyorsa sinava girmez; butun train hikayeleriyle
    data["valid_in_train"] = _found(data["train"], *stories["train"], data["valid"], data["valid_start"],
                                    data["valid_length"])
    # sinav kumesi: valid hikaye sirasi -> bu seq_len'deki valid penceresi
    exam = np.load(os.path.join(root, "exam_stories.npy"))
    exam_saved = json.load(open(os.path.join(root, "exam_stories.json"), encoding="utf-8"))
    exam_hash = hashlib.sha256(np.ascontiguousarray(exam)).hexdigest()
    assert exam_hash == exam_saved["sha256"], "exam_stories.npy izi exam_stories.json ile tutmuyor -- DURDU (kural 9)"
    assert exam_saved["tags"].get(tag) == saved["fingerprint"], "sinav kumesi baska bir %s verisiyle uretilmis" % tag
    rows_of_story = np.flatnonzero(keep)
    rows = np.searchsorted(rows_of_story, exam)
    assert (rows < len(rows_of_story)).all() and (rows_of_story[np.minimum(rows, len(rows_of_story) - 1)] == exam).all(), \
        "sinav hikayesi seq_len %d'ye sigmiyor" % seq_len
    data["exam"] = rows
    tokenizer_file = open(os.path.join(d, "tokenizer.json"), "rb").read()
    source, parts = fingerprint(vocab, data["train"], data["valid"], tokenizer=tokenizer_file, **whole)
    if source != saved["fingerprint"]:
        differ = sorted(k for k in set(parts) | set(saved["fingerprints"]) if parts.get(k) != saved["fingerprints"].get(k))
        raise AssertionError("%s: iz tutmuyor, hesaplanan %s, fingerprint.json %s, farkli parca %s -- DURDU (kural 9)"
                             % (d, source, saved["fingerprint"], differ))
    table = b"".join(np.ascontiguousarray(data["train_" + k]).tobytes() for k in ("start", "length", "head"))
    data["fingerprint"], data["fingerprints"] = fingerprint(
        vocab, parts["train"], parts["valid"], **dict({k: v for k, v in parts.items() if k not in ("vocab", "train", "valid")},
                                                    windows=table, exam=exam_hash[:12]))
    log("%s: sozluk %d | train %d token, %d hikaye, %d pencere (%d hikaye bolundu), hedef %d | valid %d hikaye, %d "
        "pencere, %d'i train'de birebir | sinav %d hikaye"
        % (tag, len(vocab), c["train_tokens"], c["train_stories"], c["train_kept"], c["train_split"], c["train_targets"],
           c["valid_stories"], c["valid_kept"], len(data["valid_in_train"]), len(rows)))
    log("iz %s (kaynak %s = fingerprint.json)  (%s)  SEQ_LEN %d  dolgu %%%.1f  bayt/token %.3f" % (
        data["fingerprint"], source, " ".join("%s %s" % kv for kv in sorted(data["fingerprints"].items())), seq_len,
        100 * (1 - (data["train_length"] + 2).sum() / (seq_len * max(c["train_kept"], 1))),
        c["train_bytes"] / max(c["train_tokens"] - c["train_stories"], 1)))
    if reference:
        ref = data["text_reference"] = _read_text_reference(os.path.join(d, _TEXT_REFERENCE), source)
        log("metin referansi %s (%s): train kelime %d, 'X and X' cifti %d" % (
            ref["sha256"], _TEXT_REFERENCE, len(ref["words"]), len(ref["xax"])))
    return data


def audit(data):
    """Sozluk benzersiz ve uint16'ya sigar; akislar uint16, [0, n) araliginda, eos ile biter, bos hikaye yok.  train
    pencereleri: her token tam bir kez hedef (hedef toplami = akis boyu), hikaye basina tek son pencere (eos ile biter),
    bas pencere eos ile baslar, pencere icinde eos yok.  valid: pencere eos hikaye eos, bayt dizisi hizali.  Sinav
    kumesi sizintisiz.  Bozuksa DURUR.  -> sayimlar."""
    vocab = data["vocab"]
    assert len(set(vocab)) == len(vocab) and len(vocab) <= 65536, "sozlukte tekrar ya da uint16'ya sigmiyor"
    eos = vocab.index(EOS_TOKEN)
    for split in ("train", "valid"):
        a, L = data[split], data[split + "_length"]
        head = data.get(split + "_head", np.ones(len(L), dtype=bool))
        assert a.dtype == np.uint16 and len(a) and a[-1] == eos, "%s: uint16 ve eos ile bitmeli" % split
        assert int(a.max()) < len(vocab), (split, int(a.max()))
        assert data["counts"][split + "_empty"] == 0, "%s: bos hikaye" % split
        assert len(L) and L.min() >= 0 and L.max() + 2 <= data["seq_len"] and len(head) == len(L), split
        rows = np.concatenate([np.arange(min(64, len(L))), np.flatnonzero(~head)[:64]])
        ids, mask = sequences(data, split, rows)
        last = ids[torch.arange(len(rows)), mask.sum(1) - 1]
        h = torch.from_numpy(head[rows])
        assert ((ids[:, 0] == eos) == h).all(), "%s: bas pencere eos ile baslamali, devam penceresi baslamamali" % split
        assert (((ids == eos) & mask).sum(1) == h.long() + (last == eos).long()).all(), "%s: pencere icinde eos" % split
    last_index = data["train_start"] + data["train_length"] + 1 - data["train_head"].astype(np.int64)
    stories = data["counts"]["train_stories"]
    assert int((data["train_length"] + 1).sum()) == len(data["train"]), "train: hedef toplami akis boyu degil"
    assert int((data["train"][last_index] == eos).sum()) == stories == int(data["train_head"].sum()), \
        "train: hikaye basina tek bas ve tek son pencere olmali"
    assert data["valid_length"].min() > 0 and len(data["valid_bytes"]) == len(data["valid_length"]) \
        and data["valid_bytes"].min() > 0, "valid: bayt dizisi pencerelerle hizali degil"
    exam = data["exam"]
    assert len(exam) and (np.diff(exam) > 0).all() and not np.isin(exam, data["valid_in_train"]).any(), \
        "sinav kumesi sirali, tekrarsiz ve sizintisiz olmali"
    return dict(data["counts"], valid_in_train=len(data["valid_in_train"]), exam=len(exam))


def sequences(data, split, rows, width=None):
    """Pencereler -> (ids, mask), (len(rows), width); sagdan dolgu, mask gercek token.  Bas pencere <eos> +
    a[start:start + length + 1], devam penceresi a[start:start + length + 2] (<split>_head yoksa hepsi bas: <eos> hikaye
    <eos>).  width: varsayilan seq_len; daha dar verilirse en uzun pencere sigmali."""
    vocab, T = data["vocab"], data["seq_len"] if width is None else width
    eos = vocab.index(EOS_TOKEN)
    rows = np.asarray(rows, dtype=np.int64)
    starts, lengths = data[split + "_start"][rows], data[split + "_length"][rows]
    head = data[split + "_head"][rows] if split + "_head" in data else np.ones(len(rows), dtype=bool)
    assert len(rows) == 0 or int(lengths.max()) + 2 <= T, "pencere genislige sigmiyor"
    ids = np.full((len(rows), T), vocab.index(PAD_TOKEN), dtype=np.int64)
    a = data[split]
    for i, (s, L, h) in enumerate(zip(starts.tolist(), lengths.tolist(), head.tolist())):
        if h:
            ids[i, 0] = eos
            ids[i, 1:L + 2] = a[s:s + L + 1]
        else:
            ids[i, :L + 2] = a[s:s + L + 2]
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
    bucket=None: sekil hep (batch_size, seq_len).  bucket=K: bucket_plan; genislik batch'in en uzun penceresi,
    WIDTH_MULTIPLE'in katina yuvarlanir."""
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


# --- metin referansi (exam_simplestories.count_text_errors): birimler ve train'in kelime / 'X and X' sayimi

def _string_hash(s):
    """Metnin 64 bit polinom ozeti ve _P^uzunluk: H(a + b) = H(a) x _P^len(b) + H(b) (tasma sarar)."""
    h, p = 0, 1
    for c in s:
        h, p = (h * int(_P) + ord(c)) % (1 << 64), p * int(_P) % (1 << 64)
    return h, p


def _token_table(vocab):
    """Token -> birim tablosu: pieces (kucuk harf birimler), joins (ilk birimi onceki token'in son birimine yapisir), ends
    (harfle biter), empty (birimsiz: bosluk; eos haric), xword (tek birim, [a-z']+), hash / power (birimlerin bitisik
    metninin _string_hash'i: ayni metin, farkli token'lama ayni anahtar), conj ('and' 1, 'or' 2).  WordPiece (ss4096):
    '##' parca yapisir.  Byte BPE (gpt2): bosluksuz harfle baslayan token, harfle biten token'a yapisir."""
    hit = _TABLES.get(id(vocab))
    if hit is not None and hit[0] is vocab:
        return hit[1]
    from tokenizers.models import WordPiece
    V, eos = len(vocab), vocab.index(EOS_TOKEN)
    tok = _tokenizer(vocab)[0]
    if isinstance(tok.model, WordPiece):
        pieces = [[t[2:]] if t.startswith("##") and len(t) > 2 else [t] for t in vocab]
        joins = np.array([t.startswith("##") and len(t) > 2 for t in vocab])
        ends = np.ones(V, dtype=bool)
    else:
        text = [tok.decode([i]) for i in range(V)]
        pieces = [_UNIT.findall(s.lower()) for s in text]
        joins = np.array([s[:1].isalpha() for s in text])
        ends = np.array([s[-1:].isalpha() for s in text])
    pieces[eos], joins[eos], ends[eos] = [], False, False
    hashes = [_string_hash("".join(p)) for p in pieces]
    tab = dict(pieces=pieces, joins=joins, ends=ends, joins_list=joins.tolist(), ends_list=ends.tolist(),
               empty=np.array([not p for p in pieces]) & (np.arange(V) != eos),
               xword=np.array([len(p) == 1 and bool(_XWORD.fullmatch(p[0])) for p in pieces]),
               hash=np.array([h for h, _ in hashes], dtype=np.uint64), power=np.array([p for _, p in hashes], dtype=np.uint64),
               conj={_string_hash("and")[0]: 1, _string_hash("or")[0]: 2})
    _TABLES[id(vocab)] = (vocab, tab)
    return tab


def _units(ids, tab):
    """id listesi -> birimler (kelime '##' / bosluksuz parcalarla birlesik, tek rakam, tek noktalama); eos ve bosluk sinir."""
    pieces, joins, ends = tab["pieces"], tab["joins_list"], tab["ends_list"]
    out, letter = [], False
    for t in ids:
        p = pieces[t]
        if p and joins[t] and letter and out:
            out[-1] += p[0]
            out.extend(p[1:])
        else:
            out.extend(p)
        letter = ends[t] and bool(p)
    return out


def _segments(x, tab):
    """Token dizisi (numpy) -> birim segmentleri (baslangic, token sayisi); bosluk token'lari atilir, eos kalir (sinir)."""
    join = tab["joins"][x[1:]] & tab["ends"][x[:-1]]
    starts = np.concatenate([[0], np.flatnonzero(~join) + 1]).astype(np.int64)
    lengths = np.diff(np.append(starts, len(x)))
    keep = ~tab["empty"][x[starts]]
    return starts[keep], lengths[keep]


def _segment_keys(x, starts, lengths, tab):
    """Segment basina anahtar: birimlerinin bitisik metninin _string_hash'i, token'lar uzerinden katlanarak."""
    keys = tab["hash"][x[starts]]
    multi = np.flatnonzero(lengths > 1)
    if len(multi):
        s, L = starts[multi], lengths[multi]
        h = keys[multi]
        for j in range(1, int(L.max())):
            m = np.flatnonzero(L > j)
            t = x[s[m] + j]
            h[m] = h[m] * tab["power"][t] + tab["hash"][t]
        keys[multi] = h
    return keys


def _xax_counts(x, starts, lengths, keys, tab):
    """Token parcasinda her 'and' / 'or' icin en uzun X (en cok _XAX_WORDS [a-z']+ birim, bagdan hemen once ve hemen
    sonra ayni) -> Counter {(X, bag): sayi}.  Ortusme sayilir: zincir 'dug and dug and dug' iki kez."""
    bad = np.concatenate([[0], np.cumsum(~tab["xword"][x])])
    ok = bad[starts + lengths] == bad[starts]
    conj = np.zeros(len(keys), dtype=np.int64)
    for h, c in tab["conj"].items():
        conj[keys == np.uint64(h)] = c
    ci = np.flatnonzero(ok & (conj > 0))
    best = np.zeros(len(ci), dtype=np.int64)
    for m in range(1, _XAX_WORDS + 1):
        live = np.flatnonzero((ci >= m) & (ci + m < len(starts)))
        for q in range(m):
            i = ci[live]
            live = live[ok[i - m + q] & (keys[i - m + q] == keys[i + 1 + q])]
        best[live] = m
    hit = np.flatnonzero(best)
    out = collections.Counter()
    if not len(hit):
        return out
    i, m = ci[hit], best[hit]
    kind = conj[i]
    h = (m * 3 + kind).astype(np.uint64)
    for q in range(_XAX_WORDS):
        sel = np.flatnonzero(m > q)
        h[sel] = h[sel] * _P + keys[i[sel] - m[sel] + q]
    _, first, count = np.unique(h, return_index=True, return_counts=True)
    for f, c in zip(first.tolist(), count.tolist()):
        X = _units(x[starts[i[f] - m[f]]:starts[i[f]]].tolist(), tab)
        out[(" ".join(X), "and" if kind[f] == 1 else "or")] += c
    return out


def _text_reference(vocab, a, log=lambda s: None):
    """Token akisi (eos ile biten hikayeler) -> (alfabetik kelime kumesi, 'X and X' sayilari {(X, bag): sayi}).  Parca
    parca (_CHUNK token, eos sinirinda)."""
    tab, eos = _token_table(vocab), vocab.index(EOS_TOKEN)
    single = np.zeros(len(vocab), dtype=bool)
    multi, xax, c0, t0 = [], collections.Counter(), 0, time.time()
    while c0 < len(a):
        c1, w = c0 + _CHUNK, 4096                     # parca sonu: c0 + _CHUNK'tan sonraki ilk eos (dahil)
        while c1 < len(a):
            after = np.flatnonzero(a[c1:c1 + w] == eos)
            if len(after):
                c1 += int(after[0]) + 1
                break
            c1, w = c1 + w, 2 * w
        x = a[c0:min(c1, len(a))]
        c1 = c0 + len(x)
        starts, lengths = _segments(x, tab)
        keys = _segment_keys(x, starts, lengths, tab)
        one = lengths == 1
        single[x[starts[one]]] = True
        _, first = np.unique(keys[~one], return_index=True)
        multi.append((keys[~one][first], c0 + starts[~one][first], lengths[~one][first]))
        xax.update(_xax_counts(x, starts, lengths, keys, tab))
        c0 = c1
        if c0 == len(a) or len(multi) % 8 == 0:
            log("metin referansi %d / %d token  (%.0f sn)" % (c0, len(a), time.time() - t0))
    words = {w for t in np.flatnonzero(single).tolist() for w in tab["pieces"][t] if w.isalpha()}
    k, s, L = (np.concatenate(z) for z in zip(*multi))
    for j in np.unique(k, return_index=True)[1].tolist():
        words.update(w for w in _units(a[s[j]:s[j] + L[j]].tolist(), tab) if w.isalpha())
    return words, xax


def _reference_body(words, xax):
    """-> (dosya govdesi, sha256): kelimeler sirali, 'X|bag' anahtarlari; iz govdenin sirali JSON'undan."""
    body = dict(words=sorted(words), xax={"%s|%s" % k: int(c) for k, c in sorted(xax.items())})
    return body, hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_text_reference(path, source):
    """text_reference.json -> dict(words, xax, sha256).  Icerik izi, kaynak izi (fingerprint.json) ya da kural tutmazsa
    DURUR; yeniden hesaplamaz (tazeleme karar: dosya elle tasinir, build_text_reference)."""
    ref = json.load(open(path, encoding="utf-8"))
    body, sha = _reference_body(ref["words"], {tuple(k.rsplit("|", 1)): c for k, c in ref["xax"].items()})
    assert sha == ref["sha256"], "%s: icerik izi tutmuyor (%s, dosyada %s) -- DURDU (kural 9)" % (path, sha[:12],
                                                                                              ref["sha256"][:12])
    assert ref["source"] == source, "%s: baska bir veriden uretilmis (kaynak izi %s, fingerprint.json %s) -- DURDU (kural 9)" \
        % (path, ref["source"], source)
    assert ref["rule"] == _TEXT_REFERENCE_RULE, "%s: kural degismis -- dosyayi tasi, build_text_reference yeniden uretir" % path
    return dict(words=set(body["words"]), sha256=sha[:12],
                xax=collections.Counter({tuple(k.rsplit("|", 1)): c for k, c in body["xax"].items()}))


def _root_log(root):
    def log(line):
        line = time.strftime("%H:%M:%S ") + line
        print(line, flush=True)
        with open(os.path.join(root, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def build_text_reference(root, tag, log=None):
    """Tek seferlik (kural 9; kullanici, 29 Eylul: "Evet"): <root>/<tag>/text_reference.json -- train'deki alfabetik
    kelimeler (uydurma kelime) ve 'X and X' sayilari (legit), kural, kaynak izi (fingerprint.json; train once build ile
    dogrulanir) ve icerigin sha256'si.  Dosya varsa dokunmaz (tazelemek icin elle tasinir).  -> ust bilgi (varsa None)."""
    log = log or _root_log(root)
    path = os.path.join(root, tag, _TEXT_REFERENCE)
    if os.path.exists(path):
        log("%s: %s var, ATLANDI (tazelemek icin dosyayi elle tasi)" % (tag, _TEXT_REFERENCE))
        return None
    t0 = time.time()
    data = _build(root, tag, SEQ_LEN, lambda s: None, reference=False)     # train fingerprint.json ile dogrulanir
    source = json.load(open(os.path.join(root, tag, "fingerprint.json"), encoding="utf-8"))["fingerprint"]
    words, xax = _text_reference(data["vocab"], data["train"], log)
    body, sha = _reference_body(words, xax)
    meta = dict(tag=tag, source=source, sha256=sha, rule=_TEXT_REFERENCE_RULE,
                counts=dict(words=len(body["words"]), xax_pairs=len(body["xax"]), xax_events=sum(body["xax"].values()),
                            train_tokens=len(data["train"])),
                created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0))
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(dict(meta, **body), f, ensure_ascii=False)
    os.replace(path + ".part", path)
    log("%s: %s YAZILDI iz %s  %s" % (tag, _TEXT_REFERENCE, sha[:12], json.dumps(meta)))
    return meta


# --- tek seferlik uretim

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, path, sha256, log):
    """Dosya yoksa ya da sha256 tutmuyorsa indirir (.part, sonra ad degisir); sha256 dogrulanir.  -> sha256."""
    if os.path.exists(path):
        h = _sha256(path)
        if sha256 in (None, h):
            return h
        log("%s: sha256 tutmuyor, yeniden indiriliyor" % path)
    assert url, "%s yok ve indirme adresi yok" % path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, n, t0 = hashlib.sha256(), 0, time.time()
    request = urllib.request.Request(url, headers={"User-Agent": "model_y data_simplestories"})
    with urllib.request.urlopen(request, timeout=120) as r, open(path + ".part", "wb") as f:
        for chunk in iter(lambda: r.read(1 << 23), b""):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    assert sha256 in (None, h.hexdigest()), "%s: indirilen sha256 %s, beklenen %s" % (url, h.hexdigest(), sha256)
    os.replace(path + ".part", path)
    log("indirildi %s  %.1f MB  %.0f sn" % (path, n / 1e6, time.time() - t0))
    return h.hexdigest()


def _write_exam(root, tags, seq_len, log):
    """Sinav kumesi (kullanici, 28 Eylul: "3 kabul"): butun tag'lerde seq_len'e sigan ve hicbir tag'de train'de birebir
    gecmeyen valid hikayeleri -> exam_stories.npy (valid hikaye sirasi, int64) + exam_stories.json (sha256, kural,
    tag izleri).  Varsa dokunmaz."""
    path = os.path.join(root, "exam_stories.npy")
    if os.path.exists(path) or not all(os.path.isdir(os.path.join(root, t)) for t in tags):
        return
    fits, leaked, sources = None, set(), {}
    for tag in tags:
        meta = json.load(open(os.path.join(root, tag, "fingerprint.json"), encoding="utf-8"))
        v, t = (np.load(os.path.join(root, tag, s + ".npy")) for s in ("valid", "train"))
        vs, vl = _stories(v, meta["eos_id"])
        f = (vl > 0) & (vl + 2 <= seq_len)
        fits = f if fits is None else fits & f
        leaked |= set(_found(t, *_stories(t, meta["eos_id"]), v, vs, vl).tolist())
        sources[tag] = meta["fingerprint"]
        del t
    exam = np.array(sorted(set(np.flatnonzero(fits).tolist()) - leaked), dtype=np.int64)
    np.save(path, exam)
    info = dict(sha256=hashlib.sha256(np.ascontiguousarray(exam)).hexdigest(), stories=len(exam), seq_len=seq_len,
                fits_all_tags=int(fits.sum()), leaked=sorted(leaked), tags=sources, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                rule="valid hikayesi butun tag'lerde L + 2 <= seq_len ve hicbir tag'de train'de birebir (token dizisi) degil")
    json.dump(info, open(os.path.join(root, "exam_stories.json"), "w", encoding="utf-8"), indent=1)
    log("sinav kumesi YAZILDI: %d hikaye (butun tag'lerde sigan %d, sizinti %s)  sha256 %s"
        % (len(exam), info["fits_all_tags"], info["leaked"], info["sha256"][:12]))


def build_cache(root, tags=TAGS, dataset=DATASET, tokenizers=TOKENIZERS, log=None, chunk=20000, seq_len=SEQ_LEN):
    """Tek seferlik uretim (kural 9).  parquet'ler raw/'a (varsa sha256 dogrulanir, yeniden indirilmez), tokenizer
    <tag>/'a; butun hikayeler (metin oldugu gibi) her tokenizer'la token'lanir.  Var olan <tag>/ ATLANIR (tazeleme ayri
    karar: klasor elle tasinir); is <tag>_partial/'da yapilir, bitince <tag>/ olur.  fingerprint.json'a: iz, kaynaklar,
    sayimlar (UNK, valid'de gidis-donus, metin duzeyinde sizinti ve tekrar).  dataset / tokenizers: testte yerel
    kaynak (repo yok, tokenizer "path").  Sonda sinav kumesi ve tag'lerin text_reference.json'u yoksa yazilir
    (_write_exam, seq_len; build_text_reference).  -> {tag: fingerprint.json icerigi}"""
    import pyarrow.parquet as pq
    from tokenizers.models import WordPiece
    os.makedirs(root, exist_ok=True)
    log = log or _root_log(root)

    def finish(out):                                   # sinav kumesi, sonra (sinav kumesi varsa) metin referanslari
        _write_exam(root, tags, seq_len, log)
        if os.path.exists(os.path.join(root, "exam_stories.npy")):
            for tag in tags:
                if os.path.isdir(os.path.join(root, tag)) and not os.path.exists(os.path.join(root, tag, _TEXT_REFERENCE)):
                    build_text_reference(root, tag, log)
        return out
    t0 = time.time()
    hf = "https://huggingface.co/%s/resolve/%s/%s"
    splits = ("valid", "train")                                      # valid once: kusur erken gorunsun
    raw = {}
    for split in splits:
        raw[split] = []
        for name in dataset["files"][split]:
            path = os.path.join(root, "raw", *name.split("/"))
            url = hf % (dataset["repo"], dataset["revision"], name) if dataset.get("repo") else None
            raw[split].append(dict(file=name, path=path, sha256=_fetch(url, path, dataset["sha256"].get(name), log)))
    work = {}
    for tag in tags:
        if os.path.isdir(os.path.join(root, tag)):
            log("%s: klasor var, ATLANDI (iz build'de dogrulanir; tazelemek icin klasoru elle tasi)" % tag)
            continue
        src, part = tokenizers[tag], os.path.join(root, tag + "_partial")
        if os.path.isdir(part):
            shutil.rmtree(part)
        os.makedirs(part)
        path = os.path.join(part, "tokenizer.json")
        if "path" in src:
            shutil.copyfile(src["path"], path)
        url = None if "path" in src else hf % (src["repo"], src["revision"], src["file"])
        sha = _fetch(url, path, src.get("sha256"), log)
        tok, vocab = _load(path, src["eos"])
        assert len(vocab) <= 65536, "%s: sozluk uint16'ya sigmiyor" % tag
        work[tag] = dict(tok=tok, vocab=vocab, eos=vocab.index(EOS_TOKEN), part=part, sha256=sha, src=src,
                         valid=[], train=[])
    if not work:
        return finish({})
    nbytes, text_hash = {s: [] for s in splits}, {s: [] for s in splits}
    for split in splits:
        done, total = 0, sum(pq.ParquetFile(r["path"]).metadata.num_rows for r in raw[split])
        for r in raw[split]:
            for rb in pq.ParquetFile(r["path"]).iter_batches(batch_size=chunk, columns=[dataset["column"]]):
                stories = rb.column(0).to_pylist()
                encoded = [s.encode("utf-8") for s in stories]
                nbytes[split].append(np.array([len(b) for b in encoded], dtype=np.int32))
                text_hash[split].append(np.array([int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")
                                                  for b in encoded], dtype=np.uint64))
                for tag, w in work.items():
                    ids = [e.ids for e in w["tok"].encode_batch_fast(stories, add_special_tokens=False)]
                    flat = np.fromiter(itertools.chain.from_iterable(x + [w["eos"]] for x in ids), dtype=np.int64,
                                       count=sum(len(x) for x in ids) + len(ids))
                    assert int((flat == w["eos"]).sum()) == len(ids), "%s %s: hikaye icinde eos" % (tag, split)
                    w[split].append(flat.astype(np.uint16))
                done += len(stories)
                if done == total or done % (10 * chunk) < chunk:
                    log("%s %d / %d hikaye  %s  (%.0f sn)" % (split, done, total, " ".join(
                        "%s %d token" % (t, sum(len(a) for a in w[split])) for t, w in work.items()), time.time() - t0))
    th = {s: np.concatenate(text_hash[s]) for s in splits}
    text_counts = dict(valid_text_in_train=int(np.isin(th["valid"], th["train"]).sum()),
                       valid_text_duplicates=int(len(th["valid"]) - len(np.unique(th["valid"]))),
                       train_text_duplicates=int(len(th["train"]) - len(np.unique(th["train"]))))
    valid_text = [s for r in raw["valid"] for s in pq.read_table(r["path"], columns=[dataset["column"]]).column(0).to_pylist()]
    out = {}
    for tag, w in work.items():
        arrays = {s: np.concatenate(w[s]) for s in splits}
        per_story = {s: np.concatenate(nbytes[s]) for s in splits}
        for s in splits:
            np.save(os.path.join(w["part"], s + ".npy"), arrays[s])
            np.save(os.path.join(w["part"], s + "_bytes.npy"), per_story[s])
        # valid'de gidis-donus: toplu encode = tek tek encode; gpt2 kayipsiz, WordPiece id'leri geri verir
        vocab, eos = w["vocab"], w["eos"]
        ends = np.flatnonzero(arrays["valid"] == eos)
        starts = np.concatenate([[0], ends[:-1] + 1])
        lossless = not isinstance(w["tok"].model, WordPiece)
        fast_differs = roundtrip_fails = 0
        for text, s, e in zip(valid_text, starts.tolist(), ends.tolist()):
            ids = arrays["valid"][s:e].tolist()
            fast_differs += encode(text, vocab) != ids
            back = decode(ids, vocab)
            roundtrip_fails += (back != text) if lossless else (encode(back, vocab) != ids)
        counts = {"%s_%s" % (s, k): v for s in splits for k, v in (
            ("stories", len(per_story[s])), ("tokens", len(arrays[s])), ("bytes", int(per_story[s].sum())))}
        if UNK_TOKEN in vocab:
            counts.update({s + "_unk": int((arrays[s] == vocab.index(UNK_TOKEN)).sum()) for s in splits})
        counts.update(text_counts, valid_fast_encode_differs=fast_differs,
                      valid_roundtrip_fails=roundtrip_fails,
                      valid_roundtrip=("decode(ids) == metin" if lossless else "encode(decode(ids)) == ids"))
        tokenizer_file = open(os.path.join(w["part"], "tokenizer.json"), "rb").read()
        fp, parts = fingerprint(vocab, arrays["train"], arrays["valid"], tokenizer=tokenizer_file,
                                train_bytes=per_story["train"], valid_bytes=per_story["valid"])
        meta = dict(tag=tag, fingerprint=fp, fingerprints=parts, eos=w["src"]["eos"], eos_id=eos, vocab_size=len(vocab),
                    tokenizer=dict({k: v for k, v in w["src"].items() if k != "path"}, sha256=w["sha256"]),
                    dataset=dict(repo=dataset.get("repo"), revision=dataset.get("revision"), column=dataset["column"],
                                 files={s: [dict(file=r["file"], sha256=r["sha256"]) for r in raw[s]] for s in splits}),
                    counts=counts, created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0))
        json.dump(meta, open(os.path.join(w["part"], "fingerprint.json"), "w", encoding="utf-8"), indent=1)
        os.rename(w["part"], os.path.join(root, tag))
        log("%s: YAZILDI iz %s  %s" % (tag, fp, json.dumps(counts)))
        out[tag] = meta
    return finish(out)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build_cache":
        build_cache(sys.argv[2], tuple(sys.argv[3:]) or TAGS)
    elif len(sys.argv) >= 3 and sys.argv[1] == "build_text_reference":
        for _tag in sys.argv[3:] or TAGS:
            build_text_reference(sys.argv[2], _tag)
    else:
        print(__doc__)
