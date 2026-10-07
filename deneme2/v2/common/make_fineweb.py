"""make_fineweb -- FineWeb-Edu shard'ini V2 bicimine cevirir (belge 47 plan, 48 uygulama).  Kaynak klasor SALT OKUNUR:
hicbir dosyasi yazilmaz / silinmez (cikti kaynagin icinde olamaz; kaynak dosyalar yalniz okuma kipinde acilir).

    prepare  <src>/gpt2/shard_NNN.{bin,json}, _offsets.npy, _bytes.npy, tokenizer.json -> <out>/ (stream_root = data_dir):
             gpt2/{train,valid}.npy (belge belge metin + kapanis eot; SS akisi bicimi), gpt2/valid_bytes.npy,
             gpt2/tokenizer.json, gpt2/{train,valid}_doc_ids.npy (shard belge indeksi), {split}_sentence_offsets /
             _story_offsets / _boundaries (data.build_boundaries, profile="web"), train_story_continues.npy (uzun belge
             cumle sinirinda satira sigan parcalara; devam eden parcada EOS hedefi yok), exam_stories.{npy,json},
             train_pack_plan_e1.npz, exam_pack_plan.npz, train_token_counts.npy, reading_prompts.json, source.json (en son).
    gate     cumle bolme kapisi: rastgele N train belgesinde web profili sinirlari = satir + nltk punkt sinirlari mi
             (kesinlik / duyarlilik, uyusmayan ornekler siniflanmis, cumle boyu, zorla bolunen oran).

Ayirma: bos (bosluk-yalniz) ya da icinde ikinci eot olan belge duser; valid = shard belge indeksi % VALID_STRIDE == 0,
train = kalan.  exam: valid'de satira sigan (1 + sum(L + 1) <= ROW_LEN), train'de birebir ya da ilk 64 token'i ayni
belgesi olmayan belgelerden tohum 0 permutasyonunun ilk EXAM_DOCS'u, sirali.  Cumle duzeyi sizinti (>= 8 token'lik exam
cumlelerinin train'de birebir gecme orani) source.json'a yazilir.

    python make_fineweb.py prepare --src <fineweb koku> --out <v2/fineweb_edu_s000> [--local <yerel kopya>] [--workers N]
    python make_fineweb.py gate --src <fineweb koku> [--docs 1000] [--row_groups 0 (hepsi)] [--out <rapor klasoru>]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

SHARD = 0
VALID_STRIDE = 50           # data_fineweb ile ayni adim (orada shard_013), burada shard_000 (kullanici, 7 Ekim: "1 parça")
EXAM_DOCS = 1000            # SS'teki 1.000 hikaye karsiligi
PREFIX_TOKENS = 64          # yakin kopya: ilk 64 token
LEAK_MIN_TOKENS = 8         # cumle sizintisi bu boydan
READING_DOCS = 10           # reading_prompts.json: 8 greedy + 2 sample (SS duzeni)
GPT2_TOKENIZER_CANON_SHA = "347233c4a8bf33f5ca7884e9db53207443731e17eb1238967e4e221f4e62fba1"   # SS tokenizer'inin icerik sha'si (_tokenizer_canon_sha)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def _tokenizer_canon_sha(path):
    """Icerik sha'si: vocab, merges, added_tokens, normalizer / pre_tokenizer / post_processor / decoder; surume bagli
    varsayilan alanlar (use_regex True, type BPE, byte_fallback / ignore_merges False) atilir. 7 Ekim: fineweb/gpt2 tokenizer.json'u
    yeni surumle kaydedilmis, dosya sha'si SS'ten farkli; vocab + merges ayni, 2.000 belge birebir ayni token."""
    t = json.load(open(path, encoding="utf-8"))
    m = dict(t["model"])
    m["merges"] = [" ".join(x) if isinstance(x, list) else x for x in m["merges"]]
    for k, v in (("type", "BPE"), ("byte_fallback", False), ("ignore_merges", False)):
        if m.get(k) == v:
            m.pop(k)
    parts = {k: dict(t[k]) if isinstance(t.get(k), dict) else t.get(k)
             for k in ("normalizer", "pre_tokenizer", "post_processor", "decoder")}
    for v in parts.values():
        if isinstance(v, dict) and v.get("use_regex") is True:
            v.pop("use_regex")
    canon = dict(model=m, added_tokens=t.get("added_tokens"), **parts)
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _guard(src, out):
    """Cikti kaynagin icinde (ya da kaynak) olamaz."""
    s, o = os.path.realpath(src), os.path.realpath(out)
    assert not (o == s or o.startswith(s.rstrip(os.sep) + os.sep)), "DUR: cikti kaynak klasorun icinde: %s" % out


def _shard_files(root, i=SHARD):
    base = os.path.join(root, "gpt2", "shard_%03d" % i)
    return dict(bin=base + ".bin", json=base + ".json", offsets=base + "_offsets.npy", bytes=base + "_bytes.npy",
                tokenizer=os.path.join(root, "gpt2", "tokenizer.json"))


def _local_copy(src, local):
    """Kaynak shard dosyalari -> yerel kopya (yalniz okunur kaynaktan; varsa ve boyu tutuyorsa atlanir)."""
    files = _shard_files(src)
    out = _shard_files(local)
    os.makedirs(os.path.join(local, "gpt2"), exist_ok=True)
    for k, p in files.items():
        if not (os.path.exists(out[k]) and os.path.getsize(out[k]) == os.path.getsize(p)):
            shutil.copyfile(p, out[k] + ".part")
            os.replace(out[k] + ".part", out[k])
    return local


def _docs(files):
    """-> (akis uint16 mmap ([eot] + metin), ofsetler, belge boylari (eot dahil), baytlar, shard json)."""
    meta = json.load(open(files["json"], encoding="utf-8"))
    off = np.load(files["offsets"])
    x = np.memmap(files["bin"], dtype=np.uint16, mode="r")
    assert len(x) == meta["tokens"] and len(off) == meta["docs"], "shard json ile akis / ofset uyusmuyor"
    L = np.r_[off[1:], len(x)] - off
    return x, off, L, np.load(files["bytes"]), meta


def _keep(x, off, tok):
    """Belge basina: en az bir bosluk-disi token ve tek eot (basta) -> bool."""
    flags, _ = D.stream_tables(tok)
    space = (flags & D._FLAG["space"]) != 0                              # sozluk tablosu (bellek: akis basina bool)
    eos = np.asarray(x == D.EOS_ID)
    word = ~space[np.asarray(x)] & ~eos
    return (np.add.reduceat(word, off) > 0) & (np.add.reduceat(eos, off) == 1), flags


def _write_stream(x, off, L, ids, path):
    """Secilen belgeler -> metin + kapanis eot (SS akisi bicimi), uint16 .npy."""
    rot = np.empty(len(x), np.uint16)
    rot[:-1] = x[1:]
    rot[-1] = D.EOS_ID
    sel = np.zeros(len(off), bool)
    sel[ids] = True
    np.save(path, rot[np.repeat(sel, L)])


def _intro(text, n_tokens):
    """Parcanin sonunda kalmamasi gereken satir: ':' ile biter ya da kisa baslik (<= 8 token, satir sonu, noktalamasiz)."""
    t = text.rstrip()
    return t.endswith(":") or (text.endswith("\n") and n_tokens <= 8 and not t.rstrip("\"')]}\u201d\u2019").endswith(
        (".", "!", "?")))


def _piece_starts(L1, is_intro, row_len):
    """Cumle boylari (L + 1) -> parca baslangiclari (0'dan): 1 + sum(L + 1) <= row_len, sirali; parcanin son cumlesi giris
    / baslik ise (is_intro(i)) sonraki parcaya gecer."""
    starts, k, b = [], 0, len(L1)
    while k < b:
        n = max(1, int(np.searchsorted(np.cumsum(L1[k:]), row_len - 1, "right")))
        if 1 < n and k + n < b and is_intro(k + n - 1):
            n -= 1
        starts.append(k)
        k += n
    return starts


def _pieces(st, row_len, tok):
    """Uzun hikaye (belge) -> cumle sinirinda parcalar (_piece_starts).  -> (yeni story ofsetleri, continues, parca basina
    belge sirasi)."""
    L1 = st.sent[:, 1] - st.sent[:, 0] + 1
    starts, cont, doc = [], [], []
    for i in range(st.n):
        a, b = int(st.story[i]), int(st.story[i + 1])
        if 1 + int(L1[a:b].sum()) <= row_len:
            ks = [0]
        else:
            ks = _piece_starts(L1[a:b], lambda j: _intro(tok.decode(np.asarray(st.stream[st.sent[a + j, 0]:st.sent[a + j, 1]])
                                                                   .astype(np.int64).tolist()), int(L1[a + j] - 1)), row_len)
        for m, k in enumerate(ks):
            starts.append(a + k)
            cont.append(m + 1 < len(ks))
            doc.append(i)
    return np.r_[np.array(starts, np.int64), len(st.sent)], np.array(cont, bool), np.array(doc, np.int64)


def _hash(a):
    return hashlib.blake2b(np.ascontiguousarray(a, dtype=np.uint16).tobytes(), digest_size=8).digest()


def prepare(args):
    from tokenizers import Tokenizer
    t0 = time.time()
    log = lambda m: print("[%6.0f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    _guard(args.src, args.out)
    if args.local:
        _guard(args.src, args.local)
    src = _local_copy(args.src, args.local) if args.local else args.src
    files = _shard_files(src)
    x, off, L, nbytes, meta = _docs(files)
    sha_bin = _sha256(files["bin"])
    assert sha_bin == meta["sha256"], "DUR: shard akisinin sha256'si shard json'dakiyle ayni degil"
    tok_sha = _sha256(files["tokenizer"])
    assert _tokenizer_canon_sha(files["tokenizer"]) == GPT2_TOKENIZER_CANON_SHA, "DUR: tokenizer SS'tekiyle ayni degil"
    log("kaynak: %s | %d belge, %d token, sha tutuyor" % (files["bin"], meta["docs"], meta["tokens"]))
    out = args.out
    os.makedirs(os.path.join(out, "gpt2"), exist_ok=True)
    shutil.copyfile(files["tokenizer"], os.path.join(out, "gpt2", "tokenizer.json"))
    tok = Tokenizer.from_file(files["tokenizer"])
    keep, flags = _keep(x, off, tok)
    idx = np.arange(len(off))
    valid_ids = idx[keep & (idx % VALID_STRIDE == 0)]
    train_ids = idx[keep & (idx % VALID_STRIDE != 0)]
    for split, ids in (("train", train_ids), ("valid", valid_ids)):
        _write_stream(x, off, L, ids, os.path.join(out, "gpt2", split + ".npy"))
        np.save(os.path.join(out, "gpt2", split + "_doc_ids.npy"), ids)
    np.save(os.path.join(out, "gpt2", "valid_bytes.npy"), nbytes[valid_ids].astype(np.int64))
    log("ayirma: train %d, valid %d belge, dusen %d (bos ya da ic eot)" % (len(train_ids), len(valid_ids), int((~keep).sum())))
    metas = {sp: D.build_boundaries(out, out, sp, profile="web", workers=args.workers) for sp in ("valid", "train")}
    longest = max(m["max_sentence_tokens"] for m in metas.values())
    for sp, m in metas.items():
        json.dump(dict(m, max_sentence_tokens_all=longest), open(os.path.join(out, sp + "_boundaries.json"), "w",
                                                                 encoding="utf-8"), indent=1)
    train = D.TokenStories(out, out, "train")
    valid = D.TokenStories(out, out, "valid")
    assert train.n == len(train_ids) and valid.n == len(valid_ids), "bos hikaye dustu: belge / hikaye sirasi kaydi"
    story, cont, pdoc = _pieces(train, D.ROW_LEN, tok)
    np.save(os.path.join(out, "train_story_offsets.npy"), story)
    np.save(os.path.join(out, "train_story_continues.npy"), cont)
    bj = os.path.join(out, "train_boundaries.json")
    m = json.load(open(bj, encoding="utf-8"))
    json.dump(dict(m, stories=int(len(cont)), documents=int(len(train_ids)), pieces_continuing=int(cont.sum()),
                   piece_rule="belge 47 s3 A: 1 + sum(L + 1) <= %d, cumle sinirinda, sirali; parca sonundaki giris / baslik "
                               "satiri sonraki parcaya" % D.ROW_LEN),
              open(bj, "w", encoding="utf-8"), indent=1)
    train = D.TokenStories(out, out, "train")
    log("parca: %d belge -> %d parca (%d devam eden)" % (len(train_ids), train.n, int(cont.sum())))
    # sinav: tam belge, satira sigan, train'de birebir / ilk 64 token kopyasi olmayan
    tr_doc = np.r_[train.story[np.r_[np.flatnonzero(~np.r_[False, cont[:-1]])]], len(train.sent)]   # belge bas cumlesi
    full, pre = set(), set()
    for a, b in zip(tr_doc[:-1], tr_doc[1:]):
        s, t = int(train.sent[a, 0]), int(train.sent[b - 1, 1])
        full.add(_hash(train.stream[s:t]))
        pre.add(_hash(train.stream[s:min(t, s + PREFIX_TOKENS)]))
    vl = valid.lengths()
    leak_full, leak_pre, cand = 0, 0, []
    for i in np.flatnonzero(vl <= D.ROW_LEN).tolist():
        s, t = int(valid.sent[valid.story[i], 0]), int(valid.sent[valid.story[i + 1] - 1, 1])
        f, p = _hash(valid.stream[s:t]) in full, _hash(valid.stream[s:min(t, s + PREFIX_TOKENS)]) in pre
        leak_full += f
        leak_pre += p and not f
        if not (f or p):
            cand.append(i)
    cand = np.array(cand, dtype=np.int64)
    pick = np.sort(cand[np.random.default_rng(0).permutation(len(cand))[:EXAM_DOCS]])
    sents = set()
    for a, b in train.sent[(train.sent[:, 1] - train.sent[:, 0]) >= LEAK_MIN_TOKENS].tolist():
        sents.add(_hash(train.stream[a:b]))
    ex = [(a, b) for i in pick.tolist() for a, b in valid.sent[valid.story[i]:valid.story[i + 1]].tolist()
          if b - a >= LEAK_MIN_TOKENS]
    sent_leak = sum(_hash(valid.stream[a:b]) in sents for a, b in ex) / max(1, len(ex))
    sha = hashlib.sha256(np.ascontiguousarray(pick)).hexdigest()
    np.save(os.path.join(out, "exam_stories.npy"), pick)
    rule = ("valid'de 1 + sum(L + 1) <= %d, train'de birebir ya da ilk %d token'i ayni belgesi olmayanlardan tohum 0 "
            "permutasyonunun ilk %d'i, sirali" % (D.ROW_LEN, PREFIX_TOKENS, EXAM_DOCS))
    json.dump(dict(sha256=sha, stories=int(len(pick)), rule=rule, candidates=int(len(cand)), fits=int((vl <= D.ROW_LEN).sum()),
                   leaked_full=int(leak_full), leaked_prefix=int(leak_pre), sentence_leak_rate=round(sent_leak, 4),
                   sentence_leak_rule="exam cumleleri (>= %d token) train cumlelerinde birebir" % LEAK_MIN_TOKENS),
              open(os.path.join(out, "exam_stories.json"), "w", encoding="utf-8"), indent=1)
    log("sinav: %d aday, %d secildi; sizinti birebir %d, ilk %d token %d; cumle sizintisi %.4f" % (
        len(cand), len(pick), leak_full, PREFIX_TOKENS, leak_pre, sent_leak))
    ro, rs = D.pack_plan(train.lengths(), D.ROW_LEN, 0, 1)
    np.savez(os.path.join(out, "train_pack_plan_e1.npz"), row_offsets=ro, row_stories=rs, seed=0, epoch=1, row_len=D.ROW_LEN)
    fill = train.lengths().sum() / ((len(ro) - 1) * D.ROW_LEN)
    ro2, rs2 = D.pack_plan(vl[pick], D.ROW_LEN, None)
    np.savez(os.path.join(out, "exam_pack_plan.npz"), row_offsets=ro2, row_stories=pick[rs2].astype(np.int32), seed=-1,
             epoch=0, row_len=D.ROW_LEN, exam_set_sha256=sha)
    log("paket: train %d satir (%d adim, doluluk %.3f), sinav %d satir" % (len(ro) - 1, -(-(len(ro) - 1) // D.BATCH_ROWS),
                                                                        fill, len(ro2) - 1))
    D.token_counts(out, out)
    many = [int(i) for i in pick if valid.story[i + 1] - valid.story[i] >= 6][:READING_DOCS]
    json.dump(dict(rule="exam belgelerinden ilk %d tanesi (>= 6 cumle), ilk 3 cumle istem; 8 greedy + 2 sample" % READING_DOCS,
                   prompts=[dict(label="okuma %d" % (k + 1), story=s, sentences=3, decode="greedy" if k < 8 else "sample")
                            for k, s in enumerate(many)]),
              open(os.path.join(out, "reading_prompts.json"), "w", encoding="utf-8"), indent=1)
    src_meta = dict(source=dict(shard=meta["shard"], parquet=meta["source"], parquet_sha256=meta["source_sha256"],
                                stream_sha256=sha_bin, tokenizer_sha256=tok_sha, docs=meta["docs"], tokens=meta["tokens"]),
                    split=dict(rule="bos / ic eot'lu belge duser; valid = indeks %% %d == 0; train = kalan" % VALID_STRIDE,
                               train_docs=int(len(train_ids)), valid_docs=int(len(valid_ids)), dropped=int((~keep).sum())),
                    streams={sp: _sha256(os.path.join(out, "gpt2", sp + ".npy")) for sp in ("train", "valid")},
                    sentences=dict(profile="web", max_sentence_tokens=D.MAX_SENTENCE_TOKENS, longest=longest,
                                   forced_splits={sp: m["forced_splits"] for sp, m in metas.items()}),
                    pieces=dict(row_len=D.ROW_LEN, train_pieces=int(train.n), continuing=int(cont.sum())),
                    exam=json.load(open(os.path.join(out, "exam_stories.json"), encoding="utf-8")),
                    created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1))
    json.dump(src_meta, open(os.path.join(out, "source.json"), "w", encoding="utf-8"), indent=1)
    log("BITTI: %s" % out)
    return src_meta


# --- cumle bolme kapisi
_URL = re.compile(r"https?://|www\.|\.(com|org|net|edu|gov)\b", re.I)
_NUMLINE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\d{1,3}|[a-zA-Z])\.\s*$")   # satir basi liste numarasi (bilerek birlesik)
_LIST = re.compile(r"^\s*([-*•·]|\d+[.)]|[a-z][.)])\s", re.I)


def _punkt():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import PunktTokenizer
    return PunktTokenizer("english")


def _norm(text, pos):
    """Sinir karakteri -> ilk bosluk-disi karakter."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _reference(text, punkt):
    """punkt (butun metin) + yeni satir sinirlari (satir kucuk harfle baslamiyorsa: baslik, madde; kucuk harfle devam
    eden satir sonu sarmasi sinir degil) -> karakter (belge basi haric)."""
    out = {_norm(text, a) for a, _ in punkt.span_tokenize(text)}
    for m in re.finditer("\n", text):
        p = _norm(text, m.end())
        if p < len(text) and not text[p].islower():
            out.add(p)
    out.discard(_norm(text, 0))
    return {p for p in out if p < len(text)}


def _category(text, p):
    """Uyusmayan sinir -> kaba sinif (oneri; gozle dogrulanir)."""
    left, right = text[max(0, p - 40):p], text[p:p + 40]
    line = text[text.rfind("\n", 0, p) + 1:p + 40]
    if _URL.search(left[-25:] + right[:25]):
        return "URL"
    if re.search(r"\d\W*$", left) or re.match(r"^\W*\d", right):
        return "sayi"
    if re.search(r"(\b[A-Z]|\b\w{1,4})\.\s*$", left):
        return "kisaltma"
    if _LIST.match(line) or ";" in left[-3:]:
        return "liste"
    if re.search("[\"“”'’]", left[-3:] + right[:3]):
        return "tirnak"
    if "\n" in left[-2:] or "\n" in right[:2]:
        return "baslik / satir"
    return "gercek hata?"


def gate(args):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    t0 = time.time()
    files = _shard_files(args.src)
    x, off, L, _, meta = _docs(files)
    tok = Tokenizer.from_file(files["tokenizer"])
    pf = pq.ParquetFile(os.path.join(args.src, "raw", os.path.basename(meta["source"])))
    rng = np.random.default_rng(args.seed)
    groups = np.arange(pf.num_row_groups)
    if args.row_groups:
        groups = np.sort(rng.permutation(groups)[:args.row_groups])
    size = pf.metadata.row_group(0).num_rows
    pool = np.concatenate([np.arange(g * size, min((g + 1) * size, meta["docs"])) for g in groups])
    pool = pool[pool % VALID_STRIDE != 0]
    pick = np.sort(rng.permutation(pool)[:args.docs])
    punkt = _punkt()
    web = D.stream_tables(tok, "web")
    keep_max = D.MAX_SENTENCE_TOKENS
    tp = fp = fn = listnum = 0
    forced = n_sent = 0
    lens, mism, bad_tok = [], [], 0
    by_group = {}
    for i in pick.tolist():
        by_group.setdefault(i // size, []).append(i)
    for g, docs in by_group.items():
        texts = pf.read_row_group(int(g), columns=["text"]).column("text").to_pylist()
        for i in docs:
            text = texts[i - g * size]
            enc = tok.encode(text)
            ids = np.asarray(x[off[i] + 1:off[i] + L[i]]).astype(np.int64)
            if enc.ids != ids.tolist():
                bad_tok += 1
                continue
            s = np.r_[ids, D.EOS_ID]
            b, _, _, f = D._boundaries(s, *web, tok, "web")
            forced += f
            n_sent += len(b)
            lens += (b[:, 1] - b[:, 0]).tolist()
            D.MAX_SENTENCE_TOKENS = 10 ** 9                               # kiyas: zorla bolme haric
            try:
                b2 = D._boundaries(s, *web, tok, "web")[0]
            finally:
                D.MAX_SENTENCE_TOKENS = keep_max
            ours = {_norm(text, enc.offsets[a][0]) for a in b2[1:, 0].tolist()} - {len(text)}
            ref = _reference(text, punkt)
            tp, fp, fn = tp + len(ours & ref), fp + len(ours - ref), fn + len(ref - ours)
            listnum += sum(bool(_NUMLINE.match(text[text.rfind("\n", 0, p) + 1:p])) for p in ref - ours)
            mism += [(i, "bizde var, punkt'ta yok", p) for p in sorted(ours - ref)]
            mism += [(i, "punkt'ta var, bizde yok", p) for p in sorted(ref - ours)]
    texts_all = {}
    sample = [mism[j] for j in rng.permutation(len(mism))[:args.examples]] if mism else []
    for g in sorted({i // size for i, _, _ in sample}):
        texts_all[g] = pf.read_row_group(int(g), columns=["text"]).column("text").to_pylist()
    rows = []
    for i, kind, p in sample:
        text = texts_all[i // size][i - (i // size) * size]
        ctx = (text[max(0, p - 70):p] + " || " + text[p:p + 70]).replace("\n", "\\n")
        rows.append(dict(doc=i, kind=kind, category=_category(text, p), context=ctx))
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    lens = np.array(lens)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = dict(docs=int(len(pick)), row_groups=int(len(groups)), token_mismatch_docs=bad_tok,
               reference="nltk punkt (english) butun metinde + kucuk harfle baslamayan yeni satir; sinir = cumlenin ilk "
                         "bosluk-disi karakteri",
               method="bizim sinir: token baslangicinin karakter ofseti (tokenizer offsets; token'lar shard'la birebir), ilk "
                      "bosluk-disi karaktere; zorla bolme (MAX_SENTENCE_TOKENS) kiyasta kapali",
               precision=round(prec, 4), recall=round(rec, 4), f1=round(2 * prec * rec / max(1e-9, prec + rec), 4),
               tp=tp, ours_only=fp, ref_only=fn, sentences=n_sent, ref_only_listnum=listnum,
               recall_listnum_ok=round((tp + listnum) / max(1, tp + fn), 4),
               forced_rate=round(forced / max(1, n_sent), 5),
               length=dict(median=int(np.median(lens)), p95=int(np.percentile(lens, 95)), max=int(lens.max())),
               mismatch_categories=cats, examples=rows, seconds=round(time.time() - t0, 1))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "sentence_gate.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1, ensure_ascii=False), flush=True)
    return res


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True, help="fineweb koku (SALT OKUNUR)")
    p.add_argument("--out", required=True)
    p.add_argument("--local", default=None, help="kaynak shard'in yerel kopyasi (Colab /content/...)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    g = sub.add_parser("gate")
    g.add_argument("--src", required=True)
    g.add_argument("--docs", type=int, default=1000)
    g.add_argument("--row_groups", type=int, default=0, help="0: hepsi; yerelde kucuk ornek icin birkac row group")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--examples", type=int, default=50)
    g.add_argument("--out", default=".")
    return ap.parse_args(argv)


def main(argv=None):
    if os.name == "nt":
        sys.path.insert(0, HERE)
        import train as TR
        TR._no_power_throttling()
    args = _args(argv)
    return prepare(args) if args.cmd == "prepare" else gate(args)


if __name__ == "__main__":
    main()
