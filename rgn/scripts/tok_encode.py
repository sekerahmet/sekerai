"""Türkçe veriyi seçilen tokenizer'la kodlar (Colab CPU'da uzun süre arka planda; kaldığı yerden sürer).

Kaynak: FineWeb-2 tur_Latn (Hugging Face, 30 parquet; dosya başına ~10 GB metin, ~2B token).  Her belge
tok_corpus.clean ile temizlenir (NFC, Latin dışı yazı > %5 atılır, < 20 karakter atılır), kodlanır, sonuna
<|endoftext|> (kimlik 0) eklenir.  Çıktı uint16 .npy parçaları (~250M token):
    <out>/train/fw2_<dosya>_<parça>.npy    <out>/valid.npy (son dosyanın baştan --valid_tokens kadarı; eğitime girmez)
    <out>/meta.json  (parça başına token; bitmiş dosyalar atlanır; yarım dosya son yazılan parçadan sürer)

    python scripts/tok_encode.py --tokenizer <drive>/tokenizer/egitilen/tr_bpe_40k_v2.json \\
        --out <drive>/v2/fineweb2_tr/tok40k --files 0 1 2       # bir kısmı; --files boş: hepsi
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tok_compare import to_tiktoken  # noqa: E402
from tok_corpus import FW_DIR, clean, fs, parquet_docs  # noqa: E402

EOS = 0
THREADS = os.cpu_count() or 2
PART = 250_000_000


def save(path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def encode_file(tok, src, out, tag, meta, batch=4000, limit=0, resume=None, on_part=None):
    """Bir parquet dosyasını kodlar; PART token'da bir parça yazar.  limit > 0: o kadar token sonra durur.
    resume: önceki çalışmanın son yazılan parçasından sonraki durum (rows = okunmuş kaynak satırı, part, sayaçlar) →
    o kadar satır atlanır, parça numarası sürer.  on_part(durum): her parçadan sonra (meta.json'a kayıt)."""
    r = resume or {}
    buf, docs, t0 = [], [], time.time()
    n_tok, n_bytes, n_docs, part = r.get("tokens", 0), r.get("bytes", 0), r.get("docs", 0), r.get("part", 0)
    skip, rows = r.get("rows", 0), 0

    def flush_docs():
        nonlocal n_tok, n_bytes, n_docs
        for ids in tok.encode_ordinary_batch(docs, num_threads=THREADS):
            ids = np.asarray(ids, dtype=np.uint16)
            buf.append(ids)
            buf.append(np.array([EOS], dtype=np.uint16))
            n_tok += len(ids) + 1
        n_bytes += sum(len(d.encode("utf-8")) for d in docs)
        n_docs += len(docs)
        docs.clear()

    def flush_part():
        nonlocal part, buf
        if not buf:
            return
        arr = np.concatenate(buf)
        buf = []
        name = "%s_%02d.npy" % (tag, part)
        save(out / name, arr)
        meta["parts"][name] = dict(tokens=int(len(arr)))
        part += 1
        if on_part:
            on_part(dict(rows=rows, part=part, tokens=n_tok, bytes=n_bytes, docs=n_docs))

    if skip:
        print("  %s: kaldığı yerden — %d satır atlanıyor, parça %d" % (tag, skip, part), flush=True)
    cur = 0
    for d in parquet_docs([src]):
        rows += 1
        if rows <= skip:
            continue
        c = clean(d)
        if c is not None:
            docs.append(c)
        if len(docs) >= batch:
            before = n_tok
            flush_docs()
            cur += n_tok - before
            if cur >= PART:
                flush_part()
                cur = 0
                print("  %s: %.0fM token, %.1f GB metin, %d belge, %.0f sn" % (tag, n_tok / 1e6, n_bytes / 1e9, n_docs,
                                                                              time.time() - t0), flush=True)
            if limit and n_tok >= limit:
                break
    if docs:
        flush_docs()
    flush_part()
    return n_tok, n_bytes, n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", type=int, nargs="*", default=[], help="FineWeb-2 dosya sırası (0–28); boş: hepsi")
    ap.add_argument("--valid_file", type=int, default=29)
    ap.add_argument("--valid_tokens", type=int, default=20_000_000)
    ap.add_argument("--limit_tokens", type=int, default=0, help="deneme: dosya başına en çok bu kadar token")
    a = ap.parse_args()
    tok = to_tiktoken(a.tokenizer)                                        # HF ile aynı token'lar, ~4× hızlı
    out = Path(a.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    mp = out / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else dict(tokenizer=a.tokenizer, files={}, parts={})
    src = sorted(fs().ls(FW_DIR, detail=False))
    if "valid" not in meta:                                            # doğrulama: son dosyanın başı
        vdir = out / "_valid"
        vdir.mkdir(exist_ok=True)
        n, b, k = encode_file(tok, src[a.valid_file], vdir, "valid", dict(parts={}), limit=a.valid_tokens)
        arr = np.concatenate([np.load(p) for p in sorted(vdir.glob("valid_*.npy"))])
        save(out / "valid.npy", arr)
        meta["valid"] = dict(tokens=int(len(arr)), bytes=b, docs=k, bytes_per_token=b / n, file=src[a.valid_file])
        mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print("doğrulama: %d token, bayt / token %.3f" % (len(arr), b / n), flush=True)
    for i in (a.files or [j for j in range(len(src)) if j != a.valid_file]):
        tag = "fw2_%02d" % i
        if tag in meta["files"]:
            print("atlandı (bitmiş): %s" % tag, flush=True)
            continue
        t0 = time.time()
        prog = meta.setdefault("progress", {})

        def on_part(state, tag=tag):                                    # parça yazıldı → kaldığı yer kaydı
            prog[tag] = state
            mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        n, b, k = encode_file(tok, src[i], out / "train", tag, meta, limit=a.limit_tokens, resume=prog.get(tag),
                              on_part=on_part)
        prog.pop(tag, None)
        meta["files"][tag] = dict(tokens=n, bytes=b, docs=k, seconds=round(time.time() - t0), src=src[i])
        mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tot = sum(f["tokens"] for f in meta["files"].values())
        print("%s bitti: %.0fM token, %.1f GB metin, %.0f dk | toplam %.2fB token" % (
            tag, n / 1e6, b / 1e9, (time.time() - t0) / 60, tot / 1e9), flush=True)


if __name__ == "__main__":
    main()
