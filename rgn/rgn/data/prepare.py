"""Ham metin -> GPT-2 BPE token akışı (uint16 .bin) + meta.json.

Düzen SS ile aynı: her belge "metin + EOS".  tiktoken gerekir (pip install tiktoken).

    python scripts/prepare_data.py --dataset tinystories
"""
import json
import time

import numpy as np

from rgn.data.registry import EOS_ID
from rgn.paths import drive_root

SEPARATOR = "<|endoftext|>"


def _encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _iter_documents(path, chunk_chars=64 << 20):
    """Büyük metin dosyasını belleğe tümüyle almadan ayırıcıya göre belgelere böler."""
    buf = ""
    with open(path, encoding="utf-8") as f:
        while True:
            block = f.read(chunk_chars)
            if not block:
                break
            buf += block
            parts = buf.split(SEPARATOR)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    yield p
    buf = buf.strip()
    if buf:
        yield buf


def tokenize_text_file(src, dst, batch_docs=4096, threads=8):
    """src (ayırıcılı metin) -> dst (.bin, uint16).  -> (belge, token, bayt) sayıları."""
    enc = _encoder()
    n_docs = n_tok = n_bytes = 0
    t0 = time.time()
    tmp = dst.with_suffix(".bin.part")
    with open(tmp, "wb") as out:
        batch = []

        def flush():
            nonlocal n_docs, n_tok, n_bytes
            for ids, text in zip(enc.encode_ordinary_batch(batch, num_threads=threads), batch):
                arr = np.asarray(ids + [EOS_ID], dtype=np.uint16)
                out.write(arr.tobytes())
                n_docs, n_tok, n_bytes = n_docs + 1, n_tok + len(arr), n_bytes + len(text.encode("utf-8"))
            batch.clear()

        for doc in _iter_documents(src):
            batch.append(doc)
            if len(batch) >= batch_docs:
                flush()
                if n_docs % (batch_docs * 50) == 0:
                    print("  %s: %d belge, %.1fM token, %.0f s" % (dst.name, n_docs, n_tok / 1e6, time.time() - t0),
                          flush=True)
        if batch:
            flush()
    tmp.replace(dst)
    return n_docs, n_tok, n_bytes


def count_bytes(path, chunk=50_000_000):
    """Hazır .bin dosyasının metin bayt sayısı (EOS hariç): token'lar çözülerek sayılır."""
    enc = _encoder()
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n_bytes = 0
    for i in range(0, len(arr), chunk):
        ids = np.asarray(arr[i:i + chunk])
        n_bytes += len(enc.decode_bytes(ids[ids != EOS_ID].tolist()))
    return n_bytes


def prepare_tinystories():
    root = drive_root() / "tinystories"
    out = root / "gpt2"
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({"source": "TinyStoriesV2-GPT4", "tokenizer": "gpt2 (tiktoken)", "eos_id": EOS_ID, "layout": "metin + EOS"})

    def save_meta():                                   # her bölümden sonra: yarıda kesilirse yapılan iş kaybolmaz
        meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    for split, fname in (("valid", "TinyStoriesV2-GPT4-valid.txt"), ("train", "TinyStoriesV2-GPT4-train.txt")):
        dst = out / ("%s.bin" % split)
        if dst.exists():
            print("[hazır] %s var, atlandı" % dst)
            meta.setdefault("%s_tokens" % split, dst.stat().st_size // 2)
            if "%s_bytes" % split not in meta:
                print("  bayt sayısı eksik, token'lardan sayılıyor ...", flush=True)
                meta["%s_bytes" % split] = count_bytes(dst)
            save_meta()
            continue
        print("[tokenize] %s" % (root / fname), flush=True)
        docs, toks, nbytes = tokenize_text_file(root / fname, dst)
        meta.update({"%s_docs" % split: docs, "%s_tokens" % split: toks, "%s_bytes" % split: nbytes})
        save_meta()
    print("[tamam] %s" % meta_path)
    return meta


PREPARERS = {"tinystories": prepare_tinystories}
