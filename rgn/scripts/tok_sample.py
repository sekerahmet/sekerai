"""Tokenizer için Türkçe örnek metin: Hugging Face'ten indirir, eğitim / ölçüm parçalarına ayırır (Drive/tokenizer).

  wiki : wikimedia/wikipedia 20231101.tr (tamamı indirilir, 2 parquet, ~0,55 GB)
  web  : HuggingFaceFW/fineweb-2 tur_Latn, ilk dosyanın baştan satır grupları (yalnız gereken kadarı okunur)
Her kaynakta her 50. belge ölçüme (en çok --eval_mb), gerisi eğitime (en çok --train_mb).  Çıktı satır başına bir
belge (JSON: {"text": ...}): <drive>/tokenizer/ornek/{train,eval}_tr_{wiki,web}.jsonl

    python scripts/tok_sample.py --train_mb 300 200 --eval_mb 10
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("rgn_paths", Path(__file__).resolve().parents[1] / "rgn" / "paths.py")
drive_root = importlib.util.module_from_spec(_spec)                       # rgn paketini (torch) yüklemeden: pyarrow
_spec.loader.exec_module(drive_root)                                       # sonrası torch Windows'ta DLL hatası verir
drive_root = drive_root.drive_root

WIKI = ("wikimedia/wikipedia", ["20231101.tr/train-00000-of-00002.parquet", "20231101.tr/train-00001-of-00002.parquet"])
WEB = "datasets/HuggingFaceFW/fineweb-2/data/tur_Latn/train/000_00000.parquet"


def docs_wiki(raw):
    from huggingface_hub import hf_hub_download
    for f in WIKI[1]:
        p = hf_hub_download(WIKI[0], f, repo_type="dataset", local_dir=raw)
        pf = pq.ParquetFile(p)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i, columns=["text"]).column("text").to_pylist()


def docs_web():
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(WEB, "rb", block_size=16 << 20) as f:      # satır grubu başına yalnız o bayt aralığı okunur
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i, columns=["text"]).column("text").to_pylist()


def split(docs, out_train, out_eval, train_mb, eval_mb, every=50):
    tb = eb = n = 0
    with open(out_train, "w", encoding="utf-8") as ft, open(out_eval, "w", encoding="utf-8") as fe:
        for d in docs:
            if not d or not d.strip():
                continue
            b = len(d.encode("utf-8"))
            if n % every == 0:
                if eb < eval_mb * 1e6:
                    fe.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
                    eb += b
            elif tb < train_mb * 1e6:
                ft.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
                tb += b
            n += 1
            if tb >= train_mb * 1e6 and eb >= eval_mb * 1e6:
                break
    return tb / 1e6, eb / 1e6, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_mb", type=float, nargs=2, default=[300, 200], help="web, wiki eğitim metni (MB)")
    ap.add_argument("--eval_mb", type=float, default=10)
    a = ap.parse_args()
    root = drive_root() / "tokenizer"
    out = root / "ornek"
    out.mkdir(parents=True, exist_ok=True)
    for name, docs, mb in (("web", docs_web(), a.train_mb[0]), ("wiki", docs_wiki(root / "ham"), a.train_mb[1])):
        tb, eb, n = split(docs, out / ("train_tr_%s.jsonl" % name), out / ("eval_tr_%s.jsonl" % name), mb, a.eval_mb)
        print("%s: %d belge, eğitim %.0f MB, ölçüm %.1f MB" % (name, n, tb, eb), flush=True)


if __name__ == "__main__":
    main()
