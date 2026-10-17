"""Tokenizer derlemi (Türkçe, çeşitli kaynak): Hugging Face'ten yalnız gereken kısmı okur, temizler, eğitim / ölçüm
parçalarına ayırır.  Çıktı: <drive>/tokenizer/derlem/{train,eval}_<kaynak>.jsonl (satır başına {"text": ...}).

Temizlik: Unicode NFC (İ gibi harflerin iki farklı kodlanışı tek); harflerin > %5'i Latin dışı yazı (Kiril, Arap,
Gürcü, Yunan, Çin …) olan belge atılır; 20 karakterden kısa belge atılır.
Ölçüm parçası eğitimden ayrı dosyadan (yoksa her 50. belge); HPLT yalnız ölçüm (hiç görülmemiş web taraması).

  kaynak       lisans        eğitim                             ölçüm
  fineweb2     ODC-By        9 dosyaya yayılmış (0, 3, … 24)     dosya 27
  hplt         CC0           —                                  tur_Latn_1 / _3 (başka tarama)
  wiki         CC BY-SA      tokenizer/ornek'ten (önceki örnek)  aynı
  ozenli       CC BY-SA 4.0  BellaTurca ÖzenliDerlem cfile 1–40  cfile 200–216
  akademik     CC BY-SA 4.0  AkademikDerlem dergi_17             dergi_10
  cosmos       CC BY 4.0     Cosmos-Turkish-Corpus 0–3           dosya 113
  instructurca CC BY-SA 4.0  InstrucTurca (Input + Output)       her 50. belge

    python scripts/tok_corpus.py
"""
import argparse
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location("rgn_paths", Path(__file__).resolve().parents[1] / "rgn" / "paths.py")
_paths = importlib.util.module_from_spec(_spec)        # rgn paketini (torch) yüklemeden: pyarrow sonrası torch Windows'ta
_spec.loader.exec_module(_paths)                       # DLL hatası verir
drive_root = _paths.drive_root

FW_DIR = "datasets/HuggingFaceFW/fineweb-2/data/tur_Latn/train"
BT = "datasets/turkish-nlp-suite/BellaTurca/"
CO = "datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0/data/train-%05d-of-00114.parquet"
HP = "datasets/HPLT/HPLT2.0_cleaned/tur_Latn_%d"
IT = "datasets/turkish-nlp-suite/InstrucTurca/data/train.jsonl"


def fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def parquet_docs(paths, col="text"):
    f = fs()
    for p in paths:
        with f.open(p, "rb", block_size=16 << 20) as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=2000, columns=[col]):
                yield from batch.column(0).to_pylist()


def jsonl_docs(paths, fn=lambda d: d["text"]):
    f = fs()
    for p in paths:
        with f.open(p, "rb", block_size=8 << 20) as fh:
            for line in fh:
                try:
                    yield fn(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue


def local_docs(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["text"]


def latin(c):
    o = ord(c)
    return o < 0x250 or 0x1E00 <= o < 0x1F00                              # Latin, Latin-1, Genişletilmiş A / B / Ek


def clean(t, max_foreign=0.05, min_chars=20):
    if not t or len(t) < min_chars:
        return None
    t = unicodedata.normalize("NFC", t)
    letters = [c for c in t if c.isalpha()]
    if not letters or sum(not latin(c) for c in letters) > max_foreign * len(letters):
        return None
    return t


def take(docs, mb, out, stats, every=0, eval_out=None, eval_mb=0):
    """docs'tan temizlenmiş belgeleri out'a (en çok mb MB); every > 0: her every. belge eval_out'a (en çok eval_mb)."""
    tb = eb = n = drop = 0
    with open(out, "a", encoding="utf-8") as ft:
        fe = open(eval_out, "a", encoding="utf-8") if eval_out else None
        for d in docs:
            c = clean(d)
            if c is None:
                drop += 1
                continue
            b = len(c.encode("utf-8"))
            if fe and every and n % every == 0:
                if eb < eval_mb * 1e6:
                    fe.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
                    eb += b
            elif tb < mb * 1e6:
                ft.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
                tb += b
            n += 1
            if tb >= mb * 1e6 and (not fe or eb >= eval_mb * 1e6):
                break
        if fe:
            fe.close()
    stats.append((Path(out).name, tb / 1e6, eb / 1e6, n, drop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_mb", type=float, default=10)
    ap.add_argument("--only", nargs="*", default=[], help="yalnız bu kaynaklar")
    a = ap.parse_args()
    root = drive_root() / "tokenizer"
    out = root / "derlem"
    out.mkdir(parents=True, exist_ok=True)
    E = a.eval_mb
    fw = sorted(fs().ls(FW_DIR, detail=False))                           # 30 dosya (000_00000 … 000_00029)
    FW = lambda i: fw[i]
    jobs = {
        "fineweb2": lambda st: ([take(parquet_docs([FW(i)]), 150, out / "train_fineweb2.jsonl", st) for i in range(0, 25, 3)],
                                take(parquet_docs([FW(27)]), 0, out / "train_fineweb2.jsonl", st, 1,
                                     out / "eval_fineweb2.jsonl", E)),
        "hplt": lambda st: [take(parquet_docs([sorted(fs().ls(HP % i, detail=False))[0]]), 0, out / "train_hplt.jsonl", st, 1, out / "eval_hplt.jsonl", E / 2)
                            for i in (1, 3)],
        "wiki": lambda st: (take(local_docs(root / "ornek" / "train_tr_wiki.jsonl"), 200, out / "train_wiki.jsonl", st),
                            take(local_docs(root / "ornek" / "eval_tr_wiki.jsonl"), 0, out / "train_wiki.jsonl", st, 1,
                                 out / "eval_wiki.jsonl", E)),
        "ozenli": lambda st: (take(jsonl_docs([BT + "ozenli-derlem/cfile_%d.jsonl" % i for i in range(1, 41)]), 300,
                                   out / "train_ozenli.jsonl", st),
                              take(jsonl_docs([BT + "ozenli-derlem/cfile_%d.jsonl" % i for i in range(200, 217)]), 0,
                                   out / "train_ozenli.jsonl", st, 1, out / "eval_ozenli.jsonl", E)),
        "akademik": lambda st: (take(jsonl_docs([BT + "akademik-derlem/dergi_17.jsonl"]), 200, out / "train_akademik.jsonl", st),
                                take(jsonl_docs([BT + "akademik-derlem/dergi_10.jsonl"]), 0, out / "train_akademik.jsonl", st,
                                     1, out / "eval_akademik.jsonl", E)),
        "cosmos": lambda st: (take(parquet_docs([CO % i for i in range(4)]), 300, out / "train_cosmos.jsonl", st),
                              take(parquet_docs([CO % 113]), 0, out / "train_cosmos.jsonl", st, 1, out / "eval_cosmos.jsonl", E)),
        "instructurca": lambda st: take(jsonl_docs([IT], lambda d: d["Input"] + "\n" + d["Output"]), 150,
                                        out / "train_instructurca.jsonl", st, 50, out / "eval_instructurca.jsonl", E),
    }
    for name, job in jobs.items():
        if a.only and name not in a.only:
            continue
        for f in out.glob("*_%s.jsonl" % name):
            f.unlink()
        st = []
        job(st)
        for f, tb, eb, n, drop in st:
            print("%-14s %-26s eğitim %6.0f MB  ölçüm %5.1f MB  %7d belge  (atılan %d)" % (name, f, tb, eb, n, drop), flush=True)


if __name__ == "__main__":
    main()
