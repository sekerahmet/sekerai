"""Koşuları karşılaştırma tablosu: runs/*/results.json -> Markdown tablo (ve isteğe bağlı CSV).

    python scripts/summarize_runs.py                 # varsayılan runs klasörü
    python scripts/summarize_runs.py --runs G:/Drive'ım/RGN/runs --csv ozet.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgn.system import setup_process  # noqa: E402
from rgn.paths import runs_dir  # noqa: E402

COLS = ("name", "attn", "pos", "mlp", "params_M", "steps", "loss", "bpb", "acc", "ms_step", "ktok_s", "mem_gb", "min")


def rows(root):
    for f in sorted(Path(root).glob("*/results.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        m, fin = r["config"]["model"], r["final"]
        yield dict(name=r["name"], attn=m["attn"], pos=m["pos"], mlp=m["mlp"], params_M=round(r["params"] / 1e6, 2),
                   steps=r["steps"], loss=round(fin["loss"], 4), bpb=fin["bpb"] and round(fin["bpb"], 4),
                   acc=round(fin["acc"], 4), ms_step=r["ms_per_step"] and round(r["ms_per_step"], 1),
                   ktok_s=r["tok_per_s"] and round(r["tok_per_s"] / 1e3), mem_gb=r["peak_mem_gb"] and
                   round(r["peak_mem_gb"], 1), min=round(r["wall_min"], 1))


def main():
    setup_process()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    data = sorted(rows(args.runs or runs_dir()), key=lambda r: r["loss"])
    print("| " + " | ".join(COLS) + " |")
    print("|" + "---|" * len(COLS))
    for r in data:
        print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in COLS) + " |")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
