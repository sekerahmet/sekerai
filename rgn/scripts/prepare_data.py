"""Veri seti hazırlama ve durum kontrolü.

    python scripts/prepare_data.py --list                      # kayıtlı veri setleri ve Drive'daki durumları
    python scripts/prepare_data.py --dataset tinystories       # ham metin -> GPT-2 token (.bin) + meta.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgn.system import setup_process  # noqa: E402
from rgn.data.prepare import PREPARERS  # noqa: E402
from rgn.data.registry import DATASETS  # noqa: E402
from rgn.paths import drive_root  # noqa: E402


def status():
    root = drive_root()
    print("Drive kökü: %s" % root)
    for name, spec in DATASETS.items():
        if spec.fmt == "synthetic":
            continue
        files = spec.train + spec.valid
        ok = sum((root / f).exists() for f in files)
        size = sum((root / f).stat().st_size for f in files if (root / f).exists()) / 2 ** 30
        state = "hazır" if ok == len(files) else ("hazırlanmalı: --dataset %s" % name if spec.prepare else "EKSİK")
        print("  %-12s %2d/%2d dosya  %6.2f GB  %-28s %s" % (name, ok, len(files), size, state, spec.description))


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="RGN veri hazırlama")
    ap.add_argument("--dataset", choices=sorted(PREPARERS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.dataset:
        PREPARERS[args.dataset]()
    status()


if __name__ == "__main__":
    main()
