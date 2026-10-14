"""Kodu Drive'a eşitle (Colab oradan çalıştırır): <Drive>/RGN/code/ altına rgn/, scripts/, configs/, tests/, colab/.

    python scripts/sync_drive.py            # değişen dosyaları kopyala
    python scripts/sync_drive.py --dry      # yalnız listele

runs/ ve cache/ kopyalanmaz.  Colab'da colab/setup_colab.py bu klasörü /content/RGN'ye alır.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgn.system import setup_process  # noqa: E402
from rgn.paths import ROOT, drive_root  # noqa: E402

FOLDERS = ("rgn", "scripts", "configs", "tests", "colab")
FILES = ("README.md", "requirements.txt")
SKIP = ("__pycache__", ".pyc", ".part")


def files():
    for name in FILES:
        if (ROOT / name).exists():
            yield ROOT / name
    for folder in FOLDERS:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and not any(s in str(p) for s in SKIP):
                yield p


def main():
    setup_process()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    dst_root = drive_root() / "RGN" / "code"
    n = 0
    for src in files():
        dst = dst_root / src.relative_to(ROOT)
        if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        n += 1
        print("  %s" % src.relative_to(ROOT))
        if not args.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print("[%s] %d dosya -> %s" % ("deneme" if args.dry else "eşitlendi", n, dst_root))


if __name__ == "__main__":
    main()
