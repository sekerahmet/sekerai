"""Drive'daki büyük dosyalar: 100 MB'tan büyük dosyaları klasöre göre toplar (checkpoint temizliği için; hiçbir şey silmez).

    python scripts/drive_sizes.py [--root /content/drive/MyDrive] [--min_mb 100]
"""
import argparse
import os
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/drive/MyDrive")
    ap.add_argument("--min_mb", type=float, default=100)
    a = ap.parse_args()
    by_dir, files, total = defaultdict(float), [], 0.0
    for dp, _, fs in os.walk(a.root):
        for f in fs:
            try:
                st = os.stat(os.path.join(dp, f))
            except OSError:
                continue
            mb = st.st_size / 2 ** 20
            total += mb
            if mb >= a.min_mb:
                rel = os.path.relpath(dp, a.root)
                by_dir[rel] += mb
                files.append((mb, rel, f, time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))))
    print("Drive toplam %.1f GB; %d dosya ≥ %.0f MB, toplam %.1f GB" % (total / 1024, len(files), a.min_mb,
                                                                       sum(f[0] for f in files) / 1024))
    print("\nklasör başına (≥ %.0f MB dosyalar):" % a.min_mb)
    for d, mb in sorted(by_dir.items(), key=lambda kv: -kv[1])[:60]:
        print("  %8.2f GB  %s" % (mb / 1024, d))
    print("\nen büyük 40 dosya:")
    for mb, d, f, t in sorted(files, reverse=True)[:40]:
        print("  %8.2f GB  %s  %s/%s" % (mb / 1024, t, d, f))


if __name__ == "__main__":
    main()
