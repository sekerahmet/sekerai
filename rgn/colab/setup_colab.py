"""Colab kurulumu: Drive'ı bağla, kodu /content/RGN'ye al, bağımlılıkları kur, ortamı raporla.

Notebook'tan:
    !python /content/drive/MyDrive/RGN/code/colab/setup_colab.py            # kod Drive'dan (scripts/sync_drive.py)
    !python setup_colab.py --git https://github.com/<kullanıcı>/<depo>.git --branch main --subdir rgn

Drive bağlama notebook hücresinde yapılmalıdır (from google.colab import drive; drive.mount("/content/drive")).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive")
TARGET = Path("/content/RGN")


def sh(cmd):
    print("$ " + cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", help="kodu bu git deposundan al (yoksa Drive/RGN/code)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--subdir", default="", help="depo içinde RGN klasörü (örneğin rgn)")
    args = ap.parse_args()
    os.chdir("/content")                      # çalışma klasörü /content/RGN ise silinince git "getcwd" hatası verir
    assert DRIVE.is_dir(), "Drive bağlı değil: önce drive.mount('/content/drive')"
    if TARGET.exists():
        shutil.rmtree(TARGET)
    if args.git:
        tmp = Path("/content/_repo")
        if tmp.exists():
            shutil.rmtree(tmp)
        sh("git clone --depth 1 --branch %s %s %s" % (args.branch, args.git, tmp))
        shutil.copytree(tmp / args.subdir if args.subdir else tmp, TARGET)
    else:
        src = DRIVE / "RGN" / "code"
        assert src.is_dir(), "%s yok: yerelde python scripts/sync_drive.py çalıştırın" % src
        shutil.copytree(src, TARGET)
    sh("%s -m pip install -q tiktoken" % sys.executable)
    import torch
    print("torch %s | CUDA %s | GPU %s" % (torch.__version__, torch.cuda.is_available(),
                                          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
    print("kod: %s" % TARGET)


if __name__ == "__main__":
    main()
