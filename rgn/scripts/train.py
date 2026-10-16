"""Eğitim.

    python scripts/train.py --config configs/ss/ss_d512_tf_yg_ozel.json
    python scripts/train.py --config configs/ss/ss_d512_tf_yg.json --set train.max_steps=200 --set data.eval_tokens=262144
    python scripts/train.py --config configs/ss/ss_d512_tf_yg_ozel.json --resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgn.system import setup_process  # noqa: E402
from rgn.config import load_config  # noqa: E402
from rgn.train import train  # noqa: E402


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="eğitim")
    ap.add_argument("--config", required=True, help="deney dosyası (configs/**.json)")
    ap.add_argument("--set", action="append", default=[], metavar="BÖLÜM.ALAN=DEĞER", help="alan ez (tekrarlanabilir)")
    ap.add_argument("--name", help="koşu adı (varsayılan: yapılandırmadaki ad)")
    ap.add_argument("--out", help="çıktı klasörü (varsayılan: runs/<ad>)")
    ap.add_argument("--resume", action="store_true", help="checkpoint.pt'den sürdür")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg.name = args.name
    train(cfg, out_dir=args.out, resume=args.resume)


if __name__ == "__main__":
    main()
