"""Hız ve bellek profili.

    python scripts/bench_speed.py attention                          # dikkat çekirdekleri, N = 512..4096
    python scripts/bench_speed.py attention --seq 1024 2048 --batch 32
    python scripts/bench_speed.py model --configs configs/ss/ss_d512_rgn.json configs/ss/ss_d512_transformer.json

Sonuçlar runs/bench/<tür>_<zaman>.json dosyasına yazılır.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgn.system import setup_process  # noqa: E402
from rgn.bench import bench_attention, bench_model, save  # noqa: E402
from rgn.config import load_config  # noqa: E402
from rgn.paths import runs_dir  # noqa: E402


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="RGN hız profili")
    ap.add_argument("what", choices=("attention", "model"))
    ap.add_argument("--seq", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=8)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--pos", default="additive")
    ap.add_argument("--configs", nargs="+", default=[])
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    if args.what == "attention":
        rows = bench_attention(args.seq, args.batch or 16, args.heads, args.kv_heads, args.d_head, args.pos)
    else:
        assert args.configs, "--configs gerekli"
        rows = bench_model([load_config(c, args.set) for c in args.configs], batch=args.batch,
                           seq_len=args.seq[0] if len(args.seq) == 1 else None)
    path = save(rows, runs_dir() / "bench" / ("%s_%s.json" % (args.what, time.strftime("%Y%m%d_%H%M%S"))))
    print("[kaydedildi] %s" % path)


if __name__ == "__main__":
    main()
