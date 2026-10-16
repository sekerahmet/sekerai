"""Eğitim adımı profili: bölüm bölüm süre ve en çok CUDA süresi alan işlemler.

    python scripts/profile_step.py --config configs/fineweb/fw_s000_d512_tf_yg_ozel_rms.json
    python scripts/profile_step.py --config ... --set train.micro_batch=64 --steps 20 --no_table   # yalnız ms/adım

Eğitimin kendi adımını (rgn.train.step.train_step: kelimeye özel satır kopyası, ileri / geri, kırpma, optimizer) birkaç
kez çalıştırır; ısınmadan sonra senkronlu ölçüm ve torch.profiler tablosu.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rgn.config import load_config  # noqa: E402
from rgn.data.loader import load_streams  # noqa: E402
from rgn.model import RGNModel  # noqa: E402
from rgn.train.optim import build_optimizer  # noqa: E402
from rgn.train.step import train_step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--set", action="append", default=[], help="bölüm.alan=değer (load_config ezmesi)")
    ap.add_argument("--no_table", action="store_true", help="yalnız ms/adım (profiler tablosu yok)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    tc, dc = cfg.train, cfg.data
    dev = torch.device("cuda")
    train_s, _, _ = load_streams(dc)
    model = RGNModel(cfg.model).to(dev)
    opt = build_optimizer(model, tc)
    fwd = torch.compile(model)
    rng = np.random.default_rng(0)
    T, micro = dc.seq_len, tc.micro_batch or tc.batch_size
    accum = tc.batch_size // micro
    ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    def step():
        batches = [(*train_s.random_batch(micro, T, rng), None) for _ in range(accum)]
        train_step(model, fwd, opt, batches, ctx, tc.grad_clip, dev, torch.bfloat16)

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    print("adım %.1f ms  (%s; mikro %d × %d, tepe bellek %.1f GB)" % (
        1000 * (time.time() - t0) / a.steps, " ".join(a.set) or cfg.name, accum, micro,
        torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    if a.no_table:
        return
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))


if __name__ == "__main__":
    main()
