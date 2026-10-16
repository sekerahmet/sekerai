"""Eğitim adımı profili: GPU süresinin hangi işlemlere gittiği (kullanıcı, 16 Ekim: kelimeye özel modelde +%25 adım süresi).

    python scripts/profile_step.py --config configs/fineweb/fw_s000_d512_tf_yg_ozel.json

Gerçek eğitim döngüsünün adımını (veri, ileri, geri, kırpma, optimizer) birkaç kez çalıştırır; ısınmadan sonra
torch.profiler ile en çok CUDA süresi alan işlemleri ve CPU'nun GPU'yu beklediği senkron noktalarını yazar.
Ayrıca bölüm bölüm duvar saati: veri + satır kopyası, ileri+geri, kırpma, optimizer.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(a.config)
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
    sec = {}

    def tick(name, t0):
        torch.cuda.synchronize()
        sec[name] = sec.get(name, 0.0) + time.time() - t0
        return time.time()

    def step(timed):
        t = time.time()
        mbs = [tuple(z.to(dev) for z in train_s.random_batch(micro, T, rng)) for _ in range(accum)]
        sub = priv = None
        if model.priv_map is not None:                       # trainer ile aynı yol
            uniq = torch.unique(torch.cat([model.priv_map[x].flatten() for x, _ in mbs]))
            sub = torch.cat([b.private.detach().index_select(0, uniq) for b in model.blocks], 1).requires_grad_()
        if timed:
            t = tick("veri + satır kopyası", t)
        for x, y in mbs:
            if sub is not None:
                priv = torch.nn.functional.embedding(torch.searchsorted(uniq, model.priv_map[x]), sub.to(torch.bfloat16))
            with ctx:
                loss = (fwd(x, y, None, priv) if priv is not None else fwd(x, y)) / accum
            loss.backward()
        if timed:
            t = tick("ileri + geri", t)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + ([sub] if sub is not None else []), tc.grad_clip)
        if timed:
            t = tick("kırpma", t)
        if opt.row_adam is not None:
            k = sub.shape[1] // len(model.blocks)
            opt.row_adam.set_rows(uniq, [sub.grad[:, i * k:(i + 1) * k] for i in range(len(model.blocks))])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if timed:
            tick("optimizer", t)

    for _ in range(4):
        step(False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        step(True)
    total = time.time() - t0
    print("adım %.1f ms (senkronlu ölçüm) — bölümler:" % (1000 * total / a.steps))
    for k, v in sec.items():
        print("   %-24s %7.1f ms" % (k, 1000 * v / a.steps))
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step(False)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))


if __name__ == "__main__":
    main()
