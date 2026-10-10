"""bench_step -- Model S govde hizi (GPU): ileri + geri, derlenmis bloklar, bf16 autocast; kollar op x PROJ_FOLD (Parca 7).
Egitim degil: rastgele token, nedensel maske (flex BlockMask yerine SDPA is_causal), kayip h^2.  Hiz iddiasi once GPU olcumu.

    python model_s/bench_step.py --d 1024 --heads 16 --widths 1x12 --rows 16 --arms split:0,split:1,mix:0,mix:1
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "common"))
import operators as O  # noqa: E402
from model import ModelS  # noqa: E402


def widths(spec):
    """'1x12' ya da '1,1,2,2' -> liste."""
    if "x" in spec and "," not in spec:
        w, n = spec.split("x")
        return [int(w)] * int(n)
    return [int(t) for t in spec.split(",")]


def run(op, fold, a):
    O.PROJ_FOLD = bool(fold)
    torch.manual_seed(0)
    m = ModelS(a.d, widths(a.widths), a.heads, op=op).cuda()
    for b in m.blocks:
        b.compile(mode=a.mode)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    tok = torch.randint(0, 50000, (a.rows, a.T), device="cuda")
    pos = torch.arange(a.T, device="cuda").expand(a.rows, a.T)

    def step():
        with torch.autocast("cuda", torch.bfloat16):
            x = m.E(tok)
            for b in m.blocks:
                x = b(x, pos, None)
            loss = m.norm(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.iters):
        step()
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - t) / a.iters
    n = sum(p.numel() for p in m.blocks.parameters())
    print("%-6s fold %d | %8.2f ms / adim (govde ileri + geri) | blok parametresi %d | tepe %.1f GB"
          % (op, fold, ms, n, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    del m, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return ms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--widths", default="1x12")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--arms", default="split:0,split:1,mix:0,mix:1")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--mode", default="max-autotune-no-cudagraphs")
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "GPU yok"
    print("bench_step d %d heads %d widths %s satir %d x %d | %s" % (a.d, a.heads, a.widths, a.rows, a.T, torch.cuda.get_device_name(0)))
    res = {arm: run(arm.split(":")[0], int(arm.split(":")[1]), a) for arm in a.arms.split(",")}
    base = res.get("split:0")
    if base:
        print("split:0'a gore: " + ", ".join("%s %+.1f%%" % (k, 100 * (v / base - 1)) for k, v in res.items()))


if __name__ == "__main__":
    main()
