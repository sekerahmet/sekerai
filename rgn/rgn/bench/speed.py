"""Hız ve bellek profilleri.

    attention  yalnız dikkat çekirdeği, ileri + geri: softmax (SDPA / flash), rgn flex (fp32 / bf16), rgn dense
    model      tam eğitim adımı (ileri + geri + optimizer) — yapılandırma dosyalarındaki modeller

Not: rgn flex, FlexAttention'ın score_mod yoludur (içeride exp / log); RGN'nin donanım iddiası (V7) bu ölçümle değil,
özel çekirdekle sınanır.  Buradaki sayılar kalite deneylerinin maliyetini gösterir.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rgn.model.attention import rgn_dense, rgn_flex, rgn_triton
from rgn.model.position import geometric


def _time(fn, warmup=3, iters=10):
    """-> (ms / çağrı, tepe bellek GB).  CUDA'da senkronize ölçüm."""
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
    return ms, mem


def _attention_case(kind, B, H, Hkv, N, dh, pos, device):
    fp32 = kind in ("rgn_flex_fp32", "rgn_dense_fp32", "rgn_triton")
    dt = torch.float32 if fp32 else torch.bfloat16
    q = F.normalize(torch.randn(B, H, N, dh, device=device), dim=-1).to(dt).requires_grad_()
    k = F.normalize(torch.randn(B, Hkv, N, dh, device=device), dim=-1).to(dt).requires_grad_()
    v = torch.randn(B, Hkv, N, dh, device=device, dtype=dt, requires_grad=True)
    gamma = geometric(4.0, 512.0, H).to(device).requires_grad_()
    beta = geometric(1.0, 1 / 65536, H).to(device).requires_grad_()

    def run():
        if kind == "softmax_sdpa":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=Hkv != H)
        elif kind == "rgn_triton":
            o = rgn_triton(q, k, v, gamma, 1)
        elif kind.startswith("rgn_flex"):
            o = rgn_flex(q, k, v, gamma, beta, pos, 1)
        else:
            o = rgn_dense(q, k, v, gamma, beta, pos, 1)
        o.float().sum().backward()
        for t in (q, k, v, gamma, beta):
            t.grad = None
    return run


def bench_attention(seq_lens=(512, 1024, 2048, 4096), batch=16, heads=8, kv_heads=8, d_head=64, pos="rope",
                    kinds=("softmax_sdpa", "rgn_triton", "rgn_dense_fp32"), dense_max=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for N in seq_lens:
        for kind in kinds:
            if (kind.startswith("rgn_flex") or kind == "rgn_triton") and device.type != "cuda":
                continue
            if kind.startswith("rgn_dense") and N > dense_max:
                continue
            try:
                ms, mem = _time(_attention_case(kind, batch, heads, kv_heads, N, d_head, pos, device))
                rows.append(dict(kind=kind, N=N, B=batch, H=heads, Hkv=kv_heads, dh=d_head, ms=ms, mem_gb=mem,
                                 tok_s=batch * N / (ms / 1000)))
            except torch.OutOfMemoryError:
                rows.append(dict(kind=kind, N=N, B=batch, ms=None, mem_gb=None, error="OOM"))
                torch.cuda.empty_cache()
            print("  %-16s N=%5d  %s" % (kind, N, "%.2f ms  %.2f GB" % (rows[-1]["ms"], rows[-1]["mem_gb"])
                                         if rows[-1].get("ms") else rows[-1].get("error")), flush=True)
    return rows


def bench_model(configs, steps=10, batch=None, seq_len=None):
    """configs: RunConfig listesi -> tam eğitim adımı süresi (derlenmiş, bf16)."""
    from rgn.model import RGNModel
    from rgn.train.optim import build_optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for cfg in configs:
        B = batch or (cfg.train.micro_batch or cfg.train.batch_size)
        T = seq_len or cfg.data.seq_len
        model = RGNModel(cfg.model).to(device)
        model.loss_checkpoint = cfg.train.loss_checkpoint
        opt = build_optimizer(model, cfg.train)
        fwd = torch.compile(model) if (cfg.train.compile and device.type == "cuda") else model
        x = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu",
                                                                                                         enabled=False)

        def step():
            with ctx:
                loss = fwd(x, x)
                if isinstance(loss, tuple):                   # gözlemci: ana + gözlemci kaybı
                    loss = loss[0] + loss[1]
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        ms, mem = _time(step, warmup=3, iters=steps)
        rows.append(dict(name=cfg.name, params=model.num_params(), B=B, T=T, ms=ms, mem_gb=mem, tok_s=B * T / (ms / 1000)))
        print("  %-28s %.1fM  B=%d T=%d  %.1f ms  %.0fk tok/s  %.1f GB" % (
            cfg.name, model.num_params() / 1e6, B, T, ms, B * T / ms, mem), flush=True)
        del model, opt, fwd
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def save(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    path.write_text(json.dumps({"env": meta, "rows": rows}, indent=1), encoding="utf-8")
    return path
