"""Koşu sonu hız profili (her eğitimin sonunda trainer çağırır) -> runs/<koşu>/speed.json.

    attention   koşunun dikkat çekirdeği ile softmax (SDPA) aynı boyutta, ileri + geri (ms, GB) — CUDA'da derlenmiş
                (eğitimdeki gibi); eager dense ölçümü oranı ~20 kat abartıyordu (ss_smoke, 15 Ekim)
    block       tek blokta dikkat ve MLP alt katmanlarının ileri + geri süresi (pay)
    inference   ileri geçiş (eğitim boyunda) token/s
    generate    önbelleksiz greedy üretim token/s (batch 1, 64 token)
"""
import json
import time

import torch
import torch.nn.functional as F

from rgn.bench.speed import _time
from rgn.model.attention import rgn_dense, rgn_flex


def _attn_kernel(attn, B, N, device):
    """Koşunun dikkat modülüyle aynı tür / boyutta çekirdek süresi ve aynı boyutta softmax."""
    cfg = attn.cfg
    H, Hkv, dh = attn.H, attn.Hkv, attn.dh
    out = {}
    dense = torch.compile(rgn_dense, dynamic=False) if device.type == "cuda" else rgn_dense
    for kind in ("model", "softmax_sdpa"):
        if kind == "softmax_sdpa" or cfg.attn == "softmax":
            dt = torch.bfloat16 if device.type == "cuda" else torch.float32
        else:
            dt = torch.float32 if cfg.rgn_fp32 else torch.bfloat16
        q = F.normalize(torch.randn(B, H, N, dh, device=device), dim=-1).to(dt).requires_grad_()
        k = F.normalize(torch.randn(B, Hkv, N, dh, device=device), dim=-1).to(dt).requires_grad_()
        v = torch.randn(B, Hkv, N, dh, device=device, dtype=dt, requires_grad=True)

        def run(kind=kind, q=q, k=k, v=v, dt=dt):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=dt == torch.bfloat16):
                if kind == "softmax_sdpa" or cfg.attn == "softmax":
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=Hkv != H)
                else:
                    fn = rgn_flex if attn.impl(device, N) == "flex" else dense
                    o = fn(q, k, v, attn.gamma().detach(), None if attn.beta() is None else attn.beta().detach(),
                           cfg.pos, cfg.power, cfg.sharpness)
            o.float().sum().backward()
            q.grad = k.grad = v.grad = None
        ms, mem = _time(run, warmup=2, iters=5)
        out[kind] = {"ms": ms, "mem_gb": mem, "dtype": str(dt).replace("torch.", "")}
    out["ratio_model_vs_softmax"] = out["model"]["ms"] / out["softmax_sdpa"]["ms"]
    return out


def _block_split(model, B, N, device, ctx):
    blk = model.blocks[0]
    x = torch.randn(B, N, model.cfg.d_model, device=device, requires_grad=True)
    rope = model.rope(N)
    res = {}
    wrap = (lambda f: torch.compile(f, dynamic=False)) if device.type == "cuda" else (lambda f: f)
    for name, fn in (("attention", wrap(lambda: blk.attn(blk.norm1(x), rope))),
                     ("mlp", wrap(lambda: blk.mlp(blk.norm2(x))))):
        def run(fn=fn):
            with ctx:
                y = fn()
            y.float().sum().backward()
            x.grad = None
        res[name] = {"ms": _time(run, warmup=2, iters=5)[0]}
    tot = res["attention"]["ms"] + res["mlp"]["ms"]
    for v in res.values():
        v["share"] = v["ms"] / tot
    return res


@torch.no_grad()
def _inference(model, B, N, device, ctx):
    x = torch.randint(0, model.cfg.vocab_size, (B, N), device=device)

    def run():
        with ctx:
            model.hidden(x)
    ms = _time(run, warmup=2, iters=5)[0]
    return {"ms": ms, "tok_s": B * N / (ms / 1000)}


@torch.no_grad()
def _generate(model, device):
    idx = torch.randint(0, model.cfg.vocab_size, (1, 16), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(idx, 64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"tok_s": 64 / (time.perf_counter() - t0), "note": "önbelleksiz başvuru üretimi"}


def profile_after_run(model, cfg, device, ctx, out_path):
    """Hata olursa koşuyu bozmaz; bulduğunu yazar."""
    B = min(cfg.train.micro_batch or cfg.train.batch_size, 16)
    N = cfg.data.seq_len
    model.eval()
    res = {"B": B, "N": N, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    for key, fn in (("attention", lambda: _attn_kernel(model.blocks[0].attn, B, N, device)),
                    ("block", lambda: _block_split(model, B, N, device, ctx)),
                    ("inference", lambda: _inference(model, B, N, device, ctx)),
                    ("generate", lambda: _generate(model, device))):
        try:
            res[key] = fn()
        except Exception as e:                       # profil koşuyu düşürmesin
            res[key] = {"error": "%s: %s" % (type(e).__name__, e)}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.train()
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    a, b = res.get("attention", {}), res.get("block", {})
    if "model" in a:
        print("  [hız] dikkat %.2f ms (softmax %.2f ms, oran %.2f) | blok payı: dikkat %.0f%% / MLP %.0f%% | çıkarım %s tok/s"
              % (a["model"]["ms"], a["softmax_sdpa"]["ms"], a["ratio_model_vs_softmax"],
                 100 * b.get("attention", {}).get("share", 0), 100 * b.get("mlp", {}).get("share", 0),
                 "%.0fk" % (res["inference"]["tok_s"] / 1e3) if "tok_s" in res.get("inference", {}) else "-"), flush=True)
    return res
