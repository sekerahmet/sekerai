"""ngram_profile -- bigram embedding'in GPU adim maliyeti (deneme; belge 90b ek).  Ayni model (train.py kurulumu: d / L / G /
GQA varsayilanlari, bloklar derlenmis, bf16, summaries_last, ayni optimizer) ve ayni gercek batch'lerle kollar:
    base           ngram yok
    dense_all      --ngram_embed N (bugunku: her katmana, yogun AdamW)
    dense_k1       + --ngram_layers 1 (yalniz ilk blok girdisi)
    sparse_all     + --ngram_sparse 1 (seyrek satir Adam, beta1 0)
    sparse_k1      + ikisi
    frozen_all     dense_all, tablo donuk (requires_grad False): tablo gradyani + optimizer'i olmadan maliyet
Kol basina isinmadan sonra --steps adim, CUDA olaylariyla ms/adim (ortanca) ve tepe bellek; dense_all'da torch.profiler ile
ilk 25 CUDA kalemi (kaynak dokumu).  Egitim, kayit, sinav yok; agirlik yazilmaz.

    python ngram_profile.py --data /content/drive/MyDrive/v2/fineweb_edu_s000 [--rows 251520] [--steps 30] [--warmup 12]
                            [--arms base,dense_all,...] [--d 768 --layers 10 --heads 12]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
import data as D  # noqa: E402
import train as TR  # noqa: E402

ARMS = dict(base=[], dense_all=["E"], dense_k1=["E", "--ngram_layers", "1"], sparse_all=["E", "--ngram_sparse", "1"],
            sparse_k1=["E", "--ngram_layers", "1", "--ngram_sparse", "1"], frozen_all=["E"])


def _batches(data, layout, n):
    """Plan e1'in ilk n batch'i, train.py'nin cpu_batch'i gibi (summaries_last), cihazda."""
    from model import summaries_last
    st = D.TokenStories(data, data, "train")
    f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
    ro, rs, row_len = f["row_offsets"], f["row_stories"], int(f["row_len"])
    out = []
    for k in range(n):
        rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(k * TR.BATCH_ROWS, (k + 1) * TR.BATCH_ROWS)]
        out.append(TR._to_device(summaries_last(D.build_batch(st, rows, layout, "cpu", row_len))[0], torch.device("cuda")))
    return out


def arm(name, a, batches, steps, warmup, prof):
    extra = [x for x in ARMS[name] if x != "E"] + (["--ngram_embed", str(a.rows)] if "E" in ARMS[name] else [])
    args = TR._args(["--model", "model_z", "--d", str(a.d), "--layers", str(a.layers), "--heads", str(a.heads),
                     "--data", a.data, "--out", "x"] + extra)
    model, _, _ = TR._build(args, torch.device("cuda"))
    mask_fn = model._masks(True)
    if name == "frozen_all":
        model.ngram.weight.requires_grad_(False)
    for block in model.blocks:
        block.compile(dynamic=False, **TR._compile_kwargs())
    opt, _ = TR._optimizer(model, args.optimizer, args.lr, True)
    for g in opt.param_groups:
        g["lr"] = args.lr * g.get("lr_mult", 1.0)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        b = batches[i % len(batches)]
        if i >= warmup:
            ev[i - warmup][0].record()
        TR._step(model, b, mask_fn, opt, True)
        if i >= warmup:
            ev[i - warmup][1].record()
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in ev]
    peak = torch.cuda.max_memory_allocated() / 1e9
    table = None
    if prof:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for i in range(5):
                TR._step(model, batches[i % len(batches)], mask_fn, opt, True)
            torch.cuda.synchronize()
        table = p.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return float(np.median(ms)), float(np.min(ms)), peak, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=251520)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "GPU yok"
    print("GPU %s, bos bellek %.1f GB" % (torch.cuda.get_device_name(0), torch.cuda.mem_get_info()[0] / 1e9), flush=True)
    for k in ("cache_size_limit", "recompile_limit"):                    # train.py'nin CUDA ayarlari
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, 32)
    import torch._inductor.config as inductor_config
    if hasattr(inductor_config, "autotune_in_subproc"):
        inductor_config.autotune_in_subproc = True
    batches = _batches(a.data, "model_z", 8)
    res = {}
    for name in a.arms.split(","):
        med, best, peak, table = arm(name, a, batches, a.steps, a.warmup, name == "dense_all")
        res[name] = med
        print("%-11s ms/adim ortanca %.1f (en iyi %.1f) | tepe %.1f GB%s" % (
            name, med, best, peak, "" if "base" not in res else " | base'e gore %+.1f ms (%+.1f%%)" % (
                med - res["base"], 100 * (med - res["base"]) / res["base"])), flush=True)
        if table:
            print(table, flush=True)
    print("NGRAM_PROFILE_BITTI", flush=True)


if __name__ == "__main__":
    main()
