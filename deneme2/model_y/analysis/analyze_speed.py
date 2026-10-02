# -*- coding: utf-8 -*-
"""analyze_speed -- Model Y performans analizi (ajan C, 2 Ekim 2026): hiz, verim, maliyet; DOGRULUK DEGIL.

Kullanici: "Performans da bir kriter olsun ama ayrica performans icin de ajan tayin et olur mu ?"

    train     egitim adimi (kosunun config.json'u, gercek paketli FineWeb pencereleri): train_seq'in adimi parca parca
              (ileri / geri / grup kopyasi / coherence / clip / Muon + Newton-Schulz / Adam / normalize_weights / EMA),
              train_seq'in kendisiyle yan yana; parca bilesenleri (govde, kayip basligi, attention, flex, FactUnits);
              kernel kategorileri (torch.profiler); MFU; hizlandirma denemeleri (esdegerlik kontroluyle).
    sizes     buyuk model adaylari: train_seq'in kendisi, birkac adim -> ms/adim, token/sn, bellek, MFU.
    generate  uretim: onbellekli (AttentionCache) batch 1 / buyuk batch, baglam boyu, prefill; sabit onbellek + CUDA graph
              (esdegerlik kontroluyle); bf16; son turlarin attention'i sabit (DAVRANIS DEGISIR).

    python analysis/analyze_speed.py <kosu klasoru> train --data <FineWeb koku>
    python analysis/analyze_speed.py <kosu klasoru> --selftest          (CPU, kucuk model: kod yollari)
Cikti: $KUYRUK_SONUC (yoksa <kosu>/analysis) altina speed_<olcum>.json; satirlar stdout'a.
"""
import argparse
import contextlib
import copy
import json
import os
import tempfile
import sys
import time
import traceback
import types

import torch  # pyarrow / tokenizers'tan once (Windows c10.dll)
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, "train_fineweb"), os.path.join(SRC, "train_simplestories")]
import internals_y as I  # noqa: E402
import model_y as M  # noqa: E402
import train_y as TR  # noqa: E402

# G4 = RTX PRO 6000 Blackwell.  NVIDIA RTX Blackwell PRO whitepaper v1.0, Ek A: Workstation Edition "Peak BF16 Tensor
# TFLOPS with FP32 Accumulate 503.8/1007.6" (ikinci sayi seyreklikle).  Server Edition sayfasi "BF16 Tensor Core 1 PFLOP"
# (seyrek) -> yogun ~500.  MFU bu tepeyle; ayrica olculen GEMM tavaniyla.
PEAK_BF16 = {"RTX PRO 6000": 503.8e12, "A100": 312e12, "H100": 989e12, "L4": 121e12}
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def peak_for(name):
    return next((v for k, v in PEAK_BF16.items() if k in name), None)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, reps=5, warmup=2):
    """fn'in medyan suresi (ms); GPU'da CUDA event'leriyle."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(reps):
        if torch.cuda.is_available():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        else:
            t = time.perf_counter()
            fn()
            times.append(1e3 * (time.perf_counter() - t))
    times.sort()
    return times[len(times) // 2]


def out_dir(run):
    d = os.environ.get("KUYRUK_SONUC") or os.path.join(run, "analysis") if os.path.isdir(run) else tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    return d


def write(run, name, payload):
    path = os.path.join(out_dir(run), "speed_%s.json" % name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload, lines=LINES), f, indent=1, ensure_ascii=False)
    say("yazildi: %s" % path)


# ---- veri: shard 13'un butun belgeleri, egitimdeki best-fit paketleme (gercek belge boyu dagilimi)

def packed_batches(root, rows, count, seq_len=8192):
    """count adet (ids, mask, document_positions) batch, rows x (seq_len + 1); data_fineweb.pack / windows ile."""
    import data_fineweb as DF
    v = DF.load_valid(root, log=lambda s: None)
    stream, offsets, _ = v["valid_shard"]
    data = dict(seq_len=seq_len, train=stream, train_starts=offsets, eot=v["eot"], fingerprint="speed_shard13")
    data["items"] = DF._items(data)
    plan = DF.pack(data, 0, 0)
    out = [DF.windows(data, plan, plan["order"][i * rows:(i + 1) * rows]) for i in range(count)]
    return out, v


def fake_batches(rows, count, seq_len, vocab, doc=300):
    """CPU denemesi: rastgele token, sabit boy belgeler."""
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(count):
        ids = torch.randint(0, vocab, (rows, seq_len + 1), generator=g)
        pos = torch.arange(seq_len + 1).remainder(doc).expand(rows, -1).contiguous()
        out.append((ids, pos > 0, pos))
    return out


def keys_per_target(batch):
    _, mask, pos = batch
    valid = mask[:, 1:]
    return float(((pos[:, :-1] + 1) * valid).sum()) / float(valid.sum()), int(valid.sum())


def model_flops(model):
    """colab_simplestories._model_flops: (N matris parametresi / token, L x d)."""
    import colab_simplestories as CS
    return CS._model_flops(model)


def train_flops(flops, targets, keys_avg):
    """6 N hedef + 12 L d anahtar (PaLM ek B; colab_simplestories._mfu ile ayni)."""
    return 6 * flops[0] * targets + 12 * flops[1] * targets * keys_avg


# ---- egitim adimi: train_seq'in kurulumu ve adimi parca parca

def setup(config, device, vocab):
    """train_seq'in kendi kurulumu (internals_y._profile_setup), kucuk bir batch'le 1 adim: model, Muon / Adam, EMA."""
    cfg = dict(config, micro_batches=1)
    g = torch.Generator().manual_seed(1)
    tiny = [(torch.randint(0, vocab, (2, 65), generator=g), torch.ones(2, 65, dtype=torch.bool))]
    ctx = I._profile_setup(cfg, tiny, device)
    ctx["loss_fn"] = torch.compile(ctx["model"].loss, dynamic=False) if ctx["compile"] else ctx["model"].loss
    return ctx


def fwd_context(ctx):
    stack = contextlib.ExitStack()
    stack.enter_context(sdpa_kernel(ctx["kernels"]))
    if ctx["bf16"]:
        stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
    return stack


class Clock:
    """Parca sureleri: CUDA event cifti (GPU), host saati (CPU).  Ayni ad toplanir."""

    def __init__(self, cuda):
        self.cuda, self.rows = cuda, []

    @contextlib.contextmanager
    def part(self, name):
        if self.cuda:
            a = torch.cuda.Event(enable_timing=True)
            a.record()
        t = time.perf_counter()
        yield
        if self.cuda:
            b = torch.cuda.Event(enable_timing=True)
            b.record()
            self.rows.append((name, a, b))
        else:
            self.rows.append((name, 1e3 * (time.perf_counter() - t), None))

    def read(self):
        sync()
        out = {}
        for name, a, b in self.rows:
            out[name] = out.get(name, 0.0) + (a.elapsed_time(b) if b is not None else a)
        self.rows = []
        return out


def accumulate(ctx, batch, micro, clock):
    """train_seq'in micro_batches > 1 yolu (accumulate + coherence birlestirmesi), ayni sirayla: grup basina parca parca
    ileri + geri, grup gradyaninin kopyasi; sonra iki yarinin gradyanlari p.grad'a ve birlesik gradyan.  -> p.grad dolu."""
    opt, params, named = ctx["opt"], ctx["params"], ctx["named"]
    ids, mask, pos = (t.to(ctx["device"]) for t in batch)
    k, split = micro, ctx["split"]
    groups = [[h + 2 * r for r in range(k // 2)] for h in (0, 1)] if split and k > 1 else [list(range(k))]
    saved, counts = [], []
    for group in groups:
        count = sum(mask[s::k, 1:].sum() for s in group)
        opt.zero_grad()
        for s in group:
            with clock.part("ileri (govde + kayip basligi)"), fwd_context(ctx):
                part_total, _ = ctx["loss_fn"](ids[s::k].contiguous(), mask[s::k].contiguous(), pos[s::k].contiguous())
            with clock.part("geri yayilim"), sdpa_kernel(ctx["kernels"]):
                (part_total * (mask[s::k, 1:].sum() / count)).backward()
        with clock.part("grup gradyani kopyasi"):
            saved.append([None if p.grad is None else p.grad.detach().clone() for p in params])
        counts.append(count)
    opt.zero_grad()
    with clock.part("coherence (yukleme, birlestirme, carpimlar)"):
        if len(saved) == 1:
            for p, g in zip(params, saved[0]):
                p.grad = g
            return
        grads = None
        for gs in saved:                                 # _Gradients.backward
            for p, g in zip(params, gs):
                if g is not None:
                    p.grad = g.clone() if p.grad is None else p.grad.add_(g)
            if grads is None:
                grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
        sums = []
        w0, w1 = (float(w) for w in counts)
        for (key, p), a in zip(named, grads):
            if a is None:
                p.grad = None
                continue
            b = p.grad - a
            p.grad = (w0 * a + w1 * b) / (w0 + w1)
            if key in ctx["unit_axis"]:
                a, b = (v - (v * p.detach()).sum(ctx["unit_axis"][key], keepdim=True) * p.detach() for v in (a, b))
            sums.append(torch.stack([(a * b).sum(), (a * a).sum(), (b * b).sum()]))
        torch.stack(sums).tolist()


def ema_loop(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


def ema_foreach(ema, model, decay):
    """ONERI: ayni iki islem, parametre listesi tek cagrida (torch._foreach_*)."""
    pe, p = list(ema.parameters()), [q.detach() for q in model.parameters()]
    torch._foreach_mul_(pe, decay)
    torch._foreach_add_(pe, p, alpha=1 - decay)


def optimizer_tail(ctx, clock, ema_fn=ema_loop):
    """train_seq'in adim sonu: clip, Muon (Newton-Schulz ayrica) / Adam, normalize_weights, link_u, EMA."""
    opt, model = ctx["opt"], ctx["model"]
    with torch.no_grad():
        if ctx["grad_clip"] is not None:
            with clock.part("clip"):
                torch.nn.utils.clip_grad_norm_(ctx["params"], ctx["grad_clip"])
        groups = opt.param_groups

        def orthogonalize(G, steps=5, precision="fp32"):
            with clock.part("  Newton-Schulz (Muon icinde)"):
                return TR.Muon.orthogonalize(G, steps, precision)
        opt.orthogonalize = orthogonalize
        try:
            for name, muon in (("Muon (Newton-Schulz dahil)", True), ("Adam", False)):
                opt.param_groups = [g for g in groups if g["use_muon"] == muon]
                with clock.part(name):
                    opt.step()
        finally:
            opt.param_groups = groups
            del opt.orthogonalize
        if getattr(model, "sphere_weights", False):
            with clock.part("normalize_weights"):
                model.normalize_weights()
        if getattr(model, "output_link", False):
            model.link_u.clamp_(min=0)
        if ctx["ema"] is not None:
            with clock.part("EMA (+ normalize_weights)"):
                ema_fn(ctx["ema"]["model"], model, ctx["ema"]["decay"])
                if getattr(model, "sphere_weights", False):
                    ctx["ema"]["model"].normalize_weights()


def replica_step(ctx, batch, micro, ema_fn=ema_loop):
    clock = Clock(ctx["cuda"])
    with clock.part("ADIM"):
        accumulate(ctx, batch, micro, clock)
        optimizer_tail(ctx, clock, ema_fn)
    return clock.read()


def real_step_ms(config, batch, device, K=2, micro=None, rows=None, model_kw=None):
    """train_seq'in kendisi, ayni batch 2K adim; ikinci K adimin ms/adim'i, tepe bellek, govde, FLOP."""
    kw = I._train_kwargs(config)
    if micro is not None:
        kw["micro_batches"] = micro
    if model_kw:
        kw["model_kw"] = dict(kw["model_kw"], **model_kw)
    if rows is not None:
        batch = tuple(t[:rows * kw["micro_batches"]] for t in batch)
    marks, info = [], {}

    def cb(step, model, nll):
        marks.append(time.perf_counter())
        if "flops" not in info:
            info.update(flops=model_flops(model), params=sum(p.numel() for p in model.parameters()),
                        body=sum(p.numel() for k, p in model.named_parameters() if not k.startswith(("tokens.", "input_"))))
        info["nll"] = nll
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    TR.train_seq(config.get("setting", "shared"), None, None, config["vocab"], steps=2 * K, device=device,
                 compile=config.get("compile", True), log_at=(), every=K, callback=cb, batches=lambda s: batch, **kw)
    info.update(step_ms=1e3 * (marks[2] - marks[1]) / K, micro=kw["micro_batches"],
                rows=batch[0].shape[0] // kw["micro_batches"],
                peak_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None)
    return info


# ---- hizli tavanlar

def ceilings(device):
    if torch.device(device).type != "cuda":
        return {}
    res = {}
    for n in (8192, 4096):
        a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        ms = bench(lambda: a @ b, reps=10)
        res["gemm_bf16_%d" % n] = 2 * n ** 3 / ms / 1e9
    for (m, k, n) in ((32768, 1024, 2752), (32768, 2752, 1024), (32768, 1024, 1024)):
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        res["gemm_bf16_%dx%dx%d" % (m, k, n)] = 2 * m * k * n / bench(lambda: a @ b, reps=10) / 1e9
    x = torch.empty(2 ** 28, device=device, dtype=torch.float32)          # 1 GB
    y = torch.empty_like(x)
    res["copy_GBps"] = 2 * x.numel() * 4 / bench(lambda: y.copy_(x), reps=10) / 1e6
    del a, b, x, y
    torch.cuda.empty_cache()
    say("TAVAN (olculen): bf16 GEMM 8192^3 %.0f TFLOP/s, 4096^3 %.0f | model sekilleri %s | kopya %.0f GB/s" % (
        res["gemm_bf16_8192"], res["gemm_bf16_4096"],
        " ".join("%s %.0f" % (k[10:], v) for k, v in res.items() if k.count("x") == 2), res["copy_GBps"]))
    return res


# ---- bilesenler: bir parca (micro-batch) uzerinde

def kernel_categories(fn, cuda, top=30):
    """Bir cagrinin GPU kernel sureleri kategori kategori (ad kalibindan) + en pahali kernel'ler."""
    if not cuda:
        return None
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        fn()
        sync()
    cats, rows = {}, []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        name = e.key.lower()
        if not t or name.startswith(("aten::", "torch", "autograd", "compiled", "##", "inductor", "triton_kernel_wrapper")):
            continue                                        # CPU tarafi olay: kernel'leri ayrica sayiliyor
        if any(s in name for s in ("gemm", "nvjet", "cutlass", "xmma", "cublas", "sm90_", "sm100_", "sm120_", "ampere_")):
            cat = "GEMM (cuBLAS/CUTLASS)"
        elif "flex" in name or "triton_tem" in name:
            cat = "flex attention (triton template)"
        elif name.startswith("triton_red") or name.startswith("triton_per"):
            cat = "triton fused reduction (norm, softmax, toplam)"
        elif name.startswith("triton_poi"):
            cat = "triton fused elementwise"
        elif "memcpy" in name or "memset" in name:
            cat = "memcpy / memset"
        elif name.startswith("cuda"):                       # cudaLaunchKernel vb. (CPU API)
            continue
        else:
            cat = "diger kernel"
        cats[cat] = cats.get(cat, 0.0) + t / 1e3
        rows.append((t / 1e3, e.count, e.key[:110]))
    rows.sort(reverse=True)
    return dict(categories=cats, top=rows[:top])


def components(ctx, batch, flops, peak):
    """Bir parca (micro-batch, rows x T): tam kayip, govde, kayip basligi, attention modulu, flex, FactUnits."""
    model, cuda = ctx["model"], ctx["cuda"]
    dev = ctx["device"]
    ids, mask, pos = (t.to(dev) for t in batch)
    res = {}
    loss = ctx["loss_fn"]

    def full_fwd():
        with fwd_context(ctx):
            loss(ids, mask, pos)

    def full_fb():
        with fwd_context(ctx):
            total, _ = loss(ids, mask, pos)
        total.backward()
        model.zero_grad(set_to_none=True)
    res["loss_fwd_ms"], res["loss_fwd_bwd_ms"] = bench(full_fwd), bench(full_fb)
    targets = int(mask[:, 1:].sum())
    keys = float(((pos[:, :-1] + 1) * mask[:, 1:]).sum()) / targets
    res["mfu_micro"] = train_flops(flops, targets, keys) / (res["loss_fwd_bwd_ms"] / 1e3) / peak if peak else None
    inner = pos[:, :-1].contiguous()
    x_ids = ids[:, :-1].contiguous()
    body = torch.compile(lambda i, p: model.hidden(i, None, p)[-1], dynamic=False) if ctx["compile"] else \
        (lambda i, p: model.hidden(i, None, p)[-1])
    g = torch.randn(x_ids.shape + (model.tokens.fixed_points.shape[1],), device=dev)

    def body_fwd():
        with fwd_context(ctx):
            body(x_ids, inner)

    def body_fb():
        with fwd_context(ctx):
            h = body(x_ids, inner)
        (h.float() * g).sum().backward()
        model.zero_grad(set_to_none=True)
    res["body_fwd_ms"], res["body_fwd_bwd_ms"] = bench(body_fwd), bench(body_fb)
    res["head_fwd_ms"] = res["loss_fwd_ms"] - res["body_fwd_ms"]
    res["head_fwd_bwd_ms"] = res["loss_fwd_bwd_ms"] - res["body_fwd_bwd_ms"]
    # attention modulu (q/k/v, norm, RoPE, log-n, flex, W_value) ve FactUnits, tek tur; maske bir kez
    B, T = x_ids.shape
    d = model.tokens.fixed_points.shape[1]
    blk = model.blocks[1]
    mask_obj = model.document_mask(inner)
    res["block_mask_ms"] = bench(lambda: model.document_mask(inner))
    xa = F.normalize(torch.randn(B, T, d, device=dev), dim=-1).requires_grad_()
    att = torch.compile(lambda x: blk.attention(x, document_positions=inner, document_mask=mask_obj), dynamic=False) \
        if ctx["compile"] else (lambda x: blk.attention(x, document_positions=inner, document_mask=mask_obj))
    units = blk.facts
    fac = torch.compile(lambda x: units(x * d ** 0.5), dynamic=False) if ctx["compile"] else (lambda x: units(x * d ** 0.5))
    for name, fn in (("attention_module", att), ("facts", fac)):
        def fwd(fn=fn):
            with fwd_context(ctx):
                fn(xa)

        def fb(fn=fn):
            with fwd_context(ctx):
                y = fn(xa)
            y.float().sum().backward()
            xa.grad = None
            model.zero_grad(set_to_none=True)
        res[name + "_fwd_ms"], res[name + "_fwd_bwd_ms"] = bench(fwd), bench(fb)
    res["kernels_fwd_bwd"] = kernel_categories(full_fb, cuda)
    return res, mask_obj, inner


def flex_variants(model, inner, mask_obj, device, cuda):
    """flex_attention tek basina (bf16, tur basina bir cagri) ve kernel_options / autotune denemeleri; cikti farki
    varsayilana gore (ayni maske, ayni q k v)."""
    if not cuda:
        return None
    from torch.nn.attention.flex_attention import flex_attention
    at = model.blocks[0].attention
    B, T = inner.shape
    H, dh = at.heads, model.tokens.fixed_points.shape[1] // at.heads
    g = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(B, H, T, dh, device=device, generator=g, dtype=torch.bfloat16) for _ in range(3))
    q, k = F.normalize(q.float(), dim=-1).bfloat16(), F.normalize(k.float(), dim=-1).bfloat16()
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    go = torch.randn(B, H, T, dh, device=device, dtype=torch.bfloat16)
    out, ref = {}, None
    variants = [("varsayilan (egitimdeki)", M._flex_compiled, None),
                ("fwd BLOCK 64x64", M._flex_compiled, dict(BLOCK_M=64, BLOCK_N=64)),
                ("fwd BLOCK 128x64", M._flex_compiled, dict(BLOCK_M=128, BLOCK_N=64)),
                ("fwd BLOCK 64x128", M._flex_compiled, dict(BLOCK_M=64, BLOCK_N=128)),
                ("fwd BLOCK 128x128", M._flex_compiled, dict(BLOCK_M=128, BLOCK_N=128)),
                ("bwd 64/128/128/64", M._flex_compiled, dict(BLOCK_M1=64, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=64)),
                ("bwd 32/64/64/32", M._flex_compiled, dict(BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32)),
                ("max-autotune", None, None)]
    for name, fn, opts in variants:
        try:
            if fn is None:
                fn = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

            def fwd():
                return fn(q, k, v, block_mask=mask_obj, scale=at.scale, kernel_options=opts)

            def fb():
                fwd().backward(go)
                for t in (q, k, v):
                    t.grad = None
            o = fwd().detach().float()
            fwd().backward(go)
            gq = q.grad.detach().float().clone()
            for t in (q, k, v):
                t.grad = None
            if ref is None:
                ref = (o, gq)
            out[name] = dict(fwd_ms=bench(fwd), fwd_bwd_ms=bench(fb), out_maxdiff=float((o - ref[0]).abs().max()),
                             grad_q_maxdiff=float((gq - ref[1]).abs().max()))
        except Exception as e:                           # paylasimli bellek / desteklenmeyen blok
            out[name] = dict(error=repr(e)[:200])
    return out


@contextlib.contextmanager
def guarded(name, res):
    """Bolum hata verirse kaydedilir, sonraki bolumler kosar."""
    try:
        yield
    except Exception:
        tb = traceback.format_exc()
        say("HATA (%s): %s" % (name, tb[-1500:]))
        res.setdefault("errors", {})[name] = tb[-4000:]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextlib.contextmanager
def norms_off():
    """DAVRANIS DEGISIR (yalniz sure): model_y icindeki F.normalize kimlik -- kure normlarinin (akis, q/k, girdi, PL) bedeli."""
    original = M.F
    ns = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    ns.normalize = lambda x, dim=-1, eps=1e-12, **kw: x
    M.F = ns
    torch._dynamo.reset()
    try:
        yield
    finally:
        M.F = original
        torch._dynamo.reset()


def train_measure(args, config, device):
    cuda = torch.device(device).type == "cuda"
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    peak = peak_for(gpu)
    say("analyze_speed train | %s | torch %s | %s | kosu %s" % (time.strftime("%Y-%m-%d %H:%M"), torch.__version__, gpu,
                                                               config.get("name")))
    res = dict(gpu=gpu, torch=torch.__version__, peak_bf16=peak, ceilings=ceilings(device))
    rows, micro, T = config["batch_size"], config["micro_batches"], config["seq_len"]
    if args.data:
        steps_data, _ = packed_batches(args.data, rows * micro, 2, T)
    else:
        steps_data = fake_batches(rows * micro, 2, T, config["vocab"])
    kp = [keys_per_target(b) for b in steps_data]
    res["keys_per_target"] = [k for k, _ in kp]
    res["targets_per_step"] = [n for _, n in kp]
    say("veri: %d adimlik batch, %d satir x %d (%d parca x %d) | hedef/adim %s | ortalama anahtar/hedef %s" % (
        len(steps_data), rows * micro, T, micro, rows, [n for _, n in kp], ["%.0f" % k for k, _ in kp]))
    ctx = setup(config, device, config["vocab"])
    model = ctx["model"]
    flops = model_flops(model)
    res["flops"] = dict(matrix_per_token=flops[0], attention_Ld=flops[1])
    # 1) adim parca parca (ilk adim isinma / compile)
    t0 = time.time()
    replica_step(ctx, steps_data[0], micro)
    say("isinma (compile dahil) %.0f sn" % (time.time() - t0))
    parts = [replica_step(ctx, steps_data[i % 2], micro) for i in range(args.steps)]
    names = list(parts[0])
    med = {n: sorted(p[n] for p in parts)[len(parts) // 2] for n in names}
    res["step_parts_ms"] = med
    total = med["ADIM"]
    targets, keys = kp[0][1], kp[0][0]
    res["replica_mfu"] = train_flops(flops, targets, keys) / (total / 1e3) / peak if peak else None
    say("ADIM PARCALARI (medyan, %d adim, ms; pay ADIM'a gore):" % len(parts))
    for n in names:
        say("   %-46s %9.1f  %5.1f%%" % (n, med[n], 100 * med[n] / total))
    say("   token/sn %.0f | MFU %s (6N + 12Ld x anahtar; tepe %s)" % (
        targets / (total / 1e3), "-" if peak is None else "%.1f%%" % (100 * res["replica_mfu"]),
        "-" if peak is None else "%.1f TFLOP/s" % (peak / 1e12)))
    # 2) bilesenler (bir parca)
    with guarded('bilesenler (bir parca)', res):
        one = tuple(t[0::micro][:rows].contiguous() for t in steps_data[0])
        comp, mask_obj, inner = components(ctx, one, flops, peak)
        res["components"] = comp
        turns, fturns = model.turns, sum(1 for t in range(model.turns) if I._turn_facts(model, t) is not None)
        say("BIR PARCA (%d satir x %d), ms:  kayip ileri %.1f, ileri+geri %.1f (MFU %s) | govde %.1f / %.1f | kayip basligi %.1f / %.1f"
            % (rows, T, comp["loss_fwd_ms"], comp["loss_fwd_bwd_ms"],
               "-" if comp["mfu_micro"] is None else "%.1f%%" % (100 * comp["mfu_micro"]), comp["body_fwd_ms"],
               comp["body_fwd_bwd_ms"], comp["head_fwd_ms"], comp["head_fwd_bwd_ms"]))
        say("   attention modulu tur basina %.2f / %.2f (x%d tur = %.1f ileri+geri) | FactUnits %.2f / %.2f (x%d = %.1f) | "
            "BlockMask %.2f" % (comp["attention_module_fwd_ms"], comp["attention_module_fwd_bwd_ms"], turns,
                                turns * comp["attention_module_fwd_bwd_ms"], comp["facts_fwd_ms"], comp["facts_fwd_bwd_ms"],
                                fturns, fturns * comp["facts_fwd_bwd_ms"], comp["block_mask_ms"]))
        kc = comp["kernels_fwd_bwd"]
        if kc:
            tot = sum(kc["categories"].values())
            say("   kernel kategorileri (bir parca ileri+geri, toplam %.1f ms):" % tot)
            for c, v in sorted(kc["categories"].items(), key=lambda x: -x[1]):
                say("      %-46s %8.1f  %5.1f%%" % (c, v, 100 * v / tot))
            say("   en pahali kernel'ler (ms, cagri, ad):")
            for t, n, name in kc["top"][:20]:
                say("      %8.2f %5d  %s" % (t, n, name))
    # 3) flex varyantlari
    with guarded('flex varyantlari', res):
        fv = flex_variants(model, inner, mask_obj, device, cuda)
        res["flex"] = fv
        if fv:
            say("FLEX tek basina (B %d, H %d, T %d, bf16; tur basina), ms ve varsayilana gore fark:" % (rows, model.blocks[0].attention.heads, T))
            for n, r in fv.items():
                say("   %-28s %s" % (n, r.get("error") or "ileri %.2f  ileri+geri %.2f  cikti farki %.1e  grad_q farki %.1e" % (
                    r["fwd_ms"], r["fwd_bwd_ms"], r["out_maxdiff"], r["grad_q_maxdiff"])))
    # 4) kure normlari kapali (DAVRANIS DEGISIR, yalniz sure)
    with guarded('kure normlari kapali (DAVRANIS DEGISIR', res):
        ids, mask, pos = (t.to(device) for t in one)
        with norms_off():
            lf = torch.compile(model.loss, dynamic=False) if ctx["compile"] else model.loss

            def fb():
                with fwd_context(ctx):
                    total_, _ = lf(ids, mask, pos)
                total_.backward()
                model.zero_grad(set_to_none=True)
            res["norms_off_fwd_bwd_ms"] = bench(fb)
        ctx["loss_fn"] = torch.compile(model.loss, dynamic=False) if ctx["compile"] else model.loss
        say("KURE NORMLARI KAPALI (DAVRANIS DEGISIR, yalniz sure): bir parca ileri+geri %.1f ms (normlu %.1f; fark %.1f%%)" % (
            res["norms_off_fwd_bwd_ms"], comp["loss_fwd_bwd_ms"],
            100 * (comp["loss_fwd_bwd_ms"] - res["norms_off_fwd_bwd_ms"]) / comp["loss_fwd_bwd_ms"]))
    # 5) LOSS_CHUNK
    with guarded('LOSS_CHUNK', res):
        res["loss_chunk"] = {}
        ref = None
        for chunk in (4096, 8192, 16384, 0):
            model.loss_chunk = chunk
            torch._dynamo.reset()
            lf = torch.compile(model.loss, dynamic=False) if ctx["compile"] else model.loss
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                with fwd_context(ctx):
                    total_, nll = lf(ids, mask, pos)
                total_.backward()
                gP = model.tokens.shift.grad.detach().clone() if model.tokens.shift.grad is not None else None
                model.zero_grad(set_to_none=True)
                if ref is None:
                    ref = (float(nll.detach()), gP)

                def fb():
                    with fwd_context(ctx):
                        t_, _ = lf(ids, mask, pos)
                    t_.backward()
                    model.zero_grad(set_to_none=True)
                res["loss_chunk"][chunk] = dict(fwd_bwd_ms=bench(fb), nll_diff=float(nll.detach()) - ref[0],
                                                grad_shift_maxdiff=None if gP is None else float((gP - ref[1]).abs().max()),
                                                peak_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None)
            except Exception as e:
                res["loss_chunk"][chunk] = dict(error=repr(e)[:200])
        model.loss_chunk = config["model_kw"].get("loss_chunk", 4096)
        torch._dynamo.reset()
        ctx["loss_fn"] = torch.compile(model.loss, dynamic=False) if ctx["compile"] else model.loss
        say("LOSS_CHUNK (bir parca ileri+geri ms, nll farki, shift gradyani farki, tepe GB): " + " | ".join(
            "%s %s" % (c, r.get("error") or "%.1f %.1e %s %.1f" % (r["fwd_bwd_ms"], r["nll_diff"],
                                                                 "-" if r["grad_shift_maxdiff"] is None else "%.1e" % r["grad_shift_maxdiff"],
                                                                 r["peak_gb"] or 0)) for c, r in res["loss_chunk"].items()))
    # 6) optimizer tarafi: EMA foreach (bit duzeyinde ayni mi), Newton-Schulz bf16 (DAVRANIS DEGISIR)
    with guarded('optimizer tarafi: EMA foreach (bit duz', res):
        if ctx["ema"] is not None:
            ema = ctx["ema"]["model"]
            e1, e2 = copy.deepcopy(ema), copy.deepcopy(ema)
            with torch.no_grad():
                ema_loop(e1, model, 0.999)
                ema_foreach(e2, model, 0.999)
            same = all(torch.equal(a, b) for a, b in zip(e1.parameters(), e2.parameters()))
            with torch.no_grad():
                t_loop = bench(lambda: ema_loop(e1, model, 0.999))
                t_each = bench(lambda: ema_foreach(e2, model, 0.999))
                t_norm = bench(lambda: e1.normalize_weights())
            res["ema"] = dict(loop_ms=t_loop, foreach_ms=t_each, bit_identical=same, normalize_weights_ms=t_norm)
            del e1, e2
            say("EMA: dongu %.2f ms, foreach %.2f ms (bit duzeyinde ayni: %s) | normalize_weights %.2f ms" % (
                t_loop, t_each, same, t_norm))
        params_muon = [p for g in ctx["opt"].param_groups if g["use_muon"] for p in g["params"]]
        shapes = {}
        for p in params_muon:
            shapes.setdefault(tuple(p.shape), []).append(p)
        ns = {}
        for prec in ("fp32", "bf16"):
            ns[prec] = sum(bench(lambda G=torch.randn(len(ps), *s, device=device): TR.Muon.orthogonalize(G, precision=prec))
                           for s, ps in shapes.items())
        G = torch.randn(len(next(iter(shapes.values()))), *next(iter(shapes)), device=device)
        diff = (TR.Muon.orthogonalize(G, precision="bf16").float() - TR.Muon.orthogonalize(G, precision="fp32")).norm() / \
            TR.Muon.orthogonalize(G, precision="fp32").norm()
        res["newton_schulz"] = dict(fp32_ms=ns["fp32"], bf16_ms=ns["bf16"], rel_diff=float(diff),
                                    shapes={str(s): len(ps) for s, ps in shapes.items()})
        say("NEWTON-SCHULZ (adim basina, sekil gruplari %s): fp32 %.1f ms, bf16 %.1f ms (DAVRANIS DEGISIR: goreli fark %.1e)" % (
            res["newton_schulz"]["shapes"], ns["fp32"], ns["bf16"], diff))
    # 7) adimin satir / parca bolusu (ayni 64 satir, ayni gradyan): bellek ve sure
    with guarded('adimin satir / parca bolusu (ayni 64 s', res):
        res["micro_split"] = {}
        full = steps_data[0]
        clock = Clock(cuda)
        accumulate(ctx, full, micro, clock)
        clock.read()
        ref_grad = [None if p.grad is None else p.grad.detach().clone() for p in ctx["params"]]
        ctx["opt"].zero_grad()
        for m in sorted({micro, micro // 2, micro // 4} - {0}):
            if (rows * micro) % m or m % 2:
                continue
            try:
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                accumulate(ctx, full, m, Clock(cuda))            # isinma (yeni sekil: compile)
                gdiff = max(float((p.grad - r).abs().max() / (r.abs().max() + 1e-30)) for p, r in zip(ctx["params"], ref_grad)
                            if r is not None)
                ctx["opt"].zero_grad()
                ms = bench(lambda: (accumulate(ctx, full, m, Clock(cuda)), ctx["opt"].zero_grad()), reps=2, warmup=0)
                res["micro_split"][m] = dict(rows=rows * micro // m, accumulate_ms=ms, grad_rel_maxdiff=gdiff,
                                             peak_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None)
            except torch.cuda.OutOfMemoryError as e:
                res["micro_split"][m] = dict(rows=rows * micro // m, error="OOM " + repr(e)[:120])
                ctx["opt"].zero_grad()
                if cuda:
                    torch.cuda.empty_cache()
        say("ADIMIN BOLUSU (64 satirin ileri+geri birikimi; gradyan farki %d parcaya gore): " % micro + " | ".join(
            "%d parca x %d satir %s" % (m, r["rows"], r.get("error") or "%.0f ms, tepe %.1f GB, grad farki %.1e" % (
                r["accumulate_ms"], r["peak_gb"] or 0, r["grad_rel_maxdiff"])) for m, r in res["micro_split"].items()))
    # 8) train_seq'in kendisi (sadakat): ayni batch
    with guarded("train_seq'in kendisi (sadakat): ayni b", res):
        del ctx, model
        torch._dynamo.reset()
        if cuda:
            torch.cuda.empty_cache()
        if args.real:
            real = real_step_ms(config, steps_data[0], device, K=args.real)
            res["real"] = real
            say("TRAIN_SEQ'IN KENDISI: %.0f ms/adim (parca parca kopya %.0f; kosu log'u 5.458-5.511), tepe %.1f GB, token/sn %.0f, "
                "MFU %s" % (real["step_ms"], total, real["peak_gb"] or 0, targets / (real["step_ms"] / 1e3),
                            "-" if not peak else "%.1f%%" % (100 * train_flops(flops, targets, keys) / (real["step_ms"] / 1e3) / peak)))
    write(args.run, "train", res)


# ---- buyuk model adaylari

CANDIDATES = (("d1024_8x2 (bugunku)", dict(d=1024, layers=8, turns=16, heads=8, units=2752), 4),
              ("d1280_8x2", dict(d=1280, layers=8, turns=16, heads=10, units=3456), 4),
              ("d1536_8x2", dict(d=1536, layers=8, turns=16, heads=12, units=4096), 2),
              ("d1536_12x2", dict(d=1536, layers=12, turns=24, heads=12, units=4096), 2),
              ("d2048_8x2", dict(d=2048, layers=8, turns=16, heads=16, units=5504), 2))


def sizes_measure(args, config, device):
    cuda = torch.device(device).type == "cuda"
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    peak = peak_for(gpu)
    say("analyze_speed sizes | %s | torch %s | %s" % (time.strftime("%Y-%m-%d %H:%M"), torch.__version__, gpu))
    T = config["seq_len"]
    total_rows = config["batch_size"] * config["micro_batches"]
    batch = (packed_batches(args.data, total_rows, 1, T)[0] if args.data else fake_batches(total_rows, 1, T, config["vocab"]))[0]
    keys, targets = keys_per_target(batch)
    out = []
    chosen = CANDIDATES if not args.only else [c for c in CANDIDATES if c[0].split()[0] in args.only.split(",")]
    for label, kw, rows in chosen:
        if not cuda:
            kw, rows = dict(kw, d=64, heads=2, units=128), 2
        r = None
        while rows >= 1 and r is None:
            try:
                info = real_step_ms(config, batch, device, K=args.real or 2, micro=total_rows // rows, rows=rows,
                                    model_kw=dict(kw, rope_base=config["model_kw"]["rope_base"]))
                tps = targets / (info["step_ms"] / 1e3)
                mfu = train_flops(info["flops"], targets, keys) / (info["step_ms"] / 1e3) / peak if peak else None
                r = dict(label=label, model_kw=kw, rows=rows, micro=info["micro"], step_ms=info["step_ms"], tokens_per_sec=tps,
                         peak_gb=info["peak_gb"], mfu=mfu, params=info["params"], body=info["body"],
                         flops=info["flops"], nll_after=info["nll"])
            except torch.cuda.OutOfMemoryError:
                say("   %s: %d satirda bellek yetmedi, %d satir" % (label, rows, rows // 2))
                rows //= 2
            finally:
                torch._dynamo.reset()
                if cuda:
                    torch.cuda.empty_cache()
        if r is None:
            out.append(dict(label=label, error="OOM"))
            continue
        out.append(r)
        h10 = 10e9 / r["tokens_per_sec"] / 3600
        say("%-20s govde %6.1fM toplam %6.1fM | %d satir x %d parca | %6.0f ms/adim %7.0f token/sn | tepe %5.1f GB | MFU %s | "
            "10G token %5.1f saat = %5.0f CU" % (label, r["body"] / 1e6, r["params"] / 1e6, r["rows"], r["micro"], r["step_ms"],
                                                  r["tokens_per_sec"], r["peak_gb"] or 0,
                                                  "-" if mfu is None else "%.1f%%" % (100 * r["mfu"]), h10, 8.9 * h10))
    write(args.run, "sizes" + ("_" + args.only.replace(",", "_") if args.only else ""),
          dict(gpu=gpu, torch=torch.__version__, peak_bf16=peak, keys_per_target=keys, targets=targets, results=out))


# ---- uretim

class StaticCache(M.AttentionCache):
    """ONERI: AttentionCache'in sabit sekilli hali -- attention butun kapasite uzerinde maskeyle (dilim boyu degismez),
    konum sayaci yerinde artar: adim CUDA graph'a alinabilir.  Istem (ilk cagri) AttentionCache'in kendisi."""

    def attend(self, at, x):
        if self.keys is None:
            return super().attend(at, x)
        H = at.heads
        v = x if H == 1 else (x @ at.W_value.T).unflatten(-1, (H, -1)).transpose(1, 2)
        pos = self.next_position[:, None].clone()
        q, k = at.queries_keys(x, pos)
        at_pos = pos[..., None] if H == 1 else pos[:, None, :, None]
        self.keys.scatter_(-2, at_pos.expand(*k.shape), k)
        self.values.scatter_(-2, at_pos.expand(*v.shape), v)
        self.next_position.add_(1)
        allowed = torch.arange(self.capacity, device=x.device)[None, None, :] <= pos[..., None]
        if H > 1:
            allowed = allowed[:, None]
        out = F.scaled_dot_product_attention(q, self.keys, self.values, attn_mask=allowed, scale=at.scale)
        return out if H == 1 else out.transpose(1, 2).flatten(-2)


class ConstantAttentionCache(M.AttentionCache):
    """DAVRANIS DEGISIR: bu turun attention ciktisi (W_context'ten once, head'ler yan yana) sabit vektor; q/k/v, onbellek
    ve Canon karisimi hesaplanmaz (Canon yalniz attention girdisi)."""

    def __init__(self, lengths, capacity, constant):
        super().__init__(lengths, capacity)
        self.constant = constant

    def canon_cache(self, x):
        return F.pad(x, (0, 0, 3, 0))

    def attend(self, at, x):
        return self.constant.to(x.dtype).expand(x.shape[0], x.shape[1], -1)


def make_caches(model, lengths, capacity, kind="dynamic", constants=None):
    out = []
    for t in range(model.turns):
        if constants and t in constants:
            out.append(ConstantAttentionCache(lengths, capacity, constants[t]))
        elif kind == "static":
            out.append(StaticCache(lengths, capacity))
        else:
            out.append(M.AttentionCache(lengths, capacity))
    return out


class GraphDecoder:
    """ONERI: istemden sonra tek token'lik adim CUDA graph'ta (sabit onbellek); acgozlu token graph icinde, CPU'ya inmez.
    compile=True: adim once torch.compile'la birlestirilir, sonra graph'a alinir."""

    def __init__(self, model, prompt, n, compile=False, autocast=False, constants=None):
        self.model, dev = model, prompt.device
        B, L = prompt.shape
        self.caches = make_caches(model, torch.full((B,), L, device=dev), L + n + 1, "static", constants)
        self.autocast = autocast
        with self._ctx():
            first = model.logits(prompt, self.caches)[:, -1].float()
        self.first_logits = first
        self.token = first.argmax(-1).clone()
        self.logits = None
        fn = self._step
        self.fn = torch.compile(fn, dynamic=False) if compile else fn
        state = self._snapshot()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.fn()
        torch.cuda.current_stream().wait_stream(s)
        self._restore(state)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self.fn()
        self._restore(state)

    def _ctx(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext()

    def _step(self):
        with self._ctx():
            logits = self.model.logits(self.token[:, None], self.caches)[:, 0].float()
        self.token.copy_(logits.argmax(-1))
        return logits

    def _snapshot(self):
        return [(c.next_position.clone(), None if c.canon_inputs is None else c.canon_inputs.clone()) for c in self.caches]

    def _restore(self, state):
        for c, (p, ci) in zip(self.caches, state):
            c.next_position.copy_(p)
            if ci is not None:
                c.canon_inputs.copy_(ci)

    def run(self, n, forced=None):
        """n adim; forced (B, n): token'lar disaridan (esdegerlik kontrolu) -> (token'lar (B, n), logits listesi ya da None)."""
        toks, logs = [self.token.clone()], []
        for i in range(n - 1):
            if forced is not None:
                self.token.copy_(forced[:, i])
            self.graph.replay()
            toks.append(self.token.clone())
            if forced is not None:
                logs.append(self.logits.clone())
        return torch.stack(toks, 1), logs


@torch.no_grad()
def eager_decode(model, prompt, n, forced=None, autocast=False, constants=None):
    """Bugunku yol (train_y.generate_cached / konus_fineweb.generate gibi): AttentionCache, adim adim, acgozlu."""
    B, L = prompt.shape
    caches = make_caches(model, torch.full((B,), L, device=prompt.device), L + n + 1, "dynamic", constants)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with ctx:
        logits = model.logits(prompt, caches)[:, -1].float()
    tok, toks, logs = logits.argmax(-1), [], []
    toks.append(tok)
    for i in range(n - 1):
        if forced is not None:
            tok = forced[:, i]
        with ctx:
            logits = model.logits(tok[:, None], caches)[:, 0].float()
        tok = logits.argmax(-1)
        toks.append(tok)
        if forced is not None:
            logs.append(logits)
    return torch.stack(toks, 1), logs


def mean_attention_outputs(model, docs, turns):
    """Tur basina attention ciktisinin (head'ler yan yana, W_context'ten once) ortalamasi -- internals_y ablate'in head
    ortalamasi gibi, tek belge tek tek."""
    acc = {t: [] for t in turns}

    def tap(t, c):
        if t in acc:
            acc[t].append(c.float().mean(dim=(0, 2)).flatten())        # (H, dh) -> d
    with torch.no_grad():
        for ids in docs:
            x = ids[None, :-1]
            I._run(model, model.input_states(x), I._plan(model), dict(heads=tap))
    return {t: torch.stack(v).mean(0) for t, v in acc.items()}


def nll_with(model, ids, constants=None):
    """Istem ileri hesabi (onbellekli yolun ilk cagrisi) -> ortalama nll."""
    caches = make_caches(model, torch.full((1,), ids.shape[0] - 1, device=ids.device), ids.shape[0], "dynamic", constants)
    with torch.no_grad():
        logits = model.logits(ids[None, :-1], caches)[0].float()
    return float(F.cross_entropy(logits, ids[1:]))


def generate_measure(args, config, device):
    cuda = torch.device(device).type == "cuda"
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    say("analyze_speed generate | %s | torch %s | %s" % (time.strftime("%Y-%m-%d %H:%M"), torch.__version__, gpu))
    res = dict(gpu=gpu, torch=torch.__version__)
    if args.selftest:
        model = I._build(config).eval().requires_grad_(False).to(device)
        docs = [torch.randint(0, config["vocab"], (300,)) for _ in range(6)]
        decode = None
    else:
        model = I._load_model(args.run, args.weights).to(device)
        import data_fineweb as DF
        v = DF.load_valid(args.data, log=lambda s: None)
        starts = list(v["valid_starts"]) + [len(v["valid"])]
        docs = [torch.from_numpy(v["valid"][starts[i]:starts[i + 1]].astype("int64")) for i in range(len(starts) - 1)]
        decode = lambda ids: v["tokenizer"].decode(ids, skip_special_tokens=False)
    stream = torch.cat(docs)                                 # uzun istemler icin art arda belgeler (yalniz hiz)
    n = args.tokens

    def prompt(B, L, offset=0):
        return torch.stack([stream[offset + r * L: offset + (r + 1) * L] for r in range(B)]).to(device)

    def timed(fn):
        sync()
        t = time.perf_counter()
        r = fn()
        sync()
        return r, 1e3 * (time.perf_counter() - t)
    lengths = (16, 1024, 4096, 7680) if not args.selftest else (16, 64)
    # 1) bugunku yol, batch 1: istem (prefill) + n token
    res["eager_b1"] = []
    for L in lengths:
        p = prompt(1, L)
        eager_decode(model, p, 4)                            # isinma
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        _, pre = timed(lambda: eager_decode(model, p, 1))
        _, tot = timed(lambda: eager_decode(model, p, n))
        r = dict(L=L, prefill_ms=pre, decode_ms_per_token=(tot - pre) / (n - 1), tokens_per_sec=(n - 1) / ((tot - pre) / 1e3),
                 prefill_tokens_per_sec=L / (pre / 1e3), peak_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None)
        res["eager_b1"].append(r)
        say("BUGUNKU YOL fp32 batch 1, istem %5d: prefill %7.1f ms (%6.0f token/sn) | uretim %.2f ms/token = %.0f token/sn | tepe %.2f GB"
            % (L, pre, r["prefill_tokens_per_sec"], r["decode_ms_per_token"], r["tokens_per_sec"], r["peak_gb"] or 0))
    # 2) bugunku yol, buyuk batch
    res["eager_batch"] = []
    for B, L in ((8, 128), (32, 128), (128, 128), (8, 2048), (32, 2048)) if not args.selftest else ((4, 16),):
        try:
            p = prompt(B, L)
            eager_decode(model, p, 4)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            _, pre = timed(lambda: eager_decode(model, p, 1))
            _, tot = timed(lambda: eager_decode(model, p, n))
            r = dict(B=B, L=L, decode_ms_per_step=(tot - pre) / (n - 1), tokens_per_sec=B * (n - 1) / ((tot - pre) / 1e3),
                     peak_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None)
        except Exception as e:
            r = dict(B=B, L=L, error=repr(e)[:160])
        res["eager_batch"].append(r)
        say("BUGUNKU YOL fp32 batch %3d, istem %4d: %s" % (B, L, r.get("error") or "%.2f ms/adim = %.0f token/sn | tepe %.2f GB" % (
            r["decode_ms_per_step"], r["tokens_per_sec"], r["peak_gb"] or 0)))
    if not cuda:
        write(args.run, "generate_selftest", res)
        return
    # 3) ONERI: sabit onbellek + CUDA graph (esdegerlik: zorlanmis token'larla logits farki)
    res["graph"] = []
    p = prompt(1, 1024)
    forced = prompt(1, 64, offset=5000)
    _, ref_logs = eager_decode(model, p, 65, forced=forced)
    for compile_ in (False, True):
        for ac in (False, True):
            label = "graph%s%s" % ("+compile" if compile_ else "", " bf16 (DAVRANIS DEGISIR)" if ac else " fp32")
            try:
                g = GraphDecoder(model, p, 65, compile=compile_, autocast=ac)
                _, logs = g.run(65, forced=forced)
                diff = max(float((a - b).abs().max()) for a, b in zip(logs, ref_logs))
                agree = sum(int((a.argmax(-1) == b.argmax(-1)).all()) for a, b in zip(logs, ref_logs)) / len(logs)
                rows = []
                for B, L in ((1, 16), (1, 1024), (1, 4096), (1, 7680), (8, 1024), (32, 1024), (128, 128)):
                    pp = prompt(B, L)
                    torch.cuda.reset_peak_memory_stats()
                    dec = GraphDecoder(model, pp, n, compile=compile_, autocast=ac)
                    _, ms = timed(lambda: dec.run(n))
                    eager_ms = None
                    rows.append(dict(B=B, L=L, ms_per_step=ms / (n - 1), tokens_per_sec=B * (n - 1) / (ms / 1e3),
                                     peak_gb=torch.cuda.max_memory_allocated() / 1e9))
                    del dec
                r = dict(label=label, logits_maxdiff=diff, argmax_agree=agree, rows=rows)
                del g
            except Exception as e:
                r = dict(label=label, error=repr(e)[:300])
            res["graph"].append(r)
            say("%s: %s" % (label, r.get("error") or "logits farki %.2e, acgozlu token ayni %.0f%% | %s" % (
                r["logits_maxdiff"], 100 * r["argmax_agree"], " | ".join(
                    "B%d L%d %.2f ms/adim %.0f token/sn" % (x["B"], x["L"], x["ms_per_step"], x["tokens_per_sec"])
                    for x in r["rows"]))))
            torch.cuda.empty_cache()
    # 4) DAVRANIS DEGISIR: son iki turun attention'i sabit (ortalama cikti; ablate: +0,000 nat)
    with guarded("sabit attention", res):
        turns = [model.turns - 2, model.turns - 1]
        ref_docs = [d[:2049] for d in docs[-40:-24] if len(d) > 64]
        constants = {t: v.to(device) for t, v in mean_attention_outputs(model, [d.to(device) for d in ref_docs], turns).items()}
        test_docs = [d[:2049].to(device) for d in docs[-24:] if len(d) > 64][:16]
        base = [nll_with(model, d) for d in test_docs]
        const = [nll_with(model, d, constants) for d in test_docs]
        toks_ref, _ = eager_decode(model, prompt(8, 256), 128)
        toks_c, _ = eager_decode(model, prompt(8, 256), 128, constants=constants)
        same_prefix = (toks_ref == toks_c).int().cumprod(1).sum(1).float().mean()
        texts = []
        for r in range(2):
            pr = prompt(8, 256)[r, -40:].tolist()
            texts.append(dict(prompt=decode(pr), full=decode(toks_ref[r, :80].tolist()), constant=decode(toks_c[r, :80].tolist())))
            flat = lambda t: t.replace("\n", " / ")
            say("   METIN %d istem sonu: ...%s" % (r, flat(texts[-1]["prompt"])))
            say("      tam  : %s" % flat(texts[-1]["full"]))
            say("      sabit: %s" % flat(texts[-1]["constant"]))
        rows = []
        for B, L in ((1, 1024), (1, 7680), (32, 1024)):
            pp = prompt(B, L)
            out = {}
            for name, cst in (("tam", None), ("sabit", constants)):
                dec = GraphDecoder(model, pp, n, compile=True, constants=cst)
                _, ms = timed(lambda: dec.run(n))
                out[name] = ms / (n - 1)
                del dec
            rows.append(dict(B=B, L=L, full_ms=out["tam"], constant_ms=out["sabit"]))
        res["constant_last_attention"] = dict(turns=[t + 1 for t in turns], nll_base=sum(base) / len(base),
                                              nll_constant=sum(const) / len(const), greedy_same_prefix=float(same_prefix),
                                              texts=texts, nll_per_doc=list(zip(base, const)),
                                              rows=rows)
        say("SON IKI TUR (%s) ATTENTION SABIT (DAVRANIS DEGISIR): nll %.4f -> %.4f (%d belge) | acgozlu 128 token'da ayni onek %.1f | %s"
            % ([t + 1 for t in turns], sum(base) / len(base), sum(const) / len(const), len(test_docs), float(same_prefix),
               " | ".join("B%d L%d %.2f -> %.2f ms/adim" % (r["B"], r["L"], r["full_ms"], r["constant_ms"]) for r in rows)))
    write(args.run, "generate", res)


# ---- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="kosu klasoru (config.json)")
    ap.add_argument("measure", nargs="?", choices=("train", "sizes", "generate"))
    ap.add_argument("--data", help="FineWeb koku (gpt2/shard_013 ...); yoksa rastgele token")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=3, help="train: parca parca olculen adim")
    ap.add_argument("--real", type=int, default=2, help="train_seq'in kendisi: K isinma + K olcum adimi (0: yok)")
    ap.add_argument("--only", help="sizes: aday adlari, virgulle (d1280_8x2,...)")
    ap.add_argument("--weights", default="last", choices=("last", "ema"))
    ap.add_argument("--tokens", type=int, default=256, help="generate: istem basina uretilen token")
    ap.add_argument("--selftest", action="store_true", help="CPU, kucuk model: kod yollari")
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 64
    if args.selftest:
        config = dict(json.load(open(os.path.join(args.run, "config.json"))) if os.path.exists(
            os.path.join(args.run, "config.json")) else {})
        config["model_kw"] = dict(config["model_kw"], d=64, heads=2, units=128, layers=2, turns=4, t_max=256)
        config.update(seq_len=256, batch_size=2, micro_batches=4, vocab=1000)
        os.environ.setdefault("KUYRUK_SONUC", tempfile.gettempdir())
        args.device, args.steps, args.real, args.tokens = "cpu", 1, 1, 8
        for m in ("train", "generate", "sizes"):
            LINES.clear()
            args.measure = m
            {"train": train_measure, "sizes": sizes_measure, "generate": generate_measure}[m](args, config, "cpu")
        return
    config = I._config(args.run)
    {"train": train_measure, "sizes": sizes_measure, "generate": generate_measure}[args.measure](args, config, args.device)


if __name__ == "__main__":
    main()
