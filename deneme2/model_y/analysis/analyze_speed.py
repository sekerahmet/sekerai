# -*- coding: utf-8 -*-
"""analyze_speed -- Model Y performans analizi (ajan C, 2 Ekim 2026): hiz, verim, maliyet; DOGRULUK DEGIL.

Kullanici: "Performans da bir kriter olsun ama ayrica performans icin de ajan tayin et olur mu ?"

    train      egitim adimi parca parca (ileri / geri / grup kopyasi / coherence / clip / Muon + Newton-Schulz / Adam /
               normalize_weights / EMA), train_seq'in kendisiyle yan yana; bilesenler (govde, kayip basligi, attention,
               FactUnits), kernel kategorileri, MFU; kure normlari kapali (DAVRANIS DEGISIR, yalniz sure); LOSS_CHUNK.
    tune       hizlandirma denemeleri train_seq'in kendisiyle (parca bolusu, Newton-Schulz bf16); veri yolu suresi.
    steptrace  kosunun gercek veri sirasiyla adim adim sure, flex blok sayisiyla iliski, GPU saati / kisma.
    sizes      buyuk model adaylari: ms/adim, token/sn, bellek, MFU.
    generate   uretim: onbellekli batch 1 / buyuk batch, baglam boyu, prefill; sabit onbellek + CUDA graph (esdegerlik
               kontroluyle); bf16; son iki turun attention'i sabit (DAVRANIS DEGISIR).
    decode     egitim sirasinda kendi devami uretiminin maliyeti: t_dec(B), her k adimda ek yuk.

    python analysis/analyze_speed.py <kosu klasoru> train --data <FineWeb koku>
    python analysis/analyze_speed.py <kosu klasoru> --selftest          (CPU, kucuk model: kod yollari)
Cikti: $KUYRUK_SONUC (yoksa <kosu>/analysis) altina speed_<olcum>.json; satirlar stdout'a.
"""
import argparse
import contextlib
import json
import math
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
    data = dict(seq_len=seq_len, train=stream, train_starts=offsets, eot=v["eot"],
                fingerprint="speed_shard13_%d" % seq_len)          # pack() plani izle onbellekte: baglam boyu izde olmali
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


def optimizer_tail(ctx, clock):
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
                ema_loop(ctx["ema"]["model"], model, ctx["ema"]["decay"])
                if getattr(model, "sphere_weights", False):
                    ctx["ema"]["model"].normalize_weights()


def replica_step(ctx, batch, micro):
    clock = Clock(ctx["cuda"])
    with clock.part("ADIM"):
        accumulate(ctx, batch, micro, clock)
        optimizer_tail(ctx, clock)
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
        info.setdefault("nll_by_step", []).append((step, nll))
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
        if "CUDA" not in str(getattr(e, "device_type", "CUDA")):   # triton kernel'in CPU tarafi olayi da ayni adla (cift sayim)
            continue
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
    # 3) kure normlari kapali (DAVRANIS DEGISIR, yalniz sure)
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
    # 4) LOSS_CHUNK
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
    # 5) train_seq'in kendisi (sadakat): ayni batch
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


# ---- hizlandirma denemeleri: train_seq'in kendisi (F1)

def data_cost(root, rows, steps=5):
    """Kosudaki veri yolu: adim basina pencere kurma (data_fineweb.windows, CPU) ve cihaza kopya."""
    import data_fineweb as DF
    v = DF.load_valid(root, log=lambda s: None)
    stream, offsets, _ = v["valid_shard"]
    data = dict(seq_len=8192, train=stream, train_starts=offsets, eot=v["eot"], fingerprint="speed_shard13_8192")
    data["items"] = DF._items(data)
    draw = DF.batches(data, rows)
    draw(0)
    t = time.perf_counter()
    got = [draw(s) for s in range(1, steps + 1)]
    cpu_ms = 1e3 * (time.perf_counter() - t) / steps
    sync()
    t = time.perf_counter()
    for b in got:
        [x.to("cuda") for x in b]
    sync()
    return cpu_ms, 1e3 * (time.perf_counter() - t) / steps


def tune_measure(args, config, device):
    gpu = torch.cuda.get_device_name(0)
    peak = peak_for(gpu)
    say("analyze_speed tune | %s | torch %s | %s" % (time.strftime("%Y-%m-%d %H:%M"), torch.__version__, gpu))
    total_rows = config["batch_size"] * config["micro_batches"]
    batch = packed_batches(args.data, total_rows, 1, config["seq_len"])[0][0]
    keys, targets = keys_per_target(batch)
    cpu_ms, copy_ms = data_cost(args.data, total_rows)
    say("VERI YOLU (kosuda her adimda, egitim dongusunde sirayla): pencere kurma %.1f ms (CPU), cihaza kopya %.1f ms" % (
        cpu_ms, copy_ms))
    res = dict(gpu=gpu, torch=torch.__version__, data_windows_ms=cpu_ms, data_copy_ms=copy_ms, variants=[])
    K = args.real or 3
    variants = (("bugunku", None, "fp32"),
                ("8 parca x 8 satir", 8, "fp32"),
                ("+ Newton-Schulz bf16 (DAVRANIS DEGISIR)", 8, "bf16"))
    base = None
    for label, micro, ns in variants:
        cfg = dict(config, newton_schulz_precision=ns)
        try:
            info = real_step_ms(cfg, batch, device, K=K, micro=micro, rows=None if micro is None else total_rows // micro)
        except Exception:
            say("HATA %s: %s" % (label, traceback.format_exc()[-800:]))
            continue
        finally:
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        info["nlls"] = info.pop("nll")
        mfu = train_flops(info["flops"], targets, keys) / (info["step_ms"] / 1e3) / peak
        base = base or info["step_ms"]
        res["variants"].append(dict(label=label, step_ms=info["step_ms"], peak_gb=info["peak_gb"], mfu=mfu,
                                    nll_after=info["nlls"], tokens_per_sec=targets / (info["step_ms"] / 1e3)))
        say("%-42s %6.0f ms/adim (%+.1f%%)  %6.0f token/sn  MFU %.1f%%  tepe %.1f GB  nll(adim %d) %.6f" % (
            label, info["step_ms"], 100 * (info["step_ms"] / base - 1), targets / (info["step_ms"] / 1e3), 100 * mfu,
            info["peak_gb"], 2 * K, info["nlls"]))
    write(args.run, "tune", res)


# ---- adim suresi gercek veri sirasiyla (kosu ile olcum arasindaki fark)

def batch_stats(batch, block=128):
    """Adimin attention isi: hedef, anahtar/hedef, flex'in BlockMask'indaki etkin (q, kv) blok sayisi (belge ici nedensel),
    belge parcasi sayisi, en uzun belge."""
    _, mask, pos = batch
    p = pos[:, :-1]
    B, T = p.shape
    first = torch.arange(T)[None] - p                     # konumun belgesinin penceredeki ilk konumu
    lo = first.view(B, T // block, block).min(-1).values // block
    blocks = int((torch.arange(T // block)[None] - lo + 1).sum())
    valid = mask[:, 1:]
    starts = (p == 0).sum().item()
    return dict(targets=int(valid.sum()), keys=float(((p + 1) * valid).sum()) / float(valid.sum()), blocks=blocks,
                docs=starts, longest=int(p.max()) + 1)


_FLEX_NOTE = dict(on=False, log=[])     # tip kaydi yalniz derleme disindaki tek prob cagrisinda (on=True); egitimde yazilmaz


@contextlib.contextmanager
def training_flex_path(run_path):
    """run_path True: kosunun yolu (7b3c1fa oncesi) -- egitimde flex fp32, kucuk blok (BLOCK 32x32, num_stages 1); bugunku kod
    q, k'yi bf16'ya cevirip geri yayilim bloklarini veriyor, bu sarmalayici o cagriyi fp32'ye geri cevirir (degerler bf16
    yuvarlamali, sure ayni yol).  False: bugunku cagri aynen.  Kayit (D4) _FLEX_NOTE["on"] iken; derlenmis bolgede buyuyen
    liste her adimda yeniden derletiyordu (C_007: 64 derlemeden sonra eager, ~21 sn/adim)."""
    from torch.nn.attention.flex_attention import flex_attention
    original = M._flex_compiled
    inner = torch.compile(lambda q, k, v, block_mask=None, scale=None, kernel_options=None: flex_attention(
        q, k, v, block_mask=block_mask, scale=scale, kernel_options=kernel_options), dynamic=False)

    def call(q, k, v, block_mask=None, scale=None, kernel_options=None):
        if run_path and kernel_options and "BLOCK_M1" in kernel_options:
            q, k, v, kernel_options = q.float(), k.float(), v.float(), dict(BLOCK_M=32, BLOCK_N=32, num_stages=1)
        if _FLEX_NOTE["on"]:
            _FLEX_NOTE["log"].append(dict(q=str(q.dtype), k=str(k.dtype), v=str(v.dtype), kernel_options=kernel_options))
        return inner(q, k, v, block_mask=block_mask, scale=scale, kernel_options=kernel_options)
    M._flex_compiled = call
    torch._dynamo.reset()
    try:
        yield
    finally:
        M._flex_compiled = original
        torch._dynamo.reset()


def probe_flex_call(config, batch, device):
    """Derlemesiz tek ileri hesap (1 satir, autocast bf16): flex'e giden tip ve secenekler."""
    _FLEX_NOTE.update(on=True, log=[])
    try:
        model = I._build(config).to(device)
        ids, mask, pos = (t[:1].contiguous().to(device) for t in batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.loss(ids, mask, pos)
        del model
        torch.cuda.empty_cache()
        return _FLEX_NOTE["log"][:1]
    finally:
        _FLEX_NOTE.update(on=False, log=[])


def gpu_sampler(samples, stop, every=5.0):
    """Arka planda nvidia-smi: SM saati, sicaklik, guc, kisma nedeni (uzun kosuda isil kisma var mi)."""
    import subprocess
    q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            samples.append([time.time()] + [x.strip() for x in out.split(",")])
        except Exception as e:
            samples.append([time.time(), repr(e)[:80]])
        stop.wait(every)


def fit_line(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx if sxx else 0.0
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    return my - slope * mx, slope, r


def steptrace_measure(args, config, device):
    import threading
    import data_fineweb as DF
    gpu = torch.cuda.get_device_name(0)
    say("analyze_speed steptrace | %s | torch %s | %s" % (time.strftime("%Y-%m-%d %H:%M"), torch.__version__, gpu))
    data = DF.load(args.data, log=say)
    assert data["fingerprint"] == config["fingerprint"], "veri izi kosununkiyle ayni degil"
    rows = config["batch_size"] * config["micro_batches"]
    draw = DF.batches(data, rows, config.get("seed", 0))
    starts = [int(s) for s in args.at.split(",")]
    N = args.window
    seq = [s for a in starts for s in range(a, a + N + 1)]
    cached = [draw(s) for s in seq]
    stats = [batch_stats(b) for b in cached]
    g = torch.Generator().manual_seed(0)
    sample = [batch_stats(draw(int(s))) for s in torch.randint(0, config["steps"], (args.sample,), generator=g)]
    mean = lambda xs: sum(xs) / len(xs)
    res = dict(gpu=gpu, torch=torch.__version__, at=starts, steps_per_window=N, stats=stats,
               epoch_sample=dict(n=len(sample), **{k: mean([s[k] for s in sample]) for k in ("keys", "blocks", "docs")},
                                 blocks_sd=float(torch.tensor([float(s["blocks"]) for s in sample]).std())),
               modes={})
    say("EPOK ORNEGI (%d rastgele adim): anahtar/hedef %.0f, flex blok %.0f (sd %.0f), belge parcasi %.0f | olculen adimlar: "
        "anahtar/hedef %.0f, blok %.0f" % (len(sample), res["epoch_sample"]["keys"], res["epoch_sample"]["blocks"],
                                          res["epoch_sample"]["blocks_sd"], res["epoch_sample"]["docs"],
                                          mean([s["keys"] for s in stats]), mean([s["blocks"] for s in stats])))
    kw = I._train_kwargs(config)
    for label in ("kosudaki yol (fp32 flex)", "bugunku kod (bf16 flex, 7b3c1fa)"):
        log, marks, samples = [], [], []
        stop = threading.Event()
        sampler = threading.Thread(target=gpu_sampler, args=(samples, stop), daemon=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sampler.start()
        try:
            with training_flex_path(label.startswith("kosu")):
                log = probe_flex_call(config, cached[0], device)
                TR.train_seq(config.get("setting", "shared"), None, None, config["vocab"], steps=len(seq) - 1, device=device,
                             compile=config.get("compile", True), log_at=(), every=1,
                             callback=lambda s, m, nll: marks.append(time.perf_counter()),
                             batches=lambda s: cached[s], **kw)
        except Exception:
            say("HATA %s: %s" % (label, traceback.format_exc()[-800:]))
            continue
        finally:
            stop.set()
            sampler.join(timeout=15)
            torch._dynamo.reset()
        # aralik k: adim k'nin geri yayilimi + optimizer + adim k+1'in ileri hesabi; pencere sinirindaki ve ilk 2 aralik disari
        ms, x_blocks, x_keys, window = [], [], [], []
        for k in range(2, len(marks) - 1):
            if k % (N + 1) == N:                          # pencere siniri: sonraki adim baska yerden
                continue
            ms.append(1e3 * (marks[k + 1] - marks[k]))
            x_blocks.append((stats[k]["blocks"] + stats[k + 1]["blocks"]) / 2)
            x_keys.append((stats[k]["keys"] + stats[k + 1]["keys"]) / 2)
            window.append(k // (N + 1))
        a, b, r = fit_line(x_blocks, ms)
        ak, bk, rk = fit_line(x_keys, ms)
        srt = sorted(ms)
        sms = [float(s[1]) for s in samples if len(s) > 2 and s[1].replace(".", "").isdigit()]
        temps = [float(s[3]) for s in samples if len(s) > 3 and s[3].replace(".", "").isdigit()]
        flags = sorted({s[5] for s in samples if len(s) > 5})
        pred = a + b * res["epoch_sample"]["blocks"]
        m = res["modes"][label] = dict(record=log[:2], intervals=len(ms), ms_mean=mean(ms), ms_median=srt[len(srt) // 2],
                                       ms_min=srt[0], ms_max=srt[-1], ms_sd=float(torch.tensor(ms).std()),
                                       fit_blocks=dict(intercept=a, slope=b, r=r), fit_keys=dict(intercept=ak, slope=bk, r=rk),
                                       predicted_epoch_ms=pred, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                                       sm_clock=[min(sms), max(sms)] if sms else None,
                                       temperature=[min(temps), max(temps)] if temps else None, throttle_flags=flags,
                                       samples=samples, ms=ms)
        say("%s | kayit %s" % (label, log[:1]))
        say("   %d aralik: ortalama %.0f ms, medyan %.0f, en az %.0f, en cok %.0f, sd %.0f | tepe %.1f GB" % (
            len(ms), m["ms_mean"], m["ms_median"], m["ms_min"], m["ms_max"], m["ms_sd"], m["peak_gb"]))
        say("   ms ~ flex blok: %.0f + %.4f x blok (r %.2f) -> epok ortalamasinda %.0f ms | ms ~ anahtar/hedef: egim %.3f (r %.2f)" % (
            a, b, r, pred, bk, rk))
        say("   GPU: SM saati %s MHz, sicaklik %s C, kisma bayraklari %s (%d ornek)" % (
            m["sm_clock"], m["temperature"], flags, len(samples)))
        for i, start in enumerate(starts):
            part = [v for v, w in zip(ms, window) if w == i]
            if part:
                say("   pencere adim %d: ortalama %.0f ms" % (start, mean(part)))
    write(args.run, "steptrace", res)


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
        """Isinma adimlari konumu, Canon halkasini ve token tamponunu ilerletir; uchu de geri yuklenir (token tamponu
        unutulunca serbest uretim istemin argmax'i yerine 3 adim sonrasinin token'iyla basliyordu: C_008)."""
        return self.token.clone(), [(c.next_position.clone(), None if c.canon_inputs is None else c.canon_inputs.clone())
                                    for c in self.caches]

    def _restore(self, state):
        token, caches = state
        self.token.copy_(token)
        for c, (p, ci) in zip(self.caches, caches):
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


# ---- egitim sirasinda kendi devami uretimi (mathematician O15): t_dec(B), ek yuk

def decode_measure(args, config, device):
    """Onbellekli acgozlu uretim, istem P token, devam G token, B = 32 / 64 / 128: bugunku ortak yol (train_y.generate_cached,
    eager) fp32 ve autocast bf16; KIYAS: sabit onbellek + CUDA graph (GraphDecoder, fp32).  Kosu basina egitim adimi da
    olculur (train_seq'in kendisi, shard 13 paketi) -> "her k adimda bir uretim" ek yuku."""
    import data_fineweb as DF
    gpu = torch.cuda.get_device_name(0)
    say("analyze_speed decode | %s | torch %s | %s" % (time.strftime("%Y-%m-%d %H:%M"), torch.__version__, gpu))
    P, G = args.prefix, args.tokens
    v = DF.load_valid(args.data, log=lambda s: None)
    starts = list(v["valid_starts"]) + [len(v["valid"])]
    docs = [torch.from_numpy(v["valid"][a:b].astype("int64")) for a, b in zip(starts, starts[1:]) if b - a > P]
    res = dict(gpu=gpu, torch=torch.__version__, prefix=P, generated=G, runs={})

    def timed(fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        return out, 1e3 * (time.perf_counter() - t)
    for run in [args.run] + [r for r in (args.extra or "").split(",") if r]:
        cfg = I._config(run)
        name = cfg.get("name", os.path.basename(run))
        r = res["runs"][name] = dict(rows=[])
        with guarded("adim " + name, r):                # egitim adimi: train_seq'in kendisi, bugunku ortak kod
            batch = packed_batches(args.data, cfg["batch_size"] * cfg["micro_batches"], 1, cfg["seq_len"])[0][0]
            info = real_step_ms(cfg, batch, device, K=2)
            r.update(step_ms=info["step_ms"], step_tokens=cfg["step_tokens"])
            say("%s: egitim adimi %.0f ms (%d token; shard 13 paketi, sabit batch)" % (name, info["step_ms"], cfg["step_tokens"]))
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        model = I._load_model(run, "last").to(device)
        for B in (32, 64, 128):
            prompts = [d[:P].tolist() for d in docs[:B]]
            ids = torch.tensor(prompts, device=device)
            row = dict(B=B)
            for label, ac in (("eager fp32 (generate_cached)", False), ("eager bf16 autocast", True)):
                ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if ac else contextlib.nullcontext
                with ctx(), torch.no_grad():               # D4: bu baglamda logits tipi
                    dtype = str(model.logits(ids[:1, :8]).dtype)
                    TR.generate_cached(model, prompts, 4)
                    toks, pre = timed(lambda: TR.generate_cached(model, prompts, 1))
                    toks, tot = timed(lambda: TR.generate_cached(model, prompts, G))
                row[label] = dict(logits_dtype=dtype, prefill_ms=pre, t_dec_ms=(tot - pre) / (G - 1), total_ms=tot)
                if not ac:
                    ref_tokens = torch.tensor(toks)
            try:
                dec, build = timed(lambda: GraphDecoder(model, ids, G))
                (out, _), ms = timed(lambda: dec.run(G))
                row["KIYAS graph fp32"] = dict(logits_dtype=str(dec.logits.dtype), build_ms=build, t_dec_ms=ms / (G - 1),
                                               total_ms=build + ms,
                                               same_tokens=float((out.cpu() == ref_tokens).float().mean()))
                del dec
            except Exception as e:
                row["KIYAS graph fp32"] = dict(error=repr(e)[:200])
            torch.cuda.empty_cache()
            r["rows"].append(row)
            for label in ("eager fp32 (generate_cached)", "eager bf16 autocast", "KIYAS graph fp32"):
                x = row[label]
                if "error" in x:
                    say("   B %3d %-30s HATA %s" % (B, label, x["error"]))
                    continue
                train_ms = B * (P + G) * r["step_ms"] / r["step_tokens"] if "step_ms" in r else float("nan")
                cost = x["total_ms"] + train_ms
                x.update(train_on_generated_ms=train_ms, cost_ms=cost,
                         overhead_every4=cost / (4 * r["step_ms"]) if "step_ms" in r else None,
                         overhead_every8=cost / (8 * r["step_ms"]) if "step_ms" in r else None)
                say("   B %3d %-30s logits %s | t_dec %.2f ms/adim | uretim %.0f ms (istem %.0f) + uretilende egitim %.0f ms = "
                    "%.0f ms | ek yuk her 4 adimda %.1f%%, her 8 adimda %.1f%%%s" % (
                        B, label, x["logits_dtype"], x["t_dec_ms"], x["total_ms"], x.get("prefill_ms", x.get("build_ms", 0)),
                        train_ms, cost, 100 * (x["overhead_every4"] or 0), 100 * (x["overhead_every8"] or 0),
                        " | token'lar eager fp32 ile ayni %.3f" % x["same_tokens"] if "same_tokens" in x else ""))
        del model
        torch.cuda.empty_cache()
    write(args.run, "decode", res)


# ---- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="kosu klasoru (config.json)")
    ap.add_argument("measure", nargs="?", choices=("train", "sizes", "generate", "tune", "steptrace", "decode"))
    ap.add_argument("--data", help="FineWeb koku (gpt2/shard_013 ...); yoksa rastgele token")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=3, help="train: parca parca olculen adim")
    ap.add_argument("--real", type=int, default=2, help="train_seq'in kendisi: K isinma + K olcum adimi (0: yok)")
    ap.add_argument("--only", help="sizes: aday adlari, virgulle (d1280_8x2,...)")
    ap.add_argument("--prefix", type=int, default=256, help="decode: istem boyu P")
    ap.add_argument("--extra", help="decode: ek kosu klasorleri, virgulle")
    ap.add_argument("--at", default="1000,9000,17000", help="steptrace: pencerelerin ilk adimi (kosunun veri sirasi)")
    ap.add_argument("--window", type=int, default=30, help="steptrace: pencere basina adim")
    ap.add_argument("--sample", type=int, default=300, help="steptrace: epok dagilimi icin rastgele adim (yalniz CPU)")
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
    {"train": train_measure, "sizes": sizes_measure, "generate": generate_measure, "tune": tune_measure,
     "steptrace": steptrace_measure,
     "decode": decode_measure}[args.measure](
        args, config, args.device)


if __name__ == "__main__":
    main()
