"""step_profile -- egitim adiminin ic + dis profili, her model (transformer, Model Z) ve her boy icin standart arac (belge 94;
kullanici, 10 Ekim: "core içine ekle her modele uyacak şekilde standart bir kod olsun her büyük koşuda standart bakalım").
Yalniz olcum; depo kodu degismez.

Yontem: GERCEK train.main dongusu (ayni derleme modu, ayni veri, ayni optimizer, ayni gunluk / checkpoint) bu surecte
calisir; depo fonksiyonlari CAGRI ANINDA sarmalanir (record_function etiketi + CPU sayaci).  Derlenen bloklarin icine
etiket girmez: blok sinifi ornek basina alt sinifa cevrilir, etiket __call__'da (derlenen _call_impl'in disinda).
Geri yayilim cekirdekleri ileri islemlerine autograd "Sequence number" ile baglanir.  Blok etiketi: Model Z yerel "loc",
G "glob"; transformer "tf".

Her yapilandirma (--config AD="train.py bayraklari", birden cok) ayri alt surecte (gercek kosu gibi taze surec): --steps
N + 1 --stop_step N.  Iki profil penceresi (w1: adim --prof_on, w2: --prof_off; --nprof adim): bilesen basina GPU ms (blok
blok yerel / G / tf, ileri / geri, cikis + kayip, embedding, optimizer, clip, maske, H2D), cekirdek turu (matmul /
attention / norm / swiglu / rope / ce ...), GPU bosluklari, senkronlar, CPU etiketleri.  Profil sonrasi ayni surecte izole
olcumler (senkronlu adim, art arda adim, maske, H2D, clip, AdamW, Muon / NorMuon, Newton-Schulz sekil basina, n-gram yolu
varsa).  Cikti: <out>/step_profile.json, <out>/<ad>/part.json, <out>/<ad>/<pencere>_trace.json; son satir
STEP_PROFILE_BITTI.

    python step_profile.py --config d1280="--model model_z --d 1280 --layers 24 --heads 20 --micro_batches 2"
                           [--config ...] [--data <veri>] [--local <yerel kopya>|none] [--steps 300] [--out /content/profile]
    CPU kuru kosu (test): --device cpu --dry 1 --dry_rows 4 (kucuk veri, sozde cekirdek = yaprak cpu_op)
"""
import argparse
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPY_GBS = 1462.0           # olculmus kopya bant genisligi (GB/s), roofline HESAP'i icin json'a yazilir
T0 = time.time()


def log(msg):
    print("[prof %7.1f sn] %s" % (time.time() - T0, msg), flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# isci: tek yapilandirma
# ----------------------------------------------------------------------------------------------------------------------
def worker(a):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    sys.path.insert(0, os.path.join(a.v2, "common"))
    import train as TR                                                   # noqa: E402
    import recipe as R                                                   # noqa: E402
    D = TR.D
    cuda = a.device == "cuda"
    if cuda:
        assert torch.cuda.is_available(), "GPU YOK"
        free, total = (x / 1e9 for x in torch.cuda.mem_get_info())
        need = a.min_free_gb if a.min_free_gb is not None else 0.85 * total
        log("GPU %s, bos %.1f / %.1f GB" % (torch.cuda.get_device_name(0), free, total))
        if free < need:
            sys.exit("DUR: GPU'da bos bellek %.1f GB < %.0f: baska is calisiyor olabilir (olcum tek basina olmali)" % (
                free, need))
    TR.LOG_EVERY = a.log_every
    if a.dry:
        TR.BATCH_ROWS = a.dry_rows
    cfg = a.worker
    out = os.path.join(a.out, cfg)
    run_dir = os.path.join(out, "run_%s" % time.strftime("%H%M%S"))
    os.makedirs(out, exist_ok=True)
    S = dict(prof=None, rf_iter=None, step=0, model=None, opt=None, captured={}, cpu=collections.defaultdict(list),
             entry=[], exit_=[], ev=[], traces={}, mem={})
    prof_at = {a.prof_on: "w1", a.prof_off: "w2"}
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])

    def timed(fn, label, cpu_key=None):
        def w(*args, **kw):
            t = time.perf_counter()
            with record_function("P:" + label):
                r = fn(*args, **kw)
            S["cpu"][cpu_key or label].append(1000 * (time.perf_counter() - t))
            return r
        return w

    subs = {}

    def relabel(mod, label):         # sinif basina TEK alt sinif (dynamo tip korumasi bloklar arasinda ortak kalsin);
        cls = type(mod)              # etiket __call__'da, derlenen _call_impl'in disinda
        if cls not in subs:
            subs[cls] = type("P" + cls.__name__, (cls,), {"__call__": lambda self, *x, **k: _rcall(self, cls, x, k)})
        mod.__class__ = subs[cls]
        mod._prof_label = label

    def _rcall(self, cls, x, k):
        with record_function("P:" + self._prof_label):
            return cls.__call__(self, *x, **k)

    orig_build, orig_opt, orig_step = TR._build, TR._optimizer, TR._step
    S["orig_step"] = orig_step
    orig = dict(to_device=TR._to_device, attn=TR._attn, build_batch=D.build_batch, mtp_targets=D.mtp_targets,
                out_loss=R.output_loss, out_loss_mtp=R.output_loss_mtp, save=R.Checkpoint.save,
                clip=torch.nn.utils.clip_grad_norm_, backward=torch.Tensor.backward)

    def build(args, dev):
        model, mask_fn, layout = orig_build(args, dev)
        MZ = sys.modules[type(model).__module__]                         # model_z/model.py ya da transformer/baseline.py
        for fn, lab in (("summaries_last", "summaries_last"), ("bigram_prev", "ngram_prev"), ("bigram_ids", "ngram_hash")):
            if hasattr(MZ, fn):
                setattr(MZ, fn, timed(getattr(MZ, fn), lab))
        is_z = hasattr(model, "global_layers")
        nloc = len(model.blocks) - (model.global_layers if is_z else 0)
        for l, blk in enumerate(model.blocks):
            relabel(blk, "block%02d_%s" % (l, ("glob" if l >= nloc else "loc") if is_z else "tf"))
        relabel(model.E, "E")
        relabel(model.norm, "norm")
        if getattr(model, "ngram", None) is not None:
            relabel(model.ngram, "ngram_lookup")
            model._ngram = timed(model._ngram, "ngram")
        model._batch_hidden = timed(model._batch_hidden, "fwd")
        S["model"] = model
        return model, mask_fn, layout

    def optimizer(model, kind, lr, cuda_):
        opt, info = orig_opt(model, kind, lr, cuda_)
        S["orig_opt"] = dict(zero=opt.zero_grad)
        if hasattr(opt, "adamw"):
            S["orig_opt"].update(adamw=opt.adamw.step, muon=opt.muon.step)
            opt.adamw.step = timed(opt.adamw.step, "opt_adamw")
            opt.muon.step = timed(opt.muon.step, "opt_muon")
        opt.zero_grad = timed(opt.zero_grad, "zero_grad")
        S["opt"], S["opt_info"] = opt, info
        return opt, info

    def to_device(batch, dev):
        S["captured"]["cpu_batch"] = batch
        return orig["to_device"](batch, dev)

    def step(model, batch, mask_fn, opt, cuda_, timer=None, cont=None, mtp=None, micro=1, skip=()):
        k = S["step"]
        t = time.perf_counter()
        S["entry"].append(t)
        if cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            S["ev"].append(e)
        if S["rf_iter"] is not None:                                     # onceki tur (adim + adimlar arasi) kapanir
            S["rf_iter"].__exit__(None, None, None)
            S["rf_iter"] = None
        if S["prof"] is not None and k == S["prof_end"]:
            if cuda:
                torch.cuda.synchronize()
            S["prof"].stop()
            S["traces"][S["prof_name"]] = S["prof"]
            S["prof"] = None
            if cuda:
                st = torch.cuda.memory_stats()
                S["mem"][S["prof_name"]] = {key: st.get(key) for key in (
                    "allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current",
                    "reserved_bytes.all.peak", "inactive_split_bytes.all.current", "num_alloc_retries", "num_ooms",
                    "num_device_alloc", "num_device_free", "num_sync_all_streams")}
        if k == a.prof_on - 2 and S["prof"] is None:                     # isinma oturumu (atilir): 8 Ekim GPU'da surecin
            S["warm"] = profile(activities=acts)                         # ILK oturumunda butun CUPTI olaylari ts = 0 geldi
            S["warm"].start()
        elif S.get("warm") is not None:
            if cuda:
                torch.cuda.synchronize()
            S["warm"].stop()
            S["warm"] = None
        if k in prof_at and S["prof"] is None:
            S["prof"] = profile(activities=acts, record_shapes=False, with_stack=False)
            S["prof"].start()
            S["prof_end"], S["prof_name"] = k + a.nprof, prof_at[k]
        if S["prof"] is not None:
            S["rf_iter"] = record_function("P:iter%d" % k)
            S["rf_iter"].__enter__()
        if k in (a.cap_on, a.cap_off):                                   # izole olcum icin cihaz batch'i ve MTP girdisi
            S["captured"][("on" if k == a.cap_on else "off")] = (batch, mask_fn, mtp, micro, skip)
        with record_function("P:step"):
            r = orig_step(model, batch, mask_fn, opt, cuda_, timer, cont, mtp, micro, skip)
        S["exit_"].append(time.perf_counter())
        S["cpu"]["step_issue_on" if mtp is not None else "step_issue_off"].append(1000 * (S["exit_"][-1] - t))
        S["step"] = k + 1
        return r

    TR._build, TR._optimizer, TR._step = build, optimizer, step
    TR._to_device = timed(to_device, "h2d")
    TR._attn = timed(orig["attn"], "block_mask")
    D.build_batch = timed(orig["build_batch"], "build_batch")
    D.mtp_targets = timed(orig["mtp_targets"], "mtp_targets")
    R.output_loss = timed(orig["out_loss"], "out_loss")
    R.output_loss_mtp = timed(orig["out_loss_mtp"], "out_loss_mtp")
    R.Checkpoint.save = staticmethod(timed(orig["save"], "ckpt_save"))
    torch.nn.utils.clip_grad_norm_ = timed(orig["clip"], "clip")
    torch.Tensor.backward = timed(orig["backward"], "backward")

    argv = [*a.cfgs[cfg], "--steps", str(a.steps + 1), "--stop_step",
            str(a.steps), "--checkpoint_minutes", "10", "--data", a.data, "--stream", a.stream, "--out", run_dir,
            "--device", a.device] + (["--local", a.local] if a.local else []) + a.extra.split()
    log("%s: train.main %s" % (cfg, " ".join(argv)))
    t_main = time.time()
    res = TR.main(argv)
    wall_main = time.time() - t_main
    if S["rf_iter"] is not None:
        S["rf_iter"].__exit__(None, None, None)
    if S["prof"] is not None:                                            # dongu profil bitmeden bittiyse
        S["prof"].stop()
        S["traces"][S["prof_name"]] = S["prof"]
    if cuda:
        torch.cuda.synchronize()
    # geri al (izole olcumler gercek fonksiyonlarla)
    torch.nn.utils.clip_grad_norm_, torch.Tensor.backward = orig["clip"], orig["backward"]
    R.output_loss, R.output_loss_mtp = orig["out_loss"], orig["out_loss_mtp"]
    TR._attn, TR._to_device = orig["attn"], orig["to_device"]

    rep = dict(config=cfg, argv=argv, wall_main_s=round(wall_main, 1), torch=torch.__version__,
               device=torch.cuda.get_device_name(0) if cuda else "cpu", identity=res.get("identity"),
               params=res.get("params"), optimizer=S["opt_info"]["split"])
    rep["windows"] = [{k: w.get(k) for k in ("step", "steps", "seconds", "ms_per_step", "first", "peak_gb",
                                              "peak_reserved_gb", "out_fwd_ms", "mtp_w", "loss")} for w in res["log"]]
    # adim basina GPU periyodu (giris olaylari arasi; senkronsuz) ve CPU sureleri
    if cuda:
        per = [S["ev"][i].elapsed_time(S["ev"][i + 1]) for i in range(len(S["ev"]) - 1)]
        rep["gpu_period_ms"] = [round(x, 2) for x in per]
    gaps = [1000 * (S["entry"][i + 1] - S["exit_"][i]) for i in range(len(S["exit_"]) - 1)]
    rep["cpu_between_steps_ms"] = [round(x, 2) for x in gaps]
    rep["cpu_ms"] = {k: dict(n=len(v), median=round(statistics.median(v), 3), max=round(max(v), 3),
                             total=round(sum(v), 1)) for k, v in S["cpu"].items() if v}
    rep["mem_at_profile"] = S["mem"]
    try:
        from torch._dynamo.utils import compile_times, counters
        rep["dynamo_counters"] = {k: dict(v) for k, v in counters.items()}
        rep["compile_times"] = compile_times(repr="str", aggregate=True)
    except Exception as e:                                               # noqa: BLE001
        rep["compile_times"] = "yok: %r" % e
    # izler
    rep["profiles"] = {}
    for name, p in S["traces"].items():
        path = os.path.join(out, "%s_trace.json" % name)
        p.export_chrome_trace(path)
        try:
            rep["profiles"][name] = analyze(path, dry=a.dry)
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            rep["profiles"][name] = dict(error=repr(e))
        log("%s %s: iz %.0f MB, cozumlendi" % (cfg, name, os.path.getsize(path) / 1e6))
    rep["isolated"] = guarded(lambda: isolated(S, a, TR, R, orig, cuda))
    json.dump(rep, open(os.path.join(out, "part.json"), "w"), indent=1)
    print_report(rep)


def guarded(fn):
    try:
        return fn()
    except Exception as e:                                               # noqa: BLE001
        import traceback
        traceback.print_exc()
        return dict(error=repr(e))


# ----------------------------------------------------------------------------------------------------------------------
# izole olcumler (dongu bittikten sonra, ayni surec, ayni model / optimizer)
# ----------------------------------------------------------------------------------------------------------------------
def isolated(S, a, TR, R, orig, cuda):
    import torch
    model, opt = S["model"], S["opt"]
    res = {}

    def tm(fn, n=8, warm=2):
        for _ in range(warm):
            fn()
        if not cuda:
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                ts.append(1000 * (time.perf_counter() - t))
            return dict(median=round(statistics.median(ts), 3), min=round(min(ts), 3))
        torch.cuda.synchronize()
        ts, cpu = [], []
        for _ in range(n):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            s.record()
            fn()
            e.record()
            cpu.append(1000 * (time.perf_counter() - t))
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return dict(median=round(statistics.median(ts), 3), min=round(min(ts), 3),
                    cpu_issue_median=round(statistics.median(cpu), 3))

    step_fn = S["orig_step"]                                             # train._step'in kendisi (sarmalayicisiz)
    for phase in ("on", "off"):
        if phase not in S["captured"]:
            continue
        b, mask_fn, mtp, micro, skip = S["captured"][phase]
        one = lambda: step_fn(model, b, mask_fn, opt, cuda, None, None, mtp, micro, skip)  # noqa: E731
        r = dict(sync_step=tm(one, n=6))
        if cuda:                                                         # art arda, senkronsuz (dongu gibi)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(6):
                one()
            e.record()
            torch.cuda.synchronize()
            r["back_to_back_ms"] = round(s.elapsed_time(e) / 6, 3)
        bm = TR._rows(b, 0, micro) if micro > 1 else b
        r["block_mask"] = tm(lambda: orig["attn"](bm, mask_fn, cuda))
        res["step_" + phase] = r
    if "cpu_batch" in S["captured"]:
        cb = S["captured"]["cpu_batch"]
        dev = torch.device(a.device)
        res["h2d"] = tm(lambda: orig["to_device"](cb, dev))
        nbytes = sum(getattr(cb, f).numel() * getattr(cb, f).element_size() for f in cb.__dataclass_fields__
                     if torch.is_tensor(getattr(cb, f)))
        res["h2d_bytes"] = int(nbytes)
    # gradyanlar dolu (son adim): optimizer parcalari
    ng = model.ngram.weight if getattr(model, "ngram", None) is not None else None
    if ng is not None and ng.grad is not None:
        gr = ng.grad
        res["ngram_grad"] = dict(layout=str(gr.layout), dtype=str(gr.dtype), shape=list(gr.shape),
                                 MB=round(gr.numel() * gr.element_size() / 1e6, 1), is_sparse=bool(gr.is_sparse))
    params = [p for p in model.parameters() if p.grad is not None]
    no_ng = [p for p in params if p is not ng]
    clip = orig["clip"]
    res["clip_all"] = tm(lambda: clip(params, TR.CLIP))
    res["clip_without_ngram"] = tm(lambda: clip(no_ng, TR.CLIP))
    if "adamw" in S.get("orig_opt", {}):
        oa, om = S["orig_opt"]["adamw"], S["orig_opt"]["muon"]
        res["adamw_all"] = tm(oa)
        if ng is not None:
            g_ = ng.grad
            ng.grad = None
            res["adamw_without_ngram"] = tm(oa)
            ng.grad = g_
        res["muon_all"] = tm(om)
        mu = opt.muon
        shapes = collections.defaultdict(list)
        for g in mu.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    shapes[tuple(p.shape)].append(p.grad)
            coef, steps_, eps = g["ns_coefficients"], g["ns_steps"], g["eps"]
        res["muon_groups"] = {str(k): len(v) for k, v in shapes.items()}
        stacks = [torch.stack(v) for v in shapes.values()]
        res["ns_only"] = tm(lambda: [R._ns_batched(G, coef, steps_, eps) for G in stacks])
        res["ns_per_shape"] = {str(k): tm(lambda G=G: R._ns_batched(G, coef, steps_, eps), n=4)
                               for k, G in zip(shapes, stacks)}
    if ng is not None:                                                   # n-gram yolu tek basina: ileri + geri (tabloya)
        MZ = sys.modules[type(model).__module__]
        cb = S["captured"]["on"][0] if "on" in S["captured"] else None
        if cb is not None:
            prev = MZ.bigram_prev(cb.tokens, cb.kind, cb.doc, cb.sent)
            is_tok = cb.kind == MZ.TOKEN
            gout = torch.randn(*cb.tokens.shape, ng.shape[1], device=ng.device)

            def ngram_fb():
                ng.grad = None
                with torch.autocast(cb.tokens.device.type, dtype=torch.bfloat16, enabled=cuda):
                    g = model._ngram(cb.tokens, prev, is_tok)
                    (g * gout).sum().backward()
            res["ngram_fwd_bwd_to_table"] = tm(ngram_fb, n=5)
            res["ngram_fwd_only"] = tm(lambda: model._ngram(cb.tokens, prev, is_tok).sum(), n=5)
    if cuda:
        st = torch.cuda.memory_stats()
        res["memory_end"] = {k: st.get(k) for k in ("allocated_bytes.all.peak", "reserved_bytes.all.peak",
                                                    "num_alloc_retries", "num_device_alloc", "num_device_free")}
    return res


# ----------------------------------------------------------------------------------------------------------------------
# iz cozumleme (Kineto chrome trace)
# ----------------------------------------------------------------------------------------------------------------------
DEV_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
RT_CATS = ("cuda_runtime", "cuda_driver")
CPU_CATS = ("cpu_op", "user_annotation")


def kcat(name):
    n = name.lower()
    if "flex_attention_backward" in n or "flex_bwd" in n:
        return "attn_bwd"
    if "flex_attention" in n or "flex_fwd" in n:
        return "attn_fwd"
    if any(s in n for s in ("gemm", "cutlass", "nvjet", "xmma", "cublas", "sm90_", "sm100_", "sm120_")) or \
            re.search(r"triton_tem_fused.*b?mm", n):
        return "matmul"
    if "memset" in n:
        return "memset"
    if "memcpy" in n:
        return "memcpy"
    if n.startswith("triton_"):
        if "rsqrt" in n:
            return "fused_norm(rsqrt)"
        if "sigmoid" in n:
            return "fused_gate(sigmoid)"
        if "silu" in n:
            return "fused_swiglu(silu)"
        if "cos" in n or "sin" in n:
            return "fused_rope"
        if "log_softmax" in n or "logsumexp" in n or "nll" in n or "exp" in n:
            return "fused_ce"
        return "fused_other"
    if "multi_tensor_apply" in n or "foreach" in n:
        return "foreach"
    if "fused_adam" in n or "adam" in n:
        return "fused_adam"
    if "reduce" in n:
        return "aten_reduce"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "aten_eltwise"
    if "sort" in n or "radix" in n or "scan" in n:
        return "sort/scan"
    if "index" in n or "embedding" in n or "gather" in n or "scatter" in n:
        return "index/embedding"
    return "other"


def coarse(path):
    """Etiket yolu -> kaba bilesen (tablo)."""
    bwd = path.startswith("bwd:")
    p = path[4:] if bwd else path
    ph = " bwd" if bwd else " fwd"
    m = re.match(r"fwd/block\d+_(loc|glob|tf)", p)
    if m:
        return "blocks " + m.group(1) + ph
    if p.startswith("fwd/ngram"):
        return ("ngram " + ({"fwd/ngram/ngram_hash": "hash", "fwd/ngram_prev": "prev",
                             "fwd/ngram/ngram_lookup": "lookup"}.get(p, "mask/other"))) + ph
    if p == "fwd/E":
        return "embedding E" + ph
    if p == "fwd/norm":
        return "final norm" + ph
    if p == "fwd":
        return "fwd misc (lambda-add, ZTOK where)" + ph
    if p.startswith("out_loss"):
        return p.split("/")[0] + ph
    return path


def _merge(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ((ce - cs) if cs is not None else 0.0)


def analyze(path, dry=False):
    tr = json.load(open(path))
    evs = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
    dev = [e for e in evs if e.get("cat") in DEV_CATS]
    rt = [e for e in evs if e.get("cat") in RT_CATS]
    cpu = [e for e in evs if e.get("cat") in CPU_CATS]
    # yuvalama: ayni is parcacigindaki kapsayan CPU olaylari
    by_tid = collections.defaultdict(list)
    for i, e in enumerate(cpu):
        by_tid[(e.get("pid"), e.get("tid"))].append(("c", i, e))
    for i, e in enumerate(rt):
        by_tid[(e.get("pid"), e.get("tid"))].append(("r", i, e))
    parent_c, parent_r = {}, {}
    for items in by_tid.values():
        items.sort(key=lambda x: (x[2]["ts"], -x[2].get("dur", 0), 0 if x[0] == "c" else 1))
        stack = []
        for kind, i, e in items:
            while stack and stack[-1][1]["ts"] + stack[-1][1].get("dur", 0) <= e["ts"]:
                stack.pop()
            par = stack[-1][0] if stack else None
            if kind == "c":
                parent_c[i] = par
                stack.append((i, e))
            else:
                parent_r[i] = par

    def chain(ci):
        out = []
        while ci is not None:
            out.append(ci)
            ci = parent_c.get(ci)
        return out                                                       # icten disa

    def plabels(ch):
        return [cpu[c]["name"][2:] for c in reversed(ch) if cpu[c].get("cat") == "user_annotation"
                and cpu[c]["name"].startswith("P:")]

    def fwd_path(ch):
        ls = [x for x in plabels(ch) if not x.startswith("iter") and x not in ("step", "backward")]
        ls = [x for j, x in enumerate(ls) if j == 0 or x != ls[j - 1]]  # ozyineli _attn: block_mask/block_mask
        return "/".join(ls)

    def seq_of(i):
        return (cpu[i].get("args") or {}).get("Sequence number")

    # ileri sira numarasi -> dugumu kuran ileri olay.  Profil, grad acikken HER islemin basinda sayacin o anki degerini
    # yazar: N degerli islemler, N. dugumu kuran isleme kadar olanlardir -> N'li SON islem, ayni N'li en dis atasina kadar.
    in_bwd = {}

    def under_bwd(i):
        if i not in in_bwd:
            p = parent_c.get(i)
            in_bwd[i] = cpu[i]["name"].startswith("autograd::engine") or (p is not None and under_bwd(p))
        return in_bwd[i]

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    last_seq = {}
    for i, e in enumerate(cpu):
        sq = seq_of(i)
        if sq is None or e.get("cat") != "cpu_op" or "Backward" in e["name"] or under_bwd(i):
            continue
        if sq not in last_seq or e["ts"] >= cpu[last_seq[sq]]["ts"]:
            last_seq[sq] = i
    seq_fwd = {}
    for sq, i in last_seq.items():
        while parent_c.get(i) is not None and seq_of(parent_c[i]) == sq:
            i = parent_c[i]
        seq_fwd[sq] = i
    iters = sorted([(e["ts"], e["ts"] + e["dur"], int(e["name"][6:])) for e in cpu
                    if e.get("cat") == "user_annotation" and e["name"].startswith("P:iter")])

    def iter_of(ts):
        for s, e_, k in iters:
            if s <= ts < e_:
                return k
        return None

    cache = {}

    def label_of(ci):
        if ci in cache:
            return cache[ci]
        ch = chain(ci)
        fp = fwd_path(ch)
        lab = None
        for c in ch:                                                     # geri: en icteki sira numarali geri olay
            e = cpu[c]
            sq = (e.get("args") or {}).get("Sequence number")
            if sq is not None and (e["name"].startswith("autograd::engine") or "Backward" in e["name"]):
                f = seq_fwd.get(sq)
                if f is not None:
                    lab = "bwd:" + (fwd_path(chain(f)) or "?" + cpu[f]["name"])
                else:
                    lab = "bwd:?" + e["name"].replace("autograd::engine::evaluate_function: ", "")
                break
        if lab is None or (fp and not fp.startswith("fwd") and not lab.startswith("bwd:fwd")):
            if fp:
                lab = fp
            elif lab is None:
                names = [cpu[c]["name"] for c in ch if cpu[c].get("cat") == "cpu_op"]
                lab = "loop:" + (names[0] if names else "?")
        cache[ci] = lab
        return lab

    corr_rt = {}
    for i, e in enumerate(rt):
        c = (e.get("args") or {}).get("correlation")
        if c is not None:
            corr_rt[c] = i
    items = []                                                          # (ts, end, name, label, iter)
    if dry or not dev:                                                  # CPU kuru kosu: yaprak cpu_op'lar sozde cekirdek
        has_child = set(v for v in parent_c.values() if v is not None)
        for i, e in enumerate(cpu):
            if e.get("cat") == "cpu_op" and i not in has_child:
                items.append((e["ts"], e["ts"] + e["dur"], e["name"], label_of(i), iter_of(e["ts"])))
        unmatched = 0
    else:
        unmatched = 0
        for e in dev:
            c = (e.get("args") or {}).get("correlation")
            r = corr_rt.get(c)
            if r is None:
                unmatched += 1
                items.append((e["ts"], e["ts"] + e["dur"], e["name"], "unmatched", None))
                continue
            par = parent_r.get(r)
            lab = label_of(par) if par is not None else "loop:?"
            items.append((e["ts"], e["ts"] + e["dur"], e["name"], lab, iter_of(rt[r]["ts"])))
    zero_ts = sum(1 for e in dev if e.get("ts", 0) == 0)
    if dev and zero_ts == len(dev):                                     # CUPTI zaman damgasi yok (ilk oturum, 8 Ekim)
        return dict(error="CUPTI zaman damgalari 0 (%d GPU olayi): iz cozumlenemez; isinma oturumu gerekir" % zero_ts,
                    n_device_events=len(dev), kernel_names=collections.Counter(e["name"][:100] for e in dev).most_common(30))
    items.sort()
    ks = sorted(set(k for *_, k in items if k is not None))
    nit = max(len(ks), 1)
    # GPU periyodu: iterasyonun ilk cekirdegi -> sonrakinin ilk cekirdegi (son: kendi son cekirdegi)
    first = {}
    for ts, en, nm, lab, k in items:
        if k is not None and k not in first:
            first[k] = ts
    period = {}
    for j, k in enumerate(ks):
        end = first[ks[j + 1]] if j + 1 < len(ks) else max(en for ts, en, nm, lab, kk in items if kk == k)
        period[k] = (first[k], end)
    agg = collections.defaultdict(lambda: [0.0, 0, 0.0])                # label -> [ms, n, idle_before_ms]
    lab_cat = collections.defaultdict(lambda: [0.0, 0])                 # (label, kcat) -> [ms, n]
    kern = collections.defaultdict(lambda: [0.0, 0, collections.Counter()])
    block_cat = collections.defaultdict(lambda: [0.0, 0])
    prev_end = None
    busy, per_iter = [], {}
    seq_dump = collections.defaultdict(list)
    for ts, en, nm, lab, k in items:
        if k is None:
            continue
        d = (en - ts) / 1000
        gap = max(0.0, (ts - prev_end) / 1000) if prev_end is not None else 0.0
        prev_end = en if prev_end is None else max(prev_end, en)
        agg[lab][0] += d
        agg[lab][1] += 1
        agg[lab][2] += gap
        lab_cat[(lab, kcat(nm))][0] += d
        lab_cat[(lab, kcat(nm))][1] += 1
        kern[nm][0] += d
        kern[nm][1] += 1
        kern[nm][2][lab] += 1
        m = re.match(r"(bwd:)?fwd/block(\d+)_(loc|glob|tf)$", lab)
        if m:
            block_cat[(m.group(3), "bwd" if m.group(1) else "fwd", kcat(nm))][0] += d
            block_cat[(m.group(3), "bwd" if m.group(1) else "fwd", kcat(nm))][1] += 1
            if k == ks[0] and (m.group(2) in ("00",) or lab.endswith("glob")) and len(seq_dump[lab]) < 80:
                seq_dump[lab].append((nm[:90], round(d * 1000, 1)))
        per_iter.setdefault(k, []).append((ts, en))
    for k in ks:
        busy.append(_merge(per_iter[k]) / 1000)
    periods = [(period[k][1] - period[k][0]) / 1000 for k in ks]
    # yalniz ilk glob ve ilk loc blogunun dokumu
    keep = {}
    for lab in sorted(seq_dump):
        kind = lab.rsplit("_", 1)[-1]
        key = (kind, lab.startswith("bwd:"))
        if key not in keep:
            keep[key] = lab
    seqs = {keep[key]: seq_dump[keep[key]] for key in keep}
    comp = collections.defaultdict(lambda: [0.0, 0, 0.0])
    for lab, (ms, n, idle) in agg.items():
        c = coarse(lab)
        comp[c][0] += ms
        comp[c][1] += n
        comp[c][2] += idle
    # senkron noktalari ve tikanan baslatmalar
    syncs = []
    blocked = collections.defaultdict(float)
    for i, e in enumerate(rt):
        nm = e["name"]
        k = iter_of(e["ts"])
        par = parent_r.get(i)
        if "ynchronize" in nm or nm in ("cudaMemcpy", "cuMemcpyDtoH_v2", "cudaStreamWaitEvent_sync"):
            syncs.append(dict(iter=k, name=nm, ms=round(e["dur"] / 1000, 3), label=label_of(par) if par is not None
                              else "?"))
        if "aunch" in nm and e.get("dur", 0) > 200:
            blocked[k] += e["dur"] / 1000
    cpu_top = collections.defaultdict(float)                             # CPU: en dis P etiketi (iter/step disi)
    for i, e in enumerate(cpu):
        if e.get("cat") != "user_annotation" or not e["name"].startswith("P:") or e["name"].startswith("P:iter"):
            continue
        anc = [cpu[c]["name"] for c in chain(i)[1:] if cpu[c].get("cat") == "user_annotation"
               and cpu[c]["name"].startswith("P:") and not cpu[c]["name"].startswith("P:iter")]
        if not anc or anc == ["P:step"]:
            cpu_top[e["name"][2:]] += e["dur"] / 1000
    iter_cpu = [(e_ - s) / 1000 for s, e_, k in iters]
    return dict(
        iters=ks, n_iters=len(ks), unmatched_device_events=unmatched,
        gpu_period_ms=[round(x, 3) for x in periods], gpu_busy_ms=[round(x, 3) for x in busy],
        gpu_idle_ms=[round(p - b, 3) for p, b in zip(periods, busy)], cpu_iter_ms=[round(x, 3) for x in iter_cpu],
        labels={lab: dict(ms=round(v[0] / nit, 4), n=round(v[1] / nit, 1), idle_before_ms=round(v[2] / nit, 4))
                for lab, v in sorted(agg.items(), key=lambda kv: -kv[1][0])},
        components={c: dict(ms=round(v[0] / nit, 4), n=round(v[1] / nit, 1), idle_before_ms=round(v[2] / nit, 4))
                    for c, v in sorted(comp.items(), key=lambda kv: -kv[1][0])},
        block_categories={"%s %s %s" % kc: dict(ms=round(v[0] / nit, 4), n=round(v[1] / nit, 1))
                          for kc, v in sorted(block_cat.items())},
        top_kernels=[dict(name=nm[:120], ms=round(v[0] / nit, 4), n=round(v[1] / nit, 1), cat=kcat(nm),
                          labels=dict(v[2].most_common(3))) for nm, v in sorted(kern.items(), key=lambda kv: -kv[1][0])[:40]],
        label_categories={"%s | %s" % lc: dict(ms=round(v[0] / nit, 4), n=round(v[1] / nit, 1))
                          for lc, v in sorted(lab_cat.items(), key=lambda kv: -kv[1][0]) if v[0] / nit >= 0.05
                          and not re.match(r"(bwd:)?fwd/block", lc[0])},
        block_sequences=seqs, syncs=syncs, blocked_launch_ms={str(k): round(v, 2) for k, v in blocked.items()},
        cpu_top_labels_ms={k: round(v / nit, 3) for k, v in sorted(cpu_top.items(), key=lambda kv: -kv[1])})


# ----------------------------------------------------------------------------------------------------------------------
# yazdirma
# ----------------------------------------------------------------------------------------------------------------------
def print_report(rep):
    cfg = rep["config"]
    print("\n=== %s: %s, %s parametre ===" % (cfg, rep.get("device"), rep.get("params")), flush=True)
    for w in rep["windows"]:
        print("  pencere -> adim %4d (%3d adim) %8.1f ms/adim  tepe %s GB (ayrilan %s)  mtp_w %s%s" % (
            w["step"], w["steps"], w["ms_per_step"], w["peak_gb"], w["peak_reserved_gb"], w["mtp_w"],
            "  (ilk: derleme)" if w["first"] else ""), flush=True)
    print("  CPU (ms, ortanca / en cok / n): " + ", ".join("%s %.2f / %.1f / %d" % (k, v["median"], v["max"], v["n"])
                                                        for k, v in rep["cpu_ms"].items()), flush=True)
    for name, p in rep["profiles"].items():
        if "error" in p:
            print("  %s: COZUMLEME HATASI %s" % (name, p["error"]), flush=True)
            continue
        per = statistics.mean(p["gpu_period_ms"]) if p["gpu_period_ms"] else 0
        print("\n  --- %s (iterasyon %s): GPU periyodu %s ms, mesgul %s, bos %s; CPU tur %s; eslesmeyen %d ---" % (
            name, p["iters"], p["gpu_period_ms"], p["gpu_busy_ms"], p["gpu_idle_ms"], p["cpu_iter_ms"],
            p["unmatched_device_events"]), flush=True)
        print("  %-48s %9s %6s %7s %9s" % ("bilesen", "ms/adim", "%", "cekirdek", "bos once"), flush=True)
        tot = 0.0
        for c, v in p["components"].items():
            tot += v["ms"] + v["idle_before_ms"]
            print("  %-48s %9.3f %6.1f %7.0f %9.3f" % (c[:48], v["ms"], 100 * v["ms"] / per if per else 0, v["n"],
                                                      v["idle_before_ms"]), flush=True)
        print("  toplam (cekirdek + bosluk) %.3f ms | periyot ortalamasi %.3f" % (tot, per), flush=True)
        print("  blok kategorileri (ms/adim, cekirdek):", flush=True)
        for k, v in p["block_categories"].items():
            print("    %-40s %9.3f %6.0f" % (k, v["ms"], v["n"]), flush=True)
        print("  etiket x cekirdek turu (blok disi, >= 0,05 ms):", flush=True)
        for k, v in list(p["label_categories"].items())[:40]:
            print("    %-60s %9.3f %6.0f" % (k[:60], v["ms"], v["n"]), flush=True)
        print("  en buyuk 25 cekirdek:", flush=True)
        for t in p["top_kernels"][:25]:
            print("    %9.3f ms %5.0f x  %-16s %s  %s" % (t["ms"], t["n"], t["cat"], t["name"][:80],
                                                       list(t["labels"])[:2]), flush=True)
        for lab, seq in p["block_sequences"].items():
            print("  dizi %s: %s" % (lab, " | ".join("%s %.0f" % (n[:40], u) for n, u in seq)), flush=True)
        print("  senkronlar: %s" % p["syncs"][:20], flush=True)
        print("  tikanan baslatma (ms / tur): %s" % p["blocked_launch_ms"], flush=True)
        print("  CPU en dis etiketler (ms / tur): %s" % p["cpu_top_labels_ms"], flush=True)
    if "isolated" in rep:
        print("\n  izole: %s" % json.dumps(rep["isolated"])[:4000], flush=True)
    print("  derleme sureleri:\n%s" % str(rep.get("compile_times"))[:3000], flush=True)


# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v2", default=V2, help="deneme2/v2 (varsayilan: bu betigin deposu)")
    ap.add_argument("--config", action="append", default=[], help='AD="train.py bayraklari" (--model dahil); birden cok')
    ap.add_argument("--min_free_gb", type=float, default=None, help="GPU tek basina mi (varsayilan toplam bellegin %%85'i)")
    ap.add_argument("--data", default="/content/drive/MyDrive/v2/fineweb_edu_s000")
    ap.add_argument("--stream", default=None, help="varsayilan --data")
    ap.add_argument("--local", default="/content/v2fw_cache")
    ap.add_argument("--out", default="/content/profile")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--prof_on", type=int, default=97)
    ap.add_argument("--prof_off", type=int, default=227)
    ap.add_argument("--nprof", type=int, default=4)
    ap.add_argument("--cap_on", type=int, default=150)
    ap.add_argument("--cap_off", type=int, default=290)
    ap.add_argument("--budget_s", type=float, default=840)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", default="", help="train.py'ye ek bayraklar (bos: varsayilanlar)")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--dry", type=int, default=0, help="CPU kuru kosu (kucuk veri)")
    ap.add_argument("--dry_rows", type=int, default=4)
    a = ap.parse_args(argv)
    assert a.config, 'en az bir --config AD="train.py bayraklari" gerek'
    a.cfgs = {}
    for c in a.config:
        name, _, flags = c.partition("=")
        assert name and "--model" in flags.split() and name not in a.cfgs, '--config AD="--model ...": %r' % c
        a.cfgs[name] = flags.split()
    a.stream = a.stream or a.data
    if a.local == "none":
        a.local = None
    if a.worker:
        return worker(a)
    os.makedirs(a.out, exist_ok=True)
    head = subprocess.run(["git", "-C", a.v2, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout[:90]
    log("depo: %s" % head.strip())
    log("step_profile: %s, adim %d, profil %d / %d (+%d), butce %.0f sn" % (",".join(a.cfgs), a.steps, a.prof_on,
                                                                            a.prof_off, a.nprof, a.budget_s))
    parts, times = {}, {}
    for cfg in a.cfgs:
        el = time.time() - T0
        if parts and el > a.budget_s - 330:
            log("BUTCE: %s atlandi (gecen %.0f sn)" % (cfg, el))
            parts[cfg] = dict(skipped="budget")
            continue
        t = time.time()
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", cfg] + list(argv if argv is not None else sys.argv[1:])
        rc = subprocess.call(cmd)
        times[cfg] = round(time.time() - t, 1)
        pj = os.path.join(a.out, cfg, "part.json")
        parts[cfg] = json.load(open(pj)) if rc == 0 and os.path.exists(pj) else dict(error="alt surec %d" % rc)
        log("%s bitti: %.1f sn, cikis %d" % (cfg, times[cfg], rc))
    res = dict(script="step_profile.py", repo_head=head.strip(), copy_gbs=COPY_GBS, configs=parts, seconds=times,
               total_s=round(time.time() - T0, 1))
    json.dump(res, open(os.path.join(a.out, "step_profile.json"), "w"), indent=1)
    log("yazildi: %s" % os.path.join(a.out, "step_profile.json"))
    print("STEP_PROFILE_BITTI", flush=True)
    return res


if __name__ == "__main__":
    main()
