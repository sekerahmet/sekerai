# -*- coding: utf-8 -*-
"""bench_speed -- hizlandirma seceneklerinin Colab GPU olcumu: adim suresi (ms) ve sayisal sapma.  Egitim BASLATMAZ:
her secenek sabit ve kucuk adim sayisi kosar, Drive'a yazmaz.  Kod yolu gercek egitiminki (train_20.train_seq).

    import bench_speed as BS
    BS.environment()                  torch / GPU / SDPA'nin sectigi yol (3 ve 4 boyut, fp32 ve bf16)
    BS.compile_repro()                eski kayip + eski compile ayari, akrabalik verisi (Inductor AssertionError'i yakalar)
    BS.tinystories(names, data=None)  TinyStories ayari (D=384, sozluk 8.004, batch 64 x 512); data=None: sentetik batch
                                      (ayni sekil, ~%37 gercek token); data=DATA: gercek batch'ler
    BS.math(names)                    train_math ayari (sozluk 13, D=64, batch 256)

Her secenek: ortak baslangic (PREP adim fp32, W_context / W_fact_out sifir olmasin), sonra secenekle WARM + TIMED adim;
sure son TIMED adimin ortalamasi.  Sapma 'base'e gore: (1) ileri hesap, ayni agirlik: logit farki, KL, top-1 ayni;
(2) egri: EVERY adimda bir nll farki, son agirliklarin uzakligi / guncellemenin boyu.  *_repeat: ayni ayar ikinci kez;
0 ise deterministik.
"""
import copy
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "train_tinystories"), os.path.join(ROOT, "train_math"),
           os.path.join(ROOT, "train_kinship")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_20 as M  # noqa: E402
import train_20 as TR  # noqa: E402

DEVICE = "cuda"          # testler CPU'da kosturur (tests_speed)
TINYSTORIES = dict(n=8004, batch=64, seq_len=512, model_kw=dict(d=384, turns=2, units=1536, t_max=512),
                   prep=20, warm=5, timed=20, every=5, eval_rows=16)
MATH = dict(n=13, batch=256, model_kw=dict(d=64, turns=2, units=256, t_max=512), prep=50, warm=50, timed=500, every=50,
            eval_rows=256)

OPTIONS = {                                   # ad -> train_seq ayarlari (adlar ONERI)
    "base": dict(),
    "base_repeat": dict(),
    "old_loss": dict(),                       # onceki kayip (boolean indeks): masked_nll'in eager'daki maliyeti
    "tf32": dict(precision="tf32"),
    "bf16": dict(precision="bf16"),
    "fused_attention": dict(attention_kernel="fused"),
    "fused_attention_repeat": dict(attention_kernel="fused"),
    "tf32_fused_attention": dict(precision="tf32", attention_kernel="fused"),
    "bf16_fused_attention": dict(precision="bf16", attention_kernel="fused"),
    "fused_adam": dict(fused_adam=True),
    "compile": dict(compile=True),
    "compile_repeat": dict(compile=True),
    "compile_tf32": dict(compile=True, precision="tf32"),
    "compile_bf16": dict(compile=True, precision="bf16"),
    "compile_bf16_fused_attention": dict(compile=True, precision="bf16", attention_kernel="fused"),
    "compile_cudagraphs": dict(compile="reduce-overhead"),
    "compile_cudagraphs_fused_adam": dict(compile="reduce-overhead", fused_adam=True),
}
TINYSTORIES_NAMES = [k for k in OPTIONS if not k.startswith("compile_cudagraphs")]
MATH_NAMES = ["base", "base_repeat", "old_loss", "fused_adam", "compile", "compile_cudagraphs", "compile_cudagraphs_fused_adam"]
BACKENDS = {0: "math", 1: "flash", 2: "efficient", 3: "cudnn", 4: "overrideable"}


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def environment():
    """Surumler ve SDPA'nin bizim sekillerde sectigi cekirdek (sayi degil, yol)."""
    import triton
    print("torch %s  cuda %s  cudnn %s  triton %s  GPU %s" % (torch.__version__, torch.version.cuda,
          torch.backends.cudnn.version(), triton.__version__, torch.cuda.get_device_name(0)))
    for d in (384, 64):
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(64, 512, d, device="cuda", dtype=dtype, requires_grad=True)
            three = torch._fused_sdp_choice(q, q, q, is_causal=True, scale=10.8)
            four = torch._fused_sdp_choice(q[:, None], q[:, None], q[:, None], is_causal=True, scale=10.8)
            print("  SDPA d=%d %s: 3 boyut -> %s, 4 boyut -> %s" % (d, str(dtype)[6:], BACKENDS.get(three, three),
                                                                   BACKENDS.get(four, four)))


def _old_nll(logits, targets, valid):
    return F.cross_entropy(logits[valid], targets[valid])


def _old_attention_forward(self, x, points=None):
    """CausalAttention.forward'in onceki hali (3 boyutlu SDPA), repro icin birebir."""
    q, k = self.queries_keys(x)
    return F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=self.scale)


def compile_repro():
    """Akrabalik verisinde eski kod (boolean indeks, 3 boyutlu SDPA) + eski ayar torch.compile(model.loss): hata varsa
    izini basar.  Ikinci veri ayni surecte (otomatik dinamik sekil); son satir yeni kayip + yeni ayar."""
    import data_20
    import exam_kinship as EK
    cases = [("eski kod, eski ayar, veri 2.560x16", dict(), True),
             ("eski kod, eski ayar, veri 2.240x28 (ayni surecte ikinci)", dict(step_answers=True), True),
             ("yeni kod, yeni ayar (fullgraph, dynamic=False), 2.240x28", dict(step_answers=True), False)]
    torch._dynamo.reset()
    for label, kw, old in cases:
        d = data_20.build(**kw)
        ids, mask = (t.to(DEVICE) for t in EK.sequences(d))
        model = M.BlockModel(len(d["vocab"]), seed=0).to(DEVICE)
        original = M.masked_nll, M.CausalAttention.forward
        if old:
            M.masked_nll, M.CausalAttention.forward = _old_nll, _old_attention_forward
        t = time.time()
        try:
            fn = torch.compile(model.loss) if old else torch.compile(model.loss, fullgraph=True, dynamic=False)
            with torch.autocast(DEVICE, enabled=False) if old else TR.numerics(device=DEVICE, autocast=True):
                total, nll = fn(ids, mask)
            total.backward()
            ref = model.loss(ids, mask)[1].item()
            print("  %-60s GECTI  %.0f sn  nll %.6f (eager %.6f)" % (label, time.time() - t, nll.item(), ref), flush=True)
        except Exception as e:
            print("  %-60s HATA %s  %.0f sn\n%s" % (label, type(e).__name__, time.time() - t,
                                                    "\n".join(traceback.format_exc().splitlines()[-25:])), flush=True)
        finally:
            M.masked_nll, M.CausalAttention.forward = original
    torch._dynamo.reset()


def synthetic_batches(n, batch, seq_len, mean_len=190):
    """TinyStories sekli: <eos> hikaye <eos> <dolgu>...; uzunluk esit dagilim, ortalama mean_len (~%37 gercek)."""
    def batch_fn(step):
        g = torch.Generator().manual_seed(100003 + step)
        lengths = torch.randint(20, min(2 * mean_len - 20, seq_len - 2), (batch,), generator=g)
        ids = torch.randint(4, n, (batch, seq_len), generator=g)
        mask = torch.arange(seq_len)[None, :] < (lengths + 2)[:, None]
        ids[:, 0] = 1
        ids[torch.arange(batch), lengths + 1] = 1
        ids[~mask] = 0
        return ids, mask
    return batch_fn


def _prepare(cfg, batches):
    """Ortak baslangic: PREP adim fp32 -> {step, model, optimizer} (surdurme paketi bicimi)."""
    pack = {}

    def save(step, model, opt):
        pack.update(step=step, model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()))
    TR.train_seq("shared", None, None, cfg["n"], steps=cfg["prep"], device=DEVICE, log_at=(), batches=batches,
                 model_kw=cfg["model_kw"], save_every=cfg["prep"], save=save)
    return pack


def _model(cfg, pack):
    model = M.BlockModel(cfg["n"], seed=0, **cfg["model_kw"]).to(DEVICE)
    model.load_state_dict(pack["model"])
    return model


@torch.no_grad()
def _forward_deviation(cfg, pack, options, ids, mask, base_logits):
    model = _model(cfg, pack)
    fn = torch.compile(model.logits, fullgraph=True, dynamic=False) if options.get("compile") else model.logits
    with TR.numerics(options.get("precision", "fp32"), options.get("attention_kernel", "math"), DEVICE, autocast=True):
        out = fn(ids[:, :-1])[mask[:, 1:]].double()
    if base_logits is None:
        return out, None
    lp, lq = base_logits.log_softmax(-1), out.log_softmax(-1)
    diff = (out - base_logits).abs()
    return out, dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(),
                     kl_mean=(lp.exp() * (lp - lq)).sum(-1).mean().item(),
                     top1_same=(out.argmax(-1) == base_logits.argmax(-1)).double().mean().item())


def _run(cfg, pack, batches, options, base):
    marks = {}

    def callback(step, model, nll):
        _sync()
        marks[step] = (time.perf_counter(), nll)
    torch._dynamo.reset()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    steps = cfg["prep"] + cfg["warm"] + cfg["timed"]
    model, _ = TR.train_seq("shared", None, None, cfg["n"], steps=steps, device=DEVICE, every=cfg["every"],
                            callback=callback, log_at=(), batches=batches, model_kw=cfg["model_kw"],
                            checkpoint=copy.deepcopy(pack), **options)
    t0, t1 = marks[cfg["prep"] + cfg["warm"]][0], marks[steps][0]
    r = dict(ms=1000 * (t1 - t0) / cfg["timed"], nll={s: v[1] for s, v in sorted(marks.items())},
             peak_gb=torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else float("nan"),
             params=[p.detach().double().clone() for p in model.parameters()])
    if base is not None:
        start = [p.detach().double() for p in _model(cfg, pack).parameters()]
        update = sum(((b - s) ** 2).sum() for b, s in zip(base["params"], start)).sqrt().item()
        r["param_dist"] = sum(((a - b) ** 2).sum() for a, b in zip(r["params"], base["params"])).sqrt().item() / update
        r["dnll"] = [abs(r["nll"][s] - base["nll"][s]) for s in r["nll"]]
    return r


def _bench(title, cfg, names, batches, eval_batch, forward_pack=None):
    """forward_pack: ileri sapma bu agirlikla (egitilmis kosunun paketi); yoksa ortak baslangicla."""
    print("== %s: %s" % (title, ", ".join(names)), flush=True)
    pack = _prepare(cfg, batches)
    forward_pack = forward_pack or pack
    ids, mask = (t[:cfg["eval_rows"]].to(DEVICE) for t in eval_batch)
    base_logits, _ = _forward_deviation(cfg, forward_pack, {}, ids, mask, None)
    results, base = {}, None
    for name in names:
        options = OPTIONS[name]
        original = M.masked_nll
        if name == "old_loss":
            M.masked_nll = _old_nll
        try:
            r = _run(cfg, pack, batches, options, base)
            _, r["forward"] = _forward_deviation(cfg, forward_pack, options, ids, mask, base_logits)
        except Exception as e:
            print("  %-30s HATA %s: %s\n%s" % (name, type(e).__name__, str(e).splitlines()[0][:150] if str(e) else "",
                                                "\n".join(traceback.format_exc().splitlines()[-12:])), flush=True)
            continue
        finally:
            M.masked_nll = original
        if name == "base":
            base = r
        results[name] = r
        fw, nan = r["forward"], float("nan")
        print("  %-30s %7.2f ms/adim  x%.2f  tepe %.2f GB | ileri max|dlogit| %.1e KL %.1e top1 %.4f | egri max|dnll| %.1e"
              "  son %.1e  agirlik uzakligi/guncelleme %.1e" % (
                  name, r["ms"], base["ms"] / r["ms"] if base else 1.0, r["peak_gb"], fw["max_abs"], fw["kl_mean"],
                  fw["top1_same"], max(r.get("dnll", [nan])), r.get("dnll", [nan])[-1], r.get("param_dist", nan)),
              flush=True)
    summary = {k: dict(ms=round(v["ms"], 3), peak_gb=round(v["peak_gb"], 2), forward=v["forward"],
                       dnll_max=max(v["dnll"]) if "dnll" in v else None, param_dist=v.get("param_dist"))
               for k, v in results.items()}
    print("JSON %s %s" % (title, json.dumps(summary)), flush=True)
    return summary


def tinystories(names=None, data=None, cfg=None, checkpoint=None):
    """data: HAZIRLIK'in DATA'si (gercek batch'ler, data_tinystories.batches) ya da None (sentetik).  checkpoint: kosunun
    surdurme paketinin yolu (Drive, yalniz okunur); ileri sapma o egitilmis agirlikla olculur."""
    cfg = cfg or TINYSTORIES
    forward_pack = torch.load(checkpoint, map_location=DEVICE) if checkpoint else None
    if data is not None:
        import data_tinystories as DT
        import exam_tinystories as ET
        batches = DT.batches(data, cfg["batch"], 0)
        eval_batch = DT.sequences(data, "valid", ET.exam_rows(data)[:cfg["eval_rows"]])
        title = "TinyStories GERCEK batch"
    else:
        batches = synthetic_batches(cfg["n"], cfg["batch"], cfg["seq_len"])
        eval_batch, title = batches(10 ** 6), "TinyStories SENTETIK batch"
    return _bench(title, cfg, names or TINYSTORIES_NAMES, batches, eval_batch, forward_pack)


def math(names=None, cfg=None):
    """train_math ayari, veri koddan (data_math 'dur', answer_only); olcum batch'i egitimde olmayan bir adimin parcasi."""
    import data_math as DM
    cfg = cfg or MATH
    batches = DM.batches(DM.build("dur"), cfg["batch"], 0, True)
    return _bench("math (dur)", cfg, names or MATH_NAMES, batches, batches(10 ** 5))
