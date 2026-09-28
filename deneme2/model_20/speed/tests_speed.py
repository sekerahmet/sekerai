# -*- coding: utf-8 -*-
"""tests_speed -- hizlandirma onerilerinin CPU kapilari: hangisi bit duzeyinde ayni, hangisi degil.  Akrabalik verisi
(train_kinship, iz 7ae5615623aa: Inductor'un Colab'da AssertionError verdigi veri).  CPU, saniyeler.

    python tests_speed.py
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train_kinship"))
sys.path.insert(0, HERE)
torch.set_num_threads(1)   # CPU cok iplikte P[ids] geri yayilimi (index_put_ accumulate) deterministik degil

import data_20 as D  # noqa: E402
import exam_kinship as EK  # noqa: E402
import model_20 as M  # noqa: E402
import model_20_transformer as MT  # noqa: E402
import train_20 as TR  # noqa: E402
from numerics_emulation import bf16_emulation, cuda_bf16_autocast, round_tf32, tf32_emulation  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

RESULTS = []


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-80s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def old_nll(logits, targets, valid):
    """Onceki kayip (boolean indeks) -- kiyas icin."""
    return F.cross_entropy(logits[valid], targets[valid])


def kinship():
    d = D.build(step_answers=True)
    assert d["fingerprint"] == "7ae5615623aa", d["fingerprint"]
    ids, mask = EK.sequences(d)
    return ids, mask, len(d["vocab"])


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def grads(model, loss):
    model.zero_grad()
    loss.backward()
    return [p.grad.clone() for p in model.parameters() if p.requires_grad]


def t_masked_nll(ids, mask, n):
    """masked_nll: gradyan bit duzeyinde ayni, deger son bitte; uc model."""
    print("masked_nll (boolean indeks yerine agirlikli ortalama)")
    for name, model in (("BlockModel (Model X)", M.BlockModel(n, seed=0)),
                        ("SequenceModel", M.SequenceModel(n, seed=0)),
                        ("TransformerModel", MT.TransformerModel(n, seed=0))):
        with torch.no_grad():                               # sifir baslayan matrisler: gradyan her yola aksin
            for k, p in model.named_parameters():
                if k.endswith(("W_context", "W_fact_out")):
                    p.normal_(0, 0.05, generator=torch.Generator().manual_seed(7))
        logits = model.logits(ids[:, :-1])
        a, b = M.masked_nll(logits, ids[:, 1:], mask[:, 1:]), old_nll(logits, ids[:, 1:], mask[:, 1:])
        ga, gb = grads(model, a), grads(model, model_old_loss(model, ids, mask))
        check("%s: deger |fark| <= 2e-6 (goreli)" % name, abs(a.item() - b.item()) <= 2e-6 * abs(b.item()),
              "fark %.1e" % abs(a.item() - b.item()))
        check("%s: butun gradyanlar BIT DUZEYINDE ayni" % name, same(ga, gb))


def model_old_loss(model, ids, mask):
    return old_nll(model.logits(ids[:, :-1]), ids[:, 1:], mask[:, 1:])


def t_trajectory(ids, mask, n):
    """12 adim egitim (Adam, cosine, clip): yeni kayipla agirliklar eskisiyle bit duzeyinde ayni."""
    print("egitim yolu: masked_nll ile boolean indeks, 12 adim")
    new, _ = TR.train_seq("shared", ids, mask, n, steps=12, log_at=())
    original = M.masked_nll
    M.masked_nll = old_nll
    try:
        old, _ = TR.train_seq("shared", ids, mask, n, steps=12, log_at=())
    finally:
        M.masked_nll = original
    a, b = new.state_dict(), old.state_dict()
    check("Model X 12 adim: butun agirliklar BIT DUZEYINDE ayni", all(torch.equal(a[k], b[k]) for k in a))


def t_attention_4d(ids, n):
    """CausalAttention 4 boyutlu SDPA: 3 boyutluyla ayni (math yolunda ve CPU varsayilaninda)."""
    print("SDPA (B, 1, T, d) ile (B, T, d)")
    att = M.CausalAttention(64, rope=True)
    x = M.TokenPoints(n).points()[ids[:64]].detach().requires_grad_(True)
    q, k = att.queries_keys(x)
    for label, backends in (("math", [SDPBackend.MATH]), ("CPU varsayilan (flash)", None)):
        ctx = sdpa_kernel(backends) if backends else torch.autocast("cpu", enabled=False)
        with ctx:
            three = F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=att.scale)
            four = att(x)
        g3 = torch.autograd.grad(three.sum(), x, retain_graph=True)[0]
        g4 = torch.autograd.grad(four.sum(), x, retain_graph=True)[0]
        check("%s: cikis ve gradyan BIT DUZEYINDE ayni" % label, torch.equal(three, four) and torch.equal(g3, g4))
    with sdpa_kernel([SDPBackend.MATH]):
        m = F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=att.scale)
    fl = F.scaled_dot_product_attention(q, k, x, is_causal=True, scale=att.scale)
    check("bilgi: CPU flash ile math farkli yol (fark kucuk)", (m - fl).abs().max() < 1e-5,
          "max |fark| %.1e" % (m - fl).abs().max())


def t_compile(ids, mask, n):
    """torch.compile (aot_eager, derleyicisiz): yeni kayip TEK grafik; eski kayip fullgraph'ta kirilir."""
    print("torch.compile (backend aot_eager; bu makinede C++ derleyicisi yok, Inductor kostrulamaz)")
    model = M.BlockModel(n, seed=0)
    eager_total, eager_nll = model.loss(ids, mask)
    ge = grads(model, eager_total)
    torch._dynamo.reset()
    fn = torch.compile(model.loss, backend="aot_eager", fullgraph=True, dynamic=False)
    total, nll = fn(ids, mask)
    gc = grads(model, total)
    check("yeni kayip fullgraph=True ile derlendi (grafik kirilmasi yok)", True)
    check("aot_eager: kayip ve gradyanlar eager ile BIT DUZEYINDE ayni", torch.equal(total, eager_total) and same(gc, ge))
    explain = torch._dynamo.explain(model.loss)(ids, mask)
    check("dynamo: 1 grafik, 0 kirilma", explain.graph_count == 1 and explain.graph_break_count == 0,
          "grafik %d kirilma %d" % (explain.graph_count, explain.graph_break_count))
    torch._dynamo.reset()
    original = M.masked_nll
    M.masked_nll = old_nll
    try:
        explain = torch._dynamo.explain(model.loss)(ids, mask)
    finally:
        M.masked_nll = original
    reasons = " | ".join(str(b.reason).splitlines()[0][:70] for b in explain.break_reasons[:1])
    check("bilgi: eski kayip (logits[valid]) grafigi boler", explain.graph_break_count >= 1,
          "grafik %d kirilma %d: %s" % (explain.graph_count, explain.graph_break_count, reasons))
    torch._dynamo.reset()


def t_numerics_flags(ids, mask, n):
    """numerics(): ayarlar cikista geri gelir; CPU'da tf32 hesabi degistirmez; bf16 autocast kayipla calisir."""
    print("numerics() ve train_seq secenekleri")
    before = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with TR.numerics("tf32", "fused", "cpu", autocast=True):
        inside = torch.get_float32_matmul_precision()
    with TR.numerics("bf16", "math", "cpu", autocast=True):
        red = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        dtype = (torch.ones(2, 2) @ torch.ones(2, 2)).dtype
    after = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    check("tf32 icinde 'high'; bf16 icinde matmul bf16, split-K toplama fp32; cikista eski ayar",
          inside == "high" and dtype == torch.bfloat16 and red is False and after == before)
    base, _ = TR.train_seq("shared", ids, mask, n, steps=3, log_at=())
    tf32, _ = TR.train_seq("shared", ids, mask, n, steps=3, log_at=(), precision="tf32")
    check("CPU'da precision='tf32' agirliklari degistirmez (bu makine: AVX2, TF32 yok)",
          all(torch.equal(a, b) for a, b in zip(base.state_dict().values(), tf32.state_dict().values())))
    bf, curve = TR.train_seq("shared", ids, mask, n, steps=3, log_at=(0, 3), precision="bf16")
    check("precision='bf16': kayip sonlu, agirliklar fp32 kalir",
          all(torch.isfinite(torch.tensor(c["nll"])) for c in curve) and
          all(p.dtype == torch.float32 for p in bf.parameters()), "nll %s" % [round(c["nll"], 4) for c in curve])
    fused, _ = TR.train_seq("shared", ids, mask, n, steps=3, log_at=(), fused_adam=True)
    diff = max((a - b).abs().max().item() for a, b in zip(base.state_dict().values(), fused.state_dict().values()))
    check("bilgi: fused Adam (CPU) ile varsayilan Adam, 3 adim", diff < 1e-4, "max |fark| %.1e" % diff)


def t_emulation():
    """Taklidin kendisi: TF32 yuvarlamasi 10 bit; bf16 taklidinde normalize fp32 doner (CUDA politikasi)."""
    print("taklit (numerics_emulation)")
    x = torch.tensor([1.0, 1.0 + 2 ** -10, 1.0 + 2 ** -11, 1.0 + 3 * 2 ** -11, -1.0 - 2 ** -12], dtype=torch.float32)
    check("TF32 en yakina: 1+2^-11 -> 1 (cift), 1+3*2^-11 -> 1+2^-9, 1+2^-10 aynen",
          round_tf32(x, "nearest").tolist() == [1.0, 1.0 + 2 ** -10, 1.0, 1.0 + 2 ** -9, -1.0])
    check("TF32 kesme: alt 13 bit silinir", round_tf32(x, "truncate").tolist() == [1.0, 1.0 + 2 ** -10, 1.0, 1.0 + 2 ** -10, -1.0])
    a, b = torch.randn(16, 32, requires_grad=True), torch.randn(32, 8, requires_grad=True)
    with tf32_emulation(None):
        y = a @ b
    check("tf32_emulation(None): a @ b ile ayni", torch.allclose(y, a.detach() @ b.detach(), atol=0, rtol=0))
    with cuda_bf16_autocast():
        h = torch.randn(4, 64) @ torch.randn(64, 64)
        n = F.normalize(h, dim=-1)
    check("cuda_bf16_autocast: matmul bf16, normalize fp32 (CUDA'daki gibi)", h.dtype == torch.bfloat16 and n.dtype == torch.float32)
    ids = torch.randint(4, 300, (2, 64), generator=torch.Generator().manual_seed(5))
    for name, model in (("Model X", M.BlockModel(300, d=64, units=128, seed=0)),
                        ("transformer", MT.TransformerModel(300, d=64, units=128, seed=0))):
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k.endswith(("W_context", "W_fact_out")):
                    p.normal_(0, 0.1, generator=torch.Generator().manual_seed(7))
            ref = model.logits(ids)
            with sdpa_kernel([SDPBackend.MATH]):      # CUDA math yolu gibi: bf16 girdi, fp32 hesap
                with cuda_bf16_autocast():
                    slow = model.logits(ids).float()
                with bf16_emulation():
                    fast = model.logits(ids).float()
        dev, gap = (slow - ref).abs().mean().item(), (fast - slow).abs().mean().item()
        check("bf16_emulation = cuda_bf16_autocast (%s, math SDPA): ortalama fark bf16 sapmasinin %%10'undan az" % name,
              gap <= dev / 10, "ort bf16 sapma %.1e, iki taklit arasi %.1e (yuvarlama sinirinda 1 bf16 ulp; bias ekleme sirasi)" % (dev, gap))


def old_rope(x):
    """apply_rope'un onceki hali: konum ve aci x.dtype'ta."""
    T, dh = x.shape[-2], x.shape[-1]
    freq = 10000.0 ** (-torch.arange(0, dh, 2, dtype=x.dtype) / dh)
    angle = torch.arange(T, dtype=x.dtype)[:, None] * freq[None, :]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def t_rope():
    """apply_rope acilari en az fp32'de: fp32'de oncekiyle bit duzeyinde ayni; bf16 girdide konum hatasi yok."""
    print("RoPE ve bf16")
    x = F.normalize(torch.randn(2, 512, 384, generator=torch.Generator().manual_seed(3)), dim=-1)
    check("fp32: yeni apply_rope eskisiyle BIT DUZEYINDE ayni", torch.equal(M.apply_rope(x), old_rope(x)))
    ref = old_rope(x.double())
    old_err = (old_rope(x.bfloat16()).double() - ref).abs().max().item()
    new_err = (M.apply_rope(x.bfloat16()).double() - ref).abs().max().item()
    check("bf16 girdi: eski hata buyuk (konum/aci bf16), yeni hata bf16 yuvarlamasi kadar", old_err > 0.05 and new_err < 0.01,
          "eski %.2e  yeni %.2e" % (old_err, new_err))


def t_bench_cpu():
    """bench_speed'in mantigi CPU'da, minik ayarla: *_repeat 0 sapma, bf16 sifirdan farkli, sure olculur."""
    print("bench_speed (CPU, minik ayar; GPU sureleri Colab'da)")
    import bench_speed as BS
    BS.DEVICE = "cpu"
    cfg = dict(n=50, batch=4, seq_len=32, model_kw=dict(d=16, turns=2, units=32, t_max=32), prep=2, warm=1, timed=2,
               every=1, eval_rows=4)
    s = BS.tinystories(["base", "base_repeat", "old_loss", "tf32", "bf16", "fused_attention", "fused_adam"], cfg=cfg)
    check("base_repeat: egri ve agirlik farki 0 (CPU'da deterministik)",
          s["base_repeat"]["dnll_max"] == 0 and s["base_repeat"]["param_dist"] == 0)
    check("bf16: ileri ve agirlik farki > 0 (secenek gercekten uygulandi)",
          s["bf16"]["forward"]["max_abs"] > 0 and s["bf16"]["param_dist"] > 0)
    check("old_loss (boolean indeks): agirliklar base ile BIT DUZEYINDE ayni", s["old_loss"]["param_dist"] == 0,
          "egri max|dnll| %.1e (yalniz kayip degerinin son biti)" % s["old_loss"]["dnll_max"])
    check("butun secenekler sure verdi", all(v["ms"] > 0 for v in s.values()) and len(s) == 7)


if __name__ == "__main__":
    ids, mask, n = kinship()
    print("veri: akrabalik 7ae5615623aa, %s, sozluk %d" % (tuple(ids.shape), n))
    t_emulation()
    t_rope()
    t_masked_nll(ids, mask, n)
    t_trajectory(ids[:256], mask[:256], n)            # bit esitligi boydan bagimsiz; sure icin ilk 256 cumle
    t_attention_4d(ids, n)
    t_compile(ids[:512], mask[:512], n)
    t_numerics_flags(ids[:256], mask[:256], n)
    t_bench_cpu()
    print("\n%d/%d GECTI" % (sum(RESULTS), len(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
