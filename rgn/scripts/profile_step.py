"""Eğitim adımı profili: bölüm bölüm süre ve en çok CUDA süresi alan işlemler.

    python scripts/profile_step.py --config configs/fineweb/fw_s000_d512_tf_yg_ozel_rms.json
    python scripts/profile_step.py --config ... --set train.micro_batch=64 --steps 20 --no_table   # yalnız ms/adım
    python scripts/profile_step.py --config ... --buckets      # süre nereye: dikkat / özel / LM head / MLP / ... (%)

Eğitimin kendi adımını (rgn.train.step.train_step: kelimeye özel satır kopyası, ileri / geri, kırpma, optimizer) birkaç
kez çalıştırır; ısınmadan sonra senkronlu ölçüm ve torch.profiler tablosu.
"""
import argparse
import re
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
from rgn.train.step import train_step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--set", action="append", default=[], help="bölüm.alan=değer (load_config ezmesi)")
    ap.add_argument("--no_table", action="store_true", help="yalnız ms/adım (profiler tablosu yok)")
    ap.add_argument("--buckets", action="store_true", help="CUDA süresinin bileşen dökümü (tablo yerine)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
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
    opt_ms = []
    if a.buckets:                                                       # optimizer süresi: senkronlu sarmal
        inner = opt.step

        def timed_step(*args, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = inner(*args, **kw)
            torch.cuda.synchronize()
            opt_ms.append(1000 * (time.perf_counter() - t))
            return r
        opt.step = timed_step

    def step():
        batches = [(*train_s.random_batch(micro, T, rng), None) for _ in range(accum)]
        train_step(model, fwd, opt, batches, ctx, tc.grad_clip, dev, torch.bfloat16)

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    print("adım %.1f ms  (%s; mikro %d × %d, tepe bellek %.1f GB)" % (
        1000 * (time.time() - t0) / a.steps, " ".join(a.set) or cfg.name, accum, micro,
        torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    if a.no_table:
        return
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=a.buckets) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    if a.buckets:
        buckets(prof, cfg.model, 3, opt_ms[-3:])
        return
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))


KERNELS = [  # (kova, çekirdek adı deseni) — sırayla ilk eşleşen
    ("dikkat çekirdeği", r"flex|attention|fmha|flash|sdpa"),
    ("özel nöron çekirdekleri", r"^(fwd|bwd_tok|bwd_row)$"),
    ("matris çarpımı", r"gemm|cutlass|nvjet|xmma|sm\d+_|cublas|matmul|^triton_tem_fused_.*mm"),
    ("kayıp (CE / softmax)", r"cross_entropy|log_softmax|nll|softmax"),
    ("kaynaşmış elementwise (inductor)", r"^triton_"),
    ("kopya / sıfırlama", r"memcpy|memset|copy"),
]
ANNOT = r"^## |^Optimizer\.|^ProfilerStep|#"           # GPU üzerindeki etiket aralıkları (çekirdek değil; çift sayım)


def buckets(prof, mc, steps, opt_ms):
    """CUDA çekirdek süresini kovalara ayırır; matris çarpımlarını aten şekline göre LM head / MLP / diğer'e böler."""
    ev = prof.events()
    seen, dev = set(), []
    for e in ev:                                       # GPU çekirdekleri; aynı çekirdeğin yinelenen kaydı bir kez
        if e.device_type.name != "CUDA" or getattr(e, "is_user_annotation", False) or re.search(ANNOT, e.name):
            continue
        key = (e.name, e.time_range.start, e.time_range.end)
        if key not in seen:
            seen.add(key)
            dev.append(e)
    total = sum(e.device_time for e in dev) / 1000 / steps
    sums = {}
    for e in dev:
        k = next((b for b, p in KERNELS if re.search(p, e.name, re.I)), "diğer")
        sums[k] = sums.get(k, 0.0) + e.device_time / 1000 / steps
    V, F = mc.vocab_size, mc.ff_dim()
    mm = {}
    for e in ev:
        if e.device_type.name != "CPU" or e.name not in ("aten::mm", "aten::addmm", "aten::bmm", "aten::_scaled_mm"):
            continue
        dims = {d for s in (e.input_shapes or []) for d in s}
        k = ("LM head" if V in dims else "ortak MLP" if (F in dims or 2 * F in dims)       # MLP a: d → 2·d_ff [G | U]
             else "diğer (projeksiyon, optimizer, ...)")
        mm[k] = mm.get(k, 0.0) + e.device_time_total / 1000 / steps
    head = mm.get("LM head", 0.0) + sums.get("kayıp (CE / softmax)", 0.0)       # gövde: LM head ve kayıp hariç
    body = total - head
    print("\nCUDA süresi dökümü (adım başına, %d adım ortalaması): toplam %.1f ms; LM head + kayıp %.1f ms, gövde %.1f ms"
          % (steps, total, head, body))
    print("  %-34s %8s  %6s  %7s" % ("", "ms", "toplam", "gövde"))
    for k, v in sorted(sums.items(), key=lambda kv: -kv[1]):
        g = "" if k == "kayıp (CE / softmax)" else "%%%5.1f" % (100 * v / body)
        print("  %-34s %8.1f  %%%5.1f  %7s" % (k, v, 100 * v / total, g))
    if mm:
        print("  matris çarpımı, şekle göre (aten):")
        for k, v in sorted(mm.items(), key=lambda kv: -kv[1]):
            g = "" if k == "LM head" else "%%%5.1f" % (100 * v / body)
            print("    %-32s %8.1f  %%%5.1f  %7s" % (k, v, 100 * v / total, g))
    if opt_ms:
        print("  optimizer adımı (senkronlu, duvar)  %8.1f ms" % (sum(opt_ms) / len(opt_ms)))
    other = {}
    for e in dev:
        if not any(re.search(p, e.name, re.I) for _, p in KERNELS):
            other[e.name] = other.get(e.name, 0.0) + e.device_time / 1000 / steps
    if other:
        print("  'diğer' içinde en büyükler:")
        for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:10]:
            print("    %8.1f ms  %s" % (v, k[:110]))


if __name__ == "__main__":
    main()
