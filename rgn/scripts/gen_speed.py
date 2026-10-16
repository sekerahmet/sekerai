"""Üretim (çıkarım) hızı profili: token/s ve token başına ms, bağlam boyuna ve batch'e göre; ileri geçişin parçalara
göre süresi (gömme, dikkat, ortak MLP, özel nöron, LM head).

    python scripts/gen_speed.py --runs fw_s000_d512_tf_yg fw_s000_d512_tf_yg_ozel_rms

İki yol: önbellekli (rgn/model/decode.py: KV tamponu + torch.compile + CUDA graph) ve önbelleksiz başvuru (her yeni
token'da bütün bağlam yeniden).  Her durum ölçülmeden önce aynı şekilde bir kez çalıştırılır (derleme süresi sayılmaz).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rgn.train.checkpoint import load_run  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def gen_rate(m, batch, prompt, new, dev, dtype, cache=True):
    idx = torch.randint(0, 50000, (batch, prompt), device=dev)
    with torch.autocast(dev.type, dtype=dtype, enabled=dev.type == "cuda"):
        m.generate(idx, new, cache=cache)                              # ısınma: aynı şekil (derleme, graph, maske)
        _sync()
        t = time.perf_counter()
        m.generate(idx, new, cache=cache)
        _sync()
    sec = time.perf_counter() - t
    return batch * new / sec, 1000 * sec / new


@torch.no_grad()
def forward_parts(m, batch, ctx, dev, dtype, reps=5):
    """Tek ileri geçişin parçaları (bağlam ctx): kanca ile katman alt modülleri ölçülür."""
    idx = torch.randint(0, 50000, (batch, ctx), device=dev)
    times = {}
    def wrap(name, fn):
        def f(*a, **k):
            _sync(); t = time.perf_counter(); out = fn(*a, **k); _sync()
            times[name] = times.get(name, 0.0) + time.perf_counter() - t
            return out
        return f
    for b in m.blocks:
        b.attn.forward = wrap("dikkat", b.attn.forward)
        if b.mlp is not None:
            b.mlp.forward = wrap("ortak MLP", b.mlp.forward)
        b.private_out = wrap("özel nöron", b.private_out)
    lm = m.lm_head.forward
    m.lm_head.forward = wrap("LM head", lm)
    with torch.autocast(dev.type, dtype=dtype, enabled=dev.type == "cuda"):
        m(idx)
        times.clear()
        _sync(); t = time.perf_counter()
        for _ in range(reps):
            m(idx)
        _sync()
        total = (time.perf_counter() - t) / reps
    return total, {k: v / reps for k, v in times.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--new", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    for run in a.runs:
        m, step = load_run(run, dev)
        print("\n== %s (adım %s) — %s" % (run, step, torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))
        with torch.no_grad():                                          # doğruluk: önbellekli = önbelleksiz (fp32 greedy)
            idx = torch.randint(0, 50000, (4, 200), device=dev)
            a_, b_ = m.generate(idx, 48), m.generate(idx, 48, cache=False)
            print("doğruluk (fp32 greedy, 4 × 48 token): önbellekli = önbelleksiz %%%.1f" % (
                100 * (a_[:, 200:] == b_[:, 200:]).float().mean().item()))
        print("%-34s %12s %14s" % ("durum", "token/s", "ms / yeni token"))
        for cache in (True, False):
            for batch, prompt in ((1, 32), (1, 512), (1, 1024), (16, 512), (64, 512)):
                if not cache and batch == 64:
                    continue
                tps, ms = gen_rate(m, batch, prompt, a.new, dev, dtype, cache)
                print("%-34s %12.0f %14.2f" % ("%s, batch %d, bağlam %d" % ("önbellekli" if cache else "önbelleksiz",
                                                                            batch, prompt), tps, ms))
        for batch, ctx in ((1, 1024),):
            total, parts = forward_parts(m, batch, ctx, dev, dtype)
            print("tek ileri geçiş (batch %d, bağlam %d): %.2f ms" % (batch, ctx, 1000 * total))
            for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
                print("   %-12s %7.2f ms  (%%%.0f)" % (k, 1000 * v, 100 * v / total))
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
