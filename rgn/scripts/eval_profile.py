"""Değerlendirme hızı profili (CUDA): ara değerlendirmeyle aynı pencereler, aynı batch.

Kollar: eski (her logit parçasında .item() senkronu, her parçada lm_head fp32 kopyası) / yeni (tek senkron, kopya bir
kez; sayılar aynı olmalı) / derlenmiş (yeni + torch.compile(model.hidden); kayıp farkı yazılır).  Ayrıca süre dökümü:
yalnız ileri geçiş (eager, derlenmiş) ve yalnız logit + kayıp.

    python scripts/eval_profile.py --run fw_s000_d1024L9_yari_epok
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rgn.config import from_dict  # noqa: E402
from rgn.data.loader import load_streams  # noqa: E402
from rgn.paths import drive_root, runs_dir  # noqa: E402
from rgn.system import setup_process  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402
from rgn.train.evaluate import evaluate  # noqa: E402


@torch.no_grad()
def evaluate_old(model, windows, batch, device, ctx, chunk=8192):
    """Değişiklikten önceki yol (yalnız kayıp ve doğruluk)."""
    W = model.lm_head.weight
    total, correct, count = 0.0, 0, 0
    for i in range(0, len(windows), batch):
        w = windows[i:i + batch].to(device, non_blocking=True)
        x, y = w[:, :-1], w[:, 1:].reshape(-1)
        with ctx:
            h = model.hidden(x)
        h = h.reshape(-1, h.shape[-1]).float()
        for j in range(0, y.numel(), chunk):
            logits = h[j:j + chunk] @ W.float().T
            t = y[j:j + chunk]
            total += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            correct += (logits.argmax(-1) == t).sum().item()
        count += y.numel()
    return {"loss": total / count, "acc": correct / count}


def _timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t


@torch.no_grad()
def main():
    setup_process()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", type=int, default=0, help="0: yapılandırmadaki eval_tokens")
    ap.add_argument("--batch", type=int, default=0, help="0: yapılandırmadaki micro_batch")
    a = ap.parse_args()
    dev = torch.device("cuda")
    d = next(p for p in (runs_dir() / a.run, drive_root() / "RGN" / "runs" / a.run) if p.is_dir())
    cfg = from_dict(json.loads((d / "config.json").read_text(encoding="utf-8")))
    m, step = load_run(a.run, dev)
    _, valid_s, bpt = load_streams(cfg.data)
    T, batch = cfg.data.seq_len, a.batch or cfg.train.micro_batch
    windows = valid_s.fixed_windows(T, a.tokens or cfg.data.eval_tokens)
    ntok = (windows.shape[0]) * T
    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    print("%s (adım %s) — %d pencere × %d = %.1fM token, batch %d, %s" % (
        a.run, step, len(windows), T, ntok / 1e6, batch, torch.cuda.get_device_name(0)), flush=True)

    hid_c = torch.compile(m.hidden, dynamic=False)
    x0 = windows[:batch, :-1].to(dev)
    with ctx:
        _, t_comp = _timed(lambda: hid_c(x0))
    print("derleme (ilk çağrı): %.1f sn" % t_comp, flush=True)

    def fwd_only(fn):
        def run():
            for i in range(0, len(windows) - batch + 1, batch):
                with ctx:
                    fn(windows[i:i + batch, :-1].to(dev, non_blocking=True))
        return run
    with ctx:
        m.hidden(x0)                                                              # ısınma (flex, blok maskesi)
    _, t_fe = _timed(fwd_only(m.hidden))
    _, t_fc = _timed(fwd_only(hid_c))
    print("yalnız ileri geçiş: eager %.1f sn, derlenmiş %.1f sn" % (t_fe, t_fc), flush=True)

    rows = []
    r_old, t_old = _timed(lambda: evaluate_old(m, windows, batch, dev, ctx))
    rows.append(("eski", r_old, t_old))
    r_new, t_new = _timed(lambda: evaluate(m, windows, batch, dev, ctx, bpt))
    rows.append(("yeni", r_new, t_new))
    r_cmp, t_cmp = _timed(lambda: evaluate(m, windows, batch, dev, ctx, bpt, hidden=hid_c))
    rows.append(("yeni + derlenmiş", r_cmp, t_cmp))
    print("\n%-18s %10s %9s %8s %10s %11s" % ("kol", "kayıp", "doğruluk", "sn", "token/s", "fark (eski)"))
    for name, r, t in rows:
        print("%-18s %10.6f %9.5f %8.1f %10.0f %+11.2e" % (name, r["loss"], r["acc"], t, ntok / t,
                                                            r["loss"] - r_old["loss"]))
    print("\nlogit + kayıp payı (yeni − yalnız eager ileri): ~%.1f sn" % (t_new - t_fe))


if __name__ == "__main__":
    main()
