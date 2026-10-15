"""Gözlemci öngörüleri (kullanıcı, 15 Ekim: G gözlemcisi + belirsizlik geri beslemesi).

    python scripts/observer_probe.py --runs ss_d512_tf_yg_gozlem

Aynı model, özellikler açık / ortalamayla değiştirilmiş (bilgi silinir, ölçek korunur):
    öngörü 1  kalibrasyon: son tahminin güveni (max p) ile doğruluğu ne kadar uyumlu?  ECE (15 kutu) — özellikler
              açıkken daha düşük olmalı
    öngörü 2  özellikler silinince kayıp artışı: bütün token'lar, gözlemcinin "nadir kelime" beklediği konumlar (n en üst
              çeyrek), belirsizliği en yüksek çeyrek
    ek        gözlemcinin kendi doğruluğu, özelliklerin dağılımı, akışa katkı oranı |E·(2f−1)| / |x|
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.system import setup_process  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_anatomy import load, windows  # noqa: E402


@torch.no_grad()
def run(m, idx, mean_f=None):
    T = idx.shape[1]
    rope = m.rope(T)
    x = m.embed(idx)
    info = {}
    for i, blk in enumerate(m.blocks):
        out = blk(x, rope)
        x = out[0] if isinstance(out, tuple) else out
        if i == m.cfg.observer_layer - 1:
            ol, f = m.observe(x)
            info["f"] = f
            info["obs_acc"] = (ol.argmax(-1)[:, :-1] == m.obs_map[idx[:, 1:]]).float()
            fu = mean_f.expand_as(f) if mean_f is not None else f
            add = m.obs_feat((2.0 * fu - 1.0).to(x.dtype))
            info["ratio"] = (add.float().norm(dim=-1) / x.float().norm(dim=-1).clamp_min(1e-6)).mean().item()
            x = x + add
    logits = m.lm_head(m.norm_f(x)).float()[:, :-1]
    tgt = idx[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
    p = torch.softmax(logits, -1)
    conf, pred = p.max(-1)
    info.update(loss=loss, conf=conf, correct=(pred == tgt).float())
    return info


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            e += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return e


def main():
    setup_process()
    ap = argparse.ArgumentParser(description="gözlemci öngörüleri")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=16)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_all = windows(a.windows).to(device)
    for runname in a.runs:
        m, step = load(runname, device)
        assert m.obs_head is not None, "gözlemcisiz model"
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        batches = [idx_all[i:i + 4] for i in range(0, len(idx_all), 4)]
        with ctx:
            on = [run(m, b) for b in batches]
        mean_f = torch.cat([o["f"].reshape(-1, 3) for o in on]).mean(0)
        with ctx:
            off = [run(m, b, mean_f) for b in batches]
        cat = lambda rs, k: torch.cat([r[k].flatten() for r in rs]).float().cpu().numpy()  # noqa: E731
        L1, L0 = cat(on, "loss"), cat(off, "loss")
        F_ = torch.cat([o["f"][:, :-1].reshape(-1, 3) for o in on]).float().cpu().numpy()
        b, g, n = F_[:, 0], F_[:, 1], F_[:, 2]
        d = L0 - L1
        hi_n = n >= np.quantile(n, 0.75)
        hi_b = b >= np.quantile(b, 0.75)
        print("\n== %s (adım %s)" % (runname, step))
        print("  gözlemci: kendi doğruluğu %.3f | özellik ort (belirsizlik %.2f, güven %.2f, nadir %.2f) | akışa katkı oranı %.3f"
              % (cat(on, "obs_acc").mean(), b.mean(), g.mean(), n.mean(), np.mean([o["ratio"] for o in on])))
        print("  öngörü 1 — kalibrasyon ECE: özellikler açık %.4f | silinmiş %.4f" % (
            ece(cat(on, "conf"), cat(on, "correct")), ece(cat(off, "conf"), cat(off, "correct"))))
        print("  öngörü 2 — özellikler silinince kayıp artışı: bütün %+.4f | nadir beklenen (üst çeyrek) %+.4f | "
              "belirsizlik üst çeyrek %+.4f | geri kalan %+.4f" % (d.mean(), d[hi_n].mean(), d[hi_b].mean(),
                                                                  d[~hi_n & ~hi_b].mean()))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
