"""Inceltme siniri: ucuz model (Thinker'siz yerel) her token'da; pahali model (transformer) yalniz ucuzun belirsizligine gore secilen
f oranindaki token'da. Kayip(f) egrisi -- belirsizlikle secim vs rastgele vs kahin (fark en buyuk). Ust sinir: pahalinin baglam
durumlari her konumda hazir varsayilir (MoD tipi gercekleme bunu bedelsiz vermez).
Kullanim: python model_s/sx_refine_bound.py --cheap <kosu> --tf <kosu> [--rows 6]"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("deneme2/v2/common", "deneme2/v2/transformer", "deneme2/v2/diag", "model_s"):
    sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np  # noqa: E402
import data as D  # noqa: E402
from generate_readings import load_run  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--data", default=r"G:/Drive'ım/v2/simplestories_gpt2")
    ap.add_argument("--stream", default=r"G:/Drive'ım/simplestories")
    a = ap.parse_args(argv)
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    m, _ = load_run(a.cheap, a.data, "cpu")
    tf, _ = load_run(a.tf, a.data, "cpu")
    m.eval()
    tf.eval()
    valid = D.TokenStories(a.stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    ro, rs = ep["row_offsets"], ep["row_stories"]
    b = D.build_batch(valid, [rs[ro[r]:ro[r + 1]].tolist() for r in range(a.rows)], "transformer", "cpu")
    keep = b.target >= 0
    tgt = b.target[keep]
    with torch.no_grad():
        hc = m.forward_states(b.tokens, b.pos, b.doc)[0][keep]
        Hc, nc = [], []
        for i in range(0, hc.shape[0], 2048):
            lp = m._logits(hc[i:i + 2048]).float().log_softmax(-1)
            Hc.append(-(lp.exp() * lp).sum(-1))
            nc.append(-lp.gather(1, tgt[i:i + 2048, None])[:, 0])
        Hc, nc = torch.cat(Hc), torch.cat(nc)
        nt = tf.loss_per_target(b)[0]
    gain = nc - nt
    n = len(nc)
    print("hedef %d | ucuz %.4f  pahali %.4f  fark %.4f" % (n, float(nc.mean()), float(nt.mean()), float(gain.mean())))
    g = torch.Generator().manual_seed(0)
    orders = {"belirsizlik": torch.argsort(Hc, descending=True), "rastgele": torch.randperm(n, generator=g),
              "kahin": torch.argsort(gain, descending=True)}
    print("%6s  %s" % ("f", "  ".join("%-22s" % k for k in orders)))
    for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        k = int(f * n)
        row = []
        for nm, o in orders.items():
            loss = (nc.sum() - gain[o[:k]].sum()) / n
            row.append("%.4f (%3.0f%% geri)" % (float(loss), 100 * float(gain[o[:k]].sum() / gain.sum())))
        print("%6.1f  %s" % (f, "  ".join("%-22s" % r for r in row)))


if __name__ == "__main__":
    main()
