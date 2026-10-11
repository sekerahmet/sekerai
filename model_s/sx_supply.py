"""KKT'nin iki terimi (matematikci, tur 4): s konumundaki hesabin marjinal degeri = kendi terimi (l_s) + tedarik terimi (t > s, K/V yoluyla).
Transformer checkpoint'inde katman --split cikisinda h_s'ye gore: toplam gradyan (normal) ve kendi gradyani (ust katmanlarda K / V
dondurulmus: s'den baskalarina giden yol kesik). tedarik = toplam - kendi.  Gostergeler (ucuz model, nedensel): H_c(s) = s'de bir
sonraki token belirsizligi; surpriz_c(s) = -log p_c(x_s) (okunan token).
Kullanim: python model_s/sx_supply.py --tf <transformer> --cheap <ucuz strata kosusu> [--rows 3] [--split 4]"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("deneme2/v2/common", "deneme2/v2/transformer", "deneme2/v2/diag", "model_s"):
    sys.path.insert(0, os.path.join(ROOT, p))
import numpy as np  # noqa: E402
import data as D  # noqa: E402
from generate_readings import load_run  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402


def block_fwd(blk, x, pos, mask, freeze_kv):
    q, _, _ = blk._qkv(x, pos)
    _, k, v = blk._qkv(x.detach() if freeze_kv else x, pos)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
    return blk._finish(x, a)


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--split", type=int, default=4, help="gradyan bu kadar katmandan sonraki kalintida")
    ap.add_argument("--data", default=r"G:/Drive'ım/v2/simplestories_gpt2")
    ap.add_argument("--stream", default=r"G:/Drive'ım/simplestories")
    a = ap.parse_args(argv)
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    tf, _ = load_run(a.tf, a.data, "cpu")
    m, _ = load_run(a.cheap, a.data, "cpu")
    tf.eval()
    m.eval()
    valid = D.TokenStories(a.stream, a.data, "valid")
    ep = np.load(os.path.join(a.data, "exam_pack_plan.npz"))
    ro, rs = ep["row_offsets"], ep["row_stories"]
    b = D.build_batch(valid, [rs[ro[r]:ro[r + 1]].tolist() for r in range(a.rows)], "transformer", "cpu")
    keep = b.target >= 0
    T = b.doc.shape[1]
    mask = (b.doc[:, :, None] == b.doc[:, None, :]) & torch.ones(T, T, dtype=torch.bool).tril()
    with torch.no_grad():
        x = tf.E(b.tokens)
        for blk in tf.blocks[:a.split]:
            x = blk(x, b.pos, mask)
    grads = {}
    for mode in ("toplam", "kendi"):
        h = x.clone().requires_grad_(True)
        y = h
        for blk in tf.blocks[a.split:]:
            y = block_fwd(blk, y, b.pos, mask, freeze_kv=(mode == "kendi"))
        lp = tf._logits(tf.norm(y)).float().log_softmax(-1)
        nll = -lp.gather(-1, b.target.clamp(min=0)[..., None])[..., 0]
        (nll * keep).sum().backward()
        grads[mode] = h.grad.detach()
        print("  %s gecti, kayip %.4f" % (mode, float(nll[keep].mean())), flush=True)
    g_tot, g_own = grads["toplam"], grads["kendi"]
    g_sup = g_tot - g_own
    no, ns = g_own.norm(dim=-1), g_sup.norm(dim=-1)
    real = b.doc >= 0
    with torch.no_grad():
        hc = m.forward_states(b.tokens, b.pos, b.doc)[0]
        lpc = m._logits(hc[real]).float().log_softmax(-1)
        Hc = torch.zeros(real.shape)
        Hc[real] = -(lpc.exp() * lpc).sum(-1)
        nxt = torch.zeros(real.shape)                                  # -log p_c(x_{s+1}) konum s'de
        nxt[real] = -lpc.gather(1, b.target.clamp(min=0)[real][:, None])[:, 0]
        surp = torch.zeros(real.shape)                                 # okunan token x_s'nin surprizi: onceki konumun tahmini
        surp[:, 1:] = nxt[:, :-1]
        surp = torch.where(torch.cat([torch.zeros_like(real[:, :1]), b.doc[:, 1:] == b.doc[:, :-1]], 1), surp, torch.zeros_like(surp))
    sel = keep & real
    o, s_, H, su = no[sel], ns[sel], Hc[sel], surp[sel]
    print("hedef %d | katman %d kalintisi" % (int(sel.sum()), a.split))
    print("tedarik payi  sum|g_sup| / sum(|g_own| + |g_sup|) = %.3f   (ortanca oran g_sup/g_own %.2f)" % (
        float(s_.sum() / (o.sum() + s_.sum())), float((s_ / o.clamp(min=1e-12)).median())))
    print("Spearman:  kendi~H_c %.3f   kendi~surpriz %.3f   tedarik~surpriz %.3f   tedarik~H_c %.3f   kendi~tedarik %.3f" % (
        spearman(o, H), spearman(o, su), spearman(s_, su), spearman(s_, H), spearman(o, s_)))
    for nm, v in (("surpriz", su), ("H_c", H)):
        q = torch.quantile(v, torch.tensor([0.25, 0.5, 0.75]))
        bins = [v <= q[0], (v > q[0]) & (v <= q[1]), (v > q[1]) & (v <= q[2]), v > q[2]]
        print("%-8s ceyrekleri: |g_own| %s | |g_sup| %s" % (nm, " ".join("%.2e" % float(o[k].mean()) for k in bins),
                                                         " ".join("%.2e" % float(s_[k].mean()) for k in bins)))


if __name__ == "__main__":
    main()
