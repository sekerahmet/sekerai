"""Tur C tek bilinmeyen adayi: FFN (SwiGLU) gizli birimlerinin token basina seyrekligi. h = silu(g) * u (hidden H). Olcu: |h| kutlesinin
(L1) token basina ilk %k birimde kalan payi; ve butun katmanlarda yalniz ilk %k birim tutulunca (gerisi 0) kayip (sifir atis). V2 bicimi (fw_docs).
Kullanim: python model_s/sx_ffn_sparsity.py [--docs 20]"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
import numpy as np  # noqa: E402
from baseline import BaselineTransformer  # noqa: E402
from fw_docs import fw_docs  # noqa: E402
from sx_code_recall import RUN  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402

FRACS = (0.01, 0.03, 0.1, 0.3)


def forward(tf, tok, keep=None, mass=None):
    x = tf.E(torch.tensor([tok]))
    pos = torch.arange(len(tok))[None]
    for li, blk in enumerate(tf.blocks):
        q, k, v = blk._qkv(x, pos)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, T, d = x.shape
        x = x + blk.proj(a.transpose(1, 2).reshape(B, T, d))
        g, u = blk.gate_up(blk.n2(x)).chunk(2, -1)
        h = F.silu(g) * u                                                    # (1, T, H)
        if mass is not None:
            s = h.abs().sort(-1, descending=True).values
            tot = s.sum(-1)
            for i, f in enumerate(FRACS):
                mass[li, i] += float((s[..., :max(1, int(f * h.shape[-1]))].sum(-1) / tot).mean())
        if keep is not None:
            kk = max(1, int(keep * h.shape[-1]))
            thr = h.abs().topk(kk, -1).values[..., -1:]
            h = torch.where(h.abs() >= thr, h, torch.zeros_like(h))
        x = x + blk.down(h)
    return tf._logits(tf.norm(x)).float().log_softmax(-1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20)
    a = ap.parse_args()
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.eval()
    docs = fw_docs(a.docs, start=4000, minlen=128)
    mass = np.zeros((idt["layers"], len(FRACS)))
    loss = {None: [0.0, 0]}
    for f in FRACS:
        loss[f] = [0.0, 0]
    with torch.no_grad():
        for tok, tgt in docs:
            t = torch.tensor(tgt)
            ok = t >= 0
            for f in loss:
                lp = forward(tf, tok, keep=f, mass=mass if f is None else None)
                loss[f][0] += float(-lp[ok].gather(1, t[ok][:, None]).sum())
                loss[f][1] += int(ok.sum())
    mass /= len(docs)
    H = tf.blocks[0].down.in_features
    print("belge %d, gizli birim %d | katman basina |h| kutlesinin ilk %%k birimdeki payi (k: %s)" % (
        len(docs), H, " / ".join("%g%%" % (100 * f) for f in FRACS)))
    for li in range(len(mass)):
        print("  L%d  " % li + "  ".join("%.3f" % m for m in mass[li]))
    base = loss[None][0] / loss[None][1]
    print("tam FFN kayip %.4f" % base)
    for f in FRACS:
        lf = loss[f][0] / loss[f][1]
        print("  yalniz ilk %%%g birim (butun katmanlar): %.4f (%+.4f)" % (100 * f, lf, lf - base))


if __name__ == "__main__":
    main()
