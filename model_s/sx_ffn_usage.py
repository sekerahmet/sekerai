"""Tur C tek bilinmeyen (danisman/07): katki matrisi C[t,i] = a_{t,i} (v_i . g_t), g_t kaybin FFN cikisina gore gradyani. Sutun gorunumu:
birim i'nin "gerektigi" token orani f_i (token t'de |C[t,:]| kutlesinin %90'ini kapsayan en kucuk birim kumesinde mi); yogun pay = #{f_i > 1/alpha}/M,
alpha 2 / 4 / 8. Satir gorunumu: k_eff = token basina o kume boyu / M. V2 bicimi (fw_docs).
Kullanim: python model_s/sx_ffn_usage.py [--docs 12]"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--cover", type=float, default=0.9)
    a = ap.parse_args()
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.eval()
    for p in tf.parameters():
        p.requires_grad_(False)
    L = idt["layers"]
    M = tf.blocks[0].down.in_features
    need = np.zeros((L, M))
    keff = [[] for _ in range(L)]
    ntok = 0
    for tok, tgt in fw_docs(a.docs, start=5000, minlen=128):
        x = tf.E(torch.tensor([tok]))
        pos = torch.arange(len(tok))[None]
        acts, outs = [], []
        for blk in tf.blocks:
            q, k, v = blk._qkv(x, pos)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            B, T, d = x.shape
            x = x + blk.proj(att.transpose(1, 2).reshape(B, T, d))
            g, u = blk.gate_up(blk.n2(x)).chunk(2, -1)
            h = (F.silu(g) * u).detach()
            acts.append(h)
            o = blk.down(h).detach().requires_grad_(True)                    # yaprak: kaybin FFN cikisina gore gradyani o.grad
            outs.append(o)
            x = x + o
        t = torch.tensor(tgt)
        ok = t >= 0
        lp = tf._logits(tf.norm(x)).float().log_softmax(-1)[0]
        (-lp[ok].gather(1, t[ok][:, None]).sum()).backward()
        with torch.no_grad():
            for li in range(L):
                gt = outs[li].grad[0][ok]                                     # (n, d)
                C = (acts[li][0][ok] * (gt @ tf.blocks[li].down.weight)).abs()   # |a_ti (v_i . g_t)|, (n, M)
                s, idx = C.sort(-1, descending=True)
                cum = s.cumsum(-1) / s.sum(-1, keepdim=True).clamp(min=1e-12)
                kk = (cum < a.cover).sum(-1) + 1                             # %90'i kapsayan kume boyu
                mask = torch.arange(M)[None] < kk[:, None]
                sel = torch.zeros_like(C, dtype=torch.bool).scatter_(1, idx, mask)
                need[li] += sel.float().sum(0).numpy()
                keff[li] += (kk.float() / M).tolist()
        ntok += int(ok.sum())
    f = need / ntok
    print("token %d, birim %d / katman | katki kutlesinin %%%.0f'i icin token basina birim orani (k_eff) ve yogun pay" % (ntok, M, 100 * a.cover))
    print("  katman  k_eff ort / ortanca | yogun pay (f_i > 1/alpha): a2 / a4 / a8 | f_i ortanca, %90'lik")
    for li in range(L):
        print("  L%-2d     %.3f / %.3f     |  %.3f / %.3f / %.3f  | %.3f, %.3f" % (
            li, np.mean(keff[li]), np.median(keff[li]), (f[li] > 0.5).mean(), (f[li] > 0.25).mean(), (f[li] > 0.125).mean(),
            np.median(f[li]), np.percentile(f[li], 90)))
    print("tum katmanlar yogun pay a4: %.3f" % float((f > 0.25).mean()))


if __name__ == "__main__":
    main()
