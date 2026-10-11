"""Tur B'nin tek bilinmeyeni (danisman/06): head basina keskinlik. Sorgu basina skor sapmasi s = q.k/sqrt(d_h) (gorulen anahtarlar uzerinde),
sigma_s^2; ve dogrusallastirmanin gercek hatasi: softmax ciktisi ile e^s'nin ikinci derece Taylor'u (1 + s + s^2/2, skorlar sorgu basina
ortalanmis) agirlikli ciktisi arasindaki bagil fark. phi = sigma_s^2 > 2 eps olan head payi (eps 0,05 / 0,1).  V2 bicimi (fw_docs).
Kullanim: python model_s/sx_sharpness.py [--docs 30]"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
import numpy as np  # noqa: E402
from baseline import BaselineTransformer  # noqa: E402
from fw_docs import fw_docs  # noqa: E402
from sx_code_recall import RUN  # noqa: E402
from sx_voice_read import _no_power_throttling  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=30)
    a = ap.parse_args()
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.eval()
    L, H = idt["layers"], idt["heads"]
    var = np.zeros((L, H))
    err = np.zeros((L, H))
    n = 0
    with torch.no_grad():
        for tok, _ in fw_docs(a.docs, start=500, minlen=256):
            x = tf.E(torch.tensor([tok]))
            pos = torch.arange(len(tok))[None]
            T = len(tok)
            causal = torch.ones(T, T, dtype=torch.bool).tril()
            cnt = causal.float().sum(-1)
            for li, blk in enumerate(tf.blocks):
                q, k, v = blk._qkv(x, pos)
                s = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5                       # (1, H, T, T)
                s = s.masked_fill(~causal, 0.0)
                mu = s.sum(-1, keepdim=True) / cnt[:, None]
                c = (s - mu).masked_fill(~causal, 0.0)                                 # sorgu basina ortalanmis skor
                var[li] += (c.pow(2).sum(-1) / cnt).mean(-1)[0].numpy()
                w = c.masked_fill(~causal, float("-inf")).softmax(-1)
                t2 = (1 + c + c.pow(2) / 2).masked_fill(~causal, 0.0)                  # e^c'nin ikinci derece Taylor'u
                wl = t2 / t2.sum(-1, keepdim=True)
                o, ol = w @ v, wl @ v
                err[li] += ((o - ol).norm(dim=-1) / o.norm(dim=-1).clamp(min=1e-6)).mean(-1)[0].numpy()
                x = blk._finish(x, w @ v)
            n += 1
    var /= n
    err /= n
    print("belge %d | katman x head: sigma_s^2 (ortalama) / dogrusal okuma bagil hatasi" % n)
    for li in range(L):
        print("L%-2d " % li + "  ".join("%5.2f/%4.2f" % (var[li, h], err[li, h]) for h in range(H)))
    for eps in (0.05, 0.1, 0.25):
        print("phi (sigma_s^2 > 2 eps = %.2f): %.3f | bagil hata > %.2f olan head payi: %.3f" % (
            2 * eps, float((var > 2 * eps).mean()), eps, float((err > eps).mean())))
    print("ortanca sigma_s^2 %.2f, ortanca hata %.3f" % (float(np.median(var)), float(np.median(err))))


if __name__ == "__main__":
    main()
