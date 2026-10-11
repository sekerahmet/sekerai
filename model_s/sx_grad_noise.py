"""Tur D tek bilinmeyen (danisman/08): dizi basina gradyanlardan (i) gurultu olcegi B_noise = tr(Sigma) / |G|^2 (McCandlish; dizi ve token
cinsinden), (ii) onem orneklemesi carpani rho_IS = mean|g_s|^2 / (mean|g_s|)^2, (iii) ileri gecisten bedava gostergelerle (dizi kaybi, ortalama
1 - p_hedef) |g_s|'nin Spearman'i. FineWeb transformer, V2 bicimi. Bellek: gradyanlarin yuruyen toplami.
Kullanim: python model_s/sx_grad_noise.py [--seqs 32] [--maxlen 512]"""
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


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=512)
    a = ap.parse_args()
    _no_power_throttling()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    pack = torch.load(os.path.join(RUN, "agent.pt"), map_location="cpu", weights_only=False)
    idt = pack["identity"]
    tf = BaselineTransformer(idt["d"], idt["layers"], idt["heads"], vocab_rows=pack["state"]["E.weight"].shape[0])
    tf.load_state_dict(pack["state"])
    tf.train(False)
    params = [p for p in tf.parameters()]
    gsum = [torch.zeros_like(p) for p in params]
    sq, nrm, loss_s, ent_s, ntoks = [], [], [], [], []
    for tok, tgt in fw_docs(a.seqs, start=6000, maxlen=a.maxlen, minlen=a.maxlen // 2):
        tf.zero_grad(set_to_none=True)
        x = torch.tensor([tok])
        t = torch.tensor(tgt)
        ok = t >= 0
        lp = tf._logits(tf.hidden(x, torch.arange(len(tok))[None], None)).float().log_softmax(-1)[0]
        nll = -lp[ok].gather(1, t[ok][:, None])[:, 0]
        nll.mean().backward()                                                # token basina ortalama: dizi gradyani
        with torch.no_grad():
            s2 = sum(float(p.grad.pow(2).sum()) for p in params)
            for gs, p in zip(gsum, params):
                gs += p.grad
            sq.append(s2)
            nrm.append(s2 ** 0.5)
            loss_s.append(float(nll.mean()))
            ent_s.append(float((1 - nll.neg().exp()).mean()))
            ntoks.append(int(ok.sum()))
    S = len(sq)
    with torch.no_grad():
        gbar2 = sum(float((gs / S).pow(2).sum()) for gs in gsum)
    m2 = float(np.mean(sq))
    G2 = (S * gbar2 - m2) / (S - 1)                                          # |G|^2 yansiz
    trS = (m2 - gbar2) * S / (S - 1)                                         # tr(Sigma) dizi basina
    Bseq = trS / max(G2, 1e-30)
    tpseq = float(np.mean(ntoks))
    rho = m2 / float(np.mean(nrm)) ** 2
    print("dizi %d (ort. %.0f token) | |G|^2 %.3e, tr(Sigma) %.3e" % (S, tpseq, G2, trS))
    print("B_noise = %.1f dizi = %.0f token (bugunku batch ~65k token)" % (Bseq, Bseq * tpseq))
    print("rho_IS = mean|g|^2 / (mean|g|)^2 = %.3f" % rho)
    print("Spearman(|g_s|, dizi kaybi) %.3f | Spearman(|g_s|, ort. 1 - p_hedef) %.3f" % (spearman(nrm, loss_s), spearman(nrm, ent_s)))


if __name__ == "__main__":
    main()
