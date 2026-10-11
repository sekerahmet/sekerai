"""sx_flex_check -- GPU: Strata-x FLEX (BlockMask) yolu = yogun maske yolu (fp32; ileri durum, sinir, gradyan).
Cok belgeli satirlar (belge sinirlari, dolgu), pencere < T, birim kovasi 128'in kati.

    python model_s/sx_flex_check.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strata_x as SX  # noqa: E402


def main():
    assert torch.cuda.is_available(), "GPU yok"
    torch.manual_seed(0)
    dev = torch.device("cuda")
    m = SX.StrataX(d_local=64, heads_local=4, enc_layers=2, dec_layers=2, d_global=128, heads_global=4, global_layers=2,
                   theta=6.0, window=48).to(dev)
    m.out.set_order(torch.rand(m.out.vocab))
    B, T = 3, 256
    tok = torch.randint(0, 5000, (B, T), device=dev)
    doc = torch.zeros(B, T, dtype=torch.long, device=dev)
    doc[0, 100:] = 1
    doc[1, 60:200] = 1
    doc[1, 200:] = -1                                                    # dolgu
    doc[2, 30:] = 1
    doc[2, 140:] = 2
    pos = torch.zeros_like(doc)
    for b in range(B):
        for d_ in doc[b].unique():
            idx = (doc[b] == d_).nonzero()[:, 0]
            pos[b, idx] = torch.arange(len(idx), device=dev)
    res = {}
    for flex in (False, True):
        SX.FLEX = flex
        m.zero_grad()
        h, aux, ends, info = m.forward_states(tok, pos, doc)
        (h.pow(2).mean() + aux).backward()
        res[flex] = (h.detach(), ends, info, [p.grad.clone() for p in m.parameters() if p.grad is not None])
    SX.FLEX = True
    d_h = float((res[True][0] - res[False][0]).abs().max())
    same_e = torch.equal(res[True][1], res[False][1])
    d_g = max(float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in zip(res[True][3], res[False][3]))
    ok = d_h < 1e-4 and same_e and d_g < 1e-3
    print("%s strata-x FLEX = yogun maske (fp32): durum %.1e, sinirlar %s, gradyan (goreli) %.1e, birim / token %.3f"
          % ("OK  " if ok else "HATA", d_h, "ayni" if same_e else "FARKLI", d_g, float(res[True][1].float().mean())))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
