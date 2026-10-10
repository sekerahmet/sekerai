"""tests_model_s -- Model S parcalarinin testleri (CPU, saniyeler).  python tests_model_s.py [--only ops,...]"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
if os.name == "nt":                                                     # kural 2a: EcoQoS kapali
    import train as _T
    _T._no_power_throttling()
from operators import AttentionOp, Context, FeedForwardOp, Projection, rope  # noqa: E402

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%s %s %s" % ("OK  " if ok else "HATA", name, info), flush=True)


def t_ops():
    """Parca 1: P yaricap kisiti; A = elle yazilmis integral (nedensel softmax cekirdegi, GQA, RoPE, QK izdusumu; fp32), A
    nedensel, A dense maske yolu = nedensel yol; F her token'da ayri."""
    torch.manual_seed(0)
    B, T, d, H, Hkv = 2, 9, 32, 4, 2
    u = torch.randn(B, T, d)
    pos = torch.arange(T).expand(B, T)
    ctx = Context(pos)
    P = Projection(d)
    with torch.no_grad():
        rms = P(u).pow(2).mean(-1).sqrt()
    check("P: cikti yaricapi 1 (g = 1)", float((rms - 1).abs().max()) < 1e-5, "max sapma %.1e" % float((rms - 1).abs().max()))
    A = AttentionOp(d, H, Hkv)
    with torch.no_grad():
        for p in (A.P.g, A.q_P.g, A.k_P.g):
            p.normal_(1, 0.2)
        got = A(u, ctx)
        h = A.P(u)                                                      # elle: integral formulu, dongu ile
        hd = d // H
        q = A.q_P(A.w_q(h).view(B, T, H, hd)).permute(0, 2, 1, 3)
        kv = A.w_kv(h).view(B, T, 2, Hkv, hd)
        k, v = A.k_P(kv[:, :, 0]).permute(0, 2, 1, 3), kv[:, :, 1].permute(0, 2, 1, 3)
        q, k = rope(q, pos), rope(k, pos)
        out = torch.zeros(B, T, H, hd)
        for b in range(B):
            for hh in range(H):
                kh = hh // (H // Hkv)
                for x in range(T):
                    s = torch.stack([q[b, hh, x] @ k[b, kh, x2] for x2 in range(x + 1)]) / hd ** 0.5
                    g = torch.softmax(s, 0)
                    out[b, x, hh] = sum(g[i] * v[b, kh, i] for i in range(x + 1))
        ref = A.w_o(out.reshape(B, T, d))
        d_ref = float((got - ref).abs().max())
        u2 = u.clone()
        u2[:, 6] += 1.0
        d_causal = float((A(u2, ctx)[:, :6] - got[:, :6]).abs().max())
        mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, T, T)
        d_mask = float((A(u, Context(pos, mask)) - got).abs().max())
    check("A: = elle integral (nedensel softmax, GQA, RoPE, QK izdusumu)", d_ref < 1e-5, "fark %.1e" % d_ref)
    check("A: nedensel (konum 6 degisince 0-5 ayni)", d_causal == 0, "fark %.1e" % d_causal)
    check("A: dense maske yolu = nedensel yol", d_mask < 1e-5, "fark %.1e" % d_mask)
    Fop = FeedForwardOp(d, 48)
    with torch.no_grad():
        f1 = Fop(u)
        u3 = u.clone()
        u3[:, 4] += 1.0
        f2 = Fop(u3)
        other = torch.cat([f2[:, :4], f2[:, 5:]], 1) - torch.cat([f1[:, :4], f1[:, 5:]], 1)
    check("F: her token'da ayri (konum 4 degisince digerleri ayni, kendisi farkli)",
          float(other.abs().max()) == 0 and float((f2[:, 4] - f1[:, 4]).abs().max()) > 1e-4)


TESTS = dict(ops=t_ops)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), " | HATA: " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)
