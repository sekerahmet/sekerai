"""moe_check -- deneme/moe2 MoE katmaninin GPU denetimi (belge 107 s6 adim 1; uzmanlar ozel operator modelz::moe_experts): gercek boy (d 1024, G MLP 5.440, 16 + 1 uzman,
token basina 3, 32 x 2048 token), egitimle ayni derleme modu.  (1) derlenmis ileri + geri = uzman dongusu basvurusu (ayni
yonlendirme, bf16 goreli fark); (2) derlenmis adimda GPU -> CPU senkron sayisi (ozel operator cagri basina bir; ileri + geri 2);
(3) ileri + geri ms: MoE ve ayni etkin genislikte yogun SwiGLU MLP.  Son satir MOE_CHECK_BITTI.

    python moe_check.py [--experts 16] [--topk 3] [--shared 1] [--rows 32] [--reps 20]
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import os
import sys
import time

import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import model as S  # noqa: E402
import train as TR  # noqa: E402


def reference(m, x):
    """Uzman dongusu (bf16 matmul, ayni yonlendirme): ortak + sum p_j uzman_j."""
    idx, p = m.route(x)
    y = torch.zeros_like(x, dtype=torch.float32)
    w = m.width
    for e in range(m.n):
        tok, j = (idx == e).nonzero(as_tuple=True)
        if not len(tok):
            continue
        h = x[tok].bfloat16() @ m.w_gate_up[e].bfloat16().T
        a = F.silu(h[:, :w]) * h[:, w:]
        y.index_add_(0, tok, (a @ m.w_down[e].bfloat16().T).float() * p[tok, j, None])
    g, u = F.linear(x.bfloat16(), m.shared_gate_up.weight.bfloat16()).chunk(2, -1)
    return y + F.linear(F.silu(g) * u, m.shared_down.weight.bfloat16()).float()


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / reps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=5440)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--shared", type=int, default=1)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "GPU YOK"
    import torch._inductor.config as ic
    ic.autotune_in_subproc = True                                      # train.py ile ayni (bozuk aday alt surecte)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    m = S.MoE(a.d, a.hidden, a.experts, a.topk, a.shared).to(dev)
    with torch.no_grad():
        for n_, p_ in m.named_parameters():
            p_.normal_(0, 0.02 / (24 ** 0.5) if n_.endswith(("w_down", "down.weight")) else 0.02)
    dense = torch.nn.ModuleDict(dict(gate_up=torch.nn.Linear(a.d, 2 * a.hidden, bias=False),
                                     down=torch.nn.Linear(a.hidden, a.d, bias=False))).to(dev)
    N = a.rows * 2048
    x = torch.randn(N, a.d, device=dev)
    cm = torch.compile(m, dynamic=False, mode=TR.COMPILE_MODE)

    def dense_f(h):
        g, u = dense["gate_up"](h).chunk(2, -1)
        return dense["down"](F.silu(g) * u)
    cd = torch.compile(dense_f, dynamic=False, mode=TR.COMPILE_MODE)

    def step(fn):
        xx = x.view(a.rows, 2048, a.d).clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fn(xx)
        y.float().square().mean().backward()
        return xx.grad
    t0 = time.time()
    step(cm)
    print("derleme %.0f sn" % (time.time() - t0), flush=True)
    m.zero_grad(set_to_none=True)
    import warnings
    torch.cuda.set_sync_debug_mode("warn")                               # senkronlar uyari olarak sayilir
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        step(cm)
    torch.cuda.set_sync_debug_mode("default")
    n_sync = sum("synchroniz" in str(w_.message) for w_ in ws)
    sync = "%d (beklenen 2: ozel operator ileri + geri)" % n_sync
    g_mod = {n_: p_.grad.clone() for n_, p_ in m.named_parameters() if p_.grad is not None}
    m.zero_grad(set_to_none=True)
    xr = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        yr = reference(m, xr)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        yc = cm(x.view(a.rows, 2048, a.d)).reshape(N, a.d).float()
    d_y = float((yc - yr).abs().max() / yr.abs().max())
    yr.float().square().mean().backward()
    d_g = {n_: float((g_mod[n_] - p_.grad).abs().max() / p_.grad.abs().max()) for n_, p_ in m.named_parameters()
           if p_.grad is not None and n_ in g_mod}
    print("ileri: derlenmis / dongu goreli %.2e" % d_y, flush=True)
    print("geri (agirlik, goreli): " + ", ".join("%s %.2e" % (k, v) for k, v in d_g.items()), flush=True)
    print("senkron (derlenmis adim, ileri + geri): %s" % sync, flush=True)
    t_m, t_d = timed(lambda: step(cm), a.reps), timed(lambda: step(cd), a.reps)
    print("ileri + geri ms: MoE %.2f | yogun %.2f | fark %+.2f (katman basina; 4 G katmani %+.1f ms / adim)"
          % (t_m, t_d, t_m - t_d, 4 * (t_m - t_d)), flush=True)
    ok = n_sync <= 2 and d_y < 2e-2 and max(d_g.values()) < 5e-2
    print("SONUC: %s" % ("GECTI" if ok else "KALDI"), flush=True)
    print("MOE_CHECK_BITTI", flush=True)


if __name__ == "__main__":
    main()
