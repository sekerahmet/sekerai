"""tests_model_s -- Model S parcalarinin testleri (CPU, saniyeler).  python tests_model_s.py [--only ops,...]"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "common"))   # ortak veri / egitim kodu
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


def t_steps():
    """Parca 2: lie adimi = bugunku transformer blogu (transformer/baseline.Block, agirliklar kopyalanir; GQA, fp32);
    strang = elle F1/2 -> A -> F2/2 ve parametre sayisi lie ile ayni; parallel = elle A + F; uc sema birbirinden farkli."""
    from integrator import Step
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
    from baseline import Block
    torch.manual_seed(1)
    B, T, d, H, Hkv, hid = 2, 11, 32, 4, 2, 64
    u = torch.randn(B, T, d)
    pos = torch.arange(T).expand(B, T)
    ctx = Context(pos)
    lie = Step(d, H, hid, Hkv, "lie")
    blk = Block(d, H, hid, Hkv)
    with torch.no_grad():
        for p in lie.parameters():
            if p.dim() == 1:
                p.normal_(1, 0.2)                                       # izdusum kazanclari 1 olmasin (kopya sinansin)
        blk.qkv.weight.copy_(torch.cat([lie.A.w_q.weight, lie.A.w_kv.weight]))
        blk.proj.weight.copy_(lie.A.w_o.weight)
        blk.gate_up.weight.copy_(lie.F.w_gate_up.weight)
        blk.down.weight.copy_(lie.F.w_down.weight)
        for dst, src in ((blk.n1, lie.A.P), (blk.n2, lie.F.P), (blk.q_norm, lie.A.q_P), (blk.k_norm, lie.A.k_P)):
            dst.weight.copy_(src.g)
            dst.eps = src.eps
        d_lie = float((lie(u, ctx) - blk(u, pos, None)).abs().max())
    check("lie adimi = bugunku transformer blogu (baseline.Block, GQA, fp32)", d_lie < 1e-5, "fark %.1e" % d_lie)
    st = Step(d, H, hid, Hkv, "strang")
    n = lambda m: sum(p.numel() for p in m.parameters() if p.dim() == 2)   # noqa: E731  matrisler (izdusum kazanci F2'de +d)
    with torch.no_grad():
        v = u + 0.5 * st.F1(u)
        v = v + st.A(v, ctx)
        ref = v + 0.5 * st.F2(v)
        d_st = float((st(u, ctx) - ref).abs().max())
    pa = Step(d, H, hid, Hkv, "parallel")
    with torch.no_grad():
        d_pa = float((pa(u, ctx) - (u + pa.A(u, ctx) + pa.F(u))).abs().max())
        pa.load_state_dict(lie.state_dict())
        st_out = st(u, ctx)
        diff_lp = float((lie(u, ctx) - pa(u, ctx)).abs().max())
    check("strang = elle F1/2 -> A -> F2/2, matris parametresi lie ile ayni (%d)" % n(lie), d_st < 1e-6 and n(st) == n(lie),
          "fark %.1e, strang %d" % (d_st, n(st)))
    check("parallel = elle u + A(u) + F(u); ayni agirlikla lie'den farkli (sira onemli)", d_pa < 1e-6 and diff_lp > 1e-4,
          "fark %.1e, lie - parallel %.1e" % (d_pa, diff_lp))


def t_stages():
    """Parca 3: k = 1 asamalar = sirali adimlar; k = 2 asama = u + D1(u) + D2(u); kucuk adimda hibrit ile sirali farki adim
    boyunun karesiyle kuculur (BCH: fark / eps^2 sabit)."""
    from integrator import Stage, Step, stages
    torch.manual_seed(2)
    B, T, d, H, hid = 1, 13, 32, 4, 64
    u = torch.randn(B, T, d) * 3
    ctx = Context(torch.arange(T).expand(B, T))
    seq = stages(d, H, hid, (1, 1, 1, 1))
    hyb = stages(d, H, hid, (2, 2))
    with torch.no_grad():
        flat = [s.steps[0] for s in seq]
        for i, st in enumerate([st for s in hyb for st in s.steps]):
            st.load_state_dict(flat[i].state_dict())
        x = u
        for s in seq:
            x = s(x, ctx)
        y = u
        for st in flat:
            y = st(y, ctx)
        d_seq = float((x - y).abs().max())
        s0 = hyb[0]
        d_par = float((s0(u, ctx) - (u + (s0.steps[0](u, ctx) - u) + (s0.steps[1](u, ctx) - u))).abs().max())
        ratios = []
        for eps in (0.2, 0.1, 0.05):
            a, b = Step(d, H, hid, scheme="lie", dt=eps), Step(d, H, hid, scheme="lie", dt=eps)
            torch.manual_seed(3)
            a.load_state_dict(flat[0].state_dict()), b.load_state_dict(flat[1].state_dict())
            sq = b(a(u, ctx), ctx)
            pr = Stage([a, b])(u, ctx)
            ratios.append(float((sq - pr).norm() / u.norm()) / eps ** 2)
        a, b, w = Step(d, H, hid, 2), Step(d, H, hid, 2), Step(d, H, hid, 2, width=2)   # GQA (kv 2)
        kvr = a.A.kv_heads * a.A.hd
        cat0 = lambda f: torch.cat([f(a), f(b)], 0)                       # noqa: E731
        w.A.w_q.weight.copy_(cat0(lambda s: s.A.w_q.weight))
        w.A.w_kv.weight.copy_(torch.cat([a.A.w_kv.weight[:kvr], b.A.w_kv.weight[:kvr],
                                         a.A.w_kv.weight[kvr:], b.A.w_kv.weight[kvr:]]))
        w.A.w_o.weight.copy_(torch.cat([a.A.w_o.weight, b.A.w_o.weight], 1))
        gu = [s.F.w_gate_up.weight.chunk(2) for s in (a, b)]
        w.F.w_gate_up.weight.copy_(torch.cat([gu[0][0], gu[1][0], gu[0][1], gu[1][1]]))
        w.F.w_down.weight.copy_(torch.cat([a.F.w_down.weight, b.F.w_down.weight], 1))
        u1 = u + a.A(u, ctx) + b.A(u, ctx)
        ref = u1 + a.F(u1) + b.F(u1)
        d_wide = float((w(u, ctx) - ref).abs().max())
        n_ab = sum(p.numel() for s in (a, b) for p in s.parameters() if p.dim() == 2)
        n_w = sum(p.numel() for p in w.parameters() if p.dim() == 2)
    check("hibrit: Step(width 2) = wide formulu u1 = u + A1 + A2, u1 + F1(u1) + F2(u1) (GQA)", d_wide < 1e-5 and n_w == n_ab,
          "fark %.1e, matris %d / %d" % (d_wide, n_w, n_ab))
    check("hibrit: k = 1 asamalar = sirali adimlar", d_seq == 0, "fark %.1e" % d_seq)
    check("hibrit: k = 2 asama = u + D1(u) + D2(u)", d_par < 1e-6, "fark %.1e" % d_par)
    check("hibrit: sirali - hibrit farki eps^2 ile kuculur (BCH)", max(ratios) / min(ratios) < 1.3,
          "fark / eps^2: %s" % ", ".join("%.4f" % r for r in ratios))


def t_model():
    """Parca 4: widths hepsi 1 = BaselineTransformer (agirlik kopyasi, kayip ve uretim ayni); genis model: sinav kaybi
    (loss_per_target) ve toplu uretim (StaticCache) = token token KV cache uretimi; FLOP birimi sum(widths)."""
    from model import ModelS
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
    import baseline as BL
    torch.manual_seed(4)
    d, H, Hkv = 32, 4, 2
    tf = BL.BaselineTransformer(d, 3, H, Hkv)
    ms = ModelS(d, (1, 1, 1), H, Hkv)
    with torch.no_grad():
        ms.E.weight.copy_(tf.E.weight)
        ms.norm.weight.copy_(tf.norm.weight)
        for blk, sb in zip(tf.blocks, ms.blocks):
            for p in sb.parameters():
                if p.dim() == 1:
                    p.normal_(1, 0.2)
            blk.qkv.weight.copy_(torch.cat([sb.A.w_q.weight, sb.A.w_kv.weight]))
            blk.proj.weight.copy_(sb.A.w_o.weight)
            blk.gate_up.weight.copy_(sb.F.w_gate_up.weight)
            blk.down.weight.copy_(sb.F.w_down.weight)
            for dst, src in ((blk.n1, sb.A.P), (blk.n2, sb.F.P), (blk.q_norm, sb.A.q_P), (blk.k_norm, sb.A.k_P)):
                dst.weight.copy_(src.g)
                dst.eps = src.eps
    toks = torch.randint(0, 1000, (2, 17))
    pos = torch.arange(17).expand(2, 17)
    with torch.no_grad():
        d_h = float((tf.hidden(toks, pos, None) - ms.hidden(toks, pos, None)).abs().max())
    check("model: widths (1, 1, 1) = BaselineTransformer (ayni agirlik, fp32)", d_h < 1e-5, "fark %.1e" % d_h)
    wide = ModelS(d, (1, 2, 3), H, Hkv).eval()
    with torch.no_grad():
        wide.E.weight[[BL.END_ID, BL.EOS_ID]] = 0                         # rastgele model hemen END / EOS uretmesin
    prompts = [[[5, 6, 7], [8, 9]], [[11, 12, 13, 14]], [[3]]]
    one = [wide.generate([p], 2, 5, batch_size=1)[0] for p in prompts]
    rows = wide.generate(prompts, 2, 5, batch_size=8)
    check("model: genis (1, 2, 3) toplu uretim (StaticCache) = tek tek KV cache uretimi", one == rows, "%s" % (rows[0],))
    with torch.no_grad():
        x = torch.randint(0, 1000, (1, 9))
        full = wide._logits(wide.hidden(x, torch.arange(9)[None], None))[0, -1]
        cache = dict(k=[None] * 3, v=[None] * 3, T=16)
        wide._logits(wide._step(x[:, :5], cache, 0)[None])
        step = wide._logits(wide._step(x[:, 5:], cache, 5)[None])[0]
    d_c = float((full - step).abs().max())
    n2 = lambda m: sum(p.numel() for p in m.blocks.parameters() if p.dim() == 2)   # noqa: E731
    check("model: KV cache adimi = tam ileri (son konum logit); matris = 6 x tek adim", d_c < 1e-4 and
          n2(wide) == 6 * n2(ModelS(d, (1,), H, Hkv)), "fark %.1e" % d_c)


def t_mix():
    """MixOp (tek operator): linear values = parallel adim (ayni agirlik); swiglu = elle formul (nedensel softmax, GQA, RoPE);
    nedensel; esit matris parametresi (64'e yuvarlama payi icinde); ModelS op mix: KV cache adimi = tam ileri, toplu uretim =
    tek tek; widths ile genis mix asamasi."""
    from integrator import Step
    from model import ModelS
    from operators import MixOp
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
    import baseline as BL
    torch.manual_seed(5)
    B, T, d, H, Hkv, hid = 2, 11, 32, 4, 2, 64
    u = torch.randn(B, T, d)
    pos = torch.arange(T).expand(B, T)
    ctx = Context(pos)
    pa, mx = Step(d, H, hid, Hkv, "parallel"), Step(d, H, hid, Hkv, "mix", values="linear")
    with torch.no_grad():
        g = torch.randn(d) * 0.2 + 1
        pa.A.P.g.copy_(g), pa.F.P.g.copy_(g), mx.M.P.g.copy_(g)
        pa.A.q_P.g.normal_(1, 0.2), pa.A.k_P.g.normal_(1, 0.2)
        mx.M.q_P.g.copy_(pa.A.q_P.g), mx.M.k_P.g.copy_(pa.A.k_P.g)
        kvr = Hkv * (d // H)
        mx.M.w_in.weight.copy_(torch.cat([pa.A.w_q.weight, pa.A.w_kv.weight[:kvr], pa.A.w_kv.weight[kvr:], pa.F.w_gate_up.weight]))
        mx.M.w_out.weight.copy_(torch.cat([pa.A.w_o.weight, pa.F.w_down.weight], 1))
        d_lin = float((pa(u, ctx) - mx(u, ctx)).abs().max())
    check("mix linear = parallel adim (u + A(Pu) + F(Pu), ayni agirlik, GQA)", d_lin < 1e-5, "fark %.1e" % d_lin)
    M = MixOp(d, H, 48, Hkv, values="swiglu")
    with torch.no_grad():
        for p in (M.P.g, M.q_P.g, M.k_P.g):
            p.normal_(1, 0.2)
        got = M(u, ctx)
        hd = d // H
        q, k, v, loc = M.w_in(M.P(u)).split(M.split, -1)
        vg, vu = v.chunk(2, -1)
        v = torch.nn.functional.silu(vg) * vu
        q = M.q_P(q.view(B, T, H, hd)).permute(0, 2, 1, 3)
        k = M.k_P(k.view(B, T, Hkv, hd)).permute(0, 2, 1, 3)
        v = v.view(B, T, Hkv, hd).permute(0, 2, 1, 3)
        q, k = rope(q, pos), rope(k, pos)
        k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
        s = (q @ k.transpose(-1, -2)) / hd ** 0.5
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, d)
        gl, ul = loc.chunk(2, -1)
        ref = M.w_out(torch.cat([a, torch.nn.functional.silu(gl) * ul], -1))
        d_sw = float((got - ref).abs().max())
        u2 = u.clone()
        u2[:, 7] += 1.0
        d_c = float((M(u2, ctx)[:, :7] - got[:, :7]).abs().max())
    check("mix swiglu = elle formul (W_out [softmax(QK) silu(Vg) Vu | silu(g) up]); nedensel", d_sw < 1e-5 and d_c == 0,
          "fark %.1e, nedensellik %.1e" % (d_sw, d_c))
    n2 = lambda m: sum(p.numel() for p in m.parameters() if p.dim() == 2)  # noqa: E731
    lie, mixs = Step(64, 4, 176, None, "lie"), Step(64, 4, 176, None, "mix")
    check("mix swiglu matris parametresi = lie (yerel genislik 64'e asagi yuvarlanir: fark < 3 d 64)",
          0 <= n2(lie) - n2(mixs) < 3 * 64 * 64, "lie %d, mix %d" % (n2(lie), n2(mixs)))
    for op in ("mix", "parallel"):
        wide = ModelS(d, (1, 2), H, Hkv, op=op).eval()
        with torch.no_grad():
            wide.E.weight[[BL.END_ID, BL.EOS_ID]] = 0
            x = torch.randint(0, 1000, (1, 9))
            full = wide._logits(wide.hidden(x, torch.arange(9)[None], None))[0, -1]
            cache = dict(k=[None] * 2, v=[None] * 2, T=16)
            wide._step(x[:, :5], cache, 0)
            d_k = float((full - wide._logits(wide._step(x[:, 5:], cache, 5)[None])[0]).abs().max())
        prompts = [[[5, 6, 7], [8, 9]], [[11, 12, 13, 14]], [[3]]]
        one = [wide.generate([p], 2, 5, batch_size=1)[0] for p in prompts]
        check("model op %s (widths 1, 2): KV cache adimi = tam ileri; toplu uretim = tek tek" % op,
              d_k < 1e-4 and one == wide.generate(prompts, 2, 5, batch_size=8), "fark %.1e" % d_k)


def t_fold():
    """Parca 7, P katlama: PROJ_FOLD acik = kapali (ayni agirlik, fp32): lie / mix swiglu / mix linear adim ciktisi ve butun
    parametre gradyanlari; q, k'de s dusmesi QK izdusumunun olcekten bagimsizligina dayanir (fark yalniz eps)."""
    import operators as O
    from integrator import Step
    torch.manual_seed(6)
    B, T, d, H, Hkv, hid = 2, 9, 32, 4, 2, 64
    u = torch.randn(B, T, d) * 5
    ctx = Context(torch.arange(T).expand(B, T))
    for name, kw in (("lie", dict(scheme="lie")), ("mix swiglu", dict(scheme="mix")), ("mix linear", dict(scheme="mix", values="linear"))):
        st = Step(d, H, hid, Hkv, **kw)
        with torch.no_grad():
            for p in st.parameters():
                if p.dim() == 1:
                    p.normal_(1, 0.3)
        res = []
        for fold in (False, True):
            O.PROJ_FOLD = fold
            st.zero_grad()
            y = st(u, ctx)
            (y * torch.linspace(-1, 1, d)).sum().backward()
            res.append((y.detach(), [p.grad.clone() for p in st.parameters()]))
        O.PROJ_FOLD = False
        d_y = float((res[0][0] - res[1][0]).abs().max())
        d_g = max(float((a - b).abs().max() / (a.abs().max() + 1e-12)) for a, b in zip(res[0][1], res[1][1]))
        check("P katlama %s: cikis ve gradyanlar = katlamasiz (fp32)" % name, d_y < 1e-4 and d_g < 1e-4,
              "cikis %.1e, gradyan (goreli) %.1e" % (d_y, d_g))


def t_qk():
    """Parca 8, qk_dim < head boyu: A = elle formul (q / k qk boyutunda, softmax olcegi 1 / sqrt(qk)); matris azalmasi tam
    q / k kadar; split ve mix modelinde KV cache adimi = tam ileri, toplu uretim (StaticCache) = tek tek."""
    from model import ModelS
    from operators import AttentionOp
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
    import baseline as BL
    torch.manual_seed(7)
    B, T, d, H, Hkv, qk = 2, 9, 32, 4, 2, 4
    u = torch.randn(B, T, d)
    pos = torch.arange(T).expand(B, T)
    A = AttentionOp(d, H, Hkv, qk_dim=qk)
    with torch.no_grad():
        got = A(u, Context(pos))
        h = A.P(u)
        q = A.q_P(A.w_q(h).view(B, T, H, qk)).permute(0, 2, 1, 3)
        kv = A.w_kv(h)
        k = A.k_P(kv[..., :Hkv * qk].view(B, T, Hkv, qk)).permute(0, 2, 1, 3)
        v = kv[..., Hkv * qk:].view(B, T, Hkv, d // H).permute(0, 2, 1, 3)
        q, k = rope(q, pos), rope(k, pos)
        k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
        s = (q @ k.transpose(-1, -2)) / qk ** 0.5
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        ref = A.w_o((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, d))
        d_a = float((got - ref).abs().max())
    full = AttentionOp(d, H, Hkv)
    n2 = lambda m: sum(p.numel() for p in m.parameters() if p.dim() == 2)  # noqa: E731
    check("qk_dim: A = elle formul (q / k %d boyut, olcek 1 / sqrt(qk)); matris azalmasi = (H + Hkv) (hd - qk) d" % qk,
          d_a < 1e-5 and n2(full) - n2(A) == (H + Hkv) * (d // H - qk) * d, "fark %.1e" % d_a)
    for op in ("split", "mix"):
        m = ModelS(d, (1, 2), H, Hkv, op=op, qk_dim=qk).eval()
        with torch.no_grad():
            m.E.weight[[BL.END_ID, BL.EOS_ID]] = 0
            x = torch.randint(0, 1000, (1, 9))
            fl = m._logits(m.hidden(x, torch.arange(9)[None], None))[0, -1]
            cache = dict(k=[None] * 2, v=[None] * 2, T=16)
            m._step(x[:, :5], cache, 0)
            d_k = float((fl - m._logits(m._step(x[:, 5:], cache, 5)[None])[0]).abs().max())
        prompts = [[[5, 6, 7], [8, 9]], [[11, 12, 13, 14]], [[3]]]
        one = [m.generate([p], 2, 5, batch_size=1)[0] for p in prompts]
        check("qk_dim model op %s: KV cache adimi = tam ileri; toplu uretim = tek tek" % op,
              d_k < 1e-4 and one == m.generate(prompts, 2, 5, batch_size=8), "fark %.1e" % d_k)


def t_gate():
    """Parca 9, gate-once FFN: keep 1 = yogun (bit); keep 0,5 = elle formul (token basina |silu(g)|^2 blok enerjisi en yuksek
    yarisi, gerisi 0) ve token basina tam keep x blok sayisi kadar blok acik; split ve mix modelinde ilk / son adim yogun,
    KV cache adimi = tam ileri."""
    import torch.nn.functional as F_
    from model import ModelS
    from operators import FeedForwardOp
    torch.manual_seed(8)
    d, hid = 32, 128
    u = torch.randn(2, 7, d)
    dense, sp = FeedForwardOp(d, hid), FeedForwardOp(d, hid, keep=0.5)
    sp.load_state_dict(dense.state_dict())
    with torch.no_grad():
        g, up = dense.w_gate_up(dense.P(u)).chunk(2, -1)
        ga = F_.silu(g)
        e = ga.view(2, 7, 2, 64).pow(2).sum(-1)                          # 128 = 2 x 64: token basina 1 blok
        m = (e == e.max(-1, keepdim=True).values).float().repeat_interleave(64, -1)
        ref = dense.w_down(ga * up * m)
        d_s = float((sp(u) - ref).abs().max())
        on = float(((ga * up * m) != 0).view(2, 7, 2, 64).any(-1).sum(-1).float().mean())
        ref1 = dense.w_down(ga * up)
    check("gate-once: keep 1 = yogun; keep 0,5 = elle formul (en yuksek gate enerjili blok), token basina 1 / 2 blok acik",
          torch.equal(dense(u), ref1) and d_s < 1e-6 and on == 1.0, "fark %.1e, acik blok %.2f" % (d_s, on))
    for op in ("split", "mix"):
        mm = ModelS(64, (1, 1, 1), 4, op=op, ffn_keep=0.5).eval()
        ks = [(b.M.keep if b.scheme == "mix" else b.F.keep) for b in mm.blocks]
        with torch.no_grad():
            x = torch.randint(0, 1000, (1, 9))
            fl = mm._logits(mm.hidden(x, torch.arange(9)[None], None))[0, -1]
            cache = dict(k=[None] * 3, v=[None] * 3, T=16)
            mm._step(x[:, :5], cache, 0)
            d_k = float((fl - mm._logits(mm._step(x[:, 5:], cache, 5)[None])[0]).abs().max())
        check("gate-once model op %s: keep ilk / son adimda 1, arada 0,5; KV cache adimi = tam ileri" % op,
              ks == [1.0, 0.5, 1.0] and d_k < 1e-4, "keep %s, fark %.1e" % (ks, d_k))


def t_out():
    """Parca 10, carpanli cikis: siklik sirasi / ters sira; tam log p normlu (toplam 1); loss = -ortalama log p[hedef] (kuyruk
    satir secimi dahil, -100 atlanir); geri yayilim sonlu ve butun cikis parametrelerine ulasir; ModelS: KV cache adimi = tam
    ileri, toplu uretim = tek tek."""
    from model import ModelS
    from output import FactorizedOutput
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
    import baseline as BL
    torch.manual_seed(9)
    d, V = 32, 300
    fo = FactorizedOutput(d, V, cuts=(40, 120), dims=(16, 8))
    counts = torch.randint(0, 1000, (V,)).float()
    fo.set_order(counts)
    ok_order = bool((counts[fo.order][:-1] >= counts[fo.order][1:]).all()) and torch.equal(fo.order[fo.rank], torch.arange(V))
    E = torch.randn(V + 4, d, requires_grad=True)
    h = torch.randn(23, d)
    lp = fo.logprobs(h, E)
    tot = float((lp.exp().sum(-1) - 1).abs().max())
    tgt = torch.randint(0, V, (23,))
    tgt[[3, 7]] = -100
    loss = fo.loss(h, E, tgt)
    keep = tgt >= 0
    ref = -lp[keep].gather(-1, tgt[keep][:, None]).mean()
    loss.backward()
    grads = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in fo.parameters()) and E.grad is not None
    hit = [int(((fo.rank[tgt[keep]] >= a) & (fo.rank[tgt[keep]] < b)).sum()) for a, b in zip(fo.edges, fo.edges[1:])]
    check("carpanli cikis: siklik sirasi ve ters sira; tam log p toplami 1; loss = -ort log p[hedef] (-100 atlanir); gradyan",
          ok_order and tot < 1e-5 and abs(float(loss) - float(ref)) < 1e-5 and grads and min(hit) > 0,
          "toplam sapma %.1e, loss %.5f ref %.5f, kume isabeti %s" % (tot, float(loss), float(ref), hit))
    m = ModelS(64, (1, 2), 4, out_cuts=(4096, 16384)).eval()
    m.out.set_order(torch.randint(0, 1000, (BL.VOCAB,)).float())
    with torch.no_grad():
        m.E.weight[[BL.END_ID, BL.EOS_ID]] = 0
        x = torch.randint(0, 1000, (1, 9))
        fl = m._logits(m.hidden(x, torch.arange(9)[None], None))[0, -1]
        cache = dict(k=[None] * 2, v=[None] * 2, T=16)
        m._step(x[:, :5], cache, 0)
        d_k = float((fl - m._logits(m._step(x[:, 5:], cache, 5)[None])[0]).abs().max())
        s1 = float((fl.exp().sum() - 1).abs())
    prompts = [[[5, 6, 7], [8, 9]], [[11, 12, 13, 14]], [[3]]]
    one = [m.generate([p], 2, 5, batch_size=1)[0] for p in prompts]
    check("carpanli cikis ModelS: log p toplami 1; KV cache adimi = tam ileri; toplu uretim = tek tek",
          s1 < 1e-4 and d_k < 1e-4 and one == m.generate(prompts, 2, 5, batch_size=8), "toplam sapma %.1e, fark %.1e" % (s1, d_k))


def t_strata_x():
    """Strata-x: birim sinirlari = elle (belge ici floor(C / theta)); nedensellik (t'den sonrasi degisince <= t ayni: h ve sinirlar);
    belge yalitimi (satirda iki belge = ikinci belge tek basina); output_loss degeri = ana CE, gradyan yardimci CE'yi de tasir;
    loss_per_target ve uretim calisir."""
    from strata_x import StrataX
    torch.manual_seed(10)
    m = StrataX(d_local=32, heads_local=4, enc_layers=1, dec_layers=1, d_global=64, heads_global=4, global_layers=2,
                theta=20.0, window=6)
    m.out.set_order(torch.randint(0, 1000, (m.out.vocab,)).float())
    info = torch.rand(2, 12) * 9
    doc = torch.tensor([[0] * 7 + [1] * 5, [0] * 10 + [-1] * 2])
    ends = m.units(info, doc)
    ref = torch.zeros_like(ends)
    for b in range(2):
        acc, last_q, cur = 0.0, 0, None
        for t in range(12):
            d_ = int(doc[b, t])
            if d_ < 0:
                continue
            if d_ != cur:
                cur, acc, last_q = d_, 0.0, 0
            acc += float(info[b, t])
            q = math.floor(acc / 20.0)
            ref[b, t] = q > last_q
            last_q = q
    check("strata-x: birim sinirlari = elle (belge ici floor(C / theta), dolgu yok)", torch.equal(ends, ref), "%s" % ends.int().tolist())
    m.eval()
    T = 16
    a = torch.randint(0, 1000, (1, T))
    b = a.clone()
    b[0, 9:] = torch.randint(0, 1000, (T - 9,))
    pos = torch.arange(T)[None]
    z = torch.zeros(1, T, dtype=torch.long)
    with torch.no_grad():
        ha, _, ea, ia = m.forward_states(a, pos, z)
        hb, _, eb, ib = m.forward_states(b, pos, z)
    d_c = float((ha[0, :9] - hb[0, :9]).abs().max())
    check("strata-x: nedensellik (konum 9'dan sonrasi degisince 0-8 ayni: durum, sinir, bilgi); birim / token %.2f" % (ea.float().mean()),
          d_c < 1e-6 and torch.equal(ea[0, :9], eb[0, :9]) and torch.equal(ia[0, :9], ib[0, :9]), "fark %.1e" % d_c)
    t2 = torch.randint(0, 1000, (1, 7))
    row = torch.cat([a[:, :9], t2], 1)
    drow = torch.tensor([[0] * 9 + [1] * 7])
    prow = torch.cat([torch.arange(9), torch.arange(7)])[None]
    with torch.no_grad():
        hr = m.forward_states(row, prow, drow)[0]
        hs = m.forward_states(t2, torch.arange(7)[None], torch.zeros(1, 7, dtype=torch.long))[0]
    d_d = float((hr[0, 9:] - hs[0]).abs().max())
    check("strata-x: belge yalitimi (satirdaki ikinci belge = tek basina)", d_d < 1e-5, "fark %.1e" % d_d)
    m.train()
    tok = torch.randint(0, 1000, (2, 12))
    tgt = torch.roll(tok, -1, 1)
    tgt[:, -1] = -100
    B_ = type("B", (), dict(tokens=tok, pos=torch.arange(12).expand(2, 12), doc=torch.zeros(2, 12, dtype=torch.long)))()
    h = m._batch_hidden(B_)
    loss = m.output_loss(h.flatten(0, 1), tgt.flatten())
    main = m._nll(h.flatten(0, 1)[tgt.flatten() >= 0], tgt.flatten()[tgt.flatten() >= 0]).mean()
    loss.backward()
    g1 = m.blocks[0].A.w_q.weight.grad.clone()
    m.zero_grad()
    m.aux_weight = 0.0
    h = m._batch_hidden(B_)
    m.output_loss(h.flatten(0, 1), tgt.flatten()).backward()
    g0 = m.blocks[0].A.w_q.weight.grad.clone()
    m.aux_weight = 1.0
    check("strata-x: output_loss degeri = ana CE; yardimci CE kodlayici gradyanina girer",
          abs(float(loss) - float(main)) < 1e-6 and float((g1 - g0).abs().max()) > 1e-8, "loss %.4f ana %.4f" % (float(loss), float(main)))
    mp = StrataX(d_local=32, heads_local=4, enc_layers=1, dec_layers=1, d_global=64, heads_global=4, global_layers=2,
                 theta=20.0, window=6, plan=3, next_weight=0.5).eval()
    with torch.no_grad():
        pa_ = mp.forward_states(a, pos, z)[0]
        pb_ = mp.forward_states(b, pos, z)[0]
        hr2 = mp.forward_states(row, prow, drow)[0]
        hs2 = mp.forward_states(t2, torch.arange(7)[None], torch.zeros(1, 7, dtype=torch.long))[0]
    mp.train()
    hh = mp._batch_hidden(B_)
    mp.output_loss(hh.flatten(0, 1), tgt.flatten()).backward()
    gn = mp.pred_next.weight.grad
    check("strata-x plan 3 + sonraki birim: nedensellik, belge yalitimi; pred_next gradyan alir",
          float((pa_[0, :9] - pb_[0, :9]).abs().max()) < 1e-6 and float((hr2[0, 9:] - hs2[0]).abs().max()) < 1e-5
          and gn is not None and float(gn.abs().sum()) > 0, "")
    m.eval()
    out = m.generate([[[5, 6, 7]], [[8]]], 2, 4)
    check("strata-x: uretim calisir (tek tek, tam yeniden hesap)", len(out) == 2 and all(len(o) == 3 for o in out), str(out[0])[:80])


def t_train():
    """Parca 5: train.py --model model_s --widths 1,2 (CPU, d 16; tests_v2'nin kucuk verisi): kayip duser; kimlikte widths,
    layers = adim sayisi; kesilip --resume ile surdurulen = kesintisiz (agent.pt bit duzeyinde, widths kimlikten);
    generate_readings.load_run ayni agirlik; --widths model_s disinda ve model_s'de eksikse DUR."""
    import tests_v2 as TV
    import train as TR
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "diag"))
    import generate_readings as GR
    tp = TV.tokenizer_path()
    if tp is None:
        print("ATLA train: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = TV._train_root(tp)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=3, max_tokens=4)
    out = lambda name: os.path.join(TV.TMP, "runs", name)                 # noqa: E731
    base = ["--data", data, "--stream", root, "--device", "cpu", "--d", "16", "--heads", "2", "--lr", "1e-2",
            "--checkpoint_minutes", "0", "--model", "model_s", "--widths", "1,2", "--epochs", "2"]
    a = TR.main(base + ["--out", out("s_A")])
    L = [w["loss"] for w in a["log"]]
    idt = torch.load(os.path.join(out("s_A"), "agent.pt"), weights_only=False)
    check("train model_s: 2 epok kosar, kayip duser; kimlik widths [1, 2], layers 2",
          sum(L[-3:]) / 3 < sum(L[:3]) / 3 - 0.5 and idt["identity"]["widths"] == [1, 2] and idt["identity"]["layers"] == 2,
          "kayip %.3f -> %.3f" % (sum(L[:3]) / 3, sum(L[-3:]) / 3))
    cut = base[:base.index("--widths")] + base[base.index("--widths") + 2:]
    TR.main(base + ["--out", out("s_B"), "--stop_step", "5"])
    TR.main(cut + ["--out", out("s_B"), "--resume", "1"])
    sa, sb = (torch.load(os.path.join(out(n), "agent.pt"), weights_only=False)["state"] for n in ("s_A", "s_B"))
    check("train model_s: kesilip surdurulen (widths kimlikten) = kesintisiz (bit duzeyinde)",
          sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa))
    m, _ = GR.load_run(out("s_A"), data, torch.device("cpu"))
    check("train model_s: load_run ayni agirlik", all(torch.equal(m.state_dict()[k], sa[k]) for k in sa))
    fo = ["--strata_out", "4096,16384", "--strata_qk", "4", "--strata_ffn_keep", "0.5"]
    o = TR.main(base + fo + ["--out", out("s_O")])
    Lo = [w["loss"] for w in o["log"]]
    TR.main(base + fo + ["--out", out("s_O2"), "--stop_step", "5"])
    TR.main(cut + ["--out", out("s_O2"), "--resume", "1"])
    so, so2 = (torch.load(os.path.join(out(n), "agent.pt"), weights_only=False) for n in ("s_O", "s_O2"))
    rk = so["state"]["out.rank"]
    check("train model_s carpanli cikis + qk 4 + keep 0,5: kayip duser; kimlikte; siklik sirasi END / EOS basta; surdurulen = kesintisiz",
          sum(Lo[-3:]) / 3 < sum(Lo[:3]) / 3 - 0.5 and so["identity"]["strata_out"] == [4096, 16384]
          and so["identity"]["strata_qk"] == 4 and int(rk[50257]) < 2 and int(rk[50256]) < 2
          and all(torch.equal(so["state"][k], so2["state"][k]) for k in so["state"]),
          "kayip %.3f -> %.3f" % (sum(Lo[:3]) / 3, sum(Lo[-3:]) / 3))
    sx = ["--data", data, "--stream", root, "--device", "cpu", "--d", "16", "--heads", "2", "--layers", "1", "--lr", "1e-2",
          "--checkpoint_minutes", "0", "--model", "strata_x", "--sx_local", "16,2,1,1", "--sx_theta", "6", "--sx_window", "8", "--sx_plan", "2", "--sx_next", "0.5",
          "--epochs", "2"]
    x1 = TR.main(sx + ["--out", out("sx_A")])
    Lx = [w["loss"] for w in x1["log"]]
    TR.main(sx + ["--out", out("sx_B"), "--stop_step", "5"])
    i_ = sx.index("--sx_local")
    sxc = sx[:i_] + sx[i_ + 10:]                                          # sx_* kimlikten
    TR.main(sxc + ["--out", out("sx_B"), "--resume", "1"])
    xa, xb = (torch.load(os.path.join(out(n), "agent.pt"), weights_only=False) for n in ("sx_A", "sx_B"))
    mx, _ = GR.load_run(out("sx_A"), data, torch.device("cpu"))
    check("train strata_x: kayip duser; kimlikte sx_*; surdurulen (sx_* kimlikten) = kesintisiz; load_run ayni agirlik",
          sum(Lx[-3:]) / 3 < sum(Lx[:3]) / 3 - 0.5 and xa["identity"]["sx_local"] == [16, 2, 1, 1] and xa["identity"]["sx_theta"] == 6
          and all(torch.equal(xa["state"][k], xb["state"][k]) for k in xa["state"])
          and all(torch.equal(mx.state_dict()[k], xa["state"][k]) for k in xa["state"]),
          "kayip %.3f -> %.3f" % (sum(Lx[:3]) / 3, sum(Lx[-3:]) / 3))
    c = TR.main(base + ["--strata_op", "mix", "--out", out("s_M")])
    Lm = [w["loss"] for w in c["log"]]
    TR.main(base + ["--strata_op", "mix", "--out", out("s_M2"), "--stop_step", "5"])
    TR.main(cut + ["--out", out("s_M2"), "--resume", "1"])
    sm, sm2 = (torch.load(os.path.join(out(n), "agent.pt"), weights_only=False) for n in ("s_M", "s_M2"))
    check("train model_s --strata_op mix: kayip duser, kimlikte mix; surdurulen (op kimlikten) = kesintisiz",
          sum(Lm[-3:]) / 3 < sum(Lm[:3]) / 3 - 0.5 and sm["identity"]["strata_op"] == "mix"
          and all(torch.equal(sm["state"][k], sm2["state"][k]) for k in sm["state"]),
          "kayip %.3f -> %.3f" % (sum(Lm[:3]) / 3, sum(Lm[-3:]) / 3))
    bad = [TV._exit_msg(TR.main, base[:-4] + ["--model", "transformer", "--widths", "1,2", "--out", out("s_x1")]),
           TV._exit_msg(TR.main, cut + ["--out", out("s_x2")])]
    check("train model_s: --widths model_s disinda / model_s'de eksik -> DUR", all(m_ and "DUR" in m_ for m_ in bad), str(bad))


TESTS = dict(ops=t_ops, steps=t_steps, stages=t_stages, model=t_model, mix=t_mix, fold=t_fold, qk=t_qk, gate=t_gate, out=t_out, strata_x=t_strata_x, train=t_train)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), " | HATA: " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)
