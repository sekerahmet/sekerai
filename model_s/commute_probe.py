"""commute_probe -- ardisik adimlar yer degistirebilir mi (Model S; kullanici, 11 Ekim: "istediğin herşeye bakabilirsin").

Operator bolmenin hatasi komutatorden gelir: iki ardisik adim S_l, S_{l+1} birbirinin girdisini az etkiliyorsa
    S_{l+1}(S_l(u))  ~  u + D_l(u) + D_{l+1}(u)      (paralel: ikisi de u'dan)
ve iki adim TEK sirali adimda hesaplanabilir.  Egitilmis transformer'da her (l, l+1) cifti tek tek paralellestirilir, sinav
satirlarinda kayip artisi olculur (geri kalan katmanlar ayni).  Kucuk artis = o cift neredeyse yer degistirir.

Asama (Parca 3, hibrit): --groups "1,1,2,2,2,2;1,1,8" -- asama basina katman sayisi; asamadaki katmanlar ayni girdiden:
    sum   u + sum_i (S_i(u) - u)                       (integrator.Stage)
    wide  u1 = u + sum_i A_i(u);  u2 = u1 + sum_i F_i(u1)   (k kat genis tek lie adimi: F'ler butun A'lari gorur)

Uyarlamali adim (--skip): ODE cozuculeri adimi yerel hataya gore secer; token basina artis kucukse o token'in o adimi
atlanabilir mi (FLOP: atlanan token'in F'i ve A sorgusu hesaplanmaz).  skip_loss.

    python commute_probe.py --run <kosu> --data <fineweb_edu_s000> [--rows 4] [--groups ... | --skip]
"""
import torch  # noqa: I001

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "common"))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "diag"))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "transformer"))
import data as D  # noqa: E402
import recipe as R  # noqa: E402
import train as T  # noqa: E402
import generate_readings as GR  # noqa: E402
import gap_v2 as G  # noqa: E402


def delta(blk, x, pos, attn):
    return blk(x, pos, attn) - x


def d_attn(blk, x, pos, attn):
    from baseline import gqa_sdpa
    B, T, d = x.shape
    q, k, v = blk._qkv(x, pos)
    a = gqa_sdpa(q, k, v, attn[:, None])
    return blk.proj(a.transpose(1, 2).reshape(B, T, d))


def d_ffn(blk, x):
    g, u = blk.gate_up(blk.n2(x)).chunk(2, -1)
    return blk.down(torch.nn.functional.silu(g) * u)


@torch.no_grad()
def stage_loss(model, batch, attn, groups, mode):
    """groups: asama basina katman sayisi (toplam L)."""
    x = model.E(batch.tokens)
    blocks = list(model.blocks)
    assert sum(groups) == len(blocks), "groups toplami %d, L %d" % (sum(groups), len(blocks))
    l = 0
    for k in groups:
        grp = blocks[l:l + k]
        l += k
        if k == 1:
            x = grp[0](x, batch.pos, attn)
        elif mode == "sum":
            x = x + sum(delta(b, x, batch.pos, attn) for b in grp)
        else:
            x = x + sum(d_attn(b, x, batch.pos, attn) for b in grp)
            x = x + sum(d_ffn(b, x) for b in grp)
    h = model.norm(x)
    keep = batch.target >= 0
    return float(torch.nn.functional.cross_entropy(model._logits(h[keep]).float(), batch.target[keep]))


@torch.no_grad()
def loss(model, batch, attn, par=None):
    """Kayip; par = l ise l ve l+1 bloklari ayni girdiden (paralel)."""
    x = model.E(batch.tokens)
    blocks = list(model.blocks)
    l = 0
    while l < len(blocks):
        if par is not None and l == par:
            x = x + delta(blocks[l], x, batch.pos, attn) + delta(blocks[l + 1], x, batch.pos, attn)
            l += 2
        else:
            x = blocks[l](x, batch.pos, attn)
            l += 1
    h = model.norm(x)
    keep = batch.target >= 0
    lg = model._logits(h[keep]).float()
    return float(torch.nn.functional.cross_entropy(lg, batch.target[keep]))


@torch.no_grad()
def skip_loss(model, batch, attn, op, frac, rule, start=2, seed=0):
    """Token basina uyarlamali adim (sifir atis): katman >= start'ta op'un ("A", "F", "AF") artisi en kucuk frac token'da
    atlanir.  rule: oracle (bu katmanin |D| / |u|), prev (onceki katmanin ayni op orani: hesaplamadan once bilinir), rand."""
    x = model.E(batch.tokens)
    real = batch.kind != D.Kind.PAD
    gen = torch.Generator().manual_seed(seed)
    prev = {}
    for l, blk in enumerate(model.blocks):
        for name in ("A", "F"):
            du = d_attn(blk, x, batch.pos, attn) if name == "A" else d_ffn(blk, x)
            r = du.norm(dim=-1) / x.norm(dim=-1)
            if l >= start and name in op and frac > 0:
                s = {"oracle": r, "prev": prev.get(name, r), "rand": torch.rand(r.shape, generator=gen)}[rule]
                cut = torch.quantile(s[real].float(), frac)
                du = du * (s > cut)[..., None]
            prev[name] = r
            x = x + du
    h = model.norm(x)
    keep = batch.target >= 0
    return float(torch.nn.functional.cross_entropy(model._logits(h[keep]).float(), batch.target[keep]))


@torch.no_grad()
def qk_bases(model, batch, attn):
    """Head basina q / k ana bilesenleri (QK izdusumunden sonra, RoPE'den once; fit satirlari): katman -> (Uq, Uk) (H, hd, hd)."""
    out, x = [], model.E(batch.tokens)
    real = (batch.kind != D.Kind.PAD)
    for blk in model.blocks:
        q, k = _qk_pre(blk, x)
        bases = []
        for t in (q, k):                                                 # (B, H, T, hd)
            z = t.permute(1, 0, 2, 3)[:, real]                           # (H, N, hd)
            bases.append(torch.linalg.svd(z.float(), full_matrices=False).Vh.transpose(-1, -2))   # (H, hd, hd) sutunlar
        out.append(bases)
        x = blk(x, batch.pos, attn)
    return out


def _qk_pre(blk, x):
    B, T, d = x.shape
    hd = d // blk.heads
    q, k, _ = blk.qkv(blk.n1(x)).split([d, blk.kv_heads * hd, blk.kv_heads * hd], -1)
    q, k = q.view(B, T, blk.heads, hd).transpose(1, 2), k.view(B, T, blk.kv_heads, hd).transpose(1, 2)
    return blk.q_norm(q.float()).to(x.dtype), blk.k_norm(k.float()).to(x.dtype)


@torch.no_grad()
def qk_rank_loss(model, batch, attn, bases, r, start=0):
    """Katman >= start'ta q ve k head basina ilk r ana bilesene izdusurulur (RoPE'den once): kuadraturde d_qk ne kadar dusebilir."""
    from baseline import gqa_sdpa, rope
    x = model.E(batch.tokens)
    for l, blk in enumerate(model.blocks):
        if l < start:
            x = blk(x, batch.pos, attn)
            continue
        B, T, d = x.shape
        q, k = _qk_pre(blk, x)
        Uq, Uk = bases[l][0][:, :, :r].to(q.dtype), bases[l][1][:, :, :r].to(k.dtype)
        q = q @ Uq @ Uq.transpose(-1, -2)
        k = k @ Uk @ Uk.transpose(-1, -2)
        v = blk.qkv(blk.n1(x))[..., d + blk.kv_heads * (d // blk.heads):].view(B, T, blk.kv_heads, -1).transpose(1, 2)
        a = gqa_sdpa(rope(q, batch.pos), rope(k, batch.pos), v, attn[:, None])
        x = blk._finish(x, a)
    h = model.norm(x)
    keep = batch.target >= 0
    return float(torch.nn.functional.cross_entropy(model._logits(h[keep]).float(), batch.target[keep]))


@torch.no_grad()
def inner_loss(model, batch, attn, layers):
    """Katman ici bolme (A ile F ayni girdiden): layers'taki katmanlarda u + A(u) + F(u), oteki katmanlar sirali."""
    x = model.E(batch.tokens)
    for l, blk in enumerate(model.blocks):
        x = x + d_attn(blk, x, batch.pos, attn) + d_ffn(blk, x) if l in layers else blk(x, batch.pos, attn)
    h = model.norm(x)
    keep = batch.target >= 0
    return float(torch.nn.functional.cross_entropy(model._logits(h[keep]).float(), batch.target[keep]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--groups", default="")
    ap.add_argument("--skip", action="store_true", help="token basina uyarlamali adim (skip_loss)")
    ap.add_argument("--inner", action="store_true", help="katman ici A ile F ayni girdiden (inner_loss)")
    ap.add_argument("--qk_rank", action="store_true", help="q / k head basina ilk r ana bilesen (qk_rank_loss)")
    a = ap.parse_args(argv)
    if os.name == "nt":
        T._no_power_throttling()
    dev = torch.device("cpu")
    model, idt = GR.load_run(a.run, a.data, dev)
    model.eval()
    stories = D.TokenStories(a.data, a.data, "valid")
    rows = G.exam_rows(a.data)[:a.rows]
    batch = D.build_batch(stories, rows, "transformer", dev)
    attn = R.dense_mask(batch, R.document_mask)
    base = loss(model, batch, attn)
    print("taban (sirali) kayip %.4f" % base, flush=True)
    L = len(model.blocks)
    if a.qk_rank:
        fit = D.build_batch(stories, G.exam_rows(a.data)[a.rows:a.rows + 4], "transformer", dev)   # taban fit satirlari ayri
        bases = qk_bases(model, fit, R.dense_mask(fit, R.document_mask))
        print("tam (r 64) kontrol: %+.4f" % (qk_rank_loss(model, batch, attn, bases, 64) - base), flush=True)
        for r in (48, 32, 24, 16, 8):
            print("q, k head basina ilk %2d ana bilesen (64'ten): hepsi %+.4f | katman >= 3 %+.4f"
                  % (r, qk_rank_loss(model, batch, attn, bases, r) - base, qk_rank_loss(model, batch, attn, bases, r, 2) - base), flush=True)
        return
    if a.inner:
        for l in range(L):
            print("katman ici A || F, yalniz katman %2d: %+.4f" % (l + 1, inner_loss(model, batch, attn, {l}) - base), flush=True)
        for name, s in (("katman >= 3", set(range(2, L))), ("hepsi", set(range(L)))):
            print("katman ici A || F, %s: %+.4f" % (name, inner_loss(model, batch, attn, s) - base), flush=True)
        return
    if a.skip:
        for op in ("F", "A", "AF"):
            for frac in (0.25, 0.5, 0.75):
                r = {k: skip_loss(model, batch, attn, op, frac, k) for k in ("oracle", "prev", "rand")}
                print("atla %-2s %%%2d (katman >= 3) | oracle %+.4f | prev %+.4f | rand %+.4f"
                      % (op, frac * 100, r["oracle"] - base, r["prev"] - base, r["rand"] - base), flush=True)
        return
    if a.groups:
        for spec in a.groups.split(";"):
            g = [int(t) for t in spec.split(",")]
            r = {m: stage_loss(model, batch, attn, g, m) for m in ("sum", "wide")}
            print("asamalar %-22s sirali derinlik %2d | sum %.4f (%+.4f) | wide %.4f (%+.4f)"
                  % (spec, len(g), r["sum"], r["sum"] - base, r["wide"], r["wide"] - base), flush=True)
        return
    for l in range(L - 1):
        print("katman %2d + %2d paralel: kayip %.4f  artis %+.4f" % (l + 1, l + 2, x := loss(model, batch, attn, l), x - base),
              flush=True)


if __name__ == "__main__":
    main()
