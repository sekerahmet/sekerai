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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--groups", default="")
    ap.add_argument("--skip", action="store_true", help="token basina uyarlamali adim (skip_loss)")
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
