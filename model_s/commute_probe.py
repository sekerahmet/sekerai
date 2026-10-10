"""commute_probe -- ardisik adimlar yer degistirebilir mi (Model S; kullanici, 11 Ekim: "istediğin herşeye bakabilirsin").

Operator bolmenin hatasi komutatorden gelir: iki ardisik adim S_l, S_{l+1} birbirinin girdisini az etkiliyorsa
    S_{l+1}(S_l(u))  ~  u + D_l(u) + D_{l+1}(u)      (paralel: ikisi de u'dan)
ve iki adim TEK sirali adimda hesaplanabilir.  Egitilmis transformer'da her (l, l+1) cifti tek tek paralellestirilir, sinav
satirlarinda kayip artisi olculur (geri kalan katmanlar ayni).  Kucuk artis = o cift neredeyse yer degistirir.

    python commute_probe.py --run <kosu> --data <fineweb_edu_s000> [--rows 4]
"""
import torch  # noqa: I001

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "common"))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "diag"))
import data as D  # noqa: E402
import recipe as R  # noqa: E402
import train as T  # noqa: E402
import generate_readings as GR  # noqa: E402
import gap_v2 as G  # noqa: E402


def delta(blk, x, pos, attn):
    return blk(x, pos, attn) - x


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=4)
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
    for l in range(L - 1):
        print("katman %2d + %2d paralel: kayip %.4f  artis %+.4f" % (l + 1, l + 2, x := loss(model, batch, attn, l), x - base),
              flush=True)


if __name__ == "__main__":
    main()
