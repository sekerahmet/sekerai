"""step_diagnostics -- egitilmis bir transformer'da (transformer/baseline) adim dinamigi: Model S semasinin dayandigi iki varsayim
(kullanici, 11 Ekim: "serbestsin ... olayın matematiğine odaklan"; TASARIM.md).

  1. Etkin adim: katman basina |du| / |u| (artisin duruma orani) ve |u| (durumun buyumesi); A ve F ayri.  Pre-norm'da |u|
     buyudukce oran kuculuyorsa adimlar bosaliyor.
  2. Cekirdek degisimi: ardisik katmanlarin attention cekirdekleri gamma_l (softmax(QK^T)) ne kadar benzer.  Her l+1 head'i icin
     l'nin en yakin head'i (toplam degisim mesafesi TV = 1/2 sum |g - g'|, sorgu basina ortalama); ve dondurulmus cekirdek hatasi
     |gamma_l V_{l+1} - gamma_{l+1} V_{l+1}| / |gamma_{l+1} V_{l+1}| (eslesen head ile).

    python step_diagnostics.py --run <kosu> --data <fineweb_edu_s000> [--rows 2]
"""
import torch  # noqa: I001

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "common"))
sys.path.insert(0, os.path.join(ROOT, "deneme2", "v2", "diag"))
import data as D  # noqa: E402
import train as T  # noqa: E402
import generate_readings as GR  # noqa: E402
import gap_v2 as G  # noqa: E402


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=2)
    a = ap.parse_args(argv)
    if os.name == "nt":
        T._no_power_throttling()
    dev = torch.device("cpu")
    model, idt = GR.load_run(a.run, a.data, dev)
    model.eval()
    assert idt["model"] == "transformer", "baseline transformer bekleniyor"
    stories = D.TokenStories(a.data, a.data, "valid")
    rows = G.exam_rows(a.data)[:a.rows]
    L = len(model.blocks)
    stats = {k: np.zeros(L) for k in ("u", "dA", "dF")}
    tv, err = np.zeros(L - 1), np.zeros(L - 1)
    rank_layers, ranks = {0, 2, 5, L - 1}, {}
    for r in rows:
        batch = D.build_batch(stories, [r], "transformer", dev)
        x = model.E(batch.tokens)
        pos = batch.pos
        mask = T._attn(batch, model.mask_fn if hasattr(model, "mask_fn") else None, False) if False else None
        real = batch.kind[0] != D.Kind.PAD
        grams, vals = [], []
        for l, blk in enumerate(model.blocks):
            q, k, v = blk._qkv(x, pos)
            Tn = q.shape[2]
            H, Hkv = q.shape[1], k.shape[1]
            kk, vv = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
            s = (q.float() @ kk.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
            causal = torch.ones(Tn, Tn, dtype=torch.bool).tril()
            doc = batch.doc[0]
            same = (doc[:, None] == doc[None, :]) & causal
            g = torch.softmax(s.masked_fill(~same, float("-inf")), -1)[0]   # (H, T, T)
            grams.append(g)
            vals.append(vv[0].float())
            a_out = (g @ vv[0].float()).transpose(0, 1).reshape(1, Tn, -1)
            dA = blk.proj(a_out.to(x.dtype))
            x1 = x + dA
            gg, uu = blk.gate_up(blk.n2(x1)).chunk(2, -1)
            dF = blk.down(torch.nn.functional.silu(gg) * uu)
            nu = x[0][real].norm(dim=-1)
            stats["u"][l] += float(nu.mean()) / len(rows)
            stats["dA"][l] += float((dA[0][real].norm(dim=-1) / nu).mean()) / len(rows)
            stats["dF"][l] += float((dF[0][real].norm(dim=-1) / x1[0][real].norm(dim=-1)).mean()) / len(rows)
            if l in rank_layers:                                         # 3. rank: guncellemeler ve cekirdek kac boyutta
                for name, t in (("dA", dA[0][real]), ("dF", dF[0][real])):
                    sv = torch.linalg.svdvals(t.float())
                    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
                    ranks.setdefault((l, name), []).append(int((e < 0.9).sum()) + 1)
                gr = g[:, real][:, :, real]
                for h in range(0, gr.shape[0], 4):
                    sv = torch.linalg.svdvals(gr[h])
                    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
                    ranks.setdefault((l, "gamma"), []).append(int((e < 0.9).sum()) + 1)
            x = x1 + dF
        idx = torch.nonzero(real).flatten()
        for l in range(L - 1):
            g0, g1 = grams[l][:, idx][:, :, idx], grams[l + 1][:, idx][:, :, idx]
            d = 0.5 * (g1[:, None] - g0[None]).abs().sum(-1).mean(-1)           # (H1, H0) sorgu ortalamasi TV
            best = d.argmin(1)
            tv[l] += float(d.min(1).values.mean()) / len(rows)
            v1 = vals[l + 1][:, idx]
            true = g1 @ v1
            froz = g0[best] @ v1
            err[l] += float((froz - true).norm() / true.norm()) / len(rows)
        print("satir %s bitti" % (r[:3],), flush=True)
    print("\nkatman | |u| ort | |dA|/|u| | |dF|/|u| | TV(l, l+1) en yakin head | dondurulmus cekirdek hatasi")
    for l in range(L):
        extra = ("%.3f | %.3f" % (tv[l], err[l])) if l < L - 1 else "-"
        print("%6d | %7.2f | %8.3f | %8.3f | %s" % (l + 1, stats["u"][l], stats["dA"][l], stats["dF"][l], extra))
    n_tok = int(sum((D.build_batch(stories, [r], "transformer", dev).kind[0] != D.Kind.PAD).sum() for r in rows) / len(rows))
    print("\n%%90 enerji icin boyut (d %d, satir basina ~%d token): katman | dA | dF | gamma (head ort)" % (model.E.weight.shape[1], n_tok))
    for l in sorted(rank_layers):
        print("%6d | %4.0f | %4.0f | %6.0f" % (l + 1, np.mean(ranks[(l, "dA")]), np.mean(ranks[(l, "dF")]), np.mean(ranks[(l, "gamma")])))


if __name__ == "__main__":
    main()
