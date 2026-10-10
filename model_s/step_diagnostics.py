"""step_diagnostics -- egitilmis bir transformer'da (transformer/baseline) adim dinamigi: Model S semasinin dayandigi iki varsayim
(kullanici, 11 Ekim: "serbestsin ... olayın matematiğine odaklan"; TASARIM.md).

  1. Etkin adim: katman basina |du| / |u| (artisin duruma orani) ve |u| (durumun buyumesi); A ve F ayri.  Pre-norm'da |u|
     buyudukce oran kuculuyorsa adimlar bosaliyor.
  2. Cekirdek degisimi: ardisik katmanlarin attention cekirdekleri gamma_l (softmax(QK^T)) ne kadar benzer.  Her l+1 head'i icin
     l'nin en yakin head'i (toplam degisim mesafesi TV = 1/2 sum |g - g'|, sorgu basina ortalama); ve dondurulmus cekirdek hatasi
     |gamma_l V_{l+1} - gamma_{l+1} V_{l+1}| / |gamma_{l+1} V_{l+1}| (eslesen head ile).
  3. Yerellik: gamma kutlesinin son w token'da (kendisi dahil) kalan payi, w = 1, 4, 16, 64; ilk token'a (EOS) giden pay.
     Yakin alan (FMM): kutle birkac token'daysa o katmanin A'si kisa bir evrisimdir.

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
def depth_exit(model, stories, rows, dev):
    """5. Derinlik ekseni (token basina kac adim): her katmandan sonra son norm + cikisla erken okuma (logit lens).  Katman l'de
    token "yerlesmis": argmax son katmanin argmax'i ve KL(son || l) < 0,1.  Yerlesme katmani dagilimi + yerlesmis token'larin
    son kayip ortalamasi (kolay mi)."""
    import torch.nn.functional as F_
    batch = D.build_batch(stories, rows, "transformer", dev)
    keep = batch.target >= 0
    T_ = batch.doc.shape[1]
    attn = (batch.doc[:, :, None] == batch.doc[:, None, :]) & torch.ones(T_, T_, dtype=torch.bool).tril()
    x = model.E(batch.tokens)
    logps = []
    for blk in model.blocks:
        x = blk(x, batch.pos, attn)
        logps.append(F_.log_softmax(model._logits(model.norm(x)[keep]).float(), -1))
    fin = logps[-1]
    tgt = batch.target[keep]
    nll = -fin.gather(-1, tgt[:, None])[:, 0]
    am = fin.argmax(-1)
    L = len(logps)
    settled = torch.full((len(tgt),), L, dtype=torch.long)
    for l in range(L - 1, -1, -1):                                       # en erken katman: o katmandan sonra hep yerlesmis
        kl = (fin.exp() * (fin - logps[l])).sum(-1)
        ok = (logps[l].argmax(-1) == am) & (kl < 0.1)
        settled = torch.where(ok & (settled == l + 1), torch.full_like(settled, l), settled)
    print("token %d | son kayip ortalamasi %.3f" % (len(tgt), float(nll.mean())))
    print("yerlesme katmani (1'den; L+1 = hic) | token payi | birikimli | bu token'larin son kaybi")
    cum = 0.0
    for l in range(L + 1):
        m = settled == l
        f = float(m.float().mean())
        cum += f
        print("%6d | %.3f | %.3f | %s" % (l + 1, f, cum, "%.3f" % float(nll[m].mean()) if m.any() else "-"))
    print("katman | argmax = son argmax | KL(son || l) ortanca | argmax = hedef (dogru)")
    for l in range(L):
        kl = (fin.exp() * (fin - logps[l])).sum(-1)
        print("%6d | %.3f | %.3f | %.3f" % (l + 1, float((logps[l].argmax(-1) == am).float().mean()), float(kl.median()),
                                           float((logps[l].argmax(-1) == tgt).float().mean())))
    half = settled < L // 2
    print("yari derinlikte (katman <= %d) yerlesmis: %.3f, son kayiplari %.3f; geri kalan %.3f" %
          (L // 2, float(half.float().mean()), float(nll[half].mean()), float(nll[~half].mean())))


@torch.no_grad()
def ffn_sparsity(model, stories, rows, dev):
    """4. FFN (SwiGLU) seyrekligi: h = silu(g) * up, token basina |h|^2 enerjisinin %90 / %99'u kac birimde (hidden'in payi);
    g < 0 payi (silu yari kapali) ve |silu(g)| < 0,05 payi; down ciktisinin yalniz en buyuk %k birimle yeniden kurulma hatasi.
    Izdusum bakisi (makale 3.3.3: aktivasyon = S2'ye izdusum): az birim cogu isi yapiyorsa seyrek cekirdek / ReLU^2 kaldiraci."""
    import torch.nn.functional as F_
    batch = D.build_batch(stories, rows, "transformer", dev)
    real = batch.kind != D.Kind.PAD
    attn = (batch.doc[:, :, None] == batch.doc[:, None, :]) & torch.ones(batch.doc.shape[1], batch.doc.shape[1], dtype=torch.bool).tril()
    x = model.E(batch.tokens)
    print("katman | hidden | %90 enerji | %99 enerji | g<0 | |silu(g)|<0,05 | ust %10 birimle down hatasi | ust %25")
    for l, blk in enumerate(model.blocks):
        q, k, v = blk._qkv(x, batch.pos)
        from baseline import gqa_sdpa
        x1 = x + blk.proj(gqa_sdpa(q, k, v, attn[:, None]).transpose(1, 2).reshape(*x.shape))
        g, up = blk.gate_up(blk.n2(x1)).chunk(2, -1)
        h = (F_.silu(g) * up)[real].float()                              # (N, hidden)
        e = h.pow(2)
        srt = e.sort(-1, descending=True).values.cumsum(-1) / e.sum(-1, keepdim=True)
        n = h.shape[1]
        f90, f99 = float((srt < 0.9).sum(-1).float().mean() + 1) / n, float((srt < 0.99).sum(-1).float().mean() + 1) / n
        gr = g[real].float()
        neg, off = float((gr < 0).float().mean()), float((F_.silu(gr).abs() < 0.05).float().mean())
        full = blk.down(h.to(x.dtype)).float()
        errs = []
        for frac in (0.10, 0.25):
            kk = int(n * frac)
            idx = h.abs().topk(kk, -1).indices
            hk = torch.zeros_like(h).scatter(-1, idx, h.gather(-1, idx))
            errs.append(float((blk.down(hk.to(x.dtype)).float() - full).norm() / full.norm()))
        print("%6d | %6d | %9.3f | %9.3f | %.3f | %.3f | %.3f | %.3f" % (l + 1, n, f90, f99, neg, off, errs[0], errs[1]), flush=True)
        x = x1 + blk.down(F_.silu(g) * up)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--ffn", action="store_true", help="yalniz FFN seyrekligi (ffn_sparsity)")
    ap.add_argument("--exit", action="store_true", help="yalniz derinlik ekseni: katman katman erken okuma (depth_exit)")
    a = ap.parse_args(argv)
    if os.name == "nt":
        T._no_power_throttling()
    dev = torch.device("cpu")
    model, idt = GR.load_run(a.run, a.data, dev)
    model.eval()
    assert idt["model"] == "transformer", "baseline transformer bekleniyor"
    stories = D.TokenStories(a.data, a.data, "valid")
    rows = G.exam_rows(a.data)[:a.rows]
    if a.ffn:
        return ffn_sparsity(model, stories, rows, dev)
    if a.exit:
        return depth_exit(model, stories, rows, dev)
    L = len(model.blocks)
    stats = {k: np.zeros(L) for k in ("u", "dA", "dF")}
    tv, err = np.zeros(L - 1), np.zeros(L - 1)
    rank_layers, ranks = {0, 2, 5, L - 1}, {}
    WS = (1, 4, 16, 64)
    loc = np.zeros((L, len(WS) + 1))
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
            ri = torch.nonzero(real).flatten()
            gr_ = g[:, ri][:, :, ri]                                         # (H, n, n) gercek token'lar
            n_ = len(ri)
            dist = torch.arange(n_)[:, None] - torch.arange(n_)[None, :]     # sorgu - anahtar
            for j, w in enumerate(WS):
                loc[l, j] += float((gr_ * ((dist >= 0) & (dist < w))).sum(-1).mean()) / len(rows)
            loc[l, -1] += float(gr_[:, :, 0].mean()) / len(rows)
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
    print("\nyerellik: gamma kutlesinin son w token'daki payi (head ve sorgu ortalamasi) | ilk token payi")
    print("katman | " + " | ".join("w %2d" % w for w in WS) + " | ilk")
    for l in range(L):
        print("%6d | " % (l + 1) + " | ".join("%.3f" % v for v in loc[l]))
    n_tok = int(sum((D.build_batch(stories, [r], "transformer", dev).kind[0] != D.Kind.PAD).sum() for r in rows) / len(rows))
    print("\n%%90 enerji icin boyut (d %d, satir basina ~%d token): katman | dA | dF | gamma (head ort)" % (model.E.weight.shape[1], n_tok))
    for l in sorted(rank_layers):
        print("%6d | %4.0f | %4.0f | %6.0f" % (l + 1, np.mean(ranks[(l, "dA")]), np.mean(ranks[(l, "dF")]), np.mean(ranks[(l, "gamma")])))


if __name__ == "__main__":
    main()
