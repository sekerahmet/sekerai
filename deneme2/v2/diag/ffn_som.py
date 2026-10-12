"""ffn_som (prototip) -- MLP noronlari arasinda PPMI komsulugu var mi? (SOM fikri on olcumu; kullanici 12 Ekim: "yap, ölç
bakalım ama fikri öldürme")

Noronun yazdigi token t_i = argmax_v Ec[v] . (g * W_down[:, i]) (ffn_decode statik okuma).  Cift turleri (katman ici):
  komsu   t_j, t_i'nin PPMI k5 komsusu (tablo 2)
  ayni    t_i == t_j (ayni token'i yazan iki noron)
  kontrol rastgele j', t_j' komsunun t_j'si ile ayni siklik kovasinda (siklik / tur etkisi esitlenir)
Okumalar: (1) aktivasyon korelasyonu (Pearson, valid hikayeler, butun konumlar); (2) W_down sutun kosinusu (agirlik uzayi);
(3) W_up satir kosinusu (anahtar uzayi).  Iki model: erken (3.000 adim) ve olgun (tam epok) -> yapi egitimle buyuyor mu.
Sira komsulari (deneme/som'un yaratmasi beklenen yapi; kullanici 12 Ekim: "iki nöron yapısını ölçersin Z3 ve şu ankini"): (i, i+o),
o = 1 / 4 / 16 ve uzak (i, i + H/2): ayni uc okuma + konu tutarliligi = yazdiklari token PPMI komsu ya da ayni olan cift orani.
"""
import torch  # noqa: I001

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("common", "model_z", "diag"):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), p))
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402
import token_trace as TT  # noqa: E402
from ffn_decode import out_weight  # noqa: E402


@torch.no_grad()
def written_tokens(model):
    E = out_weight(model)[:D.VOCAB].float()
    Ec = (E - E.mean(0)) * model.norm.weight.float()
    out = []
    for b in model.blocks:
        Wd = b.down.weight.float()                                       # (d, H)
        t = torch.cat([(Ec @ Wd[:, s:s + 512]).argmax(0) for s in range(0, Wd.shape[1], 512)])
        out.append(t.cpu().numpy())
    return out


def pairs(tok, nbr, cnt, rng, max_pairs=200_000):
    """-> komsu (i, j), ayni (i, j), kontrol (i, j') dizileri."""
    by_tok = {}
    for i, t in enumerate(tok):
        by_tok.setdefault(int(t), []).append(i)
    H = len(tok)
    bucket = np.floor(np.log10(np.maximum(cnt, 1))).astype(int)
    by_bucket = {}
    for i, t in enumerate(tok):
        by_bucket.setdefault(bucket[t], []).append(i)
    nb, same = [], []
    for i, t in enumerate(tok):
        for u in nbr[t]:
            if u >= 0 and u != t:
                nb += [(i, j) for j in by_tok.get(int(u), [])]
        same += [(i, j) for j in by_tok[int(t)] if j > i]
    nb = np.array(nb, dtype=np.int64).reshape(-1, 2)
    same = np.array(same, dtype=np.int64).reshape(-1, 2)
    if len(nb) > max_pairs:
        nb = nb[rng.choice(len(nb), max_pairs, replace=False)]
    if len(same) > max_pairs:
        same = same[rng.choice(len(same), max_pairs, replace=False)]
    ctrl = []
    for i, j in nb:
        cand = by_bucket[bucket[tok[j]]]
        k = cand[rng.integers(len(cand))]
        while k == i and len(cand) > 1:
            k = cand[rng.integers(len(cand))]
        ctrl.append((i, k))
    ctrl = np.array(ctrl, dtype=np.int64).reshape(-1, 2)
    rand = np.stack([rng.integers(H, size=len(nb)), rng.integers(H, size=len(nb))], 1) if len(nb) else nb
    return dict(komsu=nb, ayni=same, kontrol=ctrl, rastgele=rand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="ad=klasor")
    ap.add_argument("--data", required=True)
    ap.add_argument("--pmi", required=True)
    ap.add_argument("--stories", type=int, default=150)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--map", type=int, default=512, help="harita: G katmanlarinda ilk N noronun korelasyon / W_down kosinus "
                    "matrisi ve yazdigi token'lar <out>.npz'e (0 kapali)")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup(a.device, log)
    f = np.load(a.pmi)
    nbr, cnt = f["nbr"][:D.VOCAB], f["counts"][:D.VOCAB]
    tok_, vocab = G.vocab_text(a.data)
    profile = json.load(open(os.path.join(a.data, "valid_boundaries.json"), encoding="utf-8")).get("profile", "ss")
    valid = D.TokenStories(a.data, a.data, "valid")
    items = []
    for i in range(a.stories):
        sents, n = [], 0
        for s_ in valid.sentences(i):
            if n + len(s_) + 1 > a.max_tokens:
                break
            sents.append(s_.tolist())
            n += len(s_) + 1
        if sents:
            items.append(sents)
    st = TT.text_story(tok_, items, profile)
    res = {}
    for spec in a.runs:
        name, run = spec.split("=", 1)
        model = G.load(run, a.data, dev)[0].float().eval()
        L = len(model.blocks)
        H = [b.down.weight.shape[1] for b in model.blocks]
        S1 = [torch.zeros(h, dtype=torch.float64, device=dev) for h in H]
        S2 = [torch.zeros(h, h, dtype=torch.float64, device=dev) for h in H]
        N = 0
        rec = {}
        hooks = [b.down.register_forward_hook(lambda m, ar, o, l=l: rec.__setitem__(l, ar[0][0])) for l, b in enumerate(model.blocks)]
        with torch.no_grad():
            for j, sents in enumerate(items):
                batch = D.build_batch(st, [[j]], "model_z", dev, row_len=1 + sum(len(s_) + 1 for s_ in sents))
                model._batch_hidden(batch)
                keep = batch.kind[0] != int(D.Kind.PAD)
                for l in range(L):
                    A = rec[l][keep].double()
                    S1[l] += A.sum(0)
                    S2[l] += A.T @ A
                N += int(keep.sum())
                if j % 25 == 0:
                    log("%s hikaye %d / %d" % (name, j, len(items)))
        for h in hooks:
            h.remove()
        wt = written_tokens(model)
        rng = np.random.default_rng(0)
        out, maps = [], {}
        for l in range(L):
            mu = S1[l] / N
            cov = S2[l] / N - torch.outer(mu, mu)
            sd = cov.diagonal().clamp_min(1e-12).sqrt()
            corr = (cov / torch.outer(sd, sd)).float().cpu().numpy()
            S2[l] = None
            Wd = model.blocks[l].down.weight.detach().float()
            Wd = (Wd / Wd.norm(dim=0, keepdim=True)).cpu().numpy()
            Wu = model.blocks[l].gate_up.weight.detach().float()[:H[l]]          # SiLU kapisi satirlari (anahtar)
            Wu = (Wu / Wu.norm(dim=1, keepdim=True)).cpu().numpy()
            P = pairs(wt[l], nbr, cnt, rng)
            row = dict(kind="G" if model.is_glob[l] else "Y", H=H[l], uniq_tokens=int(len(set(wt[l].tolist()))))
            for k, pr in P.items():
                if len(pr) == 0:
                    row[k] = None
                    continue
                i_, j_ = pr[:, 0], pr[:, 1]
                row[k] = dict(n=int(len(pr)), corr=float(np.mean(corr[i_, j_])),
                              wdown=float(np.mean((Wd[:, i_] * Wd[:, j_]).sum(0))),
                              wup=float(np.mean((Wu[i_] * Wu[j_]).sum(1))))
            nbr_set = [set(nbr[t].tolist()) | {int(t)} for t in range(len(nbr))]
            for o in (1, 4, 16, H[l] // 2):
                i_ = np.arange(H[l] - o)
                j_ = i_ + o
                row["sira_%s" % ("uzak" if o == H[l] // 2 else o)] = dict(
                    n=int(len(i_)), corr=float(np.mean(corr[i_, j_])), wdown=float(np.mean((Wd[:, i_] * Wd[:, j_]).sum(0))),
                    wup=float(np.mean((Wu[i_] * Wu[j_]).sum(1))),
                    konu=float(np.mean([int(wt[l][b]) in nbr_set[int(wt[l][a_])] for a_, b in zip(i_, j_)])))
            if a.map and model.is_glob[l]:
                n_ = min(a.map, H[l])
                maps["%s_L%d_corr" % (name, l)] = corr[:n_, :n_].astype(np.float16)
                maps["%s_L%d_wdown" % (name, l)] = (Wd[:, :n_].T @ Wd[:, :n_]).astype(np.float16)
                maps["%s_L%d_tok" % (name, l)] = wt[l][:n_].astype(np.int32)
            out.append(row)
            log("%s katman %d %s: %s" % (name, l, row["kind"], {k: (v["n"], round(v["corr"], 4), round(v["wdown"], 4),
                                                                   round(v["wup"], 4)) if v else None
                                                               for k, v in row.items() if isinstance(v, dict) or v is None}))
        res[name] = dict(run=run, positions=N, layers=out)
        if maps:
            np.savez_compressed(a.out.replace(".json", "") + "_%s_map.npz" % name, **maps)
        del model, S1, S2
    json.dump(res, open(a.out, "w"), indent=1)
    log("bitti")


if __name__ == "__main__":
    main()
