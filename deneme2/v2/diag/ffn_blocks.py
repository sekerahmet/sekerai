"""ffn_blocks -- G MLP'lerinde bitisik noron bloklari islevsel mi? (deneme/som; kullanici 12 Ekim: "önce blok ölçümlerini yap")

Blok = sirada bitisik B noron (varsayilan 64).  Kontrol = ayni boyda, noron sirasi rastgele karistirilmis "blok" (katman basina
sabit permutasyon).  SOM'suz modelde iki okuma ayni beklenir; SOM bloklari islevselse sirali / karisik farki cikar.
  yogunlasma  token basina |aktivasyon| kutlesinin en aktif %10 blokta toplanan payi (ortalama)
  kapatma     tek blok sifirlaninca (down girdisi) hedef basina nll artisi; yerellik = toplam artisin en cok zarar goren %10
              hikayeden gelen payi ve en cok zarar goren %2 token'dan gelen payi; ortalama artis
    python ffn_blocks.py --runs ad=klasor [...] --data <fineweb_edu_s000> --out <json> [--stories 40] [--blocks 12]
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


def stories(data, n, max_tokens):
    tok, _ = G.vocab_text(data)
    profile = json.load(open(os.path.join(data, "valid_boundaries.json"), encoding="utf-8")).get("profile", "ss")
    valid = D.TokenStories(data, data, "valid")
    items = []
    for i in range(n):
        sents, k = [], 0
        for s_ in valid.sentences(i):
            if k + len(s_) + 1 > max_tokens:
                break
            sents.append(s_.tolist())
            k += len(s_) + 1
        if sents:
            items.append(sents)
    return TT.text_story(tok, items, profile), items


@torch.no_grad()
def nll_all(model, st, items, dev, hook=None):
    """-> (hedef basina nll birlesik, hikaye kimligi) ; hook: (blok, fn) down girdisine pre-hook."""
    hs = [] if hook is None else [hook[0].down.register_forward_pre_hook(hook[1])]
    out, sid = [], []
    try:
        for j, sents in enumerate(items):
            b = D.build_batch(st, [[j]], "model_z", dev, row_len=1 + sum(len(s_) + 1 for s_ in sents))
            n = model.loss_per_target(b)[0].float().cpu()
            out.append(n)
            sid.append(torch.full_like(n, j, dtype=torch.long))
    finally:
        for h in hs:
            h.remove()
    return torch.cat(out).numpy(), torch.cat(sid).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stories", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=12, help="katman basina kapatilan blok sayisi (sirali ve karisik ayri)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup(a.device, log)
    st, items = stories(a.data, a.stories, a.max_tokens)
    res = {}
    for spec in a.runs:
        name, run = spec.split("=", 1)
        model = G.load(run, a.data, dev)[0].float().eval()
        gl = [l for l, g in enumerate(model.is_glob) if g]
        rng = np.random.default_rng(0)
        H = {l: model.blocks[l].down.weight.shape[1] for l in gl}
        perm = {l: torch.from_numpy(rng.permutation(H[l])).to(dev) for l in gl}
        # yogunlasma
        acc = {l: {"sirali": [], "karisik": []} for l in gl}
        rec = {}
        hooks = [model.blocks[l].down.register_forward_pre_hook(lambda m, ar, l=l: rec.__setitem__(l, ar[0][0])) for l in gl]
        with torch.no_grad():
            for j, sents in enumerate(items):
                b = D.build_batch(st, [[j]], "model_z", dev, row_len=1 + sum(len(s_) + 1 for s_ in sents))
                model._batch_hidden(b)
                keep = b.kind[0] != int(D.Kind.PAD)
                for l in gl:
                    A = rec[l][keep].abs().float()                                  # (T, H)
                    nb = H[l] // a.block
                    for k, X in (("sirali", A), ("karisik", A[:, perm[l]])):
                        m_ = X[:, :nb * a.block].reshape(len(X), nb, a.block).sum(-1)
                        top = m_.topk(max(1, nb // 10), -1).values.sum(-1) / m_.sum(-1).clamp_min(1e-12)
                        acc[l][k].append(top.cpu())
        for h in hooks:
            h.remove()
        base, sid = nll_all(model, st, items, dev)
        log("%s: %d hedef, taban nll %.4f" % (name, len(base), base.mean()))
        layers = {}
        for l in gl:
            nb = H[l] // a.block
            pick = rng.choice(nb, a.blocks, replace=False)
            row = {k: float(torch.cat(v).mean()) for k, v in acc[l].items()}
            for kind in ("sirali", "karisik"):
                ds, st10, tk2 = [], [], []
                for bi in pick:
                    idx = torch.arange(bi * a.block, (bi + 1) * a.block, device=dev)
                    if kind == "karisik":
                        idx = perm[l][idx]
                    mask = torch.ones(H[l], device=dev)
                    mask[idx] = 0

                    def fn(m, ar, mask=mask):
                        return (ar[0] * mask,)
                    nll, _ = nll_all(model, st, items, dev, (model.blocks[l], fn))
                    d = nll - base
                    pos = np.clip(d, 0, None)
                    tot = pos.sum() + 1e-12
                    per_story = np.bincount(sid, weights=pos, minlength=len(items))
                    ds.append(d.mean())
                    st10.append(np.sort(per_story)[::-1][:max(1, len(items) // 10)].sum() / tot)
                    tk2.append(np.sort(pos)[::-1][:max(1, len(pos) // 50)].sum() / tot)
                row["kapatma_" + kind] = dict(ort_artis=float(np.mean(ds)), hikaye_ust10=float(np.mean(st10)),
                                             token_ust2=float(np.mean(tk2)))
            layers[l] = row
            log("%s katman %d: %s" % (name, l, json.dumps(row)))
        res[name] = dict(run=run, layers=layers)
        del model
    json.dump(res, open(a.out, "w"), indent=1)
    log("bitti")


if __name__ == "__main__":
    main()
