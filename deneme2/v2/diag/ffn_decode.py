"""ffn_decode -- FFN cozucu: her konumda her katmanin FFN'i (ve attention'i) dogru token'i ne kadar itti, kelime uzayinda ne
yazdi, hangi noronlar (kullanici, 12 Ekim: "Ffn decoder yazabilir misin peki ?" -> FFN'i okuyan arac).

Ayristirma (kesin, yaklasik degil): son artik x_L = x_0 + sum_l (attn_l + ffn_l) + diger (n-gram katman >= 1 eklemeleri);
cikis RMSNorm'u konum basina olcek r = rsqrt(mean(x_L^2) + eps) ve agirlik g ile dogrusal, bagli E ile logit = E (g x_L) r.
Merkezli logit (logit - sozluk ortalamasi; softmax'a gore ayni) icin bilesen katkisi
    katki(v) = (Ec[hedef] * g) . v * r,   Ec = E[:VOCAB] - ortalama satir
ve bilesenlerin toplami = merkezli hedef logiti (testte sinanir).  Noron i katkisi = aktivasyon_i * katki(W_down[:, i]);
katmanin noronlari toplami = o katmanin FFN katkisi.  Noronun "yazdigi" (statik): Ec (g * W_down[:, i]) en buyuk 5 token.
Konum basina: dogru token olasiligi, katman basina attention / FFN katkisi, FFN'in en cok ittigi 5 / bastirdigi 3 token,
hedefe en cok katki veren 3 noron (+ statik token'lari).  Ozet: katman basina ortalama katki (hepsi, hedef turu, --counts
verilirse hedef sikligi kovasi), en buyuk 15 tek FFN katkisi.

    python ffn_decode.py --run <kosu> --data <v2/fineweb_edu_s000> [--stream <kok>] [--prompts <json>] [--text "..."]
                         [--counts <npz 'counts' | npy>] [--device cuda] --out <klasor>
Cikti: <out>/ffn_decode.json, <out>/ffn_decode.md.  fp32, dense maske (token_trace gibi), G.load ile.
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402  (load, setup, KINDS)
import token_trace as TT  # noqa: E402  (text_story)

BUCKETS = ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, float("inf")))


@torch.no_grad()
def decompose(model, batch, n_up=5, n_down=3, n_neurons=3):
    """Tek hikayelik batch (B 1) -> dict (hedefli konumlar K): logp (K,), total (K,) merkezli hedef logiti, inp / other (K,),
    attn / ffn (L, K) katki, up_ids / down_ids (L, K, n) FFN'in en cok ittigi / bastirdigi token'lar, nrn_idx / nrn_val
    (L, K, n_neurons) hedefe en cok katki veren noronlar, pos (K,) konum, target (K,), target_kind (K,)."""
    assert batch.kind.shape[0] == 1, "tek hikaye"
    rec = {"attn": {}, "ffn": {}, "act": {}}
    hooks = [model.blocks[0].register_forward_pre_hook(lambda m, a: rec.__setitem__("x0", a[0][0])),
             model.norm.register_forward_pre_hook(lambda m, a: rec.__setitem__("xL", a[0][0]))]
    def down_hook(l):
        def f(m, a, o):                                                         # None doner: cikti degismez
            rec["ffn"][l], rec["act"][l] = o[0], a[0][0]
        return f
    for l, b in enumerate(model.blocks):
        hooks.append(b.proj.register_forward_hook(lambda m, a, o, l=l: rec["attn"].__setitem__(l, o[0])))
        hooks.append(b.down.register_forward_hook(down_hook(l)))
    try:
        h = model._batch_hidden(batch)[0]
    finally:
        for hk in hooks:
            hk.remove()
    keep = batch.target[0] >= 0
    tgt = batch.target[0][keep]
    lp = torch.log_softmax(model._logits(h[keep]).float(), -1)
    E = model.E.weight[:D.VOCAB].float()
    Ec = E - E.mean(0)
    xL = rec["xL"][keep].float()
    eps = model.norm.eps if model.norm.eps is not None else torch.finfo(xL.dtype).eps
    r = torch.rsqrt(xL.pow(2).mean(-1, keepdim=True) + eps)                     # (K, 1)
    gam = model.norm.weight.float()
    w = Ec[tgt] * gam * r                                                       # (K, d): katki(v) = w . v
    L = len(model.blocks)
    out = dict(logp=lp.gather(1, tgt[:, None])[:, 0], pos=torch.nonzero(keep)[:, 0], target=tgt,
               target_kind=batch.target_kind[0][keep], total=(w * xL).sum(-1), inp=(w * rec["x0"][keep].float()).sum(-1))
    attn, ffn = torch.empty(L, len(tgt)), torch.empty(L, len(tgt))
    up_ids = torch.empty(L, len(tgt), n_up, dtype=torch.long)
    down_ids = torch.empty(L, len(tgt), n_down, dtype=torch.long)
    nrn_idx = torch.empty(L, len(tgt), n_neurons, dtype=torch.long)
    nrn_val = torch.empty(L, len(tgt), n_neurons)
    rest = xL - rec["x0"][keep].float()
    for l, b in enumerate(model.blocks):
        a_, f_ = rec["attn"][l][keep].float(), rec["ffn"][l][keep].float()
        rest = rest - a_ - f_
        attn[l], ffn[l] = (w * a_).sum(-1).cpu(), (w * f_).sum(-1).cpu()
        lg = (f_ * gam * r) @ Ec.T                                              # (K, V): FFN'in merkezli logit katkisi
        up_ids[l], down_ids[l] = lg.topk(n_up, -1).indices.cpu(), (-lg).topk(n_down, -1).indices.cpu()
        per = rec["act"][l][keep].float() * (w @ b.down.weight.float())         # (K, H) noron katkisi
        v, i = per.topk(n_neurons, -1)
        nrn_idx[l], nrn_val[l] = i.cpu(), v.cpu()
    out.update(attn=attn, ffn=ffn, other=(w * rest).sum(-1), up_ids=up_ids, down_ids=down_ids, nrn_idx=nrn_idx,
               nrn_val=nrn_val)
    return {k: v.cpu() if torch.is_tensor(v) else v for k, v in out.items()}


@torch.no_grad()
def neuron_tokens(model, layer, idx, n=5):
    """Noronun statik yazdigi: Ec (g * W_down[:, idx]) en buyuk n token (merkezli, cikis normu agirligiyla)."""
    E = model.E.weight[:D.VOCAB].float()
    v = model.blocks[layer].down.weight[:, idx].float() * model.norm.weight.float()
    return ((E - E.mean(0)) @ v).topk(n).indices.tolist()


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="Model Z kosu klasoru (agent.pt)")
    ap.add_argument("--data", required=True, help="sinir dosyalari ve reading_prompts.json")
    ap.add_argument("--stream", default=None, help="gpt2/{valid.npy, tokenizer.json} koku (varsayilan --data)")
    ap.add_argument("--prompts", default=None, help="istem dosyasi (varsayilan <data>/reading_prompts.json; --text varsa "
                                                     "yalniz acikca verilirse)")
    ap.add_argument("--text", action="append", default=[], help="serbest metin (tekrarlanabilir)")
    ap.add_argument("--counts", default=None, help="token sikliklari (npz 'counts' ya da npy): ozet siklik kovasina gore")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup(args.device, log)
    stream = args.stream or args.data
    model, _, layout, idt = G.load(args.run, args.data, dev)
    if layout != "model_z":
        sys.exit("DUR: ffn_decode yalniz Model Z (%s)" % args.run)
    model = model.float().eval()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(stream, "gpt2", "tokenizer.json"))
    dec = lambda w: "<END>" if w == D.END_ID else "<EOS>" if w == D.EOS_ID else tok.decode([int(w)])  # noqa: E731
    profile = json.load(open(os.path.join(args.data, "valid_boundaries.json"), encoding="utf-8")).get("profile", "ss")
    items = []                                                                   # (etiket, cumleler)
    pfile = args.prompts or (None if args.text else os.path.join(args.data, "reading_prompts.json"))
    if pfile:
        valid = D.TokenStories(stream, args.data, "valid")
        items += [(p["label"], [s_.tolist() for s_ in valid.sentences(p["story"])])
                  for p in json.load(open(pfile, encoding="utf-8"))["prompts"]]
    for k, t in enumerate(args.text):
        items.append(("metin %d" % (k + 1), [s_.tolist() for s_ in TT.text_story(tok, [t], profile).sentences(0)]))
    counts = None
    if args.counts:
        c = np.load(args.counts)
        counts = c["counts"] if hasattr(c, "files") else c
    st = TT.text_story(tok, [it[1] for it in items], profile)
    L = len(model.blocks)
    kinds = ["G" if g else "Y" for g in model.is_glob]
    texts, allr, ncache = [], [], {}
    for j, (label, sents) in enumerate(items):
        batch = D.build_batch(st, [[j]], "model_z", dev, row_len=1 + sum(len(s_) + 1 for s_ in sents))
        r = decompose(model, batch)
        ids = batch.tokens[0].cpu()
        toks = []
        for k in range(len(r["target"])):
            p = int(r["pos"][k])
            kind = int(batch.kind[0, p])
            layers = []
            for l in range(L):
                nr = []
                for i_, v_ in zip(r["nrn_idx"][l, k].tolist(), r["nrn_val"][l, k].tolist()):
                    if (l, i_) not in ncache:
                        ncache[(l, i_)] = [dec(x) for x in neuron_tokens(model, l, i_)]
                    nr.append([i_, round(v_, 3), ncache[(l, i_)]])
                layers.append(dict(attn=round(float(r["attn"][l, k]), 3), ffn=round(float(r["ffn"][l, k]), 3),
                                   ffn_up=[dec(x) for x in r["up_ids"][l, k].tolist()],
                                   ffn_down=[dec(x) for x in r["down_ids"][l, k].tolist()], neurons=nr))
            toks.append(dict(i=p, text=dec(ids[p]) if kind == int(D.Kind.TOKEN) else TT.KIND_NAME[kind],
                             target=dec(r["target"][k]), target_kind=G.KINDS.get(int(r["target_kind"][k])),
                             p_target=round(float(np.exp(r["logp"][k])), 5), total=round(float(r["total"][k]), 3),
                             input=round(float(r["inp"][k]), 3), other=round(float(r["other"][k]), 3), layers=layers))
        texts.append(dict(label=label, tokens=toks))
        allr.append(r)
        log("%s: %d konum, FFN katkisi %.3f, attention %.3f, giris %.3f (merkezli hedef logiti ortalamasi %.3f)" % (
            label, len(toks), float(r["ffn"].sum(0).mean()), float(r["attn"].sum(0).mean()), float(r["inp"].mean()),
            float(r["total"].mean())))
    cat = lambda k: torch.cat([x[k] for x in allr], -1)  # noqa: E731
    A, F_, tk, tg = cat("attn"), cat("ffn"), cat("target_kind"), cat("target")

    def row(sel):
        return dict(n=int(sel.sum()), attn=[round(float(A[l, sel].mean()), 4) for l in range(L)],
                    ffn=[round(float(F_[l, sel].mean()), 4) for l in range(L)],
                    input=round(float(cat("inp")[sel].mean()), 4), total=round(float(cat("total")[sel].mean()), 4))
    every = torch.ones_like(tg, dtype=torch.bool)
    summary = dict(all=row(every), by_kind={G.KINDS[k]: row(tk == int(k)) for k in G.KINDS if bool((tk == int(k)).any())})
    if counts is not None:
        n_ = torch.from_numpy(np.asarray(counts, np.float64)[tg.clamp(0, len(counts) - 1).numpy()])
        summary["by_freq"] = {"%.0e-%.0e" % (lo, hi): row((n_ >= lo) & (n_ < hi)) for lo, hi in BUCKETS
                              if bool(((n_ >= lo) & (n_ < hi)).any())}
    big = sorted(((t["layers"][l]["ffn"], x["label"], t, l) for x in texts for t in x["tokens"] for l in range(L)),
                 key=lambda z: -z[0])[:15]
    res = dict(run=os.path.basename(os.path.normpath(args.run)), identity=idt, layers=kinds, texts=texts, summary=summary)
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "ffn_decode.json"), "w", encoding="utf-8"), ensure_ascii=False)
    md = ["# ffn_decode: %s" % res["run"], "",
          "Merkezli hedef logitine katki (EKSI = dogru token'i bastirdi); katman turu %s.  Toplam = giris + sum(attention + "
          "FFN) + diger." % "".join(kinds), ""]
    for name, grp in [("hepsi", {"hepsi": summary["all"]}), ("hedef turu", summary["by_kind"]),
                      ("hedef sikligi", summary.get("by_freq", {}))]:
        if not grp:
            continue
        md += ["## %s" % name, "", "| grup | n | giris | " + " | ".join("A%d %s" % (l, kinds[l]) for l in range(L)) + " | "
               + " | ".join("F%d %s" % (l, kinds[l]) for l in range(L)) + " | toplam |", "|" + "---|" * (4 + 2 * L)]
        for g_, v in grp.items():
            md.append("| %s | %d | %.3f | %s | %s | %.3f |" % (g_, v["n"], v["input"], " | ".join("%.3f" % a for a in v["attn"]),
                                                               " | ".join("%.3f" % f for f in v["ffn"]), v["total"]))
        md.append("")
    md += ["## En buyuk 15 tek FFN katkisi", ""]
    md += ["- %s | konum %d | %r -> %r (p %.3f) | F%d %s %+.2f | itti %s | noronlar %s" % (
        lab, t["i"], t["text"], t["target"], t["p_target"], l, kinds[l], v, t["layers"][l]["ffn_up"],
        ["%d (%+.2f) %s" % (n[0], n[1], n[2]) for n in t["layers"][l]["neurons"]]) for v, lab, t, l in big]
    open(os.path.join(args.out, "ffn_decode.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    log("BITTI: %s" % args.out)
    return res


if __name__ == "__main__":
    main()
