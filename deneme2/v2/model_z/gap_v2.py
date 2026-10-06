"""gap_v2 -- V2 acik analizi: Model Z - transformer, sinav hedefi basina nll ve kirilimlar (belge 28; V1 karsiligi belge 23
gap.py).  Ad onay bekliyor.

Iki model ayni sinavda (exam_pack_plan.npz, 1.000 hikaye), egitimle ayni maske yolu (common/train.py: CUDA'da block_mask +
bloklar compile, bf16 autocast; CPU'da dense).  Hedefler satir sirasiyla iki duzende ayni (belge 22 s3); hedef ozellikleri
hikayelerden bagimsiz bir donguyle kurulur ve iki modelin batch hedefleriyle birebir karsilastirilir.

    hedef ozellikleri: tur (first / mid / end / eos), icerik / islev (train sikliginda ilk 50 ya da harf-rakamsiz token
    islev), onceki cumlelerde gecti mi / yalniz bu cumlede gecti mi / yeni, parca (kelime devami; train_meaning glue),
    cumle ici konum, hikayedeki cumle sirasi k.
    acik = nll_mz - nll_tf; kirilimda pay = sum(acik) / toplam acik.

    python gap_v2.py --tf <tf agent.pt> --mz <mz agent.pt> --meaning <meaning agent.pt> --data <v2/simplestories_gpt2>
                     --stream <simplestories> --out <klasor> [--device cuda] [--stories N] [--synthetic 1]
Cikti: <out>/gap_targets.npz (hedef basina), gap.json (kirilimlar), gap.md (tablolar).
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import train as T  # noqa: E402  (_build, _attn: egitimle ayni model kurulumu ve maske yolu)
from train_meaning import glue_tables  # noqa: E402

FUNCTION_TOP = 50           # islev: train sikliginda ilk 50 token (V1 gap.py ile ayni sinir)
KINDS = {D.TargetKind.FIRST: "first", D.TargetKind.MID: "mid", D.TargetKind.END: "end", D.TargetKind.EOS: "eos"}
K_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 10 ** 6)]
POS_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 6)]
SYN = dict(d=32, layers=2, heads=2)     # --synthetic: rastgele kucuk modeller (yol sinamasi)


def target_features(stories, rows):
    """Bagimsiz donguyle hedef ozellikleri, build_batch'in hedef sirasiyla (satir -> hikaye -> BOS, cumleler, END'ler).
    -> dict of arrays: tgt, kind, k (hedefin cumlesi; EOS: n), pos (cumle ici sira; END: cumle boyu), prev (girdi token'i;
    BOS / END girdisi -1), prior (onceki cumlelerde gecti), insent (yalniz bu cumlede daha once gecti), story, n_sent."""
    out = {k: [] for k in ("tgt", "kind", "k", "pos", "prev", "prior", "insent", "story", "n_sent")}

    def add(tgt, kind, k, pos, prev, prior, insent, sid, n):
        for key, v in zip(out, (tgt, kind, k, pos, prev, prior, insent, sid, n)):
            out[key].append(v)
    for row in rows:
        for sid in row:
            S = [s.tolist() for s in stories.sentences(sid)]
            n, seen = len(S), set()
            for k, s in enumerate(S):
                add(s[0], D.TargetKind.FIRST, k, 0, -1, s[0] in seen, False, sid, n)
                here = {s[0]}
                for i in range(1, len(s)):
                    add(s[i], D.TargetKind.MID, k, i, s[i - 1], s[i] in seen, s[i] in here and s[i] not in seen, sid, n)
                    here.add(s[i])
                add(D.END_ID, D.TargetKind.END, k, len(s), s[-1], False, False, sid, n)
                seen |= here
            add(D.EOS_ID, D.TargetKind.EOS, n, 0, -1, False, False, sid, n)
    return {k: np.asarray(v) for k, v in out.items()}


def _load(kind, path, meaning, longest, dev, synthetic):
    """agent.pt (train.py biçimi: state, identity, args) -> (model, mask_fn, layout)."""
    if synthetic:
        a = SimpleNamespace(model=kind, seed=0, meaning=meaning, shared_vocab=0, **SYN)
        model, mask_fn, layout, _ = T._build(a, longest, dev)
        return model.eval(), mask_fn, layout
    pack = torch.load(path, map_location="cpu", weights_only=False)
    idt = pack["identity"]
    assert idt["model"] == kind and idt["longest"] == longest, (idt, kind, longest)
    if kind == "model_z":
        sha = hashlib.sha256(open(meaning, "rb").read()).hexdigest()
        assert sha == idt["meaning_sha256"], "meaning agent.pt modelin egitildigi dosya degil"
    a = SimpleNamespace(model=kind, seed=idt["seed"], d=idt["d"], layers=idt["layers"], heads=idt["heads"], meaning=meaning,
                        shared_vocab=idt.get("shared_vocab", 0))                  # ortak sozluk (belge 29)
    model, mask_fn, layout, _ = T._build(a, longest, dev)
    model.load_state_dict({k.replace("._orig_mod", ""): v for k, v in pack["state"].items()})
    return model.eval(), mask_fn, layout


@torch.no_grad()
def per_target(model, mask_fn, layout, stories, rows, dev, batch_rows=D.BATCH_ROWS):
    """-> (nll (K,) float64, hedef (K,), hedef turu (K,)) numpy, satir sirasiyla (exam_scores ile ayni batch'leme)."""
    cuda = dev.type == "cuda"
    nll, tgt, tk = [], [], []
    for c in range(0, len(rows), batch_rows):
        batch = D.build_batch(stories, rows[c:c + batch_rows], layout, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
            n, _, k = model.loss_per_target(batch, T._attn(batch, mask_fn, cuda))
        nll.append(n.double().cpu().numpy())
        tgt.append(batch.target[batch.target >= 0].cpu().numpy())
        tk.append(k.cpu().numpy())
    return np.concatenate(nll), np.concatenate(tgt), np.concatenate(tk)


def _bins(x, bins):
    lab = np.full(len(x), -1)
    for i, (lo, hi) in enumerate(bins):
        lab[(x >= lo) & (x <= hi)] = i
    return lab


def breakdown(nll_tf, nll_mz, f, is_function, piece):
    """Kirilimlar -> {kirilim: [{grup, n, tf, mz, fark, pay, mz_iyi}]}; fark = ort(nll_mz - nll_tf), pay = toplam acik payi,
    mz_iyi = fark < 0 olan hedef orani."""
    gap = nll_mz - nll_tf
    total = gap.sum()
    tokmid = (f["kind"] == D.TargetKind.FIRST) | (f["kind"] == D.TargetKind.MID)
    hist = np.where(f["prior"], "onceki cumlede gecti", np.where(f["insent"], "yalniz bu cumlede gecti", "yeni"))

    def rows_of(labels, order=None):
        out = []
        for g in (order if order is not None else sorted(set(labels.tolist()))):
            m = labels == g
            if not m.any():
                continue
            out.append(dict(grup=str(g), n=int(m.sum()), tf=round(float(nll_tf[m].mean()), 4),
                            mz=round(float(nll_mz[m].mean()), 4), fark=round(float(gap[m].mean()), 4),
                            pay=round(float(gap[m].sum() / total), 4), mz_iyi=round(float((gap[m] < 0).mean()), 4)))
        return out
    kind = np.array([KINDS[int(k)] for k in f["kind"]])
    func = np.where(~tokmid, "END / EOS", np.where(is_function[np.maximum(f["tgt"], 0)], "islev", "icerik"))
    pc = np.where(~tokmid, "END / EOS", np.where(piece, "parca (kelime devami)", "kelime basi / tek token"))
    hist_tok = np.where(tokmid, hist, "END / EOS")
    cross = np.where(func == "icerik", np.char.add("icerik / ", hist), np.where(func == "islev",
                                                                                 np.char.add("islev / ", hist), func))
    kb = _bins(f["k"], K_BINS)
    klab = np.array(["k %d-%s" % (lo, hi if hi < 10 ** 6 else "") for lo, hi in K_BINS])[kb]
    pb = _bins(f["pos"], POS_BINS)
    plab = np.where(f["kind"] == D.TargetKind.MID, np.array(["i %d-%s" % (lo, hi if hi < 10 ** 6 else "")
                                                              for lo, hi in POS_BINS])[pb], kind)
    return dict(
        toplam=dict(n=int(len(gap)), tf=round(float(nll_tf.mean()), 4), mz=round(float(nll_mz.mean()), 4),
                    fark=round(float(gap.mean()), 4), mz_iyi=round(float((gap < 0).mean()), 4)),
        hedef_turu=rows_of(kind, ["first", "mid", "end", "eos"]),
        icerik_islev=rows_of(func, ["icerik", "islev", "END / EOS"]),
        hikayede_gecmis=rows_of(hist_tok, ["onceki cumlede gecti", "yalniz bu cumlede gecti", "yeni", "END / EOS"]),
        icerik_x_gecmis=rows_of(cross),
        parca=rows_of(pc, ["parca (kelime devami)", "kelime basi / tek token", "END / EOS"]),
        cumle_ici_konum=rows_of(plab, ["first"] + ["i %d-%s" % (lo, hi if hi < 10 ** 6 else "") for lo, hi in POS_BINS]
                                + ["end", "eos"]),
        cumle_sirasi=rows_of(klab, ["k %d-%s" % (lo, hi if hi < 10 ** 6 else "") for lo, hi in K_BINS]))


def token_table(nll_tf, nll_mz, tgt, text, min_n=30, top=20):
    """Token basina ortalama acik (en az min_n hedef): en kotu ve en iyi (Model Z lehine) top token."""
    gap = nll_mz - nll_tf
    u, inv, cnt = np.unique(tgt, return_inverse=True, return_counts=True)
    s = np.bincount(inv, weights=gap)
    ok = cnt >= min_n
    mean = np.where(ok, s / np.maximum(cnt, 1), np.nan)
    order = np.argsort(np.where(ok, mean, np.inf))
    item = lambda i: dict(token=text[u[i]] if u[i] < len(text) else "<END>", n=int(cnt[i]),  # noqa: E731
                          fark=round(float(mean[i]), 4), pay=round(float(s[i] / gap.sum()), 4))
    best = [item(i) for i in order[:top] if ok[i]]
    worst = [item(i) for i in order[::-1] if ok[i]][:top]
    return dict(mz_iyi=best, mz_kotu=worst)


def _md(res):
    lines = ["# V2 acik (Model Z - transformer)", "", "toplam: %s" % res["toplam"], ""]
    for name, rows in res.items():
        if not isinstance(rows, list):
            continue
        lines += ["## " + name, "", "| grup | n | tf | mz | fark | pay | mz iyi |", "|---|---|---|---|---|---|---|"]
        lines += ["| %s | %d | %.4f | %.4f | %+.4f | %.3f | %.3f |" % (r["grup"], r["n"], r["tf"], r["mz"], r["fark"],
                                                                     r["pay"], r["mz_iyi"]) for r in rows]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tf", default=None)
    ap.add_argument("--mz", default=None)
    ap.add_argument("--meaning", required=True, help="Model Z'nin meaning agent.pt'si (sha identity ile denetlenir)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stories", type=int, default=None, help="yalniz ilk N sinav satirinin hikayeleri (smoke)")
    ap.add_argument("--synthetic", type=int, default=0, help="1: rastgele kucuk modeller (yol sinamasi)")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if cuda:
        assert torch.cuda.is_available(), "GPU YOK"
        for k in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, k):
                setattr(torch._dynamo.config, k, 32)
    elif os.name == "nt":
        log("guc kisitlamasi (EcoQoS) kapali: %s" % T._no_power_throttling())
        torch.set_num_threads(8)
    valid = D.TokenStories(args.stream, args.data, "valid")
    train_meta = json.load(open(os.path.join(args.data, "train_boundaries.json"), encoding="utf-8"))
    longest = train_meta.get("max_sentence_tokens_all", train_meta["max_sentence_tokens"])     # train.py ile ayni
    ep = np.load(os.path.join(args.data, "exam_pack_plan.npz"))
    ro, rs = ep["row_offsets"], ep["row_stories"]
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(len(ro) - 1)]
    if args.stories:                                                  # smoke: ilk satirlardan ~N hikaye
        keep, n = [], 0
        for r in rows:
            if n >= args.stories:
                break
            keep.append(r)
            n += len(r)
        rows = keep
    f = target_features(valid, rows)
    log("sinav: %d satir, %d hikaye, %d hedef" % (len(rows), sum(map(len, rows)), len(f["tgt"])))
    res_nll = {}
    for kind, path in (("transformer", args.tf), ("model_z", args.mz)):
        model, mask_fn, layout = _load(kind, path, args.meaning, longest, dev, args.synthetic)
        if cuda:
            for block in model.blocks:
                block.compile(dynamic=False)
        nll, tgt, tk = per_target(model, mask_fn, layout, valid, rows, dev)
        assert np.array_equal(tgt, f["tgt"]) and np.array_equal(tk, f["kind"]), \
            "%s: batch hedefleri bagimsiz donguyle ayni degil" % kind
        res_nll[kind] = nll
        log("%s: ort nll %.4f (%d hedef; hedef ve tur bagimsiz donguyle birebir)" % (kind, nll.mean(), len(nll)))
        rj = os.path.join(os.path.dirname(path), "results.json") if path else ""
        if not args.synthetic and not args.stories and os.path.exists(rj):          # egitim sinaviyla ayni sayi mi
            want = json.load(open(rj, encoding="utf-8"))["exam"]["loss"]
            log("%s: egitimin sinav kaybi %.4f, burada %.4f, fark %.1e" % (kind, want, nll.mean(), nll.mean() - want))
        del model
        if cuda:
            torch.cuda.empty_cache()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.stream, "gpt2", "tokenizer.json"))
    text = [tok.decode([i]) for i in range(tok.get_vocab_size())]
    count = torch.load(args.meaning, map_location="cpu", weights_only=False).get("count")
    count = np.bincount(np.asarray(valid.stream), minlength=len(text))[:len(text)] if count is None else count.numpy()
    is_function = np.zeros(len(text) + 1, bool)
    is_function[np.argsort(-count)[:FUNCTION_TOP]] = True
    is_function[:len(text)] |= np.array([not any(c.isalnum() for c in s) for s in text])
    start, end = (g.numpy() for g in glue_tables(text))
    piece = (f["kind"] == D.TargetKind.MID) & ((start[np.maximum(f["tgt"], 0) % len(text)]
                                                & end[np.maximum(f["prev"], 0) % len(text)]) != 0)
    res = breakdown(res_nll["transformer"], res_nll["model_z"], f, is_function, piece)
    res["tokenler"] = token_table(res_nll["transformer"], res_nll["model_z"], f["tgt"], text)
    res["girdi"] = dict(tf=args.tf, mz=args.mz, meaning=args.meaning, synthetic=args.synthetic, stories=args.stories,
                        device=args.device, longest=longest, function_top=FUNCTION_TOP)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "gap_targets.npz"), nll_tf=res_nll["transformer"],
                        nll_mz=res_nll["model_z"], piece=piece, function=is_function[np.maximum(f["tgt"], 0)], **f)
    json.dump(res, open(os.path.join(args.out, "gap.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    md = _md(res)
    open(os.path.join(args.out, "gap.md"), "w", encoding="utf-8").write(md)
    print(md, flush=True)
    log("BITTI: %s" % args.out)
    return res


if __name__ == "__main__":
    main()
