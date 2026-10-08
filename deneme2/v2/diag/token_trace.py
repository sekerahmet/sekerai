"""token_trace -- Model Z token token: her konumda tahmin, katman katman tahmin (logit lens), attention kategorileri ve
kapatma etkisi (belge 80; kullanici, 8 Ekim: "bir cümleyi alıp model adım adım ne üretiyor ona bakmadık. Yani nerede ne
üretiyor, nerede ne işe yarıyor onu da kontrol etsek güzel olur"; adlar: "İsimler ok").

Kosu generate_readings.load_run ile; tek hikayelik batch, Z'ler arada duzen (egitim summaries_last ile; agirlik ayni,
cikti konum basina esit), fp32, dense maske (modelin kendi kurali).  Konum basina:
    tahmin   ilk --top_k aday + olasilik; dogru (sonraki) token'in olasiligi ve sirasi (0: ilk aday)
    lens     her blok ciktisi -> model.norm + bagli E: dogru token'in log-olasiligi / sirasi / ilk aday
    attn     katman basina head ortalamasi (--heads: head basina da) CATEGORIES kutlesi + en cok bakilan 3 konum;
             attention_weights = softmax(q k^T / sqrt(hd)) modelin maskesiyle (SDPA tanimi; agirlik x v = blok ciktisi)
    ablation kosul basina dogru token log-olasilik farki (kosul - none); --conds:
             read_off  Z_k kendi cumlesini okumaz (z_ablate.ablated), G aynen
             z_unseen  token'lar onceki Z'leri gormez (model_z_unseen_mask; Z'ler yine yazilir), G aynen
             g_off     G bloklari atlanir (cikti = girdi)
             g_local   G bloklari yerel blok gibi (yerel maske + mantiksal konum)
             'a+b'     birlesim (ornek read_off+g_off: gecmissiz taban); G'siz modelde g_* null
Uyari (z_ablate gibi): kapatma yon icindir; model o kosulu egitimde gormedi.
Metinler: --prompts (varsayilan <data>/reading_prompts.json) hikayelerinin tamami, gercek metin; --text serbest metin
(veri klasorunun cumle profiliyle bolunur); --generate N: istem + modelin acgozlu N cumlesi (generate) izlenir.

    python token_trace.py --run <kosu> --data <v2/fineweb_edu_s000> [--stream <kok>] [--prompts <json>] [--text "..."]
                          [--generate N] [--top_k 5] [--heads] [--conds read_off,z_unseen,g_off,g_local,read_off+g_off]
                          [--device cuda] --out <klasor>
Cikti: <out>/token_trace.json (sayfa icin duz; belge 80 s3), <out>/token_trace.md (ozet).
"""
import torch  # noqa: I001  (Windows: torch once)

import argparse
import contextlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import gap_v2 as G  # noqa: E402  (load, setup, KINDS)
import z_ablate as ZA  # noqa: E402  (read_off)
import sentence as S  # noqa: E402

CATEGORIES = ("self", "own_sentence", "bos", "prev_z", "prev_tokens")
CONDS = "read_off,z_unseen,g_off,g_local,read_off+g_off"
KIND_NAME = {int(D.Kind.BOS): "BOS", int(D.Kind.TOKEN): "TOKEN", int(D.Kind.ZTOK): "Z"}


def model_z_unseen_mask(kind, doc, sent):
    """model_z_read_mask, ama TOKEN sorgusu ZTOK anahtarini (onceki Z'ler) gormez; Z_k ve BOS aynen.  Dolgu kurali icinde."""
    read = S.model_z_read_mask(kind, doc, sent)

    def mask_mod(b, h, q, kv):
        return read(b, h, q, kv) & ~((kind[b, q] == S.TOKEN) & (kind[b, kv] == S.ZTOK))
    return mask_mod


model_z_unseen_mask.includes_padding = True


def attention_weights(block, x, pos, mask):
    """Blok girdisi x (B, T, d), konum, dense maske (B, T, T) -> attention agirliklari (B, H, T, T) fp32; GQA'da k head
    boyunca tekrarlanir (SDPA enable_gqa tanimi)."""
    q, k, v = block._qkv(x, pos)
    k = k.repeat_interleave(block.heads // block.kv_heads, 1)
    s = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
    return torch.softmax(s.masked_fill(~mask[:, None], float("-inf")), -1)


def text_story(tok, texts, profile):
    """texts: metin (str; tokenizer + data._boundaries profile ile cumleler) ya da hazir cumle listesi (token listeleri) ->
    TokenStories arayuzlu nesne (stream, sent, story, continues, n, sentences(i)); hikaye i = texts[i]."""
    tables = D.stream_tables(tok, profile)
    stream, sent, story = [], [], [0]
    for t in texts:
        if isinstance(t, str):
            x = np.array(tok.encode(t).ids + [D.EOS_ID], np.int64)
            spans = D._boundaries(x, *tables, tok, profile)[0]
        else:
            x = np.array([w for s_ in t for w in s_] + [D.EOS_ID], np.int64)
            ends = np.cumsum([len(s_) for s_ in t])
            spans = np.stack([np.r_[0, ends[:-1]], ends], 1)
        base = sum(len(a) for a in stream)
        stream.append(x)
        sent += (spans + base).tolist()
        story.append(len(sent))
    st = SimpleNamespace(stream=np.concatenate(stream), sent=np.array(sent, np.int64).reshape(-1, 2),
                         story=np.array(story, np.int64), continues=None, n=len(texts))
    st.sentences = lambda i: [st.stream[a:b] for a, b in st.sent[st.story[i]:st.story[i + 1]]]
    return st


@contextlib.contextmanager
def _condition(model, cond, batch):
    """cond ('a+b') -> attn (dense) ve G hook'lari; G'siz modelde g_* icin None."""
    parts = cond.split("+")
    assert set(parts) <= {"none", "read_off", "z_unseen", "g_off", "g_local"}, cond
    G_ = model.global_layers
    if any(p.startswith("g_") for p in parts) and not G_:
        yield None
        return
    B, T = batch.kind.shape
    if "read_off" in parts:
        with ZA.ablated(model, "read_off") as fn:
            mask_fn = fn
    elif "z_unseen" in parts:
        mask_fn = (model_z_unseen_mask, model.mask_fn[1]) if G_ else model_z_unseen_mask
    else:
        mask_fn = model.mask_fn
    fns = mask_fn if isinstance(mask_fn, tuple) else (mask_fn,)
    attn = tuple(S._dense(f(batch.kind, batch.doc, batch.sent), B, T, batch.kind.device) for f in fns)
    attn = attn if G_ else attn[0]
    hooks = []
    for block in model.blocks[len(model.blocks) - G_:]:
        if "g_off" in parts:
            hooks.append(block.register_forward_hook(lambda m, args, out: args[0]))
        elif "g_local" in parts:
            hooks.append(block.register_forward_pre_hook(lambda m, args, loc=attn[0]: (args[0], batch.pos, loc)))
    try:
        yield attn
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def trace(model, batch, conds, top_k, heads):
    """Tek hikayelik batch (B 1, model_z duzeni) -> dict: input_kind, sent, target, target_kind, logp, rank, top_ids,
    top_p (konum basina); lens_logp / lens_rank / lens_top1 (L, T); attn (L, T, 5) head ortalamasi, attn_top (L, T, 3)
    konum ve agirlik, attn_heads (L, H, T, 5) (heads); ablation {kosul: (T,) fark ya da None}.  fp32, dense maske."""
    assert batch.kind.shape[0] == 1, "tek hikaye"
    kind, sent, tgt = batch.kind[0], batch.sent[0], batch.target[0]
    T = kind.shape[0]
    has = tgt >= 0
    ins, outs = [], []
    hooks = [b.register_forward_pre_hook(lambda m, args: ins.append(args)) for b in model.blocks]
    hooks += [b.register_forward_hook(lambda m, args, out: outs.append(out)) for b in model.blocks]
    try:
        h = model._batch_hidden(batch)[0]
    finally:
        for hk in hooks:
            hk.remove()
    E = model.E.weight
    t_ = tgt.clamp_min(0)

    def score(hn):
        lp = torch.log_softmax((hn @ E.T).float(), -1)
        tl = lp.gather(1, t_[:, None])[:, 0]
        return lp, tl, (lp > tl[:, None]).sum(1)
    lp, logp, rank = score(h)
    top_p, top_ids = lp.exp().topk(top_k, -1)
    L = len(model.blocks)
    lens_logp, lens_rank, lens_top1 = (torch.empty(L, T, dtype=d_) for d_ in (torch.float32, torch.long, torch.long))
    for l, x in enumerate(outs):
        lp_l, tl, rk = score(model.norm(x[0]))
        lens_logp[l], lens_rank[l], lens_top1[l] = tl.cpu(), rk.cpu(), lp_l.argmax(-1).cpu()
    q_ = torch.arange(T, device=kind.device)
    cat = torch.stack([q_[:, None] == q_[None, :],                                                   # self
                       (kind[None, :] == S.TOKEN) & (sent[None, :] == sent[:, None]) & (q_[:, None] != q_[None, :]),
                       (kind[None, :] == S.BOS) & (q_[:, None] != q_[None, :]),
                       (kind[None, :] == S.ZTOK) & (q_[:, None] != q_[None, :]),
                       (kind[None, :] == S.TOKEN) & (sent[None, :] != sent[:, None])], -1).float()    # (T, T, 5)
    attn = torch.empty(L, T, len(CATEGORIES))
    attn_top_i, attn_top_w = torch.empty(L, T, 3, dtype=torch.long), torch.empty(L, T, 3)
    attn_heads = torch.empty(L, model.blocks[0].heads, T, len(CATEGORIES)) if heads else None
    for l, (block, (x, pos, mask)) in enumerate(zip(model.blocks, ins)):
        w = attention_weights(block, x, pos, mask)[0]                                                # (H, T, T)
        per = torch.einsum("hqk,qkc->hqc", w, cat)
        attn[l] = per.mean(0).cpu()
        if heads:
            attn_heads[l] = per.cpu()
        tw, ti = w.mean(0).topk(min(3, T), -1)
        attn_top_w[l, :, :tw.shape[1]], attn_top_i[l, :, :ti.shape[1]] = tw.cpu(), ti.cpu()
    ablation = {}
    for cond in conds:
        with _condition(model, cond, batch) as a:
            ablation[cond] = None if a is None else (score(model._batch_hidden(batch, a)[0])[1] - logp).cpu()
    keep = has.cpu()
    return dict(input_kind=kind.cpu(), sent=sent.cpu(), target=tgt.cpu(), target_kind=batch.target_kind[0].cpu(),
                has_target=keep, logp=logp.cpu(), rank=rank.cpu(), top_ids=top_ids.cpu(), top_p=top_p.cpu(),
                lens_logp=lens_logp, lens_rank=lens_rank, lens_top1=lens_top1, attn=attn, attn_top=(attn_top_i, attn_top_w),
                attn_heads=attn_heads, ablation=ablation)


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="Model Z kosu klasoru (agent.pt)")
    ap.add_argument("--data", required=True, help="sinir dosyalari ve reading_prompts.json")
    ap.add_argument("--stream", default=None, help="gpt2/{valid.npy, tokenizer.json} koku (varsayilan --data)")
    ap.add_argument("--prompts", default=None, help="istem dosyasi (varsayilan <data>/reading_prompts.json); --text varsa "
                                                     "yalniz acikca verilirse")
    ap.add_argument("--text", action="append", default=[], help="serbest metin (tekrarlanabilir)")
    ap.add_argument("--generate", type=int, default=0, help="istemden sonra N cumle acgozlu uretip izle (0: gercek metin)")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--heads", action="store_true", help="head basina attention kategorileri de")
    ap.add_argument("--conds", default=CONDS, help="virgulle kapatma kosullari ('a+b' birlesim; bos: yok)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def _entries(tok, r, n_prompt, generated, top_k, heads):
    """trace sonucu -> JSON token listesi (hedefli konumlar)."""
    dec = lambda w: "<END>" if w == D.END_ID else "<EOS>" if w == D.EOS_ID else tok.decode([int(w)])  # noqa: E731
    out, pos_in, last_sent = [], 0, -1
    ids = r["ids"]
    for i in range(len(r["input_kind"])):
        k, s_ = int(r["input_kind"][i]), int(r["sent"][i])
        pos_in = 0 if s_ != last_sent else pos_in + 1
        last_sent = s_ if k == int(D.Kind.TOKEN) else last_sent
        if not bool(r["has_target"][i]):
            continue
        L = r["lens_logp"].shape[0]
        e = dict(i=i, text="<BOS>" if k == int(D.Kind.BOS) else "<Z_%d>" % (s_ + 1) if k == int(D.Kind.ZTOK) else dec(ids[i]),
                 input_kind=KIND_NAME[k], sent=s_, pos_in_sent=pos_in if k == int(D.Kind.TOKEN) else None,
                 generated=bool(generated and s_ >= n_prompt and k != int(D.Kind.BOS)),
                 target=dec(r["target"][i]), target_kind=G.KINDS.get(int(r["target_kind"][i])),
                 p_target=round(float(np.exp(r["logp"][i])), 5), rank=int(r["rank"][i]),
                 top=[[dec(w), round(float(p), 5)] for w, p in zip(r["top_ids"][i][:top_k].tolist(), r["top_p"][i].tolist())],
                 lens=[dict(layer=l, p_target=round(float(np.exp(r["lens_logp"][l, i])), 5), rank=int(r["lens_rank"][l, i]),
                            top1=dec(r["lens_top1"][l, i])) for l in range(L)],
                 attn=[dict(layer=l, **{c: round(float(r["attn"][l, i, j]), 4) for j, c in enumerate(CATEGORIES)},
                            top=[[int(a), round(float(b), 4)] for a, b in zip(r["attn_top"][0][l, i].tolist(),
                                                                              r["attn_top"][1][l, i].tolist())])
                       for l in range(L)],
                 ablation={c: None if v is None else round(float(v[i]), 4) for c, v in r["ablation"].items()})
        if heads:
            e["attn_heads"] = np.round(r["attn_heads"][:, :, i], 4).tolist()
        out.append(e)
    return out


def _summary(tokens, n_layers, n_glob):
    """Konum listesi -> ozet: none log-olasilik ortalamasi, kosul farklari (hepsi / hedef turu / girdi turu), olusma katmani
    (dogru token'in ilk kez ilk aday oldugu katman; son tahmini dogru olanlarda), katman basina attention kategorileri."""
    out = dict(positions=len(tokens), logp=round(float(np.mean([np.log(max(t["p_target"], 1e-12)) for t in tokens])), 4),
               top1=round(float(np.mean([t["rank"] == 0 for t in tokens])), 4))
    conds = list(tokens[0]["ablation"]) if tokens else []
    by = lambda key: sorted({t[key] for t in tokens if t[key] is not None})  # noqa: E731
    out["ablation"] = {c: None if tokens[0]["ablation"][c] is None else dict(
        all=round(float(np.mean([t["ablation"][c] for t in tokens])), 4),
        **{"target_" + k: round(float(np.mean([t["ablation"][c] for t in tokens if t["target_kind"] == k])), 4)
           for k in by("target_kind")},
        **{"input_" + k: round(float(np.mean([t["ablation"][c] for t in tokens if t["input_kind"] == k])), 4)
           for k in by("input_kind")}) for c in conds}
    formed = [next(l for l in range(n_layers) if all(t["lens"][m]["rank"] == 0 for m in range(l, n_layers)))
              for t in tokens if t["rank"] == 0]
    out["formed_layer"] = dict(counts=np.bincount(formed, minlength=n_layers).tolist() if formed else [0] * n_layers,
                               in_global=round(float(np.mean([f >= n_layers - n_glob for f in formed])), 4) if formed else None)
    out["attn_by_layer"] = [{c: round(float(np.mean([t["attn"][l][c] for t in tokens])), 4) for c in CATEGORIES}
                            for l in range(n_layers)]
    return out


def main(argv=None):
    args = _args(argv)
    t0 = time.time()
    log = lambda m: print("[%6.1f sn] %s" % (time.time() - t0, m), flush=True)  # noqa: E731
    dev = G.setup(args.device, log)
    stream = args.stream or args.data
    model, _, layout, idt = G.load(args.run, args.data, dev)
    if layout != "model_z":
        sys.exit("DUR: token_trace yalniz Model Z (%s)" % args.run)
    model = model.float()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(stream, "gpt2", "tokenizer.json"))
    profile = json.load(open(os.path.join(args.data, "valid_boundaries.json"), encoding="utf-8")).get("profile", "ss")
    conds = [c for c in args.conds.split(",") if c]
    items = []                                                           # (etiket, kaynak, hikaye, cumleler, istem cumle)
    pfile = args.prompts or (None if args.text else os.path.join(args.data, "reading_prompts.json"))
    if pfile:
        valid = D.TokenStories(stream, args.data, "valid")
        for p in json.load(open(pfile, encoding="utf-8"))["prompts"]:
            sents = [s_.tolist() for s_ in valid.sentences(p["story"])]
            if args.generate:
                pr = sents[:p["sentences"]]
                gen = model.generate([pr], max_sentences=args.generate, max_tokens=128)[0][0]
                items.append((p["label"], "generate", p["story"], pr + gen, len(pr)))
            else:
                items.append((p["label"], "prompt", p["story"], sents, p["sentences"]))
    for k, t in enumerate(args.text):
        st = text_story(tok, [t], profile)
        items.append(("metin %d" % (k + 1), "text", None, [s_.tolist() for s_ in st.sentences(0)], 0))
    st = text_story(tok, [it[3] for it in items], profile)
    texts, n_layers = [], len(model.blocks)
    for j, (label, source, story, sents, n_prompt) in enumerate(items):
        n_tok = 1 + sum(len(s_) + 1 for s_ in sents)
        batch = D.build_batch(st, [[j]], "model_z", dev, row_len=n_tok)
        r = trace(model, batch, conds, args.top_k, args.heads)
        r = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in r.items()}
        r["ids"] = batch.tokens[0].cpu().numpy()
        tokens = _entries(tok, r, n_prompt, source == "generate", args.top_k, args.heads)
        texts.append(dict(label=label, source=source, story=story, prompt_sentences=n_prompt, length=n_tok,
                          tokens=tokens, summary=_summary(tokens, n_layers, model.global_layers)))
        log("%s (%s): %d konum, logp %.3f, ilk aday %.3f | %s" % (label, source, len(tokens), texts[-1]["summary"]["logp"],
                                                               texts[-1]["summary"]["top1"],
                                                               {c: (v or {}).get("all") for c, v in
                                                                texts[-1]["summary"]["ablation"].items()}))
    every = [t for x in texts for t in x["tokens"]]
    res = dict(run=os.path.basename(os.path.normpath(args.run)), identity=idt, top_k=args.top_k, conditions=conds,
               categories=list(CATEGORIES), heads=model.blocks[0].heads,
               layers=[dict(index=l, kind="glob" if l >= n_layers - model.global_layers else "loc") for l in range(n_layers)],
               texts=texts, summary=_summary(every, n_layers, model.global_layers))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "token_trace.json"), "w", encoding="utf-8"), ensure_ascii=False)
    s_ = res["summary"]
    md = ["# token_trace: %s" % res["run"], "",
          "%d metin, %d konum; none log-olasilik %.4f, ilk aday %.4f.  Kapatma: kosul - none (dogru token log-olasiligi; "
          "yon icindir)." % (len(texts), s_["positions"], s_["logp"], s_["top1"]), ""]
    keys = sorted({k for v in s_["ablation"].values() if v for k in v}, key=lambda k: (k != "all", k))
    md += ["| kosul | " + " | ".join(keys) + " |", "|" + "---|" * (1 + len(keys))]
    for c in conds:
        v = s_["ablation"][c] or {}
        md.append("| %s | %s |" % (c, " | ".join("%+.4f" % v[k] if k in v else "-" for k in keys)))
    md += ["", "Olusma katmani (dogru token'in ilk kez kalici ilk aday oldugu katman; son tahmini dogru konumlar): %s; G "
           "katmaninda olusan pay %s." % (s_["formed_layer"]["counts"], s_["formed_layer"]["in_global"]), "",
           "| katman | tur | " + " | ".join(CATEGORIES) + " |", "|" + "---|" * (2 + len(CATEGORIES))]
    for l, a in enumerate(s_["attn_by_layer"]):
        md.append("| %d | %s | %s |" % (l, res["layers"][l]["kind"], " | ".join("%.3f" % a[c] for c in CATEGORIES)))
    worst = sorted(((t["ablation"].get(c) or 0, c, x["label"], t["i"], t["text"], t["target"]) for x in texts
                    for t in x["tokens"] for c in conds), key=lambda z: z[0])[:10]
    md += ["", "En cok etkilenen 10 konum (kosul, metin, konum, girdi -> hedef, fark):", ""]
    md += ["- %s | %s | %d | %r -> %r | %+.3f" % (c, lab, i, a, b, d) for d, c, lab, i, a, b in worst]
    open(os.path.join(args.out, "token_trace.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    log("BITTI: %s" % args.out)
    return res


if __name__ == "__main__":
    main()
