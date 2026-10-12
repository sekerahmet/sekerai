"""make_pmi -- egitim cumlelerinden sabit PMI tablosu (deneme/pmi): cumle ici butun kelime ciftleri sayilir, PPMI =
max(0, log P(a, b) / (P(a) P_0.75(b)) - shift) (kaydirilmis, varsayilan log 5; aynı kelimenin BPE parcalari cift sayilmaz) (baglam dagilimi 0,75 yumusatmali; Levy, Goldberg & Dagan 2015), kelime basina en yakin
K komsu + agirlik (ham PPMI / kuresel olcek: satir toplami 1 degil); yaninda SVD ile 64'luk vektor.  Kullanici, 12 Ekim: "n-gram gibi ama sabit bir tablo gibi baştan okunan";
"PMI aslında bir kelimeye yakın 10 20 kelimeyi verebiliyor".  Yalniz train bolumu (sinav valid'den; kural 14).

    python make_pmi.py --data <fineweb_edu_s000> --out <pmi.npz> [--k 20 --dim 64 --vocab 0 --device cuda --max_tokens 0]

Cikti npz: nbr (VOCAB x K int32, -1 bos), nbr_w (VOCAB x K, ham PPMI / kuresel olcek -- satirlarin medyan toplami 1), nbr_pmi (ham PPMI), S, T (VOCAB x dim), counts, meta (json).
GPU bellegi: sayim matrisi V^2 int32 (V 50.258: ~10 GB) + PPMI float32 (~10 GB).  --vocab N: yalniz en sik N token (CPU sinamasi).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

LOOK = [" Paris", " France", " red", " apple", " doctor", " river", " king", " water", " she", " because", " the"]


def log(*a):
    print("[%7.1f sn]" % (time.time() - T0), *a, flush=True)


def sentence_pairs(tok, lens, dev, wid=None):
    """Cumle parcasi (token'lar ardisik, boylar lens) -> ayni cumledeki butun sirali (a, b) ciftleri, a != konum b (pencere =
    cumle).  Cift sayisi sum L (L - 1).  wid (konum basina kelime kimligi) verilirse ayni kelimenin BPE parcalari cift sayilmaz."""
    tok, lens = torch.as_tensor(tok, device=dev), torch.as_tensor(lens, device=dev)
    sid = torch.repeat_interleave(torch.arange(len(lens), device=dev), lens)
    start = torch.cumsum(lens, 0) - lens
    per = lens[sid] - 1                                                  # her token icin baska konum sayisi
    src = torch.repeat_interleave(torch.arange(len(tok), device=dev), per)
    k = torch.arange(len(src), device=dev) - torch.repeat_interleave(torch.cumsum(per, 0) - per, per)
    i = src - start[sid[src]]                                            # kaynagin cumle icindeki sirasi
    j = k + (k >= i).long()                                              # kendisi haric
    dst = start[sid[src]] + j
    if wid is not None:
        wid = torch.as_tensor(wid, device=dev)
        keep = wid[src] != wid[dst]                                      # " red" + "iscovered" = ayni kelime
        src, dst = src[keep], dst[keep]
    return tok[src], tok[dst]


def word_starts(root):
    """GPT-2 sozlugu -> token kelime basi mi (bosluk / satir / noktalama ile baslar); degilse onceki token'in devami."""
    tk = json.load(open(os.path.join(root, "gpt2", "tokenizer.json"), encoding="utf-8"))
    st = np.ones(D.VOCAB, bool)
    for k_, v in tk["model"]["vocab"].items():
        if v < D.VOCAB and k_ and k_[0].isalnum() and k_[0] not in "ĠĊĉ":   # bosluk / satir / sekme isaretleri harf sayilir
            st[v] = False
    return st


def main(argv=None):
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FineWeb klasoru (gpt2/train.npy + train_sentence_offsets.npy)")
    ap.add_argument("--stream", default=None, help="ham akis koku (varsayilan --data)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20, help="kelime basina komsu")
    ap.add_argument("--dim", type=int, default=64, help="SVD vektor boyu")
    ap.add_argument("--min_pair", type=int, default=5, help="komsu icin en az birlikte gecme")
    ap.add_argument("--min_count", type=int, default=100, help="komsu olabilmek icin en az siklik (nadir gurultu)")
    ap.add_argument("--shift", type=float, default=float(np.log(5)), help="kaydirilmis PPMI: max(0, PMI - shift) (Levy & Goldberg 2014)")
    ap.add_argument("--same_word", type=int, default=0, help="0: ayni kelimenin BPE parcalari cift sayilmaz; 1: sayilir (eski)")
    ap.add_argument("--vocab", type=int, default=0, help="0: butun sozluk; N: yalniz en sik N token (sinama)")
    ap.add_argument("--chunk", type=int, default=200_000, help="parca basina cumle")
    ap.add_argument("--max_tokens", type=int, default=0, help="0: butun train; yoksa ilk N token'lik cumleler (sinama)")
    ap.add_argument("--plan_steps", type=int, default=0,
                    help="0: butun train; N: yalniz epok 1 planinin ilk N adiminda (N x BATCH_ROWS satir) gorulen hikayeler "
                         "(kullanici, 12 Ekim: model ham metnin tamamini ancak belirli bir adimdan gorur -> adil tablo)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")   # Windows konsolu: yazilamayan token karakteri
    if os.name == "nt":
        from train import _no_power_throttling
        log("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling())
    dev = torch.device(a.device)
    root = a.stream or a.data
    stream = np.load(os.path.join(root, "gpt2", "train.npy"), mmap_mode="r")
    sent = np.load(os.path.join(a.data, "train_sentence_offsets.npy"))
    if a.max_tokens:
        sent = sent[:int(np.searchsorted(sent[:, 1], a.max_tokens)) + 1]
    seen = None
    if a.plan_steps:                                                     # yalniz kosunun gorecegi hikayeler (sirasi onemsiz)
        from train import BATCH_ROWS
        f = np.load(os.path.join(a.data, "train_pack_plan_e1.npz"))
        ro, rs = f["row_offsets"], f["row_stories"]
        seen = np.unique(rs[ro[0]:ro[min(a.plan_steps * BATCH_ROWS, len(ro) - 1)]])
        so = np.load(os.path.join(a.data, "train_story_offsets.npy"))
        spans = np.stack([sent[so[seen], 0], sent[so[seen + 1] - 1, 1]], 1)  # hikaye token araligi (sayim icin)
        sent = sent[np.concatenate([np.arange(so[i], so[i + 1]) for i in seen])]
        log("plan_steps %d: %d / %d hikaye, %d cumle" % (a.plan_steps, len(seen), len(so) - 1, len(sent)))
    sent = sent[sent[:, 1] - sent[:, 0] >= 2]                            # tek token'li cumlede cift yok
    counts = np.zeros(D.VOCAB, np.int64)
    if seen is None:
        for i in range(0, len(sent), a.chunk):
            s = sent[i:i + a.chunk]
            counts += np.bincount(np.asarray(stream[int(s[0, 0]):int(s[-1, 1])]), minlength=D.VOCAB)[:D.VOCAB]
    else:                                                                # yalniz secilen hikayeler (aradaki gorulmeyen metin yok)
        for i in range(0, len(spans), 20_000):
            s = spans[i:i + 20_000]
            n = s[:, 1] - s[:, 0]
            idx = np.repeat(s[:, 0], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
            counts += np.bincount(np.asarray(stream[idx]), minlength=D.VOCAB)[:D.VOCAB]
    order = np.argsort(-counts, kind="stable")
    V = a.vocab or int((counts > 0).sum())
    keep = order[:V]                                                     # sozluk kimligi -> sik sira
    rank = np.full(D.VOCAB, -1, np.int64)
    rank[keep] = np.arange(V)
    rank_t = torch.from_numpy(rank).to(dev)
    starts = word_starts(root)
    log("cumle %d, token %d, sozluk %d, cihaz %s, sayim matrisi %.1f GB" % (len(sent), int(counts.sum()), V, dev, V * V * 4 / 1e9))
    C = torch.zeros(V * V, dtype=torch.int32, device=dev)
    n_pairs = 0
    for ci, i in enumerate(range(0, len(sent), a.chunk)):
        s = sent[i:i + a.chunk]
        lens = (s[:, 1] - s[:, 0]).astype(np.int64)
        base = int(s[0, 0])
        pos = np.repeat(s[:, 0] - base, lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        tok = np.asarray(stream[base:int(s[-1, 1])]).astype(np.int64)[pos]
        ws = starts[tok].copy()
        ws[np.cumsum(lens) - lens] = True                               # cumle basi her zaman kelime basi
        x, y = sentence_pairs(rank_t[torch.from_numpy(tok).to(dev)], lens, dev, None if a.same_word else np.cumsum(ws))
        ok = (x >= 0) & (y >= 0)
        u, c = torch.unique(x[ok] * V + y[ok], return_counts=True)
        C[u] += c.to(torch.int32)
        n_pairs += int(ok.sum())
        if ci % 20 == 0:
            log("parca %d / %d, cift %d" % (ci + 1, -(-len(sent) // a.chunk), n_pairs))
    C = C.view(V, V)
    row = C.sum(1, dtype=torch.float64)                                  # P(a) (cift marjinali)
    col = row ** 0.75                                                    # P_0.75(b) (yumusatma)
    lrow = (-torch.log(row.clamp_min(1))).float()                        # - log c(a)
    lcol = (torch.log(col.sum()) - torch.log(col.clamp_min(1e-30))).float()   # - log P_0.75(b)
    rare = torch.from_numpy(counts[keep] < a.min_count).to(dev)
    pmi = torch.empty(V, V, dtype=torch.float32, device=dev)
    w, nbr_r = torch.zeros(V, a.k, device=dev), torch.full((V, a.k), -1, dtype=torch.long, device=dev)
    for r0 in range(0, V, 4096):                                         # satir bloklari: ara matrisler kucuk kalir
        c = C[r0:r0 + 4096].float()
        p = torch.where(c > 0, torch.log(c.clamp_min(1)) + lrow[r0:r0 + 4096, None] + lcol[None] - a.shift,
                        torch.zeros((), device=dev)).clamp_min(0)       # PPMI: log c(a,b) - log c(a) - log P_0.75(b)
        pmi[r0:r0 + 4096] = p
        sc = p.masked_fill(rare[None] | (c < a.min_pair), 0)            # komsu: sik ve en az min_pair kez birlikte
        sc[torch.arange(len(sc), device=dev), torch.arange(r0, r0 + len(sc), device=dev)] = 0   # kendisi degil
        wb, jb = torch.topk(sc, a.k, dim=1)
        nbr_r[r0:r0 + 4096] = torch.where(wb > 0, jb, torch.full_like(jb, -1))
        w[r0:r0 + 4096] = wb
    del C
    tot = w.sum(1)                                                       # satirin bag gucu (ham PPMI toplami)
    scale = float(tot[tot > 0].median()) if bool((tot > 0).any()) else 1.0
    raw, w = w, w / scale                     # kuresel olcek: guclu bagli kelime guclu, zayif bagli (the) zayif torba
    U, Sg, Vt = torch.svd_lowrank(pmi, q=a.dim, niter=4)                 # PPMI ~ U diag(Sg) V^T
    Sv, Tv = (U * Sg.sqrt()).cpu().numpy(), (Vt * Sg.sqrt()).cpu().numpy()
    nbr = np.full((D.VOCAB, a.k), -1, np.int32)
    nbr_w = np.zeros((D.VOCAB, a.k), np.float32)
    nbr_pmi = np.zeros((D.VOCAB, a.k), np.float32)
    S = np.zeros((D.VOCAB, a.dim), np.float32)
    T = np.zeros((D.VOCAB, a.dim), np.float32)
    nr = nbr_r.cpu().numpy()
    nbr[keep] = np.where(nr >= 0, keep[np.clip(nr, 0, None)], -1)
    nbr_w[keep], nbr_pmi[keep] = w.cpu().numpy(), raw.cpu().numpy()
    S[keep], T[keep] = Sv, Tv
    tk = json.load(open(os.path.join(root, "gpt2", "tokenizer.json"), encoding="utf-8"))
    voc = {v: k_.replace("Ġ", " ") for k_, v in tk["model"]["vocab"].items()}
    inv = {s_: i_ for i_, s_ in voc.items()}
    for w_ in LOOK:
        i_ = inv.get(w_)
        if i_ is not None and nbr[i_, 0] >= 0:
            log("komsu %r (torba agirligi %.2f): %s" % (w_, nbr_w[i_].sum(), ", ".join(
                "%r %.2f" % (voc.get(int(b), b), nbr_pmi[i_, n]) for n, b in enumerate(nbr[i_]) if b >= 0)))
    meta = dict(data=a.data, plan_steps=a.plan_steps, stories=None if seen is None else int(len(seen)), k=a.k, shift=a.shift, same_word=a.same_word, dim=a.dim, min_pair=a.min_pair, min_count=a.min_count, vocab=V, sentences=int(len(sent)),
                tokens=int(counts.sum()), pairs=int(n_pairs), svd_top=[float(x) for x in Sg[:5].cpu()], weight_scale=scale,
                stream_sha256=json.load(open(os.path.join(a.data, "train_boundaries.json")))["stream_sha256"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp.npz"                                                # kesilirse yarim dosya hedefte kalmaz
    np.savez(tmp, nbr=nbr, nbr_w=nbr_w, nbr_pmi=nbr_pmi, S=S, T=T, counts=counts, meta=json.dumps(meta))
    os.replace(tmp, a.out)
    log("yazildi:", a.out, meta)
    return meta


if __name__ == "__main__":
    main()
