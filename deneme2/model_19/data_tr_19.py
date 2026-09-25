# -*- coding: utf-8 -*-
"""data_tr_19 -- Turkce iliski verisi (model_16'nin veri_16'si), BPE token'lariyla.

Kullanici, 25 Eylul: "Veri 16 en düzgün hali olabilir onu bpe tokenizer kullanabiliriz"; secenekten "Doygun BPE, 1.483".

Hat (tasinan dosyalar model_16'dan, icerik birebir; izler REF_* ile sinanir):
    tr_graph_19   graf: 1608 varlik, 24 iliski             (veri_16)
    tr_text_19    graf -> duz Turkce cumle                    (metin_16)
    tr_corpus_19  cumle -> belge, sizinti kapisi              (korpus_16)
    tr_splits_19  sinav bolmeleri ezber_* / cikarim_*         (taban_16)
    tr_chars_19   karakter sabitleri (korpus kullaniyor)       (jeton_16)
    data_tr_19    belge -> PARCA -> BPE -> pencere; sinav      <-- BU DOSYA

Parca = iki nokta arasi (model_16'nin son egitim bicimi): bir bildirim ya da SORU + CEVABI.  Pencere <eos> parca
<eos>, dolgulu; kayip butun token'larda.  Sinav: soru verilir, model <eos>'a kadar yazar; noktadan onceki metin cevabin
KENDISI ya da cevap + "'" ise dogru (olcme_16'nin kurali: "Elif Aydin'dir" dogru).

BPE doygun: korpusta 1.269 farkli kelime; 1.483 token'da her kelime tek token, kesme isaretli ek ayri ("▁Yilmaz ' in").
Olculdu (25 Eylul): 1.000'de seyrek adlar dograniyor (Bahcelievler 6 parca), ekler hicbir boyda paylasilmiyor.

Kural 9: korpus bir kez YERELDE uretilir (`python data_tr_19.py --yaz <klasor>`), Colab `load` ile okur ve izleri
yeniden hesaplayip karsilastirir.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_stories_19 as DS  # noqa: E402
import tr_corpus_19 as KP  # noqa: E402
import tr_graph_19 as V  # noqa: E402
import tr_splits_19 as TS  # noqa: E402
import tr_text_19 as TT  # noqa: E402

# ayar_16.AYAR ile birebir (yalniz ad ve veri_ad); kayarsa REF_PARTS / REF_EXAM kapisi durdurur
SETTINGS = TS.Ayar(ad="model_19", veri_ad="tr_graph_19", veri_tohum=0, tohum=0, jeton_ad="tam",
                   ek_kip="ezber_zincir", t_len=512, kopya=40, tetik=16, zincir_pay=0.20, n3=10000, ret_pay=0.05,
                   ret_tut=0.20, tam_kayip=True, yabanci_pay=0.20, cikarim_pay=0.10, arama_pay=0.25,
                   ayrik_ood_pay=0.05, kati_pay=0.0, belge_pay=0.0, n_olcum_max=3000)
DOC_WORDS = 62          # birim_16.BELGE_KELIME: tetiklenmis belge en cok kac kelime
EXAMS = ("ezber_olgu", "ezber_zincir", "cikarim_gorulmemis", "cikarim_yabanci",   # cevap korpusta yazar mi
         "kapi_kisayolsuz", "ayrik_arama", "ayrik_ood", "ayrik_kati")             # kapi ve hukum vermeyenler
THREE = ("ezber_3r", "cikarim_3r")      # uc adimli sinav, metne karsi kurulur (three_hop_splits)
MAIN = EXAMS[:4] + THREE
REF_GRAPH, REF_PARTS, REF_EXAM = "3cd9a2575e47", "abc21db46a1ae416", "2f5706cfb55e7415"   # model_16'nin kendi kodundan
BPE_TARGET = 4000       # ulasilmaz: 1.483'te doyar (her kelime tek token)
PAD, EOS = DS.PAD_TOKEN, DS.EOS_TOKEN                    # kimlik 0 ve 1, hikaye verisiyle ayni
MAX_ANSWER = 12         # sinavda en cok kac token uretilir (en uzun cevap 3 ad token'i + ek + nokta)


# --- metin: belge -> parca (birim_16.belgeler ve sirala'nin aynisi) ---
def documents(v, log=print):
    """KORPUS -> (Belge listesi, cakisan varlik ciftleri)."""
    G = V.kur(SETTINGS.veri_tohum)
    bb, bs = KP.sayfalar(v, G, SETTINGS.kopya, SETTINGS.tohum, SETTINGS.tetik, DOC_WORDS, SETTINGS.zincir_pay,
                         SETTINGS.n3, log, olc=lambda c: len(c.split()))
    kim = KP.kimlik_belgeleri(v, G, tohum=SETTINGS.tohum, yaz=log)
    ns = sum(len(b.cumle) for b in bs)
    red = KP.reddetme_belgeleri(v, G, int(ns * SETTINGS.ret_pay), SETTINGS.tohum,
                                bolme=KP.reddetme_bolme(G, SETTINGS.ret_tut, SETTINGS.tohum), yaz=log)
    return bb + bs + kim + red, KP.cakisan_ciftler(v)


def order(docs, clash, seed=0):
    """Belgeleri karistir; cakisan iki varligi 4 belge menzilinde yan yana koyma."""
    rs = np.random.default_rng(9000 + seed)
    rest = [int(i) for i in rs.permutation(len(docs))]
    if not clash:
        return rest
    back, ahead = 4, 64
    last, out = [], []
    while rest:
        pick = 0
        for j in range(min(ahead, len(rest))):
            e = docs[rest[j]].e
            if e < 0 or not any((e, x) in clash for x in last):
                pick = j
                break
        i = rest[pick]
        rest = rest[:pick] + rest[pick + 1:] if pick else rest[1:]
        out.append(i)
        last = (last + [docs[i].e])[-back:]
    return out


def raw_items(v, G, name):
    """Bolme -> [(bas, iliskiler, kopruler, cevap)] graf adlariyla; 1 adimda kopru yok, 2 adimda tek kopru."""
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    rel = list(V.ILISKI)
    out = []
    for z in getattr(v, name):
        z = [int(x) for x in z]
        if len(z) == 3:
            out.append((E[z[0]], (rel[z[1]],), (), E[z[2]]))
        else:
            out.append((E[z[0]], (rel[z[1]], rel[z[2]]), (E[z[3]],), E[z[4]]))
    return out


def surfaces(v, G, name):
    """Bolmeden (soru, cevap) ciftleri; yuzey tr_text_19.sinav_yuzeyi'nden -- korpusunkiyle yapi geregi ayni."""
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    tip = {a: v.tip_ad[int(v.tip[i])] for i, a in enumerate(E)}
    return [TT.sinav_yuzeyi(x, list(rr), a, tip[a])[:2] for x, rr, _, a in raw_items(v, G, name)]


def audit(parts, v, G, log=print):
    """Her bolmenin TANIMI egitim metnine karsi: ezber -> zincir ifadesi cevabiyla yazili; cikarim -> zincir ifadesi
    X icin HIC yazilmamis ve tek adimli olgular (kopru dahil) cevaplariyla yazili.  -> {bolme: {olcu: oran}}."""
    idx = possessor_index(parts)
    rep = {}
    for name in EXAMS:
        items = raw_items(v, G, name)
        if not items:
            continue
        with_ans = np.mean([written(idx, parts, x, rr, a) for x, rr, _, a in items])
        phrase_any = np.mean([written(idx, parts, x, rr) for x, rr, _, a in items])
        steps = []
        for x, rr, br, a in items:
            chain = (x,) + tuple(br) + (a,)
            steps.append(all(written(idx, parts, chain[k], rr[k:k + 1], chain[k + 1]) for k in range(len(rr))))
        rep[name] = {"n": len(items), "zincir+cevap yazili": float(with_ans), "zincir ifadesi yazili": float(phrase_any),
                     "tek adimlar yazili": float(np.mean(steps))}
        log("  denetim %-20s n %6d   zincir+cevap %5.3f   zincir ifadesi %5.3f   tek adimlar %5.3f" % (
            name, len(items), with_ans, phrase_any, np.mean(steps)))
    return rep


def fingerprint(obj):
    """bytes, metin ya da JSON'a donusen nesne -> 16 haneli iz."""
    if isinstance(obj, str):
        obj = obj.encode()
    elif not isinstance(obj, bytes):
        obj = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(obj).hexdigest()[:16]


def build_text(log=print, with_audit=True):
    """-> {parts, exam, graph, parts_fp, exam_fp[, audit, exam3_fp]}; parca ve sinav izleri model_16'nin kendi
    kodununkiyle ayni olmali.  with_audit: bolme tanimlarinin metne karsi denetimi + 3R bolmeleri (metni DEGISTIRMEZ)."""
    t0 = time.time()
    v = TS.veri_kur(SETTINGS, yaz=lambda *a: None)
    docs, clash = documents(v, log)
    texts = [docs[i].metin for i in order(docs, clash, SETTINGS.tohum)]
    parts = [p for d in texts for p in re.split(r"(?<=\.)\s+", d.strip()) if p]
    G = V.kur(SETTINGS.veri_tohum)
    exam = {k: surfaces(v, G, k) for k in EXAMS}
    out = dict(parts=parts, exam=exam, graph=V.IZ, parts_fp=fingerprint("\n".join(parts)),
               exam_fp=fingerprint({k: [list(x) for x in exam[k]] for k in EXAMS}))
    log("metin: %d belge -> %d parca   sinav %s   %.0f sn" % (
        len(texts), len(parts), "  ".join("%s %d" % (k, len(exam[k])) for k in EXAMS[:4]), time.time() - t0))
    if with_audit:
        out["audit"] = audit(parts, v, G, log)
        out["exam"].update(three_hop_splits(parts, G, log=log))
        out["exam3_fp"] = fingerprint({k: [list(x) for x in out["exam"][k]] for k in THREE})
    return out


# --- denetim: sinav tanimlari EGITIM METNINE karsi ---
POSSESSOR = re.compile(r"((?:[A-ZÇĞİÖŞÜ][\wçğıöşü]*\s)*[A-ZÇĞİÖŞÜ][\wçğıöşü]*'n?[ıiuü]n)\s")


def possessor_index(parts):
    """'X'in' tamlayan -> o tamlayanin gectigi parca kimlikleri.  Zincir ifadesi hep 'X'in ...' ile baslar."""
    idx = collections.defaultdict(list)
    for i, p in enumerate(parts):
        for m in POSSESSOR.finditer(p):
            full = m.group(1)
            words = full.split()
            for k in range(len(words)):                     # "Ali Yilmaz'in" ve "Yilmaz'in" ikisi de anahtar
                idx[" ".join(words[k:])].append(i)
    return idx


def phrase(head, rels, answer=None):
    """-> (tamlayan, zincir ifadesi 'X'in r1'inin r2'si', cevap yuzeyi) -- sinav yuzeyiyle ayni kurucudan."""
    xn, il, _rd, y, _yd = TT._yol_parca(head, rels, answer if answer is not None else head)
    return xn, "%s %s" % (xn, il), y


def written(idx, parts, head, rels, answer=None):
    """Zincir ifadesi X icin egitimde yaziyor mu; cevap verilirse AYNI parcada cevapla birlikte mi."""
    xn, p, y = phrase(head, rels, answer)
    hits = [i for i in idx.get(xn, ()) if p in parts[i]]
    return bool(hits) if answer is None else any(y in parts[i] for i in hits)


def three_hop(G, n, seed=7900):
    """Uc adimli yol ornekle, KOPRULERIYLE: (x, (r1, r2, r3), (b, c), d).  tr_corpus_19.uclu_yollar'in mantigi,
    ayri tohum (egitimdeki 3 adimli cumleler 7700'den)."""
    fwd = collections.defaultdict(list)
    for (a, r), h in G["olgu"].items():
        fwd[a].append((r, h))
    rs = np.random.default_rng(seed)
    names = sorted(G["tip"])
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 80:
        tries += 1
        x = names[int(rs.integers(len(names)))]
        if not fwd[x]:
            continue
        r1, b = fwd[x][int(rs.integers(len(fwd[x])))]
        if not fwd[b] or b == x:
            continue
        r2, c = fwd[b][int(rs.integers(len(fwd[b])))]
        if not fwd[c] or c in (x, b):
            continue
        r3, d = fwd[c][int(rs.integers(len(fwd[c])))]
        if d in (x, b, c) or (x, r1, r2, r3) in seen:
            continue
        seen.add((x, r1, r2, r3))
        out.append((x, (r1, r2, r3), (b, c), d))
    return out


def three_hop_splits(parts, G, n=3000, log=print):
    """3R sinavi, metne karsi kurulur:
       ezber_3r    uc adimli zincir X icin egitimde CEVABIYLA yazili
       cikarim_3r  uc adimli zincir de iki adimli alt zincirler (X r1 r2, b r2 r3) de YAZILMAMIS; uc tek adimli olgu
                   (X r1 b, b r2 c, c r3 d) CEVABIYLA yazili -- cevap ancak uc adim birlestirilerek bulunur."""
    idx = possessor_index(parts)
    _, tip = KP._adlar(G)
    memo, infer = [], []
    for x, rr, (b, c), d in three_hop(G, n * 40):
        q, a, _ = TT.sinav_yuzeyi(x, rr, d, tip[d])
        if written(idx, parts, x, rr, d):
            if len(memo) < n:
                memo.append((q, a))
        elif (len(infer) < n and not written(idx, parts, x, rr) and not written(idx, parts, x, rr[:2])
              and not written(idx, parts, b, rr[1:])
              and written(idx, parts, x, rr[:1], b) and written(idx, parts, b, rr[1:2], c)
              and written(idx, parts, c, rr[2:], d)):
            infer.append((q, a))
        if len(memo) >= n and len(infer) >= n:
            break
    log("3R: ezber_3r %d   cikarim_3r %d" % (len(memo), len(infer)))
    return {"ezber_3r": memo, "cikarim_3r": infer}


# --- BPE ---
def train_bpe(parts, target=BPE_TARGET):
    """Doygun BPE yalniz EGITIM parcalarinda: bosluk ▁ ile saklanir (gidis-donus birebir), noktalama ayri token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"),
        pre_tokenizers.Punctuation(behavior="isolated")])
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tok.train_from_iterator(parts, trainer=trainers.BpeTrainer(vocab_size=target, special_tokens=[PAD, EOS],
                                                               show_progress=False))
    return tok


def tokenizer_from(js):
    from tokenizers import Tokenizer
    return Tokenizer.from_str(js)


def vocab_of(tok):
    """Kimlik sirasinda token listesi (model ve decompose bunu kullanir)."""
    v = tok.get_vocab()
    out = [None] * len(v)
    for t, i in v.items():
        out[i] = t
    return out


def encode(tok, texts):
    return [e.ids for e in tok.encode_batch(list(texts))]


def detok(tok, ids):
    return tok.decode([int(i) for i in ids], skip_special_tokens=True).strip()


# --- dosya (kural 9) ---
def write(folder, log=print):
    """Metin + BPE + token'lar -> <folder>/tr_bpe.json, tr_data.pt.  Izler REF_*'la tutmazsa YAZMAZ."""
    t = build_text(log)
    for got, want, what in ((t["graph"], REF_GRAPH, "graf"), (t["parts_fp"], REF_PARTS, "parca"),
                            (t["exam_fp"], REF_EXAM, "sinav")):
        assert got == want, "%s izi %s, model_16'ninki %s -- tasinan kod kaymis" % (what, got, want)
    tok = train_bpe(t["parts"])
    ids = encode(tok, t["parts"])
    back = sum(tok.decode(x) == p for x, p in zip(ids, t["parts"]))
    assert back == len(ids), "gidis-donus %d / %d" % (back, len(ids))
    flat = np.fromiter((i for x in ids for i in x), np.int16, sum(len(x) for x in ids))
    offsets = np.cumsum([0] + [len(x) for x in ids]).astype(np.int64)
    js = tok.to_str()
    d = dict(ids=torch.from_numpy(flat), offsets=torch.from_numpy(offsets), exam=t["exam"], tokenizer=js,
             vocab=vocab_of(tok), graph=t["graph"], parts_fp=t["parts_fp"], exam_fp=t["exam_fp"],
             ids_fp=fingerprint(flat.tobytes()), tokenizer_fp=fingerprint(js))
    os.makedirs(folder, exist_ok=True)
    torch.save(d, os.path.join(folder, "tr_data.pt"))
    with open(os.path.join(folder, "tr_bpe.json"), "w", encoding="utf-8") as f:
        f.write(js)
    log("yazildi %s   sozluk %d   parca %d   token %d   en uzun parca %d   izler: token %s  bpe %s" % (
        folder, len(d["vocab"]), len(ids), len(flat), int(np.diff(offsets).max()), d["ids_fp"], d["tokenizer_fp"]))
    return d


def load(folder):
    """tr_data.pt'yi okur; token ve tokenizer izini YENIDEN hesaplayip karsilastirir."""
    d = torch.load(os.path.join(folder, "tr_data.pt"), weights_only=False)
    assert fingerprint(d["ids"].numpy().tobytes()) == d["ids_fp"], "token izi tutmuyor"
    assert fingerprint(d["tokenizer"]) == d["tokenizer_fp"], "tokenizer izi tutmuyor"
    assert (d["graph"], d["parts_fp"], d["exam_fp"]) == (REF_GRAPH, REF_PARTS, REF_EXAM), "metin izleri tutmuyor"
    return d


def make_windows(d, T=None):
    """-> (W (N,T) int32, M (N,T) bool): <eos> parca <eos>, dolgu 0 (maske disi)."""
    ids, off = d["ids"].numpy().astype(np.int32), d["offsets"].numpy()
    lens = np.diff(off)
    T = T or int(lens.max()) + 2
    eos = d["vocab"].index(EOS)
    W = np.zeros((len(lens), T), np.int32)
    M = np.zeros((len(lens), T), bool)
    W[:, 0] = eos
    col = np.arange(T)[None, :]
    inside = (col >= 1) & (col <= lens[:, None])
    W[inside] = ids
    W[np.arange(len(lens)), lens + 1] = eos
    M[col <= lens[:, None] + 1] = True
    return torch.from_numpy(W), torch.from_numpy(M)


# --- sinav ---
def correct(said, answer):
    """olcme_16'nin kurali: noktadan onceki metin cevabin kendisi ya da cevap + "'" (kuyruktaki ek tolere edilir)."""
    s = said.split(".")[0].strip()
    return s == answer or s.startswith(answer + "'")


@torch.no_grad()
def ask(m, tok, questions, device="cuda", chunk=512, steps=MAX_ANSWER):
    """Soru -> modelin yazdigi metin (acgozlu, <eos>'a kadar).  Ayni boydaki sorular birlikte: dolgu yok (zincir
    konuma bagli)."""
    vocab = tok.get_vocab()
    eos, pad = vocab[EOS], vocab[PAD]
    enc = [[eos] + x for x in encode(tok, questions)]
    out = [""] * len(questions)
    groups = collections.defaultdict(list)
    for i, e in enumerate(enc):
        groups[len(e)].append(i)
    for L, idx in groups.items():
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            w = torch.tensor([enc[i] for i in part], device=device)
            done = torch.zeros(len(part), dtype=torch.bool, device=device)
            for _ in range(min(steps, m.t_max - L)):
                lp = m.scoreboard(w)[:, -1].clone()
                lp[:, pad] = -float("inf")
                c = lp.argmax(-1)
                w = torch.cat([w, torch.where(done, torch.full_like(c, eos), c)[:, None]], 1)
                done |= c == eos
                if bool(done.all()):
                    break
            for i, row in zip(part, w[:, L:].tolist()):
                row = row[:row.index(eos)] if eos in row else row
                out[i] = detok(tok, row)
    return out


@torch.no_grad()
def answer_ce(m, tok, pairs, device="cuda", chunk=512):
    """Ogretmen zorlamasiyla cevabin (ciplak ad) token basina ortalama NLL'i."""
    vocab = tok.get_vocab()
    eos = vocab[EOS]
    rows = [([eos] + q, a) for q, a in zip(encode(tok, [p[0] for p in pairs]), encode(tok, [p[1] for p in pairs]))]
    groups = collections.defaultdict(list)
    for q, a in rows:
        groups[(len(q), len(a))].append(q + a)
    nll = n = 0.0
    for (lq, la), seqs in groups.items():
        for s in range(0, len(seqs), chunk):
            w = torch.tensor(seqs[s:s + chunk], device=device)
            lp = m.scoreboard(w)[:, lq - 1:lq - 1 + la]
            nll -= float(lp.gather(-1, w[:, lq:lq + la, None]).sum())
            n += w.shape[0] * la
    return nll / max(n, 1)


def exam_accuracy(m, tok, d, name, limit=None, device="cuda"):
    pairs = d["exam"][name][:limit] if limit else d["exam"][name]
    if not pairs:
        return float("nan"), 0
    said = ask(m, tok, [q for q, _ in pairs], device)
    return sum(correct(s, a) for s, (_, a) in zip(said, pairs)) / len(pairs), len(pairs)


def make_metric(d, train_windows, device="cuda", limit=2000, exam_limit=500, probes=4):
    """train_19'un bekledigi metric(m, "train"|"heldout", full, save, health) -> {accuracy, ce, diag}.
    train: ilk `limit` pencerede token accuracy ve ce.  heldout: SINAV -- ana dort bolmenin (ezber_olgu, ezber_zincir,
    cikarim_gorulmemis, cikarim_yabanci) birebir dogru ortalamasi; ce cevap token'larinda.  full: bolmelerin tamami."""
    tok = tokenizer_from(d["tokenizer"])
    W, M = train_windows

    def f(m, side, full=False, save=False, health=None):
        t0 = time.time()
        if side == "train":
            r = DS.evaluate(m, W[:limit], M[:limit], eos=d["vocab"].index(EOS), device=device)
            r["diag"] = {}
        else:
            n = None if full else exam_limit
            acc = {k: exam_accuracy(m, tok, d, k, n, device)[0] for k in MAIN}
            pairs = [p for k in MAIN for p in d["exam"][k][:n or None]]
            r = {"accuracy": float(np.mean(list(acc.values()))), "ce": answer_ce(m, tok, pairs, device),
                 "diag": dict(acc, kapi_kisayolsuz=exam_accuracy(m, tok, d, "kapi_kisayolsuz", n, device)[0])}
        r["timing"] = {"eval": time.time() - t0}
        return r

    f.health = False
    # tam yedekte decompose icin sabit istemler: her ana bolmeden ilk `probes` soru, token'lari hazir (DC.greedy <eos> ekler)
    f.probes = [["%s %d" % (k, i + 1), encode(tok, [q])[0]] for k in MAIN for i, (q, _) in enumerate(d["exam"][k][:probes])]
    return f


if __name__ == "__main__":
    if "--yaz" in sys.argv:
        write(sys.argv[sys.argv.index("--yaz") + 1])
    else:
        t = build_text()
        print("izler  graf %s  parca %s  sinav %s   (model_16: %s %s %s)" % (
            t["graph"], t["parts_fp"], t["exam_fp"], REF_GRAPH, REF_PARTS, REF_EXAM))
