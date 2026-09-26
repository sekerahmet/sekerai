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
import random
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
SHOW = ("ezber_olgu", "ezber_zincir", "ezber_3r", "cikarim_gorulmemis", "cikarim_yabanci", "cikarim_3r")   # EZBER + CIKARIM
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
    out = dict(parts=parts, documents=texts, exam=exam, graph=V.IZ, parts_fp=fingerprint("\n".join(parts)),
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
def genitive_pattern():
    """Iliski kelimesinin tamlayan eki ayri parca: "▁kardesi|nin", "▁cocugu|nun", "▁yer|in".  Tek adimli arama iliski
    kelimesinin konumunda yapiliyor; 2R'de ilk iliski ancak boyle 1R'dekiyle AYNI token olur (OLCULENLER §1l, 26 Eylul).
    Ek, metni kuran tr_text_19._nin'den (yol ifadesiyle ayni)."""
    alts = {"(?<=▁%s)%s" % (re.escape(s.split()[-1]), re.escape(TT._nin(s, False))) for s in V.TR_ILISKI.values()}
    return "|".join(sorted(alts))


def train_bpe(parts, target=BPE_TARGET, split_genitive=False):
    """Doygun BPE yalniz EGITIM parcalarinda: bosluk ▁ ile saklanir (gidis-donus birebir), noktalama ayri token;
    split_genitive: iliski kelimelerinin tamlayan eki ayri token (genitive_pattern)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=None))
    steps = [pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"),
             pre_tokenizers.Punctuation(behavior="isolated")]
    if split_genitive:
        steps.append(pre_tokenizers.Split(Regex(genitive_pattern()), behavior="isolated"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(steps)
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tok.train_from_iterator(parts, trainer=trainers.BpeTrainer(vocab_size=target, special_tokens=[PAD, EOS],
                                                               show_progress=False))
    return tok


class MorphTokenizer:
    """Kok ayri token, ek ayri token (tr_morph_19 = model_16/ek_16).  Kullanici, 20 Eylul: "kok ayri token ek ayri
    token ... ben bunlarin ayri token olmasini net istedim".  Kok YUZEY bicimiyle ("▁çocuğ", "▁Bölüm"), kesme isareti
    ayri token: metin birebir geri cozulur, sinav kurali ve araclar degismez.  HF Tokenizer'in kullanilan arayuzu."""
    KIND = "morph_19"

    def __init__(self, vocab, ozel):
        self.vocab = list(vocab)
        self.ix = {w: i for i, w in enumerate(self.vocab)}
        self.ozel = frozenset(ozel)
        self.memo = {}

    @staticmethod
    def pieces(word, ozel):
        """Bosluksuz kelime -> ["▁kok", ("'"), ek, ..., noktalama]; birlesimi "▁" + kelime.  Ek yuzeyi kelimenin sonuyla
        ya da kesme isareti bir parca sinirinda tutmazsa kelime bolunmez."""
        import tr_morph_19 as MO
        core, tail = word, []
        while core and core[-1] in MO.NOKTALAMA:
            tail.insert(0, core[-1])
            core = core[:-1]
        if not core:
            return ["▁" + tail[0]] + tail[1:] if tail else []
        plain = core.replace("'", "")
        sufs = [u[1:] for u in MO.bol(core, korunan=ozel)[1:] if u.startswith("-")]
        n = sum(map(len, sufs))
        out = [plain[:len(plain) - n]] + sufs if sufs and plain.endswith("".join(sufs)) else [plain]
        a = core.find("'")
        if a >= 0:
            ends = list(np.cumsum([len(p) for p in out]))
            out = out[:ends.index(a) + 1] + ["'"] + out[ends.index(a) + 1:] if a in ends else [core]
        return ["▁" + out[0]] + out[1:] + tail

    def _ids(self, text):
        out = []
        for w in text.split(" "):
            if w not in self.memo:
                self.memo[w] = [self.ix[p] for p in self.pieces(w, self.ozel)]
            out += self.memo[w]
        return out

    def encode_batch(self, texts):
        from types import SimpleNamespace
        return [SimpleNamespace(ids=self._ids(t)) for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        s = "".join(self.vocab[int(i)] for i in ids
                    if not (skip_special_tokens and self.vocab[int(i)] in (PAD, EOS))).replace("▁", " ")
        return s[1:] if s.startswith(" ") else s

    def decode_batch(self, batch):
        return [self.decode(x) for x in batch]

    def get_vocab(self):
        return dict(self.ix)

    def to_str(self):
        return json.dumps({"type": self.KIND, "vocab": self.vocab, "ozel": sorted(self.ozel)}, ensure_ascii=False)

    @classmethod
    def build(cls, texts, ozel):
        """Sozluk: [PAD, EOS] + metinlerdeki butun parcalar, sirali."""
        pieces = {p for w in {w for t in texts for w in t.split(" ")} for p in cls.pieces(w, ozel)}
        return cls([PAD, EOS] + sorted(pieces), ozel)


def proper_names(G):
    """Graftaki butun adlarin butun parcalari (birim_16.kur'daki `ozel`): bolucu bunlari kok sayar, soymaz."""
    words = lambda a: [x for w in a.split("_") for x in V.TR.get(w, w).split()]
    return frozenset(w for t in V.TIPLER for a in G["ad"][t] for w in words(a))


def tokenizer_from(js):
    if js.lstrip().startswith("{") and json.loads(js).get("type") == MorphTokenizer.KIND:
        d = json.loads(js)
        return MorphTokenizer(d["vocab"], d["ozel"])
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
def write(folder, log=print, split_genitive=False, morph=False, unit="part"):
    """Metin + tokenizer + token'lar -> <folder>/tr_bpe.json, tr_data.pt.  Izler REF_*'la tutmazsa YAZMAZ.
    morph: kok ayri ek ayri (MorphTokenizer); split_genitive: BPE + tamlayan eki ayri; ikisi de yoksa doygun BPE.
    unit: egitim birimi -- "part" (cumle ya da soru+cevap) ya da "document" (BUTUN belge: sayfa, biyografi; model_16
    gibi, belge bolunmez).  Izler, denetim ve sinav her iki halde CUMLE duzeyinde."""
    t = build_text(log)
    for got, want, what in ((t["graph"], REF_GRAPH, "graf"), (t["parts_fp"], REF_PARTS, "parca"),
                            (t["exam_fp"], REF_EXAM, "sinav")):
        assert got == want, "%s izi %s, model_16'ninki %s -- tasinan kod kaymis" % (what, got, want)
    if morph:
        import tr_morph_19 as MO
        ozel = proper_names(V.kur(SETTINGS.veri_tohum))
        texts = t["parts"] + [s for k in t["exam"] for qa in t["exam"][k] for s in qa]
        words = {w.rstrip(MO.NOKTALAMA).replace("'", "") for s in texts for w in s.split(" ")} - {""}
        _, missing = MO.denetle(words, ozel=ozel)
        log("bolucu: %d farkli kelime, bilinen kok + eke ayrismayan (butun kalan) %d: %s" % (
            len(words), len(missing), sorted(missing)[:12]))
        tok = MorphTokenizer.build(texts, ozel)
    else:
        tok = train_bpe(t["parts"], split_genitive=split_genitive)
    units = t["parts"] if unit == "part" else t["documents"]
    ids = encode(tok, units)
    back = sum(tok.decode(x) == p for x, p in zip(ids, units))
    assert back == len(ids), "gidis-donus %d / %d" % (back, len(ids))
    flat = np.fromiter((i for x in ids for i in x), np.int16, sum(len(x) for x in ids))
    offsets = np.cumsum([0] + [len(x) for x in ids]).astype(np.int64)
    js = tok.to_str()
    d = dict(ids=torch.from_numpy(flat), offsets=torch.from_numpy(offsets), exam=t["exam"], tokenizer=js, unit=unit,
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


EXAM_SEED = 13


def exam_pairs(d, name, limit=None):
    """Bolmenin sorulari; limit verilirse sabit tohumlu rastgele ornek, ilk N DEGIL: bolmeler iliskiye gore sirali
    (ezber_olgu'nun ilk 500'u yalniz dort aile iliskisi, 26 Eylul)."""
    pairs = d["exam"][name]
    if not limit or limit >= len(pairs):
        return pairs
    return [pairs[i] for i in sorted(random.Random(EXAM_SEED).sample(range(len(pairs)), limit))]


def exam_accuracy(m, tok, d, name, limit=None, device="cuda"):
    pairs = exam_pairs(d, name, limit)
    if not pairs:
        return float("nan"), 0
    said = ask(m, tok, [q for q, _ in pairs], device)
    return sum(correct(s, a) for s, (_, a) in zip(said, pairs)) / len(pairs), len(pairs)


def make_metric(d, train_windows, device="cuda", limit=2000, exam_limit=500, probes=4):
    """train_19'un bekledigi metric(m, "train"|"heldout", full, save, health) -> {accuracy, ce, diag}.
    train: ilk `limit` pencerede token accuracy ve ce.  heldout: SINAV -- MAIN'in alti bolmesinin birebir dogru
    ortalamasi; ce cevap token'larinda.  full: bolmelerin tamami."""
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
            pairs = [p for k in MAIN for p in exam_pairs(d, k, n)]
            r = {"accuracy": float(np.mean(list(acc.values()))), "ce": answer_ce(m, tok, pairs, device),
                 "diag": dict(acc, kapi_kisayolsuz=exam_accuracy(m, tok, d, "kapi_kisayolsuz", n, device)[0])}
        r["timing"] = {"eval": time.time() - t0}
        return r

    f.health = False
    # tam yedekte decompose icin sabit istemler: her ana bolmeden `probes` soru, token'lari hazir (DC.greedy <eos> ekler)
    f.probes = [["%s %d" % (k, i + 1), encode(tok, [q])[0]] for k in MAIN for i, (q, _) in enumerate(exam_pairs(d, k, probes))]
    return f


# --- nitel sinav (kural 12) ---
def exam_text(m, tok, d, per_split=6, seed=7, device="cpu"):
    """Her bolmeden sabit tohumla `per_split` soru -> satirlar (soru, modelin yazdigi, dogrusu); defterin 6 SINAV
    hucresiyle ayni ornek."""
    lines = []
    for split in SHOW:
        pairs = random.Random(seed).sample(d["exam"][split], per_split)
        said = ask(m, tok, [q for q, _ in pairs], device=device)
        lines.append("-- %s  (%d / %d)" % (split, sum(correct(s, a) for s, (_, a) in zip(said, pairs)), per_split))
        for s, (q, a) in zip(said, pairs):
            lines.append("   %s %s\n       model: %s\n       dogru: %s" % ("+" if correct(s, a) else "x", q, s, a))
    return lines


HOP_SPLITS = ("ezber_zincir", "cikarim_gorulmemis", "cikarim_yabanci")


def chain_items(d, splits=HOP_SPLITS):
    """2R bolmelerinin yapisi -> ({bolme: [(bas, (r1, r2), kopru, cevap)]}, soru kurucusu, graf), sinavdaki sirayla.
    Yapi grafdan yeniden kurulur; yuzeyi Drive'daki sinavla birebir tutmazsa DURUR."""
    v = TS.veri_kur(SETTINGS, yaz=lambda *a: None)
    G = V.kur(SETTINGS.veri_tohum)
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    tip = {a: v.tip_ad[int(v.tip[i])] for i, a in enumerate(E)}
    surface = lambda x, rr, a: tuple(TT.sinav_yuzeyi(x, list(rr), a, tip[a])[:2])
    out = {}
    for s in splits:
        out[s] = [(x, rr, br[0], a) for x, rr, br, a in raw_items(v, G, s)]
        assert [surface(x, rr, a) for x, rr, _, a in out[s]] == [tuple(p) for p in d["exam"][s]], \
            s + ": grafdan kurulan sorular Drive'daki sinavla tutmuyor"
    return out, surface, G, tip


def chain_class(G, x, r2, a):
    """2R zincirinin sinifi (tr_splits_19): DONUS cevap basin kendisi; AYNI kisayol r2(bas) ayni cevabi verir;
    AYIRT kisayol baska cevap; YOK basin r2'si yok; EKSIK tip r2'yi alir ama olgu yok.  ezber_zincir hepsini alir,
    cikarim bolmeleri yalniz AYIRT / YOK."""
    cut = G["olgu"].get((x, r2))
    if a == x:
        return "DONUS"
    if cut == a:
        return "AYNI"
    if cut is None:
        return "EKSIK" if G["tip"][x] in G["sema"][r2] else "YOK"
    return "AYIRT"


def class_accuracy(m, tok, items, surface, G, limit=200, seed=5, device="cpu"):
    """Bolme basina 2R dogrulugu zincir sinifina gore (sinif basina en cok `limit` soru, sabit tohum): DONUS ve AYNI
    iki adim istemez, bolme ortalamasini sisirir."""
    lines = ["bolme                sinif    pay     dogru   (soru)"]
    for split, its in items.items():
        cls = [chain_class(G, x, rr[1], a) for x, rr, _, a in its]
        rng = random.Random(seed)
        for c in sorted(set(cls)):
            idx = [i for i, z in enumerate(cls) if z == c]
            pick = rng.sample(idx, min(limit, len(idx)))
            qa = [surface(its[i][0], its[i][1], its[i][3]) for i in pick]
            said = ask(m, tok, [q for q, _ in qa], device=device)
            acc = np.mean([correct(x, g) for x, (_, g) in zip(said, qa)])
            lines.append("%-20s %-7s %6.3f %8.3f   (%d)" % (split, c, len(idx) / len(cls), acc, len(pick)))
    return lines


@torch.no_grad()
def passes_report(m, tok, d, splits=SHOW, limit=200, device="cpu"):
    """LoopedRelation: soru isteminin son konumunda (cevabin ilk token'ini tahmin eden) gecis; bolme basina medyan,
    ortalama, K_MAX'a varan pay.  IKI olcu: durma kurali (onek: bir konum oncekiler durmadan duramaz, istem boyuyla
    kendiliginden artar -- 2R istemi 1R'den uzun) ve konumun KENDI degisiminin esik altina ilk dustugu gecis (boydan
    bagimsiz).  Hop sayisini izleyip izlemedigi ikincisiyle okunur."""
    vocab = tok.get_vocab()
    lines = ["bolme                 soru  boy |  durma kurali: medyan  ort  K_MAX'a  |  kendi degisimi: medyan  ort"]
    for split in splits:
        groups = collections.defaultdict(list)
        for q, _ in exam_pairs(d, split, limit):
            e = [vocab[EOS]] + encode(tok, [q])[0]
            groups[len(e)].append(e)
        got, own, lens = [], [], []
        for L, rows in groups.items():
            t = torch.tensor(rows, device=device)
            got += m.run(t, stop=True)[1]["passes"][:, -1].tolist()
            own += m.settle_passes(t)[:, -1].tolist()
            lens += [L] * len(rows)
        lines.append("%-20s %5d %4.1f |  %20.1f %5.2f %7.0f%%  |  %22.1f %4.2f" % (
            split, len(got), float(np.mean(lens)), float(np.median(got)), float(np.mean(got)),
            100 * float(np.mean([g == m.k_max for g in got])), float(np.median(own)), float(np.mean(own))))
    return lines


def classes_cli(run_dir, step=None):
    """python data_tr_19.py --classes <kosu klasoru> [--adim N]: 2R dogrulugu zincir sinifina gore (iki mimari de);
    LoopedRelation'da ayrica bolme basina durma gecisi.  -> <kosu>/sinav/<paket>_classes.txt"""
    from looped_19 import model_from_package
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = model_from_package(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    items, surface, G, tip = chain_items(d)
    lines = ["%s   %s   adim %s   %s, CPU" % (os.path.basename(os.path.abspath(run_dir)), os.path.basename(path),
                                        f"{k['step']:,}", k.get("arch")), "",
             "2R DOGRULUK x ZINCIR SINIFI (sinif basina 200 soru, tohum 5)"] + class_accuracy(m, tok, items, surface, G)
    if getattr(m, "arch", None) == "looped_relation":
        lines += ["", "DURMA: soru isteminin son konumunda gecis (esik %g, en cok %d)" % (m.stop_eps, m.k_max)]
        lines += passes_report(m, tok, d)
    lines.append("(%.0f sn)" % (time.time() - t0))
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + "_classes.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> " + out)


def hop_report(m, tok, items, surface, G, device="cpu"):
    """Her 2R sorusu: 2R cevabi, 1. adim (bas r1 -> kopru), 2. adim (kopru r2 -> cevap), kisayol (bas r2) ve yanlisin
    turu (kopru: 1. adimda kaldi, kisayol: r2'yi basa uyguladi, diger).  Sonra modelin dogru cevaptan ILK AYRILDIGI
    token'da decompose: dogru token - secilen token, parca parca (dogruysa ilk token, ikinci adaya karsi).
    -> (satirlar, [(2R dogru, 1. adim dogru, 2. adim dogru, tur)])."""
    import decompose_19 as DC
    vocab = vocab_of(tok)
    ix = {a: i for i, a in enumerate(vocab)}
    qs = []
    for x, (r1, r2), b, a in items:
        cut = G["olgu"].get((x, r2))
        qs += [surface(x, (r1, r2), a), surface(x, (r1,), b), surface(b, (r2,), a)]
        qs.append(surface(x, (r2,), cut) if cut else None)
    said = ask(m, tok, [q[0] for q in qs if q], device=device)
    said = iter(said)
    said = [next(said) if q else None for q in qs]
    lines, stats = [], []
    for i, (x, (r1, r2), b, a) in enumerate(items):
        (q2, a2), (q1, b1), (qb, a1), cut = qs[4 * i:4 * i + 4]
        s2, s1, sb, sc = said[4 * i:4 * i + 4]
        ok2, ok1, okb = correct(s2, a2), correct(s1, b1), correct(sb, a1)
        kind = "dogru" if ok2 else "kopru" if correct(s2, b1) else "kisayol" if cut and correct(s2, cut[1]) else "diger"
        stats.append((ok2, ok1, okb, kind))
        lines.append("\n[%d] %s" % (i + 1, q2))
        lines.append("   2R       model: %-34s dogru: %-26s -> %s" % (s2, a2, kind.upper()))
        lines.append("   adim 1   %s  model: %s  dogru: %s  %s" % (q1, s1, b1, "+" if ok1 else "x"))
        lines.append("   adim 2   %s  model: %s  dogru: %s  %s" % (qb, sb, a1, "+" if okb else "x"))
        if cut:
            lines.append("   kisayol  %s -> %s" % (cut[0], cut[1]))
        # decompose: dogru cevabin token'lari ile modelin devami ILK nerede ayriliyor
        tokens, n_prompt = DC.greedy(m, encode(tok, [q2])[0], vocab, ix, steps=MAX_ANSWER)
        gold = encode(tok, [a2])[0]
        own = tokens[n_prompt:].tolist()
        k = next((j for j in range(len(gold)) if j >= len(own) or own[j] != gold[j]), 0)
        t = n_prompt - 1 + k
        R = DC.forward_parts(m, tokens)
        DC.verify(m, R, tokens)
        lp = R["logp"][t]
        order = lp.argsort(descending=True).tolist()
        a_tok = gold[k]
        b_tok = own[k] if k < len(own) and own[k] != a_tok else next(j for j in order if j != a_tok)
        where = lambda w: "%s p %.3f sira %d" % (vocab[w], float(lp[w].exp()), order.index(w) + 1)
        probe = ["dogru " + where(a_tok), "secilen " + where(b_tok)]
        if k == 0:
            probe.append("kopru " + where(encode(tok, [b1])[0][0]))
            if cut:
                probe.append("kisayol " + where(encode(tok, [cut[1]])[0][0]))
        lines.append("   ayrilan token %d (%s): %s" % (k + 1, vocab[int(tokens[t])], "  |  ".join(probe)))
        DC.show(m, R, tokens, vocab, t, a_tok, b_tok, log=lambda s: lines.append("  " + s))
    return lines, stats


def hops_cli(run_dir, step=None, per_split=6, seed=7):
    """python data_tr_19.py --hops <kosu klasoru> [--adim N] [--n 6]: 2R teshisi CPU'da, <kosu>/sinav/<paket>_hops.txt.
    Sorular nitel sinavinkiyle ayni (tohum 7)."""
    from model_19 import PointRelation
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = point_relation_only(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    items, surface, G, _ = chain_items(d)
    lines = ["%s   %s   adim %s   2R teshisi: bolme basina %d soru, tohum %d, CPU" % (
        os.path.basename(os.path.abspath(run_dir)), os.path.basename(path), f"{k['step']:,}", per_split, seed)]
    summary = []
    for split in HOP_SPLITS:
        pick = random.Random(seed).sample(range(len(items[split])), per_split)
        body, st = hop_report(m, tok, [items[split][i] for i in pick], surface, G)
        kinds = collections.Counter(x[3] for x in st)
        summary.append("%-20s 2R %d/%d   adim 1 %d/%d   adim 2 %d/%d   iki adim da dogru %d, onlarda 2R %d   yanlis: %s" % (
            split, sum(x[0] for x in st), len(st), sum(x[1] for x in st), len(st), sum(x[2] for x in st), len(st),
            sum(x[1] and x[2] for x in st), sum(x[0] and x[1] and x[2] for x in st),
            "  ".join("%s %d" % (y, kinds[y]) for y in ("kopru", "kisayol", "diger") if kinds[y])))
        lines += ["", "=" * 100, "-- " + split] + body
    lines = lines[:1] + ["OZET"] + summary + lines[1:] + ["(%.0f sn)" % (time.time() - t0)]
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + "_hops.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:2 + len(summary)]))
    print("-> " + out)


# --- 2R analizi: katman katman, konum konum, tek adimlarla karsilastirma, egitimdeki siklik ---
@torch.no_grad()
def stages(m, R, t):
    """Konum t'de ara durumlar modelin kendi sirasiyla: C, +katman 0, +attention, ..., son = C_m.
    norm_inputs (Oe15): her girdi once kureye."""
    enter = unit_input(m)
    x, out = R["C"][t], [("C", R["C"][t])]
    for i, L in enumerate(R["layers"]):
        x = enter(x) + L["delta"][t]
        out.append(("katman %d" % i, x))
        for k, a in enumerate(R["attns"]):
            if a["after"] == i:
                x = enter(x) + a["out"][t]
                out.append(("attention %d" % (k + 1), x))
    assert torch.allclose(x, R["C_m"][t], atol=1e-4 * (1 + float(x.abs().max()))), "asamalar C_m'yi vermiyor"
    return out


def unit_input(m):
    """Katman / attention girdisi: norm_inputs (Oe15) acikken kureye, degilse oldugu gibi."""
    return (lambda x: torch.nn.functional.normalize(x, dim=-1)) if getattr(m, "norm_inputs", False) else (lambda x: x)


def stage_turns(m, R, t):
    """Konum t'de asama basina: (ad, eklenenin boyu, girdinin boyu, durumun dondugu aci derece).  Bir parca oncekileri
    ezerse aci ~90, ondan sonraki parca ~0 (Oe15)."""
    enter = unit_input(m)
    out, prev = [], None
    for name, x in stages(m, R, t):
        if prev is not None:
            base = enter(prev)
            cos = torch.nn.functional.cosine_similarity(x, base, dim=0).clamp(-1, 1)
            out.append((name, float((x - base).norm()), float(prev.norm()), float(torch.rad2deg(torch.arccos(cos)))))
        prev = x
    return out


def turn_report(m, tok, items, surface, per_split=60, seed=7):
    """Cevap konumunda asama basina medyan: eklenen boy / girdi boyu ve aci; 2R ve iki tek adim (Oe15 okumasi)."""
    import decompose_19 as DC
    vocab = vocab_of(tok)
    ix = {a: i for i, a in enumerate(vocab)}
    lines = []
    for split, its in items.items():
        pick = random.Random(seed).sample(range(len(its)), min(per_split, len(its)))
        got = collections.defaultdict(list)
        for i in pick:
            x, (r1, r2), b, a = its[i]
            for kind, q in (("2R", surface(x, (r1, r2), a)[0]), ("adim 1", surface(x, (r1,), b)[0]),
                            ("adim 2", surface(b, (r2,), a)[0])):
                tokens = torch.tensor([ix[EOS]] + encode(tok, [q])[0])
                R = DC.forward_parts(m, tokens)
                for name, add, base, deg in stage_turns(m, R, len(tokens) - 1):
                    got[(kind, name)].append((add, base, deg))
        names = [n for (k, n) in got if k == "2R"]
        lines.append("%-20s cevap konumu, medyan  eklenen boy / girdi boyu ; aci derece" % split)
        for kind in ("2R", "adim 1", "adim 2"):
            lines.append("   %-8s %s" % (kind, "   ".join("%s %.1f/%.1f ;%3.0f" % (
                n.replace("attention ", "att").replace("katman ", "k"),
                np.median([v[0] for v in got[(kind, n)]]), np.median([v[1] for v in got[(kind, n)]]),
                np.median([v[2] for v in got[(kind, n)]])) for n in names)))
    return lines


def stage_rank(m, x, w, pool=None):
    """Ara durum sozluge yansitilirsa (okuma gibi: norm(x) . P) token w kacinci; pool verilirse ayrica yalniz o
    token'lar arasinda (tip ici).  readout varsa o dahil degil.  -> (sira, tip ici sira ya da None, en ustteki)"""
    s = torch.nn.functional.normalize(x, dim=-1) @ m.P.T
    return (int((s > s[w]).sum()) + 1, None if pool is None else int((s[pool] > s[w]).sum()) + 1, int(s.argmax()))


def type_pools(tok, tip):
    """Tip -> o tipteki butun varliklarin ilk token'lari (tip ici sira icin; ayni ilk token bir kez)."""
    pools = collections.defaultdict(set)
    for a, ty in tip.items():
        pools[ty].add(encode(tok, [TT.ad_tr(a)])[0][0])
    return {ty: torch.tensor(sorted(s)) for ty, s in pools.items()}


GENITIVE = ("nın", "nin", "nun", "nün", "ın", "in", "un", "ün")


def relation_keys(tokens, vocab):
    """Istemde bas varligin ekinden sonraki iliskilerin ARAMA konumlari, sirayla (r1, r2, ...): tamlayan ekli kelimede
    ekten onceki son parca ("▁kardeş i|nin" -> "i"; kelime basina token'da kelimenin kendisi), son iliskide soru
    kelimesinden onceki son parca ("yaşadığı yer" -> "yer")."""
    t = len(tokens) - 1
    ap = tokens.index(vocab.index("'"))
    starts = [j for j in range(ap + 1, t) if vocab[tokens[j]].startswith("▁")]
    words = list(zip(starts, starts[1:] + [t]))
    keys = []
    for a, b in words[:-2]:
        last = vocab[tokens[b - 1]].lstrip("▁")
        if b - 1 > a and last in GENITIVE:
            keys.append(b - 2)
        elif last.endswith(GENITIVE) and b - 1 == a:
            keys.append(a)
    return keys + [words[-2][1] - 1]


def score_rank(R, t, w, pool=None):
    """Modelin gercek okumasinda (puan, R_PC dahil, defter haric) token w kacinci -> stage_rank gibi."""
    s = R["score"][t]
    return (int((s > s[w]).sum()) + 1, None if pool is None else int((s[pool] > s[w]).sum()) + 1, int(s.argmax()))


@torch.no_grad()
def answer_parts(m, R, tokens, t, a, names, focus=()):
    """Konum t'de a'nin sozluk ortalamasina karsi puani parca parca, her attention ayri; bas varligin ad konumlarindan
    gelen ayri.  A<k> ad / A<k> odak: attention k'nin (baslar ortalamasi) ad konumlarina / focus konumlarina agirligi.
    Parcalarin toplami "toplam"."""
    import decompose_19 as DC
    ex = DC.explain(m, R, tokens, t, a)
    lp = R["logp"][t]
    nm = torch.zeros(t + 1, dtype=torch.bool)
    nm[names] = True
    chain = ex["chain"].sum(1)
    out = {"p": float(lp[a].exp()), "sira": int((lp > lp[a]).sum()) + 1, "zincir ad": float(chain[nm].sum()),
           "zincir diger": float(chain[~nm].sum()), "toplam": ex["total"]}
    for i in range(len(m.moves)):
        out["katman %d" % i] = ex["parts"]["layer_%d" % i]
    for k, h in enumerate(ex["heads"]):
        hs, A = h.sum(0), R["attns"][k]["A"][:, t, :t + 1]
        out.update({"att%d ad" % (k + 1): float(hs[nm].sum()), "att%d diger" % (k + 1): float(hs[~nm].sum()),
                    "A%d ad" % (k + 1): float(A[:, nm].sum(-1).mean()),
                    "A%d odak" % (k + 1): float(A[:, list(focus)].sum(-1).mean()) if len(focus) else float("nan")})
    if "readout" in ex["parts"]:
        out["readout"] = ex["parts"]["readout"]
    return out


def part_rows(m):
    """answer_parts'in puan satirlari modelin kendi sirasiyla; son satir "toplam" (ustundekilerin toplami)."""
    rows = ["zincir ad", "zincir diger"]
    for i in range(len(m.moves)):
        rows.append("katman %d" % i)
        if i in m.attn_after:
            k = m.attn_after.index(i) + 1
            rows += ["att%d ad" % k, "att%d diger" % k]
    return tuple(rows + (["readout"] if m.readout is not None else []) + ["toplam"])


def weight_rows(m):
    """answer_parts'in attention agirligi satirlari."""
    return tuple("A%d ad" % (k + 1) for k in range(len(m.attns))) + tuple("A%d odak" % (k + 1) for k in range(len(m.attns)))


@torch.no_grad()
def embedding_report(m, tok, pools, pairs=(("▁Ayşe", "▁Fatma"), ("▁Ayşe", "▁Ahmet"), ("▁Ayşe", "▁Ankara"),
                                           ("▁Yılmaz", "▁Kaya"))):
    """E'nin ogrendigi yakinlik: tip basina ilk token'larin E cos ortalamasi, tip ici ve tip disi; ornek ciftler."""
    vocab = vocab_of(tok)
    ix = {w: i for i, w in enumerate(vocab)}
    E = torch.nn.functional.normalize(m.embed(torch.arange(len(vocab))), dim=-1)
    lines = ["E YAKINLIGI (%s)" % ("embed acik" if m.embed_delta is not None else "embed kapali: E = P")]
    for ty in sorted(pools):
        ids = pools[ty]
        n = len(ids)
        rest = torch.cat([pools[o] for o in sorted(pools) if o != ty])
        within = float(((E[ids] @ E[ids].T).sum() - n) / max(n * (n - 1), 1))
        lines.append("   %-11s n %3d   tip ici cos %+.3f   tip disi %+.3f" % (ty, n, within, float((E[ids] @ E[rest].T).mean())))
    lines.append("   " + "   ".join("%s.%s %+.3f" % (a, b, float(E[ix[a]] @ E[ix[b]])) for a, b in pairs if a in ix and b in ix))
    return lines


def training_parts(d, tok):
    """tr_data token'larindan egitim parcalarinin (cumle) metni; belge birimi cumlelere bolunur (build_text'teki gibi)."""
    ids, off = d["ids"].numpy(), d["offsets"].numpy()
    texts = tok.decode_batch([ids[off[i]:off[i + 1]].tolist() for i in range(len(off) - 1)])
    return [p for x in texts for p in re.split(r"(?<=\.)\s+", x.strip()) if p]      # belge birimi -> cumleler


def count_written(idx, parts, head, rels, answer):
    """Zincir ifadesi X icin kac egitim parcasinda CEVABIYLA birlikte yazili."""
    xn, p, y = phrase(head, rels, answer)
    return sum(p in parts[i] and y in parts[i] for i in set(idx.get(xn, ())))


def frequency_report(idx, parts, items):
    """Bolme basina: 2R zincirin ve iki tek adimin egitimde cevabiyla kac kez yazildigi, dagilim."""
    q = lambda c: "ort %6.1f  medyan %4d  %%10 %4d  %%90 %4d  en az %4d  en cok %4d" % (
        np.mean(c), np.median(c), np.percentile(c, 10), np.percentile(c, 90), min(c), max(c))
    lines = []
    for split, its in items.items():
        c2 = [count_written(idx, parts, x, rr, a) for x, rr, _, a in its]
        c1 = [count_written(idx, parts, x, rr[:1], b) for x, rr, b, _ in its]
        cb = [count_written(idx, parts, b, rr[1:], a) for _, rr, b, a in its]
        lines += ["%-20s n %d" % (split, len(its)), "   2R zincir   " + q(c2), "   adim 1      " + q(c1),
                  "   adim 2      " + q(cb)]
    return lines


def accuracy_by_frequency(m, tok, idx, parts, items, surface, limit=3000, seed=11, device="cpu"):
    """Ezber 2R: dogruluk, zincirin egitimde cevabiyla kac kez yazildigina gore (rastgele `limit` soru)."""
    pick = random.Random(seed).sample(range(len(items)), min(limit, len(items)))
    its = [items[i] for i in pick]
    qa = [surface(x, rr, a) for x, rr, _, a in its]
    said = ask(m, tok, [q for q, _ in qa], device=device)
    rows = [(count_written(idx, parts, x, rr, a), correct(s, g)) for (x, rr, _, a), s, (_, g) in zip(its, said, qa)]
    lines = ["kac kez yazili      soru    dogru"]
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 10 ** 6)):
        ok = [c for n, c in rows if lo <= n <= hi]
        if ok:
            lines.append("   %-14s %6d    %.3f" % ("%d" % lo if lo == hi else "%d+" % lo if hi > 10 ** 5 else "%d-%d" % (lo, hi),
                                                  len(ok), np.mean(ok)))
    lines.append("   %-14s %6d    %.3f" % ("hepsi", len(rows), np.mean([c for _, c in rows])))
    return lines


def hop_analysis(m, tok, items, surface, G, tip, pools, idx, parts, detail=6):
    """Her 2R sorusu ve iki tek adimi (istemin son konumunda, dogru cevabin ilk token'i):
    (a) katman katman sira -- 2R'de dogru, kopru ve kisayol; tek adimlarda dogru; hem butun sozlukte hem tip icinde;
    (b) dogru token'in puani parca parca (answer_parts);
    (d) 2R isteminde kopru ve dogru cevap konum konum: r1 kelimesi, r2 kelimesi, cevap konumu (tip ici sira).
    Ayrinti satirlari ilk `detail` soru icin.  -> (satirlar, {tur: [answer_parts]}, {(tur, asama): [sira]},
    {(tur, asama): [tip ici sira]}, {(izlenen, konum, asama): [tip ici sira]})"""
    import decompose_19 as DC
    vocab = vocab_of(tok)
    ix = {a: i for i, a in enumerate(vocab)}
    first = lambda s: encode(tok, [s])[0][0]
    kinds = ("2R", "adim 1", "adim 2")
    lines, table = [], {k: [] for k in kinds}
    ranks, type_ranks, lens = (collections.defaultdict(list) for _ in range(3))
    for n, (x, (r1, r2), b, a) in enumerate(items):
        show = n < detail
        (q2, a2), (q1, b1), (qb, a1) = surface(x, (r1, r2), a), surface(x, (r1,), b), surface(b, (r2,), a)
        cut_e = G["olgu"].get((x, r2))
        cut = surface(x, (r2,), cut_e)[1] if cut_e else None
        if show:
            seen = (count_written(idx, parts, x, (r1, r2), a), count_written(idx, parts, x, (r1,), b),
                    count_written(idx, parts, b, (r2,), a))
            lines.append("\n[%d] %s   dogru: %s   kopru: %s%s" % (n + 1, q2, a2, b1, "   kisayol: " + cut if cut else ""))
            lines.append("   egitimde cevabiyla: 2R zincir %d kez   adim 1 %d   adim 2 %d" % seen)
        rows = {}
        for kind, q, gold, gold_e in (("2R", q2, a2, a), ("adim 1", q1, b1, b), ("adim 2", qb, a1, a)):
            tokens = torch.tensor([ix[EOS]] + encode(tok, [q])[0])
            R = DC.forward_parts(m, tokens)
            DC.verify(m, R, tokens)
            t = len(tokens) - 1
            apos = tokens.tolist().index(ix["'"])
            words = relation_keys(tokens.tolist(), vocab)
            table[kind].append(answer_parts(m, R, tokens, t, first(gold), list(range(1, apos)), focus=words[:1]))
            watch = [("dogru", first(gold), pools[tip[gold_e]])]
            if kind == "2R":
                watch.append(("kopru", first(b1), pools[tip[b]]))
                if cut_e:
                    watch.append(("kisayol", first(cut), pools[tip[cut_e]]))
            at = lambda p: [(s, [stage_rank(m, v, w, pool) for _, w, pool in watch]) for s, v in stages(m, R, p)] + [
                ("okuma", [score_rank(R, p, w, pool) for _, w, pool in watch])]
            for stage, got in at(t):
                ranks[(kind, stage)].append(got[0][0])
                type_ranks[(kind, stage)].append(got[0][1])
                if kind == "2R":
                    type_ranks[("2R kopru", stage)].append(got[1][1])
                rows.setdefault(stage, []).append("  ".join("%s %d/%d" % (lab, r, rp) for (lab, _, _), (r, rp, _) in
                                                            zip(watch, got)) + "  (ust %s)" % vocab[got[0][2]])
            if kind == "2R":
                for label, p in (("r1 kelimesi", words[0]), ("r2 kelimesi", words[1] if len(words) > 1 else None),
                                 ("cevap konumu", t)):
                    if p is None:
                        continue
                    here = at(p)
                    for stage, got in here:
                        for (lab, _, _), g in zip(watch[:2], got[:2]):
                            lens[(lab, label, stage)].append(g[1])
                    if show:
                        lines.append("   konum %-13s %-14s kopru tip ici sira: %s" % (
                            label, vocab[int(tokens[p])], "  ".join("%s %d" % (s, got[1][1]) for s, got in here)))
        if show:
            lines.append("   asama          " + " | ".join("%-50s" % k for k in kinds) + "   (sira/tip ici sira)")
            for stage, cells in rows.items():
                lines.append("   %-14s %s" % (stage, " | ".join("%-50s" % c for c in cells)))
            lines.append("   dogru token'in puani (sozluk ortalamasina karsi)   " + "   ".join("%-8s" % k for k in kinds))
            lines.append("   %-50s %s" % ("p / sira", "   ".join("%.3f/%-3d" % (table[k][-1]["p"], table[k][-1]["sira"])
                                                                    for k in kinds)))
            for row in part_rows(m) + weight_rows(m):
                lines.append("   %-50s %s" % (row, "   ".join("%+8.2f" % table[k][-1][row] for k in kinds)))
    return lines, table, ranks, type_ranks, lens


def analysis_cli(run_dir, step=None, per_split=60, detail=6, seed=7):
    """python data_tr_19.py --analysis <kosu klasoru> [--adim N] [--n 60]: 2R analizi CPU'da,
    <kosu>/sinav/<paket>_analysis.txt.  Ilk `detail` soru nitel sinavinkiyle ayni (tohum 7); ozet `per_split` soruda;
    siklik butun bolmede; siklik x dogruluk ezber 2R'den 3.000 soruda."""
    from model_19 import PointRelation
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = point_relation_only(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    items, surface, G, tip = chain_items(d)
    pools = type_pools(tok, tip)
    parts = training_parts(d, tok)
    idx = possessor_index(parts)
    head = ["%s   %s   adim %s   2R analizi: ozet bolme basina %d soru, ayrinti ilk %d, tohum %d, CPU" % (
        os.path.basename(os.path.abspath(run_dir)), os.path.basename(path), f"{k['step']:,}", per_split, detail, seed),
        "tip ici havuz (ilk token): " + "  ".join("%s %d" % (ty, len(p)) for ty, p in sorted(pools.items())),
        "asamalar norm(durum).P ile okunur (R_PC haric); 'okuma' modelin gercek puani (R_PC dahil, defter haric)",
        "A<k> odak: attention k'nin ilk iliski kelimesine (2R'de r1) agirligi", ""] + embedding_report(m, tok, pools)
    head += ["", "EGITIMDE KAC KEZ YAZILI (cevabiyla, butun bolme)"] + frequency_report(idx, parts, items)
    head += ["", "EZBER 2R: DOGRULUK x SIKLIK (3.000 soru, tohum 11)"] + accuracy_by_frequency(
        m, tok, idx, parts, items["ezber_zincir"], surface)
    head += ["", "2R DOGRULUK x ZINCIR SINIFI (sinif basina 200 soru, tohum 5)"] + class_accuracy(m, tok, items, surface, G)
    head += ["", "OLCEK (Oe15): asama basina eklenen boy ve aci, cevap konumunda (bolme basina %d soru)" % per_split] + \
        turn_report(m, tok, items, surface, per_split, seed)
    summary, body = [], []
    for split in HOP_SPLITS:
        pick = random.Random(seed).sample(range(len(items[split])), per_split)
        lines, table, ranks, type_ranks, lens = hop_analysis(m, tok, [items[split][i] for i in pick], surface, G, tip,
                                                             pools, idx, parts, detail)
        kinds = list(table)
        stage_names = [s for (kk, s) in ranks if kk == "2R"]
        summary += ["", "-- %s  (%d soru)" % (split, per_split),
                    "   dogru token'in puani, parca parca, ortalama" + " " * 7 + "   ".join("%-8s" % x for x in kinds)]
        for row in ("p",) + part_rows(m) + weight_rows(m):
            summary.append("   %-50s %s" % (row, "   ".join("%+8.3f" % np.mean([r[row] for r in table[x]]) for x in kinds)))
        summary.append("   (a) sira asama asama, medyan: butun sozluk / tip ici" + " " * 1 + "   ".join(
            "%-11s" % x for x in kinds + ["2R kopru"]))
        for s in stage_names:
            summary.append("   %-50s %s" % (s, "   ".join("%5d/%-5d" % (np.median(ranks[(x, s)]), np.median(type_ranks[(x, s)]))
                                                          for x in kinds) + "   %5d" % np.median(type_ranks[("2R kopru", s)])))
        summary.append("   (d) 2R isteminde tip ici sira, medyan (konum x asama)   " + "  ".join("%-11s" % s for s in stage_names))
        for lab in ("kopru", "dogru"):
            for label in ("r1 kelimesi", "r2 kelimesi", "cevap konumu"):
                vals = [lens.get((lab, label, s)) for s in stage_names]
                if all(vals):
                    summary.append("   %-7s %-46s %s" % (lab, label, "  ".join("%-11d" % np.median(v) for v in vals)))
        body += ["", "=" * 100, "-- " + split] + lines
    out_lines = head + summary + body + ["(%.0f sn)" % (time.time() - t0)]
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + "_analysis.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\n".join(head + summary))
    print("-> " + out)


# --- tek soru hata ayiklama: gecmise ulasiyor mu ---
def find_item(items, surface, question):
    """Soru metni -> (bolme, (bas, iliskiler, kopru, cevap))."""
    for split, its in items.items():
        for it in its:
            if surface(it[0], it[1], it[3])[0] == question:
                return split, it
    raise KeyError(question)


def training_evidence(idx, parts, head, rels, answer):
    """Zincir ifadesinin cevabiyla gectigi egitim parcalari -> [(parca, ileri)].  ileri: cevap ifadeden SONRA gelir
    (soldan saga okuyan model ifadeyi gorup cevabi tahmin edebilir); geri: cevap once gelir."""
    xn, p, y = phrase(head, rels, answer)
    out = []
    for i in sorted(set(idx.get(xn, ()))):
        s = parts[i]
        if p in s and y in s:
            out.append((s, s.find(y, s.index(p) + len(p)) >= 0))
    return out


@torch.no_grad()
def answer_in_context(m, tok, text, answer):
    """Egitim cumlesi cevaba kadar verilir -> ([p her cevap token'i, ogretmen zorlamali], modelin acgozlu devami)."""
    import decompose_19 as DC
    vocab = vocab_of(tok)
    ix = {a: i for i, a in enumerate(vocab)}
    ids = [ix[EOS]] + encode(tok, [text])[0]
    gold = encode(tok, [answer])[0]
    j = [s for s in range(len(ids) - len(gold) + 1) if ids[s:s + len(gold)] == gold][-1]
    lp = m.scoreboard(torch.tensor(ids)[None])[0]
    tokens, n = DC.greedy(m, ids[1:j], vocab, ix, steps=len(gold) + 3)
    return [float(lp[j - 1 + k, gold[k]].exp()) for k in range(len(gold))], detok(tok, tokens[n:].tolist())


def slot_variants(tok, surface, tip, x, rr, a, slot, limit=24, seed=5):
    """Yalniz bir yuvasi degismis, AYNI uzunlukta sorular.  slot: 'ad' (bas varlik, ayni tip), 'r1' ya da 'r2'
    (iliski kelimesi).  -> [(etiket, token'lar <eos> haric)]"""
    base = encode(tok, [surface(x, rr, a)[0]])[0]
    if slot == "ad":
        pool = [(TT.ad_tr(e), surface(e, rr, a)[0]) for e in sorted(tip) if tip[e] == tip[x] and e != x]
    else:
        k = int(slot[1]) - 1
        pool = [(V.TR_ILISKI[r], surface(x, rr[:k] + (r,) + rr[k + 1:], a)[0]) for r in V.TR_ILISKI if r != rr[k]]
    random.Random(seed).shuffle(pool)
    out = []
    for lab, q in pool:
        ids = encode(tok, [q])[0]
        if len(ids) == len(base) and ids != base:
            out.append((lab, ids))
        if len(out) == limit:
            break
    return out


@torch.no_grad()
def reach(m, eos, base, swaps, positions):
    """Degistirilen yuvanin izi: konum x asama goreli durum farki |x - x'| / |x| (ortalama); cevap konumunda cikis
    dagiliminin toplam degisim mesafesi (TV) ve en ustteki token'in degisme orani."""
    import decompose_19 as DC
    tb = torch.tensor([eos] + base)
    t = len(tb) - 1
    R0 = DC.forward_parts(m, tb)
    S0 = {p: stages(m, R0, p) for p in positions}
    p0 = R0["logp"][t].exp()
    diff, tv, flip = collections.defaultdict(list), [], []
    for _, s in swaps:
        R1 = DC.forward_parts(m, torch.tensor([eos] + s))
        for p in positions:
            for (name, x0), (_, x1) in zip(S0[p], stages(m, R1, p)):
                diff[(p, name)].append(float((x0 - x1).norm() / x0.norm()))
        p1 = R1["logp"][t].exp()
        tv.append(0.5 * float((p0 - p1).abs().sum()))
        flip.append(int(p0.argmax()) != int(p1.argmax()))
    return {k: float(np.mean(v)) for k, v in diff.items()}, float(np.mean(tv)), float(np.mean(flip))


def debug_one(m, tok, surface, tip, idx, parts, x, rr, b, a, title=""):
    """Tek soru, bes bolum: 1 egitimde nasil yazili, 2 model ne diyor, 3 gecmise ulasiyor mu (yuva degistirme),
    4 cevap konumunda attention, 5 defter."""
    import decompose_19 as DC
    vocab = vocab_of(tok)
    ix = {w: i for i, w in enumerate(vocab)}
    eos = ix[EOS]
    q, gold = surface(x, rr, a)
    lines = ["", "=" * 110, "%sSORU  %s   dogru: %s%s" % (title, q, gold, "   kopru: " + TT.ad_tr(b) if b else "")]
    ev = training_evidence(idx, parts, x, rr, a)
    lines.append("1. EGITIMDE cevabiyla %d parca: ileri (cevap ifadeden sonra) %d, geri (cevap once) %d" % (
        len(ev), sum(f for _, f in ev), sum(not f for _, f in ev)))
    for s, f in ev[:10]:
        lines.append("      %s  %s" % ("ileri" if f else "geri ", s))
    said = ask(m, tok, [q], device="cpu")[0]
    lines.append("2. SORU  model: %s  %s" % (said, "+" if correct(said, gold) else "x"))
    for s, f in [e for e in ev if e[1]][:4]:
        ps, cont = answer_in_context(m, tok, s, gold)
        lines.append("   egitim cumlesi cevaba kadar: devam '%s'   p(cevap token'lari, ogretmen zorlamali) %s" % (
            cont, " ".join("%.3f" % p for p in ps)))
    base = encode(tok, [q])[0]
    tokens = [eos] + base
    t = len(tokens) - 1
    apos = tokens.index(ix["'"])
    positions = list(range(apos + 1, t + 1))
    R0 = DC.forward_parts(m, torch.tensor(tokens))
    names = [s for s, _ in stages(m, R0, t)]
    lines.append("3. GECMISE ULASIYOR MU -- tek yuva degistirilince durum ne kadar degisiyor (|x - x'| / |x|, ortalama)")
    for slot in ["ad", "r1"] + (["r2"] if len(rr) > 1 else []):
        sw = slot_variants(tok, surface, tip, x, rr, a, slot)
        if not sw:
            lines.append("   yuva %s: ayni uzunlukta degisik soru yok" % slot)
            continue
        here = positions if slot == "ad" else [t]
        diff, tv, flip = reach(m, eos, base, sw, here)
        lines.append("   yuva %-3s %2d degisik soru (%s ...)   cevap konumunda: cikis TV %.3f, en ust token degisti %%%.0f" % (
            slot, len(sw), ", ".join(lab for lab, _ in sw[:3]), tv, 100 * flip))
        lines.append("      konum              " + "".join("%-12s" % s for s in names))
        for p in here:
            lines.append("      %2d %-15s %s" % (p, vocab[tokens[p]], "".join("%-12.3f" % diff[(p, s)] for s in names)))
    lines.append("4. ATTENTION cevap konumunda (bas x kaynak token)")
    for k, att in enumerate(R0["attns"]):
        A = att["A"][:, t, :t + 1]
        for h in range(A.shape[0]):
            lines.append("   att%d bas %d  %s" % (k + 1, h, "  ".join("%s %.2f" % (vocab[tokens[j]], A[h, j])
                                                                       for j in range(t + 1))))
    if "gate" in R0:
        pc = torch.zeros(len(vocab)).scatter_add_(0, R0["cache_next"], R0["cache_w"][t])
        g = encode(tok, [gold])[0][0]
        lines.append("5. DEFTER cevap konumunda: gate %.4f   p_defter(dogru ilk token) %.3f   dogru token baglamda %s" % (
            float(R0["gate"][t]), float(pc[g]), "VAR" if g in tokens else "YOK"))
    return lines


def direction_report(m, tok, idx, parts, its, surface, limit=3000, seed=11):
    """Ezber 2R: zincirin egitimde kac kez ILERI (cevap sonra) yazildigina gore dogruluk; tek adimlarin ileri payi."""
    pick = random.Random(seed).sample(range(len(its)), min(limit, len(its)))
    sel = [its[i] for i in pick]
    qa = [surface(x, rr, a) for x, rr, _, a in sel]
    said = ask(m, tok, [q for q, _ in qa], device="cpu")
    rows = []
    for (x, rr, _, a), s, (_, g) in zip(sel, said, qa):
        ev = training_evidence(idx, parts, x, rr, a)
        rows.append((sum(f for _, f in ev), sum(not f for _, f in ev), correct(s, g)))
    lines = ["   ileri kac kez     soru    dogru     (ortalama geri)"]
    for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 10 ** 6)):
        sub = [(f, r, c) for f, r, c in rows if lo <= f <= hi]
        if sub:
            lines.append("   %-14s %6d    %.3f     %.1f" % ("%d" % lo if lo == hi else "%d+" % lo if hi > 10 ** 5 else
                                                           "%d-%d" % (lo, hi), len(sub), np.mean([c for _, _, c in sub]),
                                                           np.mean([r for _, r, _ in sub])))
    one = [training_evidence(idx, parts, x, rr[:1], b) for x, rr, b, _ in sel[:300]]
    lines.append("   toplam ileri %d, geri %d (2R zincir);  tek adim (ilk 300 sorunun 1. adimi): ileri medyan %d, geri medyan %d" % (
        sum(f for f, _, _ in rows), sum(r for _, r, _ in rows), np.median([sum(f for _, f in e) for e in one]),
        np.median([sum(not f for _, f in e) for e in one])))
    return lines


@torch.no_grad()
def patch_name_head(m, tok, q2, q1, watch):
    """Mudahale: 2R sorusunda cevap konumunda ilk attention'in ada bakan basinin (1R'de ada en cok bakan bas) satiri,
    1R sorusundaki satiriyla degistirilir (ad kismi 1R'deki gibi, kalan pay 2R'nin kendi dagilimiyla); model o konumda
    bu durumla ilerler (sonraki katmanlar ve attention'lar dahil), okuma R_PC dahil, defter haric.
    watch: [(etiket, metin)] -> ilk token'larinin sirasi.  -> satirlar"""
    import decompose_19 as DC
    vocab = vocab_of(tok)
    ix = {w: i for i, w in enumerate(vocab)}
    t2s, t1s = ([ix[EOS]] + encode(tok, [q])[0] for q in (q2, q1))
    t2, t1 = len(t2s) - 1, len(t1s) - 1
    apos = t2s.index(ix["'"])
    assert t1s[:apos + 2] == t2s[:apos + 2], "1R ve 2R istemi ad kisminda ayni degil"
    R2, R1 = DC.forward_parts(m, torch.tensor(t2s)), DC.forward_parts(m, torch.tensor(t1s))
    att = next(i for i, a in enumerate(R2["attns"]) if a["after"] == 0)
    A2, A1, ov = R2["attns"][att]["A"], R1["attns"][att]["A"], R2["attns"][att]["ov"]
    h = int(A1[:, t1, 1:apos + 2].sum(-1).argmax())
    after0 = unit_input(m)(dict(stages(m, R2, t2))["katman 0"])
    C = R2["C"][None]

    def read(A_row):
        A = A2[:, t2, :t2 + 1].clone()
        if A_row is not None:
            A[h] = A_row
        fixed = after0 + torch.einsum("hj,hjd->d", A, ov[:, :t2 + 1])
        x = C
        for i, layer in enumerate(m.moves):
            x = layer(x)[0]
            if i in m.attn_after:
                x = m.attns[m.attn_after.index(i)](C, x)
                if i == 0:
                    x = x.clone()
                    x[0, t2] = fixed
        s = torch.log_softmax(m.point(x, C)[0, t2] @ m.P.T, -1)
        top = s.topk(5)
        ranks = "  ".join("%s %s sira %d" % (lab, vocab[w], int((s > s[w]).sum()) + 1)
                          for lab, w in ((lab, encode(tok, [txt])[0][0]) for lab, txt in watch))
        return "ilk 5: %s   |  %s" % (" ".join("%s %.2f" % (vocab[int(i)], float(v.exp())) for i, v in zip(top.indices, top.values)), ranks)

    head = A1[h, t1, :apos + 2]
    rest = A2[h, t2, apos + 2:t2 + 1]
    row = torch.cat([head, rest / rest.sum() * (1 - head.sum())])
    return ["6. MUDAHALE: bas %d (1R'de ada en cok bakan) 2R'de ada bakiyor: %.2f -> %.2f (1R'deki gibi)" % (
        h, float(A2[h, t2, 1:apos + 2].sum()), float(row[1:apos + 2].sum())),
        "   oldugu gibi   " + read(None), "   mudaheleli    " + read(row)]


@torch.no_grad()
def reroute(m, R, t, k, moves):
    """Mudahale: konum t'de attention k'nin (butun baslar) agirligi src'den dst'ye tasinir (moves: [(src, dst)]); model
    o konumda bu durumla ilerler (sonraki katmanlar ve attention'lar dahil).  -> puan (n,), R_PC dahil, defter haric."""
    A = R["attns"][k]["A"][:, t, :t + 1].clone()
    for src, dst in moves:
        A[:, dst] += A[:, src]
        A[:, src] = 0
    after = R["attns"][k]["after"]
    fixed = (unit_input(m)(dict(stages(m, R, t))["katman %d" % after])
             + torch.einsum("hj,hjd->d", A, R["attns"][k]["ov"][:, :t + 1]))
    C = R["C"][None]
    x = C
    for i, layer in enumerate(m.moves):
        x = layer(x)[0]
        if i in m.attn_after:
            x = m.attns[m.attn_after.index(i)](C, x)
            if i == after:
                x = x.clone()
                x[0, t] = fixed
    return (m.point(x, C) @ m.P.T)[0, t]


def reroute_cli(run_dir, step=None, per_split=60, seed=7):
    """python data_tr_19.py --reroute <kosu klasoru> [--adim N] [--n 60]: 2R sorusunda cevap konumunda SON attention'in
    r1'in tamlayan ekine ("▁kardeş i nin" -> "nin") giden agirligi r1'in arama konumuna ("i": kopru orada) tasinirsa
    dogru cevap ve kopru kacinci; kontrol: ayni agirlik <eos>'a.  Ayrica ekin ve arama konumunun durumunda kopru
    (tip ici sira, asama asama).  -> <kosu>/sinav/<paket>_reroute.txt"""
    import decompose_19 as DC
    from model_19 import PointRelation
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = point_relation_only(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    items, surface, G, tip = chain_items(d)
    pools = type_pools(tok, tip)
    vocab = vocab_of(tok)
    ix = {w: i for i, w in enumerate(vocab)}
    first = lambda s: encode(tok, [s])[0][0]
    att = len(m.attns) - 1
    variants = (("oldugu gibi", lambda key, gen: []), ("ek -> arama", lambda key, gen: [(gen, key)]),
                ("kontrol ek -> <eos>", lambda key, gen: [(gen, 0)]))
    lines = ["%s   %s   adim %s   yeniden yonlendirme: cevap konumunda attention %d, bolme basina %d soru, tohum %d, CPU" % (
        os.path.basename(os.path.abspath(run_dir)), os.path.basename(path), f"{k['step']:,}", att + 1, per_split, seed),
        "okuma R_PC dahil, defter haric; ilk token dogrulugu = en ust token dogru cevabin ilk token'i"]
    for split in HOP_SPLITS:
        pick = random.Random(seed).sample(range(len(items[split])), per_split)
        res = collections.defaultdict(list)
        at_pos = collections.defaultdict(list)
        moved = []
        for n in pick:
            x, (r1, r2), b, a = items[split][n]
            q2, a2 = surface(x, (r1, r2), a)
            b1 = surface(x, (r1,), b)[1]
            tokens = torch.tensor([ix[EOS]] + encode(tok, [q2])[0])
            R = DC.forward_parts(m, tokens)
            t = len(tokens) - 1
            key = relation_keys(tokens.tolist(), vocab)[0]
            gen = key + 1
            assert vocab[int(tokens[gen])].lstrip("▁") in GENITIVE, "r1'den sonra tamlayan eki yok: %s" % q2
            wa, wb = first(a2), first(b1)
            pa, pb = pools[tip[a]], pools[tip[b]]
            moved.append(float(R["attns"][att]["A"][:, t, gen].mean()))
            for label, mv in variants:
                sc = reroute(m, R, t, att, mv(key, gen))
                if not mv(key, gen):
                    assert torch.allclose(sc, R["score"][t], atol=1e-3 * (1 + float(sc.abs().max()))), "mudahalesiz puan tutmuyor"
                res[(label, "dogru")].append(int((sc[pa] > sc[wa]).sum()) + 1)
                res[(label, "kopru")].append(int((sc[pb] > sc[wb]).sum()) + 1)
                res[(label, "ilk token")].append(int(sc.argmax()) == wa)
                res[(label, "p")].append(float(torch.softmax(sc, -1)[wa]))
            for where, pos in (("arama (%s)" % "i", key), ("ek", gen)):
                for stage, v in stages(m, R, pos):
                    at_pos[(where, stage)].append(stage_rank(m, v, wb, pb)[1])
        lines += ["", "-- %s  (%d soru)   son attention'in cevap konumunda eke agirligi (baslar ortalamasi) %.2f" % (
            split, per_split, np.mean(moved)),
            "   %-22s %10s %12s %14s %12s" % ("", "ilk token", "p(dogru)", "dogru tip ici", "kopru tip ici")]
        for label, _ in variants:
            lines.append("   %-22s %10.3f %12.3f %14d %12d" % (
                label, np.mean(res[(label, "ilk token")]), np.mean(res[(label, "p")]),
                np.median(res[(label, "dogru")]), np.median(res[(label, "kopru")])))
        stage_names = [s for s, _ in stages(m, R, key)]
        lines.append("   kopru tip ici sira, medyan   " + "  ".join("%-11s" % s for s in stage_names))
        for where in ("arama (i)", "ek"):
            lines.append("   %-28s %s" % (where, "  ".join("%-11d" % np.median(at_pos[(where, s)]) for s in stage_names)))
    lines.append("(%.0f sn)" % (time.time() - t0))
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + "_reroute.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> " + out)


def debug_cli(run_dir, questions, step=None):
    """python data_tr_19.py --debug <kosu klasoru> --soru "<2R soru>" [--soru ...] [--adim N]: her 2R soru ve iki tek
    adimi icin debug_one; basta butun ezber 2R icin yon x dogruluk.  -> <kosu>/sinav/<paket>_debug.txt"""
    from model_19 import PointRelation
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = point_relation_only(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    items, surface, G, tip = chain_items(d)
    parts = training_parts(d, tok)
    idx = possessor_index(parts)
    lines = ["%s   %s   adim %s   tek soru hata ayiklama, CPU" % (
        os.path.basename(os.path.abspath(run_dir)), os.path.basename(path), f"{k['step']:,}"),
        "", "EZBER 2R, 3.000 soru (tohum 11): zincir egitimde kac kez ILERI yazili -> dogruluk"]
    lines += direction_report(m, tok, idx, parts, items["ezber_zincir"], surface)
    for q in questions:
        split, (x, rr, b, a) = find_item(items, surface, q)
        lines += debug_one(m, tok, surface, tip, idx, parts, x, rr, b, a, "[%s 2R] " % split)
        lines += debug_one(m, tok, surface, tip, idx, parts, x, rr[:1], None, b, "[kontrol: 1. adim] ")
        lines += debug_one(m, tok, surface, tip, idx, parts, b, rr[1:], None, a, "[kontrol: 2. adim] ")
        cut = G["olgu"].get((x, rr[1]))
        watch = [("dogru", surface(x, rr, a)[1]), ("kopru", TT.ad_tr(b))] + (
            [("kisayol", surface(x, rr[1:], cut)[1])] if cut else [])
        lines += ["", "[%s 2R] %s" % (split, q)] + patch_name_head(m, tok, q, surface(x, rr[:1], b)[0], watch)
    lines.append("(%.0f sn)" % (time.time() - t0))
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + "_debug.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> " + out)


def history_cli(run_dir, questions, steps=(0, 500, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 16000, 20000)):
    """python data_tr_19.py --history <kosu klasoru> --soru "<2R soru>" ...: egitim boyunca cevap konumunda bas basina
    bas varligin ADINA (ad token'lari + ek) giden attention payi ve dogru ilk token'in olasiligi; 2R soru ve 1. adimi.
    -> <kosu>/sinav/history.txt"""
    import decompose_19 as DC
    from model_19 import PointRelation
    d, tok = run_data(run_dir)
    vocab = vocab_of(tok)
    ix = {w: i for i, w in enumerate(vocab)}
    items, surface, _, _ = chain_items(d)
    probes = []
    for q in questions:
        _, (x, rr, b, a) = find_item(items, surface, q)
        probes += [("2R", q, surface(x, rr, a)[1]), ("1. adim", *surface(x, rr[:1], b))]
    lines = ["%s   egitim boyunca, cevap konumunda: attention 1'in bas varligin adina (ad + ek) payi, bas 0-3; "
             "att2>r1: ikinci attention'in (varsa) ilk iliski kelimesine payi, baslar ortalamasi; p = dogru ilk token"
             % os.path.basename(os.path.abspath(run_dir))]
    rows = {i: [] for i in range(len(probes))}
    for step in steps:
        try:
            path = latest_package(run_dir, step)
        except AssertionError:
            continue
        m = point_relation_only(torch.load(path, weights_only=False, map_location="cpu"))
        for i, (kind, q, gold) in enumerate(probes):
            tokens = [ix[EOS]] + encode(tok, [q])[0]
            t = len(tokens) - 1
            apos = tokens.index(ix["'"])
            r1 = relation_keys(tokens, vocab)[0]
            R = DC.forward_parts(m, torch.tensor(tokens))
            A = R["attns"][0]["A"][:, t, 1:apos + 2].sum(-1)
            a2 = "%.2f" % float(R["attns"][1]["A"][:, t, r1].mean()) if len(R["attns"]) > 1 else "-"
            p = float(R["logp"][t, encode(tok, [gold])[0][0]].exp())
            rows[i].append("   %6s   %s   %7s   p %.3f" % (f"{step:,}", "  ".join("%.2f" % float(v) for v in A), a2, p))
    for i, (kind, q, gold) in enumerate(probes):
        lines += ["", "%-8s %s   dogru: %s" % (kind, q, gold), "   adim     bas 0 bas 1 bas 2 bas 3   att2>r1"] + rows[i]
    out = os.path.join(run_dir, "sinav", "history.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> " + out)


def ablation_cli(run_dir, step=None, limit=500):
    """python data_tr_19.py --ablation <kosu klasoru> [--adim N] [--n 500]: ayni agirlikla R_PC ve/veya defter
    kapatilinca bolme basina birebir dogru.  Egitim onlarla yapildi: kapatmak parcasiz egitilmis bir modeli OLCMEZ, parcanin
    su anki katkisini olcer.  -> <kosu>/sinav/<paket>_ablation.txt"""
    from model_19 import PointRelation
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = point_relation_only(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    conds = [("tam", ())]
    if m.readout is not None:
        conds.append(("R_PC yok", ("readout",)))
    if m.ledger is not None:
        conds.append(("defter yok", ("ledger",)))
    if len(conds) == 3:
        conds.append(("ikisi de yok", ("readout", "ledger")))
    lines = ["%s   %s   adim %s   parca kapatma, bolme basina %d soru (sabit tohumlu ornek), CPU" % (
        os.path.basename(os.path.abspath(run_dir)), os.path.basename(path), f"{k['step']:,}", limit),
        "   %-14s %s" % ("", "  ".join("%-18s" % s for s in SHOW))]
    for name, off in conds:
        saved = {a: getattr(m, a) for a in off}
        for a in off:
            setattr(m, a, None)
        acc = [exam_accuracy(m, tok, d, s, limit, "cpu")[0] for s in SHOW]
        for a, v in saved.items():
            setattr(m, a, v)
        lines.append("   %-14s %s" % (name, "  ".join("%-18.3f" % x for x in acc)))
        print(lines[-1], flush=True)
    lines.append("(%.0f sn)" % (time.time() - t0))
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + "_ablation.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> " + out)


def run_data(run_dir, k=None):
    """Kosunun egitildigi veri (paketteki ad: data_tr, data_tr_morph ...) ve tokenizer; sozluk pakettekiyle tutmazsa durur."""
    k = k if k is not None else torch.load(latest_package(run_dir), weights_only=False, map_location="cpu")
    name = k["data"].split()[0]
    d = load(os.path.join(os.path.dirname(os.path.abspath(run_dir)), name))
    assert list(k["vocab"]) == d["vocab"], "paketin sozlugu %s'ninkiyle tutmuyor" % name
    return d, tokenizer_from(d["tokenizer"])


def point_relation_only(k):
    """PointRelation'a ozgu araclar (--analysis, --debug, --history, --ablation, --reroute, --hops) icin paketten
    model; baska mimaride acik hata."""
    from model_19 import PointRelation
    if k.get("arch", PointRelation.arch) != PointRelation.arch:
        raise SystemExit("bu arac PointRelation'a ozgu, paket %s -- --sinav ya da --classes kullan" % k.get("arch"))
    return PointRelation.from_package(k)


def latest_package(run_dir, step=None):
    """Kosu klasorundeki EN YENI agirlik (w<N>.pt ya da t<N>.pt); step verilirse o adimin paketi."""
    found = sorted((int(re.findall("[0-9]+", f)[0]), f) for f in os.listdir(run_dir) if re.match("[tw][0-9]+[.]pt$", f))
    if step is not None:
        found = [x for x in found if x[0] == step]
    assert found, "paket YOK -- %s%s" % (run_dir, "" if step is None else " adim %d" % step)
    return os.path.join(run_dir, found[-1][1])


def exam_cli(run_dir, step=None, per_split=6):
    """python data_tr_19.py --sinav <kosu klasoru> [--adim N] [--n 6]: CPU'da sorar, <kosu>/sinav/<paket>.txt'ye yazar.
    Iki mimari de (paketteki arch)."""
    from looped_19 import model_from_package
    path = latest_package(run_dir, step)
    k = torch.load(path, weights_only=False, map_location="cpu")
    m = model_from_package(k)
    d, tok = run_data(run_dir, k)
    t0 = time.time()
    lines = ["%s   %s   adim %s   bolme basina %d soru, tohum 7, CPU" % (
        os.path.basename(os.path.abspath(run_dir)), os.path.basename(path), f"{k['step']:,}", per_split)]
    lines += exam_text(m, tok, d, per_split)
    lines.append("(%.0f sn)" % (time.time() - t0))
    out = os.path.join(run_dir, "sinav", os.path.splitext(os.path.basename(path))[0] + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> " + out)


if __name__ == "__main__":
    arg = lambda key, default=None: sys.argv[sys.argv.index(key) + 1] if key in sys.argv else default
    if "--yaz" in sys.argv:
        write(arg("--yaz"), split_genitive="--split-genitive" in sys.argv, morph="--morph" in sys.argv,
              unit="document" if "--document" in sys.argv else "part")
    elif "--sinav" in sys.argv:
        exam_cli(arg("--sinav"), None if arg("--adim") is None else int(arg("--adim")), int(arg("--n", 6)))
    elif "--hops" in sys.argv:
        hops_cli(arg("--hops"), None if arg("--adim") is None else int(arg("--adim")), int(arg("--n", 6)))
    elif "--ablation" in sys.argv:
        ablation_cli(arg("--ablation"), None if arg("--adim") is None else int(arg("--adim")), int(arg("--n", 500)))
    elif "--history" in sys.argv:
        history_cli(arg("--history"), [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--soru"])
    elif "--debug" in sys.argv:
        debug_cli(arg("--debug"), [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--soru"],
                  None if arg("--adim") is None else int(arg("--adim")))
    elif "--reroute" in sys.argv:
        reroute_cli(arg("--reroute"), None if arg("--adim") is None else int(arg("--adim")), int(arg("--n", 60)))
    elif "--classes" in sys.argv:
        classes_cli(arg("--classes"), None if arg("--adim") is None else int(arg("--adim")))
    elif "--analysis" in sys.argv:
        analysis_cli(arg("--analysis"), None if arg("--adim") is None else int(arg("--adim")), int(arg("--n", 60)))
    else:
        t = build_text()
        print("izler  graf %s  parca %s  sinav %s   (model_16: %s %s %s)" % (
            t["graph"], t["parts_fp"], t["exam_fp"], REF_GRAPH, REF_PARTS, REF_EXAM))
