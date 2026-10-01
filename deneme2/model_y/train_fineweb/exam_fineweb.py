# -*- coding: utf-8 -*-
"""exam_fineweb -- FineWeb-Edu sinavi.  Model ve genel egitim iki ust klasorde (model_y.py, train_y.py); SimpleStories'in
genel olculeri (loop_check, alpha_summary, onbellekli devam) exam_simplestories'ten.

    exam                  valid belgeleri BELGE BELGE (data_fineweb.exam_windows; paketli, attention belge icinde): nll, ppl,
                          accuracy, bits_per_byte (eot haric / eot dahil), eos_ok, acc_ar / acc_other, belge ici konum
                          bantlari (bands: hedef sayisi, nll, accuracy, bits_per_byte), nll_by_frequency
    exam_long             SEQ_LEN'den uzun valid belgeleri tek basina, LONG_TOKENS'e kadar (bant 8k-16k); sonda bir kez
    texts                 sabit istemlerden ([eot] + istem) acgozlu devam, ilk eot'ta kesilir -- GOZLE okunur (kural 12)
    continuation_repeats  sinav belgelerinin ilk yarisindan acgozlu ve ornekleme devam: dongu, tekrar eden 8'li payi,
                          farkli 4'lu; gercek devam referans
    distant_copy          uzun baglam: bir parca uzaklik kadar sonra tekrar edilir; ikinci kopyada accuracy / nll, ilk
                          geciste (kopya yok) taban.  Kullanici, 30 Eylul: "Uzun bağlamı ölçen bir sınav yok. En kritiği bu.
                          bunu hallet." (ppl ve bant accuracy'si uzaktan getirememeyi gostermez, Men 2024)
SimpleStories'ten TASINMAYANLAR: 'X and X', kalip cumle, uydurma kelime (web metninde ad, adres, kod, sayi mesru; train
kelime listesi anlam tasimaz), soz tekrari (diyalog olcusu), exam_train (egitim 1 epoktan az: ezber okunmaz).
"""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train_simplestories"))
sys.path.insert(0, os.path.dirname(HERE))                           # model_y: model ve genel egitim

import data_fineweb as DF  # noqa: E402
import data_simplestories as DS  # noqa: E402
import exam_simplestories as ES  # noqa: E402
from train_y import generate  # noqa: E402

BANDS = ((0, 2048), (2048, 4096), (4096, 8192))    # tahminin yapildigi belge ici konum (gorulen anahtar - 1)
LONG_BANDS = BANDS + ((8192, 16384),)
LONG_TOKENS = 16384      # exam_long: belge basina en cok bu kadar girdi
LOGITS_BUDGET = 1 << 29  # sinav batch'i: satir x T x sozluk <= bu (fp32 2 GB); 8.192 x 50.257'de 1 satir
FREQUENCY_BANDS = (64, 512, 4096, 16384)           # nll_by_frequency: train siklik sirasi sinirlari (0 = en sik)
# Egitici web metnine uygun sabit istemler (ilk PROBE_PROMPTS her sinavda, hepsi sonda); her biri [eot] ile baslar (belge basi)
PROMPTS = (
    "The water cycle is",
    "In 1789,",
    "Photosynthesis is the process by which",
    "To calculate the area of a circle,",
    "The main causes of the First World War were",
    "The human heart has four chambers:",
    "One of the most important inventions of the Industrial Revolution was",
    "Climate change affects",
)
PROBE_PROMPTS = 4
# Dogru cevap sinavi: (istem, dogru cevabin ozu).  Puanlama OTOMATIK DEGIL, gozle (kullanici, 1 Ekim: "bence 10 tane yap ama
# otomatik kontrol değil sen bak"); son model ve EMA, acgozlu + 3 ornekleme, cevaplar kosu klasorunde questions.json
QUESTIONS = (
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Water boils at a temperature of", "100 C / 212 F"),
    ("Plants need sunlight, water and", "carbon dioxide"),
    ("The chemical symbol for gold is", "Au"),
    ("Isaac Newton is famous for", "laws of motion / gravity"),
    ("World War II ended in the year", "1945"),
    ("The Amazon rainforest is located in", "South America / Brazil"),
    ("If a rectangle is 3 meters long and 4 meters wide, its area is", "12 square meters"),
    ("The seasons on Earth are caused by", "the tilt of Earth's axis"),
)
REPEAT_DOCS = 64         # continuation_repeats: baglama sigan sinav belgelerinin ilk bu kadari (en az MIN_DOC token)
REPEAT_TOKENS = 256      # continuation_repeats: devam (ve karsilastirilan gercek devam) en cok bu kadar token; dongu olcusu
                         # icin yeter, 4.000'lik devam her sinavda cok yavas (A100 TEST, 30 Eylul)
MIN_DOC = 16             # istem + devam icin en kisa belge (token)
COPY_PASSAGES = 32       # distant_copy: sinav belgelerinin (COPY_TOKENS'ten uzun) ilk bu kadarinin basi
COPY_TOKENS = 64         # distant_copy: parca boyu (token)
COPY_FIRST = 256         # distant_copy: en kisa uzaklik; 2 katlarla en uzaga kadar
_CACHE = {}


def _frequency_band(data):
    """token -> train siklik bandi (0: en sik FREQUENCY_BANDS[0] token, ...); sayim train akisindan bir kez."""
    key = ("band", data["fingerprint"])
    if key not in _CACHE:
        a, V = data["train"], len(data["vocab"])
        counts = sum(np.bincount(a[c:c + (1 << 26)], minlength=V) for c in range(0, len(a), 1 << 26))
        rank = np.empty(V, dtype=np.int64)
        rank[np.argsort(-counts, kind="stable")] = np.arange(V)
        _CACHE[key] = np.searchsorted(np.array(FREQUENCY_BANDS), rank, side="right")
    return _CACHE[key]


def _token_bytes(data):
    key = ("bytes", data["fingerprint"])
    if key not in _CACHE:
        _CACHE[key] = DF.token_bytes(data["vocab"])
    return _CACHE[key]


def _repeated_bigram(target, prev, doc, valid, n):
    """Hedef, (onceki, hedef) ikilisini AYNI belgede daha once tamamlamis mi (Zoology'nin 'AR hit'i; belge siniri doc)."""
    order = torch.arange(target.numel(), device=target.device).view_as(target)
    keys = torch.where(valid, (doc * n + prev) * n + target, -1 - order).reshape(-1)
    idx = torch.sort(keys, stable=True).indices
    rep = torch.zeros_like(keys, dtype=torch.bool)
    rep[idx[1:]] = keys[idx[1:]] == keys[idx[:-1]]
    return rep.view_as(target)


class _Tally:
    """Batch'lerden toplanan sayimlar -> sinav sozlugu."""

    def __init__(self, data, bands, device):
        self.data, self.bands, self.V = data, bands, len(data["vocab"])
        self.edges = torch.tensor([b for b, _ in bands[1:]], device=device)
        self.band_of = torch.as_tensor(_frequency_band(data), device=device)
        self.tbytes = torch.as_tensor(_token_bytes(data), device=device)
        self.c = torch.zeros(5, len(bands), dtype=torch.float64, device=device)   # hedef, nll, isabet, metin nll, bayt
        self.freq = torch.zeros(3, len(FREQUENCY_BANDS) + 1, dtype=torch.float64, device=device)
        self.n = dict(all=[0, 0], eos=[0, 0], ar=[0, 0], other=[0, 0])
        self.nll, self.story_nll = 0.0, 0.0

    @torch.no_grad()
    def add(self, model, ids, mask, pos, doc):
        z = model.logits(ids[:, :-1], document_positions=pos[:, :-1])
        logits = z.to(torch.promote_types(z.dtype, torch.float32))   # en az fp32 (float64 model float64 kalir)
        target, valid = ids[:, 1:], mask[:, 1:]
        lse = logits.logsumexp(-1)
        ce = lse - logits.gather(-1, target[..., None])[..., 0]
        hit = logits.argmax(-1) == target
        eos = target == self.data["eot"]
        story = valid & ~eos
        self.nll += float(ce[valid].sum())
        self.story_nll += float(ce[story].sum())
        band = torch.bucketize(pos[:, :-1].contiguous(), self.edges, right=True)          # tahmin konumu: gorulen anahtar - 1
        inside = valid & (pos[:, :-1] < self.bands[-1][1])
        for k, sel in enumerate((inside, inside, inside & hit, inside & story, inside & story)):
            vals = (torch.ones_like(ce), ce, torch.ones_like(ce), ce, self.tbytes[target].double())[k]
            self.c[k].index_add_(0, band[sel], vals[sel].double())
        rep = _repeated_bigram(target, ids[:, :-1], doc[:, :-1], valid, self.V)
        for name, sel in dict(all=valid, eos=valid & eos, ar=valid & rep, other=valid & ~rep).items():
            self.n[name][0] += int((hit & sel).sum())
            self.n[name][1] += int(sel.sum())
        tb = self.band_of[target[valid]]
        self.freq[0].index_add_(0, tb, torch.ones_like(tb, dtype=torch.float64))
        self.freq[1].index_add_(0, tb, ce[valid].double())
        probs = logits.sub_(lse[..., None]).exp_().reshape(-1, self.V)   # yerinde: logits bundan sonra kullanilmaz
        self.freq[2].index_add_(0, self.band_of, (valid.reshape(1, -1).to(probs.dtype) @ probs)[0].double())

    def result(self, n_bytes, docs):
        rate = lambda c: c[0] / c[1] if c[1] else float("nan")
        n = self.n["all"][1]
        out = dict(n=n, nll=self.nll / max(n, 1), accuracy=rate(self.n["all"]), docs=docs, bytes=n_bytes)
        out.update(ppl=math.exp(out["nll"]), eos_ok=rate(self.n["eos"]), acc_ar=rate(self.n["ar"]),
                   acc_other=rate(self.n["other"]), bits_per_byte=DS.bits_per_byte(self.story_nll, n_bytes),
                   bits_per_byte_eos=DS.bits_per_byte(self.nll, n_bytes + docs))
        c = self.c.cpu().tolist()
        out["bands"] = [dict(range=list(b), targets=int(c[0][k]), nll=c[1][k] / c[0][k] if c[0][k] else None,
                             accuracy=c[2][k] / c[0][k] if c[0][k] else None,
                             bits_per_byte=c[3][k] / math.log(2) / c[4][k] if c[4][k] else None,
                             bytes=int(c[4][k])) for k, b in enumerate(self.bands)]
        V = self.V
        edges = [min(e, V) for e in (0,) + FREQUENCY_BANDS + (V,)]
        f = self.freq.cpu().tolist()
        out["nll_by_frequency"] = [dict(ranks=[edges[b], edges[b + 1]], targets=int(f[0][b]),
                                        nll=f[1][b] / f[0][b] if f[0][b] else None,
                                        mass_ratio=f[2][b] / f[0][b] if f[0][b] else None) for b in range(len(f[0]))]
        return out


def exam(model, data, docs=None, batch_size=None):
    """docs (valid siralari; None: data["exam"]) belge belge: exam_windows'un pencereleri paketli (document_positions)
    modelden gecer; her belgenin skoru tek basina hesabiyla ayni.  Hedef: belgenin metni + kapanis eot'u.
      nll, ppl, accuracy        butun hedefler; bits_per_byte: eot haric nll / ln 2 / belgelerin UTF-8 bayti
      bits_per_byte_eos         butun hedefler / ln 2 / (bayt + belge)
      bands                     tahmin konumu (belge ici, gorulen anahtar - 1) BANDS'te: hedef, nll, accuracy, bpb
                                (bayt: hedef token'larin bayti, data_fineweb.token_bytes)
      eos_ok, acc_ar/other      belge sonunda eot; hedef ayni belgede tamamlanmis bir ikiliyi tamamliyor mu
      nll_by_frequency          hedefin train siklik bandi: hedef, nll, mass_ratio (tahmin kutlesi / hedef; 1 kalibre)"""
    device = next(model.parameters()).device
    docs = data["exam"] if docs is None else np.asarray(docs)
    w = DF.exam_windows(data, docs)
    T = w["ids"].shape[1] - 1
    batch_size = batch_size or max(1, LOGITS_BUDGET // (len(data["vocab"]) * T))
    tally = _Tally(data, BANDS, device)
    for i in range(0, len(w["ids"]), batch_size):
        tally.add(model, *(torch.as_tensor(w[k][i:i + batch_size], device=device)
                           for k in ("ids", "mask", "document_positions", "doc")))
    return tally.result(int(data["valid_bytes"][docs].sum()), len(docs))


def exam_long(model, data, max_tokens=LONG_TOKENS):
    """SEQ_LEN'den uzun valid belgeleri tek basina (bir satir, konum 0'dan), ilk max_tokens girdiyle; LONG_BANDS.  Belge
    butun sigarsa kapanis eot'u da hedef.  Bellek yetmezse dict(error=...)."""
    device = next(model.parameters()).device
    s = data["valid_starts"]
    lengths = np.append(s[1:], len(data["valid"])) - s
    docs = np.flatnonzero(lengths > data["seq_len"])
    tally = _Tally(data, LONG_BANDS, device)
    n_bytes = 0
    try:
        for k in docs.tolist():
            unit = DF.valid_doc(data, k) + [data["eot"]]
            ids = torch.tensor([unit[:max_tokens + 1]], device=device)
            L = ids.shape[1]
            ar = torch.arange(L, device=device)[None]
            tally.add(model, ids, torch.ones_like(ids, dtype=torch.bool), ar, torch.zeros_like(ids))
            n_bytes += int(_token_bytes(data)[unit[1:L]].sum())
    except torch.cuda.OutOfMemoryError as e:
        return dict(error="bellek yetmedi: %s" % str(e)[:200], docs=len(docs))
    return dict(tally.result(n_bytes, len(docs)), max_tokens=max_tokens)


@torch.no_grad()
def texts(model, data, prompts, n=None, reals=None):
    """Acgozlu devam (train_y.generate, tek belge), ilk eot'ta kesilir.  prompts: [eot] ile baslayan id listeleri.  n None:
    satir basina gercek devamin boyu + 1 (kapanis eot'u; reals sart) -- metnin tamami okunur.
    -> [dict(prompt, model, ended[, real], ids)] okunur metin."""
    vocab, eot = data["vocab"], data["eot"]
    limits = [len(r) + 1 for r in reals] if n is None else [n] * len(prompts)
    if not prompts:
        return []
    out = []
    for i, (p, g) in enumerate(zip(prompts, generate(model, prompts, max(limits)))):
        g = g[:limits[i]]
        ended = eot in g
        g = g[:g.index(eot)] if ended else g
        row = dict(prompt=DS.decode(p[1:], vocab), model=DS.decode(g, vocab), ended=ended, ids=g)
        if reals is not None:
            row["real"] = DS.decode(reals[i], vocab)
        out.append(row)
    return out


def prompt_ids(data, prompts=PROMPTS):
    return [[data["eot"]] + DS.encode(p, data["vocab"]) for p in prompts]


def fitting_docs(data, docs, min_tokens=MIN_DOC):
    """docs'tan baglama sigan ([eot] + metin + kapanis <= SEQ_LEN + 1) ve en az min_tokens token'li belgeler, sirayla."""
    s = data["valid_starts"]
    lengths = np.append(s[1:], len(data["valid"])) - s
    return [int(k) for k in docs if min_tokens <= lengths[k] <= data["seq_len"]]


def doc_prompts(data, docs):
    """Belgelerin ilk yarisi ([eot] dahil) istem, kalani gercek devam (uretim o kadar: metnin tamami okunur)."""
    prompts, reals = [], []
    for k in docs:
        unit = DF.valid_doc(data, int(k))
        prompts.append(unit[:len(unit) // 2])
        reals.append(unit[len(unit) // 2:])
    return prompts, reals


def _repeats(prompts, gens, ended):
    """Devam basina: loop (kendi icinde tekrar eden 8'li), repeat8 (8'lisi istemde ya da devamda daha once gecmis token
    payi), distinct4 (farkli 4'lu orani)."""
    loops, rep, toks, f4 = 0, 0, 0, []
    for p, g in zip(prompts, gens):
        g8 = [tuple(g[i:i + 8]) for i in range(len(g) - 7)]
        loops += len(set(g8)) < len(g8)
        seen = {tuple(p[i:i + 8]) for i in range(len(p) - 7)}
        full = p + g
        for j in range(len(p), len(full)):
            if j >= 7:
                gram = tuple(full[j - 7:j + 1])
                rep += gram in seen
                seen.add(gram)
        toks += len(g)
        g4 = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
        f4.append(len(set(g4)) / len(g4) if g4 else 1.0)
    n = max(len(gens), 1)
    return dict(docs=len(gens), tokens=toks, ended=round(sum(ended) / n, 4), loop=round(loops / n, 4),
                repeat8=round(rep / max(toks, 1), 4), distinct4=round(sum(f4) / n, 4))


@torch.no_grad()
def continuation_repeats(model, data, count=REPEAT_DOCS, seed=0, real=False):
    """Baglama sigan sinav belgelerinin ilk count'u (fitting_docs): ilk yaridan (doc_prompts) acgozlu ve ornekleme
    (sicaklik 1, tohum seed) devam, belgenin kalani (en cok REPEAT_TOKENS) + 1 token ya da eot.  -> greedy / sampled [/ real]: _repeats."""
    prompts, reals = doc_prompts(data, fitting_docs(data, data["exam"])[:count])
    reals = [r[:REPEAT_TOKENS] for r in reals]
    out = {}
    if model is not None and prompts:
        limits = [len(r) + 1 for r in reals] * 2
        gens, ended = ES._continue(model, prompts + prompts, max(limits), [False] * len(prompts) + [True] * len(prompts),
                                   seed, data["eot"], len(data["vocab"]))
        gens = [g[:m] for g, m in zip(gens, limits)]
        ended = [e and len(g) < m for e, g, m in zip(ended, gens, limits)]
        k = len(prompts)
        out["greedy"] = _repeats(prompts, gens[:k], ended[:k])
        out["sampled"] = _repeats(prompts, gens[k:], ended[k:])
    if real:
        out["real"] = _repeats(prompts, reals, [False] * len(reals))
    return out


def copy_distances(max_distance, first=COPY_FIRST):
    """first, 2 first, 4 first, ... < max_distance, sonra max_distance."""
    out = []
    while first < max_distance:
        out.append(first)
        first *= 2
    return out + [max_distance]


@torch.no_grad()
def distant_copy(model, data, passages=COPY_PASSAGES, k=COPY_TOKENS, context=None, first=COPY_FIRST):
    """Satir = [eot] + P + dolgu + P, tek belge (konum 0'dan): P sinav belgesinin ilk k token'i (en cok baglamin 1/4'u),
    dolgu baska valid belgelerin metni; uzaklik = iki P'nin baslari arasi, copy_distances(context - k): satir context'e
    sigar (None: SEQ_LEN; 2 x SEQ_LEN egitilmemis uzaklik).  Uzaklik basina P'nin 2..k. token'lari: ikinci kopyada (bilgi uzakta) ve ilk geciste (taban, kopya yok)
    accuracy ve nll.  Kopya tabani gecmiyorsa model o uzakliga bakamiyor.  Bellek yetmezse kalan uzakliklar yazilmaz,
    error."""
    device = next(model.parameters()).device
    V, eot = len(data["vocab"]), data["eot"]
    k = min(k, data["seq_len"] // 4)
    max_distance = (context or data["seq_len"]) - k
    s = data["valid_starts"]
    lengths = np.append(s[1:], len(data["valid"])) - s
    chosen = [int(i) for i in data["exam"] if lengths[i] > k][:passages]
    P = torch.tensor([DF.valid_doc(data, i)[1:k + 1] for i in chosen], dtype=torch.long)
    pool, taken = [], set(chosen)                      # dolgu: parcalarin belgeleri disindaki valid metni, sirayla
    for i in range(len(s)):
        if len(pool) >= max_distance + 1024:
            break
        if i not in taken:
            pool += DF.valid_doc(data, i)[1:]
    pool = torch.tensor(pool, dtype=torch.long)
    pool = pool.repeat(-(-(max_distance + 1024) // len(pool)))    # kucuk veride (testler) dolgu kendini tekrar eder
    out = dict(passages=len(chosen), tokens=k, rows=[])
    j = torch.arange(1, k)                              # P'nin hedefleri 1..k-1: ilk geciste konum j, kopyada d + j
    for d in copy_distances(max_distance, first):
        g = d - k
        fill = torch.stack([pool[o:o + g] for o in ((r * 997) % (len(pool) - g + 1) for r in range(len(P)))])
        ids = torch.cat([torch.full((len(P), 1), eot), P, fill, P], 1).to(device)      # (n, d + k + 1)
        L = ids.shape[1] - 1
        pos = torch.arange(L, device=device).expand(len(P), L)
        at = torch.cat([j, d + j]).to(device)
        sums = torch.zeros(4, dtype=torch.float64)      # ilk nll, ilk isabet, kopya nll, kopya isabet
        step = max(1, LOGITS_BUDGET // (V * L))
        try:
            for b in range(0, len(P), step):
                z = model.logits(ids[b:b + step, :-1], document_positions=pos[b:b + step])[:, at]
                z = z.to(torch.promote_types(z.dtype, torch.float32))
                y = ids[b:b + step, 1:][:, at]
                ce = z.logsumexp(-1) - z.gather(-1, y[..., None])[..., 0]
                hit = (z.argmax(-1) == y).double()
                sums += torch.stack([ce[:, :k - 1].sum(), hit[:, :k - 1].sum(), ce[:, k - 1:].sum(),
                                     hit[:, k - 1:].sum()]).double().cpu()
        except torch.cuda.OutOfMemoryError as e:
            out["error"] = "uzaklik %d: bellek yetmedi: %s" % (d, str(e)[:200])
            break
        n = len(P) * (k - 1)
        out["rows"].append(dict(distance=d, targets=n, accuracy=float(sums[3]) / n, nll=float(sums[2]) / n,
                                first_accuracy=float(sums[1]) / n, first_nll=float(sums[0]) / n))
    return out
