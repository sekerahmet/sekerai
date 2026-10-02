# -*- coding: utf-8 -*-
"""analyze_errors -- Model Y (FineWeb) dogruluk ve hata teshisi (analiz ajani D).  Modeli DEGISTIRMEZ.

    answers      olgu sorulari: cevap konumunda ilk 10 token, aday cevaplarin (dogru + modelin yanlislari) toplam log p'si,
                 dogru cevabin ilk token'inin sirasi / olasiligi; ayni olgu baska yoldan (ters yon, baglamli, az ornekli);
                 logit lens: dogru ve secilen adayin ilk token'inin tur tur sirasi (cevap hangi turda kuruluyor / eziliyor)
    decoding     ayni istemler (8 sabit istem + 10 soru) farkli uretim ayarlariyla: acgozlu, tekrar cezasi (pencere 64 / 512),
                 tekrar eden 4'luyu yasaklama, ornekleme (s 1 / 0,8 p 0,9); dongu olculeri + metinler (gozle okunur)
    repetition   kendini besleme: bir cumle R kez tekrarlaninca k. tekrarda token olasiligi (Xu 2022 yontemi); final.json'daki
                 acgozlu dongulerde ve tekrarli cumlelerde tur x head attention'inin onceki gecise (kopya hedefi) dusen payi
    loop_heads   head basina ortalama ablasyonu: tekrarli cumlede kopya log p'si ve normal metinde nll ne kadar degisiyor
                 (BAGIMLILIK olcer, "o parca olmadan egitilseydi" degil); en secici head'ler kapaliyken acgozlu uretim
    corpus       egitim parcalarinda istem ifadelerinden sonra gelen token'lar ve yakin pencerede aday cevap sayilari:
                 verinin kendisi hangi cevabi destekliyor

    python analysis/analyze_errors.py <kosu klasoru> <olcum> --data <FineWeb koku> [--device cuda] [--weights last|ema|both]
Cikti: KUYRUK_SONUC ortam degiskenindeki klasor (yoksa <kosu>/internals/errors/), errors_<olcum>_<agirlik>_<zaman>.txt + .json.
Calisma klasoru deneme2/model_y (importlar oradan).
"""
import argparse
import json
import os
import re
import sys
import time

import torch                                           # once torch (Windows c10.dll), sonra digerleri
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for p in (os.path.join(SRC, "train_simplestories"), os.path.join(SRC, "train_fineweb"), SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import data_simplestories as DS  # noqa: E402
import internals_y as I  # noqa: E402
from model_y import AttentionCache  # noqa: E402

EOT = "<|endoftext|>"
PROMPTS = (                                            # exam_fineweb.PROMPTS (final.json'daki 8 istem)
    "The water cycle is",
    "In 1789,",
    "Photosynthesis is the process by which",
    "To calculate the area of a circle,",
    "The main causes of the First World War were",
    "The human heart has four chambers:",
    "One of the most important inventions of the Industrial Revolution was",
    "Climate change affects",
)
# exam_fineweb.QUESTIONS + aday cevaplar (dogrular basta, sonra modelin questions.json'daki yanlislari) + otomatik yardimci
# anahtar (yalniz tablo icin; puan GOZLE verilir)
QUESTIONS = (
    ("The capital of France is", (" Paris", " Lyon", " London"), 1, ("Paris",)),
    ("The largest planet in our solar system is",
     (" Jupiter", " Saturn", " the Earth", " Earth", " Venus", " Mercury", " Pluto", " the Sun"), 1, ("Jupiter",)),
    ("Water boils at a temperature of",
     (" 100 degrees Celsius", " 212 degrees Fahrenheit", " 100 degrees", " 98.6 degrees", " 38 degrees"), 3, ("100", "212")),
    ("Plants need sunlight, water and", (" carbon dioxide", " air", " nutrients", " soil", " oxygen", " minerals"), 2,
     ("carbon dioxide", "CO2")),
    ("The chemical symbol for gold is", (" Au", " Cu", " Ag", " Fe", " Pb", " gold"), 1, ("Au",)),
    ("Isaac Newton is famous for", (" his laws of motion", " his law of gravity", " his work on the laws of motion",
                                    " his theory of relativity", " the discovery of electricity"), 3, ("motion", "gravit")),
    ("World War II ended in the year", (" 1945", " 1944", " 1918", " 1939"), 1, ("1945",)),
    ("The Amazon rainforest is located in", (" South America", " Brazil", " the Brazilian", " Africa", " Asia"), 3,
     ("South America", "Brazil")),
    ("If a rectangle is 3 meters long and 4 meters wide, its area is",
     (" 12 square meters", " 12", " 7 square meters", " 3 x 4", " 14 square meters", " 24 square meters"), 2, (" 12",)),
    ("The seasons on Earth are caused by", (" the tilt of the Earth", " the tilt of Earth's axis", " the Earth's tilt",
                                            " the rotation of the Earth", " the distance from the Sun",
                                            " the Earth's distance from the sun"), 3, ("tilt",)),
)
# Ayni olgu baska yoldan: (grup, istem, adaylar, dogru aday sayisi).  Grup: reverse = ters yon / baska soru kalibi,
# context = cevabi baglamda vererek, fewshot = ayni kalipta iki dogru ornek, binding = gozle bulunan yanlis birlesimlerin olgusu
PROBES = (
    ("reverse", "The planet Jupiter is the", (" largest", " smallest", " second", " fifth"), 1),
    ("reverse", "Au is the chemical symbol for", (" gold", " silver", " copper", " aluminum"), 1),
    ("reverse", "The chemical symbol Au stands for", (" gold", " silver", " copper"), 1),
    ("reverse", "The chemical symbol for silver is", (" Ag", " Si", " Au", " S"), 1),
    ("reverse", "The chemical symbol for iron is", (" Fe", " Ir", " I"), 1),
    ("reverse", "The chemical symbol for sodium is", (" Na", " So", " S"), 1),
    ("reverse", "The chemical symbol for copper is", (" Cu", " Co", " C"), 1),
    ("reverse", "Photosynthesis uses sunlight, water and", (" carbon dioxide", " nutrients", " oxygen", " soil"), 1),
    ("reverse", "During photosynthesis, plants take in", (" carbon dioxide", " oxygen", " nutrients", " water"), 1),
    ("reverse", "Earth has seasons because its axis is", (" tilted", " rotating", " spinning", " vertical"), 1),
    ("reverse", "The tilt of the Earth's axis causes", (" the seasons", " day and night", " the tides"), 1),
    ("reverse", "3 x 4 =", (" 12", " 7", " 34", " 1"), 1),
    ("reverse", "Three times four equals", (" twelve", " seven", " 12", " 7"), 2),
    ("reverse", "The area of a rectangle is its length times its", (" width", " height", " length", " area"), 1),
    ("reverse", "The area of a circle is equal to pi times the", (" radius squared", " square of the radius", " diameter",
                                                                  " circumference"), 2),
    ("context", "Jupiter is the biggest of the eight planets. The largest planet in our solar system is",
     (" Jupiter", " Earth", " the Earth", " Saturn"), 1),
    ("context", "Gold (Au) is a precious metal. The chemical symbol for gold is", (" Au", " Cu", " Ag"), 1),
    ("context", "Q: What is the largest planet in our solar system?\nA:", (" Jupiter", " Earth", " The Earth", " Saturn"), 1),
    ("context", "Which planet is the largest: Mars, Earth or Jupiter? The answer is", (" Jupiter", " Earth", " Mars"), 1),
    ("fewshot", "The chemical symbol for silver is Ag. The chemical symbol for iron is Fe. The chemical symbol for gold is",
     (" Au", " Cu", " Ag", " Fe"), 1),
    ("fewshot", "The smallest planet in our solar system is Mercury. The planet closest to the Sun is Mercury. "
                "The largest planet in our solar system is", (" Jupiter", " Earth", " Mercury", " Saturn"), 1),
    ("fewshot", "2 x 3 = 6. 5 x 2 = 10. 3 x 4 =", (" 12", " 7", " 15", " 34"), 1),
    ("binding", "George Washington was born in", (" Virginia", " Westmoreland", " New York", " a log cabin", " Kentucky"), 2),
    ("binding", "Abraham Lincoln was born in", (" a log cabin", " Kentucky", " Virginia", " New York"), 2),
    ("binding", "The first president of the United States was", (" George Washington", " Thomas Jefferson",
                                                                 " Abraham Lincoln", " John Adams"), 1),
    ("binding", "The Declaration of Independence was written by", (" Thomas Jefferson", " George Washington",
                                                                   " James Madison", " Benjamin Franklin"), 1),
    ("binding", "The Constitution of the United States was written in", (" 1787", " 1789", " 1776", " 1791"), 1),
    ("binding", "The French Revolution began in", (" 1789", " 1799", " 1776", " 1815"), 1),
    ("binding", "The steam engine was invented by", (" Thomas Newcomen", " James Watt", " Thomas Edison",
                                                     " George Stephenson"), 2),
    ("binding", "Thomas Newcomen built his first steam engine in", (" 1712", " 1769", " 1793", " 1800"), 1),
    ("binding", "The city of Kano is located in", (" Nigeria", " northern Nigeria", " Ghana", " Kenya"), 2),
    ("binding", "The human heart has two atria and two", (" ventricles", " arteries", " valves", " chambers"), 1),
    ("binding", "Evaporation is the process by which liquid water changes into", (" water vapor", " a gas", " ice",
                                                                                  " liquid"), 2),
    ("binding", "Condensation is the process by which water vapor changes into", (" liquid water", " a liquid", " gas",
                                                                                  " ice"), 2),
    ("binding", "The radius of a circle is the distance from the center to the", (" edge", " circumference",
                                                                                  " center", " diameter"), 2),
)
# decoding: ayar adi -> (sicaklik, top-p, tekrar cezasi, ceza penceresi, yasakli tekrar n'lisi, tohum sayisi)
SETTINGS = (
    ("greedy", 0.0, 1.0, 1.0, 0, 0, 1),
    ("greedy_pen64", 0.0, 1.0, 1.2, 64, 0, 1),
    ("greedy_pen512", 0.0, 1.0, 1.2, 512, 0, 1),
    ("greedy_no4gram", 0.0, 1.0, 1.0, 0, 4, 1),
    ("t1.0", 1.0, 1.0, 1.0, 0, 0, 3),
    ("t0.8_p0.9", 0.8, 0.9, 1.0, 0, 0, 3),
    ("t0.8_p0.9_pen512", 0.8, 0.9, 1.2, 512, 0, 3),
    ("t0.8_p0.9_no4gram", 0.8, 0.9, 1.0, 0, 4, 3),
)
# corpus: istem ifadesi (bosluklu, cumle icindeki yazimiyla) -> sonraki token'lar; ve (capa, adaylar): capadan sonraki
# NEAR_WINDOW token icinde aday sayisi
PHRASES = (
    " largest planet in our solar system is", " largest planet in the solar system is", " largest planet is",
    " chemical symbol for gold is", " symbol for gold is", " seasons are caused by", " seasons on Earth are caused by",
    " sunlight, water and", " sunlight, water, and", " water boils at a temperature of", " capital of France is",
    " World War II ended in", " Amazon rainforest is located in", " Newton is famous for", " its area is",
    " 3 x 4 =", " George Washington was born in", " steam engine was invented by", " Constitution was written by",
    " heart has four chambers", " Thomas Newcomen", " Newcomen engine",
)
NEAR = (
    (" largest planet", (" Jupiter", " Earth", " Saturn", " Venus", " Pluto", " Mercury")),
    (" symbol for gold", (" Au", " Cu", " Ag")),
    (" seasons", (" tilt", " tilted", " rotation", " distance")),
    (" sunlight, water", (" carbon dioxide", " nutrients", " air", " soil", " oxygen")),
    (" Washington was born", (" Virginia", " New York", " Westmoreland", " log cabin")),
    (" steam engine", (" Newcomen", " Watt", " 1712", " 1769", " 1793")),
)
NEAR_WINDOW = 32
REPEATS = 10           # repetition: kendini besleme egrisi icin cumle tekrar sayisi
COPY_REPEATS = 4       # loop_heads ve attention: tekrar sayisi
SENTENCES = 64         # tekrarlanan dogal cumle sayisi (valid belgelerinden, 12-40 token)


def _say(s):
    print(s, flush=True)


def _rnd(v, k=4):
    return None if v is None else round(float(v), k)


# ---- kurulum

def load_vocab(fw_root):
    tok, vocab = DS._load(os.path.join(fw_root, "gpt2", "tokenizer.json"), EOT)
    return vocab, vocab.index(DS.EOS_TOKEN)


def encode(text, vocab):
    return DS.encode(text, vocab)


def decode(ids, vocab):
    return DS.decode(list(ids), vocab)


def load_model(run_dir, weights, device, step=None):
    m = I._load_model(run_dir, weights, step)
    return m.to(device)


# ---- answers

@torch.no_grad()
def _seq_logprob(model, prefix, cont):
    """log p(cont | prefix) token token -> (toplam, [token log p'leri])."""
    dev = next(model.parameters()).device
    x = torch.tensor([prefix + cont[:-1]], device=dev)
    lp = torch.log_softmax(model.logits(x)[0].float(), -1)
    ys = torch.tensor(cont, device=dev)
    per = lp[len(prefix) - 1:, :].gather(-1, ys[:, None])[:, 0]
    return float(per.sum()), [round(float(v), 4) for v in per]


@torch.no_grad()
def _answer_position(model, prefix, vocab, top=10):
    """Cevap konumunda (istemin son token'i) dagilim: ilk top token, olasilik."""
    dev = next(model.parameters()).device
    z = model.logits(torch.tensor([prefix], device=dev))[0, -1].float()
    p = torch.softmax(z, -1)
    v, i = p.topk(top)
    return p, [(decode([int(t)], vocab), round(float(q), 4)) for q, t in zip(v, i)]


@torch.no_grad()
def _lens_ranks(model, prefix, token_ids):
    """Logit lens: her durumda (h0, A1, F1, ...) cevap konumunda verilen token'larin sirasi (1 = en yuksek)."""
    dev = next(model.parameters()).device
    P = model.tokens.points()
    names, _ = I._state_names(model)
    states = I._states(model, torch.tensor([prefix], device=dev), I._plan(model), P)
    out = {}
    for name, s in zip(names, states):
        z = I._scores(model, s[:, -1:], P)[0, 0].float()
        out[name] = [int((z > z[t]).sum()) + 1 for t in token_ids]
    return out


def probe(model, vocab, eot, prompt, cands, n_right, lens=False):
    prefix = [eot] + encode(prompt, vocab)
    p, top = _answer_position(model, prefix, vocab)
    rows = []
    for j, c in enumerate(cands):
        ids = encode(c, vocab)
        total, per = _seq_logprob(model, prefix, ids)
        first = ids[0]
        rows.append(dict(cand=c, right=j < n_right, tokens=len(ids), logp=round(total, 4), per_token=per,
                         first_token=decode([first], vocab), first_p=_rnd(p[first], 5),
                         first_rank=int((p > p[first]).sum()) + 1))
    best = max(rows, key=lambda r: r["logp"])
    right = [r for r in rows if r["right"]]
    best_right = max(right, key=lambda r: r["logp"])
    out = dict(prompt=prompt, top10=top, cands=rows, best=best["cand"], best_is_right=best["right"],
               right_first_rank=min(r["first_rank"] for r in right),
               margin_logp=round(best_right["logp"] - max([r["logp"] for r in rows if not r["right"]] or [-1e9]), 4))
    if lens:
        wrong = max([r for r in rows if not r["right"]], key=lambda r: r["logp"])
        ids = [encode(best_right["cand"], vocab)[0], encode(wrong["cand"], vocab)[0]]
        out["lens"] = dict(tokens=[decode([t], vocab) for t in ids], ranks=_lens_ranks(model, prefix, ids))
    return out


def measure_answers(model, vocab, eot, args):
    qs = [probe(model, vocab, eot, q, c, n, lens=True) for q, c, n, _ in QUESTIONS]
    ps = [dict(probe(model, vocab, eot, q, c, n), group=g) for g, q, c, n in PROBES]
    return dict(questions=qs, probes=ps)


def text_answers(res):
    L = ["# CEVAP KONUMU: dogru cevabin ilk token'i ve adaylarin toplam log p'si (aday metni teacher forcing)",
         "istem | dogru ilk token sira | en iyi aday (log p) | dogru-yanlis log p farki | ilk 5 token"]
    for block, key in (("10 SORU", "questions"), ("AYNI OLGU BASKA YOLDAN", "probes")):
        L.append("")
        L.append("## " + block)
        for r in res[key]:
            g = ("[%s] " % r["group"]) if "group" in r else ""
            L.append("%s%r" % (g, r["prompt"][-70:]))
            L.append("    dogru ilk token sirasi %d | en iyi aday %r %s | fark %+.2f nat" % (
                r["right_first_rank"], r["best"], "DOGRU" if r["best_is_right"] else "YANLIS", r["margin_logp"]))
            L.append("    ilk 5: " + "  ".join("%r %.3f" % t for t in r["top10"][:5]))
            L.append("    adaylar: " + "  ".join("%s%r %.2f (ilk %r sira %d p %.4f)" % (
                "*" if c["right"] else "", c["cand"], c["logp"], c["first_token"], c["first_rank"], c["first_p"])
                for c in r["cands"]))
            if "lens" in r:
                lens = r["lens"]
                names = [n for n in lens["ranks"] if n.startswith("F") or n == "A1"]
                L.append("    logit lens sira %r / %r: " % tuple(lens["tokens"]) + "  ".join(
                    "%s %d/%d" % (n, lens["ranks"][n][0], lens["ranks"][n][1]) for n in names))
    return L


# ---- uretim (batch'li, onbellekli)

def _loop_stats(prompt_ids, gen):
    """loop_first: daha once (istem dahil) gecmis 8'linin bittigi ilk uretim konumu; repeat8: o konumlarin payi;
    distinct4: farkli 4'lu orani (uretimde)."""
    full, seen, rep = list(prompt_ids) + list(gen), set(), []
    for j in range(len(full)):
        gram = tuple(full[j - 7:j + 1]) if j >= 7 else None
        if j >= len(prompt_ids):
            rep.append(gram in seen if gram else False)
        if gram:
            seen.add(gram)
    g4 = [tuple(gen[i:i + 4]) for i in range(len(gen) - 3)]
    return dict(loop_first=rep.index(True) if True in rep else None, repeat8=round(sum(rep) / max(len(rep), 1), 4),
                distinct4=round(len(set(g4)) / len(g4), 4) if g4 else 1.0)


@torch.no_grad()
def generate_batch(model, prompts, n, eot, temp=0.0, top_p=1.0, penalty=1.0, window=64, no_repeat=0, seed=0,
                   forbid_eot=True):
    """Satir basina istem (id listeleri, [eot] ile), n token; onbellekli (AttentionCache, sagdan dolgu).  eot yasak
    (long_write gibi) -- eot_first: eot'un en olasi oldugu ilk adim.  penalty: konus_fineweb.generate'in cezasi (son window
    token, pozitif skor bolunur, negatif carpilir).  no_repeat k: metinde (istem dahil) gecmis k'liyi tamamlayan token yasak.
    -> (uretimler, eot_first listesi, adim basina [p_top1, p_top2] (ceza oncesi, sicaklik 1))."""
    dev = next(model.parameters()).device
    B = len(prompts)
    lengths = torch.tensor([len(p) for p in prompts], device=dev)
    L = int(lengths.max())
    x = torch.full((B, L), eot, dtype=torch.long, device=dev)
    for r, p in enumerate(prompts):
        x[r, :len(p)] = torch.tensor(p, device=dev)
    caches = [AttentionCache(lengths, L + n) for _ in range(model.turns)]
    g = torch.Generator(device=dev).manual_seed(seed)
    hist = [list(p) for p in prompts]
    grams = [{} for _ in range(B)]
    if no_repeat:
        for r, h in enumerate(hist):
            for j in range(len(h) - no_repeat + 1):
                grams[r].setdefault(tuple(h[j:j + no_repeat - 1]), set()).add(h[j + no_repeat - 1])
    out, eot_first, probs = [[] for _ in range(B)], [None] * B, [[] for _ in range(B)]
    z = model.logits(x, caches).float()
    z = z[torch.arange(B, device=dev), lengths - 1]
    for i in range(n):
        p1 = torch.softmax(z, -1).topk(2)[0].cpu().numpy()
        am = z.argmax(-1).tolist()
        for r in range(B):
            probs[r].append([round(float(p1[r, 0]), 4), round(float(p1[r, 1]), 4)])
            if eot_first[r] is None and am[r] == eot:
                eot_first[r] = i
        if forbid_eot:
            z[:, eot] = -float("inf")
        if penalty != 1.0:
            recent = torch.full((B, window), eot, dtype=torch.long, device=dev)
            for r in range(B):
                tail = hist[r][-window:]
                recent[r, :len(tail)] = torch.tensor(tail, device=dev)
            hit = torch.zeros_like(z, dtype=torch.bool).scatter_(1, recent, True)
            if forbid_eot:
                hit[:, eot] = False
            z = torch.where(hit, torch.where(z > 0, z / penalty, z * penalty), z)
        if no_repeat:
            for r in range(B):
                ban = grams[r].get(tuple(hist[r][-(no_repeat - 1):]))
                if ban:
                    z[r, list(ban)] = -float("inf")
        if temp <= 0:
            t = z.argmax(-1)
        else:
            p = torch.softmax(z / temp, -1)
            if top_p < 1.0:
                ps, order = p.sort(-1, descending=True)
                keep = ps.cumsum(-1) - ps < top_p
                p = torch.zeros_like(p).scatter(-1, order, ps * keep)
            t = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=g)[:, 0]
        tl = t.tolist()
        for r in range(B):
            out[r].append(tl[r])
            hist[r].append(tl[r])
            if no_repeat:
                grams[r].setdefault(tuple(hist[r][-no_repeat:-1]), set()).add(tl[r])
        if i + 1 < n:
            z = model.logits(t[:, None], caches)[:, -1].float()
    return out, eot_first, probs


def measure_decoding(model, vocab, eot, args):
    texts = list(PROMPTS) + [q for q, _, _, _ in QUESTIONS]
    prompts = [[eot] + encode(s, vocab) for s in texts]
    n = args.tokens
    res = []
    for name, temp, top_p, pen, win, nrep, seeds in SETTINGS:
        t0 = time.time()
        rows = []
        for seed in range(seeds):
            gens, eot_first, _ = generate_batch(model, prompts, n, eot, temp, top_p, pen, win, nrep, seed)
            for k, (s, p, g) in enumerate(zip(texts, prompts, gens)):
                st = _loop_stats(p, g)
                head = decode(g[:48], vocab)
                hit = None
                if k >= len(PROMPTS):
                    keys = QUESTIONS[k - len(PROMPTS)][3]
                    hit = any(key in head for key in keys)
                rows.append(dict(prompt=s, seed=seed, kind="question" if k >= len(PROMPTS) else "prompt",
                                 eot_first=eot_first[k], key_in_first48=hit, text=decode(g, vocab), **st))
        q = [r for r in rows if r["kind"] == "question"]
        summary = dict(setting=name, temp=temp, top_p=top_p, penalty=pen, window=win, no_repeat=nrep, seeds=seeds,
                       loop_rate=round(sum(r["loop_first"] is not None for r in rows) / len(rows), 4),
                       loop_first_median=_median([r["loop_first"] for r in rows]),
                       repeat8=round(float(np.mean([r["repeat8"] for r in rows])), 4),
                       distinct4=round(float(np.mean([r["distinct4"] for r in rows])), 4),
                       key_hits=round(sum(bool(r["key_in_first48"]) for r in q) / len(q), 4), secs=round(time.time() - t0, 1))
        _say("decoding %-20s dongu %.2f  ilk dongu medyan %s  tekrar8 %.3f  farkli4 %.3f  anahtar %.2f  (%.0f sn)" % (
            name, summary["loop_rate"], summary["loop_first_median"], summary["repeat8"], summary["distinct4"],
            summary["key_hits"], summary["secs"]))
        res.append(dict(summary=summary, rows=rows))
    return dict(tokens=n, settings=res)


def _median(v):
    v = [x if x is not None else 10 ** 9 for x in v]   # dongusuz = sonsuz
    m = float(np.median(v))
    return None if m >= 10 ** 9 else m


def text_decoding(res):
    L = ["# URETIM AYARLARI (%d token, eot yasak; 8 istem + 10 soru; ornekleme 3 tohum)" % res["tokens"],
         "ayar | dongu orani | ilk dongu konumu medyan | tekrar8 | farkli4 | anahtar ilk 48 token'da (soru, OTOMATIK YARDIMCI)"]
    for s in res["settings"]:
        m = s["summary"]
        L.append("%-20s %.2f  %8s  %.3f  %.3f  %.2f" % (m["setting"], m["loop_rate"], m["loop_first_median"], m["repeat8"],
                                                       m["distinct4"], m["key_hits"]))
    for s in res["settings"]:
        L += ["", "## %s" % s["summary"]["setting"]]
        for r in s["rows"]:
            if r["seed"] > 0 and r["kind"] == "prompt":
                continue
            cut = 260 if r["kind"] == "question" else 600
            L.append("[%s t%d dongu@%s tekrar8 %.2f] %s ||%s" % (r["kind"][0], r["seed"], r["loop_first"], r["repeat8"],
                                                               r["prompt"], r["text"][:cut].replace("\n", " / ")))
    return L


# ---- tekrar: kendini besleme ve attention

def _sentences(fw_root, vocab, count, offset=0, lo=12, hi=40):
    """Valid belgelerinden dogal cumleler: belgenin ilk lo..hi token'lik cumlesi (nokta + bosluktan bolunmus)."""
    import data_fineweb as DF
    v = DF.load_valid(fw_root, log=lambda s: None)
    out = []
    for i in range(offset, len(v["valid_starts"])):
        text = decode(DF.valid_doc(v, i)[1:400], vocab)
        for sent in re.split(r"(?<=[.!?])\s+", text)[1:-1]:
            ids = encode(" " + sent.strip(), vocab)
            if lo <= len(ids) <= hi and "\n" not in sent:
                out.append(ids)
                break
        if len(out) >= count:
            break
    return out, v


@torch.no_grad()
def self_reinforcement(model, sents, eot, repeats):
    """[eot] + s x repeats; k. tekrarda (k = 0..repeats-1) token olasiligi (ilk token haric), argmax isabeti."""
    dev = next(model.parameters()).device
    rows = []
    for s in sents:
        x = torch.tensor([[eot] + s * repeats], device=dev)
        p = torch.softmax(model.logits(x[:, :-1])[0].float(), -1)
        y = x[0, 1:]
        pt = p.gather(-1, y[:, None])[:, 0]
        hit = p.argmax(-1) == y
        P = len(s)
        row = []
        for k in range(repeats):
            sl = slice(k * P + 1, (k + 1) * P)              # tekrarin ilk token'i haric (cumle siniri)
            row.append((float(pt[sl].mean()), float(hit[sl].float().mean())))
        rows.append(row)
    a = np.array(rows)                                      # (cumle, k, 2)
    return dict(p_mean=[round(float(v), 4) for v in a[:, :, 0].mean(0)],
                p_median=[round(float(v), 4) for v in np.median(a[:, :, 0], 0)],
                acc=[round(float(v), 4) for v in a[:, :, 1].mean(0)], sentences=len(rows))


def _copy_targets(seq, start, n=8):
    """Konum t (t >= start) icin, t'de biten n'li daha once j'de bittiyse kopya hedefi j + 1 (onceki gecisin devami)."""
    last, targets = {}, {}
    for t in range(len(seq)):
        if t >= n - 1:
            gram = tuple(seq[t - n + 1:t + 1])
            if gram in last and t >= start and last[gram] + 1 < t:
                targets[t] = last[gram] + 1
            last[gram] = t
    return targets


@torch.no_grad()
def copy_attention(model, seqs, starts):
    """seqs: id listeleri; her konumda (kopya hedefi olanlar) tur x head attention'inin hedefe (j) ve hedefin bir oncesine
    (j - 1: ayni token'in onceki gecisi) dusen payi; taban: o konumdaki tekduze pay 1 / (t + 1).  -> (tur, head) dizileri."""
    dev = next(model.parameters()).device
    T, H = model.turns, model.blocks[0].attention.heads
    sums = np.zeros((T, H, 2))
    base, count = 0.0, 0
    for seq, start in zip(seqs, starts):
        tg = _copy_targets(seq, start)
        if not tg:
            continue
        q = torch.tensor(sorted(tg), device=dev)
        j = torch.tensor([tg[int(t)] for t in q.tolist()], device=dev)
        x = torch.tensor([seq], device=dev)

        def on_att(t, a):                                   # a (1, H, L, L)
            rows = a[0][:, q]                               # (H, nq, L)
            sums[t, :, 0] += rows.gather(-1, j[None, :, None].expand(H, -1, 1))[..., 0].sum(-1).double().cpu().numpy()
            sums[t, :, 1] += rows.gather(-1, (j - 1)[None, :, None].expand(H, -1, 1))[..., 0].sum(-1).double().cpu().numpy()
        I._run(model, model.input_states(x), I._plan(model), dict(attention=on_att))
        base += float((1.0 / (q.double() + 1)).sum())
        count += len(q)
    m = sums / max(count, 1)
    return dict(target=m[..., 0].round(4).tolist(), previous=m[..., 1].round(4).tolist(), uniform=round(base / max(count, 1), 5),
                positions=count)


@torch.no_grad()
def loop_confidence(model, seqs, starts):
    """Uretilmis metinde (final.json acgozlu) konum basina p(top1), top1-top2 farki; dongu disi / ici, ici: o 8'linin
    kacinci gecisi (2, 3, 4, 5+)."""
    dev = next(model.parameters()).device
    buckets = {}
    for seq, start in zip(seqs, starts):
        x = torch.tensor([seq], device=dev)
        p = torch.softmax(model.logits(x)[0].float(), -1)
        top = p.topk(2, -1)[0].cpu().numpy()
        seen = {}
        for t in range(start - 1, len(seq) - 1):            # konum t, sonraki token t + 1 (uretilmis)
            gram = tuple(seq[max(0, t - 7):t + 1])
            k = seen.get(gram, 0)
            seen[gram] = k + 1
            key = "yeni" if k == 0 else ("gecis%d" % (k + 1) if k < 4 else "gecis5+")
            b = buckets.setdefault(key, [])
            b.append((top[t, 0], top[t, 0] - top[t, 1]))
    return {k: dict(n=len(v), p_top1=round(float(np.mean([a for a, _ in v])), 4),
                    margin=round(float(np.mean([b for _, b in v])), 4)) for k, v in sorted(buckets.items())}


def _final_generations(run_dir, eot, vocab, weights):
    """final.json'daki acgozlu uretimler (istem + ids): 8 istem ve 8 belge devami.  EMA icin weight_ema altindaki."""
    f = json.load(open(os.path.join(run_dir, "final.json"), encoding="utf-8"))
    src = f.get("weight_ema", {}) if weights == "ema" else f
    seqs, starts, names = [], [], []
    for key in ("prompts", "docs"):
        for r in src.get(key) or f.get(key) or []:
            pre = [eot] + encode(r["prompt"], vocab)
            seqs.append(pre + list(r["ids"]))
            starts.append(len(pre))
            names.append(r["prompt"][:40])
    return seqs, starts, names


def measure_repetition(model, vocab, eot, args, weights):
    sents, _ = _sentences(args.data, vocab, SENTENCES)
    rng = np.random.default_rng(0)
    rand = [rng.integers(1000, 50000, len(s)).tolist() for s in sents[:32]]
    out = dict(natural=self_reinforcement(model, sents, eot, REPEATS),
               random=self_reinforcement(model, rand, eot, REPEATS))
    seqs = [[eot] + s * COPY_REPEATS for s in sents]
    out["attention_repeated"] = copy_attention(model, seqs, [len(s) + 1 for s in sents])
    gseqs, gstarts, names = _final_generations(args.run, eot, vocab, weights)
    out["attention_greedy_loops"] = copy_attention(model, gseqs, gstarts)
    out["confidence_greedy"] = loop_confidence(model, gseqs, gstarts)
    out["greedy_texts"] = names
    return out


def _head_table(m, title, H):
    L = [title, "tur  " + " ".join("  h%d " % h for h in range(H)) + " | en buyuk"]
    for t, row in enumerate(m):
        L.append("%3d  " % (t + 1) + " ".join("%.3f" % v for v in row) + " | %.3f" % max(row))
    return L


def text_repetition(res):
    L = ["# KENDINI BESLEME (Xu 2022 yontemi): cumle k kez tekrarlaninca k. tekrarda token olasiligi (ilk token haric)",
         "k                 " + " ".join("%6d" % k for k in range(len(res["natural"]["p_mean"])))]
    for key in ("natural", "random"):
        r = res[key]
        L.append("%-8s p ort.  " % key + " ".join("%6.3f" % v for v in r["p_mean"]))
        L.append("%-8s p med.  " % key + " ".join("%6.3f" % v for v in r["p_median"]))
        L.append("%-8s isabet  " % key + " ".join("%6.3f" % v for v in r["acc"]))
    L.append("")
    L.append("# ACGOZLU URETIMDE (final.json) GUVEN: konum basina p(top1) ve top1-top2 farki, 8'linin kacinci gecisi")
    for k, v in res["confidence_greedy"].items():
        L.append("%-8s n %5d  p(top1) %.3f  fark %.3f" % (k, v["n"], v["p_top1"], v["margin"]))
    for key, title in (("attention_repeated", "TEKRARLI CUMLELER"), ("attention_greedy_loops", "ACGOZLU DONGULER (final.json)")):
        a = res[key]
        H = len(a["target"][0])
        L += ["", "# ATTENTION, %s: kopya hedefine (onceki gecisin devami) dusen pay; %d konum, tekduze pay %.4f" % (
            title, a["positions"], a["uniform"])]
        L += _head_table(a["target"], "## hedef j (siradaki token'i tasiyan konum)", H)
        L += _head_table(a["previous"], "## j - 1 (ayni token'in onceki gecisi)", H)
    return L


# ---- loop_heads: ortalama ablasyonu

@torch.no_grad()
def measure_loop_heads(model, vocab, eot, args):
    sents, v = _sentences(args.data, vocab, SENTENCES)
    import data_fineweb as DF
    dev = next(model.parameters()).device
    order = np.random.default_rng(0).permutation(len(v["valid_starts"]))
    normal = [DF.valid_doc(v, int(i))[:513] for i in order[:64] if len(DF.valid_doc(v, int(i))) > 64][:48]
    ref = [DF.valid_doc(v, int(i))[:513] for i in order[64:112]]     # head ortalamalari: normal'den ayri belgeler
    assert ref, "head ortalamasi icin valid belgesi yok"
    means = I._head_means(model, ref, False, 8)
    P = model.tokens.points()
    rep = [[eot] + s * COPY_REPEATS for s in sents]

    def batches(seqs, fns):
        """-> [(x, {ad: hedef agirligi (B, L - 1)}, tur girdileri + son durum)]; fns: ad -> f(dizi) (hedef basina 0/1)."""
        out = []
        for x, valid, idx in I._batches(seqs, 16, False):
            ws = {}
            for name, fn in fns.items():
                w = torch.zeros(valid.shape, dtype=torch.float32)
                for r, i in enumerate(idx):
                    w[r, :len(seqs[i]) - 1] = torch.tensor(fn(seqs[i]), dtype=torch.float32)
                ws[name] = w.to(dev)
            x = x.to(dev)
            hs = []
            h = I._run(model, model.input_states(x[:, :-1]), I._plan(model), dict(input=lambda t, v: hs.append(v)))
            out.append((x, ws, hs + [h]))                   # hs[tur] tur girdisi, hs[turns] son durum
        return out

    def part(s, copy):
        """hedef t = s[t + 1]: tekrar t // Pn; tekrarin ilk token'i (t % Pn == 0, cumle siniri) sayilmaz."""
        Pn = (len(s) - 1) // COPY_REPEATS
        return [1.0 if t % Pn and ((t // Pn >= 1) if copy else (t // Pn == 0)) else 0.0 for t in range(len(s) - 1)]

    data = batches(rep, dict(copy=lambda s: part(s, True), first=lambda s: part(s, False)))
    data += batches(normal, dict(normal=lambda s: [1.0] * (len(s) - 1)))

    def score(plan):
        res = dict(copy=[0.0, 0.0], first=[0.0, 0.0], normal=[0.0, 0.0])
        for x, ws, hs in data:
            h = I._run(model, hs[plan["start"]], plan, start=plan["start"])
            lp = torch.log_softmax(I._scores(model, h, P).float(), -1).gather(-1, x[:, 1:, None])[..., 0]
            for key, w in ws.items():
                res[key][0] += float((lp * w).sum())
                res[key][1] += float(w.sum())
        return {k: v[0] / v[1] for k, v in res.items()}

    def plan_for(heads=(), turns=()):
        case = {}
        if heads:
            case["heads"] = {k: means[k] for k in heads}
        if turns:
            case["skip_attention"] = list(turns)
        return I._plan(model, case)

    base = score(I._plan(model))
    _say("taban: kopya log p %.4f  ilk gecis %.4f  normal %.4f" % (base["copy"], base["first"], base["normal"]))
    T, H = model.turns, model.blocks[0].attention.heads
    rows = []
    for t in range(T):
        for h in range(H):
            r = score(plan_for(heads=[(t, h)]))
            rows.append(dict(case="H%d.%d" % (t + 1, h), turn=t, head=h, d_copy=round(base["copy"] - r["copy"], 4),
                             d_first=round(base["first"] - r["first"], 4), d_normal=round(base["normal"] - r["normal"], 4)))
        r = score(plan_for(turns=[t]))
        rows.append(dict(case="A%d" % (t + 1), turn=t, head=None, d_copy=round(base["copy"] - r["copy"], 4),
                         d_first=round(base["first"] - r["first"], 4), d_normal=round(base["normal"] - r["normal"], 4)))
        _say("tur %d bitti" % (t + 1))
    heads = [r for r in rows if r["head"] is not None]
    for r in heads:                                     # secicilik: kopya kaybi - normal metin kaybi
        r["selective"] = round(r["d_copy"] - r["d_normal"], 4)
    top = sorted(heads, key=lambda r: -r["selective"])
    picks = dict(top1=[top[0]], top3=top[:3], top8=top[:8])
    combos = {}
    for name, sel in picks.items():
        r = score(plan_for(heads=[(x["turn"], x["head"]) for x in sel]))
        combos[name] = dict(heads=[x["case"] for x in sel], d_copy=round(base["copy"] - r["copy"], 4),
                            d_first=round(base["first"] - r["first"], 4), d_normal=round(base["normal"] - r["normal"], 4))
    # kendini besleme egrisi ve acgozlu uretim, secici head'ler kapali
    gens = {}
    prompts = [[eot] + encode(s, vocab) for s in PROMPTS]
    for name in ("none", "top3", "top8"):
        plan = I._plan(model) if name == "none" else plan_for(heads=[(x["turn"], x["head"]) for x in picks[name]])
        g = _greedy_with_plan(model, prompts, args.ablate_tokens, eot, plan)
        gens[name] = [dict(prompt=s, text=decode(x, vocab), **_loop_stats(p, x)) for s, p, x in zip(PROMPTS, prompts, g)]
        _say("uretim %s: dongu %d/%d" % (name, sum(r["loop_first"] is not None for r in gens[name]), len(gens[name])))
    return dict(base=base, rows=rows, combos=combos, generations=gens, note=I._DEPENDENCE_NOTE,
                sentences=len(sents), normal_docs=len(normal))


@torch.no_grad()
def _greedy_with_plan(model, prompts, n, eot, plan):
    """Mudahaleli acgozlu uretim, onbelleksiz (her adim tam ileri hesap, I._run); eot yasak."""
    dev = next(model.parameters()).device
    P = model.tokens.points()
    seqs = [list(p) for p in prompts]
    for _ in range(n):
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), eot, dtype=torch.long, device=dev)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s, device=dev)
        h = I._run(model, model.input_states(x), plan)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        z = I._scores(model, h[torch.arange(len(seqs), device=dev), last][:, None], P)[:, 0].float()
        z[:, eot] = -float("inf")
        for r, t in enumerate(z.argmax(-1).tolist()):
            seqs[r].append(t)
    return [s[len(p):] for s, p in zip(seqs, prompts)]


def text_loop_heads(res):
    b = res["base"]
    L = ["# HEAD ORTALAMA ABLASYONU (%s)" % res["note"],
         "taban log p: kopya (tekrar 2-%d) %.4f | ilk gecis %.4f | normal metin %.4f  (%d cumle, %d belge)" % (
             COPY_REPEATS, b["copy"], b["first"], b["normal"], res["sentences"], res["normal_docs"]),
         "Δ = taban - mudahale (pozitif: o konumlarda log p dustu)", "",
         "## en secici 20 head (Δkopya - Δnormal)", "case      Δkopya  Δilk    Δnormal  secicilik"]
    heads = sorted([r for r in res["rows"] if r["head"] is not None], key=lambda r: -r["selective"])
    for r in heads[:20]:
        L.append("%-8s %7.4f %7.4f %8.4f %8.4f" % (r["case"], r["d_copy"], r["d_first"], r["d_normal"], r["selective"]))
    L += ["", "## tur attention'i kapali (A<t>)", "case      Δkopya  Δilk    Δnormal"]
    for r in [r for r in res["rows"] if r["head"] is None]:
        L.append("%-8s %7.4f %7.4f %8.4f" % (r["case"], r["d_copy"], r["d_first"], r["d_normal"]))
    L += ["", "## birlikte kapali"]
    for k, v in res["combos"].items():
        L.append("%-5s %s  Δkopya %.4f  Δilk %.4f  Δnormal %.4f" % (k, ",".join(v["heads"]), v["d_copy"], v["d_first"],
                                                                     v["d_normal"]))
    for k, rows in res["generations"].items():
        L += ["", "## acgozlu uretim, %s kapali" % k]
        for r in rows:
            L.append("[dongu@%s tekrar8 %.2f] %s ||%s" % (r["loop_first"], r["repeat8"], r["prompt"],
                                                         r["text"][:500].replace("\n", " / ")))
    return L


# ---- corpus

def measure_corpus(vocab, eot, args):
    import data_fineweb as DF
    d = os.path.join(args.data, DF.TAG)
    shards = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".bin"))
    phrases = {s: encode(s, vocab) for s in PHRASES}
    nexts = {s: {} for s in PHRASES}
    counts = {s: 0 for s in PHRASES}
    near = {a: dict(anchor=encode(a, vocab), cands={c: encode(c, vocab) for c in cs}, n=0, hits={c: 0 for c in cs})
            for a, cs in NEAR}
    total = 0
    for i in shards[:args.shards] if args.shards else shards:
        t0 = time.time()
        a = np.fromfile(os.path.join(d, "shard_%03d.bin" % i), dtype=np.uint16)
        total += len(a)
        for s, ids in phrases.items():
            pos = _find(a, ids)
            counts[s] += len(pos)
            for p in pos:
                key = tuple(int(t) for t in a[p + len(ids):p + len(ids) + 3])
                nexts[s][key] = nexts[s].get(key, 0) + 1
        for an, e in near.items():
            pos = _find(a, e["anchor"])
            e["n"] += len(pos)
            for c, cid in e["cands"].items():
                cp = _find(a, cid)
                if len(pos) and len(cp):                    # capadan sonra NEAR_WINDOW token icinde aday basliyor mu
                    k = np.searchsorted(cp, pos + len(e["anchor"]))
                    ok = (k < len(cp)) & (cp[np.minimum(k, len(cp) - 1)] < pos + len(e["anchor"]) + NEAR_WINDOW)
                    e["hits"][c] += int(ok.sum())
        _say("parca %03d: %.0f M token, %.0f sn" % (i, len(a) / 1e6, time.time() - t0))
        del a
    out = dict(tokens=total, phrases=[], near=[])
    for s in PHRASES:
        top = sorted(nexts[s].items(), key=lambda kv: -kv[1])[:15]
        out["phrases"].append(dict(phrase=s, count=counts[s], next=[(decode(k, vocab), c) for k, c in top]))
    for an, e in near.items():
        out["near"].append(dict(anchor=an, n=e["n"], hits=e["hits"]))
    return out


def _find(a, ids):
    """a icinde ids dizisinin basladigi konumlar (artan)."""
    ids = list(ids)
    if not ids:
        return np.array([], dtype=np.int64)
    cand = np.flatnonzero(a[:len(a) - len(ids) + 1] == ids[0])
    for k, t in enumerate(ids[1:], 1):
        if not len(cand):
            break
        cand = cand[a[cand + k] == t]
    return cand


def text_corpus(res):
    L = ["# EGITIM VERISI: %.2f milyar token" % (res["tokens"] / 1e9), "", "## ifadeden sonraki 3 token (en sik 15)"]
    for r in res["phrases"]:
        L.append("%r  n=%d" % (r["phrase"], r["count"]))
        L.append("    " + "  ".join("%r %d" % (k, c) for k, c in r["next"]))
    L += ["", "## capadan sonra %d token icinde aday" % NEAR_WINDOW]
    for r in res["near"]:
        L.append("%r n=%d: " % (r["anchor"], r["n"]) + "  ".join("%r %d" % kv for kv in r["hits"].items()))
    return L


# ---- CLI

def _out_dir(run_dir):
    d = os.environ.get("KUYRUK_SONUC") or os.path.join(run_dir, "internals", "errors")
    os.makedirs(d, exist_ok=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="kosu klasoru")
    ap.add_argument("measure", choices=("answers", "decoding", "repetition", "loop_heads", "corpus"))
    ap.add_argument("--data", required=True, help="FineWeb koku (gpt2/tokenizer.json, gpt2/shard_*.bin)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="last", choices=("last", "ema", "both"))
    ap.add_argument("--checkpoint", type=int, help="answers: checkpoint_t<adim>.pt (EMA yedekteki ortalama)")
    ap.add_argument("--tokens", type=int, default=641, help="decoding: uretim boyu (final.json gibi 641)")
    ap.add_argument("--ablate-tokens", type=int, default=200, help="loop_heads: mudahaleli acgozlu uretim boyu")
    ap.add_argument("--shards", type=int, default=0, help="corpus: ilk K parca (0 = hepsi)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    vocab, eot = load_vocab(args.data)
    out_dir = _out_dir(args.run)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    weights_list = ["none"] if args.measure == "corpus" else (["last", "ema"] if args.weights == "both" else [args.weights])
    for w in weights_list:
        t1 = time.time()
        if args.measure == "corpus":
            res, lines = measure_corpus(vocab, eot, args), None
            lines = text_corpus(res)
            source = "egitim parcalari"
        else:
            model = load_model(args.run, w, args.device, args.checkpoint)
            source = ("checkpoint_t%06d.pt" % args.checkpoint) if args.checkpoint else (
                "model_weight_ema.pt" if w == "ema" else "model.pt")
            if args.measure == "answers":
                res = measure_answers(model, vocab, eot, args)
                lines = text_answers(res)
            elif args.measure == "decoding":
                res = measure_decoding(model, vocab, eot, args)
                lines = text_decoding(res)
            elif args.measure == "repetition":
                res = measure_repetition(model, vocab, eot, args, w)
                lines = text_repetition(res)
            else:
                res = measure_loop_heads(model, vocab, eot, args)
                lines = text_loop_heads(res)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        header = ["analyze_errors %s | kosu %s | agirlik %s (%s) | %s | sure %.0f sn" % (
            args.measure, os.path.basename(os.path.normpath(args.run)), w, source, time.strftime("%Y-%m-%d %H:%M"),
            time.time() - t1)]
        name = "errors_%s_%s%s_%s" % (args.measure, w, "_t%06d" % args.checkpoint if args.checkpoint else "", stamp)
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(header + [""] + lines) + "\n")
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(dict(measure=args.measure, weights=w, source=source, result=res), f, ensure_ascii=False)
        print("\n".join(header + [""] + lines), flush=True)
        _say("yazildi: %s/%s.txt / .json" % (out_dir, name))
    _say("toplam %.0f sn" % (time.time() - t0))


if __name__ == "__main__":
    main()
