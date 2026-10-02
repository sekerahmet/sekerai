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
                 verinin kendisi hangi cevabi destekliyor; "symbol for <ad> is" kalibinin cevap yuvasi
    answer_split 10 soru: cevap konumunda son turlar (F16, F15+F16, F16 alpha x 0,5) degisince dogru ve secilen ilk token;
                 acgozlu ilk token'dan sonraki dagilim; isin aramasi (beam 8 x 6 token).  Mudahale BAGIMLILIK olcer
    cooccur      metal adlari (Cu'nun doldurucu oldugu / olmadigi) cevresinde (+-COOC_WINDOW token, ayni belge) " copper",
                 " Cu", " iron", " Fe" gecme payi -- Cu doldurucusunun birlikte gecme adayi
    training_curve  checkpoint'ler boyunca (son + EMA): kendini besleme (dogal / rastgele, k = 0..5), acgozlu dongu (18 istem +
                 32 belge, 256 token), 10 soru cevap konumu, 312 olguda (analyze_capacity.freq_facts) acgozlu ilk token dogru mu ve
                 64 token'lik devamda dongu -- dongu egilimi ile bilgi birlikte mi degisiyor
    ss_profile   SimpleStories kosusu (son + varsa EMA): kendini besleme, acgozlu / ornekleme dongu (12 istem + 64 hikaye yarisi),
                 metinler; --data SimpleStories koku
    copy_odds    matematikci O7: kopya head'lerinde (tur.head, 1'den) kopya hedeflerine dusen kutlenin ln odds'u; m (onceki kopya
                 sayisi), s(n) = ln(0,99 (n - 1) / 0,01), Delta c (eslesen kosinus - otekilerin ortalamasi), sigma_o^2; saf dongu
                 ([eot] + cumle x 10) ve dongu oncesi (400 token yeni metin + cumle x 10)
    copy_calibration  matematikci O12: gercek valid metninde induction tahmini olan konumlarda p(induction token) kalibrasyonu,
                 baglam boyu 2 ve 4, m = 1 / 2 / 3+ ayri
    copy_ceiling DITTO-X Adim 0: g(l, m) = gercek metinde verbatim kopyanin bir sonraki token'da devam etme orani; l = en uzun
                 eslesen sonek (kova 2/4/8/16/32/64+), m = o sonekin belgede onceki gecis sayisi (kova 1..5+); --corpus-tokens > 0:
                 egitim parcalarindan (CPU), --model-docs > 0: valid belgelerinde ayni hucrelerde veri orani ve modelin p'si
    repeat_entry tekrara GIRIS: valid belgelerinde (ve --corpus-docs ile egitim parcalarinda) taze metin ilk kez tam cumle
                 tekrarina / l >= 8, 16 verbatim kopyaya hangi konumda giriyor, girince kalan metnin tekrar payi; ayni belgelerin
                 yarisindan gercek devam / acgozlu / s 1 / s 0,8 p 0,9 (256 token) ayni olcuyle; giris aninda modelin dagilimi
    beam_facts   matematikci O13: 312 olguda (analyze_capacity.freq_facts) isin (8 x 6) ve acgozlu ilk token / ilk ayrisan dogru
                 mu; isin dogru / acgozlu yanlis payi (kendi ciktisinda damitmanin bilgi kolu ust siniri)
    repeat_chain matematikci O14: tekrar cumlesinden hemen sonraki cumle de tekrar mi (g_S: herhangi bir tekrar / kopyalanan
                 parcanin devami); cumle sinirinda kopya token'i secilmis mi ve modelin p'si; veri (valid + egitim) ve acgozlu
    symbol_filler  86 element (analyze_capacity.ELEMENTS) x 5 kalip: dogru sembol ve " Cu" olasiligi / sirasi, top1
                 dagilimi (Cu kalibin genel doldurucusu mu); 133 baskentte top1 dagilimi

    python analysis/analyze_errors.py <kosu klasoru> <olcum> --data <FineWeb koku> [--device cuda] [--weights last|ema|both]
Cikti: KUYRUK_SONUC ortam degiskenindeki klasor (yoksa <kosu>/internals/errors/), errors_<olcum>_<agirlik>_<zaman>.txt + .json.
Calisma klasoru deneme2/model_y (importlar oradan).
"""
import argparse
import json
import math
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
    ("arith", "7 + 5 =", (" 12", " 75", " 13", " 2"), 1),
    ("arith", "3 + 4 =", (" 7", " 34", " 12", " 1"), 1),
    ("arith", "Seven plus five equals", (" twelve", " 12", " seven", " five"), 2),
    ("arith", "2 + 2 = 4. 3 + 5 = 8. 7 + 5 =", (" 12", " 13", " 75", " 8"), 1),
    ("arith", "2 + 2 = 4\n3 + 5 = 8\n7 + 5 =", (" 12", " 13", " 75", " 8"), 1),
    ("arith", "Two plus two equals four. Three plus five equals eight. Seven plus five equals",
     (" twelve", " thirteen", " eight", " seven"), 1),
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
# decoding: ayar adi -> (sicaklik, top-p, tekrar cezasi, ceza penceresi, yasakli tekrar n'lisi, tohum sayisi, DRY)
SETTINGS = (
    ("greedy", 0.0, 1.0, 1.0, 0, 0, 1, False),
    ("greedy_pen64", 0.0, 1.0, 1.2, 64, 0, 1, False),
    ("greedy_pen512", 0.0, 1.0, 1.2, 512, 0, 1, False),
    ("greedy_no4gram", 0.0, 1.0, 1.0, 0, 4, 1, False),
    ("greedy_dry", 0.0, 1.0, 1.0, 0, 0, 1, True),
    ("t1.0", 1.0, 1.0, 1.0, 0, 0, 3, False),
    ("t0.8_p0.9", 0.8, 0.9, 1.0, 0, 0, 3, False),
    ("t0.8_p0.9_pen512", 0.8, 0.9, 1.2, 512, 0, 3, False),
    ("t0.8_p0.9_no4gram", 0.8, 0.9, 1.0, 0, 4, 3, False),
    ("t0.8_p0.9_dry", 0.8, 0.9, 1.0, 0, 0, 3, True),
)
# DRY (dontrepeat2026_verbatimloops.txt, Ek A ve Tablo 8): z(v) -= lam * base^(n - allowed), n = v'yi uretmenin uzatacagi
# baglam sonekiyle eslesen onceki parcanin boyu (n >= allowed); ayiricilar (satir sonu, iki nokta, tirnak, yildiz) eslesmeyi keser
DRY = dict(lam=0.8, base=1.75, allowed=2, range=1024, cap=64)
# corpus: istem ifadesi (bosluklu, cumle icindeki yazimiyla) -> sonraki token'lar; ve (capa, adaylar): capadan sonraki
# NEAR_WINDOW token icinde aday sayisi
PHRASES = (
    " largest planet in our solar system is", " largest planet in the solar system is", " largest planet is",
    " chemical symbol for gold is", " symbol for gold is", " seasons are caused by", " seasons on Earth are caused by",
    " sunlight, water and", " sunlight, water, and", " water boils at a temperature of", " capital of France is",
    " World War II ended in", " Amazon rainforest is located in", " Newton is famous for", " its area is",
    " 3 x 4 =", " George Washington was born in", " steam engine was invented by", " Constitution was written by",
    " heart has four chambers", " Thomas Newcomen", " Newcomen engine",
    " 7 + 5 =", " 3 + 4 =", " 2 + 2 =", " plus five equals", " times four equals",
    " first president of the United States was", " capital of Australia is",
)
NEAR = (
    (" largest planet", (" Jupiter", " Earth", " Saturn", " Venus", " Pluto", " Mercury")),
    (" symbol for gold", (" Au", " Cu", " Ag")),
    (" seasons", (" tilt", " tilted", " rotation", " distance")),
    (" sunlight, water", (" carbon dioxide", " nutrients", " air", " soil", " oxygen")),
    (" Washington was born", (" Virginia", " New York", " Westmoreland", " log cabin")),
    (" steam engine", (" Newcomen", " Watt", " 1712", " 1769", " 1793")),
    (" capital of Australia", (" Canberra", " Sydney", " Melbourne")),
    (" first president", (" Washington", " Lincoln", " Jefferson", " Adams")),
    (" symbol Au", (" gold", " copper", " silver")),
)
NEAR_WINDOW = 32
# corpus: (capa, baglac) -> capadan sonra en cok SLOT_GAP token icinde baglac, ardindan gelen token sayilir (cevap yuvasi)
SLOTS = ((" chemical symbol for", " is"), (" symbol for", " is"), (" capital of", " is"))
SLOT_GAP = 6
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
def _dry_lengths(h, pos, breakers, cfg):
    """DRY n_t(v): h baglam (liste), pos {token: konumlar}; son token'in onceki her gecisi i icin geriye eslesme boyu m,
    aday v = h[i + 1].  -> {v: en uzun m}."""
    t = len(h) - 1
    last = h[t]
    if last in breakers:
        return {}
    lo = max(0, len(h) - cfg["range"])
    out = {}
    for i in pos.get(last, ()):
        if i >= t or i < lo:
            continue
        m = 1
        while m < cfg["cap"] and i - m >= lo and h[t - m] == h[i - m] and h[t - m] not in breakers:
            m += 1
        v = h[i + 1]
        if m > out.get(v, 0):
            out[v] = m
    return out


def generate_batch(model, prompts, n, eot, temp=0.0, top_p=1.0, penalty=1.0, window=64, no_repeat=0, seed=0,
                   forbid_eot=True, dry=None, breakers=frozenset()):
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
    pos = [{} for _ in range(B)]
    if dry:
        for r, h in enumerate(hist):
            for j, tok in enumerate(h):
                pos[r].setdefault(tok, []).append(j)
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
        if dry:
            for r in range(B):
                ns = _dry_lengths(hist[r], pos[r], breakers, dry)
                vs = [v for v, m in ns.items() if m >= dry["allowed"] and v not in breakers]
                if vs:
                    pen = torch.tensor([dry["lam"] * dry["base"] ** (ns[v] - dry["allowed"]) for v in vs], device=dev)
                    z[r, vs] = z[r, vs] - pen
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
            if dry:
                pos[r].setdefault(tl[r], []).append(len(hist[r]) - 1)
            if no_repeat:
                grams[r].setdefault(tuple(hist[r][-no_repeat:-1]), set()).add(tl[r])
        if i + 1 < n:
            z = model.logits(t[:, None], caches)[:, -1].float()
    return out, eot_first, probs


def _breakers(vocab):
    """DRY ayiricilari: metninde satir sonu, iki nokta, tirnak ya da yildiz olan token'lar."""
    return frozenset(i for i in range(len(vocab)) if any(c in decode([i], vocab) for c in '\n:"*'))


def _doc_prompts(fw_root, count, max_prompt=512):
    """Sinav belgelerinin (exam) ilk yarisi istem (en cok max_prompt token): continuation_repeats gibi."""
    import data_fineweb as DF
    v = DF.load_valid(fw_root, log=lambda s: None)
    out = []
    for k in v["exam"]:
        d = DF.valid_doc(v, int(k))
        if 32 <= len(d) // 2 <= max_prompt:
            out.append(d[:len(d) // 2])
        if len(out) >= count:
            break
    return out


def measure_decoding(model, vocab, eot, args):
    texts = list(PROMPTS) + [q for q, _, _, _ in QUESTIONS]
    prompts = [[eot] + encode(s, vocab) for s in texts]
    docs = _doc_prompts(args.data, args.docs) if args.docs else []
    n = args.tokens
    chosen = [x for x in SETTINGS if not args.settings or x[0] in args.settings.split(",")]
    breakers = _breakers(vocab) if any(x[7] for x in chosen) else frozenset()
    res, longs = [], []
    for name, temp, top_p, pen, win, nrep, seeds, dry in chosen:
        t0 = time.time()
        rows = []
        for seed in range(seeds):
            gens, eot_first, _ = generate_batch(model, prompts + docs, n, eot, temp, top_p, pen, win, nrep, seed,
                                                dry=DRY if dry else None, breakers=breakers)
            for k, (p, g) in enumerate(zip(prompts + docs, gens)):
                if k >= len(prompts):                       # belge devami: dongu olculeri ilk 256 token'da
                    g = g[:256]
                    rows.append(dict(prompt=decode(p[-30:], vocab), seed=seed, kind="doc", eot_first=eot_first[k],
                                     key_in_first48=None, text=decode(g, vocab)[:300], **_loop_stats(p, g)))
                    continue
                st = _loop_stats(p, g)
                head = decode(g[:48], vocab)
                hit = None
                if k >= len(PROMPTS):
                    keys = QUESTIONS[k - len(PROMPTS)][3]
                    hit = any(key in head for key in keys)
                rows.append(dict(prompt=texts[k], seed=seed, kind="question" if k >= len(PROMPTS) else "prompt",
                                 eot_first=eot_first[k], key_in_first48=hit, text=decode(g, vocab), **st))
        summary = dict(setting=name, temp=temp, top_p=top_p, penalty=pen, window=win, no_repeat=nrep, seeds=seeds, dry=dry,
                       secs=round(time.time() - t0, 1))
        for kind in ("prompt", "question", "doc"):
            rs = [r for r in rows if r["kind"] == kind]
            if rs:
                summary[kind] = dict(n=len(rs), loop_rate=round(sum(r["loop_first"] is not None for r in rs) / len(rs), 4),
                                     loop_first_median=_median([r["loop_first"] for r in rs]),
                                     repeat8=round(float(np.mean([r["repeat8"] for r in rs])), 4),
                                     distinct4=round(float(np.mean([r["distinct4"] for r in rs])), 4))
        q = [r for r in rows if r["kind"] == "question"]
        summary["key_hits"] = round(sum(bool(r["key_in_first48"]) for r in q) / len(q), 4)
        a = summary["prompt"]
        _say("decoding %-20s istem dongu %.2f (ilk %s)  belge dongu %s  anahtar %.2f  (%.0f sn)" % (
            name, a["loop_rate"], a["loop_first_median"], summary.get("doc", {}).get("loop_rate"), summary["key_hits"],
            summary["secs"]))
        res.append(dict(summary=summary, rows=rows))
        if args.long and (temp > 0 or dry):                 # long_write gibi: tek istem, baglam dolana kadar
            t0 = time.time()
            g, _, _ = generate_batch(model, [prompts[0]], args.long, eot, temp, top_p, pen, win, nrep, 0,
                                     dry=DRY if dry else None, breakers=breakers)
            g = g[0]
            st = _loop_stats(prompts[0], g)
            segs = []
            for s0 in range(0, len(g), 512):
                part = g[s0:s0 + 512]
                g4 = [tuple(part[i:i + 4]) for i in range(len(part) - 3)]
                segs.append(dict(start=s0, distinct4=round(len(set(g4)) / max(len(g4), 1), 4),
                                 repeat8=_loop_stats(prompts[0] + g[:s0], part)["repeat8"]))
            longs.append(dict(setting=name, tokens=len(g), segments=segs, text=decode(g, vocab)[:4000], **st))
            _say("long %-20s ilk dongu %s  tekrar8 %.3f  (%.0f sn)" % (name, st["loop_first"], st["repeat8"], time.time() - t0))
    return dict(tokens=n, docs=len(docs), settings=res, long=longs)


def _median(v):
    v = [x if x is not None else 10 ** 9 for x in v]   # dongusuz = sonsuz; medyan dongusuz yarida ise None
    m = float(np.median(v))
    return None if m >= 10 ** 8 else m


def text_decoding(res):
    L = ["# URETIM AYARLARI (%d token, eot yasak; 8 istem + 10 soru + %d belge devami (ilk 256 token); ornekleme 3 tohum)" % (
        res["tokens"], res.get("docs", 0)),
         "ayar | tur: dongu orani / ilk dongu medyan / tekrar8 / farkli4 | anahtar ilk 48 token'da (soru, OTOMATIK YARDIMCI)"]
    for s in res["settings"]:
        m = s["summary"]
        parts = ["%s %.2f/%s/%.3f/%.3f" % (k, m[k]["loop_rate"], m[k]["loop_first_median"], m[k]["repeat8"], m[k]["distinct4"])
                 for k in ("prompt", "question", "doc") if k in m]
        L.append("%-20s %s | %.2f" % (m["setting"], "  ".join(parts), m["key_hits"]))
    for r in res.get("long") or []:
        L += ["", "## uzun yazim %s: %d token, ilk dongu %s, tekrar8 %.3f" % (r["setting"], r["tokens"], r["loop_first"],
                                                                         r["repeat8"]),
              "dilim farkli4/tekrar8: " + "  ".join("%d:%.2f/%.2f" % (g["start"], g["distinct4"], g["repeat8"])
                                                    for g in r["segments"]),
              r["text"][:1500].replace("\n", " / ")]
    for s in res["settings"]:
        L += ["", "## %s" % s["summary"]["setting"]]
        for r in s["rows"]:
            if (r["seed"] > 0 and r["kind"] == "prompt") or r["kind"] == "doc":
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
    """[eot] + s x repeats; k. tekrarda (k = 0..repeats-1) token olasiligi (ilk token haric), argmax isabeti (Xu 2022 WR),
    olasiligi ilk gecistekinden yuksek token payi (Xu 2022 IP); cumleler ilk gecis olasiligina gore ucte bire bolunur
    (Xu 2022: baslangic olasiligi yuksek cumlede etki daha guclu)."""
    dev = next(model.parameters()).device
    rows = []
    for s in sents:
        x = torch.tensor([[eot] + s * repeats], device=dev)
        p = torch.softmax(model.logits(x[:, :-1])[0].float(), -1)
        y = x[0, 1:]
        pt = p.gather(-1, y[:, None])[:, 0]
        hit = p.argmax(-1) == y
        P = len(s)
        first = pt[1:P]
        row = []
        for k in range(repeats):
            sl = slice(k * P + 1, (k + 1) * P)              # tekrarin ilk token'i haric (cumle siniri)
            row.append((float(pt[sl].mean()), float(hit[sl].float().mean()), float((pt[sl] > first).float().mean())))
        rows.append(row)
    a = np.array(rows)                                      # (cumle, k, 3)

    def curves(b):
        return dict(p_mean=[round(float(v), 4) for v in b[:, :, 0].mean(0)],
                    p_median=[round(float(v), 4) for v in np.median(b[:, :, 0], 0)],
                    acc=[round(float(v), 4) for v in b[:, :, 1].mean(0)],
                    ip=[round(float(v), 4) for v in b[:, :, 2].mean(0)], sentences=len(b))
    out = curves(a)
    order = np.argsort(a[:, 0, 0])
    out["by_initial"] = [dict(initial=round(float(a[t, 0, 0].mean()), 4), **curves(a[t]))
                         for t in np.array_split(order, 3) if len(t)]
    return out


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
            bin_key = "konum%03d" % (64 * ((t + 1 - start) // 64))   # uretimdeki konum (64'luk): p(top1), tekrar payi
            buckets.setdefault(bin_key, []).append((top[t, 0], float(k > 0)))
    return {k: dict(n=len(v), p_top1=round(float(np.mean([a for a, _ in v])), 4),
                    margin=round(float(np.mean([b for _, b in v])), 4)) for k, v in sorted(buckets.items())}


@torch.no_grad()
def loop_lens(model, seqs, starts):
    """Acgozlu uretimde uretilen token'in her durumdaki (h0, A1, F1, ...) logit lens sirasi; konumlar: yeni (8'li ilk kez)
    ve tekrar (8'li daha once gecmis).  -> {kova: {durum: medyan sira, sira 1 payi}}."""
    dev = next(model.parameters()).device
    P = model.tokens.points()
    names, _ = I._state_names(model)
    ranks = {}
    for seq, start in zip(seqs, starts):
        x = torch.tensor([seq], device=dev)
        states = I._states(model, x[:, :-1], I._plan(model), P)
        y = x[0, 1:]
        seen, kind = set(), []
        for t in range(len(seq) - 1):
            gram = tuple(seq[max(0, t - 7):t + 1])
            kind.append("tekrar" if gram in seen else "yeni")
            seen.add(gram)
        pos = [t for t in range(start - 1, len(seq) - 1)]
        for name, s in zip(names, states):
            z = I._scores(model, s[0], P).float()                  # (L - 1, V)
            r = (z > z.gather(-1, y[:, None])).sum(-1) + 1
            r = r.cpu().numpy()
            for t in pos:
                ranks.setdefault(kind[t], {}).setdefault(name, []).append(int(r[t]))
    return {k: {n: dict(median=float(np.median(v)), top1=round(float(np.mean(np.array(v) == 1)), 4), n=len(v))
                for n, v in d.items()} for k, d in ranks.items()}


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
    out["lens_greedy"] = loop_lens(model, gseqs, gstarts)
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
        if "ip" in r:
            L.append("%-8s IP      " % key + " ".join("%6.3f" % v for v in r["ip"]))
        for b in r.get("by_initial", []):
            L.append("  ilk gecis p %.3f (%d cumle) p ort. " % (b["initial"], b["sentences"]) + " ".join(
                "%6.3f" % v for v in b["p_mean"]))
    L.append("")
    L.append("# ACGOZLU URETIMDE (final.json) GUVEN: konum basina p(top1) ve top1-top2 farki, 8'linin kacinci gecisi")
    for k, v in res["confidence_greedy"].items():
        L.append("%-9s n %5d  p(top1) %.3f  %s %.3f" % (k, v["n"], v["p_top1"], "tekrar payi" if k.startswith("konum")
                                                        else "fark", v["margin"]))
    L += ["", "# ACGOZLU URETIMDE LOGIT LENS: uretilen token'in her durumdaki medyan sirasi / sira 1 payi (yeni = 8'li ilk kez)"]
    lens = res["lens_greedy"]
    for k, d in lens.items():
        L.append("%-7s n %d: " % (k, next(iter(d.values()))["n"]) + "  ".join(
            "%s %g/%.2f" % (n, v["median"], v["top1"]) for n, v in d.items() if n == "h0" or n[1:] .isdigit()))
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

    def plan_for(heads=(), turns=(), facts=()):
        case = {}
        if facts:
            case["skip_facts"] = list(facts)
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
        if t > 0 or model.first_turn_facts:                # FactUnits alt adimi yok (tur 1'de takim yok)
            r = score(plan_for(facts=[t]))
            rows.append(dict(case="F%d" % (t + 1), turn=t, head=None, d_copy=round(base["copy"] - r["copy"], 4),
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
    T1 = model.turns - 1
    fact_cases = {"F%d" % (T1 + 1): [T1], "F%d+F%d" % (T1, T1 + 1): [T1 - 1, T1]}
    for name, fs in fact_cases.items():
        r = score(plan_for(facts=fs))
        combos[name] = dict(heads=[name], d_copy=round(base["copy"] - r["copy"], 4),
                            d_first=round(base["first"] - r["first"], 4), d_normal=round(base["normal"] - r["normal"], 4))
    for name in ("none", "top3", "top8") + tuple(fact_cases):
        plan = (I._plan(model) if name == "none" else plan_for(facts=fact_cases[name]) if name in fact_cases
                else plan_for(heads=[(x["turn"], x["head"]) for x in picks[name]]))
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
    L += ["", "## tur attention'i (A<t>) / FactUnits'i (F<t>) kapali", "case      Δkopya  Δilk    Δnormal"]
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
    slots = {an: dict(anchor=encode(an, vocab), conn=encode(cn, vocab), n=0, filled={}) for an, cn in SLOTS}
    total = 0
    for i in shards[:args.shards] if args.shards else shards:
        t0 = time.time()
        a = np.fromfile(os.path.join(d, "shard_%03d.bin" % i), dtype=np.uint16)
        total += len(a)
        for an, e in slots.items():                         # "<capa> <1..SLOT_GAP token> <baglac> <cevap>"
            pos = _find(a, e["anchor"]) + len(e["anchor"])
            e["n"] += len(pos)
            conn = e["conn"][0]
            done = np.zeros(len(pos), bool)
            for g in range(1, SLOT_GAP + 1):
                q = pos + g
                ok = ~done & (q + 1 < len(a))
                ok[ok] = a[q[ok]] == conn
                for t in a[q[ok] + 1].tolist():
                    e["filled"][t] = e["filled"].get(t, 0) + 1
                done |= ok
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
    out = dict(tokens=total, phrases=[], near=[], slots=[])
    for an, e in slots.items():
        top = sorted(e["filled"].items(), key=lambda kv: -kv[1])[:25]
        out["slots"].append(dict(anchor=an, n=e["n"], filled=sum(e["filled"].values()),
                                 top=[(decode([k], vocab), c) for k, c in top]))
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
    L += ["", "## cevap yuvasi: <capa> ... is <token> (capadan sonra en cok %d token icinde ' is')" % SLOT_GAP]
    for r in res.get("slots", []):
        L.append("%r n=%d, yuva dolu %d: " % (r["anchor"], r["n"], r["filled"]) + "  ".join("%r %d" % kv for kv in r["top"]))
    return L


# ---- answer_split: son turlar ve ilk token bolunmesi

def _variants(model):
    """Cevap konumunda karsilastirilan mudahaleler (tur 0'dan): son tur FactUnits yok / alpha yarim, son iki tur yok."""
    T1 = model.turns - 1
    return (("none", {}), ("F%d yok" % (T1 + 1), dict(skip_facts=[T1])),
            ("F%d+F%d yok" % (T1, T1 + 1), dict(skip_facts=[T1 - 1, T1])),
            ("aF%d x0,5" % (T1 + 1), dict(alpha_facts={T1: model.alpha_facts[T1].detach() * 0.5})))


@torch.no_grad()
def _beam(model, prefix, width, steps):
    """Isin aramasi: toplam log p'ye gore en iyi width devam, steps token.  -> [(token'lar, toplam log p)]."""
    dev = next(model.parameters()).device
    beams = [([], 0.0)]
    for _ in range(steps):
        x = torch.tensor([prefix + b for b, _ in beams], device=dev)
        lp = torch.log_softmax(model.logits(x)[:, -1].float(), -1)
        cand = []
        for (b, s0), row in zip(beams, lp):
            v, i = row.topk(width)
            cand += [(b + [int(t)], s0 + float(q)) for q, t in zip(v, i)]
        beams = sorted(cand, key=lambda c: -c[1])[:width]
    return beams


def measure_answer_split(model, vocab, eot, args):
    rows = []
    items = [(q, c[:n], k) for q, c, n, k in QUESTIONS] + [(q, c[:n], ("Jupiter",)) for g, q, c, n in PROBES
                                                           if "planet" in q and g in ("context", "fewshot")]
    dev = next(model.parameters()).device
    for prompt, rights, keys in items:
        prefix = [eot] + encode(prompt, vocab)
        right = encode(rights[0], vocab)[0]
        x = torch.tensor([prefix], device=dev)
        out = dict(prompt=prompt, right=decode([right], vocab), variants=[])
        greedy = None
        for name, case in _variants(model):
            z = (I._logits(model, x, case) if case else model.logits(x))[0, -1].float()
            p = torch.softmax(z, -1)
            g = int(p.argmax())
            greedy = g if greedy is None else greedy
            v, i = p.topk(5)
            out["variants"].append(dict(case=name, top5=[(decode([int(t)], vocab), round(float(q), 4)) for q, t in zip(v, i)],
                                        right_rank=int((p > p[right]).sum()) + 1, right_p=round(float(p[right]), 4),
                                        greedy_token=decode([greedy], vocab), greedy_p=round(float(p[greedy]), 4)))
        z2 = model.logits(torch.tensor([prefix + [greedy]], device=dev))[0, -1].float()
        p2 = torch.softmax(z2, -1)
        v, i = p2.topk(10)
        out["after_greedy"] = [(decode([int(t)], vocab), round(float(q), 4)) for q, t in zip(v, i)]
        beams = _beam(model, prefix, args.beam, args.beam_steps)
        out["beams"] = [(decode(b, vocab), round(s0, 3), any(k in decode(b, vocab) for k in keys)) for b, s0 in beams]
        rows.append(out)
        _say("%s: dogru %r sira %s" % (prompt[:40], out["right"], [r["right_rank"] for r in out["variants"]]))
    return dict(rows=rows, note=I._DEPENDENCE_NOTE, beam=args.beam, beam_steps=args.beam_steps)


def text_answer_split(res):
    L = ["# CEVAP KONUMU, SON TURLAR DEGISINCE (%s)" % res["note"],
         "istem | mudahale: dogru ilk token sirasi / p | acgozlu ilk token / p | ilk 3"]
    for r in res["rows"]:
        L.append("%r  (dogru ilk token %r)" % (r["prompt"][-70:], r["right"]))
        for v in r["variants"]:
            L.append("    %-12s dogru %4d / %.4f | secilen %r %.3f | %s" % (
                v["case"], v["right_rank"], v["right_p"], v["greedy_token"], v["greedy_p"],
                "  ".join("%r %.3f" % t for t in v["top5"][:3])))
        L.append("    acgozlu ilk token'dan sonra: " + "  ".join("%r %.3f" % t for t in r["after_greedy"]))
        L.append("    isin %d x %d: " % (res["beam"], res["beam_steps"]) + "  ".join(
            "%s%r %.2f" % ("*" if hit else "", b, s0) for b, s0, hit in r["beams"][:5]))
    return L


# ---- symbol_filler: "chemical symbol" kalibinin doldurucusu

SYMBOL_TEMPLATES = (
    "The chemical symbol for {name} is",
    "The symbol for {name} is",
    "{Name}'s chemical symbol is",
    "The element {name} has the chemical symbol",
    "The chemical symbol for silver is Ag. The chemical symbol for iron is Fe. The chemical symbol for {name} is",
)


@torch.no_grad()
def _answer_dists(model, prefixes, batch=64):
    """Istemlerin son konumundaki olasilik dagilimi (sagdan dolgu, nedensel) -> (N, V) CPU float."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(prefixes), batch):
        part = prefixes[i:i + batch]
        L = max(len(x) for x in part)
        x = torch.zeros(len(part), L, dtype=torch.long, device=dev)
        for r, q in enumerate(part):
            x[r, :len(q)] = torch.tensor(q, device=dev)
        z = model.logits(x).float()
        last = torch.tensor([len(q) - 1 for q in part], device=dev)
        out.append(torch.softmax(z[torch.arange(len(part), device=dev), last], -1).cpu())
    return torch.cat(out)


def measure_symbol_filler(model, vocab, eot, args):
    import analyze_capacity as AC
    cu = encode(" Cu", vocab)[0]
    res = dict(templates=[], capitals=None)
    for tpl in SYMBOL_TEMPLATES:
        rows = []
        prefixes = [[eot] + encode(tpl.format(name=n, Name=n[0].upper() + n[1:]), vocab) for n, _ in AC.ELEMENTS]
        P = _answer_dists(model, prefixes)
        for (name, sym), p in zip(AC.ELEMENTS, P):
            right = encode(" " + sym, vocab)[0]
            top = p.topk(3)
            rows.append(dict(name=name, symbol=sym, top3=[(decode([int(t)], vocab), round(float(q), 4))
                                                          for q, t in zip(top.values, top.indices)],
                             right_p=round(float(p[right]), 5), right_rank=int((p > p[right]).sum()) + 1,
                             cu_p=round(float(p[cu]), 5), cu_rank=int((p > p[cu]).sum()) + 1))
        other = [r for r in rows if r["symbol"] != "Cu"]
        tops = {}
        for r in rows:
            tops[r["top3"][0][0]] = tops.get(r["top3"][0][0], 0) + 1
        res["templates"].append(dict(
            template=tpl, n=len(rows), acc=round(sum(r["right_rank"] == 1 for r in rows) / len(rows), 4),
            top1_cu=sum(r["top3"][0][0] == " Cu" for r in other), n_other=len(other),
            cu_p_mean=round(float(np.mean([r["cu_p"] for r in other])), 4),
            cu_p_median=round(float(np.median([r["cu_p"] for r in other])), 4),
            cu_rank_median=float(np.median([r["cu_rank"] for r in other])),
            cu_beats_right=sum(r["cu_p"] > r["right_p"] for r in other),
            top1_counts=sorted(tops.items(), key=lambda kv: -kv[1])[:12], rows=rows))
        m = res["templates"][-1]
        _say("%-60s acc %.2f  top1=Cu %d/%d  Cu>dogru %d  p(Cu) ort %.3f" % (
            tpl[-60:], m["acc"], m["top1_cu"], m["n_other"], m["cu_beats_right"], m["cu_p_mean"]))
    P = _answer_dists(model, [[eot] + encode("The capital of %s is" % c[0], vocab) for c in AC.CAPITALS])
    tops, acc = {}, 0
    for c, p in zip(AC.CAPITALS, P):
        t = decode([int(p.argmax())], vocab)
        tops[t] = tops.get(t, 0) + 1
        acc += int(p.argmax()) == encode(" " + c[2], vocab)[0]
    res["capitals"] = dict(n=len(AC.CAPITALS), acc_first_token=round(acc / len(AC.CAPITALS), 4),
                           top1_counts=sorted(tops.items(), key=lambda kv: -kv[1])[:15])
    return res


def text_symbol_filler(res):
    L = ["# 'CHEMICAL SYMBOL' KALIBI: dogru sembolun ilk token'i ve ' Cu' (86 element, analyze_capacity.ELEMENTS)",
         "kalip | top1 dogru | top1 = Cu (bakir disi) | Cu > dogru | p(Cu) ort / medyan | Cu sirasi medyan"]
    for m in res["templates"]:
        L.append("%r" % m["template"])
        L.append("    %.3f | %d/%d | %d/%d | %.4f / %.4f | %g" % (
            m["acc"], m["top1_cu"], m["n_other"], m["cu_beats_right"], m["n_other"], m["cu_p_mean"], m["cu_p_median"],
            m["cu_rank_median"]))
        L.append("    top1 dagilimi: " + "  ".join("%r %d" % kv for kv in m["top1_counts"]))
    L += ["", "## element basina (ilk kalip): ad sembol | dogru sira / p | Cu sira / p | top3"]
    for r in res["templates"][0]["rows"]:
        L.append("%-12s %-3s | %5d / %.4f | %5d / %.4f | %s" % (
            r["name"], r["symbol"], r["right_rank"], r["right_p"], r["cu_rank"], r["cu_p"],
            "  ".join("%r %.3f" % t for t in r["top3"])))
    c = res["capitals"]
    L += ["", "## baskentler ('The capital of X is', %d): ilk token dogru %.3f; top1 dagilimi: " % (c["n"], c["acc_first_token"])
          + "  ".join("%r %d" % kv for kv in c["top1_counts"])]
    return L


# ---- cooccur: metal adi cevresinde aday token'lar

COOC_ANCHORS = (" gold", " silver", " tin", " platinum", " iron", " aluminum", " zinc", " nickel", " lead", " mercury")
COOC_TARGETS = (" copper", " Cu", " iron", " Fe", " silver", " Ag")
COOC_WINDOW = 32


def measure_cooccur(vocab, eot, args):
    """Capa (metal adi, tek token) gecislerinin kacinda +-COOC_WINDOW token icinde, ayni belgede hedef token var."""
    import data_fineweb as DF
    d = os.path.join(args.data, DF.TAG)
    shards = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".bin"))
    ids = {w: encode(w, vocab) for w in COOC_ANCHORS + COOC_TARGETS}
    assert all(len(v) == 1 for v in ids.values()), {w: v for w, v in ids.items() if len(v) != 1}
    tab = {a: dict(n=0, hits={t: 0 for t in COOC_TARGETS}) for a in COOC_ANCHORS}
    tot = {t: 0 for t in COOC_TARGETS}
    total = 0
    for i in shards[:args.shards] if args.shards else shards:
        t0 = time.time()
        a = np.fromfile(os.path.join(d, "shard_%03d.bin" % i), dtype=np.uint16)
        offsets = np.load(os.path.join(d, "shard_%03d_offsets.npy" % i))
        total += len(a)
        tpos = {t: np.flatnonzero(a == ids[t][0]) for t in COOC_TARGETS}
        for t in COOC_TARGETS:
            tot[t] += len(tpos[t])
        for an in COOC_ANCHORS:
            ap = np.flatnonzero(a == ids[an][0])
            tab[an]["n"] += len(ap)
            if not len(ap):
                continue
            adoc = np.searchsorted(offsets, ap, side="right") - 1
            for t in COOC_TARGETS:
                tp = tpos[t]
                if not len(tp):
                    continue
                lo = np.searchsorted(tp, ap - COOC_WINDOW)          # pencerenin ilk adayi
                j = np.minimum(lo, len(tp) - 1)
                ok = (lo < len(tp)) & (tp[j] <= ap + COOC_WINDOW) & (tp[j] != ap)
                ok &= np.searchsorted(offsets, tp[j], side="right") - 1 == adoc
                tab[an]["hits"][t] += int(ok.sum())
        _say("parca %03d: %.0f M token, %.0f sn" % (i, len(a) / 1e6, time.time() - t0))
        del a
    return dict(tokens=total, window=COOC_WINDOW, totals=tot,
                rows=[dict(anchor=an, n=e["n"], hits=e["hits"],
                           rate={t: round(e["hits"][t] / max(e["n"], 1), 5) for t in COOC_TARGETS}) for an, e in tab.items()])


def text_cooccur(res):
    L = ["# METAL ADI CEVRESINDE ADAYLAR (+-%d token, ayni belge; %.2f milyar token)" % (res["window"], res["tokens"] / 1e9),
         "hedeflerin toplam gecisi: " + "  ".join("%r %d" % kv for kv in res["totals"].items()),
         "capa | n | " + " | ".join("%r pay (sayi)" % t for t in COOC_TARGETS)]
    for r in res["rows"]:
        L.append("%-10r %9d | " % (r["anchor"], r["n"]) + " | ".join(
            "%.4f (%d)" % (r["rate"][t], r["hits"][t]) for t in COOC_TARGETS))
    return L


# ---- training_curve / ss_profile: dongu ve bilgi birlikte

CURVE_REPEATS = 6       # kendini besleme: tekrar sayisi (k = 0..5)
CURVE_TOKENS = 256      # acgozlu dongu olcusunun uretim boyu
CURVE_DOCS = 32         # sinav belgesi devami
FACT_TOKENS = 64        # olgu isteminden sonra acgozlu devam


def _doc_prompts_from(v, count, max_prompt=512):
    import data_fineweb as DF
    out = []
    for k in v["exam"]:
        d = DF.valid_doc(v, int(k))
        if 32 <= len(d) // 2 <= max_prompt:
            out.append(d[:len(d) // 2])
        if len(out) >= count:
            break
    return out


def _loop_summary(rows):
    """rows: _loop_stats sozlukleri -> dongu orani (± SE), ilk dongu medyani, tekrar8, farkli4."""
    n = len(rows)
    rate = sum(r["loop_first"] is not None for r in rows) / max(n, 1)
    return dict(n=n, loop_rate=round(rate, 4), loop_se=round(math.sqrt(rate * (1 - rate) / max(n, 1)), 4),
                loop_first_median=_median([r["loop_first"] for r in rows]),
                repeat8=round(float(np.mean([r["repeat8"] for r in rows])), 4),
                distinct4=round(float(np.mean([r["distinct4"] for r in rows])), 4))


def _facts_check(model, vocab, eot, facts):
    """Olgu istemi -> acgozlu ilk token dogru mu (dogru ya da kabul edilen yazimin ilk token'i) ve FACT_TOKENS'lik
    devamda dongu.  -> satirlar ve ozet (dogru / yanlis olgularda dongu orani)."""
    prompts = [[eot] + encode(f["prompt"], vocab) for f in facts]
    gens, _, _ = generate_batch(model, prompts, FACT_TOKENS, eot)
    rows = []
    for f, p, g in zip(facts, prompts, gens):
        rights = {encode(a, vocab)[0] for a in (f["answer"],) + tuple(f.get("also", ()))}
        rows.append(dict(kind=f["kind"], prompt=f["prompt"], correct=g[0] in rights, text=decode(g[:16], vocab),
                         **_loop_stats(p, g)))
    out = {}
    for kind in sorted({r["kind"] for r in rows}) + ["all"]:
        rs = [r for r in rows if kind in ("all", r["kind"])]
        ok, bad = [r for r in rs if r["correct"]], [r for r in rs if not r["correct"]]
        out[kind] = dict(n=len(rs), acc=round(len(ok) / len(rs), 4), loop_correct=_loop_summary(ok) if ok else None,
                         loop_wrong=_loop_summary(bad) if bad else None)
    return rows, out


def _curve_point(model, vocab, eot, sents, rand, prompts, texts, facts):
    sr = dict(natural=self_reinforcement(model, sents, eot, CURVE_REPEATS),
              random=self_reinforcement(model, rand, eot, CURVE_REPEATS))
    for k in ("natural", "random"):
        sr[k].pop("by_initial", None)
    gens, _, _ = generate_batch(model, prompts, CURVE_TOKENS, eot)
    rows = [dict(kind="question" if len(PROMPTS) <= i < len(PROMPTS) + len(QUESTIONS) else
                 ("prompt" if i < len(PROMPTS) else "doc"), **_loop_stats(p, g)) for i, (p, g) in enumerate(zip(prompts, gens))]
    loops = {k: _loop_summary([r for r in rows if k in ("all", r["kind"])]) for k in ("prompt", "question", "doc", "all")}
    answers = [probe(model, vocab, eot, q, c, n) for q, c, n, _ in QUESTIONS]
    q_texts = [decode(g[:24], vocab) for g in gens[len(PROMPTS):len(PROMPTS) + len(QUESTIONS)]]
    frows, fsum = _facts_check(model, vocab, eot, facts)
    return dict(self_reinforcement=sr, loops=loops,
                answers=[dict(prompt=a["prompt"], right_first_rank=a["right_first_rank"], best_is_right=a["best_is_right"],
                              margin=a["margin_logp"], greedy=t) for a, t in zip(answers, q_texts)],
                facts=fsum, fact_rows=[dict(prompt=r["prompt"], correct=r["correct"], loop_first=r["loop_first"])
                                       for r in frows],
                prompt_texts=[decode(g[:80], vocab) for g in gens[:len(PROMPTS)]])


def measure_training_curve(vocab, eot, args):
    import analyze_capacity as AC
    config = I._config(args.run)
    sents, v = _sentences(args.data, vocab, SENTENCES)
    rng = np.random.default_rng(0)
    rand = [rng.integers(1000, 50000, len(x)).tolist() for x in sents[:32]]
    prompts = [[eot] + encode(x, vocab) for x in list(PROMPTS) + [q for q, _, _, _ in QUESTIONS]]
    prompts += _doc_prompts_from(v, args.docs or CURVE_DOCS)
    facts = AC.freq_facts()
    packs = I._checkpoints(args.run)
    points = []
    for step in args.steps.split(","):
        t0 = time.time()
        model = I._build(config)
        if step == "final":
            pairs = [("last", "model.pt"), ("ema", "model_weight_ema.pt")]
            for w, f in pairs:
                model.load_state_dict(torch.load(os.path.join(args.run, f), map_location="cpu", weights_only=True))
                m = model.eval().requires_grad_(False).to(args.device)
                points.append(dict(step="final", weights=w, **_curve_point(m, vocab, eot, sents, rand, prompts, None, facts)))
                _say(_curve_line(points[-1]))
                model = m.cpu()
        else:
            pack = torch.load(packs[int(step)], map_location="cpu", weights_only=True)
            model.load_state_dict(pack["model"])
            m = model.eval().requires_grad_(False).to(args.device)
            points.append(dict(step=int(step), weights="last", **_curve_point(m, vocab, eot, sents, rand, prompts, None, facts)))
            _say(_curve_line(points[-1]))
            m = m.cpu().requires_grad_(True)                    # _load_weight_ema gradyanli parametreleri sayar
            I._load_weight_ema(m, pack["optimizer"], config)
            m = m.eval().requires_grad_(False).to(args.device)
            points.append(dict(step=int(step), weights="ema", **_curve_point(m, vocab, eot, sents, rand, prompts, None, facts)))
            _say(_curve_line(points[-1]))
            del pack
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        _say("adim %s: %.0f sn" % (step, time.time() - t0))
    return dict(points=points, repeats=CURVE_REPEATS, tokens=CURVE_TOKENS, sentences=len(sents), facts=len(facts))


def _curve_line(pt):
    sr, lp, f = pt["self_reinforcement"], pt["loops"], pt["facts"]["all"]
    return ("%-6s %-4s | SR dogal p k0..2 %.3f %.3f %.3f  rastgele k1 %.3f | dongu istem %.2f soru %.2f belge %.2f "
            "(ilk %s) | 10 soru dogru-en-iyi %d | olgu acc %.3f, dongu dogru %s / yanlis %s") % (
        pt["step"], pt["weights"], sr["natural"]["p_mean"][0], sr["natural"]["p_mean"][1], sr["natural"]["p_mean"][2],
        sr["random"]["p_mean"][1], lp["prompt"]["loop_rate"], lp["question"]["loop_rate"], lp["doc"]["loop_rate"],
        lp["all"]["loop_first_median"], sum(a["best_is_right"] for a in pt["answers"]), f["acc"],
        f["loop_correct"] and f["loop_correct"]["loop_rate"], f["loop_wrong"] and f["loop_wrong"]["loop_rate"])


def text_training_curve(res):
    L = ["# CHECKPOINT'LER BOYUNCA DONGU VE BILGI (son + EMA; EMA < 4000 baslangic agirligiyla kirli, B7)",
         "SR = kendini besleme, k. tekrarda token olasiligi (ilk token haric); dongu = 8'li tekrar, %d token acgozlu; "
         "olgu = %d olguda acgozlu ilk token dogru" % (res["tokens"], res["facts"])]
    L += [_curve_line(pt) for pt in res["points"]]
    L += ["", "## olgu turune gore (acc / dongu dogru / dongu yanlis)"]
    for pt in res["points"]:
        L.append("%-6s %-4s " % (pt["step"], pt["weights"]) + "  ".join(
            "%s %.3f/%s/%s" % (k, v["acc"], v["loop_correct"] and v["loop_correct"]["loop_rate"],
                               v["loop_wrong"] and v["loop_wrong"]["loop_rate"]) for k, v in pt["facts"].items()))
    L += ["", "## 10 soru: dogru ilk token sirasi (son agirlik)"]
    for pt in res["points"]:
        if pt["weights"] == "last":
            L.append("%-6s " % pt["step"] + " ".join("%5d" % a["right_first_rank"] for a in pt["answers"]))
    L += ["", "## istem metinleri (son agirlik, ilk 80 token)"]
    for pt in res["points"]:
        if pt["weights"] == "last":
            L.append("### adim %s" % pt["step"])
            L += ["  %s" % t[:300].replace("\n", " / ") for t in pt["prompt_texts"][:4]]
    return L


def _ss_sentences(stories, vocab, count, lo=8, hi=30):
    """SimpleStories hikayelerinden '.' token'iyla biten lo..hi token'lik cumleler (ilk cumle haric)."""
    dots = {i for i, t in enumerate(vocab) if t.strip(" \u0120") == "."}
    out = []
    for st in stories:
        ends = [j for j, t in enumerate(st) if t in dots]
        for a, b in zip(ends, ends[1:]):
            if lo <= b - a <= hi:
                out.append(st[a + 1:b + 1])
                break
        if len(out) >= count:
            break
    return out


def measure_ss_profile(args):
    import exam_simplestories as ES
    config = I._config(args.run)
    data = I._simplestories(config, args.data)
    vocab, eos = data["vocab"], data["eos"]
    _, stories = data["stories"](SENTENCES * 2 + 64, 0)
    sents = _ss_sentences(stories[:SENTENCES * 2], vocab, SENTENCES)
    rng = np.random.default_rng(0)
    rand = [rng.integers(10, len(vocab) - 1, len(x)).tolist() for x in sents[:32]]
    story_prompts = [st[:min(len(st) // 2, 200)] for st in stories[SENTENCES * 2:]][:64]
    prompts = [[eos] + data["encode"](x) for x in ES.PROMPTS] + story_prompts
    res = dict(config=dict(tag=config.get("tag"), steps=config.get("steps"), **{k: config.get("model_kw", {}).get(k) for k in (
        "d", "turns", "layers", "heads", "units", "output_link", "input_embedding", "input_embedding_sphere",
        "shared_facts", "first_turn_facts", "attention_bias")}), points=[])
    for w in ("last", "ema"):
        if not os.path.exists(os.path.join(args.run, "model_weight_ema.pt" if w == "ema" else "model.pt")):
            continue
        model = I._load_model(args.run, w).to(args.device)
        sr = dict(natural=self_reinforcement(model, sents, eos, CURVE_REPEATS),
                  random=self_reinforcement(model, rand, eos, CURVE_REPEATS))
        pt = dict(weights=w, self_reinforcement=sr, sentences=len(sents))
        for name, temp in (("greedy", 0.0), ("t1.0", 1.0)):
            gens, _, _ = generate_batch(model, prompts, CURVE_TOKENS, eos, temp=temp, top_p=1.0, seed=0)
            rows = [dict(kind="prompt" if i < len(ES.PROMPTS) else "story", **_loop_stats(p, g))
                    for i, (p, g) in enumerate(zip(prompts, gens))]
            pt[name] = {k: _loop_summary([r for r in rows if k in ("all", r["kind"])]) for k in ("prompt", "story", "all")}
            pt[name + "_texts"] = [DS.decode(g[:120], vocab) for g in gens[:4]]
        res["points"].append(pt)
        _say("%s %s | SR dogal k1 %.3f rastgele k1 %.3f | acgozlu dongu %.2f (ilk %s) | s1 dongu %.2f" % (
            os.path.basename(os.path.normpath(args.run)), w, sr["natural"]["p_mean"][1], sr["random"]["p_mean"][1],
            pt["greedy"]["all"]["loop_rate"], pt["greedy"]["all"]["loop_first_median"], pt["t1.0"]["all"]["loop_rate"]))
        del model
    return res


def text_ss_profile(res):
    L = ["# SIMPLESTORIES PROFILI  ayarlar: " + ", ".join("%s=%s" % kv for kv in res["config"].items())]
    for pt in res["points"]:
        sr = pt["self_reinforcement"]
        L += ["", "## %s (cumle %d)" % (pt["weights"], pt["sentences"]),
              "SR dogal p k0.. " + " ".join("%.3f" % v for v in sr["natural"]["p_mean"]) + "  IP " +
              " ".join("%.3f" % v for v in sr["natural"]["ip"]),
              "SR rastgele p   " + " ".join("%.3f" % v for v in sr["random"]["p_mean"])]
        for name in ("greedy", "t1.0"):
            L.append("%-6s " % name + "  ".join("%s n %d dongu %.3f ± %.3f ilk %s tekrar8 %.3f farkli4 %.3f" % (
                k, v["n"], v["loop_rate"], v["loop_se"], v["loop_first_median"], v["repeat8"], v["distinct4"])
                for k, v in pt[name].items()))
            L += ["    %s" % t[:400].replace("\n", " / ") for t in pt[name + "_texts"][:2]]
    return L


# ---- O7: kopya head odds'u

COPY_HEADS = ((5, 7), (7, 5), (8, 7), (10, 2))   # (tur, head), tur 1'den (D_003 kopya hedefi tablosu)
ODDS_REPEATS = 10
ODDS_PREFIX = 400


def _s_of_n(n):
    return math.log(0.99 * (max(n, 2) - 1) / 0.01)


@torch.no_grad()
def _copy_head_rows(model, seq, starts_period, heads):
    """seq: id listesi; starts_period: (tekrar bloğunun basi b, periyot P, tekrar sayisi R).  Konum t (tekrar k >= 1, ilk
    token haric): kopya hedefleri {t - r P + 1, r = 1..k}.  Head basina: kutle M, ln odds, cos ort. (kopya / oteki), oteki
    varyansi."""
    dev = next(model.parameters()).device
    x = torch.tensor([seq], device=dev)
    want = {t - 1 for t, _ in heads}
    inputs, atts = {}, {}
    taps = dict(canon=lambda t, v: inputs.__setitem__(t, v[1] + v[0]) if t in want else None,
                attention=lambda t, a: atts.__setitem__(t, a[0]) if t in want else None)
    I._run(model, model.input_states(x), I._plan(model), taps)
    blocks = model.turn_blocks()
    b, P, R = starts_period
    rows = []
    for (tt, hh) in heads:
        t0 = tt - 1
        at = blocks[t0].attention
        q, k = at.queries_keys(inputs[t0])                        # (1, H, T, dh); q log-n olcekli
        T = q.shape[-2]
        qs = at.query_scale(T, None, q.device).to(q.dtype) if at.attention_log_scale else torch.ones(T, 1, device=q.device)
        cos = (q[0, hh] @ k[0, hh].T) / qs                        # (T, T) ham kosinus
        a = atts[t0][hh]                                          # (T, T)
        for kk in range(1, R):
            for off in range(1, P):                               # tekrar k, tekrarin ilk token'i haric
                t = b + kk * P + off - 1                          # sorgu konumu (hedefi t + 1)
                tg = [t - r * P + 1 for r in range(1, kk + 1)]
                M = float(a[t, tg].sum())
                mask = torch.ones(t + 1, dtype=torch.bool, device=a.device)
                mask[tg] = False
                co = cos[t, :t + 1][mask]
                rows.append(dict(head="%d.%d" % (tt, hh), k=kk, m=kk, n=t + 1, s=_s_of_n(t + 1),
                                 ln_odds=math.log(max(M, 1e-9) / max(1 - M, 1e-9)), mass=M,
                                 dc=float(cos[t, tg].mean() - co.mean()), var_o=float(co.var())))
    return rows


def _fit(xs, ys):
    """En kucuk kareler egimi ve standart hatasi."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3 or x.std() == 0:
        return None, None
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    se = math.sqrt((resid @ resid) / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    return float(coef[0]), se


def measure_copy_odds(model, vocab, eot, args):
    heads = tuple(tuple(int(z) for z in h.split(".")) for h in args.heads.split(",")) if args.heads else COPY_HEADS
    sents, v = _sentences(args.data, vocab, 32)
    import data_fineweb as DF
    rng = np.random.default_rng(0)
    rand = [rng.integers(1000, 50000, len(x)).tolist() for x in sents[:16]]
    order = np.random.default_rng(1).permutation(len(v["valid_starts"]))
    prefixes = [DF.valid_doc(v, int(i))[:ODDS_PREFIX + 1] for i in order if len(DF.valid_doc(v, int(i))) > ODDS_PREFIX][:32]
    sets = dict(pure_natural=[([eot] + st * ODDS_REPEATS, (1, len(st), ODDS_REPEATS)) for st in sents],
                pure_random=[([eot] + st * ODDS_REPEATS, (1, len(st), ODDS_REPEATS)) for st in rand],
                preloop_natural=[(pf + st * ODDS_REPEATS, (len(pf), len(st), ODDS_REPEATS)) for pf, st in zip(prefixes, sents)])
    out = {}
    for name, items in sets.items():
        rows = []
        for seq, sp in items:
            rows += _copy_head_rows(model, seq, sp, heads)
        per = {}
        for tt, hh in heads:
            h = "%d.%d" % (tt, hh)
            rs = [r for r in rows if r["head"] == h]
            byk = []
            for kk in range(1, ODDS_REPEATS):
                rk = [r for r in rs if r["k"] == kk]
                byk.append(dict(k=kk, n=len(rk), ln_odds=round(float(np.mean([r["ln_odds"] for r in rk])), 3),
                                ln_odds_se=round(float(np.std([r["ln_odds"] for r in rk]) / math.sqrt(len(rk))), 3),
                                mass=round(float(np.mean([r["mass"] for r in rk])), 4),
                                s=round(float(np.mean([r["s"] for r in rk])), 3),
                                dc=round(float(np.mean([r["dc"] for r in rk])), 4),
                                var_o=round(float(np.mean([r["var_o"] for r in rk])), 5)))
            early = [r for r in rs if r["k"] <= 3]
            late = [r for r in rs if r["k"] >= 3]
            sl_m, se_m = _fit([math.log(r["m"]) for r in early], [r["ln_odds"] for r in early])
            sl_s, se_s = _fit([r["s"] for r in late], [r["ln_odds"] for r in late])
            pred = float(np.mean([r["dc"] - r["s"] * r["var_o"] for r in late])) if late else None
            per[h] = dict(by_k=byk, slope_ln_m_k1to3=sl_m, slope_ln_m_se=se_m, slope_s_k3plus=sl_s, slope_s_se=se_s,
                          predicted_slope_s=pred)
            _say("%s %s: d lnodds/d ln m (k<=3) %s  d lnodds/d s (k>=3) %s  tahmin Delta c - s sigma^2 %s" % (
                name, h, _rnd(sl_m, 3), _rnd(sl_s, 3), _rnd(pred, 3)))
        out[name] = per
    return dict(sets=out, heads=["%d.%d" % h for h in heads], repeats=ODDS_REPEATS, prefix=ODDS_PREFIX)


def text_copy_odds(res):
    L = ["# O7: KOPYA HEAD ODDS'U (tur.head 1'den; m = onceki kopya sayisi = k; s(n) = ln(0,99 (n-1)/0,01))",
         "pure_* = [eot] + cumle x %d (saf dongu); preloop_natural = %d token yeni metin + cumle x %d" % (
             res["repeats"], res["prefix"], res["repeats"])]
    for name, per in res["sets"].items():
        L += ["", "## %s" % name]
        for h, d in per.items():
            L.append("%s  egim ln m (k 1-3) %s ± %s | egim s(n) (k >= 3) %s ± %s | tahmin ort(Delta c - s sigma_o^2) %s" % (
                h, _rnd(d["slope_ln_m_k1to3"], 3), _rnd(d["slope_ln_m_se"], 3), _rnd(d["slope_s_k3plus"], 3),
                _rnd(d["slope_s_se"], 3), _rnd(d["predicted_slope_s"], 3)))
            L.append("    k: " + "  ".join("%d:%.2f±%.2f m%.3f s%.2f dc%.3f v%.4f" % (
                b["k"], b["ln_odds"], b["ln_odds_se"], b["mass"], b["s"], b["dc"], b["var_o"]) for b in d["by_k"]))
    return L


# ---- O12: m = 1 / 2 kopya kalibrasyonu

CAL_DOCS = 256
CAL_BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)


@torch.no_grad()
def measure_copy_calibration(model, vocab, eot, args):
    import data_fineweb as DF
    v = DF.load_valid(args.data, log=lambda s: None)
    docs = [DF.valid_doc(v, int(k))[:2049] for k in v["exam"][:CAL_DOCS]]
    dev = next(model.parameters()).device
    rows = {ctx: [] for ctx in (2, 4)}                     # (m, p_model(ind), hit, p_model(actual))
    for d in docs:
        x = torch.tensor([d[:-1]], device=dev)
        lp = torch.log_softmax(model.logits(x)[0].float(), -1)
        for ctx in (2, 4):
            seen, found = {}, []
            for t in range(ctx - 1, len(d) - 1):           # konum t, hedef d[t + 1]
                key = tuple(d[t - ctx + 1:t + 1])
                nxt = seen.get(key)
                if nxt and len(set(nxt)) == 1:             # onceki butun gecisler ayni devamda: induction tahmini
                    found.append((t, nxt[0], len(nxt), int(d[t + 1] == nxt[0])))
                seen.setdefault(key, []).append(d[t + 1])
            if found:
                ts = torch.tensor([f[0] for f in found], device=dev)
                ys = torch.tensor([f[1] for f in found], device=dev)
                pv = lp[ts, ys].exp().tolist()
                rows[ctx] += [(f[2], q, f[3]) for f, q in zip(found, pv)]
    out = {}
    for ctx, rs in rows.items():
        out[ctx] = {}
        for mname, sel in (("m1", lambda m: m == 1), ("m2", lambda m: m == 2), ("m3+", lambda m: m >= 3)):
            r = [x for x in rs if sel(x[0])]
            if not r:
                continue
            ps, hs = np.array([x[1] for x in r]), np.array([x[2] for x in r])
            bins = []
            ece = 0.0
            for lo, hi in zip(CAL_BINS, CAL_BINS[1:]):
                sel_b = (ps >= lo) & (ps < hi)
                if sel_b.sum():
                    bins.append(dict(lo=lo, hi=round(min(hi, 1.0), 2), n=int(sel_b.sum()), p=round(float(ps[sel_b].mean()), 4),
                                     hit=round(float(hs[sel_b].mean()), 4),
                                     se=round(float(math.sqrt(hs[sel_b].mean() * (1 - hs[sel_b].mean()) / sel_b.sum())), 4)))
                    ece += sel_b.sum() / len(ps) * abs(ps[sel_b].mean() - hs[sel_b].mean())
            out[ctx][mname] = dict(n=len(r), p_mean=round(float(ps.mean()), 4), hit=round(float(hs.mean()), 4),
                                   hit_se=round(float(math.sqrt(hs.mean() * (1 - hs.mean()) / len(r))), 4),
                                   ece=round(float(ece), 4), bins=bins)
            _say("baglam %d %s: n %d  p ort %.3f  isabet %.3f  ECE %.3f" % (ctx, mname, len(r), ps.mean(), hs.mean(), ece))
    return dict(docs=len(docs), by_context=out)


def text_copy_calibration(res):
    L = ["# O12: INDUCTION KALIBRASYONU (%d valid sinav belgesi, en cok 2048 token)" % res["docs"],
         "konum t: son <baglam> token daha once gectiyse ve onceki gecislerin hepsi ayni token'la surduyse induction tahmini; "
         "m = onceki gecis sayisi.  p = modelin o token'a verdigi olasilik, isabet = gercek sonraki token o mu"]
    for ctx, d in res["by_context"].items():
        for mname, r in d.items():
            L += ["", "## baglam %s, %s: n %d  p ort %.3f  isabet %.3f ± %.3f  ECE %.3f" % (
                ctx, mname, r["n"], r["p_mean"], r["hit"], r["hit_se"], r["ece"])]
            L += ["    p [%.2f, %.2f)  n %6d  p ort %.3f  isabet %.3f ± %.3f" % (b["lo"], b["hi"], b["n"], b["p"], b["hit"], b["se"])
                  for b in r["bins"]]
    return L


# ---- copy_ceiling: DITTO-X tavani g(l, m)

CEIL_LENGTHS = (2, 4, 8, 16, 32, 64)        # kova alt siniri; 64 = 64+
CEIL_M = 5                                  # m kovasi 1..4, 5 = 5+
CEIL_CHUNK = 50_000_000                     # parca (belge sinirinda) token
_HASH_P = np.uint64(0x9E3779B97F4A7C15)


def _copy_cells(a, starts):
    """a: belgeler arka arkaya (uint16 akis), starts: belge baslari (artan; belge [eot] + metin).  Konum t (hedef a[t + 1],
    ayni belgede): l = belgede daha once (bitisi < t) gecmis en uzun sonek, CEIL_LENGTHS kovasi (eot haric pencere); m = o
    kova boyundaki sonekin onceki gecis sayisi; tahmin = onceki gecislerin hepsi ayni token'la surduyse o token (O12 gibi),
    surmediyse tutarsiz.  -> dict(lb, m, consistent, hit, t, pred) (yalniz l >= 2 olan konumlar)."""
    n = len(a)
    doc = np.zeros(n, np.int64)
    doc[starts[1:]] = 1
    doc = np.cumsum(doc)
    pos = np.arange(n, dtype=np.int64) - starts[doc]           # belge ici konum (eot = 0)
    ends = np.append(starts[1:], n)
    has_next = np.arange(n) + 1 < ends[doc]
    nxt = np.zeros(n, np.int64)
    nxt[:-1] = a[1:]
    h = a.astype(np.uint64) + np.uint64(1)
    L = 1
    best = dict(lb=np.full(n, -1, np.int8), m=np.zeros(n, np.int64), cons=np.zeros(n, bool), pred=np.zeros(n, np.int64))
    docmix = (doc.astype(np.uint64) + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93)
    with np.errstate(over="ignore"):
        for k, target in enumerate(CEIL_LENGTHS):
            while L < target:                                   # h_{2L}[t] = h_L[t - L] * P^L + h_L[t]
                shifted = np.zeros(n, np.uint64)
                shifted[L:] = h[:-L]
                h = shifted * (_HASH_P ** np.uint64(L)) + h
                L *= 2
            valid = (pos - L + 1 >= 1) & has_next               # pencere eot'u icermez; hedef ayni belgede
            idx = np.flatnonzero(valid)
            key = h[idx] ^ docmix[idx]
            order = np.argsort(key, kind="stable")              # anahtar, sonra konum
            ks, ii = key[order], idx[order]
            first = np.ones(len(ks), bool)
            first[1:] = ks[1:] != ks[:-1]
            gstart = np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
            occ = np.arange(len(ks)) - gstart                   # onceki gecis sayisi
            fnext = nxt[ii[gstart]]
            diff = (nxt[ii] != fnext).astype(np.int64)
            cs = np.cumsum(diff)
            before = cs - diff - (cs[gstart] - diff[gstart])    # grup icinde onceki uyusmazlik sayisi
            has = occ >= 1
            t = ii[has]
            best["lb"][t] = k                                   # buyuk L sonra yazilir: en uzun kova kalir
            best["m"][t] = occ[has]
            best["cons"][t] = before[has] == 0
            best["pred"][t] = fnext[has]
    t = np.flatnonzero(best["lb"] >= 0)
    return dict(t=t, lb=best["lb"][t].astype(np.int64), m=np.minimum(best["m"][t], CEIL_M), consistent=best["cons"][t],
                pred=best["pred"][t], hit=(nxt[t] == best["pred"][t]))


def _cell_table(cells, p=None):
    """-> hucre listesi: l kovasi, m kovasi, n (tutarli), isabet orani ± SE, tutarsiz payi, (p: modelin ortalama p'si)."""
    rows = []
    for k, L in enumerate(CEIL_LENGTHS):
        for m in range(1, CEIL_M + 1):
            sel = (cells["lb"] == k) & (cells["m"] == m)
            ok = sel & cells["consistent"]
            n, n_all = int(ok.sum()), int(sel.sum())
            g = float(cells["hit"][ok].mean()) if n else None
            row = dict(l=L, m=m, n=n, n_all=n_all, inconsistent=round(1 - n / n_all, 4) if n_all else None,
                       g=None if g is None else round(g, 4), se=None if g is None else round(math.sqrt(g * (1 - g) / n), 4))
            if p is not None and n:
                row["p_model"] = round(float(p[ok].mean()), 4)
            rows.append(row)
    return rows


def _merge_counts(acc, cells):
    for k in range(len(CEIL_LENGTHS)):
        for m in range(1, CEIL_M + 1):
            sel = (cells["lb"] == k) & (cells["m"] == m)
            ok = sel & cells["consistent"]
            c = acc.setdefault((k, m), [0, 0, 0])
            c[0] += int(ok.sum())
            c[1] += int(cells["hit"][ok].sum())
            c[2] += int(sel.sum())


def measure_copy_ceiling(args):
    import data_fineweb as DF
    out = dict(lengths=CEIL_LENGTHS, m_max=CEIL_M)
    if args.corpus_tokens:
        d = os.path.join(args.data, DF.TAG)
        shards = sorted(int(f[6:9]) for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".bin")
                        and int(f[6:9]) != DF.VALID_SHARD)
        per_shard = int(args.corpus_tokens / len(shards))       # her parcanin basindan esit pay (belge sinirinda)
        acc, total = {}, 0
        for i in shards:
            t0 = time.time()
            a = np.memmap(os.path.join(d, "shard_%03d.bin" % i), dtype=np.uint16, mode="r")
            offs = np.load(os.path.join(d, "shard_%03d_offsets.npy" % i))
            lo = 0
            while lo < min(per_shard, len(a)):                  # parcalar belge baslarinda kesilir
                k_lo = int(np.searchsorted(offs, lo))
                j = max(int(np.searchsorted(offs, lo + CEIL_CHUNK, side="right")) - 1, k_lo + 1)
                hi = int(offs[j]) if j < len(offs) else len(a)
                chunk = np.asarray(a[lo:hi])
                st = offs[k_lo:j] - lo
                _merge_counts(acc, _copy_cells(chunk, st.astype(np.int64)))
                total += len(chunk)
                lo = hi
            done = lo
            _say("parca %03d: %.0f M token, toplam %.0f M, %.0f sn" % (i, done / 1e6, total / 1e6, time.time() - t0))
        rows = []
        for (k, m), (n, hit, n_all) in sorted(acc.items()):
            g = hit / n if n else None
            rows.append(dict(l=CEIL_LENGTHS[k], m=m, n=n, n_all=n_all, inconsistent=round(1 - n / n_all, 4) if n_all else None,
                             g=None if g is None else round(g, 4), se=None if g is None else round(math.sqrt(g * (1 - g) / n), 5)))
        out["corpus"] = dict(tokens=total, shards=shards, rows=rows)
    if args.model_docs:
        v = DF.load_valid(args.data, log=lambda s: None)
        order = np.random.default_rng(0).permutation(len(v["valid_starts"]))[:args.model_docs]
        docs = [DF.valid_doc(v, int(k))[:args.model_tokens + 1] for k in sorted(order)]
        stream = np.concatenate([np.asarray(x, np.int64) for x in docs])
        starts = np.cumsum([0] + [len(x) for x in docs[:-1]]).astype(np.int64)
        cells = _copy_cells(stream.astype(np.uint16), starts)
        out["valid"] = dict(docs=len(docs), tokens=int(len(stream)), weights={})
        for w in (["last", "ema"] if args.weights == "both" else [args.weights]):
            model = load_model(args.run, w, args.device)
            p = np.zeros(len(stream))
            dev = next(model.parameters()).device
            with torch.no_grad():
                for s0, x in zip(starts, docs):
                    lp = torch.softmax(model.logits(torch.tensor([x[:-1]], device=dev))[0].float(), -1)
                    sel = (cells["t"] >= s0) & (cells["t"] < s0 + len(x) - 1)
                    tt = cells["t"][sel] - s0
                    if len(tt):
                        p[cells["t"][sel]] = lp[torch.tensor(tt, device=dev), torch.tensor(cells["pred"][sel], device=dev)].cpu().numpy()
            out["valid"]["weights"][w] = _cell_table(cells, p[cells["t"]])
            del model
    return out


def text_copy_ceiling(res, n_min=400):
    L = ["# COPY_CEILING g(l, m): verbatim kopyanin sonraki token'da devam etme orani",
         "l = konum t'de biten ve belgede daha once gecmis EN UZUN sonekin boyu, kova alt siniri (2: 2-3, 4: 4-7, ..., 64: 64+);",
         "m = o kova boyundaki sonekin belgede onceki gecis sayisi (5 = 5+); tahmin = onceki gecislerin devami (hepsi ayniysa, "
         "O12 gibi; degilse 'tutarsiz', g'ye girmez); g = gercek sonraki token = tahmin orani"]
    if "corpus" in res:
        c = res["corpus"]
        L += ["", "## egitim parcalari: %.2f milyar token (parca basina esit pay, belge sinirinda)" % (c["tokens"] / 1e9),
              "l \\ m | " + " | ".join("m=%d%s" % (m, "+" if m == res["m_max"] else "") for m in range(1, res["m_max"] + 1))]
        for Lb in res["lengths"]:
            cells = {r["m"]: r for r in c["rows"] if r["l"] == Lb}
            L.append("%3d%s | " % (Lb, "+" if Lb == res["lengths"][-1] else " ") + " | ".join(
                ("%.3f ± %.3f (n %d%s)" % (r["g"], r["se"], r["n"], "" if r["n"] >= n_min else ", <n_min") if r and r["n"]
                 else "-") for r in (cells.get(m) for m in range(1, res["m_max"] + 1))))
    if "valid" in res:
        v = res["valid"]
        for w, rows in v["weights"].items():
            L += ["", "## valid (%d belge, %d token), agirlik %s: veri g ± SE / model p ort. (n)" % (v["docs"], v["tokens"], w)]
            for Lb in res["lengths"]:
                cells = {r["m"]: r for r in rows if r["l"] == Lb}
                L.append("%3d | " % Lb + " | ".join(
                    ("%.3f±%.3f / %.3f (%d)" % (r["g"], r["se"], r["p_model"], r["n"]) if r and r["n"] else "-")
                    for r in (cells.get(m) for m in range(1, res["m_max"] + 1))))
    return L


# ---- repeat_entry: tekrara giris

ENTRY_MIN_SENT = 5          # tam cumle en az bu kadar token
ENTRY_MAX_L = 32            # l en cok bu kadar izlenir
ENTRY_REGION = 256          # devam bolgesi (token)
ENTRY_CUTS = (64, 128, 256, 512, 1024, 2048)
_MOD = (1 << 61) - 1


def _dec(vocab, ids):
    """vocab: bizim sozluk listesi ya da cagrilabilir cozucu (hazir modelin tokenizer'i, len() ile)."""
    return vocab(list(ids)) if callable(vocab) else decode(ids, vocab)


def _sentence_end_table(vocab):
    """token -> cumle sonu mu: metni '.', '!', '?' ile biten (sondaki bosluk / tirnak / parantez atilarak) ya da satir sonu
    iceren token."""
    out = np.zeros(len(vocab), bool)
    for i in range(len(vocab)):
        t = _dec(vocab, [i])
        out[i] = "\n" in t or t.rstrip(" \"')]").endswith((".", "!", "?"))
    return out


def _match_trace(x):
    """Konum t icin l_t = t'de biten ve daha once (bitisi < t) gecmis en uzun sonek (<= ENTRY_MAX_L), m_t = o sonekin onceki
    gecis sayisi, copy_t = en son onceki gecisin devami (kopya token'i; l_t = 0 ise -1).  Polinom hash, O(n ENTRY_MAX_L)."""
    n = len(x)
    pre = [0] * (n + 1)
    pw = [1] * (ENTRY_MAX_L + 1)
    B = 1_000_003
    for i in range(n):
        pre[i + 1] = (pre[i] * B + int(x[i]) + 1) % _MOD
    for i in range(1, ENTRY_MAX_L + 1):
        pw[i] = pw[i - 1] * B % _MOD
    last = [dict() for _ in range(ENTRY_MAX_L + 1)]
    count = [dict() for _ in range(ENTRY_MAX_L + 1)]
    ell, mm, cp = np.zeros(n, np.int64), np.zeros(n, np.int64), np.full(n, -1, np.int64)
    prev_l = 0
    for t in range(n):
        top = min(prev_l + 1, ENTRY_MAX_L, t + 1)
        found = 0
        for L in range(top, 0, -1):
            h = (pre[t + 1] - pre[t + 1 - L] * pw[L]) % _MOD
            j = last[L].get(h)
            if j is not None:
                found, ell[t], mm[t], cp[t] = L, L, count[L][h], int(x[j + 1])
                break
        prev_l = found
        for L in range(1, min(ENTRY_MAX_L, t + 1) + 1):  # t'de biten sonekler (t + 1 varsa)
            if t + 1 < n:
                h = (pre[t + 1] - pre[t + 1 - L] * pw[L]) % _MOD
                last[L][h] = t
                count[L][h] = count[L].get(h, 0) + 1
    return ell, mm, cp


def _sentences_of(x, is_end, vocab):
    """-> [(bas, son, anahtar)]: cumle sonu token'iyla biten parcalar (ilk parca metin basindan); anahtar = cozulmus metin,
    bosluklari atilmis (satir basi / bosluklu yazim farki silinir).  ENTRY_MIN_SENT'ten kisa ya da bos olanlar None."""
    out, st = [], 0
    for i, t in enumerate(x):
        if is_end[t]:
            key = _dec(vocab, x[st:i + 1]).strip() if i + 1 - st >= ENTRY_MIN_SENT else None
            out.append((st, i, key or None))
            st = i + 1
    return out


def _entry_stats(x, region_start, is_end, vocab, trace=None):
    """x: tam metin (baglam + bolge); bolge = x[region_start:].  -> giris konumlari (bolge basindan) ve kalan tekrar payi:
    sent (bolgede biten ilk cumle, daha once tam olarak gecmis), l8 / l16 (l_t >= L olan ilk konum, giris = t - L + 1)."""
    ell, mm, cp = trace if trace is not None else _match_trace(x)
    n, R = len(x), len(x) - region_start
    rep_tok = np.zeros(n, bool)
    seen, sent_entry = set(), None
    for a, b, key in _sentences_of(x, is_end, vocab):
        if key is not None:
            if key in seen and b >= region_start:
                rep_tok[a:b + 1] = True
                if sent_entry is None:
                    sent_entry = max(a, region_start) - region_start
            seen.add(key)
    out = dict(R=R, sent=sent_entry,
               sent_rest=float(rep_tok[region_start + sent_entry:].mean()) if sent_entry is not None and sent_entry < R else None)
    for L in (8, 16):
        hit = np.flatnonzero(ell[region_start:] >= L)
        if len(hit):
            e = max(int(hit[0]) - L + 1, 0)
            out["l%d" % L] = e
            out["l%d_rest" % L] = float((ell[region_start + e:] >= L).mean()) if e < R else None
        else:
            out["l%d" % L] = None
            out["l%d_rest" % L] = None
    return out


def _entry_summary(rows, cuts):
    """Giris konumu dagilimi: kesimler icinde giris orani (± SE; bolgesi kesimden kisa metinler o kesimde sayilmaz),
    medyan giris, girince kalan tekrar payi (ort.)."""
    out = {}
    for kind in ("sent", "l8", "l16"):
        d = dict(n=len(rows))
        for c in cuts:
            elig = [r for r in rows if r["R"] >= c]
            if elig:
                k = sum(r[kind] is not None and r[kind] < c for r in elig) / len(elig)
                d["by_%d" % c] = (round(k, 4), round(math.sqrt(k * (1 - k) / len(elig)), 4), len(elig))
        ent = [r[kind] for r in rows if r[kind] is not None]
        rest = [r[kind + "_rest"] for r in rows if r.get(kind + "_rest") is not None]
        d["median_entry"] = float(np.median(ent)) if ent else None
        d["rest_share"] = round(float(np.mean(rest)), 4) if rest else None
        d["rest_n"] = len(rest)
        out[kind] = d
    return out


@torch.no_grad()
def _decision_stats(model, texts, region_starts, traces, entry_rows, logits_fn=None):
    """Bolgedeki konumlarda (l_t >= 1, m_t = 1; l kovasi 1, 2, 3, 4-7, 8-15, 16+): gercek / uretilmis sonraki token kopya mi,
    modelin p(kopya), kopya top1 mi, kopya disi kutle, kopya disi dagilimin entropisi, en buyuk rakibin p'si.  Ayrica l8
    girisinin acilis yolunda (giris + 0..7) ayni sayilar, l adimina gore.  logits_fn(x) -> (len(x) - 1, V); None = Model Y."""
    dev = next(model.parameters()).device
    if logits_fn is None:
        logits_fn = lambda x: model.logits(torch.tensor([x[:-1]], device=dev))[0]
    buckets = ((1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, ENTRY_MAX_L))
    acc = {b: [] for b in buckets}
    path = {k: [] for k in range(8)}
    for x, r0, (ell, mm, cp), er in zip(texts, region_starts, traces, entry_rows):
        pos = np.arange(r0 - 1, len(x) - 1)                    # konum t, sonraki token x[t + 1] bolgede
        sel = pos[(cp[pos] >= 0)]
        if not len(sel):
            continue
        z = logits_fn(x).float()
        P = torch.softmax(z[torch.tensor(sel, device=dev)], -1)
        c = torch.tensor(cp[sel], device=dev)
        pc = P.gather(-1, c[:, None])[:, 0]
        top1 = P.argmax(-1) == c
        Q = P.scatter(-1, c[:, None], 0.0)
        mass = Q.sum(-1)
        Qn = Q / mass[:, None].clamp(min=1e-12)
        H = -(Qn * Qn.clamp(min=1e-12).log()).sum(-1)
        rival = Q.max(-1).values
        vals = torch.stack([pc, top1.float(), mass, H, rival], -1).cpu().numpy()
        real = (np.asarray(x)[sel + 1] == cp[sel]).astype(float)
        for i, t in enumerate(sel):
            if mm[t] == 1:
                for b in buckets:
                    if b[0] <= ell[t] <= b[1]:
                        acc[b].append(np.append(vals[i], real[i]))
        if er.get("l8") is not None:
            e = r0 + er["l8"]                                   # kopyanin ilk token'i; karar konumu e - 1 + k
            for k in range(8):
                t = e - 1 + k
                idx = np.flatnonzero(sel == t)
                if len(idx):
                    path[k].append(np.append(vals[idx[0]], [real[idx[0]], ell[t]]))
    names = ("p_copy", "top1", "noncopy_mass", "noncopy_entropy", "rival_p", "copy_rate")
    out = dict(by_l={}, entry_path={})
    for b, v in acc.items():
        if v:
            a = np.array(v)
            out["by_l"]["%d-%d" % b] = dict(n=len(a), **{k: round(float(a[:, j].mean()), 4) for j, k in enumerate(names)},
                                            copy_rate_se=round(float(a[:, 5].std() / math.sqrt(len(a))), 4))
    for k, v in path.items():
        if v:
            a = np.array(v)
            out["entry_path"][k] = dict(n=len(a), l_mean=round(float(a[:, 6].mean()), 2),
                                        **{n_: round(float(a[:, j].mean()), 4) for j, n_ in enumerate(names)})
    return out


def _entry_examples(texts, rows, vocab, k=6, width=60):
    """Veride cumle tekrarina giren belgelerden k pencere: tekrar eden cumlenin cevresi (gozle)."""
    out = []
    for x, r in zip(texts, rows):
        if r["sent"] is not None:
            e = r["sent"]
            out.append(dict(entry=e, before=decode(x[max(0, e - width):e], vocab), at=decode(x[e:e + width], vocab)))
            if len(out) >= k:
                break
    return out


def measure_repeat_entry(model_needed, vocab, eot, args):
    import data_fineweb as DF
    is_end = _sentence_end_table(vocab)
    out = dict(cuts=ENTRY_CUTS, region=ENTRY_REGION, min_sentence=ENTRY_MIN_SENT)
    v = DF.load_valid(args.data, log=lambda s: None)
    order = np.random.default_rng(0).permutation(len(v["valid_starts"]))
    # (1) veri, belge basindan: butun valid belgeleri, en cok 2048 token
    full = [DF.valid_doc(v, int(i))[:2049] for i in order]
    rows = [_entry_stats(d, 1, is_end, vocab) for d in full]           # bolge = eot sonrasi metnin tamami
    out["data_from_start"] = dict(docs=len(rows), summary=_entry_summary(rows, ENTRY_CUTS),
                                  examples=_entry_examples([d[1:] for d in full], rows, vocab))
    _say("veri (belge basindan): %d belge" % len(rows))
    if args.corpus_docs:
        d0 = os.path.join(args.data, DF.TAG)
        crow, per = [], max(1, args.corpus_docs // 4)
        for sh in (0, 3, 6, 9):
            if not os.path.exists(os.path.join(d0, "shard_%03d.bin" % sh)):
                continue
            a = np.memmap(os.path.join(d0, "shard_%03d.bin" % sh), dtype=np.uint16, mode="r")
            offs = np.load(os.path.join(d0, "shard_%03d_offsets.npy" % sh))
            pick = np.random.default_rng(sh).choice(len(offs) - 1, per, replace=False)
            for i in pick:
                doc = np.asarray(a[offs[i]:offs[i + 1]][:2049]).tolist()
                crow.append(_entry_stats(doc, 1, is_end, vocab))
        out["corpus_from_start"] = dict(docs=len(crow), summary=_entry_summary(crow, ENTRY_CUTS))
        _say("corpus (belge basindan): %d belge" % len(crow))
    if not model_needed:
        return out
    # (2) ayni belgelerin yarisindan devam
    docs = [d for d in full if len(d) >= 2 * ENTRY_REGION + 1][:args.entry_docs]
    prompts = [d[:min(len(d) // 2, args.prompt_max)] for d in docs]
    reals = [d[len(p):len(p) + ENTRY_REGION] for d, p in zip(docs, prompts)]
    conds = dict(real=reals)
    out["prompt_max"] = args.prompt_max
    for name, temp, top_p in (("greedy", 0.0, 1.0), ("t1.0", 1.0, 1.0), ("t0.8_p0.9", 0.8, 0.9)):
        gens = []
        for i in range(0, len(prompts), 64):
            g, _, _ = generate_batch(args._model, prompts[i:i + 64], ENTRY_REGION, eot, temp, top_p, seed=0)
            gens += g
        conds[name] = gens
        _say("uretim %s: %d" % (name, len(gens)))
    out["continuation"] = dict(docs=len(docs), conditions={})
    for name, gens in conds.items():
        texts = [p + g for p, g in zip(prompts, gens)]
        traces = [_match_trace(t) for t in texts]
        rows = [_entry_stats(t, len(p), is_end, vocab, tr) for t, p, tr in zip(texts, prompts, traces)]
        res = dict(summary=_entry_summary(rows, (64, 128, 256)))
        if name in ("real", "greedy"):
            res["decisions"] = _decision_stats(args._model, texts, [len(p) for p in prompts], traces, rows)
        if name != "real":
            res["examples"] = [decode(g[:120], vocab) for g in gens[:3]]
        out["continuation"]["conditions"][name] = res
        _say("devam %s: cumle girisi <=256 %s" % (name, res["summary"]["sent"].get("by_256")))
    return out


def text_repeat_entry(res):
    L = ["# TEKRARA GIRIS",
         "cumle = cumle sonu token'iyla (metni . ! ? ile biten ya da satir sonu iceren) biten parca, >= %d token; tekrar = "
         "metni (bosluk atilmis) daha once tam olarak gecmis cumle; l_t = t'de biten ve daha once gecmis en uzun sonek, m_t = "
         "onceki gecis sayisi; giris = bolgede ilk tekrar (cumle) / l >= 8, 16'nin basladigi konum" % res["min_sentence"]]

    def block(title, summ, cuts):
        out = ["", "## " + title, "olcu | " + " | ".join("<= %d: oran ± SE (n)" % c for c in cuts) +
               " | medyan giris | girince kalan tekrar payi (n)"]
        for kind, d in summ.items():
            out.append("%-5s | " % kind + " | ".join(
                ("%.3f ± %.3f (%d)" % d["by_%d" % c]) if "by_%d" % c in d else "-" for c in cuts) +
                " | %s | %s (%d)" % (d["median_entry"], d["rest_share"], d["rest_n"]))
        return out
    if res.get("data_from_start"):
        L += block("veri, valid belge basindan (%d belge, <= 2048 token)" % res["data_from_start"]["docs"],
                   res["data_from_start"]["summary"], res["cuts"])
    if res.get("corpus_from_start"):
        L += block("veri, egitim parcalari belge basindan (%d belge)" % res["corpus_from_start"]["docs"],
                   res["corpus_from_start"]["summary"], res["cuts"])
    if res.get("data_from_start"):
        L += ["", "### veride cumle tekrarina giris ornekleri (gozle)"]
    for e in (res.get("data_from_start") or {}).get("examples", []):
        L.append("[%d] ...%s || %s" % (e["entry"], e["before"][-200:].replace("\n", " / "), e["at"][:240].replace("\n", " / ")))
    if "continuation" in res:
        c = res["continuation"]
        for name, r in c["conditions"].items():
            L += block("devam %s (%d belge, istem = ilk yari <= %d token, bolge %d token%s)" % (
                name, c["docs"], res.get("prompt_max", 1024), res["region"], res.get("token_note", "")),
                       r["summary"], (64, 128, 256))
            if "decisions" in r:
                L += ["karar konumlari (m = 1): l kovasi | n | gercek kopya orani ± SE | model p(kopya) | kopya top1 | kopya disi "
                      "kutle | kopya disi entropi | en buyuk rakip p"]
                for b, d in r["decisions"]["by_l"].items():
                    L.append("  l %-5s | %6d | %.3f ± %.3f | %.3f | %.3f | %.3f | %.2f | %.3f" % (
                        b, d["n"], d["copy_rate"], d["copy_rate_se"], d["p_copy"], d["top1"], d["noncopy_mass"],
                        d["noncopy_entropy"], d["rival_p"]))
                L.append("l8 girisinin acilis yolu (k = kopyanin k. token'i; karar konumu): k | n | l ort | p(kopya) | top1 | "
                         "rakip p | gercekte kopya")
                for k, d in r["decisions"]["entry_path"].items():
                    L.append("  k %d | %5d | %.1f | %.3f | %.3f | %.3f | %.3f" % (
                        k, d["n"], d["l_mean"], d["p_copy"], d["top1"], d["rival_p"], d["copy_rate"]))
            for t in r.get("examples", []):
                L.append("    %s" % t[:300].replace("\n", " / "))
    return L


# ---- repeat_entry_hf: ayni olcu hazir modellerde (transformers)

class _HFDecoder:
    def __init__(self, tok):
        self.tok, self.n = tok, len(tok)

    def __call__(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def __len__(self):
        return self.n


def _hf_card(name):
    """Model kartindan egitim verisi / token satirlari (kural 10: birebir alinti icin)."""
    try:
        from huggingface_hub import hf_hub_download
        text = open(hf_hub_download(name, "README.md"), encoding="utf-8").read()
    except Exception as e:                                  # kart yoksa olcum surer
        return ["(kart okunamadi: %s)" % e]
    keys = ("token", "dataset", "trained", "training data", "corpus", "fineweb", "webtext", "cosmopedia")
    return [ln.strip() for ln in text.splitlines() if any(k in ln.lower() for k in keys)][:25]


@torch.no_grad()
def measure_repeat_entry_hf(vocab, eot, args):
    """repeat_entry'nin (D_032) ayni 1.200 valid belgesi ve tanimlariyla hazir bir modelde: istem = belgenin ilk yarisi (<=
    --prompt-max gpt2 token), bolge 256 token.  gpt2 ailesi: id'ler birebir; baska tokenizer: metin kendi tokenizer'iyla
    yeniden kodlanir (bolge 256 KENDI token'i; l-tabanli satirlar o tokenizer'a ait)."""
    import data_fineweb as DF
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    hf = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.float32).to(args.device).eval()
    same = tok.get_vocab().get("<|endoftext|>") == eot and len(tok) == len(vocab)
    hv = _HFDecoder(tok)
    is_end = _sentence_end_table(hv)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    v = DF.load_valid(args.data, log=lambda s: None)
    order = np.random.default_rng(0).permutation(len(v["valid_starts"]))
    full = [DF.valid_doc(v, int(i))[:2049] for i in order]
    docs = [d for d in full if len(d) >= 2 * ENTRY_REGION + 1][:args.entry_docs]
    prompts_g = [d[:min(len(d) // 2, args.prompt_max)] for d in docs]
    reals_g = [d[len(p):len(p) + ENTRY_REGION] for d, p in zip(docs, prompts_g)]
    if same:
        prompts, reals = prompts_g, reals_g
    else:                                                   # metin -> kendi token'lari
        prompts = [[bos] + tok.encode(decode(p[1:], vocab), add_special_tokens=False) for p in prompts_g]
        reals = [tok.encode(decode(r, vocab), add_special_tokens=False)[:ENTRY_REGION] for r in reals_g]
    dev = args.device
    ctx = getattr(hf.config, "n_positions", None) or getattr(hf.config, "max_position_embeddings", 2048)
    keep = [i for i, p in enumerate(prompts) if len(p) + ENTRY_REGION <= ctx]
    prompts, reals = [prompts[i] for i in keep], [reals[i] for i in keep]

    def gen(temp, top_p):
        out = []
        torch.manual_seed(0)
        for i in range(0, len(prompts), 32):
            part = prompts[i:i + 32]
            L = max(len(q) for q in part)
            pad = eos if tok.pad_token_id is None else tok.pad_token_id
            x = torch.full((len(part), L), pad, dtype=torch.long)
            m = torch.zeros((len(part), L), dtype=torch.long)
            for r, q in enumerate(part):                     # soldan dolgu
                x[r, L - len(q):] = torch.tensor(q)
                m[r, L - len(q):] = 1
            kw = dict(do_sample=temp > 0, max_new_tokens=ENTRY_REGION, min_new_tokens=ENTRY_REGION,
                      suppress_tokens=[eos], pad_token_id=pad)
            if temp > 0:
                kw.update(temperature=temp, top_p=top_p, top_k=0)
            y = hf.generate(input_ids=x.to(dev), attention_mask=m.to(dev), **kw)
            out += [row[L:].tolist() for row in y.cpu()]
        return out

    conds = dict(real=reals)
    for name, temp, top_p in (("greedy", 0.0, 1.0), ("t1.0", 1.0, 1.0), ("t0.8_p0.9", 0.8, 0.9)):
        conds[name] = gen(temp, top_p)
        _say("%s %s: %d" % (args.hf_model, name, len(conds[name])))
    logits_fn = lambda x: hf(torch.tensor([x[:-1]], device=dev)).logits[0]
    out = dict(cuts=ENTRY_CUTS, region=ENTRY_REGION, min_sentence=ENTRY_MIN_SENT, prompt_max=args.prompt_max,
               hf_model=args.hf_model, same_tokenizer=same, context=ctx, card=_hf_card(args.hf_model),
               params=sum(p_.numel() for p_ in hf.parameters()),
               token_note="" if same else "; TOKENIZER FARKLI: l satirlari o tokenizer'in token'lariyla",
               continuation=dict(docs=len(prompts), conditions={}))
    for name, gens in conds.items():
        texts = [q + g for q, g in zip(prompts, gens)]
        traces = [_match_trace(t) for t in texts]
        rows = [_entry_stats(t, len(q), is_end, hv, tr) for t, q, tr in zip(texts, prompts, traces)]
        res = dict(summary=_entry_summary(rows, (64, 128, 256)))
        if name in ("real", "greedy"):
            res["decisions"] = _decision_stats(hf, texts, [len(q) for q in prompts], traces, rows, logits_fn)
        if name != "real":
            res["examples"] = [hv(g[:120]) for g in gens[:3]]
        out["continuation"]["conditions"][name] = res
        _say("devam %s: cumle girisi <=256 %s" % (name, res["summary"]["sent"].get("by_256")))
    return out


def text_repeat_entry_hf(res):
    L = ["# HAZIR MODEL: %s (%.0fM parametre, baglam %d, tokenizer %s)" % (
        res["hf_model"], res["params"] / 1e6, res["context"], "gpt2 ile ayni" if res["same_tokenizer"] else "FARKLI"),
         "model kartindan (README.md, egitim verisi / token satirlari, birebir):"]
    L += ["    " + ln for ln in res["card"]]
    body = text_repeat_entry(dict(res, data_from_start=None))
    return L + body


# ---- O13: isin dogru / acgozlu yanlis

@torch.no_grad()
def measure_beam_facts(model, vocab, eot, args):
    import analyze_capacity as AC
    facts = AC.freq_facts()
    rows = []
    dev = next(model.parameters()).device
    for f in facts:
        prefix = [eot] + encode(f["prompt"], vocab)
        answers = [encode(a, vocab) for a in (f["answer"],) + tuple(f.get("also", ()))]
        z = model.logits(torch.tensor([prefix], device=dev))[0, -1]
        g = int(z.argmax())
        beams = _beam(model, prefix, args.beam, args.beam_steps)
        top = beams[0][0]
        starts = lambda seq: bool(seq) and any(seq[:len(a)] == a[:len(seq)] for a in answers)   # isin cevapla basliyor
        texts = [decode(b, vocab) for b, _ in beams]
        rows.append(dict(kind=f["kind"], prompt=f["prompt"], answer=f["answer"], greedy_first=any(g == a[0] for a in answers),
                         beam_start=starts(top), beam_any=any(a.strip() in texts[0] for a in (f["answer"],) +
                                                              tuple(f.get("also", ()))),
                         beam_in_top8=any(starts(b) for b, _ in beams), beam_text=texts[0], greedy_token=decode([g], vocab)))
    out = {}
    for kind in sorted({r["kind"] for r in rows}) + ["all"]:
        rs = [r for r in rows if kind in ("all", r["kind"])]
        n = len(rs)
        cell = lambda a, b: sum((r["greedy_first"] == a) and (r["beam_start"] == b) for r in rs)
        k = cell(False, True)
        out[kind] = dict(n=n, greedy_right=round(sum(r["greedy_first"] for r in rs) / n, 4),
                         beam_right=round(sum(r["beam_start"] for r in rs) / n, 4),
                         beam_right_greedy_wrong=k, beam_wrong_greedy_right=cell(True, False),
                         share=round(k / n, 4), share_se=round(math.sqrt(k / n * (1 - k / n) / n), 4),
                         beam_any=round(sum(r["beam_any"] for r in rs) / n, 4),
                         beam_in_top8=round(sum(r["beam_in_top8"] for r in rs) / n, 4))
    return dict(summary=out, beam=args.beam, beam_steps=args.beam_steps,
                examples=[r for r in rows if r["beam_start"] and not r["greedy_first"]][:12],
                losses=[r for r in rows if r["greedy_first"] and not r["beam_start"]][:6])


def text_beam_facts(res):
    L = ["# O13: ISIN %d x %d ILE ACGOZLU, 312 OLGU (analyze_capacity.freq_facts)" % (res["beam"], res["beam_steps"]),
         "dogru = devam dogru cevabin (ya da kabul edilen yazimin) token'lariyla basliyor; acgozlu = ilk token",
         "tur | n | acgozlu dogru | isin dogru | isin dogru & acgozlu yanlis (pay ± SE) | acgozlu dogru & isin yanlis | "
         "cevap isin metninde herhangi yerde | ilk 8 isindan biri dogru"]
    for k, d in res["summary"].items():
        L.append("%-8s | %d | %.3f | %.3f | %d (%.3f ± %.3f) | %d | %.3f | %.3f" % (
            k, d["n"], d["greedy_right"], d["beam_right"], d["beam_right_greedy_wrong"], d["share"], d["share_se"],
            d["beam_wrong_greedy_right"], d["beam_any"], d["beam_in_top8"]))
    L += ["", "## isin dogru, acgozlu yanlis ornekleri"]
    L += ["%r -> acgozlu %r | isin %r" % (r["prompt"], r["greedy_token"], r["beam_text"]) for r in res["examples"]]
    L += ["", "## acgozlu dogru, isin yanlis"]
    L += ["%r -> acgozlu %r | isin %r" % (r["prompt"], r["greedy_token"], r["beam_text"]) for r in res["losses"]]
    return L


# ---- O14: tekrar cumlesinden sonra yine tekrar

def _chain_stats(x, region_start, is_end, vocab, trace=None):
    """Bolgede biten tekrar cumlesi i (>= ENTRY_MIN_SENT token, daha once tam gecmis) ve hemen ardindan tam cumle i + 1 icin:
    any = i + 1 de tekrar; seq = i + 1, i'nin en son onceki gecisinden sonraki cumleyle ayni (kopyalanan parcanin devami);
    sinir: i'nin son token'inda kopya token'i (en uzun sonek eslesmesinin devami) gercekte / uretimde secilmis mi.
    -> [dict(any, seq, boundary_pos, copy_token, chose)]"""
    ell, mm, cp = trace if trace is not None else _match_trace(x)
    sents = _sentences_of(x, is_end, vocab)
    last_at, out = {}, []
    for i, (a, b, key) in enumerate(sents):
        if key is not None and key in last_at and b >= region_start and i + 1 < len(sents) and sents[i + 1][2] is not None:
            j = last_at[key]
            nxt_key = sents[i + 1][2]
            follow = sents[j + 1][2] if j + 1 < len(sents) else None
            seen_before = {s_[2] for s_ in sents[:i + 1] if s_[2] is not None}
            out.append(dict(any=nxt_key in seen_before, seq=follow is not None and nxt_key == follow, boundary=b,
                            copy=int(cp[b]), chose=bool(cp[b] >= 0 and b + 1 < len(x) and x[b + 1] == cp[b])))
        if key is not None:
            last_at[key] = i
    return out


def _chain_summary(rows):
    n = len(rows)
    if not n:
        return dict(n=0)
    f = lambda k: (round(sum(r[k] for r in rows) / n, 4), round(math.sqrt(sum(r[k] for r in rows) / n *
                                                                         (1 - sum(r[k] for r in rows) / n) / n), 4))
    return dict(n=n, g_any=f("any"), g_seq=f("seq"), boundary_copy=f("chose"),
                has_copy=round(sum(r["copy"] >= 0 for r in rows) / n, 4))


@torch.no_grad()
def _boundary_probs(model, texts, chains):
    """Cumle sinirinda (tekrar cumlesinin son token'i) modelin p(kopya), kopya top1 mi, en buyuk rakip."""
    dev = next(model.parameters()).device
    vals = []
    for x, rows in zip(texts, chains):
        rows = [r for r in rows if r["copy"] >= 0]
        if not rows:
            continue
        z = model.logits(torch.tensor([x[:-1]], device=dev))[0].float()
        pos = torch.tensor([r["boundary"] for r in rows], device=dev)
        c = torch.tensor([r["copy"] for r in rows], device=dev)
        P = torch.softmax(z[pos], -1)
        pc = P.gather(-1, c[:, None])[:, 0]
        top1 = P.argmax(-1) == c
        rival = P.scatter(-1, c[:, None], 0.0).max(-1).values
        vals += torch.stack([pc, top1.float(), rival], -1).cpu().tolist()
    if not vals:
        return None
    a = np.array(vals)
    return dict(n=len(a), p_copy=round(float(a[:, 0].mean()), 4), top1=round(float(a[:, 1].mean()), 4),
                rival_p=round(float(a[:, 2].mean()), 4))


def measure_repeat_chain(model, vocab, eot, args):
    import data_fineweb as DF
    is_end = _sentence_end_table(vocab)
    v = DF.load_valid(args.data, log=lambda s: None)
    order = np.random.default_rng(0).permutation(len(v["valid_starts"]))
    full = [DF.valid_doc(v, int(i))[:2049] for i in order]
    rows = [r for d in full for r in _chain_stats(d, 1, is_end, vocab)]
    out = dict(valid=dict(docs=len(full), summary=_chain_summary(rows)))
    _say("valid: %s" % out["valid"]["summary"])
    if args.corpus_docs:
        d0 = os.path.join(args.data, DF.TAG)
        crow, per = [], max(1, args.corpus_docs // 4)
        for sh in (0, 3, 6, 9):
            if not os.path.exists(os.path.join(d0, "shard_%03d.bin" % sh)):
                continue
            a = np.memmap(os.path.join(d0, "shard_%03d.bin" % sh), dtype=np.uint16, mode="r")
            offs = np.load(os.path.join(d0, "shard_%03d_offsets.npy" % sh))
            for i in np.random.default_rng(sh).choice(len(offs) - 1, per, replace=False):
                crow += _chain_stats(np.asarray(a[offs[i]:offs[i + 1]][:2049]).tolist(), 1, is_end, vocab)
        out["corpus"] = dict(docs=per * 4, summary=_chain_summary(crow))
        _say("corpus: %s" % out["corpus"]["summary"])
    docs = [d for d in full if len(d) >= 2 * ENTRY_REGION + 1][:args.entry_docs]
    prompts = [d[:min(len(d) // 2, 1024)] for d in docs]
    gens = []
    for i in range(0, len(prompts), 64):
        gens += generate_batch(model, prompts[i:i + 64], ENTRY_REGION, eot)[0]
    for name, conts in (("real", [d[len(p):len(p) + ENTRY_REGION] for d, p in zip(docs, prompts)]), ("greedy", gens)):
        texts = [p + c for p, c in zip(prompts, conts)]
        chains = [_chain_stats(t, len(p), is_end, vocab) for t, p in zip(texts, prompts)]
        out[name] = dict(docs=len(texts), summary=_chain_summary([r for c in chains for r in c]),
                         boundary_model=_boundary_probs(model, texts, chains))
        _say("%s: %s %s" % (name, out[name]["summary"], out[name]["boundary_model"]))
    return out


def text_repeat_chain(res):
    L = ["# O14: TEKRAR CUMLESINDEN HEMEN SONRA YINE TEKRAR (g_S)",
         "cumle / tekrar tanimi repeat_entry (D_032) ile ayni.  Olay: bolgede biten tekrar cumlesi i, ardindan tam cumle i + 1.",
         "g_any = i + 1 de daha once gecmis bir cumle; g_seq = i + 1, i'nin en son onceki gecisinden sonra gelen cumleyle ayni "
         "(kopyalanan parca suruyor); sinir kopya = i'nin son token'inda en uzun sonek eslesmesinin devami secilmis",
         "kaynak | n (olay) | g_any ± SE | g_seq ± SE | sinirda kopya secildi ± SE | sinirda kopya adayi var | model p(kopya) / "
         "top1 / rakip (n)"]
    for name in ("valid", "corpus", "real", "greedy"):
        if name not in res:
            continue
        d, bm = res[name]["summary"], res[name].get("boundary_model")
        if not d.get("n"):
            L.append("%-7s | 0" % name)
            continue
        L.append("%-7s | %d | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f | %s" % (
            name, d["n"], d["g_any"][0], d["g_any"][1], d["g_seq"][0], d["g_seq"][1], d["boundary_copy"][0],
            d["boundary_copy"][1], d["has_copy"],
            "%.3f / %.3f / %.3f (%d)" % (bm["p_copy"], bm["top1"], bm["rival_p"], bm["n"]) if bm else "-"))
    return L


# ---- CLI

def _out_dir(run_dir):
    d = os.environ.get("KUYRUK_SONUC") or os.path.join(run_dir, "internals", "errors")
    os.makedirs(d, exist_ok=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="kosu klasoru")
    ap.add_argument("measure", choices=("answers", "decoding", "repetition", "loop_heads", "corpus",
                                               "answer_split", "symbol_filler", "cooccur", "training_curve",
                                               "ss_profile", "copy_odds", "copy_calibration", "copy_ceiling",
                                               "repeat_entry", "beam_facts", "repeat_chain", "repeat_entry_hf"))
    ap.add_argument("--data", required=True, help="FineWeb koku (gpt2/tokenizer.json, gpt2/shard_*.bin)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="last", choices=("last", "ema", "both"))
    ap.add_argument("--checkpoint", type=int, help="answers: checkpoint_t<adim>.pt (EMA yedekteki ortalama)")
    ap.add_argument("--tokens", type=int, default=641, help="decoding: uretim boyu (final.json gibi 641)")
    ap.add_argument("--heads", help="copy_odds: tur.head listesi, tur 1'den (varsayilan 5.7,7.5,8.7,10.2)")
    ap.add_argument("--corpus-docs", type=int, default=0, help="repeat_entry: egitim parcalarindan belge (0 = yok)")
    ap.add_argument("--prompt-max", type=int, default=1024, help="repeat_entry(_hf): istem en cok bu kadar gpt2 token")
    ap.add_argument("--hf-model", default="gpt2", help="repeat_entry_hf: HuggingFace model adi")
    ap.add_argument("--entry-docs", type=int, default=1200, help="repeat_entry: devam olcumu icin belge (>= 513 token)")
    ap.add_argument("--corpus-tokens", type=float, default=0, help="copy_ceiling: egitim parcalarindan token (0 = yok)")
    ap.add_argument("--model-docs", type=int, default=0, help="copy_ceiling: valid belgesi (0 = model olcumu yok)")
    ap.add_argument("--model-tokens", type=int, default=4096, help="copy_ceiling: valid belgesi en cok bu kadar token")
    ap.add_argument("--steps", default="final", help="training_curve: virgulle adimlar ve/veya final")
    ap.add_argument("--beam", type=int, default=8, help="answer_split: isin genisligi")
    ap.add_argument("--beam-steps", type=int, default=6, help="answer_split: isin boyu (token)")
    ap.add_argument("--settings", help="decoding: virgulle ayar adlari (varsayilan hepsi)")
    ap.add_argument("--docs", type=int, default=0, help="decoding / training_curve: sinav belgesi devami sayisi (ilk yari istem; training_curve 0 = 32)")
    ap.add_argument("--long", type=int, default=0, help="decoding: ornekleme / DRY ayarlarinda tek istemden bu kadar token")
    ap.add_argument("--ablate-tokens", type=int, default=200, help="loop_heads: mudahaleli acgozlu uretim boyu")
    ap.add_argument("--shards", type=int, default=0, help="corpus: ilk K parca (0 = hepsi)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    vocab, eot = (None, None) if args.measure == "ss_profile" else load_vocab(args.data)
    out_dir = _out_dir(args.run)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    weights_list = ["none"] if args.measure in ("corpus", "cooccur", "training_curve", "ss_profile", "copy_ceiling",
                                                "repeat_entry_hf") else (["last", "ema"] if args.weights == "both" else [args.weights])
    for w in weights_list:
        t1 = time.time()
        if args.measure in ("corpus", "cooccur"):
            res = measure_corpus(vocab, eot, args) if args.measure == "corpus" else measure_cooccur(vocab, eot, args)
            lines = text_corpus(res) if args.measure == "corpus" else text_cooccur(res)
            source = "egitim parcalari"
        elif args.measure == "training_curve":
            res = measure_training_curve(vocab, eot, args)
            lines = text_training_curve(res)
            source = "checkpoint'ler %s" % args.steps
        elif args.measure == "copy_ceiling":
            res = measure_copy_ceiling(args)
            lines = text_copy_ceiling(res)
            source = "egitim parcalari / valid"
        elif args.measure == "repeat_entry_hf":
            res = measure_repeat_entry_hf(vocab, eot, args)
            lines = text_repeat_entry_hf(res)
            source = "hazir model %s" % args.hf_model
        elif args.measure == "ss_profile":
            res = measure_ss_profile(args)
            lines = text_ss_profile(res)
            source = "model.pt / model_weight_ema.pt"
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
            elif args.measure == "answer_split":
                res = measure_answer_split(model, vocab, eot, args)
                lines = text_answer_split(res)
            elif args.measure == "copy_odds":
                res = measure_copy_odds(model, vocab, eot, args)
                lines = text_copy_odds(res)
            elif args.measure == "copy_calibration":
                res = measure_copy_calibration(model, vocab, eot, args)
                lines = text_copy_calibration(res)
            elif args.measure == "repeat_entry":
                args._model = model
                res = measure_repeat_entry(True, vocab, eot, args)
                lines = text_repeat_entry(res)
            elif args.measure == "beam_facts":
                res = measure_beam_facts(model, vocab, eot, args)
                lines = text_beam_facts(res)
            elif args.measure == "repeat_chain":
                res = measure_repeat_chain(model, vocab, eot, args)
                lines = text_repeat_chain(res)
            elif args.measure == "symbol_filler":
                res = measure_symbol_filler(model, vocab, eot, args)
                lines = text_symbol_filler(res)
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
