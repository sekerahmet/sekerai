# -*- coding: utf-8 -*-
"""exam_tinystories -- TinyStories sinavi.  Model ve genel egitim bir ust klasorde (model_20.py, train_20.py).

    exam        sabit validation alt kumesinde (exam_rows) kayip ve sonraki token accuracy'si; olculer model_18
                data_stories.evaluate'inkiler (konum bantlari, eos_ok, acc_ar)
    texts       sabit istemlerden acgozlu devam, ilk <eos>'ta kesilir -- GOZLE okunur (kural 12)
    loop_check  uretilen metinde tekrar (farkli 4'lu orani, tekrar eden 8'li)
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_20: model ve genel egitim

import data_tinystories as DT  # noqa: E402
from train_20 import generate  # noqa: E402

EXAM_STORIES = 1000      # ~190 bin hedef token (ortalama hikaye 194 token): accuracy'nin standart hatasi ~0,001
BANDS = ((0, 64), (64, 256), (256, 512))     # hedefin penceredeki konumu (model_18 ile ayni)
NAME_PROMPT = ("Once upon a time, there was a girl named Lily. She had a red ball. One day, Lily went to the "
               "park with her ball. At the park, she met a boy named Tom. Tom asked,")
# Makalenin "Evaluation prompts.yaml"inden butun kelimeleri v4 sozlugunde olanlar (dosyadaki sirayla 1, 2, 3, 5, 7, 9,
# 10, 11, 12, 13, 17; paragraf sonlari korunur) + isim istemi.  Ilk PROBE_PROMPTS = model_18'in sonda seti.
PROMPTS = (
    "One day a girl walked into the living room and noticed something very strange. There was a huge cabinet standing "
    "in the corner. It looked very old and heavy. She walked over and tried to open it, when suddenly",
    "Once upon a time, there lived a hamster in the forest. Every day, he would walked around the forest and looking for "
    "adventures. One day, he heard someone calling out from behind the bushes.\n\nThe hamster listened carefully. He "
    "realised that it was a small mouse calling out for help. It got stuck under a heavy log and couldn't get out. The "
    "hamster immediately realized that",
    "Jack asked his mom if he could ride the bike all the way to his grandmother's house. She agreed, but she said that "
    "he shouldn't ride too fast because it was raining and the path was wet.\n\nHe started riding and realized he was "
    "hungry, so he decided that he should get to his grandmother's house as fast as possible. But then he remembered "
    "his mum's words \"",
    NAME_PROMPT,
    "Alice walked up to her friend Ben's house. She was planning to ask him to go to the park with her. When Ben opened "
    "the door, she asked him if he had any plans. He said, \"I'm sorry, Alice, but",
    "Alice walked into the kitchen and saw Ben who was looking for something but looked frustrated. She said, \"Ben, why "
    "are you",
    "Once upon a time, there was tiger who liked to play the guitar. One day, a bunny heard the guitar from a distance and",
    "Alice wanted to play with her doll, but she couldn't remember where she had put it. She looked all around the house "
    "but couldn't find it, so she decided",
    "\"Ben, what do you have in your pocket?\", Alice asked. \"Oh, nothing.\", Ben replied. But Alice saw that there was "
    "definitely something in Ben's pocket, and she was very curious what it was, so she",
    "One day, a bird was flying high over the sea. At some point the bird noticed small boat with a boy sitting inside. "
    "The boy looked lost so",
    "Jimmy was on his way to school with his father when he noticed a tree with strange looking fruits on it. He asked "
    "his father, \"Can I try",
    "Anna was a popular girl, who walked to school every day carrying a very heavy backpack. It was so heavy that she "
    "could hardly walk.\n\nOne day, Anna's friends asked her why her backpack was so heavy. She said that",
)
PROBE_PROMPTS = 4        # her sinavda ilk 4 istem; sonda hepsi


def exam_rows(data, count=EXAM_STORIES):
    """Sabit alt kume: train'de birebir gecmeyen valid pencerelerinden tohum 0 ile secilen count tanesi, sirali.
    count=None: hepsi."""
    ok = np.setdiff1d(np.arange(len(data["valid_start"])), data["valid_in_train"])
    if count is None or count >= len(ok):
        return ok
    return np.sort(ok[np.random.default_rng(0).permutation(len(ok))[:count]])


def repeated_bigram(w, a):
    """w (B,T) token, a (B,T-1) sayilan hedef -> (B,T-1) bool: hedef w_(t+1), (w_t, w_(t+1)) ikilisini BU hikayede
    daha once tamamlanmis (model_18 data_stories'ten kopya; Zoology 2312.04927'nin "AR hit"i)."""
    B, L = a.shape
    n = int(w.max()) + 1
    order = torch.arange(B * L, device=w.device).view(B, L)
    row_offset = torch.arange(B, device=w.device)[:, None] * n * n
    keys = torch.where(a, row_offset + w[:, :-1] * n + w[:, 1:], -1 - order).reshape(-1)   # sayilmayan eslesmez
    sort_idx = torch.sort(keys, stable=True).indices
    repeated = torch.zeros_like(keys, dtype=torch.bool)
    repeated[sort_idx[1:]] = keys[sort_idx[1:]] == keys[sort_idx[:-1]]
    return repeated.view(B, L)


@torch.no_grad()
def exam(model, data, rows, batch_size=64):
    """rows (valid pencereleri) uzerinde, dolgu disindaki her hedefte:
      nll, ppl        ortalama -log p(hedef), e^nll   (model.loss'un nll'i, capasiz)
      accuracy        en yuksek skor hedef mi
      acc_a_b         hedefin konumu a..b arasinda olanlarda accuracy
      eos_ok          hikaye bittiginde en yuksek skor <eos> mu
      acc_ar/other    hedef bu hikayede daha once tamamlanmis bir ikiliyi tamamliyor mu / geri kalan
    Pencereler boya gore siralanip parcanin en uzununa kirpilir: dolgu sagda ve attention nedensel, hesap ayni."""
    device = next(model.parameters()).device
    eos = data["vocab"].index(DT.EOS_TOKEN)
    rows = np.asarray(rows)
    order = rows[np.argsort(data["valid_length"][rows], kind="stable")]
    nll_sum, count = 0.0, {"all": [0, 0], "eos": [0, 0], "ar": [0, 0], "other": [0, 0]}
    count.update({b: [0, 0] for b in BANDS})
    for i in range(0, len(order), batch_size):
        ids, mask = DT.sequences(data, "valid", order[i:i + batch_size])
        T = int(mask.sum(1).max())
        ids, mask = ids[:, :T].to(device), mask[:, :T].to(device)
        logits = model.logits(ids[:, :-1]).float()
        target, valid = ids[:, 1:], mask[:, 1:]
        ce = logits.logsumexp(-1) - logits.gather(-1, target[..., None])[..., 0]
        hit = logits.argmax(-1) == target
        nll_sum += float(ce[valid].sum())
        k = torch.arange(1, T, device=device)                 # hedefin penceredeki konumu
        rep = repeated_bigram(ids, valid)
        groups = {"all": valid, "eos": valid & (target == eos), "ar": valid & rep, "other": valid & ~rep}
        groups.update({(b0, b1): valid & (k >= b0) & (k < b1) for b0, b1 in BANDS})
        for name, sel in groups.items():
            count[name][0] += int((hit & sel).sum())
            count[name][1] += int(sel.sum())
    rate = lambda c: c[0] / c[1] if c[1] else float("nan")
    n = count["all"][1]
    out = dict(n=n, nll=nll_sum / max(n, 1), accuracy=rate(count["all"]))
    out["ppl"] = math.exp(out["nll"])
    out.update({"acc_%d_%d" % b: rate(count[b]) for b in BANDS})
    out.update(eos_ok=rate(count["eos"]), acc_ar=rate(count["ar"]), acc_other=rate(count["other"]))
    return out


def story_prompts(data, rows):
    """Valid hikayelerinin ilk yarisi istem, ikinci yarisi gercek devam: ([<eos> + ilk yari], [ikinci yari])."""
    a, eos = data["valid"], data["vocab"].index(DT.EOS_TOKEN)
    prompts, reals = [], []
    for s, L in zip(data["valid_start"][rows].tolist(), data["valid_length"][rows].tolist()):
        prompts.append([eos] + a[s:s + L // 2].tolist())
        reals.append(a[s + L // 2:s + L].tolist())
    return prompts, reals


@torch.no_grad()
def texts(model, data, prompts, n, reals=None):
    """Acgozlu devam (train_20.generate), ilk <eos>'ta kesilir.  prompts: <eos> ile baslayan id listeleri; istem + n
    pencereye (seq_len) sigacak kadar uretilir.  -> [dict(prompt, model, ended[, real], ids)] metinler okunur halde."""
    vocab = data["vocab"]
    eos = vocab.index(DT.EOS_TOKEN)
    n = min(n, data["seq_len"] - max(len(p) for p in prompts))
    assert n > 0, "istem pencereyi dolduruyor"
    out = []
    for i, (p, g) in enumerate(zip(prompts, generate(model, prompts, n))):
        ended = eos in g
        g = g[:g.index(eos)] if ended else g
        in_quote = sum(vocab[t] == '"' for t in p[1:]) % 2 == 1          # istem tirnagi acik biraktiysa
        row = dict(prompt=DT.decode(p[1:], vocab), model=DT.decode(g, vocab, in_quote=in_quote), ended=ended, ids=g)
        if reals is not None:
            row["real"] = DT.decode(reals[i], vocab, in_quote=in_quote)
        out.append(row)
    return out


def loop_check(generated):
    """id listeleri -> distinct4: farkli 4'lulerin orani (1: tekrar yok), loop: tekrar eden 8'lisi olan metin sayisi."""
    f4, loops = [], 0
    for t in generated:
        g4 = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
        g8 = [tuple(t[i:i + 8]) for i in range(len(t) - 7)]
        f4.append(len(set(g4)) / len(g4) if g4 else 1.0)
        loops += len(set(g8)) < len(g8)
    return dict(distinct4=sum(f4) / max(len(f4), 1), loop=loops)
