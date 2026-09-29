# -*- coding: utf-8 -*-
"""exam_simplestories -- SimpleStories sinavi (exam_tinystories duzeni).  Model ve genel egitim bir ust klasorde
(model_y.py, train_y.py).

    exam        sinav kumesinin (data["exam"]) sabit alt kumesinde (exam_rows) kayip ve sonraki token accuracy'si, konum
                bantlari, eos_ok, acc_ar (exam_tinystories ile ayni olculer) + bits_per_byte (EOS haric, esas; kullanici,
                28 Eylul: "4 kabul") ve bits_per_byte_eos (EOS dahil, hikaye basina +1 bayt)
    texts       sabit istemlerden acgozlu devam, ilk <eos>'ta kesilir -- GOZLE okunur (kural 12)
    loop_check  uretilen metinde tekrar (farkli 4'lu orani, tekrar eden 8'li)
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_y: model ve genel egitim

import data_simplestories as DS  # noqa: E402
from train_y import generate  # noqa: E402

EXAM_STORIES = 1000      # ~287 bin hedef token (ortalama hikaye ~287 token)
BANDS = ((0, 64), (64, 256), (256, 512))     # hedefin penceredeki konumu (exam_tinystories ile ayni)
LOGITS_BUDGET = 1 << 28  # sinav batch'i: batch x seq_len x sozluk <= bu (fp32 1 GB); ss4096'da 64, gpt2'de 10
# SimpleStories valid'inden (sinav kumesi): tohum 0 ile karisik sirada temasi farkli ilk 12 hikaye, istem = ilk paragrafin
# ilk 30 kelimesi (paragraf >= 36 kelime).  TinyStories istemleri SimpleStories'in dagilimina uymayabilir.  Yorum: valid
# sirasi | theme | topic | style.  Ilk PROBE_PROMPTS her sinavda, hepsi sonda.
PROMPTS = (
    "Whispers danced through the air as the sun rose over the hills. A girl named Lily woke up with a smile. Today was the "
    "day she would find the hidden",                                             # 2307 | Trust | hidden treasures | lyric
    "Tightly, she clutched the map, a guide to find the treasure that would mend her heart. In the days before, she had "
    "faced pain and betrayal, but now, hope filled",                             # 8375 | Revenge | treasure hunts | romantic
    "A fiery comet streaked across the night sky, lighting up the world below. A young boy watched in awe, wishing he "
    "could ride it to the stars. He gathered his",                               # 3073 | Failure | the sky | mythological
    "Near the edge of a small town, a young girl found an old map in her attic. The map was worn and faded, with strange "
    "lines and marks all over",                                                  # 14456 | Hardship | mysterious maps | classic
    "Rugged mountains towered above Leo as he set out for school. He always dreamed of climbing to the top. One afternoon, "
    "during gym class, a mysterious fog rolled in and",                          # 15764 | Adventure | school life | mystical
    "Boredom swept over Alex as he watched a puppet show in the square. The puppets were funny, but he thought he could "
    "do better. So, he got his friend, Rita,",                                   # 17271 | Friendship | the arts | humorous
    "Mushrooms grew in clusters around the base of the ancient tree. A curious girl had always felt drawn to the forest. "
    "She had heard tales of a magical land hidden",                              # 14786 | Belonging | magical lands | heartwarming
    "Murmurs filled the air as she watched the world change around her. She felt trapped, like a bird in a cage. Each "
    "day, she wished for a chance to fly.",                                      # 18224 | Intelligence | time travel | romantic
    "Inside a dark cabin, a pirate named Alex was counting his coins. He loved gold more than anything. \"I am the richest "
    "pirate!\" he bragged. But even as he laughed,",                             # 2189 | Self-Acceptance | pirates | modern
    "Carefully, a brave knight walked through the dark forest. The tall trees loomed over him, their leaves whispering "
    "secrets of old. In his hand, he held a shining sword, a",                   # 2787 | Long-Term Thinking | fantasy worlds | action-packed
    "Gently, Leo brushed the dust off an old globe he found. It spun slowly, revealing lands he had never seen before. "
    "Fascinated, he spotted a tiny island marked with a",                        # 19306 | Planning | magical lands | surreal
    "In a cozy little corner of her room, Alice discovered a tiny mouse with a broken toy. The mouse looked sad as it "
    "tried to play. Alice knelt down and",                                       # 4323 | Helping Others | miniature worlds | humorous
)
PROBE_PROMPTS = 4        # her sinavda ilk 4 istem; sonda hepsi


def exam_rows(data, count=EXAM_STORIES):
    """Sabit alt kume: sinav kumesinin (iki tokenizer'da da sigan, sizintisiz valid pencereleri) tohum 0 ile secilen count
    tanesi, sirali.  count=None: hepsi."""
    ok = data["exam"]
    if count is None or count >= len(ok):
        return ok
    return np.sort(ok[np.random.default_rng(0).permutation(len(ok))[:count]])


def repeated_bigram(w, a):
    """w (B,T) token, a (B,T-1) sayilan hedef -> (B,T-1) bool: hedef w_(t+1), (w_t, w_(t+1)) ikilisini BU hikayede
    daha once tamamlanmis (exam_tinystories'ten kopya; Zoology 2312.04927'nin "AR hit"i)."""
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
def exam(model, data, rows, batch_size=None):
    """rows (valid pencereleri) uzerinde, dolgu disindaki her hedefte:
      nll, ppl           ortalama -log p(hedef), e^nll   (model.loss'un nll'i, capasiz; eos hedefi dahil)
      accuracy           en yuksek skor hedef mi
      acc_a_b            hedefin konumu a..b arasinda olanlarda accuracy
      eos_ok             hikaye bittiginde en yuksek skor <eos> mu
      acc_ar/other       hedef bu hikayede daha once tamamlanmis bir ikiliyi tamamliyor mu / geri kalan
      bits_per_byte      eos haric hedeflerin toplam nll'i / ln 2 / hikayelerin UTF-8 bayti (esas; tokenizer'dan bagimsiz)
      bits_per_byte_eos  butun hedefler / ln 2 / (bayt + hikaye sayisi)
    Pencereler boya gore siralanip parcanin en uzununa kirpilir: dolgu sagda ve attention nedensel, hesap ayni.
    batch_size: varsayilan LOGITS_BUDGET'tan (sozluk buyudukce kuculur)."""
    device = next(model.parameters()).device
    eos = data["vocab"].index(DS.EOS_TOKEN)
    if batch_size is None:
        batch_size = max(1, min(64, LOGITS_BUDGET // (len(data["vocab"]) * data["seq_len"])))
    rows = np.asarray(rows)
    order = rows[np.argsort(data["valid_length"][rows], kind="stable")]
    nll_sum, story_nll = 0.0, 0.0
    count = {"all": [0, 0], "eos": [0, 0], "ar": [0, 0], "other": [0, 0]}
    count.update({b: [0, 0] for b in BANDS})
    for i in range(0, len(order), batch_size):
        ids, mask = DS.sequences(data, "valid", order[i:i + batch_size])
        T = int(mask.sum(1).max())
        ids, mask = ids[:, :T].to(device), mask[:, :T].to(device)
        logits = model.logits(ids[:, :-1]).float()
        target, valid = ids[:, 1:], mask[:, 1:]
        ce = logits.logsumexp(-1) - logits.gather(-1, target[..., None])[..., 0]
        hit = logits.argmax(-1) == target
        nll_sum += float(ce[valid].sum())
        story_nll += float(ce[valid & (target != eos)].sum())
        k = torch.arange(1, T, device=device)                 # hedefin penceredeki konumu
        rep = repeated_bigram(ids, valid)
        groups = {"all": valid, "eos": valid & (target == eos), "ar": valid & rep, "other": valid & ~rep}
        groups.update({(b0, b1): valid & (k >= b0) & (k < b1) for b0, b1 in BANDS})
        for name, sel in groups.items():
            count[name][0] += int((hit & sel).sum())
            count[name][1] += int(sel.sum())
    rate = lambda c: c[0] / c[1] if c[1] else float("nan")
    n, n_bytes = count["all"][1], int(data["valid_bytes"][rows].sum())
    out = dict(n=n, nll=nll_sum / max(n, 1), accuracy=rate(count["all"]))
    out["ppl"] = math.exp(out["nll"])
    out.update({"acc_%d_%d" % b: rate(count[b]) for b in BANDS})
    out.update(eos_ok=rate(count["eos"]), acc_ar=rate(count["ar"]), acc_other=rate(count["other"]),
               bits_per_byte=DS.bits_per_byte(story_nll, n_bytes),
               bits_per_byte_eos=DS.bits_per_byte(nll_sum, n_bytes + len(rows)), bytes=n_bytes, stories=len(rows))
    return out


def story_prompts(data, rows):
    """Valid hikayelerinin ilk yarisi istem, ikinci yarisi gercek devam: ([<eos> + ilk yari], [ikinci yari])."""
    a, eos = data["valid"], data["vocab"].index(DS.EOS_TOKEN)
    prompts, reals = [], []
    for s, L in zip(data["valid_start"][rows].tolist(), data["valid_length"][rows].tolist()):
        prompts.append([eos] + a[s:s + L // 2].tolist())
        reals.append(a[s + L // 2:s + L].tolist())
    return prompts, reals


@torch.no_grad()
def texts(model, data, prompts, n, reals=None):
    """Acgozlu devam (train_y.generate), ilk <eos>'ta kesilir.  prompts: <eos> ile baslayan id listeleri; istem + n
    pencereye (seq_len) sigacak kadar uretilir.  -> [dict(prompt, model, ended[, real], ids)] metinler okunur halde."""
    vocab = data["vocab"]
    eos = vocab.index(DS.EOS_TOKEN)
    n = min(n, data["seq_len"] - max(len(p) for p in prompts))
    assert n > 0, "istem pencereyi dolduruyor"
    out = []
    for i, (p, g) in enumerate(zip(prompts, generate(model, prompts, n))):
        ended = eos in g
        g = g[:g.index(eos)] if ended else g
        in_quote = sum(vocab[t] == '"' for t in p[1:]) % 2 == 1          # istem tirnagi acik biraktiysa (WordPiece)
        row = dict(prompt=DS.decode(p[1:], vocab), model=DS.decode(g, vocab, in_quote=in_quote), ended=ended, ids=g)
        if reals is not None:
            row["real"] = DS.decode(reals[i], vocab, in_quote=in_quote)
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
