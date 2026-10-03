# -*- coding: utf-8 -*-
"""exam_simplestories -- SimpleStories sinavi (exam_tinystories duzeni).  Model ve genel egitim bir ust klasorde
(model_y.py, train_y.py).

    exam               sinav kumesinin (data["exam"]) sabit alt kumesinde (exam_rows) kayip ve sonraki token accuracy'si,
                       konum bantlari, eos_ok, acc_ar (exam_tinystories ile ayni olculer) + bits_per_byte (EOS haric, esas;
                       kullanici, 28 Eylul: "4 kabul") ve bits_per_byte_eos (EOS dahil, hikaye basina +1 bayt) +
                       nll_by_frequency (hedefin train siklik bandina gore)
    exam_train         ayni sinav, boyu sinav hikayelerine eslenmis train hikayelerinde (ezber; epok > 1 karari)
    alpha_summary      tur basina alpha_A ve alpha_F: medyan ve |alpha|'nin en buyugu
    count_text_errors  sinav hikayelerinin ilk yarisindan acgozlu ve ornekleme devam; metin hatalari
    texts              sabit istemlerden acgozlu devam, ilk <eos>'ta kesilir -- GOZLE okunur (kural 12)
    loop_check         uretilen metinde tekrar (farkli 4'lu orani, tekrar eden 8'li)
Son dort olcu ve adlari: kullanici, 29 Eylul: "Önerilerinin hepsi kabul".
"""
import collections
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_y: model ve genel egitim

import data_simplestories as DS  # noqa: E402
from model_y import AttentionCache, BlockModel  # noqa: E402
from train_y import generate  # noqa: E402

EXAM_STORIES = 1000      # ~287 bin hedef token (ortalama hikaye ~287 token)
BANDS = ((0, 64), (64, 256), (256, 512))     # hedefin penceredeki konumu (exam_tinystories ile ayni)
LOGITS_BUDGET = 1 << 28  # sinav batch'i: batch x seq_len x sozluk <= bu (fp32 1 GB); ss4096'da 64, gpt2'de 10
CACHE_BUDGET = 8 << 30   # onbellekli uretimde satir parcasi: K/V onbellegi + istem attention tablosu (fp32) <= bu bayt
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
XAX_LEGIT_MIN = 5        # 'X and X' train'de en az bu kadar geciyorsa legit ("higher and higher")
STORY_CONTINUATIONS = 100   # count_text_errors: sabit alt kumenin ilk bu kadar hikayesinin ilk yarisindan devam
EXAM_TRAIN_STORIES = 128    # exam_train: ezber olcusu icin train hikayesi
FREQUENCY_BANDS = (64, 512, 2048)    # nll_by_frequency: train siklik sirasi sinirlari (0 = en sik token)
_LONG_SENTENCE, _LONG_QUOTE = 5, 3   # tekrari sayilan cumle / soz en az bu kadar birim (A)
_CACHE = {}              # iz basina bir kez: siklik bandi (bincount), exam_train secimi, valid kalip cumleleri


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


def _frequency_band(data):
    """token -> train siklik bandi (0: en sik FREQUENCY_BANDS[0] token, ...); sayim train'den bir kez (np.bincount)."""
    key = ("band", data["fingerprint"])
    if key not in _CACHE:
        a, V = data["train"], len(data["vocab"])
        counts = sum(np.bincount(a[c:c + DS._CHUNK], minlength=V) for c in range(0, len(a), DS._CHUNK))
        rank = np.empty(V, dtype=np.int64)
        rank[np.argsort(-counts, kind="stable")] = np.arange(V)
        _CACHE[key] = np.searchsorted(np.array(FREQUENCY_BANDS), rank, side="right")
    return _CACHE[key]


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
      nll_by_frequency   hedefin train siklik sirasi bandina gore (ranks): hedef sayisi, nll, mass_ratio = bantta
                         tahmin kutlesi (softmax toplami) / bantta hedef sayisi (1: kalibre)
    Pencereler boya gore siralanip parcanin en uzununa kirpilir: dolgu sagda ve attention nedensel, hesap ayni.
    batch_size: varsayilan LOGITS_BUDGET'tan (sozluk buyudukce kuculur)."""
    device = next(model.parameters()).device
    eos = data["vocab"].index(DS.EOS_TOKEN)
    V = len(data["vocab"])
    if batch_size is None:
        batch_size = max(1, min(64, LOGITS_BUDGET // (V * data["seq_len"])))
    rows = np.asarray(rows)
    order = rows[np.argsort(data["valid_length"][rows], kind="stable")]
    nll_sum, story_nll = 0.0, 0.0
    count = {"all": [0, 0], "eos": [0, 0], "ar": [0, 0], "other": [0, 0]}
    count.update({b: [0, 0] for b in BANDS})
    band = torch.as_tensor(_frequency_band(data), device=device)
    freq = torch.zeros(3, len(FREQUENCY_BANDS) + 1, dtype=torch.float64, device=device)   # hedef, nll, kutle
    for i in range(0, len(order), batch_size):
        ids, mask = DS.sequences(data, "valid", order[i:i + batch_size])
        T = int(mask.sum(1).max())
        ids, mask = ids[:, :T].to(device), mask[:, :T].to(device)
        logits = model.logits(ids[:, :-1]).float()
        target, valid = ids[:, 1:], mask[:, 1:]
        lse = logits.logsumexp(-1)
        ce = lse - logits.gather(-1, target[..., None])[..., 0]
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
        tb = band[target[valid]]
        freq[0].index_add_(0, tb, torch.ones_like(tb, dtype=torch.float64))
        freq[1].index_add_(0, tb, ce[valid].double())
        probs = logits.sub_(lse[..., None]).exp_().reshape(-1, V)   # yerinde: logits bundan sonra kullanilmaz
        freq[2].index_add_(0, band, (valid.reshape(1, -1).float() @ probs)[0].double())
    rate = lambda c: c[0] / c[1] if c[1] else float("nan")
    n, n_bytes = count["all"][1], int(data["valid_bytes"][rows].sum())
    out = dict(n=n, nll=nll_sum / max(n, 1), accuracy=rate(count["all"]))
    out["ppl"] = math.exp(out["nll"])
    out.update({"acc_%d_%d" % b: rate(count[b]) for b in BANDS})
    out.update(eos_ok=rate(count["eos"]), acc_ar=rate(count["ar"]), acc_other=rate(count["other"]),
               bits_per_byte=DS.bits_per_byte(story_nll, n_bytes),
               bits_per_byte_eos=DS.bits_per_byte(nll_sum, n_bytes + len(rows)), bytes=n_bytes, stories=len(rows))
    edges = [min(e, V) for e in (0,) + FREQUENCY_BANDS + (V,)]
    f = freq.cpu().tolist()
    out["nll_by_frequency"] = [dict(ranks=[edges[b], edges[b + 1]], targets=int(f[0][b]),
                                    nll=f[1][b] / f[0][b] if f[0][b] else None,
                                    mass_ratio=f[2][b] / f[0][b] if f[0][b] else None) for b in range(len(f[0]))]
    return out


def exam_train(model, data):
    """exam'in aynisi, egitimde gorulmus EXAM_TRAIN_STORIES train hikayesinde (ezber olcusu: train - sinav farki).  Secim
    bir kez: exam_rows(data, 128)'in her hikayesine, pencereye sigan train hikayelerinin boya gore sirali
    listesinde kendi boyunun yerinden +-200 icinde rastgele biri (tohum 7)."""
    key = ("train_view", data["fingerprint"])
    if key not in _CACHE:
        eos = data["vocab"].index(DS.EOS_TOKEN)
        starts, lengths = DS._stories(data["train"], eos)
        fits = np.flatnonzero((lengths > 0) & (lengths + 2 <= data["seq_len"]))
        order = fits[np.argsort(lengths[fits], kind="stable")]
        sizes, rng = lengths[order], np.random.default_rng(7)
        pick = np.array([order[int(np.clip(np.searchsorted(sizes, L) + rng.integers(-200, 200), 0, len(order) - 1))]
                         for L in data["valid_length"][exam_rows(data, EXAM_TRAIN_STORIES)].tolist()], dtype=np.int64)
        _CACHE[key] = dict(data, valid=data["train"], valid_start=starts[pick], valid_length=lengths[pick],
                           valid_bytes=data["train_bytes"][pick])
    view = _CACHE[key]
    return exam(model, view, np.arange(len(view["valid_start"])))


@torch.no_grad()
def alpha_summary(model):
    """Tur basina alpha_attention ve alpha_facts: median ve max_abs (|alpha|'nin en buyugu; medyan tek basina yaniltir:
    bir turda medyan 0,0055, en buyuk 2,67).  FIRST_TURN_FACTS=False'ta tur 1'in alpha_F'si kullanilmiyor: None."""
    out = {name: dict(median=[round(float(x), 4) for x in a.float().quantile(0.5, dim=-1)],
                      max_abs=[round(float(x), 4) for x in a.abs().max(-1).values])
           for name, a in (("attention", model.alpha_attention), ("facts", model.alpha_facts))}
    if not getattr(model, "first_turn_facts", True):
        out["facts"]["median"][0] = out["facts"]["max_abs"][0] = None
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


# ---- metin hatalari (count_text_errors)

def _sentences(units):
    """birimler -> cumleler (tuple, tirnaklar atilir; . ! ? ile biter, sondaki yarim cumle de)."""
    out, cur = [], []
    for w in units:
        if w == '"':
            continue
        cur.append(w)
        if w in (".", "!", "?"):
            out.append(tuple(cur))
            cur = []
    if cur:
        out.append(tuple(cur))
    return out


def _quotes(units, inside=False):
    """tirnak icindeki sozler (tuple, , . ! ? haric); inside: metin acik bir tirnagin icinde basliyor."""
    out, cur = [], []
    for w in units:
        if w == '"':
            if inside:
                q = tuple(x for x in cur if x not in (",", ".", "!", "?"))
                if q:
                    out.append(q)
            inside, cur = not inside, []
        elif inside:
            cur.append(w)
    return out


def _xax(units):
    """'X and X' / 'X or X' (X 1..DS._XAX_WORDS birim, [a-z']+) -> [(X, bag)]; soldan, ortusmeden, en uzun X -- A'nin
    ifadesinin finditer'i gibi (zincir 'dug and dug and dug' bir kez)."""
    out, i, n = [], 0, len(units)
    while i < n:
        run = 0
        while run < DS._XAX_WORDS and i + run < n and DS._XWORD.fullmatch(units[i + run]):
            run += 1
        m = next((m for m in range(run, 0, -1) if i + m < n and units[i + m] in ("and", "or")
                  and units[i + m + 1:i + 2 * m + 1] == units[i:i + m]), 0)
        if m:
            out.append((" ".join(units[i:i + m]), units[i + m]))
            i += 2 * m + 1
        else:
            i += 1
    return out


def _stock_reference(data, count):
    """valid'in uzun cumleleri (>= _LONG_SENTENCE birim, tirnaksiz; metin), sabit alt kumenin ilk count hikayesi haric."""
    key = ("stock", data["fingerprint"], count)
    if key not in _CACHE:
        tab, v = DS._token_table(data["vocab"]), data["valid"]
        skip = set(data["valid_start"][exam_rows(data)[:count]].tolist())
        out = set()
        for s, L in zip(*(z.tolist() for z in DS._stories(v, data["vocab"].index(DS.EOS_TOKEN)))):
            if s not in skip:
                out.update(" ".join(t) for t in _sentences(DS._units(v[s:s + L].tolist(), tab)) if len(t) >= _LONG_SENTENCE)
        _CACHE[key] = out
    return _CACHE[key]


def _references(data):
    """Sinavdan once bir kez: siklik bandi, valid kalip cumleleri (bellekte) -> ozet, train referansi (text_reference.json,
    build'in okudugu) ile."""
    _frequency_band(data)
    ref = data["text_reference"]
    return dict(sha256=ref["sha256"], words=len(ref["words"]), xax_pairs=len(ref["xax"]),
                xax_legit=sum(c >= XAX_LEGIT_MIN for c in ref["xax"].values()),
                stock_sentences=len(_stock_reference(data, STORY_CONTINUATIONS)))


@torch.no_grad()
def _continue(model, prompts, n, sampled, seed, eos, vocab_size):
    """prompts (<eos> ile baslayan id listeleri) -> (devamlar, ended): en cok n token, ilk <eos>'ta kesilir.  sampled[i]:
    sicaklik 1 ornekleme (tek uretec, tohum seed), degilse acgozlu.  BlockModel onbellekli, butun satirlar tek batch'te;
    obur modeller her adimda diziyi bastan hesaplar, satir parcalari LOGITS_BUDGET'a sigar."""
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    cached = isinstance(model, BlockModel)
    width = max(map(len, prompts))
    if cached:                                          # FineWeb'de istem 4.000 token'a kadar: 128 satir tek batch'te 80 GB'i asti
        d = model.blocks[0].attention.W_query.shape[0]
        per_row = 4 * (2 * model.turns * d * (width + n) + model.heads * width * width)
        step = max(1, min(len(prompts), CACHE_BUDGET // per_row))
    else:
        step = max(1, LOGITS_BUDGET // (vocab_size * (width + n)))
    gens = [g for c in range(0, len(prompts), step)
            for g in _continue_batch(model, prompts[c:c + step], n, sampled[c:c + step], gen, eos, cached)]
    return [g[:g.index(eos)] if eos in g else g for g in gens], [eos in g for g in gens]


def _continue_batch(model, prompts, n, sampled, gen, eos, cached):
    """_continue'nun bir batch'i -> kesilmemis n'lik id listeleri.  Onbellekte istem son token'i haric model.hidden'dan
    gecer (istem boyunca logits tablosu olusmaz), son token ilk adim olur."""
    device = next(model.parameters()).device
    lengths = torch.tensor([len(p) for p in prompts], device=device)
    width = int(lengths.max())
    ids = torch.full((len(prompts), width + n), eos, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p)
    ids, rows = ids.to(device), torch.arange(len(prompts), device=device)
    sampled = torch.tensor(sampled, device=device)
    if cached:
        head = int(width > 1)                        # butun istemler tek token'liysa istem gecisi yok: ilk cagri istemdir
        caches = [AttentionCache(lengths - head, width + n) for _ in range(model.turns)]
        if head:
            model.hidden(ids[:, :width - 1], caches)
        logits = model.logits(ids[rows, lengths - 1][:, None], caches)[:, 0]
    else:
        logits = model.logits(ids[:, :width])[rows, lengths - 1]
    out = []
    for k in range(n):
        z = logits.float()
        token = torch.where(sampled, torch.multinomial(z.softmax(-1), 1, generator=gen)[:, 0], z.argmax(-1))
        out.append(token)
        if k == n - 1 or (k % 16 == 15 and bool((torch.stack(out, 1) == eos).any(1).all())):
            break
        if cached:
            logits = model.logits(token[:, None], caches)[:, 0]
        else:
            ids[rows, lengths + k] = token
            logits = model.logits(ids[:, :width + k + 1])[rows, lengths + k]
    return torch.stack(out, 1).tolist()


def _count(prompts, continuations, ended, tab, words, xax_ref, stock):
    """Metin hatasi sayaclari, devam basina; istem yalniz tekrarin kaynagi."""
    c, ex = collections.Counter(), dict(xax=[], nonword=[])
    for p, g in zip(prompts, continuations):
        pu, both = DS._units(p, tab), DS._units(p + g, tab)
        gu = both[len(pu) - (len(pu) > 0 and both[len(pu) - 1] != pu[-1]):]   # istem kelime ortasinda kesildiyse o kelime devamin
        c["tokens"] += len(g)
        g8 = [tuple(g[i:i + 8]) for i in range(len(g) - 7)]
        c["loop"] += len(set(g8)) < len(g8)
        seen, rep = set(_sentences(pu)), 0
        for s in _sentences(gu):
            if len(s) >= _LONG_SENTENCE:
                c["sentences"] += 1
                rep += s in seen
                c["stock"] += " ".join(s) in stock
            seen.add(s)
        seen_q, rep_q = _quotes(pu), 0
        for q in _quotes(gu, sum(w == '"' for w in pu) % 2 == 1):
            rep_q += len(q) >= _LONG_QUOTE and q in seen_q
            seen_q.append(q)
        found = _xax(gu)
        bad = [x for x in found if xax_ref.get(x, 0) < XAX_LEGIT_MIN]
        alpha = [w for w in gu if w.isalpha()]
        new = [w for w in alpha if w not in words]
        c.update(sentence_repeat=rep, sentence_repeat_stories=rep > 0, quote_repeat=rep_q, quote_repeat_stories=rep_q > 0,
                 xax=len(bad), xax_stories=bool(bad), xax_legit=len(found) - len(bad), words=len(alpha),
                 nonword=len(new), nonword_stories=bool(new))
        ex["xax"] += ["%s %s %s" % (x, j, x) for x, j in bad]
        ex["nonword"] += new
    n, per1k = len(continuations), lambda k, d: round(1000 * c[k] / max(c[d], 1), 4)
    share = lambda k: round(c[k] / max(n, 1), 4)
    return dict(stories=n, tokens=c["tokens"], ended=round(sum(ended) / max(n, 1), 4), loop=share("loop"),
                sentence_repeat=share("sentence_repeat_stories"), sentence_repeat_per1k=per1k("sentence_repeat", "tokens"),
                quote_repeat=share("quote_repeat_stories"), quote_repeat_per1k=per1k("quote_repeat", "tokens"),
                xax=share("xax_stories"), xax_per1k=per1k("xax", "tokens"), xax_legit_per1k=per1k("xax_legit", "tokens"),
                stock=round(c["stock"] / max(c["sentences"], 1), 4), sentences=c["sentences"],
                nonword=share("nonword_stories"), nonword_per1k=per1k("nonword", "words"), words=c["words"],
                examples=dict(xax=ex["xax"][:10], nonword=ex["nonword"][:10]))


@torch.no_grad()
def count_text_errors(model, data, n, count=STORY_CONTINUATIONS, seed=0, real=False):
    """Sabit alt kumenin ilk count hikayesinin ilk yarisindan (story_prompts) devam, en cok n token (pencereye sigacak
    kadar) ya da <eos>: greedy (acgozlu) ve sampled (sicaklik 1, tohum seed); real=True: gercek ikinci yari da; model
    None: yalniz real.  Her biri (hikaye payi, _per1k 1000 token basina olay):
      loop             tekrar eden token 8'lisi olan hikaye
      sentence_repeat  istemde ya da devamda once gecmis uzun cumlenin (>= 5 birim, tirnaksiz) birebir tekrari
      quote_repeat     tekrar eden soz (tirnak ici, >= 3 birim)
      xax              legit olmayan 'X and X' / 'X or X' (train'de XAX_LEGIT_MIN'den az); xax_legit_per1k legitler
      stock            uzun cumlelerin valid'de (bu hikayeler haric) birebir gecen payi (kalip cumle)
      nonword          train'de hic gecmeyen alfabetik kelime; nonword_per1k 1000 kelime basina
    Birim: kucuk harf kelime, tek rakam, tek noktalama (data_simplestories._token_table); iki tokenizer'da ayni tanim."""
    rows = exam_rows(data)[:count]
    prompts, reals = story_prompts(data, rows)
    tab = DS._token_table(data["vocab"])
    words, xax_ref = data["text_reference"]["words"], data["text_reference"]["xax"]
    stock = _stock_reference(data, max(count, STORY_CONTINUATIONS))
    out = {}
    if model is not None:
        n = min(n, data["seq_len"] - max(len(p) for p in prompts))
        assert n > 0, "istem pencereyi dolduruyor"
        gens, ended = _continue(model, prompts + prompts, n, [False] * len(prompts) + [True] * len(prompts), seed,
                                data["vocab"].index(DS.EOS_TOKEN), len(data["vocab"]))
        k = len(prompts)
        out["greedy"] = _count(prompts, gens[:k], ended[:k], tab, words, xax_ref, stock)
        out["sampled"] = _count(prompts, gens[k:], ended[k:], tab, words, xax_ref, stock)
    if real:
        out["real"] = _count(prompts, reals, [True] * len(reals), tab, words, xax_ref, stock)
    return out


# ---- duzeltme molasi (train_y.train_seq correction_break; kullanici, 3 Ekim: "Hepsi onaylı yani ss böyle yapalım")

BREAK_PROMPT_TOKENS = 6  # mola istemi: train hikayesinin ilk bu kadar token'i (kisa istemde dongu daha sik, D_067)
BREAK_TOKENS = 256       # mola basina hikaye basina acgozlu devam
SENTENCE_MAX = 40        # aday cumle en cok bu kadar token
CANDIDATES = 4           # tekrar sinirinda denenen en olasi ilk token (+ eos)


def _sentence_end_table(vocab):
    """Token -> cumle sonu mu: metni (sondaki tirnak / parantez atilinca) . ! ? ile biter ya da satir sonu icerir."""
    tok = DS._tokenizer(vocab)[0]
    return [("\n" in s) or s.rstrip().rstrip("\"')]}”’").endswith((".", "!", "?"))
            for s in (tok.decode([i]) for i in range(len(vocab)))]


def _repeats(context, sentence, tab):
    """sentence, context'te birebir gecmis cumle mi (>= 5 birim, count_text_errors tanimi) ya da sentence'a degen bir
    token 8'lisi context'te zaten var mi (yakin tekrar; 8'li dongu olcusunun karsiligi)."""
    s = _sentences(DS._units(context + sentence, tab))
    if len(s) > 1 and len(s[-1]) >= _LONG_SENTENCE and s[-1] in set(s[:-1]):
        return True
    full, k = context + sentence, len(context)
    old = {tuple(full[j:j + 8]) for j in range(k - 7)}
    return any(tuple(full[j:j + 8]) in old for j in range(max(k - 7, 0), len(full) - 7))


@torch.no_grad()
def _next_logprobs(model, prefixes, eos):
    """Onekler (degisken boy) -> son konumdaki log olasiliklar (cpu).  Onbellekli: onek son token'i haric hidden'dan."""
    dev = next(model.parameters()).device
    out = []
    for c in range(0, len(prefixes), 64):
        part = prefixes[c:c + 64]
        lengths = torch.tensor([len(p) for p in part], device=dev)
        width = int(lengths.max())
        ids = torch.full((len(part), width), eos, dtype=torch.long)
        for i, p in enumerate(part):
            ids[i, :len(p)] = torch.tensor(p)
        ids, rows = ids.to(dev), torch.arange(len(part), device=dev)
        caches = [AttentionCache(lengths - 1, width) for _ in range(model.turns)]
        model.hidden(ids[:, :width - 1], caches)
        out.append(model.logits(ids[rows, lengths - 1][:, None], caches)[:, 0].float().log_softmax(-1).cpu())
    return torch.cat(out)


def correction_break(data, prompts=None, seed=0):
    """Duzeltme molasi (bulgular 34, 36, 37) -> f(step, model) -> (ids, mask, kayit).  Train hikayelerinden (seed, step)
    tohumlu prompts tanesinin ilk BREAK_PROMPT_TOKENS token'i istem; model BREAK_TOKENS token acgozlu yazar.  Her hikayede
    ILK tekrar cumlesinin (_repeats) sinirinda D1: en olasi CANDIDATES ilk token + eos, her aday cumle sonuna kadar
    acgozlu; tekrar olmayanlarin en olasisi hedef (eos: hikaye biter).  Satir = istem + model metni (sinira kadar) +
    hedef cumle; mask yalniz hedef.  Kayit: hikaye, giren (tekrara giren hikaye), duzeltilen, zorlanan (hicbir aday
    tekrarsiz degil), saniye."""
    import time
    from train_y import BREAK_PROMPTS
    prompts = prompts or BREAK_PROMPTS
    vocab, a = data["vocab"], data["train"]
    eos = vocab.index(DS.EOS_TOKEN)
    end, tab = _sentence_end_table(vocab), DS._token_table(vocab)
    heads = data["train_start"][data["train_head"] & (data["train_length"] >= BREAK_PROMPT_TOKENS)]

    def run(step, model):
        t0 = time.time()
        rows = np.random.default_rng([seed, step]).choice(len(heads), prompts, replace=False)
        firsts = [[eos] + a[s:s + BREAK_PROMPT_TOKENS].tolist() for s in heads[rows].tolist()]
        entries = []                                   # (istem + metin sinira kadar, tekrar cumlesi)
        for p, g in zip(firsts, generate(model, firsts, BREAK_TOKENS)):
            g = g[:g.index(eos)] if eos in g else g
            text, s = p[1:] + g, len(p) - 1
            for i in range(len(p) - 1, len(text)):
                if end[text[i]] or i == len(text) - 1:
                    if i + 1 > s and _repeats(text[:s], text[s:i + 1], tab):
                        entries.append(([eos] + text[:s], text[s:i + 1]))
                        break
                    s = i + 1
        ids, targets, forced = [], [], 0
        if entries:
            lp = _next_logprobs(model, [pre for pre, _ in entries], eos)
            top = lp.topk(CANDIDATES, -1).indices.tolist()
            jobs = [(r, t) for r, c in enumerate(top) for t in c if t != eos]
            rolls = generate(model, [entries[r][0] + [t] for r, t in jobs], SENTENCE_MAX - 1)
            roll = {job: x for job, x in zip(jobs, rolls)}
            for r, (pre, _) in enumerate(entries):
                cands = sorted([(float(lp[r, t]), t) for t in top[r]] + [(float(lp[r, eos]), eos)], reverse=True)
                pick = None
                for _, t in cands:
                    if t == eos:
                        pick = [eos]
                        break
                    sent = [t]
                    for x in ([] if end[t] else roll[(r, t)]):
                        if x == eos:
                            break
                        sent.append(x)
                        if end[x]:
                            break
                    if not _repeats(pre[1:], sent, tab):
                        pick = sent
                        break
                if pick is None:
                    forced += 1
                    continue
                ids.append(pre + pick)
                targets.append(len(pick))
        width = max((len(x) for x in ids), default=1)
        out = torch.full((len(ids), width), eos, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.bool)
        for i, (x, n) in enumerate(zip(ids, targets)):
            out[i, :len(x)] = torch.tensor(x)
            mask[i, len(x) - n:len(x)] = True
        return out, mask, dict(step=step, stories=prompts, entered=len(entries), corrected=len(ids), forced=forced,
                               target_tokens=int(sum(targets)), secs=round(time.time() - t0, 1))

    return run
