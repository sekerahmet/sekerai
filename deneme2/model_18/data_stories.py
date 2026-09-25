# -*- coding: utf-8 -*-
"""data_stories -- TinyStories, SONRAKI KELIME (model_17 veri_t17 + olcme_17'den).

Kullanici, 24 Eylul: "matematik de çok aritmetik bir işlem var ama dil de bağlam
ve ilişki matematik gibi değil. o yüzden görmek istiyorum."

Veri Drive'daki onbellekten (kural 9): model_17'nin 4.000 kelimelik sozlugu
(+ <dolgu> <eos> <bilinmeyen> = 4.003), akis ve pencere AYNEN onunki.
Dosyadaki eski ad <hikaye> == <eos> (to_general_eos()), kimlik 1.

OLCUT  accuracy (sonraki kelime birebir), ce (ppl = e^ce), tani: konuma gore
       accuracy (acc_0_64, acc_64_256, acc_256_512) ve eos_ok.
METIN  generate(): istemden hikaye, GOZLE okunur.

KELIME SEVIYESI, buyuk harf korunmus; nadir kelime <bilinmeyen>.
LISANS  cdla-sharing-1.0.   Eldan & Li, arXiv 2305.07759
"""
from __future__ import annotations

import collections
import hashlib
import math
import os
import re
import time

import numpy as np
import torch

SEPARATOR = "<|endoftext|>"       # dosyadaki ayrac
# GENEL SINIR: her birimin (hikaye, soru) basi ve sonu -- BOS ve EOS ayni
# token.  Kullanici: "hikaye ve ya sinir yerine genel bir terim. model nerde
# durması gerektiğini öğrenmesi adına".
EOS_TOKEN = "<eos>"
OLD_STORY_TOKEN = "<hikaye>"           # ESKI ad: onbellekteki sozluklerde bu yaziyor
UNK_TOKEN = "<bilinmeyen>"
PAD_TOKEN = "<dolgu>"          # hikaye bitince kalan yer; maske ile duser
# Kesme isaretli kisaltma TEK birim (don't, Lily's); noktalama AYRI.
TOKEN_RE = re.compile(r"[A-Za-z]+'[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")

# KIVRIK TIRNAKLAR DUZLESTIRILIYOR.  Korpus ikisini de kullaniyor ve
# kisaltma kurali yalniz duz kesmeyi taniyordu -- OLCULDU (48 MB):
#   kisaltma 93.704:  duz 91.062,  kivrik 2.642  (%2,8 UCE BOLUNUYORDU)
#   "Let's" bazen tek birim, bazen  Let | ' | s
# Ayni kelimenin iki ayri jetonlanmasi.  Tire (- , en, em) DOKUNULMUYOR:
# kurali bozmuyorlar ve anlamlari ayri.
QUOTE_MAP = str.maketrans({"’": "'", "‘": "'",
                       "“": '"', "”": '"'})
NO_SPACE_BEFORE = set(".,!?;:)]}'\"")      # oncesine bosluk KOYMA
NO_SPACE_AFTER = set("([{")                  # sonrasina bosluk KOYMA


def _stories(path, chunk_mb=64, limit_mb=None):
    """Dosyayi parca parca oku, HIKAYE SINIRINDA kes, hikaye hikaye ver."""
    return _stories_chars(path, chunk_mb * 1024 * 1024,
                          None if limit_mb is None else limit_mb * 1024 * 1024)


def _stories_chars(path, chunk, stop_at=None):
    """parca/dur KARAKTER.  En fazla `dur` karakter okunur, yarim kalan son
    hikaye atilir.  Eskiden bolumden artan sayilmiyordu: en_mb=64 fiilen
    128 MB okuyordu (onkayit model_17_TAM1)."""
    rest, n_read = "", 0
    with open(path, encoding="utf-8", errors="replace") as f:
        while stop_at is None or n_read < stop_at:
            p = f.read(chunk if stop_at is None else min(chunk, stop_at - n_read))
            if not p:
                if rest.strip():
                    yield rest.translate(QUOTE_MAP)          # dosya bitti: son hikaye tam
                return
            n_read += len(p)
            p = rest + p
            k = p.rfind(SEPARATOR)
            if k < 0:
                rest = p
                continue
            rest, p = p[k + len(SEPARATOR):], p[:k]
            for h in p.split(SEPARATOR):
                if h.strip():
                    yield h.translate(QUOTE_MAP)


def to_general_eos(vocab):
    """Eski sozlukteki <hikaye>'yi genel <eos>'a cevir.  Kimlik AYNI kalir,
    yani onbellekteki akislar ve eski paketler oldugu gibi gecerli."""
    return [EOS_TOKEN if a == OLD_STORY_TOKEN else a for a in vocab]


def build_vocab(path, limit: int, chunk_mb=64, limit_mb=None, log=print):
    """En sik `en` kelime + <dolgu> + <eos> + <bilinmeyen>.  EGITIMDEN cikar."""
    counts = collections.Counter()
    for h in _stories(path, chunk_mb, limit_mb):
        counts.update(TOKEN_RE.findall(h))
    top = sum(counts.values())
    vocab = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN] + [a for a, _ in counts.most_common(limit)]
    coverage = sum(c for _, c in counts.most_common(limit)) / top
    log("  sozluk  %s farkli kelime gorundu -> en sik %s tutuldu"
        % ("{:,}".format(len(counts)), "{:,}".format(limit)))
    log("          kapsama %%%.3f   disarda %s kelime"
        % (100 * coverage, "{:,}".format(max(0, len(counts) - limit))))
    return vocab, {a: i for i, a in enumerate(vocab)}


def build_stream(path, token_index, chunk_mb=64, limit_mb=None, log=print):
    """Hikayeleri TEK akisa diz, aralarina <eos>.  int16 dizi.

    int16: en buyuk kimlik 4.002 < 32.767, KAYIPSIZ.  Yarim dosya,
    yarim yukleme, Colab'da yarim RAM."""
    assert len(token_index) <= np.iinfo(np.int16).max + 1, "sozluk int16'ya sigmiyor"
    unk_id, eos_id = token_index[UNK_TOKEN], token_index[EOS_TOKEN]
    parts, n = [], 0
    for h in _stories(path, chunk_mb, limit_mb):
        parts.append(np.fromiter((token_index.get(t, unk_id) for t in TOKEN_RE.findall(h)),
                               dtype=np.int16))
        parts.append(np.array([eos_id], dtype=np.int16))
        n += 1
    a = np.concatenate(parts)
    log("  akis    %s hikaye   %s jeton   bilinmeyen %%%.2f"
        % ("{:,}".format(n), "{:,}".format(len(a)),
           100 * float((a == unk_id).mean())))
    return a


def make_windows(a: np.ndarray, T: int, rng=None, story_id=None, pad_id=0):
    """HER HIKAYE = BIR PENCERE, basinda ve sonunda <eos>.  Doner: (P, M).

    Kullanici: *"her hikaye bence 1 pencere olmali yoksa modele dogru tam
    hikaye ogretmemis oluruz"* ve *"bos ve eos gerekli"*.  `story_id`: <eos>'un
    kimligi.

        <eos> w1 w2 ... wL <eos> <dolgu> ...
        bastaki BOS: w1 de tahmin edilir.  sondaki EOS: model bitisi
        ogrenir, uretim durabilir.

    Kalan yer <dolgu>, maskede False.  Tasma YOK.  L + 2 > T olan hikayeler
    ATILIR (bolmek "tam hikaye" ilkesini bozardi).
    OLCULDU, T=256, BOS/EOS'SUZ hali: hikayelerin %89,8'i kaliyor, dolgu
    %33,7.  T=384 -> %95,7 ama dolgu %53,5;  T=512 -> %98,5 ama dolgu %63,7.

    hikaye verilmezse akis DUZ kesilir (maske hep True).
    """
    if story_id is None:
        n = len(a) // T
        P = a[:n * T].reshape(n, T)
        M = np.ones(P.shape, dtype=bool)
    else:
        ends = np.flatnonzero(a == story_id)
        starts = np.concatenate([[0], ends + 1])[:len(ends)]   # her hikayenin basi
        lengths = ends - starts                                   # ayrac haric kelime
        keep = (lengths > 0) & (lengths + 2 <= T)                  # + BOS + EOS
        starts, lengths = starts[keep], lengths[keep]
        P = np.full((len(starts), T), pad_id, dtype=a.dtype)
        M = np.zeros((len(starts), T), dtype=bool)
        P[:, 0] = story_id
        for i, (b, L) in enumerate(zip(starts, lengths)):
            P[i, 1:L + 1] = a[b:b + L]
            P[i, L + 1] = story_id
            M[i, :L + 2] = True
    if rng is not None:
        j = rng.permutation(len(P))
        P, M = P[j], M[j]
    return P, M


def decode(P, vocab, i=0, limit=None, in_quote=False) -> str:
    """Pencereyi OKUNUR metne cevir.  Sayi degil METIN gormek icin.
    in_quote: metin acik bir tirnagin ICINDE basliyor (istemin devami)."""
    d = P[i] if getattr(P, "ndim", 1) > 1 else P      # yigin ya da tek dizi
    d = d[:limit]
    s, open_ = "", False
    for x in d:
        t = vocab[int(x)]
        if t in (EOS_TOKEN, OLD_STORY_TOKEN):
            s += "\n\n---\n\n"
            in_quote = False
            continue
        # Tirnak hem acar hem kapar; sirayi sayarak ayiriyoruz.
        if t in "\"'" and t != "'":
            attached, opens = in_quote, not in_quote
            in_quote = not in_quote
        else:
            attached, opens = t in NO_SPACE_BEFORE, t in NO_SPACE_AFTER
        s += t if (not s or s.endswith(("\n", " ")) or attached or open_) \
            else " " + t
        open_ = opens
    return s.strip()


def fingerprint(*arrays) -> str:
    """Parmak izi -- veri kayarsa kapi yakalasin."""
    h = hashlib.sha256()
    for d in arrays:
        h.update(repr(d).encode() if isinstance(d, list)
                 else np.ascontiguousarray(d).tobytes())
    return h.hexdigest()[:16]


def build(root: str, T: int = 128, limit: int = 4000, limit_mb=None, seed: int = 0,
        aligned: bool = True, log=print):
    """TinyStories -> (ad, EG, DG).

    kok    TinyStories dosyalarinin durdugu klasor (Drive)
    T      pencere uzunlugu.  dizi() dikkati (B,T,T) -- T ile KARESEL.
    en     sozluk kirpmasi (+ <dolgu> + <eos> + <bilinmeyen>)
    en_mb  yalniz ilk N MB (deneme icin).  None -> hepsi.
    hizali pencereler HIKAYE BASINA hizalansin mi.  Duz kesimde
           tahminlerin %40,8'i hikayesinin basini GORMUYORDU (olculdu).

    Uretilen akis onbellege yazilir; ikinci cagri OKUR (kural 9).
    Veri izini train_17 egitim tensorunden hesaplayip pakete yazar."""
    cache_dir = os.path.join(root, "onbellek")
    os.makedirs(cache_dir, exist_ok=True)
    path = {b: os.path.join(root, "TinyStoriesV2-GPT4-%s.txt" % b)
           for b in ("train", "valid")}
    # SURUM onbellek anahtarinda: yapi degisince eski onbellek SESSIZCE
    # kullanilmasin.  en_mb'li akislar v3: eskisi (v2) iki kati okumustu.
    tag = ("tam_n%d_v2" % limit if limit_mb is None
              else "%dmb_n%d_v3" % (limit_mb, limit))

    vocab_file = os.path.join(cache_dir, "sozluk_%s.npy" % tag)
    if os.path.exists(vocab_file):
        vocab = to_general_eos(list(np.load(vocab_file, allow_pickle=True)))
        log("  sozluk  onbellekten  %s birim" % "{:,}".format(len(vocab)))
    else:
        vocab, _ = build_vocab(path["train"], limit, limit_mb=limit_mb, log=log)
        np.save(vocab_file, np.array(vocab, dtype=object))
    token_index = {a: i for i, a in enumerate(vocab)}

    A = {}
    for b in ("train", "valid"):
        p = os.path.join(cache_dir, "akis_%s_%s.npy" % (b, tag))
        if os.path.exists(p):
            A[b] = np.load(p)
            log("  %-6s onbellekten  %s jeton" % (b, "{:,}".format(len(A[b]))))
        else:
            A[b] = build_stream(path[b], token_index, limit_mb=limit_mb if b == "train" else None,
                        log=log)
            np.save(p, A[b])

    rng = np.random.default_rng(seed)
    eos_id = token_index[EOS_TOKEN] if aligned else None
    TRAIN, TRAIN_MASK = make_windows(A["train"], T, rng, eos_id, token_index[PAD_TOKEN])
    HELDOUT, HELDOUT_MASK = make_windows(A["valid"], T, rng, eos_id, token_index[PAD_TOKEN])
    log("  pencere T=%d   %s   egitim %s   dogrulama %s"
        % (T, "HER HIKAYE BIR PENCERE, <eos> ... <eos>" if aligned
           else "duz kesim", "{:,}".format(len(TRAIN)), "{:,}".format(len(HELDOUT))))
    log("  dolgu %%%.1f   etkin is %%%.1f   (L+2 > T olan hikayeler atildi)"
        % (100 * (1 - TRAIN_MASK.mean()), 100 * TRAIN_MASK.mean()))
    return vocab, (TRAIN, TRAIN_MASK), (HELDOUT, HELDOUT_MASK)


# ============================================================
# OLCUT -- olcme_17'den.  Hedef: SONRAKI KELIME, dolgu disinda her konum.
# ============================================================
BANDS = ((0, 64), (64, 256), (256, 512))     # hedefin penceredeki konumu
# URETIM ayari (egitimi degistirmez): son TEKRAR_PENCERESI token'da gecmis KELIMELERIN
# olasiligi TEKRAR_CEZASI'na bolunur; noktalama cezasiz.  1,0 = kapali.
# REL07 sicaklik 0'da "The dog was very happy." dongusune giriyordu (24 Eylul).
REPEAT_PENALTY = 1.0
REPEAT_WINDOW = 20
# URETIM ayari: nucleus (top-p).  Olasiliklari buyukten kucuge toplami TOP_P'ye ulasan kume
# disindaki kelimeler atilir, kalanlardan secilir.  1,0 = kapali.  Holtzman ve ark. 2019.
TOP_P = 1.0
# SAGLIK sondasi, her kosuda AYNI: ilk SONDA_PENCERE tutulan pencere (m.health) ve sabit istemler (dongu).
# Kullanici, 25 Eylul: "nereye bakacağımızı anlamak için ölçüm ... eğitim boyunca takip edebilmek için".
PROBE_WINDOWS = 64
PROBE_PROMPTS = 3        # istem dosyasindan butun kelimeleri sozlukte olan ilk 3 + isim istemi
NAME_PROMPT = ("Once upon a time, there was a girl named Lily. She had a red ball. One day, Lily went to the "
               "park with her ball. At the park, she met a boy named Tom. Tom asked,")


def repeated_bigram(w, a):
    """w (B,T) token, a (B,T-1) sayilan hedef -> (B,T-1) bool: hedef w_(t+1), (w_t, w_(t+1)) ikilisini BU
    hikayede daha once tamamlanmis -- Zoology'nin (2312.04927) "AR hit"i, egitimdeki siklik suzgeci olmadan."""
    B, L = a.shape
    n = int(w.max()) + 1
    order = torch.arange(B * L, device=w.device).view(B, L)
    row_offset = torch.arange(B, device=w.device)[:, None] * n * n
    keys = torch.where(a, row_offset + w[:, :-1] * n + w[:, 1:], -1 - order).reshape(-1)   # sayilmayan eslesmez
    sort_idx = torch.sort(keys, stable=True).indices          # esit anahtarlar konum sirasini korur
    repeated = torch.zeros_like(keys, dtype=torch.bool)
    repeated[sort_idx[1:]] = keys[sort_idx[1:]] == keys[sort_idx[:-1]]     # ilk gecis haric hepsi
    return repeated.view(B, L)


def evaluate(m, W, M, eos, device="cuda", chunk=64):
    """accuracy, ce ve tani.  W, M CPU'da; parca parca tasinir.  Pencereler boya gore siralanip parcalanir,
    her parca kendi en uzun penceresine kirpilir: dolgu hesaplanmaz, sonuc ayni (pencereler birbirini
    gormez, dolgu sagda ve nedensel).

      accuracy       sonraki kelime BIREBIR (en yuksek puan)
      ce             ayni hedeflerde kayip; ppl = e^ce
      acc_a_b        hedef konumu a..b arasinda olanlarda accuracy --
                     C hikayenin neresinde doyuyor
      eos_ok         hikaye BITTIGINDE en yuksek puan EOS mu
      acc_ar         hedef, hikayede daha once gecmis bir ikiliyi tamamliyor (tekrar_ikili): geri cagirma
      acc_other      geri kalan hedefler"""
    correct = total = 0
    ce_sum = 0.0
    band = {b: [0, 0] for b in BANDS}
    eos_counts = [0, 0]
    recall = {"ar": [0, 0], "other": [0, 0]}
    last = (M * torch.arange(1, M.shape[1] + 1)).max(1).values        # son gercek konum + 1
    order = torch.sort(last, stable=True).indices
    with torch.no_grad():
        for i in range(0, W.shape[0], chunk):
            j = order[i:i + chunk]
            L = max(2, int(last[j].max()))
            w = W[j, :L].to(device).long()
            a = M[j, :L].to(device)[:, 1:]
            p = m.scoreboard(w)[:, :-1]
            h = w[:, 1:]
            # cross_entropy'nin ayni: defterde p zaten log p (logsumexp 0), deftersiz ham puan.  (B,T,n) kopyasi yok.
            ce = p.logsumexp(-1) - p.gather(-1, h[..., None])[..., 0]
            d = (p.argmax(-1) == h) & a
            correct += int(d.sum())
            total += int(a.sum())
            ce_sum += float(ce[a].sum())
            k = torch.arange(1, h.shape[1] + 1, device=device)      # hedefin konumu
            for (b0, b1) in BANDS:
                s = a & (k >= b0) & (k < b1)
                band[(b0, b1)][0] += int((d & s).sum())
                band[(b0, b1)][1] += int(s.sum())
            s = a & (h == eos)
            eos_counts[0] += int((d & s).sum())
            eos_counts[1] += int(s.sum())
            r = repeated_bigram(w, a)
            for name, s in (("ar", a & r), ("other", a & ~r)):
                recall[name][0] += int((d & s).sum())
                recall[name][1] += int(s.sum())
    diag = {"acc_%d_%d" % b: (c / n if n else float("nan")) for b, (c, n) in band.items()}
    diag["eos_ok"] = eos_counts[0] / max(eos_counts[1], 1)
    for name, (c, n) in recall.items():
        diag["acc_" + name] = c / n if n else float("nan")
    return {"accuracy": correct / max(total, 1), "ce": ce_sum / max(total, 1), "diag": diag}


def make_metric(TRAIN, HELDOUT, eos, device="cuda", limit=2000, limit_full=None, vocab=None, prompts=()):
    """train'in bekledigi metric(m, "train"|"heldout", full, save) -> dict.
    Egitim sirasinda ilk `limit` pencere.  full: held-out HEPSI; egitim bolmesinden
    `limit_full` pencere (None: held-out kadar) -- 2,6 milyon pencerenin tamami
    bir epokluk ileri gecis olurdu.
    health: held-out'ta her olcumde sabit sondada m.health(); vocab verilirse tam yedekte (save)
    ve sonda (full) sabit istemlerle dongu ve metin de.  f(..., health=h): ayni agirlikla olculmus
    saglik -- yeniden hesaplanmaz, eksikse yalniz dongu eklenir.  Sonucta "timing": eval / health / loop sn."""
    splits = {"train": TRAIN, "heldout": HELDOUT}
    token_index = {a: i for i, a in enumerate(vocab)} if vocab is not None else None
    probe = probe_prompts(prompts, token_index) if token_index is not None else []
    w, mk = HELDOUT[0][:PROBE_WINDOWS], HELDOUT[1][:PROBE_WINDOWS]
    L = int(mk.any(0).nonzero().max()) + 1                    # sondanin en uzun hikayesine kirpilir

    def f(m, side, full=False, save=False, health=None):
        W, M = splits[side]
        if full:
            n = len(W) if side == "heldout" else (limit_full or len(HELDOUT[0]))
        else:
            n = limit
        t = time.time()
        r = evaluate(m, W[:n], M[:n], eos, device)
        r["timing"] = {"eval": time.time() - t}
        if side == "heldout" and hasattr(m, "health"):
            if health is None:
                t = time.time()
                h = m.health(w[:, :L].to(device).long(), mk[:, :L].to(device))
                r["timing"]["health"] = time.time() - t
            else:
                h = dict(health)
            if probe and (save or full) and "distinct4" not in h:
                t = time.time()
                h.update(loop_check(m, vocab, token_index, probe, device=device))
                r["timing"]["loop"] = time.time() - t
            r["health"] = h
        return r

    f.health = True
    return f


def position_table(m, W, M, device="cuda", slices=8, chunk=64, log=print):
    """accuracy'yi KONUMA gore boler: C'nin doydugu yer."""
    T = W.shape[1]
    correct, total = torch.zeros(T - 1), torch.zeros(T - 1)
    with torch.no_grad():
        for i in range(0, W.shape[0], chunk):
            w = W[i:i + chunk].to(device).long()
            a = M[i:i + chunk].to(device)[:, 1:]
            d = (m.scoreboard(w)[:, :-1].argmax(-1) == w[:, 1:]) & a
            correct += d.sum(0).float().cpu()
            total += a.sum(0).float().cpu()
    n = len(correct)
    log("%12s %10s %11s" % ("konum", "accuracy", "hedef"))
    for k in range(slices):
        a, b = k * n // slices, (k + 1) * n // slices
        s = float(total[a:b].sum())
        log("%12s %10.4f %11s" % ("%d-%d" % (a + 1, b), float(correct[a:b].sum()) / max(s, 1),
                                  "{:,}".format(int(s))))
    log("%12s %10.4f %11s" % ("TUMU", float(correct.sum() / total.sum().clamp(min=1)),
                              "{:,}".format(int(total.sum()))))


# ============================================================
# METIN -- olcme_17.devam ve konus.istemler'den.  Sayi degil, modelin YAZDIGI.
# ============================================================
def prompts(root):
    """Makalenin degerlendirme istemleri (Evaluation prompts.yaml)."""
    p = os.path.join(root, "Evaluation prompts.yaml")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        y = f.read()
    return [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n- \|-\n", y)[1:]]


def generate(m, prompt, vocab, token_index, steps=120, device="cuda", temperature=0.0, seed=None,
          banned=(PAD_TOKEN, UNK_TOKEN), penalty=REPEAT_PENALTY, window=REPEAT_WINDOW,
          top_p=TOP_P):
    """Istemin devamini URET; model <eos> uretince durur.
    sicaklik 0: hep en yuksek puan.  yasak: uretilmeyecek token'lar.
    tekrar: son `window` token'da gecmis kelimelerin olasiligi tekrar'a bolunur.
    top_p: sicaklik > 0 iken yalniz olasilik toplami top_p'ye ulasan en olasi kelimelerden secilir."""
    is_word = torch.tensor([any(ch.isalpha() for ch in a) for a in vocab], device=device)
    g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    w = [token_index[EOS_TOKEN]] + [token_index.get(t, token_index[UNK_TOKEN]) for t in TOKEN_RE.findall(prompt.translate(QUOTE_MAP))]
    n_prompt = len(w)
    y = torch.tensor([token_index[t] for t in banned], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(min(steps, m.t_max - len(w))):
            p = m.scoreboard(torch.tensor([w], device=device))[0, -1]
            p = torch.log_softmax(p.index_fill(0, y, -float("inf")), -1)
            if penalty > 1.0:
                last = torch.tensor(w[-window:], device=device)
                last = last[is_word[last]]
                p = p.index_add(0, last.unique(), torch.full((len(last.unique()),),
                                                            -math.log(penalty), device=device))
            if temperature <= 0:
                c = int(p.argmax())
            else:
                q = (p / temperature).softmax(-1)
                if top_p < 1.0:
                    qs, qi = q.sort(descending=True)
                    at = (qs.cumsum(0) - qs) >= top_p            # kumeden once toplam top_p'yi gecti
                    q = q.index_fill(0, qi[at], 0.0)
                c = int(torch.multinomial(q / q.sum(), 1, generator=g))
            w.append(c)
            if c == token_index[EOS_TOKEN]:
                break
    # Istem tirnagi acik biraktiysa devam onun icinde baslar; yoksa kapanis tirnagi acilis sanilir.
    open_quote = sum(vocab[t] == '"' for t in w[1:n_prompt]) % 2 == 1
    return (decode(np.array(w[1:n_prompt]), vocab),
            decode(np.array(w[n_prompt:]), vocab, in_quote=open_quote))


def probe_prompts(prompts, token_index, k=PROBE_PROMPTS):
    """Sabit istemler: kelimelerinin hepsi sozlukte olan ilk k istem + isim istemi.  <bilinmeyen> uretimde
    yasak; istemdeki bir isim <bilinmeyen> ise model onu hic yazamaz (CMNORM t4000 okumasi, 25 Eylul)."""
    clean = [s for s in prompts if all(t in token_index for t in TOKEN_RE.findall(s.translate(QUOTE_MAP)))]
    return clean[:k] + [NAME_PROMPT]


def loop_check(m, vocab, token_index, prompts, steps=60, device="cuda"):
    """Acgozlu uretim (sicaklik 0).  farkli4: farkli 4'lulerin orani (1: hic tekrar yok);
    dongu: tekrar eden 8'lisi olan istemlerin payi; metin: yazilanlar (pakete girer, sonradan gozle okunur)."""
    f4, d8, texts = [], [], []
    for prompt in prompts:
        _, generated = generate(m, prompt, vocab, token_index, steps=steps, device=device)
        t = TOKEN_RE.findall(generated.replace("---", " "))
        g4 = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
        g8 = [tuple(t[i:i + 8]) for i in range(len(t) - 7)]
        f4.append(len(set(g4)) / len(g4) if g4 else 1.0)
        d8.append(len(set(g8)) < len(g8))
        texts.append(generated)
    return {"distinct4": sum(f4) / len(f4), "loop": sum(d8) / len(d8), "texts": texts}
