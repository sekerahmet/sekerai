"""data -- V2 veri: GPT-2 token akisinda cumle / hikaye sinirlari, END, hikaye duzeyinde paket plani, paketli batch.
Belgeler: belge/model_z_temel/20 (tasarim), 21 (arayuz; adlar onayli, kullanici 6 Ekim), 22 (Model Z duzeni).

Cumle siniri V1 make_ss_sentences'in KOPYASI (f4d2a63 kurali + acik tirnakta kesme yok; import yok): sinir token
indeksinde, token ortasina dusmez.  tests_v2 'drive' grubu V1 kelime cumleleriyle birebir esitligi sinar.  Yalniz bosluk
token'larindan olusan cumle V1'de dusuyordu; burada komsusuna katilir (kelimeler ayni, token kaybolmaz).
profile="web" (FineWeb-Edu, belge 47 s2 / 48): acik tirnak bastirmasi yok; genis kisaltma listesi ve tek buyuk harf + "."
(U.S.) kesilmez; yalniz kapanis / bosluk olan cumle oncekine katilir; MAX_SENTENCE_TOKENS'tan uzun cumle ; : , ya da satir
sonundan (yoksa bosluklu token'dan, yoksa sinirda) bolunur.  profile="ss" (varsayilan) bugunku kural, bit duzeyinde.
Parca (belge 47 s3 A): uzun belge cumle sinirinda satira sigan parcalara bolunur; <split>_story_continues.npy (bool) devam
eden parcayi isaretler, build_batch onun son konumuna EOS hedefi koymaz (-100).

Hikaye duzeni (iki model AYNI konum ve hedef; belge 22 §3): EOS(BOS) s_1 END s_2 END ... s_n END
    hedef: BOS -> s_1'in ilk token'i; s_k'nin token'i -> sonraki ya da END; END_k -> s_(k+1)'in ilk token'i ya da EOS.
    layout model_z: END_k'nin yerinde ZTOK (token 0, girdisi E(END); ayri z token'i yok, dizi ayni boy); konum (RoPE):
    transformer hikaye ici sira, model_z mantiksal (BOS 0, s_k'nin i. token'i (k-1)+i, Z_k k; k ve i 1'den).
Drive'a bir kez (kural 9), <out>/:  <split>_sentence_offsets.npy (int64 (N, 2): ham akista [bas, son)), <split>_story_
offsets.npy (int64 H+1: hikayenin ilk cumlesi), <split>_boundaries.json, train_pack_plan_e1.npz, exam_pack_plan.npz,
train_token_counts.npy (token_counts).

    python data.py <simplestories koku (gpt2/, exam_stories.npy)> <cikti klasoru>
"""
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

EOS_ID, END_ID, VOCAB = 50256, 50257, 50258
ROW_LEN, BATCH_ROWS = 2048, 32
CHUNK = 20_000_000          # sinir hesabi bu kadar token'lik parcalarla (hikaye sinirinda)
MAX_SENTENCE_TOKENS = 128   # web profili: daha uzun cumle bolunur (belge 47 s2)
EXAM_STORIES = 1000         # sinav alt kumesi (model_y exam_simplestories.exam_rows ile ayni kural)


class Kind:
    """PackedBatch.kind: token turu.  layout model_z'de END konumu ZTOK (Z_k)."""
    BOS, TOKEN, END, ZTOK, PAD = 0, 1, 2, 3, 4


class TargetKind:
    """PackedBatch.target_kind: hedefin turu (hedefsiz -1)."""
    FIRST, MID, END, EOS = 0, 1, 2, 3


# --- V1 make_ss_sentences kopyasi (cumle siniri; etiket kismi yok)
_SPEECH = frozenset("""said asked whispered shouted replied cried exclaimed called yelled answered added thought wondered
muttered murmured screamed declared announced explained insisted begged pleaded shouts says asks replies cries calls
whispers continued agreed admitted suggested""".split())
_TITLES = frozenset(("Mr", "Mrs", "Ms", "Dr", "St", "Prof", "Sr", "Jr"))
_TITLES_WEB = _TITLES | frozenset("""Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec Inc Ltd Co Corp Mt Gen Gov Sen Rep Rev
Capt Col Lt Sgt Dept Univ Ave Blvd Ph Jr vs v approx ca c cf ed eds Ed Eds etc www Rs Sts al et ibid Ibid op cit
art Art Med Mohd Refs Th
Pa Va Mass Calif Conn Ill Wis Minn Fla Ga Tenn Ky Md Mich Penn Ariz Colo Okla Ore Wash Ala Ark Del Neb Nev""".split())
_NUMABBR_WEB = frozenset("No NO Nos no Fig Figs Vol Vols vol vols Eq Eqs Ch Sec p pp Iss d".split())   # yalniz rakamdan once
_SEQ_WEB = ("e.g.", "i.e.", " e.g.", " i.e.", "(e.g.", "(i.e.", " eds.", " Eds.", " op. cit.", " et al.", " Mohd.")
                                                                   # cok token'li kisaltma: son noktasinda kesme yok
_REF_WORDS = frozenset("""sources source and further reading references reference bibliography works cited notes citations
literature list selected additional suggested resources footnotes endnotes for more information""".split())
_CITE_LINE = re.compile(r"^\s*\([^()\n]{1,80}(?:1[5-9]|20)\d\d[a-z]?\)\s*$")   # (Koppelman, 2004) satiri
_REF_KEYS = frozenset("sources references bibliography cited reading notes citations footnotes endnotes resources".split())
_FUNC_WEB = frozenset("""the a an to of and or in on for with from as &""".split())   # okuyucu listesi; kaydirilmamis belgede
_CHARS = {}                                                          # tokenizer -> token basina karakter sayisi


def _char_len(tok):
    if id(tok) not in _CHARS:
        _CHARS[id(tok)] = np.array([len(tok.decode([i])) for i in range(tok.get_vocab_size())], np.int64)
    return _CHARS[id(tok)]


def _dot_dash(tok):
    """'.-' tek token'inin kimligi (M.-L. bas harfleri); yoksa -1."""
    ids = tok.encode(".-").ids
    return ids[0] if len(ids) == 1 else -1


def _func_ids(tok):
    """_FUNC_WEB kelimelerinin token kimlikleri (bosluklu ve bosluksuz)."""
    key = ("func", id(tok))
    if key not in _CHARS:
        _CHARS[key] = np.array(sorted({t for w in _FUNC_WEB for p in (" ", "") for t in tok.encode(p + w).ids[:1]
                                       if len(tok.encode(p + w).ids) == 1}), np.int64)
    return _CHARS[key]


def _wrapped_docs(x, f, nl_end, eos, tok):
    """Belge (eos ile ayrilan) basina kaydirilmis metin mi: noktalamasiz biten satirlarin (nl_end: o satirin '\n' token'i)
    karakter boyu -- >= 4 satir, ortanca >= 45, %60'i ortancanin +-%25'inde.  -> token basina bool (belgesi kaydirilmis)."""
    doc = np.cumsum(x == eos) - (x == eos)
    out = np.zeros(len(x), bool)
    pos = np.flatnonzero(nl_end)
    if len(pos) < 4:
        return out
    cl = np.r_[0, np.cumsum(_char_len(tok)[np.minimum(x, len(_char_len(tok)) - 1)])]
    nlv = np.maximum.accumulate(np.where(((f & _FLAG["lines"]) != 0) | (x == eos), np.arange(len(x)), -1))
    start = np.r_[0, nlv[:-1] + 1][pos]
    chars = cl[pos] - cl[start]
    for d in np.unique(doc[pos]).tolist():
        c = chars[doc[pos] == d]
        c = c[c >= 20]
        if len(c) >= 4:
            m = float(np.median(c))
            if m >= 45 and np.mean(np.abs(c - m) <= 0.25 * m) >= 0.6:
                out[doc == d] = True
    return out
_ATTRIBUTION = re.compile(r"^\s*(?:The\s+(?:\w+\s+)?\w+|[A-Z][\w']*(?:\s+[A-Z][\w']*)?)\s+(\w+)\b(?!\s*,\s*[\"“])")
_FLAG = dict(end=1, dot=2, digit=4, title=8, space=16, closer=32, lines=64, lower=128, quote=256, speech=512,
             initial=1024, soft=2048, lead=4096, oparen=8192, listnum=16384, bullet=32768, numabbr=65536, upper=131072,
             contp=262144, colon=524288, punctnl=1048576, note=2097152, numlead=4194304, notec=8388608)   # 1024+: yalniz web
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")       # V1 kelimesi (normalize_words, V1 esleme)


def stream_tables(tok, profile="ss"):
    """tokenizer -> (token basina bayraklar, token basina tirnak sayisi).  V1 _tables.  web: genis kisaltma listesi,
    initial (tek buyuk harf), soft (; : , satir sonu), lead (boslukla baslar) bayraklari; tirnak sayisi 0 (acik tirnak
    bastirmasi yok)."""
    assert profile in ("ss", "web"), profile
    web = profile == "web"
    titles = _TITLES_WEB if web else _TITLES
    n = tok.get_vocab_size()
    word = lambda s: (not web or s[:1].isspace() or s.strip()[:1].isupper()   # noqa: E731  (web: 'ed' kelime eki degil)
                      or s.strip() in ("pp", "p", "vol", "v", "al", "cit", "art"))
    text = [tok.decode([i]) for i in range(n)]
    speech = np.zeros(n, dtype=bool)
    speech[[tok.encode(p + w, add_special_tokens=False).ids[0] for w in _SPEECH for p in (" ", "")]] = True
    closers = "\"')]}”’" + ("`" if web else "")
    nl_head = lambda s: s.split("\n")[0].rstrip().rstrip(closers).endswith((".", "!", "?", ":"))  # noqa: E731
    tab = dict(end=np.array([("\n" in s) or s.rstrip().rstrip(closers).endswith((".", "!", "?"))
                             or (web and s in (".[", "?[", "![")) for s in text])          # web: .[4] atif
               & ~np.array([s.strip().endswith(".") and s.strip()[:-1] in titles and word(s) for s in text]),
               dot=np.array([s.strip() == "." for s in text]), digit=np.array([s[:1].isdigit() for s in text]),
               lines=np.array(["\n" in s for s in text]), lower=np.array([s.lstrip()[:1].islower() for s in text]),
               closer=np.array([bool(s) and s == s.lstrip() and not s.strip(closers) for s in text]),
               space=np.array([not s.strip() for s in text]),
               quote=np.array([any(q in s for q in '"“”') for s in text]),
               title=np.array([s.strip() in titles and word(s) for s in text]), speech=speech,
               initial=np.array([web and len(s.strip()) == 1 and s.strip().isupper() for s in text]),
               soft=np.array([web and (any(c in s for c in ";:,") or "\n" in s) for s in text]),
               lead=np.array([web and s[:1].isspace() for s in text]),
               oparen=np.array([web and s.strip() == "(" for s in text]),
               listnum=np.array([web and (bool(re.fullmatch(r"\d{1,3}", s.strip())) or bool(re.fullmatch("[a-z]", s.strip())))
                                 for s in text]),
               bullet=np.array([web and s.strip() in ("-", "*", "\u2022", "\u2013") for s in text]),
               numabbr=np.array([web and s.strip() in _NUMABBR_WEB for s in text]),
               upper=np.array([web and s.lstrip()[:1].isupper() for s in text]),
               contp=np.array([web and s.lstrip()[:1] in tuple(",;:)]") for s in text]),
               colon=np.array([web and s.rstrip().endswith(":") for s in text]),
               punctnl=np.array([web and "\n" in s and nl_head(s) for s in text]),
               note=np.array([web and bool(re.fullmatch("[\\d\\[\\],\u2013-]*[\\d\\[][\\d\\[\\],\u2013-]*", s))
                             for s in text]),
               notec=np.array([web and bool(re.fullmatch("[\\d\\[\\],\u2013-]+", s)) for s in text]),
               numlead=np.array([web and s.strip()[:1].isdigit() for s in text]))
    flags = sum(tab[k].astype(np.uint32) * v for k, v in _FLAG.items()).astype(np.uint32)
    quotes = np.array([sum(s.count(q) for q in '"“”') for s in text], dtype=np.int64)
    return flags, quotes * (not web)


def sentence_starts(x, flags, quotes, tok, eos, profile="ss"):
    """Hikayeler (eos ile biten akis, int64) -> cumle baslangiclari (token indeksi).  V1 sentence_starts, aynen.  web:
    tek harf kisaltmasi (bosluklu harf ya da noktadan sonraki harf) kesilmez; cok baytli kapanis tirnagi (” ’; GPT-2'de
    iki bayt token'i) kapanis sayilir."""
    f = flags[x]
    on = lambda a, name: (a & _FLAG[name]) != 0  # noqa: E731
    fn, fp = np.append(f[1:], flags[eos]), np.insert(f[:-1], 0, flags[eos])
    fpp = np.insert(fp[:-1], 0, flags[eos])
    not_eos = x != eos
    xpp0 = np.r_[eos, eos, x[:-2]]
    ini = on(fp, "initial") & (on(fp, "lead") | on(fpp, "dot") | on(fpp, "lines") | (xpp0 == eos)   # ss'de initial bos
                               | (on(fpp, "bullet") & np.insert(on(fpp, "dot")[:-1], 0, False))     # M.-L.
                               | ((xpp0 == _dot_dash(tok)) & (profile == "web")))
    keep = on(f, "dot") & (on(fn, "digit") | on(fp, "title") | ini)
    web = profile == "web"
    if web:
        shift = lambda a, k, v: np.r_[np.full(k, v, a.dtype), a[:-k]]  # noqa: E731
        fppp, xpp, xppp = shift(fpp, 1, flags[eos]), shift(x, 2, eos), shift(x, 3, eos)
        line0 = lambda fl, xx: on(fl, "lines") | (xx == eos)  # noqa: E731
        keep |= on(f, "dot") & on(fp, "listnum") & (line0(fpp, xpp) | (on(fpp, "bullet") & line0(fppp, xppp)))
        fnn2 = np.append(f[2:], [flags[eos]] * 2)
        keep |= on(f, "dot") & on(fp, "numabbr") & on(fn, "numlead")                  # No. 5, p. 94, d. 1890
        keep |= (on(f, "dot") & on(fp, "numlead") & on(fn, "numlead")                 # 12. 31) -- 999. 999 calls degil
                 & ~(on(fnn2, "lead") & on(fnn2, "lower")))
        keep |= on(f, "dot") & on(fp, "numlead") & on(fpp, "dot") & (on(fppp, "title") | on(fppp, "numabbr"))   # (v. 1.
        idx = np.arange(len(x))                                         # noktadan sonraki ilk not-disi token
        nxt = np.minimum.accumulate(np.where(~on(f, "notec") | ~not_eos, idx, len(x) - 1)[::-1])[::-1]
        after = f[np.minimum(np.r_[nxt[1:], len(x) - 1], len(x) - 1)]
        note_end = (on(fn, "note") & ((on(after, "lead") & on(after, "upper")) | on(after, "lines"))
                    & ~on(fp, "digit") & ~on(fp, "numlead"))                    # 29.156 UT ondalik, dipnot degil
        qstart = np.zeros(len(x), bool)                                 # cok baytli kapanis tirnaginin ilk token'i
        for a, b in (tok.encode(c).ids for c in "”’"):
            qstart[:-1] |= (x[:-1] == a) & (x[1:] == b)
        attached = (~on(fn, "lead") & ~on(fn, "lines") & ~on(fn, "space") & ~on(fn, "closer") & np.r_[not_eos[1:], False]
                    & ~np.r_[qstart[1:], False])
        keep |= on(f, "dot") & attached & ~note_end                     # QC981.8.C5, Ra.One, GOV.UK
        keep &= ~(on(f, "dot") & note_end & ~on(fp, "title") & ~on(fp, "numabbr"))   # self-image.2 The: dipnot
        for seq in {tuple(tok.encode(q).ids) for q in _SEQ_WEB}:
            n = len(seq)
            hit = np.ones(len(x) - n + 1, bool) if len(x) >= n else np.zeros(0, bool)
            for k, t in enumerate(seq):
                hit &= x[k:len(x) - n + 1 + k] == t
            keep[np.flatnonzero(hit) + n - 1] = True
    e = on(f, "end") & ~keep
    grow = (on(f, "space") | (on(f, "closer") & ~on(fp, "space") & ~on(fp, "lines"))) & not_eos
    if web:
        for a, b in (tok.encode(c).ids for c in "\u201d\u2019"):          # (ilk bayt, son bayt) token cifti
            pair = (x[:-1] == a) & (x[1:] == b)
            grow |= np.r_[pair, False] | np.r_[False, pair]
        dot_pre = np.insert(on(fp, "digit") | on(fp, "numlead"), 0, False)[:-1]   # dipnotun noktasindan once rakam yok
        notes = on(f, "notec") & ~on(f, "lead") & ~on(fp, "lines") & ~on(fp, "space") & not_eos
        run0 = np.maximum.accumulate(np.where(~notes, np.arange(len(x)), -1))     # not dizisinin onceki token'i
        grow |= notes & ~dot_pre[np.maximum(run0, 0)]
    while True:
        more = grow[1:] & e[:-1] & ~e[1:]
        if not more.any():
            break
        e[1:] |= more
    go = ~on(fn, "lower")
    if web:
        fnn = np.append(fn[1:], flags[eos])
        punct = on(f, "lines") & ((np.r_[False, e[:-1]] & ~on(fp, "lines")) | on(f, "punctnl") | on(fp, "colon"))
        bare = on(f, "lines") & ~punct                                       # noktalamasiz satir sonu
        idx = np.arange(len(x))
        nl = np.maximum.accumulate(np.where(on(f, "lines") | ~not_eos, idx, -1))
        ls = np.r_[0, nl[:-1] + 1]                                          # satirin ilk token'i
        first = f[np.minimum(ls, len(x) - 1)]
        heading = (idx - ls < 8) & on(first, "upper") & on(fn, "upper")
        listed = on(fn, "bullet") | (on(fn, "listnum") & on(fnn, "dot"))       # sonraki satir madde / numara
        wrapped = _wrapped_docs(x, f, bare & e, eos, tok)
        func = np.isin(x, _func_ids(tok)) & ~wrapped                        # kaydirilmamis belgede islev kelimesi + '\n'
        hold = bare & ((wrapped & ~heading & ~listed) | (np.r_[False, func[:-1]] & ~listed))
        go = ((go & ~(on(fn, "oparen") & on(fnn, "lower")) & ~hold) | punct) & ~on(fn, "contp")
    cut = np.flatnonzero(e & ~np.append(e[1:], False) & go & np.append(not_eos[1:], False) & not_eos)
    count = np.cumsum(quotes[x])                       # acik tirnak: satir / hikaye basindan beri tek sayida tirnak
    last = np.maximum.accumulate(np.where(on(f, "lines") | ~not_eos, np.arange(len(x)), -1))
    base = np.where(last >= 0, count[np.maximum(last, 0)], 0)
    cut = cut[(count[cut] - base[cut]) % 2 == 0]
    first = np.flatnonzero(e & ~np.r_[False, e[:-1]])
    run0 = first[np.searchsorted(first, cut, "right") - 1]
    quoted, broken = np.zeros(len(cut), dtype=bool), np.zeros(len(cut), dtype=bool)
    for d in range(int((cut - run0).max()) + 1 if len(cut) else 0):
        inside = run0 + d <= cut
        g = f[np.minimum(run0 + d, cut)]
        quoted |= inside & on(g, "quote")
        broken |= inside & on(g, "lines")
    talk = cut[quoted & ~broken]
    pad = np.append(f, [flags[eos]] * 9)
    talk = talk[np.any([on(pad[talk + k], "speech") for k in range(1, 9)], axis=0)] if len(talk) else talk
    merged = []
    for i in talk.tolist():
        w = x[i + 1:i + 13].tolist()
        w = w[:w.index(eos)] if eos in w else w
        m = _ATTRIBUTION.match(tok.decode(w))
        if m and m.group(1).lower() in _SPEECH:
            merged.append(i)
    heads = np.flatnonzero(np.r_[True, ~not_eos[:-1]] & not_eos)
    return np.union1d(heads, np.setdiff1d(cut, merged) + 1)


def _chunks(a, eos, size):
    """Akis -> hikaye sinirinda kesilmis parcalarin (bas, son) sinirlari."""
    start = 0
    while start < len(a):
        stop = min(start + size, len(a))
        if stop < len(a):
            stop = start + int(np.flatnonzero(np.asarray(a[start:stop]) == eos)[-1]) + 1
        yield start, stop
        start = stop


def _split_long(x, flags, S, stop, limit):
    """limit'ten uzun cumleleri bol (web): pencerede son soft token'dan sonra, yoksa son lead token'dan once, yoksa
    sinirda; parca en az limit // 4.  -> (S, stop, bolunen cumle sayisi)."""
    long = np.flatnonzero(stop - S > limit)
    if not len(long):
        return S, stop, 0
    add_s, add_t = [], []
    for i in long.tolist():
        s, t = int(S[i]), int(stop[i])
        while t - s > limit:
            w = flags[x[s:s + limit]]
            lo = limit // 4
            soft = np.flatnonzero((w[lo - 1:limit - 1] & _FLAG["soft"]) != 0)
            lead = np.flatnonzero((w[lo:limit] & _FLAG["lead"]) != 0)
            cut = s + lo + int(soft[-1]) if len(soft) else s + lo + int(lead[-1]) if len(lead) else s + limit
            add_s.append(s)
            add_t.append(cut)
            s = cut
        S[i] = s                                                     # son parca yerinde
    S2, t2 = np.r_[S, np.array(add_s, np.int64)], np.r_[stop, np.array(add_t, np.int64)]
    o = np.argsort(S2, kind="stable")
    return S2[o], t2[o], len(long)


def _ref_heading(text):
    """Kaynakca basligi mi: <= 6 kelime, hepsi _REF_WORDS'te, en az biri _REF_KEYS'te (Sources and Further Reading)."""
    w = re.findall(r"[a-z]+", text.lower())
    return 0 < len(w) <= 6 and set(w) <= _REF_WORDS and bool(set(w) & _REF_KEYS)


def _merge_references(x, S, stop, story, tok, flags, small=4):
    """web: <= small token'lik parca ayni satirdaki onceki cumleye (onceki cumle satir sonuyla bitmez), (a) kaynakca
    basligindan sonra ya da (b) madde satirinda (- FB — Fullback. Optional.); (c) atif etiketi satiri '(Ad, Yil)' onceki
    cumleye."""
    n = stop - S
    short = np.flatnonzero(n <= 8)
    heads = [i for i in short.tolist() if _ref_heading(tok.decode(x[S[i]:stop[i]].tolist()))]
    f = flags[x]
    nl = np.maximum.accumulate(np.where(((f & _FLAG["lines"]) != 0) | (x == EOS_ID), np.arange(len(x)), -1))
    drop = np.zeros(len(S), bool)
    ref = np.zeros(len(S), bool)
    for h in heads:
        for i in range(h + 1, len(S)):
            if story[i] != story[h]:
                break
            ref[i] = True
    prev = -1
    for i in range(len(S)):
        if (prev >= 0 and story[i] == story[prev] and n[i] <= 20
                and _CITE_LINE.match(tok.decode(x[S[i]:stop[i]].tolist()))):
            stop[prev] = stop[i]
            drop[i] = True
            continue
        if prev >= 0 and story[i] == story[prev] and n[i] <= small:
            ls = nl[S[i] - 1] + 1 if S[i] > 0 else 0
            joined = (f[stop[prev] - 1] & _FLAG["lines"]) == 0
            item = (f[min(ls, len(x) - 1)] & (_FLAG["bullet"] | _FLAG["listnum"])) != 0
            if joined and ((ref[i] and ref[prev]) or (item and ls <= S[prev])):
                stop[prev] = stop[i]
                drop[i] = True
                continue
        prev = i
    return S[~drop], stop[~drop], story[~drop]


def _boundaries(x, flags, quotes, tok, profile="ss"):
    """Parca (eos ile biten, int64) -> (cumle [bas, son) (N, 2), hikaye basina cumle sayisi, bosluk-yalniz katilan sayisi,
    zorla bolunen cumle sayisi).  Bos hikaye (cumlesiz) duser (V1 gibi).  web: yalniz kapanis / bosluk olan cumle de
    "bos" sayilip komsusuna katilir; uzun cumle bolunur (MAX_SENTENCE_TOKENS)."""
    S = sentence_starts(x, flags, quotes, tok, EOS_ID, profile)
    ends = np.flatnonzero(x == EOS_ID)
    story = np.searchsorted(ends, S)
    stop = np.minimum(np.r_[S[1:], len(x)], ends[np.minimum(story, len(ends) - 1)])
    blank = _FLAG["space"] | (_FLAG["closer"] if profile == "web" else 0)
    word = (flags[x] & blank) == 0                                # bosluk (web: kapanis da) olmayan token
    cw = np.r_[0, np.cumsum(word)]
    empty = cw[stop] - cw[S] == 0
    if profile == "web":                                          # kapanis cok baytli: token token degil, metinden
        for i in np.flatnonzero(~empty & (stop - S <= 4)).tolist():
            empty[i] = not tok.decode(x[S[i]:stop[i]].tolist()).strip().strip("\"')]}”’")
    # bosluk-yalniz cumle: hikayede onceki tutulan cumleye katilir; yoksa sonrakine (bas noktasi geri alinir)
    kept = -1
    for i in range(len(S)) if empty.any() else ():
        if i and story[i] != story[i - 1]:
            kept = -1
        if not empty[i]:
            kept = i
        elif kept >= 0:
            stop[kept] = stop[i]
        elif i + 1 < len(S) and story[i + 1] == story[i]:
            S[i + 1] = S[i]
    keep = ~empty
    S, stop, story = S[keep], stop[keep], story[keep]
    forced = 0
    if profile == "web":
        S, stop, story = _merge_references(x, S, stop, story, tok, flags)
        S, stop, forced = _split_long(x, flags, S, stop, MAX_SENTENCE_TOKENS)
        story = np.searchsorted(ends, S)
    counts = np.bincount(story, minlength=len(ends))
    return np.stack([S, stop], 1), counts[counts > 0], int(empty.sum()), forced


_JOB = {}


def _job_init(tok_path, profile):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    _JOB.update(tok=tok, profile=profile, tables=stream_tables(tok, profile))


def _job(args):
    """Cok surecli sinir: (akis yolu, bas, son) -> (cumleler (akis indeksi), sayimlar, katilan, bolunen)."""
    path, s, t = args
    x = np.asarray(np.load(path, mmap_mode="r")[s:t]).astype(np.int64)
    b, c, m, f = _boundaries(x, *_JOB["tables"], _JOB["tok"], _JOB["profile"])
    return b + s, c, m, f


def build_boundaries(stream_root, out_dir, split, profile="ss", workers=1):
    """<stream_root>/gpt2/<split>.npy -> <out_dir>/<split>_sentence_offsets.npy, _story_offsets.npy, _boundaries.json.
    workers > 1: parcalar (CHUNK, hikaye sinirinda) surec havuzunda; sonuc tek surecle ayni (sira korunur)."""
    from tokenizers import Tokenizer
    t0 = time.time()
    tok_path = os.path.join(stream_root, "gpt2", "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    assert tok.token_to_id("<|endoftext|>") == EOS_ID
    path = os.path.join(stream_root, "gpt2", split + ".npy")
    a = np.load(path, mmap_mode="r")
    jobs = [(path, s, t) for s, t in _chunks(a, EOS_ID, CHUNK)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, _job_init, (tok_path, profile)) as pool:
            out = pool.map(_job, jobs)
    else:
        _job_init(tok_path, profile)
        out = [_job(j) for j in jobs]
    sents, counts = np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])
    merged, forced = sum(o[2] for o in out), sum(o[3] for o in out)
    print("%s: %d token, %d parca, %d surec, %.0f sn" % (split, len(a), len(jobs), workers, time.time() - t0), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, split + "_sentence_offsets.npy"), sents)
    np.save(os.path.join(out_dir, split + "_story_offsets.npy"), np.r_[0, np.cumsum(counts)].astype(np.int64))
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            sha.update(chunk)
    L = sents[:, 1] - sents[:, 0]
    meta = dict(split=split, stream=path, stream_sha256=sha.hexdigest(),
                tokenizer_sha256=hashlib.sha256(open(tok_path, "rb").read()).hexdigest(),
                rule="make_ss_sentences.sentence_starts (V1 f4d2a63 + acik tirnakta kesme yok); bosluk-yalniz cumle komsusuna"
                     + ("; web profili (belge 47 s2), MAX_SENTENCE_TOKENS %d" % MAX_SENTENCE_TOKENS if profile == "web" else ""),
                profile=profile, forced_splits=forced,
                stories=len(counts), sentences=len(sents), tokens_in_sentences=int(L.sum()), merged_blank=merged,
                max_sentence_tokens=int(L.max()), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(out_dir, split + "_boundaries.json"), "w", encoding="utf-8"), indent=1)
    print("%s: %d hikaye, %d cumle, en uzun cumle %d token, katilan bosluk cumlesi %d | %.0f sn" % (
        split, meta["stories"], meta["sentences"], meta["max_sentence_tokens"], merged, time.time() - t0), flush=True)
    return meta


class TokenStories:
    """Ham GPT-2 akisi (mmap) + sinir dosyalari -> hikaye hikaye token cumleleri.  max_sentence_tokens: boundaries.json
    (main yazdiysa train + valid en uzunu; z konum anahtari bundan, belge 22)."""

    def __init__(self, stream_root, data_dir, split):
        self.stream = np.load(os.path.join(stream_root, "gpt2", split + ".npy"), mmap_mode="r")
        self.sent = np.load(os.path.join(data_dir, split + "_sentence_offsets.npy"))
        self.story = np.load(os.path.join(data_dir, split + "_story_offsets.npy"))
        self.meta = json.load(open(os.path.join(data_dir, split + "_boundaries.json"), encoding="utf-8"))
        self.max_sentence_tokens = self.meta.get("max_sentence_tokens_all", self.meta["max_sentence_tokens"])
        self.n = len(self.story) - 1
        cont = os.path.join(data_dir, split + "_story_continues.npy")
        self.continues = np.load(cont) if os.path.exists(cont) else None   # parca devam ediyor (son konumda EOS yok)
        assert self.continues is None or len(self.continues) == self.n, "continues hikaye sayisiyla ayni degil"

    def sentences(self, i):
        """Hikaye i -> cumle token dizileri (int64, END yok)."""
        return [np.asarray(self.stream[s:t]).astype(np.int64) for s, t in self.sent[self.story[i]:self.story[i + 1]]]

    def lengths(self, layout="transformer"):
        """Hikaye basina dizi boyu 1 + sum(L_k + 1) -- iki duzende ayni (belge 22 §3)."""
        assert layout in ("transformer", "model_z")
        L = self.sent[:, 1] - self.sent[:, 0] + 1
        return np.add.reduceat(L, self.story[:-1]) + 1


def pack_plan(lengths, row_len=ROW_LEN, seed=0, epoch=1):
    """Hikaye boylari -> (row_offsets int64 R+1, row_stories int32): satir r = row_stories[row_offsets[r]:row_offsets[r+1]]
    (lengths indeksi).  seed None: karistirma yok (sinav); yoksa default_rng([seed, epoch]) karistirir.  Hikaye bolunmez;
    son acik 16 satirdan ilk sigana (first-fit); dolu satir kapanir."""
    lengths = np.asarray(lengths)
    assert lengths.max() <= row_len, "satira sigmayan hikaye: %d > %d" % (lengths.max(), row_len)
    order = np.arange(len(lengths)) if seed is None else np.random.default_rng([seed, epoch]).permutation(len(lengths))
    rows, free, done = [], [], []                      # acik satirlar: hikaye listesi, kalan yer
    for i in order.tolist():
        n = int(lengths[i])
        for j, f in enumerate(free):
            if f >= n:
                rows[j].append(i)
                free[j] -= n
                break
        else:
            rows.append([i])
            free.append(row_len - n)
            if len(rows) > 16:                          # en eski acik satir kapanir (plan sirasi korunur)
                done.append(rows.pop(0))
                free.pop(0)
    done += rows
    return np.r_[0, np.cumsum([len(r) for r in done])].astype(np.int64), np.array([i for r in done for i in r],
                                                                                   dtype=np.int32)


@dataclass
class PackedBatch:
    """build_batch ciktisi; hepsi (B, T) ve cihazda (story_ids haric).  belge 21 §5."""
    tokens: torch.Tensor          # girdi token'i (END konumunda END_ID, model_z'de 0; dolgu 0)
    kind: torch.Tensor            # Kind
    pos: torch.Tensor             # RoPE konumu (duzene gore)
    doc: torch.Tensor             # satir ici hikaye no (dolgu -1)
    sent: torch.Tensor            # hikaye ici cumle no 0'dan (BOS, dolgu -1); END cumlesine ait
    target: torch.Tensor          # hedef token ya da -100
    target_kind: torch.Tensor     # TargetKind (hedefsiz -1)
    story_ids: torch.Tensor       # (B, S_max) satirdaki hikaye kimlikleri, -1 dolgu
    real_pos: torch.Tensor = None  # yalniz summaries_last duzeninde: gercek hikaye konumu (sutundan turetilemez)


def build_batch(stories, row_stories_list, layout, device="cpu", row_len=ROW_LEN):
    """stories (TokenStories: stream, sent (N, 2), story (H+1)), satir basina hikaye kimlikleri -> PackedBatch.  Vektorel
    (numpy; olculdu: Python dongusunun ~10 kati hizli, tests_v2 'pack' dongulu basvuruyla esitligi sinar)."""
    assert layout in ("transformer", "model_z")
    B = len(row_stories_list)
    per_row = np.array([len(r) for r in row_stories_list])
    sid = np.full((B, per_row.max()), -1, np.int64)
    for r, row in enumerate(row_stories_list):
        sid[r, :len(row)] = row
    h = sid[sid >= 0]                                                    # hikayeler, satir sirasiyla
    hrow, hdoc = np.repeat(np.arange(B), per_row), np.concatenate([np.arange(k) for k in per_row])
    n = stories.story[h + 1] - stories.story[h]                          # hikaye basina cumle
    excl = lambda a: np.r_[0, np.cumsum(a)[:-1]]  # noqa: E731
    first_sent = excl(n)
    si = np.repeat(stories.story[h] - first_sent, n) + np.arange(n.sum())   # cumlenin genel indeksi
    sh = np.repeat(np.arange(len(h)), n)                                 # cumlenin hikayesi (batch ici)
    sk = np.arange(n.sum()) - first_sent[sh]                             # hikaye ici cumle no (0'dan)
    st0, L = stories.sent[si, 0], stories.sent[si, 1] - stories.sent[si, 0]
    slen = np.add.reduceat(L + 1, first_sent) + 1                        # hikaye boyu 1 + sum(L + 1)
    cs = excl(slen)
    start = cs - cs[np.r_[0, np.cumsum(per_row)[:-1]]][hrow]             # hikayenin satirdaki ilk sutunu
    assert (start + slen <= row_len).all(), "satir tasti"
    cl = excl(L + 1)
    s0 = start[sh] + 1 + cl - cl[first_sent[sh]]                         # cumlenin ilk sutunu
    ts = np.repeat(np.arange(len(L)), L)                                 # token'in cumlesi
    j = np.arange(L.sum()) - excl(L)[ts]                                 # cumle ici sira (0'dan)
    trow, tcol = hrow[sh[ts]], s0[ts] + j
    erow, ecol = hrow[sh], s0 + L                                        # END / Z_k
    tokens = np.zeros((B, row_len), np.int64)
    kind = np.full((B, row_len), Kind.PAD, np.int8)
    pos = np.zeros((B, row_len), np.int64)
    doc = np.full((B, row_len), -1, np.int32)
    sent = np.full((B, row_len), -1, np.int32)
    tokens[hrow, start], kind[hrow, start], doc[hrow, start] = EOS_ID, Kind.BOS, hdoc
    tokens[trow, tcol] = np.asarray(stories.stream[st0[ts] + j], dtype=np.int64)
    kind[trow, tcol], doc[trow, tcol], sent[trow, tcol] = Kind.TOKEN, hdoc[sh[ts]], sk[ts]
    tokens[erow, ecol], kind[erow, ecol], doc[erow, ecol], sent[erow, ecol] = END_ID, Kind.END, hdoc[sh], sk
    target = np.full((B, row_len), -100, np.int64)
    target[:, :-1] = tokens[:, 1:]                                       # sonraki token (hikaye icinde)
    last = start + slen - 1
    target[hrow, last] = EOS_ID                                          # son END'in hedefi EOS
    target[kind == Kind.PAD] = -100
    tkind = np.full((B, row_len), -1, np.int8)
    real = kind != Kind.PAD
    tkind[real & (target == END_ID)] = TargetKind.END
    tkind[real & ((kind == Kind.BOS) | (kind == Kind.END))] = TargetKind.FIRST
    tkind[real & (kind == Kind.TOKEN) & (target != END_ID)] = TargetKind.MID
    tkind[hrow, last] = TargetKind.EOS
    cont = getattr(stories, "continues", None)
    if cont is not None:                                                # devam eden parca: belge bitmedi, EOS hedefi yok
        c = np.asarray(cont, dtype=bool)[h]
        target[hrow[c], last[c]] = -100
        tkind[hrow[c], last[c]] = -1
    if layout == "transformer":                                          # hikaye ici sira
        pos[trow, tcol] = tcol - start[sh[ts]]
        pos[erow, ecol] = ecol - start[sh]
    else:                                                                # mantiksal: BOS 0, (k-1)+i, Z_k k
        pos[trow, tcol] = sk[ts] + j + 1
        pos[erow, ecol] = sk + 1
        tokens[erow, ecol], kind[erow, ecol] = 0, Kind.ZTOK
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return PackedBatch(t(tokens), t(kind), t(pos), t(doc), t(sent), t(target), t(tkind), t(sid))


def token_counts(stream_root, out_dir=None, split="train", chunk=CHUNK):
    """<stream_root>/gpt2/<split>.npy -> token basina sayim (int64, uzunluk EOS_ID + 1), chunk'lik parcalarla (bellek: parca
    basina 8 x chunk bayt).  out_dir verilirse <out_dir>/<split>_token_counts.npy (belge 33 s2: teshis araclarinin islev /
    icerik ayrimi; bir kez uretilir)."""
    a = np.load(os.path.join(stream_root, "gpt2", split + ".npy"), mmap_mode="r")
    c = np.zeros(EOS_ID + 1, np.int64)
    for s in range(0, len(a), chunk):
        c += np.bincount(np.asarray(a[s:s + chunk]).astype(np.int64), minlength=EOS_ID + 1)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, split + "_token_counts.npy"), c)
    return c


def main(stream_root, out_dir):
    """Bir kez: train ve valid sinirlari, train paket plani (epok 1), sinav paket plani."""
    t0 = time.time()
    for split in ("valid", "train"):
        build_boundaries(stream_root, out_dir, split)
    metas = {sp: json.load(open(os.path.join(out_dir, sp + "_boundaries.json"), encoding="utf-8")) for sp in ("train", "valid")}
    longest = max(m["max_sentence_tokens"] for m in metas.values())        # en uzun cumle: train + valid (kimlikte longest)
    for sp, m in metas.items():
        json.dump(dict(m, max_sentence_tokens_all=longest), open(os.path.join(out_dir, sp + "_boundaries.json"), "w",
                                                                 encoding="utf-8"), indent=1)
    train = TokenStories(stream_root, out_dir, "train")
    ro, rs = pack_plan(train.lengths(), ROW_LEN, 0, 1)
    fill = train.lengths().sum() / ((len(ro) - 1) * ROW_LEN)
    np.savez(os.path.join(out_dir, "train_pack_plan_e1.npz"), row_offsets=ro, row_stories=rs, seed=0, epoch=1,
             row_len=ROW_LEN)
    print("train paket plani: %d satir, %d adim, doluluk %.3f | %.0f sn" % (len(ro) - 1, -(-(len(ro) - 1) // BATCH_ROWS),
                                                                       fill, time.time() - t0), flush=True)
    valid = TokenStories(stream_root, out_dir, "valid")
    E = np.load(os.path.join(stream_root, "exam_stories.npy"))
    sha = hashlib.sha256(np.ascontiguousarray(E)).hexdigest()
    assert sha == json.load(open(os.path.join(stream_root, "exam_stories.json"), encoding="utf-8"))["sha256"]
    assert valid.n == len(np.load(os.path.join(stream_root, "gpt2", "valid_bytes.npy"))), "valid hikaye sirasi kaydi"
    pick = np.sort(E[np.random.default_rng(0).permutation(len(E))[:EXAM_STORIES]])
    ro, rs = pack_plan(valid.lengths()[pick], ROW_LEN, None)
    np.savez(os.path.join(out_dir, "exam_pack_plan.npz"), row_offsets=ro, row_stories=pick[rs].astype(np.int32), seed=-1,
             epoch=0, row_len=ROW_LEN, exam_set_sha256=sha)
    print("sinav paket plani: %d hikaye, %d satir | %.0f sn" % (len(pick), len(ro) - 1, time.time() - t0), flush=True)
    token_counts(stream_root, out_dir)


if __name__ == "__main__":
    if os.name == "nt":                                 # EcoQoS kapali; yoksa ~10 kat yavas (kullanici, 6 Ekim)
        import ctypes
        from ctypes import wintypes

        class _State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        _s = _State(1, 0x1, 0)
        print("guc kisitlamasi (EcoQoS) kapali:", bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4,
                                                                                ctypes.byref(_s), ctypes.sizeof(_s))))
    main(sys.argv[1], sys.argv[2])
