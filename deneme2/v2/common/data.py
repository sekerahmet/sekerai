"""data -- V2 veri: GPT-2 token akisinda cumle / hikaye sinirlari, END, hikaye duzeyinde paket plani, paketli batch.
Belgeler: belge/model_z_temel/20 (tasarim), 21 (arayuz; adlar onayli, kullanici 6 Ekim), 22 (Model Z duzeni).

Cumle siniri V1 make_ss_sentences'in KOPYASI (f4d2a63 kurali + acik tirnakta kesme yok; import yok): sinir token
indeksinde, token ortasina dusmez.  tests_v2 'drive' grubu V1 kelime cumleleriyle birebir esitligi sinar.  Yalniz bosluk
token'larindan olusan cumle V1'de dusuyordu; burada komsusuna katilir (kelimeler ayni, token kaybolmaz).

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
_ATTRIBUTION = re.compile(r"^\s*(?:The\s+(?:\w+\s+)?\w+|[A-Z][\w']*(?:\s+[A-Z][\w']*)?)\s+(\w+)\b(?!\s*,\s*[\"“])")
_FLAG = dict(end=1, dot=2, digit=4, title=8, space=16, closer=32, lines=64, lower=128, quote=256, speech=512)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")       # V1 kelimesi (normalize_words, V1 esleme)


def stream_tables(tok):
    """tokenizer -> (token basina bayraklar, token basina tirnak sayisi).  V1 _tables."""
    n = tok.get_vocab_size()
    text = [tok.decode([i]) for i in range(n)]
    speech = np.zeros(n, dtype=bool)
    speech[[tok.encode(p + w, add_special_tokens=False).ids[0] for w in _SPEECH for p in (" ", "")]] = True
    closers = "\"')]}”’"
    tab = dict(end=np.array([("\n" in s) or s.rstrip().rstrip(closers).endswith((".", "!", "?")) for s in text])
               & ~np.array([s.strip().endswith(".") and s.strip()[:-1] in _TITLES for s in text]),
               dot=np.array([s.strip() == "." for s in text]), digit=np.array([s[:1].isdigit() for s in text]),
               lines=np.array(["\n" in s for s in text]), lower=np.array([s.lstrip()[:1].islower() for s in text]),
               closer=np.array([bool(s) and s == s.lstrip() and not s.strip(closers) for s in text]),
               space=np.array([not s.strip() for s in text]),
               quote=np.array([any(q in s for q in '"“”') for s in text]),
               title=np.array([s.strip() in _TITLES for s in text]), speech=speech)
    flags = sum(tab[k].astype(np.uint16) * v for k, v in _FLAG.items()).astype(np.uint16)
    quotes = np.array([sum(s.count(q) for q in '"“”') for s in text], dtype=np.int64)
    return flags, quotes


def sentence_starts(x, flags, quotes, tok, eos):
    """Hikayeler (eos ile biten akis, int64) -> cumle baslangiclari (token indeksi).  V1 sentence_starts, aynen."""
    f = flags[x]
    on = lambda a, name: (a & _FLAG[name]) != 0  # noqa: E731
    fn, fp = np.append(f[1:], flags[eos]), np.insert(f[:-1], 0, flags[eos])
    not_eos = x != eos
    e = on(f, "end") & ~(on(f, "dot") & (on(fn, "digit") | on(fp, "title")))
    grow = (on(f, "space") | (on(f, "closer") & ~on(fp, "space") & ~on(fp, "lines"))) & not_eos
    while True:
        more = grow[1:] & e[:-1] & ~e[1:]
        if not more.any():
            break
        e[1:] |= more
    cut = np.flatnonzero(e & ~np.append(e[1:], False) & ~on(fn, "lower") & np.append(not_eos[1:], False) & not_eos)
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


def _boundaries(x, flags, quotes, tok):
    """Parca (eos ile biten, int64) -> (cumle [bas, son) (N, 2), hikaye basina cumle sayisi, bosluk-yalniz katilan sayisi).
    Bos hikaye (cumlesiz) duser (V1 gibi)."""
    S = sentence_starts(x, flags, quotes, tok, EOS_ID)
    ends = np.flatnonzero(x == EOS_ID)
    story = np.searchsorted(ends, S)
    stop = np.minimum(np.r_[S[1:], len(x)], ends[np.minimum(story, len(ends) - 1)])
    word = (flags[x] & _FLAG["space"]) == 0                       # bosluk olmayan token
    cw = np.r_[0, np.cumsum(word)]
    empty = cw[stop] - cw[S] == 0
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
    counts = np.bincount(story, minlength=len(ends))
    return np.stack([S, stop], 1), counts[counts > 0], int(empty.sum())


def build_boundaries(stream_root, out_dir, split):
    """<stream_root>/gpt2/<split>.npy -> <out_dir>/<split>_sentence_offsets.npy, _story_offsets.npy, _boundaries.json."""
    from tokenizers import Tokenizer
    t0 = time.time()
    tok_path = os.path.join(stream_root, "gpt2", "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    assert tok.token_to_id("<|endoftext|>") == EOS_ID
    flags, quotes = stream_tables(tok)
    path = os.path.join(stream_root, "gpt2", split + ".npy")
    a = np.load(path, mmap_mode="r")
    sents, counts, merged = [], [], 0
    for s, t in _chunks(a, EOS_ID, CHUNK):
        b, c, m = _boundaries(np.asarray(a[s:t]).astype(np.int64), flags, quotes, tok)
        sents.append(b + s)
        counts.append(c)
        merged += m
        print("%s: %d / %d token, %.0f sn" % (split, t, len(a), time.time() - t0), flush=True)
    sents, counts = np.concatenate(sents), np.concatenate(counts)
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
                rule="make_ss_sentences.sentence_starts (V1 f4d2a63 + acik tirnakta kesme yok); bosluk-yalniz cumle komsusuna",
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
