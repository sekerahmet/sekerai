"""tests_model_z -- V2-Model Z testleri (CPU; belge 21, 22).  Sentetik meaning gecici klasore yazilir; GPU yok; Drive
yalniz meaning_valid (SS valid; yoksa atlanir).

    python tests_model_z.py [--only layout,cache,flex,z,direct,meaning,meaning_resume,meaning_keys,meaning_valid,
                            meaning_resume_mid,generate_longest,meaning_d1,meaning_wsd,shared,shared_cache]
"""
import os
import sys
import dataclasses
import math
import tempfile
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.name == "nt":                     # EcoQoS: yoksa ~10 kat yavas (kullanici, 6 Ekim)
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    _k = ctypes.windll.kernel32
    _k.GetCurrentProcess.restype = wintypes.HANDLE
    _k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _s = _State(1, 0x1, 0)
    print("guc kisitlamasi (EcoQoS) kapali:",
          bool(_k.SetProcessInformation(_k.GetCurrentProcess(), 4, ctypes.byref(_s), ctypes.sizeof(_s))), flush=True)
torch.set_num_threads(4)
import numpy as np  # noqa: E402

import sentence_z as SZ  # noqa: E402
from sentence import (BOS, PAD, TOKEN, ZTOK, SentenceTransformer, SummaryCache, _dense, model_z_mask,  # noqa: E402
                      z_slot_positions)
import data as D  # noqa: E402  (sentence_z common/'u yola ekledi)

FIRST, MID, END_T, EOS_T = D.TargetKind.FIRST, D.TargetKind.MID, D.TargetKind.END, D.TargetKind.EOS

TMP = tempfile.mkdtemp(prefix="tests_model_z_")
RESULTS = []
STORIES = [[[3, 4, 5], [6, 7], [8, 9, 10, 11]], [[12], [13, 14, 15, 16, 17]], [[18, 19, 20, 21], [22]]]


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


def keys_and_model(d=16, layers=2, heads=2, longest=8):
    torch.manual_seed(0)
    path = os.path.join(TMP, "meaning.pt")
    if not os.path.exists(path):
        torch.save(dict(state={"source.weight": torch.randn(D.END_ID, 256)}), path)
    keys = SZ.build_keys(path, longest)
    return keys, SentenceTransformer(keys, d=d, layers=layers, heads=heads).eval()


def token_stories(stories):
    """Hikaye -> cumle -> token listeleri -> build_batch'in okudugu nesne (stream, sent (N, 2), story (H+1))."""
    flat, sent, story = [], [], [0]
    for st in stories:
        for s in st:
            sent.append((len(flat), len(flat) + len(s)))
            flat += s
        story.append(len(sent))
    return SimpleNamespace(stream=np.array(flat, np.int64), sent=np.array(sent, np.int64), story=np.array(story))


def real_batch(rows, T):
    """Gercek common.data.build_batch(layout="model_z"); rows: STORIES indeksleri."""
    return D.build_batch(token_stories(STORIES), rows, "model_z", row_len=T)


def reference_pack(rows, T):
    """Bagimsiz basvuru (dongulu): satirlar (hikaye listeleri) -> PackedBatch alanlari, rapor 22 duzeni."""
    B = len(rows)
    f = {k: torch.full((B, T), v, dtype=torch.long) for k, v in
         dict(tokens=0, kind=PAD, pos=0, doc=-1, sent=-1, target=-100, target_kind=-1).items()}
    slots_r, slots_c, zs = [], [], []
    for r, stories in enumerate(rows):
        c = 0
        for di, st in enumerate(stories):
            seq = [(D.EOS_ID, BOS, 0, -1)]                                      # (token, kind, pos, sent)
            for k, s in enumerate(st):
                seq += [(t, TOKEN, k + 1 + i, k) for i, t in enumerate(s)]
                seq += [(0, ZTOK, k + 1, -1)]
            tg = []
            for j, (t, kd, p, sn) in enumerate(seq):
                if kd == TOKEN:
                    nxt = seq[j + 1]
                    tg.append((nxt[0], MID) if nxt[1] == TOKEN else (D.END_ID, END_T))
                else:
                    k = 0 if kd == BOS else sum(1 for x in seq[:j + 1] if x[1] == ZTOK)
                    tg.append((st[k][0], FIRST) if k < len(st) else (D.EOS_ID, EOS_T))
            for j, ((t, kd, p, sn), (tt, tk)) in enumerate(zip(seq, tg)):
                for name, v in (("tokens", t), ("kind", kd), ("pos", p), ("doc", di), ("sent", sn), ("target", tt),
                                ("target_kind", tk)):
                    f[name][r, c + j] = v
                if kd == ZTOK:
                    slots_r.append(r)
                    slots_c.append(c + j)
                    zs.append(st[sum(1 for x in seq[:j + 1] if x[1] == ZTOK) - 1])
            c += len(seq)
        assert c <= T
    Lm = max(len(z) for z in zs)
    ids = torch.tensor([z + [0] * (Lm - len(z)) for z in zs])
    mask = torch.tensor([[j < len(z) for j in range(Lm)] for z in zs])
    return SimpleNamespace(**f, z_slots=(torch.tensor(slots_r), torch.tensor(slots_c)), z_sentences=(ids, mask))


def full_logits(model, batch):
    h = model._batch_hidden(batch)
    keep = batch.target >= 0
    return h[keep] @ model.E.weight.T, batch.target[keep]


def t_z():
    keys, model = keys_and_model()
    sents = [s for st in STORIES for s in st]
    z = model._z(sents, "cpu")
    back = SZ.decode_z(keys, z, max_len=8)
    check("z: [konum ; torba] 1024, geri acma (V 50.258) birebir", z.shape == (len(sents), 1024) and back == sents,
          "%d / %d" % (sum(b == s for b, s in zip(back, sents)), len(sents)))
    try:
        SZ.encode_z(keys, torch.ones(1, 9, dtype=torch.long), torch.ones(1, 9, dtype=torch.bool))
        ok = False
    except AssertionError:
        ok = True
    check("z: longest'ten uzun cumle durur (konum anahtari veriden: en uzun + 1)", ok and len(keys["signs"]) == 9)


def t_layout():
    """test_z_layout_equals_short: tek dizi + model_z_mask + z_slot_positions = her cumle ayri kisa dizi (V1 yolu)."""
    keys, model = keys_and_model()
    batch = real_batch([[0, 1], [2]], 40)
    ref = reference_pack([STORIES[:2], STORIES[2:]], 40)
    same = all(torch.equal(getattr(batch, k).long(), getattr(ref, k)) for k in
               ("tokens", "kind", "pos", "doc", "target", "target_kind"))
    tok = batch.kind == TOKEN
    same &= torch.equal(batch.sent[tok].long(), ref.sent[tok]) and all(torch.equal(a, b) for a, b in zip(
        batch.z_slots, ref.z_slots)) and all(torch.equal(a, b) for a, b in zip(batch.z_sentences, ref.z_sentences))
    check("build_batch(model_z) = bagimsiz basvuru (token, kind, pos, doc, hedef, hedef turu, z yerleri ve cumleleri)",
          same)
    check("z_slot_positions = build_batch pos (BOS 0, Z_k k, token k-1+i; iki hikaye ayni satirda, dolgu)",
          torch.equal(z_slot_positions(batch.kind, batch.doc), batch.pos))
    lens = [1 + sum(len(s) + 1 for s in st) for st in STORIES]
    used = (batch.kind != PAD).sum().item()
    check("Model Z dizisi = transformer akisi boyu", used == sum(lens), "%d / %d" % (used, sum(lens)))
    lg_full, tg_full = full_logits(model, batch)
    lg_s, tg_s = [], []
    zall = model._z([s for st in STORIES for s in st], "cpu")
    zi = 0
    with torch.no_grad():
        for st in STORIES:
            zst = zall[zi:zi + len(st)]
            zi += len(st)
            for k in range(len(st) + 1):                         # ornek k: k onceki z
                w = st[k] if k < len(st) else []
                T = 1 + k + len(w)
                tokens = torch.tensor([[D.EOS_ID] + [0] * k + w])
                kind = torch.tensor([[BOS] + [ZTOK] * k + [TOKEN] * len(w)])
                zvec = torch.zeros(1, T, keys["z"])
                zvec[0, 1:1 + k] = zst[:k]
                h = model.hidden(tokens, kind, torch.arange(T)[None], zvec, None)
                rows = list(range(k, T))
                lg_s.append(h[0, rows] @ model.E.weight.T)
                tg_s += ([w[0]] if w else [D.EOS_ID]) + w[1:] + ([D.END_ID] if w else [])
    lg_s = torch.cat(lg_s)
    d = (lg_full - lg_s).abs().max().item()
    check("test_z_layout_equals_short (gercek build_batch): logit ve hedefler kisa dizilerle ayni",
          torch.equal(tg_full, torch.tensor(tg_s)) and d < 1e-5, "en buyuk fark %.1e, %d hedef" % (d, len(tg_s)))
    model.train()
    nll, _, tk = model.loss_per_target(batch)
    nll.mean().backward()
    finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    check("loss_per_target: hedef sayisi ve turleri; dolgulu dense yolda geri yayilim sonlu", len(nll) == len(tg_s)
          and int((tk == END_T).sum()) == sum(len(st) for st in STORIES) and int((tk == EOS_T).sum()) == len(STORIES)
          and finite)


def t_cache():
    """test_cached_logits_equal_full: SummaryCache, token token = tek dizi tam hesap (hikaye basi, tek token'lik cumle,
    ayni satirda iki hikaye, cumle onbellegi silindikten sonraki ilk token)."""
    keys, model = keys_and_model()
    batch = real_batch([[0, 1], [2]], 40)
    lg_full, _ = full_logits(model, batch)
    got = []
    with torch.no_grad():
        for st in [STORIES[0], STORIES[1], STORIES[2]]:
            cache = SummaryCache(model)
            got.append(cache.logits[None])
            for s in st:
                for t in s:
                    got.append(cache.append_token(t)[None])
                got.append(cache.close_sentence(model._z([s], "cpu")[0])[None])
    got = torch.cat(got)
    d = (got[:, :] - lg_full).abs().max().item()
    check("test_cached_logits_equal_full: onbellekli logit = tam hesap (fp32)", d < 1e-5, "en buyuk fark %.1e" % d)
    out = model.generate([STORIES[0][:1], STORIES[1][:1]], max_sentences=3, max_tokens=6)
    a = model.generate([STORIES[0][:1]], 3, 6, torch.Generator().manual_seed(0))
    b = model.generate([STORIES[0][:1]], 3, 6, torch.Generator().manual_seed(0))
    check("generate: cumle <= max_tokens, en cok max_sentences; ayni tohum ayni metin",
          all(len(gen) <= 3 and all(len(s) <= 6 for s in gen) and len(ended) == len(gen) for gen, ended, _ in out)
          and a == b, str(out[0]))


def t_flex():
    """model_z_mask tek mask_mod: FlexAttention BlockMask (CPU, ileri, fp32, eager) = dense."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        print("ATLA flex: flex_attention yok", flush=True)
        return
    keys, model = keys_and_model()
    batch = real_batch([[0, 1], [2]], 128)
    mm = model_z_mask(batch.kind, batch.doc, batch.sent)
    try:
        bm = create_block_mask(mm, 2, None, 128, 128, device="cpu")
        with torch.no_grad():
            h1 = model._batch_hidden(batch, bm)
    except Exception as e:  # noqa: BLE001
        print("ATLA flex: CPU'da kosmadi (%s)" % str(e).splitlines()[0][:120], flush=True)
        return
    with torch.no_grad():
        h0 = model._batch_hidden(batch)
    keep = batch.kind != PAD
    d = (h1[keep] - h0[keep]).abs().max().item()
    dense = _dense(mm, 2, 128, "cpu")
    check("model_z_mask: BlockMask (flex, CPU ileri) = dense; bos satir yok", d < 1e-4 and bool(dense.any(-1).all()),
          "en buyuk fark %.1e" % d)


def old_batch_hidden(model, batch, kind=None):
    """Eski yol (belge 24 s5 B oncesi): zvec (B, T, z) + hidden (_embed boolean indeks).  kind: _embed'e verilen (maske
    her zaman batch.kind'dan)."""
    B, T = batch.tokens.shape
    zvec = torch.zeros(B, T, model.keys["z"])
    ids, mask = batch.z_sentences
    if len(ids):
        zvec[batch.z_slots] = SZ.encode_z(model.keys, ids, mask)
    attn = _dense(model_z_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    return model.hidden(batch.tokens, batch.kind if kind is None else kind, batch.pos, zvec, attn)


def _h_and_grads(model, fn, w):
    model.zero_grad(set_to_none=True)
    h = fn()
    (h * w).sum().backward()
    return h.detach(), {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}


def _diff(a, b):
    """(h, grad) ciftleri -> en buyuk mutlak fark; grad'in biri None digeri degilse inf."""
    d = (a[0] - b[0]).abs().max().item()
    for n in a[1]:
        ga, gb = a[1][n], b[1][n]
        if (ga is None) != (gb is None):
            return float("inf")
        if ga is not None:
            d = max(d, (ga - gb).abs().max().item())
    return d


def t_direct():
    """_batch_hidden (z dogrudan Z_k satirlarina; belge 24 s5 B) = eski yol (hidden + zvec): cikti ve gradyan, fp32 CPU,
    gercek build_batch."""
    keys, model = keys_and_model()
    model.train()
    with torch.no_grad():                                               # z_norm birim olmasin: agirligi da sinansin
        model.z_norm.weight.mul_(1 + 0.3 * torch.randn_like(model.z_norm.weight))
    batch = real_batch([[0, 1], [2]], 40)
    order = torch.nonzero(batch.kind == ZTOK, as_tuple=True)
    check("z_slots satir sirasiyla = kind == ZTOK sirasi (gercek build_batch)",
          all(torch.equal(a.long(), b) for a, b in zip(batch.z_slots, order)))
    w = torch.randn(2, 40, 16, generator=torch.Generator().manual_seed(1))
    new = _h_and_grads(model, lambda: model._batch_hidden(batch), w)
    old = _h_and_grads(model, lambda: old_batch_hidden(model, batch), w)
    d = _diff(new, old)
    check("_batch_hidden = hidden + zvec: cikti ve butun gradyanlar (z_in, z_norm dahil) <= 1e-6", d <= 1e-6
          and new[1]["z_in.weight"] is not None and new[1]["z_in.weight"].abs().sum() > 0, "en buyuk fark %.1e" % d)
    ids, mask = batch.z_sentences
    r, c = batch.z_slots
    empty = dataclasses.replace(batch, z_sentences=(ids[:0], mask[:0]), z_slots=(r[:0], c[:0]))
    kind_noz = torch.where(batch.kind == ZTOK, torch.full_like(batch.kind, TOKEN), batch.kind)
    new = _h_and_grads(model, lambda: model._batch_hidden(empty), w)
    old = _h_and_grads(model, lambda: old_batch_hidden(model, empty, kind_noz), w)
    d = _diff(new, old)
    check("Z'siz batch (ids bos; build_batch bos hikaye uretemez, gercek batch'ten kesildi): = eski yol, z_in gradyani yok",
          d <= 1e-6 and new[1]["z_in.weight"] is None, "en buyuk fark %.1e" % d)


# --- meaning (train_meaning.py; belge 25)
DRIVE_ROOT, DRIVE_OFFSETS = "G:/Drive'ım/simplestories", "G:/Drive'ım/v2/simplestories_gpt2"
_SYN_TEXT = {10: " sw", 11: "am", 12: " the", 13: " dog", 14: ".", 15: " gl", 16: "owed", 17: " Fl", 18: "ame",
             19: "wing", 20: " sat", 21: "Let", 22: "'s", 23: " 1", 24: "2", 25: " go"}
_SYN_WORDS = [[10, 11], [12], [13], [14], [15, 16], [17, 18, 19], [20], [21, 22], [23, 24], [25]]


def _syn_text():
    text = [" w%d" % i for i in range(D.EOS_ID + 1)]
    for k, v in _SYN_TEXT.items():
        text[k] = v
    return text


def _syn_stories(n_stories, seed):
    """Konulu sentetik hikayeler: her hikaye bir konu grubundan (6 token) ve ortak kelimelerden (_SYN_WORDS) cumle kurar ->
    (stream, sent (N, 2), story (H+1)); hikayeler EOS ile biter (gercek akis gibi)."""
    rng = np.random.default_rng(seed)
    topics = [list(range(100 + 6 * g, 106 + 6 * g)) for g in range(6)]
    stream, sent, story = [], [], [0]
    for _ in range(n_stories):
        topic = topics[rng.integers(len(topics))]
        for _ in range(rng.integers(2, 8)):
            s0 = len(stream)
            for _ in range(rng.integers(2, 9)):
                stream += _SYN_WORDS[rng.integers(len(_SYN_WORDS))] if rng.random() < 0.4 else [int(rng.choice(topic))]
            sent.append((s0, len(stream)))
        story.append(len(sent))
        stream.append(D.EOS_ID)
    return SimpleNamespace(stream=np.array(stream, np.uint16), sent=np.array(sent, np.int64), story=np.array(story))


def _meaning_args(out=None, **kw):
    import train_meaning as TM
    a = TM.parse(["--d", "16", "--batch", "16", "--epochs", "2", "--subsample", "0.05", "--lr", "0.05"]
                 + (["--out", out] if out else []))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _brute_windows(st, s, W, max_tokens, start, end):
    """Bagimsiz basvuru: cumle s'nin penceresi -> (farkli token kumesi, ayni kelime token ciftleri)."""
    h = int(np.searchsorted(st.story, s, "right")) - 1
    lo = st.sent[max(st.story[h], s - W + 1), 0]
    hi = st.sent[s, 1]
    lo = max(lo, hi - max_tokens)
    toks = [int(t) for t in st.stream[lo:hi]]
    words, cur = [], [toks[0]]
    for p, t in zip(toks, toks[1:]):
        if int(start[t]) & int(end[p]):
            cur.append(t)
        else:
            words.append(cur)
            cur = [t]
    words.append(cur)
    pairs = {(x, y) for w in words for x in w for y in w if x != y}
    return set(toks), pairs


def t_meaning():
    """Pencere ve ayni-kelime maskesi = bagimsiz basvuru; forward = acik formul; sentetik egitimde kazanc artar."""
    import train_meaning as TM
    st, text = _syn_stories(60, 0), _syn_text()
    glue = TM.glue_tables(text)
    ok_w = ok_s = True
    n_same = 0
    for mt in (TM.MAX_TOKENS, 7):                                       # 7: pencere kirpma yolu da
        old, TM.MAX_TOKENS = TM.MAX_TOKENS, mt
        try:
            win = TM.Windows(st.stream, st.sent, st.story, 5, torch.device("cpu"), glue)
            rows = torch.arange(win.n)
            ids, present, same = win.batch(rows, same_word=True)
        finally:
            TM.MAX_TOKENS = old
        for s in range(win.n):
            toks, pairs = _brute_windows(st, s, 5, mt, *glue)
            got = set(ids[s][present[s]].tolist())
            ok_w &= got == toks and bool((ids[s][~present[s]] == TM.PAD).all())
            gp = {(int(ids[s, x]), int(ids[s, y])) for x, y in same[s].nonzero().tolist()}
            ok_s &= gp == pairs
            n_same += len(pairs)
    check("meaning: pencere token kumesi = basvuru (hikaye siniri, 5 cumle, MAX_TOKENS kirpma; dolgu PAD)", ok_w)
    check("meaning: ayni-kelime ciftleri = basvuru (glue tablosu; ' sw'+'am', ' Fl'+'ame'+'wing', 'Let'+\"'s\", rakam)",
          ok_s and n_same > 0, "%d cift" % n_same)
    torch.manual_seed(0)
    agent = TM.MeaningAgent(TM.V, 8)
    with torch.no_grad():
        agent.bias.normal_()
    ids, present, same = win.batch(torch.tensor([3, 17, 40]), same_word=True)
    present = present & (torch.rand(present.shape, generator=torch.Generator().manual_seed(1)) < 0.8)
    logp, okv = agent(ids, present, same)
    full = (agent.source.weight @ agent.target.weight.T / 8 ** 0.5 + agent.bias).log_softmax(1)     # log P(j | i)
    err, cnt = 0.0, 0
    for b in range(len(ids)):
        for j in range(ids.shape[1]):
            if not present[b, j]:
                continue
            vs = [i for i in range(ids.shape[1]) if present[b, i] and i != j and not same[b, i, j]]
            if not vs:
                ok_ = not okv[b, j]
            else:
                ref = torch.stack([full[ids[b, i], ids[b, j]] for i in vs]).logsumexp(0) - math.log(len(vs))
                err = max(err, abs(float(ref - logp[b, j])))
                ok_ = bool(okv[b, j])
            cnt += ok_
    check("meaning: forward = acik formul log ort_i P(j | i), oy veren ayni kelimeden degil", err < 1e-5,
          "en buyuk fark %.1e, %d gizli" % (err, cnt))
    a = _meaning_args(epochs=8)                                     # lr 0,05: 50.257'lik paydada az adim
    _, hist = TM._train(a, _syn_stories(120, 0), _syn_stories(20, 1), text)
    g = [h["valid"]["gain"] for h in hist]
    check("meaning: sentetik egitim (konulu hikayeler) -- valid kazanci artar ve > 0 (konu bilgisi ogrenildi)",
          g[-1] > g[0] and g[-1] > 0 and all(math.isfinite(h["train_loss"]) for h in hist), "kazanc %s" % g)


def t_meaning_resume():
    """Kesilip surdurulen (epok 1 + surdur) = kesintisiz (2 epok): agirlik, optimizer, olcu gecmisi birebir."""
    import train_meaning as TM
    st, va, text = _syn_stories(60, 0), _syn_stories(20, 1), _syn_text()
    a_dir, b_dir = os.path.join(TMP, "m_full"), os.path.join(TMP, "m_cut")
    TM._train(_meaning_args(a_dir), st, va, text)
    TM._train(_meaning_args(b_dir), st, va, text, stop_after=1)
    mid = torch.load(os.path.join(b_dir, "checkpoint.pt"), weights_only=False)["epoch"]
    TM._train(_meaning_args(b_dir, resume=1), st, va, text)
    A = torch.load(os.path.join(a_dir, "agent.pt"), weights_only=False)
    B = torch.load(os.path.join(b_dir, "agent.pt"), weights_only=False)
    ca = torch.load(os.path.join(a_dir, "checkpoint.pt"), weights_only=False)
    cb = torch.load(os.path.join(b_dir, "checkpoint.pt"), weights_only=False)
    same_w = all(torch.equal(A["state"][k], B["state"][k]) for k in A["state"])
    same_o = all(torch.equal(x, y) for sa, sb in zip(ca["opt"]["state"].values(), cb["opt"]["state"].values())
                 for x, y in zip(sa.values(), sb.values()))
    hist = [{k: v for k, v in h.items() if k != "seconds"} for h in A["history"]]
    hist_b = [{k: v for k, v in h.items() if k != "seconds"} for h in B["history"]]
    check("meaning: kesilip surdurulen = kesintisiz (agirlik, Adam durumu, gen, olcu gecmisi; birebir)",
          mid == 1 and same_w and same_o and torch.equal(ca["gen"], cb["gen"]) and hist == hist_b)
    try:
        TM._train(_meaning_args(b_dir, resume=1, lr=1e-2), st, va, text)
        ok = False
    except AssertionError:
        ok = True
    check("meaning: farkli ayarla surdurme durur", ok)


def t_meaning_keys():
    """Entegrasyon: train_meaning agent.pt -> sentence_z.build_keys (gercek arayuz) -> SentenceTransformer, gercek
    build_batch."""
    import train_meaning as TM
    out = os.path.join(TMP, "m_keys")
    TM._train(_meaning_args(out, epochs=1), _syn_stories(30, 0), _syn_stories(10, 1), _syn_text())
    path = os.path.join(out, "agent.pt")
    src = torch.load(path, weights_only=False)["state"]["source.weight"]
    keys = SZ.build_keys(path, 8)
    m = keys["F"][:D.END_ID, :src.shape[1]] * 2 ** 0.5                 # F = birim([birim(m) ; birim(kimlik)])
    err = (m - src / src.norm(dim=1, keepdim=True)).abs().max().item()
    check("meaning -> build_keys: source.weight (50.257 x d) okunur, F'nin meaning yarisi = birim(source)",
          src.shape == (D.END_ID, 16) and keys["F"].shape == (D.VOCAB, 512) and err < 1e-5, "fark %.1e" % err)
    torch.manual_seed(0)
    model = SentenceTransformer(keys, d=16, layers=2, heads=2).eval()
    batch = real_batch([[0, 1], [2]], 40)
    with torch.no_grad():
        h = model._batch_hidden(batch)
    sents = [s for st in STORIES for s in st]
    z = model._z(sents, "cpu")
    # geri acma birebirligi burada sinanmaz: sentetik d 16 meaning'de satirlar cok ilisik (ort |cos| ~0,5), SIC karisir;
    # geri acma tesis (belge 22 s1), gercek ajanla olculur
    check("meaning -> build_keys -> gercek build_batch ileri hesap sonlu; z (n, 1024) sonlu",
          bool(torch.isfinite(h).all()) and z.shape == (len(sents), 1024) and bool(torch.isfinite(z).all()))


def t_meaning_valid():
    """Gercek SS valid'in kucuk kismi (Drive): ilk 300 hikaye egitim, sonraki 100 olcu; birkac adim, gercek tokenizer."""
    import train_meaning as TM
    if not os.path.exists(os.path.join(DRIVE_OFFSETS, "valid_sentence_offsets.npy")):
        check("meaning: gercek valid (Drive yok, ATLANDI)", True)
        return
    vs = D.TokenStories(DRIVE_ROOT, DRIVE_OFFSETS, "valid")
    part = lambda a, b: SimpleNamespace(stream=vs.stream, sent=vs.sent[vs.story[a]:vs.story[b]],  # noqa: E731
                                        story=vs.story[a:b + 1] - vs.story[a])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DRIVE_ROOT, "gpt2", "tokenizer.json"))
    text = [tok.decode([i]) for i in range(TM.V)]
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    win = TM.Windows(vs.stream, vs.sent[:vs.story[300]], vs.story[:301], 5, torch.device("cpu"), TM.glue_tables(text))
    ids, present, same = win.batch(torch.arange(win.n), same_word=True)
    sw, am = enc(" swam")
    rows = [b for b in range(win.n) if {sw, am} <= set(ids[b][present[b]].tolist())]
    hit = [bool(same[b, (ids[b] == sw).nonzero()[0, 0], (ids[b] == am).nonzero()[0, 0]]) for b in rows]
    check("meaning (gercek valid): ' sw'+'am' ayni pencerede -> oy veremez; maske dolu", len(rows) > 0 and all(hit)
          and bool(same.any()), "%d pencere, ayni-kelime cifti olan pencere %%%.1f" % (
              len(rows), 100 * same.flatten(1).any(1).float().mean()))
    a = TM.parse(["--d", "32", "--batch", "64", "--epochs", "1", "--lr", "0.05"])     # lr: 100 adimda unigram ogrenilsin
    _, hist = TM._train(a, part(0, 300), part(300, 400), text, enc)
    h = hist[-1]
    check("meaning (gercek valid): 1 epok (300 hikaye, ~106 adim) kayip sonlu, valid kaybi ln V'nin 1 nat alti; olcu ve "
          "goz listesi calisir (kalite degil: yol)", math.isfinite(h["train_loss"]) and h["valid"]["nll"] < math.log(TM.V) - 1
          and h["valid"]["targets"] > 0 and h["valid"]["bands"]["parca"]["targets"] > 0,
          "kayip %.3f, valid %s" % (h["train_loss"], h["valid"]))


class _Crash(Exception):
    pass


def t_meaning_resume_mid():
    """Epok ICI kayit (--checkpoint_minutes, kullanici 6 Ekim: "en fazla 10 dk kayıp"): epok 1 ve epok 2 ortasinda kesilip
    surdurulen = kesintisiz (veri sirasi, seyreltme maskesi / RNG, Adam, kayip toplami, olcu gecmisi birebir)."""
    import train_meaning as TM
    st, va, text = _syn_stories(60, 0), _syn_stories(20, 1), _syn_text()
    old_save = TM._save
    kw = dict(checkpoint_minutes=0)                                    # her adimdan once kayit
    try:
        ref_dir = os.path.join(TMP, "mid_ref")
        TM._train(_meaning_args(ref_dir, **kw), st, va, text)
        A = torch.load(os.path.join(ref_dir, "agent.pt"), weights_only=False)
        ca = torch.load(os.path.join(ref_dir, "checkpoint.pt"), weights_only=False)
        hist = [{k: v for k, v in h.items() if k != "seconds"} for h in A["history"]]
        for cut in ((0, 3 * 16), (1, 5 * 16)):                         # (biten epok, siradaki pencere)
            out = os.path.join(TMP, "mid_%d_%d" % cut)

            def crash(obj, path, cut=cut):
                old_save(obj, path)
                if (obj.get("epoch"), obj.get("pos")) == cut:
                    raise _Crash
            TM._save = crash
            try:
                TM._train(_meaning_args(out, **kw), st, va, text)
                crashed = False
            except _Crash:
                crashed = True
            TM._save = old_save
            pos = torch.load(os.path.join(out, "checkpoint.pt"), weights_only=False)["pos"]
            TM._train(_meaning_args(out, resume=1, checkpoint_minutes=30), st, va, text)   # aralik kimlige girmez
            B = torch.load(os.path.join(out, "agent.pt"), weights_only=False)
            cb = torch.load(os.path.join(out, "checkpoint.pt"), weights_only=False)
            same_w = all(torch.equal(A["state"][k], B["state"][k]) for k in A["state"])
            same_o = all(torch.equal(x, y) for sa, sb in zip(ca["opt"]["state"].values(), cb["opt"]["state"].values())
                         for x, y in zip(sa.values(), sb.values()))
            hist_b = [{k: v for k, v in h.items() if k != "seconds"} for h in B["history"]]
            check("meaning: epok %d ortasinda (pencere %d) kesilip surdurulen = kesintisiz (agirlik, Adam, gen, epok "
                  "kaybi, olcu gecmisi)" % (cut[0] + 1, cut[1]), crashed and pos == cut[1] and same_w and same_o
                  and torch.equal(ca["gen"], cb["gen"]) and hist == hist_b)
    finally:
        TM._save = old_save


def t_generate_longest():
    """B2 (belge 26): generate'te max_tokens > longest -> cumle longest'te kesilir (ended False), encode_z durmaz."""
    keys, model = keys_and_model(longest=8)
    model.END = model.EOS = -1                                         # cumle hic bitmesin: yalniz sinirlar keser
    try:
        gen, ended, _ = model.generate([STORIES[0][:1]], max_sentences=2, max_tokens=20)[0]
        ok = [len(s) for s in gen] == [8, 8] and ended == [False, False]
    except AssertionError:
        ok = False
    gen5, _, _ = model.generate([STORIES[0][:1]], max_sentences=2, max_tokens=5)[0]
    check("generate: max_tokens 20 > longest 8 -> cumleler 8 token, ended False; max_tokens 5 < longest -> 5",
          ok and [len(s) for s in gen5] == [5, 5])


def t_meaning_d1():
    """Belge 27 D1: forward'daki F.embedding yolu = eski bias[ids] / lz[inv] yolu (kayip ve uc gradyan)."""
    import train_meaning as TM
    st, text = _syn_stories(60, 0), _syn_text()
    win = TM.Windows(st.stream, st.sent, st.story, 5, torch.device("cpu"), TM.glue_tables(text))
    ids, present, same = win.batch(torch.arange(64), same_word=True)
    torch.manual_seed(0)
    agent = TM.MeaningAgent(TM.V, 8)
    with torch.no_grad():
        agent.bias.normal_()

    def old_forward():
        u, v = agent.source(ids), agent.target(ids)
        uq, inv = ids.unique(return_inverse=True)
        lz = TM._log_z(agent.source(uq), agent.target.weight, agent.bias)
        k = ids.shape[1]
        voters = present[:, :, None] & present[:, None, :] & ~torch.eye(k, dtype=torch.bool) & ~same
        return TM._mix(u, v, agent.bias[ids], lz[inv], voters)

    out = []
    for fn in (lambda: agent(ids, present, same), old_forward):
        agent.zero_grad()
        logp, ok = fn()
        use = present & ok
        loss = -(logp * use).sum() / use.sum()
        loss.backward()
        out.append([loss.detach()] + [p.grad.clone() for p in (agent.source.weight, agent.target.weight, agent.bias)])
    rel = max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)) for a, b in zip(*out))
    exact = all(torch.equal(a, b) for a, b in zip(*out))
    check("meaning D1: F.embedding yolu = bias[ids] / lz[inv] yolu (kayip, source / target / bias gradyani)", rel <= 1e-6,
          "en buyuk goreli fark %.1e, birebir %s" % (rel, exact))


def t_meaning_wsd():
    """Belge 27 D3: lr dizisi = recipe.wsd_lr (genel adim, epok sinirinda kaymaz); uzatma (1 epok + --epochs 2 --resume 1,
    decay_start'tan) = bastan 2 epok."""
    import train_meaning as TM
    import recipe as R
    st, va, text = _syn_stories(60, 0), _syn_stories(20, 1), _syn_text()
    lrs, Adam = [], torch.optim.Adam

    class Recorder(Adam):
        def step(self, *a, **k):
            lrs.append(self.param_groups[0]["lr"])
            return super().step(*a, **k)
    torch.optim.Adam = Recorder
    try:
        a = _meaning_args()
        TM._train(a, st, va, text)
    finally:
        torch.optim.Adam = Adam
    steps = -(-len(st.sent) // a.batch)
    want = [R.wsd_lr(g, 2 * steps, a.lr, decay=TM.DECAY) for g in range(2 * steps)]
    check("meaning wsd: lr dizisi = recipe.wsd_lr (2 epok x %d adim)" % steps, lrs == want and a.schedule == "wsd",
          "isinma %g, tepe %g, son %g" % (lrs[0], max(lrs), lrs[-1]))
    full, ext = os.path.join(TMP, "wsd_full"), os.path.join(TMP, "wsd_ext")
    TM._train(_meaning_args(full), st, va, text)
    TM._train(_meaning_args(ext, epochs=1), st, va, text)
    dpack = torch.load(os.path.join(ext, "decay_start", "checkpoint.pt"), weights_only=False)
    g_down = dpack["epoch"] * steps + dpack["pos"] // a.batch
    TM._train(_meaning_args(ext, epochs=2, resume=1), st, va, text)
    A = torch.load(os.path.join(full, "agent.pt"), weights_only=False)
    B = torch.load(os.path.join(ext, "agent.pt"), weights_only=False)
    strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731
    same_w = all(torch.equal(A["state"][k], B["state"][k]) for k in A["state"])
    check("meaning wsd: uzatma 1 -> 2 epok (decay_start'tan, adim %d) = bastan 2 epok (agirlik, olcu gecmisi birebir); "
          "eski ciktilar epochs_1/" % g_down, g_down == steps - round(TM.DECAY * steps) and same_w
          and strip(A["history"]) == strip(B["history"]) and os.path.exists(os.path.join(ext, "epochs_1", "agent.pt")))
    try:
        TM._train(_meaning_args(ext, epochs=1, resume=1), st, va, text)
        ok = False
    except AssertionError:
        ok = True
    check("meaning wsd: kisaltma (2 -> 1 epok) durur", ok)


# --- ortak sozluk (shared_vocab, belge 29)
def shared_model(d=264, layers=2, heads=2, longest=8):
    """Ortak sozluklu kucuk model: m yarisi 256 (sentetik meaning), e yarisi d - 256."""
    keys_and_model()                                                    # sentetik meaning dosyasi
    torch.manual_seed(0)
    keys = SZ.build_keys(os.path.join(TMP, "meaning.pt"), longest, identity=False, e_dim=d - 256)
    return keys, SentenceTransformer(keys, d=d, layers=layers, heads=heads, shared_vocab=True).eval()


def _ids_mask(sents):
    L = max(len(x) for x in sents)
    ids = torch.zeros(len(sents), L, dtype=torch.long)
    mask = torch.zeros(len(sents), L, dtype=torch.bool)
    for i, x in enumerate(sents):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = True
    return ids, mask


def _z_reference(keys, Ft, sents):
    """Bagimsiz basvuru: z_pos = sum_t R_t f(t) + R_n f(END), R_t yari ici (blok blok roll(isaret * f, t)); torba
    sum f / sqrt(n)."""
    Dm, D = keys["m_dim"], keys["z_pos"]
    out = []
    for x in sents:
        zp = torch.zeros(D)
        for t, v in enumerate(list(x) + [keys["END"]]):
            y = Ft[v] * keys["signs"][t]
            zp += torch.cat([torch.roll(y[:Dm], t), torch.roll(y[Dm:], t)])
        out.append(torch.cat([zp, Ft[torch.tensor(x)].sum(0) / len(x) ** 0.5]))
    return torch.stack(out)


def t_shared():
    """shared_vocab: False iken eski yol; True iken z = basvuru formulu, m yarisi sabit, e'ye bagli, gradyan e'ye z
    yolundan da akar; E.weight = E(ids); param_groups; ileri / geri gercek build_batch ile sonlu."""
    import recipe as R
    sents = [x for st in STORIES for x in st]
    ids, mask = _ids_mask(sents)
    keys0, old = keys_and_model()
    same_old = isinstance(old.E, torch.nn.Embedding) and old.m_gain is None and torch.equal(
        old._zrows(ids, mask), SZ.encode_z(keys0, ids, mask))
    flat = SZ.encode_z_rows(keys0, keys0["F"], ids, mask)
    d_flat = (flat - SZ.encode_z(keys0, ids, mask)).abs().max().item()
    check("shared_vocab=False: E nn.Embedding, m_gain None, z yolu eski encode_z ile birebir; encode_z_rows (duz token "
          "yolu, F verilince) = encode_z", same_old and d_flat < 1e-5, "fark %.1e" % d_flat)
    keys, model = shared_model()
    Dm, D = keys["m_dim"], keys["z_pos"]
    with torch.no_grad():
        Ft = model.E.f_table()
        z = model._z(sents, "cpu")
        ref = _z_reference(keys, Ft, sents)
    d_ref = (z - ref).abs().max().item()
    check("shared: z = basvuru formulu (f = [m_hat ; e_hat]/sqrt2, R_t yari ici, END konumu, torba)", d_ref < 1e-5
          and z.shape == (len(sents), 2 * D), "fark %.1e, z %s" % (d_ref, tuple(z.shape)))
    with torch.no_grad():
        model.E.e.weight[sents[0][0]] += 1.0
        z2 = model._z(sents, "cpu")
        model.E.e.weight[sents[0][0]] -= 1.0
    m_idx = torch.cat([torch.arange(Dm), D + torch.arange(Dm)])
    e_idx = torch.cat([torch.arange(Dm, D), D + torch.arange(Dm, D)])
    check("shared: e degisince z'nin e yarilari degisir, m yarilari (konum ve torba) birebir ayni",
          torch.equal(z2[:, m_idx], z[:, m_idx]) and (z2[:, e_idx] - z[:, e_idx]).abs().max() > 1e-3)
    model.zero_grad()
    SZ.encode_z_rows(keys, model.E.f_table(), ids, mask).square().sum().backward()
    g_e = model.E.e.weight.grad
    check("shared: gradyan z yolundan e'ye akar (kullanilan satirlar), m_hat buffer (gradyansiz)",
          g_e is not None and g_e[torch.tensor(sents[0])].abs().sum() > 0 and not model.E.m_hat.requires_grad
          and "E.m_hat" not in model.state_dict())
    with torch.no_grad():
        W = model.E.weight
        rows = model.E(torch.arange(len(W)))
    groups = R.param_groups(model)
    nodec = {id(p) for p in groups[1]["params"]}
    check("shared: E.weight (cikis) = E(ids) (girdi); e ve m_gain decay'siz; m_gain = model.m_gain (skaler, 1)",
          torch.equal(W, rows) and id(model.E.e.weight) in nodec and id(model.E.m_gain) in nodec
          and model.m_gain.shape == () and float(model.m_gain) == 1.0)
    model.train()
    model.zero_grad()
    batch = real_batch([[0, 1], [2]], 40)
    nll, _, _ = model.loss_per_target(batch)
    nll.mean().backward()
    grads = [p.grad for p in model.parameters()]
    ok = torch.isfinite(nll).all() and all(g is not None and torch.isfinite(g).all() for g in grads)
    check("shared: gercek build_batch ileri / geri sonlu; m_gain, e, z_in gradyani var", bool(ok)
          and model.m_gain.grad.abs() > 0 and model.z_in.weight.grad.abs().sum() > 0)
    model.eval()
    back = SZ.decode_z_meaning(keys, z, max_len=8)
    check("shared: decode_z_meaning (m yarisindan SIC, sabit m_hat) kisa cumlelerde birebir", back == sents,
          "%d / %d" % (sum(b == x for b, x in zip(back, sents)), len(sents)))


def t_shared_cache():
    """shared_vocab: SummaryCache token token = tek dizi tam hesap; generate ayni tohum ayni metin."""
    keys, model = shared_model()
    batch = real_batch([[0, 1], [2]], 40)
    with torch.no_grad():
        lg_full, _ = full_logits(model, batch)
        got = []
        for st in [STORIES[0], STORIES[1], STORIES[2]]:
            cache = SummaryCache(model)
            got.append(cache.logits[None])
            for x in st:
                for t in x:
                    got.append(cache.append_token(t)[None])
                got.append(cache.close_sentence(model._z([x], "cpu")[0])[None])
    d = (torch.cat(got) - lg_full).abs().max().item()
    check("shared: test_cached_logits_equal_full (fp32)", d < 1e-4, "en buyuk fark %.1e" % d)
    a = model.generate([STORIES[0][:1]], 3, 6, torch.Generator().manual_seed(0))
    b = model.generate([STORIES[0][:1]], 3, 6, torch.Generator().manual_seed(0))
    check("shared: generate calisir, sinirlar tutar, ayni tohum ayni metin", a == b and all(
        len(gen) <= 3 and all(len(x) <= 6 for x in gen) for gen, _, _ in a))


TESTS = dict(z=t_z, layout=t_layout, cache=t_cache, flex=t_flex, direct=t_direct, meaning=t_meaning,
             meaning_resume=t_meaning_resume, meaning_keys=t_meaning_keys, meaning_valid=t_meaning_valid,
             meaning_resume_mid=t_meaning_resume_mid, generate_longest=t_generate_longest, meaning_d1=t_meaning_d1,
             meaning_wsd=t_meaning_wsd, shared=t_shared, shared_cache=t_shared_cache)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), " | HATA: " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)
