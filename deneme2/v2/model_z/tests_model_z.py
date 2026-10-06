"""tests_model_z -- V2-Model Z testleri (CPU; belge 21, 22, 33).  GPU yok, Drive yok.  Meaning / ortak sozluk / open_z
testleri arsivde (arsiv/v2_20261006/model_z/tests_model_z.py; git etiketi v2-before-cleanup-20261006).

    python tests_model_z.py [--only z,z_flat,layout,cache,flex,direct,generate_longest,equiv]
"""
import os
import sys
import dataclasses
import subprocess
import tempfile
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TAG = "v2-before-cleanup-20261006"
SIDE = sys.argv[sys.argv.index("--equiv_side") + 1] if "--equiv_side" in sys.argv else None   # equiv: eski kod klasoru
sys.path.insert(0, HERE if SIDE is None else SIDE)
if os.name == "nt":                     # EcoQoS: yoksa ~10 kat yavas (kullanici, 6 Ekim)
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    _k = ctypes.windll.kernel32
    _k.GetCurrentProcess.restype = wintypes.HANDLE
    _k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _s = _State(1, 0x1, 0)
    _ok = _k.SetProcessInformation(_k.GetCurrentProcess(), 4, ctypes.byref(_s), ctypes.sizeof(_s))
    if SIDE is None:
        print("guc kisitlamasi (EcoQoS) kapali:", bool(_ok), flush=True)
torch.set_num_threads(4)
import numpy as np  # noqa: E402

import sentence_z as SZ  # noqa: E402
from sentence import BOS, PAD, TOKEN, ZTOK, SentenceTransformer, SummaryCache, _dense, model_z_mask  # noqa: E402
import data as D  # noqa: E402  (sentence_z common/'u yola ekledi)

FIRST, MID, END_T, EOS_T = D.TargetKind.FIRST, D.TargetKind.MID, D.TargetKind.END, D.TargetKind.EOS

RESULTS = []
STORIES = [[[3, 4, 5], [6, 7], [8, 9, 10, 11]], [[12], [13, 14, 15, 16, 17]], [[18, 19, 20, 21], [22]]]


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


def keys_and_model(d=16, layers=2, heads=2, longest=8):
    """train.py'nin kurma sirasi: manual_seed -> build_keys (kendi Generator'i) -> manual_seed -> model."""
    torch.manual_seed(0)
    keys = SZ.build_keys(longest, d // 2)
    torch.manual_seed(0)
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


def real_batch(rows, T, stories=STORIES):
    """Gercek common.data.build_batch(layout="model_z"); rows: hikaye indeksleri."""
    return D.build_batch(token_stories(stories), rows, "model_z", row_len=T)


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
    return SimpleNamespace(**f, z_slots=(torch.tensor(slots_r), torch.tensor(slots_c)), z_sentences=(ids, mask),
                           z_flat=mask.nonzero(as_tuple=True))


def z_slot_positions(kind, doc):
    """Basvuru (eski sentence.z_slot_positions): kind, doc (B, T) -> RoPE konumu: BOS 0; Z_k k; cumle k'nin i. token'i
    (k-1)+i.  Dolgu 0."""
    B, T = kind.shape
    idx = torch.arange(T).expand(B, -1)
    new_doc = torch.ones_like(kind, dtype=torch.bool)
    new_doc[:, 1:] = doc[:, 1:] != doc[:, :-1]
    doc_start = torch.cummax(torch.where(new_doc, idx, torch.zeros_like(idx)), 1).values
    z = (kind == ZTOK).long()
    zc = z.cumsum(1)
    nz = zc - torch.gather(zc - z, 1, doc_start)
    summary = (kind == BOS) | (kind == ZTOK)
    last_summary = torch.cummax(torch.where(summary, idx, torch.zeros_like(idx)), 1).values
    pos = torch.where(kind == ZTOK, nz, nz + (idx - last_summary))
    pos = torch.where(kind == BOS, torch.zeros_like(pos), pos)
    return torch.where(kind == PAD, torch.zeros_like(pos), pos)


def ref_hidden(model, tokens, kind, pos, zvec, attn):
    """Basvuru (eski hidden + _embed): zvec (B, T, z) yalniz ZTOK'ta okunur (boolean indeks)."""
    x = model.E(tokens)
    zpos = kind == ZTOK
    if zpos.any():
        x = x.index_put((zpos,), model.z_in(model.z_norm(zvec[zpos])).to(x.dtype))
    for block in model.blocks:
        x = block(x, pos, attn)
    return model.norm(x)


def ids_mask(sents):
    L = max(len(x) for x in sents)
    ids = torch.zeros(len(sents), L, dtype=torch.long)
    mask = torch.zeros(len(sents), L, dtype=torch.bool)
    for i, x in enumerate(sents):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = True
    return ids, mask


def z_reference(keys, Ft, sents):
    """Bagimsiz basvuru: z_pos = sum_t R_t f(t) + R_n f(END), R_t yari ici (yari yari roll(isaret * f, t)); torba
    sum f / sqrt(n)."""
    h = keys["half"]
    out = []
    for x in sents:
        zp = torch.zeros(2 * h)
        for t, v in enumerate(list(x) + [keys["END"]]):
            y = Ft[v] * keys["signs"][t]
            zp += torch.cat([torch.roll(y[:h], t), torch.roll(y[h:], t)])
        out.append(torch.cat([zp, Ft[torch.tensor(x)].sum(0) / len(x) ** 0.5]))
    return torch.stack(out)


def full_logits(model, batch):
    h = model._batch_hidden(batch)
    keep = batch.target >= 0
    return h[keep] @ model.E.weight.T, batch.target[keep]


def t_z():
    """z = basvuru formulu (f = [birim(E[:h]) ; birim(E[h:])]/sqrt2, yari ici R_t, END konumu, torba); E'ye bagli, gradyan
    z yolundan E'ye; geri acma; longest siniri."""
    sents = [x for st in STORIES for x in st]
    keys, model = keys_and_model()
    with torch.no_grad():
        Ft = model.f_table()
        z = model._z(sents, "cpu")
        ref = z_reference(keys, Ft, sents)
    d_ref = (z - ref).abs().max().item()
    check("z = basvuru formulu, (n, 2d) = (n, 32); f_table (V, d) yari birim / sqrt2", d_ref < 1e-5
          and z.shape == (len(sents), 32) and Ft.shape == (D.VOCAB, 16)
          and torch.allclose(Ft[:, :8].norm(dim=1), torch.full((D.VOCAB,), 2 ** -0.5)), "fark %.1e" % d_ref)
    with torch.no_grad():
        model.E.weight[sents[0][0]] += 1.0
        z2 = model._z(sents, "cpu")
        model.E.weight[sents[0][0]] -= 1.0
    model.zero_grad()
    ids, mask = ids_mask(sents)
    SZ.encode_z(keys, model.f_table(), ids, mask, mask.nonzero(as_tuple=True)).square().sum().backward()
    g = model.E.weight.grad
    check("z: E degisince z degisir; gradyan z yolundan E'ye akar", (z2 - z).abs().max() > 1e-3
          and g is not None and g[torch.tensor(sents[0])].abs().sum() > 0)
    g = torch.Generator().manual_seed(0)
    E = torch.randn(D.VOCAB, 512, generator=g)                           # geri acma: genis rastgele tablo (h 256)
    F = torch.cat([E[:, :256] / E[:, :256].norm(dim=1, keepdim=True), E[:, 256:] / E[:, 256:].norm(dim=1, keepdim=True)],
                  1) / 2 ** 0.5
    k512 = SZ.build_keys(8, 256)
    zz = SZ.encode_z(k512, F, ids, mask, mask.nonzero(as_tuple=True))
    back = SZ.decode_z(k512, F, zz, max_len=8)
    check("decode_z (F verilir, SIC): kisa cumlelerde geri acma birebir (V 50.258, h 256)", back == sents,
          "%d / %d" % (sum(b == s for b, s in zip(back, sents)), len(sents)))
    try:
        o = torch.ones(1, 9, dtype=torch.long)
        SZ.encode_z(keys, model.f_table(), o, o.bool(), o.bool().nonzero(as_tuple=True))
        ok = False
    except AssertionError:
        ok = True
    errs = 0
    for fn in (lambda: SentenceTransformer(keys, d=32, layers=1, heads=2), lambda: SZ.encode_z(keys, Ft, ids, mask)):
        try:
            fn()
        except (AssertionError, TypeError):
            errs += 1
    check("z: longest'ten uzun cumle durur (konum anahtari en uzun + 1); d != 2 * half durur; flat'siz encode_z durur",
          ok and len(keys["signs"]) == 9 and errs == 2)


def t_z_flat():
    """Belge 33 s3: batch.z_flat (gercek build_batch) = z_sentences mask'inin nonzero sirasi; encode_z(flat) = basvuru;
    _batch_hidden + geri yayilim nonzero cagirmadan kosar (GPU senkronu yok; yol duzeyinde vekil)."""
    rng = np.random.default_rng(0)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 14))] for _ in range(rng.integers(1, 6))]
               for _ in range(120)]
    rows = [list(range(i, i + 10)) for i in range(0, 120, 10)]
    batch = D.build_batch(token_stories(stories), rows, "model_z", row_len=400)
    ids, mask = batch.z_sentences
    nz = mask.nonzero(as_tuple=True)
    check("z_flat (gercek build_batch, 120 hikaye) = z_sentences mask.nonzero (sira dahil)",
          all(torch.equal(a.long(), b) for a, b in zip(batch.z_flat, nz)) and len(batch.z_flat[0]) == int(mask.sum()))
    keys, model = keys_and_model(longest=13)
    with torch.no_grad():
        Ft = model.f_table()
        z = SZ.encode_z(keys, Ft, ids, mask, batch.z_flat)
        sents = [r[m].tolist() for r, m in zip(ids, mask)]
        ref = z_reference(keys, Ft, sents)
    d = (z - ref).abs().max().item()
    check("encode_z(flat = batch.z_flat) = basvuru formulu (%d cumle)" % len(sents), d < 1e-5, "fark %.1e" % d)
    model.train()
    real = torch.Tensor.nonzero

    def boom(*a, **k):
        raise RuntimeError("nonzero cagrildi")
    torch.Tensor.nonzero = boom
    try:
        model.zero_grad()
        nll, _, _ = model.loss_per_target(batch)
        nll.mean().backward()
        ok = bool(torch.isfinite(nll).all()) and model.E.weight.grad is not None
    except RuntimeError as e:
        ok = False
        print("  ", e, flush=True)
    finally:
        torch.Tensor.nonzero = real
    check("loss_per_target + geri yayilim nonzero cagirmadan kosar (torch.Tensor.nonzero hata firlatirken)", ok)


def t_layout():
    """test_z_layout_equals_short: tek dizi + model_z_mask + z_slot_positions = her cumle ayri kisa dizi (V1 yolu)."""
    keys, model = keys_and_model()
    batch = real_batch([[0, 1], [2]], 40)
    ref = reference_pack([STORIES[:2], STORIES[2:]], 40)
    same = all(torch.equal(getattr(batch, k).long(), getattr(ref, k)) for k in
               ("tokens", "kind", "pos", "doc", "target", "target_kind"))
    tok = batch.kind == TOKEN
    same &= torch.equal(batch.sent[tok].long(), ref.sent[tok]) and all(torch.equal(a, b) for a, b in zip(
        batch.z_slots, ref.z_slots)) and all(torch.equal(a, b) for a, b in zip(batch.z_sentences, ref.z_sentences)) \
        and all(torch.equal(a.long(), b) for a, b in zip(batch.z_flat, ref.z_flat))
    check("build_batch(model_z) = bagimsiz basvuru (token, kind, pos, doc, hedef, hedef turu, z yerleri, cumleleri, "
          "z_flat)", same)
    check("z_slot_positions (basvuru) = build_batch pos (BOS 0, Z_k k, token k-1+i; iki hikaye ayni satirda, dolgu)",
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
                h = ref_hidden(model, tokens, kind, torch.arange(T)[None], zvec, None)
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
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    check("loss_per_target: hedef sayisi ve turleri; dolgulu dense yolda geri yayilim sonlu, butun gradyanlar var",
          len(nll) == len(tg_s) and int((tk == END_T).sum()) == sum(len(st) for st in STORIES)
          and int((tk == EOS_T).sum()) == len(STORIES) and ok)


def t_cache():
    """test_cached_logits_equal_full: SummaryCache, token token = tek dizi tam hesap (hikaye basi, tek token'lik cumle,
    ayni satirda iki hikaye, cumle onbellegi silindikten sonraki ilk token)."""
    keys, model = keys_and_model()
    batch = real_batch([[0, 1], [2]], 40)
    got = []
    with torch.no_grad():
        lg_full, _ = full_logits(model, batch)
        for st in STORIES:
            cache = SummaryCache(model)
            got.append(cache.logits[None])
            for s in st:
                for t in s:
                    got.append(cache.append_token(t)[None])
                got.append(cache.close_sentence(model._z([s], "cpu")[0])[None])
    d = (torch.cat(got) - lg_full).abs().max().item()
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


def ref_batch_hidden(model, batch, kind=None):
    """Basvuru yol (belge 24 s5 B oncesi): zvec (B, T, z) + ref_hidden (boolean indeks); z flat'siz basvuru
    formuluyle.  kind: _embed'e verilen (maske her zaman batch.kind'dan)."""
    B, T = batch.tokens.shape
    zvec = torch.zeros(B, T, model.keys["z"])
    ids, mask = batch.z_sentences
    if len(ids):
        zvec[batch.z_slots] = SZ.encode_z(model.keys, model.f_table(), ids, mask, mask.nonzero(as_tuple=True))
    attn = _dense(model_z_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    return ref_hidden(model, batch.tokens, batch.kind if kind is None else kind, batch.pos, zvec, attn)


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
    """_batch_hidden (z dogrudan Z_k satirlarina; belge 24 s5 B) = basvuru yol (hidden + zvec): cikti ve gradyan, fp32
    CPU, gercek build_batch."""
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
    old = _h_and_grads(model, lambda: ref_batch_hidden(model, batch), w)
    d = _diff(new, old)
    check("_batch_hidden = hidden + zvec: cikti ve butun gradyanlar (E z yolu, z_in, z_norm dahil) <= 1e-6", d <= 1e-6
          and new[1]["z_in.weight"] is not None and new[1]["z_in.weight"].abs().sum() > 0, "en buyuk fark %.1e" % d)
    ids, mask = batch.z_sentences
    r, c = batch.z_slots
    fb, ft = batch.z_flat
    empty = dataclasses.replace(batch, z_sentences=(ids[:0], mask[:0]), z_slots=(r[:0], c[:0]), z_flat=(fb[:0], ft[:0]))
    kind_noz = torch.where(batch.kind == ZTOK, torch.full_like(batch.kind, TOKEN), batch.kind)
    new = _h_and_grads(model, lambda: model._batch_hidden(empty), w)
    old = _h_and_grads(model, lambda: ref_batch_hidden(model, empty, kind_noz), w)
    d = _diff(new, old)
    check("Z'siz batch (ids bos; build_batch bos hikaye uretemez, gercek batch'ten kesildi): = basvuru, z_in gradyani yok",
          d <= 1e-6 and new[1]["z_in.weight"] is None, "en buyuk fark %.1e" % d)


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


# --- eski kodla esdegerlik (belge 33 s5): etiketteki own_vocab yolu = bugunku varsayilan, bit duzeyinde
def _equiv_side(old):
    """Ayni tohum, ayni veri: ilk agirlik, 3 AdamW adiminin kayip ve gradyanlari, son agirlik, z, generate."""
    d, layers, heads = 64, 2, 4
    rng = np.random.default_rng(0)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 14))] for _ in range(rng.integers(1, 6))]
               for _ in range(12)]
    longest = max(len(s) for st in stories for s in st)
    torch.manual_seed(0)
    if old:
        keys = SZ.build_keys(None, longest, identity=False, e_dim=d // 2)
        torch.manual_seed(0)
        model = SentenceTransformer(keys, d, layers, heads, own_vocab=True)
    else:
        keys = SZ.build_keys(longest, d // 2)
        torch.manual_seed(0)
        model = SentenceTransformer(keys, d, layers, heads)
    res = dict(init={k: v.clone() for k, v in model.state_dict().items()}, signs=keys["signs"], nll=[], grads=[])
    batch = real_batch([list(range(0, 4)), list(range(4, 8)), list(range(8, 12))], 160, stories)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        nll, _, _ = model.loss_per_target(batch)
        nll.mean().backward()
        res["nll"].append(nll.detach().clone())
        res["grads"].append({n: p.grad.clone() for n, p in model.named_parameters()})
        opt.step()
    res["final"] = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        res["z"] = model._z([s for st in stories for s in st], "cpu")
    prompts = [st[:1] for st in stories[:4]]
    res["greedy"] = model.generate(prompts, 4, 10)
    res["sample"] = model.generate(prompts, 4, 10, torch.Generator().manual_seed(3))
    return res


def _same(x, y):
    if torch.is_tensor(x):
        return torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_same(i, j) for i, j in zip(x, y))
    return x == y


def t_equiv():
    """Etiket v2-before-cleanup-20261006'daki own_vocab yolu (git show, ayri surec) = bugunku Model Z: ayni tohumda
    agirlik, kayip, gradyan, 3 adim sonrasi agirlik, z, generate (acgozlu + ornekleme) bit duzeyinde."""
    tmp = tempfile.mkdtemp(prefix="tests_model_z_equiv_")
    try:
        for src, dst in (("common/data.py", "common/data.py"), ("model_z/sentence.py", "model_z/sentence.py"),
                         ("model_z/sentence_z.py", "model_z/sentence_z.py")):
            txt = subprocess.run(["git", "-C", REPO, "show", "%s:deneme2/v2/%s" % (TAG, src)], capture_output=True,
                                 check=True).stdout
            os.makedirs(os.path.dirname(os.path.join(tmp, dst)), exist_ok=True)
            with open(os.path.join(tmp, dst), "wb") as f:
                f.write(txt)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("ATLA equiv: git etiketi %s okunamadi (%s)" % (TAG, e), flush=True)
        return
    out = os.path.join(tmp, "old.pt")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--equiv_side", os.path.join(tmp, "model_z"), out],
                       capture_output=True, text=True)
    if p.returncode:
        check("equiv: eski kod sureci kosar", False, p.stderr[-500:])
        return
    a, b = torch.load(out, weights_only=False), _equiv_side(False)
    bad = [k for k in a if not _same(a[k], b[k])]
    check("equiv: etiketteki own_vocab yolu = bugunku Model Z, bit duzeyinde (%s)" % ", ".join(a), not bad,
          "farkli: %s" % bad if bad else "")


TESTS = dict(z=t_z, z_flat=t_z_flat, layout=t_layout, cache=t_cache, flex=t_flex, direct=t_direct,
             generate_longest=t_generate_longest, equiv=t_equiv)

if __name__ == "__main__":
    if SIDE is not None:
        torch.save(_equiv_side(True), sys.argv[sys.argv.index("--equiv_side") + 2])
        sys.exit(0)
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), " | HATA: " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)
