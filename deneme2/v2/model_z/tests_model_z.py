"""tests_model_z -- V2-Model Z testleri (CPU; belge 21, 22, 35, 43, 44).  GPU yok, Drive yok.  Formullu z testleri
(z, z_flat, direct, generate_longest, formullu onbellek) kaldirildi (belge 44); eski hali git etiketi
v2-before-formula-cleanup-20261007.

    python tests_model_z.py [--only layout,flex,learned,global_,prefill,equiv,mask,summaries_last,gqa,carry,vocab,limit,flex_ranges,
                            gate]
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
TAG = "v2-before-formula-cleanup-20261007"
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

from sentence import (BOS, PAD, TOKEN, ZTOK, SentenceTransformer, SummaryCache, _dense, model_z_mask,  # noqa: E402
                      model_z_read_mask)
import data as D  # noqa: E402  (sentence common/'u yola ekledi)

FIRST, MID, END_T, EOS_T = D.TargetKind.FIRST, D.TargetKind.MID, D.TargetKind.END, D.TargetKind.EOS

RESULTS = []
STORIES = [[[3, 4, 5], [6, 7], [8, 9, 10, 11]], [[12], [13, 14, 15, 16, 17]], [[18, 19, 20, 21], [22]]]


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


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
            c += len(seq)
        assert c <= T
    return SimpleNamespace(**f)


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


def full_logits(model, batch):
    h = model._batch_hidden(batch)
    keep = batch.target >= 0
    return h[keep] @ model.E.weight.T, batch.target[keep]


def learned_model(d=32, layers=2, heads=2, global_layers=0):
    torch.manual_seed(0)
    return SentenceTransformer(d=d, layers=layers, heads=heads, global_layers=global_layers).eval()


def t_layout():
    """build_batch(model_z) = bagimsiz basvuru (duzen, mantiksal konum); dizi transformer akisi boyunda;
    loss_per_target hedef sayilari ve dolgulu dense yolda geri yayilim."""
    batch = real_batch([[0, 1], [2]], 40)
    ref = reference_pack([STORIES[:2], STORIES[2:]], 40)
    same = all(torch.equal(getattr(batch, k).long(), getattr(ref, k)) for k in
               ("tokens", "kind", "pos", "doc", "target", "target_kind"))
    tok = batch.kind == TOKEN
    same &= torch.equal(batch.sent[tok].long(), ref.sent[tok])
    check("build_batch(model_z) = bagimsiz basvuru (token, kind, pos, doc, hedef, hedef turu)", same)
    check("z_slot_positions (basvuru) = build_batch pos (BOS 0, Z_k k, token k-1+i; iki hikaye ayni satirda, dolgu)",
          torch.equal(z_slot_positions(batch.kind, batch.doc), batch.pos))
    lens = [1 + sum(len(s) + 1 for s in st) for st in STORIES]
    used = (batch.kind != PAD).sum().item()
    check("Model Z dizisi = transformer akisi boyu", used == sum(lens), "%d / %d" % (used, sum(lens)))
    model = learned_model(d=16)
    model.train()
    nll, _, tk = model.loss_per_target(batch)
    nll.mean().backward()
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    check("loss_per_target: hedef sayisi ve turleri; dolgulu dense yolda geri yayilim sonlu, butun gradyanlar var",
          len(nll) == int((batch.target >= 0).sum()) and int((tk == END_T).sum()) == sum(len(st) for st in STORIES)
          and int((tk == EOS_T).sum()) == len(STORIES) and ok)


def t_flex():
    """model_z_mask (okumasiz; z_ablate read_off) tek mask_mod: FlexAttention BlockMask (CPU, ileri, fp32) = dense."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        print("ATLA flex: flex_attention yok", flush=True)
        return
    model = learned_model(d=16)
    batch = real_batch([[0, 1], [2]], 128)
    mm = model_z_mask(batch.kind, batch.doc, batch.sent)
    dense = _dense(mm, 2, 128, "cpu")
    try:
        bm = create_block_mask(mm, 2, None, 128, 128, device="cpu")
        with torch.no_grad():
            h1 = model._batch_hidden(batch, bm)
    except Exception as e:  # noqa: BLE001
        print("ATLA flex: CPU'da kosmadi (%s)" % str(e).splitlines()[0][:120], flush=True)
        return
    with torch.no_grad():
        h0 = model._batch_hidden(batch, dense)
    keep = batch.kind != PAD
    d = (h1[keep] - h0[keep]).abs().max().item()
    check("model_z_mask: BlockMask (flex, CPU ileri) = dense; bos satir yok", d < 1e-4 and bool(dense.any(-1).all()),
          "en buyuk fark %.1e" % d)


# --- ogrenilen z (belge 35 Yol A (b); kullanici, 6 Ekim: "formüllü Z üzerine yatırım yapmıyoruz")
def ref_read_mask(kind, doc):
    """Bagimsiz basvuru (dongulu): cumle kimligi ZTOK sayimindan (sent alani kullanilmaz).  q kv'yi gorur: ayni hikaye,
    kv <= q, ve kv BOS / ZTOK ya da (kv TOKEN, q TOKEN ya da ZTOK, ayni cumle); dolgu dolguyu."""
    B, T = kind.shape
    out = torch.zeros(B, T, T, dtype=torch.bool)
    for b in range(B):
        sid, nz, last = [], 0, None
        for j in range(T):
            if doc[b, j] != last:
                nz, last = 0, doc[b, j]
            sid.append(nz)                                                  # Z_k kendi cumlesini kapatir
            nz += int(kind[b, j] == ZTOK)
        for q in range(T):
            for kv in range(q + 1):
                kq, kk = int(kind[b, q]), int(kind[b, kv])
                if doc[b, q] != doc[b, kv]:
                    continue
                if kq == PAD or kk == PAD:
                    out[b, q, kv] = kq == PAD and kk == PAD
                    continue
                out[b, q, kv] = kk in (BOS, ZTOK) or (kk == TOKEN and kq in (TOKEN, ZTOK) and sid[q] == sid[kv])
    return out


def _old_read_mask(kind, doc, sent):
    """Aralik bicimi oncesi model_z_read_mask (belge 65 (a) oncesi, basvuru)."""
    def mask_mod(b, h, q, kv):
        kq, kk = kind[b, q], kind[b, kv]
        summary = (kk == BOS) | (kk == ZTOK)
        word = ((kq == TOKEN) | (kq == D.Kind.END) | (kq == ZTOK)) & ((kk == TOKEN) | (kk == D.Kind.END)) & (
            sent[b, q] == sent[b, kv])
        pad = (kq == PAD) & (kk == PAD)
        return (doc[b, q] == doc[b, kv]) & (kv <= q) & (summary | word | pad)
    return mask_mod


def _old_global_mask(kind, doc, sent):
    def mask_mod(b, h, q, kv):
        return (doc[b, q] == doc[b, kv]) & (kv <= q)
    return mask_mod


def _range_masks(batch, tag="sentetik"):
    """Aralik maskesi (belge 65 (a)) = eski formul + recipe._with_padding, dense, butun (q, kv); recipe.dense_mask
    (includes_padding: sarmasiz) ve _with_padding(yeni) de ayni.  -> (yerel fark, global fark, gorulen)."""
    import recipe as R
    from sentence import model_z_global_mask
    B, T = batch.kind.shape
    k, d_, s_ = batch.kind, batch.doc, batch.sent
    out = []
    for new, old in ((model_z_read_mask, _old_read_mask), (model_z_global_mask, _old_global_mask)):
        ref = _dense(R._with_padding(old(k, d_, s_), k, d_), B, T, "cpu")
        got = _dense(new(k, d_, s_), B, T, "cpu")
        wrapped = _dense(R._with_padding(new(k, d_, s_), k, d_), B, T, "cpu")
        out.append((int((got != ref).sum()) + int((wrapped != ref).sum())
                    + int((R.dense_mask(batch, new) != ref).sum()), int(ref.sum())))
    check("mask %s: aralik bicimi (yerel, global) = eski formul + dolgu kurali; sarmasiz, _with_padding'li ve "
          "recipe.dense_mask ayni (%d satir x %d)" % (tag, B, T), all(o[0] == 0 for o in out),
          "fark / gorulen: %s" % out)


def _range_masks_drive():
    """Gercek satirlar (Drive varsa): SS valid, FineWeb valid, FineWeb train, 8'er satir x 2048."""
    G = "G:/Drive'ım"
    sets = (("SS valid", G + "/simplestories", G + "/v2/simplestories_gpt2", "valid"),
            ("FineWeb valid", G + "/v2/fineweb_edu_s000", G + "/v2/fineweb_edu_s000", "valid"),
            ("FineWeb train", G + "/v2/fineweb_edu_s000", G + "/v2/fineweb_edu_s000", "train"))
    if not os.path.isdir(G + "/v2"):
        print("ATLANDI mask gercek satirlar: Drive yok", flush=True)
        return
    for tag, root, ddir, split in sets:
        ts = D.TokenStories(root, ddir, split)
        if split == "train":
            f = np.load(os.path.join(ddir, "train_pack_plan_e1.npz"))
            ro, rs = f["row_offsets"], f["row_stories"]
            rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(8)]
        else:
            lens = ts.lengths()
            fit = np.nonzero(lens <= 2048)[0][:4000]
            ro, rs = D.pack_plan(lens[fit], 2048, 0, 1)
            rows = [fit[rs[ro[r]:ro[r + 1]]].tolist() for r in range(8)]
        _range_masks(D.build_batch(ts, rows, "model_z", "cpu", 2048), tag)


def t_mask():
    """Aralik maskesi (belge 65 (a)): sentetik (dolgulu) ve gercek SS / FineWeb satirlari, eski formule esit."""
    rng = np.random.default_rng(5)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 20))] for _ in range(rng.integers(1, 8))]
               for _ in range(40)]
    _range_masks(real_batch([list(range(i, i + 5)) for i in range(0, 40, 5)], 400, stories))
    _range_masks_drive()


def _perm_dense(batch, perm, fn):
    """Bugunku dense maske (recipe dolgu kuraliyla) permute edilmis: M[perm[i], perm[j]]."""
    import recipe as R
    B, T = batch.kind.shape
    Md = R.dense_mask(batch, fn)
    return Md.gather(1, perm[:, :, None].expand(B, T, T)).gather(2, perm[:, None, :].expand(B, T, T))


def _last_ranges(batch, tag):
    """summaries_last aralik maskesi (yerel, global) = bugunku maskenin permute dense'i, butun (q, kv)."""
    import recipe as R
    from sentence import _LAST_GLOB, model_z_global_mask, model_z_summaries_last_ranges, summaries_last
    pb, perm = summaries_last(batch)
    out = []
    for old, new in ((model_z_read_mask, model_z_summaries_last_ranges), (model_z_global_mask, _LAST_GLOB)):
        ref = _perm_dense(batch, perm, old)
        got = R.dense_mask(pb, new)
        out.append((int((got != ref).sum()), int(ref.sum())))
    order = pb.kind.long()
    grouped = bool(((order == TOKEN).long().diff(dim=1) <= 0).all())               # token'lar satir basinda
    check("summaries_last %s: iki aralik (yerel, global) = bugunku maskenin permute dense'i; token'lar basta" % tag,
          all(o[0] == 0 for o in out) and grouped, "fark / gorulen: %s" % out)


def t_summaries_last():
    """summaries_last (belge 66): (a) aralik maskesi = permute dense (sentetik dolgulu + Drive'da SS / FineWeb satirlari);
    (b) ayni model ve batch: konum basina hidden, loss_per_target ve parametre gradyani duzenden bagimsiz (fp32 dense),
    G0 / G1 / G2."""
    import recipe as R
    from sentence import summaries_last
    rng = np.random.default_rng(6)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 15))] for _ in range(rng.integers(1, 7))]
               for _ in range(24)]
    rows = [list(range(i, i + 4)) for i in range(0, 24, 4)]
    batch = real_batch(rows, 300, stories)
    _last_ranges(batch, "sentetik")
    G = "G:/Drive'ım"
    if os.path.isdir(G + "/v2"):
        for tag, root, ddir in (("SS valid", G + "/simplestories", G + "/v2/simplestories_gpt2"),
                                ("FineWeb valid", G + "/v2/fineweb_edu_s000", G + "/v2/fineweb_edu_s000")):
            ts = D.TokenStories(root, ddir, "valid")
            lens = ts.lengths()
            fit = np.nonzero(lens <= 2048)[0][:3000]
            ro, rs = D.pack_plan(lens[fit], 2048, 0, 1)
            _last_ranges(D.build_batch(ts, [fit[rs[ro[r]:ro[r + 1]]].tolist() for r in range(6)], "model_z", "cpu", 2048),
                         tag)
    else:
        print("ATLANDI summaries_last gercek satirlar: Drive yok", flush=True)
    pb, perm = summaries_last(batch)
    B, T = batch.kind.shape
    res = []
    for kw in (dict(global_layers=0), dict(global_layers=1), dict(global_layers=2)):
        torch.manual_seed(0)
        m = SentenceTransformer(d=32, layers=3, heads=2, **kw)
        h0 = m._batch_hidden(batch)
        h1 = m._batch_hidden(pb).gather(1, torch.argsort(perm, 1)[..., None].expand(-1, -1, 32))
        dh = float((h0 - h1).abs().max())
        nll0 = m.loss_per_target(batch, R.dense_mask(batch, m.mask_fn) if not isinstance(m.mask_fn, tuple) else
                                 tuple(R.dense_mask(batch, f) for f in m.mask_fn))[0]
        mf = m._masks(True)
        attn1 = R.dense_mask(pb, mf) if not isinstance(mf, tuple) else tuple(R.dense_mask(pb, f) for f in mf)
        nll1 = m.loss_per_target(pb, attn1)[0]
        old = (torch.arange(B)[:, None] * T + perm)[pb.target >= 0]
        dn = float((nll0 - nll1[torch.argsort(old)]).abs().max())
        g0 = torch.autograd.grad(nll0.mean(), list(m.parameters()), allow_unused=True)
        g1 = torch.autograd.grad(nll1.mean(), list(m.parameters()), allow_unused=True)
        dg = max(float((a - b_).norm() / max(float(a.norm()), 1e-12)) for a, b_ in zip(g0, g1) if a is not None)
        res.append((str(kw), dh, dn, dg))
    check("summaries_last: ayni model / batch -- hidden (konum basina), loss_per_target (hedef sirasina geri) ve parametre "
          "gradyani duzenden bagimsiz (fp32 dense; G0, G1, G2)",
          all(r[1] < 1e-5 and r[2] < 1e-5 and r[3] < 1e-5 for r in res),
          "; ".join("%s h %.1e nll %.1e grad %.1e" % r for r in res))


def _gqa_sdpa_cases(fn):
    """fn (gqa_sdpa) -> (ok, bilgi): (B 3, H 4, Hkv 2, hd 8, S 7); maskeler Tq 1: yok, (B, 1, 1, S), (1, S); Tq 5: (B, 1, 5, S),
    (5, S) (her satirda kv 0 acik).  Basvuru SDPA enable_gqa (math tanimi: k / v repeat_interleave); esit head torch.equal."""
    F_ = torch.nn.functional
    g = torch.Generator().manual_seed(3)
    k, v = torch.randn(3, 2, 7, 8, generator=g), torch.randn(3, 2, 7, 8, generator=g)
    worst = 0.0
    for tq in (1, 5):
        q = torch.randn(3, 4, tq, 8, generator=g)
        for shape in ((None,), (3, 1, tq, 7), (tq, 7)):
            m = None if shape == (None,) else torch.rand(*shape, generator=g) < 0.6
            if m is not None:
                m[..., 0] = True
            ref = F_.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
            worst = max(worst, float((fn(q, k, v, m) - ref).abs().max()))
    q = torch.randn(3, 2, 5, 8, generator=g)
    m = torch.rand(3, 1, 5, 7, generator=g) < 0.6
    m[..., 0] = True
    same = torch.equal(fn(q, k, v, m), F_.scaled_dot_product_attention(q, k, v, attn_mask=m))
    return worst < 1e-6 and same, "en buyuk fark %.1e, esit head bit ayni %s" % (worst, same)


def t_gqa():
    """glob_kv_heads (GQA, 8 Ekim): None / heads = bugunku (agirlik ve hidden bit); kv 2 (heads 4): yalniz glob bloklari
    daralir, blok ciktisi = k / v'yi acikca tekrarlayan basvuru (dense), G1 ve G2; summaries_last ayni;
    sizinti yok; SummaryCache adim adim + prefill = tam ileri (onbellek glob'ta kv head); BatchedMuon / NorMuon sekil
    gruplari calisir."""
    import recipe as R
    import train as TR
    from sentence import gqa_sdpa, model_z_global_mask, story_positions, summaries_last
    rng = np.random.default_rng(9)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 9))] for _ in range(rng.integers(2, 6))]
               for _ in range(12)]
    rows = [list(range(i, i + 4)) for i in range(0, 12, 4)]
    batch = real_batch(rows, 160, stories)
    B, T = batch.kind.shape

    def make(**kw):
        torch.manual_seed(0)
        return SentenceTransformer(d=32, layers=3, heads=4, **kw).eval()
    base = make(global_layers=1)
    same = all(torch.equal(a_, b_) for a_, b_ in zip(base.state_dict().values(), make(global_layers=1, glob_kv_heads=4)
                                                     .state_dict().values()))
    with torch.no_grad():
        same &= torch.equal(base._batch_hidden(batch), make(global_layers=1, glob_kv_heads=4)._batch_hidden(batch))
    check("gqa: glob_kv_heads = heads (ya da None) bugunku model (agirlik ve hidden bit)", same)
    ok, info = True, []
    glob = _dense(model_z_global_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    for kw in (dict(global_layers=1), dict(global_layers=2)):
        m = make(glob_kv_heads=2, **kw)
        kv = [blk.kv_heads for blk in m.blocks]
        shapes = [tuple(blk.qkv.weight.shape) for blk in m.blocks]
        l = kv.index(2)
        blk = m.blocks[l]
        with torch.no_grad():
            x = torch.randn(B, T, 32)
            pos = story_positions(batch.kind)
            got = blk(x, pos, glob)
            q, k, v = blk._qkv(x, pos)
            a_ = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                                                                  attn_mask=glob[:, None])
            ref = blk._finish(x, a_)
        ok &= float((got - ref).abs().max()) < 1e-5 and k.shape[1] == 2 and q.shape[1] == 4
        info.append("%s kv %s qkv %s fark %.1e" % (kw, kv, shapes, float((got - ref).abs().max())))
        with torch.no_grad():
            h0 = m._batch_hidden(batch)
            pb, perm = summaries_last(batch)
            h1 = m._batch_hidden(pb).gather(1, torch.argsort(perm, 1)[..., None].expand(-1, -1, 32))
        ok &= float((h0 - h1).abs().max()) < 1e-5
    check("gqa kv 2: yalniz glob bloklari daralir, blok = k / v tekrarli basvuru (dense), summaries_last ayni", ok,
          "; ".join(info))
    m = make(global_layers=1, glob_kv_heads=2)
    alt = [[list(s_) for s_ in st] for st in stories]
    alt[0][-1] = [(t + 7) % D.END_ID for t in alt[0][-1]]
    with torch.no_grad():
        g1, g2 = m._batch_hidden(batch), m._batch_hidden(real_batch(rows, 160, alt))
        keep = batch.target >= 0
        lg_full = g1[keep] @ m.E.weight.T
        out = []
        for row in rows:
            for si in row:
                cache = SummaryCache(m)
                out.append(cache.logits[None])
                for s_ in stories[si]:
                    out += [cache.append_token(t)[None] for t in s_] + [cache.close_sentence()[None]]
        pre = SummaryCache(m)
        pre.prefill(stories[0][:2])
    first = batch.doc[0] == 0
    last = int((first & (batch.sent[0] == int(batch.sent[0][first].max()))).nonzero()[0, 0])
    k_pre = 1 + sum(len(s_) + 1 for s_ in stories[0][:2]) - 1
    d3 = float((torch.cat(out) - lg_full).abs().max())
    check("gqa kv 2: sizinti yok; SummaryCache adim adim ve prefill = tam ileri (glob onbellegi kv head)",
          torch.equal(g1[0, :last], g2[0, :last]) and d3 < 1e-5 and float((pre.logits - torch.cat(out)[k_pre]).abs().max())
          < 1e-5 and cache.all_k[2].shape[1] == 2, "fark %.1e" % d3)
    check("gqa_sdpa (belge 74): Tq 1 katlama ve Tq > 1 genisletme = enable_gqa basvurusu, esit head bit ayni",
          *_gqa_sdpa_cases(gqa_sdpa))
    nm = SentenceTransformer(d=32, layers=3, heads=4, global_layers=1, glob_kv_heads=2)
    opt = TR._optimizer(nm, "normuon", 1e-2, False)[0]
    names = [n for n, _ in R.muon_params(nm)]
    loss = nm.loss_per_target(batch)[0].mean()
    loss.backward()
    w0 = nm.blocks[2].qkv.weight.clone()
    opt.step()
    check("gqa: muon_params deseni ayni (glob qkv %s dahil), NorMuon adimi sekil gruplariyla calisir" % (
        tuple(nm.blocks[2].qkv.weight.shape),), "blocks.2.qkv.weight" in names and not torch.equal(w0, nm.blocks[2].qkv.weight)
          and torch.isfinite(nm.blocks[2].qkv.weight).all())


def t_learned():
    """Ogrenilen z: model_z_read_mask = basvuru; z parcasi yok; nedensellik; gecmisin tek yolu okuma (gradyan); onbellek =
    tam hesap; flex = dense; gercek build_batch."""
    rng = np.random.default_rng(0)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 14))] for _ in range(rng.integers(1, 6))]
               for _ in range(30)]
    rows = [list(range(i, i + 5)) for i in range(0, 30, 5)]
    batch = real_batch(rows, 300, stories)
    B, T = batch.kind.shape
    got = _dense(model_z_read_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    old = _dense(model_z_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    extra = got & ~old
    ntok = int((batch.kind == TOKEN).sum())
    check("learned: model_z_read_mask = dongulu basvuru (gercek build_batch, 30 hikaye); bos satir yok; = model_z_mask + "
          "yalniz ZTOK satirlarinda kendi cumlesinin token'lari (ek = token sayisi)",
          torch.equal(got, ref_read_mask(batch.kind, batch.doc)) and bool(got.any(-1).all()) and bool((old <= got).all())
          and not extra[batch.kind != ZTOK].any() and int(extra.sum()) == ntok, "ek %d, token %d" % (int(extra.sum()), ntok))
    m = learned_model()
    names = set(dict(m.named_parameters()))
    check("learned: z_norm / z_in / anahtar yok (yalniz E, bloklar, norm), mask_fn = model_z_read_mask",
          not hasattr(m, "keys") and not any(n.startswith(("z_norm", "z_in")) for n in names)
          and {n.split(".")[0] for n in names} == {"E", "blocks", "norm"} and m.mask_fn is model_z_read_mask)
    with torch.no_grad():
        h0 = m._batch_hidden(batch)
    bad, read = 0, True
    cols = (batch.kind[0] == TOKEN).nonzero()[:, 0].tolist()[::3]
    for c in cols:
        tk = batch.tokens.clone()
        tk[0, c] = (tk[0, c] + 7) % D.END_ID
        with torch.no_grad():
            diff = (m._batch_hidden(dataclasses.replace(batch, tokens=tk)) - h0).abs().amax(-1)
        bad += int((diff[0, :c] > 0).sum()) + int((diff[1:] > 0).sum())
        zc = c + int((batch.kind[0, c:] == ZTOK).nonzero()[0, 0])          # ayni cumlenin Z'si
        read &= bool(diff[0, zc] > 0)
    check("learned: nedensellik (sutun c'deki token degisince c'den onceki konumlar ve diger satirlar birebir, %d deneme); "
          "Z_k kendi cumlesindeki degisimi gorur" % len(cols), bad == 0 and read)
    first = (batch.sent == 0) & (batch.kind == TOKEN)
    keep = (batch.target >= 0) & (batch.sent >= 1) & (batch.kind == TOKEN)
    only = [i for i in batch.tokens[first].unique().tolist()
            if not (batch.tokens[~first] == i).any() and not (batch.target[keep] == i).any()]
    g = {}
    for name, fn in (("read", model_z_read_mask), ("model_z_mask", model_z_mask)):
        m.zero_grad()
        h = m._batch_hidden(batch, _dense(fn(batch.kind, batch.doc, batch.sent), B, T, "cpu"))
        torch.nn.functional.cross_entropy(h[keep] @ m.E.weight.detach().T, batch.target[keep]).backward()   # cikis yolu kapali
        g[name] = m.E.weight.grad[only].abs().sum().item()
    check("learned: gecmisin tek yolu okuma -- 2. cumleden sonraki kayiptan yalniz 1. cumlede gecen %d token'in E "
          "satirina gradyan: read_mask > 0, model_z_mask = 0" % len(only), len(only) > 10 and g["read"] > 0
          and g["model_z_mask"] == 0, "%.2e / %.1e" % (g["read"], g["model_z_mask"]))
    m.train()
    m.zero_grad()
    nll, _, _ = m.loss_per_target(batch)
    nll.mean().backward()
    check("learned: loss_per_target (gercek build_batch) sonlu; butun parametrelerin gradyani var (olu parametre yok)",
          bool(torch.isfinite(nll).all()) and all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()))
    m.eval()
    with torch.no_grad():
        lg_full, _ = full_logits(m, batch)
        got_l = []
        for row in rows:
            for si in row:
                cache = SummaryCache(m)
                got_l.append(cache.logits[None])
                for s in stories[si]:
                    for t in s:
                        got_l.append(cache.append_token(t)[None])
                    got_l.append(cache.close_sentence()[None])
                    assert all(k is None for k in cache.sen_k)
    d = (torch.cat(got_l) - lg_full).abs().max().item()
    a = m.generate([stories[0][:1], stories[1][:2]], 3, 6, torch.Generator().manual_seed(0))
    b = m.generate([stories[0][:1], stories[1][:2]], 3, 6, torch.Generator().manual_seed(0))
    check("learned: SummaryCache (Z_k kendi cumlesine bakar, sonra cumle silinir) = tam hesap (fp32, %d hedef); generate "
          "sinirlar, ayni tohum ayni metin" % len(lg_full), d < 1e-5 and a == b and all(
              len(gen) <= 3 and all(len(x) <= 6 for x in gen) for gen, _, _ in a), "fark %.1e" % d)
    try:
        from torch.nn.attention.flex_attention import create_block_mask
        b4 = real_batch(rows[:2], 384, stories)
        bm = create_block_mask(model_z_read_mask(b4.kind, b4.doc, b4.sent), 2, None, 384, 384, device="cpu")
        with torch.no_grad():
            h1, h0 = m._batch_hidden(b4, bm), m._batch_hidden(b4)
    except Exception as e:  # noqa: BLE001
        print("ATLA learned flex: CPU'da kosmadi (%s)" % str(e).splitlines()[0][:120], flush=True)
        return
    keepf = b4.kind != PAD
    d = (h1[keepf] - h0[keepf]).abs().max().item()
    check("learned: model_z_read_mask BlockMask (flex, CPU ileri) = dense", d < 1e-4, "fark %.1e" % d)


def t_global():
    """global_layers (belge 40 s6.2 Deney G): 0 = kurucunun varsayilani (bit); son N blok tam causal (ayni hikaye), oteki
    bloklar read_mask; global blokta konum = gercek (transformer duzeni) konum, bagimsiz basvuru ileri gecisiyle ayni ve
    mantiksal konumla farkli; onbellek = tam hesap (N 1, 2); flex = dense; yanlis maske bicimi ve N > katman DURUR; butun
    bloklar global = transformer (bit)."""
    from sentence import model_z_global_mask, story_positions
    rng = np.random.default_rng(1)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 10))] for _ in range(rng.integers(1, 6))]
               for _ in range(20)]
    rows = [list(range(i, i + 5)) for i in range(0, 20, 5)]
    batch = real_batch(rows, 256, stories)
    tb = D.build_batch(token_stories(stories), rows, "transformer", row_len=256)
    B, T = batch.kind.shape

    def model(gl, layers=3):
        return learned_model(d=32, layers=layers, global_layers=gl)
    m0 = model(0)
    torch.manual_seed(0)
    plain = SentenceTransformer(d=32, layers=3, heads=2).eval()
    with torch.no_grad():
        same0 = torch.equal(m0._batch_hidden(batch), plain._batch_hidden(batch))
    check("global: global_layers 0 = kurucunun varsayilani (agirlik ve hidden bit duzeyinde; mask_fn = read_mask)",
          same0 and all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), plain.state_dict().values()))
          and m0.mask_fn is model_z_read_mask)
    m1 = model(1)
    seen = []
    hooks = [blk.register_forward_pre_hook(lambda mod, a: seen.append((a[1], a[2]))) for blk in m1.blocks]
    with torch.no_grad():
        h1 = m1._batch_hidden(batch)
    for hk in hooks:
        hk.remove()
    real = batch.kind != PAD
    full = (batch.doc[:, :, None] == batch.doc[:, None, :]) & torch.ones(T, T, dtype=torch.bool).tril()
    read = _dense(model_z_read_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    glob = _dense(model_z_global_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    rr = real[:, :, None] & real[:, None, :]
    check("global N=1 maske: son blok tam causal (ayni hikaye, kelime + BOS + Z; gercek konumlar arasi = transformer "
          "maskesi), ilk iki blok read_mask; mask_fn (yerel, global); bos satir yok",
          torch.equal(seen[2][1], glob) and torch.equal(seen[0][1], read) and torch.equal(seen[1][1], read)
          and torch.equal(glob & rr, full & rr) and not (glob & (real[:, :, None] ^ real[:, None, :])).any()
          and bool(glob.any(-1).all()) and m1.mask_fn == (model_z_read_mask, model_z_global_mask))
    sp = story_positions(batch.kind)
    dup = uniq = True
    for r, row in enumerate(rows):
        for d_ in range(len(row)):
            sel = batch.doc[r] == d_
            dup &= len(batch.pos[r][sel].unique()) < int(sel.sum())       # mantiksal konum cakisir
            uniq &= len(sp[r][sel].unique()) == int(sel.sum())
    check("global konum: global bloga giden konum = transformer duzeninin pos'u (gercek konumlarda), yerel bloklara "
          "mantiksal pos; her hikayede mantiksal konum cakisiyor, gercek konum cakismiyor",
          torch.equal(seen[2][0][real], tb.pos[real]) and torch.equal(seen[0][0], batch.pos) and dup and uniq)

    def manual(gpos):
        """Bagimsiz ileri gecis: yerel bloklar mantiksal pos + read_mask, son blok gpos + tam causal maske."""
        with torch.no_grad():
            x = m1.E(torch.where(batch.kind == ZTOK, torch.full_like(batch.tokens, D.END_ID), batch.tokens))
            for i, blk in enumerate(m1.blocks):
                x = blk(x, gpos if i == 2 else batch.pos, glob if i == 2 else read)
            return m1.norm(x)
    d_real = (manual(torch.where(real, tb.pos, sp)) - h1)[real].abs().max().item()
    d_log = (manual(batch.pos) - h1)[real].abs().max().item()
    check("global konum (RoPE, q.k goreli konuma bagli): model = gercek konumlu basvuru (fark %.1e); mantiksal konumla "
          "basvurudan farkli (%.1e)" % (d_real, d_log), d_real == 0 and d_log > 1e-3)
    ok_cache, info = True, []
    for gl in (1, 2):
        mg = model(gl)
        with torch.no_grad():
            lg_full, _ = full_logits(mg, batch)
            got = []
            for row in rows:
                for si in row:
                    cache = SummaryCache(mg)
                    got.append(cache.logits[None])
                    for s in stories[si]:
                        for t in s:
                            got.append(cache.append_token(t)[None])
                        got.append(cache.close_sentence()[None])
        d = (torch.cat(got) - lg_full).abs().max().item()
        a = mg.generate([stories[0][:1], stories[1][:2]], 3, 6, torch.Generator().manual_seed(0))
        b = mg.generate([stories[0][:1], stories[1][:2]], 3, 6, torch.Generator().manual_seed(0))
        ok_cache &= d < 1e-5 and a == b and cache.all_k[0] is None and cache.all_k[2].shape[2] == cache.t + 1
        info.append("N=%d fark %.1e" % (gl, d))
    check("global: SummaryCache (global bloklar butun gecmis, gercek konum) adim adim logit = tam ileri gecis (fp32, %d "
          "hedef), N 1 ve 2; generate ayni tohum ayni metin" % len(lg_full), ok_cache, "; ".join(info))
    stops = []
    for attn in (read, (read, glob)):                                    # gl 1'e tek maske; gl 0'a ikili
        for mm in ((m1,) if attn is read else (m0,)):
            try:
                mm._batch_hidden(batch, attn)
                stops.append(False)
            except AssertionError:
                stops.append(True)
    try:
        SentenceTransformer(d=32, layers=2, heads=2, global_layers=3)
        stops.append(False)
    except AssertionError:
        stops.append(True)
    check("global: global_layers'a tek maske, global_layers 0'a ikili maske, katmandan cok global_layers DURUR",
          all(stops), str(stops))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "transformer"))
    from baseline import BaselineTransformer
    torch.manual_seed(0)
    tf = BaselineTransformer(d=32, layers=3, heads=2).eval()
    mall = model(3)
    with torch.no_grad():
        ht, ha = tf._batch_hidden(tb), mall._batch_hidden(batch)
    check("global: butun bloklar global (N = katman) Model Z = transformer, bit duzeyinde (ayni ilk agirlik; Z_k girdisi "
          "E(END) = transformer END'i, tam causal maske, gercek konum; gercek konumlarda hidden)",
          all(torch.equal(a, b) for a, b in zip(tf.state_dict().values(), mall.state_dict().values()))
          and tf.state_dict().keys() == mall.state_dict().keys() and torch.equal(ht[real], ha[real]))
    try:
        from torch.nn.attention.flex_attention import create_block_mask
        b4 = real_batch(rows[:2], 384, stories)
        bms = tuple(create_block_mask(f(b4.kind, b4.doc, b4.sent), 2, None, 384, 384, device="cpu") for f in m1.mask_fn)
        with torch.no_grad():
            hf, hd = m1._batch_hidden(b4, bms), m1._batch_hidden(b4)
    except Exception as e:  # noqa: BLE001
        print("ATLA global flex: CPU'da kosmadi (%s)" % str(e).splitlines()[0][:120], flush=True)
        return
    keepf = b4.kind != PAD
    d = (hf[keepf] - hd[keepf]).abs().max().item()
    check("global: (read_mask, global_mask) BlockMask ikilisi (flex, CPU ileri) = dense", d < 1e-4, "fark %.1e" % d)


def _generate_stepwise(model, prompts, max_sentences, max_tokens, generator=None):
    """Basvuru: prefill oncesi generate (istem token token: append_token / close_sentence; 86ed345 sentence.py)."""
    out = []
    for sents in prompts:
        cache = SummaryCache(model)
        logits = cache.logits
        for s in sents:
            for t in s:
                logits = cache.append_token(t)
            logits = cache.close_sentence()
        gen, ended, eos = [], [], False
        while len(gen) < max_sentences:
            cur, done = [], False
            while len(cur) < max_tokens:
                p = logits.float()
                w = int(p.argmax()) if generator is None else int(torch.multinomial(
                    torch.softmax(p, -1).cpu(), 1, generator=generator))
                if w == model.EOS and not cur:
                    eos = True
                    break
                if w in (model.END, model.EOS):
                    done = True
                    break
                cur.append(w)
                logits = cache.append_token(w)
            if eos:
                break
            gen.append(cur)
            ended.append(done)
            logits = cache.close_sentence()
        out.append((gen, ended, eos))
    return out


def t_prefill():
    """SummaryCache.prefill (belge 46): istem tek ileri geciste = token token (logit ve onbellek fp32 ~1e-5, sayaclar
    ayni); generate (prefill'li) = token token generate, acgozlu ve ornekleme, G 0 / 1 / 2, bos, kisa ve ~400 token'lik
    istem."""
    rng = np.random.default_rng(5)
    rs = lambda n: [[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 15))] for _ in range(n)]  # noqa: E731
    prompts = [[], rs(1), rs(3), rs(50)]
    info, ok_cache, ok_gen = [], True, True
    for gl in (0, 1, 2):
        m = learned_model(d=32, layers=3, global_layers=gl)
        dmax = 0.0
        with torch.no_grad():
            for sents in prompts:
                a, b = SummaryCache(m), SummaryCache(m)
                la = a.prefill(sents)
                lb = b.logits
                for x in sents:
                    for t in x:
                        lb = b.append_token(t)
                    lb = b.close_sentence()
                d = (la - lb).abs().max().item()
                for ka, kb in ((a.sum_k, b.sum_k), (a.sum_v, b.sum_v), (a.all_k, b.all_k), (a.all_v, b.all_v)):
                    for x, y in zip(ka, kb):
                        ok_cache &= (x is None) == (y is None) and (x is None or x.shape == y.shape)
                        d = max(d, 0.0 if x is None or x.shape != y.shape else (x - y).abs().max().item())
                ok_cache &= (a.n_z, a.i, a.t) == (b.n_z, b.i, b.t) and all(k is None for k in a.sen_k)
                dmax = max(dmax, d)
                # prefill sonrasi decode da ayni: bir cumle daha token token
                for t in (sents[0] if sents else [7, 8]):
                    dmax = max(dmax, (a.append_token(t) - b.append_token(t)).abs().max().item())
                dmax = max(dmax, (a.close_sentence() - b.close_sentence()).abs().max().item())
        ok_cache &= dmax < 1e-5
        g_new = m.generate(prompts, 4, 12)
        g_old = _generate_stepwise(m, prompts, 4, 12)
        s_new = m.generate(prompts, 4, 12, torch.Generator().manual_seed(11))
        s_old = _generate_stepwise(m, prompts, 4, 12, torch.Generator().manual_seed(11))
        ok_gen &= g_new == g_old and s_new == s_old
        info.append("G%d fark %.1e" % (gl, dmax))
    from sentence import StaticCache
    sd = 0.0                                                              # StaticCache = SummaryCache (belge 84)
    for kw in (dict(global_layers=0), dict(global_layers=1, glob_kv_heads=1), dict(global_layers=3)):
        torch.manual_seed(0)
        m = SentenceTransformer(32, 3, 2, **kw).eval()
        with torch.no_grad():
            for sents in prompts:
                a, b = SummaryCache(m), StaticCache(m, 64, 8, 16)
                sd = max(sd, float((a.prefill(sents) - b.prefill(sents)).abs().max()))
                for t in [5, 6, -1, 7, -1]:
                    la, lb = (a.close_sentence(), b.close_sentence()) if t < 0 else (a.append_token(t), b.append_token(t))
                    sd = max(sd, float((la - lb).abs().max()))
    try:
        c = StaticCache(m, 64, 8, 2)
        c.prefill([])
        [c.append_token(5) for _ in range(300)]                          # cumle kademesi 256 asilir
        stops = False
    except AssertionError:
        stops = True
    check("StaticCache (belge 84) adim adim = SummaryCache (fp32 < 1e-5; G 0 / 1 GQA / hepsi glob; bos, kisa, uzun istem, "
          "token ve Z adimi), tampon asimi DURUR, carry desteklenmez (SummaryCache)", sd < 1e-5 and stops
          and not StaticCache.supports(SentenceTransformer(32, 3, 2, global_layers=1, carry_group=2)), "fark %.1e" % sd)
    from sentence import decode_sdpa, _SPLIT                                 # parcali (split-KV, belge 92) = fp64 basvuru
    g_ = torch.Generator().manual_seed(4)
    q_, k_ = torch.randn(2, 4, 1, 16, generator=g_), torch.randn(2, 2, 4 * _SPLIT, 16, generator=g_)
    v_, m_ = torch.randn(2, 2, 4 * _SPLIT, 16, generator=g_), torch.arange(4 * _SPLIT)[None] < torch.tensor([[_SPLIT + 5], [3]])
    ref_ = torch.nn.functional.scaled_dot_product_attention(q_.double(), k_.double().repeat_interleave(2, 1),
                                                            v_.double().repeat_interleave(2, 1), attn_mask=m_[:, None, None])
    sp_ = float((decode_sdpa(q_, k_, v_, m_).double() - ref_).abs().max())
    check("decode_sdpa parcali (S = 4 x _SPLIT, satir basina farkli uzunluk, tamamen maskeli parcalar) = fp64 basvuru",
          sp_ < 1e-5, "fark %.1e" % sp_)
    n = sum(len(x) + 1 for x in prompts[-1]) + 1
    check("prefill: istem tek ileri gecis = token token (son logit, ozet ve global K/V, sayaclar; sonraki decode adimlari; "
          "fp32 < 1e-5), G 0 / 1 / 2, istem 1 / 4 / %d konum" % n, ok_cache, "; ".join(info))
    check("prefill: generate = token token generate (acgozlu + ornekleme, G 0 / 1 / 2, bos / kisa / %d konumluk istem), "
          "token token ayni" % n, ok_gen)


# --- eski kodla esdegerlik (belge 44; belge 33 s5 deseni): etiketteki ogrenilen z = bugunku Model Z, bit duzeyinde
def _equiv_side(old):
    """Ayni tohum, ayni veri, G'siz ve G 1: ilk agirlik, 3 AdamW adiminin kayip ve gradyanlari, son agirlik, generate
    (acgozlu + ornekleme)."""
    d, layers, heads = 64, 2, 4
    rng = np.random.default_rng(0)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 14))] for _ in range(rng.integers(1, 6))]
               for _ in range(12)]
    batch = real_batch([list(range(0, 4)), list(range(4, 8)), list(range(8, 12))], 160, stories)
    res = {}
    for gl in (0, 1):
        torch.manual_seed(0)
        if old:                                                          # etiket: learned_z bayrakli kurucu
            model = SentenceTransformer(None, d, layers, heads, learned_z=True, global_layers=gl)
        else:
            model = SentenceTransformer(d, layers, heads, global_layers=gl)
        r = dict(init={k: v.clone() for k, v in model.state_dict().items()}, nll=[], grads=[])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            nll, _, _ = model.loss_per_target(batch)
            nll.mean().backward()
            r["nll"].append(nll.detach().clone())
            r["grads"].append({n: p.grad.clone() for n, p in model.named_parameters()})
            opt.step()
        r["final"] = {k: v.clone() for k, v in model.state_dict().items()}
        model.eval()
        prompts = [st[:1] for st in stories[:4]]
        r["greedy"] = model.generate(prompts, 4, 10)
        r["sample"] = model.generate(prompts, 4, 10, torch.Generator().manual_seed(3))
        res.update({"g%d_%s" % (gl, k): v for k, v in r.items()})
    return res


def _same(x, y):
    if torch.is_tensor(x):
        return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_same(i, j) for i, j in zip(x, y))
    return x == y


def t_equiv():
    """Etiket v2-before-formula-cleanup-20261007'deki ogrenilen z (G'siz ve G 1; git show, ayri surec) = bugunku Model Z:
    ayni tohumda agirlik, kayip, gradyan, 3 adim sonrasi agirlik, generate (acgozlu + ornekleme) bit duzeyinde."""
    tmp = tempfile.mkdtemp(prefix="tests_model_z_equiv_")
    try:
        for src in ("common/data.py", "common/recipe.py", "model_z/sentence.py", "model_z/sentence_z.py"):   # recipe: ust import
            txt = subprocess.run(["git", "-C", REPO, "show", "%s:deneme2/v2/%s" % (TAG, src)], capture_output=True,
                                 check=True).stdout
            os.makedirs(os.path.dirname(os.path.join(tmp, src)), exist_ok=True)
            with open(os.path.join(tmp, src), "wb") as f:
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
    check("equiv: etiketteki ogrenilen z = bugunku Model Z (G'siz ve G 1), bit duzeyinde (%d alan: init, nll, grads, final, "
          "greedy, sample)" % len(a), not bad and a.keys() == b.keys(), "farkli: %s" % bad if bad else "")


def t_carry():
    """--carry_summaries (belge 83): (a) G'siz modelde carry batch'indeki parcalar = bolunmemis belgenin tam ileri gecisi;
    G1'de glob maskesi "sonraki parca onceki parcalarin yalniz Z'lerini gorur" olan bolunmemis basvuruyla ayni (fp32);
    (b) sizinti: bellek sutunlari yalniz devam satirinin ilk hikayesine, gecerli yuvalara (glob'da BOS haric), kaynak
    yalniz onceki satirlar; (c) SummaryCache carry (parca dolunca glob'da yalniz Z, carry_group'ta sifirlama) adim adim =
    tam ileri.  G1 kollari GQA ile de (glob_kv_heads 1, heads 4; GQA varsayilan, kullanici 8 Ekim)."""
    import recipe as R
    from sentence import model_z_read_mask, summaries_last
    rng = np.random.default_rng(1)
    sents = [[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 9))] for _ in range(14)]
    pieces = [sents[:5], sents[5:9], sents[9:]]
    other = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 6))] for _ in range(2)] for _ in range(3)]
    st = token_stories(pieces + other)
    st.continues = np.array([True, True, False, False, False, False])
    gpos = np.array([0, 1, 2, 0, 0, 0])
    T = 80
    rows = [[0, 3], [1, 4], [2, 5]]
    b = D.build_batch(st, rows, "model_z", row_len=T, carry=dict(gpos=gpos, memory=True, m_max=32))
    bu = D.build_batch(token_stories([sents]), [[0]], "model_z", row_len=2 * T)
    lens = [sum(len(x) + 1 for x in p_) + (1 if i == 0 else 0) for i, p_ in enumerate(pieces)]
    res = []
    for G, kv, h in ((0, None, 2), (1, None, 2), (1, 1, 4)):
        torch.manual_seed(0)
        m = SentenceTransformer(d=32, layers=3, heads=h, global_layers=G, glob_kv_heads=kv).eval()
        with torch.no_grad():
            pb, perm = summaries_last(b)
            hc = m._batch_hidden(pb).gather(1, torch.argsort(perm, 1)[..., None].expand(-1, -1, 32))
            if G == 0:
                pu, permu = summaries_last(bu)
                hu = m._batch_hidden(pu).gather(1, torch.argsort(permu, 1)[..., None].expand(-1, -1, 32))
            else:
                k_, d_, s_ = bu.kind, bu.doc, bu.sent
                ar = torch.arange(2 * T)
                pc = torch.bucketize(ar, torch.tensor(np.cumsum(lens)), right=True)        # konumun parcasi
                real, causal = d_[0] >= 0, ar[:, None] >= ar[None, :]
                glob = causal & (((pc[:, None] == pc[None, :]) | ((k_[0][None, :] == ZTOK) & (pc[None, :] < pc[:, None])))
                                 & real[:, None] & real[None, :] | (~real[:, None] & ~real[None, :]))
                hu = m._batch_hidden(bu, (_dense(model_z_read_mask(k_, d_, s_), 1, 2 * T, "cpu"), glob[None]))
        off = np.r_[0, np.cumsum(lens)]
        res.append(max(float((hc[r, :lens[r]] - hu[0, off[r]:off[r + 1]]).abs().max()) for r in range(3)))
    pb, _ = summaries_last(b)
    M = pb.mem_rows.shape[1]
    first = (pb.doc == 0) & (pb.kind != PAD)
    leak = True
    for f, glob in ((m._masks(True)[0], False), (m._masks(True)[-1], True)):
        dm = R.dense_mask(pb, f)[:, :, T:]                                   # (B, T, M) bellek bolumu
        n = (pb.mem_rows >= 0).sum(1)
        want = first[:, :, None] & (torch.arange(M)[None, None, :] < n[:, None, None]) & (
            torch.arange(M)[None, None, :] >= int(glob))
        leak &= torch.equal(dm, want)
    src_ok = all(int(r_) < r for r in range(3) for r_ in pb.mem_rows[r].tolist() if r_ >= 0)
    check("carry: G'siz parcalar = bolunmemis belge (fark %.1e), G1 = 'onceki parcalarin yalniz Z'si' glob maskeli basvuru "
          "(fark %.1e; GQA kv 1 %.1e); bellek yalniz devam satirinin ilk hikayesine, gecerli yuvalara (glob'da BOS'suz), kaynak yalniz "
          "onceki satirlar" % tuple(res), max(res) < 1e-5 and leak and src_ok)
    sents = [[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 9))] for _ in range(30)]
    RL, G_ = 160, 2
    pieces, cur, used, k = [], [], 1, 0                                       # uretim kurali (SummaryCache)
    for x in sents:
        cur.append(x)
        used += len(x) + 1
        if used > RL - D.MAX_SENTENCE_TOKENS - 1:
            pieces.append(cur)
            k += 1
            cur, used = [], (1 if k % G_ == 0 else 0)
    pieces += [cur] if cur else []
    st = token_stories(pieces)
    st.continues = np.array([i < len(pieces) - 1 for i in range(len(pieces))])
    gp = np.arange(len(pieces)) % G_
    out = []
    for gl, kv, h in ((0, None, 2), (1, None, 2), (1, 1, 4)):
        torch.manual_seed(0)
        m = SentenceTransformer(d=32, layers=3, heads=h, global_layers=gl, glob_kv_heads=kv, carry_group=G_).eval()
        m.row_len = RL
        b = D.build_batch(st, [[i] for i in range(len(pieces))], "model_z", row_len=RL,
                          carry=dict(gpos=gp, memory=True, m_max=128))
        with torch.no_grad():
            pb, perm = summaries_last(b)
            lg = m._batch_hidden(pb).gather(1, torch.argsort(perm, 1)[..., None].expand(-1, -1, 32)) @ m.E.weight.T
            want = []
            for r, p_ in enumerate(pieces):
                n = sum(len(x) + 1 for x in p_) + (1 if gp[r] == 0 else 0)
                reset = r + 1 < len(pieces) and gp[r + 1] == 0                # sifirlamada son Z'nin hedefi yok
                want += [lg[r, c] for c in range(n - int(reset))]
            c = SummaryCache(m)
            got = [c.logits]
            for x in sents:
                got += [c.append_token(t) for t in x] + [c.close_sentence()]
            pre = SummaryCache(m)
            pre.prefill(sents[:9])
        out.append((max(float((a - b_).abs().max()) for a, b_ in zip(got, want)), len(want),
                    float((pre.logits - got[sum(len(x) + 1 for x in sents[:9])]).abs().max())))
    check("carry uretim: SummaryCache adim adim (parca dolunca glob'da yalniz Z'ler, %d parcada sifirlama) = carry batch'inin "
          "tam ileri gecisi, %d parca; prefill = adim adim; G0, G1, G1 GQA kv 1" % (G_, len(pieces)),
          all(o[0] < 1e-5 and o[2] < 1e-5 for o in out), "; ".join("%s fark %.1e (%d konum), prefill %.1e" % (
              nm, o[0], o[1], o[2]) for nm, o in zip(("G0", "G1", "G1 kv1"), out)))


def t_vocab():
    """Sozluk dolgusu (belge 89): vocab_rows 50.304 -> ilk agirlik gercek satirlarda VOCAB'li modelle bit, dolgu satirlari
    sifir.  Dolgu satirlari buyuk rastgele yapilinca (duz h E^T'de argmax dolguya duser: test bos degil) cikti degismez:
    _logits VOCAB sutun ve loss_per_target ayni (<= 1e-5, fp32 matmul blok sirasi; tahmin ayni), output_loss ayni ve
    dolgu satiri gradyani 0, generate (StaticCache ve
    SummaryCache; acgozlu ve ornekleme) VOCAB'li modelle token token ayni, uretilen her token < VOCAB."""
    import recipe as R
    import sentence as S
    rows = 50304
    torch.manual_seed(0)
    a = SentenceTransformer(32, 3, 2, global_layers=1).eval()
    torch.manual_seed(0)
    b = SentenceTransformer(32, 3, 2, global_layers=1, vocab_rows=rows).eval()
    sa, sb = a.state_dict(), b.state_dict()
    init_ok = all(torch.equal(sa[k], sb[k][:D.VOCAB] if k == "E.weight" else sb[k]) for k in sa) \
        and tuple(sb["E.weight"].shape) == (rows, 32) and bool((sb["E.weight"][D.VOCAB:] == 0).all())
    with torch.no_grad():
        b.E.weight[D.VOCAB:] = torch.randn(rows - D.VOCAB, 32, generator=torch.Generator().manual_seed(3)) * 50
    batch = real_batch([[0, 1], [2]], 64)
    with torch.no_grad():
        h = b._batch_hidden(batch)
        raw_pad = float(((h @ b.E.weight.T).argmax(-1) >= D.VOCAB).float().mean())
        la, lb = a._logits(h), b._logits(h)
        na, pa, _ = a.loss_per_target(batch)
        nb, pb, _ = b.loss_per_target(batch)
    hh = h.flatten(0, 1).detach()
    E = b.E.weight.detach().clone().requires_grad_(True)
    loss_b = R.output_loss(hh, E, batch.target.flatten())
    loss_b.backward()
    loss_a = R.output_loss(hh, a.E.weight.detach(), batch.target.flatten())
    near = lambda x, y: float((x - y).abs().max()) <= 1e-5  # noqa: E731  (fp32 matmul blok sirasi ~1e-7)
    out_ok = la.shape[-1] == D.VOCAB and near(la, lb) and near(na, nb) and torch.equal(pa, pb) \
        and float((loss_a - loss_b).abs()) < 1e-6 and bool((E.grad[D.VOCAB:] == 0).all()) and bool(E.grad[:D.VOCAB].any())
    rng = np.random.default_rng(4)
    prompts = [[], [[int(x) for x in rng.integers(0, D.END_ID, 6)]], [[3, 4, 5], [6, 7, 8, 9]]]
    gens, saved = {}, S.STATIC_DECODE
    try:
        for static in (True, False):
            S.STATIC_DECODE = static
            for name, m in (("a", a), ("b", b)):
                with torch.no_grad():
                    gens[name, static] = (m.generate(prompts, 3, 8), m.generate(prompts, 3, 8, torch.Generator().manual_seed(5)))
    finally:
        S.STATIC_DECODE = saved
    toks = [t for g in gens.values() for out in g for gen, _, _ in out for s_ in gen for t in s_]
    gen_ok = all(gens["a", s_] == gens["b", s_] for s_ in (True, False)) and toks and max(toks) < D.VOCAB
    check("sozluk dolgusu (vocab_rows 50.304): ilk agirlik gercek satirlarda bit, dolgu sifir; dolgu satirlari buyuk "
          "rastgele iken (duz h E^T argmax'inin %.0f%%'i dolgu) _logits VOCAB sutun, _logits / loss_per_target <= 1e-5, "
          "output_loss ayni ve dolgu gradyani 0, generate (StaticCache + SummaryCache, acgozlu + ornekleme) ayni, uretilen "
          "token < VOCAB (%d token)" % (100 * raw_pad, len(toks)), init_ok and raw_pad > 0.5 and out_ok and gen_ok,
          "init %s, cikis %s, uretim %s" % (init_ok, out_ok, gen_ok))


def t_flex_ranges():
    """FlexAttention yolu (recipe.block_mask, CPU ileri) = dense yol (hakem B B10, betik hakemB/b6): egitimin varsayilan
    maskeleri summaries_last aralik maskesi (G0 / G1) ve summaries_last + carry bellegi (anahtar T + M), gercek
    build_batch; dolgu disi konumlarda hidden <= 1e-5."""
    import recipe as R
    from sentence import summaries_last
    rng = np.random.default_rng(1)
    rs = lambda n, k: [[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, k))] for _ in range(n)]  # noqa: E731
    sents = rs(30, 9)
    groups = [sents[:12], sents[12:22], sents[22:]] + [rs(2, 6) for _ in range(6)]
    stream, sent, story = [], [], [0]
    for g in groups:
        for x in g:
            sent.append((len(stream), len(stream) + len(x)))
            stream += x
        story.append(len(sent))
        stream.append(D.EOS_ID)
    st = SimpleNamespace(stream=np.array(stream, np.int64), sent=np.array(sent, np.int64), story=np.array(story),
                         continues=np.array([True, True] + [False] * 7))
    rows, diffs = [[0, 3, 4], [1, 5, 6], [2, 7, 8]], {}
    for name, carry in (("aralik", None), ("aralik+carry", dict(gpos=np.array([0, 1, 2] + [0] * 6), memory=True, m_max=64))):
        pb, _ = summaries_last(D.build_batch(st, rows, "model_z", row_len=128, carry=carry))
        for G in (0, 1):
            torch.manual_seed(0)
            m = SentenceTransformer(d=32, layers=3, heads=2, global_layers=G).eval()
            mf = m._masks(True)
            fns = mf if isinstance(mf, tuple) else (mf,)
            wrap = (lambda t: t) if G else (lambda t: t[0])  # noqa: E731
            with torch.no_grad():
                hf = m._batch_hidden(pb, wrap(tuple(R.block_mask(pb, f) for f in fns)))
                hd = m._batch_hidden(pb, wrap(tuple(R.dense_mask(pb, f) for f in fns)))
            keep = pb.kind != D.Kind.PAD
            diffs["%s G%d" % (name, G)] = float((hf[keep] - hd[keep]).abs().max())
    check("FlexAttention = dense: summaries_last aralik maskesi ve aralik + carry bellegi (anahtar T + M), G0 / G1 "
          "(<= 1e-5)", max(diffs.values()) <= 1e-5, ", ".join("%s %.1e" % kv for kv in diffs.items()))


def t_limit():
    import sentence as MOD
    MAKE = lambda: (torch.manual_seed(0), SentenceTransformer(32, 2, 2, global_layers=1).eval())[1]  # noqa: E731
    """Uretim konum siniri (belge 89b; hakem A b3 / B B7): row_len 30 iken ornekleme uretimi (ayni tohum) row_len'siz
    uretimin oneki, son cumle kesik (ended False), islenen konum (BOS + token + kapanis) <= 30; sinirsiz uretim 30'u asiyor
    (test bos degil); StaticCache ve eski yol, istem kapali / acik (open_last)."""
    def used(prompt, gen, opened, closes):                      # islenen konum: BOS + istem + token + kapanis
        return 1 + sum(len(s_) + 1 for s_ in prompt) - opened + sum(map(len, gen)) + closes
    prompt, ok, info = [[3, 4, 5, 6], [7, 8, 9]], [], []
    saved = MOD.STATIC_DECODE
    try:
        for static in (True, False):
            MOD.STATIC_DECODE = static
            for opened in (False, True):
                m = MAKE()
                m.row_len = 100000
                full = m.generate([prompt], 12, 6, torch.Generator().manual_seed(1), open_last=opened)[0]
                m.row_len = 30
                lim = m.generate([prompt], 12, 6, torch.Generator().manual_seed(1), open_last=opened)[0]
                g, f = lim[0], full[0]
                prefix = len(g) >= 1 and g[:-1] == f[:len(g) - 1] and f[len(g) - 1][:len(g[-1])] == g[-1]
                ok.append(prefix and lim[1][-1] is False and used(prompt, g, opened, len(g) - 1) <= 30
                          and used(prompt, f, opened, len(f)) > 30 and m.max_positions() == 30)
                info.append("%d/%d" % (used(prompt, g, opened, len(g) - 1), used(prompt, f, opened, len(f))))
    finally:
        MOD.STATIC_DECODE = saved
    check("uretim konum siniri (row_len 30): sinirli uretim = sinirsizin oneki, son cumle kesik, konum <= 30 (sinirsiz "
          "> 30); StaticCache + eski yol, open_last 0 / 1", all(ok), "konum " + ", ".join(info))


def t_gate():
    """attn_gate (belge 88a, 90a): (a) kapali = bugunku model (parametre yok, agirlik ve hidden bit); acik: kapi disindaki
    ilk agirlik kapisizla bit ayni, kapi sifir (0,5); (b) blok = bagimsiz basvuru (SDPA ciktisi (B, T, H, hd) x
    sigmoid(n1(x) W^T), sonra proj, MLP; yerel ve glob GQA); summaries_last ayni; sizinti yok; (c) SummaryCache adim adim
    ve prefill = tam ileri, StaticCache = SummaryCache, generate = token token (acgozlu + ornekleme), G 0 / 1 GQA / hepsi
    glob; carry uretimi = carry batch'inin tam ileri gecisi; (d) gradyan kapiya ulasir, NorMuon grubunda."""
    import torch.nn.functional as F_
    import train as TR
    import recipe as R
    from sentence import StaticCache, story_positions, summaries_last, model_z_global_mask
    DM = 128                                                              # kapi 2: girdi d // 64 = 2 boyut
    rng = np.random.default_rng(13)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 9))] for _ in range(rng.integers(2, 6))]
               for _ in range(12)]
    rows = [list(range(i, i + 4)) for i in range(0, 12, 4)]
    batch = real_batch(rows, 160, stories)
    B, T = batch.kind.shape

    def make(gate=0, rand=True, **kw):
        torch.manual_seed(0)
        m = SentenceTransformer(d=DM, layers=3, heads=4, attn_gate=gate, **kw).eval()
        if gate and rand:                                                 # kapi 0,5'ten uzak: head / satir farkli
            g = torch.Generator().manual_seed(1)
            with torch.no_grad():
                for blk in m.blocks:
                    blk.attn_gate.copy_(1.5 / blk.attn_gate.shape[1] ** 0.5 * torch.randn(blk.attn_gate.shape, generator=g))
        return m
    torch.manual_seed(0)
    off = SentenceTransformer(d=DM, layers=3, heads=4, global_layers=1).eval()     # bayraksiz kurucu
    base, on = make(global_layers=1, gate=0), make(global_layers=1, gate=1, rand=False)
    sd_on = on.state_dict()
    with torch.no_grad():
        same = all(torch.equal(a_, b_) for a_, b_ in zip(off.state_dict().values(), base.state_dict().values())) and \
            torch.equal(off._batch_hidden(batch), base._batch_hidden(batch))
    gk = [k for k in sd_on if k.endswith("attn_gate")]
    init_ok = all(torch.equal(v, base.state_dict()[k]) for k, v in sd_on.items() if k not in gk) and \
        set(sd_on) - set(gk) == set(base.state_dict()) and len(gk) == 3 and all(
            not sd_on[k].any() and tuple(sd_on[k].shape) == (4, DM) for k in gk)
    check("gate: kapali = bugunku model (attn_gate parametresi yok, agirlik ve hidden bit); acik: kapi disindaki ilk "
          "agirlik bit ayni, kapi (heads, d) sifir", same and init_ok and not any("attn_gate" in k for k in base.state_dict()))

    ok, info = True, []
    glob = _dense(model_z_global_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    loc = _dense(model_z_read_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    for gv, l, (mask, pos) in [(gv, l, mp) for gv in (1, 2) for l, mp in enumerate(
            ((loc, batch.pos), (loc, batch.pos), (glob, story_positions(batch.kind))))]:
        blk = make(global_layers=1, glob_kv_heads=2, gate=gv).blocks[l]
        with torch.no_grad():
            x = torch.randn(B, T, DM)
            got = blk(x, pos, mask)
            q, k, v = blk._qkv(x, pos)
            rep = blk.heads // blk.kv_heads
            a_ = F_.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                                 attn_mask=mask[:, None]).transpose(1, 2)
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.finfo(x.dtype).eps) * blk.n1.weight
            gate = torch.sigmoid(xn[..., :blk.attn_gate.shape[1]] @ blk.attn_gate.T)   # (B, T, H); 2: ilk d // 64 boyut
            h = x + (a_ * gate[..., None]).reshape(B, T, DM) @ blk.proj.weight.T
            gg, u = blk.gate_up(blk.n2(h)).chunk(2, -1)
            ref = h + blk.down(F_.silu(gg) * u)
        d_ = float((got - ref).abs().max())
        ok &= d_ < 1e-5 and float(gate.std()) > 0.1 and tuple(blk.attn_gate.shape) == (4, DM if gv == 1 else DM // 64)
        info.append("kapi %d blok %d (kv %d) %.1e" % (gv, l, blk.kv_heads, d_))
    first = batch.doc[0] == 0
    last = int((first & (batch.sent[0] == int(batch.sent[0][first].max()))).nonzero()[0, 0])
    alt = [[list(s_) for s_ in st] for st in stories]
    alt[0][-1] = [(t + 7) % D.END_ID for t in alt[0][-1]]
    for gv in (1, 2):
        m = make(global_layers=1, glob_kv_heads=2, gate=gv)
        with torch.no_grad():
            h0 = m._batch_hidden(batch)
            pb, perm = summaries_last(batch)
            h1 = m._batch_hidden(pb).gather(1, torch.argsort(perm, 1)[..., None].expand(-1, -1, DM))
            h2 = m._batch_hidden(real_batch(rows, 160, alt))
        ok &= float((h0 - h1).abs().max()) < 1e-5 and torch.equal(h0[0, :last], h2[0, :last]) and not torch.equal(
            h0[0, last:], h2[0, last:])
    check("gate 1 / 2: blok = bagimsiz basvuru (SDPA ciktisi x sigmoid(n1(x) W^T) head basina, 2'de n1(x)[..., :d // 64] ve W "
          "(H, d // 64), d 128; proj, MLP; yerel + glob GQA); summaries_last ayni; sizinti yok (son cumle degisince onceki konumlar bit "
          "ayni)", ok, "; ".join(info))

    rs = lambda n: [[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 15))] for _ in range(n)]  # noqa: E731
    prompts = [[], rs(1), rs(3), rs(30)]
    res = []
    for gv, kw in [(gv, kw) for gv in (1, 2) for kw in (dict(global_layers=0), dict(global_layers=1, glob_kv_heads=2),
                                                          dict(global_layers=3))]:
        m = make(gate=gv, **kw)
        with torch.no_grad():
            keep = batch.target >= 0
            lg_full = m._batch_hidden(batch)[keep] @ m.E.weight.T
            out = []
            for row in rows:
                for si in row:
                    c = SummaryCache(m)
                    out.append(c.logits[None])
                    for s_ in stories[si]:
                        out += [c.append_token(t)[None] for t in s_] + [c.close_sentence()[None]]
            d_step = float((torch.cat(out) - lg_full).abs().max())
            d_pre, d_static = 0.0, 0.0
            for sents in prompts:
                a, b, s = SummaryCache(m), SummaryCache(m), StaticCache(m, 64, 8, 16)
                la, ls = a.prefill(sents), s.prefill(sents)
                lb = b.logits
                for x_ in sents:
                    for t in x_:
                        lb = b.append_token(t)
                    lb = b.close_sentence()
                d_pre = max(d_pre, float((la - lb).abs().max()))
                d_static = max(d_static, float((la - ls).abs().max()))
                for t in [5, 6, -1, 7, -1]:
                    x1, x2 = (a.close_sentence(), s.close_sentence()) if t < 0 else (a.append_token(t), s.append_token(t))
                    d_static = max(d_static, float((x1 - x2).abs().max()))
        gen_ok = m.generate(prompts, 4, 12) == _generate_stepwise(m, prompts, 4, 12) and m.generate(
            prompts, 4, 12, torch.Generator().manual_seed(11)) == _generate_stepwise(
            m, prompts, 4, 12, torch.Generator().manual_seed(11))
        res.append((gv, kw, d_step, d_pre, d_static, gen_ok))
    check("gate 1 / 2: SummaryCache adim adim = tam ileri, prefill = adim adim, StaticCache = SummaryCache (fp32 < 1e-5), generate "
          "= token token (acgozlu + ornekleme); G0 / G1 GQA kv 2 / hepsi glob",
          all(r_[2] < 1e-5 and r_[3] < 1e-5 and r_[4] < 1e-5 and r_[5] for r_ in res),
          "; ".join("kapi %d G%d adim %.1e prefill %.1e static %.1e gen %s" % (r_[0], r_[1]["global_layers"], *r_[2:])
                    for r_ in res))

    sents = [[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 9))] for _ in range(30)]
    RL, G_ = 160, 2
    pieces, cur, used, k = [], [], 1, 0                                       # uretim kurali (SummaryCache, t_carry)
    for x_ in sents:
        cur.append(x_)
        used += len(x_) + 1
        if used > RL - D.MAX_SENTENCE_TOKENS - 1:
            pieces.append(cur)
            k += 1
            cur, used = [], (1 if k % G_ == 0 else 0)
    pieces += [cur] if cur else []
    st = token_stories(pieces)
    st.continues = np.array([i < len(pieces) - 1 for i in range(len(pieces))])
    gp = np.arange(len(pieces)) % G_
    cres = []
    for gv, gl in ((1, 0), (1, 1), (2, 1)):
        m = make(gate=gv, global_layers=gl, carry_group=G_)
        m.row_len = RL
        b = D.build_batch(st, [[i] for i in range(len(pieces))], "model_z", row_len=RL,
                          carry=dict(gpos=gp, memory=True, m_max=128))
        with torch.no_grad():
            pb, perm = summaries_last(b)
            lg = m._batch_hidden(pb).gather(1, torch.argsort(perm, 1)[..., None].expand(-1, -1, DM)) @ m.E.weight.T
            want = []
            for r, p_ in enumerate(pieces):
                n = sum(len(x_) + 1 for x_ in p_) + (1 if gp[r] == 0 else 0)
                reset = r + 1 < len(pieces) and gp[r + 1] == 0
                want += [lg[r, c] for c in range(n - int(reset))]
            c = SummaryCache(m)
            got = [c.logits] + [y for x_ in sents for y in [c.append_token(t) for t in x_] + [c.close_sentence()]]
        cres.append(max(float((a_ - b_).abs().max()) for a_, b_ in zip(got, want)))
    check("gate carry: SummaryCache carry adim adim = carry batch'inin tam ileri gecisi (%d parca; kapi 1 G0 / G1, kapi 2 G1)" % len(pieces),
          max(cres) < 1e-5, "fark %s" % ["%.1e" % v for v in cres])

    m = make(global_layers=1, gate=1, rand=False)
    m.train()
    opt = TR._optimizer(m, "normuon", 1e-2, False)[0]
    names = [n for n, _ in R.muon_params(m)]
    m.loss_per_target(batch)[0].mean().backward()
    gn = [float(blk.attn_gate.grad.abs().sum()) for blk in m.blocks]
    opt.step()
    moved = all(blk.attn_gate.abs().sum() > 0 and torch.isfinite(blk.attn_gate).all() for blk in m.blocks)
    check("gate: sifir baslangicta gradyan her bloga ulasir; attn_gate NorMuon grubunda (muon_params), adim kapiyi oynatir",
          all(g_ > 0 for g_ in gn) and all("blocks.%d.attn_gate" % i in names for i in range(3)) and moved,
          "grad %s" % ["%.2e" % g_ for g_ in gn])
    try:
        SentenceTransformer(d=32, layers=1, heads=4, attn_gate=2)
        small = False
    except AssertionError as e:
        small = "d 32 < 64" in str(e)
    w = {d_: tuple(SentenceTransformer(d=d_, layers=1, heads=4, attn_gate=2).blocks[0].attn_gate.shape) for d_ in (64, 768)}
    check("gate 2: girdi genisligi d // 64 (d64 1, d768 12 = onceki sabit 12), d < 64 DURUR (assert)",
          small and w == {64: (4, 1), 768: (4, 12)}, str(w))


TESTS = dict(layout=t_layout, flex=t_flex, learned=t_learned, global_=t_global, prefill=t_prefill,
             equiv=t_equiv, mask=t_mask, summaries_last=t_summaries_last, gqa=t_gqa, carry=t_carry, vocab=t_vocab, limit=t_limit, flex_ranges=t_flex_ranges,
             gate=t_gate)

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
