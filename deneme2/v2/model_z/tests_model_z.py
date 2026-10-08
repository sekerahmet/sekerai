"""tests_model_z -- V2-Model Z testleri (CPU; belge 21, 22, 35, 43, 44).  GPU yok, Drive yok.  Formullu z testleri
(z, z_flat, direct, generate_longest, formullu onbellek) kaldirildi (belge 44); eski hali git etiketi
v2-before-formula-cleanup-20261007.

    python tests_model_z.py [--only layout,flex,learned,global_,prefill,equiv,bag,plan,mask,summaries_last,z_bow]
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
import recipe as R  # noqa: E402

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
    (b) ayni model ve batch: konum basina hidden, loss_per_target ve parametre gradyani duzenden bagimsiz (fp32 dense), plan
    (mid, glob her yerde) dahil."""
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
    for kw in (dict(global_layers=0), dict(global_layers=1), dict(layer_plan="loc1,mid1,glob1"),
               dict(layer_plan="glob1,loc1,glob1")):
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
          "gradyani duzenden bagimsiz (fp32 dense; G0, G1, mid, glob her yerde)",
          all(r[1] < 1e-5 and r[2] < 1e-5 and r[3] < 1e-5 for r in res),
          "; ".join("%s h %.1e nll %.1e grad %.1e" % r for r in res))


def _bow_brute(batch):
    """Kaba kuvvet: Z_k (duz konum) -> cumle k + 1'in TOKEN token'lari (sirali liste), satir / hikaye / cumle dongusuyle."""
    B, T = batch.kind.shape
    out = {}
    for r in range(B):
        for c in range(T):
            if int(batch.kind[r, c]) != ZTOK:
                continue
            d, k = int(batch.doc[r, c]), int(batch.sent[r, c])
            w = sorted(int(batch.tokens[r, j]) for j in range(T) if int(batch.kind[r, j]) == TOKEN
                       and int(batch.doc[r, j]) == d and int(batch.sent[r, j]) == k + 1)
            if w:
                out[r * T + c] = w
    return out


def _bow_of(tgt):
    return {int(z): sorted(tgt["word"][tgt["pz"] == i].tolist()) for i, z in enumerate(tgt["zpos"].tolist())}


def t_z_bow():
    """z_bow (belge 68 fikir 1): hedef = kaba kuvvet (sentetik dolgulu + Drive'da SS / FineWeb satirlari), summaries_last'ta
    da (ayni Z -> ayni torba); kayip summaries_last 0 / 1 ayni (fp32); gradyan yalniz hedefli Z satirlarina; z_bow_layer
    kurali; sonraki cumleyi degistirmek ozelligi degistirmez (hedef yalniz kayipta)."""
    from sentence import summaries_last, z_bow_loss, z_bow_targets
    rng = np.random.default_rng(8)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 12))] for _ in range(rng.integers(1, 6))]
               for _ in range(20)]
    batch = real_batch([list(range(i, i + 4)) for i in range(0, 20, 4)], 260, stories)
    sets = [("sentetik", batch)]
    G = "G:/Drive'ım"
    if os.path.isdir(G + "/v2"):
        for tag, root, ddir in (("SS valid", G + "/simplestories", G + "/v2/simplestories_gpt2"),
                                ("FineWeb valid", G + "/v2/fineweb_edu_s000", G + "/v2/fineweb_edu_s000")):
            ts = D.TokenStories(root, ddir, "valid")
            lens = ts.lengths()
            fit = np.nonzero(lens <= 2048)[0][:2000]
            ro, rs = D.pack_plan(lens[fit], 2048, 0, 1)
            sets.append((tag, D.build_batch(ts, [fit[rs[ro[r]:ro[r + 1]]].tolist() for r in range(3)], "model_z", "cpu",
                                            2048)))
    else:
        print("ATLANDI z_bow gercek satirlar: Drive yok", flush=True)
    ok, info = True, []
    for tag, b in sets:
        ref = _bow_brute(b)
        got = _bow_of(z_bow_targets(b))
        pb, perm = summaries_last(b)
        T = b.kind.shape[1]
        back = {int(perm.flatten()[z] + (z // T) * T): w for z, w in _bow_of(z_bow_targets(pb)).items()}  # yeni -> eski
        ok &= got == ref and back == ref
        info.append("%s %d Z" % (tag, len(ref)))
    check("z_bow: hedef (Z_k -> cumle k + 1'in token'lari, tekrarli) = kaba kuvvet; summaries_last'ta ayni; son Z hedefsiz",
          ok, "; ".join(info))
    torch.manual_seed(0)
    kinds = {}
    for kw in (dict(global_layers=0), dict(global_layers=1), dict(layer_plan="glob1,loc1,glob1"),
               dict(layer_plan="loc1,mid1,glob1"), dict(global_layers=3)):
        kinds[str(kw)] = SentenceTransformer(d=16, layers=3, heads=2, **kw).z_bow_layer()
    m = SentenceTransformer(d=32, layers=3, heads=2, global_layers=1)
    m.z_bow_norm = torch.nn.RMSNorm(32)
    torch.nn.init.normal_(m.z_bow_norm.weight, 1.0, 0.3)
    h0, xt0 = m._batch_hidden(batch, tap=True)
    l0, n0 = z_bow_loss(m, xt0, z_bow_targets(batch))
    pb, perm = summaries_last(batch)
    _, xt1 = m._batch_hidden(pb, tap=True)
    l1, n1 = z_bow_loss(m, xt1, z_bow_targets(pb))
    xg = xt0.detach().requires_grad_(True)
    tg = z_bow_targets(batch)
    z_bow_loss(m, xg, tg)[0].backward()
    rows = (xg.grad.abs().sum(-1) > 0).flatten().nonzero()[:, 0]
    ref_t = torch.zeros(1)
    man = []
    for z, w in _bow_brute(batch).items():
        r_, c_ = divmod(z, batch.kind.shape[1])
        lg = torch.log_softmax(torch.nn.functional.rms_norm(xt0[r_, c_], (32,), m.z_bow_norm.weight) @ m.E.weight.T, -1)
        man.append(float(-lg[w].mean()))
    alt = [[list(s_) for s_ in st] for st in stories]
    alt[0][-1] = [(t + 7) % D.END_ID for t in alt[0][-1]]
    b2 = real_batch([list(range(i, i + 4)) for i in range(0, 20, 4)], 260, alt)
    _, xt2 = m._batch_hidden(b2, tap=True)
    first = batch.doc[0] == 0
    lastc = int((first & (batch.sent[0] == int(batch.sent[0][first].max()))).nonzero()[0, 0])
    check("z_bow: kayip = elle (Z basina ortalama, Z'ler uzerinde ortalama); summaries_last 0 / 1 ayni; gradyan yalniz "
          "hedefli Z satirlarinda; ozellik sonraki cumleden bagimsiz; z_bow_layer = son glob olmayan blok %s" % kinds,
          abs(float(l0) - float(np.mean(man))) < 1e-5 and abs(float(l0) - float(l1)) < 1e-5 and n0 == n1 == len(man)
          and torch.equal(rows.sort().values, tg["zpos"].sort().values)
          and torch.equal(xt0[0, :lastc], xt2[0, :lastc])
          and kinds == {"{'global_layers': 0}": 2, "{'global_layers': 1}": 1, "{'layer_plan': 'glob1,loc1,glob1'}": 1,
                        "{'layer_plan': 'loc1,mid1,glob1'}": 1, "{'global_layers': 3}": None},
          "kayip %.6f / elle %.6f / sirali %.6f" % (float(l0), float(np.mean(man)), float(l1)))


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
        for src in ("common/data.py", "model_z/sentence.py", "model_z/sentence_z.py"):
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


BAG_STORIES = [[[3, 4, 3], [5, 4, 6], [7, 3, 8]], [[9], [9, 10], [11]], [[12, 13], [14]]]


def _bag_reference(rows, stories, core):
    """Dongulu basvuru: satir satir torbalar [(satir, hikaye, cumle k (-1 BOS), P kumesi, hedefler)]."""
    out = []
    for r, row in enumerate(rows):
        for di, i in enumerate(row):
            st = stories[i]
            for k in range(-1, len(st)):
                P = {t for s in st[:k + 1] for t in s if not core[t]}
                nxt = st[k + 1] + [D.END_ID] if k + 1 < len(st) else []
                tg = ([nxt[0]] + nxt[1:]) if nxt else [D.EOS_ID]
                out.append((r, di, k, P, tg))
    return out


def bag_model(k=7):
    """Kucuk Model Z + G, torbali (C = 4 + END + EOS, R = k - |C|), secici ve DIGER rastgele."""
    model = learned_model(global_layers=1)
    counts = np.zeros(D.EOS_ID + 1, np.int64)
    counts[[4, 9]] = 100
    counts[3:20] += 5
    core = R.core_ids(counts, 2)
    R.attach_bag(model, k, len(core))
    model.bag.fill(core, counts)
    torch.nn.init.normal_(model.bag.other, std=0.5)
    torch.nn.init.normal_(model.bag.q.weight, std=0.5)
    return model.eval()


def t_bag():
    """Ogrenen torba (belge 53-55): konum -> torba ve P_k donguyle ayni; ZTOK (0) / BOS P'ye girmez; sizinti yok (sonraki
    cumle degisince onceki torbalarin P'si, secici puani ve torbasi bit ayni); paketleme degismezligi; uretim yolu
    (SummaryCache token token ve prefill) = sinav yolu (output_logprobs, iki asamali; belge 55 K3)."""
    core = torch.zeros(D.VOCAB, dtype=torch.bool)
    core[[4, D.END_ID, D.EOS_ID]] = True
    rows, T = [[0, 1], [2]], 40
    b = D.build_batch(token_stories(BAG_STORIES), rows, "model_z", row_len=T)
    ids, br, bc = R.bag_index(b)
    pb, pw = R.bag_copy(b, ids, br, bc, core)
    P = [set(pw[pb == j].tolist()) for j in range(len(br))]
    ref = _bag_reference(rows, BAG_STORIES, core)
    keep = b.target >= 0
    got_t = [[] for _ in ref]
    for bag, y in zip(ids[keep].tolist(), b.target[keep].tolist()):
        got_t[bag].append(y)
    check("bag: konum -> torba ve P_k (cumle <= k, C disi) dongulu basvuruyla ayni (%d torba); ZTOK (0) / EOS P'de yok"
          % len(ref), len(ref) == len(br) and all(P[j] == x[3] and got_t[j] == x[4] and int(br[j]) == x[0]
                                                  and int(b.doc[br[j], bc[j]]) == x[1] for j, x in enumerate(ref))
          and not any({0, D.EOS_ID} & p for p in P))
    model = bag_model(12)                                                        # R 8: P kesilmez

    def sel_of(stories, rows_):
        bb = D.build_batch(token_stories(stories), rows_, "model_z", row_len=T)
        with torch.no_grad():
            return model.bag.batch_select(bb, model._batch_hidden(bb), model.E.weight)
    s0 = sel_of(BAG_STORIES, rows)
    s1 = sel_of([[[3, 4, 3], [5, 4, 6], [20, 21, 22]]] + BAG_STORIES[1:], rows)    # hikaye 0 cumle 2 degisti
    check("bag: sizinti yok -- cumle 2 degisince torba 0-2'nin P'si, secici puani ve torbasi bit ayni, torba 3'unku degisir",
          all(torch.equal(s0[k][:3], s1[k][:3]) for k in ("pbit", "score", "inbag"))
          and not torch.equal(s0["pbit"][3], s1["pbit"][3]) and not torch.equal(s0["score"][3], s1["score"][3]))
    s2 = sel_of(BAG_STORIES, [[1, 0], [2]])                                      # hikaye 0 satirda ikinci
    n0, n1 = len(BAG_STORIES[0]) + 1, len(BAG_STORIES[1]) + 1
    check("bag: paketleme degismezligi (hikaye 0 satir basinda / ikinci sirada): P ve torba ayni, puan <= 1e-5",
          torch.equal(s0["pbit"][:n0], s2["pbit"][n1:n1 + n0]) and torch.equal(s0["inbag"][:n0], s2["inbag"][n1:n1 + n0])
          and torch.allclose(s0["score"][:n0], s2["score"][n1:n1 + n0], atol=1e-5))
    model = bag_model()                                                          # R 3: P kesilir
    with torch.no_grad():
        bb = D.build_batch(token_stories(BAG_STORIES), [[0]], "model_z", row_len=T)
        h = model._batch_hidden(bb)
        n = int((bb.kind[0] != PAD).sum())
        want = R.output_logprobs(model, bb, h, torch.arange(n))
        cache = SummaryCache(model)
        got = [cache.logits]
        for s in BAG_STORIES[0]:
            got += [cache.append_token(t) for t in s] + [cache.close_sentence()]
        pre = SummaryCache(model)
        pre.prefill(BAG_STORIES[0][:2])
    zc = int((bb.kind[0] == ZTOK).nonzero()[1])
    check("bag: uretim yolu (SummaryCache token token ve prefill) = sinav yolu output_logprobs (iki asamali, konumun B_k'si)",
          float((torch.stack(got) - want).abs().max()) < 1e-4 and float((pre.logits - want[zc]).abs().max()) < 1e-4,
          "fark %.1e" % float((torch.stack(got) - want).abs().max()))


def t_plan():
    """layer_plan (belge 52): 'loc2,glob1' = global_layers 1 (bit); mid = hikaye hikaye dolgusuz basvuru (ozet satirlari
    BOS + Z_k, aralarinda causal, konum k; token'lar degismez); MID_PAD dolgusu sonucu degistirmez; sonraki cumleyi
    degistirmek onceki konumlari degistirmez (sizinti yok); uretim onbellegi mid'de DURUR."""
    import sentence as S
    from sentence import model_z_global_mask, story_positions
    rng = np.random.default_rng(3)
    stories = [[[int(x) for x in rng.integers(0, D.END_ID, rng.integers(1, 9))] for _ in range(rng.integers(2, 6))]
               for _ in range(12)]
    rows = [list(range(i, i + 4)) for i in range(0, 12, 4)]
    batch = real_batch(rows, 160, stories)
    B, T = batch.kind.shape

    def make(**kw):
        torch.manual_seed(0)
        return SentenceTransformer(d=32, layers=3, heads=2, **kw).eval()
    with torch.no_grad():
        same = torch.equal(make(layer_plan="loc2,glob1")._batch_hidden(batch), make(global_layers=1)._batch_hidden(batch))
    check("plan: 'loc2,glob1' = global_layers 1 (hidden bit duzeyinde)", same)
    m = make(layer_plan="loc1,mid1,glob1")
    read = _dense(model_z_read_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    glob = _dense(model_z_global_mask(batch.kind, batch.doc, batch.sent), B, T, "cpu")
    with torch.no_grad():
        got = m._batch_hidden(batch)
        x = m.E(torch.where(batch.kind == ZTOK, torch.full_like(batch.tokens, D.END_ID), batch.tokens))
        x = m.blocks[0](x, batch.pos, read)
        before = x.clone()
        for r in range(B):
            for d_ in batch.doc[r].unique().tolist():
                if d_ < 0:
                    continue
                p = (((batch.kind[r] == BOS) | (batch.kind[r] == ZTOK)) & (batch.doc[r] == d_)).nonzero()[:, 0]
                x[r, p] = m.blocks[1](before[r, p][None], batch.pos[r, p][None], None)[0]
        tok = ~((batch.kind == BOS) | (batch.kind == ZTOK))
        ref = m.norm(m.blocks[2](x, story_positions(batch.kind), glob))
        saved = S.MID_PAD
        S.MID_PAD = 1
        try:
            got1 = m._batch_hidden(batch)
        finally:
            S.MID_PAD = saved
    check("plan mid: = hikaye hikaye dolgusuz basvuru (ozet satirlari causal, konum k); token'lar mid'de degismez; "
          "MID_PAD 64 = 1", float((got - ref).abs().max()) < 1e-5 and torch.equal(x[tok], before[tok])
          and float((got - got1).abs().max()) < 1e-5, "fark %.1e" % float((got - ref).abs().max()))
    alt = [[list(s) for s in st] for st in stories]
    alt[0][-1] = [(t + 7) % D.END_ID for t in alt[0][-1]]
    b2 = real_batch(rows, 160, alt)
    with torch.no_grad():
        g2 = m._batch_hidden(b2)
    first = batch.doc[0] == 0
    last = int((first & (batch.sent[0] == int(batch.sent[0][first].max()))).nonzero()[0, 0])   # son cumlenin ilk konumu
    check("plan mid: hikayenin son cumlesini degistirmek ondan onceki konumlari degistirmez (sizinti yok)",
          torch.equal(got[0, :last], g2[0, :last]) and not torch.equal(got[0, last:], g2[0, last:]))
    try:
        SummaryCache(m)
        stopped = False
    except AssertionError:
        stopped = True
    check("plan mid: uretim onbellegi (SummaryCache) DURUR", stopped)
    m2 = make(layer_plan="loc1,mid1,loc1")                                       # glob'suz plan (belge 57 K2)
    with torch.no_grad():
        ref2 = m2.norm(m2.blocks[2](x, batch.pos, read))                         # x: mid'den sonraki basvuru (blok 0-1 ayni)
        got2 = m2._batch_hidden(batch)
        got3 = m2._batch_hidden(batch, read)                                     # egitim yolu: tek maske
    check("plan glob'suz (loc1,mid1,loc1): tek maske, dense ve egitim yolu = basvuru",
          m2.global_layers == 0 and not isinstance(m2.mask_fn, tuple) and float((got2 - ref2).abs().max()) < 1e-5
          and torch.equal(got2, got3), "fark %.1e" % float((got2 - ref2).abs().max()))
    m3 = make(layer_plan="glob1,loc1,glob1")                                     # glob her yerde (belge 60 B, 62)
    real = story_positions(batch.kind)
    with torch.no_grad():
        x3 = m3.E(torch.where(batch.kind == ZTOK, torch.full_like(batch.tokens, D.END_ID), batch.tokens))
        x3 = m3.blocks[0](x3, real, glob)
        x3 = m3.blocks[1](x3, batch.pos, read)
        ref3 = m3.norm(m3.blocks[2](x3, real, glob))
        got4 = m3._batch_hidden(batch)
        got5 = m3._batch_hidden(batch, (read, glob))
        g6 = m3._batch_hidden(b2)
    check("plan glob1,loc1,glob1: global_layers 2, ilk ve son katman tam causal + gercek konum = katman katman basvuru "
          "(dense ve egitim yolu); son cumleyi degistirmek onceki konumlari degistirmez",
          m3.global_layers == 2 and float((got4 - ref3).abs().max()) < 1e-5 and torch.equal(got4, got5)
          and torch.equal(got4[0, :last], g6[0, :last]) and not torch.equal(got4[0, last:], g6[0, last:]),
          "fark %.1e" % float((got4 - ref3).abs().max()))
    with torch.no_grad():
        keep = batch.target >= 0
        lg_full = (got4[keep] @ m3.E.weight.T)
        out = []
        for row in rows:
            for si in row:
                cache = SummaryCache(m3)
                out.append(cache.logits[None])
                for s_ in stories[si]:
                    out += [cache.append_token(t)[None] for t in s_] + [cache.close_sentence()[None]]
        pre = SummaryCache(m3)
        pre.prefill(stories[0][:2])
        seq = [D.EOS_ID] + [t for s_ in stories[0][:2] for t in s_ + [D.END_ID]]
        k_pre = len(seq) - 1                                                    # Z_2'nin hedef sirasi (hikaye 0)
    d3 = float((torch.cat(out) - lg_full).abs().max())
    check("plan glob1,loc1,glob1: SummaryCache (katman basina glob bayragi) adim adim logit ve prefill = tam ileri gecis",
          d3 < 1e-5 and float((pre.logits - torch.cat(out)[k_pre]).abs().max()) < 1e-5
          and cache.all_k[1] is None and cache.all_k[0].shape[2] == cache.t + 1, "fark %.1e" % d3)


TESTS = dict(layout=t_layout, flex=t_flex, learned=t_learned, global_=t_global, prefill=t_prefill,
             equiv=t_equiv, mask=t_mask, summaries_last=t_summaries_last, z_bow=t_z_bow, bag=t_bag, plan=t_plan)

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
