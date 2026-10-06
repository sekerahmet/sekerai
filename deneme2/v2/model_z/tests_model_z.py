"""tests_model_z -- V2-Model Z testleri (CPU; belge 21, 22).  Sentetik meaning gecici klasore yazilir; GPU / Drive yok.

    python tests_model_z.py [--only layout,cache,flex,z]
"""
import os
import sys
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
import sentence_z as SZ  # noqa: E402
from sentence import (BOS, END, EOS_T, END_T, FIRST, MID, PAD, TOKEN, ZTOK, SentenceTransformer, SummaryCache,  # noqa
                      _dense, model_z_mask, z_slot_positions)

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
        torch.save(dict(state={"source.weight": torch.randn(SZ.END_ID, 256)}), path)
    keys = SZ.build_keys(path, longest)
    return keys, SentenceTransformer(keys, d=d, layers=layers, heads=heads).eval()


def pack(rows, T):
    """Satirlar (her biri hikaye listesi) -> PackedBatch benzeri (belge 21 alanlari, rapor 22 duzeni).  pos bagimsiz
    sayimla kurulur (z_slot_positions'a karsi sinanir)."""
    B = len(rows)
    f = {k: torch.full((B, T), v, dtype=torch.long) for k, v in
         dict(tokens=0, kind=PAD, pos=0, doc=-1, sent=-1, target=-100, target_kind=-1).items()}
    slots_r, slots_c, zs = [], [], []
    for r, stories in enumerate(rows):
        c = 0
        for di, st in enumerate(stories):
            seq = [(SZ.EOS_ID, BOS, 0, -1)]                                      # (token, kind, pos, sent)
            for k, s in enumerate(st):
                seq += [(t, TOKEN, k + 1 + i, k) for i, t in enumerate(s)]
                seq += [(0, ZTOK, k + 1, -1)]
            tg = []
            for j, (t, kd, p, sn) in enumerate(seq):
                if kd == TOKEN:
                    nxt = seq[j + 1]
                    tg.append((nxt[0], MID) if nxt[1] == TOKEN else (SZ.END_ID, END_T))
                else:
                    k = 0 if kd == BOS else sum(1 for x in seq[:j + 1] if x[1] == ZTOK)
                    tg.append((st[k][0], FIRST) if k < len(st) else (SZ.EOS_ID, EOS_T))
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
    return SimpleNamespace(**f, z_slots=(torch.tensor(slots_r), torch.tensor(slots_c)), z_sentences=zs)


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
    batch = pack([STORIES[:2], STORIES[2:]], 40)
    check("z_slot_positions = bagimsiz sayim (BOS 0, Z_k k, token k-1+i; iki hikaye ayni satirda, dolgu)",
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
                tokens = torch.tensor([[SZ.EOS_ID] + [0] * k + w])
                kind = torch.tensor([[BOS] + [ZTOK] * k + [TOKEN] * len(w)])
                zvec = torch.zeros(1, T, keys["z"])
                zvec[0, 1:1 + k] = zst[:k]
                h = model.hidden(tokens, kind, torch.arange(T)[None], zvec, None)
                rows = list(range(k, T))
                lg_s.append(h[0, rows] @ model.E.weight.T)
                tg_s += ([w[0]] if w else [SZ.EOS_ID]) + w[1:] + ([SZ.END_ID] if w else [])
    lg_s = torch.cat(lg_s)
    d = (lg_full - lg_s).abs().max().item()
    check("test_z_layout_equals_short: logit ve hedefler kisa dizilerle ayni", torch.equal(tg_full, torch.tensor(tg_s))
          and d < 1e-5, "en buyuk fark %.1e, %d hedef" % (d, len(tg_s)))
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
    batch = pack([STORIES[:2], STORIES[2:]], 40)
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
    batch = pack([STORIES[:2], STORIES[2:]], 128)
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


TESTS = dict(z=t_z, layout=t_layout, cache=t_cache, flex=t_flex)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), " | HATA: " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)
