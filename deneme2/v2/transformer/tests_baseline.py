"""tests_baseline (V2) -- V2-transformer testleri (CPU; belge 20, 21).  common/data.build_batch ile sentetik hikayeler;
GPU / Drive yok.  Model Z dosyasindan import yok.

    python tests_baseline.py [--only mask,targets,loss,cache,generate,recipe,flex,imports]
"""
import ast
import math
import os
import sys
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

from baseline import BaselineTransformer  # noqa: E402
import data as D  # noqa: E402  (baseline common/'u yola ekledi)

RESULTS = []
STORIES = [[[3, 4, 5], [6, 7], [8, 9, 10, 11]], [[12], [13, 14, 15, 16, 17]], [[18, 19, 20, 21], [22]],
           [[23, 24], [25, 26, 27], [28], [29, 30]]]


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
            flat.append(D.EOS_ID) if s is st[-1] else None
        story.append(len(sent))
    return SimpleNamespace(stream=np.array(flat, np.int64), sent=np.array(sent, np.int64),
                           story=np.array(story, np.int64), n=len(stories),
                           sentences=lambda i: stories[i])


def tiny(seed=0, d=32, layers=2, heads=2):
    torch.manual_seed(seed)
    return BaselineTransformer(d, layers, heads).eval()


def logits_at_targets(model, batch):
    h = model._batch_hidden(batch)
    keep = batch.target >= 0
    return h[keep] @ model.E.weight.T


@torch.no_grad()
def t_mask():
    model = tiny()
    ts = token_stories(STORIES)
    batch = D.build_batch(ts, [[0, 1], [2, 3]], "transformer", row_len=64)
    lg = logits_at_targets(model, batch)
    alone = torch.cat([logits_at_targets(model, D.build_batch(ts, [[i]], "transformer", row_len=64)) for i in range(4)])
    d = float((lg - alone).abs().max())
    check("paketli satir = her hikaye tek basina (belge maskesi, konum hikaye basinda 0)", d < 1e-5,
          "en buyuk fark %.1e, %d hedef" % (d, len(lg)))
    first = batch.doc == 0
    pos_ok = all(torch.equal(batch.pos[r][batch.doc[r] == k], torch.arange(int((batch.doc[r] == k).sum())))
                 for r in range(2) for k in range(2))
    check("konum: her hikayede 0, 1, 2 ...", pos_ok and bool((batch.pos[first.any(1)][:, 0] == 0).all()))
    other = SimpleNamespace(**vars(batch))
    other.tokens = batch.tokens.clone()
    other.tokens[(batch.doc == 0) & (batch.kind == D.Kind.TOKEN)] = 99                # 1. hikayeyi boz
    h1, h2 = model._batch_hidden(batch), model._batch_hidden(other)
    d2 = float((h1[batch.doc == 1] - h2[batch.doc == 1]).abs().max())
    check("onceki hikaye degisince sonraki hikayenin cikisi ayni (sizinti yok)", d2 == 0.0, "fark %.1e" % d2)


def t_targets():
    ts = token_stories(STORIES)
    rows = [[0, 1], [2, 3]]
    a = D.build_batch(ts, rows, "transformer", row_len=64)
    b = D.build_batch(ts, rows, "model_z", row_len=64)
    same = torch.equal(a.target, b.target) and torch.equal(a.target_kind, b.target_kind) and torch.equal(a.doc, b.doc)
    end = a.kind == D.Kind.END
    tok_ok = torch.equal(a.tokens[~end], b.tokens[~end]) and bool((b.tokens[end] == 0).all()) \
        and bool((b.kind[end] == D.Kind.ZTOK).all()) and bool((a.tokens[end] == D.END_ID).all())
    check("iki duzende hedef ve hedef turu konum konum ayni (build_batch transformer / model_z)", same,
          "%d hedef" % int((a.target >= 0).sum()))
    check("iki duzende girdi yalniz END -> ZTOK (token 0) farkli", tok_ok)
    W = sum(len(s) for st in STORIES for s in st)
    S = sum(len(st) for st in STORIES)
    check("hedef sayisi = W + S + hikaye", int((a.target >= 0).sum()) == W + S + len(STORIES))


def t_loss():
    model = tiny()
    ts = token_stories(STORIES)
    batch = D.build_batch(ts, [[0, 1], [2, 3]], "transformer", row_len=64)
    nll, pred, tk = model.loss_per_target(batch)
    lg = logits_at_targets(model, batch)
    ref = torch.nn.functional.cross_entropy(lg, batch.target[batch.target >= 0], reduction="none")
    S = sum(len(st) for st in STORIES)
    counts = [int((tk == k).sum()) for k in (D.TargetKind.FIRST, D.TargetKind.MID, D.TargetKind.END, D.TargetKind.EOS)]
    check("loss_per_target = cross-entropy; turler FIRST = END = cumle, EOS = hikaye",
          float((nll - ref).abs().max().detach()) < 1e-6 and counts[0] == S and counts[2] == S and counts[3] == len(STORIES)
          and torch.equal(pred, lg.argmax(-1)), str(counts))
    model.train()
    nll, _, _ = model.loss_per_target(batch)
    nll.mean().backward()
    ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    check("dense yolda geri yayilim sonlu (dolgu satiri bos degil)", ok and math.isfinite(float(nll.mean())))


@torch.no_grad()
def t_cache():
    model = tiny()
    seq = [D.EOS_ID] + [t for s in STORIES[3] for t in s + [D.END_ID]]
    T = len(seq)
    full = model.hidden(torch.tensor([seq]), torch.arange(T)[None], None)[0] @ model.E.weight.T
    worst = 0.0
    for m in (1, 4):                                            # istem m token, sonra birer birer
        cache = dict(k=[None] * len(model.blocks), v=[None] * len(model.blocks), T=T)
        lg = [model._logits(model._step(torch.tensor([seq[:m]]), cache, 0))[0]]
        for i in range(m, T):
            lg.append(model._logits(model._step(torch.tensor([[seq[i]]]), cache, i))[0])
        worst = max(worst, float((torch.stack(lg) - full[m - 1:]).abs().max()))
    check("test_cached_logits_equal_full: onbellekli logit = tam hesap (fp32)", worst < 1e-5, "en buyuk fark %.1e" % worst)


def naive_generate(model, sents, max_sentences, max_tokens):
    """Onbelleksiz acgozlu referans: her adimda bastan tam hesap, generate ile ayni kural."""
    seq = [D.EOS_ID] + [t for s in sents for t in list(s) + [D.END_ID]]
    nxt = lambda: int((model.hidden(torch.tensor([seq]), torch.arange(len(seq))[None], None)[0, -1]  # noqa: E731
                       @ model.E.weight.T).argmax())
    gen, ended, eos = [], [], False
    while len(gen) < max_sentences:
        cur, done = [], False
        while len(cur) < max_tokens:
            w = nxt()
            if w == D.EOS_ID and not cur:
                eos = True
                break
            if w in (D.END_ID, D.EOS_ID):
                done = True
                break
            cur.append(w)
            seq.append(w)
        if eos:
            break
        gen.append(cur)
        ended.append(done)
        seq.append(D.END_ID)
    return gen, ended, eos


@torch.no_grad()
def t_generate():
    model = tiny()
    prompts = [STORIES[0][:1], STORIES[3]]
    got = model.generate(prompts, max_sentences=4, max_tokens=6)
    ref = [naive_generate(model, p, 4, 6) for p in prompts]
    check("generate (KV cache, acgozlu) = onbelleksiz tam hesap", got == ref,
          "cumle %s, END %s" % ([len(g[0]) for g in got], [g[1] for g in got]))
    lim = all(len(g) <= 4 and all(len(s) <= 6 for s in g) and len(e) == len(g) for g, e, _ in got)
    g1 = model.generate(prompts, 3, 5, torch.Generator().manual_seed(7))
    g2 = model.generate(prompts, 3, 5, torch.Generator().manual_seed(7))
    check("generate: cumle <= max_tokens, en cok max_sentences; ayni tohum ayni ornek", lim and g1 == g2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gb = model.generate(prompts, 2, 4)
    check("generate bf16 autocast altinda calisir (onbellek etkinlesme dtype'inda)", len(gb) == 2)
    script, fed = [5, D.EOS_ID, 6, 7, 9, D.EOS_ID], []       # cumle ici EOS -> END; 2 token.ta kesilir (9 okunmaz); basta EOS biter

    def scripted(tokens, cache, n):
        fed.append(tokens[0].tolist())
        out = torch.full((D.VOCAB,), -1e9)
        out[script[len(fed) - 1]] = 0
        return out[None]                                                  # _logits asagida birim: (1, V) logit
    import baseline                                                       # kural sinamasi _step'i sarar: eski yol
    model._step, model._logits, baseline.STATIC_DECODE = scripted, lambda h: h, False
    got = model.generate([STORIES[0][:1]], max_sentences=5, max_tokens=2)
    del model._step, model._logits
    baseline.STATIC_DECODE = True
    want_fed = [[D.EOS_ID] + STORIES[0][0] + [D.END_ID], [5], [D.END_ID], [6], [7], [D.END_ID]]
    check("generate kurali: cumle ici EOS cumleyi bitirir, max_tokens'ta kesilen kapanir (girdiye END), basta EOS biter",
          got == [([[5], [6, 7]], [True, False], True)] and fed == want_fed, "%s" % (got,))


def t_recipe():
    big = BaselineTransformer(512, 8, 8)
    n = sum(p.numel() for p in big.parameters())
    hidden = -(-int(8 * 512 / 3) // 8) * 8
    want = D.VOCAB * 512 + 8 * (4 * 512 * 512 + 3 * 512 * hidden + 2 * 512 + 2 * 64) + 512
    check("parametre (d 512, 8 katman, 8 head, VOCAB 50.258, tied) = %d" % want, n == want, "%d, SwiGLU %d" % (n, hidden))
    model = tiny()
    seen, lin = [], []
    model.blocks[0].q_norm.register_forward_pre_hook(lambda m, a: seen.append(a[0].dtype))
    model.blocks[0].qkv.register_forward_hook(lambda m, a, o: lin.append(o.dtype))
    ts = token_stories(STORIES)
    batch = D.build_batch(ts, [[0, 1]], "transformer", row_len=64)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        h = model._batch_hidden(batch)
    check("QK-norm fp32 (bf16 autocast altinda; qkv cikisi bf16)", seen == [torch.float32] and lin == [torch.bfloat16],
          "%s / %s" % (seen, lin))
    check("tied embedding (giris = cikis), bias yok",
          not any(n_.endswith("bias") for n_, _ in model.named_parameters()))


@torch.no_grad()
def t_flex():
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        print("ATLANDI flex: torch'ta yok")
        return
    model = tiny()
    ts = token_stories(STORIES)
    batch = D.build_batch(ts, [[0, 1], [2, 3]], "transformer", row_len=128)
    doc = batch.doc
    bm = create_block_mask(lambda b, h, q, kv: (doc[b, q] == doc[b, kv]) & (kv <= q), 2, None, 128, 128, device="cpu")
    a = model._batch_hidden(batch, bm)
    b = model._batch_hidden(batch)
    d = float((a - b).abs().max())
    check("belge maskesi: FlexAttention BlockMask (CPU ileri) = dense", d < 1e-4, "fark %.1e" % d)
    try:
        import recipe as R
    except ImportError:
        print("ATLANDI recipe: common/recipe.py yok")
        return
    dm = R.dense_mask(batch, R.document_mask)
    mine = (doc[:, :, None] == doc[:, None, :]) & torch.ones(128, 128, dtype=torch.bool).tril()
    c = model._batch_hidden(batch, R.block_mask(batch, R.document_mask))
    check("recipe.dense_mask / block_mask(document_mask) = modelin dense maskesi",
          torch.equal(dm, mine) and float((c - b).abs().max()) < 1e-4)


def t_imports():
    bad = []
    for f in ("baseline.py",) + (("generate_baseline.py",) if os.path.exists(os.path.join(HERE, "generate_baseline.py"))
                                 else ()):
        for node in ast.walk(ast.parse(open(os.path.join(HERE, f), encoding="utf-8").read())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            bad += ["%s: %s" % (f, x) for x in names if x and x.split(".")[0] in ("sentence", "sentence_z", "model_z")]
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "model_z" == node.value:
                bad.append("%s: model_z yolu" % f)
    check("Model Z dosyasindan import yok", not bad, ", ".join(bad))


@torch.no_grad()
def t_gqa():
    """kv_heads (GQA kiyasi): kv = heads bugunkuyle bit ayni; kv 2: blok = k / v tekrarli basvuru; onbellekli generate =
    onbelleksiz acgozlu."""
    torch.manual_seed(0)
    a = BaselineTransformer(32, 2, 4)
    torch.manual_seed(0)
    b = BaselineTransformer(32, 2, 4, kv_heads=4)
    same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    torch.manual_seed(0)
    m = BaselineTransformer(32, 2, 4, kv_heads=2).eval()
    blk = m.blocks[0]
    x = torch.randn(2, 9, 32)
    pos = torch.arange(9)[None].expand(2, 9)
    q, k, v = blk._qkv(x, pos)
    ref = blk._finish(x, torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                                                         v.repeat_interleave(2, 1), is_causal=True))
    gen = m.generate([STORIES[0][:1]], 3, 4)
    check("gqa: kv_heads = heads bit ayni; kv 2 blok = tekrarli basvuru; onbellekli generate = onbelleksiz",
          same and float((blk(x, pos, None) - ref).abs().max()) < 1e-5 and k.shape[1] == 2
          and gen[0] == naive_generate(m, STORIES[0][:1], 3, 4))
    from baseline import gqa_sdpa
    g = torch.Generator().manual_seed(3)
    kk, vv = torch.randn(3, 2, 7, 8, generator=g), torch.randn(3, 2, 7, 8, generator=g)
    worst = 0.0
    for tq in (1, 5):
        qq = torch.randn(3, 4, tq, 8, generator=g)
        for mm in (None, torch.rand(3, 1, tq, 7, generator=g) < 0.6, torch.rand(tq, 7, generator=g) < 0.6):
            if mm is not None:
                mm[..., 0] = True
            want = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=mm, enable_gqa=True)
            worst = max(worst, float((gqa_sdpa(qq, kk, vv, mm) - want).abs().max()))
    hn = m._step(torch.tensor([[D.EOS_ID] + STORIES[0][0] + [D.END_ID]]), dict(k=[None] * 2, v=[None] * 2, T=16), 0)
    cache = dict(k=[None] * 2, v=[None] * 2, T=16)
    seq = [D.EOS_ID] + STORIES[0][0] + [D.END_ID]
    m._step(torch.tensor([seq[:2]]), cache, 0)
    steps = [m._step(torch.tensor([[t]]), cache, 2 + i) for i, t in enumerate(seq[2:])]
    check("gqa_sdpa (belge 74): Tq 1 katlama, Tq > 1 genisletme = enable_gqa basvurusu; _step parca parca = tek gecis",
          worst < 1e-6 and float((steps[-1] - hn).abs().max()) < 1e-5, "en buyuk fark %.1e, adim %.1e" % (
              worst, float((steps[-1] - hn).abs().max())))


GROUPS = dict(mask=t_mask, targets=t_targets, loss=t_loss, cache=t_cache, generate=t_generate, recipe=t_recipe,
              flex=t_flex, imports=t_imports, gqa=t_gqa)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(GROUPS)
    for g in only:
        GROUPS[g]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d GECTI%s" % (len(RESULTS) - len(bad), len(RESULTS), ("  KALDI: " + "; ".join(bad)) if bad else ""))
    sys.exit(1 if bad else 0)
