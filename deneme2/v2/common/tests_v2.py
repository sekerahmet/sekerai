"""tests_v2 -- V2 common/ testleri (CPU).  Gruplar: data (hizli; GPT-2 tokenizer'i yerel HF onbelleginden ya da Drive'dan,
yoksa tokenizer'li sinamalar ATLANIR), pack (sentetik), recipe (maske, WSD, gruplar, surdurme, hiz), metrics (sentence_repeat,
normalize_words, story_generation, exam_scores), integration (iki modelin loss_per_target'i ve recipe.output_loss'u
gercek build_batch ile; model dosyalari yalniz testte import edilir), train (train.py uctan uca, iki model, kucuk veri;
eski kimlik reddi dahil; ~1-2 dk), drive (valid akisi: V1 ile birebir esleme, okuma istemleri), tokens (data.token_counts).  Teshis araclari:
diag/tests_diag.py.

    python tests_v2.py [--only data,pack,recipe,metrics,integration,train,drive,tokens]
"""
import torch

import atexit  # noqa: E402
import atexit  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

if os.name == "nt":                     # EcoQoS: yoksa ~10 kat yavas (kullanici, 6 Ekim)
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    _k = ctypes.windll.kernel32
    _k.GetCurrentProcess.restype = wintypes.HANDLE
    _k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _st = _State(1, 0x1, 0)
    print("guc kisitlamasi (EcoQoS) kapali:", bool(_k.SetProcessInformation(_k.GetCurrentProcess(), 4, ctypes.byref(_st),
                                                                           ctypes.sizeof(_st))), flush=True)

GPT2_TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"   # Drive fingerprint.json
DRIVE = next((d for d in ("G:/Drive'ım", "/content/drive/MyDrive") if os.path.isdir(d)), None)
TMP = tempfile.mkdtemp(prefix="tests_v2_")
atexit.register(shutil.rmtree, TMP, True)          # train grubu ~200 MB birakiyordu
RESULTS = []

GOLDEN_TORCH = "2.14.0+cpu"     # belge 33 adim 4: v2-before-cleanup-20261006 --own_vocab 1 ile bit duzeyinde ayni olculdu
GOLDEN = {
 "transformer": {
  "first6": [
   10.8394,
   10.597,
   10.3123,
   10.0434,
   9.7698,
   9.4932
  ],
  "sha": "960799ace9cc5ec81208fa6002fd7ef35d087b2a3af6a93b582e63f8be0c96ea"
 },
 "model_z": {
  "first6": [
   10.8346,
   10.6259,
   10.3602,
   10.0859,
   9.8245,
   9.5621
  ],
  "sha": "654ac2820cf7d28f7e0c0bff79e4181e2b447bedc0b6454867385c71bf21ad68"
 }
}


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


def tokenizer_path():
    """GPT-2 tokenizer.json (Drive'daki ile ayni sha) ya da None."""
    cands = ([DRIVE + "/simplestories/gpt2/tokenizer.json"] if DRIVE else []) + glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/*/tokenizer.json"))
    for p in cands:
        if os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == GPT2_TOKENIZER_SHA:
            return p
    return None


def tiny_root(texts, tok_path):
    """Hikaye metinleri -> gecici kok (gpt2/valid.npy + tokenizer.json), V2 sinirlari yazilmis."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    root = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(root, "gpt2"))
    ids = [i for t in texts for i in tok.encode(t).ids + [D.EOS_ID]]
    np.save(os.path.join(root, "gpt2", "valid.npy"), np.array(ids, dtype=np.uint16))
    open(os.path.join(root, "gpt2", "tokenizer.json"), "wb").write(open(tok_path, "rb").read())
    D.build_boundaries(root, root, "valid")
    return tok, D.TokenStories(root, root, "valid")


def t_data():
    tp = tokenizer_path()
    if tp is None:
        print("ATLA data: GPT-2 tokenizer yok", flush=True)
        return
    texts = ['"Help! Help!" she cried. The dog said, "Woof!" Mr. Smith had 3.5 apples. Why? Alice asked.',
             '\n\nOnce upon a time, there was a cat.\n\nThe end.',
             'One sentence only']
    tok, st = tiny_root(texts, tp)
    words = [[D._WORD.findall(tok.decode(s.tolist())) for s in st.sentences(i)] for i in range(st.n)]
    want = [[['"', 'Help', '!', 'Help', '!', '"', 'she', 'cried', '.'], ['The', 'dog', 'said', ',', '"', 'Woof', '!', '"'],
             ['Mr', '.', 'Smith', 'had', '3', '.', '5', 'apples', '.'], ['Why', '?'], ['Alice', 'asked', '.']],
            [['Once', 'upon', 'a', 'time', ',', 'there', 'was', 'a', 'cat', '.'], ['The', 'end', '.']],
            [['One', 'sentence', 'only']]]
    check("cumle siniri: tirnak, Mr., ondalik, soz atfi; bosluk-yalniz ilk cumle sonrakine katiliyor", words == want,
          str(words) if words != want else "")
    whole = [np.concatenate(st.sentences(i)).tolist() for i in range(st.n)]
    raw = np.load(os.path.join(os.path.dirname(st.meta["stream"]), "valid.npy")).astype(np.int64)
    ends = np.flatnonzero(raw == D.EOS_ID)
    stories_raw = [raw[s:e].tolist() for s, e in zip(np.r_[0, ends[:-1] + 1], ends)]
    check("cumleler hikayenin butun token'larini sirayla, kayipsiz kapliyor", whole == stories_raw)
    check("boundaries.json: sayimlar ve en uzun cumle", st.meta["stories"] == 3 and st.meta["sentences"] == 8
          and st.max_sentence_tokens == max(len(s) for i in range(st.n) for s in st.sentences(i))
          and st.meta["merged_blank"] == 1, str({k: st.meta[k] for k in ("stories", "sentences", "merged_blank")}))
    b = D.build_batch(st, [[0, 1, 2]], "transformer")
    real = (b.kind != D.Kind.PAD).sum().item()
    check("lengths = build_batch'teki gercek boy", real == st.lengths().sum() and
          st.lengths("model_z").tolist() == st.lengths().tolist())


class _Synthetic:
    """Sentetik hikayeler -> TokenStories gibi: stream (hikaye sonu EOS), sent (N, 2), story (H+1), sentences(i)."""

    def __init__(self, stories):
        flat, sent, story = [], [], [0]
        for st in stories:
            for x in st:
                sent.append((len(flat), len(flat) + len(x)))
                flat += list(x)
            flat.append(D.EOS_ID)
            story.append(story[-1] + len(st))
        self.stream, self.sent, self.story = np.array(flat, np.int64), np.array(sent, np.int64), np.array(story, np.int64)
        self.n = len(stories)

    def sentences(self, i):
        return [self.stream[a:b] for a, b in self.sent[self.story[i]:self.story[i + 1]]]


def _reference_batch(stories, row_stories_list, layout, row_len):
    """build_batch'in dongulu basvurusu (adim 1 surumu): alanlar numpy."""
    K, TK = D.Kind, D.TargetKind
    B = len(row_stories_list)
    out = {f: np.full((B, row_len), v, np.int64) for f, v in (("tokens", 0), ("kind", K.PAD), ("pos", 0), ("doc", -1),
                                                               ("sent", -1), ("target", -100), ("target_kind", -1))}
    zr, zc, zs = [], [], []
    for r, row in enumerate(row_stories_list):
        c = 0
        for d, h in enumerate(row):
            ss = stories.sentences(int(h))
            seq = [D.EOS_ID] + [t for x in ss for t in list(x) + [D.END_ID]]
            T = len(seq)
            kd = [K.BOS] + [k for x in ss for k in [K.TOKEN] * len(x) + [K.END]]
            sn = [-1] + [k for k, x in enumerate(ss) for _ in range(len(x) + 1)]
            nxt = seq[1:] + [D.EOS_ID]
            tk = [TK.EOS if i == T - 1 else TK.END if x == D.END_ID else TK.FIRST if kk in (K.BOS, K.END) else TK.MID
                  for i, (x, kk) in enumerate(zip(nxt, kd))]
            if layout == "transformer":
                ps = list(range(T))
            else:
                ps = [0] + [q for k, x in enumerate(ss, 1) for q in list(range(k, k + len(x))) + [k]]
                for k, x in enumerate(ss):
                    zr.append(r)
                    zc.append(c + sum(len(y) + 1 for y in ss[:k + 1]))
                    zs.append(list(x))
                seq = [0 if x == D.END_ID else x for x in seq]
                kd = [K.ZTOK if x == K.END else x for x in kd]
            sl = slice(c, c + T)
            for f, v in (("tokens", seq), ("kind", kd), ("pos", ps), ("doc", d), ("sent", sn), ("target", nxt),
                         ("target_kind", tk)):
                out[f][r, sl] = v
            c += T
    return out, zr, zc, zs


def t_pack():
    rng = np.random.default_rng(0)
    L = rng.integers(20, 900, 3000)
    ro, rs = D.pack_plan(L, 2048, 0, 1)
    rows = [rs[ro[r]:ro[r + 1]] for r in range(len(ro) - 1)]
    check("pack_plan: her hikaye bir kez, satir <= 2048, hikaye bolunmez",
          sorted(rs.tolist()) == list(range(len(L))) and all(L[r].sum() <= 2048 for r in rows),
          "%d satir, doluluk %.3f" % (len(rows), L.sum() / (len(rows) * 2048)))
    ro2, rs2 = D.pack_plan(L, 2048, 0, 1)
    ro3, rs3 = D.pack_plan(L, 2048, 0, 2)
    roN, rsN = D.pack_plan(L[:50], 2048, None)
    check("pack_plan: ayni tohum/epok ayni plan, baska epok baska; seed None karistirmaz (ilk satir ilk hikayeyle)",
          np.array_equal(rs, rs2) and np.array_equal(ro, ro2) and not np.array_equal(rs, rs3)
          and sorted(rsN.tolist()) == list(range(50)) and rsN[0] == 0)
    E, P = D.EOS_ID, D.END_ID
    st = _Synthetic([[[10, 11, 12], [13], [14, 15]], [[20, 21]], [[30], [31, 32, 33]]])
    bt = D.build_batch(st, [[0, 1], [2]], "transformer", row_len=16)
    bz = D.build_batch(st, [[0, 1], [2]], "model_z", row_len=16)
    K, T = D.Kind, D.TargetKind
    want_tok = [E, 10, 11, 12, P, 13, P, 14, 15, P, E, 20, 21, P, 0, 0]
    want_tgt = [10, 11, 12, P, 13, P, 14, 15, P, E, 20, 21, P, E, -100, -100]
    want_kind = [K.BOS, 1, 1, 1, K.END, 1, K.END, 1, 1, K.END, K.BOS, 1, 1, K.END, K.PAD, K.PAD]
    want_tk = [T.FIRST, T.MID, T.MID, T.END, T.FIRST, T.END, T.FIRST, T.MID, T.END, T.EOS,
               T.FIRST, T.MID, T.END, T.EOS, -1, -1]
    want_zpos = [0, 1, 2, 3, 1, 2, 2, 3, 4, 3, 0, 1, 2, 1, 0, 0]
    ok = (bt.tokens[0].tolist() == want_tok and bt.target[0].tolist() == want_tgt and bt.kind[0].tolist() == want_kind
          and bt.target_kind[0].tolist() == want_tk and bt.pos[0].tolist()[:14] == list(range(10)) + list(range(4))
          and bz.pos[0].tolist() == want_zpos)
    check("build_batch: token, hedef, tur, konum (transformer hikaye ici, model_z mantiksal) elle yazilmisla ayni", ok)
    same = all(torch.equal(getattr(bt, f), getattr(bz, f)) for f in ("doc", "sent", "target", "target_kind", "story_ids"))
    endpos = bt.kind == K.END
    ztok = torch.equal(bz.kind == K.ZTOK, endpos) and bool((bz.tokens[endpos] == 0).all()) and torch.equal(
        bz.tokens[~endpos], bt.tokens[~endpos]) and torch.equal(bz.kind[~endpos], bt.kind[~endpos])
    check("iki duzen ayni boy ve hedef; model_z'de END'in yerinde ZTOK (token 0), baska fark yok (belge 22)",
          same and ztok)
    zr, zc = bz.z_slots
    ids, m = bz.z_sentences
    zs = [ids[i][m[i]].tolist() for i in range(len(ids))]
    zf = [bz.z_flat[0].tolist(), bz.z_flat[1].tolist()]
    nz = [x.tolist() for x in m.nonzero(as_tuple=True)]
    check("z_flat = z_sentences mask.nonzero (ayni sira, int64, CPU numpy'dan); transformer'da None", zf == nz
          and all(x.dtype == torch.int64 for x in bz.z_flat) and bt.z_flat is None, str(zf))
    check("ZTOK konumunun sent alani = kendi cumle numarasi (belge 35 Yol A okuma maskesi buna dayaniyor)",
          bz.sent[bz.kind == K.ZTOK].tolist() == [0, 1, 2, 0, 0, 1] and torch.equal(bz.sent[zr, zc], torch.tensor(
              [0, 1, 2, 0, 0, 1], dtype=bz.sent.dtype)), str(bz.sent[bz.kind == K.ZTOK].tolist()))
    check("z_slots ZTOK konumlarini, z_sentences o Z'nin cumlesini veriyor (cumle sirasiyla)",
          all(bz.kind[r, c] == K.ZTOK for r, c in zip(zr.tolist(), zc.tolist()))
          and zs == [[10, 11, 12], [13], [14, 15], [20, 21], [30], [31, 32, 33]] and bt.z_slots is None)
    rng = np.random.default_rng(1)
    st = _Synthetic([[rng.integers(0, 50000, int(rng.integers(1, 30))).tolist() for _ in range(int(rng.integers(1, 40)))]
                     for _ in range(120)])
    ro, rs = D.pack_plan(1 + np.add.reduceat(st.sent[:, 1] - st.sent[:, 0] + 1, st.story[:-1]))
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(min(8, len(ro) - 1))]
    ok = True
    for lay in ("transformer", "model_z"):
        b = D.build_batch(st, rows, lay)
        ref, zr_, zc_, zs_ = _reference_batch(st, rows, lay, D.ROW_LEN)
        ok &= all(np.array_equal(getattr(b, f).numpy(), ref[f]) for f in ref)
        if lay == "model_z":
            ids, m = b.z_sentences
            ok &= b.z_slots[0].tolist() == zr_ and b.z_slots[1].tolist() == zc_ and [
                ids[i][m[i]].tolist() for i in range(len(ids))] == zs_
            ok &= all(torch.equal(x, y) for x, y in zip(b.z_flat, m.nonzero(as_tuple=True)))
    check("build_batch (vektorel) = dongulu basvuru, 120 rastgele hikaye, 8 satir, iki duzen; z_flat = mask.nonzero", ok)
    check("doc / sent / story_ids", bt.doc[0].tolist() == [0] * 10 + [1] * 4 + [-1, -1]
          and bt.sent[0].tolist() == [-1, 0, 0, 0, 0, 1, 1, 2, 2, 2, -1, 0, 0, 0, -1, -1]
          and bt.story_ids.tolist() == [[0, 1], [2, -1]])


def t_drive():
    """valid akisi: V2 sinirlari V1 kelime cumleleriyle birebir; okuma istemleri V1 metniyle ayni.  (~2-3 dk)"""
    if DRIVE is None:
        print("ATLA drive: Drive yok", flush=True)
        return
    from tokenizers import Tokenizer
    ss, zr = DRIVE + "/simplestories", DRIVE + "/model_z/simplestories_full"
    out = os.path.join(TMP, "valid_bounds")
    meta = D.build_boundaries(ss, out, "valid")
    st = D.TokenStories(ss, out, "valid")
    tok = Tokenizer.from_file(ss + "/gpt2/tokenizer.json")
    vocab = json.load(open(zr + "/ss_vocab.json", encoding="utf-8"))
    ix = {w: i for i, w in enumerate(vocab)}
    texts = tok.decode_batch([np.asarray(st.stream[s:t]).tolist() for s, t in st.sent])
    v2 = [[ix.get(w, 0) for w in D._WORD.findall(t.strip())] for t in texts]
    v1_ids = np.load(zr + "/ss_exam_story_ids.npy")
    v1_so = np.load(zr + "/ss_exam_story_sentence_offsets.npy")
    v1_ho = np.load(zr + "/ss_exam_story_offsets.npy")
    v1 = [v1_ids[v1_so[k]:v1_so[k + 1]].tolist() for k in range(len(v1_so) - 1)]
    check("V1 esleme: valid'in butun cumleleri V1 kelime cumleleriyle birebir, hikaye sinirlari ayni",
          v2 == v1 and np.array_equal(st.story, v1_ho) and st.n == 21371,
          "%d cumle, %d hikaye, katilan %d, en uzun %d token" % (len(v2), st.n, meta["merged_blank"],
                                                                  st.max_sentence_tokens))
    rp = json.load(open(os.path.join(HERE, "reading_prompts.json"), encoding="utf-8"))["prompts"]
    ok = len(rp) == 10
    for r in rp:
        mine = [D._WORD.findall(tok.decode(s.tolist()).strip()) for s in st.sentences(r["story"])[:r["sentences"]]]
        first = int(v1_ho[r["story"]])
        theirs = [[vocab[w] for w in v1[k]] for k in range(first, first + r["sentences"])]
        ok &= mine == theirs
    check("reading_prompts.json: 10 istem, metinleri V1 sinav hikayeleriyle ayni", ok,
          "; ".join("%s: %s" % (r["label"], " ".join(D._WORD.findall(tok.decode(st.sentences(r["story"])[0].tolist())))[:40])
                    for r in rp[:3]))


def _batch_two_rows():
    st = _Synthetic([[[10, 11, 12], [13], [14, 15]], [[20, 21]], [[30], [31, 32, 33]]])
    return st, D.build_batch(st, [[0, 1], [2]], "transformer", row_len=24)


def t_recipe():
    import torch.nn.functional as F
    import recipe as R
    st, b = _batch_two_rows()
    K = D.Kind
    m = R.dense_mask(b, R.document_mask)
    B, T = b.kind.shape
    ok = True
    for r in range(B):
        for q in range(T):
            for kv in range(T):
                pq, pk = b.kind[r, q] == K.PAD, b.kind[r, kv] == K.PAD
                want = (kv <= q) and ((pq and pk) or (not pq and not pk and b.doc[r, q] == b.doc[r, kv]))
                ok &= bool(m[r, q, kv]) == bool(want)
    check("dense_mask(document_mask) = elle: causal, ayni hikaye, dolgu yalniz onceki dolguyu goruyor; bos satir yok",
          ok and bool(m.any(-1).all()))

    def sentence_mask(kind, doc, sent):                                   # Model Z benzeri: ozet ya da ayni cumle
        def mod(b_, h, q, kv):
            summ = (kind[b_, kv] == K.BOS) | (kind[b_, kv] == K.END)
            return (doc[b_, q] == doc[b_, kv]) & (kv <= q) & (summ | (sent[b_, q] == sent[b_, kv]))
        return mod
    m2 = R.dense_mask(b, sentence_mask)
    check("dense_mask maske fonksiyonunu arguman aliyor (cumle maskesi daha seyrek, dolgu kurali ayni)",
          bool((m2 <= m).all()) and not torch.equal(m2, m) and bool(m2.any(-1).all()))
    q, k, v = (torch.randn(B, 2, T, 8) for _ in range(3))
    flex_ok = True
    try:
        from torch.nn.attention.flex_attention import flex_attention
        for fn in (R.document_mask, sentence_mask):
            out = flex_attention(q, k, v, block_mask=R.block_mask(b, fn))
            ref = F.scaled_dot_product_attention(q, k, v, attn_mask=R.dense_mask(b, fn)[:, None])
            flex_ok &= (out - ref).abs().max().item() < 1e-5
    except Exception as e:  # noqa: BLE001
        flex_ok, ref = False, e
    check("block_mask (FlexAttention, CPU ileri, fp32) = SDPA dense_mask, iki maske fonksiyonu", flex_ok)

    total, peak = 1000, 1e-3
    lr = [R.wsd_lr(s, total, peak) for s in range(total)]
    down = total - round(0.2 * total)
    check("wsd_lr: isinma 10 adim, sabit tepe, inisin ilk adimi tepe, son %20 dogrusal, son adim > 0",
          lr[0] == peak / 10 and lr[9] == peak and lr[down - 1] == peak and lr[down] == peak
          and all(a > c for a, c in zip(lr[down:], lr[down + 1:])) and 0 < lr[-1] < peak / 100
          and abs(lr[down + 100] - peak * (total - down - 100) / (total - down)) < 1e-15)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Embedding(50, 8)
            self.z_in = torch.nn.Linear(16, 8, bias=False)
            self.norm = torch.nn.RMSNorm(8)
            self.lin = torch.nn.Linear(8, 8)
    tm = Tiny()
    g = R.param_groups(tm)
    dec, nod = {id(p) for p in g[0]["params"]}, {id(p) for p in g[1]["params"]}
    check("param_groups: z_in ve lin.weight decay'li; embedding, norm, bias decay'siz",
          dec == {id(tm.z_in.weight), id(tm.lin.weight)} and nod == {id(tm.E.weight), id(tm.norm.weight), id(tm.lin.bias)}
          and g[0]["weight_decay"] == 0.1 and g[1]["weight_decay"] == 0.0)

    def run(steps, save_at=None, dir_=None, load=False):
        torch.manual_seed(0)
        model = Tiny()
        opt = torch.optim.AdamW(R.param_groups(model), lr=1e-2)
        s0 = 0
        if load:
            s0 = R.Checkpoint.load(dir_, model, opt)["step"]
        for s in range(s0, steps):
            for gr in opt.param_groups:
                gr["lr"] = R.wsd_lr(s, steps, 1e-2)
            x = torch.randn(4, 16)                                       # RNG: surdurmede geri konmali
            loss = model.lin(model.norm(model.z_in(x))).pow(2).mean() + model.E(torch.tensor([1, 2])).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if s + 1 == save_at:
                R.Checkpoint.save(dir_, model, opt, s + 1, dict(seed=0, epoch=1), [dict(step=s + 1)], dict(a=1))
                return None
        return model
    full = run(10)
    d_ = os.path.join(TMP, "ckpt")
    run(10, save_at=4, dir_=d_)
    resumed = run(10, dir_=d_, load=True)
    meta = torch.load(os.path.join(d_, "checkpoint.pt"), weights_only=False)
    check("Checkpoint: 4 + 6 adim = kesintisiz 10 adim (agirlik bit duzeyinde, RNG dahil); plan / gecmis / args yazili",
          all(torch.equal(a, c) for a, c in zip(full.state_dict().values(), resumed.state_dict().values()))
          and meta["plan"] == dict(seed=0, epoch=1) and meta["history"] == [dict(step=4)])

    class OnDevice:                                                     # vekil: CPU'da olmayan RNG tensoru (CUDA yok)
        def __init__(self, t):
            self.t = t

        def cpu(self):
            return self.t
    load = torch.load

    def load_moved(*a, **k):
        pack = load(*a, **k)
        pack["rng"]["cpu"] = OnDevice(pack["rng"]["cpu"])
        return pack
    torch.load = load_moved
    try:
        moved = run(10, dir_=d_, load=True)
    finally:
        torch.load = load
    check("Checkpoint.load: baska cihazdan gelmis RNG durumu .cpu() ile geri konur, surdurme yine kesintisiz (vekil; "
          "CUDA RNG listesi CPU'da sinanamaz)", all(torch.equal(a, c) for a, c in zip(
              full.state_dict().values(), moved.state_dict().values())) and _raises(TypeError, torch.set_rng_state,
                                                                                     OnDevice(torch.get_rng_state())))
    import time as _t
    sw = R.SpeedWindow()
    sw.start(5)
    _t.sleep(0.2)
    r = sw.stop(15, 1000)
    check("SpeedWindow: adim sayisi, sure, token / sn", r["steps"] == 10 and 0.18 < r["seconds"] < 0.5
          and abs(r["tokens_per_sec"] * r["seconds"] / 1000 - 1) < 0.01, str(r))


def t_metrics():
    import metrics as M
    p = [["The", "cat", "sat", "."]]
    g = [["The", "cat", "sat", "."], ["A", "dog", "."], ["A", "dog", "."], [], ["A", "dog", "."]]
    e = [True, True, False, True, True]
    check("sentence_repeat: yalniz END ile biten bos olmayan sayilir; kesik cumle gecmise girer",
          M._repeat_counts(p, g, e) == (2, 3) and M.sentence_repeat(p, g, e) == 2 / 3
          and M.sentence_repeat(p, [], []) is None)
    tp = tokenizer_path()
    if tp is None:
        print("ATLA metrics (tokenizer'li kisim): GPT-2 tokenizer yok", flush=True)
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tp)
    ids = tok.encode('"Hi!" she said.\n\nThe end.').ids
    check("normalize_words: V1 kelimeleri (bosluk / satir sonu yok)", M.normalize_words(ids, tok) == [
        '"', 'Hi', '!', '"', 'she', 'said', '.', 'The', 'end', '.'])
    s1, s2 = tok.encode("The cat sat.").ids, tok.encode(" A dog ran.").ids

    class Fake:
        def generate(self, prompts, max_sentences, max_tokens, generator):
            return [([s1, s2, s2], [True, True, True], True), ([s2, s2[:2]], [True, False], False)]
    texts, res = M.story_generation(Fake(), [[s1], [s2]], "greedy", 0, tokenizer=tok)
    check("story_generation: sozlesmeyi cagiriyor, olculer ve V1 bicimi metin", res["sentence_repeat"] == round(3 / 4, 4)
          and res["counted"] == 4 and res["eos_rate"] == 0.5 and res["no_end"] == 0.2 and res["sentences"] == 2.5
          and texts[0]["story"] == "The cat sat .\nA dog ran .\nA dog ran ." and texts[0]["prompt"] == "The cat sat .",
          str(res))
    enc = lambda lines: [tok.encode(" " + x).ids for x in lines]  # noqa: E731
    # gercek: v2_tf_d512_l8_lr1e-3_20261006_103227_sweep okuma 3 (kisaltilmis; aslinda 54'ten sonra 14 kez ayni cumle)
    loop_p = enc(["Mysterious woods are always calling to me .", "The trees stand tall , their leaves whispering secrets ."])
    loop_g = enc(["I will feel proud .", "I will be proud of her .", "I will know she is brave and kind .",
                  "I will tell her about the girl and the girl .", "I will tell her about the girl and the girl .",
                  "I will know she is brave and kind .", "I will be proud of her .", "I will be proud of her .",
                  "I will be proud of her ."])
    # gercek: v2_mz_d512_l8_lr5e-4_20261006_112843_sweep okuma 4, cumle 47-56 (51 = 54 = 48, 55 = 52; son 5'in 3'u yeni)
    spread_g = enc(["As they walked , the boy felt a change inside .", "He had faced his fears and helped a friend .",
                    "The giant creature smiled and said , \" You are brave . You are brave . \"",
                    "The boy smiled , feeling proud .", "He had faced his fears and helped a friend .",
                    "The boy knew he would always remember this adventure .",
                    "As they walked back , the boy felt different .", "He had faced his fears and helped a friend .",
                    "The boy knew he would always remember this adventure .",
                    "He would always remember the giant and the giant ."])
    spread_p = enc(["Near a tall mountain , a small fox felt cold and hungry ."])

    class Real:
        def generate(self, prompts, max_sentences, max_tokens, generator):
            return [(loop_g, [True] * len(loop_g), True), (spread_g, [True] * len(spread_g), True)]
    _, res = M.story_generation(Real(), [loop_p, spread_p], "greedy", 0, tokenizer=tok, labels=["okuma 3", "okuma 4"])
    check("story_loop (gercek metin): son 5 cumlesi tekrar olan hikaye 1, daginik tekrarli hikaye 0; word_loop ikisinde 0 "
          "(cumle ici tanim donguyu gormez); istem basina sentence_repeat",
          res["story_loop"] == 0.5 and res["story_loop_prompts"] == ["okuma 3"] and res["word_loop"] == 0
          and res["sentence_repeat_by_prompt"] == {"okuma 3": round(5 / 9, 4), "okuma 4": round(3 / 10, 4)}, str(res))
    w = lambda *xs: [x.split() for x in xs]  # noqa: E731
    cut = w("the the the the", "the the the the", "the the the the", "the the the the", "the the the the")
    check("story_loop kenar: kesik (END'siz) ayni cumleler kuyrukta sayilir; kuyrukta bos cumle -> 0; 5'ten az cumle -> 0; "
          "istemdeki cumleyi tekrar sayar",
          M._story_loop(w("a b ."), w("the the the the") + cut) == 1 and M._story_loop([], w("a .", "a .", "a .", "a .",
                                                                                        "a .") + [[]]) == 0
          and M._story_loop(w("a ."), w("a .", "a .", "a .", "a .")) == 0
          and M._story_loop(w("a .", "b ."), w("a .", "b .", "a .", "b .", "a .")) == 1)
    st, _ = _batch_two_rows()
    nb = np.array([40, 10, 30])

    class FakeLoss:
        def loss_per_target(self, batch):
            keep = batch.target >= 0
            tk, tgt = batch.target_kind[keep], batch.target[keep]
            nll = torch.full(tgt.shape, 2.0)
            nll[tk == D.TargetKind.EOS] = 5.0
            pred = torch.where(tk == D.TargetKind.MID, tgt, torch.full_like(tgt, -1))
            return nll, pred, tk
    plan = (np.array([0, 2, 3]), np.array([0, 1, 2]))
    r = M.exam_scores(FakeLoss(), st, plan, nb, "transformer", batch_rows=1)
    n_all, n_eos, n_mid = 21, 3, 6                                       # 3 hikaye: 10 + 4 + 7 hedef (elle)
    want_bpb = (2.0 * (n_all - n_eos)) / math.log(2) / 80
    check("exam_scores: kayip, acc_token (MID dogru), eos / end, bpb (EOS haric) / bpb_eos formulu",
          r["by_kind"]["eos"]["n"] == n_eos and r["by_kind"]["mid"]["n"] == n_mid and r["acc_token"] == round(
              n_mid / (n_mid + r["by_kind"]["first"]["n"]), 4) and r["eos_ok"] == 0 and r["bits_per_byte"] == round(
              want_bpb, 4) and r["bits_per_byte_eos"] == round((2.0 * (n_all - n_eos) + 5.0 * n_eos) / math.log(2) / 83, 4),
          str(r))


def t_integration():
    """Iki modelin loss_per_target'i GERCEK build_batch ciktisiyla (kural, 6 Ekim): sozlesme, maske argumani, exam_scores.
    Model dosyalari yalniz burada import edilir (common/ kodu import etmez)."""
    import traceback
    import metrics as M
    import recipe as R
    root = os.path.dirname(HERE)
    rng = np.random.default_rng(2)
    st = _Synthetic([[rng.integers(0, 50000, int(rng.integers(1, 7))).tolist() for _ in range(int(rng.integers(1, 6)))]
                     for _ in range(12)])
    L = 1 + np.add.reduceat(st.sent[:, 1] - st.sent[:, 0] + 1, st.story[:-1])
    ro, rs = D.pack_plan(L, 128, None)
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(len(ro) - 1)]
    nb = np.full(st.n, 50)
    models = []
    try:
        sys.path.insert(0, os.path.join(root, "transformer"))
        from baseline import BaselineTransformer
        torch.manual_seed(0)
        models.append(("transformer", BaselineTransformer(d=16, layers=1, heads=2).eval(), R.document_mask))
    except Exception:  # noqa: BLE001
        check("entegrasyon: transformer modeli kurulur", False, traceback.format_exc(limit=1).splitlines()[-1])
    try:
        sys.path.insert(0, os.path.join(root, "model_z"))
        import sentence_z as SZ
        from sentence import SentenceTransformer, model_z_mask
        torch.manual_seed(0)
        models.append(("model_z", SentenceTransformer(SZ.build_keys(8, 8), d=16, layers=1, heads=2).eval(),
                       model_z_mask))
    except Exception:  # noqa: BLE001
        check("entegrasyon: Model Z modeli kurulur", False, traceback.format_exc(limit=1).splitlines()[-1])
    for layout, model, mask_fn in models:
        try:
            with torch.no_grad():
                b = D.build_batch(st, rows[:2], layout, row_len=128)
                nll, pred, tk = model.loss_per_target(b)
                nll2, _, _ = model.loss_per_target(b, R.dense_mask(b, mask_fn))
            k = int((b.target >= 0).sum())
            ok = len(nll) == len(pred) == len(tk) == k and bool(torch.isfinite(nll).all()) and torch.equal(
                tk, b.target_kind[b.target >= 0]) and (nll - nll2).abs().max().item() < 1e-5
            check("entegrasyon %s: loss_per_target(build_batch) sozlesmesi; recipe.dense_mask(maske fonksiyonu) = "
                  "modelin kendi maskesi" % layout, ok, "hedef %d, fark %.1e" % (k, (nll - nll2).abs().max().item()))
            e = M.exam_scores(model, st, (ro, rs), nb, layout, batch_rows=2)
            check("entegrasyon %s: exam_scores uctan uca" % layout, math.isfinite(e["loss"]) and e["stories"] == st.n,
                  "kayip %.3f bpb %.3f" % (e["loss"], e["bits_per_byte"]))
            dense = R.dense_mask(b, mask_fn)
            ws = (model.E.weight, model.blocks[0].qkv.weight)
            la = R.output_loss(model._batch_hidden(b, dense).flatten(0, 1), model.E.weight, b.target.flatten())
            ga = torch.autograd.grad(la, ws)
            lb = model.loss_per_target(b, dense)[0].mean()
            gb = torch.autograd.grad(lb, ws)
            rel = max(((x - y).norm() / y.norm()).item() for x, y in zip(ga, gb))
            rel_loss = abs(la.item() - lb.item()) / abs(lb.item())
            check("entegrasyon %s: output_loss = loss_per_target ortalamasi (fp32 CPU; kayip ve E / qkv gradyani)"
                  % layout, rel_loss < 1e-6 and rel < 1e-5,
                  "kayip goreli farki %.1e, gradyan goreli %.1e" % (rel_loss, rel))
            if layout == "model_z":
                check("entegrasyon model_z: z_flat ile _batch_hidden + geri yayilim torch.nonzero'suz (GPU senkronu "
                      "yok; vekil: nonzero yasak); z_flat'siz batch DURUR (sessiz geri donus yok)", *_no_nonzero(model, b, dense))
        except Exception:  # noqa: BLE001
            check("entegrasyon %s: loss_per_target(build_batch)" % layout, False,
                  traceback.format_exc(limit=2).splitlines()[-1])


def _no_nonzero(model, b, dense):
    """torch.nonzero / Tensor.nonzero yasakken model_z _batch_hidden + output_loss geri yayilimi -> (gecti, bilgi)."""
    import dataclasses
    import recipe as R
    real = (torch.Tensor.nonzero, torch.nonzero)

    def banned(*a, **k):
        raise AssertionError("nonzero cagrildi (GPU senkronu)")
    model.zero_grad(set_to_none=True)
    torch.Tensor.nonzero, torch.nonzero = banned, banned
    try:
        R.output_loss(model._batch_hidden(b, dense).flatten(0, 1), model.E.weight, b.target.flatten()).backward()
        ok, info = all(p.grad is not None for p in model.parameters()), "butun gradyanlar var"
    except AssertionError as e:
        ok, info = False, str(e)
    finally:
        torch.Tensor.nonzero, torch.nonzero = real
    stops = _raises(Exception, model._batch_hidden, dataclasses.replace(b, z_flat=None), dense)
    return ok and stops, info + ("" if stops else " | z_flat=None ile kostu")


def _train_root(tok_path):
    """Kucuk SS benzeri veri, gercek boru hattiyla: gpt2/{train,valid}.npy + valid_bytes + tokenizer, build_boundaries,
    train_pack_plan_e1.npz (satir 64), exam_pack_plan.npz, 3 istemlik okuma dosyasi.  -> (stream koku, data klasoru,
    istem dosyasi)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    names, things = ["Lily", "Tom", "Mia", "Sam"], ["ball", "kite", "cat", "box"]
    story = lambda i: "%s had a %s. %s liked the %s. The end." % (  # noqa: E731
        names[i % 4], things[i // 4 % 4], names[i % 4], things[i // 4 % 4])
    root = tempfile.mkdtemp(dir=TMP)
    data = os.path.join(root, "v2")
    os.makedirs(os.path.join(root, "gpt2"))
    for split, n in (("train", 64), ("valid", 8)):
        texts = [story(i + (100 if split == "valid" else 0)) for i in range(n)]
        np.save(os.path.join(root, "gpt2", split + ".npy"),
                np.array([i for t in texts for i in tok.encode(t).ids + [D.EOS_ID]], dtype=np.uint16))
        if split == "valid":
            np.save(os.path.join(root, "gpt2", "valid_bytes.npy"), np.array([len(t.encode()) for t in texts]))
    open(os.path.join(root, "gpt2", "tokenizer.json"), "wb").write(open(tok_path, "rb").read())
    for split in ("train", "valid"):
        D.build_boundaries(root, data, split)
    tr, va = D.TokenStories(root, data, "train"), D.TokenStories(root, data, "valid")
    ro, rs = D.pack_plan(tr.lengths(), 64, 0, 1)
    np.savez(os.path.join(data, "train_pack_plan_e1.npz"), row_offsets=ro, row_stories=rs, seed=0, epoch=1, row_len=64)
    ro, rs = D.pack_plan(va.lengths(), 64, None)
    np.savez(os.path.join(data, "exam_pack_plan.npz"), row_offsets=ro, row_stories=rs.astype(np.int32), seed=-1, epoch=0,
             row_len=64, exam_set_sha256="test")
    prompts = os.path.join(root, "reading_prompts.json")
    json.dump(dict(prompts=[dict(label="okuma 1", story=0, sentences=1, decode="greedy"),
                            dict(label="okuma 2", story=1, sentences=2, decode="greedy"),
                            dict(label="okuma 3", story=2, sentences=1, decode="sample")]), open(prompts, "w"))
    return root, data, prompts


def t_train():
    """train.py uctan uca (CPU, d 16, 1 katman, gercek build_batch / modeller / exam_scores / story_generation): kayip
    duser, ilk adim kaybi bagimsiz hesapla ayni, lr = wsd_lr, results / samples alanlari, kesilip surdurulen = kesintisiz
    (bit duzeyinde), uzatma = bastan uzun kosu, bitmis kosu durur, ayar farki durur, --steps, yerel kopya sha'si."""
    import traceback
    import recipe as R
    import train as TR
    tp = tokenizer_path()
    if tp is None:
        print("ATLA train: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = _train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=3, max_tokens=4)
    base = ["--data", data, "--stream", root, "--device", "cpu", "--d", "16", "--layers", "1", "--heads", "2",
            "--lr", "1e-2", "--checkpoint_minutes", "0"]                # 0: her gunluk sinirinda kayit
    out = lambda name: os.path.join(TMP, "runs", name)  # noqa: E731
    state = lambda o: torch.load(os.path.join(o, "agent.pt"), weights_only=False)["state"]  # noqa: E731
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()  # noqa: E731
    exits = lambda argv: _raises(SystemExit, TR.main, argv)  # noqa: E731
    try:
        for model in ("transformer", "model_z"):
            cmd = base + ["--model", model] + (["--learned_z", "0"] if model == "model_z" else [])   # formullu yol
            a = TR.main(cmd + ["--epochs", "2", "--out", out(model + "_A")])
            total, per = a["plan"]["total"], a["plan"]["per_epoch"]
            L = [w["loss"] for w in a["log"]]
            check("train %s: 2 epok kosar, kayip duser" % model, len(L) == total and np.mean(L[-3:]) < np.mean(L[:3]) - 0.5,
                  "%d adim (%s / epok), kayip %.3f -> %.3f" % (total, per, np.mean(L[:3]), np.mean(L[-3:])))
            args = TR._args(cmd + ["--out", "x"])
            st = D.TokenStories(root, data, "train")
            m, mask_fn, layout = TR._build(args, st.max_sentence_tokens, torch.device("cpu"))
            f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
            ro, rs = f["row_offsets"], f["row_stories"]
            b = D.build_batch(st, [rs[ro[r]:ro[r + 1]].tolist() for r in range(4)], layout, "cpu", 64)
            with torch.no_grad():
                want = m.loss_per_target(b, R.dense_mask(b, mask_fn))[0].mean().item()
            lr_ok = all(w["lr"] == R.wsd_lr(w["step"] - 1, total, 1e-2) for w in a["log"])
            check("train %s: ilk adim kaybi = loss_per_target (plan e1 satirlari 0-3, ayni tohum); lr = wsd_lr" % model,
                  abs(a["log"][0]["loss"] - want) < 1e-4 and lr_ok, "%.4f / %.4f" % (a["log"][0]["loss"], want))
            keys = {"identity", "plan", "args", "params", "env", "data", "run", "finished", "exam", "exams", "epochs",
                    "generation", "log", "speed"}
            ex_keys = {"loss", "ppl", "acc", "acc_token", "end_ok", "eos_ok", "bits_per_byte", "bits_per_byte_eos",
                       "by_kind", "stories", "bytes", "seconds", "step", "epoch", "full_epoch"}
            dk = torch.load(os.path.join(out(model + "_A"), "decay_start", "checkpoint.pt"), weights_only=False)["step"]
            check("train %s: results.json alanlari; epok basina sinav; decay_start inis basinda" % model,
                  keys <= set(json.load(open(os.path.join(out(model + "_A"), "results.json")))) and ex_keys <= set(a["exam"])
                  and [(e["step"], e["full_epoch"]) for e in a["exams"]] == [(per[0], True), (total, True)]
                  and dk == total - round(0.2 * total) == a["plan"]["decay_start"]
                  and a["speed"]["windows"] == total - 1, "decay_start %d" % dk)
            sj = json.load(open(os.path.join(out(model + "_A"), "samples.json"), encoding="utf-8"))
            txt = open(os.path.join(out(model + "_A"), "samples.txt"), encoding="utf-8").read()
            gk = {"decode", "prompts", "eos_rate", "sentences", "sentence_repeat", "counted", "word_loop", "no_end", "empty",
                  "story_loop", "story_loop_prompts", "sentence_repeat_by_prompt"}
            check("train %s: okuma ciktisi (samples.txt / .json): 3 istem sirayla, istem / model / gercek devam, greedy + "
                  "sample olculeri" % model,
                  [r["label"] for r in sj["rows"]] == ["okuma 1", "okuma 2", "okuma 3"] and all(
                      {"prompt", "story_text", "real", "eos", "story", "sentences", "decode"} <= set(r) for r in sj["rows"])
                  and all("=== okuma %d " % i in txt for i in (1, 2, 3)) and txt.count("--- gercek devam") == 3
                  and set(sj["generation"]) == {"greedy", "sample"} and all(gk <= set(g) for g in sj["generation"].values())
                  and sj["rows"][1]["prompt"].count("\n") == 1, sj["rows"][0]["prompt"] + " | " + sj["rows"][0]["real"])
            orig = R.Checkpoint.save

            class Stop(Exception):
                pass

            def save_then_stop(dir_, *rest):
                orig(dir_, *rest)
                if rest[2] == 4 and os.path.basename(dir_) != "decay_start":
                    raise Stop()
            R.Checkpoint.save = staticmethod(save_then_stop)
            try:
                TR.main(cmd + ["--epochs", "2", "--out", out(model + "_B")])
                stopped = False
            except Stop:
                stopped = True
            finally:
                R.Checkpoint.save = staticmethod(orig)
            bres = TR.main(cmd + ["--epochs", "2", "--out", out(model + "_B"), "--resume", "1",
                                  "--checkpoint_minutes", "30"])                   # kimlige girmez
            check("train %s: adim 4'te kesilip surdurulen = kesintisiz (agirlik bit duzeyinde, sinav ayni)" % model,
                  stopped and same(state(out(model + "_A")), state(out(model + "_B"))) and bres["exam"] == dict(
                      a["exam"], seconds=bres["exam"]["seconds"]) and bres["log"][0]["step"] == 1)
            TR.main(cmd + ["--epochs", "1", "--out", out(model + "_C")])
            cres = TR.main(cmd + ["--epochs", "2", "--out", out(model + "_C"), "--resume", "1"])
            arch = os.path.join(out(model + "_C"), "total_%d" % per[0])
            check("train %s: uzatma (1 -> 2 epok, inis basindan) = bastan 2 epok (bit duzeyinde); eski ciktilar %s" % (
                model, os.path.basename(arch)), same(state(out(model + "_A")), state(out(model + "_C")))
                and all(os.path.exists(os.path.join(arch, n)) for n in TR.OUTPUTS) and cres["plan"]["total"] == total)
        TR.READING_LIMITS = dict(max_sentences=3, max_tokens=500)
        early = _raises(AssertionError, TR.main, base + ["--model", "model_z", "--out", out("long")])
        TR.READING_LIMITS = dict(max_sentences=3, max_tokens=4)
        check("train model_z: okuma max_tokens > z konum anahtari egitimden ONCE durur (checkpoint yok)", early
              and not os.path.exists(os.path.join(out("long"), "checkpoint.pt")))
        cmd = base + ["--model", "transformer"]
        A = out("transformer_A")
        mt = os.path.getmtime(os.path.join(A, "checkpoint.pt"))
        r = TR.main(cmd + ["--epochs", "2", "--out", A, "--resume", "1"])
        check("train: bitmis kosu --resume ile durur, dosyalara dokunmaz",
              r["finished"] and os.path.getmtime(os.path.join(A, "checkpoint.pt")) == mt)
        check("train: checkpoint'li klasore --resume'suz yeni kosu, farkli lr ile surdurme, paketsiz surdurme, kisaltma, "
              "farkli d (agirlik yuklenmeden) DURUR", exits(cmd + ["--epochs", "2", "--out", A]) and exits(
                  [x if x != "1e-2" else "2e-2" for x in cmd] + ["--epochs", "2", "--out", A, "--resume", "1"])
              and exits(cmd + ["--out", out("empty"), "--resume", "1"])
              and exits([x if x != "16" else "32" for x in cmd] + ["--epochs", "2", "--out", A, "--resume", "1"])
              and exits(cmd + ["--epochs", "1", "--out", A, "--resume", "1"]))
        s = TR.main(cmd + ["--steps", "3", "--out", out("steps")])
        check("train --steps 3: WSD toplam 3, inis basi 2, sonda tek sinav (epok ortasi)", s["plan"]["total"] == 3
              and s["plan"]["decay_start"] == 2 and [w["lr"] for w in s["log"]] == [R.wsd_lr(i, 3, 1e-2) for i in range(3)]
              and len(s["exams"]) == 1 and not s["exam"]["full_epoch"])
        orig, saves = R.Checkpoint.save, []
        R.Checkpoint.save = staticmethod(lambda dir_, *rest: (saves.append((os.path.basename(dir_), rest[2])),
                                                              orig(dir_, *rest)))
        try:
            got = {}
            for sec in (0, 10 ** 9):
                saves[:] = []
                TR.main(cmd + ["--steps", "5", "--out", out("sec%d" % sec), "--checkpoint_minutes", str(sec / 60)])
                got[sec] = list(saves)
        finally:
            R.Checkpoint.save = staticmethod(orig)
        check("train --checkpoint_minutes: esik asilinca her gunluk sinirinda kayit (0 sn: adim 1-5), asilmazsa yalniz "
              "bitis + decay_start", got[0] == [("sec0", i) for i in (1, 2, 3)] + [("decay_start", 4), ("sec0", 4),
                                                                                  ("sec0", 5)]
              and got[10 ** 9] == [("decay_start", 4), ("sec1000000000", 5)] and json.load(open(os.path.join(
                  out("sec0"), "config.json")))["args"]["checkpoint_minutes"] == 0, str(got))
        local = os.path.join(TMP, "local")
        TR._local_copy(root, local, data)
        p = os.path.join(local, "gpt2", "train.npy")
        ok1 = open(p, "rb").read() == open(os.path.join(root, "gpt2", "train.npy"), "rb").read()
        with open(p, "r+b") as fh:                                          # bozuk kopya: ayni boy, farkli bayt
            fh.seek(200)
            fh.write(b"\x00\x01")
        TR._local_copy(root, local, data)
        ok2 = open(p, "rb").read() == open(os.path.join(root, "gpt2", "train.npy"), "rb").read()
        bj = os.path.join(data, "train_boundaries.json")
        meta = json.load(open(bj))
        json.dump(dict(meta, stream_sha256="0" * 64), open(bj, "w"))
        os.remove(p)
        ok3 = _raises(AssertionError, TR._local_copy, root, local, data)
        json.dump(meta, open(bj, "w"))
        check("train _local_copy: bayt bayt kopya; bozuk kopya yeniden kopyalanir; sinir dosyasinin sha'si tutmazsa durur",
              ok1 and ok2 and ok3)
        for model in ("transformer", "model_z"):                       # sabit deger (belge 33 adim 4)
            if torch.__version__ != GOLDEN_TORCH:
                print("ATLANDI: sabit deger torch %s ile, burada %s" % (GOLDEN_TORCH, torch.__version__), flush=True)
                break
            g = TR.main(base + ["--model", model, "--steps", "6", "--out", out("golden_" + model)]
                        + (["--learned_z", "0"] if model == "model_z" else []))
            st_ = torch.load(os.path.join(out("golden_" + model), "agent.pt"), weights_only=False)["state"]
            per = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in st_.items()}
            whole = hashlib.sha256("".join(k + per[k] for k in sorted(per)).encode()).hexdigest()
            check("train %s: sabit deger (temizlik oncesi --own_vocab 1 / transformer ile bit duzeyinde ayni olculen "
                  "kosu): ilk 6 kayip ve agirlik sha256" % model, [w["loss"] for w in g["log"]][:6] == GOLDEN[model][
                      "first6"] and whole == GOLDEN[model]["sha"], whole[:16])
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
        import generate_readings as GR
        legacy = lambda d, **kv: (d.update({k: kv.get(k, 0) for k in TR.LEGACY}), d)[1]  # noqa: E731
        st = D.TokenStories(root, data, "train")
        dev = torch.device("cpu")
        msgs, loads = {}, {}
        for kind, src, kv in (("tf", out("transformer_A"), dict(meaning_sha256=None)),
                              ("own", out("model_z_A"), dict(meaning_sha256=None, own_vocab=1)),
                              ("iota", out("model_z_A"), dict(meaning_sha256="0" * 64))):
            dst = out("legacy_" + kind)                                   # temizlik oncesi kimlikli kopya
            shutil.copytree(src, dst)
            for name, key in (("checkpoint.pt", "args"), ("agent.pt", "identity")):
                pack = torch.load(os.path.join(dst, name), weights_only=False)
                legacy(pack[key], **kv)
                if kind == "iota":
                    del pack[key]["own_vocab"], pack[key]["shared_vocab"], pack[key]["open_z"]   # e6e7847 kimligi
                torch.save(pack, os.path.join(dst, name))
            mt = os.path.getmtime(os.path.join(dst, "checkpoint.pt"))
            msgs[kind] = (_exit_msg(TR.main, base + ["--model", "model_z" if kind != "tf" else "transformer",
                                                     "--epochs", "3", "--out", dst, "--resume", "1"]) or "",
                          os.path.getmtime(os.path.join(dst, "checkpoint.pt")) == mt)
            loads[kind] = _exit_msg(GR.load_run, dst, data, dev)
        m = GR.load_run(out("legacy_own"), data, dev)[0]
        same_w = same(m.state_dict(), state(out("model_z_A")))
        silent = not _raises(Exception, TR._build(TR._args(base + ["--model", "model_z", "--learned_z", "0", "--out", "x"]),
                                                  st.max_sentence_tokens, dev)[0].load_state_dict,
                             torch.load(os.path.join(out("legacy_iota"), "agent.pt"), weights_only=False)["state"])
        fresh = json.load(open(os.path.join(out("model_z_A"), "config.json")))["identity"]
        check("train: eski kimlikli checkpoint SURDURULMEZ (transformer, own, iota; iletide etiket, dosyaya dokunulmaz); "
              "yeni kimlikte eski alanlar yok", all(TR.TAG in msg and kept for msg, kept in msgs.values())
              and not set(fresh) & set(TR.LEGACY), msgs["own"][0][:120])
        check("load_run: eski transformer ve eski own_vocab=1 Model Z yuklenir (agirlik ayni); eski iota Model Z DURUR "
              "(iletide etiket; agirlik sekilleri ayni, strict yukleme hatasiz = sessiz risk)",
              loads["tf"] is None and loads["own"] is None and same_w and TR.TAG in (loads["iota"] or "") and silent,
              (loads["iota"] or "")[:120])
        gone = [base + ["--model", "model_z", a, v, "--out", out("gone%d" % i)] for i, (a, v) in enumerate(
            (("--meaning", "x.pt"), ("--own_vocab", "1"), ("--shared_vocab", "1"), ("--open_z", "2")))]
        check("train: kaldirilan argumanlar (--meaning, --own_vocab, --shared_vocab, --open_z) veri yuklenmeden DURUR",
              all(exits(c) and not os.path.exists(c[-1]) for c in gone))
        if _learned_z_ready():
            _train_learned(base, root, data, out, state, same, exits, TR)
        else:
            print("BEKLIYOR: learned_z modeli yok (SentenceTransformer(learned_z) / model_z_read_mask)", flush=True)
    except Exception:  # noqa: BLE001
        check("train", False, traceback.format_exc(limit=3))
    finally:
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS = saved


def t_tokens():
    """data.token_counts: parca parca sayim = np.bincount (butun akis); dosya yazilir; toplam = akis boyu."""
    rng = np.random.default_rng(3)
    root = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(root, "gpt2"))
    x = rng.integers(0, D.EOS_ID + 1, 100_003).astype(np.uint16)
    np.save(os.path.join(root, "gpt2", "train.npy"), x)
    out = os.path.join(root, "v2")
    c = D.token_counts(root, out, chunk=7_919)
    disk = np.load(os.path.join(out, "train_token_counts.npy"))
    want = np.bincount(x.astype(np.int64), minlength=D.EOS_ID + 1)
    check("token_counts: parcali sayim = np.bincount (100.003 token, parca 7.919); dosya = donus; uzunluk GPT-2 sozlugu "
          "(EOS dahil); toplam = akis boyu", np.array_equal(c, want) and np.array_equal(disk, c)
          and len(c) == D.EOS_ID + 1 and int(c.sum()) == len(x) and c.dtype == np.int64)


def _learned_z_ready():
    """model_z/ learned_z'yi (belge 35 (b)) taniyor mu: SentenceTransformer(learned_z) ve model_z_read_mask."""
    import inspect
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
    import sentence as SM
    return "learned_z" in inspect.signature(SM.SentenceTransformer).parameters and hasattr(SM, "model_z_read_mask")


def _cut_and_resume(TR, cmd, out_dir):
    """cmd'yi adim 4'teki checkpoint'ten hemen sonra kes, --resume 1 ile bitir -> (kesildi mi, sonuc)."""
    import recipe as R
    orig = R.Checkpoint.save

    class Stop(Exception):
        pass

    def save_then_stop(dir_, *rest):
        orig(dir_, *rest)
        if rest[2] == 4 and os.path.basename(dir_) != "decay_start":
            raise Stop()
    R.Checkpoint.save = staticmethod(save_then_stop)
    try:
        TR.main(cmd + ["--out", out_dir])
        stopped = False
    except Stop:
        stopped = True
    finally:
        R.Checkpoint.save = staticmethod(orig)
    return stopped, TR.main(cmd + ["--out", out_dir, "--resume", "1"])


def _train_learned(base, root, data, out, state, same, exits, TR):
    """--learned_z 1 (belge 35 (b)) gercek SentenceTransformer ve gercek build_batch ile: kayip duser, kimlikte bayrak,
    ilk adim (train._attn: model_z_read_mask) = modelin KENDI maskesiyle loss_per_target, okuma maskesi Z_k'ye kendi
    cumlesini acar, kesilip surdurulen = kesintisiz, bayrak farkiyla surdurme ve transformer + bayrak DURUR."""
    import traceback
    import recipe as R
    try:
        cmd = base + ["--model", "model_z", "--learned_z", "1"]
        A = out("mzl_A")
        a = TR.main(cmd + ["--epochs", "2", "--out", A])
        L = [w["loss"] for w in a["log"]]
        st = D.TokenStories(root, data, "train")
        m, mask_fn, layout = TR._build(TR._args(cmd + ["--out", "x"]), st.max_sentence_tokens, torch.device("cpu"))
        f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
        ro, rs = f["row_offsets"], f["row_stories"]
        b = D.build_batch(st, [rs[ro[r]:ro[r + 1]].tolist() for r in range(4)], layout, "cpu", 64)
        with torch.no_grad():
            want = m.loss_per_target(b)[0].mean().item()                  # modelin kendi (recipe'siz) maskesi
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model_z"))
        from sentence import model_z_mask
        read, base_m = R.dense_mask(b, mask_fn), R.dense_mask(b, model_z_mask)
        z = b.kind == D.Kind.ZTOK
        tok = b.kind == D.Kind.TOKEN
        own = (b.sent[:, :, None] == b.sent[:, None, :]) & (b.doc[:, :, None] == b.doc[:, None, :]) & tok[:, None, :]
        extra = read & ~base_m
        check("train model_z --learned_z 1: 2 epok kosar, kayip duser; kimlikte learned_z 1; ilk adim (train._attn, "
              "model_z_read_mask) = modelin kendi maskesiyle loss_per_target; okuma maskesi = model_z_mask + yalniz Z_k "
              "satirinda kendi cumlesinin token'lari", np.mean(L[-3:]) < np.mean(L[:3]) - 0.5
              and a["identity"]["learned_z"] == 1 and abs(a["log"][0]["loss"] - want) < 1e-4
              and bool((base_m <= read).all()) and bool(extra.any()) and bool((extra <= (z[:, :, None] & own)).all())
              and int(extra.sum()) == int((z[:, :, None] & own).sum()),
              "kayip %.3f -> %.3f; ilk %.4f / %.4f; ek cift %d" % (np.mean(L[:3]), np.mean(L[-3:]), a["log"][0]["loss"],
                                                                 want, int(extra.sum())))
        stopped, bres = _cut_and_resume(TR, cmd + ["--epochs", "2"], out("mzl_B"))
        check("train model_z --learned_z 1: adim 4'te kesilip surdurulen = kesintisiz (agirlik bit duzeyinde, sinav ayni)",
              stopped and same(state(A), state(out("mzl_B"))) and bres["exam"] == dict(a["exam"],
                                                                                         seconds=bres["exam"]["seconds"]))
        mt = os.path.getmtime(os.path.join(A, "checkpoint.pt"))
        plain = base + ["--model", "model_z", "--learned_z", "0"]
        check("train: learned_z farkiyla surdurme checkpoint yuklenmeden DURUR (1 -> 0, 0 -> 1); transformer + "
              "--learned_z 1 veri yuklenmeden DURUR",
              exits(plain + ["--epochs", "2", "--out", A, "--resume", "1"])
              and exits(cmd + ["--epochs", "2", "--out", out("model_z_A"), "--resume", "1"])
              and os.path.getmtime(os.path.join(A, "checkpoint.pt")) == mt
              and exits(base + ["--model", "transformer", "--learned_z", "1", "--out", out("tf_learned")])
              and not os.path.exists(out("tf_learned")))
        default = {m: TR._args(base + ["--model", m, "--out", "x"]).learned_z for m in ("model_z", "transformer")}
        check("train: varsayilan learned_z model_z'de 1, transformer'da 0 (kullanici, 7 Ekim)",
              default == {"model_z": 1, "transformer": 0}, str(default))
    except Exception:  # noqa: BLE001
        check("train model_z --learned_z 1", False, traceback.format_exc(limit=3).splitlines()[-1])


def _exit_msg(fn, *a):
    """fn SystemExit verirse iletisi, yoksa None."""
    try:
        fn(*a)
    except SystemExit as e:
        return str(e.code)
    return None


def _raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return True
    return False


TESTS = dict(data=t_data, pack=t_pack, recipe=t_recipe, metrics=t_metrics, integration=t_integration,
             train=t_train, drive=t_drive, tokens=t_tokens)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
