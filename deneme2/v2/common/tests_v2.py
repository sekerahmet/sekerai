"""tests_v2 -- V2 common/ testleri (CPU).  Gruplar: data (hizli; GPT-2 tokenizer'i yerel HF onbelleginden ya da Drive'dan,
yoksa tokenizer'li sinamalar ATLANIR), pack (sentetik), recipe (maske, WSD, gruplar, Muon, surdurme, hiz), metrics (sentence_repeat,
normalize_words, story_generation, exam_scores), integration (iki modelin loss_per_target'i ve recipe.output_loss'u
gercek build_batch ile; model dosyalari yalniz testte import edilir), train (train.py uctan uca, iki model, kucuk veri;
eski kimlik reddi, etiketteki kodla esdegerlik dahil; ~3 dk), drive (valid akisi: V1 ile birebir esleme, okuma istemleri), tokens (data.token_counts), fineweb (belge 48:
web profili, parca / continues, make_fineweb, FineWeb ile train, knowledge_exam, generate open_last, d 768).  Teshis araclari:
diag/tests_diag.py.

    python tests_v2.py [--only data,pack,recipe,metrics,integration,train,drive,tokens,fineweb]
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

GOLDEN_TORCH = "2.14.0+cpu"     # transformer: belge 33 adim 4 (v2-before-cleanup-20261006 ile bit duzeyinde ayni olculdu);
# model_z (G 1) / model_z_g0: etiket v2-before-formula-cleanup-20261007 kodundan olculdu (7 Ekim, belge 44; --layers 2,
# adamw, --steps 6; scratchpad golden_measure.py), bugunku kod ayni degeri verdi
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
  "first6": [10.8696, 10.6366, 10.3411, 10.0874, 9.8309, 9.5415],
  "sha": "b6783484b90c6d6e97ad9e2683f143b99567739503bad1d93cc1f7776c4e54ae"
 },
 "model_z_g0": {
  "first6": [10.8705, 10.6408, 10.3417, 10.0936, 9.831, 9.5423],
  "sha": "bab46c04017cb4a3cb0abbf0a6a25c9025b64ebf98c13ac54f4c616ad429b0ae"
 }
}
GOLDEN_ARGS = {"transformer": ["--model", "transformer"],                # eski davranis acik bayrakla (8 Ekim
               "model_z": ["--model", "model_z", "--layers", "2", "--global_layers", "1", "--summaries_last", "0"],
               "model_z_g0": ["--model", "model_z", "--layers", "2", "--global_layers", "0", "--summaries_last", "0"]}


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
                seq = [0 if x == D.END_ID else x for x in seq]
                kd = [K.ZTOK if x == K.END else x for x in kd]
            sl = slice(c, c + T)
            for f, v in (("tokens", seq), ("kind", kd), ("pos", ps), ("doc", d), ("sent", sn), ("target", nxt),
                         ("target_kind", tk)):
                out[f][r, sl] = v
            c += T
    return out


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
    check("ZTOK konumunun sent alani = kendi cumle numarasi (belge 35 okuma maskesi buna dayaniyor)",
          bz.sent[bz.kind == K.ZTOK].tolist() == [0, 1, 2, 0, 0, 1], str(bz.sent[bz.kind == K.ZTOK].tolist()))
    rng = np.random.default_rng(1)
    st = _Synthetic([[rng.integers(0, 50000, int(rng.integers(1, 30))).tolist() for _ in range(int(rng.integers(1, 40)))]
                     for _ in range(120)])
    ro, rs = D.pack_plan(1 + np.add.reduceat(st.sent[:, 1] - st.sent[:, 0] + 1, st.story[:-1]))
    rows = [rs[ro[r]:ro[r + 1]].tolist() for r in range(min(8, len(ro) - 1))]
    ok = True
    for lay in ("transformer", "model_z"):
        b = D.build_batch(st, rows, lay)
        ref = _reference_batch(st, rows, lay, D.ROW_LEN)
        ok &= all(np.array_equal(getattr(b, f).numpy(), ref[f]) for f in ref)
    check("build_batch (vektorel) = dongulu basvuru, 120 rastgele hikaye, 8 satir, iki duzen", ok)
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
    _recipe_muon(R)
    _recipe_bag(R)
    import time as _t
    sw = R.SpeedWindow()
    sw.start(5)
    _t.sleep(0.2)
    r = sw.stop(15, 1000)
    check("SpeedWindow: adim sayisi, sure, token / sn", r["steps"] == 10 and 0.18 < r["seconds"] < 0.5
          and abs(r["tokens_per_sec"] * r["seconds"] / 1000 - 1) < 0.01, str(r))


def _newton_schulz(G, steps=5, eps=1e-7):
    """Basvuru: Keller Jordan Muon NS5 (jordan2024_muon; katsayilar 3,4445 / -4,7750 / 2,0315), bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    tall = G.size(0) > G.size(1)
    X = G.bfloat16()
    X = X.T if tall else X
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if tall else X).float()


def _recipe_muon(R):
    """--optimizer muon'un tarifi: bolusum, Moonshot olcegi (liu2025_muonscalable denklem 4), MuonAdamW ile surdurme."""
    import train as TR

    class TinyBlocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = torch.nn.Embedding(50, 8)
            self.z_in = torch.nn.Linear(16, 8, bias=False)
            self.blocks = torch.nn.ModuleList([torch.nn.ModuleDict(dict(norm=torch.nn.RMSNorm(8),
                                                                        lin=torch.nn.Linear(8, 8)))])
            self.norm = torch.nn.RMSNorm(8)
    tm = TinyBlocks()
    blk = tm.blocks[0]
    opt, info = TR._optimizer(tm, "muon", 1e-2, False)
    mu = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    ad = [{id(p) for p in g["params"]} for g in opt.adamw.param_groups]
    groups = [mu] + ad
    every = sum(len(g) for g in groups) == len(set().union(*groups)) == len(list(tm.parameters()))
    check("muon bolusumu: her parametre tam bir grupta; Muon'da yalniz blok 2-B agirligi; E, z_in (decay'li), blok bias / "
          "norm AdamW'de; Muon wd 0,1 ve match_rms_adamw", every and mu == {id(blk.lin.weight)}
          and ad == [{id(tm.z_in.weight)}, {id(tm.E.weight), id(blk.norm.weight), id(blk.lin.bias), id(tm.norm.weight)}]
          and opt.muon.param_groups[0]["weight_decay"] == 0.1 and opt.muon.param_groups[0]["adjust_lr_fn"] ==
          "match_rms_adamw" and info["split"]["muon"]["names"] == {"blocks.*.lin.weight": 1}, str(info["split"]))
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(256, 64) * 0.02)
    W0, g = W.detach().clone(), torch.randn(256, 64)
    W.grad = g.clone()
    lr = 1e-3
    torch.optim.Muon([W], lr=lr, weight_decay=0.1, **TR.MUON).step()
    want = -lr * 0.1 * W0 - lr * 0.2 * math.sqrt(256) * _newton_schulz(g)     # ilk adim: momentum yalniz olcek, NS'te gider
    rel = ((W.detach() - W0) - want).norm().item() / want.norm().item()
    rms = (_newton_schulz(g) * 0.2 * math.sqrt(256)).pow(2).mean().sqrt().item()
    check("Muon (train.MUON) bir adim = W - lr (0,2 sqrt(max(A, B)) NS5(g) + wd W) (Moonshot denklem 4; bf16 NS gurultusu "
          "~%2-4, 'original' olcekle fark %37); guncelleme RMS / lr ~0,2 (AdamW'ninki)", rel < 0.08 and 0.12 < rms < 0.28,
          "goreli fark %.1e, RMS/lr %.3f" % (rel, rms))

    def run(steps, save_at=None, dir_=None, load=False):
        torch.manual_seed(0)
        model = TinyBlocks()
        opt = TR._optimizer(model, "muon", 1e-2, False)[0]
        s0 = R.Checkpoint.load(dir_, model, opt)["step"] if load else 0
        for s in range(s0, steps):
            for gr in opt.param_groups:
                gr["lr"] = R.wsd_lr(s, steps, 1e-2)
            x = torch.randn(4, 16)
            h = model.blocks[0]["lin"](model.blocks[0]["norm"](model.z_in(x)))
            loss = model.norm(h).pow(2).mean() + h.pow(2).mean() + model.E(torch.tensor([1, 2])).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if s + 1 == save_at:
                R.Checkpoint.save(dir_, model, opt, s + 1, {}, [], {})
                return None
        return model
    full = run(10)
    d_ = os.path.join(TMP, "ckpt_muon")
    run(10, save_at=4, dir_=d_)
    resumed = run(10, dir_=d_, load=True)
    pack = torch.load(os.path.join(d_, "checkpoint.pt"), weights_only=False)["opt"]
    check("MuonAdamW: 4 + 6 adim = kesintisiz 10 adim (agirlik bit duzeyinde); checkpoint'te iki optimizer (Muon "
          "momentum_buffer, AdamW exp_avg); lr takvimi iki optimizer'in gruplarina", all(
              torch.equal(a, c) for a, c in zip(full.state_dict().values(), resumed.state_dict().values()))
          and set(pack) == {"adamw", "muon"} and "momentum_buffer" in pack["muon"]["state"][0]
          and "exp_avg" in pack["adamw"]["state"][0] and len(opt.param_groups) == 3)
    shapes = [(8, 24), (24, 8), (8, 24), (16, 16)]                         # belge 57 D1
    g = torch.Generator().manual_seed(0)
    init = [torch.randn(*s_, generator=g) * 0.02 for s_ in shapes]
    pa, pb = [torch.nn.Parameter(p.clone()) for p in init], [torch.nn.Parameter(p.clone()) for p in init]
    oa = torch.optim.Muon(pa, lr=2e-3, weight_decay=0.1, **TR.MUON)
    ob = R.BatchedMuon(pb, lr=2e-3, weight_decay=0.1, **TR.MUON)
    for _ in range(3):
        for x, y in zip(pa, pb):
            x.grad = torch.randn(x.shape, generator=g)
            y.grad = x.grad.clone()
        oa.step()
        ob.step()
    oa2 = torch.optim.Muon([torch.nn.Parameter(p.clone()) for p in init], lr=2e-3, weight_decay=0.1, **TR.MUON)
    oa2.load_state_dict(ob.state_dict())
    check("BatchedMuon = torch.optim.Muon (CPU, 3 adim, nesterov, wd, match_rms_adamw; ayni bicimli matrisler yiginda): "
          "agirlik ve momentum_buffer bit esit; state_dict torch Muon'a yuklenir",
          all(torch.equal(x, y) for x, y in zip(pa, pb))
          and all(torch.equal(oa.state[x]["momentum_buffer"], ob.state[y]["momentum_buffer"]) for x, y in zip(pa, pb)))
    _recipe_normuon(R, TR)


def _recipe_normuon(R, TR):
    """NorMuon (li2025_normuon Algorithm 1; kullanici, 8 Ekim): (a) ortak yol (momentum, nesterov, wd, NS) BatchedMuon'la
    bit ayni (yalniz _apply farkli); (b) yiginli NorMuon = matris matris basvuru (Algorithm 1 satir 5-11, NS torch'un tek
    matris NS'i); (c) ilk adimda W farkinin satir normlari esit (CV ~0), guncelleme RMS'i 0,2 lr; surdurme: 2 + 3 adim =
    kesintisiz 5 (state_dict, second_momentum_buffer dahil)."""
    from torch.optim._muon import _zeropower_via_newtonschulz
    shapes = [(8, 24), (24, 8), (8, 24), (16, 16)]
    g = torch.Generator().manual_seed(1)
    init = [torch.randn(*s_, generator=g) * 0.02 for s_ in shapes]
    grads = [[torch.randn(*s_, generator=g) for s_ in shapes] for _ in range(5)]
    kw = dict(lr=2e-3, weight_decay=0.1, **TR.MUON)

    def run(cls, steps, patch=None, opt_state=None, start=None):
        ps = [torch.nn.Parameter((start[i] if start else init[i]).clone()) for i in range(len(shapes))]
        opt = cls(ps, **kw)
        if patch:
            opt._apply = patch.__get__(opt)
        if opt_state:
            opt.load_state_dict(opt_state)
        for t in steps:
            for x, gr in zip(ps, grads[t]):
                x.grad = gr.clone()
            opt.step()
        return ps, opt
    a, oa = run(R.BatchedMuon, range(3))
    b, ob = run(R.NorMuon, range(3), patch=R.BatchedMuon._apply)
    same_a = all(torch.equal(x, y) for x, y in zip(a, b)) and all(
        torch.equal(oa.state[x]["momentum_buffer"], ob.state[y]["momentum_buffer"]) for x, y in zip(a, b))
    c, oc = run(R.NorMuon, range(3))
    ref = [w.clone() for w in init]
    M = [torch.zeros_like(w) for w in init]
    v = [torch.zeros(w.shape[0], 1) for w in init]
    for t in range(3):
        for i, (m_, n_) in enumerate(shapes):
            M[i].lerp_(grads[t][i], 1 - kw["momentum"])                      # satir 5 (EMA; nesterov resmi koddaki gibi)
            u = grads[t][i].lerp(M[i], kw["momentum"])
            pg = oc.param_groups[0]
            O = _zeropower_via_newtonschulz(u, pg["ns_coefficients"], pg["ns_steps"], pg["eps"]).float()   # satir 6
            v[i] = R.NORMUON_BETA2 * v[i] + (1 - R.NORMUON_BETA2) * (O * O).mean(1, keepdim=True)          # satir 7
            Oh = O / (v[i].sqrt() + R.NORMUON_EPS)                                                          # satir 9
            eta = 0.2 * kw["lr"] * (m_ * n_) ** 0.5 / Oh.norm()                                            # satir 10
            ref[i] = ref[i] - kw["lr"] * kw["weight_decay"] * ref[i] - eta * Oh                            # satir 11
    rel = max(float((x.detach() - r).norm() / (r - w).norm()) for x, r, w in zip(c, ref, init))
    d, od = run(R.NorMuon, range(1), patch=None)
    dw = [(x.detach() - w * (1 - kw["lr"] * kw["weight_decay"])) for x, w in zip(d, init)]
    cv = max(float(r.std() / r.mean()) for r in (u_.norm(dim=1) for u_ in dw))
    rms = [float(u_.square().mean().sqrt()) / kw["lr"] for u_ in dw]
    e1, oe1 = run(R.NorMuon, range(2))
    e2, _ = run(R.NorMuon, range(2, 5), opt_state=oe1.state_dict(), start=[x.detach() for x in e1])
    full_, ofull = run(R.NorMuon, range(5))
    check("NorMuon: (a) ortak yol = BatchedMuon (bit); (b) yiginli = matris matris Algorithm 1 basvurusu (goreli %.1e); "
          "(c) ilk adim satir normlari esit (CV %.1e), RMS / lr %s; 2 + 3 adim = kesintisiz 5 (bit, ikinci moment dahil)"
          % (rel, cv, [round(x, 3) for x in rms]),
          same_a and rel < 1e-5 and cv < 1e-4 and all(abs(x - 0.2) < 1e-4 for x in rms)
          and all(torch.equal(x, y) for x, y in zip(e2, full_))
          and all("second_momentum_buffer" in ofull.state[x] for x in full_))


def _recipe_bag(R):
    """Ogrenen torba (belge 54 s1.3, 55 K1 / K3 / O7 / O8): kahin kapisi (s_O = LSE disari -> log_softmax, gradyan dahil),
    B = V'de DIGER kapali, toplam 1; core_ids; hizli yol (bag_train_loss: torba basina bmm, birden cok dilim + dolgu, P
    kesilmesi, kacan + pay satirlari) = output_logprobs (two_stage_logprobs, konum basina B_k) NLL'i ve gradyani; secici
    kaybi elle; kaynak sayilari toplami; bag_full_mask adim tohumuyla ayni.  Iki duzen (Model Z: ZTOK, transformer: END)."""
    g = torch.Generator().manual_seed(0)
    V = 50
    lg = torch.randn(6, V, generator=g, dtype=torch.float64).float()
    t = torch.randint(0, V, (6,), generator=g)
    ok = True
    for k in (1, 7, 25, V - 1):
        inb = torch.zeros(V, dtype=torch.bool)
        inb[torch.randperm(V, generator=g)[:k]] = True
        a = lg.clone().requires_grad_(True)
        b_ = lg.clone().requires_grad_(True)
        lp = R.two_stage_logprobs(a, inb, a.masked_fill(inb, float("-inf")).logsumexp(-1))
        ref = torch.log_softmax(b_, -1)
        lp.gather(1, t[:, None]).sum().backward()
        ref.gather(1, t[:, None]).sum().backward()
        ok &= torch.allclose(lp, ref, atol=1e-6) and torch.allclose(a.grad, b_.grad, atol=1e-6)
    allin = torch.ones(V, dtype=torch.bool)
    free = R.two_stage_logprobs(lg, inb, torch.randn(6, generator=g))
    check("torba: kahin DIGER (LSE disari) iken iki asamali = log_softmax, gradyan dahil (|B| 1 / 7 / 25 / V-1); B = V'de "
          "DIGER kapali (birebir); serbest DIGER'de toplam 1",
          ok and torch.equal(R.two_stage_logprobs(lg, allin, torch.full((6,), 50.0)), torch.log_softmax(lg, -1))
          and float((free.exp().sum(-1) - 1).abs().max()) < 1e-5)
    counts = np.arange(D.EOS_ID + 1)[::-1].copy()                          # 0 en sik
    check("torba: core_ids = en sik 30 + END + EOS, sirali tekil",
          R.core_ids(counts, 30).tolist() == sorted(set(range(30)) | {D.END_ID, D.EOS_ID}))
    stories = [[[3, 40, 41, 3, 42], [43, 44, 40], [45, 46, 47, 48, 41]], [[50, 51], [52, 50, 53]],
               [[60, 61, 62], [63], [64, 65, 60, 66]]]
    counts = np.zeros(D.EOS_ID + 1, np.int64)
    counts[[3, 4, 5]] = 1000
    counts[40:70] = 5
    core = R.core_ids(counts, 3)
    saved = R.BAG_GROUP, R.BAG_CHUNK
    R.BAG_GROUP, R.BAG_CHUNK = 2, 4                                        # birden cok dilim + dolgu
    try:
        res = []
        for layout in ("model_z", "transformer"):
            b = D.build_batch(_Synthetic(stories), [[0, 1], [2]], layout, row_len=40)
            torch.manual_seed(1)
            m = torch.nn.Module()
            m.E = torch.nn.Embedding(D.VOCAB, 8)
            R.attach_bag(m, len(core) + 3, len(core))                      # R = 3: P kesilir, L kucuk, kacan var
            m.bag.fill(core, counts)
            torch.nn.init.normal_(m.bag.other, std=0.5)
            h0 = torch.randn(*b.kind.shape, 8, generator=g)
            full = R.bag_full_mask(b.target.numpy(), 0.3, np.random.default_rng(np.random.SeedSequence(0, spawn_key=(5,))))
            hs = h0.clone().requires_grad_(True)
            goal, nll, ex = R.bag_train_loss(m, b, hs, torch.zeros(full.shape, dtype=torch.bool), 0.0)   # pay yok, lambda 0
            goal.backward()
            gs = (hs.grad.clone(), m.E.weight.grad.clone(), m.bag.other.grad.clone())
            m.zero_grad()
            hr = h0.clone().requires_grad_(True)
            pos = (b.target.flatten() >= 0).nonzero()[:, 0]
            lp = R.output_logprobs(m, b, hr, pos)
            ref = -lp.gather(1, b.target.flatten()[pos][:, None])[:, 0].mean()
            ref.backward()
            gr = (hr.grad, m.E.weight.grad, m.bag.other.grad)
            rel = max(float((a - c).norm() / c.norm()) for a, c in zip(gs, gr))
            m.zero_grad()
            goal1, nll1, ex1 = R.bag_train_loss(m, b, h0, full, 0.5)
            sel = m.bag.batch_select(b, h0, m.E.weight)
            fi = full & (b.target.flatten() >= 0)
            ce = torch.nn.functional.cross_entropy(h0.flatten(0, 1)[fi] @ m.E.weight.T, b.target.flatten()[fi], reduction="sum")
            ids, y = sel["ids"].flatten()[pos], b.target.flatten()[pos]
            allw, fsc = sel["allowed"], R.Bag.full_score(sel)
            man = sum(float(torch.logsumexp(fsc[i][allw[i]], 0) - fsc[i, w])
                      for i, w in zip(ids.tolist(), y.tolist()) if allw[i, w])
            nsel = sum(bool(allw[i, w]) for i, w in zip(ids.tolist(), y.tolist()))
            res.append(dict(layout=layout, nll=abs(float(nll) - float(ref)), rel=rel, goal=float(goal) == float(nll),
                            extra=abs(float(goal1 - nll1) - (float(ce) / len(pos) + 0.5 * man / nsel)),
                            src=int(ex["c"] + ex["p"] + ex["l"] + ex["miss"]) == len(pos), miss=int(ex["miss"]),
                            p_over=int(ex["p_over"]), full=int(ex1["n_full"]) == round(0.3 * len(pos))))
        full2 = R.bag_full_mask(b.target.numpy(), 0.3, np.random.default_rng(np.random.SeedSequence(0, spawn_key=(5,))))
        check("torba: hizli yol = output_logprobs NLL'i ve gradyani (h, E, other), iki duzen; amac - nll = pay satirlarinin "
              "tam CE'si / hedef + lambda x secici kaybi (elle); kaynaklar toplami = hedef; kacan ve P kesilmesi var; pay "
              "adim tohumuyla ayni",
              all(r["nll"] < 1e-5 and r["rel"] < 1e-5 and r["goal"] and r["extra"] < 1e-4 and r["src"] and r["miss"] > 0
                  and r["p_over"] > 0 and r["full"] for r in res) and torch.equal(full, full2), str(res))
        R.BAG_GROUP = 256
        long = [[[40 + i % 30 for i in range(230)], [41, 42]], [[43, 44]]]          # SS train en uzun cumle 227 (GPU coktu)
        out_ = []
        for layout in ("model_z", "transformer"):
            b = D.build_batch(_Synthetic(long), [[0, 1]], layout, row_len=256)
            hl = torch.randn(*b.kind.shape, 8, generator=g)
            nll_l = R.bag_train_loss(m, b, hl, torch.zeros(b.kind.numel(), dtype=torch.bool), 0.0)[1]
            pos = (b.target.flatten() >= 0).nonzero()[:, 0]
            ref_l = -R.output_logprobs(m, b, hl, pos).gather(1, b.target.flatten()[pos][:, None])[:, 0].mean()
            out_.append(abs(float(nll_l) - float(ref_l)))
        check("torba: 128'den uzun torba (230 token'lik cumle, satir 256) iki duzende hizli yol = output_logprobs",
              max(out_) < 1e-5, str(out_))
        R.BAG_GROUP = 2
        ok_plan, ok_sel = True, True
        for layout in ("model_z", "transformer"):                         # belge 57 D2
            b = D.build_batch(_Synthetic(stories), [[0, 1], [2]], layout, row_len=40)
            torch.manual_seed(1)
            m = torch.nn.Module()
            m.E = torch.nn.Embedding(D.VOCAB, 8)
            R.attach_bag(m, len(core) + 3, len(core))
            m.bag.fill(core, counts)
            torch.nn.init.normal_(m.bag.other, std=0.5)
            torch.nn.init.normal_(m.bag.q.weight, std=0.5)
            h0 = torch.randn(*b.kind.shape, 8, generator=g)
            for sf in (1.0, 0.5):
                flags = R.bag_full_mask(b.target.numpy(), 0.3, np.random.default_rng(np.random.SeedSequence(0, spawn_key=(5,))),
                                        sf)
                out = []
                for plan in (R.bag_plan(b, flags, torch.as_tensor(core)), None):
                    m.zero_grad()
                    hp = h0.clone().requires_grad_(True)
                    goal_, nll_, ex_ = R.bag_train_loss(m, b, hp, flags, 0.5, plan=plan)
                    goal_.backward()
                    out.append((float(goal_), float(nll_), {k: float(v) for k, v in ex_.items()}, hp.grad.clone(),
                                m.bag.q.weight.grad.clone()))
                ok_plan &= out[0][:3] == out[1][:3] and torch.equal(out[0][3], out[1][3]) and torch.equal(out[0][4],
                                                                                                         out[1][4])
                sel = m.bag.batch_select(b, h0, m.E.weight)
                T = b.kind.shape[1]
                keep = {j for j in range(len(sel["rows"]))
                        if int(flags[int(sel["rows"][j]) * T + int(sel["cols"][j])]) & 2}
                fsc, allw = R.Bag.full_score(sel), sel["allowed"]
                tot = n = 0
                for q in (b.target.flatten() >= 0).nonzero()[:, 0].tolist():
                    j, w = int(sel["ids"].flatten()[q]), int(b.target.flatten()[q])
                    if j in keep and bool(allw[j, w]):
                        sc = fsc[j].masked_fill(~allw[j], float("-inf"))
                        tot, n = tot + float(torch.logsumexp(sc, 0) - sc[w]), n + 1
                got = out[0][2]["loss_selector"] / max(out[0][2]["n_selector"], 1)
                ok_sel &= abs(got - tot / max(n, 1)) < 1e-4 and (sf == 1.0) == (len(keep) == len(sel["rows"]))
        t_ = np.arange(50) - 5
        a_ = R.bag_full_mask(t_, 0.3, np.random.default_rng(7))
        old = np.zeros(50, bool)
        old[np.random.default_rng(7).choice(np.flatnonzero(t_ >= 0), int(round(0.3 * 45)), replace=False)] = True
        check("torba: bag_plan (CPU'da kurulmus) = plansiz yol (amac, nll, ekler, h ve q gradyani bit; iki duzen, sel_frac 1 "
              "ve 0,5); sel_frac 0,5'te secici kaybi = yalniz bit 1'li torbalarin elle hesabi; sel_frac 1'de bit 0 eski rng "
              "sirasi, bit 1 hepsi", ok_plan and ok_sel and np.array_equal(a_.numpy() & 1, old)
              and bool(((a_.numpy() & 2) > 0).all()))
    finally:
        R.BAG_GROUP, R.BAG_CHUNK = saved


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
        from sentence import SentenceTransformer
        torch.manual_seed(0)
        mz = SentenceTransformer(d=16, layers=2, heads=2, global_layers=1).eval()     # train.py varsayilani (G 1)
        models.append(("model_z", mz, mz.mask_fn))
    except Exception:  # noqa: BLE001
        check("entegrasyon: Model Z modeli kurulur", False, traceback.format_exc(limit=1).splitlines()[-1])
    for layout, model, mask_fn in models:
        try:
            with torch.no_grad():
                b = D.build_batch(st, rows[:2], layout, row_len=128)
                nll, pred, tk = model.loss_per_target(b)
                nll2, _, _ = model.loss_per_target(b, _dense_pair(b, mask_fn))
            k = int((b.target >= 0).sum())
            ok = len(nll) == len(pred) == len(tk) == k and bool(torch.isfinite(nll).all()) and torch.equal(
                tk, b.target_kind[b.target >= 0]) and (nll - nll2).abs().max().item() < 1e-5
            check("entegrasyon %s: loss_per_target(build_batch) sozlesmesi; recipe.dense_mask(maske fonksiyonu) = "
                  "modelin kendi maskesi (Model Z: yerel + global ikilisi)" % layout, ok, "hedef %d, fark %.1e" % (k, (nll - nll2).abs().max().item()))
            e = M.exam_scores(model, st, (ro, rs), nb, layout, batch_rows=2)
            check("entegrasyon %s: exam_scores uctan uca" % layout, math.isfinite(e["loss"]) and e["stories"] == st.n,
                  "kayip %.3f bpb %.3f" % (e["loss"], e["bits_per_byte"]))
            dense = _dense_pair(b, mask_fn)
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
                check("entegrasyon model_z: _batch_hidden + geri yayilim torch.nonzero'suz (GPU senkronu yok; vekil: "
                      "nonzero yasak)", *_no_nonzero(model, b, dense))
        except Exception:  # noqa: BLE001
            check("entegrasyon %s: loss_per_target(build_batch)" % layout, False,
                  traceback.format_exc(limit=2).splitlines()[-1])


def _no_nonzero(model, b, dense):
    """torch.nonzero / Tensor.nonzero yasakken model_z _batch_hidden + output_loss geri yayilimi -> (gecti, bilgi)."""
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
    return ok, info


def _dense_pair(b, mask_fn):
    """recipe.dense_mask; mask_fn ikiliyse (global_layers) iki maske (train._attn'in CPU yolu)."""
    import recipe as R
    return tuple(R.dense_mask(b, f) for f in mask_fn) if isinstance(mask_fn, tuple) else R.dense_mask(b, mask_fn)


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
    """train.py uctan uca (CPU, d 16, 1 katman (Model Z 2: yerel + global), gercek build_batch / modeller / exam_scores /
    story_generation): kayip duser, ilk adim kaybi bagimsiz hesapla ayni, lr = wsd_lr, results / samples alanlari, kesilip surdurulen = kesintisiz
    (bit duzeyinde), uzatma = bastan uzun kosu, bitmis kosu durur, ayar farki durur, --steps, yerel kopya sha'si."""
    import traceback
    import recipe as R
    import train as TR
    tp = tokenizer_path()
    if tp is None:
        print("ATLA train: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = _train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, TR.MODEL_Z_GLOBAL_RATIO,
             TR.MODEL_Z_SUMMARIES_LAST)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.MODEL_Z_GLOBAL_RATIO, TR.MODEL_Z_SUMMARIES_LAST = 0.6, 0            # eski varsayilan: L1 / L2 G1 (8 Ekim)
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=3, max_tokens=4)
    base = ["--data", data, "--stream", root, "--device", "cpu", "--d", "16", "--layers", "1", "--heads", "2",
            "--lr", "1e-2", "--checkpoint_minutes", "0", "--optimizer", "adamw"]   # 0: her gunluk sinirinda kayit;
    # AdamW sabit (varsayilan normuon, 8 Ekim); Muon testleri acikca muon, varsayilan testleri _unpin(base)
    out = lambda name: os.path.join(TMP, "runs", name)  # noqa: E731
    state = lambda o: torch.load(os.path.join(o, "agent.pt"), weights_only=False)["state"]  # noqa: E731
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()  # noqa: E731
    exits = lambda argv: _raises(SystemExit, TR.main, argv)  # noqa: E731
    try:
        for model in ("transformer", "model_z"):
            cmd = base + ["--model", model] + (["--layers", "2"] if model == "model_z" else [])   # Model Z G 1
            a = TR.main(cmd + ["--epochs", "2", "--out", out(model + "_A")])
            total, per = a["plan"]["total"], a["plan"]["per_epoch"]
            L = [w["loss"] for w in a["log"]]
            check("train %s: 2 epok kosar, kayip duser" % model, len(L) == total and np.mean(L[-3:]) < np.mean(L[:3]) - 0.5,
                  "%d adim (%s / epok), kayip %.3f -> %.3f" % (total, per, np.mean(L[:3]), np.mean(L[-3:])))
            args = TR._args(cmd + ["--out", "x"])
            st = D.TokenStories(root, data, "train")
            m, mask_fn, layout = TR._build(args, torch.device("cpu"))
            f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
            ro, rs = f["row_offsets"], f["row_stories"]
            b = D.build_batch(st, [rs[ro[r]:ro[r + 1]].tolist() for r in range(4)], layout, "cpu", 64)
            with torch.no_grad():
                want = m.loss_per_target(b, _dense_pair(b, mask_fn))[0].mean().item()
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
        for model in GOLDEN:                                            # sabit deger (belge 33 adim 4, 44)
            if torch.__version__ != GOLDEN_TORCH:
                print("ATLANDI: sabit deger torch %s ile, burada %s" % (GOLDEN_TORCH, torch.__version__), flush=True)
                break
            g = TR.main(base + GOLDEN_ARGS[model] + ["--steps", "6", "--out", out("golden_" + model)])
            st_ = torch.load(os.path.join(out("golden_" + model), "agent.pt"), weights_only=False)["state"]
            per = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in st_.items()}
            whole = hashlib.sha256("".join(k + per[k] for k in sorted(per)).encode()).hexdigest()
            check("train %s: sabit deger (etiketteki kodla olculen kosu, GOLDEN yorumu): ilk 6 kayip ve agirlik sha256"
                  % model, [w["loss"] for w in g["log"]][:6] == GOLDEN[model][
                      "first6"] and whole == GOLDEN[model]["sha"], whole[:16])
        _train_learned(base, root, data, out, state, same, exits, TR)
        _train_archived(base, data, out, state, same, exits, TR)
        _train_equiv(base, root, data, prompts, out, state, same, TR)
        _train_muon(base, root, data, out, state, same, exits, TR)
        _train_global(base, root, data, out, exits, TR)
        _train_bag(base, root, data, out, state, same, exits, TR)
        import copy                                                     # epok sonu eksik batch dolgusu (8 Ekim)
        import types
        import recipe as R
        st = D.TokenStories(root, data, "train")
        f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
        ro, rs = f["row_offsets"], f["row_stories"]
        n_rows = len(ro) - 1
        part = [rs[ro[r]:ro[r + 1]].tolist() for r in range(n_rows - n_rows % TR.BATCH_ROWS, n_rows)]
        padded = part + [[]] * (TR.BATCH_ROWS - len(part))
        diffs = {}
        mz = ["--model", "model_z", "--layers", "2", "--global_layers", "1", "--summaries_last", "0"]
        for name, extra in (("transformer", ["--model", "transformer"]), ("model_z G1", mz),
                            ("model_z G0", mz[:4] + ["--global_layers", "0", "--summaries_last", "0"]),
                            ("model_z summaries_last", mz[:-1] + ["1"])):   # torba dolgulanmaz (bag_index)
            args = TR._args(base + extra + ["--out", "x"])
            m0, mask_fn, layout = TR._build(args, torch.device("cpu"))
            if args.summaries_last:
                from sentence import summaries_last
                mask_fn = m0._masks(True)
            res = []
            for rows in (part, padded):
                m = copy.deepcopy(m0)
                b = D.build_batch(st, rows, layout, "cpu", 64)
                b = summaries_last(b)[0] if args.summaries_last else b
                loss, gn, _ = TR._step(m, b, mask_fn, torch.optim.SGD(m.parameters(), lr=0.0), False)
                res.append((float(loss), float(gn), [torch.zeros_like(p) if p.grad is None else p.grad
                                                     for p in m.parameters()]))
            (l0, g0, d0), (l1, g1, d1) = res
            diffs[name] = max([abs(l0 - l1), abs(g0 - g1)] + [float((x - y).abs().max()) for x, y in zip(d0, d1)])
        seen = []
        fake = types.SimpleNamespace(loss_per_target=lambda b_, attn: (seen.append(b_), 0, 0, 0)[1:])
        TR._Exam(fake, R.document_mask, False).loss_per_target(D.build_batch(st, part, "transformer", "cpu", 64))
        want = D.build_batch(st, padded, "transformer", "cpu", 64)
        fields = ("tokens", "kind", "pos", "doc", "sent", "target", "target_kind", "story_ids")
        check("train epok sonu eksik batch (%d / %d satir): bos satirla tamamlanan batch (CUDA egitim yolu) dolgusuzla ayni "
              "kayip / gradyan normu / gradyan (fp32 <= 1e-6; transformer, Model Z G1 / G0 / summaries_last); "
              "build_batch'in bos satiri = tam dolgu satiri, dolu satirlar aynen; sinav (_Exam) eksik batch'i ayni dolguyla "
              "tamamlar" % (len(part), TR.BATCH_ROWS),
              0 < len(part) < TR.BATCH_ROWS and max(diffs.values()) <= 1e-6
              and all(torch.equal(getattr(seen[0], k), getattr(want, k)) for k in fields)
              and all(torch.equal(getattr(want, k)[:len(part)], getattr(D.build_batch(st, part, "transformer", "cpu", 64), k))
                      for k in fields) and bool((want.kind[len(part):] == D.Kind.PAD).all())
              and bool((want.target[len(part):] == -100).all()),
              "en buyuk fark %s" % {k: "%.1e" % v for k, v in diffs.items()})
    except Exception:  # noqa: BLE001
        check("train", False, traceback.format_exc(limit=3))
    finally:
        (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, TR.MODEL_Z_GLOBAL_RATIO,
         TR.MODEL_Z_SUMMARIES_LAST) = saved


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
    """G'siz Model Z (--global_layers 0) gercek SentenceTransformer ve gercek build_batch ile: kayip duser; ilk adim
    (train._attn: model_z_read_mask) = modelin KENDI maskesiyle loss_per_target; okuma maskesi Z_k'ye kendi cumlesini
    acar; kesilip surdurulen = kesintisiz."""
    import traceback
    import recipe as R
    try:
        cmd = base + ["--model", "model_z", "--global_layers", "0"]
        A = out("mzl_A")
        a = TR.main(cmd + ["--epochs", "2", "--out", A])
        L = [w["loss"] for w in a["log"]]
        st = D.TokenStories(root, data, "train")
        m, mask_fn, layout = TR._build(TR._args(cmd + ["--out", "x"]), torch.device("cpu"))
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
        check("train model_z --global_layers 0: 2 epok kosar, kayip duser; kimlikte learned_z 1; ilk adim (train._attn, "
              "model_z_read_mask) = modelin kendi maskesiyle loss_per_target; okuma maskesi = model_z_mask + yalniz Z_k "
              "satirinda kendi cumlesinin token'lari", np.mean(L[-3:]) < np.mean(L[:3]) - 0.5
              and a["identity"]["learned_z"] == 1 and abs(a["log"][0]["loss"] - want) < 1e-4
              and bool((base_m <= read).all()) and bool(extra.any()) and bool((extra <= (z[:, :, None] & own)).all())
              and int(extra.sum()) == int((z[:, :, None] & own).sum()),
              "kayip %.3f -> %.3f; ilk %.4f / %.4f; ek cift %d" % (np.mean(L[:3]), np.mean(L[-3:]), a["log"][0]["loss"],
                                                                 want, int(extra.sum())))
        stopped, bres = _cut_and_resume(TR, cmd + ["--epochs", "2"], out("mzl_B"))
        check("train model_z --global_layers 0: adim 4'te kesilip surdurulen = kesintisiz (agirlik bit duzeyinde, sinav "
              "ayni)", stopped and same(state(A), state(out("mzl_B"))) and bres["exam"] == dict(
                  a["exam"], seconds=bres["exam"]["seconds"]))
    except Exception:  # noqa: BLE001
        check("train model_z --global_layers 0", False, traceback.format_exc(limit=3).splitlines()[-1])


def _train_archived(base, data, out, state, same, exits, TR):
    """Eski kimlikler (belge 33, 44): temizlik oncesi (6 Ekim; LEGACY alanli) ve formullu (learned_z 0) Model Z kosulari
    load_run'da ve surdurmede DURUR (iletide git etiketi, dosyaya dokunulmaz); eski transformer yuklenir (surdurulmez);
    global_layers / optimizer alani olmayan ogrenilen z kosusu (v2_mzl_d512_l8_lr5e-4_20261006_172433 kimligi) G'siz
    yuklenir.  Kaldirilan argumanlar (--learned_z dahil) veri yuklenmeden DURUR."""
    import traceback
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
        import generate_readings as GR
        dev = torch.device("cpu")
        legacy = lambda d, **kv: (d.update({k: kv.get(k, 0) for k in TR.LEGACY}), d)[1]  # noqa: E731
        formula = "v2-before-formula-cleanup-20261007"
        msgs, loads = {}, {}
        for kind, src, edit in (
                ("tf", out("transformer_A"), lambda d: legacy(d, meaning_sha256=None)),
                ("own", out("mzl_A"), lambda d: (d.pop("learned_z"), legacy(d, meaning_sha256=None, own_vocab=1))),
                ("iota", out("mzl_A"), lambda d: [d.pop(k, None) for k in ("learned_z", "own_vocab", "shared_vocab",
                                                                           "open_z")] and d.update(meaning_sha256="0" * 64)),
                ("formula", out("mzl_A"), lambda d: d.update(learned_z=0)),
                ("formula_old", out("mzl_A"), lambda d: [d.pop(k) for k in ("learned_z", "global_layers", "optimizer")]),
                ("learned_old", out("mzl_A"), lambda d: [d.pop(k) for k in ("global_layers", "optimizer")])):
            dst = out("old_" + kind)
            shutil.copytree(src, dst)
            for name, key in (("checkpoint.pt", "args"), ("agent.pt", "identity")):
                pack = torch.load(os.path.join(dst, name), weights_only=False)
                edit(pack[key])
                torch.save(pack, os.path.join(dst, name))
            mt = os.path.getmtime(os.path.join(dst, "checkpoint.pt"))
            cmd = base + ["--model", "transformer" if kind == "tf" else "model_z"] + (
                [] if kind == "tf" else ["--global_layers", "0"])
            msgs[kind] = (_exit_msg(TR.main, cmd + ["--epochs", "2", "--out", dst, "--resume", "1"]) or "",
                          os.path.getmtime(os.path.join(dst, "checkpoint.pt")) == mt)
            loads[kind] = _exit_msg(GR.load_run, dst, data, dev)
        m = GR.load_run(out("old_learned_old"), data, dev)[0]
        fresh = json.load(open(os.path.join(out("mzl_A"), "config.json")))["identity"]
        tags = dict(tf=TR.TAG, own=TR.TAG, iota=TR.TAG, formula=formula, formula_old=formula)
        check("train: eski kimlikli checkpoint SURDURULMEZ (transformer, own, iota: etiket %s; formullu, alanlari "
              "olmayan formullu: etiket %s; iletide etiket, dosyaya dokunulmaz); alanlari olmayan ogrenilen z bitmis kosu "
              "olarak durur; yeni kimlikte eski alanlar yok, learned_z 1" % (TR.TAG, formula),
              all(tags[k] in msgs[k][0] and msgs[k][1] for k in tags) and msgs["learned_old"] == ("", True) and not set(fresh) & set(TR.LEGACY) and fresh["learned_z"] == 1,
              msgs["formula"][0][:140])
        check("load_run: eski transformer yuklenir; temizlik oncesi (own, iota) ve formullu Model Z DURUR (iletide etiket); "
              "global_layers / optimizer alani olmayan ogrenilen z G'siz yuklenir (agirlik ayni)",
              loads["tf"] is None and all(tags[k] in (loads[k] or "") for k in ("own", "iota", "formula", "formula_old"))
              and loads["learned_old"] is None and m.global_layers == 0 and same(m.state_dict(), state(out("mzl_A"))),
              (loads["formula_old"] or "")[:140])
        gone = [base + ["--model", "model_z", a, v, "--out", out("gone%d" % i)] for i, (a, v) in enumerate(
            (("--meaning", "x.pt"), ("--own_vocab", "1"), ("--shared_vocab", "1"), ("--open_z", "2"), ("--learned_z", "1"),
             ("--learned_z", "0")))]
        check("train: kaldirilan argumanlar (--meaning, --own_vocab, --shared_vocab, --open_z, --learned_z) veri "
              "yuklenmeden DURUR", all(exits(c) and not os.path.exists(c[-1]) for c in gone))
    except Exception:  # noqa: BLE001
        check("train eski kimlikler", False, traceback.format_exc(limit=3))


def _train_muon(base, root, data, out, state, same, exits, TR):
    """--optimizer muon uctan uca (iki model, gercek modeller): kayip duser, ilk adim kaybi AdamW kosusuyla ayni (ayni
    agirlik), ikinci farkli (optimizer gercekten degisti); bolusum (her parametre tam bir grupta, Muon'da yalniz blok
    matrisleri, E degil; formullu Model Z'de z_in AdamW'de); kimlikte optimizer; kesilip surdurulen = kesintisiz; optimizer
    farkiyla surdurme DURUR; optimizer alani olmayan eski checkpoint adamw sayilir; Muon yoksa veri yuklenmeden DURUR."""
    import traceback
    try:
        st = D.TokenStories(root, data, "train")
        mats = {"blocks.*.%s.weight" % k: 1 for k in ("qkv", "proj", "gate_up", "down")}
        for model, adamw_run in (("transformer", "transformer_A"), ("model_z", "mzl_A")):
            cmd = base + ["--model", model, "--optimizer", "muon", "--global_layers", "0"]   # AdamW kosusuyla ayni model
            A = out("muon_%s_A" % model)
            a = TR.main(cmd + ["--epochs", "2", "--out", A])
            L = [w["loss"] for w in a["log"]]
            ref = [w["loss"] for w in json.load(open(os.path.join(out(adamw_run), "results.json")))["log"]]
            ck = torch.load(os.path.join(A, "checkpoint.pt"), weights_only=False)
            check("train %s --optimizer muon: 2 epok kosar, kayip duser; ilk adim kaybi AdamW kosusuyla ayni, ikinci "
                  "farkli; kimlikte optimizer muon; results.json'da bolusum; checkpoint'te iki optimizer" % model,
                  np.mean(L[-3:]) < np.mean(L[:3]) - 0.5 and L[0] == ref[0] and L[1] != ref[1]
                  and a["identity"]["optimizer"] == "muon" and a["optimizer"]["split"]["muon"]["names"] == mats
                  and a["optimizer"]["muon"]["adjust_lr_fn"] == "match_rms_adamw" and set(ck["opt"]) == {"adamw", "muon"},
                  "kayip %.3f -> %.3f (AdamW %.3f -> %.3f); ilk %.4f / %.4f" % (
                      np.mean(L[:3]), np.mean(L[-3:]), np.mean(ref[:3]), np.mean(ref[-3:]), L[0], ref[0]))
            args = TR._args(base + ["--model", model, "--optimizer", "muon", "--out", "x"])
            m = TR._build(args, torch.device("cpu"))[0]
            opt, info = TR._optimizer(m, "muon", 1e-2, False)
            name = {id(p): n for n, p in m.named_parameters()}
            mu = [name[id(p)] for g in opt.muon.param_groups for p in g["params"]]
            ad = [name[id(p)] for g in opt.adamw.param_groups for p in g["params"]]
            want = sorted(n for n, p in m.named_parameters() if n.startswith("blocks.") and p.dim() == 2)
            check("train %s muon bolusumu: her parametre tam bir grupta; Muon = bloklarin 2-B matrisleri; E AdamW'de"
                  % model, sorted(mu + ad) == sorted(name.values()) and len(set(mu + ad)) == len(mu + ad)
                  and sorted(mu) == want and "E.weight" in ad and not info["split"]["adamw_decay"]["tensors"],
                  "Muon %d, AdamW %d tensor" % (len(mu), len(ad)))
            stopped, bres = _cut_and_resume(TR, cmd + ["--epochs", "2"], out("muon_%s_B" % model))
            check("train %s --optimizer muon: adim 4'te kesilip surdurulen = kesintisiz (agirlik bit duzeyinde, sinav ayni)"
                  % model, stopped and same(state(A), state(out("muon_%s_B" % model))) and bres["exam"] == dict(
                      a["exam"], seconds=bres["exam"]["seconds"]))
            plain = base + ["--model", model]
            mt = [os.path.getmtime(os.path.join(p, "checkpoint.pt")) for p in (A, out(adamw_run))]
            check("train %s: optimizer farkiyla surdurme checkpoint yuklenmeden DURUR (muon -> adamw, adamw -> muon), "
                  "dosyalara dokunulmaz" % model, exits(plain + ["--epochs", "2", "--out", A, "--resume", "1"])
                  and exits(cmd + ["--epochs", "2", "--out", out(adamw_run), "--resume", "1"])
                  and mt == [os.path.getmtime(os.path.join(p, "checkpoint.pt")) for p in (A, out(adamw_run))])
        old = out("noopt")                                                # optimizer alanindan onceki kosu
        shutil.copytree(out("transformer_A"), old)
        pack = torch.load(os.path.join(old, "checkpoint.pt"), weights_only=False)
        del pack["args"]["optimizer"]
        torch.save(pack, os.path.join(old, "checkpoint.pt"))
        cmd = base + ["--model", "transformer", "--epochs", "2", "--out", old, "--resume", "1"]
        r = _exit_msg(TR.main, cmd)
        check("train: optimizer alani olmayan checkpoint adamw sayilir (bitmis kosu olarak durur), muon ile DURUR",
              r is None and exits(cmd + ["--optimizer", "muon"]), str(r))
        real = torch.optim.Muon
        del torch.optim.Muon
        try:
            msg = _exit_msg(TR.main, base + ["--model", "transformer", "--optimizer", "muon", "--out", out("nomuon")])
        finally:
            torch.optim.Muon = real
        try:
            del torch.optim.Muon
            msg2 = _exit_msg(TR.main, _unpin(base) + ["--model", "transformer", "--out", out("nomuon_default")])
        finally:
            torch.optim.Muon = real
        check("train: torch.optim.Muon yoksa --optimizer muon ve varsayilan (normuon) veri yuklenmeden DURUR (AdamW'ye "
              "dusmez)",
              msg is not None and "Muon" in msg and not os.path.exists(out("nomuon")) and msg2 is not None and "Muon" in msg2
              and not os.path.exists(out("nomuon_default")), str(msg))
    except Exception:  # noqa: BLE001
        check("train --optimizer muon", False, traceback.format_exc(limit=3))


def _train_global(base, root, data, out, exits, TR):
    """--global_layers (belge 40 s6.2 Deney G) gercek SentenceTransformer ve build_batch ile: 2 epok kosar, kayip duser;
    kimlikte global_layers; ilk adim kaybi (train._attn ikilisi) = modelin kendi (dense) maskesiyle loss_per_target;
    load_run kimlikten okur; transformer, formullu yol, N > katman DURUR; global_layers farkiyla surdurme checkpoint
    yuklenmeden DURUR; alani olmayan eski checkpoint 0 sayilir.  Bit duzeyinde surdurme sinanmaz (kullanici, 7 Ekim:
    SS kisa deneme, kural 3 istisnasi)."""
    import traceback
    import recipe as R
    try:
        cmd = base + ["--model", "model_z", "--layers", "2", "--global_layers", "1"]
        A = out("glob_A")
        a = TR.main(cmd + ["--epochs", "2", "--out", A])
        L = [w["loss"] for w in a["log"]]
        st = D.TokenStories(root, data, "train")
        m, mask_fn, layout = TR._build(TR._args(cmd + ["--out", "x"]), torch.device("cpu"))
        f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
        ro, rs = f["row_offsets"], f["row_stories"]
        b = D.build_batch(st, [rs[ro[r]:ro[r + 1]].tolist() for r in range(4)], layout, "cpu", 64)
        with torch.no_grad():
            want = m.loss_per_target(b)[0].mean().item()                  # modelin kendi dense maske ikilisi
            via = m.loss_per_target(b, TR._attn(b, mask_fn, False))[0].mean().item()
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
        import generate_readings as GR
        lm = GR.load_run(A, data, torch.device("cpu"))[0]
        check("train model_z --global_layers 1: 2 epok kosar, kayip duser; kimlikte global_layers 1; ilk adim kaybi = "
              "modelin kendi maskesiyle loss_per_target (train._attn ikilisi); load_run global_layers'i kimlikten okur",
              np.mean(L[-3:]) < np.mean(L[:3]) - 0.5 and a["identity"]["global_layers"] == 1
              and abs(a["log"][0]["loss"] - want) < 1e-4 and via == want and isinstance(mask_fn, tuple)
              and lm.global_layers == 1 and m.global_layers == 1,
              "kayip %.3f -> %.3f; ilk %.4f / %.4f" % (np.mean(L[:3]), np.mean(L[-3:]), a["log"][0]["loss"], want))
        gone = [base + ["--model", "transformer", "--global_layers", "1", "--out", out("g_tf")],
                base + ["--model", "model_z", "--global_layers", "2", "--out", out("g_big")]]   # base: 1 katman
        mt = [os.path.getmtime(os.path.join(p, "checkpoint.pt")) for p in (A, out("mzl_A"))]
        check("train: transformer + --global_layers ve N > katman veri yuklenmeden DURUR; "
              "global_layers farkiyla surdurme (1 -> 0, 0 -> 1) checkpoint yuklenmeden DURUR, dosyalara dokunulmaz",
              all(exits(c) and not os.path.exists(c[-1]) for c in gone)
              and exits(base + ["--model", "model_z", "--layers", "2", "--global_layers", "0", "--epochs", "2", "--out", A,
                                "--resume", "1"])
              and exits(base + ["--model", "model_z", "--global_layers", "1", "--epochs", "2", "--out", out("mzl_A"),
                                "--resume", "1"])
              and mt == [os.path.getmtime(os.path.join(p, "checkpoint.pt")) for p in (A, out("mzl_A"))])
        old = out("noglobal")                                             # global_layers alanindan onceki kosu
        shutil.copytree(out("mzl_A"), old)
        pack = torch.load(os.path.join(old, "checkpoint.pt"), weights_only=False)
        del pack["args"]["global_layers"]
        torch.save(pack, os.path.join(old, "checkpoint.pt"))
        c = base + ["--model", "model_z", "--global_layers", "0", "--epochs", "2", "--out", old, "--resume", "1"]
        r = _exit_msg(TR.main, c)
        check("train: global_layers alani olmayan checkpoint 0 sayilir (bitmis kosu olarak durur), 1 ile DURUR",
              r is None and exits(c + ["--global_layers", "1"]), str(r))
        pinned = (TR.MODEL_Z_GLOBAL_RATIO, TR.MODEL_Z_SUMMARIES_LAST)
        TR.MODEL_Z_GLOBAL_RATIO, TR.MODEL_Z_SUMMARIES_LAST = 1 / 3, 1       # train.py'nin gercek varsayilani
        try:
            pick = lambda x: (x.global_layers, x.summaries_last, x.optimizer)  # noqa: E731
            arg = lambda a: TR._args(_unpin(base) + a + ["--out", "x"])  # noqa: E731
            nolr = lambda a: a[:a.index("--lr")] + a[a.index("--lr") + 2:]  # noqa: E731  (--lr verilmez: varsayilan auto)
            dflt = {k: pick(arg(a)) for k, a in (
                ("model_z L10", ["--model", "model_z", "--layers", "10"]),
                ("model_z L12", ["--model", "model_z", "--layers", "12"]),
                ("model_z L24", ["--model", "model_z", "--layers", "24"]),
                ("model_z L12 auto", ["--model", "model_z", "--layers", "12", "--global_layers", "auto"]),
                ("transformer", ["--model", "transformer", "--layers", "12"]),
                ("transformer auto", ["--model", "transformer", "--layers", "12", "--global_layers", "auto"]),
                ("model_z acik 0", ["--model", "model_z", "--global_layers", "0", "--summaries_last", "0"]),
                ("model_z torba", ["--model", "model_z", "--layers", "12", "--bag_k", "64"]),
                ("model_z adamw", ["--model", "model_z", "--layers", "12", "--optimizer", "adamw"]))}
            r_def = TR.main(nolr(_unpin(base)) + ["--model", "model_z", "--layers", "6", "--steps", "3",
                                                  "--out", out("g_default")])
            idt = r_def["identity"]
            def_again = _exit_msg(TR.main, nolr(_unpin(base)) + ["--model", "model_z", "--layers", "6", "--lr", "auto",
                                                                 "--steps", "3", "--out", out("g_default"), "--resume", "1"])
            OLD = _unpin(base) + ["--model", "model_z", "--layers", "4", "--epochs", "2"]    # eski kosu: G3 (auto 1)
            TR.main(OLD + ["--global_layers", "3", "--summaries_last", "0", "--optimizer", "muon", "--stop_step", "4",
                           "--out", out("old_run")])
            ck = os.path.join(out("old_run"), "checkpoint.pt")
            pack = torch.load(ck, weights_only=False)
            del pack["args"]["summaries_last"], pack["args"]["glob_kv_heads"]   # alanlarindan onceki kosu
            torch.save(pack, ck)
            r_old = TR.main(nolr(OLD) + ["--out", out("old_run"), "--resume", "1"])   # optimizer, lr de kimlikten
            r_ref = TR.main(OLD + ["--global_layers", "3", "--summaries_last", "0", "--optimizer", "muon", "--out",
                                   out("old_ref")])
            fp8_cpu = _exit_msg(TR.main, OLD + ["--global_layers", "1", "--fp8", "tensorwise", "--out", out("fp8_cpu")])
            stop_bag = _exit_msg(TR.main, OLD + ["--summaries_last", "1", "--bag_k", "64", "--out", out("sl_bag")])
            kv = {h: arg(["--model", "model_z", "--layers", "3", "--heads", str(h), "--glob_kv_heads", "auto"]).glob_kv_heads
                  for h in (16, 12, 8, 4)}
            kv_bad = [_exit_msg(TR._args, _unpin(base) + ["--model", "model_z", "--heads", str(h), "--glob_kv_heads", "auto",
                                                         "--out", "x"]) for h in (6, 2)]
            kv_def = (arg(["--model", "model_z"]).glob_kv_heads, arg(["--model", "transformer"]).glob_kv_heads)
            lr = {d_: TR._args(nolr(_unpin(base)) + ["--model", "model_z", "--d", str(d_), "--out", "x"]).lr
                  for d_ in (768, 1024, 1280)}
            lr_auto = arg(["--model", "model_z", "--d", "1024", "--lr", "auto"]).lr
            lr_adamw = [_exit_msg(TR._args, a + ["--model", "model_z", "--out", "x"])
                        for a in (nolr(base), base + ["--lr", "auto"])]
            mt = os.path.getmtime(os.path.join(out("mzl_A"), "checkpoint.pt"))
            lr_inherit = _exit_msg(TR.main, _unpin(base) + ["--model", "model_z", "--global_layers", "0", "--lr", "auto",
                                                            "--epochs", "2", "--out", out("mzl_A"), "--resume", "1"])
            adamw_resume = _exit_msg(TR.main, nolr(_unpin(base)) + ["--model", "model_z", "--global_layers", "0",
                                                                    "--epochs", "2", "--out", out("mzl_A"), "--resume", "1"])
        finally:
            TR.MODEL_Z_GLOBAL_RATIO, TR.MODEL_Z_SUMMARIES_LAST = pinned
        sw = lambda n_: torch.load(os.path.join(out(n_), "agent.pt"), weights_only=False)["state"]  # noqa: E731
        check("train: varsayilanlar (8 Ekim): model_z G auto = round(L / 3) (L10 3, L12 4, L24 8) + summaries_last 1, "
              "transformer 0 / 0 (auto da 0), torbada summaries_last 0 (acik 1 DURUR); optimizer "
              "normuon; varsayilan kosu (L6 -> G2, --lr verilmeden auto) egitir, sinav ve okuma uretir, segments'ta fp8 "
              "none, --lr auto ile --resume kimlik denetiminden gecer; eski kimlikli kosu (G3 L4, muon, lr 1e-2, "
              "summaries_last / glob_kv_heads alani yok) bayraksiz (--lr dahil) --resume ile G3 / lr 1e-2 kalir (auto "
              "1'e / 0,0139'a donmez) = kesintisiz (bit); CPU'da --fp8 DURUR",
              dflt == {"model_z L10": (3, 1, "normuon"), "model_z L12": (4, 1, "normuon"),
                       "model_z L24": (8, 1, "normuon"), "model_z L12 auto": (4, 1, "normuon"),
                       "transformer": (0, 0, "normuon"), "transformer auto": (0, 0, "normuon"),
                       "model_z acik 0": (0, 0, "normuon"), "model_z torba": (4, 0, "normuon"),
                       "model_z adamw": (4, 1, "adamw")}
              and (idt["global_layers"], idt["summaries_last"], idt["optimizer"]) == (2, 1, "normuon")
              and idt["lr"] == 2e-3 * (768 / 16) ** 0.5 and def_again is None and r_old["identity"]["lr"] == 1e-2
              and r_def["segments"] == [dict(start=0, fp8="none", fp8_linears=0)]
              and r_old["identity"]["optimizer"] == "muon" and len(r_old["segments"]) == 2
              and fp8_cpu is not None and "CUDA" in fp8_cpu
              and np.isfinite(r_def["exam"]["loss"]) and r_def["generation"]
              and os.path.exists(os.path.join(out("g_default"), "samples.txt"))
              and (r_old["identity"]["global_layers"], r_old["identity"]["summaries_last"],
                   r_old["identity"]["glob_kv_heads"]) == (3, 0, 0) and r_old["finished"]
              and [w["loss"] for w in r_old["log"]] == [w["loss"] for w in r_ref["log"]]
              and all(torch.equal(v, sw("old_ref")[k]) for k, v in sw("old_run").items()) and stop_bag is not None,
              "%s; resume %s" % (dflt, def_again))
        check("train auto (8 Ekim): --glob_kv_heads auto = heads / 4 (16 4, 12 3, 8 2, 4 1), 4'e bolunmezse (6, 2) DUR, "
              "varsayilan 0; --lr verilmezse auto = 2e-3 sqrt(768 / d) (d768 tam 2e-3, d1024 1,732e-3, d1280 1,549e-3; "
              "acik auto ayni), adamw'de (varsayilan ya da acik auto) DUR, --resume'da kimlikten gelen adamw ile de DUR "
              "(checkpoint'e dokunulmaz); adamw kosusu --lr'siz --resume ile kimlikten lr alir",
              kv == {16: 4, 12: 3, 8: 2, 4: 1} and all(m_ is not None and "bolunmuyor" in m_ for m_ in kv_bad)
              and kv_def == (0, 0) and lr[768] == 2e-3 and abs(lr[1024] - 1.7320508e-3) < 1e-10
              and abs(lr[1280] - 1.5491933e-3) < 1e-10 and lr_auto == lr[1024]
              and all(m_ is not None and "adamw" in m_ for m_ in lr_adamw)
              and lr_inherit is not None and "adamw" in lr_inherit and adamw_resume is None
              and os.path.getmtime(os.path.join(out("mzl_A"), "checkpoint.pt")) == mt,
              "kv %s, lr %s, kotu %s / %s / %s" % (kv, lr, kv_bad, lr_inherit, adamw_resume))
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
        import generate_readings as GR
        N_ = base + ["--model", "model_z", "--layers", "2", "--optimizer", "normuon", "--epochs", "2"]   # kullanici, 8 Ekim
        nr = TR.main(N_ + ["--out", out("normuon_A")])
        stopped, nr2 = _cut_and_resume(TR, N_, out("normuon_cut"))
        st_a = torch.load(os.path.join(out("normuon_A"), "agent.pt"), weights_only=False)["state"]
        st_b = torch.load(os.path.join(out("normuon_cut"), "agent.pt"), weights_only=False)["state"]
        ck = torch.load(os.path.join(out("normuon_cut"), "checkpoint.pt"), weights_only=False)["opt"]["muon"]
        check("train --optimizer normuon: kosar, kayip duser, kimlikte normuon; adim 4'te kesilip surdurulen = kesintisiz "
              "(bit); checkpoint'te second_momentum_buffer ve beta2",
              nr["log"][-1]["loss"] < nr["log"][0]["loss"] and nr["identity"]["optimizer"] == "normuon" and stopped
              and all(torch.equal(st_a[k], st_b[k]) for k in st_a)
              and [w["loss"] for w in nr2["log"]] == [w["loss"] for w in nr["log"]]
              and "second_momentum_buffer" in ck["state"][0] and ck["param_groups"][0]["beta2"] == R.NORMUON_BETA2)
        S_ = base + ["--model", "model_z", "--layers", "2", "--epochs", "2"]   # --stop_step (kullanici, 8 Ekim)
        sa = TR.main(S_ + ["--out", out("stop_full")])
        ss = TR.main(S_ + ["--stop_step", "4", "--out", out("stop_cut")])
        sj = json.load(open(os.path.join(out("stop_cut"), "results.json")))
        ok_files = all(os.path.exists(os.path.join(out("stop_cut"), f)) for f in ("agent.pt", "checkpoint.pt"))
        ck_step = torch.load(os.path.join(out("stop_cut"), "checkpoint.pt"), weights_only=False)["step"]
        early = _exit_msg(TR.main, S_ + ["--stop_step", "4", "--out", out("stop_cut"), "--resume", "1"])   # N <= adim: DUR
        sr = TR.main(S_ + ["--out", out("stop_cut"), "--resume", "1"])
        full_l = [w["loss"] for w in sa["log"]]
        check("train --stop_step 4: durur, checkpoint.pt (adim 4) + agent.pt + results.json (finished False, stopped_at 4, "
              "okuma yok); kayip egrisi stop'suz kosuyla adim 4'e kadar bit ayni; --resume 1 devam = kesintisiz (agirlik bit); N <= surdurulen adim DURUR",
              ok_files and ck_step == 4 and sj["finished"] is False and sj["stopped_at"] == 4
              and sj["readings_skipped"] == "stop_step" and [w["loss"] for w in ss["log"]] == full_l[:4]
              and [w["loss"] for w in sr["log"]] == full_l and sr["finished"]
              and all(torch.equal(x, y) for x, y in zip(*(torch.load(os.path.join(out(n_), "agent.pt"),
                                                                     weights_only=False)["state"].values()
                                                          for n_ in ("stop_full", "stop_cut"))))
              and early is not None,
              "%s / %s" % ([w["loss"] for w in ss["log"]], full_l[:4]))
        L_ = base + ["--model", "model_z", "--layers", "2", "--epochs", "2"]   # --summaries_last (belge 66)
        l0 = TR.main(L_ + ["--out", out("last_off")])
        l1 = TR.main(L_ + ["--summaries_last", "1", "--out", out("last_on")])
        stopped_l, l2 = _cut_and_resume(TR, L_ + ["--summaries_last", "1"], out("last_cut"))
        sl = lambda n_: torch.load(os.path.join(out(n_), "agent.pt"), weights_only=False)["state"]  # noqa: E731
        d_loss = max(abs(a_["loss"] - b_["loss"]) for a_, b_ in zip(l0["log"], l1["log"]))
        check("train --summaries_last 1: kayip egrisi bugunku duzenle esit (<= 1e-3, fp32 toplama sirasi), sinav kaybi "
              "esit (<= 1e-3), acc ayni; kimlikte summaries_last; kesilip surdurulen = kesintisiz (bit); transformer ve "
              "torba ile DURUR",
              d_loss <= 1e-3 and abs(l0["exam"]["loss"] - l1["exam"]["loss"]) <= 1e-3
              and abs(l0["exam"]["acc"] - l1["exam"]["acc"]) <= 1e-3 and l1["identity"]["summaries_last"] == 1
              and stopped_l and all(torch.equal(sl("last_on")[k], sl("last_cut")[k]) for k in sl("last_on"))
              and [w["loss"] for w in l2["log"]] == [w["loss"] for w in l1["log"]]
              and exits(base + ["--model", "transformer", "--summaries_last", "1", "--out", out("last_tf")])
              and exits(L_ + ["--summaries_last", "1", "--bag_k", "64", "--out", out("last_bag")]),
              "kayip farki %.1e, sinav %.4f / %.4f" % (d_loss, l0["exam"]["loss"], l1["exam"]["loss"]))
        old = {}                                                          # kod temizligi (belge 77)
        for name_, fields in (("kept", dict(z_bow_weight=0.0, layer_plan=None)), ("zbow", dict(z_bow_weight=0.5)),
                              ("mid", dict(layer_plan="loc1,mid1,glob1"))):
            D_ = out("clean_" + name_)
            TR.main(S_ + ["--stop_step", "4", "--out", D_])                    # 7bec0ae'nin yazdigi alanlar
            for fn, key in (("checkpoint.pt", "args"), ("agent.pt", "identity")):
                pack = torch.load(os.path.join(D_, fn), weights_only=False)
                pack[key].update(fields)
                for f_ in ("z_reads_all", "glob_drop"):                   # 2759b46 kimliginde bu alanlar yok
                    pack[key].pop(f_)
                torch.save(pack, os.path.join(D_, fn))
            mt_ = os.path.getmtime(os.path.join(D_, "checkpoint.pt"))
            old[name_] = (_exit_msg(TR.main, S_ + ["--out", D_, "--resume", "1"]),
                          _exit_msg(GR.load_run, D_, data, torch.device("cpu")), mt_)
        kept = json.load(open(os.path.join(out("clean_kept"), "results.json")))
        check("kod temizligi (belge 77): kimliginde z_bow_weight 0 / layer_plan None olan, z_reads_all / glob_drop alani "
              "olmayan kosu (d1024 tam kosusu gibi) yeni "
              "kodla --resume edilir = kesintisiz (kayip egrisi bit); z_bow_weight 0,5 ya da layer_plan'li kosu surdurmede "
              "ve load_run'da DURUR (iletide commit 7bec0ae), checkpoint'e dokunulmaz; --z_bow_weight / --layer_plan "
              "argumanlari yok",
              old["kept"][0] is None and kept["finished"] and [w["loss"] for w in kept["log"]] == full_l
              and all(old[k][0] is not None and "7bec0ae" in old[k][0] and old[k][1] is not None and "7bec0ae" in old[k][1]
                      and os.path.getmtime(os.path.join(out("clean_" + k), "checkpoint.pt")) == old[k][2]
                      for k in ("zbow", "mid"))
              and _raises(SystemExit, TR._args, S_ + ["--z_bow_weight", "0.5", "--out", "x"])
              and _raises(SystemExit, TR._args, S_ + ["--layer_plan", "loc1,glob1", "--out", "x"]),
              str({k: (v[0] or "")[:60] for k, v in old.items()}))
        ZR = base + ["--model", "model_z", "--layers", "3", "--global_layers", "1", "--z_reads_all", "2", "--summaries_last",
                     "1", "--epochs", "2"]                                  # belge 79 oneri 1 + 3 (kullanici, 8 Ekim)
        zr = TR.main(ZR + ["--glob_drop", "0.25", "--out", out("zr_A")])
        stopped_zr, zr2 = _cut_and_resume(TR, ZR + ["--glob_drop", "0.25"], out("zr_cut"))
        zr0 = TR.main(ZR + ["--out", out("zr_nodrop")])
        drops = [bool(np.random.default_rng(np.random.SeedSequence(0, spawn_key=(s_, 1))).random() < 0.25)
                 for s_ in range(len(zr["log"]))]
        k_ = drops.index(True)                                             # ilk birakilan adim (3): oncesi ayni, kendisi farkli
        la, l0 = [w["loss"] for w in zr["log"]], [w["loss"] for w in zr0["log"]]
        szr = lambda n_: torch.load(os.path.join(out(n_), "agent.pt"), weights_only=False)["state"]  # noqa: E731
        lzr = GR.load_run(out("zr_A"), data, torch.device("cpu"))[0]
        bad_zr = [_exit_msg(TR.main, base + a + ["--out", out("zr_bad%d" % i)]) for i, a in enumerate((
            ["--model", "transformer", "--z_reads_all", "1"], ["--model", "model_z", "--layers", "3", "--global_layers", "1",
                                                              "--z_reads_all", "3"],
            ["--model", "model_z", "--layers", "2", "--global_layers", "0", "--glob_drop", "0.5"],
            ["--model", "model_z", "--layers", "2", "--global_layers", "1", "--glob_drop", "1.5"]))]
        check("train --z_reads_all 2 --glob_drop 0,25 (G1, summaries_last 1): kosar, sinav / okuma uretir, kimlikte; kesilip "
              "surdurulen = kesintisiz (bit; birakma adim tohumlu); glob_drop 0 kosusuyla ilk birakilan adima (%d) kadar kayip "
              "bit ayni, o adimda farkli; load_run z_reads_all'u kimlikten kurar; transformer / yerel katmandan fazla / "
              "glob'suz / 0..1 disi DURUR" % k_,
              zr["identity"]["z_reads_all"] == 2 and zr["identity"]["glob_drop"] == 0.25 and np.isfinite(zr["exam"]["loss"])
              and zr["generation"] and stopped_zr and all(torch.equal(szr("zr_A")[k], szr("zr_cut")[k]) for k in szr("zr_A"))
              and [w["loss"] for w in zr2["log"]] == la and k_ > 0 and la[:k_] == l0[:k_] and la[k_] != l0[k_]
              and lzr.z_reads_all == 2 and all(m_ is not None and "DUR" in m_ for m_ in bad_zr),
              "birakilan adimlar %s; DUR %s" % ([i for i, d_ in enumerate(drops) if d_][:6], [(m_ or "")[:40] for m_ in bad_zr]))
        QK = base + ["--model", "model_z", "--layers", "2", "--heads", "2", "--global_layers", "1", "--epochs", "2",
                     "--glob_kv_heads", "1"]                                 # GQA (8 Ekim)
        q1 = TR.main(QK + ["--out", out("gqa_A")])
        stopped_q, q2 = _cut_and_resume(TR, QK, out("gqa_cut"))
        sq = lambda n_: torch.load(os.path.join(out(n_), "agent.pt"), weights_only=False)["state"]  # noqa: E731
        qt = TR.main(base + ["--model", "transformer", "--heads", "2", "--glob_kv_heads", "1", "--steps", "3",
                             "--out", out("gqa_tf")])
        QK0 = [x for x in QK if x != "--glob_kv_heads"][:-1]               # bayraksiz: kimlikten 1 (INHERIT)
        kv_inherit = _exit_msg(TR.main, QK0 + ["--out", out("gqa_A"), "--resume", "1"])
        check("train --glob_kv_heads 1 (heads 2): Model Z kosar, kimlikte; glob qkv daralmis (agent.pt), load_run yukler; "
              "kesilip surdurulen = kesintisiz (bit); transformer'da da kosar; bolen degil / glob'suz model_z / acik 0 ile "
              "surdurme DURUR; bayraksiz surdurme kimlikten 1 alir (bitmis kosu olarak doner)",
              q1["identity"]["glob_kv_heads"] == 1 and tuple(sq("gqa_A")["blocks.1.qkv.weight"].shape) == (32, 16)
              and tuple(sq("gqa_A")["blocks.0.qkv.weight"].shape) == (48, 16) and stopped_q
              and all(torch.equal(sq("gqa_A")[k], sq("gqa_cut")[k]) for k in sq("gqa_A"))
              and [w["loss"] for w in q2["log"]] == [w["loss"] for w in q1["log"]]
              and GR.load_run(out("gqa_A"), data, torch.device("cpu"))[0].blocks[1].kv_heads == 1
              and np.isfinite(qt["exam"]["loss"])
              and exits(base + ["--model", "model_z", "--heads", "2", "--glob_kv_heads", "3", "--out", out("gqa_bad")])
              and exits(base + ["--model", "model_z", "--global_layers", "0", "--glob_kv_heads", "1", "--out", out("gqa_g0")])
              and exits(QK0 + ["--glob_kv_heads", "0", "--out", out("gqa_A"), "--resume", "1", "--epochs", "3"])
              and kv_inherit is None, str(kv_inherit))
    except Exception:  # noqa: BLE001
        check("train --global_layers", False, traceback.format_exc(limit=3))


def _train_bag(base, root, data, out, state, same, exits, TR):
    """--bag_k (ogrenen torba) gercek modellerle: Model Z + G ve transformer kosar, kayip duser; ilk gunluk kaybi (hizli
    yol) = ayni ilk agirlikla loss_per_target (output_logprobs; sinav tanimi); ana modelin ilk agirliklari torbasizla ayni;
    gunlukte kaynaklar, p(DIGER), tam CE, secici kaybi; kesilip surdurulen = kesintisiz (bit); load_run strict yukler,
    uretir; sayim dosyasi yok / pay disarida / bag_k farkiyla surdurme DURUR."""
    import traceback
    import recipe as R
    try:
        cpath = os.path.join(data, "train_token_counts.npy")
        np.save(cpath, D.token_counts(root))
        cmd = base + ["--model", "model_z", "--layers", "2", "--global_layers", "1", "--bag_k", "12", "--bag_core", "5",
                      "--bag_full_frac", "0.2", "--bag_weight", "0.5", "--steps", "8"]
        A = out("bag_A")
        a = TR.main(cmd + ["--out", A])
        L = [w["loss"] for w in a["log"]]
        args = TR._args(cmd + ["--out", "x"])
        core = R.core_ids(np.load(cpath), 5)
        args.bag_n_core = len(core)
        m, mask_fn, layout = TR._build(args, torch.device("cpu"))
        m.bag.fill(core, np.load(cpath))
        plain = TR._build(TR._args([x for x in cmd if x not in ("--bag_k", "12")] + ["--out", "x"]),
                          torch.device("cpu"))[0].state_dict()
        st = D.TokenStories(root, data, "train")
        f = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
        ro, rs = f["row_offsets"], f["row_stories"]
        b = D.build_batch(st, [rs[ro[r]:ro[r + 1]].tolist() for r in range(4)], layout, "cpu", 64)
        with torch.no_grad():
            want = m.loss_per_target(b)[0].mean().item()
        idt, bg = a["identity"], a["log"][-1]["bag"]
        check("train --bag_k (Model Z + G): kosar, kayip duser; ilk gunluk kaybi (hizli yol) = loss_per_target; ana model "
              "ilk agirliklari torbasizla ayni; kimlikte bag_k / bag_core / bag_weight / bag_full_frac / sha; gunlukte "
              "C + P + L + kacan = 1, p(DIGER), tam CE, secici kaybi",
              L[-1] < L[0] and abs(a["log"][0]["loss"] - want) < 1e-4
              and all(torch.equal(v, m.state_dict()[k]) for k, v in plain.items())
              and (idt["bag_k"], idt["bag_core"], idt["bag_weight"], idt["bag_full_frac"]) == (12, 5, 0.5, 0.2)
              and len(idt["bag_core_sha256"]) == 64 and abs(bg["c"] + bg["p"] + bg["l"] + bg["miss"] - 1) < 2e-4
              and 0 < bg["p_other"] < 1 and np.isfinite(bg["full_ce"]) and np.isfinite(bg["loss_selector"])
              and np.isfinite(a["exam"]["loss"]),
              "kayip %.3f -> %.3f; ilk %.4f / %.4f; %s" % (L[0], L[-1], a["log"][0]["loss"], want, bg))
        t = TR.main(base + ["--model", "transformer", "--bag_k", "12", "--bag_core", "5", "--steps", "4",
                            "--out", out("bag_tf")])
        check("train --bag_k (transformer, ozet = END): ayni yol kosar, sinav sonlu",
              t["identity"]["bag_k"] == 12 and np.isfinite(t["exam"]["loss"]) and "bag" in t["log"][-1])
        stopped, r = _cut_and_resume(TR, cmd, out("bag_cut"))
        check("train --bag_k: adim 4'te kesilip surdurulen = kesintisiz (model, secici, DIGER bit duzeyinde)",
              stopped and same(state(A), state(out("bag_cut"))) and [w["loss"] for w in r["log"]] == L)
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
        import generate_readings as GR
        lm = GR.load_run(A, data, torch.device("cpu"))[0]
        gen = lm.generate([[[int(x) for x in st.sentences(0)[0]]]], 2, 4)
        check("train: load_run torbali kosuyu strict yukler (C state_dict'ten), generate calisir",
              torch.equal(lm.bag.core, torch.as_tensor(core)) and len(gen) == 1 and isinstance(gen[0][0], list))
        os.rename(cpath, cpath + ".x")
        no_counts = exits(cmd + ["--out", out("bag_nocounts")])
        os.rename(cpath + ".x", cpath)
        check("train: sayim dosyasi yok, --bag_full_frac 0..1 disi, bag_k farkiyla surdurme DURUR",
              no_counts and exits(cmd + ["--bag_full_frac", "2", "--out", out("bag_frac")])
              and exits([x if x != "12" else "13" for x in cmd] + ["--out", A, "--resume", "1", "--steps", "10"]))
    except Exception:  # noqa: BLE001
        check("train --bag_k", False, traceback.format_exc(limit=3))


EQUIV_TAG = "v2-before-formula-cleanup-20261007"
EQUIV_RUNNER = """import torch  # noqa: I001
import json
import sys
v2, argv = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, v2 + "/common")
import train as TR
TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
TR.READING_PROMPTS, TR.READING_LIMITS = sys.argv[3], dict(max_sentences=3, max_tokens=4)
TR.main(argv)
"""


def _train_equiv(base, root, data, prompts, out, state, same, TR):
    """Formul temizligi esdegerligi (belge 44; belge 33 adim 4 deseni): etiketteki train.py (git archive, ayri surec) =
    bugunku train.py, varsayilan Muon ile Model Z (G 1), G'siz Model Z ve transformer, 2 epok: son agirlik bit duzeyinde,
    adim adim kayip / gradyan normu / lr, sinavlar, okuma metni ve olculeri ayni."""
    import subprocess
    import traceback
    import zipfile
    try:
        tmp = tempfile.mkdtemp(dir=TMP)
        z = os.path.join(tmp, "tag.zip")
        repo = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
        subprocess.run(["git", "-C", repo, "archive", "--format=zip", "-o", z, EQUIV_TAG, "deneme2/v2"], check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("ATLA esdegerlik: git etiketi %s okunamadi (%s)" % (EQUIV_TAG, e), flush=True)
        return
    try:
        zipfile.ZipFile(z).extractall(tmp)
        runner = os.path.join(tmp, "runner.py")
        open(runner, "w", encoding="utf-8").write(EQUIV_RUNNER)
        drop = lambda r: [{k: v for k, v in dict(w, seconds=0, ms_per_step=0, tokens_per_sec=0).items()  # noqa: E731
                           if k != "peak_reserved_gb"} for w in r["log"]]            # belge 48'de eklenen alan
        exams = lambda r: [dict(e, seconds=0) for e in r["exams"]]  # noqa: E731
        bad, info = [], []
        for name, extra in (("model_z", ["--model", "model_z", "--layers", "2"]),
                            ("model_z_g0", ["--model", "model_z", "--layers", "2", "--global_layers", "0"]),
                            ("transformer", ["--model", "transformer"])):
            argv = _unpin(base) + extra + ["--epochs", "2", "--optimizer", "muon"]   # etiketin varsayilani muon
            old, new = out("equiv_old_" + name), out("equiv_new_" + name)
            p = subprocess.run([sys.executable, runner, os.path.join(tmp, "deneme2", "v2"), json.dumps(argv + [
                "--out", old]), prompts], capture_output=True, text=True)
            if p.returncode:
                bad.append(name + ": eski kod kosmadi " + p.stderr[-300:])
                continue
            b = TR.main(argv + ["--out", new])
            a = json.load(open(os.path.join(old, "results.json")))
            sa, sb = (json.load(open(os.path.join(d, "samples.json"), encoding="utf-8")) for d in (old, new))
            ok = dict(agirlik=same(state(old), state(new)), kayip=drop(a) == drop(b), sinav=exams(a) == exams(b),
                      okuma=sa == sb, optimizer=a["identity"]["optimizer"] == b["identity"]["optimizer"] == "muon")
            bad += ["%s: %s" % (name, k) for k, v in ok.items() if not v]
            info.append("%s %d adim" % (name, len(b["log"])))
        check("esdegerlik: etiket %s train.py = bugunku (Muon varsayilan; Model Z G 1, G'siz, transformer; 2 epok): son "
              "agirlik bit duzeyinde, adim adim kayip / gradyan normu / lr, sinav, okuma ayni" % EQUIV_TAG, not bad,
              "; ".join(bad) if bad else ", ".join(info))
    except Exception:  # noqa: BLE001
        check("esdegerlik (etiket %s)" % EQUIV_TAG, False, traceback.format_exc(limit=3))


def _unpin(argv):
    """argv'den --optimizer ciftini cikarir (varsayilan sinamasi)."""
    i = argv.index("--optimizer")
    return argv[:i] + argv[i + 2:]


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


def _fake_fineweb(tok_path, n_docs=150, long_every=40):
    """Kucuk FineWeb benzeri kaynak (belge 48): gpt2/shard_000.{bin,json}, _offsets, _bytes, tokenizer.json ve
    raw/000_00000.parquet (row group 25).  Belgeler: kisaltma, kapanis tirnagi, liste; her long_every'de bir > 2.048 token."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    rng = np.random.default_rng(4)
    names, topics = ["Paris", "Berlin", "Rome", "Madrid"], ["France", "Germany", "Italy", "Spain"]
    texts = []
    for i in range(n_docs):
        k = i % 4
        base = ("The capital of %s is %s. It is a large city in the U.S. sense of the word, said Dr. Smith.\n"
                "“It is old.” The people there like music.\n- one item\n- two items\n"
                "World War II began in 1939. Water boils at 100 degrees." % (topics[k], names[k]))
        extra = " ".join("Sentence number %d is here." % j for j in range(int(rng.integers(2, 12))))
        text = "Document %d. " % i + base + " " + extra
        if i % long_every == 7:
            text += " " + " ".join("Long part %d continues the long document." % j for j in range(330))
        texts.append(text)
    root = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(root, "gpt2"))
    os.makedirs(os.path.join(root, "raw"))
    ids = [[D.EOS_ID] + tok.encode(t).ids for t in texts]
    flat = np.array([t for d in ids for t in d], np.uint16)
    flat.tofile(os.path.join(root, "gpt2", "shard_000.bin"))
    np.save(os.path.join(root, "gpt2", "shard_000_offsets.npy"), np.r_[0, np.cumsum([len(d) for d in ids])[:-1]].astype(np.int64))
    np.save(os.path.join(root, "gpt2", "shard_000_bytes.npy"), np.array([len(t.encode()) for t in texts], np.int32))
    sha = hashlib.sha256(open(os.path.join(root, "gpt2", "shard_000.bin"), "rb").read()).hexdigest()
    json.dump(dict(shard="shard_000", source="sample/10BT/000_00000.parquet", source_sha256="test", docs=n_docs,
                   tokens=int(len(flat)), eot=D.EOS_ID, doc_format="[eot] + text", sha256=sha),
              open(os.path.join(root, "gpt2", "shard_000.json"), "w"))
    shutil.copyfile(tok_path, os.path.join(root, "gpt2", "tokenizer.json"))
    pq.write_table(pa.table(dict(text=texts, dump=["CC-MAIN-test"] * n_docs)), os.path.join(root, "raw", "000_00000.parquet"),
                   row_group_size=25)
    return root, texts


def t_fineweb():
    """FineWeb-Edu hazirligi (belge 47 / 48): web profili (ss profili bit duzeyinde ayni), parca + continues (devam eden
    parcada EOS hedefi yok), make_fineweb prepare uctan uca (kaynak dokunulmaz, ayirma, sinav, sizinti, planlar), train.py
    FineWeb klasoruyle (Model Z + G), knowledge_exam count / run, generate open_last (iki model), d 768 / 10 / 12 sekli."""
    import traceback
    import recipe as R
    import train as TR
    import make_fineweb as MF
    tp = tokenizer_path()
    if tp is None:
        print("ATLA fineweb: GPT-2 tokenizer yok", flush=True)
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_LIMITS, MF.VALID_STRIDE, D.MAX_SENTENCE_TOKENS,
             TR.MODEL_Z_GLOBAL_RATIO, TR.MODEL_Z_SUMMARIES_LAST)
    try:
        fl_ss, q_ss = D.stream_tables(tok)
        fl_ss2, q_ss2 = D.stream_tables(tok, "ss")
        fl_web, q_web = D.stream_tables(tok, "web")
        txt = ['“Have you got good religion?” and others respond. He lives in the U.S. now. Visit edX. It works.',
               'food, then.” That is a good step.\n”\nNext line here. Keywords: ' + "; ".join("w%d" % i for i in range(90))]
        x = np.array([t for s in txt for t in tok.encode(s).ids + [D.EOS_ID]], np.int64)
        b_web, _, _, forced = D._boundaries(x, fl_web, q_web, tok, "web")
        web_txt = [tok.decode(x[s:t].tolist()) for s, t in b_web]
        check("web profili: ss tablolari ve ss sinirlari varsayilanla bit duzeyinde ayni; web'de U.S. kesilmez, edX. kesilir, "
              "cok baytli kapanis tirnagi cumlede kalir, yalniz kapanis satiri katilir, 128'den uzun liste bolunur, token'lar "
              "kayipsiz", np.array_equal(fl_ss, fl_ss2) and np.array_equal(q_ss, q_ss2) and any("U.S. now." in t for t in web_txt)
              and any(t.endswith("edX.") for t in web_txt)
              and web_txt[0].endswith("respond.") and any(t.endswith("then.”") for t in web_txt) and any(t.endswith("\n\u201d\n") for t in web_txt)
              and forced >= 1 and max(t - s for s, t in b_web) <= D.MAX_SENTENCE_TOKENS
              and int((b_web[:, 1] - b_web[:, 0]).sum()) == len(x) - len(txt),
              " | ".join(t[:30] for t in web_txt))
        cases = [   # (metin, beklenen cumleler) -- 1. ve 2. insan kontrolu (okuma/bolme/bolme_kontrol*.md)
            ('Body.\nSources and Further Reading\nCeobanu, A. and X. Escandell. 2010. Comparative Analyses. Available '
             'Online.\n', ['Body.\n', 'Sources and Further Reading\n', 'Ceobanu, A. and X. Escandell. 2010.',
                           ' Comparative Analyses. Available Online.\n']),
            ('- FB — Fullback. Optional.\n- QB — Quarterback. Mandatory.\n',
             ['- FB — Fullback. Optional.\n', '- QB — Quarterback. Mandatory.\n']),
            # v4: kaydirma belge duzeyinde (_wrapped_docs: >= 4 noktalamasiz satir, ortanca >= 45 karakter); kisa parcada
            # yalniz islev kelimesi kurali (from / as: 2. insan kontrolu onerisi; "by" DEGIL -- 3. kontrol "by\nVideo" bolunmeli)
            ('In 1892 the young painter was hired by the publishing house of\nHarper Brothers in New York, where he drew '
             'covers for their weekly\nmagazines and illustrated several popular novels of the period for\nreaders across '
             'the country. The San Francisco Art Association later\ninvited him to teach a course on drawing from life to '
             'its students.\nNew Heading\nThe text.\n1. First item\n2. Second item',
             ['In 1892 the young painter was hired by the publishing house of\nHarper Brothers in New York, where he drew '
              'covers for their weekly\nmagazines and illustrated several popular novels of the period for\nreaders across '
              'the country.', ' The San Francisco Art Association later\ninvited him to teach a course on drawing from life '
              'to its students.\n', 'New Heading\n', 'The text.\n', '1. First item\n', '2. Second item']),
            ('It was sent from\nNew York as\nPart of a series. Done.',
             ['It was sent from\nNew York as\nPart of a series.', ' Done.']),
            ('characteristics1\n, promote early identification2\n, and inform.',
             ['characteristics1\n, promote early identification2\n, and inform.']),
            ('See QC981.8.C5 B738 and Ra.One and GOV.UK now. Next one.', ['See QC981.8.C5 B738 and Ra.One and GOV.UK now.',
                                                                           ' Next one.']),
            ('Miranda v. Arizona held. Over Rs. 300 crore. Kasper et al. 2005 said. No. Shrek goes. See No. 5 (pp. 292-321).',
             ['Miranda v. Arizona held.', ' Over Rs. 300 crore.', ' Kasper et al. 2005 said.', ' No.', ' Shrek goes.',
              ' See No. 5 (pp. 292-321).']),
            ('love yourself.` (Mk: 12. 31) Adam, the first man.', ['love yourself.`', ' (Mk: 12. 31) Adam, the first man.']),
            ('the self-image.2 The premise. they have."1 This is it. used plastics.[4,5,22] It is used. Ages 3.5 here.',
             ['the self-image.2', ' The premise. they have."1', ' This is it. used plastics.[4,5,22]', ' It is used.',
              ' Ages 3.5 here.']),
            ('usually negative.”\nPublic versus private interests\nAt issue is it. He said “No.” Then left.',
             ['usually negative.”\n', 'Public versus private interests\n', 'At issue is it.', ' He said “No.”',
              ' Then left.']),
            # 3. insan kontrolu (bolme_kontrol_v3.md): ondalik + birim, 999. 999, baslik / madde, kisaltma kuyrugu, atif etiketi
            ('Triggered on March 29.156 UT and it was big. A torque of 1.0 N m was used. Call 999. 999 calls are free.',
             ['Triggered on March 29.156 UT and it was big.', ' A torque of 1.0 N m was used.', ' Call 999.',
              ' 999 calls are free.']),
            ('See his article.\nAlpha Omega Academy\nFor an example.\nLast Updated on May 2, 2022 by\nVideo games are fun.\n'
             '- Note that we use it instead of the\nTI-83 listed here.\n- Next item.',
             ['See his article.\n', 'Alpha Omega Academy\n', 'For an example.\n', 'Last Updated on May 2, 2022 by\n',
              'Video games are fun.\n', '- Note that we use it instead of the\nTI-83 listed here.\n', '- Next item.']),
            ('Settled (art. I). S. Mohd. Ali works at Upstate Med. Univ. 2000 now. He died (d. 1890) in Rome. Authors N. R. '
             'Crockett, M.-L. Dubernet wrote. The work (v. 1. Our climate) is long.',
             ['Settled (art. I).', ' S. Mohd. Ali works at Upstate Med. Univ. 2000 now.', ' He died (d. 1890) in Rome.',
              ' Authors N. R. Crockett, M.-L. Dubernet wrote.', ' The work (v. 1. Our climate) is long.']),
            ('Health is a state of mind.\n (Koppelman, 2004)\nNext paragraph here.',
             ['Health is a state of mind.\n (Koppelman, 2004)\n', 'Next paragraph here.'])]
        got = []
        for t, want in cases:
            xx = np.array(tok.encode(t).ids + [D.EOS_ID], np.int64)
            got.append([tok.decode(xx[s:e].tolist()) for s, e in D._boundaries(xx, fl_web, q_web, tok, "web")[0]] == want)
        check("web profili insan kontrolu kurallari (kaynakca / madde birlestirme, satir kaydirmasi, baslik, liste numarasi, "
              "bosluksuz nokta, kisaltma, No. yalniz rakamdan once, ters tirnak, dipnot / atif, kapanis tirnagi + satir sonu): "
              "%d / %d ornek" % (sum(got), len(got)), all(got), str(got))
        st = _Synthetic([[[10, 11, 12], [13]], [[20, 21]], [[30], [31, 32]]])
        st.continues = np.array([True, False, False])
        b1 = D.build_batch(st, [[0, 1], [2]], "model_z", row_len=16)
        st.continues = None
        b0 = D.build_batch(st, [[0, 1], [2]], "model_z", row_len=16)
        diff = (b1.target != b0.target)
        check("continues: devam eden parcanin yalniz son konumunda hedef -100 ve tur -1 (EOS yok); oteki her alan ayni",
              int(diff.sum()) == 1 and int(b1.target[diff][0]) == -100 and int(b0.target[diff][0]) == D.EOS_ID
              and int(b1.target_kind[diff][0]) == -1 and torch.equal(b1.tokens, b0.tokens) and torch.equal(b1.pos, b0.pos))
        src, texts = _fake_fineweb(tp)
        before = {f: (os.path.getmtime(os.path.join(src, f)), os.path.getsize(os.path.join(src, f)))
                  for f in ("gpt2/shard_000.bin", "gpt2/shard_000.json", "raw/000_00000.parquet")}
        out = os.path.join(TMP, "fw_out")
        MF.VALID_STRIDE = 5
        guard = _raises(AssertionError, MF._guard, src, os.path.join(src, "x"))
        MF.main(["prepare", "--src", src, "--out", out, "--local", os.path.join(TMP, "fw_local"), "--workers", "2"])
        after = {f: (os.path.getmtime(os.path.join(src, f)), os.path.getsize(os.path.join(src, f))) for f in before}
        sj = json.load(open(os.path.join(out, "source.json"), encoding="utf-8"))
        tr, va = D.TokenStories(out, out, "train"), D.TokenStories(out, out, "valid")
        tid, vid = np.load(os.path.join(out, "gpt2", "train_doc_ids.npy")), np.load(os.path.join(out, "gpt2", "valid_doc_ids.npy"))
        cont = tr.continues
        firsts = np.flatnonzero(~np.r_[False, cont[:-1]])                   # belge ilk parcasi
        rebuilt = []
        for k, a in enumerate(firsts.tolist()):
            b = firsts[k + 1] if k + 1 < len(firsts) else tr.n
            rebuilt.append(tok.decode(np.concatenate([np.concatenate(tr.sentences(i)) for i in range(a, b)]).tolist()))
        exam = np.load(os.path.join(out, "exam_stories.npy"))
        check("make_fineweb prepare: kaynak dosyalara dokunulmaz; cikti kaynak icinde DURUR; ayirma (valid indeks %% 5, train "
              "kalan, ayrik); train parcalari birlestirilince belge metinleri birebir; uzun belge parcalandi (devam eden "
              "parca var, hepsi satira sigar); sinav tam belge ve satira sigar; source.json", before == after and guard
              and set(vid.tolist()) == set(range(0, len(texts), 5)) and not set(tid) & set(vid)
              and len(tid) + len(vid) == len(texts) and rebuilt == [texts[i] for i in tid] and cont.any()
              and tr.lengths().max() <= D.ROW_LEN and len(exam) and (va.lengths()[exam] <= D.ROW_LEN).all()
              and sj["split"]["train_docs"] == len(tid) and sj["pieces"]["continuing"] == int(cont.sum())
              and va.n == len(np.load(os.path.join(out, "gpt2", "valid_bytes.npy"))), "parca %d / devam %d" % (
                  tr.n, int(cont.sum())))
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_LIMITS = 4, 1, dict(max_sentences=3, max_tokens=4)
        TR.MODEL_Z_GLOBAL_RATIO, TR.MODEL_Z_SUMMARIES_LAST = 0.6, 0            # eski varsayilan: L1 / L2 G1 (8 Ekim)
        base = ["--data", out, "--stream", out, "--device", "cpu", "--d", "16", "--layers", "2", "--heads", "2", "--lr",
                "1e-2", "--checkpoint_minutes", "0", "--optimizer", "muon"]          # 8 Ekim varsayilani normuon'dan once
        run = os.path.join(TMP, "fw_runs", "mzg")
        a = TR.main(base + ["--model", "model_z", "--epochs", "2", "--out", run])
        L = [w["loss"] for w in a["log"]]
        f = np.load(os.path.join(out, "train_pack_plan_e1.npz"))
        rows = [f["row_stories"][f["row_offsets"][r]:f["row_offsets"][r + 1]].tolist() for r in range(len(f["row_offsets"]) - 1)]
        bb = [D.build_batch(tr, rows[r:r + 4], "model_z", "cpu", D.ROW_LEN) for r in range(0, len(rows), 4)]
        n_eos = sum(int((b.target_kind == D.TargetKind.EOS).sum()) for b in bb)
        check("train.py FineWeb klasoruyle (Model Z + G varsayilan, Muon): 2 epok kosar, kayip duser; kimlikte G 1 ve muon; "
              "okuma istemleri veri klasorundeki reading_prompts.json'dan; plandaki EOS hedefi sayisi = belge sayisi "
              "(devam eden parcada yok)", np.mean(L[-3:]) < np.mean(L[:3]) - 0.5 and a["identity"]["global_layers"] == 1
              and a["identity"]["optimizer"] == "muon" and n_eos == len(tid) and json.load(open(os.path.join(
                  run, "samples.json"), encoding="utf-8"))["rows"][0]["story"] == json.load(open(os.path.join(
                      out, "reading_prompts.json")))["prompts"][0]["story"],
              "kayip %.3f -> %.3f, EOS hedefi %d / belge %d" % (np.mean(L[:3]), np.mean(L[-3:]), n_eos, len(tid)))
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
        import run_summary as RS
        rs = RS.summarize(run, out)
        kg, kl = RS.attention_keys(out)
        T_ = tr.lengths().astype(np.float64)
        check("run_summary: sinav / ms / token / bellek / Recompiling alanlari; MFU = token/sn x (6N + 12 d (G k_g + (L-G) "
              "k_l)) / tepe; k_global = sum T(T+1)/2 / sum T (elle)", rs["exam_loss"] == a["exam"]["loss"]
              and rs["recompiling"] == 0 and abs(kg - (T_ * (T_ + 1) / 2).sum() / T_.sum()) < 1e-6 and 0 < kl < kg
              and abs(rs["mfu"] - rs["tokens_per_sec"] * (6 * a["params"] + 12 * 16 * (kg + kl)) / RS.PEAK) < 1e-4
              and "peak_reserved_gb" in a["log"][0], str({k: rs[k] for k in ("mfu", "k_global", "k_local")}))
        _fineweb_knowledge(src, out, run, tok)
        _fineweb_extend(src, out, texts, tok, TR, MF)
        _open_last(tok)
        _d768()
    except Exception:  # noqa: BLE001
        check("fineweb", False, traceback.format_exc(limit=4))
    finally:
        (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_LIMITS, MF.VALID_STRIDE, D.MAX_SENTENCE_TOKENS, TR.MODEL_Z_GLOBAL_RATIO,
         TR.MODEL_Z_SUMMARIES_LAST) = saved


def _fineweb_extend(src, out, texts, tok, TR, MF):
    """make_fineweb extend / merge (belge 75): 2 ek sentetik shard; birlesik klasor = birlesik akista bastan
    build_boundaries + _pieces (bit); --base sinav belgesinin birebir ve ilk 64 token kopyasi duser; valid / sinav / okuma
    --base ile bayt ayni; train.py birlesikle shard karisik batch'te ve epok sonu eksik batch oncesinde kesilip surdurulen =
    kesintisiz (bit); _local_copy yer yetmezse DURUR; prepare (s000) HEAD koduyla bit ayni."""
    import importlib.util
    import subprocess
    import traceback
    try:
        vid = np.load(os.path.join(out, "gpt2", "valid_doc_ids.npy"))
        exam = np.load(os.path.join(out, "exam_stories.npy"))
        full_copy, pre_copy = texts[vid[exam[0]]], texts[vid[exam[1]]] + " An extra tail sentence is here."
        assert len(tok.encode(texts[vid[exam[1]]]).ids) > 64, "on ek sinamasi icin sinav belgesi kisa"
        for shard, n0, extra in ((1, 1000, [full_copy, pre_copy]), (2, 2000, [])):
            docs = ["Document %d. The capital of %s is near. It has many people, said Dr. Lee.\n- one item\n%s" % (
                n0 + i, ["France", "Italy"][i % 2], " ".join("Line %d is fine." % j for j in range(3 + i % 5)))
                for i in range(30)]
            docs[7] += " " + " ".join("Long part %d continues the long document." % j for j in range(330))
            docs = docs[:10] + extra + docs[10:]
            ids = [[D.EOS_ID] + tok.encode(t).ids for t in docs]
            base = os.path.join(src, "gpt2", "shard_%03d" % shard)
            flat = np.array([t for d in ids for t in d], np.uint16)
            flat.tofile(base + ".bin")
            np.save(base + "_offsets.npy", np.r_[0, np.cumsum([len(d) for d in ids])[:-1]].astype(np.int64))
            np.save(base + "_bytes.npy", np.array([len(t.encode()) for t in docs], np.int32))
            json.dump(dict(shard="shard_%03d" % shard, source="sample/10BT/%03d_00000.parquet" % shard,
                           source_sha256="test", docs=len(docs), tokens=int(len(flat)), eot=D.EOS_ID,
                           doc_format="[eot] + text", sha256=hashlib.sha256(flat.tobytes()).hexdigest()),
                      open(base + ".json", "w"))
        p1, p2, mo = (os.path.join(TMP, "fw_parts", "s001"), os.path.join(TMP, "fw_parts", "s002"),
                      os.path.join(TMP, "fw_10bt"))
        MF.main(["extend", "--src", src, "--base", out, "--shard", "1", "--out", p1, "--local",
                 os.path.join(TMP, "fw_local_x"), "--workers", "2"])
        MF.main(["extend", "--src", src, "--base", out, "--shard", "2", "--out", p2, "--workers", "1"])
        MF.main(["merge", "--base", out, "--parts", p1, p2, "--out", mo])
        ref = os.path.join(TMP, "fw_ref")                                     # basvuru: birlesik akista bastan
        os.makedirs(os.path.join(ref, "gpt2"))
        np.save(os.path.join(ref, "gpt2", "train.npy"), np.concatenate(
            [np.load(os.path.join(d, "gpt2", "train.npy")) for d in (out, p1, p2)]))
        shutil.copyfile(os.path.join(out, "gpt2", "tokenizer.json"), os.path.join(ref, "gpt2", "tokenizer.json"))
        D.build_boundaries(ref, ref, "train", profile="web", workers=1)
        story, cont, _ = MF._pieces(D.TokenStories(ref, ref, "train"), D.ROW_LEN, tok)
        ld = lambda d, f: np.load(os.path.join(d, f))  # noqa: E731
        mt = D.TokenStories(mo, mo, "train")
        f = ld(mo, "train_pack_plan_e1.npz")
        ro, rs = D.pack_plan(mt.lengths(), D.ROW_LEN, 0, 1)
        bj = json.load(open(os.path.join(mo, "train_boundaries.json"), encoding="utf-8"))
        same_bytes = lambda a, b: open(a, "rb").read() == open(b, "rb").read()  # noqa: E731
        copied = ("gpt2/valid.npy", "gpt2/valid_bytes.npy", "gpt2/tokenizer.json", "valid_sentence_offsets.npy",
                  "valid_story_offsets.npy", "valid_boundaries.json", "exam_stories.npy", "exam_stories.json",
                  "exam_pack_plan.npz", "reading_prompts.json")
        check("make_fineweb merge: birlesik akis = s000 + s001 + s002 train (bayt); cumle / hikaye ofsetleri, continues = "
              "birlesik akista bastan build_boundaries + _pieces (bit); plan = pack_plan(birlesik, tohum 0); sayim = "
              "token_counts(birlesik); boundaries stream_sha256 = dosya; valid / sinav / okuma --base ile bayt ayni",
              np.array_equal(ld(mo, "gpt2/train.npy"), ld(ref, "gpt2/train.npy"))
              and np.array_equal(ld(mo, "train_sentence_offsets.npy"), ld(ref, "train_sentence_offsets.npy"))
              and np.array_equal(ld(mo, "train_story_offsets.npy"), story)
              and np.array_equal(ld(mo, "train_story_continues.npy"), cont)
              and np.array_equal(f["row_offsets"], ro) and np.array_equal(f["row_stories"], rs)
              and np.array_equal(ld(mo, "train_token_counts.npy"), D.token_counts(ref))
              and bj["stream_sha256"] == hashlib.sha256(open(os.path.join(mo, "gpt2", "train.npy"), "rb").read()).hexdigest()
              and bj["max_sentence_tokens_all"] == D.TokenStories(out, out, "train").max_sentence_tokens
              and all(same_bytes(os.path.join(out, c), os.path.join(mo, c)) for c in copied),
              "%d parca, %d satir" % (mt.n, len(ro) - 1))
        s1, s2, sm = (json.load(open(os.path.join(d, "source.json"), encoding="utf-8")) for d in (p1, p2, mo))
        t1 = ld(p1, "gpt2/train_doc_ids.npy")
        shards = ld(mo, "gpt2/train_doc_shards.npy")
        check("make_fineweb extend: --base sinav belgesinin birebir (1) ve ilk 64 token (1) kopyasi duser, sayilar "
              "source.json'da (birlesikte toplam); oteki shard'da 0; train_doc_shards belge sayisiyla; sinav cumlesi "
              "sizintisi orani yazili",
              (s1["split"]["dropped_exam_full"], s1["split"]["dropped_exam_prefix"]) == (1, 1)
              and not {10, 11} & set(t1.tolist()) and len(t1) == 30
              and (s2["split"]["dropped_exam_full"], s2["split"]["dropped_exam_prefix"]) == (0, 0)
              and (sm["split"]["dropped_exam_full"], sm["split"]["dropped_exam_prefix"]) == (1, 1)
              and np.bincount(shards).tolist() == [sm["split"]["train_docs"] - 60, 30, 30]
              and 0 <= sm["exam_sentence_leak_rate"] <= 1 and "exam_sentence_leak_rate" in s1,
              str(s1["split"]))
        piece_doc = np.cumsum(np.r_[True, ~mt.continues[:-1]]) - 1           # parca -> belge -> shard
        per_e1 = -(-(len(ro) - 1) // TR.BATCH_ROWS)
        mixed = [s_ for s_ in range(1, per_e1) if len({int(shards[piece_doc[k]]) for r in range(
            s_ * TR.BATCH_ROWS, min((s_ + 1) * TR.BATCH_ROWS, len(ro) - 1)) for k in rs[ro[r]:ro[r + 1]]}) > 1]
        cut1, cut2 = mixed[0], per_e1 - 1                                     # karisik batch; epok sonu eksik batch onu
        base = ["--data", mo, "--stream", mo, "--device", "cpu", "--d", "16", "--layers", "2", "--heads", "2", "--lr",
                "1e-2", "--checkpoint_minutes", "0", "--model", "model_z", "--epochs", "2"]
        a = TR.main(base + ["--out", os.path.join(TMP, "fw_10bt_A")])
        B = os.path.join(TMP, "fw_10bt_B")
        TR.main(base + ["--stop_step", str(cut1), "--out", B])
        if cut2 > cut1:
            TR.main(base + ["--stop_step", str(cut2), "--out", B, "--resume", "1"])
        b = TR.main(base + ["--out", B, "--resume", "1"])
        sa, sb = (torch.load(os.path.join(d, "agent.pt"), weights_only=False)["state"] for d in
                  (os.path.join(TMP, "fw_10bt_A"), B))
        check("train.py birlesik klasorle: kosar; adim %d (batch'te birden cok shard) ve %d'de (epok sonu eksik batch'ten "
              "once; son batch %d / %d satir) kesilip surdurulen = kesintisiz (kayip egrisi ve agirlik bit)" % (
                  cut1, cut2, (len(ro) - 1) % TR.BATCH_ROWS or TR.BATCH_ROWS, TR.BATCH_ROWS),
              cut2 > cut1 and b["finished"] and [w["loss"] for w in a["log"]] == [w["loss"] for w in b["log"]]
              and all(torch.equal(sa[k], sb[k]) for k in sa) and a["identity"]["train_stream_sha256"] == bj["stream_sha256"],
              "kesim %s" % [cut1, cut2])
        real = TR.shutil.disk_usage
        TR.shutil.disk_usage = lambda p: real(p)._replace(free=1000)
        try:
            msg = _exit_msg(TR._local_copy, mo, os.path.join(TMP, "fw_local_full"), mo)
        finally:
            TR.shutil.disk_usage = real
        check("train _local_copy: yer yetmezse kopyadan once DURUR (gereken / bos GB iletide)",
              msg is not None and "GB" in msg and not os.path.exists(os.path.join(TMP, "fw_local_full", "gpt2", "train.npy")),
              str(msg))
        hd = os.path.join(TMP, "mf_head")                                     # prepare: HEAD kodu = bugunku (bit)
        os.makedirs(hd)
        code = subprocess.run(["git", "-C", HERE, "show", "HEAD:deneme2/v2/common/make_fineweb.py"], capture_output=True,
                              check=True).stdout
        open(os.path.join(hd, "make_fineweb_head.py"), "wb").write(code)
        spec = importlib.util.spec_from_file_location("make_fineweb_head", os.path.join(hd, "make_fineweb_head.py"))
        MH = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MH)
        MH.VALID_STRIDE = MF.VALID_STRIDE
        oh = os.path.join(TMP, "fw_out_head")
        MH.main(["prepare", "--src", src, "--out", oh, "--workers", "2"])
        drop = lambda j: {k: v for k, v in j.items() if k not in ("created", "seconds", "stream")}  # noqa: E731
        diff = []
        for root_, _, files in os.walk(out):
            for fn in files:
                rel = os.path.relpath(os.path.join(root_, fn), out)
                a_, b_ = os.path.join(out, rel), os.path.join(oh, rel)
                if fn.endswith(".json"):
                    ok = drop(json.load(open(a_, encoding="utf-8"))) == drop(json.load(open(b_, encoding="utf-8")))
                else:
                    ok = same_bytes(a_, b_)
                if not ok:
                    diff.append(rel)
        check("make_fineweb prepare (s000 yolu) HEAD koduyla bit ayni (json'larda tarih / sure / yol haric)", not diff,
              str(diff))
    except Exception:  # noqa: BLE001
        check("fineweb extend / merge", False, traceback.format_exc(limit=4))



def _fineweb_knowledge(src, out, run, tok):
    """knowledge_exam: sayim (ayni belge / pencere, bant, istem birebir, arama tablosu) ve kosu (uretim + puan)."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diag"))
    import knowledge_exam as KE
    facts = dict(facts=[
        dict(id="cap_france", category="baskent", topic=["France"], key=["Paris"], distractors=["Lyon"],
             prompt="The capital of France is"),
        dict(id="ww2", category="tarih", topic=["World War II"], key=["1939"], distractors=["1914"],
             prompt="World War II began in"),
        dict(id="none", category="bilim", topic=["photosynthesis"], key=["sunlight"], distractors=[],
             prompt="Photosynthesis is the process by which plants")])
    fp = os.path.join(TMP, "fw_facts.json")
    json.dump(facts, open(fp, "w"))
    kd = os.path.join(TMP, "fw_knowledge")
    tid = np.load(os.path.join(out, "gpt2", "train_doc_ids.npy"))
    KE.main(["count", "--src", src, "--data", out, "--out", kd, "--facts", fp, "--workers", "2"])
    c = {x["id"]: x for x in json.load(open(os.path.join(kd, "counts.json")))["facts"]}
    lk = {x["id"]: x for x in json.load(open(os.path.join(kd, "lookup.json")))["facts"]}
    n_fr = int(sum(1 for i in tid if i % 4 == 0))                              # France belgeleri (sentetik: i % 4 == 0)
    check("knowledge_exam count: ayni belge / pencere sayisi = elle (France + Paris, WWII + 1939, yok = 0), bantlar, istem "
          "birebir belge sayisi, arama tablosu cevabi ve puani",
          c["cap_france"]["docs_same"] == n_fr == c["cap_france"]["docs_window"] and c["ww2"]["docs_same"] == len(tid)
          and c["none"]["docs_same"] == 0 and c["none"]["band"] == "0" and c["ww2"]["band"] == KE.band(len(tid))
          and c["cap_france"]["prompt_exact_docs"] == n_fr and lk["ww2"]["score"] == "DOGRU"
          and lk["cap_france"]["answer"].startswith("Paris"), str({k: (v["docs_same"], v["band"]) for k, v in c.items()}))
    check("knowledge_exam score: anahtar once -> DOGRU, celdirici once -> YANLIS, ikisi yok -> BOS, kelime siniri (19390 "
          "1939 sayilmaz)", KE.score("began in 1939, not 1914", facts["facts"][1])[0] == "DOGRU"
          and KE.score("began in 1914 and 1939", facts["facts"][1])[0] == "YANLIS"
          and KE.score("began long ago", facts["facts"][1])[0] == "BOS"
          and KE.score("code 19390", facts["facts"][1])[0] == "BOS")
    r = KE.main(["run", "--run", run, "--data", out, "--knowledge", kd, "--device", "cpu"])
    check("knowledge_exam run: kayitli kosudan acgozlu uretim (open_last) ve puan; ozet bantlari; dosyalar yazildi",
          len(r["rows"]) == 3 and all(x["score"] in ("DOGRU", "YANLIS", "BOS") for x in r["rows"]) and "hepsi" in r["summary"]
          and os.path.exists(os.path.join(run, "knowledge_exam.txt")), str(r["summary"]["hepsi"]))


def _open_last(tok):
    """generate(open_last=True): son istem cumlesi acik; ilk uretilen cumle = onbellekli / tam ileri gecisli basvuru
    acgozlu devami (iki model); open_last False varsayilan = bugunku cagri."""
    import importlib
    root = os.path.dirname(HERE)
    sys.path.insert(0, os.path.join(root, "model_z"))
    sys.path.insert(0, os.path.join(root, "transformer"))
    SM, BL = importlib.import_module("sentence"), importlib.import_module("baseline")
    s1, s2 = tok.encode("The cat sat on the mat.").ids, tok.encode(" The dog").ids
    ok = True
    for name, m in (("model_z", None), ("transformer", None)):
        torch.manual_seed(0)
        m = SM.SentenceTransformer(32, 2, 2, global_layers=1).eval() if name == "model_z" else BL.BaselineTransformer(32, 2, 2).eval()
        m.END = -1                                                          # cumle bitmesin: tam max_tokens devam
        g = m.generate([[s1, s2]], 1, 6, open_last=True)[0][0][0]
        with torch.no_grad():
            if name == "model_z":
                c = SM.SummaryCache(m)
                lg = c.prefill([s1])
                for t in s2:
                    lg = c.append_token(t)
                ref = []
                for _ in range(6):
                    w = int(lg.argmax())
                    ref.append(w)
                    lg = c.append_token(w)
            else:
                seq, ref = [D.EOS_ID] + s1 + [D.END_ID] + s2, []
                for _ in range(6):
                    T = len(seq)
                    h = m.hidden(torch.tensor([seq]), torch.arange(T)[None], None)
                    w = int((h[0, -1] @ m.E.weight.T).argmax())
                    ref.append(w)
                    seq.append(w)
        dflt = m.generate([[s1, s2]], 2, 5) == m.generate([[s1, s2]], 2, 5, open_last=False)
        ok &= g == ref and dflt
    check("generate open_last (Model Z + G ve transformer): acik son cumlenin acgozlu devami = basvuru (onbellek / tam ileri "
          "gecis); open_last=False = varsayilan", ok)


def _d768():
    """d 768 / 12 head: 10 katman parametre sayisi (meta cihazda). Gercek kosu GPU'da (5p profil hucresi): CPU'da derleme
    + 2 adim ~6 dk suruyordu (7 Ekim)."""
    import importlib
    for sub in ("model_z", "transformer"):
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), sub))
    SM, BL = importlib.import_module("sentence"), importlib.import_module("baseline")
    with torch.device("meta"):
        nz = sum(p.numel() for p in SM.SentenceTransformer(768, 10, 12, global_layers=1).parameters())
        nt = sum(p.numel() for p in BL.BaselineTransformer(768, 10, 12).parameters())
    check("d 768 / 10 katman / 12 head: Model Z + G ve transformer ayni parametre sayisi (~109M)",
          nz == nt and 105e6 < nz < 112e6, "%.1fM" % (nz / 1e6))


def t_fp8():
    """--fp8 (8 Ekim): torchao varsa MLP Float8Linear donusumu state_dict adlarini / sekillerini ve Muon ayrimini
    degistirmez (CUDA yoksa yalniz bu); torchao yoksa ATLANDI."""
    import train as TR
    try:
        import torchao.float8  # noqa: F401
    except ImportError:
        print("ATLANDI fp8: torchao yok (GPU sinamasi ana oturumda)", flush=True)
        return
    import argparse
    for model in ("model_z", "transformer"):
        args = argparse.Namespace(model=model, d=64, layers=2, heads=2, seed=0, global_layers=1 if model == "model_z" else 0,
                                  bag_k=0)
        m = TR._build(args, torch.device("cpu"))[0]
        names = [n for n, _ in __import__("recipe").muon_params(m)]
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        n = TR._fp8(m, "tensorwise")
        m.load_state_dict(sd)
        check("fp8 %s: %d Linear donustu, state_dict adlari / sekilleri ayni, bf16 state_dict yuklenir, Muon ayrimi ayni"
              % (model, n), n == 2 * 2 and [n_ for n_, _ in __import__("recipe").muon_params(m)] == names)


TESTS = dict(fp8=t_fp8, data=t_data, pack=t_pack, recipe=t_recipe, metrics=t_metrics, integration=t_integration,
             train=t_train, drive=t_drive, tokens=t_tokens, fineweb=t_fineweb)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
