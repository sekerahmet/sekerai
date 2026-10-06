"""tests_v2 -- V2 common/ testleri (CPU).  Gruplar: data (hizli; GPT-2 tokenizer'i yerel HF onbelleginden ya da Drive'dan,
yoksa tokenizer'li sinamalar ATLANIR), pack (sentetik), recipe (maske, WSD, gruplar, surdurme, hiz), metrics (sentence_repeat,
normalize_words, story_generation, exam_scores), integration (iki modelin loss_per_target'i
gercek build_batch ile; model dosyalari yalniz testte import edilir), drive (valid akisi: V1 ile birebir esleme, okuma istemleri).

    python tests_v2.py [--only data,pack,recipe,metrics,integration,drive]
"""
import torch

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
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
RESULTS = []


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
        path = os.path.join(TMP, "meaning_int.pt")
        torch.save(dict(state={"source.weight": torch.randn(D.END_ID, 256)}), path)
        torch.manual_seed(0)
        models.append(("model_z", SentenceTransformer(SZ.build_keys(path, 8), d=16, layers=1, heads=2).eval(),
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
        except Exception:  # noqa: BLE001
            check("entegrasyon %s: loss_per_target(build_batch)" % layout, False,
                  traceback.format_exc(limit=2).splitlines()[-1])


TESTS = dict(data=t_data, pack=t_pack, recipe=t_recipe, metrics=t_metrics, integration=t_integration,
             drive=t_drive)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
