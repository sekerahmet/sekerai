"""tests_v2 -- V2 common/ testleri (CPU).  Gruplar: data (hizli; GPT-2 tokenizer'i yerel HF onbelleginden ya da Drive'dan,
yoksa tokenizer'li sinamalar ATLANIR), pack (hizli, sentetik), drive (valid akisi: V1 ile birebir esleme, okuma istemleri).

    python tests_v2.py [--only data,pack,drive]
"""
import torch

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
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
    """Sentetik hikayeler: .sentences(i)."""

    def __init__(self, stories):
        self.s = [[np.array(x, dtype=np.int64) for x in st] for st in stories]
        self.n = len(stories)

    def sentences(self, i):
        return self.s[i]


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


TESTS = dict(data=t_data, pack=t_pack, drive=t_drive)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
