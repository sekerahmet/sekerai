"""test_z -- Model Z cumle yolunun test takimi (CPU).  Sentetik meaning / grammar ajanlari gecici klasore yazilir; ulke
verisi depodan (data/countries).  Drive gereken sinamalar (SS sinav kumesi, bayt dosyasi) Drive yoksa ATLANIR.

    python test_z.py [--only roundtrip,examples,...]
"""
import torch

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("core/sentence", "core/grammar", "core/meaning", ""):
    sys.path.insert(0, os.path.join(HERE, p))
from train_grammar import _no_power_throttling  # noqa: E402

if os.name == "nt":                     # EcoQoS: yoksa ~10 adimdan sonra ~10 kat yavas (kullanici, 6 Ekim)
    print("guc kisitlamasi (EcoQoS) kapali:", _no_power_throttling(), flush=True)
torch.set_num_threads(4)
import sentence_z as SZ  # noqa: E402
import train_sentence as TS  # noqa: E402
from grammar import GrammarAgent  # noqa: E402
from meaning import MeaningAgent  # noqa: E402
from sentence import PAD, SentenceTransformer  # noqa: E402

DRIVE = next((d for d in ("G:/Drive'ım", "/content/drive/MyDrive") if os.path.isdir(d)), None)
TMP = tempfile.mkdtemp(prefix="test_z_")
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


def agents(vocab, d=16, heads=4):
    torch.manual_seed(0)
    mp, gp = os.path.join(TMP, "meaning.pt"), os.path.join(TMP, "grammar_h%d.pt" % heads)
    torch.save(dict(vocab=vocab, state=MeaningAgent(vocab, d).state_dict(), args={}), mp)
    torch.save(dict(vocab=vocab, state=GrammarAgent(vocab, d, heads=heads).state_dict(), args=dict(d=d, heads=heads)), gp)
    return mp, gp


def tiny_model(V=30, z=8):
    torch.manual_seed(0)
    return SentenceTransformer(["<unk>"] + ["w%d" % i for i in range(V - 1)], z, d=16, layers=1, heads=2).eval()


def t_roundtrip():
    vocab = ["<unk>"] + ["w%d" % i for i in range(199)]
    mp, gp = agents(vocab)
    on, off = SZ.build_keys(mp, gp, z=512, longest=12), SZ.build_keys(mp, gp, z=512, longest=12, roles=False)
    g = torch.Generator().manual_seed(0)
    sents = [torch.randint(1, 200, (n,), generator=g).tolist() for n in range(1, 9) for _ in range(15)]
    x, m = TS.pad_sentences(sents)
    check("rol acik / kapali: konum anahtarlari ayni (A/B yalniz rol terimi)",
          all(torch.equal(on[k], off[k]) for k in ("F", "signs", "shift", "P")) and on["roles"] and not off["roles"])
    z_off = SZ.encode_z(off, x, m)
    back = SZ.decode_z(off, z_off)
    hit = sum(b == s for b, s in zip(back, sents))
    check("z geri acma (rol kapali, 1-8 kelime, V 200, z 512) birebir", hit == len(sents), "%d / %d" % (hit, len(sents)))
    check("decode_z(max_len) = tam Lk (rol kapali; gurultude ikisi farkli aday sirasi izleyebilir)",
          SZ.decode_z(off, z_off, max_len=8) == back)
    z = SZ.encode_z(on, x, m)
    ref, old = SZ.decode_z(on, z, max_len=8), SZ.DECODE_BYTES
    SZ.DECODE_BYTES = 9 * len(on["F"]) * 4 * 7                       # parca basina 7 satir
    check("decode_z parcalara bolunce ayni", SZ.decode_z(on, z, max_len=8) == ref)
    SZ.DECODE_BYTES = old
    xp = torch.cat([x, torch.zeros(len(x), 3, dtype=torch.long)], 1)          # genislik 11 < longest 12
    mpad = torch.cat([m, torch.zeros(len(x), 3, dtype=torch.bool)], 1)
    check("encode_z dolgudan ve batch bilesiminden bagimsiz", torch.equal(SZ.encode_z(on, xp, mpad), z)
          and torch.equal(SZ.encode_z(on, x[7:9], m[7:9]), z[7:9]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z16 = SZ.encode_z(on, x, m)
        back16 = SZ.decode_z(on, z16, max_len=8)
    check("autocast icinde z ve geri acma fp32 ile ayni (egitim / sinav ayni z)",
          torch.equal(z16, z) and back16 == SZ.decode_z(on, z, max_len=8), "en buyuk fark %.2e" % (z16 - z).abs().max())
    e0 = TS.pad_sentences([[], []])
    e2 = torch.zeros(1, 2, dtype=torch.long)
    check("bos cumle (L=0 ve dolgu) z = R_0 f(END), geri acma []", e0[0].shape == (2, 0)
          and SZ.decode_z(on, SZ.encode_z(on, *e0)) == [[], []]
          and SZ.decode_z(on, SZ.encode_z(on, e2, e2.bool())) == [[]])
    k2 = SZ.build_keys(*agents(vocab, heads=2), z=512, longest=12)
    check("grammar heads agent.pt args'tan", k2["grammar"].reader.layers[0].self_attn.num_heads == 2)


def t_examples():
    stories = [[[3, 4, 5], [6, 7], [8, 9, 10, 11]], [[12], [13, 14, 15, 16, 17]], [[18, 19]]]
    st = TS.Stories.from_lists(stories, "cpu")
    ids, mask = st.batch(torch.arange(3))
    zg = torch.randn(3, ids.shape[1], 8)
    END, EOS = 30, 31
    ex = TS.examples(ids, mask, zg, END, EOS)
    ok, row = ids.dtype == torch.long, 0
    for b, sents in enumerate(stories):
        for k in range(len(sents) + 1):
            w = sents[k] if k < len(sents) else []
            tgt = [-100] * k + ([w[0]] if w else [EOS]) + w[1:] + ([END] if w else [])
            T = len(tgt)
            ok &= ex["tok"][row, :T].tolist() == [0] * (1 + k) + w and ex["target"][row, :T].tolist() == tgt
            ok &= bool((ex["kind"][row, T:] == PAD).all()) and all(torch.equal(ex["zvec"][row, 1 + j], zg[b, j])
                                                                    for j in range(k))
            row += 1
    check("examples = kaba kuvvet (hedef kaydirma, END, EOS, z yerleri)", ok and row == len(ex["tok"]))
    check("story_z zero: z yok", not TS.story_z(dict(z=8), ids, mask, "zero").any())


def t_generate():
    model = tiny_model()
    out = model.generate(torch.randn(3, 0, 8), 5) + model.generate(torch.randn(3, 2, 8), 5)
    check("generate: hikaye basi (k=0) ve k=2; EOS yalniz [EOS] olarak",
          all(o == [model.EOS] or (model.EOS not in o and len(o) <= 6) for o in out))
    vocab = model.vocab
    keys = SZ.build_keys(*agents(vocab), z=32, longest=6)
    model = SentenceTransformer(vocab, 32, d=16, layers=1, heads=2).eval()
    prompts = [[[3, 4, 5]], [[6, 7]], [[8, 9], [10]]]
    ok = True
    for gen in (None, torch.Generator().manual_seed(0)):
        stories, res = TS.story_generation(model, keys, prompts, max_sentences=4, max_words=20, generator=gen)
        ok &= len(stories) == 3 and all(len(s) <= 4 and all(len(x) <= 6 for x in s) for s in stories) and all(
            0 <= res[k] <= 1 for k in ("eos_rate", "sentence_repeat", "loop", "no_end", "empty")) and (
            res["z_ok"] is None or 0 <= res["z_ok"] <= 1)
    s1 = TS.story_generation(model, keys, prompts, 4, 20, torch.Generator().manual_seed(0))[0]
    s2 = TS.story_generation(model, keys, prompts, 4, 20, torch.Generator().manual_seed(0))[0]
    s3 = TS.story_generation(model, keys, prompts, 4, 20)[0]
    check("story_generation greedy / sample: en cok 4 cumle, cumle <= longest, olculer [0, 1]; ayni tohum ayni metin, "
          "ornekleme acgozluden farkli", ok and s1 == s2 and s1 != s3, str(res))
    other = dict(keys, F=torch.randn_like(keys["F"]))                # z degisse de zero modda metin ayni olmali
    check("story_generation mode zero: model z gormuyor (z anahtari degisince metin ayni), true modda goruyor",
          TS.story_generation(model, keys, prompts, 3, 6, mode="zero")[0]
          == TS.story_generation(model, other, prompts, 3, 6, mode="zero")[0]
          and TS.story_generation(model, keys, prompts, 3, 6)[0] != TS.story_generation(model, other, prompts, 3, 6)[0])
    orig = model.generate
    model.generate = lambda zs, m, g=None: [[model.EOS] if i == 0 else [3, 4] for i in range(len(zs))]
    stories, res = TS.story_generation(model, keys, prompts, max_sentences=3, max_words=20)
    model.generate = orig
    check("story_generation: EOS'ta duruyor, otekiler max_sentences'e kadar, tekrar sayiliyor",
          stories == [[], [[3, 4]] * 3, []] and res["eos_rate"] == round(2 / 3, 4)
          and res["sentence_repeat"] == round(2 / 3, 4), str(res))


def t_exam():
    vocab, stories = TS.load_countries()
    model = SentenceTransformer(vocab, 512, d=16, layers=1, heads=2).eval()
    torch.nn.init.zeros_(model.E.weight)
    keys = SZ.build_keys(*agents(vocab), z=512, longest=20)
    exam = [s for s in stories if s["split"] == "exam"][:20]
    st = TS.Stories.from_lists([s["sents"] for s in exam], "cpu")
    e = TS.exam_scores(model, st, torch.arange(10), keys, 4, "true", nbytes=1000)
    check("exam_scores: tekduze model kaybi ln(V+2), acc_word var", abs(e["loss"] - round(math.log(len(vocab) + 2), 4))
          < 1e-4 and 0 <= e["acc_word"] <= 1, "%s" % e)
    torch.manual_seed(0)
    model = SentenceTransformer(vocab, 512, d=16, layers=1, heads=2).eval()
    g = TS.country_generation(model, exam, keys, "cpu", 0)
    check("country_generation: repeat_fact >= repeat (birebir tekrar ayni olgudur), z_ok payi [0, 1]",
          g["repeat_fact"] >= g["repeat"] and 0 <= g.get("z_ok", 0) <= 1, str(g))


def t_resume():
    vocab, _ = TS.load_countries()
    mp, gp = agents(vocab)
    TS.CHECKPOINT_SECS, TS.SHOW = -1, 0
    base = ["--data", "countries", "--meaning", mp, "--grammar", gp, "--d", "16", "--layers", "1", "--heads", "2",
            "--batch", "256", "--epochs", "2", "--device", "cpu"]
    a_dir, b_dir = os.path.join(TMP, "A"), os.path.join(TMP, "B")
    TS.main(base + ["--out", a_dir])
    orig, calls = TS.Stories.batch, [0]

    def failing(self, rows):
        calls[0] += 1
        if calls[0] == 17 and self.n > 1000:                              # 2. epok, 2. batch
            raise RuntimeError("kesinti")
        return orig(self, rows)
    TS.Stories.batch = failing
    try:
        TS.main(base + ["--out", b_dir])
    except RuntimeError:
        pass
    TS.Stories.batch = orig
    TS.main(base + ["--out", b_dir, "--resume", "1"])
    sa = torch.load(os.path.join(a_dir, "agent.pt"), weights_only=False)["state"]
    sb = torch.load(os.path.join(b_dir, "agent.pt"), weights_only=False)["state"]
    check("surdurme: kesilip surdurulen = kesintisiz (agirlik, bit duzeyinde)", all(torch.equal(sa[k], sb[k]) for k in sa))
    ra = json.load(open(os.path.join(a_dir, "results.json")))
    rb_text = open(os.path.join(b_dir, "results.json")).read()
    rb = json.loads(rb_text)
    same = [x["exam"] for x in ra["history"]] == [x["exam"] for x in rb["history"]]
    check("surdurme: results.json butun epoklari tasiyor, sinav sayilari ayni", len(rb["history"]) == 2 and same,
          "A %d, B %d epok" % (len(ra["history"]), len(rb["history"])))
    out = TS.main(base + ["--out", b_dir, "--resume", "1"])
    check("bitmis kosuyu --resume: durur, results.json degismez", out is None
          and open(os.path.join(b_dir, "results.json")).read() == rb_text)
    TS.main(base + ["--out", os.path.join(TMP, "C"), "--epochs", "1", "--train_z", "zero", "--z_roles", "0"])
    rc = json.load(open(os.path.join(TMP, "C", "results.json")))
    h = rc["history"][0]
    check("--train_z zero --z_roles 0: kosu bitiyor, args'ta; ana sinav zero modunda (exam = exam_zero_z, exam_z zero)",
          rc["args"]["train_z"] == "zero" and rc["args"]["z_roles"] == 0 and len(rc["history"]) == 1
          and h["exam_z"] == "zero" and h["exam"]["loss"] == h["exam_zero_z"]["loss"], "%s / %s" % (
              h["exam"]["loss"], h["exam_zero_z"]["loss"]))
    try:
        TS.main(base + ["--out", b_dir, "--resume", "1", "--z_roles", "0"])
        refused = False
    except AssertionError:
        refused = True
    check("surdurme kapisi: z_roles farkliysa durur", refused)


def t_drive():
    """SS sinav kumesi ve bayt dosyasi: Z sinav hikayesi i = valid hikayesi i; make_ss_sentences --stories'in bugunku kodu
    Drive'daki ss_exam_story_*.npy'yi ve bayt dosyasini birebir uretiyor (yeni cikti bellekte, Drive'a yazilmaz)."""
    if DRIVE is None:
        print("ATLA drive: Drive yok", flush=True)
        return
    import hashlib
    import make_ss_sentences as M
    ss, zr = DRIVE + "/simplestories", DRIVE + "/model_z/simplestories_full"
    ex = np.load(ss + "/exam_stories.npy")
    sha = hashlib.sha256(np.ascontiguousarray(ex)).hexdigest()
    check("sinav kumesi: exam_stories.npy sha256 = exam_stories.json", sha == json.load(open(ss + "/exam_stories.json"))
          ["sha256"], "%s, %d hikaye" % (sha[:12], len(ex)))
    a = np.load(ss + "/gpt2/valid.npy", mmap_mode="r")
    eos = 50256
    got = [M._story_chunk((ss, zr, "valid", s, t)) for s, t in M.chunks(a, eos, M.CHUNK)]
    ids, lengths, counts, nbytes = (np.concatenate([g[i] for g in got]) for i in range(4))
    zb = np.load(zr + "/ss_exam_story_bytes.npy")
    vb = np.load(ss + "/gpt2/valid_bytes.npy")
    check("make_ss_sentences --stories: yeni bayt dizisi = Drive'daki ss_exam_story_bytes.npy (deger deger)",
          np.array_equal(nbytes, zb), "%d hikaye" % len(nbytes))
    check("Z sinav hikayesi i = valid hikayesi i: bayt dizisi model_y valid_bytes.npy ile ayni, sayi ayni",
          np.array_equal(nbytes, vb.astype(np.int64)) and len(counts) == len(vb) and ex.max() < len(counts))
    check("make_ss_sentences --stories: kelime / cumle / hikaye dizileri Drive'dakiyle ayni",
          np.array_equal(ids, np.load(zr + "/ss_exam_story_ids.npy"))
          and np.array_equal(np.r_[0, np.cumsum(lengths)], np.load(zr + "/ss_exam_story_sentence_offsets.npy"))
          and np.array_equal(np.r_[0, np.cumsum(counts)], np.load(zr + "/ss_exam_story_offsets.npy")))


TESTS = dict(roundtrip=t_roundtrip, examples=t_examples, generate=t_generate, exam=t_exam, resume=t_resume,
             drive=t_drive)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
