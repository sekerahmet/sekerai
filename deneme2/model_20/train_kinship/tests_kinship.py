# -*- coding: utf-8 -*-
"""tests_kinship -- akrabalik egitiminin kapilari: veri (data_20), sinav (exam_kinship), yedek/surdurme ve Colab calistirici
(colab_kinship).  CPU, saniyeler.

    python tests_kinship.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # model_20: model ve genel egitim
torch.set_num_threads(1)

import data_20 as D  # noqa: E402
import exam_kinship as EK  # noqa: E402
import train_20 as TR  # noqa: E402

RESULTS = []
FINGERPRINT = "454d53e81c67"      # veri degisirse burasi bilerek guncellenir (27 Eylul: 32 aile; once 90024739fb8f)


def check(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("  %-78s %s  %s" % (name, "GECTI" if ok else "KALDI", note))


def graph_from_text(train):
    """Egitim METNINDEKI tek iliskili cumlelerden ('X's r is Y.') graf -- data_20'nin ic yapisindan bagimsiz."""
    g = {}
    for s in train:
        text = D.detokenize(s)
        if text.startswith("Who ") or text.count("'s") != 1:
            continue
        left, y = text[:-1].split(" is ")
        x, r = left.split("'s ")
        g.setdefault((x, r), set()).add(y)
    return g


def walk(g, x, path):
    now = {x}
    for r in path:
        now = {y for z in now for y in g.get((z, r), ())}
    return now


DEFINITIONS = {
    "grandfather": [("father", "father"), ("mother", "father")],
    "grandmother": [("father", "mother"), ("mother", "mother")],
    "aunt": [("father", "sister"), ("mother", "sister")],
    "uncle": [("father", "brother"), ("mother", "brother")],
    "cousin": [(p, s, c) for p in ("father", "mother") for s in ("sister", "brother") for c in ("son", "daughter")],
}


def t_data():
    d = D.build()
    check("veri: iz sabit (degisirse bilerek guncellenir)", d["fingerprint"] == FINGERPRINT, d["fingerprint"])
    check("veri: iki kurulus birebir ayni", D.build()["train"] == d["train"] and D.build()["exam"] == d["exam"])
    check("veri: audit gecer (tutulan zincir metinde yok, donen zincir yok, adlar benzersiz)", D.audit(d) == 480)

    people = d["people"]
    classes = {}
    for e in d["exam"]:
        classes[e["cls"]] = classes.get(e["cls"], 0) + 1
    ok = (len(people) == 192 and len(d["train"]) == 2560 and len(d["vocab"]) == 242
          and classes == dict(memory_base=640, memory_derived=480, chain2=192, chain3=128, named2=128, named3=32))
    check("veri: sayilar (192 kisi, 2560 cumle, sozluk 242, sinif sayilari)", ok, str(classes))

    texts = [D.detokenize(s) for s in d["train"]] + [D.detokenize(e["prompt"]) for e in d["exam"]]
    check("veri: tokenize(detokenize(x)) == x butun cumle ve sorularda",
          all(D.tokenize(D.detokenize(s)) == s for s in d["train"]) and
          all(D.tokenize(D.detokenize(e["prompt"])) == e["prompt"] for e in d["exam"]), texts[0])

    vocab = set(d["vocab"])
    ids = [D.encode(s, d["vocab"]) for s in d["train"]]
    eos = d["vocab"].index(D.EOS)
    check("veri: sozluk her token'i kapsar, encode <eos> ile cevirir",
          all(t in vocab for s in d["train"] for t in s) and all(t in vocab for e in d["exam"] for t in e["prompt"]) and
          all(a.split()[0] in vocab for e in d["exam"] for a in e["answers"]) and
          all(i[0] == eos == i[-1] and 0 <= min(i) and max(i) < len(vocab) for i in ids))

    # bagimsiz referans: cevaplar yalniz egitim METNINDEKI tek adimlardan ve buradaki tanimlardan
    g = graph_from_text(d["train"])
    wrong = []
    for e in d["exam"]:
        x, path = e["subject"], e["path"]
        if path[0] in DEFINITIONS and len(path) == 1:
            want = set().union(*(walk(g, x, p) for p in DEFINITIONS[path[0]]))
        else:
            want = walk(g, x, path)
        if want != set(e["answers"]):
            wrong.append((D.detokenize(e["prompt"]), sorted(want), e["answers"]))
    check("veri: her sorunun cevabi yalniz yazili tek adimlardan (metinden) ayni cikar", not wrong, str(wrong[:2]))

    named_written = {}
    for s in d["train"]:
        t = D.detokenize(s)
        if not t.startswith("Who ") and t.count("'s") == 1:
            left, y = t[:-1].split(" is ")
            x, r = left.split("'s ")
            if r in DEFINITIONS:
                named_written.setdefault((x, r), set()).add(y)
    ok = all(ys == set().union(*(walk(g, x, p) for p in DEFINITIONS[r])) for (x, r), ys in named_written.items())
    check("veri: yazili adli olgular (aunt, cousin ...) tanimin birlesimine esit", ok and len(named_written) == 160,
          "%d adli soru" % len(named_written))

    held = set(d["held"])
    households = {}
    for x, p in people.items():
        if p["household"]:
            households.setdefault(p["household"], set()).add(x)
    split_ok = all(h <= held or not h & held for h in households.values()) and len(held) == 32
    leak = [D.detokenize(s) for s in d["train"] if s[0] + " " + s[1] in held and ("'s" in s[3:] or s[3] in DEFINITIONS)]
    leak += [D.detokenize(s) for s in d["train"] if s[0] == "Who" and s[2] + " " + s[3] in held
             and ("'s" in s[5:s.index("?")] or s[5] in DEFINITIONS)]
    check("veri: ayirma hane hane; tutulan torunun turemis iliskisi HIC yazilmadi", split_ok and not leak, str(leak[:2]))

    reverse = []
    pairs = set()
    for s in d["train"]:
        names = [" ".join(s[i:i + 2]) for i in range(len(s) - 1) if " ".join(s[i:i + 2]) in people]
        pairs.update((a, b) for a in names for b in names)
    for e in d["exam"]:
        if not e["cls"].startswith("memory"):
            reverse.append(any((e["subject"], y) in pairs for y in e["answers"]) == e["co_written"])
    check("veri: co_written = cevap ile ozne ayni egitim cumlesinde gecti (metinden)", all(reverse))

    check("veri: HIC YAZILMAMIS siniflarda co_written yalniz kuzen yolunda (kenar haneler)",
          all(not e["co_written"] or e["path"][-1] in ("son", "daughter", "cousin")
              for e in d["exam"] if not e["cls"].startswith("memory")))

    # Adim 4: yalniz 1R + ara adimli 2R; her adim Adim 0-3 metninde yazili bir 1R cumlesi olmali
    s = D.build(step_answers=True)
    stmts = {D.detokenize(t) for t in d["train"] if t[0] != "Who"}
    bad = []
    trained = [t for t in s["train"] if t[0] == "<steps>"]
    walks = [(e["subject"], e["steps"]) for e in s["exam"] if e["cls"] in ("2R_T", "2R_UT")]
    walks += [(" ".join(t[3:5]), t[t.index("?") + 1:]) for t in trained]
    for who, rest in walks:
        for i in range(0, len(rest), 8):
            txt = D.detokenize(rest[i:i + 8])
            if txt not in stmts or not txt.startswith(who + "'s "):
                bad.append(txt)
            who = txt[:-1].split(" is ")[1]
    held = set(d["held"])
    text = "\n".join(D.detokenize(t) for t in s["train"])
    leak = [e for e in s["exam"] if e["cls"] == "2R_UT" and D.detokenize(e["steps"]) in text]
    short = [t for t in s["train"] if t[0] != "<steps>" and (t.count("'s") > 1 or any(r in t for r in DEFINITIONS))]
    want = {tuple(e["prompt"] + e["steps"]) for e in s["exam"] if e["cls"] == "2R_T"}
    counts = {}
    for e in s["exam"]:
        counts[e["cls"]] = counts.get(e["cls"], 0) + 1
    check("veri, STEP_ANSWERS: egitimde yalniz 1R + 2 adimli ara adimli 2R (kisa 2R, adli, 3R yok); adimlar yazili 1R "
          "cumleleri; 160 ozne, tutulan torun hic ozne degil, 2R_UT adimlari metinde yok; 2R_T egitimde; audit gecer",
          not bad and not leak and not short and want <= {tuple(t) for t in trained} and len(trained) == 960
          and all(t.index("?") == 9 for t in trained)
          and len({" ".join(t[3:5]) for t in trained}) == 160 and not {" ".join(t[3:5]) for t in trained} & held
          and len(s["train"]) == 2240 and D.audit(s) == 192
          and counts == {"1R_T": 640, "2R_T": 192, "2R_UT": 192}, str(bad[:2] or leak[:1] or short[:1] or counts))

    u = D.build(step_answers=True, step_marker=False)
    same_chains = sorted(tuple(t[1:]) for t in trained) == sorted(tuple(t) for t in u["train"] if t[0] == "Who" and t.index("?") == 8)
    same_exam = [(e["cls"], e["prompt"][1:] if e["prompt"][0] == "<steps>" else e["prompt"], e["answers"]) for e in s["exam"]] \
        == [(e["cls"], e["prompt"], e["answers"]) for e in u["exam"]]
    check("veri, STEP_MARKER=False: ayni zincirler ve sinav, yalniz '<steps>' yok (sozluk 237, iz 3898f9b3d551); audit gecer; "
          "varsayilan (True) iz degismedi",
          same_chains and same_exam and "<steps>" not in u["vocab"] and len(u["vocab"]) == 237 and u["fingerprint"] == "3898f9b3d551"
          and D.audit(u) == 192 and s["fingerprint"] == "7ae5615623aa" and len(u["train"]) == 2240, u["fingerprint"])

    L = D.build(step_answers=True, long_1r=True)
    one = [e for e in L["exam"] if e["cls"] == "1R_T"]
    long_qa = [t for t in L["train"] if t[0] == "Who" and t.index("?") == 6]
    textL = "\n".join(D.detokenize(t) for t in L["train"])
    owen = "Who is Owen Evans's father? Owen Evans's father is David Evans."
    check("veri, LONG_1R: 1R soru-cevabi tam cumle (640, tutulanlar dahil); zincirler ve 2R sinavi ayni; 1R_T'de beklenen "
          "cumle; 2R_UT adimlari metinde yok; audit gecer; kapaliyken iz degismedi",
          len(long_qa) == 640 and all(t[7:9] == t[2:4] and t[9:11] == ["'s", t[5]] for t in long_qa) and owen in textL
          and [t for t in L["train"] if t[0] == "<steps>"] == trained
          and [e for e in L["exam"] if e["cls"] != "1R_T"] == [e for e in s["exam"] if e["cls"] != "1R_T"]
          and all(e["steps"] == e["subject"].split() + ["'s", e["path"][0], "is"] + e["answers"][0].split() + ["."] for e in one)
          and not [e for e in L["exam"] if e["cls"] == "2R_UT" and D.detokenize(e["steps"]) in textL]
          and D.audit(L) == 192 and L["long_1r"] and not s["long_1r"] and s["fingerprint"] == "7ae5615623aa",
          L["fingerprint"])


def t_resume():
    import copy
    import tempfile
    import colab_kinship as C
    s = D.build(step_answers=True)
    ids, mask = EK.sequences(s)
    ids, mask = ids[:40], mask[:40]
    nv = len(s["vocab"])
    for kw in (dict(), dict(weight_decay=0.1, optimizer="adam")):
        packs, seen_full, seen_resumed = {}, [], []

        def keep(step, model, opt):
            packs[step] = dict(step=step, model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()))
        full, _ = TR.train_seq("shared", ids, mask, nv, steps=6, log_at=(), save_every=2, save=keep, every=2,
                               callback=lambda st, m, nll: seen_full.append(st), **kw)
        resumed, _ = TR.train_seq("shared", ids, mask, nv, steps=6, log_at=(), checkpoint=packs[4], every=2,
                                  callback=lambda st, m, nll: seen_resumed.append(st), **kw)
        same = all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))
        check("surdurme: 4. adim paketinden surdurulen = kesintisiz 6 adim, bit duzeyinde; paketler 2, 4, 6; sinav tekrarlanmaz %s"
              % (kw or ""), same and sorted(packs) == [2, 4, 6] and seen_full == [0, 2, 4, 6] and seen_resumed == [6],
              "%s %s %s" % (sorted(packs), seen_full, seen_resumed))

    out = tempfile.mkdtemp() + "/r"
    run = C.start("TEST_R", s, out, steps=3, every=1, device="cpu", save_every=1)
    run["thread"].join(600)
    files = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t"))
    os.remove(os.path.join(out, "checkpoint_t00003.pt"))            # 2. adimdan sonra kesilmis gibi
    lines_before = open(os.path.join(out, "log.txt"), encoding="utf-8").read().count("\n")
    run2 = C.start("TEST_R", s, out, steps=3, every=1, device="cpu", save_every=1, resume=True)
    run2["thread"].join(600)
    log = open(os.path.join(out, "log.txt"), encoding="utf-8").read()
    try:
        C.start("TEST_R2", s, out, steps=5, every=100, device="cpu", save_every=1, resume=True)
        refused = False
    except RuntimeError:
        refused = True
    check("surdurme, colab_kinship: her adimda checkpoint_tNNNNN.pt; resume=True son paketten devam eder, gunluge yazar, "
          "klasoru tasimaz; exams.json'da adim tekrarlanmaz; ayar farkliysa reddeder",
          files == ["checkpoint_t00001.pt", "checkpoint_t00002.pt", "checkpoint_t00003.pt"] and run2["done"]
          and not run2["error"] and "SURDURULDU adim 2'den" in log and log.count("\n") > lines_before
          and os.path.exists(os.path.join(out, "checkpoint_t00003.pt")) and refused
          and [e["step"] for e in json.load(open(os.path.join(out, "exams.json")))] == [0, 1, 2, 3], str(run2["error"] or files))


def t_colab():
    import tempfile
    import colab_kinship as C
    from model_20 import BlockModel
    s = D.build(step_answers=True)
    e = next(e for e in s["exam"] if e["cls"] == "2R_UT")
    good = EK.score_steps(list(e["steps"]) + [D.EOS], e)
    bad = list(e["steps"]) + [D.EOS]
    bad[-4] = "Tom"
    worse = EK.score_steps(bad, e)
    going = EK.score_steps(list(e["steps"]) + ["Who"], e)          # dogru cevap, ama durmadan devam ediyor
    check("adim 4: score_steps -- dogru cevap + <eos>'ta SC BC AC EX FC hepsi evet; son ad yanlissa yalniz AC ve EX hayir; "
          "cevap dogru ama <eos> yoksa yalniz EX hayir",
          all(good.values()) and sorted(good) == ["AC", "BC", "EX", "FC", "SC"]
          and not worse["AC"] and not worse["EX"] and worse["BC"] and worse["SC"] and worse["FC"]
          and not going["EX"] and going["AC"] and going["SC"] and going["BC"] and going["FC"])

    m = BlockModel(len(s["vocab"]), d=16, units=8)
    gr = torch.Generator().manual_seed(5)
    with torch.no_grad():                        # egitilmemis model birim donusum gibi: '?'tan '?' uretir, test bos gecerdi
        for p_ in m.parameters():
            if p_.requires_grad:
                p_.copy_(0.5 * torch.randn(p_.shape, generator=gr))
    prompts = [EK.questions(s, "2R_UT")[i][0] for i in (0, 1)] + [EK.questions(s, "1R_T")[0][0]]
    out = TR.generate(m, prompts, 5)
    manual = []
    with torch.no_grad():
        for p in prompts:
            ids = list(p)
            for _ in range(5):
                ids.append(int(m.logits(torch.tensor([ids]))[0, -1].argmax()))
            manual.append(ids[len(p):])
    check("adim 4: generate toplu uretim = tek tek acgozlu uretim (farkli uzunluklar birlikte, rastgele model)",
          out == manual and len({tuple(o) for o in out}) > 1, str(out))

    seen = []
    ids, mask = EK.sequences(s)
    TR.train_seq("shared", ids[:20], mask[:20], len(s["vocab"]), steps=4, log_at=(), every=2,
                 callback=lambda step, model, nll: seen.append(step))
    check("train_seq: callback her every adimda, guncellemeden once (0, 2, 4)", seen == [0, 2, 4], str(seen))

    out_dir = tempfile.mkdtemp()
    run = C.start("TEST", s, out_dir + "/r", steps=2, every=100, device="cpu")
    run["thread"].join(600)
    files = sorted(os.listdir(out_dir + "/r"))
    check("colab_kinship: CPU'da start -> sinav, model, son olcum dosyalari; pulse/stop hatasiz; config'de rope True (Model X)",
          run["done"] and not run["error"] and files == ["config.json", "exams.json", "final.json", "log.txt", "model.pt"]
          and json.load(open(out_dir + "/r/config.json"))["rope"] is True,
          str(run["error"] or files))
    C.pulse(1)
    C.stop()

    L = D.build(step_answers=True, long_1r=True)
    run = C.start("TEST_L", L, out_dir + "/l", steps=1, every=100, device="cpu")
    run["thread"].join(600)
    fin = json.load(open(out_dir + "/l/final.json", encoding="utf-8")) if os.path.exists(out_dir + "/l/final.json") else {}
    ex = json.load(open(out_dir + "/l/exams.json")) if os.path.exists(out_dir + "/l/exams.json") else [{}]
    check("colab_kinship: LONG_1R verisinde 1R_T tam cumleyle puanlanir (AC, EX), final.json'da 1R_T_given0",
          run["done"] and not run["error"] and "1R_T_given0" in fin and "1R_T_EX" in ex[0]
          and len(fin["1R_T_given0"]["rows"]) == 640, str(run["error"] or sorted(fin)))

    run = C.start("TEST_T", s, out_dir + "/t", steps=2, every=100, device="cpu", setting="transformer")
    run["thread"].join(600)
    cfg = json.load(open(out_dir + "/t/config.json"))
    check("colab_kinship: setting='transformer' ile start; config ayarlarin tamamini yazar (varsayilanlar dahil)",
          run["done"] and not run["error"] and cfg["setting"] == "transformer"
          and cfg["lr"] == TR.LR and cfg["grad_clip"] == TR.GRAD_CLIP and cfg["rope"] is True
          and "final.json" in os.listdir(out_dir + "/t"),
          str(run["error"] or cfg))


if __name__ == "__main__":
    print("tests (train_kinship)")
    for f in (t_data, t_resume, t_colab):
        f()
    print("\n%d GECTI   %d KALDI" % (sum(RESULTS), len(RESULTS) - sum(RESULTS)))
    sys.exit(0 if all(RESULTS) else 1)
