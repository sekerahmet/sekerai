"""tests_diag -- V2 teshis araclari testleri (CPU; belge 33 adim 5).  Gruplar: readings (generate_readings: kayitli
kosudan okuma = train.py'ninki; arsiv kimligi durur; ek istem secimi Drive'dan), tools (gap_v2, order_probe,
z_ablate; model-z-mathematician), knowledge, trace (token_trace, belge 80).  Formullu z cesitleri (belge 44) ve torba
(bag_report; belge 77) kaldirildi.  Yardimcilar common/tests_v2'den (_train_root, tokenizer_path, DRIVE).

    python tests_diag.py [--only readings,tools,knowledge,trace]
"""
import torch

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "common")
sys.path.insert(0, COMMON)
sys.path.insert(0, HERE)
import tests_v2 as T2  # noqa: E402  (EcoQoS kapatma, gecici klasor, veri kurucu)

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print("%-4s %s %s" % ("OK" if ok else "HATA", name, info), flush=True)


def t_readings():
    """generate_readings: kayitli kosudan (agent.pt) ayni istemler -> train.py'nin samples.json'u ile birebir ayni metin ve
    olcu (greedy + sample, iki model); kosu dosyalarina dokunmaz; istem dosyasi adi denetlenir; temizlik oncesi Model Z
    kimligi durur.  Ek istem dosyasi (Drive varsa) extra_prompts'tan yeniden hesaplanir."""
    import traceback
    import generate_readings as GR
    import train as TR
    tp = T2.tokenizer_path()
    if tp is None:
        print("ATLA readings: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = T2._train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, TR.MODEL_Z_GLOBAL_RATIO)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.MODEL_Z_GLOBAL_RATIO = 0.6                                       # eski varsayilan: L1 / L2 G1 (8 Ekim)
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=6, max_tokens=4)
    base = ["--data", data, "--stream", root, "--device", "cpu"]
    same_prompts = os.path.join(root, "reading_prompts_same.json")
    shutil.copyfile(prompts, same_prompts)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    variants = [("transformer", []), ("model_z_learned", ["--global_layers", "0"]),          # belge 35 (b), G'siz
                ("model_z_global", ["--layers", "2"])]                                       # varsayilan G 1 (belge 43)
    try:
        for name, extra in variants:                                       # name: model ya da model_z_<cesit>
            model = "transformer" if name == "transformer" else "model_z"
            run = os.path.join(T2.TMP, "runs_readings", name)
            TR.main(base + ["--model", model, "--d", "16", "--layers", "1", "--heads", "2", "--lr", "1e-2", "--steps", "6",
                            "--out", run, "--checkpoint_minutes", "0", "--optimizer", "adamw"] + extra)
            before = {n: (sha(os.path.join(run, n)), os.path.getmtime(os.path.join(run, n))) for n in TR.OUTPUTS}
            r = GR.main(base + ["--out", run, "--prompts", same_prompts])
            sj = json.load(open(os.path.join(run, "samples.json"), encoding="utf-8"))
            new = json.load(open(os.path.join(run, "samples_same.json"), encoding="utf-8"))
            txt = open(os.path.join(run, "samples_same.txt"), encoding="utf-8").read()
            after = {n: (sha(os.path.join(run, n)), os.path.getmtime(os.path.join(run, n))) for n in TR.OUTPUTS}
            labels = [x["label"] for x in sj["rows"]]
            check("readings %s: kayitli agent.pt'den ayni istemler = train.py samples.json (satirlar ve olculer birebir, "
                  "greedy + sample); samples_same.txt = samples.txt; reproduced hepsi ayni" % name,
                  new["rows"] == sj["rows"] and new["generation"] == sj["generation"] and r["rows"] == sj["rows"]
                  and txt == open(os.path.join(run, "samples.txt"), encoding="utf-8").read()
                  and new["reproduced"]["same"] == labels and new["reproduced"]["different"] == []
                  and new["reproduced"]["generation"] == sj["generation"]
                  and sum(len(x["story_text"].split()) for x in sj["rows"]) > 0,
                  " | ".join(x["story_text"].replace("\n", " / ")[:40] for x in new["rows"]))
            check("readings %s: kosu dosyalari (%s) degismedi" % (name, ", ".join(TR.OUTPUTS)), before == after)
            gk = {"story_loop", "story_loop_prompts", "sentence_repeat_by_prompt", "sentence_repeat", "word_loop"}
            check("readings %s: samples_<ek>.json olculeri (story_loop, sentence_repeat_by_prompt ...) ve kaynak" % name,
                  all(gk <= set(g) for g in new["generation"].values()) and new["source"]["prompts_sha256"] == sha(
                      same_prompts) and new["source"]["identity"]["model"] == model)
            sj["rows"][1]["story_text"] += "\nX"                          # bozuk samples.json: fark yakalanir
            json.dump(sj, open(os.path.join(run, "samples.json"), "w", encoding="utf-8"))
            r2 = GR.main(base + ["--out", run, "--prompts", same_prompts])
            check("readings %s: samples.json'dan farkli metin reproduced.different'ta" % name,
                  r2["reproduced"]["different"] == [labels[1]] and r2["reproduced"]["same"] == [labels[0], labels[2]])
            check("readings %s: reading_prompts.json (samples.* uzerine yazar) DURUR" % name,
                  T2._raises(AssertionError, GR.main, base + ["--out", run, "--prompts", prompts]))
            if name == "model_z_learned":
                pack = torch.load(os.path.join(run, "agent.pt"), weights_only=False)
                pack["identity"]["learned_z"] = 0                          # formullu Model Z kimligi (belge 44)
                torch.save(pack, os.path.join(run, "agent.pt"))
                os.remove(os.path.join(run, "samples_same.json"))
                msg = T2._exit_msg(GR.main, base + ["--out", run, "--prompts", same_prompts]) or ""
                check("readings model_z: formullu kimlik DURUR (iletide git etiketi), cikti yazilmaz",
                      "v2-before-formula-cleanup-20261007" in msg
                      and not os.path.exists(os.path.join(run, "samples_same.json")), msg[:120])
    except Exception:  # noqa: BLE001
        check("readings", False, traceback.format_exc(limit=3))
    finally:
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, TR.MODEL_Z_GLOBAL_RATIO = saved
    if T2.DRIVE is None:
        print("ATLA readings (ek istem secimi, Drive kosulari): Drive yok", flush=True)
        return
    data = T2.DRIVE + "/v2/simplestories_gpt2"
    runs = T2.DRIVE + "/v2/runs/"
    got = {}
    for r, dd in (("v2_mzl_d512_l8_lr5e-4_20261006_172433", data), ("v2_mzl_g1_d512_l8_muon_lr2e-3_20261007_075626", data),
                  ("v2_mzown_d512_l8_lr5e-4_20261006_143438", data),
                  ("v2_mzl_g3_fw_d768_l10_muon_lr2e-3_20261008_065800", T2.DRIVE + "/v2/fineweb_edu_s000")):
        if not os.path.exists(runs + r + "/agent.pt"):
            got[r] = "yok"
            continue
        try:
            m, idt = GR.load_run(runs + r, dd, torch.device("cpu"))
            got[r] = (m.global_layers, idt.get("global_layers"))
            del m
        except SystemExit as e:
            got[r] = str(e.code)
    tag = "v2-before-cleanup-20261008"
    check("Drive kosulari (belge 77): 8 Ekim oncesi Model Z (ogrenilen z, G1 muon; summaries_last alani yok) ve temizlik "
          "oncesi own_vocab DURUR (iletide etiketler); 8 Ekim G3 kosusu (065800) yuklenir",
          tag in str(got["v2_mzl_d512_l8_lr5e-4_20261006_172433"]) and tag in str(got["v2_mzl_g1_d512_l8_muon_lr2e-3_20261007_075626"])
          and "v2-before-cleanup-20261006" in str(got["v2_mzown_d512_l8_lr5e-4_20261006_143438"])
          and got["v2_mzl_g3_fw_d768_l10_muon_lr2e-3_20261008_065800"] in ((3, 3), "yok"),
          str({k[:22]: (v if not isinstance(v, str) else v[:60]) for k, v in got.items()}))
    spec, again = GR.extra_prompts(data), GR.extra_prompts(data)
    disk = json.load(open(GR.EXTRA_PROMPTS, encoding="utf-8"))
    fixed = {p["story"] for p in json.load(open(TR.READING_PROMPTS, encoding="utf-8"))["prompts"]}
    v1 = set(GR.V1_POOL)
    exam = set(np.load(os.path.join(data, "exam_pack_plan.npz"))["row_stories"].tolist())
    n_sent = np.diff(np.load(os.path.join(data, "valid_story_offsets.npy")))
    p = disk["prompts"]
    stories = [x["story"] for x in p]
    check("reading_prompts_extra.json = extra_prompts(Drive) (deterministik); 5 yeni sinav hikayesi x {1, 3} cumle, "
          "greedy, okuma 11-20; sabit 10 ve V1 havuzuyla kesisim yok; en az 8 cumle",
          spec == again == disk and len(set(stories)) == 5 and not (set(stories) & (fixed | v1)) and v1 >= fixed
          and set(stories) <= exam and all(n_sent[s] >= GR.MIN_SENTENCES for s in stories)
          and [x["label"] for x in p] == ["okuma %d" % i for i in range(11, 21)]
          and [x["sentences"] for x in p] == [1, 3] * 5 and stories[::2] == stories[1::2]
          and all(x["decode"] == "greedy" for x in p), str(stories[::2]))


def t_tools():
    """Matematikcinin teshis araclari (gap_v2, order_probe, z_ablate; belge 33 adim 5): gercek train.py kosulari (transformer,
    Model Z G'siz ve G 1; varsayilan duzen summaries_last ile egitilir / sinanir) load_run ile; gap_v2 nll'i egitimin sinav
    kaybi; hedefler bagimsiz donguyle (aracin icinde assert); sayim dosyasi = akistan sayim; order_probe sonlu; z_ablate
    kosullari (none = gap, read_off kaybi degistirir)."""
    import traceback
    import gap_v2 as G
    import order_probe as OP
    import z_ablate as ZA
    import train as TR
    tp = T2.tokenizer_path()
    if tp is None:
        print("ATLA tools: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = T2._train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, TR.MODEL_Z_GLOBAL_RATIO)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.MODEL_Z_GLOBAL_RATIO = 0.6                                       # eski varsayilan: L1 / L2 G1 (8 Ekim)
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=6, max_tokens=4)
    base = ["--data", data, "--stream", root]
    runs = {}
    try:
        for name, extra in (("transformer", []), ("model_z_learned", ["--global_layers", "0"]),
                            ("model_z_global", ["--layers", "2"])):
            run = os.path.join(T2.TMP, "runs_tools", name)
            TR.main(base + ["--device", "cpu", "--model", "transformer" if name == "transformer" else "model_z", "--d", "16",
                            "--layers", "1", "--heads", "2", "--lr", "1e-2", "--steps", "6", "--out", run,
                            "--checkpoint_minutes", "0", "--optimizer", "adamw"] + extra)
            runs[name] = run
        exam = {n: json.load(open(os.path.join(r, "results.json"), encoding="utf-8"))["exam"]["loss"]
                for n, r in runs.items()}
        out = os.path.join(T2.TMP, "tools_gap")
        res = G.main(base + ["--tf", runs["transformer"], "--mz", runs["model_z_learned"], "--out", out + "_a"])
        cpath = os.path.join(T2.TMP, "tools_counts.npy")
        import data as DD
        np.save(cpath, DD.token_counts(root))
        res_c = G.main(base + ["--tf", runs["transformer"], "--mz", runs["model_z_learned"], "--out", out + "_b",
                               "--counts", cpath])
        res_g = G.main(base + ["--tf", runs["transformer"], "--mz", runs["model_z_global"], "--out", out + "_g"])
        strip = lambda r: {k: v for k, v in r.items() if k != "girdi"}  # noqa: E731
        check("tools gap_v2: tf / mz nll = egitimin sinav kaybi (load_run, egitimle ayni maske yolu; G'siz ve G 1); "
              "--counts dosyasi = akistan sayim; ciktilar yazildi",
              abs(res["toplam"]["tf"] - exam["transformer"]) < 1e-4
              and abs(res["toplam"]["mz"] - exam["model_z_learned"]) < 1e-4
              and abs(res_g["toplam"]["mz"] - exam["model_z_global"]) < 1e-4 and strip(res) == strip(res_c)
              and res_g["girdi"]["identity"]["model_z"]["global_layers"] == 1
              and all(os.path.exists(os.path.join(out + "_a", f)) for f in ("gap.json", "gap.md", "gap_targets.npz")),
              "tf %.4f / %.4f, mz %.4f / %.4f, G %.4f / %.4f" % (res["toplam"]["tf"], exam["transformer"],
                                                               res["toplam"]["mz"], exam["model_z_learned"],
                                                               res_g["toplam"]["mz"], exam["model_z_global"]))
        op = OP.main(base + ["--runs"] + list(runs.values()) + ["--stories", "4", "--out",
                                                                 os.path.join(T2.TMP, "tools_order.json")])
        fin = lambda d: all(np.isfinite(v) for v in d.values() if isinstance(v, float))  # noqa: E731
        check("tools order_probe: uc kosu (model turu kimlikten), kim kime ve gecmis karistirma sonlu",
              set(op) == {os.path.basename(r) for r in runs.values()} and all(
                  fin(r["kim_kime"]) and all(fin(c) for c in r["gecmis_karistirma"].values()) and "sira_etkisi" in
                  r["kim_kime"] for r in op.values()) and op["model_z_global"]["identity"]["global_layers"] == 1,
              str({n: r["kim_kime"]["sira_etkisi"] for n, r in op.items()}))
        za = ZA.main(base + ["--runs", runs["model_z_learned"], runs["model_z_global"]])
        lz, gz = za["model_z_learned"]["kosullar"], za["model_z_global"]["kosullar"]
        check("tools z_ablate: kosullar none / read_off; none = gap_v2 nll (G'siz, G 1); read_off kaybi degistirir (G 1'de "
              "global blok aynen)", set(lz) == set(gz) == set(ZA.CONDS)
              and abs(lz["none"]["hepsi"] - res["toplam"]["mz"]) < 1e-4
              and abs(gz["none"]["hepsi"] - res_g["toplam"]["mz"]) < 1e-4
              and lz["read_off"]["hepsi"] != lz["none"]["hepsi"] and gz["read_off"]["hepsi"] != gz["none"]["hepsi"],
              "read_off %+.4f / %+.4f" % (lz["read_off"]["fark"]["hepsi"], gz["read_off"]["fark"]["hepsi"]))
    except Exception:  # noqa: BLE001
        check("tools", False, traceback.format_exc(limit=4))
    finally:
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, TR.MODEL_Z_GLOBAL_RATIO = saved


def t_knowledge():
    """knowledge_exam run hizlandirmasi (8 Ekim): tek uretim -> satirlar eski iki uretimli kodla (5 + 80 cumle) birebir
    (loop_tail 0); loop_tail 5'te cevap / puan ayni, durma uretimi tam uretimin oneki, dongu isaretli; stop_when iki
    modelde onek.  Kucuk rastgele modeller (acgozlu, hizla donguye girer), GPT-2 tokenizer."""
    import importlib
    import knowledge_exam as KE
    tp = T2.tokenizer_path()
    if tp is None:
        print("ATLA knowledge: GPT-2 tokenizer yok", flush=True)
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tp)
    root = os.path.dirname(HERE)
    sys.path.insert(0, os.path.join(root, "model_z"))
    sys.path.insert(0, os.path.join(root, "transformer"))
    SM, BL = importlib.import_module("sentence"), importlib.import_module("baseline")
    facts = [dict(id="f%d" % i, category="c", topic=["x"], key=[k], distractors=[d_], prompt=pr) for i, (pr, k, d_) in
             enumerate((("The capital of France is", "Paris", "Lyon"), ("World War II began in", "1939", "1914"),
                        ("Water boils at", "100", "90"), ("The largest planet is", "Jupiter", "Saturn")))]
    counts = {f["id"]: dict(band="0", docs_same=0) for f in facts}
    lookup = {f["id"]: dict(answer="", score="BOS") for f in facts}
    dec = lambda ids: tok.decode(list(ids))  # noqa: E731
    prompts = [[tok.encode(f["prompt"]).ids] for f in facts]
    ok_old, ok_new, ok_pre, info = True, True, True, []
    for name in ("model_z", "transformer"):
        torch.manual_seed(0)
        m = (SM.SentenceTransformer(32, 2, 2, global_layers=1) if name == "model_z" else BL.BaselineTransformer(32, 2, 2)).eval()
        with torch.no_grad():
            ans = m.generate(prompts, open_last=True, **KE.ANSWER)              # eski kod: iki ayri uretim
            stop = m.generate(prompts, open_last=True, **KE.STOP)
        old = []
        for f, p_, (g, e, eos), (g2, e2, eos2) in zip(facts, prompts, ans, stop):
            w = "".join(dec(s_) for s_ in g[:2])
            old.append(dict(answer=w, score=KE.score(w, f)[0], stop_eos=bool(eos2), stop_sentences=len(g2),
                            stop_tokens=int(sum(len(s_) for s_ in g2)), stop_text=dec(p_[0]) + "".join(dec(s_) for s_ in g2)))
        keys = list(old[0])
        r0 = KE.answer_rows(m, facts, prompts, counts, lookup, dec, 0)
        r5 = KE.answer_rows(m, facts, prompts, counts, lookup, dec, 5)
        ok_old &= [{k: r[k] for k in keys} for r in r0] == old and not any(r["stop_loop"] for r in r0)
        ok_new &= all(a_["answer"] == b_["answer"] and a_["score"] == b_["score"] and a_["stop_text"].startswith(
            b_["stop_text"]) and b_["stop_sentences"] <= a_["stop_sentences"] for a_, b_ in zip(r0, r5))
        with torch.no_grad():
            pre = m.generate(prompts, open_last=True, stop_when=KE.loop_stop(5), **KE.STOP)
        ok_pre &= all(gp == gf[:len(gp)] and ep == ef[:len(ep)] for (gp, ep, _), (gf, ef, _) in zip(pre, stop))
        info.append("%s: cumle %s -> %s, dongu %d" % (name, [r["stop_sentences"] for r in r0],
                                                       [r["stop_sentences"] for r in r5], sum(r["stop_loop"] for r in r5)))
    check("knowledge_exam: tek uretim (loop_tail 0) = eski iki uretimli satirlar birebir (cevap, puan, EOS, cumle / token, "
          "metin); loop_tail 5 cevap / puan ayni, durma metni tam uretimin oneki; stop_when iki modelde onek",
          ok_old and ok_new and ok_pre and all("dongu 0" not in x for x in info), "; ".join(info))
    st = KE.loop_stop(2)
    check("knowledge_exam loop_stop: son 2 cumle daha once uretilmis -> dur; yeni cumle -> surer",
          st([[1], [2], [1], [2]]) and not st([[1], [2], [1], [3]]) and not st([[1], [1]]) and st([[1], [1], [1]])
          and KE.loop_stop(0) is None)


def t_trace():
    """token_trace (belge 80): G2 + GQA (kv 2) + summaries_last 1 ve G'siz kosu, gercek build_batch.  Elle attention x v =
    blok ciktisi (her katman), kategoriler toplami 1, yerel katmanda onceki cumle token'i 0; son katman lens'i = son tahmin;
    none log-olasiligi = loss_per_target = summaries_last sinav yolu (_Exam); read_off = z_ablate maskesiyle; z_unseen maskesi
    = okuma maskesi eksi (TOKEN -> ZTOK); hook'lar sonra model bit ayni; text_story metni kayipsiz; main: JSON / md, istem +
    serbest metin, --generate = model.generate (uretilen konumda ilk aday), G'siz kosuda g_* null."""
    import traceback
    import data as D
    import gap_v2 as G
    import token_trace as TT
    import train as TR
    import z_ablate as ZA
    tp = T2.tokenizer_path()
    if tp is None:
        print("ATLA trace: GPT-2 tokenizer yok", flush=True)
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tp)
    root, data, prompts = T2._train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=3, max_tokens=4)
    try:
        base = ["--data", data, "--stream", root, "--device", "cpu", "--model", "model_z", "--d", "16", "--heads", "4",
                "--steps", "6", "--checkpoint_minutes", "0"]
        runs = {}
        for name, extra in (("g2kv", ["--layers", "3", "--global_layers", "2", "--glob_kv_heads", "2"]),
                            ("g0", ["--layers", "2", "--global_layers", "0"]),
                            ("rand", ["--layers", "3", "--global_layers", "2", "--glob_kv_heads", "2", "--lr", "1e-9",
                                      "--steps", "1"])):     # neredeyse ilk agirlik: uretim END / EOS'a erken dusmez
            runs[name] = os.path.join(T2.TMP, "runs_trace", name)
            TR.main(base + extra + ["--out", runs[name]])
        cpu = torch.device("cpu")
        model = G.load(runs["g2kv"], data, cpu)[0]
        S = sys.modules[type(model).__module__]
        valid = D.TokenStories(root, data, "valid")
        sents = [s.tolist() for s in valid.sentences(1)]
        st = TT.text_story(tok, [sents], "ss")
        n = 1 + sum(len(s) + 1 for s in sents)
        batch = D.build_batch(st, [[0]], "model_z", "cpu", row_len=n)
        with torch.no_grad():
            before = model._batch_hidden(batch)
        ins = []
        hooks = [b.register_forward_pre_hook(lambda m, args: ins.append(args)) for b in model.blocks]
        with torch.no_grad():
            model._batch_hidden(batch)
            worst = 0.0
            for b, (x, pos, mask) in zip(model.blocks, ins):
                w = TT.attention_weights(b, x, pos, mask)
                q, k, v = b._qkv(x, pos)
                a = w @ v.repeat_interleave(b.heads // b.kv_heads, 1)
                worst = max(worst, float((b._finish(x, a) - b(x, pos, mask)).abs().max()))
        for h in hooks:
            h.remove()
        r = TT.trace(model, batch, TT.CONDS.split(","), 5, True)
        has = r["has_target"]
        with torch.no_grad():
            after = model._batch_hidden(batch)
            nll = model.loss_per_target(batch)[0]
            nll_last = TR._Exam(model, model._masks(True), False, S.summaries_last).loss_per_target(batch)[0]
            with ZA.ablated(model, "read_off") as fn:
                ro = model.loss_per_target(batch, tuple(S._dense(f(batch.kind, batch.doc, batch.sent), 1, n, "cpu")
                                                        for f in fn))[0]
        read = S._dense(S.model_z_read_mask(batch.kind, batch.doc, batch.sent), 1, n, "cpu")
        uns = S._dense(TT.model_z_unseen_mask(batch.kind, batch.doc, batch.sent), 1, n, "cpu")
        tz = (batch.kind[:, :, None] == S.TOKEN) & (batch.kind[:, None, :] == S.ZTOK)
        lp = r["logp"][has]
        check("token_trace: elle attention x v = blok ciktisi (3 katman, GQA kv 2; fark %.1e); kategoriler toplami 1; yerel "
              "katmanda onceki cumle token'i 0; head basina boyut; son katman lens'i = son tahmin" % worst,
              worst < 1e-5 and bool(torch.allclose(r["attn"].sum(-1), torch.ones(()), atol=1e-5))
              and float(r["attn"][0, :, 4].abs().max()) < 1e-6 and tuple(r["attn_heads"].shape) == (3, 4, n, 5)
              and bool(torch.allclose(r["lens_logp"][-1], r["logp"], atol=1e-5))
              and torch.equal(r["lens_rank"][-1], r["rank"]))
        check("token_trace: none log-olasiligi = loss_per_target = summaries_last sinav yolu; read_off farki = z_ablate "
              "maskesiyle; z_unseen = okuma maskesi eksi TOKEN -> ZTOK; hook'lar kaldirildi (model bit ayni); g_off / g_local "
              "sonlu", float((lp + nll).abs().max()) < 1e-5 and float((lp + nll_last).abs().max()) < 1e-5
              and float((r["ablation"]["read_off"][has] - (-ro - lp)).abs().max()) < 1e-5
              and torch.equal(uns, read & ~tz) and bool((read & tz).any()) and torch.equal(before, after)
              and all(bool(torch.isfinite(r["ablation"][c]).all()) for c in ("g_off", "g_local", "read_off+g_off")),
              "nll fark %.1e / %.1e" % (float((lp + nll).abs().max()), float((lp + nll_last).abs().max())))
        text = "Lily had a red ball. She liked it a lot. Then she went home."
        ts = TT.text_story(tok, [text], "ss")
        check("token_trace text_story: serbest metin cumlelere bolunur, token'lar kayipsiz (decode = metin)",
              tok.decode(np.concatenate(ts.sentences(0)).tolist()) == text and len(ts.sentences(0)) == 3,
              str([tok.decode(s.tolist()) for s in ts.sentences(0)]))
        out = os.path.join(T2.TMP, "trace_out")
        TT.main(["--run", runs["g2kv"], "--data", data, "--stream", root, "--prompts", prompts, "--text", text,
                       "--device", "cpu", "--out", out, "--heads"])
        js = json.load(open(os.path.join(out, "token_trace.json"), encoding="utf-8"))
        keys = {"i", "text", "input_kind", "sent", "pos_in_sent", "generated", "target", "target_kind", "p_target", "rank",
                "top", "lens", "attn", "ablation", "attn_heads"}
        lengths = [x["length"] for x in js["texts"]]
        gen_out = os.path.join(T2.TMP, "trace_gen")
        rg = TT.main(["--run", runs["rand"], "--data", data, "--stream", root, "--prompts", prompts, "--generate", "2",
                      "--device", "cpu", "--out", gen_out, "--conds", "read_off"])
        pj = json.load(open(prompts, encoding="utf-8"))["prompts"]
        m2 = G.load(runs["rand"], data, cpu)[0]
        want = [m2.generate([[s.tolist() for s in valid.sentences(p["story"])[:p["sentences"]]]], max_sentences=2,
                            max_tokens=128)[0][0] for p in pj]
        got = [[t for t in x["tokens"] if t["generated"] and t["input_kind"] == "TOKEN"] for x in rg["texts"]]
        n_gen = [sum(len(s) for s in w) for w in want]
        near = all(t["p_target"] >= t["top"][0][1] - 1e-5 for g in got for t in g)
        r0 = TT.main(["--run", runs["g0"], "--data", data, "--stream", root, "--prompts", prompts, "--device", "cpu",
                      "--out", os.path.join(T2.TMP, "trace_g0")])
        check("token_trace main: JSON (3 istem + 1 serbest metin; konum sayisi = hikaye boyu; alanlar tam; katman turleri), "
              "md yazildi; --generate 2 = model.generate (uretilen token sayisi, uretilen konumda ilk aday); G'siz kosuda "
              "g_* null, read_off sayi",
              len(js["texts"]) == 4 and [len(x["tokens"]) for x in js["texts"]] == lengths
              and all(set(t) == keys for x in js["texts"] for t in x["tokens"])
              and [l_["kind"] for l_ in js["layers"]] == ["loc", "glob", "glob"]
              and os.path.exists(os.path.join(out, "token_trace.md")) and js["texts"][3]["source"] == "text"
              and [len(g) for g in got] == n_gen and min(n_gen) > 0 and near
              and r0["summary"]["ablation"]["g_off"] is None and r0["summary"]["ablation"]["read_off"] is not None,
              "uretilen %s / %s" % ([len(g) for g in got], n_gen))
    except Exception:  # noqa: BLE001
        check("trace", False, traceback.format_exc(limit=4))
    finally:
        (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS) = saved


TESTS = dict(readings=t_readings, tools=t_tools, knowledge=t_knowledge, trace=t_trace)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
