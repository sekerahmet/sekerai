"""tests_diag -- V2 teshis araclari testleri (CPU; belge 33 adim 5).  Gruplar: readings (generate_readings: kayitli
kosudan okuma = train.py'ninki; arsiv kimligi durur; ek istem secimi Drive'dan), tools (gap_v2, order_probe,
z_ablate; model-z-mathematician).  Formullu z cesitleri kaldirildi (belge 44).  Yardimcilar common/tests_v2'den
(_train_root, tokenizer_path, DRIVE).

    python tests_diag.py [--only readings,tools]
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
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
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
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS = saved
    if T2.DRIVE is None:
        print("ATLA readings (ek istem secimi, Drive kosulari): Drive yok", flush=True)
        return
    data = T2.DRIVE + "/v2/simplestories_gpt2"
    runs = T2.DRIVE + "/v2/runs/"
    got = {}
    for r in ("v2_mzl_d512_l8_lr5e-4_20261006_172433", "v2_mzl_g1_d512_l8_muon_lr2e-3_20261007_075626",
              "v2_mzown_d512_l8_lr5e-4_20261006_143438"):
        if not os.path.exists(runs + r + "/agent.pt"):
            got[r] = "yok"
            continue
        try:
            m, idt = GR.load_run(runs + r, data, torch.device("cpu"))
            got[r] = (m.global_layers, idt.get("global_layers"))
            del m
        except SystemExit as e:
            got[r] = str(e.code)
    check("Drive kosulari (belge 44): eski ogrenilen z (global_layers alani yok) G'siz yuklenir, G kosusu G 1 ile; "
          "temizlik oncesi formullu own_vocab kosusu DURUR (iletide etiket)",
          got["v2_mzl_d512_l8_lr5e-4_20261006_172433"] == (0, None)
          and got["v2_mzl_g1_d512_l8_muon_lr2e-3_20261007_075626"] == (1, 1)
          and "v2-before-cleanup-20261006" in str(got["v2_mzown_d512_l8_lr5e-4_20261006_143438"]),
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
    formullu Model Z, learned_z) load_run ile; gap_v2 nll'i egitimin sinav kaybi; hedefler bagimsiz donguyle (aracin icinde
    assert); sayim dosyasi = akistan sayim; order_probe sonlu; z_ablate kosullari (none = gap, kapatma etkili, encode_z geri
    yuklenir, sutun bolumleri)."""
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
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
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
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS = saved


TESTS = dict(readings=t_readings, tools=t_tools)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
