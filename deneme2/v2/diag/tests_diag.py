"""tests_diag -- V2 teshis araclari testleri (CPU; belge 33 adim 5).  Gruplar: readings (generate_readings: kayitli
kosudan okuma = train.py'ninki; arsiv kimligi durur; ek istem secimi Drive'dan), tools (gap_v2, order_probe,
z_ablate; model-z-mathematician), bag (bag_report: aday havuzu A0, belge 53-55).  Formullu z cesitleri kaldirildi
(belge 44).  Yardimcilar common/tests_v2'den (_train_root, tokenizer_path, DRIVE).

    python tests_diag.py [--only readings,tools,bag]
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


def t_bag():
    """bag_report (A0): gercek kucuk train.py kosusu (Model Z + G) uzerinde secici uydurma + rapor, onsel acik / kapali;
    sayim dosyasi depodaki kodla uretilir; C'de END ve EOS; kaynak paylari toplami 1, kacan M ile artmaz; uydurma ana
    modeli degistirmez; meaning onseli elle hesapla ayni; ikinci kosu ayni klasorde durur."""
    import math
    import traceback
    import bag_report as BR
    import data as DD
    import generate_readings as GR
    import train as TR
    tp = T2.tokenizer_path()
    if tp is None:
        print("ATLA bag: GPT-2 tokenizer yok", flush=True)
        return
    root, data, prompts = T2._train_root(tp)
    saved = (TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, DD.BATCH_ROWS)
    TR.BATCH_ROWS, TR.LOG_EVERY = 4, 1
    TR.READING_PROMPTS, TR.READING_LIMITS = prompts, dict(max_sentences=6, max_tokens=4)
    try:
        run = os.path.join(T2.TMP, "runs_bag", "mz")
        TR.main(["--data", data, "--stream", root, "--device", "cpu", "--model", "model_z", "--d", "16", "--layers", "2",
                 "--heads", "2", "--lr", "1e-2", "--steps", "4", "--out", run, "--checkpoint_minutes", "0"])
        DD.BATCH_ROWS = 4
        g = torch.Generator().manual_seed(0)
        mpath = os.path.join(T2.TMP, "runs_bag", "meaning")
        os.makedirs(mpath, exist_ok=True)
        V = DD.EOS_ID + 1
        count = torch.zeros(V)
        count[torch.as_tensor(DD.token_counts(root)[:V] > 0)] = 1.0
        torch.save(dict(state={"source.weight": torch.randn(V, 8, generator=g), "target.weight": torch.randn(V, 8, generator=g),
                               "bias": torch.randn(V, generator=g)}, count=count,
                        tokenizer_sha256=BR._sha256(os.path.join(root, "gpt2", "tokenizer.json"))),
                   os.path.join(mpath, "agent.pt"))
        base = ["--run", run, "--data", data, "--stream", root, "--device", "cpu", "--bag_core", "5", "--steps", "3",
                "--seen_batches", "1"]
        cpath = os.path.join(data, "train_token_counts.npy")
        had = os.path.exists(cpath)
        rep = {k: BR.main(base + ["--out", os.path.join(T2.TMP, "bag_" + k)] + x)
               for k, x in (("off", []), ("on", ["--bag_prior", mpath, "--queries", "2"]), ("cl", ["--bag_clusters", "4"]))}
        check("bag: sayim dosyasi yoktu, depodaki kodla uretildi (= data.token_counts)",
              not had and np.array_equal(np.load(cpath), DD.token_counts(root)))
        cfg = json.load(open(os.path.join(T2.TMP, "bag_on", "config.json"), encoding="utf-8"))
        check("bag: C = en sik 5 + END + EOS; onsel sha kimlikte (bag_prior_sha256)",
              {DD.END_ID, DD.EOS_ID} <= set(cfg["core_tokens"]) and cfg["core"] in (6, 7)
              and cfg["bag_prior_sha256"] == BR._sha256(os.path.join(mpath, "agent.pt")))
        ok, names = True, set()
        for k, r in rep.items():
            for part in ("exam", "train_seen"):
                e = r[part]
                for sc in [x for x in e if isinstance(e[x], dict)]:
                    names.add(sc)
                    bm = e[sc]["by_m"]
                    ms = [bm[str(m)]["miss"] for m in BR.M_LIST]
                    ok &= all(abs(e["share_c"] + e["share_p"] + bm[str(m)]["share_l"] + bm[str(m)]["miss"] - 1) < 2e-4
                              for m in BR.M_LIST)
                    ok &= all(a >= b for a, b in zip(ms, ms[1:])) and abs(ms[0] - (1 - e["share_c"] - e["share_p"])) < 2e-4
        check("bag: paylar C + P + L + kacan = 1 her M'de, kacan M ile artmaz, M 0 = C u P disi (%s)" % sorted(names),
              ok and names == {"selector", "freq", "meaning", "cluster", "cluster_freq"})
        ccfg = json.load(open(os.path.join(T2.TMP, "bag_cl", "config.json"), encoding="utf-8"))["clusters"]
        z = np.load(os.path.join(T2.TMP, "bag_cl", "bag_clusters.npz"))
        lm_ok = all(v["l_actual"] <= int(m) for v_ in (rep["cl"]["exam"]["cluster"]["by_m"],) for m, v in v_.items())
        check("bag kume: dosya sha'si config'te; her uygun token tek kumede, boy <= sinir; |L| <= M; maliyet raporda",
              ccfg["sha256"] == BR._sha256(os.path.join(T2.TMP, "bag_cl", "bag_clusters.npz"))
              and len(z["token"]) == ccfg["tokens"] == len(set(z["token"].tolist()))
              and np.bincount(z["cluster"]).max() <= ccfg["limit"] and lm_ok
              and rep["cl"]["cost"]["selector_flop_per_bag"] == 2 * 16 * (16 + 4), str(ccfg))
        check("bag: ornek dosyasi dolu", os.path.getsize(os.path.join(T2.TMP, "bag_on", "bag_examples.txt")) > 100)
        try:
            BR.main(base + ["--out", os.path.join(T2.TMP, "bag_off")])
            stopped = False
        except SystemExit as e:
            stopped = "bag_head.pt var" in str(e)
        check("bag: ayni klasorde ikinci kosu durur (kisa deneme, surdurme yok)", stopped)
        model, _ = GR.load_run(run, data, torch.device("cpu"))
        model.eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        counts = np.load(cpath)
        core, rank = BR.core_mask(counts, 5, torch.device("cpu"))
        prior = BR.load_prior(mpath, BR._sha256(os.path.join(root, "gpt2", "tokenizer.json")), torch.device("cpu"))
        tr = DD.TokenStories(root, data, "train")
        plan = np.load(os.path.join(data, "train_pack_plan_e1.npz"))
        rows = [plan["row_stories"][plan["row_offsets"][i]:plan["row_offsets"][i + 1]].tolist() for i in range(4)]
        batch = DD.build_batch(tr, rows, "model_z", "cpu", int(plan["row_len"]))
        head = BR.BagHead(16, 1).float()
        h0 = {k: v.clone() for k, v in head.state_dict().items()}
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
        b = BR.bag_batch(model, model.mask_fn, batch, core, False, prior, rank)
        l1 = [BR.fit_step(head, opt, model.E.weight, b, core, True)[0] for _ in range(3)]
        check("bag: uydurma ana modeli degistirmez, seciciyi degistirir, ayni batch'te kayip duser",
              all(torch.equal(before[k], v) for k, v in model.state_dict().items())
              and not torch.equal(h0["q.weight"], head.q.weight) and l1[-1] < l1[0], str([round(x, 4) for x in l1]))
        skip, BR.PRIOR_SKIP = BR.PRIOR_SKIP, 0                              # kucuk sozluk: her token oy versin
        b = BR.bag_batch(model, model.mask_fn, batch, core, False, prior, rank)
        BR.PRIOR_SKIP = skip
        ids, br_, bc_ = BR.bag_index(batch)
        j = int(((batch.kind[br_, bc_] == DD.Kind.ZTOK) & (batch.sent[br_, bc_] >= 2)).nonzero()[0, 0])
        r, c = int(br_[j]), int(bc_[j])
        d_, k_ = int(batch.doc[r, c]), int(batch.sent[r, c])
        sel = ((batch.doc[r] == d_) & (batch.kind[r] == DD.Kind.TOKEN) & (batch.sent[r] <= k_)
               & (batch.sent[r] > k_ - BR.PRIOR_WINDOW))
        voters = sorted({t for t in batch.tokens[r][sel].tolist() if count[t] > 0})
        P = [torch.softmax(prior["bias"] + prior["src"][i] @ prior["tgt"].T / math.sqrt(8), -1) for i in voters]
        want = (torch.stack(P).mean(0) if P else torch.softmax(prior["bias"], -1)).clamp_min(1e-30).log()
        check("bag: meaning onseli (torba %d, %d oy) elle hesapla ayni" % (j, len(voters)),
              len(voters) > 1 and torch.allclose(b["prior"][j, :V], want, atol=1e-5))
        g2 = torch.Generator().manual_seed(3)
        nC, V_ = 5, DD.VOCAB
        of = torch.full((V_,), -1, dtype=torch.long)
        toks = torch.arange(100, 120)
        of[toks] = torch.arange(20) % nC
        cl = dict(of=of, size=torch.bincount(of[toks], minlength=nC), n=nC)
        P = torch.zeros(3, V_, dtype=torch.bool)
        P[1, [100, 105, 101]] = True
        y = torch.tensor([100, 102, 107, 113, 119, 101, 104, 200])
        bb = dict(P=P, bag=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]), y=y, n=3, cl=of[y],
                  src=torch.tensor([BR.SRC_REST] * 5 + [BR.SRC_P, BR.SRC_REST, BR.SRC_REST]))
        sc = torch.randn(3, nC, generator=g2)
        rk, lsz = BR.cluster_ranks(bb, dict(c=lambda b0, b1: sc[b0:b1]), cl)
        want, wl = [], []
        for t in range(len(y)):
            k, c = int(bb["bag"][t]), int(of[y[t]])
            new = [int(cl["size"][j]) - int(P[k][toks][of[toks] == j].sum()) for j in range(nC)]
            order = sorted(range(nC), key=lambda j: -float(sc[k, j]))
            cum = np.cumsum([new[j] for j in order])
            want.append(-1 if int(bb["src"][t]) != BR.SRC_REST else 10 ** 9 if c < 0 else int(cum[order.index(c)]) - 1)
            wl.append([max([0] + [x for x in cum if x <= m]) for m in BR.M_LIST])
        check("bag kume: gereken butce (sirali kumelerin birikimli YENI kelimesi) ve gercek |L| (butceyi asan kume "
              "alinmaz) kaba kuvvetle ayni; kumesiz hedef hic yakalanmaz",
              rk["c"].tolist() == want and lsz["c"].tolist() == [[float(x) for x in r] for r in wl], str(rk["c"].tolist()))
        vec = torch.nn.functional.normalize(torch.randn(40, 6, generator=g2), dim=1)
        a1, c1 = BR.cluster_vocab(vec, 4, 0)
        a2, _ = BR.cluster_vocab(vec, 4, 0)
        check("bag kume: k-means deterministik (ayni tohum ayni atama), boy <= ceil(1,5 x 10) = 15, merkez birim",
              torch.equal(a1, a2) and int(torch.bincount(a1).max()) <= 15 and torch.allclose(c1.norm(dim=1), torch.ones(4)))
    except Exception:  # noqa: BLE001
        check("bag", False, traceback.format_exc(limit=4))
    finally:
        TR.BATCH_ROWS, TR.LOG_EVERY, TR.READING_PROMPTS, TR.READING_LIMITS, DD.BATCH_ROWS = saved


TESTS = dict(readings=t_readings, tools=t_tools, bag=t_bag)

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(TESTS)
    for name in only:
        TESTS[name]()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d / %d gecti%s" % (len(RESULTS) - len(bad), len(RESULTS), "" if not bad else " | HATA: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
