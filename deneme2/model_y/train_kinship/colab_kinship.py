# -*- coding: utf-8 -*-
"""colab_kinship -- akrabalik egitiminin Colab kosulari.  Egitim arka planda bir iplikte; hucre hemen doner (CLAUDE.md kural 8).

    import colab_kinship as C
    C.start(NAME, DATA, OUT, steps=S, seed=0, every=100, device="cuda")     compile=True varsayilan (model_19 gibi)
    C.pulse()      her kosunun durumu ve son satirlari
    C.stop()       bayrak: iplik bir sonraki sinavda modeli kaydedip cikar

OUT/config.json kosu ayarlari, OUT/log.txt her sinavin satiri, OUT/exams.json butun sinavlar, OUT/model.pt son agirlik,
OUT/final.json sonda <steps> siniflari uc kosulla (model yazar / ozne verilir / ilk cumle verilir) ve yazilanlar,
OUT/checkpoint_tNNNNN.pt her save_every adimda surdurme paketi (kesilirse start(..., resume=True)).
"""
import json
import os
import sys
import threading
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_y: model ve genel egitim

import exam_kinship as EK  # noqa: E402
import model_y as M  # noqa: E402
import train_y as TR  # noqa: E402

SHORT = ("1R_T",)                    # ilk token: tek adimli bilgi, egitimde yazili
STEPS_CLS = ("2R_T", "2R_UT")         # "<steps>" ile uretim: egitimde gorulen / hic gorulmemis 2R
RUNS = {}


def start(name, data, out, steps, seed=0, every=100, device="cuda", compile=True, setting="shared", save_every=None,
          resume=False, model_kw=None, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez.
    setting: "shared" (Model X, varsayilan ayarlarla) ya da "transformer" (kiyas modeli, model_y_transformer).
    save_every: her save_every adimda out/checkpoint_tNNNNN.pt {step, model, optimizer}.  resume=True: out'taki son
    paketten surdurur -- klasor tasinmaz, gunluk ve sinavlar uzar; ayarlar config.json ile ayni olmali.
    model_kw: BlockModel ayarlari (ornek heads=4); config'te duz yazilir, varsayilanin yerine gecer."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    n = {c: len(EK.questions(data, c)) for c in SHORT + STEPS_CLS}
    # varsayilanlar da yazilir (lr, cosine tabani, clip, wd, rope): config tek basina koşuyu tarif etsin
    compile = compile and torch.device(device).type == "cuda"     # train_seq ile ayni kural: compile yalniz GPU'da
    config = dict(name=name, setting=setting, steps=steps, seed=seed, every=every, device=device, compile=compile,
                  fingerprint=data["fingerprint"], train=len(data["train"]), sizes=n, save_every=save_every,
                  **dict(dict(lr=TR.LR, lr_floor=TR.LR_FLOOR, grad_clip=TR.GRAD_CLIP, weight_decay=TR.WEIGHT_DECAY,
                              optimizer=TR.OPTIMIZER, schedule=TR.SCHEDULE, cooldown=TR.COOLDOWN,
                              stream_norm=TR.STREAM_NORM, layer_norm=TR.LAYER_NORM,
                              normalized_update=TR.NORMALIZED_UPDATE if setting in TR.STEP3 else False,
                              sphere_weights=TR.SPHERE_WEIGHTS if setting in TR.STEP3 else False,
                              canon=TR.CANON if setting in TR.STEP3 else False,
                              turns=M.TURNS if setting in TR.STEP3 else None,
                              layers=M.LAYERS if setting in TR.STEP3 else None,
                              heads=M.HEADS if setting in TR.STEP3 else None,
                              fact_activation=M.FACT_ACTIVATION if setting in TR.STEP3 else None,
                              rope=True if setting.startswith("transformer") else TR.ROPE if setting in TR.STEP3 else False),
                         **(model_kw or {}), **train_kw))
    checkpoint = None
    if resume:
        packs = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t")) if os.path.isdir(out) else []
        if not packs:
            raise RuntimeError("%s: surdurme paketi yok; bastan kosmak ayri karar (resume=False)" % out)
        saved = json.load(open(os.path.join(out, "config.json")))
        assert not saved.pop("copy_path", False), "kopya yolu (Oneri A) 28 Eylul'de kaldirildi: bu kosu surdurulemez"
        assert not saved.pop("output_skip", False), "output_skip (28 Eylul) kaldirildi: bu kosu surdurulemez"
        # 27 Eylul oncesi config'lerde optimizer / takvim yok: o kosular Adam + cosine idi.  compile sonucu degistirir: karsilastirilir
        # 28 Eylul oncesi config'lerde turns / layers yok: Adim 3 modeli tek Block x 2 tur idi
        saved = dict(dict(optimizer="adam", schedule="cosine", cooldown=TR.COOLDOWN,
                          normalized_update=False, sphere_weights=False, canon=False,
                          turns=2 if saved.get("setting") in TR.STEP3 else None,
                          layers=1 if saved.get("setting") in TR.STEP3 else None,
                          heads=1 if saved.get("setting") in TR.STEP3 else None,
                          fact_activation="relu" if saved.get("setting") in TR.STEP3 else None), **saved)
        differ = sorted(k for k in set(saved) | set(config) if k != "device" and saved.get(k) != config.get(k))
        if differ:
            raise RuntimeError("surdurme: ayarlar config.json'dan farkli %s -- ayni ayarlarla surdurulur" % differ)
        checkpoint = torch.load(os.path.join(out, packs[-1]), map_location=device)
        path = os.path.join(out, "exams.json")
        exams = [e for e in json.load(open(path)) if e["step"] <= checkpoint["step"]] if os.path.exists(path) else []
    else:
        exams = []
        if os.path.isdir(out) and os.listdir(out):
            os.rename(out, out + "_eski_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out, exist_ok=True)
        json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
    run = dict(name=name, out=out, lines=[], stop=False, error=None, done=False, t0=time.time(), exams=exams)

    class Stopped(Exception):
        pass

    def note(line):
        run["lines"].append(line)
        with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def callback(step, model, nll):
        e = dict(step=step, nll=nll, secs=round(time.time() - run["t0"], 1))
        for c in SHORT:
            if data.get("long_1r"):              # 1R cevabi tam cumle: yakinin adi cevapta dogru mu (AC)
                counts, _ = EK.exam_steps(model, data, c)
                e[c], e[c + "_EX"] = counts["AC"], counts["EX"]
                continue
            qs = EK.questions(data, c)
            pred = EK.read_questions(model, qs, data["vocab"])[0].tolist()
            e[c] = sum(int(p) in q[1] for p, q in zip(pred, qs))
        for c in STEPS_CLS:
            counts, _ = EK.exam_steps(model, data, c)
            e.update({"%s_%s" % (c, k): v for k, v in counts.items()})
        run["exams"].append(e)
        json.dump(run["exams"], open(os.path.join(out, "exams.json"), "w"), indent=1)
        note("adim %5d  nll %.3f  1R_T %d/%d | 2R_T EX %d/%d | 2R_UT AC %d EX %d BC %d SC %d FC %d /%d  (%.0f sn)" % (
                 step, nll, e["1R_T"], n["1R_T"], e["2R_T_EX"], n["2R_T"], e["2R_UT_AC"], e["2R_UT_EX"], e["2R_UT_BC"],
                 e["2R_UT_SC"], e["2R_UT_FC"], n["2R_UT"], e["secs"]))
        if run["stop"]:
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            raise Stopped()

    def save(step, model, opt):
        torch.save(dict(step=step, model=model.state_dict(), optimizer=opt.state_dict()),
                   os.path.join(out, "checkpoint_t%05d.pt" % step))

    def job():
        try:
            if checkpoint is not None:
                note("SURDURULDU adim %d'den (%s)" % (checkpoint["step"], "checkpoint_t%05d.pt" % checkpoint["step"]))
            ids, mask = EK.sequences(data)
            model, _ = TR.train_seq(setting, ids, mask, len(data["vocab"]), steps=steps, seed=seed, device=device,
                                    every=every, callback=callback, log_at=(), compile=compile, save_every=save_every,
                                    save=save, checkpoint=checkpoint, model_kw=model_kw, **train_kw)
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            final = {}
            for c, givens in [(c, (0,)) for c in SHORT if data.get("long_1r")] + [(c, (0, 2, 8)) for c in STEPS_CLS]:
                for given in givens:
                    counts, rows = EK.exam_steps(model, data, c, given)
                    final["%s_given%d" % (c, given)] = dict(counts=counts, rows=rows)
                    note("SON %-13s verilen %d: %s" % (c, given, "  ".join("%s %d" % kv for kv in counts.items())))
            json.dump(final, open(os.path.join(out, "final.json"), "w"), indent=1, ensure_ascii=False)
            run["done"] = True
            note("BITTI  %.0f sn" % (time.time() - run["t0"]))
        except Stopped:
            note("DURDURULDU (model.pt kaydedildi)")
        except Exception:
            run["error"] = traceback.format_exc()
            note("HATA\n" + run["error"])

    run["thread"] = threading.Thread(target=job, daemon=True)
    RUNS[name] = run
    run["thread"].start()
    return run


def pulse(k=3):
    """Her kosu: durum, sure ve son k satir."""
    if not RUNS:
        print("kosu yok")
    for name, run in RUNS.items():
        state = ("KOSUYOR" if run["thread"].is_alive() else "BITTI" if run["done"] else "HATA" if run["error"]
                 else "DURDU")
        print("%s  %s  %.0f sn  %s" % (name, state, time.time() - run["t0"], run["out"]))
        for line in run["lines"][-k:]:
            print("   " + line)


def stop():
    """Canli kosulara bayrak; iplik bir sonraki sinavda modeli kaydedip cikar."""
    for name, run in RUNS.items():
        if run["thread"].is_alive():
            run["stop"] = True
            print("%s: durdurma bayragi kondu" % name)
