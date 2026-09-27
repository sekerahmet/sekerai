# -*- coding: utf-8 -*-
"""colab_20 -- model_20 Colab kosulari.  Egitim arka planda bir iplikte; hucre hemen doner (CLAUDE.md kural 8).

    import colab_20 as C
    C.start(NAME, DATA, OUT, steps=S, seed=0, every=100, device="cuda")     compile=True varsayilan (model_19 gibi)
    C.pulse()      her kosunun durumu ve son satirlari
    C.stop()       bayrak: iplik bir sonraki sinavda modeli kaydedip cikar

OUT/config.json kosu ayarlari, OUT/log.txt her sinavin satiri, OUT/exams.json butun sinavlar, OUT/model.pt son agirlik,
OUT/final.json sonda <steps> siniflari uc kosulla (model yazar / ozne verilir / ilk cumle verilir) ve yazilanlar.
"""
import json
import os
import threading
import time
import traceback

import torch

import train_20 as TR

SHORT = ("1R_T",)                    # ilk token: tek adimli bilgi, egitimde yazili
STEPS_CLS = ("2R_T", "2R_UT")         # "<steps>" ile uretim: egitimde gorulen / hic gorulmemis 2R
RUNS = {}


def start(name, data, out, steps, seed=0, every=100, device="cuda", compile=True, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    if os.path.isdir(out) and os.listdir(out):
        os.rename(out, out + "_eski_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    n = {c: len(TR.questions(data, c)) for c in SHORT + STEPS_CLS}
    run = dict(name=name, out=out, lines=[], exams=[], stop=False, error=None, done=False, t0=time.time())
    json.dump(dict(name=name, steps=steps, seed=seed, every=every, device=device, compile=compile, fingerprint=data["fingerprint"],
                   train=len(data["train"]), sizes=n, **train_kw), open(os.path.join(out, "config.json"), "w"), indent=1)

    class Stopped(Exception):
        pass

    def note(line):
        run["lines"].append(line)
        with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def callback(step, model, nll):
        e = dict(step=step, nll=nll, secs=round(time.time() - run["t0"], 1))
        for c in SHORT:
            qs = TR.questions(data, c)
            pred = TR.read_questions(model, qs, data["vocab"])[0].tolist()
            e[c] = sum(int(p) in q[1] for p, q in zip(pred, qs))
        for c in STEPS_CLS:
            counts, _ = TR.exam_steps(model, data, c)
            e.update({"%s_%s" % (c, k): v for k, v in counts.items()})
        run["exams"].append(e)
        json.dump(run["exams"], open(os.path.join(out, "exams.json"), "w"), indent=1)
        note("adim %5d  nll %.3f  1R_T %d/%d | 2R_T birebir %d/%d | 2R_UT son %d birebir %d kopru %d ozne %d /%d  (%.0f sn)" % (
                 step, nll, e["1R_T"], n["1R_T"], e["2R_T_exact"], n["2R_T"], e["2R_UT_final"], e["2R_UT_exact"],
                 e["2R_UT_bridge"], e["2R_UT_subject"], n["2R_UT"], e["secs"]))
        if run["stop"]:
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            raise Stopped()

    def job():
        try:
            ids, mask = TR.sequences(data)
            model, _ = TR.train_seq("shared", ids, mask, len(data["vocab"]), steps=steps, seed=seed, device=device,
                                    every=every, callback=callback, log_at=(), compile=compile, **train_kw)
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            final = {}
            for c in STEPS_CLS:
                for given in (0, 2, 8):
                    counts, rows = TR.exam_steps(model, data, c, given)
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
