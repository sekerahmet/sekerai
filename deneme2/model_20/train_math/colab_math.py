# -*- coding: utf-8 -*-
"""colab_math -- toplama egitiminin (model_15 verisi) Colab kosulari.  Egitim arka planda bir iplikte; hucre hemen doner
(CLAUDE.md kural 8).  Duzen colab_kinship / colab_tinystories ile ayni.

    import colab_math as C
    C.start(NAME, DATA, OUT, steps=S, seed=0, device="cuda", setting="shared", model_kw=dict(d=64, units=256))
    C.pulse()      her kosunun durumu ve son satirlari
    C.stop()       bayrak: iplik bir sonraki sinavda modeli kaydedip cikar

setting "shared" = Model X; "transformer_novalue" / "transformer" = kiyas transformer'i (model_20_transformer); ayni veri,
batch'ler, sinav ve tarif (train_20.train_seq).  OUT/config.json kosu ayarlari, OUT/log.txt her sinavin satiri,
OUT/exams.json butun sinavlar, OUT/model.pt son agirlik, OUT/final.json sonda butun egitim ve tutulan sorularla olcum
(terim ve hane kirilimi, kayit.txt gibi), saglik, tani sorulari ve modelin yazdiklari; OUT/checkpoint_tNNNNNN.pt her
save_every adimda surdurme paketi (resume=True).
"""
import json
import os
import sys
import threading
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # model_20: model ve genel egitim

import data_math as DM  # noqa: E402
import exam_math as EM  # noqa: E402
import model_20 as M  # noqa: E402
import model_20_transformer as MT  # noqa: E402
import train_20 as TR  # noqa: E402

BATCH_SIZE = 256         # kullanici karari (27 Eylul)
EVERY = 5000             # sinav ve yedek; kullanici: "her 5.000 adımda ölçme ve yedek olsun kesilirse kaldığı yerden devam eder"
EXAM_LIMIT = 20000       # sinavin sabit ornegi (tohum 12345), model_15'in gunlugundeki egri gibi
ROWS = 300               # final.json'da tutulan ornegin ilk 300 sorusu: modelin yazdigi (kural 12)
RUNS = {}


def _total(parts):
    """Kirilimdan butun kumenin sonucu (n agirlikli)."""
    n = sum(p["n"] for p in parts.values())
    return dict({k: sum(p[k] * p["n"] for p in parts.values()) / n for k in ("accuracy", "length_ok", "first_digit")}, n=n)


def start(name, data, out, steps, seed=0, every=EVERY, device="cuda", compile=True, setting="shared", save_every=EVERY,
          resume=False, batch_size=BATCH_SIZE, answer_only=True, exam_limit=EXAM_LIMIT, model_kw=None, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez.
    answer_only: kayip yalniz cevap rakamlarinda ve son <eos>'ta (mask H); False: butun gercek token'lar (mask M).
    model_kw bos kalan ayarlar modelin varsayilanlari (config'e acik yazilir).  resume=True: out'taki son paketten
    surdurur -- klasor tasinmaz, gunluk uzar, paketten sonraki sinavlar atilir; ayarlar config.json ile ayni olmali."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    if setting in TR.STEP3:
        model_kw = dict(dict(d=M.D, turns=M.TURNS, layers=M.LAYERS, heads=M.HEADS, output_skip=M.OUTPUT_SKIP,
                             units=M.FACT_UNITS, t_max=M.T_MAX),
                        **(model_kw or {}))
    elif setting.startswith("transformer"):
        model_kw = dict(dict(d=MT.D, layers=MT.LAYERS, heads=MT.HEADS, units=MT.FACT_UNITS), **(model_kw or {}))
    per = len(data["train"]) // batch_size
    assert per > 0, "batch_size (%d) > egitim sorusu (%d)" % (batch_size, len(data["train"]))
    # varsayilanlar da yazilir: config tek basina kosuyu tarif etsin
    compile = compile and torch.device(device).type == "cuda"     # train_seq ile ayni kural: compile yalniz GPU'da
    config = dict(name=name, setting=setting, steps=steps, seed=seed, every=every, device=device, compile=compile,
                  data=data["name"], fingerprint=data["fingerprint"], width=data["width"], train=len(data["train"]),
                  heldout=len(data["heldout"]), batch_size=batch_size, steps_per_epoch=per, epochs=round(steps / per, 4),
                  answer_only=answer_only, exam_limit=exam_limit, save_every=save_every, model_kw=model_kw,
                  **dict(dict(lr=TR.LR, lr_floor=TR.LR_FLOOR, grad_clip=TR.GRAD_CLIP, weight_decay=TR.WEIGHT_DECAY,
                              optimizer=TR.OPTIMIZER, schedule=TR.SCHEDULE, cooldown=TR.COOLDOWN,
                              stream_norm=TR.STREAM_NORM, layer_norm=TR.LAYER_NORM,
                              normalized_update=TR.NORMALIZED_UPDATE if setting in TR.STEP3 else False,
                              sphere_weights=TR.SPHERE_WEIGHTS if setting in TR.STEP3 else False,
                              canon=TR.CANON if setting in TR.STEP3 else False,
                              rope=True if setting.startswith("transformer") else TR.ROPE if setting in TR.STEP3 else False),
                         **train_kw))
    config = json.loads(json.dumps(config))
    checkpoint, exams = None, []
    if resume:
        packs = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t")) if os.path.isdir(out) else []
        if not packs:
            raise RuntimeError("%s: surdurme paketi yok; bastan kosmak ayri karar (resume=False)" % out)
        saved = json.load(open(os.path.join(out, "config.json")))
        assert not saved.pop("copy_path", False), "kopya yolu (Oneri A) 28 Eylul'de kaldirildi: bu kosu surdurulemez"
        # 27 Eylul oncesi config'lerde optimizer / takvim yok: o kosular Adam + cosine idi.  compile sonucu degistirir: karsilastirilir
        saved = dict(dict(optimizer="adam", schedule="cosine", cooldown=TR.COOLDOWN,
                          normalized_update=False, sphere_weights=False, canon=False), **saved)
        if setting in TR.STEP3 and saved.get("model_kw"):   # 28 Eylul oncesi model_kw'de layers yok: tek Block idi
            saved["model_kw"] = dict(dict(layers=1, heads=1, output_skip=False), **saved["model_kw"])   # eski: yoklar
        differ = sorted(k for k in set(saved) | set(config) if k != "device" and saved.get(k) != config.get(k))
        if differ:
            raise RuntimeError("surdurme: ayarlar config.json'dan farkli %s -- ayni ayarlarla surdurulur" % differ)
        checkpoint = torch.load(os.path.join(out, packs[-1]), map_location=device)
        path = os.path.join(out, "exams.json")
        exams = [e for e in json.load(open(path)) if e["step"] <= checkpoint["step"]] if os.path.exists(path) else []
    else:
        if os.path.isdir(out) and os.listdir(out):
            os.rename(out, out + "_eski_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out, exist_ok=True)
        json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
    run = dict(name=name, out=out, lines=[], stop=False, error=None, done=False, t0=time.time(), exams=exams)
    heldout_sample = EM._sample(data["heldout"], exam_limit)
    heldout_windows = DM.make_windows(heldout_sample, data["width"])

    class Stopped(Exception):
        pass

    def note(line):
        run["lines"].append(line)
        with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def callback(step, model, nll):
        if "params" not in run:
            run["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            note("%s  parametre %d  1 epok = %d adim  genislik %d" % (setting, run["params"], per, data["width"]))
        e = dict(step=step, epoch=round(step / per, 3), nll=nll, secs=round(time.time() - run["t0"], 1),
                 train=EM.ask(model, data["train"], device, exam_limit), heldout=EM.ask(model, data["heldout"], device, exam_limit),
                 heldout_terms={str(k): v for k, v in EM.breakdown(model, heldout_sample, device, "terms").items()},
                 heldout_ce=EM.answer_ce(model, heldout_windows, device), panel=EM.show(model, EM.PANEL, device))
        run["exams"].append(e)
        json.dump(run["exams"], open(os.path.join(out, "exams.json"), "w"), indent=1)
        note("adim %6d  epok %6.2f  nll %.4f  egitim %.4f  tutulan %.4f (%s)  uzunluk %.4f  ilk rakam %.4f  ce %.4f  (%.0f sn)" % (
            step, e["epoch"], nll, e["train"]["accuracy"], e["heldout"]["accuracy"],
            " ".join("%s terim %.4f" % (k, v["accuracy"]) for k, v in e["heldout_terms"].items()),
            e["heldout"]["length_ok"], e["heldout"]["first_digit"], e["heldout_ce"], e["secs"]))
        if run["stop"]:
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            raise Stopped()

    def save(step, model, opt):
        torch.save(dict(step=step, model=model.state_dict(), optimizer=opt.state_dict()),
                   os.path.join(out, "checkpoint_t%06d.pt" % step))

    def job():
        try:
            if checkpoint is not None:
                note("SURDURULDU adim %d'den (checkpoint_t%06d.pt)" % (checkpoint["step"], checkpoint["step"]))
            model, _ = TR.train_seq(setting, None, None, DM.N, steps=steps, seed=seed, device=device, every=every,
                                    callback=callback, log_at=(), compile=compile, save_every=save_every, save=save,
                                    checkpoint=checkpoint, model_kw=model_kw,
                                    batches=DM.batches(data, batch_size, seed, answer_only), **train_kw)
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            final = dict(health=EM.ask(model, data["train"], device, exam_limit), breakdown={})
            for side in ("train", "heldout"):              # butun kume, kayit.txt'deki gibi terim ve hane kirilimi
                final["breakdown"][side] = {m: {str(k): v for k, v in EM.breakdown(model, data[side], device, m).items()}
                                            for m in ("terms", "digits")}
                final[side] = _total(final["breakdown"][side]["terms"])
            final["panel"] = EM.show(model, EM.PANEL, device)
            final["rows"] = EM.show(model, heldout_sample[:ROWS], device)
            json.dump(final, open(os.path.join(out, "final.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            note("SAGLIK  egitim ornegi (%d) %.4f  %s (esik %.2f)" % (
                final["health"]["n"], final["health"]["accuracy"],
                "GECTI" if final["health"]["accuracy"] >= EM.HEALTH else "KALDI", EM.HEALTH))
            for side in ("train", "heldout"):
                note("SON %-8s %.4f  uzunluk %.4f  ilk rakam %.4f  n %d   terim: %s   hane: %s" % (
                    side, final[side]["accuracy"], final[side]["length_ok"], final[side]["first_digit"], final[side]["n"],
                    " ".join("%s %.4f" % (k, v["accuracy"]) for k, v in final["breakdown"][side]["terms"].items()),
                    " ".join("%s %.4f" % (k, v["accuracy"]) for k, v in final["breakdown"][side]["digits"].items())))
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
