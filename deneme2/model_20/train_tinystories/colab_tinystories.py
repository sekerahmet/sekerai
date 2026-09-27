# -*- coding: utf-8 -*-
"""colab_tinystories -- TinyStories egitiminin Colab kosulari.  Egitim arka planda bir iplikte; hucre hemen doner (kural 8).

    import colab_tinystories as C
    C.start(NAME, DATA, OUT, steps=S, seed=0, every=500, device="cuda", batch_size=64, save_every=500)
    C.pulse()      her kosunun durumu ve son satirlari
    C.stop()       bayrak: iplik bir sonraki sinavda modeli kaydedip cikar

Mini-batch: data_tinystories.batches (adimin fonksiyonu; surdurmede ayni parca).  OUT/config.json kosu ayarlari,
OUT/log.txt her sinavin satiri, OUT/exams.json butun sinavlar (sayilar + ilk PROBE_PROMPTS istemin metni),
OUT/model.pt son agirlik, OUT/final.json sonda butun valid (sizintisiz) + sabit alt kume + butun istemler + valid
hikayelerinin devami (gercegiyle), OUT/checkpoint_tNNNNNN.pt her save_every adimda surdurme paketi (resume=True).
"""
import json
import os
import sys
import threading
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_20: model ve genel egitim

import data_tinystories as DT  # noqa: E402
import exam_tinystories as ET  # noqa: E402
import model_20 as M  # noqa: E402
import train_20 as TR  # noqa: E402

BATCH_SIZE = 64          # model_18 v4 ile ayni: 1 epok 41.602 adim; logits fp32 64 x 511 x 8.004 = 1,05 GB
PROBE_TOKENS = 80        # her sinavda istem basina uretilen token
FINAL_TOKENS = 200       # sonda istem basina (ortalama hikaye 194 token)
FINAL_STORIES = 8        # sonda ilk yarisi verilen valid hikayesi (sabit alt kumenin ilk 8'i)
RUNS = {}


def start(name, data, out, steps, seed=0, every=500, device="cuda", compile=True, setting="shared", save_every=None,
          resume=False, batch_size=BATCH_SIZE, model_kw=None, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez.
    setting "shared": Model X; model_kw bos kalan ayarlar model_20 varsayilanlari (config'e acik yazilir).
    compile=True (varsayilan; kullanici: "bu sabit ayar ve yes olsun").
    save_every: her save_every adimda out/checkpoint_tNNNNNN.pt {step, model, optimizer}.  resume=True: out'taki son
    paketten surdurur -- klasor tasinmaz, gunluk uzar, paketten sonraki sinavlar atilir; ayarlar config.json ile ayni olmali."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    if setting in TR.STEP3:
        model_kw = dict(dict(d=M.D, turns=M.TURNS, units=M.FACT_UNITS, t_max=M.T_MAX, anchor=M.ANCHOR), **(model_kw or {}))
    per_epoch = len(data["train_start"]) // batch_size
    assert per_epoch > 0, "batch_size (%d) > train penceresi (%d)" % (batch_size, len(data["train_start"]))
    rows = ET.exam_rows(data)
    # varsayilanlar da yazilir: config tek basina kosuyu tarif etsin
    compile = compile and torch.device(device).type == "cuda"     # train_seq ile ayni kural: compile yalniz GPU'da
    config = dict(name=name, setting=setting, steps=steps, seed=seed, every=every, device=device, compile=compile,
                  fingerprint=data["fingerprint"], seq_len=data["seq_len"], vocab=len(data["vocab"]),
                  train_windows=len(data["train_start"]), exam_stories=len(rows), batch_size=batch_size,
                  steps_per_epoch=per_epoch, epochs=round(steps / per_epoch, 4), save_every=save_every, model_kw=model_kw,
                  **dict(dict(lr=TR.LR, lr_floor=TR.LR_FLOOR, grad_clip=TR.GRAD_CLIP, weight_decay=TR.WEIGHT_DECAY,
                              optimizer=TR.OPTIMIZER, schedule=TR.SCHEDULE, cooldown=TR.COOLDOWN,
                              copy_path=TR.COPY_PATH, stream_norm=TR.STREAM_NORM, layer_norm=TR.LAYER_NORM,
                              rope=True if setting.startswith("transformer") else TR.ROPE if setting in TR.STEP3 else False),
                         **train_kw))
    checkpoint = None
    if resume:
        packs = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t")) if os.path.isdir(out) else []
        if not packs:
            raise RuntimeError("%s: surdurme paketi yok; bastan kosmak ayri karar (resume=False)" % out)
        saved = json.load(open(os.path.join(out, "config.json")))
        # 27 Eylul oncesi config'lerde optimizer / takvim yok: o kosular Adam + cosine idi.  compile sonucu degistirir: karsilastirilir
        saved = dict(dict(optimizer="adam", schedule="cosine", cooldown=TR.COOLDOWN), **saved)
        differ = sorted(k for k in set(saved) | set(config) if k != "device" and saved.get(k) != config.get(k))
        if differ:
            raise RuntimeError("surdurme: ayarlar config.json'dan farkli %s -- ayni ayarlarla surdurulur" % differ)
        checkpoint = torch.load(os.path.join(out, packs[-1]), map_location=device)
    else:
        if os.path.isdir(out) and os.listdir(out):
            os.rename(out, out + "_eski_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out, exist_ok=True)
        json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
    exams = []
    if resume:                                         # paketten sonraki sinavlar yeniden kosulacak: cift satir olmasin
        exams = [e for e in json.load(open(os.path.join(out, "exams.json"), encoding="utf-8"))
                 if e["step"] <= checkpoint["step"]]
    run = dict(name=name, out=out, lines=[], stop=False, error=None, done=False, t0=time.time(), exams=exams)
    vocab = data["vocab"]
    eos = vocab.index(DT.EOS_TOKEN)
    probes = [[eos] + DT.encode(p, vocab) for p in ET.PROMPTS[:ET.PROBE_PROMPTS]]

    class Stopped(Exception):
        pass

    def note(line):
        run["lines"].append(line)
        with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def callback(step, model, nll):
        if "params" not in run:
            run["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            note("parametre %d  |  %s" % (run["params"], json.dumps(model_kw)))
        e = dict(step=step, epoch=round(step / per_epoch, 4), train_nll=nll, **ET.exam(model, data, rows))
        written = ET.texts(model, data, probes, PROBE_TOKENS)
        e.update(ET.loop_check([w["ids"] for w in written]), texts=[w["model"] for w in written],
                 secs=round(time.time() - run["t0"], 1))
        run["exams"].append(e)
        json.dump(run["exams"], open(os.path.join(out, "exams.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        note("adim %6d  epok %.3f  nll %.3f | val nll %.3f ppl %.2f acc %.4f (%.3f/%.3f/%.3f) eos %.2f | dongu %d/%d "
             "farkli4 %.2f  (%.0f sn)" % (step, e["epoch"], nll, e["nll"], e["ppl"], e["accuracy"], e["acc_0_64"],
                                          e["acc_64_256"], e["acc_256_512"], e["eos_ok"], e["loop"], len(probes),
                                          e["distinct4"], e["secs"]))
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
            else:
                note("veri iz %s  train %d pencere  epok = %d adim (batch %d)  %d adim = %.3f epok  sinav %d hikaye" % (
                    data["fingerprint"], len(data["train_start"]), per_epoch, batch_size, steps, steps / per_epoch,
                    len(rows)))
            model, _ = TR.train_seq(setting, None, None, len(vocab), steps=steps, seed=seed, device=device, every=every,
                                    callback=callback, log_at=(), compile=compile, save_every=save_every, save=save,
                                    checkpoint=checkpoint, batches=DT.batches(data, batch_size, seed), model_kw=model_kw,
                                    **train_kw)
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            final = dict(step=steps, valid=ET.exam(model, data, ET.exam_rows(data, None)), subset=ET.exam(model, data, rows))
            final["prompts"] = ET.texts(model, data, [[eos] + DT.encode(p, vocab) for p in ET.PROMPTS], FINAL_TOKENS)
            halves, reals = ET.story_prompts(data, rows[:FINAL_STORIES])
            final["stories"] = ET.texts(model, data, halves, FINAL_TOKENS, reals=reals)
            final["loops"] = ET.loop_check([w["ids"] for w in final["prompts"]])
            json.dump(final, open(os.path.join(out, "final.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            for k in ("valid", "subset"):
                v = final[k]
                note("SON %-6s n %d  nll %.4f ppl %.2f acc %.4f (%.3f/%.3f/%.3f) eos %.3f ar %.3f" % (
                    k, v["n"], v["nll"], v["ppl"], v["accuracy"], v["acc_0_64"], v["acc_64_256"], v["acc_256_512"],
                    v["eos_ok"], v["acc_ar"]))
            note("SON istemler: dongu %d/%d  farkli4 %.2f" % (final["loops"]["loop"], len(ET.PROMPTS),
                                                            final["loops"]["distinct4"]))
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
