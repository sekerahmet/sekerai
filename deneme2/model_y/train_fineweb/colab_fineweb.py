# -*- coding: utf-8 -*-
"""colab_fineweb -- FineWeb-Edu egitiminin Colab kosulari (colab_simplestories duzeni).  Egitim arka planda bir iplikte;
hucre hemen doner (kural 8).

    import data_fineweb as DF, colab_fineweb as C
    DATA = DF.load(FW_ROOT)                        butun parcalar (10B token)
    C.start(C.RUN_NAME, DATA, OUT, batch_size=B)   MODEL_KW, RECIPE, TOKENS_PER_STEP, SAVE_EVERY varsayilan; 1 epok
    C.pulse()      her kosunun durumu ve son satirlari
    C.stop()       bayrak: iplik bir sonraki sinavda modeli kaydedip cikar

Adim: batch_size satirlik micro_batches parca (gradyan birikimi, train_y.MICRO_BATCHES); micro_batches = adim basina token
hedefi / (batch_size x SEQ_LEN), coherence'ta cifte yuvarlanir.  Adim sayisi: steps, yoksa token_budget / adimin gercek
token'i, o da yoksa 1 epok (pencere / adimin satiri).  stop_at: o adimin sinavindan sonra durur (A100 TEST; ayarlar gercek
kosununki).  Paketli pencere (maskeli paketleme; model_y PACKED_ATTENTION, GPU'da flex_attention).
OUT/config.json kosu ayarlari (veri, iz, baglam, adim ve token hesabi), OUT/log.txt her sinavin satiri, OUT/exams.json
butun sinavlar (sinav alt kumesinde nll / acc / bpb ve konum bantlari, ilk PROBE_PROMPTS istemin metni, calisma), OUT/model.pt
son agirlik, OUT/final.json sonda butun valid + alt kume + uzun belgeler (exam_long, 16k) + butun istemler + valid
belgelerinin devami (gercegiyle) + tekrar olculeri, OUT/checkpoint_tNNNNNN.pt her save_every adimda surdurme paketi
(resume=True).  Calisma olculeri (step_ms, tokens_per_sec, mfu, save_secs, gpu_peak_gb) colab_simplestories'teki gibi; MFU'da
attention anahtari belge icinde sayilir (hedef t'nin gordugu konum + 1).
"""
import json
import math
import os
import sys
import threading
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train_simplestories"))
sys.path.insert(0, os.path.dirname(HERE))                           # model_y: model ve genel egitim

import colab_simplestories as CS  # noqa: E402  (MFU tablosu ve FLOP hesabi, alpha satiri)
import data_fineweb as DF  # noqa: E402
import exam_fineweb as EF  # noqa: E402
import exam_simplestories as ES  # noqa: E402
import model_y as M  # noqa: E402
import train_y as TR  # noqa: E402

# Varsayilan kosu (kullanici, 30 Eylul: "onaylıyorum", "CU satın alırım sorun değli"; not.md §9): model_y varsayilanlarinin
# uzerine, aday (3) d 1280
RUN_NAME = "fineweb_modely_6x2_d1280_gpt2_s0"
MODEL_KW = dict(d=1280, layers=6, turns=12, heads=20, fact_activation="swiglu", units=3456, shared_facts=False,
                first_turn_facts=False, input_embedding=True, input_embedding_sphere=True, input_bigrams=0, output_link=True,
                learn_output_scale=True, attention_log_scale=True, loss_chunk=4096)
RECIPE = dict(schedule="coherence", final_cooldown=0.95, final_cooldown_shape="log", lr_floor=0.0, optimizer="muon",
              weight_ema=0.999, matmul_precision="bf16")      # lr: peak_lr(d)
# A100 TEST adaylari (profile_sizes): model ayarlari disinda her sey ayni; units ~ 8/3 d, 64'un kati
CANDIDATES = (("d1024_6x2", dict(d=1024, layers=6, turns=12, heads=16, units=2752)),
              ("d1024_8x2", dict(d=1024, layers=8, turns=16, heads=16, units=2752)),
              ("d1280_6x2", dict(d=1280, layers=6, turns=12, heads=20, units=3456)))
BATCH_SIZE = 8           # parca basina satir (SEQ_LEN token); A100 TEST'le bellege gore secilir
TOKENS_PER_STEP = 64 * 8192   # adim basina token hedefi: 64 x 8.192 ~ 0,5M
SAVE_EVERY = 500
# ON KOSU (L4; kullanici, 30 Eylul: "kod gelince L4'te küçük modelle ... ~1 saatlik, ~200M token'lık bir ön koşu. ve detaylı
# analiz", baglam 4.096): uctan uca hata yakalamak -- veri, maske, sinav, surdurme, metin.  lr peak_lr(384) = 0,01
# valid_shard 1: tokenize bitmeden (shard_013 yokken) baslayabilsin; egitim yalniz shard_000'dan (200M < 0,7G)
PILOT = dict(name="fineweb_modely_3x2_d384_gpt2_pilot_s0", seq_len=4096, token_budget=200e6, tokens_per_step=16 * 4096,
             batch_size=8, valid_shard=1, model_kw=dict(d=384, layers=3, turns=6, heads=4, units=1024))
PROBE_TOKENS = 64        # her sinavda istem basina uretilen token
FINAL_TOKENS = 256       # sonda sabit istem basina (belge devami: belgenin kalani kadar)
FINAL_DOCS = 8           # sonda ilk yarisi verilen sinav belgesi (gercek devamla)
REPEAT_EVERY = 4         # continuation_repeats her 4. sinavda (adim / every % 4 == 0) ve sonda
RUNS = {}


def peak_lr(d):
    """Tepe lr: aci / adim ~ lr x 0,2 x sqrt(d) (kure + Muon) d'den bagimsiz kalsin -- d 384'te 0,01 (SimpleStories), buradan
    sqrt(384 / d) ile (d 1024: 0,0061, d 1280: 0,0055)."""
    return round(0.01 * math.sqrt(384 / d), 4)


def _counting(draw, work):
    """batches sarmalayici: cekilen her batch'in hedef token'i ve attention anahtari (hedef t kendi belgesinde konum + 1
    anahtar gorur) work'e."""
    def batch(step):
        ids, mask, pos = draw(step)
        valid = mask[:, 1:]
        work["targets"] += int(valid.sum())
        work["keys"] += int(((pos[:, :-1] + 1) * valid).sum())
        return ids, mask, pos
    return batch


def plan(data, token_budget=None, steps=None, tokens_per_step=TOKENS_PER_STEP, batch_size=BATCH_SIZE, schedule=None,
         seed=0):
    """Adim hesabi (steps; yoksa token_budget / adimin token'i; o da yoksa 1 epok) -> dict(micro_batches, rows_per_step,
    step_tokens, steps, steps_per_epoch, epochs, windows, padding); pencere ve dolgu epok 0'in paketlemesinden."""
    T = data["seq_len"]
    micro = max(1, round(tokens_per_step / (batch_size * T)))
    if (schedule or TR.SCHEDULE) == "coherence" and micro > 1 and micro % 2:
        micro += 1                                         # iki yari esit parca
    rows = batch_size * micro
    packed = DF.pack(data, seed, 0)
    per_epoch = packed["windows"] // rows
    assert per_epoch > 0, "adimin satiri (%d) > pencere (%d)" % (rows, packed["windows"])
    steps = steps if steps is not None else math.ceil(token_budget / (rows * T)) if token_budget else per_epoch
    return dict(micro_batches=micro, rows_per_step=rows, step_tokens=rows * T, steps=steps, steps_per_epoch=per_epoch,
                epochs=round(steps / per_epoch, 4), windows=packed["windows"], padding=round(packed["padding"], 6))


def _band_text(e):
    return "/".join("-" if b["accuracy"] is None else "%.3f" % b["accuracy"] for b in e["bands"])


def _bpb_text(e):
    return "/".join("-" if b["bits_per_byte"] is None else "%.3f" % b["bits_per_byte"] for b in e["bands"])


def _repeats_line(r, head):
    parts = ["%s dongu %.0f%% tekrar8 %.1f%% farkli4 %.2f" % (label, 100 * r[k]["loop"], 100 * r[k]["repeat8"],
                                                              r[k]["distinct4"])
             for k, label in (("greedy", "acgozlu"), ("sampled", "ornek"), ("real", "gercek")) if k in r]
    return "%s (%d belge): %s" % (head, next(iter(r.values()))["docs"], " | ".join(parts)) if r else head + ": belge yok"


def start(name, data, out, token_budget=None, steps=None, tokens_per_step=TOKENS_PER_STEP, batch_size=BATCH_SIZE, seed=0,
          every=500, device="cuda", compile=True, setting="shared", save_every=SAVE_EVERY, resume=False, model_kw=None,
          stop_at=None, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez.
    data: data_fineweb.load -- iz, parcalar ve baglam config'e.  model_kw: MODEL_KW'nin uzerine (bos kalanlar model_y
    varsayilanlari, t_max = SEQ_LEN; config'e acik yazilir); train_kw: RECIPE'nin uzerine (train_seq ayarlari).  Adim: plan.
    save_every: her save_every adimda checkpoint_tNNNNNN.pt.  resume=True: out'taki son paketten -- ayarlar config.json ile
    ayni olmali.  stop_at: o adimin sinavindan sonra model.pt yazilir ve durur (config'e girmez)."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    assert setting in TR.STEP3, "paketli pencere yalniz BlockModel (setting shared / separate)"
    assert "micro_batches" not in train_kw, "micro_batches plan'dan (tokens_per_step / batch_size)"
    model_kw = dict(dict(d=M.D, turns=M.TURNS, layers=M.LAYERS, shared_facts=M.SHARED_FACTS, heads=M.HEADS,
                         fact_activation=M.FACT_ACTIVATION, learn_output_scale=M.LEARN_OUTPUT_SCALE, output_link=M.OUTPUT_LINK,
                         units=M.FACT_UNITS, t_max=data["seq_len"], anchor=M.ANCHOR, loss_chunk=M.LOSS_CHUNK,
                         last_facts_alpha_init=M.LAST_FACTS_ALPHA_INIT, input_embedding=M.INPUT_EMBEDDING,
                         input_bigrams=0, first_turn_facts=M.FIRST_TURN_FACTS,
                         input_embedding_sphere=M.INPUT_EMBEDDING_SPHERE, packed_attention=M.PACKED_ATTENTION,
                         attention_log_scale=M.ATTENTION_LOG_SCALE), **dict(MODEL_KW, **(model_kw or {})))
    train_kw = dict(RECIPE, lr=peak_lr(model_kw["d"]), **train_kw)
    assert not model_kw["input_bigrams"], "paketli pencerede INPUT_BIGRAMS yok"
    steps_plan = plan(data, token_budget, steps, tokens_per_step, batch_size, train_kw.get("schedule"), seed)
    cuda = torch.device(device).type == "cuda"
    compile = compile and cuda                                 # train_seq ile ayni kural: compile yalniz GPU'da
    config = dict(name=name, setting=setting, seed=seed, every=every, device=device, compile=compile, dataset="fineweb-edu",
                  packing=DF.PACKING, tag=data["tag"], fingerprint=data["fingerprint"], shards=data["shards"],
                  seq_len=data["seq_len"], vocab=len(data["vocab"]),
                  exam_docs=len(data["exam"]), valid_docs=len(data["valid_starts"]), token_budget=token_budget,
                  tokens_per_step=tokens_per_step, batch_size=batch_size, save_every=save_every, model_kw=model_kw,
                  **steps_plan,
                  **dict(dict(lr=TR.LR, lr_floor=TR.LR_FLOOR, grad_clip=TR.GRAD_CLIP, weight_decay=TR.WEIGHT_DECAY,
                              optimizer=TR.OPTIMIZER, schedule=TR.SCHEDULE, cooldown=TR.COOLDOWN,
                              coherence_window=TR.COHERENCE_WINDOW, final_cooldown=TR.FINAL_COOLDOWN,
                              weight_ema=TR.WEIGHT_EMA, matmul_precision=TR.MATMUL_PRECISION,
                              coherence_power=TR.COHERENCE_POWER, muon_tangent=TR.MUON_TANGENT,
                              final_cooldown_shape=TR.FINAL_COOLDOWN_SHAPE, attention_kernel=TR.ATTENTION_KERNEL,
                              newton_schulz_precision=TR.NEWTON_SCHULZ_PRECISION, log_cooldown_kappa=TR.LOG_COOLDOWN_KAPPA,
                              stream_norm=TR.STREAM_NORM, layer_norm=TR.LAYER_NORM, normalized_update=TR.NORMALIZED_UPDATE,
                              sphere_weights=TR.SPHERE_WEIGHTS, canon=TR.CANON, rope=TR.ROPE), **train_kw))
    steps, per_epoch, rows = config["steps"], config["steps_per_epoch"], config["rows_per_step"]
    checkpoint = None
    if resume:
        packs = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t") and f.endswith(".pt")) if os.path.isdir(out) else []
        if not packs:
            raise RuntimeError("%s: surdurme paketi yok; bastan kosmak ayri karar (resume=False)" % out)
        saved = json.load(open(os.path.join(out, "config.json")))
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
    run = dict(name=name, out=out, lines=[], stop=False, error=None, done=False, t0=time.time(), exams=exams, mark=None)
    probes = EF.prompt_ids(data)[:EF.PROBE_PROMPTS]
    peak = CS._peak_for(torch.cuda.get_device_name(device), config["matmul_precision"]) if cuda else None
    work = dict(targets=0, keys=0, save_secs=0.0)      # son sinavdan bu yana: hedef token, attention anahtari, yedek suresi

    class Stopped(Exception):
        pass

    def note(line):
        run["lines"].append(line)
        with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def callback(step, model, nll):
        perf = {}
        if run["mark"] is not None and step > run["mark"][0]:   # sinav disi: son sinavin bitisinden bu yana
            now = time.time()
            perf["step_ms"] = round(1000 * (now - run["mark"][1]) / (step - run["mark"][0]), 1)
            secs = max(now - run["mark"][1] - work["save_secs"], 1e-9)
            perf.update(tokens_per_sec=round(work["targets"] / secs, 1), mfu=CS._mfu(run["flops"], work, secs, peak),
                        save_secs=round(work["save_secs"], 3))
        if cuda:
            perf["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if "params" not in run:
            run["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            run["body_params"] = sum(p.numel() for k, p in model.named_parameters()   # govde: token tablolari haric
                                     if p.requires_grad and not k.startswith(("tokens.", "input_")))
            run["flops"] = CS._model_flops(model)
            note("parametre %d (govde %d)  |  %s  |  FLOP: matris/token %s, attention L x d %s, GPU tepe %s" % (
                run["params"], run["body_params"], json.dumps(model_kw), *(run["flops"] or ("?", "?")),
                "%.3g" % peak if peak else "yok"))
        e = dict(step=step, epoch=round(step / per_epoch, 4), tokens=step * config["step_tokens"], train_nll=nll,
                 **EF.exam(model, data), **perf)
        alpha = ES.alpha_summary(model)
        if alpha:
            e.update(alpha_summary=alpha)
        if getattr(model, "coherence", None):
            e.update(coherence=dict(model.coherence))
        if getattr(model, "weight_ema", None):        # ortalama model: ayni sinav
            e.update(weight_ema=EF.exam(model.weight_ema["model"], data))
        if getattr(model, "learn_output_scale", False):
            e.update(output_scale=float(model.log_output_scale.detach().exp()))
        if getattr(model, "output_link", False):
            e.update(link_q=float(model.link_q.detach()), link_u=float(model.link_u.detach()))
        if (step // every) % REPEAT_EVERY == 0:
            e.update(repeats=EF.continuation_repeats(model, data))
        written = EF.texts(model, data, probes, PROBE_TOKENS)
        e.update(ES.loop_check([w["ids"] for w in written]), texts=[w["model"] for w in written],
                 secs=round(time.time() - run["t0"], 1))
        run["exams"].append(e)
        part = os.path.join(out, "exams.json.part")   # once .part, sonra yerine: yazarken olen cekirdek dosyayi bozmaz
        with open(part, "w", encoding="utf-8") as f:
            json.dump(run["exams"], f, indent=1, ensure_ascii=False)
        os.replace(part, os.path.join(out, "exams.json"))
        rare = [b for b in e["nll_by_frequency"] if b["targets"]][-1]
        note("adim %6d  epok %.3f  %.2fG token  nll %.3f%s | val nll %.3f ppl %.2f acc %.4f (%s) bpb %.4f (%s) eos %.2f ar %.3f "
             "| dongu %d/%d farkli4 %.2f  (%.0f sn)"
             % (step, e["epoch"], e["tokens"] / 1e9, nll, "" if math.isfinite(nll) else " SONLU DEGIL", e["nll"], e["ppl"],
                e["accuracy"], _band_text(e), e["bits_per_byte"], _bpb_text(e), e["eos_ok"], e["acc_ar"], e["loop"],
                len(probes), e["distinct4"], e["secs"])
             + (" | %.0f ms/adim" % e["step_ms"] if "step_ms" in e else "")
             + (" %.1fk token/sn" % (e["tokens_per_sec"] / 1000) if "tokens_per_sec" in e else "")
             + (" mfu %.1f%%" % (100 * e["mfu"]) if e.get("mfu") is not None else "")
             + (" yedek %.1f sn" % e["save_secs"] if e.get("save_secs") else "")
             + (" gpu %.2f GB" % e["gpu_peak_gb"] if "gpu_peak_gb" in e else "")
             + " | siklik %d+ x%.2f" % (rare["ranks"][0], rare["mass_ratio"])
             + (CS._alpha_text(e["alpha_summary"]) if "alpha_summary" in e else "")
             + (" | c %.3f rho %.3f ort %.3f lr %.5f" % tuple(e["coherence"][k] for k in ("c", "rho", "mean", "lr"))
                if "coherence" in e else "")
             + (" | ema ppl %.2f bpb %.4f (fark %.3f nat)" % (e["weight_ema"]["ppl"], e["weight_ema"]["bits_per_byte"],
                                                             e["nll"] - e["weight_ema"]["nll"]) if "weight_ema" in e else "")
             + (" | olcek %.2f" % e["output_scale"] if "output_scale" in e else "")
             + (" | bag q %.3f u %.3f" % (e["link_q"], e["link_u"]) if "link_q" in e else ""))
        if "repeats" in e:
            note(_repeats_line(e["repeats"], "       tekrar"))
        if run["stop"] or (stop_at is not None and step >= stop_at):
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            raise Stopped()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        run["mark"] = (step, time.time())
        work.update(targets=0, keys=0, save_secs=0.0)

    def save(step, model, opt):
        t = time.time()
        path = os.path.join(out, "checkpoint_t%06d.pt" % step)
        torch.save(dict(step=step, model=model.state_dict(), optimizer=opt.state_dict()), path + ".part")
        os.replace(path + ".part", path)             # surdurme yarim pakete dusmesin
        work["save_secs"] += time.time() - t

    def final_exam(model):
        """Sonda bir kez: butun valid, alt kume, uzun belgeler, istemler, belge devamlari, tekrar."""
        f = dict(valid=EF.exam(model, data, range(len(data["valid_starts"]))), subset=EF.exam(model, data),
                 long=EF.exam_long(model, data))
        f["prompts"] = EF.texts(model, data, EF.prompt_ids(data), FINAL_TOKENS)
        halves, reals = EF.doc_prompts(data, EF.fitting_docs(data, data["exam"])[:FINAL_DOCS])
        f["docs"] = EF.texts(model, data, halves, reals=reals)
        f["loops"] = ES.loop_check([w["ids"] for w in f["prompts"]])
        f["repeats"] = EF.continuation_repeats(model, data, real=True)
        alpha = ES.alpha_summary(model)
        if alpha:
            f["alpha_summary"] = alpha
        return f

    def final_lines(f, head):
        for k in ("valid", "subset"):
            v = f[k]
            note("%s %-6s n %d  nll %.4f ppl %.2f acc %.4f (%s)  bpb %.4f (%s, eos dahil %.4f)  eos %.3f ar %.3f" % (
                head, k, v["n"], v["nll"], v["ppl"], v["accuracy"], _band_text(v), v["bits_per_byte"], _bpb_text(v),
                v["bits_per_byte_eos"], v["eos_ok"], v["acc_ar"]))
        g = f["long"]
        note("%s uzun   %s" % (head, g["error"] if "error" in g else "%d belge (> %d token, en cok %d)  acc %s  bpb %s" % (
            g["docs"], data["seq_len"], g["max_tokens"], _band_text(g), _bpb_text(g))))
        note("%s istemler: dongu %d/%d  farkli4 %.2f" % (head, f["loops"]["loop"], len(EF.PROMPTS), f["loops"]["distinct4"]))
        note(_repeats_line(f["repeats"], "%s tekrar" % head))

    def job():
        try:
            if checkpoint is not None:
                note("SURDURULDU adim %d'den (checkpoint_t%06d.pt)" % (checkpoint["step"], checkpoint["step"]))
            else:
                note("veri fineweb-edu %s iz %s  parca %s  sozluk %d  baglam %d  pencere %d (best-fit, dolgu %%%.2f) | adim %d "
                     "x %d satir (%d parca x %d) = %d token, %d adim = %.2fG token = %.3f epok | sinav %d belge | %s  "
                     "loss_chunk %s  matmul %s  compile %s" % (
                         data["tag"], data["fingerprint"], data["shards"], len(data["vocab"]), data["seq_len"],
                         config["windows"], 100 * config["padding"], steps, rows, config["micro_batches"], batch_size,
                                     config["step_tokens"], steps, steps * config["step_tokens"] / 1e9, config["epochs"],
                                     len(data["exam"]), model_kw["packed_attention"], model_kw["loss_chunk"],
                                     config["matmul_precision"], compile))
            model, _ = TR.train_seq(setting, None, None, len(data["vocab"]), steps=steps, seed=seed, device=device,
                                    every=every, callback=callback, log_at=(), compile=compile, save_every=save_every,
                                    save=save, checkpoint=checkpoint,
                                    batches=_counting(DF.batches(data, rows, seed), work), model_kw=model_kw,
                                    **dict(train_kw, micro_batches=config["micro_batches"],
                                           matmul_precision=config["matmul_precision"],
                                           coherence_power=config["coherence_power"], muon_tangent=config["muon_tangent"],
                                           attention_kernel=config["attention_kernel"],
                                           newton_schulz_precision=config["newton_schulz_precision"]))
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            final = dict(step=steps, **final_exam(model))
            peaks = [e["gpu_peak_gb"] for e in run["exams"] if "gpu_peak_gb" in e]
            times = [e["step_ms"] for e in run["exams"] if "step_ms" in e]
            speeds = [e["tokens_per_sec"] for e in run["exams"] if "tokens_per_sec" in e]
            mfus = [e["mfu"] for e in run["exams"] if e.get("mfu") is not None]
            final["perf"] = dict(step_ms=times[-1] if times else None, gpu_peak_gb=max(peaks) if peaks else None,
                                 tokens_per_sec=speeds[-1] if speeds else None, mfu=mfus[-1] if mfus else None)
            if getattr(model, "learn_output_scale", False):
                final["output_scale"] = float(model.log_output_scale.detach().exp())
            if getattr(model, "output_link", False):
                final.update(link_q=float(model.link_q.detach()), link_u=float(model.link_u.detach()))
            if getattr(model, "weight_ema", None):    # ortalama model de ayni son sinavdan gecer
                em = model.weight_ema["model"]
                torch.save(em.state_dict(), os.path.join(out, "model_weight_ema.pt"))
                final["weight_ema"] = final_exam(em)
            json.dump(final, open(os.path.join(out, "final.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            final_lines(final, "SON")
            if "weight_ema" in final:
                final_lines(final["weight_ema"], "SON ortalama")
            note("SON calisma %s" % json.dumps(final["perf"]))
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
    """Her kosu: durum, sure ve son k satir; GPU'da o anki ve tepe bellek."""
    if not RUNS:
        print("kosu yok")
    for name, run in RUNS.items():
        state = ("KOSUYOR" if run["thread"].is_alive() else "BITTI" if run["done"] else "HATA" if run["error"]
                 else "DURDU")
        mem = (" | gpu %.2f GB (tepe %.2f)" % (torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9)
               if torch.cuda.is_available() else "")
        print("%s  %s  %.0f sn  %s%s" % (name, state, time.time() - run["t0"], run["out"], mem))
        for line in run["lines"][-k:]:
            print("   " + line)


def stop():
    """Canli kosulara bayrak; iplik bir sonraki sinavda modeli kaydedip cikar."""
    for name, run in RUNS.items():
        if run["thread"].is_alive():
            run["stop"] = True
            print("%s: durdurma bayragi kondu" % name)


def profile_sizes(data, out_root, candidates=CANDIDATES, stop_at=100, every=50, batch_size=BATCH_SIZE, token_budget=10e9,
                  cu_per_hour=5.4, max_hours=48.0, min_ratio=50.0, device="cuda", **start_kw):
    """A100 TEST, arka planda: adaylar SIRAYLA, her biri gercek kosunun ayarlariyla (MODEL_KW + aday, RECIPE, lr peak_lr(d),
    1 epokluk takvim) stop_at adima kadar, <out_root>/<ad>.  Aday basina satir: token/sn ve ms/adim (son sinav araligi,
    compile sonrasi), tepe bellek, mfu, govde parametresi (token tablolari haric), token_budget / govde, token_budget icin saat
    ve CU; kural: token / govde >= min_ratio ve saat <= max_hours.  Sonda SECIM: kurali gecen en buyuk (govde) aday.
    Ilerleme pulse(), durdurma stop()."""
    name = "A100 TEST"
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    run = dict(name=name, out=out_root, lines=[], stop=False, error=None, done=False, t0=time.time(), results=[])

    def note(line):
        run["lines"].append(line)
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "profile_sizes.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def job():
        try:
            note("kural: token / govde >= %g ve %.3gG token <= %g saat (CU %.2f / saat); %d adim, sinav her %d" % (
                min_ratio, token_budget / 1e9, max_hours, cu_per_hour, stop_at, every))
            for label, kw in candidates:
                if run["stop"]:
                    break
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                r = start("A100_TEST_" + label, data, os.path.join(out_root, label), model_kw=kw, stop_at=stop_at, every=every,
                          save_every=None, batch_size=batch_size, device=device, **start_kw)
                r["thread"].join()
                timed = [e for e in r["exams"] if "tokens_per_sec" in e]
                if r["error"] or not timed:
                    note("%-10s HATA ya da olcum yok %s" % (label, (r["error"] or "")[-400:]))
                    continue
                e = timed[-1]
                hours = token_budget / e["tokens_per_sec"] / 3600
                res = dict(label=label, model_kw=kw, tokens_per_sec=e["tokens_per_sec"], step_ms=e["step_ms"],
                           gpu_peak_gb=e.get("gpu_peak_gb"), mfu=e.get("mfu"), params=r["params"],
                           body_params=r["body_params"], ratio=token_budget / r["body_params"], hours=hours,
                           cu=hours * cu_per_hour)
                res["passes"] = res["ratio"] >= min_ratio and hours <= max_hours
                run["results"].append(res)
                note("%-10s %.1fk token/sn  %.0f ms/adim  tepe %s GB  mfu %s  govde %.1fM (token/govde %.0f)  %.3gG token: "
                     "%.1f saat, %.0f CU  %s" % (label, res["tokens_per_sec"] / 1000, res["step_ms"], res["gpu_peak_gb"],
                                                 "-" if res["mfu"] is None else "%.1f%%" % (100 * res["mfu"]),
                                                 res["body_params"] / 1e6, res["ratio"], token_budget / 1e9, hours, res["cu"],
                                                 "KURALI GECTI" if res["passes"] else "kurali gecmedi"))
            passing = [x for x in run["results"] if x["passes"]]
            best = max(passing, key=lambda x: x["body_params"]) if passing else None
            note("SECIM: %s" % (best["label"] if best else "kurali gecen aday yok"))
            with open(os.path.join(out_root, "profile_sizes.json"), "w", encoding="utf-8") as f:
                json.dump(run["results"], f, indent=1)
            run["done"] = True
        except Exception:
            run["error"] = traceback.format_exc()
            note("HATA\n" + run["error"])

    run["thread"] = threading.Thread(target=job, daemon=True)
    RUNS[name] = run
    run["thread"].start()
    return run
