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
kosununki).  Paketli pencere (maskeli paketleme; GPU'da flex_attention).
OUT/config.json kosu ayarlari (veri, iz, baglam, adim ve token hesabi), OUT/log.txt her sinavin satiri, OUT/exams.json
butun sinavlar (sinav alt kumesinde nll / acc / bpb ve konum bantlari, ilk PROBE_PROMPTS istemin metni, calisma), OUT/model.pt
son agirlik, OUT/final.json sonda butun valid + alt kume + uzun belgeler (exam_long, 16k) + butun istemler + valid
belgelerinin devami (gercegiyle) + tekrar olculeri, OUT/checkpoint_tNNNNNN.pt her save_every adimda surdurme paketi
(resume=True).  Calisma olculeri (step_ms, tokens_per_sec, mfu, save_secs, gpu_peak_gb) colab_simplestories'teki gibi; MFU'da
attention anahtari belge icinde sayilir (hedef t'nin gordugu konum + 1).
"""
import functools
import json
import math
import os
import sys
import threading
import time
import traceback

import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train_simplestories"))
sys.path.insert(0, os.path.dirname(HERE))                           # model_y: model ve genel egitim

import colab_simplestories as CS  # noqa: E402  (MFU tablosu ve FLOP hesabi, alpha satiri)
import data_fineweb as DF  # noqa: E402
import exam_fineweb as EF  # noqa: E402
import exam_simplestories as ES  # noqa: E402
import model_y as M  # noqa: E402
import train_y as TR  # noqa: E402

# Varsayilan kosu: ana kosu (d1024_8x2); geri kalan model ayarlari model_y varsayilanlari
RUN_NAME = "fineweb_modely_8x2_d1024_gpt2_s0"
MODEL_KW = dict(d=1024, layers=8, turns=16, heads=8, units=2752)
RECIPE = dict(schedule="coherence", final_cooldown=0.95, final_cooldown_shape="log", lr_floor=0.0,
              weight_ema=0.999, matmul_precision="bf16")      # lr: peak_lr(d)
# A100 TEST adaylari (profile_sizes): model ayarlari disinda her sey ayni; units ~ 8/3 d, 64'un kati; head boyu 128 (RoPE
# tabani 8.192'de 64'luk head'de tirtikli, 128'likte duzgun)
# batch_size: parca basina satir (model_kw'ye girmez); 8 satirda d1024_8x2 egitim adimi 80 GB'i asar
CANDIDATES = (("d1024_6x2", dict(d=1024, layers=6, turns=12, heads=8, units=2752)),
              ("d1024_8x2", dict(d=1024, layers=8, turns=16, heads=8, units=2752, batch_size=4)),
              ("d1280_6x2", dict(d=1280, layers=6, turns=12, heads=10, units=3456, batch_size=4)))
BATCH_SIZE = 4           # parca basina satir (SEQ_LEN token): ana kosununki; A100 TEST'le bellege gore secilir
TOKENS_PER_STEP = 64 * 8192   # adim basina token hedefi: 64 x 8.192 ~ 0,5M
SAVE_EVERY = 500
PROBE_TOKENS = "auto"    # her sinavda istem basina uretilen token; "auto" = EF.typical_doc_tokens (valid medyan belge).
                         # Config'e girmez: log'da
FINAL_TOKENS = "auto"    # sonda sabit istem basina, ayni kural; belge devami: belgenin kalani kadar
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
    parts = ["%s dongu %.0f%% tekrar8 %.1f%% farkli4 %.2f%s" % (
        label, 100 * r[k]["loop"], 100 * r[k]["repeat8"], r[k]["distinct4"],
        " nadir %.1f%%" % (100 * r[k]["token_bands"][-1]) if "token_bands" in r[k] else "")
             for k, label in (("greedy", "acgozlu"), ("sampled", "ornek"), ("real", "gercek")) if k in r]
    return "%s (%d belge): %s" % (head, next(iter(r.values()))["docs"], " | ".join(parts)) if r else head + ": belge yok"


def _copy_line(c, head):
    """distant_copy: uzaklik basina kopya accuracy'si (ilk gecisteki taban parantezde)."""
    rows = " ".join("%d %.3f (%.3f)" % (r["distance"], r["accuracy"], r["first_accuracy"]) for r in c["rows"])
    return "%s (%d parca x %d token, uzaklik kopya (taban)): %s%s" % (head, c["passages"], c["tokens"], rows,
                                                                       "  " + c["error"] if "error" in c else "")


def start(name, data, out, token_budget=None, steps=None, tokens_per_step=TOKENS_PER_STEP, batch_size=BATCH_SIZE, seed=0,
          every=500, device="cuda", compile=True, setting="shared", save_every=SAVE_EVERY, resume=False, model_kw=None,
          stop_at=None, init_from=None, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez.
    data: data_fineweb.load -- iz, parcalar ve baglam config'e.  model_kw: MODEL_KW'nin uzerine (bos kalanlar model_y
    varsayilanlari, t_max = SEQ_LEN; config'e acik yazilir); train_kw: RECIPE'nin uzerine (train_seq ayarlari).  Adim: plan.
    save_every: her save_every adimda checkpoint_tNNNNNN.pt.  resume=True: out'taki son paketten -- ayarlar config.json ile
    ayni olmali.  stop_at: o adimin sinavindan sonra model.pt (ve model_weight_ema.pt) yazilir ve durur (config'e girmez).
    init_from: baska kosunun checkpoint_tNNNNNN.pt'si -- model ayarlari onun config'inden (eski config'te yazilmamis anahtar o
    gunun degeriyle, internals_y._model_kw: rope_base 10.000), agirlik ve optimizer oradan; EMA ve coherence bastan, adim 0
    (0. adim sinavi kosar).  Veri izi kaynaginkiyle ayniysa pencereler kaynagin son adimindan sonra (gormedigi veri).
    train_kw ditto_x_weight > 0: DITTO-X kendi ciktisinda (train_y); butun DITTO-X ayarlari config'e acik yazilir,
    self_prompts = adimin (yetmezse sonrakilerin) pencerelerinden belge basindan P + G token'lik satirlar, sentence_ends =
    exam_fineweb._sentences_of (D_032 / O14 cumle tanimi); sinavda ditto_x_self ozeti."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    assert setting == "shared", "paketli pencere yalniz BlockModel (setting shared)"
    assert "micro_batches" not in train_kw, "micro_batches plan'dan (tokens_per_step / batch_size)"
    source = None
    if init_from is not None:                          # model ayarlari kaynak kosunun config'inden
        import internals_y as I
        assert not model_kw, "init_from: model ayarlari kaynak kosunun config'inden gelir"
        source = json.load(open(os.path.join(os.path.dirname(init_from), "config.json")))
        assert source["seq_len"] == data["seq_len"], "init_from: baglam farkli (%d / %d)" % (source["seq_len"], data["seq_len"])
        model_kw = I._model_kw(source)
    model_kw = dict(dict(d=M.D, turns=M.TURNS, layers=M.LAYERS, shared_facts=M.SHARED_FACTS, heads=M.HEADS,
                         output_link=M.OUTPUT_LINK, units=M.FACT_UNITS, t_max=data["seq_len"], anchor=M.ANCHOR, loss_chunk=M.LOSS_CHUNK,
                         last_facts_alpha_init=M.LAST_FACTS_ALPHA_INIT, input_embedding=M.INPUT_EMBEDDING,
                         first_turn_facts=M.FIRST_TURN_FACTS,
                         input_embedding_sphere=M.INPUT_EMBEDDING_SPHERE,
                         attention_log_scale=M.ATTENTION_LOG_SCALE, rope_base=M.ROPE_BASE),
                    **dict(MODEL_KW, **(model_kw or {})))
    if model_kw.get("rope_base") == "auto":      # config'e SAYI yazilir: formul sonra degisse de kosu ayni tabanla kurulur
        model_kw["rope_base"] = M.rope_base_for(model_kw["d"] // model_kw["heads"], model_kw["t_max"])
    train_kw = dict(dict(RECIPE, lr=peak_lr(model_kw["d"])), **train_kw)   # lr verilirse onunki (devam egitimi)
    ditto_x = train_kw.get("ditto_x_weight", TR.DITTO_X_WEIGHT) > 0
    if ditto_x:                                        # butun DITTO-X ayarlari acik (varsayilan sonra degisse de kosu ayni)
        for k in ("ditto_x_weight", "ditto_x_share", "ditto_x_self_prefix", "ditto_x_self_tokens", "ditto_x_self_every",
                  "ditto_x_margin", "ditto_x_params", "copy_ceiling"):
            train_kw.setdefault(k, getattr(TR, k.upper()))
    steps_plan = plan(data, token_budget, steps, tokens_per_step, batch_size, train_kw.get("schedule"), seed)
    cuda = torch.device(device).type == "cuda"
    compile = compile and cuda                                 # train_seq ile ayni kural: compile yalniz GPU'da
    config = dict(name=name, setting=setting, seed=seed, every=every, device=device, compile=compile, dataset="fineweb-edu",
                  packing=DF.PACKING, tag=data["tag"], fingerprint=data["fingerprint"], shards=data["shards"],
                  seq_len=data["seq_len"], vocab=len(data["vocab"]),
                  exam_docs=len(data["exam"]), valid_docs=len(data["valid_starts"]), token_budget=token_budget,
                  tokens_per_step=tokens_per_step, batch_size=batch_size, save_every=save_every,
                  model_kw=dict(model_kw, packed_attention="flex", fact_activation="swiglu",   # kaldirilan secenekler
                                learn_output_scale=True, input_bigrams=0),   # config'te sabit degerle: surdurme
                  **steps_plan,
                  **dict(dict(lr=TR.LR, lr_floor=TR.LR_FLOOR, grad_clip=TR.GRAD_CLIP, weight_decay=0.0,
                              optimizer="muon", schedule=TR.SCHEDULE, cooldown=TR.COOLDOWN,
                              coherence_window=TR.COHERENCE_WINDOW, final_cooldown=TR.FINAL_COOLDOWN,
                              weight_ema=TR.WEIGHT_EMA, matmul_precision=TR.MATMUL_PRECISION,
                              coherence_power=1.0, muon_tangent=True,
                              final_cooldown_shape=TR.FINAL_COOLDOWN_SHAPE, attention_kernel="math",
                              newton_schulz_precision=TR.NEWTON_SCHULZ_PRECISION, log_cooldown_kappa=TR.LOG_COOLDOWN_KAPPA,
                              stream_norm=True, layer_norm=False, normalized_update=True, sphere_weights=True, canon=True,
                              rope=TR.ROPE), **train_kw))
    if init_from is not None:
        config["init_from"] = init_from
    config = json.loads(json.dumps(config))            # surdurmede json'dan okunanla ayni bicim (demet -> liste)
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
        if init_from is not None:                      # adim 0 = kaynagin agirligi; EMA ve coherence bastan
            pack = torch.load(init_from, map_location=device)
            for state in pack["optimizer"]["state"].values():
                state.pop("weight_ema", None)
            for group in pack["optimizer"]["param_groups"]:
                group.update(coherence_mean=1.0, coherence_frozen=None)
                for k in ("coherence_signal", "coherence_total", "ditto_x_self"):
                    group.pop(k, None)
            checkpoint = dict(step=0, model=pack["model"], optimizer=pack["optimizer"])
            del pack
    offset = 0
    if source is not None and source["fingerprint"] == data["fingerprint"]:   # ayni veri: kaynagin gormedigi pencereler
        assert source["rows_per_step"] == config["rows_per_step"] and source["seed"] == seed, \
            "init_from ayni veride: adim basina satir ve tohum kaynaginkiyle ayni olmali"
        offset = source["steps"] + 1                   # kaynak son adimin penceresini yalniz ileri hesapta gordu
    draw = DF.batches(data, rows, seed)
    extra = {}
    if ditto_x:
        P, G, B, eot = (train_kw["ditto_x_self_prefix"], train_kw["ditto_x_self_tokens"], train_kw["ditto_x_share"],
                        data["eot"])

        def self_prompts(step):
            """Adimin (yetmezse sonraki adimlarin) pencerelerinden belge basindan ([eot] ile) P + G token'lik B satir."""
            found, s = [], step
            while len(found) < B:
                ids, mask, pos = draw(offset + s)
                for r, a in (pos == 0).nonzero().tolist():
                    if (ids[r, a] == eot and a + P + G <= ids.shape[1] and pos[r, a + P + G - 1] == P + G - 1
                            and bool(mask[r, a + 1:a + P + G].all())):
                        found.append(ids[r, a:a + P + G])
                s += 1
                assert s - step < 64, "self_prompts: 64 adimda %d belge yok" % B
            return torch.stack(found[:B])
        extra = dict(self_prompts=self_prompts, eot=eot,
                     sentence_ends=functools.partial(EF._sentences_of, is_end=EF._sentence_end_table(data["vocab"]),
                                                     vocab=data["vocab"]))
    exams = []
    if resume:                                         # paketten sonraki sinavlar yeniden kosulacak: cift satir olmasin
        exams = [e for e in json.load(open(os.path.join(out, "exams.json"), encoding="utf-8"))
                 if e["step"] <= checkpoint["step"]]
    run = dict(name=name, out=out, lines=[], stop=False, error=None, done=False, t0=time.time(), exams=exams, mark=None)
    probes = EF.prompt_ids(data)[:EF.PROBE_PROMPTS]
    probe_tokens, final_tokens = (EF.typical_doc_tokens(data) if x == "auto" else x for x in (PROBE_TOKENS, FINAL_TOKENS))
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
        records = getattr(model, "ditto_x_self", None)
        if records:                                    # DITTO-X kendi adimlari, son sinavdan bu yana
            band = EF._frequency_band(data)
            mean = lambda k: sum(r[k] for r in records) / len(records)
            agree = [sum(r["agreement"][i] for r in records) for i in (0, 1)]
            agg = dict(steps=len(records), beta=records[-1]["beta"], ratio=mean("ratio"), clip=mean("clip"), loss=mean("loss"),
                       gen_secs=mean("gen_secs"), capped=sum(r["capped"] for r in records),
                       agreement=agree[0] / max(agree[1], 1))
            for side, bins_of in (("own", lambda r: r["bins"]), ("real", lambda r: r["real"]["bins"])):
                sums = {k: [sum(bins_of(r)[k][i] for r in records) for i in range(3)] for k in bins_of(records[0])}
                agg[side] = dict(bins={k: dict(n=n, rate=c / n if n else None, selected=k_)
                                       for k, (n, c, k_) in sums.items()})
            for side in ("own", "real"):
                by_l = {k: [sum(r[side]["by_l"][k][i] for r in records) for i in range(4)] for k in records[0][side]["by_l"]}
                entry = [sum(r[side]["entry"][i] for r in records) for i in range(4)]
                ids = np.concatenate([r[side]["ids"].ravel() for r in records])
                agg[side].update(by_l={k: dict(n=n, p_copy=p / n if n else None, top1=t / n if n else None,
                                               copy_rate=c / n if n else None) for k, (n, p, t, c) in by_l.items()},
                                 entry=dict(rate=entry[0] / max(entry[1], 1), rest=entry[2] / entry[3] if entry[3] else None),
                                 token_bands=(np.bincount(band[ids], minlength=len(EF.FREQUENCY_BANDS) + 1)
                                              / max(len(ids), 1)).tolist())
            e["ditto_x_self"] = agg
            records.clear()
        if getattr(model, "weight_ema", None):        # ortalama model: ayni sinav
            e.update(weight_ema=EF.exam(model.weight_ema["model"], data))
        if getattr(model, "learn_output_scale", False):
            e.update(output_scale=float(model.log_output_scale.detach().exp()))
        if getattr(model, "output_link", False):
            e.update(link_q=float(model.link_q.detach()), link_u=float(model.link_u.detach()))
        if (step // every) % REPEAT_EVERY == 0:
            e.update(repeats=EF.continuation_repeats(model, data), distant_copy=EF.distant_copy(model, data))
        written = EF.texts(model, data, probes, probe_tokens)
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
        if "distant_copy" in e:
            note(_copy_line(e["distant_copy"], "       uzak kopya"))
        if "ditto_x_self" in e:                        # cumle duzeyi once (kullanici, 2 Ekim: "papağan gibi tekrarlaması")
            d = e["ditto_x_self"]
            fmt = lambda v, f="%.2f": "-" if v is None else f % v
            note("       ditto-x (%d kendi adimi, beta %.4g, oran %.2f, kesilen %d, kirpma %.2f, uyum %.3f, uretim %.1f sn): "
                 "cumle tekrarina giris kendi %s / gercek %s, kalan pay %s / %s | sinir secimi %s (n %d, ceza %d; gercek %s) | "
                 "l kovasi secimi %s | gercekte p(kopya)/top1 %s | nadir (16384+) kendi %.1f%% gercek %.1f%%" % (
                     d["steps"], d["beta"], d["ratio"], d["capped"], d["clip"], d["agreement"], d["gen_secs"],
                     fmt(d["own"]["entry"]["rate"]), fmt(d["real"]["entry"]["rate"]), fmt(d["own"]["entry"]["rest"]),
                     fmt(d["real"]["entry"]["rest"]), fmt(d["own"]["bins"]["sentence"]["rate"], "%.3f"),
                     d["own"]["bins"]["sentence"]["n"], d["own"]["bins"]["sentence"]["selected"],
                     fmt(d["real"]["bins"]["sentence"]["rate"], "%.3f"),
                     " ".join("%s %s" % (k, fmt(b["rate"], "%.3f")) for k, b in d["own"]["bins"].items() if k != "sentence"),
                     " ".join("%s %s/%s" % (k, fmt(b["p_copy"], "%.3f"), fmt(b["top1"], "%.3f"))
                              for k, b in d["real"]["by_l"].items()),
                     100 * d["own"]["token_bands"][-1], 100 * d["real"]["token_bands"][-1]))
        if run["stop"] or (stop_at is not None and step >= stop_at):
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            if getattr(model, "weight_ema", None):
                torch.save(model.weight_ema["model"].state_dict(), os.path.join(out, "model_weight_ema.pt"))
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

    def final_exam(model, full=True):
        """Sonda bir kez: butun valid (full), alt kume, uzun belgeler, istemler, belge devamlari, tekrar, uzak kopya.  Ortalama
        modelde full=False (kullanici, 30 Eylul: "sınav kısaltması ok"): alt kume +-0,005 nat yetiyor, butun valid sureyi ikiye
        katliyordu (on kosu: son sinav 16 dk)."""
        f = dict(subset=EF.exam(model, data), long=EF.exam_long(model, data))
        if full:
            f["valid"] = EF.exam(model, data, range(len(data["valid_starts"])))
        f["prompts"] = EF.texts(model, data, EF.prompt_ids(data), final_tokens)
        halves, reals = EF.doc_prompts(data, EF.fitting_docs(data, data["exam"])[:FINAL_DOCS])
        f["docs"] = EF.texts(model, data, halves, reals=reals)
        f["loops"] = ES.loop_check([w["ids"] for w in f["prompts"]])
        f["repeats"] = EF.continuation_repeats(model, data, real=True)
        f["distant_copy"] = EF.distant_copy(model, data, context=2 * data["seq_len"])   # egitilmemis uzakliga kadar
        f["long_write"] = EF.long_write(model, data)
        alpha = ES.alpha_summary(model)
        if alpha:
            f["alpha_summary"] = alpha
        return f

    def final_lines(f, head):
        for k in [k for k in ("valid", "subset") if k in f]:
            v = f[k]
            note("%s %-6s n %d  nll %.4f ppl %.2f acc %.4f (%s)  bpb %.4f (%s, eos dahil %.4f)  eos %.3f ar %.3f" % (
                head, k, v["n"], v["nll"], v["ppl"], v["accuracy"], _band_text(v), v["bits_per_byte"], _bpb_text(v),
                v["bits_per_byte_eos"], v["eos_ok"], v["acc_ar"]))
        g = f["long"]
        note("%s uzun   %s" % (head, g["error"] if "error" in g else "%d belge (> %d token, en cok %d)  acc %s  bpb %s" % (
            g["docs"], data["seq_len"], g["max_tokens"], _band_text(g), _bpb_text(g))))
        note("%s istemler: dongu %d/%d  farkli4 %.2f" % (head, f["loops"]["loop"], len(EF.PROMPTS), f["loops"]["distinct4"]))
        note(_repeats_line(f["repeats"], "%s tekrar" % head))
        note(_copy_line(f["distant_copy"], "%s uzak kopya" % head))
        w = f["long_write"]
        note("%s uzun yazim (%d token, eot yasak): eot en olasi ilk %s, ilk tekrar eden 8'li %s | dilim farkli4 %s | tekrar8 %s"
             % (head, w["tokens"], w["eot_first"], w["loop_first"], "/".join("%.2f" % s["distinct4"] for s in w["segments"]),
                "/".join("%.2f" % s["repeat8"] for s in w["segments"])))

    def job():
        try:
            note("uretim: sinav istemi %d token, son istem %d token (PROBE_TOKENS %s, FINAL_TOKENS %s; auto = valid medyan belge)"
                 % (probe_tokens, final_tokens, PROBE_TOKENS, FINAL_TOKENS))
            if checkpoint is not None and checkpoint["step"] > 0:
                note("SURDURULDU adim %d'den (checkpoint_t%06d.pt)" % (checkpoint["step"], checkpoint["step"]))
            else:
                if init_from is not None:
                    note("BASLANGIC %s (adim 0; EMA ve coherence bastan) | veri penceresi kaynagin %d. adimindan" % (
                        init_from, offset))
                note("veri fineweb-edu %s iz %s  parca %s  sozluk %d  baglam %d  pencere %d (best-fit, dolgu %%%.2f) | adim %d "
                     "x %d satir (%d parca x %d) = %d token, %d adim = %.2fG token = %.3f epok | sinav %d belge | flex  "
                     "loss_chunk %s  matmul %s  compile %s" % (
                         data["tag"], data["fingerprint"], data["shards"], len(data["vocab"]), data["seq_len"],
                         config["windows"], 100 * config["padding"], steps, rows, config["micro_batches"], batch_size,
                                     config["step_tokens"], steps, steps * config["step_tokens"] / 1e9, config["epochs"],
                                     len(data["exam"]), model_kw["loss_chunk"],
                                     config["matmul_precision"], compile))
            model, _ = TR.train_seq(setting, None, None, len(data["vocab"]), steps=steps, seed=seed, device=device,
                                    every=every, callback=callback, log_at=(), compile=compile, save_every=save_every,
                                    save=save, checkpoint=checkpoint,
                                    batches=_counting(lambda step: draw(offset + step), work), model_kw=model_kw,
                                    **extra, **dict(train_kw, micro_batches=config["micro_batches"],
                                           matmul_precision=config["matmul_precision"],
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
                final["weight_ema"] = final_exam(em, full=False)
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
    """GPU TEST (ad GPU'dan: A100 TEST, G4 TEST ...), arka planda: adaylar SIRAYLA, her biri gercek kosunun ayarlariyla (MODEL_KW + aday, RECIPE, lr peak_lr(d),
    1 epokluk takvim) stop_at adima kadar, <out_root>/<ad>.  Aday basina satir: token/sn ve ms/adim (son sinav araligi,
    compile sonrasi), tepe bellek, mfu, govde parametresi (token tablolari haric), token_budget / govde, token_budget icin saat
    ve CU; kural: token / govde >= min_ratio ve saat <= max_hours.  Sonda SECIM: kurali gecen en buyuk (govde) aday.
    Ilerleme pulse(), durdurma stop()."""
    gpu = torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else "CPU"
    tag = next((t for k, t in (("H100", "H100"), ("A100", "A100"), ("RTX PRO 6000", "G4"), ("L4", "L4")) if k in gpu), "GPU")
    name = "%s TEST" % tag
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
                kw = dict(kw)
                rows = kw.pop("batch_size", batch_size)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                r = start("%s_TEST_%s" % (tag, label), data, os.path.join(out_root, label), model_kw=kw, stop_at=stop_at, every=every,
                          save_every=None, batch_size=rows, device=device, **start_kw)
                r["thread"].join()
                timed = [e for e in r["exams"] if "tokens_per_sec" in e]
                if r["error"] or not timed:
                    note("%-10s HATA ya da olcum yok %s" % (label, (r["error"] or "")[-400:]))
                    continue
                e = timed[-1]
                hours = token_budget / e["tokens_per_sec"] / 3600
                res = dict(label=label, model_kw=kw, batch_size=rows, tokens_per_sec=e["tokens_per_sec"], step_ms=e["step_ms"],
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
