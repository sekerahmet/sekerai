# -*- coding: utf-8 -*-
"""colab_simplestories -- SimpleStories egitiminin Colab kosulari (colab_tinystories duzeni).  Egitim arka planda bir
iplikte; hucre hemen doner (kural 8).

    import data_simplestories as DS, colab_simplestories as C
    DATA = DS.build(SS_DIR, TAG)                   TAG: "ss4096" | "gpt2"; sozluk buyuklugu veriden
    C.start(NAME, DATA, OUT, steps=S, seed=0, every=500, device="cuda", batch_size=64, save_every=500)
    C.pulse()      her kosunun durumu ve son satirlari
    C.stop()       bayrak: iplik bir sonraki sinavda modeli kaydedip cikar

Mini-batch: data_simplestories.batches (adimin fonksiyonu; surdurmede ayni parca; uzun hikaye bolunmus pencereler).
OUT/config.json kosu ayarlari (tag, iz), OUT/log.txt her sinavin satiri, OUT/exams.json butun sinavlar (sayilar, ilk
PROBE_PROMPTS istemin metni, calisma: step_ms ve gpu_peak_gb), OUT/model.pt son agirlik, OUT/final.json sonda butun sinav
kumesi + sabit alt kume + butun istemler + valid hikayelerinin devami (gercegiyle), OUT/checkpoint_tNNNNNN.pt her
save_every adimda surdurme paketi (resume=True).
Calisma olcumu (gpt2'de 50k sozluk; kullanici, 28 Eylul: "calisma performansina bakacagiz"): step_ms = son sinavdan bu
yana adim basina sure (sinav suresi haric; ilk aralik compile'i da icerir), gpu_peak_gb = o aralikta GPU tepe bellegi
(torch.cuda.max_memory_allocated; sinavdan sonra sifirlanir).  Kayip parcali (model_kw loss_chunk) ve matmul_precision
config'te ve ilk satirda.
Her sinavda ayrica (kullanici, 29 Eylul: "Önerilerinin hepsi kabul", "Evet"): tokens_per_sec (aralikta cekilen
batch'lerin gercek hedef token'i / egitim suresi; sinav ve yedek yazimi haric), save_secs (aralikta yedek yazma suresi),
mfu (egitim FLOP'u / sure / GPU tepesi; CPU'da ve tabloda olmayan GPU'da None), exam_train, alpha_summary,
nll_by_frequency (exam'in icinde) ve her TEXT_ERRORS_EVERY sinavda bir text_errors (count_text_errors); final.json'da
hepsi, metin hatalari gercek devamla birlikte.
"""
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # model_y: model ve genel egitim

import data_simplestories as DS  # noqa: E402
import exam_simplestories as ES  # noqa: E402
import model_y as M  # noqa: E402
import train_y as TR  # noqa: E402
from model_y_transformer import TransformerModel  # noqa: E402

BATCH_SIZE = 64          # colab_tinystories ile ayni; 1 epok = 35.382 adim (ss4096) / 35.251 (gpt2), bolunmus pencerelerle
PROBE_TOKENS = 80        # her sinavda istem basina uretilen token
FINAL_TOKENS = 300       # sonda istem basina (ortalama hikaye ~287 token)
FINAL_STORIES = 8        # sonda ilk yarisi verilen valid hikayesi (sabit alt kumenin ilk 8'i)
TEXT_ERRORS_EVERY = 4   # count_text_errors (100 hikaye x 2 cozumleme) her 4. sinavda (adim / every % 4 == 0) ve sonda
# GPU tepe matmul FLOP/s, yogun (seyrekliksiz; veri sayfasindaki "with sparsity" degerinin yarisi), matmul_precision'a gore.
# Kaynak: NVIDIA veri sayfalari (L4; A100; H100 SXM ve PCIe; T4).  Ad torch.cuda.get_device_name'den, ilk eslesen.
_PEAK_FLOPS = ((r"\bH100\b.*\bPCIe\b", dict(bf16=756e12, tf32=378e12, fp32=51e12)),
               (r"\bH100\b", dict(bf16=989e12, tf32=495e12, fp32=67e12)),
               (r"\bA100\b", dict(bf16=312e12, tf32=156e12, fp32=19.5e12)),
               (r"\bL4\b", dict(bf16=121e12, tf32=60e12, fp32=30.3e12)),
               (r"\bT4\b", dict(fp32=8.1e12)))                 # T4'te bf16 / tf32 tensor core yok
RUNS = {}


def _peak_for(name, precision):
    """GPU adi ve matmul_precision -> tepe FLOP/s; tabloda yoksa None."""
    for pattern, peaks in _PEAK_FLOPS:
        if re.search(pattern, name):
            return peaks.get(precision)
    return None


def _model_flops(model):
    """(N, L x d): token basina uygulanan matris parametresi -- tur basina (paylasilan blok her turda yeniden sayilir) +
    cikis (V x d) -- ve attention katmani x d.  Egitim FLOP'u = 6 N hedef + 12 L d anahtar (PaLM ek B,
    belge/makaleler/2022/chowdhery2022_palm.txt ~4765; anahtar: hedefin nedensel attention'da gordugu konum).  Bilinmeyen
    model None."""
    if isinstance(model, M.BlockModel):
        n = 0
        for i, b in enumerate(model.turn_blocks()):
            at = b.attention
            facts = model.extra_facts[i - model.layers] if not model.shared_facts and i >= model.layers else b.facts
            if i == 0 and not getattr(model, "first_turn_facts", True):
                facts = None                                  # FIRST_TURN_FACTS=False: tur 1'de FactUnits yok
            n += sum(w.numel() for w in (at.W_query, at.W_key, at.W_context) + ((at.W_value,) if at.heads > 1 else ()))
            n += sum(p.numel() for k, p in facts.named_parameters() if k.startswith("W_")) if facts is not None else 0
        V, d = model.tokens.fixed_points.shape
        return n + V * d, model.turns * d
    if isinstance(model, TransformerModel):
        V, d = model.embedding.weight.shape
        n = sum(m.weight.numel() for layer in model.layers
                for m in (layer.W_query, layer.W_key, layer.W_value, layer.W_out, layer.W_mlp_in, layer.W_mlp_out)
                if m is not None)
        return n + V * d, len(model.layers) * d
    return None


def _mfu(flops, work, secs, peak):
    """Araliktaki egitim FLOP'u / sure / tepe; flops ya da tepe yoksa None."""
    if not flops or not peak:
        return None
    return round((6 * flops[0] * work["targets"] + 12 * flops[1] * work["keys"]) / secs / peak, 5)


def _counting(draw, work):
    """batches sarmalayici: cekilen her batch'in hedef token'i ve attention anahtari (hedef t, t konum gorur) work'e."""
    def batch(step):
        ids, mask = draw(step)
        n = mask[:, 1:].sum(1)
        work["targets"] += int(n.sum())
        work["keys"] += int((n * (n + 1) // 2).sum())
        return ids, mask
    return batch


def _alpha_text(a):
    f = lambda xs: "/".join("-" if x is None else "%.2f" % x for x in xs)      # None: o turda alt blok yok
    top = lambda xs: max(x for x in xs if x is not None)
    return " | aA %s (en buyuk %.2f) aF %s (%.2f)" % (f(a["attention"]["median"]), top(a["attention"]["max_abs"]),
                                                     f(a["facts"]["median"]), top(a["facts"]["max_abs"]))


def _text_errors_line(t, head):
    parts = ["%s dongu %.0f%% cumle %.0f%% soz %.0f%% XaX %.0f%% kalip %.0f%% uydurma %.1f/1k" % (
        label, 100 * m["loop"], 100 * m["sentence_repeat"], 100 * m["quote_repeat"], 100 * m["xax"], 100 * m["stock"],
        m["nonword_per1k"]) for label, m in ((lb, t[k]) for k, lb in (("greedy", "acgozlu"), ("sampled", "ornek"),
                                                                         ("real", "gercek")) if k in t)]
    return "%s (%d hikaye): %s" % (head, next(iter(t.values()))["stories"], " | ".join(parts))


def start(name, data, out, steps, seed=0, every=500, device="cuda", compile=True, setting="shared", save_every=None,
          resume=False, batch_size=BATCH_SIZE, model_kw=None, bucket=None, **train_kw):
    """Egitimi arka planda baslatir, hemen doner.  out doluysa once out_eski_<zaman>'a TASINIR, silinmez.
    data: data_simplestories.build(root, tag) -- tag ve iz config'e yazilir, sozluk buyuklugu len(data["vocab"]).
    setting "shared": Model X; model_kw bos kalan ayarlar model_y varsayilanlari (config'e acik yazilir).
    compile=True (varsayilan; kullanici: "bu sabit ayar ve yes olsun").
    save_every: her save_every adimda out/checkpoint_tNNNNNN.pt {step, model, optimizer}.  resume=True: out'taki son
    paketten surdurur -- klasor tasinmaz, gunluk uzar, paketten sonraki sinavlar atilir; ayarlar config.json ile ayni olmali.
    bucket: data_simplestories.batches'e gider (None: rastgele batch; K: uzunluga gore gruplama)."""
    if name in RUNS and RUNS[name]["thread"].is_alive():
        raise RuntimeError("%s zaten kosuyor" % name)
    if setting in TR.STEP3:
        model_kw = dict(dict(d=M.D, turns=M.TURNS, layers=M.LAYERS, shared_facts=M.SHARED_FACTS, heads=M.HEADS, fact_activation=M.FACT_ACTIVATION,
                             learn_output_scale=M.LEARN_OUTPUT_SCALE, output_link=M.OUTPUT_LINK, units=M.FACT_UNITS, t_max=M.T_MAX,
                             anchor=M.ANCHOR, loss_chunk=M.LOSS_CHUNK, last_facts_alpha_init=M.LAST_FACTS_ALPHA_INIT,
                             input_embedding=M.INPUT_EMBEDDING, input_bigrams=M.INPUT_BIGRAMS,
                             first_turn_facts=M.FIRST_TURN_FACTS, input_embedding_sphere=M.INPUT_EMBEDDING_SPHERE),
                        **(model_kw or {}))
    keys = None                                       # INPUT_BIGRAMS: ikili listesi modele tensor, config'e izi
    if model_kw and model_kw.get("bigram_keys") is not None and not isinstance(model_kw["bigram_keys"], str):
        keys = torch.as_tensor(model_kw["bigram_keys"], dtype=torch.long)
        model_kw = dict(model_kw, bigram_keys="%d anahtar, sha256 %s" % (
            len(keys), hashlib.sha256(keys.cpu().numpy().tobytes()).hexdigest()[:12]))
    if model_kw and model_kw.get("input_bigrams") and not resume:
        assert keys is not None and len(keys) == model_kw["input_bigrams"], "INPUT_BIGRAMS: model_kw'de bigram_keys (liste)"
    per_epoch = len(data["train_start"]) // batch_size
    assert per_epoch > 0, "batch_size (%d) > train penceresi (%d)" % (batch_size, len(data["train_start"]))
    rows = ES.exam_rows(data)
    compile = compile and torch.device(device).type == "cuda"     # train_seq ile ayni kural: compile yalniz GPU'da
    cuda = torch.device(device).type == "cuda"
    # varsayilanlar da yazilir: config tek basina kosuyu tarif etsin
    config = dict(name=name, setting=setting, steps=steps, seed=seed, every=every, device=device, compile=compile,
                  tag=data["tag"], fingerprint=data["fingerprint"], seq_len=data["seq_len"], vocab=len(data["vocab"]),
                  train_windows=len(data["train_start"]), exam_stories=len(rows), batch_size=batch_size,
                  steps_per_epoch=per_epoch, epochs=round(steps / per_epoch, 4), save_every=save_every, model_kw=model_kw,
                  bucket=bucket,
                  **dict(dict(lr=TR.LR, lr_floor=TR.LR_FLOOR, grad_clip=TR.GRAD_CLIP, weight_decay=TR.WEIGHT_DECAY,
                              optimizer=TR.OPTIMIZER, schedule=TR.SCHEDULE, cooldown=TR.COOLDOWN,
                              coherence_window=TR.COHERENCE_WINDOW, final_cooldown=TR.FINAL_COOLDOWN,
                              weight_ema=TR.WEIGHT_EMA, matmul_precision=TR.MATMUL_PRECISION,
                              coherence_power=TR.COHERENCE_POWER, muon_tangent=TR.MUON_TANGENT,
                              final_cooldown_shape=TR.FINAL_COOLDOWN_SHAPE, attention_kernel=TR.ATTENTION_KERNEL,
                              newton_schulz_precision=TR.NEWTON_SCHULZ_PRECISION, log_cooldown_kappa=TR.LOG_COOLDOWN_KAPPA,
                              stream_norm=TR.STREAM_NORM, layer_norm=TR.LAYER_NORM,
                              normalized_update=TR.NORMALIZED_UPDATE if setting in TR.STEP3 else False,
                              sphere_weights=TR.SPHERE_WEIGHTS if setting in TR.STEP3 else False,
                              canon=TR.CANON if setting in TR.STEP3 else False,
                              rope=True if setting.startswith("transformer") else TR.ROPE if setting in TR.STEP3 else False),
                         **train_kw))
    checkpoint = None
    if resume:
        packs = sorted(f for f in os.listdir(out) if f.startswith("checkpoint_t") and f.endswith(".pt")) if os.path.isdir(out) else []
        if not packs:
            raise RuntimeError("%s: surdurme paketi yok; bastan kosmak ayri karar (resume=False)" % out)
        saved = json.load(open(os.path.join(out, "config.json")))
        saved.setdefault("final_cooldown_shape", "sqrt")          # 29 Eylul oncesi kosularda yazilmadi: sqrt idi
        saved.setdefault("log_cooldown_kappa", TR.LOG_COOLDOWN_KAPPA)   # 30 Eylul oncesi: "log" yoktu, k etkisiz
        saved.setdefault("attention_kernel", "math")              # 29 Eylul oncesi: hep math, fp32 Newton-Schulz
        saved.setdefault("newton_schulz_precision", "fp32")
        if setting in TR.STEP3 and saved.get("model_kw"):   # 29 Eylul oncesi kosularda bu ayarlar yazilmadi: yoktu
            saved["model_kw"] = dict(dict(output_link=False, shared_facts=True, input_embedding=False, input_bigrams=0,
                                          first_turn_facts=True, input_embedding_sphere=False), **saved["model_kw"])
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
    vocab = data["vocab"]
    eos = vocab.index(DS.EOS_TOKEN)
    probes = [[eos] + DS.encode(p, vocab) for p in ES.PROMPTS[:ES.PROBE_PROMPTS]]
    peak = _peak_for(torch.cuda.get_device_name(device), config["matmul_precision"]) if cuda else None
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
            secs = max(now - run["mark"][1] - work["save_secs"], 1e-9)     # egitim: yedek yazimi da haric
            perf.update(tokens_per_sec=round(work["targets"] / secs, 1), mfu=_mfu(run["flops"], work, secs, peak),
                        save_secs=round(work["save_secs"], 3))
        if cuda:
            perf["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if "params" not in run:
            run["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            run["flops"] = _model_flops(model)
            note("parametre %d  |  %s  |  FLOP: matris/token %s, attention L x d %s, GPU tepe %s" % (
                run["params"], json.dumps(model_kw), *(run["flops"] or ("?", "?")),
                "%.3g" % peak if peak else "yok"))
        e = dict(step=step, epoch=round(step / per_epoch, 4), train_nll=nll, **ES.exam(model, data, rows), **perf)
        e.update(exam_train=ES.exam_train(model, data))
        alpha = ES.alpha_summary(model)
        if alpha:
            e.update(alpha_summary=alpha)
        if getattr(model, "coherence", None):         # schedule coherence: son adimin olcumu (c, rho, ortalama, lr)
            e.update(coherence=dict(model.coherence))
        if getattr(model, "weight_ema", None):        # ortalama model: ayni sinav; fark = titresimin bedeli
            em = model.weight_ema["model"]
            e.update(weight_ema=dict(ES.exam(em, data, rows), exam_train=ES.exam_train(em, data)))
        if getattr(model, "learn_output_scale", False):   # ogrenilen cikis olcegi e^tau
            e.update(output_scale=float(model.log_output_scale.detach().exp()))
        if getattr(model, "output_link", False):          # cikis bagi phi: q, u
            e.update(link_q=float(model.link_q.detach()), link_u=float(model.link_u.detach()))
        if (step // every) % TEXT_ERRORS_EVERY == 0:
            e.update(text_errors=ES.count_text_errors(model, data, FINAL_TOKENS))
        written = ES.texts(model, data, probes, PROBE_TOKENS)
        e.update(ES.loop_check([w["ids"] for w in written]), texts=[w["model"] for w in written],
                 secs=round(time.time() - run["t0"], 1))
        run["exams"].append(e)
        part = os.path.join(out, "exams.json.part")   # once .part, sonra yerine: yazarken olen cekirdek dosyayi bozmaz
        with open(part, "w", encoding="utf-8") as f:
            json.dump(run["exams"], f, indent=1, ensure_ascii=False)
        os.replace(part, os.path.join(out, "exams.json"))
        rare = [b for b in e["nll_by_frequency"] if b["targets"]][-1]
        note("adim %6d  epok %.3f  nll %.3f%s | val nll %.3f ppl %.2f acc %.4f (%.3f/%.3f/%.3f) eos %.2f bpb %.4f "
             "(eos %.4f) | dongu %d/%d farkli4 %.2f  (%.0f sn)"
             % (step, e["epoch"], nll, "" if math.isfinite(nll) else " SONLU DEGIL", e["nll"], e["ppl"], e["accuracy"],
                e["acc_0_64"], e["acc_64_256"], e["acc_256_512"], e["eos_ok"], e["bits_per_byte"], e["bits_per_byte_eos"],
                e["loop"], len(probes), e["distinct4"], e["secs"])
             + (" | %.0f ms/adim" % e["step_ms"] if "step_ms" in e else "")
             + (" %.1fk token/sn" % (e["tokens_per_sec"] / 1000) if "tokens_per_sec" in e else "")
             + (" mfu %.1f%%" % (100 * e["mfu"]) if e.get("mfu") is not None else "")
             + (" yedek %.1f sn" % e["save_secs"] if e.get("save_secs") else "")
             + (" gpu %.2f GB" % e["gpu_peak_gb"] if "gpu_peak_gb" in e else "")
             + " | train nll %+.3f | siklik %d+ x%.2f" % (e["exam_train"]["nll"] - e["nll"], rare["ranks"][0],
                                                        rare["mass_ratio"])
             + (_alpha_text(e["alpha_summary"]) if "alpha_summary" in e else "")
             + (" | c %.3f rho %.3f ort %.3f lr %.5f" % tuple(e["coherence"][k] for k in ("c", "rho", "mean", "lr"))
                if "coherence" in e else "")
             + (" | ema ppl %.2f (fark %.3f nat)" % (e["weight_ema"]["ppl"], e["nll"] - e["weight_ema"]["nll"])
                if "weight_ema" in e else "")
             + (" | olcek %.2f" % e["output_scale"] if "output_scale" in e else "")
             + (" | bag q %.3f u %.3f" % (e["link_q"], e["link_u"]) if "link_q" in e else ""))
        if "text_errors" in e:
            note(_text_errors_line(e["text_errors"], "       metin"))
        if run["stop"]:
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

    def job():
        try:
            if checkpoint is not None:
                note("SURDURULDU adim %d'den (checkpoint_t%06d.pt)" % (checkpoint["step"], checkpoint["step"]))
            else:
                note("veri %s iz %s  sozluk %d  train %d pencere  epok = %d adim (batch %d)  %d adim = %.3f epok  sinav %d "
                     "hikaye  |  loss_chunk %s  matmul %s  compile %s" % (
                         data["tag"], data["fingerprint"], len(vocab), len(data["train_start"]), per_epoch, batch_size,
                         steps, steps / per_epoch, len(rows), (model_kw or {}).get("loss_chunk"),
                         config["matmul_precision"], compile))
            t = time.time()
            ref = ES._references(data)                # siklik bandi ve valid kalip cumleleri bir kez (bellekte)
            note("referans %.0f sn: text_reference.json %s (train kelime %d, 'X and X' cifti %d, legit %d), valid kalip "
                 "cumle %d" % (time.time() - t, ref["sha256"], ref["words"], ref["xax_pairs"], ref["xax_legit"],
                               ref["stock_sentences"]))
            model, _ = TR.train_seq(setting, None, None, len(vocab), steps=steps, seed=seed, device=device, every=every,
                                    callback=callback, log_at=(), compile=compile, save_every=save_every, save=save,
                                    checkpoint=checkpoint, batches=_counting(DS.batches(data, batch_size, seed, bucket), work),
                                    model_kw=model_kw if keys is None else dict(model_kw, bigram_keys=keys),
                                    **dict(train_kw, matmul_precision=config["matmul_precision"],
                                           coherence_power=config["coherence_power"], muon_tangent=config["muon_tangent"],
                                           attention_kernel=config["attention_kernel"],
                                           newton_schulz_precision=config["newton_schulz_precision"]))
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            final = dict(step=steps, valid=ES.exam(model, data, ES.exam_rows(data, None)), subset=ES.exam(model, data, rows),
                         exam_train=ES.exam_train(model, data))
            final["prompts"] = ES.texts(model, data, [[eos] + DS.encode(p, vocab) for p in ES.PROMPTS], FINAL_TOKENS)
            halves, reals = ES.story_prompts(data, rows[:FINAL_STORIES])
            final["stories"] = ES.texts(model, data, halves, FINAL_TOKENS, reals=reals)
            final["loops"] = ES.loop_check([w["ids"] for w in final["prompts"]])
            final["text_errors"] = ES.count_text_errors(model, data, FINAL_TOKENS, real=True)
            alpha = ES.alpha_summary(model)
            if alpha:
                final["alpha_summary"] = alpha
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
                final["weight_ema"] = dict(valid=ES.exam(em, data, ES.exam_rows(data, None)), subset=ES.exam(em, data, rows),
                                           exam_train=ES.exam_train(em, data),
                                           prompts=ES.texts(em, data, [[eos] + DS.encode(p, vocab) for p in ES.PROMPTS],
                                                            FINAL_TOKENS),
                                           stories=ES.texts(em, data, halves, FINAL_TOKENS, reals=reals),
                                           text_errors=ES.count_text_errors(em, data, FINAL_TOKENS))
                final["weight_ema"]["loops"] = ES.loop_check([w["ids"] for w in final["weight_ema"]["prompts"]])
                if alpha:
                    final["weight_ema"]["alpha_summary"] = ES.alpha_summary(em)
            json.dump(final, open(os.path.join(out, "final.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            for k in ("valid", "subset", "exam_train"):
                v = final[k]
                note("SON %-6s n %d  nll %.4f ppl %.2f acc %.4f (%.3f/%.3f/%.3f) eos %.3f ar %.3f  bpb %.4f (eos %.4f)" % (
                    k.replace("exam_", ""), v["n"], v["nll"], v["ppl"], v["accuracy"], v["acc_0_64"], v["acc_64_256"],
                    v["acc_256_512"], v["eos_ok"], v["acc_ar"], v["bits_per_byte"], v["bits_per_byte_eos"])
                    + (" | train - valid %+.4f nat" % (v["nll"] - final["valid"]["nll"]) if k == "exam_train" else ""))
            note("SON istemler: dongu %d/%d  farkli4 %.2f  |  calisma %s" % (
                final["loops"]["loop"], len(ES.PROMPTS), final["loops"]["distinct4"], json.dumps(final["perf"])))
            note(_text_errors_line(final["text_errors"], "SON metin"))
            if "weight_ema" in final:
                w_ = final["weight_ema"]
                note("SON ortalama model: valid ppl %.2f bpb %.4f acc %.4f | subset ppl %.2f | dongu %d/%d | train - valid "
                     "%+.4f nat" % (w_["valid"]["ppl"], w_["valid"]["bits_per_byte"], w_["valid"]["accuracy"],
                                    w_["subset"]["ppl"], w_["loops"]["loop"], len(ES.PROMPTS),
                                    w_["exam_train"]["nll"] - w_["valid"]["nll"]))
                note(_text_errors_line(w_["text_errors"], "SON ortalama metin"))
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
