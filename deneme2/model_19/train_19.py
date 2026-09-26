# -*- coding: utf-8 -*-
"""train_19 -- model_19'un egitim dongusu: optimizer, batch, kayit, SURDURME, her tam yedekte decompose.

GOREV SONRAKI TOKEN, kayip YALNIZ hedeflerde:  loss = model.loss(tokens, targets_mask).
data = (questions, filled_mask, targets_mask): questions (n, T) diziler, filled_mask gercek token'lar, targets_mask
kayba sayilan hedefler (hikayede her gercek token, matematikte cevap ve EOS).

Model kurucu ayarlari (model_kw) paketin "config" alanina model.config() olarak yazilir; surdurmede config birebir
ayni olmali.  Adam, weight decay YOK (model_18'den; kullanici, 24 Eylul: "weight decay sanki anlamsiz gerek yok").
weight_decay > 0 (egitim denemesi L1): AdamW, decay YALNIZ hareket (start/finish) ve attention matrislerinde.
KOSU ARKA PLANDA (kural 8): `start` hemen doner, ilerleme `show_log` ile OKUNUR; her kosu kendi gunluk.txt'sine EKLER.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import threading
import time
import traceback

import torch

from looped_19 import LoopedRelation
from model_19 import PointRelation

ARCHS = {PointRelation.arch: PointRelation, LoopedRelation.arch: LoopedRelation}     # model_kw'de arch; verilmezse point_relation

LR = 0.002
BATCH = 64
RUNS = {}
MUST_MATCH = ("arch", "config", "T", "batch", "lr", "data_fingerprint", "vocab", "weight_decay")   # surdurmede ayni
DECAY = re.compile(r"^(moves\.\d+\.(start|finish)|attns\.\d+\.W_(q|k|v|o|qc)"      # L1: agirlik buyumesi burada olculdu
                   r"|lookups\.\d+\.(start|finish)|find\.W_(q|k|v|o))$")             # LoopedRelation'da ayni roller
SHAPES = 8           # trim: boylar bu kadar basamaga yuvarlanir (torch.compile her batch'te yeniden derlemesin)
HEALTH_PROBE = 64    # olcut saglik vermiyorsa (matematik) model.health bu kadar egitim penceresinde olculur
# Parca basina gradyan normu, olcum adimlarinda (ilk eslesen onek): olu parca ve patlama gorunsun.
GRAD_PARTS = (("moves.", "hareket"), ("lam_w", "icerik"), ("beta_w", "icerik"), ("attns.0.", "attn1"),
              ("attns.", "attn2"), ("ledger.relation.", "R_CC"), ("ledger.", "defter"), ("readout.", "R_PC"),
              ("embed_", "E"), ("find.", "find"), ("lookups.", "lookup"), ("source_weights", "kaynak"), ("alpha", "alpha"),
              ("gate_", "gate"))


def trim(questions, filled_mask, targets_mask, shapes=SHAPES):
    """CPU'daki batch -> (tokens, targets_mask, rows): en uzun diziye kirpilmis; rows sayilan hedeflerin duz
    konumlari (B*(T-1)), -1 ile yuvarlanmis.  GPU'ya dokunmaz."""
    B, T = questions.shape
    step = max(1, T // shapes)
    real = (filled_mask.bool() | targets_mask.bool()).any(0).nonzero()
    L = min(T, -(-(int(real.max()) + 1 if len(real) else 1) // step) * step)
    tokens, mask = questions[:, :L], targets_mask[:, :L]
    rows = mask[:, 1:].bool().reshape(-1).nonzero()[:, 0]
    unit = max(1, B * T // shapes ** 2)
    R = -(-max(1, len(rows)) // unit) * unit
    return tokens, mask, torch.cat([rows, rows.new_full((R - len(rows),), -1)])


class Run:
    """Bir kosunun durumu: gunluk, sonuc, iplik, dur istegi, kayit."""

    def __init__(self, name, root=None):
        self.name, self.root = name, root
        self.log, self._pending = [], []
        self.result, self.thread, self.stop_requested = {}, None, False

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    @property
    def folder(self):
        return f"{self.root}/{self.name}"

    def note(self, line):
        line = f"[{self.name}] {line}"
        self.log.append(line)
        self._pending.append(line)

    def flush(self):
        """Yeni satirlari gunluk.txt'ye EKLER."""
        if self._pending and self.root:
            os.makedirs(self.folder, exist_ok=True)
            with open(f"{self.folder}/gunluk.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(self._pending) + "\n")
        self._pending.clear()

    def save(self, package, full=True):
        """full: t<adim>.pt surdurmeye yeten TAM yedek (+ model_<ad>.pt son hal); degilse w<adim>.pt yalniz agirlik."""
        if not self.root:
            return
        os.makedirs(self.folder, exist_ok=True)
        torch.save(package, f"{self.folder}/{'t' if full else 'w'}{package['step']}.pt")
        if full:
            torch.save(package, f"{self.root}/model_{self.name}.pt")

    def archive_old(self):
        """Ayni adla yeni kosu eskiyi EZMEZ, yan klasore TASIR."""
        if not self.root or not os.path.isdir(self.folder) or not os.listdir(self.folder):
            return None
        new = f"{self.folder}_eski_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(self.folder, new)
        last = f"{self.root}/model_{self.name}.pt"
        if os.path.exists(last):
            shutil.move(last, f"{new}/model_{self.name}.pt")
        return os.path.basename(new)

    def stop(self):
        self.stop_requested = True


def _no_metric(model, side, full=False):
    raise RuntimeError("metric verilmedi -- start(..., metric=...) sart")


def _ppl(ce):
    return "      -" if ce is None else f"{math.exp(ce):7.2f}"


def _fingerprint(*tensors):
    h = hashlib.sha256()
    for x in tensors:
        h.update(f"{tuple(x.shape)} {x.dtype}".encode())
        h.update(x.contiguous().numpy())
    return h.hexdigest()[:16]


def _snapshot(model, optimizer, sampler, record):
    """TAM anlik goruntu: agirlik + optimizer + RNG."""
    return dict(record, weights=model.state_dict(), opt=optimizer.state_dict(),
                rng=torch.get_rng_state(), sampler_rng=sampler.get_state())


def lr_at(step, lr, steps, warmup=0, decay_start=None, decay_floor=0.1):
    """Adimin LR'si: ilk `warmup` adimda dogrusal isinma (lr (adim+1)/warmup), sonra sabit; decay_start'tan steps'e
    cosine sogutma lr -> lr x decay_floor."""
    w = min(1.0, (step + 1) / warmup) if warmup else 1.0
    if decay_start is None:
        return lr * w
    k = min(1.0, max(0.0, (step - decay_start) / max(1, steps - decay_start)))
    return lr * w * (decay_floor + (1 - decay_floor) * 0.5 * (1 + math.cos(math.pi * k)))


def _check_resume(package, fixed):
    """Surdurme paketi bu cagriyla AYNI kosu mu?  Degilse kosu BASLAMAZ.  Isinma birebir ayni olmali.  LR plani:
    planin LR'ye dokundugu paket (adim > decay_start) ayni planla surer; dokunmadigindan yeni planla dal acilabilir."""
    diffs = [a for a in MUST_MATCH if package.get(a) is not None and fixed.get(a) is not None and package[a] != fixed[a]]
    if package.get("warmup", 0) != fixed.get("warmup", 0):
        diffs.append("warmup")
    if diffs:
        raise ValueError("surdurme paketi bu cagriyla uyusmuyor: " + ", ".join(diffs))
    plan = lambda k: (k.get("decay_start"), k.get("decay_end"), k.get("decay_floor"))
    started = package.get("decay_start") is not None and package.get("step", 0) > package["decay_start"]
    if started and plan(package) != plan(fixed):
        raise ValueError("LR sogutma plani (baslangic, son, taban): paket %s, cagri %s" % (plan(package), plan(fixed)))


def _grad_norms(model):
    top = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            part = next((label for prefix, label in GRAD_PARTS if name.startswith(prefix)), "olcek")
            top[part] = top.get(part, 0) + p.grad.detach().float().pow(2).sum()
    return {part: float(v.sqrt()) for part, v in top.items()}


def _header(record):
    return ("  step   epok     loss    train    heldout      ppl  "
            + "".join(f"{k:>{_width(k)}}" for k in record["heldout_diag"]) + "       s")


def _width(key):
    """Sutun genisligi: en az 12, uzun ad (cikarim_gorulmemis) bir bosluk birakir -- yapismasin."""
    return max(12, len(key) + 1)


def _line(record, elapsed, mark=""):
    s = record.get("eval_sec")
    epoch = record["step"] * record["batch"] / record["windows"] if record.get("windows") else float("nan")
    return (f"{record['step']:6d}  {epoch:5.2f}  {record['loss']:7.3f}  {record['train_acc']:7.4f}  "
            f"{record['heldout_acc']:9.4f}  {_ppl(record['heldout_ce'])}  "
            + "".join(f"{v:{_width(k)}.4f}" for k, v in record["heldout_diag"].items())
            + f"  {elapsed:6.0f}{mark}" + (f"  olcum {s['train'] + s['heldout']:.1f} sn" if s else "")
            + (f"  lr {record['lr_now']:.2e}" if record.get("decay_start") is not None or record.get("warmup") else ""))


def _health_line(h, grads=None, timing=None):
    """Saglik sozlugu -> tek satir: hareketler, boy (Oe15), olcekler, defter (Oe3, Oe9), attention, yeni parcalarin
    boyu (Oe6), cikis, gradyan, dongu, olcum suresi."""
    if "passes" in h:
        return _health_line_looped(h, grads, timing)
    L = sum(1 for k in h if k.startswith("vec_all_"))
    mean = lambda x: sum(x) / len(x)
    s = ["saglik  hareket " + " ".join("%.0f/%.0f" % (h["vec_each_%d" % i], h["vec_all_%d" % i]) for i in range(L)),
         "olu " + "/".join(str(h["dead_%d" % i]) for i in range(L)),
         "yon C %d C_m %d" % (h["dir_C"], h["dir_Cm"]),
         "boy C %.2f C_m %.2f girdi %s" % (h["norm_C"], h["norm_Cm"], "/".join("%.1f" % h["norm_in_%d" % i] for i in range(L))),
         "e^S_p %.1f lam_c %.2f" % (h["exp_S_p"], h["lam_c"])]
    if "gate" in h:
        s.append("gate %.3f (yon %+.2f benz %+.2f) komsu %.1f" % (h["gate"], h["gate_dir"], h["gate_sim"], h["ledger_eff"]))
    k = 0
    while "attn%d_ent" % k in h:
        s.append("attn%d ent %.2f bas0 %.2f mesafe %.1f%s" % (
            k + 1, mean(h["attn%d_ent" % k]), mean(h["attn%d_first" % k]), mean(h["attn%d_dist" % k]),
            " m " + "/".join("%+.3f" % x for x in h["attn%d_m" % k]) if "attn%d_m" % k in h else ""))
        k += 1
    new = [("embed_shift", "|dE|"), ("readout", "|R_PC|"), ("chain_sim", "|R_CC|"), ("induction_q", "|W_qc|")]
    if any(a in h for a, _ in new):
        s.append("yeni " + " ".join("%s %.3f" % (lab, h[a]) for a, lab in new if a in h))
    s.append("cikis %d kelime ent %.2f" % (h["pred_distinct"], h["pred_ent"]))
    if grads:
        s.append("grad " + " ".join("%s %.2g" % kv for kv in sorted(grads.items())))
    if "distinct4" in h:
        s.append("farkli4 %.2f dongu %.2f" % (h["distinct4"], h["loop"]))
    if timing:
        extras = " ".join("%s %.1f" % (x, timing[x]) for x in ("health", "loop") if x in timing)
        s.append("olcum sn train %.1f heldout %.1f%s" % (timing["train"], timing["heldout"], " (%s)" % extras if extras else ""))
    return "  |  ".join(s)


def _health_line_looped(h, grads=None, timing=None):
    """LoopedRelation saglik satiri: gecis (durma kuraliyla), gecis basina degisim ve boy, lookup kullanimi, find,
    kaynaklar."""
    K = sum(1 for k in h if k.startswith("change_"))
    s = ["saglik  gecis %.2f (K_MAX'a varan %%%.0f)" % (h["passes"], 100 * h["passes_at_max"]),
         "degisim " + "/".join("%.3f" % h["change_%d" % k] for k in range(K)),
         "boy C %.2f x %s" % (h["norm_C"], "/".join("%.1f" % h["norm_x_%d" % k] for k in range(K))),
         "lookup " + " ".join("%.0f/%.0f olu %d" % (h["vec_each_%d" % i], h["vec_all_%d" % i], h["dead_%d" % i])
                         for i in range(2)),
         "find ent " + "/".join("%.2f" % h["attn_ent_%d" % k] for k in range(K))
         + " mesafe " + "/".join("%.1f" % h["attn_dist_%d" % k] for k in range(K))
         + " m " + "/".join("%+.3f" % v for v in h["attn_m"]),
         "ayni yon %.2f" % h["same_dir"] + ("  alpha " + "/".join("%.3f" % v for v in h["alpha"]) if "alpha" in h else ""),
         "kaynak a %+.3f b %+.3f  defter komsu %.1f e^S_c %.1f" % (h["src_a"], h["src_b"], h["ledger_eff"], h["exp_S_c"]),
         "e^S_p %.1f  |dE| %.3f |R_PC| %.2f |R_CC| %.2f" % (h["exp_S_p"], h["embed_shift"], h["readout"], h["chain_sim"])]
    if "gate" in h:
        s.append("gate %.3f" % h["gate"])
    s.append("cikis %d kelime ent %.2f" % (h["pred_distinct"], h["pred_ent"]))
    if grads:
        s.append("grad " + " ".join("%s %.2g" % kv for kv in sorted(grads.items())))
    if timing:
        s.append("olcum sn train %.1f heldout %.1f" % (timing["train"], timing["heldout"]))
    return "  |  ".join(s)


def _note(run, record, elapsed, mark=""):
    run.note(_line(record, elapsed, mark))
    if record.get("health") is not None:
        run.note(_health_line(record["health"], record.get("grads"), record.get("eval_sec")))


def _optimizer(model, lr, weight_decay):
    """weight_decay 0: Adam, butun parametreler (zemin).  > 0: AdamW; decay YALNIZ DECAY'e uyanlarda -- kelime basina
    tablolar (lam_w, beta_w), olcekler (S_*) ve gate disarida: seyrek gradyanli tablolar sifira cekilmesin."""
    if not weight_decay:
        return torch.optim.Adam(model.parameters(), lr=lr)
    named = list(model.named_parameters())
    decay = [p for n, p in named if DECAY.match(n)]
    rest = [p for n, p in named if not DECAY.match(n)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}],
                             lr=lr)


def _run(run, data, n_vocab, metric, device, lr, steps, seed, batch, eval_every, save_every, weights_every=100,
         resume=None, compile=False, vocab=None, extra=None, warmup=0, decay_start=None, decay_floor=0.1, decompose=True,
         weight_decay=0.0, model_kw=None):
    torch.manual_seed(seed)
    kw = dict(model_kw or {})
    model = ARCHS[kw.pop("arch", PointRelation.arch)](n_vocab, seed=seed, **kw).to(device)
    loss_fn = torch.compile(model.loss) if compile else model.loss
    if compile:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 2 * SHAPES)
        run.note("torch.compile ACIK -- ilk adim DERLEME yuzunden yavas")
    optimizer = _optimizer(model, lr, weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(seed)

    questions, filled_mask, targets_mask = (x.cpu() for x in data)
    n_questions, max_length = questions.shape
    assert max_length <= model.t_max, f"dizi {max_length} token > t_max {model.t_max}"
    n_params = sum(p.numel() for p in model.parameters())
    fixed = dict(extra or {}, arch=model.arch, config=model.config(), n=n_vocab, lr=lr, seed=seed, batch=batch,
                 windows=int(n_questions),
                 T=max_length, n_params=n_params, compile=compile, warmup=int(warmup), weight_decay=float(weight_decay),
                 decay_start=None if decay_start is None else int(decay_start),
                 decay_end=None if decay_start is None else int(steps),
                 decay_floor=None if decay_start is None else float(decay_floor),
                 data_fingerprint=_fingerprint(questions, filled_mask, targets_mask),
                 vocab=None if vocab is None else [str(a) for a in vocab])

    last_step, prev_step_losses, loss = -1, None, None
    if resume:
        package = torch.load(resume, weights_only=False, map_location=device)
        _check_resume(package, fixed)
        fixed["vocab"] = fixed["vocab"] or package.get("vocab")
        model.load_state_dict(package["weights"])
        optimizer.load_state_dict(package["opt"])
        torch.set_rng_state(package["rng"].cpu())
        sampler.set_state(package["sampler_rng"].cpu())
        last_step, prev_step_losses = package["step"], package.get("step_losses")
        loss = torch.tensor(package["loss"])
        run.note(f"SURDURULUYOR  {os.path.basename(resume)}  adim {last_step}")
    last_saved_step = last_step

    step_losses = torch.full((max(steps, last_step) + 1,), float("nan"), device=device)
    if prev_step_losses is not None:
        step_losses[:len(prev_step_losses)] = prev_step_losses.to(device)

    # olcut saglik vermiyorsa (matematik) model.health sabit bir egitim sondasinda: Oe6 ve Oe15 olculeri her kolda
    probe = trim(questions[:HEALTH_PROBE], filled_mask[:HEALTH_PROBE], filled_mask[:HEALTH_PROBE])[:2]
    fill_ratio = float(filled_mask.sum()) / filled_mask.numel()
    run.note(f"sorular {n_questions:,} x {max_length}   dolgu %{100 * (1 - fill_ratio):.1f}")
    cfg = fixed["config"]
    new_parts = [k for k in ("embed", "readout", "chain_sim", "distance") if cfg.get(k)]
    if len(cfg.get("attn_after", ())) > 1:
        new_parts.append("ikinci attention")
    if fixed["config"].get("induction_query"):
        new_parts.append("induction sorgusu")
    if fixed["config"].get("gate_sim") is False:
        new_parts.append("gate'te benzerlik YOK")
    if fixed["config"].get("norm_inputs"):
        new_parts.append("girdi kureye (Oe15)")
    if model.arch != PointRelation.arch:
        new_parts = ["mimari " + model.arch]
    if weight_decay:
        run.note(f"AdamW  weight decay {weight_decay} YALNIZ hareket ve attention matrislerinde (DECAY)")
    run.note(f"arch {model.arch}  config {fixed['config']}  yeni parcalar: {', '.join(new_parts) or 'YOK (zemin)'}  "
             f"parametre {n_params:,}")
    run.note(f"batch {batch}   epok = {n_questions / batch:,.0f} adim   veri izi {fixed['data_fingerprint']}   olcum her "
             f"{eval_every}   tam yedek her {save_every}   agirlik her {weights_every}   lr {lr}"
             + (f"   ISINMA {warmup} adim" if warmup else "")
             + (f"   LR SOGUTMA cosine {decay_start} -> {steps}, taban lr x {decay_floor}" if decay_start is not None else ""))
    if "trial" in fixed:     # egitim ve model denemeleri AYRI izlenir (kullanici, 25 Eylul)
        run.note(f"DENEME {str(fixed['trial']).upper()}   degisen {fixed.get('changed')}")
    run.note("OLCUT: accuracy (train / heldout); kayip yalniz hedeflerde")
    run.flush()
    decomposed = set()

    def decompose_hook(at):
        if decompose and at not in decomposed:
            decomposed.add(at)
            _decompose_backup(run, model, fixed["vocab"], metric, at)
            run.flush()

    def evaluate(step, full=False, save=False, grads=None, health=None):
        kw = {"save": save, "health": health} if getattr(metric, "health", False) else {}
        t0 = time.time()
        train_result = metric(model, "train", full=full)
        t1 = time.time()
        heldout_result = metric(model, "heldout", full=full, **kw)
        timing = {"train": t1 - t0, "heldout": time.time() - t1}
        timing.update((k, v) for k, v in (heldout_result.get("timing") or {}).items() if k != "eval")
        h = heldout_result.get("health")
        if h is None:
            h = health if health is not None else model.health(probe[0].to(device).long(), probe[1].to(device))
        record = dict(fixed, step=step, loss=float(loss.detach()), step_losses=step_losses[:step + 1].cpu(),
                      train_acc=train_result["accuracy"], heldout_acc=heldout_result["accuracy"],
                      train_ce=train_result.get("ce"), heldout_ce=heldout_result.get("ce"),
                      train_diag=train_result.get("diag", {}), heldout_diag=heldout_result.get("diag", {}),
                      health=h, grads=grads, eval_sec=timing, lr_now=optimizer.param_groups[0]["lr"])
        if not run.result:
            run.note(_header(record))
        run.result = dict(record, model=model)
        return record

    started_at, record = time.time(), None
    for step in range(last_step + 1, steps + 1):
        if run.stop_requested:
            break
        batch_ids = torch.randint(0, n_questions, (batch,), generator=sampler)
        optimizer.zero_grad()
        tokens, mask, rows = trim(questions[batch_ids], filled_mask[batch_ids], targets_mask[batch_ids])
        total, loss = loss_fn(tokens.to(device, non_blocking=True).long(), mask.to(device, non_blocking=True),
                              parts=True, rows=rows.to(device, non_blocking=True))
        step_losses[step] = loss.detach()
        total.backward()
        full_save, weights_save = step % save_every == 0, step % weights_every == 0
        measure = step % eval_every == 0 or full_save or weights_save
        grads = _grad_norms(model) if measure else None
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, lr, steps, warmup, decay_start, decay_floor)
        optimizer.step()
        last_step = step
        if measure:
            record, mark = evaluate(step, save=full_save, grads=grads), ""
            if full_save:
                run.save(_snapshot(model, optimizer, sampler, record))
                last_saved_step, mark = step, "  yedek"
            elif weights_save:
                run.save(dict(record, weights=model.state_dict()), full=False)
            _note(run, record, time.time() - started_at, mark)
            run.flush()
            if full_save:
                decompose_hook(step)

    if last_step < 0:
        run.note("DURDURULDU -- hic adim atilmadi, kayit yok")
        run.flush()
        return
    _finish(run, model, optimizer, sampler, evaluate, record, last_step, last_saved_step, started_at, decompose_hook)


def _finish(run, model, optimizer, sampler, evaluate, record, last_step, last_saved_step, started_at, decompose_hook):
    """ONCE KAYDET, SONRA OLC: tam olcum uzun surebilir."""
    if last_saved_step != last_step:
        noted = record is not None and record["step"] == last_step
        if not noted:
            record = evaluate(last_step)
        run.save(_snapshot(model, optimizer, sampler, record))
        if noted:
            run.note(f"yedek t{last_step}  (son adim; olcum satiri yukarida)")
        else:
            _note(run, record, time.time() - started_at, "  yedek")
    if run.stop_requested:
        run.result = dict(record or run.result, model=model, done=True, stopped=True)
        run.note(f"DURDURULDU  son tamamlanan adim {last_step} kaydedildi  -- tam olcum ATLANDI")
        run.flush()
        return
    prior = record.get("health") if record is not None and record["step"] == last_step else None
    record = dict(evaluate(last_step, full=True, grads=(record or {}).get("grads"), health=prior), done=True)
    run.result = dict(record, model=model)
    run.save(_snapshot(model, optimizer, sampler, record))
    s = record["eval_sec"]
    run.note(f"BITTI   train {record['train_acc']:.4f}   heldout {record['heldout_acc']:.4f}   ppl "
             f"{_ppl(record['heldout_ce']).strip()}   ({time.time() - started_at:.0f} sn, tam olcum "
             f"{s['train'] + s['heldout']:.0f} sn)")
    run.flush()
    decompose_hook(last_step)


def _decompose_backup(run, model, vocab, metric, step):
    """Tam yedekte decompose (diagnose_19): istem setinde acgozlu uretim, her secim parcalarina.  Yazar:
    <kosu>/decompose/t<adim><ek>.json (diagnose_19'un yerel kaydiyla ayni bicim ve anahtar), .html, _ozet.txt;
    gunluge tek satir.  Kosunun cihazinda; hata egitimi DURDURMAZ."""
    if not run.root or not vocab:
        return
    if model.arch != PointRelation.arch:
        run.note(f"decompose t{step}: {model.arch} icin yok (diagnose_19 PointRelation'a ozgu)")
        return
    import diagnose_19 as DG
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")          # TF32 yuvarlamasi verify'i haksiz yere dusurmesin
    was_training = model.training
    model.eval()
    try:
        t0 = time.time()
        vocab = list(vocab)
        if DG.is_math(vocab):
            prompts, tag = DG.package_prompts(None, vocab)
        elif DG.is_bpe(vocab):                             # Turkce iliski verisi: olcutun sinav sorulari, token'lari hazir
            prompts, tag = [list(p) for p in getattr(metric, "probes", ())], "_tr"
        else:
            prompts, tag = DG.prompt_set(getattr(metric, "probes", ())), ""
        stories = DG.compute(model, vocab, f"{run.name} t{step}", prompts, DG.STEPS, log=lambda s: None)
        base = f"{run.folder}/{DG.FOLDER}/t{step}{tag}"
        DG.save_record(base + ".json", DG.cache_key(prompts, DG.STEPS), step, stories)
        DG.write_outputs(base, stories, f"{run.name} t{step} Decompose")
        run.note(f"decompose t{step}: {DG.headline(stories)}  ({time.time() - t0:.0f} sn) -> {DG.FOLDER}/t{step}{tag}.html")
    except Exception as h:
        run.note(f"decompose t{step} HATA: {h!r}")
    finally:
        torch.set_float32_matmul_precision(precision)
        model.train(was_training)


def _run_safe(run, **settings):
    try:
        _run(run, **settings)
    except Exception:
        error = traceback.format_exc()
        for line in error.rstrip().splitlines():
            run.note("HATA  " + line)
        run.result = dict(run.result, error=error)
        try:
            run.flush()
        except Exception:
            pass


def start(run_name, data, n_vocab, *, metric=_no_metric, device="cuda", root=None, extra=None, lr=LR, steps=20000,
          seed=0, batch=BATCH, eval_every=100, save_every=1000, weights_every=100, resume=None, compile=True, vocab=None,
          warmup=0, decay_start=None, decay_floor=0.1, decompose=True, weight_decay=0.0, **model_kw):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    data        (questions, filled_mask, targets_mask)
    metric      metric(model, "train"|"heldout", full=False) -> accuracy (ANA OLCUT), ce, diag (gunlukte sutun)
    model_kw    arch ("point_relation" varsayilan ya da "looped_relation") ve o modelin ayarlari; PointRelation icin
                (d_order, d_content, embed, readout, chain_sim, distance, attn_after, rank ...);
                verilmeyen model_19'un varsayilani.  Varsayilan ZEMIN: yeni parcalar yalniz acikca verilince acilir
                (embed=True, readout=True, chain_sim=True, distance=True, attn_after=(0, 2)).
    eval_every / save_every / weights_every   olcum, TAM yedek (t<adim>.pt, surdurme), agirlik (w<adim>.pt) araliklari
    resume      tam yedekten KALDIGI YERDEN (kural 1: uzatma surdurmedir); config, veri ya da plan tutmazsa BASLAMAZ
    warmup      ilk kac adim dogrusal isinma (0: yok)
    decay_start LR sogutmasi (None: sabit): decay_start'tan steps'e cosine, lr'den lr x decay_floor'a
    decompose   her tam yedekte ve bitiste decompose_19 (varsayilan ACIK; egitimin yorungesine dokunmaz)
    weight_decay 0: Adam (zemin).  > 0: AdamW, yalniz DECAY'e uyan parametrelerde (egitim denemesi L1)"""
    old = RUNS.get(run_name)
    if old is not None and old.alive:
        raise RuntimeError(f"{run_name} hala kosuyor -- once stop('{run_name}')")
    run = RUNS[run_name] = Run(run_name, root)
    if root is None:
        run.note("UYARI: root YOK, agirlik KAYDEDILMIYOR")
    if not resume:
        moved = run.archive_old()
        if moved:
            print(f"  ESKI YEDEKLER KORUNDU -> {moved}/")
            run.note(f"eski yedekler tasindi -> {moved}/")
    run.thread = threading.Thread(target=_run_safe, args=(run,), daemon=True, kwargs=dict(
        data=data, n_vocab=n_vocab, metric=metric, device=device, lr=lr, steps=steps, seed=seed, batch=batch,
        eval_every=eval_every, save_every=save_every, weights_every=weights_every, resume=resume, compile=compile,
        vocab=vocab, extra=extra, warmup=warmup, decay_start=decay_start, decay_floor=decay_floor, decompose=decompose,
        weight_decay=weight_decay, model_kw=model_kw))
    run.thread.start()
    return f"{run_name} basladi" + (f"  ({os.path.basename(resume)}'den)" if resume else "")


def show_log(last=40):
    """Her kosunun durumu ve son satirlari.  HICBIR SEY KOSTURMAZ (kural 8)."""
    for run in RUNS.values():
        state = "CANLI" if run.alive else "HATA" if run.result.get("error") else "bitti"
        print(f"{run.name}: {state}   gunluk {len(run.log)} satir" + ("   DURDUR istendi" if run.stop_requested else ""))
        for line in run.log[-last:]:
            print(line)


def stop(run_name=None):
    """Dur istegi: adi verilen kosuya, yoksa canli olanlarin hepsine.  Tam olcum ATLANIR."""
    for run in ([RUNS[run_name]] if run_name else [r for r in RUNS.values() if r.alive]):
        run.stop()
    print("durdurma istendi:", sorted(r.name for r in RUNS.values() if r.stop_requested))
