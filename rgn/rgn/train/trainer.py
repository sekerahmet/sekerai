"""Eğitim döngüsü.

Çıktı klasörü (runs/<ad>/):
    config.json     tam yapılandırma
    metrics.jsonl   satır başına bir kayıt: {"type": "train" | "eval" | "diag", "step": ..., ...}
    checkpoint.pt   sürdürme durumu (model + optimizer + rng; ckpt_minutes'te bir ve bitişte)
    model.pt        bitişte model ağırlıkları + yapılandırma
    results.json    özet: son / en iyi doğrulama, hız, bellek, parametre, ortam
    samples.json    bitişte greedy örnekler
    stopped         SIGTERM (colab/runner.py kapat) ile durdurulduysa: checkpoint kaydedilip çıkılır; --resume sürdürür
"""
import contextlib
import json
import math
import platform
import signal
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch

from rgn.config import RunConfig
from rgn.data.loader import load_streams
from rgn.model import RGNModel, story_masks
from rgn.paths import runs_dir
from rgn.train import checkpoint
from rgn.train.diagnostics import diagnose, summary_line
from rgn.train.evaluate import evaluate, evaluate_rows, probes, samples
from rgn.train.optim import build_optimizer, param_groups
from rgn.train.schedule import lr_factor
from rgn.train.step import train_step


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device, dtype):
    if device.type == "cuda" and dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class _Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, **rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


def _env():
    env = {"torch": torch.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def train(cfg: RunConfig, out_dir=None, resume=False):
    tc, dc, mc = cfg.train, cfg.data, cfg.model
    device = _device()
    torch.manual_seed(tc.seed)
    # TF32 kapalı: eğitim çarpımları bf16; fp32 kalan (değerlendirmede logit) ölçüm tutarlılığı için tam fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = _autocast(device, tc.dtype)

    out = Path(out_dir) if out_dir else runs_dir() / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    log = _Log(out / "metrics.jsonl")

    train_s, valid_s, bytes_per_token = load_streams(dc)
    story = dc.layout == "story"
    if story:                                                    # hikâye / cümle dizilimi (rgn/data/stories.py)
        from rgn.data.stories import load_story_rows
        train_rows = load_story_rows(train_s.arrays[0], dc.story_root, "train")
        valid_rows = load_story_rows(valid_s.arrays[0], dc.story_root, "valid")
    model = RGNModel(mc).to(device)
    model.loss_checkpoint = tc.loss_checkpoint
    n_params = model.num_params()
    T = dc.seq_len
    tokens_per_step = tc.batch_size * T
    total_steps = tc.max_steps or max(1, math.ceil(tc.epochs * train_s.total / tokens_per_step))
    micro = tc.micro_batch or tc.batch_size
    assert tc.batch_size % micro == 0, "batch_size micro_batch'in katı olmalı"
    accum = tc.batch_size // micro
    opt = build_optimizer(model, tc)
    rng = np.random.default_rng(tc.seed)
    step, t_start = 0, time.time()
    best = None

    def say(msg):                                   # günlük biçimi: [geçen sn] mesaj
        print("[%8.1f sn] %s" % (time.time() - t_start, msg), flush=True)

    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        st = checkpoint.load(ckpt_path, map_location=device)
        model.load_state_dict(st["model"])
        hp = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
        opt.load_state_dict(st["optim"])
        changed = set()
        for g, h in zip(opt.param_groups, hp):                  # hiperparametreler checkpoint'ten değil yapılandırmadan
            for k in ("base_lr", "weight_decay", "betas", "momentum", "nesterov"):
                if k in h and g.get(k) != h[k]:
                    changed.add(k)
                    g[k] = h[k]
        if changed:
            say("SÜRDÜRME: yapılandırmadan gelen yeni değerler kullanılıyor: %s" % ", ".join(sorted(changed)))
        rng.bit_generator.state = st["rng"]
        step, best = st["step"], st.get("best")
        say("SÜRDÜRÜLDÜ adım %d'den (checkpoint.pt)" % step)

    fwd = torch.compile(model) if (tc.compile and device.type == "cuda") else model
    # kelimeye özel satır kopyası: bf16 eğitimde bf16 (çekirdek fp32 biriktirir), fp32 eğitimde fp32
    priv_dtype = torch.bfloat16 if (device.type == "cuda" and tc.dtype == "bf16") else torch.float32
    val_windows = valid_s.fixed_windows(T, dc.eval_tokens)
    diag_x = val_windows[:4, :-1].to(device)
    if story:
        val_rows = valid_rows.fixed_rows(T, dc.eval_tokens)
        say("hikâye dizilimi: doğrulama %d satır, %d token (dolgu %%%.1f) | yerel katman %s" % (
            len(val_rows[0]), int((val_rows[1] != -100).sum()), 100 * float((val_rows[1] == -100).float().mean()),
            "cümle + hikâye başı" if mc.local_sentence else "pencere %d" % mc.local_window))
    eval_batch = max(1, micro)

    log.write(type="start", step=step, total_steps=total_steps, stop_step=tc.stop_step, tokens_per_step=tokens_per_step,
              params=n_params, resumed=bool(resume and step), t=time.time())
    per_epoch = math.ceil(train_s.total / tokens_per_step)
    decay_start = int(total_steps * (1 - tc.decay)) if tc.schedule == "wsd" else None
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    n_priv = sum(p.numel() for _, _, p in model.private_tables())
    priv_kinds = " + ".join(k for k, on in (("nöron×%d" % mc.private_neurons, mc.private_neurons), ("V", mc.private_value))
                            if on)
    say("%s | katmanlar %s (Y: son %d token) | D %d, head %d (kv %d), d_ff %d | kapı %s | kelimeye özel %s | "
        "%d parametre (%d özel)" % (
            cfg.name, mc.layer_kinds(), mc.local_window, mc.d_model, mc.n_heads, mc.n_kv_heads, mc.ff_dim() if mc.shared_mlp else 0,
            mc.attn_gate or "yok",
            "%s, %d satır, %s" % (priv_kinds, model.private_tables()[0][2].shape[0], tc.private_opt) if n_priv else "yok",
            n_params, n_priv))
    say("veri %s %.1fM token | pencere %d, %d × %d = %d token/adım (mikro %d × %d) | %d adım/epok, toplam %d adım "
        "(%.2f epok)%s | lr %g (adam %g) %s, ısınma %d adım | cihaz %s, compile %s, %s | değerlendirme %d pencere" % (
            dc.dataset, train_s.total / 1e6, T, tc.batch_size, T, tokens_per_step, accum, micro, per_epoch, total_steps,
            total_steps / per_epoch, ", iniş başı %d" % decay_start if decay_start is not None else "", tc.lr,
            tc.adam_lr or tc.lr, tc.schedule, max(1, int(total_steps * tc.warmup)), gpu, tc.compile and device.type == "cuda",
            tc.dtype if device.type == "cuda" else "fp32", len(val_windows)))
    _mu, _de, _nd = param_groups(model, tc.optimizer in ("muon", "normuon"))
    say("optimizer %s | %s: %d tensör, %d parametre | adamw decay: %d tensör, %d parametre | adamw no_decay: %d tensör, "
        "%d parametre" % (tc.optimizer, tc.optimizer, len(_mu), sum(x.numel() for x in _mu), len(_de),
                          sum(x.numel() for x in _de), len(_nd), sum(x.numel() for x in _nd)))

    def run_eval(final=False):
        nonlocal best
        t_ev = time.time()
        if story:
            rows = valid_rows.fixed_rows(T, dc.final_eval_tokens) if final else val_rows
            r = evaluate_rows(model, rows, eval_batch, device, ctx, bytes_per_token,
                              window=0 if mc.local_sentence else mc.local_window)
        else:
            windows = valid_s.fixed_windows(T, dc.final_eval_tokens) if final else val_windows
            r = evaluate(model, windows, eval_batch, device, ctx, bytes_per_token)
        # cümle yerel modeli maskesiz üretemez (örnek / teşhis maskesiz yol)
        r["probes"] = probes(model, device, tc.probe_tokens) if tc.probe_tokens and not mc.local_sentence else []
        best = r["loss"] if best is None else min(best, r["loss"])
        log.write(type="eval", step=step, final=final, t=time.time(), **r)
        say("DEĞERLENDİRME%s adım %d: kayıp %.4f  ppl %.2f  bpb %s  doğruluk %.4f  bant %s | %d token (%.1f sn)" % (
            " SON" if final else "", step, r["loss"], r["ppl"], "%.4f" % r["bpb"] if r["bpb"] else "-", r["acc"],
            " ".join("%s:%.3f" % kv for kv in r["acc_bands"].items()), r["tokens"], time.time() - t_ev))
        for pr in r["probes"]:
            say("   örnek farklı4 %.2f%s | %r" % (pr["distinct4"], " DÖNGÜ" if pr["loop"] else "", pr["text"][:160]))
        return r

    def run_diag():
        if mc.local_sentence:
            return
        d = diagnose(model, diag_x, ctx)
        log.write(type="diag", step=step, **d)
        say("TEŞHİS adım %d: %s" % (step, summary_line(d)))

    def save_ckpt():
        checkpoint.save(ckpt_path, model=model.state_dict(), optim=opt.state_dict(), rng=rng.bit_generator.state,
                        step=step, best=best, config=cfg.to_dict())

    stop = {"flag": False}
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    step_ms, last_ckpt, start_step = [], time.time(), step
    t_log, s_log = time.time(), step
    model.train()
    while step < total_steps:
        if stop["flag"] or (tc.stop_step and step >= tc.stop_step):
            save_ckpt()
            (out / "stopped").write_text("adım %d\n" % step, encoding="utf-8")
            say("DURDURULDU adım %d — checkpoint kaydedildi; sürdürmek için --resume" % step)
            return None
        f = lr_factor(step, total_steps, tc.schedule, tc.warmup, tc.decay, tc.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        batches = []
        for _ in range(accum):
            masks = None
            if story:
                x, y, doc, sent = train_rows.random_batch(micro, T, rng)
                masks = story_masks(doc.to(device), sent.to(device), 0 if mc.local_sentence else mc.local_window)
            else:
                x, y = train_s.random_batch(micro, T, rng)
            batches.append((x, y, masks))
        loss_sum, gn = train_step(model, fwd, opt, batches, ctx, tc.grad_clip, device, priv_dtype)
        step += 1

        if step % tc.log_every == 0 or step == total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            win_steps, win_sec = step - s_log, time.time() - t_log
            sec = win_sec / max(1, win_steps)
            first_win = not step_ms and s_log == start_step
            if not first_win:                             # ilk pencere derleme süresini içerir
                step_ms.append(sec * 1000)
            t_log, s_log = time.time(), step
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            lv = float(loss_sum)
            log.write(type="train", step=step, loss=lv, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                      ms=sec * 1000, tok_s=tokens_per_step / sec, mem_gb=mem, t=time.time())
            say("adım %d / %d (epok %.3f)  lr %.3g  kayıp %.4f  grad %.3f | pencere %d adım %.1f sn  %.1f ms/adım  "
                "%d tok/sn  tepe %.1f GB  kalan ~%.0f dk%s" % (
                    step, total_steps, step / per_epoch, opt.param_groups[0]["lr"], lv, float(gn), win_steps, win_sec,
                    sec * 1000, tokens_per_step / sec, mem, (total_steps - step) * sec / 60,
                    "  (ilk pencere: derleme dahil)" if first_win else ""))
            if not math.isfinite(lv):
                raise FloatingPointError("kayıp sonlu değil (adım %d)" % step)
        if tc.eval_every and step % tc.eval_every == 0 and step < total_steps:
            run_eval()
        if tc.diag_every and step % tc.diag_every == 0:
            run_diag()
        if (time.time() - last_ckpt) / 60 >= tc.ckpt_minutes and step < total_steps:
            save_ckpt()
            last_ckpt = time.time()

    save_ckpt()                                                  # eğitim bitti: son durum güvende
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)          # bitiş aşamasında kapat sinyali süreci hemen sonlandırır
    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
    final = run_eval(final=True)
    run_diag()
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out / "model.pt")
    smp = [] if mc.local_sentence else samples(model, device, tc.sample_tokens)
    (out / "samples.json").write_text(json.dumps(smp, indent=1, ensure_ascii=False), encoding="utf-8")
    for s in smp[:2]:
        say("SON örnek | %r" % s["text"][:200])
    med = statistics.median(step_ms) if step_ms else None
    results = {
        "name": cfg.name, "params": n_params, "steps": step, "tokens": step * tokens_per_step,
        "final": final, "best_loss": best,
        "ms_per_step": med, "tok_per_s": tokens_per_step / (med / 1000) if med else None,
        "peak_mem_gb": peak_mem,
        "wall_min": (time.time() - t_start) / 60, "env": _env(), "config": cfg.to_dict(),
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "stopped").unlink(missing_ok=True)
    say("BİTTİ: %s | son kayıp %.4f  bpb %s  doğruluk %.4f | %s ms/adım  %.1f dk" % (
        cfg.name, final["loss"], "%.4f" % final["bpb"] if final["bpb"] else "-", final["acc"],
        "%.1f" % med if med else "-", results["wall_min"]))
    return results
