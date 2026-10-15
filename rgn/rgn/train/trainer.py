"""Eğitim döngüsü.

Çıktı klasörü (runs/<ad>/):
    config.json     tam yapılandırma
    metrics.jsonl   satır başına bir kayıt: {"type": "train" | "eval" | "diag", "step": ..., ...}
    checkpoint.pt   sürdürme durumu (model + optimizer + rng; ckpt_minutes'te bir ve bitişte)
    model.pt        bitişte model ağırlıkları + yapılandırma
    results.json    özet: son / en iyi doğrulama, hız, bellek, parametre, ortam
    samples.json    bitişte greedy örnekler
    speed.json      bitişte hız profili (rgn/bench/profile_run.py)
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
from rgn.model import RGNModel
from rgn.paths import runs_dir
from rgn.train import checkpoint
from rgn.train.diagnostics import diagnose, summary_line
from rgn.train.evaluate import evaluate, probes, samples
from rgn.train.optim import build_optimizer, param_groups
from rgn.train.schedule import lr_factor


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
    tc, dc = cfg.train, cfg.data
    device = _device()
    torch.manual_seed(tc.seed)
    # TF32 bilinçli olarak KAPALI: bf16 autocast altında fp32 kalan tek matris çarpımı çekim dikkatidir (rgn_fp32) ve
    # TF32 onu ~1e-3 göreli hataya düşürür (yakın eşleşme iptal hatası; doküman 3.5).  Diğer çarpımlar zaten bf16.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = _autocast(device, tc.dtype)

    out = Path(out_dir) if out_dir else runs_dir() / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    log = _Log(out / "metrics.jsonl")

    train_s, valid_s, bytes_per_token = load_streams(dc)
    model = RGNModel(cfg.model).to(device)
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

    def say(msg):                                   # AI_NEW_MODEL v2 günlük biçimi: [geçen sn] mesaj
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
    val_windows = valid_s.fixed_windows(T, dc.eval_tokens)
    diag_x = val_windows[:4, :-1].to(device)
    eval_batch = max(1, micro)

    log.write(type="start", step=step, total_steps=total_steps, stop_step=tc.stop_step, tokens_per_step=tokens_per_step,
              params=n_params,
              resumed=bool(resume and step), t=time.time())
    mc = cfg.model
    per_epoch = math.ceil(train_s.total / tokens_per_step)
    decay_start = int(total_steps * (1 - tc.decay)) if tc.schedule == "wsd" else None
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    if mc.local_window:
        say("katmanlar %s (Y: yerel, son %d token, konum %s, d_ff %d | G: genel, tam, konum %s, d_ff %d)" % (
            mc.layer_kinds(), mc.local_window, mc.pos, mc.ff_dim("Y"), mc.global_pos or mc.pos, mc.ff_dim("G")))
    say("%s | attn %s pos %s mlp %s p %d | D %d, katman %d, head %d (kv %d), d_ff %d, %d parametre" % (
        cfg.name, mc.attn, mc.pos, mc.mlp, mc.power, mc.d_model, mc.n_layers, mc.n_heads, mc.n_kv_heads, mc.ff_dim(),
        n_params))
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
        windows = valid_s.fixed_windows(T, dc.final_eval_tokens) if final else val_windows
        t_ev = time.time()
        r = evaluate(model, windows, eval_batch, device, ctx, bytes_per_token)
        r["probes"] = probes(model, device, tc.probe_tokens) if tc.probe_tokens else []
        best = r["loss"] if best is None else min(best, r["loss"])
        log.write(type="eval", step=step, final=final, t=time.time(), **r)
        say("DEĞERLENDİRME%s adım %d: kayıp %.4f  ppl %.2f  bpb %s  doğruluk %.4f  bant %s | %d token (%.1f sn)" % (
            " SON" if final else "", step, r["loss"], r["ppl"], "%.4f" % r["bpb"] if r["bpb"] else "-", r["acc"],
            " ".join("%s:%.3f" % kv for kv in r["acc_bands"].items()), r["tokens"], time.time() - t_ev))
        for pr in r["probes"]:
            say("   örnek farklı4 %.2f%s | %r" % (pr["distinct4"], " DÖNGÜ" if pr["loop"] else "", pr["text"][:160]))
        return r

    def run_diag():
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
        loss_sum = 0.0
        for _ in range(accum):
            x, y = train_s.random_batch(micro, T, rng)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ctx:
                loss = fwd(x, y) / accum
            loss.backward()
            loss_sum += loss.detach()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip) if tc.grad_clip else torch.tensor(0.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % tc.log_every == 0 or step == total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            win_steps, win_sec = step - s_log, time.time() - t_log
            dt = win_sec / max(1, win_steps)
            first_win = not step_ms and s_log == start_step
            if not first_win:                             # ilk pencere derleme süresini içerir
                step_ms.append(dt * 1000)
            t_log, s_log = time.time(), step
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            lv = float(loss_sum)
            log.write(type="train", step=step, loss=lv, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                      ms=dt * 1000, tok_s=tokens_per_step / dt, mem_gb=mem, t=time.time())
            say("adım %d / %d (epok %.3f)  lr %.3g  kayıp %.4f  grad %.3f | pencere %d adım %.1f sn  %.1f ms/adım  "
                "%d tok/sn  tepe %.1f GB  kalan ~%.0f dk%s" % (
                    step, total_steps, step / per_epoch, opt.param_groups[0]["lr"], lv, float(gn), win_steps, win_sec,
                    dt * 1000, tokens_per_step / dt, mem, (total_steps - step) * dt / 60,
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
    speed = None
    if tc.profile_after:                                         # örnek üretimden ÖNCE (derleme önbelleği temizken)
        from rgn.bench.profile_run import profile_after_run
        speed = profile_after_run(model, cfg, device, ctx, out / "speed.json")
    smp = samples(model, device, tc.sample_tokens)
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
    if speed:
        results["speed"] = {k: speed.get(k) for k in ("attention", "block", "inference")}
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "stopped").unlink(missing_ok=True)
    say("BİTTİ: %s | son kayıp %.4f  bpb %s  doğruluk %.4f | %s ms/adım  %.1f dk" % (
        cfg.name, final["loss"], "%.4f" % final["bpb"] if final["bpb"] else "-", final["acc"],
        "%.1f" % med if med else "-", results["wall_min"]))
    return results
