# -*- coding: utf-8 -*-
"""train -- PV'nin egitim dongusu: optimizer, yigin, kayit, SURDURME (model_17 train_17'den).

GOREV SONRAKI TOKEN, kayip YALNIZ hedeflerde:

    loss = m.loss(w, H)     w (B,T) pencere, H hedef maskesi (cevabin rakamlari ve EOS)

data = (W, M, H): W (n, T) pencereler, M dolgu maskesi (yalniz gunluk icin;
PV'de konumlar birbirini gormez), H hedef maskesi.

Adam, weight decay YOK.  Kullanici, 24 Eylul: "weight decay sanki anlamsiz
gerek yok" -- start ve finish nokta; onlari 0'a cekmenin anlami yok.

KOSU ARKA PLANDA (CLAUDE.md kural 8).  `start` hemen doner, ilerleme
`show_log` ile OKUNUR.  Her kosu KENDI gunluk.txt'sine her olcumde EKLER;
surdurmede eski satirlar durur.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
import time
import traceback

import torch

import model_18 as M18
from model_18 import PV

LR = 0.002       # model_17'den: rakam tokenli veride 0,001/0,002/0,004 ayirt edilemedi
BATCH = 512
RUNS = {}        # ad -> Run

# Surdurmede paketteki degerle AYNI olmali.  Farkliysa yorunge sessizce
# baskalasir: opt.load_state_dict lr'yi paketten alir, gunluk cagriyi yazar.
MUST_MATCH = ("arch", "n", "T", "batch", "lr", "seed", "d", "vectors",
              "active", "layers", "t_max", "data_fingerprint", "vocab")


class Run:
    """Bir kosunun durumu tek yerde: gunluk, sonuc, iplik, dur istegi, kayit."""

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

    def note(self, s):
        """Satir hem kosunun gunlugune (show_log) hem disk kuyruguna."""
        s = f"[{self.name}] {s}"
        self.log.append(s)
        self._pending.append(s)

    def flush(self):
        """Yeni satirlari gunluk.txt'ye EKLER -- uzerine yazmaz."""
        if self._pending and self.root:
            os.makedirs(self.folder, exist_ok=True)
            with open(f"{self.folder}/gunluk.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(self._pending) + "\n")
        self._pending.clear()

    def save(self, pkg):
        """Anlik goruntuyu diske yaz -- KOSUNUN ICINDE.  Eskiler silinmez."""
        if not self.root:
            return
        os.makedirs(self.folder, exist_ok=True)
        torch.save(pkg, f"{self.folder}/t{pkg['step']}.pt")
        torch.save(pkg, f"{self.root}/model_{self.name}.pt")

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
        """Iplik bir sonraki adimda cikar ve cikmadan KAYDEDER."""
        self.stop_requested = True


def _no_metric(m, side, full=False):
    raise RuntimeError("metric verilmedi -- start(..., metric=...) sart")


def _ppl(ce):
    return "      -" if ce is None else f"{math.exp(ce):7.2f}"


def _fingerprint(*t):
    """Egitim tensorlerinin parmak izi -- surdurme AYNI veriyle mi."""
    h = hashlib.sha256()
    for x in t:
        h.update(f"{tuple(x.shape)} {x.dtype}".encode())
        h.update(x.contiguous().numpy())
    return h.hexdigest()[:16]


def _snapshot(m, opt, sampler, info):
    """TAM anlik goruntu -- surdurmeye yeten her sey."""
    return dict(info, weights=m.state_dict(), opt=opt.state_dict(),
                rng=torch.get_rng_state(), sampler_rng=sampler.get_state())


def _check_resume(pkg, fixed):
    """Surdurme paketi bu cagriyla AYNI kosu mu?  Degilse kosu BASLAMAZ."""
    short = lambda v: f"<{len(v)} birim>" if isinstance(v, list) else repr(v)
    diffs = [f"{a}: paket {short(pkg[a])}, cagri {short(fixed[a])}"
             for a in MUST_MATCH
             if pkg.get(a) is not None and fixed.get(a) is not None
             and pkg[a] != fixed[a]]
    if diffs:
        raise ValueError("surdurme paketi bu cagriyla uyusmuyor -- "
                         + "; ".join(diffs))


def _header(info):
    """Sutun adlari; diag sutunlari verinin metric'inden gelir."""
    return ("  step     loss    train    heldout      ppl  "
            + "".join(f"{k:>12}" for k in info["heldout_diag"]) + "       s")


def _line(info, elapsed, mark=""):
    return (f"{info['step']:6d}  {info['loss']:7.3f}  {info['train_acc']:7.4f}  "
            f"{info['heldout_acc']:9.4f}  {_ppl(info['heldout_ce'])}  "
            + "".join(f"{v:12.4f}" for v in info["heldout_diag"].values())
            + f"  {elapsed:6.0f}{mark}")


def _run(r, data, n_vocab, metric, device, lr, steps, seed, batch,
         eval_every, save_every, resume=None, compile=False, vocab=None,
         extra=None, d=M18.d, vectors=M18.VECTORS, active=M18.ACTIVE,
         layers=M18.LAYERS, t_max=M18.T_MAX):
    torch.manual_seed(seed)
    m = PV(n_vocab, d=d, vectors=vectors, active=active, layers=layers,
           t_max=t_max, seed=seed).to(device)
    # torch.compile: eski mimaride 3,90 kat olculdu (model_17 train_17); PV'de OLCULMEDI.
    loss_fn = torch.compile(m.loss) if compile else m.loss
    if compile:
        r.note("torch.compile ACIK -- ilk adim DERLEME yuzunden yavas")
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sampler = torch.Generator(device="cpu").manual_seed(seed)

    # PENCERELER CPU'DA KALIR, yigin yigin tasinir.
    W, M, H = (x.cpu() for x in data)
    n, T = W.shape
    assert T <= t_max, f"pencere {T} > t_max {t_max}"
    n_params = sum(p.numel() for p in m.parameters())
    fixed = dict(extra or {}, arch=m.arch, n=n_vocab, d=d, vectors=vectors,
                 active=active, layers=layers, t_max=t_max, lr=lr,
                 seed=seed, batch=batch, T=T, n_params=n_params, compile=compile,
                 data_fingerprint=_fingerprint(W, M, H),
                 vocab=None if vocab is None else [str(a) for a in vocab])

    last, prev, loss_t = -1, None, None   # son TAMAMLANAN adim
    if resume:
        pkg = torch.load(resume, weights_only=False, map_location=device)
        _check_resume(pkg, fixed)
        fixed["vocab"] = fixed["vocab"] or pkg.get("vocab")
        m.load_state_dict(pkg["weights"])
        opt.load_state_dict(pkg["opt"])
        torch.set_rng_state(pkg["rng"].cpu())
        sampler.set_state(pkg["sampler_rng"].cpu())
        last, prev = pkg["step"], pkg.get("step_losses")
        loss_t = torch.tensor(pkg["loss"])      # yeni adim atilmazsa pakettekini tasir
        r.note(f"SURDURULUYOR  {os.path.basename(resume)}  adim {last}")
    last_saved = last                     # paket o adimin kaydi zaten

    # HER ADIMIN kaybi: egrinin tam cozunurlugu.  GPU'da birikir (adim basina
    # senkron yok), olcum noktasinda pakete girer, surdurmede geri yuklenir.
    step_losses = torch.full((max(steps, last) + 1,), float("nan"), device=device)
    if prev is not None:
        step_losses[:len(prev)] = prev.to(device)

    fill = float(M.sum()) / M.numel()
    r.note(f"pencere {n:,} x {T}   {n * T:,} yuva   dolgu %{100 * (1 - fill):.1f}")
    r.note(f"arch {m.arch}  d {d} vectors {vectors} active {active} "
           f"layers {layers}  lr {lr} seed {seed}  sozluk {n_vocab}  "
           f"parametre {n_params:,}")
    r.note(f"yigin {batch}   epok = {n / batch:,.0f} adim   veri izi "
           f"{fixed['data_fingerprint']}   olcum her {eval_every}   yedek her {save_every}")
    r.note("OLCUT: accuracy (train / heldout); kayip yalniz hedeflerde")
    r.flush()

    def evaluate(i, full=False):
        e, h = metric(m, "train", full=full), metric(m, "heldout", full=full)
        info = dict(fixed, step=i, loss=float(loss_t.detach()),
                    step_losses=step_losses[:i + 1].cpu(),
                    train_acc=e["accuracy"], heldout_acc=h["accuracy"],
                    train_ce=e.get("ce"), heldout_ce=h.get("ce"),
                    train_diag=e.get("diag", {}), heldout_diag=h.get("diag", {}))
        if not r.result:
            r.note(_header(info))
        r.result = dict(info, model=m)
        return info

    # Yedek, o adimin opt.step()'i BITTIKTEN sonra yazilir.  Yani t<N>
    # N adimi ICERIR ve surdurme N+1'den baslar; N'den baslamak o adimi
    # IKI KEZ atar ve yorunge kayar (tests: surdurme == kesintisiz).
    t0, info = time.time(), None
    for i in range(last + 1, steps + 1):
        if r.stop_requested:
            break
        j = torch.randint(0, n, (batch,), generator=sampler)
        opt.zero_grad()
        loss_t = loss_fn(W[j].to(device, non_blocking=True).long(),
                         H[j].to(device, non_blocking=True))
        step_losses[i] = loss_t.detach()
        loss_t.backward()
        opt.step()
        last = i
        # yedek eval_every'nin katina bagli DEGIL: 100 ve 250 -> 250'de yazar.
        if i % eval_every == 0 or i % save_every == 0:
            info, mark = evaluate(i), ""
            if i % save_every == 0:
                r.save(_snapshot(m, opt, sampler, info))
                last_saved, mark = i, "  yedek"
            r.note(_line(info, time.time() - t0, mark))
            r.flush()

    if last < 0:
        r.note("DURDURULDU -- hic adim atilmadi, kayit yok")
        r.flush()
        return
    _finish(r, m, opt, sampler, evaluate, info, last, last_saved, t0)


def _finish(r, m, opt, sampler, evaluate, info, last, last_saved, t0):
    """Kosunun sonu.  ONCE KAYDET, SONRA OLC: tam olcum uzun surebilir; o arada
    oturum duserse son yedekten beri yapilan is giderdi."""
    if last_saved != last:
        if info is None or info["step"] != last:
            info = evaluate(last)
        r.save(_snapshot(m, opt, sampler, info))
        r.note(_line(info, time.time() - t0, "  yedek"))
    if r.stop_requested:
        # Durdurmanin amaci GPU'yu HEMEN birakmak; tam olcum ATLANIR.
        r.result = dict(info or r.result, model=m, done=True, stopped=True)
        r.note(f"DURDURULDU  son tamamlanan adim {last} kaydedildi  -- tam olcum ATLANDI")
        r.flush()
        return
    info = dict(evaluate(last, full=True), done=True)
    r.result = dict(info, model=m)
    r.save(_snapshot(m, opt, sampler, info))
    r.note(f"BITTI   train {info['train_acc']:.4f}   heldout "
           f"{info['heldout_acc']:.4f}   ppl {_ppl(info['heldout_ce']).strip()}"
           f"   ({time.time() - t0:.0f} sn)")
    r.flush()


def _run_safe(r, **k):
    """_run'i sarar: iplik olurse show_log bunu GORSUN, kosu suruyor sanilmasin."""
    try:
        _run(r, **k)
    except Exception:
        tb = traceback.format_exc()
        for s in tb.rstrip().splitlines():
            r.note("HATA  " + s)
        r.result = dict(r.result, error=tb)
        try:
            r.flush()
        except Exception:
            pass


def start(run, data, n_vocab, *, metric=_no_metric, device="cuda", root=None,
          extra=None, lr=LR, steps=20000, seed=0, batch=BATCH,
          eval_every=100, save_every=500, resume=None, compile=True, vocab=None,
          d=M18.d, vectors=M18.VECTORS, active=M18.ACTIVE, layers=M18.LAYERS,
          t_max=M18.T_MAX):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    data        (W, M, H) -- W (n, T) pencereler, M dolgu maskesi, H hedef maskesi
    metric      metric(m, "train"|"heldout", full=False) -> dict: accuracy (ANA OLCUT),
                ce, diag (veriye ozgu analiz olculeri; gunlukte sutun olur)
    eval_every  olcum araligi;  save_every  anlik goruntu araligi -- birbirinden BAGIMSIZ.
                HER ADIMIN kaybi ikisinden de bagimsiz: pakette `step_losses`.
    resume      bir anlik goruntu yolu verilirse KALDIGI YERDEN devam eder
                (agirlik + optimizer + RNG).  Kural 1: uzatma SURDURMEDIR.
                Paketteki ayar ya da veri bu cagriyla tutmazsa kosu BASLAMAZ.
    vocab       token listesi -- pakete yazilir
    compile     torch.compile, ACIK.  Eski mimaride 3,90 kat olculdu; PV'de OLCULMEDI.
                Ilk adim derleme yuzunden yavas.
    d, vectors, active, layers, t_max   PV'nin ayarlari (model_18)
    """
    old = RUNS.get(run)
    if old is not None and old.alive:
        raise RuntimeError(f"{run} hala kosuyor -- once stop('{run}')")
    r = RUNS[run] = Run(run, root)
    if root is None:
        r.note("UYARI: root YOK, agirlik KAYDEDILMIYOR")
    if not resume:
        moved = r.archive_old()
        if moved:
            print(f"  ESKI YEDEKLER KORUNDU -> {moved}/")
            r.note(f"eski yedekler tasindi -> {moved}/")
    r.thread = threading.Thread(target=_run_safe, args=(r,), daemon=True, kwargs=dict(
        data=data, n_vocab=n_vocab, metric=metric, device=device, lr=lr,
        steps=steps, seed=seed, batch=batch, eval_every=eval_every,
        save_every=save_every, resume=resume, compile=compile, vocab=vocab,
        extra=extra, d=d, vectors=vectors, active=active, layers=layers,
        t_max=t_max))
    r.thread.start()
    return f"{run} basladi" + (f"  ({os.path.basename(resume)}'den)" if resume else "")


def show_log(last=40):
    """Her kosunun durumu ve son satirlari.  HICBIR SEY KOSTURMAZ (kural 8)."""
    for r in RUNS.values():
        state = ("CANLI" if r.alive else "HATA" if r.result.get("error") else "bitti")
        print(f"{r.name}: {state}   gunluk {len(r.log)} satir"
              + ("   DURDUR istendi" if r.stop_requested else ""))
        for s in r.log[-last:]:
            print(s)


def stop(run=None):
    """Dur istegi: adi verilen kosuya, yoksa canli olanlarin hepsine.
    Tam olcum ATLANIR -- durdurmanin amaci GPU'yu hemen birakmak."""
    for r in ([RUNS[run]] if run else [r for r in RUNS.values() if r.alive]):
        r.stop()
    print("durdurma istendi:", sorted(r.name for r in RUNS.values() if r.stop_requested))
