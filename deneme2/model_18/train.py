# -*- coding: utf-8 -*-
"""train -- PV'nin egitim dongusu: optimizer, batch, kayit, SURDURME (model_17 train_17'den).

GOREV SONRAKI TOKEN, kayip YALNIZ hedeflerde:

    loss = model.loss(tokens, targets_mask)    tokens (B,T) sorular,
                                                targets_mask cevabin rakamlari ve EOS

data = (questions, filled_mask, targets_mask): questions (n, T) sorular,
filled_mask gercek token'lar (yalniz gunluk icin; PV'de konumlar birbirini
gormez), targets_mask kayba sayilacak hedefler.

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
RUNS = {}        # run_name -> Run

# Surdurmede paketteki degerle AYNI olmali.  Farkliysa yorunge sessizce
# baskalasir: optimizer.load_state_dict lr'yi paketten alir, gunluk cagriyi yazar.
MUST_MATCH = ("arch", "n", "T", "batch", "lr", "seed", "d_sum", "vectors",
              "active", "layers", "t_max", "squared", "S_p", "start_norm", "lam", "chain", "c_cache", "cache_topk",
              "cache_skip", "s_v_init", "s_c_init", "gate_0_init", "s_p_init", "query", "query_vectors",
              "query_active", "query_by", "c_content", "d_order", "d_content", "lam_w_init", "beta_w_init", "select", "load_balance",
              "data_fingerprint", "vocab")
# Alan eklenmeden once yazilan paketlerdeki deger: skor -D^2, carpansiz.
BEFORE_FIELD = {"squared": True, "S_p": 1.0, "start_norm": "randn", "lam": 1.0,
                "chain": "absolute", "c_cache": False,
                "cache_topk": M18.CACHE_TOPK, "cache_skip": M18.CACHE_SKIP,
                "s_v_init": 0.0, "s_c_init": 3.0, "gate_0_init": -2.0, "s_p_init": 0.0,
                "query": False, "query_vectors": 0, "query_active": 8, "query_by": "C_m",
                "c_content": False, "d_order": 0, "d_content": 0, "select": "distance", "load_balance": 0.0,
                "lam_w_init": 0.9, "beta_w_init": 0.5}


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

    def note(self, line):
        """Satir hem kosunun gunlugune (show_log) hem disk kuyruguna."""
        line = f"[{self.name}] {line}"
        self.log.append(line)
        self._pending.append(line)

    def flush(self):
        """Yeni satirlari gunluk.txt'ye EKLER -- uzerine yazmaz."""
        if self._pending and self.root:
            os.makedirs(self.folder, exist_ok=True)
            with open(f"{self.folder}/gunluk.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(self._pending) + "\n")
        self._pending.clear()

    def save(self, package, full=True):
        """Diske yaz -- KOSUNUN ICINDE.  Eskiler silinmez.
        full: t<adim>.pt, surdurmeye yeten TAM yedek (+ model_<ad>.pt son hal).
        degilse: w<adim>.pt, yalniz agirliklar ve olcum -- analiz icin."""
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
        """Iplik bir sonraki adimda cikar ve cikmadan KAYDEDER."""
        self.stop_requested = True


def _no_metric(model, side, full=False):
    raise RuntimeError("metric verilmedi -- start(..., metric=...) sart")


def _ppl(ce):
    return "      -" if ce is None else f"{math.exp(ce):7.2f}"


def _fingerprint(*tensors):
    """Egitim tensorlerinin parmak izi -- surdurme AYNI veriyle mi."""
    h = hashlib.sha256()
    for x in tensors:
        h.update(f"{tuple(x.shape)} {x.dtype}".encode())
        h.update(x.contiguous().numpy())
    return h.hexdigest()[:16]


def _snapshot(model, optimizer, sampler, record):
    """TAM anlik goruntu -- surdurmeye yeten her sey."""
    return dict(record, weights=model.state_dict(), opt=optimizer.state_dict(),
                rng=torch.get_rng_state(), sampler_rng=sampler.get_state())


# Kapali ozelligin ayarlari modele girmez: iki taraf da kapaliysa karsilastirilmaz.
SWITCHED = {"query": ("query_vectors", "query_active", "query_by"),
            "c_content": ("d_order", "d_content", "lam_w_init", "beta_w_init")}


def _check_resume(package, fixed):
    """Surdurme paketi bu cagriyla AYNI kosu mu?  Degilse kosu BASLAMAZ."""
    short = lambda v: f"<{len(v)} birim>" if isinstance(v, list) else repr(v)
    packed = {a: package.get(a, BEFORE_FIELD.get(a)) for a in MUST_MATCH}
    packed["d_sum"] = package.get("d_sum", package.get("d"))        # eski paketin 'd' alani
    off = {a for s, alt in SWITCHED.items() if not packed[s] and not fixed.get(s) for a in alt}
    diffs = [f"{a}: paket {short(packed[a])}, cagri {short(fixed[a])}"
             for a in MUST_MATCH
             if a not in off and packed[a] is not None and fixed.get(a) is not None
             and packed[a] != fixed[a]]
    if diffs:
        raise ValueError("surdurme paketi bu cagriyla uyusmuyor -- "
                         + "; ".join(diffs))


def _header(record):
    """Sutun adlari; diag sutunlari verinin metric'inden gelir."""
    return ("  step     loss    train    heldout      ppl  "
            + "".join(f"{k:>12}" for k in record["heldout_diag"]) + "       s")


def _line(record, elapsed, mark=""):
    return (f"{record['step']:6d}  {record['loss']:7.3f}  {record['train_acc']:7.4f}  "
            f"{record['heldout_acc']:9.4f}  {_ppl(record['heldout_ce'])}  "
            + "".join(f"{v:12.4f}" for v in record["heldout_diag"].values())
            + f"  {elapsed:6.0f}{mark}")


def _run(run, data, n_vocab, metric, device, lr, steps, seed, batch,
         eval_every, save_every, weights_every=100, resume=None, compile=False, vocab=None,
         extra=None, vectors=M18.VECTORS, active=M18.ACTIVE,
         layers=M18.LAYERS, t_max=M18.T_MAX, squared=None, S_p=None,
         start_norm=M18.START_NORM, lam=M18.LAM, chain=M18.CHAIN, c_cache=M18.C_CACHE,
         cache_topk=M18.CACHE_TOPK, cache_skip=M18.CACHE_SKIP, s_v_init=M18.S_V_INIT,
         s_c_init=M18.S_C_INIT, gate_0_init=M18.GATE_0_INIT, s_p_init=M18.S_P_INIT,
         query=M18.QUERY, query_vectors=M18.QUERY_VECTORS, query_active=M18.QUERY_ACTIVE,
         query_by=M18.QUERY_BY, c_content=M18.C_CONTENT,
         d_order=M18.D_ORDER, d_content=M18.D_CONTENT, lam_w_init=M18.LAM_W_INIT, beta_w_init=M18.BETA_W_INIT,
         select=M18.SELECT, load_balance=M18.LOAD_BALANCE):
    torch.manual_seed(seed)
    model = PV(n_vocab, vectors=vectors, active=active, layers=layers,
               t_max=t_max, seed=seed, squared=squared, S_p=S_p,
               start_norm=start_norm, lam=lam, chain=chain, c_cache=c_cache,
               cache_topk=cache_topk, cache_skip=cache_skip, s_v_init=s_v_init,
               s_c_init=s_c_init, gate_0_init=gate_0_init, s_p_init=s_p_init,
               query=query, query_vectors=query_vectors, query_active=query_active, query_by=query_by,
               c_content=c_content, d_order=d_order, d_content=d_content, lam_w_init=lam_w_init, beta_w_init=beta_w_init,
               select=select, load_balance=load_balance).to(device)
    # torch.compile: eski mimaride 3,90 kat olculdu (model_17 train_17); PV'de OLCULMEDI.
    loss_fn = torch.compile(model.loss) if compile else model.loss
    if compile:
        run.note("torch.compile ACIK -- ilk adim DERLEME yuzunden yavas")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = torch.Generator(device="cpu").manual_seed(seed)

    # SORULAR CPU'DA KALIR, batch batch tasinir.
    questions, filled_mask, targets_mask = (x.cpu() for x in data)
    n_questions, max_length = questions.shape
    assert max_length <= t_max, f"soru {max_length} token > t_max {t_max}"
    n_params = sum(p.numel() for p in model.parameters())
    fixed = dict(extra or {}, arch=model.arch, n=n_vocab, d_sum=model.d_sum, vectors=vectors,
                 active=active, layers=layers, t_max=t_max, lr=lr,
                 squared=model.squared,
                 S_p=M18.LEARNED if model.S_p_learned else model.S_p,
                 start_norm="randn" if start_norm is None else float(start_norm),
                 lam=float(lam), chain=chain, c_cache=bool(c_cache),
                 cache_topk=None if cache_topk is None else int(cache_topk), cache_skip=int(cache_skip),
                 s_v_init=float(s_v_init), s_c_init=float(s_c_init),
                 gate_0_init=float(gate_0_init), s_p_init=float(s_p_init),
                 query=bool(query), query_vectors=int(query_vectors), query_active=int(query_active),
                 query_by=query_by, c_content=bool(c_content), d_order=model.d_order, d_content=model.d_content, lam_w_init=float(lam_w_init), beta_w_init=float(beta_w_init), select=select,
                 load_balance=float(load_balance),
                 seed=seed, batch=batch, T=max_length, n_params=n_params,
                 compile=compile,
                 data_fingerprint=_fingerprint(questions, filled_mask, targets_mask),
                 vocab=None if vocab is None else [str(a) for a in vocab])

    last_step, prev_step_losses, loss = -1, None, None   # son TAMAMLANAN adim
    if resume:
        package = torch.load(resume, weights_only=False, map_location=device)
        _check_resume(package, fixed)
        fixed["vocab"] = fixed["vocab"] or package.get("vocab")
        model.load_state_dict(package["weights"])
        optimizer.load_state_dict(package["opt"])
        torch.set_rng_state(package["rng"].cpu())
        sampler.set_state(package["sampler_rng"].cpu())
        last_step, prev_step_losses = package["step"], package.get("step_losses")
        loss = torch.tensor(package["loss"])     # yeni adim atilmazsa pakettekini tasir
        run.note(f"SURDURULUYOR  {os.path.basename(resume)}  adim {last_step}")
    last_saved_step = last_step           # paket o adimin kaydi zaten

    # HER ADIMIN kaybi: egrinin tam cozunurlugu.  GPU'da birikir (adim basina
    # senkron yok), olcum noktasinda pakete girer, surdurmede geri yuklenir.
    step_losses = torch.full((max(steps, last_step) + 1,), float("nan"), device=device)
    if prev_step_losses is not None:
        step_losses[:len(prev_step_losses)] = prev_step_losses.to(device)

    fill_ratio = float(filled_mask.sum()) / filled_mask.numel()
    run.note(f"sorular {n_questions:,} x {max_length}   dolgu %{100 * (1 - fill_ratio):.1f}")
    run.note(f"arch {model.arch}  D_SUM {model.d_sum} ({model.d_order}+{model.d_content}) vectors {vectors} active {active} "
             f"layers {layers}  squared {fixed['squared']} S_p {fixed['S_p']}  "
             f"start_norm {fixed['start_norm']}  lam {fixed['lam']}  chain {chain}  c_cache {c_cache} (topk {cache_topk} skip {cache_skip} Q {query} {query_vectors}/{query_active} {query_by})  c_content {c_content} (lam_w {lam_w_init} beta_w {beta_w_init})  select {select}  load_balance {load_balance}  lr {lr} seed {seed}  sozluk {n_vocab}  "
             f"parametre {n_params:,}")
    run.note(f"batch {batch}   epok = {n_questions / batch:,.0f} adim   veri izi "
             f"{fixed['data_fingerprint']}   olcum her {eval_every}   tam yedek her "
             f"{save_every}   agirlik her {weights_every}")
    run.note("OLCUT: accuracy (train / heldout); kayip yalniz hedeflerde")
    run.flush()

    def evaluate(step, full=False):
        train_result = metric(model, "train", full=full)
        heldout_result = metric(model, "heldout", full=full)
        record = dict(fixed, step=step, loss=float(loss.detach()),
                      step_losses=step_losses[:step + 1].cpu(),
                      train_acc=train_result["accuracy"],
                      heldout_acc=heldout_result["accuracy"],
                      train_ce=train_result.get("ce"), heldout_ce=heldout_result.get("ce"),
                      train_diag=train_result.get("diag", {}),
                      heldout_diag=heldout_result.get("diag", {}))
        if not run.result:
            run.note(_header(record))
        run.result = dict(record, model=model)
        return record

    # Yedek, o adimin optimizer.step()'i BITTIKTEN sonra yazilir.  Yani t<N>
    # N adimi ICERIR ve surdurme N+1'den baslar; N'den baslamak o adimi
    # IKI KEZ atar ve yorunge kayar (tests: surdurme == kesintisiz).
    started_at, record = time.time(), None
    for step in range(last_step + 1, steps + 1):
        if run.stop_requested:
            break
        batch_ids = torch.randint(0, n_questions, (batch,), generator=sampler)
        optimizer.zero_grad()
        loss = loss_fn(questions[batch_ids].to(device, non_blocking=True).long(),
                       targets_mask[batch_ids].to(device, non_blocking=True))
        step_losses[step] = loss.detach()
        loss.backward()
        optimizer.step()
        last_step = step
        # Uc aralik birbirinden BAGIMSIZ: olcum, tam yedek (surdurme), agirlik (analiz).
        full_save, weights_save = step % save_every == 0, step % weights_every == 0
        if step % eval_every == 0 or full_save or weights_save:
            record, mark = evaluate(step), ""
            if full_save:
                run.save(_snapshot(model, optimizer, sampler, record))
                last_saved_step, mark = step, "  yedek"
            elif weights_save:
                run.save(dict(record, weights=model.state_dict()), full=False)
            run.note(_line(record, time.time() - started_at, mark))
            run.flush()

    if last_step < 0:
        run.note("DURDURULDU -- hic adim atilmadi, kayit yok")
        run.flush()
        return
    _finish(run, model, optimizer, sampler, evaluate, record, last_step,
            last_saved_step, started_at)


def _finish(run, model, optimizer, sampler, evaluate, record, last_step,
            last_saved_step, started_at):
    """Kosunun sonu.  ONCE KAYDET, SONRA OLC: tam olcum uzun surebilir; o arada
    oturum duserse son yedekten beri yapilan is giderdi."""
    if last_saved_step != last_step:
        if record is None or record["step"] != last_step:
            record = evaluate(last_step)
        run.save(_snapshot(model, optimizer, sampler, record))
        run.note(_line(record, time.time() - started_at, "  yedek"))
    if run.stop_requested:
        # Durdurmanin amaci GPU'yu HEMEN birakmak; tam olcum ATLANIR.
        run.result = dict(record or run.result, model=model, done=True, stopped=True)
        run.note(f"DURDURULDU  son tamamlanan adim {last_step} kaydedildi"
                 "  -- tam olcum ATLANDI")
        run.flush()
        return
    record = dict(evaluate(last_step, full=True), done=True)
    run.result = dict(record, model=model)
    run.save(_snapshot(model, optimizer, sampler, record))
    run.note(f"BITTI   train {record['train_acc']:.4f}   heldout "
             f"{record['heldout_acc']:.4f}   ppl {_ppl(record['heldout_ce']).strip()}"
             f"   ({time.time() - started_at:.0f} sn)")
    run.flush()


def _run_safe(run, **settings):
    """_run'i sarar: iplik olurse show_log bunu GORSUN, kosu suruyor sanilmasin."""
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


def start(run_name, data, n_vocab, *, metric=_no_metric, device="cuda", root=None,
          extra=None, lr=LR, steps=20000, seed=0, batch=BATCH,
          eval_every=100, save_every=1000, weights_every=100, resume=None,
          compile=True, vocab=None,
          vectors=M18.VECTORS, active=M18.ACTIVE, layers=M18.LAYERS,
          t_max=M18.T_MAX, squared=None, S_p=None, start_norm=M18.START_NORM,
          lam=M18.LAM, chain=M18.CHAIN, c_cache=M18.C_CACHE,
          cache_topk=M18.CACHE_TOPK, cache_skip=M18.CACHE_SKIP, s_v_init=M18.S_V_INIT,
          s_c_init=M18.S_C_INIT, gate_0_init=M18.GATE_0_INIT, s_p_init=M18.S_P_INIT,
          query=M18.QUERY, query_vectors=M18.QUERY_VECTORS, query_active=M18.QUERY_ACTIVE,
         query_by=M18.QUERY_BY, c_content=M18.C_CONTENT,
          d_order=M18.D_ORDER, d_content=M18.D_CONTENT, lam_w_init=M18.LAM_W_INIT, beta_w_init=M18.BETA_W_INIT,
         select=M18.SELECT, load_balance=M18.LOAD_BALANCE):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    data        (questions, filled_mask, targets_mask)
    metric      metric(model, "train"|"heldout", full=False) -> dict: accuracy (ANA OLCUT),
                ce, diag (veriye ozgu analiz olculeri; gunlukte sutun olur)
    eval_every     olcum araligi
    save_every     TAM yedek araligi (t<adim>.pt, surdurme bundan; 3,2 MB)
    weights_every  agirlik kaydi araligi (w<adim>.pt, analiz icin; ~1 MB)
                   Kullanici, 24 Eylul: "bu kadar yedek çok fazla".
                   Uc aralik BAGIMSIZ; HER ADIMIN kaybi hepsinden bagimsiz: `step_losses`.
    resume      bir anlik goruntu yolu verilirse KALDIGI YERDEN devam eder
                (agirlik + optimizer + RNG).  Kural 1: uzatma SURDURMEDIR.
                Paketteki ayar ya da veri bu cagriyla tutmazsa kosu BASLAMAZ.
    vocab       token listesi -- pakete yazilir
    compile     torch.compile, ACIK.  Eski mimaride 3,90 kat olculdu; PV'de OLCULMEDI.
                Ilk adim derleme yuzunden yavas.
    vectors, active, layers, t_max   PV'nin ayarlari (model_18)
    squared, S_p   skor; None ise sozluk sayisindan (model_18.SCORE_BY_VOCAB)
    start_norm  start'larin baslangic boyu (None: randn, boy ~sqrt(d))
    lam         zincirin solma carpani: C_t = lam*C_(t-1) + RM_t*P[w_t]  (1: solmaz)
    chain       "absolute" (RM_t, mutlak konum) ya da "relative" (kaydirma, kelimenin yasi)
    c_cache     hikayenin kendi gecmisinden kopya + ogrenilen gate (model_18.CCache)
    cache_topk, cache_skip   defterden kac komsu (None: butun gecmis); son kac konum aranmaz
    query       defteri Q ile ara; query_vectors, query_active, query_by Q'nun oklari (kapali: Q = C)
    d_order, d_content   nokta uzayi D_SUM = d_order + d_content
    c_content   C = [C_order | C_content]; C_content kaydirmasiz, lam_w/beta_w ile solar
                (lam_w_init, beta_w_init).  Kapali: D_SUM'in tamami relative
    select      aktif vektor secimi: "distance" ya da "direction" (katman 1'den itibaren yon)
    load_balance  kayba eklenen denge terimi katsayisi (0: yok): farkli C'ler farkli vektor kullansin
    s_v_init, s_c_init, gate_0_init, s_p_init   ogrenilen S_v, S_c, gate_0, S_p'nin baslangici
    """
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
        data=data, n_vocab=n_vocab, metric=metric, device=device, lr=lr,
        steps=steps, seed=seed, batch=batch, eval_every=eval_every,
        save_every=save_every, weights_every=weights_every, resume=resume,
        compile=compile, vocab=vocab,
        extra=extra, vectors=vectors, active=active, layers=layers,
        t_max=t_max, squared=squared, S_p=S_p, start_norm=start_norm, lam=lam,
        chain=chain, c_cache=c_cache, cache_topk=cache_topk, cache_skip=cache_skip,
        s_v_init=s_v_init, s_c_init=s_c_init, gate_0_init=gate_0_init, s_p_init=s_p_init,
        query=query, query_vectors=query_vectors, query_active=query_active, query_by=query_by,
        c_content=c_content, d_order=d_order, d_content=d_content, lam_w_init=lam_w_init, beta_w_init=beta_w_init,
        select=select, load_balance=load_balance))
    run.thread.start()
    return f"{run_name} basladi" + (f"  ({os.path.basename(resume)}'den)" if resume else "")


def show_log(last=40):
    """Her kosunun durumu ve son satirlari.  HICBIR SEY KOSTURMAZ (kural 8)."""
    for run in RUNS.values():
        state = ("CANLI" if run.alive else "HATA" if run.result.get("error") else "bitti")
        print(f"{run.name}: {state}   gunluk {len(run.log)} satir"
              + ("   DURDUR istendi" if run.stop_requested else ""))
        for line in run.log[-last:]:
            print(line)


def stop(run_name=None):
    """Dur istegi: adi verilen kosuya, yoksa canli olanlarin hepsine.
    Tam olcum ATLANIR -- durdurmanin amaci GPU'yu hemen birakmak."""
    for run in ([RUNS[run_name]] if run_name else [r for r in RUNS.values() if r.alive]):
        run.stop()
    print("durdurma istendi:", sorted(r.name for r in RUNS.values() if r.stop_requested))
