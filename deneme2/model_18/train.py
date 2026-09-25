# -*- coding: utf-8 -*-
"""train -- PV'nin egitim dongusu: optimizer, batch, kayit, SURDURME (model_17 train_17'den).

GOREV SONRAKI TOKEN, kayip YALNIZ hedeflerde:

    loss = model.loss(tokens, targets_mask)    tokens (B,T) diziler (hikaye ya da soru),
                                                targets_mask kayba sayilacak hedefler

data = (questions, filled_mask, targets_mask): questions (n, T) diziler,
filled_mask gercek token'lar (gunluk ve trim), targets_mask kayba sayilacak
hedefler (hikayede her gercek token, rakam verisinde cevap ve EOS).

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
BATCH = 64       # hikaye kosularinin degeri (defterde ORTAK); rakam defteri kendi degerini verir
RUNS = {}        # run_name -> Run

# Surdurmede paketteki degerle AYNI olmali.  Farkliysa yorunge sessizce
# baskalasir: optimizer.load_state_dict lr'yi paketten alir, gunluk cagriyi yazar.
MUST_MATCH = ("arch", "n", "T", "batch", "lr", "seed", "d_sum", "vectors",
              "active", "layers", "t_max", "squared", "S_p", "start_norm", "lam", "chain", "c_cache", "cache_topk",
              "cache_skip", "s_v_init", "s_c_init", "gate_0_init", "s_p_init", "query", "query_vectors",
              "query_active", "query_by", "c_content", "d_order", "d_content", "lam_w_init", "beta_w_init",
              "content_scalar", "select", "load_balance", "c_m_norm",
              "attention", "attn_heads", "attn_dim", "attn_after", "attn_value",
              "data_fingerprint", "vocab")
# Alan eklenmeden once yazilan paketlerdeki deger: skor -D^2, carpansiz.
BEFORE_FIELD = {"squared": True, "S_p": 1.0, "start_norm": "randn", "lam": 1.0,
                "chain": "absolute", "c_cache": False,
                "cache_topk": M18.CACHE_TOPK, "cache_skip": M18.CACHE_SKIP,
                "s_v_init": 0.0, "s_c_init": 3.0, "gate_0_init": -2.0, "s_p_init": 0.0,
                "query": False, "query_vectors": 0, "query_active": 8, "query_by": "C_m",
                "c_content": False, "d_order": 0, "d_content": 0, "select": "distance", "load_balance": 0.0, "c_m_norm": False,
                "lam_w_init": 0.9, "beta_w_init": 0.5, "content_scalar": False, "attention": False, "attn_heads": 0,
                "attn_dim": 0,
                "attn_after": 0, "attn_value": "state"}

# HIZ, hesap AYNI (tests: t_speed): batch en uzun hikayesine kirpilir (nedensel, dolgu sagda), puan
# tablosu yalniz sayilan hedeflerde kurulur.  Boylar SHAPES basamaga yuvarlanir: torch.compile her
# batch'te yeniden derlemesin.  Kullanici, 24 Eylul: "Düzelt bunları".
SHAPES = 8


def trim(questions, filled_mask, targets_mask, shapes=SHAPES):
    """CPU'daki batch -> (tokens, targets_mask, rows): en uzun hikayeye kirpilmis, rows sayilan hedeflerin
    duz konumlari (B*(T-1)), -1 ile yuvarlanmis.  GPU'ya dokunmaz: senkron yok."""
    B, T = questions.shape
    step = max(1, T // shapes)
    real = (filled_mask.bool() | targets_mask.bool()).any(0).nonzero()
    L = min(T, -(-(int(real.max()) + 1 if len(real) else 1) // step) * step)
    tokens, mask = questions[:, :L], targets_mask[:, :L]
    rows = mask[:, 1:].bool().reshape(-1).nonzero()[:, 0]
    unit = max(1, B * T // shapes ** 2)                  # satir sayisi bu birime yuvarlanir
    R = -(-max(1, len(rows)) // unit) * unit
    rows = torch.cat([rows, rows.new_full((R - len(rows),), -1)])
    return tokens, mask, rows


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
            "c_content": ("d_order", "d_content", "lam_w_init", "beta_w_init", "content_scalar"),
            "attention": ("attn_heads", "attn_dim", "attn_after", "attn_value")}


def _check_resume(package, fixed):
    """Surdurme paketi bu cagriyla AYNI kosu mu?  Degilse kosu BASLAMAZ."""
    short = lambda v: f"<{len(v)} birim>" if isinstance(v, list) else repr(v)
    packed = {a: package.get(a, BEFORE_FIELD.get(a)) for a in MUST_MATCH}
    packed["d_sum"] = package.get("d_sum", package.get("d"))        # eski paketin 'd' alani
    off = {a for s, subfields in SWITCHED.items() if not packed[s] and not fixed.get(s) for a in subfields}
    diffs = [f"{a}: paket {short(packed[a])}, cagri {short(fixed[a])}"
             for a in MUST_MATCH
             if a not in off and packed[a] is not None and fixed.get(a) is not None
             and packed[a] != fixed[a]]
    if diffs:
        raise ValueError("surdurme paketi bu cagriyla uyusmuyor -- "
                         + "; ".join(diffs))
    # LR sogutma plani: planli paket AYNI planla surer; plansiz (sabit LR) paketten yeni planla dal acilabilir.
    plan = lambda k: (k.get("decay_start"), k.get("decay_end"), k.get("decay_floor"))
    if package.get("decay_start") is not None and plan(package) != plan(fixed):
        raise ValueError("surdurme paketi bu cagriyla uyusmuyor -- LR sogutma plani (baslangic, son, taban): "
                         "paket %s, cagri %s" % (plan(package), plan(fixed)))


def _header(record):
    """Sutun adlari; diag sutunlari verinin metric'inden gelir."""
    return ("  step     loss    train    heldout      ppl  "
            + "".join(f"{k:>12}" for k in record["heldout_diag"]) + "       s")


def _line(record, elapsed, mark=""):
    """elapsed: baslangictan beri sn (olcum dahil); olcum: bu olcum noktasinin kendi suresi."""
    s = record.get("eval_sec")
    return (f"{record['step']:6d}  {record['loss']:7.3f}  {record['train_acc']:7.4f}  "
            f"{record['heldout_acc']:9.4f}  {_ppl(record['heldout_ce'])}  "
            + "".join(f"{v:12.4f}" for v in record["heldout_diag"].values())
            + f"  {elapsed:6.0f}{mark}" + (f"  olcum {s['train'] + s['heldout']:.1f} sn" if s else "")
            + (f"  lr {record['lr_now']:.2e}" if record.get("decay_start") is not None else ""))


# Parca basina gradyan normu, yalniz olcum adimlarinda: olu parca (0) ve patlama gorunsun.
# Olculdu (24 Eylul): top-8'de secilmeyen vektor hic gradyan almiyordu; 6 parca sonra fark edildi.
GRAD_PARTS = (("V.", "vektor"), ("lam_w", "icerik"), ("beta_w", "icerik"), ("attn.", "attn"), ("cache.", "defter"))


def _grad_norms(model):
    top = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            part = next((label for prefix, label in GRAD_PARTS if name.startswith(prefix)), "olcek")
            top[part] = top.get(part, 0) + p.grad.detach().float().pow(2).sum()
    return {part: float(v.sqrt()) for part, v in top.items()}


def _health_line(h, grads=None, timing=None):
    """Saglik sozlugu -> tek satir (gunluk, NABIZ): vektor her/butun konum, olu, yon, boy, olcekler,
    attention, cikis, gradyan; tam yedekte dongu; olcumun suresi (heldout, saglik ve donguyu icerir)."""
    L = sum(1 for k in h if k.startswith("vec_all_"))
    mean = lambda x: sum(x) / len(x)
    s = ["saglik  vektor " + " ".join("%.0f/%.0f" % (h["vec_each_%d" % i], h["vec_all_%d" % i]) for i in range(L)),
         "olu " + "/".join(str(h["dead_%d" % i]) for i in range(L)),
         "yon C %d C_m %d" % (h["dir_C"], h["dir_Cm"]),
         "boy C %.2f C_m %.2f" % (h["norm_C"], h["norm_Cm"])]
    if "exp_S_p" in h:
        s.append("e^S_p %.1f" % h["exp_S_p"])
    if "gate" in h:
        s.append("gate %.3f" % h["gate"])
    if "lam_c" in h:
        s.append("lam_c %.2f" % h["lam_c"])
    if "attn_ent" in h:
        s.append("attn ent %.2f bas0 %.2f mesafe %.1f"
                 % (mean(h["attn_ent"]), mean(h["attn_first"]), mean(h["attn_dist"])))
    s.append("cikis %d kelime ent %.2f" % (h["pred_distinct"], h["pred_ent"]))
    if grads:
        s.append("grad " + " ".join("%s %.2g" % kv for kv in sorted(grads.items())))
    if "distinct4" in h:
        s.append("farkli4 %.2f dongu %.2f" % (h["distinct4"], h["loop"]))
    if timing:
        extras = " ".join("%s %.1f" % (k, timing[k]) for k in ("health", "loop") if k in timing)
        s.append("olcum sn train %.1f heldout %.1f%s" % (timing["train"], timing["heldout"], " (%s)" % extras if extras else ""))
    return "  |  ".join(s)


def _note(run, record, elapsed, mark=""):
    """Olcum satiri ve (varsa) saglik satiri."""
    run.note(_line(record, elapsed, mark))
    if record.get("health") is not None:
        run.note(_health_line(record["health"], record.get("grads"), record.get("eval_sec")))


def _health_summary(history):
    """Bu bolumde ilk olu vektor ve ilk dongu hangi adimda goruldu, son olcumde ne durumda.  Adim 0
    sayilmaz: egitilmemis model acgozlu uretimde hep doner, ozet hep "adim 0" derdi."""
    h = [(s, x) for s, x in history if s > 0]
    dead_list = lambda x: [x["dead_%d" % i] for i in range(sum(1 for k in x if k.startswith("dead_")))]
    dead = next((s for s, x in h if any(dead_list(x))), None)
    loop = next((s for s, x in h if x.get("loop", 0) > 0), None)
    last_loop = next((x["loop"] for _, x in reversed(h) if "loop" in x), None)
    return "saglik ozeti (bu bolum, adim 0 haric): ilk olu vektor %s (son %s)  |  ilk dongu %s (son %s)" % (
        "yok" if dead is None else "adim %d" % dead, "/".join(map(str, dead_list(h[-1][1]))) if h else "-",
        "yok" if loop is None else "adim %d" % loop, "-" if last_loop is None else "%.2f" % last_loop)


def _run(run, data, n_vocab, metric, device, lr, steps, seed, batch,
         eval_every, save_every, weights_every=100, resume=None, compile=False, vocab=None,
         extra=None, vectors=M18.VECTORS, active=M18.ACTIVE,
         layers=M18.LAYERS, t_max=M18.T_MAX, squared=None, S_p=None,
         start_norm=M18.START_NORM, lam=M18.LAM, chain=M18.CHAIN, c_cache=M18.C_CACHE,
         cache_topk=M18.CACHE_TOPK, cache_skip=M18.CACHE_SKIP, s_v_init=M18.S_V_INIT,
         s_c_init=M18.S_C_INIT, gate_0_init=M18.GATE_0_INIT, s_p_init=None,
         query=M18.QUERY, query_vectors=M18.QUERY_VECTORS, query_active=M18.QUERY_ACTIVE,
         query_by=M18.QUERY_BY, c_content=M18.C_CONTENT,
         d_order=M18.D_ORDER, d_content=M18.D_CONTENT, lam_w_init=M18.LAM_W_INIT, beta_w_init=M18.BETA_W_INIT,
         content_scalar=M18.CONTENT_SCALAR,
         select=M18.SELECT, load_balance=M18.LOAD_BALANCE, c_m_norm=M18.C_M_NORM, attention=M18.ATTENTION,
         attn_heads=M18.ATTN_HEADS, attn_dim=M18.ATTN_DIM, attn_after=M18.ATTN_AFTER, attn_value=M18.ATTN_VALUE,
         decay_start=None, decay_floor=0.1):
    torch.manual_seed(seed)
    model = PV(n_vocab, vectors=vectors, active=active, layers=layers,
               t_max=t_max, seed=seed, squared=squared, S_p=S_p,
               start_norm=start_norm, lam=lam, chain=chain, c_cache=c_cache,
               cache_topk=cache_topk, cache_skip=cache_skip, s_v_init=s_v_init,
               s_c_init=s_c_init, gate_0_init=gate_0_init, s_p_init=s_p_init,
               query=query, query_vectors=query_vectors, query_active=query_active, query_by=query_by,
               c_content=c_content, d_order=d_order, d_content=d_content, lam_w_init=lam_w_init, beta_w_init=beta_w_init,
               content_scalar=content_scalar,
               select=select, load_balance=load_balance, c_m_norm=c_m_norm, attention=attention,
               attn_heads=attn_heads, attn_dim=attn_dim, attn_after=attn_after, attn_value=attn_value).to(device)
    # torch.compile: eski mimaride 3,90 kat olculdu (model_17 train_17); PV'de OLCULMEDI.
    loss_fn = torch.compile(model.loss) if compile else model.loss
    if compile:
        # trim: en cok SHAPES farkli boy; her biri bir derleme (C_content dongusu boya ozgulenir).
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 2 * SHAPES)
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
                 gate_0_init=float(gate_0_init), s_p_init=model.s_p_init,
                 query=bool(query), query_vectors=int(query_vectors), query_active=int(query_active),
                 query_by=query_by, c_content=bool(c_content), d_order=model.d_order, d_content=model.d_content, lam_w_init=float(lam_w_init), beta_w_init=float(beta_w_init),
                 content_scalar=bool(content_scalar), select=select,
                 load_balance=float(load_balance), c_m_norm=bool(c_m_norm), attention=bool(attention),
                 attn_heads=int(attn_heads), attn_dim=int(attn_dim), attn_after=int(attn_after), attn_value=attn_value,
                 seed=seed, batch=batch, T=max_length, n_params=n_params,
                 compile=compile,
                 decay_start=None if decay_start is None else int(decay_start),
                 decay_end=None if decay_start is None else int(steps),
                 decay_floor=None if decay_start is None else float(decay_floor),
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
             f"start_norm {fixed['start_norm']}  lam {fixed['lam']}  chain {chain}  c_cache {c_cache} (topk {cache_topk} skip {cache_skip} Q {query} {query_vectors}/{query_active} {query_by})  c_content {c_content} (lam_w {lam_w_init} beta_w {beta_w_init}, {'kelime basina tek deger' if content_scalar else 'kelime x boyut'})  select {select}  load_balance {load_balance}  c_m_norm {c_m_norm} (S_p {model.s_p_init:.3f})  attention {attention} ({attn_heads}x{attn_dim} katman {attn_after}'dan sonra, {attn_value})  lr {lr} seed {seed}  sozluk {n_vocab}  "
             f"parametre {n_params:,}")
    run.note(f"batch {batch}   epok = {n_questions / batch:,.0f} adim   veri izi "
             f"{fixed['data_fingerprint']}   olcum her {eval_every}   tam yedek her "
             f"{save_every}   agirlik her {weights_every}"
             + (f"   LR SOGUTMA cosine {decay_start} -> {steps}, taban lr x {decay_floor}" if decay_start is not None else ""))
    run.note("OLCUT: accuracy (train / heldout); kayip yalniz hedeflerde")
    run.flush()

    history = []                          # (adim, saglik): kosu sonu ozeti

    def evaluate(step, full=False, save=False, grads=None, health=None):
        """save: tam yedek adimi -- saglik olcutu varsa (metric.health) dongu ve metin de.
        health: ayni agirlikla olculmus saglik sozlugu; yeniden hesaplanmaz (kosu sonu).
        eval_sec: bu olcumun suresi -- train, heldout (saglik ve dongu dahil) ve parcalari."""
        kw = {"save": save, "health": health} if getattr(metric, "health", False) else {}
        t0 = time.time()
        train_result = metric(model, "train", full=full)
        t1 = time.time()
        heldout_result = metric(model, "heldout", full=full, **kw)
        timing = {"train": t1 - t0, "heldout": time.time() - t1}
        timing.update((k, v) for k, v in (heldout_result.get("timing") or {}).items() if k != "eval")
        record = dict(fixed, step=step, loss=float(loss.detach()),
                      step_losses=step_losses[:step + 1].cpu(),
                      train_acc=train_result["accuracy"],
                      heldout_acc=heldout_result["accuracy"],
                      train_ce=train_result.get("ce"), heldout_ce=heldout_result.get("ce"),
                      train_diag=train_result.get("diag", {}),
                      heldout_diag=heldout_result.get("diag", {}),
                      health=heldout_result.get("health"), grads=grads, eval_sec=timing,
                      lr_now=optimizer.param_groups[0]["lr"])
        if not run.result:
            run.note(_header(record))
        run.result = dict(record, model=model)
        if record["health"] is not None:
            history.append((step, record["health"]))
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
        # total: geri yayilan (NLL + load balance); loss: NLL -- gunluk ve step_losses bunu yazar,
        # kosular denge teriminden bagimsiz kiyaslanir.
        tokens, mask, rows = trim(questions[batch_ids], filled_mask[batch_ids], targets_mask[batch_ids])
        total, loss = loss_fn(tokens.to(device, non_blocking=True).long(), mask.to(device, non_blocking=True),
                              parts=True, rows=rows.to(device, non_blocking=True))
        step_losses[step] = loss.detach()
        total.backward()
        # Uc aralik birbirinden BAGIMSIZ: olcum, tam yedek (surdurme), agirlik (analiz).
        full_save, weights_save = step % save_every == 0, step % weights_every == 0
        measure = step % eval_every == 0 or full_save or weights_save
        grads = _grad_norms(model) if measure else None       # yalniz olcum adiminda senkron
        if decay_start is not None:                            # cosine: decay_start'ta lr, steps'te lr x taban
            k = min(1.0, max(0.0, (step - decay_start) / max(1, steps - decay_start)))
            for g in optimizer.param_groups:
                g["lr"] = lr * (decay_floor + (1 - decay_floor) * 0.5 * (1 + math.cos(math.pi * k)))
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

    if last_step < 0:
        run.note("DURDURULDU -- hic adim atilmadi, kayit yok")
        run.flush()
        return
    _finish(run, model, optimizer, sampler, evaluate, record, last_step,
            last_saved_step, started_at, history)


def _finish(run, model, optimizer, sampler, evaluate, record, last_step,
            last_saved_step, started_at, history=()):
    """Kosunun sonu.  ONCE KAYDET, SONRA OLC: tam olcum uzun surebilir; o arada
    oturum duserse son yedekten beri yapilan is giderdi."""
    if last_saved_step != last_step:
        noted = record is not None and record["step"] == last_step       # dongude olculdu, satiri yazildi
        if not noted:
            record = evaluate(last_step)
        run.save(_snapshot(model, optimizer, sampler, record))
        if noted:
            run.note(f"yedek t{last_step}  (son adim; olcum satiri yukarida)")
        else:
            _note(run, record, time.time() - started_at, "  yedek")
    if run.stop_requested:
        # Durdurmanin amaci GPU'yu HEMEN birakmak; tam olcum ATLANIR.
        run.result = dict(record or run.result, model=model, done=True, stopped=True)
        run.note(f"DURDURULDU  son tamamlanan adim {last_step} kaydedildi"
                 "  -- tam olcum ATLANDI")
        run.flush()
        return
    # Son adimin sagligi ayni agirlikla zaten olculduyse tam olcumde yeniden hesaplanmaz.
    prior = record.get("health") if record is not None and record["step"] == last_step else None
    record = dict(evaluate(last_step, full=True, grads=(record or {}).get("grads"), health=prior),   # son adimin gradyani
                  done=True)
    run.result = dict(record, model=model)
    run.save(_snapshot(model, optimizer, sampler, record))
    s = record["eval_sec"]
    run.note(f"BITTI   train {record['train_acc']:.4f}   heldout "
             f"{record['heldout_acc']:.4f}   ppl {_ppl(record['heldout_ce']).strip()}"
             f"   ({time.time() - started_at:.0f} sn, tam olcum {s['train'] + s['heldout']:.0f} sn)")
    if record.get("health") is not None and record["health"] != prior:     # ayniysa satiri zaten yukarida
        run.note(_health_line(record["health"]))
    if history:
        run.note(_health_summary(history))
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
          s_c_init=M18.S_C_INIT, gate_0_init=M18.GATE_0_INIT, s_p_init=None,
          query=M18.QUERY, query_vectors=M18.QUERY_VECTORS, query_active=M18.QUERY_ACTIVE,
         query_by=M18.QUERY_BY, c_content=M18.C_CONTENT,
          d_order=M18.D_ORDER, d_content=M18.D_CONTENT, lam_w_init=M18.LAM_W_INIT, beta_w_init=M18.BETA_W_INIT,
          content_scalar=M18.CONTENT_SCALAR,
         select=M18.SELECT, load_balance=M18.LOAD_BALANCE, c_m_norm=M18.C_M_NORM, attention=M18.ATTENTION,
          attn_heads=M18.ATTN_HEADS, attn_dim=M18.ATTN_DIM, attn_after=M18.ATTN_AFTER, attn_value=M18.ATTN_VALUE,
          decay_start=None, decay_floor=0.1):
    """ARKA PLANDA baslatir, HEMEN doner (kural 8).

    data        (questions, filled_mask, targets_mask)
    metric      metric(model, "train"|"heldout", full=False) -> dict: accuracy (ANA OLCUT),
                ce, diag (veriye ozgu analiz olculeri; gunlukte sutun olur)
    eval_every     olcum araligi
    save_every     TAM yedek araligi (t<adim>.pt, surdurme bundan: agirlik + optimizer + RNG)
    weights_every  agirlik kaydi araligi (w<adim>.pt, analiz icin: yalniz agirlik)
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
                (lam_w_init, beta_w_init; content_scalar: kelime basina tek deger).  Kapali: D_SUM'in tamami relative
    select      aktif vektor secimi: "distance", "direction" (katman 1'den itibaren yon) ya da "direction_all"
    load_balance  kayba eklenen denge terimi katsayisi (0: yok): farkli C'ler farkli vektor kullansin
    c_m_norm    puan C_m/|C_m| ile (kelimelerin kuresi); s_p_init None iken C_M_NORM_P formulunden
    attention   sozluk katmani attn_after'den sonra Gecmisten (attn_heads x attn_dim; attn_value "state" / "point")
    s_v_init, s_c_init, gate_0_init, s_p_init   ogrenilen S_v, S_c, gate_0, S_p'nin baslangici
    decay_start LR sogutmasi (None: sabit LR): decay_start'tan steps'e cosine, lr'den lr x decay_floor'a
                (CLAUDE.md kural 4'un varsayilani: taban lr/10).  Plansiz pakette yeni planla dal acilabilir.
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
        content_scalar=content_scalar,
        select=select, load_balance=load_balance, c_m_norm=c_m_norm, attention=attention, attn_heads=attn_heads,
        attn_dim=attn_dim, attn_after=attn_after, attn_value=attn_value,
        decay_start=decay_start, decay_floor=decay_floor))
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
