# -*- coding: utf-8 -*-
"""train_y -- model_y GENEL egitim: veriden bagimsiz.  Veriye ozgu sinav ve raporlar egitim klasorlerinde (train_<veri>/).
    train_seq   dizi egitimi: Model Y (BlockModel); yedek ve surdurme
    Muon        optimizer: gizli matrisler Muon, gerisi Adam (tek sinif, tek state_dict)
    pad         id listeleri -> (ids, mask);  generate: acgozlu uretim
"""
import contextlib
import copy
import math
import threading

import torch
import torch._inductor.config
from torch.nn.attention import SDPBackend, sdpa_kernel

from model_y import CANON, NORMALIZED_UPDATE, ROPE, SPHERE_WEIGHTS, AttentionCache, BlockModel, deviation

STEPS, LR = 4000, 0.01  # lr dayanagi (kure agirliklari + Muon): adim basina donme ~ LR x 0,2 x sqrt(d); nGPT 2026 tepe lr
                         # 0,24 / sqrt(d).  Veri / batch / D degisince yeniden hesaplanir
LOG_AT = (0, 10, 50, 200, 500, 1000)
LR_FLOOR = 0.0       # inisin tabani: lr sonda LR x LR_FLOOR (wsd, coherence)
GRAD_CLIP = 1.0      # gradient clipping: adimdaki gradient'in boyu bunu gecerse buna indirilir
WEIGHT_DECAY = 0.0   # weight decay (AdamW), yalniz W_ matrislerine; 0 = kapali
OPTIMIZER = "muon"   # "muon": gizli matrisler (W_context, W_value, W_fact_in, W_fact_up, W_fact_out) Muon, gerisi Adam |
                     # "adam": hepsi Adam
SCHEDULE = "coherence"   # "coherence": asagida | "wsd": lr sabit, son COOLDOWN kisminda 1 - sqrt ile LR x LR_FLOOR'a
# Inductor'un bellek yerlesimi analizi dinamik sekilde (bucket) ic kontrolde patladi; yalniz tiling sezgisi, kapatmak sonucu
# degistirmez
torch._inductor.config.triton.coalesce_tiling_analysis = False
COMPILE_LOCK = threading.Lock()   # torch.compile iplikler arasi guvenli degil: ileri hesap bu kilit altinda
COOLDOWN = 0.2       # WSD'de inisin payi (son %20).  Hagele 2024: <= %20 yeter, 1 - sqrt dogrusaldan iyi
# SCHEDULE "coherence": lr = LR x ortalama(rho), rho = adimin sinyal payi, batch'in iki yarisinin gradyanlarindan olculur
# (g1, g2; N >> B):
#   rho = g1.g2 / (g1.g2 + |g1 - g2|^2 / 4)        (= 2c / (1 + c), c = cos(g1, g2), esit boylarda)
# kurede adim = aci: theta = LR x 0,2 x sqrt(d) x rho.  Tam batch'te (ids verilmis) gurultu yok: rho = 1, olculmez.
COHERENCE_WINDOW = 200   # rho'nun hareketli ortalamasi (adim); tek adimin olcumu gurultulu
FINAL_COOLDOWN = 0.95    # coherence'ta inisin payi: ilk %5'te lr olculur, sonra o degerden x LR_FLOOR'a
FINAL_COOLDOWN_SHAPE = "log"      # son inisin bicimi: "linear" 1 - p | "sqrt" 1 - sqrt(p) | "log" 1 - log(1 + p/k) /
                                  # log(1 + 1/k) (nGPT 2026, onden yuklu)
LOG_COOLDOWN_KAPPA = 0.05          # "log" inisin k'si: kucuk k basta hizli dusus, uzun kuyruk
WEIGHT_EMA = 0.999      # agirliklarin hareketli ortalamasi: titresimi siler.  None = kapali
MATMUL_PRECISION = "bf16"   # egitim hesabi: "fp32" | "bf16" (ileri hesap autocast; parametre, gradyan, optimizer, Newton-Schulz
                            # ve weight EMA fp32).  Sinav hep fp32.  Yalniz GPU'da
NEWTON_SCHULZ_PRECISION = "fp32"   # Muon'un ortogonallestirmesi: "fp32" | "bf16"
MICRO_BATCHES = 1        # adimin batch'i bu kadar parcada (satirlar s::k) ileri + geri, gradyan birikir: bellek bir parcalik,
                         # adim tam batch'in gradyanini alir (parca kaybi hedef payiyla agirlikli).  coherence'ta cift sayi
                         # (yari basina k / 2 parca)


class Muon(torch.optim.Optimizer):
    """Muon (K. Jordan 2024, MIT) ve ayni sinifta Adam.  use_muon=True gruplar: momentum (Nesterov) -> Newton-Schulz ile
    ortogonallestirme (5 adim, fp32) -> 0,2 x sqrt(max(satir, sutun)) olcegi (Liu 2025: boylece AdamW'nin lr'si aynen
    kullanilir).  use_muon=False gruplar: torch.optim.Adam ile ayni formul (beta 0,9 / 0,999, eps 1e-8)."""

    def __init__(self, groups, lr, momentum=0.95, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(groups, dict(lr=lr, momentum=momentum, betas=betas, eps=eps, use_muon=False))
        self.tangent_axis = {}      # parametre -> kurede birim eksen (1 satir, 0 sutun): Nesterov birlesimi once kure
                                    # tegetine izdusulur, sonra ortogonallestirilir; bos = izdusum yok
        self.newton_schulz_precision = "fp32"   # NEWTON_SCHULZ_PRECISION (train_seq kurar)

    @staticmethod
    def orthogonalize(G, steps=5, precision="fp32"):
        """G'ye en yakin yari-ortogonal matris (tekil degerler ~1), besinci derece Newton-Schulz; G ayni boyda matrislerin
        yigini da olabilir (... x m x n, her matris ayri).  precision "bf16": dongu bf16'da, sonuc G'nin tipinde."""
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        if precision == "bf16":
            X = X.bfloat16()
        tall = X.shape[-2] > X.shape[-1]
        if tall:
            X = X.mT
        for _ in range(steps):
            A = X @ X.mT
            X = a * X + (b * A + c * A @ A) @ X
        return (X.mT if tall else X).to(G.dtype)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if g["use_muon"]:
                by_shape = {}                               # ayni boydaki matrisler tek Newton-Schulz cagrisinda: sure
                for p in g["params"]:                       # hesapta degil cagri sayisinda
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(g["momentum"]).add_(p.grad)
                    nesterov = p.grad.add(buf, alpha=g["momentum"])
                    axis = self.tangent_axis.get(p)
                    if axis is not None:                    # kure tegeti: satir / sutun basina v - <v, w> w
                        nesterov = nesterov - (nesterov * p).sum(axis, keepdim=True) * p
                    by_shape.setdefault(tuple(p.shape), []).append((p, nesterov))
                for items in by_shape.values():
                    updates = self.orthogonalize(torch.stack([v for _, v in items]), precision=self.newton_schulz_precision)
                    for (p, _), update in zip(items, updates):
                        p.add_(update, alpha=-g["lr"] * 0.2 * max(p.shape) ** 0.5)
                continue
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"], state["exp_avg_sq"] = torch.zeros_like(p), torch.zeros_like(p)
                state["step"] += 1
                (b1, b2), m, v = g["betas"], state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denom = (v.sqrt() / math.sqrt(1 - b2 ** state["step"])).add_(g["eps"])
                p.addcdiv_(m, denom, value=-g["lr"] / (1 - b1 ** state["step"]))


class _Gradients:
    """Birikmis gradyanlar, backward arayuzuyle: backward() p.grad'a ekler (yoksa yazar) -- train_seq'in split / tek
    batch yolu autograd'in birikimiyle ayni toplamayi gorur."""

    def __init__(self, params, grads):
        self.params, self.grads = params, grads

    def backward(self):
        for p, g in zip(self.params, self.grads):
            if g is not None:
                p.grad = g.clone() if p.grad is None else p.grad.add_(g)


# ---- dizi egitimi

def pad(rows, vocab):
    """Id listeleri -> (ids, mask), sagdan <pad>; nedensel attention'da sagdaki dolgu oncekileri etkilemez."""
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), vocab.index("<pad>"), dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        mask[i, :len(r)] = True
    return ids, mask


def train_seq(setting, ids, mask, n, steps=STEPS, lr=LR, log_at=LOG_AT, seed=0, lr_floor=LR_FLOOR, grad_clip=GRAD_CLIP,
              weight_decay=WEIGHT_DECAY, device="cpu", every=None, callback=None, compile=True,
              save_every=None, save=None, checkpoint=None, rope=ROPE,
              batches=None, model_kw=None, optimizer=OPTIMIZER, schedule=SCHEDULE, cooldown=COOLDOWN,
              normalized_update=NORMALIZED_UPDATE, sphere_weights=SPHERE_WEIGHTS, canon=CANON,
              coherence_window=COHERENCE_WINDOW,
              final_cooldown=FINAL_COOLDOWN, weight_ema=WEIGHT_EMA, matmul_precision=MATMUL_PRECISION,
              final_cooldown_shape=FINAL_COOLDOWN_SHAPE, newton_schulz_precision=NEWTON_SCHULZ_PRECISION,
              micro_batches=MICRO_BATCHES):
    """Tarif: Muon (gizli matrisler) + Adam, takvim (wsd ya da coherence), gradient clipping.  weight_decay > 0: AdamW,
    yalniz W_ matrisleri (yalniz adam).
    callback(step, model, nll): her `every` adimda, o adimin guncellemesinden ONCE (sinav, kayit, durdurma).
    compile: kayip hesabi (ileri + geri) torch.compile ile; varsayilan acik, yalniz GPU'da uygulanir (CPU'da kendiliginden
    kapanir).
    setting: "shared" (BlockModel).  rope: attention'da RoPE.
    save(step, model, opt): her save_every adimda, callback'ten sonra, guncellemeden ONCE -- adim s paketi s guncelleme
    gormus modeli ve optimizer'i tasir.  callback hata atarsa (durdurma dahil) o adimin paketi de yazilir.
    checkpoint {step, model, optimizer}: o adimdan surdurur (ayni steps ve tarifle kesintisiz kosuyla bit duzeyinde ayni --
    CPU'da; GPU'da compile token gradyanini atomik toplar, son bitler oynayabilir); o adimin callback'i ve kaydi tekrarlanmaz.
    batches(step) -> (ids, mask): buyuk veri icin her adimda bir parca (mini-batch); verilirse ids, mask kullanilmaz (None
    olabilir).  Adimin fonksiyonu olmali (surdurmede ayni parca gelsin).  Sekil adimdan adima degisebilir (bucket):
    compile ikinci sekilde dinamik sekilli tek grafige gecer.  (ids, mask, document_positions): paketli pencere
    (BlockModel.loss).
    model_kw: modele gecen ayarlar (d, turns, layers, units, t_max ...).
    schedule="coherence": lr = lr x ortalama(rho) (COHERENCE_WINDOW), alt sinir lr_floor; son final_cooldown kisminda
    final_cooldown_shape ile x lr_floor'a.  Mini-batch'te adim iki yarida hesaplanir (satirlar tek / cift), birlestirilen
    gradyan tam batch'inkiyle ayni; ortalama optimizer'in grup kaydinda (checkpoint'e girer).  model.coherence: son olcum.
    Ortalama yansiz: pay ve payda ayri hareketli ortalama, ortalama(rho) = EMA(dot+) / EMA(dot+ + |g1 - g2|^2 / 4) (oranlarin
    ortalamasi dusuk rankli gurultude yukari yanli).
    Muon'da sphere_weights: Nesterov birlesimi once kure tegetine izdusulur (Muon.tangent_axis).
    weight_ema=d: her adimdan sonra ortalama <- d x ortalama + (1 - d) x agirlik (kurede satirlar yeniden birim);
    model.weight_ema = dict(model=<ortalama model>, decay=d).  Ortalama optimizer durumunda (checkpoint'e girer).
    matmul_precision: "bf16" ileri hesap ve kayip autocast (bf16) icinde, geri yayilim disinda; "fp32" hicbir seye
    dokunmaz.  CPU'da (compile gibi) etkisiz.  SDPA hep math cekirdegiyle: flash / mem-efficient'in geri yayilimi
    deterministik degil (surdurme bit duzeyinde kalmaz).
    micro_batches: gradyan birikimi (batch'in satir sayisi bunun kati olmali)."""
    assert batches is not None or ids is not None, "ids/mask ya da batches verilmeli"
    assert setting == "shared", "setting: yalniz shared (BlockModel), bu: %s" % setting
    assert optimizer in ("muon", "adam") and schedule in ("wsd", "coherence") and 0 < cooldown <= 1
    assert schedule != "coherence" or (0 < final_cooldown <= 1 and coherence_window >= 1)
    assert final_cooldown_shape in ("sqrt", "linear", "log"), "final_cooldown_shape: sqrt | linear | log"
    assert weight_ema is None or 0 < weight_ema < 1, "weight_ema: 0 ile 1 arasi (ornek 0,999) ya da None"
    assert optimizer == "adam" or not weight_decay, "weight_decay yalniz optimizer='adam' ile (AdamW)"
    assert matmul_precision in ("fp32", "bf16"), "matmul_precision: fp32 | bf16"
    assert newton_schulz_precision in ("fp32", "bf16"), "newton_schulz_precision: fp32 | bf16"
    assert micro_batches >= 1 and (schedule != "coherence" or micro_batches % 2 == 0 or micro_batches == 1), \
        "micro_batches: >= 1; coherence'ta 1 ya da cift"
    model = BlockModel(n, seed=seed, rope=rope, normalized_update=normalized_update, sphere_weights=sphere_weights,
                       canon=canon, **(model_kw or {})).to(device)
    if ids is not None:
        ids, mask = ids.to(device), mask.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    unit_axis = {}                                         # kure tegetinin tek tablosu (coherence ve Muon)
    for k, _ in named:
        kind = k.split(".")[-1]
        if sphere_weights and kind in ("W_query", "W_key", "W_fact_in", "W_fact_up", "W_value"):   # girdisi durum: satir
            unit_axis[k] = 1
        elif sphere_weights and kind in ("W_context", "W_fact_out"):                               # duruma yazan: sutun
            unit_axis[k] = 0
    if optimizer == "muon":
        # Muon yalniz gizli 2 boyutlu matrislerde ("VO + FFN" duzeni, Wang 2025); token noktalari, esikler, W_query, W_key,
        # alpha'lar, Canon, cikis olcegi ve bag Adam'da
        hidden = ("W_context", "W_value", "W_fact_in", "W_fact_up", "W_fact_out")
        opt = Muon([dict(params=[p for k, p in named if k.endswith(hidden)], use_muon=True),
                    dict(params=[p for k, p in named if not k.endswith(hidden)], use_muon=False)], lr=lr)
        opt.tangent_axis = {p: unit_axis[k] for k, p in named if k in unit_axis}   # yalniz Muon grubunda uygulanir
        opt.newton_schulz_precision = newton_schulz_precision
    elif weight_decay:
        # her adimda once W <- W - lr · weight_decay · W (kaybin desteklemedigi agirlik soner), sonra Adam adimi.
        # shift'in capasi var, fact_threshold bir esik: ikisine uygulanmaz
        opt = torch.optim.AdamW([dict(params=[p for k, p in named if k.split(".")[-1].startswith("W_")], weight_decay=weight_decay),
                                 dict(params=[p for k, p in named if not k.split(".")[-1].startswith("W_")], weight_decay=0.0)],
                                lr=lr)
    else:
        opt = torch.optim.Adam(params, lr=lr)
    first = 0
    if checkpoint is not None:                             # surdurme: agirlik + Adam momentleri + adim
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        first = checkpoint["step"]
    if torch.device(device).type != "cuda":               # CPU: compile C++ derleyicisi ister (bu makinede yok) -> kapali
        compile = False
        matmul_precision = "fp32"                          # CPU'da etkisiz: autocast yalniz GPU'da
    kernels = [SDPBackend.MATH]                            # deterministik geri yayilim: surdurme bit duzeyinde
    loss_fn = torch.compile(model.loss) if compile else model.loss
    static_loss = [False]                                  # paketli (sabit boy) batch gelince bir kez: dynamic=False
    start = round((1 - cooldown) * steps)                  # WSD: inis bu adimda baslar
    final = round((1 - final_cooldown) * steps)            # coherence: son inis bu adimda baslar
    split = schedule == "coherence" and batches is not None  # tam batch'te gurultu yok: rho = 1, olculmez

    def forward_context():
        """Ileri hesabin baglami: compile kilidi, SDPA cekirdegi, bf16 autocast."""
        stack = contextlib.ExitStack()
        if compile:
            stack.enter_context(COMPILE_LOCK)
        stack.enter_context(sdpa_kernel(kernels))
        if matmul_precision == "bf16":
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        return stack

    def accumulate(ids, mask, packed):
        """micro_batches > 1: parca s = satirlar s::k (k = micro_batches) ayri ileri + geri; parca kaybi grubun hedef
        payiyla agirlikli.  Grup: coherence'ta iki yari (h + 2r, satirlar h::2'nin parcalari), degilse hepsi.  -> grup
        basina (_Gradients, nll), hedef sayilari, nll: split yolunun (halves, weights) ve tek batch'in (total) yerine."""
        k = micro_batches
        groups = [[h + 2 * r for r in range(k // 2)] for h in (0, 1)] if split else [list(range(k))]
        out, counts = [], []
        for group in groups:
            count = sum(mask[s::k, 1:].sum() for s in group)
            opt.zero_grad()
            nll_group = 0.0
            for s in group:
                with forward_context():
                    part_total, part_nll = loss_fn(ids[s::k].contiguous(), mask[s::k].contiguous(),
                                                   *(p[s::k].contiguous() for p in packed))
                w = mask[s::k, 1:].sum() / count
                with sdpa_kernel(kernels):
                    (part_total * w).backward()
                nll_group = nll_group + w * part_nll.detach()
            out.append((_Gradients(params, [None if p.grad is None else p.grad.detach().clone() for p in params]),
                        nll_group))
            counts.append(count)
        opt.zero_grad()
        return out, counts, sum(c * n for c, (_, n) in zip(counts, out)) / sum(counts)
    ema = None
    if weight_ema is not None:                             # ortalama model; surdurmede optimizer durumundan geri gelir
        ema = copy.deepcopy(model).requires_grad_(False)
        for pe, p in zip(ema.parameters(), model.parameters()):
            if "weight_ema" in opt.state.get(p, {}):
                pe.data = opt.state[p]["weight_ema"]
        model.weight_ema = dict(model=ema, decay=weight_ema)
    for group in opt.param_groups:                         # coherence durumu: surdurmede optimizer'la birlikte gelir
        group.setdefault("coherence_mean", 1.0)
        group.setdefault("coherence_frozen", None)
    curve, packed = [], []
    for step in range(first, steps + 1):
        resumed_here = checkpoint is not None and step == first
        if schedule == "coherence":                        # t < final: ortalama(rho);  sonra o deger · (floor + (1 - floor)(1 - done))
            g0 = opt.param_groups[0]
            if step < final:
                factor = min(max(g0["coherence_mean"], lr_floor), 1.0)
            else:
                if g0["coherence_frozen"] is None:
                    for group in opt.param_groups:
                        group["coherence_frozen"] = min(max(g0["coherence_mean"], lr_floor), 1.0)
                p = (step - final) / max(steps - final, 1)
                done = (p if final_cooldown_shape == "linear" else math.sqrt(p) if final_cooldown_shape == "sqrt"
                        else math.log1p(p / LOG_COOLDOWN_KAPPA) / math.log1p(1 / LOG_COOLDOWN_KAPPA))
                factor = g0["coherence_frozen"] * (lr_floor + (1 - lr_floor) * (1 - done))
        else:                                              # WSD: t < start: lr;  sonra lr · (floor + (1 - floor)(1 - sqrt(p)))
            factor = 1.0 if step < start else (
                lr_floor + (1 - lr_floor) * (1 - math.sqrt((step - start) / max(steps - start, 1))))
        for group in opt.param_groups:
            group["lr"] = lr * factor
        if batches is not None:                            # mini-batch: bu adimin parcasi; paketli: + document_positions
            ids, mask, *packed = (t.to(device) for t in batches(step))
            if compile and packed and not static_loss[0]:  # paketli pencere hep ayni boy: ayni surecteki onceki kosunun
                # boyu derleyiciye boyu dinamik saydiriyordu, dinamik kod uretilemedi (CantSplit)
                loss_fn, static_loss[0] = torch.compile(model.loss, dynamic=False), True
        # bf16: ileri hesap ve kayip autocast'te; geri yayilim disarida (autocast'in kaydettigi tiplerle)
        if micro_batches > 1:                              # parca parca ileri + geri; asagidaki backward'lar birikimi yazar
            halves, weights, nll = accumulate(ids, mask, packed)
            total = halves[0][0]
        else:
            with forward_context():
                if split:                                  # iki yari: satirlar tek / cift (uzunluk dagilimi benzer)
                    halves = [loss_fn(ids[h::2].contiguous(), mask[h::2].contiguous(),
                                      *(p[h::2].contiguous() for p in packed)) for h in (0, 1)]
                    weights = [mask[h::2, 1:].sum() for h in (0, 1)]     # kayip hedef token ortalamasi: token sayisiyla
                    nll = sum(w * hn for w, (_, hn) in zip(weights, halves)) / sum(weights)
                else:
                    total, nll = loss_fn(ids, mask, *packed)   # butun cumleler (ya da adimin parcasi), butun konumlar
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), dev=deviation(model).clone(),
                              W_context=sum(b.attention.W_context.norm().item() for b in model.blocks)))
        if callback is not None and every and step % every == 0 and not resumed_here:
            try:
                callback(step, model, nll.item())
            except Exception:                              # durdurma ya da hata: o adimdan surdurulebilsin
                if save is not None and step > 0:
                    save(step, model, opt)
                raise
        if save is not None and save_every and step > 0 and step % save_every == 0 and not resumed_here:
            save(step, model, opt)
        if step == steps:
            break
        opt.zero_grad()                                    # onceki adimin gradyanlarini sil
        if split:                                          # iki yarinin gradyani -> rho; toplam = tam batch'in gradyani
            grads = None                                   # tek kopya: g1; ikinci yari p.grad'a birikir (g1 + g2)
            for (half_total, _) in halves:
                with sdpa_kernel(kernels):
                    half_total.backward()
                if grads is None:
                    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
            sums = []                                       # parametre basina (a.b, a.a, b.b): sonda tek senkron
            w0, w1 = (float(w) for w in weights)
            for (k, p), a in zip(named, grads):
                if a is None:
                    p.grad = None
                    continue
                b = p.grad - a                              # g2 = (g1 + g2) - g1
                p.grad = (w0 * a + w1 * b) / (w0 + w1)      # birlesik gradyan = tam batch'inki
                if k in unit_axis:                          # kurede: teget (satir ya da sutun basina)
                    a, b = (v - (v * p.detach()).sum(unit_axis[k], keepdim=True) * p.detach() for v in (a, b))
                sums.append(torch.stack([(a * b).sum(), (a * a).sum(), (b * b).sum()]))
            dot, na, nb = 0.0, 0.0, 0.0
            for x, y, z in torch.stack(sums).tolist():      # eski sirayla toplanir (bit duzeyinde ayni)
                dot, na, nb = dot + x, na + y, nb + z
            diff = na + nb - 2 * dot                        # |g1 - g2|^2
            rho = max(dot, 0.0) / (max(dot, 0.0) + diff / 4 + 1e-30)
            for group in opt.param_groups:                  # yansiz: pay ve payda ayri ortalama, sonra oran
                if group.get("coherence_total") is None:    # ilk olcum: onceki ortalama (basta 1) bu olcumun boyuyla
                    group["coherence_total"] = max(dot, 0.0) + diff / 4
                    group["coherence_signal"] = group["coherence_mean"] * group["coherence_total"]
                group["coherence_signal"] += (max(dot, 0.0) - group["coherence_signal"]) / coherence_window
                group["coherence_total"] += (max(dot, 0.0) + diff / 4 - group["coherence_total"]) / coherence_window
                group["coherence_mean"] = group["coherence_signal"] / (group["coherence_total"] + 1e-30)
            model.coherence = dict(c=dot / (na * nb + 1e-30) ** 0.5, rho=rho, mean=opt.param_groups[0]["coherence_mean"],
                                   lr=opt.param_groups[0]["lr"])
        else:
            with sdpa_kernel(kernels):
                total.backward()                           # her ogrenilen sayi x icin ∂kayip/∂x (zincir kurali, otomatik)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)  # butun gradyanlarin toplam boyu > grad_clip ise olcekle indir
        opt.step()                                         # Adam: x <- x - lr · m / (√v + eps); Muon: ortogonal adim
        if sphere_weights:                                 # agirlik kureye geri; adim boyunu yalniz lr belirler
            model.normalize_weights()
        if getattr(model, "output_link", False):          # phi hep artan kalsin: u >= 0
            with torch.no_grad():
                model.link_u.clamp_(min=0)
        if ema is not None:                                # ortalama <- d x ortalama + (1 - d) x agirlik
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(weight_ema).add_(p.detach(), alpha=1 - weight_ema)
                    state = opt.state.get(p)
                    if state and "weight_ema" not in state:   # optimizer durumuna bagla: checkpoint'e girer
                        state["weight_ema"] = pe.data
                if sphere_weights:
                    ema.normalize_weights()
    return model, curve


@torch.no_grad()
def generate(model, prompts, n, cached=True):
    """Acgozlu uretim, her isteme n token.  -> id listeleri (yalniz uretilen).
    cached: istem bir kez, sonra her token yalniz kendi konumunu hesaplar (AttentionCache); butun istemler tek batch'te
    (farkli uzunluk: sagdan dolgu, satir basina konum).  Degilse (test referansi): ayni uzunluktaki istemler birlikte ve
    her token'da butun dizi yeniden hesaplanir (ayni token'lar, skorlar float yuvarlamasina kadar)."""
    if cached:
        return generate_cached(model, prompts, n)
    device = next(model.parameters()).device
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for idx in groups.values():
        ids = torch.tensor([prompts[i] for i in idx], device=device)
        for _ in range(n):
            ids = torch.cat([ids, model.logits(ids)[:, -1].argmax(-1, keepdim=True)], 1)
        for row, i in zip(ids[:, ids.shape[1] - n:].tolist(), idx):
            out[i] = row
    return out


@torch.no_grad()
def generate_cached(model, prompts, n):
    """generate'in onbellekli hali: istem ileri hesabi bir kez (tur basina AttentionCache doldurulur), sonra
    her adimda satir basina tek yeni konum turlardan gecer.  Is: istem + n - 1 konum (tam yeniden hesapta sum(L + i))."""
    if n <= 0:
        return [[] for _ in prompts]
    device = next(model.parameters()).device
    lengths = [len(p) for p in prompts]
    ids = torch.zeros(len(prompts), max(lengths), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p)
    ids, L = ids.to(device), torch.tensor(lengths, device=device)
    caches = [AttentionCache(L, max(lengths) + n) for _ in range(model.turns)]
    token = model.logits(ids, caches)[torch.arange(len(prompts), device=device), L - 1].argmax(-1)
    out = [token]
    for _ in range(n - 1):
        token = model.logits(token[:, None], caches)[:, 0].argmax(-1)
        out.append(token)
    return torch.stack(out, 1).tolist()
