# -*- coding: utf-8 -*-
"""train_y -- model_y GENEL egitim: veriden bagimsiz.  Veriye ozgu sinav ve raporlar egitim klasorlerinde
(akrabalik: train_kinship/exam_kinship.py).
    train       Adim 1: ardisik token ciftleri (BigramModel)
    train_seq   dizi egitimi: Model X (varsayilan), deneme ayarlari ve kiyas transformer'i; yedek ve surdurme
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

from model_y import (CANON, LAYER_NORM, NORMALIZED_UPDATE, ROPE, SPHERE_WEIGHTS, STREAM_NORM, AttentionCache,
                     BigramModel, BlockModel, SequenceModel, deviation)
from model_y_transformer import TransformerModel

SETTINGS = {                      # capa lr/wd gibi deneme sayisi; 1e-2 fazla sertti (kayip 1,284 > 1,270)
    "fixed": dict(learn_points=False),
    "free": dict(learn_points=True, anchor=0.0),
    "anchored": dict(learn_points=True, anchor=1e-3),
}


STEPS, LR = 4000, 0.01  # full batch: 1 adim = 1 epoch.  32 aile icin ilk deger (kullanici, 27 Eylul: "ilk olarak 4.000");
                         # 8 ailede 1000 idi (ezber 600'de tam), ondan once 2000.  Veri buyurse yeniden belirlenir.
                         # lr dayanagi (kure agirliklari + Muon): adim basina donme ~ LR x 0,2 x sqrt(d) (D=384'te ~2,2
                         # derece); nGPT 2026 tepe lr 0,24 / sqrt(d).  Veri / batch / D degisince yeniden hesaplanir.
LOG_AT = (0, 10, 50, 200, 500, 1000)
LR_FLOOR = 0.0       # inisin tabani: lr sonda LR x LR_FLOOR (wsd, cosine, coherence).  0 (kullanici, 29 Eylul: "1 evet
                     # varsayılan olsun"; 28 Eylul'e kadar 0,1): taban asil kaldirac (BULGULAR_y 1.3 B)
GRAD_CLIP = 1.0      # gradient clipping: adimdaki gradient'in boyu bunu gecerse buna indirilir; train_seq standardi
WEIGHT_DECAY = 0.0   # weight decay (AdamW), yalniz W_ matrislerine; 0 = kapali (standart).  Deger olculuyor
OPTIMIZER = "muon"   # "muon": gizli matrisler (W_context, W_fact_in, W_fact_out; transformer'da W_value, W_out, W_mlp_in,
                     # W_mlp_out) Muon, gerisi Adam | "adam": hepsi Adam (27 Eylul'e kadarki butun kosular).
                     # Kullanici, 27 Eylul: "WSD ve muon uygun", varsayilan "Hemen Muon + WSD"
SCHEDULE = "wsd"     # "wsd": lr sabit, son COOLDOWN kisminda 1 - sqrt ile LR x LR_FLOOR'a | "cosine": 27 Eylul'e kadarki
# Inductor'un bellek yerlesimi analizi dinamik sekilde (bucket) ic kontrolde patladi (Colab, 28 Eylul); yalniz tiling
# sezgisi, kapatmak sonucu degistirmez
torch._inductor.config.triton.coalesce_tiling_analysis = False
COMPILE_LOCK = threading.Lock()   # torch.compile iplikler arasi guvenli degil: bir kosu derlerken digerinin derlenmis
                                  # cagrisi "FX ile izleme" hatasi verdi (Colab, 27 Eylul); ileri hesap bu kilit altinda
COOLDOWN = 0.2       # WSD'de inisin payi (son %20).  Hagele 2024: <= %20 yeter, 1 - sqrt dogrusaldan iyi
# SCHEDULE "coherence" (kullanici, 28 Eylul: "modelden ölçtüğümiz bir bilgiye göre Lr bu olsun diyemiyor muyuz ?"): lr = LR x
# rho, rho = adimin sinyal payi, batch'in iki yarisinin gradyanlarindan olculur (g1, g2; N >> B):
#   rho = g1.g2 / (g1.g2 + |g1 - g2|^2 / 4)        (= 2c / (1 + c), c = cos(g1, g2), esit boylarda)
# kurede adim = aci: theta = LR x 0,2 x sqrt(d) x rho.  Tam batch'te (ids verilmis) gurultu yok: rho = 1, olculmez.
COHERENCE_WINDOW = 200   # rho'nun hareketli ortalamasi (adim); tek adimin olcumu gurultulu
FINAL_COOLDOWN = 0.95    # coherence'ta inisin payi: ilk %5'te lr olculur, sonra o degerden x LR_FLOOR'a.  0,95 (kullanici,
                         # 29 Eylul: "1 evet varsayılan olsun"; oncesi 0,05 / 0,2): SimpleStories 1 epok, phi ile EMA bpb
                         # 0,6090 -> 0,5932, acc +0,7 puan (plato evresi kazandirmiyordu)
FROZEN_LR = None         # coherence'ta inisin basladigi lr'yi elle vermek (None: o anki olculen deger).  Kullanici, 29 Eylul:
                         # "Tepe lr'yi φ'li koşununkine sabitle" (phi acik / kapali kiyasinda tek fark phi kalsin)
FINAL_COOLDOWN_SHAPE = "linear"   # son inisin bicimi: "linear" 1 - p | "sqrt" 1 - sqrt(p) (29 Eylul'e kadar; kisa
                                  # inislerde).  Kullanici, 29 Eylul: "final cooldown olsun", "1 evet varsayılan olsun"
COHERENCE_POWER = 1.0    # coherence'ta lr carpani = ortalama(rho) ^ bu us; alt / ust sinir ve son inis aynen.  0,5 = sqrt(rho)
                         # (kullanici, 28 Eylul: "Sonra tek farkı lr = LR·√ρ olan bir 10k koşusu, 6,94'e karşı."); 1,0 = bugunku
WEIGHT_EMA = None       # agirliklarin hareketli ortalamasi (ornek 0,999; kullanici onayli ad, 28 Eylul): titresimi
                         # siler; L(w) - L(ortalama) = o anki titresimin bedeli.  None = kapali
MATMUL_PRECISION = "bf16"   # egitim hesabi: "fp32" (28 Eylul'e kadarki) | "tf32" (egitim boyunca TF32 matmul; KL 2e-7) | "bf16"
                            # (ileri hesap autocast, kayip ~3e-4 nat; parametre, gradyan, optimizer ve Muon Newton-Schulz, weight
                            # EMA fp32).  Sinav hep fp32.  Yalniz GPU'da (kullanici, 28 Eylul: "bunu da yapalım model y de")
ATTENTION_KERNEL = "math"   # egitimde SDPA cekirdegi: "math" (bit duzeyinde surdurme, CPU testleri) | "flash" (GPU'da flash,
                            # olmazsa mem-efficient; geri yayilim deterministik degil).  CPU'da hep math.  Kullanici, 29 Eylul:
                            # "önerin kabul" (math, gövde ileri hesabinin ~%70'i; GPU'da surdurme zaten bit duzeyinde degil)
NEWTON_SCHULZ_PRECISION = "fp32"   # Muon'un ortogonallestirmesi: "fp32" | "bf16" (Muon'un yaygin kullanimi; adimin %14'u)
MUON_TANGENT = True      # sphere_weights'te Muon'a giren Nesterov birlesimi (g + mu buf) once agirligin kure tegetine izdusulur,
                         # sonra ortogonallestirilir (kullanici, 28 Eylul: "tamam alalım"): kurede birinci mertebe dususu en buyuk
                         # yapan adim polar(T(G)); Training nGPT 1B'de "slightly improves".  Momentum tamponu ve Adam degismez


class Muon(torch.optim.Optimizer):
    """Muon (K. Jordan 2024, MIT) ve ayni sinifta Adam.  use_muon=True gruplar: momentum (Nesterov) -> Newton-Schulz ile
    ortogonallestirme (5 adim, fp32) -> 0,2 x sqrt(max(satir, sutun)) olcegi (Liu 2025: boylece AdamW'nin lr'si aynen
    kullanilir).  use_muon=False gruplar: torch.optim.Adam ile ayni formul (beta 0,9 / 0,999, eps 1e-8)."""

    def __init__(self, groups, lr, momentum=0.95, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(groups, dict(lr=lr, momentum=momentum, betas=betas, eps=eps, use_muon=False))
        self.tangent_axis = {}      # MUON_TANGENT: parametre -> kurede birim eksen (1 satir, 0 sutun); bos = izdusum yok
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
                for p in g["params"]:                       # hesapta degil cagri sayisinda (profil, 29 Eylul: 26 ms)
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


def bigram_floor(inputs, targets, n):
    """Yalniz son token'a bakan bir modelin ulasabilecegi en dusuk kayip (sayimlardan)."""
    C = torch.zeros(n, n, dtype=torch.float64)
    C.index_put_((inputs, targets), torch.ones(len(inputs), dtype=torch.float64), accumulate=True)
    p = C / C.sum(1, keepdim=True).clamp_min(1)
    nz = C > 0
    return float(-(C[nz] * p[nz].log()).sum() / C.sum())


def train(setting, inputs, targets, n, steps=STEPS, lr=LR, log_at=LOG_AT):
    model = BigramModel(n, **SETTINGS[setting])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    for step in range(steps + 1):
        total, nll = model.loss(inputs, targets)
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), W_next=model.next.W_next.norm().item(), dev=deviation(model).clone()))
        if step == steps:
            break
        opt.zero_grad()
        total.backward()
        opt.step()
    return model, curve


# ---- dizi egitimi

STEP2 = {"step1": dict(attention=False), "step2": dict(attention=True)}
STEP3 = {"shared": dict(shared=True), "separate": dict(shared=False)}


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
              save_every=None, save=None, checkpoint=None, stream_norm=STREAM_NORM, layer_norm=LAYER_NORM, rope=None,
              batches=None, model_kw=None, optimizer=OPTIMIZER, schedule=SCHEDULE, cooldown=COOLDOWN,
              normalized_update=None, sphere_weights=None, canon=None, coherence_window=COHERENCE_WINDOW,
              final_cooldown=FINAL_COOLDOWN, weight_ema=WEIGHT_EMA, matmul_precision=MATMUL_PRECISION,
              coherence_power=COHERENCE_POWER, muon_tangent=MUON_TANGENT, final_cooldown_shape=FINAL_COOLDOWN_SHAPE,
              frozen_lr=FROZEN_LR, attention_kernel=ATTENTION_KERNEL, newton_schulz_precision=NEWTON_SCHULZ_PRECISION):
    """Standart tarif (27 Eylul'den): Muon (gizli matrisler) + Adam, WSD takvimi (lr sabit, son cooldown kisminda
    1 - sqrt ile LR x lr_floor'a), gradient clipping.  optimizer="adam", schedule="cosine": 27 Eylul'e kadarki tarif.
    lr_floor=None, grad_clip=None: en eski tarif (sabit lr).  weight_decay > 0: AdamW, yalniz W_ matrisleri (yalniz adam).
    callback(step, model, nll): her `every` adimda, o adimin guncellemesinden ONCE (sinav, kayit, durdurma).
    compile: kayip hesabi (ileri + geri) torch.compile ile; VARSAYILAN ACIK (kullanici: "bu sabit ayar ve yes olsun").  Yalniz
    GPU'da uygulanir: CPU'da kendiliginden kapanir (asagidaki if), elle compile=False yazmak gerekmez.
    setting "transformer": kiyas modeli (model_y_transformer), ayni tarif.  rope: attention'da RoPE; None = modelin kendi
    varsayilani (transformer True, BlockModel ROPE).
    save(step, model, opt): her save_every adimda, callback'ten sonra, guncellemeden ONCE -- adim s paketi s guncelleme
    gormus modeli ve optimizer'i tasir.  callback hata atarsa (durdurma dahil) o adimin paketi de yazilir.  checkpoint {step, model, optimizer}: o adimdan surdurur (ayni steps ve tarifle
    kesintisiz kosuyla bit duzeyinde ayni -- CPU'da; GPU'da compile token gradyanini atomik toplar, son bitler oynayabilir);
    o adimin callback'i ve kaydi tekrarlanmaz.
    batches(step) -> (ids, mask): buyuk veri icin her adimda bir parca (mini-batch); verilirse ids, mask kullanilmaz (None
    olabilir).  Adimin fonksiyonu olmali (surdurmede ayni parca gelsin).  Sekil adimdan adima degisebilir (bucket):
    compile ikinci sekilde dinamik sekilli tek grafige gecer.
    model_kw: modele gecen ayarlar (BlockModel: d, turns, units, t_max ...; transformer: d, layers, heads, units).
    schedule="coherence": lr = lr x ortalama(rho) (COHERENCE_WINDOW), alt sinir lr_floor; son final_cooldown kisminda
    1 - sqrt ile x lr_floor'a.  Mini-batch'te adim iki yarida hesaplanir (satirlar tek / cift), birlestirilen gradyan tam
    batch'inkiyle ayni; ortalama optimizer'in grup kaydinda (checkpoint'e girer).  model.coherence: son olcum (okuma).
    Ortalama yansiz: pay ve payda ayri hareketli ortalama, ortalama(rho) = EMA(dot+) / EMA(dot+ + |g1 - g2|^2 / 4) (oranlarin
    ortalamasi dusuk rankli gurultude yukari yanli; kullanici: "Önce yansız ortalama (bedava, doğru)").
    coherence_power: lr carpani ortalama(rho) ^ coherence_power (sinirlar ve son inisteki dondurulan deger de bu).
    muon_tangent: MUON_TANGENT (yalniz sphere_weights'te etkili).
    weight_ema=d: her adimdan sonra ortalama <- d x ortalama + (1 - d) x agirlik (kurede satirlar yeniden birim);
    model.weight_ema = dict(model=<ortalama model>, decay=d).  Ortalama optimizer durumunda (checkpoint'e girer).
    matmul_precision: "bf16" ileri hesap ve kayip autocast (bf16) icinde, geri yayilim disinda; "tf32" egitim boyunca TF32
    matmul (surec geneli ayar: ayni surecteki eszamanli kosular da etkilenir), sinav fp32, bitince onceki ayar; "fp32"
    hicbir seye dokunmaz.  CPU'da (compile gibi) etkisiz.
    attention_kernel: ATTENTION_KERNEL (CPU'da hep math).  newton_schulz_precision: NEWTON_SCHULZ_PRECISION (yalniz Muon)."""
    assert batches is not None or ids is not None, "ids/mask ya da batches verilmeli"
    assert setting in STEP3 or stream_norm, "stream_norm=False yalniz Adim 3 (BlockModel) icin"
    assert setting in STEP3 or not layer_norm, "layer_norm yalniz Adim 3 (BlockModel) icin"
    assert not setting.startswith("transformer") or not weight_decay, "transformer icin weight decay gruplari tanimli degil"
    assert optimizer in ("muon", "adam") and schedule in ("wsd", "cosine", "coherence") and 0 < cooldown <= 1
    assert schedule != "coherence" or (lr_floor is not None and 0 < final_cooldown <= 1 and coherence_window >= 1)
    assert final_cooldown_shape in ("sqrt", "linear"), "final_cooldown_shape: sqrt | linear"
    assert frozen_lr is None or (schedule == "coherence" and frozen_lr > 0), "frozen_lr: yalniz coherence'ta, pozitif"
    assert weight_ema is None or 0 < weight_ema < 1, "weight_ema: 0 ile 1 arasi (ornek 0,999) ya da None"
    assert optimizer == "adam" or not weight_decay, "weight_decay yalniz optimizer='adam' ile (AdamW)"
    assert matmul_precision in ("fp32", "tf32", "bf16"), "matmul_precision: fp32 | tf32 | bf16"
    assert coherence_power > 0, "coherence_power: pozitif us (1,0 = rho, 0,5 = sqrt(rho))"
    assert attention_kernel in ("math", "flash"), "attention_kernel: math | flash"
    assert newton_schulz_precision in ("fp32", "bf16"), "newton_schulz_precision: fp32 | bf16"
    if normalized_update is None:                          # Model X'in varsayilani; transformer ve Adim 1-2'de yok
        normalized_update = NORMALIZED_UPDATE if setting in STEP3 else False
    if sphere_weights is None:
        sphere_weights = SPHERE_WEIGHTS if setting in STEP3 else False
    if canon is None:
        canon = CANON if setting in STEP3 else False
    assert setting in STEP3 or not (normalized_update or sphere_weights or canon), \
        "normalized_update / sphere_weights / canon yalniz Adim 3 icin"
    if rope is None:                                       # modelin kendi varsayilani; Adim 1-2 modellerinde RoPE yok
        rope = True if setting.startswith("transformer") else ROPE if setting in STEP3 else False
    assert setting in STEP3 or setting.startswith("transformer") or not rope, "rope yalniz Adim 3 ve transformer icin"
    if setting in ("transformer", "transformer_novalue"):   # novalue: V matrisi yok (tek head'de V.O tek matris)
        model = TransformerModel(n, seed=seed, value_matrix=setting == "transformer", rope=rope, **(model_kw or {}))
    elif setting in STEP3:
        model = BlockModel(n, seed=seed, stream_norm=stream_norm, layer_norm=layer_norm, rope=rope,
                           normalized_update=normalized_update, sphere_weights=sphere_weights, canon=canon, **STEP3[setting],
                           **(model_kw or {}))
    else:
        model = SequenceModel(n, seed=seed, **STEP2[setting], **(model_kw or {}))
    model = model.to(device)
    if ids is not None:
        ids, mask = ids.to(device), mask.to(device)
    # ogrenilenler: Δ (shift), W_query, W_key, W_context; Adim 1-2: + W_next; Adim 3: + FactUnits (W_next yok)
    # (PF buffer, listede yok)
    params = [p for p in model.parameters() if p.requires_grad]
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    unit_axis = {}                                         # kure tegetinin tek tablosu (coherence ve MUON_TANGENT)
    for k, _ in named:
        kind = k.split(".")[-1]
        if sphere_weights and kind in ("W_query", "W_key", "W_fact_in", "W_fact_up", "W_value"):   # girdisi durum: satir
            unit_axis[k] = 1
        elif sphere_weights and kind in ("W_context", "W_fact_out"):                               # duruma yazan: sutun
            unit_axis[k] = 0
    if optimizer == "muon":
        # Muon yalniz gizli 2 boyutlu matrislerde ("VO + FFN" duzeni, Wang 2025); token noktalari, esikler, W_query, W_key,
        # bias, norm katsayilari ve cikis olcegi (log_output_scale) Adam'da
        hidden = ("W_context", "W_value", "W_fact_in", "W_fact_up", "W_fact_out", "W_value.weight", "W_out.weight",
                  "W_mlp_in.weight",
                  "W_mlp_out.weight")
        opt = Muon([dict(params=[p for k, p in named if k.endswith(hidden)], use_muon=True),
                    dict(params=[p for k, p in named if not k.endswith(hidden)], use_muon=False)], lr=lr)
        if muon_tangent:                                   # yalniz Muon grubunda uygulanir (Adam degismez)
            opt.tangent_axis = {p: unit_axis[k] for k, p in named if k in unit_axis}
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
        matmul_precision = "fp32"                          # CPU'da etkisiz: autocast ve TF32 yalniz GPU'da
        attention_kernel = "math"                          # flash / mem-efficient yalniz GPU'da
    kernels = ([SDPBackend.MATH] if attention_kernel == "math" else
               [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    loss_fn = torch.compile(model.loss) if compile else model.loss
    start = round((1 - cooldown) * steps)                  # WSD: inis bu adimda baslar
    final = round((1 - final_cooldown) * steps)            # coherence: son inis bu adimda baslar
    split = schedule == "coherence" and batches is not None  # tam batch'te gurultu yok: rho = 1, olculmez
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
    curve = []
    precision = torch.get_float32_matmul_precision()      # tf32: surec geneli ayar; egitim bitince (hata olsa da) geri
    if matmul_precision == "tf32":
        torch.set_float32_matmul_precision("high")
    try:
        for step in range(first, steps + 1):
            resumed_here = checkpoint is not None and step == first
            if lr_floor is not None:
                if schedule == "cosine":                       # lr_t = lr · (floor + (1 - floor) · (1 + cos(π t / T)) / 2)
                    factor = lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
                elif schedule == "coherence":                 # t < final: ortalama(rho);  sonra o deger · (floor + (1 - floor)(1 - sqrt(p)))
                    g0 = opt.param_groups[0]
                    if step < final:
                        factor = min(max(g0["coherence_mean"] ** coherence_power, lr_floor), 1.0)
                    else:
                        if g0["coherence_frozen"] is None:
                            for group in opt.param_groups:
                                group["coherence_frozen"] = (frozen_lr / lr if frozen_lr is not None else
                                                              min(max(g0["coherence_mean"] ** coherence_power, lr_floor), 1.0))
                        p = (step - final) / max(steps - final, 1)
                        factor = g0["coherence_frozen"] * (
                            lr_floor + (1 - lr_floor) * (1 - (p if final_cooldown_shape == "linear" else math.sqrt(p))))
                else:                                          # WSD: t < start: lr;  sonra lr · (floor + (1 - floor)(1 - sqrt(p)))
                    factor = 1.0 if step < start else (
                        lr_floor + (1 - lr_floor) * (1 - math.sqrt((step - start) / max(steps - start, 1))))
                for group in opt.param_groups:
                    group["lr"] = lr * factor
            if batches is not None:                            # mini-batch: bu adimin parcasi
                ids, mask = (t.to(device) for t in batches(step))
            # attention_kernel "math": torch 2.14'ten itibaren SDPA kendiliginden flash / mem-efficient'e gidiyor, onlarin
            # geri yayilimi deterministik degil (surdurme bit duzeyinde ayni kalmaz; hiz ajani, 27 Eylul)
            # bf16: ileri hesap ve kayip autocast'te; geri yayilim disarida (autocast'in kaydettigi tiplerle)
            with COMPILE_LOCK if compile else contextlib.nullcontext(), sdpa_kernel(kernels), (
                    torch.autocast("cuda", dtype=torch.bfloat16) if matmul_precision == "bf16" else contextlib.nullcontext()):
                if split:                                      # iki yari: satirlar tek / cift (uzunluk dagilimi benzer)
                    halves = [loss_fn(ids[h::2].contiguous(), mask[h::2].contiguous()) for h in (0, 1)]
                    weights = [mask[h::2, 1:].sum() for h in (0, 1)]     # kayip hedef token ortalamasi: token sayisiyla
                    nll = sum(w * hn for w, (_, hn) in zip(weights, halves)) / sum(weights)
                else:
                    total, nll = loss_fn(ids, mask)            # butun cumleler (ya da adimin parcasi), butun konumlar
            if step in log_at:
                curve.append(dict(step=step, nll=nll.item(), dev=deviation(model).clone() if hasattr(model, "tokens") else None,
                                  W_context=(sum(b.attention.W_context.norm().item() for b in model.blocks)
                                             if isinstance(model, BlockModel) else
                                             model.attention.W_context.norm().item() if getattr(model, "attention", None) is not None
                                             else 0.0)))
            if callback is not None and every and step % every == 0 and not resumed_here:
                if matmul_precision == "tf32":             # sinav fp32
                    torch.set_float32_matmul_precision(precision)
                try:
                    callback(step, model, nll.item())
                except Exception:                              # durdurma ya da hata: o adimdan surdurulebilsin
                    if save is not None and step > 0:
                        save(step, model, opt)
                    raise
                if matmul_precision == "tf32":
                    torch.set_float32_matmul_precision("high")
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
            if grad_clip is not None:                          # butun gradyanlarin toplam boyu > grad_clip ise olcekle indir
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()                                         # Adam: x <- x - lr · m / (√v + eps); Muon: ortogonal adim
            if sphere_weights:                                 # kusur 2: agirlik kureye geri; adim boyunu yalniz lr belirler
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
    finally:
        if matmul_precision == "tf32":
            torch.set_float32_matmul_precision(precision)
    return model, curve


@torch.no_grad()
def generate(model, prompts, n, cached=True):
    """Acgozlu uretim, her isteme n token.  -> id listeleri (yalniz uretilen).
    cached (BlockModel): istem bir kez, sonra her token yalniz kendi konumunu hesaplar (AttentionCache); butun istemler
    tek batch'te (farkli uzunluk: sagdan dolgu, satir basina konum).  Degilse: ayni uzunluktaki istemler birlikte ve her
    token'da butun dizi yeniden hesaplanir (transformer; ayni token'lar, skorlar float yuvarlamasina kadar)."""
    if cached and isinstance(model, BlockModel):
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
    """generate'in onbellekli hali (BlockModel): istem ileri hesabi bir kez (tur basina AttentionCache doldurulur), sonra
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
