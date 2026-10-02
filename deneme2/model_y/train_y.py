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
import time

import torch
import torch._inductor.config
from torch.nn.attention import SDPBackend, sdpa_kernel

from model_y import ROPE, AttentionCache, BlockModel, deviation

STEPS, LR = 4000, 0.01  # lr dayanagi (kure agirliklari + Muon): adim basina donme ~ LR x 0,2 x sqrt(d); nGPT 2026 tepe lr
                         # 0,24 / sqrt(d).  Veri / batch / D degisince yeniden hesaplanir
LOG_AT = (0, 10, 50, 200, 500, 1000)
LR_FLOOR = 0.0       # inisin tabani: lr sonda LR x LR_FLOOR (wsd, coherence)
GRAD_CLIP = 1.0      # gradient clipping: adimdaki gradient'in boyu bunu gecerse buna indirilir
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
# DITTO-X kendi ciktisinda (bulgular 18, 21): her DITTO_X_SELF_EVERY adimda gercek onekten acgozlu devam; secim orani veri
# oranini asan bolmelerde en dusuk p'li kopyalarin logit'i indirilir.  Kullanici, 2 Ekim: "doğru bilgiye zarar vermez ama
# faydasız döngüyü de azaltır."
DITTO_X_WEIGHT = 0.0      # r*: kendi teriminin MLE'ye orani; ilk kendi adiminda beta'ya cevrilir, sonra sabit.  0 = kapali
DITTO_X_SHARE = 64        # kendi devami satir sayisi (B)
DITTO_X_SELF_PREFIX = 256  # gercek belge oneki (P)
DITTO_X_SELF_TOKENS = 128  # acgozlu devam (G)
DITTO_X_SELF_EVERY = 8    # uretim ve ceza her bu kadar adimda (yalniz o adimda)
DITTO_X_MARGIN = 0.0      # mu (nat)
DITTO_X_PARAMS = ("W_query", "W_key", "W_value", "W_context", "W_fact_in", "W_fact_up", "W_fact_out")   # kendi teriminin
                          # gradyani yalniz bu ad sonekli parametrelere (tau, q, u, alpha, Canon, E, Delta ayrik); None = hepsi
COPY_CEILING = {"1": 0.191, "2": 0.277, "3": 0.388, "4-7": 0.477, "8-15": 0.697, "sentence": 0.19}   # secim tavani: l kovasi
                          # (m = 1) gercek metinde kopya orani (D_032 Tablo 3); sentence = tekrar cumlesinden sonra yine o parca (O14)


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
              device="cpu", every=None, callback=None, compile=True,
              save_every=None, save=None, checkpoint=None, rope=ROPE,
              batches=None, model_kw=None, schedule=SCHEDULE, cooldown=COOLDOWN,
              coherence_window=COHERENCE_WINDOW,
              final_cooldown=FINAL_COOLDOWN, weight_ema=WEIGHT_EMA, matmul_precision=MATMUL_PRECISION,
              final_cooldown_shape=FINAL_COOLDOWN_SHAPE, newton_schulz_precision=NEWTON_SCHULZ_PRECISION,
              micro_batches=MICRO_BATCHES, ditto_x_weight=DITTO_X_WEIGHT, ditto_x_share=DITTO_X_SHARE,
              ditto_x_self_prefix=DITTO_X_SELF_PREFIX, ditto_x_self_tokens=DITTO_X_SELF_TOKENS,
              ditto_x_self_every=DITTO_X_SELF_EVERY, ditto_x_margin=DITTO_X_MARGIN, ditto_x_params=DITTO_X_PARAMS,
              copy_ceiling=COPY_CEILING, self_prompts=None, sentence_ends=None, eot=None):
    """Tarif: Muon (gizli matrisler: W_context, W_value, W_fact_in, W_fact_up, W_fact_out) + Adam (gerisi), takvim (wsd ya
    da coherence), gradient clipping.
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
    Kure agirliklari: her adimdan sonra model.normalize_weights; Muon'da Nesterov birlesimi once kure tegetine izdusulur
    (Muon.tangent_axis).
    weight_ema=d: her adimdan sonra ortalama <- d x ortalama + (1 - d) x agirlik (kurede satirlar yeniden birim);
    model.weight_ema = dict(model=<ortalama model>, decay=d).  Ortalama optimizer durumunda (checkpoint'e girer).
    matmul_precision: "bf16" ileri hesap ve kayip autocast (bf16) icinde, geri yayilim disinda; "fp32" hicbir seye
    dokunmaz.  CPU'da (compile gibi) etkisiz.  SDPA hep math cekirdegiyle: flash / mem-efficient'in geri yayilimi
    deterministik degil (surdurme bit duzeyinde kalmaz).
    micro_batches: gradyan birikimi (batch'in satir sayisi bunun kati olmali).
    checkpoint step 0: baslangic agirligi (baska kosudan, init_from) -- 0. adimin callback'i kosar.
    ditto_x_weight > 0 (DITTO-X kendi ciktisinda): her ditto_x_self_every adimda self_prompts(step) (>= P + G token'lik gercek
    satirlar, ilk ditto_x_share'i) -> onek P'den acgozlu devam (self_continuations, eot yasak) -> ditto_x_self_loss; MLE
    gradyani bugunku gibi kirpildiktan SONRA beta g_self eklenir, yalniz ditto_x_params parametrelerine (MLE yolu P0 ile ayni).
    beta = r* |c g_MLE| / |g_self| ilk kendi adiminda (normlar izinli parametrelerde), sonra sabit (grup kaydinda
    "ditto_x_self"); kendi terimi en cok 1 x |c g_MLE|.  Ayni oneklerin gercek devami gradyansiz gecer (sizma izleyicisi).
    Kayitlar model.ditto_x_self listesinde (kosucu sinavda okur ve bosaltir).  sentence_ends(ids) -> [(bas, son, anahtar)]
    (exam_fineweb._sentences_of); eot: uretimde yasak token."""
    assert batches is not None or ids is not None, "ids/mask ya da batches verilmeli"
    assert setting == "shared", "setting: yalniz shared (BlockModel), bu: %s" % setting
    assert schedule in ("wsd", "coherence") and 0 < cooldown <= 1
    assert schedule != "coherence" or (0 < final_cooldown <= 1 and coherence_window >= 1)
    assert final_cooldown_shape in ("sqrt", "linear", "log"), "final_cooldown_shape: sqrt | linear | log"
    assert weight_ema is None or 0 < weight_ema < 1, "weight_ema: 0 ile 1 arasi (ornek 0,999) ya da None"
    assert matmul_precision in ("fp32", "bf16"), "matmul_precision: fp32 | bf16"
    assert newton_schulz_precision in ("fp32", "bf16"), "newton_schulz_precision: fp32 | bf16"
    assert micro_batches >= 1 and (schedule != "coherence" or micro_batches % 2 == 0 or micro_batches == 1), \
        "micro_batches: >= 1; coherence'ta 1 ya da cift"
    model = BlockModel(n, seed=seed, rope=rope, **(model_kw or {})).to(device)
    if ids is not None:
        ids, mask = ids.to(device), mask.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    unit_axis = {}                                         # kure tegetinin tek tablosu (coherence ve Muon)
    for k, _ in named:
        kind = k.split(".")[-1]
        if kind in ("W_query", "W_key", "W_fact_in", "W_fact_up", "W_value"):   # girdisi durum: satir
            unit_axis[k] = 1
        elif kind in ("W_context", "W_fact_out"):                               # duruma yazan: sutun
            unit_axis[k] = 0
    # Muon yalniz gizli 2 boyutlu matrislerde ("VO + FFN" duzeni, Wang 2025); token noktalari, W_query, W_key, alpha'lar,
    # Canon, cikis olcegi ve bag Adam'da
    hidden = ("W_context", "W_value", "W_fact_in", "W_fact_up", "W_fact_out")
    opt = Muon([dict(params=[p for k, p in named if k.endswith(hidden)], use_muon=True),
                dict(params=[p for k, p in named if not k.endswith(hidden)], use_muon=False)], lr=lr)
    opt.tangent_axis = {p: unit_axis[k] for k, p in named if k in unit_axis}   # yalniz Muon grubunda uygulanir
    opt.newton_schulz_precision = newton_schulz_precision
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
    self_term = ditto_x_weight > 0
    if self_term:
        assert self_prompts is not None and sentence_ends is not None and eot is not None, \
            "DITTO-X: self_prompts, sentence_ends ve eot gerekli"
        allowed = [p for k, p in named if ditto_x_params is None or k.endswith(tuple(ditto_x_params))]
        model.ditto_x_self = []                            # kendi adimi kayitlari; kosucu sinavda okur ve bosaltir
    curve, packed = [], []
    for step in range(first, steps + 1):
        resumed_here = checkpoint is not None and step == first and first > 0
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
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)  # toplam boy > grad_clip ise olcekle indir
        if self_term and step % ditto_x_self_every == 0:  # DITTO-X: MLE kirpildiktan sonra eklenir (MLE yolu P0 ile ayni)
            t0 = time.time()
            P, G = ditto_x_self_prefix, ditto_x_self_tokens
            real = self_prompts(step)[:ditto_x_share].to(device)
            own = torch.cat([real[:, :P], self_continuations(model, real[:, :P], G, eot)], 1)
            gen_secs = time.time() - t0
            with forward_context():
                loss_self, own_stats = ditto_x_self_loss(model, own, P, sentence_ends, eot, copy_ceiling, ditto_x_margin)
                with torch.no_grad():                      # sizma izleyicisi: ayni oneklerin gercek devami
                    real_stats = ditto_x_self_loss(model, real[:, :P + G], P, sentence_ends, eot, copy_ceiling,
                                                   ditto_x_margin)[1]
            grads = [None] * len(allowed)
            if loss_self.requires_grad:
                with sdpa_kernel(kernels):
                    grads = torch.autograd.grad(loss_self, allowed, allow_unused=True)
            norm = lambda gs: math.sqrt(sum(float(g.double().pow(2).sum()) for g in gs if g is not None))
            n_mle, n_self = norm([p.grad for p in allowed]), norm(grads)
            if (opt.param_groups[0].get("ditto_x_self") or {}).get("beta") is None and n_self > 0:
                for group in opt.param_groups:             # beta ilk gradyanli kendi adiminda, sonra sabit (21 §8-1)
                    group["ditto_x_self"] = dict(beta=ditto_x_weight * n_mle / n_self)
            beta = (opt.param_groups[0].get("ditto_x_self") or {}).get("beta") or 0.0
            weight = min(beta, n_mle / n_self) if n_self > 0 else 0.0   # kendi terimi <= 1 x |c g_MLE|
            if weight:
                for p, g in zip(allowed, grads):
                    if g is not None:
                        p.grad = g * weight if p.grad is None else p.grad.add_(g, alpha=weight)
            model.ditto_x_self.append(dict(
                step=step, beta=beta, ratio=weight * n_self / max(n_mle, 1e-30), capped=bool(n_self and weight < beta),
                clip=min(1.0, grad_clip / (float(grad_norm) + 1e-6)), loss=own_stats["loss"], gen_secs=gen_secs,
                bins=own_stats["bins"], agreement=own_stats["agreement"],
                own=dict(by_l=own_stats["by_l"], entry=own_stats["entry"], ids=own[:, P:].cpu().numpy()),
                real=dict(by_l=real_stats["by_l"], entry=real_stats["entry"], bins=real_stats["bins"],
                          ids=real[:, P:P + G].cpu().numpy())))
        opt.step()                                         # Adam: x <- x - lr · m / (√v + eps); Muon: ortogonal adim
        model.normalize_weights()                          # agirlik kureye geri; adim boyunu yalniz lr belirler
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


# ---- DITTO-X kendi ciktisinda (bulgular 18, 21)

def copy_trace(x):
    """x (B, N) token -> (l, m, copy), her biri (B, N) int64: l_t = t'de biten ve daha once (bitisi < t) gecmis en uzun sonek
    (<= 32), m_t = o sonekin onceki gecis sayisi, copy_t = en son onceki gecisin devami (l_t = 0: -1).
    analyze_errors._match_trace ile ayni tanim (D_032), vektorel."""
    B, N = x.shape
    pos = torch.arange(N, device=x.device)
    alive = (pos[None, :] < pos[:, None]).expand(B, N, N)          # [b, t, j]: j < t ve ortak sonek suruyor
    run = torch.zeros(B, N, N, dtype=torch.long, device=x.device)
    for k in range(32):
        back = x[:, (pos - k).clamp(min=0)]                         # x_(t - k)
        alive = alive & (back[:, :, None] == back[:, None, :]) & (pos >= k)
        run += alive
    ell = run.max(-1).values
    found = ell > 0
    last = torch.where(run == ell[..., None], pos, -1).max(-1).values
    copy = torch.where(found, x.gather(1, (last + 1).clamp(max=N - 1)), -1)
    count = torch.where(found, (run >= ell[..., None]).sum(-1), 0)
    return ell, count, copy


@torch.no_grad()
def self_continuations(model, prompts, n, eot):
    """Kendi devami: prompts (B, P > 1) ayni boyda, her satir kendi belgesi -> (B, n) acgozlu devam, eot yasak.  Onbellekli;
    istemin son token'i disi hidden'dan gecer (logits yalniz son konumda)."""
    B, P = prompts.shape
    assert P > 1, "onek en az 2 token"
    caches = [AttentionCache(torch.full((B,), P - 1, device=prompts.device), P + n) for _ in range(model.turns)]
    model.hidden(prompts[:, :-1], caches)
    z = model.logits(prompts[:, -1:], caches)[:, 0].float()
    out = []
    for i in range(n):
        z[:, eot] = -float("inf")
        out.append(z.argmax(-1))
        if i + 1 < n:
            z = model.logits(out[-1][:, None], caches)[:, 0].float()
    return torch.stack(out, 1)


def ditto_x_self_loss(model, rows, prefix, sentence_ends, eot, copy_ceiling=COPY_CEILING, margin=DITTO_X_MARGIN):
    """rows (B, N) = onek (prefix token) + devam; karar konumu t, prefix - 1 <= t <= N - 2 (sonraki token devamda).
    Bolmeler: l kovasi (copy_ceiling anahtari, m = 1) ve "sentence" = cumle siniri (O14 / _chain_stats: devamda biten tekrar
    cumlesinin son token'i, ardindan tam cumle, kopya token'i var); cumle siniri l kovasindan cikarilir.  Bolmede secim a_t:
    l kovasinda x_(t+1) = kopya token'i, cumle sinirinda sonraki cumle onceki gecisin ardindan gelen cumle.
    k = max(0, |A| - floor(g |S|)); A'nin p_c'si en dusuk k konumu:
        L = (1 / |S|) sum 1/2 (z_c - z*_r + margin)_+^2,   z*_r = max_{j != c, eot} z_j (gradyansiz), |S| butun bolmeler
    sentence_ends(ids) -> [(bas, son, anahtar)].  -> (L, istatistik): bins {bolme: [n, secilen, cezalanan]}, agreement
    [uyusan, n] (l kovasinda a_t ile z_c > z*_r), by_l {"lo-hi": [n, p_copy, top1, kopya]} (m = 1; _decision_stats'in
    toplamlari), entry [giren, satir, kalan pay toplami, kalan n] (_entry_stats'in cumle girisi), loss."""
    B, N = rows.shape
    dev = rows.device
    ell, mm, cp = copy_trace(rows)
    pos = torch.arange(N, device=dev)
    decide = (pos >= prefix - 1) & (pos <= N - 2)
    nxt = torch.cat([rows[:, 1:], rows.new_full((B, 1), -2)], 1)
    chosen = nxt == cp
    sentence = torch.zeros(B, N, dtype=torch.bool)
    seq = torch.zeros(B, N, dtype=torch.bool)
    entry = [0, B, 0.0, 0]
    cp_host = cp.cpu()
    for r, x in enumerate(rows.tolist()):
        sents, last_at, rep, first = sentence_ends(x), {}, [False] * N, None
        for i, (a, b, key) in enumerate(sents):
            if key is not None and key in last_at and b >= prefix:
                rep[a:b + 1] = [True] * (b + 1 - a)
                first = max(a, prefix) if first is None else first
                if i + 1 < len(sents) and sents[i + 1][2] is not None and cp_host[r, b] >= 0:
                    j = last_at[key]
                    sentence[r, b] = True
                    seq[r, b] = j + 1 < len(sents) and sents[i + 1][2] == sents[j + 1][2]
            if key is not None:
                last_at[key] = i
        if first is not None:
            entry[0] += 1
            if first < N:
                entry[2] += sum(rep[first:]) / (N - first)
                entry[3] += 1
    sentence, seq = sentence.to(dev) & decide, seq.to(dev)
    bins = {}
    for key in copy_ceiling:
        if key == "sentence":
            bins[key] = (sentence, seq)
        else:
            lo, hi = int(key.split("-")[0]), int(key.split("-")[-1])
            bins[key] = (decide & (mm == 1) & (ell >= lo) & (ell <= hi) & ~sentence, chosen)
    buckets = ((1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 32))
    watch = decide & (mm == 1) & (cp >= 0)
    use = (watch | sentence)[:, :-1]
    where = torch.full((B, N), -1, dtype=torch.long, device=dev)
    rt = use.nonzero()
    where[rt[:, 0], rt[:, 1]] = torch.arange(len(rt), device=dev)
    h = model.hidden(rows[:, :-1])[-1][rt[:, 0], rt[:, 1]]                      # karar konumlarinin son durumu
    c = cp[rt[:, 0], rt[:, 1]]
    P = model.tokens.points()
    scale = model.scale * torch.exp(model.log_output_scale - math.log(model.scale))
    link = (lambda v, q, u: v * (1 + v * (q + v * (q * q / 3 + u)))) if model.output_link else (lambda v, q, u: v)
    q, u = (model.link_q, model.link_u) if model.output_link else (None, None)
    work = torch.promote_types(h.dtype, torch.float32)                           # en az fp32 (float64 model float64)
    with torch.no_grad(), torch.autocast(dev.type, enabled=False):              # p_c, top1, z*_r: butun sozluk, parca parca
        Pd, sd = P.detach().to(work), scale.detach().to(work)
        qd, ud = (q.detach().to(work), u.detach().to(work)) if model.output_link else (None, None)
        z_copy, p_copy, top1, rival = [], [], [], []
        for s in range(0, len(rt), 1024):
            z = sd * link(h[s:s + 1024].detach().to(work) @ Pd.T, qd, ud)
            cc = c[s:s + 1024, None]
            z_copy.append(z.gather(1, cc)[:, 0])
            p_copy.append((z_copy[-1] - z.logsumexp(-1)).exp())
            top1.append(z.argmax(-1) == cc[:, 0])
            z.scatter_(1, cc, -float("inf"))
            z[:, eot] = -float("inf")
            rival.append(z.max(-1).values)
        z_copy, p_copy, top1, rival = (torch.cat(v) if v else torch.zeros(0, device=dev)
                                       for v in (z_copy, p_copy, top1, rival))
    stats = dict(bins={}, by_l={})
    total, picked, agree = 0, [], [0, 0]
    for key, (mask, sel) in bins.items():
        idx = where[mask]
        hit = idx[sel[mask]]
        n = len(idx)
        k = max(0, len(hit) - math.floor(copy_ceiling[key] * n + 1e-9))
        picked.append(hit[torch.argsort(p_copy[hit], stable=True)[:k]])
        stats["bins"][key] = [n, len(hit), k]
        total += n
        if key != "sentence":
            agree[0] += int((sel[mask] == (z_copy[idx] > rival[idx])).sum())
            agree[1] += n
    for lo, hi in buckets:
        idx = where[watch & (ell >= lo) & (ell <= hi)]
        stats["by_l"]["%d-%d" % (lo, hi)] = [len(idx), float(p_copy[idx].sum()), float(top1[idx].float().sum()),
                                             float(chosen[watch & (ell >= lo) & (ell <= hi)].float().sum())]
    stats.update(agreement=agree, entry=entry)
    picked = torch.cat(picked)
    if not len(picked) or not torch.is_grad_enabled():
        stats["loss"] = 0.0
        return torch.zeros((), device=dev), stats
    zc = scale * link((h[picked].to(work) * P[c[picked]].to(work)).sum(-1), q, u)   # canli: izinli parametreler ayrilmaz
    loss = 0.5 * (zc - rival[picked] + margin).clamp(min=0).pow(2).sum() / max(total, 1)
    stats["loss"] = float(loss.detach())
    return loss, stats
