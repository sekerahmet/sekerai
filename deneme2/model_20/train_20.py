# -*- coding: utf-8 -*-
"""train_20 -- model_20 GENEL egitim: veriden bagimsiz.  Veriye ozgu sinav ve raporlar egitim klasorlerinde
(akrabalik: train_kinship/exam_kinship.py).
    train       Adim 1: ardisik token ciftleri (BigramModel)
    train_seq   dizi egitimi: Model X (varsayilan), deneme ayarlari ve kiyas transformer'i; yedek ve surdurme
    pad         id listeleri -> (ids, mask);  generate: acgozlu uretim
"""
import contextlib
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model_20 import (COPY_PATH, LAYER_NORM, ROPE, STREAM_NORM, BigramModel, BlockModel, SequenceModel, deviation)
from model_20_transformer import TransformerModel

SETTINGS = {                      # capa lr/wd gibi deneme sayisi; 1e-2 fazla sertti (kayip 1,284 > 1,270)
    "fixed": dict(learn_points=False),
    "free": dict(learn_points=True, anchor=0.0),
    "anchored": dict(learn_points=True, anchor=1e-3),
}


STEPS, LR = 4000, 0.01  # full batch: 1 adim = 1 epoch.  32 aile icin ilk deger (kullanici, 27 Eylul: "ilk olarak 4.000");
                         # 8 ailede 1000 idi (ezber 600'de tam), ondan once 2000.  Veri buyurse yeniden belirlenir.
LOG_AT = (0, 10, 50, 200, 500, 1000)
LR_FLOOR = 0.1       # cosine decay: lr sonda LR x LR_FLOOR (taban lr/10); train_seq standardi
GRAD_CLIP = 1.0      # gradient clipping: adimdaki gradient'in boyu bunu gecerse buna indirilir; train_seq standardi
WEIGHT_DECAY = 0.0   # weight decay (AdamW), yalniz W_ matrislerine; 0 = kapali (standart).  Deger olculuyor
PRECISION = "fp32"   # egitim adiminin sayisalligi: "fp32" bugunku | "tf32" CUDA'da fp32 matmul TF32 tensor core'da |
                     # "bf16" autocast bf16 (agirliklar, Adam, normlar, softmax, kayip fp32).  Sinavlar hep fp32
ATTENTION_KERNEL = "math"   # "math": SDPA hep math yolu (bugunku; deterministik) | "fused": flash / mem-efficient uygunsa
                            # (hizli; geri yayilimi deterministik DEGIL -> surdurme bit duzeyinde ayni olmaz)
FUSED_ADAM = False   # True: Adam tek CUDA cekirdegi (fused); False: torch varsayilani (CUDA'da foreach)


@contextlib.contextmanager
def numerics(precision=PRECISION, attention_kernel=ATTENTION_KERNEL, device="cpu", autocast=False):
    """Egitim adiminin ileri (autocast=True) ve geri hesabinin sayisalligi; cikista onceki ayarlar geri gelir."""
    assert precision in ("fp32", "tf32", "bf16") and attention_kernel in ("math", "fused")
    matmul = torch.backends.cuda.matmul
    saved = torch.get_float32_matmul_precision(), matmul.allow_bf16_reduced_precision_reduction
    backends = [SDPBackend.MATH] if attention_kernel == "math" else [
        SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    try:
        torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
        if precision == "bf16":                        # split-K ara toplamlari da fp32 kalsin
            matmul.allow_bf16_reduced_precision_reduction = False
        cast = (torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if autocast and precision == "bf16"
                else contextlib.nullcontext())
        with sdpa_kernel(backends), cast:
            yield
    finally:
        torch.set_float32_matmul_precision(saved[0])
        matmul.allow_bf16_reduced_precision_reduction = saved[1]


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
              weight_decay=WEIGHT_DECAY, device="cpu", every=None, callback=None, compile=False, copy_path=COPY_PATH,
              save_every=None, save=None, checkpoint=None, stream_norm=STREAM_NORM, layer_norm=LAYER_NORM, rope=None,
              batches=None, model_kw=None, precision=PRECISION, attention_kernel=ATTENTION_KERNEL, fused_adam=FUSED_ADAM):
    """Standart tarif: cosine decay (LR -> LR x lr_floor) + gradient clipping.  lr_floor=None, grad_clip=None: eski tarif
    (sabit lr); 27 Eylul oncesi kayitli Adim 2-3 sonuclari onunla uretildi.  weight_decay > 0: AdamW, yalniz W_ matrisleri.
    callback(step, model, nll): her `every` adimda, o adimin guncellemesinden ONCE (sinav, kayit, durdurma).
    compile: yalniz kayip hesabi (ileri + geri) torch.compile ile, tek grafik (fullgraph) ve sabit sekil; True ya da mod adi
    ("reduce-overhead": CUDA graph, kucuk modelde cekirdek baslatma yukunu siler).  Inductor sonucu eager'la bit duzeyinde
    ayni degil.  precision, attention_kernel: numerics(); fused_adam: FUSED_ADAM.
    copy_path: Oneri A (kopya yolu ve kapisi), yalniz Adim 3 modelinde.  setting "transformer": kiyas modeli
    (model_20_transformer), ayni tarif.  rope: attention'da RoPE; None = modelin kendi varsayilani (transformer True,
    BlockModel ROPE).
    save(step, model, opt): her save_every adimda, callback'ten sonra, guncellemeden ONCE -- adim s paketi s guncelleme
    gormus modeli ve optimizer'i tasir.  checkpoint {step, model, optimizer}: o adimdan surdurur (ayni steps ve tarifle
    kesintisiz kosuyla bit duzeyinde ayni); o adimin callback'i ve kaydi tekrarlanmaz.
    batches(step) -> (ids, mask): buyuk veri icin her adimda bir parca (mini-batch); verilirse ids, mask kullanilmaz (None
    olabilir).  Adimin fonksiyonu olmali (surdurmede ayni parca gelsin); compile icin parcalarin sekli sabit olmali.
    model_kw: modele gecen ayarlar (BlockModel: d, turns, units, t_max ...; transformer: d, layers, heads, units)."""
    assert batches is not None or ids is not None, "ids/mask ya da batches verilmeli"
    assert setting in STEP3 or not copy_path, "copy_path yalniz Adim 3 (BlockModel) icin"
    assert setting in STEP3 or stream_norm, "stream_norm=False yalniz Adim 3 (BlockModel) icin"
    assert setting in STEP3 or not layer_norm, "layer_norm yalniz Adim 3 (BlockModel) icin"
    assert not setting.startswith("transformer") or not weight_decay, "transformer icin weight decay gruplari tanimli degil"
    if rope is None:                                       # modelin kendi varsayilani; Adim 1-2 modellerinde RoPE yok
        rope = True if setting.startswith("transformer") else ROPE if setting in STEP3 else False
    assert setting in STEP3 or setting.startswith("transformer") or not rope, "rope yalniz Adim 3 ve transformer icin"
    if setting in ("transformer", "transformer_novalue"):   # novalue: V matrisi yok (tek head'de V.O tek matris)
        model = TransformerModel(n, seed=seed, value_matrix=setting == "transformer", rope=rope, **(model_kw or {}))
    elif setting in STEP3:
        model = BlockModel(n, seed=seed, copy_path=copy_path, stream_norm=stream_norm, layer_norm=layer_norm, rope=rope,
                           **STEP3[setting], **(model_kw or {}))
    else:
        model = SequenceModel(n, seed=seed, **STEP2[setting], **(model_kw or {}))
    model = model.to(device)
    if ids is not None:
        ids, mask = ids.to(device), mask.to(device)
    # ogrenilenler: Δ (shift), W_query, W_key, W_context; Adim 1-2: + W_next; Adim 3: + FactUnits (W_next yok)
    # (PF buffer, listede yok)
    params = [p for p in model.parameters() if p.requires_grad]
    if weight_decay:
        # her adimda once W <- W - lr · weight_decay · W (kaybin desteklemedigi agirlik soner), sonra Adam adimi.
        # shift'in capasi var, fact_threshold bir esik: ikisine uygulanmaz
        named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([dict(params=[p for k, p in named if k.split(".")[-1].startswith("W_")], weight_decay=weight_decay),
                                 dict(params=[p for k, p in named if not k.split(".")[-1].startswith("W_")], weight_decay=0.0)],
                                lr=lr, **(dict(fused=True) if fused_adam else {}))
    else:   # fused=False'u acik vermek foreach'i de kapatir (tek tek dongu): yalniz True iken verilir
        opt = torch.optim.Adam(params, lr=lr, **(dict(fused=True) if fused_adam else {}))
    first = 0
    if checkpoint is not None:                             # surdurme: agirlik + Adam momentleri + adim
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        first = checkpoint["step"]
    loss_fn = (torch.compile(model.loss, fullgraph=True, dynamic=False, mode=compile if isinstance(compile, str) else None)
               if compile else model.loss)
    curve = []
    for step in range(first, steps + 1):
        resumed_here = checkpoint is not None and step == first
        if lr_floor is not None:                           # lr_t = lr · (floor + (1 - floor) · (1 + cos(π t / T)) / 2)
            for group in opt.param_groups:
                group["lr"] = lr * (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1))))
        if batches is not None:                            # mini-batch: bu adimin parcasi
            ids, mask = (t.to(device) for t in batches(step))
        with numerics(precision, attention_kernel, device, autocast=True):
            if compile == "reduce-overhead":               # CUDA graph: onceki adimin ciktilari artik kullanilmiyor
                torch.compiler.cudagraph_mark_step_begin()
            total, nll = loss_fn(ids, mask)                # butun cumleler (ya da adimin parcasi), butun konumlar
        if step in log_at:
            curve.append(dict(step=step, nll=nll.item(), dev=deviation(model).clone() if hasattr(model, "tokens") else None,
                              W_context=(sum(b.attention.W_context.norm().item() for b in model.blocks)
                                         if isinstance(model, BlockModel) else
                                         model.attention.W_context.norm().item() if getattr(model, "attention", None) is not None
                                         else 0.0)))
        if callback is not None and every and step % every == 0 and not resumed_here:
            callback(step, model, nll.item())
        if save is not None and save_every and step > 0 and step % save_every == 0 and not resumed_here:
            save(step, model, opt)
        if step == steps:
            break
        opt.zero_grad()                                    # onceki adimin gradyanlarini sil
        with numerics(precision, attention_kernel, device):   # geri hesap da ayni matmul ayariyla (TF32 / bf16 toplama)
            total.backward()                               # her ogrenilen sayi x icin ∂kayip/∂x (zincir kurali, otomatik)
        if grad_clip is not None:                          # butun gradyanlarin toplam boyu > grad_clip ise olcekle indir
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()                                         # Adam: x <- x - lr · m / (√v + eps); m, v gradyanin
                                                           # yuruyen ortalamasi ve karesininki (her sayi kendi adimini atar)
    return model, curve


@torch.no_grad()
def generate(model, prompts, n):
    """Acgozlu uretim: ayni uzunluktaki istemler birlikte, her birine n token.  -> id listeleri (yalniz uretilen)."""
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
