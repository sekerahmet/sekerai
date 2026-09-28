# -*- coding: utf-8 -*-
"""measure_precision -- TF32 ve bf16'nin Model X'e ve ayni boyuttaki transformer'a etkisi, CPU taklidiyle (GPU yok).
TinyStories ayari (sozluk 8.004, D=384, FactUnits 1.536, 2 tur / 2 katman, pencere 512), gercek valid hikayeleri.
Her cagri tek is yapar (her biri < 3 dk):

    python measure_precision.py prepare <model>          fp32 PREP_STEPS adim egitir -> <out>/<model>_prep.pt
    python measure_precision.py forward <model>          ayni agirlikla ileri hesap: fp64 / tf32 / bf16 -- fp32'ye gore
    python measure_precision.py curve <model> <prec>     hazir agirliktan CURVE_STEPS adim; kayip egrisi -> json
    python measure_precision.py report                   egrileri fp32 ile kiyaslar
    python measure_precision.py checkpoint <yol> <ad>    Colab kosusunun surdurme paketi (agirlik + Adam + adim) -> <ad>;
                                                         egri o adimin lr'si ve Adam momentleriyle devam eder

<model>: modelx | transformer | checkpoint'ten <ad> (Model X).  <prec>: fp32 | fp64 | tf32_nearest | tf32_truncate | bf16
(bf16_emulation, SDPA math yolu); ileri hesapta ayrica bf16_fused_attention ve bf16_fp32_head.
Olcu: logit farki (max, ortalama, logit std'sine oranla), KL(p_fp32 || p), nll farki, top-1 ayni mi; egride adim adim
|nll farki| ve son agirliklarin fp32 kosusundan uzakligi (guncellemenin boyuna oranla).
"""
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train_tinystories"))
sys.path.insert(0, HERE)

import data_tinystories as DT  # noqa: E402
import model_20 as M  # noqa: E402
import model_20_transformer as MT  # noqa: E402
from numerics_emulation import bf16_emulation, cuda_bf16_autocast, tf32_emulation  # noqa: E402

CACHE = os.environ.get("TS_CACHE", r"G:\Drive'ım\tinystories\onbellek")
OUT = os.environ.get("PREC_OUT", os.path.join(tempfile.gettempdir(), "model20_precision"))
V, T, D_MODEL, UNITS = 8004, 512, 384, 1536
BATCH, PREP_STEPS, CURVE_STEPS, LR, CLIP = 2, 30, 15, 0.01, 1.0
EVAL_ROWS = 4


def valid_data():
    vocab = [str(a) for a in np.load(os.path.join(CACHE, "sozluk_%s.npy" % DT.TAG), allow_pickle=True)]
    a = np.load(os.path.join(CACHE, "akis_valid_%s.npy" % DT.TAG))
    ends = np.flatnonzero(a == vocab.index(DT.EOS_TOKEN))
    starts = np.concatenate([[0], ends[:-1] + 1])
    lengths = ends - starts
    keep = (lengths > 0) & (lengths + 2 <= T)
    assert len(vocab) == V
    return dict(vocab=vocab, seq_len=T, valid=a, valid_start=starts[keep], valid_length=lengths[keep])


def rows(data, which):
    """Sabit pencereler: egitim adimlari icin 'train' (ilk 10.000'in permutasyonu), olcum icin 'eval' (sonrakiler)."""
    n = len(data["valid_start"])
    perm = np.random.default_rng(0).permutation(n)
    return perm[:10000] if which == "train" else perm[10000:10000 + EVAL_ROWS]


def build(name):
    if name.startswith("modelx"):
        return M.BlockModel(V, d=D_MODEL, units=UNITS, turns=2, t_max=T, seed=0)
    return MT.TransformerModel(V, d=D_MODEL, layers=2, heads=1, units=UNITS, seed=0)


def context(prec):
    if prec.startswith("tf32"):
        return tf32_emulation(prec.split("_")[1])
    if prec == "bf16":
        return bf16_emulation()
    if prec == "bf16_fused_attention":               # CPU bf16 cekirdekleri, SDPA flash (P bf16'ya yuvarlanir: CUDA
        return cuda_bf16_autocast()                  # flash / mem-efficient gibi); yavas, yalniz ileri hesap
    return torch.autocast("cpu", enabled=False)


def train(model, data, steps, prec, first_step=0, opt_state=None):
    """Adam (lr 0,01 sabit: gercek kosuda cosine 41.602 adima yayili, ilk adimlarda ~0,01), clip 1,0.  opt_state:
    checkpoint'in Adam durumu ve lr'si.  -> nll listesi."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    if opt_state is not None:
        opt.load_state_dict(opt_state["optimizer"])
        for group in opt.param_groups:
            group["lr"] = opt_state["lr"]
    order, out = rows(data, "train"), []
    dtype = torch.float64 if prec == "fp64" else torch.float32
    for s in range(first_step, first_step + steps):
        ids, mask = DT.sequences(data, "valid", order[s * BATCH:(s + 1) * BATCH])
        with context(prec):
            total, nll = model.loss(ids, mask)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        out.append(nll.item())
        assert all(p.dtype == dtype for p in params)
    return out


def load_prepared(name, dtype=torch.float32):
    model = build(name)
    model.load_state_dict(torch.load(os.path.join(OUT, "%s_prep.pt" % name)))
    return model.to(dtype)


def prepare(name, data):
    torch.manual_seed(0)
    model = build(name)
    t = time.time()
    curve = train(model, data, PREP_STEPS, "fp32")
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "%s_prep.pt" % name))
    print("%s: %d adim fp32, nll %.3f -> %.3f  (%.0f sn)" % (name, PREP_STEPS, curve[0], curve[-1], time.time() - t))


def logits_fp32_head(model, ids):
    """bf16 govde, cikis katmani (8.004 genis matmul) fp32: cikisin payini ayirmak icin."""
    with bf16_emulation():
        if isinstance(model, M.BlockModel):
            h = model.hidden(ids)[-1]
        else:
            h = model.embedding(ids)
            for layer in model.layers:
                h = layer(h)
    if isinstance(model, M.BlockModel):
        return model.scale * h.float() @ model.tokens.points().T
    return model.norm_final(h.float()) @ model.embedding.weight.T


@torch.no_grad()
def forward(name, data):
    ids, mask = DT.sequences(data, "valid", rows(data, "eval"))
    valid = mask[:, 1:]
    ref_model = load_prepared(name)
    ref = ref_model.logits(ids[:, :-1])[valid].double()
    ref_nll = F.cross_entropy(ref, ids[:, 1:][valid])
    logp_ref = ref.log_softmax(-1)
    std = ref.std().item()
    print("%s: %d konum, logit std %.3f, fp32 nll %.4f" % (name, int(valid.sum()), std, ref_nll.item()))
    result = {}
    for prec in ("fp64", "tf32_nearest", "tf32_truncate", "bf16", "bf16_fused_attention", "bf16_fp32_head"):
        model = load_prepared(name, torch.float64 if prec == "fp64" else torch.float32)
        if prec == "bf16_fp32_head":
            out = logits_fp32_head(model, ids[:, :-1])[valid].double()
        else:
            with context(prec):
                out = model.logits(ids[:, :-1])[valid].double()
        diff = (out - ref).abs()
        kl = (logp_ref.exp() * (logp_ref - out.log_softmax(-1))).sum(-1)
        nll = F.cross_entropy(out, ids[:, 1:][valid])
        r = dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(), max_over_std=diff.max().item() / std,
                 kl_mean=kl.mean().item(), kl_max=kl.max().item(), dnll=nll.item() - ref_nll.item(),
                 top1_same=(out.argmax(-1) == ref.argmax(-1)).double().mean().item())
        result[prec] = r
        print("  %-14s max|dlogit| %.2e  ort %.2e  max/std %.2e  KL ort %.2e max %.2e  dnll %+.2e  top1 ayni %.4f" % (
            prec, r["max_abs"], r["mean_abs"], r["max_over_std"], r["kl_mean"], r["kl_max"], r["dnll"], r["top1_same"]))
    json.dump(result, open(os.path.join(OUT, "%s_forward.json" % name), "w"), indent=1)


def from_checkpoint(path, name):
    """Surdurme paketi -> <ad>_prep.pt (agirlik) + <ad>_opt.pt (Adam durumu, o adimin cosine lr'si)."""
    pack = torch.load(path, map_location="cpu")
    step, steps = pack["step"], 41602
    lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
    os.makedirs(OUT, exist_ok=True)
    torch.save(pack["model"], os.path.join(OUT, "%s_prep.pt" % name))
    torch.save(dict(optimizer=pack["optimizer"], lr=lr), os.path.join(OUT, "%s_opt.pt" % name))
    print("%s: adim %d, lr %.5f" % (name, step, lr))


def curve(name, prec, data):
    model = load_prepared(name, torch.float64 if prec == "fp64" else torch.float32)
    start = [p.detach().double().clone() for p in model.parameters()]
    t = time.time()
    opt_path = os.path.join(OUT, "%s_opt.pt" % name)
    opt_state = torch.load(opt_path) if os.path.exists(opt_path) else None
    nll = train(model, data, CURVE_STEPS, prec, first_step=PREP_STEPS, opt_state=opt_state)
    torch.save([p.detach().double() for p in model.parameters()], os.path.join(OUT, "%s_%s_params.pt" % (name, prec)))
    torch.save(start, os.path.join(OUT, "%s_start.pt" % name))
    json.dump(nll, open(os.path.join(OUT, "%s_%s_curve.json" % (name, prec)), "w"))
    print("%s %s: %d adim, nll %.4f -> %.4f  (%.0f sn)" % (name, prec, CURVE_STEPS, nll[0], nll[-1], time.time() - t))


@torch.no_grad()
def eval_nll(name, params, data):
    """Kosunun son agirligi, sabit olcum pencerelerinde fp32 nll (egitim batch'inin gurultusu olmadan)."""
    model = build(name)
    for p, v in zip(model.parameters(), params):
        p.copy_(v.float())
    ids, mask = DT.sequences(data, "valid", rows(data, "eval"))
    return model.loss(ids, mask)[1].item()


def report():
    data = valid_data()
    for name in sorted({f.split("_fp32_curve")[0] for f in os.listdir(OUT) if f.endswith("_fp32_curve.json")}):
        path = lambda p, k: os.path.join(OUT, "%s_%s_%s" % (name, p, k))
        if not os.path.exists(path("fp32", "curve.json")):
            continue
        base = json.load(open(path("fp32", "curve.json")))
        base_p = torch.load(path("fp32", "params.pt"))
        start = torch.load(os.path.join(OUT, "%s_start.pt" % name))
        update = math.sqrt(sum(((b - s) ** 2).sum().item() for b, s in zip(base_p, start)))
        base_eval = eval_nll(name, base_p, data)
        print("%s: fp32 %d adim, nll %.4f -> %.4f, guncelleme boyu %.3f, son agirlik olcum nll %.5f" % (
            name, len(base), base[0], base[-1], update, base_eval))
        for prec in ("fp64", "tf32_nearest", "tf32_truncate", "bf16"):
            if not os.path.exists(path(prec, "curve.json")):
                continue
            c = json.load(open(path(prec, "curve.json")))
            p = torch.load(path(prec, "params.pt"))
            d = [abs(a - b) for a, b in zip(c, base)]
            dist = math.sqrt(sum(((a - b) ** 2).sum().item() for a, b in zip(p, base_p)))
            print("  %-14s |dnll| ilk %.1e  max %.1e  son %.1e  ort %.1e | agirlik uzakligi / guncelleme %.2e | "
                  "olcum nll farki %+.1e" % (prec, d[0], max(d), d[-1], sum(d) / len(d), dist / update,
                                            eval_nll(name, p, data) - base_eval))


if __name__ == "__main__":
    job = sys.argv[1]
    if job == "report":
        report()
    elif job == "checkpoint":
        from_checkpoint(sys.argv[2], sys.argv[3])
    else:
        data = valid_data()
        if job == "prepare":
            prepare(sys.argv[2], data)
        elif job == "forward":
            forward(sys.argv[2], data)
        elif job == "curve":
            curve(sys.argv[2], sys.argv[3], data)
