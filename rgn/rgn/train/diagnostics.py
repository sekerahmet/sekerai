"""RGN'ye özgü teşhisler (doküman V4 / V5 riskleri) — tek bir doğrulama batch'i üzerinde.

    gamma, beta      head başına öğrenilmiş γ ve β
    mlp              rasyonel MLP: ön-aktivasyon u'nun std'si ve |u| > 3 oranı (doyma)
    attn             head başına: ortalama entropi (nats), en büyük ağırlık (odak), ortalama dikkat mesafesi (token)
                     — son yarı konumlarda (bağlamı dolu satırlar)
"""
import torch


@torch.no_grad()
def diagnose(model, x, ctx):
    was_training = model.training
    model.eval()
    inputs = {}
    hooks = [blk.attn.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args))
             for i, blk in enumerate(model.blocks)]
    for blk in model.blocks:
        blk.mlp.track = True
    try:
        with ctx:
            model.hidden(x)
    finally:
        for h in hooks:
            h.remove()
        for blk in model.blocks:
            blk.mlp.track = False
    out = {"layers": []}
    N = x.shape[1]
    half = slice(N // 2, N)
    dist = (torch.arange(N, device=x.device)[:, None] - torch.arange(N, device=x.device)[None, :]).clamp_min(0).float()
    for i, blk in enumerate(model.blocks):
        a = blk.attn.weights(*inputs[i]).float()                    # (B, H, N, N)
        ent = -(a * a.clamp_min(1e-30).log()).sum(-1)               # (B, H, N)
        layer = {
            "entropy": ent[:, :, half].mean((0, 2)).tolist(),
            "top1": a.max(-1).values[:, :, half].mean((0, 2)).tolist(),
            "mean_dist": (a * dist).sum(-1)[:, :, half].mean((0, 2)).tolist(),
            "mlp": dict(blk.mlp.stats),
        }
        if blk.attn.cfg.attn == "rgn":
            layer["gamma"] = blk.attn.gamma().tolist()
            b = blk.attn.beta()
            layer["beta"] = None if b is None else b.tolist()
            k = blk.attn.sharpness()
            layer["sharpness"] = k.tolist() if torch.is_tensor(k) else [float(k)] * blk.attn.H
        out["layers"].append(layer)
    model.train(was_training)
    return out


def summary_line(diag):
    """Kısa özet: katmanlar ortalaması entropi / top1 / mesafe ve MLP doyma oranı."""
    L = diag["layers"]
    mean = lambda key: sum(sum(l[key]) / len(l[key]) for l in L) / len(L)  # noqa: E731
    sat = [l["mlp"].get("sat_frac") for l in L if l["mlp"]]
    s = "entropi %.2f | top1 %.3f | mesafe %.1f" % (mean("entropy"), mean("top1"), mean("mean_dist"))
    if sat:
        s += " | |u|>3 %.4f" % (sum(sat) / len(sat))
    return s
