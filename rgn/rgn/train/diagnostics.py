"""Teşhis — tek bir doğrulama batch'i üzerinde, katman başına:

    attn   head başına ortalama entropi (nats), en büyük pay (odak), ortalama dikkat mesafesi (token) — son yarı konumlar
    mlp    SwiGLU kapı girdisinin std'si ve |G| > 3 oranı
"""
import torch


@torch.no_grad()
def diagnose(model, x, ctx):
    was_training = model.training
    model.eval()
    inputs = {}
    hooks = [blk.attn.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args))
             for i, blk in enumerate(model.blocks) if blk.attn is not None]           # evrişimli Y katmanı atlanır
    mlps = [blk.mlp for blk in model.blocks if blk.mlp is not None]
    for mlp in mlps:
        mlp.track = True
    try:
        with ctx:
            model.hidden(x)
    finally:
        for h in hooks:
            h.remove()
        for mlp in mlps:
            mlp.track = False
    out = {"layers": []}
    N = x.shape[1]
    half = slice(N // 2, N)
    dist = (torch.arange(N, device=x.device)[:, None] - torch.arange(N, device=x.device)[None, :]).clamp_min(0).float()
    for i, blk in enumerate(model.blocks):
        if blk.attn is None:
            continue
        a = blk.attn.weights(*inputs[i][:2]).float()                # (B, H, N, N); girdi x, rope
        ent = -(a * a.clamp_min(1e-30).log()).sum(-1)
        out["layers"].append({
            "entropy": ent[:, :, half].mean((0, 2)).tolist(),
            "top1": a.max(-1).values[:, :, half].mean((0, 2)).tolist(),
            "mean_dist": (a * dist).sum(-1)[:, :, half].mean((0, 2)).tolist(),
            "mlp": dict(blk.mlp.stats) if blk.mlp is not None else {},
        })
    model.train(was_training)
    return out


def summary_line(diag):
    """Kısa özet: katman ortalaması entropi / top1 / mesafe ve MLP doyma oranı."""
    L = diag["layers"]
    mean = lambda key: sum(sum(l[key]) / len(l[key]) for l in L) / len(L)  # noqa: E731
    sat = [l["mlp"].get("sat_frac") for l in L if l["mlp"]]
    s = "entropi %.2f | top1 %.3f | mesafe %.1f" % (mean("entropy"), mean("top1"), mean("mean_dist"))
    if sat:
        s += " | |u|>3 %.4f" % (sum(sat) / len(sat))
    return s
