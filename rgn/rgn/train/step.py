"""Bir eğitim adımı (trainer ve scripts/profile_step.py ortak): mikro adımlar, kelimeye özel satır kopyası, kırpma,
optimizer.

Kelimeye özel nöron (model.priv_map varsa): adımdaki bütün mikro adımlarda geçen satırlar (uniq) CPU'da bulunur
(senkron yok); her katman için yalnız bu satırların kopyası (R, 3d) yaprak tensör olur.  İleri / geri geçiş bu kopyayla
(birleşik çekirdek), gradyanı kopyada toplanır; RowAdam yalnız bu satırları günceller.  Tam boy gradyan oluşmaz.
"""
import torch

from rgn.model.private_kernel import make_plan

_PMAP = {}


def _cpu_map(model):
    """priv_map'in CPU kopyası (token -> satır), model başına bir kez."""
    key = id(model)
    if key not in _PMAP:
        _PMAP[key] = model.priv_map.cpu()
    return _PMAP[key]


def private_rows(model, xs, device, dtype):
    """xs: mikro adımların token tensörleri (CPU) -> (uniq cihazda, uniq sayısı, katman başına satır kopyası) ya da
    (None, 0, None)."""
    if model.priv_map is None:
        return None, 0, None
    pm = _cpu_map(model)
    uniq = torch.unique(torch.cat([pm[x.flatten()] for x in xs]))
    u = uniq.to(device, non_blocking=True)
    sub = [b.private.detach().index_select(0, u).to(dtype).requires_grad_() for b in model.private_blocks()]
    return u, len(uniq), sub


def train_step(model, fwd, opt, batches, ctx, grad_clip, device, dtype):
    """batches: [(x, y, masks)] mikro adımlar (x, y CPU; masks cihazda ya da None).  -> (kayıp tensörü, grad normu)."""
    accum = len(batches)
    uniq, n_rows, sub = private_rows(model, [x for x, _, _ in batches], device, dtype)
    loss_sum = torch.zeros((), device=device)
    for x, y, masks in batches:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        priv = None
        if sub is not None:
            loc = torch.searchsorted(uniq, model.priv_map[x])
            priv = (loc, sub, make_plan(loc.flatten(), n_rows) if x.is_cuda else None)   # plan katmanlarda ortak
        with ctx:
            loss = fwd(x, y, masks, priv) / accum
        loss.backward()
        loss_sum += loss.detach()
    params = list(model.parameters()) + (sub or [])
    gn = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip else torch.zeros(())
    if sub is not None:
        opt.row_adam.set_rows(uniq, [s.grad for s in sub])
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss_sum, gn
