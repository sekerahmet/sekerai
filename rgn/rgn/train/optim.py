"""Optimizer: blokların 2-B matrisleri Muon / NorMuon'a; embedding, normlar, bias'lar, γ̂ ve β̂ AdamW'ye.

Muon güncellemesinin RMS'i AdamW'ninkiyle eşlenir (ölçek 0.2·sqrt(max(m, n)), torch.optim.Muon "match_rms_adamw"); bu
yüzden lr AdamW ölçeğinde verilir ve AI_NEW_MODEL/v2 ayarları (lr 2e-3) doğrudan taşınabilir.
"""
import math

import torch

NS_COEF = (3.4445, -4.7750, 2.0315)


def newton_schulz(G, steps=5, eps=1e-7):
    """G'nin yaklaşık ortogonal karşılığı (tekil değerler ~1), bf16."""
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = NS_COEF
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    """Momentum (Nesterov) + Newton–Schulz ortogonalleştirme.  normuon=True: NorMuon — çıktı nöronu (satır) başına ikinci
    moment ile normalizasyon, ardından ||güncelleme||_F = 0.2·sqrt(m·n)·lr."""

    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5, normuon=False,
                 beta2=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps, normuon=normuon, beta2=beta2))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p, dtype=torch.float32)
                grad = p.grad.float()
                st["m"].lerp_(grad, 1 - g["momentum"])
                upd = grad.lerp(st["m"], g["momentum"]) if g["nesterov"] else st["m"]
                O = newton_schulz(upd, g["ns_steps"]).float()
                rows, cols = p.shape
                if g["normuon"]:
                    if "v" not in st:
                        st["v"] = torch.zeros(rows, 1, device=p.device)
                    st["v"].lerp_(O.square().mean(-1, keepdim=True), 1 - g["beta2"])
                    O = O / (st["v"].sqrt() + 1e-10)
                    O = O * (0.2 * math.sqrt(rows * cols) / O.norm().clamp_min(1e-10))
                else:
                    O = O * (0.2 * math.sqrt(max(rows, cols)))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(O.to(p.dtype), alpha=-g["lr"])


class CombinedOptimizer:
    """Birden çok optimizer'ı tek optimizer gibi yönetir (lr takvimi tüm gruplara, checkpoint hepsine)."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)


def param_groups(model, use_muon):
    """-> (muon, decay, no_decay) parametre listeleri."""
    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if use_muon and name.startswith("blocks.") and p.dim() == 2:
            muon.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return muon, decay, no_decay


def build_optimizer(model, tcfg):
    use_muon = tcfg.optimizer in ("muon", "normuon")
    muon, decay, no_decay = param_groups(model, use_muon)
    adam_lr = tcfg.adam_lr or tcfg.lr
    fused = next(model.parameters()).is_cuda
    adamw = torch.optim.AdamW([dict(params=decay, weight_decay=tcfg.weight_decay, base_lr=adam_lr),
                               dict(params=no_decay, weight_decay=0.0, base_lr=adam_lr)],
                              lr=adam_lr, betas=tuple(tcfg.betas), fused=fused)
    opts = [adamw]
    if muon:
        opts.append(Muon([dict(params=muon, base_lr=tcfg.lr)], lr=tcfg.lr, momentum=tcfg.momentum,
                         weight_decay=tcfg.weight_decay, normuon=tcfg.optimizer == "normuon"))
    return CombinedOptimizer(opts)
