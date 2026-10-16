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

    _ns_compiled = None

    def _ns(self, X, steps):
        """Yığın (n, m, k) Newton–Schulz; CUDA'da derlenmiş (biçim başına bir kez)."""
        if X.is_cuda:
            if Muon._ns_compiled is None:
                Muon._ns_compiled = torch.compile(newton_schulz, dynamic=False)
            return Muon._ns_compiled(X, steps)
        return newton_schulz(X, steps)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            ps = [p for p in g["params"] if p.grad is not None]
            if not ps:
                continue
            for p in ps:
                if "m" not in self.state[p]:
                    self.state[p]["m"] = torch.zeros_like(p, dtype=torch.float32)
            grads = [p.grad.float() for p in ps]
            bufs = [self.state[p]["m"] for p in ps]
            torch._foreach_lerp_(bufs, grads, 1 - g["momentum"])
            upds = torch._foreach_lerp(grads, bufs, g["momentum"]) if g["nesterov"] else bufs
            groups = {}
            for i, p in enumerate(ps):
                groups.setdefault(tuple(p.shape), []).append(i)
            for (rows, cols), idx in groups.items():              # aynı biçimli matrisler tek yığında
                O = self._ns(torch.stack([upds[i] for i in idx]), g["ns_steps"]).float()
                if g["normuon"]:
                    for i in idx:
                        self.state[ps[i]].setdefault("v", torch.zeros(rows, 1, device=ps[i].device))
                    V = torch.stack([self.state[ps[i]]["v"] for i in idx])
                    V.lerp_(O.square().mean(-1, keepdim=True), 1 - g["beta2"])
                    for j, i in enumerate(idx):
                        self.state[ps[i]]["v"].copy_(V[j])
                    O = O / (V.sqrt() + 1e-10)
                    O = O * (0.2 * math.sqrt(rows * cols) / O.flatten(1).norm(dim=1).clamp_min(1e-10))[:, None, None]
                else:
                    O = O * (0.2 * math.sqrt(max(rows, cols)))
                for j, i in enumerate(idx):
                    p = ps[i]
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(O[j].to(p.dtype), alpha=-g["lr"])


def _row_adam_eager(p, m, v, rr, gr, lr, b1, b2, eps, bc1, bc2):
    mr = m.index_select(0, rr) * b1 + gr * (1 - b1)
    vr = v.index_select(0, rr) * b2 + gr * gr * (1 - b2)
    m.index_copy_(0, rr, mr)
    v.index_copy_(0, rr, vr)
    p.index_add_(0, rr, (-lr) * (mr / bc1) / ((vr / bc2).sqrt() + eps))


def _row_rms_eager(p, v, rr, gr, lr, b2, eps, bc2):
    vr = v.index_select(0, rr) * b2 + (gr * gr).mean(1, keepdim=True) * (1 - b2)
    v.index_copy_(0, rr, vr)
    p.index_add_(0, rr, (-lr) * gr / ((vr / bc2).sqrt() + eps))


_ROW_ADAM = {}


def _row_adam(p, *args):
    """CUDA'da derlenmiş (topla / güncelle / geri yaz tek çekirdek grubunda; satır sayısı dinamik)."""
    if not p.is_cuda:
        return _row_adam_eager(p, *args)
    if "fn" not in _ROW_ADAM:
        _ROW_ADAM["fn"] = torch.compile(_row_adam_eager, dynamic=True)
    return _ROW_ADAM["fn"](p, *args)


def _row_rms(p, *args):
    if not p.is_cuda:
        return _row_rms_eager(p, *args)
    if "rms" not in _ROW_ADAM:
        _ROW_ADAM["rms"] = torch.compile(_row_rms_eager, dynamic=True)
    return _ROW_ADAM["rms"](p, *args)


class RowAdam(torch.optim.Optimizer):
    """Satır seyrek Adam (kelimeye özel tablolar): yalnız `rows` satırları (o adımda geçen kelimeler) güncellenir;
    diğer satırların ağırlığı ve momentleri dokunulmaz (tembel Adam, gömme tabloları için standart).  Bias düzeltmesi
    adım sayısıyla.  rows trainer tarafından her adımdan önce verilir (set_rows)."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, mode="adam"):
        """mode adam: m, v satır satır | rms: momentsiz, satır başına tek ikinci moment (v (satır, 1))."""
        assert mode in ("adam", "rms")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.mode = mode
        self.rows, self.grads = None, None

    def set_rows(self, rows, grads=None):
        """rows: güncellenecek satırlar; grads: (varsa) parametre sırasıyla bu satırların gradyanları (len(rows), ...)
        — satır kopyası yolu (tam boy gradyan yok)."""
        self.rows, self.grads = rows, grads

    @torch.no_grad()
    def step(self):
        r = self.rows
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for i, p in enumerate(g["params"]):
                if self.grads is None and p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    if self.mode == "rms":
                        st["v"] = torch.zeros(p.shape[0], 1, device=p.device)
                    else:
                        st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    st["step"] = 0
                st["step"] += 1
                rr = r if r is not None else torch.arange(p.shape[0], device=p.device)
                gr = (self.grads[i] if self.grads is not None else p.grad.index_select(0, rr)).float()
                t = st["step"]
                sc = lambda z: torch.tensor(z, dtype=torch.float32, device=p.device)  # noqa: E731 (derlemede sabit değil)
                if self.mode == "rms":
                    _row_rms(p, st["v"], rr, gr, sc(g["lr"]), b2, g["eps"], sc(1 - b2 ** t))
                else:
                    _row_adam(p, st["m"], st["v"], rr, gr, sc(g["lr"]), b1, b2, g["eps"], sc(1 - b1 ** t), sc(1 - b2 ** t))


class CombinedOptimizer:
    """Birden çok optimizer'ı tek optimizer gibi yönetir (lr takvimi tüm gruplara, checkpoint hepsine)."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.row_adam = next((o for o in optimizers if isinstance(o, RowAdam)), None)

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
        if ".private" in name:                     # kelimeye özel tablo: satır seyrek Adam (build_optimizer)
            continue
        elif use_muon and name.startswith("blocks.") and p.dim() == 2:
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
    private = [p for n, p in model.named_parameters() if ".private" in n and p.requires_grad]
    if private:                                  # kelimeye özel tablolar: yalnız o adımda geçen kelimelerin satırları
        opts.append(RowAdam([dict(params=private, base_lr=adam_lr)], lr=adam_lr, betas=tuple(tcfg.betas),
                            mode=tcfg.private_opt))
    if muon:
        opts.append(Muon([dict(params=muon, base_lr=tcfg.lr)], lr=tcfg.lr, momentum=tcfg.momentum,
                         weight_decay=tcfg.weight_decay, normuon=tcfg.optimizer == "normuon"))
    return CombinedOptimizer(opts)
