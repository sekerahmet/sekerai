"""Optimizer: blokların 2-B matrisleri Muon / NorMuon'a; gömme, normlar, bias'lar, kapı AdamW'ye; kelimeye özel
tablolar satır seyrek RowAdam'a.

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


def _row_adam_eager(p, m, v, n, rr, gr, lr, b1, b2, eps):
    nr = n.index_select(0, rr) + 1
    mr = m.index_select(0, rr) * b1 + gr * (1 - b1)
    vr = v.index_select(0, rr) * b2 + gr * gr * (1 - b2)
    n.index_copy_(0, rr, nr)
    m.index_copy_(0, rr, mr)
    v.index_copy_(0, rr, vr)
    p.index_add_(0, rr, (-lr) * (mr / (1 - b1 ** nr)) / ((vr / (1 - b2 ** nr)).sqrt() + eps))


def _row_rms_eager(p, v, n, rr, gr, lr, b2, eps):
    nr = n.index_select(0, rr) + 1
    vr = v.index_select(0, rr) * b2 + (gr * gr).mean(1, keepdim=True) * (1 - b2)
    n.index_copy_(0, rr, nr)
    v.index_copy_(0, rr, vr)
    p.index_add_(0, rr, (-lr) * gr / ((vr / (1 - b2 ** nr)).sqrt() + eps))


def _row_rms_own_eager(p, v, n, rr, gr, lr, b2, eps, n0):
    """rms + satırın kendi takvimi: lr · sqrt(n0 / (n + n0)), n o satırın önceki güncelleme sayısı."""
    nr = n.index_select(0, rr) + 1
    vr = v.index_select(0, rr) * b2 + (gr * gr).mean(1, keepdim=True) * (1 - b2)
    n.index_copy_(0, rr, nr)
    v.index_copy_(0, rr, vr)
    own = (n0 / (nr - 1 + n0)).sqrt()
    p.index_add_(0, rr, (-lr) * own * gr / ((vr / (1 - b2 ** nr)).sqrt() + eps))


_ROW = {}


def _row_update(name, fn, p, *args):
    """CUDA'da derlenmiş (topla / güncelle / geri yaz; satır sayısı dinamik)."""
    if not p.is_cuda:
        return fn(p, *args)
    if name not in _ROW:
        _ROW[name] = torch.compile(fn, dynamic=True)
    return _ROW[name](p, *args)


class RowAdam(torch.optim.Optimizer):
    """Satır seyrek Adam (kelimeye özel tablolar): yalnız `rows` satırları (o adımda geçen kelimeler) güncellenir; diğer
    satırların ağırlığı ve durumu dokunulmaz (tembel Adam).  Bias düzeltmesi SATIR BAŞINA güncelleme sayısıyla (n): ilk
    kez görülen satır, genel adım sayısından bağımsız olarak doğru ölçekle başlar.
    mode rms: momentsiz, satır başına tek ikinci moment (v (satır, 1)) | adam: m, v eleman eleman."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, mode="rms", row_decay=0.0, row_mult=None):
        assert mode in ("adam", "rms")
        assert not row_decay or mode == "rms", "row_decay yalnız rms modunda"
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.mode = mode
        self.row_decay = float(row_decay)              # n0 > 0: satırın kendi takvimi (rms)
        self.row_mult = row_mult                       # (satır,) kelime başına lr çarpanı ya da None
        self.rows, self.grads = None, None
        self._lr = None                                    # lr cihazda kalıcı (her adım fill_: CPU -> GPU senkronu yok)

    def set_rows(self, rows, grads=None):
        """rows: güncellenecek satırlar; grads: parametre sırasıyla bu satırların gradyanları (len(rows), ...) — satır
        kopyası yolu; None ise p.grad'dan okunur."""
        self.rows, self.grads = rows, grads

    def _state(self, p):
        st = self.state[p]
        if not st:
            st["n"] = torch.zeros(p.shape[0], 1, device=p.device)
            st["v"] = torch.zeros(p.shape[0], 1, device=p.device) if self.mode == "rms" else torch.zeros_like(p)
            if self.mode == "adam":
                st["m"] = torch.zeros_like(p)
        assert ("m" in st) == (self.mode == "adam"), "RowAdam durumu %s modunda değil (private_opt değişmiş)" % self.mode
        if "n" not in st:                                  # eski checkpoint (genel adım sayılı): sayaç oradan başlar
            st["n"] = torch.full((p.shape[0], 1), float(st.pop("step", 0)), device=p.device)
        return st

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for i, p in enumerate(g["params"]):
                if self.grads is None and p.grad is None:
                    continue
                if self._lr is None:
                    self._lr = torch.zeros((), device=p.device)
                self._lr.fill_(g["lr"])
                if self.row_decay and getattr(self, "_n0", None) is None:
                    self._n0 = torch.tensor(self.row_decay, device=p.device)
                st = self._state(p)
                rr = self.rows if self.rows is not None else torch.arange(p.shape[0], device=p.device)
                lr = self._lr
                if self.row_mult is not None:              # kelime başına lr: (len(rr), 1)
                    if self.row_mult.device != p.device:
                        self.row_mult = self.row_mult.to(p.device)
                    lr = self._lr * self.row_mult.index_select(0, rr).unsqueeze(1)
                gr = (self.grads[i] if self.grads is not None else p.grad.index_select(0, rr)).float()
                if self.mode == "rms" and self.row_decay:
                    _row_update("rms_own", _row_rms_own_eager, p, st["v"], st["n"], rr, gr, lr, b2, g["eps"], self._n0)
                elif self.mode == "rms":
                    _row_update("rms" if lr is self._lr else "rms_mult", _row_rms_eager, p, st["v"], st["n"], rr, gr, lr,
                                b2, g["eps"])
                else:
                    _row_update("adam" if lr is self._lr else "adam_mult", _row_adam_eager, p, st["m"], st["v"], st["n"],
                                rr, gr, lr, b1, b2, g["eps"])


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
        if ".private" in name:                     # kelimeye özel tablo: RowAdam (build_optimizer)
            continue
        if ".vgate." in name:                       # V vektörü kapısı (12 girdi): AdamW, ağırlık sönümsüz
            no_decay.append(p)
        elif use_muon and name.startswith("blocks.") and p.dim() == 2:
            muon.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return muon, decay, no_decay


def row_lr_mult(mcfg, tcfg):
    """Kelime başına lr çarpanı (satır,): clamp(sqrt(ref / sayı), 1, max); satır 0 (ortak "diğer") 1.  None: kapalı."""
    if not tcfg.private_lr_ref:
        return None
    import json
    from rgn.paths import ROOT
    counts = json.loads((ROOT / mcfg.private_vocab).read_text(encoding="utf-8"))["counts"]
    c = torch.tensor([float("inf")] + list(counts), dtype=torch.float32).clamp_min(1.0)
    return (tcfg.private_lr_ref / c).sqrt().clamp(1.0, tcfg.private_lr_max)


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
        plr = tcfg.private_lr or adam_lr
        opts.append(RowAdam([dict(params=private, base_lr=plr, own_schedule=bool(tcfg.private_row_decay))], lr=plr,
                            betas=tuple(tcfg.betas), mode=tcfg.private_opt, row_decay=tcfg.private_row_decay,
                            row_mult=row_lr_mult(model.cfg, tcfg)))
    if muon:
        opts.append(Muon([dict(params=muon, base_lr=tcfg.lr)], lr=tcfg.lr, momentum=tcfg.momentum,
                         weight_decay=tcfg.weight_decay, normuon=tcfg.optimizer == "normuon"))
    return CombinedOptimizer(opts)
