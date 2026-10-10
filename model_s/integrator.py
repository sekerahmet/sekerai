"""integrator -- Model S Parca 2: zaman adimi semalari (TASARIM.md s3, Parca 2).

Bir adim (bugunku "katman") u_n -> u_{n+1}; operatorler (P, A, F) operators.py'den, sira ve katsayi semadan:
    lie       u <- u + dt A(u);  u <- u + dt F(u)                         1. derece; bugunku pre-norm transformer blogu
    strang    u <- u + dt/2 F1(u);  u <- u + dt A(u);  u <- u + dt/2 F2(u)  2. derece (Macaron Net 2019); F1, F2 yarim genislik:
                                                                          parametre lie ile ayni
    parallel  u <- u + dt (A(u) + F(u))                                   ayni u'dan iki operator (makale denklem 75; GPT-J / PaLM)
    mix       u <- u + dt M(u)                                            tek operator (operators.MixOp): bolme yok; values
                                                                          linear = parallel (birlesik matmul), swiglu = yeni
dt = 1 (bugunku transformer; agirliklar olcegi tasir).  Adim basina parametre bagimsiz (theta_n); purussuz theta(t) Parca 3.
"""
import torch

from operators import AttentionOp, FeedForwardOp, MixOp, mix_hidden

SCHEMES = ("lie", "strang", "parallel", "mix")


class Step(torch.nn.Module):
    """Tek zaman adimi; operatorleri semanin sirasiyla uygular, yeni durumu dondurur."""

    def __init__(self, d, heads, hidden, kv_heads=None, scheme="lie", dt=1.0, width=1, values="swiglu", qk_dim=None,
                 keep=1.0):
        """width k: birlesik hibrit asama (Parca 3) -- k x heads (head boyu d / heads), k x hidden; = k adimin `wide` formulu.
        mix: hidden ayri A + F'nin genisligi; MixOp'un yerel genisligi esit matris parametresiyle (mix_hidden)."""
        super().__init__()
        assert scheme in SCHEMES, "scheme: %s" % (SCHEMES,)
        self.scheme, self.dt, self.width = scheme, float(dt), int(width)
        if scheme == "mix":
            kv, hd = (width * kv_heads if kv_heads else None), d // heads
            hid = mix_hidden(d, width * heads * hd, (kv or width * heads) * hd, width * hidden, values)
            self.M = MixOp(d, width * heads, hid, kv, hd, values, qk_dim=qk_dim, keep=keep)
            return
        self.A = AttentionOp(d, width * heads, width * kv_heads if kv_heads else None, head_dim=d // heads, qk_dim=qk_dim)
        hidden = width * hidden
        if scheme == "strang":                                          # esit parametre: iki yarim genislik
            self.F1, self.F2 = FeedForwardOp(d, hidden // 2, keep), FeedForwardOp(d, hidden - hidden // 2, keep)
        else:
            self.F = FeedForwardOp(d, hidden, keep)

    def forward(self, u, ctx):
        dt = self.dt
        if self.scheme == "mix":
            return u + dt * self.M(u, ctx)
        if self.scheme == "lie":
            u = u + dt * self.A(u, ctx)
            return u + dt * self.F(u)
        if self.scheme == "strang":
            u = u + 0.5 * dt * self.F1(u)
            u = u + dt * self.A(u, ctx)
            return u + 0.5 * dt * self.F2(u)
        return u + dt * (self.A(u, ctx) + self.F(u))                    # parallel


class Stage(torch.nn.Module):
    """Parca 3, hibrit bolme (makale Ek C; kullanici, 11 Ekim: "tüm bu sıralı işlemi tek bir formüle ya da daha basit bir iki
    formüle dökebilirsin"): asamadaki k adim AYNI u'dan, artislari toplanir:
        u <- u + sum_i (S_i(u) - u)
    = ardisik k adimin BCH acilimindaki ilk terim; atilan kisim 1/2 sum_{i<j} [D_j, D_i] (adim boyunun karesiyle; rastgele
    agirlikla olculdu).  k = 1: tek sirali adim (bugunku transformer).  GPU bicimi Step(width=k): A'lar toplanir, F'ler
    toplanmis A'dan sonraki durumu gorur (`wide`; egitilmis d768'de sifir atista `sum`dan iyi, commute_probe)."""

    def __init__(self, steps):
        super().__init__()
        self.steps = torch.nn.ModuleList(steps)

    def forward(self, u, ctx):
        if len(self.steps) == 1:
            return self.steps[0](u, ctx)
        return u + sum(s(u, ctx) - u for s in self.steps)


def stages(d, heads, hidden, groups, kv_heads=None, scheme="lie"):
    """groups: asama basina adim sayisi (ornek (1, 1, 2, 2, 2, 2, 2): 12 adim, 7 sirali asama)."""
    return torch.nn.ModuleList(Stage([Step(d, heads, hidden, kv_heads, scheme) for _ in range(k)]) for k in groups)
