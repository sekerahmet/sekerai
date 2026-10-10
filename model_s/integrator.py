"""integrator -- Model S Parca 2: zaman adimi semalari (TASARIM.md s3, Parca 2).

Bir adim (bugunku "katman") u_n -> u_{n+1}; operatorler (P, A, F) operators.py'den, sira ve katsayi semadan:
    lie       u <- u + dt A(u);  u <- u + dt F(u)                         1. derece; bugunku pre-norm transformer blogu
    strang    u <- u + dt/2 F1(u);  u <- u + dt A(u);  u <- u + dt/2 F2(u)  2. derece (Macaron Net 2019); F1, F2 yarim genislik:
                                                                          parametre lie ile ayni
    parallel  u <- u + dt (A(u) + F(u))                                   ayni u'dan iki operator (makale denklem 75; GPT-J / PaLM)
dt = 1 (bugunku transformer; agirliklar olcegi tasir).  Adim basina parametre bagimsiz (theta_n); purussuz theta(t) Parca 3.
"""
import torch

from operators import AttentionOp, FeedForwardOp

SCHEMES = ("lie", "strang", "parallel")


class Step(torch.nn.Module):
    """Tek zaman adimi; operatorleri semanin sirasiyla uygular, yeni durumu dondurur."""

    def __init__(self, d, heads, hidden, kv_heads=None, scheme="lie", dt=1.0):
        super().__init__()
        assert scheme in SCHEMES, "scheme: %s" % (SCHEMES,)
        self.scheme, self.dt = scheme, float(dt)
        self.A = AttentionOp(d, heads, kv_heads)
        if scheme == "strang":                                          # esit parametre: iki yarim genislik
            self.F1, self.F2 = FeedForwardOp(d, hidden // 2), FeedForwardOp(d, hidden - hidden // 2)
        else:
            self.F = FeedForwardOp(d, hidden)

    def forward(self, u, ctx):
        dt = self.dt
        if self.scheme == "lie":
            u = u + dt * self.A(u, ctx)
            return u + dt * self.F(u)
        if self.scheme == "strang":
            u = u + 0.5 * dt * self.F1(u)
            u = u + dt * self.A(u, ctx)
            return u + 0.5 * dt * self.F2(u)
        return u + dt * (self.A(u, ctx) + self.F(u))                    # parallel
