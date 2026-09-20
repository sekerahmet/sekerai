# -*- coding: utf-8 -*-
"""Modelin OGRENDIGI uzayi cizer. `kos_13.py --kaydet ...` ciktisini okur.

Dort panel, dordu de bir soruya cevap veriyor:
  1  varlik bulutu   -- tipler ayristi mi
  2  iliski boyu     -- hangi Rel[r] SIFIRA cokmus (dejenere cozum)
  3  yuruyus         -- q = c_ozne + Rel[r1] + Rel[r2] nereye dusuyor
  4  cakisma         -- ayni hedefi paylasanlar ust uste mi
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

NPZ = sys.argv[1] if len(sys.argv) > 1 else "durum_serbest.npz"
CIK = sys.argv[2] if len(sys.argv) > 2 else "uzay.png"
d = np.load(NPZ, allow_pickle=True)
C, Rel, tip = d["C"], d["Rel"], d["tip"]
tip_ad, ad, rel_ad = list(d["tip_ad"]), list(d["ad"]), list(d["rel_ad"])
facts, comp = d["facts"], d["comp"]
kip, epok = str(d["kip"]), int(d["epok"])

# --- PCA: ayni izdusum her panelde kullanilir ki paneller KIYASLANABILSIN
mu = C.mean(0)
U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
P = Vt[:2]                                   # (2, DD)
pr = lambda X: (np.atleast_2d(X) - mu) @ P.T
Z = pr(C)
acik = S[:2] ** 2 / (S ** 2).sum()

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "figure.facecolor": "white"})
fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.suptitle(f"model_13 · zincir · kip={kip} · {epok} epok · "
             f"ogrenilen uzay (PCA, %{100*acik.sum():.0f} varyans)",
             fontsize=12, y=0.98)

# ---------------------------------------------------------------- 1
A = ax[0, 0]
renk = plt.cm.tab10(np.linspace(0, 1, len(tip_ad)))
for t, tad in enumerate(tip_ad):
    ix = np.where(tip == t)[0]
    A.scatter(Z[ix, 0], Z[ix, 1], s=7, alpha=.55, color=renk[t],
              label=f"{tad} ({len(ix)})", linewidths=0)
A.legend(fontsize=7, markerscale=1.6, loc="best", framealpha=.9)
A.set_title("1 — varlik bulutu, tipe gore\nayrisma varsa tipler ayri kumelenir")
A.set_xlabel("PC1"); A.set_ylabel("PC2")

# ---------------------------------------------------------------- 2
A = ax[0, 1]
boy = np.sqrt((Rel ** 2).sum(-1))
sira = np.argsort(boy)
olc = np.sqrt(((C[np.random.default_rng(0).integers(0, len(C), 4000)]
                - C[np.random.default_rng(1).integers(0, len(C), 4000)])
               ** 2).sum(-1)).mean()
bar = A.barh(range(len(boy)), boy[sira], height=.72,
             color=["#c0392b" if b < .25 * olc else "#2c7fb8"
                    for b in boy[sira]])
A.set_yticks(range(len(boy)))
A.set_yticklabels([str(rel_ad[i])[:20] for i in sira], fontsize=6.5)
A.axvline(olc, ls="--", c="k", lw=1)
A.text(olc, len(boy) * .5, f"  varliklar arasi\n  ortalama {olc:.2f}",
       fontsize=7, va="center")
A.set_xlabel("‖Rel[r]‖  —  kaydirmanin BOYU")
A.set_title("2 — iliski vektorleri\nKIRMIZI: sifira cokmus, 'cevap oznenin yerinde'")

# ---------------------------------------------------------------- 3
A = ax[1, 0]
A.scatter(Z[:, 0], Z[:, 1], s=4, color="#cccccc", linewidths=0, zorder=1)
ok = lambda p, q, c: A.add_patch(FancyArrowPatch(
    p, q, arrowstyle="-|>", mutation_scale=13, color=c, lw=1.8,
    shrinkA=0, shrinkB=0, zorder=4))
dogru = yanlis = 0
for n, z in enumerate(comp[:3]):
    e, r1, r2, kop, ans = (int(x) for x in z[:5])
    q1 = C[e] + Rel[r1]
    q2 = q1 + Rel[r2]
    en = int(((q2 - C) ** 2).sum(-1).argmin())
    dogru += en == ans; yanlis += en != ans
    c = plt.cm.Dark2(n)
    p0, p1, p2 = pr(C[e])[0], pr(q1)[0], pr(q2)[0]
    ok(p0, p1, c); ok(p1, p2, c)
    A.scatter(*p0, s=90, marker="o", color=c, zorder=5,
              edgecolor="k", linewidth=.7)
    A.scatter(*pr(C[kop])[0], s=110, marker="s", facecolor="none",
              edgecolor=c, linewidth=1.6, zorder=5)
    A.scatter(*pr(C[ans])[0], s=150, marker="*", color=c, zorder=6,
              edgecolor="k", linewidth=.6)
    A.annotate(str(ad[e]).replace("_", " "), p0, fontsize=7,
               xytext=(4, 5), textcoords="offset points", color=c)
    A.annotate(str(ad[ans]).replace("_", " "), pr(C[ans])[0], fontsize=7,
               xytext=(4, 5), textcoords="offset points", color=c)
A.set_title("3 — yuruyus:  ozne ●  →+r1→  ara  →+r2→  hedef\n"
            "kare ▫ = GERCEK kopru,  yildiz ★ = GERCEK cevap")
A.set_xlabel("PC1"); A.set_ylabel("PC2")

# ---------------------------------------------------------------- 4
A = ax[1, 1]
fan = []
for r in range(facts.shape[1]):
    h = facts[:, r]
    g = {}
    for e in np.where(h >= 0)[0]:
        g.setdefault(int(h[e]), []).append(e)
    cok = [v_ for v_ in g.values() if len(v_) > 1]
    if cok:
        fan.append((max(len(v_) for v_ in cok), r, cok))
fan.sort(reverse=True)
if fan:
    n_fan, r, cok = fan[0]
    grup = max(cok, key=len)
    A.scatter(Z[:, 0], Z[:, 1], s=4, color="#dddddd", linewidths=0)
    A.scatter(Z[grup, 0], Z[grup, 1], s=55, color="#c0392b", zorder=3,
              edgecolor="k", linewidth=.5,
              label=f"'{rel_ad[r]}' ile AYNI hedefe bakan {len(grup)} varlik")
    hed = int(facts[grup[0], r])
    A.scatter(*pr(C[hed])[0], s=210, marker="X", color="#16a085", zorder=4,
              edgecolor="k", linewidth=.7, label=f"ortak hedef: {ad[hed]}")
    ic = np.sqrt(((C[grup][:, None] - C[grup][None]) ** 2).sum(-1))
    ic = ic[np.triu_indices(len(grup), 1)].mean()
    A.legend(fontsize=7.5, loc="best")
    A.set_title(f"4 — CAKISMA:  c_e + Rel[r] = c_hedef  butun e icin\n"
                f"grup-ici {ic:.2f}  vs  genel {olc:.2f}   "
                f"→  {olc/ic:.1f} kat SIKISMA")
A.set_xlabel("PC1"); A.set_ylabel("PC2")

fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(CIK, dpi=135)
print(f"{CIK}   PC1+PC2 varyans %{100*acik.sum():.1f}   "
      f"sifira cokmus iliski {int((boy < .25*olc).sum())}/{len(boy)}")
