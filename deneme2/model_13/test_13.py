# -*- coding: utf-8 -*-
"""test_13 -- model_13'un kapilari. `python test_13.py`

En onemlisi §6: geri yayilim ELLE yazildi. Yanlis yazilirsa kod
SESSIZCE calisir, kayip yine duser, ama ogrenilen sey Eq.'in kendisi
olmaz. Tek korunma sayisal turevle kiyas.
"""
import ast, glob, hashlib, io, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taban_13 as MT, ayar_13 as AY, veri_13 as V13
from model_13 import Zincir, sm, kisayol_hedefi

GECTI = []


def kapi(ad, kosul, not_=""):
    GECTI.append(bool(kosul))
    print(f"  {'GECTI' if kosul else '!! KALDI':<10}{ad:<46}{not_}")


# --- §0  KOL YALITIMI ------------------------------------------------
print("\n§0  KOL YALITIMI -- disariya import YOK")
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))):
    d = []
    for x in ast.walk(ast.parse(io.open(f, encoding="utf-8").read())):
        if isinstance(x, ast.ImportFrom) and x.module:
            d.append(x.module)
        elif isinstance(x, ast.Import):
            d += [al.name for al in x.names]
    kotu = [n for n in d if n.rsplit("_", 1)[-1].isdigit()
            and not n.endswith("_13")]
    kapi(os.path.basename(f), not kotu, str(kotu) if kotu else "")

# --- §1  IZLER -- kopyanin ASIL kapisi -------------------------------
print("\n§1  IZLER -- kopya kayarsa BURADA duser")
A = AY.AYAR
v = MT.veri_kur(A, yaz=lambda *a, **k: None)
L = MT.olcme_listeleri(A, v)
kapi("graf izi 3cd9a2575e47", V13.IZ == "3cd9a2575e47", V13.IZ)
kapi("olcme izi 44e6262e37f3", MT.olcme_izi(L) == "44e6262e37f3",
     MT.olcme_izi(L))

# --- §2  KOPYA BUTUNLUGU ---------------------------------------------
print("\n§2  KOPYA BUTUNLUGU")
VERI_SHA = "116f41bad2f7"
h = hashlib.sha256(io.open(
    os.path.join(os.path.dirname(__file__), "veri_13.py"),
    encoding="utf-8").read().encode()).hexdigest()[:12]
kapi("veri_13.py, veri_11.py'nin AYNISI", h == VERI_SHA, h)

# --- §3  VERI SEKLI ---------------------------------------------------
print("\n§3  VERI SEKLI")
kapi("1608 varlik / 24 iliski", (v.n_ent, v.n_rel) == (1608, 24))
kapi("7381 tek-adim olgu", len(v.one) == 7381, len(v.one))
kapi("par (1608, 3), 327 PAYLASILAN parca",
     v.par.shape == (1608, 3) and len(v.par_ad[0]) == 327)
kapi("facts ISLEVSEL -- (e,r) basina TEK cevap", v.facts.ndim == 2)

# --- §4  BIRIM TESTININ ZEMINI ---------------------------------------
# `ent_yok`ta kisayol TIP OLARAK imkansiz olmali: facts[e,r2] == -1.
# Sifir cikmasi YETENEK degil; bu satir SIFIRIN SEBEBINI siniyor.
print("\n§4  ent_yok TANIMI -- BIRIM TESTININ zemini")
ky = kisayol_hedefi(v, L["ent_yok"])
kapi("ent_yok'ta kisayol UYGUN ornek YOK", int((ky >= 0).sum()) == 0,
     f"uygun {int((ky >= 0).sum())}")
kc = kisayol_hedefi(v, L["comp"])
kapi("comp'ta kisayol UYGUN ornek VAR (bosluga karsi)",
     int((kc >= 0).sum()) > 0, f"uygun {int((kc >= 0).sum())}")

# --- §5  PARCA KOORDINATI --------------------------------------------
print("\n§5  PARCA KOORDINATI -- saklanmiyor, HESAPLANIYOR")
m = Zincir(v, D=3, kip="parca", tohum=0)
C = m.coord()
e = 7
bek = np.concatenate([m.Parca[i] if i >= 0 else np.zeros(3)
                      for i in v.par[e]])
kapi("C[e] == concat(Parca[par[e, yuva]])",
     np.allclose(C[e], bek, atol=0), f"fark {np.abs(C[e]-bek).max():.1e}")
ms = Zincir(v, D=3, kip="serbest", tohum=0)
kapi("parca parametresi serbestten AZ",
     m.n_params() < ms.n_params(), f"{m.n_params():,} < {ms.n_params():,}")

# --- §6  ELLE GRADYAN vs SAYISAL TUREV --------------------------------
print("\n§6  ELLE GRADYAN -- sonlu farkla kiyas (ASIL kapi)")


def kayip(mdl, E, R, Y, lam):
    lg, (C, q, *_) = mdl.fwd(E, R)
    P = sm(lg)
    ce = -np.log(P[np.arange(len(Y)), Y] + 1e-12).mean()
    kap = lam * ((q - C[Y]) ** 2).sum(-1).mean()
    return ce + kap


for kip in ("serbest", "parca"):
    rng = np.random.default_rng(3)
    mdl = Zincir(v, D=3, kip=kip, tohum=1)
    ix = rng.integers(0, len(v.one), 32)
    O = np.array(v.one, np.int64)[ix]
    E, R, Y = O[:, 0], O[:, 1:2], O[:, 2]
    lam = 0.5
    lg, c = mdl.fwd(E, R)
    P = sm(lg)
    g = P.copy()
    g[np.arange(len(Y)), Y] -= 1
    g /= len(Y)
    an = mdl.grad(g, c, Y, lam)
    en_kotu, nerede = 0.0, ""
    for pi, (p, ga) in enumerate(zip(mdl.ps, an)):
        yer = [tuple(rng.integers(0, s) for s in p.shape) for _ in range(12)]
        for j in yer:
            eps = 1e-5
            p[j] += eps; a2 = kayip(mdl, E, R, Y, lam)
            p[j] -= 2 * eps; a1 = kayip(mdl, E, R, Y, lam)
            p[j] += eps
            say = (a2 - a1) / (2 * eps)
            d = abs(say - ga[j]) / max(1e-8, abs(say) + abs(ga[j]))
            if d > en_kotu:
                en_kotu, nerede = d, f"p{pi}{j}"
    kapi(f"{kip:<8} bagil hata < 1e-4", en_kotu < 1e-4,
         f"en kotu {en_kotu:.2e} @ {nerede}")


# --- §7  MODELE ILISKI VERILMIYOR ------------------------------------
# Kullanici sordu, 20 Eylul: "biz modele bu iliskidir diye dogrudan
# vermiyoruz degil mi". Cevap CUMLE degil KAPI olsun.
print("")
print("§7  ILISKI SIZINTISI -- modele 'bu iliskidir' denmiyor")
import inspect
import kelime_13 as KEL
import dil_13 as D13

# 7a  kelime katmani grafa HIC bakmiyor
_ks = io.open(os.path.join(os.path.dirname(__file__), "kelime_13.py"),
              encoding="utf-8").read()
_kt = ast.parse(_ks)
_im = [x.module for x in ast.walk(_kt)
       if isinstance(x, ast.ImportFrom) and x.module]
_im += [al.name for x in ast.walk(_kt)
        if isinstance(x, ast.Import) for al in x.names]
kapi("kelime_13 veri/taban modulu IMPORT ETMIYOR",
     not [m for m in _im if m.startswith(("veri_", "taban_"))], str(_im))
kapi("kelime_13'te facts / ILISKI / Rel gecmiyor",
     not any(w in _ks for w in ("facts", "ILISKI", "Rel[")))

# 7b  iliski kelimesi KOK tablosunda isaretsiz bir satir
_sz = KEL.Sozluk(["Cem Yildiz'in annesi Irem Yildiz'dir.",
                  "Ali Kaya'nin ogrencisi Veli Demir'dir."])
_kok_of = {_sz.kelime[i]: _sz.kok_ad[_sz.wp[i, 0]] for i in range(_sz.V)}
_il = [w for w in ("annesi", "ogrencisi") if w in _kok_of]
kapi("iliski kelimesi KOK tablosunda SADE satir",
     len(_il) == 2 and all(_kok_of[w] == w for w in _il), str(_il))
kapi("iliski ile varlik parcasi AYNI tabloda, tip isareti YOK",
     "annesi" in _sz.kok_ad and "Cem" in _sz.kok_ad,
     f"kok {len(_sz.kok_ad)} satir")

# 7c  forward YALNIZ kelime id'si aliyor
_sig = list(inspect.signature(D13.ParcaDil.forward).parameters)
kapi("ParcaDil.forward(self, X) -- baska girdi YOK",
     _sig == ["self", "X"], str(_sig))

print(f"\n{'='*64}\n{sum(GECTI)}/{len(GECTI)} kapi GECTI")
sys.exit(0 if all(GECTI) else 1)
