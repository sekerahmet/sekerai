# -*- coding: utf-8 -*-
"""model_00 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_00 folderi
altinda olmali. model_00 diger hicbir model ile ayni seyi
kullanmamali."*

Dort sey tutulur:

  1. BAGIMSIZLIK  model_00/*.py icinde `model_a` / `model_b` /
     `veri_okul*` / `pencere_a` import'u YOK. Statik taramayla.
  2. MIMARI       projenin hicbir ozelligi yok: dongu yok, Phi
     darbogazi yok, maske yok, yardimci kayip yok, miras yok.
  3. VERI         `veri_00` grafi `veri_okul4`unkiyle BIREBIR AYNI.
  4. MOTOR        `taban_00` ile `model_a` AYNI SEYI olcuyor:
     egitim havuzu bit duzeyinde ayni, olcme izi ayni.

--------------------------------------------------------------------------
!! KILIT, PAYLASILAN MODULLERI IMPORT EDER -- VE ETMELIDIR

3 ve 4, model_00'in kopyalarini ORIJINALLERIYLE karsilastiriyor; bunun
icin `model_a` / `veri_okul4` / `model_b15` buraya yukleniyor. Bu,
1'deki bagimsizlik iddiasini BOZMAZ: KOSAN kol onlari gormez, yalniz
KILIT gorur. Kilidin isi zaten bu -- kopyanin sapip sapmadigini
soylemek.

Kopyanin GERCEK riski buydu: paylasilan motorda bir olcum hatasi
duzeltilirse buraya kendiliginden GELMEZ, ve model_00 ile model_b15 o
gunden sonra FARKLI KODLA olculmus olur. 4. bolum bunu sessiz olmaktan
cikariyor: sapma varsa kilit DUSER ve o gun bilerek karar verilir
(kopyayi guncelle, ya da farki onkayda yaz).

AYRINTI=1 -> gecen kontroller de basilir.
"""
import os
import subprocess
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_K = os.path.dirname(_B)
if _B not in sys.path:
    sys.path.insert(0, _B)

import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402

AYRINTI = os.environ.get("AYRINTI") == "1"
_gecti = _bozuk = 0


def ok(kosul, ad, ek=""):
    global _gecti, _bozuk
    if kosul:
        _gecti += 1
        if AYRINTI:
            print(f"    ok   {ad}" + (f"  {ek}" if ek else ""))
    else:
        _bozuk += 1
        print(f"    !!   BOZUK: {ad}" + (f"  {ek}" if ek else ""))


# --- 0) BAGIMSIZLIK ------------------------------------------------------
print("=== 0) BAGIMSIZLIK: model_00/ disariya BAGLI MI ===")
# Metinde arama YAPMIYORUZ -- ilk denemem oyleydi ve `ayar_00.py`nin
# DOCSTRING'inde gecen "from model_b15 import ..." cumlesini (kaldirilan
# seyi ANLATAN cumleyi) import sandi. Dogrusu AST: yalnizca gercek
# import dugumlerine bakilir.
import ast                                                   # noqa: E402

YASAK = ("model_a", "model_b", "pencere_a", "tani_a", "asama1", "sor",
         "kos", "veri_dok", "graf_dok", "havuz_dok")
YASAK_ON = ("veri_okul", "model_a", "model_b")
_bu = os.path.basename(__file__)


def _disa_bagli(yol):
    """Dosyanin import ETTIGI paylasilan modulleri dondur."""
    agac = ast.parse(open(yol, encoding="utf-8").read())
    adlar = []
    for d in ast.walk(agac):
        if isinstance(d, ast.Import):
            adlar += [a.name for a in d.names]
        elif isinstance(d, ast.ImportFrom) and d.module:
            adlar.append(d.module)
        elif (isinstance(d, ast.Call)
              and getattr(d.func, "attr", "") == "import_module"
              and d.args and isinstance(d.args[0], ast.Constant)):
            adlar.append(d.args[0].value)      # importlib ile gizlenmesin
    return [a for a in adlar
            if a in YASAK or any(a.startswith(o) for o in YASAK_ON)]


for _f in sorted(x for x in os.listdir(_B) if x.endswith(".py")):
    if _f == _bu:                     # kilidin KENDISI muaf (bkz. baslik)
        continue
    _kotu = _disa_bagli(os.path.join(_B, _f))
    ok(not _kotu, f"{_f} paylasilan modul import ETMIYOR", str(_kotu))

print("\n=== 1) model_00 MIMARISI ===")
import taban_00 as M                                         # noqa: E402
import model_00 as S                                         # noqa: E402

A = S.AYAR
ok(S.M is M, "model_00 motoru taban_00", S.M.__name__)
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64, "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, taban_00.Model'den MIRAS ALMIYOR")
ok(issubclass(S.ModelSade, nn.Module), "ModelSade bir nn.Module")

v = M.veri_kur(A, yaz=lambda *a: None)
net = S.ModelSade(A, v.vocab)
ok(len(net.bloklar) == 8, "8 AYRI blok kuruldu", str(len(net.bloklar)))
_ag = {id(b.qkv.weight) for b in net.bloklar}
ok(len(_ag) == 8, "8 blok AYRI agirlik (paylasim YOK)", str(len(_ag)))
ok(net.head.weight is net.emb.weight, "BAGLI GOMME (head.weight is emb.weight)")
ok(not any(getattr(m, "bias", None) is not None for m in net.modules()),
   "hicbir katmanda BIAS YOK")
ok(not hasattr(net, "pos"), "OGRENILMIS pozisyon gommesi YOK (RoPE var)")
ok(hasattr(net, "rope_cos") and hasattr(net, "rope_sin"), "RoPE tablosu var")
ok(net.rope_cos.shape == (A.t_len, A.d // A.nh // 2),
   "RoPE tablosu (t_len, head_dim/2)", str(tuple(net.rope_cos.shape)))
ok(not any(p is net.rope_cos or p is net.rope_sin for p in net.parameters()),
   "RoPE OGRENILMIYOR (buffer, parametre DEGIL)")
_b0 = net.bloklar[0]
ok(all(hasattr(_b0, x) for x in ("w1", "w2", "w3")),
   "SwiGLU UC matris (w1, w2, w3)")
ok(_b0.w1.out_features == A.dff and _b0.w2.in_features == A.dff,
   "SwiGLU gizli boyutu d_ff")

# --- 2) ILERI GECIS ------------------------------------------------------
print("\n=== 2) ILERI GECIS ===")
x = torch.randint(0, v.vocab, (4, A.t_len))
with torch.no_grad():
    lg = net(x)
ok(tuple(lg.shape) == (4, A.t_len, v.vocab), "cikti sekli",
   f"{tuple(lg.shape)} beklenen {(4, A.t_len, v.vocab)}")
ok(torch.isfinite(lg).all(), "cikti SONLU (NaN/Inf yok)")
x2 = x.clone()
x2[:, -1] = (x2[:, -1] + 1) % v.vocab
with torch.no_grad():
    lg2 = net(x2)
ok(torch.allclose(lg[:, :-1], lg2[:, :-1], atol=0),
   "CAUSAL: son jetonu degistirmek onceki pozisyonlari BOZMUYOR")
# RoPE gercekten POZISYONA duyarli mi -- DOGRU TEST.
#
# ILK denemem yanlisti: dizi TEK bir jetonla dolduruluyordu ve
# "iki pozisyonun ciktisi ayni" cikti. O bir kusur DEGIL: butun `v`
# vektorleri aynidir, herhangi bir konveks birlesimi de ayni cikar --
# yani test RoPE'u degil ORTALAMAYI olcuyordu.
#
# Dogrusu: PERMUTASYON. Pozisyon bilgisi YOKSA dikkat onekin sirasina
# duyarsizdir; ilk iki jetonu takas etmek SON pozisyonun ciktisini
# DEGISTIRMEZ. RoPE varsa DEGISTIRIR.
y1 = torch.randint(0, v.vocab, (1, A.t_len))
y2 = y1.clone()
y2[0, 0], y2[0, 1] = y1[0, 1].clone(), y1[0, 0].clone()
with torch.no_grad():
    l1, l2 = net(y1), net(y2)
ok(not torch.allclose(l1[0, -1], l2[0, -1], atol=1e-5),
   "RoPE POZISYONA duyarli (onekteki takas SON ciktiyi degistiriyor)",
   f"en buyuk fark {(l1[0,-1]-l2[0,-1]).abs().max():.3e}")

# --- 3) VERI: kopya ORIJINALLE ayni mi ----------------------------------
print("\n=== 3) VERI: veri_00 == veri_okul4 (ICERIK) ===")
sys.path.insert(0, _K)                       # YALNIZ KILIT icin
sys.path.insert(0, os.path.join(_K, "model_a"))
sys.path.insert(0, os.path.join(_K, "model_b"))
import veri_00 as V00                                        # noqa: E402
import veri_okul4 as V4                                      # noqa: E402

ok(A.veri_ad == "veri_00", "model_00 KENDI veri modulunu okur", A.veri_ad)
_G0, _G4 = V00.kur(0), V4.kur(0)
ok(sorted(_G0) == sorted(_G4), "graf anahtarlari AYNI")
for _k in sorted(_G0):
    ok(_G0[_k] == _G4[_k], f"graf[{_k}] BIREBIR AYNI")
ok(V00.graf_izi(_G0) == V00.IZ, "graf izi veri_00.IZ ile TUTUYOR", V00.IZ)
ok(V00.zincirler(_G0) == V4.zincirler(_G4), "ZINCIRLER birebir ayni",
   str(len(V00.zincirler(_G0))))
ok(list(V00.ILISKI) == list(V4.ILISKI) and V00.TIPLER == V4.TIPLER,
   "ILISKI + TIPLER AYNI", f"|R|={len(V00.ILISKI)}")
ok(V00.SEMA == V4.SEMA and V00.GEREKTIRIR == V4.GEREKTIRIR,
   "SEMA + GEREKTIRIR AYNI")
ok(V00.turetilebilir(_G0) == V4.turetilebilir(_G4), "TURETILEBILIR ciftler AYNI")
for _g in ("kur", "zincirler", "TIPLER", "ILISKI", "SEMA", "GEREKTIRIR",
           "BLOK", "yaz", "turetilebilir"):
    ok(hasattr(V00, _g), f"veri_00.{_g} var (sozlesme)")

# --- 4) MOTOR: taban_00 ile model_a AYNI SEYI mi olcuyor ----------------
print("\n=== 4) MOTOR ESDEGERLIGI: taban_00 vs model_a ===")
import model_a as MA                                         # noqa: E402
from model_b15 import AYAR as B15                            # noqa: E402
from ayar_00 import GOREV_ALAN                               # noqa: E402

ok(M.ESKI_VARSAYILAN == MA.ESKI_VARSAYILAN, "ESKI_VARSAYILAN AYNI")
ok([f.name for f in M.dc.fields(M.Ayar)]
   == [f.name for f in MA.dc.fields(MA.Ayar)], "Ayar ALANLARI ayni")
ok(M.T_LEN == MA.T_LEN and M.SPECIAL == MA.SPECIAL, "sabitler AYNI")

for f in GOREV_ALAN:
    ok(getattr(A, f) == getattr(B15, f), f"gorev alani {f} model_b15 ile AYNI",
       f"{getattr(A, f)!r} vs {getattr(B15, f)!r}")
ok("veri_ad" not in GOREV_ALAN, "veri_ad GOREV_ALAN'da YOK (ad degil icerik)")
_fark = sorted(set(B15.fark(A)) | {"ad"})
ok(_fark == ["ad", "betas", "dar_alfa", "dar_kapi", "dff", "dongu",
             "l", "ort_bas", "veri_ad"],
   f"model_00 <-> model_b15 farki {_fark}",
   "MIMARI + STANDART TARIF (betas, ort_bas) + veri ADI (icerik AYNI)")

v15 = MA.veri_kur(B15, yaz=lambda *a: None)
ok(M.olcme_izi(M.olcme_listeleri(A, v))
   == MA.olcme_izi(MA.olcme_listeleri(B15, v15)),
   "OLCME IZI ayni -> AYNI SINAV, sayilar ayni tabloda okunur",
   M.olcme_izi(M.olcme_listeleri(A, v)))
X0, P0, T0, _, _, _ = M.egitim_havuzu(A, v, yaz=lambda *a: None)
X1, P1, T1, _, _, _ = MA.egitim_havuzu(B15, v15, yaz=lambda *a: None)
ok(X0.shape == X1.shape and (X0 == X1).all(),
   "EGITIM HAVUZU BIT DUZEYINDE ayni", f"{X0.shape} vs {X1.shape}")
ok((P0 == P1).all() and (T0 == T1).all(), "havuz P ve T de ayni")

# --- 5) STANDART TARIF -- DEVRALINMAYANLAR ------------------------------
print("\n=== 5) STANDART TARIF ===")
ok(A.ort_bas == 0, "LOOKAHEAD ORTALAMASI KAPALI (ort_bas=0)",
   f"b15 {B15.ort_bas} -> 00 {A.ort_bas}")
ok(B15.ort_bas == 10000, "model_b15'te ACIK (DEGISMEDI)", str(B15.ort_bas))
ok(A.betas == (0.9, 0.95),
   "betas (0.9, 0.95) -- nanoGPT/GPT-3/Llama/Pythia", str(A.betas))
ok(B15.betas == (0.9, 0.999),
   "model_b15'te PyTorch varsayilani (DEGISMEDI)", str(B15.betas))
ok(A.wd == 0.1 and A.sabit_lr is False,
   "wd 0.1 + cosine (zaten standart, CLAUDE.md kural 4)")
ok(A.isinma == 2000, "isinma 2000 -- nanoGPT/Llama MUTLAK degeriyle AYNI")
ok(A.lr == 1e-3, "lr 1e-3 -- Pythia-70m ile ayni mertebe")

# --- 6) PARAMETRE --------------------------------------------------------
print("\n=== 6) PARAMETRE ===")
_bek = 8 * (4 * A.d * A.d + 3 * A.d * A.dff)
ok(abs(net.n_param() - (_bek + v.vocab * A.d)) < 5000,
   "parametre ~ 8*(4d^2 + 3*d*dff) + vocab*d",
   f"{net.n_param():,} (blok {_bek:,} + gomme {v.vocab*A.d:,})")
ok(net.n_param() > 6_000_000, "6M+ parametre (model_b15 3,23M)",
   f"{net.n_param():,}")

# --- 7) MODUL SOZLESMESI -------------------------------------------------
print("\n=== 7) SOZLESME ===")
for g in ("AYAR", "egit", "fark_bas", "ModelSade", "TABAN"):
    ok(hasattr(S, g), f"model_00.{g} var", "kos_00.py duser")
for ad in ("veri_00", "taban_00", "ayar_00", "pencere_00", "tani_00",
           "asama1_00", "sor_00", "kos_00"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True, cwd=_B)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import (yalniz model_00/)",
       _r2.stderr.strip()[-200:])

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
