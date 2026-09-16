# -*- coding: utf-8 -*-
"""model_00 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Iki sey tutulur:

  1. MIMARIDE projenin hicbir ozelligi YOK: dongu yok, Phi darbogazi
     yok, maske yok, yardimci kayip yok, model_a.Model'den miras yok.
  2. VERIDE model_b15 ile BIREBIR AYNI. Ayni veriyi gormezse kol
     hicbir sey olcmez; fark YALNIZ mimari olmali.

Ayrica taban `model_a` kilidi de kosulur (kos.py'nin 3. hucresi bu
klasordeki ILK test_*.py'yi calistiriyor).

AYRINTI=1 -> gecen kontroller de basilir.
"""
import os
import subprocess
import sys

_B = os.path.dirname(os.path.abspath(__file__))
_K = os.path.dirname(_B)
for _p in (os.path.join(_K, "model_a"), os.path.join(_K, "model_b"), _B, _K):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
import model_a as M                                          # noqa: E402

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


# --- 0) TABAN model_a kilidi --------------------------------------------
print("=== 0) TABAN (model_a/test_sabit.py) ===")
_r = subprocess.run([sys.executable, os.path.join(_K, "model_a", "test_sabit.py")],
                    cwd=os.path.join(_K, "model_a"), capture_output=True,
                    text=True, env={**os.environ})
print(_r.stdout.rstrip()[-400:] or "(cikti YOK)")
ok(_r.returncode == 0, "model_a kilidi", f"cikis {_r.returncode}")

print("\n=== 1) model_00 MIMARISI ===")
import model_00 as S                                         # noqa: E402
from model_b15 import AYAR as B15                            # noqa: E402

A = S.AYAR
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64,
   "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, model_a.Model'den MIRAS ALMIYOR")
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
# NEDENSELLIK: sondaki jetonu degistirmek ONCEKI pozisyonlari
# DEGISTIRMEMELI. Causal maske gercekten calisiyor mu.
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
# yani test RoPE'u degil ortalamayi olcuyordu.
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

# --- 3) VERI model_b15 ILE AYNI MI --------------------------------------
print("\n=== 3) VERI model_b15 ILE BIREBIR AYNI MI ===")
# VERI_ALAN listesi `ayar_00.py`den okunur -- TEK KAYNAK. Burada
# kopyalansaydi iki liste ayri ayri degisir ve kilit sessizce EKSIK
# denetler hale gelirdi.
from ayar_00 import VERI_ALAN                                # noqa: E402
for f in VERI_ALAN:
    ok(getattr(A, f) == getattr(B15, f), f"veri alani {f} model_b15 ile AYNI",
       f"{getattr(A, f)!r} vs {getattr(B15, f)!r}")

# --- AD DEGIL, ICERIK -----------------------------------------------
# Kullanici karari, 16 Eylul: "verisi de veri_00 olsun". Iki kol artik
# FARKLI ad kullaniyor (veri_00 / veri_okul4), o yuzden `veri_ad` esitligi
# ARANMAZ -- yerine GRAFIN KENDISI karsilastirilir. Asagidaki havuz ve
# olcme izi denetimleri bunu ayrica bit duzeyinde tekrarliyor.
ok("veri_ad" not in VERI_ALAN, "veri_ad VERI_ALAN'da YOK (ad degil icerik)")
ok(A.veri_ad == "veri_00", "model_00 KENDI veri modulunu okur", A.veri_ad)
ok(B15.veri_ad == "veri_okul4", "model_b15 DEGISMEDI", B15.veri_ad)
import veri_00 as V00                                        # noqa: E402
import veri_okul4 as V4                                      # noqa: E402
_G0, _G4 = V00.kur(A.veri_tohum), V4.kur(B15.veri_tohum)
ok(V00.graf_izi(_G0) == V00.graf_izi(_G4),
   "GRAF model_b15'inkiyle AYNI (parmak izi)", V00.graf_izi(_G0))
ok(V00.graf_izi(_G0) == V00.IZ,
   "graf izi veri_00.IZ ile TUTUYOR -- veri sessizce kaymadi", V00.IZ)
ok(len(V00.zincirler(_G0)) == len(V4.zincirler(_G4)),
   "zincir sayisi AYNI", str(len(V00.zincirler(_G0))))
ok(list(V00.ILISKI) == list(V4.ILISKI) and V00.TIPLER == V4.TIPLER,
   "ILISKI ve TIPLER AYNI", f"|R|={len(V00.ILISKI)}")
for _g in ("kur", "zincirler", "TIPLER", "ILISKI"):
    ok(hasattr(V00, _g), f"veri_00.{_g} var (model_a.veri_kur sozlesmesi)")
for _g in ("SEMA", "GEREKTIRIR", "BLOK", "yaz", "turetilebilir"):
    ok(hasattr(V00, _g), f"veri_00.{_g} var (veri_dok/graf_dok kullaniyor)")

_fark = sorted(set(B15.fark(A)) | {"ad"})
ok(_fark == ["ad", "betas", "dar_alfa", "dar_kapi", "dff", "dongu",
             "l", "ort_bas", "veri_ad"],
   f"model_00 <-> model_b15 farki {_fark}",
   "MIMARI + STANDART TARIF (betas, ort_bas) + veri ADI (icerik AYNI)")
v15 = M.veri_kur(B15, yaz=lambda *a: None)
ok(M.olcme_izi(M.olcme_listeleri(A, v))
   == M.olcme_izi(M.olcme_listeleri(B15, v15)),
   "olcme izi model_b15 ile AYNI -> AYNI SINAV, kiyas GECERLI")
X0, _, _, _, _, _ = M.egitim_havuzu(A, v, yaz=lambda *a: None)
X1, _, _, _, _, _ = M.egitim_havuzu(B15, v15, yaz=lambda *a: None)
ok(X0.shape == X1.shape and (X0 == X1).all(),
   "EGITIM HAVUZU model_b15 ile BIREBIR AYNI", f"{X0.shape} vs {X1.shape}")

# --- 3b) STANDART TARIF -- DEVRALINMAYANLAR -----------------------------
print("\n=== 3b) STANDART TARIF -- devralinMAYANLAR ===")
# `ort_bas` en onemlisi: paylasilan ayar 10.000. adimdan sonra LOOKAHEAD
# (yavas agirlik) ortalamasi yapiyor -- kosu logunda "ORTALAMA ACILDI".
# Bu bir optimizer SARMALAYICISI ve nanoGPT/Llama/Pythia/GPT-2 tarifinde
# YOK. Standart modelin ne yaptigini olcecek bir kol, standart olmayan
# bir numarayla kosamaz.
ok(A.ort_bas == 0, "LOOKAHEAD ORTALAMASI KAPALI (ort_bas=0)",
   f"b15 {B15.ort_bas} -> 00 {A.ort_bas}")
ok(B15.ort_bas == 10000, "model_b15'te ACIK (DEGISMEDI)", str(B15.ort_bas))
ok(A.betas == (0.9, 0.95),
   "betas (0.9, 0.95) -- nanoGPT/GPT-3/Llama/Pythia", str(A.betas))
ok(B15.betas == (0.9, 0.999),
   "model_b15'te PyTorch varsayilani (DEGISMEDI)", str(B15.betas))
# Devralinanlar ZATEN standart olmali:
ok(A.wd == 0.1 and A.sabit_lr is False,
   "wd 0.1 + cosine DEVRALINDI (zaten standart, CLAUDE.md kural 4)")
ok(A.isinma == 2000,
   "isinma 2000 DEVRALINDI -- nanoGPT/Llama MUTLAK degeriyle AYNI")
ok(A.lr == 1e-3, "lr 1e-3 DEVRALINDI -- Pythia-70m ile ayni mertebe")
ok(A.mask_poz is None and not A.mask_blok and A.kopru_kayip == 0,
   "projeye ozgu DIGER numaralarin hepsi KAPALI")

# --- 4) PARAMETRE --------------------------------------------------------
print("\n=== 4) PARAMETRE ===")
_bek = 8 * (4 * A.d * A.d + 3 * A.d * A.dff)
ok(abs(net.n_param() - (_bek + v.vocab * A.d)) < 5000,
   "parametre ~ 8*(4d^2 + 3*d*dff) + vocab*d",
   f"{net.n_param():,} (blok {_bek:,} + gomme {v.vocab*A.d:,})")
ok(net.n_param() > 6_000_000, "6M+ parametre (model_b15 3,23M)",
   f"{net.n_param():,}")

# --- 5) MODUL SOZLESMESI -------------------------------------------------
print("\n=== 5) SOZLESME ===")
for g in ("AYAR", "egit", "fark_bas", "ModelSade"):
    ok(hasattr(S, g), f"model_00.{g} var", "kos.py duser")
for ad in ("pencere_00", "tani_00", "sor_00"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import",
       _r2.stderr.strip()[-200:])

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
