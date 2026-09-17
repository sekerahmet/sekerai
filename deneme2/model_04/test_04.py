# -*- coding: utf-8 -*-
"""model_04 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_04 folderi
altinda olmali. model_04 diger hicbir model ile ayni seyi
kullanmamali."*

Dort sey tutulur:

  1. BAGIMSIZLIK  model_04/*.py icinde `model_a` / `model_b` /
     `veri_okul*` / `pencere_a` import'u YOK. Statik taramayla.
  2. MIMARI       projenin hicbir ozelligi yok: dongu yok, Phi
     darbogazi yok, maske yok, yardimci kayip yok, miras yok.
  3. VERI         `veri_04` grafi `veri_okul4`unkiyle BIREBIR AYNI.
  4. MOTOR        `taban_04` ile `model_a` AYNI SEYI olcuyor:
     egitim havuzu bit duzeyinde ayni, olcme izi ayni.

--------------------------------------------------------------------------
!! KILIT, PAYLASILAN MODULLERI IMPORT EDER -- VE ETMELIDIR

3 ve 4, model_04'in kopyalarini ORIJINALLERIYLE karsilastiriyor; bunun
icin `model_a` / `veri_okul4` / `model_b15` buraya yukleniyor. Bu,
1'deki bagimsizlik iddiasini BOZMAZ: KOSAN kol onlari gormez, yalniz
KILIT gorur. Kilidin isi zaten bu -- kopyanin sapip sapmadigini
soylemek.

Kopyanin GERCEK riski buydu: paylasilan motorda bir olcum hatasi
duzeltilirse buraya kendiliginden GELMEZ, ve model_04 ile model_b15 o
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
print("=== 0) BAGIMSIZLIK: model_04/ disariya BAGLI MI ===")
# Metinde arama YAPMIYORUZ -- ilk denemem oyleydi ve `ayar_04.py`nin
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

print("\n=== 1) model_04 MIMARISI ===")
import taban_04 as M                                         # noqa: E402
import model_04 as S                                         # noqa: E402

A = S.AYAR
ok(S.M is M, "model_04 motoru taban_04", S.M.__name__)
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64, "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, taban_04.Model'den MIRAS ALMIYOR")
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

# --- 1b) OKUMA ARACLARININ MODELDEN ISTEDIGI YUZEY ----------------------
# KUSUR (16 Eylul hakemligi): `asama1_04.gizli()` ileri gecisi ELLE
# kuruyordu -- `net1.emb + net1.pos` ve TEK argumanli `blk(h)`. Ikisi de
# `model_a.Model`e ozgu; ModelSade'de `.pos` YOK (RoPE var) ve
# `Blok.forward` UC argumanli. DOGRUSAL SONDA, yani bu kolun onkayitta
# yazili EK OKUMASI, 20.000 adimdan SONRA AttributeError ile duserdi.
# Kilit sadece import ediyordu, KOSMUYORDU -- o yuzden gormedi.
ok(hasattr(S.ModelSade, "govde"), "ModelSade.govde() var (sonda bunu cagirir)")
ok(not hasattr(net, "pos"), "ogrenilmis pozisyon gommesi YOK -- elle ileri "
                            "gecis kuran her arac DUSER")
_gx = torch.randint(0, v.vocab, (4, A.t_len))
with torch.no_grad():
    _g = net.govde(_gx)
    ok(tuple(_g.shape) == (4, A.t_len, A.d), "govde() sekli (B,T,d)",
       str(tuple(_g.shape)))
    ok(torch.allclose(net(_gx), net.head(_g), atol=0),
       "forward() == head(govde()) -- sonda modelin GERCEK hesabini goruyor")
import asama1_04 as A1                                      # noqa: E402
_lst = v.comp[:8]
# !! MODELI M.DEV'E TASI. `gizli()` girdiyi `.to(M.DEV)` ediyor; model
# CPU'da kalirsa GPU'lu makinede "index is on cuda:0, other tensors on
# cpu" ile duser. ILK YAZIMDA BU EKSIKTI ve yerelde (DEV="cpu") FARK
# EDILMEDI -- kusur Colab'da, kilidin ilk GPU kosusunda cikti.
# Gercek kullanimda `asama1_04.main` modeli zaten `.to(M.DEV)` ediyor;
# yani hata ARACIN degil, BU TESTIN hatasiydi. Sonra CPU'ya geri
# aliniyor: asagidaki bolumler CPU tensorleriyle devam ediyor.
try:
    net.to(M.DEV)
    _q = A1.gizli(net, v, _lst, 6)
    ok(_q.shape == (8, A.d), "asama1_04.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
       f"{_q.shape} cihaz {M.DEV}")
except Exception as _e:                                      # noqa: BLE001
    ok(False, "asama1_04.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
       f"{type(_e).__name__}: {_e}")
finally:
    net.to("cpu")

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
print("\n=== 3) VERI: veri_04 == veri_okul4 (ICERIK) ===")
sys.path.insert(0, _K)                       # YALNIZ KILIT icin
sys.path.insert(0, os.path.join(_K, "model_a"))
sys.path.insert(0, os.path.join(_K, "model_b"))
import veri_04 as V00                                        # noqa: E402
import veri_okul4 as V4                                      # noqa: E402

ok(A.veri_ad == "veri_04", "model_04 KENDI veri modulunu okur", A.veri_ad)
_G0, _G4 = V00.kur(0), V4.kur(0)
ok(sorted(_G0) == sorted(_G4), "graf anahtarlari AYNI")
for _k in sorted(_G0):
    ok(_G0[_k] == _G4[_k], f"graf[{_k}] BIREBIR AYNI")
ok(V00.graf_izi(_G0) == V00.IZ, "graf izi veri_04.IZ ile TUTUYOR", V00.IZ)
ok(V00.zincirler(_G0) == V4.zincirler(_G4), "ZINCIRLER birebir ayni",
   str(len(V00.zincirler(_G0))))
ok(list(V00.ILISKI) == list(V4.ILISKI) and V00.TIPLER == V4.TIPLER,
   "ILISKI + TIPLER AYNI", f"|R|={len(V00.ILISKI)}")
ok(V00.SEMA == V4.SEMA and V00.GEREKTIRIR == V4.GEREKTIRIR,
   "SEMA + GEREKTIRIR AYNI")
ok(V00.turetilebilir(_G0) == V4.turetilebilir(_G4), "TURETILEBILIR ciftler AYNI")

# --- 3b) HAKEMLIKTE BULUNAN IKI KUSUR -- GERI GELMESIN ------------------
# (1) TABAN GRAF kendi basina tutarli mi. Ilk birlestirmede TEK bir SEMA
#     vardi; `_taban_kur()` semasi olgusu OLMAYAN tip ciftleri vaat
#     ediyordu ve `zincirler` KeyError veriyordu. Sonuc grafi dogruydu,
#     ama mayin oradaydi.
_GT = V00._taban_kur(0)
ok(_GT["sema"]["kardes"] == {"KISI": "KISI"},
   "taban graf TABAN semayi tasiyor (genis semayi DEGIL)",
   str(_GT["sema"]["kardes"]))
try:
    _zt = V00.zincirler(_GT)
    ok(len(_zt) == 59140, "zincirler(_taban_kur(0)) CALISIYOR", f"{len(_zt)} zincir")
except KeyError as _e:
    ok(False, "zincirler(_taban_kur(0)) CALISIYOR", f"KeyError {_e}")
ok(V00.SEMA is not V00.SEMA_TABAN and set(V00.SEMA) == set(V00.SEMA_TABAN),
   "SEMA ve SEMA_TABAN AYRI sozluk, AYNI semboller")

# (2) IZ KOR MU. Ilk surum `ad` listelerini SIRALAYIP kariyordu ve `sema`yi
#     hic karmiyordu: varlik sirasini ters cevirmek (butun jeton id'leri
#     kaydirir) ve SEMA'ya tip cifti eklemek (sinavi degistirir) IZ'i
#     KIPIRDATMIYORDU. Uc kor nokta da burada sinaniyor.
import copy as _cp                                           # noqa: E402
_iz0 = V00.graf_izi(_G0)
for _ad, _boz in (
        ("varlik SIRASI", lambda g: g["ad"].__setitem__(
            "KISI", list(reversed(g["ad"]["KISI"])))),
        ("SEMA tip cifti", lambda g: g["sema"]["onkosul"].__setitem__(
            "KISI", "DERS")),
        ("sozluk SIRASI", lambda g: g.__setitem__(
            "sozluk", list(reversed(g["sozluk"]))))):
    _h = _cp.deepcopy(_G0)
    _boz(_h)
    ok(V00.graf_izi(_h) != _iz0, f"IZ {_ad} degisimini GORUYOR")

# (3) `jeton_ad` EK JETONLARINDA cokmuyor. veri_dok/analiz_04 `ek_kip`
#     gelen her kolda IndexError veriyordu -- model_b15'in verisi bu
#     yuzden hic dokulmemisti.
sys.path.insert(0, _B)
import analiz_04 as AZ                                       # noqa: E402
_d = AZ.Dok("model_04")
_ekler = [_d.jeton_ad(i) for i in range(_d.v.ek0, _d.v.vocab)]
ok(_ekler == ["'", "<NIN>", "<SI>", "<DIR>"],
   "analiz_04.jeton_ad EK JETONLARINI adlandiriyor", str(_ekler))
for _g in ("kur", "zincirler", "TIPLER", "ILISKI", "SEMA", "GEREKTIRIR",
           "BLOK", "yaz", "turetilebilir"):
    ok(hasattr(V00, _g), f"veri_04.{_g} var (sozlesme)")

# --- 4) MOTOR: taban_04 ile model_a AYNI SEYI mi olcuyor ----------------
print("\n=== 4) MOTOR ESDEGERLIGI: taban_04 vs model_a ===")
import model_a as MA                                         # noqa: E402
from model_b15 import AYAR as B15                            # noqa: E402
from ayar_04 import GOREV_ALAN                               # noqa: E402

# !! MOTOR ARTIK model_a ILE BIREBIR DEGIL -- ve olmamali. model_04'un
# TANIMI odul; o da Ayar'a alan ekliyor. Ayrisma SAYILARAK kabul edilir:
# asagidaki liste DISINDA tek bir alan eklenirse kopya SAPMIS demektir.
# (Onceki surum "alanlar AYNI" diyordu ve odul eklenince dustu -- kilit
#  dogru davrandi, tanim guncellendi.)
ODUL_ALANLARI = {
    "odul_ac", "odul_bolme", "odul_g", "odul_batch", "odul_sicaklik",
    "odul_denetimli",
    "odul_kl", "odul_zemin", "odul_kisayol", "odul_tip", "odul_e",
    "odul_f",
    "odul_g_aile", "odul_h",
}
_e4 = set(M.ESKI_VARSAYILAN) - set(MA.ESKI_VARSAYILAN)
ok(_e4 == ODUL_ALANLARI,
   "ESKI_VARSAYILAN yalniz ODUL alanlari kadar farkli", str(sorted(_e4)))
ok({k: M.ESKI_VARSAYILAN[k] for k in MA.ESKI_VARSAYILAN}
   == MA.ESKI_VARSAYILAN,
   "ESKI_VARSAYILAN'in ORTAK anahtarlari BIREBIR ayni")
_a4 = [f.name for f in M.dc.fields(M.Ayar)]
_aa = [f.name for f in MA.dc.fields(MA.Ayar)]
ok(set(_a4) - set(_aa) == ODUL_ALANLARI,
   "Ayar'a YALNIZ odul alanlari eklendi", str(sorted(set(_a4) - set(_aa))))
ok([x for x in _a4 if x not in ODUL_ALANLARI] == _aa,
   "geri kalan Ayar alanlari AYNI SIRADA ve AYNI")
ok(M.T_LEN == MA.T_LEN and M.SPECIAL == MA.SPECIAL, "sabitler AYNI")
# Ayar VARSAYILANLARI: yalniz BILEREK degistirilen ikisi farkli olmali.
# Ucuncusu cikarsa kopya sapmis demektir. (Varsayilanlar yeniden kurulusta
# KULLANILMIYOR -- `ayar_oku` eksik alanlari ESKI_VARSAYILAN'dan
# dolduruyor -- ama sapma yine de GORULSUN.)
import dataclasses as _dcc                                   # noqa: E402
_vf = sorted(f.name for f in _dcc.fields(M.Ayar)
             if f.name not in ODUL_ALANLARI
             and getattr(M.Ayar(), f.name) != getattr(MA.Ayar(), f.name))
ok(_vf == ["ad", "veri_ad"],
   "Ayar VARSAYILANLARI (odul disi) yalniz ad + veri_ad'da farkli", str(_vf))
# ODUL alanlarinin VARSAYILANI KAPALI olmali: `odul_ac=False` iken motor
# model_03 ile BIT DUZEYINDE ayni kosar. Acan sey `ayar_04.py`, motor DEGIL.
ok(M.Ayar().odul_ac is False,
   "Ayar VARSAYILANINDA odul KAPALI -- motor kendiliginden odul kosmaz")

for f in GOREV_ALAN:
    ok(getattr(A, f) == getattr(B15, f), f"gorev alani {f} model_b15 ile AYNI",
       f"{getattr(A, f)!r} vs {getattr(B15, f)!r}")
ok("veri_ad" not in GOREV_ALAN, "veri_ad GOREV_ALAN'da YOK (ad degil icerik)")
_fark = sorted(set(B15.fark(A)) | {"ad"})
# `lr` LISTEYE 17 Eylul'de EKLENDI: odul bir INCE AYAR asamasi ve
# 1e-3 modeli 30 adimda siliyor (olculdu). `odul_*` alanlari `fark`ta
# CIKMAZ cunku model_b15'in Ayar'inda YOKLAR -- onlari ODUL_ALANLARI
# denetimi yakaliyor (bolum 4).
ok(_fark == ["ad", "betas", "dar_alfa", "dar_kapi", "dff", "dongu",
             "l", "lr", "ort_bas", "sabit_lr", "veri_ad", "wd"],
   f"model_04 <-> model_b15 farki {_fark}",
   "MIMARI + veri ADI + RECETE (wd, sabit_lr) + ODUL ASAMASI LR'i")

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
# !! wd VE sabit_lr ARTIK STANDART DEGIL -- bu kolun RECETESI.
# CLAUDE.md kural 4 ("wd 0.1 ARTIK ARANMAZ") kullanici karariyla
# 17 Eylul'de bu kolda delindi. Kilit GEVSETILMEDI, HEDEFI DEGISTI.
ok(A.wd == 0.5, "wd 0.5 -- model_a3 recetesi", str(A.wd))
ok(A.sabit_lr is True, "LR SABIT -- cosine KAPALI, recetenin ikinci yarisi")
ok(B15.wd == 0.1 and B15.sabit_lr is False,
   "model_b15'te wd 0.1 + cosine (DEGISMEDI)",
   f"{B15.wd} / {B15.sabit_lr}")
# Recetenin IKI yarisi da a3'te BIRLIKTE vardi; biri eksik alinirsa
# a3'un kosulu KURULMUS OLMAZ.
ok(A.wd == 0.5 and A.sabit_lr is True,
   "RECETE TAM: wd 0.5 VE sabit LR -- yarisi alinmadi")
ok(A.isinma == 2000, "isinma 2000 -- nanoGPT/Llama MUTLAK degeriyle AYNI")
# !! BU KOLDA lr STANDART TARIFTEN CIKTI. 1e-3 SIFIRDAN egitimin
# degeri (Pythia-70m mertebesi) ve model_03 onu tasiyordu. Odul bir
# INCE AYAR asamasi; RLVR literaturu 1e-6..1e-5 kullaniyor ve bizde
# olculdu (30 odul adimi, model_03'un agirliklarindan):
#     1e-3 -> one 0.0067  (TAM COKME)
#     1e-5 -> one 0.9233  (hasar YOK)
ok(A.lr == 1e-5,
   "lr 1e-5 -- ODUL ASAMASI degeri, sifirdan egitimin 1e-3'u DEGIL",
   f"{A.lr}")

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
    ok(hasattr(S, g), f"model_04.{g} var", "kos_04.py duser")
for ad in ("veri_04", "taban_04", "ayar_04", "pencere_04", "tani_04",
           "asama1_04", "sor_04", "kos_04"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True, cwd=_B)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import (yalniz model_04/)",
       _r2.stderr.strip()[-200:])

# --- 8) ODUL: BU KOLUN TANIMI --------------------------------------------
# Odul YANLISSA kosu BASLAMAMALI. Merdiven sessizce ters donerse 20.000
# adim yanlis seyi ogrenir ve bunu ancak sonunda anlariz.
print("")
print("=== 8) ODUL MERDIVENI ===")
import numpy as np                                            # noqa: E402
import odul_04 as OD                                          # noqa: E402

ok(A.odul_ac is True, "odul ACIK -- model_04'un TANIMI",
   "kapaliysa bu kol model_03'tur")
ok(A.odul_bolme == "ent_arama", "odul bolmesi ent_arama",
   f"{A.odul_bolme!r} -- `ent` bir HUKUM bolmesi, uzerinde EGITILMEZ")
ok(A.odul_denetimli is False, "denetimli kayip KAPALI -- ODUL TEK OGRETMEN")
ok(A.odul_kl == 0.0, "KL KAPALI -- acilirsa 'odul mu KL mi' ayrilamaz")
ok(A.odul_e == 0.00,
   "E (menzil disi) = 0.00 -- agirlik MENZILE GIRMEYE kaydirildi",
   f"{A.odul_e} -- olculdu: E->F gecisi gruplarin %36'sinda ayrisiyor")
ok(A.odul_tip < A.odul_e,
   "YANLIS TIP basamagi E'nin ALTINDA")
ok(A.odul_kisayol < A.odul_tip,
   "KISAYOL yanlis tipin de ALTINDA -- ana ariza o")
ok(A.lr == 1e-5, "ODUL ASAMASI LR'i 1e-5",
   f"{A.lr} -- 1e-3 modeli 30 adimda siliyor (olculdu 17 Eylul)")
ok(A.wd == 0.5, "wd 0.5 DEGISMEDI -- model_03 recetesi tasiniyor")
ok(A.odul_batch < A.batch,
   "odul_batch < batch -- zincirleme ornekleme yuva basina bir gecis",
   f"odul_batch {A.odul_batch} vs batch {A.batch}")
ok(A.batch == 512, "batch DEGISMEDI -- GOREV_ALAN'da, b15 kiyasi")

# ODUL BOLMESI ile HUKUM BOLMESI AYRI VARLIKLARDAN OLMALI.
# Bu kolun gecerliligi TAMAMEN buna dayaniyor: odul, ent_arama
# varliklarini zincir-basi YAPAR. `ent` varliklari dokunulmamis kalmazsa
# `ent` artik tutulmus bir sinav DEGILDIR.
_bo = {int(z[0]) for z in v.ent_arama}
_be = {int(z[0]) for z in v.ent}
ok(not (_bo & _be), "ent_arama ile ent ZINCIR-BASI varliklari AYRIK",
   f"ortak {len(_bo & _be)} varlik -- ODUL SINAVI KIRLETIR")
ok(len(_bo) > 0 and len(_be) > 0,
   f"ent_arama {len(_bo)} / ent {len(_be)} zincir-basi varlik")

# --- MERDIVEN: SEKIZ DAL DA BEKLENEN PUANI VERMELI --------------------
_par = np.asarray(v.par)
_adl = v.par_ad[0]
_yk = [i for i, x in enumerate(_adl) if str(x) == "<YOK>"]
ok(len(_yk) == 1, "<YOK> jetonu sozlukte TAM BIR KEZ")
_oa = OD.OdulAyar(A.odul_zemin, A.odul_kisayol, A.odul_tip,
                  A.odul_e, A.odul_f, A.odul_g_aile, A.odul_h)
_pz = OD.Puanlayici(_par, _yk[0], v.facts, _oa)
_tip = np.asarray(v.tip)
_z = None
for _q in v.ent:
    _e, _r1, _r2, _kp, _cv = _q
    _k = int(v.facts[_e, _r2])
    if _k < 0 or _par[_cv][1] == _par[_e][1]:
        continue
    _s2 = np.flatnonzero(_pz.menzil.h2[_e])
    _s1 = np.flatnonzero(_pz.menzil.h1[_e])
    _ay = [x for x in _s2 if _tip[x] == _tip[_cv]
           and _par[x][1] == _par[_cv][1] and x != _cv]
    _fk = [x for x in _s2 if _tip[x] == _tip[_cv]
           and _par[x][1] != _par[_cv][1]]
    _uz = [x for x in range(len(_par)) if _tip[x] == _tip[_cv]
           and x not in set(_s2.tolist()) | set(_s1.tolist())]
    if _ay and _fk and _uz:
        _z = (_e, _k, _cv, _ay[0], _fk[0], _uz[0])
        break
ok(_z is not None, "merdiven sinavi icin uygun bir `ent` sorusu bulundu")
if _z is not None:
    _e, _k, _cv, _ayl, _fkl, _uzl = _z
    _bzk = np.array([_par[_cv][0], _yk[0], _par[_cv][1]])        # <YOK> ORTADA
    _uc = np.array([_par[_cv][0], _par[_cv][1], _par[_uzl][0]])  # 3 jeton
    _sahte = None
    for _t1 in range(len(_adl)):
        _u = np.array([_t1, _par[_cv][1], _yk[0]])
        if _t1 != _yk[0] and _pz.varlik_ara(_u[None])[0] < 0:
            _sahte = _u
            break
    ok(_sahte is not None, "KAPI3 icin OLMAYAN bir uclu uretilebildi")
    # _uc: dogru cevabin tipinden FARKLI jeton sayisi -> artik ZEMIN
    # DEGIL, TIP basamagi. Ama `_uc` uydurma bir uclu oldugu icin KAPI 3'e
    # takilir; TIP basamagini GERCEK ama yanlis tipte bir varlikla sinar.
    _yt = None
    for _x in range(len(_par)):
        if (_par[_x] != _yk[0]).sum() != (_par[_cv] != _yk[0]).sum()                 and not _pz.menzil.h1[_e, _x]:
            _yt = _par[_x]
            break
    ok(_yt is not None, "TIP basamagi icin yanlis jeton sayili varlik bulundu")
    _dal = [("KAPI1 <YOK> ortada", _bzk, A.odul_zemin),
            ("KAPI3 olmayan varlik (uydurma uclu)", _uc, A.odul_zemin),
            ("KAPI3 olmayan varlik", _sahte, A.odul_zemin),
            ("TIP gecerli ama yanlis jeton sayisi", _yt, A.odul_tip),
            ("KISAYOL", _par[_k], A.odul_kisayol),
            ("menzil disi", _par[_uzl], A.odul_e),
            ("2 adim, aile yanlis", _par[_fkl], A.odul_f),
            ("2 adim + aile", _par[_ayl], A.odul_g_aile),
            ("TAM DOGRU", _par[_cv], A.odul_h)]
    _X = np.stack([d[1] for d in _dal])[None]
    _odv, _ = _pz.puanla(_X, np.array([_e]), np.array([_k]), np.array([_cv]))
    for (_nm, _, _bek), _got in zip(_dal, _odv[0]):
        ok(abs(float(_got) - _bek) < 1e-6,
           f"merdiven: {_nm} -> {_bek:+.2f}", f"gelen {float(_got):+.4f}")

# --- AVANTAJ: SABIT TERIM IPTAL OLMALI --------------------------------
# "Modelin ZATEN %100 yaptigi bir kriter odule ne agirlikla girerse
# girsin gradyani DEGISTIRMEZ" iddiasi burada sinaniyor. Iddia yanlissa
# merdivenin gerekcesi de yanlistir (onkayit model_04.md 3).
_r = torch.tensor([[0.1, 0.4, 0.9, 0.2]])
ok(torch.allclose(OD.avantaj(_r), OD.avantaj(_r + 7.0), atol=1e-5),
   "avantaj: SABIT terim TAM OLARAK iptal oluyor")
ok(float(OD.avantaj(torch.tensor([[0.3, 0.3, 0.3, 0.3]])).abs().max()) < 1e-6,
   "avantaj: VARYANSSIZ grup -> gradyan SIFIR")

# --- MERDIVEN SIRASI KILIDI -------------------------------------------
try:
    OD.OdulAyar(0.5, 0.4, 0.3, 0.2, 0.1, 1.0)
    _sira = False
except AssertionError:
    _sira = True
ok(_sira, "OdulAyar TERS merdiveni REDDEDIYOR",
   "sirasi bozuk bir merdiven sessizce kabul edilirdi")

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
