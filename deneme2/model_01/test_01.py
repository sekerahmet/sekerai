# -*- coding: utf-8 -*-
"""model_01 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_01 folderi
altinda olmali. model_01 diger hicbir model ile ayni seyi
kullanmamali."*

Dort sey tutulur:

  1. BAGIMSIZLIK  model_01/*.py icinde `model_a` / `model_b` /
     `veri_okul*` / `pencere_a` import'u YOK. Statik taramayla.
  2. MIMARI       projenin hicbir ozelligi yok: dongu yok, Phi
     darbogazi yok, maske yok, yardimci kayip yok, miras yok.
  3. VERI         `veri_01` grafi `veri_okul4`unkiyle BIREBIR AYNI.
  4. MOTOR        `taban_01` ile `model_a` AYNI SEYI olcuyor:
     egitim havuzu bit duzeyinde ayni, olcme izi ayni.

--------------------------------------------------------------------------
!! KILIT, PAYLASILAN MODULLERI IMPORT EDER -- VE ETMELIDIR

3 ve 4, model_01'in kopyalarini ORIJINALLERIYLE karsilastiriyor; bunun
icin `model_a` / `veri_okul4` / `model_b15` buraya yukleniyor. Bu,
1'deki bagimsizlik iddiasini BOZMAZ: KOSAN kol onlari gormez, yalniz
KILIT gorur. Kilidin isi zaten bu -- kopyanin sapip sapmadigini
soylemek.

Kopyanin GERCEK riski buydu: paylasilan motorda bir olcum hatasi
duzeltilirse buraya kendiliginden GELMEZ, ve model_01 ile model_b15 o
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
print("=== 0) BAGIMSIZLIK: model_01/ disariya BAGLI MI ===")
# Metinde arama YAPMIYORUZ -- ilk denemem oyleydi ve `ayar_01.py`nin
# DOCSTRING'inde gecen "from model_b15 import ..." cumlesini (kaldirilan
# seyi ANLATAN cumleyi) import sandi. Dogrusu AST: yalnizca gercek
# import dugumlerine bakilir.
import ast                                                   # noqa: E402

YASAK = ("model_a", "model_b", "pencere_a", "tani_a", "asama1", "sor",
         "kos", "veri_dok", "graf_dok", "havuz_dok",
         # !! `*_00` DA YASAK. model_01, model_00'in KOPYASI olarak dogdu;
         # yeniden adlandirmada BIR dosya atlanirsa model_01 sessizce
         # model_00'in motorunu/verisini kullanirdi ve iki kol AYNI
         # seyi kosardi. Bu satir onu imkansiz kiliyor.
         "taban_00", "veri_00", "ayar_00", "model_00", "pencere_00",
         "tani_00", "asama1_00", "sor_00", "kos_00", "analiz_00",
         "graf_00", "havuz_00", "test_00")
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

print("\n=== 1) model_01 MIMARISI ===")
import taban_01 as M                                         # noqa: E402
import model_01 as S                                         # noqa: E402

A = S.AYAR
ok(S.M is M, "model_01 motoru taban_01", S.M.__name__)
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64, "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, taban_01.Model'den MIRAS ALMIYOR")
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
# KUSUR (16 Eylul hakemligi): `asama1_01.gizli()` ileri gecisi ELLE
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
import asama1_01 as A1                                      # noqa: E402
_lst = v.comp[:8]
# !! MODELI M.DEV'E TASI. `gizli()` girdiyi `.to(M.DEV)` ediyor; model
# CPU'da kalirsa GPU'lu makinede "index is on cuda:0, other tensors on
# cpu" ile duser. ILK YAZIMDA BU EKSIKTI ve yerelde (DEV="cpu") FARK
# EDILMEDI -- kusur Colab'da, kilidin ilk GPU kosusunda cikti.
# Gercek kullanimda `asama1_01.main` modeli zaten `.to(M.DEV)` ediyor;
# yani hata ARACIN degil, BU TESTIN hatasiydi. Sonra CPU'ya geri
# aliniyor: asagidaki bolumler CPU tensorleriyle devam ediyor.
try:
    net.to(M.DEV)
    _q = A1.gizli(net, v, _lst, 6)
    ok(_q.shape == (8, A.d), "asama1_01.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
       f"{_q.shape} cihaz {M.DEV}")
except Exception as _e:                                      # noqa: BLE001
    ok(False, "asama1_01.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
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
print("\n=== 3) VERI: veri_01 == veri_okul4 (ICERIK) ===")
sys.path.insert(0, _K)                       # YALNIZ KILIT icin
sys.path.insert(0, os.path.join(_K, "model_a"))
sys.path.insert(0, os.path.join(_K, "model_b"))
import veri_01 as V00                                        # noqa: E402
import veri_okul4 as V4                                      # noqa: E402

ok(A.veri_ad == "veri_01", "model_01 KENDI veri modulunu okur", A.veri_ad)
_G0, _G4 = V00.kur(0), V4.kur(0)
ok(sorted(_G0) == sorted(_G4), "graf anahtarlari AYNI")
for _k in sorted(_G0):
    ok(_G0[_k] == _G4[_k], f"graf[{_k}] BIREBIR AYNI")
ok(V00.graf_izi(_G0) == V00.IZ, "graf izi veri_01.IZ ile TUTUYOR", V00.IZ)
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

# (3) `jeton_ad` EK JETONLARINDA cokmuyor. veri_dok/analiz_01 `ek_kip`
#     gelen her kolda IndexError veriyordu -- model_b15'in verisi bu
#     yuzden hic dokulmemisti.
sys.path.insert(0, _B)
import analiz_01 as AZ                                       # noqa: E402
_d = AZ.Dok("model_01")
_ekler = [_d.jeton_ad(i) for i in range(_d.v.ek0, _d.v.vocab)]
ok(_ekler == ["'", "<NIN>", "<SI>", "<DIR>"],
   "analiz_01.jeton_ad EK JETONLARINI adlandiriyor", str(_ekler))
for _g in ("kur", "zincirler", "TIPLER", "ILISKI", "SEMA", "GEREKTIRIR",
           "BLOK", "yaz", "turetilebilir"):
    ok(hasattr(V00, _g), f"veri_01.{_g} var (sozlesme)")

# --- 4) MOTOR: taban_01 ile model_a AYNI SEYI mi olcuyor ----------------
print("\n=== 4) MOTOR ESDEGERLIGI: taban_01 vs model_a ===")
import model_a as MA                                         # noqa: E402
from model_b15 import AYAR as B15                            # noqa: E402
from ayar_01 import GOREV_ALAN                               # noqa: E402

# !! model_01 motoru `taban_00`a gore TEK BIR ALAN ekliyor: `hop2_pay`.
# Bu kolun DUGMESI o. Baska hicbir fark OLMAMALI.
_ek = set(M.ESKI_VARSAYILAN) - set(MA.ESKI_VARSAYILAN)
ok(_ek == {"hop2_pay"}, "ESKI_VARSAYILAN'a YALNIZ hop2_pay eklendi", str(_ek))
ok(all(M.ESKI_VARSAYILAN[k] == MA.ESKI_VARSAYILAN[k]
       for k in MA.ESKI_VARSAYILAN), "diger ESKI_VARSAYILAN degerleri AYNI")
ok(M.ESKI_VARSAYILAN["hop2_pay"] == 1.0,
   "hop2_pay ESKI VARSAYILANI 1.0 -- eski kosularda 2-hop VARDI")
_a1 = [f.name for f in M.dc.fields(M.Ayar)]
_a0 = [f.name for f in MA.dc.fields(MA.Ayar)]
ok(set(_a1) - set(_a0) == {"hop2_pay"} and set(_a0) - set(_a1) == set(),
   "Ayar ALANLARI: yalniz hop2_pay EKLENDI",
   f"{sorted(set(_a1)-set(_a0))} / {sorted(set(_a0)-set(_a1))}")
ok(M.T_LEN == MA.T_LEN and M.SPECIAL == MA.SPECIAL, "sabitler AYNI")
# Ayar VARSAYILANLARI: yalniz BILEREK degistirilen ikisi farkli olmali.
# Ucuncusu cikarsa kopya sapmis demektir. (Varsayilanlar yeniden kurulusta
# KULLANILMIYOR -- `ayar_oku` eksik alanlari ESKI_VARSAYILAN'dan
# dolduruyor -- ama sapma yine de GORULSUN.)
import dataclasses as _dcc                                   # noqa: E402
# !! ORTAK alanlar uzerinden: `hop2_pay` model_a'da YOK (bu kolun
# dugmesi), onu burada karsilastirmak AttributeError verir.
_ortak = ({f.name for f in _dcc.fields(M.Ayar)}
          & {f.name for f in _dcc.fields(MA.Ayar)})
_vf = sorted(k for k in _ortak
             if getattr(M.Ayar(), k) != getattr(MA.Ayar(), k))
ok(_vf == ["ad", "veri_ad"],
   "ORTAK alanlarin VARSAYILANLARI yalniz ad + veri_ad'da farkli", str(_vf))
ok(M.Ayar().hop2_pay == 1.0,
   "taban_01 VARSAYILANI hop2_pay=1.0 -- dugmeye DOKUNULMAZSA eski davranis")

for f in GOREV_ALAN:
    ok(getattr(A, f) == getattr(B15, f), f"gorev alani {f} model_b15 ile AYNI",
       f"{getattr(A, f)!r} vs {getattr(B15, f)!r}")
ok("veri_ad" not in GOREV_ALAN, "veri_ad GOREV_ALAN'da YOK (ad degil icerik)")
# !! `Ayar.fark` CAGIRANIN alanlari uzerinden yuruyor; `hop2_pay`
# model_b15'te YOK, o yuzden listeye kendiliginden GIRMEZ. Bu kolun
# dugmesi gorunmez kalmasin diye ACIKCA ekleniyor.
_fark = sorted(set(B15.fark(A)) | {"ad", "hop2_pay"})
ok(_fark == ["ad", "betas", "dar_alfa", "dar_kapi", "dff", "dongu",
             "hop2_pay", "l", "ort_bas", "veri_ad", "wd"],
   f"model_01 <-> model_b15 farki {_fark}",
   "MIMARI + STANDART TARIF + veri ADI + IKI DUGME (hop2_pay, wd)")

v15 = MA.veri_kur(B15, yaz=lambda *a: None)
ok(M.olcme_izi(M.olcme_listeleri(A, v))
   == MA.olcme_izi(MA.olcme_listeleri(B15, v15)),
   "OLCME IZI ayni -> AYNI SINAV, sayilar ayni tabloda okunur",
   M.olcme_izi(M.olcme_listeleri(A, v)))

# --- 4b) BU KOLUN DUGMESI: hop2_pay --------------------------------------
# Iki yonlu sinaniyor:
#   1.0 -> havuz model_00'inkiyle BIT DUZEYINDE ayni olmali. Degilse
#          degisiklik sessizce BASKA bir seyi de bozmus demektir.
#   0.0 -> egitimde TEK BIR 2-hop satiri kalmamali.
print("")
print("=== 4b) BU KOLUN DUGMESI: hop2_pay ===")
ok(A.hop2_pay == 0.0, "model_01 hop2_pay = 0.0 (EGITIMDE 2-HOP YOK)",
   str(A.hop2_pay))
X0, P0, T0, _, _, _ = M.egitim_havuzu(A, v, yaz=lambda *a: None)

# veri_ad DEGISTIRILMIYOR: havuz GRAFA bagli, modul ADINA degil ve
# veri_01 ile veri_00'in icerigi bu kolda zaten BIREBIR ayni (bolum 3).
_A1 = A.degistir(hop2_pay=1.0)
_v1 = M.veri_kur(_A1, yaz=lambda *a: None)
Xf, Pf, Tf, _, _, _ = M.egitim_havuzu(_A1, _v1, yaz=lambda *a: None)
sys.path.insert(0, os.path.join(_K, "model_00"))
import taban_00 as T00                                       # noqa: E402
from ayar_00 import AYAR as A00                              # noqa: E402
_v0 = T00.veri_kur(A00, yaz=lambda *a: None)
X00, P00, T00x, _, _, _ = T00.egitim_havuzu(A00, _v0, yaz=lambda *a: None)
ok(Xf.shape == X00.shape and (Xf == X00).all(),
   "hop2_pay=1.0 -> HAVUZ model_00 ile BIT DUZEYINDE ayni",
   f"{Xf.shape} vs {X00.shape}")
ok((Pf == P00).all() and (Tf == T00x).all(), "ayni durumda P ve T de ayni")
ok(M.olcme_izi(M.olcme_listeleri(A, v))
   == T00.olcme_izi(T00.olcme_listeleri(A00, _v0)),
   "SINAV model_00 ile AYNI -> iki kol DOGRUDAN kiyaslanir", "d6751004648c")

# hop2_pay=0 gercekten 2-hop BIRAKMIYOR mu -- SATIRLARDAN sayilarak.
# 2-hop dizisinde IKI iliski jetonu vardir, 1-hop'ta BIR.
_R0, _NR = M.SPECIAL, v.n_rel
_ril = ((X0 >= _R0) & (X0 < _R0 + _NR)).sum(1)
ok(int((_ril >= 2).sum()) == 0,
   "havuzda IKI iliskili (2-hop) SATIR YOK",
   f"{int((_ril >= 2).sum())} satir")
ok(X0.shape[0] < Xf.shape[0] // 2,
   "havuz KUCULDU", f"{X0.shape[0]:,} vs model_00 {Xf.shape[0]:,}")
_rilf = ((Xf >= _R0) & (Xf < _R0 + _NR)).sum(1)
ok(int((_rilf >= 2).sum()) > 0,
   "KONTROL: hop2_pay=1.0'da 2-hop satirlari VAR",
   f"{int((_rilf >= 2).sum()):,} satir")

# --- 5) STANDART TARIF -- DEVRALINMAYANLAR ------------------------------
print("\n=== 5) STANDART TARIF ===")
ok(A.ort_bas == 0, "LOOKAHEAD ORTALAMASI KAPALI (ort_bas=0)",
   f"b15 {B15.ort_bas} -> 00 {A.ort_bas}")
ok(B15.ort_bas == 10000, "model_b15'te ACIK (DEGISMEDI)", str(B15.ort_bas))
ok(A.betas == (0.9, 0.95),
   "betas (0.9, 0.95) -- nanoGPT/GPT-3/Llama/Pythia", str(A.betas))
ok(B15.betas == (0.9, 0.999),
   "model_b15'te PyTorch varsayilani (DEGISMEDI)", str(B15.betas))
# !! wd ARTIK STANDART DEGIL -- bu kosunun DUGMESI. CLAUDE.md kural 4
# "wd 0.1 ARTIK ARANMAZ" diyordu; kullanici 17 Eylul'de bu kolda deldi.
# Kilit gevsetilmedi, HEDEFI DEGISTI: 0.1 degil 0.5 bekliyor.
ok(A.wd == 0.5, "wd 0.5 -- 0.1'in BES KATI, IKINCI KOSUNUN DUGMESI",
   str(A.wd))
ok(B15.wd == 0.1, "model_b15'te 0.1 (DEGISMEDI)", str(B15.wd))
ok(A.sabit_lr is False, "cosine -> lr/10 (CLAUDE.md kural 4, DEGISMEDI)")
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
    ok(hasattr(S, g), f"model_01.{g} var", "kos_01.py duser")
for ad in ("veri_01", "taban_01", "ayar_01", "pencere_01", "tani_01",
           "asama1_01", "sor_01", "kos_01"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True, cwd=_B)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import (yalniz model_01/)",
       _r2.stderr.strip()[-200:])

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
