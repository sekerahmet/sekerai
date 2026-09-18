# -*- coding: utf-8 -*-
"""model_08 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_08 folderi
altinda olmali. model_08 diger hicbir model ile ayni seyi
kullanmamali."*

Dort sey tutulur:

  1. BAGIMSIZLIK  model_08/*.py icinde `model_a` / `model_b` /
     `veri_okul*` / `pencere_a` import'u YOK. Statik taramayla.
  2. MIMARI       projenin hicbir ozelligi yok: dongu yok, Phi
     darbogazi yok, maske yok, yardimci kayip yok, miras yok.
  3. VERI         `veri_08` KENDI iddialarini tutuyor: sema, soy
                  agaci (cinsiyet/kusak/ensest), zincir siniflari,
                  turetilebilirlik, notrluk, cografya, IZ.
  4. MOTOR        `taban_08` ile `model_a` AYNI SEYI olcuyor:
     egitim havuzu bit duzeyinde ayni, olcme izi ayni.

--------------------------------------------------------------------------
!! KILIT, PAYLASILAN MODULLERI IMPORT EDER -- VE ETMELIDIR

3 ve 4, model_08'in kopyalarini ORIJINALLERIYLE karsilastiriyor; bunun
icin `model_a` / `veri_okul4` / `model_b15` buraya yukleniyor. Bu,
1'deki bagimsizlik iddiasini BOZMAZ: KOSAN kol onlari gormez, yalniz
KILIT gorur. Kilidin isi zaten bu -- kopyanin sapip sapmadigini
soylemek.

Kopyanin GERCEK riski buydu: paylasilan motorda bir olcum hatasi
duzeltilirse buraya kendiliginden GELMEZ, ve model_08 ile model_b15 o
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
print("=== 0) BAGIMSIZLIK: model_08/ disariya BAGLI MI ===")
# Metinde arama YAPMIYORUZ -- ilk denemem oyleydi ve `ayar_08.py`nin
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

print("\n=== 1) model_08 MIMARISI ===")
import taban_08 as M                                         # noqa: E402
import model_08 as S                                         # noqa: E402

A = S.AYAR
ok(S.M is M, "model_08 motoru taban_08", S.M.__name__)
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64, "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, taban_08.Model'den MIRAS ALMIYOR")
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
# KUSUR (16 Eylul hakemligi): `asama1_08.gizli()` ileri gecisi ELLE
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
import asama1_08 as A1                                      # noqa: E402
_lst = v.comp[:8]
# !! MODELI M.DEV'E TASI. `gizli()` girdiyi `.to(M.DEV)` ediyor; model
# CPU'da kalirsa GPU'lu makinede "index is on cuda:0, other tensors on
# cpu" ile duser. ILK YAZIMDA BU EKSIKTI ve yerelde (DEV="cpu") FARK
# EDILMEDI -- kusur Colab'da, kilidin ilk GPU kosusunda cikti.
# Gercek kullanimda `asama1_08.main` modeli zaten `.to(M.DEV)` ediyor;
# yani hata ARACIN degil, BU TESTIN hatasiydi. Sonra CPU'ya geri
# aliniyor: asagidaki bolumler CPU tensorleriyle devam ediyor.
try:
    net.to(M.DEV)
    _q = A1.gizli(net, v, _lst, 6)
    ok(_q.shape == (8, A.d), "asama1_08.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
       f"{_q.shape} cihaz {M.DEV}")
except Exception as _e:                                      # noqa: BLE001
    ok(False, "asama1_08.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
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

# --- 3) VERI: veri_08'in KENDI IDDIALARI --------------------------------
# !! model_03'ten DEVRALINAN "kopya == orijinal" TESTI KALDIRILDI.
# Orada `veri_03` bir KOPYAYDI (icerigi `veri_okul4` ile birebir) ve test
# kopyanin kaymadigini siniyordu. `veri_08` KOPYA DEGIL: yeni sema, yeni
# tipler, gercek soy agaci, gercek cografya. Kiyaslanacak bir ORIJINAL
# YOK. O yuzden test artik verinin KENDI IDDIALARINI siniyor -- ve o
# iddialarin cogu `veri_04`te TUTMUYORDU ("Fatma'nin annesi Huseyin").
print("\n=== 3) VERI: veri_08 KENDI IDDIALARINI tutuyor mu ===")
import veri_08 as V00                                        # noqa: E402

ok(A.veri_ad == "veri_08", "BU KOL KENDI veri modulunu okur", A.veri_ad)
_G0 = V00.kur(0)
E05 = [x for t in V00.TIPLER for x in _G0["ad"][t]]
_z = V00.zincirler(_G0)
ok(V00.graf_izi(_G0) == V00.IZ, "graf izi veri_08.IZ ile TUTUYOR", V00.IZ)
ok(V00.graf_izi(V00.kur(0)) == V00.IZ, "IZ ikinci kurulusta da AYNI")
ok(set(_G0["ad"]) == set(V00.TIPLER), "ad sozlugu TIPLER'i tam kapsiyor",
   f"{len(V00.TIPLER)} tip, {sum(_G0['n'].values())} varlik")
ok(max(len(a.split("_")) for a in _G0["tip"]) == 3,
   "varlik EN COK 3 kelime (t_len = 3*yuva + 15'in dayandigi sayi)")

# --- 3a) SEMA SOZLESMESI: her olgu semanin VAAT ETTIGI tipte ------------
_sema_ihlal = [(e, r, h) for (e, r), h in _G0["olgu"].items()
          if _G0["tip"][e] not in V00.SEMA[r]
          or _G0["tip"][h] != V00.SEMA[r][_G0["tip"][e]]]
ok(not _sema_ihlal, "her olgu SEMA'nin vaat ettigi tipte",
   str(_sema_ihlal[:2]))
ok(not [1 for (e, r), h in _G0["olgu"].items() if h == e],
   "kendine giden olgu YOK")

# --- 3b) SOY AGACI -- `veri_04`te bunlarin HICBIRI tutmuyordu ----------
_c, _k = _G0["cinsiyet"], _G0["kusak"]
ok(all(_c[h] == "K" for (e, r), h in _G0["olgu"].items() if r == "annesi"),
   "annesi HER ZAMAN kadin")
ok(all(_c[h] == "E" for (e, r), h in _G0["olgu"].items() if r == "babasi"),
   "babasi HER ZAMAN erkek")
ok(all(_k[h] == _k[e] - 1 for (e, r), h in _G0["olgu"].items()
       if r in ("annesi", "babasi")), "ebeveyn TAM BIR ust kusak")
ok(all(_k[h] == _k[e] for (e, r), h in _G0["olgu"].items() if r == "kardesi"),
   "kardes AYNI kusak")
_ks = [(x, _G0["olgu"][(x, "kardesi")]) for x in _G0["ad"]["KISI"]]
ok(all(_G0["olgu"].get((a, r)) == _G0["olgu"].get((b, r))
       for a, b in _ks for r in ("annesi", "babasi")),
   "kardesler AYNI anne VE AYNI babadan")
ok(all(_G0["olgu"].get((_G0["olgu"][(x, "annesi")], "kardesi"))
       != _G0["olgu"].get((x, "babasi"))
       for x in _G0["ad"]["KISI"] if (x, "annesi") in _G0["olgu"]),
   "ENSEST YOK: ebeveyn cifti kardes DEGIL")
ok(all(_G0["olgu"][(h, "annesi" if _c[e] == "K" else "babasi")] == e
       for (e, r), h in _G0["olgu"].items() if r == "cocugu"),
   "cocugu, annesi/babasi'nin TERSI")
# KENAR gercekten kenarda: ebeveyni olmayan YALNIZ kusak 0
ok({_k[x] for x in _G0["ad"]["KISI"] if (x, "annesi") not in _G0["olgu"]} == {0},
   "ebeveyni kayitli olmayan YALNIZ kusak 0")
ok({_k[x] for x in _G0["ad"]["KISI"] if (x, "cocugu") not in _G0["olgu"]} == {2},
   "cocugu kayitli olmayan YALNIZ son kusak")

# --- 3c) ZINCIR SINIFLARI: YOK ile EKSIK AYRI olmali -------------------
# `ent_yok` bolmesinin TANIMI "kisayol TIP OLARAK imkansiz" (CLAUDE.md).
# Ikisi karisirsa o bolme sessizce kirlenir.
_sn = {}
for _x in _z:
    _sn[_x[6]] = _sn.get(_x[6], 0) + 1
ok(set(_sn) == {"AYIRT", "YOK", "AYNI", "DONUS", "EKSIK"},
   "bes sinif da var", str(_sn))
ok(all(_G0["tip"][x[0]] not in V00.SEMA[x[2]] for x in _z if x[6] == "YOK"),
   "YOK sinifinda kisayol GERCEKTEN tip olarak imkansiz")
ok(all(_G0["tip"][x[0]] in V00.SEMA[x[2]] and x[5] is None
       for x in _z if x[6] == "EKSIK"),
   "EKSIK sinifinda kisayol TIP OLARAK MUMKUN ama olgu YOK")
ok(all(x[5] is not None for x in _z if x[6] in ("AYIRT", "AYNI")),
   "AYIRT/AYNI sinifinda kisayol VAR")
ok(all(_G0["olgu"].get((x[0], x[1])) == x[3]
       and _G0["olgu"].get((x[3], x[2])) == x[4] for x in _z[:2000]),
   "zincirin IKI ADIMI da gercek olgu")

# --- 3d) TURETILEBILIR: yalniz GEREKTIRIR'dekiler ----------------------
# !! ESIK 0.30, 0.50 DEGIL. Olculdu (17 Eylul): dort turetilebilir ciftin
# ikisi TAM 0.500'de duruyor --
#     annesi + cocugu -> kardesi   0.500
#     babasi + cocugu -> kardesi   0.500
# cunku o zincirin diger yarisi DONUS (kisinin kendisi). `turetilebilir`
# esigi KATI BUYUKLUK ile suzuyor, yani 0.50 esigi bu ikisini KACIRIR ve
# test "hepsi bildirilmis" derken aslinda YARISINI gormemis olurdu.
_t = [(r1, r2) for r1, r2, r3, o in V00.turetilebilir(_G0, esik=0.30)]
ok(set(_t) == set(V00.GEREKTIRIR),
   "turetilebilir cift (esik 0.30) YALNIZ GEREKTIRIR'dekiler", str(_t))

# --- 3e) NOTRLUK: ad USTUNDEKINI sizdirmasin ---------------------------
ok(all(a.split("_")[0] != _G0["olgu"][(a, "universitesi")].split("_")[0]
       for a in _G0["ad"]["FAKULTE"]), "FAKULTE adi universitesini SIZDIRMIYOR")
ok(all(a.split("_")[0] != _G0["olgu"][(a, "fakultesi")].split("_")[0]
       for a in _G0["ad"]["BOLUM"]), "BOLUM adi fakultesini SIZDIRMIYOR")
_kop, _yv = V00.sizinti(_G0, _z)
ok(0 < _kop < len(_z) // 2,
   "AILE soyadi sizintisi VAR ama azinlikta (KASITLI, olculuyor)",
   f"{_kop}/{len(_z)} zincir, yuva bazinda {dict(sorted(_yv.items()))}")

# --- 3f) GERCEK COGRAFYA ------------------------------------------------
ok(all(_G0["olgu"][(s, "bolgesi")] == V00.SEHIR_BOLGE[s] + "_Bolgesi"
       for s in _G0["ad"]["SEHIR"]),
   "sehir -> bolge GERCEK cografya (rastgele DEGIL)")

# --- 3g) IZ KOR MU -- model_03'ten devralinan uc bozma testi ------------
# Ilk surum `ad` listelerini SIRALAYIP kariyordu ve `sema`yi hic
# karmiyordu: varlik sirasini ters cevirmek (butun jeton id'leri kaydirir)
# ve SEMA'ya tip cifti eklemek (SINAVI degistirir) IZ'i KIPIRDATMIYORDU.
import copy as _cp                                           # noqa: E402
_iz0 = V00.graf_izi(_G0)
for _ad, _boz in (
        ("varlik SIRASI", lambda g: g["ad"].__setitem__(
            "KISI", list(reversed(g["ad"]["KISI"])))),
        ("SEMA tip cifti", lambda g: g["sema"]["onkosulu"].__setitem__(
            "KISI", "DERS")),
        ("sozluk SIRASI", lambda g: g.__setitem__(
            "sozluk", list(reversed(g["sozluk"]))))):
    _h = _cp.deepcopy(_G0)
    _boz(_h)
    ok(V00.graf_izi(_h) != _iz0, f"IZ {_ad} degisimini GORUYOR")

# (3) `jeton_ad` EK JETONLARINDA cokmuyor. veri_dok/analiz_08 `ek_kip`
#     gelen her kolda IndexError veriyordu -- model_b15'in verisi bu
#     yuzden hic dokulmemisti.
sys.path.insert(0, _B)
import analiz_08 as AZ                                       # noqa: E402
_d = AZ.Dok("model_08")
_ekler = [_d.jeton_ad(i) for i in range(_d.v.ek0, _d.v.vocab)]
# ek_kip="tr2": <SI> YOK (iliski kendi iyelik ekini tasiyor),
# yerine SORU SOZCUKLERI var. Kullanici karari, 17 Eylul.
# ek_kip="tr2" ek blogu:  '  + tamlayan allomorflari (8) +
# bildirme allomorflari (8) + soru sozcukleri.  Hicbiri <SOYUT>
# isaretleyici DEGIL -- hepsi GERCEK Turkce ek ya da kelime.
# model_06 UC JETON EKLEDI, hepsi sozlugun SONUNDA (REL_OFF ve ent_off
# KAYMASIN diye):
#   <BOS>   bosluk doldurmada cikarilan parcanin yeri
#   <AYIR>  "cumle bitti, simdi o parca geliyor"
#   ,       devrik cumlede OGE SINIRI -- onsuz "Kardesidir Ozlem Yilmaz
#           Ibrahim Yilmaz'in"da iki ad yan yana geliyor ve cevabin
#           nerede bittigi BELIRSIZ kaliyordu (havuz_06 yakaladi).
_bek = (["'"] + list(V00.EK_NIN) + list(V00.EK_DIR)
        + sorted(set(V00.SORU_SOZ.values()))
        + ["<BOS>", "<AYIR>", ","])
ok(_ekler == _bek, "analiz_08.jeton_ad EK JETONLARINI adlandiriyor",
   f"{len(_ekler)} jeton")
ok(_d.v.vocab - _d.v.ek0 == 23, "ek blogu 20 -> 23 jeton (BOSLUK + virgul)",
   f"{_d.v.vocab - _d.v.ek0}")
ok(M.REL_OFF == 3 and _d.v.ent_off == 3 + _d.v.n_rel,
   "REL_OFF ve ent_off KAYMADI (yeni jetonlar SONA eklendi)",
   f"REL_OFF {M.REL_OFF}  ent_off {_d.v.ent_off}")
# Allomorf SECIMI dogru mu -- unlu uyumu ve sert unsuz.
_G05 = V00.kur(0)
_es = V00.ek_secim(_G05)
_sin = [("Yilmaz", "ın", "dır"), ("Kaya", "nın", "dır"),
        ("Demir", "in", "dir"), ("Celik", "in", "tir"),
        ("Ozturk", "ün", "tür")]
for _soy, _n, _d in _sin:
    _a = next((x for x in _G05["ad"]["KISI"] if x.endswith("_" + _soy)), None)
    if _a is None:
        continue
    ok(V00.EK_NIN[_es["nin_varlik"][_a]] == _n
       and V00.EK_DIR[_es["dir_varlik"][_a]] == _d,
       f"{_soy}: tamlayan '{_n}, bildirme '{_d}",
       f"'{V00.EK_NIN[_es['nin_varlik'][_a]]} "
       f"'{V00.EK_DIR[_es['dir_varlik'][_a]]}")
ok(V00.EK_NIN[_es["nin_iliski"]["kardesi"]] == "nin"
   and V00.EK_NIN[_es["nin_iliski"]["bolumu"]] == "nün",
   "iliski + tamlayan: kardesi+nin, bolumu+nun")
# `yaz` LISTEDEN CIKTI: veri_04'te vardi ama hicbir yerde cagrilmiyor
# (tarandi, 17 Eylul). Sozlesme GERCEKTEN kullanilani sayar.
for _g in ("kur", "zincirler", "TIPLER", "ILISKI", "SEMA", "GEREKTIRIR",
           "BLOK", "turetilebilir", "graf_izi", "IZ", "sizinti",
           "SEHIR_BOLGE", "yuzey"):
    ok(hasattr(V00, _g), f"veri_08.{_g} var (sozlesme)")

# `model_a` / `model_b15` YALNIZ BU KILIT icin yukleniyor -- bu bolum
# motorun eski aileyle AYNI SEYI olctugunu siniyor, kosuda kullanilmaz.
# (Yol eklemeleri eskiden 3. bolumun icindeydi; o bolum `veri_okul4`
# kiyasiyla birlikte kaldirilinca buraya tasindi.)
sys.path.insert(0, _K)
sys.path.insert(0, os.path.join(_K, 'model_a'))
sys.path.insert(0, os.path.join(_K, 'model_b'))

# --- 4) MOTOR: taban_08 ile model_a AYNI SEYI mi olcuyor ----------------
print("\n=== 4) MOTOR ESDEGERLIGI: taban_08 vs model_a ===")
import model_a as MA                                         # noqa: E402
from model_b15 import AYAR as B15                            # noqa: E402
from ayar_08 import GOREV_ALAN                               # noqa: E402

# BILDIRILMIS AYRISMA -- IKI YENI ALAN, ve ikisi de KAPALI GELIYOR:
#   fim_kat   bosluk doldurma varyant sayisi   (model_06)
#   soru_kat  soru satiri sayisi               (model_07)
#   kisayol_kat  iyelik kisa yolu satiri       (model_08)
# ESKI_VARSAYILAN'a ikisi de 0 diye girildi, cunku alanlar eklenmeden
# ONCE kodun FIILEN yaptigi sey buydu: boyle bir satir tipi yoktu.
# Yani eski kosular SESSIZCE yanlis etiketlenmiyor.
_YENI_ALAN = {"fim_kat", "soru_kat", "kisayol_kat"}
ok(set(MA.ESKI_VARSAYILAN) | _YENI_ALAN == set(M.ESKI_VARSAYILAN),
   "ESKI_VARSAYILAN model_a ile AYNI, YALNIZ yeni alanlar eklendi",
   f"fazla {sorted(set(M.ESKI_VARSAYILAN) - set(MA.ESKI_VARSAYILAN))}")
ok(all(M.ESKI_VARSAYILAN[k] == 0 for k in _YENI_ALAN),
   "yeni alanlarin ESKI VARSAYILANI 0 -- eskiden boyle satir YOKTU",
   f"{ {k: M.ESKI_VARSAYILAN[k] for k in sorted(_YENI_ALAN)} }")
ok(all(MA.ESKI_VARSAYILAN[k] == M.ESKI_VARSAYILAN[k]
       for k in MA.ESKI_VARSAYILAN),
   "model_a'nin ESKI VARSAYILANLARININ HICBIRI degismedi")
_alan06 = [f.name for f in M.dc.fields(M.Ayar)]
_alanA = [f.name for f in MA.dc.fields(MA.Ayar)]
ok(set(_alan06) - set(_alanA) == _YENI_ALAN,
   "Ayar alanlari model_a ile AYNI, YALNIZ fim_kat + soru_kat + "
   "kisayol_kat eklendi",
   f"fazla {sorted(set(_alan06) - set(_alanA))}")
ok(not (set(_alanA) - set(_alan06)), "model_a'nin hicbir alani SILINMEDI",
   f"eksik {sorted(set(_alanA) - set(_alan06))}")
# !! `SPECIAL` BILEREK AYRILDI (17 Eylul). model_a'da 8, burada 3.
# Olculdu: sekiz ozel jetonun BESI havuzda HIC gecmiyordu --
#   [S1] [S2] [KIMLIK] ve iki adlandirilmamis yuva.
# Bunlari silmek `ek_kip=""` ve `ek_kip="tr"` kodlama yollarinin da
# silinmesini gerektirdi (Q1/Q2/IDENT yalniz onlar icindi), yani
# `taban_08` artik ortak motorun BIREBIR KOPYASI DEGIL.
# Kullanici karari: "silinsin tabii ki".
#
# Kopya-sapma bekcisi BU YUZDEN KALDIRILMADI, YON DEGISTIRDI:
# degismesi GEREKEN ayrildi mi, degismemesi gereken AYNI mi.
ok(M.T_LEN == MA.T_LEN, "T_LEN model_a ile AYNI (eksiz taban)", str(M.T_LEN))
ok(M.SPECIAL == 3 and MA.SPECIAL == 8,
   "SPECIAL model_a'dan BILEREK ayrildi (olu jetonlar silindi)",
   f"model_a {MA.SPECIAL} -> model_08 {M.SPECIAL}")
ok(not hasattr(M, "IDENT") and not hasattr(M, "Q1"),
   "Q1 / Q2 / IDENT sabitleri SILINDI (kodlama yollariyla birlikte)")
ok(M.REL_OFF == M.SPECIAL, "REL_OFF hala SPECIAL'a bagli", str(M.REL_OFF))
# Ayar VARSAYILANLARI: yalniz BILEREK degistirilen ikisi farkli olmali.
# Ucuncusu cikarsa kopya sapmis demektir. (Varsayilanlar yeniden kurulusta
# KULLANILMIYOR -- `ayar_oku` eksik alanlari ESKI_VARSAYILAN'dan
# dolduruyor -- ama sapma yine de GORULSUN.)
import dataclasses as _dcc                                   # noqa: E402
# !! YALNIZ IKISINDE DE OLAN alanlar kiyaslanir. `fim_kat` model_06'nin
# YENI alani; `getattr(MA.Ayar(), "fim_kat")` AttributeError verirdi.
_ortak = {f.name for f in _dcc.fields(MA.Ayar)}
_vf = sorted(f.name for f in _dcc.fields(M.Ayar)
             if f.name in _ortak
             and getattr(M.Ayar(), f.name) != getattr(MA.Ayar(), f.name))
ok(_vf == ["ad", "veri_ad"],
   "ORTAK alanlarin varsayilanlari yalniz ad + veri_ad'da farkli", str(_vf))
ok(M.Ayar().fim_kat == 0,
   "fim_kat VARSAYILANI 0 -- bosluk doldurma KAPALI gelir",
   "acan sey ayar_08, yani BU KOLUN karari")

# BILDIRILMIS AYRISMA. `GOREV_ALAN` "sinavin AYNI kalmasi GEREKEN
# alanlari" listesi ve model_03'e kadar hepsi model_b15 ile birebirdi.
# model_08 ikisini BILEREK degistiriyor ve ikisi de BU KOLUN TANIMI:
#   veri_ad  yeni veri (zaten GOREV_ALAN'da degil)
#   ek_kip   "tr" -> "tr2": iliski jetonu KELIMENIN KENDISI, <SI> YOK,
#            soru sozcugu VAR. Kullanici, 17 Eylul: "ben duzgun bir
#            turkce ile egitim istiyorum."
# Susturmuyoruz -- LISTELIYORUZ. Listede OLMAYAN bir alan kayarsa test
# yine duser.
AYRILAN = {"ek_kip": "tr2: iliski kendi ekini tasir, <SI> yok, soru sozcugu var"}
for f in GOREV_ALAN:
    if f in AYRILAN:
        ok(getattr(A, f) != getattr(B15, f),
           f"gorev alani {f} model_b15'ten BILEREK ayrildi",
           f"{getattr(B15, f)!r} -> {getattr(A, f)!r}   ({AYRILAN[f]})")
        continue
    ok(getattr(A, f) == getattr(B15, f), f"gorev alani {f} model_b15 ile AYNI",
       f"{getattr(A, f)!r} vs {getattr(B15, f)!r}")
ok("veri_ad" not in GOREV_ALAN, "veri_ad GOREV_ALAN'da YOK (ad degil icerik)")
_fark = sorted(set(B15.fark(A)) | {"ad"})
# !! `ort_her` BU KOLDA LISTEYE GIRDI. model_b15 `ort_her=0` (yani
# "olc_her'i kullan") birakmisti; model_08 2000'i ACIKCA yaziyor.
# Ikisi SAYICA ayni sonucu verir -- fark BEYANDIR, davranis degil:
# kolun dugmesi ort_bas ise, esik araligi da GORUNUR olmali.
ok(_fark == ["ad", "betas", "dar_alfa", "dar_kapi", "dff", "dongu",
             "ek_kip", "l", "ort_bas", "ort_her", "sabit_lr", "veri_ad",
             "wd"],
   f"model_08 <-> model_b15 farki {_fark}",
   "MIMARI + veri ADI + RECETE (wd, sabit_lr) + BU KOLUN DUGMESI (ort_*)")

v15 = MA.veri_kur(B15, yaz=lambda *a: None)
_iz05 = M.olcme_izi(M.olcme_listeleri(A, v))
_iz15 = MA.olcme_izi(MA.olcme_listeleri(B15, v15))
X0, P0, T0, _, _, _ = M.egitim_havuzu(A, v, yaz=lambda *a: None)
X1, P1, T1, _, _, _ = MA.egitim_havuzu(B15, v15, yaz=lambda *a: None)

# !! model_03'ten DEVRALINAN IKI ESITLIK TESTI, BILDIRILMIS AYRISMAYA
# CEVRILDI. Orada `veri_03` bir kopyaydi, sinav ve havuz model_b15 ile
# BIT DUZEYINDE aynidir diye sinaniyordu -- bu kolun BUTUN degeri de
# "sayilar ayni tabloda okunuyor" olmasindaydi.
#
# `veri_08` bu kolun DUGMESI: yeni sema, yeni tipler, gercek soy agaci.
# Sinavin ve havuzun degismesi bu kolun ARIZASI DEGIL, TANIMI. CLAUDE.md:
#   "olcme izi degisirse -> KOL IPTAL DEGIL. Not duser, devam edilir."
#
# Ama "degisti" demek yetmez, YONU sinanir: DEGISMESI GEREKENLER degisti,
# DEGISMEMESI GEREKENLER (dizi uzunlugu, yuva sayisi, kodlama sekli)
# AYNI kaldi. Yoksa sessiz bir kodlama hatasi "veri degisti zaten" diye
# gecerdi.
ok(_iz05 != _iz15,
   "OLCME IZI model_b15'ten AYRILDI -- BEKLENEN (veri bu kolun dugmesi)",
   f"model_08 {_iz05}  vs  model_b15 {_iz15}")
ok(X0.shape != X1.shape,
   "EGITIM HAVUZU da ayrildi -- BEKLENEN", f"{X0.shape} vs {X1.shape}")
# t_len de DEGISTI (17 -> 16) ve bu da BILEREK: <SI> cikti (-2, iki
# iliski icin), soru sozcugu girdi (+1). Turetim `Ayar.t_len`de yazili:
#   2-hop bicim 0 = e(3) ' <NIN> r1 <NIN> r2 SORU ? a(3) ' <DIR> EOS
#                 = 2*yuva + 10 = 16
# Sinanan sey artik "ayni kaldi" DEGIL, "TURETIMLE TUTUYOR".
# t_len = 3*yuva + 15: soru (yuva+7) + TAM CUMLE cevap (2*yuva+8).
# Kullanici karari 17 Eylul: cevap soruyu YENIDEN YAZIP oyle cevapliyor.
# model_06: SORU BICIMI YOK (kullanici, 17 Eylul: "yalniz bildirim
# kullanalim"). En uzun satir DEVRIK bildirim:
#   a(3) ' <DIR> ,  e(3) ' <NIN> r1 <NIN> r2 .  = 2*yuva + 9 = 15
# BOSLUK DOLDURMA +2 (bosluk isareti + ayirac). t_len = 2*yuva + 11.
# model_05'te 3*yuva + 15 = 24 idi; farkin tamami cevabin soruyu
# birebir tekrar etmesiydi (o 7 jetonun kosullu entropisi 0,000).
ok(X0.shape[1] == A.t_len == 2 * v.yuva + 11,
   "t_len TURETIMLE tutuyor (2*yuva + 11)", f"{A.t_len} (model_b15 {X1.shape[1]})")
ok(A.t_len < 24, "t_len model_05'ten KUCUK (soru bicimi kalkti)",
   f"{A.t_len} < 24")
# !! CEVAP YUVA SAYISI 3 -> 4, ve bu BILEREK (17 Eylul, kullanici:
# "<YOK> sil, gereksiz"). Adlar artik DEGISKEN uzunlukta; cevap
# hedefleri "adin kelimeleri + KESME ISARETI" oldu, yani en fazla
# yuva+1 = 4. Kisa adlarda kalan yuvalar -1 ile MASKELI.
# Olcunun ANLAMI da degisti: artik "dogru adi yazdi" DEGIL,
# "dogru adi yazdi VE dogru yerde bitirdi".
ok(P0.shape[1] == T0.shape[1] == v.cevap_yuva == v.yuva + 1,
   "cevap yuvasi = yuva + 1 (ad + KESME)", f"{P0.shape[1]} (model_b15 {P1.shape[1]})")
# !! model_06: IKI bicim TAMAMEN maskeli (cevap yuvasi olcumune
# girmiyor) -- devrik bildirim ve bosluk doldurma. Ikisinde de cevap
# oznesinden ONCE geliyor, yani soldan tahmin EDILEMEZ; olcseydik
# sahte dusuk sayi uretirdi. "Ilk yuva maskeli degil" iddiasi artik
# YALNIZ OLCULEN satirlar icin gecerli.
_tam0 = (T0 < 0).all(1)
ok(int(_tam0.sum()) > 0, "TAM MASKELI satirlar VAR (devrik + bosluk)",
   f"{int(_tam0.sum()):,}/{len(T0):,}")
ok(int((T0[~_tam0] < 0).any(1).sum()) > 0
   and int((T0[~_tam0][:, 0] < 0).sum()) == 0,
   "kisa adlarda MASKE var, ilk yuva HIC maskeli degil (OLCULEN satirlar)",
   f"maskeli satir {int((T0 < 0).any(1).sum()):,}/{len(T0):,}")
_kes = v.ek0
ok(all(int(t[int((t >= 0).sum()) - 1]) == _kes for t in T0[:500]),
   "her cevabin SON hedefi KESME ISARETI (sinir)")
# !! BOS YUVA -1 TASIYOR ve Python'da -1 SON ELEMANDIR.
# `par_ad[j][par[e,j]]` diye dogrudan indeksleyen alti arac vardi ve
# hepsi SESSIZCE sozlugun son kelimesini basiyordu ("Ibrahim Yilmaz"
# -> "Ibrahim | Yilmaz | Zonguldak", 17 Eylul). Tek kaynak
# `taban_08.kelimeler`; bu test onu bir daha kacirmamak icin.
ok(all(len(M.kelimeler(v, e)) == int(v.n_yuva[e]) for e in range(v.n_ent)),
   "kelimeler() dolgu URETMIYOR -- her ad kendi kelime sayisinda")
ok(not any("<YOK>" in w for e in range(v.n_ent) for w in M.kelimeler(v, e)),
   "hicbir adda <YOK> kalmadi")
_son = v.par_ad[0][-1]
ok(not any(M.kelimeler(v, e)[-1] == _son and v.n_yuva[e] < v.yuva
           for e in range(v.n_ent) if len(M.kelimeler(v, e)) < v.yuva
           and E05[e].split("_")[-1] != _son),
   f"kisa adlarin sonuna sozlugun SON kelimesi ('{_son}') sizmiyor")
ok(v.cevap_ara == (v.ent_off, v.ek0 + 1),
   "cevap araligi = VARLIK kelimeleri + kesme", str(v.cevap_ara))
ok(int(X0.max()) < v.vocab and int(X0.min()) >= 0,
   "havuzdaki her jeton SOZLUK ICINDE", f"max {int(X0.max())} < {v.vocab}")

# HAVUZ KUCULDU ve bu RAPORLANIR: `veri_08` daha kucuk bir graf
# (6254 olgu / 38157 zincir) ve ayni 20.000 adimda ayni batch ile
# EPOK SAYISI artiyor. Hukum vermez -- ama `seen` yukselirken `comp`
# durursa ilk bakilacak yer burasi.
_ep0 = A.batch * A.adim / X0.shape[0]
_ep1 = B15.batch * B15.adim / X1.shape[0]
# !! model_06'da havuz BUYUDU (model_05'te kuculmustu): bosluk
# doldurma her bildirim satirindan `fim_kat` varyant uretiyor.
# Epok sayisi DUSUYOR ve bu RAPORLANIR -- hukum vermez, ama `seen`
# yukselirken `comp` durursa ilk bakilacak yer burasi.
ok(X0.shape[0] > X1.shape[0],
   "havuz BUYUDU -- epok sayisi DUSTU, RAPORLANIR (hukum DEGIL)",
   f"{X0.shape[0]} dizi / ~{_ep0:.0f} epok   vs   "
   f"{X1.shape[0]} / ~{_ep1:.0f} epok")

# --- 5) STANDART TARIF -- DEVRALINMAYANLAR ------------------------------
print("\n=== 5) STANDART TARIF ===")
# !! BU KILIT GEVSETILMEDI, HEDEFI DEGISTI (CLAUDE.md'deki wd=0.5
# emsali). model_08..model_07'de `ort_bas == 0` bekleniyordu: "standart
# tarif, projeye ozgu ne varsa KAPALI". model_08'de geri beslemeli
# ortalama BU KOLUN DUGMESI (onkayit belge/onkayit/model_08.md), o
# yuzden kilit artik ACIK olmasini ve DOGRU DEGERDE olmasini bekliyor.
ok(A.ort_bas == 6000,
   "LOOKAHEAD ACIK -- BU KOLUN DUGMESI (6000'e kadar normal egitim)",
   f"b15 {B15.ort_bas} -> 08 {A.ort_bas}")
ok(A.ort_her == 2000 == A.olc_her,
   "ortalama esigi olcum noktasiyla AYNI -- her olcum bir esik",
   f"ort_her {A.ort_her}  olc_her {A.olc_her}")
ok(A.ort_alfa == 0.5,
   "alfa 0.5 -- 'bir onceki ile yari yariya' (kullanici, 18 Eylul)",
   str(A.ort_alfa))
ok(A.ort_bas % A.ort_her == 0 and A.isinma < A.ort_bas,
   "ort_bas esige OTURUYOR ve isinma BITMIS oluyor",
   f"isinma {A.isinma} < ort_bas {A.ort_bas}, {A.ort_bas} % {A.ort_her} = "
   f"{A.ort_bas % A.ort_her}")
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
    ok(hasattr(S, g), f"model_08.{g} var", "kos_08.py duser")
# !! `sor_08` LISTEDEN CIKTI (18 Eylul, kullanici dosyayi SILDI:
# *"sor py sacma olmustu"*). O arac soruyu kendi cozumluyor, diziyi
# `kodla_1hop` ile KENDI kuruyor ve modele yalniz cevap yuvalarini
# sorduruyordu -- yani ekranda gorulen sey modelin dil uretimi DEGILDI.
# Yerini `konus_08` aldi: kisit yok, model kendi yazdigini okur.
for ad in ("veri_08", "taban_08", "ayar_08", "pencere_08", "tani_08",
           "asama1_08", "kos_08", "konus_08"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True, cwd=_B)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import (yalniz model_08/)",
       _r2.stderr.strip()[-200:])

# --- 8) KOPYALAMADA KAYBOLAN DUZELTMELER ---------------------------------
# !! BU BOLUM BIR KAZADAN DOGDU. `model_07/` klasoru `model_06/`dan
# KOPYALANDI ve kopya, duzeltmeleri tasiyan commit'ten ONCEYDI. Uc sey
# sessizce geri gitti:
#   konus.bat     hala `konus_05.py` cagiriyordu (kullanici IKI KEZ
#                 carpti; klasorden turetilen surumle degistirildi)
#   --genislik    5 -> 1'e dondu. Olculdu (model_06, 20.000): tek anlik
#                 goruntu one 0.7530, pencere 0.9237. Yani arac
#                 OLCULENDEN BASKA bir modeli gosteriyordu.
#   bosluk cevabi tek jeton yerine hepsi basiliyordu; modelin hatasini
#                 OLDUGUNDAN BUYUK gosteriyordu.
# Ucunu de INSAN yakaladi, kod DEGIL. Burasi artik kodun yakalamasi icin.
# --- 7b) IYELIK KISA YOLU: TUTULAN VERIYE DOKUNUYOR MU ---------------
# !! BU BOLUM BIR KUSURDAN DOGDU (18 Eylul). Kisa yol satirlari
# ("Ayse Sahin'in dekani kimdir?") EGITIM verisidir ve COK ADIMLI bir
# gercegi 1-hop yuzeyle soyler. Eklendiklerinde TEK bir kapi konmustu
# -- "bu satir bir SINAV ZINCIRININ cevabi mi". Iki kapi daha
# gerekiyordu; ikisi de OLCULDU, ikisi de OLMUSTU:
#
#   TUTULAN BAS   `ent` tanimi "hic zincir basi olmamis"tir. 120 ent
#                 varliginin 73'u kisayolda BAS konumundaydi.
#   OOD KENARI    atomic_OOD kenarlari egitimde YOK. 1730 satirin
#                 67'si yolunu o kenarlardan geciriyordu.
#
# !! O KUSUR BU DOSYANIN 175 DENETIMINDEN GECMISTI. Ucu de artik
# `taban_08.veri_kur` icinde assert'le kapali; burasi BAGIMSIZ olarak
# yeniden turetiyor -- kapi ile denetim AYNI KODDAN gelmesin.
print("\n=== 7b) KISA YOL TUTULAN VERIYE DOKUNUYOR MU ===")
_ky = {r: y for r, y in M.KISAYOL_YOLU}
_rid = {r: i for i, r in enumerate(V00.ILISKI)}
_tut = {x[0] for L in (v.ent, v.ent_yok, v.ent_arama, v.ent_kati) for x in L}
_ood = set()
for _e, _r1, _r2, _b, _a in v.ood:
    _ood.add((_e, _r1)); _ood.add((_b, _r2))
_snv = {(x[0], x[1], x[2]) for L in
        (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood, v.ent_kati) for x in L}
_b1 = [e for e, r, a in v.kisayol if e in _tut]
_b2, _b3 = [], []
for _e, _r, _a in v.kisayol:
    _y = _ky[V00.ILISKI[_r]]
    if len(_y) == 2 and (_e, _rid[_y[0]], _rid[_y[1]]) in _snv:
        _b3.append(_e)
    _c = _e
    for _rr in _y:
        if (_c, _rid[_rr]) in _ood:
            _b2.append(_e); break
        _c = int(v.facts[_c, _rid[_rr]])
        if _c < 0:
            break
ok(not _b1, "kisa yol TUTULAN varligi ZINCIR BASI yapmiyor", f"{len(_b1)} satir")
ok(not _b2, "kisa yol OOD kenarindan GECMIYOR", f"{len(_b2)} satir")
ok(not _b3, "kisa yol SINAV ZINCIRININ cevabini soylemiyor", f"{len(_b3)} satir")
ok(len(v.kisayol) > 0, "kisa yol satiri URETILIYOR (kapilar hepsini yemedi)",
   f"{len(v.kisayol)} satir, atilan {dict(v.kisayol_atilan)}")

# --- 7c) DEFTER KAPISI -----------------------------------------------
# !! BU BOLUM DE BIR KUSURDAN DOGDU (18 Eylul). Veri gun icinde uc kez
# degisti ve her seferinde `olcme_izi` yenilendi; ama defterdeki
# `IZ_08` ELLE YAZILI bir sabit ve kimse onu denetlemiyordu. Defter
# eski izle kaldi -- 7. hucredeki `assert _iz == IZ_08` KOSUYU
# ORTASINDA DUSURECEKTI, ve bu ancak GPU saati harcandiktan sonra
# gorulecekti.
#
# Ayni hucredeki `KIYAS` sozlugu de BASKA BIR SINAVIN sayilarini
# tasiyordu (model_05/06/07, iz f4ce53fd1555). `model_b8` kusuru:
# baska sinavin sayisini ayni tabloya koymak.
print("\n=== 7c) DEFTER KAPISI ===")
import json as _json, io as _io, re as _re               # noqa: E402
_nbY = os.path.join(_B, os.path.basename(_B) + ".ipynb")
if not os.path.exists(_nbY):
    print("    --   defter YOK -- atlandi")
else:
    _nb = _json.load(_io.open(_nbY, encoding="utf-8"))
    _src = "".join("".join(c["source"]) for c in _nb["cells"])
    _iz_ger = M.olcme_izi(M.olcme_listeleri(A, v))
    _m = _re.search(r'IZ_08\s*=\s*"([0-9a-f]+)"', _src)
    ok(_m is not None, "defterde IZ_08 tanimli")
    if _m:
        ok(_m.group(1) == _iz_ger,
           "defterdeki IZ_08 = GERCEK olcme izi",
           f"defter {_m.group(1)}  veri {_iz_ger}")
    # KIYAS: ya BOS ya da SATIRLARI bu sinavdan. Bos degilse EN AZINDAN
    # eski izleri ANMAMALI.
    _eski_iz = ("f4ce53fd1555", "3431c633b6b1", "d6751004648c",
                "c69b6b028380", "57717784a417", "b5f24205744d")
    _k = _re.search(r"KIYAS = \{(.*?)^\}", _src, _re.S | _re.M)
    ok(_k is not None, "defterde KIYAS sozlugu var")
    if _k:
        _kotu = [z for z in _eski_iz if z in _k.group(1)]
        ok(not _kotu, "KIYAS sozlugu ESKI SINAV izi anmiyor", str(_kotu))
    ok(V00.IZ in _src, "defter GUNCEL graf izini aniyor", V00.IZ)

    # !! DEFTERIN KENDI AYAR KILITLERI. 3. hucre `assert M.AYAR.X == Y`
    # diye onlarca kilit tasiyor ve bunlar `ayar_08.py`nin KOPYASI --
    # yani ayar degisince SESSIZCE eskiyor.
    #
    # OLDU (18 Eylul): kolun dugmesi `ort_bas` 0 -> 6000 oldu, `test_08`
    # kilidi yeniden hedeflendi, ama DEFTERDEKI kopya `== 0` kaldi.
    # Defter 3. hucrede DUSECEKTI -- GPU ayrildiktan SONRA.
    _kilit = _re.findall(r"assert M{0}.AYAR{0}.({1}w+) == ([^,{0}n]+),".format(
        "\\", "\\"), _src)
    _kot = []
    for _al, _bek in _kilit:
        if not hasattr(A, _al):
            _kot.append((_al, "ALAN YOK")); continue
        _bek = _bek.strip()
        try:
            _d = eval(_bek, {"M": type("x", (), {"AYAR": A})})
        except Exception:
            continue                       # zincirli/karmasik ifade -- atla
        if isinstance(_d, bool) and not isinstance(getattr(A, _al), bool):
            continue                       # `a == b == c` zinciri
        if _d != getattr(A, _al):
            _kot.append((_al, f"defter {_bek} != ayar {getattr(A, _al)}"))
    ok(not _kot, f"defterin {len(_kilit)} AYAR kilidi ayar_08 ile TUTUYOR",
       str(_kot))
    ok(len(_kilit) >= 20,
       "defter 3. hucresi AYAR kilitlerini hala tasiyor (silinmemis)",
       f"{len(_kilit)} kilit")

# --- 7d) NULL TABANI -------------------------------------------------
# Statik denetimler "veri KENDI ICINDE tutarli mi" diye sorar. Bu bolum
# BASKA bir sey sorar: bir bolme APTAL bir stratejiyle cozulebiliyor mu?
# Cozulebiliyorsa modelin oradaki basarisi bilesim DEGIL, hiledir.
# (Kullanici, 18 Eylul: "oncul test yaparak da dogrula".)
#
# Esik %5. `one` ve `seen` HARIC: `one`da KISAYOL 1.0000 cikar ve bu
# DOGRUDUR (1-hop sorusunda facts[e,r] cevabin kendisidir), `seen`de de
# DONUS zincirleri KOPYA ile cozulur -- ikisi de saglik kapisi, hukum
# bolmesi degil.
print("\n=== 7d) NULL TABANI ===")
import null_08 as _NL                                       # noqa: E402
_nl = _NL.null("08", yaz=lambda *a, **k: None, v=v)
for _b, _x in sorted(_nl.items()):
    ok(_x <= 0.05, f"{_b}: aptal strateji %5'in ALTINDA", f"{_x:.4f}")
ok(set(_nl) >= {"comp", "ent", "ent_yok"},
   "null tabani HUKUM bolmelerinin hepsini olcuyor", str(sorted(_nl)))

# --- 7e) KOL NUMARASI KAPISI -----------------------------------------
# !! BU BOLUM BIR KUSURDAN DOGDU (18 Eylul). Kullanici test CIKTISINDA
# `model_05.AYAR var`, `veri_05 KENDI IDDIALARINI tutuyor mu` gibi
# satirlar gordu ve sordu: *"bu niye var model 05?"*
#
# SEBEP YAPISALDI: her kopyada yeniden adlandirma BIR NESIL geriye
# bakiyordu (`_07` -> `_08`). `_05`te kalmis atiflar hicbir nesilde
# CEVRILMEDI -- model_06'dan beri yanlistilar. Assert'ler DOGRU modulu
# kullaniyordu (testler GECIYORDU), ama her MESAJ hangi kol oldugunu
# YANLIS soyluyordu. 137 satir.
#
# Daha kotusu: toptan yeniden adlandirma eskimis bir IDDIAYI da cevirdi
# -- "`f4ce53fd1555` model_05 ile AYNI" satiri "model_08 ile AYNI" oldu,
# ki model_08'in izi cfafdcc15a23. Kor yeniden adlandirma eskimis bir
# cumleyi KENDINDEN EMIN bir yalana cevirir.
#
# !! KAPI DAR TUTULDU: yalniz EKRANA BASILAN metin (ok(...) aciklamasi
# ve baslik) taranir. Yorumlardaki gecmise atif (`havuz_06 yakaladi`,
# `veri_04'te`) MESRU ve cok -- onlari taramak kapiyi gurultuye bogar
# ve gurultulu kapi kapatilir.
print("\n=== 7e) KOL NUMARASI KAPISI ===")
import glob as _glob, io as _io2, re as _re2                 # noqa: E402
_KOL = "08"
# EKRANA BASILAN metinde BASKA kol numarasi. Gecmise atif ACIKCA yazilir.
_GECMIS_MSJ = ("model_b15", "model_05'ten KUCUK", "model_a ")
_kot = []
for _f in sorted(_glob.glob(os.path.join(_B, "test_*.py"))):
    for _i, _l in enumerate(_io2.open(_f, encoding="utf-8"), 1):
        _t = _l.strip()
        if _t.startswith("#"):
            continue
        for _q in _re2.findall(r'"([^"]*)"', _l) + _re2.findall(r"'([^']*)'", _l):
            _m = _re2.search(r"(?:model|veri|taban|ayar|asama1|kos|analiz|"
                             r"pencere|sor|tani|konus|havuz|graf|test)_(0[0-9])",
                             _q)
            if _m and _m.group(1) != _KOL and not any(
                    _g in _q for _g in _GECMIS_MSJ):
                _kot.append(f"{os.path.basename(_f)}:{_i} {_q[:60]}")
                break
ok(not _kot,
   "test MESAJLARI BASKA bir kol numarasi ANMIYOR",
   f"{len(_kot)} mesaj: {_kot[:3]}")

print("\n=== 8) KOPYALAMADA KAYBOLAN DUZELTMELER ===")
import inspect, io                                          # noqa: E402
import konus_08 as _K                                       # noqa: E402

# !! BU IKI KILIT GEVSETILMEDI, HEDEFI DEGISTI (18 Eylul).
# CLAUDE.md'deki `wd 0.5` ve `ort_bas 6000` emsali: kilit bir SAYIYI
# degil bir KURALI koruyor -- *arac, HUKMUN DAYANDIGI modeli
# gostermeli*. Kural aynen duruyor, bu kolda hukum baska okumaya
# dayaniyor.
#
# Once 5 bekleniyordu, cunku `model_06`da pencere tek anlik goruntuden
# COK iyiydi (one 0.7530 -> 0.9237) ve varsayilan 1 kalirsa arac
# olculenden KOTU bir modeli gosteriyordu. O duzeltme kopyalamada IKI
# KEZ kayboldu, kilit onun icin konuldu.
#
# model_08 GERI BESLEMELI: agirlik zaten egitim icinde her 2000 adimda
# ortalaniyor ve ONKAYIT §3 birincil okumayi EGRI yaziyor. Kilidin
# hedefi bu yuzden 1 oldu -- ONCEDEN YAZILMIS okuma kurali.
#
# !! GEREKCE BIR KEZ YANLIS YAZILDI (ayni gun). "Bu kolda pencere ZARAR
# veriyor" denmisti; dayanagi 20.000 olcumuydu ve 40.000'de DORT
# olcude de isaret dondu:
#     adim   comp EGRI / PENCERE        one EGRI / PENCERE
#     20000    0.6338 / 0.5774            0.9210 / 0.9030
#     40000    0.7958 / 0.8100            0.9543 / 0.9703
# Pencere, egri DIKKEN geri ceker; egri DUZLESINCE yardim eder. Yani
# "hangisi iyi" ADIMA BAGLI ve varsayilan ona dayandirilamaz.
# AYAKTA KALAN: bu kolda fark KUCUK (|0.02| civari); model_07'de ayni
# fark +0.18'e cikiyor. Ayrinti konus_08.kur docstring'inde.
_sig = inspect.signature(_K.kur).parameters["genislik"].default
ok(_sig == 1, "konus_08.kur VARSAYILANI 1 (ORTALAMA YOK -- egri okumasi)",
   f"genislik={_sig} -- 5 ise arac OLCULENDEN BASKA modeli gosterir")

_ks = io.open(os.path.join(_B, "konus_08.py"), encoding="utf-8").read()
ok('ap.add_argument("--genislik", type=int, default=1' in _ks,
   "konus_08 --genislik VARSAYILANI da 1",
   "argparse varsayilani `kur`dan AYRI; ikisi ayrisirsa CLI kazanir")

# !! TURKCE CIKTI KAPISI (18 Eylul). `konus.bat` chcp 65001 +
# PYTHONIOENCODING yaziyordu, yani CIFT TIKLAYINCA dogru basiyordu ama
# terminalden/IDE'den `Elif Ayd?n'd?r` cikiyordu. Bu arac "gozlerimle
# gormek istiyorum" diye istendi; okunmayan cikti o isi gormez, ve bozuk
# metin bir arizayi GIZLEYEBILIR (ayni ders §9'da alinmisti).
ok("reconfigure(encoding=\"utf-8\"" in _ks,
   "konus_08 stdout'u KENDI UTF-8'e ceviriyor",
   "kodlama BASLATICIYA birakilmis -- konus.bat tek giris noktasi DEGIL")
ok("_c = _p[:1]" in _ks and "model DURMADI" in _ks,
   "bosluk cevabi TEK JETON basiliyor",
   "egitimde bosluk 187.296/187.296 kez tek jetonluk")

# --- SINAV GOZU (18 Eylul) ----------------------------------------------
# !! BIR KUSURDAN DOGDU. Kullanici sinav dosyasindan satir kopyalayip
# kisaltarak test etti ("Elif Çelik'in danışmanı"), model 2-hop'a uzatti,
# ve "konus'un arizasi mi?" diye sordu. Ariza degildi -- olculdu (300
# olgu): %81,3 2-hop'a uzatiyor, urettigi cumle %92,7 DOGRU. Model
# olguyu BILIYOR, baska cumle kurmayi SECIYOR. Ama arac bunu
# soylemiyordu: "bilmiyor" ile "baska soruyu cevapladi" ekranda AYNI
# goruntuyu veriyordu, ve cevap dogru ciktiginda EZBER mi GENELLEME mi
# oldugu da gorunmuyordu.
#
# Kapi SUBSTRING DEGIL, DAVRANIS siniyor: `sinav_coz` neyi kabul edip
# neyi reddettigi. Substring kapisi yeniden yazimda sessizce gecerdi.
_sc = _K.sinav_coz
_1hop = (M._e(v, 0) + [M._kesme(v), M._nin(v, e=0), M.REL_OFF + 0])
_2hop = (M._e(v, 0) + [M._kesme(v), M._nin(v, e=0), M.REL_OFF + 0,
                       M._nin(v, r=0), M.REL_OFF + 1])
ok(_sc(v, _1hop) == (tuple(M._e(v, 0)), [0]),
   "sinav_coz 1-hop sorusunu TANIYOR", f"{_sc(v, _1hop)}")
ok(_sc(v, _2hop) == (tuple(M._e(v, 0)), [0, 1]),
   "sinav_coz 2-hop sorusunu TANIYOR", f"{_sc(v, _2hop)}")
# SORU SOZCUGU eklenmisse artik "sinav satiri" degil -> SUSMALI.
ok(_sc(v, _1hop + [v.soru0]) is None,
   "sinav_coz soru sozcugu gelince SUSUYOR",
   "yoksa yarim cumleye sinav notu basar")
# CEVAP da yazilmissa sorulacak bir sey kalmamis -> SUSMALI.
ok(_sc(v, _1hop + M._e(v, 1)) is None,
   "sinav_coz cevap yazilmissa SUSUYOR",
   "kullanici tam cumle yazdiysa sinav gozu devreye GIRMEMELI")
# !! DESEN `_sb = sinav_bakisi(` -- CAGRI, tanim DEGIL. Ilk surum
# "sinav_bakisi(v, net, D, jet, BOL, E2)" ariyordu ve o dize dosyada
# IKI KEZ geciyor (def satiri + cagri): cagri silinse bile kapi
# GECIYORDU. Mutasyonla yakalandi, 18 Eylul. Bir kapinin arayacagi dize
# TEK yerde gecmeli, yoksa kapi degil dekordur.
ok("_sb = sinav_bakisi(" in _ks,
   "sinav gozu KONUSMA DONGUSUNDE cagriliyor",
   "fonksiyon var ama kullanilmiyorsa ekranda hicbir sey degismez")

_bat = io.open(os.path.join(_B, "konus.bat"), encoding="utf-8",
               errors="replace").read()
# !! YORUMA DEGIL KOMUTA bak: bat'in gecmis notunda "konus_05.py"
# gecmesi normaldir, o bir ACIKLAMA. Aranan sey, `python` satirinin
# betik adini ELLE yazip yazmadigi.
_cagri = [x.strip() for x in _bat.splitlines()
          if x.strip().lower().startswith("python")]
ok("KLASOR:~6" in _bat, "konus.bat betik adini KLASORDEN turetiyor",
   "elle yazilan ad, klasor kopyalaninca KAYIYOR")
ok(_cagri and all("konus_0" not in c for c in _cagri),
   "konus.bat CAGRISI elle yazilmis bir ad TASIMIYOR", str(_cagri))

# --- 9) KOPYA KAPISI -----------------------------------------------------
# 8. bolum UC BILINEN gerilemeyi kilitliyor. Bu bolum GENEL olani sorar:
# ebeveyn klasorde (model_07) OLUP burada OLMAYAN her satir ya TASINMIS
# ya da KOPYA.json'da BEYAN EDILMIS mi?
#
# Kapinin kendisi `deneme2/kopya_kapisi.py` -- kol DISINDA, cunku iki
# klasore birden bakiyor. Buradan ALT SUREC olarak cagriliyor; boylece
# `0) BAGIMSIZLIK` taramasi bozulmuyor (o tarama test_*.py'yi zaten
# disarida birakiyor).
print("\n=== 9) KOPYA KAPISI ===")
_kk = os.path.join(os.path.dirname(_B), "kopya_kapisi.py")
if not os.path.exists(_kk):
    print("    --   kopya_kapisi.py YOK -- atlandi")
else:
    # !! encoding="utf-8" ACIKCA VERILIYOR. `text=True` tek basina YEREL
    # kodlamayi (cp1254) kullaniyordu ve kapinin Turkce mesajlari bozuk
    # basiliyordu ("§" -> "Â§"). Bozuk metin bir hatayi GIZLEYEBILIR.
    _r3 = subprocess.run([sys.executable, _kk, os.path.basename(_B)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=os.path.dirname(_B))
    _ozet = [x for x in _r3.stdout.splitlines()
             if x.strip().startswith(("!!", "GECTI", "KAPI"))
             or "GECTI --" in x]
    for x in _ozet[:12]:
        print("   ", x.strip()[:110])
    ok(_r3.returncode == 0,
       "ebeveynde OLUP burada OLMAYAN her satir TASINMIS ya da BEYANLI",
       (_r3.stdout + _r3.stderr).strip()[-600:])

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
