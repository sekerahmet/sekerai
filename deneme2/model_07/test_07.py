# -*- coding: utf-8 -*-
"""model_07 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_05 folderi
altinda olmali. model_05 diger hicbir model ile ayni seyi
kullanmamali."*

Dort sey tutulur:

  1. BAGIMSIZLIK  model_05/*.py icinde `model_a` / `model_b` /
     `veri_okul*` / `pencere_a` import'u YOK. Statik taramayla.
  2. MIMARI       projenin hicbir ozelligi yok: dongu yok, Phi
     darbogazi yok, maske yok, yardimci kayip yok, miras yok.
  3. VERI         `veri_05` KENDI iddialarini tutuyor: sema, soy
                  agaci (cinsiyet/kusak/ensest), zincir siniflari,
                  turetilebilirlik, notrluk, cografya, IZ.
  4. MOTOR        `taban_05` ile `model_a` AYNI SEYI olcuyor:
     egitim havuzu bit duzeyinde ayni, olcme izi ayni.

--------------------------------------------------------------------------
!! KILIT, PAYLASILAN MODULLERI IMPORT EDER -- VE ETMELIDIR

3 ve 4, model_05'in kopyalarini ORIJINALLERIYLE karsilastiriyor; bunun
icin `model_a` / `veri_okul4` / `model_b15` buraya yukleniyor. Bu,
1'deki bagimsizlik iddiasini BOZMAZ: KOSAN kol onlari gormez, yalniz
KILIT gorur. Kilidin isi zaten bu -- kopyanin sapip sapmadigini
soylemek.

Kopyanin GERCEK riski buydu: paylasilan motorda bir olcum hatasi
duzeltilirse buraya kendiliginden GELMEZ, ve model_05 ile model_b15 o
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
print("=== 0) BAGIMSIZLIK: model_07/ disariya BAGLI MI ===")
# Metinde arama YAPMIYORUZ -- ilk denemem oyleydi ve `ayar_05.py`nin
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

print("\n=== 1) model_05 MIMARISI ===")
import taban_07 as M                                         # noqa: E402
import model_07 as S                                         # noqa: E402

A = S.AYAR
ok(S.M is M, "model_05 motoru taban_05", S.M.__name__)
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64, "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, taban_05.Model'den MIRAS ALMIYOR")
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
# KUSUR (16 Eylul hakemligi): `asama1_05.gizli()` ileri gecisi ELLE
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
import asama1_07 as A1                                      # noqa: E402
_lst = v.comp[:8]
# !! MODELI M.DEV'E TASI. `gizli()` girdiyi `.to(M.DEV)` ediyor; model
# CPU'da kalirsa GPU'lu makinede "index is on cuda:0, other tensors on
# cpu" ile duser. ILK YAZIMDA BU EKSIKTI ve yerelde (DEV="cpu") FARK
# EDILMEDI -- kusur Colab'da, kilidin ilk GPU kosusunda cikti.
# Gercek kullanimda `asama1_05.main` modeli zaten `.to(M.DEV)` ediyor;
# yani hata ARACIN degil, BU TESTIN hatasiydi. Sonra CPU'ya geri
# aliniyor: asagidaki bolumler CPU tensorleriyle devam ediyor.
try:
    net.to(M.DEV)
    _q = A1.gizli(net, v, _lst, 6)
    ok(_q.shape == (8, A.d), "asama1_05.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
       f"{_q.shape} cihaz {M.DEV}")
except Exception as _e:                                      # noqa: BLE001
    ok(False, "asama1_05.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
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

# --- 3) VERI: veri_05'in KENDI IDDIALARI --------------------------------
# !! model_03'ten DEVRALINAN "kopya == orijinal" TESTI KALDIRILDI.
# Orada `veri_03` bir KOPYAYDI (icerigi `veri_okul4` ile birebir) ve test
# kopyanin kaymadigini siniyordu. `veri_05` KOPYA DEGIL: yeni sema, yeni
# tipler, gercek soy agaci, gercek cografya. Kiyaslanacak bir ORIJINAL
# YOK. O yuzden test artik verinin KENDI IDDIALARINI siniyor -- ve o
# iddialarin cogu `veri_04`te TUTMUYORDU ("Fatma'nin annesi Huseyin").
print("\n=== 3) VERI: veri_05 KENDI IDDIALARINI tutuyor mu ===")
import veri_07 as V00                                        # noqa: E402

ok(A.veri_ad == "veri_07", "model_05 KENDI veri modulunu okur", A.veri_ad)
_G0 = V00.kur(0)
E05 = [x for t in V00.TIPLER for x in _G0["ad"][t]]
_z = V00.zincirler(_G0)
ok(V00.graf_izi(_G0) == V00.IZ, "graf izi veri_05.IZ ile TUTUYOR", V00.IZ)
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

# (3) `jeton_ad` EK JETONLARINDA cokmuyor. veri_dok/analiz_05 `ek_kip`
#     gelen her kolda IndexError veriyordu -- model_b15'in verisi bu
#     yuzden hic dokulmemisti.
sys.path.insert(0, _B)
import analiz_07 as AZ                                       # noqa: E402
_d = AZ.Dok("model_07")
_ekler = [_d.jeton_ad(i) for i in range(_d.v.ek0, _d.v.vocab)]
# ek_kip="tr2": <SI> YOK (iliski kendi iyelik ekini tasiyor),
# yerine SORU SOZCUKLERI var. Kullanici karari, 17 Eylul.
# ek_kip="tr2" ek blogu:  '  + tamlayan allomorflari (8) +
# bildirme allomorflari (8) + soru sozcukleri.  Hicbiri <SOYUT>
# isaretleyici DEGIL -- hepsi GERCEK Turkce ek ya da kelime.
# model_07 UC JETON EKLEDI, hepsi sozlugun SONUNDA (REL_OFF ve ent_off
# KAYMASIN diye):
#   <BOS>   bosluk doldurmada cikarilan parcanin yeri
#   <AYIR>  "cumle bitti, simdi o parca geliyor"
#   ,       devrik cumlede OGE SINIRI -- onsuz "Kardesidir Ozlem Yilmaz
#           Ibrahim Yilmaz'in"da iki ad yan yana geliyor ve cevabin
#           nerede bittigi BELIRSIZ kaliyordu (havuz_06 yakaladi).
_bek = (["'"] + list(V00.EK_NIN) + list(V00.EK_DIR)
        + sorted(set(V00.SORU_SOZ.values()))
        + ["<BOS>", "<AYIR>", ","])
ok(_ekler == _bek, "analiz_07.jeton_ad EK JETONLARINI adlandiriyor",
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
    ok(hasattr(V00, _g), f"veri_05.{_g} var (sozlesme)")

# `model_a` / `model_b15` YALNIZ BU KILIT icin yukleniyor -- bu bolum
# motorun eski aileyle AYNI SEYI olctugunu siniyor, kosuda kullanilmaz.
# (Yol eklemeleri eskiden 3. bolumun icindeydi; o bolum `veri_okul4`
# kiyasiyla birlikte kaldirilinca buraya tasindi.)
sys.path.insert(0, _K)
sys.path.insert(0, os.path.join(_K, 'model_a'))
sys.path.insert(0, os.path.join(_K, 'model_b'))

# --- 4) MOTOR: taban_05 ile model_a AYNI SEYI mi olcuyor ----------------
print("\n=== 4) MOTOR ESDEGERLIGI: taban_05 vs model_a ===")
import model_a as MA                                         # noqa: E402
from model_b15 import AYAR as B15                            # noqa: E402
from ayar_07 import GOREV_ALAN                               # noqa: E402

# BILDIRILMIS AYRISMA -- IKI YENI ALAN, ve ikisi de KAPALI GELIYOR:
#   fim_kat   bosluk doldurma varyant sayisi   (model_06)
#   soru_kat  soru satiri sayisi               (model_07)
# ESKI_VARSAYILAN'a ikisi de 0 diye girildi, cunku alanlar eklenmeden
# ONCE kodun FIILEN yaptigi sey buydu: boyle bir satir tipi yoktu.
# Yani eski kosular SESSIZCE yanlis etiketlenmiyor.
_YENI_ALAN = {"fim_kat", "soru_kat"}
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
   "Ayar alanlari model_a ile AYNI, YALNIZ fim_kat + soru_kat eklendi",
   f"fazla {sorted(set(_alan06) - set(_alanA))}")
ok(not (set(_alanA) - set(_alan06)), "model_a'nin hicbir alani SILINMEDI",
   f"eksik {sorted(set(_alanA) - set(_alan06))}")
# !! `SPECIAL` BILEREK AYRILDI (17 Eylul). model_a'da 8, burada 3.
# Olculdu: sekiz ozel jetonun BESI havuzda HIC gecmiyordu --
#   [S1] [S2] [KIMLIK] ve iki adlandirilmamis yuva.
# Bunlari silmek `ek_kip=""` ve `ek_kip="tr"` kodlama yollarinin da
# silinmesini gerektirdi (Q1/Q2/IDENT yalniz onlar icindi), yani
# `taban_05` artik ortak motorun BIREBIR KOPYASI DEGIL.
# Kullanici karari: "silinsin tabii ki".
#
# Kopya-sapma bekcisi BU YUZDEN KALDIRILMADI, YON DEGISTIRDI:
# degismesi GEREKEN ayrildi mi, degismemesi gereken AYNI mi.
ok(M.T_LEN == MA.T_LEN, "T_LEN model_a ile AYNI (eksiz taban)", str(M.T_LEN))
ok(M.SPECIAL == 3 and MA.SPECIAL == 8,
   "SPECIAL model_a'dan BILEREK ayrildi (olu jetonlar silindi)",
   f"model_a {MA.SPECIAL} -> model_05 {M.SPECIAL}")
ok(not hasattr(M, "IDENT") and not hasattr(M, "Q1"),
   "Q1 / Q2 / IDENT sabitleri SILINDI (kodlama yollariyla birlikte)")
ok(M.REL_OFF == M.SPECIAL, "REL_OFF hala SPECIAL'a bagli", str(M.REL_OFF))
# Ayar VARSAYILANLARI: yalniz BILEREK degistirilen ikisi farkli olmali.
# Ucuncusu cikarsa kopya sapmis demektir. (Varsayilanlar yeniden kurulusta
# KULLANILMIYOR -- `ayar_oku` eksik alanlari ESKI_VARSAYILAN'dan
# dolduruyor -- ama sapma yine de GORULSUN.)
import dataclasses as _dcc                                   # noqa: E402
# !! YALNIZ IKISINDE DE OLAN alanlar kiyaslanir. `fim_kat` model_07'nin
# YENI alani; `getattr(MA.Ayar(), "fim_kat")` AttributeError verirdi.
_ortak = {f.name for f in _dcc.fields(MA.Ayar)}
_vf = sorted(f.name for f in _dcc.fields(M.Ayar)
             if f.name in _ortak
             and getattr(M.Ayar(), f.name) != getattr(MA.Ayar(), f.name))
ok(_vf == ["ad", "veri_ad"],
   "ORTAK alanlarin varsayilanlari yalniz ad + veri_ad'da farkli", str(_vf))
ok(M.Ayar().fim_kat == 0,
   "fim_kat VARSAYILANI 0 -- bosluk doldurma KAPALI gelir",
   "acan sey ayar_07, yani BU KOLUN karari")

# BILDIRILMIS AYRISMA. `GOREV_ALAN` "sinavin AYNI kalmasi GEREKEN
# alanlari" listesi ve model_03'e kadar hepsi model_b15 ile birebirdi.
# model_05 ikisini BILEREK degistiriyor ve ikisi de BU KOLUN TANIMI:
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
ok(_fark == ["ad", "betas", "dar_alfa", "dar_kapi", "dff", "dongu",
             "ek_kip", "l", "ort_bas", "sabit_lr", "veri_ad", "wd"],
   f"model_05 <-> model_b15 farki {_fark}",
   "MIMARI + veri ADI + BU KOLUN RECETESI (wd, sabit_lr)")

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
# `veri_05` bu kolun DUGMESI: yeni sema, yeni tipler, gercek soy agaci.
# Sinavin ve havuzun degismesi bu kolun ARIZASI DEGIL, TANIMI. CLAUDE.md:
#   "olcme izi degisirse -> KOL IPTAL DEGIL. Not duser, devam edilir."
#
# Ama "degisti" demek yetmez, YONU sinanir: DEGISMESI GEREKENLER degisti,
# DEGISMEMESI GEREKENLER (dizi uzunlugu, yuva sayisi, kodlama sekli)
# AYNI kaldi. Yoksa sessiz bir kodlama hatasi "veri degisti zaten" diye
# gecerdi.
ok(_iz05 != _iz15,
   "OLCME IZI model_b15'ten AYRILDI -- BEKLENEN (veri bu kolun dugmesi)",
   f"model_05 {_iz05}  vs  model_b15 {_iz15}")
ok(X0.shape != X1.shape,
   "EGITIM HAVUZU da ayrildi -- BEKLENEN", f"{X0.shape} vs {X1.shape}")
# t_len de DEGISTI (17 -> 16) ve bu da BILEREK: <SI> cikti (-2, iki
# iliski icin), soru sozcugu girdi (+1). Turetim `Ayar.t_len`de yazili:
#   2-hop bicim 0 = e(3) ' <NIN> r1 <NIN> r2 SORU ? a(3) ' <DIR> EOS
#                 = 2*yuva + 10 = 16
# Sinanan sey artik "ayni kaldi" DEGIL, "TURETIMLE TUTUYOR".
# t_len = 3*yuva + 15: soru (yuva+7) + TAM CUMLE cevap (2*yuva+8).
# Kullanici karari 17 Eylul: cevap soruyu YENIDEN YAZIP oyle cevapliyor.
# model_07: SORU BICIMI YOK (kullanici, 17 Eylul: "yalniz bildirim
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
# !! model_07: IKI bicim TAMAMEN maskeli (cevap yuvasi olcumune
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
# `taban_05.kelimeler`; bu test onu bir daha kacirmamak icin.
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

# HAVUZ KUCULDU ve bu RAPORLANIR: `veri_05` daha kucuk bir graf
# (6254 olgu / 38157 zincir) ve ayni 20.000 adimda ayni batch ile
# EPOK SAYISI artiyor. Hukum vermez -- ama `seen` yukselirken `comp`
# durursa ilk bakilacak yer burasi.
_ep0 = A.batch * A.adim / X0.shape[0]
_ep1 = B15.batch * B15.adim / X1.shape[0]
# !! model_07'da havuz BUYUDU (model_05'te kuculmustu): bosluk
# doldurma her bildirim satirindan `fim_kat` varyant uretiyor.
# Epok sayisi DUSUYOR ve bu RAPORLANIR -- hukum vermez, ama `seen`
# yukselirken `comp` durursa ilk bakilacak yer burasi.
ok(X0.shape[0] > X1.shape[0],
   "havuz BUYUDU -- epok sayisi DUSTU, RAPORLANIR (hukum DEGIL)",
   f"{X0.shape[0]} dizi / ~{_ep0:.0f} epok   vs   "
   f"{X1.shape[0]} / ~{_ep1:.0f} epok")

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
    ok(hasattr(S, g), f"model_05.{g} var", "kos_05.py duser")
for ad in ("veri_07", "taban_07", "ayar_07", "pencere_07", "tani_07",
           "asama1_07", "sor_07", "kos_07", "konus_07"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True, cwd=_B)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import (yalniz model_05/)",
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
print("\n=== 8) KOPYALAMADA KAYBOLAN DUZELTMELER ===")
import inspect, io                                          # noqa: E402
import konus_07 as _K                                       # noqa: E402

_sig = inspect.signature(_K.kur).parameters["genislik"].default
ok(_sig == 5, "konus_07.kur VARSAYILANI 5 (pencere ortalamasi)",
   f"genislik={_sig} -- 1 ise arac OLCULENDEN BASKA modeli gosterir")

_ks = io.open(os.path.join(_B, "konus_07.py"), encoding="utf-8").read()
ok('ap.add_argument("--genislik", type=int, default=5' in _ks,
   "konus_07 --genislik VARSAYILANI da 5",
   "argparse varsayilani `kur`dan AYRI; ikisi ayrisirsa CLI kazanir")
ok("_c = _p[:1]" in _ks and "model DURMADI" in _ks,
   "bosluk cevabi TEK JETON basiliyor",
   "egitimde bosluk 187.296/187.296 kez tek jetonluk")

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

print(f"\n{_gecti} gecti, {_bozuk} BOZUK")
sys.exit(1 if _bozuk else 0)
