# -*- coding: utf-8 -*-
"""model_10 KILIDI -- "her seyden bagimsiz" iddiasi SINANIR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_09 folderi
altinda olmali. model_09 diger hicbir model ile ayni seyi
kullanmamali."*

Dort sey tutulur:

  1. BAGIMSIZLIK  model_10/*.py icinde `model_a` / `model_b` /
     `veri_okul*` / `pencere_a` import'u YOK. Statik taramayla.
  2. MIMARI       projenin hicbir ozelligi yok: dongu yok, Phi
     darbogazi yok, maske yok, yardimci kayip yok, miras yok.
  3. VERI         `veri_10` KENDI iddialarini tutuyor: sema, soy
                  agaci (cinsiyet/kusak/ensest), zincir siniflari,
                  turetilebilirlik, notrluk, cografya, IZ.
  4. MOTOR        `taban_09` ile `model_a` AYNI SEYI olcuyor:
     egitim havuzu bit duzeyinde ayni, olcme izi ayni.

--------------------------------------------------------------------------
!! KILIT, PAYLASILAN MODULLERI IMPORT EDER -- VE ETMELIDIR

3 ve 4, model_09'in kopyalarini ORIJINALLERIYLE karsilastiriyor; bunun
icin `model_a` / `veri_okul4` / `model_b15` buraya yukleniyor. Bu,
1'deki bagimsizlik iddiasini BOZMAZ: KOSAN kol onlari gormez, yalniz
KILIT gorur. Kilidin isi zaten bu -- kopyanin sapip sapmadigini
soylemek.

Kopyanin GERCEK riski buydu: paylasilan motorda bir olcum hatasi
duzeltilirse buraya kendiliginden GELMEZ, ve model_09 ile model_b15 o
gunden sonra FARKLI KODLA olculmus olur. 4. bolum bunu sessiz olmaktan
cikariyor: sapma varsa kilit DUSER ve o gun bilerek karar verilir
(kopyayi guncelle, ya da farki onkayda yaz).

AYRINTI=1 -> gecen kontroller de basilir.
"""
import os
import ast
import importlib
import glob
import re
import io
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
print("=== 0) BAGIMSIZLIK: model_10/ disariya BAGLI MI ===")
# Metinde arama YAPMIYORUZ -- ilk denemem oyleydi ve `ayar_10.py`nin
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

print("\n=== 1) model_10 MIMARISI ===")
import taban_10 as M                                         # noqa: E402
import model_10 as S                                         # noqa: E402

A = S.AYAR
ok(S.M is M, "model_10 motoru taban_10", S.M.__name__)
ok(A.dongu == 1, "dongu = 1 (DONGU YOK)", str(A.dongu))
ok(A.l == 8, "l = 8 katman", str(A.l))
ok(A.dar_alfa == 0.0 and not A.dar_kapi,
   "Phi DARBOGAZI YOK (dar_alfa=0, dar_kapi=False)")
ok(A.kopru_kayip == 0.0, "YARDIMCI KAYIP YOK")
ok(A.mask_poz is None and not A.mask_blok, "MASKE YOK")
ok(A.dff == 704, "d_ff 704 (8/3 * 256, 64'un kati)", str(A.dff))
ok(A.d % A.nh == 0 and A.d // A.nh == 64, "head_dim 64", f"d={A.d} nh={A.nh}")
ok(not issubclass(S.ModelSade, M.Model),
   "ModelSade, taban_10.Model'den MIRAS ALMIYOR")
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

# --- 0b) KORPUS TEK YERDEN KURULUR -------------------------------------
# `korpus_10.havuz` 10 parametre aliyor. `tani_10` / `asama1_10` /
# `pencere_10` onu DOGRUDAN cagiriyordu ve 6'sini geciyordu -- kalan
# dordu (`zincir_pay`, `n3`, `ret_pay`, `ret_tut`) VARSAYILANA dusuyordu.
# `n3` egitimde 10.000, orada 0: okuma araclari EGITILEN KORPUSTAN BASKA
# bir korpus kuruyordu ve hicbir sey bunu soylemiyordu. Tek kapi:
# `KOR.havuz` yalniz `taban_10` icinden cagrilir.
# !! YORUM SATIRLARI ATILIR. Duzeltmenin kendi aciklamasi `KOR.havuz(`
# yaziyor; kapi KODU taramali, DUZYAZIYI degil -- yoksa hatayi
# anlatan yorum hatanin kendisi sayilirdi.
def _kodsuz(t):
    return NL_.join(l for l in t.split(NL_)
                    if not l.lstrip().startswith('#'))

NL_ = chr(10)
_kaynaklar = {os.path.basename(f): _kodsuz(io.open(f, encoding='utf-8').read())
              for f in glob.glob(os.path.join(_B, '*.py'))}
_dogrudan = sorted(f for f, t in _kaynaklar.items()
                   if f not in ('taban_10.py', 'korpus_10.py')
                   and re.search(r'KOR\w*\.havuz\(', t))
ok(not _dogrudan,
   'korpus YALNIZ taban_10 icinden kurulur (ayardan turer)',
   str(_dogrudan))

# --- 0c) CAGRILAN NITELIK GERCEKTEN VAR MI (statik) --------------------
# !! BU KAPI GUNUN EN PAHALI HATA SINIFINI KAPATIYOR. Bir fonksiyon bir
# dosyada yeniden adlandirilinca, onu CAGIRAN oteki dosya calisana kadar
# sessiz kaliyor. Bugun uc kez oldu:
#   KOR.biyografiler -> `sayfalar` oldu; `tani_10 --biyografi` (VARSAYILAN
#                       yol) her kosuda AttributeError ile duserdi
#   MT.cumle(...,3,tip) -> imza degisti; sinav her halukarda duserdi
#   _NL.null('09')     -> `veri_09` ariyordu, test CALISMA ZAMANINDA coktu
# Ucu de ancak KOSUNCA gorunurdu; ikisi GPU saati harcandiktan SONRA.
#
# YONTEM: her dosya AST ile ayristirilir, `import X as A` eslemeleri
# cikarilir, ve `A.nitelik` kullanimlarinin hepsi GERCEK modulde aranir.
# Yalniz BU KLASORUN modulleri denetlenir (torch/np disarida).
_yerli = {os.path.splitext(f)[0] for f in os.listdir(_B) if f.endswith('.py')}
_yok = []
for _f in sorted(glob.glob(os.path.join(_B, '*.py'))):
    _ad = os.path.basename(_f)
    if _ad.startswith('test_'):
        continue
    try:
        _ag = ast.parse(io.open(_f, encoding='utf-8').read())
    except SyntaxError as _e:
        _yok.append(f'{_ad}: AYRISTIRILAMADI {_e}')
        continue
    _esle = {}
    for _n in ast.walk(_ag):
        if isinstance(_n, ast.Import):
            for _al in _n.names:
                if _al.name in _yerli:
                    _esle[_al.asname or _al.name] = _al.name
    for _n in ast.walk(_ag):
        if (isinstance(_n, ast.Attribute) and isinstance(_n.value, ast.Name)
                and _n.value.id in _esle):
            _mod = importlib.import_module(_esle[_n.value.id])
            if not hasattr(_mod, _n.attr):
                _yok.append(f'{_ad}:{_n.lineno} {_n.value.id}.{_n.attr} -> {_esle[_n.value.id]}.{_n.attr} YOK')
# TASINMAYAN BORCU -- acikca sayilir, susturulmaz.
# Bu araclar HALA model_08'in JETON semasinda: `kelimeler`, `kodla_*`,
# `BICIM` karakter surumunde YOK. Kosulurlarsa duserler ve bu BILINIYOR
# (`konus_10.TASINMADI` ayni sebeple True). Liste KISALMALI, uzamamali:
# sayi BUYURSE kapi duser.
_BORC = {
    'analiz_10.py': 3,       # M.kelimeler, M.kodla_1hop, M.kodla_2hop
    'dokum_10.py': 4,        # K.biyografiler, MT.BICIM, MT.N_BICIM
    'egitim_dok_10.py': 17,  # M.kodla_* ailesi
    'graf_10.py': 1,         # M.kelimeler
    'havuz_10.py': 1,        # M.kelimeler
    # konus_10.py BORCTAN DUSTU (18 Eylul): karakter surumu yazildi,
    # 10 olu cagri kapandi. §8 artik BEYAN degil DAVRANIS siniyor.
}
_say = {}
for _x in _yok:
    _say[_x.split(':')[0]] = _say.get(_x.split(':')[0], 0) + 1
_yeni = [f'{_f}: {_n} (borc {_BORC.get(_f, 0)})'
         for _f, _n in sorted(_say.items()) if _n > _BORC.get(_f, 0)]
ok(not _yeni,
   '0c cagrilan her modul niteligi VAR (TASINMAYAN borcu haric)',
   str(_yeni))
_kapanan = [f'{_f}: {_n} -> {_say.get(_f, 0)}'
            for _f, _n in sorted(_BORC.items()) if _say.get(_f, 0) < _n]
ok(not _kapanan,
   '0c TASINMAYAN borcu KOPYA.json ile TUTUYOR (kapanan varsa yenile)',
   str(_kapanan))

# --- 1b) OKUMA ARACLARININ MODELDEN ISTEDIGI YUZEY ----------------------
# KUSUR (16 Eylul hakemligi): `asama1_10.gizli()` ileri gecisi ELLE
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
import asama1_10 as A1                                      # noqa: E402
import olcme_10 as OLC                                      # noqa: E402
import veri_10 as V00                                       # noqa: E402
_lst = v.comp[:8]
# !! MODELI M.DEV'E TASI. `gizli()` girdiyi `.to(M.DEV)` ediyor; model
# CPU'da kalirsa GPU'lu makinede "index is on cuda:0, other tensors on
# cpu" ile duser. ILK YAZIMDA BU EKSIKTI ve yerelde (DEV="cpu") FARK
# EDILMEDI -- kusur Colab'da, kilidin ilk GPU kosusunda cikti.
# Gercek kullanimda `asama1_10.main` modeli zaten `.to(M.DEV)` ediyor;
# yani hata ARACIN degil, BU TESTIN hatasiydi. Sonra CPU'ya geri
# aliniyor: asagidaki bolumler CPU tensorleriyle devam ediyor.
try:
    net.to(M.DEV)
    _Xh, _Sh = M.egitim_havuzu(A, v, yaz=lambda *a: None)
    _net63 = S.ModelSade(A, _Sh.vocab).to(M.DEV)
    _svh = OLC.Sinav(_Sh, v, V00.kur(A.veri_tohum), _lst, 2, "comp")
    _q = A1.gizli(_net63, _svh)
    ok(_q.shape == (8, A.d), "asama1_10.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
       f"{_q.shape} cihaz {M.DEV}")
    # ONEGIN SON KONUMU -- sabit bir poz DEGIL. Oneklerin uzunlugu
    # farkli ("...kimdir? " vs "...neresidir? "), yani sabit poz
    # satirlarin bir kismini YANLIS yerden okurdu.
    ok(len(set(_svh.bas.tolist())) > 1,
       "onek uzunlugu SATIRDAN SATIRA degisiyor -- sabit poz OLMAZ",
       f"{sorted(set(_svh.bas.tolist()))[:4]}...")
except Exception as _e:                                      # noqa: BLE001
    ok(False, "asama1_10.gizli() KOSUYOR (DOGRUSAL SONDA yolu)",
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

# --- 3) VERI: veri_10'in KENDI IDDIALARI --------------------------------
# !! model_03'ten DEVRALINAN "kopya == orijinal" TESTI KALDIRILDI.
# Orada `veri_03` bir KOPYAYDI (icerigi `veri_okul4` ile birebir) ve test
# kopyanin kaymadigini siniyordu. `veri_10` KOPYA DEGIL: yeni sema, yeni
# tipler, gercek soy agaci, gercek cografya. Kiyaslanacak bir ORIJINAL
# YOK. O yuzden test artik verinin KENDI IDDIALARINI siniyor -- ve o
# iddialarin cogu `veri_04`te TUTMUYORDU ("Fatma'nin annesi Huseyin").
print("\n=== 3) VERI: veri_10 KENDI IDDIALARINI tutuyor mu ===")
import veri_10 as V00                                        # noqa: E402

ok(A.veri_ad == "veri_10", "BU KOL KENDI veri modulunu okur", A.veri_ad)
_G0 = V00.kur(0)
E05 = [x for t in V00.TIPLER for x in _G0["ad"][t]]
_z = V00.zincirler(_G0)
ok(V00.graf_izi(_G0) == V00.IZ, "graf izi veri_10.IZ ile TUTUYOR", V00.IZ)
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

# --- 3j) METIN DENETIMI: her iliski x her kalip TURKCE mi --------------
# !! `metin_10.denetle` KOPYALAMADA DUSMUSTU ve KOPYA KAPISI gosterdi
# (kullanici: *"diff calistirip bakmadin mi?"*). Geri getirildi, ve ILK
# KOSUSUNDA bir kusur buldu: BILDIRIM kalibi 5 Turkce degildi
# ("Kardesidir, Ahmet Aydin'in Elif Aydin."). Buraya bagli ki bir daha
# sessizce dusmesin -- 600 cumlenin hepsi HER KOSUDA uretilip sinaniyor.
import metin_10 as MT10                                          # noqa: E402
_orn = MT10.denetle(0, yaz=lambda *a, **k: None)
ok(len(_orn) == len(V00.ILISKI),
   f"3j her iliskiden SINAV cumlesi uretiliyor ({len(_orn)}/{len(V00.ILISKI)})")

# --- 3h) REDDETME VERISI: modele YALAN ogretmiyor mu -------------------
# !! BU KAPI BIR HATADAN DOGDU. `reddetme_belgeleri` ilk surumde
# "ILISKI OLMAZ" ciftlerini SEMANIN TUMLEYENINDEN aliyordu ve uretilen
# cumlelerin bir kismi YANLIS ti:
#     "Bolumun ogrencisi olmaz."  bolumde ogrenci VAR
#     "Fakultenin sehri olmaz."   fakulte bir sehirde
#     "Tezin danismani olmaz."    en standart bag
# Modele DURUSTLUK ogretirken YALAN ogretecektik. Hicbir sayisal kapi
# bunu gostermedi -- METNI OKUMAK gosterdi. Kapi metni okuyor.
import korpus_10 as KOR10                                        # noqa: E402
import metin_10 as MT10                                          # noqa: E402

_red = [x for b in KOR10.reddetme_belgeleri(None, _G0, 600, tohum=0,
                                            yaz=lambda *a, **k: None)
        for x in b.cumle]
# GERCEK adlarin Turkce yuzeyleri, UZUNDAN KISAYA (kisa onek once
# eslesirse yanlis varliga gider)
_yuz = sorted({MT10._tr(a) for t in V00.TIPLER for a in _G0["ad"][t]},
              key=len, reverse=True)


def _gercek(x):
    return next((y for y in _yuz if x.startswith(y)), None)


# TUR, CEVAP cumlesinden ayrilir. !! Regexle ayirmak DENENDI ve
# YANLIS ALARM verdi: ILISKI kalibi 2 "... diye bir SEY yok" diyor ve
# VARLIK kalibi "... diye bir <tip> yok"a benziyor.
_rsoz = set(V00.TR_ILISKI.values())
_tur = lambda x: ("ILISKI" if any(r in x.split("? ", 1)[1] for r in _rsoz)
                  else "VARLIK")
_vy = [x for x in _red if _tur(x) == "VARLIK"]
_io = [x for x in _red if _tur(x) == "ILISKI"]

ok(_vy and not [x for x in _vy if _gercek(x)],
   f"3h-A VARLIK YOK: oznesi GRAFTA OLAN cumle yok "
   f"({len([x for x in _vy if _gercek(x)])}/{len(_vy)})")
ok(_io and not [x for x in _io if not _gercek(x)],
   f"3h-B ILISKI OLMAZ: oznesi GRAFTA OLMAYAN cumle yok "
   f"({len([x for x in _io if not _gercek(x)])}/{len(_io)})")

_var = {}
for (_a, _r) in _G0["olgu"]:
    _var.setdefault(_G0["tip"][_a], set()).add(_r)
_cak = [(t, r) for r, ts in V00.IMKANSIZ.items() for t in ts
        if r in _var.get(t, ())]
ok(not _cak, f"3h-C IMKANSIZ cift grafta YOK ({_cak[:2]})")

# YARI YARIYA: havuz tukendiginde KIRPILMAMALI. `uydurma_adlar` 1.460
# ad veriyor; 4.000 istenirse tekrar etmeli, azalmamali.
_buyuk = [x for b in KOR10.reddetme_belgeleri(None, _G0, 4000, tohum=0,
                                              yaz=lambda *a, **k: None)
          for x in b.cumle]
_bv = len([x for x in _buyuk if _tur(x) == "VARLIK"])
ok(abs(_bv - 2000) <= 1,
   f"3h-D havuz TUKENINCE tekrar eder, kirpilmaz (varlik yok {_bv}/4000)")

# TABLO ELLE YAZILIR: tumleyen ALINMAMIS olmali. Tumleyen 167 cift;
# tablo ondan KUCUK olmali, yoksa eleme yapilmamis demektir.
_tum = len(V00.TIPLER) * len(V00.ILISKI) - sum(len(v) for v in V00.SEMA.values())
ok(V00.N_IMKANSIZ < _tum,
   f"3h-E IMKANSIZ ELLE secilmis ({V00.N_IMKANSIZ} < {_tum} sema disi)")

# --- 3i) RET SEZGISI: `ret_mi` KOKLERI dogru mu -----------------------
# `olcme_10.RET_KOK` bes kok iceriyor ve DURUSTLUK olcusunun tamami ona
# dayaniyor. Kok listesi YANLISSA olcum sessizce bozulur: eksik kok ->
# dogru ret DUSUK gorunur; fazla kok -> olgu cumlesi "ret" sayilir ve
# kacamak SISER. O yuzden liste GERCEK KORPUSA karsi sinanir.
import olcme_10 as OLC10                                          # noqa: E402

_rc = [x for b in KOR10.reddetme_belgeleri(
       None, _G0, 400, tohum=1, bolme=KOR10.reddetme_bolme(_G0, 0.2, 1),
       yaz=lambda *a, **k: None) for x in b.cumle]
_rcev = [x.split("? ", 1)[1] for x in _rc]
_kacir = [x for x in _rcev if not OLC10.ret_mi(x)]
ok(not _kacir, f"3i-A her REDDETME cevabi yakalaniyor ({_kacir[:2]})")

# OLGU cumleleri: sayfalardan. Hicbiri ret SAYILMAMALI.
_bb, _bs = KOR10.sayfalar(v, _G0, kopya=1, tohum=1,
                          yaz=lambda *a, **k: None)
_olgu = [c for b in (_bb + _bs) for c in b.cumle]
_yanlis = [c for c in _olgu if OLC10.ret_mi(c)]
ok(not _yanlis,
   f"3i-B hicbir OLGU cumlesi ret sayilmiyor ({len(_yanlis)}/{len(_olgu)}"
   f"  {_yanlis[:2]})")

# TUTULAN, egitimde GECMIYOR mu -- sinavin tamami buna dayaniyor
_bl = KOR10.reddetme_bolme(_G0, 0.20, 0)
_egt = set()
for _b in KOR10.reddetme_belgeleri(None, _G0, 4000, tohum=0, bolme=_bl,
                                   yaz=lambda *a, **k: None):
    _egt.update(_b.cumle)
_egt_m = " ".join(_egt)
_sz = [MT10._tr(a) for L in _bl["ad_tut"].values() for a in L]
_gecen = [a for a in _sz if a in _egt_m]
ok(not _gecen,
   f"3i-C TUTULAN ad egitim reddetmelerinde GECMIYOR ({_gecen[:2]})")

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

# (3) `jeton_ad` EK JETONLARINDA cokmuyor. veri_dok/analiz_09 `ek_kip`
#     gelen her kolda IndexError veriyordu -- model_b15'in verisi bu
#     yuzden hic dokulmemisti.
sys.path.insert(0, _B)
import analiz_10 as AZ                                       # noqa: E402
_d = AZ.Dok("model_10")
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
ok(_ekler == _bek, "analiz_10.jeton_ad EK JETONLARINI adlandiriyor",
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
    ok(hasattr(V00, _g), f"veri_10.{_g} var (sozlesme)")

# `model_a` / `model_b15` YALNIZ BU KILIT icin yukleniyor -- bu bolum
# motorun eski aileyle AYNI SEYI olctugunu siniyor, kosuda kullanilmaz.
# (Yol eklemeleri eskiden 3. bolumun icindeydi; o bolum `veri_okul4`
# kiyasiyla birlikte kaldirilinca buraya tasindi.)
sys.path.insert(0, _K)
sys.path.insert(0, os.path.join(_K, 'model_a'))
sys.path.insert(0, os.path.join(_K, 'model_b'))

# --- 4) MOTOR: taban_09 ile model_a AYNI SEYI mi olcuyor ----------------
print("\n" + "=== 4) BILDIRILMIS AYRISMA: model_10 <-> model_09 ===")
import model_a as MA                                         # noqa: E402
import metin_10 as MT                                        # noqa: E402
import olcme_10 as OLC                                       # noqa: E402
from ayar_10 import GOREV_ALAN                               # noqa: E402
import ayar_10 as AY10                                       # noqa: E402
from model_b15 import AYAR as B15    # §5 RECETE KIYASI icin          # noqa: E402

# model_09 model_08'in KOPYASI olarak dogdu ve EGITIM PARADIGMASINI
# degistirdi: satir tablosu -> paketlenmis metin, jeton semasi ->
# karakter. Bu bir sapma DEGIL, kolun TANIMI (kullanici, 18 Eylul:
# "biz artik dil egitimine gectik").
#
# Ama "degisti" demek yetmez. SINANAN SEY YON:
#   DEGISMESI GEREKEN degisti mi   -- egitim yolu
#   DEGISMEMESI GEREKEN ayni mi    -- SINAV ve GRAF
#
# Ikincisi bu kolun butun kiyaslanabilirligi. Sinav degisirse
# model_08'in sayilariyla ayni tabloda okunamaz ve kol bir sey
# soylemez.
_alan09 = {f.name for f in M.dc.fields(M.Ayar)}
_alanA = {f.name for f in MA.dc.fields(MA.Ayar)}
ok(_alan09 - _alanA == {"fim_kat", "soru_kat", "kisayol_kat",
                        "t_len", "kopya", "tetik",
                        "zincir_pay", "n3", "ret_pay", "ret_tut"},
   "Ayar'a EKLENEN alanlar: fim/soru/kisayol (06-09) + korpus (10)",
   f"eklenen {sorted(_alan09 - _alanA)}")
ok(not (_alanA - _alan09), "model_a'nin hicbir alani SILINMEDI",
   f"eksik {sorted(_alanA - _alan09)}")
ok(all(MA.ESKI_VARSAYILAN[k] == M.ESKI_VARSAYILAN[k]
       for k in MA.ESKI_VARSAYILAN),
   "model_a'nin ESKI VARSAYILANLARININ HICBIRI degismedi")
# !! `t_len` ARTIK PROPERTY DEGIL, ALAN. model_08'de jeton semasindan
# TURETILIYORDU (2*yuva+11 = 17) cunku her satir TEK cumleydi. Burada
# pencere bir VERI KARARI ve turetilecek bir sey yok.
ok(not isinstance(M.Ayar.__dict__.get("t_len"), property),
   "t_len ALAN (model_08'de TURETILMIS property idi)", str(A.t_len))
ok(A.t_len == 512 and "t_len" in A.sozluk(),
   "t_len ayar_t<N>.json'a YAZILIYOR -- turetilse yazilmazdi")

# --- DEGISMEMESI GEREKEN: SINAV ve GRAF ------------------------------
IZ_08_OLCME = "cfafdcc15a23"
IZ_08_GRAF = "3cd9a2575e47"
_L09 = M.olcme_listeleri(A, v)
_iz09 = M.olcme_izi(_L09)
ok(_iz09 == IZ_08_OLCME,
   "OLCME IZI model_08 ile AYNI -- SINAV DEGISMEDI (onkayit 2)", _iz09)
ok(V00.IZ == IZ_08_GRAF, "GRAF IZI model_08 ile AYNI", V00.IZ)
ok(set(_L09) >= {"one", "seen", "comp", "ent", "ent_yok", "ood"},
   "ALTI bolme de olculuyor")

# --- DEGISMESI GEREKEN: EGITIM YOLU ----------------------------------
ok(not any(hasattr(M, x) for x in
           ("kodla_1hop", "kodla_2hop", "kodla_fim", "kopru_hedefi")),
   "model_08'in KODLAYICILARI silindi (jeton semasi yok)")
ok(not hasattr(M, "dogruluk") and not hasattr(M, "kisayol_orani"),
   "KISITLI ARGMAX olcusu silindi -- yerine olcme_10 SERBEST URETIM")
_hav = M.egitim_havuzu(A, v, yaz=lambda *a: None)
ok(isinstance(_hav, tuple) and len(_hav) == 2,
   "egitim_havuzu (X, S) donduruyor -- P/T/kimlik/KPOZ/KTR YOK",
   f"{len(_hav)} deger")
X0, S0 = _hav
ok(X0.shape[1] == A.t_len == 512, "havuz penceresi t_len", str(X0.shape))
# !! 63 -> 64. Tek yeni karakter ":" ve UC kalibdan geliyor
# ("Sunu da ekleyelim:", "Bir baska kayit:", "Bir de sunu sorayim:").
# Sayi DEGISTIGI icin degil, NEDEN degistigi yazili oldugu icin geciyor:
# sessiz bir karakter artisi bozuk metnin ilk isaretidir.
ok(S0.vocab == 64, "sozluk 64 jeton (62 karakter + PAD + EOS)", str(S0.vocab))
ok(":" in S0.harf and len(S0.harf) == 62, "yeni karakter ':' (3 kalip)")
ok(int(X0.max()) < S0.vocab and int(X0.min()) >= 0,
   "havuzdaki her jeton SOZLUK ICINDE", f"max {int(X0.max())}")
ok(len(S0.korpus_izi) == 12, "KORPUS IZI kuruldu", S0.korpus_izi)

# --- MARUZIYET: butce TURETILDI, tahmin edilmedi ----------------------
# `adim` bir tercih degil, bu esitligin cozumu. Biri degisip digeri
# degismezse kol model_08 ile ayni tabloda okunamaz.
# !! ESKI HALI "maruziyet model_08'inkine (442) ESIT" diyordu ve
# 425..460 arasi bekliyordu. IKI gerekce birden cokdu:
#   1) Kullanici: *"model 8 alakasi yok karsilastirma bitti."*
#      (CLAUDE.md kural 5) -- esitlenecek kol kalmadi.
#   2) Korpus 22,3M -> 9,1M jetona indi; ayni adim sayisi artik 44,0
#      degil 107,6 EPOK demek. Kapi bunu GORMEDEN gecirirdi.
# Yerine: EPOK sayisi LITERATURUN soyledigi aralikta mi.
_jeton = int((X0 != 0).sum())
_gecis = A.batch * A.t_len * A.adim / _jeton
ok(abs(_jeton - AY10.KORPUS_JETON) / AY10.KORPUS_JETON < 0.02,
   "ayar_10.KORPUS_JETON gercek korpusa UYUYOR -- butce ondan turuyor",
   f"olculen {_jeton:,} / yazili {AY10.KORPUS_JETON:,}")
ok(30 < _gecis < 40,
   "ilk kosu 30..40 epok (Muennighoff 2305.16264: 44 epok BASARISIZ rejim)",
   f"{_gecis:.1f} epok / {_jeton:,} jeton / adim {A.adim:,}")
ok(A.adim == 20000, "CLAUDE.md kural 1: ILK KOSU 20.000", str(A.adim))
ok((A.fim_kat, A.soru_kat, A.kisayol_kat, A.ident_frac, A.bicim)
   == (0, 0, 0, 0.0, 1),
   "SATIR TABLOSU alanlari KAPALI -- ayar_t<N>.json YALAN yazmiyor")

# --- SINAV KENDI GENISLIGINDE ----------------------------------------
_sv = OLC.Sinav(S0, v, V00.kur(A.veri_tohum), _L09["comp"], 2, "comp")
ok(_sv.genislik < A.t_len // 3,
   "SINAV egitim penceresinden COK dar -- olcum o kadar ucuz",
   f"{_sv.genislik} vs {A.t_len}")
_onek = OLC.sorular(v, V00.kur(A.veri_tohum), _L09["comp"][:50], 2)
ok(all(o.endswith("? ") for o, _b, _c, _k in _onek),
   "sinav onegi SORU ISARETIYLE bitiyor -- tek devami var")
ok(all(b.endswith(".") for _o, b, _c, _k in _onek),
   "beklenen cevap NOKTAYLA bitiyor -- uretim orada duruyor")


print("\n" + "=== 4b) PUANLAMA KAPISI: sahte ciktiyla ===")
# !! BU BOLUM BIR KUSURDAN DOGDU (18 Eylul, hakemlik). `olc` once
# `cev in cikti` diye bakiyordu. OLCULDU: varlik adlari arasinda
# 48 ALT-DIZGE cakismasi var -- "Aydin" bir SEHIR ve ayni zamanda
# "Mehmet Aydin"in icinde. Sehir sorusuna "Mehmet Aydin'dir." diyen
# model "yakin" sayilirdi; kisayol orani da ayni sekilde SISERDI.
# Onek kiyasinda cakisma: 0 cift.
#
# Kusur ancak 20.000 adimlik kosudan SONRA, sayilar "biraz tuhaf"
# gorununce fark edilirdi. Puanlama bu yuzden modelden AYRILDI
# (`olcme_10.puanla`) ve burada SAHTE CIKTIYLA sinaniyor.
_BEK = ["Sirnak'tir.", "Sirnak'tir.", "Sirnak'tir.", "Aydin'dir.",
        "Aydin'dir.", "Onur Demir'dir."]
_CEV = ["Sirnak", "Sirnak", "Sirnak", "Aydin", "Aydin", "Onur Demir"]
_KSY = [None, None, "Bolu", None, None, "Onur"]
_CIK = ["Sirnak'tir.",        # TAM
        "Sirnak'ta.",         # YAKIN  (varlik dogru, ek yanlis)
        "Bolu'dur.",          # KISAYOL
        "Mehmet Aydin'dir.",  # ALT-DIZGE TUZAGI -> hicbiri
        "Aydin'dir.",         # TAM
        "Onur Demir'dir."]    # TAM  (ksy "Onur" ONEK -- tam once gelmeli)
_p = OLC.puanla(_BEK, _CEV, _KSY, _CIK)
ok(abs(_p["tam"] - 3 / 6) < 1e-9, "puanla: TAM 3/6", f"{_p['tam']:.4f}")
ok(abs(_p["yakin"] - 1 / 6) < 1e-9, "puanla: YAKIN 1/6 (ek yanlis)",
   f"{_p['yakin']:.4f}")
ok(abs(_p["kisayol"] - 1 / 6) < 1e-9, "puanla: KISAYOL 1/6",
   f"{_p['kisayol']:.4f}")
# MUTASYON: `startswith` yerine `in` konsaydi 4. satir YAKIN sayilirdi.
ok(sum(1 for b_, c_, k_, c in zip(_BEK, _CEV, _KSY, _CIK)
       if c != b_ and c_ in c) == 2,
   "ALT-DIZGE olcusu 2 YAKIN sayardi (biri YANLIS) -- onek 1 sayiyor",
   "\"Mehmet Aydin'dir.\" icinde \"Aydin\" GECIYOR")
# KISAYOL, dogru cevabin ONEKI olabilir -> SIRA onemli.
ok(OLC.puanla(["Onur Demir'dir."], ["Onur Demir"], ["Onur"],
              ["Onur Demir'dir."])["tam"] == 1.0,
   "kisayol adi dogru cevabin ONEKI ise TAM sayiliyor",
   "sira: tam -> yakin -> kisayol")

# --- ENT-YOK: kisayol YAPISAL olarak yok -----------------------------
# `erken_teshis`in BIRIM TESTI'si buna dayaniyor. Kapinin OLCUM kodunu
# denetlemedigi ACIKCA yaziyor (yukaridaki §4b onu denetliyor); burada
# sinanan sey VERI: bolme mantigi ENT-YOK'a kisayolu MUMKUN bir zincir
# sizdirmis mi.
_qy = OLC.sorular(v, V00.kur(A.veri_tohum), _L09["ent_yok"], 2)
ok(all(k is None for _o, _b, _c, k in _qy),
   "ENT-YOK'un HICBIR zincirinde kisayol varligi YOK (facts = -1)",
   f"{len(_qy):,} zincir")
_qc = OLC.sorular(v, V00.kur(A.veri_tohum), _L09["ent"], 2)
ok(sum(k is not None for _o, _b, _c, k in _qc) > len(_qc) // 2,
   "ENT'te kisayol varligi VAR -- yani olcu ORADA calisabiliyor",
   f"{sum(k is not None for _o, _b, _c, k in _qc):,}/{len(_qc):,}")

# --- YENI SIRALAMA: eksik hatirlamayi 'yeni' saymamali ----------------
# Ayni aileden UCUNCU kusur (18 Eylul). Olcu once TAM ESITLIK bakiyordu
# ve ezber testinde soyle cikti:
#     EOS yok  recall 1.0000  yeni_sira 0.0000
#     EOS var  recall 0.8889  yeni_sira 1.0000   <- model KOTULESTI,
#                                                  olcu IYILESTI dedi
# Cunku bir olgu atlanınca dizi hicbir egitim siralamasina ESIT olamaz.
_EGT = [(1, 2, 3, 4), (4, 3, 2, 1)]
ok(not OLC.yeni_siralama((1, 2, 3, 4), _EGT),
   "TAM esitlik -> geri oynatma")
ok(not OLC.yeni_siralama((1, 3, 4), _EGT),
   "EKSIK ama SIRA tutarli -> geri oynatma (eski olcu YENI sayardi)",
   "asil kusur buydu")
ok(OLC.yeni_siralama((1, 3, 2), _EGT),
   "sira TUTARSIZ -> gercekten YENI")
ok(not OLC.yeni_siralama((3,), _EGT),
   "tek olguda 'sira' diye bir sey YOK -> yeni sayilmaz")


# --- 5) STANDART TARIF -- DEVRALINMAYANLAR ------------------------------
print("\n=== 5) STANDART TARIF ===")
# !! BU KILIT GEVSETILMEDI, HEDEFI DEGISTI (CLAUDE.md'deki wd=0.5
# emsali). model_10..model_07'de `ort_bas == 0` bekleniyordu: "standart
# tarif, projeye ozgu ne varsa KAPALI". model_09'de geri beslemeli
# ortalama BU KOLUN DUGMESI (onkayit belge/onkayit/model_10.md), o
# yuzden kilit artik ACIK olmasini ve DOGRU DEGERDE olmasini bekliyor.
# !! ESIKLER ADIM DEGIL **ORAN** olarak sinaniyor (18 Eylul). Butce
# 20.000 -> 60.000'e cikinca (esit maruziyetin cozumu) mutlak 6000
# kosunun %30'u olmaktan cikip %10'u olurdu; LR/ortalama cizelgesinin
# SEKLI model_08'inkinden ayrilirdi ve kiyas zemini oydu.
ok(A.ort_bas == 3 * A.adim // 10,
   "LOOKAHEAD ACIK -- kosunun %30'una kadar NORMAL egitim",
   f"ort_bas {A.ort_bas:,} / adim {A.adim:,}  (model_08: 6000/20000)")
ok(A.ort_her == A.olc_her and A.adim // A.olc_her == 10,
   "ortalama esigi olcum noktasiyla AYNI, 10 nokta (model_08 ile ayni sayi)",
   f"ort_her {A.ort_her:,}  olc_her {A.olc_her:,}")
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
ok(A.isinma == A.adim // 10,
   "isinma kosunun %10 -- model_08 ORANI (2000/20000)",
   f"{A.isinma:,}/{A.adim:,}. MUTLAK 2000 tasinsaydi %3,3 olurdu; "
   f"nanoGPT'nin 2000'i da kendi kosusunun %0,33'u, yani 'mutlak 2000' "
   f"zaten nanoGPT'yi TAKLIT ETMIYORDU.")
ok(A.lr == 1e-3, "lr 1e-3 -- Pythia-70m ile ayni mertebe")

# --- 6) PARAMETRE --------------------------------------------------------
print("\n=== 6) PARAMETRE ===")
_bek = 8 * (4 * A.d * A.d + 3 * A.d * A.dff)
ok(abs(net.n_param() - (_bek + v.vocab * A.d)) < 5000,
   "parametre ~ 8*(4d^2 + 3*d*dff) + vocab*d",
   f"{net.n_param():,} (blok {_bek:,} + gomme {v.vocab*A.d:,})")
ok(net.n_param() > 6_000_000, "6M+ parametre (model_b15 3,23M)",
   f"{net.n_param():,}")
# !! SAYILAR DOCSTRING'DE DEGIL BURADA (dis hakemlik, 18 Eylul:
# "sayilar docstring'de degil testte olmali -- orada CURUR").
# Asagidakilerin hepsi `model_10.py` basindaki tabloda YAZILI; yazili
# olan ile HESAPLANAN ayrilirsa bu satirlar duser.
_sw = 3 * A.d * A.dff
_gl = 2 * A.d * 4 * A.d
ok(_sw == 540_672 and _gl == 524_288,
   "SwiGLU 540.672 / GELU+4d 524.288 -- tablodaki sayilar",
   f"{_sw:,} vs {_gl:,}")
ok(abs(_sw / _gl - 1.031) < 0.002,
   "SwiGLU GELU+4d'den %3,1 FAZLA -- 'ayni parametre' DEGIL",
   f"%{100 * (_sw / _gl - 1):.1f}, 8 katmanda +{(_sw - _gl) * A.l:,}")
ok(A.dff == 704 and A.dff % 64 == 0,
   "d_ff 704 = 8/3*256 -> 64'un kati (LLaMA 256'ya yuvarlar: 768)",
   f"{A.dff}  8/3*{A.d} = {8 * A.d / 3:.1f}")
# BAGLI GOMME: `parameters()` zaten tekillestiriyor.
ok(net.n_param() == sum(_p.numel() for _p in net.parameters()),
   "n_param() == parameters() toplami (dedup sozlugu GEREKSIZDI)")
ok(net.n_param(bagli_ayri=True) - net.n_param() == net.emb.weight.numel(),
   "bagli_ayri=True gommeyi IKI kez sayiyor",
   f"{net.n_param(bagli_ayri=True):,}")

# --- STANDART TARIF FIILEN KOSUYOR MU (docstring degil, KAYNAK) -------
# Dis hakemlik sordu: "wd yalniz >=2D tensorlere mi? warmup var mi?
# grad clip var mi?" Ucu de VAR ama hicbiri sinanmiyordu -- yani
# `egit` yeniden yazilsa iddia SESSIZCE yalan olurdu.
import inspect as _ins                                      # noqa: E402
_egit_src = _ins.getsource(M.egit)
ok("p.dim() >= 2" in _egit_src and 'weight_decay": 0.0' in _egit_src,
   "wd YALNIZ dim>=2 tensorlere (RMSNorm kazanci decay DISI)")
ok("adim < ayar.isinma" in _egit_src, "ISINMA var", f"isinma={A.isinma}")
ok("clip_grad_norm_(model.parameters(), 1.0)" in _egit_src,
   "GRAD CLIP 1.0 var")
ok(A.ort_bas > 0, "LOOKAHEAD ACIK -- bu kolun DUGMESI",
   f"ort_bas={A.ort_bas} ort_her={A.ort_her} alfa={A.ort_alfa}")

# --- RoPE GORELILIGI fp16 YOLUNDA (OLCULUR, iddia edilmez) ------------
# Donme fp16'da yapiliyor. LLaMA/HF fp32'de yapip cast eder; bedeli
# burada OLCULUYOR ki "sayisal fark yok" bir iddia degil bir sayi olsun.
_c, _sn = S._rope_tablo(A.t_len, A.d // A.nh)
torch.manual_seed(0)
_u, _w = torch.randn(A.d // A.nh), torch.randn(A.d // A.nh)


def _skor(m, n, dt):
    _q = torch.zeros(1, 1, A.t_len, A.d // A.nh)
    _k = torch.zeros(1, 1, A.t_len, A.d // A.nh)
    _q[0, 0, m] = _u
    _k[0, 0, n] = _w
    _Q = S._rope(_q.to(dt), _c, _sn).float()
    _K = S._rope(_k.to(dt), _c, _sn).float()
    return float((_Q[0, 0, m] * _K[0, 0, n]).sum())


_p32 = [_skor(_m, _m - 3, torch.float32) for _m in (10, 100, 250, 400, 508)]
_p16 = [_skor(_m, _m - 3, torch.float16) for _m in (10, 100, 250, 400, 508)]
ok(max(_p32) - min(_p32) < 1e-4,
   "RoPE GORELI (fp32): q_m.k_n yalniz (m-n)'ye bagli",
   f"yayilim {max(_p32) - min(_p32):.2e}")
ok(max(_p16) - min(_p16) < 1e-2,
   "RoPE fp16 yolunda da goreli -- 100 kat kotu ama skorun %0,05'i",
   f"yayilim {max(_p16) - min(_p16):.2e}  (fp32 {max(_p32) - min(_p32):.2e})")

# --- 7) MODUL SOZLESMESI -------------------------------------------------
print("\n=== 7) SOZLESME ===")
for g in ("AYAR", "egit", "fark_bas", "ModelSade", "TABAN"):
    ok(hasattr(S, g), f"model_10.{g} var", "kos_10.py duser")
# !! `sor_08` LISTEDEN CIKTI (18 Eylul, kullanici dosyayi SILDI:
# *"sor py sacma olmustu"*). O arac soruyu kendi cozumluyor, diziyi
# `kodla_1hop` ile KENDI kuruyor ve modele yalniz cevap yuvalarini
# sorduruyordu -- yani ekranda gorulen sey modelin dil uretimi DEGILDI.
# Yerini `konus_09` aldi: kisit yok, model kendi yazdigini okur.
for ad in ("veri_10", "taban_10", "ayar_10", "pencere_10", "tani_10",
           "asama1_10", "kos_10", "konus_10"):
    _r2 = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {_B!r}); import {ad}"],
        capture_output=True, text=True, cwd=_B)
    ok(_r2.returncode == 0, f"{ad} TEK BASINA import (yalniz model_10/)",
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
# `taban_10.veri_kur` icinde assert'le kapali; burasi BAGIMSIZ olarak
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
# `IZ_10` ELLE YAZILI bir sabit ve kimse onu denetlemiyordu. Defter
# eski izle kaldi -- 7. hucredeki `assert _iz == IZ_10` KOSUYU
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
    _m = _re.search(r'IZ_10\s*=\s*"([0-9a-f]+)"', _src)
    ok(_m is not None, "defterde IZ_10 tanimli")
    if _m:
        ok(_m.group(1) == _iz_ger,
           "defterdeki IZ_10 = GERCEK olcme izi",
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
    # diye onlarca kilit tasiyor ve bunlar `ayar_10.py`nin KOPYASI --
    # yani ayar degisince SESSIZCE eskiyor.
    #
    # OLDU (18 Eylul): kolun dugmesi `ort_bas` 0 -> 6000 oldu, `test_09`
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
    ok(not _kot, f"defterin {len(_kilit)} AYAR kilidi ayar_10 ile TUTUYOR",
       str(_kot))
    ok(len(_kilit) >= 20,
       "defter 3. hucresi AYAR kilitlerini hala tasiyor (silinmemis)",
       f"{len(_kilit)} kilit")

    # --- DEFTERDEKI BAYRAKLAR ARACTA VAR MI ---------------------------
    # !! BIR KUSURDAN DOGDU (18 Eylul, hakemlik). Defterin 9. hucresi
    #     !python {ASAMA1} ... --genislik 5 --sonda --birim --bolme comp,ent,ood,seen
    # diyordu; `asama1_09` karakter duzeyine cevrilince o dort bayragin
    # UCU kalmadi. Hucre KOSUDAN SONRA calisiyor -- yani hata GPU saati
    # harcandiktan, kosu bittikten sonra cikardi. Ayni sey `tani_09
    # --genislik` icin de gecerliydi.
    #
    # Kapi MEKANIK: defterdeki her `!python {ARAC} ... --bayrak`
    # satirindan bayraklar cikarilir, aracin `add_argument`larindan
    # tanimlar cikarilir, fark bakilir. Arac yeniden yazilinca defter
    # SESSIZ kalamaz.
    _ARAC = {"PENCERE": "pencere_10.py", "ASAMA1": "asama1_10.py",
             "TANI": "tani_10.py", "ANALIZ": "analiz_10.py",
             "NULL": "null_10.py", "KONUS": "konus_10.py"}
    _kotu = []
    for _sat in _src.split(chr(10)):
        _t = _sat.strip()
        if not _t.startswith("!python "):
            continue
        _mv = _re.search(r"\{(\w+)\}", _t)
        if not _mv or _mv.group(1) not in _ARAC:
            continue
        _dy = os.path.join(_B, _ARAC[_mv.group(1)])
        if not os.path.exists(_dy):
            _kotu.append((_mv.group(1), "DOSYA YOK")); continue
        _kod = _io.open(_dy, encoding="utf-8").read()
        _var = set(_re.findall(r'add_argument\(\s*"(--[\w-]+)"', _kod))
        # `!python {X} {EV}/...` satirinin sonunda bir sonraki hucrenin
        # ilk satiri yapisik gelebiliyor (kaynak birlestirildi) --
        # bayraklar yine dogru ayrisir, yorum `#` ile baslayan parca
        # atilir.
        _kul = set(_re.findall(r"(?<![\w-])(--[\w-]+)", _t.split("#")[0]))
        _eks = sorted(_kul - _var)
        if _eks:
            _kotu.append((_ARAC[_mv.group(1)], f"BAYRAK YOK: {_eks}"))
    ok(not _kotu, "defterdeki her bayrak aracta TANIMLI", str(_kotu))

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
import null_10 as _NL                                       # noqa: E402
# !! KOL NUMARASI KLASORDEN. Elle yazili "09" burada duruyordu ve
# `veri_09` modulunu ariyordu -- model_10 klasorunde YOK, test
# CALISMA ZAMANINDA coktu ve arkasindaki §7e HIC KOSMADI.
_kol_no = os.path.basename(_B).split("_")[-1]
_nl = _NL.null(_kol_no, yaz=lambda *a, **k: None, v=v)
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
# ki model_08'in izi cfafdcc15a23. (Bu OLAY model_08'de oldu; ayni
# hata model_09 kopyalanirken TEKRARLANDI ve yine elle duzeltildi.)
# Kor yeniden adlandirma eskimis bir
# cumleyi KENDINDEN EMIN bir yalana cevirir.
#
# !! KAPI DAR TUTULDU: yalniz EKRANA BASILAN metin (ok(...) aciklamasi
# ve baslik) taranir. Yorumlardaki gecmise atif (`havuz_06 yakaladi`,
# `veri_04'te`) MESRU ve cok -- onlari taramak kapiyi gurultuye bogar
# ve gurultulu kapi kapatilir.
print("\n=== 7e) KOL NUMARASI KAPISI ===")
import glob as _glob, io as _io2, re as _re2                 # noqa: E402
# !! ELLE YAZILMAZ, KLASORDEN TURER. Onceki hali `_KOL = "09"` idi:
# yani model_10 klasorunde kapi "09"u DOGRU kol sayiyor, "10"u YABANCI
# sayiyordu -- tam TERSINE calisiyordu. Kapinin kendisi kopyalanirken
# eskimisti ve bunu gorecek bir sey yoktu.
_KOL = os.path.basename(_B).split("_")[-1]
assert _KOL.isdigit(), f"kol numarasi klasorden okunamadi: {_B}"
# EKRANA BASILAN metinde BASKA kol numarasi. Gecmise atif ACIKCA yazilir.
# !! model_09'UN EKLEDIGI UC MUAFIYET. Hepsi §4'te ve hepsi AYNI
# sebeple: bu kol model_08'den BILEREK ayrildi ve ayrismanin YONU
# sinaniyor. "model_08" kelimesi burada bir KOPYA ARTIGI degil, kolun
# kiyas zemini.
#
# Muafiyet DAR: modul adi degil, TAM CUMLE. "model_08" gecen her metni
# muaf tutmak kapiyi kapatirdi -- korunmak istenen sey tam olarak o:
# kor yeniden adlandirmanin ureteceği cumleler bu listede OLMAZ.
#
# Ve bu cumleler KENDI KENDINI DENETLIYOR: `IZ_08_OLCME` sabiti canli
# degerle karsilastiriliyor. `veri_10` degisirse assert DUSER -- yani
# cumle eskirse sessiz kalmiyor.
_GECMIS_MSJ = ("model_b15", "model_05'ten KUCUK", "model_a ",
               "model_08 ile AYNI",
               # ESIK ORANLARI: model_08'in cizelge SEKLI kiyas zemini.
               "(model_08: 6000/20000)",
               "model_08 ile ayni sayi",
               "model_08 ORANI (2000/20000)",
               "model_08'de TURETILMIS property idi",
               "model_08'in KODLAYICILARI silindi",
               "model_08'inkine (442) ESIT",
               # §4 BASLIGI: bu bolum gercekten model_09 ile kiyas
               # yapiyor, "model_09" burada KOPYA ARTIGI degil KONU.
               "BILDIRILMIS AYRISMA: model_10 <-> model_09")
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

print("\n=== 8) KONUS_10 DAVRANISI ===")
import io                                                    # noqa: E402
import konus_10 as _K                                        # noqa: E402

# !! BU BOLUM UC KEZ HEDEF DEGISTIRDI ve ucu de KAYITLI.
#   1. model_08'de 108 satirdi, JETON surumunu siniyordu (`sinav_coz`,
#      `--genislik` varsayilani, bosluk cevabi tek jeton mu).
#   2. model_09'da hepsi ANLAMSIZ kaldi -- karakter duzeyinde yuva yok.
#      Kilit SILINMEDI, "arac KENDINI ne ilan ediyor"a cevrildi
#      (`TASINMADI = True`) ve soyle bir not dusuldu: *"bayrak dusunce
#      bu bolum DAVRANIS kapilariyla yeniden yazilmali."*
#   3. 18 Eylul: karakter surumu yazildi, bayrak DUSTU. Not tutuldu --
#      bu bolum artik DAVRANIS siniyor.
ok(not getattr(_K, "TASINMADI", False),
   "konus_10 TASINDI (karakter surumu)",
   "bayrak hala True ise arac KOSMAZ ve bu bolum BEYAN kapisina doner")
ok(not hasattr(_K, "_tasinmadi_kapisi"),
   "tasinma kapisi KALDIRILDI -- arac artik gercekten kosuyor")
for _g in ("yukle", "sor", "panel", "panel_sorulari"):
    ok(hasattr(_K, _g), f"konus_10.{_g} var (sozlesme)")

# --- URETIM SINAVIN KODUYLA AYNI OLMALI ------------------------------
# Ayri bir uretim yolu yazilsaydi "konusta boyle diyor ama sinavda
# baska" diye bir ayrisma dogardi ve hangisinin dogru oldugu
# bilinemezdi. KAYNAK taranir, docstring degil.
_ks = io.open(os.path.join(_B, "konus_10.py"), encoding="utf-8").read()
ok("OLC.uret(" in _ks,
   "konus_10 SINAVIN uretim kodunu kullaniyor (olcme_10.uret)")
ok("M.egitim_havuzu(" in _ks,
   "konus_10 korpusu AYARDAN kuruyor (§0b ile ayni kural)")

# --- NITELIK PANELI: 12 SONDA, DORT BOYUT ----------------------------
# Onkayit model_10.md §4. Sorular GRAFTAN uretiliyor; elle yazilsaydi
# veri degisince "eski soruyu yeni veriye sormak" kusuru dogardi.
_P = _K.panel_sorulari(_G0)
ok(len(_P) == 12, f"panel 12 sonda ({len(_P)})")
_boy = {x[0].split()[0] for x in _P}
ok(_boy == {"DOGRULUK", "DURUSTLUK", "TUTARLILIK", "AKICILIK"},
   "panel DORT boyutu da kapsiyor", str(sorted(_boy)))
# DURUSTLUK sondalari TUTULAN veriden olmali -- gordugunu reddetmek
# EZBER olurdu, olctugumuz GENELLEME.
_bl = KOR10.reddetme_bolme(_G0, 0.20, 0)
_tut_ad = {MT10._tr(x) for L in _bl["ad_tut"].values() for x in L}
_dur = [x for x in _P if x[0].startswith("DURUSTLUK")]
ok(len(_dur) == 3, "DURUSTLUK uc sonda: olmayan ad + imkansiz cift + KACAMAK")
ok(any(any(t in x[1] for t in _tut_ad) for x in _dur),
   "DURUSTLUK sondasinin adi TUTULAN havuzdan (egitimde GECMEZ)")
ok(sum(1 for x in _dur if x[2] == "REDDETMELI") == 2
   and sum(1 for x in _dur if x[2] not in ("REDDETMELI", None)) == 1,
   "iki sonda REDDETMELI, biri BILIYOR -- kacamak olmadan ret OKUNMAZ")
# AKICILIK'in beklenen cevabi OLMAMALI: gozle okunur, puanlanmaz.
ok(all(x[2] is None for x in _P if x[0].startswith("AKICILIK")),
   "AKICILIK sondalarinin BEKLENENI YOK -- gozle okunur")
ok(all(x[2] is not None for x in _P if not x[0].startswith("AKICILIK")),
   "diger dokuz sondanin beklenen cevabi VAR")

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
