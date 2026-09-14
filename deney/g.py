# -*- coding: utf-8 -*-
"""DENEY G — maskeleme kazanci BASKA BIR VERI URETECINDE de duruyor mu.

Onceden kayit: ONKAYIT_G_GERCEK_VERI.md  (kosudan once yazildi, degismez)

    G    maskesiz,                         0 -> 120.000
    GM   maskeli, YERI ARAMA BELIRLER,     0 -> 120.000

Bugune kadarki 92 eslestirilmis noktanin HEPSI tek bir veri uretecinden
geldi: `sifirdan.build_data()`'nin rastgele grafi. Orada kisayol f(e,r2)
HER ZAMAN bir cevap uretebiliyordu. G, elde yapilmis TIPLI bir grafta
(veri_okul.py) ayni soruyu soruyor; orada ENT sorularinin %19'unda kisayol
TIP OLARAK imkansiz -> gomulu kontrol.

PROSEDUR D3.1'IN AYNISI (yer VARSAYILMAZ):
    1. G kosar.
    2. ON KAPI: G'nin ENT-AYIRT kisayol orani >= 0.30 mu? Degilse bu veride
       bastirilacak bir mekanizma YOK, GM anlamsiz -> DUR.
    3. ARAMA: Asama A + B, G'nin DISKTEKI anlik goruntulerinde.
       Asama B'nin kazanan skoru POZITIF degilse SECIM YOK -> DUR.
       (CLAUDE.md 10b: esik baska bir rejime TASINMAZ; phi 5.06'da tetik
        ilk durakta ateslemis ama kazanan -0.0010 vermisti.)
    4. GM, bulunan (parca, bloklar) ile ADIM 0'dan kosar.

    python g.py <konfig_giris.json>
"""
import os, sys, json, time, glob, subprocess

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS", "WARM_OF"):
    os.environ.pop(_v, None)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    # VERI bayragi KALIR: duman testi de BIZIM veriyi kossun, yoksa
    # test ettigimiz sey deneyin kendisi olmaz. Kucultme ENT_PAY/adim ile.
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      COMPILE="0", VERI="okul")
assert not (set(KON["ORT"]) & {"MASK_KEY", "MASK_BLK", "RESUME_FROM",
                               "INIT_FROM", "OUT", "STEPS"}), \
    f"ORT kosuya ozel degisken tasiyor: {KON['ORT']}"
os.environ.update(KON["ORT"])

import numpy as np
import torch
import sifirdan as S
from kosu import Kosu

G_, GM_ = "cikti_g", "cikti_gm"

KON.update(
    DERINLIK=list(range(1, S.CFG["L"])),
    HEDEF_SON=120000,
    KOL_SAYISI=2,
    RAPOR_TOPLAM=2 * 120000,
    PENCERE=[100000, 105000, 110000, 115000, 120000],   # onkayit 5.1: SON %20
    PHI_BEKLENEN=5.09,
    OLGUNLUK=0.50,          # comp(G @ pencere) -- gorev ogrenilmis mi
    MEKANIZMA=0.30,         # ON KAPI: kisayol(G @ pencere), onkayit 5.0
    TOL_YORUNGE=9.9,
    REF_AD="G",
    KONTROL_ALT=G_,
    UYARI={"SAGLIK":  ["comp", 0.10, -1, 1],
           "YOL":     ["ent", 0.00, -1, 1],
           "DOLANMA": ["ent_shortcut", 0.00, +1, 1]},
)
if SMOKE:
    # MEKANIZMA -1 DEGIL 0.0: -1 olsaydi ON KAPI blogu tumuyle atlanir ve
    # `pencerede()` hic kosmazdi -- duman testi tam da o yolu denemeli.
    KON.update(RAPOR_TOPLAM=2 * 600, HEDEF_SON=600, PENCERE=[400, 600],
               OLGUNLUK=-1.0, MEKANIZMA=0.0, PHI_BEKLENEN=None)

_SEC = [x.strip() for x in os.environ.get("KOLLAR", "G,GM").split(",") if x.strip()]
assert _SEC and set(_SEC) <= {"G", "GM"}, f"bilinmeyen kol: {_SEC}"
KON["KOL_SAYISI"] = len(_SEC)
KON["RAPOR_TOPLAM"] = len(_SEC) * KON["HEDEF_SON"]

K = Kosu(KON)
DERINLIK = tuple(KON["DERINLIK"])
HEDEF_SON = KON["HEDEF_SON"]
MBLK = ",".join(str(b) for b in DERINLIK)

# --- VERI KAPISI --------------------------------------------------------
# phi'yi KODDAN hesapla (CLAUDE.md 6). D3.3'teki `N_ENT * N_REL` formulu
# BURADA YANLIS OLUR: graf TIPLI, `facts` hucrelerinin yarisi bos (-1).
# Atomik olgu sayisi `one` listesinin UZUNLUGUDUR.
_d = S.build_data()
_atom = len(_d[2])
assert _atom < S.CFG["N_ENT"] * S.CFG["N_REL"], \
    "olgu sayisi tam carpim -- VERI bayragi acik mi?"
PHI = len(_d[3]) / _atom
K.log(f"DENEY G basliyor  commit {KON['commit']}")
K.log(f"  VERI={os.environ.get('VERI', '(YOK)')}  sozluk {S.VOCAB} "
      f"({S.CFG['N_REL']} iliski + {S.CFG['N_ENT']} varlik)")
K.log(f"  atomik olgu {_atom}  egitim-2hop {len(_d[3])}  COMP {len(_d[4])}  "
      f"ENT-AYIRT {len(_d[5])}  ENT-YOK {len(S.ENT_YOK)}  ->  PHI = {PHI:.2f}")
assert os.environ.get("VERI"), "VERI bayragi YOK -> eski rastgele graf kosuyor"
assert S.ENT_YOK, "ENT-YOK bos -> gomulu kontrol OLCULMEZ"

if KON.get("PHI_BEKLENEN") is not None:
    assert abs(PHI - KON["PHI_BEKLENEN"]) < 0.05, (
        f"PHI={PHI:.3f} ama onkayit {KON['PHI_BEKLENEN']} diyor -> DURDURULDU.")
    K.log(f"  PHI KAPISI GECTI   {PHI:.3f} ~ {KON['PHI_BEKLENEN']}")
K.kaydet(rapor_ek=[f"phi = {PHI:.2f}  (D3.3 = 5.06, ayni rejim)",
                   f"veri = veri_okul.py, sozluk {S.VOCAB}",
                   "kollar: " + " | ".join(_SEC),
                   "GM'nin maske yeri ARAMA ile bulunur, D3'ten alinmaz"])


def egri(alt):
    y = K.y(alt, f"egri_{KON['KOL']}_s{KON['SEED']}.json")
    return json.load(open(y)) if os.path.exists(y) else []


def pencerede(alt, alan):
    """PENCERE adimlarindaki EGRI degerlerinin ortalamasi. Birincil olcu
    DEGIL (o agirlik ortalamasi, pencere.py). Burada sadece KAPI icin."""
    c = [r[alan] for r in egri(alt)
         if r["step"] in KON["PENCERE"] and alan in r]
    return sum(c) / len(c) if c else None


# ====================================================== FAZ 1: G kolu
if "G" in _SEC and not K.bitti_mi("G", G_, KON["PENCERE"]):
    K.kaydet(faz=1, faz_ad=f"G kosuyor (maskesiz), hedef {HEDEF_SON}")
    t = time.time()
    sn = K.egit(HEDEF_SON, None, alt=G_)
    K.kaydet(G_bitti=True, adim=HEDEF_SON)
    K.log(f"G bitti ({sn} sn = {sn/60:.0f} dk)")
elif "G" in _SEC:
    K.log("G zaten bitmis (olcum noktalari diskte dogrulandi), atlaniyor")

if "GM" not in _SEC:
    K.kaydet(faz=3, faz_ad="BITTI (yalniz G istendi)")
    K.log("GM istenmedi, duruluyor."); sys.exit(0)

# ============================================ FAZ 2: ON KAPI (onkayit 5.0)
ksy = pencerede(G_, "ent_shortcut")
K.log(f"ON KAPI  G'nin ENT-AYIRT kisayol orani = "
      f"{'yok' if ksy is None else f'{ksy:.4f}'}   (esik {KON['MEKANIZMA']})")
if KON["MEKANIZMA"] >= 0:
    assert ksy is not None, "G egrisi yok -> on kapi olculemez"
    if ksy < KON["MEKANIZMA"]:
        K.kaydet(faz=9, faz_ad="DURDU: ON KAPI GECILEMEDI")
        K.log(f"  !! kisayol {ksy:.4f} < {KON['MEKANIZMA']} -> bu veride "
              f"bastirilacak bir mekanizma YOK. GM KOSULMAYACAK.")
        K.log("     Bu bir basarisizlik degil, SONUCTUR (onkayit 5.0).")
        sys.exit(0)
    K.log("  ON KAPI GECTI -> arama basliyor")

# ============================================== FAZ 3: ARAMA (D3.1 prosedürü)
# ARAMA IKI KEZ KOSAR ve SIRA ONEMLI:
#   1. KARAR turu  -- duz prosedur, ilk esik gecisinde DURUR ve secer.
#   2. TANI turu    -- ARAMA_TAM=1, her durakta Asama B, KARAR VERMEZ.
# ARAMA_TAM ile ARAMA_TABLO d33_arama.py'de BAGLI (tablo yalniz `if TAM`
# blogunda basiliyor) ve TAM modu `atesledi` bayragini HIC yazmiyor.
# Ikisini birlikte acmistim: duman testi yakaladi -- G kolu saatlerce
# kosup bitecek, arama guzel bir tablo basacak, sonra "yer bulunamadi"
# diye dusecekti. Karar turu ONCE; tani turu sonra ve BEST-EFFORT.
ARJ = os.path.join(KON["CALIS"], "ARAMA_G.json")
if not os.path.exists(ARJ):
    K.kaydet(faz=2, faz_ad="ARAMA kosuyor (Asama A + B, G'nin goruntuleri)")
    r = subprocess.run([sys.executable,
                        os.path.join(KON["KOD"], "deney", "d33_arama.py"),
                        sys.argv[1]],
                       env=dict(os.environ, ARAMA_ALT=G_,
                                ARAMA_CIKTI="ARAMA_G.json", ARAMA_TAM="0"))
    assert r.returncode == 0, f"arama basarisiz: rc={r.returncode}"
A = json.load(open(ARJ))
if not A.get("atesledi"):
    _ates = [x for x in A.get("asamaA", []) if x["sinyal"] >= A["esik"]]
    K.kaydet(faz=9, faz_ad="DURDU: ASAMA A ATESLEMEDI")
    K.log(f"  !! Asama A hicbir durakta esigi gecmedi "
          f"({len(A.get('asamaA', []))} durak bakildi, esik {A['esik']}). "
          f"GM KOSULMAYACAK.")
    assert not _ates, ("TUTARSIZ: esigi gecen durak var ama atesledi=False "
                       "-> arama TANI modunda kosmus olabilir (ARAMA_TAM)")
    sys.exit(0)
SKOR = A["kazanan"]["skor"]
K.log(f"ARAMA  parca {A['secilen_parca']}  bloklar {A['MASK_BLK']}  "
      f"skor {SKOR:+.4f}  (atesleme adimi {A['atesleme_adimi']})")

# Onkayit 4.3: SKOR POZITIF DEGILSE SECIM YOKTUR. D3.1'de kendiliginden
# pozitifti ve prosedure yazilmamisti; phi 5.06'da -0.0010 cikti.
if SKOR <= 0:
    K.kaydet(faz=9, faz_ad="DURDU: ASAMA B SKORU POZITIF DEGIL")
    K.log(f"  !! skor {SKOR:+.4f} <= 0 -> SECIM YOK. GM KOSULMAYACAK.")
    K.log("     Sonuc: 'bu veride yer ARANARAK bulunamadi' (onkayit 4.3).")
    sys.exit(0)

MK, MB = str(A["secilen_parca"]), A["MASK_BLK"]

# TANI turu: KAYIT icin, karari ETKILEMEZ. Basarisiz olursa yalniz loglanir
# (CLAUDE.md 9: "yapacagin analiz henuz icat edilmedi, buna gore kaydet").
_rt = subprocess.run([sys.executable,
                      os.path.join(KON["KOD"], "deney", "d33_arama.py"),
                      sys.argv[1]],
                     env=dict(os.environ, ARAMA_ALT=G_,
                              ARAMA_CIKTI="ARAMA_G_TANI.json",
                              ARAMA_TAM="1", ARAMA_TABLO="1"))
K.log(f"  TANI turu (24 aday x her durak) rc={_rt.returncode} "
      f"-> ARAMA_G_TANI.json" + ("" if _rt.returncode == 0 else "  !! basarisiz"))
K.kaydet(rapor_ek=[f"ARAMA: parca {MK} @ {MB}, skor {SKOR:+.4f} "
                   f"(D3/D3.1/D3.3 hep parca 1 bulmustu)"])

# ================================================== FAZ 4: GM kolu
if not K.bitti_mi("GM", GM_, KON["PENCERE"]):
    K.kaydet(faz=2, faz_ad=f"GM kosuyor (maskeli, parca {MK} @ {MB}), "
                           f"hedef {HEDEF_SON}")
    sn = K.egit(HEDEF_SON, dict(MASK_KEY=MK, MASK_BLK=MB), alt=GM_)
    K.kaydet(GM_bitti=True, adim=2 * HEDEF_SON)
    K.log(f"GM bitti ({sn} sn = {sn/60:.0f} dk)")
    K.konfig_kapisi(dict(MASK_KEY=MK,
                         MASK_BLK=[int(x) for x in MB.split(",")]), alt=GM_)
else:
    K.log("GM zaten bitmis, atlaniyor")

K.kaydet(faz=3, faz_ad="BITTI", adim=len(_SEC) * HEDEF_SON)
_np = len(KON["PENCERE"])
h = lambda a: sum(os.path.exists(
    K.y(a, f"snap_{KON['KOL']}_s{KON['SEED']}_{s:06d}.pt")) for s in KON["PENCERE"])
K.log("BITTI.  pencere noktalari: "
      + "  ".join(f"{k} {h({'G': G_, 'GM': GM_}[k])}/{_np}" for k in _SEC))

# --------- Birincil okuma komutu; ONCEDEN YAZILAN KAPILAR komutun ICINDE
_p = ",".join(str(s) for s in KON["PENCERE"])
_mb = f"{DERINLIK[0]}-{DERINLIK[-1]}"
ARG = ["--cikti", K.y("BIRINCIL_G.json"),
       "--kol", f"G:{K.y(G_)}:{_p}:yok",
       "--kol", f"GM:{K.y(GM_)}:{_p}:{MK}@{_mb}",
       # KAPI DILBILGISI: "<etiket>: <sol> <alan> <op> <sayi>" -- TAM 4 sozcuk.
       # Etikette BOSLUK, kuralda parantezli not OLAMAZ (split 4'u asar).
       # Alan adi pencere.py'nin sonuc sozlugunden gelir: EGRIDE `ent_shortcut`,
       # pencere.py'de `ent_kisayol` -- ayni buyuklugun IKI ADI var ve yanlisi
       # sessizce "ATLANDI" diye gecerdi.
       "--kapi", f"OLGUNLUK: G comp >= {KON['OLGUNLUK']}",
       "--kapi", f"ON-KAPI-kisayol-var: G ent_kisayol >= {KON['MEKANIZMA']}",
       "--kapi", "BIRINCIL-5.1-kazanc: GM/G ent >= 2.0",
       "--kapi", "MEKANIZMA-5.2-kisayol-dustu: GM-G ent_kisayol < 0.0",
       # 5.3 cebirsel sadelesme: (GM.ent/G.ent)/(GM.ent_yok/G.ent_yok)
       #                       = (GM.ent/GM.ent_yok)/(G.ent/G.ent_yok)
       "--kapi", "GOMULU-KONTROL-5.3: GM/G ent_bolu_yok > 1.0",
       "--kapi", "OZGULLUK-5.4-G: G seen >= 0.95",
       "--kapi", "OZGULLUK-5.4-GM: GM seen >= 0.95",
       "--kapi", "SAGLIK-1hop: GM bir_hop >= 0.98"]
# ORTAM ONEKI SART: pencere.py `import sifirdan` yapiyor, sifirdan CFG'yi
# ORTAMDAN kuruyor. VERI/PRESET olmadan VOCAB 1085 yerine 4016 olur ve
# BUTUN kollar "OLCULEMEDI" der -- denendi, oyle oluyor. Onceki hali oneki
# basmiyordu; komut kopyalaninca dusuyordu.
_ONEK = " ".join(f"{k}={v}" for k, v in sorted(KON["ORT"].items())
                 if k in ("VERI", "PRESET", "ENT_PAY", "COMP_PAY",
                          "MEM_AT", "N_PAIR", "P_TRAIN"))
_YOL = os.path.join(KON["KOD"], "sablon", "pencere.py")
# TIRNAK SART: kapi dizeleri BOSLUK iceriyor ve icinde ">=" var. Tirnaksiz
# basilan komut kabuga yapistirildiginda ">=" YONLENDIRME olur, "=" adinda
# dosya yaratir ve argparse'a cop gider. `" ".join(ARG)` bunu yapiyordu.
_t = lambda x: f'"{x}"' if (" " in x or ">" in x or "<" in x) else x
_KOMUT = _ONEK + " python " + _t(_YOL) + " " + " ".join(_t(x) for x in ARG)
K.log("BIRINCIL OKUMA:" + chr(10) + "  " + _KOMUT)
json.dump(dict(onek=_ONEK, arg=ARG, komut=_KOMUT),
          open(K.y("BIRINCIL_KOMUT_G.json"), "w"), indent=1)
