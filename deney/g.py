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

# BUTCE -- adim sayisi. Varsayilan 80.000; ONCEDEN 120.000 idi.
# OLCULDU (G kolu, 14 Eylul): phi 5.09'da BUTUN egriler 50.000 adimda
# doyuyor.  50.000 -> 75.000 arasi degisim:
#     comp +0.007   ent 0.000   entyok +0.006   kisayol +0.006
# Yani 70.000 adim daha kosup hicbir sey degismiyor; iki kol icin
# ~140.000 adim ~= 78 dk GPU. D3.3'te de ayni desen vardi (comp 20.000'de
# doymus, pencere yine 60-80k'ya konmustu).
# UZATMA KULLANICI KARARIDIR: HUCRE 0'daki ORT'ta BUTCE degistirilir.
# Bu kosu ICINDE kendiliginden uzamaz.
BUTCE = int(os.environ.get("BUTCE", "80000"))
_E = int(S.CFG["EVERY"])
assert BUTCE % _E == 0, f"BUTCE ({BUTCE}) EVERY'nin ({_E}) kati olmali"
# PENCERE = son %20, olcum noktalarina oturtulmus (onkayit 5.1).
PENCERE = [BUTCE - 4 * _E + i * _E for i in range(5)]

KON.update(
    DERINLIK=list(range(1, S.CFG["L"])),
    HEDEF_SON=BUTCE,
    KOL_SAYISI=2,
    RAPOR_TOPLAM=2 * BUTCE,
    PENCERE=PENCERE,
    PHI_BEKLENEN=5.09,
    OLGUNLUK=0.50,          # comp(G @ pencere) -- gorev ogrenilmis mi
    MEKANIZMA=0.30,         # BILGI: D3.3'un tasinmis esigi, DURDURMAZ
    SANS_KAT=10,            # ON KAPI: kisayol >= SANS_KAT x sans (onkayit 5.0)
    TOL_YORUNGE=9.9,
    REF_AD="G",
    KONTROL_ALT=G_,
    UYARI={"SAGLIK":  ["comp", 0.10, -1, 1],
           "YOL":     ["ent", 0.00, -1, 1],
           "DOLANMA": ["ent_shortcut", 0.00, +1, 1]},
)
if SMOKE:
    KON.update(RAPOR_TOPLAM=2 * 600, HEDEF_SON=600, PENCERE=[400, 600],
               OLGUNLUK=-1.0, SANS_KAT=0, PHI_BEKLENEN=None)
    # SANS_KAT=0: kapi KOSAR (pencerede() sinanir) ama gecer -> duman testi
    # FAZ 3 ve 4'e de ulasir. MEKANIZMA=-1 yapsaydik kapi blogu TUMUYLE
    # atlanirdi; 0.0 yapsaydik gercek esik (0.05) yine durdururdu -- ilk
    # denemede tam oyle oldu ve arama/GM yolu sinanmadan kaldi.

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

# --- SANS SEVIYESI ---------------------------------------------------------
# Birincil olcu bir ORAN (GM/G). Paydasi sans seviyesine yapisiksa oran
# etkiyi degil GURULTUYU olcer: G 0.004'e duserse oran sisirilir, G tam
# sansta ise oran 1'e cakilir. D3.3'te maskesiz kolun ENT'i 0.0090 idi;
# BURADA sans 0.005 civari, yani oyle bir deger sansin ancak 1.8 kati.
# Onkayitta boyle bir taban YOKTU -- eklendi (kosudan ONCE).
_G0 = _VM_G = None
import veri_okul as _VOK
_gg = _VOK.kur()
_tipsay = {t: len(_gg["ad"][t]) for t in _VOK.TIPLER}
_E = [a for t in _VOK.TIPLER for a in _gg["ad"][t]]
def _sans(lst):
    if not lst:
        return None
    return sum(1.0 / _tipsay[_gg["tip"][_E[a]]] for *_, a in lst) / len(lst)
SANS_AYIRT, SANS_YOK = _sans(_d[5]), _sans(S.ENT_YOK)
K.log(f"  SANS SEVIYESI  ENT-AYIRT {SANS_AYIRT:.5f} (1/{1/SANS_AYIRT:.0f})   "
      f"ENT-YOK {SANS_YOK:.5f} (1/{1/SANS_YOK:.0f})")
KON["SANS_AYIRT"], KON["SANS_YOK"] = SANS_AYIRT, SANS_YOK
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
# ESIK TASINDI, ONU BILEREK KULLANIYORUZ. 0.30 D3.3'un RASTGELE grafindaki
# A5'ten geliyor; bu veri farkli ve CLAUDE.md 10b tam olarak "bir esik baska
# bir rejime tasinmaz" diyor. O yuzden DURDURAN esik 0.30 DEGIL, SANSIN 10
# KATI: mekanizmanin VAR olup olmadigini sorar. 0.30 yalniz "D3.3 ile ayni
# guclulukte mi" diye BILGI olarak basilir.
DUR_ESIK = KON["SANS_KAT"] * SANS_AYIRT
K.log(f"ON KAPI  G'nin ENT-AYIRT kisayol orani = "
      f"{'yok' if ksy is None else f'{ksy:.4f}'}")
K.log(f"  durduran esik  {DUR_ESIK:.4f}  "
      f"(sansin {KON['SANS_KAT']} kati, sans {SANS_AYIRT:.5f})")
K.log(f"  bilgi: D3.3'un tasinmis esigi 0.30 -> "
      f"{'ustunde' if (ksy or 0) >= 0.30 else 'ALTINDA (ayni guclulukte degil)'}")
if KON["SANS_KAT"] >= 0:
    assert ksy is not None, "G egrisi yok -> on kapi olculemez"
    if ksy < DUR_ESIK:
        K.kaydet(faz=9, faz_ad="DURDU: ON KAPI GECILEMEDI")
        K.log(f"  !! kisayol {ksy:.4f} < {DUR_ESIK:.4f} -> bu veride "
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

# EKSIK KONTROL: YANLIS YERE maskelenmis kol (D3.3'teki K5) G'de de YOK.
# d33.py bunu acikca logluyor; G'nin onkaydinda yazili degildi.
# SONUC: GM > G cikarsa "maskeleme yardim etti" DENIR, ama "YER belirleyici"
# DENEMEZ -- kazanc yerden mi maliyetten mi, bu deney soyleyemez.
# (phi 3.03'te D/K = 8.1x olculmustu; bu veride olculmemis olacak.)
K.log("!! KONTROL KOLU YOK (yanlis yere maskeli). 'Kazanc YERDEN mi "
      "MALIYETTEN mi' sorusu bu deneyde CEVAPSIZ kalir.")
K.kaydet(rapor_ek=["KONTROL KOLU YOK: 'dogru YERE' kaydi bu deneyde "
                   "SINANMIYOR (D3.3'te K5 de yoktu)"])
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
       "--kapi", f"ON-KAPI-kisayol-var: G ent_kisayol >= {DUR_ESIK:.4f}",
       # BIRINCIL IKI PARCALI. Oran tek basina yeterli DEGIL: payda sans
       # seviyesine yapisiksa (G ~ 0.005) oran gurultuyu buyutur. FARK kapisi
       # paydadan bagimsiz ve ayni hukmu verir. IKISI DE gecmeli.
       "--kapi", "BIRINCIL-5.1a-oran: GM/G ent >= 2.0",
       f"--kapi", f"BIRINCIL-5.1b-fark: GM-G ent >= {2*SANS_AYIRT:.4f}",
       # Oranin okunabilir olmasi icin PAYDA sansin ustunde olmali.
       # Gecmezse oran "ATLANMAZ" ama hukum HATALI OKUNUR -> etiketle.
       f"--kapi", f"PAYDA-SAGLIGI: G ent >= {2*SANS_AYIRT:.4f}",
       "--kapi", "MEKANIZMA-5.2-kisayol-dustu: GM-G ent_kisayol < 0.0",
       # 5.3 cebirsel sadelesme: (GM.ent/G.ent)/(GM.ent_yok/G.ent_yok)
       #                       = (GM.ent/GM.ent_yok)/(G.ent/G.ent_yok)
       "--kapi", "GOMULU-KONTROL-5.3: GM/G ent_bolu_yok > 1.0",
       # 5.4'un AMACI "GM'de cokmus mu" -- bu bir KIYAS. Mutlak kapi bunu
       # test etmiyordu: ikisi birden 0.93'te olsa GM'e haksizca "maske
       # kapasiteyi yedi" denirdi. FARK asil kapi, mutlak olan saglik kontrolu.
       "--kapi", "OZGULLUK-5.4a-fark: GM-G seen >= -0.02",
       "--kapi", "SAGLIK-seen-G: G seen >= 0.90",
       "--kapi", "SAGLIK-seen-GM: GM seen >= 0.90",
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
# BAYRAK DISINDAKI HER SEY TIRNAKLANIR. Onceki kosul "bosluk ya da <> varsa"
# idi ve yollar disarida kaliyordu: Windows'ta uretilen
# C:\AI_NEW_MODEL\sablon\pencere.py bash'te "\s" -> "s" diye cozulup
# C:AI_NEW_MODELsablonpencere.py oluyordu. Colab'da yollar egik cizgili
# oldugu icin orada gorunmezdi -- yani YEREL'de denenmedikce yakalanmazdi.
_t = lambda x: x if x.startswith("--") else '"' + x + '"'
_KOMUT = _ONEK + " python " + _t(_YOL) + " " + " ".join(_t(x) for x in ARG)
K.log("BIRINCIL OKUMA:" + chr(10) + "  " + _KOMUT)
json.dump(dict(onek=_ONEK, arg=ARG, komut=_KOMUT),
          open(K.y("BIRINCIL_KOMUT_G.json"), "w"), indent=1)
