# -*- coding: utf-8 -*-
"""D3.3 — maskeleme kazanci phi ile hayatta kaliyor mu.

Onceden kayit: ONKAYIT_D33_PHI_EKSENI.md  (kosudan once yazildi, degismez)

    A5   maskesiz,                        phi 5.06, 0 -> 120.000
    D5   MASK_KEY=1 MASK_BLK=1..7,        phi 5.06, 0 -> 120.000
         (D3'un konfiginin AYNISI, sadece phi farkli)

Dusuk-phi karsiliklari ZATEN olculdu (A 0.0603 / D3 0.3607, oran 5.98x),
o yuzden burada sadece iki yeni kol var.

ARAMA YOK: bu deney tarifi degil, ALTINDAKI ETKIYI siniyor. Etki yoksa
tarif zaten konusuz. Arama olmayinca parcalamaya da gerek yok -> her kol
TEK surecte kosar (D3.2'de ek yukun %95 oldugu olculdu).

    python d33.py <konfig_giris.json>
"""
import os, sys, json, time, glob

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS", "WARM_OF"):
    os.environ.pop(_v, None)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      COMPILE="0", N_ENT="200", N_REL="6", N_PAIR="20",
                      P_TRAIN="16")
os.environ.update({k: v for k, v in KON["ORT"].items()
                   if k in ("PRESET", "ARMS", "SEEDS", "HOP2_FRAC", "MEM_AT",
                            "N_ENT", "N_REL", "N_PAIR", "P_TRAIN")})

import numpy as np
import torch
import sifirdan as S
from kosu import Kosu

A5, D5, K5 = "cikti_a5", "cikti_d5", "cikti_k5"

KON.update(
    DERINLIK=list(range(1, S.CFG["L"])),           # D3'un konfigi
    HEDEF_SON=120000,
    PENCERE=[60000, 65000, 70000, 75000, 80000],
    OLGUNLUK=0.50,                                 # comp(A5 @ pencere)
    MEKANIZMA=0.30,                                # kisayol(A5 @ pencere)
    TOL_YORUNGE=9.9,                               # referans kol yok
    REF_AD="A5",
    KONTROL_ALT=A5,
    # FAZ 1 yazilmali: bu deneyde maske ADIM 0'dan acik, faz gecisi YOK.
    # rapor.py fazi atesleme adimindan turetiyor; asamaB olmadigi icin
    # at=None -> f hep 1. Faz-2 yazilsaydi uyarilar HIC atesmezdi.
    UYARI={"SAGLIK":  ["comp", 0.10, -1, 1],
           "YOL":     ["ent", 0.00, -1, 1],
           "DOLANMA": ["ent_shortcut", 0.00, +1, 1]},
)
if SMOKE:
    KON.update(HEDEF_SON=600, PENCERE=[400, 600], OLGUNLUK=-1.0, MEKANIZMA=-1.0)

K = Kosu(KON)
DERINLIK = tuple(KON["DERINLIK"])
HEDEF_SON = KON["HEDEF_SON"]
MBLK = ",".join(str(b) for b in DERINLIK)

# phi'yi KODDAN hesapla, elle yazma (CLAUDE.md 6)
_d = S.build_data()
_atom = S.CFG["N_ENT"] * S.CFG["N_REL"]
PHI = len(_d[3]) / _atom
K.log(f"D3.3 basliyor  commit {KON['commit']}")
K.log(f"  veri {S.CFG['N_ENT']}x{S.CFG['N_PAIR']} ({S.CFG['P_TRAIN']} egitimde) "
      f"R={S.CFG['N_REL']}  ->  PHI = {PHI:.2f}   (dusuk-phi referans: 3.03)")
K.log(f"  atomik olgu {_atom}  egitim-2hop {len(_d[3])}  "
      f"COMP {len(_d[4])}  ENT {len(_d[5])}  ENT2 {len(_d[8])}")
K.log(f"  kollar: A5 (maskesiz) | D5 (poz 1 @ {MBLK}) | "
      f"K5 (poz 2 @ {MBLK}, KONTROL: eslesmis maliyet, yanlis yer)")
K.kaydet(rapor_ek=[f"phi = {PHI:.2f}  (dusuk-phi referans 3.03)",
                   f"A5 maskesiz | D5 poz 1 @ {MBLK} | K5 poz 2 (KONTROL)",
                   f"dusuk-phi sonuc: A 0.0603  D3 0.3607  oran 5.98x"])

# ============================================================== ANA DONGU
#   Her kol TEK surecte. Surdurme paketi varsa kaldigi yerden devam eder,
#   yani kopma halinde HUCRE 0-4 tekrar kosulunca kayip olmaz.
# K5: ESLESMIS MALIYET, YANLIS YER. Deney 7'de K kolu bir bulguyu oldurdu
# ("maskeleme ogrenmeyi hizlandiriyor") ve digerini kurtardi. CLAUDE.md 10b:
# "Kontrolu sonraya birakma. Kontrol kolu tasariminin parcasidir, ek degil."
for ad, alt, ek in (("A5", A5, None),
                    ("D5", D5, dict(MASK_KEY="1", MASK_BLK=MBLK)),
                    ("K5", K5, dict(MASK_KEY="2", MASK_BLK=MBLK))):
    if K.st.get(f"{ad}_bitti"):
        K.log(f"{ad} zaten bitmis, atlaniyor")
        continue
    K.kaydet(faz=1 if ad == "A5" else 2,
             faz_ad=f"{ad} kosuyor ({'maskesiz' if ek is None else 'maskeli'}), "
                    f"hedef {HEDEF_SON}")
    t = time.time()
    sn = K.egit(HEDEF_SON, ek, alt=alt)
    K.kaydet(**{f"{ad}_bitti": True}, adim=HEDEF_SON)
    K.log(f"{ad} bitti ({sn} sn = {sn/60:.0f} dk)")
    if ek is not None:
        K.konfig_kapisi(dict(MASK_KEY=ek["MASK_KEY"],
                             MASK_BLK=list(DERINLIK)), alt=alt)

K.kaydet(faz=3, faz_ad="BITTI")
_np = len(KON["PENCERE"])          # elle "/5" yazma yok (CLAUDE.md 6)
h = lambda a: sum(os.path.exists(
    K.y(a, f"snap_{KON['KOL']}_s{KON['SEED']}_{s:06d}.pt")) for s in KON["PENCERE"])
K.log(f"BITTI.  pencere noktalari: A5 {h(A5)}/{_np}  D5 {h(D5)}/{_np}  "
      f"K5 {h(K5)}/{_np}")

# Birincil okuma komutu — ONCEDEN YAZILAN KAPILAR da komutun icinde.
# Kapiyi sadece insan okursa unutulabilir ya da sonucu gorup gevsetilebilir.
_p = ",".join(str(s) for s in KON["PENCERE"])
_mb = f"{DERINLIK[0]}-{DERINLIK[-1]}"
_komut = [
    f"python sablon/pencere.py --cikti {K.y('BIRINCIL_D33.json')}",
    f"  --kol 'A5:{K.y(A5)}:{_p}:yok'",
    f"  --kol 'D5:{K.y(D5)}:{_p}:1@{_mb}'",
    f"  --kol 'K5:{K.y(K5)}:{_p}:2@{_mb}'",
    f"  --kapi 'A5 comp >= {KON['OLGUNLUK']}'          # OLGUNLUK",
    f"  --kapi 'A5 ent_kisayol >= {KON['MEKANIZMA']}'  # MEKANIZMA",
    f"  --kapi 'D5/A5 ent >= 3.0'                      # BIRINCIL",
    f"  --kapi 'D5/K5 ent >= 2.0'                      # YER mi MALIYET mi",
    f"  --kapi 'A5 ent < 0.18'                         # ANTITEZ (kalirsa antitez kazandi)",
    f"  --kapi 'D5 bir_hop >= 0.98'                    # SAGLIK",
]
K.log("  birincil okuma (kapilar komutun icinde):")
for _l in _komut:
    K.log("    " + _l)
open(K.y("BIRINCIL_KOMUT.sh"), "w").write((" \\\n").join(_komut) + "\n")
K.not_(f"BITTI — pencere noktalari A5 {h(A5)}/{_np} D5 {h(D5)}/{_np} K5 {h(K5)}/{_np}",
       f"phi = {PHI:.2f}   dusuk-phi referans: A 0.0603 / D3 0.3607 = 5.98x",
       f"okuma komutu: BIRINCIL_KOMUT.sh")
K.log("SURUCU BITTI")
