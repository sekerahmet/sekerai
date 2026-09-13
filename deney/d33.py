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

A5, D5 = "cikti_a5", "cikti_d5"

KON.update(
    DERINLIK=list(range(1, S.CFG["L"])),           # D3'un konfigi
    HEDEF_SON=120000,
    PENCERE=[60000, 65000, 70000, 75000, 80000],
    OLGUNLUK=0.50,                                 # comp(A5 @ pencere)
    MEKANIZMA=0.30,                                # kisayol(A5 @ pencere)
    TOL_YORUNGE=9.9,                               # referans kol yok
    REF_AD="A5",
    KONTROL_ALT=A5,
    UYARI={"SAGLIK":  ["comp", 0.10, -1, 2],
           "YOL":     ["ent", 0.00, -1, 2],
           "DOLANMA": ["ent_shortcut", 0.00, +1, 2]},
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
K.log(f"  kollar: A5 (maskesiz)  ve  D5 (poz 1, bloklar {MBLK})")
K.kaydet(rapor_ek=[f"phi = {PHI:.2f}  (dusuk-phi referans 3.03)",
                   f"A5 maskesiz  |  D5 poz 1 @ {MBLK}",
                   f"dusuk-phi sonuc: A 0.0603  D3 0.3607  oran 5.98x"])

# ============================================================== ANA DONGU
#   Her kol TEK surecte. Surdurme paketi varsa kaldigi yerden devam eder,
#   yani kopma halinde HUCRE 0-4 tekrar kosulunca kayip olmaz.
for ad, alt, ek in (("A5", A5, None),
                    ("D5", D5, dict(MASK_KEY="1", MASK_BLK=MBLK))):
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
        K.konfig_kapisi(dict(MASK_KEY="1", MASK_BLK=list(DERINLIK)), alt=alt)

K.kaydet(faz=3, faz_ad="BITTI")
h = lambda a: sum(os.path.exists(
    K.y(a, f"snap_{KON['KOL']}_s{KON['SEED']}_{s:06d}.pt")) for s in KON["PENCERE"])
K.log(f"BITTI.  pencere noktalari: A5 {h(A5)}/{len(KON['PENCERE'])}   "
      f"D5 {h(D5)}/{len(KON['PENCERE'])}")
K.log("  birincil okuma:")
K.log(f"    python sablon/pencere.py --cikti {K.y('BIRINCIL_D33.json')} \\")
K.log(f"      --kol 'A5:{K.y(A5)}:"
      + ",".join(str(s) for s in KON["PENCERE"]) + ":yok' \\")
K.log(f"      --kol 'D5:{K.y(D5)}:"
      + ",".join(str(s) for s in KON["PENCERE"])
      + f":1@{DERINLIK[0]}-{DERINLIK[-1]}'")
_np = len(KON["PENCERE"])          # elle "/5" yazma (CLAUDE.md 6)
K.not_(f"BITTI — A5 {h(A5)}/{_np}, D5 {h(D5)}/{_np} pencere noktasi hazir",
       f"phi = {PHI:.2f}   dusuk-phi referans: A 0.0603 / D3 0.3607 = 5.98x")
K.log("SURUCU BITTI")
