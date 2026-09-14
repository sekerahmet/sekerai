# -*- coding: utf-8 -*-
"""DENEY H — maske DOZ-YANITI: kac blok maskelenirse ne oluyor?

Onceden kayit: belge/onkayit/ONKAYIT_H_MASKE_DOZ.md (kosudan ONCE yazildi)

NEDEN SIMDI ANLAMLI. 14 Eylul, TEK BLOK taramasi (TEK_BLOK.txt, G kolu,
cikarim ani): kacis yolu BLOK 0'DA. Bloklar 1-7'nin hicbiri tek basina
kisayola dokunmuyor (d(ksy) -0.019 .. +0.057); blok 0'da kesince
0.3135 -> 0.0365. Ve butun kosularda MASK_BLK=1..7, yani blok 0 HIC
maskeli degildi.
    -> Kuyruk maskesi kacis yoluna YAPISAL OLARAK ulasamiyor.
    -> Yine de GM, G'den +0.0917 iyi (20/20 pencere).
    -> Yani bu kosuda olculecek her fayda, TANIM GEREGI kisayol
       bastirmasi DEGILDIR. Geriye "genel kisitlama" hipotezi kaliyor
       ve doz-yanit onun ilk gercek sinavi.

DORT DOZ NOKTASI, IKISI ZATEN VAR:
    0 blok   maske yok        G    VAR   (120.000 kostu)
    1 blok   MASK_BLK=7       H7   YENI
    4 blok   MASK_BLK=4,5,6,7 H4   YENI
    7 blok   MASK_BLK=1..7    GM   VAR   (120.000 kostu)
MASK_KEY=1 dordunde de ayni. Degisen tek sey blok sayisi.

WARM_OF=120000 ZORUNLU, 80000 DEGIL. Isinma `max(10, WARM_OF//20)`
(sifirdan.py:1276) -> 6000, G/GM'nin AYNISI. 80000 verilseydi 4000 olur
ve yeni kollar G/GM'nin yorungesinden ayrilirdi; §6'nin WARM_OF maddesi
tam bu hatadan yazilmisti.

    python h.py <konfig_giris.json>
"""
import os, sys, json

KON = json.load(open(sys.argv[1]))
sys.path.insert(0, KON["KOD"])
sys.path.insert(0, os.path.join(KON["KOD"], "sablon"))

for _v in ("MASK_KEY", "MASK_BLK", "RESUME_FROM", "INIT_FROM", "OUT",
           "SHARE", "IDENT_MODE", "STEPS"):
    os.environ.pop(_v, None)

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    # VERI bayragi KALIR: duman testi de BIZIM veriyi kossun.
    KON["ORT"] = dict(KON["ORT"], PRESET="smoke", MEM_AT="2", EVERY="200",
                      COMPILE="0", VERI="okul", WARM_OF="")
assert not (set(KON["ORT"]) & {"MASK_KEY", "MASK_BLK", "RESUME_FROM",
                               "INIT_FROM", "OUT", "STEPS"}), \
    f"ORT kosuya ozel degisken tasiyor: {KON['ORT']}"
os.environ.update({k: v for k, v in KON["ORT"].items() if v != ""})

import sifirdan as S
from kosu import Kosu

BUTCE = int(os.environ.get("BUTCE", "80000"))
_E = int(S.CFG["EVERY"])
assert BUTCE % _E == 0, f"BUTCE ({BUTCE}) EVERY'nin ({_E}) kati olmali"
PENCERE = [BUTCE - 4 * _E + i * _E for i in range(5)]

# DOZ NOKTALARI -- onkayit 4. Sira: cok maskeliden az maskeliye DEGIL,
# ucundan ucuna: once 4 blok, sonra 1 blok. Ikisi de yeni.
KOLLAR = [("H4", "4,5,6,7", "cikti_h4"),
          ("H7", "7",       "cikti_h7")]

KON.update(DERINLIK=list(range(1, S.CFG["L"])), HEDEF_SON=BUTCE,
           KOL_SAYISI=len(KOLLAR), RAPOR_TOPLAM=len(KOLLAR) * BUTCE,
           PENCERE=PENCERE)

K = Kosu(KON)
K.log(f"DENEY H basliyor   butce {BUTCE}   pencere {PENCERE}")
K.log(f"  WARM_OF={os.environ.get('WARM_OF')}  -> isinma "
      f"{max(10, int(os.environ.get('WARM_OF') or BUTCE) // 20)} "
      f"(G/GM ile AYNI olmali: 6000)")
K.not_(f"doz-yanit: 0 blok (G, var) / 1 blok (H7) / 4 blok (H4) / 7 blok (GM, var)",
       f"birincil: ENT, pencere {PENCERE}, her kol KENDI maskesiyle",
       "kapilar: H-1 monotonluk, H-2 esik, H-3 ters  (ONKAYIT_H_MASKE_DOZ.md)")

for ad, blk, alt in KOLLAR:
    if K.bitti_mi(ad, alt, PENCERE):
        K.log(f"{ad} ZATEN BITTI, atlaniyor")
        continue
    K.kaydet(faz=1, faz_ad=f"{ad} kosuyor (MASK_KEY=1 @ {blk}), hedef {BUTCE}")
    sn = K.egit(BUTCE, dict(MASK_KEY="1", MASK_BLK=blk), alt=alt)
    _b = sum(1 for x, _, _ in KOLLAR if K.st.get(f"{x}_bitti"))
    K.kaydet(**{f"{ad}_bitti": True}, adim=(_b + 1) * BUTCE)
    K.log(f"{ad} bitti ({sn} sn = {sn/60:.0f} dk)")

K.kaydet(faz=3, faz_ad="BITTI")
K.log("DENEY H TAMAM. Olcum: sablon/pencere.py, dort kol, pencere "
      f"{PENCERE}. Kapilar ONKAYIT_H_MASKE_DOZ.md 6'da.")
