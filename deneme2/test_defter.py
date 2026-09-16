# -*- coding: utf-8 -*-
"""test_defter — DEFTERLER SABLONA PARALEL MI?

    python test_defter.py            butun aile defterleri
    python test_defter.py model_b6   tek defter

CLAUDE.md: "Defterler sablona paralel gider... yalniz 0. ve 2. hucrede
ayrilir." Uygulamada kol defterleri iki hucrede daha ayriliyor: 8 (kolun
sartlari) ve 14 (kiyas tabani). Bu dosya bunu ELLE degil MEKANIK olarak
denetler.

NEDEN VAR (olculdu, 16 Eylul): her kol bir oncekinden uretiliyordu ve
duzeltmeler BIRIKTI --

    model_b4.ipynb   `B1 = dict(...)` 2 kez
    model_b5.ipynb   3 kez
    model_b6.ipynb   3 kez, ve rapor tablosunun basligi `model_b5`

Uc tanesi de calisiyordu (son atama kazaniyor), yani hicbir hata
vermiyordu -- yalniz YANLIS KOL ADI basiyordu. Sessiz bozulma; kullanici
sordu, ben bakmamistim.

HUKUM VEREN kontroller (dusurur):
    * ayni kiyas tabani birden cok kez yaziliyor
    * bir KOD hucresinde BASKA bir kolun adi geciyor
Geri kalani RAPOR: kosmus kollarin defteri KAYITTIR, geriye donuk
duzeltilmez.
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

KOK = os.path.dirname(os.path.abspath(__file__))
SABLON = os.path.join(KOK, "SABLON.ipynb")
# Sablondan ayrilmasi BEKLENEN hucreler ve sebepleri.
BEKLENEN = {0: "baslik", 2: "MODEL satiri",
            8: "kolun sartlari", 14: "kiyas tabani"}


def _oz(c):
    return "".join(c["source"])


def _taban_oku(py):
    """Kolun TABANI: <ad>.py icindeki `from model_bX import ... TABAN`
    ya da `TABAN = model_bX.AYAR`. Elle liste tutmuyoruz -- taban zaten
    kodda YAZILI, oradan okunur."""
    if not os.path.exists(py):
        return set()
    s_ = io.open(py, encoding="utf-8").read()
    return set(re.findall(r"from\s+(model_[a-z]\d*)\s+import.*?TABAN", s_))         | set(re.findall(r"TABAN\s*=\s*(model_[a-z]\d*)\.", s_))


def denetle(yol, T, yaz=print):
    ad = os.path.basename(yol)[:-6]
    N = json.load(io.open(yol, encoding="utf-8"))["cells"]
    kotu = []
    not_ = []
    if len(N) < len(T):
        not_.append(f"sablondan KISA ({len(N)} < {len(T)})")
    elif len(N) > len(T):
        not_.append(f"{len(N) - len(T)} ek hucre (sonda)")
    ort = min(len(N), len(T))
    tip = [i for i in range(ort) if N[i]["cell_type"] != T[i]["cell_type"]]
    if tip:
        kotu.append(f"hucre TIPI sablondan farkli: {tip}")
    farkli = [i for i in range(ort) if _oz(T[i]) != _oz(N[i])]
    fazla = [i for i in farkli if i not in BEKLENEN]
    if fazla:
        not_.append(f"beklenmeyen hucrede farkli: {fazla}")
    # --- HUKUM VEREN 1: tekrarlanmis kiyas tabani
    for i, c in enumerate(N):
        s = _oz(c)
        if c["cell_type"] == "code" and s.count("B1 = dict(") > 1:
            kotu.append(f"{i}. hucrede kiyas tabani {s.count('B1 = dict(')} "
                        "kez -- onceki koldan DEVRALINMIS")
    # --- HUKUM VEREN 2: kod hucresinde AYNI AILEDEN baska kolun adi
    # Serbest olanlar, ve NEDEN:
    #   sablonun kendi metni   -> defterin kusuru degil
    #   ailenin taban kolu     -> kiyas tabani, yorumda adi GECER
    #   BASKA ailenin kolu     -> capraz kiyas (model_b1 <-> model_a8)
    # Yakalanan tam olarak sudur: model_b6'nin kodunda "model_b5".
    # AILE ADI: harfli ailelerde `model_<harf>` (model_b15 -> model_b);
    # RAKAMLI ailede ad zaten TEK BASINA ailedir (model_00). Eskiden
    # regex yalniz [a-z] tutuyordu ve `model_00` eklenince bu satir
    # AttributeError ile DUSTU -- 16 Eylul.
    _m = re.match(r"(model_[a-z])", ad)
    aile = _m.group(1) if _m else ad
    # TABAN kolun adi serbest -- ve KOLUN KENDI .py'sinden okunur, elle
    # yazilmaz. model_b1 hepsinin tabani DEGIL: model_b7'nin tabani
    # model_b6. Sabit "aile+1" kurali dogru defteri BOZUK gosteriyordu.
    taban = _taban_oku(os.path.join(os.path.dirname(yol), ad + ".py"))
    serbest = {ad, aile + "1"} | taban
    for i, c in enumerate(N):
        if c["cell_type"] != "code":
            continue
        sab = _oz(T[i]) if i < len(T) else ""
        # KOLUN KENDI ADI bu hucrede geciyorsa, baska kol adlari
        # REFERANSTIR (kiyas tabani, tavan) ve sorun DEGIL. Yakalamak
        # istedigim hata suydu: model_b6'nin rapor tablosu kendi adi
        # YERINE `model_b5` yaziyordu -- yani kol KENDINI etiketlemiyordu.
        # Ilk kural "taban disinda hicbir ad" diyordu ve model_b9'u
        # (tavan referansi model_b8) YANLIS yere BOZUK gosterdi.
        kendi = ad in _oz(c)
        for m in set(re.findall(aile + r"\d+", _oz(c))):
            if m not in serbest and m not in sab and not kendi:
                kotu.append(f"{i}. hucrede KOLUN KENDI ADI YOK ama "
                            f"{m} var")
    yaz(f"  {ad:<12} {'BOZUK' if kotu else ' ok  '}  "
        + ("; ".join(not_) if not_ else "sablonla birebir"))
    for k in kotu:
        yaz(f"     !! {k}")
    return kotu


def main():
    T = json.load(io.open(SABLON, encoding="utf-8"))["cells"]
    hedef = sys.argv[1:]
    yollar = sorted(glob.glob(os.path.join(KOK, "model_*", "*.ipynb")))
    if hedef:
        yollar = [y for y in yollar
                  if os.path.basename(y)[:-6] in hedef]
        if not yollar:
            print(f"!! defter bulunamadi: {hedef}")
            return 1
    print(f"=== DEFTERLER SABLONA PARALEL MI ({len(T)} hucre) ===")
    n = 0
    for y in yollar:
        n += len(denetle(y, T))
    print(f"\n{len(yollar)} defter, {n} HUKUM VEREN kusur")
    if not hedef:
        # ARGUMANSIZ = RAPOR KIPI, cikis 0. Kosmus kollarin defteri
        # KAYITTIR; geriye donuk duzeltilmez, ama GORUNUR kalir.
        # (model_b4 ve model_b5 bu yuzden BOZUK gorunur ve oyle KALIR.)
        print('  (rapor kipi -- hukum icin kol adi ver: '
              'python test_defter.py model_b6)')
        return 0
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
