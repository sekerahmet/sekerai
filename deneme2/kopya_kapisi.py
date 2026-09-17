# -*- coding: utf-8 -*-
"""KOPYA KAPISI — bir kol klasoru ebeveyninden GERI KALMIS mi?

    python deneme2/kopya_kapisi.py model_07 model_06

!! BU ARAC BIR KAZADAN DOGDU (kullanici, 17 Eylul 2026).

`model_07/`, `model_06/`dan kopyalandi. Ama kopya TEK BIR ANDA
alinmadi -- dosyalar 20:09 ile 20:57 arasinda tek tek tasindi, ve o
arada `model_06` calismaya devam etti. Sonuc: BES duzeltme sessizce
eskiye dondu.

    konus.bat        hala `konus_05.py` cagiriyordu       (kullanici IKI KEZ carpti)
    konus --genislik 5 -> 1                                (arac OLCULENDEN
                                                            BASKA modeli gosteriyordu)
    konus bosluk     tek jeton yerine hepsi basiliyordu
    kos --hiz-dogrula HIC YOKTU
    GPU_HAVUZ        varsayilan 0 -> 1'e dondu

Besini de INSAN yakaladi, kod DEGIL. Ve hicbiri "hata" gibi
gorunmuyordu: kod calisiyordu, testler geciyordu, yalniz ESKIYDI.

KURAL: kol klasoru kopyalanirsa, ebeveynde OLUP kopyada OLMAYAN her
satir ya TASINIR ya da BEYAN EDILIR. Kapi ucuncu bir secenek
birakmiyor.

BEYAN DOSYASI:  <kol>/KOPYA.json

    {
      "ebeveyn": "model_06",
      "beyan": {
        "taban_07.py": "soru bicimi eklendi; ebeveynde olmayan kod VAR,
                        ebeveynden gelmeyen satir YOK diye sinandi",
        ...
      }
    }

Kapi, ebeveynde olup kopyada olmayan satirlari SAYAR. Bir dosyada
boyle satir varsa `beyan` icinde o dosya icin bir GEREKCE olmali.
Gerekce yoksa kapi DUSER ve satirlari BASAR.

!! BU BIR "DIFF YOK" TESTI DEGIL. Kolun kendi degisiklikleri
(kopyada VAR, ebeveynde YOK) hic bakilmaz -- onlar kolun TANIMI.
Bakilan tek sey: ebeveynin bildigi ama kopyanin bilmedigi sey.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys

_K = os.path.dirname(os.path.abspath(__file__))
UZANTI = (".py", ".bat")


def _oku(yol: str) -> list:
    with io.open(yol, encoding="utf-8", errors="replace") as f:
        return [x.rstrip("\r\n") for x in f]


def _cevir(satirlar: list, e_no: str, c_no: str) -> list:
    """Ebeveyn metnini KOPYANIN adlandirmasina cevirir.

    Yalniz `_NN` soneki ve `model_NN` -- baska hicbir sey. Boylece
    'ebeveynde var, kopyada yok' karari ADLANDIRMA farkindan DEGIL,
    gercek icerikten cikar."""
    a = re.compile(r"_" + e_no + r"\b")
    b = re.compile(r"\bmodel_" + e_no + r"\b")
    return [b.sub("model_" + c_no, a.sub("_" + c_no, x)) for x in satirlar]


def _onemli(s: str) -> bool:
    """Bos satir ve ayirac cizgileri GURULTU -- sayilmaz."""
    t = s.strip()
    return bool(t) and t not in ("#", '"""') and not re.fullmatch(r"#\s*-{3,}\s*", t)


def kapi(kol: str, ebeveyn: str = None, yaz=print) -> int:
    kol_y = os.path.join(_K, kol)
    assert os.path.isdir(kol_y), f"klasor YOK: {kol_y}"
    man_y = os.path.join(kol_y, "KOPYA.json")
    man = json.load(io.open(man_y, encoding="utf-8")) if os.path.exists(man_y) else {}
    ebeveyn = ebeveyn or man.get("ebeveyn")
    assert ebeveyn, (
        f"EBEVEYN BELLI DEGIL. {kol}/KOPYA.json yaz ya da komut satirinda ver.\n"
        f"  python kopya_kapisi.py {kol} <ebeveyn>")
    eb_y = os.path.join(_K, ebeveyn)
    assert os.path.isdir(eb_y), f"ebeveyn klasoru YOK: {eb_y}"
    c_no, e_no = kol.rsplit("_", 1)[1], ebeveyn.rsplit("_", 1)[1]
    beyan = man.get("beyan", {})

    yaz(f"KOPYA KAPISI   {kol}  <-  {ebeveyn}")
    yaz("=" * 74)
    eksik_dosya, bakilan = {}, 0
    for ad in sorted(os.listdir(eb_y)):
        if not ad.endswith(UZANTI):
            continue
        c_ad = ad.replace("_" + e_no, "_" + c_no)
        c_yol, e_yol = os.path.join(kol_y, c_ad), os.path.join(eb_y, ad)
        if not os.path.exists(c_yol):
            eksik_dosya[c_ad] = ["<DOSYANIN KENDISI YOK>"]
            continue
        bakilan += 1
        eb = _cevir(_oku(e_yol), e_no, c_no)
        ko = _oku(c_yol)
        # Ebeveynde OLUP kopyada OLMAYAN satirlar. Tersi BAKILMAZ.
        kalan = [x[2:] for x in difflib.ndiff(eb, ko)
                 if x.startswith("- ") and _onemli(x[2:])]
        if kalan:
            eksik_dosya[c_ad] = kalan

    if not eksik_dosya:
        yaz(f"  GECTI -- {bakilan} dosya, ebeveynde olup kopyada olmayan "
            f"satir YOK")
        return 0

    dusen = []
    yaz(f"  (sayilar KOPYA.json 'satir' tablosuna karsi kilitli)")
    for ad in sorted(eksik_dosya):
        kalan = eksik_dosya[ad]
        g = beyan.get(ad)
        # !! SAYI DA KILITLI. Beyan DOSYA basina; o yuzden beyanli bir
        # dosyada SONRADAN olusan yeni bir gerileme affedilirdi.
        # `satir` tablosu o deligi kapatiyor: sayi BUYURSE kapi duser.
        # Kucukse sorun yok (fark TASINMIS demektir) ama hatirlatilir.
        _bek = (man.get("satir") or {}).get(ad)
        if g and _bek is not None and len(kalan) > _bek:
            dusen.append(ad)
            yaz(f"  !! SAYI BUYUDU {ad:<18} {_bek} -> {len(kalan)} satir")
            yaz(f"        beyan var ama YENI bir fark olusmus:")
            for x in kalan[-6:]:
                yaz(f"        - {x[:88]}")
        elif g:
            _n = f"{len(kalan):>4}"
            if _bek is not None and len(kalan) < _bek:
                _n += f" (beyan {_bek}, DUSTU -- KOPYA.json yenilenmeli)"
            yaz(f"  BEYANLI  {ad:<22} {_n} satir   {g}")
        else:
            dusen.append(ad)
            yaz(f"  !! BEYANSIZ {ad:<20} {len(kalan):>4} satir")
            for x in kalan[:12]:
                yaz(f"        - {x[:88]}")
            if len(kalan) > 12:
                yaz(f"        ... {len(kalan) - 12} satir daha")
    yaz("=" * 74)
    if dusen:
        yaz(f"  KAPI DUSTU. Ebeveynde OLUP kopyada OLMAYAN satirlar var ve")
        yaz(f"  BEYAN EDILMEMIS: {', '.join(dusen)}")
        yaz(f"  Ya TASI, ya {kol}/KOPYA.json icinde GEREKCESINI yaz.")
        return 1
    yaz(f"  GECTI -- farklarin hepsi BEYANLI ({len(eksik_dosya)} dosya)")
    return 0


if __name__ == "__main__":
    _kol = sys.argv[1] if len(sys.argv) > 1 else None
    assert _kol, __doc__
    sys.exit(kapi(_kol, sys.argv[2] if len(sys.argv) > 2 else None))
