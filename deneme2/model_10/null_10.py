# -*- coding: utf-8 -*-
"""NULL TABANI: her sinav bolmesi APTAL bir stratejiyle ne kadar cozulur?

    python null_10.py

Kullanici, 18 Eylul: *"oncul test yaparak da dogrula, bu sekilde
hatalari tespit etmek kolay ise."*  Kolay -- ve statik denetimlerin
GOREMEDIGI seyi goruyor: bir bolme ucuz bir hileyle cozulebiliyorsa o
bolme BOZUKTUR, ve modelin oradaki basarisi bilesim degil hile olur.

`kayip_tabani_10.py` KAYBIN tabanini veriyor (kosullu entropi). Burasi
DOGRULUGUN tabanini veriyor -- ikisi farkli soru.

Hicbir strateji "anlamiyor"; hepsi ya yuzeyden ya sayimdan:

    SANS-TIP     cevabin TIPINDEN rastgele biri          1/n_tip
    EN-SIK       bolmedeki EN SIK cevap
    r2-EN-SIK    o iliskinin EGITIMDEKI en sik hedefi
    KOPYA        cevap = OZNE          (DONUS zincirlerini yakalar)
    KOPRU        cevap = KOPRU
    KISAYOL      cevap = facts[ozne, r2]   -- projenin belgeli arizasi
    SOYAD        ayni soyadli kisilerden rastgele biri (aile sizintisi)

!! `one` satirinda KISAYOL 1.0000 CIKAR ve bu DOGRUDUR: 1-hop sorusunda
`facts[e, r]` cevabin KENDISIDIR. O satir betigin calistiginin kanitidir.
`ent_yok`ta KISAYOL 0.0000 cikmalidir -- bolmenin TANIMI bu.
"""
from __future__ import annotations

import collections
import importlib
import os
import sys

_B = os.path.dirname(os.path.abspath(__file__))
if _B not in sys.path:
    sys.path.insert(0, _B)

STRATEJI = ("SANS-TIP", "EN-SIK", "r2-EN-SIK", "KOPYA", "KOPRU",
            "KISAYOL", "SOYAD")


def null(kol: str = "10", yaz=print, v=None):
    """`v` verilirse veri YENIDEN KURULMAZ -- `test_09` boyle cagiriyor."""
    V = importlib.import_module("veri_" + kol)
    M = importlib.import_module("taban_" + kol)
    A = importlib.import_module("ayar_" + kol)
    AZ = importlib.import_module("analiz_" + kol)

    if v is None:
        v = M.veri_kur(A.AYAR, yaz=lambda *a, **k: None)
    L = M.olcme_listeleri(A.AYAR, v)
    d = AZ.Dok("model_" + kol)
    ham = {i: d.ham(i) for i in range(v.n_ent)}
    tip_ad = {i: V.TIPLER[int(v.tip[i])] for i in range(v.n_ent)}
    n_tip = collections.Counter(tip_ad.values())
    soyad = {i: ham[i].split("_")[-1] for i in range(v.n_ent)}
    soyad_n = collections.Counter(soyad[i] for i in range(v.n_ent)
                                  if tip_ad[i] == "KISI")
    # r2-EN-SIK tablosu EGITIMDEN sayilir, SINAVDAN DEGIL -- yoksa hile
    # kendi cevabiyla beslenir ve taban oldugundan YUKSEK cikar.
    _c = collections.defaultdict(collections.Counter)
    for e, r, h in v.one:
        _c[r][h] += 1
    r2_sik = {r: cc.most_common(1)[0][0] for r, cc in _c.items()}

    yaz("=" * 92)
    yaz(f"NULL TABANI -- model_{kol}   (egitim YOK)")
    yaz("=" * 92)
    yaz(f"{'bolme':<9}{'n':>6}" + "".join(f"{s:>11}" for s in STRATEJI))
    yaz("-" * 92)
    en_yuksek = {}
    for nm in ("one", "seen", "comp", "ent", "ent_yok", "ent_kati", "ood"):
        lst = L.get(nm) or []
        if not lst:
            continue
        n = len(lst)
        p = dict.fromkeys(STRATEJI, 0.0)
        _en = collections.Counter(x[-1] for x in lst).most_common(1)[0][0]
        for x in lst:
            if len(x) == 3:
                e, r2, a = x
                b = None
            else:
                e, _r1, r2, b, a = x
            p["SANS-TIP"] += 1.0 / n_tip[tip_ad[a]]
            p["EN-SIK"] += (a == _en)
            p["r2-EN-SIK"] += (r2_sik.get(r2) == a)
            p["KOPYA"] += (a == e)
            p["KOPRU"] += (b is not None and a == b)
            _k = int(v.facts[e, r2])
            p["KISAYOL"] += (_k >= 0 and _k == a)
            if (tip_ad[a] == "KISI" and tip_ad[e] == "KISI"
                    and soyad[a] == soyad[e]):
                p["SOYAD"] += 1.0 / soyad_n[soyad[a]]
        yaz(f"{nm:<9}{n:>6}" + "".join(f"{p[s]/n:>11.4f}" for s in STRATEJI))
        if nm not in ("one", "seen"):
            en_yuksek[nm] = max(p[s] / n for s in STRATEJI)

    yaz("")
    yaz("HUKUM  (yalniz HUKUM VEREN bolmeler -- `one`/`seen` haric)")
    kotu = {k: x for k, x in en_yuksek.items() if x > 0.05}
    for k, x in sorted(en_yuksek.items(), key=lambda kv: -kv[1]):
        yaz(f"   {k:<10} en iyi aptal strateji {x:.4f}"
            + ("   !! %5'IN USTUNDE" if x > 0.05 else ""))
    yaz("")
    yaz("   " + ("!! BOZUK: bir bolme ucuz bir hileyle cozuluyor"
                 if kotu else
                 "GECTI -- hicbir hukum bolmesi %5'in ustunde cozulmuyor"))
    return en_yuksek


if __name__ == "__main__":
    for _k in sys.argv[1:] or ["10"]:
        null(_k)
