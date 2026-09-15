# -*- coding: utf-8 -*-
"""kos.py — defterin BASLATTIGI surec.  Modele ait DEGIL, altyapi.

    python kos.py --model model_a --ev <cikti koku> --tohum 0 [--commit X]
                  [--ustune]

Cikti:  <ev>/t<tohum>/     her tohum KENDI klasorune yazar.

--------------------------------------------------------------------------
NEDEN AYRI DOSYA

Eskiden defter bu betigi KENDI ICINDE metin olarak kuruyordu:

    kos = "\\n".join(['import sys; sys.path.insert(0, %r)' % ...,
                      'import %s as M' % MODEL, ...])
    open("/content/kos.py", "w").write(kos)

Yani kosuyu fiilen baslatan kod DEPODA DEGILDI. Gozden gecirilemiyordu,
test edilemiyordu, commit'e girmiyordu ve `%`/tirnak kacislari sessizce
bozulabilirdi. Simdi klonla birlikte geliyor: neyi kosturdugun, kosan
kodun kendisiyle AYNI commit'te.
"""
from __future__ import annotations

import argparse, glob, importlib, os, sys

KOK = os.path.dirname(os.path.abspath(__file__))       # deneme2/


def aile_yolu(model: str) -> str:
    """`<model>.py` hangi AILE klasorunde?  deneme2/model_a/model_a1.py gibi.

    Klasor adini isimden TURETMIYORUZ (model_a1 -> model_a gibi bir kural
    sessizce yanlis klasoru secebilirdi). Dosyayi ARIYORUZ ve tam bir tane
    bulmasini SART kosuyoruz."""
    aday = glob.glob(os.path.join(KOK, "*", f"{model}.py"))
    assert len(aday) == 1, (
        f"{model}.py deneme2/*/ altinda TAM BIR KEZ bulunmali, "
        f"{len(aday)} bulundu: {aday}")
    return os.path.dirname(aday[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="orn. model_a")
    ap.add_argument("--ev", required=True, help="cikti koku; t<N>/ altina yazar")
    ap.add_argument("--tohum", type=int, nargs="+", default=[0])
    ap.add_argument("--commit", default=None,
                    help="defterin klonladigi commit; kunyeye yazilir")
    ap.add_argument("--ustune", action="store_true",
                    help="DOLU klasorun uzerine yaz (varsayilan: REDDET)")
    a = ap.parse_args()

    sys.path.insert(0, aile_yolu(a.model))
    M = importlib.import_module(a.model)
    # Cikti dosyalari `AYAR.ad` ile adlandirilir. Dosya adi ile AYAR.ad
    # ayrilirsa dosya adi YALAN SOYLER: model_b.py, snap_model_a_*.pt yazar.
    # Arsivde tam bu oldu -- butun kollar `snap_A_s0_*.pt` yaziyordu.
    assert M.AYAR.ad == a.model, (
        f"AD TUTMUYOR: dosya {a.model}.py ama AYAR.ad {M.AYAR.ad!r}. "
        "Kopyalanan bir model dosyasinda `ad` degistirilmemis olabilir "
        "(deneme2/ISIMLENDIRME.md).")

    print(f"kos.py   model {a.model}   ev {a.ev}   tohum {a.tohum}   "
          f"commit {a.commit or '(yerel)'}", flush=True)
    for t in a.tohum:
        M.egit(M.AYAR.degistir(tohum=t), alt=f"{a.ev}/t{t}",
               ustune=a.ustune, commit=a.commit)


if __name__ == "__main__":
    main()
