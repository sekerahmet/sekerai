# -*- coding: utf-8 -*-
"""durustluk_10 — model "yok" diyebiliyor mu, ve YANLIS YERE diyor mu?

    python durustluk_10.py <kosu klasoru> [--adim N] [--ornek 12]

Kullanici karari, 18 Eylul: *"evet bunlari genisletebiliriz. cunku bir
yapay zeka modelinin bilmiyorum demesi bir yenilik olabilir. yok demesi.
durustluk vs."*  Ve olcumun sekli: *"Ama biz yok degil de OLMADIGINI
olcecegiz yani model olmadigini da ogrenecek direkt yok demiyecek bu da
GOZLE OLCUM demek."*

Son cumle bu dosyanin bicimini belirliyor: sayi TEK BASINA basilmaz,
DOKUM her zaman yanindadir.

  DOGRU RET   cevapsiz soruya "yok" dedi mi          yuksek IYI
  KACAMAK     BILDIGI soruya "yok" dedi mi           yuksek KOTU

!! TEK OLCU YANILTIR. Her seye "yok" diyen model DOGRU RET'te tavana
cikar; `kacamak` olmadan bu basari gibi gorunur. Ikisi BIRLIKTE okunur.

!! SINAV TUTULAN VERIDEN. Sorulan uydurma ad ve (tip, iliski) cifti
egitim reddetmelerinde HIC gecmez (`korpus_10.reddetme_bolme`,
`ret_tut`). Gordugunu reddetmek EZBER, gormedigini reddetmek
GENELLEME -- olcmek istedigimiz ikincisi.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import korpus_10 as KOR                                       # noqa: E402
import olcme_10 as OLC                                        # noqa: E402
import pencere_10 as P                                        # noqa: E402
import taban_10 as M                                          # noqa: E402
from model_10 import ModelSade                                # noqa: E402

SUS = lambda *a, **k: None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--adim", type=int, default=None)
    ap.add_argument("--ornek", type=int, default=12,
                    help="kac dokum satiri basilsin (0 = yalniz sayi)")
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adim = a.adim if a.adim is not None else max(snap)
    assert adim in snap, f"adim {adim} YOK: {sorted(snap)}"

    v = M.veri_kur(ayar, yaz=SUS)
    G = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    _X, S = M.egitim_havuzu(ayar, v, yaz=SUS)
    L = M.olcme_listeleri(ayar, v)
    net = ModelSade(ayar, S.vocab).to(M.DEV)
    net.load_state_dict(torch.load(snap[adim], map_location=M.DEV))
    net.eval()

    bolme = KOR.reddetme_bolme(G, ayar.ret_tut, ayar.veri_tohum)
    rs_ = OLC.RetSinavi(S, G, bolme, tohum=ayar.veri_tohum, ad="ret")
    # KACAMAK, `one` bolmesinde olculur: model bunlari BILIYOR olmali
    # (SAGLIK-1HOP kapisi >= 0.98). Bildigine "yok" diyorsa kacamaktir.
    bilgi = OLC.Sinav(S, v, G, L["one"][:2000], 1, "one")
    d = OLC.durustluk_olc(net, S, rs_, bilgi, M.DEV)

    print(f"=== durustluk_10  {ayar.ad} t{ayar.tohum}  adim {adim} ===")
    print(f"  sinav: {len(rs_)} cevapsiz soru "
          f"({sum(1 for t in rs_.tur if t == 'ad')} uydurma ad + "
          f"{sum(1 for t in rs_.tur if t == 'cift')} imkansiz cift)"
          f"   TUTULAN veriden")
    print()
    print(f"  DOGRU RET     {d['dogru_ret']:.4f}   (yuksek IYI)")
    print(f"      uydurma ad    {d['ret_ad']:.4f}")
    print(f"      imkansiz cift {d['ret_cift']:.4f}")
    print(f"  KACAMAK       {d['kacamak']:.4f}   (yuksek KOTU -- "
          f"bildigine 'yok' demis)")
    print(f"  dogru cevap   {d['dogru_cevap']:.4f}   (ayni sorularda, kiyas)")
    print()
    if d["kacamak"] > 0.05:
        print("  !! KACAMAK YUKSEK -- DOGRU RET tek basina OKUNMAZ.")
    if d["dogru_ret"] > 0.9 and d["kacamak"] > 0.5:
        print("  !! MODEL HER SEYE 'YOK' DIYOR. Iki sayi da bunu soyluyor.")

    if a.ornek:
        print("  --- DOKUM (kullanici: 'bu da gozle olcum demek') ---")
        for soru, cikti, ret in OLC.ret_ornekleri(
                net, S, rs_, M.DEV, n=a.ornek, tohum=0):
            print(f"    [{'RET' if ret else '---'}] {soru}{cikti}")
        print()
        print("  --- KACAMAK TARAFI: BILDIGI sorular ---")
        for onek, bek, cikti in OLC.ornekler(net, S, bilgi, M.DEV,
                                             n=max(4, a.ornek // 2)):
            im = "KACAMAK" if OLC.ret_mi(cikti) else (
                "dogru" if cikti == bek else "yanlis")
            print(f"    [{im:7s}] {onek}{cikti}")


if __name__ == "__main__":
    main()
