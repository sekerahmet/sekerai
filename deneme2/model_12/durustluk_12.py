# -*- coding: utf-8 -*-
"""durustluk_12 — model "yok" diyebiliyor mu, ve YANLIS YERE diyor mu?

    python durustluk_12.py <kosu klasoru> [--adim N] [--ornek 12]

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
egitim reddetmelerinde HIC gecmez (`korpus_12.reddetme_bolme`,
`ret_tut`). Gordugunu reddetmek EZBER, gormedigini reddetmek
GENELLEME -- olcmek istedigimiz ikincisi.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import korpus_12 as KOR                                       # noqa: E402
import olcme_12 as OLC                                        # noqa: E402
import pencere_12 as P                                        # noqa: E402
import taban_12 as M                                          # noqa: E402
from model_12 import ModelHibrit                                # noqa: E402

SUS = lambda *a, **k: None
NL_ = chr(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--adim", type=int, default=None)
    ap.add_argument("--ornek", type=int, default=12,
                    help="kac dokum satiri basilsin (0 = yalniz sayi)")
    ap.add_argument("--hepsi", action="store_true",
                    help="BUTUN anlik goruntulerde olc -- EGRI")
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adim = a.adim if a.adim is not None else max(snap)
    assert adim in snap, f"adim {adim} YOK: {sorted(snap)}"

    v = M.veri_kur(ayar, yaz=SUS)
    G = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    _X, S = M.egitim_havuzu(ayar, v, yaz=SUS)
    L = M.olcme_listeleri(ayar, v)
    net = ModelHibrit(ayar, S.vocab).to(M.DEV)
    net.load_state_dict(torch.load(snap[adim], map_location=M.DEV))
    net.eval()

    bolme = KOR.reddetme_bolme(G, ayar.ret_tut, ayar.veri_tohum)
    rs_ = OLC.RetSinavi(S, G, bolme, tohum=ayar.veri_tohum, ad="ret")
    # KACAMAK, `one` bolmesinde olculur: model bunlari BILIYOR olmali
    # (SAGLIK-1HOP kapisi >= 0.98). Bildigine "yok" diyorsa kacamaktir.
    bilgi = OLC.Sinav(S, v, G, L["one"][:2000], 1, "one")
    # --- EGRI KIPI: DURUSTLUK egitim sirasinda OLCULMEDI ama anlik
    # goruntuler duruyor, yani GERIYE DONUK cikarilabilir.
    # !! Kullanici sordu (18 Eylul): *"yani onlari raporda olcemiyor
    # muyuz?"*  Olculebiliyordu; olcum egitim dongusune KONMAMISTI.
    # Bu kip o eksigi kosuyu yeniden baslatmadan kapatiyor. Sonraki
    # kolda `egit()` icine girmeli -- o zaman canli gorulur.
    if a.hepsi:
        # !! SONUC JSON'A YAZILIR, ekrana DEGIL SADECE.
        # Kullanici: *"rapor 6 icin anlik verileri okuma olayi
        # vardi onu da ekle artik rapor duzgun ciksin."*
        # Tasarim: OLCUM yazar, RAPOR okur. Boylece 6. hucre
        # CPU ve ucuz kalir (yalniz json okur) ama butun
        # kriterler TEK YERDEN gorunur.
        _egri = []
        print(f"=== durustluk_12 EGRI  {ayar.ad} t{ayar.tohum} ===")
        print(f"  sinav: {len(rs_)} cevapsiz soru   TUTULAN veriden")
        print(f"  {'adim':>8}{'DOGRU RET':>11}{'ad':>9}{'cift':>9}"
              f"{'KACAMAK':>10}{'dogru cev':>11}")
        for _a in sorted(snap):
            net.load_state_dict(torch.load(snap[_a], map_location=M.DEV))
            net.eval()
            _d = OLC.durustluk_olc(net, S, rs_, bilgi, M.DEV)
            print(f"  {_a:>8d}{_d['dogru_ret']:>11.4f}{_d['ret_ad']:>9.4f}"
                  f"{_d['ret_cift']:>9.4f}{_d['kacamak']:>10.4f}"
                  f"{_d['dogru_cevap']:>11.4f}")
            _egri.append(dict(adim=int(_a), **{k: round(float(x), 6)
                                               for k, x in _d.items()}))
        print()
        print("  DOGRU RET tek basina OKUNMAZ: her seye 'yok' diyen model")
        print("  orada tavana cikar. KACAMAK ile BIRLIKTE okunur.")
        # --- JSON: rapor hucresi BUNU okur
        _jy = os.path.join(a.klasor, f"durustluk_t{ayar.tohum}.json")
        with open(_jy, "w", encoding="utf-8") as _f:
            json.dump(dict(ad=ayar.ad, tohum=ayar.tohum,
                           n_soru=len(rs_),
                           n_ad=sum(1 for t in rs_.tur if t == "ad"),
                           n_cift=sum(1 for t in rs_.tur if t == "cift"),
                           ret_tut=ayar.ret_tut, egri=_egri),
                      _f, ensure_ascii=False, indent=1)
        print(f"{NL_}  -> {_jy}   (6. hucre BUNU okuyor)")
        # Son anlik goruntude dokum -- gozle bakmak icin
        net.load_state_dict(torch.load(snap[max(snap)], map_location=M.DEV))
        net.eval()
        adim = max(snap)

    d = OLC.durustluk_olc(net, S, rs_, bilgi, M.DEV)

    print(f"=== durustluk_12  {ayar.ad} t{ayar.tohum}  adim {adim} ===")
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
