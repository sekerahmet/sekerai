# -*- coding: utf-8 -*-
"""tani_11 — ARIZA SEKLI: model YANLIS cevap verirken NE diyor?

    python tani_11.py <kosu klasoru> [--adim N]

`pencere_09` "ne kadari dogru" der; bu dosya "yanlislar NEYE benziyor"
der. HUKUM VERMEZ -- arizanin seklini gosterir.

    DOGRU         cikti beklenenin BIREBIR AYNISI
    YAKIN         dogru varlikla basliyor, eki/noktasi tutmuyor
    KISAYOL       facts[e, r2] -- r2 KOPRUYE degil OZNEYE uygulanmis
    KOPRU         ara varligi yazmis, ikinci hop'u YAPMAMIS
    VARLIK_BASKA  baska bir varlik adiyla basliyor
    VARLIK_DEGIL  hicbir varlik adiyla baslamiyor

KISAYOL bu projenin olctugu ana ariza; KOPRU ise "ilk hop calisti,
ikincisi calismadi" demek. Ikisi cok farkli seyler soyler ve tek bir
'yanlis' sutununda ayirt EDILEMEZLER.

!! model_08'IN YUVA AYRISTIRMASI DUSTU. Orada cevap yuva basina
kisitli argmax'la okunuyordu ve `VARLIK_DEGIL` "jeton birlesimi bir
varlik degil" demekti. Karakterde yuva yok: model ne yazarsa onu
yaziyor, ve "varlik degil" artik gercekten "hicbir adla baslamiyor".
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import korpus_11 as KOR                                       # noqa: E402
import metin_11 as MT                                         # noqa: E402
import olcme_11 as OLC                                        # noqa: E402
import pencere_11 as P                                        # noqa: E402
import taban_11 as M                                          # noqa: E402
import veri_11 as V                                           # noqa: E402
from model_11 import ModelSade                                # noqa: E402

MODEL_SINIFI = ModelSade
SUS = lambda *a, **k: None
KATEGORI = ("DOGRU", "YAKIN", "KISAYOL", "KOPRU", "VARLIK_BASKA",
            "VARLIK_DEGIL")


def _adlar(G):
    return [MT.ad_tr(a) for t in V.TIPLER for a in G["ad"][t]]


def _en_uzun_onek(ad_sirali, c):
    """Ciktinin basladigi VARLIK. Uzun adlar ONCE denenir.

    !! SIRA SART. "Aydin" bir SEHIR ve ayni zamanda "Mehmet Aydin"in
    oneki; kisa adlar once denenirse "Mehmet Aydin'dir." ciktisi SEHIR
    Aydin diye etiketlenirdi. Olculdu: 48 boyle cift var."""
    for a in ad_sirali:
        if c.startswith(a):
            return a
    return None


def ayristir(G, lst, sv: OLC.Sinav, cikti):
    """Her ornegi KATEGORI'lerden birine koyar."""
    E = _adlar(G)
    ad_sirali = sorted(set(E), key=len, reverse=True)
    out = []
    for z, bek, cev, ksy, c in zip(lst, sv.bek, sv.cev, sv.ksy, cikti):
        if c == bek:
            out.append("DOGRU"); continue
        if c.startswith(cev):
            out.append("YAKIN"); continue
        if ksy and c.startswith(ksy):
            out.append("KISAYOL"); continue
        # `E` ZATEN Turkce (`_adlar` ad_tr uyguluyor) -- ikinci kez
        # uygulamak "Ibrahim Yilmaz"i bozardi.
        kop = E[int(z[3])] if len(z) > 3 else None
        if kop and c.startswith(kop):
            out.append("KOPRU"); continue
        out.append("VARLIK_BASKA" if _en_uzun_onek(ad_sirali, c)
                   else "VARLIK_DEGIL")
    return out


def ozet(ad, kat, yaz=print) -> dict:
    n = max(1, len(kat))
    say = collections.Counter(kat)
    pay = {k: say.get(k, 0) / n for k in KATEGORI}
    yaz(f"  {ad:<10}{n:>6,}" + "".join(f"{pay[k]:>14.4f}" for k in KATEGORI))
    return pay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--adim", type=int, default=None)
    ap.add_argument("--ornek", type=int, default=4,
                    help="kategori basina kac ornek basilsin")
    ap.add_argument("--biyografi", type=int, default=160,
                    help="kac varligin biyografisi uretilsin (0 = atla)")
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    adim = a.adim if a.adim is not None else max(snap)
    assert adim in snap, f"adim {adim} YOK: {sorted(snap)}"
    print(f"=== tani_11  {ayar.ad} t{ayar.tohum}  adim {adim} ===")
    print("  HUKUM VERMEZ -- arizanin SEKLI.")

    v = M.veri_kur(ayar, yaz=SUS)
    G = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    # !! `KOR.havuz` DOGRUDAN CAGRILMAZ -- `M.egitim_havuzu(ayar, v)`.
    # Onceki hali `KOR.havuz(v, G, ayar.kopya, ayar.t_len, ...)` idi ve
    # 10 parametrenin 6'sini geciyordu: `zincir_pay`, `n3`, `ret_pay`,
    # `ret_tut` VARSAYILANA dusuyordu. `n3` egitimde 10.000, burada 0 --
    # yani okuma araci EGITILEN KORPUSTAN BASKA bir korpus kuruyordu.
    # Sozluk ayni cikabilir (ayni harfler) ama `korpus_izi` tutmaz ve
    # hicbir sey bunu SOYLEMEZDI. Tek yol: ayardan turet.
    _X, S = M.egitim_havuzu(ayar, v, yaz=SUS)
    L = M.olcme_listeleri(ayar, v)
    net = (MODEL_SINIFI or M.Model)(ayar, S.vocab).to(M.DEV)
    net.load_state_dict(torch.load(snap[adim], map_location=M.DEV))
    net.eval()

    print(f"\n  {'bolme':<10}{'n':>6}" + "".join(f"{k:>14}" for k in KATEGORI))
    sonuc, ornekler = {}, {}
    for k in ("seen", "comp", "ent", "ent_yok", "ood"):
        if not len(L.get(k, ())):
            continue
        sv = OLC.Sinav(S, v, G, L[k], 2, k)
        c = OLC.uret(net, S, sv.X, sv.bas, sv.n_uret, M.DEV)
        kat = ayristir(G, L[k], sv, c)
        sonuc[k] = ozet(k, kat)
        ornekler[k] = kat, sv, c

    for k in ("comp", "ent"):
        if k not in ornekler:
            continue
        kat, sv, c = ornekler[k]
        print(f"\n  --- {k} ORNEKLERI ---")
        for et in ("KISAYOL", "KOPRU", "VARLIK_BASKA", "VARLIK_DEGIL"):
            g = [i for i, x in enumerate(kat) if x == et][:a.ornek]
            for i in g:
                print(f"    {et:<13}BEKLEN {sv.bek[i]:<28}MODEL {c[i]}")

    # --- BIYOGRAFI OKUMASI --------------------------------------------
    # Tek olgulu sinav "cevabi biliyor mu" der. Bu BASKA bir sey sorar:
    # varligin BUTUN bilgisi erisilebilir mi, ve model onu KENDI kurdugu
    # bir sirayla mi veriyor yoksa egitim belgesini geri mi oynatiyor.
    #
    # !! YALNIZ COK OLGULU VARLIKLAR. Olculdu: 782 varligin 2 olgusu var
    # (sehir, tez, bolum...) ve orada "siralama" diye bir soru YOK --
    # iki olgunun iki sirasi var, ikisi de egitimde. Karisik olcmek
    # `yeni_sira`yi seyreltirdi. Kisiler 9-11 olgulu, soru ORADA anlamli.
    biyo = None
    if a.biyografi:
        _ait = {}
        for _e, _r, _x in v.one:
            _ait.setdefault(int(_e), []).append(1)
        _cok = [e for e in sorted(_ait) if len(_ait[e]) >= 5][:a.biyografi]
        if _cok:
            # !! `KOR.biyografiler` ARTIK YOK -- model_11'da `sayfalar`
            # oldu. Bu satir kopyadan geldi ve `--biyografi`nin
            # VARSAYILAN yolu oldugu icin `tani_11` HER KOSUDA
            # AttributeError ile duserdi. §0c kapisi artik bunu
            # statik olarak denetliyor.
            _bb, _ = KOR.sayfalar(v, G, kopya=ayar.kopya,
                                  tohum=ayar.veri_tohum, tetik=ayar.tetik,
                                  t_len=ayar.t_len,
                                  zincir_pay=ayar.zincir_pay,
                                  n3=ayar.n3, yaz=SUS)
            _egt = OLC.egitim_siralari(v, G, _bb)
            # !! t_len AYARDAN. Varsayilani 640'ti ve RoPE tablosu
            # `ayar.t_len` (512) uzunlugunda kuruluyor -- 640'lik
            # dizi tablonun DISINA tasardi. model_09'da tam bu
            # yuzden biyografi olcusu hic calismadi.
            _X, _bas, _hed = OLC.biyografi_sinavi(S, v, G, _cok,
                                                  t_len=ayar.t_len)
            biyo = OLC.biyografi_olc(net, S, _X, _bas, _hed,
                                     [_egt.get(e, set()) for e in _cok],
                                     M.DEV)
            print(f"\n  === BIYOGRAFI ({len(_cok)} varlik, >=5 olgulu) ===")
            print(f"    recall    {biyo['recall']:.4f}   "
                  "varligin olgularinin kaci SOYLENDI")
            print(f"    kesinlik  {biyo['kesinlik']:.4f}   "
                  "soylenen cumlelerin kaci GERCEK olgu")
            print(f"    yeni_sira {biyo['yeni_sira']:.4f}   "
                  "EGITIMDEKI belgelerden HICBIRI degil")
            print(f"    cumle     {biyo['cumle']:.1f}    uretilen cumle sayisi")
            print("    !! yeni_sira DUSUKSE model belgeyi GERI OYNATIYOR;")
            print("       YUKSEKSE olgulari kendi siralayabiliyor demek.")
            for _o in biyo["ornek"][:2]:
                print(f"    URETTI  {_o[:200]}")

    yol = os.path.join(a.klasor, f"tani_{ayar.ad}_t{ayar.tohum}_{adim}.json")
    if biyo:
        biyo = {k: x for k, x in biyo.items() if k != "ornek"}
    json.dump(dict(adim=adim, kategori=list(KATEGORI), bolme=sonuc,
                   biyografi=biyo),
              open(yol, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {yol}")


if __name__ == "__main__":
    main()
