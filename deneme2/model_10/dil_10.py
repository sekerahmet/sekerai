# -*- coding: utf-8 -*-
"""dil_10 — DIL ALANI: modelin EK KURALINI ogrenip ogrenmedigi.

    python dil_10.py <kosu klasoru> [--adim N] [--n 300]

Kullanici, 19 Eylul: *"doğru bir dil ölçümü mü bu?"*  Hayirdi. Mevcut
olculerin hepsi OLGU uzerinden geciyordu; bu dosya KURALI olcer.

==========================================================================
AYRIM: KURAL vs OLGU

    KURAL   ek uyumu, cumle kaliplari, semanin izin verdigi iliskiler
            -> URETILEBILIR. Model kurali OGRENEBILIR.
    OLGU    "Ahmet'in kardesi kim"
            -> URETILEMEZ. Arama tablosu.

"Iyi bir dil modeli mi" sorusu KURAL hakkindadir. O yuzden olcu su
ilkeyi tutar: **OLGUYU SABITLE, KURALI DEGISTIR.**

==========================================================================
YONTEM: ASGARI CIFT (minimal pair)

Bir ada sekiz tamlayan eki eklenebilir; Turkce'de YALNIZ BIRI dogru:

    Alasehir Bilgisayar Bolumu'in / 'in / 'un / 'un / 'nin / 'nin /
    'nun / 'nun                       <- 'nun DOGRU

Iki ayri karar var ve ikisi AYRI sinaniyor:

    TAMPON   ad UNLUYLE bitiyorsa -n- girer      ('nun)
             UNSUZLE bitiyorsa girmez            ('in)
    UNLU     son unluye gore dortlu uyum         i / i / u / u

!! UZUNLUK YANLILIGI: 'in' 2 karakter, 'nin' 3. Uzun dizgenin olasiligi
tek basina daha dusuktur. O yuzden ASIL SAYI `unlu` sutunudur -- orada
dort aday da AYNI UZUNLUKTA. `tam(8)` sutunu raporlanir ama bu yanliligi
tasir.

==========================================================================
UC MARUZIYET SEVIYESI -- asil olcu FARK, mutlak deger DEGIL

    GRAF        gercek varliklar        korpusta ONLARCA kez
    RED-EGITIM  uydurma ad, reddetme cumlelerinde     1-2 kez
    TUTULAN     uydurma ad, egitimde HIC YOK          0 kez

Kural ogrenildiyse UCU DE AYNI cikar. Duserse ezber vardir, ve
DUSUSUN SEKLI maruziyetle olcuyu verir.

!! SINIRI DA YAZILI: uydurma adlarin SON KELIMESI ("Bolumu", "Tezi",
"Yilmaz") korpusta GECIYOR -- parcalardan birlestiriliyorlar. Yani bu
test "tam dizgeyi ezberledin mi" degil, "SONUNA BAKIP KURALI
UYGULUYOR MUSUN" diye sorar. Gercek ama DAR bir iddia.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import os
import sys

import numpy as np
import torch

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

for _akis in (sys.stdout, sys.stderr):
    try:
        _akis.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

import korpus_10 as KOR                                       # noqa: E402
import metin_10 as MT                                         # noqa: E402
import veri_10 as V                                           # noqa: E402

# !! `pencere`, `taban` ve `model` BURADA ICE AKTARILMAZ -- `main()`in
# icinde aktarilir. Sebep: `taban_10` bu dosyayi ice aktariyor (WUG
# olcumu `egit()`in olcum noktasinda kosuyor) ve ucu de taban'a geri
# baglaniyor, yani ust duzeyde DONGU olurdu. Olcum cekirdegi
# (`_puan`, `ek_sinavi`, `wug_*`) yalniz `V` ve `MT`ye dayanir; ikisi
# de yaprak modul.

SUS = lambda *a, **k: None

# --- UNLU OZELLIKLERI. `--unlu` kipi bunlari kullanir.
U4 = {"i": "i", "ı": "ı", "u": "u", "ü": "ü",
      "e": "i", "a": "ı", "o": "u", "ö": "ü"}
ON = frozenset("eiöü")          # ince  (front)
YUV = frozenset("oöuü")         # yuvarlak (rounded)
UNLULER = "aeıioöuü"


def _son_unlu(s):
    for c in reversed(s.lower()):
        if c in U4:
            return c
    return None


def wug_alfabesi(S):
    """Uydurma kokun kurulacagi harfler -- SOZLUKTEN turer, elle YAZILMAZ.

    Ilk surumde `_SESSIZ` elle yaziliydi, icinde 'j' vardi ve buyuk
    harfe cevrilince "J" cikti: korpusta J HIC gecmiyor, sozlukte de
    yok, `kodla` patladi. Uydurma kok de OLAN karakterlerden kurulur;
    yoksa test dilin DISINA cikar."""
    _kucuk = [c for c in S.harf if c.islower()]
    sessiz = "".join(c for c in _kucuk if c not in U4 and c.upper() in S.harf)
    unlu = "".join(c for c in U4 if c in S.harf)
    assert len(sessiz) >= 10 and len(unlu) == 8, (
        f"wug alfabesi yetersiz: {sessiz!r} / {unlu!r}")
    return sessiz, unlu


def maruziyet(X, S) -> collections.Counter:
    """Egitim tensorunu COZUP ' takilarini ONUNDEKI SON UNLUYE gore sayar.

    TIP degil JETON: bir ad 300 cumlede geciyorsa 300 sayilir -- modelin
    gordugu sey budur. Metin `taban.egitim_havuzu`nun dondurdugu
    tensorden cozuluyor, yani EGITIMIN TA KENDISI (kapi §0b)."""
    metin = "\n".join(S.coz(r) for r in X)
    say = collections.Counter()
    for i, c in enumerate(metin):
        if c != "'":
            continue
        j = i - 1
        while j >= 0 and metin[j] not in " \n":
            j -= 1
        u = _son_unlu(metin[j + 1:i])
        if u is not None:
            say[u] += 1
    return say


def wug_kok(hedef_unlu, uyumlu, sessiz, unlu, rw):
    """SON UNLUSU `hedef_unlu` olan uydurma kok.

    uyumlu=True  -> butun unluler AYNI incelik sinifinda (dogal Turkce)
    uyumlu=False -> onceki unluler TERS sinifta -- dogal Turkce'de YOK,
                    ve tam bu yuzden degerli: model SON unluye mi
                    bakiyor, yoksa kelimeye geneline mi?"""
    ince = hedef_unlu in ON
    havuz = [c for c in unlu if (c in ON) == (ince if uyumlu else not ince)]
    n_h = int(rw.integers(2, 4))
    w = ""
    for h in range(n_h):
        w += sessiz[int(rw.integers(len(sessiz)))]
        w += hedef_unlu if h == n_h - 1 else havuz[int(rw.integers(len(havuz)))]
        if h < n_h - 1 and rw.random() < 0.5:
            w += sessiz[int(rw.integers(len(sessiz)))]
    if rw.random() < 0.5:                      # yarisi unsuzle bitsin
        w += sessiz[int(rw.integers(len(sessiz)))]
    return w[0].upper() + w[1:]


def wug_havuzu(S, n, tohum=7):
    """DENGELI wug havuzu: her unluden esit, yarisi uyumlu yarisi uyumsuz.

    `--unlu` kipi bunu kova kova ayirarak kullanir; ana tablo ve
    `taban_*.egit()` icindeki olcum ise havuzun tamamini tek sayi
    olarak okur. UCU DE AYNI URETECI kullanir."""
    sessiz, unlu = wug_alfabesi(S)
    rw = np.random.default_rng(tohum)
    kova, gor = max(1, n // (len(unlu) * 2)), set()
    out = []
    for u in unlu:
        for uyumlu in (True, False):
            k = 0
            while k < kova:
                w = wug_kok(u, uyumlu, sessiz, unlu, rw)
                if w in gor:
                    continue
                gor.add(w)
                out.append(w)
                k += 1
    return out

def unlu_raporu(net, S, X, dev, n_kova=80, tohum=11):
    """SON UNLU basina dogruluk x KORPUS MARUZIYETI x uyum.

    Kullanici, 19 Eylul: *"hatta dogru sekilde nasil olceriz kafa yor"*.
    `ek_sinavi` "kurali ogrendi mi" diye soruyor; bu kip "NEREDE
    ogrendi, nerede ogrenemedi" diye soruyor -- ve cevabi maruziyetle
    yan yana koyuyor. Her kova ESIT n, yani kovalar kiyaslanabilir."""
    sessiz, unlu = wug_alfabesi(S)
    mar = maruziyet(X, S)
    rw = np.random.default_rng(tohum)
    print(f"  korpusta ' takisi: {sum(mar.values()):,} jeton")
    for ek_ad, ek_tab, ek_fn in (("tamlayan  -in/-nin", V.EK_NIN, MT._nin),
                                 ("bildirme  -dir/-tir", V.EK_DIR, MT._dir)):
        print(f"\n{'=' * 78}\n  {ek_ad}    adaylar: {list(ek_tab)}")
        print(f"  {'son unlu':<9}{'sinif':>6}{'ozellik':>12}"
              f"{'UYUMLU':>9}{'UYUMSUZ':>9}{'HEPSI':>9}{'maruziyet':>12}")
        kar = collections.Counter()
        satir = []
        for u in UNLULER:
            ozl = ("ince" if u in ON else "kalin") + \
                  ("+yuv" if u in YUV else "+duz")
            d_uy = ek_sinavi(net, S, [wug_kok(u, True, sessiz, unlu, rw)
                                      for _ in range(n_kova)], dev,
                             ek_tab, ek_fn)
            d_uz = ek_sinavi(net, S, [wug_kok(u, False, sessiz, unlu, rw)
                                      for _ in range(n_kova)], dev,
                             ek_tab, ek_fn)
            kar += d_uy["kar"] + d_uz["kar"]
            hep = (d_uy["unlu"] + d_uz["unlu"]) / 2
            satir.append((u, d_uy["unlu"], d_uz["unlu"], hep, mar[u]))
            print(f"  {u:<9}{U4[u]:>6}{ozl:>12}{d_uy['unlu']:>9.4f}"
                  f"{d_uz['unlu']:>9.4f}{hep:>9.4f}{mar[u]:>12,}")
        print(f"  {'-' * 76}")
        _uy = float(np.mean([r[1] for r in satir]))
        _uz = float(np.mean([r[2] for r in satir]))
        print(f"  {'ORTALAMA':<27}{_uy:>9.4f}{_uz:>9.4f}"
              f"{(_uy + _uz) / 2:>9.4f}{sum(mar.values()):>12,}")
        print(f"  sans 0.2500   n = {n_kova} x 8 kova x 2 = {n_kova * 16}")
        # --- YUVARLAK vs DUZ: hipotez, model [+yuvarlak] OZELLIGINI
        #     dusuk maruziyette guvenli kuramamis olabilir.
        _y = float(np.mean([r[3] for r in satir if r[0] in YUV]))
        _d = float(np.mean([r[3] for r in satir if r[0] not in YUV]))
        print(f"  YUVARLAK (o o u u) {_y:.4f}   DUZ (a e i i) {_d:.4f}"
              f"   fark {_y - _d:+.4f}")
        if kar:
            _ky = sum(n for (a, b), n in kar.items()
                      if _son_unlu(a) in YUV and _son_unlu(b) not in YUV)
            _kf = sum(n for (a, b), n in kar.items()
                      if _son_unlu(b) in YUV and _son_unlu(a) not in YUV)
            _t = sum(kar.values())
            print(f"  HATA {_t}: yuvarlak KAYIP {_ky}, yuvarlak FAZLA "
                  f"{_kf}, yalniz incelik {_t - _ky - _kf}")
            for (a, b), n in kar.most_common(6):
                print(f"    {a:<8} -> {b:<8} {n:>4}")
    print(f"\n{'=' * 78}\n  MARUZIYET (jeton, ' oncesi son unlu):")
    _t = max(1, sum(mar.values()))
    for u in UNLULER:
        print(f"    {u}  {mar[u]:>8,}   {mar[u] / _t * 100:>5.2f}%")




@torch.no_grad()
def _puan(net, S, onek: str, adaylar: list, dev) -> list:
    """Her aday icin log P(aday | onek). Toplam, ortalama DEGIL.

    Aday dizgeleri FARKLI UZUNLUKTA olabilir; cagiran taraf bunu
    biliyor ve esit uzunluktaki gruplari AYRICA kiyasliyor."""
    j0 = S.kodla(onek)
    out = []
    diz = [j0 + S.kodla(a) for a in adaylar]
    n = max(len(d) for d in diz)
    X = np.zeros((len(diz), n), np.int64)
    for i, d in enumerate(diz):
        X[i, :len(d)] = d
    xb = torch.from_numpy(X).to(dev)
    lg = net(xb).float().log_softmax(-1)
    for i, d in enumerate(diz):
        # aday karakterleri: poz len(j0) .. len(d)-1
        t = 0.0
        for p in range(len(j0), len(d)):
            t += float(lg[i, p - 1, d[p]])
        out.append(t)
    return out


def ek_sinavi(net, S, adlar: list, dev, ek_tab, ek_fn, bs=64) -> dict:
    """Asgari cift: her ad icin 8 aday, hangisine en yuksek olasilik.

    Doner: tam8 / unlu / tampon oranlari.
      tam8    sekiz aday arasinda argmax DOGRU mu     (uzunluk yanli)
      tampon  -n- var/yok kararini dogru verdi mi     (4'e karsi 4)
      unlu    DOGRU tampon grubunun icinde, dort esit
              uzunluktaki adaydan dogrusunu sectI mi  <- ASIL SAYI
    """
    # !! EV KURALI: `olcme_*.olc` ve `durustluk_olc` ikisi de
    # `onceki = model.training` / `model.train(onceki)` yapiyor. Bu
    # fonksiyon `egit()`in olcum noktasindan da cagriliyor, yani
    # EGITIMIN ORTASINDA. Su an modelde dropout KATMANI YOK ve fark
    # etmiyor; ama tek bir olcum yolunun kurali atlamasi, dropout
    # acildigi gun SESSIZCE gurultulu bir sayi uretirdi.
    onceki = net.training
    net.eval()
    n_t8 = n_un = n_tp = 0
    kar = collections.Counter()          # (dogru ek, modelin sectigi)
    for ad in adlar:
        # WUG kokleri graf adi DEGIL -- `_tr` cevirisi GEREKMEZ.
        # Alt cizgi varsa graf adidir, yoksa zaten Turkce yazim.
        A = MT._tr(ad) if "_" in ad else ad
        dogru = ek_fn(A, True)                # "'nun" gibi, tirnak DAHIL
        adaylar = ["'" + e for e in ek_tab]
        p = _puan(net, S, A, adaylar, dev)
        # 1) sekiz aday arasinda argmax
        n_t8 += (adaylar[int(np.argmax(p))] == dogru)
        # 2) TAMPON: ilk dort tamponsuz, son dort tamponlu
        yari = len(ek_tab) // 2
        tamponlu_mu = dogru[1:] in ek_tab[yari:]
        n_tp += (max(p[yari:]) > max(p[:yari])) == tamponlu_mu
        # 3) UNLU: DOGRU tampon grubunun icinde, dort ESIT uzunluk
        g = slice(yari, None) if tamponlu_mu else slice(0, yari)
        alt = adaylar[g]
        _sec = alt[int(np.argmax(p[g]))]
        n_un += (_sec == dogru)
        if _sec != dogru:
            kar[(dogru, _sec)] += 1
    net.train(onceki)
    n = max(1, len(adlar))
    return dict(n=len(adlar), tam8=n_t8 / n, tampon=n_tp / n, unlu=n_un / n,
                kar=kar)


def main():
    import pencere_10 as P                                        # noqa: E402
    import taban_10 as M                                          # noqa: E402
    from model_10 import ModelSade                                # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--adim", type=int, default=None)
    ap.add_argument("--n", type=int, default=300, help="grup basina ad")
    ap.add_argument("--hepsi", action="store_true",
                    help="BUTUN anlik goruntulerde olc -- EGRI")
    ap.add_argument("--unlu", action="store_true",
                    help="SON UNLU basina coz: dogruluk x MARUZIYET")
    ap.add_argument("--n_kova", type=int, default=80,
                    help="--unlu kipinde unlu basina kok (80 yetiyor)")
    a = ap.parse_args()

    ayar = P.ayar_oku(a.klasor)
    snap = P.anlik_goruntuler(a.klasor)
    assert snap, f"anlik goruntu YOK: {a.klasor}"
    v = M.veri_kur(ayar, yaz=SUS)
    _X, S = M.egitim_havuzu(ayar, v, yaz=SUS)
    G = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    net = ModelSade(ayar, S.vocab).to(M.DEV)

    # --- UC MARUZIYET GRUBU
    rs = np.random.default_rng(0)
    bl = KOR.reddetme_bolme(G, ayar.ret_tut, ayar.veri_tohum)
    graf = [x for t in V.TIPLER for x in G["ad"][t]]
    red = sorted(x for L in bl["ad_egitim"].values() for x in L)
    tut = sorted(x for L in bl["ad_tut"].values() for x in L)
    sec = lambda L: [L[int(i)] for i in rs.permutation(len(L))[:a.n]]
    # --- WUG: HIC VAR OLMAYAN KOK -----------------------------------
    # !! UC GRUP DA 1.0000 CIKTI ve bu SUPHE UYANDIRDI. Sebep aciktir:
    # uydurma adlar PARCALARDAN birlestiriliyor, yani SON KELIME
    # ("Bolumu", "Yilmaz", "Tezi") korpusta ONLARCA kez geciyor. Model
    # tam dizgeyi bilmese de SON KELIMEYI eslestirip dogru eki secebilir.
    #
    # WUG TESTI o bosluğu kapatir (dilbilimin klasik yontemi): kok
    # TAMAMEN uydurma, hicbir parcasi korpusta YOK. Turkce ses
    # duzenine uyuyor ama Turkce bir kelime DEGIL. Model dogru eki
    # secebiliyorsa kurali GERCEKTEN ogrenmis demektir -- eslestirecek
    # hicbir sey yok.
    #
    # Dogru cevap `veri_*._ek_nin` ile HESAPLANIYOR (son unlu + son
    # harf sesli mi), tablodan bakilmiyor -- yani uydurma kok icin de
    # tanimli.
    # !! ALFABE SOZLUKTEN TURER, elle yazilmaz. Ilk surumde
    # `_SESSIZ = "bcdfghjklmnprstvyz"` yaziliydi ve buyuk harfe
    # cevrilince "J" cikti -- korpusta J HIC GECMIYOR, sozlukte
    # de yok, `kodla` patladi. Uydurma kok de OLAN karakterlerden
    # kurulmali; yoksa test dilin DISINA cikar.
    # !! TEK URETEC. Burada bir zamanlar AYRI bir satir ici dongu vardi
    # ve `wug_kok` ile ayni seyi baska turlu yapiyordu: unluleri
    # TAMAMEN rastgele seciyordu, yani son unlu dagilimi da rastgeleydi.
    # Iki tanim ayni kavrami olcunce biri otekinden KAYAR. `wug_kok`
    # kaldi cunku son unluyu SABITLEYEBILIYOR -- `--unlu` kipinin ve
    # `egit()` icindeki olcumun ihtiyaci bu.
    #
    # DENGELI: her unluden esit sayida, yarisi uyumlu yarisi uyumsuz.
    # Rastgele karisim `i` ve `a`ya bogulurdu (korpusta %41 ve %27) ve
    # zayif kovalari (`o`, `ö`) gorunmez yapardi.
    wug = wug_havuzu(S, a.n, tohum=7)
    gruplar = [("GRAF        (onlarca kez)", sec(graf)),
               ("RED-EGITIM  (1-2 kez)", sec(red)),
               ("TUTULAN     (HIC)", sec(tut)),
               ("WUG  (kok UYDURMA)", wug)]

    if a.unlu:
        _ad = a.adim if a.adim is not None else max(snap)
        net.load_state_dict(torch.load(snap[_ad], map_location=M.DEV))
        net.eval()
        print(f"=== dil_10  --unlu   {ayar.ad} t{ayar.tohum}  "
              f"adim {_ad} ===")
        print(f"  egitim tensoru {tuple(_X.shape)}   sozluk {S.vocab}")
        unlu_raporu(net, S, _X, M.DEV, a.n_kova)
        return

    adimlar = sorted(snap) if a.hepsi else [
        a.adim if a.adim is not None else max(snap)]
    print(f"=== dil_10  EK UYUMU / asgari cift   {ayar.ad} t{ayar.tohum} ===")
    print(f"  sozluk {S.vocab}   sans: tam8 {1/8:.3f}  unlu {1/4:.3f} "
          f" tampon {1/2:.3f}")
    print("  !! ASIL SAYI `unlu` -- dort aday ESIT UZUNLUKTA.")
    print("     `tam8` uzunluk yanliligi tasiyor ('in' 2, 'nin' 3 karakter).")
    for ek_ad, ek_tab, ek_fn in (("tamlayan  -in/-nin", V.EK_NIN, MT._nin),
                                 ("bildirme  -dir/-tir", V.EK_DIR, MT._dir)):
        print(f"\n  --- {ek_ad} ---")
        print(f"  {'adim':>7}  {'grup':<26}{'n':>5}{'unlu':>9}"
              f"{'tampon':>9}{'tam8':>9}")
        for adim in adimlar:
            net.load_state_dict(torch.load(snap[adim], map_location=M.DEV))
            net.eval()
            _son = {}
            for g_ad, L in gruplar:
                d = ek_sinavi(net, S, L, M.DEV, ek_tab, ek_fn)
                _son[g_ad] = d
                print(f"  {adim:>7}  {g_ad:<26}{d['n']:>5}{d['unlu']:>9.4f}"
                      f"{d['tampon']:>9.4f}{d['tam8']:>9.4f}")
            _a = _son["GRAF        (onlarca kez)"]["unlu"]
            _b = _son["WUG  (kok UYDURMA)"]["unlu"]
            print(f"  {'':>7}  {'FARK graf - WUG':<26}{'':>5}"
                  f"{_a - _b:>+9.4f}   (graf - WUG)")
            if abs(_a - _b) < 0.05:
                print(f"  {'':>7}  -> KURAL. Gormedigi adda da ayni.")
            else:
                print(f"  {'':>7}  -> EZBER payi VAR: gormedigi adda "
                      f"{_b:.4f}, gordugunde {_a:.4f}")
            print()
    print("  !! SINIR: uydurma adlarin SON KELIMESI korpusta geciyor")
    print("     (parcalardan birlestiriliyorlar). Bu test 'tam dizgeyi")
    print("     ezberledin mi' degil, 'SONUNA BAKIP KURALI UYGULUYOR")
    print("     MUSUN' diye sorar. Gercek ama DAR bir iddia.")


if __name__ == "__main__":
    main()
