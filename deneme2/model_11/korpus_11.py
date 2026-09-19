# -*- coding: utf-8 -*-
"""korpus_11 — CUMLELERDEN BELGE, BELGELERDEN PAKET. model_09'un havuzu.

Physics of LM 3.1 (2309.14316) Ek C duzeni: girdiler rastgele
ornekleniyor, BIRLESTIRILEREK sabit uzunluga paketleniyor, aralarina
<EOS>; bir dizi ya TAMAMEN bildirim ya TAMAMEN soru.

Belge VARLIK ODAKLI: tek varligin cumleleri. Kopruye ait olgular
belgede YOK -- ama PAKETLEME iki belgeyi ayni pencereye koyabilir ve
sizinti ORADA olur. `sizinti_kapisi` bu yuzden BELGEYE degil PENCEREYE
bakar.

!! PAKETLEME SABIT. Her epokta yeniden karistirilirsa tek epoktaki
kucuk cakisma olasiligi 44 epokta birikir.
"""
from __future__ import annotations

import hashlib

import numpy as np

import jeton_11 as J
import metin_11 as MT
import veri_11 as V

DUZ = tuple(range(MT.N_BILDIRIM))        # 17 bildirim kalibi
SOR = tuple(range(MT.N_SORU))            # 8 soru kalibi
EOS_IM = chr(10)          # akista <EOS> yerine gecer; sozlukte AYRI jeton


class Belge:
    """Bir varligin cumleleri + ICINDE HANGI OLGULARI SOYLEDIGI.

    `olgu` sizinti kapisinin tek dayanagi: pencere kurulunca birlesir.

    `butun=True` -> belge BOLUNEMEZ, tek pencerede durur. TETIKLENMIS
    biyografi icin sart: olculdu (18 Eylul) ki biyografilerin %44,5'i
    iki pencereye dagiliyor. Tetikleyici bir pencerede, olgularin yarisi
    otekinde kalirsa model soruyu gorup cevabin YARISINI ogrenir."""

    __slots__ = ("e", "cumle", "olgu", "butun")

    def __init__(self, e, cumle, olgu, butun=False):
        self.e, self.cumle, self.olgu, self.butun = e, cumle, olgu, butun

    @property
    def metin(self):
        return " ".join(self.cumle)


def _adlar(G):
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    return E, {a: t for t in V.TIPLER for a in G["ad"][t]}


def kopru_yasagi(v):
    """kopru -> {(e, r1)} : bu kenar O KOPRUNUN sayfasinda YAZILAMAZ.

    Tutulan her zincir (e, r1, r2, b, a) icin b'nin sayfasi hem
    (b, r2)'yi KONU olarak hem (e, r1)'i TERS IFADEYLE tasirdi --
    yani zincirin IKI KENARI ayni belgede bulusur ve bileşim
    KOPYALAMAYA duser.

    OLCULDU (18 Eylul): budanmadan tutulan 8.318 zincirin 8.318'i
    (%100) kopyalanabilir hale geliyor. Budama SART.

    !! BEDELI ONKAYITTA: budama "hangi ciftler tutuldu" bilgisini
    egitim dagilimina gomer. Model bundan CEVABI cikaramaz (atomik
    olgu kendi sayfasinda duruyor) ama istatistiksel bir imza kalir."""
    d = {}
    for lst in (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood, v.ent_kati):
        for z in lst:
            d.setdefault(int(z[3]), set()).add((int(z[0]), int(z[1])))
    return d


# --- ZINCIR BUTCESI ------------------------------------------------------
# OLCULDU 19 Eylul: `v.tr2` 29.510 zincir tasiyor ama korpus bunlarin
# ancak 6.819'unu YAZABILIYOR (iki-adimli bildirim slotu). Yani `zincir`
# olcusunun TAVANI %23,1 ve esigi 0.95 -- ULASILAMAZ bir kapiydi.
# Olculen %19,1; aradaki 4 puan ORNEKLEME ISRAFI (her kopya havuzdan
# YENIDEN cekiyordu, ayni zinciri iki kez yazip baskasini hic yazmiyordu).
#
# !! ONKAYIT §2'DE "ornekleme duzeltilirse %18,5 -> ~%60" YAZIYORDU.
#    YANLIS. O sayi bildirim + soru slotlarini AYRI saymisti; DISTINCT
#    zincir slotu 6.819. Duzeltme 4 puan verir, 40 degil.
#
# Cozum iki parcali ve ikisi AYNI karar:
#   1. `tr2` slot kadarina BUDANIR  -> `zincir` YAZILANI sorar
#   2. `sayfalar` ARTIK ORNEKLEMEZ  -> elindekinin hepsini yazar
# Korpus BUYUMEZ, yogunluk DEGISMEZ, epok DEGISMEZ. Yazilan FARKLI
# zincir 5.635 -> ~8.266 (+%47) cikar, tavan 1.00 olur.
UCLU_PAY = 0.26      # zincir slotlarinin bu kadari UC ADIMLIYA gider
#                      OLCULDU: yazilan 9.198 zincir cumlesinin 2.379'u
#                      uc adimli (%25,9). Ayar niyeti %23 idi, tutuyor.


def zincir_butcesi(one, tr2, yasak, kopya: int, zincir_pay: float,
                   uclu_pay: float = UCLU_PAY, tohum: int = 0, yaz=print):
    """`tr2`yi korpusun YAZABILECEGI kadarina budar.

    Bir varligin sayfasina `len(sat) * zincir_pay` zincir cumlesi
    giriyor ve sayfa `kopya` kez yaziliyor -> o varlik icin toplam
    `kopya * zincir_pay * len(sat)` slot. Bunun `1 - uclu_pay` kadari
    iki adimliya ayrilir.

    Secim TOHUMLU ve SIRALI: ayni ayar ayni budamayi verir."""
    konu, anilan = {}, {}
    for e, r, h in one:
        e, r, h = int(e), int(r), int(h)
        konu.setdefault(e, []).append((e, r, h))
        anilan.setdefault(h, []).append((e, r, h))
    bas = {}
    for i, x in enumerate(tr2):
        bas.setdefault(int(x[0]), []).append(i)
    rs = np.random.default_rng(4400 + tohum)
    tut, hedef_top = set(), 0
    for e, ix in sorted(bas.items()):
        ya = yasak.get(e, ())
        ge = [t for t in anilan.get(e, []) if (t[0], t[1]) not in ya]
        n_sat = len(konu.get(e, [])) + len(ge)
        if not n_sat:
            continue
        # !! round, int DEGIL. `int` her varlikta ~0,5 slot kirpar
        # ve 1.608 sayfada ~800 slot bosa giderdi.
        hedef = int(round(n_sat * zincir_pay * kopya * (1.0 - uclu_pay)))
        hedef_top += hedef
        if hedef >= len(ix):
            tut.update(ix)
            continue
        for j in rs.permutation(len(ix))[:hedef]:
            tut.add(ix[int(j)])
    out = [x for i, x in enumerate(tr2) if i in tut]
    yaz(f"  ZINCIR BUTCESI: {len(tr2):,} -> {len(out):,} "
        f"(slot {hedef_top:,}; kopya {kopya}, zincir_pay {zincir_pay}, "
        f"uclu pay {uclu_pay:.0%})")
    yaz(f"     ARTIK HEPSI YAZILIR -> `zincir` sinavi GORULEN zinciri "
        f"sorar (tavan 1.00, onceki 0.23)")
    return out


def sayfalar(v, G, kopya: int = 1, tohum: int = 0, tetik: int = 0,
             t_len: int = 512, zincir_pay: float = 0.0, n3: int = 0,
             yaz=print):
    """HER VARLIGIN kendi sayfasi. model_09'da yalniz OLGU SAHIBI vardi.

    model_09 OLCTU: kisi 10 olguda anlatiliyor, universite 3, tez 2,
    BOLGE 0 -- ve kopru basarisi bunu birebir izliyor (KISI 0.186,
    TEZ 0.000, BOLGE hic kopru olamiyor). Yani sorun yalniz "ayni olguyu
    kac bicimde soyledik" degil, "her varligi KAC cumlede anlattik".

    Sayfa IKI kaynaktan:
      KONU      (e, r) -> h        varligin KENDI olgulari
      ANILAN    (a, r) -> e        e'yi NESNE yapan olgular, TERS IFADEYLE
                                   yeniden yazilir (metin_11.TERS).
                                   Ters ifadesi olmayan iliskilerde
                                   CEVAP BASTA kalip kullanilir ki
                                   varligin adi yine BASTA olsun.

    YENI KENAR EKLENMEZ -- ayni kenar, konu ters cevrilmis. `graf_izi`
    korunur.

    `kopya` = Physics 3.1 `multiM`. `tetik` = kacinin onunde
    "X hakkinda ne biliyoruz?" olsun."""
    E, tipi = _adlar(G)
    rs = np.random.default_rng(7000 + tohum)
    yasak = kopru_yasagi(v)
    # --- ZINCIR CUMLELERI SAYFANIN ICINE ------------------------------
    # Kullanici, 18 Eylul: zincir "dilde var olan birsey ve kullaniyoruz
    # ama %70 olmasina gerek yok". Ayri akista paketlense model arka
    # arkaya sekiz tane iki-uc katli tamlama okurdu -- gercek metinde
    # zincir cumlesi BASIT cumlelerin ARASINDA gecer.
    #
    # !! YALNIZ EGITIM zincirleri (`v.tr2`). Tutulanlar zaten disarida;
    # ve zincir cumlesi KOPRUYU YAZMADIGI icin hicbir ATOMIK olguyu ele
    # vermez -> `olgu` kumesine GIRMEZ, sizinti kapisinda notrdur.
    z2 = {}
    for x in v.tr2:
        e_, r1_, r2_, _b_, a_ = (int(t) for t in x)
        z2.setdefault(e_, []).append(((V.ILISKI[r1_], V.ILISKI[r2_]), E[a_]))
    z3 = {}
    if n3:
        ix = {a_: i for i, a_ in enumerate(E)}
        for x, rr, hedef in uclu_yollar(v, G, n3, tohum, yasak):
            z3.setdefault(ix[x], []).append((rr, hedef))
    konu, anilan = {}, {}
    for e, r, h in v.one:
        e, r, h = int(e), int(r), int(h)
        konu.setdefault(e, []).append((e, r, h))
        anilan.setdefault(h, []).append((e, r, h))
    tum = sorted(set(konu) | set(anilan))
    bil, sor = [], []
    n_tet = n_parca = n_carp = n_bud = 0
    for e in tum:
        # ANILAN kenarlardan BUDANMIS liste
        ya = yasak.get(e, ())
        ge = [(a, r, h) for (a, r, h) in anilan.get(e, []) if (a, r) not in ya]
        n_bud += len(anilan.get(e, [])) - len(ge)
        sat = konu.get(e, []) + ge
        if not sat:
            continue
        # Bu sayfaya kac zincir cumlesi girsin -- ORANLA, tam sayiya
        # yuvarlamadan (0.20 x 7 cumle = 1,4 -> bazi sayfada 1, bazisinda 2).
        # !! ARTIK ORNEKLENMIYOR. `veri_kur` `tr2`yi zaten slot kadarina
        # budadi (`zincir_butcesi`), yani elde ne varsa HEPSI yazilir.
        # Eski hali her KOPYADA havuzdan yeniden cekiyordu ve ayni zinciri
        # tekrar yazip baskasini hic yazmiyordu -- olculdu: tavan %23,1
        # iken gerceklesen %19,1.
        # UC ADIMLI yollar KALAN slota sigdigi kadar girer; iki adimli
        # ONCELIKLI cunku olculen bolme o.
        _z2 = z2.get(e, [])
        _slot = int(round(len(sat) * zincir_pay * kopya))
        zin = _z2 + z3.get(e, [])[:max(0, _slot - len(_z2))]
        n_zin = 0
        if zin and zincir_pay:
            # kopyalara BOL: tavana yuvarla ki hicbiri disarida kalmasin
            n_zin = -(-len(zin) // kopya)
        _zsira = [int(i) for i in rs.permutation(len(zin))] if zin else []
        gorulen = set()
        for k in range(kopya):
            for _ in range(32):
                sira = tuple(int(i) for i in rs.permutation(len(sat)))
                yuz = tuple(int(rs.choice(DUZ)) for _ in sat)
                if (sira, yuz) not in gorulen:
                    break
                n_carp += 1
            gorulen.add((sira, yuz))
            # TIP CUMLESI ILK -- karistirilmaz. Ansiklopedi maddesi
            # gibi acilir ("Manisa bir sehirdir. Manisa'nin valisi...")
            # ve modele tutarli bir "bu NEDIR" isareti verir.
            # Kullanici karari, 18 Eylul. OLGU SOYLEMEZ -> sizinti
            # kapisina NOTR (`olgu` kumesine girmez).
            # !! TIP CUMLESI AYRI TUTULUR, `d`nin BASINA KONMAZ.
            # Ilk surumde `d = [_tc]` yazmistim ve asagidaki dongu
            # `d[j]` ile okuyordu -- indeks BIR KAYDI: tetiklenmis
            # sayfada tip cumlesi IKI KEZ basildi ve SON OLGU DUSTU.
            # Sayilarda gorunmuyordu; metne bakinca cikti.
            _tc = MT.tip_cumlesi(E[e], tipi[E[e]], k)
            # ZINCIR cumleleri: olgu cumleleriyle KARISTIRILIR.
            zc = []
            if n_zin:
                # TEK karistirma, kopyalara BLOK BLOK. Modulo ile sariyor:
                # kopya * n_zin >= len(zin) oldugundan her zincir EN AZ
                # bir kez yazilir, artan slot bastan tekrar eder (sayfa
                # yogunlugu boylece butun kopyalarda AYNI kalir).
                for _t in (_zsira[(k * n_zin + _j) % len(_zsira)]
                           for _j in range(n_zin)):
                    rr, hedef = zin[int(_t)]
                    _k = int(rs.integers(MT.N_BILDIRIM))
                    zc.append((MT.yol(E[e], rr, hedef, _k),
                               MT.yol_soru(E[e], rr, hedef, tipi[hedef],
                                           _k % MT.N_SORU)))
            d, s_, olgu = [], [], set()
            for j, i in enumerate(sira):
                o, r, h = sat[i]
                if o == e:                       # KONU -- dogrudan
                    d.append(MT.cumle(E[o], V.ILISKI[r], E[h], yuz[j]))
                else:                            # ANILAN -- ters ifade
                    t = MT.ters(E[o], V.ILISKI[r], E[h], tipi[E[h]], tipi[E[o]])
                    d.append(t if t else
                             MT.cumle(E[o], V.ILISKI[r], E[h],
                                      MT.CEVAP_BASTA[yuz[j] % len(MT.CEVAP_BASTA)]))
                s_.append(MT.soru(E[o], V.ILISKI[r], E[h], tipi[E[h]],
                                  yuz[j] % MT.N_SORU))
                olgu.add((o, r))
            # Zincirleri ARAYA serp -- sona eklemek "once olgular,
            # sonra zincirler" diye yapay bir duzen kurardi.
            for _zd, _zs in zc:
                _y = int(rs.integers(len(d) + 1))
                d.insert(_y, _zd)
                s_.insert(min(_y, len(s_)), _zs)
            # SORU sayfasi TIP SORUSUYLA acilir (bildirim sayfasi
            # tip CUMLESIYLE). Ikisi ayri yuzey, ayni bilgi.
            sor.append(Belge(e, [MT.tip_sorusu(E[e], tipi[E[e]], k)]
                             + s_, olgu))
            if k >= tetik:
                bil.append(Belge(e, [_tc] + d, olgu))
                continue
            # TETIKLENMIS: onek + olgular, t_len'e SIGACAK parcalar.
            # Sigmayan sayfa ATILMAZ, IKI CEVABA bolunur -- gercek bir
            # cevap da her zaman eksiksiz degildir.
            onek = MT.biyografi_sorusu(E[e])
            assert len(onek) + 1 + len(_tc) + 1 + max(
                len(c) for c in d) <= t_len, E[e]
            kur, n, olg = [onek, _tc], len(onek) + 1 + len(_tc), set()
            for j, i in enumerate(sira):
                c = d[j]
                if n + 1 + len(c) > t_len:
                    bil.append(Belge(e, kur, olg, butun=True))
                    n_parca += 1
                    kur, n, olg = [onek], len(onek), set()
                kur.append(c)
                n += 1 + len(c)
                olg.add((sat[i][0], sat[i][1]))
            bil.append(Belge(e, kur, olg, butun=True))
            n_tet += 1
            n_parca += 1
    _uz = [len(konu.get(e, [])) + len(
        [1 for (a, r, h) in anilan.get(e, []) if (a, r) not in yasak.get(e, ())])
        for e in tum]
    yaz(f"  sayfa {len(bil):,} bildirim + {len(sor):,} soru"
        f"   ({len(tum):,} varlik x {kopya} kopya)")
    yaz(f"     sayfa basina {min(_uz)}..{max(_uz)} cumle "
        f"(ort {sum(_uz)/len(_uz):.1f})   model_09: KONU olgusu kadardi")
    yaz(f"     BUDANDI {n_bud:,} anilan kenar (tutulan zinciri sizdirirdi)")
    if zincir_pay:
        _nz = sum(len(b.cumle) for b in bil) - sum(
            1 + len(konu.get(b.e, [])) + len(
                [1 for (a_, r_, h_) in anilan.get(b.e, [])
                 if (a_, r_) not in yasak.get(b.e, ())]) for b in bil)
        yaz(f"     ZINCIR sayfalara SERPILDI (pay {zincir_pay:.0%}, "
            f"3 adim havuzu {sum(len(x) for x in z3.values()):,})")
    if tetik:
        yaz(f"     TETIKLENMIS {n_tet:,} -> {n_parca:,} BUTUN belge")
    yaz(f"     tekil belge {len({(b.e, tuple(b.cumle)) for b in bil}):,}"
        f"/{len(bil):,}   (carpisma {n_carp:,} kez reddedildi)")
    return bil, sor


def uclu_yollar(v, G, n, tohum=0, yasak=None):
    """UC ADIMLI yollardan `n` tane ORNEKLE. (e, (r1,r2,r3), cevap).

    Grafta 262.403 uclu yol var -- hepsi KULLANILMAZ, orneklenir.
    Kopruler (b, c) tutulan bir zincirin kenarlarini tamamlamamali;
    `yasak` verilirse o yollar ATILIR."""
    import collections
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    ix = {a: i for i, a in enumerate(E)}
    ileri = collections.defaultdict(list)
    for (a_, r), h in G["olgu"].items():
        ileri[a_].append((r, h))
    rs = np.random.default_rng(7700 + tohum)
    ad = sorted(G["tip"])
    out, gor = [], set()
    dene = 0
    while len(out) < n and dene < n * 60:
        dene += 1
        x = ad[int(rs.integers(len(ad)))]
        if not ileri[x]:
            continue
        r1, b = ileri[x][int(rs.integers(len(ileri[x])))]
        if not ileri[b] or b == x:
            continue
        r2, c = ileri[b][int(rs.integers(len(ileri[b])))]
        if not ileri[c] or c in (x, b):
            continue
        r3, d = ileri[c][int(rs.integers(len(ileri[c])))]
        if d in (x, b, c):
            continue
        k = (x, r1, r2, r3)
        if k in gor:
            continue
        # TUTULAN zincirin kenarini tamamliyor mu
        if yasak and ((ix[x], V.ILISKI.index(r1)) in yasak.get(ix[b], ())
                      or (ix[b], V.ILISKI.index(r2)) in yasak.get(ix[c], ())):
            continue
        gor.add(k)
        out.append((x, (r1, r2, r3), d))
    return out


def zincirler(v, G, n2: int, n3: int = 0, tohum: int = 0, grup=6,
              yasak=None, yaz=print):
    """2 ve 3 ADIMLI zincir cumleleri. SAYIYLA, kopya ile CARPILMAZ.

    !! model_09'da `kopya` ile carpiliyordu ve korpusun %75'i zincir
    cumlesi oluyordu. Kullanici, 18 Eylul: *"cok sik olsun diye
    demedim, sadece dilde var olan birsey ve kullaniyoruz ama %70
    olmasina gerek yok"*. Dogal metinde basit cumle cogunluk, zincir
    azinlik, derinlestikce daha seyrek.

    !! Bu cumleler KOPRUYU YAZMAZ, yani hicbir ATOMIK olguyu ele
    vermezler -- `olgu` kumesi BOS ve sizinti kapisinda notrdurlar."""
    E, tipi = _adlar(G)
    rs = np.random.default_rng(7500 + tohum)
    d, s = [], []
    ix = rs.permutation(len(v.tr2))[:n2]
    for i in ix:
        e, r1, r2, _b, a = (int(x) for x in v.tr2[int(i)])
        k = int(rs.integers(MT.N_BILDIRIM))
        d.append(MT.yol(E[e], (V.ILISKI[r1], V.ILISKI[r2]), E[a], k))
        s.append(MT.yol_soru(E[e], (V.ILISKI[r1], V.ILISKI[r2]), E[a],
                             tipi[E[a]], k % MT.N_SORU))
    n_u = 0
    if n3:
        for x, rr, hedef in uclu_yollar(v, G, n3, tohum, yasak):
            k = int(rs.integers(MT.N_BILDIRIM))
            d.append(MT.yol(x, rr, hedef, k))
            s.append(MT.yol_soru(x, rr, hedef, tipi[hedef], k % MT.N_SORU))
            n_u += 1
    B = lambda c: [Belge(-1, c[i:i + grup], set())
                   for i in range(0, len(c), grup)]
    yaz(f"  zincir {len(d):,} bildirim + {len(s):,} soru"
        f"   (2 adim {len(ix):,}  3 adim {n_u:,})")
    return B(d), B(s)


def uydurma_adlar(G):
    """GRAFTA OLMAYAN ama DOGAL ad havuzu -- tip -> [ad].

    Kullanici, 18 Eylul: *"Niye hep ad ve soy ad uzerinden gidiyorsun?
    Yoksa fakulte universite vs bunlarda varlik."*  Ilk surum yalniz
    KISI uyduruyordu ve havuz 160'ta TUKENIYORDU (64 ad x 10 soyad = 640
    kombinasyonun 480'i GERCEK kisi). kopya buyudukce "varlik yok"
    cumleleri ayni 160 adi tekrarlar, reddetme fiilen KISI'ye ozel bir
    numaraya donerdi.

    YONTEM: adlar zaten BILESIK (`<SEMT>_<ALAN>_Bolumu`), ve parca
    listeleri tipler arasi AYRILMIS -- `SEMT_FAK` fakultelere,
    `SEMT_BOL` bolumlere. Parcalari CAPRAZ birlestirince hem grafta
    olmayan hem de Turkce'de kusursuz duran ad cikiyor:

        Alasehir Bilgisayar Bolumu     SEMT(fakulte yakasi) x BOL_ALAN
        Alasehir Egitim Fakultesi      SEMT(bolum yakasi)   x FAK_ALAN
        Alasehir Universitesi          SEMT                 x -
        Analitik Arastirma             DERS_ONEK            x TEZ_KOK
        Asamali Akiskanlar Tezi        TEZ_ONEK             x DERS_KOK

    !! `<SEMT>_<BOL_ALAN>_Fakultesi` KULLANILMIYOR: "Bilgisayar
    Fakultesi" diye bir sey YOK (Muhendislik Fakultesi olur). Havuzu
    320 buyuturdu ama DOGAL OLMAYAN ad uretirdi.
    !! SEHIR ve BOLGE UYDURULMAZ: adlari GERCEK cografyadan geliyor,
    bilesik degil. "X diye bir sehir yok" demek dunya hakkinda bir
    iddiadir ve bizim isimiz degil.

    !! Uretilen her ad GRAFTA OLMADIGI icin SUZULUR -- `olan` kumesi
    BUTUN tipleri kapsar, yalniz kendi tipini degil."""
    import itertools
    olan = {a for t in V.TIPLER for a in G["ad"][t]}
    P = lambda L: [x[0] for x in L]
    SEMT = P(V.SEMT)
    tarif = [
        ("KISI", (P(V.ERKEK) + P(V.KADIN), P(V.SOYAD)),
         lambda a, b: f"{a}_{b}"),
        ("BOLUM", (SEMT, P(V.BOL_ALAN)), lambda a, b: f"{a}_{b}_Bolumu"),
        ("BOLUM", (SEMT, P(V.FAK_ALAN)), lambda a, b: f"{a}_{b}_Bolumu"),
        ("FAKULTE", (SEMT, P(V.FAK_ALAN)), lambda a, b: f"{a}_{b}_Fakultesi"),
        ("UNIVERSITE", (SEMT,), lambda a: f"{a}_Universitesi"),
        ("DERS", (P(V.DERS_ONEK), P(V.TEZ_KOK)), lambda a, b: f"{a}_{b}"),
        ("TEZ", (P(V.TEZ_ONEK), P(V.DERS_KOK)), lambda a, b: f"{a}_{b}_Tezi"),
    ]
    out = {}
    for t, parts, sek in tarif:
        for c in itertools.product(*parts):
            a = sek(*c)
            if a not in olan:
                out.setdefault(t, []).append(a)
    for t in out:
        out[t] = sorted(set(out[t]))
    return out


def reddetme_bolme(G, tut_pay: float = 0.20, tohum: int = 0):
    """Reddetme verisini EGITIM / TUTULAN diye ayirir. IKI EKSENDE.

    Kullanici, 18 Eylul: *"biz yok degil de olmadigini olcecegiz yani
    model olmadigini da ogrenecek direkt yok demiyecek."*  "Olmadigini
    ogrenmek" ile "yok demeyi ogrenmek" ancak TUTULAN veriyle ayrilir:
    egitimde gordugu adi reddetmek EZBER, hic gormedigini reddetmek
    GENELLEME.

        AD    uydurma adlarin `tut_pay`'i egitimde HIC gecmez
        CIFT  IMKANSIZ ciftlerin `tut_pay`'i egitimde HIC gecmez

    Ikisi AYRI sey soruyor:
        ad tutulur   -> "bu ad listemde yok" genellesiyor mu
        cift tutulur -> "bu tip bu iliskiyi TASIMAZ" genellesiyor mu
                        (tip cumlesi + baska ciftler uzerinden)

    !! Ciftler TIP BAZINDA degil CIFT bazinda tutulur, ve her iliskiden
    en az bir cift EGITIMDE kalir -- yoksa "tutulan cifti bilemedi"
    sonucu "o iliskiyi hic gormedi" ile karisirdi."""
    rs = np.random.default_rng(9100 + tohum)
    uyd = uydurma_adlar(G)
    ad_e, ad_t = {}, {}
    for t, L in uyd.items():
        ix = rs.permutation(len(L))
        k = int(len(L) * tut_pay)
        ad_t[t] = sorted(L[int(i)] for i in ix[:k])
        ad_e[t] = sorted(L[int(i)] for i in ix[k:])
    cift_e, cift_t = [], []
    for r, ts in V.IMKANSIZ.items():
        ix = rs.permutation(len(ts))
        # en az 1 cift EGITIMDE kalir
        k = min(int(len(ts) * tut_pay), len(ts) - 1)
        cift_t += [(ts[int(i)], r) for i in ix[:k]]
        cift_e += [(ts[int(i)], r) for i in ix[k:]]
    return dict(ad_egitim=ad_e, ad_tut=ad_t,
                cift_egitim=sorted(cift_e), cift_tut=sorted(cift_t))


def reddetme_belgeleri(v, G, n, tohum=0, grup=6, bolme=None, yaz=print):
    """CEVAPSIZ sorular ve reddedilisleri. IKI tur, YARI YARIYA.

      VARLIK YOK    "Zeynep Kayabasi'nin danismani kimdir?
                     Zeynep Kayabasi diye bir kisi yok."
                    -> modelin VARLIK LISTESINE hakim olmasi gerek
      ILISKI OLMAZ  "Adana'nin tezi hangisidir? Sehrin tezi olmaz."
                    -> modelin SEMAYA hakim olmasi gerek
                       (tip cumlesi bunun temeli: "Adana bir sehirdir")

    !! ILISKI OLMAZ ciftleri `V.IMKANSIZ`dan gelir, SEMANIN TUMLEYENINDEN
    DEGIL. Ilk surum tumleyeni aliyordu ve YANLIS cumle uretiyordu:
    "Bolumun ogrencisi olmaz" (bolumde ogrenci VAR, grafta yalniz ters
    yon tanimli), "Fakultenin sehri olmaz" (fakulte bir sehirde).
    Ayrinti `veri_11.IMKANSIZ` basindaki notta.

    !! `n` TURETILMEDI, SECILDI -- onkayit §3d. Literatur (R-Tuning,
    2311.09677) reddetme payinin BELIRSIZLIK temelli durumda
    secilmedigini, modelin YANLIS yaptigi orana esit oldugunu soyluyor;
    ama bizimkiler NESNEL OLARAK cevapsiz sorular, farkli bir rejim ve
    orada sayi veren bir calisma bulunamadi.
    Havuz zaten korpustan BUYUK (asagida basiliyor) -- hepsini koymak
    korpusu reddetmeden ibaret birakirdi.
    Kullanici karari, 18 Eylul: %5 ile basla, OLCUMLE duzelt.

    !! OLGU SOYLEMEZ -> `olgu` kumesi BOS, sizinti kapisinda notr.
    !! OLMAYAN AD, GERCEK PARCALARDAN birlestirilir ve GRAFTA OLMADIGI
    dogrulanir -- yoksa gercek bir varliga "yok" ogretirdik."""
    E, tipi = _adlar(G)
    rs = np.random.default_rng(8800 + tohum)
    olan = set(E)
    # tipe gore: OZNE olarak hangi iliskiler VAR
    var = {}
    for (a, r) in G["olgu"]:
        var.setdefault(tipi[a], set()).add(r)
    hedef = {}
    for (a, r), h in G["olgu"].items():
        hedef[r] = tipi[h]
    # --- VARLIK YOK: BES TIPTE uydurma ad (bak: `uydurma_adlar`)
    # !! YALNIZ EGITIM YAKASI. Tutulanlar DURUSTLUK sinavini kuruyor.
    bl = bolme if bolme is not None else reddetme_bolme(G, 0.0, tohum)
    uyd = bl["ad_egitim"]
    havuz = [(t, a) for t in sorted(uyd) if var.get(t) for a in uyd[t]]
    assert not [a for _t, a in havuz if a in olan], "uydurma ad GRAFTA VAR"
    c, n1 = [], n // 2
    # !! HAVUZ TUKENIRSE TEKRAR EDILIR, KIRPILMAZ. Kirpilsaydi kopya
    # buyudukce yari-yari bolunme SESSIZCE kayardi (kopya 5'te havuz
    # 1.460, gereken ~1.815). Ayni ad tekrar ederken ILISKI ve KALIP
    # yeniden cekilir -- birebir ayni cumle degil.
    _ix = []
    while len(_ix) < n1:
        _ix += [int(i) for i in rs.permutation(len(havuz))]
    for i in _ix[:n1]:
        t, a = havuz[int(i)]
        rl = sorted(var[t])
        r = rl[int(rs.integers(len(rl)))]
        c.append(MT.yok_varlik(a, t, r, hedef[r],
                               int(rs.integers(len(MT.YOK_VARLIK)))))
    # ILISKI OLMAZ: ELLE yazilmis KATEGORI HATASI ciftleri, EGITIM yakasi
    cift = [(t, r) for t, r in bl["cift_egitim"] if G["ad"][t]]
    for _ in range(n - len(c)):
        t, r = cift[int(rs.integers(len(cift)))]
        assert r not in var.get(t, ()), f"{t} {r} GRAFTA VAR"
        a = G["ad"][t][int(rs.integers(len(G["ad"][t])))]
        c.append(MT.yok_iliski(a, t, r, hedef[r],
                               int(rs.integers(len(MT.YOK_ILISKI)))))
    c = [c[int(i)] for i in rs.permutation(len(c))]
    _hav = sum(len(G["ad"][t]) for r, ts in V.IMKANSIZ.items() for t in ts)
    _nv = n1
    yaz(f"  reddetme {len(c):,} cumle -> {-(-len(c) // grup):,} belge"
        f"   (varlik yok {_nv:,} + iliski olmaz {len(c) - _nv:,})")
    yaz("     UYDURMA AD " + "  ".join(
        f"{t} {len(uyd[t])}" for t in sorted(uyd)) +
        f"   = {len(havuz):,}")
    yaz(f"     TUTULAN (sinav) ad {sum(len(x) for x in bl['ad_tut'].values()):,}"
        f"  cift {len(bl['cift_tut'])}   -- EGITIMDE HIC GECMEZ")
    yaz(f"     IMKANSIZ cift {V.N_IMKANSIZ}/167 sema disi cift (ELLE) "
        f"-> {_hav:,} olasi soru; ORNEKLENIYOR")
    return [Belge(-1, c[i:i + grup], set()) for i in range(0, len(c), grup)]


def kimlik_belgeleri(v, G, n=8, tohum=0, yaz=print):
    """!! KARISTIRILIR. `E` tipe ve uretim sirasina gore sirali; altisarli
    boluNce belge "Balcova Makine Bolumu / Yenimahalle Makine Bolumu /
    Maltepe Makine Bolumu ..." diye NEREDEYSE AYNI alti cumle oluyordu
    (olculdu, dokumde gorundu)."""
    E, tipi = _adlar(G)
    rs = np.random.default_rng(8500 + tohum)
    c = [MT.kimlik(E[int(i)], tipi[E[int(i)]]) for i in rs.permutation(len(E))]
    b = [Belge(-1, c[i:i + n], set()) for i in range(0, len(c), n)]
    yaz(f"  kimlik {len(c):,} cumle -> {len(b):,} belge")
    return b


def yasak_ciftler(v):
    """(e,r1) -> {(b,r2)} : bu IKI olgu AYNI PENCEREDE bulunamaz.

    Tutulan zincir (e, r1, r2, b, a) pencerede hem (e,r1) hem (b,r2)
    soylenmisse KOPYALANARAK cevaplanir -- bileşim yapmadan.

    !! model_09'da kisit SAHIP duzeyindeydi (`cakisan_ciftler`, iki
    VARLIGIN belgesi yan yana gelmesin). Orada yetiyordu cunku belge
    yalniz kendi sahibinin olgularini soyluyordu. model_11'da sayfa
    ANILAN kenarlari da tasiyor, yani BASKA varliklarin olgularini da
    soyluyor -- sahibe bakmak YETMIYOR.
    OLCULDU (18 Eylul): sahip duzeyinde kisitla 16 zincir pencereden
    kopyalanabilir cikti (belge duzeyinde 0'di). Kapi yakaladi."""
    d = {}
    for lst in (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood, v.ent_kati):
        for z in lst:
            e, r1, r2, b = int(z[0]), int(z[1]), int(z[2]), int(z[3])
            d.setdefault((e, r1), set()).add((b, r2))
            d.setdefault((b, r2), set()).add((e, r1))
    return d


def cakisan_ciftler(v):
    """TUTULAN her zincir (e, r1, r2, b) icin (e, b) CIFTI.

    Bu iki varligin biyografisi ayni dilime duserse, zincir bileşim
    yapilmadan KOPYALANARAK cevaplanabilir hale gelir."""
    c = set()
    for lst in (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood, v.ent_kati):
        for z in lst:
            c.add((int(z[0]), int(z[3])))
            c.add((int(z[3]), int(z[0])))
    return c


def paketle(bel, t_len: int, tohum: int = 0, yasak=None,
            cift=None, yaz=print):
    """BIRLESTIR ve BOL -- Physics 3.1 Ek C'nin duzeni.

        belgeler siralanir -> aralarina <EOS> -> tek akis -> t_len'lik
        dilimler. DOLGU YOK.

    Onceki surum belgeyi BOLMEDEN kutuluyordu: doluluk %62-81'de
    kaliyordu ve TEK bir 514 karakterlik biyografi t_len 512'yi
    imkansiz kiliyordu (olculdu).

    `yasak` verilirse SIRALAMA KISITLI: cakisan iki varligin belgesi
    ayni dilime dusemez. Sansa birakilmaz -- kapi 2/2296 sizinti
    yakalamisti (t_len 256) ve yeniden karistirmak bir COZUM DEGIL,
    baska bir tohumda geri gelir."""
    rs = np.random.default_rng(9000 + tohum)
    sira = [int(i) for i in rs.permutation(len(bel))]
    if yasak and any(b.e >= 0 for b in bel):
        # SINIRLI ILERI BAKIS. Cakisma seyrek (6.034 cift / 1.601 varlik),
        # o yuzden ilk aday genelde uyar. Onceki surum `havuz.pop(j)` ile
        # listeyi her adimda yeniden kuruyordu -- kopya=5'te (147.550
        # belge) O(n^2) olup DONMEDI.
        _kisa = min(len(b.metin) for b in bel) + 1
        geri = max(2, t_len // _kisa + 1)
        ILERI = 64
        son, yer = [], []
        kalan = sira
        while kalan:
            sec = 0
            for j in range(min(ILERI, len(kalan))):
                e = bel[kalan[j]].e
                if e < 0 or not any((e, x) in yasak for x in son):
                    sec = j
                    break
            i = kalan[sec]
            kalan = kalan[:sec] + kalan[sec + 1:] if sec else kalan[1:]
            yer.append(i)
            son = (son + [bel[i].e])[-geri:]
        sira = yer
    # --- CUMLE SINIRINDA BOL ------------------------------------------
    # !! HAM "birlestir ve bol" BIRAKILDI. Olculdu (t_len 512): dilimlerin
    # %88-89'u KELIME ORTASINDA basliyordu ("... Ozlem Kaya. Sin" /
    # "ari Halil Ozturk'tur."). Gercek LM egitiminde bu olagandir, ama
    # BIZDE degil: SINAV cumlesi her zaman konum 0'da basliyor. Egitimde
    # konum 0 cogunlukla kelime ortasiysa modelin o konumdaki davranisi
    # sinavla UYUSMAZ.
    # Bedeli olculdu: cumle sinirinda kesmek %4,5-5,3 dolgu israfi.
    # Karsiliginda her egitim dizisi CUMLE BASINDA basliyor.
    pen, olg, kur, kur_o, n, n_kes = [], [], [], set(), 0, 0
    for k, i in enumerate(sira):
        bel_ = bel[i]
        # OLGU CIFTI KISITI: bu belgenin olgularindan biri, pencerede
        # ZATEN soylenmis bir olguyla tutulan bir zincirin IKI KENARINI
        # tamamliyor mu? Tamamliyorsa pencere SIMDI kapanir.
        # Yeniden siralamak yerine KAPATMAK secildi: siralama sansa
        # birakir ve baska tohumda geri gelir, kapatmak GARANTI eder.
        if cift and kur_o and any(
                p in kur_o for f in bel_.olgu for p in cift.get(f, ())):
            pen.append(" ".join(kur))
            olg.append(kur_o)
            kur, kur_o, n = [], set(), 0
            n_kes += 1
        # BUTUN BELGE: sigmiyorsa pencereyi SIMDI kapat. Yoksa asagidaki
        # dongu onu cumle sinirindan boler ve tetikleyici cevabindan
        # ayrilir.
        if bel_.butun and n + len(bel_.metin) + (1 if kur else 0) > t_len:
            if kur:
                pen.append(" ".join(kur))
                olg.append(kur_o)
            kur, kur_o, n = [], set(), 0
        for j, c in enumerate(bel_.cumle):
            ek = len(c) + (1 if kur else 0)
            if n + ek > t_len:
                pen.append(" ".join(kur)); olg.append(kur_o)
                kur, kur_o, n = [], set(), 0
                ek = len(c)
            kur.append(c); kur_o |= bel_.olgu; n += ek
        # BELGE SINIRI. `" ".join` ayraci da yer kaplar -- ilk surumde
        # yalniz <EOS> karakteri sayilmisti ve dilimler t_len'i 2-5
        # karakter ASIYORDU (516/517 olculdu).
        if k + 1 < len(sira) and kur and n + 2 <= t_len:
            kur.append(EOS_IM); n += 2
        # SIGMAZSA TASINMAZ: dilim siniri zaten ayiriyor. Tasiyinca
        # sonraki dilim <EOS> ile BASLIYORDU (53 dilim, olculdu).
    if kur:
        pen.append(" ".join(kur)); olg.append(kur_o)
    asan = [len(p) for p in pen if len(p) > t_len]
    assert not asan, (
        f"!! {len(asan)} dilim t_len'i ASIYOR (en uzun {max(asan)} > "
        f"{t_len}) -- uzunluk muhasebesi bozuk")
    dol = sum(len(p) for p in pen)
    yaz(f"  paket {len(pen):,} dilim x {t_len}   {dol:,} karakter   "
        f"doluluk %{100 * dol / (len(pen) * t_len):.1f}"
        + ("   (kisitli siralama)" if yasak else "")
        + (f"   CIFT KISITI {n_kes:,} kez pencere kapatti" if n_kes else ""))
    return pen, olg


def sizinti_kapisi(v, olgular, yaz=print):
    """TUTULAN zincirin IKI KENARI ayni PENCEREDE mi -- OLCUM.

    Zincir (e, r1, r2), kopru b. Kopyalayarak cevaplanabilmesi icin
    pencerede hem (e, r1) hem (b, r2) olgusunun SOYLENMIS olmasi gerek.
    Ikisi de varsa model bileşim yapmadan cevabi okuyabilir."""
    toplam = 0
    for ad, lst in (("comp", v.comp), ("ent", v.ent),
                    ("ent_yok", v.ent_yok), ("ood", v.ood)):
        if not lst:
            continue
        n = 0
        for z in lst:
            e, r1, r2, b = (int(z[0]), int(z[1]), int(z[2]), int(z[3]))
            if any((e, r1) in o and (b, r2) in o for o in olgular):
                n += 1
        toplam += n
        yaz(f"  {ad:<8} {n:>4}/{len(lst):<5} tutulan zincir PENCEREDE "
            f"kopyalanabilir" + ("" if n == 0 else "   !! SIZINTI"))
    assert toplam == 0, (
        f"!! {toplam} tutulan zincir pencereden KOPYALANARAK cevaplanabilir")
    return toplam


def havuz(v, G, kopya: int = 5, t_len: int = 512, tohum: int = 0,
          tetik: int = 0, zincir_pay: float = 0.20, n3: int = 0,
          ret_pay: float = 0.05, ret_tut: float = 0.20, yaz=print):
    """KORPUS -> (X, S). Egitim tensoru ve onu ureten sozluk.

    X (N, t_len) int64. Kuyruk PAD ile dolar (~%5): `tam_kayip` PAD
    hedeflerini zaten atliyor, yani dolgu gradyan URETMEZ.

    Sozluk KORPUSTAN cikar -- once metin, sonra jeton. model_08'de ters
    yondeydi (once jeton semasi, metin ancak geri okunurdu)."""
    # !! AYRI ZINCIR AKISI KALDIRILDI (18 Eylul). Zincir cumleleri artik
    # SAYFALARIN ICINE serpiliyor (`zincir_pay`). Ayri akista paketlense
    # model arka arkaya sekiz tane iki-uc katli tamlama okurdu; gercek
    # metinde zincir BASIT cumlelerin ARASINDA gecer.
    bb, bs = sayfalar(v, G, kopya, tohum, tetik, t_len, zincir_pay, n3, yaz)
    kim = kimlik_belgeleri(v, G, tohum=tohum, yaz=yaz)
    # REDDETME: soru cumlelerinin ~ret_pay'i. SECILDI, turetilmedi.
    _ns = sum(len(b.cumle) for b in bs)
    red = reddetme_belgeleri(v, G, int(_ns * ret_pay), tohum,
                             bolme=reddetme_bolme(G, ret_tut, tohum),
                             yaz=yaz) \
        if ret_pay else []
    yasak = cakisan_ciftler(v)
    cift = yasak_ciftler(v)
    dilim, olgu = [], []
    for L in (bb, bs, kim, red):
        d, o = paketle(L, t_len, tohum, yasak, cift, yaz)
        dilim += d
        olgu += o
    sizinti_kapisi(v, olgu, yaz)
    # EOS_IM (chr(10)) SOZLUGE GIRMEZ -- `Sozluk` satir sonunu reddediyor.
    # Belge siniri OZEL jeton, harf degil: metinden dusurulur, kodlamada
    # J.EOS'a cevrilir.
    S = J.Sozluk("".join(dilim).replace(EOS_IM, ""))
    X = np.full((len(dilim), t_len), J.PAD, np.int64)
    for i, p in enumerate(dilim):
        j = [J.EOS if c == EOS_IM else S.ileri[c] for c in p]
        assert len(j) <= t_len, (len(j), t_len)
        X[i, :len(j)] = j
    S.korpus_izi = hashlib.md5(X.tobytes()).hexdigest()[:12]
    _havuz_kapisi(X, dilim, S, t_len, yaz)
    return X, S


def _havuz_kapisi(X, dilim, S, t_len, yaz=print):
    """GIDIS-DONUS: tensordeki her dilim metnine GERI cozulmeli.

    Kodlama sessizce kayarsa (bir karakter atlanir, EOS yanlis yere
    duser) kayip yine duser ve egri normal gorunur -- hata ancak
    sinavda, aylar sonra, 'model ogrenemedi' diye ortaya cikar."""
    geri = dict(S.geri)
    geri[J.EOS] = EOS_IM
    kotu = 0
    for i in range(0, len(X), max(1, len(X) // 2000)):
        s = "".join(geri[int(t)] for t in X[i] if int(t) != J.PAD)
        kotu += (s != dilim[i])
    assert not kotu, f"!! {kotu} dilim gidis-donusu GECMEDI"
    dolu = int((X != J.PAD).sum())
    n_eos = int((X == J.EOS).sum())
    yaz(f"  HAVUZ {X.shape[0]:,} x {t_len}   {dolu:,} jeton   "
        f"doluluk %{100 * dolu / X.size:.1f}   <EOS> {n_eos:,}")
    yaz(f"  sozluk {S.vocab} jeton (iz {S.iz})   gidis-donus GECTI")
    yaz(f"  KORPUS IZI {S.korpus_izi}   -- tensorun kendisi. `metin_11` ya "
        f"da `korpus_11` degisirse bu deger degisir, `ayar` DEGISMEZ.")
