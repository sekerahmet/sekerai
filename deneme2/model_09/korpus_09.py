# -*- coding: utf-8 -*-
"""korpus_09 — CUMLELERDEN BELGE, BELGELERDEN PAKET. model_09'un havuzu.

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

import jeton_09 as J
import metin_09 as MT
import veri_09 as V

DUZ = tuple(range(MT.N_BICIM - 1))       # bildirim bicimleri (soru HARIC)
SORU = MT.N_BICIM - 1
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


def biyografiler(v, G, kopya: int = 1, tohum: int = 0, tetik: int = 0,
                 t_len: int = 512, yaz=print):
    """BIYOGRAFI: bir varligin ATOMIK OLGULARI. bioS'in karsiligi.

    !! 2-HOP ZINCIRLER BURAYA GIRMEZ -- ayri akis (`zincirler`). Ikisi
    birlestirilince belge 4.243 karaktere cikiyordu (olculdu) ve hicbir
    makul pencereye sigmiyordu.

    `kopya` = Physics 3.1'in `multiM`si: her varlik icin M belge, her
    birinde SIRA ve YUZEY bagimsiz secilir. `multi5` bioS'in kendi
    sayisi -- buyutmek AYRI bir dugme olur.

    `tetik` = bu M belgenin kaci "X hakkinda ne biliyoruz?" onegiyle
    gelsin. Kalani onekSIZ kalir, yani veride HER IKI davranis da var
    ve korpus BUYUMEZ (kullanici karari, 18 Eylul).

    !! PERMUTASYON TEKRARI REDDEDILIR. Olculdu: varlik basina ortalama
    4,73 FARKLI belge cikiyordu, 386 varlikta (%24) 5'ten az -- yani
    ayni belge iki kez uretilmis. Az olgulu varliklarda (3 olgu = 6
    siralama) carpisma kacinilmaz gibi gorunuyor ama yuzey secimi de
    caristigi icin asil sebep ornekleme. Kullanicinin sarti "bilgiler
    dogru SIRASI FARKLI"ydi; tekrar o sarti zayiflatiyor."""
    E, tipi = _adlar(G)
    rs = np.random.default_rng(7000 + tohum)
    ait = {}
    for e, r, a in v.one:
        ait.setdefault(int(e), []).append((int(e), int(r), int(a)))
    bil, sor = [], []
    n_tet = n_parca = n_carp = 0
    for e in sorted(ait):
        gorulen = set()
        for k in range(kopya):
            for _ in range(32):          # TEKRARI reddet, yeniden cek
                sira = tuple(int(i) for i in rs.permutation(len(ait[e])))
                yuz = tuple(int(rs.choice(DUZ)) for _ in ait[e])
                if (sira, yuz) not in gorulen:
                    break
                n_carp += 1
            gorulen.add((sira, yuz))
            d, s_, olgu = [], [], set()
            for j, i in enumerate(sira):
                o, r, a = ait[e][i]
                d.append(MT.cumle(E[o], V.ILISKI[r], E[a], yuz[j], tipi[E[a]]))
                s_.append(MT.cumle(E[o], V.ILISKI[r], E[a], SORU, tipi[E[a]]))
                olgu.add((o, r))
            sor.append(Belge(e, s_, olgu))
            if k >= tetik:
                bil.append(Belge(e, d, olgu))
                continue
            # --- TETIKLENMIS: onek + olgular, t_len'e SIGACAK parcalar --
            # Sigmayan biyografi ATILMAZ, IKI CEVABA bolunur. Gercek bir
            # biyografi cevabi da her zaman eksiksiz degildir; bolmek
            # modele cevap UZUNLUGUNUN degistigini de ogretir.
            onek = MT.biyografi_sorusu(E[e])
            assert len(onek) + 1 + max(len(c) for c in d) <= t_len, (
                f"tek olgu bile sigmiyor: {E[e]}")
            kur, n, olg = [onek], len(onek), set()
            for j, i in enumerate(sira):
                c = d[j]
                if n + 1 + len(c) > t_len:
                    bil.append(Belge(e, kur, olg, butun=True))
                    n_parca += 1
                    kur, n, olg = [onek], len(onek), set()
                kur.append(c)
                n += 1 + len(c)
                olg.add((ait[e][i][0], ait[e][i][1]))
            bil.append(Belge(e, kur, olg, butun=True))
            n_tet += 1
            n_parca += 1
    _tekil = len({(b.e, tuple(b.cumle)) for b in bil})
    yaz(f"  biyografi {len(bil):,} bildirim + {len(sor):,} soru"
        f"   ({len(ait):,} varlik x {kopya} kopya)")
    if tetik:
        yaz(f"     TETIKLENMIS {n_tet:,} biyografi -> {n_parca:,} BUTUN belge"
            f"   ({n_parca - n_tet:,} tanesi t_len'e sigmadigi icin bolundu)")
        yaz(f'     onek: "{MT.biyografi_sorusu(E[min(ait)])}"')
    yaz(f"     tekil belge {_tekil:,}/{len(bil):,}"
        f"   (permutasyon carpismasi {n_carp:,} kez REDDEDILDI)")
    return bil, sor


def zincirler(v, G, kopya: int = 1, tohum: int = 0, n=6, yaz=print):
    """2-HOP zincir cumleleri. Bagimsiz ifadeler, `n`erli gruplanir.

    !! Bu cumleler KOPRUYU YAZMAZ, yani hicbir ATOMIK olguyu ele
    vermezler -- `olgu` kumesi BOS ve sizinti kapisinda notrdurlar."""
    E, tipi = _adlar(G)
    rs = np.random.default_rng(7500 + tohum)
    d, s = [], []
    for _k in range(kopya):
        for i in rs.permutation(len(v.tr2)):
            e, r1, r2, _b, a = (int(x) for x in v.tr2[int(i)])
            bb = int(rs.choice(DUZ))
            d.append(MT.zincir(E[e], V.ILISKI[r1], V.ILISKI[r2], E[a], bb,
                               tipi[E[a]]))
            s.append(MT.zincir(E[e], V.ILISKI[r1], V.ILISKI[r2], E[a], SORU,
                               tipi[E[a]]))
    B = lambda c: [Belge(-1, c[i:i + n], set()) for i in range(0, len(c), n)]
    yaz(f"  zincir {len(d):,} bildirim + {len(s):,} soru cumlesi")
    return B(d), B(s)


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


def paketle(bel, t_len: int, tohum: int = 0, yasak=None, yaz=print):
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
    pen, olg, kur, kur_o, n = [], [], [], set(), 0
    for k, i in enumerate(sira):
        bel_ = bel[i]
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
        + ("   (kisitli siralama)" if yasak else ""))
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
          tetik: int = 0, yaz=print):
    """KORPUS -> (X, S). Egitim tensoru ve onu ureten sozluk.

    X (N, t_len) int64. Kuyruk PAD ile dolar (~%5): `tam_kayip` PAD
    hedeflerini zaten atliyor, yani dolgu gradyan URETMEZ.

    Sozluk KORPUSTAN cikar -- once metin, sonra jeton. model_08'de ters
    yondeydi (once jeton semasi, metin ancak geri okunurdu)."""
    bb, bs = biyografiler(v, G, kopya, tohum, tetik, t_len, yaz)
    zb, zs = zincirler(v, G, kopya, tohum, yaz=yaz)
    kim = kimlik_belgeleri(v, G, tohum=tohum, yaz=yaz)
    yasak = cakisan_ciftler(v)
    dilim, olgu = [], []
    for L in (bb, bs, zb, zs, kim):
        d, o = paketle(L, t_len, tohum, yasak, yaz)
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
    yaz(f"  KORPUS IZI {S.korpus_izi}   -- tensorun kendisi. `metin_09` ya "
        f"da `korpus_09` degisirse bu deger degisir, `ayar` DEGISMEZ.")
