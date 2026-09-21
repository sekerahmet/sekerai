# -*- coding: utf-8 -*-
"""olcme_14 -- DOGRULUK.

Kullanici, 20 Eylul 2026:

  *"olcme ana baslik: Dogruluk. dogruluk 2'ye ayriliyor bicim ve
  bilgi. bicim de ... turkce cumle dogrulugu. bilgi ise cumlenin
  icinde verilen bilginin dogrulugu. bilgi dogrulugu da ogretilen ve
  cikarim diye ikiye bolunebilir."*

  *"bicim ekler uzerinden olmaz ki sadece. bicim dedigim cumlenin
  dogru kurulmasi OGRETILDIGI GIBI."*

  *"bicim kisi soruyorsa kisi olarak cevap veriyor mu mesela. YANLIS
  KISI ayri, kisi olarak cevap vermesi gerektigini BILMESI ayri."*

  *"kalip ve tip degil ayni zamanda EKLER, yani genel olarak
  GRAMERIN TUM PARCALARI."*

    DOGRULUK
    |
    +-- BICIM   gramerin tamami: cumle OGRETILDIGI GIBI kuruldu mu
    |   +-- kalip   iskelet korpusun ogrettiklerinden biri mi
    |   +-- ek      ekler dogru birime, dogru SIRADA takilmis mi
    |   +-- tip     kisi soruluyorsa KISI cevabi verdi mi
    |   +-- kapanmadi / yozlasma      tani
    |
    +-- BILGI   cumledeki bilginin dogrulugu -- HANGI kisi
        +-- OGRETILEN   cevap korpusta SOYLENIYOR
        +-- CIKARIM     cevap korpusta SOYLENMIYOR

!! `one` / `seen` / `comp` / `ent` / `ent_yok` YOK.
Kullanici, 20 Eylul: *"artik seen comp vs yok, yeni olcu kriterlerini
soylemistim; eski kavramlar alakasiz."* Onlar model_09'un GRAF
bolmeleriydi. Bir soru OGRETILEN mi CIKARIM mi, ETIKETINDEN degil
KORPUSTAN belirlenir: cevabi soyleyen cumle metinde geciyor mu.
Olcut dogrudan, denetlenebilir, ve grafin etiketlemesinden bagimsiz.

ISKELET KUMESI ve EK KURALI da KORPUSTAN turiyor -- elle dilbilgisi
yazilmaz. Kalip listesi yazsak modelin ogrenmesi gerekeni BIZ tarif
etmis oluruz; oysa olcu "ogretildigi gibi mi".

SINAV GRAFI KULLANABILIR, MODEL KULLANAMAZ.
"""
from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F

import ek_14 as EK
import metin_14 as MT

YUVA = -1          # iskelette VARLIK yuvasi


# =====================================================================
# VARLIK -> BIRIM   (graftan; SINAV tarafi)
# =====================================================================
def _yuzey(ad: str, TR: dict) -> list[str]:
    """Graf adi ASCII, korpus Turkce; ceviri tablodan.

    !! model_13'te bu atlanmisti: `Yilmaz` korpustaki Turkce yazimla
    eslesmiyordu ve 3.000 sorunun 2.881'i dusmustu.
    !! TAM ANAHTAR ONCE (`yasadigi_yer`), sonra parca parca; ve ceviri
    sonrasi BOSLUKTAN da bolunur (`Mimarsinan` -> "Mimar Sinan")."""
    if ad in TR:
        return TR[ad].split()
    return [x for w in ad.split("_") for x in TR.get(w, w).split()]


def birimle(ad, TR, kokler, bx, korunan) -> tuple:
    """Ad -> birim indeksleri. Bir parca sozlukte yoksa BOS."""
    ix = []
    for w in _yuzey(ad, TR):
        for x in EK.bol(w, kokler, korunan=korunan):
            if x not in bx:
                return ()
            ix.append(bx[x])
    return tuple(ix)


def _gecer(ham: bytes, dizi) -> bool:
    """`dizi` korpus akisinda geciyor mu.  YAVAS AMA BASIT -- REFERANS.

    Bayt aramasi: sayilari metne cevirip `in` demek 15,4 M birimde
    ~100 MB'lik bir metin kuruyor ve dakikalarca asiliyordu. uint16
    baytlarinda `bytes.find` C hizinda; hizalama (cift ofset) elle
    denetleniyor.

    Uretimde `Arama` kullanilir; bu, onun kapisidir (test 30)."""
    des = np.asarray(dizi, np.uint16).tobytes()
    i = ham.find(des)
    while i >= 0 and i % 2:
        i = ham.find(des, i + 1)
    return i >= 0


class Arama:
    """Birim akisinda dizi arama -- IKILI indeksle.

    *Gerekce OLCULDU (20 Eylul):* `_gecer` dogru ama BULUNMAYAN bir
    dizide 30,7 MB'lik akisin TAMAMINI tariyor. Sinavin 14.123
    sorusunun 11.123'unde cevap korpusta YOK (zaten CIKARIM olmalari
    bu demek), yani neredeyse hepsi tam tarama: olcum 5 dakikayi
    gecti ve HER KOSUDA odenecekti.

    Indeks: her (w_j, w_{j+1}) ciftinin gectigi konumlar, sirali.
    Sorgu ilk ciftin kovasina bakar, kalani vektorel dogrular.
    Kurulum ~2 sn ve BIR KEZ; sorgu kovanin boyu kadar.
    """

    def __init__(self, dizi):
        self.d = d = np.asarray(dizi, np.int64)
        self.n = n = int(d.max()) + 1
        anahtar = d[:-1] * n + d[1:]
        self.sira = np.argsort(anahtar, kind="stable")
        self.anahtar = anahtar[self.sira]

    def __call__(self, q) -> bool:
        q = [int(x) for x in q]
        if not q:
            return True
        # !! ARALIK DISI = YOK. Anahtar q0*n + q1 ile kodlaniyor ve n
        # AKISIN en buyugunden geliyor; sozlukte olup akista hic
        # gecmeyen bir birim sorulursa kodlama CAKISIR ve baska bir
        # cift bulunmus gibi olur. (Kapi 30 bunu yakaladi.)
        if any(x < 0 or x >= self.n for x in q):
            return False
        if len(q) == 1:
            return bool((self.d == q[0]).any())
        a = q[0] * self.n + q[1]
        i, j = np.searchsorted(self.anahtar, [a, a + 1])
        if i == j:
            return False
        poz = self.sira[i:j]
        poz = poz[poz + len(q) <= len(self.d)]
        for k in range(2, len(q)):
            if not len(poz):
                return False
            poz = poz[self.d[poz + k] == q[k]]
        return bool(len(poz))


# =====================================================================
# BICIM -- iskelet kumesi korpustan
# =====================================================================
class Bicim:
    """Korpusun OGRETTIGI cumle iskeletleri.

    Iskelet: cumle, varlik birimlerinden olusan her kesintisiz kosu
    tek bir YUVA ile degistirilmis hali.

        Hasan Yilmaz -TAMLAYAN tezi Gorgul Elestiri Tezi -BILDIRME .
        ->  <V> -TAMLAYAN tezi <V> -BILDIRME .
    """

    def __init__(self, dizi, varlik_ix, bitis_ix, ek_ix=frozenset(), n=None):
        # Maske SOZLUK boyunda: model sozlukteki her birimi uretebilir,
        # akista hic gecmeyeni de.
        self.n = n = int(n or max(int(dizi.max()),
                                  max(varlik_ix, default=0)) + 1)
        self.varlik = np.zeros(n, bool)
        self.varlik[list(varlik_ix)] = True
        self.bitis, self.ek = set(bitis_ix), set(ek_ix)

        self.iskelet = collections.Counter()
        for c in self._cumleler(dizi):
            self.iskelet[self._iskelet(c)] += 1

        # EK KURALI korpustan: her ekin ONUNDE hangi birimler gorulmus.
        # Hem "koke mi takildi" hem "ek SIRASI dogru mu" tek tabloda.
        self.ek_oncesi = collections.defaultdict(set)
        yer = np.isin(dizi, list(self.ek)).nonzero()[0]
        for i in yer[yer > 0]:
            self.ek_oncesi[int(dizi[i])].add(int(dizi[i - 1]))

    def _cumleler(self, dizi):
        bas = 0
        for i in np.isin(dizi, list(self.bitis)).nonzero()[0]:
            if 1 < i + 1 - bas <= 40:
                yield dizi[bas:i + 1]
            bas = i + 1

    def _iskelet(self, c) -> tuple:
        cik, yuva = [], False
        for w in c:
            w = int(w)
            if 0 <= w < self.n and self.varlik[w]:
                if not yuva:
                    cik.append(YUVA)
                yuva = True
            else:
                cik.append(w)
                yuva = False
        return tuple(cik)

    def _ek_say(self, c):
        """(dogru yerlesmis ek, toplam ek).

        !! CUMLE BASINA DEGIL EK BASINA. Cumle basina 0/1 verilince
        hic ek icermeyen cikti VACUOUS geciyordu; egitimsiz model
        0,94 aliyordu."""
        iyi = tot = 0
        for i, w in enumerate(c):
            if w in self.ek:
                tot += 1
                iyi += int(i > 0 and c[i - 1] in self.ek_oncesi[w])
        return iyi, tot

    def puanla(self, dizi) -> dict:
        d = [int(x) for x in dizi]
        kap = next((i for i, w in enumerate(d) if w in self.bitis), None)
        if kap is None:
            iy, n_ = self._ek_say(d)     # kapanmasa da EKLER okunabilir
            return dict(kalip=0, ek_iyi=iy, ek_n=n_, kapanmadi=1, yozlasma=0)
        c = d[:kap + 1]
        iy, n_ = self._ek_say(c)
        return dict(kalip=int(self._iskelet(np.array(c)) in self.iskelet),
                    ek_iyi=iy, ek_n=n_, kapanmadi=0,
                    yozlasma=int(any(c[i] == c[i + 1] == c[i + 2]
                                     for i in range(len(c) - 2))))


# =====================================================================
# SORULAR -- OGRETILEN / CIKARIM ayrimi KORPUSTAN
# =====================================================================
class Soru:
    """onek modele verilir, cevap uretilir.

    kisayol    r2 kopruye degil OZNEYE uygulanmis hali
    soylenmis  cevabi soyleyen cumle KORPUSTA geciyor mu
    """
    __slots__ = ("onek", "cevap", "kisayol", "soylenmis")

    def __init__(self, onek, cevap, kisayol=(), soylenmis=False):
        self.onek, self.cevap = onek, cevap
        self.kisayol, self.soylenmis = kisayol, soylenmis


class Sorular:
    """Graftan soru kurar, sonra KORPUSA BAKARAK siniflar."""

    def __init__(self, v, E_ad, ILISKI, TIPLER, TR, TR_ILISKI, kokler, bx,
                 korunan):
        self.v, self.E_ad, self.ILISKI = v, E_ad, ILISKI
        self.TIPLER, self.TR, self.TR_ILISKI = TIPLER, TR, TR_ILISKI
        self.bx, self.kokler, self.korunan = bx, kokler, korunan
        self._nb = {}

    def birim(self, ad) -> tuple:
        if ad not in self._nb:
            self._nb[ad] = birimle(ad, self.TR, self.kokler, self.bx,
                                   self.korunan)
        return self._nb[ad]

    def jetonla(self, metin: str) -> tuple:
        """METIN -> birim indeksleri. Bir parca sozlukte yoksa BOS.

        Korpus akisi da boyle kuruldu (`birim_14.kur`): bosluktan bol,
        her kelimeyi `EK.bol` ile ayir. Ayni yol, ayni sonuc."""
        ix = []
        for w in metin.split():
            for x in EK.bol(w, self.kokler, korunan=self.korunan):
                if x not in self.bx:
                    return ()
                ix.append(self.bx[x])
        return tuple(ix)

    def kur(self, zincir, adim):
        """-> (Soru, KANIT dizisi) ya da None.

        Yuzey `metin_14.sinav_yuzeyi`den gelir, JETON ELLE DIZILMEZ --
        korpusu yazan modulun ta kendisi. Gerekce orada."""
        z = [int(x) for x in zincir]
        e, rs, ans = z[0], z[1:1 + adim], z[-1]
        onek_m, cev_m, kanit_m = MT.sinav_yuzeyi(
            self.E_ad[e], [self.ILISKI[r] for r in rs], self.E_ad[ans],
            self.TIPLER[int(self.v.tip[ans])])
        onek, cev, kanit = (self.jetonla(onek_m), self.jetonla(cev_m),
                            self.jetonla(kanit_m))
        if not onek or not cev or not kanit:
            return None
        ksy = ()
        if adim == 2:
            h = int(self.v.facts[e, z[2]])
            if h >= 0 and h != ans:
                ksy = self.birim(self.E_ad[h])
        return Soru(onek, cev, ksy), kanit

    def tum(self, listeler: dict, dizi, yaz=print) -> dict:
        """Butun zincirleri TEK HAVUZDA toplar ve KORPUSA gore siniflar.

        !! BOLME ADI HICBIR YERDE KULLANILMAZ -- ne secmek ne elemek
        icin. `listeler` sozlugunun anahtarlari (one/seen/comp/ent/
        ent_yok/ood) model_09'un graf bolmeleriydi; bu olcunun onlarla
        isi yok. Adim ZINCIRIN UZUNLUGUNDAN, sinif KORPUSTAN.

        *`ood` de eleniyordu, kaldirildi (20 Eylul):* gerekcesi "80
        ornek, cozunurluksuz -- tek ornek ~0,014 oynatir" idi ve bu
        `ood` KENDI SUTUNUYKEN dogruydu. 14.123'luk tek havuzda artik
        sutun degil; yapisi comp/ent ile ayni (iki adimli zincir), tek
        farki hangi kenarlarin tutuldugu -- onu da bayt aramasi zaten
        soyluyor.

        ZINCIR DUZENI -- OLCULDU, varsayilmadi:
            1 adim   (e, r, cevap)                  uzunluk 3
            2 adim   (e, r1, r2, KOPRU, cevap)      uzunluk 5
        Kopru z[3]'te ve soruya GIRMEZ; adim = (len-1)//2.

        *Gerekce OLCULDU (20 Eylul):* burasi `3 <= len <= 4` suzuyor
        ve `adim = len-2` diyordu. Uzunluk 5 hicbirine uymuyor, yani
        BUTUN 2 adimli zincirler SESSIZCE dusuyordu: 14.043 zincirin
        11.043'u. Olcum yalniz 1 adimi siniyordu ve `CIKARIM` 8
        ornege dusuyordu (comp/ent'in tamami elenmisti)."""
        gecer = Arama(dizi)
        cik = {"OGRETILEN": [], "CIKARIM": []}
        atilan = 0
        havuz, gorulen = [], set()
        for ad, zs in listeler.items():
            if not hasattr(zs, "__len__"):
                continue
            for z in zs:
                t = tuple(int(x) for x in z)
                if len(t) in (3, 5) and t not in gorulen:
                    gorulen.add(t)
                    havuz.append(t)
        for z in havuz:
            adim = (len(z) - 1) // 2
            s = self.kur(z, adim)
            if s is None:
                atilan += 1
                continue
            soru, bildirim = s
            soru.soylenmis = gecer(bildirim)
            cik["OGRETILEN" if soru.soylenmis else "CIKARIM"].append(soru)
        bir = sum(1 for z in havuz if len(z) == 3)
        yaz("zincir %d (tekil: %d bir adim + %d iki adim)   "
            "OGRETILEN %d   CIKARIM %d   kurulamayan %d"
            % (len(havuz), bir, len(havuz) - bir, len(cik["OGRETILEN"]),
               len(cik["CIKARIM"]), atilan))
        return cik


def kapi(sorular: dict, dizi, n=200, en_az=0.5) -> str:
    """Sorunun ONEGI korpusta gercekten geciyor mu.

    *Gerekce OLCULDU (model_13):* bolucu ozel adlarin son unlusunu
    yiyordu (Kaya->Kay) ve 3.000 sorunun yalniz 119'u korpusta
    geciyordu. Hicbir SAYISAL kapi bunu gostermemisti."""
    hep = sorular["OGRETILEN"] + sorular["CIKARIM"]
    if not hep:
        return "SORU YOK"
    gecer = Arama(dizi)
    ornek = hep[:n]
    tut = sum(gecer(s.onek[:-3]) for s in ornek)   # soru kuyrugu haric
    oran = tut / len(ornek)
    assert oran >= en_az, (
        "SORU KAPISI: %d ornegin yalniz %d'u korpusta geciyor (%.1f%%) -- "
        "bolme ya da ad esleme bozuk" % (len(ornek), tut, 100 * oran))
    return "soru kapisi GECTI  %d/%d (%.1f%%)" % (tut, len(ornek), 100 * oran)


def tip_haritasi(v, E_ad, s: Sorular):
    """birim -> o birimle BASLAYAN varliklarin TIP kumesi."""
    h = collections.defaultdict(set)
    for i, ad in enumerate(E_ad):
        b = s.birim(ad)
        if b:
            h[b[0]].add(int(v.tip[i]))
    return lambda w: h.get(int(w), set())


# =====================================================================
# PUANLAMA
# =====================================================================
def _span(u, bitis) -> tuple:
    cik = []
    for w in u:
        if int(w) in bitis:
            break
        cik.append(int(w))
    return tuple(cik)


def bicim_puanla(sorular, uretilen, bicim: Bicim, bitis, tip_of=None) -> dict:
    """Cumle OGRETILDIGI GIBI kuruldu mu.

    !! YALNIZ URETILEN puanlanir. Onek eklenirse `puanla` ilk bitis
    biriminde keser ve SORUYU puanlar; egitimsiz modelde bile
    kalip=1,0000 cikiyordu. Korpusta zaten iki ayri cumle var:
    "... hangisidir?" ve "... -dir."."""
    kal = [bicim.puanla(u) for u in uretilen]
    n = max(1, len(sorular))
    d = {a: sum(x[a] for x in kal) / n
         for a in ("kalip", "kapanmadi", "yozlasma")}
    ek_n = sum(x["ek_n"] for x in kal)
    d["ek"] = sum(x["ek_iyi"] for x in kal) / ek_n if ek_n else float("nan")
    d["ek_n"] = ek_n
    if tip_of is not None:
        d["tip"] = sum(
            bool(sp and s.cevap and tip_of(sp[0]) & tip_of(s.cevap[0]))
            for s, sp in ((s, _span(u, bitis))
                          for s, u in zip(sorular, uretilen))) / n
    return d


def bilgi_puanla(sorular, uretilen, bitis, ek_ix=frozenset()) -> dict:
    """Cumledeki bilgi dogru mu. KIMLIK uzerinden.

        tam      cevap araligi dogru                <- HUKUM
        aile     SON parca dogru, varlik yanlis     -- tani
                 480 kisi / 10 soyad: aileyi bulup icinden secmek
                 1/48 = %2,08 verir; bu sutun onu DOGRUDAN sayar
        kisayol  r2 kopruye degil OZNEYE uygulanmis -- tani
                 DENKLEM §6.1 bunun YAPISAL OLARAK imkansiz oldugunu
                 soyluyor; sutun o iddiayi SINAR
        bos      hic birim uretmemis                -- bosluga karsi

    !! KUYRUKTAKI EK ATILIR, iki tarafta da.  Onceki hal BIREBIR
    esitlik ariyordu: beklenen cevap CIPLAK ad (`sinav_yuzeyi` `Y`yi
    donduruyor) ama modelin urettigi dilbilgisel Turkce, yani
    bildirme ekini tasiyor -- `[Agri]` vs `[Agri -dir]`.  OLCULDU
    (21 Eylul): `tam` DORT kosuda da BIREBIR 0,0000 verdi; ek
    toleransiyla 0,0003 / 0,0004 / 0,0008 / 0,0138.  Ilk uc kosuda
    fark yoktu (zaten sansin altinda) ama dorduncude GERCEK bir
    sinyal sifir diye raporlandi.  Kapi 40.
    Morfoloji ayri bir sutunda zaten olculuyor (`ek`); burasi
    BILGIYI olcer.

    SIRA ONEMLI: kisayol dogru cevabin oneki olabilir, once `tam`."""
    def kirp(p):
        p = list(p)
        while p and int(p[-1]) in ek_ix:
            p.pop()
        return tuple(p)

    t = a = k = b = 0
    for s, u in zip(sorular, uretilen):
        sp = _span(u, bitis)
        if not sp:
            b += 1
            continue
        sk, ck = kirp(sp), kirp(tuple(s.cevap))
        if sk and sk == ck:
            t += 1
        elif s.kisayol and sk == kirp(tuple(s.kisayol)):
            k += 1
        elif ck and sk and sk[-1] == ck[-1]:
            a += 1
    n = max(1, len(sorular))
    return dict(tam=t / n, aile=a / n, kisayol=k / n, bos=b / n, n=n)


# =====================================================================
# URETIM -- toplu, acgozlu
# =====================================================================
@torch.no_grad()
def uret_toplu(mdl, onekler, n_yeni=10, r=0.25, bs=2048, dev="cuda"):
    """Onekler farkli uzunlukta olabilir; her biri KENDI uzunlugunca
    zorlanir, sonra model serbest devam eder.

    Acgozlu. Isin aramasi `mdl.uret` ile soru basina yapilir ve
    pahalidir; toplu olcumde acgozlu, isin ORNEKLEMDE.

    !! ADIM `mdl.adim`DAN gelir, burada YENIDEN YAZILMAZ.  Once
    yazilmisti ve hafizayi okumuyordu: §12b kosusu hafizayla egitilip
    HAFIZASIZ olculdu, BICIM sayilari gecersiz cikti (21 Eylul).
    Kapi 38 bu iki yolu birbirine bagliyor."""
    assert not mdl.saat, "saat acikken toplu uretim saat izini tutmali"
    R, C, P = mdl.donme(), mdl.kod(), mdl.p
    Rd, Ct, esik = (R.reshape(mdl.n, mdl.D * mdl.D), C.t().contiguous(),
                    mdl.esik(r))
    D, d = mdl.D, mdl.d
    cik = []
    for i in range(0, len(onekler), bs):
        gr = onekler[i:i + bs]
        L, B = max(len(o) for o in gr), len(gr)
        pad = torch.zeros(B, L, dtype=torch.long, device=dev)
        boy = torch.tensor([len(o) for o in gr], device=dev)
        for j, o in enumerate(gr):
            pad[j, :len(o)] = torch.as_tensor(o, device=dev)
        z = P.new_zeros(B, D)
        z[:, :d] = P[pad[:, 0]]
        yol = pad[:, :1]
        for adim in range(1, L + n_yeni):
            z = mdl.adim(z, yol[:, -1], Rd, C, Ct, esik)[0]
            q = F.normalize(z[:, :d], dim=-1)
            sec = (2 - 2 * (q @ P.T)).argmin(1)
            if adim < L:
                sec = torch.where(adim < boy, pad[:, adim], sec)
            yol = torch.cat([yol, sec[:, None]], 1)
        for j, o in enumerate(gr):
            cik.append(yol[j, len(o):].tolist())
    return cik


# =====================================================================
# DOGRULUK
# =====================================================================
def dogruluk(mdl, bicim: Bicim, sorular: dict, bitis, tip_of=None,
             n_yeni=10, r=0.25, dev="cuda") -> dict:
    """Tek cagri, tam agac. BICIM ayni uretimden okunur.

    `bicim.ek` BILGIYE de gidiyor: kuyruktaki bildirme eki iki
    tarafta da atilsin diye (bkz. `bilgi_puanla`)."""
    s = {"BICIM": {}, "BILGI": {}}
    for grup in ("OGRETILEN", "CIKARIM"):
        q = sorular.get(grup) or []
        if not q:
            continue
        u = uret_toplu(mdl, [x.onek for x in q], n_yeni, r, dev=dev)
        s["BICIM"][grup] = bicim_puanla(q, u, bicim, bitis, tip_of)
        s["BILGI"][grup] = bilgi_puanla(q, u, bitis, bicim.ek)
    return s


def yaz(s: dict, yaz=print):
    """Iki blok AYRI okunur: BICIM tutup BILGI dusuyorsa model dili
    ogrenmis, olgulari ogrenmemis demektir."""
    yaz("DOGRULUK")
    yaz("  BICIM       kalip    ek       tip      kapanmadi yozlasma  (ek n)")
    for ad, v in s["BICIM"].items():
        yaz("    %-9s %.4f   %.4f   %.4f   %.4f    %.4f    %d"
            % (ad, v["kalip"], v["ek"], v.get("tip", float("nan")),
               v["kapanmadi"], v["yozlasma"], v["ek_n"]))
    yaz("  BILGI       tam      aile     kisayol  bos       (n)")
    for ad, v in s["BILGI"].items():
        yaz("    %-9s %.4f   %.4f   %.4f   %.4f    %d"
            % (ad, v["tam"], v["aile"], v["kisayol"], v["bos"], v["n"]))
