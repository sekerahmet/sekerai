# -*- coding: utf-8 -*-
"""olcme_14 -- DOGRULUK.  Bastan yazildi.

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
    |   +-- kalip      iskelet korpusun ogrettiklerinden biri mi
    |   |              (kelime yerlesimi, cumle kurulusu)
    |   +-- ek         ekler dogru birime, dogru SIRADA takilmis mi
    |   +-- tip        kisi soruluyorsa KISI cevabi verdi mi
    |   +-- kapanmadi / yozlasma        tani sutunlari
    |
    +-- BILGI   cumledeki bilginin dogrulugu -- HANGI kisi
        +-- OGRETILEN   cevap korpusta VAR    one, seen
        +-- CIKARIM     cevap korpusta YOK    comp, ent
            sutunlar: tam / aile / kisayol / bos

ISKELET KUMESI KORPUSTAN TURETILIR -- elle dilbilgisi yazilmaz. Bir
kalip listesi yazsak, modelin ogrenmesi gerekeni BIZ tarif etmis
oluruz; oysa olcu "ogretildigi gibi mi" sorusu.

SINAV GRAFI KULLANABILIR, MODEL KULLANAMAZ. Varlik listesi burada
serbest; `model_14` onu hic gormuyor.
"""
from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F

import ek_14 as EK

YUVA = -1          # iskelette VARLIK yuvasi
BITIS = ".?!"      # cumle kapatan birimler


# =====================================================================
# VARLIK -> BIRIM  (graftan; SINAV tarafi, model gormuyor)
# =====================================================================
def _tr_ad(ad: str, TR: dict) -> list[str]:
    """Graf adi ASCII, korpus Turkce. TR tablosundan gecirilir.

    !! model_13'te bu atlanmisti: `Yilmaz` ile korpustaki `Yilmaz`
    eslesmiyordu ve 3.000 sorunun 2.881'i dusmustu.

    !! CEVIRI SONRASI BOSLUKTAN DA BOLUNUR: tablo bazi adlari cok
    kelimeye ceviriyor (`Mimarsinan` -> "Mimar Sinan",
    `yasadigi_yer` -> "yasadigi yer"). Bolunmezse tek parca olarak
    aranir ve sozlukte bulunamaz."""
    if ad in TR:                       # once TAM anahtar (`yasadigi_yer`)
        return TR[ad].split()
    return [x for w in ad.split("_") for x in TR.get(w, w).split()]


def varlik_birim(ad, TR, kokler, bx, korunan) -> tuple[int, ...]:
    """Varlik adi -> birim indeksleri. Bilinmeyen parca varsa BOS."""
    ix = []
    for w in _tr_ad(ad, TR):
        for x in EK.bol(w, kokler, korunan=korunan):
            if x not in bx:
                return ()
            ix.append(bx[x])
    return tuple(ix)


# =====================================================================
# BICIM -- iskelet kumesi korpustan
# =====================================================================
class Bicim:
    """Korpusun OGRETTIGI cumle iskeletleri.

    Iskelet: cumle, varlik birimlerinden olusan her kesintisiz kosu
    tek bir YUVA ile degistirilmis hali.

        Hasan Yilmaz -TAMLAYAN tezi Gorgul Elestiri Tezi -BILDIRME .
        ->  <V> -TAMLAYAN tezi <V> -BILDIRME .

    !! YAKLASIM: kosu birlestirme, varlik SINIRLARINI bilmiyor; yan
    yana iki varlik tek yuva olur. Korpusta bu yalniz ad+soyad icinde
    oluyor, yani istenen davranis. Iliski sozcukleri (`tezi`) ile
    varlik adlari (`Tezi`) BUYUK HARFLE ayriliyor, carpismiyor."""

    def __init__(self, dizi, varlik_ix: set[int], bitis_ix: set[int],
                 ek_ix: set[int] = frozenset(), n: int | None = None):
        # !! Maske SOZLUK boyunda olmali, akisin maksimumunda DEGIL:
        # model sozlukteki her birimi uretebilir, akista gecmeyeni de.
        self.n = n = int(n or max(int(dizi.max()), max(varlik_ix,
                                                       default=0)) + 1)
        self.varlik = np.zeros(n, bool)
        self.varlik[list(varlik_ix)] = True
        self.bitis = set(bitis_ix)
        self.ek = set(ek_ix)
        self.iskelet = collections.Counter()
        for c in self._cumleler(dizi):
            self.iskelet[self._iskelet(c)] += 1
        # EK KURALI korpustan: her ekin ONUNDE hangi birimler gorulmus.
        # Boylece hem "ek koke mi takildi", hem "ek sirasi dogru mu"
        # (-IYELIK -TAMLAYAN gecerli, tersi degil) tek tabloyla olculur.
        self.ek_oncesi = collections.defaultdict(set)
        d = dizi
        yer = np.isin(d, list(self.ek)).nonzero()[0]
        for i in yer[yer > 0]:
            self.ek_oncesi[int(d[i])].add(int(d[i - 1]))

    def _cumleler(self, dizi):
        """Akisi bitis birimlerinde cumlelere boler."""
        bas = 0
        son = np.isin(dizi, list(self.bitis)).nonzero()[0]
        for i in son:
            if 1 < i + 1 - bas <= 40:
                yield dizi[bas:i + 1]
            bas = i + 1

    def _iskelet(self, c) -> tuple:
        cik = []
        onceki_yuva = False
        for w in c:
            w = int(w)
            if 0 <= w < self.n and self.varlik[w]:
                if not onceki_yuva:
                    cik.append(YUVA)
                onceki_yuva = True
            else:
                cik.append(w)
                onceki_yuva = False
        return tuple(cik)

    # ---------------------------------------------------------------
    def puanla(self, dizi) -> dict:
        """Tek bir uretilmis dizi -> BICIM hukmu + tani.

        `kalip` hukum verir; otekiler NEDEN dustugunu soyler."""
        d = [int(x) for x in dizi]
        kap = next((i for i, w in enumerate(d) if w in self.bitis), None)
        if kap is None:
            # Kapanmamis cumlenin KALIBI yok, ama EKLERI yine de
            # okunabilir -- iki sutun birbirine karismasin.
            iy, n_ = self._ek_say(d)
            return dict(kalip=0, ek_iyi=iy, ek_n=n_, kapanmadi=1, yozlasma=0)
        c = d[:kap + 1]
        yoz = any(c[i] == c[i + 1] == c[i + 2] for i in range(len(c) - 2))
        isk = self._iskelet(np.array(c))
        iy, n_ = self._ek_say(c)
        return dict(kalip=int(isk in self.iskelet), ek_iyi=iy, ek_n=n_,
                    kapanmadi=0, yozlasma=int(yoz))

    def _ek_say(self, c):
        """(dogru yerlesmis ek sayisi, toplam ek sayisi).

        !! CUMLE BASINA DEGIL EK BASINA. Once cumle basina 0/1
        veriliyordu ve hic ek icermeyen cikti VACUOUS geciyordu;
        egitimsiz model 0,94 aliyordu. Ek yoksa paydaya da girmiyor."""
        iyi = tot = 0
        for i, w in enumerate(c):
            if w in self.ek:
                tot += 1
                iyi += int(i > 0 and c[i - 1] in self.ek_oncesi[w])
        return iyi, tot

    def toplu(self, diziler) -> dict:
        p = [self.puanla(d) for d in diziler]
        n = max(1, len(p))
        return {k: sum(x[k] for x in p) / n for k in p[0]}


# =====================================================================
# BILGI -- soru kurma
# =====================================================================
class Sinav:
    """Graftan birim duzeyinde soru kurar.

    Soru = (onek, cevap, kisayol). Onek modele VERILIR, cevap
    uretilir. `kisayol` r2'nin kopruye degil OZNEYE uygulanmis hali --
    ayri sayilir, cunku makul duran bir hata ile rastgele hata ayni
    sey degil."""

    def __init__(self, v, E_ad, ILISKI, TR, kokler, bx, korunan,
                 soru_tip: dict, TR_ILISKI: dict | None = None):
        self.v, self.E_ad, self.ILISKI = v, E_ad, ILISKI
        self.TR, self.TR_ILISKI, self.bx = TR, TR_ILISKI or TR, bx
        self.kokler, self.korunan = kokler, korunan
        self.soru_tip = soru_tip            # varlik tipi -> soru sozcugu
        self.TAMLAYAN = bx["-TAMLAYAN"]
        self.BILDIRME = bx["-BILDIRME"]
        self.SORU = bx["?"]
        self._nb = {}

    def birim(self, ad) -> tuple:
        if ad not in self._nb:
            self._nb[ad] = varlik_birim(
                ad, self.TR, self.kokler, self.bx, self.korunan)
        return self._nb[ad]

    def iliski(self, r) -> tuple:
        """Iliski sozcugu -> birim dizisi (`tezi` -> [tez, -IYELIK]).

        !! `TR` DEGIL `TR_ILISKI`. Graf adi ASCII (`bolumu`), korpus
        Turkce (`bölümü`), ve iliskilerin cevirisi AYRI tabloda.
        Yanlis tabloyla 24 iliskinin 14'u cevrilemiyordu; bolme dogru
        cikiyor ama `bolum` diye bir BIRIM yok, korpusta `bölüm` var.
        Sonuc: soruların cogu kurulamiyordu (seen 57/400)."""
        ix = []
        for w in _tr_ad(self.ILISKI[r], self.TR_ILISKI):
            for x in EK.bol(w, self.kokler, korunan=self.korunan):
                if x not in self.bx:
                    return ()
                ix.append(self.bx[x])
        return tuple(ix)

    def kur(self, zincir, adim):
        """(e, r1[, r2], cevap) -> (onek, cevap, kisayol).

        Yuzey korpusun SORU cumlesiyle ayni dizilis:
          <ozne> -TAMLAYAN <r1> [-TAMLAYAN <r2>] <soru sozcugu> -BILDIRME ?
        `soru_kapisi` bunu korpusta gercekten geciyor mu diye dogrular."""
        z = [int(x) for x in zincir]
        e, rs, ans = z[0], z[1:1 + adim], z[-1]
        onek = self.birim(self.E_ad[e])
        if not onek:
            return None
        for r in rs:
            ri = self.iliski(r)
            if not ri:
                return None
            onek = onek + (self.TAMLAYAN,) + ri
        sz = self.soru_tip.get(int(self.v.tip[ans]))
        if sz is None:
            return None
        onek = onek + (sz, self.BILDIRME, self.SORU)
        cev = self.birim(self.E_ad[ans])
        if not cev:
            return None
        ksy = ()
        if adim == 2:                        # r2 KOPRUYE degil OZNEYE
            h = int(self.v.facts[e, z[2]])
            if h >= 0 and h != ans:
                ksy = self.birim(self.E_ad[h])
        return onek, cev, ksy

    def bolme(self, zincirler, adim):
        """Bir bolmenin butun sorulari. Kurulamayan ATILIR ve SAYILIR."""
        cik, atilan = [], 0
        for z in zincirler:
            s = self.kur(z, adim)
            if s is None:
                atilan += 1
            else:
                cik.append(s)
        return cik, atilan


def tip_haritasi(v, E_ad, sinav: Sinav):
    """birim -> o birimle BASLAYAN varliklarin TIP kumesi.

    `tip` sutunu icin. Hukum vermez, TANI verir: model_13'te tip
    0,9862 iken kimlik 0,0258 idi ve arizanin sekli ancak bu ayrimla
    gorulmustu."""
    h = collections.defaultdict(set)
    for i, ad in enumerate(E_ad):
        b = sinav.birim(ad)
        if b:
            h[b[0]].add(int(v.tip[i]))
    return lambda w: h.get(int(w), set())


def span(u, bitis_ix) -> tuple:
    """Uretilenden CEVAP araligi: ilk bitis birimine kadar."""
    cik = []
    for w in u:
        if int(w) in bitis_ix:
            break
        cik.append(int(w))
    return tuple(cik)


def bicim_puanla(sorular, uretilen, bicim: "Bicim", bitis_ix,
                 tip_of=None) -> dict:
    """BICIM: cumle OGRETILDIGI GIBI kuruldu mu.

    Kullanici, 20 Eylul: *"bicim kisi soruyorsa kisi olarak cevap
    veriyor mu mesela. YANLIS KISI ayri, kisi olarak cevap vermesi
    gerektigini BILMESI ayri."*

    Kullanici, 20 Eylul: *"kalip ve tip degil ayni zamanda EKLER,
    yani genel olarak GRAMERIN TUM PARCALARI."*

        kalip      iskelet korpusun ogrettiklerinden biri mi
                   (kelime yerlesimi, cumle kurulusu)
        ek         her ek, korpusta onunde gorulmus bir birime mi
                   takilmis -- hem "koke mi takildi" hem "ek SIRASI"
                   (-IYELIK -TAMLAYAN gecerli, tersi degil)
        tip        kisi soruluyorsa KISI cevabi verdi mi
                   -- HANGI kisi oldugu BILGI'nin isi, buraya girmez
        kapanmadi  cumle bitmedi
        yozlasma   ayni birim 3+ kez ust uste ("gore gore gore")

    model_13'te bu iki sutun ~1,0 iken BILGI ~0,0 idi; ayrim ancak
    boyle gorunuyor."""
    # !! YALNIZ URETILEN. Onek EKLENMEZ: onek bir SORU ve `?` ile
    # bitiyor; `puanla` ilk bitis biriminde kestigi icin SORUYU
    # puanliyordu ve egitimsiz modelde bile kalip=1,0000 cikiyordu.
    # Korpusta zaten iki ayri cumle var: "... hangisidir?" / "... -dir."
    kal = [bicim.puanla(u) for u in uretilen]
    n = max(1, len(sorular))
    d = {k: sum(x[k] for x in kal) / n
         for k in ("kalip", "kapanmadi", "yozlasma")}
    ek_n = sum(x["ek_n"] for x in kal)
    d["ek"] = sum(x["ek_iyi"] for x in kal) / ek_n if ek_n else float("nan")
    d["ek_n"] = ek_n
    if tip_of is not None:
        ti = 0
        for (_o, cev, _k), u in zip(sorular, uretilen):
            s = span(u, bitis_ix)
            if s and cev and tip_of(s[0]) & tip_of(cev[0]):
                ti += 1
        d["tip"] = ti / n
    return d


def bilgi_puanla(sorular, uretilen, bitis_ix) -> dict:
    """BILGI: cumledeki bilgi dogru mu. KIMLIK uzerinden.

        tam      cevap araligi BIREBIR dogru
        aile     SON parca dogru, varlik yanlis
                 480 kisi / 10 soyad -> aileyi bulup icinden secmek
                 1/48 = %2,08 verir. Bu sutun onu DOGRUDAN sayar.
        kisayol  r2 kopruye degil OZNEYE uygulanmis -- makul duran
                 hata ile rastgele hata ayni sey degil
        bos      hic birim uretmemis -- BOSLUGA KARSI kapi

    SIRA ONEMLI: kisayol dogru cevabin oneki olabilir, once `tam`."""
    t = a = k = b = 0
    for (_onek, cev, ksy), u in zip(sorular, uretilen):
        s = span(u, bitis_ix)
        if not s:
            b += 1
        elif s == cev:
            t += 1
        elif ksy and s == ksy:
            k += 1
        elif cev and s[-1] == cev[-1]:
            a += 1
    n = max(1, len(sorular))
    return dict(tam=t / n, aile=a / n, kisayol=k / n, bos=b / n)


# =====================================================================
# SORU KAPISI -- bosluga karsi
# =====================================================================
def soru_kapisi(sorular, dizi, n=200, en_az=0.5) -> str:
    """Sorunun oneki korpusta GERCEKTEN geciyor mu.

    *Gerekce OLCULDU (model_13, 20 Eylul):* bolucu ciplak -a/-e ekini
    ozel adlarin son unlusunu yiyerek uyguluyordu (Kaya->Kay,
    Manisa->Manis) ve 3.000 sorunun yalniz 119'u korpusta geciyordu.
    Hicbir sayisal kapi bunu gostermemisti."""
    if not sorular:
        return "SORU YOK"
    # !! BAYT ARAMASI. Onceki hal 15,4 M sayiyi tek bir ~100 MB metne
    # ceviriyordu ve 200 arama dakikalar suruyordu. uint16 baytlarinda
    # `bytes.find` C hizinda; hizalamayi (cift ofset) elle denetliyoruz.
    tut = 0
    ornek = list(sorular[:n])
    ham = np.asarray(dizi, np.uint16).tobytes()
    for onek, _c, _k in ornek:
        des = np.asarray(onek, np.uint16).tobytes()
        i = ham.find(des)
        while i >= 0 and i % 2:                  # tek ofset = yanlis hiza
            i = ham.find(des, i + 1)
        tut += i >= 0
    oran = tut / len(ornek)
    assert oran >= en_az, (
        f"SORU KAPISI: {len(ornek)} ornegin yalniz {tut}'u korpusta geciyor "
        f"({oran:.1%}) -- bolme ya da ad esleme bozuk")
    return f"soru kapisi GECTI  {tut}/{len(ornek)} ({oran:.1%})"


# =====================================================================
# URETIM -- toplu, acgozlu.  model_14 API'si
# =====================================================================
@torch.no_grad()
def uret_toplu(mdl, onekler, n_yeni=10, r=0.25, bs=2048, dev="cuda"):
    """Onekler farkli uzunlukta olabilir; her biri KENDI uzunlugunca
    zorlanir, sonra model serbest devam eder.

    Acgozlu. Isin aramasi `mdl.uret` ile soru basina yapilir ve
    PAHALIDIR; toplu olcumde acgozlu kullanilir, isin ORNEKLEMDE."""
    assert not mdl.saat, "saat acikken toplu uretim saat izini tutmali"
    R, C, P = mdl.donme(), mdl.kod(), mdl.p
    D, d = mdl.D, mdl.d
    cik = []
    for i in range(0, len(onekler), bs):
        gr = onekler[i:i + bs]
        L = max(len(o) for o in gr)
        B = len(gr)
        pad = torch.zeros(B, L, dtype=torch.long, device=dev)
        boy = torch.tensor([len(o) for o in gr], device=dev)
        for j, o in enumerate(gr):
            pad[j, :len(o)] = torch.as_tensor(o, device=dev)
        z = P.new_zeros(B, D)
        z[:, :d] = P[pad[:, 0]]
        yol = pad[:, :1]
        for adim in range(1, L + n_yeni):
            z = torch.bmm(R[yol[:, -1]], z.unsqueeze(-1)).squeeze(-1)
            yak2, k = (2 - 2 * (z @ C.T)).clamp(min=0).min(1)
            z = torch.where((yak2 < r * r)[:, None], C[k], z)
            q = F.normalize(z[:, :d], dim=-1)
            sec = (2 - 2 * (q @ P.T)).argmin(1)
            zorla = adim < L
            if zorla:
                sec = torch.where(adim < boy, pad[:, adim], sec)
            yol = torch.cat([yol, sec[:, None]], 1)
        for j, o in enumerate(gr):
            cik.append(yol[j, len(o):].tolist())
    return cik


# =====================================================================
# DOGRULUK -- agacin tamami
# =====================================================================
def dogruluk(mdl, bicim: Bicim, bolmeler: dict, bitis_ix,
             tip_of=None, n_yeni=10, r=0.25, dev="cuda") -> dict:
    """Tek cagri, tam agac.

    `bolmeler`  {"one": [...], "seen": [...], "comp": [...], ...}
    BICIM ayni uretimden okunur -- ayri kosu gerekmez."""
    OGR, CIK = ("one", "seen"), ("comp", "ent")
    sonuc = {"BICIM": {}, "BILGI": {"OGRETILEN": {}, "CIKARIM": {}}}
    for ad, sor in bolmeler.items():
        if not sor:
            continue
        u = uret_toplu(mdl, [s[0] for s in sor], n_yeni, r, dev=dev)
        sonuc["BICIM"][ad] = bicim_puanla(sor, u, bicim, bitis_ix, tip_of)
        grup = "OGRETILEN" if ad in OGR else "CIKARIM" if ad in CIK else None
        b = bilgi_puanla(sor, u, bitis_ix)
        (sonuc["BILGI"][grup] if grup else sonuc.setdefault("DIGER", {}))[ad] = b
    return sonuc


def yaz(s: dict, yaz=print):
    """Agaci okunur bas. Iki blok AYRI okunur: bicim tutup bilgi
    dusuyorsa model dili ogrenmis olgulari ogrenmemis demektir."""
    yaz("DOGRULUK")
    yaz("  BICIM      kalip    ek       tip      kapanmadi  yozlasma   (ek n)")
    for ad, v in s["BICIM"].items():
        yaz(f"    {ad:<8s} {v['kalip']:.4f}   {v['ek']:.4f}   "
            f"{v.get('tip', 0):.4f}   {v['kapanmadi']:.4f}     "
            f"{v['yozlasma']:.4f}     {v['ek_n']}")
    yaz("  BILGI      tam      aile     kisayol  bos")
    for grup in ("OGRETILEN", "CIKARIM"):
        yaz(f"    {grup}")
        for ad, v in s["BILGI"][grup].items():
            yaz(f"      {ad:<6s} {v['tam']:.4f}   {v['aile']:.4f}   "
                f"{v['kisayol']:.4f}   {v['bos']:.4f}")
    for ad, v in s.get("DIGER", {}).items():
        yaz(f"    (hukum disi) {ad:<8s} tam {v['tam']:.4f}")
