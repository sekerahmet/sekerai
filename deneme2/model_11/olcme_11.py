# -*- coding: utf-8 -*-
"""olcme_11 — SINAV: serbest uretim, tam eslesme.

model_08'de `dogruluk()` cevap yuvalarinda KISITLI argmax yapiyordu
(yalniz varlik jetonlari yarisir) ve her konumu GERCEK onekle
puanliyordu. Karakter duzeyinde "varlik jetonu" diye bir blok YOK, o
yuzden kisit da yok: modele soru verilir, gerisini KENDI yazar, cikan
dizge beklenenle TAM ESLESMELI.

Gecis guvenli: model_08'de olculdu (18 Eylul) ki belirsizligi olmayan
yuzeyde kisitli ile serbest ayni sayiyi veriyor -- alti bolmede de
fark +0.0000.

SORU YUZEYI kullanilir, bildirim DEGIL. Bildirim onegi bu dilde
belirsiz: "X'in danismani" oneginden sonra havuzda 2-hop devami 12,
1-hop cevabi 1 satir (olculdu) ve model dogru olani secip zinciri
uzatiyor. Soru onegi `?` ile bittigi icin tek devami var.
"""
from __future__ import annotations

import re

import numpy as np
import torch

import jeton_11 as J
import metin_11 as MT
import veri_11 as V


def _adlar(G):
    E = [a for t in V.TIPLER for a in G["ad"][t]]
    return E, {a: t for t in V.TIPLER for a in G["ad"][t]}


def sorular(v, G, lst, hop):
    """(onek, beklenen, cevap_adi, kisayol_adi). SORU yuzeyi.

    `kisayol_adi` = facts[e, r2], yani r2'yi KOPRUYE degil OZNEYE
    uygulayinca cikan varlik. Bu kolun ana arizasi o; olculmezse
    "yanlis" ile "kisayol" ayni kefeye girer."""
    E, tipi = _adlar(G)
    out = []
    for z in lst:
        if hop == 1:
            e, r1, a = int(z[0]), int(z[1]), int(z[2])
            # !! SINAV YUZEYI `MT.SINAV_KALIBI` (=0) ile SABIT.
            # model_09'dan kopyalanirken burada `MT.cumle(..., 3, tip)`
            # duruyordu: `cumle` BILDIRIM uretir ve "? " ICERMEZ, ustelik
            # model_11'da 5 degil 4 parametre aliyor -- asagidaki
            # `tam.index("? ")` her halukarda duserdi.
            tam = MT.soru(E[e], V.ILISKI[r1], E[a], tipi[E[a]],
                          MT.SINAV_KALIBI)
            ksy = None
        else:
            e, r1, r2, a = int(z[0]), int(z[1]), int(z[2]), int(z[4])
            tam = MT.zincir_soru(E[e], V.ILISKI[r1], V.ILISKI[r2], E[a],
                                 tipi[E[a]], MT.SINAV_KALIBI)
            h = int(v.facts[e, r2])
            # h < 0: o olgu GRAFTA YOK -> kisayol TIP OLARAK imkansiz.
            # `ent_yok` bolmesinin tanimi bu, ve orada oran 0.000 CIKMALI.
            ksy = MT.ad_tr(E[h]) if h >= 0 and h != a else None
        i = tam.index("? ") + 2
        out.append((tam[:i], tam[i:], MT.ad_tr(E[a]), ksy))
    return out


class Sinav:
    """Bir bolmenin sinavi. BIR KEZ kodlanir, her olcum noktasinda degil.

    Genislik OLCULUR, elle yazilmaz: egitim penceresi 512 ama en uzun
    soru+cevap 98 karakter. 512'de olcmek uretimi 5 kat pahalilastirir
    ve HICBIR SEY eklemez -- model oneki gormeden sonrasini yazmiyor."""

    __slots__ = ("ad", "X", "bas", "bek", "cev", "ksy", "n_uret", "genislik")

    def __init__(self, S, v, G, lst, hop, ad=""):
        q = sorular(v, G, lst, hop)
        self.ad = ad
        self.bek = [b for _o, b, _c, _k in q]
        self.cev = [c for _o, _b, c, _k in q]
        self.ksy = [k for _o, _b, _c, k in q]
        self.n_uret = max(len(b) for b in self.bek) + 2
        # !! GENISLIK = EN UZUN ONEK + n_uret, "onek+cevap" DEGIL.
        # Ilki olculdu ve TASTI: kisa onekli bir satira model uzun bir
        # cevap yazmaya kalkinca imlec dizinin sonunu gecti (IndexError,
        # egitilmemis modelde her seferinde). Uretim ONEK BASINA
        # sinirlidir, cevap uzunluguna gore degil.
        self.genislik = max(len(o) for o, _b, _c, _k in q) + self.n_uret + 1
        self.X = np.full((len(q), self.genislik), J.PAD, np.int64)
        self.bas = np.zeros(len(q), np.int64)
        for i, (o, _b, _c, _k) in enumerate(q):
            j = S.kodla(o)
            self.X[i, :len(j)] = j
            self.bas[i] = len(j)

    def __len__(self):
        return len(self.bek)


@torch.no_grad()
def uret(model, S, X, bas, n_uret, dev, bs=256, nokta_durur=True):
    """OZYINELI serbest uretim. Kisit YOK: butun sozluk uzerinde argmax.

    !! SATIR DONGUSU YOK. Ilk surum her satir icin `bool(bitti[r])` ve
    `int(nx[r])` cagiriyordu; ikisi de GPU->CPU SENKRONU. Olculdu: bir
    olcum noktasi 2.246 ileri gecis x 256 satir = ~575.000 senkron.
    Artik tek tensor, tek `.cpu()`.

    DURMA: nokta, <EOS> ya da <PAD>.
      nokta   cumle bitti -- beklenen cevap da noktayla bitiyor
      EOS     belge siniri. `coz` onu SESSIZCE atiyordu, yani
              "Onur Demir<EOS>'dir." cikti "Onur Demir'dir." diye
              okunur ve YANLIS cevap DOGRU sayilirdi.
      PAD     dolgu jetonu. Egitimde hicbir zaman HEDEF degil
              (ignore_index) ama TAHMIN edilebilir; `coz` onu da
              atiyordu."""
    # BIYOGRAFI icin `nokta_durur=False`: cevap COK CUMLELI, ilk
    # noktada durmak birinci olgudan sonra kesmek olurdu.
    _d = [S.ileri["."], J.EOS, J.PAD] if nokta_durur else [J.EOS, J.PAD]
    dur = torch.tensor(_d, device=dev)
    cik = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].copy()).to(dev)
        p = torch.from_numpy(bas[i:i + bs].copy()).to(dev)
        n, son = len(xb), xb.shape[1]
        ar = torch.arange(n, device=dev)
        bitti = torch.zeros(n, dtype=torch.bool, device=dev)
        kayit = torch.full((n, n_uret), J.PAD, dtype=torch.long, device=dev)
        for t in range(n_uret):
            with torch.autocast(dev, dtype=torch.float16,
                                enabled=(dev == "cuda")):
                lg = model(xb)
            nx = lg.float()[ar, p - 1].argmax(-1)
            nx = torch.where(bitti, torch.full_like(nx, J.PAD), nx)
            xb[ar, p] = torch.where(bitti, xb[ar, p], nx)
            kayit[:, t] = nx
            bitti |= (nx[:, None] == dur[None, :]).any(1)
            p = torch.where(bitti, p, p + 1)
            bitti |= (p >= son)
            if bool(bitti.all()):
                break
        cik += [S.coz(r) for r in kayit.cpu().numpy()]
    return cik


def olc(model, S, sv: "Sinav", dev, bs=256) -> dict:
    """Doner: tam / yakin / kisayol.

    tam      cikti beklenenin BIREBIR AYNISI
    yakin    cikti DOGRU varlik adiyla BASLIYOR, gerisi tutmuyor
             ("Sirnak'ta" / "Sirnak")
    kisayol  cikti KISAYOL varliginin adiyla BASLIYOR (r2 oznede,
             kopru atlanmis)

    !! `in` DEGIL `startswith` -- OLCULDU (18 Eylul). Alt-dizge ile
    48 CIFT varlik adi cakisiyor: "Aydin" bir SEHIR ve ayni zamanda
    "Mehmet Aydin"in icinde. Model sehir sorusuna "Mehmet Aydin'dir."
    deseydi `"Aydin" in cikti` DOGRU doner ve tamamen baska bir varlik
    "yakin" sayilirdi; kisayol orani da ayni sekilde SISERDI.
    Onekle cakisma: 0 cift. Uretim zaten cevabin ILK karakterinden
    basliyor, yani dogru kiyas onek kiyasi."""
    onceki = model.training
    model.eval()
    u = uret(model, S, sv.X, sv.bas, sv.n_uret, dev, bs)
    model.train(onceki)
    return puanla(sv.bek, sv.cev, sv.ksy, u)


def puanla(bek, cev, ksy, cikti) -> dict:
    """PUANLAMA MODELDEN AYRI -- sahte ciktiyla SINANABILSIN diye.

    Ilk surumde `olc`un icindeydi ve tek sinama yolu bir model
    egitmekti. Bir puanlama kusuru ancak 20.000 adimlik kosudan sonra,
    sayilar "biraz tuhaf" gorundugunde fark edilirdi. `test_09` artik
    bunu SAHTE CIKTIYLA siniyor: alt-dizge cakismasi, ek farki, kisayol.

    SIRA ONEMLI: tam -> yakin -> kisayol. Kisayol adi dogru cevabin
    ONEKI olabilir; o durumda once dogru cevap sayilmali."""
    t = y = k = 0
    for b, c_, ks, c in zip(bek, cev, ksy, cikti):
        if c == b:
            t += 1
        elif c.startswith(c_):
            y += 1
        elif ks and c.startswith(ks):
            k += 1
    n = max(1, len(bek))
    return dict(tam=t / n, yakin=y / n, kisayol=k / n)



# ======================= DURUSTLUK =====================================
# Kullanici karari, 18 Eylul: *"bir yapay zeka modelinin bilmiyorum
# demesi bir yenilik olabilir. yok demesi. durustluk vs."*  ve
# *"biz yok degil de OLMADIGINI olcecegiz."*
#
# TEK OLCU YANILTIR: her seye "yok" diyen model DOGRU RET'te tavana
# cikar. O yuzden CIFT olculur ve ikisi BIRLIKTE raporlanir.
#
#     DOGRU RET   cevapsiz soruya "yok" dedi mi        (yuksek IYI)
#     KACAMAK     BILDIGI soruya "yok" dedi mi         (yuksek KOTU)
#
# Sinav TUTULAN veriden kurulur: o ad ve o (tip, iliski) cifti egitimde
# HIC gecmez (`korpus_11.reddetme_bolme`). Gordugunu reddetmek EZBER,
# gormedigini reddetmek GENELLEME -- olcmek istedigimiz ikincisi.
RET_KOK = ("yok", "olmaz", "bilmiyorum", "tanımıyorum", "tanımam")
_RET_RE = re.compile(r"\b(" + "|".join(RET_KOK) + r")", re.IGNORECASE)


def ret_mi(c: str) -> bool:
    """Cikti bir REDDETME mi.

    !! KELIME BASI ile eslenir, alt dizge DEGIL. Kokler TAHMIN degil:
    `metin_11.YOK_VARLIK` / `YOK_ILISKI` kaliplarindan geliyorlar ve
    `test_11` §3i ikisini de GERCEK korpusa karsi siniyor -- uretilen
    her reddetme cumlesi YAKALANMALI, hicbir OLGU cumlesi yakalanMAMALI.
    Boylece bu liste benim Turkce sezgime degil VERIYE bagli."""
    return bool(_RET_RE.search(c))


class RetSinavi:
    """TUTULAN ad ve ciftlerden kurulan CEVAPSIZ sorular.

    `Sinav`dan farki: BEKLENEN metin YOK. Dogru cevap tek bir dizge
    degil -- "reddet" diye bir DAVRANIS. O yuzden puanlama `ret_mi`
    ile, tam eslesme ile degil."""

    __slots__ = ("ad", "X", "bas", "n_uret", "genislik", "tur", "onek")

    def __init__(self, S, G, bolme, n_ad=200, n_cift=200, tohum=0, ad=""):
        import metin_11 as _MT
        rs = np.random.default_rng(9300 + tohum)
        E, tipi = _adlar(G)
        var = {}
        for (a_, r_) in G["olgu"]:
            var.setdefault(tipi[a_], set()).add(r_)
        hedef = {}
        for (a_, r_), h_ in G["olgu"].items():
            hedef[r_] = tipi[h_]
        q, tur = [], []
        # 1) TUTULAN UYDURMA AD -- tipin GERCEKTEN tasidigi bir iliski
        hav = [(t, a_) for t, L in sorted(bolme["ad_tut"].items())
               if var.get(t) for a_ in L]
        for i in rs.permutation(len(hav))[:n_ad]:
            t, a_ = hav[int(i)]
            rl = sorted(var[t])
            r_ = rl[int(rs.integers(len(rl)))]
            c = _MT.yok_varlik(a_, t, r_, hedef[r_], 0)
            q.append(c[:c.index("? ") + 2]); tur.append("ad")
        # 2) TUTULAN IMKANSIZ CIFT -- GERCEK varlik, imkansiz iliski
        ct = [x for x in bolme["cift_tut"] if G["ad"][x[0]]]
        for _ in range(n_cift):
            t, r_ = ct[int(rs.integers(len(ct)))]
            a_ = G["ad"][t][int(rs.integers(len(G["ad"][t])))]
            c = _MT.yok_iliski(a_, t, r_, hedef[r_], 0)
            q.append(c[:c.index("? ") + 2]); tur.append("cift")
        self.ad, self.onek, self.tur = ad, q, tur
        # Uretim uzunlugu: en uzun REDDETME cevabi kadar. Kisa tutulursa
        # model reddetmeye BASLAYIP kesilir ve "ret degil" sayilirdi.
        self.n_uret = 48
        self.genislik = max(len(o) for o in q) + self.n_uret + 1
        self.X = np.full((len(q), self.genislik), J.PAD, np.int64)
        self.bas = np.zeros(len(q), np.int64)
        for i, o in enumerate(q):
            j = S.kodla(o)
            self.X[i, :len(j)] = j
            self.bas[i] = len(j)

    def __len__(self):
        return len(self.onek)


def durustluk_puanla(ret_cikti, ret_tur, bilgi_cikti, bilgi_bek) -> dict:
    """PUANLAMA MODELDEN AYRI -- `puanla` ile ayni gerekce.

    dogru_ret    cevapsiz sorularin kaci REDDEDILDI      (yuksek IYI)
    ret_ad       ...bunlarin UYDURMA AD yakasi
    ret_cift     ...IMKANSIZ CIFT yakasi
    kacamak      CEVAPLANABILIR sorularin kaci REDDEDILDI (yuksek KOTU)
    dogru_cevap  cevaplanabilirlerin kaci DOGRU           (kiyas icin)

    !! `kacamak` olmadan `dogru_ret` OKUNMAZ. Her seye "yok" diyen
    model dogru_ret = 1.000, kacamak = 1.000 verir; ikincisi olmasa
    birincisi basari gibi gorunurdu."""
    n = max(1, len(ret_cikti))
    d = sum(ret_mi(c) for c in ret_cikti)
    na = max(1, sum(1 for t in ret_tur if t == "ad"))
    nc = max(1, sum(1 for t in ret_tur if t == "cift"))
    m = max(1, len(bilgi_cikti))
    return dict(
        dogru_ret=d / n,
        ret_ad=sum(ret_mi(c) for c, t in zip(ret_cikti, ret_tur)
                   if t == "ad") / na,
        ret_cift=sum(ret_mi(c) for c, t in zip(ret_cikti, ret_tur)
                     if t == "cift") / nc,
        kacamak=sum(ret_mi(c) for c in bilgi_cikti) / m,
        dogru_cevap=sum(c == b for c, b in zip(bilgi_cikti, bilgi_bek)) / m,
    )


@torch.no_grad()
def durustluk_olc(model, S, rs_: "RetSinavi", bilgi_sv: "Sinav", dev,
                  bs=256) -> dict:
    """DURUSTLUK: iki sinav, tek rapor."""
    onceki = model.training
    model.eval()
    u_ret = uret(model, S, rs_.X, rs_.bas, rs_.n_uret, dev, bs)
    u_bil = uret(model, S, bilgi_sv.X, bilgi_sv.bas, bilgi_sv.n_uret, dev, bs)
    model.train(onceki)
    return durustluk_puanla(u_ret, rs_.tur, u_bil, bilgi_sv.bek)


def ret_ornekleri(model, S, rs_: "RetSinavi", dev, n=8, tohum=0):
    """GOZLE BAKMAK icin: (soru, model ne yazdi, ret sayildi mi)."""
    r = np.random.default_rng(tohum)
    i = r.choice(len(rs_), size=min(n, len(rs_)), replace=False)
    onceki = model.training
    model.eval()
    u = uret(model, S, rs_.X[i], rs_.bas[i], rs_.n_uret, dev, bs=len(i))
    model.train(onceki)
    return [(rs_.onek[int(j)], c, ret_mi(c)) for j, c in zip(i, u)]


def ornekler(model, S, sv: "Sinav", dev, n=8, tohum=0):
    """GOZLE BAKMAK icin: (onek, beklenen, model ne yazdi)."""
    rs = np.random.default_rng(tohum)
    i = rs.choice(len(sv), size=min(n, len(sv)), replace=False)
    X, bas = sv.X[i], sv.bas[i]
    onceki = model.training
    model.eval()
    u = uret(model, S, X, bas, sv.n_uret, dev, bs=len(i))
    model.train(onceki)
    return [(S.coz(X[k, :bas[k]]), sv.bek[int(j)], c)
            for k, (j, c) in enumerate(zip(i, u))]


# ======================= BIYOGRAFI ======================================
# Kullanici karari, 18 Eylul: *"bu bir dil modeli. dil modelinin basarili
# olmasi lazim. artik bu bir test degil."*
#
# Tek olgulu sinav "cevabi biliyor mu" der. Biyografi sinavi BASKA bir
# sey sorar: varligin BUTUN bilgisi erisilebilir mi, ve model onu KENDI
# kurdugu bir sirayla mi veriyor yoksa egitimdeki belgeyi mi geri
# oynatiyor. Ikincisi bu projenin ana sorusunun (EZBER mi KODLAMA mi)
# yeni bir yuzeyde sorulmus hali.


def biyografi_sinavi(S, v, G, ents, t_len=640):
    """(X, bas, hedef) -- tetikleyici onegi + varligin GERCEK olgulari.

    `hedef[i]` = {(o, r): {o cumlenin butun yuzeyleri}}. Uretilen bir
    cumle bu kumelerden birinde ise DOGRU, ve HANGI olgu oldugu bilinir.
    Yuzey kumesi kullaniliyor cunku model bicimi serbest seciyor --
    "Mus, X'in memleketidir." ile "X'in memleketi Mus'tur." AYNI olgu."""
    E, tipi = _adlar(G)
    ait = {}
    for e, r, a in v.one:
        ait.setdefault(int(e), []).append((int(e), int(r), int(a)))
    X = np.full((len(ents), t_len), J.PAD, np.int64)
    bas = np.zeros(len(ents), np.int64)
    hedef = []
    for i, e in enumerate(ents):
        o = MT.biyografi_sorusu(E[e]) + " "
        j = S.kodla(o)
        X[i, :len(j)] = j
        bas[i] = len(j)
        # !! ESKI API: `MT.cumle(..., bicim, tip)` + `MT.N_BICIM`.
        # model_11'da `cumle` 4 parametre aliyor ve bildirim sayisi
        # `N_BILDIRIM` (17). Kopyadan gelen hali CALISMA ZAMANINDA
        # duserdi -- model_09'da biyografi olcusu zaten hic
        # kosmamisti, o yuzden gorulmedi. §0c kapisi artik statik
        # olarak yakaliyor.
        hedef.append({(oo, rr): {MT.cumle(E[oo], V.ILISKI[rr], E[aa], b)
                                 for b in range(MT.N_BILDIRIM)}
                      for oo, rr, aa in ait[e]})
    return X, bas, hedef


def _yuzey_haritasi(v, G):
    """cumle dizgesi -> (ozne, iliski). Butun BILDIRIM yuzeyleri.

    Hem egitim belgesini hem modelin ciktisini AYNI haritayla
    cozumluyoruz; yoksa "yeni siralama" olcusu iki farkli tanimi
    kiyaslardi."""
    E, tipi = _adlar(G)
    h = {}
    for e, r, a in v.one:
        e, r, a = int(e), int(r), int(a)
        for b in range(MT.N_BILDIRIM):
            h[MT.cumle(E[e], V.ILISKI[r], E[a], b)] = (e, r)
    return h


def olgu_sirasi(cumleler, harita):
    """Cumle listesinden OLGU SIRASI. Taninmayan cumle ATLANIR."""
    out = []
    for c in cumleler:
        k = harita.get(c.strip())
        if k is not None and k not in out:
            out.append(k)
    return tuple(out)


def egitim_siralari(v, G, belgeler):
    """varlik -> egitimde GORULEN olgu siralarinin kumesi.

    `Belge.olgu` bir KUME, yani SIRAYI TASIMIYOR -- ilk yazimda onu
    kullanmistim ve butun varliklar "tek siralama" cikiyordu. Sira
    ancak CUMLELERDEN cozumlenir."""
    h = _yuzey_haritasi(v, G)
    out = {}
    for b in belgeler:
        if b.e < 0:
            continue
        # Tetiklenmis belgenin ILK cumlesi onek -- haritada yok, atlanir.
        out.setdefault(b.e, set()).add(olgu_sirasi(b.cumle, h))
    return out


def _alt_dizi(kisa, uzun) -> bool:
    """`kisa`, `uzun` icinde SIRAYI BOZMADAN geciyor mu."""
    it = iter(uzun)
    return all(x in it for x in kisa)


def yeni_siralama(uretilen, egitim) -> bool:
    """Uretilen olgu sirasi egitimdeki HICBIR belgeyle tutarli mi?

    !! TAM ESITLIK DEGIL, ALT DIZI. Ilk surum `tuple(bulunan) not in
    egt` diyordu ve OLCULDU (18 Eylul, ezber testi) ki bu SAHTE pozitif
    uretiyor: model tek bir olguyu atlayinca dizi hicbir egitim
    siralamasina ESIT olamiyor ve "YENI SIRALAMA" sayiliyordu.
    Sayilar:

        EOS yok   recall 1.0000   yeni_sira 0.0000
        EOS var   recall 0.8889   yeni_sira 1.0000   <- MODEL DAHA KOTU,
                                                       olcu DAHA IYI dedi

    Yani olcu, EKSIK HATIRLAMAYI 'kendi sirasini kurdu' diye okuyordu --
    tam ters yon. Alt-dizge/`startswith` kusuruyla ayni aileden.

    Dogrusu: uretilen olgularin GORELI sirasi egitimdeki bir siralamayla
    tutarli mi. Tutarliysa model o belgeyi (kismen de olsa) GERI
    OYNATIYOR; hicbiriyle tutarli degilse sirayi KENDI kurmus."""
    if len(uretilen) < 2:
        return False        # tek olguda "sira" diye bir sey YOK
    return not any(_alt_dizi(uretilen, t) for t in egitim)


def biyografi_olc(model, S, X, bas, hedef, egitim_sirasi, dev, bs=64,
                  n_uret=520):
    """Doner: recall / kesinlik / yeni_sira / uzunluk + ornekler.

    recall      varligin olgularinin kaci SOYLENDI
    kesinlik    soylenen cumlelerin kaci GERCEK bir olgu
    yeni_sira   uretilen olgu SIRASI egitimdeki 5 belgeden HICBIRI degil
                -> belgeyi geri oynatmiyor, KENDI kuruyor
    """
    onceki = model.training
    model.eval()
    u = uret(model, S, X, bas, n_uret, dev, bs, nokta_durur=False)
    model.train(onceki)
    rec, kes, yeni, uzun = [], [], [], []
    for c, h, egt in zip(u, hedef, egitim_sirasi):
        cum = [x.strip() + "." for x in c.split(". ") if x.strip()]
        cum = [x for x in cum if len(x) > 1]
        bulunan, dogru = [], 0
        for x in cum:
            for k, yuzeyler in h.items():
                if x in yuzeyler:
                    dogru += 1
                    if k not in bulunan:
                        bulunan.append(k)
                    break
        rec.append(len(bulunan) / max(1, len(h)))
        kes.append(dogru / max(1, len(cum)))
        uzun.append(len(cum))
        yeni.append(yeni_siralama(tuple(bulunan), egt))
    n = max(1, len(u))
    return dict(recall=sum(rec) / n, kesinlik=sum(kes) / n,
                yeni_sira=sum(yeni) / n, cumle=sum(uzun) / n,
                ornek=u[:3])
