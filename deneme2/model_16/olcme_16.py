"""OLCU -- TEK ANALIZ: soru soruldu, cevap DOGRU MU.

Kullanici, 22 Eylul: "yuva 1 hic bir zaman olcu olmadi, o bir analizdi.
burda tek analiz var: soru sorduk, cevap dogru mu yanlis mi, bu kadar."
ve: "one / comp / ent gibi kavramlari bir turlu kaldiramadik, onun yerine
ezber cikarim gelmesi gerekiyor."

AYRIM TEK SORUDA: cevap korpusta YAZIYOR MU?

  EZBER    ezber_olgu      tek olgu, tek cumlede ogretildi
           ezber_zincir    bilesim, egitimde GORULDU
  CIKARIM  cikarim_gorulmemis  o (varlik,r1,r2) uclusu korpusta YOK
           cikarim_yabanci     varlik HIC zincir basi olmamis

YUZEY `metin_16.sinav_yuzeyi`den gelir, ELLE DIZILMEZ -- sinavin yuzeyi
korpusunkiyle YAPI GEREGI ayni kalsin diye.  model_14'te bu bir kez elle
dizilmisti, tokenizer degisince sinav sessizce KOSMAZ hale gelmisti.

KUYRUKTAKI EK TOLERE EDILIR.  Beklenen cevap CIPLAK ad ("Elif Aydin"),
modelin urettigi dilbilgisel Turkce ("Elif Aydin'dir").  Birebir esitlik
ararsak gercek sinyal sifir diye raporlanir; model_14'te tam bu olmustu
(dort kosuda da 0,0000).  Kural: uretilen ya ADIN KENDISI, ya da ad + "'".

ZINCIRDEKI YERI.  Kim kimi cagiriyor, bu dosya nerede:
(model BIRIM goruyor -- karakter yalnizca ara adim ve kapi)

  veri_16     graf: 1608 varlik, 24 iliski, olgular
  metin_16    graf -> duz Turkce cumle
  korpus_16   cumle -> belge -> paketlenmis akis
  jeton_16    KARAKTER sozlugu + GIDIS-DONUS KAPISI
  birim_16    metin -> sayim -> kok havuzu -> BIRIM AKISI -> pencere
  ek_16       kelime -> kok + ek    (Turkce morfolojisi)

  taban_16    bolmeler (ezber_* / cikarim_*) + Ayar tanimi
  ayar_16     dugmeler
  hazirla_16  veriyi dosyaya yazar, Colab Drive'dan OKUR

  model_16    MIMARI -- model_15'ten
  kos_16      egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu   <-- BU DOSYA
"""
import torch

import metin_16 as MT

BITIS = "."          # cevap cumlesi noktayla biter
ENUZUN = 48          # bir cevap icin en fazla kac karakter uretilir

EZBER = ("ezber_olgu", "ezber_zincir")
CIKARIM = ("cikarim_gorulmemis", "cikarim_yabanci")


def _kodlayici(d):
    ileri = {c: i + 2 for i, c in enumerate(d["harf"])}
    geri = {i + 2: c for i, c in enumerate(d["harf"])}
    geri[d["PAD"]] = "\x00"
    geri[d["EOS"]] = "\n"
    return ileri, geri


def yuzeyler(d, ad, en=None, tohum=0):
    """Bolmeden (onek, cevap) ciftleri.  Metin, jeton DEGIL."""
    E, IL, TIP = d["varlik"], d["iliski"], d["tip_ad"]
    tip = d["tip"]
    L = d["bolme"][ad]
    if en and len(L) > en:
        g = torch.Generator().manual_seed(tohum + 7)
        L = [L[i] for i in torch.randperm(len(L), generator=g)[:en].tolist()]
    out = []
    for z in L:
        z = [int(x) for x in z]
        e, cev = z[0], z[-1]
        rs = z[1:2] if len(z) == 3 else z[1:3]      # 1 adim mi 2 adim mi
        onek, c, _kanit = MT.sinav_yuzeyi(
            E[e], [IL[r] for r in rs], E[cev], TIP[int(tip[cev])])
        out.append((onek, c))
    return out


def sor(m, d, ad, aygit="cuda", en=2000, tohum=0, parca=1000, ayrinti=False):
    """TEK ANALIZ: onegi ver, uret, DOGRU MU.

    Onekler farkli uzunlukta oldugu icin UZUNLUGA GORE obeklenip
    yiginlaniyor -- tensor dikdortgen olmak zorunda.  Obek bir mufredat
    degil, yalniz sekil."""
    ileri, geri = _kodlayici(d)
    yuz = yuzeyler(d, ad, en, tohum)
    if not yuz:
        return (float("nan"), []) if ayrinti else float("nan")

    kova = {}
    for onek, cev in yuz:
        j = [ileri[c] for c in onek]
        kova.setdefault(len(j), []).append((j, cev, onek))

    dog = say = 0
    ornek = []
    with torch.no_grad():
        for kalem in kova.values():
            for i in range(0, len(kalem), parca):
                oh = kalem[i:i + parca]
                w = torch.tensor([j for j, _, _ in oh], device=aygit)
                U = m.uret_dizi(w, ENUZUN).cpu()
                for (_, cev, onek), u in zip(oh, U):
                    s = "".join(geri.get(int(t), "?") for t in u.tolist())
                    s = s.split(BITIS)[0].lstrip()   # cevap araligi
                    ok = s == cev or s.startswith(cev + "'")
                    dog += ok; say += 1
                    if ayrinti and len(ornek) < 12:
                        ornek.append((onek, s, cev, ok))
    oran = dog / say
    return (oran, ornek) if ayrinti else oran


def olcut(d, aygit="cuda", en=2000, tam_en=10 ** 9):
    """kos_16'nin bekledigi bicim:  olcut(m, "ezber"|"cikarim", tam=False)"""
    def f(m, taraf, tam=False):
        adlar = EZBER if taraf == "ezber" else CIKARIM
        n = tam_en if tam else en
        p = [sor(m, d, a, aygit=aygit, en=n) for a in adlar]
        a = [len(d["bolme"][x]) for x in adlar]
        return sum(o * k for o, k in zip(p, a)) / sum(a)   # buyukluge gore
    return f


def kirilim(m, d, aygit="cuda", en=2000):
    """SONUC istatistigi, bolme bolme.  Yine 'cevap dogru mu'."""
    return {a: (sor(m, d, a, aygit=aygit, en=en), len(d["bolme"][a]))
            for a in EZBER + CIKARIM + ("kapi_kisayolsuz",)}


def etiket_kapisi(d, ornek=60, yaz=print):
    """OLCUNUN KENDI SAGLIGI -- yetenek olcmez.

    `cikarim_*` demek "cevap korpusta YOK" demek.  Dogruysa kanit dizisi
    korpusta BULUNMAMALI; `ezber_zincir`inki ise BULUNMALI.  Tersi cikarsa
    etiketler yalan ve BUTUN sayilar okunamaz."""
    _, geri = _kodlayici(d)
    metin = "".join(geri.get(int(t), "?") for t in d["X"].reshape(-1).tolist())
    E, IL, TIP = d["varlik"], d["iliski"], d["tip_ad"]
    tip = d["tip"]
    sonuc = {}
    for ad, beklenen in (("ezber_zincir", True),
                         ("cikarim_gorulmemis", False),
                         ("cikarim_yabanci", False)):
        L = d["bolme"][ad][:ornek]
        n = 0
        for z in L:
            z = [int(x) for x in z]
            _o, _c, kanit = MT.sinav_yuzeyi(
                E[z[0]], [IL[r] for r in z[1:3]], E[z[-1]],
                TIP[int(tip[z[-1]])])
            n += kanit in metin
        sonuc[ad] = (n, len(L), beklenen)
        yaz(f"  {ad:<20s} {n:3d}/{len(L):<3d} korpusta"
            f"   beklenen {'VAR' if beklenen else 'YOK'}"
            f"   {'GECTI' if (n > 0) == beklenen else 'KALDI'}")
    return sonuc
