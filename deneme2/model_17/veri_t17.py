# -*- coding: utf-8 -*-
"""veri_t17 -- TinyStories.  ETIKET YOK: gorev SONRAKI JETON.

Kullanici, 22 Eylul: *"etiket olmasin. amac dil modeli olarak calisiyor mu
onu anlamak"*.

KELIME SEVIYESI, buyuk harf KORUNMUS.  Kullanici: *"kelime yapalim
avantaji az olabilir ama okumasi daha rahat olur"*.  Avantaj gercekten
az, OLCULDU (ayni 40.000 hikaye):

    birim     FARKLI   hikaye ort   parametre (durum 32)
    KELIME    11.718       192      12.752.289
    BPE       12.588       200      13.698.849

BPE'nin bolmesi nadir kelimede belirgin (hamster -> ham|ster, untangles ->
unt|angles) ama TinyStories'in jetonlarinin cogu sik kelime ve onlar
bolunmuyor; toplama %4,3 olarak yansiyor.  Secim okunabilirlik icin.

BEDELI: nadir kelime <bilinmeyen>e duser ve model onu ASLA uretemez.
BPE'de parcalardan kurulabilirdi.

BUYUK HARF KORUNDU.  Kirpma oldugu icin neredeyse bedava:
    N=4000 kapsama   buyuk korunmus %99,263   hepsi kucuk %99,468
Korunmazsa uretilen metinde ozel ad ve cumle basi kaybolur.

SOZLUK KIRPILIYOR.  Olculdu (24,8M jeton): 2.772 jeton TAM BIR KEZ
geciyor.  Her jeton E + b + M tasiyor (durum=32'de 1.088 sayi), yani bir
kez gecenler olu parametre olur.  Ve C'de ayni kusur olculdu: cevap
olamayacak siniflar kayipta durunca gradyan seyreliyordu, duzeltince
sinav 0,083 -> 0,145.

SURUM V2.  README: "GPT-3.5 uretimleri daha dusuk kalitede".

HIKAYE SINIRI AKISTA ve PENCEREDE.  <hikaye> akista ayrac; pencerede hem
BOS hem EOS (pencere()).  model_16'da tam burasi kaybolmustu: ardisik
birimlerin %12,9'u farkli varliklara aitti ve hicbir sey onu
isaretlemiyordu.

LISANS.  cdla-sharing-1.0.   Eldan & Li, arXiv 2305.07759
"""
from __future__ import annotations

import collections
import hashlib
import os
import re

import numpy as np

SINIR = "<|endoftext|>"       # dosyadaki ayrac
HIKAYE = "<hikaye>"           # bizim jetonumuz
BILINMEYEN = "<bilinmeyen>"
DOLGU = "<dolgu>"          # hikaye bitince kalan yer; maske ile duser
# Kesme isaretli kisaltma TEK birim (don't, Lily's); noktalama AYRI.
JETON = re.compile(r"[A-Za-z]+'[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")

# KIVRIK TIRNAKLAR DUZLESTIRILIYOR.  Korpus ikisini de kullaniyor ve
# kisaltma kurali yalniz duz kesmeyi taniyordu -- OLCULDU (48 MB):
#   kisaltma 93.704:  duz 91.062,  kivrik 2.642  (%2,8 UCE BOLUNUYORDU)
#   "Let's" bazen tek birim, bazen  Let | ' | s
# Ayni kelimenin iki ayri jetonlanmasi.  Tire (- , en, em) DOKUNULMUYOR:
# kurali bozmuyorlar ve anlamlari ayri.
DUZLE = str.maketrans({"’": "'", "‘": "'",
                       "“": '"', "”": '"'})
YAPISIK = set(".,!?;:)]}'\"")      # oncesine bosluk KOYMA
ACAN = set("([{")                  # sonrasina bosluk KOYMA


def _hikayeler(yol, parca_mb=64, en_mb=None):
    """Dosyayi parca parca oku, HIKAYE SINIRINDA kes, hikaye hikaye ver."""
    return _hikayeler_kar(yol, parca_mb * 1024 * 1024,
                          None if en_mb is None else en_mb * 1024 * 1024)


def _hikayeler_kar(yol, parca, dur=None):
    """parca/dur KARAKTER.  En fazla `dur` karakter okunur, yarim kalan son
    hikaye atilir.  Eskiden bolumden artan sayilmiyordu: en_mb=64 fiilen
    128 MB okuyordu (onkayit model_17_TAM1)."""
    art, okunan = "", 0
    with open(yol, encoding="utf-8", errors="replace") as f:
        while dur is None or okunan < dur:
            p = f.read(parca if dur is None else min(parca, dur - okunan))
            if not p:
                if art.strip():
                    yield art.translate(DUZLE)          # dosya bitti: son hikaye tam
                return
            okunan += len(p)
            p = art + p
            k = p.rfind(SINIR)
            if k < 0:
                art = p
                continue
            art, p = p[k + len(SINIR):], p[:k]
            for h in p.split(SINIR):
                if h.strip():
                    yield h.translate(DUZLE)


def sozluk(yol, en: int, parca_mb=64, en_mb=None, yaz=print):
    """En sik `en` kelime + <hikaye> + <bilinmeyen>.  EGITIMDEN cikar."""
    say = collections.Counter()
    for h in _hikayeler(yol, parca_mb, en_mb):
        say.update(JETON.findall(h))
    top = sum(say.values())
    ad = [DOLGU, HIKAYE, BILINMEYEN] + [a for a, _ in say.most_common(en)]
    kap = sum(c for _, c in say.most_common(en)) / top
    yaz("  sozluk  %s farkli kelime gorundu -> en sik %s tutuldu"
        % ("{:,}".format(len(say)), "{:,}".format(en)))
    yaz("          kapsama %%%.3f   disarda %s kelime"
        % (100 * kap, "{:,}".format(max(0, len(say) - en))))
    return ad, {a: i for i, a in enumerate(ad)}


def akis(yol, ix, parca_mb=64, en_mb=None, yaz=print):
    """Hikayeleri TEK akisa diz, aralarina <hikaye>.  int16 dizi.

    int16: en buyuk kimlik 4.002 < 32.767, KAYIPSIZ.  Yarim dosya,
    yarim yukleme, Colab'da yarim RAM."""
    assert len(ix) <= np.iinfo(np.int16).max + 1, "sozluk int16'ya sigmiyor"
    bl, hk = ix[BILINMEYEN], ix[HIKAYE]
    cik, n = [], 0
    for h in _hikayeler(yol, parca_mb, en_mb):
        cik.append(np.fromiter((ix.get(t, bl) for t in JETON.findall(h)),
                               dtype=np.int16))
        cik.append(np.array([hk], dtype=np.int16))
        n += 1
    a = np.concatenate(cik)
    yaz("  akis    %s hikaye   %s jeton   bilinmeyen %%%.2f"
        % ("{:,}".format(n), "{:,}".format(len(a)),
           100 * float((a == bl).mean())))
    return a


def pencere(a: np.ndarray, T: int, uret=None, hikaye=None, dolgu=0):
    """HER HIKAYE = BIR PENCERE, basinda ve sonunda <hikaye>.  Doner: (P, M).

    Kullanici: *"her hikaye bence 1 pencere olmali yoksa modele dogru tam
    hikaye ogretmemis oluruz"* ve *"bos ve eos gerekli"*.

        <hikaye> w1 w2 ... wL <hikaye> <dolgu> ...
        bastaki BOS: w1 de tahmin edilir.  sondaki EOS: model bitisi
        ogrenir, uretim durabilir.

    Kalan yer <dolgu>, maskede False.  Tasma YOK.  L + 2 > T olan hikayeler
    ATILIR (bolmek "tam hikaye" ilkesini bozardi).
    OLCULDU, T=256, BOS/EOS'SUZ hali: hikayelerin %89,8'i kaliyor, dolgu
    %33,7.  T=384 -> %95,7 ama dolgu %53,5;  T=512 -> %98,5 ama dolgu %63,7.

    hikaye verilmezse akis DUZ kesilir (maske hep True).
    """
    if hikaye is None:
        n = len(a) // T
        P = a[:n * T].reshape(n, T)
        M = np.ones(P.shape, dtype=bool)
    else:
        sn = np.flatnonzero(a == hikaye)
        bas = np.concatenate([[0], sn + 1])[:len(sn)]   # her hikayenin basi
        uz = sn - bas                                   # ayrac haric kelime
        tut = (uz > 0) & (uz + 2 <= T)                  # + BOS + EOS
        bas, uz = bas[tut], uz[tut]
        P = np.full((len(bas), T), dolgu, dtype=a.dtype)
        M = np.zeros((len(bas), T), dtype=bool)
        P[:, 0] = hikaye
        for i, (b, L) in enumerate(zip(bas, uz)):
            P[i, 1:L + 1] = a[b:b + L]
            P[i, L + 1] = hikaye
            M[i, :L + 2] = True
    if uret is not None:
        j = uret.permutation(len(P))
        P, M = P[j], M[j]
    return P, M


def coz(P, ad, i=0, en=None) -> str:
    """Pencereyi OKUNUR metne cevir.  Sayi degil METIN gormek icin."""
    d = P[i] if getattr(P, "ndim", 1) > 1 else P      # yigin ya da tek dizi
    d = d[:en]
    s, acik, tirnak = "", False, False
    for x in d:
        t = ad[int(x)]
        if t == HIKAYE:
            s += "\n\n---\n\n"
            tirnak = False
            continue
        # Tirnak hem acar hem kapar; sirayi sayarak ayiriyoruz.
        if t in "\"'" and t != "'":
            yapisik, acacak = tirnak, not tirnak
            tirnak = not tirnak
        else:
            yapisik, acacak = t in YAPISIK, t in ACAN
        s += t if (not s or s.endswith(("\n", " ")) or yapisik or acik) \
            else " " + t
        acik = acacak
    return s.strip()


def iz(*diziler) -> str:
    """Parmak izi -- veri kayarsa kapi yakalasin."""
    h = hashlib.sha256()
    for d in diziler:
        h.update(repr(d).encode() if isinstance(d, list)
                 else np.ascontiguousarray(d).tobytes())
    return h.hexdigest()[:16]


def kur(kok: str, T: int = 128, en: int = 4000, en_mb=None, tohum: int = 0,
        hizali: bool = True, yaz=print):
    """TinyStories -> (ad, EG, DG).

    kok    TinyStories dosyalarinin durdugu klasor (Drive)
    T      pencere uzunlugu.  dizi() dikkati (B,T,T) -- T ile KARESEL.
    en     sozluk kirpmasi (+ <hikaye> + <bilinmeyen>)
    en_mb  yalniz ilk N MB (deneme icin).  None -> hepsi.
    hizali pencereler HIKAYE BASINA hizalansin mi.  Duz kesimde
           tahminlerin %40,8'i hikayesinin basini GORMUYORDU (olculdu).

    Uretilen akis onbellege yazilir; ikinci cagri OKUR (kural 9).
    Veri izini train_17 egitim tensorunden hesaplayip pakete yazar."""
    ob = os.path.join(kok, "onbellek")
    os.makedirs(ob, exist_ok=True)
    yol = {b: os.path.join(kok, "TinyStoriesV2-GPT4-%s.txt" % b)
           for b in ("train", "valid")}
    # SURUM onbellek anahtarinda: yapi degisince eski onbellek SESSIZCE
    # kullanilmasin.  en_mb'li akislar v3: eskisi (v2) iki kati okumustu.
    etiket = ("tam_n%d_v2" % en if en_mb is None
              else "%dmb_n%d_v3" % (en_mb, en))

    ps = os.path.join(ob, "sozluk_%s.npy" % etiket)
    if os.path.exists(ps):
        ad = list(np.load(ps, allow_pickle=True))
        yaz("  sozluk  onbellekten  %s birim" % "{:,}".format(len(ad)))
    else:
        ad, _ = sozluk(yol["train"], en, en_mb=en_mb, yaz=yaz)
        np.save(ps, np.array(ad, dtype=object))
    ix = {a: i for i, a in enumerate(ad)}

    A = {}
    for b in ("train", "valid"):
        p = os.path.join(ob, "akis_%s_%s.npy" % (b, etiket))
        if os.path.exists(p):
            A[b] = np.load(p)
            yaz("  %-6s onbellekten  %s jeton" % (b, "{:,}".format(len(A[b]))))
        else:
            A[b] = akis(yol[b], ix, en_mb=en_mb if b == "train" else None,
                        yaz=yaz)
            np.save(p, A[b])

    uret = np.random.default_rng(tohum)
    hk = ix[HIKAYE] if hizali else None
    EG, EM = pencere(A["train"], T, uret, hk, ix[DOLGU])
    DG, DM = pencere(A["valid"], T, uret, hk, ix[DOLGU])
    yaz("  pencere T=%d   %s   egitim %s   dogrulama %s"
        % (T, "HER HIKAYE BIR PENCERE, <hikaye> ... <hikaye>" if hizali
           else "duz kesim", "{:,}".format(len(EG)), "{:,}".format(len(DG))))
    yaz("  dolgu %%%.1f   etkin is %%%.1f   (L+2 > T olan hikayeler atildi)"
        % (100 * (1 - EM.mean()), 100 * EM.mean()))
    return ad, (EG, EM), (DG, DM)


if __name__ == "__main__":
    KOK = r"G:\Drive'ım\tinystories"
    ad, (EG, EM), (DG, DM) = kur(KOK, T=256, en=4000, en_mb=64)
    print()
    print("iz", iz(ad, EG[:1000], EM[:1000], DG[:1000]))
    print()
    for i in (0, 1):
        print("ORNEK PENCERE %d   %d kelime + %d dolgu"
              % (i, int(EM[i].sum()), int((~EM[i]).sum())))
        print(coz(EG[i][EM[i]], ad))
        print()
