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

HIKAYE SINIRI AKISTA.  <hikaye> jetonu sozlukte duruyor.  model_16'da
tam burasi kaybolmustu: ardisik birimlerin %12,9'u farkli varliklara
aitti ve hicbir sey onu isaretlemiyordu.

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
    art, okunan = "", 0
    n = parca_mb * 1024 * 1024
    dur = None if en_mb is None else en_mb * 1024 * 1024
    with open(yol, encoding="utf-8", errors="replace") as f:
        while True:
            p = f.read(n)
            if not p:
                break
            p = art + p
            k = p.rfind(SINIR)
            if k < 0:
                art = p
                continue
            art, p = p[k + len(SINIR):], p[:k]
            okunan += len(p)
            for h in p.split(SINIR):
                if h.strip():
                    yield h.translate(DUZLE)
            if dur and okunan >= dur:
                return
    if art.strip():
        yield art.translate(DUZLE)


def sozluk(yol, en: int, parca_mb=64, en_mb=None, yaz=print):
    """En sik `en` kelime + <hikaye> + <bilinmeyen>.  EGITIMDEN cikar."""
    say = collections.Counter()
    for h in _hikayeler(yol, parca_mb, en_mb):
        say.update(JETON.findall(h))
    top = sum(say.values())
    ad = [HIKAYE, BILINMEYEN] + [a for a, _ in say.most_common(en)]
    kap = sum(c for _, c in say.most_common(en)) / top
    yaz("  sozluk  %s farkli kelime gorundu -> en sik %s tutuldu"
        % ("{:,}".format(len(say)), "{:,}".format(en)))
    yaz("          kapsama %%%.3f   disarda %s kelime"
        % (100 * kap, "{:,}".format(max(0, len(say) - en))))
    return ad, {a: i for i, a in enumerate(ad)}


def akis(yol, ix, parca_mb=64, en_mb=None, yaz=print):
    """Hikayeleri TEK akisa diz, aralarina <hikaye>.  int32 dizi."""
    bl, hk = ix[BILINMEYEN], ix[HIKAYE]
    cik, n = [], 0
    for h in _hikayeler(yol, parca_mb, en_mb):
        cik.append(np.fromiter((ix.get(t, bl) for t in JETON.findall(h)),
                               dtype=np.int32))
        cik.append(np.array([hk], dtype=np.int32))
        n += 1
    a = np.concatenate(cik)
    yaz("  akis    %s hikaye   %s jeton   bilinmeyen %%%.2f"
        % ("{:,}".format(n), "{:,}".format(len(a)),
           100 * float((a == bl).mean())))
    return a


def pencere(a: np.ndarray, T: int, uret=None):
    """Akisi T uzunlugunda pencerelere kes.  Dolgu YOK, artan atilir.

    KARISTIRILIR: model_16'da akis blok dizilimliydi ve soru orani ilk
    %40'ta %0,54, sonrasinda %6,64 cikmisti."""
    n = len(a) // T
    P = a[:n * T].reshape(n, T)
    return P if uret is None else P[uret.permutation(n)]


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
        yaz=print):
    """TinyStories -> (ad, EG, DG).

    kok    TinyStories dosyalarinin durdugu klasor (Drive)
    T      pencere uzunlugu.  dizi() dikkati (B,T,T) -- T ile KARESEL.
    en     sozluk kirpmasi (+ <hikaye> + <bilinmeyen>)
    en_mb  yalniz ilk N MB (deneme icin).  None -> hepsi.

    Uretilen akis onbellege yazilir; ikinci cagri OKUR (kural 9).
    Iz her cagrida YENIDEN hesaplanir."""
    ob = os.path.join(kok, "onbellek")
    os.makedirs(ob, exist_ok=True)
    yol = {b: os.path.join(kok, "TinyStoriesV2-GPT4-%s.txt" % b)
           for b in ("train", "valid")}
    etiket = "%s_n%d" % ("tam" if en_mb is None else "%dmb" % en_mb, en)

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
    EG, DG = pencere(A["train"], T, uret), pencere(A["valid"], T, uret)
    yaz("  pencere T=%d   egitim %s   dogrulama %s   hikaye siniri %s kez"
        % (T, "{:,}".format(len(EG)), "{:,}".format(len(DG)),
           "{:,}".format(int((EG == ix[HIKAYE]).sum()))))
    return ad, EG, DG


if __name__ == "__main__":
    KOK = r"G:\Drive'ım\tinystories"
    ad, EG, DG = kur(KOK, T=128, en=4000, en_mb=64)
    print()
    print("iz", iz(ad, EG[:1000], DG[:1000]))
    print()
    print("ORNEK PENCERE  (egitim, T=128)")
    print(coz(EG, ad, 0))
