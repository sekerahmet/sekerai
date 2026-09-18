# -*- coding: utf-8 -*-
"""metin_09 — GRAFTAN DUZ TURKCE METIN. model_09'un veri katmani.

model_08'de zincir once JETONA cevriliyordu, metin ancak `analiz.oku`
ile GERI okunuyordu. Burada yon tersine dondu: **metin asil kaynak**,
jeton ondan cikar. Gercek dil modeli egitiminde de boyledir.

    graf --> metin_09 --> karakter jetonlari

!! BIR CUMLE = BIR OLGU, ISTISNASIZ. Bir cumle iki iliskiyi birden
soyleyemez: soylerse kopru baglama yazilmis olur, tutulmus 2-hop
zinciri kopyalanarak cevaplanir ve sinav coker. (bioS de bu kisiti
saglar: her cumle tek oznitelik.)

Unlu uyumu `veri_09`un tablolarindan gelir, burada YENIDEN HESAPLANMAZ.
"""
from __future__ import annotations

import veri_09 as V

# --- bicimler ------------------------------------------------------------
# Hepsi DOGAL Turkce. model_08'in "ILISKI BASTA" bicimi (`Kardesidir Elif
# Aydin, Ahmet Aydin'in.`) BURAYA ALINMADI -- kimse oyle konusmaz ve
# kullanici da isaret etmisti.
# "kopulasiz" ({X'in} {r} {Y}.) KALDIRILDI: listede dogal ama AKAN
# METINDE eksik cumle -- "Merve Aydin'in ogrencisi Hulya Ozturk."
# Bildirim cumlelerinin %25'i oyleydi. Geri almak tek satir.
BICIM = ("kanonik", "yuklem basta", "ozne basta", "soru")
N_BICIM = len(BICIM)


def _tr(ad: str) -> str:
    """Graf adi (Ibrahim_Yilmaz) -> ekranda gorunen Turkce (İbrahim Yılmaz)."""
    return " ".join(V.TR.get(w, w) for w in ad.split("_"))


ad_tr = _tr          # disari acik ad: graf adi -> Turkce


def _nin(k: str, ozel: bool) -> str:
    """Tamlayan eki. Ozel isimde kesme isareti VAR, cins isimde YOK."""
    return ("'" if ozel else "") + V.EK_NIN[V._ek_nin(k)]


def _dir(k: str, ozel: bool) -> str:
    """Bildirme eki. Ayni kural."""
    return ("'" if ozel else "") + V.EK_DIR[V._ek_dir(k)]


def cumle(ozne: str, iliski: str, cevap: str, bicim: int, tip: str) -> str:
    """TEK olgunun cumlesi. `ozne`/`cevap` GRAF ADI, `tip` cevabin tipi."""
    X, Y = _tr(ozne), _tr(cevap)
    r = V.TR_ILISKI[iliski]
    Xn = X + _nin(X, True)                      # Ahmet Aydın'ın
    Yd = Y + _dir(Y, True)                      # Elif Aydın'dır
    rd = r + _dir(r, False)                     # kardeşidir
    b = bicim % N_BICIM
    if b == 0:
        return f"{Xn} {r} {Yd}."
    if b == 1:
        return f"{Yd}, {Xn} {r}."
    if b == 2:
        return f"{Y}, {Xn} {rd}."
    sz = V.SORU_SOZ[tip]
    return f"{Xn} {r} {sz}{_dir(sz, False)}? {Yd}."


def zincir(ozne: str, r1: str, r2: str, cevap: str, bicim: int,
           tip: str) -> str:
    """IKI adimli cumle. KOPRU GECMEZ -- bu kolun butun sorusu bu."""
    X, Y = _tr(ozne), _tr(cevap)
    a, b_ = V.TR_ILISKI[r1], V.TR_ILISKI[r2]
    Xn = X + _nin(X, True)
    il = f"{a}{_nin(a, False)} {b_}"             # kardeşinin bölümü
    Yd = Y + _dir(Y, True)
    bd = b_ + _dir(b_, False)
    b = bicim % N_BICIM
    if b == 0:
        return f"{Xn} {il} {Yd}."
    if b == 1:
        return f"{Yd}, {Xn} {il}."
    if b == 2:
        return f"{Y}, {Xn} {a}{_nin(a, False)} {bd}."
    sz = V.SORU_SOZ[tip]
    return f"{Xn} {il} {sz}{_dir(sz, False)}? {Yd}."


def kimlik(ad: str, tip: str) -> str:
    """Sifir adim ozdeslik (identity bridge, arXiv 2509.24653)."""
    X = _tr(ad)
    sz = V.SORU_SOZ[tip]
    return f"{X} {sz}{_dir(sz, False)}? {X + _dir(X, True)}."


def biyografi_sorusu(ad: str) -> str:
    """BIYOGRAFI TETIKLEYICISI: cok olguluk bir cevap ISTEYEN yuzey.

    Kullanici karari, 18 Eylul: *"biyografi ayri soruyla isteyebiliriz
    illaki kimdir diye sormamiza gerek yok"*. Bu karar tasarimi TEMIZLEDI:
    "kimdir?" tetikleyici olsaydi KIMLIK cumlesiyle ayni onegi paylasirdi
    ("Ayse Kaya kimdir? Ayse Kaya'dir." vs "... <biyografi>"), ve greedy
    cozumleme hangisi daha SIKSA onu secerdi -- yani istenen davranis
    SANSA kalirdi. Kacinmak icin bir ALTKUME'den kimlik cumlesini silmek
    gerekirdi, ve o altkume sinav varliklarindan gelmek zorundaydi
    (1.608 varligin 1.600'u sinavda geciyor, olculdu) -- yani confound.

    Ayri yuzeyle uc kalip AYRISIYOR ve ayrisma ADIN HEMEN ARDINDA:

        SINAV      Ayse Kaya'nin danismani kimdir?
        KIMLIK     Ayse Kaya kimdir?
        BIYOGRAFI  Ayse Kaya hakkinda ne biliyoruz?

    Altkume YOK, kimlik koprusu 1.608 varligin HEPSINDE duruyor."""
    return f"{_tr(ad)} hakkında ne biliyoruz?"


# --- denetim -------------------------------------------------------------
def denetle(tohum: int = 0, yaz=print) -> dict:
    """Her iliski, her bicim -- uretilen metin TURKCE mi, tekil mi.

    Kapilar: (1) her (iliski, bicim) bir cumle uretiyor, (2) cumle iki
    kez nokta icermiyor (bir cumle = bir olgu), (3) ek jetonlari
    `veri_09` tablolarindan cikiyor (yeniden hesaplanmiyor).
    """
    G = V.kur(tohum)
    hedef = {t: set(G["ad"][t]) for t in V.TIPLER}
    tipi = {a: t for t in V.TIPLER for a in G["ad"][t]}
    sayim, ornek = {}, {}
    for r in V.ILISKI:
        ilk = None
        for (o, rr), c in G["olgu"].items():
            if rr == r:
                ilk = (o, c)
                break
        assert ilk, f"{r} icin olgu YOK"
        o, c = ilk
        for b in range(N_BICIM):
            s = cumle(o, r, c, b, tipi[c])
            assert s.count(".") == 1 or (b == 4 and s.count(".") == 1), \
                f"{r} bicim {b}: BIR CUMLE = BIR OLGU ihlali -> {s}"
            assert s[0].isupper(), f"{r} bicim {b}: buyuk harf yok -> {s}"
            sayim[(r, b)] = len(s)
        ornek[r] = cumle(o, r, c, 0, tipi[c])
    yaz(f"  {len(V.ILISKI)} iliski x {N_BICIM} bicim = {len(sayim)} cumle GECTI")
    yaz(f"  uzunluk {min(sayim.values())}..{max(sayim.values())} karakter")
    return ornek


if __name__ == "__main__":
    import sys
    print("metin_09 DENETIMI")
    orn = denetle()
    print("\nHER ILISKIDEN BIR CUMLE (kanonik)")
    for r in V.ILISKI:
        print("   " + orn[r])
    G = V.kur(0)
    tipi = {a: t for t in V.TIPLER for a in G["ad"][t]}
    o, c = "Ahmet_Aydin", G["olgu"][("Ahmet_Aydin", "kardesi")]
    print(f"\nAYNI OLGU, {N_BICIM} BICIM")
    for b in range(N_BICIM):
        print(f"   {BICIM[b]:<14}{cumle(o, 'kardesi', c, b, tipi[c])}")
    k = G["olgu"][("Ahmet_Aydin", "bolumu")]
    f = G["olgu"][(k, "fakultesi")]
    print(f"\nIKI ADIMLI ({BICIM[0]}) -- kopru '{_tr(k)}' GECMIYOR")
    for b in range(N_BICIM):
        print(f"   {BICIM[b]:<14}{zincir(o, 'bolumu', 'fakultesi', f, b, tipi[f])}")
    print(f"\nKIMLIK\n   {kimlik(o, tipi[o])}")
