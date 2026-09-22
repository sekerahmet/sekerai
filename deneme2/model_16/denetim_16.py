"""denetim_16 -- veri yolunun kapilari.  Her kapinin NEDEN var oldugu yazili.

`test_14`in 43 kapisindan 36'si model_14'un mimarisine aitti (donme,
Rodrigues, capa, VQ, saat, hafiza butcesi, ReLU seyreklik, defter) ve
model_16'yi ilgilendirmiyor.  Buraya VERI YOLUNA ait olanlar tasindi;
her birinin gerekcesi bir KAYIPTAN geliyor, tahminden degil.

    python denetim_16.py

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
  dok_16      goz ile okunur dokum
  denetim_16  KAPILAR   <-- BU DOSYA

  model_16    MIMARI -- model_15'ten
  train_16    egitim dongusu
  olcme_16    olcu: soru soruldu, cevap dogru mu
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
_D = None


def _veri():
    global _D
    if _D is None:
        import torch
        yol = os.environ.get("KORPUS_PT",
                             r"G:/Drive'ım/model_16/korpus_16.pt")
        assert os.path.exists(yol), f"{yol} YOK -- KORPUS_PT ile yol ver"
        _D = torch.load(yol, weights_only=False)
    return _D


GECTI, KALDI = [], []


def kapi(ad):
    def sar(f):
        try:
            GECTI.append((ad, f() or ""))
        except AssertionError as e:
            KALDI.append((ad, str(e)))
        except Exception as e:                      # kodun kendisi bozuk
            KALDI.append((ad, f"{type(e).__name__}: {e}"))
        return f
    return sar


# =====================================================================
@kapi("0  IZOLASYON -- model dosyasi disariya import etmez")
def _0():
    """Kol KENDI kodunu tasir.  model_16.py baska bir kol dosyasina
    baglanirsa kol tek basina kosturulamaz hale gelir."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "model_16.py"), encoding="utf-8").read()
    izin = {"torch", "torch.nn", "torch.nn.functional", "math", "__future__"}
    n = 0
    for l in src.split("\n"):
        if l.startswith(("import ", "from ")):
            n += 1
            mod = l.split()[1]
            assert mod in izin, f"disariya import: {l}"
    return f"{n} import, hepsi izinli"


@kapi("1  BIRIM -- pencere sekli, SON PENCERE dusmuyor, kapali sinif")
def _1():
    """*Gerekce (birim_16 yorumu):* onceki hal son pencereyi DUSURUYORDU.
    `sliding_window_view` len-L+1 pencere verir ve sonuncusu gecerlidir."""
    import numpy as np
    import birim_16 as B
    dizi = np.array([3, 1, 0, 1, 2, 1, 0, 4, 1, 0], dtype=np.int64)
    b = B.Birim(["a", "b", "c", "d", "e"], dizi, {}, set(), frozenset())
    assert list(b.say) == [3, 4, 1, 1, 1], list(b.say)
    m = b.kapali(2)
    assert set(m.nonzero()[0].tolist()) == {0, 1}, "en sik iki birim secilmedi"
    P = b.pencere(4)
    assert P.shape == (7, 4), P.shape
    assert list(P[0]) == [3, 1, 0, 1], P[0]
    assert list(P[-1]) == [0, 4, 1, 0], "SON PENCERE DUSTU"
    return f"say {list(b.say)}  pencere {P.shape}"


@kapi("2  BOLME -- her kelime BILINEN KOK + BILINEN EK'e cozulur")
def _2():
    """*Gerekce (ek_16):* soyma ancak kalan BILINEN bir koke inerse kabul.
    Aksi halde `Mers`, `Gires`, `Univers`, `sehr` gibi UYDURMA kokler
    girer ve sozluk sessizce bozulur."""
    import ek_16 as EK
    for k, bek in (("kardeşidir", 3), ("danışmanının", 3),
                   ("Bahçelievler", 1)):
        p = EK.bol(k)
        assert len(p) == bek, f"{k} -> {p}  (beklenen {bek} parca)"
    p = EK.bol("kardeşidir")
    assert p[0] == "kardeş" and all(x.startswith("-") for x in p[1:]), p
    return "kok+ek ayrimi dogru, ozel ad bolunmuyor"


@kapi("3  ZINCIR DUZENI -- hicbir zincir SESSIZCE dusmemeli")
def _3():
    """*Gerekce OLCULDU (model_14, 20 Eylul):* olcum `3 <= len <= 4`
    suzuyor ve `adim = len-2` diyordu.  Gercek duzen 1 adim uzunluk 3,
    2 adim uzunluk 5.  Uzunluk 5 hicbirine uymuyordu: 14.043 zincirin
    11.043'u SESSIZCE dustu, CIKARIM 8 ornege indi.  Hicbir sayi
    gostermedi -- 'zincir 3000' satiri makul duruyordu.

    Kapi HAVUZUN TAMAMINI sayar: her zincir ya 3 ya 5 uzunlukta olmali,
    ve `olcme_16.yuzeyler` ikisini de OKUMALI."""
    import olcme_16 as O
    d = _veri()
    top = 0
    for ad, L in d["bolme"].items():
        if not L:
            continue
        uz = {len(z) for z in L}
        assert uz <= {3, 5}, f"{ad}: beklenmeyen uzunluk {sorted(uz)}"
        top += len(L)
        n = len(O.yuzeyler(d, ad, en=50))
        assert n == min(50, len(L)), f"{ad}: {len(L)} zincirden {n} yuzey"
    return f"{top:,} zincir, hepsi uzunluk 3 ya da 5, hepsi okunuyor"


@kapi("4  SINAV YUZEYI -- GERCEK sozlukle kurulur, TAMAMI cozulur")
def _4():
    """*Gerekce OLCULDU (model_14, 21 Eylul):* olcum `bx['-TAMLAYAN']`
    uzerinde KeyError verdi.  Yani tokenizer yeniden yazildigindan beri
    SINAV HIC KOSMAMISTI ve 34 kapinin hicbiri gormemisti -- cunku
    hicbiri sorulari GERCEK sozlukle kurmuyordu.

    Kapi gercegini kurar: uretilen her onek ve cevap, modelin sozlugune
    TAM oturmali.  Oturmazsa sinav sessizce sifir verir ve biz 'model
    ogrenemedi' deriz."""
    import olcme_16 as O
    d = _veri()
    if "harf" not in d:
        return "ATLANDI -- bu dosya BIRIM bicimde (kapi 5 onu sinar)"
    harf = set(d["harf"])
    eksik = set()
    n = 0
    for ad in O.EZBER + O.CIKARIM:
        for onek, cev in O.yuzeyler(d, ad, en=300):
            eksik |= (set(onek) | set(cev)) - harf
            n += 1
    assert not eksik, f"sozlukte YOK: {sorted(eksik)[:20]}"
    return f"{n} yuzey, {len(harf)} karakterlik sozluge TAM oturdu"


@kapi("5  SINAV YUZEYI, BIRIM sozlugunde -- KARAKTER DEGIL")
def _5():
    """Kapi 4'un BIRIM hali.  Karakterde her harf sozlukte oldugu icin
    sorun cikmaz; birimde onek KELIME KELIME cozulmek zorunda ve bir
    kelime sozlukte yoksa sinav SESSIZCE kosmaz.  model_14'un 21 Eylul'de
    yasadigi sey tam buydu.

    Birim dunyasi yoksa kapi ATLANIR -- ama atlandigini SOYLER."""
    import olcme_16 as O
    yol = os.environ.get("BIRIM_NPZ", "birim_16.npz")
    if not os.path.exists(yol):
        return f"ATLANDI -- birim dunyasi yok ({yol})"
    import birim_16 as BR
    b = BR.yukle(yol, yaz=lambda *a: None)
    d = _veri()
    eksik, n = set(), 0
    for ad in O.EZBER + O.CIKARIM:
        for onek, cev in O.yuzeyler(d, ad, en=300):
            for w in (onek + " " + cev).split():
                if w not in b.bolme:
                    eksik.add(w)
            n += 1
    assert not eksik, (f"{len(eksik)} kelime BIRIM sozlugunde YOK -- sinav "
                       f"sessizce kosmaz: {sorted(eksik)[:20]}")
    return f"{n} yuzey, {len(b)} birimlik sozluge TAM oturdu"


@kapi("6  ETIKET -- cikarim_* korpusta YOK, ezber_* VAR")
def _6():
    """*Gerekce:* `cikarim_*` demek 'cevap korpusta YAZMIYOR' demek.
    Yaziyorsa o bolme ezberi olcer ve CIKARIM diye raporlanir -- butun
    sayilar okunamaz hale gelir.  Sinif zincirin ETIKETINDEN degil
    KORPUSTAN dogrulanir."""
    import olcme_16 as O
    d = _veri()
    s = O.etiket_kapisi(d, ornek=40, yaz=lambda *a: None)
    for ad, (n, top, bek) in s.items():
        assert (n > 0) == bek, f"{ad}: {n}/{top} korpusta, beklenen {bek}"
    return "  ".join(f"{a} {n}/{t}" for a, (n, t, _) in s.items())


@kapi("7  IZ -- veri dosyasi ile kod AYNI seyi soyluyor")
def _7():
    """*Gerekce (kural 9):* her kosuda yeniden uretilen korpus bolucudeki
    en ufak degisiklikte SESSIZCE kayar ve iki kosu kiyaslanamaz hale
    gelir.  Dosyanin izi yeniden HESAPLANIP karsilastirilir."""
    import hazirla_16 as H
    import veri_16 as V
    d = _veri()
    assert H.iz(d) == d["iz"], f"iz TUTMADI {H.iz(d)} != {d['iz']}"
    assert d["graf_izi"] == V.IZ, f"graf izi {d['graf_izi']} != {V.IZ}"
    return f"veri {d['iz']}   graf {d['graf_izi']}"


if __name__ == "__main__":
    import denetim_16  # noqa: F401  (kapilar import aninda kosar)
    print("=" * 74)
    for ad, not_ in GECTI:
        print(f"  GECTI  {ad}")
        if not_:
            print(f"           {not_}")
    for ad, e in KALDI:
        print(f"  KALDI  {ad}")
        print(f"           {e}")
    print("=" * 74)
    print(f"  {len(GECTI)} GECTI   {len(KALDI)} KALDI")
    sys.exit(1 if KALDI else 0)
