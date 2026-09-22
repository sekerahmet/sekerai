"""agirlik_16 -- CEVAP ARALIKLARI.

Kullanici, 22 Eylul: *"ilk cevap token u dogru mu, sonra ilk ve ikinci bir
arada dogru mu, sonra ilk ikinci ve ucuncu dogru mu diye her sirada tum
cevaba kadar kontrol etmek"*.

Bu, kaybi AGIRLIKLANDIRARAK da yazilabilirdi (agirliklar L..1) ve
once oyle yazildi; olculdu, kaldirildi.  Bugunku bicim bir KAPI:

    konum k puanlanir  <=>  0..k-1'in HEPSI dogru bilindi

Kullanici, ayni gun: *"yuva 8 dogru cevabi almadiysa yuva 9 soru sorma
hakki yok."*  Gerekce: sinav cevabin TAMAMINA bakiyor; onek bozulduysa
soru sifir aliyor ve devaminin dogru olmasi hicbir sey kazandirmiyor.

Bu modul kapinin GIRDISINI uretir: konum basina aralik ici sira.
Kapinin kendisi `model_16._kapi`.

CEVAP ARALIGI NEREDEN GELIYOR.  Akis duz bir birim dizisi; kayip
cevabin nerede oldugunu bilmiyor.  Boru hattini degistirmek yerine
aralik AKISTAN turetiliyor:

    BILINEN bir varlik adi  +  kopula eki  +  nokta   ->  o ad bir CEVAP

Eslesme 1.608 varlik adiyla BIREBIR, yani isaretin dogrulugu tanim
geregi tam.  Geri yuruyuslu bir kural denendi ve %82,3'te kaldi
(`Melikgazi Psikoloji bolum -u` gibi adlarda kirpiliyordu, ve tip
cumlelerini `bir tez -dir` yanlislikla isaretliyordu).

KAPI: `denetle` isaretlerin gercekten olgu cevabi oldugunu ornekle
sinar ve kapsamı basar.

ZINCIRDEKI YERI:  birim_16 (akis) -> agirlik_16 (ofset dizisi) -> train_16
                  -> model_16._kapi
"""
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KOPULA = ("-dır", "-dir", "-dur", "-dür", "-tır", "-tir", "-tur", "-tür")


def varlik_tokenleri(d):
    """1.608 varligin token dizileri.  metin_16'nin kendi cevirisiyle."""
    import metin_16 as MT
    import olcme_16 as OL
    kodla = OL._kodlayici(d)[0]
    return [tuple(kodla(MT._tr(v))) for v in d["varlik"]]


def isaretle(d, yaz=print):
    """Akista CEVAP araliklarini bul.  Doner: (baslangic, uzunluk) dizileri.

    Kural: BILINEN varlik adi + kopula eki + nokta.  Eslesme birebir."""
    dz = d["dizi"].numpy()
    ix = d["ix"]
    kop = np.array([ix[e] for e in KOPULA if e in ix])
    son_im = [ix[x] for x in (".", ",") if x in ix]
    # Kopuladan sonra NOKTA ya da VIRGUL: duz cumlelerin bir kismi
    # "... -dir , bu boyle bilinir ." diye devam ediyor ve yalniz nokta
    # aransa o gecisler KACIYORDU.  Kesinlik yine tam, cunku ad eslesmesi
    # 1.608 varlikla birebir.
    bitis = np.isin(dz, kop)[:-1] & np.isin(dz, son_im)[1:]
    son = np.flatnonzero(bitis)                            # kopulanin konumu
    # uzunluga gore grupla: her varlik adi icin o uzunlukta dilim karsilastir
    ad_uz = collections.defaultdict(set)
    for u in varlik_tokenleri(d):
        ad_uz[len(u)].add(u)
    bas, uz = [], []
    for L, kume in sorted(ad_uz.items()):
        b = son - L                                        # adin baslangici
        ok = b >= 0
        if not ok.any():
            continue
        bb = b[ok]
        dilim = np.lib.stride_tricks.sliding_window_view(dz, L)[bb]
        eslesen = [i for i, sat in zip(bb, map(tuple, dilim)) if sat in kume]
        bas.extend(eslesen); uz.extend([L] * len(eslesen))
    bas = np.array(bas, np.int64); uz = np.array(uz, np.int64)
    s = np.argsort(bas)
    bas, uz = bas[s], uz[s]
    yaz(f"CEVAP araligi {len(bas):,}   akisin %{100*uz.sum()/len(dz):.2f}'i"
        f"   uzunluklar {dict(sorted(collections.Counter(uz.tolist()).items()))}")
    return bas, uz


def ofset(d, yaz=print):
    """Konum basina ARALIK ICI SIRA: 0 = cevabin ILK tokeni,
    1..L-1 = devami, -1 = cevap araligi disi.

    CEVAP KAPISI bununla kuruluyor: konum k, 0..k-1'in HEPSI dogru
    bilindiyse puanlanir; ilk hatadan sonrasi kapanir.  Gerekce:
    sinav cevabin TAMAMINA bakiyor, onek bozulduysa soru sifir
    aliyor ve devaminin dogru olmasi hicbir sey kazandirmiyor."""
    dz = d["dizi"].numpy()
    o = np.full(len(dz), -1, np.int8)
    bas, uz = isaretle(d, yaz)
    for b, L in zip(bas.tolist(), uz.tolist()):
        o[b:b + L] = np.arange(L, dtype=np.int8)
    yaz(f"ofset: aralik basi {int((o == 0).sum()):,}"
        f"   devam {int((o > 0).sum()):,}   disari {int((o < 0).sum()):,}")
    return o


def cumle_basi(d, yaz=print):
    """Pencere BASI konumlari -- noktadan SONRAKI her konum.

    Parca = iki nokta arasi; CUMLE DEGIL, ya bir bildirim ya bir
    SORU + CEVABI ('?' parca ICI, soruyu cevabindan ayirmaz).

    Gerekce OLCULDU: kayan pencere (atla=4) ile pencerelerin %71'i
    parca ORTASINDAN basliyordu; onek, ait olmadigi bir parcanin
    kuyrugunu tasiyor ve dikkat onu GORUYOR.

    Belge siniri ATLANIR: bir belgenin son noktasindan sonra `<belge>`
    gelir ve pencere oradan acilsa ilk yuvada "sayfa bitti"den baska
    bir sey olmazdi.  Sinir bir sonraki pencerenin ICINDE, yani
    bilgilendirici oldugu yerde kalir.

    Akisa DOKUNMAZ -- yalniz hangi konumdan pencere acilacagini soyler.
    """
    dz = d["dizi"].numpy()
    b = np.flatnonzero(dz == d["ix"]["."]) + 1
    sin = d["ix"].get("<belge>")
    if sin is not None:
        atla = (b < len(dz)) & (dz[np.minimum(b, len(dz) - 1)] == sin)
        b = b + atla                       # sinirin BIR SONRASI
    b = b[b < len(dz)].astype(np.int64)
    yaz(f"pencere basi {len(b):,}   ortalama parca "
        f"{len(dz) / max(len(b), 1):.1f} birim")
    return b


def parca_tablo(d, t_len=None, yaz=print):
    """TEKER TEKER egitim icin PARCA OBEKLERI.  Doner: {L: (X, OFS)}.

    Bir satir = bir PARCA (iki nokta arasi): ya bir bildirim, ya bir
    SORU + CEVABI.  Satirlar UZUNLUGA GORE obeklenir, yani her tensor
    TAM DOLU -- DOLGU YOK.

    Neden dolgu degil obek: dolgu sonucu degistirmiyordu (hedefi dolgu
    olan konum `ignore_index` ile duser, nedensel maske yuzunden gercek
    konumlar dolguya bakamaz) ama HESABI harciyordu.  OLCULDU: tek
    pencere T=32 ile dolgu %56,9, yani 2,32 kat bosa hesap.  Obeklenince
    %0.  Bedeli: bir yigin TEK uzunluktan gelir.

    Akistan FARKI: hicbir satir baska bir parcadan tek birim tasimaz.
    Model "sonraki soru"yu gormez; buna karsilik her ornek sinavin
    verdigi seyin AYNISI olur.

    Durma isareti kaybolmaz: parca "." ile bitiyor, yani son hedef
    nokta.  Model cevabi yazip noktayi koymayi ogrenir.
    """
    dz = d["dizi"].numpy()
    ix = d["ix"]
    t_len = t_len or d["t_len"]
    son = np.flatnonzero(dz == ix["."])
    bas = np.concatenate([[0], son[:-1] + 1])
    # Belge siniri ONCEKI belgenin sonuna ait; teker teker egitimde
    # belge diye bir sey yok, o yuzden parcanin BASINDAN dusuruluyor.
    sin = ix.get("<belge>")
    if sin is not None:
        bas = bas + (dz[np.minimum(bas, len(dz) - 1)] == sin)
    uz = son - bas + 1
    tut = uz <= t_len
    if not tut.all():
        yaz(f"parca_tablo: {int((~tut).sum()):,} parca t_len={t_len}'e "
            f"sigmiyor, DUSTU (en uzun {int(uz.max())})")
    bas, uz = bas[tut], uz[tut]
    ofs_tam = ofset(d, yaz=lambda *a: None)
    obek = {}
    for L in np.unique(uz):
        b = bas[uz == L]
        p = b[:, None] + np.arange(L)[None, :]
        obek[int(L)] = (torch.from_numpy(dz[p].astype(np.int16)),
                        torch.from_numpy(ofs_tam[p]))
    n = sum(X.shape[0] for X, _ in obek.values())
    ab = sum(int((O == 0).any(1).sum()) for _, O in obek.values())
    yaz(f"parca obegi {len(obek)} uzunluk   {n:,} satir   DOLGU YOK"
        f"   cevap araligi basi {ab:,} satirda")
    return obek


def parca_dolu(d, t_len=None, yaz=print):
    """parca_tablo'nun TEK TENSOR hali.  Doner: (X, OFS, MASKE).

    27 obek tek (N, t_len) tensore doldurulur.  Kayip BIREBIR AYNI
    kalir: dolgu konumlari MASKE ile hem paydan hem paydadan duser,
    nedensel maske gercek konumlarin dolguya bakmasini engeller
    (dolgu hep sonda), ve dolgu birimlerine gradyan akmaz.

    Neden: obekleme adim basina 27 kucuk ileri/geri gecis uretiyor ve
    kucuk tensorlerde GPU baslatma maliyeti hesabin onune geciyor.
    Tek gecis 2,32 kat FLOP eder ama 27 kat az cekirdek baslatir.

    Sozluge DOLGU BIRIMI EKLENMEZ -- doldurucu olarak 0. indeks
    kullanilir, o konumlar zaten puanlanmiyor.  Sozluk 445 kalir,
    veri izi degismez.

    Satirlar TEK havuzda oldugu icin duzgun rastgele cekilis her
    satira esit sans verir; obek secmedeki aclik sorunu YOK.
    """
    OB = parca_tablo(d, t_len, yaz=lambda *a: None)
    t_len = t_len or d["t_len"]
    N = sum(X.shape[0] for X, _ in OB.values())
    X = torch.zeros((N, t_len), dtype=torch.int16)
    O = torch.full((N, t_len), -1, dtype=torch.int8)
    M = torch.zeros((N, t_len - 1), dtype=torch.float32)
    i = 0
    for L in sorted(OB):
        x, o = OB[L]
        n = x.shape[0]
        X[i:i + n, :L] = x
        O[i:i + n, :L] = o
        M[i:i + n, :L - 1] = 1.0          # konum j, j+1'i tahmin eder
        i += n
    yaz(f"tek tensor {N:,} x {t_len}   puanlanan konum "
        f"{int(M.sum()):,}   dolgu %{100 * (1 - M.mean()):.1f}")
    return X, O, M


def denetle(d, n=2000, yaz=print):
    """KAPI -- isaretler gercekten OLGU CEVABI mi, ve kapsam ne."""
    import metin_16 as MT
    import olcme_16 as OL
    kodla = OL._kodlayici(d)[0]
    dz = d["dizi"].numpy()
    bas, uz = isaretle(d, yaz=lambda *a: None)
    ad = {tuple(u): i for i, u in enumerate(varlik_tokenleri(d))}
    g = np.random.default_rng(0)
    s = g.choice(len(bas), min(n, len(bas)), replace=False)
    tam = sum(tuple(dz[bas[k]:bas[k] + uz[k]]) in ad for k in s)
    assert tam == len(s), f"isaretin {len(s)-tam}'i varlik adi DEGIL"
    # KAPSAM: ezber_olgu cevaplari akista kac kez isaretlenmis
    E, IL, TIP = d["varlik"], d["iliski"], d["tip_ad"]
    isar = np.zeros(len(dz), bool)
    for b, L in zip(bas.tolist(), uz.tolist()):
        isar[b] = True
    ak = "".join(map(chr, dz))          # aranabilir hal -- BIR KEZ kurulur
    bulundu = kayip_ = 0
    for z in [[int(x) for x in y] for y in d["bolme"]["ezber_olgu"]][:200]:
        tam_c = kodla(MT.soru(E[z[0]], IL[z[1]], E[z[-1]],
                              TIP[int(d["tip"][z[-1]])], 0))
        on = kodla(MT.sinav_yuzeyi(E[z[0]], [IL[z[1]]], E[z[-1]],
                                   TIP[int(d["tip"][z[-1]])])[0])
        cev_tok = tam_c[len(on)]                       # cevabin ILK tokeni
        p = "".join(map(chr, tam_c))
        k, n_bul, n_isar = ak.find(p), 0, 0
        while k >= 0:
            n_bul += 1
            n_isar += bool(isar[k + len(on)])
            k = ak.find(p, k + 1)
        bulundu += n_bul; kayip_ += n_bul - n_isar
    yaz(f"KAPI GECTI: {len(s):,} isaretin hepsi TAM bir varlik adi")
    yaz(f"  200 olgunun sinav bicimindeki {bulundu:,} gecisinden "
        f"{bulundu-kayip_:,}'i isaretlenmis  (%{100*(bulundu-kayip_)/max(bulundu,1):.1f})")
    return bas, uz


if __name__ == "__main__":
    import torch
    d = torch.load(os.environ.get("BIRIM_PT",
                                  "/content/drive/MyDrive/model_16/birim_16.pt"),
                   weights_only=False)
    denetle(d)
    agirlik(d)
