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
    """Cumle BASI konumlari -- noktadan SONRAKI her konum.

    Pencere buradan acilirsa hicbir pencere cumle ortasindan baslamaz.
    Gerekce OLCULDU: bugunku atla=4 ile pencerelerin %71'i cumle
    ORTASINDAN basliyor; onek, ait olmadigi bir cumlenin kuyrugunu
    tasiyor ve dikkat onu GORUYOR (sinir maskesi yok).

    Akisa DOKUNMAZ -- yalniz hangi konumlardan pencere acilacagini
    soyler.  Dosya ve izi aynen kalir (kural 9).
    """
    dz = d["dizi"].numpy()
    b = np.flatnonzero(dz == d["ix"]["."]) + 1
    b = b[b < len(dz)].astype(np.int64)
    yaz(f"cumle basi {len(b):,}   ortalama cumle "
        f"{len(dz) / max(len(b), 1):.1f} birim")
    return b


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
