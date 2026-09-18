# -*- coding: utf-8 -*-
"""egitim_dok_08 — VERININ KENDISI, duz metin.

Kullanici karari, 17 Eylul 2026:
    *"veri dosyalarini uret ama uretme modelini duzelt. BIR FORMATTA
    olmali. 01. token listesi, 02. varlik listesi, 03. egitim hop1
    tip1, egitim hop1 tip2, sorular hop1 tip1 gibi bir suru detayda
    text dosyasi olarak. ama lutfen dizi icinde detay olmadan."*

FORMAT -- iki kural, istisnasiz:

  1) HER SATIR TIPI KENDI DOSYASINDA, numarali. Birlestirme yok:
     "hop1'in uc bicimi tek dosyada" YAPILMAZ, ucu de ayri dosya.
  2) DOSYANIN ICINDE DETAY YOK. Bir satir = bir cumle. Baslik yok,
     aciklama yok, jeton numarasi yok. Ne aradigin `00_ICINDEKILER`de.

    ONCE (bozuktu)                  SIMDI
    04_egitim_bildirim_1hop.txt     10_egitim_hop1_tip1.txt
      (3 bicim IC ICE)              11_egitim_hop1_tip2.txt
                                    12_egitim_hop1_tip3.txt
    14_diziler_numarali.txt         -- SILINDI: satir basina 4 satir
      cumle + "jeton:" + "numara:"     detay yaziyordu.

Numaralarda BOSLUK YOK ve dosya listesi TEK BIR TABLODAN (`PLAN`)
turetiliyor -- eskiden numaralar elle yazilmisti ve 04/05/12/13/14
diye gidiyordu, arasi bostu.

Cumleler MODELIN GORDUGU DIZIDEN okunuyor (`analiz_08.Dok.oku`),
yeniden uretilmiyor -- burada ne goruyorsan havuzda o var.

!! YERLESIM. `analiz_08`in dokumu DENETIM icin (graf oku, kisayol,
taban cizgi) ve `veri/model_08/denetim/` altinda durur. Burasi
yalniz CUMLELER.

    python egitim_dok_08.py [--klasor <yol>] [--ornek N]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import numpy as np                                             # noqa: E402
import taban_08 as M                                           # noqa: E402
import analiz_08 as AZ                                         # noqa: E402

NL = chr(10)


# ===================== DOSYA PLANI — TEK KAYNAK ==========================
# (numara, ad, ne oldugu)  -- "ne oldugu" YALNIZ 00_ICINDEKILER'e girer,
# veri dosyasinin icine GIRMEZ.
PLAN = [
    ("01", "jetonlar",          "sozlugun tamami: numara + jeton"),
    ("02", "varliklar",         "varlik adlari"),
    ("03", "iliskiler",         "iliski adlari"),

    ("10", "egitim_hop1_tip1",  "1-hop bildirim, kanonik sira"),
    ("11", "egitim_hop1_tip2",  "1-hop bildirim, yuklem basta"),
    ("12", "egitim_hop1_tip3",  "1-hop bildirim, iliski basta"),
    ("13", "egitim_hop2_tip1",  "2-hop bildirim, kanonik sira"),
    ("14", "egitim_hop2_tip2",  "2-hop bildirim, yuklem basta"),
    ("15", "egitim_hop2_tip3",  "2-hop bildirim, iliski basta"),
    ("16", "egitim_bosluk_hop1", "1-hop bosluk doldurma"),
    ("17", "egitim_bosluk_hop2", "2-hop bosluk doldurma"),
    ("18", "sorular_hop1",      "1-hop soru: '... kardesi kimdir? ...'"),
    ("19", "sorular_hop2",      "2-hop soru: '... arkadasi kimdir? ...'"),
    ("20", "egitim_kimlik",     "kimlik koprusu: 'X kimdir? X'dir.'"),

    ("30", "sinav_one",         "SINAV bolmesi: tek adimlik olgu"),
    ("31", "sinav_seen",        "SINAV bolmesi: gorulmus 2-hop"),
    ("32", "sinav_comp",        "SINAV bolmesi: gorulmemis uclu"),
    ("33", "sinav_ent",         "SINAV bolmesi: hic zincir basi olmamis"),
    ("34", "sinav_ent_yok",     "SINAV bolmesi: kisayol TIP OLARAK imkansiz"),
    ("35", "sinav_ood",         "SINAV bolmesi: dagitim disi"),

    # !! AYNI ZINCIRLER, SORU BICIMINDE. Kullanici, 17 Eylul:
    # *"tamam da bu soru degil ki, bu cumle. soru 'kimdir' diye biter.
    # burda ne soruluyor, cevap ne?"* Hakli: 30..35 yarim birakilmis
    # BILDIRIM. Soru bicimli hali OLCULUYOR (pencere_08 `soru_` tablosu)
    # ama DOSYAYA DOKULMUYORDU -- eksikti.
    #
    # HUKMU 30..35 VERIR: olcme izi f4ce53fd1555 ile model_05/06 kiyasi
    # YALNIZ orada gecerli. 40..45 IKINCIL, eklentinin kendi isi.
    ("40", "sinavsoru_one",      "AYNI zincir, SORU bicimi (ikincil)"),
    ("41", "sinavsoru_seen",     "AYNI zincir, SORU bicimi (ikincil)"),
    ("42", "sinavsoru_comp",     "AYNI zincir, SORU bicimi (ikincil)"),
    ("43", "sinavsoru_ent",      "AYNI zincir, SORU bicimi (ikincil)"),
    ("44", "sinavsoru_ent_yok",  "AYNI zincir, SORU bicimi (ikincil)"),
    ("45", "sinavsoru_ood",      "AYNI zincir, SORU bicimi (ikincil)"),

    # !! AYNI ZINCIRLER, BOSLUK DOLDURMA. Kullanici, 17 Eylul: *"ama bu
    # da bosluk doldurma tipinde ent, diger tipte ent nerede?"*
    # !! IKISI DE TEK JETON bosaltir (kullanici karari, 17 Eylul).
    # Once `ozne` OZNENIN TAMAMINI bosaltiyordu ve egitimle UYUMSUZDU:
    # egitimdeki 187.296 boslugun 187.296'si tek jetonluk, ama o sinav
    # 2-3 jeton istiyordu.
    # pencere_08 bunlari `fim_ozne1_` / `fim_rel_` diye OLCUYORDU ama
    # dosyaya dokmuyordu. Diziyi `taban_08.kodla_fim_sinav` kuruyor --
    # olcumle AYNI KAYNAK, iki kopya olmasin diye.
    ("50", "sinavbosluk_ozne_comp",    "AYNI zincir, SOYAD (oznenin son jetonu) bosluklu"),
    ("51", "sinavbosluk_ozne_ent",     "AYNI zincir, SOYAD bosluklu (ikincil)"),
    ("52", "sinavbosluk_ozne_ent_yok", "AYNI zincir, SOYAD (oznenin son jetonu) bosluklu"),
    ("53", "sinavbosluk_ozne_ood",     "AYNI zincir, SOYAD (oznenin son jetonu) bosluklu"),
    ("54", "sinavbosluk_rel_comp",     "AYNI zincir, ILISKI jetonu bosluklu"),
    ("55", "sinavbosluk_rel_ent",      "AYNI zincir, ILISKI jetonu bosluklu"),
    ("56", "sinavbosluk_rel_ent_yok",  "AYNI zincir, ILISKI jetonu bosluklu"),
    ("57", "sinavbosluk_rel_ood",      "AYNI zincir, ILISKI jetonu bosluklu"),

    # !! BILDIRME EKI SINAVI (unlu uyumu). Kullanici, 17 Eylul:
    # "verileri ona gore uretmedin ki." Hakli -- `pencere_08`ye `ek_`
    # sutunu eklendi ama KARSILIK GELEN DOSYA URETILMEDI. Ucuncu kez
    # ayni hata (once `sorular`, sonra `sinavbosluk`).
    # Bicim: <onek><TAB><beklenen ek>   -- 01_jetonlar ile ayni duzen.
    ("60", "sinavek_one",   "ad yazildiktan sonra DOGRU EK mi (unlu uyumu)"),
    ("61", "sinavek_seen",  "ad yazildiktan sonra DOGRU EK mi (unlu uyumu)"),
    ("62", "sinavek_comp",  "ad yazildiktan sonra DOGRU EK mi (unlu uyumu)"),
    ("63", "sinavek_ent",   "ad yazildiktan sonra DOGRU EK mi (unlu uyumu)"),
]


def _ayrim(d, v, X, P=None, T=None, i=0, fim=False):
    """Bir ornek satir icin (MODELE VERILEN, MODELDEN ISTENEN).

    Elle yazilmiyor: diziyi ve `kodla_*`in dondurdugu P/T'yi okuyor.
    Kullanici, 17 Eylul: *"iki kolonda modele ne veriyoruz ne
    istiyoruz diye ekler misin"*.

    !! `tam_kayip=True` -- KAYIP HER KONUMDA hesaplaniyor. Burasi
    kaybin sekli DEGIL, satirin NE OGRETTIGI: olculen cevap yuvasi.
    Devrik bicimlerde o yuva YOK (cevap kendi baglamindan once gelir),
    ve bu bir kusur degil, TANIM -- orada satir yalniz dil bilgisi
    ogretiyor."""
    import numpy as _np
    r = [int(t) for t in X[i] if int(t) != M.PAD]
    if fim:
        j = r.index(v.ayir)
        return d.oku(r[:j + 1]), d.oku(r[j + 1:]), d.oku(r[j + 1:])
    if T is not None and int((_np.asarray(T[i]) >= 0).sum()) > 0:
        # !! ISTENEN, KUYRUGUN TAMAMI DEGIL -- T'nin MASKESIZ hedefleri.
        # Kullanici sordu (17 Eylul): "13'te son 3 jetonu mu istiyoruz?"
        # Hayir: olculen sey AD JETONLARI + SINIR ISARETI. Sondaki
        # bildirme eki (`dir`) ve nokta MASKELI -- adin son unlusu belli
        # olunca `dir` zaten belli, sifir kosullu bilgi tasiyor.
        # Once buraya `r[p0+1:]` yaziliydi ve `Kocaeli'dir.` diye
        # gosteriyordu; olculen `Kocaeli'`.
        p0 = int(P[i][0])
        hedef = [int(t) for t in _np.asarray(T[i]) if int(t) >= 0]
        # ISTENEN = cumlenin DEVAMININ TAMAMI (kullanici, 17 Eylul:
        # "beklediğimiz yeri (Kocaeli'dir) diye yaz, yoksa neyi
        # istedigimizi anlamiyorum"). OLCULEN ayri satirda: ad
        # jetonlari + sinir isareti. Aradaki fark `dir` ve nokta --
        # EGITILIYOR ama birincil olcude maskeli. `pencere_08`nin
        # `ek_` sutunu onlari AYRICA olcer.
        return (d.oku(r[:p0 + 1]), d.oku(r[p0 + 1:]), d.oku(hedef))
    return ("(cumlenin tamami)",
            "HER KONUMDA sonraki jeton",
            "YOK -- cevap yuvasi maskeli "
            "(cevap kendi baglamindan ONCE geliyor)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", default=None)
    ap.add_argument("--ornek", type=int, default=0,
                    help="her dosyadan en fazla N satir (0 = HEPSI)")
    a = ap.parse_args()
    kl = a.klasor or os.path.join(_K, "veri", "model_08")
    os.makedirs(kl, exist_ok=True)
    d = AZ.Dok("model_08")
    v, ayar = d.v, d.ayar
    n_bic = max(1, ayar.bicim)
    icerik, AYRIM = {}, {}

    def satirlar(ad):
        """`ad` icin CUMLE LISTESI. Tek yerden, tek bicimde."""
        # --- sozlukler ---
        # !! CUMLELERDEKI YAZIMLA AYNI. `d.ad()` ve `d.R` graf'in ASCII
        # adlarini veriyor (Ibrahim, bolumu); cumleler ise Turkce
        # basiliyor (Ibrahim -> İbrahim, bolumu -> bölümü). Liste
        # ASCII kalirsa dosyadaki adi cumlelerde ARAYAMAZSIN.
        if ad == "jetonlar":
            def _tr(i):
                # `oku()` CUMLE BASI buyuk harf yapar; tek jetonluk bir
                # listede bu YANLIS okunur ("bolumu" -> "Bölümü", oysa
                # jeton cins isim ve kucuk harfli). Cevirilere DOGRUDAN
                # bakiliyor.
                w = d.jeton_ad(i)
                return d.VM.TR_ILISKI.get(w) or d.VM.TR.get(w, w)
            return [f"{i}\t{d.jeton_ad(i)}\t{_tr(i)}" for i in range(v.vocab)]
        if ad == "varliklar":
            return [" ".join(d.VM.TR.get(w, w) for w in M.kelimeler(v, e))
                    for e in range(v.n_ent)]
        if ad == "iliskiler":
            return [d.VM.TR_ILISKI[r] for r in d.R]

        # --- egitim: bildirim, uc sira ---
        if ad.startswith("egitim_hop"):
            hop, tip = int(ad[10]), int(ad[-1]) - 1
            lst = v.one if hop == 1 else v.tr2
            kodla = M.kodla_1hop if hop == 1 else M.kodla_2hop
            if tip >= n_bic:
                return []
            X, P, T = kodla(v, list(lst), tip)
            AYRIM[ad] = _ayrim(d, v, X, P, T)
            return [d.oku(r) for r in X]

        # --- egitim: bosluk doldurma ---
        if ad.startswith("egitim_bosluk_hop"):
            if getattr(ayar, "fim_kat", 0) <= 0:
                return []
            hop = int(ad[-1])
            lst = v.one if hop == 1 else v.tr2
            kodla = M.kodla_1hop if hop == 1 else M.kodla_2hop
            # !! HAVUZDAKININ AYNI TOHUMU. `egitim_havuzu` tek bir
            # uretec kullaniyor ve 1hop'u 2hop'tan ONCE tuketiyor; ayni
            # sirayla ilerlenmezse buradaki bosluk KONUMLARI havuzdan
            # kayar ve dosya "egitimde olan" olmaktan cikar.
            rs = np.random.default_rng(1000 + ayar.veri_tohum)
            cik = []
            for _h, _l, _k in ((1, v.one, M.kodla_1hop),
                               (2, v.tr2, M.kodla_2hop)):
                for b in range(n_bic):
                    X = _k(v, list(_l), b)[0]
                    for r in X:
                        dz = [int(t) for t in r if int(t) != M.PAD]
                        for p in rs.choice(len(dz),
                                           size=min(ayar.fim_kat, len(dz)),
                                           replace=False):
                            if _h == hop:
                                _f = M.kodla_fim(v, dz, int(p))
                                if ad not in AYRIM:
                                    _XX = np.zeros((1, v.t_len), np.int64)
                                    _XX[0, :len(_f)] = _f
                                    AYRIM[ad] = _ayrim(d, v, _XX, fim=True)
                                cik.append(d.oku(_f))
            return cik

        # --- egitim: soru bicimi ---
        if ad.startswith("sorular_hop"):
            if getattr(ayar, "soru_kat", 0) <= 0:
                return []
            hop = int(ad[-1])
            lst = v.one if hop == 1 else v.tr2
            X, P, T = M.kodla_soru(v, list(lst), hop)
            AYRIM[ad] = _ayrim(d, v, X, P, T)
            return [d.oku(r) for r in X]

        # --- egitim: kimlik ---
        if ad == "egitim_kimlik":
            if ayar.ident_frac <= 0:
                return []
            X, P, T = M.kodla_kimlik_q1(v, range(v.n_ent))
            AYRIM[ad] = _ayrim(d, v, X, P, T)
            return [d.oku(r) for r in X]

        # --- sinav bolmeleri, BILDIRIM (HUKMU BUNLAR VERIR) ---
        if ad.startswith("sinav_"):
            bol = ad[6:]
            lst = d.L.get(bol) or []
            if not lst:
                return []
            kodla = M.kodla_1hop if bol == "one" else M.kodla_2hop
            X, P, T = kodla(v, list(lst), 0)
            AYRIM[ad] = _ayrim(d, v, X, P, T)
            return [d.oku(r) for r in X]

        # --- sinav bolmeleri, BILDIRME EKI (IKINCIL) ---
        if ad.startswith("sinavek_"):
            bol = ad[8:]
            lst = d.L.get(bol) or []
            if not lst:
                return []
            kodla = M.kodla_1hop if bol == "one" else M.kodla_2hop
            X, P, T = kodla(v, list(lst), 0)
            # `ek_dogruluk` ile AYNI konum hesabi: sinir isaretinin
            # konumundan SONRAKI jeton. Iki yerde iki hesap olmasin
            # diye burada da T'nin son maskesiz hedefinden turetiliyor.
            son_j = (T >= 0).sum(1) - 1
            poz = P[np.arange(len(P)), np.maximum(son_j, 0)] + 1
            cik = []
            for i in range(len(X)):
                r = [int(t) for t in X[i] if int(t) != M.PAD]
                p = int(poz[i])
                if p + 1 >= len(r):
                    continue
                cik.append(d.oku(r[:p + 1]) + chr(9) + d.jeton_ad(r[p + 1]))
            if ad not in AYRIM and cik:
                _o, _e = cik[0].split(chr(9))
                AYRIM[ad] = (_o, _e, _e)
            return cik

        # --- sinav bolmeleri, BOSLUK DOLDURMA (IKINCIL) ---
        if ad.startswith("sinavbosluk_"):
            if getattr(ayar, "fim_kat", 0) <= 0:
                return []
            _y, bol = ad[12:].split("_", 1)
            lst = d.L.get(bol) or []
            if not lst:
                return []
            X = M.kodla_fim_sinav(
                v, list(lst), "ozne" if _y == "ozne" else "iliski")[0]
            AYRIM[ad] = _ayrim(d, v, X, fim=True)
            return [d.oku(r) for r in X]

        # --- sinav bolmeleri, SORU bicimi (IKINCIL) ---
        if ad.startswith("sinavsoru_"):
            if getattr(ayar, "soru_kat", 0) <= 0:
                return []
            bol = ad[10:]
            lst = d.L.get(bol) or []
            if not lst:
                return []
            X, P, T = M.kodla_soru(v, list(lst), 1 if bol == "one" else 2)
            AYRIM[ad] = _ayrim(d, v, X, P, T)
            return [d.oku(r) for r in X]

        raise AssertionError("PLANDA var, uretici YOK: " + ad)

    for no, ad, ne in PLAN:
        s = satirlar(ad)
        if not s:
            print(f"atlandi: {no}_{ad}  (bu kolda YOK)")
            continue
        tam = len(s)
        if a.ornek:
            s = s[:a.ornek]
        dosya = f"{no}_{ad}.txt"
        yol = os.path.join(kl, dosya)
        with io.open(yol, "w", encoding="utf-8") as f:
            for x in s:
                f.write(x + NL)
        icerik[dosya] = (tam, ne)
        print(f"yazildi: {dosya:<26} {tam:>9,} satir"
              f"   {os.path.getsize(yol)/1024/1024:>5.1f} MB")

    # --- 00_ICINDEKILER: ACIKLAMANIN TEK YERI --------------------------
    with io.open(os.path.join(kl, "00_ICINDEKILER.txt"), "w",
                 encoding="utf-8") as f:
        f.write("model_08 VERISI" + NL)
        f.write("=" * 70 + NL)
        f.write(f"t_len {v.t_len}   sozluk {v.vocab}   varlik {v.n_ent}   "
                f"iliski {v.n_rel}" + NL)
        f.write(f"<BOS> {v.bosluk}   <AYIR> {v.ayir}" + NL + NL)
        f.write("Her dosya: BIR SATIR = BIR CUMLE. Dosyalarin icinde "
                "aciklama YOK." + NL + NL)
        # TABLO: dosya adi | satir | ORNEK | amac. Kullanici, 17 Eylul:
        # *"burdaki dosyalarda dosya adi, ornek ve amac diye bir tablo
        # hazirla."* Ornek DOSYANIN KENDISINDEN okunuyor, elle
        # yazilmiyor -- dosya degisirse tablo da degisir.
        # TABLO. Kullanici, 17 Eylul: *"iki kolonda modele ne veriyoruz
        # ne istiyoruz diye ekler misin bu tablolara, ilk 3'e gerek
        # yok."*  01/02/03 SOZLUK dosyalari; onlarda "verilen/istenen"
        # diye bir sey yok, o yuzden TABLOYA GIRMIYORLAR (dosyalar
        # DURUYOR, yalniz tabloda yer almiyorlar).
        f.write(NL + "TABLO" + NL + "-" * 70 + NL)
        for dosya in sorted(icerik):
            if dosya[:2] in ("01", "02", "03"):
                continue
            n, ne = icerik[dosya]
            _ad = dosya[3:-4]
            with io.open(os.path.join(kl, dosya), encoding="utf-8") as g:
                ilk = g.readline().rstrip(NL)
            ver, ist, olc = AYRIM.get(_ad, ("?", "?", "?"))
            f.write(f"{dosya}   ({n:,} satir)" + NL)
            f.write(f"   amac    : {ne}" + NL)
            f.write(f"   ornek   : {ilk}" + NL)
            f.write(f"   VERILEN : {ver}" + NL)
            f.write(f"   ISTENEN : {ist}" + NL)
            if olc != ist:
                f.write(f"   OLCULEN : {olc}" + NL)
            f.write(NL)
        f.write("01/02/03 sozluk dosyalaridir; 'verilen/istenen' yok." + NL)
        f.write(NL + "VERILEN / ISTENEN / OLCULEN -- uc ayri sey:" + NL)
        f.write("  VERILEN  modelin GORDUGU onek" + NL)
        f.write("  ISTENEN  cumlenin DEVAMI -- tam_kayip=True oldugu icin"
                " hepsi EGITILIYOR" + NL)
        f.write("  OLCULEN  `dogruluk()`un BAKTIGI yer: ad jetonlari +"
                " sinir isareti" + NL)
        f.write("           (ISTENEN ile ayniysa satir YAZILMAZ)" + NL + NL)
        f.write("Fark `dir` ve nokta. `dir` adin SON JETONUNDAN"
                " deterministik -- olculdu:" + NL)
        f.write("73 farkli son jeton, CAKISMA 0. O yuzden birincil olcu"
                " onu maskeliyor." + NL)
        f.write("Ama determinizm MATEMATIKTE; MODELDE olup olmadigini"
                " `pencere_08`nin" + NL)
        f.write("`ek_` sutunu AYRICA olcer (unlu uyumu) -- ve o olcunun"
                " verisi 60..63'te." + NL + NL)
        f.write("60..63 BILDIRME EKI SINAVI -- iki sutun, TAB ile ayrik:"
                + NL)
        f.write("     onek                                          | ek"
                + NL)
        f.write("     Kaan Arslan'in annesinin tezi Niceliksel Optik'"
                " | tir" + NL)
        f.write("  VERILEN kesme isaretine KADAR, ISTENEN tek jeton:"
                " dogru allomorf." + NL)
        f.write("  `ek_dogruluk` ile AYNI hedef -- mekanik olarak"
                " dogrulandi." + NL)
        f.write(NL + "EGITIM HAVUZU = 10..20 arasi dosyalar." + NL)
        f.write("SINAV = 30..35. Sinav DUZ BILDIRIMLE yapilir (tip1); "
                "soru bicimi ve" + NL)
        f.write("bosluk doldurma EGITIMDE var, SINAVDA yok." + NL)
        f.write(NL + "20_egitim_kimlik BENZERSIZ satirlari verir; havuzda "
                f"payi %{ayar.ident_frac:.0%} olana" + NL)
        f.write("kadar TEKRARLANIR. Digerlerinde tekrar yoktur." + NL)
        # !! Kullanici 33_sinav_ent.txt'e bakip sordu (17 Eylul): "bu
        # nasil bir soru? ent sorulari bunlar mi?" Dosyada TAM CUMLE
        # duruyor, cunku sinav dizisi odur; modele yalniz ONEK veriliyor.
        # Bu ayrim dosyanin icine yazilamaz (orada aciklama yok), buraya
        # yaziliyor.
        f.write(NL + "SINAV DOSYALARI -- nasil okunur:" + NL)
        f.write("  Satirda TAM CUMLE var; modele yalniz BASI verilir, "
                "gerisini O yazar." + NL + NL)
        f.write("  30..35  BILDIRIM -- HUKMU BUNLAR VERIR" + NL)
        f.write("     satir  : Kaan Arslan'in annesinin tezi "
                "Niceliksel Optik'tir." + NL)
        f.write("     SORU   : Kaan Arslan'in annesinin tezi ______" + NL)
        f.write("     CEVAP  : Niceliksel Optik'tir." + NL + NL)
        f.write("  40..45  AYNI ZINCIR, SORU BICIMI -- IKINCIL" + NL)
        f.write("     satir  : Kaan Arslan'in annesinin tezi hangisidir? "
                "Niceliksel Optik'tir." + NL)
        f.write("     SORU   : Kaan Arslan'in annesinin tezi hangisidir?" + NL)
        f.write("     CEVAP  : Niceliksel Optik'tir." + NL + NL)
        f.write("  Neden ikisi birden: hukum 30..35'te, cunku model_05 ve "
                "model_06 da" + NL)
        f.write("  o yuzeyde olculdu ve ucu ancak oyle AYNI TABLODA "
                "okunur. 40..45" + NL)
        f.write("  eklentinin kendi isini gosterir." + NL + NL)
        f.write("  HER IKISINDE DE KOPRU CUMLEDE GECMIYOR:" + NL)
        f.write("     Kaan Arslan --annesi--> Tugce Arslan   (bu ad hicbir "
                "satirda YOK)" + NL)
        f.write("     Tugce Arslan --tezi--> Niceliksel Optik" + NL)
        for ln in _dogrula(icerik, ayar, v, a.ornek):
            f.write(ln + NL)
    _olcu_kapisi()
    print(f"{NL}klasor: {kl}")


# ÖLÇÜ AİLESİ -> DOSYA ÖNEKİ. `pencere_08`nin urettigi her ikincil
# olcu ailesinin BURADA bir karsiligi olmali.
#
# !! NEDEN VAR: ayni hata UC KEZ yapildi (kullanici, 17 Eylul).
#   1) `soru_` olculuyordu, dosyasi YOKTU   -> "bu soru degil ki"
#   2) `fim_` olculuyordu, dosyasi YOKTU    -> "bosluk doldurmada ent nerede"
#   3) `ek_` olculuyordu, dosyasi YOKTU     -> "verileri ona gore uretmedin ki"
# Ucunde de olcu vardi, BAKILACAK SEY yoktu. Artik kod SORUYOR.
OLCU_DOSYA = {
    "soru_":      "sinavsoru_",
    "fim_ozne1_": "sinavbosluk_ozne_",
    "fim_rel_":   "sinavbosluk_rel_",
    "ek_":        "sinavek_",
}


def _olcu_kapisi(yaz=print):
    """`pencere_08` hangi ikincil olculeri uretiyor -- hepsinin dosyasi var mi?

    Kaynagi OKUYARAK buluyor: `r[f"<aile>_{...}"]` kaliplari. Modeli
    yuklemeye gerek yok, ve yeni bir olcu eklenince BURASI patlar."""
    import re
    kaynak = io.open(os.path.join(_K, "pencere_08.py"), encoding="utf-8").read()
    # !! [A-Za-z] -- yalniz kucuk harf arayan bir desen, sinama
    # sirasinda UYDURMA_ adli sahte olcuyu KACIRDI (17 Eylul).
    aile = set(re.findall(r'r\[f"([A-Za-z0-9_]+_)\{', kaynak))
    aile -= {"fim_ozne_"}          # ESKI ad, artik uretilmiyor
    planda = {ad for _n, ad, _ne in PLAN}
    eksik = []
    for a in sorted(aile):
        onek = OLCU_DOSYA.get(a)
        if onek is None:
            eksik.append(f"{a}  -> OLCU_DOSYA tablosunda YOK")
        elif not any(p.startswith(onek) for p in planda):
            eksik.append(f"{a}  -> '{onek}*' dosyasi PLAN'da YOK")
    if eksik:
        yaz("!! OLCU KAPISI DUSTU -- olculen ama DOKULMEYEN sey var:")
        for e in eksik:
            yaz("     " + e)
        raise AssertionError("olculen her sey DOKULMELI: " + "; ".join(eksik))
    yaz(f"  olcu kapisi GECTI: {len(aile)} ikincil olcu ailesinin "
        f"{len(aile)}'sinin dosyasi var")


def _dogrula(icerik, ayar, v, ornek):
    """DOKUM ILE HAVUZ TUTUYOR MU -- her uretimde, sessizce degil.

    Sebep: bu dosyalar "egitimde ne var" diye ELLE okunuyor. Dokum
    havuzdan kayarsa yanlis bir sey okunur ve bunun hicbir belirtisi
    olmaz. Burasi sayilari `egitim_havuzu`nun KENDISINE soruyor."""
    out = ["", "DOGRULAMA (dokum <-> egitim havuzu)"]
    if ornek:
        out.append(f"  ATLANDI: --ornek {ornek} ile kirpilmis dokum.")
        print(out[-1])
        return out
    n_bic = max(1, ayar.bicim)
    n_duz = (len(v.one) + len(v.tr2)) * n_bic
    bek = {
        "duz":    (n_duz, sum(icerik[f][0] for f in icerik
                              if f[:2] in ("10", "11", "12", "13", "14", "15"))),
        "bosluk": (n_duz * getattr(ayar, "fim_kat", 0),
                   sum(icerik[f][0] for f in icerik if f[:2] in ("16", "17"))),
        "soru":   ((len(v.one) + len(v.tr2)) * getattr(ayar, "soru_kat", 0),
                   sum(icerik[f][0] for f in icerik if f[:2] in ("18", "19"))),
    }
    tamam = True
    for ad, (b, g) in bek.items():
        iyi = (b == g)
        tamam &= iyi
        out.append(f"  {'GECTI ' if iyi else '!! BOZUK'} {ad:<8} "
                   f"havuz {b:>9,}   dokum {g:>9,}")
    out.append("  " + ("HEPSI TUTUYOR" if tamam else "!! DOKUM HAVUZDAN KAYDI"))
    for ln in out[1:]:
        print(ln)
    assert tamam, "dokum havuzdan kaydi -- yukariya bak"
    return out


if __name__ == "__main__":
    main()
