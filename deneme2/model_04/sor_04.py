# -*- coding: utf-8 -*-
"""sor_03 — model_03'in KENDI ELLE SORU ARACI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_03 folderi
altinda olmali. model_03 diger hicbir model ile ayni seyi
kullanmamali."*

`model_b/sor.py`nin KOPYASI (uretici: scratchpad/kur_okuma00.py). Modeli
`ModelSade` ile kurar. Paylasilan surumde yapilan bir degisiklik buraya
GECMEZ; `test_03.py` ikisinin AYNI SEYI olctugunu her kosuda siniyor.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

_B = os.path.dirname(os.path.abspath(__file__))
if _B not in sys.path:
    sys.path.insert(0, _B)

import taban_03 as M                                         # noqa: E402
import pencere_03 as P                                       # noqa: E402
from model_03 import ModelSade                               # noqa: E402

# model_03'in TEK modeli var; kanca yok, dogrudan yazili.
MODEL_SINIFI = ModelSade


def _ad(v, e):
    """Varlik id -> okunabilir ad.  YUVA SAYISINA BAGIMSIZ.

    Onceden IKI yuvaya sabitliydi; jeton_ad="tam" 3 yuva kullaniyor ve
    ucuncu jeton SESSIZCE DUSUYORDU (Ankara_Fen_Lisesi -> Ankara_Fen)."""
    if v.par is None:
        return f"e{e}"
    return _birlestir(v.par_ad[j][int(v.par[e, j])] for j in range(v.yuva))


_TR = str.maketrans("çğıöşüÇĞIİÖŞÜ", "cgiosucgiiosu")
_YUM = {"k": "g", "p": "b", "t": "d"}        # son ses yumusamasi
_TAMLAYAN = ("nin", "nun", "nın", "nün", "in", "un", "ın", "ün")


def _sade(x):
    """Turkce harfleri ASCII'ye indirir, kucuk harfe cevirir. Kullanici
    'kardeşi' de yazabilsin 'kardesi' de."""
    return x.translate(_TR).lower()


def _yazimlar(r):
    """Bir iliskinin KABUL EDILEN yazimlari.

        anne   -> anne, annesi, annesinin, annenin ...
        okul   -> okul, okulu, okulunun ...
        cocuk  -> cocuk, cocugu, cocugunun ...   (k -> g yumusamasi)
        sehir  -> sehir, sehri, sehrinin ...     (unlu dusmesi)

    Fazla yazim URETMEK zararsiz; TEK kural iki iliskinin ayni yazimi
    PAYLASMAMASI ve bu `_iliski_sozluk` icinde assert ile siniriyor."""
    g = _sade(r)
    govde = {g}
    if g[-1] in _YUM:
        govde.add(g[:-1] + _YUM[g[-1]])          # cocuk -> cocug
    if len(g) >= 3 and g[-2] in "aeiou":
        govde.add(g[:-2] + g[-1])                # sehir -> sehr
    out = {g}
    for b in govde:
        out |= {b + "i", b + "u", b + "si", b + "su"}
    # zincirde ikinci hop'un TAMLAYANI olur: "annesi" -> "annesinin"
    return out | {a + e for a in set(out) for e in _TAMLAYAN}


def _iliski_sozluk(v):
    """yazim -> iliski id. Cakisma varsa DUSER: sessizce yanlis iliskiye
    baglamak, hic cozumlememekten kotudur."""
    d, cakisma = {}, []
    for i, r in enumerate(v.iliski):
        for y in _yazimlar(r):
            if y in d and d[y] != i:
                cakisma.append((y, v.iliski[d[y]], r))
            d[y] = i
    assert not cakisma, f"iliski yazimi CAKISIYOR: {cakisma[:5]}"
    return d


def _varlik_sadele(kelimeler):
    """Son kelimedeki tamlayan ekini atar:  Yilmaz'in -> Yilmaz.
    Kesme yoksa ek de aranmaz -- varlik adlarinda ek YOK."""
    k = list(kelimeler)
    if k and "'" in k[-1]:
        k[-1] = k[-1].split("'")[0]
    return [x for x in k if x]


def _birlestir(parcalar):
    """<YOK> dolgusunu atar, kalanini BOSLUKLA birlestirir.

    ALT CIZGI YOK. Kullanici, 16 Eylul: "ahmet_kilic denediysen sikinti
    cunku _ yok." Dogru: alt cizgi ham grafin ad dizgesinde var, MODELIN
    DUNYASINDA YOK -- model "Ahmet" ve "Kilic" diye iki jeton goruyor.
    Ciktida alt cizgi basmak, olmayan bir jetoni varmis gibi gosterirdi.
    Tek jetonlu kollarda ad zaten tek parcadir, hicbir sey degismez."""
    return " ".join(p for p in parcalar if p != "<YOK>")


def kur(klasor, genislik=5):
    ayar = P.ayar_oku(klasor)
    v = M.veri_kur(ayar, yaz=lambda *a: None)
    # ILISKI ADLARI `Veri`de YOK -- veri modulunden alinir, TEK KAYNAK.
    import importlib
    v.iliski = list(importlib.import_module(ayar.veri_ad).ILISKI)
    snap = P.anlik_goruntuler(klasor)             # {adim: yol}
    adimlar = list(snap)
    pen = adimlar[-min(genislik, len(adimlar)):]
    sd = P.agirlik_ortalamasi([snap[x] for x in pen])
    net = (MODEL_SINIFI or ModelSade)(ayar, v.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    return ayar, v, net, pen


def _bolme(v, e, r1, r2):
    """Bu soru hangi bolmeden? Bolme listeleri TEK KAYNAK -- burada
    yeniden turetilmiyor, `veri_kur`un urettigi listelerde ARANIYOR."""
    if r2 is None:
        return "one" if any(x[0] == e and x[1] == r1 for x in v.one) else "?"
    for ad in ("ood", "ent_yok", "ent", "ent_kati", "comp", "ent_arama", "tr2"):
        for x in getattr(v, ad, []):
            if x[0] == e and x[1] == r1 and x[2] == r2:
                return {"tr2": "seen"}.get(ad, ad)
    return "?"


BOLME_NOT = {
    "one":      "atomik olgu -- egitimde GORULDU",
    "seen":     "2-hop zincir -- egitimde GORULDU (ezber)",
    "comp":     "ucluyu gormedi (varlik zincir basi OLDU)",
    "ent":      "varlik hic zincir basi olmadi -- ASIL SINAV",
    "ent_yok":  "ent + kisayol TIP OLARAK imkansiz",
    "ent_kati": "varlik hicbir zincirde yok",
    "ood":      "iki kenar da atomic_OOD (Wang'in tanimi)",
    "ent_arama": "arama havuzu -- HUKUMDE KULLANILMAZ",
    "?":        "graf'ta boyle bir zincir YOK (cevabi da yok)",
}


@torch.no_grad()
def sor(v, net, metin, yaz=print):
    # ALT CIZGI ZORUNLU DEGIL. Varlik artik JETONLARLA yaziliyor
    # (Ahmet | Kilic), yani "Ahmet_Kilic" TEK bir kelime degil; kullanici
    # da "Ahmet Kilic cocuk sehir" yazabilmeli. Alt cizgi yazarsa da olur.
    p = metin.replace(",", " ").replace("_", " ").split()
    ad2id = {_sade(_ad(v, e)): e for e in range(v.n_ent)}
    yaz2id = _iliski_sozluk(v) if hasattr(v, "iliski") else None
    # COZUMLEME SAGDAN: sondaki ILISKI kelimeleri (en fazla 2) ayrilir,
    # kalani VARLIK olur. "Ayse Yilmaz'in annesinin kardesi" ->
    # varlik "Ayse Yilmaz", iliskiler [anne, kardes]. Sira DOGRU: dizide
    # once gelen ILK hop'tur ("annesinin" = 1. hop).
    rid = []
    while p and len(rid) < 2 and yaz2id is not None \
            and _sade(p[-1]) in yaz2id:
        rid.insert(0, yaz2id[_sade(p.pop())])
    p = _varlik_sadele(p)
    if not p or not rid:
        yaz("  kullanim:  <varlik>'in <iliski>[nin] [<iliski2>]")
        yaz("     ornek:  Ayse Yilmaz'in annesi")
        yaz("             Ayse Yilmaz'in annesinin kardesi")
        yaz("             Ahmet Kilic cocuk sehir        (ciplak yazim da olur)")
        if yaz2id is not None:
            yaz(f"     iliskiler: {list(v.iliski)}")
        return
    e_ad = " ".join(p)
    if _sade(e_ad) not in ad2id:
        yak = [_ad(v, i) for a, i in ad2id.items()
               if a.startswith(_sade(e_ad)[:4])][:6]
        yaz(f"  '{e_ad}' grafta YOK." + (f"  Benzer: {yak}" if yak else ""))
        return
    e = ad2id[_sade(e_ad)]
    rels = [v.iliski[i] for i in rid]

    if len(rid) == 1:
        X, Pp, _ = M.kodla_1hop(v, [(e, rid[0], 0)])
        gercek = int(v.facts[e, rid[0]])
        kopru = None
    else:
        b = int(v.facts[e, rid[0]])
        gercek = int(v.facts[b, rid[1]]) if b >= 0 else -1
        kopru = b
        X, Pp, _ = M.kodla_2hop(v, [(e, rid[0], rid[1], 0, 0)])

    # `yuva_ara` TEK KAYNAK -- dogruluk() ve kayip da onu kullaniyor.
    ARA = v.yuva_ara[:Pp.shape[1]] if v.par is not None else         [(v.ent_off, v.ent_off + v.n_ent)]
    # --- OZYINELI COZUM -- ve neden ------------------------------------
    # KUSUR (16 Eylul, hakemlikte bulundu): dizi `kodla_*(v, [(..., 0)])`
    # ile kuruluyor, yani cevap yuvalarina VARLIK 0'in jetonlari giriyor.
    # Next-token sozlesmesinde P[j] pozisyonu, j. cevap jetonunu ONCEKI
    # jetona bakarak tahmin ediyor -- ve orada YER TUTUCU duruyordu.
    # Yalnizca P[0] temizdi (nedensel maske).
    #
    # OLCULDU (model_b14, 52000-60000 penceresi, 600 ornek):
    #     seen   yer tutucu 0.3950   ogretmenli 0.9983   ozyineli 0.9983
    #     comp   yer tutucu 0.0067   ogretmenli 0.0167   ozyineli 0.0167
    # Yani `sor.py` `seen`de 60 puan YANLIS cevap basiyordu; gorunur
    # belirtisi "Ipek Cetin" (gercek "Ipek Celik") gibi ILK jetonu dogru,
    # sonrasi yanlis cevaplardi.
    #
    # Ozyineli cozum OGRETMENLI okumayla BIREBIR ayni cikti (yukaridaki
    # tablo) -- yani `dogruluk()`un teacher-forced olmasi olcuyu
    # SISIRMIYOR. Bu ayri bir bulgu, belge/OLCULENLER.md'ye yazildi.
    Xc = X.copy()
    tah = []
    for j, (lo, hi) in enumerate(ARA):
        lg = net(torch.from_numpy(Xc).to(M.DEV)).float()
        t = int(lg[0, int(Pp[0, j]), lo:hi].argmax()) + lo
        tah.append(t)
        Xc[0, int(Pp[0, j]) + 1] = t      # tahmini GERI YAZ
    if v.par is None:
        m_ad = _ad(v, tah[0] - v.ent_off)
    else:
        m_ad = _birlestir(v.par_ad[j][tah[j] - ARA[j][0]]
                          for j in range(len(ARA)))

    g_ad = _ad(v, gercek) if gercek >= 0 else "(olgu YOK)"
    bol = _bolme(v, e, rid[0], rid[1] if len(rid) > 1 else None)
    if v.par is not None:
        yaz("  jeton   soru: " + " | ".join(
            v.par_ad[j][int(v.par[e, j])] for j in range(v.yuva)))
    # COZUMLEME geri okunur. Turkce EK URETMIYORUZ -- uretseydik
    # "anne" + "u" = "anneu" gibi sacmaliklar cikardi (ilk surumde
    # tam bu oldu). Onun yerine NE ANLASILDIGI acikca yazilir.
    yaz(f"  cozum   varlik '{_ad(v, e)}'  +  iliski {rels}")
    # BOS DIZE BASMA. Model uc yuvaya da <YOK> derse `_birlestir`
    # hepsini atar ve satir BOS kalirdi -- okuyan kisi "arac mi coktu,
    # model mi 'hicbir sey' dedi" AYIRAMAZ. (Bu projede bir kez daha
    # goruntu kusuru modelin hatasi sanilmisti; ayrinti onkayitta.)
    yaz(f"  model   {m_ad if m_ad else '(uc yuva da <YOK> -- model BOS cevap verdi)'}")
    yaz(f"  gercek  {g_ad}" + ("        DOGRU" if m_ad == g_ad and gercek >= 0
                               else "        YANLIS" if gercek >= 0 else ""))
    yaz(f"  bolme   {bol}   ({BOLME_NOT.get(bol, '')})")
    if kopru is not None and kopru >= 0:
        yaz(f"  kopru   {_ad(v, kopru)}   (model bunu YAZMADAN kullanmali)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("klasor")
    ap.add_argument("--genislik", type=int, default=5)
    ap.add_argument("--soru", action="append", default=None)
    a = ap.parse_args()

    ayar, v, net, pen = kur(a.klasor, a.genislik)
    print(f"=== sor  {ayar.ad} t{ayar.tohum} ===")
    print(f"  pencere {pen[0]}-{pen[-1]} ({len(pen)} anlik goruntu)")
    print(f"  varlik {v.n_ent}  iliski {v.n_rel}  sozluk {v.vocab}  "
          f"yuva {v.yuva}")
    print("  !! BU BIR OLCU DEGIL -- elle sorulan sorular SECILMIS "
          "sorulardir.")
    print("     Hukum `pencere_b` ile verilir. Bu arac ANLAMAK icin.")
    print()
    if a.soru:
        for q in a.soru:
            print(f"> {q}")
            sor(v, net, q)
            print()
        return 0
    print("  ornek:  Ayse_Yilmaz baba        |  Ayse_Yilmaz cocuk kardes")
    print("  cikis:  bos satir\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        sor(v, net, q)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
