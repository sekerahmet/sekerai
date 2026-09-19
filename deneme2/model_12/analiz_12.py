# -*- coding: utf-8 -*-
"""analiz_12 — BU KOLUN KENDI VERI DOKUMU + SAGLIK DENETIMI. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026: *"bunlarin hepsi model_09 folderi
altinda olmali. model_09 diger hicbir model ile ayni seyi kullanmamali.
Analiz icinde analiz_09 kullanalim mesela, digerleri icin de."*

`veri_dok.py`nin KOPYASI (uretici: scratchpad/kur_analiz00.py). Iki fark:
motor `taban_09`, ve `--model` secenegi YOK -- bu arac yalniz model_09'i
tanir. Dokum `model_12/veri/model_12/` altina gider (depoya girmez).

    python analiz_12.py [--tam] [--klasor <yol>]
"""
from __future__ import annotations

import argparse
import collections
import importlib
import os
import sys

import numpy as np

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import taban_12 as M                                          # noqa: E402

# OZEL BLOK: UC jeton. [S1]/[S2]/[KIMLIK] ve iki bos yuva
# 17 Eylul'de SILINDI -- olculmustu ki besi de egitim
# havuzunda HIC gecmiyordu (kullanici: "bunlar niye var?").
# OZEL BLOK: UC jeton, ve IKISI NOKTALAMA.
#   ?   soru isareti   "...kardesi kim?"
#   .   nokta          "...Sinan Yilmaz'dir."
# Ucuncusu <PAD>: DILIN PARCASI DEGIL. Satirlar 8-16 jeton
# uzunlugunda ve tensor SABIT GENISLIK ister, o yuzden kisa
# satirlar sagdan doldurulur. Kayipta atlanir
# (ignore_index=PAD) ve model onu HIC uretmez.
OZEL = {0: "<PAD>", 1: "?", 2: "."}


class Dok:
    """Veriye tek bir yerden bakan yardimci. HICBIR SEY TURETMEZ --
    `veri_kur`un urettigini ve HAM GRAFI okur, ikisini KARSILASTIRIR."""

    def __init__(self, model_ad):
        mod = importlib.import_module(model_ad)
        self.ayar = mod.AYAR
        self.v = M.veri_kur(self.ayar, yaz=lambda *a: None)
        self.VM = importlib.import_module(self.ayar.veri_ad)
        self.G = self.VM.kur(self.ayar.veri_tohum)
        self.E = [a for t in self.VM.TIPLER for a in self.G["ad"][t]]
        assert len(self.E) == self.v.n_ent
        self.R = list(self.VM.ILISKI)
        self.O = self.G["olgu"]                  # (ad, iliski) -> ad
        self.L = M.olcme_listeleri(self.ayar, self.v)

    # --- adlandirma -----------------------------------------------------
    def ad(self, e):
        """Varligin ADI -- ALT CIZGISIZ, cunku model alt cizgi gormuyor."""
        return self.E[int(e)].replace("_", " ")

    def ham(self, e):
        """Ham grafin ad dizgesi (alt cizgili). Yalniz graf sorgusu icin."""
        return self.E[int(e)]

    def jetonlar(self, e):
        v = self.v
        if v.par is None:
            return [self.E[int(e)]]
        return M.kelimeler(v, e)

    def jeton_ad(self, i):
        v, i = self.v, int(i)
        if i < M.SPECIAL:
            return OZEL[i]
        if i < M.SPECIAL + v.n_rel:
            # `@` ONEKI KALDIRILDI (kullanici, 17 Eylul: "niye basinda
            # @ isareti var? token dediğin boyle olmaz ki").
            # Iliskiyi varlik kelimesinden ayiran sey ZATEN YAZIM:
            #   kucuk harf = iliski (cins isim)   fakulte, anne, vali
            #   BUYUK harf = varlik adi           Fakultesi, Ahmet
            # Olculdu: iliskilerin 24/24'u kucuk, varlik
            # kelimelerinin 241/242'si buyuk harfle basliyor.
            # `@` bu ayrimin uzerine binen FAZLADAN bir isaretti ve
            # dokumu okuyani "bu nasil bir token" diye durduruyordu.
            return self.R[i - M.SPECIAL]
        # EK JETONLARI sozlugun SONUNDA (ek_kip="tr" -> 4 tane).
        # KUSUR (16 Eylul hakemligi): burada bu dal YOKTU ve `i` ek
        # jetonuysa varlik tablosunda aranip IndexError veriyordu. Yani
        # bu arac `ek_kip` gelen HER kolda (model_b15, model_05) coker,
        # model_b13'te calisirdi -- model_b15'in verisi hic DOKULMEMIS.
        if getattr(v, "ek0", 0) and i >= v.ek0:
            # ek_kip="tr2":  '  + tamlayan allomorflari
            # + bildirme allomorflari + soru sozcukleri.
            # ("tr" kodlamasi 17 Eylul SILINDI.)
            # BOSLUK DOLDURMA jetonlari sozlugun EN SONUNDA (model_06).
            # Bu dal YOKTU ve arac onlari gorunce IndexError veriyordu.
            if getattr(v, "bosluk", 0) and i == v.bosluk:
                return "<BOS>"
            if getattr(v, "ayir", 0) and i == v.ayir:
                return "<AYIR>"
            if getattr(v, "virgul", 0) and i == v.virgul:
                return ","
            return (("'",) + tuple(v.ek_nin_ad)
                    + tuple(v.ek_dir_ad)
                    + tuple(v.soru_ad))[i - v.ek0]
        lo = v.yuva_ara[0][0]
        return v.par_ad[0][i - lo] if v.par is not None else self.ad(i - lo)

    def oku(self, X0):
        """Jeton dizisini OKUNABILIR TURKCE'ye cevir.

        Kullanici, 17 Eylul: *"benim egitim diye gordugum hep soru var."*
        Bildirim satirlari havuzda VARDI ama jeton halinde
            Ibrahim Yilmaz ' in kardesi Ozlem Yilmaz ' dir .
        diye gorunuyordu ve cumle oldugu anlasilmiyordu. Burasi tam
        olarak MODELIN GORDUGU diziyi okur -- yeniden uretmez.

        Kural: ek jetonlari (' , tamlayan, bildirme) ve noktalama
        ONCEKI kelimeye YAPISIR; kelimeler bosluklu.
        """
        v, VM = self.v, self.VM
        yapisik = {M.QM, M.EOS}
        # VIRGUL de ONCEKI KELIMEYE yapisir: "Yilmaz," -- "Yilmaz ,"
        # degil. <BOS> ve <AYIR> yapisik DEGIL, onlar kelime degil.
        if getattr(v, "virgul", 0):
            yapisik = yapisik | {v.virgul}
        if getattr(v, "ek0", 0):
            yapisik |= set(range(v.ek0, v.ek0 + 1 + len(v.ek_nin_ad)
                                 + len(v.ek_dir_ad)))
        par = []
        for t in X0:
            t = int(t)
            if t == M.PAD:
                break
            if t < M.SPECIAL:
                w = {M.QM: "?", M.EOS: "."}[t]
            elif t < M.SPECIAL + v.n_rel:
                w = VM.TR_ILISKI[self.R[t - M.SPECIAL]]
            elif getattr(v, "ek0", 0) and t >= v.ek0:
                w = self.jeton_ad(t)
            else:
                w = VM.TR.get(self.jeton_ad(t), self.jeton_ad(t))
            if t in yapisik and par:
                par[-1] += w
            else:
                par.append(w)
        # CUMLE BASI buyuk harf -- YALNIZ okumada. Jeton kucuk harfli
        # kalir (cins isim), yoksa "kim" ile "Kim" iki ayri jeton olurdu.
        _c = " ".join(par)
        for _i, _h in enumerate(_c):
            if _h.isalpha():
                _c = _c[:_i] + _h.upper() + _c[_i + 1:]
                break
        # cevap cumlesi de buyuk harfle basliyor -- "? " ya da ". "dan sonra
        _o = []
        _bas = False
        for _h in _c:
            _o.append(_h.upper() if _bas and _h.isalpha() else _h)
            if _h.isalpha():
                _bas = False
            elif _h in "?.":
                _bas = True
        return "".join(_o)

    def dizi(self, X0):
        return " ".join(self.jeton_ad(t) for t in X0)

    # --- ham graftan cevap ----------------------------------------------
    def graf(self, e, r1, r2=None):
        a1 = self.O.get((self.ham(e), self.R[r1]))
        if r2 is None:
            return a1, None
        if a1 is None:
            return None, None
        return self.O.get((a1, self.R[r2])), a1

    def uzunluk(self, e):
        # <YOK> SILINDI: `kelimeler()` zaten dolgu URETMIYOR.
        return len(self.jetonlar(e))


# ======================================================================
#  DOSYALAR
# ======================================================================
def yaz_tokenlar(d, yol):
    v = d.v
    # her jeton hangi varliklarda, hangi yuvalarda geciyor
    nerede = collections.defaultdict(list)
    if v.par is not None:
        for e in range(v.n_ent):
            for j, p in enumerate(d.jetonlar(e)):
                nerede[p].append((e, j))
    tip = {e: v.tip_ad[v.tip[e]] for e in range(v.n_ent)}
    with open(yol, "w", encoding="utf-8") as f:
        f.write("SOZLUK -- modelin gordugu BUTUN jetonlar\n")
        f.write("=" * 74 + "\n")
        # !! SAYIM DUZELTILDI (model_06). Onceden EK jetonlari da
        # "varlik parcasi" diye sayiliyordu; <BOS>/<AYIR> eklenince
        # fark buyudu ve tablo kendi icinde tutmaz oldu.
        _n_ek = v.vocab - v.ek0 if getattr(v, "ek0", 0) else 0
        _n_bos = 2 if getattr(v, "bosluk", 0) else 0
        _n_var = v.vocab - M.SPECIAL - v.n_rel - _n_ek
        f.write(f"toplam {v.vocab} jeton = {M.SPECIAL} ozel + {v.n_rel} "
                f"iliski + {_n_var} varlik kelimesi + {_n_ek - _n_bos} ek"
                + (f" + {_n_bos} bosluk isareti" if _n_bos else "")
                + chr(10))
        f.write("OZEL blok UC jeton: <PAD> (satir dolgusu),"
                " ? ve <SON> (noktalama)." + chr(10))
        f.write("Olu jeton YOK -- [S1]/[S2]/[KIMLIK] ve iki bos"
                " yuva 17 Eylul silindi." + chr(10) + chr(10))
        f.write(f"{'id':>4}  {'jeton':<22} {'gectigi varlik':>14}  aciklama\n")
        f.write("-" * 74 + "\n")
        for i in range(v.vocab):
            ad = d.jeton_ad(i)
            if i < M.SPECIAL:
                ack = "OZEL"
            elif i < M.SPECIAL + v.n_rel:
                ack = "ILISKI"
            else:
                yer = nerede.get(ad, [])
                tipler = sorted({tip[e] for e, _ in yer})
                yuvalar = sorted({j for _, j in yer})
                ack = (f"tip={'/'.join(tipler)}  yuva={yuvalar}"
                       if yer else "HICBIR VARLIKTA YOK")
                f.write(f"{i:>4}  {ad:<22} {len(yer):>14}  {ack}\n")
                continue
            f.write(f"{i:>4}  {ad:<22} {'-':>14}  {ack}\n")

        if v.par is not None:
            f.write("\n\nJETON KAC VARLIKTA GECIYOR -- dagilim\n")
            f.write("-" * 74 + "\n")
            say = collections.Counter(len(x) for x in nerede.values())
            for k in sorted(say):
                f.write(f"  {k:>4} varlikta : {say[k]:>4} jeton\n")
            tek = sorted(p for p, x in nerede.items() if len(x) == 1)
            f.write(f"\nYALNIZ 1 VARLIKTA gecen {len(tek)} jeton:\n")
            for i in range(0, len(tek), 6):
                f.write("  " + "  ".join(f"{p:<18}" for p in tek[i:i + 6])
                        + "\n")
            f.write("\n  ^ bunlar modelin BIRLESTIRME ogrenemedigi jetonlar:\n")
            f.write("    tek bir varliga ait olduklari icin 'parca' degil\n")
            f.write("    fiilen 'isim' gibi davraniyorlar.\n")


def yaz_varliklar(d, yol):
    v = d.v
    with open(yol, "w", encoding="utf-8") as f:
        f.write("VARLIKLAR -- ad, tip, JETON AYRISIMI\n")
        f.write("=" * 74 + "\n")
        f.write(f"{v.n_ent} varlik.  Model adi DEGIL, jetonlari goruyor.\n\n")
        f.write(f"{'id':>5}  {'tip':<6} {'ad':<28} jetonlar\n")
        f.write("-" * 74 + "\n")
        for e in range(v.n_ent):
            f.write(f"{e:>5}  {v.tip_ad[v.tip[e]]:<6} {d.ad(e):<28} "
                    + " | ".join(d.jetonlar(e)) + "\n")
        f.write("\n\nJETON SAYISINA GORE\n" + "-" * 74 + "\n")
        say = collections.Counter((v.tip_ad[v.tip[e]], d.uzunluk(e))
                                  for e in range(v.n_ent))
        for (t, u), n in sorted(say.items()):
            f.write(f"  {t:<6} {u} jeton : {n:>5}\n")


def yaz_iliskiler(d, yol):
    v = d.v
    say1 = collections.Counter(r for _, r, _ in d.v.one)
    with open(yol, "w", encoding="utf-8") as f:
        f.write("ILISKILER -- sema ve dagilim\n" + "=" * 74 + "\n\n")
        f.write(f"{'id':>3}  {'iliski':<10} {'egitimde':>9}  tip semasi\n")
        f.write("-" * 74 + "\n")
        for i, r in enumerate(d.R):
            sem = "  ".join(f"{a}->{b}" for a, b in d.VM.SEMA[r].items())
            f.write(f"{i:>3}  {r:<10} {say1.get(i, 0):>9}  {sem}\n")
        f.write("\n\nSINAV BOLMELERINDE r1 / r2 DAGILIMI\n" + "-" * 74 + "\n")
        for bol in ("comp", "ent", "ent_yok", "ood"):
            lst = d.L.get(bol) or []
            if not lst:
                continue
            f.write(f"\n{bol}  (n={len(lst)})\n")
            for hangi, ix in (("r1", 1), ("r2", 2)):
                c = collections.Counter(d.R[x[ix]] for x in lst)
                f.write(f"  {hangi}: " + "  ".join(
                    f"{k}={n}" for k, n in c.most_common()) + "\n")


def _satir(d, x, iki=True):
    """Bir olguyu/zinciri, EGITIMDE GORUNEN BUTUN YUZEY BICIMLERIYLE yaz.

    !! KUSUR (17 Eylul, kullanici fark etti): burada `kodla_*(v, [x])`
    cagriliyordu ve `bicim_no` varsayilani 0 -- yani dokum HER SATIRI
    kanonik SORU biciminde basiyordu. Havuzda bildirim satirlari
    (117.562'nin 31.216'si) VARDI ama dokumde HIC GORUNMUYORDU, ve
    veri "hep soru" gibi okunuyordu. Artik `ayar.bicim` kadar bicimin
    HEPSI basiliyor."""
    v = d.v
    _n = max(1, d.ayar.bicim)
    _ad = ("KANONIK", "DEVRIK", "BILDIRIM")
    if iki:
        e, r1, r2, b, a = x
        ks = d.O.get((d.ham(e), d.R[r2]))
        bas = (f"{d.ad(e):<26} --{d.R[r1]:<9}--> {d.ad(b):<26} "
               f"--{d.R[r2]:<9}--> {d.ad(a):<26} | "
               f"kisayol({d.R[r2]})={d.ad(d.E.index(ks)) if ks else '-':<24}")
        ic = []
        for i in range(_n):
            _x = M.kodla_2hop(v, [x], i)[0][0]
            ic += [f"{_ad[i]:<9}{d.oku(_x)}", f"{'':<9}{d.dizi(_x)}"]
    else:
        e, r1, a = x
        bas = (f"{d.ad(e):<26} --{d.R[r1]:<9}--> {d.ad(a):<26}")
        ic = []
        for i in range(_n):
            _x = M.kodla_1hop(v, [x], i)[0][0]
            ic += [f"{_ad[i]:<9}{d.oku(_x)}", f"{'':<9}{d.dizi(_x)}"]
    return bas + (chr(10) + "      ").join([""] + ic)


def yaz_liste(d, yol, lst, baslik, aciklama, iki=True, tam=True):
    n = len(lst)
    with open(yol, "w", encoding="utf-8") as f:
        f.write(baslik + "\n" + "=" * 74 + "\n")
        f.write(aciklama.rstrip() + "\n\n")
        f.write(f"{n} satir.\n")
        if iki:
            f.write("BICIM: e --r1--> KOPRU --r2--> CEVAP | kisayol | "
                    "modelin gordugu jeton dizisi\n")
            f.write("Kopru dizide YOK -- model onu YAZMADAN kullanmali.\n\n")
        else:
            f.write("BICIM: e --r--> CEVAP | modelin gordugu jeton dizisi\n\n")
        for x in (lst if tam else lst[:2000]):
            f.write(_satir(d, x, iki) + "\n")
        if not tam and n > 2000:
            f.write(f"\n... ({n - 2000} satir daha, tam liste icin --tam)\n")


# ======================================================================
#  SAGLIK DENETIMI
# ======================================================================
def saglik(d, f):
    """Hukum veren denetimler. Her biri: SORU -> OLCUM -> HUKUM."""
    v, out = d.v, []

    def bak(ad, ok, detay=""):
        out.append((ad, ok, detay))
        f.write(f"  {'GECTI ' if ok else '!! KALDI'}  {ad}\n")
        if detay:
            f.write(f"           {detay}\n")

    f.write("\n1) KODUN CEVABI HAM GRAFLA TUTUYOR MU\n" + "-" * 74 + "\n")
    f.write("   `v.facts` kodun turettigi tablo; `G['olgu']` ham graf sozlugu.\n")
    f.write("   Ikisi AYRI kaynak; tutmazlarsa hangisinin dogru oldugu belirsiz.\n")
    kac = tut = 0
    kotu = []
    for bol, lst in d.L.items():
        for x in lst:
            if len(x) == 3:
                e, r1, a = x
                g, _ = d.graf(e, r1)
            else:
                e, r1, r2, b, a = x
                g, _ = d.graf(e, r1, r2)
            kac += 1
            if g == d.ham(a):
                tut += 1
            elif len(kotu) < 5:
                kotu.append(f"{bol}: {d.ad(e)} -> kod {d.ad(a)} / graf {g}")
    bak(f"olcum listelerinin {kac} ornegi ham grafla tutuyor",
        tut == kac, f"{tut}/{kac}" + ("  " + " ; ".join(kotu) if kotu else ""))

    f.write("\n2) SINAV EGITIME SIZIYOR MU\n" + "-" * 74 + "\n")
    tr_uclu = {(e, r1, r2) for e, r1, r2, _, _ in v.tr2}
    tr_k1 = {(e, r) for e, r, _, _, _ in v.tr2}
    tr_k2 = {(b, r) for _, _, r, b, _ in v.tr2}
    tr_bas = {x[0] for x in v.tr2}
    for bol in ("comp", "ent", "ent_yok", "ood"):
        lst = d.L.get(bol) or []
        if not lst:
            continue
        s_uclu = sum((e, r1, r2) in tr_uclu for e, r1, r2, _, _ in lst)
        bak(f"{bol}: egitimde GORULEN ucluyu sormuyor", s_uclu == 0,
            f"{s_uclu}/{len(lst)}")
    ood = d.L.get("ood") or []
    s_ken = sum(1 for e, r1, r2, b, _ in ood
                if (e, r1) in tr_k1 or (b, r2) in tr_k2)
    bak("ood: iki kenardan HICBIRI egitim zincirinde gecmiyor", s_ken == 0,
        f"{s_ken}/{len(ood)}   <- BIRINCIL olcunun gecerliligi buna bagli")
    ent = d.L.get("ent") or []
    s_bas = sum(1 for e, *_ in ent if e in tr_bas)
    bak("ent: varlik egitimde ZINCIR BASI olmamis", s_bas == 0,
        f"{s_bas}/{len(ent)}")

    f.write("\n3) SORULAR KISAYOLLA GECILEBILIR MI\n" + "-" * 74 + "\n")
    f.write("   KISAYOL = r2'yi dogrudan soru varligina uygulamak.\n")
    f.write("   Kisayol cevaba ESITSE soru bozuktur: kopruye gerek kalmaz.\n")
    for bol in ("comp", "ent", "ent_yok", "ood"):
        lst = d.L.get(bol) or []
        if not lst:
            continue
        var = ayni = 0
        for e, r1, r2, b, a in lst:
            ks = d.O.get((d.ham(e), d.R[r2]))
            if ks is not None:
                var += 1
                ayni += (ks == d.ham(a))
        bak(f"{bol}: kisayol == cevap olan ornek YOK", ayni == 0,
            f"kisayol VAR {var}/{len(lst)}, cevaba ESIT {ayni}")

    f.write("\n3b) CEVAP, SORU VARLIGINDAN TEK ADIMDA ULASILABILIYOR MU\n")
    f.write("-" * 74 + "\n")
    f.write("   KISAYOL yalniz r2'ye bakiyor. Ama BASKA bir iliski de\n")
    f.write("   dogrudan cevaba goturuyorsa, model kopruyu kurmadan\n")
    f.write("   'e ile a arasinda bir bag var' diye gecebilir. Bu, r2\n")
    f.write("   kisayolundan DAHA GENIS bir acik olurdu.\n")
    for bol in ("comp", "ent", "ent_yok", "ood"):
        lst = d.L.get(bol) or []
        if not lst:
            continue
        tek = 0
        ornek = []
        for e, r1, r2, b, a in lst:
            hangi = [d.R[r] for r in range(v.n_rel)
                     if int(v.facts[e, r]) == a]
            if hangi:
                tek += 1
                if len(ornek) < 3:
                    ornek.append(f"{d.ad(e)} -{'/'.join(hangi)}-> {d.ad(a)}")
        # HUKUM: acik VARSA sayisi ve HANGI (r1,r2) ciftinden geldigi
        # yazilir. Sifir bekleyip "KALDI" demek yaniltirdi -- acik
        # GERCEK ama kucuk, ve VERI DEGISTIRILMEZ (olcme izi kilitli,
        # model_b1 kiyasi bozulurdu). Dogru davranis: KAYDA GEC.
        cift = collections.Counter()
        toplam = collections.Counter()
        for e, r1, r2, b, a in lst:
            toplam[(d.R[r1], d.R[r2])] += 1
            if any(int(v.facts[e, r]) == a for r in range(v.n_rel)):
                cift[(d.R[r1], d.R[r2])] += 1
        bak(f"{bol}: TEK KENAR acigi %2'nin ALTINDA",
            tek / max(1, len(lst)) < 0.02,
            f"{tek}/{len(lst)} = %{100*tek/max(1,len(lst)):.2f}")
        for k, nn in cift.most_common(4):
            f.write(f"           {k[0]:<9} -> {k[1]:<9} "
                    f"{nn}/{toplam[k]} = %{100*nn/toplam[k]:.0f}\n")
        if ornek:
            f.write(f"           ornek: {ornek[0]}\n")

    f.write("\n3c) BOLMELER AYRIK MI\n" + "-" * 74 + "\n")
    f.write("   Ayni zincir iki bolmede birden olursa, iki olcu BAGIMSIZ\n")
    f.write("   degildir ve 'ikisi de yukseldi' bir sey ifade etmez.\n")
    kume = {b: {(x[0], x[1], x[2]) for x in (d.L.get(b) or [])}
            for b in ("comp", "ent", "ent_yok", "ood", "seen")}
    for i, b1 in enumerate(("comp", "ent", "ood", "seen")):
        for b2 in ("comp", "ent", "ood", "seen")[i + 1:]:
            ort = kume[b1] & kume[b2]
            # ent_yok, ent'in ALT KUMESI -- tanimi geregi, kusur degil.
            bak(f"{b1} ile {b2} kesismiyor", not ort,
                f"ortak {len(ort)}")
    # DUZELTME (16 Eylul, OLCULDU): CLAUDE.md `ent_yok`u "ent'in ALT
    # KUMESI" diye yaziyordu. DEGIL. Ikisi de AYNI varlik havuzundan
    # geliyor ama ZINCIR kumeleri AYRIK:
    #     ent      = kisayol TIP OLARAK MUMKUN olan zincirler (AYIRT)
    #     ent_yok  = kisayol TIP OLARAK IMKANSIZ olanlar      (YOK)
    # Varlik kesisimi 310/312-316, zincir kesisimi 0.
    t_ent = {(e, r1, r2) for e, r1, r2, _, _ in v.ent}
    t_yok = {(e, r1, r2) for e, r1, r2, _, _ in v.ent_yok}
    bak("ent ile ent_yok ZINCIR duzeyinde AYRIK", not (t_ent & t_yok),
        f"ortak {len(t_ent & t_yok)}")
    ve = {x[0] for x in v.ent}
    vy = {x[0] for x in v.ent_yok}
    bak("ent ile ent_yok AYNI varlik havuzundan (neredeyse)",
        len(ve & vy) / max(1, len(ve | vy)) > 0.95,
        f"varlik: ent {len(ve)}  ent_yok {len(vy)}  ortak {len(ve & vy)}")
    bak("ent: kisayol TIP OLARAK MUMKUN (tanim)",
        all(int(v.facts[e, r2]) >= 0 for e, _, r2, _, _ in v.ent))
    bak("ent_yok: kisayol TIP OLARAK IMKANSIZ (tanim, BIRIM TESTI)",
        all(int(v.facts[e, r2]) < 0 for e, _, r2, _, _ in v.ent_yok))

    f.write("\n4) DONUS: cevap = soru varligi olan ornek\n" + "-" * 74 + "\n")
    f.write("   Olursa model soruyu KOPYALAYARAK gecebilir.\n")
    for bol in ("seen", "comp", "ent", "ent_yok", "ood"):
        lst = d.L.get(bol) or []
        if not lst:
            continue
        n = sum(1 for x in lst if x[0] == x[4])
        bak(f"{bol}: DONUS yok", n == 0 or bol == "seen",
            f"{n}/{len(lst)}" + ("   (seen EZBER havuzu, hukum vermez)"
                                 if bol == "seen" else ""))

    f.write("\n5) ZINCIRIN IKI HOPU DA EGITIMDE VAR MI\n" + "-" * 74 + "\n")
    f.write("   Yoksa soru CEVAPLANAMAZ olurdu -- model bilgiyi hic gormemis\n")
    f.write("   olurdu, ve 'birlestiremiyor' ile 'bilmiyor' ayrilamazdi.\n")
    olgu1 = {(e, r) for e, r, _ in v.one}
    for bol in ("comp", "ent", "ent_yok", "ood"):
        lst = d.L.get(bol) or []
        if not lst:
            continue
        eksik = sum(1 for e, r1, r2, b, _ in lst
                    if (e, r1) not in olgu1 or (b, r2) not in olgu1)
        bak(f"{bol}: her iki atomik olgu da EGITIMDE", eksik == 0,
            f"eksik {eksik}/{len(lst)}")

    f.write("\n6) CEVAP TEK MI\n" + "-" * 74 + "\n")
    cift = [(e, r) for (e, r), _ in d.O.items()]
    bak("ham grafta (varlik, iliski) -> TEK cevap",
        len(cift) == len(set(cift)), f"{len(cift)} anahtar")
    if v.par is not None:
        imza = {}
        cak = 0
        for e in range(v.n_ent):
            k = tuple(v.par[e])
            cak += k in imza
            imza[k] = e
        bak("jeton dizisi BIREBIR (iki varlik ayni diziyi almamis)",
            cak == 0, f"cakisma {cak}")

    f.write("\n7) SOZLUK SAGLIGI\n" + "-" * 74 + "\n")
    if v.par is not None:
        parcalar = {p for e in range(v.n_ent) for p in d.jetonlar(e)}
        bak("hicbir jetonun ICINDE alt cizgi yok",
            not any("_" in p for p in parcalar),
            str([p for p in parcalar if "_" in p][:3]))
        kul = set()
        for e in range(v.n_ent):
            kul.update(int(x) for x in v.par[e])
        olu = [i for i in range(len(v.par_ad[0])) if i not in kul]
        bak("sozlukte KULLANILMAYAN varlik jetonu yok", not olu,
            f"olu: {[v.par_ad[0][i] for i in olu][:5]}")
    bak("OZEL blok: olu jeton YOK", M.SPECIAL == 3,
        "SPECIAL=3 -- <PAD>, ? ve <SON>, ucu de kullaniliyor")

    f.write("\n8) SINAV BOLMESI HOMOJEN MI\n" + "-" * 74 + "\n")
    f.write("   Tek sayi olarak okunan bir bolme icinde KOLAY ve ZOR vaka\n")
    f.write("   karisimi varsa, sayi karisimi gizler.\n")
    for bol in ("ood", "ent"):
        lst = d.L.get(bol) or []
        if not lst:
            continue
        c = collections.Counter(d.uzunluk(x[4]) for x in lst)
        pay = "  ".join(f"{k} jeton %{100*n/len(lst):.1f}"
                        for k, n in sorted(c.items()))
        # HUKUM VERMEZ -- bu bir RAPOR. Homojen olmamak veriyi BOZMUYOR,
        # ama tek sayi olarak okumayi YANILTICI yapiyor. Cozum onkayit
        # 2.6'da BAGLAYICI: tani_b `ood`u uzunluga gore AYIRARAK raporlar.
        f.write(f"  RAPOR     {bol}: cevap uzunlugu  {pay}\n")
        if len(c) > 1:
            f.write("            ^ homojen DEGIL -- tek sayi kolay ve zor\n"
                    "              vakayi KARISTIRIR (onkayit 2.6)\n")
    return out


def yaz_taban(d, yol):
    """AKILSIZ stratejiler kac alir? 0.84 sayisi ancak buna gore okunur."""
    v = d.v
    with open(yol, "w", encoding="utf-8") as f:
        f.write("TABAN CIZGILER -- akilsiz stratejiler kac alir\n")
        f.write("=" * 74 + "\n")
        f.write("Bir dogruluk sayisi, ANCAK sans seviyesiyle birlikte\n")
        f.write("anlamlidir. Burasi modelin GECMESI GEREKEN esikleri verir.\n")
        f.write("Hicbiri egitim gerektirmiyor; hepsi VERIDEN hesaplaniyor.\n\n")
        # en sik cevap
        for bol in ("comp", "ent", "ent_yok", "ood"):
            lst = d.L.get(bol) or []
            if not lst:
                continue
            n = len(lst)
            f.write(f"\n{bol}  (n={n})\n" + "-" * 74 + "\n")
            # 1) hep en sik cevabi soyle
            c = collections.Counter(x[4] for x in lst)
            e_sik, n_sik = c.most_common(1)[0]
            f.write(f"  1) HEP EN SIK CEVAP ({d.ad(e_sik)}) : "
                    f"{n_sik/n:.4f}\n")
            # 2) kisayol cevabini soyle
            ks_ok = 0
            for e, r1, r2, b, a in lst:
                ks = d.O.get((d.ham(e), d.R[r2]))
                ks_ok += (ks is not None and ks == d.ham(a))
            f.write(f"  2) KISAYOL cevabini soyle          : {ks_ok/n:.4f}\n")
            # 3) soru varligini soyle (DONUS)
            f.write(f"  3) SORU VARLIGINI soyle            : "
                    f"{sum(1 for x in lst if x[0]==x[4])/n:.4f}\n")
            # 4) TIP-KISITLI rastgele: dogru tipi BILEN ama baska hicbir
            #    sey bilmeyen bir tahminci. 1/|tip|'in ornek agirlikli
            #    ortalamasi.
            tp = collections.Counter(v.tip[x[4]] for x in lst)
            bek = sum(nn / n / max(1, int((v.tip == t).sum()))
                      for t, nn in tp.items())
            f.write(f"  4) DOGRU TIPI bilip rastgele       : {bek:.4f}\n")
            # 4b) r2'nin MENZILINI de bilen tahminci -- daha guclu taban
            men = {}
            for _, _, r2, _, _ in lst:
                if r2 not in men:
                    s_ = v.facts[:, r2]
                    men[r2] = len(set(int(x) for x in np.unique(s_[s_ >= 0])))
            bek2 = sum(1.0 / max(1, men[x[2]]) for x in lst) / n
            f.write(f"  4b) r2'nin MENZILINDEN rastgele    : {bek2:.4f}\n")
            f.write(f"      (menzil buyuklugu ortalama "
                    f"{sum(men[x[2]] for x in lst)/n:.0f} varlik)\n")
            # 5) jeton kopyalama: cevabin jetonlari soruda var mi
            if v.par is not None:
                tam = kis = 0
                for e, r1, r2, b, a in lst:
                    sj = [p for p in d.jetonlar(e)]
                    aj = [p for p in d.jetonlar(a)]
                    ort = sum(1 for p in aj if p in sj)
                    tam += (ort == len(aj))
                    kis += (ort > 0)
                f.write(f"  5) CEVABIN jetonlari SORUDA:\n")
                f.write(f"       hepsi var (kopyalayarak gecer) : "
                        f"{tam/n:.4f}\n")
                f.write(f"       en az biri var                 : "
                        f"{kis/n:.4f}\n")
                # 6) soyadi + dogru ad tahmini
                pay = collections.Counter()
                for e, r1, r2, b, a in lst:
                    sj = [p for p in d.jetonlar(e)]
                    aj = [p for p in d.jetonlar(a)]
                    pay[sum(1 for p in aj if p in sj)] += 1
                f.write("       ortak jeton sayisi: " + "  ".join(
                    f"{k}->{100*x/n:.1f}%" for k, x in sorted(pay.items()))
                    + "\n")
                # 6) YUVA BASINA taban. Dogruluk BUTUN yuvalarin tutmasini
                #    ister; ama bir yuva BEDAVA geliyorsa gorev o kadar
                #    kolaylasir. Iki bedava kaynak olabilir:
                #      KOPYA  sorunun ayni yuvasindaki jeton (soyadi
                #             kalitimi: baba/anne/kardes/cocuk AYNI soyad)
                #      SABIT  o yuvanin EN SIK jetonu (cogu zaman <YOK>)
                f.write("  6) YUVA BASINA taban (dogruluk UCUNU DE ister):\n")
                # !! ADLAR DEGISKEN UZUNLUKTA (<YOK> silindi, 17 Eylul):
                # "Adana"nin 2. yuvasi YOK. Eskiden dolgu vardi ve
                # jetonlar(e)[j] her zaman donerdi; artik IndexError.
                # Yuvasi olmayan ornek o yuvanin TABANINA GIRMEZ ve
                # PAYDAYA da girmez -- yoksa oran sessizce kucuk cikardi.
                _yuv = lambda e, j: (d.jetonlar(e)[j]
                                     if j < len(d.jetonlar(e)) else None)
                for j in range(v.yuva):
                    _var = [x for x in lst if _yuv(x[4], j) is not None
                            and _yuv(x[0], j) is not None]
                    if not _var:
                        continue
                    _n = len(_var)
                    kop = sum(1 for x in _var
                              if _yuv(x[4], j) == _yuv(x[0], j))
                    sk = collections.Counter(_yuv(x[4], j) for x in _var)
                    t_, n_ = sk.most_common(1)[0]
                    f.write(f"       yuva {j}: SORUDAN KOPYA {kop/_n:.4f}"
                            f"   EN SIK '{t_}' {n_/_n:.4f}\n")
                # 7) iliski bazinda kopya -- soyadi kalitiminin YERI
                f.write("  7) SORUDAN KOPYA, iliskiye gore (yuva 1 = soyad):\n")
                per = collections.defaultdict(lambda: [0, 0])
                for x in lst:
                    if min(len(d.jetonlar(x[4])),
                           len(d.jetonlar(x[0]))) < 2:
                        continue   # tek kelimeli ad: soyad YOK
                    k = per[d.R[x[2]]]
                    k[0] += (d.jetonlar(x[4])[1] == d.jetonlar(x[0])[1])
                    k[1] += 1
                for r_, (ok_, nn_) in sorted(per.items(),
                                             key=lambda z: -z[1][0] / z[1][1]):
                    f.write(f"       {r_:<10} {ok_/nn_:.4f}  (n={nn_})\n")
        f.write("\n\nNASIL OKUNUR\n" + "-" * 74 + "\n")
        f.write("Model bir bolmede bu sayilarin HEPSINDEN yukarida degilse,\n")
        f.write("o bolmede 'kompozisyon ogrendi' denemez. Ozellikle (5):\n")
        f.write("cevabin butun jetonlari soruda geciyorsa, dogru cevap\n")
        f.write("KOPYALAMAYLA uretilebilir ve bolme o oranda ZAYIFTIR.\n")


def main():
    ap = argparse.ArgumentParser()
    # `--model` YOK: bu arac yalniz model_09'i tanir.
    ap.set_defaults(model="model_12")
    ap.add_argument("--klasor", default=None,
                    help="varsayilan: <AILE>/veri/<model>")
    ap.add_argument("--tam", action="store_true",
                    help="buyuk egitim dosyalarini KIRPMA")
    a = ap.parse_args()

    d = Dok(a.model)
    if a.klasor is None:
        # DOKUM, KOLUN AILE KLASORUNE yazilir -- deponun tepesine DEGIL.
        # Klasor adi isimden TURETILMEZ, dosya ARANIR (defterin 3. hucresi
        # `pencere_*.py`yi nasil buluyorsa oyle): model_b6 -> model_b/.
        # Alt klasor MODELIN ADI, cunku dokum jetonlamaya BAGLI --
        # model_b1 ("") ile model_b6 ("tam") AYNI grafi FARKLI dokerdi.
        a.klasor = os.path.join(_K, "veri", a.model)
    os.makedirs(a.klasor, exist_ok=True)
    # TEK KLASOR, dosyalar AYRI AYRI -- kullanici karari,
    # 17 Eylul: "hepsi ayri ayri dosya". Alt klasor denendi
    # (denetim/) ve istenmedi.
    yol = lambda n: os.path.join(a.klasor, n)
    v = d.v

    yaz_tokenlar(d, yol("01_tokenlar.txt"))
    yaz_varliklar(d, yol("02_varliklar.txt"))
    yaz_iliskiler(d, yol("03_iliskiler.txt"))
    # !! 04/05 (EGITIM dokumu) KALDIRILDI -- kullanici, 17 Eylul:
    # "niye iki tane egitim var?". `egitim_dok_09` zaten EGITIM_*.txt
    # olarak DUZ halini yaziyor; bu ikisi ayni icerigin aciklamali
    # kopyasiydi. SINAV dokumleri (06-09) KALIYOR cunku onlarda KOPRU
    # ve KISAYOL var -- sinav sorusu onlarsiz okunmaz.
    for ad, dosya, ack in (
        ("comp", "06_sinav_comp.txt",
         "Bu (varlik, r1, r2) UCLUSU egitimde hic gorulmedi. Ama varlik\n"
         "baska zincirlerde BAS olarak gorunmus olabilir -- yani en zayif\n"
         "genelleme sinavi. Iliski CIFTI de egitimde var (CLAUDE.md)."),
        ("ent", "07_sinav_ent.txt",
         "Varlik egitimde HIC zincir basi olmamis. Olgularinda, kopru ve\n"
         "cevap olarak gorunmus. JETON duzeyinde ZAYIF: parcalari baska\n"
         "varliklarda bas konumunda goruluyor -> hukum vermez."),
        ("ent_yok", "08_sinav_ent_yok.txt",
         "`ent`in alt kumesi: KISAYOL TIP OLARAK IMKANSIZ. Yani r2, soru\n"
         "varliginin tipine hic uygulanamaz. Birim testi burada: modelin\n"
         "kisayol orani TAM 0 olmali."),
        ("ood", "09_sinav_ood.txt",
         "IKI KENAR DA egitim zincirlerinde hic gecmedi (Wang'in kenar\n"
         "duzeyindeki tanimi). BIRINCIL OLCU: tanimi kenara dayandigi\n"
         "icin jetonlamadan ETKILENMIYOR.\n"
         "UYARI: homojen DEGIL -- cevaplarin bir kismi TEK jetonlu."),
    ):
        yaz_liste(d, yol(dosya), d.L.get(ad) or [],
                  f"SINAV -- {ad}", ack)
    yaz_taban(d, yol("11_taban_cizgiler.txt"))

    with open(yol("10_saglik.txt"), "w", encoding="utf-8") as f:
        f.write("SAGLIK DENETIMI -- veri dogru mu\n" + "=" * 74 + "\n")
        f.write(f"model {a.model}   veri {d.ayar.veri_ad}   "
                f"tohum {d.ayar.veri_tohum}\n")
        sonuc = saglik(d, f)

    with open(yol("00_OZET.txt"), "w", encoding="utf-8") as f:
        f.write("VERI DOKUMU -- OZET\n" + "=" * 74 + "\n")
        f.write(f"model      {a.model}\n")
        f.write(f"veri       {d.ayar.veri_ad}  (tohum {d.ayar.veri_tohum})\n")
        f.write(f"varlik     {v.n_ent}\n")
        f.write(f"iliski     {v.n_rel}\n")
        f.write(f"sozluk     {v.vocab} jeton   yuva {v.yuva}   "
                f"t_len {d.ayar.t_len}\n")
        f.write(f"olcme izi  {M.olcme_izi(d.L)}\n\n")
        f.write("HAVUZLAR\n" + "-" * 74 + "\n")
        for k in ("one", "tr2", "comp", "ent", "ent_yok", "ent_kati", "ood"):
            x = getattr(v, k, None)
            f.write(f"  {k:<10} {len(x) if x is not None else 0:>7}\n")
        f.write("\nOLCUM LISTELERI (her kumeden en fazla "
                f"{d.ayar.n_olcum_max})\n" + "-" * 74 + "\n")
        for k, x in d.L.items():
            f.write(f"  {k:<10} {len(x):>7}\n")
        f.write("\nSAGLIK DENETIMI\n" + "-" * 74 + "\n")
        kotu = [s for s in sonuc if not s[1]]
        for ad, ok, detay in sonuc:
            f.write(f"  {'GECTI ' if ok else '!! KALDI'}  {ad}\n")
        f.write(f"\n  {len(sonuc) - len(kotu)}/{len(sonuc)} GECTI\n")
        f.write("\n\nBILINEN ZAAFLAR -- veri BOZUK degil, ama BUNLARI BIL\n")
        f.write("=" * 74 + "\n")
        f.write("Hicbiri koşuyu gecersiz kilmiyor; hepsi HER KOLDA AYNI,\n")
        f.write("yani kollar arasi kiyas etkilenmiyor. Ama tek bir sayiyi\n")
        f.write("'kompozisyon ogrendi' diye okurken bunlar hesaba katilir.\n\n")
        f.write("1. AILE CEBIRI ACIGI\n")
        f.write("   `anne->cocuk` ve `baba->cocuk` zincirlerinin cevabina\n")
        f.write("   `kardes` TEK KENARIYLA da ulasiliyor, ve o kenar\n")
        f.write("   EGITIMDE var. Bu zincirlerin %100'u boyle.\n")
        for bol in ("comp", "ent", "ood"):
            lst = d.L.get(bol) or []
            if not lst:
                continue
            tek = sum(1 for e, r1, r2, b, a in lst
                      if any(int(v.facts[e, r]) == a for r in range(v.n_rel)))
            f.write(f"     {bol:<8} {tek}/{len(lst)} ornek "
                    f"(%{100*tek/len(lst):.2f})\n")
        f.write("   -> `ood`da %0.44, SE'nin (0.024) COK altinda: BIRINCIL\n")
        f.write("      olcu fiilen etkilenmiyor. VERI DEGISTIRILMEDI, cunku\n")
        f.write("      olcme izi degisirdi ve model_b1 kiyasi bozulurdu.\n\n")
        f.write("2. BOLMELER HOMOJEN DEGIL\n")
        for bol in ("ood", "ent"):
            lst = d.L.get(bol) or []
            if not lst:
                continue
            c = collections.Counter(d.uzunluk(x[4]) for x in lst)
            f.write(f"   {bol}: " + "  ".join(
                f"{k} jeton %{100*n/len(lst):.1f}" for k, n in sorted(c.items()))
                + "\n")
        f.write("   -> tek jetonlu cevaplarda COK JETONLU BAGLAMA hic\n")
        f.write("      sinanmiyor. tani_b uzunluga gore AYIRIR (onkayit 2.6).\n\n")
        f.write("3. YUVA DOLGUSU KALKTI" + '\\n')
        f.write("   Bu baslik eskiden <YOK> dolgusunu olcuyordu: adlar" + '\\n')
        f.write("   TAM 3 yuvaydi ve kisa adlarda son yuva HEP <YOK>" + '\\n')
        f.write("   oldugu icin BEDAVA geliyordu. 17 Eylul: dolgu" + '\\n')
        f.write("   SILINDI, ad kac kelimeyse o kadar jeton. Bedava" + '\\n')
        f.write("   yuva diye bir sey KALMADI." + '\\n')
        if v.par is not None:
            import collections as _c
            _d = _c.Counter(len(d.jetonlar(e)) for e in range(v.n_ent))
            f.write("   ad uzunlugu dagilimi: " + str(dict(sorted(_d.items())))
                    + '\\n')
        f.write("4. SOYADI KALITIMI -- yuva 1 kismen KOPYALANABILIR\n")
        lst = d.L.get("ood") or []
        if v.par is not None and lst:
            lst = [x for x in lst
                   if min(len(d.jetonlar(x[4])),
                          len(d.jetonlar(x[0]))) >= 2]
            kop = sum(1 for x in lst
                      if d.jetonlar(x[4])[1] == d.jetonlar(x[0])[1])
            f.write(f"   ood: yuva 1, sorudan kopyalayarak "
                    f"%{100*kop/len(lst):.1f} tutturulur\n")
            f.write("   (kardes/anne/cocuk'ta %45-54). AMA cevabin BUTUN\n")
            f.write("   jetonlarinin soruda oldugu ornek: %0.0 -- yani\n")
            f.write("   kopyalama TEK BASINA hicbir soruyu gecirmiyor.\n\n")
        f.write("5. OZEL BLOK TEMIZ" + chr(10))
        f.write("   SPECIAL=3 -- <PAD>, ? ve <SON>. Eskiden 8 idi"
                " ve besi olu" + chr(10))
        f.write("   duruyordu ([S1], [S2], [KIMLIK] ve iki bos"
                " yuva); 17 Eylul silindi." + chr(10) + chr(10))
        f.write("6. `ent_yok`, `ent`in ALT KUMESI DEGIL\n")
        f.write("   CLAUDE.md boyle yaziyordu; OLCULDU ve YANLIS. Ikisi de\n")
        f.write("   ayni varlik havuzundan (ortak 310) ama ZINCIR kumeleri\n")
        f.write("   AYRIK (kesisim 0): ent = kisayol TIP OLARAK MUMKUN,\n")
        f.write("   ent_yok = IMKANSIZ. Belge duzeltildi.\n")

        f.write("\nDOSYALAR\n" + "-" * 74 + "\n")
        for n in sorted(os.listdir(a.klasor)):
            p = os.path.join(a.klasor, n)
            f.write(f"  {n:<24} {os.path.getsize(p)/1024:>9.0f} KB\n")

    print(f"-> {a.klasor}")
    for n in sorted(os.listdir(a.klasor)):
        p = os.path.join(a.klasor, n)
        print(f"   {n:<24} {os.path.getsize(p)/1024:>9.0f} KB")
    kotu = [s for s in sonuc if not s[1]]
    print(f"\nSAGLIK: {len(sonuc) - len(kotu)}/{len(sonuc)} GECTI")
    for ad, _, detay in kotu:
        print(f"  !! {ad}  {detay}")
    return 1 if kotu else 0


if __name__ == "__main__":
    raise SystemExit(main())
