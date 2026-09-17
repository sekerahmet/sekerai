# -*- coding: utf-8 -*-
"""konus_06 — model_06 ile DUZ KONUSMA. Sablon YOK, denetim YOK.

Kullanici karari, 17 Eylul 2026: *"model benim kendimin test edebilmesi
icin yerelde bir py yaz. calistirinca cikan ekranda ben kendim sorumu
sorayim o cevap versin, sablon vs istemiyorum. sadece kendim test edip
gozlerimle gormek istiyorum."*

`sor_06.py` NEDEN YETMEDI -- ikisi AYRI SEY olcuyor:

    sor_06    soruyu COZUMLER (varlik + iliski), diziyi `kodla_1hop` ile
              KENDI KURAR ve modele yalniz CEVAP YUVALARINI sordurur.
              Yani cumlenin govdesini ARAC yaziyor, model bosluk
              dolduruyor. Yaninda graf'taki dogru cevabi, bolmeyi ve
              kopruyu basar. Bu bir DENETIM araci ve oyle kalmali.

    konus_06  yazdigin cumleyi JETONLAYIP modele verir, gerisini MODEL
              YAZAR: butun sozluk uzerinde argmax, nokta gelene kadar.
              Sablon yok -- cevabin kac kelime olacagini, kesme
              isaretinin nereye gelecegini, cumlenin nerede bitecegini
              model soyler. Arac dogru cevabi BILMEZ ve "dogru/yanlis"
              DEMEZ. Model ne yazdiysa o basilir.

!! BU BIR OLCU DEGIL. Elle secilmis sorulardir; hukum `pencere_06` ile
verilir (onkayit `belge/onkayit/model_06.md`).

    python konus_06.py                      en son anlik goruntu
    python konus_06.py --adim 8000          belli bir adim
    python konus_06.py --genislik 5         son 5'in agirlik ortalamasi
    python konus_06.py --klasor <yol>       baska bir kosu

model_06'DA SORU BICIMI YOK -- dil yalniz BILDIRIM. Yani dogal girdi
soru sozcuksuz bir ONEK, ya da BOSLUKLU bir cumle:

    > Ayse Yilmaz'in annesi
    Ayse Yilmaz'in annesi Fatma Yilmaz'dir.

    > Fatma _ 'in annesi Ayse Yilmaz'dir.
      Fatma Yilmaz'in annesi Ayse Yilmaz'dir.
      bosluga gelen: Yilmaz

Ikinci bicim bu kolun DUGMESI: ayni cumlede HER parca sorulabiliyor.
Nedensel model boslugun sagini goremedigi icin cikarilan parca SONA
tasiniyor (<BOS> ... <AYIR> parca); arac bunu senin icin yapiyor.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch

_K = os.path.dirname(os.path.abspath(__file__))
if _K not in sys.path:
    sys.path.insert(0, _K)

import taban_06 as M                                          # noqa: E402
import pencere_06 as P                                        # noqa: E402
import analiz_06 as AZ                                        # noqa: E402
from model_06 import ModelSade                                # noqa: E402

# Ham kosu verisi Drive'da, yerelde bagli (CLAUDE.md "Nerede ne okunur").
KLASOR = os.path.join("G:" + os.sep, "Drive'ım", "model_06", "t0")

# Turkce harfleri ASCII'ye indirir. Kullanici "Ayse" de yazabilsin
# "Ayşe" de -- jeton adlari ASCII (par_ad), ekranda gosterilen bicim
# Turkce (veri_05.TR). Ikisi de ayni anahtara duser.
#
# !! BUYUK/KUCUK HARF KORUNUR. Kullanici, 17 Eylul: *"Fakultesi ile
# fakultesi ayri seyler, o da ilk harften ayrisiyor."* Dogru: `fakultesi`
# bir ILISKI (cins isim), `Fakultesi` bir VARLIK KELIMESI (ozel isim) ve
# ikisi AYRI jeton. Ilk surum hepsini kucuge indiriyordu ve
# "...bolumunun fakultesi hangisi?" cumlesindeki ilişkiyi VARLIK
# kelimesi sanip yanlis jeton basiyordu -- 1800 egitim satirinin
# 409'unda (olculdu, scratchpad/dene_konus.py).
_KAT = str.maketrans("çğıöşüâîû"
                     "ÇĞİIÖŞÜ",
                     "cgiosuaiuCGIIOSU")


def kat(s: str) -> str:
    """Diyakritikleri duser, HARF BUYUKLUGUNU KORUR."""
    return s.translate(_KAT)


# --- SOZLUK: yazilan kelime -> jeton ------------------------------------
class Sozluk:
    """Serbest metni jetonlara cevirir.

    TEK KAYNAK: butun tablolar `v`den ve veri modulunden turetilir,
    hicbir kelime burada ELLE yazili degil. Bir iliski eklenirse bu
    dosya degismez.
    """

    def __init__(self, v, VM):
        self.v, self.VM = v, VM
        lo = v.yuva_ara[0][0]

        def ekle(d, anahtar, jeton, nere):
            # ANAHTAR KELIME DIZISI (tuple), tek kelime de olsa. Cunku
            # ALTI jetonun ekrandaki hali BOSLUKLU: "ön koşulu",
            # "İç Anadolu", "Mimar Sinan", "Recep Bey", "Doğu Anadolu",
            # "Güneydoğu Anadolu". Bunlar okundugu gibi yazilinca iki-uc
            # kelime gorunur ama MODEL ICIN TEK JETONdur; anahtar tek
            # kelime olsaydi arac onlari hic bulamaz, "bilmiyorum" derdi.
            a = tuple(kat(w).lower() for w in anahtar.split())
            if a in d and d[a] != jeton:
                self.cakisma.append((a, nere))
            d[a] = jeton
            self.en_uzun = max(self.en_uzun, len(a))

        self.cakisma = []
        self.en_uzun = 1
        # VARLIK KELIMELERI. Sozluk yuvalar arasinda PAYLASILAN
        # (v.paylasilan) -- yani "Yilmaz" hangi yuvada olursa olsun ayni
        # jeton. Paylasilmazsa cozumleme konuma bagli olurdu; assert ile
        # duruyoruz, sessizce yanlis jeton uretmektense.
        assert v.paylasilan, "konus_06 PAYLASILAN sozluk bekler"
        self.kel = {}
        for i, w in enumerate(v.par_ad[0]):
            ekle(self.kel, w, lo + i, "varlik")
            if w in VM.TR:
                ekle(self.kel, VM.TR[w], lo + i, "varlik(TR)")
        # ILISKILER
        self.rel = {}
        for i, r in enumerate(VM.ILISKI):
            ekle(self.rel, r, M.REL_OFF + i, "iliski")
            if r in VM.TR_ILISKI:
                ekle(self.rel, VM.TR_ILISKI[r], M.REL_OFF + i, "iliski(TR)")
        # SORU SOZCUKLERI
        self.soru = {}
        for i, s in enumerate(v.soru_ad):
            ekle(self.soru, s, v.soru0 + i, "soru")
        assert not self.cakisma, f"yazim CAKISIYOR: {self.cakisma[:5]}"

        # EK ALLOMORFLARI. Katlanmis hallerinde CAKISIYORLAR ('in ile 'in,
        # 'dir ile 'dir -- fark yalniz noktasiz i'de), o yuzden burada tek
        # jeton degil LISTE tutulur ve dogrusu ONCEKI KELIMEDEN secilir
        # (unlu uyumu; `taban_05._nin` / `_dir` ile AYNI tablo).
        self.nin, self.dir = {}, {}
        for i, e in enumerate(v.ek_nin_ad):
            self.nin.setdefault(kat(e).lower(), []).append(v.nin0 + i)
        for i, e in enumerate(v.ek_dir_ad):
            self.dir.setdefault(kat(e).lower(), []).append(v.dir0 + i)
        self.nin_tam = {e: v.nin0 + i for i, e in enumerate(v.ek_nin_ad)}
        self.dir_tam = {e: v.dir0 + i for i, e in enumerate(v.ek_dir_ad)}
        # ad (kelime dizisi) -> varlik id.  Kesme isaretinden onceki adi
        # varliga baglamak icin; ek bicimi ORADAN cikiyor.
        self.ad2e = {tuple(M.kelimeler(v, e)): e for e in range(v.n_ent)}
        # SON KELIME -> ek bicimi. Unlu uyumu SON HECEYE bagli, yani ek
        # varligin KENDISINI degil adinin son kelimesini izler. Bu tablo
        # sayesinde grafta OLMAYAN bir ad da dogru eklenir ("Zebra
        # Yilmaz'in") -- unlu uyumu burada yeniden HESAPLANMIYOR, veri
        # modulunun tablosundan okunuyor, tek kaynak bozulmasin.
        self.son_nin, self.son_dir = {}, {}
        for e in range(v.n_ent):
            s = M.kelimeler(v, e)[-1]
            self.son_nin.setdefault(s, v.nin0 + int(v.nin_ent[e]))
            self.son_dir.setdefault(s, v.dir0 + int(v.dir_ent[e]))

    # --- cozumleme ------------------------------------------------------
    def jetonla(self, metin):
        """Doner: (jetonlar, bilinmeyen kelimeler, notlar).

        Kelimeyi PARCALAMAZ. Sozlukte olmayan bir kelime UYDURULMAZ --
        adiyla geri bildirilir. (Bu projede bir kez daha arac kusuru
        modelin hatasi sanilmisti; sessiz duzeltme yapmiyoruz.)
        """
        v = self.v
        # !! VIRGUL ve ALT CIZGI de ayri jeton.
        #   ,  devrik cumlede OGE SINIRI (model_06'nin yeni jetonu).
        #      Onsuz "Yilmaz," tek kelime sayilip sozlukte bulunamazdi.
        #   _  KULLANICININ BOSLUGU. Dilde <BOS> diye yaziliyor ama
        #      klavyeden "_" yazmak kolay. Kullanici, 17 Eylul:
        #      *"Fatma ..... 'in annesi Ayse Yilmaz'dir gibi"*.
        parca = re.findall(r"[^\s'?.,_]+|['?.,]|_+", metin)
        jet, bilinmeyen, not_ = [], [], []
        ad = []                       # kesme isaretine kadar okunan ad
        i = 0
        while i < len(parca):
            p = parca[i]
            i += 1
            if p == "?":
                jet.append(M.QM); ad = []; continue
            if p == ".":
                jet.append(M.EOS); ad = []; continue
            if p == "'":
                jet.append(v.ek0); continue
            if p == ",":
                jet.append(v.virgul); ad = []; continue
            if set(p) == {"_"}:
                # BOSLUK: "buraya ne gelir?" Dizi kurulurken sona
                # <AYIR> eklenecek ve model parcayi ORADA yazacak.
                jet.append(v.bosluk); ad = []; continue
            # KESME ISARETINDEN SONRA gelen ek: 'in / 'dir
            if jet and jet[-1] == v.ek0:
                t = self._ek(p, kat(p).lower(), ad, not_)
                if t is None:
                    bilinmeyen.append(p)
                else:
                    jet.append(t)
                ad = []
                continue
            # EN UZUN ESLESME. "ön koşulu" iki kelime GORUNUR ama TEK
            # jetondur; once ucluye, sonra ikiliye, sonra tekile bakilir.
            # Kisa eslesme once denenirse "ön" bulunamaz, "koşulu" da
            # baska bir iliskiye kayardi.
            t, alinan = None, 1
            for n in range(min(self.en_uzun, len(parca) - i + 1), 0, -1):
                oge = parca[i - 1:i - 1 + n]
                if any(x in ("'", "?", ".") for x in oge):
                    continue
                t = self._bul(tuple(kat(x) for x in oge))
                if t is not None:
                    alinan = n
                    break
            if t is None:
                ad = []
                bilinmeyen.append(p)
                continue
            i += alinan - 1
            jet.extend(t)
            # ADI BIRIKTIR: kesme isaretinden onceki varlik, ek bicimini
            # (unlu uyumu) oradan secmek icin lazim.
            lo = v.yuva_ara[0][0]
            if len(t) == 1 and lo <= t[0] < lo + len(v.par_ad[0]):
                ad.append(v.par_ad[0][t[0] - lo])
            else:
                ad = []
        return jet, bilinmeyen, not_

    def _bul(self, f):
        """Kelime DIZISININ jeton(lar)i. Doner: liste ya da None.

        SIRAYI HARF BUYUKLUGU BELIRLER. `Fakultesi` bir varlik kelimesi,
        `fakultesi` bir iliski -- katlanmis hallerinde AYNI dizge, ayrimi
        tasiyan tek sey bu (kullanici, 17 Eylul: "o da ilk harften
        ayrisiyor"). Once dogru havuza bakilir; bulunamazsa digerine de
        bakilir, cunku `oku()` CUMLE BASINI buyutuyor ("Hangisi Asli
        Arslan'in..."), yani buyuk harfli bir iliski ya da soru sozcugu
        de gelebilir.
        """
        fl = tuple(x.lower() for x in f)
        ozel = f[0][:1].isupper()
        havuz = ((self.kel, self.rel, self.soru) if ozel
                 else (self.rel, self.soru, self.kel))
        for d in havuz:
            if fl in d:
                return [d[fl]]
        # BITISIK YAZILANLAR -- kelime ikiye AYRILMAK zorunda, cunku
        # dilde bu ikisinin arasinda kesme isareti YOK:
        #   "annesinin"  ->  annesi + nin   (`kodla_2hop`: r1, nin, r2)
        #   "kimdir"     ->  kim + dir      (kimlik satiri)
        #   "annesidir"  ->  annesi + dir   (model_06 bicim 2: YUKLEM
        #                                      ONE ALINMIS bildirim)
        # !! Bu dal YOKTU ve `--dene` yakaladi: biçim 2'nin 600 satirinin
        # 600'u cozumlenemiyordu ("Annesidir" BILINMEYEN diye donuyordu).
        return (self._bitisik(fl, self.rel, self.nin,
                              self.v.nin_rel, self.v.nin0, M.REL_OFF)
                or self._bitisik(fl, self.rel, self.dir,
                                 self.v.dir_rel, self.v.dir0, M.REL_OFF)
                or self._bitisik(fl, self.soru, self.dir,
                                 self.v.dir_soru, self.v.dir0, self.v.soru0))

    def _ek(self, p, f, ad, not_):
        """Kesmeden sonraki ek. Yazildigi gibi varsa AYNEN alinir; Turkce
        harf katlandigi icin belirsizse (in/ın) ONCEKI ADIN unlu
        uyumundan secilir."""
        v = self.v
        if p in self.nin_tam:
            return self.nin_tam[p]
        if p in self.dir_tam:
            return self.dir_tam[p]
        aday = self.nin.get(f) or self.dir.get(f)
        if not aday:
            return None
        if len(aday) == 1:
            return aday[0]
        # ONCE varligin kendi bicimi, yoksa ADIN SON KELIMESININKI.
        # Ikisi de ayni tabloya bakar; ikincisi grafta olmayan adlar icin
        # ("Zebra Yilmaz'in") calisir.
        tablo = self.nin if f in self.nin else self.dir
        e = self.ad2e.get(tuple(ad))
        if e is not None:
            dogru = (v.nin0 + int(v.nin_ent[e]) if tablo is self.nin
                     else v.dir0 + int(v.dir_ent[e]))
        else:
            kaynak = self.son_nin if tablo is self.nin else self.son_dir
            dogru = kaynak.get(ad[-1]) if ad else None
        if dogru is None:
            not_.append(f"'{p}' belirsiz, ilk bicim alindi")
            return aday[0]
        if dogru not in aday:              # yazim gercekten baska bir ek
            return aday[0]
        return dogru

    @staticmethod
    def _bitisik(fl, govde_d, ek_d, tablo, ek0, taban):
        """"annesinin" -> [annesi, nin];  "ön koşulunun" -> [onkosulu, nun].

        Govde sozlukte olmali, SON kelimenin kalani da bir ek. Ek bicimi
        YAZILANDAN DEGIL tablodan gelir -- unlu uyumu zaten govdeye
        bagli, iki kaynak olmasin.
        """
        for g, t in govde_d.items():
            if len(g) != len(fl) or g[:-1] != fl[:-1]:
                continue
            son, gson = fl[-1], g[-1]
            if son.startswith(gson) and len(son) > len(gson) \
                    and son[len(gson):] in ek_d:
                return [t, ek0 + int(tablo[t - taban])]
        return None


# --- MODEL ---------------------------------------------------------------
def kur(klasor, genislik=1, adim=None):
    ayar = P.ayar_oku(klasor)
    v = M.veri_kur(ayar, yaz=lambda *a: None)
    snap = P.anlik_goruntuler(klasor)
    if adim is not None:
        assert adim in snap, (f"adim {adim} yok. Var olanlar: "
                              f"{list(snap)[-10:]}")
        sec = [adim]
    else:
        sec = list(snap)[-min(genislik, len(snap)):]
    sd = P.agirlik_ortalamasi([snap[x] for x in sec])
    net = ModelSade(ayar, v.vocab).to(M.DEV)
    net.load_state_dict(sd)
    net.eval()
    return ayar, v, net, sec


@torch.no_grad()
def olasilik(v, net, jet, k=8):
    """Bir SONRAKI jetonun dagilimi -- `/o` komutu.

    Kullanici, 17 Eylul: *"Furkan'dan sonra Kaya da gelebilir Yilmaz da
    gelebilir... ama hep Demir'i seciyor."* Haklı, ve model de oyle
    diyor: ARGMAX gizliyordu. Olculdu (adim 20000, "Furkan" onekiyle):
    grafta 8 "Furkan X" varligi var, modelin gecerli sekiz soyada
    verdigi toplam kutle %99,1 ve entropi 2,073 nat -- 8 uzerinde
    duzgun dagilimin (2,079) neredeyse tamami. Tepedeki %15,4, ikinci
    %14,5. Yani karar tek basina yorumlanamaz; dagilim yorumlanir.
    """
    satir = np.zeros((1, v.t_len), dtype=np.int64)
    satir[0, :len(jet)] = jet
    lg = net(torch.from_numpy(satir).to(M.DEV)).float()
    p = torch.softmax(lg[0, len(jet) - 1], -1).cpu().numpy()
    sira = np.argsort(-p)[:k]
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return [(int(t), float(p[t])) for t in sira], ent


@torch.no_grad()
def devam(v, net, jet, ornekle=False, isi=1.0, tohum=None):
    """Diziyi MODELE YAZDIRIR. Kisit YOK: argmax butun sozluk uzerinde.

    Sozlesme `dogruluk()` ile AYNI: satir t_len genisliginde verilir,
    p konumundaki logit p+1'inci jetonu tahmin eder. Uretilen jeton
    satira geri yazilir (ozyineli cozum), yani model kendi yazdigini
    okur -- sor_06'te bunun neden onemli oldugu olculmustu.
    """
    # ARGMAX mi ORNEKLEME mi -- ve NEDEN secenek:
    # Kullanici, 17 Eylul: *"Furkan yazinca niye Demir ile devam
    # ediyor?"* Cunku argmax farkin buyuklugune bakmaz. Olculdu (adim
    # 20000, onek "Furkan"): Demir %15,40  Sahin %14,53  Kaya %14,32 --
    # 0,87 puanlik bir fark ciktinin %100'unu belirliyor. Model "Demir"
    # demiyor, "sekizinden biri" diyor. Ornekleme dagilimi OLDUGU GIBI
    # gosterir; hukum yine `pencere_05`in (o ARGMAX olcer, olcum
    # tekrarlanabilir olsun diye).
    rs = np.random.default_rng(tohum)
    x = list(jet)
    while len(x) < v.t_len:
        satir = np.zeros((1, v.t_len), dtype=np.int64)
        satir[0, :len(x)] = x
        lg = net(torch.from_numpy(satir).to(M.DEV)).float()
        z = lg[0, len(x) - 1]
        if ornekle:
            p = torch.softmax(z / max(isi, 1e-6), -1).cpu().numpy()
            t = int(rs.choice(len(p), p=p / p.sum()))
        else:
            t = int(z.argmax())
        x.append(t)
        if t in (M.EOS, M.PAD):
            break
    return x


def dene(D, S, n=300):
    """JETONLAYICI SINAMASI -- `python konus_06.py --dene`.

    Olcut: bir EGITIM satirini `oku()` ile okunabilir Turkce'ye cevirip
    elle yazilmis gibi geri jetonlarsak AYNI jetonlar cikmali. Cikmazsa
    arac modele egitimde HIC GORMEDIGI bir dizi veriyor demektir ve
    ekranda "model bilemedi" diye okunacak sey ARACIN kusuru olur.

    Kilit testine (`test_06.py`) KONMADI: o test kosudan once calisan
    onkayitli 155 denetim, sayisi koldan kola kiyaslaniyor. Bu arac
    hukumde kullanilmiyor, sinamasi da kendi yaninda durur.

    IKI KUSURU BOYLE YAKALADI (17 Eylul, 1800 satirda):
        409 satir  `Fakultesi` (varlik) ile `fakultesi` (iliski) ayni
                   kucuk harfe indiriliyordu -- ayrimi ILK HARF tasiyor
        148 satir  "ön koşulu" / "İç Anadolu" ekranda BOSLUKLU gorunur
                   ama TEK jetondur; kelime kelime aranince bulunamiyordu
    """
    import random
    rs = random.Random(7)
    v, kotu, hep = D.v, 0, 0
    for etiket, lst, kodla in (("1hop", v.one, M.kodla_1hop),
                               ("2hop", v.tr2, M.kodla_2hop)):
        for bicim in range(3):
            k = 0
            for x in rs.sample(list(lst), min(n, len(lst))):
                dizi = [int(t) for t in kodla(v, [x], bicim)[0][0]
                        if int(t) != M.PAD]
                jet, bil, _ = S.jetonla(D.oku(dizi))
                hep += 1
                if jet != dizi:
                    k += 1
                    if k == 1:
                        print(f"  !! {etiket} bicim {bicim}")
                        print(f"     metin : {D.oku(dizi)}")
                        print("     bekl  : "
                              + " ".join(D.jeton_ad(t) for t in dizi))
                        print("     cikan : "
                              + " ".join(D.jeton_ad(t) for t in jet)
                              + (f"   BILINMEYEN {bil}" if bil else ""))
            print(f"  {etiket} bicim {bicim}: {min(n, len(lst)) - k}"
                  f"/{min(n, len(lst))} birebir")
            kotu += k
    print(f"\n  TOPLAM {hep - kotu}/{hep} birebir"
          + ("   GECTI" if kotu == 0 else f"   {kotu} BOZUK"))
    return 1 if kotu else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dene", action="store_true",
                    help="jetonlayiciyi egitim satirlarina karsi sina")
    ap.add_argument("--klasor", default=KLASOR)
    ap.add_argument("--genislik", type=int, default=1,
                    help="son N anlik goruntunun agirlik ortalamasi")
    ap.add_argument("--adim", type=int, default=None)
    ap.add_argument("--soru", action="append", default=None)
    a = ap.parse_args()

    # COZUCU: `analiz_05.Dok` jeton dizisini okunabilir Turkce'ye cevirir
    # (`oku`). Kendi kopyasini cikarmak yerine O kullaniliyor -- yazim
    # kurali (ekler onceki kelimeye yapisir) TEK yerde dursun.
    D = AZ.Dok("model_06")
    if a.dene:
        return dene(D, Sozluk(D.v, D.VM))

    ayar, v, net, sec = kur(a.klasor, a.genislik, a.adim)
    assert (D.v.vocab, D.v.n_ent, D.v.ek0) == (v.vocab, v.n_ent, v.ek0), (
        "depodaki model_06.AYAR ile KOSUNUN ayari ayristi -- sozluk tutmuyor")
    D.v = v                      # kosunun kendi verisi okunsun
    S = Sozluk(v, D.VM)

    print(f"model_06 t{ayar.tohum}   anlik goruntu "
          + (f"{sec[0]}" if len(sec) == 1
             else f"{sec[0]}-{sec[-1]} ortalamasi"))
    print("soru yaz, bos satir cikar.")
    print("  /ara <parca>  varlik adi ara        /iliski  iliskiler")
    print("  _ yaz -> BOSLUK sor:  Fatma _ 'in annesi Ayse Yilmaz'dir.")
    print("  /o <cumle basi>  SONRAKI jetonun DAGILIMI    "
          "/j  jetonlari goster")
    print("  /s  ORNEKLEME ac/kapa (argmax HEP ayni cevabi verir)\n")
    jeton_goster = [False]
    ornek = [False]          # /s -- argmax yerine DAGILIMDAN cek
    # Adlari ARAMAK gerekiyor: 1087 varlik var ve olmayan bir ad
    # yazildiginda model degil ARAC susuyor. Dokumun tamami zaten
    # `veri/model_05/` altinda; bu yalniz elin altinda dursun diye.
    TR = D.VM.TR
    adlar = [(" ".join(TR.get(w, w) for w in M.kelimeler(v, e)))
             for e in range(v.n_ent)]

    def bir(q):
        if q == "/j":
            jeton_goster[0] = not jeton_goster[0]
            print("jeton gosterimi", "ACIK" if jeton_goster[0] else "KAPALI")
            return
        if q == "/s":
            ornek[0] = not ornek[0]
            print("ORNEKLEME " + ("ACIK -- ayni soru her seferinde BASKA "
                                  "cevap verebilir (dagilimdan cekiliyor)"
                                  if ornek[0] else
                                  "KAPALI -- argmax, ayni soru HEP ayni cevap"))
            return
        if q.startswith("/o"):
            # ARGMAX TEK CEVAP VERIR, model ise DAGILIM tasir. Bu komut
            # olmadan "hep ayni cevabi veriyor" diye okunan sey aslinda
            # "ikinci aday 0,9 puan geride" olabiliyor.
            jet, bil, _ = S.jetonla(q[2:].strip())
            if bil:
                print(f"  bilmiyorum: {', '.join(bil)}")
                return
            if not jet or len(jet) >= v.t_len:
                print("  kullanim:  /o <cumlenin BASI>   ornek:  /o Furkan")
                return
            ad, ent = olasilik(v, net, jet)
            for t, pr in ad:
                cb = "#" * int(round(pr * 40))
                print(f"   {pr:7.2%}  {D.jeton_ad(t):<14} {cb}")
            print(f"   entropi {ent:.3f} nat"
                  f"   (duzgun dagilim {len(ad)} aday uzerinde olsaydi"
                  f" {np.log(len(ad)):.3f})")
            return
        if q == "/iliski":
            print("  " + "  ".join(D.VM.TR_ILISKI.get(r, r)
                                   for r in D.VM.ILISKI))
            return
        if q.startswith("/ara"):
            ara = kat(q[4:].strip()).lower()
            bul = [a for a in adlar if ara in kat(a).lower()]
            print(f"  {len(bul)} ad" + (":  " + " | ".join(bul[:25])
                                        if bul else "")
                  + ("  ..." if len(bul) > 25 else ""))
            return
        jet, bilinmeyen, notlar = S.jetonla(q)
        if bilinmeyen:
            print(f"  bilmiyorum: {', '.join(bilinmeyen)}")
            return
        if not jet:
            return
        if len(jet) >= v.t_len:
            print(f"  cok uzun: {len(jet)} jeton, satir {v.t_len}")
            return
        for n in notlar:
            print(f"  ({n})")
        # BOSLUK SORULDU MU? Nedensel model boslugun sagini o
        # konumda goremez; dilde cikarilan parca SONA tasiniyor ve
        # arasina <AYIR> giriyor. Kullanici "_" yazdiysa diziyi o
        # duzene sokup modele PARCAYI yazdiriyoruz.
        _fim = v.bosluk in jet
        if _fim:
            assert jet.count(v.bosluk) == 1, "TEK bosluk sorulabilir"
            jet = jet + [v.ayir]
        cikti = devam(v, net, jet, ornekle=ornek[0])
        if jeton_goster[0]:
            print("  jeton: " + " ".join(D.jeton_ad(t) for t in jet)
                  + "  ||  " + " ".join(D.jeton_ad(t) for t in cikti[len(jet):]))
        # MODEL CUMLEYI TAMAMLIYORSA BASINI DA BAS. Yazilan sey `?` ya da
        # `.` ile bitmisse model YENI bir cumleye basliyor demektir ve
        # yalniz onu basmak dogru. Bitmemisse model SENIN cumleni
        # suruduruyor -- o durumda yalniz devami basmak okunamaz bir sey
        # uretir ve MODELIN HATASI SANILIR:
        #
        #   yazilan  "Ibrahim Yilmaz'in kardesi"
        #   model    "nin memleketi Konya'dir."      <- BILDIRIM bicimi,
        #                                              dilbilgisi KUSURSUZ
        #   ekranda  "Nin memleketi Konya'dir."      <- sacma GORUNUYOR
        #
        # (kullanici, 17 Eylul: "bu sekilde yaptigim zaman neyi yanlis
        # yapiyorum?" -- yanlis yapan kendisi degil, bu satirdi.)
        if _fim:
            # Cevap dizinin SONUNDA. Cumleyi bosluk DOLDURULMUS haliyle
            # geri yaz -- kullanici ne sordugunu ve ne geldigini yan yana
            # gorsun.
            _p = [int(t) for t in cikti[len(jet):]]
            _dolu = [t for t in jet[:-1]]
            _i = _dolu.index(v.bosluk)
            print("  " + D.oku(_dolu[:_i] + _p + _dolu[_i + 1:]))
            print("  bosluga gelen: " + D.oku(_p)
                  + f"   ({' '.join(D.jeton_ad(t) for t in _p)})")
            return
        yeni_cumle = jet[-1] in (M.QM, M.EOS)
        print(D.oku(cikti[len(jet):] if yeni_cumle else cikti))

    if a.soru:
        for q in a.soru:
            print(f"> {q}")
            bir(q)
            print()
        return 0
    while True:
        try:
            # ISTEM KIPI GOSTERIR. Kullanici, 17 Eylul: "hicbir sey
            # degismedi, rastgele bir soyad secmedi" -- `/s` yazilmamisti
            # ve ekranda argmax'ta mi ornekte mi oldugunu gosteren
            # HICBIR SEY yoktu. Yardim satirinda yazmasi yetmiyor:
            # kipi tasiyan yer, kipin kullanildigi yer olmali.
            q = input("ornek> " if ornek[0] else "argmax> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        bir(q)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
