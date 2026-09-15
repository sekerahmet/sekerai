# -*- coding: utf-8 -*-
"""model_a — TABAN.  Okul verisi, maske yok, kimlik gorevi yok.

Onceden kayit: belge/onkayit/model_a.md

Deneme 2'nin ILK kosusu. Tek kol, tez sinamiyor. Isi: kodun egittigini
gormek ve UZERINE DENEY KURULACAK ZEMINI olcmek -- doyma adimi, ENT
seviyesi, kisayol orani, en iyi pencere.

Varyasyon (model_a1 gibi) BU DOSYAYI import eder ve yalniz AYAR'in bir
alanini degistirir; mimariyi/veriyi yeniden tanimlamaz
(deneme2/ISIMLENDIRME.md). Taban olculmeden varyasyon YAZILMAZ.

--------------------------------------------------------------------------
ARSIVDEKI sifirdan.py'DEN NE DEGISTI (dordu de fiilen ariza cikarmisti)

1. AYAR ARTIK NESNE, ortam degiskeni DEGIL.
   Eskiden `MASK_KEY = os.environ.get(...)` modul seviyesindeydi ve IMPORT
   ANINDA okunuyordu; her arac import etmeden once ortami kurmak zorundaydi,
   onceki hucrelerden sizan MEM_AT assert patlatti. Burada import'un yan
   etkisi YOK: her sey `Ayar` icinde, fonksiyonlara PARAMETRE olarak gider.

2. AD DOSYADA.  Eskiden butun kollar ARMS=A ile kostu, yani G de GM de
   `snap_A_s0_*.pt` yazdi; kolu yalniz KLASOR ayiriyordu. Burada `ayar.ad`
   dosya adina giriyor: `snap_model_a_00050000.pt`.

3. ISINMA ACIK SAYI.  Eskiden `warm = max(10, STEPS // 20)` idi; kosuyu
   parcalara bolunce isinma 6000 degil 250 adim oldu ve ayni tohum baska
   yorunge izledi. Burada `ayar.isinma` dogrudan yazilir, `adim`dan
   TURETILMEZ.

4. TANI BURADA DEGIL.  Sonda/logit-lens/dikkat `tani_a.py`ye gider, olcum
   `pencere_a.py`ye. Bu dosya: ayar + veri + model + egitim. 1500 satirlik
   ic-ice yigin yok.

--------------------------------------------------------------------------
MIMARI VE OPTIMIZASYON NEREDEN GELIYOR -- IKI AYRI KAYNAK

Ikisini KASITLI olarak ayri yerlerden aldik. Gerekcesi 2603.25009'un kendi
merkezi bulgusu: "grokking dynamics are NOT primarily determined by
architecture, but by interactions between optimization stability and
regularization." Yani iki makale FARKLI seylere bakiyor.

MIMARI -- 2604.07822 "Loop, Think & Generalize" satir 594
    d=768, 12 kafa, 4 katmanlik TEKRARLI blok, batch 512, isinma 2000
    Ayni gorev (iki adimli kompozisyon, OOD) ve dongunun ise yaradigini
    olcmus. Kod: github.com/OSU-NLP-Group/Loop-Think-Generalize

OPTIMIZASYON -- 2603.25009 "A Systematic Empirical Study of Grokking" 4.1
    AdamW lr 1e-3, wd 1.0, gradyan kirpma 1.0, GELU, pre-norm, dropout yok
    Bu calisma grokking'i HIZLANDIRMAYI olcuyor ve wd'yi "dominant control
    parameter" olarak buluyor -- dar bir "Goldilocks" bandi var.

SINIR, ACIKCA: 2603.25009'un gorevi MODULAR ADDITION (mod 97), bizimki
degil. Sayilari (ozellikle wd) oradan aldik ama transfer ettigi OLCULMEDI.
Ilk kosunun isi bunu gormek.
"""
from __future__ import annotations

import dataclasses as dc
import glob, json, math, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bu dosya AILE klasorunde (deneme2/model_a/); `veri_okul.py` bir UST
# klasorde (deneme2/) cunku GOREVI tanimlar, modele ait degil -- model_a da
# model_b de ayni veriyi gorur. Ust klasoru yola eklemek TEK import yan
# etkisidir ve deterministiktir: ortam degiskeni okumuyor, ayar tasimiyor
# (sifirdan.py'nin arizasi oydu, bkz. ISIMLENDIRME.md).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import veri_okul as VO

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Ozel token'lar. ENT_OFF/VOCAB veriden TURER (Veri.__post_init__), burada
# sabit YAZILMAZ -- arsivde `ENT_OFF` bir kez elle 14 yazilmis, 16'ymis.
PAD, Q1, Q2, QM, EOS, IDENT = 0, 1, 2, 3, 4, 5
SPECIAL = 8
REL_OFF = SPECIAL
T_LEN = 8


# ======================= AYAR ============================================
@dc.dataclass(frozen=True)
class Ayar:
    ad: str = "model_a"

    # --- veri (bir ailenin butun kollarinda AYNI olmali, yoksa
    #     'sartlar esit' bozulur ve kollar farkli veri gorur)
    veri_tohum: int = 0
    ent_pay: float = 0.20      # varliklarin ne kadari ENT'e ayrilir
    comp_pay: float = 0.10     # zincirlerin ne kadari COMP'a ayrilir
    arama_pay: float = 0.25    # ENT'in ne kadari ARAMA'ya (HUKUMDEN AYRIK)

    # --- mimari
    #   MIMARI: 2604.07822 "Loop, Think & Generalize" satir 594, BIREBIR:
    #   "we use an embedding dimension of 768, 12 attention heads and a
    #    recurrent block of 4 transformer layers"
    #   Kafadan atilmadi; yayimlanmis ve ayni gorevde (2-hop OOD) calismis
    #   bir konfigurasyon. Kodu: github.com/OSU-NLP-Group/Loop-Think-Generalize
    d: int = 256               # 768 idi (Loop&Generalize). KUCULTULDU.
    #   Gerekce KESE DEGIL, 2603.25009'un merkezi bulgusu: "grokking dynamics
    #   are NOT primarily determined by architecture". O calismanin kendi
    #   transformer'i d=512, DERINLIK 1. Genislik ikinci derece bir etken
    #   olarak okundu; birinci derece olan sey (wd, lr) literaturden alindi.
    #   Bu bir CIKARIM. 768 bir dugme olarak duruyor, pahali degil (~54 dk).
    l: int = 4                 # BLOK sayisi (paylasilan agirlik)
    nh: int = 4                # 256/4 = 64 per kafa -- Loop&Generalize'in
    #                            768/12 = 64'uyle AYNI kafa boyutu.
    dff: int = 1024            # 4*d. VARSAYIM DEGIL: 2603.25009 4.1
    #                            "a feedforward dimension of 4d = 2,048" diyor.
    dongu: int = 2             # ayni bloklar kac kez uygulanacak (R)
    #   dongu=1  -> DUZ transformer (l katman)
    #   dongu=R  -> l*R katman-esdegeri hesap, AYNI parametrelerle
    # 2604.07822:24 birebir: "The model with R=1 is equivalent to a 4-layer
    # vanilla transformer" ve "systematic generalization in the 2-hop task
    # already emerges from WEIGHT SHARING under fixed recurrence."
    # Yani duz model ayri bir model DEGIL, bu eksenin R=1 kosesi.

    # --- egitim
    tohum: int = 0
    adim: int = 20000          # ILK SINIR (CLAUDE.md kural 1), tavan DEGIL.
    #   Uzatmanin yazili bir ust siniri YOK; uzatmak KULLANICI kararidir
    #   ve UZATMA = SURDURME: `kos.py --adim <yeni> --surdur`.
    batch: int = 512
    lr: float = 1e-3           # 2603.25009 Tablo 1, AdamW standardi.
    #   1e-4 idi (Loop&Generalize'dan). Ama o calisma grokking'i HIZLANDIRMAYI
    #   hedeflemiyordu; bu calisma tam onu olcuyor ve AdamW icin 1e-3 kullaniyor.
    wd: float = 0.1            # KULLANICI KARARI. KRITIK DUGME.
    #   NOT: arsivdeki butun kosular da 0.1 kullaniyordu (grok_uzun preset,
    #   G/GM'nin kayitli cfg'si). O ayarla 120.000 adim kosuldu ve ent
    #   0.14'te kaldi -- ama model DUZ idi. Burada tek fark DONGU.
    #   2603.25009'un taramasi (modular addition, AdamW):
    #   Onceki deger 0.01 idi ve o calismanin taramasinda 0.01 = "no seed
    #   grokks within 400,000 steps". Yani hicbir sey gormeyecegimiz deger.
    #     lambda 0.01  ->  hic grokking YOK
    #     lambda 1.0   ->  3/3 tohum, gecikme 44.000 adim
    #     lambda 5.0   ->  3/3 tohum, gecikme 24.000 adim   (optimal)
    #   0.1 bu taramanin ALTINDA kaliyor (0.01 ile 1.0 arasi, olculmemis).
    #   SINIR: o tarama MODULAR ADDITION'da yapildi, bizim gorevde degil.
    #   Yani "0.1 yanlis" DIYEMEM; "olculmemis aralikta" diyebilirim.
    betas: tuple = (0.9, 0.999)   # AdamW momentum katsayilari.
    #   Hicbir makale YAZMIYOR. Yazilmamis olmasi "torch varsayilani" demek
    #   olarak okundu -> (0.9, 0.999). Arsivde (0.9, 0.95) idi (GPT tarzi) ve
    #   `egit()` icine GOMULUYDU -- yani ayar.json'a bile girmiyordu.
    #   Artik alan: gorunur, kaydediliyor, degistirilebilir. CIKARIM, olcum degil.
    isinma: int = 2000         # ACIK -- `adim`dan turetilmez.
    #   2604.07822:594 "linear warmup schedule of 2000 steps" -- Wang da 2000.
    #   6000 idi (arsivden, 120000//20). 20.000 adimlik kosuda %30 ederdi.
    sabit_lr: bool = True      # True: isinmadan sonra LR SABIT (grokking icin)
    olc_her: int = 2000        # 20.000/2000 = 10 olcum noktasi (kullanici).
    n_olcum_max: int = 3000    # her olcme kumesinden en fazla

    # --- kimlik gorevi (model_a1 / model_a2 bunu degistirir)
    ident_frac: float = 0.0
    ident_kip: str = ""        # "" | "q2son" | "q1"

    # --- maske (bu deneyde kapali; aile ilerde kullanabilsin diye duruyor)
    mask_poz: int | None = None
    mask_blok: tuple = ()

    def degistir(self, **kw) -> "Ayar":
        bilinmeyen = set(kw) - {f.name for f in dc.fields(self)}
        assert not bilinmeyen, f"Ayar'da boyle alan yok: {bilinmeyen}"
        return dc.replace(self, **kw)

    def sozluk(self) -> dict:
        return dc.asdict(self)

    def fark(self, other: "Ayar") -> dict:
        a, b = self.sozluk(), other.sozluk()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k] and k != "ad"}


def fark_bas(a: Ayar, b: Ayar, yaz=print) -> dict:
    """'Tek fark su' bir IDDIA degil, CIKTI olsun (ISIMLENDIRME.md b).

    Kol C tam bunun yoklugundan gecersiz kaldi: tek okuma noktasiyla
    egitilmis paketten uc okuma noktasiyla surduruldu, hicbir sey hata
    vermedi, kimse fark etmedi."""
    d = a.fark(b)
    yaz(f"  {a.ad}  vs  {b.ad}")
    if not d:
        yaz("     FARK YOK -- ayni ayar. Kasitli mi?")
    for k, (x, y) in sorted(d.items()):
        yaz(f"     FARKLI : {k:<12} {x!r} -> {y!r}")
    yaz(f"     AYNI   : {len(a.sozluk()) - len(d) - 1} alan")
    return d


# ======================= VERI ============================================
@dc.dataclass
class Veri:
    facts: np.ndarray          # (n_ent, n_rel)  -1 = olgu YOK
    one: list                  # (e, r, hedef)              1hop
    tr2: list                  # (e, r1, r2, kopru, cevap)  egitim 2hop
    comp: list                 #  ayni   -- egitimde GORULMEMIS (e,r1,r2) UCLUSU
    #   !! DIKKAT (15 Eylul hakemligi): buraya ve CLAUDE.md'ye "gorulmemis
    #   r1-r2 CIFTI" yazilmisti. OLCULDU, YANLIS: comp'un 4281 orneginin
    #   4281'inin (r1,r2) cifti egitimde de var (115 ciftin 115'i). Bolme
    #   kodu ZINCIR tutuyor, CIFT tutmuyor. Yani comp "bu iliski ciftini hic
    #   gormedi" demek DEGIL, "bu varligin bu ciftle zincirini gormedi"
    #   demek -- cok daha zayif bir genelleme sinavi. Kod degismedi
    #   (arsivdeki build_data_dis ile ayni), ETIKET duzeltildi.
    #   ILISKI-CIFTI genellemesini olcen bir bolme HENUZ YOK.
    ent: list                  #  ayni   -- varlik hic zincir basi olmamis (HUKUM)
    ent_yok: list              #  ayni   -- kisayol TIP OLARAK imkansiz
    ent_arama: list            #  ayni   -- maske aramasi icin, HUKUMDEN AYRIK
    n_ent: int = 0
    n_rel: int = 0
    # TIP: her varligin tipi (KISI/OKUL/SEHIR/DERS). TANI icin gerekli --
    # "model YANLIS cevap verirken hic olmazsa DOGRU TIPTE bir sey mi
    # soyluyor?" sorusu bunsuz sorulamaz. `veri_kur` doldurur; tani_a
    # kendi basina TURETMEZ (turetirse iki dosya ayri siralama kurar).
    tip: np.ndarray | None = None      # (n_ent,) tip indeksi
    tip_ad: tuple = ()                 # tip indeksi -> ad

    def __post_init__(self):
        self.n_ent, self.n_rel = self.facts.shape
        self.ent_off = SPECIAL + self.n_rel
        self.vocab = self.ent_off + self.n_ent
        # phi: TURETILMIS TANI SAYISI, kontrol parametresi DEGIL. Ayarlanamaz;
        # graf yogunlugundan ve ent_pay/comp_pay'den duser. "phi'yi 7 yapalim"
        # denemez -- veri ureticisi degistirilir.
        self.phi = len(self.tr2) / max(1, len(self.one))
        # WANG'IN TANIMI AYNI DEGIL (15 Eylul hakemligi). Wang 2405.15071:155
        # "phi = |train_inferredID| / |atomicID|" ve atomicID, OOD varliklarinin
        # olgularini DISLAR. Bizim paydamiz TUM olgular. Olculdu: ayni veride
        # bizimki 5.09, Wang tanimiyla 6.36 -- %25 fark. Wang'in 3.6-18.0
        # taramasina konumlanirken WANG_PHI kullanilmali, phi degil.
        _ent = {e for e, *_ in self.ent} | {e for e, *_ in self.ent_yok} \
            | {e for e, *_ in self.ent_arama}
        _id = sum(1 for e, _, _ in self.one if e not in _ent)
        self.wang_phi = len(self.tr2) / max(1, _id)


def veri_kur(ayar: Ayar, yaz=print) -> Veri:
    """Okul grafi -> bolmeler.  Arsivdeki build_data_dis ile AYNI mantik.

    Tohum `ayar.veri_tohum`; butun kollarda AYNI olmali, yoksa kollar farkli
    veri gorur ve 'sartlar esit' bozulur."""
    # KUSUR (15 Eylul hakemligi): burada `VO.kur()` yaziyordu, yani graf
    # HER ZAMAN tohum 0 ile uretiliyordu. `ayar.veri_tohum` yalniz BOLMEYI
    # etkiliyordu. Olculdu: veri_tohum 0 ve 1 ayni `facts`, farkli `tr2`.
    # Yani "veri tohumunu degistirdim" diyen biri grafin degismedigini
    # FARK ETMEZDI.
    G = VO.kur(ayar.veri_tohum)
    zin = VO.zincirler(G)
    E = [a for t in VO.TIPLER for a in G["ad"][t]]
    # E, TIP SIRASIYLA kuruluyor -- tip dizisi AYNI comprehension'dan
    # cikarilir ki iki yerde iki siralama olmasin.
    E_tip = np.array([i for i, t in enumerate(VO.TIPLER)
                      for _ in G["ad"][t]], np.int64)
    R = list(VO.ILISKI)
    eid = {a: i for i, a in enumerate(E)}
    rid = {r: i for i, r in enumerate(R)}

    facts = np.full((len(E), len(R)), -1, np.int64)
    for (e, r), h in G["olgu"].items():
        facts[eid[e], rid[r]] = eid[h]

    bas = {}
    for x in zin:
        bas.setdefault(x[0], []).append(x)

    rng = np.random.RandomState(ayar.veri_tohum)
    ent_ad = set()
    for t in VO.TIPLER:                       # TABAKALI: tek tip secilirse
        a = [x for x in G["ad"][t] if x in bas]   # sinav o tipin karisimina
        if a:                                     # indirgenir
            k = int(round(len(a) * ayar.ent_pay))
            ent_ad |= {a[int(i)] for i in rng.permutation(len(a))[:k]}

    # ARAMA / HUKUM ayrimi VARLIK duzeyinde: ayni varligin baska bir zinciri
    # de sizinti sayilir. Olculdu: ayrim olmadan hukum kumesinin %24'u
    # aramada zaten gorulmustu.
    _ea = sorted(ent_ad)
    _k = max(1, int(round(len(_ea) * ayar.arama_pay)))
    _ix = rng.permutation(len(_ea))
    arama_ad = {_ea[int(i)] for i in _ix[:_k]}
    hukum_ad = ent_ad - arama_ad

    tr2, comp, ent_ay, ent_yk, ent_ar = [], [], [], [], []
    for e in E:                                # E sirasi SABIT -> tekrarlanabilir
        lst = bas.get(e)
        if not lst:
            continue
        if e in ent_ad:
            hedef = ent_ay if e in hukum_ad else ent_ar
            for x in lst:
                if x[6] == "AYIRT":
                    hedef.append(x)
                elif x[6] == "YOK" and e in hukum_ad:
                    ent_yk.append(x)
            continue                           # AYNI / DONUS: duser
        p = rng.permutation(len(lst))
        k = int(round(len(lst) * (1.0 - ayar.comp_pay)))
        for i, j in enumerate(p):
            x = lst[int(j)]
            if i < k:
                tr2.append(x)
            elif x[6] in ("AYIRT", "YOK"):
                comp.append(x)                 # AYNI/DONUS sinava girmez

    say = lambda L: [(eid[x[0]], rid[x[1]], rid[x[2]], eid[x[3]], eid[x[4]])
                     for x in L]
    v = Veri(facts=facts, one=[(eid[e], rid[r], eid[h])
                               for (e, r), h in G["olgu"].items()],
             tr2=say(tr2), comp=say(comp), ent=say(ent_ay),
             ent_yok=say(ent_yk), ent_arama=say(ent_ar),
             tip=E_tip, tip_ad=tuple(VO.TIPLER))

    # --- SIZINTI DENETIMI -- sessiz gecmesin
    trset = {(e, a, b) for e, a, b, _, _ in v.tr2}
    for nm, st in (("COMP", v.comp), ("ENT", v.ent), ("ENT-YOK", v.ent_yok)):
        k = sum((e, a, b) in trset for e, a, b, _, _ in st)
        assert k == 0, f"{nm} sizintisi: {k} ornek egitimde de var"
    _ent_id = {eid[a] for a in ent_ad}
    k = sum(e in _ent_id for e, *_ in v.tr2)
    assert k == 0, f"ENT varligi egitimde ZINCIR BASI olmus: {k}"
    _h = {e for e, *_ in v.ent} | {e for e, *_ in v.ent_yok}
    _a = {e for e, *_ in v.ent_arama}
    assert not (_h & _a), f"ARAMA/HUKUM varlik sizintisi: {len(_h & _a)}"

    yaz(f"  veri: olgu {len(v.one)}  egitim2 {len(v.tr2)}  COMP {len(v.comp)}  "
        f"ENT {len(v.ent)}  ENT-YOK {len(v.ent_yok)}  "
        f"ENT-ARAMA {len(v.ent_arama)}  phi {v.phi:.2f}")
    yaz(f"        n_ent {v.n_ent}  n_rel {v.n_rel}  vocab {v.vocab}  "
        f"ent_off {v.ent_off}")
    yaz(f"        phi {v.phi:.2f} (bizim tanim)   {v.wang_phi:.2f} (Wang tanimi, "
        f"payda atomicID)   -- TURETILMIS, ayar DEGIL")

    # BOLMELER NE KADAR YENI -- her kosuda BASILIR, bir daha etiket kaymasin.
    # 15 Eylul'de comp'a "gorulmemis iliski cifti" deniyordu; olculunce
    # ciftlerin TAMAMI egitimde cikti. Sayi gozukurse iddia kayamaz.
    _tc = {(a, b) for _, a, b, _, _ in v.tr2}
    _tv = {e for e, *_ in v.tr2}
    for _ad, _L in (("COMP", v.comp), ("ENT", v.ent), ("ENT-YOK", v.ent_yok)):
        if not _L:
            continue
        _yc = sum(1 for _, a, b, _, _ in _L if (a, b) not in _tc)
        _yv = sum(1 for e, *_ in _L if e not in _tv)
        yaz(f"        {_ad:<8} YENI olan: iliski-cifti {_yc}/{len(_L)}   "
            f"zincir-basi varlik {_yv}/{len(_L)}")
    yaz("        ^ COMP'ta cift YENI DEGIL: comp 'gorulmemis UCLU', "
        "'gorulmemis CIFT' DEGIL.")
    return v


# ======================= KODLAMA =========================================
def _bos(n):
    return np.zeros((n, T_LEN), np.int64)


def kodla_1hop(v: Veri, batch):
    """[Q1] e r ? cevap EOS   -> hedef pozisyon 3"""
    X = _bos(len(batch))
    for i, (e, r, a) in enumerate(batch):
        X[i, :6] = [Q1, v.ent_off + e, REL_OFF + r, QM, v.ent_off + a, EOS]
    return X, np.full(len(batch), 3, np.int64), \
        np.array([v.ent_off + a for _, _, a in batch], np.int64)


def kodla_2hop(v: Veri, batch):
    """[Q2] e r1 r2 ? cevap EOS   -> hedef pozisyon 4"""
    X = _bos(len(batch))
    for i, (e, r1, r2, _b, a) in enumerate(batch):
        X[i, :7] = [Q2, v.ent_off + e, REL_OFF + r1, REL_OFF + r2, QM,
                    v.ent_off + a, EOS]
    return X, np.full(len(batch), 4, np.int64), \
        np.array([v.ent_off + a for *_, a in batch], np.int64)


def kodla_kimlik_q1(v: Veri, ents):
    """[Q1] e IDENT ? e EOS  -> hedef = varligin KENDISI (SIFIR-HOP).

    DUZELTME (15 Eylul hakemligi): bunu daha once "ise yaramaz kontrol" diye
    anlatmistim. YANLIS. arXiv 2509.24653'un onerdigi identity bridge TAM
    BUDUR -- birebir: "a zero-hop SELF-MAPPING for each bridge token" ve
    "an identity mapping on bridge tokens". Yani e -> e.

    Bilinen itiraz (arsiv DENEY5): cevap girdide duruyor, kopyalamayla
    cozulebilir. Makale bunu bilerek yapiyor; iddiasi, gizli durumu token
    gommesiyle HIZALAMAYA zorlamasi."""
    X = _bos(len(ents))
    for i, e in enumerate(ents):
        X[i, :6] = [Q1, v.ent_off + e, IDENT, QM, v.ent_off + e, EOS]
    return X, np.full(len(ents), 3, np.int64), \
        np.array([v.ent_off + e for e in ents], np.int64)


def kodla_kimlik_q2son(v: Veri, batch):
    """[Q2] e r1 IDENT ? kopru EOS  -> hedef = facts[e,r1], yani BIRINCI HOP.

    DUZELTME (15 Eylul hakemligi): bu, makalenin identity bridge'i DEGILDIR.
    Makale SIFIR-HOP self-mapping oneriyor (e -> e; yukaridaki q1). Bu ise
    BIRINCI HOP DENETIMI -- daha guclu ve FARKLI bir mudahale.

    !! BEDELI VAR: ENT varliklari burada [Q2] cercevesinde ZINCIR BASI
    oluyor. ENT bolmesinin tanimi "varlik hic zincir basi olmamis" idi;
    q2son ile bu tanim BOZULUR. Olculdu: 8.400 kimlik orneginin 1.274'u
    (%15) bir ENT varligini zincir basi yapiyor. `egitim_havuzu` bunu her
    kosuda BASAR, sessiz gecmez.

    Cevap (kopru) girdide gecmiyor -> kopyalamayla cozulemez. kodla_2hop ile
    AYNI cerceve ve AYNI hedef pozisyonu (4)."""
    X = _bos(len(batch))
    for i, (e, r1, b) in enumerate(batch):
        X[i, :7] = [Q2, v.ent_off + e, REL_OFF + r1, IDENT, QM,
                    v.ent_off + b, EOS]
    return X, np.full(len(batch), 4, np.int64), \
        np.array([v.ent_off + b for _, _, b in batch], np.int64)


def egitim_havuzu(ayar: Ayar, v: Veri, yaz=print):
    """1hop + 2hop (+ istege bagli kimlik gorevi) -> tek havuz."""
    parca = [kodla_1hop(v, v.one), kodla_2hop(v, v.tr2)]
    kimlik = None
    if ayar.ident_frac > 0:
        assert ayar.ident_kip in ("q1", "q2son"), \
            f"ident_kip 'q1' ya da 'q2son' olmali: {ayar.ident_kip!r}"
        if ayar.ident_kip == "q1":
            kimlik = kodla_kimlik_q1(v, range(v.n_ent))
        else:
            ik = [(e, r, int(v.facts[e, r]))
                  for e in range(v.n_ent) for r in range(v.n_rel)
                  if v.facts[e, r] >= 0]        # -1 = olgu YOK, atla
            kimlik = kodla_kimlik_q2son(v, ik)
        n_tab = len(parca[0][0]) + len(parca[1][0])
        tekrar = max(1, int(round(ayar.ident_frac / max(1e-9, 1 - ayar.ident_frac)
                                  * n_tab / len(kimlik[0]))))
        parca.append(tuple(np.tile(z, (tekrar,) + (1,) * (z.ndim - 1))
                           for z in kimlik))
        yaz(f"  kimlik gorevi: {ayar.ident_kip}  {len(kimlik[0])} ornek x{tekrar}")
        # ENT TANIMI BOZULUYOR MU -- her kosuda BASILIR (15 Eylul hakemligi).
        # Mevcut sizinti assert'i yalniz `tr2`ye bakiyor; kimlik havuzu oradan
        # gecmiyor ve ENT varligini [Q2] cercevesinde zincir basi yapabiliyor.
        _ent_v = {e for e, *_ in v.ent} | {e for e, *_ in v.ent_yok}
        _bas = [e for e, _, _ in ik] if ayar.ident_kip == "q2son" else []
        _n = sum(1 for e in _bas if e in _ent_v)
        if _n:
            yaz(f"  !! DIKKAT: kimlik havuzunda {_n}/{len(_bas)} ornek bir ENT "
                f"varligini [Q2] cercevesinde ZINCIR BASI yapiyor.")
            yaz("     ENT'in tanimi ('hic zincir basi olmamis') BU KOLDA "
                "gecerli degil; taban ile ent kiyasi bunu hesaba katmali.")
    X = np.concatenate([a for a, _, _ in parca])
    P = np.concatenate([b for _, b, _ in parca])
    T = np.concatenate([c for _, _, c in parca])
    if kimlik is not None:
        yaz(f"                 havuzun %{100*len(parca[2][0])/len(X):.0f}'i")
    return X, P, T, kimlik


# ======================= MODEL ===========================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        f = x.float()
        return (self.g.float() * f
                * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)
                ).to(x.dtype)


class Blok(nn.Module):
    def __init__(self, d, nh, dff):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.po = nn.Linear(d, d, bias=False)
        self.f1 = nn.Linear(d, dff, bias=False)
        self.f2 = nn.Linear(dff, d, bias=False)
        self.mask_poz = None    # bu pozisyon bu blokta YENIDEN OKUNAMAZ

    def forward(self, x):
        B, T, D = x.shape
        q, k, val = self.qkv(self.n1(x)).chunk(3, -1)
        sh = lambda t: t.view(B, T, self.nh, self.hd).transpose(1, 2)
        if self.mask_poz is None:
            o = F.scaled_dot_product_attention(sh(q), sh(k), sh(val),
                                               is_causal=True)
        else:
            m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            m[:, self.mask_poz] = True
            m[self.mask_poz, self.mask_poz] = False      # kendine bakabilir
            o = F.scaled_dot_product_attention(sh(q), sh(k), sh(val),
                                               attn_mask=~m[None, None])
        x = x + self.po(o.transpose(1, 2).reshape(B, T, D))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class Model(nn.Module):
    def __init__(self, ayar: Ayar, vocab: int):
        super().__init__()
        self.ayar, self.vocab = ayar, vocab
        d = ayar.d
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T_LEN, d)
        self.bloklar = nn.ModuleList([Blok(d, ayar.nh, ayar.dff)
                                      for _ in range(ayar.l)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight        # bagli gomme
        for i in ayar.mask_blok:
            self.bloklar[i].mask_poz = ayar.mask_poz
        self.apply(self._ilk)

    @staticmethod
    def _ilk(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for _ in range(self.ayar.dongu):       # R kez AYNI bloklar
            for blk in self.bloklar:
                h = blk(h)
        return self.head(self.nf(h))

    def n_param(self):
        gor, tot = set(), 0
        for p in self.parameters():
            if id(p) not in gor:
                gor.add(id(p)); tot += p.numel()
        return tot


# ======================= OLCME ===========================================
def olcme_listeleri(ayar: Ayar, v: Veri):
    """Her kumeden en fazla n_olcum_max ornek. TEK KAYNAK: egitim dongusu de
    pencere_a.py de BURAYI cagirir. Arsivde bu mantik iki dosyada AYRI AYRI
    duruyordu ve bir salt degisirse farkli ornek olculurdu, sessizce."""
    def alt(lst, salt):
        if not lst:
            return []
        r = np.random.RandomState(ayar.veri_tohum + 7 + salt)
        if len(lst) > ayar.n_olcum_max:
            return [lst[i] for i in r.permutation(len(lst))[:ayar.n_olcum_max]]
        return list(lst)
    return dict(one=alt(v.one, 0), seen=alt(v.tr2, 1), comp=alt(v.comp, 2),
                ent=alt(v.ent, 3), ent_yok=alt(v.ent_yok, 5))


def olcme_izi(L: dict) -> str:
    """Olcme setlerinin PARMAK IZI -- setin KENDISINI temsil eder.

    Dis hakemlik (15 Eylul) hakli cikti. Onceki hali sadece
    `(anahtar, uzunluk, ilk 2 ornek)` hash'liyordu ve ORTADAN degisen bir
    ornegi GORMUYORDU. Olculdu: ent[1500] degistirildi, uzunluk ve ilk iki
    ornek ayni kaldi -> parmak izi BIREBIR AYNI cikti. Yani tam yakalamasi
    gereken seyi kaciriyordu.

    Simdi BAYT DUZEYINDE: her kumenin tamami int64 dizisine cevrilip
    ham baytlari hash'leniyor. `repr()`ten hem daha hizli hem tam
    deterministik (repr float/int gosterimine bagli degil).

    TEK KAYNAK: egit() de pencere_a.py de BURAYI cagirir. Ayni veri ve
    ayarla ayni izi vermeleri MEKANIK olarak dogrulanabilsin diye."""
    import hashlib
    h = hashlib.md5()
    for k in sorted(L):
        h.update(k.encode())
        arr = np.asarray(L[k], dtype=np.int64)
        h.update(repr(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:12]


@torch.no_grad()
def dogruluk(model, v: Veri, X, P, T, bs=512):
    """VARLIK-KISITLI argmax: cevap her zaman bir varliktir, ilişki/ozel
    token'lar yarismaya sokulmaz."""
    # KUSUR (15 Eylul): sonunda `model.train()` vardi -- yani olcum,
    # cagiranin kipini DEGISTIRIYORDU. pencere_a.olc() `net.eval()` deyip
    # arka arkaya olcuyor; ilk cagridan sonra model TRAIN kipine gecmis
    # oluyordu. Su an dropout/batchnorm yok, yani sonuca etkisi YOKTU --
    # ama biri dropout eklerse olcum SESSIZCE rastgelelesirdi.
    onceki = model.training
    model.eval()
    lo, hi = v.ent_off, v.ent_off + v.n_ent
    ok = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            lg = model(xb)
        idx = torch.from_numpy(P[i:i + bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        lg = lg.float()[ar, idx][:, lo:hi]
        g = torch.from_numpy(T[i:i + bs]).to(DEV) - lo
        ok.append((lg.argmax(-1) == g).cpu().numpy())
    model.train(onceki)
    return float(np.concatenate(ok).mean())


@torch.no_grad()
def kisayol_orani(model, v: Veri, lst, bs=512):
    """Model kac ornekte KISAYOL cevabini (facts[e, r2]) soyluyor?
    facts hucresi -1 ise (olgu yok) ent_off-1 cikar; bu bir ILISKI token'idir,
    hicbir varlik tahminiyle eslesmez -> oran 0.000. ENT-YOK icin dogrusu bu."""
    if not lst:
        return float("nan")
    X, P, _ = kodla_2hop(v, lst)
    ksy = np.array([v.ent_off + int(v.facts[e, r2]) for e, _, r2, _, _ in lst])
    onceki = model.training            # bkz. dogruluk()'taki ayni kusur
    model.eval()
    lo, hi = v.ent_off, v.ent_off + v.n_ent
    ok = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            lg = model(xb)
        idx = torch.from_numpy(P[i:i + bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)
        p = lg.float()[ar, idx][:, lo:hi].argmax(-1) + lo
        ok.append((p.cpu().numpy() == ksy[i:i + bs]))
    model.train(onceki)
    return float(np.concatenate(ok).mean())


# ======================= DOSYA / ORTAM ===================================
def _atomik(yol: str, yazici):
    """Once `<yol>.tmp`e yaz, sonra os.replace ile yerine koy.

    OLCULDU (15 Eylul): yarim kalmis bir anlik goruntu pencere_a'nin
    glob'una GIRIYOR (adim 4000 listeye girdi) ve orada `torch.load`
    "PytorchStreamReader failed" diye patliyor; yarim kalmis bir
    `egri.json` ise JSONDecodeError veriyor ve BUTUN egri kayboluyor.

    Iki yolla oluyordu: (a) Drive'a yazarken kosu kesilir, (b) egitim
    yazarken pencere_a AYNI ANDA okur. `.tmp` + `os.replace` ikisini de
    kapatir: okuyucu ya ESKI ya YENI dosyayi gorur, ARASINI asla.
    `.tmp` AYNI klasorde -- replace ancak ayni dosya sisteminde atomik.

    GERI DUSUS: bazi FUSE suruculeri (Drive dahil) var olan bir dosyanin
    UZERINE rename'i reddedebilir. O durumda once siler, sonra tasiriz --
    artik atomik DEGIL, ama kosu adim 4000'de cokmez. Bir kez uyarilir."""
    t = yol + ".tmp"
    yazici(t)
    try:
        os.replace(t, yol)
    except OSError as e:
        global _ATOMIK_UYARI
        if not _ATOMIK_UYARI:
            _ATOMIK_UYARI = True
            print(f"  !! os.replace calismadi ({type(e).__name__}: {e}). "
                  "Sil-sonra-tasi'ya dusuluyor:")
            print("     yazim ATOMIK DEGIL, egitim kosarken olcum "
                  "calistirma.")
        if os.path.exists(yol):
            os.remove(yol)
        os.replace(t, yol)


_ATOMIK_UYARI = False


def _yaz_json(yol: str, nesne):
    def w(t):
        with open(t, "w", encoding="utf-8") as f:
            json.dump(nesne, f, indent=1)
    _atomik(yol, w)


def _commit() -> str:
    """Bu sayilari HANGI KOD uretti.

    Defter GitHub'dan klonluyor ve commit'i EKRANA basiyordu -- ama cikti
    klasorune YAZMIYORDU. Uc gun sonra "bu kosu hangi koddan" sorusunun
    mekanik cevabi yoktu. `+KIRLI`: calisma agacinda kaydedilmemis
    degisiklik var, yani commit tek basina kosuyu TARIF ETMIYOR."""
    k = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "-C", k, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        if h.returncode:
            return "?"
        d = subprocess.run(["git", "-C", k, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=15)
        return h.stdout.strip() + ("+KIRLI" if d.stdout.strip() else "")
    except Exception:
        return "?"


def _gpu_adi() -> str:
    try:
        return torch.cuda.get_device_name(0) if DEV == "cuda" else ""
    except Exception:
        return ""


def _yazilabilir(alt: str, yaz=print):
    """Drive GERCEKTEN bagli mi -- ve oraya yazip geri okuyabiliyor muyuz?

    Colab'da drive.mount calismadiysa `/content/drive/MyDrive/...` sihirli
    bir yol degildir, siradan bir klasordur: `os.makedirs` hic sikayet
    etmeden onu GECICI DISKTE acar. Kosu 45 dk surer, biter, runtime olur
    ve HER SEY SILINIR -- hicbir yerde hata gorunmez. `ismount` gercek
    FUSE baglantisi ile sahte klasoru ayirir."""
    p = os.path.abspath(alt).replace(os.sep, "/")
    if "/content/drive" in p and not os.path.ismount("/content/drive"):
        raise SystemExit(
            os.linesep + "!! /content/drive BAGLI DEGIL." + os.linesep
            + f"   {alt} Drive gibi duruyor ama runtime'in GECICI diski;"
            + os.linesep
            + "   kosu bitince her sey silinir ve hicbir hata gorunmez."
            + os.linesep
            + "   Defterin 'Drive' hucresini (drive.mount) calistir.")
    t = os.path.join(alt, ".yazma_denemesi")
    with open(t, "w") as f:
        f.write("ok")
    with open(t) as f:
        assert f.read() == "ok", f"yazildi ama geri okunamadi: {alt}"
    os.remove(t)
    yaz(f"  yazilabilir: {alt}" + ("   (DRIVE)" if "/content/drive" in p else ""))


def _klasor_hazirla(alt: str, ustune: bool, yaz=print):
    """Bir kosu klasorune IKINCI kez yazilmasin. Hicbir sey SILINMEZ.

    Uc ariza olculdu (15 Eylul), ucu de sessizdi:

    1. COKEN KOSU KORUNMUYORDU. Ilk olcumden once coken bir kosu geriye
       yalniz `ayar_*.json` + `kosu_*.json` birakiyor -- anlik goruntu de
       egri de yok. Eski denetim yalniz onlara bakiyordu, dolayisiyla
       "bos" sayip tekrar kosuyordu ve cokme kaydini SILIYORDU. Artik
       klasordeki HERHANGI bir dosya doluluk sayilir.

    2. `ustune=True` KLASORU TEMIZLEMIYORDU, uzerine yaziyordu. Olculdu:
       8 adimlik kosunun uzerine 4 adimlik kosu koyuldu, snap/ icinde
       2,4 (yeni) ile 6,8 (eski) YAN YANA kaldi.

    3. VE bu karisim pencere_a'ya girdi: "anlik 4 goruntu: 2..8" deyip
       IKI FARKLI KOSUNUN agirliklarini ayni pencerede ortaladi. Hicbir
       sey hata vermedi. Arsivdeki kol C tam boyle gecersiz kalmisti.

    Cozum: `ustune` artik SILMEZ, TASIR -- eski klasor
    `<alt>_eski_<zaman>` olur, yenisi bos baslar. Tasima basarisiz olursa
    hata YUKARI FIRLAR; hicbir kosulda silmeye dusmeyiz."""
    var = [y for y in glob.glob(os.path.join(alt, "*")) if os.path.isfile(y)]
    var += glob.glob(os.path.join(alt, "snap", "*.pt"))
    if not var:
        return
    if not ustune:
        raise SystemExit(
            os.linesep
            + f"!! {alt} ZATEN DOLU ({len(var)} dosya) -- burada bitmis,"
            + os.linesep
            + "   yarim kalmis ya da COKMUS bir kosu var." + os.linesep
            + "   Yeni tohum BASKA klasore yazar (t1/, t2/)." + os.linesep
            + "   Ayni tohumu tekrar kosmak istiyorsan: ustune=True"
            + os.linesep
            + "   (--ustune). O da SILMEZ: eskisini _eski_<zaman> diye"
            + os.linesep + "   yan klasore tasir.")
    yedek = f"{alt}_eski_{time.strftime('%Y%m%d_%H%M%S')}"
    os.rename(alt, yedek)          # basarisiz olursa firlasin: SILMEYIZ
    os.makedirs(alt, exist_ok=True)
    yaz(f"  ustune=True -> eski kosu SILINMEDI, tasindi: "
        f"{os.path.basename(yedek)}/  ({len(var)} dosya)")


def surdurme_yaz(yol, ayar, model, opt, scaler, rs, adim, egri, iz):
    """SURDURME PAKETI -- kosuyu KALDIGI YERDEN devam ettirmeye yeter.

    Anlik goruntu (`snap/*.pt`) BUNU YAPAMAZ: icinde yalniz fp16 AGIRLIK
    var. Olculdu (15 Eylul): 28 anahtarin hepsi model agirligi; AdamW'nin
    `exp_avg`/`exp_avg_sq` momentleri, adim sayaci, GradScaler olcegi ve
    batch RNG durumu YOK. Agirliktan devam edilirse optimizer SIFIRDAN
    baslar, yorunge kesintisiz kosudan FARKLI olur ve hicbir sey hata
    vermez -- arsivdeki kol C tam boyle gecersiz kalmisti.

    CLAUDE.md kural 1 "yetmezse uzatilir" diyordu; kod bunu
    imkansiz kiliyordu. Kural ile kod CELISIYORDU.

    TEK dosya, her olcum noktasinda ATOMIK olarak ustune yazilir (~41 MB:
    fp32 agirlik + iki AdamW momenti). Yani kosu koparsa en fazla
    `olc_her` adim kaybedilir."""
    def w(t):
        torch.save(dict(
            model=model.state_dict(), opt=opt.state_dict(),
            scaler=scaler.state_dict(), rs=rs.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=(torch.cuda.get_rng_state_all() if DEV == "cuda" else None),
            adim=adim, egri=egri, ayar=ayar.sozluk(), olcme_izi=iz,
        ), t)
    _atomik(yol, w)


def surdurme_oku(yol, ayar: Ayar, model, opt, scaler, rs, iz, yaz=print):
    """Paketi geri kur. UYMAYAN her sey burada DURDURUR, sessiz gecmez."""
    # map_location="cpu": DEV verilirse paketteki HER tensor GPU'ya tasinir
    # ve RNG durumlari bozulur (asagiya bak). Optimizer durumu CPU'dan
    # yuklenince `load_state_dict` onu zaten parametrenin cihazina taşır.
    p = torch.load(yol, map_location="cpu", weights_only=False)
    eski = p["ayar"]
    yeni = ayar.sozluk()
    # `adim` DISINDA her alan ayni olmali: uzatma butceyi degistirir,
    # modeli/veriyi DEGISTIRMEZ. `ad` da serbest degil -- cikti adlarina
    # giriyor.
    fark = {k for k in yeni if k != "adim" and eski.get(k) != yeni[k]}
    if fark:
        raise SystemExit(
            os.linesep + f"!! SURDURULEMEZ: ayar degismis: {sorted(fark)}"
            + os.linesep
            + f"   eski { {k: eski.get(k) for k in sorted(fark)} }"
            + os.linesep
            + f"   yeni { {k: yeni[k] for k in sorted(fark)} }" + os.linesep
            + "   Butce disinda bir sey degistiyse bu SURDURME degil, "
            + "BASKA bir kosudur.")
    if p.get("olcme_izi") != iz:
        raise SystemExit(
            os.linesep + f"!! SURDURULEMEZ: olcme seti degismis "
            f"(paket {p.get('olcme_izi')}, simdi {iz}).")
    if p["adim"] >= ayar.adim:
        raise SystemExit(
            os.linesep + f"!! SURDURULECEK BIR SEY YOK: paket {p['adim']} "
            f"adimda, hedef {ayar.adim}. Uzatmak icin `adim` buyutulmeli.")
    model.load_state_dict(p["model"])
    opt.load_state_dict(p["opt"])
    scaler.load_state_dict(p["scaler"])
    rs.set_state(p["rs"])
    # RNG durumlari CPU ByteTensor OLMAK ZORUNDA.
    # OLCULDU (15 Eylul, T4'te ilk gercek surdurmede): paket
    # `map_location=DEV` ile yuklenince RNG tensorleri de GPU'ya tasindi ve
    # `set_rng_state` "RNG state must be a torch.ByteTensor" diye patladi.
    # CPU'da hic gorunmuyordu: orada `cuda_rng` None, yani bu dal HIC
    # calismiyordu. Bit-duzeyinde gecen 8+8 testim de CPU'daydi -- test
    # onemli dali KAPSAMIYORDU.
    _cpu = lambda x: x.cpu() if torch.is_tensor(x) else x
    torch.set_rng_state(_cpu(p["torch_rng"]))
    if DEV == "cuda" and p.get("cuda_rng"):
        torch.cuda.set_rng_state_all([_cpu(x) for x in p["cuda_rng"]])
    yaz(f"  SURDURULUYOR: adim {p['adim']} -> {ayar.adim}   "
        f"({len(p['egri'])} olcum noktasi devralindi)")
    return p["adim"], list(p["egri"])


def erken_teshis(r: dict, ayar: Ayar, yaz=print, uyarildi: set | None = None):
    """Bozuk kosuyu 45 dakika sonra degil, ILK OLCUMDE yakala.

    BIRIM TESTI onkayitta (belge/onkayit/model_a.md 5) zaten YAZILIYDI --
    ama yalniz kosu BITTIKTEN sonra pencere_a'da degerlendiriliyordu. Yani
    "olcum kodu bozuk mu" sorusunun cevabi icin butun kosuyu beklemek
    gerekiyordu. Burada adim 0'da soruluyor: saniye 0."""
    a = r["adim"]
    k = r.get("ent_yok_kisayol")
    if k is not None and k == k and abs(k) > 1e-9:      # k == k  ->  NaN degil
        raise SystemExit(
            os.linesep
            + f"!! BIRIM TESTI KALDI (adim {a}): ent_yok_kisayol {k:.4f}, "
            + "MATEMATIKSEL OLARAK 0 olmaliydi." + os.linesep
            + "   ENT-YOK zincirlerinde facts hucresi -1'dir; kisayol cevabi"
            + os.linesep
            + "   ent_off-1 cikar, yani bir ILISKI token'i -- hicbir varlik"
            + os.linesep
            + "   tahminiyle eslesemez. Sifir DEGILSE olcum kodu bozuktur ve"
            + os.linesep
            + "   bu kosunun BUTUN sayilari okunmaz. Kosu durduruldu.")
    kp = r.get("kayip")
    if kp is not None and not math.isfinite(kp):
        raise SystemExit(
            os.linesep + f"!! KAYIP {kp} (adim {a}) -- NaN/inf." + os.linesep
            + "   fp16 + GradScaler bozuk GRADYANI atlar, ama agirliklar bir"
            + os.linesep
            + "   kez NaN olursa egitim SESSIZCE devam eder ve butun"
            + os.linesep
            + "   dogruluklar sifira duser. Kosu durduruldu.")
    # UYARI BIR KEZ: her olcum noktasinda tekrarlanirsa logu doldurur ve
    # asil satirlari gozden kacirtir.
    if (a >= 2 * ayar.isinma and r.get("one", 1.0) < 0.05
            and (uyarildi is None or "one" not in uyarildi)):
        if uyarildi is not None:
            uyarildi.add("one")
        yaz(f"  !! UYARI: adim {a}, isinma ({ayar.isinma}) coktan bitti ama "
            f"one {r['one']:.3f}.")
        yaz("     Atomik olgu ezberi bu gorevin EN KOLAY parcasi, sans "
            "seviyesi ~0.001.")
        yaz("     Burada takilmak egitimin bozuk oldugunu DUSUNDURUR "
            "-- kapi degil, UYARI.")


# ======================= EGITIM ==========================================
def egit(ayar: Ayar, alt=None, yaz=print, ustune=False, commit=None,
         surdur=False) -> list:
    alt = alt or f"cikti_{ayar.ad}_t{ayar.tohum}"
    os.makedirs(alt, exist_ok=True)
    _yazilabilir(alt, yaz)              # Drive gercekten bagli mi, saniye 0'da
    sur_yol = f"{alt}/surdurme_t{ayar.tohum}.pt"
    surduruluyor = surdur and os.path.exists(sur_yol)
    if surdur and not surduruluyor:
        raise SystemExit(
            os.linesep + f"!! SURDURME PAKETI YOK: {sur_yol}" + os.linesep
            + "   Bu klasordeki kosu surdurme destegi EKLENMEDEN once"
            + os.linesep
            + "   kosulmus olabilir. O zaman uzatma bir SURDURME degil,"
            + os.linesep + "   BASTAN kosudur: --ustune ile yeniden kos.")
    if not surduruluyor:
        _klasor_hazirla(alt, ustune, yaz)   # dolu klasore IKINCI kez yazma
    # Anlik goruntuler AYRI alt klasorde: 20.000 adimda 10, uzatilirsa 20
    # dosya oluyor ve tohum klasorunde okunmasi gereken 4 json'u gomuyor.
    # _klasor_hazirla'dan SONRA: once doluluk bakilir, sonra klasor acilir.
    os.makedirs(f"{alt}/snap", exist_ok=True)
    yaz(f"=== {ayar.ad}  tohum {ayar.tohum} ===  cihaz {DEV}  cikti {alt}/")
    v = veri_kur(ayar, yaz)
    Xtr, Ptr, Ttr, kimlik = egitim_havuzu(ayar, v, yaz)

    L = olcme_listeleri(ayar, v)
    kod = {k: (kodla_1hop(v, L[k]) if k == "one" else kodla_2hop(v, L[k]))
           for k in L if L[k]}
    iz = olcme_izi(L)
    yaz("  olcme: " + "  ".join(f"{k} {len(L[k])}" for k in L if L[k])
        + f"   parmak izi {iz}")

    torch.manual_seed(ayar.tohum)
    model = Model(ayar, v.vocab).to(DEV)
    yaz(f"  parametre {model.n_param():,}  (d={ayar.d} l={ayar.l} "
        f"nh={ayar.nh} dff={ayar.dff} dongu={ayar.dongu})"
        f"  -> {ayar.l*ayar.dongu} katman-esdegeri hesap")

    # dim>=2 -> decay.  GOMME DE BURAYA GIRIYOR (dim 2) ve head'e bagli
    # oldugu icin tek sayilir. Bu bir SECIM: cok sayida LLM tarifi gommeyi
    # decay DISINDA tutar. 2603.25009 "AdamW ... weight decay = 1.0" diyor,
    # grup ayrimindan bahsetmiyor -> her seye uygulandigi okundu. wd buyudukce
    # (0.1 -> 1.0) bu secim onem kazanir; ACIK DUGME.
    dec = [p for p in model.parameters() if p.dim() >= 2]
    nodec = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": ayar.wd},
                             {"params": nodec, "weight_decay": 0.0}],
                            lr=ayar.lr, betas=tuple(ayar.betas))
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    rs = np.random.RandomState(ayar.tohum + 991)
    egri, t0 = [], time.time()

    # Ayarin YANINA olcme izini de yaz: "bu kosu hangi ornekleri olctu"
    # sorusu sonradan MEKANIK olarak cevaplanabilsin.
    _yaz_json(f"{alt}/ayar_t{ayar.tohum}.json",
              dict(ayar.sozluk(), _olcme_izi=iz))

    # KUNYE: ayar "ne isteyecektik"i, kunye "fiilen ne kostu"yu yazar. Ikisi
    # ayri sey. Commit, GPU, torch surumu ve veri sayilari SADECE LOGA
    # basiliyordu; log ise defterde tek dosyaydi ve her kosuda ustune
    # yaziliyordu -- yani bu bilgiler ikinci kosuda KAYBOLUYORDU.
    kunye_yolu = f"{alt}/kosu_t{ayar.tohum}.json"
    kunye = dict(
        ad=ayar.ad, tohum=ayar.tohum, durum="KOSUYOR",
        commit=commit or _commit(),
        cihaz=DEV, gpu=_gpu_adi(), torch=torch.__version__,
        numpy=np.__version__, python=sys.version.split()[0],
        baslangic=time.strftime("%Y-%m-%d %H:%M:%S"),
        parametre=model.n_param(), katman_esdegeri=ayar.l * ayar.dongu,
        olcme_izi=iz, havuz=int(len(Xtr)),
        veri=dict(olgu=len(v.one), egitim2=len(v.tr2), comp=len(v.comp),
                  ent=len(v.ent), ent_yok=len(v.ent_yok),
                  ent_arama=len(v.ent_arama), n_ent=v.n_ent, n_rel=v.n_rel,
                  vocab=v.vocab, phi=round(v.phi, 4),
                  wang_phi=round(v.wang_phi, 4)),
        olcme={k: len(L[k]) for k in L},
    )
    _yaz_json(kunye_yolu, kunye)
    yaz(f"  kunye: commit {kunye['commit']}  {kunye['gpu'] or DEV}  "
        f"torch {kunye['torch']}  -> {kunye_yolu}")

    egri_yolu = f"{alt}/egri_{ayar.ad}_t{ayar.tohum}.json"

    def _nokta(adim, kayip, lr, kayip_son=None):
        # `kayip` ARALIK ORTALAMASI (bkz. dongudeki birikim), `kayip_son`
        # yalniz son batch. Onkayit 4.6 "kayip doyma adimi"ni okuyacak; tek
        # batch'in kaybi bunun icin gurultulu bir tahminci -- batch 512'de
        # ornekleme sacilimi tek basina 0.05-0.1 oynatiyor. Ikisi de yaziliyor
        # ki "ortalama mi dustu, gurultu mu" sorusu sonradan sorulabilsin.
        r = dict(adim=adim, kayip=kayip, kayip_son=kayip_son, lr=float(lr),
                 sn=round(time.time() - t0, 1))
        for k in kod:
            r[k] = dogruluk(model, v, *kod[k])
        r["ent_kisayol"] = kisayol_orani(model, v, L["ent"])
        r["ent_yok_kisayol"] = kisayol_orani(model, v, L["ent_yok"])
        if kimlik is not None:
            r["kimlik"] = dogruluk(model, v, *kimlik)
        return r

    def _satir(r):
        return (f"  {r['adim']:7d}/{ayar.adim}  "
                + ("kayip   ---" if r["kayip"] is None
                   else f"kayip {r['kayip']:.3f}")
                + "  " + "  ".join(f"{k} {r[k]:.3f}" for k in
                                   ("one", "seen", "comp", "ent") if k in r)
                + f"  ksy {r['ent_kisayol']:.3f}"
                + (f"  kimlik {r['kimlik']:.3f}" if "kimlik" in r else "")
                + f"  ({r['sn']/60:.0f} dk)")

    # --- ADIM 0 -- hicbir sey egitilmeden OLCUM YOLUNUN TAMAMI kosulur.
    # Iki isi var: (a) BIRIM TESTI'ni saniye 0'da patlatmak -- olcum kodu
    # bozuksa 45 dakika beklemenin anlami yok; (b) SANS SEVIYESINI bu veride
    # olcmek (teorik 1/1060 ~ 0.001, ama tahmin degil OLCUM yazilsin).
    # ANLIK GORUNTU KAYDEDILMEZ: egitilmemis agirlik pencere_a'nin agirlik
    # ortalamasina girerse ilk pencereyi KIRLETIR.
    uyarildi = set()
    r0 = None
    tahmin = False
    bas = 0
    if surduruluyor:
        bas, egri = surdurme_oku(sur_yol, ayar, model, opt, scaler, rs, iz, yaz)
        r0 = egri[0]                    # adim 0 olcumu devralindi
        # ONCEKI KUNYE SILINMEZ. Surdurulen kosu baska bir oturumda, baska
        # bir GPU'da, baska bir commit'te baslamis olabilir; "bu sayilar
        # hangi kosudan" sorusu oturum basina cevaplanabilmeli.
        try:
            _eski = json.load(open(kunye_yolu, encoding="utf-8"))
            _onc = _eski.pop("onceki", [])
            kunye["onceki"] = _onc + [_eski]
        except Exception:
            pass
        kunye.update(surduruldu=True, surdurme_baslangic=bas)
        _yaz_json(kunye_yolu, kunye)
    # Kayip birikimi GPU'da tutulur: her adimda .item() demek her adimda
    # GPU senkronu demek olurdu. Tensor olarak toplanip yalniz olcum
    # noktasinda bir kez okunuyor -- bedeli yok.
    kayip_top = torch.zeros((), device=DEV)
    kayip_say = 0
    # ADIM 0 DA `try` ICINDE. Disaridayken burada coken bir kosu kunyeyi
    # `durum: KOSUYOR`da birakiyordu -- olculdu: ilk olcumde patlayan kosu
    # ne `HATA` yazdi ne de sebebi. Klasor "yarim mi, kosuyor mu, oldu mu"
    # belli olmadan kaliyordu.
    try:
        if not surduruluyor:
            r0 = _nokta(0, None, 0.0)
            egri.append(r0)
            yaz(_satir(r0) + "   <- SANS (egitim yok, anlik goruntu YAZILMAZ)")
            erken_teshis(r0, ayar, yaz, uyarildi)
            _yaz_json(egri_yolu, egri)

        for adim in range(bas + 1, ayar.adim + 1):
            if adim < ayar.isinma:
                lr = ayar.lr * adim / ayar.isinma
            elif ayar.sabit_lr:
                lr = ayar.lr              # grokking icin LR SONMEMELI
            else:
                lr = ayar.lr * 0.5 * (1 + math.cos(
                    math.pi * (adim - ayar.isinma)
                    / max(1, ayar.adim - ayar.isinma)))
            for g in opt.param_groups:
                g["lr"] = lr

            # YERINE KOYARAK ornekleme: ayni ornek bir batch'te tekrar
            # gelebilir ve EPOCH diye bir sey YOK. Literaturdeki butceler
            # epoch cinsinden (Loop&Generalize "7k epoch", 2603.25009
            # full-batch) -- adim sayimiz onlarla DOGRUDAN kiyaslanamaz.
            # Beklenen gecis: adim*batch/len(Xtr) = 20000*512/51120 ~ 200,
            # ama Poisson sacilimli.
            j = rs.randint(0, len(Xtr), ayar.batch)
            xb = torch.from_numpy(Xtr[j]).to(DEV)
            pb = torch.from_numpy(Ptr[j]).to(DEV)
            tb = torch.from_numpy(Ttr[j]).to(DEV)
            with torch.autocast(DEV, dtype=torch.float16,
                                enabled=(DEV == "cuda")):
                lg = model(xb)
                # xb.shape[0], ayar.batch DEGIL: ikisi burada esit ama bir
                # varyasyon degisken batch kullanirsa `ayar.batch` sessizce
                # yanlis satirlari secerdi.
                lg = lg[torch.arange(xb.shape[0], device=DEV), pb]
                kayip = F.cross_entropy(lg.float(), tb)
            kayip_top += kayip.detach()
            kayip_say += 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(kayip).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if adim % ayar.olc_her == 0 or adim == ayar.adim:
                r = _nokta(adim, float(kayip_top.item() / max(1, kayip_say)),
                           lr, kayip_son=float(kayip.item()))
                kayip_top = torch.zeros((), device=DEV)
                kayip_say = 0
                egri.append(r)
                # ANLIK GORUNTU: agirlik ortalamasi olcumunun sarti. Adim
                # adli, 8 hane sifir dolgulu -- arsivde `f"..._{20000}.pt"`
                # hicbir sey bulmamis ama 190000'i bulmustu (zaten 6
                # haneydi), yani hata KISMEN gorunmustu. Ad `ayar.ad` tasir.
                # ATOMIK: yarim .pt hem torch.load'i patlatir hem de
                # pencere_a'nin glob'una girer (bkz. _atomik).
                yol = (f"{alt}/snap/"
                       f"snap_{ayar.ad}_t{ayar.tohum}_{adim:08d}.pt")
                sd = {k: t.half() for k, t in model.state_dict().items()}
                _atomik(yol, lambda t, _s=sd: torch.save(_s, t))
                _yaz_json(egri_yolu, egri)
                # SURDURME PAKETI: tek dosya, her olcumde ustune yazilir.
                # Kosu koparsa en fazla `olc_her` adim kaybedilir; defterde
                # "surdurme yok, bastan baslar" yaziyordu, artik dogru degil.
                surdurme_yaz(sur_yol, ayar, model, opt, scaler, rs, adim,
                             egri, iz)
                yaz(_satir(r))
                erken_teshis(r, ayar, yaz, uyarildi)
                if not tahmin:
                    tahmin = True
                    # r0['sn'] = BIR olcumun maliyeti (t0 veri/model
                    # kurulumundan SONRA basliyor). r['sn'] icinde iki olcum
                    # var (adim 0 ve bu). Olcum maliyeti ayri sayilmazsa
                    # tahmin 10 olcum kadar EKSIK cikardi.
                    # Surdurulen kosuda r0['sn'] ESKI oturumun saatinden
                    # gelir; bu oturumun saatiyle karistirilamaz.
                    olcum = 0.0 if surduruluyor else r0["sn"]
                    kalan_adim = ayar.adim - bas
                    hiz = max(0.0, r["sn"] - 2 * olcum) / max(1, adim - bas)
                    n_olc = kalan_adim // ayar.olc_her + (0 if surduruluyor else 1)
                    top = hiz * kalan_adim + olcum * n_olc
                    yaz(f"     >> HIZ {hiz*1000:.0f} ms/adim, olcum basina "
                        f"{olcum:.0f} sn x{n_olc}  ->  bu oturum ~{top/60:.0f} dk, "
                        f"kalan ~{(top - r['sn'])/60:.0f} dk.")
                    yaz("        Bu rakam beklenenin cok ustundeyse SIMDI "
                        "durdur -- 45 dk mi 6 saat mi, sonunda degil BURADA "
                        "belli olsun.")
    except BaseException as e:
        kunye.update(durum="HATA", hata=f"{type(e).__name__}: {e}"[:400],
                     bitis=time.strftime("%Y-%m-%d %H:%M:%S"),
                     son_adim=(egri[-1]["adim"] if egri else -1),
                     sure_dk=round((time.time() - t0) / 60, 1))
        _yaz_json(kunye_yolu, kunye)
        raise
    kunye.update(durum="BITTI", bitis=time.strftime("%Y-%m-%d %H:%M:%S"),
                 son_adim=egri[-1]["adim"],
                 sure_dk=round((time.time() - t0) / 60, 1))
    _yaz_json(kunye_yolu, kunye)
    yaz(f"  BITTI  {kunye['sure_dk']} dk  son adim {kunye['son_adim']}")
    return egri


AYAR = Ayar()

if __name__ == "__main__":
    egit(AYAR)
