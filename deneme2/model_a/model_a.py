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
import glob, importlib, json, math, os, subprocess, sys, time
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
    veri_ad: str = "veri_okul"   # HANGI GRAF. "veri_okul2" = tam IKI KATI.
    #   15 Eylul'de eklendi. Modul adi olarak yaziliyor ki `ayar_t<N>.json`a
    #   girsin: "bu kosu hangi veriyi gordu" sorusu SONRADAN cevaplanabilsin.
    #   Alan eklemek SURDURMEYI bozabilirdi (eski paketlerde bu anahtar YOK
    #   ve karsilastirma 'degismis' derdi); `surdurme_oku` icinde ESKI
    #   VARSAYILAN tablosu var, oraya bak.
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
    # --- GERI BESLEMELI AGIRLIK ORTALAMASI (Lookahead, 1907.08610)
    #   Onceden kayit: belge/onkayit/model_a5.md  (kullanici fikri, 15 Eylul)
    #     phi   <- (1 - ort_alfa) * phi + ort_alfa * theta
    #     theta <- phi          (egitim ORTALANMIS agirliktan devam eder)
    #   VARSAYILAN KAPALI: ort_bas=0 -> hicbir sey yapilmaz, model_a..a4'un
    #   davranisi DEGISMEZ. `test_sabit.py` bunu her kosuda dogruluyor.
    # --- DARBOGAZ (model_b ailesi okur, model_a GORMEZDEN GELIR) -----------
    #   DiscoLoop (arXiv 2607.00341) Denklem 4-6. Dongu turlari arasinda
    #   kalinti akisina "cozulmus gomme" kanali eklenir:
    #       H~ = H + alfa * RMSNorm(Phi(H)),   Phi(h) = softmax(Wh/tau) @ W
    #   `model_a.Model` bu alanlari HIC OKUMAZ -- parametre sayisi ve
    #   baslangic agirligi degismez, `test_sabit` bunu dogruluyor.
    dar_alfa: float = 0.0      # 0 = KAPALI. Sabit gecit gucu.
    dar_tau: float = 1.0       # Phi'nin softmax sicakligi.
    dar_kapi: bool = False     # True = ogrenilebilir gecit (d+1 parametre)
    jeton_ad: str = ""         # Varliklari COK JETON olarak kodla.
    #   ""     KAPALI -- varlik = TEK jeton (model_a .. model_b4).
    #   "ilk"  KUSURLU. ILK alt cizgiden bolup IKI AYRIK sozluk kurar.
    #          `model_b5` bununla kosuldu ve KUSURLU ilan edildi
    #          (onkayit model_b5.md §8.1): ayni dizge iki ayri jeton
    #          oluyor (Aydin sehir vs soyad), `Fen_Lisesi` icinde alt
    #          cizgiyle TEK jeton kaliyor. Yeniden uretilebilirlik icin
    #          DURUYOR, YENI KOL ICIN KULLANILMAZ.
    #   "tam"  DOGRUSU. BUTUN alt cizgilerden boler, TEK PAYLASILAN
    #          sozluk kurar, 3 yuvaya sagdan doldurur:
    #            Ayse_Yilmaz       -> (Ayse,    Yilmaz, <YOK>)
    #            Adana_Fen_Lisesi  -> (Adana,   Fen,    Lisesi)
    #            Adana             -> (Adana,   <YOK>,  <YOK>)
    #            Nukleer_Fizik     -> (Nukleer, Fizik,  <YOK>)
    #          Boylece `Adana` sehirde de okulda da AYNI jeton, `Lisesi`
    #          butun okullarda ortak, `Fizik` her yerde ayni.
    #          Onkayit: belge/onkayit/model_b6.md §2 (8 kontrol + bolme
    #          gecerliligi + kisayol olcumu).
    #   Ayse_Yilmaz -> (Ayse, Yilmaz);  Ankara_Fen_Lisesi -> (Ankara,
    #   Fen_Lisesi);  Ankara -> (Ankara, <YOK>).  ILK alt cizgiden bolunur,
    #   cunku anlamli olan o. Sonucu: sozluk 2145 -> ~300 ve varliklar
    #   BILESIK hale gelir -- 700 kisi, 100 ad + 7 soyaddan kurulur.
    #   `T_LEN` 8 -> 11 (t_len ozelligi), cevap IKI jeton, kayip IKI hedef.
    #   MIMARI DEGISMEZ. Onkayit: belge/onkayit/model_b5.md
    #   VARSAYILAN KAPALI -- eski kollarin hepsi bit duzeyinde AYNI kalir.
    kopru_kayip: float = 0.0   # YARDIMCI KAYIP: kopruyu TAHMIN ETTIR.
    #   0.0 = KAPALI, eski kollarla BIT AYNI.
    #   >0 ise ana kayba su eklenir: r1 ve r2 pozisyonlarinda koprunun
    #   ILK iki token'ini uret. O pozisyonlar nedensel olarak (e, r1)'i
    #   GORMUS durumda, yani kopru orada BELIRLI; ve ana kayip onlari
    #   KULLANMIYOR (cevap 3+yuva'dan itibaren okunuyor).
    #   BU BIR TESHIS, mimari oneri DEGIL: kopru etiketi graftan geliyor,
    #   gercek metinde gelmezdi. SINAVDA hicbir sey enjekte edilmez --
    #   model normal kosar, comp/ood her zamanki gibi olculur.
    #   Sordugu soru: kopru hidden state'e KONULURSA bilesim duzelir mi?
    #   OLCULDU (16 Eylul, lineer sonda): model_b6'da koprunun AD token'i
    #   hidden state'ten cikarilamiyor (0.0162, en sik sinif 0.0227).
    #   Onkayit: belge/onkayit/model_b8.md
    dar_kafa: int = 1          # Phi'nin BAS sayisi (model_c).
    #   1 = KAPALI, ek parametre YOK, `model_b` ile BIT AYNI.
    #   m > 1:  Phi_m(h) = (1/m) SUM_j softmax(W A_j nf(h)/tau) @ W
    #   Her bas bir YUVA cozer -- cok jetonlu bir varligin "temiz
    #   gommesi" ancak boyle kurulabilir. OLCULDU (16 Eylul, asama1):
    #   jeton_ad="tam" iken kopru 11 pozisyonun hicbirinde cozulemiyor
    #   (0.0442), cunku Phi POZISYON-YEREL ve uc jetonlu varligin tek
    #   bir gomme satiri YOK. A_0 = I ile baslatilir.
    #   Onkayit: belge/onkayit/model_c.md
    dar_sert: bool = False     # Phi'yi SERTLESTIR: tau -> 0 limiti.
    #   Phi(h) = W[argmax(nf(h) Wᵀ)] -- yumusak ortalama yerine TEK gomme.
    #   argmax turevlenemez; straight-through (van den Oord 2017, VQ-VAE):
    #   ileri sert, geri gradyan nf(h)'ye DOGRUDAN gecer. Yani bu bir
    #   VEKTOR NICEMLEME -- gizli durum sozluk gomme tablosuna yuvarlanir.
    #   IKI SEYI BIRDEN degistirir (ileri gecis + gradyan yolu); onkayit
    #   belge/onkayit/model_b3.md §2 bunu kusur olarak yaziyor.
    #   `dar_sdpa` ile BIRLIKTE kullanilamaz (sert yolda softmax YOK).
    dar_sdpa: bool = False     # Phi'yi SDPA ile hesapla. SAYISAL DEGIL,
    #   HESAPLAMA yolu degisir: Phi(h) = softmax(nf(h) Wᵀ/tau) @ W tam
    #   olarak Attention(Q=nf(h), K=W, V=W, scale=1/tau) -- `head` bias'siz
    #   ve `head.weight is emb.weight` oldugu icin bu CEBIRSEL OZDESLIK.
    #   Kazanc (N,V) ara matrisinin HIC yazilmamasindan gelir (B=1000,
    #   V=2145 -> 34,3 MB). Onkayit: belge/onkayit/model_b2.md
    #   BIT DUZEYINDE ayni DEGIL (fp16 toplama sirasi) -> yorunge ayrisir.
    ood_pay: float = 0.0       # 0 = KAPALI. >0 ise ATOMIK OLGULARIN (kenar)
    #   bu orani atomic_OOD'ye ayrilir -- Wang 2405.15071 §3.1'in birebir
    #   tanimi: "The atomic facts are then the EDGES ... which we partition
    #   disjointly into atomic_ID and atomic_OOD (95%:5%)".
    #   EGITIM  = iki kenari da ID olan zincirler (train_inferred_ID)
    #   SINAV   = iki kenari da OOD olan zincirler (test_inferred_OOD)
    #   KARISIK = bir kenari OOD -> NE EGITIM NE SINAV, tamamen duser
    #   ATOMIK OLGULAR HEPSI EGITIMDE (Wang §2: "our training set includes
    #   ALL the atomic facts").
    #   `kati_pay`dan FARKI: orada VARLIK boluyorduk ve ikinci hop kenari
    #   egitimde 2. hop olarak %93,7 geciyordu -- yani makalenin %0'ini
    #   ureten mekanizma YOKTU. Burada tanim geregi %0.
    kati_pay: float = 0.0      # 0 = KAPALI. >0 ise varliklarin bu orani
    #   HICBIR egitim zincirinde gorunmez -- ne bas, ne kopru, ne cevap.
    #   `ent_pay`den FARKI: ENT varliklari kopru ve cevap olarak egitimde
    #   GORUNUYOR, yalniz bas olmuyor. KATI olanlar hicbir rolde yok.
    #   ATOMIK OLGULARI DURUR -- Wang 2405.15071 de atomicOOD'yi egitimde
    #   tutuyor; sinav "olgulari biliyor, zincir kurabiliyor mu".
    #   VARSAYILAN KAPALI -> model_a..a6'nin verisi DEGISMEZ.
    ort_bas: int = 0           # 0 = KAPALI. >0 ise bu adimdan SONRA ortala.
    ort_her: int = 0           # 0 -> `olc_her` kullanilir. Lookahead'in `k`si.
    #   DIKKAT: makale k=5..10 tariyor; olc_her=2000 onun 200-400 KATI.
    #   Sonumleme cikarsa once BU sorgulanmali, fikir degil.
    ort_alfa: float = 0.5      # 0.5 = "bir onceki modelle esit ortalama".
    olc_her: int = 2000        # 20.000/2000 = 10 olcum noktasi (kullanici).
    n_olcum_max: int = 3000    # her olcme kumesinden en fazla

    # --- kimlik gorevi (model_a1 / model_a2 bunu degistirir)
    ident_frac: float = 0.0
    ident_kip: str = ""        # "" | "q2son" | "q1"
    belge_pay: float = 0.0
    # --- EK ISARETLEYICILI KODLAMA (16 Eylul) --------------------------
    # Bugunku dilde rolu POZISYON tasiyor: "Fatma anne Ayse" ile
    # "Ayse anne Fatma" farkli seyler. Yani siradan bir permutasyon
    # ANLAMI BOZAR -- bu, bicim cesitliligini imkansiz kiliyordu.
    # Turkce'de sira serbesttir cunku rolu EK tasir:
    #   "Ayse'nin annesi Fatma'dir" = "Fatma'dir Ayse'nin annesi"
    # ek_kip="tr" dort jeton ekler -- ILISKI DEGIL, DILBILGISI:
    #   '     ozel adla ek arasina (Ayse'nin)
    #   <NIN> tamlayan (sahip)
    #   <SI>  tamlanan (iliski)
    #   <DIR> yuklem (cevap)
    # Jetonlar SOZLUGUN SONUNA ekleniyor -> REL_OFF/ent_off KAYMAZ,
    # ek_kip="" ile uretilen diziler BIT AYNI kalir.
    ek_kip: str = ""           # "" | "tr"
    bicim: int = 1             # kac YUZEY BICIMI (1..3), ek_kip GEREKTIRIR
    #   0 = KAPALI. >0 ise egitim havuzuna BELGE satirlari eklenir:
    #   ZINCIRLENEN IKI ATOMIK OLGU AYNI DIZIDE.
    #
    #       [S1] Ayse Yilmaz <YOK> cocuk  ? Fatma Yilmaz <YOK> <EOS>
    #       [S1] Fatma Yilmaz <YOK> kardes ? Emre Yilmaz <YOK> <EOS>
    #       ------------------- TEK DIZI, 20 jeton -------------------
    #
    #   YENI OLGU YOK: iki olgu da `one` icinde zaten AYRI AYRI var.
    #   Eklenen tek sey BITISIKLIK -- gercek metinde "Ayse'nin cocugu
    #   Fatma. Fatma'nin kardesi Emre." ayni paragraftadir; bizde
    #   hicbir dizi iki olguyu birden tasimiyordu (olculdu: koprunun
    #   ayirt edici jetonu 2-hop satirlarinin %3,28'inde geciyor, o da
    #   isim cakismasi).
    #
    #   BELGELER YALNIZ `tr2`DEN kurulur. comp/ent/ent_yok/ood/
    #   ent_arama zincirlerinden KURULMAZ -- kurulsaydi o zincirlerin
    #   koprusu baglama YAZILMIS olurdu, yani SIZINTI. `egitim_havuzu`
    #   bunu assert ile denetler.
    #
    #   `belge_pay` = havuzun ne kadari BELGE olacak (ident_frac ile
    #   ayni desen). 0.5 -> yari yariya.
    tam_kayip: bool = False
    #   False: kayip YALNIZ cevap yuvalarinda -- bu bir SORU-CEVAP kaybi.
    #   True : kayip HER pozisyonda, next-token -- yani DIL MODELI kaybi.
    #
    #   16 Eylul, kullanici: "normal dil egitimindeki veri mimarisini
    #   kullanmaliyiz." Olculdu: `lg_tam = model(xb)` butun pozisyonlarin
    #   logit'ini zaten uretiyor, biz 11'in 3'unu alip %70'ini ATIYORUZ.
    #   Bu yuzden model 'Ahmet' -> 'Yilmaz' gecisini HIC tahmin etmiyor;
    #   varligin IKI jetonunu birim olarak baglamasi icin tek baski,
    #   cevabi uretirken. Model nedensel maskeli (is_causal=True), yani
    #   next-token kaybi mesru -- ileriye bakip KOPYALAYAMAZ.

    # --- maske (bu deneyde kapali; aile ilerde kullanabilsin diye duruyor)
    mask_poz: int | None = None
    mask_blok: tuple = ()

    def degistir(self, **kw) -> "Ayar":
        bilinmeyen = set(kw) - {f.name for f in dc.fields(self)}
        assert not bilinmeyen, f"Ayar'da boyle alan yok: {bilinmeyen}"
        return dc.replace(self, **kw)

    @property
    def t_len(self) -> int:
        """Dizi uzunlugu. ALAN DEGIL, TURETILMIS -- `jeton_ad`dan duser,
        yani iki yerde iki deger olamaz.

            [Q2] e        r1 r2 ?  a       EOS       ->  8
            [Q2] e1 e2    r1 r2 ?  a1 a2   EOS       -> 11
        """
        if self.ek_kip:
            # bicim 0 (en uzun): 2-hop
            #   e1 e2 e3 ' <NIN> r1 <SI> <NIN> r2 <SI> ? a1 a2 a3 ' <DIR> EOS
            #   = 2*yuva + 11 = 17   (yuva 3)
            # 1-hop bicim 0: e ' <NIN> r <SI> ? a ' <DIR> EOS = 14
            # kimlik satiri (eksiz, degismedi) = 10 -- ikisi de siginir.
            assert self.jeton_ad == "tam", (
                f"ek_kip su an yalniz jeton_ad='tam' ile KURULDU: "
                f"{self.jeton_ad!r}")
            assert self.belge_pay == 0, (
                "ek_kip + belge_pay BIRLIKTE KURULMADI: belge satirina ek "
                "isaretleyici eklenmedi, t_len turetimi yanlis olur.")
            return 17
        if not self.jeton_ad:
            return T_LEN
        # "ilk" 2 yuva, "tam" 3 yuva. Ikisinde de 2-hop 11'e siginiyor:
        #   "ilk"  [Q2] e1 e2    r1 r2 ? a1 a2    EOS  =  9
        #   "tam"  [Q2] e1 e2 e3 r1 r2 ? a1 a2 a3 EOS  = 11
        if self.belge_pay > 0:
            # BELGE = iki 1-hop olgu YAN YANA.
            #   [S1] e1 e2 e3 r ? b1 b2 b3 EOS  = 10   (x2 = 20)
            # 2-hop sorusu 11; 20 ikisini de kapsiyor.
            #   !! BEDELI VAR: butun satirlar 20'ye DOLGULANIR ve
            #   attention T^2 -> 11^2=121'den 20^2=400'e cikar, ~3,3 kat.
            return 20
        return 11

    def sozluk(self) -> dict:
        return dc.asdict(self)

    def fark(self, other: "Ayar") -> dict:
        a, b = self.sozluk(), other.sozluk()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k] and k != "ad"}


# ESKI KOSULARLA UYUM. Bir alan Ayar'a SONRADAN eklenirse, ondan once
# yazilmis `ayar_t<N>.json` ve `surdurme_t<N>.pt` dosyalarinda o anahtar
# YOKTUR. Tablo olmasa yeni bir alan eklemek:
#   - butun eski kosularin SURDURULMESINI kirardi (surdurme_oku)
#   - butun eski kosularin OLCULMESINI kirardi (pencere_a.ayar_oku)
# Olculdu (15 Eylul, `veri_ad` eklenirken): ikisi de fiilen kirildi.
#
# BURAYA YAZILAN DEGER, ALANIN EKLENMEDEN ONCE KODUN FIILEN YAPTIGI SEY
# OLMALI -- "makul varsayilan" degil. Yanlis yazilirsa eski kosular
# SESSIZCE yanlis etiketlenir.
ESKI_VARSAYILAN = {
    "veri_ad": "veri_okul",   # 15 Eylul oncesi tek veri kaynagi buydu
    # Geri beslemeli ortalama (Lookahead) 15 Eylul'de eklendi. Ondan once
    # BOYLE BIR SEY YOKTU -> kapali. `ort_bas=0` tam olarak "kapali"
    # demek; digerleri o durumda hic okunmuyor ama alan olarak var olmali.
    "ort_bas": 0,
    "ort_her": 0,
    "ort_alfa": 0.5,
    # ent_kati bolmesi 15 Eylul'de eklendi; ondan once YOKTU -> kapali.
    "kati_pay": 0.0,
    "ood_pay": 0.0,
    "dar_alfa": 0.0, "dar_tau": 1.0, "dar_kapi": False, "dar_sdpa": False,
    "dar_sert": False, "jeton_ad": "", "dar_kafa": 1, "kopru_kayip": 0.0,
    # tam_kayip 16 Eylul'de eklendi; ondan onceki butun kosular SORU-CEVAP
    # kaybiyla egitildi -> kapali.
    "tam_kayip": False,
    # belge_pay 16 Eylul'de eklendi; ondan onceki butun kosularda satir
    # basina TEK olgu vardi -> kapali.
    "belge_pay": 0.0,
    # ek_kip/bicim 16 Eylul'de eklendi; ondan onceki butun kosular TEK
    # bicimde ve eksiz kodlamayla egitildi -> kapali.
    "ek_kip": "", "bicim": 1,
}


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
    ood: list = dc.field(default_factory=list)
    #   WANG'IN test_inferred_OOD'si: zincirin IKI kenari da atomic_OOD.
    #   Yani ne birinci hop ne ikinci hop, egitimdeki HICBIR zincirde
    #   gecmiyor. Bas varlik ise BASKA kenarlariyla egitimde zincir basi
    #   OLMUS (Wang'da da oyle) -- tutulan sey VARLIK degil KENAR.
    #   ood_pay=0 ise BOS kalir.
    ent_kati: list = dc.field(default_factory=list)
    #   varlik HICBIR egitim zincirinde gecmemis -- ne bas, ne kopru, ne
    #   cevap. Yalniz atomik olgularda var. Wang 2405.15071'in OOD'si bu;
    #   orada transformer %0 aliyor (22 milyon adimda bile). `ent`ten FARKI:
    #   ENT varliklari kopru ve cevap olarak egitimde GORUNUYOR.
    #   kati_pay=0 ise BOS kalir ve hicbir sey degismez.

    par: np.ndarray | None = None      # (n_ent, 2) jeton ciftleri
    #   jeton_ad=False ise None ve hicbir sey degismez.
    par_ad: tuple = ()                 # (yuva1 adlari, yuva2 adlari)
    t_len: int = 0                     # veri_kur doldurur (ayar.t_len)
    ek_kip: str = ""                   # "" | "tr"  (ayar.ek_kip)

    def __post_init__(self):
        self.n_ent, self.n_rel = self.facts.shape
        self.ent_off = SPECIAL + self.n_rel
        if self.par is None:
            self.yuva, self.vocab = 1, self.ent_off + self.n_ent
            self.p1_off = self.p2_off = self.ent_off
            self.n1 = self.n2 = self.n_ent
            # `yuva_ara` BURADA DA kurulur. Yoksa `dogruluk()` tek jetonlu
            # kollarda AttributeError ile duserdi -- 16 Eylul, model_b6
            # yamasinda gozden kacti, duman testinde yakalandi. Tek kaynak
            # olsun diye TEK ELEMANLI liste; sart yazmaya gerek kalmiyor.
            self.paylasilan = True
            self.yuva_ara = [(self.ent_off, self.vocab)]
        else:
            # IKI AYRIK BLOK -> yuva basina KISITLI argmax temiz kalir.
            self.yuva = self.par.shape[1]
            # PAYLASILAN sozluk ("tam") -> butun yuvalar AYNI blok.
            # AYRIK sozluk ("ilk")     -> yuva basina ayri blok.
            _ayni = all(x is self.par_ad[0] or x == self.par_ad[0]
                        for x in self.par_ad)
            self.paylasilan = _ayni
            if _ayni:
                n = len(self.par_ad[0])
                self.yuva_ara = [(self.ent_off, self.ent_off + n)] * self.yuva
                self.vocab = self.ent_off + n
            else:
                o, self.yuva_ara = self.ent_off, []
                for ad_ in self.par_ad:
                    self.yuva_ara.append((o, o + len(ad_))); o += len(ad_)
                self.vocab = o
            (self.p1_off, _h1) = self.yuva_ara[0]
            (self.p2_off, _h2) = self.yuva_ara[min(1, self.yuva - 1)]
            self.n1, self.n2 = _h1 - self.p1_off, _h2 - self.p2_off
        # EK ISARETLEYICILERI SOZLUGUN SONUNA. Boylece REL_OFF, ent_off
        # ve yuva_ara HIC KAYMAZ -- ek_kip kapaliyken uretilen diziler
        # BIT AYNI kalir, eski kosular gecerliligini korur.
        self.ek0 = 0
        if self.ek_kip:
            assert self.ek_kip == "tr", f"ek_kip: {self.ek_kip!r}"
            self.ek0 = self.vocab
            self.vocab += 4            # '  <NIN>  <SI>  <DIR>
        if not self.t_len:
            self.t_len = 11 if self.par is not None else T_LEN
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
            | {e for e, *_ in self.ent_arama} \
            | {e for e, *_ in self.ent_kati}
        # KATI varliklari da paydadan DUSER: Wang'in atomicID tanimi
        # "OOD varliklarinin olgularini dislar" diyor ve KATI tam
        # olarak onun OOD'si. Dusurulmezse wang_phi KUCUK gorunur.
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
    # HANGI GRAF -- modul adi AYARDAN geliyor, sabit degil. "veri_okul"
    # (1060 varlik) ya da "veri_okul2" (2120). Modul `kur`/`zincirler`/
    # `TIPLER`/`ILISKI` sozlesmesini saglamak zorunda; saglamazsa burada
    # AttributeError verir, sessizce yanlis veri kurmaz.
    _V = importlib.import_module(ayar.veri_ad)
    for _g in ("kur", "zincirler", "TIPLER", "ILISKI"):
        assert hasattr(_V, _g), f"{ayar.veri_ad} modulunde {_g} yok"

    G = _V.kur(ayar.veri_tohum)
    zin = _V.zincirler(G)
    E = [a for t in _V.TIPLER for a in G["ad"][t]]

    # --- IKI JETONLU KODLAMA (ayar.jeton_ad) ----------------------------
    # ILK alt cizgiden bolunur, cunku anlamli olan o:
    #   Ayse_Yilmaz       -> (Ayse, Yilmaz)        ad + soyad
    #   Ankara_Fen_Lisesi -> (Ankara, Fen_Lisesi)  sehir + tur
    #   Ankara            -> (Ankara, <YOK>)
    # Soyadi PAYLASIMI zaten var (veri_okul: "cocuk/kardes/anne/baba AYNI
    # soyadi tasir"), yani hicbir jeton TEK BASINA kisiyi belirlemiyor.
    _par = _par_ad = None
    if ayar.jeton_ad == "tam":
        # DOGRU KODLAMA: butun alt cizgiler, TEK PAYLASILAN sozluk.
        _yuva = max(len(a.split("_")) for a in E)
        _pl = sorted({p for a in E for p in a.split("_")})
        _ix = {p: i + 1 for i, p in enumerate(_pl)}      # 0 = <YOK>
        _par = np.array([[_ix.get(p, 0) for p in
                          (a.split("_") + ["<YOK>"] * _yuva)[:_yuva]]
                         for a in E], np.int64)
        _sz = ("<YOK>",) + tuple(_pl)
        _par_ad = (_sz,) * _yuva                         # AYNI sozluk, her yuva
        assert not any("_" in p for p in _pl), "jeton icinde ALT CIZGI kaldi"
        assert len({tuple(r) for r in _par}) == len(E),             "BIREBIR DEGIL -- ayni jeton dizisi birden cok varliga denk"
        _coz = lambda r: "_".join(_sz[i] for i in r if i)
        _kt = [(E[i], _coz(_par[i])) for i in range(len(E))
               if _coz(_par[i]) != E[i]]
        assert not _kt, f"GIDIS-DONUS BOZUK: {_kt[:3]}"
        yaz(f"  jeton_ad=tam: {len(E)} varlik -> {_yuva} yuva, "
            f"PAYLASILAN sozluk {len(_sz)} jeton "
            f"(tek jetonda {len(E)} idi)")
    elif ayar.jeton_ad == "ilk":
        _ik = [(a.split("_", 1) + ["<YOK>"])[:2] for a in E]
        _y1 = sorted({p[0] for p in _ik})
        _y2 = sorted({p[1] for p in _ik})
        _i1 = {a: i for i, a in enumerate(_y1)}
        _i2 = {a: i for i, a in enumerate(_y2)}
        _par = np.array([[_i1[p[0]], _i2[p[1]]] for p in _ik], np.int64)
        _par_ad = (tuple(_y1), tuple(_y2))
        # BIREBIRLIK: iki jeton bir varligi TEK SEKILDE belirlemeli, yoksa
        # "dogru cevap" tanimsiz olurdu.
        assert len({tuple(p) for p in _par}) == len(E), \
            "IKI JETON birebir DEGIL -- ayni cift birden cok varliga denk"
        yaz(f"  jeton_ad ACIK: {len(E)} varlik -> yuva1 {len(_y1)}  "
            f"yuva2 {len(_y2)}  (tek jetonda {len(E)} idi)")
    # E, TIP SIRASIYLA kuruluyor -- tip dizisi AYNI comprehension'dan
    # cikarilir ki iki yerde iki siralama olmasin.
    E_tip = np.array([i for i, t in enumerate(_V.TIPLER)
                      for _ in G["ad"][t]], np.int64)
    R = list(_V.ILISKI)
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
    for t in _V.TIPLER:                       # TABAKALI: tek tip secilirse
        a = [x for x in G["ad"][t] if x in bas]   # sinav o tipin karisimina
        if a:                                     # indirgenir
            k = int(round(len(a) * ayar.ent_pay))
            ent_ad |= {a[int(i)] for i in rng.permutation(len(a))[:k]}

    # KATI GRUBU -- ENT'ten AYRIK secilir. Bu varliklar egitim zincirinde
    # HICBIR ROLDE gorunmeyecek (bas, kopru, cevap). ENT ise yalniz bas
    # olmuyor. Tabakalama ENT ile AYNI: tek tip secilirse sinav o tipin
    # karisimina indirgenir.
    kati_ad = set()
    if ayar.kati_pay > 0:
        for t in _V.TIPLER:
            a = [x for x in G["ad"][t] if x in bas and x not in ent_ad]
            if a:
                k = int(round(len(a) * ayar.kati_pay))
                kati_ad |= {a[int(i)] for i in rng.permutation(len(a))[:k]}
        assert not (kati_ad & ent_ad), "KATI ve ENT gruplari ORTUSUYOR"

    # ARAMA / HUKUM ayrimi VARLIK duzeyinde: ayni varligin baska bir zinciri
    # de sizinti sayilir. Olculdu: ayrim olmadan hukum kumesinin %24'u
    # aramada zaten gorulmustu.
    _ea = sorted(ent_ad)
    _k = max(1, int(round(len(_ea) * ayar.arama_pay)))
    _ix = rng.permutation(len(_ea))
    arama_ad = {_ea[int(i)] for i in _ix[:_k]}
    hukum_ad = ent_ad - arama_ad

    # OOD KENARLARI -- rng'den EN SON cekilir. Onceki cekilislerin
    # (ent_ad, kati_ad, arama_ad) sirasi DEGISMEMELI; degisirse ood_pay=0
    # olsa bile eski kollarin verisi kayar. `test_sabit` bunu dogruluyor.
    ood_k = set()
    if ayar.ood_pay > 0:
        _kn = [(e, r) for e in E for r in R if facts[eid[e], rid[r]] >= 0]
        _k = int(round(len(_kn) * ayar.ood_pay))
        ood_k = {_kn[int(i)] for i in rng.permutation(len(_kn))[:_k]}

    def _ood(x):
        """Zincirin kac kenari atomic_OOD'de?  x = (e, r1, r2, kopru, cevap)"""
        return ((x[0], x[1]) in ood_k) + ((x[3], x[2]) in ood_k)

    tr2, comp, ent_ay, ent_yk, ent_ar, ent_kt, ood_ay = [], [], [], [], [], [], []
    for e in E:                                # E sirasi SABIT -> tekrarlanabilir
        lst = bas.get(e)
        if not lst:
            continue
        if e in kati_ad:                       # HICBIR ROLDE egitimde yok
            for x in lst:
                if x[6] in ("AYIRT", "YOK") and _ood(x) == 0:
                    ent_kt.append(x)
            continue                           # AYNI / DONUS: duser
        if e in ent_ad:
            hedef = ent_ay if e in hukum_ad else ent_ar
            for x in lst:
                if _ood(x):                    # OOD kenarli: ENT'e girmez
                    continue
                if x[6] == "AYIRT":
                    hedef.append(x)
                elif x[6] == "YOK" and e in hukum_ad:
                    ent_yk.append(x)
            continue                           # AYNI / DONUS: duser
        # WANG'IN UC YOLU (§2, §3.1):
        #   iki kenar da OOD -> test_inferred_OOD
        #   bir kenar OOD    -> KARISIK: ne egitim ne sinav, DUSER
        #   iki kenar da ID  -> normal (tr2 / comp)
        # `lst_id` ood_pay=0 iken `lst`in KENDISI olur ve rng akisi
        # birebir korunur -- eski kollarin verisi degismez.
        lst_id = []
        for x in lst:
            d = _ood(x)
            if d == 2:
                if x[6] in ("AYIRT", "YOK"):
                    ood_ay.append(x)
            elif d == 0:
                lst_id.append(x)
        p = rng.permutation(len(lst_id))
        k = int(round(len(lst_id) * (1.0 - ayar.comp_pay)))
        for i, j in enumerate(p):
            x = lst_id[int(j)]
            if i < k:
                tr2.append(x)
            elif x[6] in ("AYIRT", "YOK"):
                comp.append(x)                 # AYNI/DONUS sinava girmez

    # KATI varligi KOPRU ya da CEVAP olarak da gecmemeli -- bolmenin
    # tanimi "hicbir rolde yok". Bas konumunu yukaridaki `continue`
    # hallediyor; kalan iki konum BURADA siliniyor.
    # OLCULDU (15 Eylul): kati_pay=0.05'te bu, egitim zincirlerinin
    # %9,3'unu goturuyor. YERINE KOYULMUYOR -- bu kolun kiyasi KOSU ICI
    # (comp vs ent vs ent_kati, ayni model, ayni adim, ayni phi), o yuzden
    # phi'nin baska kollarla eslesmesi GEREKMIYOR. Dolgu zincir eklemek
    # egitim dagilimini bozardi (elde kalan havuz AYNI/DONUS turunden,
    # yani TRIVIAL zincirler).
    if kati_ad:
        _n0 = len(tr2)
        tr2 = [x for x in tr2 if x[3] not in kati_ad and x[4] not in kati_ad]
        # COMP ve ENT sinavlari da KATI'den arindirilir: yoksa o orneklerin
        # bir kismi gizliden ent_kati olur ve UC GRUBUN KARSITLIGI bulanir.
        comp = [x for x in comp if x[3] not in kati_ad and x[4] not in kati_ad]
        ent_ay = [x for x in ent_ay if x[3] not in kati_ad and x[4] not in kati_ad]
        ent_yk = [x for x in ent_yk if x[3] not in kati_ad and x[4] not in kati_ad]
        ent_ar = [x for x in ent_ar if x[3] not in kati_ad and x[4] not in kati_ad]
        yaz(f"  KATI: {len(kati_ad)} varlik hicbir zincirde yok. "
            f"egitim zinciri {_n0} -> {len(tr2)} "
            f"(-{_n0 - len(tr2)}, %{100*(_n0-len(tr2))/max(1,_n0):.1f})")

    say = lambda L: [(eid[x[0]], rid[x[1]], rid[x[2]], eid[x[3]], eid[x[4]])
                     for x in L]
    v = Veri(facts=facts, one=[(eid[e], rid[r], eid[h])
                               for (e, r), h in G["olgu"].items()],
             tr2=say(tr2), comp=say(comp), ent=say(ent_ay),
             ent_yok=say(ent_yk), ent_arama=say(ent_ar),
             tip=E_tip, tip_ad=tuple(_V.TIPLER), ent_kati=say(ent_kt),
             ood=say(ood_ay), par=_par, par_ad=_par_ad or (),
             t_len=ayar.t_len, ek_kip=ayar.ek_kip)

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
    # KATI: BOLMENIN TANIMI BU. Bas/kopru/cevap UC KONUMDA da denetlenir --
    # `ent` icin yalniz bas denetleniyor, cunku orada kopru/cevap SERBEST.
    if kati_ad:
        _kid = {eid[a] for a in kati_ad}
        _k = sum(1 for e, _, _, b, c in v.tr2
                 if e in _kid or b in _kid or c in _kid)
        assert _k == 0, f"KATI varligi egitim zincirinde gecti: {_k}"
        assert v.ent_kati, "kati_pay > 0 ama ent_kati BOS"
        _kt = {e for e, *_ in v.ent_kati}
        assert _kt <= _kid, "ent_kati'de KATI olmayan varlik var"
        assert not (_kt & (_h | _a)), "KATI ile ENT gruplari ORTUSUYOR"
        # Atomik olgular DURMALI -- Wang da atomicOOD'yi egitimde tutuyor.
        _ko = sum(1 for e, _, _ in v.one if e in _kid)
        assert _ko > 0, "KATI varliklarinin atomik olgulari da silinmis"
        _sz = {(e, a, b) for e, a, b, _, _ in v.tr2}
        _l = sum((e, a, b) in _sz for e, a, b, _, _ in v.ent_kati)
        assert _l == 0, f"ENT-KATI sizintisi: {_l}"

    if ood_k:
        _ok = {(eid[e], rid[r]) for e, r in ood_k}
        # 1) OOD kenari HICBIR egitim zincirinde, HICBIR hop'ta gecmemeli
        _x = sum(1 for e, r1, r2, b, _ in v.tr2
                 if (e, r1) in _ok or (b, r2) in _ok)
        assert _x == 0, f"OOD kenari egitim zincirinde gecti: {_x}"
        # 2) SINAV zincirinin IKI kenari da OOD olmali
        _y = sum(1 for e, r1, r2, b, _ in v.ood
                 if not ((e, r1) in _ok and (b, r2) in _ok))
        assert _y == 0, f"ood bolmesinde iki kenari OOD olmayan: {_y}"
        # 3) ASIL MEKANIZMA (Wang §3.3): ikinci hop kenari egitimde IKINCI
        #    HOP olarak gecmemeli. `ent_kati`de bu %93,7 geciyordu -- yani
        #    o bolme makalenin %0'ini ureten kosulu SAGLAMIYORDU.
        _ik = {(b, r2) for _, _, r2, b, _ in v.tr2}
        _z = sum(1 for _, _, r2, b, _ in v.ood if (b, r2) in _ik)
        assert _z == 0, f"OOD 2. hop kenari egitimde 2. HOP olarak gecti: {_z}"
        # 4) ATOMIK OLGULAR DURMALI (Wang §2: "all the atomic facts")
        _ao = sum(1 for e, r, _ in v.one if (e, r) in _ok)
        assert _ao == len(_ok), f"OOD atomik olgulari silinmis: {_ao}/{len(_ok)}"
        _bs = {e for e, *_ in v.tr2}
        _hb = sum(1 for e, *_ in v.ood if e in _bs)
        yaz(f"  OOD: {len(_ok)} kenar atomic_OOD (%{100*ayar.ood_pay:.1f}). "
            f"sinav {len(v.ood)} zincir. "
            f"bas varlik egitimde BASKA zincirlerde bas olmus: "
            f"{_hb}/{len(v.ood)} (%{100*_hb/max(1,len(v.ood)):.0f})")
    yaz(f"  veri: olgu {len(v.one)}  egitim2 {len(v.tr2)}  COMP {len(v.comp)}  "
        f"ENT {len(v.ent)}  ENT-YOK {len(v.ent_yok)}  "
        f"ENT-ARAMA {len(v.ent_arama)}"
        + (f"  ENT-KATI {len(v.ent_kati)}" if v.ent_kati else "")
        + (f"  OOD {len(v.ood)}" if v.ood else "")
        + f"  phi {v.phi:.2f}")
    yaz(f"        n_ent {v.n_ent}  n_rel {v.n_rel}  vocab {v.vocab}  "
        f"ent_off {v.ent_off}")
    yaz(f"        phi {v.phi:.2f} (bizim tanim)   {v.wang_phi:.2f} (Wang tanimi, "
        f"payda atomicID)   -- TURETILMIS, ayar DEGIL")

    # BOLMELER NE KADAR YENI -- her kosuda BASILIR, bir daha etiket kaymasin.
    # 15 Eylul'de comp'a "gorulmemis iliski cifti" deniyordu; olculunce
    # ciftlerin TAMAMI egitimde cikti. Sayi gozukurse iddia kayamaz.
    _tc = {(a, b) for _, a, b, _, _ in v.tr2}
    _tv = {e for e, *_ in v.tr2}
    for _ad, _L in (("COMP", v.comp), ("ENT", v.ent), ("ENT-YOK", v.ent_yok),
                    ("ENT-KATI", v.ent_kati), ("OOD", v.ood)):
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
def _bos(n, v=None):
    return np.zeros((n, v.t_len if v is not None else T_LEN), np.int64)


def _e(v, e):
    """Varligin jeton(lar)i. jeton_ad kapaliysa tek elemanli liste.
    `yuva_ara` TEK KAYNAK -- paylasilan sozlukte butun yuvalar ayni
    blogu gosterir, ayrik sozlukte her yuva kendi blogunu."""
    if v.par is None:
        return [v.ent_off + e]
    return [v.yuva_ara[j][0] + int(v.par[e, j]) for j in range(v.yuva)]


def _ekler(v):
    """(KESME, NIN, SI, DIR) -- sozlugun SONUNA eklendi (Veri.__post_init__).
    Eski jeton id'leri KAYMADI."""
    assert v.ek_kip == "tr", "ek_kip kapali, ek isaretleyici YOK"
    return v.ek0, v.ek0 + 1, v.ek0 + 2, v.ek0 + 3


def kodla_1hop(v: Veri, batch, bicim_no: int = 0):
    """[Q1] e r ? cevap EOS   -> hedef pozisyon 3
    jeton_ad: [Q1] e1 e2 r ? a1 a2 EOS -> hedefler 4 ve 5.

    ek_kip="tr": rolu POZISYON degil EK tasir, o yuzden SIRA DEGISEBILIR.
        0  Ayse Yilmaz <YOK> ' <NIN> anne <SI> ? Fatma Yilmaz <YOK> ' <DIR> EOS
        1  anne <SI> Ayse Yilmaz <YOK> ' <NIN> ? Fatma Yilmaz <YOK> ' <DIR> EOS
        2  Ayse Yilmaz <YOK> ' <NIN> anne <SI> Fatma Yilmaz <YOK> ' <DIR> EOS
           ^ bildirim: soru isareti YOK

    Ucunde de CEVAP SONDA. "Fatma'dir Ayse'nin annesi" gibi cevap-basta
    bir bicim KURULMADI: o dizide cevap hicbir seyden turemiyor, yani
    cevap pozisyonu bir sey OLCMUYOR olurdu.
    """
    X = _bos(len(batch), v)
    P, T = [], []
    for i, (e, r, a) in enumerate(batch):
        ez, az = _e(v, e), _e(v, a)
        if not v.ek_kip:
            dz = [Q1] + ez + [REL_OFF + r, QM] + az + [EOS]
            a0 = 3 + v.yuva                  # ILK cevap jetonunun yeri
        else:
            K, N, S, D = _ekler(v)
            oz, il = ez + [K, N], [REL_OFF + r, S]
            on = {0: oz + il + [QM], 1: il + oz + [QM], 2: oz + il}[bicim_no % 3]
            dz = on + az + [K, D, EOS]
            a0 = len(on)
        assert len(dz) <= v.t_len, (len(dz), v.t_len, bicim_no)
        X[i, :len(dz)] = dz
        # next-token: a0'daki jeton a0-1'den tahmin edilir
        P.append(list(range(a0 - 1, a0 - 1 + v.yuva)))
        T.append(az)
    return X, np.array(P, np.int64), np.array(T, np.int64)


def kodla_2hop(v: Veri, batch, bicim_no: int = 0):
    """[Q2] e r1 r2 ? cevap EOS   -> hedef pozisyon 4
    jeton_ad: [Q2] e1 e2 r1 r2 ? a1 a2 EOS -> hedefler 5 ve 6.

    ek_kip="tr": "Ayse Yilmaz'in cocugunun kardesi ...dir"
        cocugunun = cocuk <SI> <NIN>   -- zincir DIZIDE isaretli
        0  e ' <NIN> r1 <SI> <NIN> r2 <SI> ? a ' <DIR> EOS      (KANONIK)
        1  r2 <SI> r1 <SI> <NIN> e ' <NIN> ? a ' <DIR> EOS      (devrik)
        2  e ' <NIN> r1 <SI> <NIN> r2 <SI> a ' <DIR> EOS        (bildirim)

    !! OLCME HEP bicim 0 ile yapilir (onkayit). Egitim `ayar.bicim`
    kadar bicim gorur; sinav TEK bicimdir, yoksa "cesitlilik ogretti mi"
    sorusu "cesitlilikle mi sinandi" sorusuna karisirdi.
    """
    X = _bos(len(batch), v)
    P, T = [], []
    for i, (e, r1, r2, _b, a) in enumerate(batch):
        ez, az = _e(v, e), _e(v, a)
        if not v.ek_kip:
            dz = [Q2] + ez + [REL_OFF + r1, REL_OFF + r2, QM] + az + [EOS]
            a0 = 4 + v.yuva
        else:
            K, N, S, D = _ekler(v)
            oz = ez + [K, N]
            i1, i2 = [REL_OFF + r1, S], [REL_OFF + r2, S]
            on = {0: oz + i1 + [N] + i2 + [QM],
                  1: i2 + i1 + [N] + oz + [QM],
                  2: oz + i1 + [N] + i2}[bicim_no % 3]
            dz = on + az + [K, D, EOS]
            a0 = len(on)
        assert len(dz) <= v.t_len, (len(dz), v.t_len, bicim_no)
        X[i, :len(dz)] = dz
        P.append(list(range(a0 - 1, a0 - 1 + v.yuva)))
        T.append(az)
    return X, np.array(P, np.int64), np.array(T, np.int64)


def kodla_belge(v: Veri, batch):
    """BELGE: zincirlenen IKI atomik olgu AYNI DIZIDE.

        [S1] e r1 ? b <EOS>  [S1] b r2 ? a <EOS>

    Gercek metinde "Ayse'nin cocugu Fatma. Fatma'nin kardesi Emre."
    ayni paragraftadir. Bizde hicbir dizi iki olguyu birden tasimiyordu.
    YENI OLGU YOK -- ikisi de `one` icinde ZATEN var; eklenen tek sey
    BITISIKLIK.

    KAYIP: butun pozisyonlarda (next-token). Yani bu kodlayici ancak
    `tam_kayip` ile ANLAMLI -- yoksa belgenin ortasi hic ogrenilmez.
    `egitim_havuzu` bunu assert ile denetler.

    P/T doner ama ikinci olgunun CEVAP yuvalarini gosterir: `tam_kayip`
    kapaliyken bile satir CÖP olmasin diye. (Kolun kendisi tam_kayip
    ACIK kosuyor; bu yalniz saglamlik.)
    """
    X = _bos(len(batch), v)
    P, T = [], []
    for i, (e, r1, r2, b, a) in enumerate(batch):
        ez, bz, az = _e(v, e), _e(v, b), _e(v, a)
        d1 = [Q1] + ez + [REL_OFF + r1, QM] + bz + [EOS]
        d2 = [Q1] + bz + [REL_OFF + r2, QM] + az + [EOS]
        dz = d1 + d2
        assert len(dz) <= v.t_len, (
            f"belge {len(dz)} jeton, t_len {v.t_len} -- t_len TURETIMI YANLIS")
        X[i, :len(dz)] = dz
        p0 = len(d1) + 2 + v.yuva          # 2. olgunun QM pozisyonu
        P.append(list(range(p0, p0 + v.yuva)))
        T.append(az)
    return X, np.array(P, np.int64), np.array(T, np.int64)


def kodla_kimlik_q1(v: Veri, ents):
    """[Q1] e IDENT ? e EOS  -> hedef = varligin KENDISI (SIFIR-HOP).

    DUZELTME (15 Eylul hakemligi): bunu daha once "ise yaramaz kontrol" diye
    anlatmistim. YANLIS. arXiv 2509.24653'un onerdigi identity bridge TAM
    BUDUR -- birebir: "a zero-hop SELF-MAPPING for each bridge token" ve
    "an identity mapping on bridge tokens". Yani e -> e.

    Bilinen itiraz (arsiv DENEY5): cevap girdide duruyor, kopyalamayla
    cozulebilir. Makale bunu bilerek yapiyor; iddiasi, gizli durumu token
    gommesiyle HIZALAMAYA zorlamasi.

    COK JETONA ACILDI (16 Eylul, kullanici karari). Onceden burada
    `assert v.par is None` vardi -- yani identity bridge TAM DA bizim
    rejimimizde (cok jetonlu varlik) KAPALIYDI ve hic kosulmadi.

        tek jeton   [Q1] e        IDENT ? e        EOS    6 jeton
        cok jeton   [Q1] e1 e2 e3 IDENT ? e1 e2 e3 EOS   10 jeton

    t_len 11'e SIGIYOR, yani bu kol `t_len`e DOKUNMAZ.

    NEDEN -- kahin testimiz (belge/bulgu/kahin_testi.md) makalenin
    teshisini DOGRULADI. arXiv 2509.24653 4.1, birebir:
      "The result shows that the model completely fails to decode c on
       OOD two-hop reasoning, which indicates token b doesn't bridge
       the gap between first hop and second hop."
      "We attribute this failure to a contextual disconnect between the
       input and output spaces. The model is not explicitly required to
       establish an equivalence between the input token b and the
       output token b, which is a trivial capability for well-trained
       LLMs."
      "a straightforward solution is to augment the training data with
       b -> b 'zero-hop' sequences, which we term as identity bridge."

    GRAF BILGISI KULLANMIYOR: ne olgu, ne cevap, ne kopru. Yalniz
    varligin KENDI jetonlari. `kopru_kayip`tan (model_b8) farki BU.
    """
    ents = list(ents)
    X = _bos(len(ents), v)
    P, T = [], []
    p0 = 2 + v.yuva                          # QM'nin pozisyonu
    for i, e in enumerate(ents):
        ez = _e(v, e)
        dz = [Q1] + ez + [IDENT, QM] + ez + [EOS]
        assert len(dz) <= v.t_len, (
            f"kimlik dizisi {len(dz)} jeton, t_len {v.t_len}")
        X[i, :len(dz)] = dz
        P.append(list(range(p0, p0 + v.yuva)))
        T.append(ez)
    return X, np.array(P, np.int64), np.array(T, np.int64)


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
        assert v.par is None, "kimlik kollari jeton_ad ile KOSULMADI"
        X[i, :7] = [Q2, v.ent_off + e, REL_OFF + r1, IDENT, QM,
                    v.ent_off + b, EOS]
    return X, np.full((len(batch), 1), 4, np.int64), \
        np.array([[v.ent_off + b] for _, _, b in batch], np.int64)


def kopru_hedefi(v: Veri):
    """(pozisyonlar, kac token). Kopru, r1 ve r2 pozisyonlarindan okunur.

        [S2] e1 e2 e3 r1 r2 ?  a1 a2 a3 EOS
                      ^^ ^^              <- BURASI BOS: ana kayip
                                            cevabi 3+yuva'dan okuyor
    Iki pozisyon var, o yuzden en fazla IKI token. Ucuncu yuva zaten
    cogunlukla <YOK> dolgusu (olculdu: %94,7), bilgi tasimiyor.

    ek_kip="tr", bicim 0:
        e1 e2 e3 ' <NIN> r1 <SI> <NIN> r2 <SI> ? ...
                          ^^         ^^
    !! BICIM 0'A gore. Baska bicimde r1/r2 baska yerde -- o yuzden
    `egitim_havuzu` bicim>1 iken kopru_kayip'i REDDEDIYOR."""
    if v.ek_kip:
        return [v.yuva + 2, v.yuva + 5][:min(v.yuva, 2)], min(v.yuva, 2)
    return [1 + v.yuva, 2 + v.yuva][:min(v.yuva, 2)], min(v.yuva, 2)


def egitim_havuzu(ayar: Ayar, v: Veri, yaz=print):
    """1hop + 2hop (+ istege bagli kimlik gorevi) -> tek havuz."""
    _bic = max(1, ayar.bicim)
    assert _bic == 1 or ayar.ek_kip, (
        "bicim>1 ek_kip GEREKTIRIR: eksiz dilde sirayi degistirmek ANLAMI "
        "BOZAR ('Fatma anne Ayse' != 'Ayse anne Fatma').")
    assert _bic <= 3, f"kurulu bicim sayisi 3, istenen {_bic}"
    assert not (_bic > 1 and ayar.kopru_kayip > 0), (
        "bicim>1 + kopru_kayip: kopru pozisyonu bicimden bicime DEGISIYOR, "
        "kopru_hedefi tek bir pozisyon listesi donuyor -> hedef YANLIS satira "
        "duser.")
    _kp, _nk = kopru_hedefi(v)
    parca, _kt = [], []
    # !! BICIM CESITLILIGI YALNIZ OLGU (1-hop) SATIRLARINA. Iki sebep:
    #
    # 1) MEKANIZMA. Physics of LM 3.1'in olctugu sey ENTITENIN BILGISININ
    #    kac farkli ifadeyle gecdigi ("knowledge augmentation"); soru
    #    tarafi ayri bir sey. Bizim semptomumuz da tam orada: ezber tam
    #    (one/seen 1.0000) ama dogrusal sonda BOS -- yani olgu
    #    ezberlenmis, KODLANMAMIS.
    #
    # 2) BEDEL. Her sey 3 bicimde uretilseydi havuz 198.688 -> 593.944
    #    olurdu (3x). model_b14 bunun bedelini OLCTU: havuz 1,7 katina
    #    cikinca 20.000 adim yetmedi, 60.000'e uzatildi. Yalniz 1-hop
    #    cogaltilinca havuz ~%27 buyuyor.
    #
    # Sinav zaten bicim 0 ile yapiliyor; 2-hop'u da tek bicimde tutmak
    # egitim/sinav BICIM UYUSMAZLIGINI da ortadan kaldiriyor.
    for _b in range(_bic):
        parca.append(kodla_1hop(v, v.one, _b))
        _kt.append(np.full((len(v.one), _nk), -1, np.int64))
        parca.append(kodla_2hop(v, v.tr2, _b))
        _kt.append(np.array([_e(v, x[3])[:_nk] for x in v.tr2], np.int64))
    _n_tab = sum(len(a) for a, _, _ in parca)
    if _bic > 1:
        yaz(f"  BICIM CESITLILIGI: {_bic} yuzey bicimi -- OLGU ve SORU "
            f"satirlarinin IKISINDE de")
        yaz(f"     (1hop {len(v.one)} + 2hop {len(v.tr2)}) x{_bic} = {_n_tab}")
        yaz("     SINAV hep bicim 0 -- egitim daha cok yuzey gorur, sinav TEK.")
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
        tekrar = max(1, int(round(ayar.ident_frac / max(1e-9, 1 - ayar.ident_frac)
                                  * _n_tab / len(kimlik[0]))))
        parca.append(tuple(np.tile(z, (tekrar,) + (1,) * (z.ndim - 1))
                           for z in kimlik))
        _kim_satir = len(parca[-1][0])
        _kt.append(np.full((_kim_satir, _nk), -1, np.int64))
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
    # --- BELGE SATIRLARI (belge_pay > 0 ise) ----------------------------
    if ayar.belge_pay > 0:
        assert ayar.tam_kayip, (
            "belge_pay tam_kayip GEREKTIRIR: kayip yalniz cevap yuvalarinda "
            "hesaplanirsa belgenin ORTASI (kopru) hic ogrenilmez ve kol "
            "hicbir sey olcmez.")
        # !! SIZINTI DENETIMI: belgeler YALNIZ tr2'den. Sinav
        # zincirlerinden kurulsaydi o zincirin koprusu BAGLAMA yazilmis
        # olurdu -- yani cevabi elden vermis olurduk.
        _sinav = {(x[0], x[1], x[2]) for lst in
                  (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood)
                  for x in lst}
        _kaynak = [x for x in v.tr2]
        _sizan = [x for x in _kaynak if (x[0], x[1], x[2]) in _sinav]
        assert not _sizan, (
            f"!! SIZINTI: {len(_sizan)} belge SINAV zincirinden kurulacakti")
        assert not ayar.ek_kip, (
            "belge + ek_kip KURULMADI (kodla_belge'ye ek isaretleyici "
            "eklenmedi)")
        n_tab = _n_tab
        n_bel = int(round(ayar.belge_pay / max(1e-9, 1 - ayar.belge_pay)
                          * n_tab))
        rs_b = np.random.RandomState(ayar.veri_tohum + 7717)
        idx = rs_b.randint(0, len(_kaynak), n_bel)
        belge = kodla_belge(v, [_kaynak[j] for j in idx])
        parca.append(belge)
        # BELGE satirlarinda kopru hedefi YOK: kopru zaten dizide YAZILI,
        # ayrica tahmin ettirmenin anlami yok. -1 -> maskelenir.
        _kt.append(np.full((len(belge[0]), _nk), -1, np.int64))
        yaz(f"  BELGE: {n_bel} satir (havuzun %{100*n_bel/(n_tab+n_bel):.0f}'i)"
            f"  kaynak tr2 ({len(_kaynak)} zincir)  t_len {v.t_len}")
        yaz(f"     sizinti denetimi GECTI: sinav zincirinden belge YOK")

    X = np.concatenate([a for a, _, _ in parca])
    P = np.concatenate([b for _, b, _ in parca])
    T = np.concatenate([c for _, _, c in parca])
    # KOPRU HEDEFI `_kt` yukarida parca ile YAN YANA kuruldu (-1 = bu
    # satirda kopru YOK: 1hop, kimlik, belge -> maskelenir).
    KT = np.concatenate(_kt)
    assert len(KT) == len(X), (len(KT), len(X))
    if ayar.kopru_kayip > 0:
        yaz(f"  KOPRU KAYBI acik: agirlik {ayar.kopru_kayip}  "
            f"pozisyon {_kp}  {_nk} token  "
            f"({int((KT[:, 0] >= 0).sum())}/{len(KT)} satirda gecerli)")
    if kimlik is not None:
        yaz(f"                 havuzun %{100*_kim_satir/len(X):.0f}'i")
    return X, P, T, kimlik, np.array(_kp, np.int64), KT


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
        self.pos = nn.Embedding(ayar.t_len, d)
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
    d = dict(one=alt(v.one, 0), seen=alt(v.tr2, 1), comp=alt(v.comp, 2),
             ent=alt(v.ent, 3), ent_yok=alt(v.ent_yok, 5))
    # ANAHTAR YALNIZ DOLUYSA EKLENIR. Bos liste bile eklense `olcme_izi`
    # DEGISIR ve BUTUN eski kosular "iz tutmuyor" diye olculemez hale
    # gelirdi -- pencere_a egitim izi ile olcme izini karsilastiriyor.
    if v.ent_kati:
        d["ent_kati"] = alt(v.ent_kati, 6)
    if v.ood:
        d["ood"] = alt(v.ood, 8)
    return d


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
    # YUVA BASINA KISITLI argmax. yuva=1 iken eski davranisla AYNI.
    ARA = v.yuva_ara[:P.shape[1]]
    ok = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            lg = model(xb)
        idx = torch.from_numpy(P[i:i + bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)[:, None]
        lgp = lg.float()[ar, idx]                       # (n, yuva, V)
        tb = torch.from_numpy(T[i:i + bs]).to(DEV)
        dogru = None
        for j, (lo, hi) in enumerate(ARA):
            d = lgp[:, j, lo:hi].argmax(-1) == (tb[:, j] - lo)
            dogru = d if dogru is None else (dogru & d)
        ok.append(dogru.cpu().numpy())
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
    # KISAYOL CEVABI da yuva basina kodlanir. `_e` TEK KAYNAK -- burada
    # yeniden turetilmiyor. facts -1 ise ent_off-1 (bir ILISKI token'i)
    # cikar ve hicbir varlik tahminiyle eslesmez; yuva kipinde de oyle
    # olmasi icin -1 ozel olarak ele alinir.
    def _ksy(e, r2):
        h = int(v.facts[e, r2])
        if h < 0:
            return [v.ent_off - 1] * v.yuva      # ESLESMEZ, oran 0.000
        return _e(v, h)
    ksy = np.array([_ksy(e, r2) for e, _, r2, _, _ in lst], np.int64)
    onceki = model.training            # bkz. dogruluk()'taki ayni kusur
    model.eval()
    ARA = v.yuva_ara[:P.shape[1]]
    ok = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            lg = model(xb)
        idx = torch.from_numpy(P[i:i + bs]).to(DEV)
        ar = torch.arange(len(idx), device=DEV)[:, None]
        lgp = lg.float()[ar, idx]                        # (n, yuva, V)
        esit = None
        for j, (lo, hi) in enumerate(ARA):
            p = lgp[:, j, lo:hi].argmax(-1).cpu().numpy() + lo
            d = p == ksy[i:i + bs, j]
            esit = d if esit is None else (esit & d)
        ok.append(esit)
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


def surdurme_yaz(yol, ayar, model, opt, scaler, rs, adim, egri, iz,
                 yavas=None):
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
            # Lookahead YAVAS agirligi. ort_bas=0 iken None -- eski
            # paketlerde bu anahtar hic YOKTU, `surdurme_oku` .get() ile
            # okuyor, yani eski paketler aynen surdurulebilir.
            yavas=(None if yavas is None else
                   {k: v.cpu() for k, v in yavas.items()}),
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
    fark = {k for k in yeni
            if k != "adim" and eski.get(k, ESKI_VARSAYILAN.get(k)) != yeni[k]}
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
    # YAVAS agirlik (Lookahead). Eski paketlerde anahtar YOK -> None.
    # Yeniden kurulurken modelin cihazina tasinir.
    _yav = p.get("yavas")
    if _yav is not None:
        _yav = {k: v.to(DEV) for k, v in _yav.items()}
    yaz(f"  SURDURULUYOR: adim {p['adim']} -> {ayar.adim}   "
        f"({len(p['egri'])} olcum noktasi devralindi)"
        + ("   [yavas agirlik da devralindi]" if _yav is not None else ""))
    return p["adim"], list(p["egri"]), _yav


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
         model_kur=None,   # None -> Model. `model_b` kendi sinifini verir;
         #                   varsayilan davranis BIT DUZEYINDE ayni kalir.
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
    Xtr, Ptr, Ttr, kimlik, KPOZ, KTR = egitim_havuzu(ayar, v, yaz)

    L = olcme_listeleri(ayar, v)
    kod = {k: (kodla_1hop(v, L[k]) if k == "one" else kodla_2hop(v, L[k]))
           for k in L if L[k]}
    iz = olcme_izi(L)
    yaz("  olcme: " + "  ".join(f"{k} {len(L[k])}" for k in L if L[k])
        + f"   parmak izi {iz}")

    torch.manual_seed(ayar.tohum)
    model = (model_kur or Model)(ayar, v.vocab).to(DEV)
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
    # Lookahead'in YAVAS agirligi. ort_bas=0 iken hep None kalir ve hicbir
    # sey olmaz -- eski kollarin davranisi aynen korunur.
    yavas, ort_say = None, 0

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
                  ent_kati=len(v.ent_kati), ood=len(v.ood),
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
        bas, egri, yavas = surdurme_oku(sur_yol, ayar, model, opt, scaler,
                                        rs, iz, yaz)
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
    # YARDIMCI kaybi AYRI say. `kayip` sutunu ikisinin TOPLAMI ve
    # kopru_kayip=0 olan kollarla KIYASLANAMAZ; ayri yazilmazsa
    # "model_b8'in kaybi daha yuksek" diye YANLIS okunurdu.
    yrd_top = torch.zeros((), device=DEV)
    yrd_say = 0
    ana_top = torch.zeros((), device=DEV)
    ana_say = 0
    # KPOZ her adimda tensora ceviriliyordu; bir kez yeter.
    _KPT = torch.from_numpy(KPOZ).to(DEV)
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
                # COSINE, TABANI lr/10 -- nanoGPT / Pythia / Qwen SFT ucu de
                # boyle yapiyor. Taban OLMADAN son adimda lr TAM SIFIR olur
                # ve model fiilen DONAR; `ort_bas` ile birlikte birincil
                # pencere donmus agirliklari ortalar.
                #   nanoGPT train.py:  min_lr + coeff * (learning_rate - min_lr)
                #   nanoGPT GPT-2      min_lr 6e-5   = lr/10
                #   Pythia-70m.yml     min_lr 1e-4   = lr/10  (lr 1e-3)
                #   Qwen2.5 SFT        7e-6 -> 7e-7  = lr/10
                # Kullanici karari, 16 Eylul: "min lr olsun, tabani lr/10
                # olsun." YENI AYAR ALANI DEGIL -- `sabit_lr=False` dali
                # deneme 2'de HIC KOSMADI (olculdu: 25 kosulmus ayar
                # dosyasinin 0'inda cosine), yani hicbir kayitli sonuc
                # degismiyor. `sabit_lr=True` yolu BIT AYNI kalir.
                _alt = ayar.lr / 10.0
                _k = 0.5 * (1 + math.cos(
                    math.pi * (adim - ayar.isinma)
                    / max(1, ayar.adim - ayar.isinma)))
                lr = _alt + _k * (ayar.lr - _alt)
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
                lg_tam = model(xb)
                # xb.shape[0], ayar.batch DEGIL: ikisi burada esit ama bir
                # varyasyon degisken batch kullanirsa `ayar.batch` sessizce
                # yanlis satirlari secerdi.
                # pb/tb artik (B, yuva). yuva=1 iken eski davranisla
                # AYNI: tek pozisyon, tek hedef, ayni kayip.
                _ar = torch.arange(xb.shape[0], device=DEV)[:, None]
                lg = lg_tam[_ar, pb]                   # (B, yuva, V)
                # `ana` HER ZAMAN hesaplanir: model_b6 ile KIYASLANABILIR
                # olan sayi budur. tam_kayip acikken optimize edilen sey
                # `ana` DEGIL, ama egriye ikisi de yazilir. (model_b8'de
                # `kayip` sutunu kirlenmisti ve b6 ile kiyaslanamaz hale
                # gelmisti -- ayni hataya dusmemek icin.)
                ana = F.cross_entropy(
                    lg.float().reshape(-1, lg.shape[-1]), tb.reshape(-1))
                if ayar.tam_kayip:
                    # DIL MODELI KAYBI: pozisyon t, X[t+1]'i tahmin eder.
                    # PAD (=0) hedefleri atlanir; PAD dizinin yalniz
                    # KUYRUGUNDA var, arasinda yok.
                    kayip = F.cross_entropy(
                        lg_tam[:, :-1].float().reshape(-1, lg_tam.shape[-1]),
                        xb[:, 1:].reshape(-1), ignore_index=PAD)
                    ana_top += ana.detach()
                    ana_say += 1
                else:
                    kayip = ana
                # --- YARDIMCI KOPRU KAYBI (kopru_kayip>0 ise) -----------
                # kopru_kayip=0'da bu blok HIC calismaz -> eski kollar
                # BIT DUZEYINDE ayni kalir.
                if ayar.kopru_kayip > 0:
                    kb = torch.from_numpy(KTR[j]).to(DEV)      # (B, nk)
                    m = kb[:, 0] >= 0                          # 2hop satirlar
                    if bool(m.any()):
                        lgk = lg_tam[m][:, _KPT]               # (Bm, nk, V)
                        yrd = F.cross_entropy(
                            lgk.float().reshape(-1, lgk.shape[-1]),
                            kb[m].reshape(-1))
                        # ORTALAMA GECERLI SATIRLAR UZERINDE; ana kayip
                        # BUTUN satirlarda ortalaniyor. Havuzun %82'si
                        # 2hop, yani yardimci terimin FIILI agirligi
                        # kopru_kayip'in ~1,22 KATI. Hata degil ama
                        # "agirlik 1.0" gorunup 1,22 olmasi YANILTIR.
                        kayip = kayip + ayar.kopru_kayip * yrd
                        yrd_top += yrd.detach()
                        yrd_say += 1
            kayip_top += kayip.detach()
            kayip_say += 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(kayip).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # --- GERI BESLEMELI ORTALAMA (Lookahead) -- OLCUMDEN ONCE.
            # Sirasi onemli: once ortala, sonra olc/anlik-goruntu al.
            # Boylece egri, anlik goruntu ve surdurme paketi HEPSI ayni
            # (ortalanmis) agirligi gorur; "hangi agirlik olculdu"
            # sorusu tek cevapli kalir.
            _oh = ayar.ort_her or ayar.olc_her
            if ayar.ort_bas and adim >= ayar.ort_bas and adim % _oh == 0:
                with torch.no_grad():
                    sd_ = model.state_dict()
                    # DIKKAT: dongu degiskenleri `_k`/`_t`. `v` DISARIDA
                    # Veri nesnesi; `for k, v in ...` yazilirsa GOLGELENIR
                    # ve bir sonraki olcumde `v.ent_off` patlar. Duman
                    # testi bunu yakaladi (15 Eylul) -- gercek kosuda ilk
                    # ortalama adiminda cokerdi.
                    if yavas is None:          # ILK esik: phi <- theta
                        yavas = {_k: _t.detach().clone()
                                 for _k, _t in sd_.items()}
                        yaz(f"    ORTALAMA ACILDI adim {adim}  "
                            f"alfa {ayar.ort_alfa}  her {_oh} adim")
                    else:
                        a_ = ayar.ort_alfa
                        for _k, _t in sd_.items():
                            if yavas[_k].dtype.is_floating_point:
                                yavas[_k].mul_(1 - a_).add_(_t.detach(),
                                                            alpha=a_)
                            else:              # sayac/maske gibi alanlar
                                yavas[_k].copy_(_t.detach())
                        model.load_state_dict(yavas)
                        ort_say += 1

            if adim % ayar.olc_her == 0 or adim == ayar.adim:
                r = _nokta(adim, float(kayip_top.item() / max(1, kayip_say)),
                           lr, kayip_son=float(kayip.item()))
                if ayar.kopru_kayip > 0:
                    # AYRI SUTUN: `kayip` = ana + agirlikli yardimci.
                    # `kayip_yrd` yalniz yardimci terim (agirliksiz).
                    # `kayip_ana` = kayip - kopru_kayip * kayip_yrd, yani
                    # model_b6 ile KIYASLANABILIR olan sayi.
                    r["kayip_yrd"] = float(yrd_top.item() / max(1, yrd_say))
                    r["kayip_ana"] = round(
                        r["kayip"] - ayar.kopru_kayip * r["kayip_yrd"], 6)
                if ayar.tam_kayip:
                    # `kayip` = DIL MODELI kaybi (optimize edilen).
                    # `kayip_ana` = yalniz cevap yuvalari -- model_b6'nin
                    # `kayip` sutunuyla AYNI SEY, tek kiyaslanabilir sayi.
                    r["kayip_ana"] = round(
                        float(ana_top.item() / max(1, ana_say)), 6)
                kayip_top = torch.zeros((), device=DEV)
                kayip_say = 0
                yrd_top = torch.zeros((), device=DEV)
                yrd_say = 0
                ana_top = torch.zeros((), device=DEV)
                ana_say = 0
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
                             egri, iz, yavas)
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
