# -*- coding: utf-8 -*-
"""taban_11 — BU KOLUN KENDI motoru. TEK BASINA DURUR.

Kullanici karari, 16 Eylul 2026:
    *"bunlarin hepsi model_09 folderi altinda olmali. model_09 diger
    hicbir model ile ayni seyi kullanmamali."*

Bu dosya `model_a.py`nin KOPYASIDIR ve artik model_09'a aittir.
`model_a`/`model_b` icin yapilan bir degisiklik buraya GECMEZ.

Icindekiler: `Ayar`, `Veri`, `veri_kur`, kodlayicilar, `egitim_havuzu`,
`olcme_listeleri` / `olcme_izi`, `dogruluk`, `kisayol_orani`, anlik
goruntu + surdurme, ve `egit`.

==========================================================================
!! KOPYANIN BEDELI -- ve nasil odendigi

Paylasilan motorda bir olcum hatasi duzeltilirse, o duzeltme buraya
KENDILIGINDEN gelmez; model_09 ile model_b15 O GUNDEN SONRA FARKLI
KODLA olculmus olur. Bu, kopyanin gercek riski ve gozden kacarsa
sayilari sessizce karsilastirilamaz hale getirir.

Bunun icin `test_11.py` her kosuda DAVRANIS ESDEGERLIGI siniyor:

    egitim havuzu       taban_09 vs model_a  -> BIT DUZEYINDE ayni
    olcme listeleri     olcme_izi            -> AYNI
    Ayar alanlari       ESKI_VARSAYILAN      -> AYNI
    kodlayici ciktilari ayni girdi           -> AYNI dizi

Yani divergence SESSIZ olmuyor: kilit duser ve o gun bilerek karar
verilir (kopyayi guncelle, ya da farki onkayda yaz).
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
# NOT: `model_a.py` burada bir ust klasoru yola ekleyip `veri_okul`u
# import ediyordu. `taban_09` bunu YAPMIYOR: model_09'in verisi
# `veri_11.py`, ve o BU klasorde. Veri modulu `veri_kur` icinde
# `ayar.veri_ad`dan import ediliyor (asagida), burada degil.

import jeton_11 as J
import korpus_11 as KOR
import olcme_11 as OLC

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Ozel token'lar. ENT_OFF/VOCAB veriden TURER (Veri.__post_init__), burada
# sabit YAZILMAZ -- arsivde `ENT_OFF` bir kez elle 14 yazilmis, 16'ymis.
# !! OZEL BLOK 8'DEN 3'E INDI -- kullanici karari, 17 Eylul:
#     "bunlar niye var?"  ...  "silinsin tabii ki"
# Olculdu: sekiz ozel jetonun BESI havuzda HIC gecmiyordu.
#     [S1]  [S2]  [KIMLIK]  <KULLANILMIYOR> x2
# [S1]/[S2] ekli olmayan kodlamanin soru isaretcileriydi; gercek eklerle
# gereksizler (tamlayan eki zaten zinciri isaretliyor). [KIMLIK] kimlik
# satirinin isaretcisiydi; o satir artik Turkce yazildigi icin
# ("Ibrahim Yilmaz kimdir?") ona da gerek kalmadi.
#
# Sonuc: `ek_kip=""` (eksiz) ve `ek_kip="tr"` KODLAMA YOLLARI SILINDI --
# Q1/Q2/IDENT sabitleri yalniz onlar icindi. Bu kol yalniz "tr2" kosar.
# Bedeli: `taban_09` artik ortak motorun BIREBIR KOPYASI DEGIL, yani
# `test_09`in model_a kopya-sapma bekcisi BILDIRILMIS AYRISMAYA dondu.
PAD, QM, EOS = 0, 1, 2      # dolgu,  soru isareti '?',  nokta '.'
SPECIAL = 3
REL_OFF = SPECIAL
T_LEN = 8


# ======================= AYAR ============================================
@dc.dataclass(frozen=True)
class Ayar:
    ad: str = "model_11"

    # --- veri (bir ailenin butun kollarinda AYNI olmali, yoksa
    #     'sartlar esit' bozulur ve kollar farkli veri gorur)
    veri_ad: str = "veri_11"     # HANGI GRAF. "veri_okul2" = tam IKI KATI.
    #   15 Eylul'de eklendi. Modul adi olarak yaziliyor ki `ayar_t<N>.json`a
    #   girsin: "bu kosu hangi veriyi gordu" sorusu SONRADAN cevaplanabilsin.
    #   Alan eklemek SURDURMEYI bozabilirdi (eski paketlerde bu anahtar YOK
    #   ve karsilastirma 'degismis' derdi); `surdurme_oku` icinde ESKI
    #   VARSAYILAN tablosu var, oraya bak.
    veri_tohum: int = 0
    # --- KORPUS (model_09) ------------------------------------------
    # `t_len` ARTIK TURETILMIYOR. model_08'de jeton semasindan
    # dusuyordu (2*yuva+11 = 17) cunku her satir TEK cumleydi ve
    # uzunlugu semanin sonucuydu. Burada pencere bir VERI KARARI:
    # metin 512'lik dilimlere bolunuyor, dilim sinirinin cumleyle
    # ilgisi yok. Turetilecek bir sey kalmadi.
    t_len: int = 512           # egitim penceresi (karakter)
    kopya: int = 5             # Physics 3.1 `multiM`: varlik basina M belge
    tetik: int = 0             # bu M belgenin kaci BIYOGRAFI SORUSU onegiyle
    # --- KORPUSUN BILESIMI (model_11). Hepsi AYARDAN gelir ki onkayit
    # ve defter tek yerden okusun; `havuz`un varsayilanina GUVENILMEZ.
    zincir_pay: float = 0.20   # sayfa cumlelerinin ~payi ZINCIR cumlesi
    n3: int = 0                # UC adimli zincir havuzu (0 = yok)
    ret_pay: float = 0.05      # soru cumlelerinin ~payi REDDETME
    ret_tut: float = 0.20      # reddetme verisinin SINAVA ayrilan payi
    # --- HIZ. IKISI DE AYAR DUGMESI: kapatilinca ESKI yol aynen
    # kosar, yani bir kol 'hizli' kosarken oteki KIYAS icin eski
    # yolda kalabilir. Ikisi de YORUNGEYI DEGISTIRIR -- bit denk
    # DEGIL; o yuzden ayar, sessiz bir iyilestirme degil.
    bf16: bool = False         # fp16+GradScaler yerine bf16, SCALER YOK
    derle: bool = False        # torch.compile YALNIZ egitim ileri gecisinde
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
    # ek_kip="tr2" -- KULLANICI KARARI, 17 Eylul:
    #   *"ben duzgun bir turkce ile egitim istiyorum. Fakultesi ile
    #    fakultesi ayri seyler, o da ilk harften ayrisiyor."*
    # "tr" kipinde iliski KOK jetonuydu ve iyeligi ayri bir <SI> jetonu
    # tasiyordu (fakulte + <SI>). "tr2"de iliski KELIMENIN KENDISI:
    #   fakultesi   kucuk harf, CINS isim   <- iliski
    #   Fakultesi   BUYUK harf, OZEL adin parcasi
    # Ikisini ayiran sey YAZIM; fazladan isaretleyici GEREKMIYOR.
    # <SI> KALKTI, yerine SORU SOZCUGU geldi (kim / neresi / hangisi).
    ek_kip: str = ""           # "" | "tr" | "tr2"
    bicim: int = 1
    kisayol_kat: int = 0 # IYELIK KISA YOLU: kisi basina kac satir.
    #   "Ayse Sahin'in dekani kimdir?  Melek Aydin'dir."
    #   Yuzey 1-HOP, cevap COK ADIMLI. Turkce'de iyelik aidiyet
    #   kenarlarindan SESSIZCE gecer: okudugum bolumun fakultesinin
    #   dekani, kisa yoldan "benim dekanim" olur (kullanici, 18 Eylul).
    #   !! GRAFA KENAR EKLEMEZ. `olgu`ya girmez, `zincirler` gormez,
    #   `olcme_izi` DEGISMEZ, `ent_kisayol` olcusunun tanimi BOZULMAZ.
    #   Yalniz egitim havuzuna satir ekler -- tipki `soru_kat` gibi.
    soru_kat: int = 0    # SORU BICIMI: her zincir icin kac SORU satiri.
    #                      0 = kapali (model_06'nin durumu).
    #
    #   Ibrahim Yilmaz'in danismaninin arkadasi kimdir? Derya Yilmaz'dir.
    #
    #   NEDEN (kullanici, 17 Eylul): *"benim amacim EK YAPMAKTI, tum
    #   formati degistirmek degil. bosluk doldurma olsun, bu dil bilgisini
    #   ogrenmek icin su an iyi. ama ben su soruyu da sorabilmeliyim."*
    #
    #   !! PAY BURADA SECILIR, TUREMEZ. `fim_kat`ta ders olculdu: 2
    #   varyant secildi ve FIM havuzun %53,3'u OLDU -- kimse "yarisi
    #   olsun" demedi, sayi `fim_kat`tan DUSTU (OLCULENLER §1g).
    #   `egitim_havuzu` bu yuzden her kosuda SORU PAYINI da basiyor.
    fim_kat: int = 0     # BOSLUK DOLDURMA: bildirim satiri basina kac
    #                      varyant. 0 = kapali. Konum HER SATIRDA rastgele
    #                      (kullanici karari, 17 Eylul: "her konum esit
    #                      olasilikla"), ama VERI TOHUMUNA bagli -- yani
    #                      butun egitim tohumlari AYNI havuzu gorur.             # kac YUZEY BICIMI (1..3), ek_kip GEREKTIRIR
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
    # fim_kat / soru_kat model_06 ve model_07'de eklendi; ondan once
    # BOYLE BIR SATIR TIPI YOKTU -> ikisi de kapali.
    "fim_kat": 0,
    "soru_kat": 0,
    "kisayol_kat": 0,
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
    # `tetik` 18 Eylul'de eklendi; ondan once biyografi belgesinin
    # ONEGI YOKTU -> 0.
    "tetik": 0, "kopya": 5, "t_len": 512,
    "zincir_pay": 0.20, "n3": 0, "ret_pay": 0.05, "ret_tut": 0.20,
    # bf16/derle 19 Eylul'de eklendi; onceki butun kosular
    # fp16+GradScaler ve derlenmemis yolda kostu -> kapali.
    "bf16": False, "derle": False,
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


# --- IYELIK KISA YOLU ----------------------------------------------------
# KISI'den giden AKADEMIK zincir. Uc hedef, hepsi TEK OKUMALI:
#
#     X'in fakultesi      bolumu -> fakultesi                  2 adim
#     X'in dekani         bolumu -> fakultesi -> dekani         3 adim
#     X'in universitesi   bolumu -> fakultesi -> universitesi   3 adim
#
# !! NEDEN BU UCU. Olculdu (18 Eylul): grafta her birine giden BASKA
# yollar da var (`tezi -> konusu -> bolumu -> ...`, `memleketi ->
# universitesi`), ama Turkce onlari SESSIZCE YUTMAZ -- "Ayse'nin
# dekani" hicbir zaman "memleketinin universitesinin dekani" demez.
# Yutulanlar DISARIDA birakildi:
#     X'in sehri / bolgesi   memleket ve universite okumalari AYRISIYOR
#     X'in valisi            memleketinin valisi | yasadigi yerin valisi
#                            IKISI DE dogal -> cevap TEK DEGIL
KISAYOL_YOLU = (("fakultesi",    ("bolumu", "fakultesi")),
                ("konusu",       ("tezi", "konusu")),
                ("dekani",       ("bolumu", "fakultesi", "dekani")),
                ("universitesi", ("bolumu", "fakultesi", "universitesi")))
#   !! `konusu` 18 Eylul'de eklendi (kullanici: *"Ahmet Aydin konusu da bir
#   soru tipi olabilir mi?"*). Olabilir, ve dordu icinde EN TEMIZI:
#   KISI'den TEZ'e giden TEK kenar `tezi`, dolayisiyla "X'in konusu"nun
#   grafta TEK yolu var ("X'in fakultesi"nin iki yolu vardi).
#   Turkce'de de tek okumasi var: akademide "konun ne?" = "tez konun ne?".
#
#   !! Bu, "tezi -> konusu Turkce'de yutulmaz" diye yazdigimla CELISMIYOR.
#   Yutulma BASTAKI ADA bagli, kopru iliskiye degil:
#       "X'in KONUSU"     -> bir kisinin "konu"su ancak tezinin konusudur.
#       "X'in FAKULTESI"  -> dogrudan okumasi (bagli oldugu fakulte) BASKIN;
#                            "tezinin konusunun bolumunun fakultesi" DEGIL.


def _kisayol_kur(G, eid, rid, tut_bas, tut_kenar, tut_zincir):
    """(e, r_son, cevap) uclulerini uretir. Grafa DOKUNMAZ.

    Kisa yol satiri EGITIM verisidir ve COK ADIMLI bir gercegi 1-hop
    yuzeyle soyler. Dolayisiyla TUTULAN (held-out) her seyi ihlal
    edebilir. UC kapi var ve UCU DE gerekli -- biri eksik olursa
    sizinti SESSIZ olur:

      tut_bas     TUTULAN VARLIK. `ent` tanimi "hic zincir basi
                  olmamis"tir; "X'in dekani = Y" satiri X'i tam o
                  konuma koyar. ent / ent_yok / ent_arama / ent_kati.
      tut_kenar   TUTULAN KENAR. atomic_OOD kenarlari egitimde YOK;
                  yolu o kenardan gecen satir onu geri sokar.
      tut_zincir  TUTULAN ZINCIR. Iki adimli bir kisa yol, tam o
                  zincirin cevabidir.

    !! VARSAYILAN YOK, bilerek: izin verici bir varsayilan kapiyi
    sessizce acar. Doner: (satirlar, {sebep: sayi}).
    """
    olgu = G["olgu"]
    out, atilan = [], {"bas": 0, "kenar": 0, "zincir": 0}
    for r_son, yol in KISAYOL_YOLU:
        assert yol[-1] == r_son, (r_son, yol)
        for k in G["ad"]["KISI"]:
            if eid[k] in tut_bas:
                atilan["bas"] += 1
                continue
            c, kenar = k, False
            for r in yol:
                if (eid[c], rid[r]) in tut_kenar:
                    kenar = True
                    break
                c = olgu.get((c, r))
                if c is None:
                    break
            if kenar:
                atilan["kenar"] += 1
                continue
            if c is None:
                continue                 # yol EKSIK -- satir yok
            # !! KISA YOL BIR OLGUYU TEKRAR ETMEMELI: (k, r_son) zaten
            # kayitliysa satir yeni bir sey ogretmez. Semada KISI'den
            # bu DORT iliski CIKMIYOR, yani bu assert yapisal -- ama
            # sema degisirse SESSIZ kalmasin.
            assert (k, r_son) not in olgu, (
                f"kisayol bir OLGUYU tekrar ediyor: {k} {r_son}")
            if len(yol) == 2 and (eid[k], rid[yol[0]], rid[yol[1]]) in tut_zincir:
                atilan["zincir"] += 1
                continue
            out.append((eid[k], rid[r_son], eid[c]))
    return tuple(out), atilan


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
    ek_kip: str = ""                   # "" | "tr" | "tr2"  (ayar.ek_kip)
    soru_ad: tuple = ()                # ek_kip="tr2": SORU SOZCUKLERI,
    #   veri modulunun `SORU_SOZ`undan, SIRALI (determinizm).
    soru_tip: tuple = ()               # tip indeksi -> soru_ad indeksi.
    dir_soru: tuple = ()               # soru sozcugu -> ek_dir_ad indeksi
    #   ("kimdir", "neresidir") -- kimlik satiri icin.
    #   Motor tip ADLARINI BILMEZ; esleme VERI MODULUNDEN gelir.
    # --- EK ALLOMORFLARI (ek_kip="tr2") ---------------------------------
    # Turkce'de tamlayan eki sekiz bicimde: -in/-in/-un/-un ve sesliden
    # sonra -nin/... Ilk surumde TEK <NIN> jetonu hepsini ortuyordu;
    # kullanici fark etti (17 Eylul). Hangi bicimin gelecegi ONCEKI
    # KELIMEDEN belirli, yani YENI BILGI DEGIL -- dilin yuzeyi.
    # Unlu uyumunu VERI MODULU hesaplar, motor yalniz INDEKS tasir.
    ek_nin_ad: tuple = ()              # ("in","in","un",... ) 8 bicim
    ek_dir_ad: tuple = ()              # ("dir","dir","dur",...) 8 bicim
    nin_ent: tuple = ()                # varlik -> ek_nin_ad indeksi
    dir_ent: tuple = ()                # varlik -> ek_dir_ad indeksi
    nin_rel: tuple = ()                # iliski -> ek_nin_ad indeksi
    dir_rel: tuple = ()                # iliski -> ek_dir_ad indeksi
    kisayol: tuple = ()                # (e, r_son, cevap) IYELIK KISA YOLU
    kisayol_atilan: tuple = ()         # ((sebep, sayi), ...) URETILMEYEN satirlar
    #   `one` ile AYNI BICIMDE ama OLGU DEGIL: cevap `bolumu ->
    #   fakultesi -> ...` yolunun sonu. `facts`a GIRMEZ, `zincirler`
    #   gormez, `olcme_izi` DEGISMEZ.
    #  ^ model_06: 'annesi' -> 'annesidir'. YUKLEM ONE ALINMIS yuzey
    #    bicimleri icin; rolu KONUM degil EK tasisin diye.

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
            assert self.ek_kip == "tr2", (
                f"eksiz ve 'tr' kodlama yollari SILINDI: {self.ek_kip!r}")
            self.ek0 = self.vocab
            if True:
                # '  + NIN allomorflari + DIR allomorflari + SORU SOZCUKLERI
                assert self.soru_ad, "tr2 SORU_SOZ ister -- veri modulu vermeli"
                assert len(self.soru_tip) == len(self.tip_ad), (
                    "soru_tip her TIP icin bir deger tasimali")
                assert self.ek_nin_ad and self.ek_dir_ad, (
                    "tr2 EK ALLOMORFLARINI ister -- veri modulu vermeli")
                assert len(self.nin_ent) == len(self.dir_ent) == self.n_ent, (
                    "her VARLIK icin ek bicimi belli olmali")
                assert len(self.nin_rel) == self.n_rel, (
                    "her ILISKI icin ek bicimi belli olmali")
                self.nin0 = self.ek0 + 1
                self.dir0 = self.nin0 + len(self.ek_nin_ad)
                self.soru0 = self.dir0 + len(self.ek_dir_ad)
                self.vocab += (1 + len(self.ek_nin_ad) + len(self.ek_dir_ad)
                               + len(self.soru_ad))
                # BOSLUK DOLDURMA jetonlari -- SOZLUGUN EN SONUNA, ki
                # REL_OFF / ent_off / yuva_ara HIC KAYMASIN.
                #   <BOS>   cumleden CIKARILAN parcanin yerini tutar
                #   <AYIR>  cumle bitti, simdi o parca geliyor
                # Kullanici karari, 17 Eylul: *"Fatma ..... 'in annesi Ayse
                # Yilmaz'dir gibi"* -- yani ayni cumlede HER parca
                # sorulabilmeli. Model nedensel oldugu icin bosluk yerinde
                # sorulamaz (sagini goremez); parca SONA tasinir.
                self.bosluk = self.vocab
                self.ayir = self.vocab + 1
                # VIRGUL -- kullanici karari, 17 Eylul: *"burda , kavrami
                # devreye giriyor... kardesi'dir Ozlem Yilmaz, Ibrahim
                # Yilmaz'in"*.
                #
                # GEREKCESI OLCULDU: devrik bicimde cevap ile ozne YAN YANA
                # iki ad oluyor ve aralarinda HIC isaret yoktu --
                # "Kardesidir Ozlem Yilmaz Ibrahim Yilmaz'in": cevap
                # "Ozlem" mi "Ozlem Yilmaz" mi, dizide bunu soyleyen bir
                # sey yok. `havuz_06` yakaladi: 3000 satirin 300'unde
                # next-token sozlesmesi dusuyordu. Diger biciminde sinir
                # zaten isaretli ('dir / 'in); burada VIRGUL isaretliyor.
                self.virgul = self.vocab + 2
                self.vocab += 3
        # --- DEGISKEN UZUNLUKLU AD (17 Eylul, kullanici: "<YOK> sil,
        # gereksiz"). Once her varlik TAM `yuva` jeton kapliyordu ve kisa
        # adlar <YOK> ile SAGDAN dolduruluyordu -- havuzdaki butun
        # jetonlarin %11,5'i dolguydu ve "Adana <YOK> <YOK>" Turkce
        # DEGILDI. Artik ad kac kelimeyse o kadar jeton.
        #
        # SINIR ZATEN ISARETLI: her adin ardindan KESME ISARETI geliyor
        # ("Adana'nin"). Yani dolguya gerek yok -- model adi bitirip `'`
        # demeyi ogrenmek zorunda, ki bu DAHA GERCEK bir dil gorevi.
        if self.par is not None and self.ek_kip == "tr2":
            self.n_yuva = tuple(
                int((self.par[e] >= 0).sum()) for e in range(self.n_ent))
            assert min(self.n_yuva) >= 1, "her adin en az bir kelimesi olmali"
            # CEVAP pozisyonlarinda izin verilen kume: VARLIK kelimeleri
            # ARTI kesme isareti. Ikisi BITISIK (ent_off .. ek0 .. ek0+1),
            # yani tek aralik yetiyor.
            self.cevap_ara = (self.ent_off, self.ek0 + 1)
            self.cevap_yuva = self.yuva + 1      # en uzun ad + `'`
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
        # !! `<YOK>` SOZLUKTEN CIKTI (17 Eylul, kullanici: "sil,
        # gereksiz"). Kisa adlar artik SAGDAN DOLDURULMUYOR; `par`in
        # bos yuvalari -1 tasiyor ve `_e` onlari HIC uretmiyor.
        # Sinir zaten KESME ISARETI ile isaretli ("Adana'nin").
        _yuva = max(len(a.split("_")) for a in E)
        _pl = sorted({p for a in E for p in a.split("_")})
        _ix = {p: i for i, p in enumerate(_pl)}
        _par = np.array([[(_ix[p] if p is not None else -1) for p in
                          (a.split("_") + [None] * _yuva)[:_yuva]]
                         for a in E], np.int64)
        _sz = tuple(_pl)
        _par_ad = (_sz,) * _yuva                         # AYNI sozluk, her yuva
        assert not any("_" in p for p in _pl), "jeton icinde ALT CIZGI kaldi"
        assert len({tuple(r) for r in _par}) == len(E),             "BIREBIR DEGIL -- ayni jeton dizisi birden cok varliga denk"
        _coz = lambda r: "_".join(_sz[i] for i in r if i >= 0)
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

    # --- SORU SOZCUKLERI (ek_kip="tr2") -- VERI MODULUNDEN --------------
    # Motor tip ADLARINI bilmez: "KISI'ye kim denir"i veri modulu soyler.
    # Sirali, cunku jeton id'leri buradan duser ve DETERMINIST olmali.
    _soru_ad, _soru_tip = (), ()
    _ek_nin_ad = _ek_dir_ad = _nin_ent = _dir_ent = _nin_rel = ()
    _dir_rel = ()
    _dir_soru = ()
    if ayar.ek_kip == "tr2":
        _SS = getattr(_V, "SORU_SOZ", None)
        assert isinstance(_SS, dict) and set(_SS) == set(_V.TIPLER), (
            f"{ayar.veri_ad}.SORU_SOZ her TIP icin bir soru sozcugu "
            f"vermeli (ek_kip='tr2'): {_SS}")
        _soru_ad = tuple(sorted(set(_SS.values())))
        _soru_tip = tuple(_soru_ad.index(_SS[t]) for t in _V.TIPLER)
        _ES = _V.ek_secim(G)
        _ek_nin_ad, _ek_dir_ad = tuple(_V.EK_NIN), tuple(_V.EK_DIR)
        _nin_ent = tuple(_ES["nin_varlik"][a] for a in E)
        _dir_ent = tuple(_ES["dir_varlik"][a] for a in E)
        _nin_rel = tuple(_ES["nin_iliski"][r] for r in _V.ILISKI)
        _dir_rel = tuple(_ES["dir_iliski"][r] for r in _V.ILISKI)
        _dir_soru = tuple(_ES["dir_soru"][w] for w in _soru_ad)

    say = lambda L: [(eid[x[0]], rid[x[1]], rid[x[2]], eid[x[3]], eid[x[4]])
                     for x in L]
    # IYELIK KISA YOLU icin YASAK ucluler: butun SINAV bolmeleri.
    # --- IYELIK KISA YOLU: UC TUTMA KAPISI (bkz. `_kisayol_kur`)
    # !! `ent_kt` (KATI) de listede. Bu kolda bos (kati_pay=0) ama
    # `olcme_listeleri` doluysa onu da olcuyor -- dusurulseydi
    # kati_pay>0 olan bir kolda SESSIZ sizinti olurdu.
    _ksy_bas = {eid[x[0]] for L in (ent_ay, ent_yk, ent_ar, ent_kt) for x in L}
    _ksy_kenar = {(eid[a], rid[b]) for a, b in ood_k}
    _ksy_zincir = {(eid[x[0]], rid[x[1]], rid[x[2]])
                   for L in (comp, ent_ay, ent_yk, ent_ar, ood_ay, ent_kt)
                   for x in L}
    _ksy, _ksy_at = _kisayol_kur(G, eid, rid, _ksy_bas, _ksy_kenar, _ksy_zincir)
    v = Veri(facts=facts, one=[(eid[e], rid[r], eid[h])
                               for (e, r), h in G["olgu"].items()],
             tr2=say(tr2), comp=say(comp), ent=say(ent_ay),
             ent_yok=say(ent_yk), ent_arama=say(ent_ar),
             tip=E_tip, tip_ad=tuple(_V.TIPLER), ent_kati=say(ent_kt),
             ood=say(ood_ay), par=_par, par_ad=_par_ad or (),
             t_len=ayar.t_len, ek_kip=ayar.ek_kip,
             soru_ad=_soru_ad, soru_tip=_soru_tip,
             ek_nin_ad=_ek_nin_ad, ek_dir_ad=_ek_dir_ad,
             nin_ent=_nin_ent, dir_ent=_dir_ent, nin_rel=_nin_rel,
             dir_rel=_dir_rel,
             dir_soru=_dir_soru,
             kisayol=_ksy, kisayol_atilan=tuple(sorted(_ksy_at.items())))

    # --- SIZINTI DENETIMI -- sessiz gecmesin
    trset = {(e, a, b) for e, a, b, _, _ in v.tr2}
    for nm, st in (("COMP", v.comp), ("ENT", v.ent), ("ENT-YOK", v.ent_yok)):
        k = sum((e, a, b) in trset for e, a, b, _, _ in st)
        assert k == 0, f"{nm} sizintisi: {k} ornek egitimde de var"
    # !! KISA YOL SATIRLARI UC KAPIDAN DA GECIRILIR. Iki adimli bir
    # kisayol ("X'in fakultesi = Y") tam olarak "X'in bolumunun
    # fakultesi?" sorusunun cevabidir; ayrica X TUTULAN bir varliksa
    # onu ZINCIR BASI yapar, ve yol bir OOD kenarindan geciyorsa o
    # kenari egitime geri sokar. Ucu de OLCULDU ve UCU DE OLDU
    # (18 Eylul): sirasiyla comp'ta 28 zincir, 256 tutulan-bas satiri,
    # 55 OOD satiri.
    _ksy_yol = {r: yol for r, yol in KISAYOL_YOLU}
    _sinav3 = {(x[0], x[1], x[2]) for L in
               (v.comp, v.ent, v.ent_yok, v.ent_arama, v.ood, v.ent_kati)
               for x in L}
    _sz = []
    for e, r, a in v.kisayol:
        _y = _ksy_yol[_V.ILISKI[r]]
        if e in _ksy_bas:
            _sz.append(("TUTULAN BAS", e, _V.ILISKI[r]))
        if len(_y) == 2 and (e, rid[_y[0]], rid[_y[1]]) in _sinav3:
            _sz.append(("SINAV ZINCIRI", e, _V.ILISKI[r]))
        _c = e
        for _rr in _y:
            if (_c, rid[_rr]) in _ksy_kenar:
                _sz.append(("OOD KENARI", e, _V.ILISKI[r]))
                break
            _c = int(facts[_c, rid[_rr]])
            if _c < 0:
                break
    assert not _sz, f"KISA YOL sizintisi: {len(_sz)} satir -- {_sz[:5]}"
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
# !! MODEL_08'IN KODLAYICILARI SILINDI (18 Eylul).
#
# `_bos` `_e` `_cevap` `kelimeler` `_kesme` `_nin` `_dir` `_dir_soru`
# `_soru` `_bildirim` `kodla_1hop` `kodla_2hop` `kodla_fim_sinav`
# `kodla_soru` `kodla_fim` `kodla_kimlik_q1` `kopru_hedefi` -- 453 satir.
#
# Hepsi "varlik = 3 jeton yuvasi, iliski = 1 atomik jeton" semasina
# baglilardi. model_09 KARAKTER duzeyinde calisiyor: o semadan geriye
# hicbir sey kalmadi. Metni `metin_09` uretiyor, jetona `jeton_09`
# ceviriyor, pencereye `korpus_09` koyuyor.
#
# SILINDILER, YORUMA ALINMADILAR: calismayan ama duran kod, dosyayi
# okuyanin "demek ki hala boyle" diye okumasina sebep olur. Bu kolda
# ayni hata iki kez oldu (kos_11.py MODEL="model_08", ayar_11.py
# `from taban_08 import Ayar`).

def egitim_havuzu(ayar: Ayar, v: Veri, yaz=print):
    """KORPUS -> egitim tensoru. model_08'in SATIR TABLOSU YOK.

    model_08'de havuz `kodla_1hop` / `kodla_2hop` / FIM / soru / kimlik
    dilimlerinden kuruluyordu ve her satir TEK olguydu. Burada havuz
    PAKETLENMIS METIN: belgeler birlestirilip `t_len`lik pencerelere
    bolunuyor (Physics of LM 3.1 Ek C duzeni). Bir pencere 7..11 cumle
    tasiyor ve hangi cumlenin nerede oldugu MODELE SOYLENMIYOR.

    Doner: (X, S). `P`/`T` (cevap yuvasi) YOK -- kayip butun
    pozisyonlarda next-token, yani `tam_kayip` artik SECENEK degil
    TEK yol."""
    assert ayar.tam_kayip, (
        "model_09 DIL MODELI kaybiyla egitilir. Cevap yuvasi diye bir sey "
        "yok: pencerede 7-11 cumle var ve hangisinin cevap oldugu "
        "isaretlenmiyor.")
    V09 = importlib.import_module(ayar.veri_ad)
    # !! ADLI CAGRI. Ilk surumde `yaz` POZISYONEL gecti ve `zincir_pay`
    # yuvasina dustu -- korpus sessizce `len(sat) * print` deneyip
    # patladi. Sirasi kayan bir cagri, patlamasaydi YANLIS ORANLA
    # korpus kurardi.
    X, S = KOR.havuz(v, V09.kur(ayar.veri_tohum), kopya=ayar.kopya,
                     t_len=ayar.t_len, tohum=ayar.veri_tohum,
                     tetik=ayar.tetik, zincir_pay=ayar.zincir_pay,
                     n3=ayar.n3, ret_pay=ayar.ret_pay,
                     ret_tut=ayar.ret_tut, yaz=yaz)
    return X, S

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
# !! `dogruluk` / `ek_dogruluk` / `kisayol_orani` SILINDI (18 Eylul).
#
# Ucu de KISITLI ARGMAX yapiyordu: cevap yuvasinda yalniz VARLIK
# jetonlari yarisirdi, ve her yuva GERCEK onekle puanlanirdi (teacher
# forcing). Karakter duzeyinde "varlik jetonu blogu" diye bir sey yok,
# yani kisit kurulamaz -- ve kurulmamali da: kisitli sayi modelin
# serbest birakildiginda ne yazacagini SOYLEMIYOR.
#
# Yerine `olcme_11.Sinav` + `olcme_11.olc`: model soruyu alir, gerisini
# KENDI yazar, cikan dizge beklenenle TAM ESLESMELI. Gecisin bedelsiz
# oldugu OLCULDU (model_08, 18 Eylul): belirsizligi olmayan soru
# yuzeyinde kisitli ile serbest alti bolmede de +0.0000 fark verdi.

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
                 yavas=None, korpus_izi=""):
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
            korpus_izi=korpus_izi,
            # Lookahead YAVAS agirligi. ort_bas=0 iken None -- eski
            # paketlerde bu anahtar hic YOKTU, `surdurme_oku` .get() ile
            # okuyor, yani eski paketler aynen surdurulebilir.
            yavas=(None if yavas is None else
                   {k: v.cpu() for k, v in yavas.items()}),
        ), t)
    _atomik(yol, w)


def surdurme_oku(yol, ayar: Ayar, model, opt, scaler, rs, iz, yaz=print,
                 korpus_izi=""):
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
    # KORPUS KAPISI. `olcme_izi` SINAVI kolluyor, ayar farki AYARI --
    # ikisi de EGITIM METNINI kollamiyor. `metin_11.cumle`de tek bir ek
    # degisse korpus bastan asagi degisir ve iki kapi da SESSIZ kalirdi:
    # kosu "surduruldu" diye devam eder, yarisi bir metinle yarisi
    # baskasiyla egitilmis olurdu.
    if korpus_izi and p.get("korpus_izi", korpus_izi) != korpus_izi:
        raise SystemExit(
            os.linesep + f"!! SURDURULEMEZ: EGITIM KORPUSU degismis "
            f"(paket {p.get('korpus_izi')}, simdi {korpus_izi})."
            + os.linesep
            + "   Ayar ayni ama metin baska -- `metin_09` / `korpus_09` "
            "degismis olmali.")
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
            + "YAPISAL OLARAK 0 olmaliydi." + os.linesep
            + "   ENT-YOK'un TANIMI: facts[e, r2] = -1, yani kisayolun"
            + os.linesep
            + "   isaret edecegi varlik GRAFTA YOK. `olcme_11.sorular` o"
            + os.linesep
            + "   satirlara ksy=None veriyor ve `puanla` None'i saymiyor."
            + os.linesep
            + "   Sifir DEGILSE bolme mantigi bozulmus demektir (ENT-YOK'a"
            + os.linesep
            + "   kisayolu MUMKUN bir zincir sizmis) -- veri kusuru."
            + os.linesep
            + "   !! Bu kapi OLCUM kodunu DENETLEMIYOR: ksy=None oldugu"
            + os.linesep
            + "   surece 0 cikar. Olcum kodunun kendisi `test_09` §4b'de"
            + os.linesep
            + "   SAHTE CIKTIYLA sinaniyor. Kosu durduruldu.")
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
    Xtr, S = egitim_havuzu(ayar, v, yaz)
    L = olcme_listeleri(ayar, v)
    iz = olcme_izi(L)
    # SINAV BIR KEZ KURULUR. Her olcum noktasinda yeniden kodlansaydi
    # 10 olcum x 6 bolme = 60 kez ayni metin uretilirdi.
    sinav = {k: OLC.Sinav(S, v, importlib.import_module(ayar.veri_ad)
                          .kur(ayar.veri_tohum), L[k],
                          1 if k == "one" else 2, k)
             for k in L if len(L[k])}
    _g = max(x.genislik for x in sinav.values())
    yaz("  olcme: " + "  ".join(f"{k} {len(L[k])}" for k in L if len(L[k]))
        + f"   parmak izi {iz}")
    yaz(f"  SINAV GENISLIGI {_g} karakter (OLCULDU)  egitim penceresi "
        f"{ayar.t_len}  -> olcum {ayar.t_len / _g:.1f} kat ucuz")

    torch.manual_seed(ayar.tohum)
    model = (model_kur or Model)(ayar, S.vocab).to(DEV)
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
    # !! bf16 ile GradScaler KAPANIR. Sebep OLCULDU (torch 2.14
    # kaynagi, `GradScaler._maybe_opt_step`):
    #     if not sum(v.item() for v in
    #                optimizer_state["found_inf_per_device"].values()):
    # `.item()` ADIM BASINA bir GPU->CPU SENKRONU demek. Ayrica
    # olcek tasinca o adim ATLANIR (init_scale 65536, her 2000
    # adimda bir yoklama) -- yani `adim` ile GERCEK guncelleme
    # sayisi AYRISIR. bf16'nin ustel araligi fp32 ile AYNI
    # (1.18e-38..3.39e38), olcege ihtiyac YOK.
    _bf = bool(ayar.bf16)
    if _bf and DEV == "cuda":
        assert torch.cuda.is_bf16_supported(), (
            "bf16 DESTEKLENMIYOR (Ampere ve ustu gerekir). L4/A100 " + NL +
            "  evet, T4 HAYIR. ayar.bf16=False ile fp16 yoluna don.")
    _dtip = torch.bfloat16 if _bf else torch.float16
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda" and not _bf))

    # --- DERLENMIS SARMALAYICI. `model` HAM KALIR.
    # !! AYRIM SART, iki tuzak var ve ikisi de SESSIZ:
    #   1) `state_dict()` derlenmis nesneden alinirsa anahtarlar
    #      `_orig_mod.` oneki tasir. Anlik goruntu ve surdurme
      #    paketi bozuk anahtarla yazilir, `pencere_*` OKUYAMAZ.
    #   2) Olcum farkli batch sekilleriyle cagriliyor; derlenmis
    #      nesne her sekil icin YENIDEN DERLER. Olcum ham modeli
    #      kullanir, bu yuzden dert yok.
    # Bu kolda AYNI SINIF hata zaten yasandi (kopyada dosya adi
    # degisti, ICERIK degismedi) -- sessiz olan tehlikelidir.
    _egit_ag = model
    if ayar.derle and DEV == "cuda":
        _egit_ag = torch.compile(model)
        yaz("  torch.compile ACIK -- YALNIZ egitim ileri gecisi. "
            "Anlik goruntu / olcum / surdurme HAM modelden.")
    if _bf:
        yaz("  bf16 ACIK -- GradScaler YOK (adim basina senkron da yok)")
    # --- DURUSTLUK ve TUTARLILIK SINAVI: BIR KEZ kurulur.
    # !! model_10'da ikisi de AYRI ARAC idi ve rapor IKI PARCA
    # kaldi. Kullanici: *"niye 6 disinda ek bir rapor alani yaptin?
    # butun kriterleri tek bir yerden gorsek"* ve *"yani ben su an
    # rapordan model olgun mu degil mi bilmiyorum."*
    # Artik ana olcum noktasinda, ayni gecise binerek olculuyor ve
    # `egri_*.json`a yaziliyor -- rapor hucresi onu diger sutunlar
    # gibi basar. Maliyet ~%3 (ana sinav ~14.000 satir, bu 400).
    #
    # SINAV TUTULAN VERIDEN: o uydurma ad ve o (tip, iliski) cifti
    # egitim reddetmelerinde HIC gecmez (`ret_tut`). Gordugunu
    # reddetmek EZBER olurdu; olctugumuz GENELLEME.
    _G_ret = importlib.import_module(ayar.veri_ad).kur(ayar.veri_tohum)
    _sv_ret = None
    if ayar.ret_pay:
        _bl_ret = KOR.reddetme_bolme(_G_ret, ayar.ret_tut,
                                     ayar.veri_tohum)
        _sv_ret = OLC.RetSinavi(S, _G_ret, _bl_ret,
                                tohum=ayar.veri_tohum, ad="ret")
        yaz(f"  DURUSTLUK sinavi {len(_sv_ret)} cevapsiz soru "
            f"(TUTULAN veriden, egitimde HIC gecmez)")
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
                  vocab=S.vocab, sozluk_izi=S.iz,
                  korpus=int((Xtr != J.PAD).sum()),
                  phi=round(v.phi, 4),
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
        # DURUSTLUK -- CIFT olcu, AYNI olcum noktasinda.
        # !! `kacamak` olmadan `dogru_ret` OKUNMAZ: her seye "yok"
        # diyen model dogru rette TAVANA cikar. Ikisi de yazilir.
        if _sv_ret is not None and "one" in sinav:
            _dr = OLC.durustluk_olc(model, S, _sv_ret, sinav["one"], DEV)
            r["dogru_ret"] = _dr["dogru_ret"]
            r["ret_ad"] = _dr["ret_ad"]
            r["ret_cift"] = _dr["ret_cift"]
            r["kacamak"] = _dr["kacamak"]
        for k, sv in sinav.items():
            d = OLC.olc(model, S, sv, DEV)
            r[k] = d["tam"]
            # YAKIN AYRI SUTUN: "varligi buldu ama eki yanlis yazdi"
            # ile "baska bir varlik yazdi" ayni sey degil, ve karakter
            # duzeyinde ilki GERCEKTEN oluyor ("Sirnak'ta" / "Sirnak").
            r[k + "_yakin"] = d["yakin"]
            if k in ("ent", "ent_yok", "comp"):
                r[k + "_kisayol"] = d["kisayol"]
        return r

    def _satir(r):
        return (f"  {r['adim']:7d}/{ayar.adim}  "
                + ("kayip   ---" if r["kayip"] is None
                   else f"kayip {r['kayip']:.3f}")
                + "  " + "  ".join(f"{k} {r[k]:.3f}" for k in
                                   ("one", "seen", "comp", "ent") if k in r)
                + f"  ksy {r.get('ent_kisayol', float('nan')):.3f}"
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
                                        rs, iz, yaz, S.korpus_izi)
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
    # !! `yrd_top` / `ana_top` DUSTU. Ikisi de model_08'in IKI TERIMLI
    # kaybini ayirmak icindi (yardimci kopru + cevap yuvasi). model_09'da
    # kayip TEK terim -- ayrilacak bir sey yok.
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
            # !! GPU HAVUZU KALDIRILDI -- kullanici karari, 18 Eylul:
            # "GPU_HAVUZ ile ilgili ne varsa kaldir". model_06 ve
            # model_07'de havuzun tamamini GPU'da tutan IKINCI bir dal
            # vardi (int16, adim basina host->device kopya yok).
            # OLCULDU (`kos_07.py --hiz-dogrula 60`): yorunge BIT
            # DUZEYINDE ayni (agirlik farki 0.000e+00) ama HIZLANMA
            # 0.98x -- yani YOK. Kazanci olmayan bir yol iki kod dali,
            # bir ortam degiskeni ve bir dogrulama bayragi tasiyordu;
            # ucu de silindi. Kalan tek yol budur.
            xb = torch.from_numpy(Xtr[j]).to(DEV)
            with torch.autocast(DEV, dtype=_dtip,
                                enabled=(DEV == "cuda")):
                # !! `_egit_ag` DERLENMIS sarmalayici, `model` HAM.
                # Ayrim SART: `state_dict()` derlenmis nesneden
                # alinirsa anahtarlar `_orig_mod.` oneki tasir ve
                # `pencere_*` anlik goruntuleri OKUYAMAZ. Olcum de
                # ham modeli kullanir; yoksa her olcum noktasinda
                # farkli batch sekli YENIDEN DERLEME tetikler.
                lg_tam = _egit_ag(xb)
                # DIL MODELI KAYBI, TEK KAYIP. Pozisyon t, X[t+1]'i
                # tahmin eder; PAD hedefleri atlanir (PAD yalniz dizinin
                # KUYRUGUNDA, ~%6).
                #
                # !! model_08'in UC terimi de dustu:
                #   `ana` (cevap yuvasi)  -- cevap yuvasi diye bir sey yok
                #   FIM <BOS> maskesi     -- FIM cikti (kullanici, 18 Eylul)
                #   kopru_kayip           -- kopru pozisyonu SABIT DEGIL;
                #                            pencerede 7-11 cumle var ve
                #                            zincir cumlesi koprüyü zaten
                #                            YAZMIYOR, yani hedef YOK.
                kayip = F.cross_entropy(
                    lg_tam[:, :-1].float().reshape(-1, lg_tam.shape[-1]),
                    xb[:, 1:].reshape(-1), ignore_index=J.PAD)
            kayip_top += kayip.detach()
            kayip_say += 1
            opt.zero_grad(set_to_none=True)
            if _bf:
                kayip.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
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
                # BITS PER CHARACTER -- karakter modelinin standart olcusu
                # ve model_08'in `kayip`iyla KIYASLANAMAZ oldugunu
                # gorunur kilar (orada jeton, burada karakter).
                r["bpc"] = round(r["kayip"] / math.log(2), 4)
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
                             egri, iz, yavas, S.korpus_izi)
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


# NOT: `model_a.py` burada `AYAR = Ayar()` tanimlayip dogrudan
# kosulabiliyordu. `taban_09` bir MOTOR; model_09'in ayari `ayar_11.py`de,
# kosuyu baslatan `model_11.py`. Burada calistirilacak bir sey YOK.
