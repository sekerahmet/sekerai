# -*- coding: utf-8 -*-
"""taban_13 -- model_13'un VERI YOLU. `taban_11`den KOPYA.

Kullanici karari, 20 Eylul 2026: *"evet onlari kopyala aynisi zaten"*.

TASINAN: Ayar, Veri, veri_kur, olcme_listeleri, olcme_izi ve
bagimliliklari. Kapanis AST ile cikarildi, elle secilmedi -- IMPORTLAR
DA OYLE: torch / jeton / dil / olcme kapanista HIC gecmiyor, bu dosya
torch'suz calisiyor.

TASINMAYAN (~1000 satir): egit, Model, Blok, RMSNorm,
surdurme_*, onbellek_*, erken_teshis, _atomik ... Hepsi TRANSFORMER
egitim motoru. model_13 metne hic bakmiyor, yalniz grafa.
CLAUDE.md kural 7: cagrisiz kod tasinmaz.

KOPYANIN KAPISI IZLER:
    veri_13.IZ      == 3cd9a2575e47
    olcme_izi(L)    == 44e6262e37f3
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import importlib
import os

import numpy as np

import jeton_13 as J
import korpus_13 as KOR


SPECIAL = 3


T_LEN = 8


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


KISAYOL_YOLU = (("fakultesi",    ("bolumu", "fakultesi")),
                ("konusu",       ("tezi", "konusu")),
                ("dekani",       ("bolumu", "fakultesi", "dekani")),
                ("universitesi", ("bolumu", "fakultesi", "universitesi")))


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
        self._phi_hesapla()

    def _phi_hesapla(self):
        """phi + wang_phi. `tr2` DEGISIRSE yeniden cagrilir.

        !! METOT, cunku `zincir_butcesi` `tr2`yi BUDUYOR. Hesap
        `__post_init__` icinde tek satir kalsaydi `v.phi` budama ONCESI
        degeri tasirdi: rapor 4.00 basardi, model 1.12 gorurken.

        phi TURETILMIS bir tani sayisidir, kontrol parametresi DEGIL --
        graf yogunlugundan ve ent_pay/comp_pay'den duser.

        WANG'IN TANIMI AYNI DEGIL (15 Eylul hakemligi). Wang
        2405.15071:155 "phi = |train_inferredID| / |atomicID|" ve
        atomicID, OOD varliklarinin olgularini DISLAR; bizim paydamiz
        TUM olgular. Olculdu: ayni veride bizimki 5.09, Wang tanimiyla
        6.36 -- %25 fark. Wang'in 3.6-18.0 taramasina konumlanirken
        WANG_PHI kullanilmali, phi degil. KATI varliklari paydadan
        duser -- Wang'in OOD'si tam olarak o."""
        self.phi = len(self.tr2) / max(1, len(self.one))
        _ent = ({e for e, *_ in self.ent}
                | {e for e, *_ in self.ent_yok}
                | {e for e, *_ in self.ent_arama}
                | {e for e, *_ in self.ent_kati})
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

    # --- ZINCIR BUTCESI: `tr2` KORPUSUN YAZABILECEGI KADAR -------------
    # OLCULDU 19 Eylul: tr2 29.510 zincir tasiyordu, korpus 6.819 tanesini
    # yazabiliyordu. `zincir` sinavi TAMAMINDAN soruyordu -> tavan 0.23,
    # esik 0.95. Olcu kendi adini yalanliyordu ("gordugu bilesigi
    # hatirliyor mu" -- gormedigini soruyordu).
    #
    # !! BURADA, `veri_kur`in ICINDE. Cunku iki okuma yolu var ve ikisi
    # de once BURAYA ugruyor:
    #     egit()      : egitim_havuzu -> olcme_listeleri
    #     pencere_11  : olcme_listeleri -> egitim_havuzu   (TERS SIRA)
    # Budama `havuz`da yapilsaydi pencere BUDANMAMIS tr2 ile sinav
    # kurardi ve iki yol FARKLI `zincir` olcerdi -- sessizce.
    if ayar.zincir_pay:
        v.tr2 = KOR.zincir_butcesi(
            v.one, v.tr2, KOR.kopru_yasagi(v), ayar.kopya,
            ayar.zincir_pay, tohum=ayar.veri_tohum, yaz=yaz)
        v._phi_hesapla()          # tr2 degisti -> phi/wang_phi DE degisir

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


_K = os.path.dirname(os.path.abspath(__file__))


ONBELLEK_YOL = os.environ.get(
    "KORPUS_ONBELLEK", os.path.join(os.path.dirname(_K), "onbellek"))


ONBELLEK_KAPALI = os.environ.get("KORPUS_ONBELLEK_KAPALI", "") == "1"


ONBELLEK_KAYNAK = ("korpus", "metin", "jeton", "veri")


def _onbellek_anahtari(ayar, v) -> str:
    """AYAR + VERI + KAYNAK KOD -> 16 haneli anahtar."""
    import hashlib
    h = hashlib.sha256()
    for k in ("kopya", "t_len", "tohum", "veri_tohum", "tetik", "zincir_pay",
              "n3", "ret_pay", "ret_tut", "veri_ad", "ek_kip", "tam_kayip"):
        h.update(f"{k}={getattr(ayar, k, None)!r};".encode())
    for ad in ("one", "tr2", "comp", "ent", "ent_yok", "ent_arama",
               "ood", "ent_kati"):
        L = getattr(v, ad, None) or []
        h.update(f"{ad}={len(L)}:".encode())
        h.update(repr(L).encode())
    _kol = os.path.basename(_K).rsplit("_", 1)[-1]
    for _ad in ONBELLEK_KAYNAK:
        _f = os.path.join(_K, f"{_ad}_{_kol}.py")
        if _ad == "veri":
            _f = os.path.join(_K, ayar.veri_ad + ".py")
        with open(_f, "rb") as _fh:
            h.update(_fh.read())
    return h.hexdigest()[:16]


def _onbellek_oku(yol, yaz):
    """(X, S) ya da None. Bozuk dosya SESSIZCE atlanir, patlamaz."""
    try:
        z = np.load(yol, allow_pickle=False)
        X = z["X"].astype(np.int64)
        S = J.Sozluk("".join(str(z["harf"])))
        S.korpus_izi = str(z["korpus_izi"])
        yaz(f"  KORPUS ONBELLEKTEN: {os.path.basename(yol)}  "
            f"{X.shape[0]:,} x {X.shape[1]}  iz {S.korpus_izi}")
        return X, S
    except Exception as e:                       # pragma: no cover
        yaz(f"  onbellek OKUNAMADI ({e}) -- yeniden kurulacak")
        return None


def _onbellek_yaz(yol, X, S, yaz):
    """int16 olarak yazar: sozluk 64 sembol, int64 dort kat israf."""
    try:
        os.makedirs(os.path.dirname(yol), exist_ok=True)
        assert int(X.max()) < 32767, "int16 tasar -- sozluk buyumus"
        _gec = yol + ".gecici"
        np.savez(_gec, X=X.astype(np.int16),
                 harf="".join(S.harf), korpus_izi=S.korpus_izi)
        os.replace(_gec + ".npz", yol)           # ATOMIK: yarim dosya kalmaz
        yaz(f"  KORPUS ONBELLEGE YAZILDI: {os.path.basename(yol)}  "
            f"{os.path.getsize(yol)/1e6:.0f} MB")
    except Exception as e:                       # pragma: no cover
        yaz(f"  onbellege YAZILAMADI ({e}) -- kosu etkilenmez")


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
    # ONBELLEK: anahtar AYAR + VERI + KAYNAK KOD (bkz. yukarisi).
    _yol = None
    if not ONBELLEK_KAPALI:
        _yol = os.path.join(ONBELLEK_YOL,
                            f"korpus_{_onbellek_anahtari(ayar, v)}.npz")
        _var = _onbellek_oku(_yol, yaz) if os.path.exists(_yol) else None
        if _var is not None:
            return _var
    X, S = KOR.havuz(v, V09.kur(ayar.veri_tohum), kopya=ayar.kopya,
                     t_len=ayar.t_len, tohum=ayar.veri_tohum,
                     tetik=ayar.tetik, zincir_pay=ayar.zincir_pay,
                     n3=ayar.n3, ret_pay=ayar.ret_pay,
                     ret_tut=ayar.ret_tut, yaz=yaz)
    if _yol:
        _onbellek_yaz(_yol, X, S, yaz)
    return X, S
