# model_17 — TASARIM

Kullanıcı, 23 Eylül 2026:

> *"amacım mevcut olmayan ama llm olma potansiyeli olan bir mimari
> yapmak."*

> *"model 17 değiştirebilirsin. yeni bir model e gerek yok. bos ve eos
> gerekli eğer bir dil ve mantık modeli yapacaksak bu gerekli.
> transfomer çok sonra."*

> *"transformer ile kıyas lazım ama önce göz ile mimarinin bir
> olgunluğa eriştiğini görmek gerekiyor."*

**Bu dosya adım adım yazılıyor.** Her adım bir karar verir ve bir
sonrakinin sorusunu açar. Ölçülen sayılar burada DEĞİL:
`belge/onkayit/model_17_TAM1.md`. Eski mimarinin yedeği model_15
(kullanıcı: *"yani yedeğii model 15"*).

---

## Mimari — TAM1'den devralınan, DEĞİŞMEDİ

```
gez      s_t = normalize(M[w_t] s_{t-1} + b[w_t])     her kelime bir MATRIS
dizi     her s_t, j <= t icin s_j'lere sorar:  a = relu(q.k + hb)
puan     en yakin E[w]   ==   2 o.E_w - |E_w|^2
         boyut 32 / durum 64, 16,8 M parametre, %99,96'si kelime basina (M)
```

Kullanıcı: *"aslında dimension 32 ve durum 64 ölçümlerde güzel çalıştı
ama kontext büyüdüğünde ne olura bakmadık."* ADIM 1 mimariye dokunmuyor.

Bitiş sonrası analizin söyledikleri (onkayıt §ANALIZ): durum bir kelimeyi
~10-20 token'da unutuyor, uzak bağlamı attention taşıyor; nadir
kelimelerin matrisi rastgele yürüyüş.

---

## ADIM 1 — BAŞ VE SON (BOS/EOS)

```
ONCE    Once upon a time , ... happy .  <dolgu> ...
        ilk kelime hic HEDEF degil, bitis hic sorulmuyor
SIMDI   <hikaye> Once upon a time , ... happy . <hikaye> <dolgu> ...
        <hikaye> -> Once        hikaye nasil BASLAR
        .        -> <hikaye>    hikaye nerede BITER
```

- Kod: `veri_t17.pencere()`. `L + 2 > T` olan hikâye atılır.
- Üretim: `konus` `<hikaye>`'den başlar, model kendi `<hikaye>`'sini
  yazınca durur. `serbest` istemsiz, boş sayfadan hikâye yazar.
- Paket BOS/EOS'u kendisi anlar ve yazar (`bos_eos`, `hikaye_id`).

**Bedeli:** veri değişti, yeni koşu gerekiyor (TAM2). TAM1'den
sürdürülemez: veri izi farklı ve uzatma SÜRDÜRMEDİR (kural 1).

---

## ADIM 2 — ÖLÇÜ KENDİ İÇİNDE

Kullanıcı: *"bizim ölçümlerimiz kendi içinde olmalı. neyi ölçtüğümüz de
önemli. modelin başarı kriteri kendi mimarisine doğru feedback verecek
bir ölçüm olmalı. burada da geri besleme kriterini etkiliyor"*

**2a. Birincil sayı doğrulama perplexity'si.** Doğruluk yalnız en
üstteki tahmini görüyor. TAM1'de 15K-20K arası düz görünürken
perplexity düşmeye devam ediyordu (onkayıt §A1).

**2b. Hedefler türüne göre ayrılır.** BOS/EOS iki yeni hedef ekledi;
tek bir sayı artık TAM1'le aynı şeyi ölçmüyor:

```
bas     BOS'tan sonraki ilk kelime      yeni
govde   hikayenin 2..L. kelimeleri      TAM1'le AYNI hedefler
                                        -> surumler BURADA kiyaslanir
son     EOS                             yeni
```

Günlükte iki sütun: `dg ppl` (hepsi) ve `govde`.

**2c. Her parça kendi geri bildirimini alır** — `olcme_17.tani()`:

```
parca       soru                          olcu
durum       gecmisi ne kadar tasiyor      bir kelime degisince izi kac token surer
attention   uzak baglami okuyor mu        ateslenen konum, kutlenin uzakligi;
                                          oncesi BASKA hikaye olunca CE artisi
okuma       konuma gore tahmin            konum dilimlerinde perplexity
sinirlar    basi ve sonu ogrendi mi       bas/son perplexity'si, gercek sonda
                                          p(EOS), govdede erken EOS orani
```

tani() bir ölçümdür: kullanıcı onayıyla koşulur, her sürümde aynı
örneklemle (aynı `n`, aynı tohum).

**2d. Olgunluk gözle** — `konus`. Transformer kıyası "çok sonra".

---

## YOL HARİTASI — sıra önerisi, kararlar işaretli

```
1  TAM2    YALNIZ BOS/EOS -- mimari ve egitim TAM1'le AYNI    [KOD HAZIR, KARAR]
2  DT1     HEDEF MIMARI (asagida): durum + attention + bellek  [KOD HAZIR, KARAR]
           derinlik, kucuk butce (~3,25 M); ayni veri, ayni olcu
           (govde ppl, tani) -> TAM2'yle KENDI ICINDE
3  secim noktalari: t+1'in yaninda t+2, t+3 tahmini           [ONERI]
4  goz testi, EGITIMSIZ: guzergah tablosu konus'ta dis depo --
           depo yok / sabit oran / "ihtiyac kapisi"; hangisi olgun,
           kullanici bakar.  Egitim istemez, 1-3 ile paralel olabilir.
5  bellek ne kadar kucuk olabilir: guzergahlari K sinifa birlestir;
           buyurse seyrek bellek (product keys)
6  transformer kiyasi -- cok sonra
```

Kullanıcı: *"TAM2 yalnız BOS/EOS olsun"*. TAM2'de değişen tek şey veri
penceresi. Mimari (`model_17.py`) dokunulmadı; MLP ya da yeni katman YOK.
Defterde TAM1'in ayarları aynen: boyut 32, durum 64, lr 0,002, wd 0,01,
tohum 0, yığın 512, T 256, 20.000 adım, bas 100, yedek 250, derle açık.
Kaçınılmaz iki fark: `L + 2 > T` süzgeci (254 kelimeden uzun hikâye
düşer) ve pencere sayısı değiştiği için karışık sıra.

Bu sıranın dayandığı tartışma (23 Eylül):

- **Bilgi güzergâhta.** *"istanbul -->Ankara-->Mersin bu aslında bir
  bilgi, ama istanbul->izmir->mersin bu ayrı bilgi."* ... *"hafıza aslında
  izlediği güzergah ama varaca nokta token."* Sayım (onkayıt §GUZERGAH):
  gerçekleşen kombinasyon olasının yanında yok denecek kadar az; uzun
  güzergâhlar tekil; tablo tek başına TAM1'i geçemiyor ama ona eklenince
  perplexity düşüyor.
- **Birleştirme.** *"farklı güzergah aynı sonuca çıkıyorsa bu iki farklı
  güzergahı tek bir bilgi olarak saklamak mantıklı olabilir."*
- **Denge.** *"modeli ezbere değil öğrenmeye itmek istiyorum. ama insan
  beyni de her zaman mantık ve ya her zaman ezber kullanmaz. ihtiyaç
  duyduğunda kullanır."* Depo küçük tutulursa korpusu ezberleyemez;
  konum başına bir kapı hangisine güveneceğini seçer.
- **Köprü kuralı.** Depo çıkışta değil DÖNGÜNÜN İÇİNDE okunmalı:
  "Türkiye'nin başkenti" okununca durum Ankara'ya dönmeli, "nüfusu"
  gelmeden önce. Aksi halde ikinci adım özneye uygulanır — KISAYOL.

---

## SEÇİM NOKTALARI (öneri, yol haritası 3)

Kullanıcı, 23 Eylül: *"bir güzergah bazlı bir model olduğumuz için eğer
model girdiye göre 20 tane güzergah üretecekse ( 20 next token) ilk
döngüde belirlenecek token herşeyi belirleyen olduğu için onun doğru
belirlenmesi en kritik olan."*

**Bugünkü geri besleme:** her konumda gerçek önekten sonraki token.
Kayıp `−ln p`, doğruyu yukarı itme `1 − p`: bilmediğinde tam güç,
bildiğinde sıfıra yakın. Her token ortalamada EŞİT ağırlıkta.

**İki boşluk:**

- Seçimin SONUCU sayılmıyor: yanlış "Lily/Tom" ile yanlış "the/a" aynı
  kuralla cezalanıyor; biri hikâyeyi belirliyor, öteki hiçbir şeyi.
- Model kendi hatasından sonrasını hiç görmüyor; eğitimde sıradaki
  konuma hep gerçek token geliyor. Ranzato 2016 buna exposure bias
  diyor: *"errors may accumulate along the way."*

**İnceltme:** kritik olan "ilk token" değil, SEÇİM NOKTALARI — nerede
olurlarsa. İstemden sonraki ilk token çoğu zaman bir seçim noktası ama
her zaman değil. Seçimin sonucu ölçülebilir: TAM1'de bir kelime
değişince tahmin KL'si 1 token sonra 2,18, 10 token sonra 0,16
(onkayıt §A4).

**Öneri:** her durumdan t+1'in yanında t+2 ve t+3 de tahmin edilir
(küçük ek okuma başları). Gloeckle 2024 §5.1: böyle bir kayıp *"can
place more emphasis on consequential transitions than inconsequential
ones during teacher-forced training."* Durum yalnız sonraki adımı değil,
güzergâhın nereye gittiğini de taşımak zorunda kalır.

**Ölçü (kendi içinde):** gövde perplexity'si ve "seçim noktalarında
doğruluk". Seçim noktası: güzergâh tablosunda penceresi geniş olan ve
sonrasını değiştiren token. Tanım ve karar kuralı TAM3 koşulmadan ÖNCE
onkayıta yazılır.

Kendi hatasından öğrenme (scheduled sampling, dizi düzeyinde eğitim)
hata birikmesine doğrudan dokunur ama paralel eğitimi bozar; sona kaldı.

---

## HEDEF MİMARİ — durum takibi + attention + bellek katmanı

Kullanıcı, 23 Eylül: *"tasarıma karar vermiştik ben sadece bellek eksik
deyince ... ben o mimari çok beğendim sadece bir hafıza katmanı ... sonra
benzerlik izleri taşısın vs ayrıca normalizasyon kalkacak ve paralel
eşleme gelecek"*

Çalışma adı DT (durum takibi). TAM1'in `M[w]` tablosu gidiyor; yerine
token'dan ÜRETİLEN geçiş geliyor. Bilgi paylaşılan katmanlara taşınıyor.

```
blok (L kez ust uste), artik akis r_t (genislik d):

  durum      k = norm(r Wk)  v = r Wv  q = norm(r Wq)  beta = 2 sigmoid(r wb)
             S_t = S_{t-1} (I - beta k k^T) + beta v k^T      S: 64 x 64
             beta 1: k yuvasi v ile DEGISIR   beta 2: k boyunca YANSIR
             DURUM NORMALIZE EDILMEZ -> ozyineleme dogrusal -> PARALEL egitim
             h_t = S_t q_t                    (normalize yalniz okumada)
  attention  her h_t, j <= t icin h_j'lere sorar      uzak baglam
  bellek     r += W2 . relu(W1 . norm(r))    anahtar = guzergah izi (BENZERLIK),
                                             deger = pencereyi iten vektor
cikis       en yakin E[w]  ==  2 r.E_w - |E_w|^2        (model_15 ADIM 1b kurali)
```

**Bilgi nerede:** kullanıcının itirazına (*"parametre sayısı az. bu durumda
bilgi nerede taşınacak"*) cevap bellek katmanı. Bağlam (bu hikâyede kim,
nerede) DURUMDA, her hikâyede sıfırdan; bilgi (dil, kalıplar, güzergâh
pencereleri) BELLEKTE, kalıcı. Anahtarlar benzerlikle eşleşir: benzer
güzergâh aynı yuvayı açar — kullanıcının *"farklı güzergah aynı sonuca
çıkıyorsa ... tek bir bilgi olarak saklamak"* ilkesi. Büyüdüğünde yoğun
bellek yerine seyrek bellek (product keys, Lample 2019).

**Normalize neden kalkıyor:** her adımda normalize etmek özyinelemeyi
doğrusal olmaktan çıkarıyor, dizi boyunca sıralı hesabı zorunlu kılıyordu
(TAM1'de 0,32 sn/adım). `I - beta k k^T`'nin özdeğerleri 1 ve 1-beta;
beta (0,2)'de hepsi [-1,1] içinde: patlamaz. Konum başına normalize
(artık akışta) paralelliği bozmaz, kalabilir.

**KARAR: DERİNLİK.** Kullanıcı, 23 Eylül: *"derinlik olsun, küçük
bütçeyle başla, DT1'i yaz"*. Kod: `model_17.DT` (`baslat(...,
mimari="dt")`), 2 blok, genişlik 256, bellek 1.024, durum 64 x 64:
3.249.924 parametre (test_17 bunu kağıt üstü formülle sınıyor).

Gerilim şuydu — köprü için belleğin, ikinci adım gelmeden okunması
gerekiyor. İki yol var:

```
dongu ICINDE   bellek okumasi durumu degistirir   -> sinirsiz adim, ama
               ozyineleme dogrusal degil          -> PARALEL EGITIM YOK
DERINLIK       1. blok kopruyu cozer ("baskenti" -> Ankara), ayni konuma
               yazar; 2. blogun durumu onu okur    -> paralel korunur, ama
               cozulebilen adim sayisi <= blok sayisi
```

Transformer'da ölçülen de derinlik yolu: köprü erken katmanda çözülüyor,
ikinci adım geç katmanda (Biran 2024). Öneri: DERİNLİK (L = 2-4 blok).

**Kağıt üstü hesap** (sohbette yapıldı, `python` ile sınandı):

```
patlamaz       I - beta k k^T: ozdegerler 1 (d-1 kez) ve 1-beta; beta (0,2) -> [-1,1]
durum takibi   bardak oyunu: beta=2, k=(e_i-e_j)/sqrt2 -> H = takas; 1.000 hamlede
               kayipsiz; H12.H23 != H23.H12 (SIRA onemli)
sulanma yok    beta=1: S_yeni k = v,  k'ya dik her sey AYNEN
               (TAM1 tipi "ekle + normalize"de top sinyali 10 adimda %3'e iniyordu)
donme          tek yansimanin determinanti -1, 3'lu donmenin +1 -> 2 yansima gerekir
maliyet        gecisi uygulamak satir basina 2d (tam matriste d^2)
parametre      kelime basina V.d (TAM1'de V.d^2: V=50K, d=1024'te 52 milyar)
```

**Parametre (kağıt üstü, V = 4.003, durum 64 x 64):**

```
genislik 256, L=2, bellek 1.024 yuva        ~3,3 M   (TAM1'in beste biri)
genislik 512, L=4, bellek 2.048 yuva        ~13 M    (TAM1 duzeyi, 16,8 M)
                                            hepsi PAYLASILAN + kelime vektoru
```

**Uygulama sırası:** önce sıralı döngü (TAM1 gibi, `torch.compile` ile)
ve doğruluk kapıları; DeltaNet'in parçalı paralel hesabı (Yang 2024)
ikinci adım. TAM2 eski mimariyle koşacağı için eski sınıf da model_17'de
kalır; paket `mimari` alanıyla ayrılır, konus ve `tani()` ikisini de tanır.

İlk taslak (bellek katmanı yok, 0,54 M): `belge/analiz/model_17_TAM1/model_17_dt1_taslak.py`.

---

## SABİTLER — her sayının gerekçesi

Kullanıcı, 23 Eylül: *"model içinde sabit rakamlar var formülasyonda
onları belirli bir mantığı oturtmaya çalıştık ama bazılarınn mantığı
olmayabilir bakarsın onlara."*

```
sabit                   deger        gerekce                                      durum
--- DT
beta = 2 sigmoid(.)     (0, 2)       ozdegerler [-1,1]: patlamaz, yansiyabilir    TURETILDI (Grazzi 2025)
beta baslangici         1 (gb0=0)    araligin ortasi, klasik delta kurali         SECIM, olculmedi
genislik                256          okuma rank siniri 32 -> 256, kucuk butce     KARAR + HESAP
durum                   64 x 64      64 yuva; geri yayilim 2,1 GB/blok (B=512)    HESAP
blok                    2            kopru derinlikle: 1 adim = 2 blok            KARAR
bellek                  1.024        genisligin 4 kati                            GELENEK, olculmedi
yansima                 1            3'lu donme 2 ister; ucuz baslangic           HESAP var, secim olculmedi
init 1/sqrt(giris)      Gk Gv Gq W1  birim RMS girdi -> birim RMS cikti           TURETILDI
init 1/sqrt(durum)      Aq Ak Av     birim boylu h -> birim boylu q, k            TURETILDI
init 1/sqrt(bellek)     W2           bellek ciktisi akisla ayni olcekte           TURETILDI
init 1/sqrt(genislik)   E            baslangic puanlari O(1)                      TURETILDI
init N(0,1)             X            akis birim RMS'le baslar                     TURETILDI
hb = 0                  relu esigi   esigin VARLIGI ispatli (eski not); 0 secim    KISMEN
eps 1e-3                okuma norm   bos yuvanin gurultusu sisirilmesin; tipik    GEREKCE VAR,
                                     |Sq| 1-8, yalniz bos yuvada devreye girer    deger keyfi
eps 1e-6                RMS          sifira bolme                                  SAYISAL
relu attention          PAY=False    Gorev A: 0,782/0,289; TAM1'de seyrek, |o|    OLCULDU, ama
                                     sabit kaldi (onkayit §A5)                    BASKA gorevde
lr 0,002, wd 0,01       sabit lr     eski gorevlerden (rakam, 1003 sozluk)        DEVRALINDI, DT'de
                                     TAM1 egrisi 20K'da dusuyordu: cizelge acik   OLCULMEDI
--- VERI / OLCU
T = 256                 pencere      kalan hikaye %89,8 / dolgu %33,7             OLCULDU
sozluk 4.000            kirpma       kapsama %99,26                               OLCULDU
egitim ici olcum        2.000 pncr   ~340 bin hedef: dogrulukta std hata ~0,001   HESAP
                                     (pencere ici bagimlilik haric)
--- ESKI MIMARI -- TAM2 icin DOKUNULMADI ("TAM2 yalniz BOS/EOS")
M = I + 0,1 randn       baslangic    0,1'in gerekcesi YOK                         GEREKCESIZ
E ~ N(0,1)              baslangic    ilk kayip ~26: cok sivri dagilim             GEREKCESIZ (DT'de duzeldi)
boyut 32                okuma        rank siniri 32 (Yang 2018, onkayit #2)       OLCULDU, DAR
```

**Kelime vektörleri.** Boyutları eğitimde DEĞİŞMEZ, bir ayardır; değerleri
rastgele başlar ve her adımda güncellenir. Kelime başına sayı:
TAM1'de 32 (E) + 64 (b) + 4.096 (M) = 4.192; DT'de 256 (X) + 256 (E) =
512. Kelimeye özgü sayı 8 kat azaldı, bilgi paylaşılan katmanlara geçti.

---

## Kaynaklar — hepsi makalenin kendi metninde doğrulandı

```
durum takibi     Merrill, Petty, Sabharwal 2024  The Illusion of State   2404.08819
                 Grazzi vd. 2025  negatif ozdegerler                     2411.12537
olceklenen gecis Yang vd. 2024  DeltaNet                                 2406.06484
                 Siems vd. 2025  DeltaProduct                            2502.10297
kelime = matris  Sutskever, Martens, Hinton 2011  MRNN (ICML)
                 Foerster vd. 2017  ISAN                                 1611.09434
                 Krause vd. 2016  mLSTM                                  1609.07959
guzergah deposu  Geva vd. 2021  FFN = anahtar-deger bellegi              2012.14913
                 Khandelwal vd. 2020  kNN-LM                             1911.00172
                 Liu vd. 2024  infini-gram                               2401.17377
                 Lample vd. 2019  product keys                           1907.05242
kopru            Biran vd. 2024  Hopping Too Late                        2406.12775
baglam           Khandelwal vd. 2018  Sharp Nearby, Fuzzy Far Away       1805.04623
                 Daniluk vd. 2017  kisa attention erimi                  1702.04521
okuma            Yang vd. 2018  softmax darbogazi                        1711.03953
secim noktalari  Gloeckle vd. 2024  multi-token prediction, §5.1         2404.19737
                 Ranzato vd. 2016  exposure bias                         1511.06732
```

Metinler ve alıntıları yeniden arayan betik:
`belge/analiz/model_17_TAM1/literatur/` (`python verify.py`).

---

## Kodun kendi denetimi

`python test_17.py` — 15 kapı. Eski mimari ve altyapı (9): sürdürme,
dolgu, akış (int16, okuma sınırı), pencere (BOS/EOS), durdur (3), `coz`
gidiş-dönüş, ölçü kendi içinde. DT (6): parametre == kağıt üstü hesap,
bardak oyunu (1.000 takas), delta kuralı (yuva sulanmaz), nedensellik,
dolgu, sürdürme.
