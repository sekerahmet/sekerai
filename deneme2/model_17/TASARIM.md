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

## YOL HARİTASI — karar verilmedi, sıra önerisi

```
1  TAM2    ADIM 1 + ADIM 2, ayni mimari, 20.000 adim          [KOD HAZIR]
2  goz testi, EGITIMSIZ: guzergah tablosu konus'ta dis depo --
           depo yok / sabit oran / "ihtiyac kapisi"; hangisi olgun,
           kullanici bakar
3  depo ne kadar kucuk olabilir: guzergahlari K sinifa birlestir
4  ogrenen depo + kapi, DONGUNUN ICINDE
5  transformer kiyasi -- cok sonra
```

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

## BEKLEYEN — ölçeklenebilir kodlayıcı

"LLM potansiyeli" için kelime başına matris ölçeklenmez: V·d² (V=50K,
d=1024'te yalnız M 52 milyar). Aday: token'dan üretilen yansıma geçişi,
`S <- S(I - beta k k^T) + beta v k^T`, beta (0,2) (DeltaNet ailesi).

Taslak kodlandı ve DURDURULDU. Kullanıcının itirazı — *"parametre sayısı
az. bu durumda bilgi nerede taşınacak"* — doğru çıktı: 0,54 M parametre,
bilgi deposu yok. Taslak: `belge/analiz/model_17_TAM1/model_17_dt1_taslak.py`.

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
```

Metinler ve alıntıları yeniden arayan betik:
`belge/analiz/model_17_TAM1/literatur/` (`python verify.py`).

---

## Kodun kendi denetimi

`python test_17.py` — 9 kapı: sürdürme, dolgu, akış (int16, okuma sınırı),
pencere (BOS/EOS), durdur (3), `coz` gidiş-dönüş, ölçü kendi içinde.
