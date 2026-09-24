# model_18 — TASARIM (PV: nokta ve vektör)

Kullanıcı, 23 Eylül 2026:

> *"amacım mevcut olmayan ama llm olma potansiyeli olan bir mimari yapmak."*

Kullanıcı, 24 Eylül 2026:

> *"biz kendi koyduğumuz kuralı nasıl aşarız diye debelenip duruyoruz."*
> *"transformer mimarisinin bir kısmını ödünç alabiliriz yeni bir dünya keşfetmeye gerek yok."*
> *"unutma kelimeler hep sabit bir yerde. zincirler de sabit şu an (zinciri de hareketlendirmek
> mantıklı olabilir belki öğrenilen zincir her yeni örnekte bir yere gider. benzer zincirlerin
> benzer konumlar alması."*
> *"Bu koşuyu bekleyelim ama önerilerini bir yere yaz"*

**Bu dosya öneridir, karar değildir.** Her öneri tek değişkenli bir koşuyla ya da kâğıt üstü bir
testle sınanır; kararı kullanıcı verir. Ölçülen sayıların yeri `belge/OLCULENLER.md`; burada yalnız
önerinin gerekçesi olan bulgular kısaca anılır.

---

## Fikrin kendisi — DEĞİŞMEZ

```
P[w]      kelimeler koordinatta SABİT noktalar (birim küre, rastgele)
C         zincir: bağlam, kelimelerin toplanmasıyla oluşan nokta
V         vektör sözlüğü: noktayı cevaba doğru taşıyan öğrenilen hareketler (start -> finish)
seçim     hangi hareketin uygulanacağı uzaklıkla / yönle belirlenir
cevap     en yakın sabit nokta
defter    hikâyenin kendi geçmişi (C_j -> sonraki kelime)
```

Yolda koyduğumuz **uygulama kuralları** (top-8, serbest hareket boyu, sabit λ, sabit geometriyle
arama) fikrin parçası değil; sorun çıkaran her biri transformer ailesinin sınanmış çözümüyle
değiştirilebilir.

## Bugünkü yapı — TS_PV_LB_CONTENT2

```
C_order    relative zincir, λ 0,7, 512 boyut (sıra, son ~5 kelime)
C_content  kaydırmasız, lam_w / beta_w kelime x boyut öğrenilir, 512 boyut
seçim      katman 0 uzaklık, 1-3 yön (SELECT "direction"), top-8
denge      LOAD_BALANCE 0,01, sabit sıcaklıkla
defter     top-8, skip 3, gate
çıkış      -e^S_p * |C_m - P|^2
```

## Önerinin dayandığı bulgular (ayrıntı: OLCULENLER.md)

1. **Seçim çöküşü:** top-8'de seçilmeyen vektör gradyan almıyor; derin katmanlarda işi 5-10 vektör
   yapıyordu. Taşınmış noktanın boyu (|C| 1,4 → 14) seçimi eziyordu. SELECT + load balancing ile
   derin katmanlar 106-197 vektöre açıldı; **son katman hâlâ dar**.
2. **Yakın C çiftleri:** farklı kelime isteyen yakın çiftlerin %80'inde son 4 kelime birebir aynı
   (bilgi C'de yok → bellek); kalan %20'de fark C'nin küçük varyanslı yönlerinde (ham uzaklık boğuyor).
3. **Bellek kazancı:** C_content 4.000 adımda +4,3 puan; metinde özne korunuyor, olay örgüsü oluşuyor.
4. **Çıkış:** güven |C_m| büyütülerek ifade ediliyor (~17, noktalar 1'de); e^S_p ~2,8'de kalıyor.
5. **Giriş tarafında benzerlik yok:** P rastgele; "dog" ile "puppy", "Sam" ile "Lily" girişte
   "table" kadar uzak. Zincirler yalnız aynı kelimeler aynı yaşta geçince yakın düşüyor.

---

## Zincir ve positional embedding — karşılaştırma

Kullanıcı, 24 Eylül: *"pozisyon gömmesi vs parametresiz lineer zincir yapsana avantaj ve dezavantaj
olarak"*.

Adil kıyas **zincir** ile **PE + attention** arasındadır. PE yalnız etikettir ("neredesin"), bağlamı
attention toplar. Zincir ikisini birden yapar: etiket (roll = yaş) ve toplama (λ^yaş ağırlıklı toplam).

```
C_order_t = Σ 0,7^yaş · roll^yaş(P[w])

sabit desenli attention   içerik skoru SIFIR, eğimi m = ln(1/0,7) = 0,357 olan normalize edilmemiş ALiBi
roll = dönme              DFT tabanında 512 eşit aralıklı frekans; RoPE'nin akrabası, iki farkla:
                            RoPE yalnız q·k'yı döndürür, değer saf kalır -- bizde kelimenin KENDİSİ döner
                            RoPE frekansları geometrik, yakında benzerlik yavaş düşer
                            -- bizde eşit aralıklı: aynı kelime yaş 3'te ve yaş 4'te DİK
```

| ölçüt | lineer zincir (C_order) | PE + attention |
|---|---|---|
| parametre | 0 | mutlak PE T×d (bizde 524.288); RoPE, ALiBi, NoPE 0; + attention ağırlıkları |
| bağlamdan ne seçilir | hiçbir şey: ağırlık yalnız yaşa bağlı | içeriğe göre (q·k), öğrenilir |
| erişilebilir menzil | doğrusal okumayla ~son 4 kelime (hesap aşağıda) | pencerenin tamamı |
| üretimde token başı | sabit, O(d); ama `devam` bugün her adımda öneki baştan hesaplıyor | KV cache, O(T·d) |
| aynı bağlam başka konumda | aynı nokta (ölçüldü: kNN %13,3 → %34,8) | mutlak PE'de farklı; RoPE/ALiBi'de göreli |
| sıra kesinliği | çok yüksek: yaşlar dik | öğrenilir |
| araya bir kelime girerse | eski kelimelerin katkısı başka vektöre döner | attention o kelimeyi yine bulur |
| uzun girdi | yapıda sınır yok (kodda `assert t_max`); ÖLÇÜLMEDİ | mutlak PE: yok; kopyada ALiBi ve NoPE, RoPE'yi belirgin geçiyor ¹ |
| izlenebilirlik | CM ile geri yürünür | yok |
| attention'a anahtar olarak | induction anahtarı hazır ² | induction için 2 katman ² |

¹ Jelassi ve ark. 2024 (2402.01032): Hard-ALiBi ile ≤50 uzunlukta eğitilip 1000'de neredeyse kusursuz.
² Olsson ve ark. 2022, "smeared key": anahtara bir önceki token'ın anahtarı karışınca in-context
learning tek katmanlı modelde de oluşuyor. Zincir bunun üstel, çok adımlı hâli; defterin anahtarları
(C_j) bunu zaten kullanıyor.

**Hesap** (eğitimsiz; P rastgele, yaşlar dik, tekrar eden kelime yok):

```
|C_order|² payı, yaş k    λ^(2k)·(1 − λ²)
                          yaş 0 %51 · 1 %25 · 2 %12 · 3 %6 · 4 %3      ilk iki %76, ilk dört %94
doğrusal okuma, yaş k     sinyal 0,5·λ^k  >  4003 aday içinde en büyük gürültü ≈ 0,031 × 4,07 = 0,126
                          -> k ≤ 3: son ~4 kelime
```

- Uzaklık ya da yön ölçen her seçim fiilen son 1-2 kelimeye bakar: yakın C'lerin aynı kelime ve
  vektörlere kilitlenmesinin geometrik sebebi.
- ~4 kelime, "farklı kelime isteyen yakın çiftlerin %80'inde son 4 kelime aynı" bulgusuyla (2) uyumlu.
- Bilgi kaybolmuyor, ölçeğin altına gömülüyor: ardışık soyma (CM gibi: çöz, çıkar, λ'ya böl, geri
  kaydır) her adımda aynı sinyal/gürültüyü görür, float32 hassasiyetine kadar onlarca kelime geri
  alınır. Katmanlar (softmax RBF) bunu yapamaz; medyan hikâye (171 kelime) buna da sığmaz.

**Sonuç (öneri):**
- Zincir kalır: parametresiz, üretimde sabit maliyetli, sırayı kesin kodlar, attention'a hazır
  induction anahtarı verir.
- Eksik olan içerikle seçim, yani attention. Konum **nereye bakılacağını** belirlesin, **ne
  getirileceğini** değil: anahtar zincirden, değer saf içerikten.
- Attention'a ayrı PE gerekmeyebilir (ÖLÇÜLMEDİ). Kimi Linear (2510.26692) bütün global attention
  katmanlarında NoPE kullanıyor, konumu ve yakınlık yanlılığını tümüyle lineer katmana bırakıyor.
  Griffin (2402.19427): lineer recurrence + local attention, mutlak PE yok, RoPE yalnız attention'da.
- Khandelwal ve ark. 2018 (1805.04623): LSTM ~200 token kullanıyor; sıra yalnız son ~20 token'da
  önemli, 50'nin ötesi kaba bir anlam alanı. Zincir (yakın sıra) + içerik / attention (uzak) ayrımı
  buna uyuyor.
- Zincir içi seçenek: roll yerine geometrik frekanslı dönme. Yaşlar arası kademeli benzerlik verir,
  yine parametresizdir. Relative zinciri öngören kNN kâğıt testiyle sınanır.

---

## Kâğıt üstünde ELENENLER

| aday | neden |
|---|---|
| Expert-choice, BASE katmanları, Soft MoE | nedensel değil (seçim geleceği görüyor) |
| Hash yönlendirme | seçim bağlamdan kopar |
| Global-batch dengeleme (Qiu 2025, Qwen) | dengeyi zaten bütün batch'te (64 hikâye) hesaplıyoruz |
| Ölü vektör yenileme (Huh 2023) | load balancing sonrası kalan ölü vektör sorunu yok |
| Ortak (shared) uzman | Qwen3 kaldırdı; lehine ölçüm yok |
| FSQ, rotation trick | STE yok; tasarımı değiştirir |
| Mixture of Softmaxes | sınır rank değil kullanım (C_m 40-113 yön) |
| C_order'a çok ölçekli sabit sönme (RetNet) | öğrenilen sönme (C_content) bunu kapsıyor |
| Titans | tek kaynak, tekrar yok, karmaşık |
| Unlikelihood, SimCTG | DITTO ve contrastive decoding aynı işi daha iyi sayılarla yapıyor |
| Kayıpsız sapma dengeleme (DeepSeek) | load balancing işe yaradı — YEDEK |
| Qwen RMSNorm kazancı | X-MoE projeksiyonunun zayıf hâli — biri yeter |

---

## İLK 3 ÖNERİ

### 1. Çıkış küreye + öğrenilen ölçek  (ad önerisi: `OUT_NORM`)

```
Ĉ_m    = C_m / |C_m|
puan_w = s · cos(Ĉ_m, P_w)          s öğrenilir, e^S ≈ 6'dan başlar (2s ≈ 12)
cevap  = en yakın sabit nokta        (kürede: en yakın P = en büyük kosinüs)
```

- **Neden:** P birim kürede; C_m de küreye inince "en yakın nokta" iddiası birebir doğru olur.
  Bugün C_m noktalardan ~17 kat uzakta, "en yakın" yalnız bir yön anlamına geliyor.
- **Beklenen:** son katman "güven hareketi" yapmak zorunda kalmaz (son katman çöküşünün en olası
  sebebi); ara katmanlardaki boy sorunu hafifler.
- **Tasarım kuralı (hesap):** ölçek sabit 2,8 kalırsa kayıp tabanı ≈ 2,76 nat, ppl ≥ 15,8 — ölçek
  ÖĞRENİLMELİ ve yüksek başlamalı.
- **Literatür:** Hoffer, Hubara, Soudry 2018, *Fix your classifier* (1801.04540): sabit ortonormal
  sınıflandırıcı + birim küre + öğrenilen skaler ölçek, ImageNet ResNet50 75,3 → 75,3. vMF kaybı,
  Kumar & Tsvetkov 2019 (1812.04616): sabit gömmelere regresyonda aynı "normu büyütme" arızası.
  NormFace, Wang ve ark. 2017 (1704.06369): ölçeksiz kosinüs softmax yakınsamıyor. nGPT, Loshchilov
  ve ark. 2024 (2410.01131): bütün durum kürede, aynı doğruluğa 4-20 kat az adım (tek kaynak).
- **Risk:** sabit çıkış gömmesinin dil modelinde ölçülmüş bedeli var (Hoffer: sabit word2vec ile
  ppl 74,1 → 81,2). Bu bedeli zaten ödüyoruz; normalizasyon yeni bedel eklemiyor.

### 2. Öğrenilen zincir  (ad önerisi: `CHAIN_EMBED`, parametre `delta_w`)

```
E[w] = P[w] + Δ[w]        Δ öğrenilir, 0'dan başlar
C_order, C_content  ->  P yerine E ile kurulur;  CEVAP hâlâ sabit P
```

- **Neden:** girişte kelimeler arası benzerlik olmadığı için bütün genelleme vektörlere kalıyor.
  "Benzer zincirler benzer konumlar alsın" (kullanıcı) ancak zincire giren noktalar
  öğrenilirse mümkün.
- **İddia korunur:** kelimelerin cevap noktası (P) sabit; yalnız kelimenin zincire nasıl girdiği
  öğrenilir. Δ = 0 başlangıcı bugünkü modelle birebir aynı → kazanç temiz okunur.
- **Maliyet:** 4003 × 1024 ≈ 4,1M parametre.
- **Kâğıt üstü test (önce bu):** kNN testi (relative zincir kararını doğru öngörmüştü: %13,3 → %34,8)
  bu sefer anlamsal girişle: TinyStories birlikte-geçme (PPMI + SVD) vektörleri P'ye eklenir, kNN
  doğruluğu ölçülür. Artmıyorsa öneri elenir.
- **Diğerleriyle ilişki:** zincir anlamsal olarak düzenlenince yakın C'ler gerçekten benzer bağlam
  olur; seçim projeksiyonunun (4. aday) işinin bir kısmı zaten yapılmış olur.

### 3. Gated delta kuralı — matris bellek  (ad önerisi: `C_MEMORY`)

```
S_t = S_(t-1) · α(w_t) · (I − β(w_t) · k_t k_tᵀ) + β(w_t) · v_t k_tᵀ
k_t = P_content[w_t]   (SABİT adres)       v_t = E_content[w_t]
C_content_t = S_t · q_t                    q_t: zincirden okunan sorgu
```

- **Neden:** delta kuralı sabit, birim, birbirine dik anahtarlar ister — P tam öyle. Sabit noktalar
  belleğin ADRESLERİ olur: "Sam"in adresine "arkadaşı Lily" yazılır, yeni bilgi gelince üzerine yazılır.
  Bugünkü C_content aynı işin vektör (köşegen) hâli.
- **Literatür:** GLA, Yang ve ark. 2023 (2312.06635): sabit sönme 16,55 → boyut başına veriye bağlı
  sönme 14,77. HGRN2, Qin ve ark. 2024 (2404.07904), 44M: 24,82 → 23,73 (vektör → matris durum).
  xLSTM, Beck ve ark. 2024 (2405.04517): matris hafıza 17,70 → 13,48. Gated DeltaNet, Yang, Kautz,
  Hatamizadeh 2024 (2412.06464): sönme hızlı siler, delta kuralı hedefli yazar; ikisi birlikte
  recall testlerinde en iyi. Zoology, Arora ve ark. 2023 (2312.04927): ppl farkının %82'si geri
  gelen token'lardan — bizim %80 bulgumuzla aynı tanı.
- **Risk:** en büyük kod değişikliği; defterle kısmen örtüşür (ikisi de "geçmişten getir").

### Sıradaki adaylar (ilk 3'ten sonra)

- **4. Seçimde öğrenilen projeksiyon** (X-MoE, Chi ve ark. 2022, 2204.09179; ad önerisi `ROUTE_DIM`):
  1024 → ~128, sonra kosinüs, öğrenilen sıcaklık, dengede sabit sıcaklık. Makalede yönlendirme boyutu
  uzman sayısının yarısıyken en iyi; bizim oran (1024 / 256) en kötü satıra denk. ViT-VQGAN (Yu ve ark.
  2021, 2110.04627): kod kullanımı 256 boyutta %4, 32 boyutta %96. Kullanıcının tercihi; 1 ve 2'den
  sonra yakın C'ler hâlâ ayrışmıyorsa öne çıkar.
- **5. Döngü:** DITTO, Xu ve ark. 2022 (2206.02369): açgözlüde cümle tekrarı %14,5 → %2,85, ppl de
  iyileşiyor. Contrastive decoding, Li ve ark. 2022 (2210.15097): hikâye tutarlılığı 0,48 → 0,62;
  "amatör" olarak kısa bellekli model (ör. CCACHE) — **kâğıt üstünde, eğitimsiz sınanabilir.**
- **6. Defter:** anahtar/sorgu normalize (var), öğrenilen sıcaklık (var); ham kosinüs en yakın
  geçmişi seçer (Merity ve ark. 2016, 1609.07843) — Q ile birlikte değerlendirilir.

---

## Önerilen bütün mimari (v2)

```
SABİT (iddia)
  P[w]          birim küre, rastgele, sabit              cevaplar ve belleğin adresleri

GİRİŞ                                                                       (öneri 2)
  E[w] = P[w] + Δ[w]

ZİNCİR
  C_order_t   = 0,7 · kaydır(C_order_(t-1)) + E_order[w_t]
  S_t         = S_(t-1)·α(w_t)·(I − β(w_t)·k_t k_tᵀ) + β(w_t)·v_t k_tᵀ       (öneri 3)
  C_content_t = S_t · q_t
  C_t         = [ C_order_t | C_content_t ]                    D_SUM 1024

HAREKET (4 katman, üst üste)
  seçim     katman 0 uzaklık; 1-3 yön; top-8
  denge     LOAD_BALANCE 0,01 (sabit sıcaklık)
  C_m       = C + Σ W · (finish − start)

DEFTER      hikâyenin geçmişi, top-8, skip 3, gate

ÇIKIŞ                                                                       (öneri 1)
  puan_w = s · cos(C_m, P_w),  s öğrenilir;  cevap = en yakın sabit nokta

ÜRETİM      top-p 0,9; sonra contrastive decoding (aday 5)
```

## Sıra

```
kâğıt üstü (eğitimsiz, onayla)   PPMI kNN (öneri 2 var mı)  ·  contrastive decoding (aday 5)
tek değişkenli koşular           zemin TS_PV_LB_CONTENT2;  önce 1 (en küçük), sonra 2, sonra 3
her koşuda                        accuracy · seçim sağlığı (her C / bütün C) · örnek istemler (döngü, özne)
```

Adlar (`OUT_NORM`, `CHAIN_EMBED`, `delta_w`, `C_MEMORY`, `ROUTE_DIM`) önerimdir; kullanıcı koyar.
