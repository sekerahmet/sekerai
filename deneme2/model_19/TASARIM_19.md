# model_19 — TASLAK

**Taslaktır, sabitlenmiş tasarım değildir.**

Kullanıcı, 25 Eylül 2026: *"çünkü hala taslak bir öneri bu sabitleşmiş bir taslak değil."*

**model_18 bağlayıcı değil.** Kullanıcı, 25 Eylül:

> *"model 18 den bazı şeyleri alıyoruz ama burda yeni bir model tasarlıyoruz gibi düşünebilirsin. model 18 bizi bağlayan
> birşey değil. evet orda öğrendiğimiz şeyler var ama burda farklı birşey yapıyoruz model 18 ayağımıza pranga olmalalı."*

(Okumam: "olmamalı".)

> *"yeni birşey yazıyoruz ama herşeyi sıfırdan yazmayalım model 18 de ortak ise kullanabiliriz manasına söyledim.
> gerekiyorsa baştan bile yazılabilir ama orda değişken ve bazı kavramlar oturudu ama bir noktadan sonra çok karışık hale
> geldi."*

Öneriler kabul edildi. Kullanıcı: *"Önerilerin hepsi kabul."* (Ö1–Ö12) ve *"Önerilerini kabul ediyorum bence
mantıklı."* (Ö13–Ö15). Her parça bir koşuyla sınanır (kural 0: koşu onayla). Ölçülen sayıların yeri
`belge/OLCULENLER.md`. Görsel: artifact `model_19 Üç İlişki`.

## Fikir

Kullanıcı, 25 Eylül:

> *"Aklıma bir fikir geldi. P ler bir matris olmalı buna point to point relation diyelim. Yani once kelimesinin diğer
> kelimelerle olan relation. Eğitim boyunca öğrenilir. Burda bir skor olur. Mesela once ile upon çok yakındır ama once
> ile shower uzaktır. Bunun dışında P ile C bir matris olur yani once upon chain i a ile yakın olur bu da öğrenilir.
> Bir de C 2 C matrix olur. Model bunların değerlerini ve ne zaman kullanacağını öğrenir bunu da attention ile yapar."*

Tablo (kullanıcının çizimi): `R_P1_P1 öğrenilir` (P × P), `R_P1_C1 öğrenilir` (P × C), `R_C1_C1 öğrenilir` (C × C).
Hücreler yönlü: R_P1_P2 ile R_P2_P1 ayrı.

## Terimler

| terim | ne | öğrenilir mi |
|---|---|---|
| **P[w]** | Kelimenin **cevap noktası**: model bir kelimeyi söyleyeceği zaman buna bakar. Birim kürede, rastgele. | hayır, sabit |
| **E[w]** | Kelimenin **giriş noktası** (embedding): kelime okunup zincire girerken kullandığı yer. | evet |
| **C_t** | Zincir: t'ye kadar okunan kelimelerin E'lerinden kurulan bağlam noktası. Kullanıcı: *"modelde oluşan zincirler"*. | kurulum kuralı |
| **C_m** | Zincirin katmanlardan (hareketler, attention) geçmiş hâli. | katmanlar evet |
| **R_PC** | Zincirden kelimeye ilişki: "once upon" → "a". | evet |
| **R_CC** | Zincirden zincire benzerlik: "The little" ~ "a little". Kullanıcı: *"evet benzerlik zincir benzerliği"*. | evet |
| **defter** | Hikâyenin kendi geçmişi: şimdiki zincire benzeyen eski zincirden sonra gelen kelimeye oy verir. | benzerliği evet |
| **gate** | Konum başına 0–1 arası tek sayı: deftere ne kadar güvenilecek. | evet |

**E nedir, neden P'den ayrı?**
- Bir kelimenin iki işi var: okunmak (girişte) ve söylenmek (çıkışta). model_18'de ikisi de aynı sabit P'ydi.
  - "dog" ile "puppy" girişte "table" kadar uzaktı.
  - Kelime en çok kendine yakındı; yankının kaynağı buydu.
- Burada ikisi ayrılır: P çıkışta sabit kalır, E girişte öğrenilir. "dog" ile "puppy"nin E'leri yakınlaşabilir, P'leri
  ayrı kalır.
- Kullanıcının R_PP'si E'de yaşar. Kullanıcı: *"P0 = C0 olarak bul edersek R_P_P ihtyaç kalmaz ama o ilişki embed
  olarak girmeli sanki değil mi ?"*. İlk konumda zincir yalnız kelimenin kendisi; çıkışta ayrı bir R_PP, R_PC ile
  çakışırdı.

## Taslak mimari

```
E[w]       = norm( P[w] + ΔE[w] )                             giriş noktası, kürede (Ö1)
               ΔE: hikâyede U[w]·Wᵀ (U: V×r, W: D×r, r = 256); matematikte tam tablo V×D (Ö8)
C_t        = [ sıra yarısı | içerik yarısı ] ( E[w_0..t] )     zincir, iki yarı (aşağıda)
C_m        = V0 → attention_1 → V1 → V2 → attention_2 → V3 (C_t)                 (Ö13: ikinci attention)
               attention puanı:  q·k/√d − m_h · (t − j),   m_h baş başına öğrenilir  (Ö14: mesafe eğilimi)
puan(w)    = P[w]ᵀ · ( s · Ĉ_m  +  A_PC · B_PCᵀ · C_t )                                TOPLAM
sim(t, j)  = cos( (I + A_CC · B_CCᵀ) · Ĉ_t ,  Ĉ_j ),   j < t − 3                       (Ö2)
gate       = σ( Ĉ_m · g_d  +  g_s · max_j sim(t, j)  +  g_0 )                          (Ö7, Ö9)
p          = (1 − gate) · softmax(puan)  +  gate · p_defter
```

- **Cevap hâlâ en yakın sabit nokta.** puan(w) = P[w]ᵀ·z, z = s·Ĉ_m + A_PC·B_PCᵀ·C_t. Cevap, birleşik z noktasına en
  yakın sabit P; R_PC noktayı taşıyan öğrenilen bir hareket.
- **Zincirin iki yarısı kalır.** Kullanıcının sorusu: *"ilk yarı ve ikinci yarı model 18 den gelen bir öğrenmeydi. bu
  mimaride ihityaç olacak mı ?"*.
  - Sıra yarısı yakın sırayı tutar: her kelime yaşı kadar kaydırılır, 0,7^yaş ile söner.
  - İçerik yarısı daha uzun kalan konu torbasıdır: sırasız, kelime başına öğrenilen sönme.
  - Tek bir sönme hızı ikisini yapamaz. 0,7 on kelimede unutur; 0,95 sırayı bulandırır.
  - R_CC bunların yerini tutmaz: saklamaz, yalnız karşılaştırır. "a little dog" ~ "a little puppy" yakınlığı asıl E'den
    gelir.
  - İçerik yarısı model_18'de ölçüldü: 4.000 adımda +4,3 puan.
- **E iki yarıya da girer.** E'nin sıra kısmı sıra yarısına, içerik kısmı içerik yarısına gider; ikisi de öğrenilir.
- **İkinci attention işlenmiş durumu okur.**
  - Birinci attention: sorgu C_m'den, anahtar bir önceki ham zincirden (model_18'deki gibi). Bu yapı induction'ın ilk
    başını gömülü taşır.
  - İkinci attention: sorgu, anahtar ve değer işlenmiş durumdan (C_m). Böylece birincinin ve V katmanlarının yazdığını
    okuyabilir; üst üste binme budur.

## Açık sorulara verilen cevaplar

| soru | cevap | kaynak |
|---|---|---|
| attention kalsın mı | evet; attention'sız model sabit kurallı bir özyineleme olur, geri çağırmada zayıf | kullanıcı: *"attention olmadan doğru seçilim olmaz gibi geliyor"*; model_18 teşhisi: attention kapalıyken ad doğruluğu ~0 |
| hareket sözlüğü V kalsın mı | evet; toplamla kurulamayan birleşimleri ("upon a" → time) doğrusal olmayan tek parça o kuruyor | öneri |
| zemin | model_19'un kendi zemini: kendi kodu, yeni parçalar kapalı (E = P, R_PC = 0, R_CC = birim, ikinci attention ve mesafe eğilimi yok) | kullanıcı: *"model_19 kendi zemini olsun. çünkü bir başka zemin sürekli bizi doğruya gitmekten ve bağımsız düşünmekten alıkoyuyor"* |
| matematikte D | 128. Rank D'yi aşamaz; 128'de matrisler tam (128 × 128, 16k parametre), sıkıştırma gerekmez | kullanıcı: *"matematik çok küçük bir sözlük olduğu için deneme olarak 128 yaparız yetmezse ve yetmediğini test edersek 256 yaparız"* |
| D ne zaman büyür | "D yetmiyor" bulgusundan sonra; GPU hızlıysa 128 ve 256 yan yana da koşulabilir | kullanıcı: *"bunlar net bir şekilde dimension yetmiyor bulgusundan sonra yapmayı tercih ederima ma GPU koşusu çok hızlı ilerlerse doğrudan GPU koşusuda yapabiliriz"* |
| hikâyede D ve r | model_19'un kendi zemininde ölçülerek; r = 256 başlangıç | kullanıcı: *"R 256"* |
| kod | model_18'in ortak parçaları kullanılır, yalnız seçilmiş yol taşınır; denenip bırakılmış seçenekler taşınmaz | kullanıcının sözü yukarıda; öneri |

**"D yetmiyor" ölçütü** (sayısı koşudan önce yazılır):
- Kapasite eksikliği belirtisi: train doğru cevap oranı düşük kalır ve eğri düzleşir; model eğitim sorularını bile
  çözemez.
- Train yüksek, heldout düşükse sorun D değil.

## Transformer'ın çözdüğü sorunlar, bu modeldeki karşılıkları

Kullanıcı: *"transformer mimarisi de tek seferde çıkmadı oradaki her parça belirli bir sorunu çözmek için ... belki de
transformer mimarisinin çözdüğü sorunları bu modelde hangileri çözüyor diye bir tablo yapsak eksik noktaları da
belirlemiş oluruz"*.

| transformer parçası | çözdüğü sorun | bu modelde | durum |
|---|---|---|---|
| Token embedding | kelimeye öğrenilen anlam; benzer kelimeler yakın | E | var (model_18'de yoktu) |
| Konum kodlaması | attention sırayı bilmez | zincirin sıra yarısı + attention'da mesafe eğilimi | var (Ö14 ile) |
| Self-attention | geçmişin herhangi bir konumundan içeriğe göre bilgi getirmek | attention_1, attention_2 | var |
| Çok baş | aynı anda farklı ilişkiler (sözdizimi, kim kimi) | baş başına | var |
| Nedensel maske | geleceği görmeden eğitim | zincir, attention, defter nedensel | var |
| FFN | konum başına doğrusal olmayan hesap; bilgi deposu | V sözlüğü: yöne göre seçilen başlangıç (anahtar) ve hareket (değer) | var; seyrek (top-8) |
| Artık bağlantı | her katman ekler, bilgi kaybolmaz, gradyan akar | hareketler ve attention çıktısı noktaya eklenir | var |
| LayerNorm | ölçek kontrolü, kararlı eğitim | C_m çıkışta küreye; E küreye (Ö1); katman girdileri ölçülür (Ö15) | kısmen, ölçülecek |
| Derinlik (blok tekrarı) | bir attention'ın bulduğunu başka birinin kullanması | iki attention bloğu | var (Ö13 ile) |
| Çıkış (unembedding) | durumdan kelime dağılımına | sabit P ile cos + R_PC | var; sabit rastgele taban bilinçli seçim |
| Paralel eğitim | RNN'in sıralı eğitim darboğazı | zincir doğrusal, bütün dizi paralel | var |
| Kopyalama (induction) | bağlamdaki kalıbı ve adı sürdürmek | defter + R_CC; attention_1'in anahtarı bir önceki zincir | var |

Dayanaklar (metinde doğrulandı):
- Geva ve ark. 2020 (`2020/p_2012.14913.txt`): *"feed-forward layers in transformer-based language models operate as
  key-value memories"*.
- Olsson ve ark. 2022 (`2022/olsson2022_induction.txt`): induction *"implemented by a circuit consisting of a pair of
  attention heads in different layers"*.

## Öneriler (hepsi kabul edildi)

**Ö1. E kürede (birim).**
- Sınırsız E zinciri büyütür. model_18'de iki kez görüldü:
  - Nokta boyunun 1,4'ten 14'e büyümesi seçimi eziyordu.
  - Model güvenini boyu büyüterek ifade ediyordu.
- R_PC terimi C_t ile doğrusal olduğu için bu yolu yeniden açardı.

**Ö2. R_CC benzerliği sınırlı: cos((I + A·Bᵀ)·Ĉ_t, Ĉ_j).** Defter kodu benzerliğin −1 ile 1 arasında olduğunu
varsayıyor: −2,0 işareti, `sim > −1,5` eşiği, gate'e giren en yüksek benzerlik.

**Ö3. R_CC yalnız seçilen komşulardan öğrenmesin.**
- Eğitimde bütün geçmişe bakılır (`topk = None`).
- Sağlık satırında "farklı seçilen komşu sayısı" yazılır.
- Gerekçe: model_18'in ilk çöküşü aynı biçimdeydi, seçilmeyen gradyan almıyordu.

**Ö4. Yeni parçalar sıfır etkiyle ve ayrı bir rastgele üreteçten başlar.**
- A küçük rastgele; B, W ve m_h sıfır.
- Amaç, her parçanın katkısını ayırmak.
- Eşitlik testi: yeni parçalar kapalıyken 0. adımda puanlar zeminle aynı mı. Test gerçek token'larla beslenir.

**Ö5. R_CC'nin karar kuralında döngü yer alır.**
- model_18'de defter uzak kalıbı besliyordu.
- Heldout artarken uzak kalıp artıyorsa bu takas olarak yazılır. Gözle okuma da yapılır.

**Ö6. Yeni parçaların büyümesi ölçülür.**
- ‖A·Bᵀ‖, ‖ΔE‖ ve m_h adım adım günlüğe yazılır.
- Ayrı bir LR ya da ölçek katsayısı gerekip gerekmediğine buna bakarak karar verilir.
- Gerekçe: Adam, sıfırdan başlayan bir matrisi faydasından bağımsız bir hızla büyütür.

**Ö7. Gate küçük kalır; "büyük defter" ayrı bir kaynak olur.**
- Gate karar verir, bilgi saklamaz.
- model_18'de gate'in sorunu boyut değil, döngüde açılmasıydı: kopya dışında medyan ~0,005; sık döngüde turdan tura
  0,19 → 0,61.
- Literatürde büyük bellek bile küçük bir karıştırıcıyla birleşir (Grave 2016; kNN-LM 2019).

**Ö8. Embedding kelime başına tablo, P uzayında tek matris değil.**
- Sabit rastgele P üzerinde tek bir D × D matris, 8004 kelimeye bağımsız yer veremez (8004 > D).
- Matematikte (13 kelime) fark yok.

**Ö9. Gate'in girdileri görünür yazılır ve decompose'da okunur.** İki girdi var: Ĉ_m'nin yönü ve en yüksek defter
benzerliği.

**Ö10. Matematik zemini model_19'un kendi koduyla koşulur.**
- Veri "cok" (2 ve 3 terim). Her kol 3 tohum.
- Kayıttaki MAT_COK_PV (0,4791; erken model_18 kodu, D 128) kıyas değil.

**Ö11. Parçalar birer birer eklenir.** R_CC hikâyede sınanır; toplamada defterin işe yarama ihtimali düşük.

**Ö12. Matematikte nitel işaret:** E rakamları 0'dan 9'a sıraya diziyor mu.

**Ö13. İkinci attention katmanı (V2'den sonra), işlenmiş durumu okur.**
- Bir baş "özne kim"i bulsa bile başka bir baş bunu kullanamıyordu.
- Kim kimdir hatasıyla ilişkisi bir hipotez, ölçülmedi.

**Ö14. Attention'da baş başına öğrenilen mesafe eğilimi: −m_h·(t − j).**
- Başlar yakını uzaktan ancak içerikten ayırt edebiliyordu.
- Matematikte de anlamlı: toplamada cevabın rakamı, sorudaki belli uzaklıktaki rakamlara bağlı.

**Ö15. Katman girdilerinin ölçeği ölçülür; sorun çıkarsa her katmanın girdisi normalize edilir.** model_18'de |C_m|
~17'ye büyüyordu; hareketlerin göreli etkisi küçülüyor olabilir.

## Sıra

**Birleştirme** (her adım yalnız öncekinin ölçümü gerek gösterirse):
1. **Toplam.**
2. **Kaynak başına kapı.**
3. **Kaynaklar üzerinde attention.**

**Sınama** (karar kuralları koşudan önce; her kol 3 tohum):

| # | veri | kol |
|---|---|---|
| 1 | matematik | zemin (yeni parçalar kapalı), D 128 |
| 2 | matematik | + E |
| 3 | matematik | + R_PC |
| 4 | matematik | + mesafe eğilimi |
| 5 | matematik | + ikinci attention |
| 6 | hikâye | zemin (model_19'un kendi kodu, yeni parçalar kapalı) |
| 7 | hikâye | + E |
| 8 | hikâye | + R_PC |
| 9 | hikâye | + mesafe eğilimi |
| 10 | hikâye | + ikinci attention |
| 11 | hikâye | + R_CC (Ö5'in kuralıyla) |
| 12 | — | birleştirmenin 2. ve 3. adımları, gerekirse |

Parametre:
- Matematik (V = 13, D = 128): ΔE 1.664, R_PC 16.384, R_CC 16.384, m_h baş başına 1.
- Hikâye (V = 8004, D = 1024 ise, r = 256): ΔE 2.311.168, R_PC 524.288, R_CC 524.288.

## Eğitim (kabul edildi)

Kullanıcı: *"doğru eğitim doğru sonuç verir"* ve *"Evet eğitim kısmı onaylıyorum ama çok belirgin bir arıza varsa tek
tohum"*.

| karar | değer |
|---|---|
| hedef | sonraki token; matematikte kayıp yalnız cevap rakamları ve EOS'ta, hikâyede bütün token'larda |
| optimizer | Adam, weight decay yok |
| LR programı | sabit + son %20 cosine soğutma; ısınma ve taban Aşama 0'da seçilir (`train_19.lr_at`) |
| yeni parçalar | sıfır etkiyle başlar; boyları sağlık satırında izlenir (Ö6), gerekirse ayrı LR |
| bütçe (matematik) | batch 4096, 20.000 adım (kural 1'in ilk sınırı) |
| tohum | kol başına 3; tohum 0'da belirgin arıza varsa tek tohum |
| karar | tutulan sorularda birebir doğru; kurallar koşudan önce `belge/onkayit/model_19.md`'de |
| iki tür deneme | ayrı izlenir. **Eğitim denemesi** (`*_TRAIN_*`) yalnız tarifi değiştirir, model = zemin. **Model denemesi** (`*_ARCH_*`) yalnız modeli değiştirir, tarif sabit. İkisini birden değiştiren koşu başlamaz; tür pakette ve günlükte. Kullanıcı: *"model içinde denediklerimiz ve eğitimde denediklerimiz ayrı şekilde takip etmeliyiz"* |

**Aşamalar:**
1. **Aşama 0: tarif.** Yalnız zemin, tohum 0. LR tepesi 0,001 / 0,002 / 0,004 / 0,01; seçilen LR ile ısınma ve 0,0001
   tabanı.
2. **Aşama 1: kollar.** Tarif sabit: zemin, +E, +R_PC, +mesafe eğilimi, +ikinci attention.
3. **Aşama 2: hikâye.** Önce tarifin zeminde doğrulanması, sonra kollar ve R_CC. Döngü için eğitim tarafında bir aday
   sonraya bırakıldı: DITTO (Xu 2022, klasörde): *"the model learns to penalize probabilities of sentence-level
   repetitions from pseudo repetitive data"*.

## Transformer ile karşılaştırma: avantajlar ve dezavantajlar

Her satırda dayanağın türü yazılı:
- **ölçüldü:** model_18 kayıtlarında.
- **yapısal:** mimariden doğrudan çıkar.
- **beklenti:** ölçülmedi.

model_18'de ölçülenler o koda aittir (kural 11). model_19'a taşınmadan önce yeniden doğrulanır.

**Bu mimarinin avantajları**

| avantaj | açıklama | dayanak |
|---|---|---|
| Kesin ayrışma | Bir kelimenin puanı parçalarına kesin ayrılıyor: geçmiş kelime başına zincir payı, baş başına attention, vektör başına hareket, defter. Transformer'da LayerNorm ve MLP yüzünden ayrışma yaklaşık. | ölçüldü: decompose + verify (model_18) |
| Açık kopyalama | Kopyalama ayrı bir kaynakta (defter) ve ne kadar kullanıldığı gate'te görünüyor; kapatılabilir, ölçülebilir. Transformer'da kopyalama başlara dağılmış. | ölçüldü: döngünün defter ve zincir kaynaklı olduğu müdahalelerle gösterildi |
| Yakın sıra öğrenmesiz | Sıra yarısı konumu kaydırmayla, öğrenilen parametre olmadan veriyor; göreli olduğu için mutlak bir uzunluk sınırı yok. | yapısal; uzun metne genellemesi beklenti |
| Sabit cevap noktaları | Çıkış tarafı eğitimle kaymıyor; "en yakın sabit nokta" geometrisi okunabilir. | yapısal |
| Küçük sözlükte ucuz çıkış | model_18 aynı L4'te 0,146–0,229 sn/adım, TinyStories-3M 0,776. | ölçüldü, ama fark büyük ölçüde sözlükten: TS-3M'in 50.257'lik logit tablosu. Mimarinin avantajı sayılmaz |

**Bu mimarinin dezavantajları**

| dezavantaj | açıklama | dayanak |
|---|---|---|
| Döngü | PV açgözlü üretimde transformer'dan çok daha fazla döngüye giriyor. Sık döngü payı: PV t20000 0,223, TinyStories-3M 0,060 (3,7 kat). | ölçüldü (model_18) |
| Aynı hikâyelerde daha kötü sıkıştırma | 150 tutulan hikâyede: TinyStories-3M 0,557 bpc, PV 0,700 bpc. | ölçüldü (eski PV kodu, v2 verisi; kıyas kaba) |
| Sabit rastgele çıkış tabanı | Olasılık dağılımı P'nin rastgele alt uzayına sıkışıyor. Transformer'ın çıkış matrisi öğreniliyor. | yapısal; Yang 2018 (softmax darboğazı, klasörde) |
| Çok türlü parça, çok ayar | Zincir, seçimli sözlük, defter, gate, ilişkiler. Elle konmuş sabitler var: λ 0,7, skip 3, top-8, aktif 8. Transformer birkaç parçanın tekrarı. | yapısal; `olcumsuz-secimler` belleği |
| Seçimli FFN'nin çöküşü | Top-k seçimde seçilmeyen vektör öğrenmiyor; model_18'de derin katmanlarda işi 5–10 vektör yaptı, denge terimi gerekti. Transformer'ın yoğun MLP'sinde bu sorun yok. | ölçüldü (model_18) |
| Ölçeklenme bilinmiyor | Transformer için ölçeklenme yasaları ve donanım çekirdekleri (flash attention) var; burada hiçbiri yok. | yapısal |
| Maliyette üstünlük yok | Zincir sabit boyutlu, ama tam attention kaldıkça uzun bağlam maliyeti transformer'la aynı sınıfta. | yapısal |

**Henüz bilinmeyen: model_19'un kendi sorusu.**
- Öğrenilen ilişkiler (E, R_PC, R_CC) ve iki eksik parça (ikinci attention, mesafe eğilimi) yukarıdaki dezavantajlardan
  hangisini kapatır? Beklenti:
  - E ile R_PC yankıyı azaltır.
  - İkinci attention ve mesafe eğilimi kim kimdir hatasına yardım eder.
  - R_CC döngüyü artırabilir (Ö5).
- Hiçbiri ölçülmedi. Sınama sırası bu soruyu cevaplamak için.

## Literatür (metinde doğrulandı)

- **Elhage ve ark. 2021**, `2021/elhage2021_framework.txt`:
  - *"Zero layer transformers model bigram statistics."*
  - *"the optimal behavior of W_U W_E is to approximate the bigram log-likelihood."*
  - Levy ve Goldberg 2014'e atıfla: *"many early word embeddings can be seen as matrix factorizations of a
    log-likelihood matrix."*
- **Olsson ve ark. 2022**, `2022/olsson2022_induction.txt`: induction başları iki katmanlı bir devre; *"key shifting"*,
  önceki bir *"previous token head"* ile.
- **Geva ve ark. 2020**, `2020/p_2012.14913.txt`: FFN *"key-value memories"*.
- **Jelassi ve ark. 2024**, `2024/jelassi2024_repeat.txt`: *"a two layer transformer can copy strings of exponential
  length while GSSMs are fundamentally limited by their fixed-size latent state."*
- **Arora ve ark. 2023 (Zoology)**, `2023/arora2023_zoology.txt`: *"a 70M parameter attention model outperforms a 1.4
  billion parameter gated-convolution model on associative recall."*
- **Grave ve ark. 2016**, `2016/grave2016_cache.txt`: *p = (1−λ)·p_vocab + λ·p_cache*.
- **Khandelwal ve ark. 2019 (kNN-LM)**, `2019/p_1911.00172.txt`: *"linearly interpolating its next word distribution
  with a k-nearest neighbors (kNN) model"*.
- **Sonraya:** `2024/p_2404.19737.txt`, *Better & Faster Large Language Models via Multi-token Prediction*. Okunmadı.

## Sonraya bırakılan

- **Kelimeden zincire ilişki:** "next token değil de next 3 token". Kullanıcı: *"ama o kısım kalsın"*.
- **Büyük defter:** bütün eğitim verisinin zincir → kelime belleği, ayrı bir kaynak (Ö7).
- **İki yarıyı birleştirmek:** tek zincirde boyut başına öğrenilen sönme.
