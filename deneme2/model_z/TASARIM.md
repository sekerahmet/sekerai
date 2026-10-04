# Model Z

Kullanıcı, 3 Ekim 2026: *"belki model z deyip yeni birşey denesek öğrendiklerimizle farklı bir yaklaşım"*;
*"Yeni bir model için yeni bakış açısı gerekiyor. Dil nedir. Dil matematiksel olarak nasıl ifade edilir."*;
*"şimdi model Z klasörünü aç"*.

## Gramer ajanı (ilk parça)

Kullanıcı, 3 Ekim 2026: *"Gramer ajanının temel görevi verilen tüm kelimelerden anlamlı bir cümle kurabilmesi"*;
*"Next token mantığında değil tek seferde deneyerek bulmalı yani tek seferde bir cümle. Yuvalar doğru olacak."*;
*"en hızlı en efektif ve en doğrusu olmalı"*; *"Görevi makine bulmayacak görev diye birşey yok"*.

- Girdi: cümlenin kelimeleri, karışık. Çıktı: her kelimeye bir yuva (konum), bütün cümle bir anda.
- Kelime ↔ kelime ilişki matrisi; görev etiketi yok.
- Başarı: tek seferde doğru cümle. Ölçüler: tam doğru, doğru yuva, doğru komşu, deneme sayısı.
- Veri: `data/countries/` (hazır dosyalar). Sınav: eğitimde görülmüş ve görülmemiş olgu.

Durum (4 Ekim): kod `core/grammar/` (grammar.py ajan, train_grammar.py eğitim); dizme ilişki matrisi üzerinde tek çevrim,
ilk k aday `order_alternatives`. Veri: ülke, SS 200k, SS tamamı (`make_ss_sentences.py`, Drive).

## Akış: context agent + grammar agent

Kullanıcı, 4 Ekim 2026: *"k bağımsız bir yapı olmalı ve grammar endeks ile baştan torbalar elenmeli"*; *"benim istediğim
diye birşey yok doğru tasarım ne"*; tasarıma *"Tamam beraber eğitelim o değer de not al sonra bir ölçüye bağlarız"*.

- İş bölümü: context agent içeriği ve eksiksizliği (torba olasılığı), grammar agent biçimi (`quality_index`) ölçer.
- Seçim: endeks **kapı**, sıralama değil. Kullanıcı, 4 Ekim 2026: *"İndeks aslında bir kapı yani sıralama değil geçersiz
  olmaz demek sadece"*; *"Aynen ve test et"*. Eşiğin altındaki torba elenir; geçenler yalnız context agent olasılığıyla
  sıralanır. (Önceki öneri `log P_context + λ · quality_index` düştü: yüksek endeks bilgi taşımıyor, "Nepal ." tam puan;
  orta bölgede düzgün iki yönlü cümle −0,55, eksik torba −0,00.)
- Aday sayısı mimariden ayrı: her yön bir dağılım; yönlerden sırayla en olası torbalar çekilir, gramerden geçer, N iyi
  aday bulununca durulur. N çalışma bütçesi; K (yön sayısı) kaç konu tutulabileceği.
- Döngü: seçilen torba → grammar agent dizer → cümle → context agent okur → sonraki torba.
- Birlikte eğitim: kullanıcı, 4 Ekim 2026: *"Gramer önce eğittik ve çalışıyor context ve gramer aynı veri. Bu yüzden
  context eğitilmiş grammar ile eğitilmeli gramer değişmemeli sadece okunmalı"*; *"Yaz ve test et"*. Grammar agent donuk;
  context agent'tan torbalar çekilir, gramer dizip puanlar, kapıdan geçemeyenin olasılığı aşağı itilir; geçen
  ödüllendirilmez (yalnız ceza: kısa, kesin torba ödül almaz) (torba kesikli: gradient gramerden akmaz, örnekleme sinyali). Gerçek torbanın olasılık kaybı aynen sürer.
  Gramer biçime bakar: geçerli ama gerçek olmayan devam cezalanmaz (SS'te geçerli devam kümesi gerekmez).
  Ön koşul: donuk gramer cümle olmayan torbaları ayırıyor mu. **Sınandı (4 Ekim):** ülke grameri, context agent d64 K30
  adayları, 300 sınav hikâyesi, 57.308 aday: cümle > cümle değil %86 (aynı bağlamda da %86); eşik −0,5 cümle olmayanın
  %61'ini, cümlenin %1,5'ini eler. Kör nokta: kısa torba (2 kelimelik "Nepal ." tam puan) ve dilbilgisi düzgün ama eksik /
  yanlış torba ("Turkey has Ankara as its ."). Endeks dizilişin kesinliğini ölçüyor: "X borders Y" iki yönlü olduğu
  için gerçek cümle −0,55 alıyor. Eğitimde ödül olarak kullanılırsa kısa, kesin torbaları ödüllendirme tehlikesi var.

## Açık noktalar

- **Kapı eşiği** (önceki λ'nın yerine). Kullanıcı, 4 Ekim 2026: *"o değer de not al sonra bir ölçüye bağlarız"*. Ön
  koşul sınamasında −1,0 altında gerçek cümle yoktu; ölçüye bağlanmadı. **Kapı sınaması (4 Ekim, 2.607 bağlam):** ilk
  aday cümle değil: kapısız %3,6, −1,0 %3,0 (yanlış eleme 0), −0,5 %0,8 (yanlış eleme 544 cümle adayı; "X borders Y" −0,55
  aldığı için bu olgu türü kapıdan geçemez). Kapının orta bölgesi zayıf: dizilişi iki yönlü düzgün cümle, eksik torbadan
  düşük puan alıyor.
  **`missing` düğümü ve `is_complete` (4 Ekim; kullanıcı: "missing ok"):** gramer bozuk torbalarla (drop 0,25, add 0,25)
  yeniden eğitildi, eşik yok. Dizme görülmemişte 0,995 (eski 0,998). Context agent adaylarında ilk aday cümle değil %3,6
  → %0,1, ilk 5'te %4,6 → %0,5; farklı torbada cümle olmayanın %95,3'ü elenir, cümlenin %98,9'u geçer. Yanlış elenenler
  gramerin görmediği olgular (Parthenon, nasi lemak, Himeji Castle; sınavda complete_real unseen 0,919, seen 1,000):
  kapı tanımadığı kelime bağını "eksik" sayıyor. Geçen eksikler: "The Netherlands .", "The Eiffel is in France .".
  **Tam eğitim** (kullanıcı, 4 Ekim: *"Gramer tam eğitimli olmalı o zaman"*; `--train_all 1`, 2.498 cümle): ilk aday cümle
  değil %0,1, ilk 5'te %0,4; yanlış elenen cümle adayı 422 → 31 (4 farklı torba: "The capital of the Netherlands is
  Amsterdam" türü, iki "the"li torbada dizme adları karıştırıyor, kapı o diziyi eliyor). Cümlenin %99,8'i geçer, cümle
  olmayanın %94,7'si elenir; geçenlerin çoğu veride olmayan ama düzgün cümle ("Paris is in France ."), anlam hatası
  ("People in the Czech Republic eat Brno .") ya da az sayıda eksik ("The Czech Republic ."). SS'te görülmemiş bağ
  sorunu açık.

- **Gramer kötü torba görmedi.** Kullanıcı, 4 Ekim 2026: *"bunu açık bir nokta olarak kaydet. grammer kötü torba görmedi!
  evet seçilen torbaya puan vermek grammer işi olur katılıyorum."* Gramer, kendisine gelen torbayı dizer ve bir **kalite
  endeksi** döndürür (kullanıcı: *"judge değil quality index mi oluyor grammerin fonksiyonu ?"*); torbayı gönderen ajan
  sıralama yapmaz (*"grammere gönderen ajanın grammer gibi bir görevi olmayacak"*). Gramer yalnız doğru cümlelerle eğitildi;
  kötü torbaya verdiği puanın anlamı ölçülmedi. Sınama: gerçek torba ile içine başka cümleden kelime karışmış torba,
  endeks ayrışıyor mu.
  **Durum (4 Ekim, kısmen kapandı):** `quality_index` (ardıl + öncel log-olasılığı, bağ başına). SS tam gramer, 400
  unseen: aynı cümlede gerçek > eksik %87, > karışık %82, > rastgele kelimeli %93; rastgele çiftte %68 / %79 / %75.
  Aynı bağlamın adaylarını sıralamada güçlü, mutlak eşik olarak zayıf; eksiksizliği değil dilbilgisini ölçüyor.
