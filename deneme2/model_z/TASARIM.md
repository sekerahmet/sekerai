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

## Akış: context agent + grammar agent (son sürüm, 4 Ekim)

Kullanıcı, 4 Ekim 2026: *"k bağımsız bir yapı olmalı ve grammar endeks ile baştan torbalar elenmeli"*; *"İndeks aslında
bir kapı yani sıralama değil geçersiz olmaz demek sadece"*; *"Tamam son karar eğitimden sonra. Son sürüm b"*.

- Döngü: context agent cümleyi okur → K aday torba (olasılığıyla) → grammar agent her adayı dizer ve kapıdan geçirir
  (`is_complete`) → geçemeyen ve hikâyede birebir geçmiş torba elenir → kalanların en olasısı seçilir → dizilmiş cümle
  context agent'a döner.
- İş bölümü: context agent içerik ve eksiksizlik (torba olasılığı), grammar agent biçim (kapı). Kapı sıralamaz.
- Context agent (son sürüm B): d 64, K 30, cosine, lr 3e-3, 30 epok; kayıp gerçek sonraki torbanın karışım olasılığı.
  Eğitim kapısız; kapı yalnız seçimde (`train_context --grammar`). Eğitimde kapı cezası denendi: seçimde kapıyla aynı
  sonuç (ilk aday geçerli 0,978 / 0,976), doğru torba ilk 30'da 0,881 → 0,829, adım 1,7 kat yavaş.
- Grammar agent (kapı): `missing` düğümüyle bozuk torbalarla (kelime çıkarma / ekleme) eğitilir (`train_grammar
  --missing 1`); ülkede bütün cümlelerle (`--train_all 1`, kullanıcı: *"Gramer tam eğitimli olmalı o zaman"*).
- Son sürüm, ülke sınavı (300 hikâye, seçimde kapı): ilk aday geçerli devam 0,9985, doğru torba ilk 1 / 5 / 30
  0,143 / 0,519 / 0,880; kapı cümlenin %99,8'ini geçirir, cümle olmayanın %94,7'sini eler.
- Aday sayısı mimariden ayrı (öneri, kodlanmadı): her yön bir dağılım; yönlerden sırayla torba çekilip kapıdan geçirilir,
  N iyi aday bulununca durulur.

## Meaning agent (tasarım, 4 Ekim; kodlanmadı)

Kullanıcı, 4 Ekim 2026: *"Aslında kelimeler arası anlam bağını oluştıran bir adıma ihtiyaç yok gramerden ayrı olarak
Türkiye Ankara Asia lira gibi kelimeleri yakınlaştıran"*; *"Cümleler içinde attention ile"*; *"dizmenin bozulmaması için
ayrı birşey öneriyorum"*; *"Meaning agent ok"*; *"Benim aklımdaki şuydu aslında context bir önceki cümledeki kelimelere
bakıp ona yakın kelimeleri bulabilmesi 57.000 yerine belki çok daha az. Amaç missing değil önceki cümleye bakarak doğru
torbayı oluşturması"*.

- Amaç: kelimeler arası anlam yakınlığı (Turkey ↔ Ankara, Asia, lira). Context agent önceki cümlenin kelimelerine yakın
  kelimelerden kısa bir aday listesi alır, torbayı 57.000 yerine bu listeden kurar.
- Kısa liste (kod: `meaning.build_neighbor_table`, `shortlist`; kullanıcı: *"bir cümledeki tüm kelimeler için ... 5 yakın
  kelime ... bir liste üretmek bir tahmin yok"*; *"the of is gibi kelimeleri elemek çok kolay ... sıklık tablosu ... elle
  vermeye gerek yok"*; *"south in the a gibi değil"*): 2 cümlelik pencerede birlikte geçme sayımı; kelimenin komşusu
  **güçlü bağ**: kat P(j|i)/P(j) ≥ 3 ve i'nin pencerelerinin ≥ %5'inde birlikte; birlikte geçme payına göre ilk N. Biçim
  kelimelerinin güçlü bağı olmaz, komşu getirmez (elle liste yok). Geçiş listesi = okunan cümlenin bütün kelimelerinin
  komşuları (derinlik 1) + sık kelimeler + hikâyede geçenler.
- **Elle konan ölçüler** (kullanıcı: *"elle verdiğin ölçüleri not al"*; hiçbiri ölçülmedi): pencere 2 cümle (`WINDOW`),
  kelime başına N = 10 (`NEIGHBORS`, kullanıcı: *"n=10"*), kat ≥ 3 (`STRONG_LIFT`), birlikte ≥ %5 (`STRONG_SHARE`), sık
  kelime: cümlelerin > %2'si (`FREQUENT_SHARE`), sayımda en az 3 birlikte geçme (senaryo betiği).
- Gözle (ülke, sayım): Turkey → Turkish, Ankara, Black, Iran…; Peru'da kat sıralaması America'yı 25. sıraya atıyordu
  (seyrek kelimeler Sucre, Tiwanaku öne geçiyor); pay sıralaması + kat eşiği düzeltti. 7 komşuda 2. adım Sophia →
  Ukraine kayması (ortak kelime köprüsü).
- Gizli kelimeli (BERT türü) `MeaningAgent` denendi: bağlam içinde gizli kelimeyi doğru buluyor (pencere 2: 167/170) ama
  kelime başına komşu tablosu vermiyor (E benzerliği: yerine geçen kelimeler; Q·K: gürültü). Kısa liste sayımla.
- Ülke, context agent senaryosu (N 10, derinlik 1): liste ort 60 kelime / 897, sınavda sonraki cümlenin kelimeleri
  listede 0,976.

## Açık noktalar

- **Kapı SS'te: görülmemiş bağ.** Kullanıcı, 4 Ekim 2026: *"şu an indeks hazır değil mi? notunu al ona bakarız"*. Kısmi
  eğitimli ülke grameri görülmemiş gerçek cümlelerin %8'ini eledi (unseen complete_real 0,919): tanımadığı kelime bağını
  "eksik" sayıyor. Ülkede tam eğitimle kalktı; SS'te görülmemiş bağ hep olacak, orada yeniden bakılacak.
- **Kapıdan geçen eksikler** (az): "The Czech Republic .", "The Eiffel is in France .". Anlam hatası ("People in the Czech
  Republic eat Brno .") kapının işi değil.
- **Eğitimde kapı SS'te.** Ülkede katkısı olmadı; SS'te context agent çok daha fazla kötü aday üretirse yeniden sorulabilir.
