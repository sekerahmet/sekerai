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

## Meaning agent (4 Ekim)

Kullanıcı, 4 Ekim 2026: *"Aslında kelimeler arası anlam bağını oluştıran bir adıma ihtiyaç yok gramerden ayrı olarak
Türkiye Ankara Asia lira gibi kelimeleri yakınlaştıran"*; *"Cümleler içinde attention ile"*; *"dizmenin bozulmaması için
ayrı birşey öneriyorum"*; *"Meaning agent ok"*; *"benim 2 ve ya 3 cümle dediğim cümlenin tüm kelimeleri ortak. Yoksa cümle
tahmini değil"*; *"1 cümlenin tüm kelimeleri sırayla gizlenmezse model nasıl öğrenecek ?"*.

- Amaç: context agent 57.000 kelime yerine okuduğu cümlenin kelimelerine yakın kelimelerden kısa bir liste üzerinde
  çalışsın (kullanıcı: *"57.000 yerine belki çok daha az"*).
- Mimari (kullanıcı: *"Ya da istediğimiz tabloyu kuran bir mimari tasarla"*; *"saatlerdir bu tabloyu öğrenen modelle
  kurmanı istedim"*): bağ tablosu modelin kendi parametresi, R[i, j] = source_i · target_j / √d. Gizli kelime yalnız
  tablodan tahmin edilir: puan(j) = bias_j + Σ_i α_i R[i, j]; α attention (gizli yer `mask` sorgusu, `key`), bias genel
  sıklık. Kelime başına iki vektör (`source`, `target`).
- Öğrenme: her cümlede biten büyüyen pencere (1, 1–2, 1–3 … en çok 5 cümle; kullanıcı: *"sıralı ilk cümle sonra ilk
  cümle ve ikinci cümle sonra ilk üç cümle gibi"*), penceredeki bütün kelimeler tek torba; her kelime sırayla gizlenir.
  Pencere 1 cümle içi bağ, fazlası cümleler arası bağ.
- Çıktı: `neighbors.pt` (`build_neighbor_table`): R'nin her satırının en büyük 50 değeri.
- Kullanım `shortlist`: okunan cümlenin her kelimesi için tablodan NEIGHBORS komşu, DEPTH adım; komşu × derinlik eğitimde
  maliyet kararı, üretimde sıcaklık gibi ayar. Context agent `shortlist` ile çalışabiliyor (`context.py`).
- Doğruluk ölçüsü (eğitimden önce yazıldı; kullanıcı: *"Eğer doğru değilse biz yanlış birşey tasarlamışızdır demek"*):
  (1) tablo gözle: Turkey → Ankara, Istanbul, Turkish, lira, baklava, Hagia Sophia; Peru → Lima, Cusco, sol, ceviche;
  biçim kelimelerinin güçlü bağı yok; (2) gizli kelime bulunuyor; (3) Ankara gizliyken en çok Turkey ve capital oy
  veriyor; (4) bu tablonun kısa listesiyle context agent kısıtsız B'ye yakın.
- Önceki deneme (attention katmanlı gizli kelime modeli): gizli kelimeyi bağlam içinde buluyordu (pencere 5: 269/273) ama
  tablo çıkarılamadı (kelime temsili benzerliği, Q·K, boş mask, etki okuması); silindi.

## Açık noktalar

- **Kapı SS'te: görülmemiş bağ.** Kullanıcı, 4 Ekim 2026: *"şu an indeks hazır değil mi? notunu al ona bakarız"*. Kısmi
  eğitimli ülke grameri görülmemiş gerçek cümlelerin %8'ini eledi (unseen complete_real 0,919): tanımadığı kelime bağını
  "eksik" sayıyor. Ülkede tam eğitimle kalktı; SS'te görülmemiş bağ hep olacak, orada yeniden bakılacak.
- **Kapıdan geçen eksikler** (az): "The Czech Republic .", "The Eiffel is in France .". Anlam hatası ("People in the Czech
  Republic eat Brno .") kapının işi değil.
- **Eğitimde kapı SS'te.** Ülkede katkısı olmadı; SS'te context agent çok daha fazla kötü aday üretirse yeniden sorulabilir.
