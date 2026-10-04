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
  kurmanı istedim"*; *"Kur"*): kelime başına iki vektör, `source` (oy veren, word2vec IN) ve `target` (oy alan, OUT); her
  görünen kelime gizli kelimeyi tek başına tahmin eder, P(j | i) = softmax_j(bias_j + source_i · target_j / √d); gizli
  kelimenin olasılığı bu tahminlerin ortalaması. bias genel sıklık. d 128.
- Öğrenme: her cümlede biten büyüyen pencere (1, 1–2, 1–3 … en çok 5 cümle; kullanıcı: *"sıralı ilk cümle sonra ilk
  cümle ve ikinci cümle sonra ilk üç cümle gibi"*), penceredeki bütün kelimeler tek torba; her kelime sırayla gizlenir.
  Sık kelime seyreltmesi (word2vec, t = 0,001; kullanıcı: *"Evet"*): kelime √(t/f) + t/f olasılıkla kalır (. 0,09, the
  0,19, Turkey 1).
- Çıktı: `neighbors.pt` (`build_neighbor_table`): her kelime için kos(source_i, target_j) en büyük 50 kelime. IN-OUT
  kosinüsü birlikte gelen kelimeleri verir (Mitra ve ark. 2016, `belge/makaleler/2016/mitra2016_desm.txt`).
- Kullanım `shortlist`: okunan cümlenin her kelimesi için tablodan NEIGHBORS komşu, DEPTH adım; komşu × derinlik eğitimde
  maliyet kararı, üretimde sıcaklık gibi ayar (`train_context --meaning neighbors.pt --shortlist N --depth D`). Liste =
  okunan cümlenin kelimelerinin komşuları + hikâyede geçenler; biçim kelimeleri cümlenin kendi biçim kelimeleri ve
  onların komşuları üzerinden girer. Ülke sınavı kapsam (sonraki cümlenin tamamı listede): 10 × 1 0,301, 10 × 5 0,958,
  30 × 2 0,995 (138 kelime / 897); derinlik bu tabloda belirleyici.
- Context agent + meaning 30 × 2 (CPU, epok ~7 sn), seçimde kapı yok: doğru torba ilk 1 / 5 / 30 0,133 / 0,490 / 0,826,
  ilk aday geçerli 0,942 (kısıtsız B: 0,139 / 0,509 / 0,881, 0,964).
- Ölçü gözle (kullanıcı: *"Sınav yok bunda göz ile kontrol var"*; *"Eğer doğru değilse biz yanlış birşey
  tasarlamışızdır"*). Ülke, son sürüm (`meaning_table_countries_w5_d128_mix_sub_e4`, CPU 39 sn): Japan → Tokyo, Shinano,
  Japanese, Osaka, sushi, yen, Himeji; Peru → Chile, sol, ceviche, Cusco, Lima, Spanish, America, Amazon, Pacific;
  Ankara → Turkey, Greece, baklava, Turkish, Hagia, Euphrates, Istanbul, Black, lira; Turkey → Euphrates, borders,
  Istanbul, Hagia, is, Turkish, Greece, Ankara, … (arada kalıp kelimeleri); biçim kelimeleri yalnız birbirine bağlı.
- Denenip bırakılan: attention katmanlı gizli kelime modeli (tablo okunamadı), attention'lı oy (oylar kalıp kelimelerine
  gitti), toplamsal oy (kanıt bütün Türkiye kelimelerine bölüştü).

## Açık noktalar

- **Kapı SS'te: görülmemiş bağ.** Kullanıcı, 4 Ekim 2026: *"şu an indeks hazır değil mi? notunu al ona bakarız"*. Kısmi
  eğitimli ülke grameri görülmemiş gerçek cümlelerin %8'ini eledi (unseen complete_real 0,919): tanımadığı kelime bağını
  "eksik" sayıyor. Ülkede tam eğitimle kalktı; SS'te görülmemiş bağ hep olacak, orada yeniden bakılacak.
- **Kapıdan geçen eksikler** (az): "The Czech Republic .", "The Eiffel is in France .". Anlam hatası ("People in the Czech
  Republic eat Brno .") kapının işi değil.
- **Eğitimde kapı SS'te.** Ülkede katkısı olmadı; SS'te context agent çok daha fazla kötü aday üretirse yeniden sorulabilir.
