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

## Meaning agent (4 Ekim)

Kullanıcı, 4 Ekim 2026: *"Aslında kelimeler arası anlam bağını oluştıran bir adıma ihtiyaç yok gramerden ayrı olarak
Türkiye Ankara Asia lira gibi kelimeleri yakınlaştıran"*; *"Cümleler içinde attention ile"*; *"dizmenin bozulmaması için
ayrı birşey öneriyorum"*; *"Meaning agent ok"*; *"benim 2 ve ya 3 cümle dediğim cümlenin tüm kelimeleri ortak. Yoksa cümle
tahmini değil"*; *"1 cümlenin tüm kelimeleri sırayla gizlenmezse model nasıl öğrenecek ?"*.

- Amaç: kelimeleri anlamca yakınlaştıran tablo. İlk kullanım context agent'ın kısa listesiydi (arşivde);
  şimdi meaning vektörü z'nin kelime vektörünün yarısı (`core/sentence/sentence_z.py`).
- Mimari (kullanıcı: *"Ya da istediğimiz tabloyu kuran bir mimari tasarla"*; *"saatlerdir bu tabloyu öğrenen modelle
  kurmanı istedim"*; *"Kur"*): kelime başına iki vektör, `source` (oy veren, word2vec IN) ve `target` (oy alan, OUT); her
  görünen kelime gizli kelimeyi tek başına tahmin eder, P(j | i) = softmax_j(bias_j + source_i · target_j / √d); gizli
  kelimenin olasılığı bu tahminlerin ortalaması. bias genel sıklık. d 128.
- Öğrenme: her cümlede biten büyüyen pencere (1, 1–2, 1–3 … en çok 5 cümle; kullanıcı: *"sıralı ilk cümle sonra ilk
  cümle ve ikinci cümle sonra ilk üç cümle gibi"*), penceredeki bütün kelimeler tek torba; her kelime sırayla gizlenir.
  Sık kelime seyreltmesi (word2vec, t = 0,001; kullanıcı: *"Evet"*): kelime √(t/f) + t/f olasılıkla kalır (. 0,09, the
  0,19, Turkey 1).
- Çıktı: `neighbors.pt` (`build_neighbor_table`): her kelime için modelin kendi log P(j | i) en büyük 50 kelime
  (kullanıcı, 4 Ekim: *"log P bana daha doğru gibi geldi"*). Önce kosinüs vardı (Mitra ve ark. 2016); SS'te vektör boyu
  sıklığı kodluyor, kosinüs onu atınca komşular nadir adlara kaydı (dragon → Flamewing, Firewing; log P: scales, knight,
  cave, fierce, roared). Matematikçi ölçüsü (`belge/model_z_temel/12` Ek A), SS sınavı 5 × 5: liste 3.676 → 820 kelime,
  sonraki cümlenin tamamı listede 0,062 → 0,207. SS: boy → He, his, he, him; girl → She, her, she; Mia → She, her, she.
  Nadir adlar zayıf (Tom 71 kez geçiyor: little, big, Mia …; Tim 372: Rex 1,000).
- Ölçü gözle (kullanıcı: *"Sınav yok bunda göz ile kontrol var"*; *"Eğer doğru değilse biz yanlış birşey
  tasarlamışızdır"*). Ülke, son sürüm (`meaning_table_countries_w5_d128_mix_sub_e4`, CPU 39 sn): Japan → Tokyo, Shinano,
  Japanese, Osaka, sushi, yen, Himeji; Peru → Chile, sol, ceviche, Cusco, Lima, Spanish, America, Amazon, Pacific;
  Ankara → Turkey, Greece, baklava, Turkish, Hagia, Euphrates, Istanbul, Black, lira; Turkey → Euphrates, borders,
  Istanbul, Hagia, is, Turkish, Greece, Ankara, … (arada kalıp kelimeleri); biçim kelimeleri yalnız birbirine bağlı.
- Denenip bırakılan: attention katmanlı gizli kelime modeli (tablo okunamadı), attention'lı oy (oylar kalıp kelimelerine
  gitti), toplamsal oy (kanıt bütün Türkiye kelimelerine bölüştü).

## SentenceTransformer (5 Ekim)

Kullanıcı, 5 Ekim 2026: *"Herşeyi unut model z baştan tasarlıyoruz gibi düşün . Mantık şu cümleyi vektöre çevir.
Vektörler arası attention yapıp bir sonraki cümle değilde bir sonraki kelimeyi tahmin et. Cümle bitince onu vektör yap"*;
*"matematikçinin formülü + meaning + grammer bilgileri = Z de"*; *"Sentence transformer çok dikkatli kur en iyi
transformer modeli gibi . Safece 1 cümle sonrasına bakacak şekilde"*.

- z (`sentence_z.py`, eğitim yok): z = Σ_t R_t f(w_t) + R_n f(END) + Σ_t g_t ⊙ f(w_t).  f(w) = birim([meaning(w) ;
  rastgele kimlik]); R_t konum anahtarı (kaydırma × işaret, tersi R_tᵀ); g_t = işaret(P h_t), h_t grammar torba
  okuyucusu (kelimenin yuvası). Geri açma u_t = R_tᵀ z, en emin konum önce (SIC). Ülke, z 512: görülmemiş 356 cümlede
  birebir 0,969. Aile: TPR (Smolensky 1990), HRR (Plate 1995); matematikçinin önerisi.
- Model (`sentence.py`): [BOS][z_1..z_{k-1}] w_1..w_t → w_{t+1} (END ile biter); pre-norm RMSNorm, RoPE, QK-norm,
  SwiGLU, bias yok, tied embedding; AdamW 0,1, warmup + cosine, clip 1,0. Örnek = tek cümle, kısa dizi, düz causal.
- Adım adım (CLAUDE.md kural 13): (1) z'den ülke doğrusal okunuyor 0,989; (2) yalnız son z ile cümle içi ülke adı 0,988,
  karışık z 0,03; (3) bütün z'ler, d 64, batch 160 cümle, 15 epok: görülmemiş 300 hikâyede sonraki cümle geçerli 0,984,
  doğru olgu 0,997, tekrar 0,013, döngü 0. İlk 4 epok / 728 adım yetmedi (geçerli 0,386; adım değil örnek sayısı).
- Windows: EcoQoS kapatılmazsa ~10 adımdan sonra adım 55 → ~550 ms (`train_grammar._no_power_throttling`).
- Arşiv: context agent, öğrenilen codec ve `generate.py` `arsiv/model_z_20261005/` (5 Ekim, kullanıcı: *"context i ve
  onunla ilgili şeyleri arşive al. Sadece meaning gramer transformer ve onunla ilgili şeyler kalsın"*); git geçmişinde.
