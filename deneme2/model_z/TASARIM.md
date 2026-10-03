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

## Açık noktalar

- **Gramer kötü torba görmedi.** Kullanıcı, 4 Ekim 2026: *"bunu açık bir nokta olarak kaydet. grammer kötü torba görmedi!
  evet seçilen torbaya puan vermek grammer işi olur katılıyorum."* Gramer, kendisine gelen torbayı dizer ve bir **kalite
  endeksi** döndürür (kullanıcı: *"judge değil quality index mi oluyor grammerin fonksiyonu ?"*); torbayı gönderen ajan
  sıralama yapmaz (*"grammere gönderen ajanın grammer gibi bir görevi olmayacak"*). Gramer yalnız doğru cümlelerle eğitildi;
  kötü torbaya verdiği puanın anlamı ölçülmedi. Sınama: gerçek torba ile içine başka cümleden kelime karışmış torba,
  endeks ayrışıyor mu.
