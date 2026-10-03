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

Durum: veri hazır, gramer ajanının kodu yok. Kol Model Y'den kopyalanmadı; temel çalışması (dilin, gramerin, anlamın matematiksel
ifadesi; bilginin nasıl saklandığı; yön / açı / uzunluk / kütle) bitince adım adım kurulur (CLAUDE.md kural 13).
