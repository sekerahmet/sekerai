# Strata-x gözden geçirme 1 — derin nefes (11 Ekim 2026)

Kullanıcı, 11 Ekim: *"Bazen durup sakince tasarımı gözden geçirmek de faydalı olabiliyor ... Her bir bölümden ne bekliyorum neyi
hedefliyorum neye ihtiyacı var doğru olanı alıyor mu ? Aldığını doğru işliyor mu? Kapasitesi yeterli mi ? Tasarım teorik olarak
baştan sona akışı destekliyor mu? İçeri de verimsiz yerler var mı ? ... Bir plan yapıp bunları değerlendirir danışır matematiksel
olarak ispatlar ve son tasarım fikrimi daha sadeleştiririm"*; *"Ayrıca bir örnek üzerinden düşünme deneyi de kesin yapardım"*.

Ölçüm yeri: B3 (Writer / Reader 512, Thinker d512 / 8, θ 4, SS 2.000 adım, 1,5579) ve aynı ayarda transformer (1,4886);
sınav alt kümesinin ilk 6 satırı, 12.156 hedef. Araçlar: `sx_trace.py` (örnek), `sx_voice_read.py`, `sx_confidence.py`.

## 1. Örnek (hikâye 27, `sx_trace.py`)

"It was an icy morning when a boy set out to sea. He had a secret plan ... the lost city of Atlantis ..."
Birim sınırları tasarlandığı gibi sürprizli token'larda: *icy | morning | set | to | sea | ... | Atlantis* (birim ~2,7 token).
En büyük kayıplar (Strata − transformer, nat) ve Thinker kapatılınca değişim:

| yer | açık | Thinker kapalı | tür |
|---|---|---|---|
| icy → **morning** | +2,35 | −0,19 | yerel eşdizim |
| the ocean → **around** him | +2,42 | +1,27 | yerel kalıp |
| the only one looking → **.** | +2,28 | −0,58 | yerel sözdizim |
| I will not be → **scared** | +2,00 | +0,10 | pencere içi hikâye bilgisi (fear) |
| city of → **Atlantis** | +1,69 | +1,88 | dünya bilgisi (Thinker taşıyor ama yetmiyor) |
| With a → **deep** breath | +1,38 | +0,23 | yerel eşdizim |

Örneğin söylediği (HİPOTEZ): büyük kayıpların çoğu Thinker'a bağlı değil — transformer'ın bildiği **kalıp / eşdizim** bilgisi.

## 2. Toplu ölçüm (ÖLÇÜLDÜ)

- **Kalibrasyon** (`sx_confidence.py`): Strata aşırı güvenli değil, daha az güvenli: entropi 1,644 / tf 1,572; yanlışta top-p 0,399 /
  0,409; kalibrasyon eğrisi aynı. → Strata dürüst ama **daha az bilgili**. Açık en çok H_tf 0,5–2 (kalıp) token'larında: entropi
  1,379 / 1,255, nll +0,070 (token'ların %41'i). ~~Thinker güvenle yanlış yöne itiyor~~ (ilk okumada açığın %76'sı "Thinker zararlı"
  token'larda göründü; kalibrasyon bunu çürüttü — seçim yanılgısı: yanlış token'da her bilgi çıkarma kaybı düşürür).
- **Thinker işe yarıyor:** kapatınca +0,126 (bütün türlerde). 2× büyütünce (d1024 / 12) yalnız −0,02 / −0,04 → darboğaz değil.
- **Kanal (Writer 512'de) darboğaz değil:** Voice (çalışan kapı, 1a9efe7) 1,5593 = B3 1,5579; kapı ortalama 0,03, türe göre seçici değil;
  Voice'u kapatınca +0,036 (en çok tekrar token'larında +0,073) → kullanılıyor ama ekleme değil ikame. Pencere 512 etkisiz (1,5549).
- **Writer kapasitesi darboğaz:** genişlik 256 → 512 −0,114 (Thinker 2× büyükken de −0,100); derinlik 2+2 → 4+4 −0,045;
  token / çıkış 256 → 512 (Writer 256) −0,005.

## 3. Parça parça (kullanıcının soruları)

| parça | ne bekliyorum | doğru girdi? | doğru işliyor? | kapasite | not |
|---|---|---|---|---|---|
| Reader (2 katman) | sürpriz, birim vektörü, Writer tabanı | evet (token'lar) | sınırlar sürprizli token'da (örnek) | üç iş tek parçada | kendi çıkış başı adımın ~%20'si |
| birim kuralı | eşit bilgi parçaları | Reader sürprizi | evet | — | 2,1 token / birim: sıkıştırma zayıf |
| Thinker (8, birimlerde) | hikâye durumu, yön | up(Reader son token) | evet (+0,126) | fazla (2× ≈ etkisiz) | hesabın ~yarısı |
| Thinker → Writer | durumu iletmek | tek vektör, girişte | Writer 512'de yeterli | — | Voice eklemedi |
| Writer (2 katman) | kalıp + yön → token | he + ctx | evet | **yetersiz** | açık burada |

## 4. Akış (tümden gelim) ve HESAP

x_{t+1}'i tahmin etmek iki tür bilgi ister: **(a) kalıp bilgisi** (sözcük, eşdizim, sözdizim: her token'da, ezber) ve **(b) hikâye
durumu** (yavaş değişir: kim, ne, nereye). Strata (b)'yi birimlere taşıdı — doğru (Thinker işe yarıyor, büyümesi gerekmiyor). Ama (a)
yerel kısma kaldı ve (a) **ezber kapasitesi** ister. Token başına dokunulan FFN parametresi (HESAP, d512, hidden 1.368, SwiGLU):
transformer 8 × 2,1M = 16,8M; Strata 4 × 2,1M = 8,4M (+ Thinker'ın birimde 8 FFN'i, tek vektörle iletilen). Toplam parametre Strata'da
fazla (~37,6M / 25M), ama **her token'ın erişebildiği ezber yarı yarıya**. Writer genişleyince açık kapanıyor; Thinker büyüyünce
kapanmıyor: ikisi de bununla tutarlı. Kullanıcının ölçütüyle: *"Kapasite yapacağı iş ile ölçülür"* — kalıp token'ları yalnız yerel iş
ister ve onlar düşüyor → **kapasite**.

## 5. Verimsiz yerler

1. Thinker gereğinden büyük ve birimler kısa (2,1 token): hesabın yarısı darboğaz olmayan yerde.
2. Reader'ın ayrı çıkış başı (~%20 adım) yalnız sınır + yardımcı kayıp için.
3. Kalıp bilgisi yoğun (dense) hesapla aranıyor: ezber bir arama işi, hesap işi değil.
4. Kodda kazanmayan seçenekler: plan (1,566), next, rate (SS 1,579), memory (ertelendi), Voice (ikame) → silinir / kapalı.

## 6. ~~Sadeleşmiş fikir (HİPOTEZ) — "hesap karara, arama ezbere"~~ (çürüdü, bkz. 9)

Strata'nın ilkesi (hesabı bilgiye göre harcamak) bir adım daha: üç iş, üç maliyet —
**yön kararı** → Thinker (az birimde, derin; küçülebilir); **kalıp ezberi** → Writer'da **seyrek bellek** (çok parametre, token başına
az etkin: MoE FFN ya da arama tablosu); **sözdizim** → küçük yoğun yerel katman. Transformer her token'da üçünü de yoğun hesapla yapıyor.

## 7. ~~Ölçüm planı~~ (M1 ayırt edemez, M2 verim sorusu; yerine 9)

- **M1 ezber mi dikkat mi:** Writer 256, yalnız FFN hidden 2× (attention 256 kalır). 256 → 512 kazancının (0,114) çoğunu alırsa
  darboğaz ezber kapasitesi → seyrek bellek yolu. Almazsa genişlik / dikkat → başka yol.
- **M2 Thinker fazla mı:** B3, Thinker 8 → 4 katman. Kayıp ≤ +0,01 ise hesap Thinker'dan Writer'a aktarılır.
- (Bekleyen) Voice + Writer 256 (düzeltilmiş kapı): eşik ≤ 1,615 kanal / ≥ ~1,65 kapasite.

## 8. Danışmana

Ana iddia: "açık Writer'ın ezber kapasitesinde, kanal ve Thinker değil". Çürütme ve M1 / M2 düzeneğine itiraz istenir.

## 9. Danışman antitezi ve düzeltme (ÖLÇÜLDÜ, `scratchpad/danisman/sx_gain_split.py`, 24 satır, 48.150 hedef)

A (Writer 256) → B3 (512) kazancı 0,115, token'lar Thinker kapatma Δ'sına göre (yanlılığa karşı hem A hem B3 ile sınıflanmış):
Thinker'a duyarsız token'lar (|Δ| < 0,05) kazancın yalnız %4–15'ini alıyor ve orada B3 = transformer (+0,005 / +0,008). Kazanç
Thinker'ın işin içinde olduğu token'larda. H_tf 0,5–2 kovası "kalıp" değil: orada Thinker kapatma +0,112 (genel +0,118 kadar).
Benim 6 satırlık tablomda da Thinker'sız token'ların açığı +0,015 / payı %8'di — veriyi okudum, entropi etiketine güvendim.
**Düzeltilmiş hüküm (ÖLÇÜLDÜ + HİPOTEZ):** açık ezberde de kanalda da değil, Writer'ın Thinker bilgisini yerel bağlamla **birleştirme
kapasitesinde** (kullanıcının çekincesi: *yeni içerik ikisini birden ister*). §4'teki "sürpriz kalan bilgiyi ölçer, uygulama maliyetini
ölçmez" tespiti kavramsal olarak doğru ama bu açığı açıklamıyor (yerel token'lar zaten eşit).

Ayırıcı test (danışman; kod yok): A ve B3 Thinker'sız eğitilir (`--sx_theta 1e9`: birim kapanmaz, ctx = g0). Thinker'sız 256 → 512
kazancı ≈ 0,11 kalırsa ezber; belirgin küçülürse (< 0,05) birleştirme. Tasarım adayı (danışman, N4 iletişim ilkesi): Thinker bilgisi
her Writer katmanına kendi izdüşümüyle (katman başına down_l); HESAP: token başı katman başı ~0,1M çarpma (~%0,4).
Önceden tahmin: ben Thinker'sız kazanç ~0,05 (birleştirme), danışman — (sorulmadı).
