# 02 — Strata tezine en güçlü itiraz (11 Ekim 2026)

Soru (ana oturum, kullanıcının merakı üzerine): Strata'nın bütün tezine (N1, §0) en güçlü itiraz ne; hangi varsayım en zayıf;
ölçekte nerede kırılır; en ucuz çürütücü ölçüm; itirazlara rağmen ayakta kalan ne.
Dayanak: TASARIM N1–N4, OLCULENLER 1071–1075, danışman 01 ve sonraki CPU ölçümleri (`scratchpad/danisman/sx_prolong.py`,
`sx_copy.py`, `sx_entropy.py`, `sx_argmax.py`; B = `v2_s_sxab_B_l512_t4_ss_s2000`), literatür metinleri: `2024/pagnoni2024_blt.txt`,
`2025/hwang2025_hnet.txt`, `2023/yu2023_megabyte.txt`, `2021/nawrot2021_hourglass.txt`.

## 1. Tezin varsayımları

| # | varsayım | durum |
|---|---|---|
| V1 | Bilgi token'lara eşit dağılmıyor | ÖLÇÜLDÜ, dayanıyor: Gini 0,51; bilginin %50'si token'ların %19'unda |
| V2 | Büyük çözücünün tazelenme ihtiyacı, gelen bilgiyle (gerçekleşen sürpriz) orantılı | kısmen ÖLÇÜLDÜ, dayanıyor: bir birim bayat g yalnız +0,034, yuvadan bağımsız |
| V3 | Küçük kodlayıcının sürprizi, büyük modelin sürprizinin yeterli vekili | ölçülmedi (HAYAL R-A); FineWeb'de birim 1,8 token'a düştü |
| V4 | Yerel kısım (kodlayıcı + kod çözücü + sürpriz başı) küçük kalabilir; hesabın çoğu büyük çözücüde | **en zayıf**; SS ölçümleri aleyhte (§2) |
| V5 | Kod çözücü büyük çözücüyü birim başına tek vektörden görerek token seçebilir | aleyhte iz: açığın %65'i ilk geçişte, yüksek entropide; uzak kopya +0,32 |
| V6 | Kazanç ölçekte FLOP / bant genişliği tasarrufuna dönüşür | eğitimde HESAP olarak olası; batch'li üretimde kırılgan (§3) |

## 2. En güçlü itiraz: hesap oranı ters döndü (V4)

Tez "büyük çözücü pahalı, yerel ucuz" der. SS'te kalite ancak yerel kısım token başına büyük çözücü kadar hesap yapınca
toparlandı (HESAP, `sx_params.py`; büyük d512 / 8 katman = 25,2M parametre, birim / token 0,37–0,38):

| kol | token başı yerel çarpma | token başı büyük çözücü (25,2M × 0,37) | çıkış başları | açık (2.000 adım) |
|---|---|---|---|---|
| A yerel 256 | 3,7M | 9,3M | ~2,1M (yardımcı + ana) | +0,18 |
| B yerel 512 | 13,7M | 9,3M | ~4,2M | +0,07 |
| transformer | — | 25,2M + çıkış | 1 baş | 0 |

- B'nin token başı hesabı (≈ 27M) transformer'ınkiyle (≈ 26M) aynı; açık hâlâ +0,07. SS'te Strata henüz hiç FLOP kazanmadan
  kaybediyor. Profil de aynı yönde: B'de adımın ~%72'si yerel kısım + sürpriz başı + çıkış (17,5 + 21,3 + 18,1 + 18,5 / 105,6 ms).
- Mekanizma (HİPOTEZ, ölçümlerle uyumlu): açık, yeni içeriğin seçildiği token'larda. Bu token'lar (iii) ilk geçiş, H_tf 2–4.
  Açığın %65'i burada. B'nin argmax'ı burada 1 epok transformer'ın ilk 5'inde %86 (tf2000 %93). Örnekler: "park" → "homes",
  "hero" → "difference". Token seçimi küçük kod çözücüde yapılıyor, büyük çözücünün bilgisi ona tek bir d_local vektörle
  geliyor. Seçim ne kadar bilgi istiyorsa yerel kısım o kadar büyümek zorunda; o da tasarrufu yiyor.
- Ölçekte kırılma noktası: FineWeb'de "dünya bilgisi" token'ları (isim, sayı, terim) SS'tekinden çok daha fazla ve çok daha
  yüksek entropili. Bilgi büyük çözücünün FFN'lerinde; onu token'a çevirecek kanal yine d_local'lik tek vektör + küçük kod
  çözücü. V4'ün ölçekte tutması için gereken yerel boy, büyük çözücüden yavaş büyümeli.

**Literatürle karşılaştırma.** BLT bu varsayımı açıkça kuruyor ve ölçekte destek bildiriyor: *"When growing total parameters
20x from 400M to 8B, we only roughly double BLT's local model parameters"* (pagnoni2024_blt, s. 13).

Ama aynı metinde iki kısıt var:
- **Hesap-optimal rejimde eşitlik, BPE token boyunda.** Hesap-optimal rejimdeki eşitlik ~4,5 baytlık yamalarla geliyor. BLT,
  Llama 3 tokenizer'ının ortalama token boyunu 4,4 bayt olarak veriyor. Yani BPE token'ı başına yaklaşık bir yama: token'a
  göre sıkıştırma yok.
- **Büyük yama küçük ölçekte geride başlıyor.** 6–8 baytlık yamalar için metin şunu yazıyor: *"BLT with patch size of 8
  starts at a significantly worse point compared to bpe Llama 2 at 1B but ends up better than bpe at 7B scale"*. Kazanç
  noktası hesap-optimal bütçenin 2,5–3 katında.

Strata birimi ≈ 2,7 GPT-2 token'ı, yani yaklaşık 11 bayt (HESAP; token başına ~4 bayt TAHMİN). Bu, BLT'nin denediği en büyük
yamadan (8 bayt) büyük. Bu yüzden 60M'lik SS modelinde açık beklenen sonuç. Ama SS sonucu da ölçekte ne olacağını
söylemiyor: ne aleyhte ne lehte kanıt.

Ayrıca BLT'nin kod çözücüsü yamalara cross-attention yapıyor (s. 6, lD alternatif cross-attention + transformer katmanı).
Hourglass'ta da attention ile yukarı örnekleme en iyi sonucu veriyor: *"Attention upsampling works well regardless of the
function U and has the lowest perplexity"*. Strata'nın toplamalı tek vektörü bu ailede en kaba seçenek. Ama danışman 01 /
tur 2 ölçümü, son birimlere daha ince bakmanın SS'te açığı kapatmayacağını gösterdi (yuva profili düz). Kanal sorunu
"hangi birime bakıldığı" değil, "ne kadar genişlikle bakıldığı" (V5).

## 3. Ölçekte diğer kırılma yerleri (mekanizmayla)

- **Batch'li üretim hızı (V6).**
  - Küçük batch'te üretim adımı bellek bant genişliğiyle sınırlı. Maliyet ≈ o adımda okunan ağırlıklar.
  - B dizide, her dizi her adımda ~0,37 olasılıkla birim kapatıyor. En az birinin kapatma olasılığı 1 − 0,63^B: B = 8'de %97.
  - Yani büyük çözücünün ağırlıkları hemen her adım okunur. FLOP düşer ama bant genişliği maliyeti düşmez (HESAP). Kazanç
    ancak büyük batch'te (hesapla sınırlı rejim) ya da uzun bağlamda (KV okuma) gelir.
  - Düzensiz alt batch'ler `dynamic=False` derlemeyle de çatışıyor: her kova yeni bir şekil (danışman 01 D2).
- **Sınır kalitesi (V3).**
  - FineWeb'de küçük kodlayıcının sürprizi yüksek, birimler küçülüyor (θ 8'de 1,8 token, ÖLÇÜLDÜ). Oran θ'sı sıkıştırmayı
    sabitler ama "eşit bilgi" artık göreli olur.
  - Sınır kararından gradyan geçmiyor. Kodlayıcı yalnız kendi CE'siyle eğitiliyor; sınırı büyük çözücüye yararlı olacak
    yere koymayı öğrenmiyor.
  - H-Net bu soruna kendi yönlendirme modülüyle uçtan uca çözüm arıyor (*"jointly trainable boundary predictors are the
    ideal solution"*, hwang2025_hnet, s. 3). BLT ayrı ve donuk bir 100M entropi modeli kullanıyor. Strata ikisinin arasında.
  - BLT'de yama, bir sonraki baytın entropisi eşiği geçtiği yerde başlıyor: tahmin öncesi belirsizlik. Strata'da birim,
    gerçekleşen sürprizden sonra kapanıyor: bilgi geldikten sonra. Hangisinin doğru olduğu açık soru; Strata'nınki
    "güncelle" mantığına, BLT'ninki "zor tahmine hesap" mantığına uyuyor.
- **Hikâye belleği.** Pencere dışındaki birebir kopya (ii) token başına +0,32 kötü. Uzun bağlamda kurulan isim ve sayıların
  payı büyür; büyük çözücü bunları birim özetinden taşımak zorunda. SS'te bu açığın yalnız ~%5'i; uzun belgede payı
  ölçülmeli.

## 4. Tezi en ucuz yoldan çürütecek tek ölçüm

**V4'ün ölçek testi: yerel boyun açığa etkisi, büyük çözücü büyüdükçe küçülüyor mu?**

- SS'te iki kısa koşu: büyük çözücü d1024 / 12 katman, yerel 256 ve yerel 512, θ 4, 2.000 adım. Kod değişikliği yok;
  `--d 1024 --layers 12 --sx_local 256,4,2,2` ve `512,8,2,2`.
- d512'deki fark (A − B = 0,115) elde var.
- Karar kuralı (önceden yazılır):
  - Fark belirgin küçülürse (örneğin < 0,06), BLT'nin "yerel modüller yavaş büyür" bulgusu Strata'da da geçerli; tez yaşar.
  - Fark aynı kalır ya da büyürse, yerel kısım büyük çözücüyle birlikte büyümek zorunda. O zaman token başı tasarruf
    ölçekte de gelmez; tezin hesap tarafı çürür.
- Maliyet: iki × ~2.000 adım. Uyarı: bu kısa koşudur, büyük model hükmü değil (BLT'de kesişme 1B–7B arasında).
- Daha ucuz ama zayıf bir eşi: eldeki A ve B checkpoint'lerinde ctx kapatmanın (ctx → g0) etkisini karşılaştırmak. B'de
  +0,122 ölçüldü; A'da ölçülmedi. Büyük çözücüye bağımlılık yerel boyla nasıl değişiyor, onu gösterir. Ama ölçek sorusunu
  cevaplamaz.

## 5. İtirazlara rağmen ayakta kalan

- **V1 ve V2 ölçümle dayanıyor.**
  - Bilgi eşitsiz dağılıyor.
  - B'de büyük çözücünün katkısı +0,122 nat ve bu katkı birim içi konumdan bağımsız. Bir birim bayat g yalnız +0,034
    kaybettiriyor; dört birim bayat +0,099.
  - Yani düşük sürprizli token'lardan sonra büyük çözücüyü yeniden koşmamak ucuz. "Hesabı bilgi gelince harca" çekirdeği
    SS'te doğrulandı (ÖLÇÜLDÜ, `sx_prolong.py`).
- **Uzun bağlam maliyeti gerçekten düşüyor.** Büyük çözücünün KV önbelleği ve attention'ı birim ekseninde. 2,7 kat az
  birim, KV'de 2,7 kat, attention FLOP'unda ~7 kat daha az iş demek (HESAP). Bu, batch'li üretimdeki bant genişliği
  itirazından etkilenmez; uzun bağlamda baskın maliyet zaten KV okuma.
- **Literatürün yönü tezle aynı.** BLT ve H-Net, hesap eşitken sıkıştırılmış birimler üzerinde büyük çözücünün ölçekte BPE
  transformer'ını yakaladığını ya da geçtiğini bildiriyor. BLT'de büyük yama ölçek büyüdükçe daha iyi. Ama bunlar baytlar
  üzerinde; BPE token'ı üzerinde ~2,7 katlık ek sıkıştırmanın ölçekte bedava olduğuna dair kanıt yok. Strata'nın asıl
  sorusu bu ve ancak §4'teki gibi ölçek ölçümüyle cevaplanır.
- **Nitel taraf.** SS 1 epokta 10 / 10 hikâye EOS ile bitti, döngü yok, isimler tutarlı. Yani mimari "sağlıklı çalışan model"
  ölçütünü SS'te geçiyor; açık kalite derinliğinde (olay örgüsü dar), çöküşte değil.

**Hüküm.** Tezin ilkesi (bilgi gelince güncelle) ölçümle ayakta. Ekonomisi ("büyük çözücü yalnız birimlerde, yerel kısım
küçük") SS'te ölçüldüğü kadarıyla tutmuyor: kalite yerel hesap ≈ büyük çözücü hesabı olunca toparlandı. Bu, ölçekte BLT'nin
gösterdiği gibi düzelebilir ya da düzelmeyebilir; tek ucuz ayırıcı §4'teki test.
