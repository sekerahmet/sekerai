# 03 — Bağımsız matematik: Strata-x'in hesap ekonomisi (11 Ekim 2026, mathematician)

Nesne: `model_s/strata_x.py` (çalışma ağacı), sayılar `GOZDEN_GECIRME.md` §1–15 ve `TASARIM.md`:66–70. 01 / 02 danışman raporları
bağımsızlık için okunmadı. Hesap betiği: scratchpad `matematikci/strata_econ.py` (CPU, saniyeler). Tek tohum; her sayı tek koşu.

## 0. Önce en önemli üç bulgu

1. **SS'te Thinker bilgi değil yalnız hesap ekliyor (KANIT, §2.1), o hesap da token düzeyinden 2,0–2,5× az verimli (HESAP, §1).**
   Senin 2,44× hesabın doğru. Ama bu sayı bir alt sınır gibi okunmalı: kayıp eğrisi içbükeyse oran daha büyük çıkar. Thinker'sız
   tabandaki aux başı oranı ~2,0'a indirebilir. FLOP ve duvar saati aynı oranı veriyor (2,44× / 2,3×). Thinker büyütüldüğünde (2×)
   verim 19 kat düşüyor: Thinker doymuş.
2. **Kuramsal ölçüt: E_birim / E_token = π/ρ; tez ancak π > ρ ise geçerli** (§2.2). Amortisman modelinde bu oran w_s·τ·(1 − e^{−ℓ/τ}).
   SS'e uydurunca başabaş noktası **ℓ* ≈ 6–7 token/birim** (τ'ya duyarsız). Bugünkü birim 2,4 token; amortismanın en kötü olduğu yer.
3. **Literatür aynı tabloyu çiziyor** (§3): aynı kısaltma fikri karakter düzeyinde 1,5–3× kazandırıyor (Hourglass). BPE üstüne ~2×
   sıkıştırma ise küçük ölçekte kaybediyor, büyük ölçekte / çok veride kazanıyor (BLT ps 8, H-Net 2 aşama). Strata bugün
   kaybetmesi beklenen bölgede. Ayırıcı en ucuz test **θ taraması (ℓ ≈ 6 ve ≈ 11)**: iki kısa koşu, tahmin §4'te.

## 1. Soru 1 — marjinal verim (kayıp düşüşü / FLOP)

Varsayımlar (HESAP): ileri FLOP / pozisyon = 2·(4d² + 3d·h), h = _hidden(d) → d512 katmanı 6,30 MFLOP, d256 1,58, d1024 25,2.
SS'te dikkat terimi (2·T·d, T ≤ 256) ≤ %4, ihmal edildi. ρ = 0,42 birim/token (2,4 token/birim). Thinker katmanının token başı maliyeti
ρ katman. **eq** = d512 token katmanı eşdeğeri. Kayıplar GOZDEN_GECIRME'den (ÖLÇÜLDÜ, tek tohum).

| kıyas | düzey | kazanç (nat) | ek hesap (eq) | nat / eq |
|---|---|---|---|---|
| Thinker'sız W512 (2+2) → transformer 8 katman | token | 0,1056 | 4,0 | **0,0264** |
| Thinker'sız W512 → B3 (Thinker d512/8) | birim | 0,0363 | 3,36 | **0,0108** |
| W256 → W512 genişlik, Thinker'sız | token | 0,172 | 3,0 | 0,057 |
| W256 → W512 genişlik, Thinker'lı | token | 0,115 | 3,0 | 0,038 |
| W256 derinlik 2+2 → 4+4, Thinker'lı | token | 0,045 | 1,0 | 0,045 |
| W256 Thinker'sız → Thinker'lı | birim | 0,094 | 3,36 | 0,028 |
| Thinker d512/8 → d1024/12, W512 | birim | 0,023 | 16,8 | 0,0014 |
| aynısı, W256 | birim | 0,037 | 16,8 | 0,0022 |

- Aynı Writer'da token / birim verim oranı: W512 için **2,44×**; W256 için 1,6× (derinlikle) ve 2,0× (genişlikle) (HESAP).
- Duvar saati (ÖLÇÜLDÜ ms): Thinker 90 → 114 ms (+24 ms) karşılığında 0,036, yani 0,0015 nat/ms. Transformer'ın fazla 4 katmanı
  90 → 120 ms (+30 ms) karşılığında 0,106, yani 0,0035 nat/ms. Oran 2,3× (HESAP; aux başı farkı içinde, kaba).
- Toplamda B3 token başına 55,2 MFLOP, transformer 54,6 MFLOP (oran 1,01; ρ 0,37–0,48 için 0,96–1,07). Yani **eşit hesapta 0,069 nat
  geride**. Doğrusal aradeğerle transformer 1,558'e ~5,4 katmanda iniyor; bu B3 FLOP'unun ~%69'u (HESAP).
- Düzeltmeler: (a) L(n) içbükeyse 4→8 aralığının ortalama eğimi, ilk 3,36 katmanın eğimini olduğundan düşük gösterir; o yüzden 2,44
  alt sınırdır. (b) Thinker'sız tabanda 2. katmanda aux başı ve pencere 256 var, transformer'da yok. Aux başı tabana δ kadar zarar
  veriyorsa oran (0,106 − δ)/4/0,0108 olur; δ 0,02 için 2,0×. (c) Aux başı (4,2 MFLOP) toplam FLOP'un %7,6'sı: birim mekanizmasının
  sabit vergisi (HESAP; ölçülen adım payı ~%20).

## 2. Soru 2 — kuramsal gerekçe

**2.1 Bilgi yok, yalnız hesap var (KANIT).** g_t = G(x_{≤e(t)}) ve e(t) ≤ t. Hikâye başı t − W + 1'den sonraysa x_{≤e(t)} pencerenin
içindedir. O zaman I(x_{t+1}; g_t | pencere) = 0: sınırsız kapasiteli bir Writer'a Thinker hiçbir bilgi veremez, verebileceği tek şey
amortize edilmiş hesaptır. SS'te pencere 512'nin etkisiz kalması (ÖLÇÜLDÜ) bu koşulun büyük ölçüde sağlandığını gösteriyor.
Ölçümdeki karşılığı **ikame etkileşimi**: Δ_T(W256) − Δ_T(W512) = 0,094 − 0,036 = 0,058. Writer büyüyünce Thinker etkisinin %62'si
kayboluyor. Thinker bilgiyi tamamlayan bir parça olsaydı Δ_T Writer büyüdükçe küçülmezdi (HİPOTEZ: ayırıcı ölçüt).

**2.2 Verim formülü (tanım).** Bir token katmanı Δ kazandırıp 1 birim hesaba mal olur; bir birim katmanı πΔ kazandırıp ρ'ya mal olur.
O halde E_u/E_t = π/ρ ve **tez ⇔ π > ρ**. SS'te π = 0,41 × 0,42 ≈ 0,17. Yani bir Thinker katmanı ~0,17 token katmanı değerinde
ama 0,42'sine mal oluyor (HESAP).

**2.3 π neden küçük: amortisman modeli (HİPOTEZ, iki parametre).** Derinliğin marjinal değerinin bir kısmı **hızlı** özelliklere gider
(her token'da değişenler: son token, sözdizim durumu). Thinker bunlara yardım edemez: girdisi yalnız birim sonu vektörü, çıktısı
birim boyunca bayat kalır. Kalan w_s kesri **yavaş** özelliklere gider (ortalama ömrü τ token olanlar: kim, nerede, ne oluyor).
Birimde hesaplanan yavaş bir özellik ℓ token'a hizmet eder; birim boyunca bayatlamama olasılığı s(ℓ) = (τ/ℓ)(1 − e^{−ℓ/τ}). Buradan:

    π = w_s·s(ℓ)   ⇒   E_u/E_t = w_s·ℓ·s(ℓ) = w_s·τ·(1 − e^{−ℓ/τ})

SS'e uydurma (ℓ 2,4, oran 0,41): τ 10 / 15 / 25 için w_s 0,19 / 0,19 / 0,18. Başabaş ℓ* = −τ·ln(1 − 1/(w_s τ)) ≈ **7,3 / 6,6 / 6,3**.
Model içinde ℓ 8 → 1,06–1,23, ℓ 16 → 1,5–2,1 (HESAP). Yorum: BPE yazım içi fazlalığı zaten silmiş; Hourglass'ın karakterdeki kazancı
o fazlalıktan geliyor. 2,4 token'lık bir birim hızlı özelliklerin ömründen uzun değil, yavaş özelliklerinkinden ise çok kısa.
**Birim vektörü darboğazı** ölçümle elendi: Voice etkisiz, 2× Thinker 0,0014/eq. Darboğaz bant genişliği değil. Sorun, Thinker'ın
hesapladığı şeyi Writer'ın zaten hesaplayabilmesi, üstüne doyma.

**2.4 Doyma.** İki ölçüm noktası var: 3,36 eq → 0,036 ve 20,2 eq → 0,059 (W512). Üstel uydurma V∞ 0,059, x0 3,6 eq ve başlangıç eğimi
0,0167 veriyor; bu 0,0264'ün altında, yani ilk katmanlar da daha az verimli. Kuvvet yasası (β 0,27) ise ilk katmanları verimli
gösteriyor. İki nokta bu iki modeli ayırt etmiyor (§4 T2).

**2.5 Hangi koşulda daha verimli olur (HESAP / HİPOTEZ).**
- (i) Uzun birim, ℓ ≳ 7: §2.3'teki model. Aynı model şunu da öngörüyor: FineWeb'de sabit θ 8 birimi 1,8 token'a indirdiği için
  (TASARIM.md:70) en kötü bölgeye düşüyor; oran tabanlı θ ℓ ≥ 7'yi hedeflemeli.
- (ii) Bağımlılık pencereyi aşınca: §2.1'deki eşitlik bozulur, Thinker bilgi taşır, ikame tamamlayıcılığa döner. Maliyet tarafı:
  oran(T) = [n_loc(D + 4Wd) + n_g·ρ(D + 2ρTd)] / [n(D + 2Td)]. d512, W 256, ρ 0,42 için T 256 → 0,91; 2k → 0,77; 8k → 0,51;
  32k → 0,30. KV önbelleği 8k'da ~0,43×. 1B'de d büyüdükçe dikkat payı küçülür, aynı formülle yeniden hesaplanmalı.
- (iii) Ölçek / veri: w_s büyüyebilir, çünkü yerel istatistik küçük modelde önce doyar. Bu, BLT / H-Net'teki "önce kötü, sonra iyi"
  desenine uyuyor (HİPOTEZ).
- (iv) Veri türü: yavaş özellik payının ve τ'nun büyük olduğu metin (uzun anlatı, kod kapsamı).

## 3. Soru 3 — literatür (klasörden; metin satırla doğrulanabilir)

- **Hourglass** (`2021/nawrot2021_hourglass.txt`): Tablo 1 (satır 410–440) enwik8 karakter verisinde. "2@1 8@4 2@1" BPC 1,128;
  "10@1 (Baseline)" de 1,128. 4 tam katman + 8 kısaltılmış katman (k 4) = 6 eq, 10 katmanla aynı sonucu veriyor: π ≈ 0,75, π/ρ ≈ 3.
  Tablo 5 (satır 698, k 3, vanilya (2,2)): n 4 → 8 için 1,134 → 1,111, yani 0,0173/eq. Vanilya 8 → 10 için 1,151 → 1,128, yani
  0,0115/katman: **1,5×** (HESAP, onların tablolarından). §4.6 (satır 731): kısaltma en az 3 kat olmalı ve 2–3 vanilya katmanı korunmalı.
  Aynı fikir karakterde 1,5–3× kazandırıyor, BPE üstünde (Strata) 2,4× kaybettiriyor. Fark taban dizideki fazlalıktan.
- **Block Transformer** (`2024/ho2024_block_transformer.txt`): satır 463'e göre 2–3 kat parametreyle vanilya ile aynı perplexity'ye
  ulaşılıyor. Yani kazanç parametre başına kalitede değil, throughput'ta (KV IO). §3.6 (satır 747) eğitim FLOP'u ve throughput birlikte
  kısıtlandığında daha iyi perplexity iddia ediyor (2048 bağlam, 300B token). Ama orada token çözücü yalnız kendi bloğunu görüyor
  (satır 123); ikame olanaksız, tamamlayıcılık zorunlu. Strata'da 256'lık pencere ikameye yol açıyor. Ek Q (satır 3228): token
  çözücüyü küçültmek ağır bedel — Writer'ın büyük kalma zorunluluğuyla aynı.
- **BLT** (`2024/pagnoni2024_blt.txt`, satır 636–643): Llama 2'nin BPE'si 3,7 bayt/token, yani ps 8 ≈ 2,2 BPE token/yama. Bu Strata'nın
  2,1–2,7'siyle **aynı sıkıştırma**. Satır 641: ps 8 "starts at a significantly worse point compared to bpe Llama 2 at 1B" ve 7B'de BPE'yi
  geçiyor. Tezin en doğrudan karşılığı bu: bu sıkıştırma küçük ölçekte kaybediyor.
- **H-Net** (`2025/hwang2025_hnet.txt`): (3,3)-DC ≈ 9 bayt/iç parça ≈ ~2 BPE token. Satır 1487: bayt düzeyindeki bütün H-Net'ler
  token Transformer'ının gerisinde başlıyor, yeterli veriden sonra öne geçiyor. 2 aşamalı H-Net 30B baytta kesişiyor (satır 1489–1495).
  SS 2.000 adım bunun çok altında (adım başı token sayısı bende yok).
- İlgililik (C1): BLT ve H-Net bayttan başlıyor; Block Transformer'ın blokları sabit 4 token. "SS ölçeğinde (25–40M, 2k adım) birim
  hesabı FLOP başına daha iyi" diyen doğrudan bir karşılık bulamadım. Kazanç ya throughput'ta (ho2024) ya da ≥1B / ≥30B baytta
  (BLT, H-Net) (LİTERATÜR).

## 4. Soru 4 — en ucuz ayırıcı test (eğitim koşusuyla sınanmalı; karar kullanıcının)

- **T0 (GPU yok, kayıt okuma).** (a) B3 ve `sxnothk_512` ara kayıplarında Thinker katkısı 500 → 2.000 adımda büyüyor mu? Tahmin
  (TAHMİN): büyüyor; w_s öğrenmeyle artıyor. (b) "oran θ" koşusunun (1,579) ortalama ℓ'si ne? ℓ ≥ 4 idiyse modelimi çürütür:
  model ℓ 4'te ~1,560 diyor, ölçülen 1,579. ℓ ≈ 2–3 idiyse bilgi vermez.
- **T1 (birincil, B3 ayarında yalnız θ değişir, 2 kısa koşu).** ℓ ≈ 6 ve ℓ ≈ 11 hedeflenir (θ, koşunun kendi birim istatistiğinden
  seçilir; TASARIM:174'teki tablo başka koddan). Ölçü: Thinker katkısı = 1,5942 − kayıp; bunu 8/ℓ eq'e böl.
  Başabaş çizgisi: ℓ 6'da kayıp ≤ 1,559, ℓ 11'de ≤ 1,575. Amortisman modeli (τ 15) ℓ 6 için 1,562, ℓ 11 için ~1,566 öngörüyor.
  "π sabit" (katkı hesapla orantılı) alternatifi ℓ 6 için 1,580, ℓ 11 için 1,587 öngörüyor. Tahminim (TAHMİN): ℓ 11 çizgiyi geçer,
  ℓ 6 sınırda kalır. Geçerse tez SS'te uzun birimle yaşar (BLT / Block T. ile aynı yön). Geçmezse SS'te düşer ve yalnız §2.5
  (ii) / (iii) bölgesinde aranır.
- **T2 (ikincil).** Thinker 2 katman (0,84 eq). Başabaş 0,022 (≤ 1,572); üstel 1,582, kuvvet yasası 1,569; tahminim 1,578.

## 5. Açık sorular
Saf 4 katmanlı d512 transformer (aux başsız) yok, token eğiminin tabanı kirli olabilir (§1 (b)). ℓ büyüyünce Reader'ın birim sonu
vektörü yavaş özellikleri taşıyabilir mi? 1B'de oran(T) için gerçek d, n_loc, n_g ve ρ gerekli.
