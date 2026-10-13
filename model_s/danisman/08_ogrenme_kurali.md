# 08 — Öğrenme kuralı: problem, sınırlar, gevşeklik (11 Ekim 2026, mathematician; TERSINE §14)

Kural: önce hesap ve kanıt. Etiketler: KANIT, HESAP (varsayımıyla), ÖLÇÜLDÜ, LİTERATÜR (dosya + satır; hafızadan olan ayrıca işaretli),
HİPOTEZ / TAHMİN yalnız boşluğun adı. Proje sayıları: FineWeb V2, satır 2.048 × batch 32 = **B = 65.536 token / adım**;
d1024 / L12 tam epok 12.074 adım = **D ≈ 0,79B token**, 374,6 ms / adım, sınav 2,9716 (OLCULENLER_z, 9 Ekim).

## 0. Sonuç önce

1. **Sinyal ve geometri ekseninde ilkesel tavan sabit ve küçük; büyük gevşeklik verinin nasıl harcandığında.**
   - Geri yayılım, kesin gradyanı fonksiyon değerinin sabit katı maliyetle veriyor (ucuz gradyan ilkesi, KANIT).
   - Yalnız ileri geçişli bir kestiricinin varyansı parametre sayısı n ile büyüyor: n 1,4·10⁸ ⇒ bir geri yayılım ≈ 10⁸ ileri geçiş (KANIT, §3).
   - Geometri kazançları bu projede ölçüldü: NorMuon −0,012 nat ⇒ etkin veri çarpanı m ≈ 1,04–1,08 (HESAP, §2).
   - "Transformer kaybına 2× kısa sürede" ⇔ eşit token maliyetinde m = 2 ⇔ d1024'te sabit token'da −0,10 ile −0,21 nat.
2. **Öğrenme kuralı veri üssünü değiştiremez, yalnız çarpanını (KANIT, LİTERATÜR koşullu).** Bayes öngörücüsünün öğrenme eğrisi, metnin öngörüsel
   bilgisinin türevi (`2001/bialek2001_predictability.txt` L502–515). Hilberg tipi kuvvet yasasında üs veriden gelir
   (`2025/debowski2025_zipf_hilberg_scaling.txt` L28–40, L74–76). Kalan eksenler:
   - (i) batch / gürültü rejimi: (S/S_min − 1)(E/E_min − 1) = 1 (`2018/mccandlish2018_largebatch.txt` L174);
   - (ii) önem örneklemesi: varyans çarpanı E‖g‖² / (E‖g‖)² (KANIT);
   - (iii) seyrek bilginin tek görüşte saklanması. SGD ~100–1.000 görüş istiyor (`2024/allenzhu2024_physics33_capacity.txt` L284, L334); ideal 1.
     En büyük kuramsal gevşeklik bu, ama kural değil mimari + kural değişikliği.
3. **Tek bilinmeyen: dizi başına gradyanlardan iki sayı.** Gradyan gürültü ölçeği B_noise = tr(Σ)/‖G‖² (token cinsinden) ve önem örneklemesi
   çarpanı ρ_IS = E‖g_s‖² / (E‖g_s‖)². Checkpoint'te CPU'da ~32 dizi için geri geçişle ölçülür.
   - B_noise ≫ 65k ise veri ekseni tükenmiş (E ≈ E_min). B_noise < 65k ise token israfı var, çarpanı (1 + B/B_noise).
   - **Token başına dağılım (aday) yanlış birim:** hesap tasarrufu yalnız dizi düzeyinde mümkün (§4). Seçim dizi düzeyinde ölçülmeli.

## 1. §14 formülünün düzeltilmesi

- **(a) Hedef duvar saati, FLOP değil.** Kullanıcının hedefi "hedef kayba daha kısa sürede".
  - T = Σ_k τ(B_k). Doymuş GPU'da adım süresi τ(B) ≈ B·τ_token. Doymamış GPU'da τ ≈ sabit.
  - Değişkenler: adım S, batch B, token E = S·B. McCandlish ödünleşimi S ile E'yi bağlıyor (L174, B_crit = E_min / S_min L183).
- **(b) Takvim kuralın parçası ve ölçüm noktası.** WSD'de kalite inişten sonra okunur (proje dersi: sabit-lr penceresi yanıltır).
  Amaç: inişten sonraki kayıp.
- **(c) Popülasyon kaybı.** Tek epokta eğitim kaybı = sınav kaybı. Yeniden kullanım (epok, damıtma) eklenince ayrışır; amaç E_𝒟 olmalı
  (tekrarlanan veri daha az değerli, `2023/muennighoff2023_dataconstrained.txt` L324).
- **(d) Veri bir seçim değişkeni.** Ağırlık w_i, sıra, seçim. Seçmenin maliyeti ileri geçiş (token maliyetinin ~1/3'ü) ve bu maliyete girmeli.
- **(e) Ayar arama maliyeti.** lr / takvim taraması da hesap. Aktarılabilir parametreleme (μP) bunu düşürür.

Düzeltilmiş amaç:

    min_{R, w, B_k, takvim}  T_duvar  s.t.  L_𝒟(θ_K, iniş sonrası) ≤ L*,   T_duvar = Σ_k τ(B_k) + τ_seçim + τ_ayar

## 2. Sınırlar ve projenin uzaklığı (HESAP)

- **Etkin veri çarpanı m.** Bir değişiklik aynı kaybı D/m token'da veriyorsa duvar saati 1/m. Ölçülen ΔL'den m = exp(ΔL / s), s = dL / d ln D (token'ın e katı başına).
  - s'nin proje içi kaba kestirimi: d768 3.000 adım 3,357 → tam epok ~3,10; d1024 3.000 adım 3,45 → tam epok 3,00. Ham eğim 0,18 / 0,32 nat.
  - 3.000 adım noktaları inişsiz olduğu için eğim şişik. s ≈ 0,15–0,3 alındı.
  - NorMuon −0,012 → m 1,04–1,08. Z2 eklentileri tam epokta −0,033 → m 1,12–1,25. Hedef m = 2 → ΔL = s·ln 2 = **0,10–0,21 nat** (sabit token'da).
- **Hesap verimi.** d1024: N ≈ 1,4·10⁸ etkin parametre, adım başı 6·N·B ≈ 5,5·10¹³ FLOP / 0,375 sn ≈ 150 TFLOP/s elde (HESAP). Donanım tavanı
  bilinmiyor; MFU ayrı (speed-analyst). Kuralın asıl ekseni token verimi m.
- **Batch rejimi (LİTERATÜR + HESAP).** E/E_min = 1 + B/B_noise ve S/S_min = 1 + B_noise/B.
  - LM'lerde B_crit kayıp düştükçe büyüyor (McCandlish L30, L50).
  - Hafızadan: Kaplan 2020'nin B_crit(L) uydurması 3 nat civarında ~10⁶ token veriyor. Klasörde yok, doğrulanmadı.
  - Doğruysa B = 65k ≪ B_noise: token verimi zaten tavanda (E ≈ E_min), adım verimi değil. Sabit GPU'da adım süresi B ile büyüdüğü için kazanç yok.
    Bu eksenin gevşekliği tek ölçümle (§5) kapanır.
- **Veri üssü (KANIT, Bayes koşullu).** Λ(T) = S(T+1) − S(T) − S₀ (Bialek L513–515). Bayes öngörücüsü, önseli altında ortalama log-kayıpta en iyisi.
  - Öğrenme eğrisinin üssü, öngörüsel bilginin büyüme üssü γ'dan gelir: Λ ∝ T^{γ−1}.
  - Bir kural ancak SGD'nin üssü Bayes'inkinden kötüyse üs kazandırabilir. Ölçülmemiş; ölçümü uzun koşu ister (hafızadan: dil için γ ~0,5–0,8).

## 3. Sinyal ekseni: geri yayılımdan "daha efektif" sinyal (KANIT)

- **Ucuz gradyan ilkesi (Baur–Strassen; hafızadan).** Hesaplanabilir her f için ∇f'nin ters mod maliyeti ≤ c·maliyet(f), c ≈ 3–5.
  - Bu projede eğitim adımı ≈ 3× ileri geçiş (ileri 2N, geri 4N).
  - Kesin gradyanı veren hiçbir yöntem bu sabitin altına inemez: tasarruf tavanı ≤ 3×, o da geri geçişi tamamen atlamakla. Yani öğrenmeyi öldürmekle.
- **Yalnız ileri kestirici.** ĝ = (∇f·u)u, u ~ N(0, I). E‖ĝ‖² = (n + 2)‖∇f‖². Bir yön ölçümü ≈ 1 sayı bilgi verir.
  - Bir geri yayılımla aynı varyansa ~n ileri geçişle ulaşılır. n 1,4·10⁸ ⇒ ~10⁸ kat pahalı.
  - Yerel kurallar (Hebb, öngörüsel kodlama) bir hedefe doğru iniyorsa ya bu varyansı ödüyor ya da geri yayılımı yaklaşık yineliyor.
- **Sonuç:** sinyalin kendisi gevşek değil. Gevşeklik, sinyalin **hangi veriden** ve **ne sıklıkla** alındığında.

## 4. Veri ekseni: hangi token / dizi öğretir (KANIT)

- **Önem örneklemesi.** Örnek i'yi q_i olasılıkla seç, 1/(N q_i) ile ağırlıkla: yansız.
  - Toplam varyans q_i ∝ ‖g_i‖'de en küçük. tr Σ_IS = (E‖g‖)² − ‖G‖²; tekdüzede E‖g‖² − ‖G‖².
  - ‖G‖² küçükken gürültü baskın rejimde E_min ∝ tr Σ ⇒ **m_IS ≈ E‖g‖² / (E‖g‖)²**.
  - Örnek: log ‖g‖ sapması σ olan lognormalde m_IS = e^{σ²}: σ 0,5 → 1,28; σ 1 → 2,72.
- **Hesap tasarrufu yalnız dizi düzeyinde (KANIT, yapısal).** Geri geçiş FLOP'u işlenen konum sayısıyla orantılı, kayıp terimi sayısıyla değil.
  Bir dizi içinde token'ların kaybını sıfırlamak, attention ve öteki konumlar üzerinden geri geçişi azaltmaz (çıkış başı hariç).
  - Atlanabilen birim dizi (belge parçası). Seçim maliyeti ileri geçiş (adımın ~1/3'ü) ve dizinin ‖g_s‖'si için bir gösterge gerekir.
  - Kayıp ya da logit gradyanı ‖p − e_y‖² = 1 − 2p_y + Σp²; ileri geçişten bedava çıkar.
- **Tavan (HESAP).** f oranında dizi geri geçişe girerse adım maliyeti 1/3 + (2/3)·f. f 0,5'te 1,5× ucuz.
  Horvitz–Thompson varyansı Σ‖g_s‖²(1/q_s − 1), Σq_s = f·S. ρ_IS büyükse (≥ 2) seçim hem varyansı hem hesabı düşürür.
  ρ_IS ≈ 1 ise bu eksen boş.
- **Tek görüşte saklama (LİTERATÜR + HESAP).** 2 bit / parametre kapasite her bilgi parçası ~1.000 kez görülünce (Allen-Zhu L284); 100 görüşte düşük (L334).
  Bilgi kuramsal olarak deterministik bir olgunun tek görüşü yeter. Bağlam arşivi bunu yapıyor (Tur B), parametrik öğrenme yapmıyor.
  - Gevşeklik çarpanı nadir olgularda 10²–10³. Ama kayıptaki payı bu ölçekte bilinmiyor.
  - Kural değil, mimari (eğitimde yazılan parametrik dışı bellek) + kural değişikliği. Kayıt için yazıldı; bu turun ekseni değil.

## 5. Geometri ve yeniden kullanım (kısa)

- **Geometri.** Gürültü baskın rejimde ön koşullama yönü düzeltir ama E_min'i ancak koşul sayısı oranında düşürür.
  Ölçülen: NorMuon m ≈ 1,04–1,08 (d1024). Muon / NorMuon zaten kullanımda; kalan gevşeklik küçük (HESAP, §2).
- **Yeniden kullanım.**
  - Tekrar: veri sınırlıysa değerli, FineWeb'de sınır yok.
  - Büyüyen model / damıtma: öğretmen yoksa damıtma yok.
  - Etiket gürültüsü: tek sıcak hedef p*'den tek örnek; logit gradyan varyansı ≈ 1 − Σ_y p*(y)² (entropi ~3 nat'ta baskın). Kaldırmak p*'yi bilmeyi ister.
    Ucuz bir yaklaşık p* (bağlam içi n-gram sayımı) öğretmen olabilir mi? Ölçülmedi (HİPOTEZ, aynı veriden gelir, yeni bilgi eklemez).

## 6. Hangi eksen, hangi koşulda (HESAP özeti)

| eksen | ilkesel tavan | koşul | proje durumu |
|---|---|---|---|
| sinyal (geri yayılım yerine) | ≤ 3× (ucuz gradyan) | — | kapalı (§3) |
| geometri | koşul sayısı oranı | gürültü az | ölçülen m 1,04–1,08 |
| batch rejimi | 1 + B/B_noise | B > B_noise | **ölçülecek** |
| veri seçimi (dizi) | ρ_IS · 1/(1/3 + 2f/3) | ρ_IS büyük | **ölçülecek** |
| tek görüşte saklama | 10²–10³ (nadir olgu) | nadir bilgi payı büyük | mimari; ayrı tur |
| veri üssü | yalnız SGD üssü Bayes'ten kötüyse | — | uzun koşu ister |

## 7. Tek bilinmeyen ve ölçüm (CPU, eğitimsiz)

- **Ölçülecek:** checkpoint'te (FineWeb transformer d768 / L10; varsa ara checkpoint'ler de, B_noise kayıpla büyür) 32 dizi × 2.048 token için
  dizi başına gradyan g_s (32 geri geçiş).
  - ‖G_B‖² = ‖(1/B)Σ g_s‖², ‖g_s‖².
  - McCandlish'in iki batch kestiricisi (B_küçük 1, B_büyük 32): ‖G‖² ≈ (32‖G_32‖² − ‖G_1‖²)/31, tr Σ ≈ (‖G_1‖² − ‖G_32‖²)/(1 − 1/32).
  - **B_noise = tr Σ / ‖G‖²** (dizi), × 2.048 → token.
  - **ρ_IS = mean ‖g_s‖² / (mean ‖g_s‖)².**
  - Gösterge kalitesi: Spearman(‖g_s‖, dizinin ileri geçişten çıkan logit gradyan toplamı).
- **Karar (HESAP):**
  - B_noise > 2·10⁵ token → batch ekseni kapalı.
  - ρ_IS ≥ 2 ve gösterge Spearman ≥ 0,7 → dizi seçimi m ~1,5–2'ye aday.
  - İkisi de değilse kural ekseninde 2× için ilkesel yer yok; kalan iş mimari (tek görüşte saklama) ya da üs.
- Çekince: ham gradyan üzerinden. NorMuon'un yönü dikleştirilmiş; ön koşullanmış B_noise farklı olabilir. Dizi içi token'lar ilişkili olduğu için
  token cinsinden B_noise yaklaşık.
- TAHMİN (boşluğun adı): B_noise ~10⁵–10⁶ token; ρ_IS 1,3–2.
