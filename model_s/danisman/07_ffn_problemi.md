# 07 — FFN problemi: kuruluş ve türetme (11 Ekim 2026, mathematician; TERSINE §12)

Kural: önce hesap ve kanıt; bileşen amaçtan türer. Etiketler: KANIT, HESAP (varsayımıyla), ÖLÇÜLDÜ, LİTERATÜR (dosya + satır),
HİPOTEZ / TAHMİN yalnız boşluğun adı. Gösterim: d genişlik, M gizli birim sayısı, h_t FFN girdisi (kalıntı),
FFN(h) = Σ_i a_i(h)·v_i, a_i = silu(g_i·h)·(u_i·h), v_i = W_d'nin i. sütunu.

## 0. Sonuç önce

1. **Çözüm bilinen aileye çıkıyor.**
   - Her token'da kullanılan bilgi (kural, sözdizimi) yoğun hesapta; tek tük token'ın kullandığı bilgi (olgu) seyrek aramada.
   - Kullanım sıklığı f_i olan birim i için eşik **f_i < 1/α** ise seyrek havuza gider (KANIT, doğrusal maliyet modelinde). α, seyrek / dağınık
     hesabın yoğun matris çarpımına göre duvar saati çarpanı.
   - Seyrek kısmın taneciği (tek birim = PKM, blok = MoE), birimlerin birlikte etkinleşme küme boyuyla belirlenir.
   - Yoğun FFN λ_P → ∞ (bellek pahalı) ya da veri kıt sınırı. MoE / PKM / Engram λ_P küçük, veri bol sınırı.
   - Engram'ın U biçimli dağıtım yasası bunun ampirik hali (`2026/deepseek2026_engram.txt` L66, L182, L578).
2. **FFN'in okuma kuralı da ilkesel, attention'ınkiyle aynı çatıdan (KANIT).**
   - FFN parametrik bir arşivden normalize edilmemiş toplamla okuyor. Attention bağlam arşivinden softmax ile okuyor.
   - Softmax "tek kaynak" varsayımının Bayes çözümü (06 §5). Normalize edilmemiş toplam "bağımsız kanıtlar log uzayında toplanır" varsayımının
     çözümü (04 §2: log-doğrusal / uzman çarpımı).
   - Derinlik = gereken bileşim (sıçrama) sayısı; c sıçramalı bir bileşimi tek tabloda tutmak |V|^c giriş, c tabloda c·|V| giriş ister (KANIT, sayım).
3. **Tek bilinmeyen: katkı matrisi C[t,i] = a_{t,i}·(v_i·g_t)** (g_t = ∂ℓ/∂FFN çıkışı). Eğitimsiz, CPU'da tek ileri + geri geçişle ölçülür.
   Senin adayın (token başına ilk k) bu matrisin satır görünümü. Kararı veren sütun görünümü:
   - birim başına kullanım sıklığı f_i ve onun dağılımı (baş / kuyruk);
   - birlikte etkinleşme küme boyu c.
   - "Yoğun pay" = f_i > 1/α olan birimlerin oranı: yoğun FFN'in ne kadarının doğru, ne kadarının aramaya taşınabileceğini tek sayıyla söyler.

## 1. §12 formülünün düzeltilmesi

§12'deki amaç: min E[−log p] + λ_F·FLOP + λ_P·parametre. Eksikler:
- **(a) Veri ve öğrenilebilirlik nicel değil.**
  - Bilgi kapasitesi ~2 bit / parametre, ancak her bilgi parçası eğitimde ~1.000 kez görülürse (`2024/allenzhu2024_physics33_capacity.txt`
    L56, L242, L284). 100 kez görülürse kapasite düşük (L334).
  - Kısıt: bir olgu ancak e_f ≥ e_min kez geçiyorsa parametrede tutulabilir. Toplam eğitim hesabı 6·A·D ≤ C (A etkin parametre, D token).
  - P (toplam) ile A (etkin) ayrı değişkenler. Yoğun FFN P = A özel hali.
- **(b) λ_P iki şeyi karıştırıyor.**
  - Kapasite: bellek boyu ∝ P.
  - Bant genişliği: token başına okunan bayt. Batch B ile seyrek erişimde ≈ P·(1 − (1 − k/M)^B)/B.
  - Üretim maliyetini ikincisi belirler. Ayrı katsayı gerekir: λ_P·P + λ_B·bayt / token.
- **(c) Çıktı p değil.** FFN çıktısı kalıntıya yazılır, sonraki okumaların (bağlam ya da parametre) sorgusunu kurar. Bu yüzden derinlik L bir seçim
  değişkeni; sıralı adım sayısına (gecikmeye) bir kısıt gerekir.
- **(d) Bağlam / parametre sınırı.** Bağlamda olan bilginin parametrede tutulması gereksiz.
  - Parametrik bilginin değeri I(Y; θ | bağlam); bağlam arşivininki I(Y; arşiv | θ).
  - Amaç ikisini birlikte içermeli: Tur B'deki arşiv ve FFN'in deposu aynı problemin iki arşivi.

Düzeltilmiş amaç:

    min_{P, A, yapı, L}  E[−log p]  +  λ_F·2A  +  λ_P·P  +  λ_B·bayt/token
    kısıt:  6·A·D ≤ C,   olgu f için  e_f ≥ e_min,   L ≥ bileşim derinliği

## 2. FFN = parametrik arşivden okuma (KANIT)

- FFN(h) = Σ_i a_i(h) v_i. M girdili bir arşiv: anahtar (g_i, u_i), not v_i.
  - Geva: FFN katmanları anahtar–değer belleği gibi çalışıyor; her değer çıkış sözlüğü üzerinde bir dağılım doğuruyor (`2020/p_2012.14913.txt`, özet).
- **Okuma kuralının türetimi.** Çıkış log-doğrusal: logit = E·norm(h) (04 §2). O halde v_i bir log-kanıt vektörü, a_i onun ağırlığı.
  - Y verildiğinde koşullu bağımsız kanıtlarda Bayes kuralı log-kanıtların toplamı (04 §2, uzman çarpımı). Normalize edilmemiş, çok girdili
    toplam bu yüzden ilkesel.
  - Attention'ın softmax'ı ise "tek kaynak" varsayımının sonsalı (06 §5).
  - Fark keyfi değil: parametrik bilgi çoğu token'da birden çok bağımsız olgunun birleşimi, bağlamdan okuma çoğu zaman tek bir kaynak.
- **Sonuç:** model = iki arşiv + derinlik.
  - Bağlam arşivi: token başına bir kayıt, softmax ile okunur.
  - Parametre arşivi: M kayıt, toplamla okunur.
  - Derinlik: oku → yaz turları.

## 3. Parametrede tutmak mı, her token'da hesaplamak mı (KANIT, en kısa betimleme ilkesi)

Gereken bir g eşlemesi u kez kullanılıyor.

    tablo:  λ_P·n·b  +  u·λ_F·c_bak          (n giriş, giriş başına b bit)
    kural:  λ_P·ℓ_g  +  u·λ_F·c_hes          (ℓ_g: g'nin betimleme uzunluğu, c_hes: token başına hesap)
    tablo   ⇔  λ_P·(n·b − ℓ_g) < u·λ_F·(c_hes − c_bak)

- **Olgu sıkıştırılamaz** (ℓ_g ≈ n·b) → tablo zorunlu.
- **Kural sıkıştırılabilir** (ℓ_g ≪ n·b) → kural.
- İkisi de FFN parametresinde durur. Fark **erişimde**: kural her token'da gerekir, olgu yalnız ilgili token'da. §4 bunu sayıya çeviriyor.

## 4. Yoğun mu seyrek mi: birim başına eşik (KANIT, doğrusal maliyet modelinde)

- Birim i'nin maliyeti, token başına: yoğunda 6d (gate, up, down). Seyrekte f_i·6d·α + c_yön/M_s.
  - f_i: i'nin gerektiği token oranı, α: dağınık erişim çarpanı (≥ 1), c_yön: yönlendirme maliyeti.
- Seyrek havuz ⇔ f_i·α + c_yön/(6d·M_s) < 1. Yönlendirme ucuzsa: **f_i < 1/α.**
- **Etkin k türetilir, seçilmez:** k = Σ_{i∈seyrek} f_i. Kullanım dağılımından gelir.
- **Tanecik.** k birim, g'lik bloklarda.
  - Token başına maliyet ≈ 2d·M/g (blok yönlendirici) + 6d·g·n_blok(k, g).
  - Birimler bağımsızsa n_blok ≈ k ve g* = √(M/(3k)).
  - Birimler c'lik kümelerde birlikte etkinleşiyorsa n_blok ≈ k/c ve g* ≈ c.
  - Ürün anahtarı yönlendirmeyi 4d·√M'ye indirir ve g = 1'i (PKM) mümkün kılar.
  - Kural: **c büyük → MoE (blok), c ≈ 1 → PKM / Engram (tek giriş).**
- **Sayı (HESAP; d768).**

  | yapı | maliyet / token / katman |
  |---|---|
  | yoğun SwiGLU, M 2.048 | 6dM = 9,4M |
  | PKM, M 65.536, k 64 | yönlendirme 0,79M + 6dk = 0,29M = 1,1M; α 4 ile ~2M |
  | blok, bağımsız birim, g* = 18 | 2dM/g ≈ 5,4M + 6d·g·k ≈ 5,3M = 10,7M (kötü) |
  | blok, c 64 | 1,6M + 0,3M = 1,9M |

  Seyreğin kazancı tamamen f dağılımına ve c'ye bağlı.
- **Öğrenilebilirlik (HİPOTEZ, §1a'dan).** Seyrek erişimde bir olgunun gradyanı yalnız ilgili token'da oluşur. Olgu başına görülme sayısı aynı kalır;
  ilgisiz token'ların gürültü gradyanı kalkar. Kapasite sınırının seyrekte en az yoğundaki kadar iyi olması beklenir. Ölçülmedi.

## 5. Derinlik (KANIT, sayım)

- c adımlı bir bileşim (ör. "X'in doğduğu ülkenin başkenti" = iki parametrik sıçrama) tek tabloda |V|^c giriş, c ardışık tabloda c·|V| giriş ister.
- Her ara sonuç kalıntıya yazılıp bir sonraki okumanın sorgusu olur. Gereken derinlik ≥ parametrik ve bağlamsal sıçramaların toplam zincir uzunluğu
  (06 §6 ile aynı ilke).
- Genişlik derinliğin yerini ancak bileşim zinciri kısaysa tutar.

## 6. Hangi λ'da hangi bilinen aile

| koşul | çözüm | bilinen |
|---|---|---|
| λ_P büyük (bellek pahalı) ya da D küçük (olgu e_min'e ulaşmıyor) | P = A, hepsi yoğun | yoğun FFN |
| λ_P küçük, α orta, birimler kümeli (c ≫ 1) | yoğun baş + blok seyrek kuyruk | MoE (paylaşılan uzman + yönlendirilen uzmanlar) |
| λ_P → 0, D bol, c ≈ 1 | yoğun baş + tek girişli arama | PKM, Memory Layers, Engram |

- **Açıkça:** yeni mimari çıkmıyor. Türetilen, yoğun FFN, MoE ve PKM / Engram'ın aynı amacın farklı λ bölgelerindeki çözümleri olduğu.
- PKM: 12 katman + bellek, 24 katmanlı transformer'ı geçiyor ve çıkarımda iki kat hızlı (`2019/p_1907.05242.txt`, sonuçlar).
  Kullanım çöküşü bilinen risk (PKM kullanımı %25,8 → BatchNorm ile %80,3; TERSINE §7a).
- Engram'ın bulgusu (sabit toplam parametrede MoE ile arama belleği arasında U biçimli en iyi dağıtım) §4'teki eşiğin dağılım hali (LİTERATÜR, L182).
- Olası dar yeni nokta: f_i < 1/α eşiğinin donanım çarpanıyla, c'nin tanecikle açık bağı. Literatürde granülerlik yasaları var (hafızadan;
  klasörde aranmalı). Yeni sayılmamalı.

## 7. Tek bilinmeyen ve ölçüm (CPU, eğitimsiz; FineWeb transformer d768 / L10, 30–50 belge)

- **Ölçülecek nesne:** her katmanda C[t,i] = a_{t,i}·(v_i·g_t), g_t = ∂ℓ_t/∂FFN_çıkış_t (tek geri geçiş). Neden-sonuç seçimi için |a_{t,i}|·‖v_i‖ da.
- **Okunacaklar:**
  1. Satır, k_eff(ε): ΔL ≤ ε için token başına gereken birim sayısı.
     - İki seçimle: neden-sonuç (|a|‖v‖'ye göre ilk k) ve kâhin (|C|'ye göre). Aradaki fark yönlendiricinin tavan payı.
     - ε 0,01 / 0,05 nat.
  2. Sütun, f_i: birimin token'ların gerekli kümesine girme oranı. Dağılımı (baş / kuyruk; Zipf eğimi).
     **Yoğun pay** = #{f_i > 1/α} / M, α ∈ {2, 4, 8}.
  3. Birlikte etkinleşme: gerekli kümelerin Jaccard benzerliğiyle birim kümeleme. c = kütlenin %80'ini toplayan küme boyu.
- **Karar kuralı (HESAP):** yoğun pay ≳ 0,5 → yoğun FFN doğru (SS / FineWeb ölçeğinde). Yoğun pay küçük ve k_eff ≪ M → seyrek. c büyük → MoE, küçük → PKM.
- **Çekinceler.**
  - B2: sıfır atış kapatma bağımlılık ölçer; seyrek eğitilmiş model başka özellikler öğrenir.
  - SwiGLU'da a_i negatif ve küçük değerler alır (ReLU gibi sıfırlanmaz); sıfır atış seyreklik olduğundan düşük görünür → yoğun pay üst sınır.
  - d768 küçük ölçek: olgu payı küçük, yoğun pay büyük çıkabilir. 1B'ye taşınmaz (kural: küçük model varsayımı taşınmaz).
- TAHMİN (boşluğun adı): ilk katmanlarda yoğun pay yüksek (sözdizimi), orta-üst katmanlarda kuyruk kalın. k_eff(0,01) ~ M'nin %10–30'u.

## 8. Açık
- §3'teki kural / olgu ayrımı ölçüde f_i'nin dağılımı olarak görünüyor. Bir birimin hem kural hem olgu taşıması (süperpozisyon) eşiği bulanıklaştırır.
  C'nin birim tabanında değil, seyrek sözlük (SAE) tabanında ölçülmesi daha doğru olabilir. Pahalı; ikinci adım.
- Bağlam / parametre sınırı (§1d) ayrı bir ölçü ister: aynı olgunun bağlamda verilince FFN katkısının düşmesi. Kapsam dışı, not edildi.
