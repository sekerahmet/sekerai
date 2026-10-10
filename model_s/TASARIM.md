# Model S — transformer'ı bir denklemin çözücüsü olarak sıfırdan kurmak

Kullanıcı, 11 Ekim 2026: *"bu denklem üzerinde yoğunlaş Z ya da trasformer mimarisine takılma. bu matematik herşeyin anahtarı gibi
düşün"*; *"istersen modeline bir isim seç. her parçayı ffn, attention vs sıfırdan yazıyorsun gibi düşün. yeni klasör aç acele etme.
parçaları istersen teker teker yaz."*

Kaynak: Tai, Liu, Li, Chan, *A Mathematical Explanation of Transformers* (arXiv 2510.03989; `belge/makaleler/2025/tai2025_math_transformer.txt`).
Ad: **S** = splitting (operatör bölme) / sürekli (zaman). Mimarinin adı **Strata** (kullanıcı, 11 Ekim: *"bu mimariye bir isim de
verebilirsin kendin yepyeni orjinal bir isim"*): tabakalar -- sıralı derinlik yalnız adımların yer değiştirmediği (komütatörü
büyük) yerde, geri kalan adımlar yan yana tek geniş tabakada. Kod adı `model_s` / `--model model_s` kalır.  Klasör: `C:\AI_NEW_MODEL\model_s` (kullanıcı, 11 Ekim: "C:\AI_NEW_MODEL bunun altında model S olarak klasör aç"); veri / eğitim ortak kodu `deneme2/v2/common`.

## 1. Denklem

Durum u(x, y, t): x token konumu, y özellik boyutu, t derinlik (sürekli zaman).

    ∂u/∂t = A(u) + F(u),   u(·, ·, 0) = f   (f: embedding = başlangıç koşulu)

- **A** — attention: x ekseninde yerel olmayan integral operatörü, çekirdek γ(x, x̃; u) = softmax_x̃(⟨Q(x), K(x̃)⟩ / √d), nedensel (x̃ ≤ x).
- **F** — FFN: her x'te ayrı (y ekseninde) doğrusal olmayan operatör (SwiGLU).
- **P** — normalizasyon: kısıta izdüşüm (makale: LN = ortalama / varyans kısıtı). Biz RMSNorm (yarıçap kısıtı) kullanırız;
  pre-norm biçimde operatör, durumun izdüşümünde değerlendirilir: A(P u), F(P u).

## 2. Bugünkü transformer bu denklemin en kaba çözümü

Makale (bölüm 3): bir transformer bloğu = **Lie bölmesi** (operatörler sırayla, 1. derece) + **açık Euler** (alt adım başına tek
değerlendirme) + **Δt = 1**; her katman bir zaman adımı, her adımın kendi parametresi θ_n.

    Lie:   u ← u + A(P u);  u ← u + F(P u)                      (bugünkü pre-norm blok)

Model S'nin amacı: aynı denklemi **daha doğru şemalarla, daha az adımda** çözmek; ölçüt aynı kalitede daha az sıralı işlem
(eğitim ve üretim hızı).

## 3. Parçalar (kural 10: her parça AMAÇ / KULLANIM / ÖĞRENİLME / OKUNMA / YÖNETİM)

| # | parça | dosya | durum |
|---|---|---|---|
| 1 | P, A, F operatörleri | `operators.py` | bitti: A = elle integral (fp32 1,8e-7), nedensel |
| 2 | zaman adımı şemaları (lie, strang, parallel) | `integrator.py` | bitti: lie = bugünkü transformer bloğu (bit düzeyinde) |
| 3 | hibrit aşamalar: aşamadaki adımlar aynı u'dan (`Stage`, `stages`) | `integrator.py` | bitti: k = 1 = sıralı, fark ∝ ε² (BCH) |
| 3b | θ(t): pürüzsüz parametre | `integrator.py` | ertelendi (ölçüm: aşağıda) |
| 4 | model: başlangıç koşulu, çözücü (adım başına genişlik), okuma | `model.py` | bitti: widths hepsi 1 = transformer (bit), uretim = KV cache |
| 5 | eğitim / sınav bağlantısı (`--model model_s --widths ...`) | `deneme2/v2/common/train.py` (dal deneme/model-s) | bitti: CPU 2 epok, sürdürme bit düzeyinde |

### Parça 1 — operatörler

- **AMAÇ:** denklemin sağ tarafını ayrı ayrı, kendi başına değerlendirilebilen operatörler olarak vermek; şemalar bunları
  istediği sırada ve katsayıyla çağırır. Bugünkü kodda attention ve FFN tek blokta iç içe; şema değiştirmek mümkün değil.
- **KULLANIM:** `op(u, ctx) -> du` (artış; durumu değiştirmez). ctx: konum, maske. A ve F kendi P'sini uygular (pre-norm).
- **ÖĞRENİLME:** her operatörün kendi ağırlığı; artış ölçeği şemanın Δt katsayısıyla çarpılır (ağırlık başlangıcı buna göre).
- **OKUNMA:** her operatörün artış normu ‖du‖ / ‖u‖ adım adım kaydedilebilir (hangi operatör ne kadar iş yapıyor).
- **YÖNETİM:** test: A nedensel (gelecek token değişince geçmiş çıktı aynı), A = elle yazılmış integral formülü (fp32),
  F konum başına (bir token değişince yalnız onun çıktısı değişir), P yarıçap kısıtını sağlar.

### Parça 2 — şemalar

    lie       u ← u + Δt·A(u);  u ← u + Δt·F(u)                         1. derece (bugünkü)
    strang    u ← u + ½Δt·F₁(u);  u ← u + Δt·A(u);  u ← u + ½Δt·F₂(u)    2. derece (Macaron Net 2019)
    parallel  u ← u + Δt·(A(u) + F(u))                                    makalenin denklem 75'i (GPT-J / PaLM)

Eşit parametre: strang'de F₁ ve F₂ yarım genişlik.

**Uyarı (HİPOTEZ):** 2. derece doğruluk yalnız θ(t) zamanda pürüzsüzse geçerli. Her adımın bağımsız ağırlığı varsa
"Strang" sadece başka bir sıra olur; asıl avantaj pürüzsüz θ(t) (Parça 3b) ile gelir.

### Ölçülenler: eğitilmiş transformer denklemi nasıl çözüyor

ÖLÇÜLDÜ (`step_diagnostics.py`, `commute_probe.py`; kıyas transformer'ı `v2_tf_fw_d768_l10_muon_lr2e-3_20261007_150554`, sınav
satırları, CPU fp32):

- adım küçük değil: ‖du‖ / ‖u‖ her katmanda ~0,3; ardışık katman çekirdekleri γ_l, γ_{l+1} çok farklı (en yakın head TV ~0,55,
  dondurulmuş çekirdek hatası %50–90). Pürüzsüz θ(t) varsayımı (Parça 3b) zayıf: ağ ODE'yi küçük adımla çözmüyor.
- artışlar düşük boyutlu: %90 enerji için dA 1 / 58 / 96 / 26, dF 152 / 184 / 52 / 77 boyut (d 768; katman 1 / 3 / 6 / 10).
- ardışık katmanlar yer değiştirir mi (l ve l+1 aynı girdiden, kayıp artışı; taban 3,1103): 1+2 **+2,03**, 2+3 **+0,53**,
  3+4 +0,08, 4+5 +0,11, 5+6 +0,14, 6+7 +0,13, 7+8 +0,07, 8+9 +0,05, 9+10 +0,14.
- HESAP / rastgele ağırlık (`tests_model_s.py stages`): sıralı − paralel fark / ε² sabit (0,0082): fark komütatörden, 2. derece.

- aşamalar, SIFIR ATIŞ (eğitilmiş ağırlıklar yeniden düzenlenir, yeniden eğitim yok; taban 3,1103; `--groups`):

  | aşamalar (katman sayısı) | sıralı derinlik | sum | wide |
  |---|---|---|---|
  | 1×8, 2 | 9 | +0,142 | +0,117 |
  | 1,1,2,2,2,2 | 6 | +0,533 | +0,442 |
  | 1×6, 4 | 7 | +0,784 | +0,790 |
  | 1,1,1,1,3,3 | 6 | +0,798 | +0,729 |
  | 1,2,1,2,2,2 | 6 | +1,095 | +0,800 |
  | 1,1,4,4 | 4 | +1,531 | +1,578 |
  | 2,2,2,2,2 (ilk çift de) | 5 | +3,08 | +6,19 |
  | 1,1,8 | 3 | +3,48 | +4,03 |

  Cezalar aşağı yukarı toplanıyor (çift başına ~0,1); ilk çifti birleştirmek felaket. Sıfır atış eğitimde ne olacağının
  ÜST sınırı (ağ birbirini görmeden öğrenmeye hiç fırsat bulmadı); asıl soru eğitilerek ölçülür (E1).
- token başına uyarlamalı adım (ODE adım denetimi; katman ≥ 3'te en küçük artışlı %p token'ın artışı atlanır; `--skip`):
  F %25: oracle +0,32, önceki katmana göre +0,49, rastgele +0,56; A %25: +0,23 / +0,42 / +0,57; F %50: +0,89 / +1,33 / +1,44.
  Artış boyu atlanabilir token'ı zayıf seçiyor (oracle rastgeleden yalnız ~%40 iyi); uyarlamalı adım burada güçlü kaldıraç değil.
- yerellik (γ kütlesinin son w token'daki payı): katman 3–6'da w 4 %25–36, w 64 %68–81; katman 1 dağınık (w 64 %21, ilk
  token %17), 9–10'da w 64 %53. A yakın alan değil: kısa evrişimle değiştirilemez.

Okuma: **sıra yalnız baştaki 2–3 adımda önemli**; sonrası neredeyse yer değiştiriyor. Zaman adımını küçültmek (θ(t), strang)
yerine sıralı derinliği komütatörün büyük olduğu yere koymak.

### Parça 3 — hibrit aşamalar (makale Ek C)

    aşama:  u ← u + Σ_{i ∈ aşama} (S_i(u) − u)        (S_i: lie adımı; hepsi aynı u'dan)

= ardışık k adımın BCH açılımının ilk terimi; atılan ½ Σ_{i<j} [D_j, D_i]. Aşama tek GPU işlemine birleştirilebilir:
**k kat geniş tek adım** (k·heads attention, k·hidden FFN; FFN'ler bütün A'ları görür: `wide`). Yani hibrit şema =
sıralı derinliği genişliğe çevirmek; FLOP aynı, sıralı adım sayısı azalır.

- **AMAÇ:** aynı FLOP'la daha az sıralı adım: daha büyük matmul (GPU verimi), daha az çekirdek çağrısı.
- **KULLANIM:** `stages(d, heads, hidden, groups)` (formül); GPU biçimi `Step(width=k)` / `model.ModelS(d, widths, heads)`:
  widths ör. (1, 1, 1, 1, 2, 2, 2, 2) = 12 adımlık FLOP, 8 sıralı adım.
- **ÖĞRENİLME:** aynı kayıp; ağ aşama içindeki adımları birbirinin çıktısını görmeden öğrenir.
- **OKUNMA:** aşama içi her adımın artışı ayrı ölçülebilir; aşamalar arası komütatör (`commute_probe`).
- **YÖNETİM:** test: k = 1 aşamalar = sıralı adımlar (bit düzeyinde); k = 2 = u + D₁ + D₂; fark ∝ ε².

### Parça 3b — θ(t) (ertelendi)

    pürüzsüz   θ(t) = Σ_k c_k(t)·Θ_k

Ölçüm (yukarıda) adımların büyük ve çekirdeklerin hızla değiştiğini gösteriyor; pürüzsüzlük dayatmanın kaliteye mal olması
beklenir (TAHMİN). Strang / yüksek dereceli şemalar da bu yüzden ertelendi.

### Parça 6 — tek operatör M (bölmesiz adım)

Kullanıcı, 11 Ekim: *"işin matematiğini hızlandıran kolaylaştıran basitleştiren ya da farklı bir bakış açısıyla değerlendiren bir
yöntem. mevcut literatür, yapılan transformer mimarisi insanların zamanla biriktirdiği bir hantallık da olabilir."*

F, çekirdeği birim matris olan bir A'dır: A(u) + F(u) = Σ_x' [γ(x, x') W_ov + δ(x, x') F] u(x'). Bölme yalnız iki operatör ayrı
tutulduğu için gerekiyor. Tek operatör:

    z = W_in P(u) = [q | k | v_g v_u | g up]
    u ← u + W_out [ Σ_x' γ(x, x') silu(v_g) v_u (x')  |  silu(g) up ]

- **AMAÇ:** katman başına 1 izdüşüm, 2 matmul, 1 attention (bugün 2 izdüşüm, 4 matmul); bölme hatası yok; attention doğrusal
  olmayan özellikleri karıştırır.
- **KULLANIM:** `operators.MixOp`; `--strata_op mix` (hepsi) ya da adım başına liste (`split,mix,...,split`); `parallel` =
  doğrusal değerler (u + A(Pu) + F(Pu), kontrol).
- **ÖĞRENİLME:** aynı kayıp; eşit matris parametresi için yerel genişlik 7d/3 (d 1024: 2.368; 64'e aşağı yuvarlanır, %0,7 az).
- **OKUNMA:** yerel ve attention kanallarının W_out'taki payı; katman içi kayıp artışı (`commute_probe --inner`).
- **YÖNETİM:** test: linear = parallel adım (aynı ağırlık), swiglu = elle formül, nedensel, KV cache = tam ileri, toplu üretim.

ÖLÇÜLDÜ (sıfır atış, d768 transformer, `--inner`; A ile F aynı girdiden): katman 1 +2,83, 2 +0,12, 3–9 +0,02…+0,06,
10 +0,72; katman ≥ 3 birlikte +2,21. İlk ve son adımda F attention çıktısına muhtaç; ara adımlarda sıra az önemli →
aday düzen `split,mix,…,mix,split`.

### Parça 7 — izdüşüm katlama (makale §2.4, Th. 3.1)

LN = kısıt kümesine izdüşüm = satır başına tek ölçek → doğrusal haritayla yer değiştirir: P(u) W = s(u) (u diag(g) W),
s = 1/rms(u). Normlu kopya yazılmaz; QK izdüşümü ölçekten bağımsız olduğundan q, k'de s düşer; v / yerel kanallarda s SwiGLU'ya
katılır. Fonksiyon aynı (test: çıkış ve gradyan fp32 ~1e-6). `PROJ_FOLD` varsayılan kapalı; hız `bench_step.py` ile GPU'da.
Profil (Model Z d1024, OLCULENLER): normlar adımın ~%11'i (HESAP).

### Parça 8 — q / k boyu (makale §5: head = sürekli h'nin kuadratürü)

Head sayısı FLOP'u değiştirmez; kaldıraç çekirdeğin ⟨q, k⟩ boyu. ÖLÇÜLDÜ (sıfır atış, `--qk_rank`): head başına ilk r ana
bileşen, r 48 +0,005, 32 +0,056 (katman ≥ 3 +0,037), 16 +0,40. `qk_dim` (q, k, QK izdüşümü, RoPE qk boyunda; v head boyunda);
`--strata_qk`. HESAP d1024 / T 2048, qk 32: katman işinin ~%9'u.

### Parça 9 — gate-önce FFN (makale §3.3.3: aktivasyon = izdüşüm)

Kullanıcı, 11 Ekim: *"Swiglu yerine relu kullanmak değil kendi hızlı uygun gördüğünü tasarlamak daha hızlı ve entegre hatta daha
doğru"*. silu(g) ≈ 0 ise birim up ne olursa olsun katkısız → gate kendi yönlendiricisi: önce gate (FFN'nin 1/3'ü), token başına
|silu(g)|² enerjisi en yüksek blokların up / down'ı. ÖLÇÜLDÜ (`--ffn`): enerjinin %90'ı hidden'in %6–17'sinde; sıfır atış
blok seyreklik (`--gate_sparse`, bloklar keyfi) %50 gate +0,37 / oracle +0,24 / rastgele +2,82. Şimdilik maskeyle (kalite);
kazanç gruplu matmul çekirdeğiyle (keep 0,25: FFN 0,5, katman ~%25 az; HESAP). `--strata_ffn_keep`; ilk / son adım yoğun.

## 4. Deneyler (sırası)

| | soru | koşular (aynı FLOP: 12 birim) |
|---|---|---|
| E1 | genişliğe çevrilen derinlik aynı kaliteyi verir mi, ne kadar hızlı | `1×12` (= transformer), `1,1,1,1,2,2,2,2`, `1,1,2,2,2,2,2` |
| E1k | komütatör ilkesi: genişliği SONA koymak mı doğru | `2,2,2,2,1,1,1,1` (ters; E1'den kötüyse ilke desteklenir) |
| E2 | tek operatör (FP8 kapalı: MixOp matrisleri FP8 listesinde değil, kıyas eşit) | `split` 1×12, `mix` hepsi, `split,mix×10,split`, `parallel` hepsi |
| E3 | q / k boyu ve gate-önce FFN (kalite; FP8 kapalı) | `--strata_qk 32`, `--strata_ffn_keep 0.5`, `0.25` |

**E1 sonucu (ÖLÇÜLDÜ, d1024, 3.000 adım, FP8):** `1×12` sınav 3,2999 / 374,2 ms; `1,1,1,1,2,2,2,2` 3,3085 / 364,0 ms (+0,009,
−%2,7: aynı kayba ~%1, gürültü içinde). Ters kontrol `2,2,2,2,1,1,1,1` adım 1.500'e kadar sondakinden iyi ya da eşit: **sıfır atış
komütatör eğitim sonucunu öngörmedi**; düzen seçimi bu ilkeyle yapılmaz. Aynı iş yüklü iki genişlik kolunda ms farkı ~%3:
koşudan koşuya derleme oynaklığı; ~%3 altı hız farkı tek koşuyla okunmaz (bench aynı süreçte).

Ortak: s000 (FineWeb-Edu), d1024, heads 16 (head 64), GQA yok, tam WSD kısa koşu (`--steps`), aynı tohum, Z2 tarifi
(NorMuon, lr auto, FP8 MLP). Ölçüt: sınav kaybı + ms/adım + aynı kayba süre. Strang / θ(t) / donmuş γ ölçümle ertelendi
(adımlar büyük, çekirdekler hızlı değişiyor).

## 5. Kapsam dışı (şimdilik)

Z, cümle düzeni, G yerleşimi: Model S tam nedensel attention ile başlar (kullanıcı: "Z ya da trasformer mimarisine takılma").
Denklem çözücüsü oturursa Z ve yerel / genel ayrımı A operatörünün çekirdek seçimi olarak geri eklenebilir.
