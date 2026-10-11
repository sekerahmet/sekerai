# 01 — Strata-x kod denetimi + "açık yerel kapasitede" çıkarımına saldırı (11 Ekim 2026)

Kullanıcı, 11 Ekim 2026: *"Sanki bi danışman hem kodu inceleyen hem fikirlerini çürüten hem yeni fikir öneren bir arkadaş olan
sana böyle biri daha iyi olmaz mı ? Kodları da gözden geçirir"*. Bu tur yalnız iş 1 (kod) ve iş 2 (çürütme); yeni fikir yok.

Okunan: `model_s/TASARIM.md` (N1–N5), `HAYAL.md`, `strata_x.py`, `output.py`, `model.py`, `operators.py`, `integrator.py`,
`tests_model_s.py` (t_strata_x, t_train), dal `deneme/model-s` çalışma ağacı (`wt_s`) `deneme2/v2/common/train.py`
(_build, _step, _exam, _readings, set_order, kimlik), `recipe.py` (param_groups, muon_params), `metrics.exam_scores`,
OLCULENLER_z.md satır 1069–1075. Betikler (CPU, < 1 dk): `scratchpad/danisman/sx_audit1.py`, `sx_params.py`.

## Özet — en ciddi üç madde

1. **ORTA (ölçü temizliği):** `train.py` modeli hiçbir yerde `eval()`'e almıyor; sınav ve okuma eğitim modunda koşuyor.
   `--sx_rate > 0`'da `info_ema` (θ'nın kaynağı) **sınav batch'leriyle ve üretilen metinle güncelleniyor**: sınav kendi
   içinde batch sırasına bağlı, sınav verisi eğitim durumuna sızıyor. ÖLÇÜLDÜ (`sx_audit1.py` [1]).
2. **ORTA (çıkarımı geçersiz kılar):** E kolu ("token 512 + yerel 256") token temsilini büyütmüyor: çıkış başının girdisi
   `tok_out` ile 256 boyutlu alt uzayda kalıyor (rank = d_local), girişte de `tok_in` rank 256. "Açık token boyundan değil"
   hükmü bu kolla verilemez. ÖLÇÜLDÜ (`sx_audit1.py` [4]: h 48 boyutlu, rank 16 = d_local).
3. **ORTA (sessiz dtype):** bf16 autocast altında büyük çözücünün artık akışı (residual) **bf16** (z = `up(...)` Linear
   çıktısı); `d_token != d_local` kolunda yerel kodlayıcı + kod çözücü akışı da bf16. Transformer tabanında akış fp32
   (E çıktısı). ÖLÇÜLDÜ (`sx_audit1.py` [3]); kalite etkisi ŞÜPHE (ölçülmedi).

## İŞ 1 — kod denetimi

### K1. Sınav / okuma eğitim modunda; rate'li koşuda θ sınavla kayıyor — ORTA
- `train.py` (wt_s) 568–572 `_exam`, 575 `_readings`, 1129/1153 çağrılar: `model.eval()` yok (grep: train.py, metrics.py,
  generate_readings.py'de `eval()` / `.train(` hiç geçmiyor). Transformer / Model Z'de zararsız (dropout yok), Strata-x'te değil:
- `strata_x.py:151` `if self.rate > 0 and self.training` → her sınav batch'i ve `generate`'in her adımı (`_step` →
  `forward_states`) `info_ema`'yı %1 iter. `strata_x.py:182` pred_next yardımcı kaybı ve `:196` `_aux` sınavda da hesaplanıyor
  (yalnız boş iş).
- Senaryo: C kolu (B + oran 3) sınavında k. batch'in θ'sı 1..k-1. sınav batch'lerinin sürprizinden; çok epoklu koşuda epok sonu
  sınavından sonra eğitim kirli θ ile sürer; okuma (açgözlü metin, düşük sürpriz) sırasında θ düşer → üretimde eğitimdekinden
  fazla birim. `gap_v2.py:50` `eval()` çağırıyor → aynı checkpoint'te train.py sınavı ile gap_v2 farklı sayı verebilir.
- KANIT `sx_audit1.py` [1] (küçük rastgele model, rate 3): info_ema 11,032 → 11,099 → 11,166 (iki sınav çağrısı); aynı batch iki
  kez: token nll farkı en çok 0,11, ortalama 17,7825 / 17,7864; bir `generate` sonrası 11,404; `eval()`'de fark 0, ema sabit.
  Eğitilmiş SS modelinde kayma büyüklüğü TAHMİN: küçük (sınav ve eğitim sürprizi yakın), ama kural ihlali ("OLCU TEMIZ").
- Düzeltme: `_exam` / `_readings` / `continuation_exam` başında `model.eval()`, sonunda `model.train()`; ya da
  `strata_x.py:151` koşuluna `torch.is_grad_enabled()` ekle. Test: train.py entegrasyonunda `--sx_rate` ile sınav öncesi /
  sonrası `info_ema` eşit (kural 6: modelin kendi testi `mr.eval()` çağırdığı için bunu göremedi).

### K2. bf16 artık akışı — ORTA (dtype ÖLÇÜLDÜ, etki ŞÜPHE)
- `strata_x.py:167` `z = self.up(...)` autocast'te bf16 → `integrator.Step.forward` `u + dt*A` bf16 + bf16 = bf16; 8 (SS) / 12
  büyük çözücü katmanı bf16 akışta toplanır. `:140` `x = self.tok_in(x)` → `d_token` kolunda kodlayıcı ve (`:188` he + ctx)
  kod çözücü de bf16. `sx_audit1.py` [3]: d_token yok → blok 0 / 3 fp32, 1–2 (büyük) bf16; d_token 48 → dördü de bf16.
- Neden önemli: Strata-x ile transformer arasındaki her kalite farkına karışır; E kolu ile A kolu arasında da fark yaratır.
- Düzeltme: `z = self.up(...).float()` ve `x = self.tok_in(x).float()` (iki satır). Etki ölçümü: A kolu 2.000 adım, tek fark bu.

### K3. E kolu = düşük ranklı embedding, "token 512" değil — ORTA (İŞ 2'de ayrıntı)
- `strata_x.py:69–70, 139–144, 191`: logit = (E·W_out)·x_256; çıkış dağılımının rank'ı d_local ile sınırlı (softmax darboğazı
  değişmedi). `tok_in` / `tok_out` blok dışında → AdamW'de (`recipe.muon_params` yalnız `blocks.`), artı K2'deki bf16.

### D1. Teşhis eksikleri — DÜŞÜK, ama İŞ 2'yi kilitliyor
- `strata_x.py:197` `last_units` hesaplanıyor, train.py hiç yazmıyor (grep: `last_units` / `units` yok). Yardımcı CE değeri
  (`aux`) `output_loss` `:212`'de değer olarak atılıyor, günlükte yok. Kollar arası birim / token ve kodlayıcı CE bilinmiyor.
- Düzeltme: pencere günlüğüne `units_per_token`, `aux_ce` (ikisi de tek skaler).

### D2. Hız / ölçü notları — DÜŞÜK
- Senkronlar: `:147–148` `nxt.nonzero()` iki kez, `:155` `int(...max())`, `:156` `ends.nonzero`, `:184` `pairs.any()`, `:197`
  iki `.item()` → ileri başına ~7 ana makine senkronu (GPU kuyruğu boşalır). `:155`'teki zorunlu; diğerleri birleştirilebilir.
- `train.py:441` her adım `_attn(...)` belge BlockMask'i kuruluyor, `StrataX._batch_hidden` (`:194`) kullanmıyor: ms/adım'a
  giriyor (transformer de kurduğu için kıyas adil; mutlak Strata süresi şişik).
- ŞÜPHE (GPU'da sınanmalı): global bloklar `dynamic=False`; her yeni `Umax` kovası (128'in katı) × grad / no_grad ayrı derleme.
  Eğitimin başında birim / token hızla düştüğü için yeni kova = yeniden derleme penceresi; "öğrendikçe ucuzlama" (adım 200
  199 ms → 118 ms, N3) kısmen derleme süresi olabilir. Test: `TORCH_LOGS=recompiles` ya da adım başına `Umax` günlüğü; ucuz.

### Doğrulandı (hata bulunmadı)
- Nedensellik: 5 yapılandırma (taban, plan 3, rate 2, d_token 48, plan 2 + next 0,5 + rate 1,5 + d_token 40) × 13 kesme noktası,
  iki belge + dolgu satırı: kesmeden önceki durum ve sınırlarda fark **0** (ÖLÇÜLDÜ, `sx_audit1.py` [5], eval modu, CPU yoğun
  yol). Kod okuması: i_t yalnız x_≤t (`:145–148`), sınır C_t'den (`:122–132`), kc = t'de biten son birim (`:157, :173`),
  `ok` belge eşitliği (`:174`), yuva t − son sınır (`:179`), pred_next hedefi kesik (`:186`). FLEX mod fonksiyonları (`:99–100,
  :112–113`) yoğun maskeyle aynı ifade; GPU eşitliği `sx_flex_check.py`'de.
- Sınav tanımı: `loss_per_target` (logprobs + cross_entropy) = eğitim ana CE (`_nll`): fp32 en büyük fark 3,4e-5 (≈17 nat'ta);
  logprobs normlu (Σp − 1 ≤ 4,8e-7). Yardımcı CE sınava girmiyor (eval'de `_aux` None; train modunda da `loss_per_target`
  kullanmıyor). ÖLÇÜLDÜ [2].
- Sürdürme: `info_ema`, `out.order`, `out.rank` kalıcı buffer → `model.state_dict()` → checkpoint; `set_order` (train.py 920)
  yüklemeden önce, checkpoint üzerine yazar (aynı sayımlar). Eksik test: `--sx_rate` ile kesilip sürdürülen = kesintisiz
  (t_train strata_x rate'siz).

### Eksik kritik testler
1. train.py yoluyla sınav: `info_ema` sınavdan önce = sonra; aynı batch iki kez = aynı nll (K1'i yakalardı).
2. CPU autocast'te büyük çözücü akışı fp32 (K2).
3. `--sx_rate` sürdürme bit eşitliği.

## İŞ 2 — "Açık token temsilinden değil yerel kapasiteden; yerelde genişlik derinlikten iyi"

Kollar (HESAP, `sx_params.py`; büyük d512 / 8 katman, θ 4; çarpma = token başı ileri, yerel pencere 256 dahil):

| kol | sınav | toplam | E | yerel | yerel çarpma | baş çarpma | arayüz rank | Muon açı / d512 |
|---|---|---|---|---|---|---|---|---|
| A yerel 256 (2+2) | 1,672 | 41,6M | 12,9M | 3,2M | 3,7M | 2,1M | 256 | 0,71 |
| E token 512 + yerel 256 | 1,674 | 55,0M | 25,7M | 3,2M | 3,7M | 4,2M | **256** | 0,71 |
| yerel 256 × (4+4) | 1,627 | 44,7M | 12,9M | 6,3M | 7,4M | 2,1M | 256 | 0,71 |
| yerel 384 (2+2) | 1,604 | 52,2M | 19,3M | 7,1M | 7,9M | 3,1M | 384 | 0,87 |
| B yerel 512 (2+2) | 1,557 | 64,4M | 25,7M | 12,6M | 13,7M | 4,2M | 512 | 1,00 |

**Dayanan kısım:** yerel tarafı (genişlik + arayüz + token boyu birlikte) büyütmek açığı sırayla kapatıyor (A → 384 → B:
−0,07, −0,12); tek tohum ama farklar E1'deki kol gürültüsünün (~0,003–0,009) çok üstünde. Dayanmayan iki alt hüküm:

**S1. "Token temsilinden değil" — E kolu bunu sınamıyor.** K3: E'de çıkış rank'ı ve giriş rank'ı 256 kaldı; değişen yalnız E'nin
parametrelemesi (düşük ranklı, AdamW, bf16 akış). E ≈ A beklenen sonuç, bilgi taşımıyor. Çürütme testi (kod hazır, sıfır
satır): **`--sx_local 512,8,2,2 --sx_token 256`** (51,5M, arayüz 512, token / baş 256), 2.000 adım. ≈1,557 → token temsili
gerçekten önemsiz; ≈1,67 → açığın büyük kısmı token / çıkış tarafında; arada → ikisi.

**S2. Arayüz bant genişliği karıştırıcısı.** Büyük çözücü birimi tek bir d_local vektörden görür (`up(he[son token])`, `:167`), kod
çözücü büyük çözücüyü tek bir d_local vektörle ve yalnız girişte görür (`down`, `:172, :188`). Genişlik kolları bu iki kanalı da
genişletiyor, derinlik kolu genişletmiyor: sıralama 512 > 384 > 256×8 > 256, "yerel kapasite" ile "arayüz" hipotezlerinin ikisine
de uyuyor (HİPOTEZ). Ucuz ayırıcılar: (a) eğitim yok — mevcut A ve B checkpoint'lerinde sınav batch'inde ctx → g0 kapatma Δ
kaybı (CPU, dakikalar): B'nin Δ'sı A'nınkinden çok büyükse kazancın bir kısmı kanaldan; (b) yerel 512 + up / down rank 256
darboğazı (iki satır), 2.000 adım: ≈1,557 → yerel kapasite; ≈1,67 → arayüz.

**S3. lr / Muon karıştırıcısı.** lr global d'den (train.py 843; SS'te açık 2e-3) ve yerel bloklar da Muon'da aynı lr'yi alıyor.
Projenin kendi kuralı (açı / adım ∼ lr·0,2·√d, LR_REF: en iyi lr ∝ 1/√d) doğruysa yerel 256 blokları kendi boylarına göre √2
düşük adımla, 384 1,15 düşük, 512 tam adımla eğitildi (HESAP; kuralın yerel bloklara taşınması HİPOTEZ). 2.000 adımlık WSD'de
eksik lr kayıp yazar ve yönü genişliği kayırır. Test: A'da yalnız yerel bloklara lr × √2 (tek param grubu çarpanı), 2.000 adım.

**S4. "Genişlik derinlikten iyi" kıyası dört şeyi birden değiştiriyor.** 384 ile 256×(4+4): yerel token başı çarpma eşit (7,9 /
7,4M) ama toplam parametre +7,5M (hepsi E'de, çünkü d_token = d_local), baş çarpması 3,1 / 2,1M, arayüz 384 / 256, Muon açısı
0,87 / 0,71. Fark (0,024) bunların herhangi birinden gelebilir. Test (kod hazır): **`--sx_local 384,... --sx_token 256`**; S1
koluyla birlikte E ve baş eşitlenmiş bir genişlik / derinlik kıyası verir.

**S5. Birim / token oranı kola göre kayıyor (ölçülmüyor).** θ sabit 4: kodlayıcı iyileşince sürprizi düşer → daha az birim →
büyük çözücü token başına daha az çalışır. Derinlik kolunda kodlayıcı 4 katman: birimler muhtemelen en çok o kolda azaldı
(TAHMİN) → derinlik kolu büyük çözücüden daha az yararlanmış olabilir. `last_units` günlükte yok (D1). Test (eğitim yok): her
kolun agent.pt'sinde sınav batch'inde birim / token; fark büyükse kıyas `--sx_rate` ile (birim / token sabit) tekrarlanır.

**S6. Yardımcı CE ağırlığı 1,0.** Kodlayıcı aynı zamanda tam ağırlıklı bir LM; kod çözücünün girdisi bu LM'e hazırlanmış `he`.
4+4 kolunda yerel katmanların yarısı bu kaybın altında. aux CE değeri günlükte yok (D1). Test (eğitim yok): kollarda sınav
batch'inde kodlayıcı CE'si ve kod çözücü CE'si yan yana; kodlayıcı CE'si derinlik kolunda belirgin düşükse kapasite oraya akmış.

### Önerilen sıra (ucuzdan pahalıya)
1. Eğitimsiz, mevcut checkpoint'ler (CPU): birim / token, kodlayıcı CE'si, ctx kapatma Δ — A, B, 384, 256×(4+4) (S2a, S5, S6).
2. Kod değişikliği yok, 2 × 2.000 adım: `512 / token 256` (S1) ve `384 / token 256` (S4).
3. Küçük kod: K1 + K2 + D1 düzeltmeleri, sonra yerel lr × √2 (S3) ve rank 256 darboğazı (S2b).
