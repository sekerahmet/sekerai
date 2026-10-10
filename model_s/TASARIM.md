# Model S — transformer'ı bir denklemin çözücüsü olarak sıfırdan kurmak

Kullanıcı, 11 Ekim 2026: *"bu denklem üzerinde yoğunlaş Z ya da trasformer mimarisine takılma. bu matematik herşeyin anahtarı gibi
düşün"*; *"istersen modeline bir isim seç. her parçayı ffn, attention vs sıfırdan yazıyorsun gibi düşün. yeni klasör aç acele etme.
parçaları istersen teker teker yaz."*

Kaynak: Tai, Liu, Li, Chan, *A Mathematical Explanation of Transformers* (arXiv 2510.03989; `belge/makaleler/2025/tai2025_math_transformer.txt`).
Ad: **S** = splitting (operatör bölme) / sürekli (zaman).  Klasör: `C:\AI_NEW_MODEL\model_s` (kullanıcı, 11 Ekim: "C:\AI_NEW_MODEL bunun altında model S olarak klasör aç"); veri / eğitim ortak kodu `deneme2/v2/common`.

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
| 3 | θ(t): adımlara göre parametre (bağımsız / pürüzsüz) | `integrator.py` | sırada |
| 4 | model: başlangıç koşulu, çözücü, okuma (çıktı) | `model.py` | sırada |
| 5 | eğitim / sınav bağlantısı (`--model model_s`) | `deneme2/v2/common/train.py` | sırada |

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
"Strang" sadece başka bir sıra olur; asıl avantaj Parça 3 ile birlikte gelir.

### Parça 3 — θ(t)

    bağımsız   θ_n her adımda ayrı (bugünkü)
    pürüzsüz   θ(t) = Σ_k c_k(t)·Θ_k   (az sayıda taban ağırlık, zamanla değişen katsayılar)

Pürüzsüz θ(t): denklem gerçekten sürekli olur, yüksek dereceli şema anlam kazanır ve adım sayısı parametreden bağımsızlaşır
(daha az adımla aynı θ(t) çözülebilir: hız).

## 4. Deneyler (sırası)

| | soru | kıyas |
|---|---|---|
| E1 | strang = lie mi (aynı parametre, aynı adım) | Model S lie (= transformer) ile strang |
| E2 | daha az adımla aynı kalite | strang L9 ile lie L12 |
| E3 | pürüzsüz θ(t) + strang, adım azaltma | E2 + θ(t) |
| E4 | donmuş γ (attention çekirdeği iki adımda bir) | E1 tabanına karşı |

Taban: Model S lie, aynı veri / plan / tohum (s000, d1024 L12). Ölçüt: sınav kaybı + ms/adım + aynı kayba süre.

## 5. Kapsam dışı (şimdilik)

Z, cümle düzeni, G yerleşimi: Model S tam nedensel attention ile başlar (kullanıcı: "Z ya da trasformer mimarisine takılma").
Denklem çözücüsü oturursa Z ve yerel / genel ayrımı A operatörünün çekirdek seçimi olarak geri eklenebilir.
