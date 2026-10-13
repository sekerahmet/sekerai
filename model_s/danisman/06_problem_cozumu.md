# 06 — TERSINE §10 probleminin çözümü (11 Ekim 2026, mathematician)

Kural: önce hesap ve kanıt; bileşenler amaçtan türer, sonradan parça eklenmez. Etiketler: KANIT (türetme), HESAP (varsayımıyla),
ÖLÇÜLDÜ (önceki turlardan), LİTERATÜR (hafızadan olanlar ayrıca işaretli), HİPOTEZ / TAHMİN yalnız boşluğun adı olarak.
Gösterim: R_t = (σ_t, {m_j}_{j<t}), t anında elde olan her şey; ℓ_t = −log p(x_{t+1} | R_t).

## 0. Sonuç önce

1. **Çözüm bilinen bir aile çıkıyor; attention ilkesel olarak doğru.** Ama yalnız bir işi için: keskin okuma, arşiv üzerinden.
   - Amaçtan türeyen yapı "durum + arşiv" karması.
   - Softmax attention, arşivden okumanın kesin çözümü (Gibbs varyasyon ilkesi ve Bayes sonsalı, §5). Saf transformer λ_m → 0 sınırı,
     sabit durumlu model (SSM) λ_m → ∞ sınırı; arada karma (§5).
   - Transformer'ın ilkesel olarak fazla ödediği iki şey var:
     - (a) her token'ı tam genişlikte arşivlemek (16 kbit / katman). Su doldurma birkaç on bit diyor (§4).
     - (b) geniş ortalamayı her adımda tarayarak yeniden hesaplamak. Durumda tutmak d_h/N kadar ucuz (§3).
2. **Erit / arşivle kuralı iki eşikli.** İkisi de aynı Lagrange fonksiyonunun KKT koşulu (§3–4):
   - (i) **bit eşiği:** tedarik Fisher tayfında su doldurma. Hangi konumdan kaç bit arşivleneceğini bu verir.
   - (ii) **keskinlik eşiği:** okuma ağırlıklarının sapması σ_s² ≤ 2ε ise bilgi duruma erir (doğrusal okuma yeterli); değilse arşivlenir.
   - Problem metninde bir hata var: durum boyu ücretlendirilmezse çözüm önemsizleşiyor (§1).
3. **Ölçümlerden öngörülen yapı (§8):**
   - %76 geniş kütle → sabit durum (delta / doğrusal).
   - ~%8 yakın sıra → durumdaki FIFO pencere.
   - ~%3 birebir → kesin anahtarlı arşiv, O(1).
   - Kalan keskin okuma → b bitlik notlu arşivde softmax.
   - Tedarik %42 ve düz → token başına sabit b.
   - Tek bilinmeyen: head başına keskinlik (σ_s) dağılımı. Checkpoint'te, CPU'da ölçülür.

## 1. Problemin düzeltilmesi (KANIT)

- **(a) λ_c = 0 iken durum / arşiv ayrımı yok olur.** Her m_j geçmişin bir fonksiyonu; σ_t'ye katılabilir. O halde her çözümün yalnız
  durumlu bir eşdeğeri var. Ayrım tamamen hesap maliyetinden doğuyor.
- **(b) Amaç yalnız arşiv bitini ücretlendiriyor.** σ_t = (x_1..x_t), arşivsiz ve bedava bir çözüm. Bu yüzden durum boyu da ücretlendirilmeli:
  c_U ∝ |σ| (her adım güncelleme + okuma), ya da λ_m durum bitine de uygulanmalı. Aşağıda c_U = κ_S·|σ| / adım alınıyor.

## 2. Bilgi kuramsal biçim (KANIT)

- R_t, X_≤t'nin bir fonksiyonuysa en iyi tahmincide:
  E[ℓ_t] = H(X_{t+1} | X_≤t) + I(X_{t+1}; X_≤t | R_t).
  İlk terim entropi oranı: indirilemez. İkincisi bellek eksikliğinin bedeli.
- Bütün gelecek için gereken bellek öngörüsel bilgi darboğazı: min I(X_≤t; R_t) − β·I(R_t; X_>t).
  - β → ∞'da çözüm nedensel durum: geleceği aynı biçimde dağıtan geçmişlerin denklik sınıfı.
  - LİTERATÜR, hafızadan, klasörde aranmalı: Crutchfield; Bialek–Nemenman–Tishby 2001; Still 2014.
- Bu biçim neyin saklanacağını söyler, nasıl saklanacağını (durum mu arşiv mi) söylemez. "Nasıl" yalnız λ_c'den gelir (§1a).

## 3. Erit mi arşivle mi: maliyet koşulu (KANIT, doğrusal maliyet modelinde)

Bir bilgi parçası φ: b bit, j'de doğar, ömrü L adım, ömrü boyunca r kez okunur.

    durumda:  C_S = κ_S·b·L                       (her adım güncelleme + okuma; okuma bedava, maliyet ömre yayılı)
    arşivde:  C_A = b·(κ_w + λ_m) + r·(c_bul(N) + κ_R·b)
    arşivle  ⇔  κ_S·L  >  κ_w + λ_m + r·(c_bul/b + κ_R)

- **Okuma oranı ρ = r/L:** sık okunan (ρ ≈ 1) ve tarama gerektiren (c_bul = N) bilgi durumda kalmalı. Seyrek okunan, uzun ömürlü bilgi arşive.
- **Kapasite koşulu (KANIT, önceki turlar):**
  - Doğrusal durumda kesin geri çağırma head başına ≲ d_h–2d_h öğe.
  - Bir kesin kopya L·h bit ister (Jelassi'nin sınırının doğal metin hali).
  - Dolayısıyla keskin, öğeye özgü bilgi sayıca d_h'yi aşınca kuraldan bağımsız olarak arşive gitmek zorunda.
- **Keskinlik eşiği (KANIT + HESAP).**
  - s = q·k / √d_h için e^s = 1 + s + s²/2 + …
  - Doğrusal durum okuması (φ(q)·φ(k) = 1 + q·k) ilk iki terimi kesin verir. Bağıl ağırlık hatası ≈ E[s²]/2 = σ_s²/2.
    σ_s 0,3 → %4,5; 0,5 → %12,5; 1 → %50.
  - Kural: σ_s² ≤ 2ε ise bilgi duruma erir (geniş okuma doğrusalla kesin), değilse keskin okuma gerekir → arşiv.
  - ε seçimi: doğrusallaştırmanın kayıp artışı ½·Fisher·‖Δo‖² ile tarama tasarrufu λ_c·(4N·d_h − 4d_h²)'nin eşitlendiği nokta.
- **Sayı (HESAP; d768, d_h 64, N_eff 1.354):** geniş okuma için durum okuması 4d·d_h = 0,2M FLOP / token / katman, tarama 4Nd = 4,2M.
  Geniş bilgiyi taramak 21× pahalı.

## 4. Arşivin içeriği ve biti (KANIT, ikinci derece Taylor + Gauss oran–bozulma)

- Token j'nin arşivlenen notundaki e kadar bozulma, gelecek kaybı ≈ ½·eᵀF_j e kadar artırır. F_j = E[g_sup g_supᵀ], j'nin tedarik gradyanının kovaryansı.
- KKT (bit başına marjinal kayıp düşüşü = λ_m'ye bağlı fiyat θ): bileşen i'ye r_{j,i} = max(0, ½·log₂(λ_{j,i}/θ)) bit.
  λ_{j,i}, F_j^{1/2} Σ_h F_j^{1/2}'nin özdeğerleri. Not, özvektörlere izdüşüm.
- **"Hangi konumdan kaç bit":** b_j = Σ_i r_{j,i}.
  - Yalnız keskin okuma alt uzayındaki tayf θ'yı aşan konumlar arşive yazar. Geniş alt uzay §3 gereği duruma erir.
- **Tedarik düz → b_j ≈ sabit.** Tayf biçimi ve iz konumlar arasında benzerse ortak θ her konuma aynı biti verir. Token seçerek arşivleme
  (MoD / Strata-x birimleri) kazandırmaz (§20 ÖLÇÜLDÜ, KKT).
- **Büyüklük (HESAP, Gauss):** her bileşen ~2 bitte %93 doyar → b ≈ 2·n_ilgili. n_ilgili 16–64 ise 32–128 bit / head / token.
  Transformer 2·d_h·16 = 2.048 bit / head / token → 16–64× fazla.

## 5. Okuma: softmax'ın türetimi ve özel haller (KANIT)

- **Gibbs varyasyon ilkesi.** w* = argmax_{w∈Δ} Σ_j w_j s_j + H(w)/β ⇒ w*_j = softmax(β s)_j. Softmax, okumada en yüksek entropili seçim.
- **Bayes.** "Kaynak tek bir geçmiş konum; önsel tekdüze; olabilirlik ∝ e^{s_j}" modelinde sonsal p(j | q) = softmax ve o = E[v | q].
  Softmax attention bu varsayımlarla arşivden okumanın tam çözümü.
- **Özel haller:**

  | λ_m (arşiv biti) | λ_c (hesap) | çözüm |
  |---|---|---|
  | → 0 | küçük | her şey tam genişlikte arşiv + her adım tarama = **transformer** |
  | → ∞ | — | arşiv yok, yalnız durum = **SSM / doğrusal / delta** (öngörüsel bilgi darboğazının sıkıştırılmış nedensel durumu) |
  | sonlu | sonlu | **karma**: geniş + kısa ömürlü → durum, keskin + uzun ömürlü → b bitlik arşiv + softmax |

- **Açıkça:** çözüm bilinen bir mimari ailesi, durum + attention karması. Jelassi de karmaları öneriyor (`2024/jelassi2024_repeat.txt` L487);
  klasörde OLMo karması (`2026/ai2_2026_olmo_hybrid.txt`). Burada yeni olan mimari değil:
  - (i) iki eşikli kuralın amaçtan türetilmesi;
  - (ii) arşiv bitinin su doldurmayla (Fisher tayfı) belirlenmesi. Karmalar genellikle tam genişlikte KV saklıyor.
  - Bu ikisinin literatürde olup olmadığı web'de aranmalı (kullanıcı onayıyla).

## 6. Dört açık, aynı amaçtan

- **Birleştirme.** Log-kayıpta en iyi tahminci, (σ_t, o_t) verildiğinde Bayes tahmincisi.
  - Kanıtlar y verildiğinde koşullu bağımsızsa: log-doğrusal (uzman çarpımı). Kaynak tek bir konumsa (birebir kopya): karışım.
  - İkisi de 04 §2'de türetildi; ek parça değil, Bayes kuralının iki hali.
- **Katmanlar = sıçrama sayısı (KANIT, yapısal).** Tek okumada sorgu σ_t'den kurulur. Adresi başka bir arşiv öğesinin içeriğine bağlı bir öğe,
  o içerik önce okunup duruma yazılmadan bulunamaz.
  - Örnek: indüksiyon = "önceki token'ı oku → onunla eşleşeni bul" = 2 sıçrama.
  - Gereken derinlik ≥ okuma zincirinin uzunluğu. Her katman bir "oku → güncelle" turu.
- **Adres buluşması.** Anahtar yazma anında, gelecek sorgular bilinmeden kurulur.
  - En iyi anahtar K_j = argmax I(K_j; "j, gelecekteki bir sorgu için ilgili") (bilgi darboğazı, hedef ilgililik).
  - İlgililik yerel token'ların belirlenimci bir fonksiyonuysa (birebir tekrar) anahtar n-gram'ın kendisi: öğrenme gerektirmeyen özel hal,
    okuma O(1). Değilse vektör anahtar + softmax.
- **Yerel sıra.** Ömrü L ≤ w ve okuma oranı yüksek bilgide §3'ün koşulu durumu seçer. Kesinlik için durumun bir parçası FIFO kaydırma yazmacı
  olmalı: son w sembol, w·log₂V bit, kesin. Ayrı bir parça değil, durumun kesin bileşeni.

## 7. Token ne olmalı

- Sembol bir tamsayı. Amaçtan türeyen nesne, her sembolün iki etkisi:
  - (1) duruma katkısı Δσ_j (erime; doğrusal / delta durumunda bir güncelleme işlemi, ör. I − β k kᵀ);
  - (2) arşiv kaydı m_j = (anahtar, b_j bitlik not), keskin okuma alt uzayından.
- Transformer'ın d boyutlu h_j'si ikisini tek kapta tutuyor: hem geniş okumanın kaynağı hem tam genişlikte arşiv (K, V).
- d boyutlu vektör türetilmiş bir zorunluluk değil, bir kap. Türetilen boyutlar:
  - durum katkısı d_h × d_h durumu üzerinden;
  - arşiv notu b ≈ 32–128 bit / head.
- 08'deki operatör çürütmesiyle tutarlı: işlem olarak token yalnız durum (geniş) kısmı için doğru, keskin kopya için değil.

## 8. Ölçümlerle öngörülen yapı (HESAP; FineWeb transformer d768 / L10 ölçülerinden)

| okuma (ölçülen pay) | §3 / §4 kuralı | bileşen | maliyet / token / katman |
|---|---|---|---|
| geniş ortalama (~%76 kütle) | σ_s küçük, ρ ≈ 1 → erit | delta / doğrusal durum | 4d·d_h = 0,2M (tarama 4,2M) |
| yakın sıra (~%8) | L ≤ w, ρ yüksek → durum | FIFO, son w sembol | w·d_h (w 8: ~4k) |
| birebir tekrar (~%3 token) | seyrek okuma, kesin adres | n-gram anahtarlı arşiv, kimlik notu | O(1) + 16 bit / token |
| kalan keskin okuma (ölçülmedi) | σ_s büyük → arşiv | b bitlik not + softmax | φ·4N·d_h; bellek b / head |
| sink (~%5–10) | "boş okuma" olasılığı | durumun sabit kısmı | 0 |

- Bellek: φ, keskin head oranı. Örnek varsayım φ 0,25, b 64 bit: 12·0,25·64·10 ≈ 1,9 kbit / token. Transformer 245 kbit / token → ~125× az.
- Tedarik %42 düz: §4'e göre arşivde token seçimi yok, b sabit. Payın bir kısmı geniş okumadan (durumdan) geliyor. Keskin / geniş ayrımı ölçülmedi.

## 9. Hesabın bıraktığı tek bilinmeyen ve ölçüm (CPU, eğitimsiz)

- **Head başına keskinlik.** Bilinmeyen φ: σ_s² > 2ε olan head'lerin oranı (kütle ağırlıklı).
  - FineWeb checkpoint'inde her head için σ_s ve doğrusallaştırma hatası ölçülür: softmax yerine 1 + s ile çıkışın bağıl farkı.
  - Bu tek sayı karma oranını verir. Literatürdeki 3:1–1:7 karma oranları buradan önceden kestirilebilir mi, sınanmış olur.
- Ek (önceki tur): Fisher ağırlıklı PQ ile Δkayıp(b) eğrisi → b_doyma.
- TAHMİN (boşluğun adı): φ ≈ 0,1–0,3. Gerekçe: kütlenin %76'sı geniş, keskin kütle birkaç head'de yoğun.

## 10. Açık

- Öğrenilebilirlik türetilmedi. İki eşik eğitimde bileşen başına mı seçilecek, yoksa head türüne sabit mi atanacak?
  İkincisi bilinen karma tasarımın ta kendisi; birincisi yeni ama ayrık seçim sorununu geri getirir (TARTISMA Söz 13–14).
- Gauss varsayımı (§4) ve doğrusal maliyet modeli (§3) yaklaşıklık; eşiklerin sayıları bunlara bağlı.
