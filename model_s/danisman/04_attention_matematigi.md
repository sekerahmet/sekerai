# 04 — Attention'ın matematiği, tersinden (11 Ekim 2026, mathematician; TERSINE.md §2 ve §3)

Kural: önce hesap. HİPOTEZ / TAHMİN yalnız hesabın bıraktığı boşluğu adlandırmak için. Betik: scratchpad
`matematikci/recall.py` (CPU, saniyeler). Gösterim: d model genişliği, H head, d_h = d/H, N görülen anahtar, L katman, V sözlük.

## 0. En önemli üç madde

1. **Transformer'ın kaçınılmaz terimi okuma FLOP'u değil, token başına saklanan durum.** Token başına K/V belleği 2·d·L sayı.
   - d512 / L8'de bu 16 KB/token. Token'ın gerçek bilgisi (SS kaybı 1,49 nat = 2,15 bit) bunun ~61.000'de biri (HESAP, §3.1).
   - Üretimde bu bellek batch'e bölünemez; N > 6d/B olduğunda parametre okumasını geçer (d2048, B32: N > 384).
   - Tedarik teriminin %42 ve düz çıkması "token atla" yolunu kapatıyor. Açık kalan yol **token başına bit**: ne saklandığı ve nasıl
     okunduğu.
2. **Attention'ın altı görünümü üç hesap ailesine iniyor** (§1):
   - (i) tam liste + tarama (softmax, kNN; bellek O(N));
   - (ii) sabit boyutlu durum (doğrusal attention / hızlı ağırlık / SSM / gradyan adımı; bellek O(d·d_h));
   - (iii) sayım / hash (kesin eşleşmede kesin; bellek O(N) tamsayı).
   - Geri çağırma kapasitesi, d_h 64'te: (ii) ~d_h–2d_h çift (doğrusal: SNR = d_h/N); softmax'ta yarım kütle sınırı N½ ≈ e^{√d_h − 1/2} ≈ 1.800.
3. **Birleştirme kuralı kaynak türüne göre değişiyor** (§2):
   - karışım: "token tek bir kaynaktan doğar" varsayımında Bayes-optimal (kopya);
   - uzman çarpımı: y verildiğinde koşullu bağımsız kanıtta optimal;
   - log-doğrusal: bağımlı kanıtta en yüksek entropi çözümü.
   - Transformer'ın kalıntı + tek softmax yapısı tam olarak **log-doğrusal, ortak sıcaklıklı**. Karışımı ancak logit tümsekleriyle yaklaşık
     kurabiliyor; kopya için kesin karışım ucuz ve kapalı formda hesaplanır.

## 1. Eşdeğer görünümler (her biri: eşdeğerlik koşulu · başka yol · maliyet · kayıp)

Taban, head başına bir sorgu için: o = Σ_{j≤t} a_j v_j, a_j = exp(s_j)/Σ exp(s), s_j = q·k_j/√d_h.
Token başı FLOP (bütün head'ler): **4Nd**. Bellek: **2Nd** (katman başına).

**1.1 Çekirdek yumuşatıcı (Nadaraya–Watson).** o = Σ K(q,k_j)v_j / Σ K(q,k_j), K = exp(q·k/√d_h).
- Eşdeğerlik: her zaman tam (tanım; KANIT).
- Başka yol A, rastgele özellik: K ≈ φ(q)·φ(k), m özellik. Maliyet ~4md, bellek m·d, N'den bağımsız. Yaklaşık.
  Pozitif özelliklerde tahmin varyansı ∝ exp(‖q‖²/√d_h + ‖k‖²/√d_h)/m; keskin attention için m üstel büyür (HESAP, çekirdek varyans formülü).
- Başka yol B, hiyerarşik / çok kutuplu (uzak kümelerin özeti): O(N log N) toplam. K yumuşaksa yaklaşık iyi, keskinse kötü.
- Kayıp: keskin, içerikle seçilen tek konum (kopya). Tam da bu yollar orada kırılıyor.

**1.2 Koşullu beklenti / Gibbs sonsalı.** o = E_{j∼p(j|q)}[v_j], p(j|q) ∝ exp(s_j).
- Eşdeğerlik: tam (KANIT).
- Başka yol A, en yakın komşu (ANN) ile ilk k: o_k = yalnız ilk k anahtarla yeniden normalize edilmiş çıkış.
  - Hata kesin olarak Δo = m_dış·(o_k − o_dış), dolayısıyla ‖Δo‖ ≤ 2 m_dış max‖v‖ (KANIT, tur 4'teki cebir). m_dış = ilk k dışında kalan kütle.
  - Maliyet: indeks araması ~c·d·log N (çizge) ya da ~d·√N (IVF), artı 4kd. Bellek 2Nd artı indeks.
- Başka yol B, örnekleme: s örnekle varyans Var_p(v)/s. Normalizasyon için yine indeks gerekir.
- Kayıp: m_dış büyükse (yaygın attention) hata büyük. Bilinmeyen tek nicelik head başına m_dış(k).

**1.3 İlişkisel / modern Hopfield belleği.** Bir güncelleme adımı ξ' = V·softmax(β K^T ξ), β = 1/√d_h, attention'ın aynısı (KANIT).
- Kapasite (HESAP): ‖k‖² = d_h, rastgele anahtarlar. Doğru konum skoru √d_h, ötekiler ~N(0,1).
  - Doğru kütle m_c = e^{√d_h} / (e^{√d_h} + (N−1)·e^{1/2}).
  - Yarım kütle sınırı N½ = e^{√d_h − 1/2}: d_h 64'te 1.808.
  - Sayısal (recall.py): N 4.096'da kod çözme doğruluğu 0,999. m_c = 0,31 olsa da öteki değerler gürültü olarak birbirini götürüyor.
- Kapasite √d_h ile üstel artıyor. Bedeli bütün listenin saklanması.

**1.4 Doğrusal attention = hızlı ağırlık = RNN durumu.** Softmax yerine φ(q)·φ(k) konursa:
S_t = S_{t−1} + v_t φ(k_t)^T, o_t = S_t φ(q_t) / (z_t·φ(q_t)).
- Eşdeğerlik: softmax ile yalnız exp'in sonlu boyutlu bir özellik haritası olsaydı tam olurdu; öyle bir harita yok, dolayısıyla yaklaşık.
- Maliyet ~4d·d_h / token, bellek d·d_h. Softmax'tan ucuz olma koşulu N > d_h (= 64).
- Kapasite (HESAP + sayısal), doğrusal dış çarpımda: o = v_i + Σ_{j≠i}(k_j·k_i/d_h) v_j.
  - Çapraz konuşma varyansı / sinyal = (N−1)/d_h, yani **SNR = d_h/(N−1)**.
  - recall.py (d_h 64, 4.096 aday değerden en yakını): N 64 → 0,97; 128 → 0,84; 256 → 0,44; 1.024 → 0,05.
- Kayıp: N ≫ d_h olduğunda kesin geri çağırma.

**1.5 Bağlam içi gradyan adımı (doğrusal head).** Σ v_j k_j^T q = ΔW·q. ΔW = η·Σ v_j k_j^T, ½Σ‖W k_j − v_j‖² kaybında W = 0'dan atılan tek
gradyan adımı (KANIT, η = 1).
- Başka yol: tam en küçük kareler (RLS / delta kuralı) W* = V K^T (K K^T + λI)^{-1}. Sherman–Morrison ile token başına ~8d·d_h.
- Kapasite: rank sınırı gereği S K = V her V için ancak n ≤ d_h ise çözülebilir (KANIT).
  - recall.py: N ≤ 96 → 1,00; 128 → 0,98; 256 → 0,59; 1.024 → 0,05.
  - Çapraz konuşma d_h'ye kadar silinir, ötesinde tavan aynı.
- Kayıp: (1.4) ile aynı tavan; daha temiz geri çağırma, iki kat maliyet.

**1.6 Veri bağımlı karıştırma matrisi.** O = A·V, A = D^{-1}(exp(QK^T/√d_h) ⊙ M).
- log A'nın rankı ≤ d_h, ama exp rankı korumuyor: A tam ranklı olabilir. "Düşük ranklı" demek yaklaşım (KANIT: exp elemansal).
- Başka yol A, Toeplitz (yalnız konuma bağlı, içerikten bağımsız): FFT ile eğitimde O(d log N) / token.
  Tam eşdeğer yalnız A içerikten bağımsızsa. İçerik araması kaybolur.
- Başka yol B, yarı ayrılabilir rank r (SSM; azalımlı doğrusal attention ile SSD ikiliği): O(r·d) / token, bellek r·d.
  Tam eşdeğer: skaler azalımlı doğrusal attention ≡ yarı ayrılabilir matris (KANIT, cebir). Kapasite 1.4 / 1.5'teki gibi.

**1.7 Sayım / hash (keskin + kesin eşleşme sınırı).** Anahtar kesin eşleşmeyle tanımlanıyorsa (k_j = önceki token'ın kimliği, q = şimdiki token)
ve β → ∞ ise o = ardılların ortalaması = **bağlam içi bigram dağılımı c(x_t → y)/c(x_t)** (KANIT: sert attention, eşit skorlar, eşit ağırlık).
- Maliyet: hash + artırma O(n) tamsayı işlemi / token (n-gram boyu). Bellek O(N) tamsayı.
- Sayısal: N 2k, d_h 64'te bir indüksiyon head'i ~4N·d_h = 0,5 MFLOP / token; hash ~10² işlem.
- Kayıp: bulanık eşleşme (eş anlam, yeniden yazım) ve öğrenilmiş anahtar.

**Özet tablo (token başı, bütün head'ler; d512, d_h 64, N 2.048):**

| yol | FLOP | bellek / katman | kesin geri çağırma |
|---|---|---|---|
| softmax | 4Nd = 4,2M | 2Nd = 2,1M sayı | N½ ≈ 1.800 / head (rastgele anahtar) |
| ilk k (k 32) + ANN | ≈ 4kd + indeks ≈ 0,1M + | 2Nd + indeks | m_dış'a bağlı |
| doğrusal / SSM | 4d·d_h = 0,13M | d·d_h = 33k | ~d_h–2d_h / head |
| delta / RLS | ~8d·d_h = 0,26M | ~2d·d_h | ≤ d_h kesin, ~2d_h kod çözme |
| sayım (kesin eşleşme) | ~10² tamsayı | N tamsayı | kesin (yalnız birebir eşleşme) |

## 2. Olasılık birleştirme (TERSINE §3)

Kaynak dağılımları: p_a (önsel), p_b (bağlam içi tekrar), p_c (yavaş durum), p_d (sözdizimi).

- **Karışım** p = Σ_i π_i(x) p_i. Gizli değişkenli üretim modelinde Bayes-optimal: her token TEK bir kaynaktan doğar, π_i = p(z = i | x) (KANIT).
  - Pişmanlık sınırı: −log Σπ_i p_i ≤ −log p_i − log π_i (KANIT). En iyi kaynağa en çok −log π_i kadar ödenir.
  - Kopya / üretim ikiliği tam bu yapı (işaretçi karışımı).
- **Uzman çarpımı** p ∝ Π p_i. Kanıtlar e_i y verildiğinde koşullu bağımsızsa optimal:
  p(y | e_1..e_n) ∝ p(y)^{1−n} Π p(y | e_i) (Bayes, KANIT). Yani doğru biçimde önsel (1 − n) üssüyle düzeltilir.
- **Log-doğrusal** p ∝ exp(Σ w_i log p_i + b). E[log p_i] kısıtları altında en yüksek entropi çözümü (KANIT, MaxEnt dual).
  - Uzman çarpımını (w = 1, önsel düzeltmeli) içerir.
  - Bağımlı kanıtta w < 1 çift saymayı düzeltir.
- **Transformer:** kalıntı h = e + Σ_l f_l, logit z = E·(g ⊙ h)/rms(h). O halde z = Σ_l E(g ⊙ f_l)/rms(h) + E(g ⊙ e)/rms(h) ve
  p ∝ Π_l exp(E g f_l)^{1/rms(h)}.
  - Bu, normlanmamış uzmanların **ortak, girdiye bağlı sıcaklıklı log-doğrusal çarpımı** (KANIT, cebir).
  - Toplamsal olmayan etkileşim yalnız sonraki katmanların öncekileri okumasıyla (bileşim) geliyor.
- **Karışımı log-doğrusalla kurmak (HESAP).** Kopya ile önsel çelişiyor: p_b = δ_A, p_a(A) = 10⁻⁴; hedef karışım π_b = 0,5.
  - Log-doğrusal yol, A'nın logit'ine Δz = log(0,5/(1−0,5)) − log p_a(A) + log(1 − p_a(A)) ≈ 9,2 eklemek zorunda.
  - Bu tümsek π_b'ye ve p_a(A)'ya bağlı; dolayısıyla modelin iki niceliği de hesaplayıp logit ölçeğinde birleştirmesi gerekiyor.
  - Karışımda aynı iş tek bir π_b skaleri ve bir toplama.
- **Hangi kaynak, hangi kural (HESAP sonucu, varsayımla):**
  - (b) tekrar için karışım (token ya kopyalanır ya üretilir);
  - (a), (c) ve (d), y verildiğinde yaklaşık bağımsız kanıtlarsa log-doğrusal çarpım.
  - Bağımsızlık varsayımı ölçülmedi. Ölçüsü: Σ w_i ≠ 1 sapması (log-doğrusal yeniden uydurmada).

## 3. Kaçınılmaz terim ve başka matematik (soru 3)

**3.1 Saklanan durumun fazlalığı (HESAP).**
- Token başına K/V: 2·d·L sayı · 16 bit. d512 / L8'de 131.072 bit.
- Token'ın bilgisi: entropi oranı ≈ kayıp = 1,49 nat = 2,15 bit (SS transformer). Kimliğin kendisi bile ≤ log₂V = 15,6 bit.
- **Fazlalık 131.072 / 2,15 ≈ 61.000×** (kimliğe göre 8.400×).
- KV önbelleği bağlamın sıkıştırılmış hali değil, paralel arama için önceden hesaplanmış bir indeks.

**3.2 Üretimde bu terim baskın (HESAP).**
- Token başı bayt: parametre okuması 2·12d²·L / B (batch'e bölünür), KV okuması 2·2Nd·L (bölünmez).
- KV / parametre oranı = N·B / (6d).
  - d2048, B 32: N > 384'te KV baskın; N 8k'da 21×.
  - d512, B 32: N 8k'da 85×.
- Eğitim FLOP'unda ise okuma payı daha küçük: 4N_eff·d / 24d² = N_eff/(6d). N_eff 1.354 (FineWeb 8k, §17) için d512'de 0,44, d2048'de 0,11.
- Sonuç: eğitim maliyetinde 24d² baskın, üretimde saklanan durum baskın.

**3.3 Tedarik teriminin söylediği (§20 + KANIT).** Tedarik payı 0,42 ve göstergelere göre düz (1,6–1,7×).
- Seçerek saklamada token s'yi atlamanın kaybı onun tedarik değeri g_sup(s). Değerler düzse en iyi bütçe dağılımı tekdüze (KKT, eşit marjinal).
- Dolayısıyla token seçmek kazandırmıyor. Kazanç ancak **token başına bitten** (ne saklanıyor) ve **okuma yolundan** (nasıl bulunuyor) gelebilir.
- Üç matematik, ucuzlatma çarpanlarıyla:
  - (i) Kesin tekrarı ayrı ve kesin say (1.7). Bellek 2dL sayı/token → ~2 tamsayı/token. d512 / L8'de ~4.000× daha az, ama yalnız
    kesin eşleşen tekrar için. Bu (b) kaynağı. Birleştirme karışımla (§2), tek π_b skaleri.
  - (ii) Yavaş durumu sabit boyutlu bir durumda tut (1.4 / 1.5). Bellek N'den bağımsız d·d_h. Kapasite ~d_h çift / head.
    Bu, yavaş durumun gerçekten az sayıda olgu olduğu varsayımında yeterli. (c) kaynağı.
  - (iii) Kalan bulanık uzak aramayı ilk k + indeks ile yap (1.2). Okuma 4Nd → ~4kd. Hata ≤ 2 m_dış max‖v‖.
- **Hesabın bıraktığı tek bilinmeyen:** transformer checkpoint'inde head başına (1) m_dış(k), k = 8 / 32, ve (2) uzak kütlenin
  (pencere dışı) kesin eşleşen indüksiyon konumlarına düşen payı r_b.
  - TERSINE §4'teki ölçüm tam bunu veriyor.
  - Karar kuralı (HESAP): r_b büyükse (i) kazanır; m_dış(32) küçükse (iii) kazanır; ikisi de küçükse attention'ın bu modeldeki işi yaygın
    içerik okuması demektir ve sabit durum (ii) yetmez.
- HİPOTEZ (boşluğun adı): SS'te r_b yüksek. Hikâyeler isim ve kalıp tekrarına dayanıyor; ama bu ölçülmedi.

## 4. Açık sorular
- 1.3 ve 1.4'teki kapasite sayıları rastgele anahtar içindir; eğitilmiş anahtarlar daha iyi ayrışabilir, yani sayılar alt sınırdır.
- §2'deki bağımsızlık varsayımı ölçülmeden çarpım ile log-doğrusal arasında seçim yapılamaz.
