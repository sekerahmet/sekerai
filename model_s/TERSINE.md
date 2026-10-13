# Tersine mühendislik — transformer'ın hesapladığı şey başka nasıl hesaplanır (11 Ekim 2026)

Kullanıcı, 11 Ekim: *"yeni bir fikir ama bu sefer 3 kişisiniz ve bir de ben varım. Bu iş net bir şekilde matematik tersine mühendislik ile
olacak şeyler. Transformer başarılı olmuş bir mimari bunu kabul ediyorum ama yeni mimari için derin matematik ve olasılık hesabının efektif
yapılması demek . Yeni model bunu transformer gibi yapmak zorunda değil. ... Attention bir matris çarpımı q v k değerleri hesaplama ama
bunun dışında Ffn hesapları giriş ve çıkış yerleri de var. Yeni mimari tüm bu katmanlardaki matemstik hesabını başka hangi şekilde
yapabilirsin sorusu ile mümkün ? Yani attention ile elde edilen şey başka nasıl elde edilebilirdi"*.

Kurallar (bugünden): önce hesap, görüş / hipotez hesaptan sonra; deney yalnız hesabın bıraktığı tek bilinmeyeni ölçer; önceden tahmin.
Süzgeç: (1) maliyet formülü (token başı FLOP, bellek, bağlamla büyüme); (2) iki terimli marjinal değer (GOZDEN_GECIRME §20: kendi
terimi + tedarik terimi; tedarik %42, düz); (3) yaptığı işi (aşağıdaki "elde edilen şey") koruyor mu — ölçüyle.

## 1. Parça parça: ne hesaplıyor (formül) · maliyet (token başı, d, N görülen anahtar) · elde edilen şey

| parça | formül | FLOP / token | elde edilen şey (literatür + bizde ölçülecek) |
|---|---|---|---|
| giriş | e = E[x] | ~0 (arama) | token kimliği → vektör |
| attention | o_t = Σ_{j≤t} softmax_j(q_t·k_j/√d) v_j | 8d² (qkvo) + 4Nd | başka konumlardan okuma (tedarik terimi): kopya / indüksiyon, önceki token, varlık izleme, konum |
| FFN (SwiGLU) | W_d (silu(W_g h) ⊙ W_u h) | 16d² (hidden 8d/3) | parametrik bellek (anahtar–değer, Geva 2021 — klasörde kontrol), doğrusal olmayan birleştirme |
| norm + kalıntı | h ← h + f(RMSNorm(h)) | ~d | ortak pano: her parça aynı alana yazar / okur |
| çıkış | p = softmax(E h) | 2Vd (V 50k: d512'de ~51M ≈ 2 katman) | log-doğrusal uzman: olasılık |

## 2. Attention'ın eşdeğer görünümleri (matematikçi doldurur, hesabıyla)

çekirdek yumuşatıcı (Nadaraya–Watson) · ilişkisel / Hopfield bellek (bir enerji adımı) · doğrusal attention = hızlı ağırlık / RNN durumu ·
veri bağımlı düşük ranklı karıştırma matrisi · koşullu beklenti E[v | q] · bağlam içi gradyan adımı. Her biri: başka hangi hesap yolu (ör.
indeks / hash, sıralama, özyineleme, sayım), maliyeti, neyi kaybeder.

## 3. Olasılık tarafı (tersine)

Hedef p(x_{t+1} | x_≤t). Bilgi kaynakları (HİPOTEZ, ölçülecek): (a) dil önsel bilgisi (parametrik), (b) bağlam içi tekrar (kopya / indüksiyon
— bağlam içi bigram sayımı gibi; kesin hesaplanabilir), (c) yavaş durum (konu, varlıklar; sıkıştırılabilir), (d) sözdizimi (yerel).
Transformer hepsini aynı mekanizmayla (attention + FFN, her katman, her token) hesaplıyor. Soru: her kaynağın en ucuz doğru hesap yolu ne,
ve olasılıklar nasıl birleşir (karışım / uzman çarpımı / log-doğrusal)?

## 4. Ölçülecek (ben, CPU, eldeki transformer checkpoint'i)

Attention gerçekte ne yapıyor: head başına attention kütlesinin (i) önceki token'a, (ii) bağlamdaki önceki eş token'ın ardılına (indüksiyon),
(iii) belge başına / sink'e, (iv) geri kalan "içerik" araması payı; ve her türün kayba katkısı (o head türünü kapatınca kayıp — kapatma
ölçüsü, eğitim değil). Bu, §3'teki kaynakların bu modelde hangi paylarla hesaplandığını verir.

## Sıra

Kullanıcı, 11 Ekim: *"Attention ile başlamak mantıklı sonra da ffn bakarız"*. Tur 1: attention (§2, §4, alternatif bilançosu). FFN, çıkış ve
olasılık birleştirme (§3) attention turu bitince.

## 5. Ekip

Ben: §1 ve §4. Matematikçi: §2 (+ §3'ün matematiği). Danışman: attention'ı başka şeyle değiştiren mimarilerin metinden sayılarla bilançosu
(doğrusal attention, SSM / Mamba, Hyena, önbellek / kNN-LM, ∞-gram, ürün anahtarlı bellek, MoR). Kullanıcı: muhakeme, yön.
EOF
echo ok
## 6. Tur 1 sonucu: attention (ÖLÇÜLDÜ + matematikçi `danisman/04` + danışman `danisman/05`)

Ölçüm (`sx_heads.py`, `sx_topk.py`; transformer d512 / 8, sınav ilk 2 satır, 4.077 hedef; s2000 / 1 epok):
kütle kendi 0,048 / 0,048, önceki 0,086 / 0,082, indüksiyon (x_{j−1} = x_t) 0,028 / 0,027, belge başı 0,054 / 0,097, geri kalan geniş içerik
0,791 / 0,756; hiçbir head bir türe uzmanlaşmamış; m_dış(8) 0,36, m_dış(32) 0,14; 64'ten uzak kütle 0,23–0,25, bunun kesin indüksiyon payı
r_b 0,04. Head kapatma (s2000) her head +0,002 ile +0,064, toplam +1,02. Eğitim süresi tabloyu değiştirmiyor.
Matematikçi: attention'ın altı görünümü üç aile — tam liste + tarama / sabit durum / sayım-hash; sayım = attention'ın kesin eşleşme sınırı
(KANIT); kapasite d_h 64: dış çarpım N 128'de 0,84, delta ≤ 2 d_h, softmax ~1.800; kaçınılmaz terim saklanan durum (token başı 131 kbit,
bilginin ~61.000 katı); transformer'ın birleştirmesi log-doğrusal uzman çarpımı (KANIT).
Danışman: sabit durumlu her aile birebir geri çağırmayı kaybeder (Zoology: açığın %82'si token'ların %6,4'ünde); hibrit (1:7–3:1 tam
attention) transformer'ı eşitler / geçer; kazanç tipik bağlamda FLOP değil KV belleği ve üretim; sonlu durumla kayıpsız geri çağırma bilinmiyor.

| attention'ın işi | pay (bizde) | başka matematik | durum |
|---|---|---|---|
| geniş ortalama | ~%76 | sabit durum (doğrusal / delta) | biliniyor |
| önceki token | ~%8 | kısa evrişim | biliniyor |
| belge başı / sink | ~%5–10 | sabit terim | biliniyor |
| birebir geri çağırma | ~%3 kütle; literatürde açığın çoğu | — | **açık boşluk** |

Açık boşluk = kullanıcının not defteri (HAYAL tur 4: *"Bir bilgi nadir ama hatırlanması gereken bilgiye onu bir yerlerde saklarım bir not
defteri gibi"*): yalnız saklanmaya değeri tutan, taramadan ulaşılan, öğrenilebilir geri çağırma. Tur 2'nin sorusu.
Çekince: SS'te birebir geri çağırma az; FineWeb'de payı ölçülmeli (danışman: literatür FineWeb benzeri veride büyük diyor).

## 7. Tasarım taslağı v0 — attention (kullanıcının **sorusu**, 11 Ekim: *"varsayım olarak token = d boyutlu bir sayı + parçacık ve parçacık da d
boyutlu bir sayı ve attention nedir bunu kısaca özetlersem sonra ffn geçelim. Model parçaları uyumlu olmalı"* — kullanıcı sonra: *"Ben token =
sayı + parçacık mı diye soru sordum anlamak için böyle olsun demedim"*; test: *"modeli test etmek ancak tasarım bitince olur ve ss ile test
ederiz sadece çalıştığını görmek için"*)

**Token (soru, karar değil):** x_t = e_t + Σ_{k∈S_t} a_{t,k} P_k biçimi bir sorudur. Hesabın desteklediği (Claude, §7b sonrası): token'ın
içeriği vektör e_t (girişe parça = hash'li n-gram tablosu: 6,6B parametre / çakışma; V2'de döngüyü artırdı); parçalar **adres** — token'ın
belleğe yazıldığı / bulunduğu defterler (parça adresi rastgeleden 6–14× isabetli, `sx_part_address.py`, yeniden ölçülüyor). Yani "token bir
vektör + bellek adresi (s parça)"; açık nokta 4 (giriş parçası ile adres parçası) buna göre: giriş parçası yok, yalnız adres — HİPOTEZ, ekiple sınanacak.
**Attention = paylaşılan parça belleği** (`TARTISMA_attention.md` Söz 11–15):
1. adres: S_t = top-s parça (ürün anahtarı), yük dengesi λ Σ p̂_k² (kapasite formülünden);
2. yazma: seyrek parça → liste (j, k_j, v_j); kalabalık (n_k > ~800) → S_k += a v_t, n_k += a;
3. okuma: o_t = Σ_{j∈∪_{k∈S_t} L_k} softmax_j(q_t·k_j) v_j + Σ_{k∈S_t kalabalık} (a_{t,k} / n_k) S_k;
+ küçük yerel pencere (önceki token / sözdizimi, ölçülen kütlenin ~%8'i).
Maliyet: aday × d (yük dengeli: ~N s² / K aday) vs attention N d; bellek ~160 bit / token liste vs KV 131 kbit.
**Uyum:** aynı sözlük P giriş (token bileşimi) ve bağlam belleği adresi; FFN turunda hedef: ezber (parametrik bellek) de aynı P üzerinde.

### 7a. Açık noktalar (kullanıcı, 11 Ekim: *"İlk token ve parçacık tasarımının açıkta kaldığı bir nokta var mı ? Mesela sabit belirli ve
tanımlı parçacık mesela"*)

1. **Parça kaynağı: tanımlı mı öğrenilen mi.** Ölçüm tanımlıdan yana: geri çağırmada token kimliği k-means kodlarını geçti (0,095 vs ≤ 0,083,
   `sx_code_recall.py`); tanımlı parça kaymaz / çökmez / türevsiz seçim istemez. Eş anlamı bilmez → aday karışım: **tanımlı** (kimlik, önceki
   1–2 token ile n-gram hash'i, konum kovası) + **öğrenilen** (anlam).
2. **Seçim token'dan mı bağlamdan mı:** giriş bileşimi token'a bağlı olabilir; bellek adresi bağlama bağlı olmalı (çok anlamlılık).
3. **Bellek iddiasındaki boşluk:** "160 bit / token" yalnız indeks; adayları softmax'la seçmek için k_j, v_j gerekir — saklanırsa yine 2d
   (KV kadar), yalnız okuma ucuzlar. Bellek kazancı ancak k, v token kimliği + parçalardan **yeniden hesaplanabilirse** — tanımlı parçalar bunu verir.
4. **Konum:** parçalar sırasız; tanımlı konum parçaları (önceki token, cümle başı) — ölçülen önceki-token kütlesi ~%8.
5. **Oranlar ve sayılar:** a sabit mi hesaplanan mı; K, s seçimi (kapasite formülü aralık veriyor).
6. **Parça seçimi nasıl öğrenilecek** (kullanıcı: *"Ya da parçacık seçimi nasıl öğrenilecek gibi"*): seçim ayrık (gradyan yok) ve doğru
   aday seçilmeyen parçanın listesindeyse model onu hiç görmez → hata sinyali oluşmaz ("görmediğini öğrenemez"). Seçenekler: yumuşak
   top-s (ürün anahtarı; ölü parça: PKM kullanım %25,8 → sorgu BatchNorm'la %80,3), yük dengesi λ Σ p̂² (kapasite formülünden), tanımlı
   parçalar (öğrenme gerekmez). Claude'un önerisi (HİPOTEZ): eğitimin başında yoğun attention **öğretmen** — seçim, öğretmenin baktığı
   konumları içeren parçaları seçmeye eğitilir (hedef = yakalanan attention kütlesi, `sx_part_address.py`'nin ölçüsü); sonra öğretmen
   kalkar. Risk: öğretmenin alışkanlığını kopyalar.

### 7b. Kısa turun cevapları (matematikçi; danışman bekleniyor)

- Tanımlı parça: girişim yok, belirsizlik var; v = onehot(ardıl) ve 1/n_k ile okuma = bağlam içi ardıl dağılımı c(y)/n (KANIT) → Söz 5'in
  sayım uzmanı. Parçalar arası güven: **hiyerarşik Dirichlet zinciri** p_m = (c_m + α_m p_{m−1}) / (n_m + α_m), m = 3, 2, 1 tanımlı n-gram,
  taban p_0 = softmax(LM + öğrenilen parça okuması); öğrenilen kısma gradyan ağırlığı Π_m α_m / (n_m + α_m) — öğrenilen parçalar yalnız
  tanımlıların bilmediği yerde öğrenir. **Kural:** anahtar yerel token'ların belirlenimci fonksiyonuysa tanımlı, değilse öğrenilen.
- Bağlamsız k, v: katman başı ~93× az bellek; kayıp H(Y | n-gram eşi) − H(Y | tam bağlam) — ikinci terim hesaplanamaz (ölçülür).
- Seçim: kesik top-s'de seçilmeyen parçanın gradyanı tam 0 (KANIT); yoğun gevşetmede ilk s/K'nin kütlesi Φ(σ − z_q) (K 65.536, s 16:
  σ 1 → %0,7, σ 4 → %70). **Öğretmen önerisi (Claude) reddedildi:** hedef "yakalanan kütle" yönlendiriciyi yaygın parçalara iter (kütlenin %76'sı
  geniş ortalama), öğretmen evresi O(N d), kaldırınca kayma (Δo = m_kaçan (o_S − o_N)) birikir. **Kabul: örneklenmiş adaylar** — seçilen s'ye r
  rastgele parça + log q düzeltmeli örneklenmiş softmax (tam gradyanın tutarlı kestirimi; r 4: ~2k işlem / token) + λ Σ p̂², ileri hesap hep seyrek.
- Danışman (metin + `scratchpad/danisman/ngram_collide.py`, FineWeb 20,5M token): hash'li n-gram tablosunda ezilen pay — bigram m 1M %27,
  12,8M %2,3; trigram 1M %79, 12,8M %16 (Heaps ile 10B token'da daha kötü); m 12,8M × d 512 = 6,6B parametre. OE: 400M = 1B taban
  (`2025/huang2025_overtokenized.txt` L58, L416); Engram: kanonikleştirme sözlüğü %23 küçültür, çok başlı asal hash (`2026/deepseek2026_engram.txt`
  L293–340). Proje ölçümü: V2 n-gram embedding döngüyü artırdı (varsayılan kapalı). Öğretmen: **DeepSeek-V3.2 DSA** yoğun ısınma, KL(head toplamlı
  attention ‖ indeksleyici), 1.000 adım, sonra seyrek KL (`2025/deepseek2025_v32_dsa.txt` L201–228); yük dengesi: DeepSeek-V3 yardımcı kayıpsız
  sapma terimi (`2024/deepseek2024_v3.txt` L430–449; BPB 0,727 → 0,724).

**Karar (Claude):** (1) tanımlı parçalar **bağlam belleğinde** (belge başına kesin anahtar; parametre / çakışma yok), **girişte değil** —
girişteki parçalar öğrenilen sözlükten (hash tablosu maliyeti 6,6B parametre ve proje ölçümünde döngü artışı girişe ait). (2) Seçim öğretimi:
**örneklenmiş adaylar** (matematikçi) + **yardımcı kayıpsız sapma dengesi** (DeepSeek-V3); DSA tarzı yoğun ısınma yalnız yedek, keskin head'lerle
sınırlı (DSA geniş seçimde çalışır, bizim aday sayısı küçük; öğretmen kütlesi yaygın — matematikçinin yanlılık hesabı).

## 8. Temel: attention neyi, nasıl okumalı (kullanıcı, 11 Ekim: *"Yani ilk kısmı teorik olarak doğru eksiksiz kurarsak ffn adımı doğru olur
bir de attention neyi nasıl okumalı"*)

| ne okunur | pay (ölçülen) | nasıl | ne saklanır | düzey |
|---|---|---|---|---|
| son birkaç token (sözdizimi) | ~%8 kütle | yerel pencere, birebir | son w token | temsil |
| birebir tekrar (sonek eşinin ardılı) | FineWeb token %2–7 | tanımlı parça (bağlam içi n-gram) + Dirichlet geri çekilme | token kimliği (16 bit) | **olasılık** |
| anlamca benzer yer (eş anlam, bulanık) | ölçülmedi | öğrenilen parça adres + adaylarda softmax | **? (açık)** | temsil |
| hikâyenin geneli | ~%76 kütle | kalabalık parçaların toplamı | parça başı toplam | temsil |
| sink | ~%5–10 | sabit terim | — | temsil |
| **bastırma** ("bu zaten söylendi": tekrarı frenler) | GPT-2 Small L10H7 rolünün ≥%76,9'u kopya bastırma (`2023/mcdougall2023_copysuppression.txt` L32, L64) | tanımlı parça okumasında işaretli kol: Dirichlet logit'ine negatif katkı | token kimliği | **olasılık** |

Temsil düzeyindeki okumaların toplamı = FFN'in girdisi → FFN turu bu tablo üzerine kurulur.
**Açık hücre: öğrenilen parça adayının "notu"** — bağlamsal durumun tamamı (~16 kbit / token / katman, KV kadar), yalnız kimlik (ucuz, anlam
ayrımı yok) ya da sıkıştırılmış not (b bit; FineWeb'de alt sınır h ≈ 4,3 bit / token, transformer 131 kbit). Kullanıcının not defteri fikrinin
(HAYAL tur 4) karşılığı.

### 8a. Not hücresi ve tablo eksikleri (matematikçi + danışman)

- **Not (matematikçi, HESAP, Gauss bilgi darboğazı):** kanonik korelasyon ρ_i, bileşen başına r_i bit: I_i = ½ log₂(1/(1 − ρ_i²(1 − 2^{−2r_i})));
  her bileşen ~2 bitte doyar (ρ² 0,3: 1 bit %70, 2 bit %93) → b_doyma ≈ 2 n_ilgili; varsayım n_ilgili 16 → ~32 bit / token / katman
  (KV 16 kbit; ~500×). İçerik: H_j değil, Fisher ağırlıklı ilk bileşenler (F = E[g_sup g_supᵀ], tedarik gradyanı; KANIT: F^{1/2} Σ_h F^{1/2}
  özvektörleri, ters su doldurma); tedarik düz → eşit bit. Anahtar ile not ayrı (doluluk ≤ 1'de token başı anahtar biti ≈ 0); depolama adres
  256 bit + not 32–64 bit. CPU ölçümü: v_j'yi PQ ile b ∈ {16..256} bite nicemle → Δkayıp(b); Fisher ağırlıklı vs düz; tayf.
- **Literatür (danışman):** MLA 576 eleman / token / katman (`2024/deepseek2024_v3.txt` L1248–1250; ~9,2 kbit bf16); DeepSeek-V4.1-Flash 890 bayt /
  token toplam (`2026/deepseek2026_v41flash.txt` L11–12, ~178 bit / token / katman, FP4 eğitimde). Risk: not sorgudan önce yazılır (beklenmeyen
  sorgu kaybı H(Y | not) − H(Y | tam durum), ölçülür); darboğaz eğitimde öğrenilmeli; üretimde kendini besleme (döngü); keskin seçim ~ln N logit fark ister.
- **Eksik satır eklendi:** bastırma (danışman). İkinci eksik (zayıf): öğrenilen parçalarda katmanlar arası bileşim (indüksiyon iki katmanlı,
  `2022/olsson2022_induction.txt` L75).
- **Bellek kazancı neye yarar (kullanıcı: *"bellek kazancı bizim ne işimize yarıyor ? yani üretim mi model boyu mu ? yoksa ucuz bir okuma için
  kapasite mi ?"*):** model boyu hayır; üretim evet (bant sınırlı; d2048 B32 8k'da KV okuma parametre okumasının 21×); uzun bağlam kapasitesi
  evet; eğitim çok az (KV saklanmaz; attention okuması tipik N'de katmanın küçük kısmı, ~1,2×). → Bu attention tasarımı kullanıcının **ikinci**
  hedefine (bitmiş model) hizmet eder; **birinci** hedef (eğitim) için kaldıraç FFN (hesabın ~2/3'ü; PKM 12 katman + bellek 24 katmanı 2× hızla geçer).
- **Veri biçimi hatası (Claude):** FineWeb ölçümleri (`sx_code_recall.py`, `sx_part_address.py`, `sx_hd_backoff.py`) ham token'la yapıldı; V2
  modeli cümle sonu END token'larıyla eğitildi (p_0 4,40 vs sınav 3,10) → bu üç sonucun model durumuna dayanan kısmı **yeniden ölçülecek**
  (V2 sınırlarıyla). `sx_suffix_recall.py` (yalnız veri) etkilenmez.

### 8b. Olgunluk ölçüsü (kullanıcı, 11 Ekim: *"bellek kriterimizin olgunluk ölçüsü ne. bu arada hala hedefime takılma o başka yerde kaldı.
evet bir hedef olmalı. ama onlar doğru çalışan modelin bir optimizasyonu olarak gelir değil mi"*)

Önce doğru çalışan model; bellek / hız onun optimizasyonu. Bir okuma parçası (ör. not) ancak şunlarla olgun: (1) **kayıp** tam durumlu
okumaya göre ≤ ~0,01 nat (not boyutu Δkayıp(b) eğrisinin düzleştiği yerde seçilir); (2) **işlev** — §8 tablosundaki her okuma işi ayrı ayrı
korunur (tekrar dilimi doğruluğu, bastırma); (3) **metin** — üretimde döngü / tekrar artmaz (okuyarak). Bellek kazancı sonuç olarak raporlanır.
Veri biçimi düzeldi (`fw_docs.py`, V2 sınırları): FineWeb transformer'ı 40 belgede 3,109 (sınav 3,099).

### 8c. Doğru biçimle yeniden ölçüm (ÖLÇÜLDÜ, `fw_docs.py` + V2 sınırları; FineWeb transformer d768 / L10)

- **n-gram geri çekilme** (`sx_hd_backoff.py`, 200 belge, ölçü 54.674 token): transformer 3,0776; + bağlam içi n-gram (M 4) **3,1007 (−0,023)**;
  α ızgara ucuna (64) gider — model sayımları yok saymak ister. Hatalı biçimle görülen +0,074 **yapaydı** (model dağılım dışındaydı). Matematikçinin
  önkaydı: Δℓ ≈ 0 → tanımlı parçalar kaliteye katkı getirmez; değerleri ancak tam attention'ın yerini aldıkları mimaride (bu ölçüm o değişimi sınamaz).
- **Kod / kimlik** (`sx_code_recall.py`, 300 belge, 186.987 token): kimlik m ≥ 1 0,127 (%23), m ≥ 2 0,066, m ≥ 4 0,016; b8 0,126 / 0,063 /
  0,014; b12 0,092 / 0,036 / 0,008; b16 0,109 / 0,044 / 0,008 → kodlar kimlikleri geçmiyor (sonuç değişmedi).
- **Parça adresi** (`sx_part_address.py`, 60 belge ≥ 256 token): s 1 0,259 / 0,037; s 4 0,685 / 0,150; s 16 0,935 / 0,421 (rastgele taban =
  aday payı) → adres ayakta.
- **Ders:** bir modelle ölçümden önce modelin bilinen kaybı yeniden üretilir (burada sınav 3,099 ↔ ham token 4,40 farkı hatayı gösterdi).

### 8d. Açık: yerel pencere neden var (kullanıcı, 11 Ekim: *"Yerel pencere ne demek ? Attention mı"*; *"Bunu niye tasarıma ekledik"*)

Gerekçe: parçalar sırasız (anahtar ortalaması konumu siler — matematikçi Söz 2, danışman Söz 3); ölçülen kütlenin ~%8'i önceki token.
Ama ekiple tartışılmadan, transformer'da bilinen parça olduğu için eklendi. Seçenek: **tanımlı konum parçaları** ("1 önce", "2 önce", "cümle başı")
aynı parça belleğinde → tek mekanizma (parçaların uyumu). Risk: yerel sözdizimi tek tek parça okumalarıyla pencere kadar zengin kurulur mu.
Açık soru (ekibe): yerel sıra ayrı pencereyle mi, parça belleğinde konum parçalarıyla mı. Diğer açıklar: birleştirme kuralı, katmanlar,
yazma / okuma adresinin buluşması (s 4'te kütlenin %68'i).

## 9. Attention turunun dürüst hükmü (kullanıcı, 11 Ekim: *"Yani varsayım ile kafana göre parça eklersen böyle olur. ... Attention için yeni
matematik yok anladık sadece yeni kavramlar ekleyerek attention a yön göstermeye mi döndük ... önce full attention kazım bu sırada adres
parçacığı olur bu da bir nokta da moe nasıl yaparız sorusuna mı döndü ? Yani yeni tasarım aslında mevcut birşeyin keşifine mi döndü"*)

**Evet.** "Paylaşılan parça belleği" = yönlendirmeli seyrek attention (kova / küme adayları + softmax + MoE tipi denge): Reformer (LSH), Routing
Transformer, MoBA / NSA (blok seçimi), DeepSeek DSA (yoğun öğretmenle seçim öğretimi). Danışman akrabalığı söyledi; Claude farkı büyüttü — hata.
**Yöntem hatası:** açık → "şu parça eklenirse çözülür" (yerel pencere, bastırma, Dirichlet, konum parçası) → yamalarla büyüyen tasarım → bilinen
mimariye yakınsama. **Kazanım:** yeni matematik değil, attention'ın sınırlarının hesapla belgelenmesi (keskinlik üstel; sabit durum birebir geri
çağıramaz; kapasite / Zipf / bellek sınırları; attention'ın ölçülen işi ~%76 geniş ortalama, ~%3 birebir tekrar) → attention kendi işi için
matematiksel olarak iyi bir çözüm; çevresi keşfedilmiş.
**Karar (Claude):** tur kapanır; parça belleği özgün tasarım değil — **bilinen parçaların özgünlüğü doğrulanmamış ve hiç sınanmamış bir
birleşimi** (kullanıcı: *"gerçekten yeni bir varyant mı bulduk peki ?"*): bileşenlerin hepsi bilinen (kova adayları, küme toplamı, önbellek / PPM
geri çekilmesi, örneklenmiş aday, yardımcı kayıpsız denge); literatür yalnız klasörde arandı (web yok); hiç eğitilmedi; bilinenleri aştığına dair
hesap yok. FFN turu iki dersle:
(1) literatür ilk adımda (fikir varsa baştan söylenir); (2) parça ekleyerek açık kapatmak yok — önce modelin ne hesaplaması gerektiği matematikle,
tasarım ondan türer.

---

# Tur B — okuma ve token'ın kendisi tek bir problemden (kullanıcı, 11 Ekim: *"bu tasarım bitmeden bence bir sonrakine geçmeyeceğiz. çünkü bu
tasarım aynen kalıyorsa bir sonraki de aynen kalır"*; *"token ne olmalı sorusuyla beraber olmalı bu. yani biz şu an D boyutlu bir nesne olarak
tanımlıyoruz. bu kısıt olmadan konuyu ele almak gerekli"*)

**Bitti ölçütü:** (1) her bileşen aynı amaçtan türer, sonradan eklenen parça yok; (2) dört açık (birleştirme, katmanlar, adres buluşması, yerel
sıra) çözümle kapanır; (3) literatürdeki yeri (web dahil, kullanıcı onayıyla) kesin; (4) test en sonda, SS, yalnız çalıştığını görmek için.

## 10. Problem (taslak)

Girdi x_1..x_T. Seçilen (hiçbirinin biçimi verili değil): **durum** σ_t (vektör / kod / küme / dağılım / liste ... serbest), σ_t = U(σ_{t−1}, x_t, o_t);
**arşiv** m_j (|m_j| bit, 0 olabilir); **okuma** o_t = A(σ_t, {m_j}_{j<t}), maliyet c_A; **tahmin** p(x_{t+1} | σ_t, o_t).
**Amaç:** min E[−log p(x_{t+1})] + λ_m E[Σ_j |m_j|] + λ_c E[c_A + c_U], öğrenilebilirlik koşuluyla.
**İlkesel ölçüt (LİTERATÜR hafızadan, aranacak):** öngörüsel durum — Crutchfield nedensel durumları (geleceği aynı dağıtan geçmişler denk), öngörüsel
bilgi darboğazı (Still; Bialek, Nemenman, Tishby 2001 öngörüsel bilgi); gerekli bellek I(geçmiş; gelecek) ile sınırlı.
**"Token ne olmalı" buradan:** gelen sembol durumun içine erir mi (bilgisi özetle açıklanıyorsa) yoksa kimliğiyle arşivlenir mi (ileride birebir
gerekecekse)? Ölçülenler: kütlenin ~%76'sı geniş ortalama (eriyen), ~%3 birebir tekrar (arşiv), Jelassi: kopya özetle yapılamaz. Transformer bu
seçimi yapmaz (her token tam vektör arşivlenir, özet her adım yeniden hesaplanır); SSM hiçbir şeyi arşivlemez.
Beklenen: token'ın biçimi, arşiv biti, okuma yolu — hepsi bu amaçtan.

### 10a. Danışman: formülün eksikleri ve erken uyarı

Eksikler (kabul, Claude): (i) **paralel eğitim** — σ_t = U(σ_{t−1}, x_t, o_t) genel özyineleme T sıralı adım ister; kazanan aileler bu kısıtın
çözümleri (attention konumları bağımsız; doğrusal / delta özyineleme birleşmeli → paralel ön ek / parçalı tarama; `2026/m2rnn2026.txt` L62–70) →
amaca c_train = sıralı derinlik; (ii) **arşiv kararı çevrimiçi** (gelecek sorgu bilinmeden; önbellek politikası); (iii) **öğrenilebilirlik** = veri
bütçesi B ile SGD'nin bulduğu çözüm (transformer kopyayı GSSM'den 100× az örnekle öğrenir, `2024/jelassi2024_repeat.txt` L287); (iv) **beklenti
serbest üretimde de** (döngü); (v) c_A bayt (üretim bant sınırlı); (vi) **girdi birimi (bayt / BPE / bilgi birimi) de seçim değişkeni** —
kullanıcının "token ne olmalı" sorusunun tam karşılığı, Strata'nın ilk sorusu.
Literatür: Crutchfield / Still / Bialek–Nemenman–Tishby klasörde yok (web gerekir); amaçtan türetilmiş "arşivle / özetle" LM çalışması klasörde yok;
Hilberg yasası (hafızadan): öngörüsel bilgi ~T^β → sonlu nedensel durum yok, arşiv + özet zorunlu.
**Erken uyarı:** doğal çözüm büyük olasılıkla bilinen aile — özet: birleşmeli özyineleme (GDN) + öğrenilen seçici arşiv: Selective Attention
(aynı ppl'de attention belleği 512 / 1.024 / 2.048'de 16 / 25 / 47× az, `2024/leviathan2024_selective.txt` L60–67), FoX (`2025/lin2025_fox.txt`
L69), Titans "being surprising is more memorable" (`2025/behrouz2025_titans.txt` L221–228; Strata sürpriz ilkesinin bellek tarafı), StreamingLLM.
**Çıta:** 2.048'de aynı ppl'de 47× az attention belleği; türetilen kural bunu geçmezse yeniden keşif.

### 10b. Matematikçinin çözümü (`danisman/06_problem_cozumu.md`)

- Çözüm bilinen aile: durum + attention karması; **softmax okumada en yüksek entropili seçim (Gibbs) ve "kaynak tek geçmiş konum" modelinde Bayes
  sonsalı — arşivden okumanın tam çözümü (KANIT)**. Saf transformer λ_m → 0, sabit durum λ_m → ∞ sınırı; sonlu λ'da karma.
- Problem metninde hata (KANIT): λ_c = 0'da durum / arşiv ayrımı yok olur; durum boyu ücretsizse σ_t = bütün geçmiş → c_U ∝ |σ| eklenmeli.
- İki eşik (tek Lagrange): **arşivle** ⇔ κ_S L > κ_w + λ_m + r (c_bul / b + κ_R) (L bilgi ömrü, r okunma sayısı; doğrusal maliyet modelinde KANIT)
  + Jelassi kapasite koşulu; **duruma erit** ⇔ σ_s² ≤ 2ε (doğrusal okuma e^s'nin ilk iki terimi; bağıl hata σ_s²/2). Bit: tedarik Fisher tayfında
  su doldurma; tedarik düz → b token başına sabit.
- Dört açık amaçtan: birleştirme = Bayes (bağımsız kanıtta log-doğrusal, kopyada karışım); katmanlar = okuma zincirindeki sıçrama sayısı; adres =
  ilgililik hedefli bilgi darboğazı (n-gram anahtarı öğrenmesiz özel hal); yerel sıra = durumun kesin FIFO bileşeni.
- **Token** = duruma güncelleme katkısı + arşiv kaydı (anahtar, b bitlik not); d boyutlu vektör türetilmiş zorunluluk değil, kap.
- Transformer'ın fazlası: tam genişlikte arşiv (head başı 2.048 bit vs 32–128) ve geniş ortalamayı tarama (d768, N_eff 1.354: 4,2M vs 0,2M, 21×).
  Öngörülen yapı (varsayım φ 0,25, b 64): ~1,9 kbit / token vs 245 kbit (~125×). Tek bilinmeyen φ = σ_s² > 2ε olan head payı (CPU; `sx_sharpness.py`).
  Öğrenilebilirlik türetilmedi (eşikler öğrenilirse ayrık seçim geri gelir; head türüne sabitlenirse bilinen karma).
- **Hüküm (Claude):** mimari yeni değil (alan deneme yanılmayla aynı karmaya vardı); yeni olan türetme — iki eşikli kural ve bit hesabı; literatürde
  olup olmadığı web taramasında (danışman).

### 10c. Tek bilinmeyen ölçüldü: keskinlik (ÖLÇÜLDÜ, `sx_sharpness.py`; FineWeb transformer d768 / L10, V2 biçimi, 30 belge, 120 head)

φ (σ_s² > 2ε, ε 0,05 / 0,1 / 0,25) = **1,000 / 1,000 / 1,000** (tahmin 0,1–0,3 → çürüdü); ortanca σ_s² 5,09; ikinci derece Taylor ile doğrusal
okumanın bağıl hatası ortanca **0,597**, > 0,25 olan head %90. → Hiçbir head doğrusal (sabit durumlu) okumayla değiştirilemez; "kütlenin %76'sı
geniş → sabit durum yapar" çıkarımı (§6, §8) **yanlıştı**: kütle konumlara yayılı ama içerikçe keskin ağırlıklı. Türetilmiş kurala göre bu modelin
en iyisi her yerde arşiv + softmax = transformer; kalan kaldıraç not boyutu (KV sıkıştırma: MLA, V4.1 — bilinen).
Çekince: sıfır atış (eğitilmiş transformer'ın head'leri); doğrusal / hibrit model eğitimde kendi özelliklerini öğrenir → "transformer'ın öğrendiği
okuma doğrusal değil", "doğrusal okuma imkânsız" değil.
**Tur B hükmü (Claude):** attention'ı daha iyi yapan yeni matematik yok; softmax attention arşivden okumanın ilkesel çözümü (KANIT) ve bu modelde
ölçümle doğrulandı. Token = duruma katkı + arşiv kaydı; arşiv kaydı her token için gerekli. Açık: danışmanın web taraması (iki eşikli kural literatürde mi).

### 10d. Web taraması (danışman; 11 metin klasöre: 2026 lkv / srttt / learningtoevict / limits / gdwm; 2025 trimkv / debowski2025; 2011
debowski; 2001 bialek / shalizi; 2000 shalizi_ib — Claude metinde doğruladı)

- Kuram: dilde öngörüsel bilgi bağlamla kuvvet yasasıyla büyür (`2001/bialek2001_predictability.txt` L170; Dębowski Hilberg) → sonlu nedensel
  durum yok, **arşiv zorunlu** (keskinlik ölçümüyle aynı yön). Nedensel durum = en küçük tam öngörücü temsil (Shalizi–Crutchfield 2001 L23);
  bilgi darboğazı β → ∞ limitinde ona iner (2000 L74) → λ_m darboğaz katsayısı.
- Uygulama: "neyi arşivle" bilinen aile — TRIM-KV (saklama kapısı + kapasite kaybı + yoğundan damıtma), LKV (%15 KV ile neredeyse kayıpsız),
  Learning to Evict (RL), Selective Attention (47×). Karşı kanıt: öğrenilen önem skorlayıcısı rastgeleyi geçemiyor (`2026/limits_learnedimportance2026.txt` L27).
- **Strata dersi:** sürprizle kapılı birebir bellek düzeltilmiş deneyde çöker (birebir eşleşme 2.250 denemede %0; `2026/srttt2026_surprisal_ttt_postmortem.txt`
  L40); doğru yazma sinyali sürpriz değil uzak bağlama bağlılık — GDWM CPMI (`2026/gdwm2026_gatedworkingmemory.txt` L84; 4× az gradyan adımı).
  Kendi ölçümümüzle örtüşür (tedarik ~ sürpriz 0,276 < belirsizlik 0,319, GOZDEN_GECIRME §20).

## 11. Attention turu kapandı (Claude)

Bitti ölçütleri: (1) tek amaçtan türetme — yapıldı; (2) dört açık amaçtan kapandı; (3) literatürdeki yer web dahil belirlendi; (4) test sona.
**Sonuç:** softmax attention arşivden okumanın ilkesel çözümü (KANIT) ve bu modelde her head'de ölçümle doğrulandı (φ = 1); arşiv zorunlu (kuram + ölçüm);
kalan kaldıraçlar not boyutu ve arşiv seçimi — ikisi de bilinen, aktif alanlar. Yeni olabilecek dar şey: CPMI tabanlı arşiv kuralı ön eğitimde (attention'ı
değiştirmez, ucuzlatır). Token = duruma katkı + arşiv kaydı (her token için gerekli).

**Ana hat notu (Strata değil):** Z2'ye ilkesel eklenebilecek iki şey — (i) yerel katmanlara kapılı delta durumu (S_t = α_t S_{t−1} − β_t (S_{t−1} k_t − v_t) k_tᵀ,
o_t = S_t q_t; hata düzelten, unutan sabit ilişki belleği; Qwen3-Next / Kimi Linear / Olmo Hybrid ucuz katmanı) — Z özetleri zaten okunduğu için katkısı
bilinmiyor, Z2 vs Z2 + delta ile ölçülür; (ii) G katmanlarında MLA tarzı KV sıkıştırma (yerel katmanlarda değil: KV pencereyle sınırlı, GQA var —
kullanıcı: *"zaten yerel katmanlar sıkıştırılmış bir şey"*).

---

# Tur C — FFN (kullanıcı, 11 Ekim: *"Başlat"*; sıra: *"Attention ile başlamak mantıklı sonra da ffn bakarız"*)

Yöntem (Tur A dersleri): (1) problem tek amaçla; (2) literatür ilk adımda (yeniden keşif riski baştan söylenir); (3) tek bilinmeyen ölçülür;
(4) parça ekleyerek açık kapatılmaz. "Bitti" ölçütü Tur B ile aynı.

## 12. Problem (taslak)

Okuma katmanı (attention / durum) token t'ye bağlamdan bilgi getirdi: r_t. Tahmin için gereken, r_t'nin **dünya bilgisi ve dil kurallarıyla**
işlenmesi: f(r_t) → bir sonraki token'ın olasılığı. FFN bu işi yoğun bir parametrik fonksiyonla yapıyor: W_d (silu(W_g r) ⊙ W_u r), her token'da
bütün parametreler (token başı 16 d² FLOP; hesabın ~2/3'ü).
Seçim değişkenleri (biçim verili değil): **bilgi deposu** (neyi parametrede tutmalı: olgu, kalıp, kural), **erişim** (her token deponun ne kadarına
dokunur: yoğun / seyrek / arama), **işlem** (doğrusal olmayan birleştirmenin biçimi ve derinliği), **paylaşım** (katmanlar arası).
**Amaç:** min E[−log p] + λ_F · FLOP / token + λ_P · parametre (bellek) + öğrenilebilirlik + paralel eğitim.
Bilinen (hafızadan, metinde doğrulanacak): FFN = anahtar–değer belleği (Geva 2021); MoE / PKM / Memory Layers / Engram erişimi seyrekleştirir.
Tek bilinmeyen adayı (ölçülür, CPU): eğitilmiş modelde FFN gizli birimlerinin token başına seyrekliği — kütlenin ne kadarı ilk k birimde, yalnız ilk k
birim tutulunca kayıp ne olur (sıfır atış). Seyreklik yüksekse seyrek erişim ilkesel; düşükse FFN yoğun hesap olarak doğru.

### 12a. Danışman literatürü (Tur C) ve seyreklik ölçümü

- **Erken uyarı (danışman; Claude metinde doğruladı):** doğal çözüm bilinen aile — ince taneli, paylaşılan uzmanlı MoE + seyrek arama belleği.
  Krajewski 2024: hesap-optimal MoE 10²⁰ FLOP'ta 20× hesaplı yoğunla eşit, 10²⁵'te > 40× (`2024/krajewski2024_finegrainedmoe.txt` L108);
  DeepSeekMoE 16B ≈ LLaMA2 7B, ~%40 hesap (`2024/dai2024_deepseekmoe.txt` L51–53); UltraMem MoE'den çıkarımda 6× hızlı (`2024/huang2024_ultramem.txt` L66–70);
  KAN eşit FLOP'ta dilde MLP'nin gerisinde (`2024/yu2024_kan_or_mlp.txt` L36). FFN = anahtar–değer belleği (Geva), olgular orta katman (ROME);
  ReLU'lu T5'te token başı etkin birim %3 (`2022/li2022_lazyneuron.txt` L56–57). Sıfır atış seyreklik yanıltabilir (TASARIM Parça 9: %50 gate +0,37).
  Yeni olabilecek dar alan (baştan): katmanlar arası paylaşılan tek seyrek bellek + token başına okuma sayısının bilgiye göre değişmesi (MoR derinlikte yapıyor).
- **Seyreklik (ÖLÇÜLDÜ, `sx_ffn_sparsity.py`; FineWeb transformer d768 / L10, hidden 2.048, V2 biçimi, 20 belge, bütün katmanlarda aynı anda, sıfır atış):**
  tam 3,0843; yalnız ilk %30 birim 3,1027 (+0,018); %10 3,2780 (+0,194); %3 3,9190 (+0,835); %1 5,3296 (+2,245). İlk %10 birim |h| kütlesinin
  katmana göre %42–63'ü. → FFN işinin büyük kısmı token başına gereksiz; ama hangi birimin gerektiğini bulmak kapı / yukarı izdüşümü tam hesaplamayı
  ister (tasarruf yalnız aşağıda) — gerçek tasarruf önceden bilen yönlendirici (MoE) ister.

### 12b. Matematikçinin türetmesi (`danisman/07_ffn_problemi.md`)

- Çözüm bilinen aile. Birim başına eşik: **f_i < 1/α → seyrek erişim** (f_i birimin gerektiği token oranı, α seyrek erişimin duvar saati çarpanı;
  doğrusal maliyet modelinde KANIT); k = Σ f_i kullanım dağılımından çıkar; seyrek kısmın taneciği birlikte etkinleşme küme boyu c (c büyük → MoE,
  c ≈ 1 → PKM / Engram). Yoğun FFN: λ_P büyük / veri kıt bölgenin çözümü. Engram'ın U biçimli dağıtım yasası bu eşiğin ampirik hali.
- Formül düzeltmeleri: toplam (P) ve etkin (A) parametre ayrı; öğrenilebilirlik (olgu ~1.000 kez görülmezse 2 bit / parametre kapasiteye ulaşılmaz,
  `2024/allenzhu2024_physics33_capacity.txt`); bellek kapasitesi ile okunan bayt ayrı; FFN çıktısı sonraki okumanın sorgusu → derinlik değişken.
- **Birleşik okuma (KANIT): model = iki arşiv + derinlik.** Attention bağlam arşivinden softmax (tek kaynak varsayımının Bayes çözümü); FFN parametre
  arşivinden normalize edilmemiş toplam (birden çok olgunun log uzayında birleşimi, uzman çarpımı; Geva ile tutarlı). Derinlik = bileşim zinciri (c adım:
  tek tablo |V|^c, ardışık c |V|). Sıkıştırılamayan olgu tabloya, sıkıştırılabilen kural hesaba (en kısa betimleme).
- Tek bilinmeyen: katkı matrisi C[t,i] = a_{t,i} (v_i·g_t); sütun görünümü (f_i → yoğun pay #{f_i > 1/α}/M; c) karar verir: yoğun pay ≳ 0,5 → yoğun FFN
  bu ölçekte doğru; küçük ve k_eff ≪ M → seyrek. Tahmin (TAHMİN): k_eff(0,01) M'nin %10–30'u. Ölçüm `sx_ffn_usage.py`.

### 12c. Tek bilinmeyen ölçüldü (ÖLÇÜLDÜ, `sx_ffn_usage.py`; FineWeb transformer d768 / L10, hidden 2.048, V2 biçimi, 12 belge, 6.225 hedef)

Katkı C = a·(v·g): token başına %90 katkı için birim payı (k_eff) L0 0,33 … L3 0,43 … L9 0,26; birimin gerektiği token payı f_i ortanca 0,25–0,42;
yoğun pay (f_i > 1/α) α 2: 0,01–0,20, **α 4: 0,92 (tüm katmanlar; L9 0,50)**, α 8: ~1. → Birimler uzmanlaşmamış, olgu kuyruğu yok (son katman hariç);
kurala göre (yoğun pay ≳ 0,5) **bu ölçek / bu modelde yoğun FFN doğru**. Çekince: model yoğun eğitildi — yoğun eğitim birimleri yaygın kullanmayı
öğretir; seyrek eğitilmiş modelin daha iyi olup olmayacağını bu ölçüm söylemez (literatür: 10²⁰ FLOP'ta bile MoE kazanır).

## 13. Tur C hükmü ve günün bütünü (Claude)

FFN: ilkesel çözüm bilinen aile — sık gereken bilgi yoğun çekirdekte, seyrek gereken seyrek erişimde (ince taneli MoE / bellek; Engram U-yasası); bu ölçekte
yoğun savunulabilir, ölçekle MoE. Yeni FFN matematiği yok (KAN dilde kaybeder). Kazanım açıklama: **model = iki arşiv (bağlam: softmax; parametre:
toplam / uzman çarpımı) + derinlik (bileşim)**.
**Gün:** tersine mühendisliğin ilkesel olarak türettiği mimari, bugünkü en iyi bilinen mimari — hibrit attention (Z2 gibi) + sıkıştırılmış KV + ince taneli
MoE FFN. Strata'ya özgü fikirler sağ çıkmadı; kazanım: transformer ailesinin neden böyle olduğunun hesapla bilinmesi.
**Sorgulanmamış iki yer (orijinallik varsa büyük olasılıkla burada):** (1) girdi biriminin kendisi (bayt / BPE / bilgi birimi; danışman Tur B (vi));
(2) öğrenme kuralı (geri yayılım / gradyan inişi) — kullanıcı başta: *"Öğrenmenin özü kritik"*.

---

# Tur D — öğrenme kuralı (kullanıcı, 11 Ekim: *"Öğrenme kuralı bence"*; önceden: *"Geri yayılım fonksiyonundan daha efektif ... farklı bakış
açısı ile olur"*, *"Öğrenmenin özü kritik"*; ilk hedef: *"transformerın o kaybına daha kısa sürede ulaşmak da bir hız ölçüsü"*)

Yöntem: Tur A–C dersleri (problem tek amaçla; literatür ilk adımda; tek bilinmeyen; parça ekleyerek açık kapatma yok).

## 14. Problem (taslak)

Veri dağılımı 𝒟, model f_θ, kayıp L(θ) = E_𝒟[−log p_θ]. **Öğrenme kuralı** = θ_{k+1} = θ_k + Δ_k, Δ_k = R(o_k, geçmiş) — o_k adım k'de elde edilen
**sinyal** (geri yayılım gradyanı, yerel sinyal, yalnız ileri geçiş ...), R onu güncellemeye çeviren **geometri** (SGD, Adam, Muon / NorMuon,
doğal gradyan ...), ve adım k'nin **hangi veriyi / token'ı** gördüğü (sıra, seçim, ağırlık).
**Amaç:** belirli bir hesap C (FLOP) ve veri D (token) bütçesinde min L(θ_K); ya da hedef kayba (transformer'ın kaybı) en az C / D ile ulaşmak.
Bilinen (hafızadan, metinde doğrulanacak): eğitim FLOP ≈ 6 N D (ileri 2 N D + geri 4 N D); geri yayılım gradyanı kesin, ileri geçişin ~2× maliyeti.
Gevşeklik nerede olabilir (aday, hüküm değil): (a) adım başına yön (geometri; Muon ölçülmüş kazanç); (b) adım başına veri (hangi token öğretir —
Strata'nın "hesap bilgiye gitsin" ilkesinin öğrenmedeki karşılığı); (c) sinyalin maliyeti (geri geçişi ucuzlatmak / atlamak); (d) hesabın yeniden kullanımı
(büyüyen model, paylaşım, damıtma).

### 14a. Danışman literatürü (Tur D; 7 metin eklendi; Claude metinde doğruladı)

| eksen | en iyi ölçülmüş | bizde |
|---|---|---|
| geometri | Muon AdamW kalitesine FLOP'un ~%52'siyle (`2025/liu2025_muonscalable.txt` L19, L93); NorMuon 1,1B'de Adam'a %21,74, Muon'a %11,31 (`2025/li2025_normuon.txt` L22–23); modded-nanogpt Adam'a ~1,9× adım | NorMuon var |
| büyüyen model | G_stack 7B: 300B yerine 194B token (`2024/du2024_stacking.txt` L61) | yok |
| veri seçimi | Rho-1 alana özel sürekli ön eğitimde DeepSeekMath'i %3 token'la (`2024/lin2024_rho1.txt` L52); RHO-LOSS görüntüde 18× adım; **genel LM ön eğitiminde ölçülmemiş** | yok |
| sinyal (geri yayılım yerine) | FF / ileri gradyan / öngörüsel kodlama küçük görüntü ağlarında; MeZO yalnız ince ayar; LM ön eğitiminde kıyas yok | — |

Hesap: geri geçiş eğitim FLOP'unun 2/3'ü → bedava sinyal en fazla 3× (gradyan kalitesi korunursa); alternatiflerin hepsi kaliteyi kaybediyor.
**Tek gerçek açık:** genel ön eğitimde "hangi token öğretir" (Strata ilkesinin öğrenmedeki karşılığı). Risk: seçilmeyen token'ların öğrettiği beceri kaybolur
(Hapax: maskeleme ilgili yeteneği bastırdı). Ölçüm eğitim ister (seçici kayıp vs tam CE, aynı kayba adım) — kullanıcı onayıyla.

### 14b. Matematikçi (`danisman/08_ogrenme_kurali.md`) ve tek bilinmeyen

- Sinyal: ucuz gradyan ilkesi — kesin gradyan ileri geçişin sabit katı; geri geçişi atlamak en çok 3×; yalnız ileri kestiricinin varyansı (n+2)‖∇f‖² (KANIT;
  n 1,4·10⁸'de bir geri yayılım ~10⁸ ileri geçiş) → "geri yayılımdan etkili sinyal" kapalı. Kural **veri üssünü değil çarpanını** değiştirir (Bayes koşullu;
  üs Hilberg'den, Bialek L502–515). 2× hedef d1024'te aynı token'da 0,10–0,21 nat ister; NorMuon −0,012. En büyük kuramsal gevşeklik **tek görüşte saklama**
  (SGD ~100–1.000 görüş, Allen-Zhu) — mimari / bellek konusu (RETRO, kNN-LM). Seçim yalnız dizi düzeyinde hesap kazandırır (geri geçiş konum sayısıyla).
- **Ölçüldü (`sx_grad_noise.py`; FineWeb transformer son checkpoint, 32 dizi, ort. 466 token):** ρ_IS = **1,028** (kural ≥ 2 → dizi seçimi ekseni kapalı);
  Spearman(|g_s|, dizi kaybı) 0,565; |G|² tahmini < 0 (gürültüde) → B_noise belirlenemedi, çok büyük (son checkpoint minimuma yakın); çekince: eğitim ortası farklı olabilir.

## 15. Tur D hükmü ve günün sonucu (Claude)

Öğrenme kuralı: sinyal ilkesel kapalı, geometri (NorMuon) en iyilerden, dizi seçimi ~%3, kural üssü değiştiremez. Kullanıcının sezgisi (*"orda da muon gibi
yeni birşey çıkmaz tutar onun bi türevini bulursunuz"*) doğrulandı.
**Gün (Tur A–D):** her turda problem doğru kuruldu, ilkesel çözüldü, ölçüldü; cevap her seferinde bilinen yere çıktı → transformer + MoE + NorMuon ailesi bizim
türetebildiğimiz düzeyde ilkesel en iyiye yakın. Kazanım: neden böyle olduğunun hesabı ("iki arşiv + derinlik"; softmax = tek kaynak Bayes; arşiv zorunlu
(Hilberg); kural yalnız çarpan). Somut değer: Z2'yi ilkesel parçalarla iyileştirmek (ölçek büyüyünce MoE FFN; yerel katmanlarda delta durumu; G'de MLA KV).
Yenilik için ders: standart problemi daha iyi çözmek değil, **hangi varsayımın yanlış olabileceğiyle** başlamak (Z2'nin cümle özeti böyle bir varsayımdı).

## 16. Ana hat tasarım notu: Z'nin birimi (kullanıcı, 11 Ekim: *"Z2 ile bir kelime token aynı boy olması sezgisel olarak bana yanlış geliyor"*;
*"bundan eminim ... z nasıl büyür token ve diğer şeyler büyümeden ? Ya da katmanlara cümlenin başını sonunu vs mi dağıtmak lazım yani z nin kv si mi
her katmanda eşit paylaşılmalı ?"*; *"Niye karmaşık oluyor ? Zaten cümle ile kelimeyi aynı birim yapmak mantıksız"*)

- Bilgi (HESAP): kelime ~2 (SS) – 4,3 (FineWeb) bit; cümle (~15–25 token) ≥ 30–100 bit + hikâyedeki yeri → Z ~10–20× daha çok taşımalı, ama aynı d.
  Z yüklü: proje ölçümü Z okuması kapalı +0,51 (cümle başı +5,48). Literatür: blok özetini birden çok önek token'ına dağıtmak iyileştirir (ho2024 §2.4).
- **Katmanlara dağıtmak kapasite eklemez:** Z'nin her katmandaki K/V'si aynı d genişlikli kalıntı akışından türer; Z2'de katman başına ayrı K/V (özelleşme)
  doğru; eşit paylaşım özelleşmeyi kaybettirir. Kapasite yalnız genişlikle artar.
- Maliyet (FFN / izdüşüm ∝ genişlik², Z konumları ~%5): tek geniş vektör d_z = 4d → ~+%80, d_z = 2d → ~+%20; **k = 4 ayrı vektör → 4d kapasite, ~+%15**.
- **Öneri: Z = kendine ait matris (k × d) + kendi ağırlıkları** (kelimeden ayrı izdüşüm / FFN, `batch.kind` ile seçilir; çok kipli modellerdeki gibi; ortak
  nokta attention head uzayı). Kelime token'ları değişmez; Z kelimeden farklı birim (biçim ve işleme). "Karmaşık" değil (Claude'un abartısı düzeltildi).
  Kodda cümle başına tek Z var (`model_z/model.py`), birden çok Z denenmedi. Deneme kararı kullanıcının.

### 16a. Literatür (danışman; Claude metinde doğruladı) ve plan (kullanıcı, 11 Ekim: *"Token 512 cümle 1024 denenebilir hatta k 1 k 2 ikisini de
uyarlanabilir yaparsın yani farklı kombinasyon. Bir de bunu deneyen olmuş mu ?"*)

- k özet yuvası: cümle sonu gist, N_g arttıkça uzun bağlamda (HELMET) başarı düzenli artar, kısa görevde değişmez (`2025/sentence_anchored_gist2025.txt`
  L385–392); Block Transformer önek > 4 belirgin iyileşir (`2024/ho2024_block_transformer.txt` L645–652).
- Türe özgü ağırlık, ortak attention: Mixture-of-Transformers (kip: metin / görüntü / ses), Chameleon 7B'de yoğun kaliteye FLOP'un %55,8'iyle
  (`2024/liang2024_mot.txt` L46–48) — kelime / cümle ayrımı için kullanılmamış.
- **Aynı dizide attention'ı paylaşan, daha geniş kalıntı akışlı özet token'ı (kelime d 512, Z d 1.024): klasörde ve web taramasında yok — denenmemiş.**
- Üst birimi geniş, token'ın yerine geçen: MegaByte (P·D_G, `2023/yu2023_megabyte.txt` L150–153), Strata-x.
**Plan (dal `deneme/z-unit`, ana hat; bayraklar uyarlanabilir, varsayılan = bugünkü Z2 bit düzeyinde):** (1) cümle başına k Z yuvası; (2) türe özgü
ağırlıklar (Z kendi izdüşüm / FFN'i); (3) Z'ye daha geniş akış (d_z). Testler: varsayılan = eski model, kesilip sürdürülen = kesintisiz, üretim = adım
adım. Sonra SS'te çalışıyor mu; GPU koşuları kullanıcı onayıyla.
**Plan düzeltmesi (kullanıcı, 11 Ekim: *"Zaten ben sana dedim salla o k olayını diye mantıken 10 token d boyutta 1 tek d boyutuna sıkıştırmak
mantıksız"*; önce: *"Senin önerdiğin aynı cümleyi farklı farklı yuvaya bölmek doğru mu ?"*):** k yuva **bırakıldı** (yuva simetrisi / kopya riski de
vardı; "yuvalar uzmanlaşır" Claude'un hipoteziydi). Tek iş: **Z'ye kendi genişliği (d_z > d) ve kendi ağırlıkları**, aynı dizide, attention head
uzayında buluşma. Bayrak d_z; d_z = d → bugünkü Z2 bit düzeyinde (testle).
