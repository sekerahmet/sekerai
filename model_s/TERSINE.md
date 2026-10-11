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
