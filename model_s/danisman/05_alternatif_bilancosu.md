# 05 — Attention'ın (ve FFN'in) yerine başka matematik: literatür bilançosu (11 Ekim 2026)

Kullanıcı, 11 Ekim: *"Attention bir matris çarpımı q v k değerleri hesaplama ama bunun dışında Ffn hesapları giriş ve çıkış yerleri de var. Yeni
mimari tüm bu katmanlardaki matemstik hesabını başka hangi şekilde yapabilirsin sorusu ile mümkün ? Yani attention ile elde edilen şey başka
nasıl elde edilebilirdi"* (TERSINE.md).

**Kaynak.** Yalnız `belge/makaleler/` (26 metin).
- Sayılar üç okuma ajanının satır numaralı çıkarımından alındı. Altı tanesini metinde yeniden doğruladım: Zoology L57–58,
  Waleffe L422, MAD L514, Wang L182, Olmo Hybrid L462, Kimi L33. Ek olarak PKM / kNN-LM / Engram'ı metinde aradım.
- Satırsız dört dosyada (kNN-LM, ∞-gram, PKM, Geva) konum bayt ofsetiyle verildi.
- **Klasörde yok:**
  - Hyena'nın kendi makalesi; yalnız Zoology ve MAD içinde ölçülmüş.
  - Katharopoulos (doğrusal attention) ve Qwen3-Next. Qwen3.8-Flash-Next raporu var.
  - CALM (Schuster 2022) ve MoD (Raposo 2024).
  
  Web araştırması gerekiyorsa karar senin.

## 0. Önce hesap: neyin maliyetini değiştiriyorlar (HESAP)

Token başı ileri FLOP, d model boyu, N görülen anahtar sayısı (nedensel ortalama), d_h head boyu:
- attention alt katmanı: 8d² + 2Nd; KV önbelleği katman başına 2dN sayı;
- doğrusal attention / GDN: ≈ 8d² + 4d·d_h; durum d·d_h sabit;
- FFN: 16d².

d512, d_h 64 için hesap:

| N | attention | GDN | oran | KV (sayı) | GDN durumu |
|---|---|---|---|---|---|
| 256 | 2,36M | 2,23M | 1,1 | 0,26M | 0,033M |
| 2.048 | 4,19M | 2,23M | 1,9 | 2,1M | 0,033M |
| 8.192 | 10,5M | 2,23M | 4,7 | 8,4M | 0,033M |
| 32.768 | 35,7M | 2,23M | 16 | 33,6M | 0,033M |

Gerçek veride görülen anahtar sayısı küçük. FineWeb'de N_etkin 2k bağlamda 701, 8k'da 1.354 (GOZDEN §17). N_etkin 701'de attention
2,82M, GDN 2,23M: alt katmanda 1,26 kat, FFN dahil katmanda ~1,1 kat fark.
**Sonuç (HESAP):** tipik bağlamda bu ailelerin FLOP kazancı küçük. Asıl kazanç KV belleği ve üretimde bant genişliği (uzun N'de 10²–10³
kat). Literatürdeki hız sayıları da tam bu yerden geliyor (aşağıda).

## 1. Attention'ı sabit durumlu karıştırıcıyla değiştirenler (SSM / gated conv / doğrusal / delta / RWKV)

**Ortak bilanço (metin).** Genel dil modellemesini koruyorlar; bağlamdan birebir geri çağırma ve kopyayı kaybediyorlar. Kaybı küçük
oranda attention'la geri alıyorlar.

- **Zoology** (gated conv: H3, Hyena, RWKV; 70M–1,4B, Pile 10B / 50B token):
  - Attention'a açık en çok 2,1 PPL. *"82% of the gap is explained by"* bağlam içi geri çağırma (L58).
  - Geri çağırma token'ları Pile'ın %6,4'ü; geri kalan token'larda *"no gap"* (L330–332).
  - 70M attention, geri çağırma diliminde 1,4B Hyena'yı geçiyor: 2,41'e 3,43 PPL (L339).
  - Kuram: BaseConv'un gereken boyu dizi uzunluğuyla büyüyor (L95).
  - Çare: katmanların <%10'u attention. 360M'de hibrit 8,59, saf attention 9,44 PPL; toplam FLOP'ta %18 azalma (L639–660).
  - Seyrek seçimli attention açığın %72–85'ini kapatıyor (L660).
- **Jelassi 2024:** iki katmanlı transformer üstel uzunlukta dizi kopyalayabiliyor (L50).
  - Durum belleği mem(S) < L·log D − 1 ise hata > 1/2 (Cor. 2.8, L230).
  - Telefon rehberi: 410M transformer, 2,8B GSSM'yi geçiyor (L450).
  - Mamba'nın Pile perplexity'si daha düşük (L127).
- **Waleffe 2024** (8B, 3,5T token):
  - 5-shot MMLU, 1,1T'de 46,3'e 28–29. Fark 3,5T'de 1,37'ye iniyor (L190–224).
  - Telefon rehberi ~500 token sonra çöküyor (L391).
  - Hibrit: %7,1 attention (4 katman), %42,9 Mamba-2, %50 MLP (L503). En iyi kayıp ≈ %8 attention'da (L422).
  - Hibrit, transformer'a göre ortalamada +2,65; MMLU 53,6'ya 50,1. Üretim *"nearly 8× faster"* (L572).
- **Jamba** (7B ablasyon / 250B token): saf Mamba bağlam içi öğrenmede ve formatta zayıf; IMDB 48,8'e 84,1 (L575–597).
  - Oran a:m = 1:7; *"1:3 or 1:7 with virtually no performance difference"* (L524).
  - 256K'da KV: 4 GB, Mixtral'da 32 GB (L60–68).
- **MAD** (70M–7B, 500+ model): durum boyu geri çağırmayı doğrudan belirliyor; P* ∝ M^−0,28 (L563).
  - Hesap-optimal hibrit oranı %25; durum boyu da hesaba katılınca %10 (L514, L566).
- **Wang 2025** (72 model: 340M / 20B ve 1,3B / 100B):
  - Dil modellemesi oranlar arasında *"less than one percent"* değişiyor (L182).
  - Saf doğrusal modelde RULER 0,10–0,35, tam attention'da ~0,42 (L169).
  - Geri çağırma 3:1 civarında doyuyor; önerilen oran 3:1–6:1 (L12, L203). KV belleği 4–7 kat azalıyor (L201).
- **DeltaNet** (1,3B / 100B):
  - Wiki ppl 16,87; Transformer++ 16,85.
  - SWDE 49,5'e 66,6. 2 global attention katmanı eklenince 71,0, ortalama 51,8'e 50,9 (Tablo 1, L739–1057).
- **Gated DeltaNet** (1,3B / 100B):
  - S-NIAH-3 4K'da 27,6. Gerçek geri çağırma ortalaması 30,6, Transformer++ 37,0 (L934).
  - Hibrit (GDN + SWA): 39,0–40,1; ppl 16,07, Transformer++ 18,53.
- **Kimi Linear** (3B aktif / 48B, 1,4T): oran KDA:MLA = 3:1.
  - Tablo 1: 3:1 doğrulama 5,65; saf tam attention 5,77; 7:1'de 5,70; 15:1'de 5,82 (L459–480).
  - KV belleği %75 az; 1M bağlamda üretim 6 kat hızlı (L33).
- **Olmo Hybrid** (7B, 6T): GDN 3:1. Aynı Common Crawl kaybına %35, aynı MMLU'ya %49 daha az token (L462).
  - Katman durumu: GDN 1,05 MiB; 32K'da GQA 128 MiB (Tablo 1).
  - 1B'de saf GDN 0,677, transformer 0,682 (L2901).
- **Qwen3.8-Flash-Next** (125B / 6B aktif): dört katmandan biri tam attention.
  - Gerekçe: *"retains the direct token-level retrieval that no finite-state memory reproduces exactly"* (L55).
  - Ortalama: tam attention 49,87; SWA hibrit 51,15; GDN hibrit 53,81 (L263–273).
- **Griffin** (300B token): iki tekrarlayan blok + bir yerel MQA (pencere 1024).
  - Telefon rehberinde pencereye kadar kusursuz, sonra düşüyor (L751–775).
  - Llama-2'yi *"roughly 7 times fewer tokens"*'la yakalıyor (L113).
- **Hymba** (1B / 100B): bütün katmanlarda SWA geri çağırmayı %20'den fazla düşürüyor.
  - İlk, orta ve son katmanda global attention bunu geri alıyor: üretim 2,7 kat, önbellek 3,8 kat iyi (L226).
- **RWKV-7** (≤2,9B, 5,6T; saf RNN):
  - MQAR'da durum 8.192 ile 256 çiftte %72,9 (L1799).
  - Pass-key 2,9B'de 35K'ya kadar kusursuz (L1955).
- **Hızlı ağırlıklar** (Ba 2016): A ← λA + η h hᵀ. İlişkisel geri çağırmada hata %1,81; LSTM %60,8 (R 20, L148–173). LLM ölçeğinde sayı yok.

**Hesap okuması.**
- Kayıp, Zoology'nin %6,4'lük geri çağırma dilimine yoğun; o dilimi bütün katmanlarda değil, %7–25 attention katmanıyla kapatmak yetiyor.
- Hibritler tutarlı biçimde ya eşitliyor ya geçiyor: Waleffe +2,65; Kimi −0,12 ppl; Olmo %35 token; Qwen +3,9.
- Kazanç ölçeğe bağlı: Zoology'de en küçük açık 70M'de +2,14 PPL, 360M'de +0,35 PPL; Waleffe'de MMLU farkı 1,1T'de ~17, 3,5T'de 1,4.

## 2. Bağlamdan / veriden arama (önbellek LM, kNN-LM, ∞-gram)

- **Sürekli önbellek** (Grave 2016; LSTM üstünde): geçmiş durum → sonraki kelime, nokta çarpımla.
  - PTB 82,3 → 72,1; wt2 99,3 → 68,9; wt103 48,7 → 40,8.
  - Bağıl kazanç veri büyüdükçe küçülüyor: wt103'te ~%16, wt2'de ~%30 (L407).
  - Maliyet *"negligible"* (L189). p = (1 − λ)p + λ p_önbellek. Attention'ın kopya işinin tek katmanlık, eğitimsiz biçimi.
- **kNN-LM** (247M, WT-103): 18,65 → 16,12; önbellekle 15,79.
  - WIKI-100M LM + 3B'lik veri deposu: 13,73. 3B üzerinde eğitilen LM 15,17.
  - λ 0,25. Getirdiği şey nadir kalıplar ve olgular. Maliyet: ~1024 komşuluk FAISS araması.
- **∞-gram** (1,8T token indeks):
  - GPT-2 117M: 22,86 → 13,58 (−%42). Llama-2 70B: 4,65 → 4,20 (−%12).
  - Sorgu <200 ms, CPU'da; token başı 7 bayt.
  - Açık uçlu üretimde *"not be helpful, and even harmful"*.

**Hesap okuması.** Aramanın katkısı küçük modelde büyük, büyük modelde küçük (%42'den %12'ye). Arama, modelin eksik ezberini dolduruyor;
attention'ın bağlam içi işini değil.

## 3. FFN'in işini (parametrik bellek) seyrek aramayla yapanlar

- **Geva 2020:** FFN parametrelerin üçte ikisi; anahtarlar insanca okunabilir kalıplar, değerler sonraki token dağılımı.
- **PKM** (28B kelime):
  - 12 katman + 1 bellek 15,6; 24 katman 16,0. *"twice faster"*.
  - Bellek 16k → 1M olunca 22,8 → 18,0, *"inference time unchanged"*. Arama O(√K + k²)·d.
- **Memory layers at scale** (1,3B–8B, 1T):
  - 2–4 kat daha fazla hesap kullanan yoğun modelle eşleşiyor (L537). Olgusal QA %100'den fazla artıyor.
  - Kullanılan 3 bellek katmanı (L369); *"sparse and dense layers are both needed"*.
- **Engram** (27B, 262B token): n-gram araması, O(1).
  - MMLU 57,4 → 60,4. Çok anahtarlı NIAH 84,2 → 97,0, çünkü *"frees up attention capacity"* (L211).
  - Seyrek bütçenin %20–25'i Engram'a; U eğrisi, Δ 0,0139 (L586–603). ρ → 0'da *"memory cannot replace computation"*.
  - Üretim kaybı <%3. Çıkarınca olgusal başarının %29–44'ü kalıyor, okuduğunu anlamanın %81–93'ü.
- **Over-tokenized (OE):** girişte hash'li n-gram embedding. 400M + OE, 1B tabana eşit; ek FLOP %0,38 (L937–947).
- **Memory Mosaics v2:** attention'ı üç düzeyli ilişkisel bellekle değiştiriyor. Bilgi bilançosu transformer'la aynı (56,8'e 57,1);
  yeni bilgide +12,3. Hesap 18,9 / 16,7 GFLOP / token.

**Hesap okuması.** Parametrik ezber, FLOP'suz aramaya taşınabiliyor: PKM 2 kat hız, Engram <%3 maliyet. Ama yalnız hesabın bir payı
yerine; o pay Engram'da %20–25.

## 4. Derinliği token'a göre ayırmak (MoR)

Eşit FLOP'ta (16,5e18, FineWeb-Edu) MoR Nr 2, 167M parametreyle NLL 2,7511; vanilya 315M 2,7824 (L420–520). Eşit veride FLOP %25 az,
eğitim süresi −%19. 135M'de vanilyanın gerisinde (L615). Belirsizliğe göre hesap ayırmanın gerçeklenmiş hali; kazancın bir kısmı aynı
FLOP'ta daha çok token görmekten.

## 5. Bilanço tablosu

| attention / FFN'in işi | en ucuz ve kayıpsız bilinen yol (metindeki sayı) | kalan boşluk |
|---|---|---|
| yerel sözdizim, kısa bağlam, "diğer %93,6 token" | sabit durumlu karıştırıcı (GDN / Mamba-2 / SWA); Zoology'de açık yok; durum 1 MiB / katman | yok denecek kadar az; FLOP kazancı N_etkin ~700'de ~1,1 kat (§0) |
| bağlam içi birebir kopya / ilişkisel geri çağırma (%6,4 token, açığın %82'si) | katmanların %7–25'inde tam attention (3:1–7:1); seyrek seçim açığın %72–85'i | kuram: sonlu durum tam kopyalayamaz (Jelassi, Qwen L55); o katmanlar O(N) |
| uzun bağlam erişimi | 3:1 hibrit + seyrek seçimli attention (QSA 1M'de prefill 7,6 kat, decode 4,9 kat; RULER 90 → 93) | çok belgeli QA'da 16–32K transformer ~1 puan önde (Waleffe L698) |
| durum takibi | GDN (negatif özdeğer) attention'dan iyi (Olmo, L1590) | — |
| parametrik / olgusal ezber (FFN) | seyrek arama belleği: PKM, Memory+, Engram (%20–25 bütçe), OE; FLOP ~0 | yoğun hesabın yerini tutmuyor; ρ → 0 kötü |
| nadir kalıp / eğitim verisinden geri getirme | kNN-LM / ∞-gram, karıştırma λ | büyük modelde kazanç küçülüyor; üretimde zararlı olabiliyor |
| hesabı token zorluğuna göre dağıtma | MoR: eşit FLOP'ta yarı parametreyle eşit | küçük ölçekte (135M) geride |

## 6. Görüş (hesaptan sonra)

1. Literatürün ortak ayrıştırması üç parça:
   - token'ların çoğu için ucuz, sabit durumlu karıştırıcı;
   - az sayıda tam attention katmanı (yalnız birebir geri çağırma için);
   - parametrik ezber için seyrek arama.
   
   Her parçanın karşılığı ölçülmüş bir iş. Strata-x'in Thinker'ı bu üçünden hiçbiri değil.
2. Bizim ölçümlerimiz bu tabloya uyuyor:
   - pencere içi kopyada Strata = transformer (danışman, `sx_attn_copy`);
   - açık birleştirme ve yerel kapasitede: Writer'sız genişlik kazancı 0,172.
   
   Literatürde yerel kalıp ezberinin FLOP'suz yolu Engram / OE / PKM (HİPOTEZ: SS'te Writer'ın açığının bir kısmını bunlar alır).
   Sınaması: Writer 256 + OE ya da bigram tablosu. V2'de n-gram kodu var; etkisinin "modelden bağımsız" olduğu ölçülmüştü (CLAUDE.md).
3. Orijinal bir yol arayan için boşluk bir tane, ve metin açıkça söylüyor: birebir geri çağırmayı sonlu durumla kayıpsız yapan bilinen
   bir yol yok. Bugün herkes onu O(N) attention katmanlarıyla ödüyor. Tersine mühendislik sorusunun en değerli hedefi bu dilim:
   token'ların %6,4'ü, açığın %82'si (Zoology). Hesap: o dilimi O(1) ya da O(log N) yapan bir mekanizma, 3:1 hibritteki attention
   katmanlarının maliyetini kaldırır; 8k'da katman başına 10,5M'ye 2,2M (§0).
