# Strata — hayal / süzgeç defteri

Kullanıcı, 11 Ekim 2026: *"Hayal et önce. Sonra mantık süzgecinden geçir sonra tekrar sonra yine mantık ... Bir ve ya iki adım
değil tatmin olana kadar"*; *"hafızanı belki serbest bırak. Yeteneklerin ön plana çıksın"*.

Kural: hayal turunda eleme yok; süzgeç turunda her fikir ölçülmüş bir sayıya ya da hesaba bağlanır. Etiketler: HİPOTEZ /
HESAP / ÖLÇÜLDÜ / LİTERATÜR.

---

## Tur 1 — hayal (serbest)

Bir insan okurken her kelimeye aynı emeği vermez. Bir "durum" tutar (kim, ne, nerede); kolay kelimeleri neredeyse emeksiz
tahmin eder; yalnız şaşırınca düşünür. Kelimeleri değil özü hatırlar, ama isimleri ve sayıları birebir tutar.

- H1. Durum = birkaç yuva (varlık, konu); yalnız şaşırtıcı bilgi gelince güncellenir.
- H2. Dil sürekli bir yörünge; tahmin sürekli uzayda yapılır, sonra kelimeye çözülür.
- H3. Öğrenme yalnız ağırlıkta değil, okurken yazılan hızlı bir bellekte de olur (bağlam içi öğrenme = bellek yazma).
- H4. Çıkış 50 bin kelimeye skor değil; bir vektör üretilir ve en yakın kelime bulunur.
- H5. Derinlik sabit değil; aynı çekirdek, durum oturana kadar yinelenir.
- H6. Model token token değil, "bir sonraki fikri" söyler; yüzey (kelimeler) ucuz bir katmanda dökülür.
- H7. Token birimi BPE'nin keyfi parçası değil; bilgi birimi.

## Tur 1 — süzgeç

| fikir | süzgeç | karar |
|---|---|---|
| H1 olay güdümlü durum | Sabit boyutlu durum tam geri çağırmayı kaybeder (yinelemeli modellerin bilinen zaafı). "Yalnız şaşırınca güncelle" ise Strata-x'in birim sınırının ta kendisi | Strata-x'te zaten var |
| H2 sürekli tahmin | Büyük çözücüye doğrudan bir hedef verir: bir sonraki birimin vektörü. "Sonraki kavram tahmini" daha güçlü LM bildiriyor (LİTERATÜR, `2026/nextconcept_discrete2026.txt`, `ncp_archpreview2026.txt`) | **aday**: yardımcı kayıp |
| H3 hızlı bellek | Bağlam 2.048 token; tam attention karşılanabiliyor. Uzun bağlamda değerli | park |
| H4 en yakın kelime | Normlu olasılık kaybolur (bpb ölçülemez); çarpanlı çıkış tam normlu ve ~9× ucuz (HESAP) | hayır; Parça 10 kalır |
| H5 yinelenen derinlik | Birimler eşit bilgi taşıdığı için her birim benzer zorlukta (bilgi sd 2,4; ÖLÇÜLDÜ): bilgi eşitleme, uyarlamalı derinlik ihtiyacını zaten azaltıyor | gerek yok — ızgara bunu çözüyor |
| H6 fikir sonra yüzey | Strata-x'in kendisi | var |
| H7 bilgi birimi | Strata-x'in kendisi | var |

Süzgeçten çıkan güzel bir sonuç: **bilgiyi eşitleyen ızgara, derinlik eksenindeki uyarlamayı gereksiz kılıyor.** Üç eksenli AMR
iki eksene iniyor (x ve y); t ekseni x'in sonucu.

## Tur 2 — hayal (en zayıf noktaya)

Strata-x'in en zayıf yeri: birim içinde büyük çözücünün durumu tazelenmiyor; 4 token'lık birimin son token'ı birimin başındaki
g'yi görüyor.

- H8. Büyük çözücü tek vektör değil, bir **plan** üretir: sonraki birimin her yuvası için bir vektör (k yuva). Kod çözücü
  token'ı yazarken kendi yuvasının plan vektörünü görür.
- H9. Plan, bir sonraki birimin kodlayıcı vektörünü tahmin etmeye zorlanır (H2 ile birleşir): büyük çözücü "söyleyeceği
  fikri" açıkça öngörür.
- H10. Eğitim ilerledikçe kodlayıcı iyileşir, sürpriz düşer, birimler büyür, büyük çözücü daha az noktada çalışır:
  **model öğrendikçe ucuzlar.**

## Tur 2 — süzgeç

- H8: plan = d_g → k·d_local doğrusal haritası; birim başına bir kez (HESAP: d_g·k·d_l = 1024·4·512 = 2,1M çarpma, büyük
  çözücünün birim başına ~151M'inin yanında yok sayılır). Bilgi birim başına ~θ nat, yani yuva başına θ/k nat: plan ne kadar
  bilgi taşıyacağını biliyor. **aday.**
- H9: yardımcı kayıp, büyük çözücüye token CE'sinden daha doğrudan bir sinyal; ama hedef (kodlayıcı vektörü) kendisi
  öğreniliyor, çöküş riski (iki taraf sabite gider). Çare: hedefte gradyan kesilir (stop-grad) ve kosinüs. **aday, riskli.**
- H10: E4 günlüğünden ölçülür (birim / token adım adım). Doğruysa eğitim maliyeti zamanla düşer. **ölçüm.**

## Tur 3 — mantık (tasarımın kendisine)

- R-A. Küçük kodlayıcının sürprizi büyük modelinkinden kaba: birimler bilgi sd 2,4 (büyük model) ile 5,3 (unigram) arasında
  kalır. E4 sonrası ölçülür.
- R-B. Büyük çözücüde RoPE konumu = birim sırası; birimler farklı token uzunluğunda → mesafe bozuk. İçerik tabanlı
  attention'da önemsiz olmalı (Model Z'de G NoPE iyiydi; ÖLÇÜLDÜ). Sonra NoPE da denenebilir.
- R-C. İsim kopyalama: ismin ilk geçişi yüksek sürprizli (kendi birimi), ikinci geçişi bağlamdan tahmin edilebilir → büyük
  birime gömülür; kod çözücünün penceresi (256) dışındaysa ismi büyük çözücüden (önceki birime attention) almalı. Okumada
  gözle bakılacak.
- R-D. Yoğun maske (B, T, T) GPU'da yavaş; pencereli flex maske sonra.

## Şimdiki karar

E4 (Strata-x tabanı) çalışsın; ardından E5 = Strata-x + plan (H8) + sonraki birim tahmini (H9). Plan kodda seçenek olarak
şimdi yazılır.

## Tur 4 — hikâye belleği (11 Ekim; danışman tartışması 2–3'ten)

Ölçülen zaaf (danışman, `sx_argmax.py`): Writer, hikâyede kurulmuş bilgiyi (park, hero) kaçırıp akıcı kalıba gidiyor; geçersiz yön
seçimi transformer'ın iki katı (%14 / %7); uzak kopya token başına +0,32. Voice (doğrudan yol) bunu kapatmazsa sıradaki çare.

Soru (kullanıcıya, muhakeme): **bir hikâyede neyin hatırlanmaya değer olduğunu kim / nasıl belirler?**

- H11 **şaşırtıcı olan** hatırlanır: ilk geçişte yüksek sürprizli token (yeni isim, yer) belleğe yazılır. Reader'ın sürprizi zaten var;
  bedava. Risk: sürprizli ama önemsiz şey de yazılır (yazım hatası, nadir sıfat).
- H12 **tekrar eden** hatırlanır: ikinci kez geçen önemlidir. Risk: ilk geçişte yazılmazsa ikinci geçişi zaten kaçırır (uzak kopya).
- H13 **yönü değiştiren** hatırlanır: Thinker durumunun en çok değiştiği birim (karar, amaç, olay). Sayısal analizdeki izleme
  fonksiyonu = çözümün türevi; ölçülebilir: ‖g_u − g_{u−1}‖.

Mekanizma taslağı (HİPOTEZ): Writer her token'da küçük bir **sorgu** üretir; bellek = seçilmiş geçmiş token'ların (anahtar = Reader
durumu, değer = token kimliği / embedding) listesi; kapılı attention ile çıkışa (Voice gibi) ya da Writer girdisine. Maliyet: bellek
boyu M ≪ T. Ölçüm önce sıfır atış: H11 / H12 / H13'ün seçtiği token'lar, sonradan kopyalanan (danışmanın (i)/(ii) kovası) token'ları
ne kadar kapsıyor (recall), bellek boyu ne kadar.

### Tur 4 — kullanıcının cevabı ve sentez

Kullanıcı, 11 Ekim: *"Bir insan olarak bir şey tekrar etmiyorsa unuturum genelde. Ama bu tekrar bir metin içindeki tekrar gibidir. Bir de
sürekli tekrar ettiğim bilgiyi unutmam evimin adresi gibi. Bir bilgi nadir ama hatırlanması gereken bilgiye onu bir yerlerde saklarım bir
not defteri gibi beynim de tutmam ama o bilgiye nasıl ulaşacağımı beynim de kalıcı hafızata yazarım ve gerektiğinde açıp bakarım."*

Üç bellek türü → Strata karşılığı:
1. **Sönen çalışma belleği** (metin içi tekrar güçlendirir, tekrarsız söner) → küçük bellek, girdi başına güç; tekrar görülünce artar.
2. **Kalıcı bilgi** (adres) → ağırlıklar; ÖLÇÜLDÜ: dünya bilgisinde açık yok (H_tf < 0,5 +0,009).
3. **Not defteri + dizin** (bilgiyi değil yerini hatırla) → defter = seçilmiş geçmiş token'lar olduğu gibi (ucuz); dizin = bağlamlı
   küçük anahtar; Writer belirsizken bakar, kapıyla alır. Thinker ismi taşımaz, "bak" demeyi bilir.

Kalan sorunlar: (a) ayırt etme (iki köpek: anahtar bağlamlı olmalı), (b) seyrek öğrenme sinyali (kopya hedeflerin %1–5'i; gerekirse
yardımcı kopya kaybı), (c) yazma seçimi (sürpriz = nadir bedava; "önemli" sonradan söner). LİTERATÜR: Grave 2016 sürekli önbellek
(`belge/makaleler/2016/grave2016_cache.txt`) — kullanıcının tarifi + sönme + seçici yazma. Sıra: önce Voice sonucu; zaaf kalırsa defter.

### Tur 4 — devam: yöntem, bağ, yineleme (kullanıcı)

Kullanıcı, 11 Ekim: *"Lucy ile köpeğini yanyana koyarsak hatırlarım ... Doğal olan herkesin kendi köpeği ile olan bağıdır. Ama benim
beynim maalesef kapasiteli ikinci bir beyin ya da nöron sayımı artırma seçeneğim yok. Bu yüzden beynim yöntem geliştirir ... Bazen bir
süre neydi diye kendi kendime ipucu aramaya da çalışırım hafızamı bir kaç kez yoklarım"*.

- **Büyütme değil yöntem** → danışman 02'nin en güçlü itirazına ("yerel kısım küçük kalamıyor") cevap: Writer'ı şişirmek yerine
  hatırlama yöntemi (defter) ver. Strata ilkesiyle aynı: hesap gereken yerde.
- **Bağlı yazma** → defter girdisinin anahtarı yazıldığı anın bağlamı (o birimdeki Thinker durumu "Lucy'den bahsediliyor"u taşır):
  "Lucy–köpek–Max" birlikte; "iki köpek" ayırt etmesi kendiliğinden (HİPOTEZ).
- **Yinelemeli hatırlama** → Writer belirsizse deftere bakar; yetmezse ilk bakışın ipucuyla sorguyu düzeltip yeniden bakar (yalnız zor
  token'larda: uyarlamalı hesap). Çekince: seyrek öğrenme sinyali; önce tek bakış, sonra yineleme.

Defter tasarımı (sentez): **bağlı anahtarla yazılan, kullanılmayanın söndüğü, zor anda birkaç kez bakılan not defteri.**

## Tur 5 — okurken kurulan bağlar (kullanıcı, 11 Ekim)

Kullanıcı: *"insanlarda nöron sayısı çok değişmez ama insan büyüdükçe yeni şey öğrendikçe nöronlar kendi aralarında yeni bağlar kurarlar.
bir nöron sayısız bağlantı kurabilir ve nöronların tipleride kendi aralarından uzmanlaşmış gibidir."*

- Transformer: bağlantı yapısı sabit ve yoğun; öğrenme yalnız güç değiştirir. Strata: yapı öğrenmeyle değişir (birimler büyür; ÖLÇÜLDÜ
  199 → 118 ms).
- **Not defteri = hızlı ağırlıklar** (HİPOTEZ; LİTERATÜR hafızadan: fast weights 1980–90'lar; gated delta rule — Qwen3-Next, Kimi Linear):
  okurken  M ← (1 − λ_t) M + k_t ⊗ v_t ;  hatırlarken  v̂ = M q_t.
  Bağ = k (Lucy + köpek bağlamı) ⊗ v (Max); unutma = λ (sönme); sabit kapasite = M d×d (hikâye boyundan bağımsız; "nöron sayısı sabit,
  bağlar değişir"); uzmanlaşma = katman / head başına ayrı M.
- Strata'ya özgü olan: **ne yazılacağını Reader'ın sürprizi** (yüksek sürprizde güçlü yazma), **ne zaman bakılacağını Writer'ın
  belirsizliği** (kapı) belirler; genel bir katman değil, işi belli bir hikâye belleği.
- Ölçülecek: Voice sonrası kalan açık (hikâyede kurulmuş bilgi); önce danışmanla antitez, sonra küçük kol.
