# Strata-x gözden geçirme 1 — derin nefes (11 Ekim 2026)

Kullanıcı, 11 Ekim: *"Bazen durup sakince tasarımı gözden geçirmek de faydalı olabiliyor ... Her bir bölümden ne bekliyorum neyi
hedefliyorum neye ihtiyacı var doğru olanı alıyor mu ? Aldığını doğru işliyor mu? Kapasitesi yeterli mi ? Tasarım teorik olarak
baştan sona akışı destekliyor mu? İçeri de verimsiz yerler var mı ? ... Bir plan yapıp bunları değerlendirir danışır matematiksel
olarak ispatlar ve son tasarım fikrimi daha sadeleştiririm"*; *"Ayrıca bir örnek üzerinden düşünme deneyi de kesin yapardım"*.

Ölçüm yeri: B3 (Writer / Reader 512, Thinker d512 / 8, θ 4, SS 2.000 adım, 1,5579) ve aynı ayarda transformer (1,4886);
sınav alt kümesinin ilk 6 satırı, 12.156 hedef. Araçlar: `sx_trace.py` (örnek), `sx_voice_read.py`, `sx_confidence.py`.

## 1. Örnek (hikâye 27, `sx_trace.py`)

"It was an icy morning when a boy set out to sea. He had a secret plan ... the lost city of Atlantis ..."
Birim sınırları tasarlandığı gibi sürprizli token'larda: *icy | morning | set | to | sea | ... | Atlantis* (birim ~2,7 token).
En büyük kayıplar (Strata − transformer, nat) ve Thinker kapatılınca değişim:

| yer | açık | Thinker kapalı | tür |
|---|---|---|---|
| icy → **morning** | +2,35 | −0,19 | yerel eşdizim |
| the ocean → **around** him | +2,42 | +1,27 | yerel kalıp |
| the only one looking → **.** | +2,28 | −0,58 | yerel sözdizim |
| I will not be → **scared** | +2,00 | +0,10 | pencere içi hikâye bilgisi (fear) |
| city of → **Atlantis** | +1,69 | +1,88 | dünya bilgisi (Thinker taşıyor ama yetmiyor) |
| With a → **deep** breath | +1,38 | +0,23 | yerel eşdizim |

Örneğin söylediği (HİPOTEZ): büyük kayıpların çoğu Thinker'a bağlı değil — transformer'ın bildiği **kalıp / eşdizim** bilgisi.

## 2. Toplu ölçüm (ÖLÇÜLDÜ)

- **Kalibrasyon** (`sx_confidence.py`): Strata aşırı güvenli değil, daha az güvenli: entropi 1,644 / tf 1,572; yanlışta top-p 0,399 /
  0,409; kalibrasyon eğrisi aynı. → Strata dürüst ama **daha az bilgili**. Açık en çok H_tf 0,5–2 (kalıp) token'larında: entropi
  1,379 / 1,255, nll +0,070 (token'ların %41'i). ~~Thinker güvenle yanlış yöne itiyor~~ (ilk okumada açığın %76'sı "Thinker zararlı"
  token'larda göründü; kalibrasyon bunu çürüttü — seçim yanılgısı: yanlış token'da her bilgi çıkarma kaybı düşürür).
- **Thinker işe yarıyor:** kapatınca +0,126 (bütün türlerde). 2× büyütünce (d1024 / 12) yalnız −0,02 / −0,04 → darboğaz değil.
- **Kanal (Writer 512'de) darboğaz değil:** Voice (çalışan kapı, 1a9efe7) 1,5593 = B3 1,5579; kapı ortalama 0,03, türe göre seçici değil;
  Voice'u kapatınca +0,036 (en çok tekrar token'larında +0,073) → kullanılıyor ama ekleme değil ikame. Pencere 512 etkisiz (1,5549).
- **Writer kapasitesi darboğaz:** genişlik 256 → 512 −0,114 (Thinker 2× büyükken de −0,100); derinlik 2+2 → 4+4 −0,045;
  token / çıkış 256 → 512 (Writer 256) −0,005.

## 3. Parça parça (kullanıcının soruları)

| parça | ne bekliyorum | doğru girdi? | doğru işliyor? | kapasite | not |
|---|---|---|---|---|---|
| Reader (2 katman) | sürpriz, birim vektörü, Writer tabanı | evet (token'lar) | sınırlar sürprizli token'da (örnek) | üç iş tek parçada | kendi çıkış başı adımın ~%20'si |
| birim kuralı | eşit bilgi parçaları | Reader sürprizi | evet | — | 2,1 token / birim: sıkıştırma zayıf |
| Thinker (8, birimlerde) | hikâye durumu, yön | up(Reader son token) | evet (+0,126) | fazla (2× ≈ etkisiz) | hesabın ~yarısı |
| Thinker → Writer | durumu iletmek | tek vektör, girişte | Writer 512'de yeterli | — | Voice eklemedi |
| Writer (2 katman) | kalıp + yön → token | he + ctx | evet | **yetersiz** | açık burada |

## 4. Akış (tümden gelim) ve HESAP

x_{t+1}'i tahmin etmek iki tür bilgi ister: **(a) kalıp bilgisi** (sözcük, eşdizim, sözdizim: her token'da, ezber) ve **(b) hikâye
durumu** (yavaş değişir: kim, ne, nereye). Strata (b)'yi birimlere taşıdı — doğru (Thinker işe yarıyor, büyümesi gerekmiyor). Ama (a)
yerel kısma kaldı ve (a) **ezber kapasitesi** ister. Token başına dokunulan FFN parametresi (HESAP, d512, hidden 1.368, SwiGLU):
transformer 8 × 2,1M = 16,8M; Strata 4 × 2,1M = 8,4M (+ Thinker'ın birimde 8 FFN'i, tek vektörle iletilen). Toplam parametre Strata'da
fazla (~37,6M / 25M), ama **her token'ın erişebildiği ezber yarı yarıya**. Writer genişleyince açık kapanıyor; Thinker büyüyünce
kapanmıyor: ikisi de bununla tutarlı. Kullanıcının ölçütüyle: *"Kapasite yapacağı iş ile ölçülür"* — kalıp token'ları yalnız yerel iş
ister ve onlar düşüyor → **kapasite**.

## 5. Verimsiz yerler

1. Thinker gereğinden büyük ve birimler kısa (2,1 token): hesabın yarısı darboğaz olmayan yerde.
2. Reader'ın ayrı çıkış başı (~%20 adım) yalnız sınır + yardımcı kayıp için.
3. Kalıp bilgisi yoğun (dense) hesapla aranıyor: ezber bir arama işi, hesap işi değil.
4. Kodda kazanmayan seçenekler: plan (1,566), next, rate (SS 1,579), memory (ertelendi), Voice (ikame) → silinir / kapalı.

## 6. ~~Sadeleşmiş fikir (HİPOTEZ) — "hesap karara, arama ezbere"~~ (çürüdü, bkz. 9)

Strata'nın ilkesi (hesabı bilgiye göre harcamak) bir adım daha: üç iş, üç maliyet —
**yön kararı** → Thinker (az birimde, derin; küçülebilir); **kalıp ezberi** → Writer'da **seyrek bellek** (çok parametre, token başına
az etkin: MoE FFN ya da arama tablosu); **sözdizim** → küçük yoğun yerel katman. Transformer her token'da üçünü de yoğun hesapla yapıyor.

## 7. ~~Ölçüm planı~~ (M1 ayırt edemez, M2 verim sorusu; yerine 9)

- **M1 ezber mi dikkat mi:** Writer 256, yalnız FFN hidden 2× (attention 256 kalır). 256 → 512 kazancının (0,114) çoğunu alırsa
  darboğaz ezber kapasitesi → seyrek bellek yolu. Almazsa genişlik / dikkat → başka yol.
- **M2 Thinker fazla mı:** B3, Thinker 8 → 4 katman. Kayıp ≤ +0,01 ise hesap Thinker'dan Writer'a aktarılır.
- (Bekleyen) Voice + Writer 256 (düzeltilmiş kapı): eşik ≤ 1,615 kanal / ≥ ~1,65 kapasite.

## 8. Danışmana

Ana iddia: "açık Writer'ın ezber kapasitesinde, kanal ve Thinker değil". Çürütme ve M1 / M2 düzeneğine itiraz istenir.

## 9. Danışman antitezi ve düzeltme (ÖLÇÜLDÜ, `scratchpad/danisman/sx_gain_split.py`, 24 satır, 48.150 hedef)

A (Writer 256) → B3 (512) kazancı 0,115, token'lar Thinker kapatma Δ'sına göre (yanlılığa karşı hem A hem B3 ile sınıflanmış):
Thinker'a duyarsız token'lar (|Δ| < 0,05) kazancın yalnız %4–15'ini alıyor ve orada B3 = transformer (+0,005 / +0,008). Kazanç
Thinker'ın işin içinde olduğu token'larda. H_tf 0,5–2 kovası "kalıp" değil: orada Thinker kapatma +0,112 (genel +0,118 kadar).
Benim 6 satırlık tablomda da Thinker'sız token'ların açığı +0,015 / payı %8'di — veriyi okudum, entropi etiketine güvendim.
**Düzeltilmiş hüküm (ÖLÇÜLDÜ + HİPOTEZ):** açık ezberde de kanalda da değil, Writer'ın Thinker bilgisini yerel bağlamla **birleştirme
kapasitesinde** (kullanıcının çekincesi: *yeni içerik ikisini birden ister*). §4'teki "sürpriz kalan bilgiyi ölçer, uygulama maliyetini
ölçmez" tespiti kavramsal olarak doğru ama bu açığı açıklamıyor (yerel token'lar zaten eşit).

Ayırıcı test (danışman; kod yok): A ve B3 Thinker'sız eğitilir (`--sx_theta 1e9`: birim kapanmaz, ctx = g0). Thinker'sız 256 → 512
kazancı ≈ 0,11 kalırsa ezber; belirgin küçülürse (< 0,05) birleştirme. Tasarım adayı (danışman, N4 iletişim ilkesi): Thinker bilgisi
her Writer katmanına kendi izdüşümüyle (katman başına down_l); HESAP: token başı katman başı ~0,1M çarpma (~%0,4).
Önceden tahmin: ben Thinker'sız kazanç ~0,05 (birleştirme), danışman — (sorulmadı).

---

# Tur 2 — notlar, matematik, literatür (11 Ekim; kullanıcı: *"günlerce tartışır ama sadece 1 günde test ederiz"*)

Bu turda yeni koşu yok. GPU'da yalnız tur 1'in kararlaştırdığı Thinker'sız iki koşu (`sxnothk_*`); sonucu buraya veri olarak girer.

## 10. Notlarda çürüyüp hâlâ ayakta duranlar (düzeltilecek)

- N1 "**Küçük** yerel model ... büyük çözücü": Writer küçük kalamıyor (ÖLÇÜLDÜ, tur 1); ad zaten işlevden, varsayım da düşmeli.
- HAYAL tur 4–5 "büyütme değil yöntem → defter": defterin dayanağı "hikâye belleği zaafı"ydı; tur 1: açık birleştirmede, kopyada
  değil (uzak kopya açığın %5'i). Defter **açık soru**ya iner, sıradaki çare değil.
- HAYAL H8 plan (aday) → ÖLÇÜLDÜ 1,566 / 1,557, kazanç yok. H5 "ızgara uyarlamalı derinliği gereksiz kılar" → hiç sınanmadı (HİPOTEZ).
- N4 iletişim ilkesi "bugünkü açıkların hepsi iletişim sorunu": kanal genişliği (Voice) çürüdü; ilke "geniş kanal" değil
  "**birleştirme yeri**" olarak yeniden yazılmalı (aşağıda).

## 11. Matematik: açık hangi terimde

Sonraki token'ın logit'i ℓ_w = f_w(S, L); S = hikâye durumu (uzak), L = yerel bağlam. Fonksiyonel ANOVA ayrışımı:
**f(S, L) = a(L) + b(S) + c(S, L)** — a yalnız yerel, b toplamsal hikâye eğilimi, c etkileşim (S'nin hangi kısmının işe yaradığı
L'ye bağlı: "the small ___" → boat; hangi tekne olduğu S'de).

- a(L): Thinker'a duyarsız token'larda B3 = transformer (+0,005 / +0,008; danışman ÖLÇÜLDÜ) → **a yeterli**.
- b(S): Voice tam olarak b'yi ekler (logit'e kapılı toplam, bağlı E ile uzman çarpımı); 2 genişlikte de kazanç yok (ÖLÇÜLDÜ) → **b ya
  zaten girişten taşınıyor ya küçük**.
- c(S, L): geriye kalan; açığın yeri (HİPOTEZ, eleme ile).

c'yi kim hesaplıyor? Writer girişinde x = he + ctx (S burada **bir kez, toplamla** giriyor). SwiGLU (W_g x) ⊙ (W_u x) toplamı
açınca W_g he ⊙ W_u ctx çapraz terimi doğar: bir FFN katmanı **en çok hidden adet** çift doğrusal S×L özelliği üretir (d256: 680,
d512: 1.368); attention ise L'ye bağlı sorguyla geçmiş konumlardaki (S taşıyan) durumu çeker. Strata B3'te c için 2 Writer katmanı;
transformer'da 8 katmanın hepsi, ve her katmanda S yeniden, **o anki duruma göre** okunuyor (HESAP; sayım kaba).
Sayım tek başına yetmiyor (ÖLÇÜLDÜ, bf16 akışlı kod): Writer 256 × 4+4 1,627 ile Writer 512 × 2+2 1,558 aynı FFN çapraz özellik
sayısında (~2.700) çok farklı → **akış genişliği** de sınır: S ve L aynı d_local'lık akışta yan yana durmak zorunda.

## 12. Literatür (klasörden; metnin kendisinden)

`2024/ho2024_block_transformer.txt` (Block Transformer; global blok çözücü + yerel token çözücü, Strata-x'in yakın akrabası):
- §2.4 / Ek F: bağlam gömmesini token çözücüye vermenin üç yolu — **toplam** (bizim ctx), **cross-attention** ve **önek token'ları**
  (bağlam P adet token'a izdüşer, token çözücü onları kendi self-attention'ında görür). Önek en iyi; toplam ve cross-attention geride.
  "Expanding the number of prefix tokens ... broadens the token decoder's computation width" (§2.4).
- Ek Q: "significantly reducing token decoder severely impacts language modeling performance" — bizim ölçtüğümüzle aynı.
- Ek L: kısa blok → büyük blok çözücü, uzun blok → büyük token çözücü yararlı.
Okuma (HİPOTEZ): önek = c(S, L) için **hesap yeri** — S'ye ait P konum Writer'ın her katmanında FFN'den geçer (S kendi başına
işlenir) ve her token onlara **o anki L'ye göre** sorgu yapar (her katmanda çekme). Bizim toplam yolumuz literatürde zayıf kalan yol.
Diğer akrabalar (okunacak, tur 3): `2021/nawrot2021_hourglass.txt`, `2025/hwang2025_hnet.txt`, `2025/videau2025_aunet.txt`,
`2024/pagnoni2024_blt.txt` (BLT yerel çözücüsü cross-attention kullanıyor).

## 13. Tasarım adayları (tur 3'te çürütülecek; HESAP d512, birim / token 0,42)

| aday | c için ne ekler | ek maliyet / token | literatür |
|---|---|---|---|
| inject (her Writer katmanına toplam) | S her katmanda yeniden, toplamla | ~0,1M / katman (%0,4) | toplam yolu zayıf (ho2024) |
| **Thinker KV** (Writer attention'ı pencere + son N birimin Thinker vektörü, tek softmax) | her katmanda L'ye göre çekme | anahtar başına ~0,42 × pencere; küçük | BLT cross-attn; ho2024'te cross-attn zayıf (onlarınki ayrı modül) |
| **önek konumu** (her birim sonunda P Thinker konumu Writer dizisine; Model Z'nin Z token'ı gibi) | çekme + S'nin kendi FFN işlemi | Writer dizisi +P × 0,42 (P 1: +%42 Writer hesabı) | ho2024 en iyi; Model Z |

Önceden tahmin (sonuç görülmeden; Thinker'sız koşu çıkınca güncellenebilir): Thinker'sız eğitimde 256 → 512 kazancı ~0,05
(birleştirme). inject ≈ etkisiz (toplam yolu); önek > Thinker KV > inject (HİPOTEZ, ho2024'e dayanarak).

## 14. Thinker'sız eğitim sonucu (ÖLÇÜLDÜ, `sxnothk_*`, θ 1e9, SS 2.000 adım)

Writer 256: 1,7661 / 59,8 ms; Writer 512: 1,5942 / 90,0 ms (Thinker'lı A 1,672, B3 1,5579; transformer 1,4886).
Genişlik kazancı Thinker'sız **0,172** (Thinker'lı 0,115); Thinker katkısı Writer 256'da 0,094, Writer 512'de **0,036**.
Önceden tahminim ~0,05 → **yanlış**. Danışmanın ölçütüyle (≈0,11 ve üstü) açık yerel kapasitede; ama ayrım sanıldığı kadar temiz değil:
SS'te pencere (256) hikâyenin çoğunu görüyor, "yerel" = hikâye durumunun çoğu. Thinker ile Writer genişliği **ikame**.
HESAP (kaba): transformer'ın token düzeyindeki +4 katmanı ~0,105 (1,594 → 1,489); Thinker'ın 8 birim katmanı (~3,4 token katmanı
eşdeğeri hesap) 0,036 → SS'te birim hesabı token hesabından ~2,5× az verimli. Strata tezinin (hesap birimde) SS'te kalite başına
hesap kazancı yok. Açık soru (tur 3): tez uzun bağlamda (pencerenin göremediği yerde) mi yaşar, yoksa kazanç kalite değil üretim
maliyetinde mi (ho2024: KV ve throughput) aranmalı?

## 15. Kullanıcının girdisi (tur 3): yazma yöntemi

Kullanıcı, 11 Ekim: *"Yazarken önce akışı kafamda kurarım yani genel olarak ama yazacağım şey uzunsa önce başlarım sonra bir yerde durur
tekrar okur düzeltmelerim varsa düzeltir sonra öyle devam ederim. Ama bu benim kapasitem ve yöntemim yani sıralı yel sefer değil. Yaz oku
düzelt ve yazmaya devam et şeklinde"*.
Süzgeç (benim): (1) **plan** — bugünkü Thinker geçmişi özetler, plan yapmaz; H8 kazanç vermedi; değeri uzun metinde (seçenek a'yı
destekler, HİPOTEZ). (2) **dur ve tekrar oku** — durma yeri Strata'da var (birim sınırı); ama Thinker metni değil özeti okuyor ve §14:
özet üzerinde hesap ~2,5× az verimli. İnsan durunca cümlelerin kendisini okur → aday: birim sonunda token akışına "düşünme konumları",
yazılmış token'ları tam çözünürlükte okur (Model Z'nin Z token'ı ve pause token'larına yakın; yeni olan durma yerini bilginin seçmesi;
HİPOTEZ). (3) **düzelt** — eğitim kaybıyla ölçülemez (her token bir kez tahmin edilir); üretimde yaz–oku–düzelt yöntemi, değeri okunan
metinde; ayrı yol, not. Kullanıcının "kapasite ve yöntem" sözü: sınırlı kapasite, çabayı zor yerde artırır — Strata'nın ilk ilkesi;
yanlış olan çabayı özete harcamaktı.

## 16. Test düzeneği (tartışma 2 tur: ben, danışman, bağımsız matematikçi — `danisman/03_bagimsiz_matematik.md`)

Kullanıcı, 11 Ekim: *"ne arıyoruz ne ölçüyoruz neye hizmet ediyor ne tip bir tasarım kazancı var ve bu deney ölçüsü tek başına bir
değişken mi yoksa başka ölçülerin değişmesi gerektiğini de doğruyor mu"*.

**Ne arıyoruz:** Strata tezinin kalite yarısı — *uzun menzilli hesap, bilgi birimlerinde token'lardakinden en az eşit verimli mi?*
**Neye hizmet eder:** karar — tez (a) "uzun menzil birimde" olarak yaşar / kalite tarafında çürür, yalnız (b) üretim maliyeti kalır.
**Ölçmediği:** üretim maliyeti (KV, throughput; ayrı iş, KV önbelleği yok); FineWeb'deki gerçek uzun bağlam (pencere 64 SS'teki vekil).
**Neden pencere 64:** pencere 256'da Thinker bilgi taşımıyor (matematikçi §2.1 KANIT; ölçülen katkı 0,036) → farklar gürültüde.

Kollar (hepsi pencere 64 — Reader ve Writer aynı —, Writer 512, Thinker d512, SS 2.000 adım, aynı tohum):

| kol | ayar | hesap (eq = d512 token katmanı) |
|---|---|---|
| ref | Thinker'sız (θ 1e9) | 0 |
| T1 | her token birim (θ 1e-6), Thinker 1 katman | 1 |
| U6 | `--sx_rate 6`, 8 katman | 8 / ℓ ≈ 1,3 |
| U11 | `--sx_rate 11`, 8 katman | ≈ 0,73 |

eq koşu ORTALAMASI birim / token'dan (günlükteki `birim / token`), hedef r'den değil.

**Bağlayan ölçü (danışman):** eşit hesapta katkı — kayıp hesaba göre içbükey, farklı eq'de verim kıyası az hesaplıyı kayırır.
U katkısı (ref − kayıp) U11 ile U6 arasında doğrusal aradeğerle 1 eq'ye taşınır, T1'inkiyle kıyaslanır.
- Ön koşul: ref − T1 ≥ 0,03 (yoksa pencere 64'te de uzak bilgi az; test bilgi vermez).
- U(1 eq) ≥ T1 − 0,01 → tez (a) yaşar. U(1 eq) < T1 − 0,03 → tez kalite tarafında çürür, (b) kalır. Arası kararsız (ek tohum = kullanıcı kararı).

**Tek değişken mi:** U6 ↔ U11 yalnız birim uzunluğu. T1 ↔ U: ızgara (token / birim) — hesap aradeğerle eşitleniyor; katman sayısı bilerek
farklı (tezin sorusu "aynı hesap birimde derin mi, token'da sığ mı"). Bilinen karıştırıcılar: rate (U'da var, T / ref'te yok; matematikçi:
r 3 koşusu iki modelden de ~0,02 sapıyor), ρ eğitimde kayar, tek tohum gürültüsü ~0,01–0,02.

**Ek okuma (ek koşu yok, matematikçinin modeli):** sınav kaybı hikâye içi konuma göre t < 64 / t ≥ 64. t < 64'te pencere hikâyenin
tamamını görüyor, yalnız hesap terimi kalır: model C_U6 / C_U2 ≈ 0,9, "π sabit" alternatifi ≈ 0,39; oran ≤ 0,5 → amortisman modeli düşer
(U2 kolu gerekir; isteğe bağlı 5. koşu). Tutarlılık: T'nin eq başı verimi token katmanınınkine (~0,026) yakın olmalı; belirgin düşükse
sorun Writer'a tek vektörle toplam giriş arayüzü.

**Önceden tahminler (TAHMİN):**
- danışman: ref ≈ 1,70; katkı T1 ~0,06, U6 ~0,05, U11 ~0,035 → U(1 eq) ≈ T1 − 0,015 (kararsız, hafif aleyhte).
- matematikçi: D (pencere bedeli) 0,03–0,08; U6 eq başı 0,04–0,07 > T 0,03–0,045 (T 4 katmanla hesaplanmış) → U lehine.
- ben: T1 ~0,08 (tam çözünürlükte tek uzun menzilli katman güçlü), U6 ~0,05, U11 ~0,04 → U(1 eq) ≈ T1 − 0,03 → tez kalitede çürür.
**Tasarım kazancı:** yaşarsa "uzun birimde küçük Thinker, kalan hesap token katmanlarında"; çürürse T1 kolu kendisi bir aday:
yerel katmanlar + az sayıda tam çözünürlüklü uzun menzil katmanı (kullanıcının "dur ve metnin kendisini tekrar oku" sezgisi §15; Model Z'nin
G katmanlarına yakın) ve Strata'nın kalan iddiası üretim maliyeti.
Kod gerekmiyor (θ 1e-6 / 1e9, --sx_rate, --layers var); konuma göre bölme yerel CPU betiği (koşudan sonra).

## 17. Hesapla düzeltilmiş düzenek (kullanıcı: *"danışmanlar ... seni hesap ile ikna etmeli ... önce bunu matematik ile yapmalı"*)

Benim c(N, ℓ) kuralım (π > c) iki hesapla düzeltildi: (i) içbükeylik — doğrusal eşik 2,3–7× gevşek (danışman, üstel / kuvvet
uydurmaları); (ii) nedensel maskede attention 2Nd, d büyüdükçe c büyür (d2048'de 1,7–2,2×). "ℓ 2,4 hiçbir N'de kazanamaz" fazla güçlüydü:
c(∞; 2,4) = 0,174 ile π 0,17 beraberlik (matematikçi).
**FineWeb belge boyu (HESAP, `onbellek/fineweb/sample.parquet`, 20.000 belge, token_count):** ortanca 636, ort 1.029, %90 1.949,
%99 7.608; token ağırlıklı ortalama görülen anahtar N_eff: bağlam 2k → 701, 8k → 1.354, 32k → 1.923. c(ℓ 6, d512): tam 0,14 / 0,12 / 0,11,
hibrit rakip (3/4 pencere 256) 0,16; SS (N_eff ~150) 0,16 → **maliyet rejimi SS ≈ tipik web verisi**; attention tasarrufu gerçek veride
küçük. Bu yüzden SS pencere 64 testi karar değiştirir (önceki "koşma" kararım geri alındı).
**Ölçüt (doğrusallık yok, yalnız içbükeylik; danışman):** n_eq = 8·c ≈ 1 (U6, tipik veri) → V_U6 ile V_T1 doğrudan.
Kollar: pencere 64, Writer 512, Thinker d512 — ref (θ 1e9), T1 (θ 1e-6, 1 katman), U6 (`--sx_rate 6`, 8 katman). U11 / T2 yalnız ara bölgede.
- ön koşul ref − T1 ≥ 0,03; V_U6 ≥ V_T1 − 0,01 → tezin kalite yarısı tipik veride yaşar → uzun belge sınaması;
  V_U6 < V_T1 − 0,03 → çürür → T tasarımı (yerel + az tam çözünürlüklü katman; §15, Model Z2'ye yakın) ya da yalnız üretim maliyeti;
  arası kararsız (ek tohum kullanıcı kararı).
Bilinen boşluk (hesaplanamaz): π'nin bilgi türüne bağlılığı (uzun belgede kopya / öz oranı). Tahmin (benim): V_U6 < V_T1.
Matematikçi (son tur, `scratchpad/matematikci/rule.py`) aynı ölçüte bağımsız vardı: V_U(8) ≥ V_T(1) ise U, c ≤ 1/8 olan her N'de
(ℓ 6: N_eff ≳ 1k) 1 katmanlık T'yi yener — yalnız monotonluk varsayımı. N_eff = E[L²] / (2 E[L]) (belge boyu L). Ek HESAP:
**Reader'ın sürpriz başı 2·4.098·d FLOP = d512'de 0,67 eq** — U6'nın 8k'daki global hesabından (0,53 eq) büyük; kollar arası kıyasta
ortak (düşer), Strata–transformer kıyasında U'ya eklenir; d2048'de 0,17 eq. Ek okuma (koşu yok): sınav kaybı t < 64 / t ≥ 64.

---

# Tur 4 — çerçeve sorgusu (kullanıcı, 11 Ekim: *"Strata hedefi sadece hızlı yapmak mı ? ... Sanki ben Orjinal birşey çıkarma hedefin olduğunu
düşündüm. ... ızgara fikrinin üstünlüğü mü zayıfladı ? Matematik nerde yanıldı ? Hangi matematik daha iyi olurdu"*)

## 18. Matematik nerede yanıldı

- Hedef hız değildi: "hesap bilgiye gitsin" (AMR) ilkesiydi; gün içinde "uzun bağlamda ucuz" iddiasına daraldı (hourglass / Block'un
  iddiası, orijinal değil).
- **Yanlış nicelik eşitlendi:** okunan token'ın sürprizi (girdi bilgisi). AMR hatanın büyük olduğu yerde sıklaştırır; LM'de hata göstergesi
  bir sonraki token'ın **tahmin belirsizliği** (nedensel, a priori). Ölçüm: açığın %55'i H_tf > 2'de (%35 token), H_tf < 0,5'te açık yok.
- **İnceltme yerine kabalaştırma:** Strata-x token tabanının üstüne kaba katman ekledi; AMR ucuz kaba tabanı gereken yerde inceltir.
- **Süzgeç hatası:** HAYAL H5 (uyarlamalı derinlik) "eşit bilgili birim = eşit zorluk" diye elendi; eşit bilgi içeriği ≠ eşit tahmin zorluğu.
- **Daha iyi matematik (HESAP çerçevesi):** hesabın değeri içbükey → sabit bütçenin en iyi dağılımı KKT: her konumda **marjinal kayıp
  düşüşü / hesap eşit** (∂ℓ_t/∂c_t = λ). Gösterge: ucuz modelin kendi belirsizliği (kalibrasyonu iyi, `sx_confidence.py`).

## 19. İnceltme sınırı (HESAP, `sx_refine_bound.py`; ucuz = Thinker'sız Writer 512 (4 yerel katman), pahalı = transformer; 6 satır)

ucuz 1,6754, pahalı 1,5700 (açık 0,105). Pahalı hesabı f oranında token'a verince açığın geri gelen payı:
f 0,1 / 0,3 / 0,5 / 0,7 → belirsizlikle %25 / %51 / %75 / %93; rastgele %8 / %30 / %48 / %67.
Maliyet (kaba, ucuzun 4 katmanı yeniden kullanılırsa 4 + 4f katman): f 0,5 → transformer'ın ~%75'i, açık 0,027 (Strata-x B3: ~%100,
açık 0,068). **Üst sınır:** pahalının bağlam durumları her konumda hazır varsayıldı (MoD tipi gerçeklemede inceltilmeyen token derin
katmana durum vermez). Literatür: güvene göre erken çıkış (CALM), Mixture-of-Depths, MoR (`2025/bae2025_mor.txt`) — fikir ailesi biliniyor;
olası özgün kısım: yönlendiricinin öğrenilmiş kapı değil marjinal getiri eşitleme ilkesinden türemesi (HİPOTEZ, sınanmadı).

## 20. KKT'nin iki terimi (matematikçi tur 4, KANIT + ÖLÇÜLDÜ)

Kayıp token'lara ayrılmaz: ∂L/∂c_s = ∂ℓ_s/∂c_s (**kendi**: x_{s+1}'in tahmini) + Σ_{t>s} ∂ℓ_t/∂c_s (**tedarik**: s'yi K/V'den okuyanlar) = λ.
Strata-x yalnız tedarik göstergesini (girdi sürprizi), §18 önerim yalnız kendi göstergesini (belirsizlik) eşitliyordu. MoD tipi kesik (inceltilmeyen
token derin K/V vermez) tam olarak tedarik teriminin kaybı (Δo = m (o_S − o_N), m inceltilmeyen anahtarlara giden attention kütlesi).
ÖLÇÜLDÜ (`sx_supply.py`; transformer s2000, katman 4 kalıntısında gradyan; üst katmanlarda K/V dondurulunca = kendi, fark = tedarik;
göstergeler Thinker'sız Writer 512'den; 3 satır, 6.043 hedef; birinci derece vekil):
- tedarik payı Σ|g_sup| / Σ(|g_own| + |g_sup|) = **0,419** (ortanca oran 0,83)
- Spearman: kendi ~ H_c **0,735**; tedarik ~ sürpriz 0,276; tedarik ~ H_c 0,319; kendi ~ sürpriz 0,141
- çeyrekler: kendi H_c'ye göre 9× değişir; tedarik sürprize göre 1,6×, H_c'ye göre 1,7× (düz)
Okuma: kendi terimi belirsizlikle iyi gösteriliyor → "zor yerde derin düşün" sağlam. Tedarik büyük ve düz → her token iz bırakmalı (en az sığ
K/V); MoD tipi atlama pahalı. **Girdi sürprizi tedarik terimini belirsizlikten iyi göstermiyor** → "iki sinyal, iki iş" adayı desteklenmedi.
Ortaya çıkan tasarım ("her token okunur ve iz bırakır, belirsiz yerde derin düşünülür"; kullanıcının yazma yöntemi §15) literatürde var
(MoR'un K/V paylaşımlı varyantı). Strata'ya özgü iki fikir (sürprizle birim, sürprizin ayrı sinyali) bugünkü hesap ve ölçümden sağ çıkmadı.
Kalan açık soru (hayal turu için süzgeç: iki terimli marjinal değer): tedarik terimi büyük ve düz — okumanın sabit maliyetini ucuzlatmanın yeni yolu.
