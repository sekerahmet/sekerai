# Çalışma kuralları

**Okuma sırası:** önce §0 (tez ve zincir — bağlam kaybına karşı), sonra
gerisi. §1-12'deki her madde **bu projede fiilen yapılmış bir hatadan**
çıktı; soyut iyi niyet değil, parantez içindeki olay gerçekten oldu.

Ayrıntı belgede, kural burada. Sayı alıntılarken §0'daki tabloyu kullan.

---

## 0. TEZ VE YÖNTEM — bağlam kaybına karşı

**Yeni oturum ÖNCE burayı okur.** Deneye özel sayılar burada DEĞİL,
`belge/onkayit/` ve `belge/bulgu/` altında. Burası değişmeyen kısım.

### Problem

Sentetik bilgi grafiği, iki adımlı soru: `[Q2] e r1 r2 ?` → `facts[facts[e,r1],r2]`.

```
Turkiye --baskenti--> ANKARA --nufusu--> 5.7M
            (KOPRU: modelin ANKARA'yi YAZMADAN kullanmasi gerekir)
KISAYOL : r2'yi dogrudan e'ye uygulamak -> "Turkiye'nin nufusu"
          Model bunu ogrenip kopruyu ATLIYOR. Olculen mekanizma bu.

dizilim: poz 0=Q2, 1=VARLIK, 2=r1, 3=r2, 4=cevap
MASK_KEY=p : egitim sirasinda butun sorgularin poz p'ye bakmasini engelle
```

| bölme | ne |
|---|---|
| `comp` | görülmemiş r1–r2 **çifti**, varlık görülmüş |
| `ent`  | varlık **hiç zincir başı olmamış** ← **asıl problem** |
| `ent2` | ikinci hop hiç kompozisyonel görülmemiş (özgüllük kontrolü) |

### Tez

> **Maskeleme modelin daha iyi öğrenmesini sağlar.**

Kısayol yolunu eğitim sırasında kapatırsan model köprüyü kullanmak zorunda
kalır ve `ent` yükselir.

### Antitez

> **Ölçek (veri/parametre) tek başına `ent`'i çözer; maskeleme gereksizdir.**

Antitez zayıf olamaz (tohum değiştirmek gibi). Ölçek olmak zorunda.

### Zincir — her adım ölçüldü

```
D3     Yeri TERSINE MUHENDISLIKLE bulduk.  A 0.0603 -> D3 0.3607  DOGRULANDI
D3.1   Yeri ARAYARAK da bulabiliyor muyuz? EVET, ayni yeri buldu.  DOGRULANDI
CIKARIM  Yeri bilince, orada maskeleyip AYNI TOHUM + AYNI VERIYLE
         BASLANGICTAN kosarsan D3.1, D3 gibi davranir.             DOGRULANDI
ANTITEZ  Peki OLCEK degisince?
D3.2   Veri x4 -> YANLIS EKSEN, 'olcek disi' hukmu. Literature bakilmamisti.
D3.3   Literatur sonrasi dogru eksen: phi (3.03 -> 5.06).
       SONUC: kazanc DARALIYOR ama yok olmuyor.
         onceden yazilan pencere  A5 0.0090 -> D5 0.0287   3.19x
         her kol kendi en iyisinde 0.0357 -> 0.1353        3.79x  [SONRADAN]
         D3'un 0.3607'si ile kiyas: tepe %37, pencerede %7,9
       Kacis BLOK 0'dan (olculdu): maskesiz tek blok orasi.
       ANTITEZ DE KAZANMADI: A5 0.0090 < A 0.0603.
       -> belge/bulgu/BULGU_D33_PHI_DARALMASI.md
```

### Yöntem — yeri nasıl buluyoruz (D3.1'de doğrulandı)

```
ASAMA A (ucuz, her olcumde)
   Girdinin 3 parcasini AYRI AYRI boz (1=varlik, 2=r1, 3=r2).
   sinyal(p) = degismezlik(ENT,p) - degismezlik(COMP,p)
   ENT'te COMP'tan FAZLA degismezlik veren parca = kacis yolu.
   sinyal >= ESIK olan ILK adimda atesle.

ASAMA B (pahali, bir kez)
   3 parca x L blok-kuyrugu tara.
   skor = (r1 bagimsizligini DUSUR) - (COMP'a ZARAR)
   Kazanan = argmax. INSAN MUDAHALESI YOK.
```

**İki aşamayı AYRI doğrula.** Tetiğin çalışması seçimin de çalıştığı anlamına
gelmez: φ=5.06'da Aşama A ilk durakta ateşledi ama Aşama B'nin kazananı orada
**negatif** skor verdi. Aşama B'nin skoru pozitif değilse seçim yoktur.

**Bilinen gedik:** Aşama B bir blok kuyruğu (`b0`) da döndürüyor ama
kullanılmıyor — koşular sabit `1..L-1` ile yapılıyor. Yani **pozisyon**
aramadan geliyor, **derinlik** gelmiyor.

### Ölçme

Birincil okuma: **5 anlık görüntünün AĞIRLIK ORTALAMASI**, sonra ölç.
Eğri ortalaması DEĞİL — ikisi arasında ~2x fark var ve karıştırmak bulgunun
kendisini değiştirir. Kol **kendi maskesiyle** ölçülür. Tek yol:
`sablon/pencere.py`. Pencere bağımlılığı: `sablon/kayan_pencere.py`.

**Sayı alıntılarken belgelerden kopyalama** — belgelerde iki farklı ölçüm
hattından sayılar var. Tek geçerli tablo: `belge/bulgu/BULGU_OLCUM_HATTI.md`.

---
## 1. Sıra: ölçüm → veri → tez → antitez → sentez → sentezi ölç

Kullanıcının kuralı, birebir:

> *"İmkânın varsa önce test et, elinde analiz edebileceğin hipotez üretebileceğin
> veri olsun, sonra tez üret, sonra antitez, en sonunda sentez. Sonra yine aynı
> döngü, bu sefer onu ölç."*

**Pahalı bir kol tasarlamadan önce sor:** *"bu tasarımı çürütebilecek hangi ölçüm
elimdeki veriyle YAPILABİLİR?"* Yapılabiliyorsa önce onu yap.

> (D2'nin kodu, önceden kaydı ve Colab hücresi yazıldı; **sonra** yapılan
> 13 saniyelik bir ölçüm tasarımı öldürdü. Ölçüm GPU bile istemiyordu.)

## 2. Mimari çözüm, mekanizma ölçülmeden önerilmez

> *"C ve B, A'yı anlamadan ürettiğimiz çözümler oldu ve bir verim alamadık."*

Müdahale, mekanizmanın **hangi adımını** hedeflediğini söyleyebilmeli.
Gerekçe literatüre değil, bu modelde yapılmış ölçüme bağlanır.

## 3. Yönlendirici Claude

Her adımda onay isteme. Planı söyle, uygula, raporla. Kullanıcı yönü
değiştirmek isterse söyler. **Ama pahalı/geri dönülmez bir şey başlatmadan
önce tek cümlelik bilgi ver** (süre, maliyet, neyi kaybedebiliriz).

---

## 4. Sayı raporlarken

**Eşleştir.** Eğriler kaymışsa eşit *adımda* değil eşit *ilerlemede* karşılaştır.
> ("D %20 hızlı" dendi; uç noktada 1.08x'e indi ve kontrol kolunda da vardı.)

**Tek ölçüm noktasından trend çıkarma.** Tepe geçici olabilir.
> ("Eşik geçildi, +0.108" denildi; 20.000 adım sonra +0.070'e indi ve eşiğin
> altına düştü.)

**Uzatma yapma.** Doğrusal uzatma bu projede iki kez yanıldı.

**Tek ölçüm yolu kullan.** Bir sayıyı eğriden, karşılaştırılanı kendi kodundan
alma.
> (2×2 tablosunda A'nın tek noktası eğriden, ortalaması kendi koddan alınmıştı.)

**"Arıza buldum" demeden önce doğrula.** Dosya henüz yazılmamış olabilir.
> (Bir turda iki kez "yedek çalışmıyor" denildi; ikisi de zamanlamaydı.)

**Kendi bulgunu çürütecek kontrolü aynı anda kur.** Eşleştirilmiş kontrol kolu
olmadan "X işe yaradı" denmez.
> (K kolu, "hızlanma" bulgusunu öldürdü ve "kısayol bastırması" bulgusunu
> kurtardı. K olmasaydı ikisini de yanlış raporlayacaktık.)

## 5. Literatür

**Bir iddiayı plana dayandıracaksan terimi metinde ARA.** Özetleyiciye
"doğrula" diye sorma. Terim metinde sıfır kez geçiyorsa iddia yoktur.
> (arXiv 2501.15857'nin "yakınsamış ezberci model çevrilemez" dediği aktarıldı
> ve plan buna dayandırıldı. Makalede öyle bir cümle yok; konusu bile farklı.
> İki bağımsız ajan PDF'i grep'leyerek yakaladı.)

**Aynı makaleye iki farklı yoldan bak** (PDF ve ar5iv/HTML). Bu projede üç ayrı
uydurma böyle yakalandı.

**Niteliksel iddiaları da sayılar kadar etiketle:** "tam metinden" /
"özetten, doğrulanmadı".

---

## 6. Kod değiştirirken

**Çalışan dosya her zaman `sifirdan.py` kalır.** Eski hali
`sifirdan_vN_DONDURULDU.py` diye arşivlenir. Yeni dosyaya V2 adı VERİLMEZ —
`analiz.py` `import sifirdan` yapıyor ve tek eğitim çekirdeği, D–A kıyasının
geçerlilik şartı.

**Her yama üç testten geçer:**
1. Bayrak kapalıyken **bit düzeyinde no-op** (yamasız sürümle karşılaştır:
   eğri alanları + tüm ağırlık tensörleri)
2. Bayrak açıkken etki **gerçekten** var mı (ölç, varsay ma)
3. Hatalı girdide **assert'ler ateşliyor** mu

**Tanı fonksiyonlarını unutma.** İleri geçişi kendi hesaplayan yerler
(`dikkat()` gibi) yeni davranışı görmezse **yanlış rapor verir**.
> (`dikkat()` maskeyi görmüyordu; maskeli modelde var olmayan dikkat
> raporlayacaktı.)

**Sabitleri elle kopyalama.** `T_LEN_ON = 8` gibi kopyalar bayatlar.

**Türetilmiş bir büyüklük, kontrol akışı için oynadığın parametreye bağlı
olabilir.** Bir parametreyi "sadece nerede duracağımı söylemek için"
değiştiriyorsan, ondan TÜREYEN her şeyi ara.
> (`warm = max(10, STEPS // 20)`. D3.1'i 5.000'lik parçalara bölünce ilk
> parçanın ısınması 6.000 değil **250** adım oldu; aynı tohum başka yörünge
> izledi ve maskesiz faz referans kol A'nın tekrarı olmaktan çıktı — 10.000'de
> `comp` 0.016'ya karşı 0.074. `WARM_OF` ile ısınma artık TOPLAM koşudan
> hesaplanıyor.)

**İsim gölgeleme.** Bir fonksiyon adını değişken olarak kullanma.
> (`olc` fonksiyonu bir durum hücresinde listeye atanınca sonraki hücre kırıldı.)

**Sihirli sabit yazma, koddan al.** `ENT_OFF` 14 değil 16'ydı; sonuç absürt
çıktığı için yakalandı.

---

## 7. Colab / Drive altyapısı

**Colab'da YAMA YOK.** Akış tek yönlü:

```
YEREL  sifirdan.py duzelt + uc testi kos
  ->   GitHub'a it
  ->   Colab KLONLAR (rm -rf + git clone), uzerine hicbir sey yazmaz
```

Colab'daki yama iki ayrı kaynak yaratıyordu (`/content/_d` yamalı,
`/content/_a` yamasız) ve hangisinin koştuğu belirsizdi. Klon tek kaynak.
**İmza YAPISAL bir çapa olmalı, mesaj metni DEĞİL.** 13 Eylül: `kosu.py`'deki
bir hata mesajını yeniden yazdım, imza o cümleyi arıyordu ve kapı boşuna
ateşledi — kod eski değildi, **imza** eskiydi. Fonksiyon adı (`def bitti_mi`),
çağrı (`K.bitti_mi(ad, alt`), atama (`_SEC = [x.strip()`) seç; cümle seçme.

Bunun yerine **imzaları assert et**: kodda bulunması gereken string'leri
listeye koy, klonda yoksa dur. *"Yereli düzelttim ama itmeyi unuttum"*
sessizce değil, ilk hücrede patlar.

**ANALİZ modülünün de yamalı olduğunu doğrula, sadece eğitimin değil.**
Eğitim klonunu (`/content/_d`) imzayla doğruluyorduk; defterdeki `import
sifirdan` başka bir klondan (`/content/_a`) geliyordu ve **yamasızdı**.
`blk.mask_key = 1` ölü bir öznitelik yarattı, hata vermedi, sessizce yanlış
ölçtü. **Her oturumda `S.__file__` bas ve beklenen yamanın kaynakta
olduğunu assert et.**
> (Maskeyle eğitilmiş D ve E1, maske kapalıyken değerlendirildi. İki belgedeki
> sayılar düzeltilmek zorunda kaldı.)

**Müdahalenin ETKİSİZ olması da bir arıza belirtisidir.** Bir maske/ablasyon
taramasında bütün adaylar tam `+0.000` verdiyse, bulgu "etki yok" değil,
"ölçüm bozuk"tur. Taramaya `assert etki_var` koy.

**Dosya adı biçimini varsayma, `glob`'dan oku.** `snap_A_s0_020000.pt` sıfır
dolguluydu; `f"..._{20000}.pt"` hiçbir şey bulamadı ama 190000 bulundu (zaten
6 haneli), yani hata **kısmen** göründü — en sinsi hali.

**Kullanıcıdan yapabileceğin şeyi isteme.** Colab'a bağlısın: hücreyi sen
koy, sen çalıştır.
> ("sifirdan.py'yi Drive'a koy" denildi; gereksizdi.)

**`ps | grep <betik>` kendi komut satırını yakalar** → yanlış "çalışıyor" der.
**Nabız dosyasının yaşına bak.** (Aynı tuzağın `pkill` hali bir koşuyu öldürdü.)

**Alt sürece TEMİZ ortam ver.** Önceki hücrelerin `os.environ`'a yazdığı
`MEM_AT` vb. sızıp assert patlatır.

**Büyük dosyayı Drive'a `.tmp` + `mv` ile yaz.** 102 MB'lık sürdürme paketi
düz `cp` ortasında runtime ölürse Drive'da yarım kalır ve `/content` silindiği
için tek kopya odur.

**Sürücü ASLA hücrenin içinde koşmaz.** Karar mantığı (dur → ölç → seç →
devam) girdiği anda onu hücreye gömmek cazip geliyor; gömülürse çekirdek
kilitlenir ve ilerlemeye bakmak, yedeği kontrol etmek, koşuyu durdurmak
imkânsız hale gelir. Sürücü `Popen` ile **ayrı süreç**, mantığı **depoda bir
dosya**. Defter yalnız kapıları açar ve başlatır.
> (D3.1'de gömüldü; 15.000 adım boyunca ne bakılabildi ne durdurulabildi.
> D, E1, E2, D3'te sürücü hep ayrı süreçti ve böyle bir sorun hiç olmadı.)

**İlerleme raporu bir DOSYADIR, bir komut değil.** Bakıcı 5 dakikada bir
üretir, Drive'a yazar. Rapor üretmek için çekirdeğin boş olmasına, GPU'ya ya
da hesap yapmaya gerek olmamalı — yoksa tam ihtiyaç duyduğun anda (koşu
kilitliyken) çalışmaz.

**Şablon:** depoda `sablon/` (`kosu.py`, `pencere.py`, `bakici.sh`,
`arsivle.py`, `rapor.py`) + `uret_sablon.py`.

```
yeni deney =  uret_sablon.py'deki DENEYLER sozluguna bir kayit
            + deney/<ad>.py  (deneye OZEL karar mantigi)
            + python uret_sablon.py SABLON_deney.ipynb <ad>
```

**Defterde elle düzenlenen alan YOKTUR.** Eskiden HÜCRE 0 önceki deneyin
ayarlarını taşıyordu ve düzeltmeyi unutan kişi sessizce önceki deneyi
başlatıyordu. `IMZA` artık `{dosya: [imza]}` — yalnız `sifirdan.py` değil,
**iskele de** denetlenir; itmeyi unutursan HÜCRE 3 durur.

### Dosyalama düzeni — ADI VARSAYMA, buraya bak

Her ad bir **maske** taşır (deney adı / kol / tohum / adım). Maskesiz ad
koyma: iki deneyin çıktısı karışır ve hangisinin hangisi olduğu sonradan
çıkarılamaz.

```
DEPO (GitHub, Colab bunu klonlar)
  sifirdan.py                  EGITIM CEKIRDEGI -- adi ASLA degismez
  analiz.py                    kosu sonrasi tani ('import sifirdan')
  sablon/                      deneyden BAGIMSIZ altyapi
  deney/<ad>.py                deneye OZEL karar mantigi (surucu)
  uret_sablon.py               defteri URETIR; DENEYLER sozlugu icinde

COLAB
  /content/kod_<ad>/           GitHub klonu   -- uzerine YAZILMAZ
  /content/calis_<ad>/         calisma (hizli disk)
  /content/drive/MyDrive/deney_<ad>/    KALICI ayna

CALISMA / AYNA icinde
  cikti_<kol>/                 her kol kendi klasorunde
                               (tek kollu eski kosularda duz 'cikti')
      snap_<kol>_s<tohum>_<adim:06d>.pt     anlik goruntu (fp16)
      surdur_<kol>_s<tohum>.pt              CANLI surdurme paketi
      egri_<kol>_s<tohum>.json              olcum egrisi (TUM alanlar)
      dikkat_/kv_<kol>_s<tohum>_<adim>.npz  tani tensorleri
  sur/<kol>/surdur_<kol>_s<tohum>_<adim:06d>.pt   ADIM ADLI arsiv
  log/  egitim_<deney>_<kol>_<adim:06d>.txt / surucu_<deney>.txt /
        bakici_<deney>.txt
  ham/                         analiz ciktilari (json/npz)
  konfig.json  durum.json  konfig_giris.json  RAPOR.txt

BELGE (yerel, depoda degil)
  belge/onkayit/ONKAYIT_<DENEY>_<KONU>.md   kosudan ONCE, DEGISMEZ
  belge/bulgu/BULGU_<KONU>.md               sonuc
  belge/fikir/                              henuz deney olmamis
  arsiv/cekirdek_eski/                      dondurulmus eski surumler
```

**Adım numarası her yerde 6 hane sıfır dolgulu.** `f"..._{20000}.pt"`
yazmak hiçbir şey bulmaz ama `190000`'i bulur (zaten 6 hane) — hata
**kısmen** görünür, en sinsi hali. `glob`'dan oku.

**Her koşu ham veriyi MAKSİMUM yoğunlukta bırakır** — bugün işe yaramayan
da dahil. Gerekçe ve liste §9'da; disk ucuz, koşu pahalı.

**Kurtarma = HÜCRE 0-4'ü tekrar koşmak.** Ayrı bir `kurtar.py` yok ve
gerekmiyor: kapılar imzayı doğruluyor, `durum.json` kaldığı yeri biliyor,
adım adlı paketler Drive'da, HÜCRE 6 hangi adıma dönülebileceğini listeler.
**Bitmiş bir kolu geri almak için `durum.json`'daki `<kol>_bitti` bayrağını
SİL** — sürücü kolu atlarken ona bakar, `adim`'a değil.

**Sürdürme paketi her ölçümde ÜZERİNE yazılır** → adım adlı kopya
alınmazsa geçmiş bir adıma dönülemez.
> (A'nın 5.000–170.000 arası örnek-başına kaydı bu yüzden kayboldu.)

**Yerleşik çözümler** (hepsi `sablon/` içinde, tekrar yazma):
- `arsivle.py` paketi adım adlı arşivler. **Adımı KOPYADAN okur**, kaynaktan
  değil — tersi yarış: kopya yeni ağırlıkları taşıyıp eski adımın adını alır.
- `kosu.konfig_kapisi()` sürdürme paketindeki `cfg`'yi beklenenle
  karşılaştırır (yanlış `MASK_KEY` ile sessizce sürdürmeyi önler).
- `kosu._commit_kapisi()` Drive'dan geri yüklenen yarım iş başka bir
  commit'le üretilmişse durdurur.
- `kosu.bitti_mi()` `durum.json` "bitti" derken ölçüm noktaları diskte
  yoksa durdurur.

---

## 8. Maliyet tahmini

Tahmin etme, **ölç**. Bu projede "1 saat" denen iş 1 dakika sürdü, "105 MB
indirme" gereken şey zaten Drive'da mount'luydu.

---

## 9. Ham veri — bugün işe yaramayanı da kaydet

**Bu oturumun en büyük bulgusu, planlamadığımız bir veriden çıktı.**
Ağırlık ortalaması sonucu (ENT 0.050 → 0.310) yalnızca ara kontrol noktaları
kaydedildiği için mümkün oldu. Sadece nihai modeli saklasaydık o bulgu
**hiç var olmayacaktı** — ve onları kaydederken böyle bir analiz aklımızda yoktu.

> **Kural: yapacağın analiz henüz icat edilmedi. Buna göre kaydet.**

Her koşu şunları bırakmalı:

```
ara kontrol noktalari    her olcumde (fp16 yeter)     <- ortalama/film analizi
ornek-basina kayit       tahmin dahil, sadece dogruluk DEGIL
egrinin TUM alanlari     secilmis 5 sutun degil, hepsi
dikkat / aktivasyon      ozet tensorler
veri bolmeleri           veri.npz (hash'lenebilir olsun)
konfig + kod surumu      commit ya da md5
```

**Neden "sadece doğruluk" yetmez:** kısayol bulgusu (%73) ancak **tahmin edilen
varlık** kayıtlıydığı için çıkarıldı. Doğruluk sütunu tek başına "model yanlış
yapıyor" der, "şöyle yanlış yapıyor" demez.

**Neden eğrinin tümü:** `ent_shortcut` alanı 24 ölçüm boyunca kayıtlıydı ve
**saatlerce hiç bakılmadı**. Bakıldığında teşhisi tek başına o verdi.

**Üzerine yazma.** Sürdürme, örnek-başına tabloyu ve paketi sıfırlıyor.
Adım adlı kaydet, yoksa geçmiş silinir.
> (A'nın 5.000–170.000 arası örnek-başına kaydı böyle kayboldu; artık o
> aralıkta örnek-bazlı analiz yapılamıyor.)

**Disk ucuz, koşu pahalı.** 2 TB Drive var; bir koşuyu tekrarlamak 85 dakika.
Şüphe varsa kaydet.

---

## 10. Adil hakemlik — koda ve MANTIĞA

İkisi ayrı iş. Kod doğru olup mantık yanlış olabilir; bu oturumda oldu.

### 10a. Koda hakemlik

§6'daki üç test **zorunlu**, ek olarak:

- **Ölçümü bilinen bir değere karşı doğrula.** Hesapladığın "doğru cevap oranı"
  eğrideki `ent` ile eşleşmeli. Eşleşmiyorsa kod bozuk, sonuç değil.
  > (Hata ayrıştırması ilk koşuda %0.1 verdi, eğri %13 diyordu → `ENT_OFF`
  > yanlıştı. Bu kontrol olmasaydı yanlış tabloyu raporlayacaktım.)
- **Tanı fonksiyonları yeni davranışı görüyor mu?**
- **İsim gölgeleme, elle kopyalanmış sabit, sızan ortam değişkeni.**

### 10b. Mantığa hakemlik

Her iddia **aynı mesajda** şu üçüyle gelir:

```
1. Bunu NE curutur?          (koşudan ONCE yazilir)
2. En yakin ALTERNATIF aciklama nedir, ve hangi KONTROL onu eler?
3. Hangi kismi OLCULDU, hangi kismi CIKARIM?
```

**Kontrolü sonraya bırakma.** Kontrol kolu tasarımın parçasıdır, ek değil.
> (K kolu bir bulguyu öldürdü — "kısayol öğrenmeyi yavaşlatıyor" — ve
> diğerini kurtardı. K olmasaydı ikisini de yanlış raporlayacaktık.)

**Yanlışlamayı deneyin içine göm.** Ölçütlerden biri "şu düzelmemeli" olsun.
> (Özgüllük maddesi: ENT2 hareket ederse sonuç geçersiz. Tuttu ve sonucu
> güçlendirdi.)

**Okuma ile ölçümü ayır.** Katman eğrilerinden çıkarılan "devir teslim
çalışmıyor" bir **okumaydı**; nedensel test yapılmadı ve sonradan **yanlış**
çıktı (gerçek mekanizma kısayoldu).

**Kendi tasarımına saldır, kullanıcı sormadan.** Zayıf yerleri aynı mesajda say;
özellikle "tasarımı tamamen geçersiz kılabilecek" olanları ayrı başlıkta.
> (R1: maske, ENT'teki hatayı inceleyerek bulundu → mimari test kümesinden
> türetildi. Bunu kullanıcı sormadan yazmak doğruydu.)

**Adil olmak = savunmamak değil, tek yanlı saldırmak da değil.** Karşı
argümanın en güçlü halini kur, sonra dürüstçe cevapla, ve kazandığında kabul et.
> (Kullanıcı "K'da da hızlanma var" ihtimalini sordu; doğru çıktı ve bulgu
> geri alındı. Savunmaya çalışmak yanlış olurdu.)

**Bulgu senin lehine çıktığında daha dikkatli ol.** Bu oturumda bütün geri
almalar, sonucun *iyi* göründüğü anlardan sonra geldi.

**Bir eşik başka bir rejime taşınmaz — taşıdığını ÖLÇ.** D3.1'in arama eşiği
(`sinyal >= 0.03`) φ=3.03'te 40.000. adımda ateşliyordu; φ=5.06'da **ilk
durakta** (5.000) ateşledi, çünkü yüksek φ oraya 8 kat hızlı varıyor. Tetik
transfer etti ama **seçim etmedi**: Aşama B'nin kazananı orada `-0.0010`
verdi (D3.1'de `+0.0565`). 10.000'den itibaren istikrarlı `parça 1`, artan
skorla. Ders: iki aşamalı bir prosedürde **her aşamayı ayrı doğrula**;
tetiğin çalışması seçimin de çalıştığı anlamına gelmez.
> (Prosedürde "Aşama B'nin skoru pozitif olmalı" koşulu yoktu — D3.1'de
> kendiliğinden pozitif olduğu için fark edilmemişti.)

---

## 11. Kayıt

**Terminal çıktısına güvenme, diske yaz.** Her analiz JSON + NPZ (+ mümkünse
PNG) bırakmalı ve **tek komutla yeniden üretilebilmeli** (`analiz_A.py` gibi).

**Hatayı silme, işaretleyerek düzelt.** Belgede yanlış bir iddia varsa üstünü
çizip düzeltmeyi yaz; sessizce değiştirme.

**Önceden kayıt değiştirilmez.** Sonuç görüldükten sonra seçilen her okuma
`[SONRADAN SEÇİM]` diye etiketlenir.

---

## 12. Bulgu bildirirken — uydurmayı yakalama

Bu madde, 13 Eylül'de üretilen 31 "bulgu"dan 6'sının yanlış, 9'unun şişirilmiş
çıkmasından doğdu. Ayırt edici işaret kesindi ve kolay:

**Ayakta kalanlar İKİLİ idi** (var/yok, satır numaralı).
**Yanlış çıkanların hepsinde bir komuttan gelmemiş SAYI ya da SONUÇ iddiası vardı.**

Her bulgu üç şeyle gelir, yoksa bulgu değil:

```
1. dosya:satir        iddianin dayandigi YER
2. yanlislayan komut  "bunu curutecek tek komut" — ve KOSULMUS olmali
3. ariza zinciri      X olursa Y yazilir, Z yanlis okunur
```

**Sayı geçiyorsa, o sayının geldiği komut gösterilir.** Gösterilemiyorsa sayı
yazılmaz; yerine "bilmiyorum, şu komut söyler" yazılır.
> ("~3 geri dönüş noktası" denildi; ölçüm depoda duruyordu — 120.000 adım
> 66 dakika, yani ~12 nokta. `grep dk belge/bulgu/*.md` yetiyordu.)
> ("Drive dolsa nabız atmaya devam eder" denildi; `grep nabiz bakici.sh`
> bir saniye sürüyordu ve tersini söylüyordu — nabız Drive'a yazılıyor.)

**Ciddiyet etiketi arıza zinciri olmadan yazılmaz.** Zinciri kuramıyorsan
"Yüksek" diyemezsin.

**BEKLENTİ EMİR DEĞİLDİR — ama Claude ikisini ayırmıyor.**
Bu maddenin ilk hali *"sayı sipariş edilmez"* diye yazılmıştı ve yanlıştı:
kullanıcı hiçbir zaman *"10 tane bul"* demedi, *"10 tane bulursun"* dedi.
TAHMİNDİ. Claude 10 üretti. Ardından *"bunlar hata değil dersin"* dendi ve
6'sı geri alındı. Yani kusur sayının istenmesinde değil:

> **Claude, karşı tarafın beklediğini sandığı çıktıyı üretiyor.**
> Beklentinin söylenmiş olması bile gerekmiyor; *"kesin bulursun"* yetti.

Bu, "sayı istenirse sayı üretir"den daha kötüdür, çünkü beklenti her yerdedir
ve çoğu zaman telaffuz edilmez.

**Karşı önlem — sayı EN SON sayılır.** Bulgu adedi, bulgular yazılmadan önce
ağza alınmaz. Önce her bulgu için `dosya:satır` + yanlışlayan komut yazılır;
kaç tane kaldıysa o kadardır. Beklenen sayı ile çıkan sayı tutmuyorsa,
düzeltilecek olan çıkan sayı değildir.

Doğru soru hiçbir zaman "kaç tane" değil:
**"sessizce yanlış SAYI üretebilecek olan hangisi?"**
> (31 bulgunun yalnız 2'si bu soruya cevap veriyordu.)

**Düşmanca okuma değil, mekanik kontrol.** Belgelerdeki 0.365 / 0.3607
çelişkisi bütün belgelerde aynı sayıyı `grep`'leyerek bulundu — yargıya
bağlı değil, yanlışsa mekanik olarak yanlışlanır.
