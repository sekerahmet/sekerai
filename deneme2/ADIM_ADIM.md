# Denklemin adım adım denetimi — ve PARK EDİLENLER

Kullanıcı, 20 Eylül 2026:

> *"bu da özet olmuş, adım adım gidelim derken tüm adımları tek
> seferde değil. her bir adımı ayrı ayrı. ilk olarak bu adım
> matematik ve kodda paralel mi. ne istedik, hedefimiz neydi ve
> başarılı işe yarayan bir şey oldu mu."*

> *"buldukça bir yere not et sonra dönüp bakarız. önce not et
> sonra adım 2 geçelim."*

`DENKLEM.md` §0'daki çözülmüş örnek, sekiz adımlık bir cümle. O
tabloyu **adım adım** denetliyoruz. Her adımda dört soru:

```
1  DENKLEMLER paralel mi -- matematik satiri ile kod satiri
2  NE ISTEDIK             -- o adimin hedefleri, tek tek
3  OLDU MU                -- her hedef icin CALISTI / CALISMADI
4  ACIK KALEM             -- adim BITMEDEN kapatilir
```

**Kural:** bir adım, açık kalemi kalmadan kapanır. Ama denetim
sırasında **o adıma ait olmayan** bir şey bulunursa düzeltilmez —
aşağıya park edilir, sonra dönülür. Yoksa her adım bir sonrakine
dallanır ve hiçbiri bitmez.

---

## PARK — bulundu, sırası gelmedi

### P1  `K_TAM = 451` ama sözlük **444** birim

```
BULUNDU   adim 1 sirasinda, kapi 31'in ciktisinda
          "444 birimin 331'i varlik birimi; hepsi TAM SO(32)
           (K_TAM=451, acik sinif 0)"

NE        `Birim.kapali(k_tam)` en sik k_tam birimi tam SO(D) yapar,
          kalani tek duzlem. K_TAM = 451 > n = 444 oldugu icin
          HEPSI tam SO(D) aliyor: acik sinif BOS, §4.3'un frekans
          ayrimi ATIL.

DAVRANIS  SU AN DOGRU. K_TAM = 444 ile birebir ayni sonuc.

RISK      Deger artik adini dogrulamiyor. 451, ESKI sozlugun
          tamamiydi (`ek_14` yeniden yazilmadan once). Yeni sozluk
          444. Sozluk 451'i GECERSE frekans ayrimi kendiliginden,
          keyfi bir kesimle geri gelir ve hicbir kapi bunu soylemez:
          kapi 31 role bakiyor, varlik birimleri yine ilk 451'in
          icinde kalabilir.

NIYE SIMDI DEGIL
          Adim 1'in (z_0 -> z_1) konusu degil. Operator secimi
          §4.3'un konusu.

NEREYE BAKILACAK
          ayar_14.py:81   birim_14.Birim.kapali
          DENKLEM.md §4.3, §13/K7   kapi 31
```

### P2  §1'in "birebir" bloğu `t_j`'yi yazmıyor

```
BULUNDU   adim 2 sirasinda
NE        `yol()` her adimda t'yi hesaplayip donduruyor:
              t = t + 1;  t.masked_fill(vur, 0)
          yani  t_j = vur_j ? 0 : t_{j-1} + 1   (son capadan beri adim).
          §1'in "kodun hesapladigi sey, BIREBIR" blogunda YOK.
YUK TASIYOR MU
          Evet, saat acikken: `_kos` hedefi `H[t]` diye indeksliyor.
          Saat VARSAYILAN KAPALI oldugu icin su an atil.
NIYE SIMDI DEGIL
          Adim 2'nin (R_Yildiz'in operator olup olmadigi) konusu
          degil; §1/§6 butunlugunun konusu.
NEREYE BAKILACAK
          DENKLEM.md §1 blogu, §6 (saat);  model_14.yol, model_14._kos
```

---

## KAPANAN ADIMLAR

### Adım 1 — `z_0 -> z_1`, girdi `Cem`, hedef `Yıldız`   (KAPANDI)

```
1  DENKLEMLER PARALEL   §1'in alti satirinin altisi da kodla birebir.
                        Not: `z_0` tanimi v1'de YOKTU, koddan yazildi
                        -- teori kodu ongormedi, kod teoriyi tamamladi.

2-3  NE ISTEDIK / OLDU MU
   H1 |z_0| = 1, izometri ilk adimdan gecerli
      CALISTI.  |z_1| = 1,0000. Bos kontrol degil: C serbest
      parametre, kureden cikarsa iddia duser -- kod c_k = C_k/|C_k|
      ile kapatiyor (kapi 16: C x2,5 -> |z| = 2,5).
      KOD TEORIYI DUZELTTI, DENKLEM §1'e yazildi.

   H2 bastaki "cop" durum ILK CAPADA silinsin  (§9.5, ilk surum)
      CALISMADI -- ve mekanizma HIC YOKTU. Silmeyi yapacak olan sey
      capanin kendiliginden tetiklenmesiydi: adim 1'de s = 0,5482,
      esik 1 - r^2/2 = 0,969.
      Iddia v2'de §9.5'ten cikarilmisti ama KODDA UC yorumda ve
      §5.1/G'nin atfinda duruyordu -- belge duzelirken kod yorumu
      bayat kaldi.  HEPSINDEN KALDIRILDI (b3923d6).

   H3 okuma `Yildiz`i gostersin
      Egitilmemis model: cos -0,3182, sira 353/444.

4  ACIK KALEM  A3 (cop onek puanlaniyor)  ->  KAPANDI (137b197)
      Capayi adim 1'de tetiklemek ELENDI: r 0,25 -> 0,95 gerekirdi,
      o yaricapta 2048 baslik kureyi 1,05 kapliyor = sonlu otomat.
      SECILEN: ISINMA = 4, kayip konum 4..23'u puanliyor. Deger
      `atla`dan cikiyor (j = i mod atla), W=4 dort kalinti sinifini
      da tam 5 konumda birakan TEK deger; hicbir gecis dusmuyor.
      DENKLEM §5.2, kapi 32, §13/K10.
```

---

### Adım 2 — `z_1 -> z_2`, girdi `Yıldız`, hedef `-ın`   (KAPANDI)

```
1  DENKLEMLER PARALEL   Evet, ve bu adimda fazladan bir sey soyluyor:
                        kodda j=2'yi j=1'den ayiran HICBIR YOL YOK.
                        Dongu govdesi j cinsinden tekduze, denklem de.
                        Ilk "siradan" adim ozel bir dal kullanmiyor.
                        Eksik: §1'in "birebir" blogu t_j'yi yazmiyor -> P2

2-3  NE ISTEDIK / OLDU MU
   H1 |z_2| = 1, izometri BILESKEDE (z_2 = R_Yildiz R_Cem z_0)
      CALISTI.  |z_2| = 1,0000.

   H2 R_w BIRIMIN ozelligi -- konumdan ve baglamdan bagimsiz.
      §2/iddia 2'nin ("gorulmemis bilesim gorulmus parcalara iner")
      dayandigi sey. ASIL IDDIA.
      YARISI CALISTI, ve calismayan yarisi ASIL IDDIA.
      Matris gercekten birimin: F.embedding(X[:, j-1], Rd), satir
      birimden geliyor konumdan degil. AMA §0 tablosunda `Yildiz`
      IKI kez girdi oluyor ve ayni matrisin etkisi ayni degil:
          j=2   |Pz| 0,9638 -> 0,9418   oran 0,9772
          j=7   |Pz| 0,6448 -> 0,5338   oran 0,8279     7,5 KAT
      TEK CUMLE: R_w tam durum uzerinde operatordur, OKUNABILIR
      kisim uzerinde DEGIL.  (Pi izometri degil -- §4.1)
      Yeni kusur DEGIL: §4.1'in `!!` blogu bunu soyluyordu. Adim
      2'nin ekledigi sey, sizintinin BIRIMIN degil DURUMUN ozelligi
      oldugunun kaniti -- yani "o birimin donmesi kotu, daha iyisini
      ogrenir" turu butun PER-BIRIM cozumler ELENDI.
      DENKLEM §2.1'in "NE DEMIYOR" blokuna yazildi.

   H3 kok -> ek gecisi ("kok ayri token, ek ayri token"in mimarideki
      ilk karsiligi)
      cos 0,6327, sira 18/444 -- sekiz adimin acik ara en iyisi
      (otekiler 353 52 159 102 348 56 237). AMA TEK CEKILIS,
      egitilmemis model. HUKUM YOK. Sinamak icin gereken sey:
      ek hedeflerinin sira dagilimi ile kok hedeflerininki.

   EK GOZLEM (yazili sayilardan, kosturmadan)
      |Pz| 1,0000 -> 0,5622 boyunca: okunabilir kutlenin %43,8'i
      gizliye gecti. Adim adim oran:
         Cem .9638  Yildiz .9772  -in .9169  kardes .8391
         -i .9965   Ceren .8930   Yildiz .8279  -dir 1.0532
      `-i` ve `-dir` neredeyse hic kaybettirmiyor, ADLAR kaybettiriyor.

4  ACIK KALEM   §0'in kayip satiri ISINMA=4'ten ONCE hesaplanmisti.
      Kullanici: *"bence not düş mantıklı"*  ->  B secildi.
      Hicbir sey kosturulmadi; satirin altina konum 1..8 uzerinden
      oldugu ve ilk gercek kosuda tazelenecegi yazildi. Tablonun
      gerisi (|z|, |Pz|, s, cos, sira) ileri gecisin sayilari,
      kayip dilimine bagli degil -- etkilenmiyor.
```

---

### Adım 3 — `z_2 -> z_3`, girdi `-ın`, hedef `kardeş`   (KAPANDI)

```
1  DENKLEMLER PARALEL   Evet. Ama "paralel" bir seyi ORTUYOR: kod
                        `-ın`a `Cem`e davrandigindan hic farkli
                        davranmiyor -- ayni SO(32), ayni kayip
                        agirligi. EK ile KOK mimaride ayirt
                        EDILMIYOR, ve bu bir karar degil, P1'deki
                        bayat K_TAM yuzunden FARKINDA OLMADAN boyle.

2-3  NE ISTEDIK / OLDU MU
   H1 |z_3| = 1                   CALISTI, 1,0000
   H2 "ek ayri token" -> iliski bir OPERATOR olsun
      BEKLENTI YANLIS YERE KURULMUS. Cumlenin yapisi:
         Cem Yıldız  -ın   kardeş   -i   Ceren Yıldız  -dır .
          <sahip>  SAHIPLIK <ILISKI> IYELIK  <cevap>   kopula
      Iliskiyi KOK tasiyor (`kardeş`); `-ın` rolu isaretliyor.
      Yani umut R[kardeş]'e dusuyor, R[-ın]'a degil.
   H3 okuma  sira 52/444 (egitimsiz);  300 adim sonra 10/444

4  BULGU J -- ISPAT, egitim/veri/agirlik YOK
   `uye`yi DUSUREN bir gradyan adimi hedeflerin sirasini:
       k=1 %0,0   k=2 %0,2   k=3 %3,7   k=5 %9,7   k=11 %19,8
   k=1 TEOREM: hedefin kazanci eta, rakibinki eta<p_t,p_c> <= eta.
   ILK HIPOTEZIM YANLISTI ("kayip sirayi bozuyor"); ispat duzeltti.
   DOGRUSU: kayip SULANMIS. Taban uye >= 2 - 2/sqrt(k).
   DENKLEM §5.1/J, kapi 33.

   OLCUMLE DOGRULANDI, bizim cumlemizde: adim 3'te listenin tepesi
       bölüm(0,905)  tez(0,884)  ...  *kardeş 10.
   `Cem Yıldız -ın`den bu 20 cumlede CIKAN 6 iliski:
       anne  bölüm  danışman  kardeş  tez  yaşadığı
   Yani tepedeki rakipler KARDES ILISKILER. Model yanlis bir sey
   ogrenmemis -- "buraya bir iliski gelir"i dogru ogrenmis,
   HANGISI oldugunu secemiyor. Secemez de.

5  ACIK KALEM -> delta (eski A2).  KAPANDI: DEGISMIYOR.
   Once YANLIS gerekceyle actim ("adim 1'in sirasi 4"): ISINMA=4
   ile konum 1,2,3 PUANLANMIYOR, `dis` oraya bakmiyor bile.
   PUANLANAN her konumda sira 1.  Geri aldim.
   Sonra DOGRU gerekceyle olctum (calisan model, 20 cumle):
       HEDEFIN uzakligi   %50 0,1169   ort 0,1406
       PENCERE DISI       %1  0,4530   en kucuk 0,3460
   Hedef 0,117, en yakin rakip 0,346 -> itmeye GEREK YOK.
   Buyutmenin bedeli 0,8 -> %31,9   1,0 -> %75,8 itilir; d=8'de
   20 konumluk yoldan 319 nokta uzaklastirilamaz.
   Tam korpustaki 0,9160 basarisiz CEKME demek (uye 0,839 -> k~3),
   delta onu duzeltemez.   §13  A2 -> K11.

6  YAN URUN -- §5.1/K, 20 cumle sinavinin kendisi gosterdi
   adim  750: kayip 0,231  1.sira %98,2  cos 0,9904   <- TEPE
   adim 1500: kayip 0,204  1.sira %95,9  cos 0,9845
   Kayip DUSERKEN sira KOTULESIYOR; kazanc `uye`den degil
   `kod`+`duzen`den. Agirliklar (a1,a2,a3) okumayi GERI CEKIYOR.
   §13/A1 guncellendi -- artik olculmus bedeli var.
```

---

### Adım 4 — `z_3 -> z_4`, girdi `kardeş`, hedef `-i`   (KAPANDI)

```
1  DENKLEMLER PARALEL   Evet. Bu adimin ayricaligi: ISINMA=4 yuzunden
                        kaybin BAKTIGI ILK konum. Dort terim de
                        (uye, dis, kod, bag) buradan basliyor.
                        Adim 1-3 taniydi, adim 4 EGITIMIN KENDISI.

2-3  NE ISTEDIK / OLDU MU
   H1 |z_4| = 1                      CALISTI, 1,0000
   H2 ilk PUANLANAN konumda kayip ise yariyor mu
      CALISTI. 300 adim sonra sira 159 -> 1, cos 0,9892.
      Tepede *-i(0,989), ikinci Fatma(0,799) -- hata bicimi
      UYUM KARISIKLIGI DEGIL.
   H3 unlu uyumu (`kardeş` -> `-i`, sozlukte 7 iyelik yuzeyi var)
      KAPSAM DISI -- kullanici karari, 21 Eylul:
      *"i ü vs odaklanmak cok onemli bir hata degil sonucta bir ek
        gelmesi gerektigi belli. bizim tokenizer da problemliydi.
        onu bu modelde bakmak dogru olmaz."*
      (Veri hazirdi: 9 kok+iyelik cifti, 5 ayri yuzey, 7 aday,
       rastgele %14,3. Sirasi gelirse buradan kurulur.)

4  ACIK KALEM -> B (capayi uyandirmak).  PARK EDILDI, asagida.
```

---

## PARK (devam)

### P3  ÇAPA hiç tetiklenmiyor — ve eğitimden sonra GÜVENLİ mesafede

```
BULUNDU   adim 4'te. Egitilmis modelde s_j en yuksek degerini adim
          4'te aliyor: 0,9447.  Esik 1 - r^2/2 = 0,96875.

NE        §3 mimarinin ASIL iddiasi ("durum periyodik olarak bir koda
          oturur, gorulmemis bilesim gorulmus parcalara iner") ve
          §3.2 (reddetme = kod defteri uyeligi) ikisi de capaya
          bagli. Capa hic atesmedigi icin IKISI DE UYKUDA.

ADIM 1'DE ELENMISTI, AMA O ELEME EGITILMEMIS MODELE AITTI:
              tetiklenmesi icin   yari aci   2048 baslik kureyi
  adim 1, EGITILMEMIS  r >= 0,9506   56,8°    1,1e+00   <- COKME
  adim 4, EGITILMIS    r >= 0,3326   19,1°    1,5e-13   <- GUVENLI
  su anki r = 0,25                   14,4°    2,6e-17

PENCERE (kagit)   0,3326 <= r < 0,4156
  ust sinir: top yari acisi < donme acisi olmali, yoksa R c_k ayni
  topta kalir ve SABIT NOKTA olur. Donme acisi n^(-1/(d-1)) = 24,0°
  -> r < 2 sin(12,0°) = 0,4156.   (Kosu 21 Eylul: model "komsu
  araligi 0.419" basti, hesapla tutuyor.)

!! KARSI KANIT YANLIS ADRESE YAZILMIS OLABILIR
  §3.1 capaya karsi tek kaniti "Şanlıurfa ×12, |Πz| sabit 0,193"
  diye gosteriyor. Ama AYNI BELGE §5.1/C'de "ortalama durum capa
  yaricapinin 2,25 katinda" diyor -- atesmeyen capa sabit nokta
  uretemez. Alternatif aciklama kodda yazili (`_itme` docstring):
  "acik sinif donmesi TEK DUZLEM; duzlem z0'a dik dusunce R z0 ~ z0
  kaliyor." O sirada K_TAM = 80 idi, yani varlik birimlerinin %93'u
  tek duzlemliydi. Muhtemelen ZAYIF OPERATOR, capa degil -- ve o
  zaten K_TAM = n ile duzeltildi.

EKSIK SAYI  OGRENILMIS donme acisi. Ust sinir BASLANGIC olceginden;
  `duzen = |a|^2 + |θ|^2` acilari egitim boyunca KUCULTUYOR. 20
  cumlede toplam 1207,73 -> 343,20, ama oradan ogrenilmis aciyi
  cikaramam: 444 birimin yalniz 73'u veride var, dususun cogu HIC
  KULLANILMAYAN birimlerin sifira cekilmesinden.

NIYE SIMDI DEGIL  Gercek korpustaki aciyi okuyamiyorduk: Drive'daki
  eski t0 agirliklari OLU 451'lik sozlukle egitilmisti. 21 Eylul'de
  yeni sozlukle t0 kosusu baslatildi (commit 7f166ea); `model_t0.pt`
  Drive'a yazilinca acilar ve `s` dagilimi YERELDE okunacak, sonra
  `r` kagitta secilecek.

NEREYE BAKILACAK  DENKLEM §3, §3.1, §5.1/C;  ayar_14.R_CAPA;
  model_14.yol (esik), model_14.donme

OLCULDU 21 Eylul (t0, commit 46d9b04) -- P3'UN ONCULU YANLISTI:
  EGITIMDE  capa %0,00 -> %5,44 -> %6,44 -> %7,74 -> %7,61
  URETIMDE  capa %0,00  (s en fazla 0,89, esik 0,969)
  Yani capa r=0,25'te KENDILIGINDEN uyanmis -- ama YALNIZ egitim
  durumlarinda. Modelin kendi urettigi durumlarda HIC atesmiyor.
  §3'un mekanizmasi tam IHTIYAC DUYULAN yerde yok.
  Ve uretimdeki TEKRARLAR capadan DEGIL: bkz P4.
```

### P4  VARLIK birimlerinin donmesi durumu OYNATMIYOR  (-> §13/A10)

```
BULUNDU   21 Eylul, t0 ciktisina GOZLE bakarken. Kullanici:
          *"bir kac ornek soru sorup cevaplara gozle bakarak bu
            olcumu teyit edebilir miyiz"* -- 20 soruluk tabloda
          tekrarlar gorundu (Recep x3, Kapadokya x4, Beykoz x3),
          sayida gorunmuyordu.

OLCULDU                n    ort frek   ort |a|   1-cos   KENDINE
          VARLIK     331     18.920    1,0012   0,1085   %73
          EK          34    137.734    1,7551   0,4559   %21
          otekiler    82     74.403    1,9169   0,4111   %22
          HEPSI      444     35.845    1,2184   0,1855   %60

NE        z = [p_w; 0] alinip R_w uygulaninca okuma YINE w'yi
          veriyor. Sozlugun %60'i kendi donmesinin SABIT NOKTASI,
          varlik birimlerinde %73.

UCUNU BIRDEN ACIKLIYOR
          tekrarlar / kapanmadi 0,6967 / BILGI tam 0,0000

SEBEP     gradyan FREKANSLA gelir, `duzen = a3*S|a|^2` HERKESE ESIT
          basar -> denge |a| frekansla olcekleniyor. Varlik birimi
          ort 18.920 kez geciyor, ek birimi 137.734 -- 7 kat.

ELENEN ACIKLAMALAR
          capa emici durumu   HAYIR: uretimde capa HIC atesmiyor
          tek duzlem (§4.3)   HAYIR: K_TAM = n, acik sinif BOS

KAPI 31 BUNU GOREMIYOR
          Operatorun SINIFINI siniyor ("varlik birimi TAM SO(D)"),
          ogrenilen BUYUKLUGUNU degil. Tam SO(32) olup acisi 0,46'da
          kalan donme, tek duzlemli olmaktan iyi degil.
          §13: "kapi teoreme degil IHTIYACA kurulur."

NEREYE BAKILACAK  DENKLEM §5.1/L, §13/A10, §4.3;  kapi 31;
                  ayar_14.A3_DUZEN;  model_14.kayip (duzen terimi)
```

---

### Adım 5 — `z_4 -> z_5`, girdi `-i`, hedef `Ceren`   (SIRADA)
    !! BILGI ADIMI -- sinavin sordugu TEK gecis bu.
