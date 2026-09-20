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

### Adım 3 — `z_2 -> z_3`, girdi `-ın`, hedef `kardeş`   (SIRADA)
