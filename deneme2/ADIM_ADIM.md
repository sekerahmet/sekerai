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

### Adım 2 — `z_1 -> z_2`, girdi `Yıldız`, hedef `-ın`   (SIRADA)
