# Yol modeli — genel bir dil mimarisi

Kullanıcı, 20 Eylül 2026:

> *"bu model gerçek dil modeli öğrenmesinin bir kopyası gibi
> tasarlasak ama veri bizim verimiz gibi düşün. çünkü biz kendi
> verimize uygun bir mimari tasarlamıyoruz. genel bir dil mimarisi o
> ama bizim verimizde çalışıp çalışmadığını test edebiliriz."*

Bu belge **genel** bir mimari tarif eder. Grafımızdan, varlık
listemizden, ilişki şemamızdan hiçbir şey almaz. Korpusumuz mimarinin
kaynağı değil, **sınavı**.

---

> ## Bu belgenin ikinci sürümü — ve NİYE
>
> İlk sürüm *"hiçbir şey koşulmadı"* diye başlıyordu ve gerçekten hiçbir
> şey koşmadan yazılmıştı. Sonra kod koştu. Her çarpışmada **kodu**
> düzelttim, belgeyi değil: `DENKLEM.md` bir kez değişti, `model_14.py`
> beş kez. Sonuçta ikisi **20 yerde** ayrıştı ve 32 kapının 32'si
> geçerken bu ayrışmaların hiçbiri görünmedi — çünkü kapılar kodu
> sınıyordu, belgeyi değil.
>
> Kullanıcı, 20 Eylül: *"bir şeyi kodda yaptıysak ve doğru ise aslında
> onu matematik ile teorik olarak test etmedik demektir. kod çalışıyor,
> o zaman denklem içine açıklamalı olarak yazalım ve denklemi bir örnek
> üzerinden çözelim. kod doğru ve teoriyi düzeltiyorsa kalsın."*
>
> Bu sürüm o kurala göre yazıldı: **kodun hesapladığı şey**, her
> sapmanın nereden geldiği yazılı olarak. Ayrışmaların tam listesi ve
> hangisinde kimin haklı olduğu §13'te.

---

## 0. Denklemin bir örnek üzerinde çözümü

Korpustan gerçek bir cümle, **eğitilmemiş** modelle adım adım:

```
Cem  Yıldız  -ın  kardeş  -i  Ceren  Yıldız  -dır  .
D = 32   d = 8   K = 2048   r = 0,25   delta = 0,40

j   girdi w_{j-1}  hedef w_j   |z_j|   |Πz_j|    s_j   çapa  cos_hedef  sıra
1   Cem           Yıldız      1,0000  0,9638  0,5482   -     -0,3182    353
2   Yıldız        -ın         1,0000  0,9418  0,5633   -      0,6327     18
3   -ın           kardeş      1,0000  0,8635  0,6591   -      0,4713     52
4   kardeş        -i          1,0000  0,7246  0,5775   -      0,1604    159
5   -i            Ceren       1,0000  0,7221  0,5615   -      0,3089    102
6   Ceren         Yıldız      1,0000  0,6448  0,5818   -     -0,3081    348
7   Yıldız        -dır        1,0000  0,5338  0,5587   -      0,4520     56
8   -dır          .           1,0000  0,5622  0,4929   -     -0,0232    237

kayıp   üye 1,6561   dis 0,0000   kod 0,8643   bağ 0,0000   düzen 1207,73
        TOPLAM 2,6411
        !! BU SATIR ISINMA=4 KARARINDAN ONCE, konum 1..8 uzerinden
           hesaplandi. Simdiki kayip konum 4..8'i puanlar (§5.2).
           TABLONUN GERISI ETKILENMEZ -- |z|, |Pz|, s, cos, sira
           ileri gecisin sayilari, kayip dilimine bagli degil.
           Ilk gercek kosuda tazelenecek.
```

Bu tablo, aşağıdaki iddiaların **üçünü birden** tek bakışta gösteriyor
ve bu yüzden belgenin başında duruyor:

```
|z_j| = 1,0000 HER ADIMDA     IZOMETRI dogru (§2.1). Gozle gorunuyor.

|Πz_j| 0,96 -> 0,56           OKUNABILIR kutle her adimda azaliyor.
                              §4.1 bu bosluğu ZORUNLU kıldı; ayni
                              bosluk bilginin OKUNMAYAN yere
                              kacmasinin da yolu.  (§2.1'in NE
                              DEMEDIGI -- asagida.)

s_j ~ 0,55   esik 0,969       CAPA hic tetiklenmiyor. Esik
                              1 - r^2/2 = 0,96875; durum hicbir koda
                              o kadar yakin degil.  §3'un onkosulu
                              SAGLANMIYOR.

dis = 0,0000                  Itme terimi bu cumlede BIREBIR SIFIR.
                              §5.1'e bakiniz.
```

---

## 1. Mimari

Kodun hesapladığı şey, birebir:

```
BASLANGIC   z_0  = [ p_{w_0} ; 0_{D-d} ]                    |z_0| = 1
            t_0  = 0
DONME       zp_j = R_{w_{j-1}} z_{j-1}
KOD ARAMA   k_j  = argmax_k <zp_j , c_k>,    c_k = C_k/|C_k|
CAPA        vur_j = [ <zp_j , c_{k_j}> > 1 - r^2/2 ]
            z_j  = vur_j ? c_{k_j} : zp_j
SAAT        t_j  = vur_j ? 0 : t_{j-1} + 1        SON CAPADAN BERI ADIM
OKUMA       q_j  = Π z_j / |Π z_j|
            w_j  = argmax_w <q_j , h_{t_j, w}>    saat KAPALI: h = p
```

```
SABIT (ogrenilmez)
  p_w ∈ S^{d-1}          n birim, KUCUK okuma uzayinda, tohumdan
  Π : R^D -> R^d         ilk d koordinata izdusum

OGRENILEN
  R_w ∈ SO(D)            her birimin donmesi          (§4.3, §9.1)
  C   ∈ R^{K×D}          KOD DEFTERI
  S   ∈ SO(D)            saat.  OPSIYONEL, §6, VARSAYILAN KAPALI

ESIK
  r       capa yaricapi   -- EGITIMDE KULLANILIYOR (§5, §13/A1)
  delta   itme esigi      -- EGITIMDE KULLANILIYOR (§5, §13/A1)
```

Hesap yığını yok: ne dikkat, ne katman, ne çıktı matrisi. İleri
geçişte yalnız dönme çarpımı ve en yakın komşu.

**Belgede olmayan, kodda olan dört şey — artık burada:**

```
z_0 TANIMI      Ilk surumde YOKTU. Pencere akistan keyfi yerden
                basliyor; ilk birimin noktasi okuma uzayina konuyor,
                gizli kisim SIFIR. Yani z_0'da gizli kutle yok, §4.1'in
                kullandigi bosluk adimlarla olusuyor.

C KUREDE        §1 "C ∈ R^{K×D}" diyordu, norm kisiti yoktu. Kod ileri
                geciste c_k = C_k/|C_k| kullaniyor. GEREKLI: `C` serbest
                bir parametre, egitimde kureden cikiyor ve o zaman capa
                durumu kureden atiyor -- "norm korunur" iddiasi duser.
                OLCULDU (kapi 16): C x2,5 -> |z| = 2,5.

ESIK IC CARPIMDA  |zp - c_k| < r  ile  <zp,c_k> > 1 - r^2/2  ozdes
                (ikisi de birim normda). Kod ikincisini kullaniyor;
                (B,K) uzerinde uc elemanwise tensor daha kurmamak icin.

t_j TANIMI      Kod her adimda `t`yi hesaplayip donduruyor ama bu
                blokta YOKTU (21 Eylul, denetim sirasinda bulundu --
                blogun basligi "birebir" diyor).
                    t = t + 1                 CAPA KONTROLUNDEN ONCE
                    t = t.masked_fill(vur, 0) CAPADAN SONRA
                yani capa tetiklenen adimda t = 0.
                YUK TASIYOR: saat acikken okuma hedefi p_w degil
                H[t_j][w] (§6). Saat VARSAYILAN KAPALI oldugu icin
                su an atil -- ama `yol()` onu her kosuda hesaplayip
                donduruyor ve `saglik()` basiyor.
```

---

## 2. Üç iddia — ve her birinin NE DEMEDİĞİ

```
1  IZOMETRIK GECIS   Donme normu ve mesafeyi korur:  |R a - R b| = |a - b|
                     -> temsil cokemez, sinyal sonmez, gradyan sonmez.
                     OLCULDU: ortogonallik sapmasi 1,5e-06 (kapi 4);
                     6 adimlik bileskenin tekil degerleri 1 (kapi 6);
                     §0 tablosunda |z_j| = 1,0000.

   NE DEMIYOR:       |Π a - Π b| hakkinda HICBIR SEY. Okudugumuz sey o.
                     OLCULDU (20 Eylul): "Ayşe Yılmaz" ile "Ayşe Çelik"
                     arasindaki fark
                        adim 2  |za-zb| 0,3164   |Πza-Πzb| 0,2151
                        adim 3  |za-zb| 0,3164   |Πza-Πzb| 0,0525
                     Tam mesafe kili kilina sabit -- teorem TUTTU. Ama
                     okunabilir kismi bir adimda dorde bolundu: farkin
                     %97'si gizli 24 boyuta gecti. Orada kimse okumuyor
                     ve KAYIPTA ORAYI CEZALANDIRAN TERIM YOK.

                     VE SIZINTI BIRIMIN DEGIL, DURUMUN OZELLIGI.
                     §0 tablosunda `Yildiz` IKI kez girdi oluyor, yani
                     AYNI R_Yildiz matrisi iki kez uygulaniyor:
                        j=2   |Pz| 0,9638 -> 0,9418   oran 0,9772
                        j=7   |Pz| 0,6448 -> 0,5338   oran 0,8279
                     Ayni operator, 7,5 kat farkli kayip. Cunku R tam
                     durum uzerinde operatordur, OKUNABILIR KISIM
                     uzerinde DEGIL (§4.1: Pi izometri degil).
                     SONUC: "o birimin donmesi kotu, daha iyisini
                     ogrenir" turu butun PER-BIRIM cozumler ELENIR.

2  NICELEME          Durum PERIYODIK olarak bir koda oturur
                     -> durumlar YENIDEN KULLANILABILIR olur.
                     Gorulmemis bilesim, gorulmus parcalara iner.
                     ASIL IDDIA BU.

   NE DEMIYOR:       "Periyodik" DENKLEMDEN CIKMIYOR, VARSAYIM. Capanin
                     tetiklenip tetiklenmeyecegi r'ye ve kodlarin nereye
                     oturduguna bagli, ve ikisi de ogrenmenin sonucu.
                     OLCULDU: §0'da tek bir cumlede HIC tetiklenmedi;
                     tam korpusta %17,8, ve 2048 kodun 329'u kullanildi.

3  MESAFEYLE OKUMA   Cikti katmani yok; sozluk buyudukce okuma maliyeti
                     artar ama PARAMETRE artmaz.

   NE DEMIYOR:       SABIT d'de n buyudukce noktalarin sikismasi.
                     OLCULDU: n=451, d=8'de RASTGELE bir yonun en yakin
                     noktaya cos'u zaten 0,857 (31 derece); en kotu cift
                     0,970 (14 derece). Dogru okumak, hedefi rastgeleden
                     DAHA IYI degil, 450 rakibin EN IYISINDEN daha iyi
                     yapmak demek.
                     Noktalari itme ile yaymak en kotu cifti 14,2'den
                     44,8 dereceye cikariyor (3,2 kat) -- ve d=8 duzgun
                     yayilmis, d=16 rastgeleden IYI. Kod su an `randn`
                     kullaniyor.  ACIK SORU A8.
```

---

## 3. Çapa = niceleme (vektör kuantalama)

```
k_j = argmax_k <zp_j , c_k>
<zp_j , c_{k_j}> > 1 - r^2/2      ->   z_j <- c_{k_j}        CAPA
```

**Varlık listesi yok.** `C` öğrenilir; model hangi durumların yeniden
kullanılmaya değer olduğunu kendi bulur.

### 3.1 Neden `çıkarım`ı veriyor — TEOREM, ve teoremin SINIRI

`j`'de `c_k`'ya çapalandıysa `z_j = c_k`, geçmişten **bağımsız**.

```
TEOREM     Ara adimlarda capa TETIKLENMEZSE, herhangi bir sonek
           u = (u_1..u_m) icin
               z_{j+m} = R_{u_m} ... R_{u_1} c_k
           yalniz (k, u)'ya baglidir -- oraya nasil gelindigine DEGIL.
```

> `(k, u)` eğitimde görüldüyse, model onu `k`'ya çapalanan **her**
> bağlamda birebir tekrarlar.

Sonuç: tek-adım doğruluğu `a` ise **`çıkarım ≈ a²`**.

```
!! ILK SURUMDE "ara adimlarda capa tetiklenmezse" KOSULU YOKTU.
   Formul  z_{j+m} = R_{u_m} ... R_{u_1} c_k  diye yaziliydi, oysa
   modelin ozyinelemesi  z_{j+1} = capa(R z_j).  Capa BILESKEDEN
   DUSURULMUSTU.

   Dusurulen sey zararsiz degil: R_{u_1} c_k yine ayni kodun
   yaricapina duserse  z_{j+1} = c_k  olur ve durum SABIT NOKTA'dir.
   OLCULDU (20 Eylul), modelin yazdigi:
       ? -> Şanlıurfa(0,92) -> Şanlıurfa -> Şanlıurfa ... 12 kez
       |Πz| her adimda TAM AYNI: 0,193
   Teorem "durum gecmisi unutur" diyor ve dogru soyluyor; biz onu
   "GEREKSIZ gecmisi unutur" diye okuduk. Soruyu da unuttu.
```

### 3.1b KOD DEFTERİ NE KODLUYOR — ÖLÇÜLDÜ, ve §3'ün ASIL İDDİASI DÜŞÜYOR

Ölçüldü (21 Eylül, t0 / d=16, kapalı form, eğitim YOK). 7.381 tek
adımlı sınav öneği; öneğin **sonundaki** kod indeksi `k` tek bir
sayı (2048 sınıf) ve ondan ne okunabildiğine bakıldı. Kıyas zemini:
`k` **karıştırılmış** hâli — 2048 sınıfa bölmenin kendiliğinden
verdiği düşüşü ayıklamak için.

```
              GERCEK k     KARISIK k       NET      en-olasi dogruluk
ILISKI        1,849 bit    0,169 bit    +1,680     %36,5  vs  %12,7
OZNE          2,244        1,705        +0,539      %3,4  vs   %2,7
CEVAP         1,958        1,640        +0,318      %3,0  vs   %2,8

kullanilan AYRI kod          166 / 2048
onegin SONUNDA capa tetikleyen  %37,0
```

**Kod defteri İLİŞKİYİ kodluyor, OLGUYU değil.** `OZNE` ve `CEVAP`
kıyas zemininin içinde. 166 kod, ve ayırt ettikleri şey esasen 24
ilişki.

```
!! "CAPA OZNE KIMLIGINI SILIYOR" DIYE DUSUNDUM -- OLCULDU, YANLIS.
   Capa tetiklendiginde z <- c_k, yani durum 166 vektorden biriyle
   DEGISTIRILIYOR; kimligin silinmesi beklenirdi. Ayni orneklerde
   capa ONCESI durum (zp) ile SONRASI (c_k) karsilastirildi:
                                    sira    1.sira   SANSA GORE
       CAPA VAR  (z = c_k)        142,19     %0,2     1,10 kat
       CAPA VAR  (zp, capa ONCESI) 132,50    %1,1     1,18 kat
       CAPA YOK  (z = zp)          14,33    %23,6    11,34 kat
   Capanin atesledigi orneklerde kimlik ZATEN YOKTU (1,18 kat).
   CAPA SILMIYOR -- kimligin zaten kayboldugu yerlerde ATESLIYOR.
   Sebep degil, BELIRTI. Mantikli: kodlar yalniz iliskiyi kodluyor,
   kimligini yitirmis durum o jenerik cekicilere dogal olarak yaklasir.

!! IKI NUFUS var, ve ikisi de BILGI uretmiyor:
       %63  kimlik SAGLAM        sansin 11,3 kati, %23,6 tam isabet
       %37  kimlik ZATEN GITMIS  sansin 1,1 kati -- capa burada atesler
   BILGI tam = 0,0000 HER IKISINDE de. Yani soruyu tasimak GEREKLI
   ama YETERLI DEGIL.
```

**§3'ün asıl iddiası** — *"durum periyodik olarak bir koda oturur →
durumlar yeniden kullanılabilir olur → görülmemiş bileşim görülmüş
parçalara iner"* — şu hâliyle **çalışmıyor**: durum oturuyor (%11
üretimde), ama oturduğu kod olguyu değil ilişkiyi taşıyor, ve
bileşim inmiyor.

Eksik olan şey `(özne, ilişki) → cevap` **araması**, ve mimaride bu
işi yapan bir mekanizma **yok**. Kod defteri bunun için tasarlanmıştı;
ölçüm ilişkiyi kodladığını gösterdi.

---

### 3.2 Kod defteri üyeliği = DÜRÜSTLÜK

`Zeynep` gerçek bir ad, `Kayabaşı` gerçek bir soyad, ama
`Zeynep Kayabaşı` diye bir varlık yok — kod defterinde de kodu yok.

```
durum bir kodun r yaricapinda   ->  capa,      "var"
hicbir kodun yakininda degil    ->  capa yok,  "yok"
```

Reddetme ayrı bir mekanizma değil, **kod defteri üyeliğinin
kendisi**. Ve genel: "varlık" kavramı gerektirmiyor.

```
!! KODDA YOK. Model "yok" diyemiyor; uretim hep bir birim yaziyor.
   §3.2 bir TASARIM, bir uygulama degil.  ACIK SORU A9.
```

---

## 4. `D > d`, açısal okuma, ve operatörün zenginliği

### 4.1 Okuma neden bir İZDÜŞÜM

`D = d` olsaydı ortak sonek `A` bir izometri olur, bütün ikili
mesafeleri korurdu:

```
|q_H - q_A| = |p_Fatma  - p_Zehra|        (anne soneki)
|q_H - q_A| = |p_Görgül - p_Betimsel|     (tezi soneki)
```

Sol taraf ilişkiden bağımsız → sağ taraflar eşit olmak zorunda
kalırdı. Sabit rastgele noktalarda olmaz.

*Sayı (çember, 475 birim):* gereken hata `< 0,38°`, en iyi uzlaşmada
çıkan `~100°`. **250 kat.**

`Π` izometri değildir; `|Π A (q_x − q_y)|` artık `A`'nın farkı hangi
yöne çevirdiğine bağlıdır. Çelişki kalkar.

```
!! BU BIR GEREKLILIK KANITI, YETERLILIK DEGIL. "Π'nin ne kadarini
   gecirecegi A'ya bagli" demek, o miktarin SERBEST BIR PARAMETRE
   oldugu demektir -- ve onu pinleyen tek sey kayiptir. §5'in kaybi
   pinlemiyor: §2.1'de olculen %97 sizinti tam buradan geciyor.
   §0 tablosunda |Πz_j|'nin 0,96'dan 0,56'ya inisi ayni sey.
```

### 4.2 Okuma AÇISAL olmalı

`|Π z − hedef|²` yazılamaz: gizli kütle varken `|Π z| < 1` olur ve
terim gizli kütleyi **sıfıra iter** — oysa 4.1 onu zorunlu kılıyor.
İki terim birbiriyle kavga eder. Bu yüzden:

```
q = Π z / |Π z|
L_üye = |q - p_w|^2 = 2 - 2 cos(açı farkı)
```

İş bölümü:

```
L_üye, L_dis   OKUMA YONU    -- hangi kelime
L_çapa         TAM DURUM     -- hafiza, yalniz capa noktalarinda
```

```
!! OKUMA BIR SIRALAMA (argmax), KAYIP BIR MESAFE. Ikisini baglayan
   sey bir MARJ olmali; §5'te o isi L_dis'in yapmasi bekleniyordu ve
   yapmiyor (§5.1).
```

### 4.3 Tek düzlem — ve frekans ayrımının ÖLÇÜLEN hatası

Birim başına tam `SO(D)` gerçek ölçekte imkânsız:

```
V=50k, D=256   ->   50.000 x 32.640  =  1,63 MILYAR
```

Tek düzlemli (Givens) dönme düzleminin dışında özdeşliktir:

```
E[ |delta_duzlem|^2 / |delta|^2 ]  =  2/D
D= 32   ->  %6,3     farkin %93,7'si DOKUNULMAZ      OLCULDU 0,0629
D=256   ->  %0,8
```

İlk sürüm çözümü **frekansa** bağlıyordu: yüksek frekanslı birimler
tam `SO(D)`, kalanı tek düzlem — *"bir içerik kelimesinin işi durumu
ayırt edilebilir bir yere taşımak; bunun için tek düzlem yeter."*

```
!! O SON CUMLE ARGUMANSIZDI ve OLCUM YANLISLADI (20 Eylul).
   451 birimin 328'i VARLIK birimi (ad parcasi). K_TAM=80 iken
   bunlarin yalniz 24'u (%7,3) tam SO(D) aliyordu; kalan 304 --
   yani OZNE KIMLIGINI TASIYAN birimlerin tamami -- farkin %6,2'sine
   dokunan operatore mahkumdu. Modelin yazdiginda birebir gorundu:
       Mersin -> Mersin(0,98)   Bartın -> Bartın(0,94)
   yani R[ad] ~ I, ozne degisince cevap degismiyor.

   DUZELTILDI: K_TAM = n (hepsi tam SO(D)), parametre 129.331 ->
   289.232. BICIM acildi (kalip 0,099 -> 0,144, ek 0,554 -> 0,641)
   ama BILGI 0,0000 kaldi: darbogazdi, TEK darbogaz degildi.

   Bolmenin kendisi (frekans) bir dugme olarak duruyor -- ama artik
   kapi 31 FREKANSA degil ROLE bakiyor: varlik birimi tam SO(D)
   almali.

   AYRIM OLCEK MESELESI, KALITE DEGIL (21 Eylul).  Yukaridaki
   1,63 MILYAR sayisi V=50k, D=256 icin. Bu olcekte -- 444 birim,
   D=32 -- tamami 220 bin parametre, yani ayrima GEREK YOK.
   `K_TAM = None` (AYRIM YOK) yazildi; deger 451 idi ve sozluk
   444 oldugu icin "hepsi"yi TESADUFEN soyluyordu. Sozluk 451'i
   gecse ayrim KENDILIGINDEN, KEYFI bir kesimle geri gelirdi ve
   kapi 31'in oran sarti bunu yakalamazdi (o gun varlik birimleri
   hala ilk 451'de olabilir). Kapi 31'e ikinci sart kondu:
   `K_TAM` bir SAYIYSA sozlukten KUCUK olmali.
   Olcek buyuyunce ayrim yeniden acilir -- ama kesim FREKANSA
   degil ROLE gore kurulur.
```

---

## 5. Kayıp — YOLUN TAMAMINA

Kullanıcı: *"cümlenin tamamını değerlendirmemiz gerekiyor next token
değil."*

Kodun hesapladığı şey, birebir:

```
kos_{j,w} = <q_j , p_w>                              j = ISINMA .. L-1

üye   = 2 - 2 · ort_{b,j} kos_{j, w_j}               KONUM BASINA ORTALAMA

d2_w   = ( 2 - 2 · max_j kos_{j,w} )_+
iç_w   = [ w bu pencerede geciyor ]
dmin_w = sqrt( max( iç_w ? delta+1 : d2_w , TABAN ) )       TABAN = 1e-8
dis   = ort_b  Σ_w (delta - dmin_w)_+^2               BIRIM UZERINDE TOPLAM

kod   = ort_{b,j} | sg[zp_j] - c_{k_j} |^2            HER ADIMDA
bağ   = Σ_{vur} | zp_j - sg[c_{k_j}] |^2 / max(1, #vur)

düzen = |a|^2 + |θ|^2

L = üye + a1·dis + a2·(kod + beta·bağ) + a3·düzen
```

`L_dis` **itici kuvvet**. *Gerekçe ÖLÇÜLDÜ (model_13):* saf çekme
kaybı çöktü — bit 8,32, unigram tabanı 6,58'in üstünde. Çekmek yetmez.

`L_çapa` **örtük çıkarımın ön koşulu**: onsuz durum kodun yanından
geçer ama çapa tetiklenmez.

Sıra kısıtı YOK — sıra yolun kendi sırası.

### 5.1 Bilinen kusurlar — hepsi ÖLÇÜLDÜ

```
A  DELTA -- INCELENDI, DEGISMIYOR.  KARAR 21 Eylul, §13/K11.
   Itme ancak dmin < delta iken ateslenir. OLCULDU (20 Eylul, tam
   korpus, t0):
       hedefin uzakligi           0,9160
       rakiplerin uzakligi  %50   0,8974   %25  0,7611   %5  0,5745
       delta                      0,4000
   Itilmesi gereken ciftlerin %0,48'i esigin altinda; dis = 0,0048.
   Ilk okumam "delta cok kucuk" idi.  OLCUM AKSINI SOYLEDI.

   OLCULDU (21 Eylul, 20 cumle, 300 adim, MODEL CALISIRKEN):
       HEDEFIN uzakligi      %50 0,1169   %95 0,2970   ort 0,1406
       PENCERE DISI birim    %1  0,4530   %50 0,8802   en kucuk 0,3460
   Yani calisan rejimde hedef 0,117'de, EN YAKIN rakip 0,346'da --
   ust uste binmeyen, 3 katlik bir ayrim. ITMEYE GEREK YOK ve
   delta = 0,4 birimlerin yalniz %0,3'unu itiyor: DOGRU davranis.

   BUYUTMENIN BEDELI (ayni olcum):
       delta 0,6 -> pencere disi birimlerin  %5,9'u itilir  (~25/420)
       delta 0,8 ->                         %31,9          (~134/420)
       delta 1,0 ->                         %75,8          (~319/420)
   d = 8 boyutta, 20 konumluk bir yoldan 319 noktayi birden
   uzaklastirmak SAGLANAMAZ; yalniz gurultu olur. Ustune §5.1/H.

   ASIL MESELE: tam korpusta hedef 0,9160'ta, yani CEKME basarisiz --
   itme degil. Ve 0,9160 uzaklik  uye = 0,839  demek; §5.1/J'nin
   tabanindan geri cozulunce  k ~ 3  cikiyor: tam korpusun ortalama
   DALLANMASI. Sayi kendi kendini acikliyor.
   delta basarisiz bir cekmeyi duzeltemez.  DEGISMIYOR.

B  TOPLAM/ORTALAMA KARISIK
   Belge ikisini de TOPLAM yaziyordu; kod `üye`yi konum basina
   ORTALAMA, `dis`i birim uzerinde TOPLAM aliyor. Yani a1 = 1,0
   yaziyor ama gercek agirlik ~1/180. Degisiklik olcek/bellek icin
   yapildi, AMAC FONKSIYONUNU degistirdigi fark edilmedi.

C  L_capa'nin KOD TERIMI HER ADIMDA
   Belge "yalniz capa tetiklendigi adimlarda" diyordu. Kod `kod`u
   HER adimda uyguluyor (k-ortalama gibi), `bağ`i yalniz
   tetiklendiginde. Sonucu: kod defteri 15,9 M durumun TAMAMINA
   cekiliyor. OLCULDU: 2048 kodun 329'u kullaniliyor, |z-c| ~ 0,56
   ve r = 0,25 -- yani ortalama durum capa yaricapinin 2,25 katinda.

D  zp, z DEGIL
   Belge `z` yaziyordu. Capa SONRASI z TAM OLARAK c_k oldugu icin
   |z - c_k| = 0 ve terim OZDES SIFIR olurdu. Kod capa ONCESI
   durumu (`zp`) kullaniyor. (Kapi 19/16 yakaladi.)

E  TABAN ve MASKE
   `sqrt`in turevi 0'da tanimsiz. `kos` bir kosinus ama fp32'de 1'i
   asiyor (olculen en kucuk 2-2kos: -2,384e-07), clamp onu TAM 0
   yapiyor. OLCULDU: egitim ADIM 1'de NaN verdi.
   Duzeltme: pencerede olan birim delta+1'e konuyor (mentese zaten
   0), ve d2 TABAN=1e-8'e tabanlaniyor -> dmin >= 1e-4, gradyan
   sonsuz yerine <= 4.000; mentesenin en uc terimi 0,1600 yerine
   0,1599 (%0,05 sapma).
   !! CIHAZ FARKI: ClampBackward NaN'i CPU'da yutuyor, CUDA'da
   yutmuyor. Bu sinif hatayi yerel CPU dongusu GOREMEZ; kapi DEGERI
   sinamali, NaN'i degil (kapi 28).

F  L_duzen u ve v'yi KAPSAMIYOR
   Belge "donme ureteci" diyor; kod yalniz `a` ve `θ`. Tek duzlemin
   YONU (u, v) duzenlenmiyor. K_TAM = n iken zaten acik sinif bos.

G  COP ONEK PUANLANIYORDU  --  KAPANDI 20 Eylul, bkz §5.2
   Pencere akistan keyfi yerden basliyor (§9.5), yani bastaki durum
   cop. Ilk surum bunu "cop durum ILK CAPADA silinir" diye
   mesrulastiriyordu -- SILINMIYOR: adim 1'de s = 0,5482, esik 0,969,
   capa hic tetiklenmiyor (§0 tablosu). Ortada bir mekanizma hic yoktu.
   OLCULDU: ayni onege kac AYRI hedef dayatildigi --
       konum 1: 9,20 hedef   tavan %45,1
       konum 2: 3,34         tavan %52,9
       konum 3: 1,91         tavan %63,2
       konum 6: 1,10         tavan %90,8
   Hicbir belirlenimci model bu tavani gecemez, ve tam bu konumlar
   AD birimlerinin donmesini egiten konumlar.
   COZUM: ISINMA = 4, kayip konum 4..23'u puanliyor. Hicbir gecis
   dusmuyor -- gerekce ve alternatifin elenmesi §5.2, kapi 32.

H  L_dis HAKSIZ CEZA (ilk surumden beri biliniyor)
   Pencerede olmayan birimlerin bir kismi GECERLI alternatif
   (korpusta 17 bildirim kalibi var). Mentese softmax'tan sert.

I  OLU KOD
   VQ'nun klasik arizasi. Standart care yeniden baslatma.  A6.

J  KAYBIN GUCU DALLANMA CARPANINA BOLUNUYOR
   ISPAT -- egitim YOK, veri YOK, agirlik YOK. Yalnizca sabit okuma
   geometrisi (p, tohum 0, n=444, d=8) ve kaybin tanimi uzerinde
   aritmetik. `uye`yi DUSUREN bir gradyan adimi, hedeflerin sirasini:
       k= 1   %0,0 kotulestirir      ort sira degisimi  -31,3
       k= 2   %0,2                                      -21,7
       k= 3   %3,7                                      -17,8
       k= 5   %9,7                                      -13,6
       k=11   %19,8                                      -9,0
   k=1 satiri TEOREM: q'yu p_t'ye eta kadar iterken hedefin kazanci
   eta*1, herhangi bir rakibin kazanci eta*<p_t,p_c> <= eta. Hedef
   her rakipten EN AZ KADAR kazanir -> sira ASLA kotulesemez.
   k>1'de gradyan hedeflerin AGIRLIK MERKEZINE gidiyor, ve merkez
   hicbir hedefin yeri degil.

   ERISILEBILIR TABAN:   uye >= 2 - 2/sqrt(k)
       k=1 0,0000    k=3 0,8453    k=5 1,1056
       k=11 1,3970   k=24 1,5918
   §0'da olculen (EGITILMEMIS) uye = 1,6561. Yani k=11 bir konumda
   egitimin TAMAMI uyeyi 1,66'dan ancak 1,40'a indirebilir -- %16.

   SEMADAN GERCEK BIR ORNEK: `<kisi adi> -ın  ???` konumunda KISI
   oznesinden 11 iliski cikabiliyor (annesi arkadasi babasi bolumu
   cocugu danismani kardesi memleketi ogrencisi tezi yasadigi_yer).

   SONUC  Kayip YANLIS HIZALI DEGIL -- SULANMIS. Belirlenimci
   konumda kusursuz. Ve `uye` sayisina bakip "model ogrendi mi"
   DENEMEZ: sayinin icinde, konumdan konuma degisen, AYRILMAMIS bir
   indirilemez pay var.
   !! ISINMA (§5.2) bunu COZMEZ. O, pencerenin KESIM YERINDEN dogan
   belirsizligi atti; bu, DILIN kendi belirsizligi ve her konumda var.
   !! Tek noktali okumanin yapisal siniri: k yollu bir konumda
   hedeflerin en fazla 1/k'si 1. siraya cikabilir. Bu MARJLI kayip
   icin de gecerli -- okuma tek nokta oldukca kayip sekli degistirmek
   tavani degistirmez.                                      (kapi 33)

K  TERIM AGIRLIKLARI OKUMAYI GERI CEKIYOR
   OLCULDU (21 Eylul, 20 cumle ezber sinavi, GERCEK m.kayip):
       adim   kayip    1.SIRA%   cos_dogru
        250   0,356     98,1%     0,9801
        500   0,266     98,6%     0,9886
        750   0,231     98,2%     0,9904   <- TEPE
       1000   0,226     97,6%     0,9876
       1250   0,216     96,4%     0,9849
       1500   0,204     95,9%     0,9845
   750'den sonra TOPLAM kayip dusmeye devam ediyor ama hem sira
   hem cos_dogru KOTULESIYOR. Yani kazanc `uye`den degil, `kod`
   (VQ) ve `duzen`den geliyor: mimari dogru cevabi biliyor, sonra
   kod defteri ve duzenleyici onu GERI CEKIYOR.
   a1 = 1,0  a2 = 1,0  a3 = 1e-4  HICBIRI olculmeden secilmisti
   (`ayar_14`: "buradakiler baslangic"). Simdi olculmus bir bedeli
   var.  ACIK -- §13/A1.

L  VARLIK BIRIMLERI DURUMU OYNATMIYOR  --  BILGI=0'IN MEKANIZMASI
   OLCULDU (21 Eylul, t0, commit 46d9b04, 444 birim / 5 epok):
                      n    ort frek   ort |a|   1-cos   KENDINE DONEN
       VARLIK       331     18.920    1,0012   0,1085   243   %73
       EK            34    137.734    1,7551   0,4559     7   %21
       otekiler      82     74.403    1,9169   0,4111    18   %22
       HEPSI        444     35.845    1,2184   0,1855   268   %60
   "KENDINE DONEN": z = [p_w; 0] alinip R_w uygulandiginda okuma YINE
   w'yi veriyor -- yani R_w, w'nin kendi noktasinin OKUNABILIR yonunu
   degistirmiyor. Sozlugun %60'i kendi donmesinin SABIT NOKTASI.

   UCUNU BIRDEN ACIKLIYOR:
       uretimde TEKRAR      kendine donen birime girince cikilmiyor
                            (Recep x3, Kapadokya x4, Beykoz x3)
       kapanmadi 0,6967     dongudeyken `.` hic gelmiyor
       BILGI tam 0,0000     ozne adi SEYREK -> R[ad] ~ I -> oznenin
                            kimligi duruma HIC GECMIYOR

   MEKANIZMA -- GEOMETRIK, ve olculdu:
       okuma uzayinda EN YAKIN KOMSU acisi   ort 30,5°  (n=444, S^7)
       R_w'nin p_w'yi tasidigi aci
           VARLIK   23,5°   <- komsu araligini GECEMIYOR   %73 kendine
           EK       53,7°   <- rahat geciyor                %21 kendine
   Durum, basladigi Voronoi hucresinden CIKMIYOR. ("teta < fi/2 ->
   kendine doner" kestirimi %65,3 tutuyor; kalani YONDEN geliyor,
   buyukluk tek basina belirlemiyor.)

   !! ILK YAZDIGIM SEBEP YANLISTI: "duzen = a3*S|a|^2 seyrek birimin
   donmesini sifira cekiyor" demistim. HESAP AKSINI SOYLUYOR --
   birim basina, uye'nin CEKMESI / duzen'in ITMESI:
       VARLIK ort (frek  18.920)   1,2e-03 / 2,0e-04 = 5,9 kat
       EN SEYREK  (frek   4.087)   2,6e-04 / 1,7e-04 = 1,5 kat
   Yani a3 EN SEYREK birimlerde kiyaslanabilir, VARLIK birimlerinde
   BASKIN DEGIL. Frekansla monotonluk gercek ama sebebi a3 degil.

   NIYE BUYUMUYOR -- BIR HIPOTEZ KURDUM VE MATEMATIK CURUTTU.
   Hipotez: "amac fonksiyonu varliklari ayirt etmeyi ISTEMIYOR; bir
   ad okunduktan sonra gelecek birim kim olursa olsun `-ın`/`-dır`."
   Bu VERININ ozelligi, yani SAYILIR -- egitim gerekmez. Sayildi
   (21 Eylul, 15,9 M gecis):
       H(ardil | girdi)          VARLIK 2,434 bit  -> k_etkin 5,41
                                 en sik ardilin payi %44,5
       ardil bir AD MI           girdi VARLIK -> ardil VARLIK %56,4
       iki AYRI adin ardil
       dagilimi ortusmesi        0,3308 -> AYIRT EDICI bilgi %66,9
   HIPOTEZ YANLIS. Ad okunduktan sonra ardil belirsiz, kimlik
   uretilmesi gereken konumlar seyrek degil (yarisindan cok), ve iki
   ad birbirinden cok farkli devam ediyor. R[ad]'in ayirt edici
   olmasi icin BOL gradyan basinci VAR.
   (Bu sayim `a3 = 0` kosusunu da gereksiz kildi.)

   GERIYE KALAN, ve dort ayri izin ciktigi TEK yer:
       AMAC FONKSIYONU   ayrimi ISTIYOR     %66,9 ayirt edici bilgi
       TAM DURUM         ayrimi KORUYOR     |za-zb| 0,3164 SABIT (§2.1)
       OKUMA             ayrimi KAYBEDIYOR  |Pza-Pzb| 0,2151 -> 0,0525
       SONUC             varlik donmesi okunabilir kismi 23,5°
                         oynatiyor, komsu araligi 30,5° -> hucreden
                         cikmiyor -> %73 kendine donuyor
   Basinc var, tasiyici var, OKUMA TASIMIYOR. §4.1'in `!!` blogu
   bunu zaten yazmisti: gecirilen miktar SERBEST bir parametre ve
   onu pinleyen tek sey kayip; §5'in kaybi pinlemiyor.
   `uye` yalniz Pz'nin YONUNE bakiyor, |Pz|'ye DEGIL -- o yuzden
   egitim |Pz|'yi 0,96'dan 0,18'e indirirken kayip itiraz etmiyor.

   NEREDE OLMADIGI DA OLCULDU -- KAPALI FORMLA, EGITIM YOK
   Bir birimin butun (gelen durum z_i, hedef t_i) ciftleri alinip
   UC okuma karsilastirildi. Sira, o birimden sonra GORULEN sinif
   kumesi icinde; sinav dilimi AYRI (%70/%30).
     SU ANKI      egitilmis donme + sabit p okumasi
     EN IYI R     Procrustes -- `uye`nin KENDI amacinin (S<PiRz,p_t>)
                  matematiksel en iyisi; daha iyisi YOK
     SINIFLANDIR  ayni z'lerden dogrusal siniflandirici (one-hot ridge)

     birim     n_sn | SU ANKI  EN IYI R  SINIFLANDIR |  SANS    k
     -i        5113 |  47,22    47,93     26,00      |  91,5   182
     -si       2817 |  44,80    41,46     22,55      |  74,0   147
     -ü        2922 |  24,14    20,91      9,59      |  46,5    92
     Akiskanlar 146 |   6,21     5,53      4,08      |   9,5    18
     -dir      3322 |   1,12     1,13      1,10      |   1,5     2

   UC SONUC
     1  EGITIM EN IYI DONMEYI ZATEN BULMUS. SU ANKI ~ EN IYI R her
        satirda. Arizanin optimizasyonla ILGISI YOK -- kanitli.
     2  OKUMANIN BICIMI ~2 KAT MALIYETLI. Ayni durumlardan dogrusal
        siniflandirici 47,2 -> 26,0 ve 24,1 -> 9,6. "Sabit rastgele
        NOKTAYA isabet et" sarti pahali (§4.2 / §13-A5) -- ama tek
        basina arizayi ACIKLAMIYOR.
     3  KALAN BOSLUK DURUMDA. En iyi dogrusal okuma bile 26/182'de
        kaliyor, 1'de degil. Bilgi durumda KISMEN var (2-3,5 kat sans
        ustu), yetecek kadar DEGIL.
   Dilbilgisi tarafi CALISIYOR: `-dir` 1,12 / sans 1,5.
   Hukum veren satirlar `-i`, `-si`, `-ü` -- CEVAP tam oralardan
   sonra geliyor ve ozne onekte var.  BILGI = 0,0000'in sayisal
   karsiligi bu.

   !! `Ahmet` gibi AD satirlari hukum VERMEZ: orada siniflandirici da
   sans seviyesinde (3,45 / sans 3,5), yani modelin arizasi degil --
   bir addan sonra gelen SOYAD sol baglamda belirlenmiyor olabilir.

   KIMLIK DURUMDA VAR, OKUMA GOREMIYOR -- SIZINTININ FIYATI
   OLCULDU (21 Eylul, kapali form, egitim YOK): sinav oneginin
   SONUNDAKI durumdan OZNENIN KIM OLDUGU dogrusal probla okundu.
   480 ozne, %70/%30 ayri dilim, sans = 240,5.
       TAM durum (32 boyut)        ort sira   6,09    1.sira %35,1
       OKUNABILIR kisim (8 boyut)  ort sira  50,18    1.sira  %3,2
       |Pz| onegin sonunda 0,3901
   SIZINTININ BEDELI:  sira 8,2 kat,  tam isabet 11 kat.

   IKI SONUC
     1  TASIMA CALISIYOR -- ve bu MIMARININ LEHINE. Alti adimlik
        donme bileskesinden sonra kimlik hala durumda ve DOGRUSAL
        okunabilir. §2.1'in izometri iddiasi ISE YARIYOR.
     2  OKUMA KAYBEDIYOR. Ayni bilgi okunabilir 8 boyutta 50,18.
        §4.1'in "gecirilen miktar SERBEST bir parametre ve onu
        pinleyen tek sey kayip" cumlesinin FIYATI budur.

   !! Pi SABIT: "ilk d koordinat" (§1), ve kimligi tasiyan yonlerin
   orada olmasi icin hicbir sebep yok. Ama GLOBAL ogrenilebilir Pi
   kazanc VERMEZ -- ortogonal baz degisimi p ve R'lere sogurulur.
   Gercek lever: `d`nin kendisi, kayba |Pz| terimi, ya da dogrusal
   OLMAYAN okuma. Her biri KENDI kagit sinavini ister.

   ADAYLAR SINANDI -- KAGITTA, KOSMADAN (21 Eylul)

   ADAY 1  `d`yi buyutmek.  DESTEKLENDI.
     Ayni durumlarin ILK d koordinatinda kimlik probu:
        d      ort sira   1.sira    kazanc/boyut
        8       50,18      %3,2       -5,4      <- SU AN
        12      23,08     %10,6       -6,8      <- en buyuk kazanc
        16      15,37     %16,9       -1,9
        24       8,80     %26,0       -0,55
        32       6,09     %35,1       -0,24
     Keskin dirsek yok ama 16'dan sonra getiri sonuyor.
     d = 8 -> 16:  sira 3,3 kat, tam isabet 5,3 kat.
     PARAMETRE MALIYETI SIFIR -- `p` buffer, `Pi` bir dilim.
     §4.1'in "D > d" sarti korunuyor (32 > 16).
     !! ALT SINIR: egri d=8 ile EGITILMIS modelden okundu.

     KALABALIK ARGUMANI NOTR, ve ilk dusundugumun TERSI:
     d buyudukce noktalar UZAKLASIYOR (komsu acisi 24,7° -> 41,8°),
     yani kendi hucrenden cikmak ZORLASIYOR. Ama `ayar_14`in
     baslangic olcegi zaten `aralik = n^(-1/(d-1))`, yani komsu
     araliginin KENDISI -- oran korunuyor. `d` lehine de aleyhine
     de degil.

   ADAY 2  Kayba |Pz| terimi.  ELENDI.
     Terim ancak "|Pz| buyuk olan durumlar GERCEKTEN daha iyi
     okunuyor" ise ise yarar. Olculdu:
        |Pz| dilimi        KIMLIK sirasi   CEVAP sirasi
        0,130-0,316            56,26          232,5
        0,369-0,414            47,81          241,2
        0,464-0,651            44,58          231,4
        korelasyon  |Pz|~kimlik -0,068   |Pz|~cevap -0,014
     Korelasyon SIFIR. Cevap sirasi butun dilimlerde ~235, yani
     sans (222). Sorun okunabilir kisimdaki KUTLE degil, HANGI
     YONLERIN o altuzayda oldugu -- `d` egrisi zaten bunu
     soyluyordu: yardim eden sey daha fazla BOYUT.

   ADAY 3  Dogrusal OLMAYAN okuma.  -> §12b (olgu hafizasi tasarimi),
     ve orada §4.1'i de cozdugu gosteriliyor.

   !! d = 8 -> 16 KOSULDU, ve KAZANC DEGIL TAKAS cikti (21 Eylul).
   Ayni betik, ayni bolme, ayni 480 ozne:
       model   TAM durum (32)          okunabilir blok
       d=8     sira   6,09   %35,1     sira  50,18   %3,2
       d=16    sira 111,08   %11,2     sira 132,01   %4,7
   DIL kazandi (kalip 2,8 kat, ek 1,9 kat, kapanan cumle 4/20 ->
   11/20) ama DURUM ozne kimligini KAYBETTI (%35,1 -> %11,2).
   MEKANIZMA: gizli alan TASIYICIDIR. d 8 -> 16 olunca gizli alan
   24 -> 16 boyuta indi; okuma buyudu, TASIYICI KUCULDU.
       d artar   -> okuma iyi,  tasiyici KUCUK
       d azalir  -> okuma kotu, tasiyici BUYUK
   CLAUDE.md: "bir alani bozarak baska bir alani iyilestirmek
   KAZANC DEGIL TAKAStir ve oyle yazilir."

   ADRESLEMENIN TAVANI (LDA, kapali form): en iyi 16 boyutlu
   altuzay %8,0; tam 32 boyut %11,2. Yani bilgi 16 boyuta SIGIYOR
   ve donme onu getirebilirdi -- getirmemis. Adresleme EGITIM/KAYIP
   tarafinda iyilestirilebilir, D buyutmeden.

   ACIK -- `d` bundan sonra TEK BASINA oynatilmaz; §12b ile birlikte.

   !! CAPA BUNUN SEBEBI DEGIL. Ayni uretimlerde capa HIC tetiklenmedi
   (s en fazla 0,89, esik 0,969) -- §3.1'in "emici durum"u DEGIL.
   Ve K_TAM = n oldugu icin "tek duzlem" aciklamasi da elendi.
   !! "EGITIMDE capa %7,61, URETIMDE %0,00" DIYE YAZMISTIM -- YANLIS,
   GERI ALINDI (21 Eylul).  O rakam IKI uretim izinden, toplam ~34
   adimdan geliyordu ve d=8 modeline aitti. 8.000 adimda olculdu:
       A  EGITIM penceresi        n 69.000   ATESLEYEN %8,56
       B  sinav onegi (zorlanmis) n  7.074   ATESLEYEN %10,49
       C  URETIM (modelin kendi)  n  8.000   ATESLEYEN %11,28
   Capa uretimde ATESLIYOR, hem de egitimden DAHA COK.  Iki ornekten
   genelleme yapmistim.

   !! KAPI 31 BUNU GOREMIYOR: operatorun SINIFINI sinar ("varlik
   birimi TAM SO(D) almali"), ogrenilen BUYUKLUGUNU degil. Tam
   SO(32) olup acisi 0,46'da kalan bir donme, tek duzlemli olmaktan
   iyi degildir. §13'un kendi kurali: "kapi teoreme degil IHTIYACA
   kurulur" -- ihtiyac "varlik birimi durumu OYNATMALI".
   ACIK -- esik ve care karari verilmedi.
M  KAYIP OLGUYU SATIN ALAMIYOR -- SAYILDI + OLCULDU (21 Eylul)
   !! TEZI DUSTU -- bkz. §5.1/O.  Fiyat zaten dogru; model
      odulu ALAMIYOR.  Asagidaki OLCUMLER gecerli, HUKUM degil.
   §12b hafiza kosusu duserken sorulan soru: optimizasyon neden
   donmeleri sondurmeyi SECTI?  Cevap kayipta, aritmetikle.

   1) OLGU, PUANLANAN KONUMLARIN %15,04'U.  (kagit, GPU YOK)
      Korpusun kendi kalip karisimi (17 bildirim + 8 soru) x 7.381
      olgu = 184.525 cumle, 2.356.137 birim konumu:
          SABLON / DIL          %43,2
          OZNE anilmasi         %23,2
          CEVAP birimleri       %33,6
      Cevap birimlerinin hepsi olgu istemiyor: ek UNLU UYUMUNDAN
      deterministik, soyadin bir kismi addan cikarilabilir. 1.608 ad
      uzerinde onek-dallanmasi sayildi -- ad basina 2,85 birim, bunun
      1,92'si BELIRSIZ.   184.525 x 1,92 / 2.356.137 = %15,04.

   2) BUTUN OLGULARI BILMENIN DEGERI.
      uye = 2 - 2*ort(cos), konum basina ORTALAMA (§5.1/B). O %15,04'u
      cos = 1'e cekmenin kazanci  2 x 0,1504 x (1 - cos_simdi):
          cos >= 0   ->  EN COK 0,3008        cos ~ 0,5  ->  ~0,1504

   3) KISAYOLUN DEGERI -- iki DENGE arasi, OLCULDU.
      (t0_d16_hafizasiz / t0_d16_hafizali, ikisi de epok 5)

      ```
                  hafizasiz   hafizali    kayiptaki fark
      uye            0,9460     0,6639        +0,2821   %44
      a2*kod         0,2672     0,0203        +0,2469   %39
      a3*duzen       0,1128     0,0045        +0,1083   %17
      a2*beta*bag    0,0062     0,0038        +0,0024    %0
      a1*dis         0,0003     0,0013        -0,0010    -
      TOPLAM         1,3325     0,6938        +0,6387
      ```

   ```
   KISAYOL (olgu SIFIR)          0,6387
   BUTUN korpusu ezberlemek     <= 0,3008
   ```

   **Kisayol, butun olgulari ogrenmenin IKI KATINDAN fazla oduyor.**
   Ve yalniz `uye`den aldigi 0,2821, olgunun EN IYI HALDE degdigi
   0,3008'in **%94'u** -- hicbir olgu ogrenmeden. Optimizasyon yanlis
   bir sey yapmadi; dogru fiyata bakip kolay olani sectI.

   HANGISI TESVIK, HANGISI MUHASEBE:
   ```
   uye     zf'ye gradyan VERIR                      TESVIK
   dis     VERIR                                    TESVIK
   bag     VERIR ama `vur` ile kapili -- capa %94'e  GERI BESLEME:
           cikinca neredeyse her adimda uygulanir,   kapandikca
           zp'yi kodlara ceker, capa daha cok atesler  hizlanir
   duzen   a, th uzerinde DOGRUDAN                   TESVIK
   kod     `zf.detach()` -- yalniz C hareket eder.   MUHASEBE
           0,2469'luk dususu cokusun SONUCU, sebebi DEGIL.
           !! a2'yi bu sayiya bakip oynatmak HATA olur.
   ```

   DUZELTME: ilk yazimda 3) "a3 x (3056,93 - 45,04) = 0,3012" diyordu.
   3056,93 BASLANGIC degeri; hafizasiz kosunun oradan zaten indigi
   varsayildi. Kayit aksini soyledi -- hafizasiz kosuda `duzen`
   epok 1'de 870'e inip sonra GERI TIRMANDI (991, 1078, 1106, 1128):
   cezaya ragmen acilar BUYUYOR, cunku uye odiyordu. a3'un gercek payi
   0,1083, yani kisayolun %17'si. Sonuc degismedi, GUCLENDI: asil
   odeme `uye`den geliyor.

   SONUCU MIMARI: kapasite eklemek bu tabloyu DEGISTIRMEZ. Eklenen her
   serbestlik, 0,64'u olgu ogrenmeden toplamanin YENI bir yolunu acar
   -- §12b'de tam bu oldu. Once FIYAT duzelir, sonra kapasite.
       ADAY L1  a3 = 0.  Kisayolun yalniz %17'sini alir -- TEK BASINA
                YETMEZ.  (§5.1/K'yi yine de kapatir.)
       ADAY L2  `uye` konum basina ESIT agirlikli olmasin. Olgu
                konumlari korpus uretecinden BILINIYOR; maske pencereyle
                tasinir.  !! Modele "burasi olgu" sinyali vermek --
                TASARIM KARARI, bedava degil.
       ADAY L3  Etiketsiz surumu: zor konuma agirlik (focal). Olgu
                konumlari zaten zor oldugu icin kendiliginden agirlik
                alir; ek sinyal YOK.
       ADAY L4  `bag` geri beslemesini kes: `vur` orani bir tavani
                asarsa bag'i kapat, ya da bag'i capa oranina bol.
                Cokusun HIZLANDIRICISI budur, kaynagi degil.
   Dordu de SINANMADI.  Kisayolun %44'u `uye`de oldugu icin L2/L3
   dogrudan oraya bakiyor; L1 ve L4 tek baslarina yetmez.
N  L1..L4 KAGITTA SINANDI -- IKISI DUSTU, BIRI KIRILGAN  (21 Eylul)
   !! DAL KAPANDI -- bkz. §5.1/O.  Dordu de FIYATI degistiriyor;
      alinamayan odulu carpanla buyutmek ise yaramaz.
   Olcut tek: degisiklikten sonra KISAYOL hala OLGUDAN cok mu oduyor?
   Zemin §5.1/M.3'ten:  kisayol 0,6387   butun olgular <= 0,3008.

   ```
   L1  a3 = 0        0,6387 -> 0,5304   hala 1,76 kat   REDDEDILDI
   L4  bag kesilir   0,6387 -> 0,6363   hala 2,12 kat   REDDEDILDI
   ```
   Ikisi de DEGER muhasebesiyle kapandi, bilinmeyen gerekmedi. L4'un
   tezi zaten "hizlandirici"ydi; hiz tezi kagitta sinanamaz (gradyan
   buyuklugu gerekir) ama DEGER olarak hicbir sey degistirmiyor.

   L2 / L3 TEK BILINMEYENE INIYOR:  c_f = olgu konumlarindaki ortalama
   cos (hafizasiz denge).  Kisit:  f*c_f + g*c_g = 0,5270   (uye'den)
   Olgu konumlarina w agirligi verilince payda SADELESIR:

       OLGU     2*f*w*(1-c_f) / (f*w+g)
       KISAYOL  2*g*(c_g'-c_g) / (f*w+g)
       ->  olgunun kazanmasi icin   w > 0,1411 / (0,1504 * (1-c_f))

   ```
    c_f    c_g     L2: gereken w    L3: gereken gamma
   0,20   0,585         1,17              0,24
   0,40   0,549         1,56              1,56
   0,50   0,532         1,88              9,58
   0,60   0,514         2,34           IMKANSIZ
   0,80   0,479         4,69           IMKANSIZ
   0,86   0,468         6,70        L2 de kaybeder
   ```

   **L3 KIRILGAN.** Focal agirlik `(1-cos)^gamma` demek; olgu
   konumlarina FAZLADAN agirlik verebilmesi icin o konumlarin DAHA ZOR
   olmasi, yani c_f < c_g olmasi sart. c_g ~ 0,53'te neredeyse sabit
   (f kucuk). c_f 0,514'u gecerse focal olgu konumlarina DAHA AZ
   agirlik verir -- duzeltmez, TERS calisir.
   Ve c_f'nin yuksek olmasi icin sebep var: model "dogru bicimde,
   dogru turden, yanlis varlik" uretiyor; ayni turden varliklar okuma
   uzayinda kumeleniyorsa yanlis cevabin cos'u yuksektir.

   ```
   L2   c_f < 0,86 olan HER yerde calisir, w 1..7.      SAGLAM
   L3   yalniz c_f < ~0,45'te; gamma hizla patliyor.    KIRILGAN
   ```
   L2'nin bedeli ayri: modele "burasi olgu" sinyali vermek, yani
   deneyin kendi sorusunu zayiflatmak. TASARIM KARARI.

   EKSIK TEK SAYI: c_f.  Kayitli `t0_d16_hafizasiz/model_t0.pt` + birim
   akisi ile ileri gecis yeter -- EGITIM YOK. Yerelde uretilemez:
   `metin_14.py` 21 Eylul'de degisti, onbellek anahtari kaydi, yerel
   uretim kosunun GORDUGU korpusu vermez (kural 9'un tuzagi).
   KABA AYRIM BILE KARAR VERDIRIR: varlik birimleri (`var_ix`, hucre
   4'te zaten kuruluyor) ile geri kalanin cos'u. Varlik birimleri DAHA
   KOLAYSA L3 dogrudan oluyor.
O  §5.1/M'NIN TEZI YANLIS -- FIYAT ZATEN DOGRU  (21 Eylul, ileri gecis)
   M "kayip olguyu satin alamiyor, once FIYAT duzelir" diyordu. Fiyat
   OLCULDU ve tersi cikti. Iki kayitli agirlik, ayni 131.072 pencere,
   hedefin cos'u birim TIPINE gore ayrildi. EGITIM YOK.
   KENDI KAPISI: ortalama cos 0,5291 -> uye 0,9418, kayitli 0,9460.

   ```
     tip       pay      cos hafizasiz   cos hafizali   uye'ye KATKI FARKI
     VARLIK   26,43%       0,1282          0,1861         +0,0306   %11
     EK       29,40%       0,7557          0,9458         +0,1118   %40
     DIGER    44,16%       0,6182          0,7711         +0,1351   %49
     TOPLAM                                               +0,2775
   ```

   1) OLGU ZATEN PAHALI.  Varlik konumlari geri kalandan 5,2 kat zor
      (kalinti 1,7436 vs 0,4887 / 0,7636). `uye` kaybinin
      0,2643 x 1,7436 = **0,4609'u**, yani **%49'u**, varlik
      konumlarinda ELDE EDILMEMIS duruyor.

   2) KISAYOL ORAYA DOKUNMADI.  Kazancinin %89'u EK ve DIGER'den.
      Varlikta 0,1282 -> 0,1861; 0,128 zaten sansa yakin (444 birim,
      16 boyutlu kure).

   3) ORTAK HUKUM:
      ```
      VARLIK'ta duran alinmamis odul   0,4609
      kisayolun uye'den aldigi          0,2775
      ```
      Model odulu SECMEDI diye almadi degil -- **ALAMIYOR**. Fiyat
      dogru yerde ve yeterince yuksek; toplayacak mekanizma yok.

   BUNUN KAPATTIGI DAL:  L1, L2, L3, L4 (§5.1/N) -- hepsi FIYATI
   degistiriyor. Alinamayan bir odulu carpanla buyutmek onu
   alinabilir yapmaz. L3'un yasadigi da bu olcumle cikti
   ((1-c_f)/(1-c_g) = 2,667) ama ARTIK ONEMI YOK: 2,667 kat agirlik,
   sans seviyesindeki bir cos'u yukseltmez.
   DAL KAPANDI -- yeniden acilmasi icin "model varlik konumlarinda
   sans ustune cikabiliyor ama cikmiyor" gosterilmeli.

   TEK KAYDA DEGER ITIRAZ: pencere akistan KEYFI yerden basliyor
   (§9.5), yani penceredeki ILK varlik anilmasi HICBIR modelce
   bilinemez. 20 puanlanan konumda ~5,3 varlik konumu var; biri
   ilk-anilma olsa odulun ~%20'si dusulur -> 0,37. Hala 0,2775'in
   ustunde. Hukum degismiyor.

   YAN BULGU: cokmus model EK konumlarinda cos 0,9458'e cikmis --
   unlu uyumu / ek morfolojisi fiilen COZULMUS. Ayni model hicbir
   cumle kapatamiyor (BICIM kapanmadi 1,0000). Morfoloji ile uretim
   AYRI seyler, ve birincisi ikincisi olmadan da olabiliyor.

   DUZELTME KAYDI: M'nin 1) ve 2) sayimlari (f = %15,04, kagit)
   GERCEK AKISLA TUTMUYOR -- olculen VARLIK payi %26,43, ve akisin
   %29,40'i EK birimi. Kagit sayimi izole cumleler + kucuk ornekten
   kurulmus bir `kok_havuzu` ile yapilmisti; gercek bolme daha cok
   ek birimi uretiyor. OLCUM GECERLI, SAYIM DEGIL.

   NEREYE:  A10 (varlik birimleri durumu oynatmiyor, %73 kendine
   donuyor) ve A11. Ariza TASIYICIDA.
P  BARIYER d ILE BUYUYOR -- d=8->16 TAKASININ MEKANIZMASI  (21 Eylul)
   !! ONCEDEN KAYDI CURUDU -- bkz. §5.1/Q.  Donme d ile buyuyor ve
      d=16'da bariyeri ASIYOR.  Bariyer HESABI gecerli, ACIKLAMA degil.
   §5.1/L: durum okuma uzayinda kendi Voronoi hucresinden cikmiyor;
   olcut EN YAKIN KOMSU acisi, d=8'de 30,5°. O sayi `d`YE BAGLI ve
   §5.1/L d=8'de olculmustu. Kagitta hesaplandi (444 nokta kurede
   duzgun -- A5: p_w yerlesimi rastgele; 12 tekrar):

   ```
     d     en yakin komsu acisi     yarisi
     8            30,6°              15,3°     <- KAPI: olculen 30,5°
    16            47,3°              23,6°
    32            59,7°              29,8°
   ```

   Ornekleme d=8'de olculen degeri BIREBIR uretiyor (30,6 vs 30,5),
   yani model dogru.

   **d 8 -> 16'da asilmasi gereken aci %55 buyudu.** Varlik donmesinin
   de %55 buyumesi icin hicbir sebep yok -- ve §5.1/O olctu: varlik
   konumlarinda cos 0,1282, sansa yakin. Takas boyle calisiyor:
       d buyudu  -> DIL kazandi (okuma uzayi genis, kaliplar ayriliyor)
                 -> BARIYER buyudu, ozne kimligi daha da gecmez oldu
                    (%35,1 -> %11,2, §5.1/L)

   NICIN DONME BUYUMUYOR -- ADAY ACIKLAMA, SINANMADI:
   `[C]` R_w ORTOGONAL, yani RIJIT. Bir varlik birimi cok farkli
   baglamlarda geciyor; R[Cem] gelen HER durumu "Cem gorulmus"
   diyen bir yere tasimali. Rijit bir donme bunu yapamaz -- cok
   noktayi tek noktaya goturemez. Cok donerse gelen durumlari
   DAGITIR, az donerse HIC yazmaz. Optimizasyonun uzlasmasi
   R ~ I, cunku hic yazmamak yanlis yazmaktan ucuz.
   Eger dogruysa: varlik biriminin isi YAZMAK, ve yazma islemi
   TOPLAMSAL olmali -- donme degil. (§4.1 ile ayni yer: izometri.)
   !! Bu, §12b'nin hafizasindan FARKLI bir yer: hafiza KODLA
   adresleniyordu (= ILISKI, §3.1b) ve HER adimda okunuyordu;
   burada onerilen sey VARLIK JETONUNDA bir yazma.

   SIRADAKI OLCUM (parametre okuma, ileri gecis bile YOK):
   d=16 kontrol noktasinda varlik donme acisi ve "kendine donen"
   orani. ONCEDEN KAYIT: bariyer %55 buyudu, donme buyumediyse
   KENDINE DONEN varlik orani %73'un USTUNDE cikmali. Cikmazsa
   bu aciklama yanlis ve tekrar bakilir.
Q  §5.1/P'NIN ONCEDEN KAYDI CURUDU -- BARIYER ACIKLAMASI d=16'DA GECMIYOR
   (21 Eylul, PARAMETRE OKUMA, ileri gecis bile yok)
   Kayit: "bariyer %55 buyudu, donme buyumediyse kendine donen VARLIK
   orani %73'un USTUNDE cikmali."   CIKMADI.

   ```
                         d=8      d=16 hafizasiz   d=16 hafizali
     komsu araligi      30,5°         47,9°            47,9°
     VARLIK donme acisi 22,1°         48,9°             1,2°
     kendine donen       %75           %43             %100
     ort |a| VARLIK     0,0293        0,0330           0,0072
   ```
   (tanimlar §5.1/L ile ayni: z0=[p_w;0], z1=R_w z0, q=norm(Pi z1);
    aci = <q,p_w>; kendine donen = argmax_v q.p_v == w)

   1) DONME `d` ILE BUYUYOR, hem de bariyerden HIZLI: 2,2 kat vs
      1,57 kat. d=16'da varlik donmesi (48,9°) komsu araligini (47,9°)
      ASIYOR.  **Voronoi bariyeri BILGI=0'i d=16'da ACIKLAMIYOR.**
   2) "R_w RIJIT, yazamaz, R ~ I'de uzlasir" aciklamam (§5.1/P) OLDU.
      d=16'da I'de uzlasmiyor, tam tersine buyuyor.
   3) BUNA RAGMEN ozne okunabilirligi %35,1 -> %11,2 DUSTU (§5.1/L).
      Yani ariza "durum HAREKET ETMIYOR" degil.
   4) HAFIZALI kosu parametre duzeyinde: 1,2°, %100 kendine donen.
      §12b'nin cokusu agirlikta da gorunuyor -- SONLU OTOMAT.

   YAN BULGU, ARANMIYORDU:  `p` bir **register_buffer** -- ogrenilmiyor.
   Rastgele atanip donduruluyor, ve |p_w| = 1,000 (min=ortanca=max),
   yani `_kos` gercekten kosinus. A5 buyudu: "p_w yerlesimi rastgele"
   degil, **rastgele VE DONMUS**. 47,9°lik komsu araligi kimsenin
   optimize edemedigi SABIT bir ozellik.

   NEREYE BIRAKTI: tasima calisiyor, okuma kaybediyor. §3.1b/adim 5
   zaten olcmustu -- ozne TAM durumdan %11,2 (sans %0,3), okunabilir
   bloktan %4,7. Yani kimlik durumda VAR, `Pi` atiyor. Darbogaz
   TASIMA degil OKUMA.
   Soru netlesti: **gizli alandaki icerik CEVAP ANINDA okumaya nasil
   ulasacak.**  §12b'nin icgudusu (izometriyi EKLEYEREK kir) TUR
   olarak dogruymus; yeri ve adresi yanlisti.
R  `Pi` IYI YARIYI ATIYOR -- ve `uye` OKUNAN BLOGU SILIYOR  (21 Eylul)
   !! MEKANIZMA IDDIASI (uye birikimli siliyor) DUSTU -- §5.1/S.
      Kayip TEK ADIMDA, iliski operatorunde.  OLCUMLER gecerli.
   §5.1/Q "kimlik durumda var, Pi atiyor" diyordu. Nerede durdugu
   olculdu: 7.381 tek adimli sinav onegi, onegin SONUNDAKI durum,
   dogrusal prob (one-hot ridge, %70/%30, top-1), KARISIK etiket tabani.

   ```
   t0_d8     D=32  d=8   -> gizli 24 boyut
     blok                OZNE           ILISKI         CEVAP
     TAM z            0,186 / 0,003   0,508 / 0,103   0,003 / 0,000
     Pi z  (OKUNAN)   0,035 / 0,003   0,199 / 0,108   0,000 / 0,001
     z_gizli (ATILAN) 0,156 / 0,002   0,385 / 0,104   0,002 / 0,001

   t0_d16    D=32  d=16  -> gizli 16 boyut   (IKI BLOK ESIT BOYUT)
     TAM z            0,045 / 0,001   0,566 / 0,108   0,003 / 0,002
     Pi z  (OKUNAN)   0,009 / 0,001   0,231 / 0,108   0,001 / 0,001
     z_gizli (ATILAN) 0,015 / 0,001   0,532 / 0,107   0,004 / 0,001
   ```

   1) ATILAN BLOK, OKUNANDAN IYI -- ve d=16'da bu BOYUT FARKINDAN
      DEGIL, ikisi de 16 boyut:
          OZNE     okunan 0,009   atilan 0,015   1,7 kat
          ILISKI   okunan 0,231   atilan 0,532   2,3 kat
      `Pi` bilgiyi yalnizca ATMIYOR, IYI YARIYI atiyor.

   2) MEKANIZMA -- `uye` OKUNAN BLOGU HER ADIM SABITLIYOR.
      Kayip Pi z_j'yi p_{w_j}'ye cekiyor; p SABIT ve RASTGELE (A5).
      Sabit bir hedefe cakilan blokta baska bilgiye yer kalmaz.
      Gizli blok cakilmiyor, kimlik orada birikiyor.
      -> `d < D` "atmak" degil, okunan blogu SILMEK.

   3) d 8 -> 16 BILGIYI OKUNAN BLOGA TASIMADI, YOK ETTI.
      TAM z (ikisinde de 32 boyut, ADIL):  OZNE 0,186 -> 0,045, 4,1 kat.
      §5.1/L'nin kaydettigi %35,1 -> %11,2 ile ayni yon ve ayni
      buyukluk mertebesi (3,1 kat) -- AYRI olcu, AYNI sonuc.
      Silinen alan 8'den 16 boyuta cikinca tasiyici 24'ten 16'ya indi
      VE icindeki kimlik 10 kat azaldi -- kapasitenin acikladigindan
      fazla.

   4) CEVAP HER YERDE SANSTA.  0,003 vs 0,000-0,002, her iki modelde,
      her uc blokta. §3.1b ile ayni: cevap hic olusmuyor.
      !! ILISKI ise gizli blokta 0,532 -- okumanin ihtiyaci olan sey
      bile cogunlukla ATILAN tarafta.

   SINIR: prob DOGRUSAL. Dogrusal olmayan yapi gozden kacabilir.
   Ama hukum "atilan blok daha iyi" KIYASLI bir hukum ve iki blok
   ayni prob, ayni boyut, ayni ornek -- kiyas gecerli.

   TASARIMA SOYLEDIGI:  tasiyici GIZLI blok. Cevap aninda okumaya
   ulasmasi gereken sey orada duruyor. Yani aranan mekanizma
   "kapasite eklemek" degil, **gizli blogu cevap aninda okumaya
   baglamak**. §12b'nin icgudusu (izometriyi ekleyerek kir) burada
   dogru yerini buluyor: yeri HER ADIM degil, adresi KOD degil.
S  KIMLIK ILISKI OPERATORUNDE OLUYOR -- ve sayim bunu ONGORMUSTU
   (21 Eylul, ileri gecis + prob, EGITIM YOK)
   Ozne okunabilirligi onek boyunca, konum konum. Onek yapisi HER
   ornekte ayni:  `0:Ibrahim 1:Yilmaz 2:-in 3:kardes 4:-i 5:kim 6:-dir 7:?`
   z_j = R[jeton j-1] uygulandiktan SONRAKI durum.

   ```
   t0_d8                            t0_d16_hafizasiz
    j  capa%   TAM    Pi   gizli     j  capa%   TAM    Pi   gizli
    0   0,0   0,035 0,035 0,000      0   0,0   0,041 0,035 0,000
    1   0,0   0,048 0,034 0,042      1   0,0   0,053 0,035 0,037
    2   0,0   0,308 0,067 0,266      2   0,0   0,235 0,062 0,150   TEPE
    3   0,0   0,289 0,084 0,247      3   0,0   0,242 0,066 0,116
    4   0,2   0,147 0,022 0,153      4  13,0   0,044 0,009 0,015   COKUS
    5   0,1   0,179 0,039 0,165      5  22,8   0,070 0,020 0,026
    8   0,0   0,163 0,027 0,137      8  43,0   0,033 0,013 0,009
   ```

   1) KIMLIK KURULUYOR. j=2'de (ad tamamlandiginda) tepe: d=8'de
      0,308, d=16'da 0,235. Sans 0,001-0,003. Yani model ozneyi
      DOGRU kuruyor.
   2) SONRA TEK ADIMDA COKUYOR, ve o adim `R[kardes]` -- ILISKI
      OPERATORU.  d=8: 0,289 -> 0,147.  d=16: 0,242 -> 0,044.
      Kademeli DEGIL. "uye birikimli olarak siliyor" tezi DUSTU;
      blok-kosegen gerekcesi de onunla duser.
   3) CAPA SEBEP DEGIL: d=8'de o adimda capa %0,2 ve dusus yine var.
      d=16'da capa %13 ile ORADA basliyor ve dususu DERINLESTIRIYOR
      (5,5 kat vs 2,0 kat) -- katalizor, kaynak degil.

   NEDEN -- ve bu SAYILMIStI.  Model dogru cevabi verebilmek icin
   her ozne s ve iliski r icin sunu saglamali:
   ```
        Pi R_r z_s  ~  p_{f(s,r)}
   ```
   `Pi R_r` SABIT, rank <= d bir dogrusal harita. Serbestligi
   Stiefel(D,d) = Dd - d(d+1)/2 = 32*16 - 136 = **376**.
   Iliski basina ozne sayisi 7.381/24 ~ 308; her biri d-1 = 15 kisit.
   ```
        kisit  308 x 15 = 4.620      serbestlik 376      12,3 KAT KISA
   ```
   Bu sayi §12b'de zaten yaziyordu (376 / 4.613). Yeni olan:
   **prob egrisi onu OLAYIN ICINDE gosteriyor** -- kimlik tam o
   operatorde olcusuz kaliyor.

   VE DAHA KOTUSU: R_r bir IZOMETRI. <R z1, R z2> = <z1, z2>, yani
   butun ikili acilari KORUYOR. Olgu tablosu ise keyfi: birbirine
   yakin iki ozne bambaska cevaplara gidiyor. Rijit bir harita bunu
   yapamaz. `Pi`nin buzmesi tek kacis yolu (§4.1 tam bunu diyor) ve
   o da 376 serbestlikle sinirli.

   HUKUM: **olgu aramasi R_r'nin ICINDE OLAMAZ.** Ayar meselesi
   degil, sayim meselesi. Kendi parametresi ve kendi adresi gerekiyor.
   -> tasarim §12c.

   ADRES NEREDE: en iyi ozne okunabilirligi j=2-3'te, yani ILISKI
   OPERATORUNDEN ONCE (d=8 0,308 / d=16 0,235). Adresleme tavani bu,
   ve tavan `R_r`'den SONRA degil ONCE okunursa gecerli.
```

---

### 5.2 ISINMA — ilk konumlar niye puanlanmıyor  (KARAR, §13/A3)

Pencere akıştan **keyfi** yerden başlıyor (§9.5), yani baştaki önek
çöp. İlk sürüm bunu *"çöp durum ilk çapada silinir"* diye
meşrulaştırıyordu; **silinmiyor** — adım 1'de `s = 0,5482`, eşik
`1 − r²/2 = 0,969`, çapa hiç tetiklenmiyor (§0 tablosu). Ortada bir
mekanizma hiç yoktu.

**Çapayı adım 1'de tetiklemek ELENDİ — aritmetik kendi kendini yiyor:**

```
s_1 = 0,5482 icin gereken r:  1 - r^2/2 < 0,5482  ->  r > 0,9506   (3,8 kat)

r = 0,2500   baslik yari acisi 14,4 derece   2048 baslik kureyi 2,6e-17 kapliyor
r = 0,9506   baslik yari acisi 56,8 derece   2048 baslik kureyi 1,05    kapliyor
```

Kaplama 1'e ulaşınca **her durum bir koda oturur** — §5'in `bağ`
teriminin kaçındığı hâlin ta kendisi: model sonlu otomata çöker.

**Karar: ilk `ISINMA` konum puanlanmaz.** `ISINMA` keyfî değil,
`atla`'dan çıkıyor. Akıştaki `i` indisli geçiş, `s ≡ 0 (mod atla)`
olan her pencerede görülür ve penceredeki konumu `j = i − s`, yani
**`j ≡ i (mod atla)`** — bir geçiş hep aynı kalıntı sınıfında kalır:

```
 W   c=0 c=1 c=2 c=3   toplam   atilan konum      (L = 24, atla = 4)
 1     5   6   6   6     23     -
 2     5   5   6   6     22     1
 3     5   5   5   6     21     1,2
 4     5   5   5   5     20     1,2,3            <- SECILEN
 5     4   5   5   5     19     1,2,3,4
 8     4   4   4   4     16     1..7
```

`W = 4` **tek** değer: tavanı ölçülen üç düşük konumu (`%45,1`,
`%52,9`, `%63,2` — §5.1/G) atan **ve** dört kalıntı sınıfını da tam
5 konumda bırakan. Yani **hiçbir geçiş eğitimden düşmüyor**; her
geçiş yalnızca çöp ofsetten puanlandığı o **tek** pencereyi
kaybediyor. Böylece "ad birimlerinin dönmesi eğitilemez" itirazı da
düşüyor: ad birimleri gradyanını almaya devam ediyor, artık çöp
önekten almıyor.

```
BEDEL    puanlanan konum  23 -> 20      (%13)
KAZANC   puanlanan en dusuk tavan  %45,1 -> konum 4
         (olculen dort nokta monoton artiyor: 45,1 / 52,9 / 63,2 / 90,8)
GENEL    W, `atla`nin KATI olmali -- degilse bir gecis sinifi
         otekilerden bir konum az puanlanir.   `ayar_14` assert ediyor.
```

Kapı **32**: kayıp `uye`si elle kurulanla birebir, ve `isin=1`inkinden
farklı. Dilim sessizce `1:`e dönerse hiçbir sayı patlamaz — bu yüzden
kapı değer özdeşliği kuruyor.

## 6. Saat `S` — ÖLÇÜLECEK DÜĞME, varsayılan KAPALI

Saat, tekrar eden birim için konmuştu. Çapa geldikten sonra tekrar
bakıldı ve gereksiz görünüyor:

```
Ayşe* Yılmaz ... Fatma* Yılmaz    iki Yilmaz FARKLI capalardan gelir
                                  -> ayrimi DURUM yapiyor
Hasan* -ın kardeş -i -nin         arada capa yok, ama D>d boslugu var
                                  -> yine DURUM yetiyor
```

Ve maliyeti var:

```
BILDIRIM   cevap capadan 3 adim sonra
SORU       cevap capadan 6 adim sonra
           -> ayni olgu IKI AYRI geometrik kisit
           -> ~3 soru bicimi  =>  kisitlar 3 KAT
```

Kapasite zaten sınırda. Varsayılan **kapalı**; açmak bir ölçüm kararı.

---

## 7. Üretim

```
acgozlu      ilk yanlis okuma her seyi bozar
tumu birden  n^L
ISIN         her adimda en iyi 50 yol
```

**İçsel çapa:** üretimde çapa, YAZILAN kelimeye değil durumun kendi
konumuna bakar. Böylece köprü **yazılmadan** çapalanır.

Bu ayrım kritik: köprüyü yazdırmak (CoT/scratchpad) `CLAUDE.md`'ye
göre **ayrı bir sorudur**; bu proje **örtük** çıkarımı araştırıyor.

```
!! OLCUM ACGOZLU KOSUYOR. `olcme_14.uret_toplu` isin aramasi
   yapmiyor. Yani butun DOGRULUK sayilari, bu bolumun "her seyi
   bozar" dedigi yontemle alindi.  §13/A4.
```

---

## 8. Parametre

```
                        n=444, D=32, d=8, K=2048, K_TAM=n
tam SO(D)     444 x 496          220.224
kod defteri  2048 x  32           65.536
-------------------------------------------
                                 285.760     (saat kapali)
```

**Kıyas yanıltıcı olur.** Kullanıcı, 20 Eylül: *"çalışamayan bir
model ile neyi kıyaslayacaksın. kıyas için çalışan bir modelin bir
yönünü — mesela hız ve ya hacim, parametre etkisi — kıyaslarsın.
çalışmamış bir model analiz için lazım, o kadar."*

Modelin iddiası "daha az parametreyle aynı iş" değil — iddia **hesap
yığınının yerine nicelemeyi koymak**. Niceleme çalışmazsa parametreyi
büyütmek de kurtarmaz.

---

## 9. Eğitim

```
9.1 PARAMETRELESTIRME
    R dogrudan optimize edilemez (Adam ortogonalligi bozar).
    tam SO(D)     R = exp(ters simetrik),  torch.matrix_exp
    tek duzlem    KAPALI FORM (Rodrigues), matrix_exp gerekmez:
      R = I + sinθ (v u^T - u v^T) + (cosθ - 1)(u u^T + v v^T)

9.2 GRADYAN SONMEZ
    Capalar arasi dz/dz bir donmedir, tekil degerleri 1.  (kapi 6)

9.3 OGRETMEN ZORLAMASI
    w_{j-1} gercek birim; z_j onekin kapali formlu bileskesidir,
    ozyineleme yok. Kayip yine YOLUN TAMAMINA.

9.4 BASLANGIC
    Tipik donme acisi BIR KOMSU ARALIGI olsun:
      aralik ~ n^(-1/(d-1)),   sigma = aralik / sqrt(D).   HESAP.

9.5 VERI
    Akistan PENCERE, uzunluk L, kesme araligi `atla`.
    Semantik bolme YOK -- pencere keyfi yerden baslar.
    OLCULDU: sinavin sordugu sey SORU + CEVAP ve ikisi AYNI zincirde
    olmali. 648.281 soru-cevap cifti, ortalama 15,2 birim:
       L=16 -> %77,2    L=20 -> %97,3    L=24 -> %99,5
    L=16 OLCULMEDEN secilmisti ve ciftlerin %22,8'inde zincir cevaba
    varmadan kesiliyordu.  Simdi L=24.
    `atla` modelin ogrenebilecegini DEGISTIRMEZ, ayni gecisin epok
    icinde kac kez gradyan verdigini belirler.

9.6 IKI ADIMLI EGITIM YOK
    Cikarim yalniz sinavda olusur; boylece 3.1'in ongorusu temiz
    kalir ve gercekten dusebilir.

9.7 r ve delta EGITIMDE  --  ILK SURUM TERSINI SOYLUYORDU
    Ilk surum "r, delta egitimde YOK, tutulan bolmede aranir" diyordu
    IKI YERDE (§1 ve §9.7), ama §5'in L_dis'i delta'yi kullaniyor ve
    kod `yol(X, r)` ile capayi egitimde calistiriyor. Kod §5'i
    izliyor. Ikisi de hic ARANMADAN sabit kullanildi.

9.8 MALIYET  -- OLCULDU 20 Eylul, L4
    L=24, atla=4, batch 8192, 470 adim/epok:
      donme()          1,5 ms
      yol()           17,9 ms
      + kayip ileri   23,0 ms
      + geri + adim   87,6 ms        <- geri gecis 65 ms, ileriden 2,8 kat
      epok            27 sn
    TF32 fark vermiyor (88,3 vs 87,9): darbogaz FLOP degil BELLEK.
    Batch supurmesi 4096->32768'de is 8 kat, sure 8,08 kat -> GPU bagli.
    Sicak dongudeki buyuk gecici tensor: ileri geciste 4,97 GB ->
    1,23 GB (kod arama no_grad, 2-2x ve clamp reduksiyondan SONRA).
```

---

## 10. Açık sorular

```
A1  D, d, K kac?   (32, 8, 2048) kagit onerisi.
A2  r kac?         Kodlar arasi mesafenin yarisindan kucuk olmali.
A3  delta, a1, a2, a3 kac?   §5.1/A ve /B'den sonra YENIDEN sorulmali.
A4  Saat acilacak mi?        §6 -- 3x kisit maliyeti.
A5  Kapali/acik sinir NEREDE?  Artik K_TAM = n; bolme bir dugme.
A6  OLU KOD: 2048'in 329'u kullaniliyor. Yeniden baslatma?
A7  Isin genisligi -- ve olcum NIYE acgozlu kosuyor (§7).
A8  p_w YERLESIMI: `randn` mi, ITME ile yayilmis mi?
    OLCULDU: en kotu cift 14,2 -> 44,8 derece (3,2 kat).
A9  §3.2 reddetme kodda YOK.
```

---

## 11. Bizim verimiz NEYİ SINIYOR

Mimari genel; korpusumuz sınav. Dört düşebilir soru:

```
S1  KOD DEFTERI VARLIKLARI KESFEDIYOR MU?
    Grafi biz yazdik, cevabi BILIYORUZ. Kodlar varliklara oturuyorsa
    niceleme gercek.                            <- HUKUM VEREN SORU

S2  cikarim ~ a^2 TUTUYOR MU?
    3.1'in ongorusu. Tutmuyorsa teorem dogru ama ONKOSULU
    saglanmiyor demektir.

S3  OLMAYAN BIRLESIM REDDEDILIYOR MU?     3.2 -- once kodlanmali (A9).

S4  KAPALI/ACIK BOLMESI FREKANSTAN CIKIYOR MU?
    4.3 -- OLCULDU, CIKMIYOR. Frekans olguyu yanlis tarafa koyuyordu.
```

`S1` tek başına hüküm verir: niceleme çalışmazsa mimarinin ana
iddiası düşer ve parametre sorusu anlamsızlaşır. **Ve S1'in ön koşulu
şu an sağlanmıyor** (§0: çapa hiç tetiklenmiyor).

---

## 12. Neye benziyor, nesi yeni

```
donme ile kompozisyon      RotatE, donme tabanli graf gommeleri
vektor kuantalama          VQ-VAE ailesi
en yakin nokta ile okuma   vektor niceleme
isin aramasi               standart
```

Yeni olan birleşim: **dizi durumunun izometrik olması ve periyodik
olarak öğrenilmiş bir koda oturması**, kaybın da bir sonraki kelimeye
değil yolun tamamına bakması. Ölçülmüş bir karşılığını bilmiyorum;
literatürde aranmalı — "benzerini gördüm" diye yazmıyorum.

---

## 12b. OLGU HAFIZASI — TASARIM, KOD, ve KOŞU  (21 Eylül)

> Tasarım aşağıda OLDUĞU GİBİ duruyor; kod yazıldı, koşuldu ve
> **düştü** — hüküm bölümün SONUNDA. Dayanaklar: `[Ö]` ölçüldü,
> `[H]` hesap, `[Ç]` çıkarım — sınanmadı.

### Niçin

`[Ö]` Sınav öneğinin sonundaki durum **soruyu taşıyor, cevabı
taşımıyor** (§3.1b, adım 5):

```
ILISKI   7,4 kat sans ustu    OZNE 4,7 kat    CEVAP 1,8 kat
```

`[Ö]` Ve mimaride olguyu tutabilecek bir yer yok — kapasite sayımı:

```
                          serbestlik      kisit        durum
iliski operatoru R[r]            376      4.613     12 KAT KISA
kod defteri C              65.536      110.715    1,7 KAT KISA
MODELIN TAMAMI            285.760      110.715    2,6 kat bol
```

Toplam kapasite yeter, ama **adreslenemiyor**: parametrelerin %77'si
dönmelerde, her dönme bütün bağlamlarında paylaşılıyor ve ortogonal
olmak zorunda.

### Ne eklenir

```
HAFIZA   a_j = softmax( <zp_j , Kmem> / tau )         Kmem  (M, D)
         m_j = a_j @ Vmem                             Vmem  (M, D)
         z_j = normalize( z_j + beta_m * m_j )
```

`[H]` **Boyut**, kısıt sayımından:

```
kisit  7381 olgu x (d-1) = 110.715
M=2048  (C yeniden kullanilir)   2048 x 2D = 131.072   1,18 kat
M=3456  en az                                221.184   2,00 kat
M=4096  rahat                                262.144   2,37 kat
```

`[Ç]` **Önerilen: `Kmem = C`** — kod defteri zaten var ve zaten
adresliyor. Eklenen yalnız `Vmem`: `2048 x 32 = 65.536` parametre,
modele `+%23`. Kapasite `1,18 kat` — dar ama yeter; yetmezse `M`
büyür.

### İKİ İŞİ BİRDEN YAPAR

`[Ç]` §4.1'in şartı **`d < D` değil**, "okuma İZOMETRİ OLMASIN".
Şu anki çözüm bunu **atarak** sağlıyor (`Π`), ve atılan şey kayıp.
Hafıza okuması izometriyi **ekleyerek** kırar:

```
z -> q artik izometri DEGIL, cunku m eklemeli/dogrusal-olmayan
-> §4.1'in celiskisi KALKAR
-> `d < D` ZORUNLU olmaktan cikar, tasiyici kucultmek gerekmez
```

`[Ö]` Bu önemli, çünkü `d`'yi oynatmak **takas** olduğu ölçüldü:

```
d=8 -> 16   DIL kazandi (kalip 2,8 kat, kapanan cumle 4/20 -> 11/20)
            DURUM kaybetti (ozne %35,1 -> %11,2)
Mekanizma: gizli alan TASIYICI; d 8->16 olunca 24 -> 16 boyuta indi.
```

`[Ç]` Hafıza eklenirse `D`'yi 64'e çıkarmaya (4 kat parametre) gerek
kalmayabilir. **SINANMADI.**

### Ne ÇÖZMEZ

`[Ö]` **Adresleme tavanı.** Hafıza `(özne, ilişki)` ile okunacak;
ilişki kusursuz taşınıyor ama özne `d=16`'da `%11,2` (tam durumdan),
`%4,7` (okunabilir bloktan). Mükemmel bir hafıza bile bu tavanı
aşamaz.

```
LDA ust siniri (en iyi 16 boyutlu altuzay)   %8,0
tam 32 boyut                                 %11,2 (d=16) / %35,1 (d=8)
```

`[Ö]` Bilgi 16 boyuta **sığıyor** (LDA %8,0 ~ tam %11,2); dönme onu
getirebilirdi, getirmemiş. Yani adresleme **eğitim/kayıp** tarafında
iyileştirilebilir, `D` büyütmeden.

### Açık kalanlar

```
[Ç]  C hem KILIT hem NICELEYICI olabilir mi? `kod` terimi C'yi
     k-ortalama gibi egitiyor; kilit olmak AYIRT EDICILIK ister.
     Iki amac cakisabilir.
[Ç]  Capa (sert esik) kalir mi, yoksa yumusak okumaya mi doner?
     Yumusarsa `vur` kalkar ve §3.2 (reddetme = kod uyeligi)
     dayanaksiz kalir -- ki zaten uygulanmamis (T3).
[Ö]  Kapasite 1,18 kat DAR. Yetmezse M buyutulur.
```

### ÖNCEDEN KAYIT — ilk koşu neye karar verir

```
DEGISEN   Vmem eklenir, okuma EKLEMELI olur.  d=16 KALIR.
          Tek degisken; d ile birlikte oynatilmaz.

HUKUM     BILGI tam > 0,0000   -> hafiza DOGRU yon
          BILGI tam = 0,0000   -> ya adresleme tavani (%11) bagladi,
                                  ya hafiza yanlis kuruldu; AYIRMAK
                                  icin kimlik probu birlikte okunur
BAKILIR AMA HUKUM VERMEZ
          uye (boyutlar/terimler degisti, kiyaslanamaz)
          BICIM  (d=16 zaten degistirmisti)
          capa orani, kullanilan kod sayisi
NE YAPILMAZ
          M, tau, beta_m birlikte taranmaz. Bir kosu, bir karar.
```

### SONUÇ — KOŞULDU, HÜKÜM: **HAYIR**  (21 Eylül, commit `6057207`)

Önceden kayıt uygulandı: tek değişken (`Vmem`), `d=16` kaldı,
`M, tau` taranmadı. 302 sn, 5 epok.

**EĞİTİM — model yolu BIRAKTI:**

```
        uye      duzen     capa
bas    1,9946   3056,93    %0,00
epok 1 0,6941     77,09    %93,36
epok 2 0,6832     49,62    %94,79
epok 3 0,6742     45,80    %94,81
epok 4 0,6712     41,67    %95,07
epok 5 0,6639     45,04    %94,35
```

`duzen` 68 kat düştü, `capa` %95'e çıktı: dönmeler söndü ve durum her
adım aynı birkaç koda oturuyor. Yani `z_j = capa(R[w] z_{j-1})`
zinciri fiilen çalışmıyor — model YOL olmaktan çıktı.

**ÖLÇÜM:**

```
              hafizasiz d=16    hafizali
kalip            0,2372          0,0000    COKTU
ek               0,4344          0,0000    COKTU
kapanmadi        0,3413          1,0000    HICBIR cumle kapanmiyor
ek uretimi        6.460              63    %99 azaldi
tip              0,0998          0,3962    yukseldi -- n=63 uzerinde
BILGI tam        0,0000          0,0000    DEGISMEDI
```

**HÜKÜM.** Kayda göre `BILGI tam = 0,0000` → hafıza YANLIŞ YÖN.
Üstelik `d=16`'nın kazandırdığı dili de götürdü: bu bir takas değil,
**iki alanda birden kayıp**.

`[Ö]` Önceden kayıt iki şık sayıyordu (adresleme tavanı / hafıza yanlış
kuruldu) ve ayırmak için kimlik probu öngörüyordu. **Eğitim logu üçüncü
bir şık gösterdi ve probu gereksiz kıldı:** model adreslemeye hiç
gelmedi, ondan önce çöktü. İki şık da bu koşuda SINANMADI.

**TASARIM HATASI — nerede yanlış düşündüm:**

```
YAZDIGIM     "V sifirdan baslar, yani kosu hafizasiz modelle
              OZDES baslar -- guvenli."     (kapi 36 bunu dogruluyor)
ATLADIGIM    Sifir baslangic BASLANGICI guvenli kilar, DENGEYI degil.
             `uye` dusurmenin iki yolu var: R'yi dogru dondurmek,
             ya da V'yi doldurup R'yi bosa cikarmak. Ikincisi DAHA
             KOLAY ve kayipta onu YASAKLAYAN terim YOK.
             Optimizasyon kolayini secti.
```

`[Ö]` Bu, §12b'nin "Ne ÇÖZMEZ" listesinde yazdığım tavandan **farklı**
bir arıza. Orada "hafıza mükemmel olsa bile adresleme %11'de bağlar"
demiştim; gerçekte olan, hafızanın yolu **ikame etmesi**.

**BUNDAN SONRASI İÇİN ŞART.** Hafıza tekrar denenecekse, `R`'nin işini
elinden almasını engelleyen bir şey gerekir. Üç aday, hiçbiri
sınanmadı:

```
[C] ADAY M1  Hafiza KAYIPTA puanlanmaz -- yalniz URETIMDE okunur.
             Ikame guduzu tamamen kalkar; ama V'yi ne egitir?
[C] ADAY M2  `duzen` / dogrusal-olmayanlik tabani: R kimlige
             yaklasirsa CEZA. Ikameyi pahalilastirir, yasaklamaz.
[C] ADAY M3  Hafiza yalnizca SON adimda (cevap uretilirken) okunur;
             yol adimlarinda kapali. Zincir R'de kalir.
```

`[Ç]` M3 en ucuzu ve niyeti en iyi karşılayan: hafıza `(özne, ilişki)`
durumundan **cevabı** getirsin diye tasarlandı, yolu yürüsün diye
değil. Şu anki kod onu her adımda okuyor — bu kodun kendi hatası,
tasarımın değil.

## 12c. OLGU ARAMASI — TASARIM 2  (21 Eylül, KOD YAZILMADI)

> `[Ö]` ölçüldü · `[H]` hesap · `[Ç]` çıkarım, sınanmadı.
> §12b'nin yerine geçmez — §12b **koşuldu ve düştü**; bu tasarım
> onun ölçülen üç arızasına karşı kurulmuştur.

### Niçin — zincir, hepsi ölçülü

```
BILGI tam = 0,0000
 <- cevap hicbir blokta YOK, sansta            §5.1/R
 <- olgu aramasi R_r'nin ICINDE olamaz         §5.1/S  SAYILDI
      Pi R_r  rank<=d, serbestlik 376
      kisit   308 ozne/iliski x 15 = 4.620     12,3 KAT KISA
      ustune R_r IZOMETRI: butun ikili acilari koruyor,
      olgu tablosu ise keyfi -- rijit harita yapamaz
 <- ve olcum bunu OLAYIN ICINDE gosteriyor     §5.1/S
      ozne okunabilirligi j=2'de TEPE (0,308 / 0,235),
      j=4'te -- tam R[iliski]'den sonra -- COKUS (0,147 / 0,044)
```

`[Ö]` **Fiyat mesele değil** (§5.1/O): `üye` kaybının %49'u zaten
varlık konumlarında alınmamış duruyor. Kapasite/ağırlık kolu kapalı.

### Adres — ve neden tavan 0,044 değil 0,235

`[H]` `R_r` izometri olduğu için, **tek bir ilişki içinde**
`zp = R_r z_s` konfigürasyonu `z_s` konfigürasyonuna **birebir
eştir** — bütün ikili açılar korunur. Prob'un j=4'te çökmesi, 24
ilişkiyi tek bir doğrusal çerçevede okumasından. Bir hafıza
ilişkileri **ayırmak zorunda değil**: farklı ilişkiler uzayın farklı
yerlerine düşer ve yuvalar oraya yerleşir.

```
ADRES = zp_j = R[w_{j-1}] z_{j-1}
        iliski jetonunda  zp = R[iliski] z_{ozne}
        yani adres ZATEN (ozne, iliski).  ETIKET GEREKMEZ --
        jetonun kendisi ikinci bileseni tasiyor.
TAVAN   tepedeki ozne okunabilirligi:  d=8  0,308   d=16  0,235
        (capa hasari duselecek: d=16'da o adimda %13 atesliyor)
```

`[Ö]` Bu, §12b'nin `%11,2` diye yazdığı tavandan **2,1 kat** yüksek.
§12b tavanı **çöküşten sonraki** durumdan okumuştu — yanlış yerden.

### Ne eklenir

```
zp = R[w_{j-1}] z_{j-1}
a  = softmax( <zp, K> / tau )        K (M, D)   YENI parametre
m  = a @ V                           V (M, D)   V(0) = 0
z  = normalize( capa(zp) + m )
L += a4 * max(0, ates_orani - B)^2   BUTCE  (duz L1 SINANDI, BOS)
```
`M = 8192`, `B = 0,30`, `a4 > 1,30` -- gerekceleri asagida.

`[H]` **Boyut — ilk yazdığım sayım YANLIŞ ölçüyü saydı.**
Önce şöyle yazmıştım: kisit 7.381 × (d−1) = 110.715, `M = 2048`
→ 131.072 parametre, 1,18 kat. **Parametre sayısı doğru ölçü değil.**
`tau = 0,02` ile softmax neredeyse TOP-1; üretilebilecek AYRI çıktı
sayısı ~ `M`.

```
uretilmesi gereken AYRI cevap yonu       1.577
ayirt edilmesi gereken ADRES             7.381
  ayni cevabi paylasan adresler AYNI yuvayi kullanabilir, ama
  ancak zp uzayinda yakinlarsa -- (ozne,iliski) keyfi, yakin
  olmalari icin sebep YOK
EN IYI HAL   M >= 1.577      EN KOTU HAL   M >= 7.381
```

`M = 2048` en iyi halin ust ucunda, yani IYIMSER. **`M = 8192`**:
`2 x 8192 x 32 = 524.288` parametre, model `285.760 -> 810.048`
(**+%184**). Buyuk bir degisiklik ve oyle yazilmali; `M`
kucultulebilir ama once adreslerin kumelenip kumelenmedigi
olculmeli (ayni prob makinesi yapar).

### §12b'nin ÜÇ arızasına karşı

```
1  ADRES YANLISTI.  K = C idi; C, `kod` terimiyle k-ortalama gibi
   egitiliyor ve OLCULDU (§3.1b): ILISKIYI kodluyor (+1,680 bit),
   olguyu DEGIL (+0,318). Yani hafiza iliski basina tek satirlik
   bir tabloya donmustu.
   -> K artik SERBEST parametre. `kod` terimi ona dokunmaz.
   -> Ve §12b'nin kapasite hesabi 2048 yuva varsaydi; olculen
      kullanilan kod sayisi 166 idi, yani hesap 12,3 kat iyimserdi.
      Serbest K'da bu bagimlilik yok.

2  HER ADIMDA OKUNUYORDU ve BEDAVAYDI.  Optimizasyon donmeleri
   sondurup (duzen 3057 -> 45) isi hafizaya yaptirdi; capa %94'e
   cikti, model SONLU OTOMATA coktu. Parametre duzeyinde de
   gorunuyor: butun donmeler 1,2°, %100 kendine donen (§5.1/Q).
   -> `a4 * ort(|m|)`: KULLANIM BEDELLI. R'nin isini elinden almak
      artik ucuz degil.
   -> DUZ L1 BEDELI SINANDI ve ARALIGI BOS CIKTI:
        AMACLANAN kullanim  pay 0,2643 x kalinti 1,7436 x tavan 0,235
                            = 0,1083 kazanc,  ates orani ~0,264   [O]
        IKAME               0,6387 kazanc,    ates orani ~1,000   [O]
        maliyet = a4 x oran  ise
             ikameyi engelle   a4 > 0,6387 / 1,000 = 0,6387
             kullanimi birak   a4 < 0,1083 / 0,264 = 0,4097
             0,6387 > 0,4097  ->  **BOS, 1,56 kat**
        Ilk yazdigim "(0 , 0,46)" araligi yalniz TAVANI hesapliyordu;
        IKAME TABANINI hic hesaplamamistim.  Duz L1 CALISMAZ.
   -> BUTCE MENTESESI:   L_haf = a4 * max(0, ates_orani - B)^2
        B = 0,30   -- olculen varlik konumu payi 0,2643'un biraz ustu,
                      SECILMEDI, o sayidan turedi                 [O]
        amaclanan kullanim  oran 0,264  ->  maliyet 0  (BEDAVA)
        ikame               oran 1,000  ->  maliyet a4 x 0,49
             ikameyi engelle   a4 > 0,6387 / 0,49 = 1,30
             ust sinir YOK -- ikinin maliyeti AYRISIYOR
        ARALIK  a4 > 1,30.  Ilk deger 3,0 (guvenli tarafta; asagi
        yonde hata IKAME demek ve onu bir kez gorduk).

3  V = 0 BASLANGICI DENGEYI DEGISTIRIYORDU.  Baslangic guvenliydi
   (kapi 36 dogruladi) ama denge degil.
   -> Bedel terimi dengeyi de sabitler: V buyumesi artik ucretli.
```

### Ne ÇÖZMEZ

```
[O] ADRESLEME TAVANI 0,235 (d=16) / 0,308 (d=8).  Kusursuz bir
    hafiza bile olgularin en fazla bu kadarini bulur. BILGI tam
    icin ust sinir budur -- 1,0 beklenmiyor.
[O] CAPA HASARI.  d=16'da iliski adiminda capa %13 atesliyor ve
    zp'yi 2048 koddan birine indiriyor. Adres o orada BOZULUYOR.
    Bu tasarim capaya dokunmuyor; dokunmak ayri bir karar (§3.2
    reddetme iddiasi capaya bagli, ve o zaten uygulanmamis -- T3).
[H] MODEL 2,8 KATINA CIKIYOR (285.760 -> 810.048).  Bu artik
    "kucuk bir ekleme" degil; kiyaslarda oyle yazilmali.
[C] CIKARIM (2 adim) icin hicbir sey yapmiyor. Tek adimli olgu
    aramasi calisirsa zincir AYRI bir soru olarak acilir.
```

### Açık kalanlar

```
[C] `tau`. §12b'de 0,02 idi ve etkin yuva sayisi olculmustu.
    K degisti, o olcum tasinmaz -- yeniden bakilmali.
[C] Okuma `capa`dan ONCE mi SONRA mi? Yukarida SONRA yazildi
    (§12b ile ayni). Ama capa adresi bozuyorsa (yukarida), okuma
    capa ONCESI zp'den adreslenip SONRA eklenmeli. Ikisi ayri
    denklem; kagitta ayrilmadi.
[C] `a4` bir MENTESE mi olmali (belli bir orandan sonra ceza)?
    Duz L1 az kullanimi da cezalandiriyor.
```

### ÖNCEDEN KAYIT — ilk koşu neye karar verir

```
DEGISEN   K (serbest), V, ve a4 bedeli.  d = 16 KALIR.  Tek paket,
          parcalari ayri ayri taranmaz.

BIRINCIL  BILGI tam.  Tavan 0,235; 0,0000'dan BUYUK ise tasarim
          DOGRU yon.  0,0000 ise DUSER.

IKINCIL -- HUKUM VERMEZ ama tasarimi ACIKLAR
   duzen   45'e cokerse §12b tekrarlaniyor -> bedel YETMEDI
           1128 civarinda kalirsa bedel TUTTU
   |m| nerede ateslyor:  varlik konumlarinda mi?  Tasarimin
           iddiasi bu.  §5.1/R'nin makinesi bu olcumu zaten yapiyor.
   BICIM   §12b'de cokmustu (kalip 0,2372 -> 0,0000).  Cokerse
           yine ikame var demektir.

NE YAPILMAZ
   M, tau, a4 birlikte taranmaz.  Bir kosu, bir karar.
   capa'ya DOKUNULMAZ -- ayni kosuda iki degisken olmaz.
```

---

## 13. Kod ↔ denklem mutabakatı

Ayrışmaların tam listesi. **K** = kod haklı, denkleme yazıldı.
**T** = teori haklı, kod düzeltilmeli. **A** = açık, karar verilmedi.

> §0 tablosunun **adım adım** denetimi ve o denetimde bulunup sırası
> gelmediği için park edilen kalemler: `ADIM_ADIM.md`. Buradaki tablo
> ayrışmaları tutar, oradaki dosya denetimin kendi defteridir.

```
    NE                                        KIM   NEREDE
K1  z_0 = [p_{w_0}; 0] tanimi                  K    §1
K2  C kurede (c_k = C_k/|C_k|)                 K    §1
K3  esik ic carpimda (1 - r^2/2)               K    §1
K4  capa BILESKEDE (3.1'in kosulu)             K    §3.1
K5  L_capa'da zp, z degil                      K    §5.1/D
K6  L_dis'te TABAN + maske (NaN)               K    §5.1/E
K7  K_TAM = None (AYRIM YOK), frekans ayrimi   K    §4.3, kapi 31
    olguyu yanlis tarafa koyuyordu.  Deger
    451 idi (sozluk 444): "hepsi"yi TESADUFEN
    soyluyordu; `None` niyeti dogrudan yazar.
    Kapi 31 artik "int ise sozlukten KUCUK
    olmali" diye yasakliyor.  (21 Eylul)
K8  L = 24, atla = 4                           K    §9.5
K9  kod terimi HER ADIMDA                      K    §5.1/C  -- yazildi,
                                                    ama SONUCU acik (A6)
K10 ISINMA = 4, cop onek PUANLANMIYOR          K    §5.2, kapi 32
    (eski A3; kapandi 20 Eylul)
K11 delta = 0,40 DEGISMIYOR -- incelendi       K    §5.1/A
    (eski A2; calisan rejimde hedef 0,117
     en yakin rakip 0,346'da, itmeye GEREK YOK)

T1  olcum ACGOZLU kosuyor, §7 ISIN diyor       T    §7, A7
T2  §9.7 "r, delta egitimde yok" YANLIS        T    §9.7 -- duzeltildi
T3  §3.2 reddetme kodda YOK                    T    A9

A1  a1/a2/a3 OLCULMEDEN secildi                A    §5.1/B, K, M
    (uye ORTALAMA, dis TOPLAM; ve OLCULDU:
     750. adimdan sonra kayip DUSERKEN sira
     KOTULESIYOR -- kazanc kod+duzen'den)
    !! OLCULDU (§5.1/M): BUTUN olgulari bilmek
    kayipta EN COK 0,3008 eder; §12b'nin kisayolu
    OLGUSUZ 0,6387 etti -- iki kati. Payi: uye
    %44, kod %39 (muhasebe), duzen %17.
    Fiyat duzelmeden kapasite eklemek bosuna.
A12 d TEK BASINA OYNATILMAZ -- TAKAS           A    §5.1/L
    d=8->16 dili kazandi, durum ozne
    kimligini kaybetti (%35,1 -> %11,2).
    Gizli alan TASIYICI; d buyuyunce kuculuyor.
A11 OLGU ARAMASI MIMARIDE YOK                  A    §3.1b, §5.1/S
    -> §12b: tasarim + kod + KOSU.  DUSTU.
    -> §12c: TASARIM 2, uc arizaya karsi kuruldu.
       SAYILDI (§5.1/S): Pi R_r 376 serbestlik,
       4.620 kisit -- arama R_r'nin ICINDE OLAMAZ.
       Adres zp (= ozne,iliski), tavan 0,235.
       BILGI tam 0,0000 kaldi, BICIM coktu;
       hafiza R'yi IKAME etti (duzen 3057->45).
       Mimaride hala YOK -- ve eklerken yolu
       korumak SART (§12b/M1-M3).
    durum soruyu tasiyor (ozne %63'te 11,3 kat
    sans ustu, iliski neredeyse kusursuz), ama
    CEVAP ne durumda ne kodda. Kod defteri
    ILISKIYI kodluyor (166 kod / 24 iliski).
    (ozne,iliski)->cevap arayan sey YOK.
A10 VARLIK birimleri durumu OYNATMIYOR         A    §5.1/L, O
    (%73 kendine donuyor; kapi 31 SINIFI
     siniyor, ogrenilen BUYUKLUGU degil)
    !! ARTIK BIRINCI SIRADA.  OLCULDU (§5.1/O):
    varlik konumlarinda cos 0,1282 -- sansa
    yakin, ve `uye` kaybinin %49'u ORADA
    ALINMAMIS duruyor. Fiyat dogru, tasiyici yok.
    Kayip tarafindaki butun adaylar (L1..L4) bu
    olcumle KAPANDI.
A4  L_duzen u, v'yi kapsamiyor                 A    §5.1/F
A5  p_w RASTGELE ve DONMUS (buffer)            A    A8, §5.1/Q
    ogrenilmiyor; |p_w| = 1,000 tam, yani `_kos`
    gercek kosinus. Komsu araligi (d=8 30,5°,
    d=16 47,9°) kimsenin oynatamadigi SABIT.
A6  ek yuzeyleri AYRI token oldu               K    -- veri katmani;
    (`-ın` ile `-in` ayri), unlu uyumu               `ek_14`, 20 Eylul
    modelin girdisinde
```

### Bunun tekrarlamaması için

Kapılar **kodu** sınıyordu, belgeyi değil. 32 kapının 32'si geçerken
20 ayrışma vardı. Kural:

> **Bir iddia `DENKLEM.md`'ye yazılıyorsa, onu sınayan kapının adı
> yanında yazılır. Kapısı olmayan iddia, iddia değil temennidir.**

Ve kapı **teoreme değil İHTİYACA** kurulur:

```
teoremi sinayan kapi          ihtiyaci sinayan kapi
|Ra - Rb| = |a - b|      ->   |ΠRa - ΠRb| COKMEMELI
capa sonrasi durum ayni  ->   capa EMICI OLMAMALI
uzaklik kucultuluyor     ->   dogru cevap KACINCI SIRADA
```
