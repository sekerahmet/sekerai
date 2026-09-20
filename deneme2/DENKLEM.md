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
DONME       zp_j = R_{w_{j-1}} z_{j-1}
KOD ARAMA   k_j  = argmax_k <zp_j , c_k>,    c_k = C_k/|C_k|
CAPA        vur_j = [ <zp_j , c_{k_j}> > 1 - r^2/2 ]
            z_j  = vur_j ? c_{k_j} : zp_j
OKUMA       q_j  = Π z_j / |Π z_j|
            w_j  = argmax_w <q_j , p_w>
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

**Belgede olmayan, kodda olan üç şey — artık burada:**

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
   ACIK -- care karari verilmedi.

   !! CAPA BUNUN SEBEBI DEGIL. Ayni uretimlerde capa HIC tetiklenmedi
   (s en fazla 0,89, esik 0,969) -- §3.1'in "emici durum"u DEGIL.
   Ve K_TAM = n oldugu icin "tek duzlem" aciklamasi da elendi.
   !! EGITIMDE capa %7,61, URETIMDE %0,00. §3'un mekanizmasi tam
   IHTIYAC DUYULAN yerde yok.

   !! KAPI 31 BUNU GOREMIYOR: operatorun SINIFINI sinar ("varlik
   birimi TAM SO(D) almali"), ogrenilen BUYUKLUGUNU degil. Tam
   SO(32) olup acisi 0,46'da kalan bir donme, tek duzlemli olmaktan
   iyi degildir. §13'un kendi kurali: "kapi teoreme degil IHTIYACA
   kurulur" -- ihtiyac "varlik birimi durumu OYNATMALI".
   ACIK -- esik ve care karari verilmedi.
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
K7  K_TAM = n  (frekans ayrimi yanlis)         K    §4.3
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

A1  a1/a2/a3 OLCULMEDEN secildi                A    §5.1/B, §5.1/K
    (uye ORTALAMA, dis TOPLAM; ve OLCULDU:
     750. adimdan sonra kayip DUSERKEN sira
     KOTULESIYOR -- kazanc kod+duzen'den)
A10 VARLIK birimleri durumu OYNATMIYOR         A    §5.1/L
    (%73 kendine donuyor; kapi 31 SINIFI
     siniyor, ogrenilen BUYUKLUGU degil)
A4  L_duzen u, v'yi kapsamiyor                 A    §5.1/F
A5  p_w yerlesimi rastgele                     A    A8
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
