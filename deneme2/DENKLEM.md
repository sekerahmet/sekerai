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
T  ADRES KUSURSUZ -- TEK BOZAN `CAPA`.  §12c'nin TAVANI YANLISTI
   (21 Eylul, ileri gecis + k-ortalama, EGITIM YOK)
   Olcu: 7.381 tek adimli onek, adres = onegin SONUNDAKI durum.
   Yuva tavani = M yuvaya k-ortalama, her yuva COGUNLUK cevabini
   verirse kac olgu dogru olur.
   !! Bu olcu DOGRUSAL PROB DEGIL -- yuva atamasi ortak bir dogrusal
   cerceve istemiyor, yani §5.1/S'deki havuzlama sorunu BURADA YOK.

   ```
                      capa     cakisik(cos>0,999)  M=2048 M=4096 M=7381
     t0_d8            ACIK          %0,1           0,282  0,556  0,999
     t0_d8            KAPALI        %0,0           0,282  0,556  1,000
     t0_d16 hafizasiz ACIK         %41,8           0,172  0,333  0,588
     t0_d16 hafizasiz KAPALI        %0,0           0,287  0,558  1,000
   ```

   1) DONMELER ADRESI HIC BOZMUYOR.  Capa kapatilinca cakisma
      %41,8 -> %0,0 ve tavan 1,000. Izometri teoremi gorunur hale
      geldi: R_r butun ikili acilari koruyor, adres kayipsiz.
   2) TEK BOZAN CAPA.  d=16'da onek sonunda %37 atesliyor ve
      adreslerin %41,8'ini birbirinin AYNISI yapiyor. Bedeli
      tavanin **%41'i** (1,000 -> 0,588).
   3) `d` ADRES ICIN ONEMSIZ.  Capa kapaliyken d=8 ve d=16 AYNI:
      1,000. d=8'in iyi gorunmesi capanin orada atesmemesinden.
   4) SIKISTIRMA YOK.  Tavan M ile neredeyse DOGRUSAL (0,282 ->
      0,556 -> 1,000). Cevap adresin keyfi fonksiyonu; **olgu
      basina bir yuva** gerekiyor. §12c'nin M = 8192 duzeltmesi
      DOGRU, ve artik tahmin degil.
      !! M >= N'de saflik ZATEN 1'e gider; oradaki bilgi saflik
      degil CAKISMA ORANI. Capali halde M ne olursa olsun 0,588'i
      gecemez.

   §12c'NIN TAVANI YANLISTI -- ve KOTUMSERDI.  "0,235" yazmistim;
   o sayi DOGRUSAL PROBun tavani ve slot hafizasi icin YANLIS OLCU.
   Dogru tavan: capa ACIK 0,588, capa KAPALI 1,000.
   -> "capa'ya DOKUNULMAZ" karari da DUSER: capa artik tasarimin
      en buyuk tek kaybi.

   [C] BIRLESTIRME -- SINANMADI.  Hafiza okumasi zaten "en yakin
   anahtar + deger". Capa da "en yakin kod". Ayni makine.
        capa    z <- C[k]        DEGISTIRIR -- adresi yok eder
        hafiza  z <- z + V[k]    EKLER      -- adresi korur
   Ikisi TEK mekanizma yapilabilir: capayi kaldir, yerine toplamsal
   okuma koy. §4.1'in sarti ("okuma izometri olmasin") boylece
   ATARAK degil EKLEYEREK saglanir -- §12b'nin icgudusu, dogru yerde.
   Bedeli: §3.2 (reddetme = kod uyeligi) dayanaksiz kalir; ama o
   zaten uygulanmamis (T3).
U  8.192 YUVA AYRILDI, 10'U KULLANILDI -- ve §12c'nin HUKMU ERKENDI
   (21 Eylul, cok ornekli iz + ileri gecis, EGITIM YOK)
   Mimari izlenebilir oldugu icin tek ornek yerine 4 OZNE x 3 ILISKI
   izlendi, sonra 7.381 onegin tamamina bakildi.

   ```
                            cakisik  ATESLEYEN  M=2048 M=4096 M=7381
     hafizasiz (capa ACIK)   %41,8      166      0,172  0,333  0,588
     12c ama V SUSTURULDU     %0,1      660      0,281  0,557  1,000
     12c GERCEK              %17,9       10      0,289  0,558  1,000
   ```
   (ilk satir §5.1/T'nin "capa ACIK" satiriyla birebir: 41,8 / 0,588)

   1) ADRES BOZULMAMIS.  Yuva tavani 1,000, V susturulmus haliyle AYNI.
      "Toplamsal hafiza capanin yerine gecip adresi yok ediyor" tezi
      SINANDI ve DUSTU.
   2) **8.192 yuvanin 10'u atesliyor** -- 7.381 cevap konumunun
      tamaminda. 4x3 izde 12 sorunun tamami YALNIZ IKI yuvaya dusuyor
      (7940 ve 1708), ve hangisine dustugu ne ozneyi ne iliskiyi
      izliyor.
   3) KENDI KENDINI BESLIYOR: V susturulunca 660 yuva atesliyor, V
      acikken 10. Ayni K, farkli yorunge -- yani hafizanin CIKTISI
      durumlari birkac havzaya suruyor.

   FIILEN KULLANILAN KAPASITE:
   ```
   ayrilan     8.192 yuva x 32 = 262.144 sayi
   KULLANILAN     10 yuva x 32 =     320 sayi
   kisit                          110.715
                                  346 KAT kisa
   ```

   SEBEP -- ve bu BENIM KARARIM:
   ```
   `kod` terimi ACIK   -> K niceleyiciye zorlanir ve OLCULDU (§3.1b)
                          ki ILISKIYI kodluyor, olguyu degil
   `kod` terimi KAPALI -> K'yi YAYAN hicbir kuvvet kalmiyor;
                          kazanan yuva kazanmaya devam ediyor
   ```
   `a2 = 0`'i olculmus bir gerekceyle kapattim (§12c/1). Gerekce
   dogruydu; ama kaldirdigim sey ayni zamanda **anahtarlari yayan tek
   kuvvetti**. Iki ucta da olgu yok, ters sebeplerle.

   BUTCENIN BIR KISMI ILK ADIMDA HARCANIYOR:  izde her ornekte
   `|m|` j=1'de 4,79, sonra 0,4.  Sebep mimaride: pencere basinda
   `z = [p_{ilk jeton} ; 0]`, gizli yarisi TAM SIFIR -- cok ayirt
   edici ve HER ZAMAN ayni turden bir durum. Hafiza oraya yigilmis.
   (ISINMA o konumlari PUANLAMIYOR ama hafizanin oraya yazmasini
   ENGELLEMIYOR -- ayri sey.)

   ### §12c'NIN HUKMU GERI ALINIYOR

   §12c'nin onceden kaydi uc sart sayiyordu: *adres kusursuz,
   kapasite yeterli, fiyat dogru*. Hukmu verirken ikincisini
   "8.192 >= 7.381" diye okudum -- yani AYRILAN kapasiteyi. Iz
   gosterdi ki KULLANILAN kapasite 10 yuva.

   **Sart saglanmamis. "Mimari kol kapandi" hukmu ERKENDI ve geri
   aliniyor.**  Kol acik; ama yeniden kosmadan once yeni bir sart
   yazilir ve bu sart OLCULEBILIR:

   ```
   YENI SART   hafiza yuvalarini GERCEKTEN kullanmali.
               olcu: cevap konumunda ATESLEYEN AYRI yuva sayisi
               taban  10     (bugun)
               hedef  >> 10  ve kosu SIRASINDA izlenir, sonda degil
   ```
   Bu sart bir umut degil, egitim logunda gorunen bir sayi -- `capa`
   oraninin izlendigi gibi izlenir.
V  `BILGI tam` HIC ATESLENEMIYORMUS -- ve bir kosuda GERCEK bir
   sinyali sifir diye raporladik  (21 Eylul, uretim, EGITIM YOK)
   Kullanici: *"ama dogru baglamda cevap vermemis ki"* -- 20 ornege
   bakarken sorulan soru olcunun kendisine gitti.

   `bilgi_puanla` BIREBIR esitlik ariyordu.  Beklenen cevap CIPLAK ad
   (`sinav_yuzeyi` `_parca`nin 3. elemanini, yani `Y`yi donduruyor);
   modelin urettigi ise dilbilgisel Turkce, bildirme ekini tasiyor:

   ```
   beklenen [Agri]           span [Agri -dir]
   beklenen [Irem Ozturk]    span [Irem Ozturk -tur]
   beklenen [Melek Yilmaz]   span [Melek Yilmaz -dir]
   ```

   Yani `tam`in ateslemesi icin modelin adi yazip EK KOYMADAN noktayi
   basmasi gerekiyordu -- korpusta dilbilgisi disi bir yuzey, ve model
   tam tersini yapmaya EGITILIYOR.

   DORT KOSU EK-TOLERANSLI YENIDEN PUANLANDI (ayni kontrol noktalari,
   ayni 7.381 onek; sans 1/1577 = 0,00063):
   ```
     kosu                BIREBIR   EK-TOLERANSLI   sans kati
     t0_d8                0,0001      0,0003         0,4 x
     t0_d16_hafizasiz     0,0000      0,0004         0,6 x
     t0_d16_hafizali      0,0000      0,0008         1,3 x
     t0_d16_12c           0,0000      0,0138        21,8 x
   ```

   1) ILK UC KOSUDA HATA BIR SEY GIZLEMEMIS.  Ek toleransiyla da sansin
      altinda ya da tam sansta. O kosular hakkindaki hukumler ozunde
      DOGRUYMUS.
   2) **DORDUNCUSUNDE GERCEK BIR SINYAL SIFIR DIYE RAPORLANDI.**
      §12c, bu kolun sans ustune cikan ILK mimarisi: 21,8 kat.
      Mutlak deger hala dusuk (%1,38) ama "tam sifir" ile "sansin
      22 kati" NITEL olarak farkli iki iddia.

   BAGIMSIZ DOGRULAMA -- dogrusal prob, olcuden tamamen ayri:
   ```
                      OZNE (gercek/karisik)   CEVAP
     12c       TAM z  0,133 / 0,000           0,009 / 0,002
     hafizasiz TAM z  0,045 / 0,001           0,003 / 0,002
   ```
   Ozne 2,9 kat, cevap 3 kat iyilesmis.  Iki bagimsiz olcu ayni yonu
   gosteriyor: hafiza durumun ICERIGINI gercekten degistirdi.

   DUZELTILDI: `bilgi_puanla` kuyruktaki eki IKI TARAFTA da atiyor
   (`dogruluk` `bicim.ek`i geciriyor).  Morfoloji zaten AYRI bir
   sutunda olculuyor (`ek`); burasi BILGIYI olcer.
   KAPI 40 eklendi: dogru cevabi EKIYLE veren bir hal olcude DOGRU
   sayilmali.  Hicbir kapi bunu goremiyordu cunku hicbiri
   `bilgi_puanla`yi gercek bir cevapla cagirmiyordu.

   !! BUNUN DERSI, uretim hatasiyla (§12b) AYNI: iki kez ust uste,
   hukum veren sayiyi ureten yolun KENDISI sinanmamisti. Kapi 37
   (egitim == uretim) ve kapi 40 (olcu atesleniyor) o iki deligi
   kapatiyor.
W  DILIN KAPASITE SORUSU -- ILK KEZ SAYIYA BAGLANDI  (21 Eylul, kagit)
   Olgunun kapasite hesabini (§5.1/S: 376 serbestlik / 4.620 kisit)
   yaptik; DILINKINI hic yapmamistik.  Kullanicinin sorusu acti:

     *"Istanbul'dan Ankara'ya gidiyorsam sonrasi Izmir ya da Bursa;
     Mersin'den Ankara'ya gidiyorsam Sivas ya da Erzurum.  Bu
     guzergah dedigimiz zaten ogrenilmesi gereken sey."*

   Yani durum GECMISI degil, gecmisin DEVAMI BELIRLEYEN kismini
   tutmali.  Devami ayni olan iki guzergah AYNI duruma dusebilir --
   ve dusmelidir, sikistirma budur.

   Olculdu: korpusun kendi kalip karisimi (17 bildirim + 8 soru x
   7.381 olgu), 184.525 cumle, 2.356.137 birim.

   ```
     k   AYRI baglam   AYRI DEVAM KUMESI   dallanma   sikistirma
     1          412            232          18,87      1,8 kat
     2        7.757          1.833           5,70      4,2 kat
     3       44.077          2.164           2,49     20,4 kat
     4      108.525          2.608           1,91     41,6 kat
     5      200.309          2.145           1,63     93,4 kat
     6      302.791          1.937           1,40    156,3 kat
   ```

   1) BAGLAM PATLIYOR, DURUM DOYUYOR.  Ayri baglam 412 -> 302.791,
      ama AYIRT EDILMESI GEREKEN durum ~2.600'de doyup dusmeye
      basliyor.  Model 302.791 gecmisi degil, 2.608 sinifi ayirmali.
      **Bedava sikistirma 41 kat.**  444 cevirmeyle dilin
      ogrenilebilmesinin sebebi bu -- 444^23 ~ 10^61 yol var ama
      ayirt edilecek o kadar YER yok.

   2) DALLANMA SONUYOR:  18,87 -> 1,40.  Alti birim gecmisten sonra
      siradaki birim %71 ihtimalle ZATEN BELLI.  §5.1/J'nin dallanma
      tabani (uye >= 2 - 2/sqrt(k)) bu egrinin ustune oturur.

   3) IKI KAPASITE YAN YANA:
      ```
      DIL   ~2.600 ayirt edilecek durum   COZULDU (ek 0,92, kapanmadi 0,01)
      OLGU   7.381                         0,0138
      ```
      Olgu, BUTUN DILDEN 2,8 kat fazla ayrim istiyor.  Ama fark
      SAYIDA degil YAPIDA:
      ```
      DILDE   benzer guzergah -> benzer devam
              2.600 sinif GEOMETRIK OLARAK UYUMLU yerlesebilir
      OLGUDA  benzer ozne -> alakasiz cevap
              7.381 sinif KEYFI yerlere dusmeli
      ```
      Cevirme benzerligi KORUDUGU icin birincisini yapar, ikincisini
      yapamaz.  §5.1/S'in ispati bunun aritmetik hali.

   SINIR: sayilar IZOLE CUMLELERDEN, gercek akistan degil.  Korpusta
   cumleler sayfa icinde yan yana duruyor, yani baglam cumle sinirini
   asiyor ve k=6'daki 1,40 muhtemelen daha yuksek.  Ayni yaklasim
   daha once olgu payini %15,04 vermisti, gercek akista %26,43 cikti.
   **Mertebe dogru, ondalik degil.**  Gercek akista tekrarlanmali.
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

**ÖLÇÜM — !! BU SAYILAR GEÇERSİZ, 21 Eylül'de anlaşıldı.**
`olcme_14.uret_toplu` adımı KENDİ yazıyordu ve `mdl.V`ye hiç
dokunmuyordu; koşu hafızayla eğitilip **hafızasız** ölçüldü.
BİÇİM çöküşü bundan da gelebilir. EğİTİM LOGU geçerli
(yozlaşma orada ölçüldü), ÇIKTI ölçümü değil.
Düzeltildi: tek adım artık `Yol.adim`da, iki yol da onu çağırıyor;
kapı 37 ikisini birbirine bağlıyor.


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

## 12c. OLGU ARAMASI — TASARIM 2 → 3  (21 Eylül)

> `[Ö]` ölçüldü · `[H]` hesap · `[Ç]` çıkarım, sınanmadı.
> §12b **koşuldu ve düştü**. Bu tasarım onun üç ölçülen arızasına
> karşı kuruldu, kâğıtta sınandı, iki kere düzeltildi (§5.1/T ve
> aşağıdaki hesaplar), sonra **koşuldu**. Tasarım aşağıda OLDUĞU
> GİBİ duruyor; hüküm bölümün SONUNDA. **DİL sıçradı, BİLGİ
> sıfır kaldı, kol kapandı.**

### Tek cümle

**Çapa kaldırılır, yerine TOPLAMSAL bir okuma konur.** İkisi zaten
aynı makine — "en yakın anahtara bak" — ama biri durumu **değiştirip**
adresi yok ediyor, öteki **ekleyip** koruyor.

```
SIMDI   z = vur ? C[k] : zp            DEGISTIRIR   adres yok olur
YENI    z = normalize( zp + m )        EKLER        adres korunur
```

### Niçin — zincir, hepsi ölçülü

```
BILGI tam = 0,0000
 <- cevap hicbir blokta YOK, sansta                    §5.1/R
 <- olgu aramasi R_r'nin ICINDE OLAMAZ -- SAYILDI      §5.1/S
      Pi R_r rank<=d, serbestlik Stiefel(32,16) = 376
      kisit  308 ozne/iliski x (d-1) = 4.620      12,3 KAT KISA
      ustune R_r IZOMETRI: butun ikili acilari koruyor,
      olgu tablosu keyfi -- rijit harita yapamaz
 <- ve olcum bunu OLAYIN ICINDE gosteriyor             §5.1/S
      ozne okunabilirligi j=2'de TEPE, j=4'te -- tam
      R[iliski]'den sonra -- COKUS
 <- FIYAT mesele DEGIL: uye kaybinin %49'u zaten
    varlik konumlarinda ALINMAMIS duruyor              §5.1/O
```

### Adres — ölçüldü, kusursuz

`[Ö]` §5.1/T: 7.381 tek adımlı öneğin son durumu, yuva tavanı
(k-ortalama, her yuva çoğunluk cevabı):

```
                  capa     cakisik(cos>0,999)  M=2048 M=4096 M=7381
  t0_d8           ACIK          %0,1           0,282  0,556  0,999
  t0_d8           KAPALI        %0,0           0,282  0,556  1,000
  t0_d16          ACIK         %41,8           0,172  0,333  0,588
  t0_d16          KAPALI        %0,0           0,287  0,558  1,000
```

```
DONMELER ADRESI HIC BOZMUYOR    izometri, kayipsiz
TEK BOZAN CAPA                  d=16'da tavanin %41'i
`d` ADRES ICIN ONEMSIZ          capa kapaliyken d=8 = d=16 = 1,000
SIKISTIRMA YOK                  tavan M ile DOGRUSAL
```

`[Ö]` Adres `zp = R[w_{j-1}] z_{j-1}`. İlişki jetonunda bu **zaten
`(özne, ilişki)`** — etiket gerekmez, jetonun kendisi ikinci bileşeni
taşıyor. Doğrusal prob bunu göremiyordu (24 ilişkiyi tek çerçevede
havuzluyor); yuva ataması ortak çerçeve istemiyor, doğru ölçü bu.

### Ne eklenir

```
zp = R[w_{j-1}] z_{j-1}
kn = top_n( <zp, K> )                 K (M, D)   YENI, VQ ile EGITILMEZ
a  = softmax( <zp, K[kn]> / tau )
m  = a @ V[kn]                        V (M, D)   V(0) = 0
z  = normalize( zp + m )              CAPA YOK

L  = uye + a1*dis + a3*duzen + a4 * max(0, ort|m| - B)^2
                                      kod ve bag TERIMLERI KALKAR
```

`[H]` **`M = 8192`** — §5.1/T: sıkıştırma yok, olgu başına bir yuva.
7.381 olgu, yani `M >= 7.381`.

```
donme 220.224 + K 262.144 + V 262.144 = 744.512    (simdi 285.760, x2,61)
```

`[H]` **`BATCH = 2048`** — bellek aritmetiğinden, varsayılandan değil.
Skor matrisi `(B·L, M)`: önceki koşu `8192·24·2048·4 = 1,61 GB`;
`M = 8192`'de **aynı bütçe** `B = 2048` demek. Adım/epok 486 → 1.943.

`[H]` **`a4 > 0,795`** — ikamenin YENİ kayıptaki değeri:

```
olculen kisayol  0,6387
  eksi a2*kod    0,2469     bu terim kalkiyor
  eksi bag       0,0024     bu da
  = 0,3894
butce mentesesi  a4 * max(0, ort|m| - B)^2,   B = 0,30
  B, olculen varlik konumu payi 0,2643'ten turedi -- SECILMEDI  [O]
  ikame     ort|m| ~ 1,00  ->  maliyet a4 x 0,49
  amaclanan ort|m| ~ 0,26  ->  maliyet 0   (BEDAVA)
  engelle:  a4 > 0,3894 / 0,49 = 0,795.   UST SINIR YOK -- ayrisiyor.
ILK DEGER a4 = 2,0  (tabanin 2,5 kati).  Yukari hata "hafiza hic
kullanilmaz" demek; asagi hata IKAME demek ve onu bir kez gorduk.
```

`[Ç]` **`tau`** taşınmıyor: §12b'nin 0,02'si `K = C` içindi, K değişti.
İlk değer 0,02 ama bu bir **seçim**, ölçüm değil.

### §12b'nin ÜÇ arızasına karşı

```
1  ADRES YANLISTI.  K = C idi ve C, `kod` terimiyle k-ortalama gibi
   egitiliyordu; OLCULDU (§3.1b): ILISKIYI kodluyor (+1,680 bit),
   olguyu DEGIL (+0,318).  Hafiza iliski basina tek satirlik bir
   tabloya donmustu.  Ve kapasite hesabi 2048 yuva varsaymisti;
   kullanilan kod sayisi 166'ydi -- 12,3 kat iyimser.
   -> K SERBEST.  `kod`/`bag` terimleri KALKIYOR, K'yi hicbir sey
      niceleyiciye zorlamiyor.

2  HER ADIMDA OKUNUYORDU ve BEDAVAYDI.  Optimizasyon donmeleri
   sondurdu (duzen 3057 -> 45), capa %94'e cikti, model SONLU
   OTOMATA coktu; parametrede de gorunuyor: 1,2 derece, %100
   kendine donen (§5.1/Q).
   -> BUTCE MENTESESI.  Duz L1 SINANDI ve araligi BOS cikti
      (engelle a4 > 0,639 / birak a4 < 0,410, 1,56 kat).  Mentese
      ile maliyetler AYRISIYOR.

3  V = 0 BASLANGICI DENGEYI KAYDIRIYORDU.  Baslangic guvenliydi
   (kapi 36 dogruladi), denge degil.
   -> Butce dengeyi de sabitler: V buyumesi ucretli.
```

### Ne ÇÖZMEZ

```
[O] TAVAN, capa kalkinca 1,000 -- ADRESLEME tarafinda engel yok.
    Ama tavan "ogrenilebilir" demek DEGIL; 7.381 yuvanin dogru
    degerle dolmasi AYRI bir is ve bu tasarim onu GARANTI ETMIYOR.
[O] MODEL 2,6 KATINA CIKIYOR (285.760 -> 744.512).  Artik "kucuk
    bir ekleme" degil; her kiyasta oyle yazilacak.
[H] M kucultulebilir mi: M=4096'da tavan 0,558.  Ilk kosu M=8192
    ile; mekanizma calisirsa M asagi taranir.
[C] CIKARIM (2 adim) icin hicbir sey yapmiyor.  Tek adim calisirsa
    zincir AYRI bir soru olarak acilir.
[!] §3 GERI CEKILIYOR.  Capa kalkinca §3'un "durum bir koda oturur
    -> bilesim gorulmus parcalara iner" iddiasi mimaride KALMIYOR.
    Zaten olculmustu (§3.1b: kod iliskiyi kodluyor, bilesim inmiyor)
    ama bu bir IDDIA GERI CEKME ve oyle yazilmali.
    §3.2 (reddetme = kod uyeligi) dayanaksiz kalir -- hic
    uygulanmamisti (T3).  T3 KAPANIR: artik uygulanamaz.
    SAAT `t` capaya bagliydi (vur olunca sifirlanir); SAAT=False
    oldugu icin fiilen etkisiz, ama kod ve kapi 35 temizlenecek.
```

### ÖNCEDEN KAYIT — ilk koşu neye karar verir

```
DEGISEN   capa KALKAR; K serbest (M=8192), V, a4 butcesi;
          kod/bag terimleri kalkar; BATCH 8192 -> 2048.
          TEK PAKET -- "capa yerine toplamsal okuma".  Parcalari
          ayri ayri taranmaz.  d = 16 KALIR.

BIRINCIL  BILGI tam.  Adresleme tavani 1,000, yani bu kez tavan
          BAGLAMIYOR.  > 0,0000 ise tasarim DOGRU yon.
          = 0,0000 ise MIMARI KOL KAPANIR: adres kusursuz,
          kapasite yeterli, fiyat dogru, ve HALA yok.

IKINCIL -- HUKUM VERMEZ, tasarimi ACIKLAR
   ort |m|   0,30'un altinda mi (butce tuttu mu)
   duzen     45'e cokerse IKAME var demektir, butce YETMEDI
   |m| NEREDE buyuk: varlik konumlarinda mi?  Tasarimin iddiasi
             bu, ve olcen makine kurulu (§5.1/R).
   BICIM     §12b'de cokmustu; cokerse yine ikame var.
   kullanilan yuva sayisi

NE YAPILMAZ
   M, tau, a4 birlikte taranmaz.  Bir kosu, bir karar.
   `d` oynatilmaz -- §5.1/T'ye gore adres icin onemsiz zaten.
```

### SONUÇ — KOŞULDU.  DİL SIÇRADI, BİLGİ SIFIR, KOL KAPANDI
(21 Eylül, commit `f1b0ac9`, 725 sn eğitim + 23 sn ölçüm)

Önceden kayıt uygulandı: tek paket, `d = 16` kaldı, `M / tau / a4`
taranmadı. **Ölçüm bu kez geçerli** — §12b'yi geçersiz kılan üretim
hatası düzeltildi ve kapı 37 ile bağlandı (`soru kapısı 200/200`,
`yozlaşma 0,0064`, yani çıktı dejenere değil).

```
BICIM       kalip    ek       tip      kapanmadi  yozlasma  (ek n)
  OGRETILEN 0,7223   0,9249   0,2553   0,0099     0,0064     6.816
  CIKARIM   0,7180   0,9190   0,2447   0,0122     0,0058    11.590
BILGI       tam      aile     kisayol  bos
  OGRETILEN 0,0000   0,0000   0,0000   0,0002
  CIKARIM   0,0000   0,0001   0,0000   0,0000
```

**1) DİL — projenin gördüğü en iyi sonuç, ve TAKAS DEĞİL.**

```
              d=8      hafizasiz d=16   §12b      §12c
kalip        0,0840       0,2372       0,0000    0,7223
ek           0,2276       0,4344       0,0000    0,9249
tip          0,1542       0,0998       0,3962    0,2553
kapanmadi    0,6967       0,3413       1,0000    0,0099
```

Cümlelerin %99'u kapanıyor, ekler %92 doğru. Bozulan alan YOK:
BİLGİ zaten 0,0000'dı ve 0,0000 kaldı. Kural gereği takas diye
yazılmaz — **tek yönlü kazanç**.

**2) BÜTÇE TUTTU.** `ort |m|` beş epok boyunca 0,3068 → 0,3086;
tavan 0,30. §12b'de kaçan şey burada kilitlendi. `|V|max` 2,4 → 5,6:
model daha çok kullanmak istiyor, fiyat bırakmıyor.

**3) BİLGİ tam = 0,0000.**  Önceden kayda göre bu, mimari kolu
kapatıyordu — **ve ben öyle yazdım. Sonra §5.1/U o hükmü geri
aldırdı:** kayıt "kapasite yeterli" diyordu, ben bunu AYRILAN
kapasite (8.192 yuva) diye okudum; iz KULLANILAN kapasitenin **10
yuva** olduğunu gösterdi. Şart sağlanmamış, kol **açık**.
Aşağıdaki üç satır o yüzden artık "sağlandı" değil, "sağlandı
SANILDI" diye okunur:
Kayıtta aynen şöyle yazıyordu: *"= 0,0000 ise adres kusursuz,
kapasite yeterli, fiyat doğru, ve HÂLÂ yok."* Üçü de sağlandı:

```
adres      tavan 1,000 (§5.1/T, capa kapali)
kapasite   8.192 yuva >= 7.381 olgu
fiyat      uye'nin %49'u varlik konumlarinda ALINMAMIS (§5.1/O)
butce      tuttu (yukarida)
```

**4) TASARIMIN KENDİ İDDİASI ÇÜRÜDÜ — ve sebebi söylüyor.**
§12c "hafıza varlık konumlarında ateşleyecek" diyordu; ölçüldü:

```
ort|m| 0,3482    VARLIK 0,3094    DIGER 0,3799    oran 0,81
```

Varlık konumlarında geri kalandan **daha AZ** ateşliyor. Hafıza
çalıştı ama bütçesinin tamamını **dile** harcadı.

`[H]` Neden: dil konumları puanlananların %74'ü ve getirileri kolay;
varlık konumları %26 ve orada problem gerçekten zor. Bütçe TOPLAMI
sınırladı, **DAĞILIMI zorlamadı**. §5.1/O'nun "alınmamış ödül"
argümanının sınırı da buymuş: alınmamış ödül ancak **alınabiliyorsa**
bir şey ifade ediyor; optimizasyon kolay %74'e gitti.

**5) `duzen` 147,5 — kayıtta yazan iki değerin ARASINDA.**
"1128 = bütçe tuttu, 45 = ikame" demiştim; 147 ikisine de uymuyor ve
45'e daha yakın. Tek başına okunamaz; `yozlaşma 0,0064` ve BİÇİM'in
sıçraması §12b'deki otomat çöküşünün BURADA OLMADIĞINI söylüyor.
Kayıt bu ara bölgeyi öngörmemişti — **kayıt kusuru, sonuç değil.**

### NE KAPANDI, NE AÇIK KALDI

```
ACIK (ilkin KAPANDI yazildi, §5.1/U geri aldirdi)
         "olgu aramasini KAPASITE + ADRES + FIYAT ile cozeriz" kolu.
         Adres ve fiyat saglandi; KAPASITE saglanmadi -- 8.192 yuva
         ayrildi, 10'u kullanildi (§5.1/U).  Kol yeniden kosulabilir
         ama ONCE yuva kullanimini yayan bir sey gerekiyor, ve o
         sart kosu SIRASINDA olculur.

ACIK     Hafizanin DAGILIMINI zorlamak AYRI bir soru ve bu kosu onu
         sinamadi.  Ama sinamak icin varlik konumlarini ETIKETLEMEK
         gerekir (L2) -- ve o, deneyin kendi sorusunu zayiflatiyor
         (§5.1/N).  Bir sonraki karar bu takasla ilgili, kapasiteyle
         degil.

YAN      DIL artik cozulmus sayilir.  Bundan sonraki her kol, dili
         BOZMADIGINI gostermek zorunda -- yeni bir taban var:
         kalip 0,7223  ek 0,9249  kapanmadi 0,0099.
```

### TASARIM 3 — YUVALAR YAYILSIN  (21 Eylül, KOD YAZILMADI)

§12c'nin mekanizması duruyor ve dili çözdü; değişen tek şey, ölçülen
tek arıza: **8.192 yuvanın 10'u kullanılıyor** (§5.1/U).

#### Aday üç yol, kâğıtta ayrıldı

```
S3  TAU'yu yumusat        REDDEDILDI -- ama YALNIZ softmax icin.
    !! Bu red `softmax`in disbukeyligine dayaniyordu; TASARIM 4
       (ReLU) o zemini kaldiriyor ve red ORADA GECERSIZ.
    Okuma m = sum a_i V_i.  a yumusarsa m, k rastgele yonun
    ortalamasi olur ve normu ~1/sqrt(k) kuculur:
        etkin yuva  1 -> netlik %100     4 -> %50     8 -> %35
    Ustelik |m| zaten BUTCEYLE 0,30'a bagli, telafi edemiyor.
    Gradyani yaymak icin CEVABI bozmak gerekiyor.

S2  Olu yuvayi yeniden tohumla  (VQ-VAE'nin standart hilesi)
    Agirlik gerektirmiyor, dogrudan olcuLen arizayi hedefliyor.
    AMA K'yi VERIYE cekiyor -- olu olanlari da olsa. Ve §3.1b
    olctu ki veriye cekilen K, olguyu degil ILISKIYI kodluyor.
    Riski ACIK, ikinci sirada.

S1  YUK DENGELEME  (mixture-of-experts'in standart terimi)  <- SECILEN
        L_denge = M * sum_i f_i * P_i
        f_i  yuva i'ye giden konum PAYI
        P_i  yuva i'nin ortalama softmax OLASILIGI
    Tekduze kullanimda 1, tek yuvada M.
    !! K'yi VERIYE CEKMIYOR.  "hepsini kullan" diyor, "verinin
       ustune otur" demiyor -- §3.1b'nin arizasini TASIMIYOR.
       S2'den ayiran sey bu.
```

#### `a5` — ve ilk hesabımın hatası

```
L_denge   COKMUS halde (10 yuva)  819,2
          hedef (olgu basina bir yuva)  1,110
          BASLANGICTA (olculdu, 21 Eylul, adim 1)  1,54
```

`[Ö]` **Bu terim sabit bir itiş değil, GERİ ÇAĞIRICI KUVVET** — ve
bunu ilk yazışımda gözden kaçırmıştım. Başlangıçta yuvaların
8.078/8.192'si kullanılıyor ve `denge` 1,54; ceza
`3e-5 × 1,54 = 4,6e-5`, yani fiilen sıfır. Çöküş başlarsa 10 yuvada
`3e-5 × 819 = 0,0246`'ya çıkar. Yani terim her şey yolundayken
bedava, bozulmaya başlayınca devreye giriyor.

`[Ö]` Bunun bir sonucu da şu: **çöküş başlangıçta yok, eğitim
sırasında oluşuyor.** §5.1/U'nun "10 yuva"sı bir başlangıç durumu
değil, modelin adım adım vardığı bir yer. `a5`'in işi bir çöküşü
ONARMAK değil, ENGELLEMEK.

İlk yazdığım `a5 = 3e-4`, **yaymaya 0,2454 ödüyordu** — olgunun
değdiği 0,3008'in (§5.1/M) neredeyse tamamı. O ağırlıkla model
olguyu öğrenmeden yuvaları rastgele dağıtıp parayı alırdı; §12b'nin
ikame tuzağının aynısı, yeni bir kapıdan.

```
yayma odulu olgunun  %30'u olsun  ->  a5 <= 1,10e-04
                     %10'u                3,68e-05
                      %5'i                1,84e-05
SECIM   a5 = 3e-5   ->  odul 0,0245  = olgunun %8'i
```

`[Ö]` **Asıl gerekçe büyüklük değil, ve bu ÖLÇÜLDÜ.** Ölü yuvanın
gradyanı **tam sıfır** — softmax'ın top-8'ine hiç girmiyor. İki
koşunun 1. adımındaki eğitim logu, yan yana:

```
§12c  (a5 = 0)      C  |g|max 0.000e+00      <- TAM SIFIR
tasarim 3 (a5)      C  |g|max 1.210e-05
```

`V` sıfırdan başladığı için okuma kayba hiçbir şey katmıyor,
dolayısıyla softmax üzerinden anahtarlara geri **hiç** gradyan
akmıyordu. Terimin işi bir ihaleyi kazanmak değil, **o sıfırı
kırmak** — ve birinci adımda birebir o oldu. Küçük olması yeter; büyük olması
tehlikeli. Bu yüzden üst sınır hesaplandı, alt sınır hesaplanmadı —
ve bu bir seçim, ölçüm değil.

#### Bütçe deliği — izde görülen `|m| = 4,79`

```
kayip:  mn = y["mn"][:, isin:].mean()
```

Konum 0..3 bütçeye **hiç girmiyor**. ISINMA onları `üye`den dışlıyor
(önek çöp, §5.2) ama hafızanın oraya **yazmasını** engellemiyor —
oraya yazmak bedava. Ve o konumlardan biri her pencerede
`z = [p ; 0]`, yani gizli yarısı tam sıfır: çok ayırt edici, hep aynı
türden bir durum. İzde her örnekte `|m|` j=1'de **4,79**, sonra 0,4.

```
DUZELTME   butce TUM konumlara bakar, dilim YOK.
           Pencerenin %17'si (4/24) butce disindaydi.
BEKLENEN   olculen 0,3086 ortalamasi YUKSELIR (disarida kalan buyuk
           degerler iceri girince), yani model ayni butceyle daha AZ
           yazabilir hale gelir.
```

`[Ç]` Bu bir **düzeltme**, yeni bir değişken değil: mevcut
mekanizmanın kapsaması gereken yeri kapsaması. Yine de koşuya iki
şey birden giriyor ve bu yazılır.

#### ÖNCEDEN KAYIT

```
DEGISEN   a5 * L_denge eklenir (a5 = 3e-5);  butce dilimi kalkar.
          Baska HICBIR SEY degismez -- M, tau, a4, d, capa aynen.

ONKOSUL -- HUKUMDEN ONCE, ve KOSU SIRASINDA izlenir
   ATESLEYEN AYRI YUVA sayisi.  taban 10 (§5.1/U).
   >> 10 olmazsa mekanizma calismamistir ve BILGI okunmaz;
   kosu yayma sorununu cozmemis demektir, olgu sorusuna
   cevap VERMEZ.

BIRINCIL  BILGI tam  (onkosul saglanirsa)
   > 0,0000  -> yayma yetiyormus, kol acik kalir
   = 0,0000  -> adres kusursuz + kapasite GERCEKTEN kullanilmis +
                fiyat dogru + butce tuttu, ve hala yok.
                ISTE O ZAMAN kol kapanir.  §12c'de erken
                kapatmistim (§5.1/U); bu kez sart olculecek.

IKINCIL -- HUKUM VERMEZ
   `denge` EGRISI.  Baslangic 1,54 (olculdu).  Tirmanirsa cokus
   basliyor demektir ve a5 onu TUTAMIYOR; duz kalirsa tutuyor.
   Bu, `YUVA` sayisinin SUREKLI halidir -- ikisi birlikte okunur.
   DIL BOZULMAMALI.  Yeni taban: kalip 0,7223  ek 0,9249
   kapanmadi 0,0099.  Bozulursa TAKAS diye yazilir.
   ort |m| (butce yeni tanimla), duzen, |m| varlik/diger orani.

NE YAPILMAZ
   a5 taranmaz.  M, tau, a4 oynatilmaz.  Bir kosu, bir karar.
```

### TASARIM 4 — `softmax` YERİNE `ReLU`  (21 Eylül, KOD YAZILMADI)

Kullanıcı: *"transformer mimarisi bilgiyi hafızada nasıl tutuyor ya da
çağırıyor, belki oradan kopya çekebiliriz."* Çekilebiliyor, ve çekilen
şey Tasarım 3'ün `a5`'ini **gereksiz kılıyor**.

#### Nereden kopya

`[Ç]` Transformer'ın ileri-besleme katmanı `W₂·σ(W₁x)` zaten bir
anahtar-değer hafızası: `W₁`'in satırları anahtar, `W₂`'nin sütunları
değer, `σ(W₁x)` katsayı. Bizim `a = σ(⟨z,K⟩)`, `m = a@V` yapımızın
birebir aynısı — **tek bir yer hariç**:

```
TRANSFORMER   sigma = ReLU/GeLU, ELEMAN BAZINDA, yarisma YOK
BIZ           softmax(top-8), tau=0,02 -- neredeyse tek-sicak
```

#### 1) Ölü yuva dinamiği — yarışmanın kendisi sebep

```
softmax(top-8), tau=0,02
   bir yuva gradyan almak icin 8.184 rakibi YENMELI (top %0,098)
   kazanan anahtar z'ye yaklasir -> ayni sorgulari daha guvenli kazanir
   KENDINI BESLEYEN DONGU
ReLU(<z,k> + b)
   yalnizca ESIGI asmak yeter; i'nin ateslemesi j'yi BASTIRMIYOR
   besleme dongusu YOK
```

`[Ö]` Ölçülen imza bu okumaya uyuyor: Tasarım 3 koşusunda yuva
`8078 → 796` (bir epok), `denge 1,54 → 46,68` (30 kat). Üstel
görünümlü.

`[Ç]` **ReLU'da da ölü yuva mümkün** (`b` çok negatif olursa) ama
ölme sebebi **göreli değil mutlak**: yalnızca ateşlemenin zararlı
olduğu yerde. Kendini besleyen döngü yok.

`[Ç]` **SINIR:** ReLU'nun pratikte çökmeyi önleyeceğini kâğıtta
KANITLAYAMAM. Gösterebildiğim şey döngünün **yapısal olarak**
kalktığı. Sonuç ölçülecek.

#### 2) "Olgu başına bir yuva" şartı DÜŞÜYOR

```
softmax ~top-1  cikti M vektorden BIRI        -> M >= 7.381
                (§5.1/T'nin tavani TAM BU varsayimdan cikmisti)
ReLU            cikti bir ALT KUMENIN toplami -> ~C(M,k) ayri cikti
```

Bağlayıcı kısıt yeniden **parametre sayımı** olur:

```
kisit 7.381 x (d-1) = 110.715
  M=2048  K+V 131.072  1,18 kat    model 351.296
  M=4096      262.144  2,37 kat    model 482.368
  M=8192      524.288  4,74 kat    model 744.512
```

`M ≥ 1730` yeterli. **`M = 8192` artık gerekmiyor** — ama ilk koşuda
değiştirmiyorum: bir koşu bir karar, ve fazla kapasite güvenli taraf.
B1 tutarsa `M` aşağı taranır ve model küçülür.

#### 3) `S3` reddim düşüyor — seyrelme `softmax`'a özgüymüş

`§12c`'de "tau'yu yumuşat"ı şöyle reddetmiştim: dışbükey bileşimde
`k` yön karışınca norm `1/√k` ile küçülür (k=8'de netlik %35).
**ReLU'da toplam negatif olmayan ve sınırsız** — değerler sönmez,
toplanır. Çoklu aktivasyon artık bir bedel değil, **biçim**.

#### 4) Bedel — ve KAPI 26 ile çakışma

```
`sa` = (B, M) skor matrisi, adim basina 0,067 GB
softmax top-n   `sa` NO_GRAD, yalniz (B,8) gradyanli
ReLU            `sa` GRADYANLI, 23 adim TUTULUR
                tutulan 1,54 GB, tepe ~3,1 GB   (L4'te 23,7 GB var)
```

`[H]` Hesap maliyeti **düşüyor**: `topk` ve iki `gather` gidiyor.

`[!]` **Kapı 26'nın şartı ihlal ediliyor** (`gradyanli (B,K) tensor
sayisi = 0`). Kapı **silinmez**: amacı bunun KAZA ile olmasını
engellemekti, şimdi KASITLI. Şartı "≤ L−1 tane, ve tepe bellek
aritmetiği yanında yazılı" olarak değişir.

#### 5) Sapma `b` başlangıcı — hesapla, varsayılanla değil

```
<z,k>, birim vektorler, D=32  ->  std ~ 1/sqrt(D) = 0,1768
b = 0      -> konum basina ~%50 atesler = 4.096 yuva
b = -0,149 -> %20   (~1.638 yuva)
b = -0,291 -> %5    (~409 yuva)
b = -0,411 -> %1    (~81 yuva)
```

`[Ç]` **`b₀ = -0,29`** (~%5). Denge seyrekliği kâğıtta öngörülemez,
ölçülecek. `b` öğrenilebilir olmalı — sabitlenirse seyreklik bir
varsayım olarak kalır.

#### 6) Bütçe (`a4`) değişmiyor, `a5` kalkıyor

`[H]` `a4`'ün tabanı **ikame kazancından** geliyordu (0,3894) ve
`σ`'dan bağımsız: `a4 > 0,795` aynen geçerli. `B = 0,30` de geçerli —
bütçe `|m|` üzerinde, yani hafızanın durumu **ne kadar oynattığı**
üzerinde; kaç yuvanın katkıda bulunduğundan bağımsız.

`[H]` Ve ReLU'da `|m|` **sınırsız** (softmax'ta `≤ max|V_i|` idi),
yani bütçe artık **daha gerekli**.

`[Ç]` **`a5 = 0`.** ReLU'da "`f_i` = top-1 payı" anlamsız. Yerine
izlenecek: **ateşleyen yuva oranı** ve **konum başına ortalama aktif
yuva**. İkisi de ucuz ve ikisi de `YUVA`'nın doğru genellemesi.

#### ÖNCEDEN KAYIT

```
DEGISEN   softmax(top-n) -> ReLU(<z,K> + b),  b ogrenilir, b0 = -0,29
          a5 = 0.  Baska HICBIR SEY: M, a4, butce, d, capa AYNEN.

ONKOSUL   ATESLEYEN yuva orani, KOSU SIRASINDA.
          Tasarim 3'te bu oran bir epokta 8078 -> 796 dusuyordu.
          ReLU'da DUSMEMELI (ya da cok daha yavas dusmeli).
          Dusuyorsa B1 yapisal olarak dogru ama pratikte YETMIYOR
          demektir ve S2 (olu yuva tohumlama) siraya gecer.

BIRINCIL  BILGI tam, ek-toleransli olcu (kapi 40).
          Taban 0,0138 (§5.1/V), sansin 21,8 kati.
          ONCEDEN YAZILAN BEKLENTI (§5.1/U'nun okumasinin sinavi):
          10 yuva 102 dogru verdi = yuva basina ~10 soru.  Kullanilan
          yuva n katina cikarsa BILGI de o mertebede artmali.
          Artmazsa yuva sayisi BAGLAYICI KISIT DEGILMIS demektir ve
          §5.1/U'nun butun okumasi yeniden yazilir.

IKINCIL   DIL BOZULMAMALI: kalip 0,7223  ek 0,9249  kapanmadi 0,0099.
          ort |m| (butce tuttu mu), duzen, konum basina aktif yuva.
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
T3  §3.2 reddetme kodda YOK                    K    A9
    -> §12c KOSULDU: capa kalkti (r=0).  Kod
       uyeligi diye bir sey kalmadi, reddetme BU
       YOLLA artik UYGULANAMAZ.  T3 KAPANDI;
       reddetme AYRI bir mekanizma ister.

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
    -> §12c: capa kaldirildi, toplamsal hafiza
       konuldu, KOSULDU.  DIL SICRADI (kapanmadi
       0,3413 -> 0,0099) ama BILGI 0,0000 KALDI.
       Adres kusursuz (tavan 1,000), kapasite
       yeterli (8.192 yuva), fiyat dogru, butce
       tuttu -- ve hala yok.  KAPASITE+ADRES+FIYAT
       KOLU KAPANDI (onceden kayit geregi).
       Sebep olculdu: hafiza butcesini DILE
       harcadi (varlik 0,3094 / diger 0,3799).
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
