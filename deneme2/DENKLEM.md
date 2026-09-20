# Yol modeli — genel bir dil mimarisi

Kullanıcı, 20 Eylül 2026:

> *"bu model gerçek dil modeli öğrenmesinin bir kopyası gibi
> tasarlasak ama veri bizim verimiz gibi düşün. çünkü biz kendi
> verimize uygun bir mimari tasarlamıyoruz. genel bir dil mimarisi o
> ama bizim verimizde çalışıp çalışmadığını test edebiliriz."*

Bu belge **genel** bir mimari tarif eder. Grafımızdan, varlık
listemizden, ilişki şemamızdan hiçbir şey almaz. Korpusumuz mimarinin
kaynağı değil, **sınavı**.

**Hiçbir şey koşulmadı.** `CLAUDE.md` kural 0.

---

## 1. Mimari

```
z_j  =  capa( R[w_{j-1}] @ z_{j-1} )          durum, kurede
w_j  =  en yakin birim( PI z_j / ||PI z_j|| ) OKUMA -- ACISAL
```

```
SABIT (ogrenilmez)
  p_w in R^d, |p|=1     n birim, KUCUK okuma uzayinda
  PI : R^D -> R^d       ilk d koordinata izdusum

OGRENILEN
  R_w in SO(D)          her birimin donmesi        (bkz. 4.3)
  C   in R^{K x D}      KOD DEFTERI -- capa hedefleri
  S   in SO(D)          saat.  OPSIYONEL, bkz. 6

ESIK (kayipta YOK, tutulanda aranir)
  r                     capa yaricapi
  delta                 itme esigi
```

Hesap yığını yok: ne dikkat, ne katman, ne çıktı matrisi. İleri
geçişte yalnız dönme çarpımı ve en yakın komşu.

---

## 2. Üç iddia

```
1  IZOMETRIK GECIS   Donme normu ve mesafeyi korur.
                     -> temsil cokemez, sinyal sonmez, gradyan sonmez.
                     OLCULDU (model_13): oteleme tabanlisinda ozne
                     duyarliligi 0,22'ye soniyordu. Donmede
                     ||Ra-Rb|| = ||a-b||; sonme IMKANSIZ.

2  NICELEME          Durum periyodik olarak bir KODA oturur.
                     -> durumlar YENIDEN KULLANILABILIR olur.
                     Gorulmemis bilesim, gorulmus parcalara iner.
                     ASIL IDDIA BU. Olcusu `comp`.

3  MESAFEYLE OKUMA   Cikti katmani yok; sozluk buyudukce okuma
                     maliyeti artar ama PARAMETRE artmaz.
```

---

## 3. Çapa = niceleme (vektör kuantalama)

```
k = argmin_k || z - C_k ||
|| z - C_k || < r   ->   z <- C_k          CAPA
```

**Varlık listesi yok.** `C` öğrenilir; model hangi durumların yeniden
kullanılmaya değer olduğunu kendi bulur.

### 3.1 Neden `comp`i veriyor — TEOREM

`j`'de `C_k`'ya çapalandıysa `z_j = C_k`, geçmişten **bağımsız**. O
hâlde herhangi bir sonek `u = (u_1..u_m)` için

```
z_{j+m} = R_{u_m} ... R_{u_1} C_k
```

yalnız `(k, u)`'ya bağlı — oraya nasıl gelindiğine **değil**.

> `(k, u)` eğitimde görüldüyse, model onu `k`'ya çapalanan **her**
> bağlamda birebir tekrarlar.

Sonuç: tek-adım doğruluğu `a` ise **`comp ≈ a²`**. Düşebilir bir
öngörü; `comp ≥ 0,50` kapısı için `a ≥ 0,71` gerekir.

### 3.2 Kod defteri üyeliği = DÜRÜSTLÜK

`Zeynep` gerçek bir ad, `Kayabaşı` gerçek bir soyad, ama
`Zeynep Kayabaşı` diye bir varlık yok — kod defterinde de kodu yok.

```
durum bir kodun r yaricapinda   ->  capa,      "var"
hicbir kodun yakininda degil    ->  capa yok,  "yok"
```

Reddetme ayrı bir mekanizma değil, **kod defteri üyeliğinin
kendisi**. Ve genel: "varlık" kavramı gerektirmiyor.

Talep eden tarafı: paylaşılan dönmelerle gerçek birleşimler kodun
içine, sahte birleşimler dışına düşmeli. Bu **bileşimsel üyelik** ve
zor. Sınavın `durustluk` kalemi bunu ölçer.

---

## 4. `D > d`, açısal okuma, ve operatörün zenginliği

### 4.1 Okuma neden bir İZDÜŞÜM

`D = d` olsaydı ortak sonek `A` bir izometri olur, bütün ikili
mesafeleri korurdu:

```
||q_H - q_A|| = ||p_Fatma  - p_Zehra||        (anne soneki)
||q_H - q_A|| = ||p_Gorgul - p_Betimsel||     (tezi soneki)
```

Sol taraf ilişkiden bağımsız → sağ taraflar eşit olmak zorunda
kalırdı. Sabit rastgele noktalarda olmaz.

*Sayi (cember, 475 birim):* gereken hata `< 0,38°`, en iyi uzlaşmada
çıkan `~100°`. **250 kat.**

`Π` izometri değildir; `‖Π A (q_x − q_y)‖` artık `A`'nın farkı hangi
yöne çevirdiğine bağlıdır. Çelişki kalkar.

### 4.2 Okuma AÇISAL olmali

`‖Π z − hedef‖²` yazılamaz: gizli kütle varken `‖Π z‖ < 1` olur ve
terim gizli kütleyi **sıfıra iter** — oysa 4.1 onu zorunlu kılıyor.
İki terim birbiriyle kavga eder.

```
q = PI z / ||PI z||
L_uye = || q - p_w ||^2 = 2 - 2 cos(aci farki)
```

İş bölümü netleşiyor:

```
L_uye, L_dis   OKUMA YONU    -- hangi kelime
L_capa         TAM DURUM     -- hafiza, yalniz capa noktalarinda
```

### 4.3 Tek düzlem YETMİYOR — ölçeklenme buradan geçmiyor

Birim başına tam `SO(D)` gerçek ölçekte imkânsız:

```
V=50k, D=256   ->   50.000 x 32.640  =  1,63 MILYAR
```

Tek düzlemli (Givens) dönme denendi ve **elendi**: düzleminin dışında
özdeşliktir.

```
E[ ||delta_duzlem||^2 / ||delta||^2 ]  =  2/D
D= 32   ->  %6,3     farkin %93,7'si DOKUNULMAZ
D=256   ->  %0,8
```

4.1'deki kaçış çiftlerin ancak %6'sında çalışır.

*Ornek (D=4):* `anne` düzlemi `(e1,e3)` iken `δ=(0,1,0,1)` çiftinin
hiçbir bileşeni o düzlemde değil; açı ne olursa olsun izdüşüm ayrımı
1'de **sabit** kalıyor, ilişki o çiftin geometrisini
şekillendiremiyor.

**Çözüm dilsel, korpusa özgü değil:**

```
GEOMETRIYI SEKILLENDIREN   kapali sinif = YUKSEK FREKANSLI birimler
                           ek, noktalama, kalip, iliski sozcugu
                           ->  TAM SO(D),  D(D-1)/2 parametre

DURUMU BIR YERE KOYAN      acik sinif = kalan
                           adlar, icerik sozcukleri
                           ->  TEK DUZLEM,  2D+1 parametre
```

Bölme **frekanstan** yapılır; etiket, sözlük ya da oracle gerekmez.
Bir içerik kelimesinin işi durumu ayırt edilebilir bir yere taşımak;
bunun için tek düzlem yeter.

---

## 5. Kayıp — YOLUN TAMAMINA

Kullanıcı: *"cümlenin tamamını değerlendirmemiz gerekiyor next token
değil."*

```
L = L_uye + a1*L_dis + a2*L_capa + a3*L_duzen
```

```
L_uye   = SUM_j  || q_j - p_{w_j} ||^2               ACISAL (4.2)

L_dis   = SUM_{w pencerede DEGIL}  max(0, delta - d(p_w, YOL))^2
          d(p_w, YOL) = min_j || q_j - p_w ||

L_capa  = || sg[z] - C_k ||^2  +  beta || z - sg[C_k] ||^2
          YALNIZ capa tetiklendigi adimlarda.  Standart VQ.

L_duzen = donme uretecinin karesi.
          a3 = EZBER <-> GENELLEME dugmesi
```

`L_dis` **itici kuvvet**. *Gerekçe ÖLÇÜLDÜ (model_13):* saf çekme
kaybı çöktü — bit 8,32, unigram tabanı 6,58'in üstünde. Çekmek yetmez.

`L_capa` **örtük `comp`in ön koşulu**: onsuz durum kodun yanından
geçer ama çapa tetiklenmez.

Sıra kısıtı YOK — sıra yolun kendi sırası.

### 5.1 Bilinen tuzaklar

```
L_dis HAKSIZ CEZA   pencerede olmayan birimlerin bir kismi GECERLI
                    alternatif (korpusta 17 bildirim kalibi var).
                    Mentese softmax'tan sert.  Hafifletici: hepsi
                    delta'yi gecince terim sifirlanir.
OLU KOD             VQ'nun klasik arizasi: hic ziyaret edilmeyen kod.
                    Standart care yeniden baslatma.  A6.
```

---

## 6. Saat `S` — ÖLÇÜLECEK DÜĞME, varsayılan KAPALI

Saat, tekrar eden birim için konmuştu. Çapa geldikten sonra tekrar
bakıldı ve gereksiz görünüyor:

```
Ayse* Yilmaz ... Fatma* Yilmaz    iki Yilmaz FARKLI capalardan gelir
                                  -> ayrimi DURUM yapiyor
Hasan* -TAM karde -IYE -TAM       arada capa yok, ama D>d boslugu var
                                  -> yine DURUM yetiyor
```

Ve maliyeti var:

```
BILDIRIM   cevap capadan 3 adim sonra
SORU       cevap capadan 6 adim sonra
           -> ayni olgu IKI AYRI geometrik kisit
           -> ~3 soru bicimi  =>  kisitlar 3 KAT
```

```
saat VAR   tekrar ayrimi GARANTI     kisitlar 3x
saat YOK   ayrim OGRENILIR           kisitlar 1x
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
İlk `zincir()` tasarımı köprüyü çıktı olarak üretiyordu — yanlış
soruyu cevaplıyordu, kaldırıldı.

---

## 8. Parametre — ve dürüst kalibrasyon

```
                        bizim (n=475, D=32, d=8, K=2048)
kapali sinif   80 x 496            39.680
acik sinif    395 x  65            25.675
kod defteri  2048 x  32            65.536
-------------------------------------------
                                  130.891     (saat kapali)
```

**Kıyas yanıltıcıydı, düzeltiyorum:**

```
model_11   6,5 M     %98'i 8 BLOK (hesap yigini), %0,3'u gomme
model_14   131 bin   %100 depolama, hesap yigini YOK
```

İkisi aynı eksende kıyaslanamaz. Ve "depolama tek başına yeter mi"
sorusunun bir ölçümü zaten var:

```
model_13   77.253 parametre, cogu depolama   ->   one 0,0204   COKTU
```

131 bin, çöktüğünü bildiğimiz noktanın 1,7 katı. Bu sayı
**savunulmuyor**; kâğıdın verdiği alt sınır. Modelin iddiası "daha az
parametreyle aynı iş" değil — iddia **hesap yığınının yerine
nicelemeyi koymak**. Niceleme çalışmazsa parametreyi büyütmek de
kurtarmaz.

---

## 9. Eğitim

```
9.1 PARAMETRELESTIRME
    R dogrudan optimize edilemez (Adam ortogonalligi bozar).
    kapali sinif  R = exp(ters simetrik),  torch.matrix_exp
    acik sinif    tek duzlem, KAPALI FORM:
      R = I + sin0 (v u^T - u v^T) + (cos0 - 1)(u u^T + v v^T)
      matrix_exp gerekmez; uygulama O(D).

9.2 GRADYAN SONMEZ
    Capalar arasi dz/dz bir donmedir, tekil degerleri 1.

9.3 OGRETMEN ZORLAMASI
    w_{j-1} gercek birim; z_j onekin kapali formlu bileskesi,
    ozyineleme yok. Kayip yine YOLUN TAMAMINA.

9.4 BASLANGIC
    Tipik donme acisi BIR KOMSU ARALIGI olsun:
      aralik ~ n^(-1/(d-1)),   sigma = aralik / sqrt(D).   HESAP.

9.5 VERI
    Akistan PENCERE. Semantik bolme YOK -- pencere keyfi yerden
    baslar, "cop" durum ILK CAPADA silinir.

9.6 IKI ADIMLI EGITIM YOK
    comp yalniz cikarimda olusur; boylece 3.1'in ongorusu temiz kalir
    ve gercekten dusebilir.

9.7 r, delta EGITIMDE YOK
    Tutulan bolmede aranir; birer olcumdur.

9.8 MALIYET  -- OLCULDU 20 Eylul, onceki tahmin 26 KAT DUSUKTU

    Ileri gecis carpma sayisi (B=8192, L=16, n=475, d=8, D=32, K=2048):
      donme() kur         54 M    % 0,6
      donme uygula       126 M    % 1,4
      KOD ARAMA        8.053 M    %92,2      <- baskin kalem
      okuma              498 M    % 5,7
      TOPLAM            8,73 G  -> epok (342 adim x3)  9,0 TFLOP

    Onceki tahmin 0,35 TFLOP idi; KOD ARAMASI hesaba katilmamisti.
    L4'te ~1 sn/epok saf hesap, pratikte saniyeler. Tarama HALA
    karsilanabilir, ama K baskin dugme (olculdu, B=1024):
      K= 512  26 ms/adim      K=2048  40 ms
      K=1024  31 ms           K=4096  75 ms

    BELLEK: gather edilmis donme matrisleri geri gecis icin tutulur,
    B=8192'de 503 MB. Hesabin yalniz %1,4'u ama bellegin buyuk kismi.
    Kucultmek gerekirse ilk kaldirac B, ikincisi acik sinif icin
    Rodrigues'i MATRIS KURMADAN uygulamak (%83 birim, 503 -> ~85 MB).
    Simdilik gerekmiyor.

    NEUTRAL CIKAN: cdist -> ic carpim degisikligi. Mikro-olcum 1,5x
    vaat ediyordu, ileri+geri olcumu 1,01x verdi (fark yok). Degisiklik
    TUTULDU ama HIZ ICIN DEGIL: ||z-C||^2 = 2-2 z.C tam esitlik,
    uzakligi bedava veriyor ve bir bagimlilik eksiliyor.
```

---

## 10. Açık sorular

```
A1  D, d, K kac?   (32, 8, 2048) kagit onerisi. Tarama ucuz.
A2  r kac?         Kodlar arasi mesafenin yarisindan kucuk olmali.
A3  delta, a1, a2, a3 kac?   Olculmeden secilmez.
A4  Saat acilacak mi?        (6) -- 3x kisit maliyeti.
A5  Kapali/acik sinir NEREDE?  Frekans esigi bir dugme.
A6  OLU KOD: K kodun kaci kullaniliyor? Yeniden baslatma gerekli mi?
A7  Isin genisligi.
```

---

## 11. Bizim verimiz NEYİ SINIYOR

Mimari genel; korpusumuz sınav. Dört düşebilir soru:

```
S1  KOD DEFTERI VARLIKLARI KESFEDIYOR MU?
    Grafi biz yazdik, cevabi BILIYORUZ. Kodlar varliklara oturuyorsa
    niceleme gercek; oturmuyorsa mekanizma yanlis ve bunu UCUZA
    ogreniriz.                                  <- HUKUM VEREN SORU

S2  comp ~ a^2 TUTUYOR MU?
    3.1'in ongorusu. Tutmuyorsa teorem dogru ama onkosulu
    saglanmiyor demektir (muhtemelen L_capa yetersiz).

S3  OLMAYAN BIRLESIM REDDEDILIYOR MU?
    3.2. Sahte ad birlesimleri kodun disina dusuyor mu.

S4  KAPALI/ACIK BOLMESI FREKANSTAN CIKIYOR MU?
    4.3. En sik 80 birim gercekten ek/kalip/iliski mi.
```

`S1` tek başına hüküm verir: niceleme çalışmazsa mimarinin ana
iddiası düşer ve parametre sorusu anlamsızlaşır.

---

## 12. Neye benziyor, nesi yeni

```
donme ile kompozisyon      RotatE, donme tabanli graf gommeleri
vektor kuantalama          VQ-VAE ailesi
en yakin nokta ile okuma   vektor niceleme
isin aramasi               standart
kapali/acik sinif ayrimi   dilbilim
```

Yeni olan birleşim: **dizi durumunun izometrik olması ve periyodik
olarak öğrenilmiş bir koda oturması**, kaybın da bir sonraki kelimeye
değil yolun tamamına bakması. Ölçülmüş bir karşılığını bilmiyorum;
literatürde aranmalı — "benzerini gördüm" diye yazmıyorum.
