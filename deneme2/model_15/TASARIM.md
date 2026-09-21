# model_15 — GÜZERGÂH

Kullanıcı, 21 Eylül 2026:

> *"baştan model 15 tasarladığımızı düşün ve sadece 3 boyut olsun hatta
> 2 boyut bile yeterli. istanbul (1,1) ankara (05,05) gibi değerler.
> sonra da istanbul --> ankara --> mersin güzergahları olunca bir
> sonraki uğranacak şehir neresi olduğunu öğrenmek. izmir bursa
> üzerinden mersine geldiğimizde bir sonraki şehrin neresi olduğunu
> hesaplamak."*

> *"model 14 ön yargılarını taşıma lütfen."*

**Bu dosya adım adım yazılıyor.** Her adım bir şeye karar verir, o kararı
örnek üstünde gösterir, ve bir sonraki adımın sorusunu açar. Tek seferde
yazılmıyor.

---

## ADIM 1 — ŞEHİRLER BELLİ Mİ

**Evet. Belli, sabit, ve eğitilmiyor.** Kod: `sehir_15.py`.

Bu bir karar, varsayılan değil. Şehirlerin nerede durduğunu model
öğrenmiyor — biz veriyoruz, ve bir daha dokunulmuyor. Model'in
öğreneceği tek şey **yolun kendisi**.

```python
SEHIR = {
    "Istanbul": (1.0, 1.0),      # kullanici
    "Ankara":   (0.5, 0.5),      # kullanici
    "Izmir":   (-1.0, 0.3),
    "Bursa":    (0.2, 1.1),
    "Mersin":   (0.4, -0.9),
    "Sivas":    (1.4, 0.1),
    "Adana":    (0.9, -0.8),
    "Konya":    (0.6, -0.3),
}
KONUM = torch.tensor([...])      # (8, 2)  buffer, Parameter DEGIL
```

Verdiğin iki değer aynen duruyor. Kalan altısı örneği tamamlamak için.

### Boyut 2

*"iki ya da 3 boyut farketmez"* — 2 seçildi. 3'e çıkarmanın şu an bir
gerekçesi yok; gerekçe doğarsa `BOYUT` tek yerden değişir.

> **Ama bir uyarı şimdiden yazılı:** 2 boyutta model_14'ün çevirmesi
> (dönme) **sırayı göremez** — 2B dönmeler değişmelidir,
> `R(a)R(b) = R(b)R(a)`, yani durum uğranan şehirlerin **toplamı**
> olur, sırası değil. Sıra 3 boyutta görünür olmaya başlar. Bu,
> ADIM 2'deki çevirme kararını doğrudan bağlıyor.

### Yollar

```
Istanbul -> Ankara -> Mersin -> Sivas        [0, 1, 4, 5]
Izmir    -> Bursa  -> Mersin -> Adana        [2, 3, 4, 6]
Konya    -> Ankara -> Mersin -> Sivas        [7, 1, 4, 5]
```

Üçü birlikte işin tamamını tarif ediyor:

```
1 ve 2   son sehir AYNI (Mersin), devam FARKLI   ->  AYIRMA gerekiyor
1 ve 3   son iki sehir AYNI, devam AYNI          ->  BIRLESTIRME gerekiyor
```

**İkisi birden istenecek.** Birini yapıp ötekini yapamamak model_14'ün
düştüğü yer (§5.1/X).

### Kodun kendi denetimi

`python sehir_15.py`:

```
  ix  ad          x       y      uzunluk   aci
   0  Istanbul     1.00    1.00    1.414     45.0
   1  Ankara       0.50    0.50    0.707     45.0
   2  Izmir       -1.00    0.30    1.044    163.3
   3  Bursa        0.20    1.10    1.118     79.7
   4  Mersin       0.40   -0.90    0.985    -66.0
   5  Sivas        1.40    0.10    1.404      4.1
   6  Adana        0.90   -0.80    1.204    -41.6
   7  Konya        0.60   -0.30    0.671    -26.6

DENETIM
  en yakin iki sehir  Mersin / Adana   uzaklik 0.510
  en yakin iki DOGRULTU  cos 1.0000   Istanbul / Ankara
  !! UZUNLUK ATILIRSA bu iki sehir AYNI NOKTA olur.
```

**Son satır bir bulgu, süs değil.** Verdiğin iki koordinat aynı
doğrultuda (ikisi de 45°), yalnız uzunlukları farklı: 1,414 ve 0,707.
model_14 her durumu birim küreye bastırıp uzunluğu atıyordu — o
mimaride İstanbul ile Ankara **ayırt edilemezdi**.

```
model_15 kurali:  OKUMA uzunlugu KULLANIR.  Durum normalize EDILMEZ.
```

Bunun bir bedeli var ve şimdiden yazılıyor: normalize etmemek, uzun
yollarda durumun patlaması ya da sönmesi riskini geri getirir.
model_14 normalize'i tam bu yüzden koymuştu. **ADIM 2'nin çözmesi
gereken şey bu** — sırayı görebilen, birleştirebilen, ama 4 adımda
patlamayan bir güncelleme.

### ADIM 1b — UZUNLUĞUN VE AÇININ ÖNEMİ VAR MI

Kullanıcı sordu. Cevap koordinatlarda değil, **okuma kuralında**.
Üç aday var, ve önemi olup olmadığını kural belirliyor. Kod:
`okuma_15.py`.

```
A  IC CARPIM   argmax <s, x_c>            uzunluk = ONEM PUANI
B  KOSINUS     argmax <s, x_c / |x_c|>    uzunluk ATILIYOR
C  EN YAKIN    argmin |s - x_c|           uzunluk + aci = KONUM
```

Sorulan tek şey: bir şehri kazandıran **herhangi** bir durum var mı.

```
  sehir        A ic carpim   B kosinus   C en yakin
  Istanbul       0,164        0,105        0,136
  Ankara       OKUNAMAZ     OKUNAMAZ       0,021
  Izmir          0,294        0,298        0,309
  Bursa          0,113        0,164        0,120
  Mersin         0,144        0,215        0,166
  Sivas          0,147        0,099        0,097
  Adana          0,138        0,055        0,132
  Konya        OKUNAMAZ       0,063        0,020
```

**İki kural, daha model kurulmadan ölüyor:**

```
A  Ankara ve Konya HIC uretilemez.  Sebep: ic carpimda ancak DISBUKEY
   KABUK KOSESI olan noktalar kazanabilir.  Ikisi de kabugun ICINDE.
   Ankara 1. ve 3. yolda var -> yol daha basta imkansiz.

B  Ankara HIC uretilemez.  Sebep: x_Istanbul = 2 * x_Ankara, yani
   uzunluk atilinca AYNI NOKTA oluyorlar.  Berabere kalip kaybediyor.

C  Sekiz sehrin sekizi de okunabiliyor.
```

### Cevap

**Uzunluğun ve açının ayrı ayrı önemi yok — birlikte KONUM oldukları
için önemliler.** İki başarısız kural, konumu daha azına indirgediği
için ölüyor:

```
kosinus     konumu ACIYA indirgiyor        -> uzunluk farki kayboluyor
ic carpim   uzunlugu ONEM PUANI yapiyor    -> uzun olan hep kazaniyor,
                                              icerideki hic kazanamiyor
en yakin    konumu KONUM olarak birakiyor  -> herkes yasiyor
```

```
KARAR:  okuma  =  argmin_c |s - x_c|
```

### Bunun model_14 için söylediği

model_14 okumayı `argmax <Pi z / |Pi z|, p_w>` ile yapıyor — yani
**tam olarak B**. Orada patlamamasının tek sebebi `p_w`'nin **rastgele**
olması: yüksek boyutta rastgele birim vektörlerin hemen hepsi kabuk
köşesidir ve hiçbiri bir diğeriyle aynı doğrultuda değildir.

> **Kosinus okumasi ANLAMLI koordinatla calismiyor, ANLAMSIZ
> koordinatla calisiyor.**  Haritaya benzeyen her yerlesim onu kirar.

Bu, ADIM 1'in açtığı "harita mı etiket mi" sorusunu da çözüyor:
**harita olabilir**, ama ancak `en yakin` okumasıyla.

### Bedeli — yazılıyor, gizlenmiyor

`en yakin` kimseyi öldürmüyor ama eşit de davranmıyor:

```
Izmir   0,309   duzlemin ucta biri        -- kenardaki sehir, BUYUK hedef
Ankara  0,021   yuzde iki                 -- iceride kalan sehir, KUCUK hedef
Konya   0,020
```

Kabuğun içinde kalan şehirlerin hücresi küçük: durumun oraya **isabet
etmesi** gerekiyor, yaklaşması yetmiyor. Ankara iki yolumuzda da var.
Bu bir arıza değil, **doğruluk talebi**: ADIM 2'deki güncelleme
Ankara'yı 0,021'lik bir hücreye koyabilmeli.

### Adım 1'in açtığı soru

Koordinatlar **harita mı, etiket mi?**

```
HARITA   yakin sehirlerin koordinatlari yakin.  Ama cografi yakinlik
         GUZERGAH yakinligi DEGIL: Mersin ile Adana 0,510 uzakta
         (en yakin cift) ama bizim ornekte aralarinda yol yok.
ETIKET   koordinatlar keyfi, butun isi cevirmeler yapar.
         model_14 boyleydi (`p` rastgele ve donmus).
```

Şu anki tablo **harita gibi** yazıldı ama model bunu henüz kullanmıyor.
Fark ancak ADIM 2'deki okuma kuralı seçilince ortaya çıkacak.

---

## ADIM 2 — YOL TUTULMAZ, HESAPLANIR

Kullanıcı: *"bu yol hesaplanan bir şey olacak. ben modele sadece
istanbul diyebilirim, istanbul ankara diyebilirim veya istanbul ankara
mersin diyebilirim."*

Kod: `model_15.py`.

### Sözlük

Her token'a **rastgele** bir konum. Hem girdi hem okumanın hedefi —
tek tablo.

```python
self.E = nn.Parameter(r(n, boyut))     # (8, 2)  rastgele, EGITILIR
```

> ADIM 1'deki harita koordinatları (Istanbul 1,0/1,0 ...) **düştü**.
> Kullanıcı kararı: konumlar rastgele atanır. `sehir_15.py` artık
> yalnız ad ve yol listesi taşıyor.

### Durum AYRI boyutta

Kullanıcı: *"s iki değerli olmaması lazım bu yanlış."* Doğru —
ilk yazımda durumu sözlükle aynı boyuta zincirlemiştim.

```
sozluk   2 sayi    bir SEHRIN konumu
durum    6 sayi    bir YOLU tasiyor
```

İkisi aynı şey değil. 6 seçildi çünkü örnekteki yol üç şehir, üçünü
birden taşıyabilecek kadar yer: `3 x 2 = 6`. Keyfi değil, ama son da
değil — yol uzarsa yeniden sorulur.

### Güncelleme

```python
s_t = M[w_t] @ s_{t-1} + b[w_t]         # s: 6 sayi
golge = s @ Q                            # 6 -> 2, okumak icin
cevap = argmin_c |golge - E[c]|
```

`M` dönme **değil**, genel matris. Gerekçe ADIM 1: 2B'de dönmeler
değişmelidir, sırayı göremezdi.

`Q` durumu sözlük uzayına indiriyor — okuma orada yapılıyor.

### Girdi değişken uzunlukta

```python
m.ileri(dizi(["Istanbul"]))
m.ileri(dizi(["Istanbul", "Ankara"]))
m.ileri(dizi(["Istanbul", "Ankara", "Mersin"]))
```

Üçü de ayrı hesap. Hiçbir yerde "yol" diye saklanan bir şey yok —
yalnız `E`, `b`, `M`, `Q`, `s0` var, ve `s` her çağrıda sıfırdan
hesaplanıyor.

### Parametre

```
  E    (8, 2)      16     sozluk
  b    (8, 6)      48     token -> duruma giris
  M    (8, 6, 6)  288     guncelleme
  Q    (6, 2)      12     okuma izdusumu
  s0   (6,)         6
                  ----
                   370
```

### Eğitilmemiş çıktı

```
  Istanbul                 s = [-0,55 -0,69  0,23  0,32  0,24 -0,63]  -> Sivas
  Istanbul Ankara          s = [-0,42  0,07  0,72  0,18  0,16 -0,41]  -> Sivas
  Istanbul Ankara Mersin   s = [-0,49  0,47  0,44  0,25  0,30  0,37]  -> Sivas
```

Cevaplar rastgele — daha eğitilmedi. Üçü de Sivas diyor çünkü gölge
henüz orijin civarında dolaşıyor. Gösterdiği tek şey **mekanizma**:
önek uzadıkça durum yürüyor, ve her önekten bir cevap okunabiliyor.

### YOL — kullanıcının tarifi

Kullanıcı: *"yol şu değil mi? istanbul(x,y) U ankara U mersin. bu
sayede hangi konumdan hangi konuma gittiğimi bilirim. bu bilgi ya da
input."*

```python
def yol(self, w):
    return self.E[w].reshape(-1)        # (T,) -> (T*2,)
```

```
  Istanbul                 2 sayi   [ -1,13  -1,15]
  Istanbul Ankara          4 sayi   [ -1,13  -1,15  -0,25  -0,43]
  Istanbul Ankara Mersin   6 sayi   [ -1,13  -1,15  -0,25  -0,43   0,32  -1,26]
  Izmir Bursa Mersin       6 sayi   [  0,85   0,69  -0,32  -2,12   0,32  -1,26]
  Konya Ankara Mersin      6 sayi   [  1,12  -0,25  -0,25  -0,43   0,32  -1,26]
```

Uzunluk **yolla birlikte büyüyor**. Hiçbir şey sıkışmıyor, hiçbir şey
atılmıyor.

Son üç satır işin tamamını gösteriyor:

```
son iki sayi UCUNDE DE AYNI      hepsi Mersin'de bitiyor
1. ve 3. satir yalniz ILK CIFTTE ayriliyor   -> ayni cevap verilecek
2. satir ILK DORTTE ayriliyor                -> farkli cevap verilecek
```

Fark **açıkça orada duruyor**, adresli. Geçişler de içinde: ardışık
çiftler `(Istanbul -> Ankara)` ve `(Ankara -> Mersin)`.

> Bu seçim `M` matrisini işsiz bırakıyor — güncellenecek bir durum
> kalmadı. Yerine "bu listeye bakıp cevabı üret" diyen bir parça
> gerekiyor. **ADIM 3.**

### ADIM 3'ün sorusu

Kayıp ne. Yani `Istanbul Ankara Mersin -> Sivas` nasıl öğretilecek.
