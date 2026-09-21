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

## ADIM 1 — İŞ NE

### Şehirler

Düzlemde sabit noktalar. Verilen iki tanesi kullanıcının, kalanı örneği
tamamlamak için:

```
     sehir        x      y
     Istanbul    1,0    1,0        <- kullanicinin verdigi
     Ankara      0,5    0,5        <- kullanicinin verdigi
     Izmir      -1,0    0,3
     Bursa       0,2    1,1
     Mersin      0,4   -0,9
     Sivas       1,4    0,1
     Adana       0,9   -0,8
```

Bu noktalar **hiç değişmiyor**. Eğitilmiyor, oynatılmıyor. Dünyanın
sabit hâli.

### İş

İki güzergâh. İkisi de Mersin'de bitiyor, ve devamları **farklı**:

```
   Istanbul -> Ankara -> Mersin -> ?        cevap  Sivas
   Izmir    -> Bursa  -> Mersin -> ?        cevap  Adana
```

Model, son şehir aynı olmasına rağmen **nereden geldiğine bakarak**
ayırmalı.

Ve tersi de istenebilir — veri öyle diyorsa **birleştirmeli**:

```
   Istanbul -> Ankara -> Mersin -> ?        cevap  Sivas
   Konya    -> Ankara -> Mersin -> ?        cevap  Sivas      AYNI
```

**İkisi de istenecek. Mimari ikisini de yapabilmeli.** Birini yapıp
ötekini yapamamak, model_14'ün düştüğü yer (§5.1/X).

### Başarı ne demek

Tek bir cümle, ve ölçülebilir:

```
   Modele bir yol onekini birim birim verdikten sonra, durumdan
   OKUNAN sehir, verinin o onekten sonra soyledigi sehir olmali.
```

Kısmî başarı yok: yol bitene kadar her adımda doğru olmalı, yoksa
sonraki adımın girdisi zaten bozuk.

### Adım 1'in açtığı soru

Koordinatlar **harita mı, etiket mi?**

```
HARITA olursa    yakin sehirlerin koordinatlari yakin.
                 Model "Mersin'e yakin bir yer" diyebilir, tam
                 bilemese de yakin dusebilir.  Ama YANLIS bir
                 dusunceyi de ucuzlastirir: cografi yakinlik,
                 GUZERGAH yakinligi DEGIL.
                 (Ankara-Mersin komsu degil ama yol var;
                  Bursa-Izmir yakin ama belki yol yok.)

ETIKET olursa    koordinatlar rastgele, aralarinda anlam yok.
                 model_14 boyle yapti (`p` rastgele ve donmus).
                 Butun isi cevirmeler yapar.
```

Kullanıcının verdiği iki sayı (`1,0 / 1,0` ve `0,5 / 0,5`) **aynı
doğrultuda** — açıları eşit, yalnız uzunlukları farklı. Bu tek başına
bir şey söylüyor: eğer okuma açıya bakarsa bu iki şehir **aynı nokta**.
model_14 tam bunu yapıyordu (her şeyi birim küreye bastırıp uzunluğu
atıyordu).

**ADIM 2'de karara bağlanacak.**
