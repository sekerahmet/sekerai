# Adlandırma — deneme 2

**15 Eylül 2026.** Tek kural: **bir modelin adı her yerde aynıdır.**

```
model_a              <- AD
model_a.py           <- modelin kendisi
model_a.ipynb        <- onun Colab defteri (SABLON.ipynb'in KOPYASI)
pencere_a.py         <- onun olcumu
belge/onkayit/model_a.md    <- kosudan ONCE
belge/bulgu/model_a.md      <- sonuc
Drive: deneme2/model_a/     <- onun ciktisi
```

## Tohum

Ayni model UC TOHUMLA kosar: `t0`, `t1`, `t2`. Ayri model degil, AYNI
modelin tekrari -- tek fark rastgele baslangic ve batch sirasi.
**Veri ucunde de AYNI** (`veri_tohum` ayri bir alan, 0'da sabit).

```
Drive: deneme2/model_a/t0/  t1/  t2/
       snap_model_a_t0_00020000.pt
       egri_model_a_t0.json     ayar_t0.json
```

Sebebi: grokking tohuma bagli. 2603.25009'da "only 1 of 3 seeds grokked"
gibi sonuclar var. Tek tohumla bir sey gormezsek "konfigurasyon yanlis" mi
"bu baslangic sanssiz" mi ayiramayiz.

```
3/3 yukseldi  ->  saglam
1/3 yukseldi  ->  oluyor ama KIRILGAN
0/3 yukseldi  ->  konfigurasyon yanlis, sansizlik degil
```

**Colab: kuyruk/surucu YOK.** Bir defter = bir model. `SABLON.ipynb`i
kopyala, adini modelin adi yap, TEK SATIRI degistir. Sablonda `MODEL = None`
ve bir `assert` var: kopyalayip duzeltmeyi unutan SESSIZCE baska bir modeli
kosturamaz. (Eskiden HUCRE 0 onceki deneyin ayarlarini tasiyordu ve tam
bu oluyordu.)

Ekrana bakınca hangi modele ait olduğu belli. **Arama yapmadan.**

Eskiden böyle değildi: `sablon/pencere.py` bütün deneyleri ölçüyordu,
`deney/g.py` ile `deney/h.py` aynı `kosu.py`'yi çağırıyordu, ve bütün
koşuların çıktısı `snap_A_s0_*.pt` idi — G de GM de. Hangi dosyanın hangi
koşuya ait olduğu **klasör adından** çıkarılıyordu, o kadar.

## Harf ve sayı

```
model_a     TABAN            bir mantik
model_a1    VARYASYON        ayni mantik, tek dugme farkli
model_a2    VARYASYON
model_b     YENI MANTIK      farkli bir sey deniyoruz
model_b1    onun varyasyonu
```

**Nokta yok.** Eskiden `D3.1`, `D3.2`, `D3.3` vardı ve noktalar var olmayan
bir soy ağacı ima ediyordu: `D3.2` yanlış eksende bir çıkmaz, `D3.3` bambaşka
bir eksendi, ikisi de `D3`'ün alt sürümü değildi. Harf+sayı yalan söylemiyor.

**Harf tekrar kullanılmaz.** İptal olsa bile geri dönmez.

**Arşivdeki harflerle ilgisi yok.** `arsiv/` içindeki A, D, K, G, GM ayrı bir
dünya. Deneme 2 `model_a`'dan başlar.

## Araç AİLEYE aittir, varyasyona değil

```
model_a  model_a1  model_a2   -> hepsini pencere_a.py olcer
model_b  model_b1             -> hepsini pencere_b.py olcer
```

Bunun sebebi var: **kıyas hep aile içinde yapılır.** `model_a` ile `model_a1`
karşılaştırılıyorsa ikisi de aynı ölçüm kodundan geçmeli, yoksa fark ölçüm
farkı mı model farkı mı bilinmez. Bu bir kez oldu: bir sayı eğriden, kıyaslananı
kendi kodundan alınmıştı.

`model_a` ile `model_b` kıyaslanacaksa **ayrı bir iştir** ve ayrı yazılır —
çünkü zaten farklı mantıklar.

---

## `sifirdan.py` neyi yanlış yapıyordu

Yeni düzen bunları tekrarlamamak için. Hepsi fiilen oldu:

**1. Ayarlar `os.environ`'dan, modül seviyesinde, import anında okunuyordu.**
```python
MASK_KEY = os.environ.get("MASK_KEY", "")     # import edilince OKUNDU
```
Her araç `import` etmeden önce ortam değişkeni kurmak zorundaydı; önceki
hücrelerden sızan `MEM_AT` assert patlattı; "alt sürece temiz ortam ver" diye
bir bakım yükü doğdu. **Yeni düzende ayar bir nesne, import yan etkisi yok.**

**2. Kol kimliği ortam değişkenindeydi, dosya adında değildi.** Yukarıda.

**3. Türetilmiş büyüklük, kontrol parametresine bağlıydı.**
```python
warm = max(10, STEPS // 20)
```
Koşuyu parçalara bölünce ısınma 6.000 değil **250** adım oldu; aynı tohum
başka yörünge izledi ve referans kol olmaktan çıktı.

**4. 1500 satır, her şey iç içe:** konfig + veri + model + eğitim + ölçüm +
sonda + tanı + yedekleme. Bir yeri değiştirmek diğerini kırıyordu.

---

## "ŞARTLAR EŞİT" nasıl korunacak?

Tek dosya olmasının **tek gerçek faydası** buydu: iki kol aynı koddan
çıkıyordu. Ayrı dosyalara bölünce bu garanti kendiliğinden gitmez, yerine
bir şey koymak gerekir. İki şey:

**(a) Varyasyon tabanı IMPORT eder, kopyalamaz.**

```python
# model_a1.py
from model_a import AYAR, Model, veri_kur, egit

AYAR = AYAR.degistir(ident_frac=0.15, ident_kip="q2son")
#      ^ SADECE FARK. Mimari/veri/tohum/adim TEKRAR YAZILMAZ.
```

Varyasyon dosyası mimariyi yeniden tanımlayamaz; yalnız ilan edilmiş alanları
değiştirebilir. Kopyala-yapıştır yasak değil, **imkânsız**.

**(b) Koşu başında ayarlar MEKANİK karşılaştırılır ve loga basılır.**

```
model_a  vs  model_a1
   FARKLI :  ident_frac  (0.0 -> 0.15)    ident_kip  ("" -> "q2son")
   AYNI   :  49 alan
```

"Tek fark şu" artık bir iddia değil, **çıktı**. Kol C tam bu yüzden geçersiz
kaldı: tek okuma noktasıyla eğitilmiş paketten üç okuma noktasıyla sürdürüldü,
hiçbir şey hata vermedi, kimse fark etmedi.

---

## Klasör

```
deneme2/
    ISIMLENDIRME.md
    SABLON.ipynb     <- kopyalanacak defter (MODEL = None)
    veri_okul.py     <- veri ureteci, GOREVI tanimlar, modele ait DEGIL
    model_a.py       model_a.ipynb
    pencere_a.py     <- olcum (henuz yazilmadi)
```

Cikti depoda DEGIL, Drive'da: `deneme2/<model>/`. Kosu ciktisi ikili ve
buyuk; `.gitignore`da `*.pt *.npz *.parquet` var.
