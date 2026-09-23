# model_18 -- success factors

Kullanici, 24 Eylul: *"modelin başarısı nerde duracağını bilmek yani 13 + 25 = 38
cevabı vermesi 13+130 = 143 yani cevabın uzunluğunu bilmesi. ikincisi de doğru
bilmesi. öncelikle eğitimde olanların doğruluğu ve eğitimde olmayanların
doğruluğu."*

Kosu: `MAT_COK_PV` (notebook.ipynb, hucre 1).  Veri: model_15 `veri_cok.pt`
(2+3 terim), train ve heldout bolmesi AYNEN onlarinki.

## Olculer

```
1. NEREDE DURACAGINI BILMEK    length_ok      cevabin uzunlugu dogru, EOS dogru yerde
                                               13 + 25 = 38 (2 hane), 13 + 130 = 143 (3 hane)
2. DOGRU BILMEK                accuracy       cevap BIREBIR dogru (exact match)
     egitimde olanlar          train_acc
     egitimde olmayanlar       heldout_acc
```

Hepsi pakette ve gunlukte: `train_acc`, `heldout_acc`, `train_diag.length_ok`,
`heldout_diag.length_ok` (her 100 adimda 2.000 soruda; sonda sorularin TAMAMINDA).

## Kiyas noktalari (ayni veri, ayni olcut, 16.000 adim)

```
                         heldout_acc   train_acc   length_ok (heldout)
model_15 (eski mimari)   0,4403        0,4456      -
model_17 A (bellek 512)  0,2571        0,2602      0,9868
model_17 1b (1,2M par)   0,6929        0,7620      0,9816
```

## Esik YOK

Kullanici, 24 Eylul: *"bu tarz eşik değerleri birşey ifade ediyor mu ? yani model
%30 mesela doğru cevap veriyorsa model güvenilmez ki"* ve *"o yüzden eşik değeri yok"*.

Toplama bir kural: ogrenildiyse gorulmemis sorularda da isler.  Ara degerler
basari degil TANI (hangi basamak, nerede duruyor, train ile heldout farki).
Kiyas noktalari mimariler arasi ilerlemeyi gosterir, guvenilirligi degil.

Kullanici, 24 Eylul: *"eşik değeri diye birşey yok. biz karar veriyoruz bu model iyi
ve ya kötü çünkü göreve göre değişiir matematikte hepsini bilmektir bana göre çünkü
adres belli"*.

MATEMATIKTE BASARI = HEPSINI BILMEK: train ve heldout sorularinin hepsi birebir
dogru, hepsinde dogru yerde durmak.  Hukmu kullanici verir.
