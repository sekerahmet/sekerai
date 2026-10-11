# Strata — hayal / süzgeç defteri

Kullanıcı, 11 Ekim 2026: *"Hayal et önce. Sonra mantık süzgecinden geçir sonra tekrar sonra yine mantık ... Bir ve ya iki adım
değil tatmin olana kadar"*; *"hafızanı belki serbest bırak. Yeteneklerin ön plana çıksın"*.

Kural: hayal turunda eleme yok; süzgeç turunda her fikir ölçülmüş bir sayıya ya da hesaba bağlanır. Etiketler: HİPOTEZ /
HESAP / ÖLÇÜLDÜ / LİTERATÜR.

---

## Tur 1 — hayal (serbest)

Bir insan okurken her kelimeye aynı emeği vermez. Bir "durum" tutar (kim, ne, nerede); kolay kelimeleri neredeyse emeksiz
tahmin eder; yalnız şaşırınca düşünür. Kelimeleri değil özü hatırlar, ama isimleri ve sayıları birebir tutar.

- H1. Durum = birkaç yuva (varlık, konu); yalnız şaşırtıcı bilgi gelince güncellenir.
- H2. Dil sürekli bir yörünge; tahmin sürekli uzayda yapılır, sonra kelimeye çözülür.
- H3. Öğrenme yalnız ağırlıkta değil, okurken yazılan hızlı bir bellekte de olur (bağlam içi öğrenme = bellek yazma).
- H4. Çıkış 50 bin kelimeye skor değil; bir vektör üretilir ve en yakın kelime bulunur.
- H5. Derinlik sabit değil; aynı çekirdek, durum oturana kadar yinelenir.
- H6. Model token token değil, "bir sonraki fikri" söyler; yüzey (kelimeler) ucuz bir katmanda dökülür.
- H7. Token birimi BPE'nin keyfi parçası değil; bilgi birimi.

## Tur 1 — süzgeç

| fikir | süzgeç | karar |
|---|---|---|
| H1 olay güdümlü durum | Sabit boyutlu durum tam geri çağırmayı kaybeder (yinelemeli modellerin bilinen zaafı). "Yalnız şaşırınca güncelle" ise Strata-x'in birim sınırının ta kendisi | Strata-x'te zaten var |
| H2 sürekli tahmin | Büyük çözücüye doğrudan bir hedef verir: bir sonraki birimin vektörü. "Sonraki kavram tahmini" daha güçlü LM bildiriyor (LİTERATÜR, `2026/nextconcept_discrete2026.txt`, `ncp_archpreview2026.txt`) | **aday**: yardımcı kayıp |
| H3 hızlı bellek | Bağlam 2.048 token; tam attention karşılanabiliyor. Uzun bağlamda değerli | park |
| H4 en yakın kelime | Normlu olasılık kaybolur (bpb ölçülemez); çarpanlı çıkış tam normlu ve ~9× ucuz (HESAP) | hayır; Parça 10 kalır |
| H5 yinelenen derinlik | Birimler eşit bilgi taşıdığı için her birim benzer zorlukta (bilgi sd 2,4; ÖLÇÜLDÜ): bilgi eşitleme, uyarlamalı derinlik ihtiyacını zaten azaltıyor | gerek yok — ızgara bunu çözüyor |
| H6 fikir sonra yüzey | Strata-x'in kendisi | var |
| H7 bilgi birimi | Strata-x'in kendisi | var |

Süzgeçten çıkan güzel bir sonuç: **bilgiyi eşitleyen ızgara, derinlik eksenindeki uyarlamayı gereksiz kılıyor.** Üç eksenli AMR
iki eksene iniyor (x ve y); t ekseni x'in sonucu.

## Tur 2 — hayal (en zayıf noktaya)

Strata-x'in en zayıf yeri: birim içinde büyük çözücünün durumu tazelenmiyor; 4 token'lık birimin son token'ı birimin başındaki
g'yi görüyor.

- H8. Büyük çözücü tek vektör değil, bir **plan** üretir: sonraki birimin her yuvası için bir vektör (k yuva). Kod çözücü
  token'ı yazarken kendi yuvasının plan vektörünü görür.
- H9. Plan, bir sonraki birimin kodlayıcı vektörünü tahmin etmeye zorlanır (H2 ile birleşir): büyük çözücü "söyleyeceği
  fikri" açıkça öngörür.
- H10. Eğitim ilerledikçe kodlayıcı iyileşir, sürpriz düşer, birimler büyür, büyük çözücü daha az noktada çalışır:
  **model öğrendikçe ucuzlar.**

## Tur 2 — süzgeç

- H8: plan = d_g → k·d_local doğrusal haritası; birim başına bir kez (HESAP: d_g·k·d_l = 1024·4·512 = 2,1M çarpma, büyük
  çözücünün birim başına ~151M'inin yanında yok sayılır). Bilgi birim başına ~θ nat, yani yuva başına θ/k nat: plan ne kadar
  bilgi taşıyacağını biliyor. **aday.**
- H9: yardımcı kayıp, büyük çözücüye token CE'sinden daha doğrudan bir sinyal; ama hedef (kodlayıcı vektörü) kendisi
  öğreniliyor, çöküş riski (iki taraf sabite gider). Çare: hedefte gradyan kesilir (stop-grad) ve kosinüs. **aday, riskli.**
- H10: E4 günlüğünden ölçülür (birim / token adım adım). Doğruysa eğitim maliyeti zamanla düşer. **ölçüm.**

## Tur 3 — mantık (tasarımın kendisine)

- R-A. Küçük kodlayıcının sürprizi büyük modelinkinden kaba: birimler bilgi sd 2,4 (büyük model) ile 5,3 (unigram) arasında
  kalır. E4 sonrası ölçülür.
- R-B. Büyük çözücüde RoPE konumu = birim sırası; birimler farklı token uzunluğunda → mesafe bozuk. İçerik tabanlı
  attention'da önemsiz olmalı (Model Z'de G NoPE iyiydi; ÖLÇÜLDÜ). Sonra NoPE da denenebilir.
- R-C. İsim kopyalama: ismin ilk geçişi yüksek sürprizli (kendi birimi), ikinci geçişi bağlamdan tahmin edilebilir → büyük
  birime gömülür; kod çözücünün penceresi (256) dışındaysa ismi büyük çözücüden (önceki birime attention) almalı. Okumada
  gözle bakılacak.
- R-D. Yoğun maske (B, T, T) GPU'da yavaş; pencereli flex maske sonra.

## Şimdiki karar

E4 (Strata-x tabanı) çalışsın; ardından E5 = Strata-x + plan (H8) + sonraki birim tahmini (H9). Plan kodda seçenek olarak
şimdi yazılır.
