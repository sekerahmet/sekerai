# Klasör haritası

*13 Eylül 2026'da tasnif edildi. Yeni dosya koyarken buraya da bir satır ekle.*

Kural: **kökte yalnızca gerekçesi olan dosya durur.** Gerekçe ya "her oturumda
okunuyor" ya da "`import sifirdan` yapıyor, alt klasöre inince kırılır"dır.

---

## Kök

| dosya | neden kökte |
|---|---|
| `CLAUDE.md` | çalışma kuralları, her oturumda okunur |
| `HARITA.md` | bu dosya |
| `README.md` | projenin bir paragraflık tanımı |
| `sifirdan.py` | **çalışan çekirdek.** CLAUDE.md §6: asla taşınmaz, asla V2 adı almaz |
| `analiz.py` | `import sifirdan` — alt klasöre inerse import kırılır |
| `tani_rol.py` | `import sifirdan` — aynı sebep |
| `SABLON_deney.ipynb` | aktif deney şablonu (7 hücre) |
| `uret_sablon.py` | şablonu üreten betik — **defteri elle düzenleme, buradan üret** |

> Python, betiğin **kendi klasörünü** `sys.path`'e koyar, çalışma dizinini
> değil. `cozumleme/x.py` içinden `import sifirdan` bu yüzden çalışmaz.
> Kökte kalma gerekçesi bu, estetik değil.

## `sablon/` ve `deney/` — GitHub'dan akar

Depo ile **birebir aynı** olmalı; Colab bunları klonluyor.
Değiştirince: yerel → `git push` → Colab `git clone`. Colab'da yama yok (§7).

```
sablon/kosu.py      adim adli surdurme, parca basina log, durum/konfig,
                    yorunge ve konfig kapilari          (deneyden BAGIMSIZ)
sablon/bakici.sh    ayri surec: 5 dk'da bir Drive yedegi + arsiv + rapor
sablon/arsivle.py   surdurme paketini ICINDEKI adimla kopyala -> geri donulebilir
sablon/rapor.py     tek metin rapor; hesap yapmaz, GPU istemez, cekirdek istemez
deney/d31.py        D3.1'in karar mantigi (arama, esik, faz gecisi)
```

## `belge/`

| klasör | ne | not |
|---|---|---|
| `onkayit/` | 10 önceden kayıt | **değiştirilmez** (§11). Sonradan seçilen okuma `[SONRADAN SEÇİM]` etiketlenir |
| `bulgu/` | 9 bulgu | hata silinmez, üstü çizilip düzeltilir (§11) |
| `fikir/` | `NEW_AI_MODEL_IDEAS.md`, `ozet_rapor_denemeler.md`, `colab.md` | ham fikir ve eski özetler |
| `hakemlik/` | `HAKEMLIK_20260914_LITERATUR.md` | bulguların literatüre karşı bağımsız hakemliği. Her iddia `[METİN]` / `[ÖLÇÜLDÜ]` / `[DAYANAKSIZ]` etiketli |

## `cozumleme/` — yeniden koşulabilir araçlar

Ölü kod değil; kayıtlı çıktı üzerinden tekrar koşar, çoğu GPU istemez.

```
analiz_A.py   A kolunun tam tersine muhendisligi (kontrol noktasi + veri.npz, ~3-5 dk CPU)
kiyas_D.py    Deney 6: D kollarinin A+ ile ESLESTIRILMIS kiyasi (onceden kayitli protokol)
oyna.py       egitilmis modelle etkilesim (sohbet degil: varlik/iliski kimlikleri)
```

Kökten çağır: `python cozumleme/analiz_A.py`

## `cikti/`

```
cikti/A_kontrol/    35 MB — A kolunun ham analiz ciktisi:
                    analiz_A.json/.npz/.png, kafa_ablasyon.npz, model_A_s0.pt,
                    prob_yonu.pt, tani_rol.json, veri.npz
```

§9: *"yapacağın analiz henüz icat edilmedi, buna göre kaydet."* Ağırlık
ortalaması bulgusu (ENT 0.050 → 0.310) sırf ara kontrol noktaları
saklandığı için mümkün oldu. Buradan bir şey silme.

## `arsiv/`

Çalışmayan değil, **artık kullanılmayan**. Geçmişi okumak için duruyor.

| klasör | ne | neden arşiv |
|---|---|---|
| `cekirdek_eski/` | `sifirdan_v1/v2/v3_DONDURULDU.py` | §6: eski çekirdek dondurulur, silinmez |
| `colab_hucre/` | 16 dosya: `colab_*.py`, `_c1..7.py`, `deney9_D3.ipynb`, yedek betiği | Deney 5-9'un tek seferlik Colab hücreleri. Şablon bunların yerini aldı |
| `kopru_llm/` | 11 dosya: `kopru*.py`, `teshis*.py`, `yazim_*.py`, `son_test.py`, `asamaB2.py`, `kapi_deneyi.py`, `yeniden_analiz.py` | **11-12 Eylül'ün gerçek-LLM hattı.** Köprü adreslenebilirliğini Qwen/2WikiMultihop/Compositional Celebrities üzerinde ölçüyordu. Sentetik `sifirdan.py` hattı bunun yerine geçti; Q1b/Q2 gerçek modele buradan değil, temiz bir tasarımla döndü |

## Silinenler

| ne | neden |
|---|---|
| `_sekerai/` | **eski commit `21196a4`'te ikinci bir `sifirdan.py` kopyası** (71488 bayt, `WARM_OF` öncesi). §7'nin en pahalı hatası tam buydu: iki klon, hangisinin koştuğu belirsiz. Temizdi, her şeyi itilmişti; depo GitHub'da, taze klon tek komut |
| `__pycache__/` | derleme artığı |
| `sifirdan_out/` | boş |

---

## BEKLEYEN — Drive tasnifi (D3.1 BİTTİKTEN SONRA)

Drive'da 10 ayrı kök klasör var. Tek çatı altına toplanacak, **ama koşu
bitmeden değil**: `deney4_tekrarli_erisim/egri_A_s0.json` D3.1'in yörünge
kapısının referansı (her durakta okunuyor) ve `deney_d31/`'e bakıcı her
5 dakikada yazıyor.

```
MyDrive/sekerai/
    kol/A_tohum0/          <- deney4_tekrarli_erisim
    kol/BC_erken/          <- sifirdan_final + eski_C_80000
    kol/D_K_maske/         <- deney7_maske
    kol/E1_E2_tohum1/      <- deney8_tohum1
    kol/D3_zemin/          <- deney9_D3
    kol/D31_arama/         <- deney_d31
    gercek_model/Q2/       <- deney11_Q2
    gecersiz/              <- deney12_D31_GECERSIZ_warm_hatasi, sifirdan_out2*
```

Taşımadan sonra **mutlaka** güncellenecek: `SABLON_deney.ipynb` HÜCRE 0'daki
`REF_EGRI`. Kol envanteri: `belge/KOLLAR.md`.

## Nereye ne koyulur

```
yeni deney         -> deney/<ad>.py          + SABLON_deney.ipynb HUCRE 0
sablon degisikligi -> sablon/                + git push (Colab oradan ceker)
onceden kayit      -> belge/onkayit/         KOSMADAN ONCE yazilir
bulgu              -> belge/bulgu/
tek seferlik analiz-> cozumleme/             (sifirdan import ediyorsa KOKE)
ham cikti          -> cikti/<kol>/           silme, uzerine yazma
```
