# Hangi sürüm hangi deneyi koştu

CLAUDE.md §6: çalışan dosya **her zaman** `sifirdan.py` kalır; yamalanınca
eski hali burada dondurulur. Bu tablo olmadan "şu bulguyu hangi kod üretti"
sorusu cevapsız kalır.

| dosya | md5 | bayt | ne getirdi | koşturduğu kollar |
|---|---|---|---|---|
| `sifirdan_v1_DONDURULDU.py` | — | 66754 | ilk çekirdek | A, B, C |
| `sifirdan_v2_DONDURULDU.py` | — | 70030 | sürdürme paketi, `RESUME_FROM` | C devamı, D5 kimlik |
| `sifirdan_v3_DONDURULDU.py` | — | 71034 | `IDENT_FRAC`, kimlik denetimi | D6 |
| `sifirdan_v4_DONDURULDU.py` | `7ae697486d` | 71488 | **`MASK_KEY`/`MASK_BLK`** + SHARE/INIT_FROM/IDENT_MODE | **D, K, E1, E2, D3** — ve Qwen tarafı (Q1b, Q1c, Q2) bu sürüm yürürlükteyken yapıldı | 
| *(çalışan)* `sifirdan.py` | `ffd9709430` | 72246 | **`WARM_OF`** — ısınma toplam koşudan | **D3.1**, ve sırada D3.2 |

GitHub karşılıkları: v4 = commit `21196a4`, çalışan sürüm = `c04942e` ve sonrası.

---

## v4 → çalışan sürüm arasındaki tek fark

`WARM_OF`. Öncesinde ısınma `warm = max(10, STEPS // 20)` idi — yani **bu
koşuda gidilecek hedefe** bağlıydı. Eğitim parçalı yürütülünce (D3.1: her
5.000 adımda dur-ara-devam) ilk parçanın ısınması 6.000 yerine **250** adım
oluyor, aynı tohum başka bir yörünge izliyordu.

Bayrak kapalıyken bit düzeyinde no-op: 28/28 tensör aynı, maxfark `0.00e+00`.
Yani **v4 ile üretilmiş bulgular geçerli**; `WARM_OF` verilmediği sürece
çalışan sürüm v4 ile özdeş davranır.

Ölçülen: açıkken 28/28 tensör farklı; hatalı girdide iki assert ateşliyor
(`WARM_OF < 0`, `WARM_OF < STEPS`).

---

## v5 — anlık görüntü yazımı atomik  (commit `90fa63c`, 13 Eylül 2026)

```
md5  8d5cc03c18c33e11795c33a7cbfaf29c   (v4, D3 / D3.1 / D3.2 / A4)
md5  0fd7975bc9325d7558fb8dac91527e55   (v5, D3.3'ten itibaren)
```

**Tek değişiklik:** `snap_<kol>_s<tohum>_<adım>.pt` artık `.tmp` + 
`os.replace` ile yazılıyor. Düz yazmada runtime tam o anda koparsa geçerli
ADLI ama yarım bir dosya kalıyordu; sürdürme paketi zaten korunuyordu ama
**birincil ölçümün dayandığı** snapshot açıktaydı.

**No-op doğrulaması (CLAUDE.md §6 test 1), aynı tohum aynı konfig:**

```
3 anlik goruntu x 28 tensor = 84 tensor   BIT-AYNI
49 egri alani x 3 olcum                   AYNI  (tek fark: secs = duvar saati)
```

> **Tuzak, kayda geçsin:** ilk karşılaştırma dosya md5'i üzerinden yapıldı
> ve 5 dosya "FARKLI" çıktı. Kontrol koşusu (yamasız sürüm iki kez) farkın
> yamadan gelmediğini gösterdi: `torch.save` zip arşivinin iç önekini dosya
> adından türetiyor (`snap_..._000200/` yerine `snap_..._000200.pt/`), yani
> baytlar değişiyor, tensörler değişmiyor. §6 zaten "tüm ağırlık
> tensörleri" diyor; dosya md5'i yanlış ölçüttü.

> `mem_H` / `mem_olu` / `mem_tepe` alanları da "farklı" görünmüştü —
> üçü de `NaN` ve `NaN != NaN`. Determinizm sorunu değil, karşılaştırma
> artifaktı.

---

## Buradan bir sürümü geri çağırmak

Arşiv dosyaları **çalıştırılmak için değil, okunmak için**. Bir bulguyu
yeniden üretmek gerekirse GitHub'daki commit'ten klonla — `analiz.py`
`import sifirdan` yaptığı için dosya adının `sifirdan.py` olması şart:

```bash
git clone https://github.com/sekerahmet/sekerai.git /tmp/v4 && cd /tmp/v4 && git checkout 21196a4
```
