# Colab'da RGN

## Akış

1. **Yerelde** değişiklikleri commit edip GitHub'a gönder (kod Colab'a git'ten gider, veri Drive'dan okunur).
2. **Colab'da** `colab/RGN_colab.ipynb` (`python colab/make_notebook.py` üretir) hücrelerini çalıştır:
   - 0 HAZIRLIK: Drive bağlama, `BRANCH` kolundan taze kod, kurulum (`setup_colab.py`)
   - 1 TEST: test takımı (CUDA testleri dahil: Flex pencere, özel nöron çekirdeği)
   - 2 KOŞU: deney kuyruğu arka planda; 3 NABIZ: durum; 4 KIYAS: tabloya fark
   - 5 PROFİL: eğitim adımı bölüm bölüm; 6 DÖNGÜ: greedy tekrar testi
   - X DURDUR / Z GPU: checkpoint kaydedip durdur / çalışma zamanını kapat

## Klasörler (Colab)

| Yol | İçerik |
| :--- | :--- |
| `/content/RGN` | kodun çalışan kopyası |
| `/content/rgn_cache` | token dosyalarının yerel kopyası (Drive'dan okumak yavaş) |
| `/content/drive/MyDrive/RGN/runs/<koşu>` | koşu çıktıları — oturum kapansa da kalır, yerelde `G:\Drive'ım\RGN\runs` altında görünür |

## Notlar

- Oturum düşerse aynı komutu `--resume` ile tekrar çalıştır; checkpoint 10 dakikada bir yazılır.
- Bellek yetmezse `--set train.micro_batch=16` (adım başına token sayısı değişmez, yalnız mikro adım sayısı artar).
- GPU türü değişebilir (T4 / L4 / A100 / RTX PRO 6000); hız sonuçlarını karşılaştırırken `results.json` içindeki `env.gpu` alanına bak.
