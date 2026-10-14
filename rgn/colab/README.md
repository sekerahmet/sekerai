# Colab'da RGN

## Akış

1. **Yerelde** kodu Drive'a eşitle (Drive masaüstü uygulaması `G:\Drive'ım`'ı Google Drive'a yükler):
   ```
   python scripts/sync_drive.py
   ```
2. **Colab'da** `colab/RGN_colab.ipynb` notebook'unu aç (Drive'daki `RGN/code/colab/` altından) ve hücreleri sırayla çalıştır:
   - Drive bağlama, kurulum (`setup_colab.py`)
   - Testler (CUDA testleri dahil: flex = dense, γ/β gradyanı)
   - Hız profili (dikkat çekirdekleri)
   - Duman testi (`configs/ss/ss_smoke.json`, ~300 adım)
   - SS karşılaştırma deneyleri (Aşama 0)
   - Sonuç tablosu

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
