# RGN — Rasyonel Çekim Ağı (Rational Gravity Network)

Decoder-only bir dil modeli. Kelimeler arası etkileşim **ters kare çekim yasasıyla** (anlam uzaklığı + sıra uzaklığı),
kelime başına dönüşüm **rasyonel MLP** ile yapılır. Tasarım dokümanı (`docs/RGN_tasarim_dokumani.md`) yerelde tutulur, depoda yoktur.

```
çekim:   w_ij = 1 / (1 + γ_h ||q̂_i − k̂_j||² + β_h (i − j)),   α_ij = w_ij / Σ_k w_ik,   S_i = X_i + [Σ_j α_ij V_j] W_O
MLP:     u = W_1 RMSNorm(S) + b_1,   H = S + W_2 · u / sqrt(1 + u²) + b_2
```

## Klasör yapısı

```
├── docs/                    tasarım dokümanı — YEREL (depoya girmez)
├── rgn/                     Python paketi
│   ├── config.py            ModelConfig / DataConfig / TrainConfig, JSON deney dosyaları ("extends" + --set)
│   ├── paths.py             Drive kökü, önbellek, runs klasörü (Windows / Colab otomatik)
│   ├── model/               norm.py · position.py · attention.py · mlp.py · rgn.py (RGNModel)
│   ├── data/                registry.py (veri setleri) · loader.py · prepare.py (ham metin -> token)
│   ├── train/               trainer.py · optim.py (AdamW + Muon/NorMuon) · schedule.py · evaluate.py ·
│   │                        diagnostics.py (γ, β, MLP doyması, dikkat entropisi) · checkpoint.py
│   └── bench/speed.py       dikkat çekirdeği ve tam model hız / bellek profilleri
├── configs/                 deney dosyaları
│   ├── base/                mimari ve veri tabanları (rgn_d512, train_ss, train_tinystories, train_fineweb, cpu_tiny)
│   ├── ss/                  SimpleStories: ana model + karşılaştırma ve ablation'lar + duman testi
│   ├── tinystories/
│   └── fineweb/
├── scripts/                 train.py · prepare_data.py · bench_speed.py · summarize_runs.py · sync_drive.py
├── colab/                   RGN_colab.ipynb · setup_colab.py · README.md
└── tests/                   run_tests.py (pytest gerekmez) · test_model / test_attention / test_data / test_train
```

## Model seçenekleri (`configs` içinde `model` bölümü)

| Alan | Değerler | Varsayılan |
| :--- | :--- | :--- |
| `attn` | `rgn` (çekim) · `softmax` (karşılaştırma tabanı) | `rgn` |
| `pos` | `additive` (payda + β(i−j)) · `alibi` (payda × (1 + β(i−j))) · `rope` · `rope+alibi` · `none` | `additive` |
| `mlp` | `rational` (u/√(1+u²)) · `glu` (cebirsel GLU) · `swiglu` (karşılaştırma tabanı) | `rational` |
| `attn_impl` | `auto` (CUDA: flex, CPU: dense) · `dense` · `flex` | `auto` |
| `power` | çekim üssü p: 1/(1 + γ d^p) | 1 |

## Veri setleri

| Ad | Kaynak (Drive) | Durum |
| :--- | :--- | :--- |
| `ss` | `simplestories/gpt2/{train,valid}.npy` — 605,6M token | hazır |
| `tinystories` | `tinystories/TinyStoriesV2-GPT4-*.txt` -> `tinystories/gpt2/*.bin` | `prepare_data.py --dataset tinystories` |
| `fineweb` | `fineweb/gpt2/shard_000..013.bin` — ~750M token / parça | hazır |
| `synthetic` | bellekte Markov zinciri | testler |

## Hızlı başlangıç

```bash
python tests/run_tests.py                                              # testler (CPU)
python scripts/train.py --config configs/base/cpu_tiny.json            # yerel duman testi
python scripts/prepare_data.py --list                                  # veri durumu
python scripts/sync_drive.py                                           # kodu Colab için Drive'a eşitle
python scripts/train.py --config configs/ss/ss_d512_rgn.json           # (GPU) ana model, SS 1 epok
python scripts/summarize_runs.py                                       # sonuç tablosu
```

## Çıktılar (`runs/<koşu>/`)

`config.json` · `metrics.jsonl` (train / eval / diag kayıtları) · `checkpoint.pt` · `model.pt` · `results.json` · `samples.json`
