# YYG + kelimeye özel nöron

Küçük decoder-only dil modeli araştırması. Taban: softmax dikkat + RoPE + SwiGLU transformer, **yerel / genel katman düzeni**
(YYG: yerel katman son 128 token'a, genel katman bütün geçmişe bakar; 8 katmanda `YYGYGYYG`). Üzerine:

- **Kelimeye özel nöron** — her katmanda sözlükteki her kelimenin kendi nöronu vardır. Adres token kimliğinden bellidir;
  nöron yalnız o kelimenin geçtiği konumda çalışır ve yalnız o kelimeden gradyan alır. Ortak MLP'ye paraleldir:

  ```
  x = RMSNorm(S),   satır = [g | u | w]  (kelimenin satırı, katman başına)
  H = S + SwiGLU(x) + SiLU(g·x)·(u·x)·w
  ```

  Eğitimde yalnız adımda geçen satırlar kopyalanır; ileri / geri birleşik Triton çekirdeğiyle (`rgn/model/private_kernel.py`),
  güncelleme satır seyrek optimizer'la (`RowAdam`, satır başına bias düzeltmesi) yapılır.
- **Dikkat kapısı** (isteğe bağlı) — head ya da öğe başına çıktı kapısı, sıfır başlatmada birim.
- **Hikâye düzeni** (SimpleStories) — satırlar hikâye başından başlar, cümle sonunda kesilir; yerel katman istenirse kendi cümlesine bakar.

Eski RGN deneyleri (ters kare çekim dikkati, rasyonel MLP ve türevleri) git etiketi `arsiv/rgn`'dedir.

## Klasör yapısı

```
├── rgn/                     Python paketi
│   ├── config.py            ModelConfig / DataConfig / TrainConfig, JSON deney dosyaları ("extends" + --set)
│   ├── paths.py             Drive kökü, önbellek, runs klasörü (Windows / Colab otomatik) · system.py (süreç ayarları)
│   ├── model/               norm.py · position.py (RoPE) · attention.py (YYG, kapı, hikâye maskeleri) · mlp.py (SwiGLU) ·
│   │                        private_kernel.py (özel nöron Triton çekirdeği) · rgn.py (Block, RGNModel)
│   ├── data/                registry.py (veri setleri) · loader.py · stories.py (hikâye düzeni)
│   └── train/               trainer.py · step.py (eğitim adımı) · optim.py (AdamW, Muon / NorMuon, RowAdam) · schedule.py ·
│                            evaluate.py · diagnostics.py (dikkat entropisi, MLP doyması) · checkpoint.py
├── assets/                  özel satırı olan kelime listeleri (ss_vocab_used, fw_s000_vocab_used)
├── configs/                 deney dosyaları: base/ (train_ss, cpu_tiny) · ss/ · fineweb/
├── scripts/                 train.py · profile_step.py · loop_test.py · story_eval.py · freq_gain.py · loop_trace.py ·
│                            trace_token.py · summarize_runs.py · sync_drive.py
├── colab/                   RGN_colab.ipynb (make_notebook.py üretir) · runner.py · setup_colab.py
└── tests/                   run_tests.py (pytest gerekmez) · test_model / test_attention / test_data / test_train
```

## Model seçenekleri (`configs` içinde `model` bölümü)

| Alan | Değerler | Varsayılan |
| :--- | :--- | :--- |
| `local_window` | yerel katman penceresi; `0`: bütün katmanlar genel | 128 |
| `global_layers` | genel katman sayısı (eşit aralıklı, son katman genel) | 3 |
| `attn_gate` | `""` · `head` · `elem` | `""` |
| `private_neurons` | `1`: kelimeye özel nöron | 0 |
| `private_vocab` | özel satırı olan kelimeler (gerisi tek ortak satır) | `assets/ss_vocab_used.json` |
| `local_sentence` | yerel katman kendi cümlesine bakar (`data.layout story` gerek) | `false` |

`train.private_opt`: özel tabloların optimizer'ı — `rms` (momentsiz, satır başına tek ölçek) · `adam`.

## Veri setleri

| Ad | Kaynak (Drive) |
| :--- | :--- |
| `ss` | SimpleStories, GPT-2 BPE (`simplestories/gpt2/{train,valid}.npy`) |
| `fineweb_s000` | FineWeb-Edu ilk parça, GPT-2 BPE; doğrulama aynı parçadan ayrılmış |
| `synthetic` | bellekte Markov zinciri (testler) |

## Hızlı başlangıç

```bash
python tests/run_tests.py                                              # testler (CPU; CUDA testleri GPU'da)
python scripts/train.py --config configs/base/cpu_tiny.json            # yerel duman testi
python scripts/train.py --config configs/ss/ss_d512_tf_yg_ozel.json    # (GPU) SS, kelimeye özel nöron, 1 epok
python scripts/summarize_runs.py                                       # sonuç tablosu
```

## Çıktılar (`runs/<koşu>/`)

`config.json` · `metrics.jsonl` (train / eval / diag kayıtları) · `checkpoint.pt` · `model.pt` · `results.json` · `samples.json`
