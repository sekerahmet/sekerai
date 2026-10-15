"""Yapılandırma: model / veri / eğitim alanları ve JSON deney dosyaları.

Deney dosyası (configs/**.json) üç bölümden oluşur; "extends" ile bir taban dosyayı devralır (iç içe birleştirme):

    {"extends": "../base/ss_d512.json", "name": "ss_d512_rgn",
     "model": {"attn": "rgn", "mlp": "rational"}, "data": {...}, "train": {...}}

Komut satırından tek alan ezilebilir:  --set train.lr=1e-3 --set model.pos=alibi
"""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

ATTN = ("rgn", "softmax")
ATTN_IMPL = ("auto", "triton", "dense", "flex", "sdpa")
POS = ("rope", "alibi", "additive", "rope+alibi", "none")
MLP = ("rational", "rglu", "glu", "swiglu")
OPTIMIZERS = ("adamw", "muon", "normuon")
SCHEDULES = ("wsd", "cosine", "constant")


@dataclass
class ModelConfig:
    vocab_size: int = 50304            # GPT-2 BPE (50.257) -> 64'ün katı
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8                # < n_heads: GQA
    d_ff: int = 0                      # 0: otomatik (rational 4D; glu / swiglu 8D/3, 64'ün katı)
    attn: str = "rgn"                  # rgn: ters kare çekim | softmax: karşılaştırma tabanı
    attn_impl: str = "auto"            # auto: rgn -> CUDA + Triton + pos rope / none + power 1 ise triton (Flash-RGN); değilse
                                       # N <= dense_max_len ise dense, değilse flex; softmax -> sdpa
    dense_max_len: int = 1024          # auto'da dense'in üst sınırı (N×N bellek); ölçüldü: SS d512 N 1024'te dense 299 ms/adım,
                                       # flex 3800 ms/adım (flex, öğrenilebilir γ/β gradyanını yavaş yoldan topluyor)
    rgn_fp32: bool = True              # rgn dikkatinde q, k, v fp32 (yakın eşleşmede 2 - 2 q·k iptal hatası; doküman 3.5)
    rgn_av_bf16: bool = False          # dense yolda w·V çarpımı bf16 (CUDA; q·kᵀ fp32 kalır) — hız seçeneği
    rgn_qk_prec: str = "tf32x3"        # triton yolunda q·kᵀ: tf32x3 (fp32'ye yakın) | tf32 (tam adım ~%10 hızlı) | ieee
    pos: str = "rope"                  # rope (varsayılan, kullanıcı 15 Ekim) | additive: payda + β(i-j) | alibi: payda × (1 + β(i-j))
                                       # | rope+alibi | none
    rope_base: float = 10000.0
    max_len: int = 4096                # RoPE tablosu boyu
    power: int = 1                     # uzaklık üssü p: 1/(1 + γ d^p)
    sharpness: int = 1                 # keskinlik üssü k: (1 + γ d / k)^(-k); k = 1 ters kare, k büyüdükçe e^(-γd)
    sharpness_learn: bool = False      # head başına öğrenilen k (sharpness_range içinde, başlangıç sharpness_init)
    sharpness_range: tuple = (1.0, 8.0)   # 8: fp32'de alt taşma güvenliği ((1 + γd/k)^-k, γ ~ 1e5'e kadar)
    sharpness_init: float = 4.0
    sharpness_int: bool = False        # öğrenilen k ileri geçişte en yakın tamsayı (kuvvet zinciri, log / exp yok); gradyan
                                       # sürekli k'ye düz geçer (straight-through)
    sharpness_range_g: tuple = ()      # genel (G) katmanlarda k aralığı (yerel katmanlar sharpness_range).  Genel katmanda aday
                                       # çok: kalın kuyruğun toplam payı aday sayısıyla büyür -> daha yüksek tavan
    gamma_range: tuple = (4.0, 512.0)  # γ başlangıcı, head başına geometrik
    gamma_min: float = 1.0             # γ = γ_min + γ̂²
    learn_gamma: bool = True
    beta_range: tuple = (1.0, 1.0 / 65536)  # β başlangıcı (additive / alibi), head başına geometrik (azalan)
    local_window: int = 0              # > 0: yerel (Y) katmanlar yalnız son local_window token'a bakar (j > i - W); genel (G)
                                       # katmanlar tümüne.  0: bütün katmanlar genel (eski düzen)
    global_layers: int = 0             # local_window > 0 iken genel katman sayısı
    g_layout: str = "interleave"       # interleave: eşit aralık, son katman G (~Y-Y-G) | top: son N | elle "2,5,7"
    global_pos: str = ""               # genel katmanlarda konum ("" = pos); "none": konumsuz — ağırlık yalnız anlam uzaklığı
                                       # (saf çekim; sıra bilgisi yerel katmanlardan ve causal maskeden)
    ff_ratio: tuple = ()               # (Y, G): MLP / dikkat (4 d²) parametre oranı, katman türüne göre.  Örn. (1, 3): yerel
                                       # katman dizimi taşır (dar MLP), genel katman anlam çekimiyle toplar (geniş MLP).  (): d_ff
    # Softmax dikkatinde q / k / v düzeyinde eklemeler (kullanıcı, 15 Ekim: "softmax kalsın, q k v değişsin").  Hepsi
    # cebirsel ve sıfır başlatmalı: başlangıçta model transformer tabanıyla birebir aynı.
    #   pay(i, j) = softmax_j( odak_i · q_i·k_j / √dh + önem_j ),   çıktı_i = kapı_i · Σ_j pay(i, j) v_j
    attn_importance: bool = False      # önem (kütle): anahtar başına, sorgudan bağımsız taban puan ∈ (-importance_max, +)
    importance_max: float = 4.0
    attn_focus: bool = False           # odak: sorgu başına keskinlik çarpanı ∈ (0, focus_max), başlangıç 1
    focus_max: float = 4.0
    attn_evidence: bool = False        # kanıt: head başına iki sayı MLP kapı girdisine (softmax payları normalize ettiği için
                                       # atılan bilgi): güç = ln((1/n) Σ_j e^puan); odaklanma = güç − ortalama puan ≥ 0
                                       # (kaydırmaya duyarsız: yalnız puan farkları — softmax'ın kendi değişmezliğiyle uyumlu)
    attn_evidence_grad: bool = False   # güç teriminin (softmax paydası) gradyanı q / k'ye akar mı (flex desteği gerekir)
    attn_gate: str = ""                # kapı: "" | "head" (head başına 1 sayı) | "elem" (her çıktı sayısı); 2σ_a ∈ (0, 2), başlangıç 1
    mlp: str = "rational"              # rational: W2 u/sqrt(1+u²) | rglu: W2 [G/sqrt(1+G²) ⊙ U] | glu: cebirsel GLU | swiglu: taban
    mlp_bias: bool = True
    init_std: float = 0.02

    def layer_kinds(self):
        """Katman dizilimi: 'Y' yerel (son local_window token), 'G' genel (tam causal)."""
        L = self.n_layers
        if not self.local_window:
            return "G" * L
        if self.g_layout == "interleave":
            at = {round((i + 1) * L / self.global_layers) - 1 for i in range(self.global_layers)}
        elif self.g_layout == "top":
            at = set(range(L - self.global_layers, L))
        else:
            at = {int(v) for v in self.g_layout.split(",") if v.strip()}
        return "".join("G" if l in at else "Y" for l in range(L))

    def ff_dim(self, kind=None):
        """MLP genişliği.  kind ('Y' / 'G') ve ff_ratio verilmişse: oran × 4d² parametre (kapılı MLP 3 matris, rasyonel 2),
        64'ün katı."""
        if kind and self.ff_ratio:
            r = self.ff_ratio[0 if kind == "Y" else 1]
            mats = 2 if self.mlp == "rational" else 3
            return max(64, round(r * 4 * self.d_model / mats / 64) * 64)
        if self.d_ff:
            return self.d_ff
        if self.mlp == "rational":
            return 4 * self.d_model
        return max(64, round(8 * self.d_model / 3 / 64) * 64)

    def validate(self):
        assert self.attn in ATTN, self.attn
        assert self.attn_impl in ATTN_IMPL, self.attn_impl
        assert self.rgn_qk_prec in ("tf32x3", "tf32", "ieee"), self.rgn_qk_prec
        assert self.attn_gate in ("", "head", "elem"), self.attn_gate
        assert not (self.attn_evidence and (self.attn != "softmax" or self.attn_importance)), \
            "kanıt: softmax dikkatinde, önem olmadan (ek boyut hızlı çekirdeği bozar)"
        assert not ((self.attn_importance or self.attn_focus or self.attn_gate) and self.attn != "softmax"), \
            "önem / odak / kapı: şimdilik softmax dikkatinde"
        assert self.local_window >= 0 and 0 <= self.global_layers <= self.n_layers
        assert not self.global_pos or self.global_pos in POS, self.global_pos
        assert not self.ff_ratio or (len(self.ff_ratio) == 2 and self.local_window), "ff_ratio: (Y, G), yerel pencere gerek"
        if self.local_window:
            kinds = self.layer_kinds()
            assert len(kinds) == self.n_layers and kinds.count("G") == self.global_layers, \
                "g_layout %s: %d genel katman bekleniyor (%s)" % (self.g_layout, self.global_layers, kinds)
            assert self.attn_impl != "flex", "yerel pencere flex yolunda yok (triton / dense / sdpa)"
        assert self.pos in POS, self.pos
        assert self.mlp in MLP, self.mlp
        assert self.d_model % self.n_heads == 0 and self.n_heads % self.n_kv_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0, "RoPE çift head boyutu ister"
        assert self.power in (1, 2)
        assert 1 <= self.sharpness <= 16, "keskinlik üssü 1..16"
        for lo, hi in (self.sharpness_range,) + ((tuple(self.sharpness_range_g),) if self.sharpness_range_g else ()):
            assert 1.0 <= lo < self.sharpness_init < hi <= 16.0, "sharpness_range(_g) / sharpness_init"
        assert not self.sharpness_range_g or self.local_window, "sharpness_range_g: yerel / genel düzen gerek"
        assert not (self.attn == "softmax" and self.pos not in ("rope", "none")), "softmax tabanı: pos rope / none"
        assert not (self.attn == "softmax" and self.attn_impl in ("dense", "flex", "triton")), "softmax tabanı sdpa ile"
        assert not (self.attn_impl == "triton" and (self.pos not in ("rope", "none") or self.power != 1)), \
            "triton çekirdeği: pos rope / none, power 1"
        assert not (self.attn == "rgn" and self.attn_impl == "sdpa"), "rgn dikkati sdpa ile çalışmaz (triton / dense / flex)"


@dataclass
class DataConfig:
    dataset: str = "ss"                # rgn/data/registry.py: ss | tinystories | fineweb | synthetic
    seq_len: int = 1024
    max_train_files: int = 0           # 0: hepsi (fineweb'de parça sayısını sınırlamak için)
    local_copy: bool = True            # Colab'da token dosyalarını önce yerel diske kopyala
    eval_tokens: int = 4_194_304       # ara değerlendirme (doğrulama akışının başından sabit pencereler)
    final_eval_tokens: int = 0         # 0: tüm doğrulama kümesi


@dataclass
class TrainConfig:
    batch_size: int = 64               # adım başına pencere (mikro adımların toplamı)
    micro_batch: int = 0               # 0: batch_size (tek geçiş)
    max_steps: int = 0                 # 0: epochs'tan hesaplanır
    stop_step: int = 0                 # > 0: bu adımda checkpoint kaydedip dur (takvim max_steps'e göre sürer; --resume)
    epochs: float = 1.0
    optimizer: str = "muon"            # matrisler Muon / NorMuon, geri kalan AdamW
    lr: float = 2e-3                   # Muon lr (AdamW RMS ölçeğinde: 0.2·sqrt(max(m, n)))
    adam_lr: float = 0.0               # 0: lr ile aynı (embedding, norm, bias, γ̂, λ̂)
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    momentum: float = 0.95
    schedule: str = "wsd"              # wsd: ısınma + sabit + son kısımda doğrusal düşüş
    warmup: float = 0.01               # adımların oranı
    decay: float = 0.2                 # wsd: son oran
    min_lr_frac: float = 0.0
    grad_clip: float = 1.0
    dtype: str = "bf16"                # bf16 | fp32 (CPU'da her zaman fp32)
    compile: bool = True
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1000
    diag_every: int = 1000             # γ, |u| > 3 oranı, dikkat entropisi
    ckpt_minutes: float = 10.0
    sample_tokens: int = 64            # bitişte örnek üretim uzunluğu (0: kapalı)
    probe_tokens: int = 48             # her değerlendirmede 2 istemle greedy devam (nitel kontrol; 0: kapalı)
    profile_after: bool = True         # bitişte hız profili (runs/<koşu>/speed.json)
    loss_checkpoint: bool = False      # kayıpta logit'leri geri geçişte yeniden hesapla (bellek az, LM head 4 geçiş yerine 3)


@dataclass
class RunConfig:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    def to_dict(self):
        return dataclasses.asdict(self)


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(_read(path.parent / parent), raw)
    return raw


def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _build(section_cls, values):
    names = {f.name: f for f in dataclasses.fields(section_cls)}
    unknown = set(values) - set(names)
    assert not unknown, "%s: bilinmeyen alan(lar) %s" % (section_cls.__name__, sorted(unknown))
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return section_cls(**kw)


def from_dict(raw):
    raw = dict(raw)
    cfg = RunConfig(name=raw.pop("name", "run"), notes=raw.pop("notes", ""),
                    model=_build(ModelConfig, raw.pop("model", {})),
                    data=_build(DataConfig, raw.pop("data", {})),
                    train=_build(TrainConfig, raw.pop("train", {})))
    assert not raw, "bilinmeyen bölüm(ler): %s" % sorted(raw)
    cfg.model.validate()
    assert cfg.train.optimizer in OPTIMIZERS and cfg.train.schedule in SCHEDULES
    return cfg


def load_config(path=None, overrides=()):
    """JSON deney dosyası (+ "extends" zinciri) ve "bölüm.alan=değer" ezmeleri -> RunConfig."""
    raw = _read(path) if path else {}
    for item in overrides:
        key, _, value = item.partition("=")
        parts = key.split(".")
        node = raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return from_dict(raw)
