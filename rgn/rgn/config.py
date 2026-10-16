"""Yapılandırma: model / veri / eğitim alanları ve JSON deney dosyaları.

Deney dosyası (configs/**.json) üç bölümden oluşur; "extends" ile bir taban dosyayı devralır (iç içe birleştirme):

    {"extends": "ss_d512_tf_yg.json", "name": "ss_d512_tf_yg_ozel", "model": {"private_neurons": 1}}

Komut satırından tek alan ezilebilir:  --set train.lr=1e-3 --set model.attn_gate=head

Model: softmax dikkat + RoPE + SwiGLU, yerel / genel (YYG) katman düzeni, isteğe bağlı kapı ve kelimeye özel nöron.
Kaldırılan deneyler (RGN çekim dikkati, rasyonel MLP, önem / odak, konum özeti, gözlemci, rota...) git etiketi
`arsiv/rgn`'de; eski config.json'larda görünen alanları REMOVED tablosu karşılar.
"""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

OPTIMIZERS = ("adamw", "muon", "normuon")
SCHEDULES = ("wsd", "cosine", "constant")


@dataclass
class ModelConfig:
    vocab_size: int = 50304            # GPT-2 BPE (50.257) -> 64'ün katı
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8                # < n_heads: GQA
    d_ff: int = 0                      # 0: SwiGLU 8d/3, 64'ün katı
    rope_base: float = 10000.0
    max_len: int = 4096                # RoPE tablosu boyu
    local_window: int = 128            # yerel (Y) katman yalnız son local_window token'a bakar; 0: bütün katmanlar genel
    global_layers: int = 3             # genel (G) katman sayısı; eşit aralıklı, son katman G (8 katmanda YYGYGYYG)
    attn_gate: str = ""                # dikkat çıkış kapısı: "" | "head" (head başına) | "elem" (her çıktı); 2σ_a ∈ (0, 2)
    private_neurons: int = 0           # 1: her katmanda her kelimenin kendi nöronu (kelimeye özel nöron); 0: kapalı
    private_vocab: str = "assets/ss_vocab_used.json"   # özel satırı olan kelimeler (gerisi tek ortak "diğer" satırı)
    local_sentence: bool = False       # Y katmanı pencere yerine kendi cümlesi + hikâye başı (data.layout story gerek)
    shared_mlp: bool = True            # False: ortak MLP yok (katman = dikkat [+ kelimeye özel nöron])
    mlp_bias: bool = True
    init_std: float = 0.02

    def layer_kinds(self):
        """Katman dizilimi: 'Y' yerel, 'G' genel (eşit aralık, son katman G)."""
        L = self.n_layers
        if not self.local_window:
            return "G" * L
        at = {round((i + 1) * L / self.global_layers) - 1 for i in range(self.global_layers)}
        return "".join("G" if l in at else "Y" for l in range(L))

    def ff_dim(self):
        """SwiGLU genişliği: d_ff ya da 8d/3 (64'ün katı)."""
        return self.d_ff or max(64, round(8 * self.d_model / 3 / 64) * 64)

    def validate(self):
        assert self.attn_gate in ("", "head", "elem"), self.attn_gate
        assert self.private_neurons in (0, 1), "private_neurons: 0 ya da 1 (çekirdek satırı tek nöron)"
        assert self.local_window >= 0 and 0 <= self.global_layers <= self.n_layers
        assert not self.local_sentence or self.local_window, "local_sentence: yerel / genel düzen gerek"
        assert self.local_window or self.global_layers, "yerel pencere yoksa bütün katmanlar genel"
        assert self.d_model % self.n_heads == 0 and self.n_heads % self.n_kv_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0, "RoPE çift head boyutu ister"


@dataclass
class DataConfig:
    dataset: str = "ss"                # rgn/data/registry.py: ss | fineweb_s000 | synthetic
    seq_len: int = 512
    local_copy: bool = True            # Colab'da token dosyalarını önce yerel diske kopyala
    eval_tokens: int = 4_194_304       # ara değerlendirme (doğrulama akışının başından sabit pencereler)
    final_eval_tokens: int = 0         # 0: tüm doğrulama kümesi
    layout: str = "stream"             # stream: rastgele pencere | story: hikâye başından, cümle sonunda kesilen satırlar
    story_root: str = "v2/simplestories_gpt2"   # layout story: cümle / hikâye sınırları (Drive köküne göre)


@dataclass
class TrainConfig:
    batch_size: int = 64               # adım başına pencere (mikro adımların toplamı)
    micro_batch: int = 0               # 0: batch_size (tek geçiş)
    max_steps: int = 0                 # 0: epochs'tan hesaplanır
    stop_step: int = 0                 # > 0: bu adımda checkpoint kaydedip dur (takvim max_steps'e göre sürer; --resume)
    epochs: float = 1.0
    optimizer: str = "muon"            # matrisler Muon / NorMuon, geri kalan AdamW
    lr: float = 2e-3                   # Muon lr (AdamW RMS ölçeğinde: 0.2·sqrt(max(m, n)))
    adam_lr: float = 0.0               # 0: lr ile aynı (embedding, norm, bias, kapı, özel tablolar)
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
    diag_every: int = 1000             # dikkat entropisi, MLP doyması
    ckpt_minutes: float = 10.0
    private_opt: str = "rms"           # kelimeye özel tablolar (satır seyrek): rms (momentsiz, satır başına tek ölçek) | adam
    sample_tokens: int = 64            # bitişte örnek üretim uzunluğu (0: kapalı)
    probe_tokens: int = 48             # her değerlendirmede 2 istemle greedy devam (nitel kontrol; 0: kapalı)
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


# Kaldırılan alanlar (arsiv/rgn): eski config.json'larda görünürse değer mevcut tasarımla uyumluysa yok sayılır.
# Değer: kabul edilen değerler kümesi; ANY: değer ne olursa olsun yok sayılır (kaldırılan özelliğin yalnız ayarı).
ANY = object()
REMOVED = {
    "model": {
        "attn": {"softmax"}, "pos": {"rope"}, "mlp": {"swiglu"}, "g_layout": {"interleave"},
        "attn_importance": {False}, "attn_focus": {False}, "attn_locality": {False}, "observer_layer": {0},
        "mlp_only_layers": {0}, "mlp_vocab_neurons": {0}, "mlp_vocab_assign": {""}, "route_heads": {0},
        "route_neurons": {0}, "private_layers": {""}, "global_pos": {""}, "ff_ratio": {()}, "sharpness_range_g": {()},
        **{k: ANY for k in ("attn_impl", "dense_max_len", "rgn_fp32", "rgn_av_bf16", "rgn_qk_prec", "power",
                            "sharpness", "sharpness_learn", "sharpness_range", "sharpness_init", "sharpness_int",
                            "gamma_range", "gamma_min", "learn_gamma", "beta_range", "importance_max", "focus_max",
                            "locality_tau_local", "locality_tau_global", "observer_vocab", "observer_aux_weight")},
    },
    "data": {"max_train_files": ANY},
    "train": {"profile_after": ANY},
}


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


def _build(section, section_cls, values):
    names = {f.name for f in dataclasses.fields(section_cls)}
    kw = {}
    for k, v in values.items():
        v = tuple(v) if isinstance(v, list) else v
        if k in names:
            kw[k] = v
            continue
        ok = REMOVED.get(section, {}).get(k)
        assert ok is not None, "%s: bilinmeyen alan %r" % (section_cls.__name__, k)
        assert ok is ANY or v in ok, "%s.%s = %r: kaldırılan bir özellik (git etiketi arsiv/rgn)" % (section, k, v)
    return section_cls(**kw)


def from_dict(raw):
    raw = dict(raw)
    cfg = RunConfig(name=raw.pop("name", "run"), notes=raw.pop("notes", ""),
                    model=_build("model", ModelConfig, raw.pop("model", {})),
                    data=_build("data", DataConfig, raw.pop("data", {})),
                    train=_build("train", TrainConfig, raw.pop("train", {})))
    assert not raw, "bilinmeyen bölüm(ler): %s" % sorted(raw)
    cfg.model.validate()
    assert cfg.train.optimizer in OPTIMIZERS and cfg.train.schedule in SCHEDULES
    assert cfg.data.layout in ("stream", "story"), cfg.data.layout
    assert cfg.train.private_opt in ("rms", "adam"), cfg.train.private_opt
    assert not cfg.model.local_sentence or cfg.data.layout == "story", "local_sentence: data.layout story gerek"
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
