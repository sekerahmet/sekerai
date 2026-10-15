"""Veri seti kaydı — hepsi GPT-2 BPE token akışı (uint16), belgeler EOS (50256) ile ayrılır.

Yollar Drive köküne göredir (rgn/paths.py: Windows G:/Drive'ım, Colab /content/drive/MyDrive).

    ss           SimpleStories: hazır (simplestories/gpt2/{train,valid}.npy; 605,6M / 6,1M token)
    tinystories  TinyStoriesV2-GPT4: ham metinden hazırlanır -> tinystories/gpt2/{train,valid}.bin
                 (python scripts/prepare_data.py --dataset tinystories)
    fineweb      FineWeb-Edu sample-10BT: hazır parçalar (fineweb/gpt2/shard_000..013.bin, ~750M token / parça);
                 eğitim 000..012, doğrulama 013'ün ilk 10M token'ı
    fineweb_s000 FineWeb-Edu shard_000'ın v2 sürümü (v2/fineweb_edu_s000/gpt2/{train,valid}.npy; 737,0M / 15,1M token):
                 aynı kaynak akış (sha 0aa18c…), metin + eot, valid = belge indeksi % 50 == 0; cümle sınırları yanında
                 (train_sentence_offsets.npy, web profili, ≤ 128 token).  _c24 / _c32 sabit parçadır, cümle değil.
    synthetic    testler için rastgele token (dosya yok)

bytes_per_token: doğrulama kümesinin UTF-8 bayt / token oranı; bits-per-byte (bpb) = kayıp / ln2 / bytes_per_token.
"""
import json
from dataclasses import dataclass, field

from rgn.paths import drive_root

EOS_ID = 50256
GPT2_VOCAB = 50257


@dataclass
class DatasetSpec:
    name: str
    description: str
    train: list                          # Drive köküne göre dosya yolları
    valid: list
    fmt: str = "npy"                     # npy | bin (ham uint16)
    valid_max_tokens: int = 0            # 0: tüm doğrulama dosyası
    bytes_per_token: float = 0.0         # 0: meta dosyasından okunur
    meta: str = ""                       # bytes_per_token'ın okunacağı json (Drive köküne göre)
    prepare: str = ""                    # hazırlama gerekiyorsa prepare.py'deki fonksiyon adı
    extra: dict = field(default_factory=dict)

    def resolve_bytes_per_token(self):
        if self.bytes_per_token:
            return self.bytes_per_token
        if self.meta:
            m = json.loads((drive_root() / self.meta).read_text(encoding="utf-8"))
            return m["valid_bytes"] / m["valid_tokens"]
        return 0.0


DATASETS = {
    "ss": DatasetSpec(
        "ss", "SimpleStories (GPT-2 BPE) — kısa, sade hikâyeler; hızlı mimari karşılaştırma",
        train=["simplestories/gpt2/train.npy"], valid=["simplestories/gpt2/valid.npy"], fmt="npy",
        bytes_per_token=26_430_797 / 6_116_459),           # fingerprint.json: valid_bytes / valid_tokens
    "tinystories": DatasetSpec(
        "tinystories", "TinyStoriesV2-GPT4 (GPT-2 BPE) — çocuk hikâyeleri; SS ile çapraz kontrol",
        train=["tinystories/gpt2/train.bin"], valid=["tinystories/gpt2/valid.bin"], fmt="bin",
        meta="tinystories/gpt2/meta.json", prepare="tinystories"),
    "fineweb": DatasetSpec(
        "fineweb", "FineWeb-Edu sample-10BT (GPT-2 BPE) — genel web metni; ölçek testi",
        train=["fineweb/gpt2/shard_%03d.bin" % i for i in range(13)], valid=["fineweb/gpt2/shard_013.bin"], fmt="bin",
        valid_max_tokens=10_000_000, bytes_per_token=3_469_725_588 / 752_097_492),   # shard_000.json oranı
    "fineweb_s000": DatasetSpec(
        "fineweb_s000", "FineWeb-Edu shard_000, v2 ayrımı (her 50. belge doğrulama) — eski v2 FineWeb koşularıyla kıyaslanabilir",
        train=["v2/fineweb_edu_s000/gpt2/train.npy"], valid=["v2/fineweb_edu_s000/gpt2/valid.npy"], fmt="npy",
        bytes_per_token=69_744_540 / 15_073_888),          # valid_bytes.npy toplamı / valid token (SS ile aynı tanım)
    "synthetic": DatasetSpec("synthetic", "Rastgele token (testler)", train=[], valid=[], fmt="synthetic",
                             bytes_per_token=4.0),
}


def get_spec(name):
    assert name in DATASETS, "bilinmeyen veri seti %r; seçenekler: %s" % (name, sorted(DATASETS))
    return DATASETS[name]
