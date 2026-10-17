"""Veri seti kaydı — hepsi GPT-2 BPE token akışı (uint16), belgeler EOS (50256) ile ayrılır.

Yollar Drive köküne göredir (rgn/paths.py: Windows G:/Drive'ım, Colab /content/drive/MyDrive).

    ss           SimpleStories (simplestories/gpt2/{train,valid}.npy; 605,6M / 6,1M token); cümle / hikâye sınırları
                 v2/simplestories_gpt2 (data.layout story)
    fineweb_s000 FineWeb-Edu shard_000, v2 ayrımı (v2/fineweb_edu_s000/gpt2/{train,valid}.npy; 737,0M / 15,1M token):
                 doğrulama ve sınav aynı parçadan (belge indeksi % 50 == 0), eğitimle kesişim yok
    fineweb_s000_s002  eğitim FineWeb-Edu s000 + v2/fineweb_edu_parts/s001, s002 (2,24B token); doğrulama s000'ınki (aynı
                 sınav) — yeni veri ↔ tekrar kıyası (17 Ekim)
    synthetic    testler için rastgele token (dosya yok)

bytes_per_token: doğrulama kümesinin UTF-8 bayt / token oranı; bits-per-byte (bpb) = kayıp / ln2 / bytes_per_token.
"""
from dataclasses import dataclass

EOS_ID = 50256
GPT2_VOCAB = 50257


@dataclass
class DatasetSpec:
    name: str
    description: str
    train: list                          # Drive köküne göre dosya yolları
    valid: list
    fmt: str = "npy"                     # npy | synthetic
    bytes_per_token: float = 0.0


DATASETS = {
    "ss": DatasetSpec(
        "ss", "SimpleStories (GPT-2 BPE) — kısa, sade hikâyeler; hızlı mimari karşılaştırma",
        train=["simplestories/gpt2/train.npy"], valid=["simplestories/gpt2/valid.npy"],
        bytes_per_token=26_430_797 / 6_116_459),           # valid_bytes / valid_tokens
    "fineweb_s000": DatasetSpec(
        "fineweb_s000", "FineWeb-Edu shard_000 (GPT-2 BPE), v2 ayrımı — genel web metni",
        train=["v2/fineweb_edu_s000/gpt2/train.npy"], valid=["v2/fineweb_edu_s000/gpt2/valid.npy"],
        bytes_per_token=69_744_540 / 15_073_888),          # valid_bytes.npy toplamı / valid token
    "fineweb_s000_s002": DatasetSpec(
        "fineweb_s000_s002", "FineWeb-Edu shard_000 + 001 + 002 eğitim (2,24B token), doğrulama yine shard_000 (yeni veri ↔ tekrar)",
        train=["v2/fineweb_edu_s000/gpt2/train.npy", "v2/fineweb_edu_parts/s001/gpt2/train.npy",
               "v2/fineweb_edu_parts/s002/gpt2/train.npy"],
        valid=["v2/fineweb_edu_s000/gpt2/valid.npy"],
        bytes_per_token=69_744_540 / 15_073_888),
    "synthetic": DatasetSpec("synthetic", "Rastgele token (testler)", train=[], valid=[], fmt="synthetic",
                             bytes_per_token=4.0),
}


def get_spec(name):
    assert name in DATASETS, "bilinmeyen veri seti %r; seçenekler: %s" % (name, sorted(DATASETS))
    return DATASETS[name]
