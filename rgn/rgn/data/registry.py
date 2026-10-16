"""Veri seti kaydı — hepsi GPT-2 BPE token akışı (uint16), belgeler EOS (50256) ile ayrılır.

Yollar Drive köküne göredir (rgn/paths.py: Windows G:/Drive'ım, Colab /content/drive/MyDrive).

    ss           SimpleStories (simplestories/gpt2/{train,valid}.npy; 605,6M / 6,1M token); cümle / hikâye sınırları
                 v2/simplestories_gpt2 (data.layout story)
    fineweb_s000 FineWeb-Edu shard_000, v2 ayrımı (v2/fineweb_edu_s000/gpt2/{train,valid}.npy; 737,0M / 15,1M token):
                 doğrulama ve sınav aynı parçadan (belge indeksi % 50 == 0), eğitimle kesişim yok
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
    "synthetic": DatasetSpec("synthetic", "Rastgele token (testler)", train=[], valid=[], fmt="synthetic",
                             bytes_per_token=4.0),
}


def get_spec(name):
    assert name in DATASETS, "bilinmeyen veri seti %r; seçenekler: %s" % (name, sorted(DATASETS))
    return DATASETS[name]
