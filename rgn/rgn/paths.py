"""Ortam yolları — Windows (G:/Drive'ım) ve Colab (/content/drive/MyDrive) otomatik algılanır.

Ortam değişkenleriyle her biri ezilebilir:
    RGN_DRIVE  Drive kökü (veri setleri burada: simplestories/, v2/)
    RGN_CACHE  büyük token dosyalarının hızlı yerel kopyası (Colab'da /content/rgn_cache)
    RGN_RUNS   koşu çıktıları (Colab'da Drive/RGN/runs: oturum kapansa da kalır, yerelde de görünür)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DRIVE_CANDIDATES = ("/content/drive/MyDrive", "G:/Drive'ım", "G:/My Drive")


def in_colab():
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def drive_root():
    env = os.environ.get("RGN_DRIVE")
    if env:
        return Path(env)
    for c in _DRIVE_CANDIDATES:
        if Path(c).is_dir():
            return Path(c)
    raise FileNotFoundError("Drive kökü bulunamadı (%s); RGN_DRIVE ortam değişkenini ayarlayın" % ", ".join(_DRIVE_CANDIDATES))


def cache_dir():
    env = os.environ.get("RGN_CACHE")
    if env:
        return Path(env)
    return Path("/content/rgn_cache") if in_colab() else ROOT / "cache"


def runs_dir():
    env = os.environ.get("RGN_RUNS")
    if env:
        return Path(env)
    return drive_root() / "RGN" / "runs" if in_colab() else ROOT / "runs"
