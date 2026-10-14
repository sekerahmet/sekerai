"""Veri: pencere biçimi, kaydırma, sabit doğrulama pencereleri, kayıt ve yapılandırma dosyaları."""
from pathlib import Path

import numpy as np

from _util import ROOT
from rgn.config import DataConfig, load_config
from rgn.data import DATASETS, TokenStream, load_streams


def test_random_batch_shift():
    a = np.arange(1000, dtype=np.uint16)
    s = TokenStream([a])
    x, y = s.random_batch(4, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 16)
    assert (y[:, :-1] == x[:, 1:]).all() and (y - x == 1).all()


def test_random_batch_file_boundary():
    """Dosya tam seq_len + 1 token ise tek geçerli pencere seçilir; kısa dosya hiç seçilmez."""
    s = TokenStream([np.arange(17, dtype=np.uint16), np.arange(5, dtype=np.uint16)])
    x, y = s.random_batch(8, 16, np.random.default_rng(0))
    assert (x[:, 0] == 0).all() and (y[:, -1] == 16).all()


def test_rgn_rejects_sdpa():
    from rgn.config import ModelConfig
    try:
        ModelConfig(attn="rgn", attn_impl="sdpa").validate()
    except AssertionError:
        return
    raise AssertionError("rgn + sdpa kabul edilmemeliydi")


def test_fixed_windows_deterministic():
    s = TokenStream([np.arange(1000, dtype=np.uint16)])
    w1, w2 = s.fixed_windows(64, 600), s.fixed_windows(64, 600)
    assert w1.shape == (9, 65) and (w1 == w2).all() and (w1[1, 0] == 64)


def test_synthetic_streams():
    train, valid, bpt = load_streams(DataConfig(dataset="synthetic", seq_len=32))
    assert train.total > 0 and valid.total > 0 and bpt > 0


def test_registry_specs():
    for name, spec in DATASETS.items():
        if spec.fmt != "synthetic":
            assert spec.train and spec.valid and spec.fmt in ("npy", "bin"), name


def test_all_configs_load():
    files = sorted((ROOT / "configs").rglob("*.json"))
    assert files
    for f in files:
        cfg = load_config(f)
        cfg.model.validate()
    names = [load_config(f).name for f in files if not f.parent.name == "base"]
    assert len(names) == len(set(names)), "deney adları tekil olmalı"
