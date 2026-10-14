"""Testler için ortak yardımcılar."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgn.config import ModelConfig  # noqa: E402


class Skip(Exception):
    pass


def tiny(**kw):
    """Küçük model yapılandırması (CPU testleri)."""
    base = dict(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_len=128, attn_impl="dense")
    base.update(kw)
    return ModelConfig(**base)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise Skip("CUDA yok")
