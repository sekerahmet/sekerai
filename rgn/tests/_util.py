"""Testler için ortak yardımcılar."""
import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgn.config import ModelConfig  # noqa: E402


class Skip(Exception):
    pass


def tiny(**kw):
    """Küçük YYG model yapılandırması (CPU testleri): 3 katman YYG, pencere 8."""
    base = dict(vocab_size=97, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, max_len=128, local_window=8,
                global_layers=1)
    base.update(kw)
    return ModelConfig(**base)


@contextlib.contextmanager
def private_vocab(tokens):
    """Geçici özel kelime listesi (ROOT altında) -> göreli yol."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=ROOT) as f:
        json.dump({"tokens": list(tokens)}, f)
    try:
        yield Path(f.name).name
    finally:
        Path(f.name).unlink()


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise Skip("CUDA yok")
