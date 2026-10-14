"""Checkpoint: atomik kaydetme (.part üzerinden) ve sürdürme."""
from pathlib import Path

import torch


def save(path, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(state, tmp)
    tmp.replace(path)


def load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
