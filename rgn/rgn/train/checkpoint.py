"""Checkpoint: atomik kaydetme (.part üzerinden), sürdürme ve analiz için koşu yükleme."""
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


def load_run(run, device="cpu"):
    """Koşu adı -> (model (eval), adım).  runs_dir()'de, yoksa Drive/RGN/runs'ta; checkpoint.pt, yoksa model.pt."""
    from rgn.config import from_dict
    from rgn.model import RGNModel
    from rgn.paths import drive_root, runs_dir
    d = next((p for p in (runs_dir() / run, drive_root() / "RGN" / "runs" / run) if p.is_dir()), None)
    assert d is not None, "koşu bulunamadı: %s" % run
    ck = load(d / "checkpoint.pt" if (d / "checkpoint.pt").exists() else d / "model.pt")
    m = RGNModel(from_dict(ck["config"]).model)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck.get("step")
