"""Eğitim duman testi: sentetik veride kısa CPU eğitimi kaybı düşürmeli, çıktı dosyaları yazılmalı, sürdürme çalışmalı."""
import json
import tempfile
from pathlib import Path

from _util import ROOT
from rgn.config import load_config
from rgn.train import train
from rgn.train.schedule import lr_factor


def test_schedule_shapes():
    assert lr_factor(0, 100, "wsd", 0.1, 0.2) == 0.1
    assert lr_factor(50, 100, "wsd", 0.1, 0.2) == 1.0
    assert lr_factor(99, 100, "wsd", 0.1, 0.2) < 0.1
    assert abs(lr_factor(99, 100, "cosine", 0.1, 0.2)) < 0.01


def test_cpu_training_reduces_loss():
    for overrides in ([], ["model.attn=softmax", "model.pos=rope", "model.mlp=swiglu"], ["train.optimizer=adamw"]):
        cfg = load_config(ROOT / "configs/base/cpu_tiny.json", overrides)
        with tempfile.TemporaryDirectory() as d:
            r = train(cfg, out_dir=d)
            out = Path(d)
            for f in ("config.json", "metrics.jsonl", "checkpoint.pt", "model.pt", "results.json"):
                assert (out / f).exists(), f
            first = json.loads((out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
            assert r["final"]["loss"] < first["loss"] - 0.5, (overrides, first["loss"], r["final"]["loss"])


def test_resume_equals_uninterrupted():
    """Kesilip sürdürülen koşu = kesintisiz koşu (ağırlıklar bit düzeyinde aynı): model + optimizer + takvim + veri sırası."""
    import torch
    base = ["train.max_steps=30", "train.eval_every=10", "train.profile_after=false"]
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        train(load_config(ROOT / "configs/base/cpu_tiny.json", base), out_dir=a)
        assert train(load_config(ROOT / "configs/base/cpu_tiny.json", base + ["train.stop_step=17"]), out_dir=b) is None
        assert (Path(b) / "stopped").exists()
        r = train(load_config(ROOT / "configs/base/cpu_tiny.json", base), out_dir=b, resume=True)
        assert r["steps"] == 30 and not (Path(b) / "stopped").exists()
        wa = torch.load(Path(a) / "model.pt", weights_only=False)["model"]
        wb = torch.load(Path(b) / "model.pt", weights_only=False)["model"]
        for k in wa:
            assert torch.equal(wa[k], wb[k]), k
