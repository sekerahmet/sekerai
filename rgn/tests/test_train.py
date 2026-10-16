"""Eğitim: kısa CPU eğitimi kaybı düşürür ve dosyaları yazar; kesilip sürdürülen koşu = kesintisiz koşu (özel nöronlu
model dahil); RowAdam satır başına bias düzeltmesi."""
import json
import tempfile
from pathlib import Path

import torch

from _util import ROOT, private_vocab
from rgn.config import load_config
from rgn.train import train
from rgn.train.optim import RowAdam
from rgn.train.schedule import lr_factor

CPU_TINY = ROOT / "configs/base/cpu_tiny.json"


def test_schedule_shapes():
    assert lr_factor(0, 100, "wsd", 0.1, 0.2) == 0.1
    assert lr_factor(50, 100, "wsd", 0.1, 0.2) == 1.0
    assert lr_factor(99, 100, "wsd", 0.1, 0.2) < 0.1
    assert abs(lr_factor(99, 100, "cosine", 0.1, 0.2)) < 0.01


def test_cpu_training_reduces_loss():
    with private_vocab(range(1, 400)) as rel:
        for overrides in ([], ["model.attn_gate=head"], ["train.optimizer=adamw"],
                          ["model.private_neurons=1", "model.private_vocab=" + rel],
                          ["model.private_neurons=1", "model.private_vocab=" + rel, "model.shared_mlp=false"],
                          ["model.private_value=1", "model.private_vocab=" + rel]):
            cfg = load_config(CPU_TINY, overrides)
            with tempfile.TemporaryDirectory() as d:
                r = train(cfg, out_dir=d)
                out = Path(d)
                for f in ("config.json", "metrics.jsonl", "checkpoint.pt", "model.pt", "results.json"):
                    assert (out / f).exists(), f
                recs = [json.loads(l) for l in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                assert recs[0]["type"] == "start" and recs[0]["total_steps"] == cfg.train.max_steps
                first = next(r for r in recs if r["type"] == "train")
                assert "acc_bands" in next(r for r in recs if r["type"] == "eval")
                assert r["final"]["loss"] < first["loss"] - 0.5, (overrides, first["loss"], r["final"]["loss"])


def _resume_equal(extra):
    base = ["train.max_steps=30", "train.eval_every=10"] + extra
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        train(load_config(CPU_TINY, base), out_dir=a)
        assert train(load_config(CPU_TINY, base + ["train.stop_step=17"]), out_dir=b) is None
        assert (Path(b) / "stopped").exists()
        r = train(load_config(CPU_TINY, base), out_dir=b, resume=True)
        assert r["steps"] == 30 and not (Path(b) / "stopped").exists()
        wa = torch.load(Path(a) / "model.pt", weights_only=False)["model"]
        wb = torch.load(Path(b) / "model.pt", weights_only=False)["model"]
        for k in wa:
            assert torch.equal(wa[k], wb[k]), (extra, k)


def test_resume_equals_uninterrupted():
    """Kesilip sürdürülen koşu = kesintisiz koşu (ağırlıklar bit düzeyinde aynı): model + optimizer + takvim + veri sırası."""
    _resume_equal([])
    with private_vocab(range(1, 400)) as rel:
        for opt in ("rms", "adam"):
            _resume_equal(["model.private_neurons=1", "model.private_vocab=" + rel, "train.private_opt=" + opt])
        _resume_equal(["model.private_neurons=1", "model.private_value=1", "model.private_vocab=" + rel])


def test_row_adam_per_row_bias():
    """İlk kez görülen satırın ilk güncellemesi, genel adım sayısından bağımsız (satır başına bias düzeltmesi)."""
    for mode in ("rms", "adam"):
        p = torch.nn.Parameter(torch.zeros(4, 3))
        opt = RowAdam([dict(params=[p])], lr=0.1, mode=mode)
        g = torch.ones(1, 3)
        for _ in range(500):                                       # satır 0 çok güncellenir
            opt.set_rows(torch.tensor([0]), [g])
            opt.step()
        opt.set_rows(torch.tensor([3]), [g])                       # satır 3 ilk kez
        opt.step()
        assert torch.allclose(p.data[3], torch.full((3,), -0.1), atol=1e-4), (mode, p.data[3])
        assert (p.data[1:3] == 0).all()                            # dokunulmayan satırlar
