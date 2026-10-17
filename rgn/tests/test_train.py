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
                          ["model.private_neurons=1", "model.private_groups=4", "model.private_vocab=" + rel],
                          ["model.y_windows=[8, 16]"],
                          ["model.private_neurons=1", "model.private_vocab=" + rel, "model.shared_mlp=false"],
                          ["model.private_value=1", "model.private_vocab=" + rel],
                          ["model.ple_dim=16", "model.private_vocab=" + rel]):
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
        assert torch.load(Path(b) / "model.pt", weights_only=False)["step"] == 17   # durdurmada hafif ağırlık dosyası
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
        _resume_equal(["model.private_neurons=2", "model.private_vocab=" + rel])
        _resume_equal(["model.private_neurons=1", "model.private_vocab=" + rel, "train.private_lr=0.008",
                       "train.private_row_decay=67"])


def test_row_lr_mult():
    """Kelime başına lr çarpanı: satırın adımı lr · çarpan; çarpan sayıdan sqrt(ref / sayı), [1, max]."""
    p = torch.nn.Parameter(torch.zeros(3, 2))
    opt = RowAdam([dict(params=[p])], lr=0.1, mode="rms", row_mult=torch.tensor([1.0, 2.0, 4.0]))
    opt.set_rows(torch.tensor([0, 2]), [torch.ones(2, 2)])
    opt.step()
    assert torch.allclose(p.data[:, 0], torch.tensor([-0.1, 0.0, -0.4]), atol=1e-5)
    from types import SimpleNamespace
    from rgn.train.optim import row_lr_mult
    with private_vocab([3, 5, 9]) as rel:
        import json
        f = ROOT / rel
        d = json.loads(f.read_text(encoding="utf-8"))
        d["counts"] = [10000, 100, 1]
        f.write_text(json.dumps(d), encoding="utf-8")
        m = row_lr_mult(SimpleNamespace(private_vocab=rel), SimpleNamespace(private_lr_ref=400.0, private_lr_max=4.0))
        assert torch.allclose(m, torch.tensor([1.0, 1.0, 2.0, 4.0]))


def test_row_own_schedule():
    """row_decay: satırın adımı kendi güncelleme sayısıyla küçülür (lr·sqrt(n0/(n+n0))); dokunulmayan satır etkilenmez."""
    p = torch.nn.Parameter(torch.zeros(3, 2))
    opt = RowAdam([dict(params=[p])], lr=0.1, mode="rms", row_decay=4.0)
    g = torch.ones(1, 2)
    steps = []
    for _ in range(5):
        before = p.data[0].clone()
        opt.set_rows(torch.tensor([0]), [g])
        opt.step()
        steps.append(float((before - p.data[0])[0]))
    want = [0.1 * (4 / (k + 4)) ** 0.5 for k in range(5)]
    assert all(abs(a - b) < 1e-5 for a, b in zip(steps, want)), (steps, want)
    opt.set_rows(torch.tensor([2]), [g])
    opt.step()
    assert abs(float(p.data[2, 0]) + 0.1) < 1e-5 and (p.data[1] == 0).all()


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


def test_evaluate_matches_reference():
    """evaluate (cihazda birikim, tek senkron; son eksik batch) = açık hesap: kayıp, doğruluk, bant doğrulukları."""
    import contextlib
    from rgn.model import RGNModel
    from rgn.train.evaluate import evaluate
    from _util import tiny
    torch.manual_seed(0)
    m = RGNModel(tiny()).eval()
    win = torch.randint(0, 97, (7, 33))
    r = evaluate(m, win, 3, torch.device("cpu"), contextlib.nullcontext(), chunk=40)
    with torch.no_grad():
        logits = m(win[:, :-1]).float()
    y = win[:, 1:]
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    hit = logits.argmax(-1) == y
    assert abs(r["loss"] - nll.item()) < 1e-5 and abs(r["acc"] - hit.float().mean().item()) < 1e-9, r
    assert abs(r["acc_bands"]["0-32"] - hit.float().mean().item()) < 1e-9 and r["tokens"] == 7 * 32
