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


def test_removed_fields_compat():
    """Eski config.json'lardaki kaldırılan alanlar: mevcut tasarımla uyumlu değerde yok sayılır, değilse hata."""
    from rgn.config import from_dict
    old = {"model": {"attn": "softmax", "pos": "rope", "mlp": "swiglu", "sharpness": 1, "gamma_range": [4.0, 512.0],
                     "attn_importance": False, "local_window": 128, "global_layers": 3},
           "data": {"max_train_files": 0}, "train": {"profile_after": True}}
    cfg = from_dict(old)
    assert cfg.model.layer_kinds() == "YYGYGYYG"
    for bad in ({"attn": "rgn"}, {"mlp": "rational"}, {"observer_layer": 5}):
        try:
            from_dict({"model": bad})
        except AssertionError:
            continue
        raise AssertionError("kabul edilmemeliydi: %s" % bad)


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
            assert spec.train and spec.valid and spec.fmt == "npy" and spec.bytes_per_token > 0, name


def test_all_configs_load():
    files = sorted((ROOT / "configs").rglob("*.json"))
    assert files
    for f in files:
        cfg = load_config(f)
        cfg.model.validate()
    names = [load_config(f).name for f in files if not f.parent.name == "base"]
    assert len(names) == len(set(names)), "deney adları tekil olmalı"


def _story_fixture():
    """3 hikâye (EOS ile ayrılmış), cümle sınırları v2 biçiminde: akış konumu [başlangıç, bitiş)."""
    import numpy as np
    from rgn.data.registry import EOS_ID
    sents = [[[1, 2, 3], [4, 5]], [[6, 7], [8, 9, 10], [11]], [[12, 13, 14, 15]]]
    stream, se, so = [], [], [0]
    for st in sents:
        for s in st:
            se.append([len(stream), len(stream) + len(s)])
            stream += s
        stream.append(EOS_ID)
        so.append(len(se))
    return np.array(stream, np.uint16), np.array(so), np.array(se)


def test_story_rows():
    """Satır hikâye başı EOS'uyla başlar, yalnız cümle sonunda kesilir; doc EOS'ta artar, EOS sent -1; dolgu hedefi -100."""
    import numpy as np
    from rgn.data.registry import EOS_ID
    from rgn.data.stories import StoryRows
    stream, so, se = _story_fixture()
    r = StoryRows(stream, so, se)
    x, y, doc, sent = r._stack([r._row(1, 8)], 8)
    # hikâye 1 = EOS 6 7 8 9 10 11 (7 token); 8'e kadar sığan son cümle sonu 11'den sonra (EOS hedefi)
    assert x[0].tolist() == [EOS_ID, 6, 7, 8, 9, 10, 11, 0] and y[0].tolist() == [6, 7, 8, 9, 10, 11, EOS_ID, -100]
    assert doc[0].tolist() == [1, 1, 1, 1, 1, 1, 1, -1] and sent[0].tolist() == [-1, 2, 2, 3, 3, 3, 4, -2]
    x, y, doc, sent = r._stack([r._row(1, 6)], 6)                  # sığmayan son cümle (11) atılır, satır cümle sonunda biter
    assert x[0].tolist() == [EOS_ID, 6, 7, 8, 9, 10] and y[0, -1].item() == 11
    x, y, doc, sent = r._stack([r._row(1, 12)], 12)                # iki hikâye aynı satırda: doc artar
    assert doc[0].tolist() == [1] * 7 + [2] * 5 and x[0, 7].item() == EOS_ID and y[0, -1].item() == EOS_ID
    X, Y, D, S = r.fixed_rows(8)
    assert X.shape[0] == 2 and X[1, 0].item() == EOS_ID and X[1, 1].item() == 12   # sonraki satır sonraki hikâyeden
    assert np.all((Y == -100).numpy() == (D == -1).numpy())


def test_story_masks_and_loss():
    """Y maskesi: kendi cümlesi + hikâye başı; G: kendi hikâyesi.  Cümle yerel modelde önceki cümlenin token'ı yalnız-Y
    modelin çıktısını değiştirmez; önceki hikâye G'li modeli değiştirmez; dolgu kaybı sayılmaz."""
    import torch
    from _util import tiny
    from rgn.data.stories import StoryRows
    from rgn.model import RGNModel
    from rgn.model.attention import story_masks
    from rgn.model.rgn import lm_loss
    stream, so, se = _story_fixture()
    r = StoryRows(stream, so, se)
    x, y, doc, sent = r._stack([r._row(1, 12)], 12)
    m = story_masks(doc, sent, 0)
    Ym, Gm = m["Y"][0, 0], m["G"][0, 0]
    assert Ym[4].tolist() == [True, False, False, True, True] + [False] * 7     # token 8: EOS + kendi cümlesi (8, 9)
    assert Gm[8].tolist() == [False] * 7 + [True, True] + [False] * 3           # 2. hikâye yalnız kendine bakar
    torch.manual_seed(0)
    yonly = RGNModel(tiny(vocab_size=50304, n_kv_heads=4, local_window=4, global_layers=0, local_sentence=True,
                          n_layers=2)).eval()
    x2 = x.clone()
    x2[0, 1], x2[0, 2] = 30, 31                                                  # 1. cümleyi değiştir
    h1, h2 = yonly.hidden(x, masks=m), yonly.hidden(x2, masks=m)
    assert torch.allclose(h1[0, 3:7], h2[0, 3:7], atol=1e-5) and not torch.allclose(h1[0, 1:3], h2[0, 1:3])
    full = RGNModel(tiny(vocab_size=50304, n_kv_heads=4, local_window=4, global_layers=1, local_sentence=True,
                         n_layers=2)).eval()
    assert torch.allclose(full.hidden(x, masks=m)[0, 7:], full.hidden(x2, masks=m)[0, 7:], atol=1e-5)
    assert not torch.allclose(full.hidden(x, masks=m)[0, 3:7], full.hidden(x2, masks=m)[0, 3:7])
    h = torch.randn(1, 4, 8)
    w = torch.randn(11, 8)
    t = torch.tensor([[1, 2, -100, -100]])
    ref = torch.nn.functional.cross_entropy((h[0, :2] @ w.T), t[0, :2])
    assert torch.allclose(lm_loss(h, w, t, use_checkpoint=False), ref, atol=1e-5)
