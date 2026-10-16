"""Token akışı yükleyici: memmap dosyalarından rastgele pencere (eğitim) ve sabit pencere (doğrulama) batch'leri.

Pencereler belge sınırını aşabilir (EOS ayırıcı model tarafından görülür); nanoGPT tarzı standart düzen.
Colab'da Drive'dan okumak yavaş olduğu için dosyalar önce yerel diske kopyalanır (DataConfig.local_copy).
"""
import shutil
from pathlib import Path

import numpy as np
import torch

from rgn.data.registry import get_spec
from rgn.paths import cache_dir, drive_root, in_colab


def _open(path, fmt="npy"):
    return np.load(path, mmap_mode="r")


def _local(path, enabled):
    """Colab'da dosyanın yerel kopyası (yoksa / boyu farklıysa kopyalanır)."""
    if not (enabled and in_colab()):
        return path
    dst = cache_dir() / Path(path).relative_to(drive_root())
    if not dst.exists() or dst.stat().st_size != Path(path).stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("[veri] yerel kopya: %s -> %s" % (path, dst), flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        shutil.copyfile(path, tmp)
        tmp.replace(dst)
    return dst


class TokenStream:
    """Bir ya da daha çok uint16 token dosyası; dosya boyuyla orantılı rastgele pencere seçimi."""

    def __init__(self, arrays):
        self.arrays = [a for a in arrays if len(a) > 1]
        assert self.arrays, "boş token akışı"
        self.sizes = np.array([len(a) for a in self.arrays], dtype=np.int64)
        self.total = int(self.sizes.sum())

    def random_batch(self, batch, seq_len, rng):
        """-> x, y (batch, seq_len) int64; y = x'in bir kaydırılmışı."""
        ok = self.sizes > seq_len                                  # pencere sığmayan dosya seçilmez
        assert ok.any(), "hiçbir dosya seq_len + 1 token içermiyor"
        w = np.where(ok, self.sizes, 0)
        files = rng.choice(len(self.arrays), size=batch, p=w / w.sum())
        out = np.empty((batch, seq_len + 1), dtype=np.int64)
        for r, f in enumerate(files):
            start = rng.integers(0, self.sizes[f] - seq_len)       # üst sınır dışlayıcı: start + seq_len + 1 <= boy
            out[r] = self.arrays[f][start:start + seq_len + 1]
        t = torch.from_numpy(out)
        return t[:, :-1], t[:, 1:]

    def fixed_windows(self, seq_len, max_tokens=0):
        """Doğrulama: ilk dosyanın başından ardışık, örtüşmeyen pencereler (her çağrıda aynı).  -> (n, seq_len + 1)."""
        a = self.arrays[0]
        n_tok = len(a) - 1 if not max_tokens else min(len(a) - 1, max_tokens)
        n = n_tok // seq_len
        idx = np.arange(n)[:, None] * seq_len + np.arange(seq_len + 1)[None, :]
        return torch.from_numpy(np.asarray(a[:n * seq_len + 1])[idx].astype(np.int64))


def synthetic_arrays(n_tokens=200_000, vocab=512, seed=0, p_follow=0.9):
    """Testler için öğrenilebilir dizi: sabit bir Markov zinciri (her token %90 olasılıkla sabit bir ardılına gider,
    aksi halde rastgele).  Tohumdan bağımsız aynı geçiş tablosu: eğitim ve doğrulama aynı dağılımdan."""
    succ = np.random.default_rng(12345).permutation(vocab).astype(np.uint16)
    rng = np.random.default_rng(seed)
    data = np.empty(n_tokens, dtype=np.uint16)
    data[0] = 1
    follow = rng.random(n_tokens) < p_follow
    rand = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    for i in range(1, n_tokens):
        data[i] = succ[data[i - 1]] if follow[i] else rand[i]
    return data


def load_streams(data_cfg):
    """DataConfig -> (eğitim akışı, doğrulama akışı, bytes_per_token)."""
    spec = get_spec(data_cfg.dataset)
    if spec.fmt == "synthetic":
        return (TokenStream([synthetic_arrays(seed=0)]), TokenStream([synthetic_arrays(50_000, seed=1)]),
                spec.bytes_per_token)
    root = drive_root()
    for rel in spec.train + spec.valid:
        if not (root / rel).exists():
            raise FileNotFoundError("%s bulunamadı" % (root / rel))
    train = TokenStream([_open(_local(root / r, data_cfg.local_copy)) for r in spec.train])
    valid = TokenStream([_open(_local(root / r, data_cfg.local_copy)) for r in spec.valid])
    return train, valid, spec.bytes_per_token


def valid_tokens(dataset="ss", n=0):
    """Doğrulama akışının başından n token (0: hepsi) -> numpy (analiz betikleri)."""
    spec = get_spec(dataset)
    a = _open(drive_root() / spec.valid[0])
    return np.asarray(a[:n] if n else a)
