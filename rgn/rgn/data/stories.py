"""Hikâye / cümle dizilimi (kullanıcı, 15 Ekim: "rastgele pencere yerine cümle cümle bölmek ne kadar etki ediyor").

Aynı SS token akışı (simplestories/gpt2/{train,valid}.npy: metin + EOS), v2'nin cümle sınırları (v2/simplestories_gpt2/
{split}_sentence_offsets.npy (S, 2) [başlangıç, bitiş) akış konumu; {split}_story_offsets.npy hikâye -> cümle aralığı).
Satır: bir hikâyenin önündeki EOS'tan (hikâye başı, BOS) başlar, ardışık cümleler sığdığı kadar eklenir, yalnız cümle
sonunda kesilir (hikâye biterse sonraki hikâye EOS'uyla aynı satırda sürer); kalan dolgu (hedef -100).
Kimlikler: doc = satırdaki hikâye sırası (EOS'ta artar), sent = cümle kimliği (EOS: -1 = hikâye başı); dolgu doc -1,
sent -2.  Satıra sığmayan (> seq_len) hikâyenin kesilen kuyruğu bu satırda yok (SS: hikâyelerin %6,5'i, token'ların %1,2'si).
"""
import numpy as np
import torch

from rgn.data.registry import EOS_ID


class StoryRows:
    def __init__(self, stream, story_offsets, sentence_offsets):
        se = np.asarray(sentence_offsets)
        so = np.asarray(story_offsets)
        self.stream = stream
        self.sent_start = np.ascontiguousarray(se[:, 0])
        self.sent_end = np.ascontiguousarray(se[:, 1])
        self.story_start = self.sent_start[so[:-1]]
        self.total = len(stream)
        assert int(stream[self.story_start[1] - 1]) == EOS_ID, "hikâyeden önce EOS yok: akış ile sınırlar uyuşmuyor"

    def _end(self, begin, T):
        """begin'den başlayan satırın sonu: begin + T'yi aşmayan son cümle sonu."""
        end = int(self.sent_end[np.searchsorted(self.sent_end, begin + T, side="right") - 1])
        assert end > begin, "cümle satıra sığmıyor"
        return end

    def _row(self, s, T):
        begin = int(self.story_start[s]) - 1                       # hikâye başı EOS'u
        end = self._end(begin, T)
        x = np.asarray(self.stream[begin:end], dtype=np.int64)
        y = np.asarray(self.stream[begin + 1:end + 1], dtype=np.int64)
        pos = np.arange(begin, end)
        sent = np.searchsorted(self.sent_start, pos, side="right") - 1
        eos = x == EOS_ID
        sent[eos] = -1
        doc = np.cumsum(eos)
        return x, y, doc, sent, end

    def _stack(self, rows, T):
        B = len(rows)
        X = np.zeros((B, T), np.int64)
        Y = np.full((B, T), -100, np.int64)
        D = np.full((B, T), -1, np.int64)
        S = np.full((B, T), -2, np.int64)
        for r, (x, y, doc, sent, _) in enumerate(rows):
            n = len(x)
            X[r, :n], Y[r, :n], D[r, :n], S[r, :n] = x, y, doc, sent
        return tuple(torch.from_numpy(a) for a in (X, Y, D, S))

    def random_batch(self, batch, T, rng):
        """-> x, y, doc, sent (batch, T); rastgele hikâye başlarından (ilk hikâye hariç: önünde EOS yok)."""
        s = rng.integers(1, len(self.story_start), size=batch)
        return self._stack([self._row(int(i), T) for i in s], T)

    def fixed_rows(self, T, max_tokens=0):
        """Doğrulama: hikâye sırasıyla, örtüşmeyen satırlar (her çağrıda aynı) -> (x, y, doc, sent) (n, T)."""
        rows, s, n_tok = [], 1, 0
        while s < len(self.story_start) and (not max_tokens or n_tok < max_tokens):
            r = self._row(s, T)
            rows.append(r)
            n_tok += len(r[0])
            s = int(np.searchsorted(self.story_start, r[4], side="right"))
        return self._stack(rows, T)


def load_story_rows(stream, story_root, split):
    """story_root (Drive köküne göre, ör. v2/simplestories_gpt2) altındaki sınırlarla StoryRows."""
    from rgn.data.loader import _local
    from rgn.paths import drive_root
    root = drive_root() / story_root
    so = np.load(_local(root / ("%s_story_offsets.npy" % split), True))
    se = np.load(_local(root / ("%s_sentence_offsets.npy" % split), True))
    return StoryRows(stream, so, se)
