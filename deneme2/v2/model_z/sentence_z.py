"""sentence_z (V2) -- cumle <-> z formulu, GPT-2 token duzeyinde.  Tarif: belge/model_z_temel/22 s1, 29 (yari ici R_t),
31 (own); temizlik belge 33 (kullanici, 6 Ekim: "model Z kendi E standart varsayılan olsun").

    token vektoru   f(v) = [birim(E(v)[:h]) ; birim(E(v)[h:])] / sqrt(2)      E: modelin ogrenilen sozlugu, h = d / 2
    konum kanali    sum_t R_t f(t_t) + R_n f(END)                            R_t: her yarinin icinde kaydir_t(isaret_t * .)
    torba kanali    sum_t f(t_t) / sqrt(n)                                   (bag_channel)
    z = [konum ; torba]  (2d)

Cumle = sinirdan sinira token'lar, END YOK (END konum kanalinda n'inci konum).  f tablosu modelden gelir
(SentenceTransformer.f_table; gradyanli).  encode_z duz token yolu: gercek token indeksleri (b, t) disaridan
(PackedBatch.z_flat), GPU senkronu yok.  decode_z: u_t = R_t^T z, en emin konum once (SIC), verilen F tablosuyla; teshis.
Konum anahtari sayisi veriden: en uzun cumle + 1.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID  # noqa: E402
DECODE_BYTES = 2 ** 30      # decode_z: skor tablosu (B x konum x V) parca basina en cok bu kadar bayt


def _halves_shift(sizes, L):
    """Yari ici kaydirma: bloklar (boyutlar sizes) -> shift, unshift (L, sum(sizes)); y[j] = x[shift[t, j]]."""
    j = torch.arange(sum(sizes))
    start = torch.repeat_interleave(torch.tensor([0] + list(sizes[:-1])).cumsum(0), torch.tensor(sizes))
    size = torch.repeat_interleave(torch.tensor(sizes), torch.tensor(sizes))
    off, t = j - start, torch.arange(L)[:, None]
    return start + (off[None] - t) % size, start + (off[None] + t) % size


def build_keys(longest, half, seed=1, bag_channel=True):
    """-> anahtarlar (sabit; ayni tohum ayni z).  longest: en uzun cumle (token, END haric; veriden); half: f'nin yari
    boyu (d / 2).  keys["z"] toplam z boyutu (model z_dim'i), keys["z_pos"] = 2 * half."""
    g = torch.Generator().manual_seed(seed)
    L = longest + 1                                                             # + END konumu
    shift, unshift = _halves_shift((half, half), L)
    signs = torch.randint(0, 2, (L, 2 * half), generator=g).float() * 2 - 1
    return dict(END=END_ID, signs=signs, shift=shift, unshift=unshift, z=2 * half * (2 if bag_channel else 1),
                z_pos=2 * half, bag_channel=bag_channel, half=half)


def keys_to(keys, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in keys.items()}


def encode_z(keys, F, ids, mask, flat):
    """F (VOCAB, z_pos) f tablosu (gradyanli), ids, mask (B, L) sirali cumleler (dolgu 0, END yok), flat (b, t) gercek
    token indeksleri (mask.nonzero sirasi; PackedBatch.z_flat) -> z (B, keys["z"]) fp32.  Duz token yolu: dolgulu
    (B, L, z_pos) tensor yok, nonzero yok."""
    B, L = ids.shape
    assert L < len(keys["signs"]), "cumle build_keys(longest)'ten uzun"
    dev = ids.device
    n = mask.sum(1)
    with torch.autocast(dev.type, enabled=False):
        F = F.float()
        b, t = flat
        tok = torch.cat([ids[b, t], torch.full((B,), keys["END"], dtype=ids.dtype, device=dev)])
        b, t = torch.cat([b, torch.arange(B, device=dev)]), torch.cat([t, n])           # + END konumu n
        f = torch.nn.functional.embedding(tok, F)                                       # geri yayilim deterministik
        z = torch.zeros(B, F.shape[1], device=dev).index_add(0, b, (f * keys["signs"][t]).gather(1, keys["shift"][t]))
        if not keys["bag_channel"]:
            return z
        k = len(tok) - B                                                                # END haric token'lar
        bag = torch.zeros(B, F.shape[1], device=dev).index_add(0, b[:k], f[:k]) / n.clamp_min(1)[:, None].float().sqrt()
        return torch.cat([z, bag], 1)


@torch.no_grad()
def decode_z(keys, F, z, sic=True, max_len=None):
    """Teshis: F (VOCAB, z_pos) tablosuyla (modelin f_table'i) z (B, z) -> token listeleri (END'e kadar); yalniz konum
    kanali.  max_len: en uzun cumle biliniyorsa yalniz ilk max_len + 1 konum; satirlar DECODE_BYTES'lik parcalarla."""
    Lk = len(keys["signs"]) if max_len is None else min(len(keys["signs"]), max_len + 1)
    step = max(1, DECODE_BYTES // (Lk * len(F) * 4))
    if len(z) > step:
        return [w for c in range(0, len(z), step) for w in decode_z(keys, F, z[c:c + step], sic, max_len)]
    F = F.float()
    signs, unshift, END = keys["signs"][:Lk], keys["unshift"][:Lk], keys["END"]
    B = len(z)
    dev = z.device
    rows = torch.arange(B, device=dev)

    def scores(r):                                                              # (B, Lk, VOCAB)
        return (r[:, unshift] * signs[None]) @ F.T

    with torch.autocast(dev.type, enabled=False):
        z = z[:, :keys["z_pos"]].float()
        if not sic:
            words = scores(z).argmax(-1)
        else:
            r = z.clone()
            words = torch.full((B, Lk), END, dtype=torch.long, device=dev)
            done = torch.zeros(B, Lk, dtype=torch.bool, device=dev)
            end = torch.full((B,), Lk - 1, device=dev)
            for _ in range(Lk):
                s = scores(r)
                top = s.topk(2, dim=-1)
                margin = top.values[..., 0] - top.values[..., 1]
                open_ = ~done & (torch.arange(Lk, device=dev)[None] <= end[:, None])
                if not open_.any():
                    break
                margin = margin.masked_fill(~open_, -float("inf"))
                t = margin.argmax(1)
                act = open_.any(1)
                w = top.indices[rows, t, 0]
                words[rows[act], t[act]] = w[act]
                done[rows[act], t[act]] = True
                ta = t[act]
                r[act] -= (F[w[act]] * keys["signs"][ta]).gather(-1, keys["shift"][ta])
                end = torch.where(act & (w == END), torch.minimum(end, t), end)
    out = []
    for row in words.tolist():
        out.append(row[:row.index(END)] if END in row else row)
    return out
