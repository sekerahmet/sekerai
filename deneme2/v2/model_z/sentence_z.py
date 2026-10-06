"""sentence_z (V2) -- cumle <-> z formulu, GPT-2 token duzeyinde (egitim yok).  V1 deneme2/model_z/core/sentence/
sentence_z.py'nin kopyasi + degisiklik (import yok); adlar onayli (encode_z, decode_z; kullanici 6 Ekim).  Tarif:
belge/model_z_temel/22_V2_model_z_tarifi.md s1.

    token vektoru   f(v) = birim([meaning(v) ; kimlik(v)])     v: GPT-2 token'i ya da END; meaning 256, kimlik kalan
    konum kanali    sum_t R_t f(t_t) + R_n f(END)               R_t f = kaydir_t(isaret_t * f); tersi R_t^T
    torba kanali    sum_t f(t_t) / sqrt(n)                     (bag_channel; konumsuz okunur, rapor 15 s2)
    z = [konum ; torba]  (z_pos + z_pos)

Cumle = sinirdan sinira token'lar, END YOK (END konum kanalinda n'inci konum).  Grammar rol terimi yok (kullanici, 6
Ekim).  decode_z: u_t = R_t^T z, en emin konum once (SIC); yalniz konum kanali.  Konum anahtari sayisi veriden: en uzun
cumle + 1 (build_keys(longest)).

Ortak sozluk (belge 29; adlar onayli, kullanici 6 Ekim): build_keys(..., identity=False) -> kimlik yok, F yok; m_hat
(sabit meaning yarisi) ve yari ici R_t (m yarisi kendi Dm'sinde, e yarisi kendi De'sinde).  f = [m_hat ; e_hat] / sqrt(2)
modelin E'sinden gelir (MeaningEmbedding.f_table); encode_z_rows gradyanli, duz token yolu.  decode_z_meaning: z'nin m
yarisindan SIC, sabit m_hat tablosuyla (egitimden bagimsiz teshis).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from data import END_ID, VOCAB  # noqa: E402
Z = 512
DECODE_BYTES = 2 ** 30      # decode_z: skor tablosu (B x konum x V) parca basina en cok bu kadar bayt


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _halves_shift(sizes, L):
    """Yari ici kaydirma: bloklar (boyutlar sizes) -> shift, unshift (L, sum(sizes)); y[j] = x[shift[t, j]]."""
    j = torch.arange(sum(sizes))
    start = torch.repeat_interleave(torch.tensor([0] + list(sizes[:-1])).cumsum(0), torch.tensor(sizes))
    size = torch.repeat_interleave(torch.tensor(sizes), torch.tensor(sizes))
    off, t = j - start, torch.arange(L)[:, None]
    return start + (off[None] - t) % size, start + (off[None] + t) % size


def build_keys(meaning_path, longest, z=Z, seed=1, bag_channel=True, identity=True, e_dim=None):
    """meaning agent.pt (source.weight, en az END_ID satir) -> anahtarlar (sabit; ayni tohum ayni z).  longest: en uzun
    cumle (token, END haric; veriden).  keys["z"] toplam z boyutu (model z_dim'i buradan alir), keys["z_pos"] konum.
    identity=False (ortak sozluk, belge 29): F yok; m_hat (VOCAB, Dm) ve yari ici anahtarlar, z_pos = Dm + e_dim (e_dim
    varsayilan Dm; modelin d'si buna esit olmali)."""
    mp = torch.load(meaning_path, map_location="cpu", weights_only=False)
    src = mp["state"]["source.weight"].float()
    assert len(src) >= END_ID, "meaning satiri %d < GPT-2 sozlugu %d" % (len(src), END_ID)
    g = torch.Generator().manual_seed(seed)
    m = _unit(src[:END_ID])
    m = torch.cat([m, _unit(torch.randn(1, m.shape[1], generator=g))])          # END: rastgele birim
    if not identity:
        Dm = m.shape[1]
        D = Dm + (Dm if e_dim is None else e_dim)
        L = longest + 1
        shift, unshift = _halves_shift((Dm, D - Dm), L)
        signs = torch.randint(0, 2, (L, D), generator=g).float() * 2 - 1
        return dict(END=END_ID, m_hat=m, signs=signs, shift=shift, unshift=unshift, z=D * (2 if bag_channel else 1),
                    z_pos=D, bag_channel=bag_channel, identity=False, m_dim=Dm)
    assert m.shape[1] < z, "z meaning boyundan buyuk olmali"
    ident = _unit(torch.randn(VOCAB, z - m.shape[1], generator=g))
    F = _unit(torch.cat([m, ident], 1))                                         # (VOCAB, z), satir END_ID = END
    L = longest + 1                                                             # + END konumu
    signs = torch.randint(0, 2, (L, z), generator=g).float() * 2 - 1
    j = torch.arange(z)
    shift = (j[None] - torch.arange(L)[:, None]) % z                           # kaydir_t: y[j] = x[j - t]
    unshift = (j[None] + torch.arange(L)[:, None]) % z
    return dict(END=END_ID, F=F, signs=signs, shift=shift, unshift=unshift, z=z * (2 if bag_channel else 1), z_pos=z,
                bag_channel=bag_channel)


def keys_to(keys, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in keys.items()}


def _bind(keys, ids, t):
    """R_t f(ids): ids (...), t (...) ayni sekil -> (..., z_pos)."""
    f = keys["F"][ids] * keys["signs"][t]
    return f.gather(-1, keys["shift"][t])


@torch.no_grad()
def encode_z(keys, ids, mask):
    """ids, mask (B, L) sirali cumleler (dolgu 0, END yok) -> z (B, keys["z"]), fp32."""
    B, L = ids.shape
    assert L < len(keys["signs"]), "cumle build_keys(longest)'ten uzun"
    n = mask.sum(1)
    dev = ids.device
    with torch.autocast(dev.type, enabled=False):
        full = torch.cat([ids, torch.zeros(B, 1, dtype=ids.dtype, device=dev)], 1).scatter(1, n[:, None], keys["END"])
        keep = torch.arange(L + 1, device=dev)[None] <= n[:, None]
        t = torch.arange(L + 1, device=dev).expand(B, -1)
        z = (_bind(keys, full, t) * keep[..., None]).sum(1)
        if not keys["bag_channel"]:
            return z
        bag = (keys["F"][ids] * mask[..., None]).sum(1) / n.clamp_min(1)[:, None].float().sqrt()
        return torch.cat([z, bag], 1)


def encode_z_rows(keys, F, ids, mask):
    """encode_z, f satirlari disaridan (F (VOCAB, z_pos), gradyanli; ortak sozlukte MeaningEmbedding.f_table) ve duz
    token yolu (dolgulu (B, L, z_pos) tensor yok): ids, mask (B, L) -> z (B, keys["z"]) fp32.  Identity anahtarlari ve
    keys["F"] verilince encode_z ile ayni (test)."""
    B, L = ids.shape
    assert L < len(keys["signs"]), "cumle build_keys(longest)'ten uzun"
    dev = ids.device
    n = mask.sum(1)
    with torch.autocast(dev.type, enabled=False):
        F = F.float()
        b, t = mask.nonzero(as_tuple=True)
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
def decode_z_meaning(keys, z, max_len=None):
    """Ortak sozluk teshisi (belge 29 s1.4 a): z'nin konum kanalinin m yarisi (sqrt(2) olcekli) -> SIC, sabit m_hat
    tablosuyla.  Egitimden bagimsiz; kisa / orta cumlede birebir (olculdu: <= 10 token ~1,000)."""
    Dm = keys["m_dim"]
    sub = dict(END=keys["END"], F=keys["m_hat"], signs=keys["signs"][:, :Dm], shift=keys["shift"][:, :Dm],
               unshift=keys["unshift"][:, :Dm], z_pos=Dm)
    return decode_z(sub, z[:, :Dm] * 2 ** 0.5, max_len=max_len)


@torch.no_grad()
def decode_z(keys, z, sic=True, max_len=None):
    """z (B, z) -> token listeleri (END'e kadar); yalniz konum kanali.  max_len: en uzun cumle biliniyorsa yalniz ilk
    max_len + 1 konum cozulur; satirlar DECODE_BYTES'lik parcalarla."""
    Lk = len(keys["signs"]) if max_len is None else min(len(keys["signs"]), max_len + 1)
    step = max(1, DECODE_BYTES // (Lk * len(keys["F"]) * 4))
    if len(z) > step:
        return [w for c in range(0, len(z), step) for w in decode_z(keys, z[c:c + step], sic, max_len)]
    F, signs, unshift, END = keys["F"], keys["signs"][:Lk], keys["unshift"][:Lk], keys["END"]
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
                r[act] -= _bind(keys, w[act], t[act])
                end = torch.where(act & (w == END), torch.minimum(end, t), end)
    out = []
    for row in words.tolist():
        out.append(row[:row.index(END)] if END in row else row)
    return out
