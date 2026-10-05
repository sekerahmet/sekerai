"""sentence_z -- cumle <-> z formulu (egitim yok).  Kullanici, 5 Ekim: "matematikçinin formülü + meaning + grammer
bilgileri = Z de"; adlar onayli (encode_z, decode_z).  Baglama ailesi: TPR (Smolensky 1990), HRR (Plate 1995).

    kelime vektoru  f(w) = birim([meaning(w) ; kimlik(w)])      meaning: anlam;  kimlik: rastgele, geri acmayi ayirir
    konum anahtari  R_t f = kaydir_t(isaret_t * f)              sirayi tutar; tersi R_t^T
    rol anahtari    g_t = isaret(P h_t)  (+-1)                  h_t: grammar torba okuyucusu (kelimenin yuvasi, sirasiz)
    z = sum_t R_t f(w_t) + R_n f(END) + sum_t g_t * f(w_t)

decode_z: u_t = R_t^T z, her konumda en yakin kelime; SIC: en emin konum once cozulur, katkisi z'den cikarilir.  Rol
terimi geri acmada cikarilamaz (h_t butun cumleye bagli), gurultu gibi kalir.  Olculdu (ulke, Z 512, SIC): gorulmemis
356 cumlede birebir 0,969.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grammar"))
from grammar import GrammarAgent  # noqa: E402

Z = 512


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def build_keys(meaning_path, grammar_path, z=Z, longest=64, seed=1):
    """meaning ve grammar agent.pt -> anahtarlar (sabit; ayni tohum ayni z).  longest: en uzun cumle (kelime)."""
    mp = torch.load(meaning_path, map_location="cpu", weights_only=False)
    gp = torch.load(grammar_path, map_location="cpu", weights_only=False)
    assert mp["vocab"] == gp["vocab"], "meaning ve grammar sozlugu farkli"
    gram = GrammarAgent(gp["vocab"], d=gp["args"]["d"])
    gram.load_state_dict(gp["state"])
    gram.eval()
    g = torch.Generator().manual_seed(seed)
    V = len(mp["vocab"])
    m = _unit(mp["state"]["source.weight"].float())
    m = torch.cat([m, _unit(torch.randn(1, m.shape[1], generator=g))])          # END
    assert m.shape[1] < z, "z meaning boyundan buyuk olmali"
    ident = _unit(torch.randn(V + 1, z - m.shape[1], generator=g))
    F = _unit(torch.cat([m, ident], 1))                                         # (V + 1, z), satir V = END
    L = longest + 1                                                             # + END konumu
    signs = torch.randint(0, 2, (L, z), generator=g).float() * 2 - 1
    j = torch.arange(z)
    shift = (j[None] - torch.arange(L)[:, None]) % z                           # kaydir_t: y[j] = x[j - t]
    unshift = (j[None] + torch.arange(L)[:, None]) % z
    P = torch.randn(z, gram.E.weight.shape[1], generator=g)
    return dict(vocab=mp["vocab"], END=V, F=F, signs=signs, shift=shift, unshift=unshift, P=P, grammar=gram, z=z)


def keys_to(keys, device):
    """Anahtarlar ve grammar okuyucusu cihaza (GPU'da z hesabi)."""
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in keys.items()}
    out["grammar"] = keys["grammar"].to(device)
    return out


@torch.no_grad()
def _roles(keys, ids, mask):
    gram = keys["grammar"]
    B = len(ids)
    e = torch.cat([gram.E(ids), gram.boundary.expand(B, 1, -1)], 1)
    m = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=ids.device)], 1)
    h = gram.reader(e, src_key_padding_mask=~m)[:, :ids.shape[1]]
    return torch.sign(h @ keys["P"].T)                                          # (B, L, z)


def _bind(keys, ids, t):
    """R_t f(ids): ids (...), t (...) ayni sekil -> (..., z)."""
    f = keys["F"][ids] * keys["signs"][t]
    return f.gather(-1, keys["shift"][t])


@torch.no_grad()
def encode_z(keys, ids, mask):
    """ids, mask (B, L) sirali cumleler (dolgu 0) -> z (B, z)."""
    B, L = ids.shape
    assert L < len(keys["signs"]), "cumle build_keys(longest)'ten uzun"
    n = mask.sum(1)
    dev = ids.device
    full = torch.cat([ids, torch.zeros(B, 1, dtype=ids.dtype, device=dev)], 1).scatter(1, n[:, None], keys["END"])
    keep = torch.arange(L + 1, device=dev)[None] <= n[:, None]
    t = torch.arange(L + 1, device=dev).expand(B, -1)
    z = (_bind(keys, full, t) * keep[..., None]).sum(1)
    return z + (_roles(keys, ids, mask) * keys["F"][ids] * mask[..., None]).sum(1)


@torch.no_grad()
def decode_z(keys, z, sic=True):
    """z (B, z) -> kelime kimlik listeleri (END'e kadar)."""
    F, signs, unshift, END = keys["F"], keys["signs"], keys["unshift"], keys["END"]
    B, Lk = len(z), len(signs)
    dev = z.device
    rows = torch.arange(B, device=dev)

    def scores(r):                                                              # (B, Lk, V + 1)
        return (r[:, unshift] * signs[None]) @ F.T

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
