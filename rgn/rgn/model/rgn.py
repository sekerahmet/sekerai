"""RGNModel — decoder-only dil modeli (doküman 1.3 bileşen haritası).

    token -> embedding -> [ön-norm -> çekim dikkati -> residual -> ön-norm -> rasyonel MLP -> residual] × L
          -> final norm -> LM head (paylaşımlı embedding) -> sonraki token
"""
import dataclasses
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rgn.model.attention import Attention
from rgn.model.mlp import build_mlp
from rgn.model.norm import RMSNorm
from rgn.model.position import rope_tables


def vocab_neuron_table(V, ff, k, seed=0):
    """Sözlüğü nöronlara bölüştürür: token başına k nöron, her nöron eşit sayıda token (k·V/ff, ±1).  Token'lar rastgele
    sıraya dizilir (katman başına ayrı tohum: iki katmanda aynı nöronu paylaşan token'lar ayrışır); sıradaki i. token
    (i·k + j) mod ff nöronlarını alır.  -> (V, k) long."""
    g = torch.Generator().manual_seed(1000 + seed)
    order = torch.randperm(V, generator=g)
    slots = (torch.arange(V)[:, None] * k + torch.arange(k)[None, :]) % ff
    table = torch.empty(V, k, dtype=torch.long)
    table[order] = slots
    return table


def private_rows(cfg):
    """Kelimeye özel tablo satır sayısı: özel payı olan kelimeler + 1 ortak "diğer"."""
    from rgn.paths import ROOT
    return len(json.loads((ROOT / cfg.private_vocab).read_text(encoding="utf-8"))["tokens"]) + 1


def _onehot(idx, n, dtype):
    """(..., k) indeks -> (..., n) 0/1 maske."""
    return torch.zeros(*idx.shape[:-1], n, device=idx.device, dtype=dtype).scatter_(-1, idx, 1.0)


class Block(nn.Module):
    def __init__(self, cfg, kind="G", mlp_only=False, layer=0):
        """kind: 'G' genel (tam causal; konum cfg.global_pos ya da pos) | 'Y' yerel (son local_window token, konum pos).
        mlp_only: dikkat yok, yalnız MLP (token'lar arası bilgi akışı yok)."""
        super().__init__()
        self.kind = kind
        self.norm2 = RMSNorm(cfg.d_model)
        self.norm1 = None if mlp_only else RMSNorm(cfg.d_model)
        if cfg.ff_ratio:
            cfg = dataclasses.replace(cfg, d_ff=cfg.ff_dim(kind))
        if kind == "G" and cfg.global_pos:
            cfg = dataclasses.replace(cfg, pos=cfg.global_pos)
        if kind == "G" and cfg.sharpness_range_g:
            cfg = dataclasses.replace(cfg, sharpness_range=tuple(cfg.sharpness_range_g))
        self.attn = None if mlp_only else Attention(cfg, window=cfg.local_window if kind == "Y" else 0)
        self.mlp = build_mlp(cfg)
        if not (mlp_only and cfg.mlp_vocab_neurons):
            self.vocab_neurons = None
        else:
            if cfg.mlp_vocab_assign and layer == 0:              # PMI komşuluğuyla atama (dosyadan)
                from rgn.paths import ROOT
                a = json.loads((ROOT / cfg.mlp_vocab_assign).read_text(encoding="utf-8"))
                assert a["vocab"] == cfg.vocab_size and a["neurons"] == cfg.ff_dim(kind), "atama tablosu boyu"
                tbl = torch.tensor(a["table"], dtype=torch.long)[:, None]
            else:
                tbl = vocab_neuron_table(cfg.vocab_size, cfg.ff_dim(kind), cfg.mlp_vocab_neurons, seed=layer)
            self.register_buffer("vocab_neurons", tbl)
        self.private = None
        if cfg.private_neurons:                                  # kelimeye özel nöronlar: satır = kelime, [g | u | d] × p
            self.private = nn.Parameter(torch.empty(private_rows(cfg), 3 * cfg.private_neurons * cfg.d_model))
        self.route_h = self.route_n = None
        if not mlp_only and cfg.route_heads:                     # token rotası: head'ler ve nöronlar (katman başına ayrı)
            self.register_buffer("route_h_tbl", vocab_neuron_table(cfg.vocab_size, cfg.n_heads, cfg.route_heads,
                                                                    seed=100 + layer))
            self.route_h = True
        if not mlp_only and cfg.route_neurons:
            self.register_buffer("route_n_tbl", vocab_neuron_table(cfg.vocab_size, cfg.ff_dim(kind), cfg.route_neurons,
                                                                    seed=200 + layer))
            self.route_n = True
        self.loc_proj = None
        if cfg.attn_locality and not mlp_only:
            # konum özeti -> MLP kapı girdisi: E · (2L − 1) ∈ (−1, 1]; E sıfır başlatma (başlangıçta taban)
            taus = cfg.locality_tau_local if kind == "Y" else cfg.locality_tau_global
            self.loc_proj = nn.Linear(len(taus) * cfg.n_heads, cfg.ff_dim(), bias=False)

    def _private(self, xn, rows=None, pw=None):
        """Kelimeye özel nöronlar: xn (B, N, d) -> (B, N, d).  a = SiLU(g·x)(u·x), çıktı Σ a·d.  Satırlar: rows (B, N)
        ile tam tablodan, ya da eğitimde pw (B, N, 3pd): trainer'ın derleme dışında bir kez okuduğu satırlar."""
        B, N, D = xn.shape
        w = (F.embedding(rows, self.private) if pw is None else pw).view(B, N, -1, 3, D).to(xn.dtype)  # (B, N, p, 3, d)
        proj = (w[..., :2, :] * xn[:, :, None, None, :]).sum(-1)                        # (B, N, p, 2)
        a = F.silu(proj[..., 0]) * proj[..., 1]
        return (a.unsqueeze(-1) * w[..., 2, :]).sum(2)

    def forward(self, x, rope=None, masks=None, tokens=None, rows=None, pw=None):
        if self.attn is None:
            nm = None
            if self.vocab_neurons is not None:                   # token'ın kendi nöronları (sözlüğe bölüştürülmüş)
                assert tokens is not None, "sözlüğe bölüştürülmüş nöronlar token kimliği ister"
                idx = self.vocab_neurons[tokens]                     # (B, N, k)
                nm = torch.zeros(*idx.shape[:-1], self.mlp.c.in_features, device=x.device, dtype=x.dtype)
                nm.scatter_(-1, idx, 1.0)
            return x + (self.mlp(self.norm2(x)) if nm is None else self.mlp(self.norm2(x), neuron_mask=nm))
        hm = nm = None
        if self.route_h or self.route_n:
            assert tokens is not None, "token rotası token kimliği ister"
        if self.route_h:
            hm = _onehot(self.route_h_tbl[tokens], self.attn.H, x.dtype).transpose(1, 2)     # (B, H, N)
        if self.route_n:
            nm = _onehot(self.route_n_tbl[tokens], self.mlp.c.in_features, x.dtype)          # (B, N, ff)
        a = self.attn(self.norm1(x), rope, None if masks is None else masks[self.kind], head_mask=hm)
        extra = None
        if self.loc_proj is not None:
            a, loc = a
            extra = self.loc_proj((2.0 * loc - 1.0).to(x.dtype))
        s = x + a                                       # S_i = X_i + Attn(RMSNorm(X))_i
        xn = self.norm2(s)
        out = s + (self.mlp(xn, extra) if nm is None else self.mlp(xn, extra, neuron_mask=nm))
        if self.private is not None:                    # kelimeye özel nöronlar, ortak MLP'ye paralel
            out = out + self._private(xn, rows, pw)
        return out                                      # H_i = S_i + MLP(RMSNorm(S))_i [+ özel_i]


def _ce_sum(h, weight, target):
    return F.cross_entropy((h @ weight.T).float(), target, reduction="sum")


def lm_loss(h, weight, target, chunk=8192, use_checkpoint=True):
    """Ortalama cross-entropy; logit'ler parça parça hesaplanır.  use_checkpoint: logit'ler geri geçişte yeniden hesaplanır
    (V × N logit matrisi bellekte tutulmaz; bedeli LM head çarpımının bir kez daha yapılması)."""
    h, target = h.reshape(-1, h.shape[-1]), target.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, target.numel(), chunk):
        args = (h[i:i + chunk], weight, target[i:i + chunk])
        use = use_checkpoint and torch.is_grad_enabled()
        total = total + (checkpoint(_ce_sum, *args, use_reentrant=False) if use else _ce_sum(*args))
    return total / (target != -100).sum().clamp_min(1)          # dolgu hedefi (-100) sayılmaz


class RGNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, kind, i < cfg.mlp_only_layers, i) for i, kind in enumerate(cfg.layer_kinds()))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # W_vocab = W_embᵀ
        self.use_rope = any(b.attn is not None and b.attn.use_rope for b in self.blocks)
        if not cfg.private_neurons:
            self.priv_map = None
        else:                                  # token -> özel satır (0: ortak "diğer")
            from rgn.paths import ROOT
            used = json.loads((ROOT / cfg.private_vocab).read_text(encoding="utf-8"))["tokens"]
            pm = torch.zeros(cfg.vocab_size, dtype=torch.long)
            pm[torch.tensor(used)] = torch.arange(1, len(used) + 1)
            self.register_buffer("priv_map", pm)
        self.obs_head = None
        if cfg.observer_layer:
            # gözlemci: K. katman çıkışı (gradyansız) -> küçük sözlükte dağılım; özellikler sonraki katmandan önce akışa
            from rgn.paths import ROOT
            top = json.loads((ROOT / cfg.observer_vocab).read_text(encoding="utf-8"))
            omap = torch.full((cfg.vocab_size,), len(top), dtype=torch.long)
            omap[torch.tensor(top)] = torch.arange(len(top))
            self.register_buffer("obs_map", omap)                          # token -> sınıf (len(top) = "diğer")
            self.obs_norm = RMSNorm(cfg.d_model)
            self.obs_head = nn.Linear(cfg.d_model, len(top) + 1, bias=False)
            self.obs_feat = nn.Linear(3, cfg.d_model, bias=False)
        self.loss_checkpoint = True                     # trainer TrainConfig.loss_checkpoint ile ayarlar
        if self.use_rope:
            cos, sin = rope_tables(cfg.max_len, cfg.d_model // cfg.n_heads, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        """N(0, std); residual akışına yazan W_O ve W_2 için std / sqrt(2L); bias'lar sıfır."""
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            if name.endswith(("attn.w_aux.weight", "loc_proj.weight", "obs_feat.weight")):
                nn.init.zeros_(p)                       # önem 0, odak 1, kapı 1: başlangıçta transformer tabanı
            elif name.endswith(".private"):           # g, u ~ N(0, std); d = 0 (başlangıçta taban)
                nn.init.normal_(p, 0.0, std)
                p.data.view(p.shape[0], -1, 3, self.cfg.d_model)[:, :, 2].zero_()
            elif p.dim() == 2:
                s = std / math.sqrt(2 * self.cfg.n_layers) if name.endswith(("attn.w_o.weight", "mlp.c.weight")) else std
                nn.init.normal_(p, 0.0, s)
            elif name.endswith(".bias"):
                nn.init.zeros_(p)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def rope(self, n):
        if not self.use_rope:
            return None
        assert n <= self.cfg.max_len, "dizi %d > max_len %d (RoPE tablosu)" % (n, self.cfg.max_len)
        return self.rope_cos[:n], self.rope_sin[:n]

    def observe(self, x):
        """Gözlemci: x (gradyansız) -> (sınıf logit'leri, özellikler (B, N, 3) ∈ [0, 1]: belirsizlik, güven, nadir)."""
        ol = self.obs_head(self.obs_norm(x.detach())).float()
        with torch.no_grad():
            p = torch.softmax(ol, -1)
            H = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
            f = torch.stack([H / (1.0 + H), p.max(-1).values, p[..., -1]], -1)
        return ol, f

    def hidden(self, idx, return_obs=False, masks=None, priv=None):
        """idx (B, N) -> son normalize edilmiş durum (B, N, d) [, gözlemci logit'leri].  masks: story_masks (hikâye dizilimi)."""
        x = self.embed(idx)
        rope = self.rope(idx.shape[1])
        ol = None
        for i, block in enumerate(self.blocks):
            if priv is not None:                         # eğitim: bütün katmanların satırları (B, N, L·3pd), trainer okur
                k = priv.shape[-1] // len(self.blocks)
                x = block(x, rope, masks, idx, None, priv[..., i * k:(i + 1) * k])
            else:
                x = block(x, rope, masks, idx, None if self.priv_map is None else self.priv_map[idx])
            if self.obs_head is not None and i == self.cfg.observer_layer - 1:
                ol, f = self.observe(x)
                x = x + self.obs_feat((2.0 * f - 1.0).to(x.dtype))
        h = self.norm_f(x)
        return (h, ol) if return_obs else h

    def forward(self, idx, targets=None, masks=None, priv=None):
        """targets yoksa logit'ler (B, N, V); varsa ortalama kayıp (skaler) — gözlemci varsa (ana kayıp, gözlemci kaybı)."""
        if targets is None:
            return self.lm_head(self.hidden(idx, masks=masks))
        if self.obs_head is not None:
            h, ol = self.hidden(idx, return_obs=True, masks=masks, priv=priv)
            main = lm_loss(h, self.lm_head.weight, targets, use_checkpoint=self.loss_checkpoint)
            aux = F.cross_entropy(ol.reshape(-1, ol.shape[-1]), self.obs_map[targets].reshape(-1))
            return main, aux
        h = self.hidden(idx, masks=masks, priv=priv)
        return lm_loss(h, self.lm_head.weight, targets, use_checkpoint=self.loss_checkpoint)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, eos_id=None, generator=None):
        """Önbelleksiz başvuru üretimi.  temperature 0: argmax (üstel işlem yok); > 0: örnekleme."""
        limit = self.cfg.max_len if self.use_rope else None
        for _ in range(max_new_tokens):
            ctx = idx[:, -limit:] if limit else idx
            logits = self.lm_head(self.hidden(ctx)[:, -1]).float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx
