"""Önbellekli üretim (KV cache): istem tek ileri geçişte, sonra token başına tek adım.

Yerel (Y) katman: son local_window token için döner tampon (yuva = konum mod W); genel (G) katman: tam tampon (konuma
göre).  GQA (n_kv_heads < n_heads), dikkat kapısı, kelimeye özel nöron ve V vektörü desteklenir.  Adım dallanmasızdır;
CUDA'da ilk adımda torch.compile + CUDA graph'a alınır, sonra her adım tampona yaz + tek tekrar (v2 transformer
StaticCache ile aynı yol: belge 84).  Tampon boyu kademeli (BUCKET × {1, 1,5} × 2^k): her boy bir kez derlenir.

    cache = StaticCache(model, batch=B, positions=yeni_token_sayısı)
    logits = cache.prefill(idx)            # idx (B, T), satırlar eşit boy; yerel katman varsa istem FLEX_BLOCK'un
                                           # katına sağdan dolgulanır (causal: dolgu öncekileri etkilemez) → flex yolu
    logits = cache.step(next_token)        # (B,) -> (B, V)
"""
import torch
import torch.nn.functional as F

from rgn.model.attention import FLEX_BLOCK

_COMPILED = {}
_POOL = {}                  # (model, batch, tampon boyu, dtype) -> StaticCache: tamponlar ve CUDA graph tekrar kullanılır
BUCKET = 256


def decode_attn(q, k, v, mask):
    """Tek sorgulu dikkat: q (B, H, 1, dh), k / v (B, Hkv, S, dh), mask (B | 1, S) bool -> (B, H, 1, dh).  GQA için
    head'ler KV head'e katlanır; skor fp32."""
    B, H, _, dh = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    qf = (q * dh ** -0.5).reshape(B * Hkv, H // Hkv, dh)
    s = torch.bmm(qf, k.reshape(B * Hkv, S, dh).transpose(1, 2)).float().view(B, Hkv, H // Hkv, S)
    p = torch.softmax(s.masked_fill(~mask[:, None, None], float("-inf")), -1).to(v.dtype)
    return torch.matmul(p, v).reshape(B, H, 1, dh)


def _step(m, w, n, slot_pos, K, V, out):
    """Tek adım (yerinde, dallanmasız): token w (B,) konum n (1,) -> out (B, V) logit; n += 1."""
    x = m.embed(w)[:, None]
    ple = m.ple_context(x)
    rope = (m.rope_cos.index_select(0, n), m.rope_sin.index_select(0, n))
    rows = m.priv_map[w][:, None] if m.priv_map is not None else None
    W = m.cfg.local_window                                                 # döner tampon boyu (en büyük Y penceresi)
    slot, dist = n, None
    if W:                                                                  # yapılandırmaya göre sabit (dallanma değil)
        slot = n % W
        slot_pos.index_copy_(0, slot, n)
        dist = torch.where(slot_pos >= 0, n - slot_pos, W)                # yuvanın uzaklığı (boş yuva: W)
    kg = next((K[l] for l, b in enumerate(m.blocks) if b.kind == "G"), None)
    mask_g = (torch.arange(kg.shape[2], device=w.device) <= n)[None] if kg is not None else None
    for l, b in enumerate(m.blocks):
        xa = b.norm1(x)
        q, k, v = b.attn.project(xa, rope)
        vv = b.value_vec(xa, rows)
        if vv is not None:
            v = v + vv.view(len(w), 1, b.attn.Hkv, b.attn.dh).transpose(1, 2).to(v.dtype)
        pos = slot if b.kind == "Y" else n
        K[l].index_copy_(2, pos, k.to(K[l].dtype))
        V[l].index_copy_(2, pos, v.to(V[l].dtype))
        mask = (dist < b.attn.window)[None] if b.kind == "Y" else mask_g   # katmanın kendi penceresi (sabit)
        o = decode_attn(q.to(K[l].dtype), K[l], V[l], mask)
        x = b.finish(x, b.attn.finish(xa, o), rows, ple_ctx=None if ple is None else ple[:, :, l])
    out.copy_(m.lm_head(m.norm_f(x[:, 0])).float())
    n.add_(1)


class StaticCache:
    def __init__(self, model, batch, positions, dtype=None):
        self.m, self.B, self.positions = model, batch, positions
        self.dev = model.embed.weight.device
        self.dtype = dtype or (torch.bfloat16 if self.dev.type == "cuda" else torch.float32)
        self.run = None

    @torch.no_grad()
    def prefill(self, idx):
        """idx (B, T) -> son konumun logit'i (B, V); tamponlar doldurulur."""
        m, (B, T) = self.m, idx.shape
        assert B == self.B
        W = m.cfg.local_window
        need = T + self.positions
        assert need <= m.cfg.max_len, "istem + yeni %d > max_len %d (RoPE tablosu)" % (need, m.cfg.max_len)
        self.limit = min(m.cfg.max_len, next(t for k in range(40) for t in (BUCKET << k, (3 * BUCKET // 2) << k)
                                             if t >= need))
        Tx = -(-T // FLEX_BLOCK) * FLEX_BLOCK if W else T                      # yerel katman her boyda blok maskeli
        Tx = Tx if Tx <= m.cfg.max_len else T                                  # flex'le (yoksa yoğun T × T maske)
        xi = F.pad(idx, (0, Tx - T)) if Tx > T else idx
        x = m.embed(xi)
        ple = m.ple_context(x)
        rope = m.rope(Tx)
        rows = m.priv_map[xi] if m.priv_map is not None else None
        pos = torch.arange(T, device=self.dev)
        self.K, self.V = [], []
        for l, b in enumerate(m.blocks):
            xa = b.norm1(x)
            q, k, v = b.attn.project(xa, rope)
            vv = b.value_vec(xa, rows)
            if vv is not None:
                v = v + vv.view(B, Tx, b.attn.Hkv, b.attn.dh).transpose(1, 2).to(v.dtype)
            S = W if b.kind == "Y" else self.limit
            kb = k.new_zeros(B, b.attn.Hkv, S, b.attn.dh, dtype=self.dtype)
            vb = torch.zeros_like(kb)
            keep = pos[-W:] if b.kind == "Y" else pos                          # yerel: son W konum, yuvası konum mod W
            at = keep % W if b.kind == "Y" else keep
            kb[:, :, at], vb[:, :, at] = k[:, :, keep].to(self.dtype), v[:, :, keep].to(self.dtype)
            self.K.append(kb)
            self.V.append(vb)
            x = b.finish(x, b.attn.finish(xa, b.attn._core(q, k, v)), rows,
                         ple_ctx=None if ple is None else ple[:, :, l])
        slot_pos = torch.full((max(W, 1),), -1, dtype=torch.long, device=self.dev)
        if W:
            keep = pos[-W:]
            slot_pos[keep % W] = keep
        logits = m.lm_head(m.norm_f(x[:, T - 1])).float()
        new = dict(K=self.K, V=self.V, slot_pos=slot_pos, n=torch.tensor([T], device=self.dev),
                   w=torch.zeros(B, dtype=torch.long, device=self.dev), out=torch.zeros_like(logits))
        self.K, self.V = getattr(self, "_K", None), getattr(self, "_V", None)
        if self.run is not None:                                               # graph var: aynı tamponlara yerinde yaz
            for a, b in zip(self.K + self.V, new["K"] + new["V"]):
                a.copy_(b)
            for k_ in ("slot_pos", "n", "w", "out"):
                getattr(self, k_).copy_(new[k_])
        else:
            for k_, v_ in new.items():
                setattr(self, k_, v_)
        self._K, self._V = self.K, self.V
        self.t = T
        return logits

    def _call(self):
        _COMPILED.get("step", _step)(self.m, self.w, self.n, self.slot_pos, self.K, self.V, self.out)

    def _capture(self):
        if self.dev.type != "cuda":
            return self._call
        if "step" not in _COMPILED:
            _COMPILED["step"] = torch.compile(_step, dynamic=False)
            for k in ("recompile_limit", "cache_size_limit"):
                if getattr(torch._dynamo.config, k, 64) < 64:
                    setattr(torch._dynamo.config, k, 64)
        saved_n, saved_p = self.n.clone(), self.slot_pos.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                                                 # ısınma (derleme)
                try:
                    self._call()
                except Exception as e:  # noqa: BLE001
                    if _COMPILED["step"] is _step:
                        raise
                    print("StaticCache: torch.compile olmadı, derlemesiz graph: %s" % str(e).splitlines()[0][:150])
                    _COMPILED["step"] = _step
                    self._call()
                self.n.copy_(saved_n)
                self.slot_pos.copy_(saved_p)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._call()
        self.n.copy_(saved_n)
        self.slot_pos.copy_(saved_p)
        return self.graph.replay

    @torch.no_grad()
    def step(self, w):
        """w (B,) bir sonraki token -> (B, V) logit (out tamponu; bir sonraki adımda üzerine yazılır)."""
        self.t += 1
        assert self.t <= self.limit, "StaticCache: tampon aşıldı (%d > %d)" % (self.t, self.limit)
        self.w.copy_(w.view(-1))
        if self.run is None:
            self.run = self._capture()
        self.run()
        return self.out


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, eos_id=None, generator=None):
    """Önbellekli üretim; RGNModel.generate ile aynı arayüz ve (greedy'de) aynı sonuç."""
    B, T = idx.shape
    probe = StaticCache(model, B, max_new_tokens)
    limit = min(model.cfg.max_len, next(t for k in range(40) for t in (BUCKET << k, (3 * BUCKET // 2) << k)
                                        if t >= T + max_new_tokens))
    key = (id(model), B, limit, probe.dtype)
    cache = _POOL.get(key) or _POOL.setdefault(key, probe)
    cache.positions = max_new_tokens
    logits = cache.prefill(idx)
    outs = [idx]
    for i in range(max_new_tokens):
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
        else:
            nxt = logits.argmax(-1, keepdim=True)
        outs.append(nxt)
        if (eos_id is not None and bool((nxt == eos_id).all())) or i == max_new_tokens - 1:
            break
        logits = cache.step(nxt[:, 0])
    return torch.cat(outs, dim=1)
