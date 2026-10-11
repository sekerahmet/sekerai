"""strata_x -- Strata-x: dil denklemini bilgiye uyarlanan x izgarasinda cozmek (TASARIM.md s0; kullanici, 11 Ekim: "Token kavramini
... hepsini sorgulatabilirsin").

    token'lar -> yerel kodlayici (dar, sig, pencereli; kendi LM basi) -> token surprizi i_t = -log p_enc(x_t | x_<t)
              -> birim siniri: belge ici birikimli bilgi C_t, floor(C_t / theta) artinca t'de birim biter (nedensel)
              -> birim vektoru up(h_enc[son token]) -> buyuk cozucu (genis, derin, birimler uzerinde tam nedensel) -> g_u
              -> yerel kod cozucu: t'deki token, sonu <= t olan son birimin g'sini (down) + yerel baglami gorur -> x_{t+1}

Nedensellik: i_t yalniz x_<=t'ye bagli; birim sonu e <= t olan birimler x_<=t'den kurulur; global maske birim ekseninde nedensel.
Kayip: kod cozucu CE + kodlayici yardimci CE (yalniz gradyana; gunluk degeri ana CE: main + w (aux - aux.detach())).
Uretim: dict onbellek yolunda her adim tam yeniden hesap (dogru, yavas; hizli yol sonra).
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "deneme2", "v2", "transformer"))
from baseline import BaselineTransformer  # noqa: E402
from data import VOCAB  # noqa: E402
from model import SBlock  # noqa: E402
from operators import Projection  # noqa: E402
from output import FactorizedOutput  # noqa: E402


def _hidden(d):
    return -(-int(8 * d / 3) // 8) * 8


class StrataX(BaselineTransformer):
    def __init__(self, d_local=512, heads_local=8, enc_layers=2, dec_layers=2, d_global=1024, heads_global=16,
                 global_layers=12, theta=8.0, window=256, aux_weight=1.0, vocab_rows=VOCAB, out_cuts=(4096, 16384),
                 plan=0, next_weight=0.0):
        """plan k (HAYAL.md H8): buyuk cozucu birim basina k yuvalik plan (d_global -> k x d_local); kod cozucu token'i yazarken
        kendi yuvasinin plan vektorunu gorur (0: tek vektor, down).  next_weight (H9): sonraki birimin kodlayici vektorunu
        tahmin (kosinus, hedefte gradyan kesik) yardimci kayip agirligi."""
        super().__init__(d_local, 1, heads_local, None, vocab_rows)     # E (d_local), norm, uretim kurallari
        self.plan_k, self.next_weight = int(plan), float(next_weight)
        self.theta, self.window, self.aux_weight = float(theta), int(window), float(aux_weight)
        self.n_enc, self.n_glob = enc_layers, global_layers
        loc = lambda: SBlock(d_local, heads_local, _hidden(d_local))  # noqa: E731
        glo = lambda: SBlock(d_global, heads_global, _hidden(d_global))  # noqa: E731
        self.blocks = torch.nn.ModuleList([loc() for _ in range(enc_layers)] + [glo() for _ in range(global_layers)]
                                          + [loc() for _ in range(dec_layers)])   # Muon: blocks.* (recipe.muon_params)
        self.up = torch.nn.Linear(d_local, d_global, bias=False)
        self.down = torch.nn.Linear(d_global, d_local * max(1, self.plan_k), bias=False)
        self.pred_next = torch.nn.Linear(d_global, d_global, bias=False) if self.next_weight else None
        self.g0 = torch.nn.Parameter(torch.zeros(d_local))               # tamamlanmis birim yokken baglam
        self.enc_norm, self.glob_norm = Projection(d_local), Projection(d_global)
        self.out = FactorizedOutput(d_local, VOCAB, out_cuts) if out_cuts else None
        self._aux = None
        n_l, n_g = enc_layers + dec_layers, global_layers
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:
                i = int(name.split(".")[0])
                n = n_g if self.n_enc <= i < self.n_enc + n_g else n_l
                std = 0.02 / math.sqrt(2 * n) if name.endswith(("w_o.weight", "w_down.weight")) else 0.02
                with torch.no_grad():
                    p.normal_(0.0, std)
        for lin in (self.up, self.down):
            torch.nn.init.normal_(lin.weight, 0.0, 0.02)

    # --- parcalar ---------------------------------------------------------------------------------------------------
    def _nll(self, h, tgt):
        """h (N, d_local) normlu, tgt (N,) gecerli -> -log p (N,)."""
        if self.out is not None:
            return self.out.nll(h, self.E.weight, tgt)
        return torch.nn.functional.cross_entropy(BaselineTransformer._logits(self, h).float(), tgt, reduction="none")

    def _local_mask(self, doc):
        B, T = doc.shape
        i = torch.arange(T, device=doc.device)
        m = (doc[:, :, None] == doc[:, None, :]) & (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
        return m | torch.eye(T, dtype=torch.bool, device=doc.device)        # dolgu kendini gorur

    def units(self, info, doc):
        """info (B, T) token bilgisi, doc (B, T) -> ends (B, T) bool: belge ici floor(C_t / theta) artti mi (t'de birim biter)."""
        real = doc >= 0
        C = torch.cumsum(info * real, 1)
        first = torch.ones_like(real)
        first[:, 1:] = doc[:, 1:] != doc[:, :-1]
        start = torch.cummax(torch.where(first, torch.arange(doc.shape[1], device=doc.device), 0), 1).values
        base = torch.where(start > 0, C.gather(1, (start - 1).clamp(min=0)), torch.zeros_like(C))
        q = torch.floor((C - base) / self.theta)
        prev = torch.where(first, torch.zeros_like(q), torch.roll(q, 1, 1))
        return (q > prev) & real

    def forward_states(self, tokens, pos, doc):
        """-> (h_dec normlu (B, T, d_local), aux CE (skaler), ends (B, T), kodlayici bilgisi (B, T))."""
        B, T = tokens.shape
        m_loc = self._local_mask(doc)
        x = self.E(tokens)
        for blk in self.blocks[:self.n_enc]:
            x = blk(x, pos, m_loc)
        he = x
        hn = self.enc_norm(he)
        nxt = (doc[:, 1:] == doc[:, :-1]) & (doc[:, 1:] >= 0)             # x_{t+1} ayni belgede: i_{t+1} = -log p_enc(x_{t+1})
        nll = torch.zeros(B, T, device=tokens.device)
        nll_v = self._nll(hn[:, :-1][nxt], tokens[:, 1:][nxt])
        nll = nll.index_put((nxt.nonzero()[:, 0], nxt.nonzero()[:, 1] + 1), nll_v.detach().float())
        aux = nll_v.mean() if nll_v.numel() else he.sum() * 0
        ends = self.units(nll, doc)
        # birimler: satir basina sonlar, Umax'a dolgu
        Umax = 64                                                           # 2'nin kuvvetine yuvarla: derleme (dynamic=False)
        while Umax < int(ends.sum(1).max()):                                # en cok ~6 sekil gorur
            Umax *= 2
        r, c = ends.nonzero(as_tuple=True)
        k = torch.cumsum(ends.long(), 1) - 1                               # t'de son tamamlanmis birimin sirasi
        upos_tok = torch.zeros(B, Umax, dtype=torch.long, device=tokens.device)
        valid = torch.zeros(B, Umax, dtype=torch.bool, device=tokens.device)
        upos_tok[r, k[r, c]] = c
        valid[r, k[r, c]] = True
        udoc = torch.where(valid, doc.gather(1, upos_tok), torch.full_like(upos_tok, -1))
        ufirst = torch.ones_like(valid)
        ufirst[:, 1:] = udoc[:, 1:] != udoc[:, :-1]
        ustart = torch.cummax(torch.where(ufirst, torch.arange(Umax, device=tokens.device), 0), 1).values
        upos = torch.arange(Umax, device=tokens.device)[None] - ustart       # belge ici birim sirasi (RoPE)
        z = self.up(he.gather(1, upos_tok[..., None].expand(-1, -1, he.shape[-1])))
        z_in = z
        iu = torch.arange(Umax, device=tokens.device)
        m_g = (udoc[:, :, None] == udoc[:, None, :]) & (iu[:, None] >= iu[None, :]) & valid[:, None, :]
        m_g = m_g | torch.eye(Umax, dtype=torch.bool, device=tokens.device)
        for blk in self.blocks[self.n_enc:self.n_enc + self.n_glob]:
            z = blk(z, upos, m_g)
        g = self.down(self.glob_norm(z))                                    # (B, Umax, k x d_local)
        kc = k.clamp(min=0)
        ok = (k >= 0) & (udoc.gather(1, kc) == doc) & (doc >= 0)
        dl = he.shape[-1]
        gt = g.gather(1, kc[..., None].expand(-1, -1, g.shape[-1]))         # (B, T, k x d_local)
        if self.plan_k:                                                     # yuva = t - son birim sonu (0: sonraki birimin ilk token'i)
            last_end = upos_tok.gather(1, kc)
            slot = (torch.arange(T, device=tokens.device)[None] - last_end).clamp(0, self.plan_k - 1)
            gt = gt.view(B, T, self.plan_k, dl).gather(2, slot[..., None, None].expand(-1, -1, 1, dl))[:, :, 0]
        ctx = torch.where(ok[..., None], gt, self.g0.to(gt.dtype))
        if self.pred_next is not None and self.training:                    # H9: z_u -> kodlayicinin z_{u+1}'i (hedef kesik)
            pairs = valid[:, 1:] & valid[:, :-1] & (udoc[:, 1:] == udoc[:, :-1])
            if pairs.any():
                pr = self.pred_next(self.glob_norm(z[:, :-1]))[pairs].float()
                tg = z_in[:, 1:][pairs].detach().float()
                aux = aux + self.next_weight * (1 - torch.nn.functional.cosine_similarity(pr, tg, -1)).mean()
        x = he + ctx
        for blk in self.blocks[self.n_enc + self.n_glob:]:
            x = blk(x, pos, m_loc)
        return self.norm(x), aux, ends, nll

    # --- BaselineTransformer arayuzu ------------------------------------------------------------------------------------
    def _batch_hidden(self, batch, attn=None):
        h, aux, ends, _ = self.forward_states(batch.tokens, batch.pos, batch.doc)
        self._aux = aux if self.training else None
        self.last_units = ends.sum().item() / max((batch.doc >= 0).sum().item(), 1)   # birim / token (gunluk / teshis)
        return h

    def hidden(self, tokens, pos, attn):
        return self.forward_states(tokens, pos, torch.zeros_like(tokens))[0]

    def _logits(self, h):
        return BaselineTransformer._logits(self, h) if self.out is None else self.out.logprobs(h, self.E.weight)

    def output_loss(self, h, target):
        ok = target >= 0
        main = self._nll(h[ok], target[ok]).mean()
        if self._aux is None:
            return main
        aux, self._aux = self._aux, None
        return main + self.aux_weight * (aux - aux.detach())                # deger = ana CE, gradyan + yardimci CE

    def _step(self, tokens, cache, n):
        """Dict onbellek yolu: onbellek token gecmisi; her adim tam yeniden hesap (dogru; hizli yol sonra)."""
        seq = cache.setdefault("seq", tokens.new_zeros(1, 0))
        seq = torch.cat([seq[:, :n], tokens], 1)
        cache["seq"] = seq
        T = seq.shape[1]
        h = self.forward_states(seq, torch.arange(T, device=seq.device)[None], torch.zeros_like(seq))[0]
        return h[:, -1]

    @torch.no_grad()
    def generate(self, prompts, max_sentences, max_tokens, generator=None, open_last=False, on_token=None,
                 stop_when=None, batch_size=32):
        """Tek tek (dict onbellek, tam yeniden hesap) -- StaticCache bloklarin _qkv'sine dayanir, Strata-x'te yok."""
        import contextlib
        import baseline as BL
        old, BL.STATIC_DECODE = BL.STATIC_DECODE, False
        eager = getattr(torch.compiler, "set_stance", None)                # her adim yeni boy: derlenmis bloklar yeniden derlenmesin
        try:
            with eager("force_eager") if eager else contextlib.nullcontext():
                return super().generate(prompts, max_sentences, max_tokens, generator, open_last, on_token, stop_when, 1)
        finally:
            BL.STATIC_DECODE = old
