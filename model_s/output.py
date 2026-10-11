"""output -- Model S Parca 10: frekansla carpanlara ayrilmis cikis (kullanici, 11 Ekim: "bir transformer mimarisi değil bir strata
mimarisi kuruyorsun").  Profil (d1024): cikis katmani adimin %18,7'si; veri (s000): en sik 4.096 token hedeflerin %80'i.

    p(w) = p_bas(w | h)                                  w bas kumesinde (en sik K0 token)
    p(w) = p_bas(c_j | h) * p_j(w | h)                   w kuyruk j'de;  p_j = softmax_j((h P_j) . (E_w Q_j))

Zincir kurali: tam normlu dagilim (toplam 1).  Bas tam d'de (bagli E satirlari + kuyruk isaret vektorleri C), kuyruk j d_j
boyutta (E bagli kalir: kuyruk token vektoru E_w Q_j).  Egitimde kuyruk yalniz hedefi o kumede olan satirlar icin.
"""
import torch
import torch.nn.functional as F_


class FactorizedOutput(torch.nn.Module):
    def __init__(self, d, vocab, cuts=(4096, 16384), dims=None):
        """cuts: kume sinirlari (siklik sirasinda), dims: kuyruk boyutlari (varsayilan d / 2, d / 8)."""
        super().__init__()
        self.vocab, self.cuts = vocab, tuple(int(c) for c in cuts)
        self.edges = (0,) + self.cuts + (vocab,)
        assert all(a < b for a, b in zip(self.edges, self.edges[1:])), "cuts artan ve < vocab"
        self.dims = tuple(dims or (d // 2, d // 8))[:len(self.cuts)]
        assert len(self.dims) == len(self.cuts), "her kuyruga bir boyut"
        self.register_buffer("order", torch.arange(vocab))                 # siklik sirasi -> token (train.py doldurur)
        self.register_buffer("rank", torch.arange(vocab))                  # token -> siklik sirasi
        self.C = torch.nn.Parameter(torch.randn(len(self.cuts), d) * 0.02)  # kuyruk isaretleri (bas softmax'inda)
        self.P = torch.nn.ParameterList(torch.nn.Parameter(torch.linalg.qr(torch.randn(d, k))[0]) for k in self.dims)
        self.Q = torch.nn.ParameterList(torch.nn.Parameter(p.detach().clone()) for p in self.P)  # P Q^T = izdusum: E'ye yakin

    @torch.no_grad()
    def set_order(self, counts):
        """counts (vocab,) -> siklik sirasi (azalan, esitlikte kucuk id once)."""
        order = torch.as_tensor(counts, dtype=torch.float64).neg().argsort(stable=True)
        self.order.copy_(order.to(self.order.device))
        self.rank[self.order] = torch.arange(self.vocab, device=self.rank.device)

    def _head(self, h, E):
        w = torch.cat([E[self.order[:self.cuts[0]]], self.C.to(E.dtype)])
        return F_.log_softmax((h @ w.T).float(), -1)                       # (N, K0 + kuyruk sayisi)

    def _tail(self, h, E, j):
        a, b = self.edges[j + 1], self.edges[j + 2]
        P, Q = self.P[j].to(h.dtype), self.Q[j].to(E.dtype)
        return F_.log_softmax(((h @ P) @ (E[self.order[a:b]] @ Q).T).float(), -1)

    def logprobs(self, h, E):
        """h (..., d) -> tam log p (..., vocab), token id sirasinda."""
        lead = h.shape[:-1]
        h = h.reshape(-1, h.shape[-1])
        hd = self._head(h, E)
        K0 = self.cuts[0]
        parts = [hd[:, :K0]] + [self._tail(h, E, j) + hd[:, K0 + j:K0 + j + 1] for j in range(len(self.cuts))]
        out = torch.empty(h.shape[0], self.vocab, device=h.device, dtype=hd.dtype)
        out[:, self.order] = torch.cat(parts, -1)
        return out.view(*lead, self.vocab)

    def loss(self, h, E, target):
        """h (N, d), target (N,) (-100 hedefsiz) -> hedefli konumlarda ortalama -log p; kuyruk yalniz hedefi o kumede olan satirlarda."""
        ok = target >= 0
        return self.nll(h[ok], E, target[ok]).mean()

    def nll(self, h, E, t):
        """h (N, d), t (N,) gecerli hedef -> -log p[t] (N,)."""
        r = self.rank[t]
        hd = self._head(h, E)
        K0 = self.cuts[0]
        lp = hd.gather(-1, r.clamp(max=K0 - 1)[:, None])[:, 0]
        for j in range(len(self.cuts)):
            a, b = self.edges[j + 1], self.edges[j + 2]
            m = (r >= a) & (r < b)
            idx = m.nonzero()[:, 0]
            tl = self._tail(h[idx], E, j).gather(-1, (r[idx] - a)[:, None])[:, 0] + hd[idx, K0 + j]
            lp = lp.index_put((idx,), tl)
        return -lp
