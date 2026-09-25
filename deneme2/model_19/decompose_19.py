# -*- coding: utf-8 -*-
"""decompose_19 -- model_19'un TEK TOKEN DECOMPOSE'u.  Okuma araci; modelin hesabina girmez.

Bir konumdaki hesap yalniz toplama:
    C_m = C + katman 0 + attention 1 + katman 1 + katman 2 + attention 2 + katman 3
    puan(w) = P[w] . z,   z = 2 e^S_p C_m / |C_m|  +  R_PC(C)
Bu yuzden iki kelimenin puan farki parcalara TAM ayrilir:
    s_a - s_b = 2 e^S_p / |C_m| * sum_parca <parca, P[a] - P[b]>  +  <R_PC(C), P[a] - P[b]>
Zincir gecmis kelimelere ayrilir (sira: lam^yas kaydir^yas E[w_i]; icerik: sonme * beta_w * E[w_i]); R_PC dogrusal
oldugu icin o da gecmis kelime basina ayrilir.  Attention bas ve kaynak konum basina.  Defter olasilikta karisir,
p = (1 - gate) softmax(puan) + gate p_defter: payi ayri yazilir.
Her kullanimda verify: parcalarin toplami modelin kendi hesabiyla tutmazsa DURUR (modele yeni parca eklenirse burasi
da genisletilir).  Hazir teshis (istem seti, ozet, gezgin, kayit): diagnose_19.py.

    python decompose_19.py <paket.pt> "<istem>" [adim]     acgozlu uretir, en dar secimlerin decompose'unu basar
"""
import sys
import weakref

import numpy as np
import torch
import torch.nn.functional as F

import data_stories_19 as DS

BANNED = (DS.PAD_TOKEN, DS.UNK_TOKEN)      # uretimde yasak (sozlukte varsa)


def banned_ids(token_index, device="cpu"):
    """Sozlukte bulunan yasak token'larin kimlikleri (matematik sozlugunde yok)."""
    return torch.tensor([token_index[x] for x in BANNED if x in token_index], dtype=torch.long, device=device)


def _mult(m):
    return float(m.S_p.exp())


def _relation_matrix(rel):
    """Relation -> tam matris (d, d), float64: R(x) = x @ M^T."""
    return (rel.M if rel.M is not None else rel.A @ rel.B.T).detach().double()


@torch.no_grad()
def forward_parts(m, tokens):
    """tokens (T,) -> butun konumlarin parcalari:
         C (T,d); layers: katman basina ids, w (T,k), delta (T,d)
         attns: attention basina after (katman), A (H,T,T), ov (H,T,d) bas h'nin j'den getirdigi, out (T,d)
         C_m (T,d) modelin kendi durumu; C_m_parts (T,d) parcalarin toplami; readout (T,d) R_PC(C) ya da None
         score (T,n) ham puan; logp (T,n) defterle karisik; defter: gate, gate_dir, gate_sim (T,), cache_w (T,T),
         cache_next (T,)
       Durum modelin KENDI cagrilariyla ilerler (secimler birebir onunki); parcalar yanda hesaplanir."""
    tok, T = tokens[None], tokens.shape[0]
    C = m.chain(tok)
    C_m, R = C, {"C": C[0], "layers": [], "attns": []}
    for i, layer in enumerate(m.moves):
        D2 = layer._distance(C_m)
        if layer.active >= D2.shape[-1]:
            ids = torch.arange(D2.shape[-1], device=D2.device).expand(D2.shape)
            w = torch.softmax(-D2 * layer.S_v.exp(), -1)
        else:
            d, ids = D2.topk(layer.active, dim=-1, largest=False)
            w = torch.softmax(-d * layer.S_v.exp(), -1)
        delta = torch.einsum("tk,tkd->td", w[0], (layer.finish - layer.start)[ids[0]])
        R["layers"].append({"ids": ids[0], "w": w[0], "delta": delta})
        C_m = layer(C_m)[0]
        if i in m.attn_after:
            att = m.attns[m.attn_after.index(i)]
            A = att.weights(C, C_m)[0]
            v = (F.normalize(C_m, dim=-1) @ att.W_v.T)[0].view(T, att.heads, att.dim)
            ov = torch.einsum("dhk,jhk->hjd", att.W_o.view(-1, att.heads, att.dim), v)
            R["attns"].append({"after": i, "A": A, "ov": ov, "out": torch.einsum("htj,hjd->td", A, ov)})
            C_m = att(C, C_m)
    parts = R["C"] + sum(L["delta"] for L in R["layers"]) + sum(a["out"] for a in R["attns"])
    score = m.point(C_m, C) @ m.P.T
    R.update(C_m=C_m[0], C_m_parts=parts, score=score[0],
             readout=None if m.readout is None else m.readout(C)[0])
    if m.ledger is None:
        R["logp"] = torch.log_softmax(score[0], -1)
        return R
    R["logp"] = m.ledger.mix(tok, C, C_m, score)[0]
    W_c, sim_max, valid = m.ledger.neighbors(C)
    gate_dir, gate_sim = m.ledger.gate_inputs(C_m, sim_max)
    R.update(gate=m.ledger.gate(C_m, sim_max, valid)[0], gate_dir=gate_dir[0], gate_sim=gate_sim[0],
             cache_w=W_c[0], cache_next=torch.roll(tokens, -1, 0))
    return R


@torch.no_grad()
def chain_terms(m, tokens, t):
    """Zincirde konum i'nin (i = 0..t) C_t'ye payi -> (order (t+1, d_order), content (t+1, d_content)), float64.
    Toplamlari C_t.  Kelimeler E ile (embed kapaliyken P)."""
    x = m.embed(tokens[None, :t + 1])[0].double()
    d_o = m.d_order
    age = t - torch.arange(t + 1, device=tokens.device)
    shifted = (torch.arange(d_o, device=tokens.device)[None, :] - age[:, None]) % d_o     # kaydir^yas x [j] = x[j - yas]
    order = (m.lam ** age.double())[:, None] * x[:, :d_o].gather(1, shifted)
    lam, beta = m.lam_beta(tokens[:t + 1])
    L = lam.double().log().cumsum(0)                                         # sonme(i -> t) = e^(L_t - L_i)
    return order, (L[t] - L).exp() * beta.double() * x[:, d_o:]


@torch.no_grad()
def explain(m, R, tokens, t, a, b=None):
    """Konum t'nin tahmini: a'nin b'ye karsi puan farki parca parca; b None: sozluk ortalamasina karsi.
    -> parts {ad: pay}: chain_order, chain_content, layer_i, attn<k>_head_<h>, readout.
       chain (t+1, 2) ve readout_words (t+1,) gecmis konum basina; vectors [katman] (k, 3) = (vektor, agirlik, pay);
       heads [attention] (H, t+1) bas x kaynak konum; total = s_a - s_b.  Paylarin toplami total."""
    P = m.P.double()
    u = P[a] - (P.mean(0) if b is None else P[b])
    C_m = R["C_m"][t].double()
    scale = 2 * _mult(m) / float(C_m.norm())
    order, content = chain_terms(m, tokens, t)
    d_o = m.d_order
    chain = scale * torch.stack([order @ u[:d_o], content @ u[d_o:]], 1)
    out = {"parts": {"chain_order": float(chain[:, 0].sum()), "chain_content": float(chain[:, 1].sum())},
           "chain": chain, "vectors": [], "heads": [], "total": scale * float(C_m @ u)}
    for i, (layer, L) in enumerate(zip(m.moves, R["layers"])):
        ids = L["ids"][t]
        c = scale * L["w"][t].double() * ((layer.finish - layer.start)[ids].double() @ u)
        out["vectors"].append(torch.stack([ids.double(), L["w"][t].double(), c], 1))
        out["parts"]["layer_%d" % i] = float(c.sum())
        for k, att in enumerate(R["attns"]):
            if att["after"] == i:
                heads = scale * att["A"][:, t, :t + 1].double() * (att["ov"][:, :t + 1].double() @ u)
                out["heads"].append(heads)
                for h in range(heads.shape[0]):
                    out["parts"]["attn%d_head_%d" % (k, h)] = float(heads[h].sum())
    if R["readout"] is not None:
        Mr = _relation_matrix(m.readout)
        words = torch.cat([order, content], 1) @ Mr.T @ u                   # R_PC(C_t) = sum_i R_PC(terim_i)
        out["readout_words"] = words
        out["parts"]["readout"] = float(words.sum())
        out["total"] += float(R["readout"][t].double() @ u)
    return out


def copy_length(tokens, i):
    """tokens[i] ile biten en uzun parcanin (tokens[i] dahil) daha once, i'den once biterek gectigi boy.
    4 ve ustu: model onceki bir parcayi kopyaliyor."""
    best = 0
    for e in range(i):
        n = 0
        while n <= e and tokens[e - n] == tokens[i - n]:
            n += 1
        best = max(best, n)
    return best


@torch.no_grad()
def greedy(m, prompt, vocab, token_index, steps=60, forced=None):
    """Sicaklik 0 uretim, token kimlikleriyle -> (tokens (T,), n_prompt).  <eos> ile durur.  Yasaklar uretilmez.
    forced {k: token}: uretilen k. token (0'dan) zorlanir -- catal cevirme.  Istem DS.tokenize ile bolunur
    (matematikte rakamlar boslukla ayrik yazilir: "4 7 2 + 1 8 2 =")."""
    dev = m.P.device
    unk = token_index.get(DS.UNK_TOKEN)
    w = [token_index[DS.EOS_TOKEN]] + [token_index[x] if unk is None else token_index.get(x, unk)
                                       for x in DS.tokenize(DS.normalize(prompt))]
    n_prompt = len(w)
    y = banned_ids(token_index, dev)
    for k in range(min(steps, m.t_max - len(w))):
        if forced and k in forced:
            c = int(forced[k])
        else:
            p = m.scoreboard(torch.tensor(w, device=dev)[None])[0, -1]
            c = int(torch.log_softmax(p.index_fill(0, y, -float("inf")), -1).argmax())
        w.append(c)
        if c == token_index[DS.EOS_TOKEN]:
            break
    return torch.tensor(w, device=dev), n_prompt


@torch.no_grad()
def verify(m, R, tokens, tol=1e-4):
    """decompose modelin kendisiyle tutuyor mu: PARCALARIN toplami m'in C_m'si, log p'si m.scoreboard.  Tutmazsa DURUR."""
    tok = tokens[None]
    C_m, logp = m._layers(m.chain(tok))[0][0], m.scoreboard(tok)[0]
    dC = float(torch.maximum((R["C_m"] - C_m).abs().max(), (R["C_m_parts"] - C_m).abs().max())) / (1 + float(C_m.abs().max()))
    dp = float((R["logp"] - logp).abs().max()) / (1 + float(logp.abs().max()))
    if not (dC < tol and dp < tol):
        raise AssertionError("decompose modelle tutmuyor: C_m farki %.1e, log p farki %.1e (goreli)" % (dC, dp))


@torch.no_grad()
def trajectory(m, tokens, n_prompt, banned):
    """Uretilen her token icin bir satir: secilen a, p, fark (log p, ikinci adaya karsi), ikinci b, ilk 5, entropi,
    kopya boyu, gate ve iki girdisi, p_defter(a), defterin farka katkisi, parca paylari (a - b).
    banned: uretimde yasak kimlikler; olasiliklar onlarsiz yeniden normalize.  Her cagrida verify."""
    R = forward_parts(m, tokens)
    verify(m, R, tokens)
    lp = torch.log_softmax(R["logp"].index_fill(-1, banned.to(R["logp"].device), -float("inf")), -1)
    rows = []
    for t in range(n_prompt - 1, len(tokens) - 1):
        a = int(tokens[t + 1])
        top = lp[t].topk(min(6, lp.shape[-1]))
        b = next(int(x) for x in top.indices if int(x) != a)
        ex = explain(m, R, tokens, t, a, b)
        diff = float(R["score"][t, a] - R["score"][t, b])
        assert abs(ex["total"] - diff) <= 1e-3 * (1 + abs(diff)), "konum %d: parcalar %.4f, puan farki %.4f" % (
            t, ex["total"], diff)
        assert abs(sum(ex["parts"].values()) - ex["total"]) <= 1e-3 * (1 + abs(diff)), "konum %d: parca toplami" % t
        row = {"t": t, "a": a, "b": b, "p": float(lp[t, a].exp()), "p_b": float(lp[t, b].exp()),
               "margin": float(lp[t, a] - lp[t, b]), "argmax": int(top.indices[0]) == a,
               "top": [(int(i), float(v.exp())) for i, v in zip(top.indices[:5], top.values[:5])],
               "entropy": float(-(lp[t].exp() * lp[t].clamp_min(-1e4)).sum()),
               "copy": copy_length(tokens.tolist(), t + 1), "model_margin": ex["total"], "parts": ex["parts"]}
        if "gate" in R:
            p_cache = torch.zeros(lp.shape[-1], device=lp.device).scatter_add_(0, R["cache_next"], R["cache_w"][t])
            row.update(gate=float(R["gate"][t]), gate_dir=float(R["gate_dir"][t]), gate_sim=float(R["gate_sim"][t]),
                       p_cache=float(p_cache[a]), cache_shift=row["margin"] - ex["total"])
        rows.append(row)
    return R, rows


def show(m, R, tokens, vocab, t, a, b, log=print, top=4):
    """Konum t'de a'nin b'ye karsi decompose'u, okunur."""
    word = lambda i: vocab[int(i)]
    ex = explain(m, R, tokens, t, a, b)
    log("  PARCALAR (%s - %s, logit):  toplam %+.2f  =  %s" % (word(a), word(b), ex["total"], "  ".join(
        "%s %+.2f" % (k, v) for k, v in ex["parts"].items())))
    chain = ex["chain"].sum(1)
    for i in chain.abs().argsort(descending=True)[:top].tolist():
        log("    zincir   konum %3d %-12s yas %3d  %+.2f  (sira %+.2f  icerik %+.2f)" % (
            i, word(tokens[i]), t - i, chain[i], ex["chain"][i, 0], ex["chain"][i, 1]))
    if "readout_words" in ex:
        rw = ex["readout_words"]
        log("    R_PC     %s" % "   ".join("konum %d %s %+.2f" % (i, word(tokens[i]), rw[i])
                                          for i in rw.abs().argsort(descending=True)[:top].tolist()))
    for k, heads in enumerate(ex["heads"]):
        A = R["attns"][k]["A"]
        for h in range(heads.shape[0]):
            hc = heads[h]
            best = hc.abs().argsort(descending=True)[:3].tolist()
            log("    att%d bas %d  %+.2f   %s" % (k + 1, h, hc.sum(), "   ".join(
                "konum %d %s (A %.2f) %+.2f" % (j, word(tokens[j]), A[h, t, j], hc[j]) for j in best)))
    for i, v in enumerate(ex["vectors"]):
        best = v[:, 2].abs().argsort(descending=True)[:3].tolist()
        log("    katman %d %+.2f   %s" % (i, v[:, 2].sum(), "   ".join(
            "v%d (w %.2f) %+.2f [%s]" % (int(v[r, 0]), v[r, 1], v[r, 2], " ".join(pushes(m, i, int(v[r, 0]), vocab)))
            for r in best)))
    if "gate" in R:
        w = R["cache_w"][t]
        log("    defter  gate %.3f (yon %+.2f, benzerlik %+.2f)   %s" % (
            R["gate"][t], R["gate_dir"][t], R["gate_sim"][t], "   ".join(
                "konum %d -> %s %.2f" % (j, word(R["cache_next"][j]), w[j]) for j in w.argsort(descending=True)[:4].tolist()
                if w[j] > 0.05)))


_PUSHES = weakref.WeakKeyDictionary()      # model -> katman basina hareketlerin en cok ittigi kelimeler


def pushes(m, layer, vector, vocab, k=3):
    """Hareketin (finish - start) en cok ittigi k kelime: <V, P[w]> en buyuk."""
    table = _PUSHES.setdefault(m, {})
    if layer not in table:
        L = m.moves[layer]
        table[layer] = ((L.finish - L.start).detach() @ m.P.T).topk(min(k, m.n), dim=-1).indices
    return [vocab[int(i)] for i in table[layer][vector]]


def main():
    from model_19 import PointRelation
    path, prompt = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    k = torch.load(path, weights_only=False, map_location="cpu")
    m, vocab = PointRelation.from_package(k), list(k["vocab"])
    ix = {a: i for i, a in enumerate(vocab)}
    tokens, n_prompt = greedy(m, prompt, vocab, ix, steps)
    R, rows = trajectory(m, tokens, n_prompt, banned_ids(ix))
    print("%s  adim %s\nISTEM %s\nURETILEN %s\n" % (path, k.get("step"), prompt,
                                                    DS.decode(np.array(tokens[n_prompt:].tolist()), vocab)))
    for r in sorted(rows, key=lambda r: r["margin"])[:5]:
        print("\n== konum %d  %s -> %s (fark %.2f)" % (
            r["t"] + 1, " ".join(vocab[int(x)] for x in tokens[max(0, r["t"] - 8):r["t"] + 1]), vocab[r["a"]], r["margin"]))
        show(m, R, tokens, vocab, r["t"], r["a"], r["b"])


if __name__ == "__main__":
    main()
