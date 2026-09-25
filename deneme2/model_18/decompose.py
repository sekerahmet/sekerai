# -*- coding: utf-8 -*-
"""decompose -- TEK TOKEN DOKUMU.  Okuma araci; modelin hesabina girmez.

Kullanici, 25 Eylul: "Tek token dökümü mantıklı bence hatta en kuvvetli ölçüm olabilir."

Bir konumdaki hesap yalniz toplama:
    C_m = C + katman 0 + attention + katman 1..3        puan(w) = 2 e^S_p cos(C_m, P[w]) + sabit
Bu yuzden iki kelimenin puan farki parcalara TAM ayrilir:
    s_a - s_b = 2 e^S_p / |C_m| * sum_parca <parca, P[a] - P[b]>
Zincir de gecmis kelimelere ayrilir: C_order'da lam^yas roll^yas P[w_i], C_content'te solma * beta_w * P[w_i].
Defter olasilikta karisir, p = (1 - gate) softmax(puan) + gate p_cache: payi ayri yazilir.
Tepe ayari icin: relative zincir + C_content + C_M_NORM (squared).  Kapi: tests.t_decompose.  Her kullanimda verify:
parcalarin toplami modelin kendi hesabiyla tutmazsa durur (model_18'e yeni parca eklenirse burasi da genisletilir).
Hazir teshis (istem seti, kopya ve dar secim, ozet, gezgin, pakete gore kayit): diagnose.py.

    python decompose.py <paket.pt> "<istem>" [adim]     acgozlu uretir, her secimin dokumunu basar
"""
import sys
import weakref

import numpy as np
import torch
import torch.nn.functional as F

import data_stories as DS

BANNED = (DS.PAD_TOKEN, DS.UNK_TOKEN)      # DS.generate'in yasaklari


def _mult(m):
    """Puanin carpani: e^S_p (ogrenilen) ya da S_p."""
    return float(m.S_p.exp()) if m.S_p_learned else float(m.S_p)


@torch.no_grad()
def forward_parts(m, tokens):
    """tokens (T,) -> butun konumlarin parcalari:
         C (T,d); layers: katman basina ids, w (T,k) secilen vektorler ve agirliklari, delta (T,d)
         A (H,T,T) dikkat; ov (H,T,d) bas h'nin konum j'den getirdigi (dikkat agirligindan once); attn (T,d)
         C_m (T,d); score (T,n) ham puan; logp (T,n) defterle karisik log p
         gate (T,); cache_j, cache_next, cache_w (T,k): defterin konumlari, getirdigi kelime, agirligi
       Parcalarin toplami modelin C_m'si, logp == scoreboard (tests: t_decompose)."""
    assert m.chain == "relative" and m.c_content and m.c_m_norm and m.squared, "dokum tepe ayari icin"
    tok, T = tokens[None], tokens.shape[0]
    C = m.C(tok)[0]
    C_m, R = C, {"C": C, "layers": []}
    for i, layer in enumerate(m.V):
        D_s2, ids = layer._distance(C_m, None).topk(layer.active, dim=-1, largest=False)
        w = torch.softmax(-D_s2 * layer.S_v.exp(), -1)
        delta = torch.einsum("tk,tkd->td", w, (layer.finish - layer.start)[ids])
        R["layers"].append({"ids": ids, "w": w, "delta": delta})
        C_m = C_m + delta
        if m.attn is not None and i == m.attn_after:
            L = m.attn
            A = L.weights(C[None], C_m[None])[0]
            v = L.W_v(F.normalize(C_m if L.value == "state" else m.P[tokens], dim=-1)).view(T, L.heads, L.dim)
            ov = torch.einsum("dhk,jhk->hjd", L.W_o.weight.view(-1, L.heads, L.dim), v)
            R.update(A=A, ov=ov, attn=torch.einsum("htj,hjd->td", A, ov))
            C_m = C_m + R["attn"]
    if m.direct_chain != "all":                         # cikis zincirin dogrudan oyunu tasimaz (model_18.DIRECT_CHAIN)
        R["direct"] = m.direct_part(C[None], tokens[None])[0]
        C_m = C_m - R["direct"]
    score = m.score(C_m[None])
    R.update(C_m=C_m, score=score[0])
    if m.cache is None:
        R["logp"] = torch.log_softmax(score[0], -1)
        return R
    R["logp"] = m.cache(tok, C[None], C_m[None], score)[0]
    gate, W_c, nxt = m.cache.parts(tok, C[None], C_m[None])
    Cn, pos = F.normalize(C, dim=-1), torch.arange(T)
    sim = (Cn @ Cn.T).masked_fill(~(pos[None, :] < pos[:, None] - m.cache.skip), -2.0)
    R.update(gate=gate[0], cache_w=W_c[0], cache_next=nxt[0], cache_j=sim.topk(W_c.shape[-1], dim=-1).indices)
    return R


@torch.no_grad()
def chain_terms(m, tokens, t):
    """Zincirde konum i'nin (i = 0..t) C_t'ye payi -> (order (t+1, d_order), content (t+1, d_content)), float64.
    Toplamlari C_t."""
    x = m.P[tokens[:t + 1]].double()
    d_o = m.d_order
    age = t - torch.arange(t + 1)
    shifted = (torch.arange(d_o)[None, :] - age[:, None]) % d_o             # roll^yas x [j] = x[j - yas]
    order = (m.lam ** age.double())[:, None] * x[:, :d_o].gather(1, shifted)
    lam, beta = m.lam_beta(tokens[:t + 1])
    L = lam.double().log().cumsum(0)                                         # solma(i -> t) = e^(L_t - L_i)
    return order, (L[t] - L).exp() * beta.double() * x[:, d_o:]


@torch.no_grad()
def explain(m, R, tokens, t, a, b=None):
    """Konum t'nin tahmini (sonraki token): a'nin b'ye karsi puan farki, parca parca; b None: sozluk ortalamasina
    karsi.  -> parts {ad: pay} (zincir_order, zincir_content, katmanlar, attention baslari), chain (t+1, 2)
    gecmis konum basina, vectors [katman] (k, 3) = (vektor, agirlik, pay), heads (H, t+1) bas x konum,
    total = s_a - s_b.  Paylarin toplami total (tests: t_decompose)."""
    P = m.P.double()
    u = P[a] - (P.mean(0) if b is None else P[b])
    C_m = R["C_m"][t].double()
    scale = 2 * _mult(m) / float(C_m.norm())
    order, content = chain_terms(m, tokens, t)
    d_o = m.d_order
    chain = scale * torch.stack([order @ u[:d_o], content @ u[d_o:]], 1)
    if m.direct_chain == "none":                        # cikistan dusen zincir payi dokumde de yok
        chain = torch.zeros_like(chain)
    elif m.direct_chain == "no_self":
        chain[t] = 0
    out = {"parts": {"chain_order": float(chain[:, 0].sum()), "chain_content": float(chain[:, 1].sum())},
           "chain": chain, "vectors": [], "total": scale * float(C_m @ u)}
    for i, (layer, Ly) in enumerate(zip(m.V, R["layers"])):
        ids = Ly["ids"][t]
        c = scale * Ly["w"][t].double() * ((layer.finish - layer.start)[ids].double() @ u)
        out["vectors"].append(torch.stack([ids.double(), Ly["w"][t].double(), c], 1))
        out["parts"]["layer_%d" % i] = float(c.sum())
        if "A" in R and i == m.attn_after:
            heads = scale * R["A"][:, t, :t + 1].double() * (R["ov"][:, :t + 1].double() @ u)
            out["heads"] = heads
            for h in range(heads.shape[0]):
                out["parts"]["head_%d" % h] = float(heads[h].sum())
    return out


def copy_length(tokens, i):
    """tokens[i] ile biten en uzun parcanin (tokens[i] dahil) daha once, i'den once biterek gectigi boy.
    0: tokens[i] hic gecmemis.  4 ve ustu: model onceki bir parcayi kopyaliyor."""
    best = 0
    for e in range(i):
        n = 0
        while n <= e and tokens[e - n] == tokens[i - n]:
            n += 1
        best = max(best, n)
    return best


@torch.no_grad()
def logp_cut(m, tokens, cut=None):
    """MUDAHALE: zincirin puana DOGRUDAN oyu cikarilmis log p (T, n) -- explain'deki chain_* paylari duser.  Katmanlar,
    attention ve defter zinciri gormeye devam eder.  cut: None (mudahale yok, scoreboard ile ayni), "chain" (butun
    zincir), "age0" (yalniz simdiki kelimenin kendi terimi: P_order[w_t] ve beta_w * P_content[w_t])."""
    assert cut is None or m.direct_chain == "all", "zincirin oyu modelde zaten dusmus (direct_chain)"
    tok = tokens[None]
    C = m.C(tok)
    C_m = m._layers(C, tokens=tok)[0]
    s = m.score(C_m)
    if cut is not None:
        if cut == "chain":
            d = C
        elif cut == "age0":
            x = m.P[tok]
            d = torch.cat([x[..., :m.d_order], m.lam_beta(tok)[1] * x[..., m.d_order:]], -1)
        else:
            raise ValueError(cut)
        s = s - 2 * _mult(m) / C_m.norm(dim=-1, keepdim=True) * (d @ m.P.T)
    return (m.cache(tok, C, C_m, s) if m.cache is not None else torch.log_softmax(s, -1))[0]


@torch.no_grad()
def greedy(m, prompt, vocab, token_index, steps=60, banned=BANNED, logp=None, forced=None):
    """DS.generate'in sicaklik 0 yolu, token kimlikleriyle -> (tokens (T,), n_prompt).  <eos> ile durur.
    logp(m, tokens (T,)) -> (T, n): mudahaleli puan (None: m.scoreboard).  forced {k: token}: uretilen k. token
    (0'dan) acgozlu secim yerine zorlanir -- catal cevirme."""
    w = [token_index[DS.EOS_TOKEN]] + [token_index.get(x, token_index[DS.UNK_TOKEN])
                                       for x in DS.tokenize(DS.normalize(prompt))]
    n_prompt = len(w)
    y = torch.tensor([token_index[x] for x in banned])
    for k in range(min(steps, m.t_max - len(w))):
        if forced and k in forced:
            c = int(forced[k])
        else:
            x = torch.tensor(w)
            p = (m.scoreboard(x[None])[0] if logp is None else logp(m, x))[-1]
            c = int(torch.log_softmax(p.index_fill(0, y, -float("inf")), -1).argmax())
        w.append(c)
        if c == token_index[DS.EOS_TOKEN]:
            break
    return torch.tensor(w), n_prompt


@torch.no_grad()
def verify(m, R, tokens, tol=1e-4):
    """Dokum modelin kendisiyle tutuyor mu: parcalarin toplami m'in C_m'si, log p'si m.scoreboard.  Tutmazsa DURUR --
    model_18'e yeni bir parca eklendiyse forward_parts'a da eklenmeli (bulgu koda aittir, kural 11)."""
    tok = tokens[None]
    C_m, logp = m.move(tok)[0][0], m.scoreboard(tok)[0]
    dC = float((R["C_m"] - C_m).abs().max()) / (1 + float(C_m.abs().max()))
    dp = float((R["logp"] - logp).abs().max()) / (1 + float(logp.abs().max()))
    if not (dC < tol and dp < tol):
        raise AssertionError("dokum modelle tutmuyor: C_m farki %.1e, log p farki %.1e (goreli)" % (dC, dp))


@torch.no_grad()
def trajectory(m, tokens, n_prompt, banned_ids):
    """Uretilen her token icin bir satir: secilen a, p, fark (log p, ikinci adaya karsi), ikinci b, ilk 5, entropi,
    kopya boyu, argmax mi, gate, p_cache(a), defterin farka katkisi, parca paylari (a - b).
    banned_ids: uretimde yasak kimlikler; olasiliklar DS.generate gibi onlarsiz yeniden normalize.
    Her cagrida verify: dokum modelle tutmazsa satir yazilmaz."""
    R = forward_parts(m, tokens)
    verify(m, R, tokens)
    lp = torch.log_softmax(R["logp"].index_fill(-1, banned_ids, -float("inf")), -1)
    rows = []
    for t in range(n_prompt - 1, len(tokens) - 1):
        a = int(tokens[t + 1])
        top = lp[t].topk(6)
        b = next(int(x) for x in top.indices if int(x) != a)
        ex = explain(m, R, tokens, t, a, b)
        diff = float(R["score"][t, a] - R["score"][t, b])
        assert abs(ex["total"] - diff) <= 1e-3 * (1 + abs(diff)), "konum %d: parcalar %.4f, puan farki %.4f" % (
            t, ex["total"], diff)
        row = {"t": t, "a": a, "b": b, "p": float(lp[t, a].exp()), "p_b": float(lp[t, b].exp()),
               "margin": float(lp[t, a] - lp[t, b]), "argmax": int(top.indices[0]) == a,
               "top": [(int(i), float(v.exp())) for i, v in zip(top.indices[:5], top.values[:5])],
               "entropy": float(-(lp[t].exp() * lp[t].clamp_min(-1e4)).sum()),
               "copy": copy_length(tokens.tolist(), t + 1), "model_margin": ex["total"], "parts": ex["parts"]}
        if "gate" in R:
            p_cache = torch.zeros(lp.shape[-1]).scatter_add_(0, R["cache_next"][t], R["cache_w"][t])
            row.update(gate=float(R["gate"][t]), p_cache=float(p_cache[a]),
                       cache_shift=row["margin"] - ex["total"])
        rows.append(row)
    return R, rows


def show(m, R, tokens, vocab, t, a, b, log=print, top=4):
    """Konum t'de a'nin b'ye karsi dokumu, okunur."""
    word = lambda i: vocab[int(i)]
    ex = explain(m, R, tokens, t, a, b)
    log("  PARCALAR (%s - %s, logit):  toplam %+.2f  =  %s" % (word(a), word(b), ex["total"], "  ".join(
        "%s %+.2f" % (k, v) for k, v in ex["parts"].items())))
    chain = ex["chain"].sum(1)
    for i in chain.abs().argsort(descending=True)[:top].tolist():
        log("    zincir   konum %3d %-12s yas %3d  %+.2f  (sira %+.2f  icerik %+.2f)" % (
            i, word(tokens[i]), t - i, chain[i], ex["chain"][i, 0], ex["chain"][i, 1]))
    if "heads" in ex:
        for h in range(ex["heads"].shape[0]):
            hc = ex["heads"][h]
            best = hc.abs().argsort(descending=True)[:3].tolist()
            log("    bas %d  %+.2f   %s" % (h, hc.sum(), "   ".join(
                "konum %d %s (A %.2f) %+.2f" % (j, word(tokens[j]), R["A"][h, t, j], hc[j]) for j in best)))
    for i, v in enumerate(ex["vectors"]):
        best = v[:, 2].abs().argsort(descending=True)[:3].tolist()
        log("    katman %d %+.2f   %s" % (i, v[:, 2].sum(), "   ".join(
            "v%d (w %.2f) %+.2f [%s]" % (int(v[r, 0]), v[r, 1], v[r, 2], " ".join(pushes(m, i, int(v[r, 0]), vocab)))
            for r in best)))
    if "gate" in R:
        log("    defter  gate %.3f   %s" % (R["gate"][t], "   ".join(
            "konum %d -> %s %.2f" % (j, word(n), w) for j, n, w in
            zip(R["cache_j"][t].tolist(), R["cache_next"][t].tolist(), R["cache_w"][t].tolist()) if w > 0.05)))


_PUSHES = weakref.WeakKeyDictionary()      # model -> katman basina vektorlerin en cok ittigi kelimeler


def pushes(m, layer, vector, vocab, k=3):
    """Vektorun (finish - start) en cok ittigi k kelime: <V, P[w]> en buyuk."""
    table = _PUSHES.setdefault(m, {})
    if layer not in table:
        L = m.V[layer]
        table[layer] = ((L.finish - L.start).detach() @ m.P.T).topk(k, dim=-1).indices
    return [vocab[int(i)] for i in table[layer][vector]]


def main():
    from model_18 import PV
    path, prompt = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    k = torch.load(path, weights_only=False, map_location="cpu")
    m, vocab = PV.from_package(k), list(k["vocab"])
    ix = {a: i for i, a in enumerate(vocab)}
    tokens, n_prompt = greedy(m, prompt, vocab, ix, steps)
    R, rows = trajectory(m, tokens, n_prompt, torch.tensor([ix[x] for x in BANNED]))
    print("%s  adim %s\nISTEM %s\nURETILEN %s\n" % (path, k.get("step"), prompt,
                                                    DS.decode(np.array(tokens[n_prompt:]), vocab)))
    print("%5s %-12s %6s %7s %-12s %5s %6s  parcalar (secilen - ikinci): %s" % (
        "konum", "secilen", "p", "fark", "ikinci", "kopya", "gate", " ".join(rows[0]["parts"]) if rows else ""))
    for r in rows:
        print("%5d %-12s %6.3f %7.2f %-12s %5d %6.3f  %s" % (
            r["t"] + 1, vocab[r["a"]], r["p"], r["margin"], vocab[r["b"]], r["copy"], r.get("gate", 0),
            " ".join("%+.1f" % v for v in r["parts"].values())))
    for r in sorted(rows, key=lambda r: r["margin"])[:5]:
        print("\n== konum %d  %s -> %s (fark %.2f)" % (r["t"] + 1, " ".join(vocab[int(x)] for x in tokens[max(0, r["t"] - 8):r["t"] + 1]),
                                                     vocab[r["a"]], r["margin"]))
        show(m, R, tokens, vocab, r["t"], r["a"], r["b"])


if __name__ == "__main__":
    main()
