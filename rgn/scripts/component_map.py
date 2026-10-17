"""İşlem haritası: her katmanda her bileşenin (karıştırıcı = dikkat / evrişim, MLP, özel nöron, PLE) akışa eklediği çıktı
üç paya ayrılır — işlem neye harcanıyor, daha ucuz bir parça yapabilir mi?  (eğitilmiş modelde; yeniden eğitim yok)

  tablo  : yalnız o anki kelimeye bağlı pay: kelime başına ortalama çıktı T[kelime] (tablo kestirim pencerelerinde; az
           görülen kelime genel ortalamaya büzülür, --shrink).  Bunu bir tablo (PLE / gömme) yapabilir.
  sabit  : bileşen girdisinin sabit nedensel filtreleriyle (spectral_map ile aynı: gecikme 0–3 + bant ortalamaları)
           doğrusal tahmin — konuma bağlı, içeriğe bağlı olmayan iş (evrişim / doğrusal katman yapabilir).
  t+s    : önce tablo, kalan üzerine sabit filtre (sıralı).  İçerik = 1 − R²(t+s): gerçekten dikkat / doğrusal olmayan
           MLP gerekir.
Takas: bileşen yalnız o katmanda tahminle değiştirilince kayıp artışı (konum ≥ qmin): ort (genel ortalama — bileşenin
toplam önemi), tablo, sabit, t+s.  Karıştırıcı ve MLP'nin girdisi kendi norm'u (norm1 / norm2); özel nöron ve PLE için
norm2 çıktısı (PLE'nin kapısı blok çıktısına bakar; yaklaşık).

Kalibrasyon (--truncate): Y pencereleri çıkarımda daraltılınca kayıp artışı (yeniden eğitimsiz); yeniden eğitilmiş
koşularla kıyas (yeniden eğitimsiz ölçü kaybı ne kadar abartıyor?).

Sınır: bugünkü kullanımı ölçer; takas birlikte uyum yüzünden kaybı abartır (evrişim: takas toplamı 0,18 ↔ koşu +0,034).

    python scripts/component_map.py --run fw_s000_d1024L9_yari_epok --truncate 32,64 16,16 8,8
"""
import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from spectral_map import BANDS, DELAYS, features  # noqa: E402

NAMES = {"mix": "karış", "mlp": "mlp", "priv": "özel", "ple": "ple"}
VARIANTS = ("ort", "tablo", "sabit", "t+s")


def components(b):
    """Blok bileşenleri [(ad, girdi)]: girdi 'xa' (norm1) ya da 'xn' (norm2)."""
    out = [("mix", "xa")]
    if b.mlp is not None:
        out.append(("mlp", "xn"))
    if b.priv_names:
        out.append(("priv", "xn"))
    if b.private_ple is not None:
        out.append(("ple", "xn"))
    return out


def mixer(b):
    return b.attn if b.attn is not None else b.conv


def capture(m, l):
    """Katman l'nin bileşen girdileri ve çıktıları için kancalar -> (sözlük, kancalar)."""
    b, cap = m.blocks[l], {}
    hs = [b.norm1.register_forward_hook(lambda mod, i, o: cap.__setitem__("xa", o)),
          mixer(b).register_forward_hook(lambda mod, i, o: cap.__setitem__("mix", o))]
    if b.norm2 is not None:
        hs.append(b.norm2.register_forward_hook(lambda mod, i, o: cap.__setitem__("xn", o)))
    if b.mlp is not None:
        hs.append(b.mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("mlp", o)))
    if b.private_ple is not None:
        hs.append(b.ple_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("ple", o)))
    if b.priv_names:                                                   # özel nöron modül değil: yöntemi sar
        orig = b.private_out

        def wrapped(xn, rows=None, priv=None, name="private"):
            q = orig(xn, rows, priv, name)
            cap["priv"] = q if name == b.priv_names[0] else cap["priv"] + q
            return q
        b.private_out = wrapped
        hs.append(_Restore(b))
    return cap, hs


class _Restore:
    """Örnek üstüne yazılmış private_out'u kaldırır (kanca gibi .remove())."""
    def __init__(self, b):
        self.b = b

    def remove(self):
        self.b.__dict__.pop("private_out", None)


def replace(m, l, comp, fn, stash):
    """Katman l'de bileşen comp'un çıktısı fn(x) ile değişir (x = bileşen girdisi) -> kancalar."""
    b = m.blocks[l]
    hs = [b.norm1.register_forward_hook(lambda mod, i, o: stash.__setitem__("xa", o))]
    if b.norm2 is not None:
        hs.append(b.norm2.register_forward_hook(lambda mod, i, o: stash.__setitem__("xn", o)))
    if comp == "mix":
        hs.append(mixer(b).register_forward_hook(lambda mod, i, o: fn(stash["xa"]).to(o.dtype)))
    elif comp == "mlp":
        hs.append(b.mlp.register_forward_hook(lambda mod, i, o: fn(stash["xn"]).to(o.dtype)))
    elif comp == "ple":
        hs.append(b.ple_norm.register_forward_hook(lambda mod, i, o: fn(stash["xn"]).to(o.dtype)))
    else:
        def priv_out(xn, rows=None, priv=None, name="private"):
            return fn(xn).to(xn.dtype) if name == b.priv_names[0] else torch.zeros_like(xn)
        b.private_out = priv_out
        hs.append(_Restore(b))
    return hs


@torch.no_grad()
def window_loss(m, wins, qmin):
    """Konum ≥ qmin hedeflerinde ortalama kayıp (pencereler (n, N + 1))."""
    tot = 0.0
    for w in wins:
        tot += float(F.cross_entropy(m(w[None, :-1]).float()[0, qmin:], w[qmin + 1:]))
    return tot / len(wins)


@torch.no_grad()
def analyze(m, wins_tab, wins_fit, wins_ev, qmin, ridge=(1e-3, 1e-2, 1e-1, 1.0), shrink=4.0, log=print):
    """Katman katman: tablo, sabit filtre (normal denklemler, sırt cezası taraması), ölçüm, takas -> satırlar."""
    dev = next(m.parameters()).device
    V, d, L = m.cfg.vocab_size, m.cfg.d_model, len(m.blocks)
    nF = len(DELAYS) + len(BANDS)
    base = window_loss(m, wins_ev, qmin)
    rows = []
    for l in range(L):
        b = m.blocks[l]
        comps = components(b)
        cap, hs = capture(m, l)
        # 1 tablo: kelime başına toplam ve sayı
        S = {c: torch.zeros(V, d, device=dev) for c, _ in comps}
        C = torch.zeros(V, device=dev)
        for w in wins_tab:
            m.hidden(w[None, :-1])
            tok = w[qmin:-1]
            C.index_add_(0, tok, torch.ones_like(tok, dtype=C.dtype))
            for c, _ in comps:
                S[c].index_add_(0, tok, cap[c][0, qmin:].float())
        mu = {c: S[c].sum(0) / C.sum() for c, _ in comps}
        T = {c: mu[c] + (S[c] - C[:, None] * mu[c]) / (C[:, None] + shrink) for c, _ in comps}
        del S
        # 2 sabit filtre: girdi başına X'X, bileşen başına X'y (çıktı) ve X'r (tablodan kalan)
        keys = sorted({k for _, k in comps})
        XtX = {k: torch.zeros(nF * d, nF * d, device=dev) for k in keys}
        XtY = {(c, t): torch.zeros(nF * d, d, device=dev) for c, _ in comps for t in "yr"}
        for w in wins_fit:
            m.hidden(w[None, :-1])
            tok = w[qmin:-1]
            X = {k: features(cap[k].float())[0, qmin:].reshape(-1, nF * d) for k in keys}
            for k in keys:
                XtX[k] += X[k].T @ X[k]
            for c, k in comps:
                y = cap[c][0, qmin:].float()
                XtY[(c, "y")] += X[k].T @ y
                XtY[(c, "r")] += X[k].T @ (y - T[c][tok])
        W = {}
        for k in keys:
            A = XtX.pop(k)
            md = A.diagonal().mean()
            for r in ridge:
                A.diagonal().add_(r * md)
                for c, kk in comps:
                    if kk == k:
                        for t in "yr":
                            W[(c, t, r)] = torch.linalg.solve(A, XtY[(c, t)])
                A.diagonal().sub_(r * md)
        del XtY
        # 3 ölçüm
        R = len(ridge)
        sse = {c: {"tablo": 0.0, "sabit": torch.zeros(R), "t+s": torch.zeros(R), "sst": 0.0} for c, _ in comps}
        for w in wins_ev:
            m.hidden(w[None, :-1])
            tok = w[qmin:-1]
            X = {k: features(cap[k].float())[0, qmin:].reshape(-1, nF * d) for k in keys}
            for c, k in comps:
                y = cap[c][0, qmin:].float()
                t = T[c][tok]
                e = sse[c]
                e["tablo"] += float(((y - t) ** 2).sum())
                e["sst"] += float(((y - y.mean(0)) ** 2).sum())
                for i, r in enumerate(ridge):
                    e["sabit"][i] += float(((X[k] @ W[(c, "y", r)] - y) ** 2).sum())
                    e["t+s"][i] += float(((t + X[k] @ W[(c, "r", r)] - y) ** 2).sum())
        for h in hs:
            h.remove()
        # 4 takas
        for c, k in comps:
            e = sse[c]
            ry, rr = ridge[int(e["sabit"].argmin())], ridge[int(e["t+s"].argmin())]
            r2 = {v: 1 - float(e[v] if v == "tablo" else e[v].min()) / e["sst"] for v in ("tablo", "sabit", "t+s")}
            cur = {}

            def pred(x, v):
                if v == "ort":
                    return mu[c].expand(x.shape[0], x.shape[1], d)
                out = T[c][cur["tok"]][None] if v in ("tablo", "t+s") else 0
                if v in ("sabit", "t+s"):
                    Wv = W[(c, "y", ry)] if v == "sabit" else W[(c, "r", rr)]
                    out = out + features(x.float()).reshape(x.shape[0], x.shape[1], -1) @ Wv
                return out
            swap = {}
            for v in VARIANTS:
                stash = {}
                hk = replace(m, l, c, lambda x, v=v: pred(x, v), stash)
                tot = 0.0
                for w in wins_ev:
                    cur["tok"] = w[:-1]
                    tot += float(F.cross_entropy(m(w[None, :-1]).float()[0, qmin:], w[qmin + 1:]))
                for h in hk:
                    h.remove()
                swap[v] = tot / len(wins_ev) - base
            row = dict(layer=l, kind=b.kind + ("c" if b.attn is None else ""), comp=c, r2=r2, swap=swap)
            rows.append(row)
            log("%-6d %-3s %-6s | %6.3f %6.3f %6.3f | %6.3f | %+8.4f %+8.4f %+8.4f %+8.4f" % (
                l, row["kind"], NAMES[c], r2["tablo"], r2["sabit"], r2["t+s"], 1 - r2["t+s"],
                *(swap[v] for v in VARIANTS)))
        del T, W
    return base, rows


@torch.no_grad()
def truncate(m, wins_ev, qmin, specs, log=print):
    """Y pencereleri çıkarımda specs ile daraltılınca kayıp artışı (Y katmanlarına sırayla, döngüsel)."""
    ys = [b for b in m.blocks if b.kind == "Y" and b.attn is not None]
    orig = [b.attn.window for b in ys]
    base = window_loss(m, wins_ev, qmin)
    out = []
    for spec in specs:
        for i, b in enumerate(ys):
            b.attn.window = min(spec[i % len(spec)], orig[i])
        dl = window_loss(m, wins_ev, qmin) - base
        out.append((spec, dl))
        log("  Y %-10s (eğitimde %s)  Δ %+.4f" % ("/".join(map(str, spec)), "/".join(map(str, orig)), dl))
    for b, w in zip(ys, orig):
        b.attn.window = w
    return out


def main():
    from rgn.data.loader import valid_tokens
    from rgn.train.checkpoint import load_run
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--table", type=int, default=384, help="tablo kestirim penceresi")
    ap.add_argument("--fit", type=int, default=96, help="sabit filtre kestirim penceresi")
    ap.add_argument("--eval", type=int, default=16, help="ölçüm / takas penceresi")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--qmin", type=int, default=256)
    ap.add_argument("--shrink", type=float, default=4.0, help="tablo büzülmesi (kelime sayısına eklenen)")
    ap.add_argument("--truncate", nargs="*", default=[], help="Y pencere kırpma kalibrasyonu, ör. 32,64 16,16")
    ap.add_argument("--no_map", action="store_true", help="yalnız kırpma kalibrasyonu")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load_run(a.run, dev)
    m.float()
    N = a.seq
    n = a.table + a.fit + a.eval
    toks = valid_tokens(a.dataset, n * (N + 1))
    wins = torch.from_numpy(np.asarray(toks, dtype=np.int64)).view(-1, N + 1).to(dev)
    tab, fit, ev = wins[:a.table], wins[a.table:a.table + a.fit], wins[a.table + a.fit:]
    print("%s (adım %s) — tablo %d, sabit %d, ölçüm %d pencere × %d (konum ≥ %d)" % (
        a.run, step, a.table, a.fit, a.eval, N, a.qmin), flush=True)
    if not a.no_map:
        print("\n%-6s %-3s %-6s | %6s %6s %6s | %6s | %8s %8s %8s %8s" % (
            "katman", "tür", "bileşen", "tablo", "sabit", "t+s", "içerik", "ort", "tablo", "sabit", "t+s"))
        print("%-18s | %-20s | %-6s | %s" % ("", "R² (çıktının açıklanan payı)", "", "takas Δ kayıp (yeniden eğitimsiz)"))
        base, _ = analyze(m, tab, fit, ev, a.qmin, shrink=a.shrink, log=lambda s: print(s, flush=True))
        print("taban kayıp (konum ≥ %d): %.4f" % (a.qmin, base))
    if a.truncate:
        print("\nkalibrasyon: Y pencereleri çıkarımda daraltılınca (yeniden eğitimsiz)")
        truncate(m, ev, a.qmin, [tuple(int(v) for v in s.split(",")) for s in a.truncate])


if __name__ == "__main__":
    main()
