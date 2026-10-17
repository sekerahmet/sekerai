"""Dikkat ne yapıyor — beş soru, eğitilmiş modelde (koşu değil; varsayım üretir, karar koşuyla verilir).

  1 konum   : head'in deseni içerikten bağımsız mı?  Sorgunun uzaklık profili (bantlar: 0, 1, 2, 3, 4–7, … 512+) ile
              head'in ortalama profilinin örtüşmesi Σ min (1: hep aynı desen → sabit kaydırma / evrişimle değiştirilebilir;
              Raganato vd. 2020).  Ayrıca en büyük bant ve payı (ör. "1: 0,85" önceki token head'i).
  2 kelime  : desen sorgu kelimesinin kimliğinden ne kadar belli?  Kelime profili (aynı kelimenin sorgularının ortalaması)
              kestirim pencerelerinde, örtüşme ölçümü ayrı pencerelerde (aşırı uyum yok); kazanç = örtüşme(kelime
              profili) − örtüşme(head profili), yalnız kestirimde ≥ --min_count kez geçen kelimeler.
  3 head    : head ortalamayla değiştirilince (mean ablation; sıfırlama değil) kayıp artışı (Michel vd. 2019).
  4 taşınan : head çıktısının, aynı dikkat paylarıyla yalnız kelime kimliğinden (gömmeden) hesaplanan çıktıya benzerliği
              (kosinüs); referans: V'nin kendisinin gömme V'sine benzerliği (katman 0'da ikisi de 1).
  5 uzak    : genel katmanlar yerel pencereyle sınırlanınca token başına kayıp artışı; sık / diğer kelime ve konuma göre.

Geçerlilik (eğitilmiş modelde ölçüm): pay ≠ önem (Kobayashi vd. 2020) → 4. ölçü değer tarafını da katar; kapatma,
modelin birlikte uyum sağladığı parçayı söker (dağılım dışı) → sıfır yerine ortalama; tek tek kapatma yedekli head'leri
küçük gösterir; ölçülen şey bugünkü kullanım, yeniden eğitimde varılacak düzen değil.

    python scripts/attn_probe.py --run fw_s000_d1024L9_yari_epok
    python scripts/attn_probe.py --run ... --parts konum,kelime        # yalnız bazıları
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rgn.data.loader import valid_tokens  # noqa: E402
from rgn.data.registry import EOS_ID  # noqa: E402
from rgn.paths import ROOT  # noqa: E402
from rgn.train.checkpoint import load_run  # noqa: E402

EDGES = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]                 # uzaklık bantları: 0, 1, 2, 3, 4–7, … 512+
BAND_NAMES = ["0", "1", "2", "3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512+"]


def _ov(a, b):
    return torch.minimum(a, b).sum(-1)


@torch.no_grad()
def pattern_pass(m, wins, qmin, dev, carry):
    """Pencere başına katman katman: uzaklık profilleri (Q, L, H, NB), sorgu token'ları (Q,), taşınan kosinüsleri."""
    N = wins.shape[1]
    dist = (torch.arange(N, device=dev)[:, None] - torch.arange(N, device=dev)[None, :]).clamp_min(0)
    bins = torch.bucketize(dist, torch.tensor(EDGES, device=dev), right=True)
    qsel = torch.arange(N, device=dev) >= qmin
    L, H, NB = len(m.blocks), m.blocks[0].attn.H, len(EDGES) + 1
    unif = {}                                      # referans: izinli anahtarlara düzgün dağılım (Y: pencere, G: causal)
    for b in m.blocks:
        W = b.attn.window
        if W not in unif:
            ok = (dist <= (W - 1 if W else N)) & (torch.arange(N, device=dev)[:, None] >= torch.arange(N, device=dev))
            u = ok.float() / ok.float().sum(-1, keepdim=True)
            unif[W] = torch.zeros(int(qsel.sum()), NB, device=dev).scatter_add_(1, bins[qsel], u[qsel])
    prof, qtok = [], []
    cos_o = torch.zeros(L, H, device=dev)
    cos_v = torch.zeros(L, device=dev)
    for w in wins:
        t = w.to(dev)
        x = m.embed(t[None])
        e = x
        rope = m.rope(N)
        rows = m.priv_map[t[None]] if m.priv_map is not None else None
        ple = m.ple_context(x)
        hb = torch.zeros(int(qsel.sum()), L, H, NB, device=dev)
        for l, b in enumerate(m.blocks):
            xa = b.norm1(x)
            p = b.attn.weights(xa, rope)[0]                                      # (H, N, N)
            pq = p[:, qsel]
            hb[:, l] = torch.zeros(H, pq.shape[1], NB, device=dev).scatter_add_(
                2, bins[qsel].unsqueeze(0).expand(H, -1, -1), pq).transpose(0, 1)
            if carry:
                g = b.attn.H // b.attn.Hkv
                v = b.attn.project(xa, rope)[2][0].repeat_interleave(g, 0)      # (H, N, dh)
                vt = b.attn.project(b.norm1(e), rope)[2][0].repeat_interleave(g, 0)
                cos_o[l] += F.cosine_similarity(pq @ v, pq @ vt, dim=-1).mean(-1)
                cos_v[l] += F.cosine_similarity(v, vt, dim=-1).mean()
            x = b(x, rope, None, rows, None, None, None if ple is None else ple[:, :, l])
        prof.append(hb)
        qtok.append(t[qsel])
    n = len(wins)
    ref = torch.stack([unif[b.attn.window] for b in m.blocks], 1).repeat(n, 1, 1)      # (Q, L, NB), pencere başına aynı
    return torch.cat(prof), torch.cat(qtok), cos_o / n, cos_v / n, ref


@torch.no_grad()
def token_losses(m, X):
    logits = m(X[:, :-1]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), X[:, 1:].reshape(-1), reduction="none").view(
        X.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="fineweb_s000")
    ap.add_argument("--counts", default="assets/fw_s000_vocab_used.json")
    ap.add_argument("--parts", default="konum,kelime,head,tasinan,uzak")
    ap.add_argument("--windows", type=int, default=16, help="desen ölçüleri (yarısı kestirim, yarısı ölçüm)")
    ap.add_argument("--abl_windows", type=int, default=4, help="kapatma ölçüleri (3, 5)")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--qmin", type=int, default=256)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--top", type=int, default=30, help="5'te 'sık kelime' sayılan en sık token sayısı")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    torch.set_grad_enabled(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, step = load_run(a.run, dev)
    m.float()
    N, W, L = a.seq, m.cfg.local_window, len(m.blocks)
    kinds = [b.kind for b in m.blocks]
    toks = valid_tokens(a.dataset, (a.windows + a.abl_windows) * (N + 1))
    allw = torch.from_numpy(np.asarray(toks, dtype=np.int64)).view(-1, N + 1)
    wins, abl = allw[:a.windows, :N], allw[a.windows:]
    print("%s (adım %s) — desen: %d pencere × %d (sorgu ≥ %d); kapatma: %d pencere" % (
        a.run, step, a.windows, N, a.qmin, a.abl_windows), flush=True)

    if parts & {"konum", "kelime", "tasinan"}:
        half = a.windows // 2
        pe, te, _, _, _ = pattern_pass(m, wins[:half], a.qmin, dev, False)                  # kestirim
        pm, tm, cos_o, cos_v, ref = pattern_pass(m, wins[half:], a.qmin, dev, "tasinan" in parts)  # ölçüm
        gprof = pe.mean(0)                                                                 # (L, H, NB)
        if "konum" in parts:
            ovg = _ov(pm, gprof[None]).mean(0)                                             # (L, H)
            ovu = _ov(pm, ref[:, :, None]).mean(0)                                         # düzgün dağılım referansı
            fix = ovg - ovu                                                                # +: sabit desen, 0: yayvan
            print("\n1. konum: sorgu profilinin head profiline örtüşmesi; 'üstü' = örtüşme − düzgün dağılımla örtüşme "
                  "(büyük: desen sabit ve keskin, ~0: yayvan); en sabit head ve en büyük bandı")
            for l in range(L):
                h = int(fix[l].argmax())
                kb = int(gprof[l, h].argmax())
                print("  %d %s  örtüşme ort %.2f, düzgün %.2f, üstü ort %+.2f  en çok %+.2f (head %2d: '%s' bandı "
                      "%.2f)  üstü ≥ 0,3 olan head: %d / %d" % (
                          l, kinds[l], ovg[l].mean(), ovu[l].mean(), fix[l].mean(), fix[l, h], h, BAND_NAMES[kb],
                          gprof[l, h, kb], int((fix[l] >= 0.3).sum()), fix.shape[1]))
        if "kelime" in parts:
            u, inv = torch.unique(te, return_inverse=True)
            cnt = torch.bincount(inv, minlength=len(u)).float()
            sums = torch.zeros(len(u), *pe.shape[1:], device=dev).index_add_(0, inv, pe)
            ok = cnt >= a.min_count
            pos = torch.searchsorted(u, tm).clamp_max(len(u) - 1)
            have = (u[pos] == tm) & ok[pos]
            tprof = sums[pos[have]] / cnt[pos[have]][:, None, None, None]
            gain = (_ov(pm[have], tprof) - _ov(pm[have], gprof[None])).mean(0)            # (L, H)
            print("\n2. kelime: kelime profilinin head profiline göre örtüşme kazancı (+: desen kelimeye bağlı); %d "
                  "ölçüm sorgusu, %d kelime (≥ %d kez)" % (int(have.sum()), int(ok.sum()), a.min_count))
            for l in range(L):
                h = int(gain[l].argmax())
                print("  %d %s  ort %+.3f  en çok %+.3f (head %2d)" % (l, kinds[l], gain[l].mean(), gain[l, h], h))
        if "tasinan" in parts:
            print("\n4. taşınan: V (bağlamlı) ↔ V (yalnız gömmeden) kosinüsü — 1: katman kelime kimliğini taşıyor, düşük: "
                  "bağlamdan üretilmiş bilgi; head çıktısı aynı karşılaştırmanın dikkatle ağırlıklısı (katman 0'da ikisi 1)")
            for l in range(L):
                print("  %d %s  V %.2f   çıktı ort %.2f (head en az %.2f / en çok %.2f)" % (
                    l, kinds[l], cos_v[l], cos_o[l].mean(), cos_o[l].min(), cos_o[l].max()))

    if parts & {"head", "uzak"}:
        X = abl.to(dev)
        base = token_losses(m, X)
        print("\nkapatma pencereleri taban kaybı %.4f" % base.mean(), flush=True)
        if "head" in parts:
            print("\n3. head: ortalamayla değiştirince kayıp artışı (×1000 nat); katman satırı, head sütunu")
            for l, b in enumerate(m.blocks):
                orig, row = b.attn._core, []
                for h in range(b.attn.H):
                    def patched(q, k, v, mask=None, _h=h, _o=orig):
                        out = _o(q, k, v, mask).clone()
                        out[:, _h] = out[:, _h].mean(dim=(0, 1))
                        return out
                    b.attn._core = patched
                    row.append(1000 * (token_losses(m, X).mean() - base.mean()).item())
                del b.attn._core
                print("  %d %s  %s   toplam %+.0f" % (l, kinds[l], " ".join("%+5.0f" % v for v in row), sum(row)),
                      flush=True)
        if "uzak" in parts:
            gl = [b for b in m.blocks if b.kind == "G"]
            for b in gl:
                b.attn.window = W
            d = token_losses(m, X) - base
            for b in gl:
                b.attn.window = 0
            c = json.loads((ROOT / a.counts).read_text(encoding="utf-8"))
            order = sorted(zip(c.get("counts") or [1] * len(c["tokens"]), c["tokens"]), key=lambda ct: -ct[0])
            freq = torch.tensor(sorted({t for _, t in order[:a.top]} | {EOS_ID}), device=dev)
            tgt = X[:, 1:]
            posn = torch.arange(d.shape[1], device=dev)[None].expand_as(d)
            far = posn >= W                                                     # yalnız uzak bağlamı olan konumlar
            isf = torch.isin(tgt, freq)
            print("\n5. uzak: genel katmanlar %d pencereyle sınırlanınca hedef token başına kayıp artışı (konum ≥ %d)" % (
                W, W))
            for name, sel in (("hepsi", far), ("sık kelime (hedef)", far & isf), ("diğer kelime", far & ~isf)):
                dd = d[sel]
                print("  %-20s n %6d  ort %+.4f  Δ>0,1: %%%4.1f  Δ>1: %%%4.1f" % (
                    name, dd.numel(), dd.mean(), 100 * (dd > 0.1).float().mean(), 100 * (dd > 1).float().mean()))
            for lo, hi in ((W, 256), (256, 512), (512, 1 << 30)):
                dd = d[(posn >= lo) & (posn < hi)]
                if dd.numel():
                    print("  konum %4d–%-5s       ort %+.4f  Δ>0,1: %%%4.1f" % (
                        lo, hi - 1 if hi < 1 << 30 else "", dd.mean(), 100 * (dd > 0.1).float().mean()))


if __name__ == "__main__":
    main()
