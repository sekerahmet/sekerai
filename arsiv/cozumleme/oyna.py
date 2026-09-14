# -*- coding: utf-8 -*-
"""
OYNA — egitilmis modelle etkilesim.

DIKKAT: bu bir sohbet modeli DEGIL. Sozlugunde kelime yok; sadece varlik ve
iliski kimlikleri var. Sorabilecegin sey sudur:

    1-hop   e42 'nin r3 'u ne?            -> f(e42, r3)
    2-hop   e42 'nin r3 'unun r7 'si ne?  -> f(f(e42,r3), r7)

Ve her soruda sunu gorursun:
    - modelin ilk 5 tahmini ve olasiliklari
    - dogru cevap, ve bu zincir egitimde GORULDU mu
    - KOPRU (ara varlik) ne, model onu iceride biliyor mu
    - bellekteki hangi slotlar yandi (kol B/C icin)

Kullanim (Colab):
    %run oyna.py --out /content/out2 --arm C
sonra:
    >>> q(42, 3)            # 1-hop
    >>> q(42, 3, 7)         # 2-hop
    >>> rastgele()          # tutulmus bir zincirden rastgele soru
    >>> kiyasla(42, 3, 7)   # ayni soruyu B ve C kollarina birden sor
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F


def yukle(out_dir, arm, seed=0):
    ck = os.path.join(out_dir, f"model_{arm}_s{seed}.pt")
    if not os.path.exists(ck):
        raise SystemExit(f"bulunamadi: {ck}\n(SAVE_CKPT=1 ile egitilmis olmali)")
    d = torch.load(ck, map_location="cpu", weights_only=False)

    # sifirdan.py'yi ckpt'teki CFG ile ayni sekilde kur
    import importlib.util
    here = os.path.dirname(os.path.abspath(globals().get("__file__", "oyna.py")))
    for k, v in d["cfg"].items():
        os.environ[k] = str(v)
    os.environ["PRESET"] = "smoke"          # CFG zaten ortamdan ezilecek
    spec = importlib.util.spec_from_file_location("sf", os.path.join(here, "sifirdan.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["sf"] = m
    spec.loader.exec_module(m)

    assert m.VOCAB == d["vocab"], f"sozluk uyusmuyor: {m.VOCAB} vs {d['vocab']}"
    net = m.Net(arm, m.CFG)
    net.load_state_dict(d["state"]); net.eval()
    z = np.load(os.path.join(out_dir, "veri.npz"))
    return m, net, z


class Oyun:
    def __init__(self, out_dir, arm, seed=0):
        self.m, self.net, self.z = yukle(out_dir, arm, seed)
        self.arm = arm
        self.facts = self.z["facts"]
        self.comp = {(int(a), int(b), int(c)) for a, b, c, _, _ in self.z["comp"]}
        self.N, self.R = self.facts.shape
        print(f"kol {arm} yuklendi | {self.N} varlik | {self.R} iliski | "
              f"{self.m.nparam(self.net)/1e6:.2f}M parametre")
        print(f"tutulmus (egitimde gorulmemis) zincir sayisi: {len(self.comp)}")

    # ---- dahili
    def _ileri(self, x, poz):
        with torch.no_grad():
            lg, w = self.net(torch.tensor([x]), want_w=True)
        p = lg[0, poz].softmax(-1)
        ent_p = p[self.m.ENT_OFF:]
        top = torch.topk(ent_p, 5)
        return top.indices.numpy(), top.values.numpy(), w

    def _slotlar(self, w, poz, k=5):
        if w is None:
            return None
        ww = w[0, poz].numpy()
        idx = np.argsort(-ww)[:k]
        return [(int(i), float(ww[i])) for i in idx], float(
            -(ww * np.log(ww + 1e-12)).sum())

    # ---- kullanici arayuzu
    def q(self, e, r1, r2=None):
        m = self.m
        if not (0 <= e < self.N): raise ValueError(f"varlik 0..{self.N-1}")
        for r in (r1, r2):
            if r is not None and not (0 <= r < self.R):
                raise ValueError(f"iliski 0..{self.R-1}")

        if r2 is None:
            x = [m.Q1, m.ENT_OFF + e, m.REL_OFF + r1, m.QM, 0, 0, 0, 0]
            poz, gold, kopru = 3, int(self.facts[e, r1]), None
            basl = f"e{e} 'nin r{r1} 'u ne?"
            gorulmus = "evet (1-hop egitimde vardi)"
        else:
            x = [m.Q2, m.ENT_OFF + e, m.REL_OFF + r1, m.REL_OFF + r2, m.QM, 0, 0, 0]
            poz = 4
            kopru = int(self.facts[e, r1]); gold = int(self.facts[kopru, r2])
            basl = f"e{e} 'nin r{r1} 'unun r{r2} 'si ne?"
            gorulmus = ("HAYIR — bu zincir egitimde hic gecmedi"
                        if (e, r1, r2) in self.comp else "evet (egitimde vardi)")

        ids, pr, w = self._ileri(x, poz)
        print(f"\n  {basl}")
        print(f"  dogru cevap: e{gold}   [{gorulmus}]")
        if kopru is not None:
            print(f"  KOPRU      : e{kopru}   (soruda GECMIYOR, modelin kendisi bulmali)")
        print("  modelin tahminleri:")
        for i, (a, p) in enumerate(zip(ids, pr)):
            im = "  <-- DOGRU" if int(a) == gold else (
                 "  <-- kopru!" if kopru is not None and int(a) == kopru else "")
            print(f"    {i+1}. e{int(a):<5d} %{100*p:5.1f}{im}")
        if kopru is not None:
            kisayol = int(self.facts[e, r2])
            if int(ids[0]) == kisayol and kisayol != gold:
                print(f"    (!) 1. tahmin f(e{e},r{r2})=e{kisayol} — r{r1}'i atlamis, KISAYOL")
        s = self._slotlar(w, poz)
        if s:
            sl, H = s
            print(f"  bellek: entropi {H:.2f} (maks {np.log(self.m.CFG['M']):.2f}) | "
                  f"en cok yanan slotlar: " + ", ".join(f"#{i}({100*v:.1f}%)" for i, v in sl))
        return int(ids[0]) == gold

    def rastgele(self, n=1, sadece_tutulmus=True):
        hav = sorted(self.comp) if sadece_tutulmus else None
        r = np.random.RandomState()
        ok = 0
        for _ in range(n):
            if hav:
                e, r1, r2 = hav[r.randint(len(hav))]
            else:
                e, r1, r2 = r.randint(self.N), r.randint(self.R), r.randint(self.R)
            ok += bool(self.q(e, r1, r2))
        if n > 1:
            print(f"\n  {n} soruda {ok} dogru (%{100*ok/n:.0f})")


def kiyasla(out_dir, e, r1, r2, seed=0, arms=("A", "B", "C")):
    """Ayni soruyu birden cok kola sor."""
    for a in arms:
        try:
            g = Oyun(out_dir, a, seed)
        except SystemExit as ex:
            print(f"kol {a}: {ex}"); continue
        g.q(e, r1, r2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out2")
    ap.add_argument("--arm", default="C")
    ap.add_argument("--seed", type=int, default=0)
    a, _ = ap.parse_known_args()
    oyun = Oyun(a.out, a.arm, a.seed)
    q = oyun.q
    rastgele = oyun.rastgele
    print("\nhazir. dene:  q(42, 3)   q(42, 3, 7)   rastgele(5)")
