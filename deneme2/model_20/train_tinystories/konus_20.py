"""Modelle KONUS (model_20, TinyStories) -- bir hikaye baslangici yaz, model devamini yazsin.

Agirlik Colab'da uretilip Drive'a yaziliyor; burasi yalniz OKUYOR.  Model CPU'da calisir.  Model <eos>'tan baslar ve
kendi <eos>'unu yazinca durur.

  Once upon a time           yaz, devamini gorursun
  <bos satir>                sinavin 12 isteminden RASTGELE biri
  serbest                    istemsiz, bos sayfadan bir hikaye
  n=200                      en fazla kac token (varsayilan 200; istem + n <= 512)
  s=0.8                      sicaklik.  0 = hep en olasi token (sinavdaki acgozlu uretim)
  p=0.9                      top-p: en olasi token'lardan toplami 0,9 olan kumeden sec (varsayilan 0,9; 1 = kapali;
                             yalniz s > 0'da isler -- kapaliyken uzun kuyruktan nadir kelimeler gelir)
  r=1.3                      tekrar cezasi: son 20 token'da gecenlerin puani 1,3'e bolunur (1 = kapali)
  yasak                      <bilinmeyen>/<dolgu> uretimi kapat (varsayilan) / ac
  model                      kosulari listele (en yeni once), hangisi yuklu
  model <ad>                 baska bir kosu yukle (ad ya da basindan bir parca)
  ema                        ayni kosunun ortalama agirliklari (model_weight_ema.pt) / son agirliklar
  ?                          yardim
  q                          cik
"""
import json
import os
import random
import sys
import textwrap
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
# Agir moduller ilk kullanimda: pencere hemen acilir (import torch birkac saniye)
torch = np = DT = ET = BlockModel = AttentionCache = None

RUN_ROOTS = [r"G:\Drive'ım\model_20", r"G:\Drivem\model_20"]
TS_DIRS = [r"G:\Drive'ım\tinystories\onbellek", r"G:\Drivem\tinystories\onbellek"]


def _heavy():
    global torch, np, DT, ET, BlockModel, AttentionCache
    if DT is None:
        import numpy as _np
        import torch as _torch
        import data_tinystories as _dt
        import exam_tinystories as _et
        from model_20 import AttentionCache as _attention_cache
        from model_20 import BlockModel as _block_model
        torch, np, DT, ET, BlockModel, AttentionCache = _torch, _np, _dt, _et, _block_model, _attention_cache


def _first_existing(candidates):
    return next((d for d in candidates if os.path.isdir(d)), None)


def runs():
    """TinyStories kosulari (model.pt olan), en son degisen once."""
    root = _first_existing(RUN_ROOTS)
    if not root:
        return []
    d = [os.path.join(root, x) for x in os.listdir(root) if x.startswith("tinystories") and "_eski_" not in x]
    d = [x for x in d if os.path.exists(os.path.join(x, "model.pt"))]
    return sorted(d, key=lambda x: os.path.getmtime(os.path.join(x, "model.pt")), reverse=True)


def find_run(pattern=None):
    all_runs = runs()
    if not all_runs:
        print("KOSU YOK.  Aranan: %s\\tinystories*\\model.pt" % " ya da ".join(RUN_ROOTS))
        return None
    if not pattern:
        return all_runs[0]
    hit = [r for r in all_runs if os.path.basename(r) == pattern] or \
          [r for r in all_runs if os.path.basename(r).startswith(pattern)]
    if not hit:
        print("'%s' ile eslesen kosu yok.  'model' yazip listeye bak." % pattern)
        return None
    return hit[0]


def load_vocab():
    _heavy()
    d = _first_existing(TS_DIRS)
    return [str(a) for a in np.load(os.path.join(d, "sozluk_%s.npy" % DT.TAG), allow_pickle=True)]


def load_model(run_dir, n_vocab, averaged=False):
    """config.json'dan modeli kurar, model.pt (ya da model_weight_ema.pt) yukler."""
    _heavy()
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    kw = dict(cfg["model_kw"])
    kw.setdefault("layers", 1)                              # 28 Eylul oncesi kosular: tek Block
    assert not cfg.get("copy_path"), "kopya yolu (Oneri A) 28 Eylul'de kaldirildi"
    m = BlockModel(n_vocab, seed=0, stream_norm=cfg.get("stream_norm", True),
                   layer_norm=cfg.get("layer_norm", False), rope=cfg.get("rope", True), shared=cfg["setting"] == "shared",
                   normalized_update=cfg.get("normalized_update", False), sphere_weights=cfg.get("sphere_weights", False),
                   canon=cfg.get("canon", False), **kw)
    name = "model_weight_ema.pt" if averaged else "model.pt"
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        print("   %s yok (%s)" % (name, os.path.basename(run_dir)))
        return None, cfg
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval(), cfg


def summary(run_dir, cfg, averaged):
    kw = cfg["model_kw"]
    print("=" * 74)
    print("kosu    %s%s" % (os.path.basename(run_dir), "   (ORTALAMA agirliklar)" if averaged else ""))
    print("model   D %s  katman %s  tur %s  FactUnits %s   adim %s (%.2f epok)   takvim %s" % (
        kw.get("d"), kw.get("layers", 1), kw.get("turns"), kw.get("units"), "{:,}".format(cfg["steps"]),
        cfg.get("epochs", 0), cfg.get("schedule", "?")))
    f = os.path.join(run_dir, "final.json")
    if os.path.exists(f):
        v = json.load(open(f, encoding="utf-8"))["valid"]
        print("olcum   valid ppl %.2f  acc %.4f" % (v["ppl"], v["accuracy"]))
    print("=" * 74)


def generate(model, ids, n, vocab, temp=0.0, top_p=1.0, penalty=1.0, banned=()):
    """Tek istem, token token.  Tek head'de onbellekli (istem bir kez, sonra yalniz yeni konum; istem + n <= 512),
    cok head'de her token'da tam yeniden hesap.  <eos>'ta durur."""
    eos = vocab.index(DT.EOS_TOKEN)
    ban = [vocab.index(t) for t in banned]
    out = []
    x = list(ids)
    cached = getattr(model, "heads", 1) == 1 and len(x) + n <= 512
    caches = [AttentionCache(torch.tensor([len(x)]), len(x) + n) for _ in range(model.turns)] if cached else None
    with torch.no_grad():
        for i in range(n):
            if cached:                                      # ilk adim istemin tamami, sonra yalniz son token
                logits = model.logits(torch.tensor([x if i == 0 else x[-1:]]), caches)[0, -1].float()
            else:
                logits = model.logits(torch.tensor([x[-512:]]))[0, -1].float()
            if ban:
                logits[ban] = -float("inf")
            if penalty != 1.0:                              # son 20 token: pozitif puan boluner, negatif carpilir
                recent = torch.tensor(sorted(set(x[-20:])))
                logits[recent] = torch.where(logits[recent] > 0, logits[recent] / penalty, logits[recent] * penalty)
            if temp <= 0:
                t = int(logits.argmax())
            else:
                p = torch.softmax(logits / temp, -1)
                if top_p < 1.0:
                    ps, order = p.sort(descending=True)
                    keep = ps.cumsum(0) - ps < top_p
                    p = torch.zeros_like(p).scatter(0, order[keep], ps[keep])
                t = int(torch.multinomial(p / p.sum(), 1))
            if t == eos:
                break
            out.append(t)
            x.append(t)
    return out


def print_wrapped(title, text, g=78):
    print(title)
    for para in text.split("\n"):
        for s in textwrap.wrap(para, g) or [""]:
            print("   " + s)


def main():
    print(__doc__.split("\n\n")[-1].rstrip())
    print("\nmodel arka planda yukleniyor -- istemini simdiden yazabilirsin\n")
    state = dict(averaged=False)

    def _load():
        try:
            state["run"] = find_run(sys.argv[1] if len(sys.argv) > 1 else None)
            if state["run"]:
                state["vocab"] = load_vocab()
                state["model"], state["cfg"] = load_model(state["run"], len(state["vocab"]))
        except BaseException as e:
            state["error"] = e

    loader = threading.Thread(target=_load, daemon=True)
    loader.start()
    ready = False
    # top-p 0,9: s = 0,8'de kapaliyken kelime salatasi ("a green toot"), aciksa tutarli (kullanici denemesi, 28 Eylul)
    n, temp, top_p, penalty, banned = 200, 0.0, 0.9, 1.0, True
    while True:
        try:
            g = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if g in ("q", "quit", "cik"):
            break
        if g == "?":
            print(__doc__.split("\n\n")[-1].rstrip())
            continue
        if g.startswith("n="):
            n = max(1, int(g[2:]))
            print("   en fazla %d token" % n)
            continue
        if g.startswith("s="):
            temp = max(0.0, float(g[2:]))
            print("   sicaklik %.2f%s" % (temp, "  (acgozlu, sinavdaki gibi)" if not temp else ""))
            continue
        if g.startswith("p="):
            top_p = min(1.0, max(0.01, float(g[2:])))
            print("   top-p %.2f%s" % (top_p, "  (kapali)" if top_p == 1 else ""))
            continue
        if g.startswith("r="):
            penalty = max(1.0, float(g[2:]))
            print("   tekrar cezasi %.2f%s" % (penalty, "  (kapali)" if penalty == 1 else ""))
            continue
        if g == "yasak":
            banned = not banned
            print("   <bilinmeyen>/<dolgu> %s" % ("URETILMEZ" if banned else "uretilebilir"))
            continue
        if not ready:
            if loader.is_alive():
                print("   model yukleniyor, bekle...")
                loader.join()
            if "error" in state or not state.get("run") or state.get("model") is None:
                raise SystemExit(state.get("error") or 1)
            summary(state["run"], state["cfg"], state["averaged"])
            ready = True
        if g == "model":
            for r in runs()[:15]:
                print("   %s%s" % (os.path.basename(r), "   <- yuklu" if r == state["run"] else ""))
            continue
        if g.startswith("model ") or g == "ema":
            run = state["run"] if g == "ema" else find_run(g[6:].strip())
            averaged = (not state["averaged"]) if g == "ema" else False
            if run:
                m, cfg = load_model(run, len(state["vocab"]), averaged)
                if m is not None:
                    state.update(run=run, model=m, cfg=cfg, averaged=averaged)
                    summary(run, cfg, averaged)
            continue
        vocab = state["vocab"]
        eos = vocab.index(DT.EOS_TOKEN)
        if g == "serbest":
            g = ""
        elif not g:
            g = random.choice(ET.PROMPTS)
        ids = [eos] + DT.encode(g, vocab)
        n_unknown = sum(1 for t in ids if vocab[t] == DT.UNK_TOKEN)
        steps = min(n, 512 - len(ids))
        out = generate(state["model"], ids, steps, vocab, temp, top_p, penalty,
                       (DT.PAD_TOKEN, DT.UNK_TOKEN) if banned else ())
        in_quote = sum(vocab[t] == '"' for t in ids[1:]) % 2 == 1
        print()
        print_wrapped("ISTEM%s" % ("   (%d kelime sozlukte YOK)" % n_unknown if n_unknown else ""),
                      DT.decode(ids[1:], vocab) or "(bos)")
        print_wrapped("MODEL  (sicaklik %.2f, top-p %.2f, tekrar cezasi %.2f)%s" % (
            temp, top_p, penalty, "" if len(out) < steps else "  -- sinira geldi, kesildi"),
            DT.decode(out, vocab, in_quote=in_quote))
        print()


if __name__ == "__main__":
    main()
