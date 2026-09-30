"""Modelle KONUS (model_y, FineWeb-Edu kosulari) -- bir metin baslangici yaz, model devamini yazsin.
Kullanici, 30 Eylul: "Sen bir konuş py yazar mısın"; ad "konus_fineweb.py (Önerilen)".

Agirlik Colab'da uretilip Drive'a yaziliyor; burasi yalniz OKUYOR ve yerel onbellege alir (CACHE_DIR): model.pt / EMA
bir kez kopyalanir, checkpoint'in yalniz model agirligi (~1 GB, optimizer'siz) yazilir -- ayni dosya bir daha inmez.
Model CPU'da calisir, gpt2 tokenizer Drive'dan (fineweb/gpt2/tokenizer.json).  Model <|endoftext|>'ten (belge basi) baslar ve kendi <|endoftext|>'ini yazinca durur.
Kosu surerken model.pt yoksa son checkpoint yuklenir: egitilirken konusulur.

  The water cycle is         yaz, devamini gorursun
  <bos satir>                sinavin sabit istemlerinden RASTGELE biri
  serbest                    istemsiz, bos sayfadan bir belge
  n=200                      en fazla kac token (varsayilan 200)
  s=0.8                      sicaklik.  0 = hep en olasi token (sinavdaki acgozlu uretim)
  p=0.9                      top-p: en olasi token'lardan toplami 0,9 olan kumeden sec (1 = kapali; yalniz s > 0'da)
  r=1.2                      tekrar cezasi: son 64 token'da gecenlerin puani 1,2'ye bolunur (1 = kapali); yalniz uretim
  model                      kosulari listele (en yeni once), hangisi yuklu
  model <ad>                 baska bir kosu yukle (ad ya da basindan bir parca)
  ema                        ayni kosunun ortalama agirliklari (model_weight_ema.pt) / son agirliklar
  ck                         kosunun en son checkpoint'ini yukle (egitim surerken; adimi yazar)
  ?                          yardim
  q                          cik
"""
import json
import os
import random
import shutil
import sys
import textwrap
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "train_simplestories")]
# Agir moduller ilk kullanimda: pencere hemen acilir (import torch birkac saniye)
torch = DS = EF = I = AttentionCache = None

RUN_ROOTS = [r"G:\Drive'ım\model_y", r"G:\Drivem\model_y"]
FW_ROOTS = [r"G:\Drive'ım\fineweb", r"G:\Drivem\fineweb"]
EOT = "<|endoftext|>"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "model_y_konus")   # kosu basina alt klasor


def _heavy():
    global torch, DS, EF, I, AttentionCache
    if DS is None:
        import torch as _torch                          # torch pyarrow'dan once (Windows c10.dll)
        import data_simplestories as _ds
        import exam_fineweb as _ef
        import internals_y as _i
        from model_y import AttentionCache as _attention_cache
        torch, DS, EF, I, AttentionCache = _torch, _ds, _ef, _i, _attention_cache


def _first_existing(candidates):
    return next((d for d in candidates if os.path.isdir(d)), None)


def _checkpoints(run_dir):
    return sorted(f for f in os.listdir(run_dir) if f.startswith("checkpoint_t") and f.endswith(".pt"))


def runs():
    """FineWeb kosulari (model.pt ya da checkpoint olan), en son degisen once."""
    d = [os.path.join(root, x) for root in RUN_ROOTS if os.path.isdir(root) for x in os.listdir(root)
         if x.startswith("fineweb") and "_eski_" not in x]
    d = [x for x in d if os.path.exists(os.path.join(x, "config.json"))
         and (os.path.exists(os.path.join(x, "model.pt")) or _checkpoints(x))]
    return sorted(d, key=lambda x: os.path.getmtime(os.path.join(x, "config.json")), reverse=True)


def find_run(pattern=None):
    all_runs = runs()
    if not all_runs:
        print("KOSU YOK.  Aranan: %s\\fineweb*\\{model.pt, checkpoint_t*.pt}" % " ya da ".join(RUN_ROOTS))
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
    """gpt2 tokenizer (Drive'daki kopya) -> sozluk; DS.encode / decode bu sozlukle calisir."""
    _heavy()
    return DS._load(os.path.join(_first_existing(FW_ROOTS), "gpt2", "tokenizer.json"), EOT)[1]


def load_model(run_dir, weights="auto"):
    """config.json'dan modeli kurar (internals_y._build: eski config'lerin varsayilanlari dahil) ve agirligi yukler.
    weights: "last" model.pt, "ema" model_weight_ema.pt, "ck" son checkpoint, "auto" model.pt yoksa son checkpoint.
    -> (model ya da None, config, etiket)."""
    _heavy()
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if weights == "auto":
        weights = "last" if os.path.exists(os.path.join(run_dir, "model.pt")) else "ck"
    cache = os.path.join(CACHE_DIR, os.path.basename(run_dir))
    os.makedirs(cache, exist_ok=True)
    if weights == "ck":
        cks = _checkpoints(run_dir)
        if not cks:
            print("   checkpoint yok (%s)" % os.path.basename(run_dir))
            return None, cfg, None
        local = os.path.join(cache, cks[-1][:-3] + ".model.pt")
        if not os.path.exists(local):                   # paketten yalniz model agirligi, yerele bir kez
            print("   %s Drive'dan okunuyor (optimizer dahil ~3 GB), yalniz model yerele yaziliyor..." % cks[-1])
            pack = torch.load(os.path.join(run_dir, cks[-1]), map_location="cpu", weights_only=True)
            torch.save(pack["model"], local + ".part")
            os.replace(local + ".part", local)
        state, label = torch.load(local, map_location="cpu", weights_only=True), "checkpoint adim %d" % int(cks[-1][12:18])
    else:
        name = "model_weight_ema.pt" if weights == "ema" else "model.pt"
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            print("   %s yok (%s)" % (name, os.path.basename(run_dir)))
            return None, cfg, None
        local = os.path.join(cache, name)
        if not os.path.exists(local) or os.path.getmtime(local) < os.path.getmtime(path):   # Drive'daki daha yeni
            print("   %s yerel onbellege kopyalaniyor (%.1f GB)..." % (name, os.path.getsize(path) / 1e9))
            shutil.copyfile(path, local + ".part")
            os.replace(local + ".part", local)
        state, label = torch.load(local, map_location="cpu", weights_only=True), (
            "ORTALAMA agirliklar" if weights == "ema" else "son agirliklar")
    m = I._build(cfg)
    m.load_state_dict(state)
    return m.eval(), cfg, label


def summary(run_dir, cfg, label):
    kw = cfg["model_kw"]
    print("=" * 74)
    print("kosu    %s   (%s)" % (os.path.basename(run_dir), label))
    print("model   d %s  blok %s  tur %s  head %s  FactUnits %s  RoPE %s   adim %s (%.2f epok)   baglam %s" % (
        kw.get("d"), kw.get("layers"), kw.get("turns"), kw.get("heads"), kw.get("units"), kw.get("rope_base", 10000.0),
        "{:,}".format(cfg["steps"]), cfg.get("epochs", 0), cfg.get("seq_len")))
    f = os.path.join(run_dir, "final.json")
    if os.path.exists(f):
        v = json.load(open(f, encoding="utf-8")).get("valid") or {}
        if v:
            print("olcum   valid ppl %.2f  acc %.4f  bpb %.4f" % (v["ppl"], v["accuracy"], v["bits_per_byte"]))
    print("=" * 74)


def generate(model, ids, n, eot, temp=0.0, top_p=1.0, penalty=1.0, window=64):
    """Tek istem, token token, onbellekli (istem bir kez, sonra yalniz yeni konum).  <|endoftext|>'te durur."""
    out, x = [], list(ids)
    caches = [AttentionCache(torch.tensor([len(x)]), len(x) + n) for _ in range(model.turns)]
    with torch.no_grad():
        for i in range(n):
            logits = model.logits(torch.tensor([x if i == 0 else x[-1:]]), caches)[0, -1].float()
            if penalty != 1.0:                              # son window token: pozitif puan bolunur, negatif carpilir
                recent = torch.tensor(sorted(set(x[-window:])))
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
            if t == eot:
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
    print("\nmodel arka planda yukleniyor (Drive'dan ilk okuma birkac dakika surebilir) -- istemini simdiden yazabilirsin\n")
    state = {}

    def _load():
        try:
            state["run"] = find_run(sys.argv[1] if len(sys.argv) > 1 else None)
            if state["run"]:
                state["vocab"] = load_vocab()
                state["model"], state["cfg"], state["label"] = load_model(state["run"])
        except BaseException as e:
            state["error"] = e

    loader = threading.Thread(target=_load, daemon=True)
    loader.start()
    ready = False
    n, temp, top_p, penalty = 200, 0.0, 0.9, 1.0
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
        if not ready:
            if loader.is_alive():
                print("   model yukleniyor, bekle...")
                loader.join()
            if "error" in state or not state.get("run") or state.get("model") is None:
                raise SystemExit(state.get("error") or 1)
            summary(state["run"], state["cfg"], state["label"])
            ready = True
        if g == "model":
            for r in runs()[:15]:
                print("   %s%s" % (os.path.basename(r), "   <- yuklu" if r == state["run"] else ""))
            continue
        if g.startswith("model ") or g in ("ema", "ck"):
            run = find_run(g[6:].strip()) if g.startswith("model ") else state["run"]
            weights = {"ema": "last" if state["label"] == "ORTALAMA agirliklar" else "ema", "ck": "ck"}.get(g, "auto")
            if run:
                m, cfg, label = load_model(run, weights)
                if m is not None:
                    state.update(run=run, model=m, cfg=cfg, label=label)
                    summary(run, cfg, label)
            continue
        vocab = state["vocab"]
        eot = vocab.index(DS.EOS_TOKEN)                      # sozlukte eot'un yazimi (DS._load)
        if g == "serbest":
            g = ""
        elif not g:
            g = random.choice(EF.PROMPTS)
        ids = [eot] + DS.encode(g, vocab)
        out = generate(state["model"], ids, n, eot, temp, top_p, penalty)
        print()
        print_wrapped("ISTEM", DS.decode(ids[1:], vocab) or "(bos)")
        print_wrapped("MODEL  (sicaklik %.2f, top-p %.2f, tekrar cezasi %.2f)%s" % (
            temp, top_p, penalty, "" if len(out) < n else "  -- sinira geldi, kesildi"), DS.decode(out, vocab))
        print()


if __name__ == "__main__":
    main()
