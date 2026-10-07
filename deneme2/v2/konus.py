"""Modelle KONUS (V2: Model Z + G ve transformer; SS ve FineWeb-Edu kosulari) -- bir metin baslangici yaz, model devamini
yazsin.  Kullanici, 7 Ekim: "daha önceki modellerde yazdığımız gibi konus py ve konus bat yaz" (model_y konus_fineweb'in
V2 karsiligi).

Agirlik Colab'da uretilip Drive'a yaziliyor (v2/runs/<kosu>/agent.pt); burasi yalniz OKUR ve yerel onbellege alir
(konus_cache/<kosu>, git disi).  Model CPU'da.  Istem egitimdeki kuralla cumlelere bolunur (veri klasorunun profili: ss /
web); istem noktalama ile bitmiyorsa son cumle acik kalir, model onu tamamlar (open_last, bilgi sinavindaki gibi).

  The capital of France is   yaz, devamini gorursun
  <bos satir>                bilgi sinavinin istemlerinden (FineWeb) ya da bir SS acilisindan RASTGELE biri
  n=8                        en fazla kac cumle (varsayilan 8)
  s=0                        acgozlu (sinavdaki gibi, varsayilan);  s=1 ornekleme (sicaklik 1, okumalardaki gibi)
  model                      kosulari listele (en yeni once), hangisi yuklu
  model <ad>                 baska bir kosu yukle (ad ya da basindan bir parca)
  ?                          yardim
  q                          cik
"""
import json
import os
import random
import shutil
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "common"), os.path.join(HERE, "diag")]
torch = np = D = T = GR = None                                 # agir moduller ilk kullanimda: pencere hemen acilir

RUN_ROOTS = [r"G:\Drive'ım\v2\runs", r"G:\Drivem\v2\runs"]
COLAB_DRIVE = "/content/drive/MyDrive/"
CACHE_DIR = os.path.join(HERE, "konus_cache")              # git disi (.gitignore)
SS_OPENINGS = ["Once upon a time,", "One day, a little girl named", "In a small village,", "The old dog"]
MAX_TOKENS = 128                                          # cumle basina (web profilinin MAX_SENTENCE_TOKENS'i)


def _heavy():
    global torch, np, D, T, GR
    if D is None:
        import torch as _torch                             # torch pyarrow'dan once (Windows c10.dll)
        import numpy as _np
        import data as _d
        import train as _t
        import generate_readings as _gr
        torch, np, D, T, GR = _torch, _np, _d, _t, _gr
        if os.name == "nt":
            T._no_power_throttling()                        # kural 2a


def _drive(path):
    """Colab yolu -> yerel Drive yolu."""
    root = next((os.path.dirname(r.rstrip("\\/")) for r in RUN_ROOTS if os.path.isdir(r)), None)
    root = os.path.dirname(root) if root else None          # G:\Drive'ım
    return os.path.join(root, *path[len(COLAB_DRIVE):].split("/")) if path.startswith(COLAB_DRIVE) else path


def runs():
    """agent.pt'si olan kosular, en son degisen once."""
    d = [os.path.join(r, x) for r in RUN_ROOTS if os.path.isdir(r) for x in os.listdir(r)]
    d = [x for x in d if os.path.exists(os.path.join(x, "agent.pt"))]
    return sorted(d, key=lambda x: os.path.getmtime(os.path.join(x, "agent.pt")), reverse=True)


def find_run(pattern=None):
    all_runs = runs()
    if not all_runs:
        print("KOSU YOK.  Aranan: %s\\<kosu>\\agent.pt" % " ya da ".join(RUN_ROOTS))
        return None
    if not pattern:
        return all_runs[0]
    hit = [r for r in all_runs if os.path.basename(r) == pattern] or \
          [r for r in all_runs if os.path.basename(r).startswith(pattern)]
    if not hit:
        print("'%s' ile eslesen kosu yok.  'model' yazip listeye bak." % pattern)
        return None
    return hit[0]


def load(run_dir):
    """agent.pt -> yerel onbellek (Drive'daki daha yeniyse yeniden) -> (model, kimlik, tokenizer, profil, sinir tablolari)."""
    _heavy()
    from tokenizers import Tokenizer
    cache = os.path.join(CACHE_DIR, os.path.basename(run_dir))
    os.makedirs(cache, exist_ok=True)
    src, local = os.path.join(run_dir, "agent.pt"), os.path.join(cache, "agent.pt")
    if not os.path.exists(local) or os.path.getmtime(local) < os.path.getmtime(src):
        print("   agent.pt yerel onbellege kopyalaniyor (%.2f GB)..." % (os.path.getsize(src) / 1e9))
        shutil.copyfile(src, local + ".part")
        os.replace(local + ".part", local)
    args = torch.load(local, map_location="cpu", weights_only=False)["args"]
    data, stream = _drive(args["data"]), _drive(args["stream"])
    model, idt = GR.load_run(cache, data, torch.device("cpu"))
    tok = Tokenizer.from_file(os.path.join(stream, "gpt2", "tokenizer.json"))
    profile = json.load(open(os.path.join(data, "train_boundaries.json"), encoding="utf-8")).get("profile", "ss")
    return dict(run=run_dir, model=model.eval(), idt=idt, tok=tok, profile=profile, data=data,
                tables=D.stream_tables(tok, profile))


def summary(st):
    i = st["idt"]
    print("=" * 74)
    print("kosu    %s" % os.path.basename(st["run"]))
    print("model   %s  d %s  katman %s  head %s  global %s  optimizer %s  lr %s   veri %s (%s profili)" % (
        i["model"], i["d"], i["layers"], i["heads"], i.get("global_layers", 0), i.get("optimizer", "adamw"), i["lr"],
        os.path.basename(st["data"]), st["profile"]))
    r = os.path.join(st["run"], "results.json")
    if os.path.exists(r):
        ex = (json.load(open(r, encoding="utf-8")).get("exams") or [{}])[-1]
        if "loss" in ex:
            print("sinav   kayip %.4f  bpb %.4f  acc %.4f  (adim %s)" % (ex["loss"], ex["bits_per_byte"], ex["acc"],
                                                                    ex.get("step")))
    print("=" * 74)


def sentences(st, text):
    """Istem -> egitimdeki kuralla cumleler (token listeleri) ve son cumle acik mi."""
    tok = st["tok"]
    x = np.array(tok.encode(text).ids + [D.EOS_ID], np.int64)
    sents = [x[s:e].tolist() for s, e in D._boundaries(x, *st["tables"], tok, st["profile"])[0]]
    return sents, not text.rstrip().endswith((".", "!", "?", '"', "\u201d", ":"))


def random_prompt(st):
    if st["profile"] == "web":
        facts = json.load(open(os.path.join(HERE, "diag", "facts.json"), encoding="utf-8"))["facts"]
        return random.choice(facts)["prompt"]
    return random.choice(SS_OPENINGS)


def main():
    print(__doc__.split("\n\n")[-1].rstrip())
    print("\nmodel arka planda yukleniyor (ilk kopya Drive'dan birkac dakika surebilir) -- istemini simdiden yazabilirsin\n")
    state, box = {}, {}

    def _load():
        try:
            run = find_run(sys.argv[1] if len(sys.argv) > 1 else None)
            if run:
                state.update(load(run))
        except BaseException as e:                          # noqa: BLE001
            box["error"] = e

    loader = threading.Thread(target=_load, daemon=True)
    loader.start()
    ready, n, sample = False, 8, False
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
            print("   en fazla %d cumle" % n)
            continue
        if g.startswith("s="):
            sample = float(g[2:]) > 0
            print("   %s" % ("ornekleme (sicaklik 1)" if sample else "acgozlu (sinavdaki gibi)"))
            continue
        if not ready:
            if loader.is_alive():
                print("   model yukleniyor, bekle...")
                loader.join()
            if "error" in box or "model" not in state:
                raise SystemExit(box.get("error") or 1)
            summary(state)
            ready = True
        if g == "model":
            for r in runs()[:15]:
                print("   %s%s" % (os.path.basename(r), "   <- yuklu" if r == state["run"] else ""))
            continue
        if g.startswith("model "):
            run = find_run(g[6:].strip())
            if run:
                state.update(load(run))
                summary(state)
            continue
        if not g:
            g = random_prompt(state)
        sents, opened = sentences(state, g)
        tok = state["tok"]
        print("\nISTEM  %s   (%d cumle%s)" % (g, len(sents), ", son cumle acik" if opened else ""))
        print("MODEL  (%s)  -- akiyor, Ctrl+C keser" % ("ornekleme" if sample else "acgozlu"))
        sys.stdout.write("   " + (tok.decode(sents[-1]) if opened else ""))
        shown = {"ids": [], "text": "", "col": 0}

        def stream(w):                                      # token token bas, 78 sutunda kir; cumle sonu yeni satir degil
            if w is None:
                return
            shown["ids"].append(w)
            text = tok.decode(shown["ids"])
            if text.endswith("\ufffd"):                     # yarim UTF-8 bayti: sonraki token'i bekle
                return
            new, shown["text"] = text[len(shown["text"]):], text
            for ch in new:
                if ch == "\n" or (ch == " " and shown["col"] >= 75):
                    sys.stdout.write("\n   ")
                    shown["col"] = 0
                else:
                    sys.stdout.write(ch)
                    shown["col"] += 1
            sys.stdout.flush()

        gen = torch.Generator().manual_seed(random.randrange(1 << 31)) if sample else None
        try:
            with torch.no_grad():
                (out, ended, eos), = state["model"].generate([sents], n, MAX_TOKENS, generator=gen, open_last=opened,
                                                             on_token=stream)
            print("\n   -- %d cumle, %d token%s" % (len(out), sum(map(len, out)),
                                                  ", model durdu (EOS)" if eos else ", cumle sinirina geldi"))
        except KeyboardInterrupt:                           # Ctrl+C: o ana kadarki cikti ekranda kalir
            print("\n   -- kesildi")
        print()


if __name__ == "__main__":
    main()
