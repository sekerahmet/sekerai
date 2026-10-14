"""Colab koşu kontrolcüsü: hazırla / koş / kuyruk / nabız / kapat.  Eğitimler arka planda ayrı süreçtir; notebook hücresi ya
da bu oturum serbest kalır, koşu Drive'daki runs/<koşu>/ klasörüne yazar.

Notebook'tan:
    import sys; sys.path.insert(0, "/content/RGN/colab"); import runner as R
    R.hazirla()                                        # Drive + kod + bağımlılık + ortam raporu
    R.kos("configs/ss/ss_smoke.json")                  # arka planda başlat
    R.kuyruk(["configs/ss/ss_d512_rgn.json", "configs/ss/ss_d512_transformer.json"])   # sırayla, arka planda
    R.arkaplan("hiz", "python scripts/bench_speed.py attention")                       # herhangi bir komut
    R.nabiz()                                          # tüm işler: durum, son adım, son değerlendirme, kalan süre
    R.nabiz("ss_smoke", satir=30)                      # tek iş, günlüğün son 30 satırı
    R.kapat("ss_smoke")                                # SIGTERM: checkpoint kaydedip çıkar (sonra kos(..., resume=True))
    R.sonuc()                                          # biten koşular: sayı + metin + hız

Komut satırından (aynı işlevler):
    python colab/runner.py kos configs/ss/ss_smoke.json
    python colab/runner.py nabiz
    python colab/runner.py kapat ss_smoke
"""
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from rgn.config import load_config  # noqa: E402
from rgn.paths import runs_dir  # noqa: E402

JOBS = Path(os.environ.get("RGN_JOBS", "/content/rgn_jobs.json" if Path("/content").is_dir() else CODE / "runs" / "jobs.json"))


def _jobs():
    return json.loads(JOBS.read_text(encoding="utf-8")) if JOBS.exists() else {}


def _save_jobs(j):
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(j, indent=1, ensure_ascii=False), encoding="utf-8")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    try:                                              # zombi süreç (Linux) canlı sayılmaz
        return "Z" not in Path("/proc/%d/stat" % pid).read_text().split()[2]
    except (OSError, IndexError):
        return True


def _spawn(name, cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "a", encoding="utf-8")
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                         env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    j = _jobs()
    j[name] = {"pid": p.pid, "cmd": cmd, "log": str(log), "start": time.time()}
    _save_jobs(j)
    print("[başladı] %s (pid %d) — günlük: %s" % (name, p.pid, log))
    return name


def hazirla(git=None, branch="main", subdir=""):
    """setup_colab.py'yi çalıştırır (Drive bağlı olmalı)."""
    args = "" if not git else "--git %s --branch %s --subdir %s" % (git, branch, subdir)
    subprocess.run("%s %s %s" % (sys.executable, CODE / "colab" / "setup_colab.py", args), shell=True, check=True)


def kos(config, *sets, name=None, resume=True):
    """Tek eğitimi arka planda başlat.  sets: "train.max_steps=200" gibi ezmeler."""
    cfg = load_config(CODE / config, sets)
    name = name or cfg.name
    if name in _jobs() and _alive(_jobs()[name]["pid"]):
        print("[zaten çalışıyor] %s" % name)
        return name
    cmd = "%s scripts/train.py --config %s --name %s %s %s" % (
        shlex.quote(sys.executable), shlex.quote(config), name, " ".join("--set %s" % shlex.quote(s) for s in sets),
        "--resume" if resume else "")
    return _spawn(name, cmd, runs_dir() / name / "stdout.log")


def arkaplan(name, cmd):
    """Herhangi bir komutu (test, hız profili, veri hazırlama) arka planda çalıştır; günlük runs/_<ad>.log."""
    return _spawn(name, cmd, runs_dir() / ("_%s.log" % name))


def kuyruk(configs, name="kuyruk", resume=True):
    """Yapılandırmaları sırayla, tek arka plan sürecinde çalıştır (biri bitince ya da hata verince sonraki)."""
    steps = " ; ".join("%s scripts/train.py --config %s %s" % (shlex.quote(sys.executable), shlex.quote(c),
                                                                "--resume" if resume else "") for c in configs)
    j = _jobs()
    j.setdefault("_kuyruk_icerik", {})[name] = [load_config(CODE / c).name for c in configs]
    _save_jobs(j)
    return _spawn(name, steps, runs_dir() / ("_%s.log" % name))


# Kaba kıyas çapası (farklı veri düzeni: END token'ları, hikâye maskesi, 1000 hikâyelik sınav) — yalnız yön göstergesi
ANCHORS = {"ss": ("tf v2 d512 (AI_NEW_MODEL, farklı düzen)", 0.4732)}


def _metrics(d):
    out = {}
    p = d / "metrics.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:                # Drive eşitlemesi yarım satır bırakabilir
                    continue
                out.setdefault(r.get("type"), []).append(r)
    return out


def _plan(d, m):
    """-> (toplam adım, adım başına token): 'start' kaydından ya da günlüğün başlık satırından."""
    if m.get("start"):
        s = m["start"][-1]
        return s["total_steps"], s["tokens_per_step"]
    log = d / "stdout.log"
    if log.exists():
        for line in open(log, encoding="utf-8", errors="replace"):
            g = re.search(r"(\d+) adım × (\d+) token", line)
            if g:
                return int(g.group(1)), int(g.group(2))
    return None, None


def _bar(frac, n=24):
    k = max(0, min(n, int(round(frac * n))))
    return "█" * k + "░" * (n - k)


def _run_status(run):
    """Tek satır özet (bitmiş koşular ve sonuc() için)."""
    d = runs_dir() / run
    if (d / "results.json").exists():
        r = json.loads((d / "results.json").read_text(encoding="utf-8"))
        return "%-28s BİTTİ   kayıp %.4f  bpb %s  %.1f dk" % (run, r["final"]["loss"], "%.4f" % r["final"]["bpb"]
                                                              if r["final"]["bpb"] else "-", r["wall_min"])
    m = _metrics(d)
    total, _ = _plan(d, m)
    tr = (m.get("train") or [None])[-1]
    if not tr:
        return "%-28s %s" % (run, "DURDU" if (d / "stopped").exists() else "başlamadı / derleniyor")
    return "%-28s %sadım %d/%s  kayıp %.4f  %.0f ms" % (run, "DURDU  " if (d / "stopped").exists() else "",
                                                       tr["step"], total or "?", tr["loss"], tr["ms"])


def _run_detail(run):
    """Ayrıntılı nabız: ilerleme, eğitim, değerlendirme (+ fark), teşhis, örnek, kaba kıyas."""
    d = runs_dir() / run
    m = _metrics(d)
    total, tps = _plan(d, m)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else None
    done = (d / "results.json").exists()
    state = "BİTTİ" if done else "DURDU" if (d / "stopped").exists() else "KOŞUYOR" if m.get("train") else "BAŞLIYOR"
    L = ["%s  %s" % (run, state)]
    if cfg:
        mc = cfg["model"]
        L[0] += "   attn=%s pos=%s mlp=%s p=%s  D %d L %d H %d  |  veri %s pencere %d" % (
            mc["attn"], mc["pos"], mc["mlp"], mc.get("power", 1), mc["d_model"], mc["n_layers"], mc["n_heads"],
            cfg["data"]["dataset"], cfg["data"]["seq_len"])
    tr = m.get("train") or []
    if tr and total:
        last = tr[-1]
        frac = last["step"] / total
        ms = statistics.median(r["ms"] for r in tr[-5:])
        L.append("   ilerleme %s %5.1f%%  adım %d/%d  %.0fM token  kalan ~%.0f dk" % (
            _bar(frac), 100 * frac, last["step"], total, last["step"] * tps / 1e6, (total - last["step"]) * ms / 60000))
        k = len(tr[-10:])
        avg = sum(r["loss"] for r in tr[-10:]) / k
        L.append("   eğitim   kayıp %.4f (son %d ort. %.4f)  lr %.2e  gn %.3f  |  %.0f ms/adım  %.0fk tok/s  %.1f GB" % (
            last["loss"], k, avg, last["lr"], last["grad_norm"], ms, last["tok_s"] / 1e3, last["mem_gb"]))
        L.append("   son 5    " + "   ".join("%d: %.4f (gn %.2f, %.0f ms)" % (r["step"], r["loss"], r["grad_norm"], r["ms"])
                                          for r in tr[-5:]))
    ev = m.get("eval") or []
    if ev:
        e = ev[-1]
        delta = " (%+.4f)" % (e["loss"] - ev[-2]["loss"]) if len(ev) > 1 else ""
        s = "   değerl.  adım %d%s: kayıp %.4f%s  ppl %.2f  bpb %s  doğruluk %.4f" % (
            e["step"], " SON" if e.get("final") else "", e["loss"], delta, e["ppl"],
            "%.4f" % e["bpb"] if e.get("bpb") else "-", e["acc"])
        if e.get("acc_bands"):
            s += "  bant " + " ".join("%s:%.3f" % kv for kv in e["acc_bands"].items())
        if tr:
            s += "  |  eğitim−değerl. %+.3f" % (sum(r["loss"] for r in tr[-10:]) / len(tr[-10:]) - e["loss"])
        L.append(s)
        anchor = ANCHORS.get(cfg["data"]["dataset"]) if cfg else None
        if anchor and e.get("bpb"):
            L.append("   kıyas    %s bpb %.4f  →  fark %+.4f (kaba)" % (anchor[0], anchor[1], e["bpb"] - anchor[1]))
        for pr in (e.get("probes") or [])[:2]:
            L.append("   örnek    farklı4 %.2f%s  » %s" % (pr["distinct4"], "  DÖNGÜ" if pr["loop"] else "",
                                                         pr["text"][:150].replace("\n", " ")))
    dg = m.get("diag") or []
    if dg:
        g = dg[-1]
        lay = g["layers"]
        ent = [sum(l["entropy"]) / len(l["entropy"]) for l in lay]
        top1 = sum(sum(l["top1"]) / len(l["top1"]) for l in lay) / len(lay)
        dist = sum(sum(l["mean_dist"]) / len(l["mean_dist"]) for l in lay) / len(lay)
        sat = [l["mlp"].get("sat_frac") for l in lay if l.get("mlp")]
        s = "   teşhis   adım %d: entropi %.2f  top1 %.3f  mesafe %.0f" % (g["step"], sum(ent) / len(ent), top1, dist)
        if sat:
            s += "  |u|>3 %.4f" % (sum(sat) / len(sat))
        gam = [x for l in lay for x in (l.get("gamma") or [])]
        if gam:
            s += "  γ ort %.0f [%.0f–%.0f]" % (sum(gam) / len(gam), min(gam), max(gam))
        L.append(s)
        L.append("            katman entropisi " + " ".join("%.2f" % x for x in ent))
    return L


def _gpu():
    try:
        out = subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader",
                             shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
        return "GPU: bellek / toplam / kullanım = %s" % out if out else ""
    except Exception:
        return ""


def nabiz(name=None, satir=0):
    """Ayrıntılı durum: her iş ve koşusu için ilerleme, eğitim, değerlendirme, teşhis, örnek; satir > 0 ise günlüğün son
    satırları.  Bitmiş koşular tek satır."""
    j = _jobs()
    groups = j.get("_kuyruk_icerik", {})
    names = [name] if name else [k for k in j if not k.startswith("_")]
    if not names:
        print("iş yok")
    for n in names:
        job = j.get(n)
        alive = bool(job and _alive(job["pid"]))
        print("[%s] %s%s" % ("ÇALIŞIYOR" if alive else "durdu/bitti", n,
                            "  (%.1f dk önce başladı)" % ((time.time() - job["start"]) / 60) if job else ""))
        for run in groups.get(n, [n]):
            d = runs_dir() / run
            if not d.exists():
                print("   %-28s sırada" % run)
                continue
            detail = _run_detail(run) if (alive or not (d / "results.json").exists()) else [_run_status(run)]
            for line in detail:
                print("   " + line)
        if satir and job:
            lines = Path(job["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-satir:]
            print("\n".join("   | " + l for l in lines))
    g = _gpu()
    if g:
        print(g)


def sonuc(names=None, metin=2, karakter=400):
    """Biten koşular: sayı (son değerlendirme, hız, bellek) + metin (greedy örnekler) + hız profili özeti."""
    root = runs_dir()
    runs = names or sorted(p.parent.name for p in root.glob("*/results.json"))
    for run in runs:
        d = root / run
        if not (d / "results.json").exists():
            print("%-28s bitmedi — %s" % (run, _run_status(run).strip()))
            continue
        r = json.loads((d / "results.json").read_text(encoding="utf-8"))
        f, m = r["final"], r["config"]["model"]
        print("\n== %s  (attn=%s pos=%s mlp=%s p=%s, %.1fM)" % (run, m["attn"], m["pos"], m["mlp"], m.get("power", 1),
                                                              r["params"] / 1e6))
        print("   kayıp %.4f  ppl %.2f  bpb %s  doğruluk %.4f  |  %s ms/adım  %s tok/s  %.1f GB  |  %.1f dk" % (
            f["loss"], f["ppl"], "%.4f" % f["bpb"] if f["bpb"] else "-", f["acc"],
            "%.1f" % r["ms_per_step"] if r["ms_per_step"] else "-",
            "%.0fk" % (r["tok_per_s"] / 1e3) if r["tok_per_s"] else "-", r["peak_mem_gb"] or 0, r["wall_min"]))
        sp = (r.get("speed") or {}).get("attention") or {}
        if "model" in sp:
            print("   dikkat çekirdeği %.2f ms (softmax %.2f ms, oran %.1f)" % (
                sp["model"]["ms"], sp["softmax_sdpa"]["ms"], sp["ratio_model_vs_softmax"]))
        if (d / "samples.json").exists():
            for s in json.loads((d / "samples.json").read_text(encoding="utf-8"))[:metin]:
                print("   » %s" % s["text"][:karakter].replace("\n", " "))


def kapat(name):
    """SIGTERM gönder: eğitim checkpoint kaydedip çıkar.  Kuyruk adı verilirse kuyruk süreci grubu durur."""
    job = _jobs().get(name)
    if not job or not _alive(job["pid"]):
        print("[çalışmıyor] %s" % name)
        return
    os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
    print("[kapatılıyor] %s — checkpoint kaydedilip çıkılacak (adım sonunda)" % name)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["nabiz"]
    if cmd == "kos":
        kos(rest[0], *rest[1:])
    elif cmd == "kuyruk":
        kuyruk(rest)
    elif cmd == "arkaplan":
        arkaplan(rest[0], " ".join(rest[1:]))
    elif cmd == "nabiz":
        nabiz(*(rest[:1]), satir=int(rest[1]) if len(rest) > 1 else 0)
    elif cmd == "kapat":
        kapat(rest[0])
    elif cmd == "sonuc":
        sonuc(rest or None)
    elif cmd == "hazirla":
        hazirla(*rest)
    else:
        sys.exit("komutlar: hazirla | kos <config> [ezmeler] | kuyruk <config>... | arkaplan <ad> <komut> | nabiz [ad [satır]] | sonuc [ad...] | kapat <ad>")
