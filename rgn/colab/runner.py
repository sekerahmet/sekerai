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

Komut satırından (aynı işlevler):
    python colab/runner.py kos configs/ss/ss_smoke.json
    python colab/runner.py nabiz
    python colab/runner.py kapat ss_smoke
"""
import json
import os
import shlex
import signal
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


def _last(path, kind):
    if not Path(path).exists():
        return None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"type": "%s"' % kind in line:
                last = line
    return json.loads(last) if last else None


def _run_status(run):
    d = runs_dir() / run
    cfg = d / "config.json"
    tr, ev = _last(d / "metrics.jsonl", "train"), _last(d / "metrics.jsonl", "eval")
    s = "%-28s" % run
    if (d / "results.json").exists():
        r = json.loads((d / "results.json").read_text(encoding="utf-8"))
        return s + " BİTTİ   kayıp %.4f  bpb %s  %.1f dk" % (r["final"]["loss"], "%.4f" % r["final"]["bpb"]
                                                             if r["final"]["bpb"] else "-", r["wall_min"])
    if (d / "stopped").exists():
        s += " DURDU   "
    if tr:
        c = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else None
        s += " adım %d  kayıp %.4f  %.0f ms  %.0fk tok/s" % (tr["step"], tr["loss"], tr["ms"], tr["tok_s"] / 1e3)
        log = runs_dir() / run / "stdout.log"
        total = None
        if log.exists():
            for line in open(log, encoding="utf-8", errors="replace"):
                if line.startswith("adım ") and "/" in line:
                    total = int(line.split()[1].split("/")[1])
        if total:
            s += "  kalan ~%.0f dk" % ((total - tr["step"]) * tr["ms"] / 60000)
    if ev:
        s += "  | değerlendirme %.4f" % ev["loss"]
    return s


def nabiz(name=None, satir=0):
    """Durum özeti; name verilirse o işin günlüğünün son `satir` satırı da."""
    j = _jobs()
    groups = j.get("_kuyruk_icerik", {})
    names = [name] if name else [k for k in j if not k.startswith("_")]
    if not names:
        print("iş yok")
    for n in names:
        job = j.get(n)
        alive = job and _alive(job["pid"])
        print("[%s] %s%s" % ("ÇALIŞIYOR" if alive else "durdu/bitti", n,
                            "  (%.0f dk)" % ((time.time() - job["start"]) / 60) if job else ""))
        for run in groups.get(n, [n]):
            print("   " + _run_status(run))
        if satir and job:
            lines = Path(job["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-satir:]
            print("\n".join("   | " + l for l in lines))


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
    elif cmd == "hazirla":
        hazirla(*rest)
    else:
        sys.exit("komutlar: hazirla | kos <config> [ezmeler] | kuyruk <config>... | arkaplan <ad> <komut> | nabiz [ad [satır]] | kapat <ad>")
