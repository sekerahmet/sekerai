# -*- coding: utf-8 -*-
"""GPU analiz kuyrugunun isci adimi (Colab hucre "12 ISCI" her turda bu modulu git'ten yeniden yukler; kural degisince
hucreyi yeniden baslatmak gerekmez).  Kullanici, 2 Ekim: "benim sürekli takip etmem zor oluyor kaçırabilirim".

Kuyruk: RUNS_ROOT/kuyruk/istek_<ad>.json {"komut": "python <betik>.py ...", "sahip": "A", "aciklama": "..."} ->
sonuc_<ad>/ (istek.json, cikti.log, durum.json).  Yalniz depodaki betik, kabuk karakteri yok.
Sira (belge/ajanlar/EKIP_KURALLARI.md A3): oncelik = sahibin harcadigi GPU suresi + isin OLCULMUS tahmini suresi (ayni
betik + alt komutun gecmis medyani; yoksa bitmis islerin medyani); en kucuk once, esitse en eski.  Ajan beyani kullanilmaz.
KESILDI kalan is (hucre kesilince) kendiliginden kuyruga geri konur.
"""
import json
import os
import shlex
import statistics
import subprocess
import time

TIMEOUT = 3 * 3600
SHELL_CHARS = ';|&><`$\n'


def _check(cmd, src, repo):
    if any(c in cmd for c in SHELL_CHARS):
        return "kabuk karakteri yasak"
    a = shlex.split(cmd)
    if len(a) < 2 or a[0] != "python" or not a[1].endswith(".py"):
        return "'python <betik>.py ...' bicimi gerekli"
    p = os.path.realpath(os.path.join(src, a[1]))
    if not p.startswith(os.path.realpath(repo) + "/") or not os.path.exists(p):
        return "betik depoda yok: %s" % a[1]
    return None


def job_key(cmd):
    """Is turu: betik + ilk yer tutucu / yol / bayrak olmayan arguman (alt komut)."""
    a = shlex.split(cmd)
    sub = next((x for x in a[2:] if not x.startswith(("{", "/", "-"))), "")
    return a[1] + " " + sub if len(a) > 1 else cmd


def _load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def requeue_interrupted(q):
    """KESILDI ya da durum.json'suz (yarim) sonuc klasorunu kuyruga geri koyar; doner: geri konan adlar."""
    back = []
    for d in os.listdir(q):
        if not d.startswith("sonuc_"):
            continue
        st = _load(os.path.join(q, d, "durum.json"))
        if st is not None and st.get("sonuc") == "KESILDI" and os.path.exists(os.path.join(q, d, "istek.json")):
            name = d[len("sonuc_"):]
            if not os.path.exists(os.path.join(q, "istek_%s.json" % name)):
                with open(os.path.join(q, d, "istek.json"), encoding="utf-8") as f:
                    body = f.read()
                with open(os.path.join(q, "istek_%s.json" % name), "w", encoding="utf-8") as f:
                    f.write(body)
                os.remove(os.path.join(q, d, "durum.json"))
                back.append(name)
    return back


def history(q):
    """Bitmis isler: sahip -> harcanan sn; is turu -> sureler; varsayilan tahmin (medyan)."""
    used, durs, alld = {}, {}, []
    for d in os.listdir(q):
        if not d.startswith("sonuc_"):
            continue
        st = _load(os.path.join(q, d, "durum.json"))
        if not st:
            continue
        s = float(st.get("sure_sn") or 0)
        used[st.get("sahip")] = used.get(st.get("sahip"), 0) + s
        req = _load(os.path.join(q, d, "istek.json"))
        if st.get("sonuc") == "TAMAM" and req and req.get("komut"):
            durs.setdefault(job_key(req["komut"]), []).append(s)
            alld.append(s)
    return used, durs, (statistics.median(alld) if alld else 600.0)


def order(q):
    """Kuyruktaki istekler, oncelik sirasiyla: [(oncelik, zaman, dosya, tahmin_sn, sahip)]."""
    used, durs, default = history(q)
    rows = []
    for f in os.listdir(q):
        if not (f.startswith("istek_") and f.endswith(".json")):
            continue
        r = _load(os.path.join(q, f)) or {}
        k = job_key(r["komut"]) if r.get("komut") else None
        est = statistics.median(durs[k]) if k in durs else default
        rows.append((used.get(r.get("sahip"), 0) + est, os.path.getmtime(os.path.join(q, f)), f, est, r.get("sahip")))
    rows.sort()
    return rows, used


def step(ctx):
    """Bir tur: ekrani bas, siradaki isi kos (yoksa 30 sn bekle).  ctx: hucrenin globals'i (RUNS_ROOT, FW_ROOT, SRC, REPO,
    _git, torch, clear_output); ctx['done_log'] turlar arasi kalir."""
    q = ctx["RUNS_ROOT"] + "/kuyruk"
    os.makedirs(q, exist_ok=True)
    log = ctx.setdefault("done_log", [])
    for name in requeue_interrupted(q):
        log.append("%s  %s  KESILDI -> kuyruga geri kondu" % (time.strftime("%H:%M"), name))
    rows, used = order(q)
    ctx["clear_output"](wait=True)
    print("ISCI  %s (UTC)  kuyrukta %d  |  %s" % (time.strftime("%H:%M"), len(rows), ctx["_git"]("log", "-1", "--format=%h %s")[:60]))
    print("   harcanan GPU sn: %s" % {k: round(v) for k, v in used.items()})
    print("   sira (oncelik = harcanan + olculmus tahmin): %s" % " > ".join("%s(~%.0fs)" % (x[2][6:-5], x[3]) for x in rows[:6]))
    for line in log[-8:]:
        print("   " + line)
    torch = ctx["torch"]
    if torch.cuda.is_available():
        print("gpu %s  bos %.1f GB" % (torch.cuda.get_device_name(0), torch.cuda.mem_get_info()[0] / 1e9))
    if not rows:
        time.sleep(30)
        return
    pick, est = rows[0][2], rows[0][3]
    name = pick[len("istek_"):-len(".json")]
    out = q + "/sonuc_" + name
    os.makedirs(out, exist_ok=True)
    st = dict(ad=name, basla=time.strftime("%Y-%m-%d %H:%M:%S"), tahmin_sn=round(est))
    subs = {"{RUN}": ctx["RUN"], "{RUNS_ROOT}": ctx["RUNS_ROOT"], "{FW_ROOT}": ctx["FW_ROOT"], "{SRC}": ctx["SRC"]}
    try:
        os.replace(os.path.join(q, pick), out + "/istek.json")      # once tasi: bozuk istek kuyrugu kilitlemesin
        req = json.load(open(out + "/istek.json", encoding="utf-8"))
        st["sahip"] = req.get("sahip")
        st["kod"] = ctx["_git"]("rev-parse", "--short", "HEAD")
        cmd = req["komut"]
        for k, v in subs.items():
            cmd = cmd.replace(k, v)
        st["komut"] = cmd
        bad = _check(req["komut"], ctx["SRC"], ctx["REPO"])
        if bad:
            st.update(sonuc="REDDEDILDI", neden=bad)
        else:
            t0 = time.time()
            with open(out + "/cikti.log", "w") as f:
                r = subprocess.run(shlex.split(cmd), cwd=ctx["SRC"], stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT,
                                   env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                                            RUNS_ROOT=ctx["RUNS_ROOT"], FW_ROOT=ctx["FW_ROOT"], KUYRUK_SONUC=out))
            st.update(sonuc="TAMAM" if r.returncode == 0 else "HATA", kod_cikis=r.returncode, sure_sn=round(time.time() - t0))
    except subprocess.TimeoutExpired:
        st.update(sonuc="ZAMAN ASIMI", sure_sn=TIMEOUT)
    except KeyboardInterrupt:
        st.update(sonuc="KESILDI")
        json.dump(st, open(out + "/durum.json", "w"), indent=1, ensure_ascii=False)
        raise
    except Exception as e:
        st.update(sonuc="HATA", neden=repr(e))
    st["bitis"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(st, open(out + "/durum.json", "w"), indent=1, ensure_ascii=False)
    log.append("%s  %s  %s  %s sn (tahmin %s)  (%s)" % (st["bitis"][11:16], name, st["sonuc"], st.get("sure_sn", "-"),
                                                       st.get("tahmin_sn"), st.get("sahip")))
