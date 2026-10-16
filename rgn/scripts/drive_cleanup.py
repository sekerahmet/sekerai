"""Drive temizliği (kullanıcı onayıyla, 17 Ekim 2026).  Varsayılan yalnız listeler; --apply siler (Drive çöp kutusuna gider).

Kurallar (yalnız bunlar; veri dosyaları, videolar ve RGN koşuları dokunulmaz):
  1. model_y/<koşu>/checkpoint_t*.pt   : koşu başına en son adımdaki kalır, ara checkpoint'ler silinir
  2. v2/runs/**/decay_start/*.pt        : iniş başı kopyaları (asıl checkpoint ve sonuç dosyaları kalır)
  3. v2/inductor_cache/                 : derleme önbelleği (gerekirse yeniden oluşur)
  4. v2/profile/**/*.pt                 : hız profili koşularının checkpoint'leri (günlükler kalır)

    python scripts/drive_cleanup.py            # liste + toplam
    python scripts/drive_cleanup.py --apply    # sil
"""
import argparse
import os
import re
import shutil

ROOT = "/content/drive/MyDrive"


def plan():
    files, dirs = [], []
    my = os.path.join(ROOT, "model_y")
    for dp, _, fs in os.walk(my):
        ck = sorted((int(m.group(1)), f) for f in fs for m in [re.fullmatch(r"checkpoint_t(\d+)\.pt", f)] if m)
        files += [os.path.join(dp, f) for _, f in ck[:-1]]
    for dp, _, fs in os.walk(os.path.join(ROOT, "v2", "runs")):
        if os.path.basename(dp) == "decay_start":
            files += [os.path.join(dp, f) for f in fs if f.endswith(".pt")]
    ic = os.path.join(ROOT, "v2", "inductor_cache")
    if os.path.isdir(ic):
        dirs.append(ic)
    for dp, _, fs in os.walk(os.path.join(ROOT, "v2", "profile")):
        files += [os.path.join(dp, f) for f in fs if f.endswith(".pt")]
    return files, dirs


def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    files, dirs = plan()
    groups = {}
    for p in files + dirs:
        rel = os.path.relpath(p, ROOT)
        key = "/".join(rel.split("/")[:2])
        groups.setdefault(key, [0, 0])
        groups[key][0] += 1
        groups[key][1] += size(p)
    total = sum(v[1] for v in groups.values())
    print("%s: %d dosya + %d klasör, toplam %.1f GB" % ("SİLİNİYOR" if a.apply else "PLAN (silinmedi)", len(files),
                                                          len(dirs), total / 2 ** 30))
    for k, (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print("  %7.1f GB  %4d  %s" % (b / 2 ** 30, n, k))
    print("\nörnek dosyalar:")
    for p in (files + dirs)[:15]:
        print("  " + os.path.relpath(p, ROOT))
    if a.apply:
        for p in files:
            os.remove(p)
        for d in dirs:
            shutil.rmtree(d)
        print("bitti")


if __name__ == "__main__":
    main()
