# -*- coding: utf-8 -*-
"""model_y kodunu Colab icin Drive'a kopyalar; push gerekmez (kullanici, 28 Eylul: "Kod Drive'dan gelsin, push gerekmesin").

Kopyalanan: model_y altindaki butun .py dosyalari (alt klasorler dahil).  En son iz listesi (fingerprints.json) yazilir:
dosya basina sha256, birlesik iz, git HEAD ve commit edilmemis degisiklik bayragi.  Colab HAZIRLIK kopyayi verify() ile
izlerle karsilastirir; esitleme yarim kaldiysa durur.
Kullanim (yerelde):  python sync_code.py"""
import hashlib
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join("G:" + os.sep, "Drive'ım", "model_y", "code")     # Colab'da /content/drive/MyDrive/model_y/code
MANIFEST = "fingerprints.json"


def code_files(root):
    """root altindaki .py dosyalari: goreli yol ('/' ile), sirali."""
    out = []
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        out += [os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/") for n in names if n.endswith(".py")]
    return sorted(out)


def file_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def combined(hashes):
    """Birlesik iz: yol ve hash ciftlerinden, 12 hane."""
    return hashlib.sha256("\n".join("%s %s" % kv for kv in sorted(hashes.items())).encode()).hexdigest()[:12]


def sync(dest=DEST, root=HERE):
    """Kodu dest'e kopyalar, dest'te kaynakta olmayan .py dosyalarini siler, iz listesini EN SON yazar."""
    files = code_files(root)
    hashes = {f: file_hash(os.path.join(root, f)) for f in files}
    os.makedirs(dest, exist_ok=True)
    if os.path.exists(os.path.join(dest, MANIFEST)):
        os.remove(os.path.join(dest, MANIFEST))          # yarim kopya eski listeyle dogrulanmasin
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(dest, f)) or dest, exist_ok=True)
        shutil.copyfile(os.path.join(root, f), os.path.join(dest, f))
    stale = [f for f in code_files(dest) if f not in hashes]
    for f in stale:
        os.remove(os.path.join(dest, f))
    git = lambda *a: subprocess.run(["git", "-C", root] + list(a), capture_output=True, text=True).stdout.strip()
    manifest = dict(fingerprint=combined(hashes), files=hashes, git_head=git("log", "--oneline", "-1"),
                    uncommitted=bool(git("status", "--porcelain", "--", ".")), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(dest, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest, stale


def verify(root):
    """Colab'da: root'taki kopyayi iz listesiyle karsilastirir; tutmazsa AssertionError.  Iz listesini dondurur."""
    path = os.path.join(root, MANIFEST)
    assert os.path.exists(path), "iz listesi yok (%s) -- yerelde python sync_code.py, Drive esitlemesini bekle" % path
    manifest = json.load(open(path, encoding="utf-8"))
    found = {f: file_hash(os.path.join(root, f)) for f in code_files(root)}
    missing = sorted(set(manifest["files"]) - set(found))
    wrong = sorted(f for f in manifest["files"] if f in found and found[f] != manifest["files"][f])
    extra = sorted(set(found) - set(manifest["files"]))
    assert not (missing or wrong or extra), "kod izi tutmuyor -- Drive esitlemesi yarim olabilir: eksik %s, farkli %s, fazla %s" % (
        missing, wrong, extra)
    assert combined(found) == manifest["fingerprint"]
    return manifest


if __name__ == "__main__":
    m, stale = sync()
    print("kod izi %s  %d dosya -> %s" % (m["fingerprint"], len(m["files"]), DEST))
    print("git %s%s" % (m["git_head"], "  + COMMIT EDILMEMIS DEGISIKLIK" if m["uncommitted"] else ""))
    if stale:
        print("Drive'dan silinen eski dosyalar: %s" % stale)
