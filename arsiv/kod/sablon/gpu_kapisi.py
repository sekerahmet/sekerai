# -*- coding: utf-8 -*-
"""GPU KAPISI — egitim koserken ikinci bir GPU isi BASLATMA.

14 Eylul, deney G: Colab runtime'i UC kez geri donusturuldu. Ucunde de o
sirada egitimin YANINDA ikinci bir GPU isi kosuyordu (arama tani turu).
Kanitlanmis bir nedensellik degil, ama ucu de ayni desende ve her geri
donusum ~10-15 dk kayip demek.

Asil kusur kuralin YERINDE OLMAMASIYDI: "egitim koserken GPU'ya dokunma"
diye konustuk, sonra ben yine dokundum. Kural KODA girmezse tutulmuyor --
bu oturumda ayni sey `belge/` ve IMZA capalari icin de oldu.

    from gpu_kapisi import gpu_bos_mu
    gpu_bos_mu()                      # egitim kosuyorsa ASSERT ATESLER
    gpu_bos_mu(zorla=True)            # bilerek yan yana kosacaksan

Ortam degiskeniyle de kapatilabilir:  GPU_KAPISI=0
"""
import os
import subprocess

# Egitim surecinin izi. `ps | grep <betik>` KENDI komut satirini yakalar ve
# yanlis "kosuyor" der (CLAUDE.md 7) -> ilk harf koseli parantezde.
_DESEN = "[s]ifirdan.py"


def kosan_egitim():
    """Kosan egitim sureclerinin (pid, gecen sure, komut) listesi."""
    r = subprocess.run(["bash", "-lc",
                        f"ps -eo pid,etime,cmd | grep -E '{_DESEN}' || true"],
                       capture_output=True, text=True)
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def gpu_bos_mu(zorla=False, ad="bu is"):
    """Egitim kosuyorsa DURDUR. Kosmuyorsa sessizce gec."""
    if zorla or os.environ.get("GPU_KAPISI") == "0":
        return True
    k = kosan_egitim()
    assert not k, (
        f"GPU KAPISI: {len(k)} egitim sureci KOSUYOR, {ad} baslatilmiyor." +
        chr(10) + "  " + (chr(10) + "  ").join(k) + chr(10) +
        "  Egitim koserken ikinci GPU isi runtime'i zorluyor (14 Eylul: uc"
        " geri donusum)." + chr(10) +
        "  Bitmesini bekle, ya da bilerek yan yana kosacaksan:" + chr(10) +
        "     gpu_bos_mu(zorla=True)   /   GPU_KAPISI=0 ortam degiskeni")
    return True


if __name__ == "__main__":
    k = kosan_egitim()
    print(f"kosan egitim sureci: {len(k)}")
    for x in k:
        print("  ", x)
    print("GPU", "MESGUL -- tani/tarama baslatma" if k else "BOS -- serbest")
