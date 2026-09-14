# -*- coding: utf-8 -*-
"""SEMA HESAP — bir sema kurulmadan ONCE phi tavanini ve gomulu kontrolun
buyuklugunu soyler. Isim uretmeye, veri kurmaya gerek yok.

IKI BUYUKLUK, ikisi de SEMANIN ARITMETIGI:

  phi tavani = zincir / olgu
             = kopru tipinin ORTALAMA CIKIS DERECESI
    Yani "alan" (cografya / okul / aile) phi'yi DEGISTIRMEZ. Tip basina
    ILISKI SAYISI degistirir.

  gomulu kontrol (YOK payi) = kisayolun TIP OLARAK imkansiz oldugu zincirler
    zincir (e, r1, r2): kisayol f(e,r2) ancak tip(e) r2'nin kaynaklarindaysa
    VARDIR. Iliskiler cok tipten cikiyorsa kisayol hep var -> kontrol kucuk.

  BU IKISI TERS CALISIR. Tek bir tipe (mesela KISI) cok iliski yiginca phi
  yukselir ama her sey ayni tipten ciktigi icin kisayol hep mumkun olur ve
  YOK sinifi erir. Cozum: HER tipe cok iliski, ama her iliski DAR kaynakli.

    python sema_hesap.py
"""
import sys

# --------------------------------------------------------------- ADAYLAR
# sema: iliski -> {kaynak tipi: hedef tipi}
# boy : tip -> varlik sayisi

COGRAFYA = (dict(
    baskenti={"ULKE": "SEHIR"},
    buyuksehri={"ULKE": "SEHIR", "BOLGE": "SEHIR"},
    komsusu={"ULKE": "ULKE", "SEHIR": "SEHIR"},
    dili={"ULKE": "DIL"},
    ulkesi={"SEHIR": "ULKE"},
    bolgesi={"ULKE": "BOLGE", "SEHIR": "BOLGE"},
    merkezi={"BOLGE": "SEHIR"},
    yanbolgesi={"BOLGE": "BOLGE"},
    kokeni={"DIL": "DIL"},
), dict(ULKE=156, SEHIR=780, BOLGE=312, DIL=94))

# Kullanicinin onerisi: okul / ogretmen / kardes / ogrenci / sehir.
# KISI baskin tip, cok iliskisi var.
OKUL_KISI_AGIRLIKLI = (dict(
    annesi={"KISI": "KISI"},
    babasi={"KISI": "KISI"},
    kardesi={"KISI": "KISI"},
    esi={"KISI": "KISI"},
    ogretmeni={"KISI": "KISI"},
    arkadasi={"KISI": "KISI"},
    okulu={"KISI": "OKUL"},
    sehri={"KISI": "SEHIR", "OKUL": "SEHIR"},
    dersi={"KISI": "DERS"},
    muduru={"OKUL": "KISI"},
    komsusu={"SEHIR": "SEHIR"},
    valisi={"SEHIR": "KISI"},
    hocasi={"DERS": "KISI"},
), dict(KISI=800, OKUL=150, SEHIR=150, DERS=60))

# Ayni alan, ama IKINCIL tipler de zenginlestirilmis ve iliskiler DAR
# kaynakli tutulmus (her iliski 1-2 tipten cikiyor).
OKUL_DENGELI = (dict(
    annesi={"KISI": "KISI"},
    babasi={"KISI": "KISI"},
    kardesi={"KISI": "KISI"},
    esi={"KISI": "KISI"},
    ogretmeni={"KISI": "KISI"},
    arkadasi={"KISI": "KISI"},
    okulu={"KISI": "OKUL"},
    sehri={"KISI": "SEHIR"},
    dersi={"KISI": "DERS"},
    muduru={"OKUL": "KISI"},
    kurucusu={"OKUL": "KISI"},
    semti={"OKUL": "SEHIR"},
    rakibi={"OKUL": "OKUL"},
    anaders={"OKUL": "DERS"},
    valisi={"SEHIR": "KISI"},
    komsusu={"SEHIR": "SEHIR"},
    enbuyukokulu={"SEHIR": "OKUL"},
    zorunluders={"SEHIR": "DERS"},
    hocasi={"DERS": "KISI"},
    kitabi={"DERS": "DERS"},
    merkezokulu={"DERS": "OKUL"},
    dogdugusehir={"DERS": "SEHIR"},
), dict(KISI=700, OKUL=200, SEHIR=200, DERS=120))


def hesap(sema, boy):
    tipler = list(boy)
    cikis = {t: [r for r, m in sema.items() if t in m] for t in tipler}
    olgu = {t: len(cikis[t]) * boy[t] for t in tipler}
    N = sum(olgu.values())

    say = {"KISAYOL_MUMKUN": 0, "YOK": 0}
    kopru_tip = {}
    for r1, m1 in sema.items():
        for kt, ht in m1.items():                 # kaynak tipi kt, kopru tipi ht
            n = boy[kt]                            # bu (tip, iliski) kac olgu
            for r2 in cikis[ht]:
                if r2 == r1:
                    continue                       # kisayol = kopru olur
                anah = "KISAYOL_MUMKUN" if kt in sema[r2] else "YOK"
                say[anah] += n
                kopru_tip[ht] = kopru_tip.get(ht, 0) + n
    Z = sum(say.values())
    return N, Z, say, olgu, cikis, kopru_tip


def yaz(ad, sema, boy):
    N, Z, say, olgu, cikis, kt = hesap(sema, boy)
    print("=" * 74)
    print(f"{ad}")
    print("=" * 74)
    print("  tip      varlik  cikis derecesi  olgu")
    for t in boy:
        print(f"   {t:8s} {boy[t]:6d} {len(cikis[t]):12d} {olgu[t]:9d}"
              f"   ({', '.join(cikis[t])})")
    print(f"  ATOMIK OLGU {N:8d}")
    print(f"  ZINCIR      {Z:8d}")
    print(f"  phi TAVANI  {Z/N:8.2f}   <- hicbir bolme ayrilmadan")
    print(f"  bolmeler ayrildiktan sonra (ENT %20 + COMP %10) ~"
          f"{0.72*Z/N:.2f}")
    print(f"  kisayol TIP OLARAK mumkun {say['KISAYOL_MUMKUN']:7d}"
          f"  ({100*say['KISAYOL_MUMKUN']/Z:2.0f}%)")
    print(f"  kisayol IMKANSIZ (YOK)    {say['YOK']:7d}"
          f"  ({100*say['YOK']/Z:2.0f}%)   <- GOMULU KONTROL")
    print()


if __name__ == "__main__":
    yaz("A) COGRAFYA — bugunku sema (veri_gercek.py)", *COGRAFYA)
    yaz("B) OKUL/AILE — KISI baskin, ikincil tipler zayif", *OKUL_KISI_AGIRLIKLI)
    yaz("C) OKUL/AILE — her tip zengin, iliskiler DAR kaynakli", *OKUL_DENGELI)
    print("NOT: YOK payi YAPISALDIR (tiplerden gelir). AYNI ve DONUS siniflari"
          " degerlere bagli,")
    print("     burada hesaplanmaz -- 'kisayol mumkun' onlari da iceriyor.")
