#!/bin/bash
# BAKICI - ayri surec. Cekirdek kilitliyken de calisir.
# Her ARA saniyede bir: (1) calismayi Drive'a aynala, (2) raporu yenile, (3) nabiz.
#   bakici.sh <CALIS> <EV> <KOD> [ARA]
#
# CIKTI BIR LOGA GIDER (defter HUCRE 4 -> log/bakici_<deney>.txt). Eskiden
# DEVNULL'a gidiyordu ve hem aynalama hatasi hem arsivle.py'nin "arsivlendi
# adim N" satiri kayboluyordu: yedekleme surecinin HICBIR adli izi yoktu.
CALIS="$1"; EV="$2"; KOD="$3"; ARA="${4:-300}"
mkdir -p "$EV"
while true; do
  echo "--- tur $(date +'%Y-%m-%d %H:%M:%S')"
  # 1) kucuk dosyalar - dizin yapisini koruyarak, sadece yenileri
  ( cd "$CALIS" && find . -type f ! -name '*.pt' -print0       | xargs -0 -r -I@ cp -u --parents @ "$EV"/ )     || echo "!! kucuk dosya aynalamasi HATA"
  # 2) buyuk .pt dosyalari - .tmp + mv (yarim dosya kalmasin, CLAUDE.md 7)
  ( cd "$CALIS" && find . -type f -name '*.pt' ) | while read -r r; do
      [ -f "$EV/$r" ] && [ ! "$CALIS/$r" -nt "$EV/$r" ] && continue
      mkdir -p "$EV/$(dirname "$r")"
      cp -f "$CALIS/$r" "$EV/$r.tmp" && mv -f "$EV/$r.tmp" "$EV/$r"         || echo "!! .pt aynalanamadi: $r"
  done
  # 3) surdurme paketini ADIM ADLI arsivle -> geri donus noktasi
  python "$KOD/sablon/arsivle.py" "$CALIS" || echo "!! arsivle.py HATA"
  # 4) rapor - once .tmp, sonra mv: okurken yarim gormeyelim
  if python "$KOD/sablon/rapor.py" "$CALIS" > "$CALIS/RAPOR.txt.tmp" 2>&1; then
    mv -f "$CALIS/RAPOR.txt.tmp" "$CALIS/RAPOR.txt"
    cp -f "$CALIS/RAPOR.txt" "$EV/RAPOR.txt" || echo "!! RAPOR Drive'a yazilamadi"
  else
    echo "!! rapor.py COKTU:"; tail -5 "$CALIS/RAPOR.txt.tmp"
  fi
  # NABIZ Drive'a yazilir -> Drive erisilemezse nabiz da BAYATLAR. Kasitli:
  # nabzin taze olmasi, Drive'in yazilabilir oldugunun KANITI olsun.
  date +"%s %Y-%m-%d %H:%M:%S" > "$EV/nabiz.txt" || echo "!! nabiz yazilamadi"
  sleep "$ARA"
done
