#!/bin/bash
# BAKICI — ayri surec. Cekirdek kilitliyken de calisir.
# Her ARA saniyede bir: (1) calismayi Drive'a aynala, (2) raporu yenile, (3) nabiz.
#   bakici.sh <CALIS> <EV> <KOD> [ARA]
CALIS="$1"; EV="$2"; KOD="$3"; ARA="${4:-300}"
mkdir -p "$EV"
while true; do
  # 1) kucuk dosyalar — dizin yapisini koruyarak, sadece yenileri
  ( cd "$CALIS" && find . -type f ! -name '*.pt' -print0 \
      | xargs -0 -r -I@ cp -u --parents @ "$EV"/ ) 2>/dev/null
  # 2) buyuk .pt dosyalari — .tmp + mv (yarim dosya kalmasin, CLAUDE.md 7)
  ( cd "$CALIS" && find . -type f -name '*.pt' ) 2>/dev/null | while read -r r; do
      [ -f "$EV/$r" ] && [ ! "$CALIS/$r" -nt "$EV/$r" ] && continue
      mkdir -p "$EV/$(dirname "$r")"
      cp -f "$CALIS/$r" "$EV/$r.tmp" && mv -f "$EV/$r.tmp" "$EV/$r"
  done
  # 3) surdurme paketini ADIM ADLI arsivle -> her olcumde geri donus noktasi
  python "$KOD/sablon/arsivle.py" "$CALIS" 2>/dev/null
  # 4) rapor — once .tmp, sonra mv: okurken yarim gormeyelim
  python "$KOD/sablon/rapor.py" "$CALIS" > "$CALIS/RAPOR.txt.tmp" 2>&1 \
    && mv -f "$CALIS/RAPOR.txt.tmp" "$CALIS/RAPOR.txt" \
    && cp -f "$CALIS/RAPOR.txt" "$EV/RAPOR.txt"
  date +"%s %Y-%m-%d %H:%M:%S" > "$EV/nabiz.txt"
  sleep "$ARA"
done
