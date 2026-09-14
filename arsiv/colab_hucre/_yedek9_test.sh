while true; do
  ( cd /content/out9 && find . -type f ! -name 'surdur_*.pt' -print0 |
      xargs -0 -I@ cp -u --parents @ /content/drive/MyDrive/deney9_D3/ ) 2>/dev/null
  for f in /content/out9/*/surdur_*.pt; do
    [ -e "$f" ] || continue
    rel="${f#/content/out9/}"
    mkdir -p "/content/drive/MyDrive/deney9_D3/$(dirname "$rel")"
    cp -f "$f" "/content/drive/MyDrive/deney9_D3/$rel.tmp" && mv -f "/content/drive/MyDrive/deney9_D3/$rel.tmp" "/content/drive/MyDrive/deney9_D3/$rel"
  done
  cp -f /content/log_*.txt /content/surucu9.txt /content/drive/MyDrive/deney9_D3/ 2>/dev/null
  touch /content/yedek_nabiz9
  sleep 300
done
