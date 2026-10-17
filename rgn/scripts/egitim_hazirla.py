"""Kendi eğitim metinlerini ön eğitim belgelerine çevirir (JSONL, satır başına {"text": ..., "kaynak": ...}).

Girdi klasörü (ör. EGITIM; repoya girmez):
    *.txt                    elle yazılmış yazılar; belgeler arası kendi satırında "=====" (yoksa dosya tek belge)
    meb_kitaplar/*_full.txt  PDF'den metne çevrilmiş kitaplar ("===== PDF SAYFA n =====" işaretli): ilk sayfalar ve künye /
                             marş / içindekiler / kaynakça sayfaları (BOILER) atılır, sayfa numarası / tekrarlanan üst bilgi satırları atılır, satır sonuyla bölünmüş
                             cümleler birleştirilir; kitap başına bir belge
    site_html/**/*.html      site sayfaları: script / stil / gezinme atılır, görünen metin (tablo hücreleri " | " ile)
Çıktı: <girdi>/cikti/{yazilar,kitaplar,site}.jsonl  +  her birinden örnek (ilk 1.500 karakter) ekrana

    python scripts/egitim_hazirla.py "C:/AI new 2/rgn/EGITIM"
"""
import collections
import glob
import json
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

PAGE = re.compile(r"^===== PDF SAYFA (\d+) =====$", re.M)
BOILER = ("ISBN", "Atatürk Bulvarı", "Korkma, sönmez", "GENÇLİĞE HİTABE", "Ey Türk gençliği", "Her hakkı saklıdır",
          "yayın hakları", "İÇİNDEKİLER", "İçindekiler", "Talim ve Terbiye Kurulu", "KAYNAKÇA", "Kaynakça")   # künye vb. sayfa
SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "button", "template"}
BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "section", "article", "table", "ul", "ol",
         "blockquote", "pre", "figcaption", "summary", "details", "dt", "dd"}


def norm(t):
    t = unicodedata.normalize("NFC", t.replace("\r\n", "\n").replace("\u00a0", " "))
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def yazilar(root):
    for f in sorted(root.glob("*.txt")):
        t = f.read_text(encoding="utf-8-sig")
        for i, p in enumerate(re.split(r"\n\s*=====\s*\n", t.replace("\r\n", "\n"))):
            p = norm(p)
            if len(p) >= 20:
                yield dict(text=p, kaynak="%s#%d" % (f.name, i + 1))


def kitap(text, skip_pages=3):
    pages = PAGE.split(text)                                              # ['', '0', sayfa0, '1', sayfa1, ...]
    body = [pages[i + 1] for i in range(1, len(pages) - 1, 2)
            if int(pages[i]) >= skip_pages and not any(b in pages[i + 1] for b in BOILER)]
    lines = [l.strip() for pg in body for l in pg.strip().split("\n")]
    freq = collections.Counter(l for l in lines if l)
    head = {l for l, k in freq.items() if k >= max(5, len(body) // 4) and len(l) < 80}   # her sayfada tekrarlanan üst bilgi
    out = []
    for pg in body:
        ls = [l.strip() for l in pg.strip().split("\n")]
        ls = [l for l in ls if l and l not in head and not re.fullmatch(r"\d{1,4}", l)
              and not re.fullmatch(r"(\w+)( )+", l)                       # "AYT AYT AYT …"
              and not re.fullmatch(r"[A-ZÇĞİÖŞÜ .\-]+ - (AYT|TYT) ?\d+|(AYT|TYT) ?\d+ ?- [A-ZÇĞİÖŞÜ .\-]+", l)]  # sayfa başlığı
        out.append("\n".join(ls))
    t = "\n".join(out)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                               # satır sonunda bölünmüş kelime
    t = re.sub(r"(?<=[a-zçğıöşü,;])\n(?=[a-zçğıöşü])", " ", t)         # cümle içi satır sonu
    t = re.sub(r"\.{5,}\s*\d*", " ", t)                                   # içindekiler noktaları
    return norm(t)


def kitaplar(root, skip_pages=3):
    for f in sorted((root / "meb_kitaplar").glob("*_full.txt")):               # _uniteler / _index vb. türetilmiş
        t = kitap(f.read_text(encoding="utf-8", errors="replace"), skip_pages)
        if len(t) >= 200:
            yield dict(text=t, kaynak=f.name)


class _Html(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title = [], 0, ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, d):
        if self._in_title:
            self.title += d
        elif not self.skip:
            self.out.append(d)


def site(root):
    for f in sorted(glob.glob(str(root / "site_html" / "**" / "*.html"), recursive=True)):
        p = _Html()
        p.feed(Path(f).read_text(encoding="utf-8", errors="replace"))
        t = norm("".join(p.out))
        t = re.sub(r"(?m)^\| ", "", t)
        if len(t) >= 200:
            yield dict(text=t, kaynak=str(Path(f).relative_to(root / "site_html")).replace("\\", "/"))


def main():
    root = Path(sys.argv[1])
    out = root / "cikti"
    out.mkdir(exist_ok=True)
    for name, gen in (("yazilar", yazilar(root)), ("kitaplar", kitaplar(root)), ("site", site(root))):
        docs = list(gen)
        with open(out / ("%s.jsonl" % name), "w", encoding="utf-8") as fo:
            for d in docs:
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
        n = sum(len(d["text"].encode("utf-8")) for d in docs)
        print("\n##### %s: %d belge, %.1f MB  -> %s" % (name, len(docs), n / 1e6, out / ("%s.jsonl" % name)))
        if docs:
            d = docs[len(docs) // 2]
            print("örnek (%s):\n%s\n…" % (d["kaynak"], d["text"][:1500]))


if __name__ == "__main__":
    main()
