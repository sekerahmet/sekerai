# -*- coding: utf-8 -*-
"""build_notebook_20 -- model_20'nin adim defteri: tek sayfa, sekmeli (Tum resim, Adim 0..3).
Elle yazilan kisim notebook_master_20.html; Adim 0'daki dunya icerigi data_20'den uretilir.

    python build_notebook_20.py      ->  notebook_20.html
"""
import html
import math
import os

import data_20 as D

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "notebook_master_20.html")
OUT = os.path.join(HERE, "notebook_20.html")
e = html.escape


def world():
    d = D.build()
    D.audit(d)
    people, rel = d["people"], d["rel"]
    held = set(d["held"])

    def shown(x):
        p = people[x]
        if p["generation"] == 2 and p["gender"] == "f":
            return "%s <span class=nee>(née %s)</span>" % (e(x), e(p["birth_family"]))
        return e(x)

    n = len(D.FAMILIES)
    cx = cy = 380
    parts, fam_pos, hh_pos = [], {}, {}
    for i, f in enumerate(D.FAMILIES):
        a = math.radians(90 - i * 360 / n)
        fam_pos[f] = (cx + 300 * math.cos(a), cy - 300 * math.sin(a))
        b = math.radians(90 - (i + 0.5) * 360 / n)
        hh_pos[f] = (cx + 172 * math.cos(b), cy - 172 * math.sin(b))
    for i, f in enumerate(D.FAMILIES):
        nxt = D.FAMILIES[(i + 1) % n]
        hx, hy = hh_pos[f]
        for src, cls in ((f, "son"), (nxt, "dau")):
            x, y = fam_pos[src]
            parts.append('<line class="ln %s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (cls, x, y, hx, hy))
    for f in D.FAMILIES:
        x, y = fam_pos[f]
        gf, gm = D.NAMES[f][:2]
        parts.append('<g class="fam"><rect x="%.1f" y="%.1f" width="124" height="44" rx="3"/>'
                     '<text x="%.1f" y="%.1f" class="t1">%s</text><text x="%.1f" y="%.1f" class="t2">%s &amp; %s</text></g>'
                     % (x - 62, y - 22, x, y - 4, f, x, y + 13, gf, gm))
    for f in D.FAMILIES:
        x, y = hh_pos[f]
        kids = [k for k, p in people.items() if p["household"] == f]
        fa, mo = rel[kids[0]]["father"][0].split()[0], rel[kids[0]]["mother"][0].split()[0]
        cls = "hh held" if f in D.HELD_HOUSEHOLDS else "hh teach"
        parts.append('<g class="%s"><rect x="%.1f" y="%.1f" width="124" height="48" rx="3"/>'
                     '<text x="%.1f" y="%.1f" class="t1">%s + %s</text><text x="%.1f" y="%.1f" class="t2">%s, %s</text></g>'
                     % (cls, x - 62, y - 24, x, y - 5, fa, mo, x, y + 13, kids[0].split()[0], kids[1].split()[0]))
    svg = ('<svg viewBox="0 0 760 760" role="img" aria-label="Sekiz aile halka biçiminde evleniyor: her ailenin oğlu bir '
           'sonrakinin kızıyla evli, torunlar bu hanelerde.">%s</svg>' % "".join(parts))

    rows = []
    for f in D.FAMILIES:
        kids = [k for k, p in people.items() if p["household"] == f]
        k0 = kids[0]

        def names(key):
            if isinstance(key, str):
                ys = sorted(set().union(*(D.follow(rel, k0, p) for p in D.DERIVED[key])))
            else:
                ys = D.follow(rel, k0, key)
            return ", ".join(e(y) for y in ys)
        group = '<span class="pill held">tutulan</span>' if f in D.HELD_HOUSEHOLDS else '<span class="pill teach">öğretme</span>'
        rows.append("<tr><td><b>%s</b><br>%s</td><td>%s<br>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            e(f), group, shown(rel[k0]["father"][0]), shown(rel[k0]["mother"][0]), ", ".join(e(k) for k in kids),
            names("aunt"), names("uncle"), names("grandfather") + "<br>" + names("grandmother"), names("cousin")))

    by_subject = {}
    for s_ in d["train"]:
        subj = " ".join(s_[2:4]) if s_[0] == "Who" else " ".join(s_[:2])
        by_subject.setdefault(subj, []).append(D.detokenize(s_))
    order = sorted(people, key=lambda x: (people[x]["generation"],
                                          D.FAMILIES.index(people[x]["household"] or people[x]["birth_family"]),
                                          people[x]["gender"] != "m"))
    gen_label = {1: "I", 2: "II", 3: "III"}
    blocks = []
    for x in order:
        if x not in by_subject:
            continue
        p = people[x]
        tag = ""
        if p["generation"] == 3:
            tag = '<span class="pill held">tutulan</span>' if x in held else '<span class="pill teach">öğretme</span>'
        lines = "".join("<li>%s</li>" % e(t) for t in by_subject[x])
        blocks.append('<details class="person" data-name="%s"%s><summary><span class="gen">%s</span> %s %s'
                      '<span class="count">%d cümle</span></summary><ol>%s</ol></details>'
                      % (e(x.lower()), " open" if x == "Alice Smith" else "", gen_label[p["generation"]], shown(x), tag,
                         len(by_subject[x]), lines))

    CLS = [("chain2", "çıkarım · 2 adım, zincir"), ("named2", "çıkarım · 2 adım, adıyla"), ("chain3", "çıkarım · 3 adım, zincir"),
           ("named3", "çıkarım · 3 adım, adıyla"), ("memory_derived", "hafıza · yazılı türemiş"), ("memory_base", "hafıza · temel olgu")]
    counts = {c: sum(1 for q in d["exam"] if q["cls"] == c) for c, _ in CLS}
    chips = "".join('<button type="button" class="chip%s" data-cls="%s" aria-pressed="%s">%s <b>%d</b></button>'
                    % (" on" if not c.startswith("memory") else "", c, "true" if not c.startswith("memory") else "false",
                       label, counts[c]) for c, label in CLS)
    qrows = []
    for c, _ in CLS:
        for q in d["exam"]:
            if q["cls"] != c:
                continue
            qrows.append('<tr class="q" data-cls="%s" data-name="%s"%s><td class="cls">%s</td><td class="mono">%s</td>'
                         '<td>%s</td><td>%s</td></tr>' % (
                             c, e((" ".join([q["subject"]] + q["answers"])).lower()), " hidden" if c.startswith("memory") else "",
                             c, e(D.detokenize(q["prompt"])), " · ".join(e(a) for a in q["answers"]),
                             '<span class="pill warn">aynı cümlede</span>' if q["co_written"] and not c.startswith("memory") else ""))

    return """  <div class="world">
    <div class="ring">
      <figure>
        <div class="draw">%s</div>
        <figcaption>Dış halkada aileler ve dedeleri–nineleri (I. nesil). İçte haneler: bir ailenin oğlu (düz çizgi) sonraki ailenin kızıyla (kesikli) evli, torunlar (III. nesil) bu hanelerde. Yeşil haneler öğretme, mavi haneler tutulan grup.</figcaption>
      </figure>
      <div class="part">
        <h3>Dünyanın kuralları</h3>
        <ul class="rules">
          <li>Her çiftin bir oğlu, bir kızı var. Evlenen kadın kocasının soyadını alır; bir kişinin tek bir tam adı var ve her ad benzersiz.</li>
          <li>Bu yüzden her torunda <code>aunt</code> tek kişi ve hep babanın kız kardeşi (hala); <code>uncle</code> tek kişi ve hep annenin erkek kardeşi (dayı).</li>
          <li>Her torunun 2 <code>grandfather</code>'ı, 2 <code>grandmother</code>'ı, 4 <code>cousin</code>'i var. Kuzenler iki yandaki hanelerin çocukları.</li>
          <li>Temel ilişkiler herkes için yazılır: father, mother, brother, sister, son, daughter.</li>
          <li><span class="pill teach">öğretme</span> hanelerinin torunlarında türemiş ilişki iki biçimde yazılır: zincir (<code>father's sister</code>) ve adıyla (<code>aunt</code>).</li>
          <li><span class="pill held">tutulan</span> hanelerin torunlarında türemiş ilişki hiçbir biçimde yazılmaz; sorulur.</li>
          <li><span class="pill warn">aynı cümlede</span>: cevap ile özne bir eğitim cümlesinde birlikte geçmiş, yalnız kenar hanelerin kuzen sorularında. Bu sorular ayrı okunur.</li>
        </ul>
        <p class="fp">veri izi %s · data_20.py'den üretildi</p>
      </div>
    </div>

    <div class="part">
      <h3>Haneler ve torunların akrabaları</h3>
      <div class="scroll"><table class="houses">
        <thead><tr><th>hane</th><th>anne–baba</th><th>torunlar</th><th>aunt</th><th>uncle</th><th>grandfather / grandmother</th><th>cousin</th></tr></thead>
        <tbody>%s</tbody>
      </table></div>
    </div>

    <div class="part">
      <h3>Sorular</h3>
      <p>Varsayılan görünüm, tutulan torunlara sorulan ve cevabı hiçbir biçimde yazılmamış sorular. Hafıza sınıfları düğmeyle açılır. Adlı sorularda birden çok doğru cevap var; herhangi biri doğru sayılır.</p>
      <div class="tools">
        <input id="search" type="search" placeholder="ad ara, örn. owen" aria-label="Ada göre süz">
        %s
      </div>
      <div class="scroll"><table>
        <thead><tr><th>sınıf</th><th>soru</th><th>doğru cevap(lar)</th><th></th></tr></thead>
        <tbody id="qbody">%s</tbody>
      </table></div>
    </div>

    <div class="part">
      <h3>Eğitim cümleleri, kişiye göre</h3>
      <p>Her olgu bir cümle ve bir soru–cevap olarak yazılı. Nesil I–III. Arama kutusu bu listeyi de süzer.</p>
      <div class="people" id="people">%s</div>
    </div>
  </div>
""" % (svg, d["fingerprint"], "\n".join(rows), chips, "\n".join(qrows), "\n".join(blocks))


EXTRA_CSS = """
:root { --teach: var(--pl); --teach-tint: var(--pl-tint); --held: var(--w); --held-tint: var(--w-tint);
  --warn: var(--anchor); --warn-tint: var(--anchor-tint); --line: #9AA7A1; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --line: #56635D; } }
:root[data-theme="dark"] { --line: #56635D; }
.stepnav { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); padding-block: 10px; border-bottom: 1px solid var(--rule); }
.steps .all b { color: var(--w); }
.steps a[aria-current="page"] { background: var(--ink); border-color: var(--ink); color: var(--bg); }
.steps a[aria-current="page"] b { color: var(--bg); }
.world { display: grid; gap: 22px; }
.world .ring { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 28px; align-items: start; }
@media (max-width: 820px) { .world .ring { grid-template-columns: 1fr; } }
.world svg .ln { stroke: var(--line); stroke-width: 1.4; fill: none; }
.world svg .ln.dau { stroke-dasharray: 5 4; }
.world svg rect { stroke-width: 1.2; }
.world svg .fam rect { fill: var(--surface); stroke: var(--rule); }
.world svg .hh.teach rect { fill: var(--teach-tint); stroke: var(--teach); }
.world svg .hh.held rect { fill: var(--held-tint); stroke: var(--held); }
.world svg text { text-anchor: middle; }
.world svg .t1 { font-size: 13px; font-weight: 600; }
.world svg .t2 { font-size: 11.5px; fill: var(--muted); }
.world .rules { display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }
.world .rules li { padding-left: 14px; border-left: 2px solid var(--rule); font-size: 14.5px; }
.pill.teach { background: var(--teach-tint); color: var(--teach); }
.pill.held { background: var(--held-tint); color: var(--held); }
.pill.warn { background: var(--warn-tint); color: var(--warn); }
.nee { color: var(--muted); font-size: .9em; }
.fp { font: 12px var(--mono); color: var(--muted); }
.world .houses td { min-width: 110px; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.tools input { font: 14px var(--sans); color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 7px 10px; width: 260px; max-width: 100%; }
.tools input:focus-visible, .chip:focus-visible, summary:focus-visible { outline: 2px solid var(--w); outline-offset: 2px; }
.chip { font: 13px var(--sans); color: var(--muted); background: var(--surface); border: 1px solid var(--rule); border-radius: 999px; padding: 5px 12px; cursor: pointer; }
.chip b { font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.chip.on { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.chip.on b { color: var(--bg); }
.people { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap: 8px; align-items: start; }
.person { background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; }
.person summary { cursor: pointer; padding: 9px 12px; display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: baseline; }
.person .gen { font: 500 11px var(--mono); color: var(--muted); width: 22px; }
.person .count { margin-left: auto; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.person ol { margin: 0; padding: 4px 12px 12px 38px; font: 13px/1.6 var(--mono); max-width: none; }
.q .cls { font: 11.5px var(--mono); color: var(--muted); white-space: nowrap; }
"""

SCRIPTS = """<script>
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.steps a'));
  var secs = Array.prototype.slice.call(document.querySelectorAll('section.step'));
  var ids = secs.map(function (s) { return s.id; });
  var nav = document.querySelector('.stepnav');
  function show(id) {
    secs.forEach(function (s) { s.hidden = s.id !== id; });
    links.forEach(function (a) { a.setAttribute('aria-current', a.getAttribute('href') === '#' + id ? 'page' : 'false'); });
  }
  function current() { var h = (location.hash || '').slice(1); return ids.indexOf(h) >= 0 ? h : 'tum'; }
  links.forEach(function (a) {
    a.addEventListener('click', function (ev) {
      ev.preventDefault();
      var id = a.getAttribute('href').slice(1);
      show(id);
      try { history.replaceState(null, '', '#' + id); } catch (err) {}
      if (nav && window.scrollY > nav.offsetTop) { try { window.scrollTo(0, nav.offsetTop); } catch (err) {} }
    });
  });
  window.addEventListener('hashchange', function () { show(current()); });
  show(current());
})();
(function () {
  var search = document.getElementById('search');
  if (!search) return;
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chip'));
  var rows = Array.prototype.slice.call(document.querySelectorAll('#qbody tr'));
  var people = Array.prototype.slice.call(document.querySelectorAll('.person'));
  function apply() {
    var q = search.value.trim().toLowerCase();
    var on = {};
    chips.forEach(function (c) { on[c.dataset.cls] = c.classList.contains('on'); });
    rows.forEach(function (r) { r.hidden = !on[r.dataset.cls] || (q && r.dataset.name.indexOf(q) < 0); });
    people.forEach(function (p) { p.hidden = q && p.dataset.name.indexOf(q) < 0 && p.textContent.toLowerCase().indexOf(q) < 0; });
  }
  chips.forEach(function (c) {
    c.addEventListener('click', function () {
      c.classList.toggle('on');
      c.setAttribute('aria-pressed', c.classList.contains('on') ? 'true' : 'false');
      apply();
    });
  });
  search.addEventListener('input', apply);
})();
</script>"""


if __name__ == "__main__":
    page = open(MASTER, encoding="utf-8").read()
    page = page.replace("{{WORLD}}", world()).replace("{{EXTRA_CSS}}", EXTRA_CSS).replace("{{SCRIPTS}}", SCRIPTS)
    assert "{{" not in page
    open(OUT, "w", encoding="utf-8").write(page)
    print(OUT, len(page))
