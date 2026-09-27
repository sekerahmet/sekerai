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
    d = D.build(step_answers=True)             # bugunku egitim verisi: 1R + ara adimli 2R
    D.audit(d)
    people, rel = d["people"], d["rel"]
    held = set(d["held"])

    def shown(x):
        p = people[x]
        if p["generation"] == 2 and p["gender"] == "f":
            return "%s <span class=nee>(née %s)</span>" % (e(x), e(p["birth_family"]))
        return e(x)

    n = len(D.FAMILIES)
    cx = cy = 450

    def at(r, k):
        a = math.radians(90 - k * 360 / n)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    def label(r, k, text, cls, outward):
        """Yaricap boyunca yazi: 32 aile yan yana yatay yazilamiyor; sol yarida ters donmesin diye 180 cevrilir."""
        a = 90 - k * 360 / n
        x, y = at(r, k)
        right = math.cos(math.radians(a)) > -1e-9
        rot = -a if right else 180 - a
        side = ("start" if outward else "end") if right else ("end" if outward else "start")
        return ('<text x="%.1f" y="%.1f" class="%s" text-anchor="%s" dominant-baseline="middle" transform="rotate(%.1f %.1f %.1f)">%s</text>'
                % (x, y, cls, side, rot, x, y, text))

    parts, fam_pos, hh_pos = [], {}, {}
    for i, f in enumerate(D.FAMILIES):
        fam_pos[f] = at(355, i)
        hh_pos[f] = at(250, i + 0.5)
    for i, f in enumerate(D.FAMILIES):
        nxt = D.FAMILIES[(i + 1) % n]
        hx, hy = hh_pos[f]
        for src, cls in ((f, "son"), (nxt, "dau")):
            x, y = fam_pos[src]
            parts.append('<line class="ln %s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (cls, x, y, hx, hy))
    for i, f in enumerate(D.FAMILIES):
        x, y = fam_pos[f]
        gf, gm = D.NAMES[f][:2]
        parts.append('<g class="fam"><title>%s: %s &amp; %s</title><circle cx="%.1f" cy="%.1f" r="5"/>%s</g>'
                     % (f, gf, gm, x, y, label(368, i, f, "t1", outward=True)))
    for i, f in enumerate(D.FAMILIES):
        x, y = hh_pos[f]
        kids = [k for k, p in people.items() if p["household"] == f]
        fa, mo = rel[kids[0]]["father"][0].split()[0], rel[kids[0]]["mother"][0].split()[0]
        cls = "hh held" if f in D.HELD_HOUSEHOLDS else "hh teach"
        parts.append('<g class="%s"><title>%s hanesi: %s + %s, torunlar %s</title><rect x="%.1f" y="%.1f" width="12" height="12" rx="2"/>%s</g>'
                     % (cls, f, fa, mo, " ve ".join(kids), x - 6, y - 6,
                        label(236, i + 0.5, "%s, %s" % (kids[0].split()[0], kids[1].split()[0]), "t2", outward=False)))
    svg = ('<svg viewBox="0 0 900 900" role="img" aria-label="32 aile halka biçiminde evleniyor: her ailenin oğlu bir '
           'sonrakinin kızıyla evli, torunlar bu hanelerde. Yeşil haneler öğretme, mavi haneler tutulan.">%s</svg>' % "".join(parts))

    rows = []
    for f in D.FAMILIES:
        kids = [k for k, p in people.items() if p["household"] == f]
        k0 = kids[0]

        def names(key):
            return ", ".join(e(y) for y in sorted(set().union(*(D.follow(rel, k0, p) for p in D.DERIVED[key]))))
        group = '<span class="pill held">tutulan</span>' if f in D.HELD_HOUSEHOLDS else '<span class="pill teach">öğretme</span>'
        rows.append("<tr><td><b>%s</b><br>%s</td><td>%s<br>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            e(f), group, shown(rel[k0]["father"][0]), shown(rel[k0]["mother"][0]), ", ".join(e(k) for k in kids),
            names("aunt"), names("uncle"), names("grandfather") + "<br>" + names("grandmother")))

    by_subject = {}
    for s_ in d["train"]:
        start = s_.index("Who") + 2 if "Who" in s_[:2] else 0          # "<steps> Who is X ..." / "Who is X ..." / "X 's ..."
        by_subject.setdefault(" ".join(s_[start:start + 2]), []).append(D.detokenize(s_))
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
                      % (e(x.lower()), " open" if x == "Tom Smith" else "", gen_label[p["generation"]], shown(x), tag,
                         len(by_subject[x]), lines))

    CLS = [("2R_UT", "2R_UT · hiç görülmemiş 2R"), ("2R_T", "2R_T · eğitimdeki 2R")]      # 1R_T (640) kisi listesinde
    counts = {c: sum(1 for q in d["exam"] if q["cls"] == c) for c, _ in CLS}
    chips = "".join('<button type="button" class="chip%s" data-cls="%s" aria-pressed="%s">%s <b>%d</b></button>'
                    % (" on" if c == "2R_UT" else "", c, "true" if c == "2R_UT" else "false", name, counts[c]) for c, name in CLS)
    qrows = []
    for c, _ in CLS:
        for q in d["exam"]:
            if q["cls"] != c:
                continue
            want = D.detokenize(q["steps"]) if "steps" in q else " · ".join(q["answers"])
            qrows.append('<tr class="q" data-cls="%s" data-name="%s"%s><td class="cls">%s</td><td class="mono">%s</td><td>%s</td></tr>' % (
                c, e((" ".join([q["subject"]] + q["answers"])).lower()), "" if c == "2R_UT" else " hidden",
                c, e(D.detokenize(q["prompt"])), e(want)))

    return """  <div class="world">
    <div class="ring">
      <figure>
        <div class="draw">%s</div>
        <figcaption>Dış halkada 32 aile (I. nesil; üstüne gelince dede ve nine). İçte haneler: bir ailenin oğlu (düz çizgi) sonraki ailenin kızıyla (kesikli) evli; torunlar (III. nesil) bu hanelerde, adları hanenin yanında. Yeşil haneler öğretme, mavi haneler tutulan grup.</figcaption>
      </figure>
      <div class="part">
        <h3>Dünyanın kuralları</h3>
        <ul class="rules">
          <li>Her çiftin bir oğlu, bir kızı var. Evlenen kadın kocasının soyadını alır; bir kişinin tek bir tam adı var ve her ad benzersiz.</li>
          <li>Bu yüzden her torunda hala (<code>father's sister</code>) ve dayı (<code>mother's brother</code>) tek kişi; iki dede, iki nine var.</li>
          <li>Temel ilişkiler herkes için yazılır: father, mother, brother, sister, son, daughter. Her olgu bir cümle ve bir soru–cevap.</li>
          <li>2R yalnız ara adımlarıyla yazılır: <code>&lt;steps&gt; Who is X 's r1 's r2 ? X 's r1 is B . B 's r2 is Y .</code> Tutulan torunlar dışındaki 160 kişi için, kişi başına 6.</li>
          <li><span class="pill held">tutulan</span> hanelerin torunları (32 kişi) hiçbir 2R cümlesinde geçmez; zinciri onlardan geçen 2R de yazılmaz. Onların 2R'si sorulur.</li>
          <li>Kısa cevaplı 2R, adlı ilişkiler (aunt, uncle, grandfather, grandmother, cousin) ve 3R yazılmaz.</li>
        </ul>
        <p class="fp">veri izi %s · data_20.py'den üretildi (STEP_ANSWERS açık)</p>
      </div>
    </div>

    <div class="part">
      <h3>Haneler ve torunların akrabaları</h3>
      <div class="scroll"><table class="houses">
        <thead><tr><th>hane</th><th>anne–baba</th><th>torunlar</th><th>hala</th><th>dayı</th><th>dedeler / nineler</th></tr></thead>
        <tbody>%s</tbody>
      </table></div>
    </div>

    <div class="part">
      <h3>Sorular</h3>
      <p>Varsayılan görünüm, tutulan torunlara sorulan ve hiç görülmemiş 2R soruları; doğru cevap ara adımlarıyla. Eğitimdeki 2R düğmeyle açılır; 1R_T soruları (640) aşağıda, kişinin eğitim cümlelerinde.</p>
      <div class="tools">
        <input id="search" type="search" placeholder="ad ara, örn. owen" aria-label="Ada göre süz">
        %s
      </div>
      <div class="scroll"><table>
        <thead><tr><th>sınıf</th><th>soru</th><th>doğru cevap</th></tr></thead>
        <tbody id="qbody">%s</tbody>
      </table></div>
    </div>

    <div class="part">
      <h3>Eğitim cümleleri, kişiye göre</h3>
      <p>Her 1R olgu bir cümle ve bir soru–cevap; 2R ara adımlı cevaplar sorudaki kişinin altında. Nesil I–III. Arama kutusu bu listeyi de süzer.</p>
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
.world .ring { display: grid; gap: 22px; }
.world .ring figure { max-width: 780px; }
.world svg .ln { stroke: var(--line); stroke-width: 1.4; fill: none; }
.world svg .ln.dau { stroke-dasharray: 5 4; }
.world svg rect { stroke-width: 1.2; }
.world svg .fam circle { fill: var(--surface); stroke: var(--ink); stroke-width: 1.4; }
.world svg .hh.teach rect { fill: var(--teach-tint); stroke: var(--teach); }
.world svg .hh.held rect { fill: var(--held-tint); stroke: var(--held); }
.world svg text { fill: var(--ink); }
.world svg .t1 { font-size: 17px; font-weight: 600; }
.world svg .t2 { font-size: 14px; fill: var(--muted); }
.world svg .hh.held .t2 { fill: var(--held); }
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
