# -*- coding: utf-8 -*-
"""Model X (C' + RoPE) mimari sayfasi: tek SVG diyagram (buyuk resim).  Cikti: model_x_20.html"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_x_20.html")
W, H = 1240, 1330
els = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="t", anchor="middle", extra=""):
    els.append('<text x="%g" y="%g" class="%s" text-anchor="%s"%s>%s</text>' % (x, y, cls, anchor, extra, esc(s)))


def rect(x, y, w, h, cls, rx=6):
    els.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" class="%s"/>' % (x, y, w, h, rx, cls))


def box(x, y, w, h, cls, title, sub=None, tcls="bt", scls="bs"):
    rect(x, y, w, h, cls)
    cy = y + h / 2
    if sub:
        text(x + w / 2, cy - 3, title, tcls)
        text(x + w / 2, cy + 13, sub, scls)
    else:
        text(x + w / 2, cy + 4.5, title, tcls)


def line(pts, cls="ar", marker="ah", dash=False):
    d = "M" + " L".join("%g,%g" % p for p in pts)
    m = ' marker-end="url(#%s)"' % marker if marker else ""
    els.append('<path d="%s" class="%s%s"%s/>' % (d, cls, " dash" if dash else "", m))


def plus(cx, cy, cls="plus"):
    els.append('<circle cx="%g" cy="%g" r="11" class="%s"/>' % (cx, cy, cls))
    els.append('<path d="M%g,%g L%g,%g M%g,%g L%g,%g" class="plusx"/>' % (cx - 6, cy, cx + 6, cy, cx, cy - 6, cx, cy + 6))


def dot(cx, cy):
    els.append('<circle cx="%g" cy="%g" r="3.5" class="dot"/>' % (cx, cy))


# ---------------------------------------------------------------- 1. girdi: 28 konum
TOK = ["<eos>", "<steps>", "Who", "is", "Owen", "Evans", "'s", "mother", "'s", "father", "?", "Owen", "Evans", "'s",
       "mother", "is", "Julia", "Evans", ".", "Julia", "Evans", "'s", "father", "is", "Harold", "Fisher", ".", "<eos>"]
X0, CW, TY, TH = 40, 41, 84, 32
cx = lambda i: X0 + i * CW + CW / 2
text(40, 30, "GİRDİ", "eyebrow", "start")
text(40, 50, "bir eğitim dizisi, 28 konum", "bs", "start")
text(40, 66, "(hiç görülmemiş torun sorusu biçiminde)", "bs", "start")
for i, t in enumerate(TOK):
    cls = "cell"
    if i == 23:
        cls = "cell query"
    elif i in (22, 14):
        cls = "cell keyhl"
    elif i == 24:
        cls = "cell target"
    rect(X0 + i * CW + 1, TY, CW - 2, TH, cls, rx=3)
    text(cx(i), TY + TH / 2 + 4, t, "tok" + (" tokb" if i in (22, 23, 24, 14) else ""))
    text(cx(i), TY + TH + 14, str(i), "idx")
# RoPE yaylari: konum 23 ("is") -> 22 "father" (1), 14 "mother" (9)
for j, top, cls, lab in ((22, 64, "arc", "1 önce"), (14, 40, "arc weak", "9 önce")):
    x1, x2 = cx(23), cx(j)
    els.append('<path d="M%g,%g C%g,%g %g,%g %g,%g" class="%s" marker-end="url(#ahr)"/>' % (
        x1, TY - 2, x1, top, x2, top, x2, TY - 3, cls))
    text((x1 + x2) / 2, 58 if j == 22 else top - 5, lab, "arcl")
text(1018, 26, "RoPE: skor mesafeye de bağlı", "ropel", "start")
text(1018, 42, "'is' (23) yakındaki 'father'ı", "bs", "start")
text(1018, 56, "sorudaki 'mother'dan ayırır", "bs", "start")
# soru / cevap parantezleri
for a, b, lab in ((1, 10, "soru"), (11, 26, "cevap: iki cümle (model yazar)")):
    xa, xb, yb = X0 + a * CW + 3, X0 + (b + 1) * CW - 3, TY + TH + 24
    els.append('<path d="M%g,%g L%g,%g L%g,%g L%g,%g" class="brk"/>' % (xa, yb - 5, xa, yb, xb, yb, xb, yb - 5))
    text((xa + xb) / 2, yb + 15, lab, "bs")
text(cx(24), TY - 8, "hedef", "tgl")

# ---------------------------------------------------------------- 2. TokenPoints
BY = 196
box(40, BY, 170, 50, "fixed", "PF · 238 × 64", "sabit noktalar, öğrenilmez")
plus(236, BY + 25)
box(262, BY, 170, 50, "learn", "shift · 238 × 64", "öğrenilir, 0'dan")
line([(210, BY + 25), (224, BY + 25)])
line([(262, BY + 25), (248, BY + 25)])
box(150, 270, 172, 44, "op", "PL = norm(PF + shift)", "238 nokta, kürede")
line([(236, BY + 36), (236, 270)])
box(470, 270, 230, 44, "op", "h₀ = PL[ids]", "(B, 28, 64) · konum başına nokta")
line([(322, 292), (470, 292)])
line([(585, 170), (585, 268)])
text(596, 250, "ids", "bs", "start")
text(40, 186, "TokenPoints", "sect", "start")

# ---------------------------------------------------------------- 3. Block (2 tur)
BX, BT, BR, BB = 36, 340, 800, 1060
rect(BX, BT, BR - BX, BB - BT, "block", rx=10)
text(140, BT + 24, "Block", "sect", "start")
text(196, BT + 24, "tek blok, 2 tur, aynı ağırlıklar (SHARED_BLOCK)", "bs", "start")
SX = 120                                                   # akis (h) hatti
line([(585, 314), (585, 328), (SX, 328), (SX, 392)], "st", None)
text(SX + 8, 386, "h", "hl", "start")
# tur 2 donusu
dot(SX, 1046)
line([(SX, 1046), (58, 1046), (58, 360), (SX - 6, 360)], "loop", "ahl")
els.append('<text x="50" y="700" class="loopl" text-anchor="middle" transform="rotate(-90 50 700)">tur 2: çıkış aynı Block\'a yeniden girer</text>')

# -- CausalAttention
AX, AT, AR, AB = 170, 380, 784, 792
rect(AX, AT, AR - AX, AB - AT, "panel", rx=8)
text(AX + 16, AT + 22, "CausalAttention", "pt", "start")
text(AX + 142, AT + 22, "nedensel · tek head · V yok", "bs", "start")
dot(SX, 420)
line([(SX, 420), (200, 420)], "st", None)
els.append('<path d="M200,420 L200,690" class="st"/>')
QY, KY = 440, 506
for y, wn, lab in ((QY, "W_query · 64×64", "q_t"), (KY, "W_key · 64×64", "k_j")):
    line([(200, y + 18), (226, y + 18)], "st", "ahs")
    box(228, y, 128, 36, "learn", wn, tcls="btm")
    line([(356, y + 18), (372, y + 18)])
    box(374, y, 62, 36, "op", "norm", tcls="bt")
    line([(436, y + 18), (452, y + 18)])
    box(454, y, 76, 36, "rope", "RoPE", tcls="btr")
    text(540, y + 23, lab, "hl", "start")
text(454 + 38, KY + 56, "yeni: konuma göre döndür", "ropes")
text(454 + 38, KY + 70, "boy 1 kalır · 0 sayı", "ropes")
box(592, 444, 186, 50, "op", "s_tj = 10,83 · ⟨q_t, k_j⟩", "cosine × sabit ölçek", tcls="btm")
line([(566, QY + 18), (590, 462)])
line([(566, KY + 18), (590, 480)])
box(592, 520, 186, 40, "op", "j > t  →  −∞", "nedensel maske", tcls="btm")
line([(685, 494), (685, 518)])
box(592, 586, 186, 44, "op", "a_tj = softmax_j", "(T × T) · satır toplamı 1", tcls="btm")
line([(685, 560), (685, 584)])
box(400, 660, 370, 48, "op", "c_t = Σ_j a_tj · h_j", "getirilen: durumların kendisi (V yok)", tcls="btm")
line([(685, 630), (685, 658)])
line([(200, 690), (398, 690)], "st", "ahs")
text(300, 682, "değer = h", "bs")
box(480, 730, 210, 44, "learn", "W_context · 64×64", "0'dan başlar", tcls="btm")
line([(585, 708), (585, 728)])
plus(SX, 752)
line([(480, 752), (SX + 13, 752)])
text(300, 744, "W_context · c_t", "bs")
line([(SX, 392), (SX, 739)], "st", "ahs")
box(SX - 48, 810, 96, 40, "op", "L2 norm", "boy 1", tcls="bt")
line([(SX, 763), (SX, 808)], "st", "ahs")
text(SX + 58, 834, "h = norm(h + W_context c)", "form", "start")

# -- FactUnits
FX, FT, FR, FB = 170, 868, 784, 960
rect(FX, FT, FR - FX, FB - FT, "panel", rx=8)
text(FX + 16, FT + 22, "FactUnits", "pt", "start")
text(FX + 96, FT + 22, "her konumda ayrı · 256 birim", "bs", "start")
dot(SX, 918)
line([(SX, 850), (SX, 973)], "st", "ahs")
line([(SX, 918), (226, 918)], "st", "ahs")
box(228, 898, 140, 40, "learn", "W_fact_in · 256×64", tcls="btm")
line([(368, 918), (384, 918)])
box(386, 898, 140, 40, "learn", "− fact_threshold", "256", tcls="btm")
line([(526, 918), (542, 918)])
box(544, 898, 66, 40, "op", "ReLU", "u · 256", tcls="bt")
line([(610, 918), (626, 918)])
box(628, 898, 142, 40, "learn", "W_fact_out · 64×256", "0'dan başlar", tcls="btm")
plus(SX, 985)
line([(699, 938), (699, 985), (SX + 13, 985)])
text(420, 978, "W_fact_out · u", "bs")
box(SX - 48, 1004, 96, 32, "op", "L2 norm", tcls="bt")
line([(SX, 996), (SX, 1002)], "st", None)
text(SX + 58, 1025, "h = norm(h + FactUnits(h))", "form", "start")

# ---------------------------------------------------------------- 4. cikis
OY = 1100
line([(SX, 1036), (SX, OY - 2)], "st", "ahs")
text(SX + 8, 1084, "h₂", "hl", "start")
box(60, OY, 330, 50, "op", "skor_j = 10,06 · ⟨h₂, PL_j⟩", "238 token · 10,06 = ln(0,99 · 237 / 0,01)", tcls="btm")
line([(390, OY + 25), (428, OY + 25)])
box(430, OY, 110, 50, "op", "softmax", "olasılık", tcls="bt")
line([(540, OY + 25), (578, OY + 25)])
box(580, OY, 220, 50, "out", "sonraki token", "konum 23 → 'Harold'", tcls="bt")
# ayni PL (tied)
line([(150, 292), (22, 292), (22, OY + 25), (58, OY + 25)], "tied", "aht")
els.append('<text x="15" y="720" class="tiedl" text-anchor="middle" transform="rotate(-90 15 720)">aynı PL: giriş ve çıkış aynı noktalar</text>')
text(40, OY - 10, "Çıkış", "sect", "start")

# ---------------------------------------------------------------- 5. kayip
LY = 1184
line([(690, OY + 50), (690, LY)])
box(60, LY, 740, 46, "op", "kayıp = cross-entropy(skor, hedef) + 1e-3 · Σ‖shift‖²",
    "bütün konumlarda (<pad> hariç) · Adam", tcls="btm")

# ---------------------------------------------------------------- sag sutun
RX, RW = 840, 370


def panel_rows(y0, title, rows, colx=(0, 230), valcls="num"):
    text(RX, y0, title, "sect", "start")
    y = y0 + 26
    for r in rows:
        if r is None:
            els.append('<path d="M%g,%g L%g,%g" class="rule"/>' % (RX, y - 13, RX + RW, y - 13))
            y += 6
            continue
        k, v, cls = (r + ("",))[:3]
        text(RX + colx[0], y, k, "rk " + cls, "start")
        text(RX + RW, y, v, valcls + " " + cls, "end")
        y += 21
    return y


y = panel_rows(196, "Öğrenilen sayılar", [
    ("shift · 238 × 64", "15.232"), ("W_query · 64 × 64", "4.096"), ("W_key · 64 × 64", "4.096"),
    ("W_context · 64 × 64", "4.096"), ("W_fact_in · 256 × 64", "16.384"), ("fact_threshold · 256", "256"),
    ("W_fact_out · 64 × 256", "16.384"), ("RoPE", "0", "ropek"), None, ("toplam", "60.544", "bold")])
text(RX, y + 2, "tek blok: iki tur aynı sayıları kullanır", "bs", "start")
text(RX, y + 17, "(transformer 115.328 · V'siz 107.008)", "bs", "start")

y = panel_rows(500, "Ayarlar", [
    ("ROPE", "True · yeni", "ropek"), ("SHARED_BLOCK · TURNS", "True · 2"), ("STREAM_NORM", "True"),
    ("LAYER_NORM", "False"), ("COPY_PATH", "False"), ("D · FACT_UNITS", "64 · 256"), ("ANCHOR", "1e-3")],
    valcls="numm")

# RoPE kutusu
RY = 700
text(RX, RY, "RoPE nasıl", "sect", "start")
rect(RX, RY + 12, RW, 196, "ropebox", rx=8)
ccx, ccy, r = RX + 70, RY + 100, 50
els.append('<circle cx="%g" cy="%g" r="%g" class="unit"/>' % (ccx, ccy, r))
import math
for ang, cls, lab in ((20, "vec", "q"), (80, "vec rot", "R_t q")):
    a = math.radians(ang)
    ex, ey = ccx + r * math.cos(a), ccy - r * math.sin(a)
    els.append('<path d="M%g,%g L%g,%g" class="%s" marker-end="url(#%s)"/>' % (ccx, ccy, ex, ey, cls,
                                                                              "ahr" if "rot" in cls else "ah"))
    text(ex + (8 if ang < 50 else -4), ey - 4, lab, "vl" + (" ropek" if "rot" in cls else ""), "start" if ang < 50 else "end")
els.append('<path d="M%g,%g A%g,%g 0 0 0 %g,%g" class="angle"/>' % (
    ccx + 26 * math.cos(math.radians(20)), ccy - 26 * math.sin(math.radians(20)), 26, 26,
    ccx + 26 * math.cos(math.radians(80)), ccy - 26 * math.sin(math.radians(80))))
text(ccx + 30, ccy - 30, "θ", "vl ropek", "start")
text(ccx, ccy + r + 18, "bir boyut çifti", "bs")
tx = RX + 140
for i, s in enumerate(["q, k: 32 boyut çifti, her biri", "θ = t · 10000^(−2i/64) döner",
                       "i = 0: 1 rad / konum", "i = 31: 0,00013 rad / konum",
                       "⟨R_t q, R_j k⟩ yalnız j − t'ye bağlı", "boy değişmez: cosine ve 10,83",
                       "aynen kalır · iki turda da"]):
    text(tx, RY + 40 + i * 21, s, "rs", "start")

# egitim ve sonuc
EY = 940
y = panel_rows(EY, "Eğitim ve sonuç", [
    ("Adam · lr", "0,01 → cosine → 0,001"), ("gradient clipping", "1,0"), ("adım · batch", "4.000 · full (2.240)"),
    ("tohum · veri", "0 · 240bdd2aaebf"), None,
    ("hiç görülmemiş 2R, EX", "187 / 192", "bold"), ("C′ (RoPE'suz)", "89 / 192"),
    ("transformer · V'siz", "171 · 175")], valcls="numm")

# lejant
LG = 1270
items = [("learn", "öğrenilen parametre"), ("op", "parametresiz hesap"), ("fixed", "sabit (PF)"), ("rope", "RoPE (yeni)")]
x = 60
for cls, lab in items:
    rect(x, LG, 22, 14, cls, rx=3)
    text(x + 30, LG + 11, lab, "bs", "start")
    x += 58 + 7.2 * len(lab)
els.append('<path d="M%g,%g L%g,%g" class="st"/>' % (x, LG + 7, x + 30, LG + 7))
text(x + 38, LG + 11, "akış h", "bs", "start")
x += 100
els.append('<path d="M%g,%g L%g,%g" class="tied"/>' % (x, LG + 7, x + 30, LG + 7))
text(x + 38, LG + 11, "aynı PL (tied)", "bs", "start")

DEFS = """<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mk"/></marker>
<marker id="ahs" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mks"/></marker>
<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mkr"/></marker>
<marker id="ahl" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mkl"/></marker>
<marker id="aht" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mkt"/></marker>
</defs>"""

CSS = """
:root{
  --paper:#F5F6F2; --ink:#1B232C; --muted:#5B6773; --line:#3A4552;
  --cell:#FFFFFF; --cellb:#C9D0D6;
  --learn-f:#F4E4C4; --learn-s:#A5641A;
  --op-f:#FFFFFF; --op-s:#3A4552;
  --fixed-f:#E6E9EC; --fixed-s:#7A8793;
  --rope:#0A7A83; --rope-f:#D3EEF0;
  --stream:#2E5596; --loop:#7B5AA6; --tied:#8A6D3B;
  --block-f:#EBEFF4; --block-s:#9AA8B8; --panel-f:#F8FAFC; --panel-s:#C3CDD8;
  --out-f:#E3EFE0; --out-s:#3F7A45;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --paper:#101418; --ink:#DCE3EA; --muted:#94A1AE; --line:#B7C2CD;
    --cell:#171C22; --cellb:#39424C;
    --learn-f:#3A2B12; --learn-s:#E2A24C;
    --op-f:#161B21; --op-s:#AEB9C4;
    --fixed-f:#20262D; --fixed-s:#7F8C98;
    --rope:#4CC8D1; --rope-f:#0E3134;
    --stream:#86A8E6; --loop:#B79BE0; --tied:#CDAE74;
    --block-f:#151A21; --block-s:#3B4652; --panel-f:#12161B; --panel-s:#323C47;
    --out-f:#15261A; --out-s:#76C07E;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --paper:#101418; --ink:#DCE3EA; --muted:#94A1AE; --line:#B7C2CD;
  --cell:#171C22; --cellb:#39424C;
  --learn-f:#3A2B12; --learn-s:#E2A24C;
  --op-f:#161B21; --op-s:#AEB9C4;
  --fixed-f:#20262D; --fixed-s:#7F8C98;
  --rope:#4CC8D1; --rope-f:#0E3134;
  --stream:#86A8E6; --loop:#B79BE0; --tied:#CDAE74;
  --block-f:#151A21; --block-s:#3B4652; --panel-f:#12161B; --panel-s:#323C47;
  --out-f:#15261A; --out-s:#76C07E;
}
body{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding-inline:16px;padding-block:20px 28px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 18px;margin-bottom:10px}
h1{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;font-weight:600;font-size:28px;letter-spacing:.01em;margin:0;text-wrap:balance}
.meta{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.fig{margin:0;overflow-x:auto}
.fig svg{display:block;width:100%;min-width:1040px;height:auto;color:var(--ink)}
figcaption{font-size:13px;color:var(--muted);max-width:75ch;margin-top:8px;line-height:1.5}
svg text{fill:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif}
.eyebrow{font-size:11px;letter-spacing:.14em;font-weight:600;fill:var(--muted)}
.sect{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-size:16px;font-weight:600}
.pt{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;font-size:15px;font-weight:600}
.bt{font-size:13px;font-weight:500}
.btm{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;font-weight:500}
.btr{font-size:13px;font-weight:600;fill:var(--rope)!important}
.bs{font-size:11px;fill:var(--muted)!important}
.form{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;fill:var(--muted)!important}
.hl{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;font-weight:600;fill:var(--stream)!important}
.tok{font-size:10px}
.tokb{font-weight:700}
.idx{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:9px;fill:var(--muted)!important}
.tgl{font-size:10px;font-weight:600;fill:var(--out-s)!important}
.arcl{font-size:10px;font-weight:600;fill:var(--rope)!important}
.ropel{font-size:12.5px;font-weight:600;fill:var(--rope)!important}
.ropes{font-size:10.5px;fill:var(--rope)!important}
.ropek{fill:var(--rope)!important;font-weight:600}
.rk{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px}
.rs{font-size:12px}
.num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;font-variant-numeric:tabular-nums}
.numm{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px}
.bold{font-weight:700}
.vl{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px}
.loopl{font-size:11px;fill:var(--loop)!important;font-weight:500}
.tiedl{font-size:11px;fill:var(--tied)!important;font-weight:500}
rect.cell{fill:var(--cell);stroke:var(--cellb);stroke-width:1}
rect.query{stroke:var(--rope);stroke-width:2}
rect.keyhl{fill:var(--rope-f);stroke:var(--rope)}
rect.target{fill:var(--out-f);stroke:var(--out-s)}
rect.learn{fill:var(--learn-f);stroke:var(--learn-s);stroke-width:1.4}
rect.op{fill:var(--op-f);stroke:var(--op-s);stroke-width:1.2}
rect.fixed{fill:var(--fixed-f);stroke:var(--fixed-s);stroke-width:1.2;stroke-dasharray:4 3}
rect.rope{fill:var(--rope-f);stroke:var(--rope);stroke-width:2}
rect.out{fill:var(--out-f);stroke:var(--out-s);stroke-width:1.4}
rect.block{fill:var(--block-f);stroke:var(--block-s);stroke-width:1.4;stroke-dasharray:7 4}
rect.panel{fill:var(--panel-f);stroke:var(--panel-s);stroke-width:1}
rect.ropebox{fill:var(--panel-f);stroke:var(--rope);stroke-width:1.2}
path.ar{fill:none;stroke:var(--line);stroke-width:1.4}
path.st{fill:none;stroke:var(--stream);stroke-width:2.6}
path.loop{fill:none;stroke:var(--loop);stroke-width:1.8;stroke-dasharray:6 4}
path.tied{fill:none;stroke:var(--tied);stroke-width:1.6;stroke-dasharray:3 4}
path.arc{fill:none;stroke:var(--rope);stroke-width:2}
path.arc.weak{stroke-width:1.2;stroke-dasharray:4 3;opacity:.8}
path.brk{fill:none;stroke:var(--muted);stroke-width:1}
path.rule{stroke:var(--panel-s);stroke-width:1}
path.vec{stroke:var(--ink);stroke-width:1.8}
path.vec.rot{stroke:var(--rope);stroke-width:2.2}
path.angle{fill:none;stroke:var(--rope);stroke-width:1.2}
path.plusx{stroke:var(--ink);stroke-width:1.6}
circle.plus{fill:var(--op-f);stroke:var(--ink);stroke-width:1.4}
circle.dot{fill:var(--stream)}
circle.unit{fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.mk{fill:var(--line)} .mks{fill:var(--stream)} .mkr{fill:var(--rope)} .mkl{fill:var(--loop)} .mkt{fill:var(--tied)}
"""

svg = ('<svg viewBox="0 0 %d %d" role="img" aria-label="Model X (model_20 C′ + RoPE) mimarisi: token noktaları PL, iki tur '
       'aynı Block (RoPE\'lu nedensel attention ve FactUnits, her eklemeden sonra L2 norm), çıkışta aynı PL ile skor.">'
       % (W, H)) + DEFS + "\n".join(els) + "</svg>"

html = """<title>Model X</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>%s</style>
<div class="wrap">
<header>
<h1>Model X</h1>
<span class="meta">model_20 · C′ + RoPE · koşu shared_steps_long1r_rope_s0 · kod 6262c99 · 27 Eylül 2026</span>
</header>
<figure class="fig">
%s
<figcaption>Bir konumun yolu: token noktası PL'den başlar, aynı Block'tan iki kez geçer (attention durumlara bakar, FactUnits
dönüştürür, her eklemeden sonra L2 norm), son durum aynı PL noktalarıyla karşılaştırılıp sonraki token seçilir. C′'den tek
fark RoPE: q ve k konuma göre döndürülür, öğrenilen sayı eklenmez.</figcaption>
</figure>
</div>
""" % (CSS, svg)
open(OUT, "w", encoding="utf-8").write(html)
print(OUT, len(html))
