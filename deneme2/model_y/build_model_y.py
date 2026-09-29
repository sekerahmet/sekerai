# -*- coding: utf-8 -*-
"""Model Y (3 blok x 2 tur, SimpleStories ayari A) mimari sayfasi: tek SVG diyagram.  Cikti: model_y.html
Model X'e gore yeni parcalar yesil cerceveyle isaretli."""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_y.html")
W, H = 1240, 1480
els = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="t", anchor="middle", extra=""):
    els.append('<text x="%g" y="%g" class="%s" text-anchor="%s"%s>%s</text>' % (x, y, cls, anchor, extra, esc(s)))


def rect(x, y, w, h, cls, rx=6):
    els.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" class="%s"/>' % (x, y, w, h, rx, cls))


def box(x, y, w, h, cls, title, sub=None, tcls="bt", scls="bs", new=False):
    rect(x, y, w, h, cls)
    if new:                                                   # Model Y'de yeni
        rect(x - 3, y - 3, w + 6, h + 6, "newring", rx=8)
    cy = y + h / 2
    if sub:
        text(x + w / 2, cy - 3, title, tcls)
        text(x + w / 2, cy + 13, sub, scls)
    else:
        text(x + w / 2, cy + 4.5, title, tcls)


def line(pts, cls="ar", marker="ah"):
    d = "M" + " L".join("%g,%g" % p for p in pts)
    m = ' marker-end="url(#%s)"' % marker if marker else ""
    els.append('<path d="%s" class="%s"%s/>' % (d, cls, m))


def dot(cx, cy, cls="dot"):
    els.append('<circle cx="%g" cy="%g" r="3.5" class="%s"/>' % (cx, cy, cls))


# ---------------------------------------------------------------- 1. girdi: 22 konum
TOK = ["<eos>", "Lily", "and", "Tom", "went", "to", "the", "park", ".", "Lily", "had", "a", "red", "ball", ".", "She",
       "gave", "the", "ball", "to", "Tom", "."]
X0, CW, TY, TH = 40, 41, 84, 32
cx = lambda i: X0 + i * CW + CW / 2
Q, T_, NEAR, FAR = 19, 20, 9, 3
for i, t in enumerate(TOK):
    cls = "cell"
    if i in (16, 17, 18):
        cls = "cell canonhl"
    elif i == Q:
        cls = "cell canonhl query"
    elif i == FAR:
        cls = "cell keyhl"
    elif i == T_:
        cls = "cell target"
    rect(X0 + i * CW + 1, TY, CW - 2, TH, cls, rx=3)
    text(cx(i), TY + TH / 2 + 4, t, "tok" + (" tokb" if i in (Q, T_, NEAR, FAR) else ""))
    text(cx(i), TY + TH + 14, str(i), "idx")
for j, top, cls, lab, ly in ((FAR, 36, "arc", "16 önce: Tom", 38), (NEAR, 58, "arc weak", "10 önce: Lily", 76)):
    x1, x2 = cx(Q), cx(j)
    els.append('<path d="M%g,%g C%g,%g %g,%g %g,%g" class="%s" marker-end="url(#ahr)"/>' % (
        x1, TY - 2, x1, top, x2, top, x2, TY - 3, cls))
    text((x1 + x2) / 2, ly, lab, "arcl")
text(cx(T_), TY - 8, "hedef", "tgl")
xa, xb, yb = X0 + 16 * CW + 3, X0 + 20 * CW - 3, TY + TH + 24
els.append('<path d="M%g,%g L%g,%g L%g,%g L%g,%g" class="brkc"/>' % (xa, yb - 5, xa, yb, xb, yb, xb, yb - 5))
text((xa + xb) / 2, yb + 15, "Canon: son 4 konum (t−3 … t)", "canons")
text(980, 26, "GİRDİ", "eyebrow", "start")
text(980, 44, "örnek dizi, 22 konum", "bs", "start")
text(980, 58, "gösterim kelimeyle; gerçekte alt-kelime", "bs", "start")
text(980, 72, "token'ları (ss4096 ya da gpt2)", "bs", "start")
text(980, 96, "attention uzaktaki adı seçer", "ropel", "start")
text(980, 112, "'to' (19): 'Tom' mu, 'Lily' mi?", "bs", "start")
text(980, 136, "Canon: yakın ifadeyi karıştırır", "canonl", "start")
text(980, 152, "'gave the ball to' tek girdide", "bs", "start")

# ---------------------------------------------------------------- 2. TokenPoints
BY = 196
text(40, 186, "TokenPoints", "sect", "start")
box(40, BY, 170, 50, "fixed", "PF · V × 384", "sabit noktalar, öğrenilmez")
els.append('<circle cx="236" cy="%g" r="11" class="plus"/>' % (BY + 25))
els.append('<path d="M230,%g L242,%g M236,%g L236,%g" class="plusx"/>' % (BY + 25, BY + 25, BY + 19, BY + 31))
box(262, BY, 170, 50, "learn", "shift · V × 384", "öğrenilir, 0'dan")
line([(210, BY + 25), (224, BY + 25)])
line([(262, BY + 25), (248, BY + 25)])
box(150, 270, 172, 44, "op", "PL = norm(PF + shift)", "V nokta, kürede")
line([(236, BY + 36), (236, 270)])
box(470, 270, 230, 44, "op", "h₀ = PL[ids]", "(B, T, 384) · konum başına nokta")
line([(322, 292), (470, 292)])
line([(585, 170), (585, 268)])
text(596, 250, "ids", "bs", "start")
text(600, 214, "V = 4.096 (ss4096) · 50.257 (gpt2)", "bs", "start")

# ---------------------------------------------------------------- 3. Block: 3 farkli Block, 6 tur (A B C A B C)
BX, BT, BR, BB = 36, 322, 800, 1212
SX = 120
rect(BX, BT, BR - BX, BB - BT, "block", rx=10)
text(140, BT + 22, "Block", "sect", "start")
text(192, BT + 22, "3 farklı Block (A, B, C) · 6 tur: A B C A B C · her Block kendi ağırlıkları · α tur başına", "bs", "start")
line([(585, 314), (585, 318), (SX, 318), (SX, 864)], "st", "ahs")
text(SX + 8, 474, "h", "hl", "start")

# -- Canon
CX, CT, CR, CB = 170, 352, 784, 444
rect(CX, CT, CR - CX, CB - CT, "canonpanel", rx=8)
text(CX + 16, CT + 22, "Canon", "pt canonk", "start")
text(CX + 66, CT + 22, "attention'ın girdisi · son 4 konum, boyut başına ağırlık", "bs", "start")
box(228, 386, 166, 32, "learn", "canon_weights · 4 × 384", tcls="btm")
box(408, 388, 362, 44, "canon", "x_t = h_t + Σₖ wₖ ⊙ h_(t−k)", "k = 0..3 · ⊙ boyut başına çarpım · w 0'dan", tcls="btm")
line([(394, 402), (406, 402)])
dot(SX, 426)
line([(SX, 426), (406, 426)], "st", "ahs")

# -- CausalAttention
AX, AT, AR, AB = 170, 462, 784, 846
rect(AX, AT, AR - AX, AB - AT, "panel", rx=8)
text(AX + 16, AT + 22, "CausalAttention", "pt", "start")
text(AX + 142, AT + 22, "nedensel · 4 head × 96 boyut · her head kendi q, k, a ve değer dilimi", "bs", "start")
line([(585, 432), (585, 496), (200, 496), (200, 744), (226, 744)], "xs", "ahc")
text(596, 454, "x", "xl", "start")
QY, KY = 512, 576
for y, wn, lab in ((QY, "W_query · 384×384", "q_t"), (KY, "W_key · 384×384", "k_j")):
    line([(200, y + 18), (226, y + 18)], "xs", "ahc")
    box(228, y, 140, 36, "learn", wn, tcls="btm")
    line([(368, y + 18), (382, y + 18)])
    box(384, y, 62, 36, "op", "norm", "head başına", tcls="bt")
    line([(446, y + 18), (460, y + 18)])
    box(462, y, 76, 36, "rope", "RoPE", tcls="btr")
    text(546, y + 23, lab, "hl", "start")
text(500, KY + 56, "konuma göre döndür", "ropes")
text(500, KY + 70, "boy 1 kalır · 0 sayı", "ropes")
box(598, 516, 180, 50, "op", "s_tj = 10,83 · ⟨q_t, k_j⟩", "head başına · cosine × sabit ölçek", tcls="btm")
line([(570, QY + 18), (596, 534)])
line([(570, KY + 18), (596, 552)])
box(598, 590, 180, 40, "op", "j > t  →  −∞", "nedensel maske", tcls="btm")
line([(688, 566), (688, 588)])
box(598, 652, 180, 44, "op", "a_tj = softmax_j", "4 head × (T × T) · satır toplamı 1", tcls="btm")
line([(688, 630), (688, 650)])
box(400, 720, 370, 48, "op", "c_t = [Σ_j a¹_tj v¹_j ; … ; Σ_j a⁴_tj v⁴_j]", "4 head yan yana · vʰ = W_value x'in h. dilimi", tcls="btm")
line([(688, 696), (688, 718)])
box(228, 726, 150, 36, "learn", "W_value · 384×384", tcls="btm")
text(303, 776, "satırlar birim", "bs")
line([(378, 744), (398, 744)])
box(480, 790, 210, 44, "learn", "W_context · 384×384", "sütunlar birim", tcls="btm")
line([(585, 768), (585, 788)])

# -- guncelleme A
line([(480, 812), (150, 812), (150, 864)])
text(300, 806, "norm(W_context · c)", "bs")
box(SX - 56, 866, 112, 44, "op", "yaklaş + norm", "α_A", tcls="bt")
box(600, 868, 180, 40, "learn", "α_A · 6 × 384", "tur başına, 0,1'den", tcls="btm")
line([(600, 888), (178, 888)])
text(190, 930, "h = norm(h + α_A ⊙ (norm(W_context c) − h))", "form", "start")

# -- FactUnits (SwiGLU)
FX, FT, FR, FB = 170, 946, 784, 1100
rect(FX, FT, FR - FX, FB - FT, "panel", rx=8)
text(FX + 16, FT + 22, "FactUnits", "pt", "start")
text(FX + 96, FT + 22, "her konumda ayrı · SwiGLU · 1024 birim (8/3·d) · girdi x = √d · h", "bs", "start")
line([(SX, 910), (SX, 1110)], "st", "ahs")
dot(SX, 1000)
line([(SX, 1000), (226, 1000)], "st", "ahs")
dot(SX, 1056)
line([(SX, 1056), (226, 1056)], "st", "ahs")
box(228, 980, 150, 40, "learn", "W_fact_in · 1024×384", "satırlar birim · kapı", tcls="btm")
line([(378, 1000), (392, 1000)])
box(394, 980, 70, 40, "op", "SiLU", "kapı", tcls="bt", new=True)
box(228, 1036, 150, 40, "learn", "W_fact_up · 1024×384", "satırlar birim · içerik", tcls="btm", new=True)
line([(464, 1000), (506, 1000), (506, 1013)])
line([(378, 1056), (506, 1056), (506, 1041)])
els.append('<circle cx="506" cy="1027" r="13" class="mul"/>')
text(506, 1032, "⊙", "mulx")
text(530, 1022, "u", "hl", "start")
line([(519, 1027), (600, 1027)])
box(602, 1007, 170, 40, "learn", "W_fact_out · 384×1024", "sütunlar birim", tcls="btm")

# -- guncelleme F
line([(687, 1047), (687, 1088), (150, 1088), (150, 1110)])
text(560, 1082, "norm(W_fact_out · u)", "bs")
box(SX - 56, 1112, 112, 44, "op", "yaklaş + norm", "α_F", tcls="bt")
box(600, 1114, 180, 40, "learn", "α_F · 6 × 384", "0,1'den · son tur 1'den", tcls="btm", new=True)
line([(600, 1134), (178, 1134)])
text(190, 1176, "h = norm(h + α_F ⊙ (norm(W_fact_out u) − h))", "form", "start")

# -- tur donusu
line([(SX, 1156), (SX, 1196)], "st", None)
dot(SX, 1196)
line([(SX, 1196), (58, 1196), (58, 360), (SX - 6, 360)], "loop", "ahl")
els.append('<text x="50" y="780" class="loopl" text-anchor="middle" transform="rotate(-90 50 780)">'
           'tur 2–6: sıradaki Block (A B C A B C), aynı akış</text>')

# ---------------------------------------------------------------- 4. cikis
OY = 1248
line([(SX, 1196), (SX, OY - 2)], "st", "ahs")
text(SX + 8, 1230, "h₆", "hl", "start")
text(40, OY - 10, "Çıkış", "sect", "start")
box(60, OY, 360, 50, "op", "skor_j = e^τ · ⟨h₆, PL_j⟩", "τ = log_output_scale öğrenilir · başta ln 12,91 / ln 15,42",
    tcls="btm", new=True)
line([(420, OY + 25), (446, OY + 25)])
box(448, OY, 100, 50, "op", "softmax", "olasılık", tcls="bt")
line([(548, OY + 25), (574, OY + 25)])
box(576, OY, 224, 50, "out", "sonraki token", "konum 19 → 'Tom'", tcls="bt")
line([(150, 292), (22, 292), (22, OY + 25), (58, OY + 25)], "tied", "aht")
els.append('<text x="15" y="780" class="tiedl" text-anchor="middle" transform="rotate(-90 15 780)">'
           'aynı PL: giriş ve çıkış aynı noktalar</text>')

# ---------------------------------------------------------------- 5. kayip
LY = 1334
line([(690, OY + 50), (690, LY)])
box(60, LY, 740, 46, "op", "kayıp = cross-entropy(skor, hedef) + 1e-3 · Σ‖shift‖²",
    "LOSS_CHUNK 4096: sözlük parça parça, skor tablosu hiç oluşmaz · yalnız gerçek token'lar", tcls="btm", new=True)

# ---------------------------------------------------------------- sag sutun
RX, RW = 840, 370


def panel_rows(y0, title, rows, valcls="num"):
    text(RX, y0, title, "sect", "start")
    y = y0 + 26
    for r in rows:
        if r is None:
            els.append('<path d="M%g,%g L%g,%g" class="rule"/>' % (RX, y - 13, RX + RW, y - 13))
            y += 6
            continue
        k, v, cls = (r + ("",))[:3]
        text(RX, y, k, "rk " + cls, "start")
        text(RX + RW, y, v, valcls + " " + cls, "end")
        y += 21
    return y


y = panel_rows(196, "Öğrenilen sayılar · ayar A", [
    ("shift · V × 384", "1.572.864 / 19.298.688"),
    ("", "ss4096 / gpt2"), None,
    ("Block başına (A, B, C ayrı)", "", "bold"),
    ("canon_weights · 4 × 384", "1.536", "canonk"),
    ("W_query, W_key · 384 × 384", "2 × 147.456"),
    ("W_value, W_context · 384 × 384", "2 × 147.456"),
    ("W_fact_in, W_fact_up · 1024 × 384", "2 × 393.216"),
    ("W_fact_out · 384 × 1024", "393.216"),
    ("Block toplamı", "1.771.008"), None,
    ("3 Block", "5.313.024"),
    ("α_A, α_F · 2 × 6 × 384", "4.608"),
    ("log_output_scale", "1", "newk"),
    ("toplam · ss4096", "6.890.497", "bold"),
    ("toplam · gpt2", "24.616.321", "bold"),
    ("embedding dışı", "5.317.633")])
text(RX, y + 2, "SimpleStories-11M: 6 katman, d 384, 11M · eşit hesap", "bs", "start")
text(RX, y + 17, "(6 tur = 6 katman), %37 daha az parametre", "bs", "start")

panel_rows(620, "Ayarlar", [
    ("LAYERS · TURNS", "3 · 6 (A B C A B C)"), ("HEADS · D", "4 × 96 · 384"),
    ("FACT_ACTIVATION · FACT_UNITS", "swiglu · 1024"), ("CANON · ROPE", "True · True", "canonk"),
    ("NORMALIZED_UPDATE", "True · α 0,1'den"), ("LAST_FACTS_ALPHA_INIT", "1,0 (son tur α_F)", "newk"),
    ("LEARN_OUTPUT_SCALE", "True", "newk"), ("SPHERE_WEIGHTS · ANCHOR", "True · 1e-3"),
    ("T_MAX · üretim önbelleği", "512 · Canon 3 satır", "newk")], valcls="numm")

# yaklas + norm kutusu
UY = 862
text(RX, UY, "Yaklaş + norm nasıl", "sect", "start")
rect(RX, UY + 12, RW, 196, "updbox", rx=8)
ccx, ccy, r = RX + 72, UY + 116, 52
els.append('<circle cx="%g" cy="%g" r="%g" class="unit"/>' % (ccx, ccy, r))
pt = lambda deg: (ccx + r * math.cos(math.radians(deg)), ccy - r * math.sin(math.radians(deg)))
(hx, hy), (ux, uy) = pt(15), pt(115)
alpha = 0.35
mx, my = hx + alpha * (ux - hx), hy + alpha * (uy - hy)
nd = math.degrees(math.atan2(ccy - my, mx - ccx))
nx, ny = pt(nd)
els.append('<path d="M%g,%g L%g,%g" class="chord"/>' % (hx, hy, ux, uy))
els.append('<path d="M%g,%g L%g,%g" class="vec" marker-end="url(#ah)"/>' % (ccx, ccy, hx, hy))
els.append('<path d="M%g,%g L%g,%g" class="vec" marker-end="url(#ah)"/>' % (ccx, ccy, ux, uy))
els.append('<path d="M%g,%g L%g,%g" class="vec new" marker-end="url(#ahs)"/>' % (ccx, ccy, nx, ny))
els.append('<circle cx="%g" cy="%g" r="3" class="pt"/>' % (mx, my))
text(hx + 6, hy + 4, "h", "vl", "start")
text(ux - 4, uy - 6, "norm(u)", "vl", "end")
text(nx + 4, ny - 10, "yeni h", "vl hl", "start")
text(ccx, ccy + r + 18, "bir boyut düzlemi", "bs")
tx = RX + 158
for i, s in enumerate(["h, norm(u): kürede iki nokta", "h + α ⊙ (u − h): arada bir nokta", "norm: yeniden küreye",
                       "α boyut ve tur başına", "son tur α_F = 1'den: başta", "durum = norm(f), girdiyi", "tekrar etmez (kayıp 12,4 → 9,2)"]):
    text(tx, UY + 40 + i * 21, s, "rs" + (" newk" if i >= 4 else ""), "start")

# egitim ve sonuc
y = panel_rows(1100, "Eğitim", [
    ("optimizer", "Muon + Adam"), ("MUON_TANGENT", "önce teğete izdüş, sonra NS", "newk"),
    ("lr (coherence)", "0,01 × ort(ρ)^COHERENCE_POWER (1)"), ("ort(ρ)", "pay / payda ayrı ortalanır", "newk"),
    ("MATMUL_PRECISION", "bf16 · ağırlık ve EMA fp32", "newk"), ("LOSS_CHUNK", "4096", "newk"),
    ("koşu: son iniş · EMA", "%20 · 0,999"), ("batch", "64 × 512")], valcls="numm")
panel_rows(y + 16, "Sonuç", [
    ("ilk koşu (SimpleStories)", "bekleniyor", "bold"),
    ("kıyas", "SS-11M · aynı test · bits_per_byte"),
    ("Model X · TinyStories 1 epok", "ppl 5,99 · 0,594 bits/char")], valcls="numm")

# lejant
LG = 1432
items = [("learn", "öğrenilen parametre"), ("op", "parametresiz hesap"), ("fixed", "sabit (PF)"), ("canon", "Canon"),
         ("rope", "RoPE")]
x = 60
for cls, lab in items:
    rect(x, LG, 22, 14, cls, rx=3)
    text(x + 30, LG + 11, lab, "bs", "start")
    x += 58 + 7.2 * len(lab)
rect(x, LG - 2, 26, 18, "newring", rx=4)
text(x + 34, LG + 11, "Model Y'de yeni", "bs newk", "start")
x += 70 + 7.2 * 15
for cls, lab in (("st", "akış h"), ("tied", "aynı PL (tied)")):
    els.append('<path d="M%g,%g L%g,%g" class="%s"/>' % (x, LG + 7, x + 30, LG + 7, cls))
    text(x + 38, LG + 11, lab, "bs", "start")
    x += 60 + 7.2 * len(lab)

MARK = ('<marker id="%s" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" '
        'markerHeight="10" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="%s"/></marker>')
DEFS = "<defs>" + "".join(MARK % (i, c) for i, c in (("ah", "mk"), ("ahs", "mks"), ("ahr", "mkr"), ("ahl", "mkl"),
                                                      ("aht", "mkt"), ("ahc", "mkc"))) + "</defs>"

DARK = """--paper:#101418; --ink:#DCE3EA; --muted:#94A1AE; --line:#B7C2CD;
    --cell:#171C22; --cellb:#39424C;
    --learn-f:#3A2B12; --learn-s:#E2A24C;
    --op-f:#161B21; --op-s:#AEB9C4;
    --fixed-f:#20262D; --fixed-s:#7F8C98;
    --rope:#4CC8D1; --rope-f:#0E3134;
    --canon:#D98AD3; --canon-f:#34182F;
    --stream:#86A8E6; --loop:#B79BE0; --tied:#CDAE74;
    --block-f:#151A21; --block-s:#3B4652; --panel-f:#12161B; --panel-s:#323C47;
    --out-f:#15261A; --out-s:#76C07E; --new:#5FD08C;"""

CSS = """
:root{
  --paper:#F5F6F2; --ink:#1B232C; --muted:#5B6773; --line:#3A4552;
  --cell:#FFFFFF; --cellb:#C9D0D6;
  --learn-f:#F4E4C4; --learn-s:#A5641A;
  --op-f:#FFFFFF; --op-s:#3A4552;
  --fixed-f:#E6E9EC; --fixed-s:#7A8793;
  --rope:#0A7A83; --rope-f:#D3EEF0;
  --canon:#8E3B8A; --canon-f:#F4E3F2;
  --stream:#2E5596; --loop:#7B5AA6; --tied:#8A6D3B;
  --block-f:#EBEFF4; --block-s:#9AA8B8; --panel-f:#F8FAFC; --panel-s:#C3CDD8;
  --out-f:#E3EFE0; --out-s:#3F7A45; --new:#1E8A4C;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    %s
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  %s
}
body{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding-inline:16px;padding-block:20px 28px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 18px;margin-bottom:10px}
h1{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;font-weight:600;font-size:28px;letter-spacing:.01em;margin:0;text-wrap:balance}
.meta{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace;font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.fig{margin:0;overflow-x:auto}
.fig svg{display:block;width:100%%;min-width:1040px;height:auto;color:var(--ink)}
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
.xl{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;font-weight:600;fill:var(--canon)!important}
.tok{font-size:10px}
.tokb{font-weight:700}
.idx{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:9px;fill:var(--muted)!important}
.tgl{font-size:10px;font-weight:600;fill:var(--out-s)!important}
.arcl{font-size:10px;font-weight:600;fill:var(--rope)!important}
.ropel{font-size:12.5px;font-weight:600;fill:var(--rope)!important}
.ropes{font-size:10.5px;fill:var(--rope)!important}
.canonl{font-size:12.5px;font-weight:600;fill:var(--canon)!important}
.canons{font-size:10.5px;font-weight:600;fill:var(--canon)!important}
.canonk{fill:var(--canon)!important;font-weight:600}
.newk{fill:var(--new)!important;font-weight:600}
.rk{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px}
.rs{font-size:12px}
.num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;font-variant-numeric:tabular-nums}
.numm{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px}
.bold{font-weight:700}
.vl{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px}
.loopl{font-size:11px;fill:var(--loop)!important;font-weight:500}
.tiedl{font-size:11px;fill:var(--tied)!important;font-weight:500}
.mulx{font-size:15px;font-weight:600}
rect.cell{fill:var(--cell);stroke:var(--cellb);stroke-width:1}
rect.canonhl{fill:var(--canon-f);stroke:var(--canon)}
rect.query{stroke:var(--rope);stroke-width:2.2}
rect.keyhl{fill:var(--rope-f);stroke:var(--rope)}
rect.target{fill:var(--out-f);stroke:var(--out-s)}
rect.learn{fill:var(--learn-f);stroke:var(--learn-s);stroke-width:1.4}
rect.op{fill:var(--op-f);stroke:var(--op-s);stroke-width:1.2}
rect.fixed{fill:var(--fixed-f);stroke:var(--fixed-s);stroke-width:1.2;stroke-dasharray:4 3}
rect.rope{fill:var(--rope-f);stroke:var(--rope);stroke-width:2}
rect.canon{fill:var(--canon-f);stroke:var(--canon);stroke-width:1.8}
rect.out{fill:var(--out-f);stroke:var(--out-s);stroke-width:1.4}
rect.block{fill:var(--block-f);stroke:var(--block-s);stroke-width:1.4;stroke-dasharray:7 4}
rect.panel{fill:var(--panel-f);stroke:var(--panel-s);stroke-width:1}
rect.canonpanel{fill:var(--panel-f);stroke:var(--canon);stroke-width:1.2}
rect.updbox{fill:var(--panel-f);stroke:var(--stream);stroke-width:1.2}
rect.newring{fill:none;stroke:var(--new);stroke-width:2;stroke-dasharray:5 3}
path.ar{fill:none;stroke:var(--line);stroke-width:1.4}
path.st{fill:none;stroke:var(--stream);stroke-width:2.6}
path.xs{fill:none;stroke:var(--canon);stroke-width:2.2}
path.loop{fill:none;stroke:var(--loop);stroke-width:1.8;stroke-dasharray:6 4}
path.tied{fill:none;stroke:var(--tied);stroke-width:1.6;stroke-dasharray:3 4}
path.arc{fill:none;stroke:var(--rope);stroke-width:2}
path.arc.weak{stroke-width:1.2;stroke-dasharray:4 3;opacity:.8}
path.brkc{fill:none;stroke:var(--canon);stroke-width:1.2}
path.rule{stroke:var(--panel-s);stroke-width:1}
path.vec{stroke:var(--ink);stroke-width:1.8}
path.vec.new{stroke:var(--stream);stroke-width:2.4}
path.chord{fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 3}
path.plusx{stroke:var(--ink);stroke-width:1.6}
circle.plus{fill:var(--op-f);stroke:var(--ink);stroke-width:1.4}
circle.mul{fill:var(--op-f);stroke:var(--new);stroke-width:1.8}
circle.dot{fill:var(--stream)}
circle.pt{fill:var(--ink)}
circle.unit{fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.mk{fill:var(--line)} .mks{fill:var(--stream)} .mkr{fill:var(--rope)} .mkl{fill:var(--loop)} .mkt{fill:var(--tied)}
.mkc{fill:var(--canon)}
""" % (DARK, DARK)

svg = ('<svg viewBox="0 0 %d %d" role="img" aria-label="Model Y, 3 blok x 2 tur: token noktaları PL; altı tur A B C A B C '
       'sırasıyla üç farklı Block (Canon, RoPE\'lu nedensel attention ve SwiGLU FactUnits; her adım durumu alpha kadar '
       'yaklaştırıp küreye geri koyar); çıkışta aynı PL ile, öğrenilen ölçekle skor.">' % (W, H)) + DEFS + "\n".join(els) + "</svg>"

html = """<title>Model Y</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>%s</style>
<div class="wrap">
<header>
<h1>Model Y · 3 × 2</h1>
<span class="meta">model_y · 3 blok × 2 tur · 4 head · SwiGLU · SimpleStories ayarı A (d 384) · tokenizer ss4096 / gpt2 · 28 Eylül 2026</span>
</header>
<figure class="fig">
%s
<figcaption>Bir konumun yolu: token noktası PL'den başlar ve altı turda üç farklı Block'tan A B C A B C sırasıyla geçer. Her
turda Canon son 4 konumu boyut başına karıştırır, 4 head'li attention bu karışımlara bakıp uzaktakini getirir, SwiGLU
FactUnits konumu kendi içinde dönüştürür; her adım durumu α kadar hedefe yaklaştırıp küreye geri koyar. Son durum aynı PL
noktalarıyla karşılaştırılır; ölçek artık öğrenilir. Yeşil çerçeve: Model X'e göre yeni.</figcaption>
</figure>
</div>
""" % (CSS, svg)
open(OUT, "w", encoding="utf-8").write(html)
print(OUT, len(html))
