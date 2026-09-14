# ============================================================================
# HUCRE 2 - OLCUM HATTI + DENETIM 4 (D3 BASLAMADAN ONCE)
# CLAUDE.md 4: tek olcum yolu. CLAUDE.md 10a: bilinen degere karsi sina.
# D3'un sayisi, D'nin sayisiyla kiyaslanacak -> IKISI DE BU HATTAN gecmeli.
# Sabit 0.322'ye guvenmiyoruz; D'yi bu defter YENIDEN olcer.
# ============================================================================
facts, pairs, one, tr2, comp_, ent_ev, seen_ent, unseen_ent, ent2_ev = S.build_data()
lo, hi = S.ENT_OFF, S.ENT_OFF + S.CFG["N_ENT"]
N = 3000

SET = {}
for ad, lst in (("COMP", comp_), ("ENT", ent_ev), ("ENT2", ent2_ev)):
    l = lst[:N]                       # (e, r1, r2, b, a)
    E = S.enc_two(l)
    SET[ad] = dict(X=E[0], poz=E[1],
                   gold=(S.ENT_OFF + np.array([x[4] for x in l])).astype(np.int64),
                   kis=(S.ENT_OFF + np.array([facts[x[0], x[2]] for x in l])).astype(np.int64))
_rg = np.random.RandomState(7)
_e = _rg.randint(S.CFG["N_ENT"], size=N); _r = _rg.randint(S.CFG["N_REL"], size=N)
_E1 = S.enc_one([(int(a), int(b), int(facts[a, b])) for a, b in zip(_e, _r)])
SET["1hop"] = dict(X=_E1[0], poz=_E1[1], gold=_E1[2], kis=None)
print("olcum kumeleri:", {k: len(v["X"]) for k, v in SET.items()})

_net = S.Net("A", S.CFG).to(S.DEV)

def maske_kur(net, blok):
    for i, b in enumerate(net.blocks): b.mask_key = 1 if i in blok else None

@torch.no_grad()
def OLC(net, bs=500):
    o = {}
    for ad, d in SET.items():
        X, poz, g = d["X"], d["poz"], d["gold"]; ok, kk = [], []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(S.DEV)
            ix = torch.from_numpy(poz[i:i+bs]).to(S.DEV)
            ar = torch.arange(len(ix), device=S.DEV); lg, _ = net(xb)
            p = (lg.float()[ar, ix][:, lo:hi].argmax(-1) + lo).cpu().numpy()
            ok.append(p == g[i:i+bs])
            if d["kis"] is not None: kk.append(p == d["kis"][i:i+bs])
        o[ad] = float(np.concatenate(ok).mean())
        if kk: o[ad + "_kis"] = float(np.concatenate(kk).mean())
    return o

def olc_snap(yol, blok):
    sd = torch.load(yol, map_location=S.DEV)
    _net.load_state_dict({k: v.float() for k, v in sd.items()})
    maske_kur(_net, blok); _net.eval(); r = OLC(_net); maske_kur(_net, ()); return r

def olc_ortalama(yollar, blok):
    acc = None
    for y in yollar:
        sd = torch.load(y, map_location=S.DEV)
        if acc is None: acc = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k in acc: acc[k] += sd[k].float()
    for k in acc: acc[k] /= len(yollar)
    _net.load_state_dict(acc); maske_kur(_net, blok); _net.eval()
    r = OLC(_net); maske_kur(_net, ()); return r

# --- DENETIM 4: D'nin BILINEN egri degerini yeniden uretebiliyor muyuz? --
# Tutuyorsa: veri bolmesi + konfig + maske + olcum yolu, D ile AYNI hizada.
# Tutmuyorsa: bu defterle D3'u D ile kiyaslamak ANLAMSIZ -> devam etme.
_dk, _dm = REF["D"]
_eg = f"{_dk}/egri_A_s0.json"
assert os.path.exists(_eg), f"D'nin egrisi yok: {_eg} (deney7 Drive'da mi?)"
_E = {r["step"]: r for r in json.load(open(_eg))}
_sn = snaplar(_dk)
_ad = [a for a in (70000, 75000, 65000) if a in _sn and a in _E][:1]
assert _ad, "D'nin 65-75 bin anlik goruntusu bulunamadi"
_a = _ad[0]
_ml = olc_snap(_sn[_a], _dm)          # maske ACIK  (D'nin egitim yapilandirmasi)
_mk = olc_snap(_sn[_a], ())           # maske KAPALI (yanlis yapilandirma - karsilastirma)
_fark = abs(_ml["ENT"] - _E[_a]["ent"])
print(f"\nDENETIM 4  D @ {_a}   egri ent = {_E[_a]['ent']:.3f}")
print(f"   maske ACIK  -> ENT {_ml['ENT']:.3f}  (fark {_fark:.3f})   <- dogru olan")
print(f"   maske KAPALI-> ENT {_mk['ENT']:.3f}  (fark {abs(_mk['ENT']-_E[_a]['ent']):.3f})")
assert _fark < 0.03, ("Olcum hatti D'nin egrisini yeniden uretemiyor. "
                      "Veri bolmesi / konfig / offsetler uyusmuyor - kiyas ANLAMSIZ.")
print("DENETIM 4 GECTI   olcum hatti D ile ayni hizada (veri+konfig+offset)")

# --- DENETIM 5: EGITIMIN MASKESI -> davranistan DEGIL, KAYITLI KONFIGDEN --
# DENETIM 4 maskeyi AYIRT EDEMEZ: D'de maskeli 0.157 / maskesiz 0.150, ikisi de
# egrinin 0.03 yakininda. Cunku D kisayolu zaten kullanmiyor. Davranissal
# kontrol bu isi goremez; surdur_*.pt icindeki cfg gorur.
_sp = glob.glob(f"{_dk}/surdur_*.pt")
assert _sp, "D'nin surdurme paketi yok - konfig kapisi SINANAMIYOR"
_c = torch.load(_sp[0], map_location="cpu", weights_only=False).get("cfg", {})
assert _c.get("MASK_KEY") == "1" and tuple(_c.get("MASK_BLK") or ()) == _dm, \
    f"Konfig kapisi bozuk: MASK_KEY={_c.get('MASK_KEY')!r} MASK_BLK={_c.get('MASK_BLK')!r}"
print(f"DENETIM 5 GECTI   kayitli-konfig kapisi calisiyor "
      f"(D: MASK_KEY={_c['MASK_KEY']!r} MASK_BLK={_c['MASK_BLK']})")
