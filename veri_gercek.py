# -*- coding: utf-8 -*-
"""VERI_GERCEK — GERCEK DUNYA seklinde tipli bilgi grafigi + kendi sozlugumuz.

NEDEN DEGISTI (14 Eylul): ilk surum 10.000 atomik olgu hedefliyordu ve bunu
tutturmak icin 1.200 ulke uyduruyordu (`ulke_431`). Okunmuyordu. Dunyada
~156 ulke var; olgu sayisini degil DUNYAYI esas aldik.

    GERCEK olan  : ulke adi, baskent, ana dil, sehir adlari, dil ailesi
    URETILMIS    : komsuluk (kita icinde cografi siraya gore komsu secilir),
                   bolge adlari (<Ulke>_Kuzey/_Guney/_Bati), bolge-sehir
                   eslesmesi, listede olmayan sehirler (<Ulke>_Sehir3)

Olgu sayisi ~5.700; egitim hacmini olgu sayisi DEGIL 2-adimli zincir sayisi
ve phi belirliyor (phi = egitim 2-adimlisi / atomik olgu). Zincir sayisi
kombinatorik oldugu icin 10.000'i zaten asiyor.

GOMULU KONTROL: graf TIPLI oldugu icin kisayol f(e,r2) HER ZAMAN cevap
uretemez:

    Turkiye komsusu baskenti ?  -> Yunanistan -> Atina
        KISAYOL "Turkiye baskenti" = Ankara        VAR   (model bunu ogrenir)
    Izmir bolgesi merkezi ?     -> Turkiye_Bati -> ...
        KISAYOL "Izmir merkezi"                    YOK   (sehrin merkezi olmaz)

Ayni modelde kisayolu MUMKUN olan ve OLMAYAN zincirleri ayri olceriz.

Sozluk kelime duzeyinde: her varlik ve iliski TEK token. Dil bilgisi yok.

    python veri_gercek.py [--dok cikti.txt]
"""
import argparse
import io

# ===================================================================== DUNYA
# (ulke, baskent, ana dil).  SIRA COGRAFI -- komsuluk bu siradan uretilir.
# Kita bir ILISKI DEGIL: 5 degerli bir cevap uzayi, 780 degerli SEHIR ile
# ayni modelde kiyaslanamaz sayi uretir. Sadece komsu siralamasi icin var.
KITALAR = {
"AVRUPA": [
    ("Izlanda", "Reykjavik", "izlandaca"),
    ("Irlanda", "Dublin", "irlandaca"),
    ("Birlesik_Krallik", "Londra", "ingilizce"),
    ("Fransa", "Paris", "fransizca"),
    ("Belcika", "Bruksel", "felemenkce"),
    ("Hollanda", "Amsterdam", "felemenkce"),
    ("Luksemburg", "Luksemburg_Sehri", "fransizca"),
    ("Almanya", "Berlin", "almanca"),
    ("Danimarka", "Kopenhag", "danca"),
    ("Norvec", "Oslo", "norvecce"),
    ("Isvec", "Stokholm", "isvecce"),
    ("Finlandiya", "Helsinki", "fince"),
    ("Estonya", "Tallinn", "estonca"),
    ("Letonya", "Riga", "letonca"),
    ("Litvanya", "Vilnius", "litvanca"),
    ("Belarus", "Minsk", "belarusca"),
    ("Rusya", "Moskova", "rusca"),
    ("Ukrayna", "Kiev", "ukraynaca"),
    ("Moldova", "Kisinev", "romence"),
    ("Romanya", "Bukres", "romence"),
    ("Bulgaristan", "Sofya", "bulgarca"),
    ("Turkiye", "Ankara", "turkce"),
    ("Kibris", "Lefkosa", "turkce"),
    ("Yunanistan", "Atina", "yunanca"),
    ("Kuzey_Makedonya", "Uskup", "makedonca"),
    ("Arnavutluk", "Tiran", "arnavutca"),
    ("Karadag", "Podgorica", "karadagca"),
    ("Bosna", "Saraybosna", "bosnakca"),
    ("Sirbistan", "Belgrad", "sirpca"),
    ("Hirvatistan", "Zagreb", "hirvatca"),
    ("Slovenya", "Lyublyana", "slovence"),
    ("Macaristan", "Budapeste", "macarca"),
    ("Slovakya", "Bratislava", "slovakca"),
    ("Cekya", "Prag", "cekce"),
    ("Polonya", "Varsova", "lehce"),
    ("Avusturya", "Viyana", "almanca"),
    ("Isvicre", "Bern", "almanca"),
    ("Italya", "Roma", "italyanca"),
    ("Malta", "Valletta", "maltaca"),
    ("Ispanya", "Madrid", "ispanyolca"),
    ("Portekiz", "Lizbon", "portekizce"),
],
"ASYA": [
    ("Gurcistan", "Tiflis", "gurcuce"),
    ("Ermenistan", "Erivan", "ermenice"),
    ("Azerbaycan", "Baku", "azerice"),
    ("Iran", "Tahran", "farsca"),
    ("Irak", "Bagdat", "arapca"),
    ("Suriye", "Sam", "arapca"),
    ("Lubnan", "Beyrut", "arapca"),
    ("Israil", "Kudus", "ibranice"),
    ("Urdun", "Amman", "arapca"),
    ("Suudi_Arabistan", "Riyad", "arapca"),
    ("Kuveyt", "Kuveyt_Sehri", "arapca"),
    ("Bahreyn", "Manama", "arapca"),
    ("Katar", "Doha", "arapca"),
    ("BAE", "Abu_Dabi", "arapca"),
    ("Umman", "Maskat", "arapca"),
    ("Yemen", "Sana", "arapca"),
    ("Turkmenistan", "Asgabat", "turkmence"),
    ("Ozbekistan", "Taskent", "ozbekce"),
    ("Kazakistan", "Astana", "kazakca"),
    ("Kirgizistan", "Biskek", "kirgizca"),
    ("Tacikistan", "Dusanbe", "tacikce"),
    ("Afganistan", "Kabil", "pestuca"),
    ("Pakistan", "Islamabad", "urduca"),
    ("Hindistan", "Delhi", "hintce"),
    ("Nepal", "Katmandu", "nepalce"),
    ("Butan", "Timpu", "dzongka"),
    ("Banglades", "Dakka", "bengalce"),
    ("Myanmar", "Naypyidaw", "birmanca"),
    ("Sri_Lanka", "Kolombo", "sinhalaca"),
    ("Tayland", "Bangkok", "tayca"),
    ("Laos", "Vientiane", "laosca"),
    ("Kambocya", "Pnom_Penh", "kmerce"),
    ("Vietnam", "Hanoi", "vietnamca"),
    ("Cin", "Pekin", "cince"),
    ("Mogolistan", "Ulanbator", "mogolca"),
    ("Kuzey_Kore", "Pyongyang", "korece"),
    ("Guney_Kore", "Seul", "korece"),
    ("Japonya", "Tokyo", "japonca"),
    ("Filipinler", "Manila", "filipince"),
    ("Malezya", "Kuala_Lumpur", "malayca"),
    ("Singapur", "Singapur_Sehri", "malayca"),
    ("Endonezya", "Cakarta", "endonezce"),
],
"AFRIKA": [
    ("Fas", "Rabat", "arapca"),
    ("Cezayir", "Cezayir_Sehri", "arapca"),
    ("Tunus", "Tunus_Sehri", "arapca"),
    ("Libya", "Trablus", "arapca"),
    ("Misir", "Kahire", "arapca"),
    ("Sudan", "Hartum", "arapca"),
    ("Guney_Sudan", "Cuba", "ingilizce"),
    ("Etiyopya", "Addis_Ababa", "amharca"),
    ("Eritre", "Asmara", "tigrinyaca"),
    ("Cibuti", "Cibuti_Sehri", "fransizca"),
    ("Somali", "Mogadisu", "somalice"),
    ("Kenya", "Nairobi", "svahilice"),
    ("Uganda", "Kampala", "ingilizce"),
    ("Ruanda", "Kigali", "kinyarvandaca"),
    ("Burundi", "Gitega", "kirundice"),
    ("Tanzanya", "Dodoma", "svahilice"),
    ("Kongo", "Kinsasa", "fransizca"),
    ("Zambiya", "Lusaka", "ingilizce"),
    ("Malavi", "Lilongve", "ingilizce"),
    ("Mozambik", "Maputo", "portekizce"),
    ("Zimbabve", "Harare", "ingilizce"),
    ("Botsvana", "Gaborone", "ingilizce"),
    ("Namibya", "Vindhuk", "ingilizce"),
    ("Guney_Afrika", "Pretorya", "ingilizce"),
    ("Madagaskar", "Antananarivo", "malgasca"),
    ("Angola", "Luanda", "portekizce"),
    ("Gabon", "Librevil", "fransizca"),
    ("Kamerun", "Yaunde", "fransizca"),
    ("Cad", "Encemine", "arapca"),
    ("Nijer", "Niyamey", "fransizca"),
    ("Nijerya", "Abuja", "ingilizce"),
    ("Benin", "Porto_Novo", "fransizca"),
    ("Togo", "Lome", "fransizca"),
    ("Gana", "Akra", "ingilizce"),
    ("Fildisi", "Yamusukro", "fransizca"),
    ("Burkina_Faso", "Uagadugu", "fransizca"),
    ("Mali", "Bamako", "fransizca"),
    ("Senegal", "Dakar", "fransizca"),
    ("Gine", "Konakri", "fransizca"),
    ("Sierra_Leone", "Freetown", "ingilizce"),
    ("Liberya", "Monrovia", "ingilizce"),
    ("Moritanya", "Nuaksot", "arapca"),
],
"AMERIKA": [
    ("Kanada", "Ottava", "ingilizce"),
    ("ABD", "Vasington", "ingilizce"),
    ("Meksika", "Meksiko", "ispanyolca"),
    ("Guatemala", "Guatemala_Sehri", "ispanyolca"),
    ("El_Salvador", "San_Salvador", "ispanyolca"),
    ("Honduras", "Tegucigalpa", "ispanyolca"),
    ("Nikaragua", "Managua", "ispanyolca"),
    ("Kosta_Rika", "San_Hose", "ispanyolca"),
    ("Panama", "Panama_Sehri", "ispanyolca"),
    ("Kuba", "Havana", "ispanyolca"),
    ("Jamaika", "Kingston", "ingilizce"),
    ("Haiti", "Port_o_Prens", "fransizca"),
    ("Dominik", "Santo_Domingo", "ispanyolca"),
    ("Kolombiya", "Bogota", "ispanyolca"),
    ("Venezuela", "Karakas", "ispanyolca"),
    ("Guyana", "Corctaun", "ingilizce"),
    ("Surinam", "Paramaribo", "felemenkce"),
    ("Brezilya", "Brazilya", "portekizce"),
    ("Ekvador", "Kito", "ispanyolca"),
    ("Peru", "Lima", "ispanyolca"),
    ("Bolivya", "La_Paz", "ispanyolca"),
    ("Paraguay", "Asuncion", "ispanyolca"),
    ("Uruguay", "Montevideo", "ispanyolca"),
    ("Arjantin", "Buenos_Aires", "ispanyolca"),
    ("Sili", "Santiago", "ispanyolca"),
],
"OKYANUSYA": [
    ("Avustralya", "Kanberra", "ingilizce"),
    ("Papua", "Port_Moresby", "ingilizce"),
    ("Solomon", "Honiara", "ingilizce"),
    ("Vanuatu", "Port_Vila", "fransizca"),
    ("Fiji", "Suva", "ingilizce"),
    ("Yeni_Zelanda", "Vellington", "ingilizce"),
],
}

# Baskent DISINDAKI gercek sehirler. Listede olmayan ulkeler icin
# <Ulke>_Sehir2.. uretilir -- uydurma oldugu ADINDAN belli olsun diye.
EK_SEHIR = {
"Turkiye": ["Istanbul", "Izmir", "Bursa", "Antalya"],
"Fransa": ["Marsilya", "Lyon", "Tuluz", "Nis"],
"Almanya": ["Hamburg", "Munih", "Koln", "Frankfurt"],
"Italya": ["Milano", "Napoli", "Torino", "Floransa"],
"Ispanya": ["Barselona", "Valensiya", "Sevilla", "Bilbao"],
"Portekiz": ["Porto", "Braga", "Koimbra", "Faro"],
"Birlesik_Krallik": ["Mancester", "Birmingem", "Glaskov", "Liverpul"],
"Hollanda": ["Rotterdam", "Lahey", "Utrecht", "Eindhoven"],
"Belcika": ["Anvers", "Gent", "Liej", "Bruj"],
"Isvicre": ["Zurih", "Cenevre", "Basel", "Lozan"],
"Avusturya": ["Graz", "Linz", "Salzburg", "Innsbruck"],
"Polonya": ["Krakov", "Lodz", "Vroclav", "Poznan"],
"Cekya": ["Brno", "Ostrava", "Plzen", "Liberec"],
"Macaristan": ["Debrecen", "Szeged", "Miskolc", "Pecs"],
"Romanya": ["Kluj", "Timisoara", "Yas", "Kostanca"],
"Bulgaristan": ["Plovdiv", "Varna", "Burgaz", "Ruse"],
"Yunanistan": ["Selanik", "Patras", "Larisa", "Heraklion"],
"Sirbistan": ["Novi_Sad", "Nis_Sehri", "Kragujevac", "Subotica"],
"Hirvatistan": ["Split", "Rijeka", "Osijek", "Zadar"],
"Ukrayna": ["Harkiv", "Odesa", "Dnipro", "Lviv"],
"Rusya": ["Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan"],
"Isvec": ["Goteborg", "Malmo", "Uppsala", "Vasteras"],
"Norvec": ["Bergen", "Trondheim", "Stavanger", "Drammen"],
"Danimarka": ["Aarhus", "Odense", "Aalborg", "Esbjerg"],
"Finlandiya": ["Tampere", "Turku", "Oulu", "Espoo"],
"Irlanda": ["Kork", "Limerik", "Galway", "Vaterford"],
"Cin": ["Sanghay", "Kanton", "Sencen", "Cengdu"],
"Japonya": ["Osaka", "Nagoya", "Sapporo", "Fukuoka"],
"Guney_Kore": ["Busan", "Incheon", "Daegu", "Gwangju"],
"Hindistan": ["Mumbai", "Bangalor", "Kalkuta", "Cennai"],
"Pakistan": ["Karaci", "Lahor", "Faysalabad", "Ravalpindi"],
"Banglades": ["Citagong", "Kulna", "Rajsahi", "Silhet"],
"Iran": ["Meshed", "Isfahan", "Sitaz", "Tebriz"],
"Irak": ["Musul", "Basra", "Erbil", "Kerkuk"],
"Suriye": ["Halep", "Humus", "Lazkiye", "Hama"],
"Suudi_Arabistan": ["Cidde", "Mekke", "Medine", "Dammam"],
"BAE": ["Dubai", "Sarja", "Acman", "Al_Ayn"],
"Kazakistan": ["Almati", "Simkent", "Karaganda", "Aktobe"],
"Ozbekistan": ["Semerkant", "Buhara", "Namangan", "Andican"],
"Azerbaycan": ["Gence", "Sumgayit", "Mingecevir", "Lenkeran"],
"Tayland": ["Cangmay", "Pattaya", "Puket", "Hatyay"],
"Vietnam": ["Saygon", "Hayfong", "Danang", "Kanto"],
"Endonezya": ["Surabaya", "Bandung", "Medan", "Semarang"],
"Malezya": ["Penang", "Johor", "Ipoh", "Malaka"],
"Filipinler": ["Kezon", "Davao", "Sebu", "Makati"],
"Misir": ["Iskenderiye", "Giza", "Port_Said", "Suveys"],
"Fas": ["Kazablanka", "Marakes", "Fes", "Tanca"],
"Cezayir": ["Vahran", "Konstantin", "Annaba", "Blida"],
"Nijerya": ["Lagos", "Kano", "Ibadan", "Port_Harcourt"],
"Guney_Afrika": ["Johannesburg", "Kapkent", "Durban", "Soweto"],
"Kenya": ["Mombasa", "Kisumu", "Nakuru", "Eldoret"],
"Etiyopya": ["Dire_Dava", "Mekelle", "Gondar", "Avasa"],
"ABD": ["Nevyork", "Sikago", "Los_Angeles", "Hyuston"],
"Kanada": ["Toronto", "Montreal", "Vankuver", "Kalgari"],
"Meksika": ["Guadalahara", "Monterrey", "Puebla", "Tihuana"],
"Brezilya": ["Sao_Paulo", "Rio", "Salvador", "Fortaleza"],
"Arjantin": ["Kordoba", "Rosario", "Mendoza", "La_Plata"],
"Kolombiya": ["Medellin", "Kali", "Barrankilla", "Kartagena"],
"Peru": ["Arekipa", "Trujillo", "Cusco", "Piura"],
"Sili": ["Valparaiso", "Konsepsiyon", "Antofagasta", "Temuko"],
"Avustralya": ["Sidney", "Melburn", "Brisbane", "Perth"],
"Yeni_Zelanda": ["Oklend", "Kraycurc", "Hamilton", "Dunedin"],
}

# Dil ailesi. Iki kademe: dil -> alt aile -> ust aile. Ust aileler
# (hintavrupaca, altayca, ...) CEVAP-ONLY: kendi kokenleri yok. 10 tane,
# sozlugun %0,7'si. Ilk tasarimda sayisal yapraklar sozlugun YARISIYDI;
# fark bu.
KOKEN = {
 "fransizca": "latince", "italyanca": "latince", "ispanyolca": "latince",
 "portekizce": "latince", "romence": "latince",
 "ingilizce": "cermence", "almanca": "cermence", "felemenkce": "cermence",
 "isvecce": "cermence", "norvecce": "cermence", "danca": "cermence",
 "izlandaca": "cermence",
 "rusca": "slavca", "ukraynaca": "slavca", "belarusca": "slavca",
 "lehce": "slavca", "cekce": "slavca", "slovakca": "slavca",
 "bulgarca": "slavca", "sirpca": "slavca", "hirvatca": "slavca",
 "bosnakca": "slavca", "slovence": "slavca", "makedonca": "slavca",
 "karadagca": "slavca",
 "litvanca": "baltikca", "letonca": "baltikca",
 "irlandaca": "keltce",
 "yunanca": "hintavrupaca", "arnavutca": "hintavrupaca",
 "ermenice": "hintavrupaca",
 "farsca": "iranice", "pestuca": "iranice", "tacikce": "iranice",
 "hintce": "hintarya", "urduca": "hintarya", "bengalce": "hintarya",
 "sinhalaca": "hintarya", "nepalce": "hintarya",
 "turkce": "ortakturkce", "azerice": "ortakturkce", "kazakca": "ortakturkce",
 "ozbekce": "ortakturkce", "turkmence": "ortakturkce",
 "kirgizca": "ortakturkce",
 "fince": "finugorca", "estonca": "finugorca", "macarca": "finugorca",
 "arapca": "samice", "ibranice": "samice", "amharca": "samice",
 "tigrinyaca": "samice", "maltaca": "samice",
 "somalice": "kusitce",
 "svahilice": "bantuca", "kinyarvandaca": "bantuca", "kirundice": "bantuca",
 "cince": "cintibetce", "birmanca": "cintibetce", "dzongka": "cintibetce",
 "tayca": "taykadaice", "laosca": "taykadaice",
 "vietnamca": "avustroasyatik", "kmerce": "avustroasyatik",
 "malayca": "avustronezyaca", "endonezce": "avustronezyaca",
 "filipince": "avustronezyaca", "malgasca": "avustronezyaca",
 "gurcuce": "kafkasca",
 "japonca": "altayca", "korece": "altayca", "mogolca": "altayca",
 # alt aile -> ust aile
 "latince": "hintavrupaca", "cermence": "hintavrupaca",
 "slavca": "hintavrupaca", "baltikca": "hintavrupaca",
 "keltce": "hintavrupaca", "iranice": "hintavrupaca",
 "hintarya": "hintavrupaca",
 "ortakturkce": "altayca",
 "finugorca": "uralca",
 "samice": "afroasyatik", "kusitce": "afroasyatik",
 "bantuca": "nijerkongo",
}

# ==================================================================== SEMA
# iliski -> {kaynak tipi: hedef tipi}.  Ayni iliski FARKLI tiplerden
# cikabilir (ulkenin de komsusu var, sehrin de). Bu ortusme olmadan
# kisayollu zincir yok denecek kadar azalir ve gomulu kontrol calismaz.
SEMA = {
    "baskenti":   {"ULKE": "SEHIR"},
    "buyuksehri": {"ULKE": "SEHIR", "BOLGE": "SEHIR"},
    "komsusu":    {"ULKE": "ULKE", "SEHIR": "SEHIR"},
    "dili":       {"ULKE": "DIL"},
    "ulkesi":     {"SEHIR": "ULKE"},
    "bolgesi":    {"ULKE": "BOLGE", "SEHIR": "BOLGE"},
    "merkezi":    {"BOLGE": "SEHIR"},
    "yanbolgesi": {"BOLGE": "BOLGE"},
    "kokeni":     {"DIL": "DIL"},
}
ILISKI = list(SEMA)
TIPLER = ["ULKE", "SEHIR", "BOLGE", "DIL"]
YON = ["Kuzey", "Guney"]
SEHIR_SAY = 5          # baskent dahil


def _komsu(liste):
    """Sirali listede bitisik olan. Son eleman bir ONCEKINE baglanir --
    dairesel kapatmak Endonezya'yi Gurcistan'a komsu yapiyordu."""
    n = len(liste)
    return {liste[i]: liste[i + 1 if i < n - 1 else i - 1] for i in range(n)}


def kur():
    ulkeler = [u for k in KITALAR for u in KITALAR[k]]
    ad_ulke = [u[0] for u in ulkeler]
    baskent = {u[0]: u[1] for u in ulkeler}
    dil_of = {u[0]: u[2] for u in ulkeler}

    # --- sehirler: baskent + gercek ekler + gerekiyorsa uretilmis ---------
    sehir = {}
    for u in ad_ulke:
        s = [baskent[u]] + list(EK_SEHIR.get(u, []))[:SEHIR_SAY - 1]
        while len(s) < SEHIR_SAY:
            s.append(f"{u}_Sehir{len(s) + 1}")
        sehir[u] = s

    bolge = {u: [f"{u}_{y}" for y in YON] for u in ad_ulke}

    # --- diller: ulke dilleri + aileler ----------------------------------
    diller = []
    for d in list(dil_of.values()) + list(KOKEN) + list(KOKEN.values()):
        if d not in diller:
            diller.append(d)
    eksik = [d for d in dil_of.values() if d not in KOKEN]
    assert not eksik, f"KOKEN eksik: {eksik}"

    ad = {"ULKE": ad_ulke,
          "SEHIR": [c for u in ad_ulke for c in sehir[u]],
          "BOLGE": [b for u in ad_ulke for b in bolge[u]],
          "DIL": diller}

    # Ayni ad iki varliga verilemez: token cakisir, sessizce yanlis olcer.
    hepsi = [a for t in TIPLER for a in ad[t]]
    assert len(hepsi) == len(set(hepsi)),         "TEKRAR EDEN AD: " + str([a for a in set(hepsi) if hepsi.count(a) > 1])

    # --- OLGULAR ----------------------------------------------------------
    olgu = {}
    uk = {}
    for kita in KITALAR:          # komsuluk KITA ICINDE, cografi siradan
        uk.update(_komsu([u[0] for u in KITALAR[kita]]))

    for u in ad_ulke:
        olgu[(u, "baskenti")] = baskent[u]
        olgu[(u, "buyuksehri")] = sehir[u][1]
        olgu[(u, "komsusu")] = uk[u]
        olgu[(u, "dili")] = dil_of[u]
        # ULKE'nin bolgesi SON bolge: ilki olsaydi 'ULKE bolgesi buyuksehri'
        # ile kisayol 'ULKE buyuksehri' AYNI sehre cikardi.
        olgu[(u, "bolgesi")] = bolge[u][-1]
        # Sehirler bolgelere BLOK halinde dagilir; her bolgede en az 2 sehir
        # olmali, yoksa 'merkezi' ile 'buyuksehri' ayni sehre cikar.
        pay = [[] for _ in YON]
        for i, c in enumerate(sehir[u]):
            pay[min(i * len(YON) // SEHIR_SAY, len(YON) - 1)].append(c)
        sk = _komsu(sehir[u])
        bk = _komsu(bolge[u])
        for j, b in enumerate(bolge[u]):
            assert len(pay[j]) >= 2, (u, j, pay)
            for c in pay[j]:
                olgu[(c, "ulkesi")] = u
                olgu[(c, "bolgesi")] = b
                olgu[(c, "komsusu")] = sk[c]
            olgu[(b, "merkezi")] = pay[j][0]
            olgu[(b, "buyuksehri")] = pay[j][1]
            olgu[(b, "yanbolgesi")] = bk[b]
    for d in diller:
        if d in KOKEN:
            olgu[(d, "kokeni")] = KOKEN[d]

    ozel = ["<pad>", "<soru>", "?", "<son>"]
    sozluk = ozel + ILISKI + hepsi
    return dict(ad=ad, n={t: len(ad[t]) for t in TIPLER}, sozluk=sozluk,
                kim={s: i for i, s in enumerate(sozluk)},
                tip={a: t for t in TIPLER for a in ad[t]},
                olgu=olgu, sema=SEMA, iliski=ILISKI, baskent=baskent)


def zincirler(G):
    """Tip olarak GECERLI 2 adimli zincirler, UC SINIFA ayrilmis.

    AYIRT : kisayol hedefi VAR ve CEVAPTAN FARKLI -> modelin kisayola
            saptigini goreBILDIGIMIZ tek sinif. Olcum bunun uzerinde yapilir.
    YOK   : kisayol tip olarak IMKANSIZ -> gomulu kontrol.
    AYNI  : kisayol hedefi var ama CEVABIN KENDISI. Model kisayolu alsa da
            dogru cevap verir; hata ile dogruyu AYIRT EDEMEYIZ. Egitimde
            kullanilabilir, SINAVDA KULLANILAMAZ.

    Eski rastgele grafta AYNI sinifi neredeyse hic olusmuyordu; gercek
    dunyada olusuyor cunku iliskiler BAGIMLI ('Izmir dili' ile
    'Izmir ulkesi dili' ayni sey). Olcup ayirmak zorundayiz.
    """
    out = []
    for (e, r1), b in G["olgu"].items():
        for r2 in G["iliski"]:
            if r2 == r1:
                # r1 == r2 ise KISAYOL ile KOPRU AYNI varliga cikar.
                continue
            if G["tip"][b] not in G["sema"][r2]:       # ikinci hop gecersiz
                continue
            cev = G["olgu"][(b, r2)]
            ks = G["olgu"].get((e, r2))
            # DONUS once bakilir: cevap SORULAN VARLIGIN KENDISI ise model
            # "varligi kopyala" kuralini ogrenip dogru cikabilir; ne koprüyü
            # ne kisayolu olcer. Hepsi YOK sinifina dusuyordu ve kontrolun
            # %26'sini kirletiyordu (olculdu: 1.872/7.332).
            sinif = ("DONUS" if cev == e else
                     "YOK" if ks is None else
                     "AYNI" if ks == cev else "AYIRT")
            out.append((e, r1, r2, b, cev, ks, sinif))
    return out


def yaz(G, z, f=print):
    S = {k: [x for x in z if x[6] == k]
         for k in ("AYIRT", "YOK", "AYNI", "DONUS")}
    f("=" * 76)
    f("GERCEK DUNYA SEKLINDE TIPLI BILGI GRAFIGI")
    f("=" * 76)
    f("  tip buyuklukleri : " + "  ".join(f"{k}={v}" for k, v in G["n"].items()))
    f(f"  SOZLUK           : {len(G['sozluk'])} token  "
      f"({len(G['iliski'])} iliski + {sum(G['n'].values())} varlik + 4 ozel)")
    f(f"  ATOMIK OLGU      : {len(G['olgu'])}")
    f(f"  2-ADIMLI ZINCIR  : {len(z)}")
    for k, ac in (("AYIRT", "kisayol VAR, cevaptan FARKLI -> SINAV burada"),
                  ("YOK",   "kisayol IMKANSIZ           -> gomulu kontrol"),
                  ("AYNI",  "kisayol = cevap            -> sinavda KULLANILMAZ"),
                  ("DONUS", "cevap = sorulan varlik     -> sinavda KULLANILMAZ")):
        f(f"     {k:6s} {len(S[k]):6d}  ({100*len(S[k])/len(z):2.0f}%)   {ac}")

    f("\n--- SEMA ---")
    for r, m in G["sema"].items():
        f(f"   {r:12s} " + ",  ".join(f"{k}->{v}" for k, v in m.items()))

    f("\n--- ZINCIR TURLERI ---")
    tur = {}
    for e, r1, r2, b, a, ks, s in z:
        d = tur.setdefault((r1, r2),
                           {"AYIRT": 0, "YOK": 0, "AYNI": 0, "DONUS": 0})
        d[s] += 1
    f(f"   {'r1':12s} {'r2':12s} {'AYIRT':>7s} {'YOK':>7s} {'AYNI':>7s}"
      f" {'DONUS':>7s}")
    for (r1, r2), d in sorted(tur.items()):
        f(f"   {r1:12s} {r2:12s} {d['AYIRT']:7d} {d['YOK']:7d} {d['AYNI']:7d}"
          f" {d['DONUS']:7d}")

    f("\n--- TURKIYE'NIN BUTUN OLGULARI ---")
    trs = set(["Turkiye"]) | set(G["ad"]["SEHIR"][:0])
    for (e, r), h in G["olgu"].items():
        if e == "Turkiye" or e.startswith("Turkiye_") or e in \
                ("Ankara", "Istanbul", "Izmir", "Bursa", "Antalya"):
            f(f"   {e:16s} {r:12s} {h}")

    ornek = {"AYIRT": "model kisayola saparsa YAKALARIZ",
             "YOK": "kisayol imkansiz (tip izin vermiyor)",
             "AYNI": "kisayol dogru cevabi veriyor -- sinavda kullanilamaz",
             "DONUS": "cevap sorulan varligin kendisi -- kopyalamak yetiyor"}
    for k in ("AYIRT", "YOK", "AYNI", "DONUS"):
        f(f"\n--- ZINCIR / {k}  ({ornek[k]}) ---")
        for e, r1, r2, b, a, ks, _s in S[k][:8]:
            f(f"   {e} {r1} {r2} ?   ->  {a}")
            f(f"       kopru: {b:18s} |  KISAYOL '{e} {r2}' = {ks or 'YOK'}")

    f("\n--- TOKEN DIZILIMI (kelime duzeyi, her varlik TEK token) ---")
    for e, r1, r2, b, a, ks, _s in S["AYIRT"][:3]:
        d = ["<soru>", e, r1, r2, "?", a, "<son>"]
        f(f"   {' '.join(d)}")
        f(f"       -> {[G['kim'][t] for t in d]}")

    ur = [a for a in G["ad"]["SEHIR"] if "_Sehir" in a]
    f("\n--- URETILMIS (gercek olmayan) KISIMLAR ---")
    f("   komsuluk : kita ici cografi siradan uretildi, gercek sinir DEGIL")
    f(f"   bolgeler : {len(G['ad']['BOLGE'])} adet, <Ulke>_Kuzey / <Ulke>_Guney")
    f(f"   sehirler : {len(ur)}/{len(G['ad']['SEHIR'])} adet uretilmis "
      f"(<Ulke>_SehirN); {len(G['ad']['SEHIR'])-len(ur)} tanesi GERCEK")
    f("   GERCEK   : ulke, baskent, ana dil, sehir adlari, dil ailesi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dok", default="")
    a = ap.parse_args()
    G = kur()
    z = zincirler(G)
    yaz(G, z)
    if a.dok:
        with io.open(a.dok, "w", encoding="utf-8", newline="") as fh:
            def f(s=""):
                fh.write(str(s) + "\n")
            yaz(G, z, f)
            f("\n\n" + "=" * 76)
            f("TUM ATOMIK OLGULAR")
            f("=" * 76)
            for (e, r), h in G["olgu"].items():
                f(f"{e}\t{r}\t{h}")
            f("\n\n" + "=" * 76)
            f("TUM 2-ADIMLI ZINCIRLER  (sinif / soru / kopru / cevap / kisayol)")
            f("=" * 76)
            for e, r1, r2, b, ans, ks, s in z:
                f(f"{s:5s}\t{e} {r1} {r2} ?\t-> {ans}\tkopru={b}"
                  f"\tkisayol={ks or 'YOK'}")
        print(f"\n-> tam dokum: {a.dok}")


if __name__ == "__main__":
    main()
