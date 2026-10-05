"""Text normalisation for business names and addresses.

All functions are pure-Python / regex so they work for any country label
(US, India, France, or anything unseen). Nothing here is country-specific
except optional canonicalisation dictionaries for common state/region names.
"""
import re
import unicodedata

from translit import phonetic
from translit import to_ascii as _translit

# Tokens that denote missing values inside address strings.
NULL_TOKENS = {"null", "none", "nan", "na", "n/a"}

# Legal-form / company-type words. They are separated from the "core" name,
# because sources add, drop or reorder them freely.
LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "pvt", "private", "plc", "llp", "lp", "pc", "pllc", "pa", "sarl",
    "sas", "sasu", "sa", "sci", "eurl", "snc", "scop", "ste", "societe", "gmbh",
    "ag", "bv", "nv", "the", "ms", "m/s", "and", "of", "de", "du", "des", "la",
    "le", "les", "et",
    # more French legal forms
    "ei", "eirl", "sca", "scs", "gie", "selarl", "sel", "scp", "scm", "sprl", "l", "d",
    # "Pvt. Ltd." / "LLP" written in Indian scripts, after transliteration
    "pra", "li", "elelpi", "elalpi", "ellpi", "elaelpi",
}
# Legal words recognised by sound after transliteration (piraivet, praivet, limitet, limittad ...).
LEGAL_SOUND = {"prvt", "prpt", "lmt", "lmtt", "elp"}  # elp: elelpi / elelapi (LLP)

# Name abbreviations -> canonical form.
NAME_ABBR = {
    "corpn": "corporation", "intl": "international", "mfg": "manufacturing",
    "svc": "service", "svcs": "services", "tech": "technologies",
    "technology": "technologies", "ent": "enterprises", "entp": "enterprises",
    "bros": "brothers", "assoc": "associates", "mgmt": "management",
    "dept": "department", "ctr": "center", "cter": "center", "centre": "center",
    "grp": "group", "hldgs": "holdings", "ind": "industries", "inds": "industries",
    "natl": "national", "univ": "university", "hosp": "hospital",
    # French
    "cie": "compagnie", "ets": "etablissements", "etabl": "etablissements", "etab": "etablissements",
    "asso": "association", "fed": "federation", "gpe": "groupe", "st": "saint",
    "sts": "saints", "dvpt": "developpement",
    "dev": "developpement", "sce": "services", "svce": "services",
}

# Address abbreviations -> canonical form. "saint" maps to "st" because one
# source systematically replaces "St"(reet) with "Saint".
ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "ste": "st", "road": "rd",
    "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "plaza": "plz", "highway": "hwy",
    "parkway": "pkwy", "circle": "cir", "terrace": "ter", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt",
    "building": "bldg", "floor": "flr", "fl": "flr", "suite": "ste",
    "number": "no", "nagar": "ngr", "sector": "sec", "near": "nr", "opp": "opposite",
    "opposite": "opposite", "chemin": "ch", "allee": "all", "impasse": "imp",
    "route": "rte", "rue": "r", "township": "twp", "city": "", "town": "",
    "cove": "cv", "trail": "trl", "mount": "mt", "fort": "ft", "point": "pt",
    # French street types, written in full or abbreviated
    "sainte": "st", "allees": "all", "chem": "ch", "rt": "rte", "cours": "crs", "cour": "crs",
    "quai": "q", "qu": "q", "passage": "pass", "faubourg": "fg", "fbg": "fg", "residence": "res",
    "resid": "res", "batiment": "bldg", "bat": "bldg", "bld": "blvd", "bis": "",
    "cedex": "", "docteur": "dr", "general": "gen", "gal": "gen", "marechal": "mal",
    "president": "pdt", "lieu": "", "dit": "", "lieudit": "", "hameau": "ham", "zone": "z",
    "industrielle": "ind", "zi": "z ind", "za": "z", "zac": "z", "rdc": "", "rez": "", "chaussee": "",
    # articles and prepositions that sources add or drop freely
    "de": "", "du": "", "des": "", "la": "", "le": "", "les": "", "l": "", "d": "", "et": "", "of": "",
    "the": "",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}

# India states: Latin names, common abbreviations and native-script names.
IN_STATES = {
    "andhra pradesh": "ap", "ఆంధ్రప్రదేశ్": "ap", "telangana": "tg", "ts": "tg",
    "తెలంగాణ": "tg", "tamil nadu": "tn", "தமிழ்நாடு": "tn", "karnataka": "ka",
    "ಕರ್ನಾಟಕ": "ka", "kerala": "kl", "കേരളം": "kl", "maharashtra": "mh",
    "महाराष्ट्र": "mh", "gujarat": "gj", "ગુજરાત": "gj", "rajasthan": "rj",
    "राजस्थान": "rj", "uttar pradesh": "up", "उत्तर प्रदेश": "up",
    "madhya pradesh": "mp", "मध्य प्रदेश": "mp", "bihar": "br", "बिहार": "br",
    "west bengal": "wb", "পশ্চিমবঙ্গ": "wb", "delhi": "dl", "दिल्ली": "dl",
    "haryana": "hr", "हरियाणा": "hr", "punjab": "pb", "ਪੰਜਾਬ": "pb",
    "odisha": "od", "orissa": "od", "ଓଡ଼ିଶା": "od", "assam": "as", "অসম": "as",
    "jharkhand": "jh", "झारखंड": "jh", "chhattisgarh": "cg", "छत्तीसगढ़": "cg",
    "uttarakhand": "uk", "उत्तराखंड": "uk", "goa": "ga", "himachal pradesh": "hp",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}

# France: regions and their departments -> one region code. Only matched as a whole
# comma-separated address part, because words like "nord" also occur inside street names.
_FR_REGIONS = {
    "hdf": ["hauts de france", "nord", "pas de calais", "somme", "aisne", "oise", "nord pas de calais", "picardie"],
    "naq": ["nouvelle aquitaine", "gironde", "landes", "dordogne", "lot et garonne", "pyrenees atlantiques",
            "charente", "charente maritime", "vienne", "haute vienne", "deux sevres", "creuse", "correze",
            "aquitaine", "limousin", "poitou charentes"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "idf": ["ile de france", "paris", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise"],
    "ara": ["auvergne rhone alpes", "rhone", "isere", "loire", "ain", "savoie", "haute savoie", "drome",
            "ardeche", "puy de dome", "allier", "cantal", "haute loire", "rhone alpes", "auvergne"],
    "occ": ["occitanie", "haute garonne", "herault", "gard", "aude", "pyrenees orientales", "tarn", "aveyron",
            "lot", "gers", "tarn et garonne", "hautes pyrenees", "ariege", "lozere", "midi pyrenees",
            "languedoc roussillon"],
    "pac": ["provence alpes cote d azur", "paca", "bouches du rhone", "var", "alpes maritimes", "vaucluse",
            "alpes de haute provence", "hautes alpes"],
    "ges": ["grand est", "bas rhin", "haut rhin", "moselle", "meurthe et moselle", "marne", "aube", "ardennes",
            "vosges", "meuse", "haute marne", "alsace", "lorraine", "champagne ardenne"],
    "bre": ["bretagne", "ille et vilaine", "finistere", "morbihan", "cotes d armor"],
    "nor": ["normandie", "seine maritime", "calvados", "manche", "orne", "eure"],
    "bfc": ["bourgogne franche comte", "cote d or", "doubs", "saone et loire", "yonne", "nievre", "jura",
            "haute saone", "territoire de belfort", "bourgogne", "franche comte"],
    "cvl": ["centre val de loire", "loiret", "indre et loire", "loir et cher", "cher", "indre", "eure et loir"],
    "cor": ["corse", "corse du sud", "haute corse"],
}
_FR_MAP = {name: code for code, names in _FR_REGIONS.items() for name in names}
_FR_KEY = re.compile(r"[^a-z]+")


def _fr_key(part: str) -> str:
    """'Hauts-de-France' / 'hauts de france' / "Val-d'Oise" -> comparable key."""
    return _FR_KEY.sub(" ", _translit(part)).strip()


_STATE_MAP = {**US_STATES, **IN_STATES}
_STATE_CODES = set(_STATE_MAP.values())
# Longest names first so "west virginia" wins over "virginia".
_STATE_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(k) for k in sorted(_STATE_MAP, key=len, reverse=True)) + r")(?![\w])",
    re.IGNORECASE,
)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1+")
_DIGITS = re.compile(r"\d+")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|co|net|org|fr|biz|info)(?:\.[a-z]{2})?$")
# Digit look-alikes inside alphabetic words (R0cky, S0ns, 5alters).
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"})


def to_ascii(s: str) -> str:
    """Lower-case ASCII transliteration (handles accents and Indic scripts)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return _translit(s)


def squash(s: str) -> str:
    """Collapse repeated characters; makes transliterations and typos agree."""
    return _REPEAT.sub(r"\1", s)


def _fix_leet(tok: str) -> str:
    if tok.isdigit() or tok.isalpha():
        return tok
    letters = sum(c.isalpha() for c in tok)
    return tok.translate(_LEET) if letters >= 2 else tok


def name_tokens(name: str):
    """Return (core_tokens, legal_tokens) for a business name."""
    s = to_ascii(name).strip()
    m = _DOMAIN.match(s.replace(" ", ""))
    if m:  # "hitechengineering.com" -> single glued token
        return [m.group(1).replace("-", "")], []
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " ")
    s = s.replace("m/s", " ms ").replace("pvt.", "pvt ").replace("'", "")
    s = _NON_ALNUM.sub(" ", s)
    core, legal = [], []
    for t in s.split():
        t = _fix_leet(t)
        t = NAME_ABBR.get(t, t)
        is_legal = t in LEGAL or (len(t) >= 5 and not t.isdigit() and phonetic(t) in LEGAL_SOUND)
        (legal if is_legal else core).append(t)
    return core, legal


def normalize_name(name: str) -> str:
    core, _ = name_tokens(name)
    return " ".join(core)


def split_state(addr: str):
    """Pull a recognised state/region out of the address; returns (rest, state_code)."""
    if not addr:
        return "", ""
    found = []

    def repl(m):
        k = m.group(1)
        found.append(_STATE_MAP.get(k.lower(), _STATE_MAP.get(k, "")))
        return " "

    rest = _STATE_RE.sub(repl, addr)
    # A comma-separated component that is exactly a state code ("..., OR").
    parts = rest.split(",")
    keep = []
    for p in parts:
        q = p.strip().lower()
        if q in _STATE_CODES:
            found.append(q)
            continue
        fk = _fr_key(p) if q else ""
        if fk in _FR_MAP:
            found.append(_FR_MAP[fk])
        else:
            keep.append(p)
    return ",".join(keep), (found[-1] if found else "")


def address_tokens(addr: str):
    """Return (tokens, numbers, state) for an address string."""
    if not addr:
        return [], [], ""
    rest, state = split_state(unicodedata.normalize("NFKC", addr))
    s = to_ascii(rest)
    s = _NON_ALNUM.sub(" ", s.replace("#", " ").replace("'", " "))
    toks = []
    for t in s.split():
        if t in NULL_TOKENS:
            continue
        t = ADDR_ABBR.get(t, t)
        if t:
            toks.append(t)
    nums = []
    for t in toks:
        for d in _DIGITS.findall(t):
            d = d.lstrip("0") or "0"
            if d not in nums:
                nums.append(d)
    return toks, nums, state


def normalize_address(addr: str) -> str:
    toks, _, _ = address_tokens(addr)
    return " ".join(toks)
