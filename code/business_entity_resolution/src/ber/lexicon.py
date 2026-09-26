"""Hand-written general-language lists used by Stage 1 (src/ber/normalize.py).

Nothing here is learned from labels. Each list is general knowledge about business names and
addresses (legal forms, honorifics, administrative regions); CLEANING.md documents why each one
exists. The France entries were chosen by inspecting unlabeled test text (there is no France
training data).

All entries are in the form they take AFTER clean_text (lowercase, no accents, no punctuation,
dotted acronyms merged: "L.L.C." -> "llc", "S.A.R.L." -> "sarl").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# Legal forms (EDA 4.5, 6.5)
# ---------------------------------------------------------------------------------------------

# (family, phrase). Removed from name_core ANYWHERE in the name: word order is shuffled in S2/S3
# ("pvt astor trading ltd", "llc prairie diana"). Multi-token phrases are listed before their parts.
# A name can belong to several families ("private limited" -> {PRIVATE, LIMITED}), so a record
# truncated to "Private" does not conflict with "Private Limited".
LEGAL_UNAMBIGUOUS: list[tuple[str, str]] = [
    ("LLC", "limited liability company"),
    ("LLP", "limited liability partnership"),
    ("LP", "limited partnership"),
    ("PC", "professional corporation"),
    ("PRIVATE", "private"), ("PRIVATE", "pvt"),
    ("PUBLIC", "public limited"),  # "public" alone is a generic word (EDA 6.5), only with "limited"
    ("LIMITED", "limited"), ("LIMITED", "ltd"),
    ("LLC", "llc"), ("LLP", "llp"), ("PLLC", "pllc"),
    ("INC", "inc"), ("INC", "incorporated"),
    ("CORP", "corp"), ("CORP", "corporation"),
    ("CO", "company"),
    ("SARL", "sarl"), ("SASU", "sasu"), ("EURL", "eurl"),
    # Indic-script forms (transliterated "private" / "limited" / "LLP"), as they appear after
    # ZWNJ/ZWJ removal. "प्रा. लि." is the Devanagari abbreviation of "pvt. ltd.".
    ("PRIVATE", "प्राइवेट"), ("PRIVATE", "प्रा"), ("PRIVATE", "ಪ್ರೈವೇಟ್"), ("PRIVATE", "பிரைவேட்"),
    ("PRIVATE", "ప్రైవేట్"), ("PRIVATE", "প্রাইভেট"), ("PRIVATE", "പ്രൈവറ്റ്"), ("PRIVATE", "પ્રાઇવેટ"),
    ("PRIVATE", "ਪ੍ਰਾਈਵੇਟ"), ("PRIVATE", "ପ୍ରାଇଭେଟ୍"),
    ("LIMITED", "लिमिटेड"), ("LIMITED", "लि"), ("LIMITED", "ಲಿಮಿಟೆಡ್"), ("LIMITED", "லிமிடெட்"),
    ("LIMITED", "లిమిటెడ్"), ("LIMITED", "লিমিটেড"), ("LIMITED", "ലിമിറ്റഡ്"), ("LIMITED", "લિમિટેડ"),
    ("LIMITED", "ਲਿਮਟਿਡ"), ("LIMITED", "ଲିମିଟେଡ୍"),
    ("LLP", "एलएलपी"), ("LLP", "ಎಲ್ಎಲ್ಪಿ"), ("LLP", "ଏଲ୍ଏଲ୍ପି"), ("LLP", "ఎల్ఎల్పి"), ("LLP", "எல்எல்பி"),
    ("LLP", "এলএলপি"), ("LLP", "એલએલપી"), ("LLP", "എൽഎൽപി"),
]

# Short forms that are also ordinary words or initials ("co" in "Co Operative", "sa"/"sas" as
# Indian initials, "lp", "pc"). Removed only at the START or END of the name (after the
# unambiguous forms are gone), never in the middle ("Maa Co Services" keeps "co").
LEGAL_EDGE_ONLY: list[tuple[str, str]] = [
    ("CO", "and co"), ("CO", "co"), ("CO", "cie"),
    ("SAS", "sas"), ("SA", "sa"), ("SCI", "sci"), ("SNC", "snc"),
    ("LP", "lp"), ("PC", "pc"), ("EI", "ei"),
]

# "Messrs"/honorific prefixes injected at the start of India S2/S3 names ("Smt My Service").
# Verified on train true pairs (train_ids only): each starts ~32k S2/S3 names whose S1 partner does
# not start with it, vs <= 607 S1 names. "shree"/"sri" are NOT here: they are part of real names
# ("Shree Trading"). "ms" is only stripped when the raw text has the slash form "M/s".
NAME_PREFIXES = ["mr", "smt", "shri", "dr"]

# ---------------------------------------------------------------------------------------------
# State / region slot (EDA 5.3). Matched against WHOLE comma clauses only, never single tokens,
# so "in" or "me" inside a street name is safe. Not canonicalised in Stage 1.
# ---------------------------------------------------------------------------------------------

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia", "pr": "puerto rico", "gu": "guam", "vi": "virgin islands",
}

INDIA_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "uttaranchal", "west bengal", "andaman and nicobar islands", "chandigarh",
    "dadra and nagar haveli and daman and diu", "dadra and nagar haveli", "daman and diu", "delhi",
    "new delhi", "jammu and kashmir", "ladakh", "lakshadweep", "puducherry", "pondicherry",
]
# Vehicle-registration style codes used by India S3 ("MH", "DL"). Codes that collide with a US
# code ("ga", "or", "ct", "la", "ut") are harmless: the clause goes to addr_state_raw either way.
INDIA_CODES = ["ap", "ar", "as", "br", "cg", "ct", "ga", "gj", "hr", "hp", "jh", "jk", "ka", "kl",
               "mp", "mh", "mn", "ml", "mz", "nl", "od", "or", "pb", "rj", "sk", "tn", "tg", "ts",
               "tr", "up", "uk", "ut", "wb", "an", "ch", "dn", "dd", "dl", "la", "ld", "py"]

# France: current regions, pre-2016 regions, and departments (metropolitan + overseas).
FRANCE_REGIONS = [
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
    "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine", "occitanie",
    "pays de la loire", "provence alpes cote dazur",
    "guadeloupe", "martinique", "guyane", "la reunion", "mayotte",
    "alsace", "aquitaine", "auvergne", "basse normandie", "bourgogne", "centre", "champagne ardenne",
    "franche comte", "haute normandie", "languedoc roussillon", "limousin", "lorraine",
    "midi pyrenees", "nord pas de calais", "picardie", "poitou charentes", "rhone alpes",
]
FRANCE_DEPARTMENTS = [
    "ain", "aisne", "allier", "alpes de haute provence", "hautes alpes", "alpes maritimes", "ardeche",
    "ardennes", "ariege", "aube", "aude", "aveyron", "bouches du rhone", "calvados", "cantal",
    "charente", "charente maritime", "cher", "correze", "corse du sud", "haute corse", "cote dor",
    "cotes darmor", "creuse", "dordogne", "doubs", "drome", "eure", "eure et loir", "finistere",
    "gard", "haute garonne", "gers", "gironde", "herault", "ille et vilaine", "indre",
    "indre et loire", "isere", "jura", "landes", "loir et cher", "loire", "haute loire",
    "loire atlantique", "loiret", "lot", "lot et garonne", "lozere", "maine et loire", "manche",
    "marne", "haute marne", "mayenne", "meurthe et moselle", "meuse", "morbihan", "moselle",
    "nievre", "nord", "oise", "orne", "pas de calais", "puy de dome", "pyrenees atlantiques",
    "hautes pyrenees", "pyrenees orientales", "bas rhin", "haut rhin", "rhone", "haute saone",
    "saone et loire", "sarthe", "savoie", "haute savoie", "paris", "seine maritime",
    "seine et marne", "yvelines", "deux sevres", "somme", "tarn", "tarn et garonne", "var",
    "vaucluse", "vendee", "vienne", "haute vienne", "vosges", "yonne", "territoire de belfort",
    "essonne", "hauts de seine", "seine saint denis", "val de marne", "val doise",
]

STATE_CLAUSES: list[str] = sorted(set(
    list(US_STATES) + list(US_STATES.values()) + INDIA_STATES + INDIA_CODES + FRANCE_REGIONS
    + FRANCE_DEPARTMENTS
))
