"""Stage 1: deterministic text cleaning, shared by train and test.

Everything here is a polars expression, so it runs vectorised in Rust. No Python per-row code.
Raw columns are never modified; `clean_frame` only ADDS columns. Every rule was checked against
real rows (see CLEANING.md); the comments give the reason for each one.

Country-agnostic by design: no rule looks at the `country` column. Rules written for US/India
patterns are keyed on token shapes, so on text they were not written for (e.g. French) they do
nothing instead of failing.
"""

from __future__ import annotations

import sys
import unicodedata

import polars as pl

from ber.lexicon import LEGAL_EDGE_ONLY, LEGAL_UNAMBIGUOUS, NAME_PREFIXES, STATE_CLAUSES

NORMALIZE_VERSION = "1.2.1"

# ---------------------------------------------------------------------------------------------
# Common text steps
# ---------------------------------------------------------------------------------------------

# Non-ASCII decimal digits (Devanagari ०, Tamil ௦, fullwidth, ...) -> ASCII. Built from the Unicode
# database, not a hand-written list.
_ND = [chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == "Nd" and c > 0x7F]
_ND_ASCII = [str(unicodedata.digit(c)) for c in _ND]

# Quote-like characters are deleted, not turned into spaces, so "Shopper's" == "Shoppers".
# \x1a (SUB) is included: in this data it replaces an apostrophe (D\x1asouza, Shopper\x1aS).
_QUOTES = "\"“”'‘’`´\x1a"

# cp1252 mojibake of UTF-8 punctuation, e.g. "â\x80\x99" (’) and "Â\x80\x93" (–). Seen only as
# these exact 3-char sequences, so the rule cannot touch a real "â" (French).
_MOJIBAKE_QUOTE = r"[âÂ]\x80[\x98\x99\x9c\x9d]"
_MOJIBAKE_OTHER = r"[âÂ]\x80[\x80-\x9f]"

_FILLER = r"(^|[\s,])(?:null|n\s*/\s*a)([\s,]|$)"

# Latin letters with no canonical decomposition (œ is common in French: "Œillets").
_LIGATURES = {"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "đ": "d", "ł": "l", "ı": "i"}


def _pre(e: pl.Expr) -> pl.Expr:
    """Steps 1-3 plus lowercase, '&' and placeholders. Punctuation (incl. commas) is still present."""
    e = (
        e.str.replace_all(_MOJIBAKE_QUOTE, "")
        # other mojibake punctuation is a dash in practice ("WORLDMARK Â\x80\x93 3"); '-' is then
        # dropped between words and kept inside numbers by _punct
        .str.replace_all(_MOJIBAKE_OTHER, "-")
        .str.replace_all("´", "")  # NFKC would turn it into a space + combining accent
        # 1. NFKC, then non-ASCII digits -> ASCII
        .str.normalize("NFKC")
        .str.replace_many(_ND, _ND_ASCII)
        # 2. quotes
        .str.replace_all(f"[{_QUOTES}]", "")
        # zero-width format chars (ZWNJ/ZWJ inside Indic words) are deleted: turning them into a
        # space would split the word. Other control chars become spaces.
        .str.replace_all(r"\p{Cf}", "")
        .str.replace_all(r"\p{Cc}", " ")
        # 3. accents: remove combining marks ONLY when they follow a Latin base letter.
        #    Indic vowel signs are combining marks too and must survive.
        .str.normalize("NFD")
        .str.replace_all(r"(\p{Latin})\p{Mn}+", "${1}")
        .str.normalize("NFC")
        # 4. lowercase, ligatures, '&'
        .str.to_lowercase()
        .str.replace_many(list(_LIGATURES), list(_LIGATURES.values()))
        .str.replace_all("&", " and ")
        # "<NULL>" is a missing-value marker in S2/S3 (~0.9% of rows, never in S1). Bare "null" and
        # "n/a" are filler tokens too (87k US S2 addresses have a "null" clause, 15k India S3 "n/a").
        # Two passes: adjacent fillers share the separator. "##" / "#" go with the punctuation.
        .str.replace_all(r"<\s*null\s*>", " ")
        .str.replace_all(_FILLER, "${1} ${2}")
        .str.replace_all(_FILLER, "${1} ${2}")
        # French "N°12" -> "no 12" (the degree sign would otherwise vanish and glue n to the number)
        .str.replace_all(r"(^|[^\p{L}])n\s*°\s*", "${1}no ")
    )
    # OCR digits inside words: "H0uston", "De1hi", "Ree1sch". Only a 0/1 with Latin letters on BOTH
    # sides, in a token made only of letters and 0/1, so "2nd", "12b", "1-11-251/1b", "s1w33789" stay.
    # Each pass fixes one digit per token (matches cannot overlap); three passes cover real cases.
    for _ in range(3):
        for digit, letter in (("0", "o"), ("1", "l")):
            e = e.str.replace_all(
                rf"(^|[^\p{{L}}\p{{N}}])([\p{{Latin}}01]*\p{{Latin}}){digit}(\p{{Latin}}[\p{{Latin}}01]*)([^\p{{L}}\p{{N}}]|$)",
                f"${{1}}${{2}}{letter}${{3}}${{4}}")
    return e


def _punct(e: pl.Expr, keep_commas: bool) -> pl.Expr:
    """Punctuation -> space; keep '/' and '-' inside number-like tokens; tidy whitespace.

    With keep_commas=True, commas survive as clause separators (normalised to ", ").
    """
    keep = r"\p{L}\p{M}\p{N}/\-\s" + ("," if keep_commas else "")
    e = (
        e.str.replace_all(f"[^{keep}]", " ")
        # care-of forms stay one token (c/o, s/o, d/o, w/o), else "C/O R. Kumar" -> "cor kumar"
        .str.replace_all(r"(^|[^\p{L}])([cdsw]) ?/ ?o([^\p{L}]|$)", "${1}${2}o${3}")
        # letter-letter: "hauts-de-france" -> "hauts de france", "c/o" -> "c o" (two passes
        # because matches cannot overlap: "a-b-c")
        .str.replace_all(r"([\p{L}\p{M}])[/\-]+([\p{L}\p{M}])", "${1} ${2}")
        .str.replace_all(r"([\p{L}\p{M}])[/\-]+([\p{L}\p{M}])", "${1} ${2}")
        # word-number: "sector-17" -> "sector 17", "rz-40" -> "rz 40"; a single letter stays
        # attached ("12-b", "a-73", "g-864") because it is part of the house number
        .str.replace_all(r"(\p{L}{2,})[/\-]+(\d)", "${1} ${2}")
        .str.replace_all(r"(\d)[/\-]+(\p{L}{2,})", "${1} ${2}")
        .str.replace_all(r"([/\-])[/\-]+", "${1}")
        # '/' and '-' at token edges are punctuation, not part of a token
        .str.replace_all(r"(^|[\s,])[/\-]+", "${1}")
        .str.replace_all(r"[/\-]+([\s,]|$)", "${1}")
        .str.replace_all(r"\s+", " ")
    )
    if keep_commas:
        e = e.str.replace_all(r"\s*,[\s,]*", ", ")
    return e.str.strip_chars(" ,")


def _merge_initials(e: pl.Expr) -> pl.Expr:
    """Join runs of single Latin letters: "l l c" -> "llc", "k r puram" -> "kr puram".

    Dotted acronyms (L.L.C., S.A.R.L., R.K. Puram) become letter runs after punctuation removal.
    Never crosses a comma. Implemented with sentinels because Rust regex has no lookaround.
    """
    return (
        e.str.replace_all(" ", "  ")
        .str.replace_all(r"(^| )(\p{Latin})( |$)", "${1}\x02${2}\x03${3}")
        .str.replace_all(r"\x03 +\x02", "")
        .str.replace_all(r"[\x02\x03]", "")
        .str.replace_all(r" +", " ")
    )


def clean_text(e: pl.Expr) -> pl.Expr:
    """The common steps 1-4 (no commas). Used for name_clean and addr_clean."""
    return _merge_initials(_punct(_pre(e), keep_commas=False))


# ---------------------------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------------------------

SCRIPTS = ["Latin", "Devanagari", "Tamil", "Kannada", "Telugu", "Bengali", "Gujarati", "Gurmukhi", "Malayalam", "Odia"]
_SCRIPT_REGEX = {s: rf"\p{{{'Oriya' if s == 'Odia' else s}}}" for s in SCRIPTS}


def script_of(e: pl.Expr) -> pl.Expr:
    """Dominant script by letter count. "Other" = letters of another script, "None" = no letters."""
    counts = {s: e.str.count_matches(rx) for s, rx in _SCRIPT_REGEX.items()}
    top = pl.max_horizontal(list(counts.values()))
    out = pl.when(e.str.contains(r"\p{L}")).then(pl.lit("Other")).otherwise(pl.lit("None"))
    for s in reversed(SCRIPTS):  # earlier scripts win ties
        out = pl.when((top > 0) & (counts[s] == top)).then(pl.lit(s)).otherwise(out)
    return out


def has_nonlatin(e: pl.Expr) -> pl.Expr:
    return e.str.contains(r"[\p{L}&&\P{Latin}]")


# ---------------------------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------------------------

# (class, token pattern). Matched against the END of name_clean, first match wins, so longer
# forms come first. "private"/"pvt" alone is a truncated "private limited" (978/978 checked true
# pairs have "private limited"/"pvt ltd" on the other side). CO includes "and co" ("Ram & Co").
LEGAL_SUFFIXES: list[tuple[str, str]] = [
    ("PRIVATE_LIMITED", r"(?:private|pvt|p) (?:limited|ltd)"),
    ("PUBLIC_LIMITED", r"public (?:limited|ltd)"),
    ("LLC", r"llc|limited liability company"),
    ("LLP", r"llp|limited liability partnership"),
    ("PLLC", r"pllc"),
    ("LP", r"lp|limited partnership"),
    ("PRIVATE_LIMITED", r"private|pvt"),
    ("LIMITED", r"limited|ltd"),
    ("INC", r"inc|incorporated"),
    ("CORP", r"corp|corporation"),
    ("PC", r"pc|professional corporation"),
    ("CO", r"(?:and )?(?:co|company|cie)"),
    ("SASU", r"sasu"),
    ("SARL", r"sarl"),
    ("SAS", r"sas"),
    ("SA", r"sa"),
    ("EURL", r"eurl"),
    ("SNC", r"snc"),
    ("SCI", r"sci"),
    ("EI", r"ei"),
]
# "M/s" (Messrs) prefix: 1.3% of India S2/S3 names, 0% of S1. Detected on the raw text, because
# after cleaning it is indistinguishable from the initials "Ms".
_MS_PREFIX_RAW = r"(?i)^\s*m\s*/\s*s\b\.?"
# Trailing tags "#48688" appended to S2/S3 names (7.3k India true pairs, ~0 in S1). Removed from
# name_core only; name_clean keeps the digits, so they are still in name_numbers.
_TAG_RAW = r"#\s*\d+"
# Domains and handles: "sarthitrading.com", "www.x.co.in", "@handle" (3-4% of S2/S3 names, EDA 4.7)
_DOMAIN_RAW = r"(?i)(?:https?://)?(?:www\.)?([\p{L}\d][\p{L}\d-]*)\.(?:com|net|org|in|co|fr|biz|info|io|us)(?:\.(?:in|uk|fr))?\b"
_HANDLE_RAW = r"@([\p{L}\d_.]+)"


def _tok_alt(phrases: list[str]) -> str:
    """Alternation over token phrases in the padded format (" a  b "): inner spaces doubled."""
    return "|".join(p.replace(" ", "  ") for p in sorted(phrases, key=len, reverse=True))


# Unambiguous forms, grouped per family. Multi-token phrases that contain another family's token
# ("limited liability company" contains "limited") run first, one by one, so they are consumed
# before the single tokens are looked at.
_MULTI = [(f, p) for f, p in LEGAL_UNAMBIGUOUS if " " in p]
_SINGLE: dict[str, list[str]] = {}
for _f, _p in LEGAL_UNAMBIGUOUS:
    if " " not in _p:
        _SINGLE.setdefault(_f, []).append(_p)
_EDGE_ALT = "|".join(p for _, p in sorted(LEGAL_EDGE_ONLY, key=lambda t: -len(t[1])))
_EDGE: dict[str, list[str]] = {}
for _f, _p in LEGAL_EDGE_ONLY:
    _EDGE.setdefault(_f, []).append(_p)
LEGAL_FAMILIES = sorted({f for f, _ in LEGAL_UNAMBIGUOUS} | set(_EDGE))


def _pad(e: pl.Expr) -> pl.Expr:
    return pl.concat_str([pl.lit(" "), e.str.replace_all(" ", "  "), pl.lit(" ")])


def _unpad(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"\s+", " ").str.strip_chars(" ")


def add_legal(lf: pl.LazyFrame, base: str) -> pl.LazyFrame:
    """Add name_core and legal_families from the cleaned name in column `base` (prefixes and tags
    already removed).

    Unambiguous legal forms are removed anywhere; edge-only forms (co, sa, sas, ...) only from the
    start or end of what is left. legal_families = sorted list of every family found (possibly
    empty). If removing the forms would leave nothing, name_core falls back to `base`.
    Intermediate results are materialised as columns so the expression tree stays small.
    """
    # "(P) Ltd" is "Pvt Ltd"; a lone "p" is not a legal form
    lf = lf.with_columns(_lp=_pad(pl.col(base)).str.replace_all(" p  (ltd|limited) ", " pvt  ${1} "))
    flags: dict[str, list[str]] = {}
    steps = [(fam, f" {_tok_alt([ph])} ") for fam, ph in _MULTI]
    steps += [(fam, f" (?:{_tok_alt(toks)}) ") for fam, toks in _SINGLE.items()]
    for i, (fam, rx) in enumerate(steps):
        flags.setdefault(fam, []).append(f"_lf{i}")
        lf = lf.with_columns(**{f"_lf{i}": pl.col("_lp").str.contains(rx)}).with_columns(
            _lp=pl.col("_lp").str.replace_all(rx, " "))
    lf = lf.with_columns(_lmid=_unpad(pl.col("_lp")))
    lf = lf.with_columns(
        _llead=_pad(pl.col("_lmid").str.extract(f"^((?:(?:{_EDGE_ALT}) )+)\\S").fill_null("")),
        _ltrail=_pad(pl.col("_lmid").str.extract(f"\\S((?: (?:{_EDGE_ALT}))+)$").fill_null("")),
        _lcore=pl.col("_lmid").str.replace(f"^(?:(?:{_EDGE_ALT}) )+(\\S)", "${1}")
        .str.replace(f"(\\S)(?: (?:{_EDGE_ALT}))+$", "${1}"),
    )
    edge_flags = {}
    for fam, toks in _EDGE.items():
        rx = f" (?:{_tok_alt(toks)}) "
        edge_flags[fam] = pl.col("_llead").str.contains(rx) | pl.col("_ltrail").str.contains(rx)
    fams = sorted(set(flags) | set(edge_flags))
    fam_expr = []
    for fam in fams:
        cond = pl.any_horizontal([pl.col(c) for c in flags.get(fam, [])] + ([edge_flags[fam]] if fam in edge_flags else []))
        fam_expr.append(pl.when(cond).then(pl.lit(fam)).otherwise(pl.lit(None, pl.String)))
    lf = lf.with_columns(
        name_core=pl.when(pl.col("_lcore") == "").then(pl.col(base)).otherwise(pl.col("_lcore")),
        legal_families=pl.concat_list(fam_expr).list.drop_nulls(),
    )
    return lf.drop("_lp", "_lmid", "_llead", "_ltrail", "_lcore", *[c for cs in flags.values() for c in cs])


def legal_suffix_class(name_clean: pl.Expr) -> pl.Expr:
    """Legacy single class of the LAST legal form (kept for compatibility; see legal_families)."""
    out = pl.lit("NONE")
    for cls, pat in reversed(LEGAL_SUFFIXES):
        out = pl.when(name_clean.str.contains(f" (?:{pat})$")).then(pl.lit(cls)).otherwise(out)
    return out


def _et_to_and(e: pl.Expr) -> pl.Expr:
    # French "et" = "&" = "and" (S1 France uses "&", S2/S3 France "et" in 1%, EDA 4.6)
    return e.str.replace_all(r"(^| )et( |$)", "${1}and${2}").str.replace_all(r"(^| )et( |$)", "${1}and${2}")


def name_base(raw_name: pl.Expr) -> pl.Expr:
    """Cleaned name with tags removed, domains/handles reduced to their stem and injected
    honorific prefixes stripped from the start. Input to add_legal."""
    src = (raw_name.str.replace_all(_TAG_RAW, " ").str.replace_all(_DOMAIN_RAW, " ${1} ")
           .str.replace_all("@", " "))
    b = _et_to_and(clean_text(src))
    b = pl.when(raw_name.str.contains(_MS_PREFIX_RAW)).then(b.str.replace(r"^ms (\S)", "${1}")).otherwise(b)
    return b.str.replace(f"^(?:{'|'.join(NAME_PREFIXES)}) (\\S)", "${1}")


def name_domain_stem(raw_name: pl.Expr) -> pl.Expr:
    stem = pl.coalesce(raw_name.str.extract(_DOMAIN_RAW, 1), raw_name.str.extract(_HANDLE_RAW, 1))
    stem = clean_text(stem).str.replace_all(r"[\s/\-]", "")
    return pl.when(stem == "").then(None).otherwise(stem)


# ---------------------------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------------------------

# A landmark clause runs from the keyword to the next comma (98% of India clauses end at a comma).
# The keyword must be followed by a word in the same clause: "Opp, AL" is a town in Alabama.
# "landmark" itself is not a keyword ("Landmark Road" is a US street name).
_LANDMARK = r"(?:near|nr|opp|opposite|behind|beside|next to|in front of)"
_LANDMARK_CLAUSE = rf"(^|[\s,]){_LANDMARK} [^,]+"
_STREET_WORDS = r"street|st|road|rd|avenue|ave|drive|dr|lane|ln|boulevard|blvd|court|ct|circle|cir|way|place|pl"
# ...but a keyword directly followed by a street type is a street name ("303 Opp Avenue")
_LANDMARK_PROTECT = rf"(^|[\s,])({_LANDMARK}) ((?:{_STREET_WORDS})(?:[\s,]|$))"

_NUM = r"[^\s,]*\d[^\s,]*"  # a number-like token

# Unambiguous street-type abbreviations -> one canonical token. Not mapped on purpose:
# "ct" (Connecticut), "fl" (Florida; handled next to numbers below), "pl"/"dr"/"bd"/"av" (see below).
_ALWAYS = {
    "rd": "road", "ave": "avenue", "blvd": "boulevard", "boul": "boulevard", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "cir": "circle", "flr": "floor", "bldg": "building",
    "apt": "apartment", "ngr": "nagar", "nagr": "nagar", "mrg": "marg", "rte": "route",
    "gf": "ground floor", "grd": "ground",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}
_UNIT_WORDS = (r"n|s|e|w|ne|nw|se|sw|north|south|east|west|no|apartment|ste|suite|unit|floor|building"
               r"|street|road|avenue|drive|lane|boulevard")  # "Mesquite St Street"
_FR_ARTICLES = r"de|du|des|la|le|l|d"
_STREET_TYPES = r"street|road|avenue|drive|lane|boulevard|place|circle|court|ct|way|terrace|parkway|highway"


def _std_clauses(e: pl.Expr) -> pl.Expr:
    """Street-type canonicalisation on comma-separated text. Returns comma-separated text.

    Tokens are padded so each one is surrounded by its own spaces (" a  b , c "), which lets
    token-level rules run without lookaround and without overlapping matches.
    """
    p = pl.concat_str([pl.lit(" "), e.str.replace_all(",", " ,").str.replace_all(" ", "  "), pl.lit(" ")])
    p = p.str.replace_many([f" {k} " for k in _ALWAYS], [f" {v} " for v in _ALWAYS.values()])
    p = (
        # "st": street at the end of a clause or before a number/unit word; saint otherwise
        # (US "Wabash St," vs "St Louis", France "St-Herblain", India "St Sebastian Church")
        p.str.replace_all(r" st ( ,|$)", " street ${1}")
        .str.replace_all(rf" st ( {_NUM}| (?:{_UNIT_WORDS}) )", " street ${1}")
        .str.replace_all(" st ", " saint ")
        # "ste": suite before a number (US "Unit STE 210"), sainte otherwise (France "Ste Marie")
        .str.replace_all(rf" ste ( {_NUM})", " suite ${1}")
        .str.replace_all(" ste ", " sainte ")
        # "dr": drive only at the end of a clause (US); elsewhere it is Doctor (India "Dr Ambedkar")
        .str.replace_all(r" dr ( ,|$)", " drive ${1}")
        # "fl": floor only next to a number ("2nd fl", "fl 3"); at the end it is Florida
        .str.replace_all(rf" ({_NUM})  fl ", " ${1}  floor ")
        .str.replace_all(rf" fl  ({_NUM}) ", " floor  ${1} ")
        # "pl": place at a clause end or after a number; India "Pl No 5" (plot) is left alone
        .str.replace_all(r" pl ( ,|$)", " place ${1}")
        # "r": rue only right after a house number and not before a street type
        # (France "106 R Neuve"; US "231 R St" is R Street in Washington DC)
        .str.replace_all(rf" r  ((?:{_STREET_TYPES}) )", " \x02  ${1}")
        .str.replace_all(rf" ({_NUM})  r ", " ${1}  rue ")
        .str.replace_all("\x02", "r")
    )
    p = (
        # "ct": court when it ends a clause that has a street name before it ("067 Production Ct,");
        # a clause that is only "ct" is Connecticut (41k vs 69k US S2 addresses)
        p.str.replace_all(r"([^\s,])  ct ( ,|$)", "${1}  court ${2}")
        # "no" before a number is only a marker ("No 510", French "N° 12" -> "no 12"): drop it
        .str.replace_all(rf" no  ({_NUM}) ", " ${1} ")
    )
    for short, full in [("r", "rue"), ("bd", "boulevard"), ("av", "avenue"), ("pl", "place"), ("imp", "impasse"),
                        ("ch", "chemin")]:
        # after a house number: "47 Bd du Tertre"; India "Block-BD", "Av Colony" are initials
        if short != "r":
            p = p.str.replace_all(rf" ({_NUM})  {short} ", f" ${{1}}  {full} ")
        # clause start before a French article (reordered France S2/S3: "R du Brun Pin, Tourcoing")
        p = p.str.replace_all(rf"(^ |,  ){short}  ({_FR_ARTICLES}) ", f"${{1}}{full}  ${{2}} ")
    p = p.str.replace_all(r" (?:sec|sect) ( \d)", " sector ${1}")  # India "SEC. 4"
    return p.str.replace_all(r"\s+", " ").str.replace_all(" , ", ", ").str.strip_chars(" ,")


def _drop_commas(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(",", "")


def _strip_leading_zeros(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"(^|[^\d])0+(\d)", "${1}${2}")


# ---------------------------------------------------------------------------------------------
# Frame-level entry point
# ---------------------------------------------------------------------------------------------

# Words whose following number is a unit, not the house number (US "Unit 16B", "Fl 0", "Suite 110").
# S2 often drops the units that S1 has (EDA 5). India "Flat No 4A" / "Shop No 1" are NOT here: in
# India that number is often the only identifier (S2 "H.no 4A" = S1 "Flat No.4A"); Stage 3 showed
# excluding them lowers India house-number agreement (0.589 vs 0.622 exact on true pairs).
_UNIT_WORDS_ADDR = ["unit", "suite", "ste", "apt", "apartment", "fl", "flr", "floor", "room", "rm", "unt"]
# India S2/S3 prepend a fake number clause ("Door No 864", "H.no 15", "Plot 963", "Block D-764",
# "NO 32"); S1 also starts this way when the clause is real. When an address starts with one of
# these and has another house-number candidate, the second candidate is used.
_INJECTED_START = r"^(?:door no|door|d no|dno|h no|hno|house no|plot no|plot|block no|block|no) [^\s]*\d"
_ORDINAL = r"^\d+(?:st|nd|rd|th|er|eme|e)$"
_MAX_LETTER_RUN = 3  # a token with this many letters in a row is a word, not a house number  # 87th, 2nd, French 1er / 2eme


def _number_part(el: pl.Expr) -> pl.Expr:
    """addr_numbers element: leading zeros stripped; a token glued to a word ("chambers16/11",
    "cour2", 3+ letters in a row) keeps only its number part ("16/11", "2"), and is dropped
    (null, filtered out) if that part is an ordinal ("annexe3rd" -> "3rd"). Plain ordinals such as
    "87th" stay, as before. addr_house_number still rejects glued tokens outright."""
    run = rf"\p{{L}}{{{_MAX_LETTER_RUN},}}"
    part = el.str.replace_all(run, " ").str.strip_chars(" /-").str.extract(r"(\S*\d\S*)")
    glued = el.str.contains(run)
    out = pl.when(~glued).then(el).when(part.str.contains(_ORDINAL)).then(None).otherwise(part)
    return _strip_leading_zeros(out)


def _house_and_unit(addr_nl: pl.Expr, skip_injected: bool = True, letter_run: int | None = _MAX_LETTER_RUN,
                    glued: str = "reject") -> tuple[pl.Expr, pl.Expr]:
    """(addr_house_number, addr_unit) from comma-separated, landmark-free, initials-merged text.

    skip_injected=False, letter_run=None and glued="keep_digits" exist only for the Stage 3 ablations."""
    # commas stay as their own tokens so a unit word never reaches across a clause boundary
    # ("2nd floor, 1-11-251/1b": the number is not a floor number)
    toks = addr_nl.str.replace_all(",", " ,").str.split(" ")
    el, prev, prev2 = pl.element(), pl.element().shift(1), pl.element().shift(2)
    is_num = el.str.contains(r"\d")
    after_unit = (prev.is_in(_UNIT_WORDS_ADDR).fill_null(False)
                  | (prev.is_in(["no", "number"]).fill_null(False) & prev2.is_in(_UNIT_WORDS_ADDR).fill_null(False)))
    fraction_after_num = el.str.contains(r"^\d+/\d+$") & prev.str.contains(r"\d").fill_null(False)  # "10501 1/2"
    # a house/unit number has a digit and no run of 3+ letters: "7salemmainroad" (glued
    # "N.H.7SALEMMAINROAD"), "lane4th", "sec9", "etsu513" are words with a digit, not numbers
    if glued == "keep_digits" and letter_run:  # ablation: cut the letter runs off instead of rejecting
        el = el.str.replace_all(rf"\p{{L}}{{{letter_run},}}", "").str.strip_chars("/-")
    valid = is_num & ~el.str.contains(_ORDINAL)
    if letter_run and glued == "reject":
        valid = valid & ~el.str.contains(rf"\p{{L}}{{{letter_run},}}")
    cand = toks.list.eval(el.filter(valid & ~after_unit & ~fraction_after_num))
    units = toks.list.eval(el.filter(valid & after_unit))
    injected = addr_nl.str.contains(_INJECTED_START) & (cand.list.len() >= 2) & pl.lit(skip_injected)
    house = pl.when(injected).then(cand.list.get(1, null_on_oob=True)).otherwise(cand.list.first())
    unit = units.list.first()
    # no other number: the unit number is the best identifier we have ("Unit 16B, Main St")
    return _strip_leading_zeros(pl.coalesce(house, unit)), _strip_leading_zeros(unit)


def _state_index(comps: pl.Expr, has_latin: pl.Expr) -> pl.Expr:
    """Index of THE state/region clause, or null. A clause qualifies when it is entirely a known
    state / region name or code, or - in an address that also has Latin text - written wholly in a
    non-Latin script (India S2 writes the state in the native script: 88% of such clauses are the
    last clause and the top values are all state names).

    Only one clause is taken: the LAST clause if it qualifies (the state slot in every source),
    otherwise the first qualifying clause (S2/S3 reorder components: "TX, Houston, 12 Main St").
    So "Washington, IN" gives state "in" and keeps the city "washington"."""
    el = pl.element()
    known = el.is_in(STATE_CLAUSES)
    native = el.str.contains(r"\p{L}") & ~el.str.contains(r"\p{Latin}")
    mask = pl.when(has_latin).then(comps.list.eval(known | native)).otherwise(comps.list.eval(known))
    first = mask.list.eval(pl.element().arg_true()).list.first()
    return pl.when(mask.list.last()).then(comps.list.len() - 1).otherwise(first)


STAGE1_COLUMNS = [
    "name_clean", "legal_suffix_class", "legal_families", "name_core", "name_nospace", "name_domain_stem",
    "name_script", "name_has_nonlatin", "name_numbers",
    "addr_clean", "addr_numbers", "addr_house_number", "addr_unit", "addr_landmark", "addr_no_landmark",
    "addr_std", "addr_std_components", "addr_state_raw", "addr_core", "addr_script", "addr_has_nonlatin",
]
_ADDR_COLUMNS = [c for c in STAGE1_COLUMNS if c.startswith("addr_")]


def addr_nl(addr: pl.Expr) -> pl.Expr:
    """Comma-separated, landmark-free, initials-merged address (internal; exposed for Stage 3)."""
    c0 = _punct(_pre(addr), keep_commas=True).str.replace_all(_LANDMARK_PROTECT, "${1}\x02${2} ${3}")
    return _merge_initials(c0.str.replace_all(_LANDMARK_CLAUSE, "${1}").str.replace_all("\x02", "")
                           .str.replace_all(r"\s*,[\s,]*", ", ").str.strip_chars(" ,"))


def clean_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add all Stage 1 columns. Input needs business_name and business_address; nothing is dropped.

    Empty addresses (after removing fillers) give null in every addr_* column, never "".
    """
    name, addr = pl.col("business_name"), pl.col("business_address")
    lf = lf.with_columns(
        name_clean=_et_to_and(clean_text(name)),
        _name_base=name_base(name),
        name_domain_stem=name_domain_stem(name),
        # commas kept, initials not merged yet; \x02 shields street names from landmark detection
        _addr_c0=_punct(_pre(addr), keep_commas=True).str.replace_all(_LANDMARK_PROTECT, "${1}\x02${2} ${3}"),
    )
    lf = add_legal(lf, "_name_base")
    lf = lf.with_columns(
        legal_suffix_class=legal_suffix_class(pl.col("name_clean")),
        name_script=script_of(pl.col("name_clean")),
        name_has_nonlatin=has_nonlatin(pl.col("name_clean")),
        name_numbers=pl.col("name_clean").str.extract_all(r"\d+"),
        addr_clean=_drop_commas(_merge_initials(pl.col("_addr_c0").str.replace_all("\x02", ""))),
        # landmark detection runs before initials are merged, so "N R Colony" is not "nr colony"
        addr_landmark=pl.col("_addr_c0").str.extract_all(_LANDMARK_CLAUSE)
        .list.eval(pl.element().str.replace_all("\x02", "").str.strip_chars(" ,")).list.join(", "),
        _addr_nl=_merge_initials(
            pl.col("_addr_c0").str.replace_all(_LANDMARK_CLAUSE, "${1}").str.replace_all("\x02", "")
            .str.replace_all(r"\s*,[\s,]*", ", ").str.strip_chars(" ,")
        ),
    )
    house, unit = _house_and_unit(pl.col("_addr_nl"))
    lf = lf.with_columns(
        name_nospace=pl.col("name_core").str.replace_all(" ", ""),
        addr_numbers=pl.col("addr_clean").str.extract_all(_NUM).list.eval(_number_part(pl.element())).list.drop_nulls(),
        addr_house_number=house,
        addr_unit=unit,
        addr_no_landmark=_drop_commas(pl.col("_addr_nl")),
        addr_std_components=_std_clauses(pl.col("_addr_nl")).str.split(", ").list.eval(
            pl.element().filter(pl.element() != "")),
        addr_script=script_of(pl.col("addr_clean")),
        addr_has_nonlatin=has_nonlatin(pl.col("addr_clean")),
        _has_latin=pl.col("addr_clean").str.contains(r"\p{Latin}"),
    )
    comps, has_latin = pl.col("addr_std_components"), pl.col("_has_latin")
    lf = lf.with_columns(_state_i=_state_index(comps, has_latin))
    lf = lf.with_columns(
        addr_std=comps.list.join(" "),
        addr_state_raw=comps.list.get(pl.col("_state_i"), null_on_oob=True),
        # leading zeros stripped in each numeric part ("rz 0040" = "rz 40", "03/c" = "3/c")
        # drop the state clause; with no state, the index sentinel len() keeps every clause
        addr_core=_strip_leading_zeros(pl.concat_list([
            comps.list.head(pl.col("_state_i").fill_null(comps.list.len()).fill_null(0)),
            comps.list.slice(pl.col("_state_i").fill_null(comps.list.len()).fill_null(0) + 1),
        ]).list.join(" ")),
    )
    empty = pl.col("addr_clean").fill_null("") == ""
    lf = lf.with_columns(
        [pl.when(empty).then(None).otherwise(pl.col(c)).alias(c) for c in _ADDR_COLUMNS]
    ).with_columns(
        [pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
         for c in ("addr_house_number", "addr_unit", "addr_landmark", "addr_state_raw", "addr_core", "name_domain_stem")]
    )
    return lf.drop("_addr_c0", "_addr_nl", "_name_base", "_has_latin", "_state_i")
