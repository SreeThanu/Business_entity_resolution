"""Shared helpers for the EDA scripts.

All raw reads go through `read_tsv`, which uses the mandated call
    pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
and verifies the column count. Parsed frames are cached as parquet in
eda/cache/ (pyarrow-backed strings, far smaller in RAM than object dtype).
The files under dataset/ are never modified.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
EDA = ROOT / "eda"
CACHE = EDA / "cache"
OUT = EDA / "out"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
SEED = 42


def path_of(split: str, name: str) -> Path:
    """name in {'source1','source2','source3','ground_truth'}"""
    return DATA / split / f"{split}_{name}.tsv"


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = GT_COLS if "ground_truth" in path.name else SRC_COLS
    if list(df.columns) != expected:
        raise ValueError(f"{path}: columns {list(df.columns)} != {expected}")
    return df


def load(split: str, name: str) -> pd.DataFrame:
    """Load a file (cached). Columns are pyarrow strings."""
    cp = CACHE / f"{split}_{name}.parquet"
    if not cp.exists():
        df = read_tsv(path_of(split, name))
        df.to_parquet(cp, index=False)
        del df
    return pd.read_parquet(cp, dtype_backend="pyarrow")


def load_sources(split: str, cols=None) -> pd.DataFrame:
    """S1+S2+S3 concatenated with a 'source' column (S1/S2/S3)."""
    parts = []
    for i in (1, 2, 3):
        d = load(split, f"source{i}")
        if cols is not None:
            d = d[cols]
        d = d.assign(source=f"S{i}")
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["source"] = out["source"].astype("category")
    return out


def load_pairs() -> pd.DataFrame:
    """Ground truth exploded to one row per (s1, match). Cached."""
    cp = CACHE / "train_gt_pairs.parquet"
    if not cp.exists():
        gt = load("train", "ground_truth")
        s = gt["matched_entity_ids"].astype(str)
        lst = s.str.split(",")
        p = pd.DataFrame({"s1": gt["source1_entity_id"].astype(str).repeat(lst.str.len()).values,
                          "match": np.concatenate(lst.values)})
        p = p[p["match"].str.strip() != ""].reset_index(drop=True)
        p.to_parquet(cp, index=False)
    return pd.read_parquet(cp)


PAIR_SAMPLE = 1_000_000


def load_pair_records() -> pd.DataFrame:
    """Random sample of PAIR_SAMPLE true pairs (seed 42) joined with both records:
    s1_*, m_* columns + src (S2/S3). Cached. (All 7.6M pairs with text do not fit in 8 GB RAM.)"""
    cp = CACHE / "train_gt_pair_records.parquet"
    if not cp.exists():
        pairs = load_pairs().sample(PAIR_SAMPLE, random_state=SEED).reset_index(drop=True)
        s1 = load("train", "source1")
        s1["entity_id"] = s1["entity_id"].astype(str)
        s1 = s1[s1["entity_id"].isin(pd.Index(pairs["s1"].unique()))].astype(str)
        s1.columns = ["s1", "s1_name", "s1_addr", "s1_country"]
        need = pd.Index(pairs["match"].unique())
        parts = []
        for i in (2, 3):
            d = load("train", f"source{i}")
            d["entity_id"] = d["entity_id"].astype(str)
            d = d[d["entity_id"].isin(need)].astype(str)
            parts.append(d)
        m = pd.concat(parts, ignore_index=True)
        m.columns = ["match", "m_name", "m_addr", "m_country"]
        out = pairs.merge(s1, on="s1", how="left").merge(m, on="match", how="left")
        out["src"] = out["match"].str[:2]
        out.to_parquet(cp, index=False)
    return pd.read_parquet(cp)


# ----------------------------------------------------------------- text utils
LEGAL_SUFFIXES = {
    # India / US / UK
    "pvt", "private", "ltd", "limited", "llp", "inc", "incorporated", "corp",
    "corporation", "llc", "co", "company", "plc", "lp", "pllc", "pc", "opc",
    "pte", "lc", "ltda",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "scp", "earl", "gie",
}
# keep word chars AND combining marks (Indic vowel signs are category M, not \w in Python re)
_punct_re = re.compile(r"[^\w\s\u0300-\u036f\u0900-\u0dff\u200c\u200d]", re.UNICODE)
_ws_re = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    """Remove combining accents from Latin letters only (Indic vowel signs are kept)."""
    if s.isascii():
        return s
    out, prev_latin = [], False
    for c in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(c) and prev_latin:
            continue
        out.append(c)
        if not unicodedata.combining(c):
            prev_latin = ord(c) < 0x250
    return unicodedata.normalize("NFC", "".join(out))


def norm_basic(s: str) -> str:
    """lowercase + punctuation->space + collapse whitespace."""
    s = _punct_re.sub(" ", s.lower()).replace("_", " ")
    return _ws_re.sub(" ", s).strip()


def norm_name(s: str) -> str:
    """norm_basic + '&'->'and' + drop legal-suffix tokens."""
    s = s.lower().replace("&", " and ")
    toks = [t for t in norm_basic(s).split() if t not in LEGAL_SUFFIXES]
    return " ".join(toks)


def script_of(s: str) -> str:
    """Rough script bucket of a string: ascii / latin-accented / non-latin."""
    if s.isascii():
        return "ascii"
    for ch in s:
        if ord(ch) > 0x24F and ch.isalpha():
            return "non-latin"
    return "latin-ext"


# ----------------------------------------------------------------- postal codes
POSTAL_PATTERNS = {
    "US": re.compile(r"(?<!\d)(\d{5})(?:-(\d{4}))?(?!\d)"),
    "India": re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)"),
    "France": re.compile(r"(?<!\d)(\d{5})(?!\d)"),
}


def extract_postal(addr: str, country: str) -> str:
    """Last postal-looking token for the country ('' if none).

    US: 5-digit (ZIP+4 collapsed to 5); India: 6-digit (allows 'ddd ddd');
    France / unknown: 5-digit.
    """
    pat = POSTAL_PATTERNS.get(country, POSTAL_PATTERNS["France"])
    m = pat.findall(addr)
    if not m:
        return ""
    last = m[-1]
    if country == "India":
        return last[0] + last[1]
    return last[0] if isinstance(last, tuple) else last


# ----------------------------------------------------------------- markdown
def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = False, maxw: int = 90) -> str:
    d = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v) if np.isfinite(v) else ""
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        s = str(v).replace("|", "\\|").replace("\n", " ")
        return s if len(s) <= maxw else s[: maxw - 1] + "…"

    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_section(num: str, text: str) -> None:
    (OUT / f"section{num}.md").write_text(text, encoding="utf-8")
