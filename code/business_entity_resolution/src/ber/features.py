"""Pair features for the matcher (scripts/07_features.py). v0 baseline.

v0 BASELINE: a first, deliberately simple feature set so the Colab pipeline runs end to end.
It may be replaced wholesale; bump FEATURE_VERSION on any change (07 refuses to mix versions).

Inputs: a chunk of candidate pairs (BLOCKING.md schema) and the cleaned entity tables
(CLEANING.md, "Cleaned data contract"). Null semantics follow the contract: a null on either side
means UNKNOWN, so the feature is null (LightGBM treats it as missing), never a mismatch.

Country is used only to count chain names within a country; it is never a feature (France is
test-only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ber import config
from ber.normalize import NORMALIZE_VERSION

FEATURE_VERSION = "v0.1"

ENTITY_COLUMNS = ["entity_id", "country", "name_core", "name_nospace", "name_domain_stem", "legal_families",
                  "name_script", "name_has_nonlatin", "addr_core", "addr_house_number", "addr_numbers"]

FEATURES = [
    # name
    "name_tsr", "name_jw", "name_3g_jac", "name_tok_jac", "name_core_eq", "name_domain_match", "name_any_domain",
    "name_tok_added", "name_tok_dropped", "a_name_ntok", "b_name_ntok",
    # legal families
    "legal_agree", "legal_conflict", "legal_unknown",
    # address
    "addr_tok_jac", "addr_3g_jac", "a_addr_missing", "b_addr_missing",
    "hn_exact", "hn_gap", "a_hn_missing", "b_hn_missing",
    "addr_num_overlap", "addr_num_jac",
    "state_agree", "state_disagree", "state_unknown",
    # script
    "a_nonlatin", "b_nonlatin", "script_same",
    # blocking
    "in_p1", "in_p2", "in_p3", "in_p4", "n_passes", "p1_cos", "p2_cos", "p1_rank", "p2_rank", "p1_rank_rev",
    "mutual_best", "s1_n_cands", "p1_cos_rank_s1", "p2_cos_rank_s1", "p1_cos_gap_s1", "b_is_s3",
    # chain names
    "a_chain_n", "b_chain_n",
]


# ---------------------------------------------------------------------------------------------
# Entity tables
# ---------------------------------------------------------------------------------------------

@dataclass
class Entities:
    s1: pl.DataFrame     # S1 rows of the split + a_chain_n
    pool: pl.DataFrame   # S2 + S3 rows of the split + b_chain_n


def check_normalize_version(path) -> None:
    meta = pq.ParquetFile(path).metadata.metadata or {}
    found = meta.get(b"NORMALIZE_VERSION", b"?").decode()
    if found != NORMALIZE_VERSION:
        raise RuntimeError(f"{path}: NORMALIZE_VERSION {found}, code expects {NORMALIZE_VERSION}; re-run 03/05")


def _source(split: str, source: int) -> pl.LazyFrame:
    check_normalize_version(config.clean_source_path(split, source))
    st = pl.scan_parquet(config.stage2_source_path(split, source)).select("entity_id", "addr_state_canon")
    return pl.scan_parquet(config.clean_source_path(split, source)).select(ENTITY_COLUMNS).join(st, on="entity_id", how="left")


def chain_counts(s1: pl.DataFrame) -> pl.DataFrame:
    """How many S1 of the split share each name_core, within country."""
    return s1.group_by("country", "name_core").agg(pl.len().cast(pl.Int32).alias("chain_n"))


def build_entities(s1: pl.DataFrame, pool: pl.DataFrame) -> Entities:
    counts = chain_counts(s1)
    s1 = s1.join(counts.rename({"chain_n": "a_chain_n"}), on=["country", "name_core"], how="left")
    pool = pool.join(counts.rename({"chain_n": "b_chain_n"}), on=["country", "name_core"], how="left").with_columns(
        pl.col("b_chain_n").fill_null(0))
    return Entities(s1, pool)


def load_entities(split: str) -> Entities:
    """Every S1 and S2/S3 row of an entity split ("train" or "test"), contract columns only."""
    s1 = _source(split, 1).collect()
    pool = pl.concat([_source(split, s) for s in (2, 3)]).collect()
    return build_entities(s1, pool)


# ---------------------------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------------------------

def _grams(col: str) -> pl.Expr:
    """Set of overlapping character 3-grams (three non-overlapping scans at offsets 0, 1, 2)."""
    return pl.concat_list([pl.col(col).str.slice(i).str.extract_all(r"(?s).{3}") for i in range(3)]).list.unique()


def _tokens(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def _jaccard(x: str, y: str) -> pl.Expr:
    """|x & y| / |x | y| for two columns of SETS (unique lists); null if either is null or both empty."""
    inter = pl.col(x).list.set_intersection(pl.col(y)).list.len()
    union = pl.col(x).list.len() + pl.col(y).list.len() - inter
    return pl.when(union > 0).then(inter / union).cast(pl.Float32)


def _derive(e: pl.DataFrame) -> pl.DataFrame:
    """Per-entity sets and parsed values, computed once per entity instead of once per pair."""
    return e.with_columns(
        g_name=_grams("name_core"), t_name=_tokens("name_core"),
        g_addr=_grams("addr_core"), t_addr=_tokens("addr_core"),
        hn_int=pl.col("addr_house_number").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False),
        num_set=pl.col("addr_numbers").list.unique(),
    )


def _both(x: str, y: str) -> pl.Expr:
    return pl.col(x).is_not_null() & pl.col(y).is_not_null()


def pair_features(pairs: pl.DataFrame, ents: Entities) -> pl.DataFrame:
    """Features for candidate pairs (s1_id, cand_id, country, blocking columns).

    Per-S1 features (s1_n_cands, *_rank_s1, p1_cos_gap_s1) need every candidate of an S1 in the same
    call; 07_features.py chunks by S1, so that holds.
    Returns s1_id, cand_id, country, *FEATURES.
    """
    a = _derive(ents.s1.join(pairs.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="semi"))
    b = _derive(ents.pool.join(pairs.select(pl.col("cand_id").alias("entity_id")).unique(), on="entity_id", how="semi"))
    a = a.select(pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias(f"a_{c}") for c in a.columns
                                                       if c not in ("entity_id", "country", "a_chain_n")], "a_chain_n")
    b = b.select(pl.col("entity_id").alias("cand_id"), *[pl.col(c).alias(f"b_{c}") for c in b.columns
                                                         if c not in ("entity_id", "country", "b_chain_n")], "b_chain_n")
    df = pairs.join(a, on="s1_id", how="left").join(b, on="cand_id", how="left")
    missing = df.filter(pl.col("a_name_core").is_null() | pl.col("b_name_core").is_null())
    if missing.height:
        raise ValueError(f"{missing.height} pairs reference ids missing from the entity tables, e.g. "
                         f"{missing.select('s1_id', 'cand_id').head(3).rows()}")

    # rapidfuzz, vectorised over pairs (C++, all cores)
    an, bn = df["a_name_core"].to_list(), df["b_name_core"].to_list()
    tsr = process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    jw = process.cpdist(an, bn, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)

    k = config.BLOCK_K
    df = df.with_columns(
        name_tsr=pl.Series(tsr, dtype=pl.Float32), name_jw=pl.Series(jw, dtype=pl.Float32),
        _lf_inter=pl.col("a_legal_families").list.set_intersection(pl.col("b_legal_families")).list.len(),
        _lf_both=(pl.col("a_legal_families").list.len() > 0) & (pl.col("b_legal_families").list.len() > 0),
    )
    domain = ((pl.col("a_name_nospace") == pl.col("b_name_domain_stem")).fill_null(False)
              | (pl.col("b_name_nospace") == pl.col("a_name_domain_stem")).fill_null(False)
              | (pl.col("a_name_domain_stem") == pl.col("b_name_domain_stem")).fill_null(False))
    both_state = _both("a_addr_state_canon", "b_addr_state_canon")
    both_hn = _both("a_addr_house_number", "b_addr_house_number")
    both_addr = _both("a_addr_core", "b_addr_core")
    both_num = _both("a_addr_numbers", "b_addr_numbers")
    out = df.select(
        "s1_id", "cand_id", "country",
        "name_tsr", "name_jw",
        name_3g_jac=_jaccard("a_g_name", "b_g_name"),
        name_tok_jac=_jaccard("a_t_name", "b_t_name"),
        name_core_eq=pl.col("a_name_core") == pl.col("b_name_core"),
        name_domain_match=domain,
        name_any_domain=pl.col("a_name_domain_stem").is_not_null() | pl.col("b_name_domain_stem").is_not_null(),
        name_tok_added=pl.col("b_t_name").list.set_difference(pl.col("a_t_name")).list.len(),
        name_tok_dropped=pl.col("a_t_name").list.set_difference(pl.col("b_t_name")).list.len(),
        a_name_ntok=pl.col("a_t_name").list.len(), b_name_ntok=pl.col("b_t_name").list.len(),
        legal_agree=pl.col("_lf_both") & (pl.col("_lf_inter") > 0),
        legal_conflict=pl.col("_lf_both") & (pl.col("_lf_inter") == 0),
        legal_unknown=~pl.col("_lf_both"),
        addr_tok_jac=pl.when(both_addr).then(_jaccard("a_t_addr", "b_t_addr")),
        addr_3g_jac=pl.when(both_addr).then(_jaccard("a_g_addr", "b_g_addr")),
        a_addr_missing=pl.col("a_addr_core").is_null(), b_addr_missing=pl.col("b_addr_core").is_null(),
        hn_exact=pl.when(both_hn).then(pl.col("a_addr_house_number") == pl.col("b_addr_house_number")),
        hn_gap=(pl.col("a_hn_int") - pl.col("b_hn_int")).abs().cast(pl.Float32),
        a_hn_missing=pl.col("a_addr_house_number").is_null(), b_hn_missing=pl.col("b_addr_house_number").is_null(),
        addr_num_overlap=pl.when(both_num).then(pl.col("a_num_set").list.set_intersection(pl.col("b_num_set")).list.len()),
        addr_num_jac=pl.when(both_num).then(_jaccard("a_num_set", "b_num_set")),
        state_agree=both_state & (pl.col("a_addr_state_canon") == pl.col("b_addr_state_canon")),
        state_disagree=both_state & (pl.col("a_addr_state_canon") != pl.col("b_addr_state_canon")),
        state_unknown=~both_state,
        a_nonlatin=pl.col("a_name_has_nonlatin"), b_nonlatin=pl.col("b_name_has_nonlatin"),
        script_same=pl.col("a_name_script") == pl.col("b_name_script"),
        in_p1="in_p1", in_p2="in_p2", in_p3="in_p3", in_p4="in_p4",
        n_passes=pl.sum_horizontal(pl.col("in_p1", "in_p2", "in_p3", "in_p4").cast(pl.Int8)),
        p1_cos="p1_cos", p2_cos="p2_cos",
        p1_rank=pl.col("p1_rank").fill_null(k["p1"] + 1),       # null = not in that pass's list, i.e. worse than k
        p2_rank=pl.col("p2_rank").fill_null(k["p2"] + 1),
        p1_rank_rev=pl.col("p1_rank_rev").fill_null(k["p4"] + 1),
        mutual_best=(pl.col("p1_rank") == 1).fill_null(False) & (pl.col("p1_rank_rev") == 1).fill_null(False),
        s1_n_cands=pl.len().over("s1_id"),
        p1_cos_rank_s1=pl.col("p1_cos").rank("min", descending=True).over("s1_id"),
        p2_cos_rank_s1=pl.col("p2_cos").rank("min", descending=True).over("s1_id"),
        p1_cos_gap_s1=pl.col("p1_cos").max().over("s1_id") - pl.col("p1_cos"),
        b_is_s3=pl.col("cand_id").str.starts_with("S3-"),
        a_chain_n="a_chain_n", b_chain_n="b_chain_n",
    )
    # compact, uniform dtypes: flags Int8, counts Int32, similarities Float32
    casts = []
    for c, dt in out.select(FEATURES).schema.items():
        if dt == pl.Boolean:
            casts.append(pl.col(c).cast(pl.Int8))
        elif dt.is_integer():
            casts.append(pl.col(c).cast(pl.Int32))
        else:
            casts.append(pl.col(c).cast(pl.Float32))
    return out.with_columns(casts).select("s1_id", "cand_id", "country", *FEATURES)


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Float32 feature matrix, nulls as NaN (LightGBM's missing value)."""
    return df.select(pl.col(features).cast(pl.Float32)).to_numpy().astype(np.float32, copy=False)
