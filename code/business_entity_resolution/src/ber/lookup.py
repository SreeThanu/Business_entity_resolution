"""Stage 2: state-slot equivalence table learned from TRAINING true pairs only.

Stage 1 moves the state/region clause into addr_state_raw without canonicalising it. The sources
spell it differently (EDA 5.3): US S1/S2 codes vs S3 full names; India S1 full names vs S3 codes
("mh") vs S2 native script ("महाराष्ट्र"). This module learns variant -> S1 spelling from aligned true
pairs and writes it to a NEW column, addr_state_canon. Stage 1 columns are never modified.

France has no labels: its rows get addr_state_canon = null (unknown), never a mismatch.

LEAKAGE RULE: only pairs whose S1 entity is in data/splits/train_ids.txt are used.
`training_pairs` enforces it and raises if a validation ID gets through.

Table (data/dicts/state_map.parquet): country, variant, canonical, n, share.
"""

from __future__ import annotations

import hashlib

import polars as pl

from ber import config
from ber.normalize import NORMALIZE_VERSION

STATE_MIN_COUNT = 100   # pairs supporting variant -> canonical
STATE_MIN_SHARE = 0.90  # of all aligned pairs where the variant appears
LEARN_COUNTRIES = ("US", "India")  # countries with training labels

STATE_MAP_FILE = "state_map.parquet"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_pairs(gt_long: pl.DataFrame, train_ids: list[str], val_ids: list[str]) -> pl.DataFrame:
    """True (S1, S2/S3) pairs restricted to train_ids. Raises if any validation S1 ID remains."""
    train, val = set(train_ids), set(val_ids)
    if train & val:
        raise ValueError(f"train_ids and val_ids overlap ({len(train & val)} IDs)")
    pairs = gt_long.drop_nulls("matched_entity_id").filter(pl.col("source1_entity_id").is_in(train))
    leaked = pairs.filter(pl.col("source1_entity_id").is_in(val)).height
    if leaked:
        raise ValueError(f"{leaked} validation pairs in the learning set")
    return pairs.rename({"source1_entity_id": "s1_id", "matched_entity_id": "other_id"})


def learning_metadata(n_pairs: int, **extra) -> dict[str, str]:
    return {
        "NORMALIZE_VERSION": NORMALIZE_VERSION,
        "learned_from": "true pairs whose S1 entity is in data/splits/train_ids.txt (val_ids never used)",
        "train_ids_sha256": _sha256(config.split_ids_path("train_ids")),
        "n_training_pairs": str(n_pairs),
        **{k: str(v) for k, v in extra.items()},
    }


def learn_state_map(aligned: pl.DataFrame) -> pl.DataFrame:
    """aligned: country, s1_state, other_state (one row per train pair, single-clause states only).

    Returns (country, variant, canonical, n, share). S1 spellings map to themselves.
    """
    counts = aligned.group_by("country", "other_state", "s1_state").len("n")
    total = aligned.group_by("country", "other_state").len("n_variant")
    best = (counts.join(total, on=["country", "other_state"])
            .with_columns(share=pl.col("n") / pl.col("n_variant"))
            .sort("n", descending=True).group_by("country", "other_state", maintain_order=True).first()
            .filter((pl.col("n") >= STATE_MIN_COUNT) & (pl.col("share") >= STATE_MIN_SHARE)))
    s1_vals = (aligned.group_by("country", "s1_state").len("n").filter(pl.col("n") >= STATE_MIN_COUNT)
               .select("country", pl.col("s1_state").alias("variant"), pl.col("s1_state").alias("canonical"),
                       "n", pl.lit(1.0).alias("share")))
    learned = best.select("country", pl.col("other_state").alias("variant"), pl.col("s1_state").alias("canonical"),
                          "n", "share")
    return (pl.concat([s1_vals, learned.join(s1_vals.select("country", "variant"), on=["country", "variant"], how="anti")])
            .sort("country", "n", descending=[False, True]))


def apply_state_map(lf: pl.LazyFrame, state_map: pl.DataFrame) -> pl.LazyFrame:
    """entity_id + addr_state_canon. Null when the state is unknown, unmapped, ambiguous (several
    clauses) or the country has no training labels (France)."""
    out = pl.lit(None, pl.String)
    for (country,), g in state_map.group_by("country"):
        mapped = pl.col("addr_state_raw").replace_strict(g["variant"].to_list(), g["canonical"].to_list(),
                                                         default=None, return_dtype=pl.String)
        out = pl.when(pl.col("country") == country).then(mapped).otherwise(out)
    return lf.select("entity_id", out.alias("addr_state_canon"))
