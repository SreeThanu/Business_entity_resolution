"""Frozen validation-split logic. Used by scripts/02_make_splits.py and the tests.

Assignment depends only on the entity_id string, via md5, so it is identical across
processes, machines and Python versions. Never use the built-in hash(): it is salted
per process.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

VAL_PERCENT = 20


def is_val(entity_id: str) -> bool:
    return int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % 100 < VAL_PERCENT


def country_slug(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def make_splits(s1: pd.DataFrame) -> dict[str, list[str]]:
    """Build every ID list from the train source-1 table (needs entity_id and country).

    Returns {name: sorted IDs}: train_ids, val_ids, and for every country C in the data
    fold_C_train (C entities in train_ids) and fold_C_val (C entities in val_ids).
    A cross-country experiment trains on fold_A_train and evaluates on fold_B_val.
    """
    val = s1["entity_id"].map(is_val)
    out = {
        "train_ids": sorted(s1.loc[~val, "entity_id"]),
        "val_ids": sorted(s1.loc[val, "entity_id"]),
    }
    for country in sorted(s1["country"].unique()):
        in_c = s1["country"] == country
        slug = country_slug(country)
        out[f"fold_{slug}_train"] = sorted(s1.loc[in_c & ~val, "entity_id"])
        out[f"fold_{slug}_val"] = sorted(s1.loc[in_c & val, "entity_id"])
    return out
