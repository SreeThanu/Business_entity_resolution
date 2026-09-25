"""Data loading. This is the only module anyone should use to load data.

All loaders read the parquet files produced by scripts/01_convert_to_parquet.py and the
frozen ID lists produced by scripts/02_make_splits.py. Text is returned exactly as it
appears in the raw TSVs (no cleaning).

Source tables have columns: entity_id, business_name, business_address, country, source.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ber import config

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country", "source"]


def load_source(split: str, source: int, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load one source table.

    Args:
        split: "train" or "test".
        source: 1, 2 or 3.
        columns: optional subset of SOURCE_COLUMNS to read (faster, less memory).

    Returns:
        One row per entity, in raw file order.
    """
    if split not in config.SPLITS:
        raise ValueError(f"split must be one of {config.SPLITS}, got {split!r}")
    if source not in config.SOURCES:
        raise ValueError(f"source must be one of {config.SOURCES}, got {source!r}")
    return pd.read_parquet(config.parquet_source_path(split, source), columns=list(columns) if columns else None)


def load_ground_truth(long: bool = True) -> pd.DataFrame:
    """Load the training ground truth.

    Args:
        long: if True (default), one row per (source1_entity_id, matched_entity_id) pair;
            S1 entities with no match appear once with matched_entity_id = null.
            If False, the raw wide table: source1_entity_id, matched_entity_ids
            (comma-separated, empty string for no match).
    """
    return pd.read_parquet(config.parquet_ground_truth_path(long=long))


def load_split_ids(name: str) -> list[str]:
    """Load a frozen ID list from data/splits/, e.g. "train_ids", "val_ids", "fold_us_train".

    IDs are train source-1 entity_ids, one per line, sorted.
    """
    path = config.split_ids_path(name)
    if not path.exists():
        available = sorted(p.stem for p in config.SPLITS_DIR.glob("*.txt"))
        raise FileNotFoundError(f"no split named {name!r}; available: {available}")
    return path.read_text().split()
