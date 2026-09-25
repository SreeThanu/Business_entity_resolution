"""All project paths, relative to a single ROOT.

ROOT is the project root (the directory holding code/, data_raw/, data/, output/).
It defaults to four levels above this file and can be overridden with the BER_ROOT
environment variable. Nothing else in the codebase should build paths by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4])).resolve()

# Raw, untouched copy of the fresh download (never modified).
RAW_DIR: Path = ROOT / "data_raw"
RAW_DATASET_DIR: Path = RAW_DIR / "dataset"
CHECKSUM_DIR: Path = ROOT / "checksums"

# Derived data.
DATA_DIR: Path = ROOT / "data"
PARQUET_DIR: Path = DATA_DIR / "parquet"
SPLITS_DIR: Path = DATA_DIR / "splits"

# Submission artefacts.
OUTPUT_DIR: Path = ROOT / "output"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def raw_source_path(split: str, source: int) -> Path:
    return RAW_DATASET_DIR / split / f"{split}_source{source}.tsv"


def raw_ground_truth_path() -> Path:
    return RAW_DATASET_DIR / "train" / "train_ground_truth.tsv"


def parquet_source_path(split: str, source: int) -> Path:
    return PARQUET_DIR / f"{split}_source{source}.parquet"


def parquet_ground_truth_path(long: bool) -> Path:
    return PARQUET_DIR / ("train_ground_truth_long.parquet" if long else "train_ground_truth.parquet")


def split_ids_path(name: str) -> Path:
    return SPLITS_DIR / f"{name}.txt"
