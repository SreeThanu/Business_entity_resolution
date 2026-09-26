"""All project paths, relative to a single ROOT.

ROOT is the project root (the directory holding code/, data_raw/, data/, output/).
It defaults to four levels above this file and can be overridden with the BER_ROOT
environment variable. Nothing else in the codebase should build paths by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4])).resolve()

# Raw, untouched copy of the official download (never modified). It lives on the external SSD;
# override with BER_RAW_DATASET_DIR (the folder holding train/ and test/ TSVs).
RAW_DATASET_DIR: Path = Path(os.environ.get(
    "BER_RAW_DATASET_DIR", "/Volumes/thanu's T7/Business_entity_resolution/student_resource/dataset"))
CHECKSUM_DIR: Path = ROOT / "checksums"

# Derived data.
DATA_DIR: Path = ROOT / "data"
PARQUET_DIR: Path = DATA_DIR / "parquet"
SPLITS_DIR: Path = DATA_DIR / "splits"

# Cleaned data (scripts/03_clean.py) and the lookup tables learned from train pairs.
CLEAN_DIR: Path = DATA_DIR / "clean"
DICTS_DIR: Path = DATA_DIR / "dicts"

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


def clean_source_path(split: str, source: int) -> Path:
    return CLEAN_DIR / f"{split}_source{source}.parquet"


# ---------------------------------------------------------------------------------------------
# Blocking (src/ber/blocking.py, scripts/05_block.py). See BLOCKING.md for how these were chosen.
# ---------------------------------------------------------------------------------------------

CAND_DIR: Path = DATA_DIR / "cand"                 # candidates + vector cache (on the T7)
BLOCK_N_FEATURES = 2 ** 24                         # hashed char_wb 3-5-gram space (as in the EDA)
BLOCK_SHARD_ROWS = 500_000                         # rows per cached vector shard
BLOCK_QUERY_CHUNK = 250_000                        # S1 queries processed together (memory bound)
# Retrieval uses only n-grams whose document frequency is <= this fraction of the split's documents;
# the retrieved set is then re-ranked by the exact cosine over all n-grams.
BLOCK_DF_CAP = 0.005
BLOCK_RETRIEVE_M = {"p1": 300, "p2": 300, "p4": 100}  # rows retrieved per query per shard before the exact re-rank
BLOCK_KMAX = {"p1": 200, "p2": 100, "p4": 20}      # collected in --mode dev for the recall-vs-k curves
BLOCK_K = {"p1": 50, "p2": 20, "p4": 10}           # used in --mode full (final candidate set)
BLOCK_P4_MAX_PER_S1 = 100                          # P4 pairs kept per S1 (hub S1 records, see BLOCKING.md)
BLOCK_P3_MAX_BLOCK = 20                            # P3 key blocks with more pool records are dropped
BLOCK_DEV_QUERIES = 20_000                         # S1 queries sampled in --mode dev
BLOCK_SEED = 0


def cand_path(split: str) -> Path:
    return CAND_DIR / f"{split}_candidates.parquet"
