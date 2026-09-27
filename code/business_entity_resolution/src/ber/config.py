"""All project paths, relative to a single ROOT.

ROOT is the project root (the directory holding code/, data_raw/, data/, output/).
It defaults to four levels above this file. Every location can be moved with an environment
variable; the defaults are the Mac layout. Nothing else in the codebase should build paths by hand.

    BER_ROOT             project root                                  default: four levels above this file
    BER_DATA_DIR         local working copy of the data (fast disk)    default: ROOT/data
    BER_PERSIST_DIR      checkpoints + artifacts that must survive a   default: ROOT/artifacts
                         Colab disconnect (Google Drive on Colab)
    BER_OUTPUT_DIR       submission files                              default: ROOT/output
    BER_RAW_DATASET_DIR  the official dataset/ folder (train/, test/)  default: ROOT/data_raw/dataset
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4])).resolve()

# Raw, untouched copy of the official download (never modified): the folder holding train/ and
# test/ TSVs. On the Mac, data_raw is a symlink to the official student_resource/ folder on the
# external SSD; elsewhere set BER_RAW_DATASET_DIR.
RAW_DATASET_DIR: Path = Path(os.environ.get("BER_RAW_DATASET_DIR", ROOT / "data_raw" / "dataset"))
CHECKSUM_DIR: Path = ROOT / "checksums"

# Derived data.
DATA_DIR: Path = Path(os.environ.get("BER_DATA_DIR", ROOT / "data"))
PARQUET_DIR: Path = DATA_DIR / "parquet"
SPLITS_DIR: Path = DATA_DIR / "splits"

# Cleaned data (scripts/03_clean.py) and the lookup tables learned from train pairs.
CLEAN_DIR: Path = DATA_DIR / "clean"
DICTS_DIR: Path = DATA_DIR / "dicts"

# Submission artefacts.
OUTPUT_DIR: Path = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))

# Checkpoints and artifacts (feature chunks, models, prediction chunks). On Colab: Google Drive.
PERSIST_DIR: Path = Path(os.environ.get("BER_PERSIST_DIR", ROOT / "artifacts"))
FEATURES_DIR: Path = PERSIST_DIR / "features"
MODELS_DIR: Path = PERSIST_DIR / "models"
PREDS_DIR: Path = PERSIST_DIR / "preds"

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


def stage2_source_path(split: str, source: int) -> Path:
    return CLEAN_DIR / f"stage2_{split}_source{source}.parquet"


# ---------------------------------------------------------------------------------------------
# Blocking (src/ber/blocking.py, scripts/06_block.py). See BLOCKING.md for how these were chosen.
# ---------------------------------------------------------------------------------------------

CAND_DIR: Path = DATA_DIR / "cand"                 # candidates + vector cache (on the T7)
# Where resumable outputs go: chunk parts, the P4 cache and the final candidate files. Defaults to
# CAND_DIR; on Colab point it at Google Drive so a disconnect loses nothing (the vector cache stays
# in CAND_DIR on the fast local disk).
CAND_PERSIST_DIR: Path = Path(os.environ.get("BER_CAND_PERSIST_DIR", str(CAND_DIR)))
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



def cand_path(name: str) -> Path:
    return CAND_DIR / f"{name}_candidates.parquet"


# ---------------------------------------------------------------------------------------------
# Features, model, decision rule (scripts/07-10)
# ---------------------------------------------------------------------------------------------

# Feature split -> (candidate file name, entity split whose data/clean files the pairs join to).
FEATURE_SPLITS = {"train": ("cand_train_200k", "train"), "val": ("cand_val_50k", "train"), "test": ("test", "test")}
FEATURE_CHUNK_S1 = int(os.environ.get("BER_FEATURE_CHUNK_S1", 20_000))  # S1 per feature chunk (~1.8M pairs)
THRESHOLD_GRID = [round(0.02 * i, 2) for i in range(5, 50)]               # 0.10 .. 0.98
