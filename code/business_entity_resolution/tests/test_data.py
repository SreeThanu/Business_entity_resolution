"""Data-layer tests. Require scripts 01 and 02 to have been run."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ber import config, data
from ber.splits import is_val

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Data lines (header excluded); identical to eda/EDA_REPORT.md section 1.2.
EXPECTED_ROWS = {
    ("train", 1): 2_206_821,
    ("train", 2): 5_034_616,
    ("train", 3): 5_285_603,
    ("test", 1): 1_732_544,
    ("test", 2): 4_887_273,
    ("test", 3): 5_082_316,
}
N_GT = 2_206_821
N_SINGLETONS = 123_247
N_PAIRS = 7_638_365

pytestmark = pytest.mark.skipif(
    not config.PARQUET_DIR.exists(), reason="run scripts/01_convert_to_parquet.py first"
)


def count_data_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


@pytest.fixture(scope="module")
def gt_long():
    return data.load_ground_truth(long=True)


@pytest.fixture(scope="module")
def train_s1_ids() -> set[str]:
    return set(data.load_source("train", 1, columns=["entity_id"])["entity_id"])


# --- row counts ---------------------------------------------------------------

@pytest.mark.parametrize("split,source", sorted(EXPECTED_ROWS))
def test_row_counts(split, source):
    n = pq.read_metadata(config.parquet_source_path(split, source)).num_rows
    assert n == EXPECTED_ROWS[(split, source)]
    assert n == count_data_lines(config.raw_source_path(split, source))


def test_ground_truth_row_counts(gt_long):
    assert pq.read_metadata(config.parquet_ground_truth_path(long=False)).num_rows == N_GT
    assert len(gt_long) == N_PAIRS + N_SINGLETONS
    assert gt_long["matched_entity_id"].isna().sum() == N_SINGLETONS


# --- IDs and schema -----------------------------------------------------------

@pytest.mark.parametrize("split,source", sorted(EXPECTED_ROWS))
def test_ids_unique_and_source_column(split, source):
    df = data.load_source(split, source, columns=["entity_id", "source"])
    assert df["entity_id"].is_unique
    assert (df["source"] == source).all()
    assert df["entity_id"].str.startswith(f"S{source}-").all()


def test_text_kept_verbatim():
    # Raw line 37966 of test_source1.tsv is RFC-4180 style: "\"\"\"ehpad Club SAS\"".
    # With QUOTE_NONE the field is stored exactly as written in the file.
    df = data.load_source("test", 1, columns=["entity_id", "business_name"]).set_index("entity_id")
    assert df.loc["S1-179567829", "business_name"] == '"""ehpad Club SAS"'


# --- ground truth consistency -------------------------------------------------

def test_every_train_s1_in_ground_truth_once(train_s1_ids):
    wide = data.load_ground_truth(long=False)
    assert wide["source1_entity_id"].is_unique
    assert set(wide["source1_entity_id"]) == train_s1_ids


def test_ground_truth_ids_exist_in_train(gt_long):
    s2 = set(data.load_source("train", 2, columns=["entity_id"])["entity_id"])
    s3 = set(data.load_source("train", 3, columns=["entity_id"])["entity_id"])
    matched = gt_long["matched_entity_id"].dropna()
    assert matched.isin(s2 | s3).all()
    assert not gt_long.dropna().duplicated().any()


# --- splits -------------------------------------------------------------------

SPLIT_NAMES = [
    "train_ids", "val_ids",
    "fold_us_train", "fold_us_val", "fold_india_train", "fold_india_val",
]


@pytest.fixture(scope="module")
def splits() -> dict[str, set[str]]:
    return {n: set(data.load_split_ids(n)) for n in SPLIT_NAMES}


def test_train_val_disjoint_and_complete(splits, train_s1_ids):
    assert not splits["train_ids"] & splits["val_ids"]
    assert splits["train_ids"] | splits["val_ids"] == train_s1_ids


def test_folds_consistent(splits):
    for c in ("us", "india"):
        assert splits[f"fold_{c}_train"] <= splits["train_ids"]
        assert splits[f"fold_{c}_val"] <= splits["val_ids"]
        assert not splits[f"fold_{c}_train"] & splits[f"fold_{c}_val"]
    assert not splits["fold_us_train"] & splits["fold_india_val"]
    assert not splits["fold_india_train"] & splits["fold_us_val"]
    assert splits["fold_us_train"] | splits["fold_india_train"] == splits["train_ids"]
    assert splits["fold_us_val"] | splits["fold_india_val"] == splits["val_ids"]


def test_val_rate(splits, train_s1_ids):
    assert 0.19 < len(splits["val_ids"]) / len(train_s1_ids) < 0.21


def test_is_val_matches_md5_rule():
    for eid in ["S1-965667", "S1-55344266", "S1-925783039"]:
        expected = int(hashlib.md5(eid.encode()).hexdigest(), 16) % 100 < 20
        assert is_val(eid) is expected


def test_frozen_splits_match_committed_checksums():
    ref = config.SPLITS_DIR / "splits.sha256"
    for line in ref.read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((config.SPLITS_DIR / name).read_bytes()).hexdigest() == digest, name


def test_splits_reproducible(tmp_path):
    """Run the split script twice, in fresh processes with different hash seeds."""
    outs = []
    for seed in ("1", "2"):
        out = tmp_path / f"run{seed}"
        env = {**os.environ, "PYTHONHASHSEED": seed}
        subprocess.run(
            [sys.executable, str(SCRIPTS / "02_make_splits.py"), "--out", str(out), "--quiet"],
            check=True, env=env,
        )
        outs.append(out)
    for name in SPLIT_NAMES:
        a = (outs[0] / f"{name}.txt").read_bytes()
        b = (outs[1] / f"{name}.txt").read_bytes()
        assert a == b, name
        assert a == config.split_ids_path(name).read_bytes(), f"{name} differs from frozen copy"
