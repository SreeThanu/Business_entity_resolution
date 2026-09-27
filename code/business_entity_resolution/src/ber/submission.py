"""Submission files: output/candidate_pairs.tsv and output/matching_results.tsv.

Both have one row per test S1 (all of them, empty list when none) and comma-separated unique
S2/S3 ids. Validation runs the official utils/validate_submission.py. Only polars is needed here
(no blocking dependencies), so the Colab training environment can use it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

VALIDATOR = Path(__file__).resolve().parents[2] / "utils" / "validate_submission.py"


def _write_lists(pairs: pl.DataFrame | pl.LazyFrame, s1_ids: pl.Series, path: Path, id_col: str, list_col: str) -> None:
    lists = (pairs.lazy().select("s1_id", id_col).unique()
             .group_by("s1_id").agg(pl.col(id_col).sort().str.join(",").alias(list_col))
             .collect(engine="streaming"))
    out = (pl.DataFrame({"s1_id": s1_ids}).unique(maintain_order=True).join(lists, on="s1_id", how="left")
           .select(pl.col("s1_id").alias("source1_entity_id"), pl.col(list_col).fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    out.write_csv(tmp, separator="\t", quote_style="never")
    tmp.replace(path)


def write_candidate_tsv(cands: pl.DataFrame | pl.LazyFrame, s1_ids: pl.Series, path: Path) -> None:
    """Official candidate_pairs.tsv from candidate rows (s1_id, cand_id)."""
    _write_lists(cands, s1_ids, path, "cand_id", "candidate_entity_ids")


def write_matching_tsv(matches: pl.DataFrame | pl.LazyFrame, s1_ids: pl.Series, path: Path) -> None:
    """Official matching_results.tsv from predicted pairs (s1_id, cand_id)."""
    _write_lists(matches, s1_ids, path, "cand_id", "matched_entity_ids")


def _validator_module():
    spec = importlib.util.spec_from_file_location("validate_submission", VALIDATOR)
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    return v


def validate_candidate_tsv(path: Path, required: set[str]) -> list[str]:
    """Run the official validator's rules (utils/validate_submission.py) on a candidate file."""
    v = _validator_module()
    errors: list[str] = []
    v.validate_id_list_file(str(path), v.CANDIDATE_HEADER, "candidate_entity_ids", required, None, errors)
    return errors


def run_official_validator(matching: Path, candidate: Path | None, s1_ids: pl.Series) -> int:
    """Run utils/validate_submission.py as the organisers ship it; returns its exit code.

    The validator reads the required S1 ids from <test-dir>/test_source1.tsv (first column). The
    raw TSVs are not on Colab, so a temporary test dir with just that column is written from the
    ids passed in (the cleaned test S1 table has exactly the raw ids, in raw order).
    """
    with tempfile.TemporaryDirectory() as d:
        pl.DataFrame({"entity_id": s1_ids}).write_csv(Path(d) / "test_source1.tsv", separator="\t", quote_style="never")
        cmd = [sys.executable, str(VALIDATOR), "--matching", str(matching), "--test-dir", d]
        if candidate is not None:
            cmd += ["--candidate", str(candidate)]
        return subprocess.run(cmd).returncode
