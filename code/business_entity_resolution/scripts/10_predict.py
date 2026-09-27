"""Test inference, decision rule, submission files.

    python scripts/10_predict.py                   # newest model that has a threshold (09 done)
    python scripts/10_predict.py --model <id>

1. Scores every test feature chunk (07 --split test) and writes the probabilities to
   {BER_PERSIST_DIR}/preds/test/<model id>/part_NNNNN.parquet; chunks already scored are skipped.
2. Applies the decision rule with the model's val threshold (threshold.json from 09): one-to-one
   (each S2/S3 record goes only to its highest-probability S1), then probability >= threshold.
3. Writes output/matching_results.tsv (every test S1, empty when no match) and, if missing,
   output/candidate_pairs.tsv from data/cand/test_candidates.parquet.
4. Checks that every predicted pair is in candidate_pairs.tsv (hard failure otherwise) and runs
   the official utils/validate_submission.py on both files.
5. Copies both files to {BER_PERSIST_DIR}/output/.

Validating candidate_pairs.tsv with the official script loads ~160M ids into Python sets
(~15-20 GB RAM); on a small runtime pass --skip-candidate-validation (the subset check in step 4
still runs).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import artifacts as A
from ber import config, memguard
from ber import decision as D
from ber import features as F
from ber import submission as S


def check_subset(matches: pl.DataFrame, candidate_tsv) -> int:
    """Number of predicted pairs that are NOT in candidate_pairs.tsv (streamed; only rows whose
    candidate ids were predicted are kept in memory)."""
    wanted = matches["cand_id"].unique().implode()
    cands = (pl.scan_csv(candidate_tsv, separator="\t", quote_char=None, schema={"source1_entity_id": pl.String,
                                                                               "candidate_entity_ids": pl.String})
             .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("candidate_entity_ids").str.split(",").alias("cand_id"))
             .explode("cand_id", empty_as_null=True).filter(pl.col("cand_id").is_in(wanted)).collect(engine="streaming"))
    return matches.join(cands, on=["s1_id", "cand_id"], how="anti").height


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="override the model's val threshold")
    ap.add_argument("--rewrite-candidates", action="store_true", help="rewrite output/candidate_pairs.tsv")
    ap.add_argument("--skip-candidate-validation", action="store_true")
    args = ap.parse_args()
    memguard.start()

    md_dir = A.resolve_model(args.model, need="threshold.json")
    thr = json.loads((md_dir / "threshold.json").read_text())
    threshold = args.threshold if args.threshold is not None else thr["threshold"]
    feats = json.loads((md_dir / "features.json").read_text())
    booster = lgb.Booster(model_file=str(md_dir / "model.txt"))
    A.log(f"model {md_dir.name}, threshold {threshold} (val macro-F0.5 {thr['val_macro_f05']:.4f})")

    # 1. score chunks (resumable)
    parts = A.feature_parts("test")
    pred_dir = config.PREDS_DIR / "test" / md_dir.name
    A.check_run_manifest(pred_dir, {"model_id": md_dir.name, "features": A.feature_fingerprint("test")})
    todo = [p for p in parts if not (pred_dir / p.name).exists()]
    A.log(f"test: {len(parts)} feature chunks, {len(parts) - len(todo)} scored earlier")
    for n, p in enumerate(todo, 1):
        t = time.time()
        df = pl.read_parquet(p)
        prob = booster.predict(F.to_matrix(df, feats)).astype(np.float32)
        A.atomic_write_parquet(df.select("s1_id", "cand_id", "country").with_columns(prob=pl.Series(prob)), pred_dir / p.name)
        A.log(f"{p.name}: {df.height:,} pairs scored ({time.time() - t:.0f}s); {n}/{len(todo)} this run")

    # 2. decision rule (threshold filter first: equivalent, and keeps memory small)
    preds = pl.scan_parquet([pred_dir / p.name for p in parts])
    matches = D.decide(preds, threshold).collect(engine="streaming")
    s1 = pl.read_parquet(config.clean_source_path("test", 1), columns=["entity_id", "country"])
    per = s1.join(matches.group_by("s1_id").len("n"), left_on="entity_id", right_on="s1_id", how="left").with_columns(
        pl.col("n").fill_null(0))
    stats = per.group_by("country").agg(pl.len().alias("s1"), (pl.col("n") > 0).mean().alias("share_with_match"),
                                        pl.col("n").mean().alias("matches_per_s1")).sort("country")
    A.log(f"{matches.height:,} matches for {per.filter(pl.col('n') > 0).height:,} of {s1.height:,} test S1")
    print(stats)

    # 3. submission files
    matching = config.OUTPUT_DIR / "matching_results.tsv"
    candidate = config.OUTPUT_DIR / "candidate_pairs.tsv"
    S.write_matching_tsv(matches, s1["entity_id"], matching)
    A.log(f"wrote {matching}")
    if args.rewrite_candidates or not candidate.exists():
        S.write_candidate_tsv(pl.scan_parquet(config.cand_path("test")), s1["entity_id"], candidate)
        A.log(f"wrote {candidate}")

    # 4. checks
    missing = check_subset(matches, candidate)
    if missing:
        sys.exit(f"FAIL: {missing:,} predicted pairs are not in {candidate}")
    A.log(f"every predicted pair is in {candidate.name}")
    code = S.run_official_validator(matching, None if args.skip_candidate_validation else candidate, s1["entity_id"])
    if code != 0:
        sys.exit(f"FAIL: official validator exit code {code}")

    # 5. persist
    dest = config.PERSIST_DIR / "output"
    dest.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != config.OUTPUT_DIR.resolve():
        shutil.copyfile(matching, dest / f"matching_results_{md_dir.name}.tsv")
        shutil.copyfile(matching, dest / "matching_results.tsv")
        if not (dest / candidate.name).exists() or (dest / candidate.name).stat().st_size != candidate.stat().st_size:
            shutil.copyfile(candidate, dest / candidate.name)
    A.atomic_write_json({"model_id": md_dir.name, "threshold": threshold, "matches": matches.height,
                         "stats": stats.to_dicts()}, dest / "predict_summary.json")
    A.log(f"outputs copied to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
