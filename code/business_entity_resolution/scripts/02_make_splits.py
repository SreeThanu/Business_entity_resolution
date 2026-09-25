"""Create the frozen validation splits in data/splits/ and report their statistics.

The splits are FROZEN. This script refuses to overwrite existing split files; pass
--out DIR to write somewhere else (the tests use this to check reproducibility).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ber import config, data
from ber.splits import VAL_PERCENT, make_splits


def match_stats(gt_long: pd.DataFrame, ids: list[str]) -> tuple[float, float]:
    """(singleton rate, mean matches per matched entity) for the given S1 IDs."""
    sub = gt_long[gt_long["source1_entity_id"].isin(set(ids))]
    n_matches = sub.groupby("source1_entity_id")["matched_entity_id"].count()
    matched = n_matches[n_matches > 0]
    return float((n_matches == 0).mean()), float(matched.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=config.SPLITS_DIR)
    ap.add_argument("--quiet", action="store_true", help="skip the statistics report")
    args = ap.parse_args()

    s1 = data.load_source("train", 1, columns=["entity_id", "country"])
    splits = make_splits(s1)

    args.out.mkdir(parents=True, exist_ok=True)
    existing = [n for n in splits if (args.out / f"{n}.txt").exists()]
    if existing:
        print(f"refusing to overwrite frozen splits in {args.out}: {existing}", file=sys.stderr)
        return 1
    for name, ids in splits.items():
        (args.out / f"{name}.txt").write_text("\n".join(ids) + "\n")
    print(f"wrote {len(splits)} ID lists to {args.out}")
    if args.quiet:
        return 0

    train, val = set(splits["train_ids"]), set(splits["val_ids"])
    assert not train & val and len(train) + len(val) == len(s1), "train/val must partition train S1"

    print(f"\nholdout rate by country (target {VAL_PERCENT}%)\n| country | S1 entities | train | val | val % |\n|---|---|---|---|---|")
    for country, g in s1.groupby("country"):
        n_val = int(g["entity_id"].isin(val).sum())
        print(f"| {country} | {len(g):,} | {len(g) - n_val:,} | {n_val:,} | {100 * n_val / len(g):.2f} |")
    n_val = len(val)
    print(f"| ALL | {len(s1):,} | {len(train):,} | {n_val:,} | {100 * n_val / len(s1):.2f} |")

    gt = data.load_ground_truth(long=True)
    print("\nper split (EDA: 5.58% singletons, 3.67 mean matches for matched entities)\n"
          "| split | S1 entities | singleton % | mean matches (matched) |\n|---|---|---|---|")
    for name, ids in splits.items():
        single, mean = match_stats(gt, ids)
        print(f"| {name} | {len(ids):,} | {100 * single:.2f} | {mean:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
