"""Train the pair classifier: LightGBM binary, labels from the ground truth, early stopping on val.

    python scripts/08_train.py                     # needs 07 --split train and --split val
    python scripts/08_train.py --force             # train again even if this exact model exists

Label = 1 iff (s1_id, cand_id) is a ground-truth pair. Candidates missing a true pair (blocking
misses) cannot be learned here; that ceiling is in BLOCKING.md.

Output: {BER_PERSIST_DIR}/models/<YYYYmmdd-HHMMSS>_<git commit>/
    model.txt  features.json  params.json  meta.json  metrics.json  importance.csv
Idempotent: if a finished model with the same code commit, feature fingerprints and parameters
exists, it is reported and nothing is trained (the folder is written under a temp name and
renamed at the end, so a folder that exists is complete).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import artifacts as A
from ber import config, memguard
from ber import features as F


def labelled(split: str) -> pl.DataFrame:
    """Feature rows of a split + label (0/1)."""
    df = pl.concat([pl.read_parquet(p) for p in A.feature_parts(split)])
    gt = (pl.scan_parquet(config.parquet_ground_truth_path(long=True)).drop_nulls("matched_entity_id")
          .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_id").alias("cand_id"))
          .filter(pl.col("s1_id").is_in(df["s1_id"].unique().implode())).collect()
          .unique().with_columns(label=pl.lit(1, pl.Int8)))
    return df.join(gt, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))


def default_params(args) -> dict:
    return {"objective": "binary", "metric": ["binary_logloss", "auc"], "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
            "num_threads": os.cpu_count() or 2, "seed": 0, "deterministic": True, "force_col_wise": True,
            "verbosity": -1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=255)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    memguard.start()

    params = default_params(args)
    commit = A.git_commit()
    A.feature_parts("train"), A.feature_parts("val")  # clear error if 07 has not finished them
    fingerprint = {"commit": commit, "train": A.feature_fingerprint("train"), "val": A.feature_fingerprint("val"),
                   "params": params, "max_rounds": args.max_rounds, "early_stop": args.early_stop}
    fingerprint = json.loads(json.dumps(fingerprint))
    if not args.force and not commit.endswith("-dirty") and commit != "nogit":
        for d in A.model_dirs():
            meta = d / "meta.json"
            if meta.exists() and json.loads(meta.read_text()).get("fingerprint") == fingerprint:
                A.log(f"model already trained with this code, data and params: {d} (use --force to retrain)")
                return 0

    t0 = time.time()
    tr, va = labelled("train"), labelled("val")
    feats = F.FEATURES
    for name, df in (("train", tr), ("val", va)):
        A.log(f"{name}: {df.height:,} pairs, {df['label'].sum():,} positives ({df['label'].mean():.4f})")
    dtrain = lgb.Dataset(F.to_matrix(tr, feats), label=tr["label"].to_numpy(), feature_name=feats, free_raw_data=True)
    dval = lgb.Dataset(F.to_matrix(va, feats), label=va["label"].to_numpy(), reference=dtrain, free_raw_data=True)
    n_train, n_val, pos_train, pos_val = tr.height, va.height, int(tr["label"].sum()), int(va["label"].sum())
    del tr, va

    evals: dict = {}
    booster = lgb.train(params, dtrain, num_boost_round=args.max_rounds, valid_sets=[dtrain, dval],
                        valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(args.early_stop, first_metric_only=True, verbose=True),
                                   lgb.log_evaluation(100), lgb.record_evaluation(evals)])
    best = booster.best_iteration or booster.current_iteration()
    metrics = {"best_iteration": best,
               **{f"{split}_{m}": float(v[best - 1]) for split, ms in evals.items() for m, v in ms.items()},
               "train_seconds": round(time.time() - t0)}

    model_id = A.new_model_id()
    final = config.MODELS_DIR / model_id
    tmp = config.MODELS_DIR / f".tmp_{model_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(tmp / "model.txt"), num_iteration=best)
    (tmp / "features.json").write_text(json.dumps(feats, indent=2))
    (tmp / "params.json").write_text(json.dumps({**params, "num_boost_round": best}, indent=2))
    (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2))
    imp = pl.DataFrame({"feature": feats, "gain": booster.feature_importance("gain", iteration=best).astype(np.float64),
                        "split": booster.feature_importance("split", iteration=best)}).sort("gain", descending=True)
    imp.write_csv(tmp / "importance.csv")
    meta = {"model_id": model_id, "commit": commit, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_version": F.FEATURE_VERSION, "lightgbm": lgb.__version__, "train_pairs": n_train,
            "val_pairs": n_val, "train_positives": pos_train, "val_positives": pos_val, "fingerprint": fingerprint}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(final)
    A.log(f"model {model_id}: best iteration {best}, " + ", ".join(f"{k}={v:.5f}" for k, v in metrics.items()
                                                                     if k.startswith("val_")))
    print(imp.head(15))
    A.log(f"saved to {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
