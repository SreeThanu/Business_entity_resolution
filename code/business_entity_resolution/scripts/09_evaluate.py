"""Evaluate a model on val with the decision rule; choose and save the threshold.

    python scripts/09_evaluate.py                  # newest model in {BER_PERSIST_DIR}/models
    python scripts/09_evaluate.py --model <id>     # a specific one (or BER_MODEL_ID)
    python scripts/09_evaluate.py --no-cross       # skip the cross-country check

Decision rule (src/ber/decision.py): one-to-one (each S2/S3 record goes only to its highest-
probability S1), then keep pairs with probability >= threshold. The threshold is searched on
config.THRESHOLD_GRID for the best macro-F0.5, the official formula over ALL val S1 of
cand_val_50k (singletons score 1 on an empty prediction). S1 without candidates count too.

Cross-country check: for each ordered pair of train countries (US, India), a model trained on
the source country's train features only (same parameters, the main model's number of rounds)
is evaluated on the target country's val S1, at the main threshold and at the target's own best.

Writes into the model folder: val_preds.parquet, threshold.json (read by 10_predict.py),
cross_<src>_to_<tgt>.json, eval_report.md. Resumable: each of those is reused if present; a
finished eval (eval_report.md) is only printed again unless --force.

Caveat: val one-to-one only sees the 50k val S1, while on test every S1 competes for each
record, so val is slightly optimistic about the one-to-one step.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import artifacts as A
from ber import config, data, memguard
from ber import decision as D
from ber import features as F

VAL_QUERIES = "cand_val_50k"


def md(df: pl.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.iter_rows()])


def val_queries_truth() -> tuple[pl.DataFrame, pl.DataFrame]:
    ids = data.load_split_ids(VAL_QUERIES)
    q = (pl.scan_parquet(config.clean_source_path("train", 1)).select(pl.col("entity_id").alias("s1_id"), "country")
         .filter(pl.col("s1_id").is_in(ids)).collect())
    if q.height != len(ids):
        raise RuntimeError(f"{len(ids) - q.height} {VAL_QUERIES} ids are missing from train S1")
    t = (pl.scan_parquet(config.parquet_ground_truth_path(long=True)).drop_nulls("matched_entity_id")
         .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_id").alias("cand_id"))
         .filter(pl.col("s1_id").is_in(ids)).collect())
    return q, t


def predict_parts(booster: lgb.Booster, parts, feats: list[str], country: str | None = None) -> pl.DataFrame:
    out = []
    for p in parts:
        df = pl.read_parquet(p)
        if country is not None:
            df = df.filter(pl.col("country") == country)
        if df.height:
            prob = booster.predict(F.to_matrix(df, feats)).astype(np.float32)
            out.append(df.select("s1_id", "cand_id", "country").with_columns(prob=pl.Series(prob)))
    return pl.concat(out)


def cross_country(md_dir, feats, params, rounds, threshold, q, t, src, tgt) -> dict:
    path = md_dir / f"cross_{src}_to_{tgt}.json"
    if path.exists():
        return json.loads(path.read_text())
    A.log(f"cross-country: train on {src} only ({rounds} rounds), evaluate on {tgt}")
    tr = pl.concat([pl.read_parquet(p).filter(pl.col("country") == src) for p in A.feature_parts("train")])
    gt = (pl.scan_parquet(config.parquet_ground_truth_path(long=True)).drop_nulls("matched_entity_id")
          .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_id").alias("cand_id"))
          .filter(pl.col("s1_id").is_in(tr["s1_id"].unique().implode())).collect().unique()
          .with_columns(label=pl.lit(1, pl.Int8)))
    tr = tr.join(gt, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    booster = lgb.train({k: v for k, v in params.items() if k != "num_boost_round"},
                        lgb.Dataset(F.to_matrix(tr, feats), label=tr["label"].to_numpy(), feature_name=feats),
                        num_boost_round=rounds)
    del tr
    preds = predict_parts(booster, A.feature_parts("val"), feats, country=tgt)
    qt, tt = q.filter(pl.col("country") == tgt), t.filter(pl.col("s1_id").is_in(q.filter(pl.col("country") == tgt)["s1_id"].implode()))
    at_main = D.macro_f05(D.decide(preds, threshold), qt, tt).row(0, named=True)
    best_t, _ = D.threshold_search(preds, qt, tt, config.THRESHOLD_GRID)
    at_best = D.macro_f05(D.decide(preds, best_t), qt, tt).row(0, named=True)
    res = {"train_country": src, "eval_country": tgt, "rounds": rounds, "main_threshold": threshold,
           "macro_f05_at_main_threshold": at_main["macro_f05"], "own_best_threshold": best_t,
           "macro_f05_at_own_best_threshold": at_best["macro_f05"], "entities": at_main["entities"]}
    A.atomic_write_json(res, path)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model id (folder name); default: newest")
    ap.add_argument("--no-cross", action="store_true", help="skip the cross-country check")
    ap.add_argument("--force", action="store_true", help="recompute everything for this model")
    args = ap.parse_args()
    memguard.start()

    md_dir = A.resolve_model(args.model)
    report_path = md_dir / "eval_report.md"
    if args.force:
        for p in [report_path, md_dir / "threshold.json", md_dir / "val_preds.parquet", *md_dir.glob("cross_*.json")]:
            p.unlink(missing_ok=True)
    if report_path.exists():
        A.log(f"{md_dir.name}: already evaluated (use --force to redo)")
        print(report_path.read_text())
        return 0

    feats = json.loads((md_dir / "features.json").read_text())
    params = json.loads((md_dir / "params.json").read_text())
    booster = lgb.Booster(model_file=str(md_dir / "model.txt"))
    q, t = val_queries_truth()

    preds_path = md_dir / "val_preds.parquet"
    if preds_path.exists():
        preds = pl.read_parquet(preds_path)
    else:
        preds = predict_parts(booster, A.feature_parts("val"), feats)
        A.atomic_write_parquet(preds, preds_path)
    A.log(f"val: {preds.height:,} pairs scored, {q.height:,} S1, {t.height:,} true pairs")

    best_t, table = D.threshold_search(preds, q, t, config.THRESHOLD_GRID)
    matches = D.decide(preds, best_t)
    overall = D.macro_f05(matches, q, t).with_columns(group=pl.lit("ALL"))
    per_country = D.macro_f05(matches, q, t, ["country"]).rename({"country": "group"})
    no_o2o = D.macro_f05(preds.filter(pl.col("prob") >= best_t), q, t).with_columns(group=pl.lit("ALL, without one-to-one"))
    summary = pl.concat([overall, per_country, no_o2o], how="diagonal").select("group", pl.exclude("group"))
    by_country = {r["group"]: r["macro_f05"] for r in per_country.iter_rows(named=True)}
    A.atomic_write_json({"model_id": md_dir.name, "threshold": best_t,
                         "decision_rule": "one-to-one (each S2/S3 record -> its highest-probability S1), then prob >= threshold",
                         "val_queries": VAL_QUERIES, "val_macro_f05": overall["macro_f05"][0],
                         "val_macro_f05_by_country": by_country}, md_dir / "threshold.json")
    A.log(f"threshold {best_t}: val macro-F0.5 {overall['macro_f05'][0]:.4f} {by_country}")

    parts = [f"# Evaluation of {md_dir.name}\n",
             f"Val: {VAL_QUERIES} ({q.height:,} S1, {t.height:,} true pairs, {preds.height:,} candidate pairs). "
             "Decision rule: one-to-one, then threshold. Macro-F0.5 = official formula over all S1.\n",
             f"## Chosen threshold: {best_t}\n", md(summary), "\n## Threshold search\n", md(table)]
    if not args.no_cross:
        countries = sorted(q["country"].unique().to_list())
        rows = []
        for src, tgt in itertools.permutations(countries, 2):
            r = cross_country(md_dir, feats, params, params["num_boost_round"], best_t, q, t, src, tgt)
            r["in_country_model_macro_f05"] = by_country.get(tgt)
            rows.append(r)
        cross = pl.DataFrame(rows).select("train_country", "eval_country", "entities", "in_country_model_macro_f05",
                                          "macro_f05_at_main_threshold", "own_best_threshold",
                                          "macro_f05_at_own_best_threshold")
        parts += ["\n## Cross-country check\n",
                  "Model trained on one country only (same params and rounds), evaluated on the other's val S1. "
                  "`in_country_model_macro_f05` = the main model (trained on both) on the same S1, for reference.\n",
                  md(cross)]
    report = "\n".join(parts) + "\n"
    report_path.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
