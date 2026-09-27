"""Decision rule (pair probabilities -> matches) and the official metric.

Decision rule, used identically on val (09_evaluate.py) and test (10_predict.py):
    1. one-to-one: every S2/S3 record goes only to its highest-probability S1 (ties: smallest s1_id);
    2. threshold: keep that pair if its probability >= t.
Filtering by the threshold first and then taking the per-record argmax gives the same pairs (the
argmax survives the filter iff its probability >= t), so large prediction sets are filtered first.

Metric (official README): F0.5 per S1, macro-averaged over ALL S1 of the evaluation set.
    F0.5 = 1.25 P R / (0.25 P + R)
    S1 without true matches: 1 if nothing is predicted, else 0.
    S1 with true matches but no correct prediction (incl. an empty prediction): 0.
"""

from __future__ import annotations

import polars as pl


def one_to_one(preds: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """preds: s1_id, cand_id, prob (+ anything). Keeps, per cand_id, the row with the highest prob."""
    return (preds.sort(["cand_id", "prob", "s1_id"], descending=[False, True, False])
            .unique("cand_id", keep="first", maintain_order=True))


def decide(preds: pl.DataFrame | pl.LazyFrame, threshold: float) -> pl.DataFrame | pl.LazyFrame:
    """The full decision rule: predicted matches (s1_id, cand_id, prob)."""
    return one_to_one(preds.filter(pl.col("prob") >= threshold))


def per_s1_f05(matches: pl.DataFrame, queries: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """F0.5 for every queried S1.

    matches: s1_id, cand_id (predicted). queries: s1_id (+ any grouping columns), every S1 of the
    evaluation set. truth: s1_id, cand_id (true pairs of those S1).
    """
    m = matches.select("s1_id", "cand_id").unique()
    tp = m.join(truth.select("s1_id", "cand_id").unique(), on=["s1_id", "cand_id"]).group_by("s1_id").len("tp")
    n_pred = m.group_by("s1_id").len("n_pred")
    n_true = truth.select("s1_id", "cand_id").unique().group_by("s1_id").len("n_true")
    per = (queries.join(n_pred, on="s1_id", how="left").join(n_true, on="s1_id", how="left")
           .join(tp, on="s1_id", how="left")
           .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0).cast(pl.Int64)))
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * p * r / (0.25 * p + r)))
    return per.with_columns(f05=f)


def macro_f05(matches: pl.DataFrame, queries: pl.DataFrame, truth: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Macro-F0.5 (+ entities, singleton share, pair precision/recall), overall or per `by` group."""
    per = per_s1_f05(matches, queries, truth)
    agg = [pl.len().alias("entities"), (pl.col("n_true") == 0).mean().alias("singleton_share"),
           pl.col("f05").mean().alias("macro_f05"),
           (pl.col("tp").sum() / pl.col("n_pred").sum()).alias("pair_precision"),
           (pl.col("tp").sum() / pl.col("n_true").sum()).alias("pair_recall"),
           (pl.col("n_pred").sum() / pl.len()).alias("pred_per_s1")]
    return per.group_by(by).agg(agg).sort(by) if by else per.select(agg)


def threshold_search(preds: pl.DataFrame, queries: pl.DataFrame, truth: pl.DataFrame,
                     grid: list[float]) -> tuple[float, pl.DataFrame]:
    """Macro-F0.5 of the decision rule at every threshold in `grid`; returns (best t, table)."""
    o2o = one_to_one(preds.select("s1_id", "cand_id", "prob"))
    rows = [macro_f05(o2o.filter(pl.col("prob") >= t), queries, truth).with_columns(threshold=pl.lit(t))
            for t in grid]
    table = pl.concat(rows).select("threshold", pl.exclude("threshold"))
    best = table.sort(["macro_f05", "threshold"], descending=[True, False]).row(0, named=True)["threshold"]
    return float(best), table
