"""Tests for src/ber/decision.py: the official macro-F0.5 and the one-to-one decision rule."""

from __future__ import annotations

import polars as pl
import pytest

from ber import decision as D


def test_readme_example():
    # README: predicts [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    q = pl.DataFrame({"s1_id": ["S1-00001"]})
    m = pl.DataFrame({"s1_id": ["S1-00001"] * 3, "cand_id": ["S2-00047", "S2-00193", "S3-00812"]})
    t = pl.DataFrame({"s1_id": ["S1-00001"] * 2, "cand_id": ["S2-00047", "S3-00812"]})
    assert D.macro_f05(m, q, t)["macro_f05"][0] == pytest.approx(0.714, abs=1e-3)


def test_singletons_and_empty_predictions():
    q = pl.DataFrame({"s1_id": ["single_ok", "single_bad", "miss", "wrong"], "country": ["US", "US", "India", "India"]})
    t = pl.DataFrame({"s1_id": ["miss", "wrong"], "cand_id": ["S2-1", "S2-2"]})
    m = pl.DataFrame({"s1_id": ["single_bad", "wrong"], "cand_id": ["S2-9", "S2-8"]})
    per = {r["s1_id"]: r["f05"] for r in D.per_s1_f05(m, q, t).iter_rows(named=True)}
    assert per == {"single_ok": 1.0, "single_bad": 0.0, "miss": 0.0, "wrong": 0.0}
    by = {r["country"]: r["macro_f05"] for r in D.macro_f05(m, q, t, ["country"]).iter_rows(named=True)}
    assert by == {"India": 0.0, "US": 0.5}


def test_one_to_one_and_threshold_commute():
    preds = pl.DataFrame({"s1_id": ["a", "b", "a", "c", "b"], "cand_id": ["x", "x", "y", "y", "z"],
                          "prob": [0.9, 0.8, 0.4, 0.6, 0.3]})
    o = D.one_to_one(preds).sort("cand_id")
    assert o.select("s1_id", "cand_id").rows() == [("a", "x"), ("c", "y"), ("b", "z")]
    for t in (0.0, 0.35, 0.5, 0.7, 0.95):
        filtered_first = D.decide(preds, t).select("s1_id", "cand_id").sort("cand_id").rows()
        o2o_first = D.one_to_one(preds).filter(pl.col("prob") >= t).select("s1_id", "cand_id").sort("cand_id").rows()
        assert filtered_first == o2o_first
    # ties go to the smallest s1_id, deterministically
    tie = pl.DataFrame({"s1_id": ["b", "a"], "cand_id": ["x", "x"], "prob": [0.5, 0.5]})
    assert D.one_to_one(tie)["s1_id"].to_list() == ["a"]


def test_threshold_search_picks_best():
    q = pl.DataFrame({"s1_id": ["a", "b"]})
    t = pl.DataFrame({"s1_id": ["a"], "cand_id": ["x"]})               # b is a singleton
    preds = pl.DataFrame({"s1_id": ["a", "b"], "cand_id": ["x", "y"], "prob": [0.7, 0.4]})
    best, table = D.threshold_search(preds, q, t, [0.3, 0.5, 0.8])
    assert best == 0.5
    assert table.filter(pl.col("threshold") == 0.5)["macro_f05"][0] == 1.0
