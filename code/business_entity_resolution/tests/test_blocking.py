"""Tests for src/ber/blocking.py on toy data (no real files needed)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from ber import blocking as B
from ber import config


# --- recall ---------------------------------------------------------------------------------------

def test_recall_toy():
    truth = pl.DataFrame({"s1_id": ["a", "a", "b", "c"], "cand_id": ["x1", "x2", "y1", "z1"],
                          "country": ["US", "US", "US", "India"]})
    cands = pl.DataFrame({"s1_id": ["a", "a", "b", "c", "c"], "cand_id": ["x1", "q", "y1", "z9", "z1"]})
    r = B.recall(cands, truth).row(0, named=True)
    assert r["true_pairs"] == 4 and r["pair_recall"] == pytest.approx(3 / 4)
    assert r["entities"] == 3 and r["entity_all_found"] == pytest.approx(2 / 3)  # a misses x2
    by = {row["country"]: row for row in B.recall(cands, truth, ["country"]).iter_rows(named=True)}
    assert by["US"]["pair_recall"] == pytest.approx(2 / 3) and by["India"]["pair_recall"] == 1.0
    # duplicate candidate rows do not inflate recall
    assert B.recall(pl.concat([cands, cands]), truth)["pair_recall"][0] == pytest.approx(3 / 4)


def test_cand_stats_counts_queries_without_candidates():
    s = B.cand_stats(pl.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["1", "2", "3"]}), pl.Series(["a", "b", "c"]))
    assert (s["pairs"], s["max"], s["mean"]) == (3, 2, 1.0)


def test_select_k_uses_ranks_p3_and_p4_cap():
    c = pl.DataFrame({"s1_id": ["a", "a", "a", "a", "b", "b"],
                      "p1_rank": [1, 60, None, None, None, None], "p2_rank": [None, None, 3, None, None, None],
                      "p1_rank_rev": [None, None, None, 2, 1, 1], "in_p3": [False, False, False, True, False, False],
                      "p1_cos": [.9, .5, .4, .3, .8, .7]},
                     schema={"s1_id": pl.String, "p1_rank": pl.Int32, "p2_rank": pl.Int32, "p1_rank_rev": pl.Int32,
                             "in_p3": pl.Boolean, "p1_cos": pl.Float32})
    k = {"p1": 50, "p2": 10, "p4": 5}
    assert B.select_k(c, k, p4_cap=None).height == 5
    assert B.select_k(c, {"p1": 50, "p2": 10, "p4": 1}, p3=False, p4_cap=None).height == 4
    # P4 cap per S1 keeps the highest-cosine reverse pair of "b"
    capped = B.select_k(c, k, p3=False, p4_cap=1)
    assert capped.filter(pl.col("s1_id") == "b")["p1_cos"].to_list() == [pytest.approx(.8)]


# --- search primitives -----------------------------------------------------------------------------

def test_rowdot_and_topk_merge():
    A = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    Bm = sparse.csr_matrix(np.array([[1, 1, 1], [0, 1, 0], [2, 0, 0]], dtype=np.float32))
    got = B.rowdot(A, Bm, np.array([0, 0, 1]), np.array([0, 2, 1]))
    assert got.tolist() == [3.0, 2.0, 3.0]
    top = B.TopK(2, 2)
    top.merge(np.array([0, 0, 0, 1]), np.array([10, 11, 12, 20]), np.array([.1, .9, .5, .3], np.float32), 3)
    top.merge(np.array([0]), np.array([13]), np.array([.7], np.float32), 3)
    I, V = top.sorted()
    assert I[0].tolist() == [11, 13] and I[1].tolist() == [20, -1]


# --- candidate TSV --------------------------------------------------------------------------------

def test_candidate_tsv_one_row_per_s1_no_duplicates(tmp_path):
    cands = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"],
                          "cand_id": ["S2-5", "S3-7", "S2-5", "S2-9"]})
    s1 = pl.Series(["S1-1", "S1-2", "S1-3"])
    path = tmp_path / "candidate_pairs.tsv"
    B.write_candidate_tsv(cands, s1, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    rows = dict(line.split("\t") for line in lines[1:])
    assert list(rows) == ["S1-1", "S1-2", "S1-3"]            # one row per S1, including empty ones
    assert rows["S1-1"].split(",") == ["S2-5", "S3-7"]       # no duplicate ids
    assert rows["S1-3"] == ""
    assert B.validate_candidate_tsv(path, set(s1.to_list())) == []
    # the official rules catch a missing S1 row
    assert B.validate_candidate_tsv(path, {"S1-1", "S1-2", "S1-3", "S1-4"})


# --- end to end on a toy split: country is never crossed ------------------------------------------------

def _toy_split(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CLEAN_DIR", tmp_path / "clean")
    monkeypatch.setattr(config, "CAND_DIR", tmp_path / "cand")
    monkeypatch.setattr(config, "BLOCK_DF_CAP", 1.0)  # 12 documents: keep every n-gram
    (tmp_path / "clean").mkdir()
    (tmp_path / "cand").mkdir()
    cols = ["entity_id", "country", "name_core", "name_nospace", "name_domain_stem", "addr_core", "addr_house_number"]
    rows = {
        1: [("S1-a", "US", "acme tools", "acmetools", None, "12 main street dover", "12"),
            ("S1-b", "India", "shree steel", "shreesteel", None, "5 mg road pune", "5"),
            ("S1-c", "France", "boulangerie paul", "boulangeriepaul", None, "3 rue neuve lille", "3")],
        2: [("S2-a", "US", "acme tools services", "acmetoolsservices", None, "12 main street dover", "12"),
            ("S2-x", "India", "acme tools", "acmetools", None, "12 main street dover", "12"),  # same text, other country
            ("S2-b", "India", "shree steel", "shreesteel", None, "5 mg road pune", "5"),
            ("S2-f", "US", "boulangerie paul", "boulangeriepaul", None, "3 rue neuve lille", "3")],
        3: [("S3-a", "US", "acmetools", "acmetools", "acmetools", "dover", None),
            ("S3-y", "France", "shree steel", "shreesteel", None, "5 mg road pune", "5"),
            ("S3-c", "France", "boulangerie paul", "boulangeriepaul", None, "3 rue neuve lille", "3"),
            ("S3-u", "US", "zeta", "zeta", None, "99 elm road", "99")],
    }
    for s, rs in rows.items():
        df = pl.DataFrame(rs, schema=cols, orient="row")
        df.write_parquet(config.clean_source_path("toy", s))
        df.select("entity_id", pl.lit(None, pl.String).alias("addr_state_canon")).write_parquet(
            config.CLEAN_DIR / f"stage2_toy_source{s}.parquet")
    return {e: c for rs in rows.values() for e, c, *_ in rs}


def test_toy_end_to_end_never_crosses_country(tmp_path, monkeypatch):
    country_of = _toy_split(tmp_path, monkeypatch)
    st = {k: B.VectorStore("toy", k) for k in B.KINDS}
    for s in st.values():
        s.build(procs=1)
    q = pl.Series(["S1-a", "S1-b", "S1-c"])
    p3, stats = B.p3_pairs("toy", q)
    frames = []
    for country in ("France", "India", "US"):
        qc = q.filter(q.is_in([e for e, c in country_of.items() if c == country]))
        fwd = B.forward(st, country, qc, {"p1": 5, "p2": 5})
        p4 = B.reverse_all(st["p1"], country, 5)
        u = B.union(country, fwd, p3.filter(pl.col("q_id").is_in(qc.implode())), p4)
        frames.append(B.fill_cosines(st, country, u.rename({"s1_id": "q_id"})).rename({"q_id": "s1_id"}))
    c = pl.concat(frames)
    # country never crossed
    assert all(country_of[a] == country_of[b] == ctry for a, b, ctry in c.select("s1_id", "cand_id", "country").iter_rows())
    pairs = set(c.select("s1_id", "cand_id").iter_rows())
    assert {("S1-a", "S2-a"), ("S1-a", "S3-a"), ("S1-b", "S2-b"), ("S1-c", "S3-c")} <= pairs
    assert ("S1-a", "S2-x") not in pairs and ("S1-c", "S2-f") not in pairs
    # P2 finds the glued / domain spelling; P3 finds the shared house number + street token
    row = c.filter((pl.col("s1_id") == "S1-a") & (pl.col("cand_id") == "S3-a")).row(0, named=True)
    assert row["in_p2"] and row["p2_cos"] > 0
    assert c.filter((pl.col("s1_id") == "S1-a") & (pl.col("cand_id") == "S2-a"))["in_p3"].item()
    # every pair has both cosines filled, ranks start at 1, and pairs are unique
    assert c["p1_cos"].null_count() == 0 and c["p2_cos"].null_count() == 0
    assert c.filter(pl.col("in_p1"))["p1_rank"].min() == 1
    assert c.select("s1_id", "cand_id").is_duplicated().sum() == 0


def test_toy_full_mode_writes_parquet_and_valid_tsv(tmp_path, monkeypatch):
    country_of = _toy_split(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "BLOCK_K", {"p1": 2, "p2": 2, "p4": 1})
    monkeypatch.setattr(config, "BLOCK_KMAX", {"p1": 5, "p2": 5, "p4": 2})
    monkeypatch.setattr(config, "BLOCK_QUERY_CHUNK", 1)   # several chunks -> several part files
    tsv = tmp_path / "out" / "candidate_pairs.tsv"
    out = B.run_full("toy", tsv=tsv)
    c = pl.read_parquet(out)
    assert not out.with_suffix(".parts").exists()
    assert c.select("s1_id", "cand_id").is_duplicated().sum() == 0
    assert all(country_of[a] == country_of[b] for a, b in c.select("s1_id", "cand_id").iter_rows())
    assert c["p1_rank"].drop_nulls().max() <= 2 and c["p1_rank_rev"].drop_nulls().max() <= 1
    rows = dict(line.split("\t") for line in tsv.read_text().splitlines()[1:])
    assert sorted(rows) == ["S1-a", "S1-b", "S1-c"]
    assert set(rows["S1-a"].split(",")) == set(c.filter(pl.col("s1_id") == "S1-a")["cand_id"])
