"""Tests for src/ber/features.py on toy entity tables (no real files needed)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ber import config
from ber import features as F


def _ents(rows: list[dict]) -> pl.DataFrame:
    base = {"country": "US", "name_nospace": None, "name_domain_stem": None, "legal_families": [],
            "name_script": "Latin", "name_has_nonlatin": False, "addr_core": None, "addr_house_number": None,
            "addr_numbers": None, "addr_state_canon": None}
    full = [{**base, **r} for r in rows]
    for r in full:
        r["name_nospace"] = r["name_nospace"] or r["name_core"].replace(" ", "")
    return pl.DataFrame(full, schema={"entity_id": pl.String, "country": pl.String, "name_core": pl.String,
                                      "name_nospace": pl.String, "name_domain_stem": pl.String,
                                      "legal_families": pl.List(pl.String), "name_script": pl.String,
                                      "name_has_nonlatin": pl.Boolean, "addr_core": pl.String,
                                      "addr_house_number": pl.String, "addr_numbers": pl.List(pl.String),
                                      "addr_state_canon": pl.String})


@pytest.fixture
def toy():
    s1 = _ents([
        {"entity_id": "S1-1", "name_core": "qes induction", "legal_families": ["LIMITED", "PRIVATE"],
         "addr_core": "kh 570/13 new delhi", "addr_house_number": "570/13", "addr_numbers": ["570/13"],
         "addr_state_canon": "delhi", "country": "India"},
        {"entity_id": "S1-2", "name_core": "starbucks", "addr_core": "12 main st", "addr_house_number": "12",
         "addr_numbers": ["12"], "addr_state_canon": "tx"},
        {"entity_id": "S1-3", "name_core": "starbucks", "country": "US"},
        {"entity_id": "S1-4", "name_core": "maison dupont", "country": "France", "legal_families": ["SARL"],
         "addr_core": "4 rue x", "addr_house_number": "4", "addr_numbers": ["4"]},
    ])
    pool = _ents([
        {"entity_id": "S2-1", "name_core": "qes induction", "legal_families": ["LIMITED"],
         "addr_core": "kh 570/13 new delhi", "addr_house_number": "570/13", "addr_numbers": ["570/13"],
         "addr_state_canon": "delhi", "country": "India"},
        {"entity_id": "S3-2", "name_core": "starbucks coffee", "legal_families": ["INC"], "addr_core": "15 main st",
         "addr_house_number": "15", "addr_numbers": ["15"], "addr_state_canon": "ca", "name_domain_stem": "starbucks"},
        {"entity_id": "S2-3", "name_core": "maison dupont", "country": "France", "legal_families": ["SAS"],
         "addr_core": "4 rue x", "addr_house_number": "4", "addr_numbers": ["4"]},
    ])
    pairs = pl.DataFrame({
        "s1_id": ["S1-1", "S1-2", "S1-3", "S1-4"], "cand_id": ["S2-1", "S3-2", "S3-2", "S2-3"],
        "country": ["India", "US", "US", "France"],
        "in_p1": [True, True, False, True], "in_p2": [True, False, False, True],
        "in_p3": [True, False, False, False], "in_p4": [True, False, True, False],
        "p1_cos": [1.0, 0.6, 0.3, 0.9], "p1_rank": [1, 3, None, 1], "p1_rank_rev": [1, None, 2, None],
        "p2_cos": [1.0, 0.7, 0.7, 1.0], "p2_rank": [1, None, None, 1],
    }, schema_overrides={"p1_cos": pl.Float32, "p2_cos": pl.Float32, "p1_rank": pl.Int32,
                         "p1_rank_rev": pl.Int32, "p2_rank": pl.Int32})
    return pairs, F.build_entities(s1, pool)


def test_schema_and_order(toy):
    pairs, ents = toy
    out = F.pair_features(pairs, ents)
    assert out.columns == ["s1_id", "cand_id", "country", *F.FEATURES]
    assert out.height == pairs.height
    assert "country" not in F.FEATURES  # never a model input
    assert F.to_matrix(out, F.FEATURES).shape == (4, len(F.FEATURES))


def test_values(toy):
    pairs, ents = toy
    r = {x["s1_id"]: x for x in F.pair_features(pairs, ents).iter_rows(named=True)}
    exact, star, bare, fr = r["S1-1"], r["S1-2"], r["S1-3"], r["S1-4"]
    # names
    assert exact["name_core_eq"] == 1 and exact["name_tsr"] == pytest.approx(1) and exact["name_3g_jac"] == pytest.approx(1)
    assert star["name_tok_added"] == 1 and star["name_tok_dropped"] == 0 and star["name_core_eq"] == 0
    assert star["name_tsr"] == pytest.approx(1)  # token-set ratio: subset
    assert star["name_domain_match"] == 1 and star["name_any_domain"] == 1 and exact["name_domain_match"] == 0
    # legal families: set semantics
    assert (exact["legal_agree"], exact["legal_conflict"], exact["legal_unknown"]) == (1, 0, 0)
    assert (fr["legal_agree"], fr["legal_conflict"]) == (0, 1)
    assert bare["legal_unknown"] == 1
    # address
    assert exact["hn_exact"] == 1 and exact["hn_gap"] == 0 and exact["addr_tok_jac"] == pytest.approx(1)
    assert star["hn_exact"] == 0 and star["hn_gap"] == 3
    assert star["addr_num_overlap"] == 0 and star["addr_tok_jac"] == pytest.approx(2 / 4)
    # empty address = unknown: nulls, flags set, never a mismatch
    assert bare["a_addr_missing"] == 1 and bare["addr_tok_jac"] is None and bare["hn_exact"] is None
    assert bare["hn_gap"] is None and bare["addr_num_overlap"] is None and bare["a_hn_missing"] == 1
    # state: France null = unknown
    assert exact["state_agree"] == 1 and star["state_disagree"] == 1
    assert (fr["state_agree"], fr["state_disagree"], fr["state_unknown"]) == (0, 0, 1)
    # blocking: null ranks -> k + 1
    assert bare["p1_rank"] == config.BLOCK_K["p1"] + 1 and star["p1_rank_rev"] == config.BLOCK_K["p4"] + 1
    assert exact["mutual_best"] == 1 and bare["mutual_best"] == 0 and exact["n_passes"] == 4
    assert star["b_is_s3"] == 1 and exact["b_is_s3"] == 0
    # chain names: 2 US S1 called "starbucks"; the S3 record's name_core has no S1 twin
    assert star["a_chain_n"] == 2 and star["b_chain_n"] == 0 and exact["a_chain_n"] == 1


def test_per_s1_features():
    pairs = pl.DataFrame({"s1_id": ["a", "a", "a", "b"], "cand_id": ["S2-1", "S2-2", "S3-3", "S2-1"],
                          "country": ["US"] * 4, "in_p1": [True] * 4, "in_p2": [False] * 4, "in_p3": [False] * 4,
                          "in_p4": [False] * 4, "p1_cos": [0.9, 0.5, 0.9, 0.2], "p1_rank": [1, 3, 2, 1],
                          "p1_rank_rev": [None] * 4, "p2_cos": [0.1, 0.2, 0.3, 0.4], "p2_rank": [None] * 4},
                         schema_overrides={"p1_cos": pl.Float32, "p2_cos": pl.Float32, "p1_rank": pl.Int32,
                                           "p1_rank_rev": pl.Int32, "p2_rank": pl.Int32})
    s1 = _ents([{"entity_id": "a", "name_core": "x y"}, {"entity_id": "b", "name_core": "z"}])
    pool = _ents([{"entity_id": i, "name_core": "x"} for i in ("S2-1", "S2-2", "S3-3")])
    out = F.pair_features(pairs, F.build_entities(s1, pool)).sort("s1_id", "cand_id")
    assert out["s1_n_cands"].to_list() == [3, 3, 3, 1]
    assert out["p1_cos_rank_s1"].to_list() == [1, 3, 1, 1]
    assert out["p1_cos_gap_s1"].to_list() == pytest.approx([0, 0.4, 0, 0])


def test_missing_entity_raises(toy):
    pairs, ents = toy
    with pytest.raises(ValueError, match="missing"):
        F.pair_features(pairs.with_columns(cand_id=pl.lit("S2-404")), ents)


def test_grams_jaccard_short_strings():
    df = pl.DataFrame({"a": ["ab", "abcd", None], "b": ["ab", "bcde", "abc"]})
    j = df.select(a=F._grams("a"), b=F._grams("b")).select(F._jaccard("a", "b"))["a"].to_list()
    assert j[0] is None                      # no 3-grams on either side
    assert j[1] == pytest.approx(1 / 3)      # {abc, bcd} vs {bcd, cde}
    assert j[2] is None or math.isnan(j[2])  # null side
