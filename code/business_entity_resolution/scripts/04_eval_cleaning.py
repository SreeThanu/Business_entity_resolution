"""Stage 3: does the Stage 1 cleaning separate true pairs from look-alikes? Rules only, no model.

True pairs: 100k per country (200k), S1 in data/splits/train_ids.txt only. Hard negatives: 100k per
country, (S1 in train_ids, S2/S3) pairs with the same country and the same name_core that are NOT a
ground-truth match (the look-alikes / chain names that cost precision).

For every signal the table shows the rate on true pairs, on hard negatives and the gap, for three
text versions: raw, basic (lowercase, punctuation -> space, collapsed spaces) and Stage 1 clean.
The ablation table switches single rules off; a rule whose removal WIDENS the gap is flagged,
because it narrows the gap between true pairs and look-alikes.

    python scripts/04_eval_cleaning.py   # -> reports/stage3_cleaning_eval.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from ber import config, data, memguard
from ber.lookup import training_pairs
from ber.normalize import _house_and_unit, addr_nl

N_PER_COUNTRY = 100_000
MAX_PER_KEY = 5   # S2/S3 records kept per (country, name_core), so chain names cannot explode the join
SMALL_GAP = 10    # |house number difference| <= this counts as a small gap
SEED = 0
REPORT = Path(__file__).resolve().parents[1] / "reports" / "stage3_cleaning_eval.md"

SIDE_COLS = ["entity_id", "business_name", "business_address", "name_clean", "name_core", "name_nospace",
             "legal_families", "legal_suffix_class", "addr_clean", "addr_std", "addr_core", "addr_house_number",
             "addr_state_raw"]


def scan_train(source: int) -> pl.LazyFrame:
    return pl.scan_parquet(config.clean_source_path("train", source))


def build_pairs() -> pl.DataFrame:
    train_ids = data.load_split_ids("train_ids")
    gt = pl.read_parquet(config.parquet_ground_truth_path(long=True))
    s1 = scan_train(1).filter(pl.col("entity_id").is_in(train_ids)).select("entity_id", "country", "name_core").collect()
    true = (training_pairs(gt, train_ids, data.load_split_ids("val_ids"))
            .join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id"))
    true = pl.concat([g.sample(min(N_PER_COUNTRY, g.height), seed=SEED)
                      for _, g in true.group_by("country", maintain_order=True)])

    other = pl.concat([scan_train(s).select("entity_id", "country", "name_core") for s in (2, 3)]).collect()
    keyed = (other.sample(fraction=1.0, shuffle=True, seed=SEED)
             .group_by("country", "name_core", maintain_order=True).head(MAX_PER_KEY))
    del other
    cand = (s1.rename({"entity_id": "s1_id"})
            .join(keyed.rename({"entity_id": "other_id"}), on=["country", "name_core"])
            .join(gt.drop_nulls().rename({"source1_entity_id": "s1_id", "matched_entity_id": "other_id"}),
                  on=["s1_id", "other_id"], how="anti"))
    del keyed
    neg = pl.concat([g.sample(min(N_PER_COUNTRY, g.height), seed=SEED)
                     for _, g in cand.group_by("country", maintain_order=True)])
    del cand

    pairs = pl.concat([true.select("s1_id", "other_id", "country").with_columns(label=pl.lit("true")),
                       neg.select("s1_id", "other_id", "country").with_columns(label=pl.lit("hard_neg"))])
    a = scan_train(1).select(SIDE_COLS).filter(pl.col("entity_id").is_in(pairs["s1_id"].implode())).collect()
    b = pl.concat([scan_train(s).select(SIDE_COLS) for s in (2, 3)]).filter(
        pl.col("entity_id").is_in(pairs["other_id"].implode())).collect()
    return (pairs.join(a.rename(lambda c: f"a_{c}"), left_on="s1_id", right_on="a_entity_id")
            .join(b.rename(lambda c: f"b_{c}"), left_on="other_id", right_on="b_entity_id"))


# --- text versions -------------------------------------------------------------------------------

def basic(e: pl.Expr) -> pl.Expr:
    return e.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def toks(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.list.set_intersection(b).list.len() / a.list.set_union(b).list.len()


def first_int(e: pl.Expr) -> pl.Expr:
    return e.str.extract(r"(\d+)").cast(pl.Int64, strict=False)


def house_bucket(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    gap = (first_int(a) - first_int(b)).abs()
    return (pl.when(a.is_null() | b.is_null()).then(pl.lit("missing"))
            .when(a == b).then(pl.lit("exact"))
            .when(gap <= SMALL_GAP).then(pl.lit("small_gap"))
            .otherwise(pl.lit("big_gap")))


def features(p: pl.DataFrame) -> pl.DataFrame:
    A, B = (lambda c: pl.col(f"a_{c}")), (lambda c: pl.col(f"b_{c}"))
    fam_conflict = ((A("legal_families").list.len() > 0) & (B("legal_families").list.len() > 0)
                    & (A("legal_families").list.set_intersection(B("legal_families")).list.len() == 0))
    cls_conflict = (A("legal_suffix_class") != "NONE") & (B("legal_suffix_class") != "NONE") & (
        A("legal_suffix_class") != B("legal_suffix_class"))
    raw_house = lambda side: pl.col(f"{side}_business_address").str.extract(r"(\d+)")
    basic_house = lambda side: basic(pl.col(f"{side}_business_address")).str.extract(r"(?:^| )(\d\S*)")
    # ablations recomputed from raw text for the pair rows only
    p = p.with_columns(**{f"{s}_nl": addr_nl(pl.col(f"{s}_business_address")) for s in "ab"})
    p = p.with_columns(**{f"{s}_house_noskip": _house_and_unit(pl.col(f"{s}_nl"), skip_injected=False)[0] for s in "ab"})
    p = p.with_columns(**{f"{s}_house_noskip": pl.when(pl.col(f"{s}_house_noskip") == "").then(None)
                          .otherwise(pl.col(f"{s}_house_noskip")) for s in "ab"})
    both_addr = A("addr_core").is_not_null() & B("addr_core").is_not_null()
    f = p.with_columns(
        name_eq_raw=A("business_name") == B("business_name"),
        name_eq_basic=basic(A("business_name")) == basic(B("business_name")),
        name_eq_core=A("name_core") == B("name_core"),
        name_eq_nospace=A("name_nospace") == B("name_nospace"),
        legal_conflict_families=fam_conflict,
        legal_conflict_class_endonly=cls_conflict,
        house_raw=house_bucket(raw_house("a"), raw_house("b")),
        house_basic=house_bucket(basic_house("a"), basic_house("b")),
        house_clean=house_bucket(A("addr_house_number"), B("addr_house_number")),
        house_noskip=house_bucket(pl.col("a_house_noskip"), pl.col("b_house_noskip")),
        jacc_raw=pl.when(both_addr).then(jaccard(toks(A("business_address")), toks(B("business_address")))),
        jacc_basic=pl.when(both_addr).then(jaccard(toks(basic(A("business_address"))), toks(basic(B("business_address"))))),
        jacc_addr_clean=pl.when(both_addr).then(jaccard(toks(A("addr_clean")), toks(B("addr_clean")))),
        jacc_addr_std=pl.when(both_addr).then(jaccard(toks(A("addr_std")), toks(B("addr_std")))),
        jacc_addr_core=pl.when(both_addr).then(jaccard(toks(A("addr_core")), toks(B("addr_core")))),
    )
    for v in ("raw", "basic", "clean", "noskip"):
        f = f.with_columns(**{f"house_{v}_{k}": pl.col(f"house_{v}") == k for k in ("exact", "small_gap", "big_gap", "missing")})
    return f


# (group, label, column). Signals where a higher rate on true pairs is good; the gap is true - neg.
METRICS = [
    ("name", "exact: raw", "name_eq_raw"), ("name", "exact: basic", "name_eq_basic"),
    ("name", "exact: name_core", "name_eq_core"), ("name", "exact: name_nospace", "name_eq_nospace"),
    ("legal", "family conflict (legal_families)", "legal_conflict_families"),
    ("legal", "class conflict (legacy end-only)", "legal_conflict_class_endonly"),
    *[("house", f"{k}: {v}", f"house_{v}_{k}") for k in ("exact", "small_gap", "big_gap", "missing")
      for v in ("raw", "basic", "clean")],
    ("addr", "Jaccard: raw", "jacc_raw"), ("addr", "Jaccard: basic", "jacc_basic"),
    ("addr", "Jaccard: addr_clean", "jacc_addr_clean"), ("addr", "Jaccard: addr_std", "jacc_addr_std"),
    ("addr", "Jaccard: addr_core", "jacc_addr_core"),
]
# (rule, column with the rule ON, column with the rule OFF). Same sign convention: bigger |gap| wins.
ABLATIONS = [
    ("injected-clause skip (house number exact)", "house_clean_exact", "house_noskip_exact"),
    ("state slot removed (addr_core vs addr_std)", "jacc_addr_core", "jacc_addr_std"),
    ("abbrev + no-marker std (addr_std vs addr_clean)", "jacc_addr_std", "jacc_addr_clean"),
    ("legal forms anywhere (families vs end-only class)", "legal_conflict_families", "legal_conflict_class_endonly"),
    ("Stage 1 name_core vs basic (exact)", "name_eq_core", "name_eq_basic"),
    ("Stage 1 house number vs basic (exact)", "house_clean_exact", "house_basic_exact"),
    ("Stage 1 addr_core vs basic (Jaccard)", "jacc_addr_core", "jacc_basic"),
]


def rates(f: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return f.group_by("country", "label").agg([pl.col(c).cast(pl.Float64).mean().alias(c) for c in cols])


def gap(r: pl.DataFrame, country: str, c: str) -> tuple[float, float]:
    t = r.filter((pl.col("country") == country) & (pl.col("label") == "true"))[c].item()
    n = r.filter((pl.col("country") == country) & (pl.col("label") == "hard_neg"))[c].item()
    return t, n


def tables(f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = sorted({c for *_, c in METRICS} | {c for _, a, b in ABLATIONS for c in (a, b)})
    r = rates(f, cols)
    rows, abl = [], []
    for country in sorted(r["country"].unique()):
        for grp, name, c in METRICS:
            t, n = gap(r, country, c)
            rows.append({"country": country, "signal": grp, "metric": name, "true": t, "hard_neg": n, "gap": t - n})
        for rule, on, off in ABLATIONS:
            g_on, g_off = (lambda x: x[0] - x[1])(gap(r, country, on)), (lambda x: x[0] - x[1])(gap(r, country, off))
            abl.append({"country": country, "rule": rule, "gap_rule_on": g_on, "gap_rule_off": g_off,
                        "flag": "NARROWS" if abs(g_on) < abs(g_off) - 1e-4 else "ok"})
    return pl.DataFrame(rows), pl.DataFrame(abl)


def to_md(df: pl.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(fmt(v).replace("|", "/") for v in r) + " |" for r in df.iter_rows()])


def examples() -> str:
    cols = ["business_name", "name_core", "legal_families", "business_address", "addr_core", "addr_house_number",
            "addr_unit", "addr_state_raw"]
    out = []
    for country, split in (("US", "train"), ("India", "train"), ("France", "test")):
        # S2/S3 only: S1 is already clean, the noise is in S2/S3
        lf = pl.concat([pl.scan_parquet(config.clean_source_path(split, s)).select("country", *cols) for s in (2, 3)])
        d = (lf.filter(pl.col("country") == country).select(cols).collect().sample(20, seed=SEED)
             .with_columns(pl.col("legal_families").list.join(" ")))
        out.append(f"### {country} ({split} S2/S3), 20 random records\n\n{to_md(d)}")
    return "\n\n".join(out)


def main() -> int:
    memguard.start()
    p = build_pairs()
    counts = p.group_by("country", "label").len().sort("country", "label")
    print(counts)
    t, abl = tables(features(p))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Stage 3: cleaning evaluation (rules only)\n\n"
        "Generated by `scripts/04_eval_cleaning.py`. True pairs and hard negatives use train_ids S1 only.\n"
        "Hard negatives share country and name_core by construction, so name exact-match on name_core is 1.0 "
        "for them. Jaccard is over pairs where both addresses are non-empty. House buckets: exact string, "
        f"small gap (|diff| <= {SMALL_GAP}), big gap, missing (either side has no house number).\n\n"
        f"Pairs per group:\n\n{to_md(counts)}\n\n## Signals\n\n{to_md(t)}\n\n"
        "## Rule ablations\n\nA rule is flagged NARROWS when switching it off gives a larger |true - hard_neg| gap.\n\n"
        f"{to_md(abl)}\n\n## Before/after examples\n\n{examples()}\n")
    with pl.Config(tbl_rows=100, fmt_str_lengths=50, tbl_width_chars=200, float_precision=4):
        print(t)
        print(abl)
    print(f"written to {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
