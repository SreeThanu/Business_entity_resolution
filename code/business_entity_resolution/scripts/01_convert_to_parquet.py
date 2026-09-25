"""Convert the raw train/test TSVs to parquet (pyarrow, zstd). No cleaning of any kind.

Outputs in data/parquet/:
    {split}_source{n}.parquet           entity_id, business_name, business_address, country, source
    train_ground_truth.parquet          raw wide table (source1_entity_id, matched_entity_ids)
    train_ground_truth_long.parquet     one row per pair; singletons have matched_entity_id = null

Every validation failure raises; nothing is written for a file that fails its checks.
"""

from __future__ import annotations

import csv
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import config

ID_PATTERN = r"^S([123])-\d+$"
SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
GT_HEADER = ["source1_entity_id", "matched_entity_ids"]


class ValidationError(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def read_tsv(path) -> pd.DataFrame:
    # QUOTE_NONE: quote characters are kept as literal text, so a stray quote can never
    # make the parser swallow the following lines.
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )


def count_data_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def write_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")


def id_source(ids: pd.Series, what: str) -> pd.Series:
    """Source number (1/2/3) from the entity_id prefix. Raises on any malformed ID."""
    src = ids.str.extract(ID_PATTERN, expand=False)
    bad = ids[src.isna()]
    check(bad.empty, f"{what}: {len(bad)} malformed entity_ids, e.g. {bad.head(3).tolist()}")
    return src.astype("int8")


def convert_source(split: str, source: int) -> set[str]:
    path = config.raw_source_path(split, source)
    df = read_tsv(path)
    name = path.name
    check(list(df.columns) == SOURCE_HEADER, f"{name}: header {list(df.columns)}")
    n_lines = count_data_lines(path)
    check(len(df) == n_lines, f"{name}: parsed {len(df):,} rows but file has {n_lines:,} data lines")
    check(df["entity_id"].is_unique, f"{name}: {df['entity_id'].duplicated().sum()} duplicate entity_ids")

    df["source"] = id_source(df["entity_id"], name)
    wrong = df[df["source"] != source]
    check(wrong.empty, f"{name}: {len(wrong)} rows have a prefix other than S{source}, "
                       f"e.g. {wrong['entity_id'].head(3).tolist()}")

    out = config.parquet_source_path(split, source)
    write_parquet(df, out)
    back = pq.read_metadata(out).num_rows
    check(back == n_lines, f"{out.name}: parquet has {back:,} rows, TSV has {n_lines:,}")
    n_quote = int(df[SOURCE_HEADER[1:3]].apply(lambda c: c.str.contains('"', regex=False)).any(axis=1).sum())
    print(f"{name}: {len(df):,} rows, rows with a literal quote char: {n_quote}", flush=True)
    return set(df["entity_id"])


def convert_ground_truth(train_ids: dict[int, set[str]]) -> None:
    path = config.raw_ground_truth_path()
    gt = read_tsv(path)
    check(list(gt.columns) == GT_HEADER, f"ground truth: header {list(gt.columns)}")
    n_lines = count_data_lines(path)
    check(len(gt) == n_lines, f"ground truth: parsed {len(gt):,} rows but file has {n_lines:,} data lines")

    # Every train S1 ID appears exactly once, and nothing else appears.
    s1 = gt["source1_entity_id"]
    check(s1.is_unique, f"ground truth: {s1.duplicated().sum()} duplicate source1_entity_ids")
    check((id_source(s1, "ground truth S1") == 1).all(), "ground truth: source1_entity_id not from S1")
    gt_s1 = set(s1)
    check(gt_s1 == train_ids[1], f"ground truth S1 IDs vs train_source1: "
                                 f"{len(gt_s1 - train_ids[1])} only in GT, {len(train_ids[1] - gt_s1)} missing from GT")

    long = gt.assign(matched_entity_id=gt["matched_entity_ids"].str.split(","))
    long = long.explode("matched_entity_id", ignore_index=True)[["source1_entity_id", "matched_entity_id"]]
    long.loc[long["matched_entity_id"] == "", "matched_entity_id"] = None
    long["matched_entity_id"] = long["matched_entity_id"].astype("string")

    matched = long["matched_entity_id"].dropna()
    msrc = id_source(matched, "ground truth matches")
    all_train = set().union(*train_ids.values())
    missing = matched[~matched.isin(all_train)]
    check(missing.empty, f"ground truth: {len(missing)} matched IDs not in any train file, e.g. {missing.head(3).tolist()}")
    dup_pairs = long.dropna().duplicated().sum()
    check(dup_pairs == 0, f"ground truth: {dup_pairs} duplicate (S1, match) pairs")
    check(long["source1_entity_id"].nunique() == len(gt), "ground truth: long table lost S1 entities")

    write_parquet(gt, config.parquet_ground_truth_path(long=False))
    write_parquet(long, config.parquet_ground_truth_path(long=True))
    for p, n in [(config.parquet_ground_truth_path(False), len(gt)), (config.parquet_ground_truth_path(True), len(long))]:
        check(pq.read_metadata(p).num_rows == n, f"{p.name}: row count mismatch after write")
    n_single = int(long["matched_entity_id"].isna().sum())
    print(f"train_ground_truth.tsv: {len(gt):,} S1 rows -> {len(long):,} long rows "
          f"({len(matched):,} pairs, {n_single:,} singletons); match sources: "
          f"{msrc.value_counts().sort_index().to_dict()}", flush=True)


def main() -> int:
    train_ids: dict[int, set[str]] = {}
    for split in config.SPLITS:
        for source in config.SOURCES:
            ids = convert_source(split, source)
            if split == "train":
                train_ids[source] = ids
    # IDs must not collide across sources (the prefix makes this true by construction; check anyway).
    check(sum(map(len, train_ids.values())) == len(set().union(*train_ids.values())), "train IDs overlap across sources")
    convert_ground_truth(train_ids)
    print(f"all checks passed; parquet written to {config.PARQUET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
