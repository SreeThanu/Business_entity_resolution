"""Tiny, self-consistent copy of the data layout for the local smoke test of scripts/07-10.

    python scripts/make_dev_sample.py --out /tmp/x/data [--n-train 400 --n-val 200 --n-test 300]

Reads the real data (config.DATA_DIR) and writes, under --out, the same layout restricted to a
few S1: the first N ids of cand_train_200k / cand_val_50k (sorted) and the first N/3 test S1 per
country, their candidate rows, only the S2/S3 rows those candidates reference, the matching
ground-truth rows and the splits. Parquet key-value metadata (NORMALIZE_VERSION) is kept.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from ber import config, data


def copy_rows(src: Path, dst: Path, ids: list[str], col: str = "entity_id") -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(src, filters=[(col, "in", ids)])
    kv = {k: v for k, v in (pq.ParquetFile(src).metadata.metadata or {}).items() if k != b"ARROW:schema"}
    pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}), **kv}), dst, compression="zstd")
    return t.num_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args()
    out = args.out
    if out.resolve() == config.DATA_DIR.resolve():
        sys.exit("--out must not be the real data dir")

    train_q = data.load_split_ids("cand_train_200k")[: args.n_train]
    val_q = data.load_split_ids("cand_val_50k")[: args.n_val]
    test_s1 = pl.read_parquet(config.clean_source_path("test", 1), columns=["entity_id", "country"]).sort("entity_id")
    test_q = pl.concat([g.head(args.n_test // 3) for _, g in test_s1.group_by("country", maintain_order=True)])["entity_id"].to_list()

    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "cand_train_200k.txt").write_text("\n".join(train_q) + "\n")
    (out / "splits" / "cand_val_50k.txt").write_text("\n".join(val_q) + "\n")

    pool = {"train": set(), "test": set()}
    for name, q, split in (("cand_train_200k", train_q, "train"), ("cand_val_50k", val_q, "train"), ("test", test_q, "test")):
        c = pl.scan_parquet(config.cand_path(name)).filter(pl.col("s1_id").is_in(q)).collect()
        (out / "cand").mkdir(parents=True, exist_ok=True)
        c.write_parquet(out / "cand" / f"{name}_candidates.parquet")
        pool[split] |= set(c["cand_id"].to_list())
        print(f"{name}: {len(q)} S1, {c.height:,} candidate pairs")

    for split, s1_ids in (("train", train_q + val_q), ("test", test_q)):
        ids = {1: s1_ids, 2: sorted(i for i in pool[split] if i.startswith("S2-")),
               3: sorted(i for i in pool[split] if i.startswith("S3-"))}
        for s in config.SOURCES:
            for src in (config.clean_source_path(split, s), config.stage2_source_path(split, s)):
                n = copy_rows(src, out / "clean" / src.name, ids[s])
                print(f"clean/{src.name}: {n:,} rows")

    gt = config.parquet_ground_truth_path(long=True)
    n = copy_rows(gt, out / "parquet" / gt.name, train_q + val_q, col="source1_entity_id")
    print(f"parquet/{gt.name}: {n:,} rows")
    (out / "dicts").mkdir(parents=True, exist_ok=True)
    for f in config.DICTS_DIR.glob("*.parquet"):
        shutil.copyfile(f, out / "dicts" / f.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
