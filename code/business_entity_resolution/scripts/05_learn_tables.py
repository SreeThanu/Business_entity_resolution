"""Cleaning, Stage 2: learn the state-slot table from train pairs and write addr_state_canon.

    python scripts/05_learn_tables.py

Reads the Stage 1 files (data/clean/{split}_source{n}.parquet), never rewrites them.
Learning uses only pairs whose S1 entity is in train_ids, one country at a time, under the memory
guard (abort if RSS > 5 GiB). Outputs:
    data/dicts/state_map.parquet                          the learned table
    data/clean/stage2_{split}_source{n}.parquet           entity_id, addr_state_canon
France has no labels: addr_state_canon is null there (unknown), never a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import polars as pl

from ber import config, data, memguard
from ber.lookup import LEARN_COUNTRIES, STATE_MAP_FILE, apply_state_map, learn_state_map, learning_metadata, training_pairs

FILES = [(split, s) for split in config.SPLITS for s in config.SOURCES]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss {memguard.rss_gib():.2f} GiB] {msg}", flush=True)


def stage2_path(split: str, source: int):
    return config.CLEAN_DIR / f"stage2_{split}_source{source}.parquet"


def main() -> int:
    memguard.start()
    config.DICTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = training_pairs(pl.read_parquet(config.parquet_ground_truth_path(long=True)),
                           data.load_split_ids("train_ids"), data.load_split_ids("val_ids"))
    single = pl.col("addr_state_raw").is_not_null() & ~pl.col("addr_state_raw").str.contains(",")
    parts, n_pairs = [], 0
    for country in LEARN_COUNTRIES:
        s1 = (pl.scan_parquet(config.clean_source_path("train", 1)).filter((pl.col("country") == country) & single)
              .select(pl.col("entity_id").alias("s1_id"), pl.col("addr_state_raw").alias("s1_state")).collect())
        p = pairs.join(s1, on="s1_id")
        other = pl.concat([
            pl.scan_parquet(config.clean_source_path("train", s)).filter((pl.col("country") == country) & single)
            .select(pl.col("entity_id").alias("other_id"), pl.col("addr_state_raw").alias("other_state"))
            for s in (2, 3)]).collect()
        aligned = p.join(other, on="other_id").select(pl.lit(country).alias("country"), "s1_state", "other_state")
        n_pairs += aligned.height
        parts.append(learn_state_map(aligned))
        log(f"{country}: {aligned.height:,} aligned train pairs, {parts[-1].height} table rows")
        del s1, p, other, aligned
    state_map = pl.concat(parts)
    meta = learning_metadata(n_pairs)
    state_map.write_parquet(config.DICTS_DIR / STATE_MAP_FILE, metadata=meta)
    meta = {**meta, "stage": "2", "dict_sha256": json.dumps(
        {STATE_MAP_FILE: hashlib.sha256((config.DICTS_DIR / STATE_MAP_FILE).read_bytes()).hexdigest()})}

    for sp, s in FILES:
        src = pl.scan_parquet(config.clean_source_path(sp, s))
        out = stage2_path(sp, s)
        tmp = out.with_suffix(".tmp.parquet")
        apply_state_map(src, state_map).sink_parquet(tmp, compression="zstd", metadata=meta)
        n_in, n_out = src.select(pl.len()).collect().item(), pl.scan_parquet(tmp).select(pl.len()).collect().item()
        if n_in != n_out:
            raise RuntimeError(f"{out.name}: {n_out} rows, Stage 1 has {n_in}")
        tmp.replace(out)
        log(f"stage 2 {sp}_source{s}: {n_out:,} rows")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
