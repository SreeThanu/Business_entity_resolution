"""Cleaning, Stage 1: deterministic text cleaning (src/ber/normalize.py) on every source file.

    python scripts/03_clean.py          # data/parquet -> data/clean/{split}_source{n}.parquet
    python scripts/03_clean.py --dev    # 50k-row samples: data/dev -> data/dev/clean

Each file is streamed (sink_parquet), one at a time, under the memory guard (abort if RSS > 5 GiB).
Raw columns are kept unchanged; Stage 1 columns are only added. The output carries
NORMALIZE_VERSION in its parquet key-value metadata and is written to a temp file first, so a
failed run never leaves a half-written file under the final name.

Stage 1 files are immutable once written: later stages (scripts/05_learn_tables.py) write their
columns to separate sidecar files keyed by entity_id and never rewrite these.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import polars as pl

from ber import config, memguard
from ber.normalize import NORMALIZE_VERSION, STAGE1_COLUMNS, clean_frame

FILES = [(split, s) for split in config.SPLITS for s in config.SOURCES]


def log(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(f"[{time.strftime('%H:%M:%S')} rss {memguard.rss_gib():.2f} peak {peak:.2f} GiB] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="run on the 50k-row samples in data/dev")
    args = ap.parse_args()
    memguard.start()

    if args.dev:
        src = lambda sp, s: config.DATA_DIR / "dev" / f"{sp}_source{s}.parquet"
        dst = lambda sp, s: config.DATA_DIR / "dev" / "clean" / f"{sp}_source{s}.parquet"
    else:
        src, dst = config.parquet_source_path, config.clean_source_path
    for sp, s in FILES:
        t = time.time()
        final = dst(sp, s)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_suffix(".tmp.parquet")
        clean_frame(pl.scan_parquet(src(sp, s))).sink_parquet(
            tmp, compression="zstd", metadata={"NORMALIZE_VERSION": NORMALIZE_VERSION, "stage": "1"})
        n_in = pl.scan_parquet(src(sp, s)).select(pl.len()).collect().item()
        out = pl.scan_parquet(tmp)
        n_out = out.select(pl.len()).collect().item()
        missing = set(STAGE1_COLUMNS) - set(out.collect_schema().names())
        if n_in != n_out or missing:
            raise RuntimeError(f"{final.name}: {n_out} rows (source {n_in}), missing columns {missing}")
        tmp.replace(final)
        log(f"stage 1 {sp}_source{s}: {n_out:,} rows, {final.stat().st_size / 2**20:.0f} MiB, {time.time() - t:.0f}s")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
