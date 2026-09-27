"""Pair features, one parquet file per chunk of S1 (v0 baseline feature set: src/ber/features.py).

    python scripts/07_features.py --split train     # cand_train_200k  (17.7M pairs)
    python scripts/07_features.py --split val       # cand_val_50k     (4.4M pairs)
    python scripts/07_features.py --split test      # test candidates  (~160M pairs)

Input: data/cand/{name}_candidates.parquet + data/clean (the entity tables of the matching split
are held in memory; candidate pairs are streamed one chunk at a time and never all loaded).
Output: {BER_PERSIST_DIR}/features/{split}/part_NNNNN.parquet, then _DONE.json listing them.

Resumable and idempotent: the S1 ids are sorted and cut into fixed chunks of FEATURE_CHUNK_S1, so
chunk i is always the same S1s (and holds ALL their candidate pairs); chunks already written are
skipped. run.json records the settings; a changed feature version or candidate file is refused
(--force deletes the folder and starts over). Memory guard: BER_MEM_LIMIT_GIB (default 5).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import polars as pl
import pyarrow.parquet as pq

from ber import artifacts as A
from ber import config, memguard
from ber import features as F
from ber.normalize import NORMALIZE_VERSION


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(config.FEATURE_SPLITS), required=True)
    ap.add_argument("--chunk-s1", type=int, default=config.FEATURE_CHUNK_S1, help="S1 entities per chunk file")
    ap.add_argument("--force", action="store_true", help="delete this split's feature folder and rebuild")
    args = ap.parse_args()
    memguard.start()

    cand_name, ent_split = config.FEATURE_SPLITS[args.split]
    cand = config.cand_path(cand_name)
    if not cand.exists():
        sys.exit(f"{cand} missing (copy the data first: scripts/fetch_data.py, see COLAB.md)")
    out = A.features_dir(args.split)
    if args.force:
        A.reset_folder(out)

    t0 = time.time()
    s1_ids = pl.scan_parquet(cand).select(pl.col("s1_id").unique()).collect()["s1_id"].sort()
    n_rows = pq.ParquetFile(cand).metadata.num_rows
    fp = {"FEATURE_VERSION": F.FEATURE_VERSION, "NORMALIZE_VERSION": NORMALIZE_VERSION, "features": F.FEATURES,
          "candidates": cand.name, "candidate_rows": n_rows, "n_s1": len(s1_ids), "chunk_s1": args.chunk_s1,
          "s1_sha256": A.sha256_lines(s1_ids.to_list()), "block_k": config.BLOCK_K}
    n_chunks = math.ceil(len(s1_ids) / args.chunk_s1)
    names = [f"part_{i:05d}.parquet" for i in range(n_chunks)]
    done = out / "_DONE.json"
    if done.exists() and A.feature_fingerprint(args.split) == fp and all((out / n).exists() for n in names):
        A.log(f"{args.split}: features already complete in {out}, nothing to do")
        return 0
    A.check_run_manifest(out, fp)
    todo = [i for i in range(n_chunks) if not (out / names[i]).exists()]
    A.log(f"{args.split}: {n_rows:,} pairs, {len(s1_ids):,} S1, {n_chunks} chunks, {n_chunks - len(todo)} done earlier")
    if todo:
        t = time.time()
        ents = F.load_entities(ent_split)
        A.log(f"entity tables ({ent_split}): S1 {ents.s1.height:,}, S2+S3 {ents.pool.height:,} ({time.time() - t:.0f}s)")
        for n, i in enumerate(todo, 1):
            t = time.time()
            ids = s1_ids.slice(i * args.chunk_s1, args.chunk_s1)
            # ids are a contiguous run of the sorted unique S1 ids, so a range filter selects exactly them
            pairs = pl.scan_parquet(cand).filter(pl.col("s1_id").is_between(pl.lit(ids[0]), pl.lit(ids[-1]))).collect()
            feats = F.pair_features(pairs, ents)
            A.atomic_write_parquet(feats, out / names[i])
            A.log(f"{names[i]}: {len(ids):,} S1, {feats.height:,} pairs ({time.time() - t:.0f}s); "
                  f"{n}/{len(todo)} this run")

    total = sum(pq.ParquetFile(out / p).metadata.num_rows for p in names)
    if total != n_rows:
        raise RuntimeError(f"feature chunks hold {total:,} rows, candidates have {n_rows:,}")
    A.atomic_write_json({"fingerprint": fp, "parts": names, "rows": total, "seconds_this_run": round(time.time() - t0)}, done)
    A.log(f"{args.split}: complete, {total:,} rows in {len(names)} files -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
