"""Blocking / candidate generation (see BLOCKING.md).

    # Mac (8 GB): a query sample against the FULL S2+S3 pool, with the recall report
    python scripts/05_block.py --split train --mode dev --queries train_ids      # tuning sample
    python scripts/05_block.py --split train --mode dev --queries val_ids        # final dev numbers
    python scripts/05_block.py --split train --mode dev --queries train_ids --pool-experiment --exact-check 2000
    # Colab: every S1 query
    python scripts/05_block.py --split train --mode full    # -> data/cand/train_candidates.parquet (+ val_ids recall)
    python scripts/05_block.py --split test  --mode full    # -> data/cand/test_candidates.parquet + output/candidate_pairs.tsv

Dev mode keeps every pass's top-KMAX with ranks, so any k up to KMAX can be evaluated from one run.
Full mode keeps config.BLOCK_K. Labels are read only in the evaluation, never to make candidates.
Memory guard: BER_MEM_LIMIT_GIB (default 5).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ber import blocking as B
from ber import config, data, memguard

REPORTS = Path(__file__).resolve().parents[1] / "reports"
KS = [5, 10, 20, 50, 100, 200]


def md(df: pl.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(fmt(v).replace("|", "/") for v in r) + " |" for r in df.iter_rows()])


def s1_frame(split: str) -> pl.DataFrame:
    return pl.read_parquet(config.clean_source_path(split, 1), columns=["entity_id", "country"])


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

def truth_for(queries: pl.DataFrame) -> pl.DataFrame:
    gt = pl.read_parquet(config.parquet_ground_truth_path(long=True)).drop_nulls("matched_entity_id")
    t = gt.join(queries.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").rename(
        {"source1_entity_id": "s1_id", "matched_entity_id": "cand_id"})
    script = pl.concat([pl.scan_parquet(config.clean_source_path("train", s)).select("entity_id", "name_script")
                        for s in (2, 3)]).filter(pl.col("entity_id").is_in(t["cand_id"].implode())).collect()
    return t.join(script.rename({"entity_id": "cand_id"}), on="cand_id", how="left").with_columns(
        indic=~pl.col("name_script").is_in(["Latin", "None", "Other"]))


def recall_table(c: pl.DataFrame, truth: pl.DataFrame, label: str) -> pl.DataFrame:
    rows = [B.recall(c, truth).with_columns(group=pl.lit("ALL")),
            B.recall(c, truth, ["country"]).rename({"country": "group"}),
            B.recall(c, truth.filter("indic")).with_columns(group=pl.lit("Indic-script S2/S3 names"))]
    return pl.concat([r.select("group", "true_pairs", "pair_recall", "entities", "entity_all_found") for r in rows]
                     ).with_columns(candidates=pl.lit(label)).select("candidates", pl.exclude("candidates"))


def evaluate(c: pl.DataFrame, queries: pl.DataFrame, k: dict[str, int], title: str) -> str:
    truth = truth_for(queries)
    final = B.select_k(c, k)
    parts = [f"# Blocking evaluation: {title}\n",
             f"Queries: {queries.height:,} S1 entities; pool: the FULL train S2+S3 of the same country. "
             f"k = {k}; P4 max per S1 = {config.BLOCK_P4_MAX_PER_S1}; P3 max block = {config.BLOCK_P3_MAX_BLOCK}; "
             f"df cap = {config.BLOCK_DF_CAP}.\n"]

    # final set + EDA baseline
    tabs = [recall_table(final, truth, "union (final k)"),
            recall_table(c.filter(pl.col("p1_rank") <= 50), truth, "P1 alone, k=50 (EDA baseline set-up)")]
    parts.append("## Recall\n\nEDA baseline (raw text, P1 alone, k=50): 0.9451 overall, US 0.9757, India 0.8998.\n\n"
                 + md(pl.concat(tabs)))

    st = B.cand_stats(final, queries["entity_id"])
    parts.append("## Candidates per S1 entity (final set)\n\n" + md(pl.DataFrame([st])))

    # recall vs k for P1 and P2 alone
    curve = []
    for p in ("p1", "p2", "p4"):
        rank = {"p1": "p1_rank", "p2": "p2_rank", "p4": "p1_rank_rev"}[p]
        for kk in [x for x in KS if x <= config.BLOCK_KMAX[p]]:
            sel = c.filter(pl.col(rank) <= kk)
            r = B.recall(sel, truth)
            by = B.recall(sel, truth, ["country"])
            curve.append({"pass": p, "k": kk, "pair_recall": r["pair_recall"][0],
                          **{f"recall_{row['country']}": row["pair_recall"] for row in by.iter_rows(named=True)},
                          "mean_cands": B.cand_stats(sel, queries["entity_id"])["mean"]})
    parts.append("## Recall vs k (each pass alone)\n\n" + md(pl.DataFrame(curve)))

    # budget grid
    grid = []
    for k1 in (20, 30, 50, 100):
        for k2 in (0, 5, 10, 20):
            for k4 in (0, 3, 5, 10):
                kk = {"p1": k1, "p2": k2, "p4": k4}
                for p3 in (False, True):
                    sel = B.select_k(c, kk, p3=p3)  # includes the P4 cap
                    r = B.recall(sel, truth)
                    grid.append({"k1": k1, "k2": k2, "k4": k4, "p3": p3, "pair_recall": r["pair_recall"][0],
                                 "entity_all_found": r["entity_all_found"][0],
                                 "mean_cands": B.cand_stats(sel, queries["entity_id"])["mean"]})
    g = pl.DataFrame(grid).sort("mean_cands")
    frontier = g.filter(pl.col("pair_recall") >= pl.col("pair_recall").cum_max())
    parts.append("## Budget grid: recall-vs-mean-candidates frontier\n\n" + md(frontier))

    # marginal contribution
    found = final.join(truth.select("s1_id", "cand_id", "country", "indic"), on=["s1_id", "cand_id"])
    mc = []
    for p in ("in_p1", "in_p2", "in_p3", "in_p4"):
        others = [x for x in ("in_p1", "in_p2", "in_p3", "in_p4") if x != p]
        only = found.filter(pl.col(p) & ~pl.any_horizontal(others))
        mc.append({"pass": p[3:], "true_pairs_found": found.filter(p).height, "found_only_by_this_pass": only.height,
                   "marginal_recall": only.height / truth.height,
                   "marginal_recall_indic": only.filter("indic").height / max(1, truth.filter("indic").height),
                   "candidates_from_pass": final.filter(p).height})
    parts.append("## Marginal contribution (true pairs no other pass found)\n\n" + md(pl.DataFrame(mc)))

    # misses
    miss = truth.join(final.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="anti")
    ex = miss.sample(min(20, miss.height), seed=config.BLOCK_SEED)
    cols = ["entity_id", "name_core", "addr_core", "addr_house_number", "name_script"]
    a = pl.read_parquet(config.clean_source_path("train", 1), columns=cols).filter(pl.col("entity_id").is_in(ex["s1_id"].implode()))
    b = pl.concat([pl.scan_parquet(config.clean_source_path("train", s)).select(cols) for s in (2, 3)]).filter(
        pl.col("entity_id").is_in(ex["cand_id"].implode())).collect()
    side = (ex.select("s1_id", "cand_id", "country").join(a.rename(lambda x: f"s1_{x}"), left_on="s1_id", right_on="s1_entity_id")
            .join(b.rename(lambda x: f"c_{x}"), left_on="cand_id", right_on="c_entity_id")
            .join(c.select("s1_id", "cand_id", "p1_cos", "p1_rank", "p2_rank", "p1_rank_rev"), on=["s1_id", "cand_id"], how="left"))
    parts.append(f"## 20 true matches still missed ({miss.height:,} missed pairs in total)\n\n"
                 + md(side.drop("s1_name_script")))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------------
# Experiments (dev, train split)
# ---------------------------------------------------------------------------------------------

def pool_experiment(st: dict[str, B.VectorStore], queries: pl.DataFrame) -> str:
    """P1 recall@k when the pool is subsampled, or enlarged with test S2/S3 records (train IDF)."""
    truth = truth_for(queries)
    test = B.VectorStore("test", "p1")
    test.build()
    rows = []
    for country in sorted(queries["country"].unique()):
        q = queries.filter(pl.col("country") == country)["entity_id"]
        ids, Qf, Qc = B.load_rows(st["p1"], 1, country, q)
        train_sh = st["p1"].shards((2, 3), country)
        n_train = sum(s.n for s in train_sh)
        test_sh = test.shards((2, 3), country)
        n_test = sum(s.n for s in test_sh)
        for factor in (0.5, 0.81, 1.0, 1.23):
            keep_train = min(1.0, factor)
            add = max(0.0, factor - 1.0) * n_train / max(1, n_test)  # share of the test pool to add
            top, offset, idmap = B.TopK(Qf.shape[0], 100), 0, []
            for store, shards, frac in ((st["p1"], train_sh, keep_train), (test, test_sh, add)):
                if frac <= 0:
                    continue
                for sh in shards:
                    sid = store.ids(sh)
                    mask = (sid.hash(seed=7) % 1000 < int(frac * 1000)).to_numpy()
                    rows_ = np.flatnonzero(mask)
                    if rows_.size == 0:
                        continue
                    Xf, Xc = store.load(sh, rows=rows_, idf_from=st["p1"])
                    B.search_shard(top, Qf, Qc, Xf, Xc, offset, config.BLOCK_RETRIEVE_M["p1"])
                    idmap.append(sid.gather(rows_))
                    offset += rows_.size
            pool = pl.DataFrame({"entity_id": pl.concat(idmap)}).with_row_index("gidx").with_columns(pl.col("gidx").cast(pl.Int64))
            f = B.topk_frame(ids, top, pool, "p1_cos", "p1_rank").rename({"q_id": "s1_id"})
            t = truth.filter(pl.col("country") == country).join(pool.select(pl.col("entity_id").alias("cand_id")), on="cand_id", how="semi")
            for kk in (10, 20, 50, 100):
                rows.append({"country": country, "pool_factor": factor, "pool_size": offset, "k": kk,
                             "pair_recall": B.recall(f.filter(pl.col("p1_rank") <= kk), t)["pair_recall"][0]})
            B.log(f"pool experiment {country} x{factor}: pool {offset:,}")
    r = pl.DataFrame(rows).pivot(on="k", index=["country", "pool_factor", "pool_size"], values="pair_recall")
    return ("## Pool size vs recall (P1 alone)\n\nTrain pool subsampled (x0.5, x0.81 = train/test ratio) or enlarged "
            "to x1.23 with test S2/S3 records of the same country (the test pool has ~23% more S2/S3 per S1). "
            "True pairs whose record left the pool are excluded.\n\n" + md(r.rename({c: f"recall@{c}" for c in r.columns if c.isdigit()})))


def exact_check(st: dict[str, B.VectorStore], queries: pl.DataFrame, n: int) -> str:
    """Recall of capped retrieval + exact re-rank vs an exact brute-force search (all n-grams)."""
    q = queries.sample(min(n, queries.height), seed=config.BLOCK_SEED + 1)
    truth = truth_for(q)
    rows = []
    for country in sorted(q["country"].unique()):
        ids, Qf, Qc = B.load_rows(st["p1"], 1, country, q.filter(pl.col("country") == country)["entity_id"])
        pool = B.pool_ids(st["p1"], (2, 3), country)
        for label, exact in (("capped retrieval + exact re-rank", False), ("exact search", True)):
            t0 = time.time()
            top, offset = B.TopK(Qf.shape[0], 100), 0
            for sh in st["p1"].shards((2, 3), country):
                Xf, Xc = st["p1"].load(sh)
                B.search_shard(top, Qf, Qf if exact else Qc, Xf, Xf if exact else Xc, offset,
                               100 if exact else config.BLOCK_RETRIEVE_M["p1"])
                offset += sh.n
            f = B.topk_frame(ids, top, pool, "p1_cos", "p1_rank").rename({"q_id": "s1_id"})
            t = truth.filter(pl.col("country") == country)
            rows.append({"country": country, "method": label, "seconds": round(time.time() - t0),
                         **{f"recall@{kk}": B.recall(f.filter(pl.col("p1_rank") <= kk), t)["pair_recall"][0] for kk in (10, 50, 100)}})
            B.log(f"exact check {country} {label}: {time.time() - t0:.0f}s")
    return f"## Capped retrieval vs exact search (P1, {q.height:,} queries)\n\n" + md(pl.DataFrame(rows))


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=config.SPLITS, required=True)
    ap.add_argument("--mode", choices=("dev", "full"), required=True)
    ap.add_argument("--queries", default="train_ids", help="dev, train split: ID list to sample queries from")
    ap.add_argument("--n", type=int, default=config.BLOCK_DEV_QUERIES, help="dev: number of sampled queries")
    ap.add_argument("--pool-experiment", action="store_true")
    ap.add_argument("--exact-check", type=int, default=0, metavar="N")
    ap.add_argument("--eval-only", action="store_true", help="dev: re-evaluate the saved dev candidates")
    args = ap.parse_args()
    memguard.start()
    if not config.CAND_DIR.exists():  # never create it silently on the internal disk
        sys.exit(f"{config.CAND_DIR} missing: symlink it to the T7 (Mac) or mkdir -p it (Colab); see BLOCKING.md")
    st = B.stores(args.split)
    s1 = s1_frame(args.split)

    if args.mode == "dev":
        pool_q = s1
        if args.split == "train":
            pool_q = s1.filter(pl.col("entity_id").is_in(data.load_split_ids(args.queries)))
        queries = pool_q.sample(min(args.n, pool_q.height), seed=config.BLOCK_SEED)
        tag = f"{args.split}_{args.queries if args.split == 'train' else 'all'}_n{queries.height}"
        out = config.CAND_DIR / "dev" / f"{tag}_candidates.parquet"
        if args.eval_only:
            c = pl.read_parquet(out)
        else:
            c = B.run_queries(args.split, st, queries, config.BLOCK_KMAX)
            out.parent.mkdir(parents=True, exist_ok=True)
            c.write_parquet(out)
            B.log(f"wrote {out} ({c.height:,} rows)")
        if args.split == "train":
            report = evaluate(c, queries, config.BLOCK_K, f"dev, {args.queries} sample")
            if args.exact_check:
                report += "\n" + exact_check(st, queries, args.exact_check) + "\n"
            if args.pool_experiment:
                report += "\n" + pool_experiment(st, queries) + "\n"
            REPORTS.mkdir(exist_ok=True)
            (REPORTS / f"blocking_{tag}.md").write_text(report)
            print(report)
        return 0

    # full: every S1 of the split, final k, written chunk by chunk
    out = B.run_full(args.split, tsv=config.OUTPUT_DIR / "candidate_pairs.tsv" if args.split == "test" else None)
    if args.split == "train":
        val = s1.filter(pl.col("entity_id").is_in(data.load_split_ids("val_ids")))
        c = pl.scan_parquet(out).filter(pl.col("s1_id").is_in(val["entity_id"].implode())).collect()
        report = evaluate(c, val, config.BLOCK_K, "full, val_ids")
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "blocking_train_full_val_ids.md").write_text(report)
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
