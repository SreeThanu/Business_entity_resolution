"""Blocking / candidate generation. See BLOCKING.md.

Inputs are the Stage 1/2 cleaned files (CLEANING.md, "Cleaned data contract"); only contract
columns are used. Labels are never used here: this module only generates candidates.

Passes (the candidate set is their union; every pass stays inside one country):
    P1  char_wb 3-5-gram TF-IDF cosine on name_core + addr_core, top-k S2/S3 per S1
    P2  the same on the name only (name_core + name_nospace + name_domain_stem), smaller k
    P3  exact key join on (house number, state) and (house number, address token)
    P4  reverse P1: top-k' S1 per S2/S3 record (one-to-one matching: recovers matches that chain
        look-alikes push out of the S1-side top-k)

Scale (see BLOCKING.md for the measurements). Each text is hashed once into per-(source, country)
shards of raw n-gram counts, cached on disk with the split's document frequencies. Search runs
query-chunk x pool-shard: sparse_dot_topn retrieves the top-M pool rows per query using only
n-grams with document frequency <= BLOCK_DF_CAP (common n-grams such as "roa"/"ltd" dominate the
cost of an exact search), then the retrieved rows are re-scored with the exact cosine over all
n-grams, and a running top-k by exact cosine is kept. Memory is bounded by the chunk and shard sizes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numba
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber import config
from ber.normalize import NORMALIZE_VERSION
from ber.submission import validate_candidate_tsv, write_candidate_tsv  # noqa: F401 (re-exported)

N_THREADS = os.cpu_count() or 2
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
                        n_features=config.BLOCK_N_FEATURES, alternate_sign=False, norm=None, dtype=np.float32)
KINDS = ("p1", "p2")
TIMES: dict[str, float] = defaultdict(float)  # seconds per step, for throughput logs and --probe


def log(msg: str) -> None:
    from ber.memguard import rss_gib
    print(f"[{time.strftime('%H:%M:%S')} rss {rss_gib():.2f} GiB] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Texts (contract columns only)
# ---------------------------------------------------------------------------------------------

def text_expr(kind: str) -> pl.Expr:
    """P1: name_core + addr_core. P2: name_core + name_nospace (only when it differs, so glued
    spellings like "sarthitrading" can hit) + name_domain_stem."""
    core = pl.col("name_core")
    if kind == "p1":
        parts = [core, pl.col("addr_core").fill_null("")]
    elif kind == "p2":
        nospace = pl.when(pl.col("name_nospace") != core).then(pl.col("name_nospace")).otherwise(pl.lit(""))
        parts = [core, nospace, pl.col("name_domain_stem").fill_null("")]
    else:
        raise ValueError(kind)
    return pl.concat_str(parts, separator=" ").str.replace_all(r"\s+", " ").str.strip_chars().alias("text")


def _transform(texts: list[str]) -> sparse.csr_matrix:
    X = _HV.transform(texts)
    X.sort_indices()
    return X


# ---------------------------------------------------------------------------------------------
# Vector store: cached raw-count shards per (split, kind, source, country)
# ---------------------------------------------------------------------------------------------

@dataclass
class Shard:
    source: int
    country: str
    part: int
    n: int
    path: str  # relative to the store dir


class VectorStore:
    """Raw char n-gram counts for every record of one split and one text kind.

    Layout: CAND_DIR/cache/{split}/{kind}/ with manifest.json, df.npy (document frequency per
    hashed n-gram over S1+S2+S3 of the split), and s{source}_{country}_{part}.npz (+ _ids.parquet).
    Rebuilt when NORMALIZE_VERSION or the n-gram settings change.
    """

    def __init__(self, split: str, kind: str, root: Path | None = None):
        self.split, self.kind = split, kind
        self.dir = (root or config.CAND_DIR / "cache") / split / kind
        self._idf = self._keep = None

    # --- build ---------------------------------------------------------------------------------
    def _meta(self) -> dict:
        return {"NORMALIZE_VERSION": NORMALIZE_VERSION, "n_features": config.BLOCK_N_FEATURES,
                "ngram": "char_wb 3-5", "shard_rows": config.BLOCK_SHARD_ROWS, "text": self.kind}

    def ready(self) -> bool:
        m = self.dir / "manifest.json"
        return m.exists() and json.loads(m.read_text()).get("meta") == self._meta()

    def build(self, procs: int = N_THREADS) -> None:
        if self.ready():
            return
        if not self.dir.parents[2].exists():
            raise FileNotFoundError(
                f"{self.dir.parents[2]} does not exist. On the Mac it must be a symlink to the external SSD "
                "(ln -s <external disk>/ber_data/cand data/cand); on Colab: mkdir -p data/cand")
        self.dir.mkdir(parents=True, exist_ok=True)
        df = np.zeros(config.BLOCK_N_FEATURES, dtype=np.int32)
        shards, n_docs, t = [], 0, time.time()
        with ProcessPoolExecutor(procs, mp_context=get_context("spawn")) as ex:
            for source in config.SOURCES:
                lf = pl.scan_parquet(config.clean_source_path(self.split, source))
                for country in sorted(lf.select(pl.col("country").unique()).collect()["country"]):
                    d = lf.filter(pl.col("country") == country).select("entity_id", text_expr(self.kind)).collect()
                    for part, a in enumerate(range(0, d.height, config.BLOCK_SHARD_ROWS)):
                        chunk = d.slice(a, config.BLOCK_SHARD_ROWS)
                        texts = chunk["text"].to_list()
                        step = 20_000
                        X = sparse.vstack(list(ex.map(_transform, [texts[i:i + step] for i in range(0, len(texts), step)])),
                                          format="csr")
                        df += np.bincount(X.indices, minlength=config.BLOCK_N_FEATURES).astype(np.int32)
                        X.data = np.minimum(X.data, 255).astype(np.uint8)
                        X.indices, X.indptr = X.indices.astype(np.int32), X.indptr.astype(np.int32)
                        name = f"s{source}_{slug(country)}_{part:03d}"
                        sparse.save_npz(self.dir / f"{name}.npz", X, compressed=False)
                        chunk.select("entity_id").write_parquet(self.dir / f"{name}_ids.parquet")
                        shards.append(Shard(source, country, part, chunk.height, name).__dict__)
                        n_docs += chunk.height
                    log(f"store {self.split}/{self.kind}: S{source} {country} {d.height:,} rows ({time.time() - t:.0f}s)")
        np.save(self.dir / "df.npy", df)
        (self.dir / "manifest.json").write_text(json.dumps({"meta": self._meta(), "n_docs": n_docs, "shards": shards}))

    # --- read ----------------------------------------------------------------------------------
    @property
    def manifest(self) -> dict:
        return json.loads((self.dir / "manifest.json").read_text())

    def shards(self, sources: tuple[int, ...], country: str) -> list[Shard]:
        return [Shard(**s) for s in self.manifest["shards"] if s["source"] in sources and s["country"] == country]

    def countries(self, source: int) -> list[str]:
        return sorted({s["country"] for s in self.manifest["shards"] if s["source"] == source})

    def weights(self, idf_from: "VectorStore | None" = None) -> tuple[np.ndarray, np.ndarray]:
        """(idf, keep) from this store's df, or another store's (e.g. train IDF on test rows).
        idf = ln((1+N)/(1+df)) + 1 (sklearn smooth_idf); keep = df <= BLOCK_DF_CAP * N."""
        src = idf_from or self
        if src._idf is None:
            df, n = np.load(src.dir / "df.npy"), src.manifest["n_docs"]
            src._idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
            src._keep = df <= config.BLOCK_DF_CAP * n
        return src._idf, src._keep

    def ids(self, shard: Shard) -> pl.Series:
        return pl.read_parquet(self.dir / f"{shard.path}_ids.parquet")["entity_id"]

    def load(self, shard: Shard, rows: np.ndarray | None = None, idf_from: "VectorStore | None" = None):
        """(full, capped) L2-normalised TF-IDF CSR matrices of one shard (optionally a row subset)."""
        X = sparse.load_npz(self.dir / f"{shard.path}.npz")
        if rows is not None:
            X = X[rows]
        idf, keep = self.weights(idf_from)
        return weigh(X, idf, keep)


def slug(country: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in country.lower())


def weigh(X: sparse.csr_matrix, idf: np.ndarray, keep: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    X = X.astype(np.float32)
    X.data *= idf[X.indices]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = sparse.csr_matrix(sparse.diags(1.0 / norms).astype(np.float32) @ X)
    X.sort_indices()
    C = X.copy()
    C.data *= keep[C.indices]
    C.eliminate_zeros()
    return X, C


# ---------------------------------------------------------------------------------------------
# Top-k search: capped retrieval + exact re-ranking
# ---------------------------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _rowdot(a_ptr, a_ind, a_dat, b_ptr, b_ind, b_dat, ra, rb, out):
    for t in numba.prange(ra.size):
        p, pe = a_ptr[ra[t]], a_ptr[ra[t] + 1]
        q, qe = b_ptr[rb[t]], b_ptr[rb[t] + 1]
        s = 0.0
        while p < pe and q < qe:
            if a_ind[p] == b_ind[q]:
                s += a_dat[p] * b_dat[q]
                p += 1
                q += 1
            elif a_ind[p] < b_ind[q]:
                p += 1
            else:
                q += 1
        out[t] = s


def rowdot(A: sparse.csr_matrix, B: sparse.csr_matrix, ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Exact dot product of row ra[t] of A with row rb[t] of B (both CSR with sorted indices)."""
    out = np.zeros(ra.size, dtype=np.float32)
    if ra.size:
        _rowdot(A.indptr.astype(np.int64), A.indices, A.data, B.indptr.astype(np.int64), B.indices, B.data,
                ra.astype(np.int64), rb.astype(np.int64), out)
    return out


class TopK:
    """Running top-k per query by exact cosine. idx = global pool index (-1 = empty)."""

    def __init__(self, n: int, k: int):
        self.k = k
        self.idx = np.full((n, k), -1, dtype=np.int64)
        self.val = np.full((n, k), -np.inf, dtype=np.float32)

    def merge(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_per_row_max: int) -> None:
        if rows.size == 0:
            return
        n = self.idx.shape[0]
        order = np.lexsort((-vals, rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
        starts = np.searchsorted(rows, np.arange(n))
        pos = np.arange(rows.size) - starts[rows]
        keep = pos < n_per_row_max
        ni = np.full((n, n_per_row_max), -1, dtype=np.int64)
        nv = np.full((n, n_per_row_max), -np.inf, dtype=np.float32)
        ni[rows[keep], pos[keep]] = cols[keep]
        nv[rows[keep], pos[keep]] = vals[keep]
        I, V = np.concatenate([self.idx, ni], 1), np.concatenate([self.val, nv], 1)
        top = np.argpartition(-V, self.k - 1, axis=1)[:, :self.k]
        self.idx, self.val = np.take_along_axis(I, top, 1), np.take_along_axis(V, top, 1)

    def sorted(self) -> tuple[np.ndarray, np.ndarray]:
        o = np.argsort(-self.val, axis=1, kind="stable")
        return np.take_along_axis(self.idx, o, 1), np.take_along_axis(self.val, o, 1)


def search_shard(top: TopK, Qf, Qc, Xf, Xc, offset: int, m: int) -> None:
    """Retrieve top-m rows of one pool shard per query by capped score, re-score exactly, merge
    the best top.k of them (by exact cosine) into the running top-k."""
    t = time.time()
    XcT = Xc.T.tocsr()
    TIMES["transpose"] += time.time() - t
    t = time.time()
    R = sp_matmul_topn(Qc, XcT, top_n=m, n_threads=N_THREADS).tocsr()
    TIMES["retrieve"] += time.time() - t
    t = time.time()
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    cols = R.indices.astype(np.int64)
    exact = rowdot(Qf, Xf, rows, cols)
    top.merge(rows, cols + offset, exact, top.k)
    TIMES["rerank"] += time.time() - t


def search(store: VectorStore, Qf, Qc, pool_sources: tuple[int, ...], country: str, k: int, m: int,
           pool_store: VectorStore | None = None) -> TopK:
    """Top-k pool rows (global index over the country's pool shards, in manifest order) per query,
    re-ranked by exact cosine from the top-m capped-score rows of every shard (m >= k)."""
    pool_store = pool_store or store
    top, offset = TopK(Qf.shape[0], k), 0
    for sh in pool_store.shards(pool_sources, country):
        t = time.time()
        Xf, Xc = pool_store.load(sh, idf_from=store)
        TIMES["load"] += time.time() - t
        search_shard(top, Qf, Qc, Xf, Xc, offset, max(m, k))
        offset += sh.n
        del Xf, Xc
    return top


def pool_ids(store: VectorStore, sources: tuple[int, ...], country: str) -> pl.DataFrame:
    """entity_id <-> global pool index (gidx) for the country's pool, in search order."""
    parts = [store.ids(sh) for sh in store.shards(sources, country)]
    ids = pl.concat(parts) if parts else pl.Series("entity_id", [], dtype=pl.String)
    return pl.DataFrame({"entity_id": ids}).with_row_index("gidx").with_columns(pl.col("gidx").cast(pl.Int64))


def load_rows(store: VectorStore, source: int, country: str, ids: pl.Series,
              idf_from: VectorStore | None = None) -> tuple[pl.Series, sparse.csr_matrix, sparse.csr_matrix]:
    """Vectors of the given entity_ids of one source/country, in the order they appear in the store."""
    want = set(ids.to_list()) if ids is not None else None
    out_ids, F, C = [], [], []
    for sh in store.shards((source,), country):
        sid = store.ids(sh)
        mask = sid.is_in(ids.implode()).to_numpy() if want is not None else np.ones(sh.n, bool)
        if mask.any():
            f, c = store.load(sh, rows=np.flatnonzero(mask), idf_from=idf_from)
            out_ids.append(sid.filter(pl.Series(mask)))
            F.append(f)
            C.append(c)
    if not F:
        empty = sparse.csr_matrix((0, config.BLOCK_N_FEATURES), dtype=np.float32)
        return pl.Series("entity_id", [], dtype=pl.String), empty, empty
    return pl.concat(out_ids), sparse.vstack(F, format="csr"), sparse.vstack(C, format="csr")


def topk_frame(q_ids: pl.Series, top: TopK, pool: pl.DataFrame, cos: str, rank: str) -> pl.DataFrame:
    """Long (q_id, cand_id, cos, rank) frame from a TopK; rank is 1-based by exact cosine."""
    I, V = top.sorted()
    n, k = I.shape
    ok = (I >= 0).ravel()
    q = np.repeat(np.arange(n), k)[ok]
    f = pl.DataFrame({
        "q_id": q_ids.gather(q),
        "gidx": I.ravel()[ok],
        cos: V.ravel()[ok],
        rank: np.tile(np.arange(1, k + 1, dtype=np.int32), n)[ok],
    })
    return f.join(pool, on="gidx", how="left").select("q_id", pl.col("entity_id").alias("cand_id"), cos, rank)


# ---------------------------------------------------------------------------------------------
# P3: address keys (no text similarity)
# ---------------------------------------------------------------------------------------------

_P3_MIN_TOKEN = 4  # address tokens shorter than this are too generic to key on


def p3_keys(lf: pl.LazyFrame, state: pl.LazyFrame) -> pl.LazyFrame:
    """(entity_id, country, key) for records with a house number: one key per state and per
    alphabetic addr_core token (>= 4 letters), each combined with the house number."""
    base = (lf.filter(pl.col("addr_house_number").is_not_null())
            .select("entity_id", "country", "addr_house_number", "addr_core")
            .join(state, on="entity_id", how="left"))
    by_state = base.filter(pl.col("addr_state_canon").is_not_null()).select(
        "entity_id", "country", pl.concat_str([pl.lit("S|"), "addr_house_number", pl.lit("|"), "addr_state_canon"]).alias("key"))
    by_token = (base.select("entity_id", "country", "addr_house_number",
                            pl.col("addr_core").str.split(" ").alias("tok"))
                .explode("tok")
                .filter(pl.col("tok").str.contains(rf"^\p{{L}}{{{_P3_MIN_TOKEN},}}$"))
                .select("entity_id", "country",
                        pl.concat_str([pl.lit("T|"), "addr_house_number", pl.lit("|"), "tok"]).alias("key")))
    return pl.concat([by_state, by_token]).unique()


def p3_pairs(split: str, s1_ids: pl.Series, max_block: int = config.BLOCK_P3_MAX_BLOCK) -> tuple[pl.DataFrame, dict]:
    """Pairs (q_id, cand_id) sharing a P3 key, same country. Keys with more than `max_block` pool
    records are dropped (reported in the stats)."""
    def state(s: int) -> pl.LazyFrame:
        p = config.stage2_source_path(split, s)
        return pl.scan_parquet(p).select("entity_id", "addr_state_canon")

    q = p3_keys(pl.scan_parquet(config.clean_source_path(split, 1)).filter(pl.col("entity_id").is_in(s1_ids.implode())),
                state(1)).collect()
    pool = pl.concat([p3_keys(pl.scan_parquet(config.clean_source_path(split, s)), state(s)) for s in (2, 3)])
    pool = pool.join(q.select("country", "key").unique().lazy(), on=["country", "key"], how="semi").collect()
    size = pool.group_by("country", "key").len("block")
    big = size.filter(pl.col("block") > max_block)
    pool = pool.join(big.select("country", "key"), on=["country", "key"], how="anti")
    pairs = (q.join(pool, on=["country", "key"], suffix="_c")
             .select(pl.col("entity_id").alias("q_id"), pl.col("entity_id_c").alias("cand_id")).unique())
    stats = {"query_keys": q.height, "blocks": size.height, "blocks_dropped": big.height,
             "pool_rows_in_dropped_blocks": int(big["block"].sum() or 0)}
    return pairs, stats


# ---------------------------------------------------------------------------------------------
# Passes over one country and one S1 query set
# ---------------------------------------------------------------------------------------------

def forward(stores: dict[str, VectorStore], country: str, q_ids: pl.Series, kmax: dict[str, int]) -> pl.DataFrame:
    """P1 + P2 for the given S1 ids of one country: q_id, cand_id, p1_cos, p1_rank, p2_cos, p2_rank."""
    out = None
    for kind, (cos, rank) in (("p1", ("p1_cos", "p1_rank")), ("p2", ("p2_cos", "p2_rank"))):
        st = stores[kind]
        ids, Qf, Qc = load_rows(st, 1, country, q_ids)
        top = search(st, Qf, Qc, (2, 3), country, kmax[kind], config.BLOCK_RETRIEVE_M[kind])
        f = topk_frame(ids, top, pool_ids(st, (2, 3), country), cos, rank)
        out = f if out is None else out.join(f, on=["q_id", "cand_id"], how="full", coalesce=True)
    return out


def reverse_all(store: VectorStore, country: str, k: int, out_dir: Path) -> None:
    """P4 for every S2/S3 record of one country, one file per S2/S3 shard in out_dir
    (cand_id, q_id = S1, p1_cos, p1_rank_rev). Shards already written are skipped (resumable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s1 = None
    for sh in store.shards((2, 3), country):
        path = out_dir / f"{sh.path}.parquet"
        if path.exists():
            continue
        s1 = s1 if s1 is not None else pool_ids(store, (1,), country)
        t = time.time()
        Qf, Qc = store.load(sh)
        top = search(store, Qf, Qc, (1,), country, k, config.BLOCK_RETRIEVE_M["p4"])
        f = topk_frame(store.ids(sh), top, s1, "p1_cos", "p1_rank_rev").rename({"q_id": "cand_id", "cand_id": "q_id"})
        _atomic_write(f, path)
        log(f"P4 {store.split} {country} {sh.path}: {sh.n:,} reverse queries ({time.time() - t:.0f}s)")


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def fill_cosines(stores: dict[str, VectorStore], country: str, pairs: pl.DataFrame) -> pl.DataFrame:
    """Exact p1_cos / p2_cos for pairs where that pass did not retrieve them (pure feature fill)."""
    for kind in KINDS:
        col = f"{kind}_cos"
        need = pairs.filter(pl.col(col).is_null())
        if need.height == 0:
            continue
        st = stores[kind]
        ids, Qf, _ = load_rows(st, 1, country, need["q_id"].unique())
        qmap = pl.DataFrame({"q_id": ids}).with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int64))
        pool = pool_ids(st, (2, 3), country)
        need = need.select("q_id", "cand_id").join(qmap, on="q_id").join(pool.rename({"entity_id": "cand_id"}), on="cand_id")
        vals, offset = [], 0
        for sh in st.shards((2, 3), country):
            part = need.filter((pl.col("gidx") >= offset) & (pl.col("gidx") < offset + sh.n))
            if part.height:
                Xf, _ = st.load(sh)
                v = rowdot(Qf, Xf, part["qi"].to_numpy(), part["gidx"].to_numpy() - offset)
                vals.append(part.select("q_id", "cand_id").with_columns(pl.Series(f"_{col}", v)))
            offset += sh.n
        if vals:
            pairs = (pairs.join(pl.concat(vals), on=["q_id", "cand_id"], how="left")
                     .with_columns(pl.coalesce(col, f"_{col}").alias(col)).drop(f"_{col}"))
    return pairs


def union(country: str, fwd: pl.DataFrame, p3: pl.DataFrame, p4: pl.DataFrame) -> pl.DataFrame:
    """Candidate rows: s1_id, cand_id, country, in_p1..in_p4, cosines and ranks."""
    p3 = p3.with_columns(in_p3=pl.lit(True))
    p4 = p4.select("q_id", "cand_id", pl.col("p1_cos").alias("_p4_cos"), "p1_rank_rev")
    u = (fwd.join(p3, on=["q_id", "cand_id"], how="full", coalesce=True)
         .join(p4, on=["q_id", "cand_id"], how="full", coalesce=True))
    return u.select(
        pl.col("q_id").alias("s1_id"), "cand_id", pl.lit(country).alias("country"),
        pl.col("p1_rank").is_not_null().alias("in_p1"), pl.col("p2_rank").is_not_null().alias("in_p2"),
        pl.col("in_p3").fill_null(False), pl.col("p1_rank_rev").is_not_null().alias("in_p4"),
        pl.coalesce("p1_cos", "_p4_cos").cast(pl.Float32).alias("p1_cos"), pl.col("p1_rank").cast(pl.Int32),
        pl.col("p1_rank_rev").cast(pl.Int32), pl.col("p2_cos").cast(pl.Float32), pl.col("p2_rank").cast(pl.Int32),
    )


def select_k(c: pl.DataFrame | pl.LazyFrame, k: dict[str, int], p3: bool = True,
             p4_cap: int | None = config.BLOCK_P4_MAX_PER_S1):
    """Candidate set for given k values (ranks are 1-based).

    P4 pairs are capped at `p4_cap` per S1 (highest P1 cosine first): a short or generic S1 text
    ("onyx") can sit in the reverse top-k of thousands of S2/S3 records.
    """
    in4 = (pl.col("p1_rank_rev") <= k["p4"]).fill_null(False)
    if p4_cap:
        in4 = in4 & (pl.when(in4).then(pl.col("p1_cos")).rank("ordinal", descending=True).over("s1_id") <= p4_cap)
    keep = (pl.col("p1_rank") <= k["p1"]).fill_null(False) | (pl.col("p2_rank") <= k["p2"]).fill_null(False) | in4
    if p3:
        keep = keep | pl.col("in_p3")
    return c.filter(keep)


# ---------------------------------------------------------------------------------------------
# Pipeline: P4 cache, chunked runs, full mode
# ---------------------------------------------------------------------------------------------

def stores(split: str) -> dict[str, VectorStore]:
    out = {k: VectorStore(split, k) for k in KINDS}
    for st in out.values():
        st.build()
    return out


def p4_path(split: str, country: str) -> Path:
    """P4 cache for one country: a folder with one file per S2/S3 shard (older runs: one file with
    the same name + .parquet, still read). Keyed on every setting that changes the result."""
    return (config.CAND_PERSIST_DIR / "cache" / split /
            f"p4_{slug(country)}_k{config.BLOCK_KMAX['p4']}_cap{config.BLOCK_DF_CAP}_m{config.BLOCK_RETRIEVE_M['p4']}")


def p4_ready(split: str, st: VectorStore, country: str) -> bool:
    d = p4_path(split, country)
    return (d.parent / f"{d.name}.parquet").exists() or all(
        (d / f"{sh.path}.parquet").exists() for sh in st.shards((2, 3), country))


def p4_cached(split: str, st: VectorStore, country: str) -> pl.LazyFrame:
    """Reverse top-KMAX for every S2/S3 record of the country (independent of the queries); built
    and cached on first use, resumable per shard; returned lazily so each chunk reads only its rows."""
    d = p4_path(split, country)
    legacy = d.parent / f"{d.name}.parquet"
    if legacy.exists():
        return pl.scan_parquet(legacy)
    if not p4_ready(split, st, country):
        t = time.time()
        reverse_all(st, country, config.BLOCK_KMAX["p4"], d)
        log(f"P4 {split} {country}: cache complete ({time.time() - t:.0f}s)")
    return pl.scan_parquet(d / "*.parquet")


def process_chunk(split: str, st: dict[str, VectorStore], country: str, q: pl.Series, k: dict[str, int],
                  final: bool) -> pl.DataFrame:
    """All four passes for one chunk of S1 queries of one country. final=True applies select_k
    (incl. the P4 cap); every S1's pairs are in its chunk, so that equals applying it globally."""
    t0, times0 = time.time(), dict(TIMES)
    p3, p3_stats = p3_pairs(split, q)
    TIMES["p3"] += time.time() - t0
    fwd = forward(st, country, q, k)
    p4q = p4_cached(split, st["p1"], country).filter(
        (pl.col("p1_rank_rev") <= k["p4"]) & pl.col("q_id").is_in(q.implode())).collect()
    u = union(country, fwd, p3, p4q)
    t = time.time()
    u = fill_cosines(st, country, u.rename({"s1_id": "q_id"})).rename({"q_id": "s1_id"})
    TIMES["fill"] += time.time() - t
    if final:
        u = select_k(u, k)
    steps = {key: round(TIMES[key] - times0.get(key, 0.0)) for key in ("p3", "load", "transpose", "retrieve", "rerank", "fill")}
    log(f"{country}: {len(q):,} queries -> {u.height:,} candidates in {time.time() - t0:.0f}s {steps}; "
        f"P3 dropped {p3_stats['blocks_dropped']:,}/{p3_stats['blocks']:,} blocks")
    return u


def run_queries(split: str, st: dict[str, VectorStore], queries: pl.DataFrame, k: dict[str, int],
                final: bool = False) -> pl.DataFrame | None:
    """In-memory run for a query sample (dev mode): queries = entity_id, country."""
    out = []
    pool_countries = set(st["p1"].countries(2)) | set(st["p1"].countries(3))
    for country in sorted(queries["country"].unique()):
        if country not in pool_countries:
            log(f"{country}: no S2/S3 pool, skipped")
            continue
        q_all = queries.filter(pl.col("country") == country)["entity_id"].sort()
        for a in range(0, len(q_all), config.BLOCK_QUERY_CHUNK):
            out.append(process_chunk(split, st, country, q_all.slice(a, config.BLOCK_QUERY_CHUNK), k, final))
    return pl.concat(out) if out else None


def run_fingerprint(split: str, queries: pl.DataFrame) -> dict:
    ids = "\n".join(sorted(queries["entity_id"].to_list())).encode()
    return {"NORMALIZE_VERSION": NORMALIZE_VERSION, "split": split, "k": config.BLOCK_K,
            "p4_cap": config.BLOCK_P4_MAX_PER_S1, "p3_max_block": config.BLOCK_P3_MAX_BLOCK,
            "df_cap": config.BLOCK_DF_CAP, "retrieve_m": config.BLOCK_RETRIEVE_M,
            "query_chunk": config.BLOCK_QUERY_CHUNK, "queries_sha256": hashlib.sha256(ids).hexdigest()}


def run_candidates(split: str, queries: pl.DataFrame, name: str, tsv: Path | None = None) -> Path:
    """Resumable full run: final-k candidates for `queries` (entity_id, country) of `split`.

    Each (country, chunk) goes to CAND_DIR/{name}.parts/{country}_{i}.parquet (written atomically);
    chunks that already exist are skipped, so a run can be restarted after a disconnect. The parts
    folder records the settings; resuming with different settings is refused. When every chunk is
    present they are merged into CAND_DIR/{name}_candidates.parquet (row count checked) and, for the
    official file, output/candidate_pairs.tsv is written and validated.
    """
    st = stores(split)
    parts = config.CAND_PERSIST_DIR / f"{name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    fp, manifest = run_fingerprint(split, queries), parts / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fp:
        raise RuntimeError(f"{parts} was made with different settings or queries; delete it to start over:\n"
                           f"{manifest.read_text()}\nnow: {json.dumps(fp)}")
    manifest.write_text(json.dumps(fp))
    pool_countries = set(st["p1"].countries(2)) | set(st["p1"].countries(3))
    expected = []
    for country in sorted(queries["country"].unique()):
        if country not in pool_countries:
            log(f"{country}: no S2/S3 pool, its queries get no candidates")
            continue
        q_all = queries.filter(pl.col("country") == country)["entity_id"].sort()
        n_chunks = math.ceil(len(q_all) / config.BLOCK_QUERY_CHUNK)
        for i in range(n_chunks):
            path = parts / f"{slug(country)}_{i:04d}.parquet"
            expected.append(path)
            if path.exists():
                log(f"{path.name}: done earlier, skipped")
                continue
            q = q_all.slice(i * config.BLOCK_QUERY_CHUNK, config.BLOCK_QUERY_CHUNK)
            _atomic_write(process_chunk(split, st, country, q, config.BLOCK_K, final=True), path)
            log(f"{path.name}: chunk {i + 1}/{n_chunks} written")
    out = merge_parts(parts, expected, config.CAND_PERSIST_DIR / f"{name}_candidates.parquet")
    if tsv is not None:
        write_candidate_tsv(pl.scan_parquet(out), queries["entity_id"], tsv)
        errors = validate_candidate_tsv(tsv, set(queries["entity_id"].to_list()))
        if errors:
            raise ValueError(f"{tsv} failed validation: {errors}")
        log(f"{tsv}: valid ({queries.height:,} rows)")
    return out


def merge_parts(parts: Path, expected: list[Path], out: Path) -> Path:
    """Concatenate the chunk files into one parquet (streaming), check the row count, then delete them."""
    missing = [p.name for p in expected if not p.exists()]
    if missing or not expected:
        raise RuntimeError(f"cannot merge: missing chunks {missing}" if missing else "no chunks to merge")
    n_parts = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in expected)
    tmp = out.with_name(out.name + ".tmp")
    pl.scan_parquet(expected).sink_parquet(tmp, compression="zstd")
    n_out = pl.scan_parquet(tmp).select(pl.len()).collect().item()
    if n_out != n_parts:
        raise RuntimeError(f"merge wrote {n_out:,} rows, parts have {n_parts:,}")
    tmp.replace(out)
    for p in expected:
        p.unlink()
    (parts / "run.json").unlink(missing_ok=True)
    parts.rmdir()
    log(f"merged {len(expected)} chunks -> {out} ({n_out:,} rows)")
    return out


# ---------------------------------------------------------------------------------------------
# Probe: measured throughput -> estimated time of a full run
# ---------------------------------------------------------------------------------------------

def probe(split: str, queries: pl.DataFrame, n_probe: int = 5_000) -> dict:
    """Time every step on ONE pool shard of the largest country and extrapolate to `queries`.

    Measures per shard: load (+ weighting), transpose, and per query: retrieve + re-rank (P1, P2,
    P4 reverse). The estimate is sum over countries of chunks x shards x fixed cost +
    queries x shards x per-query cost, plus P3 per chunk and the P4 cache if it is not built yet.
    """
    t = time.time()
    st = stores(split)
    t_build = time.time() - t
    countries = [c for c in sorted(queries["country"].unique()) if st["p1"].shards((2, 3), c)]
    size = {c: sum(sh.n for sh in st["p1"].shards((2, 3), c)) for c in countries}
    country = max(countries, key=size.get)
    s1_all = pl.read_parquet(config.clean_source_path(split, 1), columns=["entity_id", "country"])
    qs = s1_all.filter(pl.col("country") == country).sample(n_probe, seed=config.BLOCK_SEED)["entity_id"]
    meas = {}
    for kind in KINDS:
        ids, Qf, Qc = load_rows(st[kind], 1, country, qs)
        sh = st[kind].shards((2, 3), country)[0]
        TIMES.clear()
        t = time.time(); Xf, Xc = st[kind].load(sh); t_load = time.time() - t
        top = TopK(Qf.shape[0], config.BLOCK_K[kind])
        search_shard(top, Qf, Qc, Xf, Xc, 0, config.BLOCK_RETRIEVE_M[kind])
        meas[kind] = {"fixed": t_load + TIMES["transpose"],
                      "per_query": (TIMES["retrieve"] + TIMES["rerank"]) / len(ids) * config.BLOCK_SHARD_ROWS / sh.n}
    # P4: reverse queries = rows of one pool shard against one S1 shard
    sh = st["p1"].shards((2, 3), country)[0]
    Qf, Qc = st["p1"].load(sh, rows=np.arange(min(n_probe, sh.n)))
    s1sh = st["p1"].shards((1,), country)[0]
    TIMES.clear()
    t = time.time(); Xf, Xc = st["p1"].load(s1sh); t_load = time.time() - t
    search_shard(TopK(Qf.shape[0], config.BLOCK_KMAX["p4"]), Qf, Qc, Xf, Xc, 0, config.BLOCK_RETRIEVE_M["p4"])
    meas["p4"] = {"fixed": t_load + TIMES["transpose"],
                  "per_query": (TIMES["retrieve"] + TIMES["rerank"]) / Qf.shape[0] * config.BLOCK_SHARD_ROWS / s1sh.n}
    t = time.time(); p3_pairs(split, qs.head(1000)); t_p3 = time.time() - t

    est = {"p1": 0.0, "p2": 0.0, "fill": 0.0, "p3": 0.0, "p4_cache": 0.0}
    for c in countries:
        nq = queries.filter(pl.col("country") == c).height
        chunks = math.ceil(nq / config.BLOCK_QUERY_CHUNK)
        n_pool_sh = len(st["p1"].shards((2, 3), c))
        for kind in KINDS:
            est[kind] += chunks * n_pool_sh * meas[kind]["fixed"] + nq * n_pool_sh * meas[kind]["per_query"]
            est["fill"] += chunks * n_pool_sh * meas[kind]["fixed"]  # one more pass over the pool shards
        est["p3"] += chunks * t_p3
        if not p4_ready(split, st["p1"], c):
            n_s1_sh = len(st["p1"].shards((1,), c))
            est["p4_cache"] += n_pool_sh * n_s1_sh * meas["p4"]["fixed"] + size[c] * n_s1_sh * meas["p4"]["per_query"]
    total = sum(est.values())
    return {"probe_country": country, "probe_queries": n_probe, "cores": N_THREADS, "store_build_s": round(t_build),
            "measured": {k: {kk: round(v, 6) for kk, v in m.items()} for k, m in meas.items()}, "p3_per_chunk_s": round(t_p3),
            "queries": queries.height, "estimate_s": {k: round(v) for k, v in est.items()},
            "estimate_total_h": round(total / 3600, 2)}


# ---------------------------------------------------------------------------------------------
# Evaluation helpers (labels only measure recall)
# ---------------------------------------------------------------------------------------------

def recall(cands: pl.DataFrame, truth: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Pair recall and "entity: all matches found".

    cands: s1_id, cand_id. truth: s1_id, cand_id (+ any `by` columns), only for the queried S1s.
    Entities without matches are excluded from the entity metric (nothing to find).
    """
    by = by or []
    t = truth.join(cands.select("s1_id", "cand_id").unique().with_columns(found=pl.lit(True)),
                   on=["s1_id", "cand_id"], how="left").with_columns(pl.col("found").fill_null(False))
    ent = t.group_by(["s1_id", *by]).agg(all_found=pl.col("found").all())
    pr = t.group_by(by).agg(true_pairs=pl.len(), pair_recall=pl.col("found").mean()) if by else \
        t.select(true_pairs=pl.len(), pair_recall=pl.col("found").mean())
    er = ent.group_by(by).agg(entities=pl.len(), entity_all_found=pl.col("all_found").mean()) if by else \
        ent.select(entities=pl.len(), entity_all_found=pl.col("all_found").mean())
    return pr.join(er, on=by) if by else pl.concat([pr, er], how="horizontal")


def oracle_f05(cands: pl.DataFrame, queries: pl.DataFrame, truth: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Macro-F0.5 of a perfect matcher restricted to the candidates: the blocking-imposed ceiling.

    queries: s1_id (+ `by` columns), ALL queried S1s, singletons included. truth: s1_id, cand_id.
    Per S1 (the challenge metric, EDA 3.6): no true match -> predicts nothing -> 1. Otherwise it
    predicts exactly the true matches among its candidates: precision 1, recall r = found / true,
    F0.5 = 1.25 r / (0.25 + r); nothing found -> 0.
    """
    by = by or []
    t = truth.join(cands.select("s1_id", "cand_id").unique().with_columns(found=pl.lit(True)),
                   on=["s1_id", "cand_id"], how="left").group_by("s1_id").agg(
        n_true=pl.len(), n_found=pl.col("found").fill_null(False).sum())
    per = (queries.select("s1_id", *by).join(t, on="s1_id", how="left")
           .with_columns(pl.col("n_true").fill_null(0), pl.col("n_found").fill_null(0))
           .with_columns(r=pl.col("n_found") / pl.col("n_true"))
           .with_columns(f05=pl.when(pl.col("n_true") == 0).then(1.0).when(pl.col("n_found") == 0).then(0.0)
                         .otherwise(1.25 * pl.col("r") / (0.25 + pl.col("r")))))
    agg = [pl.len().alias("entities"), (pl.col("n_true") == 0).mean().alias("singleton_share"),
           pl.col("f05").mean().alias("oracle_macro_f05")]
    return per.group_by(by).agg(agg).sort(by) if by else per.select(agg)


def cand_stats(cands: pl.DataFrame, queries: pl.Series) -> dict:
    per = (pl.DataFrame({"s1_id": queries}).join(cands.group_by("s1_id").len("n"), on="s1_id", how="left")
           .with_columns(pl.col("n").fill_null(0)))["n"]
    return {"queries": len(queries), "pairs": int(per.sum()), "mean": float(per.mean()),
            "p50": float(per.quantile(0.5)), "p95": float(per.quantile(0.95)), "max": int(per.max())}
