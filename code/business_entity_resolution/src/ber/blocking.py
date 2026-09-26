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

import json
import os
import time
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

N_THREADS = os.cpu_count() or 2
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
                        n_features=config.BLOCK_N_FEATURES, alternate_sign=False, norm=None, dtype=np.float32)
KINDS = ("p1", "p2")


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
                "(ln -s '/Volumes/thanu's T7/Business_entity_resolution/ber_data/cand' data/cand); on Colab: mkdir -p data/cand")
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
    R = sp_matmul_topn(Qc, Xc.T.tocsr(), top_n=m, n_threads=N_THREADS)
    R = R.tocsr()
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    cols = R.indices.astype(np.int64)
    exact = rowdot(Qf, Xf, rows, cols)
    top.merge(rows, cols + offset, exact, top.k)


def search(store: VectorStore, Qf, Qc, pool_sources: tuple[int, ...], country: str, k: int, m: int,
           pool_store: VectorStore | None = None) -> TopK:
    """Top-k pool rows (global index over the country's pool shards, in manifest order) per query,
    re-ranked by exact cosine from the top-m capped-score rows of every shard (m >= k)."""
    pool_store = pool_store or store
    top, offset = TopK(Qf.shape[0], k), 0
    for sh in pool_store.shards(pool_sources, country):
        Xf, Xc = pool_store.load(sh, idf_from=store)
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
        p = config.CLEAN_DIR / f"stage2_{split}_source{s}.parquet"
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


def reverse_all(store: VectorStore, country: str, k: int) -> pl.DataFrame:
    """P4 for every S2/S3 record of one country: cand_id, q_id (S1), p1_cos, p1_rank_rev."""
    s1 = pool_ids(store, (1,), country)
    frames = []
    for sh in store.shards((2, 3), country):
        Qf, Qc = store.load(sh)
        top = search(store, Qf, Qc, (1,), country, k, config.BLOCK_RETRIEVE_M["p4"])
        f = topk_frame(store.ids(sh), top, s1, "p1_cos", "p1_rank_rev")
        frames.append(f.rename({"q_id": "cand_id", "cand_id": "q_id"}))
        log(f"P4 {store.split} {country} {sh.path}: {sh.n:,} reverse queries")
    return pl.concat(frames) if frames else pl.DataFrame()


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
    return (config.CAND_DIR / "cache" / split /
            f"p4_{slug(country)}_k{config.BLOCK_KMAX['p4']}_cap{config.BLOCK_DF_CAP}_m{config.BLOCK_RETRIEVE_M['p4']}.parquet")


def p4_cached(split: str, st: VectorStore, country: str) -> pl.LazyFrame:
    """Reverse top-KMAX for every S2/S3 record of the country (independent of the queries), cached;
    returned lazily so each query chunk reads only its own rows."""
    path = p4_path(split, country)
    if not path.exists():
        t = time.time()
        f = reverse_all(st, country, config.BLOCK_KMAX["p4"])
        f.write_parquet(path)
        log(f"P4 {split} {country}: {f.height:,} reverse pairs in {time.time() - t:.0f}s")
    return pl.scan_parquet(path)


def run_queries(split: str, st: dict[str, VectorStore], queries: pl.DataFrame, k: dict[str, int],
                parts_dir: Path | None = None, final: bool = False) -> pl.DataFrame | None:
    """All passes for the given S1 queries (entity_id, country), chunked per country.

    final=True applies select_k (incl. the P4 cap) per chunk; every S1's pairs are in one chunk, so
    this equals applying it to the whole set. With parts_dir, each chunk is written to
    parts_dir/part-*.parquet and nothing is kept in memory (full mode); otherwise the frame is returned.
    """
    p3, p3_stats = p3_pairs(split, queries["entity_id"])
    log(f"P3: {p3.height:,} pairs, stats {p3_stats}")
    out, n_part = [], 0
    pool_countries = set(st["p1"].countries(2)) | set(st["p1"].countries(3))
    for country in sorted(queries["country"].unique()):
        if country not in pool_countries:
            log(f"{country}: no S2/S3 pool, skipped")
            continue
        q_all = queries.filter(pl.col("country") == country)["entity_id"]
        p4 = p4_cached(split, st["p1"], country)
        for a in range(0, len(q_all), config.BLOCK_QUERY_CHUNK):
            q = q_all.slice(a, config.BLOCK_QUERY_CHUNK)
            t = time.time()
            fwd = forward(st, country, q, k)
            p4q = p4.filter((pl.col("p1_rank_rev") <= k["p4"]) & pl.col("q_id").is_in(q.implode())).collect()
            u = union(country, fwd, p3.filter(pl.col("q_id").is_in(q.implode())), p4q)
            u = fill_cosines(st, country, u.rename({"s1_id": "q_id"})).rename({"q_id": "s1_id"})
            if final:
                u = select_k(u, k)
            if parts_dir is not None:
                u.write_parquet(parts_dir / f"part-{n_part:04d}.parquet")
                n_part += 1
            else:
                out.append(u)
            log(f"{country} queries {a:,}-{a + len(q):,}: {u.height:,} candidates ({time.time() - t:.0f}s)")
    if parts_dir is not None:
        return None
    return pl.concat(out) if out else None


def run_full(split: str, tsv: Path | None = None) -> Path:
    """Every S1 of the split at config.BLOCK_K -> config.cand_path(split); optionally the official
    candidate TSV (validated). Chunks go to a parts folder first, then are concatenated on disk."""
    st = stores(split)
    s1 = pl.read_parquet(config.clean_source_path(split, 1), columns=["entity_id", "country"])
    out = config.cand_path(split)
    parts = out.with_suffix(".parts")
    if parts.exists():
        for f in parts.glob("part-*.parquet"):
            f.unlink()
    parts.mkdir(parents=True, exist_ok=True)
    run_queries(split, st, s1, config.BLOCK_K, parts_dir=parts, final=True)
    tmp = out.with_suffix(".tmp.parquet")
    pl.scan_parquet(parts / "part-*.parquet").sink_parquet(tmp, compression="zstd")
    tmp.replace(out)
    for f in parts.glob("part-*.parquet"):
        f.unlink()
    parts.rmdir()
    log(f"wrote {out} ({pl.scan_parquet(out).select(pl.len()).collect().item():,} rows)")
    if tsv is not None:
        write_candidate_tsv(pl.scan_parquet(out), s1["entity_id"], tsv)
        errors = validate_candidate_tsv(tsv, set(s1["entity_id"].to_list()))
        if errors:
            raise ValueError(f"{tsv} failed validation: {errors}")
        log(f"{tsv}: valid ({s1.height:,} rows)")
    return out


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


def cand_stats(cands: pl.DataFrame, queries: pl.Series) -> dict:
    per = (pl.DataFrame({"s1_id": queries}).join(cands.group_by("s1_id").len("n"), on="s1_id", how="left")
           .with_columns(pl.col("n").fill_null(0)))["n"]
    return {"queries": len(queries), "pairs": int(per.sum()), "mean": float(per.mean()),
            "p50": float(per.quantile(0.5)), "p95": float(per.quantile(0.95)), "max": int(per.max())}


# ---------------------------------------------------------------------------------------------
# Submission file
# ---------------------------------------------------------------------------------------------

def write_candidate_tsv(cands: pl.DataFrame | pl.LazyFrame, s1_ids: pl.Series, path: Path) -> None:
    """Official candidate_pairs.tsv: one row per S1 (all of `s1_ids`), comma-separated unique
    S2/S3 ids, empty when none."""
    lists = (cands.lazy().select("s1_id", "cand_id").unique()
             .group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias("candidate_entity_ids"))
             .collect(engine="streaming"))
    out = (pl.DataFrame({"s1_id": s1_ids}).unique(maintain_order=True).join(lists, on="s1_id", how="left")
           .select(pl.col("s1_id").alias("source1_entity_id"), pl.col("candidate_entity_ids").fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")


def validate_candidate_tsv(path: Path, required: set[str]) -> list[str]:
    """Run the official validator's rules (utils/validate_submission.py) on a candidate file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validate_submission", Path(__file__).resolve().parents[2] / "utils" / "validate_submission.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    errors: list[str] = []
    v.validate_id_list_file(str(path), v.CANDIDATE_HEADER, "candidate_entity_ids", required, None, errors)
    return errors
