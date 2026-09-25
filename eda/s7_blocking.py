"""Section 7 - Quick blocking-recall check with char n-gram TF-IDF.

Spec: TF-IDF(analyzer="char_wb", ngram_range=(3,5)) on lowercased text, fit on train S1+S2+S3,
cosine top-k among all train S2+S3 records for each S1 query.

Implementation notes (memory: 8 GB machine, 12.5M documents):
* An exact TfidfVectorizer.fit on 12.5M docs needs >10 GB, so the vocabulary is replaced by
  HashingVectorizer(n_features=2**24, alternate_sign=False) with the same analyzer. IDF is computed
  exactly like sklearn's default (smooth_idf=True: idf = ln((1+N)/(1+df)) + 1) over all 12.5M train
  docs; rows are L2-normalized (sklearn default). Hash collisions only merge rare n-grams.
* Queries: random sample of N_QUERIES train S1 entities (seed 42). Pool: ALL train S2+S3 records
  (10.3M), processed in shards; exact brute-force cosine via an inverted-index kernel (numba).
* Two variants are recorded from the same scores: pool = all S2+S3 (spec) and pool restricted to
  records with the same country label as the query.

Usage: python s7_blocking.py [name|nameaddr]   (default runs both)
"""
import sys
import time
from multiprocessing import Pool

import numba
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from common import CACHE, SEED, load, load_pairs, md_table, write_section

N_FEAT = 2 ** 24
N_QUERIES = 12_000
KMAX = 100
KS = [5, 10, 20, 50, 100]
CHUNK = 100_000
SHARD = {"name": 2_000_000, "nameaddr": 600_000}
HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, n_features=N_FEAT,
                       alternate_sign=False, norm=None, dtype=np.float32)


def _transform(texts):
    return HV.transform(texts)


def transform_many(texts, procs=5):
    chunks = [texts[i:i + CHUNK] for i in range(0, len(texts), CHUNK)]
    with Pool(procs) as p:
        mats = p.map(_transform, chunks)
    return sparse.vstack(mats, format="csr")


def get_texts(split, i, mode):
    d = load(split, f"source{i}")
    if mode == "name":
        return d.business_name.tolist()
    return [f"{n} {a}" for n, a in zip(d.business_name.tolist(), d.business_address.tolist())]


def compute_idf(mode):
    cp = CACHE / f"s7_idf_{mode}.npy"
    if cp.exists():
        return np.load(cp)
    df = np.zeros(N_FEAT, dtype=np.int64)
    n = 0
    for i in (1, 2, 3):
        texts = get_texts("train", i, mode)
        for a in range(0, len(texts), 1_000_000):
            X = transform_many(texts[a:a + 1_000_000])
            df += np.bincount(X.indices, minlength=N_FEAT)  # indices are unique per row
            n += X.shape[0]
            del X
        print(f"  idf pass source{i} done ({n:,} docs)", flush=True)
    idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    np.save(cp, idf)
    return idf


def vectorize(texts, idf):
    X = transform_many(texts)
    X = X.multiply(idf).tocsr().astype(np.float32)
    return normalize(X, norm="l2", copy=False)


@numba.njit(parallel=True, fastmath=True)
def shard_topk(q_indptr, q_ind, q_dat, q_cty, xt_indptr, xt_ind, xt_dat, row_cty, nrows, k,
               out_i, out_v, out_ic, out_vc):
    nq = q_indptr.size - 1
    nt = numba.get_num_threads()
    for t in numba.prange(nt):
        acc = np.zeros(nrows, np.float32)
        for qi in range(t, nq, nt):
            for p in range(q_indptr[qi], q_indptr[qi + 1]):
                f = q_ind[p]
                w = q_dat[p]
                for r in range(xt_indptr[f], xt_indptr[f + 1]):
                    acc[xt_ind[r]] += w * xt_dat[r]
            # top-k over all rows and over same-country rows (simple replace-min scan)
            bi = np.full(k, -1, np.int64); bv = np.full(k, -1.0, np.float32); mn = -1.0; mp = 0
            ci = np.full(k, -1, np.int64); cv = np.full(k, -1.0, np.float32); cmn = -1.0; cmp_ = 0
            qc = q_cty[qi]
            for r in range(nrows):
                v = acc[r]
                if v > 0.0:
                    if v > mn:
                        bi[mp] = r; bv[mp] = v
                        mn = bv[0]; mp = 0
                        for j in range(1, k):
                            if bv[j] < mn:
                                mn = bv[j]; mp = j
                    if row_cty[r] == qc and v > cmn:
                        ci[cmp_] = r; cv[cmp_] = v
                        cmn = cv[0]; cmp_ = 0
                        for j in range(1, k):
                            if cv[j] < cmn:
                                cmn = cv[j]; cmp_ = j
                    acc[r] = 0.0
            out_i[qi, :] = bi; out_v[qi, :] = bv
            out_ic[qi, :] = ci; out_vc[qi, :] = cv


def merge_topk(idx_list, val_list, k):
    I = np.concatenate(idx_list, axis=1)
    V = np.concatenate(val_list, axis=1)
    o = np.argsort(-V, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(I, o, 1), np.take_along_axis(V, o, 1)


def run(mode):
    t0 = time.time()
    print(f"[{mode}] computing idf", flush=True)
    idf = compute_idf(mode)
    print(f"[{mode}] idf ready {time.time() - t0:.0f}s", flush=True)

    s1 = load("train", "source1")
    rng = np.random.default_rng(SEED)
    qpos = np.sort(rng.choice(len(s1), N_QUERIES, replace=False))
    q_ids = s1.entity_id.astype(str).iloc[qpos].tolist()
    q_country = s1.country.astype(str).iloc[qpos].tolist()
    all_texts = get_texts("train", 1, mode)
    Q = vectorize([all_texts[p] for p in qpos], idf)
    del all_texts, s1

    cty_codes = {"US": 0, "India": 1}
    q_cty = np.array([cty_codes.setdefault(c, len(cty_codes)) for c in q_country], dtype=np.int8)

    pool_ids, idx_l, val_l, idxc_l, valc_l = [], [], [], [], []
    offset = 0
    for i in (2, 3):
        d = load("train", f"source{i}")
        ids = d.entity_id.astype(str).tolist()
        ctys = np.array([cty_codes.setdefault(c, len(cty_codes)) for c in d.country.astype(str).tolist()], dtype=np.int8)
        del d
        texts = get_texts("train", i, mode)
        pool_ids += ids
        for a in range(0, len(texts), SHARD[mode]):
            b = min(len(texts), a + SHARD[mode])
            X = vectorize(texts[a:b], idf)
            XT = X.T.tocsr()
            del X
            nrows = b - a
            oi = np.empty((N_QUERIES, KMAX), np.int64); ov = np.empty((N_QUERIES, KMAX), np.float32)
            oic = np.empty((N_QUERIES, KMAX), np.int64); ovc = np.empty((N_QUERIES, KMAX), np.float32)
            shard_topk(Q.indptr.astype(np.int64), Q.indices.astype(np.int64), Q.data, q_cty,
                       XT.indptr.astype(np.int64), XT.indices.astype(np.int32), XT.data, ctys[a:b], nrows, KMAX,
                       oi, ov, oic, ovc)
            del XT
            for arr in (oi, oic):
                arr[arr >= 0] += offset + a
            idx_l.append(oi); val_l.append(ov); idxc_l.append(oic); valc_l.append(ovc)
            # keep memory flat: merge as we go
            if len(idx_l) > 1:
                I, V = merge_topk(idx_l, val_l, KMAX); idx_l, val_l = [I], [V]
                I, V = merge_topk(idxc_l, valc_l, KMAX); idxc_l, valc_l = [I], [V]
            print(f"[{mode}] S{i} rows {a:,}-{b:,} done {time.time() - t0:.0f}s", flush=True)
        offset += len(texts)
        del texts
    I, V = idx_l[0], val_l[0]
    IC, VC = idxc_l[0], valc_l[0]
    pool_ids = np.array(pool_ids, dtype=object)
    res = {"q_ids": np.array(q_ids, dtype=object), "q_country": np.array(q_country, dtype=object),
           "top_ids": np.where(I >= 0, pool_ids[np.maximum(I, 0)], ""), "top_vals": V,
           "top_ids_c": np.where(IC >= 0, pool_ids[np.maximum(IC, 0)], ""), "top_vals_c": VC}
    np.savez_compressed(CACHE / f"s7_topk_{mode}.npz", **res)
    print(f"[{mode}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    modes = sys.argv[1:] or ["name", "nameaddr"]
    for m in modes:
        run(m)
