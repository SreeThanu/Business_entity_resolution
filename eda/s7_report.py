"""Section 7 report - evaluates the top-k lists saved by s7_blocking.py."""
import numpy as np
import pandas as pd

from common import CACHE, SEED, load, load_pairs, md_table, write_section

pairs = load_pairs()
gt = pairs.groupby("s1").match.apply(set).to_dict()
out = ["# 7. Quick blocking-recall check (char n-gram TF-IDF)\n",
       "Setup: `char_wb` 3–5-grams on lowercased text, IDF fit on all train S1+S2+S3 (12.5M docs), L2-normalised, "
       "cosine brute force against **all 10.3M train S2+S3 records**. Vocabulary is hashed (2^24 buckets) instead of "
       "an exact `TfidfVectorizer` vocabulary because the exact fit does not fit in 8 GB RAM; IDF formula and "
       "normalisation match sklearn defaults. Queries: 12,000 random train S1 entities (seed 42). "
       "`pool=same-country` repeats the ranking with the pool restricted to records whose country label equals the query's.\n"]

KS = [5, 10, 20, 50, 100]
tables, missed_all, hi_nonmatch_all = {}, {}, {}
s1 = load("train", "source1").astype(str).set_index("entity_id")
need = set()
res_all = {}
for mode, label in (("name", "name only"), ("nameaddr", "name + address")):
    try:
        r = np.load(CACHE / f"s7_topk_{mode}.npz", allow_pickle=True)
    except FileNotFoundError:
        continue
    res_all[mode] = r
    q, qc = r["q_ids"], r["q_country"]
    rows = []
    for variant, ids, vals in (("all S2+S3", r["top_ids"], r["top_vals"]), ("same-country", r["top_ids_c"], r["top_vals_c"])):
        rec = []
        for j, (qid, cty) in enumerate(zip(q, qc)):
            t = gt.get(qid, set())
            if not t:
                continue
            lst = list(ids[j])
            pos = {e: p for p, e in enumerate(lst) if e}
            ranks = [pos.get(e, 10 ** 9) for e in t]
            rec.append((qid, cty, len(t), ranks))
        for cty in ["ALL"] + sorted(set(qc)):
            sub = [x for x in rec if cty == "ALL" or x[1] == cty]
            if not sub:
                continue
            row = {"text": label, "pool": variant, "country": cty, "queries_with_matches": len(sub),
                   "true_pairs": sum(x[2] for x in sub)}
            for k in KS:
                hit = sum(sum(rk < k for rk in x[3]) for x in sub)
                row[f"pair_recall@{k}"] = hit / row["true_pairs"]
            for k in (20, 50, 100):
                row[f"entity_all_found@{k}"] = np.mean([all(rk < k for rk in x[3]) for x in sub])
            rows.append(row)
        if variant == "all S2+S3":
            # misses at k=50
            miss = [(x[0], e) for x in rec for e, rk in zip(gt[x[0]], x[3]) if rk >= 50]
            missed_all[mode] = miss
            need |= {e for _, e in miss}
            # singletons / non-matching top-1
            hi = []
            for j, (qid, cty) in enumerate(zip(q, qc)):
                t = gt.get(qid, set())
                top = ids[j][0]
                hi.append(dict(q=qid, country=cty, singleton=not t, top1=top, top1_score=float(vals[j][0]),
                               top1_is_match=top in t,
                               best_true_score=max([float(v) for e, v in zip(ids[j], vals[j]) if e in t], default=np.nan)))
            hi_nonmatch_all[mode] = pd.DataFrame(hi)
            need |= {h for h in hi_nonmatch_all[mode].top1 if h}
    tables[mode] = pd.DataFrame(rows)

# fetch records needed for examples
recs = []
for i in (2, 3):
    d = load("train", f"source{i}")
    d["entity_id"] = d.entity_id.astype(str)
    recs.append(d[d.entity_id.isin(need)].astype(str))
recs = pd.concat(recs).set_index("entity_id")

for mode, t in tables.items():
    out.append(f"\n## 7.{1 if mode == 'name' else 2} Recall — {'name only' if mode == 'name' else 'name + address'}\n")
    out.append(md_table(t))

# score separation: top-1 score for singletons vs matched entities
out.append("\n## 7.3 Top-1 cosine: singletons vs entities with matches (pool = all S2+S3)\n")
for mode, h in hi_nonmatch_all.items():
    desc = h.groupby(["country", "singleton"]).agg(n=("q", "size"), top1_score_median=("top1_score", "median"),
                                                  top1_score_p90=("top1_score", lambda x: x.quantile(.9)),
                                                  share_top1_is_true_match=("top1_is_match", "mean"),
                                                  share_top1_score_ge_0_9=("top1_score", lambda x: (x >= 0.9).mean()),
                                                  best_true_score_median=("best_true_score", "median")).reset_index()
    out.append(f"\n{mode}:\n")
    out.append(md_table(desc))
    # high-score non-matches (precision risk)
    bad = h[(~h.top1_is_match) & (h.top1_score >= 0.85)].sample(frac=1, random_state=SEED).head(10)
    if len(bad):
        ex = pd.DataFrame({
            "country": bad.country.values, "S1 singleton?": bad.singleton.values,
            "S1 name": s1.loc[bad.q, "business_name"].values, "S1 address": s1.loc[bad.q, "business_address"].values,
            "top-1 (non-match) name": recs.reindex(bad.top1)["business_name"].values,
            "top-1 address": recs.reindex(bad.top1)["business_address"].values, "cosine": bad.top1_score.values})
        out.append(f"\nNon-matching top-1 with cosine >= 0.85 ({mode}; random 10) — likely false merges:\n")
        out.append(md_table(ex, floatfmt="{:.3f}", maxw=50))

for mode, miss in missed_all.items():
    rng = np.random.default_rng(SEED)
    sel = [miss[i] for i in rng.choice(len(miss), min(20, len(miss)), replace=False)] if miss else []
    ex = pd.DataFrame([dict(country=s1.at[a, "country"], s1_name=s1.at[a, "business_name"], match_name=recs.at[b, "business_name"],
                            s1_addr=s1.at[a, "business_address"], match_addr=recs.at[b, "business_address"], match_id=b)
                       for a, b in sel])
    out.append(f"\n## 7.{4 if mode == 'name' else 5} 20 true matches still missed at k=50 ({mode}, pool = all S2+S3; "
               f"{len(miss):,} misses in total)\n")
    out.append(md_table(ex, maxw=55))

write_section("7", "\n".join(out) + "\n")
for t in tables.values():
    print(md_table(t))
