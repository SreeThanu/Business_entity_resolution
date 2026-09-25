"""Section 6 - Chains and ambiguity (precision risk).

Normalized name = common.norm_name after accent folding. Normalized names are hashed to uint64
(pd.util.hash_array) to keep 12.5M keys in memory.
"""
import re

import numpy as np
import pandas as pd

from common import SEED, load, load_pairs, md_table, norm_basic, norm_name, strip_accents, write_section

out = ["# 6. Chains and ambiguity (precision risk)\n",
       "Keys: `nname` = norm_name(accent-folded name) (lowercase, punctuation stripped, '&'→'and', legal-suffix "
       "tokens removed); `hnum` = first number in the address (leading zeros stripped). All joins are within country.\n"]


def hnum(a):
    m = re.search(r"\d+", a)
    return m.group(0).lstrip("0") if m else ""


def idh(x):
    return pd.util.hash_array(np.asarray(x, dtype=object))


def prep(split, i):
    """Compact frame: pyarrow text columns + uint64 hashes of id / nname / nname|hnum."""
    d = load(split, f"source{i}")
    names, addrs = d.business_name.tolist(), d.business_address.tolist()
    nn = [norm_name(strip_accents(s)) for s in names]
    d["country"] = d.country.astype(str).astype("category")
    d["idh"] = idh(d.entity_id.tolist())
    d["h"] = idh(nn)
    d["hn"] = idh([n + "|" + hnum(a) for n, a in zip(nn, addrs)])
    d["empty_nname"] = [n == "" for n in nn]
    if i == 1:
        d["nname"] = nn
        d["addr_key"] = [norm_basic(a) for a in addrs]
    return d


s1 = prep("train", 1)
print("s1 prepared")
st = pd.read_parquet("cache/s1_match_stats.parquet")  # from s3
s1["n"] = s1.idh.map(pd.Series(st.n.values, index=idh(st.s1.tolist()))).fillna(0).astype(int).values
s1["country"] = s1.country.astype(str)

# ---------------------------------------------------- 6.1 chains within S1
g = s1[s1.nname != ""].groupby(["country", "nname"]).agg(s1_entities=("entity_id", "size"),
                                                          distinct_addresses=("addr_key", "nunique"),
                                                          singletons=("n", lambda x: int((x == 0).sum())),
                                                          example_address=("business_address", "first"))
g = g.reset_index()
chains = g[(g.s1_entities > 1) & (g.distinct_addresses > 1)]
share = s1.assign(k=s1.nname).merge(g[["country", "nname", "s1_entities"]], on=["country", "nname"], how="left")
amb = share.groupby("country").apply(lambda x: pd.Series({
    "S1 entities": len(x),
    "share whose nname is shared with >=1 other S1": (x.s1_entities > 1).mean(),
    "share whose nname is shared with >=5 other S1": (x.s1_entities > 5).mean(),
    "empty nname": (x.nname == "").mean()})).reset_index()
top_chains = {c: chains[chains.country == c].sort_values("s1_entities", ascending=False).head(25)
              for c in chains.country.unique()}

# ---------------------------------------------------- 6.2 exact-nname collisions S1 vs S2/S3
pairs = load_pairs()
pairs["h1"] = idh(pairs.s1.tolist())
pairs["h2"] = idh(pairs.match.tolist())
pairs["ph"] = pairs.h1 ^ (pairs.h2 * np.uint64(1000003))
rows, ex_rows = [], []
s23_parts = []
for i in (2, 3):
    d = prep("train", i)
    s23_parts.append(d[["entity_id", "business_name", "business_address", "country", "idh", "h", "hn", "empty_nname"]])
    print("s", i, "prepared")
s23 = pd.concat(s23_parts, ignore_index=True)
del s23_parts
s23["country"] = s23.country.astype(str).astype("category")
s23["is_matched_to_someone"] = s23.idh.isin(pairs.h2.unique())

for key in ("h", "hn"):
    c1 = s1[~s1.empty_nname].groupby(["country", key]).size().rename("n1")
    c23 = s23[~s23.empty_nname].groupby(["country", key], observed=True).size().rename("n23")
    j = pd.concat([c1, c23], axis=1, join="inner")
    total_pairs = float((j.n1 * j.n23).sum())
    # true pairs sharing the key
    tp = pairs[["h1", "h2"]].merge(s1[["idh", "country", key]], left_on="h1", right_on="idh") \
              .merge(s23[["idh", key]], left_on="h2", right_on="idh", suffixes=("_1", "_2"))
    same = tp[tp[f"{key}_1"] == tp[f"{key}_2"]]
    same = same[same[f"{key}_1"].isin(j.index.get_level_values(1))]
    for cty in sorted(s1.country.unique()):
        jj = j.loc[cty] if cty in j.index.get_level_values(0) else j.iloc[:0]
        tot = float((jj.n1 * jj.n23).sum())
        tpc = int((same.country == cty).sum())
        s1c = s1[s1.country == cty]
        has = s1c[key].isin(jj.index)
        rows.append({"key": "nname" if key == "h" else "nname+hnum", "country": cty,
                     "S1 entities with >=1 S2/S3 sharing key": float(has.mean()),
                     "... of which singletons (share of all singletons)": float(has[s1c.n == 0].mean()),
                     "candidate pairs sharing key": int(tot), "true pairs among them": tpc,
                     "precision of 'match iff key equal'": tpc / tot if tot else np.nan,
                     "recall of 'match iff key equal'": tpc / max(1, int((tp.country == cty).sum()))})
rule = pd.DataFrame(rows)
print(md_table(rule))

# examples: S2/S3 record with the same nname as an S1 entity but NOT in its match list
a = s1.loc[~s1.empty_nname, ["idh", "country", "h", "n"]].reset_index().rename(columns={"index": "i1"})
b = s23.loc[~s23.empty_nname, ["idh", "country", "h", "is_matched_to_someone"]].reset_index().rename(columns={"index": "i2"})
b["country"] = b.country.astype(str)
sz1 = a.groupby("h").size(); sz2 = b.groupby("h").size()
small = (sz1 * sz2).dropna()
small = small[small < 50].index  # skip mega-chains (keeps the join small and examples readable)
m = a[a.h.isin(small)].merge(b[b.h.isin(small)], on=["country", "h"], suffixes=("_s1", "_other"))
m["ph"] = m.idh_s1 ^ (m.idh_other * np.uint64(1000003))
fm = m[~m.ph.isin(pairs.ph.values)].copy()
for side, src, ii in (("s1", s1, "i1"), ("other", s23, "i2")):
    for c in ("entity_id", "business_name", "business_address"):
        fm[f"{c}_{side}"] = src[c].iloc[fm[ii].values].astype(str).values
fm_single = fm[fm.n == 0]
cols = ["country", "entity_id_s1", "business_name_s1", "business_address_s1", "entity_id_other",
        "business_name_other", "business_address_other", "is_matched_to_someone"]
ex_single = fm_single.sample(min(10, len(fm_single)), random_state=SEED)[cols]
ex_other = fm[fm.n > 0].sample(min(10, int((fm.n > 0).sum())), random_state=SEED)[cols]
fm_stats = pd.DataFrame({"stat": [
    "S1-vs-S2/S3 same-nname non-matching pairs (chains <50 excluded)",
    "... where S1 is a singleton", "... where the other record is matched to a different S1",
    "... where the other record is a distractor (matches nobody)"],
    "value": [len(fm), len(fm_single), int(fm.is_matched_to_someone.sum()), int((~fm.is_matched_to_someone).sum())]})

# ---------------------------------------------------- write
out.append("## 6.1 How ambiguous are S1 names?\n")
out.append(md_table(amb))
for c, t in top_chains.items():
    out.append(f"\nTop chain names in S1 ({c}): same nname on multiple S1 entities with different addresses\n")
    out.append(md_table(t[["nname", "s1_entities", "distinct_addresses", "singletons", "example_address"]], maxw=60))
out.append(f"\nTotal chain keys (nname on >1 S1 entity with >1 distinct address): "
           + ", ".join(f"{c}: {int((chains.country == c).sum()):,} keys covering {int(chains[chains.country == c].s1_entities.sum()):,} S1 entities" for c in chains.country.unique()) + "\n")
out.append("\n## 6.2 Exact-key collisions between S1 and S2/S3 (what would a naive exact-name rule do?)\n")
out.append(md_table(rule))
out.append("\n")
out.append(md_table(fm_stats))
out.append("\n## 6.3 Likely false merges: non-matching S2/S3 record with identical nname, S1 is a SINGLETON (random 10)\n")
out.append(md_table(ex_single, maxw=55))
out.append("\n## 6.4 Likely false merges: identical nname, S1 has other true matches, this record is not one of them (random 10)\n")
out.append(md_table(ex_other, maxw=55))
write_section("6", "\n".join(out) + "\n")
rule.to_csv("out/s6_rule.csv", index=False)
print(md_table(amb)); print(md_table(fm_stats))
for c, t in top_chains.items():
    print(c); print(md_table(t.head(10)))
print(md_table(ex_single, maxw=40))
