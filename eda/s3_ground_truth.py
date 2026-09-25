"""Section 3 - Ground truth structure."""
import numpy as np
import pandas as pd

from common import load, load_pairs, md_table, pct, write_section

out = ["# 3. Ground truth structure\n"]
s1 = load("train", "source1")[["entity_id", "business_name", "business_address", "country"]]
s1 = s1.astype(str)
s23 = pd.concat([load("train", "source2")[["entity_id", "country"]],
                 load("train", "source3")[["entity_id", "country"]]], ignore_index=True)
s23["entity_id"] = s23.entity_id.astype(str)
s23["country"] = s23.country.astype(str).astype("category")
pairs = load_pairs()
pairs["src"] = pairs["match"].str[:2]

cty1 = pd.Series(s1.country.values, index=s1.entity_id.values)
cty23 = pd.Series(s23.country.values, index=s23.entity_id.values)

# ---- matches per S1
n = pairs.groupby("s1").size()
n2 = pairs[pairs.src == "S2"].groupby("s1").size()
n3 = pairs[pairs.src == "S3"].groupby("s1").size()
st = pd.DataFrame({"s1": s1.entity_id.values, "country": s1.country.values})
st["n"] = st.s1.map(n).fillna(0).astype(int)
st["n2"] = st.s1.map(n2).fillna(0).astype(int)
st["n3"] = st.s1.map(n3).fillna(0).astype(int)
st["bucket"] = st.n.clip(upper=10).astype(str).replace("10", "10+")

dist = st.groupby(["bucket"]).size().rename("all").to_frame()
for c, g in st.groupby("country"):
    dist[c] = g.groupby("bucket").size()
dist = dist.fillna(0).astype(int)
order = [str(i) for i in range(10)] + ["10+"]
dist = dist.reindex([o for o in order if o in dist.index])
distp = dist / dist.sum()
dist_tbl = dist.reset_index().merge((distp.add_suffix("_share")).reset_index(), on="bucket")

summ = st.groupby("country").agg(s1_entities=("s1", "size"), singletons=("n", lambda x: int((x == 0).sum())),
                                 total_matches=("n", "sum"), mean_matches=("n", "mean"),
                                 mean_matches_nonsingleton=("n", lambda x: x[x > 0].mean()),
                                 max_matches=("n", "max"))
summ.loc["ALL"] = [len(st), int((st.n == 0).sum()), int(st.n.sum()), st.n.mean(), st.n[st.n > 0].mean(), st.n.max()]
summ["singleton_rate"] = summ.singletons / summ.s1_entities
for c in ("s1_entities", "singletons", "total_matches", "max_matches"):
    summ[c] = summ[c].astype(int)
singleton_rate = float((st.n == 0).mean())

# ---- source mix
def mix(r):
    if r.n == 0:
        return "none"
    if r.n2 > 0 and r.n3 > 0:
        return "S2+S3"
    return "S2 only" if r.n2 > 0 else "S3 only"
st["mix"] = np.select([st.n == 0, (st.n2 > 0) & (st.n3 > 0), st.n2 > 0], ["none", "S2+S3", "S2 only"], "S3 only")
mixt = pd.crosstab(st.mix, st.country, margins=True)
same_src = pd.DataFrame({
    "stat": ["S1 with >=2 matches from S2", "S1 with >=2 matches from S3", "max S2 matches for one S1",
             "max S3 matches for one S1"],
    "value": [int((st.n2 >= 2).sum()), int((st.n3 >= 2).sum()), int(st.n2.max()), int(st.n3.max())]})
n2d = pd.crosstab(st.n2.clip(upper=6), st.n3.clip(upper=6))
n2d.index.name = "n_S2 \\ n_S3"

# ---- one-to-one test
mc = pairs.groupby("match").s1.nunique()
multi = mc[mc > 1]
one2one = pd.DataFrame({"stat": ["distinct matched S2/S3 IDs", "IDs appearing in >1 S1 list", "max S1 lists per ID"],
                        "value": [len(mc), len(multi), int(mc.max()) if len(mc) else 0]})
ex_multi = pd.DataFrame()
if len(multi):
    mm = pairs[pairs.match.isin(multi.index[:5])].merge(s1, left_on="s1", right_on="entity_id")
    ex_multi = mm[["match", "s1", "business_name", "business_address", "country"]]

# ---- distractors
matched = set(pairs.match)
s23["is_matched"] = s23.entity_id.isin(matched)
s23["src"] = s23.entity_id.str[:2]
distr = s23.groupby(["src", "country"], observed=True).agg(records=("entity_id", "size"), matched=("is_matched", "sum"))
distr["unmatched"] = distr.records - distr.matched
distr["distractor_rate"] = distr.unmatched / distr.records
tot = s23.groupby("src").agg(records=("entity_id", "size"), matched=("is_matched", "sum"))
tot["unmatched"] = tot.records - tot.matched
tot["distractor_rate"] = tot.unmatched / tot.records
tot["country"] = "ALL"
distr = pd.concat([distr.reset_index(), tot.reset_index()])

# ---- country agreement on pairs
pairs["c1"] = pairs.s1.map(cty1)
pairs["c2"] = pairs.match.map(cty23).astype(str)
cc = pd.crosstab(pairs.c1, pairs.c2)
diff = pairs[pairs.c1 != pairs.c2]

# ---- F0.5 reference points
def f05(p, r):
    return 0 if p + r == 0 else 1.25 * p * r / (0.25 * p + r)

nz = st.n[st.n > 0]
ref = pd.DataFrame({
    "strategy (oracle-style reference)": [
        "predict empty for everyone (floor)",
        "perfect on singletons, 1 correct match per non-singleton",
        "perfect on singletons, all correct matches but +1 false match each non-singleton",
        "perfect on non-singletons, 1 false match on 5% of singletons",
    ],
    "macro_F0.5": [
        singleton_rate,
        singleton_rate + (1 - singleton_rate) * np.mean([f05(1, 1 / k) for k in nz]),
        singleton_rate + (1 - singleton_rate) * np.mean([f05(k / (k + 1), 1) for k in nz]),
        1 - 0.05 * singleton_rate,
    ]})

# ---- write
out.append("## 3.1 Matches per S1 entity\n")
out.append(md_table(dist_tbl, floatfmt="{:.4f}"))
out.append("\n")
out.append(md_table(summ, index=True, floatfmt="{:.4f}"))
out.append(f"\n**Singleton rate = {pct(singleton_rate)}.** Predicting an empty list for every train S1 entity "
           f"scores macro-F0.5 = **{singleton_rate:.4f}** - the baseline floor.\n")
out.append("## 3.2 Source mix of matches\n")
out.append(md_table(mixt, index=True))
out.append("\n")
out.append(md_table(same_src))
out.append("\nJoint distribution of #S2 matches (rows) × #S3 matches (cols), capped at 6:\n")
out.append(md_table(n2d, index=True))
out.append("\n## 3.3 One-to-one hypothesis (does an S2/S3 record belong to >1 S1 entity?)\n")
out.append(md_table(one2one))
if len(ex_multi):
    out.append("\nExamples:\n")
    out.append(md_table(ex_multi))
out.append("\n## 3.4 Distractors: S2/S3 records that match no S1 entity\n")
out.append(md_table(distr, floatfmt="{:.4f}"))
out.append("\n## 3.5 Country label agreement on matched pairs\n")
out.append(md_table(cc, index=True))
out.append(f"\nPairs with different country labels: {len(diff):,}\n")
if len(diff):
    out.append(md_table(diff.head(10)))
out.append("\n## 3.6 F0.5 reference points (computed on train GT)\n")
out.append(md_table(ref))
write_section("3", "\n".join(out) + "\n")
st.to_parquet("cache/s1_match_stats.parquet", index=False)
print(md_table(dist_tbl)); print(md_table(summ, index=True)); print(md_table(mixt, index=True))
print(md_table(same_src)); print(md_table(one2one)); print(md_table(distr)); print(md_table(cc, index=True))
print(md_table(ref)); print(ex_multi.head(10))
