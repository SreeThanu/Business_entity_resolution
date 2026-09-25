"""Section 6.5 - What separates true matches from look-alike non-matches?

Uses the name+address TF-IDF top-10 lists from s7_blocking.py (12,000 random train S1 queries).
Among candidates with cosine >= 0.8, compare true matches vs non-matches on:
house number equality / |difference|, legal-suffix family agreement, extra-token counts.
Writes out/section6b.md (appended after section 6 in the report).
"""
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import CACHE, OUT, SEED, md_table, norm_basic, norm_name, strip_accents

r = np.load(CACHE / "s7_topk_nameaddr.npz", allow_pickle=True)
q, qc, ids, vals = r["q_ids"], r["q_country"], r["top_ids"][:, :10], r["top_vals"][:, :10]
rows = [(q[j], qc[j], ids[j, i], float(vals[j, i])) for j in range(len(q)) for i in range(10)
        if ids[j, i] and vals[j, i] >= 0.8]
c = pd.DataFrame(rows, columns=["s1", "country", "cand", "cos"])

t = pq.read_table(CACHE / "train_gt_pairs.parquet")
t = t.filter(pc.is_in(t["s1"], pa.array(list(set(c.s1))))).to_pandas()
tp = set(zip(t.s1, t.match))
c["is_match"] = [(a, b) in tp for a, b in zip(c.s1, c.cand)]


def fetch(path, need):
    tb = pq.read_table(path)
    tb = tb.filter(pc.is_in(tb["entity_id"], pa.array(list(need))))
    return tb.to_pandas().set_index("entity_id")


s1 = fetch(CACHE / "train_source1.parquet", set(c.s1))
oth = pd.concat([fetch(CACHE / f"train_source{i}.parquet", set(c.cand)) for i in (2, 3)])

FAM = {"pvt": "pvt", "private": "pvt", "ltd": "ltd", "limited": "ltd", "llp": "llp", "inc": "inc",
       "incorporated": "inc", "corp": "corp", "corporation": "corp", "llc": "llc", "co": "co", "company": "co",
       "pllc": "pllc", "pc": "pc", "lp": "lp", "public": "public"}


def fams(name):
    return frozenset(FAM[w] for w in norm_basic(strip_accents(name)).replace("l l c", "llc").split() if w in FAM)


def hnum(a):
    m = re.search(r"\d+", a)
    return int(m.group(0)) if m else None


def core(name):
    return set(norm_name(strip_accents(name)).split())


out = []
for _, x in c.iterrows():
    a, b = s1.loc[x.s1], oth.loc[x.cand]
    h1, h2 = hnum(a.business_address), hnum(b.business_address)
    f1, f2 = fams(a.business_name), fams(b.business_name)
    t1, t2 = core(a.business_name), core(b.business_name)
    out.append(dict(
        hnum_status="missing" if h1 is None or h2 is None else ("equal" if h1 == h2 else "differs"),
        hnum_absdiff=np.nan if h1 is None or h2 is None or h1 == h2 else abs(h1 - h2),
        suffix_status="none on one side" if not f1 or not f2 else ("same family" if f1 == f2 else ("overlap" if f1 & f2 else "different family")),
        extra_core_tokens=len(t2 - t1), missing_core_tokens=len(t1 - t2)))
c = pd.concat([c, pd.DataFrame(out)], axis=1)
c["label"] = np.where(c.is_match, "true match", "non-match")

res = ["\n## 6.5 What separates true matches from look-alikes? (name+address TF-IDF top-10, cosine >= 0.8)\n",
       f"{len(c):,} candidates from {c.s1.nunique():,} random train S1 queries; {int(c.is_match.sum()):,} true matches, "
       f"{int((~c.is_match).sum()):,} non-matches.\n"]
for col in ("hnum_status", "suffix_status"):
    tb = pd.crosstab([c.country, c.label], c[col], normalize="index").reset_index()
    res.append(f"\n`{col}` (row-normalised):\n")
    res.append(md_table(tb))
d = c[c.hnum_status == "differs"].groupby(["country", "label"]).hnum_absdiff.describe(percentiles=[.25, .5, .75]).reset_index()
res.append("\n|house-number difference| when both present and different:\n")
res.append(md_table(d, floatfmt="{:.1f}"))
e = c.groupby(["country", "label"])[["extra_core_tokens", "missing_core_tokens"]].agg(lambda s: (s > 0).mean()).reset_index()
e.columns = ["country", "label", "share with extra core name token", "share with missing core name token"]
res.append("\nCore-name token differences (norm_name tokens; >0 means the candidate adds / drops a word):\n")
res.append(md_table(e))
cos = c.groupby(["country", "label"]).cos.describe(percentiles=[.1, .5, .9]).reset_index()
res.append("\nCosine distribution within this band:\n")
res.append(md_table(cos, floatfmt="{:.3f}"))
with open(OUT / "section6b.md", "w", encoding="utf-8") as f:
    f.write("\n".join(res) + "\n")
print("\n".join(res))
