"""Section 8 - Train vs test shift, and France in detail."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import SEED, load, md_table, norm_basic, norm_name, script_of, strip_accents, write_section

SAMPLE = 200_000
out = ["# 8. Train vs test shift\n",
       f"Token statistics use a random sample of up to {SAMPLE:,} rows per (split, source, country) (seed {SEED}).\n"]


def pctl(x):
    q = np.percentile(x, [5, 50, 95])
    return f"{x.mean():.1f} / {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}"


len_rows, vocab, tokcnt, frames = [], {}, {}, {}
for split in ("train", "test"):
    for i in (1, 2, 3):
        d = load(split, f"source{i}")
        d["country"] = d.country.astype(str)
        for cty, g in d.groupby("country"):
            smp = g.sample(min(SAMPLE, len(g)), random_state=SEED)
            names, addrs = smp.business_name.tolist(), smp.business_address.tolist()
            key = (split, f"S{i}", cty)
            nt = [norm_basic(s).split() for s in names]
            at = [norm_basic(s).split() for s in addrs]
            vocab[key + ("name",)] = Counter(t for x in nt for t in x)
            vocab[key + ("addr",)] = Counter(t for x in at for t in x)
            len_rows.append(dict(split=split, source=f"S{i}", country=cty, n=len(g),
                                 **{"name chars mean/p5/p50/p95": pctl(g.business_name.str.len().to_numpy()),
                                    "addr chars mean/p5/p50/p95": pctl(g.business_address.str.len().to_numpy()),
                                    "addr empty": float((g.business_address == "").mean()),
                                    "name non-latin": np.mean([script_of(s) == "non-latin" for s in names]),
                                    "name latin-accented": np.mean([script_of(s) == "latin-ext" for s in names]),
                                    "addr non-latin": np.mean([script_of(s) == "non-latin" for s in addrs])}))
            if cty == "France":
                frames[f"S{i}"] = g.astype(str).copy()
        del d
        print("done", split, i, flush=True)

lens = pd.DataFrame(len_rows).sort_values(["country", "source", "split"])

# token overlap: share of test token occurrences whose token appears in train vocab (same source+country)
ov = []
for (split, src, cty, fld), c in vocab.items():
    if split != "test" or cty == "France":
        continue
    tr = vocab.get(("train", src, cty, fld))
    if tr is None:
        continue
    tot = sum(c.values())
    seen = sum(v for t, v in c.items() if t in tr)
    ov.append(dict(source=src, country=cty, field=fld, test_token_types=len(c),
                   types_seen_in_train=sum(1 for t in c if t in tr) / len(c),
                   occurrences_seen_in_train=seen / tot))
# France tokens vs ANY train vocab (same source)
for src in ("S1", "S2", "S3"):
    for fld in ("name", "addr"):
        c = vocab[("test", src, "France", fld)]
        tr = vocab[("train", src, "US", fld)] + vocab[("train", src, "India", fld)]
        ov.append(dict(source=src, country="France (vs train US+India)", field=fld, test_token_types=len(c),
                       types_seen_in_train=sum(1 for t in c if t in tr) / len(c),
                       occurrences_seen_in_train=sum(v for t, v in c.items() if t in tr) / sum(c.values())))
ov = pd.DataFrame(ov)

# ------------------------------------------------------------- France detail
FR_LEGAL = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "earl", "gie", "association",
            "inc", "llc", "ltd", "corp", "co", "pvt", "limited", "company"]
FR_STREET = ["rue", "r", "avenue", "av", "ave", "boulevard", "bd", "bld", "chemin", "ch", "place", "pl", "allee", "impasse",
             "imp", "route", "rte", "quai", "cours", "square", "sq", "faubourg", "fbg", "lieu", "dit", "zi", "za", "zac"]
fr_rows, fr_legal, fr_street, fr_last, fr_city = [], [], [], {}, {}
for src, g in frames.items():
    names, addrs = g.business_name.tolist(), g.business_address.tolist()
    nt = [norm_basic(strip_accents(s)).split() for s in names]
    at = [norm_basic(strip_accents(s)).split() for s in addrs]
    comps = [[p.strip() for p in a.split(",") if p.strip()] for a in addrs]
    fr_rows.append(dict(source=src, records=len(g),
                        name_has_accent=np.mean([strip_accents(s) != s for s in names]),
                        addr_has_accent=np.mean([strip_accents(s) != s for s in addrs]),
                        addr_empty=np.mean([a.strip() == "" for a in addrs]),
                        addr_has_5digit=np.mean([bool(re.search(r"(?<!\d)\d{5}(?!\d)", a)) for a in addrs]),
                        addr_starts_with_number=np.mean([bool(re.match(r"^\s*\d", a)) for a in addrs]),
                        addr_mean_components=np.mean([len(c) for c in comps]),
                        name_UPPER=np.mean([s == s.upper() and any(ch.isalpha() for ch in s) for s in names]),
                        addr_UPPER=np.mean([s == s.upper() and any(ch.isalpha() for ch in s) for s in addrs]),
                        name_has_quote=np.mean(['"' in s for s in names]),
                        name_has_apostrophe=np.mean(["'" in s or "’" in s for s in names]),
                        name_has_hyphen=np.mean(["-" in s for s in names])))
    fr_legal.append(dict(source=src, **{t: np.mean([t in x for x in nt]) for t in FR_LEGAL}))
    fr_street.append(dict(source=src, **{t: np.mean([t in x for x in at]) for t in FR_STREET}))
    fr_last[src] = Counter(c[-1] for c in comps if c).most_common(15)
    fr_city[src] = Counter(c[-2] if len(c) >= 2 else "" for c in comps).most_common(15)
fr_rows = pd.DataFrame(fr_rows); fr_legal = pd.DataFrame(fr_legal); fr_street = pd.DataFrame(fr_street)
fr_legal = fr_legal[["source"] + [c for c in fr_legal.columns[1:] if fr_legal[c].max() >= 0.001]]
fr_street = fr_street[["source"] + [c for c in fr_street.columns[1:] if fr_street[c].max() >= 0.001]]

ex_s1 = frames["S1"].sample(30, random_state=SEED)
ex_s2 = frames["S2"].sample(10, random_state=SEED)
ex_s3 = frames["S3"].sample(10, random_state=SEED)

# top name tokens France
toptok = pd.DataFrame({src: [f"{w} ({c:,})" for w, c in vocab[("test", src, "France", "name")].most_common(25)]
                       for src in ("S1", "S2", "S3")})
topaddr = pd.DataFrame({src: [f"{w} ({c:,})" for w, c in vocab[("test", src, "France", "addr")].most_common(25)]
                        for src in ("S1", "S2", "S3")})

# ---- proxy for match rate: share of S1 with >=1 S2/S3 record of identical nname (same country), train vs test
proxy = []
for split in ("train", "test"):
    hs = {}
    for i in (1, 2, 3):
        d = load(split, f"source{i}")
        nn = [norm_name(strip_accents(s)) for s in d.business_name.tolist()]
        hs[i] = pd.DataFrame({"country": d.country.astype(str).values,
                              "h": pd.util.hash_array(np.asarray(nn, dtype=object)),
                              "is_empty": [x == "" for x in nn]})
        del d, nn
    other = pd.concat([hs[2], hs[3]])
    other = other[~other.is_empty]
    keys = set(zip(other.country, other.h))
    s1 = hs[1]
    s1["hit"] = [(c, h) in keys for c, h in zip(s1.country, s1.h)]
    for cty, g in s1.groupby("country"):
        proxy.append(dict(split=split, country=cty, s1_entities=len(g), share_with_exact_nname_hit=g.hit.mean()))
    del hs, other, keys
proxy = pd.DataFrame(proxy)
try:
    st = pd.read_parquet("cache/s1_match_stats.parquet")
    sr = st.groupby("country").n.apply(lambda x: (x == 0).mean())
    proxy["train_singleton_rate"] = proxy.country.map(sr).where(proxy.split == "train")
except FileNotFoundError:
    pass

# ------------------------------------------------------------- write
out.append("## 8.1 Length and script: train vs test (US, India) and France\n")
out.append(md_table(lens, floatfmt="{:.3f}"))
out.append("\n## 8.2 Token overlap: test tokens seen in train (same source & country)\n")
out.append(md_table(ov))
out.append("\nPostal-code / pattern rates for train vs test are in section 5.1 (both splits are listed there).\n")
out.append("\n## 8.3 Proxy for match rate: S1 entities with >=1 identical-nname S2/S3 record (same country)\n")
out.append(md_table(proxy))
out.append("\n## 8.4 France: record-level characteristics\n")
out.append(md_table(fr_rows, floatfmt="{:.3f}"))
out.append("\nLegal-form tokens in France names (share of names containing token):\n")
out.append(md_table(fr_legal, floatfmt="{:.4f}"))
out.append("\nStreet-type tokens in France addresses (accent-folded):\n")
out.append(md_table(fr_street, floatfmt="{:.4f}"))
out.append("\nMost common last address component (France):\n")
out.append(md_table(pd.DataFrame({k: [f"{w} ({c:,})" for w, c in v] + [""] * (15 - len(v)) for k, v in fr_last.items()}), maxw=45))
out.append("\nMost common second-to-last component (city slot, France):\n")
out.append(md_table(pd.DataFrame({k: [f"{w} ({c:,})" for w, c in v] + [""] * (15 - len(v)) for k, v in fr_city.items()}), maxw=45))
out.append("\nTop France name tokens:\n")
out.append(md_table(toptok, maxw=40))
out.append("\nTop France address tokens:\n")
out.append(md_table(topaddr, maxw=40))
out.append("\n## 8.5 France: 30 random S1 rows\n")
out.append(md_table(ex_s1, maxw=80))
out.append("\n10 random S2 France rows:\n")
out.append(md_table(ex_s2, maxw=80))
out.append("\n10 random S3 France rows:\n")
out.append(md_table(ex_s3, maxw=80))
write_section("8", "\n".join(out) + "\n")
print(md_table(lens)); print(md_table(ov)); print(md_table(proxy)); print(md_table(fr_rows)); print(md_table(fr_legal))
print(md_table(ex_s1.head(15), maxw=60))
