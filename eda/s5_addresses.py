"""Section 5 - Address analysis (per source, per country) + true-pair address agreement."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import (SEED, load, load_pair_records, md_table, norm_basic, script_of, strip_accents,
                    write_section)

SAMPLE = 200_000
out = ["# 5. Address analysis\n",
       f"Pattern rates use a random sample of up to {SAMPLE:,} rows per (split, source, country) group (seed {SEED}); "
       "true-pair statistics use all train pairs.\n"]

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
                "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR".split())
RX = {
    "5digit_any": re.compile(r"(?<!\d)\d{5}(?!\d)"),
    "5digit_not_first_token": re.compile(r"(?<=[\s,])\d{5}(?!\d)"),
    "zip_after_state (e.g. 'TX 75001')": re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),
    "zip+4": re.compile(r"(?<!\d)\d{5}-\d{4}(?!\d)"),
    "6digit_any (India PIN)": re.compile(r"(?<!\d)\d{6}(?!\d)"),
    "3+3 digit ('560 001')": re.compile(r"(?<!\d)\d{3}\s\d{3}(?!\d)"),
    "FR '5digit City' (e.g. '75008 Paris')": re.compile(r"(?<!\d)\d{5}\s+[A-Za-zÀ-ÿ]"),
    "PO box": re.compile(r"\bp\.?\s?o\.?\s?box\b", re.I),
    "any digit": re.compile(r"\d"),
}
LANDMARK = {
    "near": r"\bnear\b", "nr": r"\bnr\.?\b", "opp/opposite": r"\bopp\.?\b|\bopposite\b", "behind": r"\bbehind\b",
    "beside": r"\bbeside\b", "next to": r"\bnext to\b", "in front of": r"\bin front of\b", "c/o": r"\bc/o\b",
    "s/o|d/o|w/o": r"\b[sdw]/o\b", "landmark": r"\blandmark\b", "ke pass/ke paas": r"\bke pa+s\b",
    "chez/près de (fr)": r"\bchez\b|\bpr[eè]s de\b",
}
LANDMARK = {k: re.compile(v, re.I) for k, v in LANDMARK.items()}
ABBR = {  # family -> (short regex, long regex)
    "Road": (r"\brd\b", r"\broad\b"), "Street": (r"\bst\b", r"\bstreet\b"), "Avenue": (r"\bave?\b", r"\bavenue\b"),
    "Boulevard": (r"\bblvd\b", r"\bboulevard\b"), "Drive": (r"\bdr\b", r"\bdrive\b"), "Lane": (r"\bln\b", r"\blane\b"),
    "Court": (r"\bct\b", r"\bcourt\b"), "Circle": (r"\bcir\b", r"\bcircle\b"), "Highway": (r"\bhwy\b", r"\bhighway\b"),
    "Suite/Unit/Apt": (r"\b(ste|apt)\b", r"\b(suite|unit|apartment)\b"), "Floor": (r"\bflr?\b", r"\bfloor\b"),
    "Nagar": (r"\bngr\b", r"\bnagar\b"), "Marg": (r"\bmg\b", r"\bmarg\b"), "Colony": (r"\bcol\b", r"\bcolony\b"),
    "Sector": (r"\bsec\b", r"\bsector\b"), "Plot": (r"\bplt\b", r"\bplot\b"), "House no": (r"\bh\.?\s?no\b", r"\bhouse\b"),
    "Rue (fr)": (r"\br\b", r"\brue\b"), "Avenue (fr av.)": (r"\bav\b", r"\bavenue\b"), "Boulevard (fr bd)": (r"\bbd\b", r"\bboulevard\b"),
    "Place (fr pl)": (r"\bpl\b", r"\bplace\b"), "Chemin (fr ch)": (r"\bch\b", r"\bchemin\b"),
}
ABBR = {k: (re.compile(a, re.I), re.compile(b, re.I)) for k, (a, b) in ABBR.items()}


def lastcomp(a):
    parts = [p.strip() for p in a.split(",") if p.strip()]
    return parts[-1] if parts else ""


pat_rows, lm_rows, ab_rows, comp_rows, last_rows = [], [], [], [], {}
for split in ("train", "test"):
    for i in (1, 2, 3):
        d = load(split, f"source{i}")[["business_address", "country"]]
        d["country"] = d["country"].astype(str)
        for cty, g in d.groupby("country"):
            a = g["business_address"].astype(str)
            key = dict(split=split, source=f"S{i}", country=cty, n=len(a))
            smp = a.sample(min(SAMPLE, len(a)), random_state=SEED).tolist()
            ne = [s for s in smp if s.strip()]
            m = len(smp)
            pat_rows.append({**key, "empty": 1 - len(ne) / m, **{k: sum(bool(r.search(s)) for s in smp) / m for k, r in RX.items()}})
            lm_rows.append({**key, **{k: sum(bool(r.search(s)) for s in smp) / m for k, r in LANDMARK.items()}})
            row = dict(key)
            for fam, (rs, rl) in ABBR.items():
                row[f"{fam}: short"] = sum(bool(rs.search(s)) for s in smp) / m
                row[f"{fam}: long"] = sum(bool(rl.search(s)) for s in smp) / m
            ab_rows.append(row)
            ncomp = np.array([len([p for p in s.split(",") if p.strip()]) for s in smp])
            lc = [lastcomp(s) for s in ne]
            first_tok_num = sum(bool(re.match(r"^\s*(no\.?\s*|#+\s*|h\.?\s?no\.?\s*)?\d", s, re.I)) for s in ne) / max(1, len(ne))
            comp_rows.append({**key, "mean_components": ncomp.mean(), "p50_components": np.median(ncomp),
                              "1_component": (ncomp == 1).mean(), "starts_with_number": first_tok_num,
                              "last_comp_is_US_state_code": sum(x.upper() in US_STATES for x in lc) / max(1, len(ne)),
                              "has_US_state_code_anywhere": sum(any(p.strip() in US_STATES for p in s.split(",")) for s in ne) / max(1, len(ne)),
                              "non-latin_chars": sum(script_of(s) == "non-latin" for s in smp) / m,
                              "latin-accented": sum(script_of(s) == "latin-ext" for s in smp) / m,
                              "UPPERCASE": sum(s == s.upper() and any(c.isalpha() for c in s) for s in smp) / m,
                              "has_##": sum("##" in s for s in smp) / m})
            if (split == "train") or cty == "France":
                last_rows[(split, f"S{i}", cty)] = Counter(lc).most_common(15)
        print("done", split, i)

# ------------------------------------------------------------- true pairs
pr = load_pair_records()


def postal(a, c):
    if c == "India":
        m = re.findall(r"(?<!\d)(\d{6})(?!\d)", a)
    else:
        m = re.findall(r"(?<=[\s,])(\d{5})(?:-\d{4})?(?!\d)", a)  # 5-digit, not the leading house number
    return m[-1] if m else ""


def housenum(a):
    m = re.search(r"\d+[A-Za-z]?", a)
    return m.group(0).lstrip("0") if m else ""


def tri(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


p1 = [postal(a, c) for a, c in zip(pr.s1_addr, pr.s1_country)]
pm = [postal(a, c) for a, c in zip(pr.m_addr, pr.s1_country)]


def agree_cat(x, y):
    if not x and not y:
        return "both missing"
    if not x or not y:
        return "one missing"
    return "match" if x == y else "conflict"


pr["postal"] = [agree_cat(x, y) for x, y in zip(p1, pm)]
h1 = [housenum(a) for a in pr.s1_addr]
hm = [housenum(a) for a in pr.m_addr]
pr["housenum"] = [agree_cat(x, y) for x, y in zip(h1, hm)]
pr["addr_empty"] = np.select([(pr.s1_addr.str.strip() == "") & (pr.m_addr.str.strip() == ""),
                              pr.s1_addr.str.strip() == "", pr.m_addr.str.strip() == ""],
                             ["both empty", "S1 empty", "S2/S3 empty"], "both present")
na1 = [norm_basic(strip_accents(a)) for a in pr.s1_addr]
nam = [norm_basic(strip_accents(a)) for a in pr.m_addr]
pr["addr_exact_norm"] = [x == y and x != "" for x, y in zip(na1, nam)]
pr["addr_tokenset_equal"] = [set(x.split()) == set(y.split()) and x != "" for x, y in zip(na1, nam)]
pr["addr_jacc3"] = [len(tri(x) & tri(y)) / max(1, len(tri(x) | tri(y))) if x and y else np.nan for x, y in zip(na1, nam)]
pr["tok_jacc"] = [len(set(x.split()) & set(y.split())) / max(1, len(set(x.split()) | set(y.split()))) if x and y else np.nan
                  for x, y in zip(na1, nam)]


def tab(col):
    t = pd.crosstab([pr.s1_country, pr.src], pr[col], normalize="index")
    t["pairs"] = pr.groupby(["s1_country", "src"]).size()
    return t.reset_index()


sim = pr.groupby(["s1_country", "src"])[["addr_exact_norm", "addr_tokenset_equal", "addr_jacc3", "tok_jacc"]].mean().reset_index()
hard = pr[(pr.addr_empty == "both present") & (pr.addr_jacc3 < 0.2) & (pr.m_addr.map(script_of) != "non-latin")]
hard = hard.sample(min(15, len(hard)), random_state=SEED)[["s1_country", "src", "s1_name", "m_name", "s1_addr", "m_addr", "addr_jacc3"]]
low_share = pr.groupby("s1_country").addr_jacc3.apply(lambda x: pd.Series({"jacc3<0.2": (x < 0.2).mean(), "jacc3<0.4": (x < 0.4).mean(), "median": x.median()})).unstack().reset_index()

# ------------------------------------------------------------- write
out.append("## 5.1 Postal-code and numeric patterns (share of addresses)\n")
out.append("US `5digit_any` is dominated by house numbers (e.g. `17560 Ellis Road`); `5digit_not_first_token` and "
           "`zip_after_state` are the better ZIP proxies.\n")
out.append(md_table(pd.DataFrame(pat_rows)))
out.append("\n## 5.2 Address structure\n")
out.append(md_table(pd.DataFrame(comp_rows), floatfmt="{:.3f}"))
out.append("\n## 5.3 Most frequent last comma-component (state / city slot)\n")
lt = pd.DataFrame({f"{k[0]} {k[1]} {k[2]}": [f"{w} ({c:,})" for w, c in v] + [""] * (15 - len(v)) for k, v in last_rows.items()})
for j in range(0, lt.shape[1], 3):
    out.append(md_table(lt.iloc[:, j:j + 3], maxw=45))
    out.append("\n")
out.append("## 5.4 Landmark phrases (share of addresses)\n")
out.append(md_table(pd.DataFrame(lm_rows)))
out.append("\n## 5.5 Abbreviation vs long form (share of addresses containing each form)\n")
ab = pd.DataFrame(ab_rows)
keepc = ["split", "source", "country", "n"] + [c for c in ab.columns[4:] if ab[c].max() >= 0.005]
abt = ab[keepc]
# transpose for readability
abt = abt.set_index(["split", "source", "country"]).drop(columns="n").T
abt.columns = [" ".join(c) for c in abt.columns]
out.append(md_table(abt, index=True))
out.append("\n## 5.6 True pairs: postal code agreement (country-specific extraction; India 6-digit, US/FR 5-digit not leading)\n")
out.append(md_table(tab("postal")))
out.append("\nHouse/first number agreement (first number in the address, leading zeros stripped):\n")
out.append(md_table(tab("housenum")))
out.append("\nAddress emptiness on true pairs:\n")
out.append(md_table(tab("addr_empty")))
out.append("\nAddress similarity on true pairs (norm_basic + accent fold):\n")
out.append(md_table(sim))
out.append("\n")
out.append(md_table(low_share))
out.append("\n## 5.7 True pairs with very different addresses (both present, non-Latin excluded, jacc3 < 0.2, random 15)\n")
out.append(md_table(hard, floatfmt="{:.3f}", maxw=80))
write_section("5", "\n".join(out) + "\n")
pd.DataFrame(pat_rows).to_csv("out/s5_patterns.csv", index=False)
print(md_table(pd.DataFrame(pat_rows))); print(md_table(pd.DataFrame(comp_rows), floatfmt="{:.3f}"))
print(md_table(tab("postal"))); print(md_table(tab("housenum"))); print(md_table(tab("addr_empty"))); print(md_table(sim))
print(md_table(pd.DataFrame(lm_rows)))
