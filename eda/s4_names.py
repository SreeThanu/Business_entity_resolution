"""Section 4 - Name analysis (per source, per country) + true-pair name agreement."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import (LEGAL_SUFFIXES, SEED, load, load_pair_records, md_table, norm_basic, norm_name,
                    pct, script_of, strip_accents, write_section)

rng = np.random.default_rng(SEED)
SAMPLE = 200_000  # per (split, source, country) group for token-level stats

out = ["# 4. Name analysis\n",
       "True-pair statistics (4.9-4.10) use a random sample of 1,000,000 of the 7,638,365 train pairs (seed 42).\n",
       f"Length/casing/script stats use all rows; token and suffix frequencies use a random sample of up to "
       f"{SAMPLE:,} names per (split, source, country) group (seed {SEED}).\n"]

SUFFIX_VARIANTS = {
    # token after norm_basic -> canonical family
    "pvt": "Pvt/Private", "private": "Pvt/Private", "pvt.": "Pvt/Private",
    "ltd": "Ltd/Limited", "limited": "Ltd/Limited",
    "llp": "LLP", "inc": "Inc/Incorporated", "incorporated": "Inc/Incorporated",
    "corp": "Corp/Corporation", "corporation": "Corp/Corporation", "llc": "LLC",
    "co": "Co/Company", "company": "Co/Company", "plc": "PLC", "lp": "LP", "pllc": "PLLC", "opc": "OPC",
    "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "sa": "SA", "eurl": "EURL", "sci": "SCI", "snc": "SNC",
}
NONLATIN_SUFFIX = {"प्राइवेट": "Pvt (Devanagari)", "लिमिटेड": "Ltd (Devanagari)", "प्रा": "Pvt abbr (Devanagari)",
                   "लि": "Ltd abbr (Devanagari)"}


def casing(s: str) -> str:
    letters = [c for c in s if c.isalpha() and c.isascii()]
    if not letters:
        return "no-ascii-letters"
    if all(c.isupper() for c in letters):
        return "UPPER"
    if all(c.islower() for c in letters):
        return "lower"
    words = [w for w in re.findall(r"[A-Za-z]+", s)]
    if words and all(w[0].isupper() and (len(w) == 1 or not w[1:].isupper()) for w in words):
        return "Title"
    return "Mixed"


def describe_len(x: pd.Series) -> dict:
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(mean=float(x.mean()), p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4], max=int(x.max()))


len_rows, tok_rows, script_rows, case_rows, suf_rows, amp_rows, odd_rows = [], [], [], [], [], [], []
top_tokens = {}
for split in ("train", "test"):
    for i in (1, 2, 3):
        d = load(split, f"source{i}")[["business_name", "country"]]
        d["country"] = d["country"].astype(str)
        for cty, g in d.groupby("country"):
            names = g["business_name"].astype(str)
            key = dict(split=split, source=f"S{i}", country=cty, n=len(names))
            L = names.str.len()
            T = names.str.split().str.len().fillna(0)
            len_rows.append({**key, "unit": "chars", **describe_len(L)})
            tok_rows.append({**key, "unit": "tokens", **describe_len(T)})
            smp = names.sample(min(SAMPLE, len(names)), random_state=SEED).tolist()
            sc = Counter(script_of(s) for s in smp)
            script_rows.append({**key, **{k: sc.get(k, 0) / len(smp) for k in ("ascii", "latin-ext", "non-latin")}})
            cc = Counter(casing(s) for s in smp)
            case_rows.append({**key, **{k: cc.get(k, 0) / len(smp) for k in
                                        ("Title", "UPPER", "lower", "Mixed", "no-ascii-letters")}})
            toks = [norm_basic(s).split() for s in smp]
            fam = Counter()
            for t in toks:
                fams = {SUFFIX_VARIANTS[w] for w in t if w in SUFFIX_VARIANTS}
                fams |= {NONLATIN_SUFFIX[w] for w in t if w in NONLATIN_SUFFIX}
                fam.update(fams)
            suf_rows.append({**key, **{k: v / len(smp) for k, v in fam.items()},
                             "any_legal_suffix": sum(1 for t in toks if any(w in LEGAL_SUFFIXES for w in t)) / len(smp)})
            low = [s.lower() for s in smp]
            amp_rows.append({**key,
                             "has_&": sum("&" in s for s in smp) / len(smp),
                             "has_' and '": sum(" and " in s for s in low) / len(smp),
                             "has_'et' (fr)": sum(" et " in s for s in low) / len(smp)})
            odd_rows.append({**key,
                             "starts_non_alnum": sum(bool(re.match(r"^[^\w]", s)) for s in smp) / len(smp),
                             "@handle": sum(s.startswith("@") for s in smp) / len(smp),
                             "domain_like(.com/.in/.fr...)": sum(bool(re.search(r"\.(com|in|net|org|co|fr|us|biz)\b", s.lower())) for s in smp) / len(smp),
                             "has_digit": sum(any(c.isdigit() for c in s) for s in smp) / len(smp),
                             "has_dba/aka": sum(bool(re.search(r"\b(dba|d/b/a|aka|t/a)\b", s.lower())) for s in smp) / len(smp),
                             "has_parenthesis": sum("(" in s for s in smp) / len(smp),
                             "empty": sum(s.strip() == "" for s in smp) / len(smp)})
            if split == "train" or cty == "France":
                c = Counter(w for t in toks for w in t)
                top_tokens[(split, f"S{i}", cty)] = c.most_common(25)
        print("done", split, i)

# ------------------------------------------------------------- true pairs
pr = load_pair_records()
pr["script_m"] = [script_of(s) for s in pr.m_name]
pr["script_1"] = [script_of(s) for s in pr.s1_name]
a, b = pr.s1_name, pr.m_name
pr["exact"] = a == b
pr["casefold"] = a.str.lower() == b.str.lower()
n1b, nmb = [norm_basic(s) for s in a], [norm_basic(s) for s in b]
pr["norm_basic"] = [x == y for x, y in zip(n1b, nmb)]
n1n, nmn = [norm_name(strip_accents(s)) for s in a], [norm_name(strip_accents(s)) for s in b]
pr["norm_name"] = [x == y for x, y in zip(n1n, nmn)]
pr["norm_name_tokensort"] = [sorted(x.split()) == sorted(y.split()) for x, y in zip(n1n, nmn)]
pr["tokset_subset"] = [bool(set(x.split())) and bool(set(y.split())) and
                       (set(x.split()) <= set(y.split()) or set(y.split()) <= set(x.split())) for x, y in zip(n1n, nmn)]


def tri(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


pr["jacc3"] = [len(tri(x) & tri(y)) / max(1, len(tri(x) | tri(y))) for x, y in zip(n1n, nmn)]
pr["n1n"], pr["nmn"] = n1n, nmn
levels = ["exact", "casefold", "norm_basic", "norm_name", "norm_name_tokensort", "tokset_subset"]
agree = pr.groupby(["s1_country", "src"])[levels + ["jacc3"]].mean().reset_index()
allrow = pr[levels + ["jacc3"]].mean().to_frame().T.assign(s1_country="ALL", src="ALL")
agree = pd.concat([agree, allrow], ignore_index=True)
by_script = pr.groupby(["s1_country", "script_m"])[levels + ["jacc3"]].mean()
by_script["pairs"] = pr.groupby(["s1_country", "script_m"]).size()
by_script = by_script.reset_index()
jq = pr.groupby("s1_country").jacc3.describe(percentiles=[.05, .1, .25, .5]).reset_index()

# hard examples: latin script both sides, lowest trigram jaccard
cand = pr[(pr.script_m != "non-latin") & (pr.script_1 != "non-latin") & (pr.jacc3 < 0.15)]
hard = cand.sample(min(15, len(cand)), random_state=SEED)[["s1_country", "src", "s1_name", "m_name", "jacc3", "s1_addr", "m_addr"]]
nl = pr[pr.script_m == "non-latin"].sample(min(5, int((pr.script_m == "non-latin").sum())), random_state=SEED)[
    ["s1_country", "src", "s1_name", "m_name"]]
low_share = pr.groupby("s1_country").apply(lambda g: pd.Series({
    "pairs": len(g), "jacc3<0.15": (g.jacc3 < 0.15).mean(), "jacc3<0.3": (g.jacc3 < 0.3).mean(),
    "latin-only jacc3<0.15": ((g.jacc3 < 0.15) & (g.script_m != "non-latin")).mean()})).reset_index()

# ------------------------------------------------------------- write
out.append("## 4.1 Name length (characters)\n")
out.append(md_table(pd.DataFrame(len_rows), floatfmt="{:.1f}"))
out.append("\n## 4.2 Name length (whitespace tokens)\n")
out.append(md_table(pd.DataFrame(tok_rows), floatfmt="{:.1f}"))
out.append("\n## 4.3 Script (share of names)\n`latin-ext` = Latin with accents/extended chars; `non-latin` = contains a letter above U+024F (Devanagari, Kannada, Tamil, ...).\n")
out.append(md_table(pd.DataFrame(script_rows)))
out.append("\n## 4.4 Casing patterns (ASCII letters only)\n")
out.append(md_table(pd.DataFrame(case_rows)))
out.append("\n## 4.5 Legal suffix families (share of names containing at least one token of the family)\n")
suf = pd.DataFrame(suf_rows).fillna(0)
cols = ["split", "source", "country", "n", "any_legal_suffix"] + sorted(c for c in suf.columns if c not in
                                                                        ("split", "source", "country", "n", "any_legal_suffix"))
suf = suf[cols]
keep = [c for c in cols[5:] if suf[c].max() >= 0.002]
out.append(md_table(suf[cols[:5] + keep]))
out.append("\n## 4.6 '&' vs 'and'\n")
out.append(md_table(pd.DataFrame(amp_rows)))
out.append("\n## 4.7 Other name noise\n")
out.append(md_table(pd.DataFrame(odd_rows)))
out.append("\n## 4.8 Most frequent tokens (norm_basic tokens)\n")
tt = pd.DataFrame({f"{k[0]} {k[1]} {k[2]}": [f"{w} ({c:,})" for w, c in v] + [""] * (25 - len(v))
                   for k, v in top_tokens.items()})
for j in range(0, tt.shape[1], 4):
    out.append(md_table(tt.iloc[:, j:j + 4], maxw=40))
    out.append("\n")
out.append("## 4.9 True pairs: how often do names agree?\n")
out.append("Cumulative normalizations: `exact` → `casefold` → `norm_basic` (lowercase, punctuation→space) → "
           "`norm_name` (+accent fold, '&'→'and', drop legal-suffix tokens) → `norm_name_tokensort` (order-insensitive) → "
           "`tokset_subset` (one side's token set contains the other's). `jacc3` = mean char-trigram Jaccard of norm_name.\n")
out.append(md_table(agree))
out.append("\nBy script of the S2/S3 name:\n")
out.append(md_table(by_script))
out.append("\nTrigram-Jaccard distribution of true pairs:\n")
out.append(md_table(jq))
out.append("\n")
out.append(md_table(low_share))
out.append("\n## 4.10 Hard examples: true pairs with very different names (Latin script both sides, jacc3 < 0.15, random 15)\n")
out.append(md_table(hard, floatfmt="{:.3f}", maxw=70))
out.append("\nNon-Latin-script true pairs (random 5):\n")
out.append(md_table(nl))
write_section("4", "\n".join(out) + "\n")
pd.DataFrame(suf).to_csv("out/s4_suffix.csv", index=False)
agree.to_csv("out/s4_agree.csv", index=False)
by_script.to_csv("out/s4_by_script.csv", index=False)
print(md_table(pd.DataFrame(script_rows))); print(md_table(agree)); print(md_table(by_script)); print(md_table(low_share))
print(md_table(suf[cols[:5] + keep]))
print(md_table(hard[["s1_country", "src", "s1_name", "m_name", "jacc3"]]))
