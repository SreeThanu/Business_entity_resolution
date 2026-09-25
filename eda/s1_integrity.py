"""Section 1 - Integrity checks. Also builds the parquet cache."""
import re
import time

import numpy as np
import pandas as pd

from common import DATA, load, load_pairs, md_table, path_of, pct, write_section

FILES = [("train", "source1"), ("train", "source2"), ("train", "source3"), ("train", "ground_truth"),
         ("test", "source1"), ("test", "source2"), ("test", "source3")]

t0 = time.time()
out = ["# 1. Integrity\n"]

# ------------------------------------------------------------- raw byte scan
raw_rows = []
for split, name in FILES:
    p = path_of(split, name)
    b = p.read_bytes()
    bom = b.startswith(b"\xef\xbb\xbf")
    try:
        b.decode("utf-8")
        bad_utf8 = 0
    except UnicodeDecodeError as e:
        bad_utf8 = 1
    n_lines = b.count(b"\n")
    ends_nl = b.endswith(b"\n")
    lines = b.split(b"\n")
    tabs_per_line = np.array([ln.count(b"\t") for ln in lines if ln])
    n_quote_lines = sum(1 for ln in lines if b'"' in ln)
    del lines
    vals, cnts = np.unique(tabs_per_line, return_counts=True)
    raw_rows.append(dict(file=p.name, bytes=len(b), newline_count=n_lines, ends_with_newline=ends_nl,
                         utf8_bom=bom, utf8_decode_error=bool(bad_utf8), crlf=b.count(b"\r"),
                         tabs_per_line=", ".join(f"{v}:{c:,}" for v, c in zip(vals, cnts)),
                         lines_with_quote=n_quote_lines))
    del b
raw = pd.DataFrame(raw_rows)
print("raw scan done", round(time.time() - t0), "s")

# ------------------------------------------------------------- parsed checks
# One file at a time (8 GB RAM); keep only uint64 ID hashes for cross-file checks.
def H(x):
    return pd.util.hash_array(np.asarray(x, dtype=object))


rows, nulls, ws_rows, pref_rows, dup_rows, ex, qex = [], [], [], [], [], [], []
idsets = {}
for split, name in FILES:
    df = load(split, name)
    fname = path_of(split, name).name
    rr = raw.loc[raw.file == fname].iloc[0]
    rows.append(dict(file=fname, parsed_rows=len(df), columns=df.shape[1],
                     lines_minus_header=rr.newline_count - 1 + (0 if rr.ends_with_newline else 1),
                     dup_ids=int(df.iloc[:, 0].duplicated().sum()),
                     exact_dup_rows=int(df.duplicated().sum())))
    for c in df.columns:
        s = df[c]
        st = s.str.strip()
        nulls.append(dict(file=fname, column=c, empty=int((s == "").sum()), empty_rate=float((s == "").mean()),
                          whitespace_only=int(((st == "") & (s != "")).sum()),
                          literal_null_tokens=int(st.str.lower().isin(["nan", "null", "none", "na", "n/a", "-", "--", "0"]).sum())))
        if c in ("entity_id", "source1_entity_id", "matched_entity_ids", "country"):
            continue
        ws_rows.append(dict(
            file=fname, column=c,
            leading_trailing_ws=int((s != st).sum()),
            double_space=int(s.str.contains("  ", regex=False).sum()),
            nbsp=int(s.str.contains("\u00a0", regex=False).sum()),
            zero_width=int(s.str.contains("[\u200b\u200c\u200d\ufeff]", regex=True).sum()),
            control_chars=int(s.str.contains(r"[\x00-\x08\x0b-\x1f\x7f]", regex=True).sum()),
            replacement_char=int(s.str.contains("\ufffd", regex=False).sum()),
            mojibake_like=int(s.str.contains("Ã.|â€|Â", regex=True).sum()),
            non_ascii=int((~s.str.isascii()).sum()) if hasattr(s.str, "isascii") else int((~s.astype(str).map(str.isascii)).sum()),
        ))
        mask = s.str.contains("[\u200b\u200c\u200d\ufeff\u00a0]|[\x00-\x08\x0b-\x1f\x7f]", regex=True)
        for v in s[mask].head(2).tolist():
            ex.append(dict(file=fname, column=c, value=repr(v)))
    if name == "ground_truth":
        gt = df
        continue
    ids = df["entity_id"]
    exp = "S" + name[-1] + "-"
    pref_rows.append(dict(file=fname, wrong_prefix=int((~ids.str.startswith(exp)).sum()),
                          non_numeric_suffix=int((~ids.str.slice(3).str.fullmatch(r"\d+")).sum()),
                          min_id_len=int(ids.str.len().min()), max_id_len=int(ids.str.len().max())))
    idsets[(split, name)] = np.unique(H(ids.tolist()))
    dup_rows.append(dict(file=fname,
                         rows_sharing_name_addr_country=int(df[["business_name", "business_address", "country"]].duplicated(keep=False).sum()),
                         rows_sharing_name_country=int(df[["business_name", "country"]].duplicated(keep=False).sum())))
    if name == "source1":
        mask = df.business_name.str.contains('"', regex=False) | df.business_address.str.contains('"', regex=False)
        qex.append(df[mask].head(3).astype(str).assign(file=fname))
    del df
    print("parsed", fname, flush=True)

parsed = pd.DataFrame(rows); nulls = pd.DataFrame(nulls); ws = pd.DataFrame(ws_rows)
pref = pd.DataFrame(pref_rows); dupc = pd.DataFrame(dup_rows); ex = pd.DataFrame(ex); qex = pd.concat(qex)

cross = pd.DataFrame([dict(source=f"S{i}", train_ids=len(idsets[("train", f"source{i}")]),
                           test_ids=len(idsets[("test", f"source{i}")]),
                           ids_in_both=len(np.intersect1d(idsets[("train", f"source{i}")], idsets[("test", f"source{i}")])))
                      for i in (1, 2, 3)])
# cross-source overlap within a split (would indicate prefix/ID reuse)

pairs = load_pairs()
m = pairs["match"]
mh = H(m.tolist())
gt_ids = gt["source1_entity_id"]
gth = H(gt_ids.tolist())
s1h, s2h, s3h = idsets[("train", "source1")], idsets[("train", "source2")], idsets[("train", "source3")]
t2h, t3h = idsets[("test", "source2")], idsets[("test", "source3")]
in2, in3 = np.isin(mh, s2h), np.isin(mh, s3h)
raw_list = gt["matched_entity_ids"]
gt_checks = {
    "GT rows": len(gt),
    "GT rows with empty match list": int((raw_list == "").sum()),
    "GT source1 IDs duplicated": int(gt_ids.duplicated().sum()),
    "GT source1 IDs missing from train_source1": int((~np.isin(gth, s1h)).sum()),
    "train_source1 IDs missing from GT": int((~np.isin(s1h, gth)).sum()),
    "matched IDs total (pairs)": len(pairs),
    "matched IDs with S2- prefix": int(m.str.startswith("S2-").sum()),
    "matched IDs with S3- prefix": int(m.str.startswith("S3-").sum()),
    "matched IDs with other prefix (incl. S1-)": int((~m.str.startswith(("S2-", "S3-"))).sum()),
    "matched S2 IDs not in train_source2": int((m.str.startswith("S2-").values & ~in2).sum()),
    "matched S3 IDs not in train_source3": int((m.str.startswith("S3-").values & ~in3).sum()),
    "matched IDs found in a TEST S2/S3 file": int((np.isin(mh, t2h) | np.isin(mh, t3h)).sum()),
    "duplicate IDs inside one match list": int(pairs.duplicated().sum()),
    "match lists with whitespace around IDs": int(raw_list.str.contains(r"\s", regex=True).sum()),
    "match lists with empty elements (',,' or trailing ',')": int(raw_list.str.contains(r"(?:^,|,,|,$)", regex=True).sum()),
}
gtc = pd.DataFrame({"check": list(gt_checks), "value": list(gt_checks.values())})

# ------------------------------------------------------------- write
out.append("## 1.1 Raw file scan (bytes)\n")
out.append(md_table(raw))
out.append("\n`tabs_per_line` = distribution of tab count per physical line (3 expected for sources, 1 for ground truth). "
           "Double quotes appear only as RFC-4180 quoted fields (`\"\"` escapes); pandas' default QUOTE_MINIMAL parser "
           "handles them correctly (parsed rows = physical lines - header, below).\n")
out.append("## 1.2 Parsed row counts, duplicate IDs, exact-duplicate rows\n")
out.append(md_table(parsed))
out.append("\n## 1.3 Empty / null rates per column\n")
out.append(md_table(nulls))
out.append("\n`literal_null_tokens` counts values like `nan`, `null`, `none`, `n/a`, `-`, `0` (after strip, lowercase).\n")
out.append("## 1.4 Whitespace, encoding and odd characters (rows affected)\n")
out.append(md_table(ws))
if len(ex):
    out.append("\nExamples:\n")
    out.append(md_table(ex))
out.append("\n\nQuoted-field examples (parsed values):\n")
out.append(md_table(qex))
out.append("\n## 1.5 ID format\n")
out.append(md_table(pref))
out.append("\nTrain/test ID overlap per source:\n")
out.append(md_table(cross))
out.append("\n## 1.6 Same content under different IDs\n")
out.append(md_table(dupc))
out.append("\n## 1.7 Ground-truth consistency\n")
out.append(md_table(gtc))

write_section("1", "\n".join(out) + "\n")
parsed.to_csv("out/s1_parsed.csv", index=False)
nulls.to_csv("out/s1_nulls.csv", index=False)
ws.to_csv("out/s1_ws.csv", index=False)
gtc.to_csv("out/s1_gt.csv", index=False)
dupc.to_csv("out/s1_dupc.csv", index=False)
print(md_table(parsed)); print(md_table(gtc)); print(md_table(dupc)); print(md_table(ws))
print(md_table(nulls[nulls["empty"] > 0]))
print("done", round(time.time() - t0), "s")
