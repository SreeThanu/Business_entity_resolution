"""Section 2 - Country distribution."""
import pandas as pd

from common import load, md_table, write_section

out = ["# 2. Country distribution\n"]
rows = []
raw_labels = []
for split in ("train", "test"):
    for i in (1, 2, 3):
        c = load(split, f"source{i}")["country"].astype(str)
        vc = c.value_counts()
        for lab, n in vc.items():
            rows.append(dict(split=split, source=f"S{i}", country=lab, n=int(n)))
            raw_labels.append(dict(split=split, source=f"S{i}", label_repr=repr(lab), n=int(n),
                                   normalized=lab.strip().lower()))
df = pd.DataFrame(rows)
piv = df.pivot_table(index=["split", "country"], columns="source", values="n", aggfunc="sum", fill_value=0)
piv["total"] = piv.sum(axis=1)
piv = piv.reset_index()
for s in ("S1", "S2", "S3", "total"):
    tot = piv.groupby("split")[s].transform("sum")
    piv[f"{s}_share"] = piv[s] / tot
lab = pd.DataFrame(raw_labels)
variants = lab.groupby("normalized")["label_repr"].nunique().reset_index(name="distinct_raw_spellings")

# records per S1 entity (per country) - pool size
ratio = piv[["split", "country", "S1", "S2", "S3"]].copy()
ratio["S2_per_S1"] = ratio.S2 / ratio.S1
ratio["S3_per_S1"] = ratio.S3 / ratio.S1
ratio["S2+S3_per_S1"] = (ratio.S2 + ratio.S3) / ratio.S1

out.append("## 2.1 Records per country per source\n")
out.append(md_table(piv, floatfmt="{:.4f}"))
out.append("\n## 2.2 Raw label spellings (repr shows hidden whitespace)\n")
out.append(md_table(lab))
out.append("\nDistinct raw spellings per normalized label (strip+lower):\n")
out.append(md_table(variants))
out.append("\n## 2.3 Pool size: S2/S3 records per S1 entity\n")
out.append(md_table(ratio, floatfmt="{:.3f}"))
write_section("2", "\n".join(out) + "\n")
piv.to_csv("out/s2_country.csv", index=False)
ratio.to_csv("out/s2_ratio.csv", index=False)
print(md_table(piv)); print(md_table(lab)); print(md_table(ratio, floatfmt="{:.3f}"))
