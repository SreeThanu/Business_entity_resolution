"""Assemble eda/EDA_REPORT.md from the per-section outputs in eda/out/.

Run order (each script is independent and re-runnable; caches live in eda/cache/):
    python s1_integrity.py      # also builds the parquet cache
    python s2_country.py
    python s3_ground_truth.py   # writes cache/s1_match_stats.parquet (used by s6, s8)
    python s4_names.py
    python s5_addresses.py
    python s6_ambiguity.py
    python s7_blocking.py       # ~35 min on an 8 GB laptop (name, then name+address)
    python s7_report.py
    python s6b_discriminators.py  # needs s7 output
    python s8_shift.py
    python build_report.py
"""
from datetime import date

from common import EDA, OUT

HEADER = f"""# Business Entity Resolution: Exploratory Data Analysis

*Generated {date.today().isoformat()} by `eda/build_report.py`. Every number below was computed by the scripts in `eda/`
(see the run order at the top of `build_report.py`). The files in `dataset/` were never modified.*

**Reading notes**
- Files are read with `pd.read_csv(path, sep="\\t", dtype=str, keep_default_na=False)`, and the column count is checked.
- `nname` means the normalised name: accents folded (Latin only), lowercased, punctuation removed, `&` turned into `and`,
  and legal-suffix tokens dropped (`common.norm_name`). `hnum` means the first number in the address.
- To fit in 8 GB of RAM, some statistics use samples. Each section states its sample size, and the seed is always 42:
  - True-pair text statistics use 1,000,000 of the 7,638,365 pairs.
  - Token frequencies use up to 200,000 rows per (split, source, country) group.
  - The blocking check uses 12,000 random S1 queries.

## Contents
1. Integrity · 2. Country distribution · 3. Ground truth structure · 4. Names · 5. Addresses ·
6. Chains and ambiguity · 7. Blocking-recall check · 8. Train vs test shift · 9. Summary
"""

IMPL = {
"1": """
### Implications for our pipeline
- Parsing is safe with the mandated `read_csv` call. Quoted fields (`""` escapes) parse correctly, so no custom parser is needed.
- The ground truth is clean and complete. We can build a validation split directly from it with no repair step.
- S1 has no empty names or addresses, and train S1 names are 100% ASCII. Nearly all the noise sits on the S2/S3 side.
  So normalisation should target S2/S3 patterns: uppercase text, double spaces, `\\x1a` in place of an apostrophe,
  double-encoded dashes (`Â\\x80\\x93`), and ZWNJ inside Indic words (legitimate; keep it).
- About 3.4% of S2/S3 addresses are empty. The matcher needs a name-only path for these records, with a stricter threshold.
- S2/S3 contain rows with identical name, address and country under different IDs (50,933 in train S2, 37,241 in train S3).
  Because matching is one-to-one (Section 3), these are either several true matches of the same S1 entity or distractors.
  They must not be collapsed before matching.
""",
"2": """
### Implications for our pipeline
- Country labels are clean and 100% consistent within true pairs (Section 3.5). Country is a safe hard blocking key.
  The code must treat country as an open set of labels (France appears only in test).
- Test has **5.5–5.8 S2+S3 records per S1 entity, versus 4.68 in train**, for every country, US and India included.
  So test contains either more matches per entity or more distractors. Thresholds and any "expected number of matches"
  prior tuned on train may not transfer. This is an open question (Section 9).
- France is 15% of test S1 entities (259,452). If our pipeline scored 0 on France, it would lose about 0.15 of the final
  macro-F0.5.
""",
"3": """
### Implications for our pipeline
- **The precision-heavy framing is misleading here.** Only 5.58% of S1 entities are singletons. The average entity has
  3.67 true matches, spread over both S2 and S3, and often several from the same source. Returning only the single best
  match (perfect precision) caps macro-F0.5 at about **0.70**. Returning every match plus one false match scores about **0.81**.
  Recall on multi-match entities is where the score is.
- The **one-to-one constraint is strict**: no S2/S3 record belongs to more than one S1 entity. We should resolve conflicts
  globally, for example by giving each S2/S3 record only to its best-scoring S1 entity. This also suppresses chain-name
  false merges.
- About a quarter of S2/S3 records are distractors (26.6% of S2, 25.4% of S3). A good pipeline must leave many
  high-similarity candidates unmatched.
- Several matches can come from the same source. The matcher must not enforce "at most one per source".
- Singletons score 1.0 only with an empty prediction. We need a calibrated "no match" decision, but singletons are only
  5.6% of the score.
""",
"4": """
### Implications for our pipeline
- Exact name equality holds for only 4.7% of true pairs. Our `norm_name` normalisation raises that to 47.7%, token-sort
  to 50.2%, and a token-subset test to 66.5%.
  - Useful name features: normalised equality, token-set containment, char-trigram Jaccard, and edit distance.
  - Needed normalisation: legal-suffix handling, reordering (`PVT. ASTOR TRADING LTD.`), OCR digits for letters
    (`H0uston`, `De1hi`), injected accents (`Ínc`, `Frànce`), bracketed or duplicated words.
- **Non-Latin scripts:** 23% of India S2 names and 13% of India S3 names are in Indic scripts (Devanagari, Kannada, Tamil,
  Telugu, Bengali, Gujarati, Gurmukhi, Malayalam). This is 18% of India true pairs, and they have almost no character
  overlap with the Latin S1 name (mean trigram Jaccard 0.02).
  - These pairs need either a transliteration step, or matching on address and house number alone.
  - Legal suffixes also appear in script, e.g. `प्राइवेट लिमिटेड`.
- About 2% of pairs share essentially no name text even in Latin script. These are invented brand/DBA names
  (`Keloonyxavi`), acronyms (`DI`, `TS`, `FF`), web domains (`sarthitrading.com`, 3–4% of S2/S3 names) and `@handles`.
  These pairs can only be recovered through the address.
- Legal-suffix mix differs by country:
  - US: LLC 27%, Inc 18%.
  - India: Private 63%, Limited 76%.
  - France (test only): SARL 28%, SAS 20%, EURL 6.5%, SA, SASU, SCI.
  - Suffixes carry signal (Section 6.5). Strip them for similarity, but keep "suffix family agrees/conflicts" as its own feature.
- `&` and `and` are used interchangeably (US S1: `&` in 5.6%, `and` in 4.3%). Map one to the other.
""",
"5": """
### Implications for our pipeline
- **The postal-code hypothesis is false.** Addresses contain essentially no ZIP, PIN or French postal codes: under 2% of
  addresses in any source/country, and "both missing" in 97–99.98% of true pairs. Postal code cannot be a blocking key or
  a primary feature. 5-digit numbers in US addresses are house numbers.
- **House number is the key address field.** It matches on 65% of India and 75% of US true pairs. Section 6.5 shows it is
  the strongest single discriminator against look-alike distractors.
- **State and region formats differ by source:**
  - US: S1/S2 use 2-letter codes; S3 uses full names (`Texas`).
  - India: S1 uses full names; S3 uses codes (`MH`, `DL`, `KA`); in S2, about 24% of addresses contain native-script text, mostly the state name (`महाराष्ट्र`).
  - France: S1 uses regions (`Hauts-de-France`); S2/S3 often use departments (`Nord`, `Gironde`).
  - We need a canonical state/region mapping. It can be learned from train pairs, because the S1 state and the S3 code
    co-occur on true pairs.
- Components are reordered (`KS, 124 ERIE ST, WICHITA`). Use order-free token or trigram similarity, not positional comparison.
- Abbreviation expansion matters for US S2/S3 (`ST`, `DR`, `AVE`, `LN`, `RD` in 4–11% of addresses). Filler tokens
  (`null`, `N/A`, `##`) and city aliases (`Bombay`/`Mumbai`, `Calcutta`/`Kolkata`) should be normalised.
- Landmark phrases (`near`, `opp`, `c/o`, `behind`) occur almost only in India addresses (4–6%). They add noise tokens
  and should be down-weighted, not relied on.
- S2/S3 addresses are often truncated to a few components (for example `A/404, Bombay, Mumbai Suburban, MH`). Address
  similarity must tolerate one side being a subset of the other.
""",
"6": """
### Implications for our pipeline
- Names alone are very ambiguous. About half of S1 names are shared with other S1 entities (US 46%, India 53%), with
  chains of 100–550 entities (`meridian`, `summit`, `ear nose and throat group`, `shree trading`).
  "Match if normalised name is equal" has 3–6% precision.
- The data contains **deliberate near-duplicate distractors**. They reuse the S1 name, or a one-token variation of it,
  with the same street and city but:
  - a slightly shifted house number (US median difference 7), or
  - a different legal-suffix family (LLC vs Inc vs Corp), or
  - an added word (`Exports`, `Infratech`, `Overseas`, `Public`).
- Signals that separate true matches from look-alikes with cosine ≥ 0.8 (Section 6.5):
  - Exact house-number agreement: US 85% of true matches vs 2.8% of look-alikes; India 72% vs 29%.
  - Legal-suffix family conflict: 0.5% of true matches vs 22% of look-alikes (US).
  - Added core-name word: 17% of true matches vs 56% of look-alikes (US); 36% vs 90% (India).
- A small house-number difference is a **distractor signature**. When a true match's house number differs, the gap is
  usually large (median 1,000 in the US), which suggests parsing noise such as unit numbers. So the feature should be the
  size of the difference, not just a flag.
- "nname + hnum" is a high-precision rule in the US (97.8% precision, 39% recall). In India it is weaker (71% precision),
  because Indian house numbers are messier (`6-3-663/G/4`, `Rz-40` vs `Rz-0040`).
""",
"7": """
### Implications for our pipeline
- **Name-only TF-IDF is not a viable blocker:** recall@100 is only 65.7% (US 70.7%, India 58.3%). Chain names fill the
  top-k with same-name records at other addresses.
- **Name + address TF-IDF is a strong baseline blocker:**
  - Recall@10 is 90.7%, @20 92.9%, @50 94.5% (US 97.6%, India 90.0%), @100 95.5%.
  - At k=50, 85% of entities have all their matches retrieved.
  - Candidate budget at k=50 is about 50 × 1.73M test S1 entities ≈ 87M pairs, which is feasible.
- Restricting the pool to the same country changes recall by at most 0.4 points. It saves compute but not recall.
- Remaining misses at k=50 are mostly:
  1. India names in non-Latin script whose address is short or truncated.
  2. Empty S2/S3 addresses.
  3. Invented brand/domain names.
  - Fixes: a second blocking pass keyed on (country, state, house number); transliteration for Indic names; a union of
    name-only and name+address candidate lists.
- Singletons are not easy to spot from retrieval scores. Their top-1 name cosine has median 0.95–0.98, and 65–79% have a
  top-1 non-match with name cosine ≥ 0.9. The singleton decision must come from the pairwise matcher (house number,
  suffix, extra tokens), not from a similarity threshold.
- Caveat: the vocabulary was hashed into 2^24 buckets because an exact `TfidfVectorizer.fit` on 12.5M documents does not
  fit in 8 GB of RAM. The IDF formula and L2 normalisation match sklearn's defaults.
""",
"8": """
### Implications for our pipeline
- **US and India show no meaningful train→test shift** in lengths, scripts or formats:
  - Name/address length percentiles are identical to within one character.
  - 93–98% of test token occurrences were seen in train.
  - Postal-code patterns are identical (Section 5.1).
  - Models trained on train US/India should transfer.
- **The shift that matters is volume:** test has more S2/S3 records per S1 entity (Section 2).
- **France** looks like the same generator with a French lexicon:
  - Structure: 3 components in the form `N Rue X, City, Region` (S1). S2/S3 use departments and abbreviate
    `Rue`→`R`/`R.`, write `N°`/`NO` before house numbers, and add leading zeros (`005`).
  - Coverage: 15 main cities in 3 regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire).
  - Names: legal forms SARL, SAS, EURL, SA, SASU, SCI, sometimes dotted (`E.U.R.L.`); organisation words `club`,
    `amicale`, `ecole`, `comite`, `association`; genuine accents in S1 (16% of names) plus injected ones in S2/S3 (24%).
- Only 43–54% of French name-token occurrences appear in the train vocabulary. Any learned name-token model (for example
  learned token weights or embeddings trained on train only) will be weak on France. Character-level and structural
  features (house-number agreement, suffix family, token containment) should transfer. The legal-suffix list, the
  abbreviation map (`r`→`rue`, `av`→`avenue`, `bd`→`boulevard`) and the region↔department map need French entries.
- 95.9% of France S1 entities have an S2/S3 record with the same `nname`, versus 92.2–92.9% for train and test
  US/India. That suggests France has a similar or higher match rate, not a mostly-singleton population.
""",
}

SUMMARY = """
# 9. Summary

## Top 10 findings, ranked by impact on macro-F0.5

| # | Finding | Evidence | What to do |
|---|---|---|---|
| 1 | **Recall carries the score; singletons are rare.** | Singleton rate is 5.58%, so all-empty scores 0.0558. Non-singletons average 3.67 matches. Perfect precision with one match per entity caps the score at 0.70; all matches plus one false match scores 0.81 (Section 3.6). | Aim to return *all* matches. Tune thresholds on macro-F0.5 on a held-out split, not on pairwise precision. |
| 2 | **Name-only blocking loses a third of matches; name+address fixes most of it.** | TF-IDF recall@50: name 61.9%, name+address 94.5% (US 97.6%, India 90.0%). Recall@100: 65.7% vs 95.5% (Section 7). | Block on name+address char TF-IDF (k≈50), plus a secondary block on (country, state, house number). |
| 3 | **Strict one-to-one assignment.** | 0 of 7.64M matched S2/S3 IDs appear in more than one S1 list (Section 3.3). | Assign each S2/S3 record to at most one S1 entity (best score wins). This kills chain false merges. |
| 4 | **Look-alike distractors: same name, same street, shifted house number or swapped suffix.** | US look-alikes: house number differs in 97% (median gap 7); suffix family conflicts in 22% vs 0.5% for true matches (Section 6.5). | Features: house-number equality and gap size, suffix-family agreement, added/dropped core tokens. |
| 5 | **Chain names make name similarity alone useless.** | 46–53% of S1 names are shared; "equal nname" has 3–6% precision. nname+hnum: 97.8% precision in the US, 71% in India (Section 6). | Name similarity is necessary, never sufficient. Always pair it with address and house-number evidence. |
| 6 | **Indic-script names for India.** | 23% of India S2 and 13% of India S3 names; 18% of India true pairs; trigram Jaccard ≈ 0.02 (Section 4.9). | Add a transliteration step, or an address-driven match path, for non-Latin names. |
| 7 | **No postal codes.** | Under 2% of addresses in any group; 97–99.98% of true pairs have none on either side (Section 5). | Do not build on ZIP/PIN. Use house number, street tokens, city and state instead. |
| 8 | **Source-specific formats.** | US S3 writes full state names; India S3 uses state codes; about 24% of India S2 addresses contain native-script text (mostly the state); S2 US is 90% uppercase with abbreviations; components are reordered (Section 5). | Canonicalise state/region, expand abbreviations, and use order-free similarity. |
| 9 | **Heavy name noise even in Latin script.** | Exact 4.7%, normalised 47.7%, token-subset 66.5%. About 2% of pairs are brand/acronym/domain names with no text overlap (Section 4). | Use several name similarities as features, and fall back to address for brand names. |
| 10 | **France: unseen country, same structure, different vocabulary; test also has more records per entity.** | France is 15% of test S1; 43–54% of its name-token occurrences are seen in train. Test S2+S3 per S1 is 5.5–5.8 vs 4.68 in train (Sections 2, 8). | Keep features language-agnostic, add French suffix and abbreviation lists, and re-check thresholds for the volume shift. |

## Hypotheses that turned out false or needed correcting
- *"Postal codes will be useful (US ZIP/ZIP+4, India 6-digit PIN)."* **False.** They are almost never present in any source,
  and France has none either.
- *"Precision-heavy metric, so predict conservatively."* **Only partly true.** Singletons are rare and entities have about
  3.7 matches each, so aggressive recall on confident entities is worth more than extra caution.
- *"Normalising names (lowercase, strip punctuation, remove suffixes) makes most pairs equal."* **Only for 48% of pairs.**
- The one-to-one hypothesis was **confirmed** without exception.

## Open questions the data could not answer
1. **Why does test have about 23% more S2+S3 records per S1 entity?** Are these extra matches per entity or extra
   distractors? This decides whether train-tuned thresholds and match-count priors transfer. The public leaderboard is
   the only signal.
2. **France match and singleton rates are unknown.** The same-name proxy (95.9%) suggests they resemble train, but there
   are no labels.
3. **Is an offline transliteration library allowed** under the "no external data" rule? It adds no business data, but it
   ships linguistic tables. The same question applies to hand-written state and department maps. The conservative option
   is to learn these mappings from train pairs (for example S1 `Maharashtra` ↔ S3 `MH` ↔ S2 `महाराष्ट्र`).
4. **Were the test distractors generated the same way as train?** For example, house-number shifts and suffix swaps.
   Section 6.5 features assume they were.
5. **How are the public and private leaderboards split by country?** If France is unevenly split, public scores may not
   predict private ones.
6. Some statistics are sample-based (1M pairs, 12k blocking queries). Sampling error is below ±0.5 points for all rates
   quoted, but rare-pattern counts (e.g. specific suffix variants) may shift slightly on the full data.
"""


def main():
    parts = [HEADER]
    for s in ["1", "2", "3", "4", "5", "6", "7", "8"]:
        p = OUT / f"section{s}.md"
        parts.append(p.read_text(encoding="utf-8") if p.exists() else f"# {s}. (missing: run the section script)\n")
        if s == "6" and (OUT / "section6b.md").exists():
            parts.append((OUT / "section6b.md").read_text(encoding="utf-8"))
        parts.append(IMPL[s])
    parts.append(SUMMARY)
    (EDA / "EDA_REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", EDA / "EDA_REPORT.md")


if __name__ == "__main__":
    main()
