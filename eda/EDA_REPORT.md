# Business Entity Resolution: Exploratory Data Analysis

*Generated 2026-09-25 by `eda/build_report.py`. Every number below was computed by the scripts in `eda/`
(see the run order at the top of `build_report.py`). The files in `dataset/` were never modified.*

**Reading notes**
- Files are read with `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`, and the column count is checked.
- `nname` means the normalised name: accents folded (Latin only), lowercased, punctuation removed, `&` turned into `and`,
  and legal-suffix tokens dropped (`common.norm_name`). `hnum` means the first number in the address.
- To fit in 8 GB of RAM, some statistics use samples. Each section states its sample size, and the seed is always 42:
  - True-pair text statistics use 1,000,000 of the 7,638,365 pairs.
  - Token frequencies use up to 200,000 rows per (split, source, country) group.
  - The blocking check uses 12,000 random S1 queries.

## Contents
1. Integrity · 2. Country distribution · 3. Ground truth structure · 4. Names · 5. Addresses ·
6. Chains and ambiguity · 7. Blocking-recall check · 8. Train vs test shift · 9. Summary

# 1. Integrity

## 1.1 Raw file scan (bytes)

| file | bytes | newline_count | ends_with_newline | utf8_bom | utf8_decode_error | crlf | tabs_per_line | lines_with_quote |
|---|---|---|---|---|---|---|---|---|
| train_source1.tsv | 210,069,713 | 2,206,822 | 1 | 0 | 0 | 0 | 3:2,206,822 | 4 |
| train_source2.tsv | 489,301,488 | 5,034,617 | 1 | 0 | 0 | 0 | 3:5,034,617 | 6 |
| train_source3.tsv | 503,705,637 | 5,285,604 | 1 | 0 | 0 | 0 | 3:5,285,604 | 0 |
| train_ground_truth.tsv | 127,015,583 | 2,206,822 | 1 | 0 | 0 | 0 | 1:2,206,822 | 0 |
| test_source1.tsv | 175,022,086 | 1,732,545 | 1 | 0 | 0 | 0 | 3:1,732,545 | 134 |
| test_source2.tsv | 509,456,422 | 4,887,274 | 1 | 0 | 0 | 0 | 3:4,887,274 | 349 |
| test_source3.tsv | 506,002,772 | 5,082,317 | 1 | 0 | 0 | 0 | 3:5,082,317 | 330 |

`tabs_per_line` = distribution of tab count per physical line (3 expected for sources, 1 for ground truth). Double quotes appear only as RFC-4180 quoted fields (`""` escapes); pandas' default QUOTE_MINIMAL parser handles them correctly (parsed rows = physical lines - header, below).

## 1.2 Parsed row counts, duplicate IDs, exact-duplicate rows

| file | parsed_rows | columns | lines_minus_header | dup_ids | exact_dup_rows |
|---|---|---|---|---|---|
| train_source1.tsv | 2,206,821 | 4 | 2,206,821 | 0 | 0 |
| train_source2.tsv | 5,034,616 | 4 | 5,034,616 | 0 | 0 |
| train_source3.tsv | 5,285,603 | 4 | 5,285,603 | 0 | 0 |
| train_ground_truth.tsv | 2,206,821 | 2 | 2,206,821 | 0 | 0 |
| test_source1.tsv | 1,732,544 | 4 | 1,732,544 | 0 | 0 |
| test_source2.tsv | 4,887,273 | 4 | 4,887,273 | 0 | 0 |
| test_source3.tsv | 5,082,316 | 4 | 5,082,316 | 0 | 0 |

## 1.3 Empty / null rates per column

| file | column | empty | empty_rate | whitespace_only | literal_null_tokens |
|---|---|---|---|---|---|
| train_source1.tsv | entity_id | 0 | 0.0000 | 0 | 0 |
| train_source1.tsv | business_name | 0 | 0.0000 | 0 | 0 |
| train_source1.tsv | business_address | 0 | 0.0000 | 0 | 0 |
| train_source1.tsv | country | 0 | 0.0000 | 0 | 0 |
| train_source2.tsv | entity_id | 0 | 0.0000 | 0 | 0 |
| train_source2.tsv | business_name | 0 | 0.0000 | 0 | 6 |
| train_source2.tsv | business_address | 168,967 | 0.0336 | 0 | 0 |
| train_source2.tsv | country | 0 | 0.0000 | 0 | 0 |
| train_source3.tsv | entity_id | 0 | 0.0000 | 0 | 0 |
| train_source3.tsv | business_name | 0 | 0.0000 | 0 | 18 |
| train_source3.tsv | business_address | 175,916 | 0.0333 | 0 | 0 |
| train_source3.tsv | country | 0 | 0.0000 | 0 | 0 |
| train_ground_truth.tsv | source1_entity_id | 0 | 0.0000 | 0 | 0 |
| train_ground_truth.tsv | matched_entity_ids | 123,247 | 0.0558 | 0 | 0 |
| test_source1.tsv | entity_id | 0 | 0.0000 | 0 | 0 |
| test_source1.tsv | business_name | 0 | 0.0000 | 0 | 0 |
| test_source1.tsv | business_address | 0 | 0.0000 | 0 | 0 |
| test_source1.tsv | country | 0 | 0.0000 | 0 | 0 |
| test_source2.tsv | entity_id | 0 | 0.0000 | 0 | 0 |
| test_source2.tsv | business_name | 0 | 0.0000 | 0 | 49 |
| test_source2.tsv | business_address | 129,408 | 0.0265 | 0 | 0 |
| test_source2.tsv | country | 0 | 0.0000 | 0 | 0 |
| test_source3.tsv | entity_id | 0 | 0.0000 | 0 | 0 |
| test_source3.tsv | business_name | 0 | 0.0000 | 0 | 61 |
| test_source3.tsv | business_address | 136,098 | 0.0268 | 0 | 0 |
| test_source3.tsv | country | 0 | 0.0000 | 0 | 0 |

`literal_null_tokens` counts values like `nan`, `null`, `none`, `n/a`, `-`, `0` (after strip, lowercase).

## 1.4 Whitespace, encoding and odd characters (rows affected)

| file | column | leading_trailing_ws | double_space | nbsp | zero_width | control_chars | replacement_char | mojibake_like | non_ascii |
|---|---|---|---|---|---|---|---|---|---|
| train_source1.tsv | business_name | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| train_source1.tsv | business_address | 0 | 968 | 0 | 0 | 70 | 0 | 443 | 554 |
| train_source2.tsv | business_name | 0 | 554,392 | 0 | 25,643 | 3 | 0 | 0 | 764,608 |
| train_source2.tsv | business_address | 0 | 107,574 | 0 | 0 | 119 | 0 | 1,118 | 478,453 |
| train_source3.tsv | business_name | 0 | 575,616 | 0 | 14,514 | 1 | 0 | 0 | 606,737 |
| train_source3.tsv | business_address | 0 | 4,874 | 0 | 0 | 125 | 0 | 785 | 476,588 |
| test_source1.tsv | business_name | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40,789 |
| test_source1.tsv | business_address | 0 | 753 | 0 | 0 | 72 | 0 | 400 | 73,800 |
| test_source2.tsv | business_name | 0 | 500,237 | 0 | 31,104 | 0 | 0 | 1,235 | 928,158 |
| test_source2.tsv | business_address | 0 | 82,739 | 0 | 0 | 183 | 0 | 2,812 | 720,665 |
| test_source3.tsv | business_name | 0 | 519,596 | 0 | 17,468 | 0 | 0 | 1 | 737,515 |
| test_source3.tsv | business_address | 0 | 3,126 | 0 | 0 | 162 | 0 | 856 | 729,222 |

Examples:

| file | column | value |
|---|---|---|
| train_source1.tsv | business_name | 'Medchal\x1aMalkajgiri Housekeeping Private Limited' |
| train_source1.tsv | business_address | 'Second Floor Vasant Towers, Bearing Municipal No. 1-11-251/1B, Behind Shopper\x1aS Stop,… |
| train_source1.tsv | business_address | 'D.No.6-107/1B, S.No.169/6-Part, Division No 05, Opp:Tile\x1aS Mark, Chandrampalem, Madhu… |
| train_source2.tsv | business_name | 'పర్\u200cఫెక్ట్ యునైటెడ్ మీడియా ప్రైవేట్ లిమిటెడ్' |
| train_source2.tsv | business_name | 'శ్యామ్ టెక్నాలజీ హార్డ్\u200cవేర్ ప్రైవేట్ లిమిటెడ్' |
| train_source2.tsv | business_address | 'Delhi, UNIT NO.30, FLOOR 3, WORLDMARK Â\x80\x93 3, ASSET Â\x80\x93 7, AEROCITY, N.H Â\x8… |
| train_source2.tsv | business_address | 'PHASE \x1a 6B, DIVYASREE TECHNOPOLIS, 124-125, YEMLUR P.O., OFF OLD AIRPORT RO, AD, BENG… |
| train_source3.tsv | business_name | 'കൃഷ്ണ ഇന്റർനാഷണൽ ഹാർഡ്\u200cവെയർ പ്രൈവറ്റ് ലിമിറ്റഡ്' |
| train_source3.tsv | business_name | 'Southern ಇನ್ವೆಸ್ಟ್\u200cಮೆಂಟ್ ಲಿಮಿಟೆಡ್' |
| train_source3.tsv | business_address | 'Sneh Bunglow, Plot No. 26, Survey No: 712/1/28, 26, D\x1asouza Colony, College Road, Nas… |
| train_source3.tsv | business_address | 'Second Floor, Site No.409, Bbmp Katha No.4041/409, Hosur Sarjapur Road (Hsr) Layout, Sec… |
| test_source1.tsv | business_address | '\x1aRaman & Raman Shopping Complex\x1a, 48, Thiruvidaimarudur Road, \x1a, Kumbakonam, Th… |
| test_source1.tsv | business_address | '2Nd Floor, Flat No. \x1a 203 11, Durgadas Kuthi Lane, Lp-118/3/1, Howrah, West Bengal' |
| test_source2.tsv | business_name | 'సుప్రీమ్ మోడర్న్ ఇన్వెస్ట్\u200cమెంట్ ప్రైవేట్ లిమిటెడ్' |
| test_source2.tsv | business_name | 'ಇಂಟರ್\u200cನ್ಯಾಷನಲ್ ಎನರ್ಜಿ ಎಲ್ಎಲ್\u200cಪಿ' |
| test_source2.tsv | business_address | 'NELLORE, SAPTHAGIRI COLONY, NEAR GVR COLLEGE, MIN, I BY PASS, JAYA SAI LAKSHMI TOWERS D.… |
| test_source2.tsv | business_address | 'H.NO 588 FLAT NO.-1493, PLOT NO. GH-01, SEC. 4, 01ST AVENUE, BLOCK \x1aB, GAUR CITY, GRE… |
| test_source3.tsv | business_name | 'ଡାଇନାମିକ୍ ହୋଟେଲ୍ କନସଲଟାଣ୍ଟସ୍ ଏଲ୍\u200cଏଲ୍\u200cପି' |
| test_source3.tsv | business_name | 'రియల్ ఎక్స్\u200cపోర్ట్స్ ప్రైవేట్ లిమిటెడ్' |
| test_source3.tsv | business_address | 'C/O Shridhar Prasad Yadav, Village P.O. - Faga P S: Bandhuawa Kuraba, Via - Baunsi, Di, … |
| test_source3.tsv | business_address | 'M \x1a1501, Tower-m, Anjara Grand Heritage Sector 74, Noida, Gautam Buddha Nagar, उत्तर … |


Quoted-field examples (parsed values):

| entity_id | business_name | business_address | country | file |
|---|---|---|---|---|
| S1-317037102 | Royal Food Pvt Ltd | Karnataka, 4Th Floor, Unit 402, "Prestige Feroze" Building, Cunningham Road, Vasanth Naga… | India | train_source1.tsv |
| S1-830575986 | Astor Trading Pvt. Ltd. | 9Th Floor, "Niagara" Building, Embassy Taurus Techzone, Electronics Technology, Park Phas… | India | train_source1.tsv |
| S1-556372777 | Nizar Marine Pvt Ltd | Opp.Suratcitygymkhana, Piplod, Kohinoor House", Opp. Surat City Gymkhana, Piplod, Nizar, … | India | train_source1.tsv |
| S1-179567829 | "ehpad Club SAS | 4 Rue Daurat, Saint-Nazaire, Pays de la Loire | France | test_source1.tsv |
| S1-822704836 | "lieu Comite SARL | 6 RUE Arnaud Miqueu, Bordeaux, Nouvelle-Aquitaine | France | test_source1.tsv |
| S1-219186286 | Fédération du "ehpad | 13 Rue Albert Sauvage, Dunkerque, Hauts-de-France | France | test_source1.tsv |

## 1.5 ID format

| file | wrong_prefix | non_numeric_suffix | min_id_len | max_id_len |
|---|---|---|---|---|
| train_source1.tsv | 0 | 0 | 6 | 12 |
| train_source2.tsv | 0 | 0 | 6 | 12 |
| train_source3.tsv | 0 | 0 | 5 | 12 |
| test_source1.tsv | 0 | 0 | 6 | 12 |
| test_source2.tsv | 0 | 0 | 6 | 12 |
| test_source3.tsv | 0 | 0 | 4 | 12 |

Train/test ID overlap per source:

| source | train_ids | test_ids | ids_in_both |
|---|---|---|---|
| S1 | 2,206,821 | 1,732,544 | 0 |
| S2 | 5,034,616 | 4,887,273 | 0 |
| S3 | 5,285,603 | 5,082,316 | 0 |

## 1.6 Same content under different IDs

| file | rows_sharing_name_addr_country | rows_sharing_name_country |
|---|---|---|
| train_source1.tsv | 0 | 844,718 |
| train_source2.tsv | 50,933 | 868,414 |
| train_source3.tsv | 37,241 | 886,506 |
| test_source1.tsv | 0 | 623,220 |
| test_source2.tsv | 44,550 | 794,399 |
| test_source3.tsv | 32,154 | 791,077 |

## 1.7 Ground-truth consistency

| check | value |
|---|---|
| GT rows | 2,206,821 |
| GT rows with empty match list | 123,247 |
| GT source1 IDs duplicated | 0 |
| GT source1 IDs missing from train_source1 | 0 |
| train_source1 IDs missing from GT | 0 |
| matched IDs total (pairs) | 7,638,365 |
| matched IDs with S2- prefix | 3,693,619 |
| matched IDs with S3- prefix | 3,944,746 |
| matched IDs with other prefix (incl. S1-) | 0 |
| matched S2 IDs not in train_source2 | 0 |
| matched S3 IDs not in train_source3 | 0 |
| matched IDs found in a TEST S2/S3 file | 0 |
| duplicate IDs inside one match list | 0 |
| match lists with whitespace around IDs | 0 |
| match lists with empty elements (',,' or trailing ',') | 0 |


### Implications for our pipeline
- Parsing is safe with the mandated `read_csv` call. Quoted fields (`""` escapes) parse correctly, so no custom parser is needed.
- The ground truth is clean and complete. We can build a validation split directly from it with no repair step.
- S1 has no empty names or addresses, and train S1 names are 100% ASCII. Nearly all the noise sits on the S2/S3 side.
  So normalisation should target S2/S3 patterns: uppercase text, double spaces, `\x1a` in place of an apostrophe,
  double-encoded dashes (`Â\x80\x93`), and ZWNJ inside Indic words (legitimate; keep it).
- About 3.4% of S2/S3 addresses are empty. The matcher needs a name-only path for these records, with a stricter threshold.
- S2/S3 contain rows with identical name, address and country under different IDs (50,933 in train S2, 37,241 in train S3).
  Because matching is one-to-one (Section 3), these are either several true matches of the same S1 entity or distractors.
  They must not be collapsed before matching.

# 2. Country distribution

## 2.1 Records per country per source

| split | country | S1 | S2 | S3 | total | S1_share | S2_share | S3_share | total_share |
|---|---|---|---|---|---|---|---|---|---|
| test | France | 259,452 | 703,378 | 731,615 | 1,694,445 | 0.1498 | 0.1439 | 0.1440 | 0.1448 |
| test | India | 809,986 | 2,312,565 | 2,405,000 | 5,527,551 | 0.4675 | 0.4732 | 0.4732 | 0.4724 |
| test | US | 663,106 | 1,871,330 | 1,945,701 | 4,480,137 | 0.3827 | 0.3829 | 0.3828 | 0.3828 |
| train | India | 883,188 | 2,017,799 | 2,115,547 | 5,016,534 | 0.4002 | 0.4008 | 0.4002 | 0.4005 |
| train | US | 1,323,633 | 3,016,817 | 3,170,056 | 7,510,506 | 0.5998 | 0.5992 | 0.5998 | 0.5995 |

## 2.2 Raw label spellings (repr shows hidden whitespace)

| split | source | label_repr | n | normalized |
|---|---|---|---|---|
| train | S1 | 'US' | 1,323,633 | us |
| train | S1 | 'India' | 883,188 | india |
| train | S2 | 'US' | 3,016,817 | us |
| train | S2 | 'India' | 2,017,799 | india |
| train | S3 | 'US' | 3,170,056 | us |
| train | S3 | 'India' | 2,115,547 | india |
| test | S1 | 'India' | 809,986 | india |
| test | S1 | 'US' | 663,106 | us |
| test | S1 | 'France' | 259,452 | france |
| test | S2 | 'India' | 2,312,565 | india |
| test | S2 | 'US' | 1,871,330 | us |
| test | S2 | 'France' | 703,378 | france |
| test | S3 | 'India' | 2,405,000 | india |
| test | S3 | 'US' | 1,945,701 | us |
| test | S3 | 'France' | 731,615 | france |

Distinct raw spellings per normalized label (strip+lower):

| normalized | distinct_raw_spellings |
|---|---|
| france | 1 |
| india | 1 |
| us | 1 |

## 2.3 Pool size: S2/S3 records per S1 entity

| split | country | S1 | S2 | S3 | S2_per_S1 | S3_per_S1 | S2+S3_per_S1 |
|---|---|---|---|---|---|---|---|
| test | France | 259,452 | 703,378 | 731,615 | 2.711 | 2.820 | 5.531 |
| test | India | 809,986 | 2,312,565 | 2,405,000 | 2.855 | 2.969 | 5.824 |
| test | US | 663,106 | 1,871,330 | 1,945,701 | 2.822 | 2.934 | 5.756 |
| train | India | 883,188 | 2,017,799 | 2,115,547 | 2.285 | 2.395 | 4.680 |
| train | US | 1,323,633 | 3,016,817 | 3,170,056 | 2.279 | 2.395 | 4.674 |


### Implications for our pipeline
- Country labels are clean and 100% consistent within true pairs (Section 3.5). Country is a safe hard blocking key.
  The code must treat country as an open set of labels (France appears only in test).
- Test has **5.5–5.8 S2+S3 records per S1 entity, versus 4.68 in train**, for every country, US and India included.
  So test contains either more matches per entity or more distractors. Thresholds and any "expected number of matches"
  prior tuned on train may not transfer. This is an open question (Section 9).
- France is 15% of test S1 entities (259,452). If our pipeline scored 0 on France, it would lose about 0.15 of the final
  macro-F0.5.

# 3. Ground truth structure

## 3.1 Matches per S1 entity

| bucket | all | India | US | all_share | India_share | US_share |
|---|---|---|---|---|---|---|
| 0 | 123,247 | 49,351 | 73,896 | 0.0558 | 0.0559 | 0.0558 |
| 1 | 119,157 | 47,468 | 71,689 | 0.0540 | 0.0537 | 0.0542 |
| 2 | 375,212 | 149,927 | 225,285 | 0.1700 | 0.1698 | 0.1702 |
| 3 | 530,841 | 211,965 | 318,876 | 0.2405 | 0.2400 | 0.2409 |
| 4 | 484,115 | 193,669 | 290,446 | 0.2194 | 0.2193 | 0.2194 |
| 5 | 321,957 | 129,339 | 192,618 | 0.1459 | 0.1464 | 0.1455 |
| 6 | 164,868 | 66,462 | 98,406 | 0.0747 | 0.0753 | 0.0743 |
| 7 | 63,968 | 25,669 | 38,299 | 0.0290 | 0.0291 | 0.0289 |
| 8 | 18,680 | 7,463 | 11,217 | 0.0085 | 0.0085 | 0.0085 |
| 9 | 4,205 | 1,671 | 2,534 | 0.0019 | 0.0019 | 0.0019 |
| 10+ | 571 | 204 | 367 | 0.0003 | 0.0002 | 0.0003 |


| country | s1_entities | singletons | total_matches | mean_matches | mean_matches_nonsingleton | max_matches | singleton_rate |
|---|---|---|---|---|---|---|---|
| India | 883,188 | 49,351 | 3,059,843 | 3.4645 | 3.6696 | 11 | 0.0559 |
| US | 1,323,633 | 73,896 | 4,578,522 | 3.4591 | 3.6636 | 11 | 0.0558 |
| ALL | 2,206,821 | 123,247 | 7,638,365 | 3.4613 | 3.6660 | 11 | 0.0558 |

**Singleton rate = 5.58%.** Predicting an empty list for every train S1 entity scores macro-F0.5 = **0.0558** - the baseline floor.

## 3.2 Source mix of matches

| mix | India | US | All |
|---|---|---|---|
| S2 only | 57,202 | 85,827 | 143,029 |
| S2+S3 | 710,985 | 1,065,062 | 1,776,047 |
| S3 only | 65,650 | 98,848 | 164,498 |
| none | 49,351 | 73,896 | 123,247 |
| All | 883,188 | 1,323,633 | 2,206,821 |


| stat | value |
|---|---|
| S1 with >=2 matches from S2 | 1,129,968 |
| S1 with >=2 matches from S3 | 1,224,128 |
| max S2 matches for one S1 | 5 |
| max S3 matches for one S1 | 6 |

Joint distribution of #S2 matches (rows) × #S3 matches (cols), capped at 6:

| n_S2 \ n_S3 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 0 | 123,247 | 60,407 | 57,041 | 31,526 | 12,260 | 3,017 | 247 |
| 1 | 58,750 | 269,681 | 251,224 | 140,393 | 54,575 | 13,437 | 1,048 |
| 2 | 48,490 | 223,054 | 208,159 | 116,076 | 45,167 | 10,957 | 876 |
| 3 | 25,037 | 114,405 | 105,817 | 59,427 | 23,315 | 5,504 | 452 |
| 4 | 8,898 | 40,618 | 38,338 | 20,852 | 8,131 | 2,085 | 156 |
| 5 | 1,854 | 8,252 | 7,796 | 4,169 | 1,668 | 378 | 37 |

## 3.3 One-to-one hypothesis (does an S2/S3 record belong to >1 S1 entity?)

| stat | value |
|---|---|
| distinct matched S2/S3 IDs | 7,638,365 |
| IDs appearing in >1 S1 list | 0 |
| max S1 lists per ID | 1 |

## 3.4 Distractors: S2/S3 records that match no S1 entity

| src | country | records | matched | unmatched | distractor_rate |
|---|---|---|---|---|---|
| S2 | India | 2,017,799 | 1,480,545 | 537,254 | 0.2663 |
| S2 | US | 3,016,817 | 2,213,074 | 803,743 | 0.2664 |
| S3 | India | 2,115,547 | 1,579,298 | 536,249 | 0.2535 |
| S3 | US | 3,170,056 | 2,365,448 | 804,608 | 0.2538 |
| S2 | ALL | 5,034,616 | 3,693,619 | 1,340,997 | 0.2664 |
| S3 | ALL | 5,285,603 | 3,944,746 | 1,340,857 | 0.2537 |

## 3.5 Country label agreement on matched pairs

| c1 | India | US |
|---|---|---|
| India | 3,059,843 | 0 |
| US | 0 | 4,578,522 |

Pairs with different country labels: 0


## 3.6 F0.5 reference points (computed on train GT)

| strategy (oracle-style reference) | macro_F0.5 |
|---|---|
| predict empty for everyone (floor) | 0.0558 |
| perfect on singletons, 1 correct match per non-singleton | 0.6964 |
| perfect on singletons, all correct matches but +1 false match each non-singleton | 0.8074 |
| perfect on non-singletons, 1 false match on 5% of singletons | 0.9972 |


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

# 4. Name analysis

True-pair statistics (4.9-4.10) use a random sample of 1,000,000 of the 7,638,365 train pairs (seed 42).

Length/casing/script stats use all rows; token and suffix frequencies use a random sample of up to 200,000 names per (split, source, country) group (seed 42).

## 4.1 Name length (characters)

| split | source | country | n | unit | mean | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | chars | 26.4 | 13.0 | 21.0 | 27.0 | 32.0 | 38.0 | 105 |
| train | S1 | US | 1,323,633 | chars | 22.5 | 11.0 | 17.0 | 22.0 | 27.0 | 35.0 | 68 |
| train | S2 | India | 2,017,799 | chars | 27.4 | 13.0 | 21.0 | 27.0 | 33.0 | 42.0 | 104 |
| train | S2 | US | 3,016,817 | chars | 23.6 | 11.0 | 17.0 | 23.0 | 29.0 | 39.0 | 87 |
| train | S3 | India | 2,115,547 | chars | 27.0 | 12.0 | 20.0 | 27.0 | 33.0 | 43.0 | 123 |
| train | S3 | US | 3,170,056 | chars | 24.0 | 11.0 | 17.0 | 23.0 | 30.0 | 40.0 | 88 |
| test | S1 | France | 259,452 | chars | 19.4 | 12.0 | 16.0 | 19.0 | 23.0 | 29.0 | 50 |
| test | S1 | India | 809,986 | chars | 26.4 | 13.0 | 21.0 | 27.0 | 32.0 | 38.0 | 92 |
| test | S1 | US | 663,106 | chars | 22.5 | 11.0 | 17.0 | 22.0 | 27.0 | 35.0 | 66 |
| test | S2 | France | 703,378 | chars | 21.1 | 11.0 | 16.0 | 20.0 | 26.0 | 35.0 | 60 |
| test | S2 | India | 2,312,565 | chars | 28.3 | 14.0 | 22.0 | 28.0 | 34.0 | 44.0 | 102 |
| test | S2 | US | 1,871,330 | chars | 24.3 | 11.0 | 18.0 | 24.0 | 30.0 | 39.0 | 85 |
| test | S3 | France | 731,615 | chars | 21.1 | 10.0 | 16.0 | 20.0 | 26.0 | 36.0 | 69 |
| test | S3 | India | 2,405,000 | chars | 27.9 | 12.0 | 21.0 | 28.0 | 34.0 | 44.0 | 103 |
| test | S3 | US | 1,945,701 | chars | 24.6 | 11.0 | 18.0 | 24.0 | 30.0 | 41.0 | 85 |

## 4.2 Name length (whitespace tokens)

| split | source | country | n | unit | mean | p5 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | tokens | 3.7 | 2.0 | 3.0 | 4.0 | 4.0 | 5.0 | 16 |
| train | S1 | US | 1,323,633 | tokens | 3.4 | 2.0 | 3.0 | 3.0 | 4.0 | 5.0 | 12 |
| train | S2 | India | 2,017,799 | tokens | 3.7 | 2.0 | 3.0 | 4.0 | 4.0 | 5.0 | 15 |
| train | S2 | US | 3,016,817 | tokens | 3.4 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 12 |
| train | S3 | India | 2,115,547 | tokens | 3.7 | 1.0 | 3.0 | 4.0 | 4.0 | 6.0 | 18 |
| train | S3 | US | 3,170,056 | tokens | 3.4 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 | 14 |
| test | S1 | France | 259,452 | tokens | 3.1 | 2.0 | 3.0 | 3.0 | 3.0 | 4.0 | 6 |
| test | S1 | India | 809,986 | tokens | 3.7 | 2.0 | 3.0 | 4.0 | 4.0 | 5.0 | 14 |
| test | S1 | US | 663,106 | tokens | 3.4 | 2.0 | 3.0 | 3.0 | 4.0 | 5.0 | 11 |
| test | S2 | France | 703,378 | tokens | 3.1 | 1.0 | 3.0 | 3.0 | 4.0 | 5.0 | 8 |
| test | S2 | India | 2,312,565 | tokens | 3.8 | 2.0 | 3.0 | 4.0 | 5.0 | 5.0 | 15 |
| test | S2 | US | 1,871,330 | tokens | 3.5 | 1.0 | 3.0 | 3.0 | 4.0 | 6.0 | 12 |
| test | S3 | France | 731,615 | tokens | 3.1 | 1.0 | 3.0 | 3.0 | 4.0 | 5.0 | 10 |
| test | S3 | India | 2,405,000 | tokens | 3.8 | 1.0 | 3.0 | 4.0 | 5.0 | 6.0 | 16 |
| test | S3 | US | 1,945,701 | tokens | 3.5 | 1.0 | 3.0 | 3.0 | 4.0 | 6.0 | 13 |

## 4.3 Script (share of names)
`latin-ext` = Latin with accents/extended chars; `non-latin` = contains a letter above U+024F (Devanagari, Kannada, Tamil, ...).

| split | source | country | n | ascii | latin-ext | non-latin |
|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 1.0000 | 0.0000 | 0.0000 |
| train | S1 | US | 1,323,633 | 1.0000 | 0.0000 | 0.0000 |
| train | S2 | India | 2,017,799 | 0.7215 | 0.0441 | 0.2344 |
| train | S2 | US | 3,016,817 | 0.9319 | 0.0681 | 0.0000 |
| train | S3 | India | 2,115,547 | 0.8157 | 0.0527 | 0.1316 |
| train | S3 | US | 3,170,056 | 0.9326 | 0.0674 | 0.0000 |
| test | S1 | France | 259,452 | 0.8427 | 0.1573 | 0.0000 |
| test | S1 | India | 809,986 | 1.0000 | 0.0000 | 0.0000 |
| test | S1 | US | 663,106 | 1.0000 | 0.0000 | 0.0000 |
| test | S2 | France | 703,378 | 0.7527 | 0.2473 | 0.0000 |
| test | S2 | India | 2,312,565 | 0.7233 | 0.0401 | 0.2366 |
| test | S2 | US | 1,871,330 | 0.9380 | 0.0620 | 0.0000 |
| test | S3 | France | 731,615 | 0.7595 | 0.2405 | 0.0000 |
| test | S3 | India | 2,405,000 | 0.8181 | 0.0486 | 0.1333 |
| test | S3 | US | 1,945,701 | 0.9368 | 0.0632 | 0.0000 |

## 4.4 Casing patterns (ASCII letters only)

| split | source | country | n | Title | UPPER | lower | Mixed | no-ascii-letters |
|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 0.8893 | 0.0000 | 0.0000 | 0.1107 | 0.0000 |
| train | S1 | US | 1,323,633 | 0.5683 | 0.0000 | 0.0000 | 0.4317 | 0.0000 |
| train | S2 | India | 2,017,799 | 0.4344 | 0.1501 | 0.0435 | 0.1468 | 0.2253 |
| train | S2 | US | 3,016,817 | 0.4447 | 0.2146 | 0.0663 | 0.2742 | 0.0000 |
| train | S3 | India | 2,115,547 | 0.6116 | 0.0260 | 0.0524 | 0.1953 | 0.1147 |
| train | S3 | US | 3,170,056 | 0.5475 | 0.0319 | 0.0699 | 0.3506 | 0.0001 |
| test | S1 | France | 259,452 | 0.1192 | 0.0000 | 0.0001 | 0.8807 | 0.0000 |
| test | S1 | India | 809,986 | 0.8876 | 0.0000 | 0.0000 | 0.1124 | 0.0000 |
| test | S1 | US | 663,106 | 0.5687 | 0.0000 | 0.0000 | 0.4313 | 0.0000 |
| test | S2 | France | 703,378 | 0.1878 | 0.2082 | 0.0654 | 0.5386 | 0.0000 |
| test | S2 | India | 2,312,565 | 0.4460 | 0.1419 | 0.0374 | 0.1472 | 0.2276 |
| test | S2 | US | 1,871,330 | 0.4621 | 0.2042 | 0.0605 | 0.2732 | 0.0001 |
| test | S3 | France | 731,615 | 0.2627 | 0.0566 | 0.0661 | 0.6146 | 0.0000 |
| test | S3 | India | 2,405,000 | 0.6210 | 0.0248 | 0.0456 | 0.1919 | 0.1168 |
| test | S3 | US | 1,945,701 | 0.5619 | 0.0327 | 0.0628 | 0.3424 | 0.0001 |

## 4.5 Legal suffix families (share of names containing at least one token of the family)

| split | source | country | n | any_legal_suffix | Co/Company | Corp/Corporation | EURL | Inc/Incorporated | LLC | LLP | LP | Ltd (Devanagari) | Ltd abbr (Devanagari) | Ltd/Limited | PLLC | Pvt (Devanagari) | Pvt abbr (Devanagari) | Pvt/Private | SA | SARL | SAS | SASU | SCI | SNC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 0.8442 | 0.0292 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | 0.0431 | 0.0001 | 0.0000 | 0.0000 | 0.7584 | 0.0000 | 0.0000 | 0.0000 | 0.6294 | 0.0002 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| train | S1 | US | 1,323,633 | 0.5367 | 0.0116 | 0.0263 | 0.0000 | 0.1803 | 0.2692 | 0.0009 | 0.0124 | 0.0000 | 0.0000 | 0.0024 | 0.0154 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| train | S2 | India | 2,017,799 | 0.5710 | 0.0307 | 0.0204 | 0.0000 | 0.0000 | 0.0000 | 0.0305 | 0.0001 | 0.0937 | 0.0176 | 0.4153 | 0.0000 | 0.0781 | 0.0176 | 0.3552 | 0.0002 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| train | S2 | US | 3,016,817 | 0.4709 | 0.0364 | 0.0544 | 0.0000 | 0.1420 | 0.1741 | 0.0004 | 0.0157 | 0.0000 | 0.0000 | 0.0312 | 0.0077 | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| train | S3 | India | 2,115,547 | 0.6406 | 0.0310 | 0.0188 | 0.0000 | 0.0000 | 0.0000 | 0.0361 | 0.0001 | 0.0502 | 0.0097 | 0.4799 | 0.0000 | 0.0418 | 0.0097 | 0.4077 | 0.0002 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| train | S3 | US | 3,170,056 | 0.4635 | 0.0344 | 0.0502 | 0.0000 | 0.1388 | 0.1801 | 0.0005 | 0.0146 | 0.0000 | 0.0000 | 0.0271 | 0.0085 | 0.0000 | 0.0000 | 0.0002 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | S1 | France | 259,452 | 0.6735 | 0.0003 | 0.0001 | 0.0653 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0490 | 0.2839 | 0.2013 | 0.0413 | 0.0324 | 0.0000 |
| test | S1 | India | 809,986 | 0.8417 | 0.0292 | 0.0161 | 0.0000 | 0.0000 | 0.0000 | 0.0442 | 0.0001 | 0.0000 | 0.0000 | 0.7542 | 0.0000 | 0.0000 | 0.0000 | 0.6257 | 0.0001 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| test | S1 | US | 663,106 | 0.5336 | 0.0112 | 0.0267 | 0.0000 | 0.1785 | 0.2682 | 0.0008 | 0.0127 | 0.0000 | 0.0000 | 0.0024 | 0.0153 | 0.0000 | 0.0000 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | S2 | France | 703,378 | 0.5442 | 0.0003 | 0.0001 | 0.0593 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0486 | 0.1989 | 0.1417 | 0.0450 | 0.0391 | 0.0106 |
| test | S2 | India | 2,312,565 | 0.5826 | 0.0297 | 0.0202 | 0.0000 | 0.0000 | 0.0000 | 0.0357 | 0.0001 | 0.0949 | 0.0179 | 0.4264 | 0.0000 | 0.0785 | 0.0179 | 0.3540 | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| test | S2 | US | 1,871,330 | 0.4913 | 0.0456 | 0.0635 | 0.0000 | 0.1408 | 0.1695 | 0.0003 | 0.0150 | 0.0000 | 0.0000 | 0.0410 | 0.0070 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | S3 | France | 731,615 | 0.5343 | 0.0015 | 0.0000 | 0.0570 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0477 | 0.1973 | 0.1386 | 0.0438 | 0.0384 | 0.0097 |
| test | S3 | India | 2,405,000 | 0.6555 | 0.0308 | 0.0190 | 0.0000 | 0.0000 | 0.0000 | 0.0419 | 0.0001 | 0.0503 | 0.0093 | 0.4918 | 0.0000 | 0.0417 | 0.0093 | 0.4079 | 0.0001 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| test | S3 | US | 1,945,701 | 0.4852 | 0.0448 | 0.0573 | 0.0000 | 0.1398 | 0.1753 | 0.0005 | 0.0136 | 0.0000 | 0.0000 | 0.0377 | 0.0078 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 4.6 '&' vs 'and'

| split | source | country | n | has_& | has_' and ' | has_'et' (fr) |
|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 0.0429 | 0.0000 | 0.0000 |
| train | S1 | US | 1,323,633 | 0.0558 | 0.0432 | 0.0000 |
| train | S2 | India | 2,017,799 | 0.0343 | 0.0012 | 0.0000 |
| train | S2 | US | 3,016,817 | 0.0470 | 0.0356 | 0.0000 |
| train | S3 | India | 2,115,547 | 0.0329 | 0.0010 | 0.0000 |
| train | S3 | US | 3,170,056 | 0.0459 | 0.0353 | 0.0000 |
| test | S1 | France | 259,452 | 0.0592 | 0.0000 | 0.0000 |
| test | S1 | India | 809,986 | 0.0420 | 0.0000 | 0.0000 |
| test | S1 | US | 663,106 | 0.0559 | 0.0426 | 0.0000 |
| test | S2 | France | 703,378 | 0.0587 | 0.0000 | 0.0104 |
| test | S2 | India | 2,312,565 | 0.0336 | 0.0011 | 0.0000 |
| test | S2 | US | 1,871,330 | 0.0482 | 0.0367 | 0.0000 |
| test | S3 | France | 731,615 | 0.0563 | 0.0000 | 0.0100 |
| test | S3 | India | 2,405,000 | 0.0332 | 0.0010 | 0.0000 |
| test | S3 | US | 1,945,701 | 0.0464 | 0.0353 | 0.0000 |

## 4.7 Other name noise

| split | source | country | n | starts_non_alnum | @handle | domain_like(.com/.in/.fr...) | has_digit | has_dba/aka | has_parenthesis | empty |
|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 0.0014 | 0.0000 | 0.0000 | 0.0013 | 0.0000 | 0.0518 | 0.0000 |
| train | S1 | US | 1,323,633 | 0.0015 | 0.0004 | 0.0000 | 0.0262 | 0.0000 | 0.0000 | 0.0000 |
| train | S2 | India | 2,017,799 | 0.0167 | 0.0023 | 0.0337 | 0.0307 | 0.0000 | 0.0724 | 0.0000 |
| train | S2 | US | 3,016,817 | 0.0248 | 0.0037 | 0.0436 | 0.0639 | 0.0000 | 0.0350 | 0.0000 |
| train | S3 | India | 2,115,547 | 0.0179 | 0.0025 | 0.0367 | 0.0340 | 0.0098 | 0.0744 | 0.0000 |
| train | S3 | US | 3,170,056 | 0.0238 | 0.0038 | 0.0424 | 0.0628 | 0.0136 | 0.0350 | 0.0000 |
| test | S1 | France | 259,452 | 0.0004 | 0.0000 | 0.0000 | 0.0078 | 0.0000 | 0.0803 | 0.0000 |
| test | S1 | India | 809,986 | 0.0015 | 0.0000 | 0.0000 | 0.0013 | 0.0000 | 0.0506 | 0.0000 |
| test | S1 | US | 663,106 | 0.0013 | 0.0003 | 0.0000 | 0.0259 | 0.0000 | 0.0000 | 0.0000 |
| test | S2 | France | 703,378 | 0.0090 | 0.0029 | 0.0347 | 0.0111 | 0.0000 | 0.0872 | 0.0000 |
| test | S2 | India | 2,312,565 | 0.0154 | 0.0019 | 0.0271 | 0.0280 | 0.0000 | 0.0718 | 0.0000 |
| test | S2 | US | 1,871,330 | 0.0227 | 0.0029 | 0.0359 | 0.0616 | 0.0000 | 0.0334 | 0.0000 |
| test | S3 | France | 731,615 | 0.0083 | 0.0027 | 0.0344 | 0.0112 | 0.0080 | 0.0842 | 0.0000 |
| test | S3 | India | 2,405,000 | 0.0159 | 0.0020 | 0.0294 | 0.0316 | 0.0078 | 0.0751 | 0.0000 |
| test | S3 | US | 1,945,701 | 0.0220 | 0.0029 | 0.0349 | 0.0605 | 0.0107 | 0.0349 | 0.0000 |

## 4.8 Most frequent tokens (norm_basic tokens)

| train S1 India | train S1 US | train S2 India | train S2 US |
|---|---|---|---|
| limited (118,565) | llc (53,839) | limited (53,132) | llc (35,080) |
| private (98,335) | inc (36,060) | private (52,770) | inc (26,933) |
| ltd (33,114) | and (8,640) | ltd (30,478) | l (10,301) |
| pvt (27,576) | s (8,434) | pvt (18,792) | center (9,161) |
| india (13,891) | c (8,195) | लिमिटेड (18,733) | com (8,748) |
| llp (8,613) | l (7,300) | प्राइवेट (15,618) | partners (8,714) |
| services (5,438) | care (6,392) | india (11,316) | corp (8,173) |
| solutions (4,870) | of (5,668) | services (7,380) | c (7,755) |
| trading (4,618) | associates (5,557) | com (6,788) | and (7,703) |
| brothers (4,476) | center (5,081) | llp (6,148) | s (7,375) |
| co (4,363) | group (4,744) | center (5,679) | group (6,949) |
| technologies (3,731) | partners (4,433) | co (4,185) | co (6,603) |
| international (3,632) | p (4,433) | industries (3,928) | ltd (6,186) |
| foundation (3,177) | d (4,286) | enterprises (3,662) | care (5,114) |
| global (3,139) | corp (4,189) | group (3,660) | of (5,027) |
| tech (3,032) | pc (4,185) | brothers (3,531) | services (4,977) |
| industries (2,915) | health (4,080) | प्रा (3,515) | holdings (4,714) |
| consultants (2,782) | clinic (3,211) | लि (3,515) | associates (4,158) |
| enterprises (2,759) | pllc (3,073) | public (3,494) | d (3,459) |
| consultancy (2,710) | medicine (2,701) | ventures (3,449) | health (3,206) |
| ventures (2,698) | lp (2,471) | లిమిటెడ్ (3,254) | lp (3,169) |
| technology (2,684) | pediatric (2,309) | ಲಿಮಿಟೆಡ್ (3,147) | p (2,854) |
| developers (2,659) | dental (2,027) | solutions (3,128) | corporation (2,814) |
| traders (2,605) | family (1,952) | trading (2,968) | the (2,661) |
| marketing (2,515) | global (1,807) | exports (2,940) | clinic (2,443) |


| train S3 India | train S3 US | test S1 France | test S2 France |
|---|---|---|---|
| limited (63,902) | llc (36,281) | sarl (56,772) | sarl (39,787) |
| private (61,022) | inc (26,609) | sas (40,264) | sas (28,341) |
| ltd (32,671) | center (9,336) | club (17,514) | france (19,179) |
| pvt (21,100) | l (9,207) | france (16,068) | s (14,192) |
| india (10,949) | partners (8,659) | de (13,701) | club (13,899) |
| लिमिटेड (10,033) | com (8,501) | eurl (13,050) | eurl (11,859) |
| services (8,464) | corp (7,670) | amicale (11,066) | de (11,550) |
| प्राइवेट (8,361) | and (7,645) | sa (9,809) | sa (9,727) |
| com (7,398) | s (7,547) | ecole (8,476) | amicale (9,118) |
| llp (7,279) | c (7,342) | du (8,338) | sasu (8,992) |
| center (7,013) | group (6,954) | maison (8,317) | a (8,050) |
| co (4,297) | co (6,221) | sasu (8,268) | sci (7,812) |
| enterprises (4,108) | ltd (5,371) | centre (8,091) | groupe (7,578) |
| industries (4,048) | care (5,117) | comite (7,718) | maison (7,099) |
| ventures (3,703) | services (4,986) | union (6,661) | du (6,990) |
| trading (3,659) | of (4,976) | sci (6,485) | com (6,946) |
| group (3,537) | holdings (4,584) | sportive (6,400) | ecole (6,616) |
| solutions (3,505) | a (4,547) | des (6,046) | développement (6,348) |
| public (3,385) | d (4,434) | amis (5,346) | centre (6,297) |
| exports (3,322) | associates (4,152) | primaire (5,219) | comite (6,146) |
| brothers (3,249) | the (3,264) | pharmacie (4,567) | union (5,807) |
| shri (2,802) | health (3,206) | cie (4,387) | fils (5,744) |
| smt (2,794) | p (3,005) | saint (4,332) | des (5,138) |
| dr (2,790) | lp (2,943) | la (4,126) | sportive (5,022) |
| s (2,767) | service (2,598) | ets (4,110) | participations (4,871) |


| test S3 France |
|---|
| sarl (39,470) |
| sas (27,728) |
| france (18,759) |
| club (13,864) |
| s (13,640) |
| de (11,495) |
| eurl (11,404) |
| sa (9,548) |
| a (9,451) |
| amicale (9,131) |
| sasu (8,757) |
| sci (7,686) |
| groupe (7,362) |
| maison (7,017) |
| com (6,882) |
| du (6,861) |
| ecole (6,688) |
| centre (6,373) |
| développement (6,009) |
| comite (6,000) |
| fils (5,726) |
| union (5,660) |
| des (5,004) |
| sportive (4,868) |
| participations (4,785) |


## 4.9 True pairs: how often do names agree?

Cumulative normalizations: `exact` → `casefold` → `norm_basic` (lowercase, punctuation→space) → `norm_name` (+accent fold, '&'→'and', drop legal-suffix tokens) → `norm_name_tokensort` (order-insensitive) → `tokset_subset` (one side's token set contains the other's). `jacc3` = mean char-trigram Jaccard of norm_name.

| s1_country | src | exact | casefold | norm_basic | norm_name | norm_name_tokensort | tokset_subset | jacc3 |
|---|---|---|---|---|---|---|---|---|
| India | S2 | 0.0272 | 0.0641 | 0.1500 | 0.4264 | 0.4371 | 0.5570 | 0.6185 |
| India | S3 | 0.0275 | 0.0665 | 0.1676 | 0.4408 | 0.4504 | 0.6216 | 0.6655 |
| US | S2 | 0.0614 | 0.1416 | 0.2574 | 0.5255 | 0.5606 | 0.7100 | 0.8104 |
| US | S3 | 0.0579 | 0.1307 | 0.2578 | 0.4887 | 0.5221 | 0.7214 | 0.7860 |
| ALL | ALL | 0.0467 | 0.1076 | 0.2181 | 0.4773 | 0.5019 | 0.6655 | 0.7356 |

By script of the S2/S3 name:

| s1_country | script_m | exact | casefold | norm_basic | norm_name | norm_name_tokensort | tokset_subset | jacc3 | pairs |
|---|---|---|---|---|---|---|---|---|---|
| India | ascii | 0.0357 | 0.0854 | 0.2079 | 0.5161 | 0.5285 | 0.7070 | 0.7720 | 306,883 |
| India | latin-ext | 0.0000 | 0.0000 | 0.0000 | 0.7113 | 0.7232 | 0.8668 | 0.8867 | 21,945 |
| India | non-latin | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0097 | 0.0189 | 72,210 |
| US | ascii | 0.0644 | 0.1469 | 0.2783 | 0.4879 | 0.5218 | 0.7033 | 0.7889 | 554,376 |
| US | latin-ext | 0.0000 | 0.0000 | 0.0000 | 0.7378 | 0.7752 | 0.8725 | 0.9081 | 44,586 |

Trigram-Jaccard distribution of true pairs:

| s1_country | count | mean | std | min | 5% | 10% | 25% | 50% | max |
|---|---|---|---|---|---|---|---|---|---|
| India | 401038.0000 | 0.6427 | 0.3876 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.7368 | 1.0000 |
| US | 598962.0000 | 0.7978 | 0.2510 | 0.0000 | 0.3333 | 0.4375 | 0.6129 | 1.0000 | 1.0000 |


| s1_country | pairs | jacc3<0.15 | jacc3<0.3 | latin-only jacc3<0.15 |
|---|---|---|---|---|
| India | 401038.0000 | 0.1952 | 0.2339 | 0.0247 |
| US | 598962.0000 | 0.0191 | 0.0406 | 0.0191 |

## 4.10 Hard examples: true pairs with very different names (Latin script both sides, jacc3 < 0.15, random 15)

| s1_country | src | s1_name | m_name | jacc3 | s1_addr | m_addr |
|---|---|---|---|---|---|---|
| India | S3 | My Agro | Smt My Service | 0.095 | X-25, Okhla Industrial Area Phase-Ii, Delhi, South Delhi, New Delhi | X-25, DL, New Delhi, South Delhi |
| India | S3 | Real Royal Tech Private Limited | Drexveraio | 0.000 | 24/204, Aykara Building Kottaramattom, Pala, Kottayam, Kerala | 24/204, Aykara Building Kottaramattom, Pala, Kottayam, Kerala |
| India | S3 | Dynamic Infra Limited | DI | 0.062 | 122/313 Shastri Nagar, Kanpur, Uttar Pradesh | 122/313 Shastri Nagar, Kanpur, Uttar Pradesh |
| US | S3 | Upper Tri-State Plum Inc. | Círanyla | 0.000 | 2720 Forrest Street, Unit D, Sacramento, CA | Forrest St, Sacramento, California |
| US | S2 | Nariko Adams Digital LLC | Orbivera | 0.000 | 10501 87th Street, Mission, TX | 10501 1/2 87TH ST, MISSION, TX |
| India | S2 | Bawa Management Private Limited | Mr Bawa Private Limited Services | 0.138 | S. No 41/20/2, Flat No .16, Sanket Apt. Rahatni, Pune, Maharashtra | S. NO ##41/20/2, FLAT NO .16, SANKET APT. RAHATNI, PUNE, Maharashtra |
| India | S3 | Mds (India) Mechatronics Private Limited | Onyxgildcalo | 0.000 | Hooghly, 188/2/C, G T Road Baidyabati, West Bengal | Door No C-218 188/2/C, G T Road Baidyabati, Hooghly, WB |
| India | S3 | Tez Store Limited | TS | 0.083 | 371/A, Chakkeri Veettil, Kariyadu Theru, Thalassery, Kannur, Kerala | 371/A, Chakkeri Veettil, Kariyadu Theru, Thalassery, Kannur, Kerala |
| US | S3 | Taylor Bay Australian | Brixavi | 0.000 | AZ, 16654 Aspen Drive, Fountain Hills | 16654. Aspen Dr, Fountain Hills, Arizona |
| US | S2 | Merchant Nexera LLC | Jaxviopyra | 0.038 | 2007 Wake Bridge Drive, Whitsett, NC | WAKE BRIDGE DRIVE, WHITSETT, NC |
| India | S3 | Fantasy Foundation | FF | 0.048 | Ganesh Ward, Tehsil Kareli, Kareli, Narsinghpur, Madhya Pradesh | Kareli, Ganesh Ward, Madhya Pradesh, Tehsil Kareli, Narsinghpur |
| US | S2 | Figgins Bright Logistics LLC | Fayeveo | 0.031 | 111 Grant Street, Sneads Ferry, NC | 111 1/2 GRANT ST, SNEADS FERRY, NC |
| US | S3 | Apex Direct Columbia LLC | Keloonyxavi | 0.000 | 1402 Taylor Road, Knoxville, TN | 1402 Taylor Road, Knoxville, Tennessee |
| US | S3 | Metropolitan Animal Hospital Clinic | Miraecto | 0.024 | 201 Curiel Street, Eloy, AZ | 201C Curiel Street, Eloy, Arizona |
| US | S2 | Ronny Klein Systems LLC | Aviwex | 0.000 | 18258 Symeron Road, Apple Valley, CA | 18258. SYMERON ROAD, APPLE VALLEY, CA |

Non-Latin-script true pairs (random 5):

| s1_country | src | s1_name | m_name |
|---|---|---|---|
| India | S2 | City Blue Exports Private Limited | સિટી બ્લૂ એક્સપોર્ટ્સ પ્રાઇવેટ લિમિટેડ |
| India | S3 | One Power | वन पावर |
| India | S2 | All Developers Pvt Ltd | ऑल डेवलपर्स प्रा. लि. |
| India | S2 | My Logistics Private Limited | மை லாஜிஸ்டிக்ஸ் பிரைவேட் லிமிடெட் |
| India | S2 | Jai Care Pvt Ltd | ಜೈ ಕೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ |


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

# 5. Address analysis

Pattern rates use a random sample of up to 200,000 rows per (split, source, country) group (seed 42); true-pair statistics use all train pairs.

## 5.1 Postal-code and numeric patterns (share of addresses)

US `5digit_any` is dominated by house numbers (e.g. `17560 Ellis Road`); `5digit_not_first_token` and `zip_after_state` are the better ZIP proxies.

| split | source | country | n | empty | 5digit_any | 5digit_not_first_token | zip_after_state (e.g. 'TX 75001') | zip+4 | 6digit_any (India PIN) | 3+3 digit ('560 001') | FR '5digit City' (e.g. '75008 Paris') | PO box | any digit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 0.0000 | 0.0030 | 0.0013 | 0.0000 | 0.0000 | 0.0002 | 0.0035 | 0.0007 | 0.0000 | 0.9128 |
| train | S1 | US | 1,323,633 | 0.0000 | 0.1104 | 0.0158 | 0.0000 | 0.0000 | 0.0013 | 0.0061 | 0.0907 | 0.0000 | 1.0000 |
| train | S2 | India | 2,017,799 | 0.0291 | 0.0113 | 0.0053 | 0.0026 | 0.0000 | 0.0002 | 0.0115 | 0.0035 | 0.0001 | 0.9126 |
| train | S2 | US | 3,016,817 | 0.0370 | 0.1051 | 0.0134 | 0.0000 | 0.0000 | 0.0138 | 0.0052 | 0.0793 | 0.0166 | 0.9021 |
| train | S3 | India | 2,115,547 | 0.0299 | 0.0104 | 0.0049 | 0.0000 | 0.0000 | 0.0002 | 0.0110 | 0.0032 | 0.0000 | 0.9011 |
| train | S3 | US | 3,170,056 | 0.0351 | 0.1049 | 0.0143 | 0.0000 | 0.0000 | 0.0131 | 0.0050 | 0.0787 | 0.0156 | 0.9128 |
| test | S1 | France | 259,452 | 0.0000 | 0.0041 | 0.0036 | 0.0004 | 0.0000 | 0.0001 | 0.0000 | 0.0021 | 0.0000 | 0.9959 |
| test | S1 | India | 809,986 | 0.0000 | 0.0028 | 0.0013 | 0.0000 | 0.0000 | 0.0002 | 0.0035 | 0.0005 | 0.0000 | 0.9128 |
| test | S1 | US | 663,106 | 0.0000 | 0.1091 | 0.0154 | 0.0000 | 0.0000 | 0.0014 | 0.0063 | 0.0884 | 0.0000 | 1.0000 |
| test | S2 | France | 703,378 | 0.0304 | 0.0054 | 0.0028 | 0.0003 | 0.0000 | 0.0002 | 0.0000 | 0.0040 | 0.0000 | 0.9322 |
| test | S2 | India | 2,312,565 | 0.0226 | 0.0109 | 0.0051 | 0.0025 | 0.0000 | 0.0003 | 0.0114 | 0.0033 | 0.0000 | 0.9299 |
| test | S2 | US | 1,871,330 | 0.0295 | 0.1075 | 0.0139 | 0.0000 | 0.0000 | 0.0138 | 0.0050 | 0.0803 | 0.0171 | 0.9202 |
| test | S3 | France | 731,615 | 0.0301 | 0.0051 | 0.0028 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0037 | 0.0000 | 0.9329 |
| test | S3 | India | 2,405,000 | 0.0246 | 0.0100 | 0.0046 | 0.0000 | 0.0000 | 0.0002 | 0.0116 | 0.0031 | 0.0000 | 0.9191 |
| test | S3 | US | 1,945,701 | 0.0284 | 0.1067 | 0.0145 | 0.0000 | 0.0000 | 0.0136 | 0.0052 | 0.0791 | 0.0165 | 0.9296 |

## 5.2 Address structure

| split | source | country | n | mean_components | p50_components | 1_component | starts_with_number | last_comp_is_US_state_code | has_US_state_code_anywhere | non-latin_chars | latin-accented | UPPERCASE | has_## |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 5.658 | 5.000 | 0.000 | 0.331 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| train | S1 | US | 1,323,633 | 3.158 | 3.000 | 0.000 | 0.859 | 0.863 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S2 | India | 2,017,799 | 4.834 | 5.000 | 0.000 | 0.366 | 0.000 | 0.002 | 0.237 | 0.000 | 0.239 | 0.030 |
| train | S2 | US | 3,016,817 | 2.961 | 3.000 | 0.000 | 0.804 | 0.866 | 1.000 | 0.000 | 0.000 | 0.899 | 0.027 |
| train | S3 | India | 2,115,547 | 4.711 | 4.000 | 0.000 | 0.347 | 0.043 | 0.051 | 0.225 | 0.000 | 0.000 | 0.028 |
| train | S3 | US | 3,170,056 | 3.083 | 3.000 | 0.000 | 0.807 | 0.043 | 0.050 | 0.000 | 0.000 | 0.000 | 0.025 |
| test | S1 | France | 259,452 | 3.039 | 3.000 | 0.000 | 0.861 | 0.000 | 0.000 | 0.000 | 0.283 | 0.000 | 0.000 |
| test | S1 | India | 809,986 | 5.666 | 6.000 | 0.000 | 0.330 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| test | S1 | US | 663,106 | 3.157 | 3.000 | 0.000 | 0.860 | 0.863 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | France | 703,378 | 2.607 | 3.000 | 0.000 | 0.766 | 0.000 | 0.000 | 0.000 | 0.241 | 0.291 | 0.000 |
| test | S2 | India | 2,312,565 | 4.871 | 5.000 | 0.000 | 0.377 | 0.000 | 0.001 | 0.238 | 0.000 | 0.240 | 0.031 |
| test | S2 | US | 1,871,330 | 2.985 | 3.000 | 0.000 | 0.816 | 0.867 | 1.000 | 0.000 | 0.000 | 0.905 | 0.027 |
| test | S3 | France | 731,615 | 2.627 | 3.000 | 0.000 | 0.769 | 0.000 | 0.000 | 0.000 | 0.243 | 0.000 | 0.000 |
| test | S3 | India | 2,405,000 | 4.727 | 4.000 | 0.000 | 0.360 | 0.044 | 0.052 | 0.229 | 0.000 | 0.000 | 0.029 |
| test | S3 | US | 1,945,701 | 3.105 | 3.000 | 0.000 | 0.817 | 0.035 | 0.040 | 0.000 | 0.000 | 0.000 | 0.026 |

## 5.3 Most frequent last comma-component (state / city slot)

| train S1 India | train S1 US | train S2 India |
|---|---|---|
| Maharashtra (36,571) | TX (17,543) | Maharashtra (27,324) |
| Delhi (23,352) | NY (13,442) | Delhi (16,942) |
| Uttar Pradesh (14,162) | NC (12,363) | Uttar Pradesh (10,668) |
| Karnataka (13,530) | OH (11,044) | Karnataka (10,387) |
| Tamil Nadu (11,962) | IL (9,983) | महाराष्ट्र (9,474) |
| Gujarat (10,853) | TN (7,651) | Tamil Nadu (9,294) |
| West Bengal (10,847) | VA (7,571) | Gujarat (8,397) |
| Telangana (10,771) | MA (7,398) | West Bengal (8,363) |
| Haryana (6,887) | AZ (7,268) | Telangana (6,570) |
| Kerala (6,701) | IN (6,402) | दिल्ली (5,781) |
| Rajasthan (6,149) | WA (5,538) | Haryana (5,175) |
| Madhya Pradesh (4,250) | MD (5,295) | Rajasthan (4,691) |
| Bihar (4,224) | CA (4,889) | Andhra Pradesh (4,615) |
| Andhra Pradesh (3,330) | AL (4,383) | Kerala (4,191) |
| Orissa (2,617) | WI (4,150) | उत्तर प्रदेश (3,668) |


| train S2 US | train S3 India | train S3 US |
|---|---|---|
| TX (16,889) | MH (25,380) | Texas (16,028) |
| NY (12,820) | DL (16,014) | New York (12,325) |
| NC (11,896) | UP (9,630) | North Carolina (11,125) |
| OH (10,556) | KA (9,261) | Ohio (10,285) |
| IL (9,691) | महाराष्ट्र (8,423) | Illinois (9,153) |
| VA (7,544) | TN (8,419) | Tennessee (7,126) |
| TN (7,501) | GJ (7,672) | Virginia (7,069) |
| AZ (7,143) | WB (7,425) | Massachusetts (6,810) |
| MA (7,124) | TG (5,985) | Arizona (6,777) |
| IN (6,209) | दिल्ली (5,372) | Indiana (5,955) |
| WA (5,198) | HR (4,739) | Washington (5,172) |
| MD (5,154) | RJ (4,348) | Maryland (4,842) |
| CA (4,768) | KL (3,820) | California (4,491) |
| AL (4,158) | उत्तर प्रदेश (3,236) | Alabama (3,998) |
| WI (3,960) | ಕರ್ನಾಟಕ (3,094) | Minnesota (3,752) |


| test S1 France | test S2 France | test S3 France |
|---|---|---|
| Hauts-de-France (67,977) | Hauts-de-France (21,676) | Hauts-de-France (23,541) |
| Nouvelle-Aquitaine (56,786) | Nord (18,460) | Nouvelle-Aquitaine (19,637) |
| Pays de la Loire (48,565) | Gironde (18,292) | Gironde (18,004) |
| Bordeaux (2,192) | Nouvelle-Aquitaine (18,206) | Nord (17,820) |
| Nantes (1,877) | Loire-Atlantique (15,617) | Pays de la Loire (16,946) |
| Lille (1,704) | Pays de la Loire (15,359) | Loire-Atlantique (15,198) |
| Tourcoing (936) | BORDEAUX (9,757) | Bordeaux (10,526) |
| Dunkerque (849) | NANTES (8,557) | Nantes (9,307) |
| Calais (798) | LILLE (7,938) | Lille (8,412) |
| Roubaix (790) | TOURCOING (4,178) | Tourcoing (4,402) |
| Saint-Nazaire (655) | DUNKERQUE (3,895) | Dunkerque (4,353) |
| La Teste-de-Buch (614) | ROUBAIX (3,799) | Roubaix (4,007) |
| Pessac (613) | CALAIS (3,578) | Calais (3,896) |
| Mérignac (518) | Pas-de-Calais (3,501) | Pas-de-Calais (3,295) |
| Lège-Cap-Ferret (430) | PESSAC (2,957) | Pessac (3,220) |


## 5.4 Landmark phrases (share of addresses)

| split | source | country | n | near | nr | opp/opposite | behind | beside | next to | in front of | c/o | s/o|d/o|w/o | landmark | ke pass/ke paas | chez/près de (fr) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883,188 | 0.0649 | 0.0224 | 0.0390 | 0.0096 | 0.0021 | 0.0019 | 0.0006 | 0.0579 | 0.0071 | 0.0022 | 0.0002 | 0.0000 |
| train | S1 | US | 1,323,633 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |
| train | S2 | India | 2,017,799 | 0.0548 | 0.0186 | 0.0340 | 0.0080 | 0.0019 | 0.0016 | 0.0004 | 0.0559 | 0.0062 | 0.0020 | 0.0002 | 0.0000 |
| train | S2 | US | 3,016,817 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |
| train | S3 | India | 2,115,547 | 0.0424 | 0.0142 | 0.0251 | 0.0063 | 0.0014 | 0.0014 | 0.0004 | 0.0553 | 0.0059 | 0.0018 | 0.0001 | 0.0000 |
| train | S3 | US | 3,170,056 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |
| test | S1 | France | 259,452 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| test | S1 | India | 809,986 | 0.0642 | 0.0220 | 0.0397 | 0.0096 | 0.0023 | 0.0019 | 0.0006 | 0.0572 | 0.0073 | 0.0023 | 0.0002 | 0.0000 |
| test | S1 | US | 663,106 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | S2 | France | 703,378 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 |
| test | S2 | India | 2,312,565 | 0.0555 | 0.0192 | 0.0344 | 0.0077 | 0.0019 | 0.0016 | 0.0005 | 0.0565 | 0.0066 | 0.0021 | 0.0002 | 0.0000 |
| test | S2 | US | 1,871,330 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | S3 | France | 731,615 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 |
| test | S3 | India | 2,405,000 | 0.0413 | 0.0147 | 0.0255 | 0.0061 | 0.0013 | 0.0012 | 0.0004 | 0.0556 | 0.0065 | 0.0015 | 0.0002 | 0.0000 |
| test | S3 | US | 1,945,701 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |

## 5.5 Abbreviation vs long form (share of addresses containing each form)

| index | train S1 India | train S1 US | train S2 India | train S2 US | train S3 India | train S3 US | test S1 France | test S1 India | test S1 US | test S2 France | test S2 India | test S2 US | test S3 France | test S3 India | test S3 US |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Road: short | 0.0181 | 0.0029 | 0.0154 | 0.0995 | 0.0121 | 0.0945 | 0.0004 | 0.0173 | 0.0028 | 0.0003 | 0.0154 | 0.1012 | 0.0004 | 0.0125 | 0.0964 |
| Road: long | 0.2442 | 0.1867 | 0.2089 | 0.0809 | 0.1630 | 0.0868 | 0.0000 | 0.2451 | 0.1874 | 0.0000 | 0.2123 | 0.0831 | 0.0000 | 0.1635 | 0.0873 |
| Street: short | 0.0078 | 0.0069 | 0.0075 | 0.1069 | 0.0055 | 0.1005 | 0.0005 | 0.0080 | 0.0069 | 0.0181 | 0.0068 | 0.1061 | 0.0173 | 0.0055 | 0.1017 |
| Street: long | 0.0405 | 0.2064 | 0.0343 | 0.0883 | 0.0274 | 0.0946 | 0.0000 | 0.0412 | 0.2057 | 0.0000 | 0.0352 | 0.0888 | 0.0000 | 0.0274 | 0.0934 |
| Avenue: short | 0.0003 | 0.0022 | 0.0002 | 0.0737 | 0.0001 | 0.0711 | 0.0085 | 0.0002 | 0.0022 | 0.0546 | 0.0002 | 0.0740 | 0.0525 | 0.0002 | 0.0719 |
| Avenue: long | 0.0094 | 0.1371 | 0.0079 | 0.0595 | 0.0063 | 0.0628 | 0.1195 | 0.0092 | 0.1359 | 0.0655 | 0.0077 | 0.0598 | 0.0673 | 0.0061 | 0.0621 |
| Boulevard: short | 0.0000 | 0.0002 | 0.0000 | 0.0086 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0002 | 0.0042 | 0.0000 | 0.0087 | 0.0042 | 0.0000 | 0.0082 |
| Boulevard: long | 0.0003 | 0.0162 | 0.0003 | 0.0069 | 0.0002 | 0.0072 | 0.0358 | 0.0002 | 0.0162 | 0.0199 | 0.0001 | 0.0070 | 0.0203 | 0.0001 | 0.0073 |
| Drive: short | 0.0063 | 0.0021 | 0.0055 | 0.0900 | 0.0042 | 0.0865 | 0.0001 | 0.0061 | 0.0022 | 0.0001 | 0.0053 | 0.0904 | 0.0001 | 0.0042 | 0.0870 |
| Drive: long | 0.0007 | 0.1681 | 0.0006 | 0.0710 | 0.0005 | 0.0752 | 0.0000 | 0.0007 | 0.1678 | 0.0000 | 0.0006 | 0.0714 | 0.0000 | 0.0006 | 0.0761 |
| Lane: short | 0.0002 | 0.0008 | 0.0002 | 0.0384 | 0.0001 | 0.0364 | 0.0000 | 0.0002 | 0.0009 | 0.0000 | 0.0002 | 0.0387 | 0.0000 | 0.0002 | 0.0369 |
| Lane: long | 0.0147 | 0.0711 | 0.0129 | 0.0304 | 0.0106 | 0.0324 | 0.0000 | 0.0151 | 0.0718 | 0.0000 | 0.0137 | 0.0312 | 0.0000 | 0.0103 | 0.0332 |
| Court: short | 0.0004 | 0.0153 | 0.0004 | 0.0362 | 0.0003 | 0.0228 | 0.0000 | 0.0004 | 0.0148 | 0.0000 | 0.0003 | 0.0369 | 0.0000 | 0.0003 | 0.0229 |
| Court: long | 0.0033 | 0.0430 | 0.0025 | 0.0192 | 0.0020 | 0.0206 | 0.0001 | 0.0030 | 0.0434 | 0.0000 | 0.0028 | 0.0195 | 0.0001 | 0.0021 | 0.0207 |
| Circle: short | 0.0001 | 0.0003 | 0.0001 | 0.0108 | 0.0001 | 0.0111 | 0.0000 | 0.0001 | 0.0003 | 0.0000 | 0.0000 | 0.0111 | 0.0000 | 0.0001 | 0.0110 |
| Circle: long | 0.0040 | 0.0214 | 0.0035 | 0.0091 | 0.0027 | 0.0098 | 0.0000 | 0.0037 | 0.0208 | 0.0000 | 0.0033 | 0.0095 | 0.0000 | 0.0026 | 0.0101 |
| Highway: long | 0.0054 | 0.0083 | 0.0050 | 0.0053 | 0.0035 | 0.0054 | 0.0000 | 0.0049 | 0.0086 | 0.0000 | 0.0044 | 0.0055 | 0.0000 | 0.0033 | 0.0057 |
| Suite/Unit/Apt: short | 0.0044 | 0.0144 | 0.0036 | 0.0000 | 0.0029 | 0.0107 | 0.0014 | 0.0038 | 0.0141 | 0.0014 | 0.0035 | 0.0000 | 0.0014 | 0.0032 | 0.0110 |
| Suite/Unit/Apt: long | 0.0275 | 0.1361 | 0.0245 | 0.0000 | 0.0205 | 0.0617 | 0.0000 | 0.0268 | 0.1350 | 0.0000 | 0.0235 | 0.0000 | 0.0000 | 0.0201 | 0.0623 |
| Floor: short | 0.0166 | 0.0182 | 0.0144 | 0.0005 | 0.0128 | 0.0101 | 0.0000 | 0.0163 | 0.0178 | 0.0000 | 0.0151 | 0.0004 | 0.0000 | 0.0125 | 0.0097 |
| Floor: long | 0.1824 | 0.0011 | 0.1587 | 0.0000 | 0.1241 | 0.0041 | 0.0000 | 0.1822 | 0.0011 | 0.0000 | 0.1589 | 0.0000 | 0.0000 | 0.1259 | 0.0042 |
| Nagar: long | 0.1767 | 0.0000 | 0.1417 | 0.0000 | 0.1275 | 0.0000 | 0.0000 | 0.1756 | 0.0000 | 0.0000 | 0.1417 | 0.0000 | 0.0000 | 0.1263 | 0.0000 |
| Marg: long | 0.0222 | 0.0000 | 0.0190 | 0.0000 | 0.0149 | 0.0000 | 0.0000 | 0.0218 | 0.0000 | 0.0000 | 0.0193 | 0.0000 | 0.0000 | 0.0146 | 0.0000 |
| Colony: long | 0.0559 | 0.0004 | 0.0472 | 0.0004 | 0.0375 | 0.0004 | 0.0000 | 0.0556 | 0.0004 | 0.0000 | 0.0488 | 0.0004 | 0.0000 | 0.0373 | 0.0004 |
| Sector: short | 0.0091 | 0.0000 | 0.0078 | 0.0000 | 0.0065 | 0.0000 | 0.0001 | 0.0094 | 0.0000 | 0.0001 | 0.0083 | 0.0000 | 0.0001 | 0.0063 | 0.0000 |
| Sector: long | 0.0567 | 0.0000 | 0.0472 | 0.0000 | 0.0362 | 0.0000 | 0.0000 | 0.0553 | 0.0000 | 0.0000 | 0.0473 | 0.0000 | 0.0000 | 0.0359 | 0.0000 |
| Plot: long | 0.0994 | 0.0000 | 0.0872 | 0.0000 | 0.0780 | 0.0000 | 0.0000 | 0.0986 | 0.0000 | 0.0000 | 0.0880 | 0.0000 | 0.0000 | 0.0771 | 0.0000 |
| House no: short | 0.0405 | 0.0000 | 0.0766 | 0.0000 | 0.0723 | 0.0000 | 0.0000 | 0.0403 | 0.0000 | 0.0000 | 0.0784 | 0.0000 | 0.0000 | 0.0757 | 0.0000 |
| House no: long | 0.0392 | 0.0010 | 0.0360 | 0.0009 | 0.0315 | 0.0008 | 0.0000 | 0.0394 | 0.0011 | 0.0000 | 0.0360 | 0.0009 | 0.0000 | 0.0317 | 0.0011 |
| Rue (fr): short | 0.0207 | 0.0006 | 0.0186 | 0.0004 | 0.0153 | 0.0005 | 0.0002 | 0.0205 | 0.0006 | 0.2531 | 0.0185 | 0.0005 | 0.2430 | 0.0147 | 0.0006 |
| Rue (fr): long | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 0.6580 | 0.0000 | 0.0001 | 0.3826 | 0.0000 | 0.0001 | 0.3937 | 0.0000 | 0.0001 |
| Avenue (fr av.): short | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0001 | 0.0000 | 0.0392 | 0.0001 | 0.0000 | 0.0376 | 0.0001 | 0.0000 |
| Avenue (fr av.): long | 0.0094 | 0.1371 | 0.0079 | 0.0595 | 0.0063 | 0.0628 | 0.1195 | 0.0092 | 0.1359 | 0.0655 | 0.0077 | 0.0598 | 0.0673 | 0.0061 | 0.0621 |
| Boulevard (fr bd): short | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 0.0078 | 0.0001 | 0.0000 | 0.0171 | 0.0002 | 0.0000 | 0.0158 | 0.0001 | 0.0000 |
| Boulevard (fr bd): long | 0.0003 | 0.0162 | 0.0003 | 0.0069 | 0.0002 | 0.0072 | 0.0358 | 0.0002 | 0.0162 | 0.0199 | 0.0001 | 0.0070 | 0.0203 | 0.0001 | 0.0073 |
| Place (fr pl): short | 0.0043 | 0.0003 | 0.0043 | 0.0104 | 0.0039 | 0.0097 | 0.0037 | 0.0046 | 0.0003 | 0.0083 | 0.0039 | 0.0108 | 0.0080 | 0.0038 | 0.0105 |
| Place (fr pl): long | 0.0062 | 0.0199 | 0.0056 | 0.0091 | 0.0045 | 0.0097 | 0.0127 | 0.0064 | 0.0205 | 0.0068 | 0.0055 | 0.0086 | 0.0076 | 0.0043 | 0.0094 |
| Chemin (fr ch): long | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0141 | 0.0000 | 0.0000 | 0.0077 | 0.0000 | 0.0000 | 0.0081 | 0.0000 | 0.0000 |

## 5.6 True pairs: postal code agreement (country-specific extraction; India 6-digit, US/FR 5-digit not leading)

| s1_country | src | both missing | conflict | match | one missing | pairs |
|---|---|---|---|---|---|---|
| India | S2 | 0.9998 | 0.0000 | 0.0001 | 0.0001 | 194,589 |
| India | S3 | 0.9998 | 0.0000 | 0.0001 | 0.0001 | 206,449 |
| US | S2 | 0.9727 | 0.0001 | 0.0017 | 0.0255 | 289,537 |
| US | S3 | 0.9726 | 0.0001 | 0.0022 | 0.0251 | 309,425 |

House/first number agreement (first number in the address, leading zeros stripped):

| s1_country | src | both missing | conflict | match | one missing | pairs |
|---|---|---|---|---|---|---|
| India | S2 | 0.0737 | 0.2085 | 0.6585 | 0.0592 | 194,589 |
| India | S3 | 0.0742 | 0.2099 | 0.6456 | 0.0704 | 206,449 |
| US | S2 | 0.0001 | 0.1219 | 0.7470 | 0.1309 | 289,537 |
| US | S3 | 0.0002 | 0.1283 | 0.7555 | 0.1160 | 309,425 |

Address emptiness on true pairs:

| s1_country | src | S2/S3 empty | both present | pairs |
|---|---|---|---|---|
| India | S2 | 0.0384 | 0.9616 | 194,589 |
| India | S3 | 0.0399 | 0.9601 | 206,449 |
| US | S2 | 0.0483 | 0.9517 | 289,537 |
| US | S3 | 0.0463 | 0.9537 | 309,425 |

Address similarity on true pairs (norm_basic + accent fold):

| s1_country | src | addr_exact_norm | addr_tokenset_equal | addr_jacc3 | tok_jacc |
|---|---|---|---|---|---|
| India | S2 | 0.1120 | 0.1930 | 0.7712 | 0.7728 |
| India | S3 | 0.0419 | 0.0644 | 0.6173 | 0.6140 |
| US | S2 | 0.1330 | 0.1902 | 0.7177 | 0.6706 |
| US | S3 | 0.0441 | 0.0637 | 0.6026 | 0.5005 |


| s1_country | jacc3<0.2 | jacc3<0.4 | median |
|---|---|---|---|
| India | 0.0409 | 0.1513 | 0.7500 |
| US | 0.0045 | 0.0784 | 0.6522 |

## 5.7 True pairs with very different addresses (both present, non-Latin excluded, jacc3 < 0.2, random 15)

| s1_country | src | s1_name | m_name | s1_addr | m_addr | addr_jacc3 |
|---|---|---|---|---|---|---|
| India | S3 | Silver High Properties | सिल्वर हाई प्रॉपर्टीज | A/704, Deeraj Gaurav H Ii, New Link Road, Andheri W, Mumbai, Maharashtra | A/404, Bombay, Mumbai Suburban, MH | 0.100 |
| India | S3 | Smart (India) Exim Private Limited | Smart Limited Exim Private-(India) | Wework Chromium, Cts No. 106, 106/1-5, Jogeshwari-Vikhroli Link Road (Jvlr), Ne… | Hd-121, Mumbai, MH | 0.109 |
| India | S3 | New Delhi Center Private Limited | New De1hi Center Private Limited | E-12, 1St Floor, Green Park Main, Green Park Colony, Safdarjung Enclave Ward, N… | Door No 864 E-12, null, New Delhi, Bahadurgarh, DL | 0.143 |
| India | S3 | Sanskruti Industries Pvt Ltd | Pvt Sanskruti Índustries Ltd | 1123, 2Nd Floor, Gajmukh Azad Nagar 3 Veera Desai Road, Andheri (West), Mumbai,… | 01123, Mumbai, Mumbai City, MH | 0.170 |
| India | S3 | Fortune Digital Management Private Limited | ফরচুন ডিজিটাল ম্যানেজমেন্ট প্রাইভেট লিমিটেড | 134/1, Mahatma Gandhi Road 2Nd Floor, Room No.40, Kolkata, Howrah, West Bengal | N/A, 134/1, WB, Calcutta | 0.067 |
| India | S3 | Sarthi Trading | sarthitrading.com | #153, 2Nd Floor, Jkb Complex Kammanahalli Circle, Bangalore, Karnataka | H.no 15, Bangalore, KA | 0.178 |
| India | S3 | Skm Civiltech Pvt Ltd | Skm Civiltech [Pvt] | 201, Devki Apartments, E-27/28, Saket, Indore, Madhya Pradesh | Indore, MP, Indore, 201 | 0.180 |
| India | S2 | Black Future Food Private Limited | PRIVATE BLACK FUTURE FOOD LIMITED | 91, Shri Ram Enclave, Vatika Road, Sanganer, Shyosinghpura Urf Kallawala, Bagru… | JAIPUR, Rajasthan, 91 | 0.196 |
| India | S3 | Sai Business LLP | Sai Business | No.2, Mancholi Street, Tneb Colony Kalaiagal Nagar, Ekkattu Thangal, Chennai, T… | No.2, Kanchipuram, Chennai, TN | 0.189 |
| India | S3 | Sms & Co | Sms Co & | Pt No. 16, 17, 18, Flat No.202, Sri Mani Sai Kalyan Arcade, Balaji Nagar, Miyap… | Pt No. C-16, Rangareddi, TG | 0.098 |
| India | S3 | Crescent Tie Ltd | Crescent Tie Ltd | No.766, Poonamallee High Road, Kilpauk, Chennai-10., Tamil Nadu | TN, Door No 66, Chennai-10. | 0.183 |
| India | S3 | Nsr Bpo LLP | Nsr Bpo | No: 6-50, Prem Cottage Payanam, Unnamalaikadai Post, Kanniyakumari District, Ka… | H.no 94 No: 6-50, TN, N/A, Kanyakumari | 0.182 |
| US | S3 | Gilliam Mobile | gilliam mobile | 142 West Street, Boston, MA | Massachusetts, Hyde Park, West St | 0.140 |
| India | S3 | Jain Consultancy Limited | Jain Consultancy | Bangalore, 2Nd Floor, Above Food World, Arikere, Iimb Post, Bannerghatta Road, … | H.no 224, Bangalore, KA | 0.183 |
| India | S3 | Jain Estate Private Limited | जैन एस्टेट प्राइवेट लिमिटेड | P No. 18, A-Sector, Sarvdharam, Bhopal, Madhya Pradesh | Kolar Roadbhopal, P No. 18, MP | 0.182 |


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

# 6. Chains and ambiguity (precision risk)

Keys: `nname` = norm_name(accent-folded name) (lowercase, punctuation stripped, '&'→'and', legal-suffix tokens removed); `hnum` = first number in the address (leading zeros stripped). All joins are within country.

## 6.1 How ambiguous are S1 names?

| country | S1 entities | share whose nname is shared with >=1 other S1 | share whose nname is shared with >=5 other S1 | empty nname |
|---|---|---|---|---|
| India | 883188.0000 | 0.5332 | 0.3578 | 0.0000 |
| US | 1323633.0000 | 0.4625 | 0.2245 | 0.0000 |

Top chain names in S1 (India): same nname on multiple S1 entities with different addresses

| nname | s1_entities | distinct_addresses | singletons | example_address |
|---|---|---|---|---|
| mumbai india | 194 | 194 | 16 | Plot No. 135, Marol Co-Operative Industrial Estate Andheri … |
| new delhi india | 184 | 184 | 6 | Central Delhi, 37/8 E.V Charging, Delhi, Old Rajinder Nagar… |
| shree trading | 178 | 178 | 12 | 6-3-200/B/5/3 Road No. 1, Banjara Hills, Hyderabad, Telanga… |
| new delhi services | 152 | 152 | 6 | Flat No 691, 2Nd Floor Sector-D Pocket-6 Vasant Kunj New De… |
| mumbai services | 146 | 146 | 6 | 5, Ranchhodas Terrace 319, Sir Bhalchandra Road, Matunga, M… |
| shree foundation | 121 | 121 | 6 | Flat No. 002, Tower 15, L&T Rain Tree Boulevard, Airport Ro… |
| mumbai solutions | 118 | 118 | 8 | Office No 302, Floor 3, Plot - 89, Ararat Building Nagindas… |
| new delhi solutions | 117 | 117 | 5 | H.N-Wz 1800-A Kh No.698, 2Nd Floor Basai Darapur, New Delhi… |
| delhi india | 112 | 112 | 8 | B/49A, Rajan Babu Road, Adarsh Nagar, Delhi, East Delhi, De… |
| kolkata india | 110 | 110 | 3 | 91/2, Safuipara, Baidyapata, Arobindo Park P.S. -Kasba, Kol… |
| shree services | 110 | 110 | 11 | Flat No - A / 302, Pune, Sno - 135 / 1, Meadows Habitat Pas… |
| shree solutions | 109 | 109 | 8 | D.No.16-31-43, Flat No.501, Pratima Arcade 6Th Phase Kphb M… |
| shree and | 107 | 107 | 5 | No:17, Duraisamy Colony Mangalinagar, Iind Street, Arumbakk… |
| delhi solutions | 106 | 106 | 4 | Tamil Nadu, Chennai, New No.25, G2, Agasthiyar Street, East… |
| shree traders | 102 | 102 | 9 | Kgp 7/529 Dindukombu, Kanthalloor P O, Devikulam, Idukki, K… |
| shree enterprises | 100 | 100 | 7 | 311, 312Aditya Plaza, Nr Karnavati Flats, Ahmadabad City, A… |
| kolkata services | 98 | 98 | 6 | 8 Fern Place Kolkata, Kolkata, Howrah, West Bengal |
| new trading | 97 | 97 | 9 | 25 Dr. P.V. Cherian Crescent Egmore, Chennai, Tamil Nadu |
| sai trading | 97 | 97 | 5 | S-104, Bhagat Singh Colony, Bhiwadi, Alwar, Rajasthan |
| bangalore india | 96 | 96 | 0 | Block No 16, Flat No 15, Nandi Gardens Apts Phase 2. Alahal… |
| shree consultants | 92 | 92 | 5 | Ba/2 Salt Lake City Sector-1, Kolkata, North 24 Parganas, W… |
| green trading | 91 | 91 | 5 | Uttar Pradesh, K-1494, Gautam Buddha Nagar, Noida, 11Th Ave… |
| shree consultancy | 91 | 91 | 5 | Maharashtra, Pune, S No 41/1A, Pune, Chaudhari Wasti Khardi |
| blue trading | 88 | 88 | 10 | H. No-5-9-470/3/123/P, F No-302, Dream Nest Apt, Tirumalagi… |
| shree and brothers | 88 | 88 | 7 | S/O Savaranna 4-68 Wadepally, Wadepallyvg, Mahbubnagar, Mah… |

Top chain names in S1 (US): same nname on multiple S1 entities with different addresses

| nname | s1_entities | distinct_addresses | singletons | example_address |
|---|---|---|---|---|
| meridian | 556 | 556 | 29 | 4301 Laurel Oak Lane, Muncie, IN |
| cedar | 345 | 345 | 26 | 326 Oak Street, Uhrichsville, OH |
| cascade | 327 | 327 | 16 | 11173 Kinsley Street, Eden Prairie, MN |
| summit | 327 | 327 | 18 | Crossville, TN, 764 Oakmont Drive |
| family center | 326 | 326 | 20 | 840 Norton Road, Berlin, CT |
| helios | 319 | 319 | 18 | 66 Edgewood Street, Bridgeport, CT |
| granite | 309 | 309 | 20 | 524 Barrenwood Drive, Wadsworth City, OH |
| ear nose and throat group | 308 | 308 | 21 | 225 Mcalpine Drive, Fl 0, Saint Louis, MO |
| lynx | 307 | 307 | 14 | 6301 Griffith Loop, Killeen, TX |
| sterling | 306 | 306 | 9 | 3100 Wilshire Drive, Greensboro, NC |
| amber | 305 | 305 | 19 | 4607 198th Avenue, Douglas County, NE |
| sapphire | 304 | 304 | 19 | 21 Hoyt Drive, Conway, AR |
| pinnacle | 302 | 302 | 16 | 1656 Crooked Fork Road, Weston, WV |
| ember | 301 | 301 | 16 | 12005 La Padera Lane, Florissant, MO |
| slate | 299 | 299 | 12 | 1930 Fawn Drive, Cheltenham Township, PA |
| anchor | 298 | 298 | 18 | 201 Clifton Road, Rocky Mount, NC |
| ironclad | 298 | 298 | 12 | 269 Hempstead 177, Fulton, AR |
| cornerstone | 296 | 296 | 17 | 1705 Pitt Street, Greenville, NC |
| stag | 296 | 296 | 25 | 10729 Lafayette Avenue, Shirley, IL |
| cardinal | 294 | 294 | 13 | 17333 Culps Bluff Avenue, St. George City, LA |
| cinder | 294 | 294 | 15 | 3985 Galloway Drive, Memphis, TN |
| citadel | 293 | 293 | 11 | 13647 Partridge Trail, Perry, IA |
| falcon | 293 | 293 | 16 | 8848 8th Street, Unit 16B, Phoenix, AZ |
| primary care group | 293 | 293 | 15 | 8190 Tr 73, Old Fort, OH |
| halcyon | 291 | 291 | 17 | 6600 Capitol Drive, Greenbelt, MD |

Total chain keys (nname on >1 S1 entity with >1 distinct address): India: 69,858 keys covering 470,884 S1 entities, US: 133,629 keys covering 612,199 S1 entities


## 6.2 Exact-key collisions between S1 and S2/S3 (what would a naive exact-name rule do?)

| key | country | S1 entities with >=1 S2/S3 sharing key | ... of which singletons (share of all singletons) | candidate pairs sharing key | true pairs among them | precision of 'match iff key equal' | recall of 'match iff key equal' |
|---|---|---|---|---|---|---|---|
| nname | India | 0.9223 | 0.6069 | 20,978,844 | 1,328,344 | 0.0633 | 0.4341 |
| nname | US | 0.9226 | 0.6576 | 68,906,793 | 2,325,975 | 0.0338 | 0.5080 |
| nname+hnum | India | 0.6745 | 0.1497 | 1,342,181 | 954,936 | 0.7115 | 0.3121 |
| nname+hnum | US | 0.7198 | 0.0163 | 1,807,926 | 1,768,751 | 0.9783 | 0.3863 |


| stat | value |
|---|---|
| S1-vs-S2/S3 same-nname non-matching pairs (chains <50 excluded) | 1,969,365 |
| ... where S1 is a singleton | 130,565 |
| ... where the other record is matched to a different S1 | 1,409,108 |
| ... where the other record is a distractor (matches nobody) | 560,257 |

## 6.3 Likely false merges: non-matching S2/S3 record with identical nname, S1 is a SINGLETON (random 10)

| country | entity_id_s1 | business_name_s1 | business_address_s1 | entity_id_other | business_name_other | business_address_other | is_matched_to_someone |
|---|---|---|---|---|---|---|---|
| US | S1-393026612 | Dermatology Green Partners LLC | 25450 172nd Drive, Surprise, AZ | S3-362794602 | Dermatology Green Partners Inc | Surprise, Arizona, 25463- 172st Dr | 0 |
| US | S1-757856143 | Creative Unified Neuberger Inc | ND, Edinburg, 700 3rd Street | S3-324493896 | Creative Unified Neuberger Corp | 721 Third Street, Edinburg, North Dakota | 0 |
| US | S1-254684349 | Durham Northern | 1936 Debra Drive, Springfield, OR | S3-920445413 | Durham Northern | 156 Asti Ct, Clayton, North Carolina | 1 |
| US | S1-389437534 | Family Integrated Associates | 3424 Zion Lane, Unit 11, Knoxville, TN | S3-439415780 | Family  Integrated Associates Inc | 161 Champagne Boulevard, Unit Unit 6211, Branson, Miss… | 1 |
| US | S1-640781501 | Secure Mountain Aimei, Inc. | 5419 Gainsborough Drive, Fairfax County, VA | S2-639620036 | Secure Mountain Aimei, Inc | 5420 GAINSBOROUGH DR, FAIRFAX, VA | 0 |
| India | S1-574848350 | MP Producer | Cabin No.5, 1St Floor, Sga Complex Swaran Enclave, Nan… | S2-348790466 | MP Producer | PLOT NO. 059 , 1ST FLOOR, ALANKAR SOCIETY, OPP. BANK O… | 1 |
| India | S1-85387642 | KAY Infrastructure Private Limited | Rustam Bagh Main Road, Karnataka, 3Rd Floor, Oxford Ch… | S2-552025654 | KAY  INFRASTRUCTURE PVT LTD | Madhya Pradesh, 4 PALIWAL NAGAR, INDORE | 1 |
| US | S1-69389653 | Sales Pediatrics LLC | 5211 Gallant Fox Way, Unit 116, Charlotte, NC | S3-338785335 | Sales Pediatrics | Houston, 3215- Hurlingham St, Texas | 0 |
| India | S1-691395159 | Technologies Bpr Alloys LLP | Hd-270, We Work Vaswani, Chamber 2 Floor, 264-265, Mum… | S2-990593057 | PVT. TECHNOLOGIES BPR ALLOYS LTD. | SP CENTER, MUMBAI, MUMBAI CITY, Maharashtra | 1 |
| US | S1-190571352 | Paige Storage LLC | 2929 Silver Crest Drive, Mill Creek, WA | S3-442653068 | Paige Stórage Inc | 2931 Silver Crest Drive, Mill Creek, Washington | 0 |

## 6.4 Likely false merges: identical nname, S1 has other true matches, this record is not one of them (random 10)

| country | entity_id_s1 | business_name_s1 | business_address_s1 | entity_id_other | business_name_other | business_address_other | is_matched_to_someone |
|---|---|---|---|---|---|---|---|
| US | S1-930194973 | Carrell Preferred Trg | 11737 Administration Drive, Unit SUITE 2ND, Saint Loui… | S2-524151588 | Carrell Preferred Trg LP | 46 Molisee Rd, MORGANTOWN, WV | 1 |
| US | S1-115316041 | Cam's Empire Family Practice PC | Wilmington, NC, 5526 Dunmore Road | S3-933091939 | Cam's Empire Family Práctice, LLC | #16727 Square Rigger Lane, Friendswood, Texas | 1 |
| US | S1-552025056 | Elliott Integrated Motors Corp | 2202 Ashford Villa Circle, Chattanooga, TN | S2-982299442 | ELLIOTT INTEGRATED MÓTORS CO | 374 MONTCLAIR DR, GADSDEN, AL | 1 |
| India | S1-460071335 | Adarsh (India) Vintrade Ltd | H. No. 68, Block C4C, Pocket -14, Janakpuri, New Delhi… | S2-970138290 | Adarsh (India) Vintrade LLP | H. NO. 69, BLOCK C4C, POCKET -14, JANAKPURI, NEW DELHI… | 0 |
| US | S1-571481781 | Hunter Ampercap LLC | 5026 Sardis Road, Unit H, Charlotte, NC | S2-633760256 | Hunter  Ampercap | 4926  ISLAND VIEW LANE, MUKILETO CDP, WA | 1 |
| US | S1-713032182 | Dynamic Investment Services, PLLC | 2505 B Old Marathon Road, Alpine, TX | S2-632864546 | Dynamic Investment LLC Services | 1435 AMHERST ST, BURKBURNETT, TX | 1 |
| India | S1-552784277 | Sai Biotech (India) Pvt Ltd | Flat No. 405, Fourth Floor, Krishna Plaza Near Iskon T… | S2-103454269 | Sai Biotech (India) LLP | FLAT NO. 416, FOURTH FLOOR, KRISHNA PLAZA NEAR ISKON T… | 0 |
| US | S1-555780772 | Synex Ventures LLC | 1200 Ellison Loop, Fl 1, Merlin, OR | S3-943606123 | Synex Ventures | 113-A Avocet Ct, Mooresville, North Carolina | 1 |
| US | S1-492403978 | Bay Drugs | 6900 Four Peaks Way, Florence, AZ | S2-259284118 | Bay Drugs LLC | 6116 39TH CIRCLE, TOPEKA, KS | 1 |
| US | S1-560505060 | Prairie Diana Inc. | 1304 Norton Street, Unit Apartment 1, Rochester, NY | S3-127427316 | LLC Prairie Diana | 97048 3660 Road, Paden, Oklahoma | 1 |


## 6.5 What separates true matches from look-alikes? (name+address TF-IDF top-10, cosine >= 0.8)

23,796 candidates from 9,868 random train S1 queries; 19,703 true matches, 4,093 non-matches.


`hnum_status` (row-normalised):

| country | label | differs | equal | missing |
|---|---|---|---|---|
| India | non-match | 0.5866 | 0.2866 | 0.1268 |
| India | true match | 0.1871 | 0.7183 | 0.0946 |
| US | non-match | 0.9702 | 0.0282 | 0.0016 |
| US | true match | 0.0827 | 0.8497 | 0.0675 |

`suffix_status` (row-normalised):

| country | label | different family | none on one side | overlap | same family |
|---|---|---|---|---|---|
| India | non-match | 0.0591 | 0.2671 | 0.2112 | 0.4625 |
| India | true match | 0.0002 | 0.3201 | 0.1472 | 0.5324 |
| US | non-match | 0.2248 | 0.6217 | 0.0112 | 0.1422 |
| US | true match | 0.0046 | 0.5967 | 0.0024 | 0.3962 |

|house-number difference| when both present and different:

| country | label | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|---|
| India | non-match | 1300.0 | 129.9 | 346.4 | 1.0 | 4.0 | 11.0 | 60.0 | 5231.0 |
| India | true match | 1581.0 | 445.7 | 1007.0 | 1.0 | 33.0 | 216.0 | 556.0 | 21171.0 |
| US | non-match | 1821.0 | 276.1 | 1498.5 | 1.0 | 3.0 | 7.0 | 13.0 | 24166.0 |
| US | true match | 931.0 | 7737.5 | 147846.8 | 1.0 | 164.0 | 1000.0 | 3598.5 | 4511332.0 |

Core-name token differences (norm_name tokens; >0 means the candidate adds / drops a word):

| country | label | share with extra core name token | share with missing core name token |
|---|---|---|---|
| India | non-match | 0.9030 | 0.4449 |
| India | true match | 0.3604 | 0.2333 |
| US | non-match | 0.5562 | 0.0698 |
| US | true match | 0.1741 | 0.1719 |

Cosine distribution within this band:

| country | label | count | mean | std | min | 10% | 50% | 90% | max |
|---|---|---|---|---|---|---|---|---|---|
| India | non-match | 2216.000 | 0.860 | 0.041 | 0.800 | 0.809 | 0.854 | 0.918 | 0.986 |
| India | true match | 8450.000 | 0.896 | 0.053 | 0.800 | 0.823 | 0.897 | 0.967 | 1.000 |
| US | non-match | 1877.000 | 0.850 | 0.037 | 0.800 | 0.808 | 0.842 | 0.905 | 0.973 |
| US | true match | 11253.000 | 0.888 | 0.055 | 0.800 | 0.817 | 0.883 | 0.966 | 1.000 |


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

# 7. Quick blocking-recall check (char n-gram TF-IDF)

Setup: `char_wb` 3–5-grams on lowercased text, IDF fit on all train S1+S2+S3 (12.5M docs), L2-normalised, cosine brute force against **all 10.3M train S2+S3 records**. Vocabulary is hashed (2^24 buckets) instead of an exact `TfidfVectorizer` vocabulary because the exact fit does not fit in 8 GB RAM; IDF formula and normalisation match sklearn defaults. Queries: 12,000 random train S1 entities (seed 42). `pool=same-country` repeats the ranking with the pool restricted to records whose country label equals the query's.


## 7.1 Recall — name only

| text | pool | country | queries_with_matches | true_pairs | pair_recall@5 | pair_recall@10 | pair_recall@20 | pair_recall@50 | pair_recall@100 | entity_all_found@20 | entity_all_found@50 | entity_all_found@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| name only | all S2+S3 | ALL | 11,348 | 41,340 | 0.4261 | 0.5003 | 0.5581 | 0.6189 | 0.6570 | 0.2673 | 0.3239 | 0.3631 |
| name only | all S2+S3 | India | 4,538 | 16,643 | 0.3824 | 0.4428 | 0.4931 | 0.5493 | 0.5833 | 0.2140 | 0.2618 | 0.3017 |
| name only | all S2+S3 | US | 6,810 | 24,697 | 0.4555 | 0.5391 | 0.6019 | 0.6657 | 0.7066 | 0.3028 | 0.3653 | 0.4041 |
| name only | same-country | ALL | 11,348 | 41,340 | 0.4269 | 0.5026 | 0.5608 | 0.6223 | 0.6606 | 0.2702 | 0.3289 | 0.3687 |
| name only | same-country | India | 4,538 | 16,643 | 0.3844 | 0.4458 | 0.4962 | 0.5538 | 0.5876 | 0.2177 | 0.2686 | 0.3067 |
| name only | same-country | US | 6,810 | 24,697 | 0.4556 | 0.5409 | 0.6044 | 0.6684 | 0.7099 | 0.3051 | 0.3690 | 0.4100 |

## 7.2 Recall — name + address

| text | pool | country | queries_with_matches | true_pairs | pair_recall@5 | pair_recall@10 | pair_recall@20 | pair_recall@50 | pair_recall@100 | entity_all_found@20 | entity_all_found@50 | entity_all_found@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| name + address | all S2+S3 | ALL | 11,348 | 41,340 | 0.8157 | 0.9065 | 0.9286 | 0.9451 | 0.9552 | 0.8091 | 0.8490 | 0.8738 |
| name + address | all S2+S3 | India | 4,538 | 16,643 | 0.7655 | 0.8502 | 0.8775 | 0.8998 | 0.9158 | 0.6948 | 0.7409 | 0.7766 |
| name + address | all S2+S3 | US | 6,810 | 24,697 | 0.8495 | 0.9444 | 0.9630 | 0.9757 | 0.9818 | 0.8853 | 0.9211 | 0.9386 |
| name + address | same-country | ALL | 11,348 | 41,340 | 0.8158 | 0.9067 | 0.9288 | 0.9454 | 0.9554 | 0.8096 | 0.8496 | 0.8744 |
| name + address | same-country | India | 4,538 | 16,643 | 0.7655 | 0.8503 | 0.8775 | 0.9000 | 0.9158 | 0.6948 | 0.7413 | 0.7766 |
| name + address | same-country | US | 6,810 | 24,697 | 0.8498 | 0.9447 | 0.9634 | 0.9759 | 0.9821 | 0.8860 | 0.9217 | 0.9396 |

## 7.3 Top-1 cosine: singletons vs entities with matches (pool = all S2+S3)


name:

| country | singleton | n | top1_score_median | top1_score_p90 | share_top1_is_true_match | share_top1_score_ge_0_9 | best_true_score_median |
|---|---|---|---|---|---|---|---|
| India | 0 | 4,538 | 1.0000 | 1.0000 | 0.5172 | 0.9363 | 1.0000 |
| India | 1 | 261 | 0.9533 | 1.0000 | 0.0000 | 0.6475 |  |
| US | 0 | 6,810 | 1.0000 | 1.0000 | 0.5859 | 0.9498 | 1.0000 |
| US | 1 | 391 | 0.9768 | 1.0000 | 0.0000 | 0.7928 |  |

Non-matching top-1 with cosine >= 0.85 (name; random 10) — likely false merges:

| country | S1 singleton? | S1 name | S1 address | top-1 (non-match) name | top-1 address | cosine |
|---|---|---|---|---|---|---|
| US | 0 | Behavioral Health Pacific Medicine, Inc | TX, 136 Forrest Moon Lane, Kyle | behavioral health pacific medicine, inc | 107 Water Tank Road, N/A, GUIN, AL | 1.000 |
| India | 0 | Trueline Digital Private Limited | 3994/1 Astodia Road, Ahmedabad, Gujarat | Trueline Private Digital Limited | 2-543-1, Tuticorin, தமிழ்நாடு | 1.000 |
| US | 0 | Rocky Initiative | 114 Esplanade, Paragould, AR | rocky initiative | 232 LITTLE BRROK WAY, COLUMBUS, OH | 1.000 |
| US | 1 | Citadel Navios | 4208 Mortimer Avenue, Baltimore, MD | Citadel Navios | 414 Holmes St, La Porte, Texas | 1.000 |
| India | 0 | Arista Urban Ltd | 201, Jai Laxmi Narayan Co, Soc, Lt Road, Mulund, … | Arista Urban | ASWATHY, DEEPA NAGAR NP 5-4/512/B, NELLANAD PANCH… | 0.972 |
| US | 0 | Foot & Ankle Medicine LLC | 3725 Central Park Drive, Moore, OK | FOOT &  ANKLE MEDICINE LLC | 2780 AVERY DRIVE, ROCKALL CDP, TX | 1.000 |
| US | 0 | Northrup Choice Illumination, LLC | Green Valley, 970 Rio Magdalena, AZ | NORTHRUP CHOICE ILLUMINATION, LLC | FUQUAY VARIA, NC, 3110 GOLD RING RD | 1.000 |
| US | 1 | Daffy P. Mueller, DMD | UT, Washington City, 3365 Elkhorn Lane | Daffy P. Mueller, DMD Co | Utah, Wasshington, 3370 Elkhorn Ln | 0.991 |
| India | 0 | Global Construction Private Limited | Delhi, South Delhi, 52/123, New Delhi, Basement, … | private global construction limited | C/O ANU MITRA, KAMAKHYA APARTMENT NEAR MAJHI NRS … | 1.000 |
| India | 0 | Tirupati Technology Private Limited | A - 410, Siddhi Vinayak Tower - A, Behind Dcp Off… | Private Tirupati Technology Limited | FLAT NO-201/B WING S.NO., 772 OPAL ENCLAVE UNTWAD… | 1.000 |

nameaddr:

| country | singleton | n | top1_score_median | top1_score_p90 | share_top1_is_true_match | share_top1_score_ge_0_9 | best_true_score_median |
|---|---|---|---|---|---|---|---|
| India | 0 | 4,538 | 0.9117 | 0.9764 | 0.8561 | 0.5661 | 0.9098 |
| India | 1 | 261 | 0.7807 | 0.9029 | 0.0000 | 0.1226 |  |
| US | 0 | 6,810 | 0.8938 | 0.9764 | 0.8819 | 0.4709 | 0.8904 |
| US | 1 | 391 | 0.7499 | 0.8777 | 0.0000 | 0.0563 |  |

Non-matching top-1 with cosine >= 0.85 (nameaddr; random 10) — likely false merges:

| country | S1 singleton? | S1 name | S1 address | top-1 (non-match) name | top-1 address | cosine |
|---|---|---|---|---|---|---|
| India | 0 | City Infra Limited | Shop No 123 Sector 5Near Subzi Mandi Nit Faridaba… | Haryana Center | Shop No 123 Sector 5Near Subzi Mandi Nit Faridaba… | 0.946 |
| US | 0 | White Enterprises | 35 Enterprise Drive, Middletown, OH | White Enterprises L.L.C. | 46 Enterprise Drive, Ohio, Middletown | 0.859 |
| India | 0 | Vns Services Private Limited | Flat No 504, H.No. 1-10-1, Ideas Sudarshan Archad… | Vns Services Infratech Private  Limited | No 50, H.no. 1-10-1, Ideas Sudarshan Archade, Sec… | 0.851 |
| India | 0 | Hashtag & Associates | W/O Suresh Chand, Shop No. 1, Nagla Rambal, Nai A… | HASHTAG & ASSOCIATES OVERSEAS | NO 32 W/O SURESH CHAND, SHOP NO. 1, NAGLA RAMBAL,… | 0.945 |
| India | 1 | Smart Future Management Private Limited | At/Po-Subarnapur Ps/Dist-Subarnapur, Subarnapur, … | Smart Future Management Public Limited | OD, Sonepur, H.no 43 At/po-subarnapur Ps/dist-sub… | 0.957 |
| US | 0 | Judkins Clean Manufacturing | St. Albans, WV, 940 Elmhurst Drive | Judkins Clean Manufacturing Inc | 961 ELMHURST DR, ST. ALBANS, WV | 0.883 |
| US | 0 | Happy Cleaning Service Inc | 323 Highland Avenue, Wallingford, CT | Holdings Happy Cleaning Service | 336 HIGHLAND AVENUE, WALLINGFORD, CT | 0.868 |
| US | 0 | Superior Medical Enterprises Inc | 5069 Homeworth Road, Homeworth, OH | Superior Medical Enterprises Corp | 5071 Homeworth Rd, Homeworth, Ohio | 0.862 |
| India | 1 | Mack & Brothers Private Limited | 040, Bangalore, Karnataka, H-Block, Aditya Sollie… | MACK & BROTHERS EXPORTS PRIVATE LIMITED | 47, H-BLOCK, ADITYA SOLLIEVO APARTMENT NELLURAHAL… | 0.877 |
| US | 0 | Pinnacle Clear Great LLC | Easton, 22 Pinebrook Lane, MA | Pinnacle Clear Great Inc | 35 Pinebrook Ln, EASTON, MA | 0.906 |

## 7.4 20 true matches still missed at k=50 (name, pool = all S2+S3; 15,756 misses in total)

| country | s1_name | match_name | s1_addr | match_addr | match_id |
|---|---|---|---|---|---|
| US | Pacific Research Brands LLC | LLC Pacific Ree1sch Brands | 8931 Reynolda Road, Pfafftown, NC | 8931-C REYNOLDA RD, PFAFFTOWN, NC | S2-26684756 |
| India | Modern Technologies | ಮಾಡರ್ನ್ ಟೆಕ್ನಾಲಜೀಸ್ | Karnataka, Unit No. 709, Bangalore, 7Th Floor, A Wing,… | UNIT NO. 709, BANGALORE, Karnataka | S2-444570192 |
| India | Red Investments Private Limited | রেড ইনভেস্টমেন্টস প্রাইভেট লিমিটেড | 32, Pilkhana First Lane, Howrah, Howrah, West Bengal | WB, Howrah, 32 | S3-253718822 |
| India | Unique Engineering Private Limited | Unique Engaernig Private Limited | Plot No 1-Hig, Phase-V, Kukatapally, Balanagar, Hydera… | 1-HIG. , PHASE-V, KUKATAPALLY, BALANGAAR, Andhra Prade… | S2-197106935 |
| US | Graham Lithium Inc | Graham Lgitium Inc | 22 Hampton Road, MD, Fl 0, Linthicum Heights | 92 HAMPTON ROAD, LINTHICUM HEIGHTS, MD | S2-142341336 |
| US | Yeh Guggenheim of Somers PC | yehsomersguggenheim.com | 6 Valley Drive, Somers, NY | YORKTOWN HEIGHTS, 7 VALLEY DR, NY | S2-128717340 |
| US | Family Society | Family [Society] | 22723 153rd Avenue, Snohomish, WA | 22723 153rd Ave, NULL, High Bridge, Washington | S3-144220659 |
| US | Urban | Ulbban | 9925 226th Street, Kent, WA | 9925 226ND STREET, null, KENT, WA | S2-358056365 |
| India | Alpha Infra Private Limited | अल्फा इंफ्रा प्राइवेट लिमिटेड | House No B 94 Shahpura Pole- Spb-21, Bhopal, Madhya Pr… | HOUSE NO B #94 SHAHPURA POLE- SPB-21, BHOPAL, Madhya P… | S2-81108058 |
| India | Sharp Educational Society | Sharp Society Eucatioial | 6-3-663/G/4, 2Nd Floor Innovative House, Punjagutta, H… | 2-6-3-663/G/4, 2ND FLOOR INNOVATIVE HOUSE, PUNJAGUTTA,… | S2-982494257 |
| US | 75/91 Limited | 75H/SA1N LIMITED | 99 Sunset Rock Road, Andover, MA | 99 Sunset Rock Road, ANDOVER, MA | S2-467789817 |
| US | Superior Commercial Company | Superior-Commercial | 9768 Burberry Way, Highlands Ranch, CO | 9768- Burberry Way, Highlands Ranch, Colorado | S3-530412297 |
| US | Sion Assets LLC | Sion Ases LLC | 5575 Government Street, Baton Rouge City, LA | 5575-5577 Government St, Baton Rouge, Louisiana | S3-690839933 |
| India | Sree Consulting Pvt Ltd | Sree Coneuteing Pvt Ltd | 78, Ladli Bhawan, Kedarnath Road Darya Ganj, New Delhi… | PLOT 42 78, LADLI BHAWAN, KEDARNATH ROAD DARYA GANJ, D… | S2-641433450 |
| India | Arihant Sree Hospitality LLP | अरिहंत श्री हॉस्पिटैलिटी एलएलपी | Bc-053-0001 Vikramshila Colony, Bihar, Bhagalpur, Urdu… | Bhagalpur, BR, Bc-053-0001 Vikramhila Colony, Urdu Baz… | S3-227968486 |
| India | North Bharat Projects Pvt Ltd | ਨੌਰਥ ਭਾਰਤ ਪ੍ਰੋਜੈਕਟਸ ਪ੍ਰਾ. ਲਿ. | Sco 9 Phase Ix, Mohali, Chandigarh, Punjab | SCO 9 PHASE IX, MOHALI, Punjab, CHANDIGARH | S2-384273400 |
| US | Bay Four LLC | Bay Fóur LLC | 182 Main Street, Unit 27, Salt Lake City, UT | 182 MAIN STREET, SALT LAKE CITY, UT | S2-200494409 |
| India | Lakshmi Finance Private Limited | lakshmifinance.com | Enkay Heritage, Rly Stn Rd, Plot-403, Tps Shop 003, Pa… | ENKAY HERITAGE, RLY STN RD, PLOT-401, TPS SHOP 003, PA… | S2-368760650 |
| India | Lotus Global Limited | લોટસ ગ્લોબલ લિમિટેડ | Rs. No. 197, 198, 199, 201, 202, Opp Kanesara Bus Stan… | RS. NO. 197, SIDHPUR, PATAN, 198, 199, 201, 202, OPP K… | S2-732440078 |
| India | Hotel Consultancy Pvt Ltd | Hotel Cónsultancy Pvt Limited | B Ff2, Sarjanam Resicom, Beside Krishna Vatika Sayajip… | Block D-764 B Ff2, Sarjanam Resicom, Beside Krishna Va… | S3-767698763 |

## 7.5 20 true matches still missed at k=50 (nameaddr, pool = all S2+S3; 2,268 misses in total)

| country | s1_name | match_name | s1_addr | match_addr | match_id |
|---|---|---|---|---|---|
| India | Prime Exports Private Limited | प्राइम एक्सपोर्ट्स प्राइवेट लिमिटेड | Rz-40, First Floor, Back Side Of S. Extn.-Part-Ii, Utt… | Rz-0040, New Delhi, DL | S3-176672383 |
| US | Swett's Painting | Spáinting.Com | 12616 330, Elizabethtown, IN | Elizabethtwn, Indiana, 12615 330 | S3-131919456 |
| India | Mumbai Emu Pvt. Ltd. | Mbicba Emu Pvt. Ltd. | 3/C, 504, Neighbourhood, Lokhandwala Kandivali (East),… | 03/C, Mumbai, Mumbai City, MH | S3-910318169 |
| India | Black Krishna Investments Private Limited | ಬ್ಲ್ಯಾಕ್ ಕೃಷ್ಣ ಇನ್ವೆಸ್ಟ್‌ಮೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | Sai Grace, E-3/54-56, 1St Main, 3Rd Cross, Nanjappa La… | Sai Grace, Bangalore, KA | S3-819047526 |
| US | The Dent Deli | The Dént Deli | 1611 Parkview Avenue, Ozark, MO |  | S2-399021145 |
| US | Ear Nose & Throat Specialists L.L.C. | Ear-Nose & Throat Specialists | 1928 Tuckey Lane, Phoenix, AZ | 1928. Tuckey Ln, Northwest Annex, Arizona | S3-83955662 |
| India | Star Marketing | स्टार मार्केटिंग | B-6, Medhavi, 2Nd Cross Road, Lokhandwala Complex, Aza… | Plot 963 B-6, Navi Mumbai, MH | S3-830487127 |
| US | Valley Westin Group | Valley  Group Service | 32 Palmer Court, Loudoun County, VA | 32 Palmer Ct, Sterling, Virginia | S3-573100746 |
| India | Global Properties Private Limited | ಗ್ಲೋಬಲ್ ಪ್ರಾಪರ್ಟೀಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | 23/A 31 Rd, 1 Crs St, Sakamma Badavane, Bangalore Nort… | H.NO 23/A  31 RD, NULL, BANGALORE, BANGALORE NORTH, Ka… | S2-201289219 |
| India | Orion Services (India) Private Limited | Orion Srehvicse (India) Private Limited #48688 | Dhoori, Pilkhuwa, Uttar Pradesh, Pilkhuwa, Ghaziabad, … |  | S3-410081852 |
| India | My Consultants Limited | മൈ കൺസൾട്ടന്റ്സ് ലിമിറ്റഡ് | 70/482, Pisharikkavu West Residence Association, Edakk… | 70/482, CALICUT, KOZHIKODE, Kerala | S2-404503037 |
| India | Raj Academy Private Limited | Mr Raj Academy Private Ltd | 138 Rajdhani Enclave Pitampura, Delhi, North Delhi, De… |  | S2-694343081 |
| US | Howard Wisconsin LLC | Howard-Wisconsin | 3300 18th Street, Unit 102, Washington, DC |  | S2-233467050 |
| US | Pediatric Dentistry Associates | Pediatric Dentistry (Associates) | 21188 Winding Brook Square, Loudoun County, VA |  | S2-414433550 |
| India | Seven Developers Private Limited | செவன் டெவலப்பர்ஸ் பிரைவேட் லிமிடெட் | 218, Anthonipuram Salem, Salem, Tamil Nadu | SALEM, Tamil Nadu, 218/7 | S2-808946987 |
| India | QN Solutions Private Limited | QN Solutions | Plot No. 2A E Block, Hebbal Industrial Area, Hebbal, M… |  | S3-988443689 |
| India | City Infra LLP | सिटी इंफ्रा एलएलपी | Varanasi, Uttar Pradesh, Varanasi, Bada Lalpur, Varana… | 18 B.D.A. COLONY, VARANASI, Uttar Pradesh | S2-882890674 |
| India | Onkar Welfare Society | Onkar Society Center | B-106, Omicron-1A, Noida, Gautam Buddha Nagar, Uttar P… | B-c-106, Noida, UP | S3-203697337 |
| India | Big New Solutions Private Limited | బిగ్ న్యూ సొల్యూషన్స్ ప్రైవేట్ లిమిటెడ్ | Plot 410, Flat No 402, 4Th Floor, Sameeksha Spires Mat… | తెలంగాణ, Plot 410, Hyderabad | S3-856305643 |
| India | Mannat India LLP | Mannat India  LLP | Madhu Malaxmi Chambers, D. No. 32-9-17, 17A & 32-2-6, … |  | S2-666208001 |


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

# 8. Train vs test shift

Token statistics use a random sample of up to 200,000 rows per (split, source, country) (seed 42).

## 8.1 Length and script: train vs test (US, India) and France

| split | source | country | n | name chars mean/p5/p50/p95 | addr chars mean/p5/p50/p95 | addr empty | name non-latin | name latin-accented | addr non-latin |
|---|---|---|---|---|---|---|---|---|---|
| test | S1 | France | 259,452 | 19.4 / 12 / 19 / 29 | 50.1 / 40 / 48 / 65 | 0.000 | 0.000 | 0.157 | 0.000 |
| test | S2 | France | 703,378 | 21.1 / 11 / 20 / 35 | 39.5 / 22 / 40 / 59 | 0.031 | 0.000 | 0.247 | 0.000 |
| test | S3 | France | 731,615 | 21.1 / 10 / 20 / 36 | 40.0 / 22 / 41 / 60 | 0.029 | 0.000 | 0.241 | 0.000 |
| test | S1 | India | 809,986 | 26.4 / 13 / 27 / 38 | 77.7 / 47 / 76 / 116 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S1 | India | 883,188 | 26.4 / 13 / 27 / 38 | 77.7 / 47 / 76 / 116 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | India | 2,312,565 | 28.3 / 14 / 28 / 44 | 68.7 / 31 / 68 / 110 | 0.023 | 0.237 | 0.040 | 0.238 |
| train | S2 | India | 2,017,799 | 27.4 / 13 / 27 / 42 | 68.2 / 30 / 68 / 110 | 0.029 | 0.234 | 0.044 | 0.237 |
| test | S3 | India | 2,405,000 | 27.9 / 12 / 28 / 44 | 59.4 / 20 / 59 / 106 | 0.025 | 0.133 | 0.049 | 0.229 |
| train | S3 | India | 2,115,547 | 27.0 / 12 / 27 / 43 | 59.1 / 19 / 58 / 106 | 0.031 | 0.132 | 0.053 | 0.225 |
| test | S1 | US | 663,106 | 22.5 / 11 / 22 / 35 | 35.0 / 26 / 34 / 48 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S1 | US | 1,323,633 | 22.5 / 11 / 22 / 35 | 35.0 / 26 / 34 / 48 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | US | 1,871,330 | 24.3 / 11 / 24 / 39 | 31.8 / 22 / 32 / 43 | 0.029 | 0.000 | 0.062 | 0.000 |
| train | S2 | US | 3,016,817 | 23.6 / 11 / 23 / 39 | 31.5 / 21 / 32 / 43 | 0.037 | 0.000 | 0.068 | 0.000 |
| test | S3 | US | 1,945,701 | 24.6 / 11 / 24 / 41 | 38.8 / 26 / 39 / 53 | 0.028 | 0.000 | 0.063 | 0.000 |
| train | S3 | US | 3,170,056 | 24.0 / 11 / 23 / 40 | 38.4 / 25 / 39 / 53 | 0.035 | 0.000 | 0.067 | 0.000 |

## 8.2 Token overlap: test tokens seen in train (same source & country)

| source | country | field | test_token_types | types_seen_in_train | occurrences_seen_in_train |
|---|---|---|---|---|---|
| S1 | India | name | 32,443 | 0.7230 | 0.9822 |
| S1 | India | addr | 125,568 | 0.4486 | 0.9672 |
| S1 | US | name | 35,926 | 0.6797 | 0.9782 |
| S1 | US | addr | 62,846 | 0.6185 | 0.9747 |
| S2 | India | name | 57,710 | 0.4428 | 0.9508 |
| S2 | India | addr | 119,208 | 0.4358 | 0.9625 |
| S2 | US | name | 70,465 | 0.3812 | 0.9314 |
| S2 | US | addr | 80,466 | 0.5493 | 0.9575 |
| S3 | India | name | 61,917 | 0.4374 | 0.9470 |
| S3 | India | addr | 105,382 | 0.4391 | 0.9641 |
| S3 | US | name | 71,376 | 0.3887 | 0.9323 |
| S3 | US | addr | 81,115 | 0.5517 | 0.9605 |
| S1 | France (vs train US+India) | name | 30,696 | 0.3072 | 0.4298 |
| S1 | France (vs train US+India) | addr | 14,953 | 0.2854 | 0.6590 |
| S2 | France (vs train US+India) | name | 44,320 | 0.2034 | 0.4678 |
| S2 | France (vs train US+India) | addr | 34,578 | 0.1659 | 0.6312 |
| S3 | France (vs train US+India) | name | 46,331 | 0.2233 | 0.5381 |
| S3 | France (vs train US+India) | addr | 33,758 | 0.1642 | 0.6439 |

Postal-code / pattern rates for train vs test are in section 5.1 (both splits are listed there).


## 8.3 Proxy for match rate: S1 entities with >=1 identical-nname S2/S3 record (same country)

| split | country | s1_entities | share_with_exact_nname_hit | train_singleton_rate |
|---|---|---|---|---|
| train | India | 883,188 | 0.9223 | 0.0559 |
| train | US | 1,323,633 | 0.9226 | 0.0558 |
| test | France | 259,452 | 0.9590 |  |
| test | India | 809,986 | 0.9276 |  |
| test | US | 663,106 | 0.9290 |  |

## 8.4 France: record-level characteristics

| source | records | name_has_accent | addr_has_accent | addr_empty | addr_has_5digit | addr_starts_with_number | addr_mean_components | name_UPPER | addr_UPPER | name_has_quote | name_has_apostrophe | name_has_hyphen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 259,452 | 0.157 | 0.278 | 0.000 | 0.004 | 0.860 | 3.039 | 0.000 | 0.000 | 0.001 | 0.000 | 0.022 |
| S2 | 703,378 | 0.245 | 0.195 | 0.031 | 0.005 | 0.669 | 2.608 | 0.208 | 0.290 | 0.000 | 0.000 | 0.047 |
| S3 | 731,615 | 0.239 | 0.199 | 0.029 | 0.005 | 0.677 | 2.628 | 0.056 | 0.000 | 0.000 | 0.000 | 0.046 |

Legal-form tokens in France names (share of names containing token):

| source | sarl | sas | sasu | sa | eurl | sci | snc | association | co |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.2832 | 0.2015 | 0.0413 | 0.0492 | 0.0654 | 0.0322 | 0.0000 | 0.0154 | 0.0003 |
| S2 | 0.2091 | 0.1405 | 0.0444 | 0.0489 | 0.0590 | 0.0389 | 0.0106 | 0.0142 | 0.0003 |
| S3 | 0.2064 | 0.1391 | 0.0433 | 0.0480 | 0.0578 | 0.0381 | 0.0100 | 0.0139 | 0.0014 |

Street-type tokens in France addresses (accent-folded):

| source | rue | r | avenue | av | ave | boulevard | bd | chemin | ch | place | pl | allee | impasse | imp | route | rte | quai | cours | square | faubourg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.6582 | 0.0002 | 0.1194 | 0.0085 | 0.0000 | 0.0359 | 0.0076 | 0.0141 | 0.0000 | 0.0126 | 0.0037 | 0.0464 | 0.0198 | 0.0008 | 0.0171 | 0.0015 | 0.0076 | 0.0084 | 0.0031 | 0.0020 |
| S2 | 0.3827 | 0.2539 | 0.0653 | 0.0392 | 0.0153 | 0.0199 | 0.0165 | 0.0079 | 0.0051 | 0.0068 | 0.0084 | 0.0277 | 0.0110 | 0.0085 | 0.0093 | 0.0082 | 0.0042 | 0.0045 | 0.0028 | 0.0018 |
| S3 | 0.3942 | 0.2435 | 0.0678 | 0.0377 | 0.0148 | 0.0204 | 0.0160 | 0.0083 | 0.0048 | 0.0072 | 0.0082 | 0.0283 | 0.0113 | 0.0082 | 0.0095 | 0.0079 | 0.0043 | 0.0047 | 0.0030 | 0.0017 |

Most common last address component (France):

| S1 | S2 | S3 |
|---|---|---|
| Hauts-de-France (87,960) | Hauts-de-France (76,368) | Hauts-de-France (85,734) |
| Nouvelle-Aquitaine (73,819) | Gironde (65,047) | Nouvelle-Aquitaine (71,925) |
| Pays de la Loire (62,901) | Nord (64,801) | Nord (65,648) |
| Bordeaux (2,870) | Nouvelle-Aquitaine (64,246) | Gironde (65,344) |
| Nantes (2,441) | Loire-Atlantique (54,689) | Pays de la Loire (61,870) |
| Lille (2,211) | Pays de la Loire (54,500) | Loire-Atlantique (55,324) |
| Tourcoing (1,221) | BORDEAUX (34,618) | Bordeaux (38,951) |
| Dunkerque (1,110) | NANTES (30,485) | Nantes (34,117) |
| Calais (1,057) | LILLE (27,832) | Lille (31,011) |
| Roubaix (1,050) | TOURCOING (14,620) | Tourcoing (16,277) |
| Saint-Nazaire (868) | DUNKERQUE (13,672) | Dunkerque (15,656) |
| Pessac (812) | ROUBAIX (13,115) | Roubaix (14,879) |
| La Teste-de-Buch (778) | CALAIS (12,628) | Calais (14,283) |
| Mérignac (691) | Pas-de-Calais (11,974) | Pas-de-Calais (12,158) |
| Lège-Cap-Ferret (550) | PESSAC (10,193) | Pessac (11,774) |

Most common second-to-last component (city slot, France):

| S1 | S2 | S3 |
|---|---|---|
| Bordeaux (37,215) | BORDEAUX (61,176) | Bordeaux (73,194) |
| Nantes (32,250) | NANTES (52,365) | Nantes (63,172) |
| Lille (29,592) | LILLE (48,294) | Lille (57,950) |
| Tourcoing (15,715) | TOURCOING (25,753) | Tourcoing (30,886) |
| Dunkerque (14,799) | DUNKERQUE (24,296) | Dunkerque (28,969) |
| Roubaix (14,112) | ROUBAIX (23,304) | Roubaix (27,717) |
| Calais (13,647) | CALAIS (22,272) | Calais (26,967) |
| Saint-Nazaire (11,910) |  (21,717) |  (21,694) |
| Pessac (10,937) | PESSAC (17,913) | Pessac (21,366) |
| La Teste-de-Buch (9,777) | SAINT-NAZAIRE (14,396) | Saint-nazaire (14,489) |
| Mérignac (8,900) | LA TESTE-DE-BUCH (14,026) | La Teste-de-buch (13,739) |
| Lège-Cap-Ferret (6,943) | PORNIC (11,197) | Pornic (13,576) |
| Pornic (6,702) | MÉRIGNAC (10,110) | Mérignac (13,187) |
| Hauts-de-France (6,653) | LA BAULE-ESCOUBLAC (8,546) | La Baule-escoublac (8,498) |
| Saint-Herblain (6,005) | Bordeaux (7,763) | Saint-herblain (7,470) |

Top France name tokens:

| S1 | S2 | S3 |
|---|---|---|
| sarl (56,772) | sarl (39,787) | sarl (39,470) |
| sas (40,264) | sas (28,341) | sas (27,728) |
| club (17,514) | france (19,179) | france (18,759) |
| france (16,068) | s (14,192) | club (13,864) |
| de (13,701) | club (13,899) | s (13,640) |
| eurl (13,050) | eurl (11,859) | de (11,495) |
| amicale (11,066) | de (11,550) | eurl (11,404) |
| sa (9,809) | sa (9,727) | sa (9,548) |
| ecole (8,476) | amicale (9,118) | a (9,451) |
| du (8,338) | sasu (8,992) | amicale (9,131) |
| maison (8,317) | a (8,050) | sasu (8,757) |
| sasu (8,268) | sci (7,812) | sci (7,686) |
| centre (8,091) | groupe (7,578) | groupe (7,362) |
| comite (7,718) | maison (7,099) | maison (7,017) |
| union (6,661) | du (6,990) | com (6,882) |
| sci (6,485) | com (6,946) | du (6,861) |
| sportive (6,400) | ecole (6,616) | ecole (6,688) |
| des (6,046) | développement (6,348) | centre (6,373) |
| amis (5,346) | centre (6,297) | développement (6,009) |
| primaire (5,219) | comite (6,146) | comite (6,000) |
| pharmacie (4,567) | union (5,807) | fils (5,726) |
| cie (4,387) | fils (5,744) | union (5,660) |
| saint (4,332) | des (5,138) | des (5,004) |
| la (4,126) | sportive (5,022) | sportive (4,868) |
| ets (4,110) | participations (4,871) | participations (4,785) |

Top France address tokens:

| S1 | S2 | S3 |
|---|---|---|
| de (198,529) | de (108,390) | de (111,954) |
| rue (131,636) | rue (76,542) | rue (78,767) |
| la (89,437) | r (50,639) | la (51,472) |
| france (79,012) | la (49,828) | r (48,607) |
| hauts (78,548) | loire (35,833) | loire (37,028) |
| nouvelle (65,587) | bordeaux (32,451) | bordeaux (32,708) |
| aquitaine (65,555) | nantes (27,827) | nantes (27,972) |
| loire (56,196) | lille (25,943) | france (27,573) |
| pays (56,173) | du (25,913) | hauts (27,189) |
| bordeaux (33,494) | france (25,452) | du (25,922) |
| nantes (29,061) | hauts (25,089) | lille (25,817) |
| lille (26,891) | des (23,264) | des (23,404) |
| du (26,814) | nord (21,495) | nouvelle (22,708) |
| des (24,357) | nouvelle (21,185) | aquitaine (22,673) |
| avenue (23,904) | aquitaine (21,164) | gironde (20,885) |
| saint (20,214) | gironde (21,137) | nord (20,752) |
| tourcoing (14,187) | atlantique (18,221) | pays (19,546) |
| dunkerque (13,585) | pays (17,771) | atlantique (17,626) |
| roubaix (12,986) | saint (15,892) | saint (16,040) |
| calais (12,318) | calais (15,873) | calais (15,614) |
| nazaire (10,653) | tourcoing (13,731) | avenue (13,464) |
| pessac (9,891) | dunkerque (13,136) | tourcoing (13,452) |
| buch (8,745) | avenue (13,092) | dunkerque (13,234) |
| teste (8,735) | roubaix (12,665) | roubaix (12,615) |
| bis (8,543) | nazaire (10,496) | nazaire (10,465) |

## 8.5 France: 30 random S1 rows

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S1-41726612 | Mediation Visages Fetes EURL | Dunkerque, 3 Rue de la Bienfaisance, Hauts-de-France | France |
| S1-845965189 | Lille Club | 15 BIS Rue d'Arcole, Lille, Hauts-de-France | France |
| S1-881396111 | Amicale du Usine | 41 Place de la Cinquième République, Pessac, Nouvelle-Aquitaine | France |
| S1-657417418 | Pharmacie Dame | 26 Rue Mercière, Bordeaux, Nouvelle-Aquitaine | France |
| S1-872609724 | Nantes Sport SARL | Pays de la Loire, 29 Rue du Fezzan, Nantes | France |
| S1-692990298 | Établissements Faire | 7 Avenue de la Reousse Dune l'Herbe, Lège-Cap-Ferret, Nouvelle-Aquitaine | France |
| S1-357802158 | Foire Maternelle EURL | 83 Rue de Laseppe, Bordeaux, Nouvelle-Aquitaine | France |
| S1-150454433 | Maison Formule | 22 Rue de Mons, Roubaix, Hauts-de-France | France |
| S1-867280894 | Dansez Construction EURL | 4 Allée de Cherbourg, Dunkerque, Hauts-de-France | France |
| S1-553377212 | Art (France) College SARL | Bordeaux, Nouvelle-Aquitaine, 17 Rue des Trois Chandeliers | France |
| S1-182087270 | Espace (France) Service SAS | 20 Rue Maurice, Bordeaux, Nouvelle-Aquitaine | France |
| S1-117869567 | Horizons Sante SARL | 28 Chemin du Petit Marsac, Saint-Nazaire, Pays de la Loire | France |
| S1-200821668 | ET Primaire SCI | 129 Rue du Marché, Lille, Hauts-de-France | France |
| S1-532537669 | Asso Etablissement (France) SARL | 10 Avenue la Palombière Piraillan, Lège-Cap-Ferret, Nouvelle-Aquitaine | France |
| S1-809780656 | EHPAD Graines | 6 Rue Miternique, Dunkerque, Hauts-de-France | France |
| S1-449426066 | Saint-Nazaire Services SAS | Saint-Nazaire, Pays de la Loire, 11 Allée Sarah Vaughan | France |
| S1-171548044 | Alpin Comite SA | 27 Rue Antoine Leleu, Calais, Hauts-de-France | France |
| S1-239977709 | Dames Club SCI | 10 Rue d' Angers, Tourcoing, Hauts-de-France | France |
| S1-966848668 | Accords (France) Soins SARL | Pays de la Loire, 39 Allée de la Houssaie, Saint-Nazaire | France |
| S1-371152910 | Histoire Culture SARL | 25 Rue de Reims, Tourcoing, Hauts-de-France | France |
| S1-96782121 | RQ Primaire SASU | 2 ter Rue Armand Calmon, Pessac, Nouvelle-Aquitaine | France |
| S1-291935000 | PAC Logistique SAS | Lille, Hauts-de-France, 42 Rue du Chaufour | France |
| S1-313025096 | Cavaliers Primaire SARL | 11 Rue Davy, Lille, Hauts-de-France | France |
| S1-799157291 | Centre Hospitalier Anim | 68 Rue Sainte Barbe, Tourcoing, Hauts-de-France | France |
| S1-117979957 | Formation Centre SARL | 7 RUE Censive du Tertre, Nantes, Pays de la Loire | France |
| S1-620960476 | Association de Communale | 16 Rue Boris Vian, Mérignac, Nouvelle-Aquitaine | France |
| S1-923202670 | Corneille Culturelle SARL | 6 Avenue du Bassin, La Teste-de-Buch, Nouvelle-Aquitaine | France |
| S1-658704498 | Centre Médical Nicolas | 56 Rue de Rigoulet, Bordeaux, Nouvelle-Aquitaine | France |
| S1-499525702 | Pornic Collectif SARL | 35 RUE des Coeures, Pornic, Pays de la Loire | France |
| S1-357372944 | Vie Comite SARL | 24 Rue Paul Langevin, Saint-Nazaire, Pays de la Loire | France |

10 random S2 France rows:

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S2-19087216 | JX Sportive | 30 RUE DE MENIN, TOURCOING, Nord | France |
| S2-824918948 | ASSOCIATION DE VALOIS EURL | N° 14 R. DE BOULMGNE, TOURCOING | France |
| S2-533464750 | École primaire International SCI | NO 12 RUE LOUIS BRALLE, BORDEAUX | France |
| S2-677480728 | Pontet Ecole SAS | 57 RUE DE LA VILLE EN PIERRE, NANTES, Pays de la Loire | France |
| S2-11241777 | Sociale Chu Comite SARL | Hauts-de-France, NO. 14 RUE MONSEIGNEUR PIEDFORT, CALAIS | France |
| S2-751420807 | Trois & Frèrês Développement SA | 56 - RTE DES QUATRE VENTS, SAINT-NAZAIRE, Loire-Atlantique | France |
| S2-854158318 | 3eme Culturelle (Frànce) | 1 BD. DU MASSACRE, SAINT-HERBLAIN, Loire-Atlantique | France |
| S2-943119084 | MQG Amicale | 005 PLACE DES BASQUES, BORDEAUX, Nouvelle-Aquitaine | France |
| S2-484686989 | Spv Jeunes SCI | Loire-Atlantique, LA BAULE-ESCOUBLAC, 82 BOULEVARD DE L'OÉAN | France |
| S2-104097631 | Dentaire Societe (France) SCI | 91 R ROBERT CAUMONT - IMMEUBLE P, LES BUREAUX DU LAC II, BORDEAUX, Gironde | France |

10 random S3 France rows:

| entity_id | business_name | business_address | country |
|---|---|---|---|
| S3-366461218 | Tourcoing Loisirs | N°94 R Des Champs, Tourcoing | France |
| S3-911557245 | Vigan SASU Comite | 8 Rue André Chenier, Lille, Hauts-de-France | France |
| S3-295250807 | Bureau SAS Fetes (France) | 23 Bis Rue De Caulet, Bordeaux | France |
| S3-889785051 | @Alissasmaternelle | 115 R. Dupuy De Lôme, Roubaix, Nord | France |
| S3-717904579 | Team Ecole E.U.R.L. | 19 R La Clairière Des Ontiens, Mérignac, Gironde | France |
| S3-584659989 | entente comite (france) | 361 R. Jules Guesde, Roubaix | France |
| S3-521711221 | BLP Immôbilier SA | 9 Rue Paul Painlevé, Tourcoing | France |
| S3-89844898 | Balade Primaire (France) E.U.R.L. | R. De Gnad, Tourcoing, Nord | France |
| S3-189802196 | EURL IGG Auto | 15 R Thévenet, Dunkerque | France |
| S3-543623090 | Cultures Amicale | 94 Rue Notre Dame, Bordeaux, Nouvelle-Aquitaine | France |


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
