# Cleaning

Three stages. Section numbers `[x.y]` refer to `eda/EDA_REPORT.md`.

| stage | code | output | learned from labels? |
|---|---|---|---|
| 1 deterministic text rules | `src/ber/normalize.py`, `src/ber/lexicon.py`, `scripts/03_clean.py` | `data/clean/{split}_source{n}.parquet` | no |
| 2 state-slot table | `src/ber/lookup.py`, `scripts/05_learn_tables.py` | `data/dicts/state_map.parquet`, `data/clean/stage2_{split}_source{n}.parquet` | yes, train_ids S1 only |
| 3 evaluation | `scripts/04_eval_cleaning.py` | `reports/stage3_cleaning_eval.md` | reads train_ids pairs only |

Invariants

- One implementation for train and test: polars expressions only, no per-row Python.
- Raw columns (`business_name`, `business_address`, ...) are never modified; cleaned columns are added.
- Stage 1 files are immutable once written. Stage 2 writes separate sidecar files keyed by
  `entity_id`; it never rewrites a Stage 1 column. A rule change bumps NORMALIZE_VERSION and
  regenerates all files (history: 1.1.0 -> 1.1.1 house-number units; 1.1.1 -> 1.2.0 the three
  fixes below; 1.2.0 -> 1.2.1 digit part of glued tokens in addr_numbers).
- Country is an open set (France is test-only [2, 8.4]). No rule reads the `country` column in
  Stage 1; rules are keyed on token shapes, so on unseen text they do nothing
  (`test_unseen_text_is_noop`). Stage 2 maps only countries it has labels for; France gets
  `addr_state_canon = null` (unknown), never a mismatch.
- No external data or APIs. The hand-written lists in `src/ber/lexicon.py` are general language
  knowledge: legal forms, honorifics, state/region names and codes, street abbreviations.
- **France rules came from inspecting unlabeled test text** (there is no France training data):
  French legal forms, `r`/`av`/`bd`/`pl`/`ch`/`imp`/`rte`, `N°`, `et`, the France region/department list.
- Memory guard (`src/ber/memguard.py`): scripts abort cleanly if RSS > 5 GiB. Full Stage 1 peaks at
  1.3 GiB; Stage 2 at 1.0 GiB.

## Cleaned data contract

For teammates building blocking and features on these files. **NORMALIZE_VERSION = 1.2.1**
(stored in every file's parquet key-value metadata as `NORMALIZE_VERSION`; check it when you load).

### Files

| file | rows | key | content |
|---|---|---|---|
| `data/clean/{split}_source{n}.parquet` | same as the raw source, same order | `entity_id` | raw columns + all Stage 1 columns below |
| `data/clean/stage2_{split}_source{n}.parquet` | same as Stage 1, same order | `entity_id` | `addr_state_canon` only; join on `entity_id` |
| `data/dicts/state_map.parquet` | 141 | (`country`, `variant`) | learned state spellings, provenance in metadata |

`split` in {train, test}, `n` in {1, 2, 3}. Raw columns (`entity_id`, `business_name`,
`business_address`, `country`, `source`) are unchanged from `data/parquet`.

### Null semantics (read this first)

- **Empty addresses are null, not `""`.** If the address is empty after removing fillers
  (`""`, `null`, `N/A`, `<NULL>`), EVERY `addr_*` column is null (strings AND lists). About 3% of
  S2/S3 rows. Treat null as "unknown", never as a match or a mismatch.
- Names are never empty, so `name_*` columns are never null except `name_domain_stem`.
- Optional string columns (`name_domain_stem`, `addr_house_number`, `addr_unit`, `addr_landmark`,
  `addr_state_raw`, `addr_core`) are null when absent, never `""`.
- List columns on a present address are lists, possibly empty (`[]`).
- **`addr_state_canon` is null for France**: there are no France labels, so the state is UNKNOWN.
  A null on either side must never be counted as a state mismatch. It is also null for US/India rows
  whose state is missing or not in the learned table (~3-4% of US and India S2/S3 rows).

### Columns

| column | type | meaning | null when | example |
|---|---|---|---|---|
| `name_clean` | str | full cleaned name: NFKC, lowercase, Latin accents/quotes/punctuation removed, `&`/`et` -> `and`, OCR 0/1 fixed, dotted acronyms merged | never | `qes induction pvt ltd` |
| `name_core` | str | `name_clean` minus legal forms (anywhere; short ambiguous ones only at the edges), injected prefixes (M/s, Mr, Smt, Shri, Dr), `#tags`, domain TLDs. Falls back to the prefix-stripped name if nothing would be left | never | `qes induction`, `राम मार्केटिंग` |
| `name_nospace` | str | `name_core` without spaces, to compare against glued domains/handles | never | `sarthitrading` |
| `name_domain_stem` | str | stem of a domain or @handle in the raw name | no domain/handle (~96%) | `sarthitrading` |
| `legal_families` | list[str] | **a SET** (sorted, unique) of legal families found; compare with set operations, not equality of the first element. Values: CO CORP EI EURL INC LIMITED LLC LLP LP PC PLLC PRIVATE PUBLIC SA SARL SAS SASU SCI SNC | never (may be `[]`) | `["LIMITED", "PRIVATE"]` |
| `legal_suffix_class` | str | LEGACY single class of the last Latin legal form; does not see Indic forms or shuffled order. Prefer `legal_families` | never (`"NONE"`) | `PRIVATE_LIMITED` |
| `name_script` | str | dominant script: Latin, Devanagari, Tamil, Kannada, Telugu, Bengali, Gujarati, Gurmukhi, Malayalam, Odia, Other, None | never | `Devanagari` |
| `name_has_nonlatin` | bool | any non-Latin letter in the name | never | `true` |
| `name_numbers` | list[str] | digit runs in `name_clean` (includes `#tag` numbers) | never (may be `[]`) | `["67693"]` |
| `addr_clean` | str | cleaned address, commas removed | empty address | `kh no 570/13 new delhi west delhi delhi` |
| `addr_numbers` | list[str] | every number-like token, leading zeros stripped per numeric part; plain ordinals included (`87th`). A token glued to a word (3+ letters in a row) contributes only its number part: `chambers16/11` -> `16/11`, `cour2` -> `2`; dropped if that part is an ordinal (`annexe3rd`). Unlike `addr_house_number`, which rejects glued tokens | empty address | `["570/13"]`, `["33", "2"]` |
| `addr_house_number` | str | first street-clause number: skips ordinals, US unit numbers, `1/2`, the India injected leading clause, and words with a digit (3+ letters in a row); falls back to the unit number | empty address, or no valid number (~7-10%) | `570/13`, `c-558`, `1-11-251/1b` |
| `addr_unit` | str | number after unit/suite/ste/apt/fl/floor/room | none (~95%) | `16b` |
| `addr_landmark` | str | landmark clauses (near/opp/behind/...), `, `-joined | none | `opp hotel vrindhavan` |
| `addr_no_landmark` | str | `addr_clean` without landmark clauses | empty address | |
| `addr_std_components` | list[str] | comma clauses of `addr_no_landmark` with abbreviations canonicalised, in source order | empty address | `["kh 570/13", "new delhi", "west delhi", "delhi"]` |
| `addr_std` | str | `addr_std_components` joined by spaces | empty address | `kh 570/13 new delhi west delhi delhi` |
| `addr_state_raw` | str | THE state/region clause as written (one clause: the last clause if it is a state, else the first clause that is entirely a state). Not canonicalised: `tx` vs `texas` differ here | no state clause | `delhi`, `महाराष्ट्र`, `hauts de france` |
| `addr_core` | str | `addr_std` without the state clause, leading zeros stripped | empty address, or only a state | `kh 570/13 new delhi west delhi` |
| `addr_script`, `addr_has_nonlatin` | str, bool | as for names | empty address | `Latin`, `false` |
| `addr_state_canon` (stage2 file) | str | state in the S1 spelling of its country (US code, India full name), learned from train_ids pairs | France (unknown), no/unmapped state, empty address | `tx`, `maharashtra` |

### What to use for what

- **Blocking text:** `name_core` (or `name_nospace`) + `addr_core`. Both are order-free token strings
  with legal forms, fillers, state and noise removed. Do NOT block on `name_clean` / `addr_clean`
  (legal forms and states dominate the tokens) or on `addr_state_raw` (spelling differs by source).
- **Hard constraints:** `country` (true pairs always share it). Use `addr_state_canon` equality only
  when BOTH sides are non-null.
- **Strong features:** `addr_house_number` exact / numeric gap (US 72% exact on true pairs vs 0.07%
  on look-alikes), `legal_families` set conflict (both non-empty and disjoint), `addr_core` token Jaccard.
- **Do not drop generic words** (services, group, exports): look-alikes are made by adding them.
- Indic-script names are NOT transliterated: `name_script` tells you when Latin-vs-Indic text
  comparison is meaningless.

## Rules and their justification

### Common text steps (names and addresses)

| rule | EDA | example |
|---|---|---|
| NFKC; non-ASCII digits -> ASCII | 1.4 | `प्लॉट १२३` -> `प्लॉट 123` |
| quote characters deleted (incl. `\x1a`, a stand-in apostrophe); CSV-escape artifacts of the QUOTE_NONE parquet disappear with them | 1.4 | `Shopper\x1aS Stop` -> `shoppers stop`, `"""ehpad Club SAS"` -> `ehpad club sas` |
| mojibake quote `â\x80\x99` deleted; other `Â\x80\x9x` -> `-` | 1.4 | `WORLDMARK Â\x80\x93 3` -> `worldmark 3` |
| ZWNJ/ZWJ deleted (not spaced: it would split the Indic word) | 1.4 | `ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್` |
| accents removed from Latin letters only (injected `Límited`, genuine French accents: both sides lose them) | 1.4, 8.4 | `Límited` -> `limited` |
| lowercase; `&` -> `and`; `œ` -> `oe` | 1.4, 4.4, 4.6 | |
| fillers `<NULL>`, `null`, `n/a` removed as tokens; `#`/`##` are punctuation | 1.4, 1.3 | `##3305 SHARATIN RD, null` -> `3305 sharatin rd` |
| OCR digits: `0`->`o`, `1`->`l` only with Latin letters on both sides, inside a token of letters and 0/1 only | 1.4 | `H0uston`, `De1hi`, `Ree1sch`; untouched `2Nd`, `12b`, `1-11-251/1B`, `S1W33789` |
| punctuation -> space; `/` and `-` kept inside number tokens; brackets dropped, content kept | 1.4, 5.1 | `Family [Society]` -> `family society`, `10-62/1` kept |
| dotted acronyms / single letters merged | 4.5 | `L.L.C.` -> `llc`, `R.K. Puram` -> `rk puram` |
| whitespace collapsed | 1.4 (550k double spaces) | |

### Names

| rule | EDA | notes |
|---|---|---|
| unambiguous legal forms removed ANYWHERE from `name_core` | 4, 4.5, 6.5 | word order is shuffled in S2/S3: `PVT. ASTOR TRADING LTD.`, `LLC Prairie Diana`. Replaces the earlier end-only rule |
| ambiguous short forms (`co`, `and co`, `cie`, `and cie`, `sa`, `sas`, `sci`, `snc`, `lp`, `pc`, `ei`) only at the start/end | 4.5 | `Maa Co Services` keeps `co`. v1.2.0 added `and cie` (French `& Cie` / `et Cie`): `Reso & Cie SAS` -> `reso` instead of `reso and` (18,883 France rows). Whole-token only, so `pharmacie`, `sciences` are untouched |
| Indic-script legal forms (private/limited/LLP in 9 scripts, `प्रा. लि.`) | 4.3, 4.5, 4.9 | hand list, checked against the most frequent non-Latin tail tokens of India S2/S3 names (unlabeled frequency) |
| `legal_families` = set of families | 4.5 | `Private Limited` = {PRIVATE, LIMITED}, so a truncated `Private` does not conflict |
| injected prefixes stripped from the START only: `M/s` (raw slash form), `Mr`, `Smt`, `Shri`, `Dr` | 4.7 | verified on train true pairs: each starts ~32k S2/S3 names whose S1 does not; <= 607 S1 names start with them. `Shree`/`Sri` kept (real names); a name is never emptied |
| trailing tags `#48688` removed from `name_core`, kept in `name_numbers` | 4.7 | 7.3k India train pairs, ~0 in S1 |
| domains/handles reduced to stem in `name_core`; `name_domain_stem`, `name_nospace` | 4.7 | 3-4% of S2/S3 names |
| French `et` -> `and` | 4.6 | S1 France uses `&`, S2/S3 `et` (1%) |
| generic words (services, group, exports, ...) NEVER removed | 6.5 | look-alikes are built by ADDING them (2.5-3x) |
| no transliteration; `name_script` only | 4.9 | |

### Addresses

| rule | EDA | notes |
|---|---|---|
| nothing built on postal codes | 5.1, 5.6 | <2% of addresses have one |
| `addr_house_number`: first number, skipping ordinals (`87th`, `2nd`, `1er`), US unit numbers, and the `1/2` in `10501 1/2` | 5, 6.5 | exact agreement is the strongest address signal |
| India injected leading clause (`Door No`, `H.no`, `Plot`, `Block`, `NO` + number): skipped when another candidate exists | 5, 6.5 | ~533k India train pairs start this way on S2/S3 |
| no house candidate -> use the unit number | Stage 3 | |
| a house number has no run of 3+ letters (v1.2.0) | samples | `N.H.7SALEMMAINROAD`, `lane4th`, `sec9`, `etsu513` are rejected; next valid number or null. 157,840 rows changed (100,867 to null). Their number part is still available in `addr_numbers` (v1.2.1, 272,802 rows changed there). Costs India 0.6 pp exact agreement on true pairs (glued tokens that were identical on both sides); the keep-digits alternative is in the Stage 3 ablations |
| unit words are US-style only; India `flat`/`shop` numbers stay house candidates | Stage 3 | excluding them dropped India exact agreement on true pairs from 0.622 to 0.589 |
| leading zeros stripped in each numeric part | 5 | `Rz-0040` = `Rz-40`, `03/C` = `3/C` |
| US abbreviations `st rd ave ln blvd cir hwy dr ct`; `st` street vs saint, `dr` drive vs Doctor, `fl` floor vs Florida, `ct` court (inside a clause) vs Connecticut (clause alone) | 5.5 | 69k vs 41k US S2 addresses |
| France `r`/`r.` -> rue (after a number, or clause-start before an article), `av`, `bd`, `pl`, `ch`, `imp`, `rte`; `N°`/`no` before a number dropped | 5.5, 8.4 | from unlabeled test text |
| India `ngr`, `mrg`, `sec`/`sect` + number -> nagar, marg, sector | 5.5 | |
| landmark clauses (`near`, `opp`, `behind`, ...) -> `addr_landmark`; street names like `Opp Avenue` protected | 5.4 | unchanged from v1.0 |
| ONE state/region clause -> `addr_state_raw`, removed from `addr_core`: the last clause if it qualifies, else the first qualifying clause (v1.2.0: `Washington, IN` keeps city `washington`; `new delhi, delhi` keeps `new delhi`; 1,311,529 rows changed, all previously multi-clause). Qualifying = whole-clause match against US states + codes, India states/UTs + vehicle codes, France regions + departments; or a clause wholly in a non-Latin script inside a Latin address | 5.3 | India S2 writes the state in native script (88% of such clauses are the last clause; the top values are all state names) |

## Stage 2

`state_map.parquet` maps every state spelling seen on the S2/S3 side of a train pair to the S1
spelling (US codes, India full names), kept when supported by >= 100 pairs with >= 90% agreement.
Examples: `texas` -> `tx`, `mh` / `महाराष्ट्र` -> `maharashtra`. Result in `addr_state_canon`
(sidecar). Coverage (v1.2.0): ~96-97% of US and India S2/S3 rows, France 0% by design (unknown). City aliases (Bombay/Mumbai) are not handled yet.

## Stage 3 results

See `reports/stage3_cleaning_eval.md` (v1.2.1; identical to v1.2.0, whose only change was `addr_numbers`, which no Stage 3 signal uses). Summary (true pairs vs hard negatives with the same
name_core):

- `addr_core` token Jaccard widens the gap over basic normalisation: US 0.535 -> 0.726, India 0.633 -> 0.723.
- House number exact agreement: US 0.722 vs 0.0007; India 0.616 vs 0.006.
- `legal_families` conflict: India true pairs 0.17% (legacy end-only class: 5.5%), hard negatives 18.8%
  (31.4%). The absolute gap is narrower, flagged in the report; kept because it removes almost all
  false conflicts on true pairs, which matters more for a veto feature.
- The glued-word house-number rule (v1.2.0) is flagged: India exact gap 0.616 -> 0.610. Kept as
  specified; switching to "keep the digits" (`chambers16/11` -> `16/11`) would give 0.617.
- Caveat: hard negatives are defined by equal `name_core`, so name-equality rows cannot be compared
  across normalisations on that set.
