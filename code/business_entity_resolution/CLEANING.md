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
  `entity_id`; it never rewrites a Stage 1 column. (v1.1.1 replaced v1.1.0 within the same session,
  before anything consumed it; the change is the house-number fix described below.)
- Country is an open set (France is test-only [2, 8.4]). No rule reads the `country` column in
  Stage 1; rules are keyed on token shapes, so on unseen text they do nothing
  (`test_unseen_text_is_noop`). Stage 2 maps only countries it has labels for; France gets
  `addr_state_canon = null` (unknown), never a mismatch.
- No external data or APIs. The hand-written lists in `src/ber/lexicon.py` are general language
  knowledge: legal forms, honorifics, state/region names and codes, street abbreviations.
- **France rules came from inspecting unlabeled test text** (there is no France training data):
  French legal forms, `r`/`av`/`bd`/`pl`/`ch`/`imp`/`rte`, `N°`, `et`, the France region/department list.
- Memory guard (`src/ber/memguard.py`): scripts abort cleanly if RSS > 5 GiB. Full Stage 1 peaks at
  1.5 GiB; Stage 2 at 1.1 GiB.

## Stage 1 columns

| column | meaning |
|---|---|
| `name_clean` | full cleaned name (common steps below + `et` -> `and`) |
| `name_core` | name without legal forms, injected prefixes, `#tags`, domain TLDs |
| `name_nospace` | `name_core` without spaces (compare against glued domains/handles) |
| `name_domain_stem` | `sarthitrading` from `sarthitrading.com` / `@sarthi_trading`, else null |
| `legal_families` | sorted SET of legal families found (`["LIMITED", "PRIVATE"]`), possibly empty |
| `legal_suffix_class` | legacy single class of the last legal form (kept for compatibility) |
| `name_script`, `name_has_nonlatin`, `name_numbers` | dominant script, any non-Latin letter, digit runs (incl. tags) |
| `addr_clean` | cleaned address, commas removed |
| `addr_numbers` | every number-like token, leading zeros stripped per numeric part |
| `addr_house_number` | first street-clause number, see rules |
| `addr_unit` | number after a unit word (unit/suite/ste/apt/fl/floor/room) |
| `addr_landmark`, `addr_no_landmark` | landmark clauses (India) and the address without them |
| `addr_std`, `addr_std_components` | abbreviations canonicalised, per comma clause |
| `addr_state_raw` | the state/region clause(s), as written (not canonicalised) |
| `addr_core` | `addr_std` without state clauses, leading zeros stripped |
| `addr_script`, `addr_has_nonlatin` | as for names |

Empty addresses (after filler removal) give null in EVERY `addr_*` column, never `""` [1.3].

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
| ambiguous short forms (`co`, `and co`, `cie`, `sa`, `sas`, `sci`, `snc`, `lp`, `pc`, `ei`) only at the start/end | 4.5 | `Maa Co Services` keeps `co` |
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
| unit words are US-style only; India `flat`/`shop` numbers stay house candidates | Stage 3 | excluding them dropped India exact agreement on true pairs from 0.622 to 0.589 |
| leading zeros stripped in each numeric part | 5 | `Rz-0040` = `Rz-40`, `03/C` = `3/C` |
| US abbreviations `st rd ave ln blvd cir hwy dr ct`; `st` street vs saint, `dr` drive vs Doctor, `fl` floor vs Florida, `ct` court (inside a clause) vs Connecticut (clause alone) | 5.5 | 69k vs 41k US S2 addresses |
| France `r`/`r.` -> rue (after a number, or clause-start before an article), `av`, `bd`, `pl`, `ch`, `imp`, `rte`; `N°`/`no` before a number dropped | 5.5, 8.4 | from unlabeled test text |
| India `ngr`, `mrg`, `sec`/`sect` + number -> nagar, marg, sector | 5.5 | |
| landmark clauses (`near`, `opp`, `behind`, ...) -> `addr_landmark`; street names like `Opp Avenue` protected | 5.4 | unchanged from v1.0 |
| state/region clause -> `addr_state_raw`, removed from `addr_core`: whole-clause match against US states + codes, India states/UTs + vehicle codes, France regions + departments; or a clause wholly in a non-Latin script inside a Latin address | 5.3 | India S2 writes the state in native script (88% of such clauses are the last clause; the top values are all state names) |

## Stage 2

`state_map.parquet` maps every state spelling seen on the S2/S3 side of a train pair to the S1
spelling (US codes, India full names), kept when supported by >= 100 pairs with >= 90% agreement.
Examples: `texas` -> `tx`, `mh` / `महाराष्ट्र` -> `maharashtra`. Result in `addr_state_canon`
(sidecar). Coverage: US ~96%, India ~85% of S2/S3 rows (the rest have several state-like clauses or
rare spellings), France 0% by design. City aliases (Bombay/Mumbai) are not handled yet.

## Stage 3 results

See `reports/stage3_cleaning_eval.md`. Summary (true pairs vs hard negatives with the same
name_core):

- `addr_core` token Jaccard widens the gap over basic normalisation: US 0.535 -> 0.726, India 0.633 -> 0.723.
- House number exact agreement: US 0.721 vs 0.0007; India 0.622 vs 0.006.
- `legal_families` conflict: India true pairs 0.17% (legacy end-only class: 5.5%), hard negatives 18.8%
  (31.4%). The absolute gap is narrower, flagged in the report; kept because it removes almost all
  false conflicts on true pairs, which matters more for a veto feature.
- Caveat: hard negatives are defined by equal `name_core`, so name-equality rows cannot be compared
  across normalisations on that set.
