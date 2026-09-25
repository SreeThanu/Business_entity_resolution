# Validation splits (FROZEN)

**These splits are frozen and must never be regenerated or edited.** Every experiment
evaluates on these exact ID lists so results stay comparable. `scripts/02_make_splits.py` refuses
to overwrite them. The ID lists are gitignored; `splits.sha256` (committed) is the reference, and
`tests/test_data.py` checks the files on disk against it.

## How they were made

- Universe: all 2,206,821 **train source-1** entities (`train_source1.tsv`).
- An entity goes to validation iff
  `int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % 100 < 20`
  (see `src/ber/splits.py`). This depends only on the ID string, so it gives the same result on
  every machine and process. Python's built-in `hash()` is salted per process and is never used.
- Cross-country folds come from the `country` column (no hard-coded list). For each
  country C: `fold_C_train` = C entities in `train_ids`, `fold_C_val` = C entities in `val_ids`.
  Train currently has two countries, US and India, so there are four fold files. Cross-country
  experiments: train on `fold_us_train` and evaluate on `fold_india_val`, or train on
  `fold_india_train` and evaluate on `fold_us_val`.
- Created 2026-09-25 from the fresh download (sha256 in `checksums/fresh_dataset.sha256`).

## Contents

| file | S1 entities | singleton % | mean matches (matched entities) |
|---|---|---|---|
| train_ids.txt | 1,766,455 | 5.60 | 3.667 |
| val_ids.txt | 440,366 | 5.53 | 3.664 |
| fold_us_train.txt | 1,059,554 | 5.60 | 3.664 |
| fold_us_val.txt | 264,079 | 5.52 | 3.661 |
| fold_india_train.txt | 706,901 | 5.60 | 3.670 |
| fold_india_val.txt | 176,287 | 5.54 | 3.667 |

EDA reference: 5.58% singletons, 3.67 mean matches. Holdout rate: US 19.95%, India 19.96%,
overall 19.95%.

## Evaluation convention

Validation S1 entities are searched against the **full train S2 and S3 pool** (all of
`train_source2` and `train_source3`), **not** against a pool filtered down to the validation
entities' matches. This mimics test, where candidates come from the whole test S2/S3 pool. The S2/S3
records matched to validation entities are also candidates when training on `train_ids`. That is
expected, because S2/S3 records are shared and only the S1 entities are split.

## Files

One sorted train-S1 `entity_id` per line. Load with `ber.data.load_split_ids("val_ids")`.
