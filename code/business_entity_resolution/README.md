# Business Entity Resolution (Amazon ML Challenge)

Match each source-1 business to its records in sources 2 and 3.

Current status: data foundation and cleaning are done (parquet conversion, loaders, frozen splits,
Stage 1-3 cleaning; see CLEANING.md, including the "Cleaned data contract" for downstream code).
Blocking / candidate generation is in place (BLOCKING.md). There is no feature or model code yet.

## Layout

```
ber/                                  project ROOT (git repo)
  data/parquet/                       scripts/01 output                  (symlink to external SSD)
  data/clean/                         scripts/03 + 05 output             (symlink to external SSD)
  data/dicts/                         learned Stage 2 table              (symlink to external SSD)
  data/cand/                          candidates + vector cache (~21 GB) (symlink to external SSD)
  data/splits/                        frozen ID lists (gitignored); SPLITS.md, splits.sha256, log committed
  checksums/                          committed reference sha256 of the raw dataset + check log
  eda/                                EDA scripts + EDA_REPORT.md (ran on an old copy that is byte-identical to the fresh one)
  output/                             matching_results.tsv, candidate_pairs.tsv
  Documentation_template.md
  code/business_entity_resolution/
    src/ber/config.py                 every path, relative to ROOT (override with BER_ROOT)
    src/ber/data.py                   the ONLY way to load data
    src/ber/splits.py                 md5-based split rule
    scripts/00_check_raw.py           checksums + TSV field/UTF-8/row-count checks
    scripts/01_convert_to_parquet.py
    scripts/02_make_splits.py
    scripts/03_clean.py               Stage 1 cleaning -> data/clean/{split}_source{n}.parquet
    scripts/04_eval_cleaning.py       Stage 3 evaluation -> reports/stage3_cleaning_eval.md
    scripts/05_learn_tables.py        Stage 2 state table -> data/dicts, data/clean/stage2_*
    scripts/05_block.py               blocking -> data/cand/, output/candidate_pairs.tsv (run AFTER 05_learn_tables)
    src/ber/blocking.py               the four blocking passes, vector cache, recall helpers
    src/ber/normalize.py, lexicon.py  Stage 1 rules and hand-written lists
    src/ber/lookup.py                 Stage 2 learning (train_ids only)
    src/ber/memguard.py               abort cleanly if RSS > 5 GiB
    CLEANING.md                       every cleaning rule + the cleaned data contract
    utils/validate_submission.py      from the official download
    tests/test_data.py, tests/test_normalize.py
```

## Reproduce from scratch

All commands run from `code/business_entity_resolution/`. You do not need sudo; everything is
installed in a conda env.

```bash
# 1. Environment
conda create -y -n ber python=3.11
conda activate ber
pip install -r requirements.txt
pip install -e .

# 2. Raw data: the official dataset/ folder (train/{train_source1,2,3,train_ground_truth}.tsv,
#    test/test_source{1,2,3}.tsv). Default location (src/ber/config.py):
#      "/Volumes/thanu's T7/Business_entity_resolution/student_resource/dataset"
#    Elsewhere: export BER_RAW_DATASET_DIR=/path/to/dataset
#    Verify it against checksums/fresh_dataset.sha256 (prints "checksums OK" and "format OK").
python scripts/00_check_raw.py --verify           # ~15 s, writes nothing

# 3. TSV -> parquet (validates everything, fails loudly)
python scripts/01_convert_to_parquet.py           # ~5 min

# 4. Splits are FROZEN. Do not regenerate them. If data/splits/*.txt are missing on
#    your machine, this recreates them byte-for-byte (it refuses to overwrite):
python scripts/02_make_splits.py                  # < 1 min

# 5. Cleaning (all under a 5 GiB RSS guard; peak ~1.3 GiB on an 8 GB MacBook)
python scripts/03_clean.py                        # Stage 1, ~5 min for the 6 files
python scripts/05_learn_tables.py                 # Stage 2, ~15 s (needs Stage 1)
python scripts/04_eval_cleaning.py                # Stage 3 report, ~1 min

# 6. Blocking (see BLOCKING.md). Mac: query samples; Colab: --mode full
python scripts/05_block.py --split train --mode dev --queries train_ids   # first run ~1.5 h (builds caches)
python scripts/05_block.py --split train --mode dev --queries val_ids     # ~10 min once caches exist

# 7. Tests (~2 min; the raw row-count tests skip with a message if the raw dataset is missing)
python -m pytest -q
```

Run `00_check_raw.py` without `--verify` only to create new reference hashes, for example after an
official dataset update. It overwrites `checksums/fresh_dataset.sha256`.

## Loading data

```python
from ber import data

s1 = data.load_source("train", 1)                 # entity_id, business_name, business_address, country, source
gt = data.load_ground_truth(long=True)            # source1_entity_id, matched_entity_id (null = no match)
val_ids = data.load_split_ids("val_ids")
```

## Data notes

- Text is stored exactly as it appears in the TSVs, read with `quoting=csv.QUOTE_NONE`. There are
  823 rows (10 of them in train) that use RFC-4180 quoting, so their stored value keeps the
  wrapping quotes and the `""` escapes, for example `"""ehpad Club SAS"` where the logical value is
  `"ehpad Club SAS`. The cleaned columns delete all quote characters, so this is harmless.
- Train contains only US and India. Test source 1 also contains France (259,452 rows).
- Evaluation convention: see `data/splits/SPLITS.md`. Validation S1 entities are searched
  against the full train S2/S3 pool.
