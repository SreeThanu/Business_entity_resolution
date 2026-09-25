# Business Entity Resolution (Amazon ML Challenge)

Match each source-1 business to its records in sources 2 and 3.

Current status: data foundation only (parquet conversion, loaders, frozen splits). There is no
blocking, feature or model code yet.

## Layout

```
ber/                                  project ROOT (git repo)
  data_raw/dataset/                   fresh download, untouched          (gitignored)
  data/parquet/                       scripts/01 output                  (gitignored)
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
    utils/validate_submission.py      from the official download
    tests/test_data.py
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

# 2. Raw data: put the official dataset/ folder at <ROOT>/data_raw/dataset/
#    (train/{train_source1,2,3,train_ground_truth}.tsv, test/test_source{1,2,3}.tsv).
#    Verify it against the committed hashes in checksums/fresh_dataset.sha256 and check the
#    format. This must print "checksums OK" and "format OK" and exit 0. --verify writes nothing.
python scripts/00_check_raw.py --verify

# 3. TSV -> parquet (about 5 minutes, validates everything, fails loudly)
python scripts/01_convert_to_parquet.py

# 4. Splits are FROZEN. Do not regenerate them. If data/splits/*.txt are missing on
#    your machine, this recreates them byte-for-byte (it refuses to overwrite):
python scripts/02_make_splits.py

# 5. Tests (about 2-3 minutes; they also check the splits against data/splits/splits.sha256)
python -m pytest -v
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
  `"ehpad Club SAS`. Unquoting belongs in the normalisation step, which does not exist yet.
- Train contains only US and India. Test source 1 also contains France (259,452 rows).
- Evaluation convention: see `data/splits/SPLITS.md`. Validation S1 entities are searched
  against the full train S2/S3 pool.
