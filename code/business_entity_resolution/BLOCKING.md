# Blocking / candidate generation

Code: `src/ber/blocking.py`, `scripts/05_block.py`, settings in `src/ber/config.py` (`BLOCK_*`).
Inputs: `data/clean` (NORMALIZE_VERSION 1.2.1, contract columns only) and `data/splits`.
Output: `data/cand/{split}_candidates.parquet`, `output/candidate_pairs.tsv` (test).

Labels (the ground truth) are read only by the evaluation in `05_block.py`; no pass uses them.

## Passes

Every pass is restricted to one country (true pairs always share it, EDA 3.5). Countries are read
from the data, so France (test only) is handled like any other country.

| pass | what | text / key | kept |
|---|---|---|---|
| P1 | char_wb 3-5-gram TF-IDF cosine, S1 -> S2/S3 | `name_core + " " + addr_core` | top-k1 per S1 |
| P2 | same, name only | `name_core` + `name_nospace` (if different) + `name_domain_stem` | top-k2 per S1 |
| P3 | exact key join | (house number, `addr_state_canon`) and (house number, each alphabetic `addr_core` token of >= 4 letters) | all pairs sharing a key; keys with > BLOCK_P3_MAX_BLOCK pool records are dropped |
| P4 | reverse P1, S2/S3 -> S1 | as P1 | pairs where the S1 is in the record's top-k4 |

TF-IDF: HashingVectorizer (2^24 buckets, char_wb 3-5, no sign flipping), IDF fit per split on
S1+S2+S3 of that split (unsupervised, also on test): idf = ln((1+N)/(1+df)) + 1, rows L2-normalised.

### How the search scales (measured on the Mac, 8 cores)

An exact cosine top-k is dominated by frequent n-grams ("roa", "ltd", "nag"): 12.9 ms per query per
1M pool rows, i.e. ~29 h for train US alone. So each pass does:

1. **retrieve** the top-M pool rows per query with sparse_dot_topn using only n-grams whose
   document frequency is <= BLOCK_DF_CAP x N (0.5% of the split's documents; ~40% of a query's
   n-grams survive), BLOCK_RETRIEVE_M rows per shard of 500k;
2. **re-rank** those rows by the exact cosine over all n-grams (numba sparse row-dot);
3. keep a running top-k per query by exact cosine across shards.

The stored `p1_cos` / `p2_cos` are exact cosines; `p1_rank` / `p2_rank` are ranks by exact cosine
within the retrieved set. The loss against an exact brute-force search is measured below.

Each text is hashed once into cached shards of raw counts (`data/cand/cache/{split}/{p1,p2}`,
~21 GB for train+test), so later runs only load them. Memory is bounded by BLOCK_QUERY_CHUNK and
BLOCK_SHARD_ROWS; peak RSS on the Mac stays under ~2.5 GB.

## Candidate file (`data/cand/{split}_candidates.parquet`)

One row per (S1, candidate) pair, unique.

| column | type | meaning |
|---|---|---|
| s1_id, cand_id | str | S1 entity and S2/S3 candidate |
| country | str | shared country (never crossed) |
| in_p1, in_p2, in_p3, in_p4 | bool | which passes produced the pair |
| p1_cos, p2_cos | float32 | exact P1 / P2 cosine, filled for EVERY pair (also for pairs another pass found) |
| p1_rank | int32 | rank of the candidate in the S1's P1 list (1 = best); null if not in P1 |
| p1_rank_rev | int32 | rank of the S1 in the candidate's reverse P1 list (P4); null if not in P4 |
| p2_rank | int32 | rank in the S1's P2 list; null if not in P2 |

Cosine is symmetric, so the P1 cosine is the same in both directions; the two ranks differ.

## Chosen settings (`src/ber/config.py`)

| setting | value | why |
|---|---|---|
| BLOCK_K p1 / p2 / p4 | 50 / 20 / 10 | best recall at mean <= ~100 candidates per S1 on the train_ids tuning sample (frontier below) |
| BLOCK_P4_MAX_PER_S1 | 100 | "hub" S1 records with short or generic text (`onyx`, `jalgaon adv`) sit in the reverse top-k of thousands of S2/S3 records; without the cap one S1 got 3,118 candidates. The cap costs ~0.15 pp entity recall |
| BLOCK_P3_MAX_BLOCK | 20 | keys shared by more pool records are dropped (34% of the sample's P3 key blocks, mostly a small house number + a city token) |
| BLOCK_DF_CAP / BLOCK_RETRIEVE_M | 0.005 / 300 (P4: 100) | within 0.1-0.3 pp of an exact search (table below). The first setting tried, 0.002 / 100, lost 2.5 pp in US |
| BLOCK_KMAX (dev only) | 200 / 100 / 20 | ranks kept in dev mode for the recall-vs-k curves |

Tuning used a 20,000-entity sample of train_ids (seed 0); the final numbers below are a
20,000-entity sample of val_ids with the same code and settings. Both search the FULL train S2+S3
pool of the query's country (SPLITS.md convention). Numbers on ALL of val_ids come from
`--mode full` on Colab (`reports/blocking_train_full_val_ids.md`, not run yet).

## Results on val_ids (20,000 S1 entities, final settings)

EDA baseline (raw text, P1 alone, k=50): 0.9451 overall, US 0.9757, India 0.8998.

| candidates | group | true_pairs | pair_recall | entities | entity_all_found |
|---|---|---|---|---|---|
| union (final k) | ALL | 69387 | 0.9800 | 18891 | 0.9416 |
| union (final k) | US | 41705 | 0.9926 | 11341 | 0.9736 |
| union (final k) | India | 27682 | 0.9611 | 7550 | 0.8934 |
| union (final k) | Indic-script S2/S3 names | 4814 | 0.8681 | 2444 | 0.8097 |
| P1 alone, k=50 (EDA baseline set-up) | ALL | 69387 | 0.9488 | 18891 | 0.8596 |
| P1 alone, k=50 (EDA baseline set-up) | India | 27682 | 0.9021 | 7550 | 0.7499 |
| P1 alone, k=50 (EDA baseline set-up) | US | 41705 | 0.9798 | 11341 | 0.9326 |
| P1 alone, k=50 (EDA baseline set-up) | Indic-script S2/S3 names | 4814 | 0.7073 | 2444 | 0.6060 |

### Candidates per S1 entity

| queries | pairs | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| 20000 | 1775509 | 88.7755 | 83.0000 | 137.0000 | 193 |

### Marginal contribution (true pairs no other pass found)

| pass | true_pairs_found | found_only_by_this_pass | marginal_recall | marginal_recall_indic | candidates_from_pass |
|---|---|---|---|---|---|
| p1 | 66721 | 169 | 0.0024 | 0.0029 | 1200082 |
| p2 | 50665 | 204 | 0.0029 | 0.0002 | 490732 |
| p3 | 44040 | 137 | 0.0020 | 0.0237 | 197648 |
| p4 | 67001 | 344 | 0.0050 | 0.0345 | 750098 |

P4 adds the most on its own (0.5 pp overall, 3.5 pp of Indic-script pairs); P3 finds 2.4 pp of the
Indic-script pairs that no text pass finds. The union misses 1,385 of 69,387 true pairs; 20 are
listed with their cleaned fields in `reports/blocking_train_val_ids_n20000.md`. Most are Indic-script
names whose addresses share only the house number and a city token (a P3 block larger than the cap),
or noisy S2/S3 names with an empty address.

## Tuning (train_ids sample)

### Recall vs k, each pass alone

| pass | k | pair_recall | recall_India | recall_US | mean_cands |
|---|---|---|---|---|---|
| p1 | 5 | 0.8261 | 0.7674 | 0.8655 | 4.9997 |
| p1 | 10 | 0.9139 | 0.8496 | 0.9570 | 9.9995 |
| p1 | 20 | 0.9325 | 0.8765 | 0.9701 | 19.9990 |
| p1 | 50 | 0.9483 | 0.9014 | 0.9798 | 49.9975 |
| p1 | 100 | 0.9563 | 0.9144 | 0.9843 | 99.9950 |
| p1 | 200 | 0.9631 | 0.9271 | 0.9873 | 199.9900 |
| p2 | 5 | 0.4635 | 0.4110 | 0.4988 | 4.9990 |
| p2 | 10 | 0.5572 | 0.4845 | 0.6060 | 9.9980 |
| p2 | 20 | 0.6191 | 0.5345 | 0.6759 | 19.9960 |
| p2 | 50 | 0.6844 | 0.6024 | 0.7395 | 49.9900 |
| p2 | 100 | 0.7341 | 0.6648 | 0.7806 | 99.9800 |
| p4 | 5 | 0.9585 | 0.9270 | 0.9797 | 22.8406 |
| p4 | 10 | 0.9641 | 0.9353 | 0.9834 | 45.8042 |
| p4 | 20 | 0.9684 | 0.9423 | 0.9860 | 92.5846 |

### Budget frontier (union; P3 on/off; P4 cap applied)

| k1 | k2 | k4 | p3 | pair_recall | entity_all_found | mean_cands |
|---|---|---|---|---|---|---|
| 20 | 0 | 0 | False | 0.9325 | 0.8205 | 19.9990 |
| 20 | 5 | 0 | False | 0.9414 | 0.8433 | 22.7352 |
| 20 | 0 | 3 | False | 0.9650 | 0.8989 | 25.5950 |
| 20 | 5 | 3 | False | 0.9692 | 0.9111 | 28.2316 |
| 20 | 10 | 3 | False | 0.9705 | 0.9148 | 32.3606 |
| 20 | 5 | 5 | False | 0.9712 | 0.9166 | 33.4100 |
| 20 | 5 | 3 | True | 0.9734 | 0.9221 | 35.1566 |
| 20 | 10 | 3 | True | 0.9746 | 0.9258 | 39.2854 |
| 20 | 5 | 5 | True | 0.9749 | 0.9267 | 40.2441 |
| 20 | 10 | 5 | True | 0.9759 | 0.9297 | 44.2936 |
| 20 | 20 | 3 | True | 0.9760 | 0.9299 | 48.4395 |
| 20 | 5 | 10 | True | 0.9765 | 0.9313 | 51.5659 |
| 30 | 10 | 5 | True | 0.9769 | 0.9325 | 52.8963 |
| 20 | 20 | 5 | True | 0.9771 | 0.9333 | 53.3218 |
| 20 | 10 | 10 | True | 0.9775 | 0.9341 | 55.4739 |
| 30 | 20 | 5 | True | 0.9780 | 0.9361 | 61.7959 |
| 30 | 10 | 10 | True | 0.9783 | 0.9366 | 63.2069 |
| 20 | 20 | 10 | True | 0.9786 | 0.9374 | 64.2618 |
| 30 | 20 | 10 | True | 0.9793 | 0.9398 | 71.9006 |
| 50 | 20 | 10 | True | 0.9801 | 0.9420 | 88.6076 |
| 100 | 20 | 10 | True | 0.9808 | 0.9442 | 133.4959 |

### Approximate retrieval vs exact search

| country | method | seconds | recall@10 | recall@50 | recall@100 |
|---|---|---|---|---|---|
| India | capped retrieval + exact re-rank | 47 | 0.8453 | 0.8950 | 0.9057 |
| India | exact search | 67 | 0.8450 | 0.8966 | 0.9084 |
| US | capped retrieval + exact re-rank | 49 | 0.9585 | 0.9814 | 0.9865 |
| US | exact search | 86 | 0.9595 | 0.9824 | 0.9880 |

### Pool size vs recall

Train pool subsampled (x0.5, x0.81 = train/test ratio) or enlarged to x1.23 with test S2/S3 records of the same country (the test pool has ~23% more S2/S3 per S1). True pairs whose record left the pool are excluded.

| country | pool_factor | pool_size | recall@10 | recall@20 | recall@50 | recall@100 |
|---|---|---|---|---|---|---|
| India | 0.5000 | 2066945 | 0.8782 | 0.8991 | 0.9173 | 0.9301 |
| India | 0.8100 | 3347528 | 0.8590 | 0.8834 | 0.9059 | 0.9196 |
| India | 1.0000 | 4133346 | 0.8496 | 0.8765 | 0.9014 | 0.9144 |
| India | 1.2300 | 5082722 | 0.8436 | 0.8714 | 0.8967 | 0.9102 |
| US | 0.5000 | 3092423 | 0.9697 | 0.9779 | 0.9850 | 0.9881 |
| US | 0.8100 | 5010306 | 0.9612 | 0.9732 | 0.9816 | 0.9857 |
| US | 1.0000 | 6186873 | 0.9570 | 0.9702 | 0.9798 | 0.9843 |
| US | 1.2300 | 7607162 | 0.9543 | 0.9680 | 0.9785 | 0.9834 |

Going from x1.0 to x1.23 (test's pool per S1) costs P1 at k=50 about 0.5 pp in India and 0.1 pp in
US. P1 at k=100 on the x1.23 pool is still above k=50 on x1.0, so scaling k1 with the pool (50 -> ~60
on test) would recover it for ~10 more candidates per S1. BLOCK_K is NOT scaled automatically; it
is a one-line change in config.py if we want it for test.

## Runtimes (Mac, 8 cores, 8 GB; peak RSS < 2.5 GB)

| step | time |
|---|---|
| vector cache, train + test, P1 + P2 (once) | ~8 min, 21 GB on the T7 |
| P3 for 20k queries | ~1 min |
| P4 reverse cache, train (every S2/S3 record; once per split) | India 39 min, US 65 min |
| P1 + P2 + cosine fill, 20k queries | ~7 min |
| exact check (2,000 queries) + pool experiment | ~15 min |

Full mode, estimated from these (not measured): P4 is the same order of work as a full forward P1
(all S1 x all S2/S3 of a country), so train ~4-5 h and test ~4 h on 8 cores; time scales roughly
with 1/cores. Output: ~89 candidates per S1, i.e. ~196M rows for train and ~155M for test.

## Commands

```bash
python scripts/05_block.py --split train --mode dev --queries train_ids --exact-check 2000 --pool-experiment
python scripts/05_block.py --split train --mode dev --queries val_ids
python scripts/05_block.py --split train --mode dev --queries val_ids --eval-only   # re-score saved candidates
python scripts/05_block.py --split train --mode full
python scripts/05_block.py --split test  --mode full   # also writes output/candidate_pairs.tsv, validated
```

## Running --full on Colab

Full mode searches every S1 entity (train 2.2M, test 1.7M) against its country's full S2+S3 pool.
It needs more cores and RAM than the Mac. Use a High-RAM CPU runtime (more cores = faster:
sparse_dot_topn and the numba kernel use every core).

Put on Google Drive first (from the Mac): `data/clean/` (the 6 Stage 1 files and 6 `stage2_*` files),
`data/parquet/train_ground_truth_long.parquet` (train recall report only) and `data/splits/`.
The vector cache (`data/cand/cache`, ~21 GB) is rebuilt on Colab in ~10 min; copying it is optional.

```bash
# 1. code + environment
!git clone <repo-url> ber && cd ber/code/business_entity_resolution && pip install -q -r requirements.txt && pip install -q -e .

# 2. data from Drive to the LOCAL disk (Drive-mounted paths are far too slow for random reads)
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/ber/data/parquet /content/ber/data/cand /content/ber/output
!cp -r /content/drive/MyDrive/ber_data/clean  /content/ber/data/clean
!cp -r /content/drive/MyDrive/ber_data/splits /content/ber/data/splits
!cp /content/drive/MyDrive/ber_data/parquet/train_ground_truth_long.parquet /content/ber/data/parquet/

# 3. run (BER_ROOT points config.py at /content/ber; raise the memory guard to the runtime's RAM)
%cd /content/ber/code/business_entity_resolution
!BER_ROOT=/content/ber BER_MEM_LIMIT_GIB=40 python scripts/05_learn_tables.py   # only if stage2_* were not copied
!BER_ROOT=/content/ber BER_MEM_LIMIT_GIB=40 python scripts/05_block.py --split train --mode full
!BER_ROOT=/content/ber BER_MEM_LIMIT_GIB=40 python scripts/05_block.py --split test  --mode full

# 4. outputs back to Drive
!mkdir -p /content/drive/MyDrive/ber_data/cand /content/drive/MyDrive/ber_output
!cp /content/ber/data/cand/*_candidates.parquet /content/drive/MyDrive/ber_data/cand/
!cp /content/ber/output/candidate_pairs.tsv /content/drive/MyDrive/ber_output/
!cp reports/blocking_train_full_val_ids.md /content/drive/MyDrive/ber_output/
```
