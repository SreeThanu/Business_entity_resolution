# Running the pipeline on Colab

This runs features → training → evaluation → test prediction (`scripts/07-10`) on Colab Pro with
`colab/run_pipeline.ipynb`. The code comes from the private GitHub repo, and the data comes from the
shared Google Drive folder `ber_data`. Checkpoints, models and outputs go to a folder in **your own**
Drive (`ber_persist`, about 10 GB), so a disconnect loses at most the chunk that was running.

The feature set and model are a **v0 baseline** (`src/ber/features.py`, `FEATURE_VERSION = "v0.1"`). Replace or
extend them freely; see "Changing the features or model" below.

## 1. Add the shared data folder to your Drive

The data owner shares a Drive folder named `ber_data` with you (Viewer access is enough). A folder that
is shared with you does not appear under `/content/drive/MyDrive` in Colab until you add a shortcut:

1. Open [drive.google.com](https://drive.google.com), then **Shared with me** in the left sidebar.
2. Right-click `ber_data`, then **Organize → Add shortcut**, then choose **My Drive**, then **Add**.
3. Check that `ber_data` now shows up in **My Drive** with a small arrow on its icon.

In Colab it is now at `/content/drive/MyDrive/ber_data`. If you put the shortcut somewhere else, or gave it
another name, change `DRIVE_DATA` in the notebook's settings cell.

The folder has the `data/` layout: `clean/`, `splits/`, `dicts/`, `cand/` (three candidate files),
`parquet/train_ground_truth_long.parquet`. There are 29 files, 6.71 GB in total. The exact list, with the size
and sha256 of every file, is in `colab/data_manifest.json`. The notebook checks every file against it.

You need about 10 GB of free space in your own Drive for `ber_persist` (test features 4.5 GB, test
predictions 1.6 GB, submission files ~2 GB, the rest is small).

## 2. Create a GitHub token and add it to Colab Secrets

The repo is private, so the notebook needs a read-only token.

1. On GitHub: profile picture → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. Fill in:
   - **Token name:** `colab-ber-readonly`. **Expiration:** the end of the challenge.
   - **Resource owner:** the account that owns the repo (`SreeThanu`). If you are a collaborator and
     the owner is not listed, ask the owner to create the token, or to approve fine-grained tokens for the repo.
   - **Repository access:** *Only select repositories* → `Business_entity_resolution`.
   - **Permissions → Repository permissions → Contents: Read-only**. Leave everything else as *No access*
     (Metadata: Read-only is added automatically).
3. **Generate token** and copy it. GitHub shows it only once.
4. In Colab, open the notebook and click the **key icon (Secrets)** in the left sidebar → **Add new secret**:
   - Name: `GITHUB_TOKEN`, Value: the token.
   - Turn on **Notebook access** for this notebook.

The notebook reads the token with `google.colab.userdata.get("GITHUB_TOKEN")` and passes it to git as a
one-off HTTP header. It is never printed or written to disk, and it is not stored in the cloned repo's
`.git/config`. Never paste the token into a cell.

To open the notebook the first time: in Colab, **File → Open notebook → GitHub**, tick *Include private
repos*, authorise, and pick `code/business_entity_resolution/colab/run_pipeline.ipynb`. Alternatively,
download it from GitHub and use **File → Upload notebook**.

## 3. Choose the runtime

**Runtime → Change runtime type**: Python 3, hardware accelerator **CPU**, **High-RAM** on.

LightGBM trains on the CPU, so a GPU would only use up compute units. High-RAM matters: the entity tables
plus the training matrix need ~15-20 GB, and the official validator on the full `candidate_pairs.tsv` needs
up to ~20 GB more. The notebook's runtime cell prints RAM, CPU count and GPU, and warns if RAM is below 40 GiB.
If you have to use a standard runtime, add `--skip-candidate-validation` to the `10_predict.py` cell. The check
that every match is in `candidate_pairs.tsv` still runs.

## 4. Run the cells in order

| cell | what it does | time |
|---|---|---|
| 0. Settings | repo, branch, folders; defines `run()` (streams output, stops on failure) | - |
| Runtime check | RAM / CPUs / GPU | - |
| 1. Mount Drive | asks for Drive permission; checks that `ber_data` is visible | - |
| 2. Clone or update | clones the repo into `/content/ber`, or fast-forwards it | < 1 min |
| 3. Install | `requirements-colab.txt` + the `ber` package | ~1 min |
| 4. Copy + verify data | Drive → `/content/ber_data`, sha256 of every file | ~5-10 min |
| 5. Env vars | `BER_ROOT`, `BER_DATA_DIR`, `BER_PERSIST_DIR`, `BER_OUTPUT_DIR`, memory guard | - |
| 07 train / val / test | pair features, one parquet per 20,000 S1 | ~15 min / ~5 min / ~2-2.5 h |
| 08 | LightGBM with early stopping on val | ~30-60 min |
| 09 | decision rule, threshold search, per-country and cross-country report | ~30-60 min |
| 10 | test inference, `matching_results.tsv`, `candidate_pairs.tsv`, validation | ~20-30 min |

The times are estimates from the Mac measurement (about 75 s per 1.8M-pair chunk on 8 cores). The logs print
the time of every chunk.

### After a disconnect

Run all the cells again from the top (**Runtime → Run all** works). The VM's local disk is wiped when it
resets, so cells 2-4 clone the code, install it and copy the data again. Every stage then resumes from its
checkpoints on Drive:

- **07** writes one file per chunk (`ber_persist/features/{split}/part_NNNNN.parquet`), each written under
  a temp name and then renamed. Chunks that already exist are skipped, so a disconnect loses at most the
  chunk that was running. `_DONE.json` marks a finished split, and re-running a finished split does nothing.
- **08** skips training if a finished model with the same code commit, features and parameters already
  exists. A model folder is renamed into place only once it is complete. `--force` trains again.
- **09** reuses `val_preds.parquet` and the cross-country results. Once `eval_report.md` exists, re-running
  only prints the report.
- **10** skips test chunks that are already scored (`ber_persist/preds/test/<model>/`), then redoes the
  cheap decision and validation steps.

Each chunk folder records its settings in `run.json`. If you change the feature code (`FEATURE_VERSION`) or
the candidate file, a stage refuses to mix old and new chunks. Re-run that stage with `--force` to rebuild it.

## 5. Where the outputs land

All of these are in your Drive under `ber_persist/`:

| path | content |
|---|---|
| `output/matching_results.tsv` | the submission file (also `matching_results_<model>.tsv`) |
| `output/candidate_pairs.tsv` | the candidate set (needed in the final zip) |
| `output/predict_summary.json` | model, threshold, match counts per country |
| `models/<YYYYmmdd-HHMMSS>_<commit>/` | `model.txt`, `features.json`, `params.json`, `meta.json` (commit, data fingerprints), `metrics.json`, `importance.csv`, `threshold.json`, `eval_report.md`, `val_preds.parquet`, `cross_*.json` |
| `features/{train,val,test}/` | feature checkpoints (you can delete them after 10 finishes) |
| `preds/test/<model>/` | test probability checkpoints |

Both TSVs are also in `/content/ber/output/` until the runtime is recycled. The last notebook cell flushes
and unmounts Drive so that every write is uploaded.

## What the stages do

- **07_features.py `--split {train,val,test}`**: pairs from `cand_train_200k` / `cand_val_50k` / test
  candidates, joined to the cleaned entity tables. The v0 features are: name token-set ratio, Jaro-Winkler,
  character 3-gram and token Jaccard, `name_core` equality, `name_nospace`↔domain-stem match, added and dropped
  core tokens; legal families agree / conflict / unknown; `addr_core` token and 3-gram Jaccard, house number
  exact / gap / missing, `addr_numbers` overlap, state agree / disagree / unknown (null, France included, means
  unknown); script flags; blocking pass flags, cosines and ranks (a null rank becomes k+1), per-S1 rank and gap;
  chain-name counts (how many S1 of the country share the name). Country itself is never a feature.
  The candidate pairs are streamed one chunk at a time. Only the entity tables are held in memory.
- **08_train.py**: LightGBM binary classifier. The labels come from `train_ground_truth_long.parquet`, and
  early stopping uses val logloss.
- **09_evaluate.py**: applies the decision rule on val: **one-to-one** (each S2/S3 record goes only to its
  highest-probability S1), then probability ≥ threshold. The threshold is searched for the best **macro-F0.5**,
  using the official formula over all 50k val S1, where singletons score 1 on an empty prediction. It reports
  results overall and per country, plus a cross-country check (train on US only → evaluate on India, and the
  reverse). The chosen threshold is saved as `threshold.json`.
- **10_predict.py**: scores the test chunks, applies the same rule with the saved threshold, writes both
  TSVs, fails if a match is missing from `candidate_pairs.tsv`, and runs `utils/validate_submission.py`.

## Changing the features or model

- Features: edit `src/ber/features.py`, bump `FEATURE_VERSION`, and run the 07 cells with `--force`
  (`run("python scripts/07_features.py --split train --force")`, and the same for val and test).
- Model parameters: `scripts/08_train.py` flags (`--learning-rate`, `--num-leaves`, `--max-rounds`), or
  edit `default_params`.
- A specific model for 09 / 10: `--model <folder name>` or `os.environ["BER_MODEL_ID"] = "<folder name>"`.
- Before pushing a change, run the local smoke test on the Mac: `bash scripts/smoke_colab_flow.sh`. It runs
  the same four stages on a tiny sample in a temp folder, in about 1 minute.

## For the data owner: updating the Drive folder

```bash
python scripts/pack_for_drive.py                           # re-hash, rewrite colab/data_manifest.json
python scripts/pack_for_drive.py --copy-to <staging dir>   # optional: the exact files, in the data/ layout
```

Upload the files into the shared `ber_data` folder, keeping the same sub-folders, and commit the new
manifest. If a file on Drive does not match the committed manifest, the notebook stops at cell 4.
