#!/usr/bin/env bash
# Local smoke test of the Colab flow: the same four stages (07 -> 08 -> 09 -> 10) on a tiny dev
# sample, with every output in a temp folder (BER_DATA_DIR, BER_PERSIST_DIR, BER_OUTPUT_DIR), so
# the real data/, artifacts/ and output/ are never touched.
#
#   bash scripts/smoke_colab_flow.sh            # from code/business_entity_resolution/
#   KEEP=1 bash scripts/smoke_colab_flow.sh     # keep the temp folder for inspection
#
# It also re-runs every stage once to check they resume / skip instead of redoing work.
set -euo pipefail
cd "$(dirname "$0")/.."

TMP="$(mktemp -d "${TMPDIR:-/tmp}/ber_smoke.XXXXXX")"
if [[ -z "${KEEP:-}" ]]; then trap 'rm -rf "$TMP"' EXIT; fi
echo "== smoke dir: $TMP"

echo "== 0. tiny sample of the real data"
python scripts/make_dev_sample.py --out "$TMP/data" --n-train 400 --n-val 200 --n-test 300

export BER_DATA_DIR="$TMP/data" BER_PERSIST_DIR="$TMP/persist" BER_OUTPUT_DIR="$TMP/output"
export BER_FEATURE_CHUNK_S1=150        # several chunks even on a tiny sample

echo "== 07 features"
for split in train val test; do python scripts/07_features.py --split "$split"; done
echo "== 08 train"
python scripts/08_train.py --max-rounds 300 --early-stop 30
echo "== 09 evaluate"
python scripts/09_evaluate.py
echo "== 10 predict"
python scripts/10_predict.py

echo "== resume check: re-running every stage must skip finished work"
rm "$BER_PERSIST_DIR"/features/test/part_00001.parquet          # simulate a lost chunk
python scripts/07_features.py --split test | tee "$TMP/rerun07.log"
grep -q "1/1 this run" "$TMP/rerun07.log"
python scripts/09_evaluate.py > /dev/null
python scripts/10_predict.py | tee "$TMP/rerun10.log"
grep -q "scored earlier" "$TMP/rerun10.log"

echo "== outputs"
ls -la "$BER_OUTPUT_DIR" "$BER_PERSIST_DIR"/models/*/ "$BER_PERSIST_DIR/output"
head -3 "$BER_OUTPUT_DIR/matching_results.tsv"
echo "== SMOKE TEST PASSED"
