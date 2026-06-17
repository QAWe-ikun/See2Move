#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CASE_COUNT="${CASE_COUNT:-4}"
CASE_DIR="${CASE_DIR:-runs/case_studies_10k_stay_t200}"
DEMO_DIR="${DEMO_DIR:-runs/qwen_case_demos}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/case_studies_10k_stay_t200/panels}"
OUTPUT_NAME="${OUTPUT_NAME:-case_study_grid.png}"
RESULT_SOURCE="${RESULT_SOURCE:-prediction}"

for idx in $(seq 1 "$CASE_COUNT"); do
  python -m see2move.tools.record_qwen_case_demo \
    --case-dir "$CASE_DIR" \
    --output-dir "$DEMO_DIR" \
    --case-index "$idx"
done

python -m see2move.tools.render_case_study_panels \
  --case-dir "$CASE_DIR" \
  --demo-dir "$DEMO_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-cases "$CASE_COUNT" \
  --output-name "$OUTPUT_NAME" \
  --result-source "$RESULT_SOURCE"
