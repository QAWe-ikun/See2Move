#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${1:-runs/stay_threshold_sweep_10k_stay}"
CHECKPOINT="${2:-runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/checkpoint_best.pt}"
RECORDS="${3:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl}"
DATA_DIR="${4:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay}"
QWEN_FEATURES="${5:-/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt}"

mkdir -p "$OUTPUT_DIR"

thresholds=(0 50 100 200 300 500 800 1000)
files=()
for threshold in "${thresholds[@]}"; do
  output="$OUTPUT_DIR/t${threshold}.json"
  echo "Evaluating stay threshold ${threshold} -> ${output}"
  python -m see2move.training.evaluate_qwen_policy \
    --checkpoint "$CHECKPOINT" \
    --records "$RECORDS" \
    --data-dir "$DATA_DIR" \
    --qwen-features "$QWEN_FEATURES" \
    --stay-threshold "$threshold" \
    > "$output"
  files+=("$output")
done

python -m see2move.tools.summarize_evaluation_suite \
  "${files[@]}" \
  --output-json "$OUTPUT_DIR/summary.json" \
  --output-md "$OUTPUT_DIR/summary.md"

echo "Saved stay threshold sweep to ${OUTPUT_DIR}"
