#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${1:-runs/ablation_eval_10k_stay_t200}"
STAY_THRESHOLD="${2:-200}"
RECORDS="${3:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl}"
DATA_DIR="${4:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay}"
QWEN_FEATURES="${5:-/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt}"

mkdir -p "$OUTPUT_DIR"

run_eval() {
  local name="$1"
  local checkpoint="$2"
  local output="$OUTPUT_DIR/${name}.json"
  echo "Evaluating ${name} -> ${output}"
  python -m see2move.training.evaluate_qwen_policy \
    --checkpoint "$checkpoint" \
    --records "$RECORDS" \
    --data-dir "$DATA_DIR" \
    --qwen-features "$QWEN_FEATURES" \
    --stay-threshold "$STAY_THRESHOLD" \
    > "$output"
}

run_eval "full" "runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/checkpoint_best.pt"
run_eval "no_depth" "runs/ablations_10k_stay/qwen_no_depth/checkpoint_best.pt"
run_eval "no_pose" "runs/ablations_10k_stay/qwen_no_pose/checkpoint_best.pt"
run_eval "no_qwen" "runs/ablations_10k_stay/qwen_no_qwen/checkpoint_best.pt"

python -m see2move.tools.summarize_evaluation_suite \
  "$OUTPUT_DIR/full.json" \
  "$OUTPUT_DIR/no_depth.json" \
  "$OUTPUT_DIR/no_pose.json" \
  "$OUTPUT_DIR/no_qwen.json" \
  --output-json "$OUTPUT_DIR/summary.json" \
  --output-md "$OUTPUT_DIR/summary.md"

echo "Saved ablation evaluation results to ${OUTPUT_DIR}"
