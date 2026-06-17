#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${1:-runs/learned_baseline_eval_10k_stay_t200}"
STAY_THRESHOLD="${2:-200}"
RECORDS="${3:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl}"
DATA_DIR="${4:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay}"
QWEN_FEATURES="${5:-/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt}"
SIGLIP_FEATURES="${6:-/mnt/f/see2move/features/siglip_so400m_ai2thor_oracle_10k_stay.pt}"

mkdir -p "$OUTPUT_DIR"

echo "Evaluating qwen3vl_full"
python -m see2move.training.evaluate_qwen_policy \
  --checkpoint runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/checkpoint_best.pt \
  --records "$RECORDS" \
  --data-dir "$DATA_DIR" \
  --features "$QWEN_FEATURES" \
  --stay-threshold "$STAY_THRESHOLD" \
  > "$OUTPUT_DIR/qwen3vl_full.json"

echo "Evaluating siglip_full"
python -m see2move.training.evaluate_qwen_policy \
  --checkpoint runs/ai2thor_policy_10k_stay_siglip_gain_score_e25/checkpoint_best.pt \
  --records "$RECORDS" \
  --data-dir "$DATA_DIR" \
  --features "$SIGLIP_FEATURES" \
  --stay-threshold "$STAY_THRESHOLD" \
  > "$OUTPUT_DIR/siglip_full.json"

python -m see2move.tools.summarize_evaluation_suite \
  "$OUTPUT_DIR/qwen3vl_full.json" \
  "$OUTPUT_DIR/siglip_full.json" \
  --output-json "$OUTPUT_DIR/summary.json" \
  --output-md "$OUTPUT_DIR/summary.md"

bash scripts/evaluate_oracle_baselines.sh \
  --records "$RECORDS" \
  --output-dir "$OUTPUT_DIR/non_learned"

echo "Saved learned baseline evaluation to ${OUTPUT_DIR}"
