#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_CHECKPOINT="runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/checkpoint_best.pt"
DEFAULT_RECORDS="/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"
DEFAULT_DATA_DIR="/mnt/f/see2move/data/ai2thor_oracle_10k_stay"
DEFAULT_QWEN_FEATURES="/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt"

if [[ $# -gt 0 && "$1" == --* ]]; then
  CHECKPOINT="$DEFAULT_CHECKPOINT"
  RECORDS="$DEFAULT_RECORDS"
  DATA_DIR="$DEFAULT_DATA_DIR"
  QWEN_FEATURES="$DEFAULT_QWEN_FEATURES"
  EXTRA_ARGS=("$@")
else
  CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"
  RECORDS="${2:-$DEFAULT_RECORDS}"
  DATA_DIR="${3:-$DEFAULT_DATA_DIR}"
  QWEN_FEATURES="${4:-$DEFAULT_QWEN_FEATURES}"
  EXTRA_ARGS=("${@:5}")
fi

python -m see2move.training.evaluate_qwen_policy \
  --checkpoint "$CHECKPOINT" \
  --records "$RECORDS" \
  --data-dir "$DATA_DIR" \
  --qwen-features "$QWEN_FEATURES" \
  "${EXTRA_ARGS[@]}"
