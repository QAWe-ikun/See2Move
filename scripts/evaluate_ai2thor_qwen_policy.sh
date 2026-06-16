#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${1:-runs/ai2thor_policy_10k_stay_qwen3vl_e20/checkpoint_best.pt}"
RECORDS="${2:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl}"
DATA_DIR="${3:-/mnt/f/see2move/data/ai2thor_oracle_10k_stay}"
QWEN_FEATURES="${4:-/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt}"

python -m see2move.training.evaluate_qwen_policy \
  --checkpoint "$CHECKPOINT" \
  --records "$RECORDS" \
  --data-dir "$DATA_DIR" \
  --qwen-features "$QWEN_FEATURES"
