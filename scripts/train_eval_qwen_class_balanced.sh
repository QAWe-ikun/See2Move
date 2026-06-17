#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/ablations/train_qwen_class_balanced.yaml}"
OUTPUT="${2:-runs/ablation_eval_10k_stay_t200/qwen_class_balanced.json}"
mkdir -p "$(dirname "$OUTPUT")"

bash scripts/train_ai2thor_qwen_policy.sh "$CONFIG"

python -m see2move.training.evaluate_qwen_policy \
  --checkpoint runs/ablations_10k_stay/qwen_class_balanced/checkpoint_best.pt \
  --stay-threshold 200 > "$OUTPUT"

cat "$OUTPUT"
