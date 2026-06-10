#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${1:-runs/ai2thor_policy/checkpoint_best.pt}"
RECORDS="${2:-data/ai2thor_oracle/records.jsonl}"
DATA_DIR="${3:-data/ai2thor_oracle}"

python -m see2move.training.evaluate_policy \
  --checkpoint "$CHECKPOINT" \
  --records "$RECORDS" \
  --data-dir "$DATA_DIR"
