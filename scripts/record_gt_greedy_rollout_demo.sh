#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m see2move.tools.record_qwen_greedy_rollout_demo \
  --policy gt \
  --output-dir runs/gt_greedy_rollout_demo \
  --video-name gt_greedy_rollout_demo.mp4 \
  --case-indices 1,2,3,4,5,6,7,8 \
  "$@"
