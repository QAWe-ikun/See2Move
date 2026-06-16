#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-simple}"

run_simple() {
  local configs=(
    configs/ablations/train_simple_no_depth.yaml
    configs/ablations/train_simple_no_text.yaml
    configs/ablations/train_simple_no_pose.yaml
    configs/ablations/train_simple_no_rgb.yaml
    configs/ablations/train_simple_rgb_only.yaml
  )
  for config in "${configs[@]}"; do
    bash scripts/train_ai2thor_policy.sh "$config"
  done
}

run_qwen() {
  local configs=(
    configs/ablations/train_qwen_no_depth.yaml
    configs/ablations/train_qwen_no_pose.yaml
    configs/ablations/train_qwen_no_qwen.yaml
  )
  for config in "${configs[@]}"; do
    bash scripts/train_ai2thor_qwen_policy.sh "$config"
  done
}

case "$MODE" in
  simple)
    run_simple
    ;;
  qwen)
    run_qwen
    ;;
  all)
    run_simple
    run_qwen
    ;;
  *)
    echo "Usage: bash scripts/run_ai2thor_ablations.sh [simple|qwen|all]" >&2
    exit 2
    ;;
esac

bash scripts/compare_ai2thor_runs.sh
