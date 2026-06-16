#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/train_ai2thor_qwen3vl.yaml}"
python -m see2move.training.train_qwen_policy --config "$CONFIG"
