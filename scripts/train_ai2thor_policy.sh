#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/train_ai2thor_policy.yaml}"
python -m see2move.training.train_policy --config "$CONFIG"
