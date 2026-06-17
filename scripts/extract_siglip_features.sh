#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${SIGLIP_MODEL_PATH:-/mnt/e/project/SceneReVis/ckpt/google/siglip-so400m-patch14-384}"
OUTPUT="${SIGLIP_FEATURES_PATH:-/mnt/f/see2move/features/siglip_so400m_ai2thor_oracle_10k_stay.pt}"

python -m see2move.tools.extract_clip_features \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT" \
  "$@"
