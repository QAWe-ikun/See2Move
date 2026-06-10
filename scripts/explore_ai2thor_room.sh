#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCENE="${1:-FloorPlan1}"
python -m see2move.tools.explore_ai2thor --scene "$SCENE"
