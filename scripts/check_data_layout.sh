#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-"$ROOT/data"}"
VLNCE_SPLIT="${SPLIT:-val_unseen}"
SMARTWAY_SPLIT="${SMARTWAY_SPLIT:-rand100}"

status=0

check_required() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" ]]; then
    printf '[ok]      %s\n' "$label"
  else
    printf '[missing] %s\n         %s\n' "$label" "$path"
    status=1
  fi
}

check_optional() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" ]]; then
    printf '[ok]      %s\n' "$label"
  else
    printf '[optional] %s\n           %s\n' "$label" "$path"
  fi
}

printf 'Repo root:  %s\n' "$ROOT"
printf 'Data root:  %s\n\n' "$DATA_ROOT"

check_required "$ROOT/third_party/vln-ce/run.py" "VLN-CE repository"
check_required "$ROOT/third_party/smartway-code/run.py" "SmartWay repository"

printf '\nShared data\n'
check_required "$DATA_ROOT/scene_datasets/mp3d" "Matterport3D scene root"
check_required "$DATA_ROOT/ddppo-models/gibson-2plus-resnet50.pth" "VLN-CE DDPPO depth checkpoint"
check_required "$DATA_ROOT/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth" "SmartWay DDPPO depth checkpoint"

printf '\nVLN-CE data\n'
check_required "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/$VLNCE_SPLIT/${VLNCE_SPLIT}.json.gz" "R2R v1-3 split data"
check_required "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/$VLNCE_SPLIT/${VLNCE_SPLIT}_gt.json.gz" "R2R v1-3 split ground truth"
check_required "$DATA_ROOT/checkpoints/pretrained/WPN.pth" "VLN-CE waypoint checkpoint"

printf '\nSmartWay data\n'
check_required "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/$SMARTWAY_SPLIT/${SMARTWAY_SPLIT}_bertidx.json.gz" "R2R v1-2 BERTidx split data"
check_required "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed/$SMARTWAY_SPLIT/${SMARTWAY_SPLIT}_gt.json.gz" "R2R v1-2 split ground truth"
check_required "$ROOT/third_party/smartway-code/waypoint_predictor/checkpoints/final-camera-ready" "SmartWay waypoint predictor checkpoint"
check_required "$ROOT/third_party/smartway-code/ram_plus_swin_large_14m.pth" "RAM+ checkpoint"
check_required "$ROOT/third_party/smartway-code/recognize-anything" "Recognize Anything repository"
check_optional "$ROOT/third_party/smartway-code/logs/checkpoints/evaluation" "SmartWay policy checkpoint directory"

printf '\n'
if [[ "$status" -eq 0 ]]; then
  printf 'All required paths are present.\n'
else
  printf 'Some required paths are missing. See docs/WSL_REPRODUCTION.md.\n'
fi

exit "$status"
