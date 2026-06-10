#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-"$ROOT/data"}"
VLNCE_DIR="$ROOT/third_party/vln-ce"

SPLIT="${SPLIT:-val_unseen}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-1}"
VLNCE_WPN_CKPT="${VLNCE_WPN_CKPT:-"$DATA_ROOT/checkpoints/pretrained/WPN.pth"}"
RESULTS_DIR="${RESULTS_DIR:-"$ROOT/runs/vlnce_waypoint_${SPLIT}"}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

require_path "$VLNCE_DIR/run.py" "VLN-CE repository"
require_path "$VLNCE_WPN_CKPT" "VLN-CE waypoint checkpoint"
require_path "$DATA_ROOT/scene_datasets/mp3d" "Matterport3D scene root"
require_path "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/$SPLIT/${SPLIT}.json.gz" "R2R split data"
require_path "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/$SPLIT/${SPLIT}_gt.json.gz" "R2R split ground truth"
require_path "$DATA_ROOT/ddppo-models/gibson-2plus-resnet50.pth" "DDPPO depth checkpoint"

mkdir -p "$RESULTS_DIR"

cd "$VLNCE_DIR"
export PYTHONPATH="$VLNCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python run.py \
  --run-type eval \
  --exp-config vlnce_baselines/config/r2r_waypoint/2-wpn-dc.yaml \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  SIMULATOR_GPU_IDS "[$GPU_ID]" \
  TORCH_GPU_ID "$GPU_ID" \
  NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
  EVAL_CKPT_PATH_DIR "$VLNCE_WPN_CKPT" \
  RESULTS_DIR "$RESULTS_DIR" \
  EVAL.SPLIT "$SPLIT" \
  EVAL.SAMPLE False \
  EVAL.USE_CKPT_CONFIG False \
  TASK_CONFIG.DATASET.SCENES_DIR "$DATA_ROOT/scene_datasets/" \
  TASK_CONFIG.DATASET.DATA_PATH "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz" \
  TASK_CONFIG.TASK.NDTW.GT_PATH "$DATA_ROOT/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz" \
  MODEL.DEPTH_ENCODER.ddppo_checkpoint "$DATA_ROOT/ddppo-models/gibson-2plus-resnet50.pth"
