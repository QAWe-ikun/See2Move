#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-"$ROOT/data"}"
SMARTWAY_DIR="$ROOT/third_party/smartway-code"

EXP_NAME="${EXP_NAME:-evaluation}"
SMARTWAY_SPLIT="${SMARTWAY_SPLIT:-rand100}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-1}"
SMARTWAY_EVAL_CKPT_PATH="${SMARTWAY_EVAL_CKPT_PATH:-"$ROOT/runs/smartway_zero_shot/no_policy_checkpoint"}"
RESULTS_DIR_BASE="${RESULTS_DIR_BASE:-"$ROOT/runs/smartway_eval_results/"}"
CHECKPOINT_FOLDER_BASE="${CHECKPOINT_FOLDER_BASE:-"$ROOT/runs/smartway_checkpoints/"}"
TENSORBOARD_DIR_BASE="${TENSORBOARD_DIR_BASE:-"$ROOT/runs/smartway_tensorboard/"}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 1
  fi
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'Missing OPENAI_API_KEY. SmartWay uses the OpenAI API for MLLM navigation.\n' >&2
  exit 1
fi

require_path "$SMARTWAY_DIR/run.py" "SmartWay repository"
require_path "$DATA_ROOT/scene_datasets/mp3d" "Matterport3D scene root"
require_path "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/$SMARTWAY_SPLIT/${SMARTWAY_SPLIT}_bertidx.json.gz" "R2R BERTidx split data"
require_path "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed/$SMARTWAY_SPLIT/${SMARTWAY_SPLIT}_gt.json.gz" "R2R split ground truth"
require_path "$DATA_ROOT/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth" "SmartWay DDPPO depth checkpoint"
require_path "$SMARTWAY_DIR/waypoint_predictor/checkpoints/final-camera-ready" "SmartWay waypoint predictor checkpoint"
require_path "$SMARTWAY_DIR/ram_plus_swin_large_14m.pth" "RAM+ checkpoint"
require_path "$SMARTWAY_DIR/recognize-anything" "Recognize Anything repository"

mkdir -p "$RESULTS_DIR_BASE" "$CHECKPOINT_FOLDER_BASE" "$TENSORBOARD_DIR_BASE"

cd "$SMARTWAY_DIR"
export PYTHONPATH="$SMARTWAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python run.py \
  --exp_name "$EXP_NAME" \
  --run-type eval \
  --exp-config run_VLNBERT.yaml \
  SIMULATOR_GPU_IDS "[$GPU_ID]" \
  TORCH_GPU_ID "$GPU_ID" \
  TORCH_GPU_IDS "[$GPU_ID]" \
  GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
  EVAL.SPLIT "$SMARTWAY_SPLIT" \
  EVAL_CKPT_PATH_DIR "$SMARTWAY_EVAL_CKPT_PATH" \
  RESULTS_DIR "$RESULTS_DIR_BASE" \
  CHECKPOINT_FOLDER "$CHECKPOINT_FOLDER_BASE" \
  TENSORBOARD_DIR "$TENSORBOARD_DIR_BASE" \
  TASK_CONFIG.DATASET.SCENES_DIR "$DATA_ROOT/scene_datasets/" \
  TASK_CONFIG.DATASET.DATA_PATH "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz" \
  TASK_CONFIG.TASK.NDTW.GT_PATH "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz" \
  TASK_CONFIG.TASK.SDTW.GT_PATH "$DATA_ROOT/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz" \
  MODEL.DEPTH_ENCODER.ddppo_checkpoint "$DATA_ROOT/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth"
