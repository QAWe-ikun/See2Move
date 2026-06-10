# WSL Reproduction

Workspace path in WSL:

```bash
cd /mnt/e/project/see2move
```

Use `DATA_ROOT` if datasets are stored outside the repo:

```bash
export DATA_ROOT=/path/to/data
```

## VLN-CE

Runtime used by the upstream project:

- Python 3.6
- Habitat-Sim 0.1.7
- Habitat-Lab 0.1.7
- R2R_VLNCE_v1-3_preprocessed
- Matterport3D scenes

Compatibility note:

- `third_party/vln-ce/requirements.txt` pins `opencv-python-headless==4.5.5.64`.
- PyTorch is installed by the upstream pip requirements; configure your network or pip mirror before installing.

Run:

```bash
bash scripts/run_vlnce_waypoint_eval.sh
```

## SmartWay

Runtime used by the upstream project:

- Python 3.8.20
- PyTorch 2.1.1
- Habitat-Sim 0.1.7
- Habitat-Lab 0.1.7
- R2R_VLNCE_v1-2_preprocessed
- R2R_VLNCE_v1-2_preprocessed_BERTidx
- Matterport3D scenes
- OpenAI API key

Extra local assets expected by SmartWay:

```text
third_party/smartway-code/ram_plus_swin_large_14m.pth
third_party/smartway-code/recognize-anything/
third_party/smartway-code/waypoint_predictor/checkpoints/final-camera-ready
```

Run:

```bash
export OPENAI_API_KEY=...
bash scripts/run_smartway_eval.sh
```

## Data Check

Before running either baseline:

```bash
bash scripts/check_data_layout.sh
```

Useful overrides:

```bash
SPLIT=val_seen bash scripts/run_vlnce_waypoint_eval.sh
VLNCE_WPN_CKPT=/path/to/WPN.pth bash scripts/run_vlnce_waypoint_eval.sh

SMARTWAY_SPLIT=val_unseen bash scripts/run_smartway_eval.sh
SMARTWAY_EVAL_CKPT_PATH=/path/to/policy.pth bash scripts/run_smartway_eval.sh
```

Notes:

- Matterport3D requires separate access approval.
- SmartWay downloads DINOv2 assets on first use.
- SmartWay's API wrapper reads `OPENAI_API_KEY`.
