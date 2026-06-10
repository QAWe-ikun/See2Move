# WSL Baselines

These notes are for running third-party baselines used by See2Move.

Workspace path in WSL:

```bash
cd /mnt/e/project/see2move
```

Set `DATA_ROOT` if datasets are outside the repo:

```bash
export DATA_ROOT=/path/to/data
```

## VLN-CE

Reference runtime:

- Python 3.6
- Habitat-Sim 0.1.7
- Habitat-Lab 0.1.7
- R2R_VLNCE_v1-3_preprocessed
- Matterport3D scenes

Build dependency:

```bash
sudo apt update
sudo apt install -y build-essential libcrypt-dev
```

Run:

```bash
bash scripts/run_vlnce_waypoint_eval.sh
```

## SmartWay

Reference runtime:

- Python 3.8.20
- PyTorch 2.1.1
- Habitat-Sim 0.1.7
- Habitat-Lab 0.1.7
- R2R_VLNCE_v1-2_preprocessed
- R2R_VLNCE_v1-2_preprocessed_BERTidx
- Matterport3D scenes
- OpenAI API key

Extra assets:

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

## Checks

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
