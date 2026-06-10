# WSL Reproduction Guide

This guide assumes WSL2 Ubuntu with NVIDIA GPU support. The Windows workspace is available as:

```bash
/mnt/e/project/see2move
```

For better I/O performance with Matterport3D, it is also reasonable to keep large datasets on the WSL ext4 filesystem and point `DATA_ROOT` there.

The scripts can always be run through `bash scripts/name.sh`. If you prefer
direct execution, run:

```bash
chmod +x scripts/*.sh
```

## Reproduced Works

### 1. VLN-CE waypoint models

Code:

```text
third_party/vln-ce
```

Main reference:

- VLN-CE: `Beyond the Nav-Graph`
- Waypoint VLN-CE: `Waypoint Models for Instruction-guided Navigation in Continuous Environments`

Official runtime:

- Python 3.6
- Habitat-Sim 0.1.7
- Habitat-Lab 0.1.7
- Matterport3D scenes
- R2R_VLNCE_v1-3_preprocessed

The vendored VLN-CE requirements pin `opencv-python-headless==4.5.5.64`
for Python 3.6. Without this pin, pip may try to build a modern OpenCV from
source and fail with a CMake/ninja error.

The vendored VLN-CE requirements install `torch` and `torchvision` with pip.
If your connection is slow, configure pip mirrors, proxies, or cache first;
the setup script uses a longer pip timeout to reduce interrupted downloads.

Suggested environment:

```bash
bash scripts/setup_vlnce_env_wsl.sh
```

Run:

```bash
cd /mnt/e/project/see2move
bash scripts/run_vlnce_waypoint_eval.sh
```

If installation already failed while downloading a large `torch` wheel, recover
inside the existing environment with:

```bash
conda activate vlnce
python -m pip install "pip==21.3.1" "setuptools<60" wheel
cd /mnt/e/project/see2move/third_party/vln-ce
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip --default-timeout=120 install --only-binary=:all: opencv-python-headless==4.5.5.64
python -m pip --default-timeout=120 install -r requirements.txt
```

### 2. SmartWay / Fast-SmartWay family

Code:

```text
third_party/smartway-code
```

This repository is the public SmartWay implementation and is the current code path exposed by the Fast-SmartWay project page. It contains:

- enhanced waypoint predictor
- VLNBERT view selection
- RAM+ object recognition
- GPT/MLLM backtracking navigator

Official runtime:

- Python 3.8.20
- PyTorch 2.1.1
- Habitat-Sim 0.1.7
- Habitat-Lab 0.1.7
- Matterport3D scenes
- R2R_VLNCE_v1-2_preprocessed
- R2R_VLNCE_v1-2_preprocessed_BERTidx
- RAM+ checkpoint
- waypoint predictor checkpoint
- OpenAI API key for the MLLM navigator

Suggested environment:

```bash
HABITAT_SIM_TARBALL=/path/to/habitat-sim-0.1.7-py3.8_headless_linux.tar.bz2 \
  bash scripts/setup_smartway_env_wsl.sh
```

Install Recognize Anything under the SmartWay repo because the code imports it with a relative path:

```bash
cd /mnt/e/project/see2move/third_party/smartway-code
git clone https://github.com/xinyu1205/recognize-anything.git
```

Run:

```bash
cd /mnt/e/project/see2move
export OPENAI_API_KEY=...
bash scripts/run_smartway_eval.sh
```

## Shared Data Layout

By default all scripts read data from:

```text
data/
```

Override it with:

```bash
export DATA_ROOT=/path/to/data
```

Expected layout:

```text
data/
  scene_datasets/
    mp3d/
      {scene_id}/
        {scene_id}.glb
        {scene_id}.navmesh
        {scene_id}.house
        {scene_id}_semantic.ply
  datasets/
    R2R_VLNCE_v1-3_preprocessed/
      val_unseen/
        val_unseen.json.gz
        val_unseen_gt.json.gz
    R2R_VLNCE_v1-2_preprocessed/
      rand100/
        rand100_gt.json.gz
      val_unseen/
        val_unseen_gt.json.gz
    R2R_VLNCE_v1-2_preprocessed_BERTidx/
      rand100/
        rand100_bertidx.json.gz
      val_unseen/
        val_unseen_bertidx.json.gz
  ddppo-models/
    gibson-2plus-resnet50.pth
  pretrained_models/
    ddppo-models/
      gibson-2plus-resnet50.pth
  checkpoints/
    pretrained/
      WPN.pth
```

SmartWay also expects:

```text
third_party/smartway-code/
  ram_plus_swin_large_14m.pth
  recognize-anything/
  waypoint_predictor/
    checkpoints/
      final-camera-ready
```

Check the layout before running:

```bash
bash scripts/check_data_layout.sh
```

## Useful Overrides

VLN-CE:

```bash
SPLIT=val_seen NUM_ENVIRONMENTS=4 bash scripts/run_vlnce_waypoint_eval.sh
VLNCE_WPN_CKPT=/path/to/WPN.pth bash scripts/run_vlnce_waypoint_eval.sh
```

SmartWay:

```bash
SMARTWAY_SPLIT=val_unseen bash scripts/run_smartway_eval.sh
SMARTWAY_EVAL_CKPT_PATH=/path/to/policy.pth bash scripts/run_smartway_eval.sh
```

If `SMARTWAY_EVAL_CKPT_PATH` points to a file, SmartWay evaluates that checkpoint. If it points to a missing path, the current code follows its zero-shot path and still loads the waypoint predictor checkpoint.

## Known Reproducibility Notes

- Both works depend on Habitat 0.1.7, which is old and CUDA-sensitive.
- SmartWay downloads DINOv2 assets through Hugging Face on first use.
- SmartWay currently hard-codes evaluation to 100 episodes inside `BaseVLNCETrainer._eval_checkpoint`.
- The SmartWay API wrapper has been patched locally to read `OPENAI_API_KEY`.
- Matterport3D requires separate access approval and cannot be redistributed in this repo.
