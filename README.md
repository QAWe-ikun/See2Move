# See2Move

See2Move predicts the next camera motion from language, RGB observation,
depth/Z-buffer, camera pose, and interaction history.

The goal is to move the camera toward a viewpoint where the object, region, or
operation area described by the instruction becomes easier to see and act on.

```text
instruction + RGB + depth/Z-buffer + camera pose + history
    -> next camera motion / candidate viewpoint
```

## Project Scope

See2Move is the main project. The third-party repositories are included as
references and baselines, not as the final system.

- `third_party/vln-ce`: continuous VLN and waypoint navigation baseline.
- `third_party/smartway-code`: SmartWay / Fast-SmartWay related baseline and
  waypoint-selection foundation.

These are tracked as Git submodules and should point to the project forks.

## Current Workspace

The repository currently contains:

- See2Move data generation code in `see2move/`
- baseline run scripts in `scripts/`
- AI2-THOR data notes in `docs/AI2THOR.md`
- AI2-THOR experiment report in `docs/SEE2MOVE_REPORT.md`
- AI2-THOR result tables in `docs/AI2THOR_RESULTS.md`
- WSL notes in `docs/WSL_REPRODUCTION.md`
- project target notes in `docs/TARGET.md`
- third-party baselines as Git submodules

## Quickstart

From WSL:

```bash
cd /mnt/e/project/see2move
bash scripts/check_data_layout.sh
```

Run baselines after environments, datasets, and checkpoints are prepared:

```bash
# VLN-CE waypoint baseline
conda activate vlnce
bash scripts/run_vlnce_waypoint_eval.sh

# SmartWay baseline
conda activate smartway
export OPENAI_API_KEY=...
bash scripts/run_smartway_eval.sh
```

Generate the first See2Move oracle dataset:

```bash
pip install -r requirements-ai2thor.txt
bash scripts/generate_ai2thor_oracle.sh
```

Explore an AI2-THOR room manually:

```bash
bash scripts/explore_ai2thor_room.sh FloorPlan1
```

Train the main policy:

```bash
bash scripts/train_ai2thor_policy.sh
```

Render a multi-case greedy rollout MP4 demo:

```bash
conda activate see2move
bash scripts/record_qwen_greedy_rollout_demo.sh --steps 5 --max-cases 4
```

The video is written to:

```text
runs/qwen_greedy_rollout_demo/greedy_rollout_demo.mp4
```
