# See2Move

See2Move is a reproduction workspace for RGB-D vision-language navigation
baselines, with a later extension toward instruction-conditioned camera motion
prediction.

## Baselines

- `third_party/vln-ce`: VLN-CE and waypoint VLN-CE.
- `third_party/smartway-code`: SmartWay / Fast-SmartWay related code path.

These are tracked as Git submodules. Local compatibility changes are stored in
`patches/third_party/`.

## Extension Target

The extension task is:

```text
instruction + current RGB view + depth/Z-buffer + camera pose + history
    -> next camera motion / candidate viewpoint
```

The predicted motion should improve visibility of the object, region, or
operation area referenced by the instruction.

## WSL Quickstart

From WSL:

```bash
cd /mnt/e/project/see2move
bash scripts/check_data_layout.sh
```

After the conda environments, datasets, and checkpoints are prepared:

```bash
# VLN-CE waypoint baseline
conda activate vlnce
bash scripts/run_vlnce_waypoint_eval.sh

# SmartWay zero-shot baseline
conda activate smartway
export OPENAI_API_KEY=...
bash scripts/run_smartway_eval.sh
```

See `docs/WSL_REPRODUCTION.md` for dependency and data layout notes.
