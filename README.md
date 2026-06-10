# See2Move

See2Move is organized around two stages:

1. Reproduce existing continuous VLN baselines.
2. Extend them into generalized occlusion-aware camera/viewpoint planning.

The current workspace vendors two reference implementations:

- `third_party/vln-ce`: VLN-CE and waypoint VLN-CE baselines.
- `third_party/smartway-code`: SmartWay zero-shot VLN-CE code, including an enhanced waypoint predictor and MLLM navigator.

These are tracked as Git submodules. Local compatibility changes are stored in
`patches/third_party/`.

## Target Problem

The project target is not a table-specific heuristic. The intended task is:

```text
instruction + current RGB view + depth/Z-buffer + camera pose + history
    -> next camera motion or candidate viewpoint
```

The policy should move toward a viewpoint where the intended object, region, or manipulation affordance is visible and reachable. Examples include putting an item under a table, inspecting a shelf level, looking behind a sofa, placing an object inside a cabinet, or moving around any occluder that blocks a requested operation.

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

See `docs/WSL_REPRODUCTION.md` for the full dependency and data layout.
The scripts `scripts/setup_vlnce_env_wsl.sh` and
`scripts/setup_smartway_env_wsl.sh` are installation templates for WSL.

## Extension Plan

The generalized task design lives in:

- `docs/GENERALIZED_OCCLUSION_VIEWPOINT.md`
- `configs/occlusion_viewpoint.yaml`

The first project-specific milestone should be an oracle data generator:

1. Sample multiple candidate camera motions.
2. Render RGB-D from each candidate in Habitat/AI2-THOR/Isaac/Blender.
3. Score candidates by task-region visibility, occlusion reduction, collision risk, and motion cost.
4. Train a policy to imitate the best candidate.
