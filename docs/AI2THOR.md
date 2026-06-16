# AI2-THOR Data Generation

AI2-THOR is the main data source for See2Move.

The generator creates oracle labels by:

1. Sampling a scene from the configured scene list.
2. Sampling a camera pose in that scene.
3. Selecting a visible target object.
4. Saving RGB and depth observations.
5. Trying candidate camera motions plus `Stay`.
6. Choosing the action with the largest target visibility improvement.

`Stay` has a fixed score of `0`. If every movement would reduce target
visibility, the oracle label becomes `Stay` instead of forcing an unnecessary
camera motion.

Install:

```bash
pip install -r requirements-ai2thor.txt
```

All generator, smoke-test, and training commands assume those dependencies are
installed.

Run:

```bash
bash scripts/generate_ai2thor_oracle.sh
```

The generator shows a `tqdm` progress bar over sampling attempts and reports
the number of valid records collected so far.

Scene ranges can be configured in `configs/ai2thor_oracle.yaml`:

```yaml
target_records: 10000
max_attempts: 120000
samples_per_scene: 1
resume: true
output_dir: /mnt/f/see2move/data/ai2thor_oracle_10k_stay
scenes:
  - FloorPlan1-30
  - FloorPlan201-230
  - FloorPlan301-330
  - FloorPlan401-430
```

With `samples_per_scene: 1`, generation rotates through the scene list instead
of spending hundreds of attempts in one scene before moving on. This gives the
current broad-scene set better scene coverage.

Output:

```text
/mnt/f/see2move/data/ai2thor_oracle_10k_stay/
  records.jsonl
  000000_rgb.png
  000000_depth.npy
```

`records.jsonl` contains the instruction, target object, history, camera pose,
candidate action scores, and selected label.

This oracle is intentionally simple. Later versions should score operation
regions, occlusion reduction, motion cost, and collision risk.

## Qwen3-VL Features

Qwen3-VL is used as a frozen RGB+instruction encoder. Depth remains separate:

```text
instruction + RGB -> Qwen3-VL feature
depth/Z-buffer    -> depth CNN feature
camera pose       -> pose feature
all features      -> action head
```

Extract features with the local model:

```bash
bash scripts/extract_qwen3vl_features.sh \
  --model-path /mnt/f/models/qwen3_vl \
  --records /mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl \
  --data-dir /mnt/f/see2move/data/ai2thor_oracle_10k_stay \
  --output /mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt \
  --batch-size 1
```

If `transformers` cannot recognize `qwen3_vl`, install a newer Transformers
build in the `see2move` environment. Qwen3-VL processing also requires
`torchvision`; it is included in `requirements-ai2thor.txt`.

## Manual Exploration

Browse an AI2-THOR room from the terminal:

```bash
bash scripts/explore_ai2thor_room.sh FloorPlan1
```

Controls:

```text
w/s/a/d  move
j/l      rotate
i/k      look up/down
r        random reachable position
o        list visible objects
q        quit
```

Frames are saved after each action:

```text
runs/ai2thor_explorer/latest_rgb.png
runs/ai2thor_explorer/latest_depth.png
```

## Training

Train the policy on generated records:

```bash
pip install -r requirements-ai2thor.txt
bash scripts/train_ai2thor_policy.sh
```

The default config now trains on the 10k Stay-aware dataset and selects the
checkpoint by validation positive-gain rate. See `docs/AI2THOR_RESULTS.md` for
completed historical 4.8k ablation tables.

Run the residual-fusion experiment explicitly:

```bash
bash scripts/train_ai2thor_policy.sh configs/train_ai2thor_resfusion.yaml
```

The default policy config uses the compact baseline model as the first
RGB-D training baseline:

```yaml
model:
  architecture: simple
```

A stronger experimental multimodal model is also available:

```yaml
model:
  architecture: residual_fusion
```

`residual_fusion` uses a residual RGB-D encoder, a GRU text encoder, pose
features, language-conditioned visual gating, and an action-embedding scoring
head. Use `configs/train_ai2thor_resfusion.yaml` to run it.

The default training config uses a scene-level validation split:

```yaml
data:
  split_strategy: scene
  val_fraction: 0.2
```

It also supports class-weighted loss and early stopping for imbalanced action
labels:

```yaml
train:
  selection_metric: oracle_gain.positive_gain_rate
  scheduler: none
  class_weighting: inverse_frequency
  gradient_clip_norm: 1.0
  early_stopping_patience: 10
```

Training and evaluation print top-1, top-2, top-3, balanced accuracy,
per-action accuracy, and oracle visibility-gain metrics.

Summarize a saved training run:

```bash
bash scripts/summarize_training_run.sh --metrics runs/ai2thor_policy_10k_stay_simple_e35/metrics.json
```

Inspect dataset balance before or after training:

```bash
bash scripts/summarize_ai2thor_dataset.sh
```

Evaluate:

```bash
bash scripts/evaluate_ai2thor_policy.sh
```

Train the Qwen3-VL + depth policy:

```bash
bash scripts/train_ai2thor_qwen_policy.sh configs/train_ai2thor_qwen3vl.yaml
```

Evaluate it:

```bash
bash scripts/evaluate_ai2thor_qwen_policy.sh
```

## Ablations

After preparing the broad-scene AI2-THOR dataset, run the full compact baseline:

```bash
bash scripts/train_ai2thor_policy.sh
```

Then run compact-model ablations:

```bash
bash scripts/run_ai2thor_ablations.sh simple
```

These ablations keep the same model and training recipe, but zero out one or
more input modalities:

```text
simple_no_depth   RGB + text + pose
simple_no_text    RGB + depth + pose
simple_no_pose    RGB + depth + text
simple_no_rgb     depth + text + pose
simple_rgb_only   RGB only
```

After Qwen3-VL features have been extracted, train the full Qwen3-VL + depth
policy and its ablations:

```bash
bash scripts/train_ai2thor_qwen_policy.sh
bash scripts/run_ai2thor_ablations.sh qwen
```

The Qwen ablations are:

```text
qwen_no_depth   Qwen3-VL feature + pose
qwen_no_pose    Qwen3-VL feature + depth
qwen_no_qwen    depth + pose
```

Compare all finished runs:

```bash
bash scripts/compare_ai2thor_runs.sh
```

Smoke test without AI2-THOR:

```bash
bash scripts/make_smoke_ai2thor_dataset.sh
bash scripts/train_ai2thor_policy.sh configs/train_ai2thor_smoke.yaml
```
