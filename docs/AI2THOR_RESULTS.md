# AI2-THOR Results

Current dataset:

```text
records: 4872
scenes: 120
actions: 8
majority-label baseline: 20.9%
```

The validation split is scene-level, so these numbers measure transfer to held
out rooms rather than random-sample memorization.

These completed results use the original 8-action space. New Stay-aware runs
use a 9-action head where `Stay` has score 0 and is selected when all camera
motions reduce visibility.

## Compact RGB-D-Text-Pose Baseline

```text
run: runs/ai2thor_policy_4k8_simple_e35
best top-1 accuracy: 32.31%
best balanced accuracy: 25.59%
best top-2 accuracy: 53.78%
best top-3 accuracy: 68.89%
best positive-gain rate: 55.75%
```

## Compact Model Ablations

| Run | Inputs | Top-1 | Balanced | Top-2 | Top-3 | Positive gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full | RGB + depth + text + pose | 32.31 | 25.59 | 53.78 | 68.89 | 55.75 |
| no_depth | RGB + text + pose | 31.33 | 24.39 | 53.01 | 68.67 | 55.09 |
| no_text | RGB + depth + pose | 27.38 | 18.57 | 46.55 | 64.62 | 52.35 |
| no_pose | RGB + depth + text | 29.13 | 20.91 | 49.51 | 68.24 | 54.00 |
| no_rgb | depth + text + pose | 33.19 | 26.58 | 54.98 | 70.87 | 57.61 |
| rgb_only | RGB | 24.32 | 15.16 | 42.28 | 58.93 | 55.64 |

Interpretation:

- Text is important: removing it causes the largest top-1 drop.
- Pose is also important for scene-split generalization.
- Depth is useful, but RGB + text + pose partially compensates when depth is
  removed.
- In this compact CNN, removing RGB improves the result. The likely reason is
  that raw RGB overfits or adds noise at 4.8k samples, while depth + text + pose
  provides a cleaner decision signal.
- RGB-only is weak, so visual appearance alone is not enough for this oracle.

The `no_rgb` result should be treated as the strongest compact baseline, not as
the final project target. The full target remains instruction + camera view +
depth/Z-buffer + pose/history; Qwen3-VL is the next step for making RGB useful
through stronger semantic visual encoding.

## Qwen3-VL + Depth

```text
run: runs/ai2thor_policy_4k8_qwen3vl_depth_e20
best top-1 accuracy: 32.53%
best balanced accuracy: 23.23%
best top-2 accuracy: 54.22%
best top-3 accuracy: 70.65%
best mean predicted score: 999.23
best positive-gain rate: 56.85%
```

Gain-selected checkpoint:

```text
run: runs/ai2thor_policy_4k8_qwen3vl_depth_gain_e20
selection metric: oracle_gain.positive_gain_rate
selected epoch: 9
best top-1 accuracy: 29.46%
best positive-gain rate: 57.17%
best mean predicted score: 913.65
```

Compared with the compact full baseline, Qwen3-VL gives a small top-1
improvement and a clearer improvement in top-3 accuracy and predicted
visibility score. Compared with the compact `no_rgb` result, it is still lower
on top-1, but the high predicted-score result suggests that semantic RGB
features are useful for ranking view-improving actions.

## Qwen3-VL Ablations

| Run | Inputs | Top-1 | Balanced | Top-2 | Top-3 | Mean score | Positive gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_full | Qwen3-VL + depth + pose | 32.53 | 23.23 | 54.22 | 70.65 | 999.23 | 56.85 |
| qwen_no_depth | Qwen3-VL + pose | 33.30 | 26.44 | 54.87 | 69.66 | 1008.06 | 57.72 |
| qwen_no_pose | Qwen3-VL + depth | 34.28 | 26.51 | 55.53 | 69.11 | 1081.48 | 57.72 |
| qwen_no_qwen | depth + pose | 28.70 | 21.16 | 48.19 | 65.28 | 870.07 | 54.98 |

Interpretation:

- Removing Qwen3-VL hurts strongly, so the semantic RGB+instruction feature is
  useful.
- The best Qwen variant is `qwen_no_pose`, which reaches 34.28% top-1 and the
  highest mean predicted visibility score.
- `qwen_no_depth` is also strong, so Qwen3-VL already captures much of the
  semantic view-selection signal.
- The full Qwen fusion underperforms both `qwen_no_depth` and `qwen_no_pose`.
  This suggests the current shallow fusion head is not yet using all modalities
  cleanly; it may need stronger regularization, a gated fusion head, or more
  data.

The current strongest top-1 model is `qwen_no_pose`. The strongest compact
non-Qwen baseline remains `simple_no_rgb`.

## Error Analysis: Qwen No-Pose

The `qwen_no_pose` checkpoint has macro-F1 40.51% on the full 4.8k record set.
The confusion matrix shows a strong action imbalance:

| Action | Precision | Recall | F1 | Support | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoveAhead | 50.63 | 70.66 | 58.99 | 1019 | 1422 |
| MoveBack | 0.00 | 0.00 | 0.00 | 87 | 0 |
| MoveLeft | 46.09 | 36.15 | 40.52 | 603 | 473 |
| MoveRight | 45.64 | 36.11 | 40.32 | 565 | 447 |
| RotateLeft | 51.14 | 51.20 | 51.17 | 830 | 831 |
| RotateRight | 57.70 | 49.11 | 53.06 | 839 | 714 |
| LookUp | 30.30 | 15.38 | 20.41 | 195 | 99 |
| LookDown | 54.51 | 65.80 | 59.63 | 734 | 886 |

The main failure is `MoveBack`: it appears in the dataset but is never predicted
by this checkpoint. `MoveAhead` and `LookDown` are over-predicted, while
`LookUp` is under-predicted. This supports the review concern that class
imbalance is a central bottleneck and should be addressed with resampling,
class-balanced loss, or action-aware data generation.

## Online Rollout

The compact full model was also tested in AI2-THOR closed-loop rollout:

```text
checkpoint: runs/ai2thor_policy_4k8_simple_e35/checkpoint_best.pt
episodes: 50
steps: 3
mean_total_gain: -266.82
positive_total_gain_rate: 40.00%
```

This result shows a gap between single-step offline metrics and closed-loop
camera control. The model can choose plausible one-step actions, but repeated
execution may move the camera into worse viewpoints. One immediate fix is to
add a `Stay` output head so the policy can avoid movement when no candidate
action improves visibility.

Run the Qwen ablation group with:

```bash
bash scripts/run_ai2thor_ablations.sh qwen
bash scripts/compare_ai2thor_runs.sh
```

For deployment-style camera movement, the current default compact and Qwen
training configs select checkpoints by validation positive-gain rate rather
than exact top-1 classification.

## Next Stage 1-4 Checklist

1. Main Stay-aware dataset generation:

```bash
bash scripts/generate_ai2thor_oracle.sh
```

2. Dataset difficulty stratification:

```bash
bash scripts/stratify_ai2thor_analysis.sh \
  --records /mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl \
  --output runs/analysis/dataset_strata.json
```

3. Per-record prediction analysis and stratification:

```bash
bash scripts/analyze_ai2thor_predictions.sh \
  --checkpoint runs/ablations_4k8/qwen_no_pose/checkpoint_best.pt \
  --kind qwen \
  --include-records \
  --output runs/analysis/qwen_no_pose_predictions.json

bash scripts/stratify_ai2thor_analysis.sh \
  --records /mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl \
  --analysis runs/analysis/qwen_no_pose_predictions.json \
  --output runs/analysis/qwen_no_pose_strata.json
```

4. Three-step compact-policy rollout:

```bash
bash scripts/rollout_ai2thor_policy.sh \
  --checkpoint runs/ai2thor_policy_4k8_simple_e35/checkpoint_best.pt \
  --policy model \
  --episodes 50 \
  --steps 3 \
  --output runs/rollouts/simple_model_3step.json

bash scripts/rollout_ai2thor_policy.sh --policy random --episodes 50 --steps 3 --output runs/rollouts/random_3step.json
bash scripts/rollout_ai2thor_policy.sh --policy oracle --episodes 50 --steps 3 --output runs/rollouts/oracle_3step.json
```

5. Stay-aware training:

```bash
bash scripts/train_ai2thor_policy.sh
bash scripts/train_ai2thor_qwen_policy.sh configs/train_ai2thor_qwen3vl.yaml
```
