# See2Move Review Response Notes

This note tracks changes made after the internal review.

## Addressed In The Report

- Added task-boundary discussion against VLN and Next-Best-View.
- Added a Related Work positioning section.
- Clarified that the current target is object-level visibility improvement,
  while the long-term target is operation-region visibility.
- Added implementation details for the compact CNN, Qwen3-VL feature extraction,
  depth preprocessing, pose encoding, and concat fusion.
- Reframed experimental conclusions as a project-stage technical report rather
  than a submission-ready paper.
- Added explicit discussion of dataset limitations, oracle-label limitations,
  imbalanced actions, and missing error analysis.
- Added an AAAI/ICCV/CoRL-oriented improvement plan.

## Addressed In Code

- Added macro-F1 and per-class precision/recall accounting for future training
  and evaluation runs.
- Added relative gain accounting for future training and evaluation runs:

```text
relative_gain = predicted_score / max(initial_visible_pixels, 1)
```

- Updated run comparison summaries to include macro-F1 and relative gain when
  available.
- Added a prediction analysis script for confusion matrix and per-class
  precision/recall/F1:

```bash
bash scripts/analyze_ai2thor_predictions.sh --checkpoint <checkpoint>
```

- Added optional per-record prediction export and difficulty stratification:

```bash
bash scripts/analyze_ai2thor_predictions.sh --checkpoint <checkpoint> --include-records
bash scripts/stratify_ai2thor_analysis.sh --analysis runs/analysis/<file>.json
```

- Added a compact-policy AI2-THOR rollout script for 3-step greedy evaluation:

```bash
bash scripts/rollout_ai2thor_policy.sh --policy model --steps 3
```

- Added a first gated-fusion Qwen configuration with modality dropout:

```bash
bash scripts/train_ai2thor_qwen_policy.sh configs/train_ai2thor_qwen3vl_gated.yaml
```

- Updated the main oracle generation config to 10k Stay-aware data:

```bash
bash scripts/generate_ai2thor_oracle.sh
```

Existing completed runs do not contain these newly added metrics, so their
comparison output will show `null` for those fields until they are rerun.

## Still Open

- Scale AI2-THOR data from 4.8k to 20k-50k.
- Add target-size, distance, and scene-type difficulty stratification. Initial
  visibility and oracle-gain stratification are now implemented.
- Run confusion matrix and success/failure case visualizations on final
  checkpoints.
- Run and report greedy multi-step rollout numbers.
- Add stronger baselines such as CLIP/Qwen frozen feature + sequence head or
  ResNet-18/34/50 RGB-D baselines.
- Compare gated fusion against shallow concat, then consider FiLM or
  cross-modal attention.
