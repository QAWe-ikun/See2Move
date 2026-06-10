# AI2-THOR Data Generation

AI2-THOR is the main data source for See2Move.

The first generator creates oracle labels by:

1. Sampling a scene and camera pose.
2. Selecting a visible target object.
3. Saving RGB and depth observations.
4. Trying candidate camera motions.
5. Choosing the action with the largest target visibility improvement.

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

Output:

```text
data/ai2thor_oracle/
  records.jsonl
  000000_rgb.png
  000000_depth.npy
```

`records.jsonl` contains the instruction, target object, history, camera pose,
candidate action scores, and selected label.

This oracle is intentionally simple. Later versions should score operation
regions, occlusion reduction, motion cost, and collision risk.

## Training

Train the policy on generated records:

```bash
pip install -r requirements-ai2thor.txt
bash scripts/train_ai2thor_policy.sh
```

Evaluate:

```bash
bash scripts/evaluate_ai2thor_policy.sh
```

Smoke test without AI2-THOR:

```bash
bash scripts/make_smoke_ai2thor_dataset.sh
bash scripts/train_ai2thor_policy.sh configs/train_ai2thor_smoke.yaml
```
