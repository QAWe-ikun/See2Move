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

Evaluate:

```bash
bash scripts/evaluate_ai2thor_policy.sh
```

Smoke test without AI2-THOR:

```bash
bash scripts/make_smoke_ai2thor_dataset.sh
bash scripts/train_ai2thor_policy.sh configs/train_ai2thor_smoke.yaml
```
