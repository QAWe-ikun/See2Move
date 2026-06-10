# Target

See2Move is an instruction-conditioned camera motion prediction project.

Input:

```text
instruction
RGB camera view
depth or Z-buffer
camera pose
history
```

Output:

```text
next camera motion or candidate viewpoint
```

The predicted motion should improve visibility of the relevant object, area, or
operation region. The target is category- and relation-general, for example:

- view a blocked object
- see inside or under a region
- move around an occluder
- inspect a placement or manipulation area

## Role Of Third-Party Code

VLN-CE and SmartWay are used for:

- RGB-D navigation environment setup
- waypoint/action-space baselines
- comparison metrics
- candidate viewpoint generation ideas

They are not the final See2Move model.

## First Milestone

Build a baseline loop:

1. Read instruction, RGB, depth/Z-buffer, pose, and history.
2. Generate candidate camera motions.
3. Score candidates by visibility improvement, collision risk, and movement
   cost.
4. Select the best next motion.

The first implementation uses AI2-THOR to generate oracle labels. See
`docs/AI2THOR.md`.

The current trainable policy is an RGB-D, language, and pose action classifier
trained on AI2-THOR oracle records.
