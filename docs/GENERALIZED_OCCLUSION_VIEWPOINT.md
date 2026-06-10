# Generalized Occlusion-Aware Viewpoint Planning

## Goal

Convert VLN-style RGB-D navigation into active viewpoint planning for operation visibility. The system should choose the next camera motion that reveals the object, region, or affordance needed by the instruction.

This must generalize across objects and spatial relations. A table is only one case.

## Task Interface

Input:

```text
instruction
current RGB image
current depth or Z-buffer
current camera pose
optional semantic masks or open-vocabulary detections
action/viewpoint history
```

Output:

```text
next camera action
```

or:

```text
delta_pose = {dx, dy, dz, dyaw, dpitch}
```

or:

```text
candidate viewpoint index
```

## Generalized Language Targets

The instruction should be parsed into a target frame:

```text
intent: inspect | place | grasp | align | verify | navigate_to_view
target_object: any open-vocabulary category or instance
reference_object: optional object used in a relation
target_region: optional free-space or surface region
spatial_relation: under | inside | behind | between | on | near | left_of | right_of | lower_level | upper_level
visibility_requirement: see_target | see_free_space | see_contact_area | see_path | see_affordance
```

Examples:

```text
"Move so I can place the chair under the table"
"Find a view where the inside of the lower cabinet is visible"
"Move the camera to see the wall outlet behind the sofa"
"Look around the box so the handle is no longer occluded"
"Find a view of the empty area between the bed and the shelf"
```

## Reusing The Two Baselines

VLN-CE contributes:

- Habitat RGB-D sensor setup.
- Continuous navigation environment.
- Waypoint action space.
- Depth encoder and RGB-D policy patterns.
- Metrics and data pipeline conventions.

SmartWay contributes:

- Candidate waypoint generation.
- Occupancy-aware waypoint prediction.
- History-aware action selection.
- Backtracking logic.
- MLLM-based reasoning over candidates.

The project-specific extension should sit above these as an active perception task layer.

## Oracle Label Generation

For each training state:

1. Parse or sample a target frame.
2. Generate candidate camera motions.
3. Render RGB-D and optional semantic masks from each candidate.
4. Score every candidate.
5. Use the best candidate as the supervised action label.

Suggested score:

```text
score =
    w_visible_target * visible_target_area
  + w_visible_region * visible_region_area
  + w_occlusion_drop * occlusion_reduction
  + w_depth_margin * usable_depth_margin
  + w_semantic_match * target_semantic_confidence
  - w_collision * collision_risk
  - w_motion * motion_cost
  - w_repeat * revisited_view_penalty
```

The score is generic because `visible_target_area` can be an object, a free-space region, a support surface, a container interior, or a relation-defined area.

## Candidate Action Space

Start with a discrete candidate set:

```text
stop
forward 0.25m
back 0.25m
strafe_left 0.25m
strafe_right 0.25m
yaw_left 15deg
yaw_right 15deg
pitch_up 15deg
pitch_down 15deg
raise_camera 0.10m
lower_camera 0.10m
```

Then add waypoint candidates from SmartWay/VLN-CE:

```text
heading angle + distance
```

The final model can select a candidate or regress a continuous delta pose.

## Model Milestones

### M0: Reproduce baselines

Run VLN-CE waypoint and SmartWay evaluation with shared RGB-D assets.

### M1: Oracle active viewpoint baseline

No training. Use depth, semantic masks, and candidate rendering to pick the best next view.

### M2: Imitation policy

Train a model to imitate the oracle:

```text
RGB encoder + depth encoder + text encoder + history encoder -> candidate score
```

### M3: MLLM planner

Use SmartWay-style prompts, but replace route-following language with operation-visibility language. Candidate descriptions should include visible objects, target visibility, occlusion status, free-space estimate, and motion cost.

### M4: Generalization evaluation

Evaluate by held-out:

- object category
- spatial relation
- room type
- scene
- occluder type
- simulator-to-real camera depth noise

## Metrics

Use both navigation and visibility metrics:

```text
view_success: target visibility exceeds threshold
occlusion_reduction: before/after hidden target fraction
region_visibility: target free-space or affordance mask area
motion_cost: path length or number of camera moves
collision_rate
backtrack_rate
instruction_conditioned_success
```

## Simulator Candidates

Use VLN-CE/MP3D for initial reproduction. For operation-style tasks, prefer a simulator with manipulable or queryable object semantics:

- Habitat Rearrangement / ReplicaCAD
- AI2-THOR / ProcTHOR
- Isaac Sim
- BlenderProc for synthetic data generation

The first practical path is Habitat/ReplicaCAD or AI2-THOR because they make object categories, receptacles, and spatial relations easier to label than raw MP3D.
