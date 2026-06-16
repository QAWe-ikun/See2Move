# See2Move: Visibility-Driven Camera Motion Prediction for Placement Models

## Abstract

Existing Vision-Language-Action (VLA) models and World Action Models (WAMs) mainly focus on real-world robot control, indoor navigation, or general embodied control, where the goal is often to directly generate manipulation or navigation actions from visual observations and language instructions. However, in Technical Artist (TA) and 3D scene-editing workflows, many operations do not start from direct object placement. Instead, the system often needs to first obtain a clearer and less occluded viewpoint. For placement or scene rearrangement models such as SceneReVis, an occluded camera view may limit the model's ability to understand the target object, supporting surface, or operation region.

We propose See2Move, a camera motion prediction framework that provides viewpoint assistance for placement models. Given a language intent, current RGB view, depth map, and camera pose, See2Move predicts the next camera action that makes the target object or operation region more visible. Unlike conventional Vision-Language Navigation (VLN), See2Move does not primarily optimize goal-reaching or path-following. Instead, it directly predicts the visibility gain of each candidate camera action and selects the action with the highest expected gain. We generate supervision automatically in AI2-THOR by using instance segmentation masks to compute the visible-pixel change of the target object before and after each candidate action. The model fuses frozen Qwen3-VL hidden states, depth features, and camera pose features to train a lightweight gain-score policy. A Stay action and a stay-threshold inference strategy are further introduced to reduce unnecessary or negative-gain camera movements.

## Keywords

Technical Artist; scene placement; SceneReVis; viewpoint adjustment; camera motion prediction; visibility improvement; depth map; AI2-THOR

## 1 Introduction

### 1.1 Background

Recent advances in large-scale vision-language models and embodied AI have improved the ability of agents to reason over visual observations, language instructions, and actions. VLA models attempt to unify visual inputs, textual commands, and low-level actions in an end-to-end framework, with applications in robotic grasping, mobile manipulation, and interactive task execution. Meanwhile, VLN methods study how an agent follows language instructions to move through indoor environments, including continuous control in realistic spaces.

However, TA and 3D scene-editing tasks differ from conventional robotic manipulation or navigation. In DCC software, game engines, or simulation environments, the operating target is usually an existing 3D scene. For placement or scene rearrangement models such as SceneReVis, the core goal is to infer reasonable object positions, orientations, or rearrangement plans from scene context. Before making such placement predictions, the system often needs a sufficiently clear viewpoint to understand the target object, supporting surface, occlusion relation, and local geometry.

For example, when a user wants to place a chair under a table, the current view may be blocked by the tabletop, table legs, or surrounding furniture. If a placement model reasons only from the current occluded view, it may fail to determine whether the space under the table is available, whether the target region is occupied, or whether the spatial relation between objects is plausible. Therefore, a pre-placement viewpoint adjustment module is useful: it does not directly perform placement, but predicts the next camera movement to provide a clearer and less occluded observation for downstream placement models.

### 1.2 Motivation

In real-world robotic settings, models often need to estimate 3D structure from 2D images, such as depth, camera pose, point clouds, or object geometry. Recent visual geometry models can recover camera parameters, depth maps, and 3D point clouds from images, providing useful geometric priors for robotics and embodied AI. In contrast, 3D authoring environments used by TAs already contain rich geometric information. DCC tools, game engines, and simulators can directly provide meshes, depth buffers, object transforms, camera poses, instance IDs, and object visibility.

This means that TA-oriented and placement-oriented agents do not need to rely entirely on external 3D reconstruction models. They can instead leverage native geometric signals already available in the authoring environment. See2Move is based on this setting: it uses the current RGB view, depth map, and camera pose to learn how to move the camera so that the target region becomes more visible. The module can serve as an upstream assistant for placement models such as SceneReVis, actively searching for a better viewpoint before placement prediction.

### 1.3 Contributions

We propose See2Move, a visibility-driven camera motion prediction framework for placement models. The main contributions are:

1. We formulate pre-placement viewpoint adjustment as a language-conditioned camera motion prediction problem based on target visibility improvement, rather than conventional path following or waypoint prediction.

2. We build an AI2-THOR-based oracle data generation pipeline that automatically obtains target visible-pixel changes after candidate actions using instance masks, depth maps, and camera poses, avoiding manual next-camera-action annotation.

3. We design a multimodal gain-score policy that fuses Qwen3-VL hidden states, depth features, and camera pose features to directly predict the visibility gain of each candidate camera action.

4. We introduce a Stay action and a stay-threshold inference strategy, allowing the model to preserve the current viewpoint when the expected movement benefit is insufficient, thereby reducing zero-gain or negative-gain camera movements.

5. We position See2Move as a pre-perception assistant for placement and scene rearrangement models such as SceneReVis, providing clearer and less occluded observations for downstream placement prediction.

## 2 Related Work

### 2.1 Vision-Language Navigation

Vision-Language Navigation studies how an agent moves through an environment according to natural language instructions. Early VLN tasks often rely on discrete navigation graphs, where the agent moves between predefined nodes. VLN-CE extends this setting to continuous environments, requiring agents to execute low-level actions in more realistic physical spaces. Later studies explore waypoint prediction, topological planning, and cross-modal Transformers to improve navigation in continuous environments.

These methods provide important foundations for language-conditioned visual motion decision-making. However, their primary objective is usually path following or goal reaching, and their metrics include success rate, SPL, and path length. In contrast, See2Move focuses on local viewpoint adjustment. The model does not need to complete long-horizon navigation; instead, it determines whether the next camera action can improve the visibility of the target object or operation region, thereby providing a better input viewpoint for downstream placement models.

### 2.2 Vision-Language-Action Models and World Action Models

VLA models attempt to unify vision, language, and action in an end-to-end framework. Representative work such as RT-2 transfers large-scale vision-language models to robotic control, enabling models to output robot actions from images and language commands. WAM-related studies further explore learning action-conditioned state transitions from videos, trajectories, or world models. These methods usually target robotic manipulation, mobile robot control, or general embodied control.

Although VLA/WAM methods could in principle be extended to camera control, few studies explicitly target pre-placement viewpoint adjustment in TA scenarios. For placement models such as SceneReVis, action prediction is not always equivalent to directly moving objects. In many cases, the system should first adjust the viewpoint to obtain more reliable local geometry and occlusion information. See2Move can be regarded as a concrete instantiation of VLA/WAM-style reasoning in a TA setting: the action space is camera motion and viewpoint adjustment rather than robot end-effector control, and the optimization target is the visibility gain of the next camera action rather than task completion alone.

### 2.3 3D Geometry, Simulation, and Placement Models

In real-world settings, models often need to estimate depth, camera pose, or 3D structure from RGB images. Visual geometry models have demonstrated the ability to predict camera parameters, depth maps, and point clouds from one or multiple images, providing important 3D cues when explicit geometry is unavailable. However, in DCC software, game engines, and simulation environments, much of this 3D information is directly maintained by the environment.

AI2-THOR is an interactive 3D indoor environment that provides RGB images, depth maps, instance segmentation, and agent poses. We use AI2-THOR instance segmentation masks to compute the visible-pixel count of the target object before and after each candidate action, and use the difference as the action gain score. Compared with reconstructing 3D geometry from 2D images, our method directly leverages native geometric and visibility signals in the simulator, reducing data generation cost and aligning the supervision more closely with the task objective.

In relation to placement models, See2Move does not replace the placement prediction ability of models such as SceneReVis. Instead, it complements their pre-perception stage. Placement models need to understand target regions, support relations, occlusion relations, and scene context. See2Move predicts camera actions that lead to clearer viewpoints, allowing downstream placement models to reason from more reliable observations.

## 3 Method

### 3.1 Task Definition

Given the current observation:

```text
o_t = {I_t, D_t, p_t, x}
```

where `I_t` denotes the current RGB image, `D_t` denotes the depth map, `p_t` denotes the camera pose, and `x` denotes the language instruction or placement intent. The model selects the next camera action from the candidate action set:

```text
A = {MoveAhead, MoveBack, MoveLeft, MoveRight,
     RotateLeft, RotateRight, LookUp, LookDown, Stay}
```

Instead of formulating the task as simple action classification, we predict the visibility gain of each candidate action:

```text
g(a) = V_after(a) - V_current
```

where `V_current` is the visible-pixel count of the target object or target region in the current view, and `V_after(a)` is the visible-pixel count after executing action `a`. The model outputs a predicted gain `g_hat(a)` for each action and selects the action with the highest predicted gain:

```text
a* = argmax_a g_hat(a)
```

When integrated with placement models such as SceneReVis, the output action from See2Move updates the camera viewpoint. The updated RGB-D observation and camera pose are then passed to the placement model, improving the visibility condition for downstream placement prediction.

### 3.2 Oracle Data Generation

We use AI2-THOR to automatically generate training data. For each sampled state, we randomly select a scene, camera position, camera orientation, and target object. We then record the current RGB image, depth map, camera pose, and language instruction. For each candidate action, the environment temporarily executes the action, computes the instance mask area of the target object in the new view, and restores the camera to the original state.

The visible-pixel count is obtained from AI2-THOR instance segmentation masks. Specifically, `event.instance_masks` contains a binary mask for each object instance. For the target object `target_id`, summing the mask pixels gives the number of visible pixels in the current camera view:

```text
V = sum(mask(target_id))
```

The supervision score for a candidate action is defined as:

```text
score(a) = visible_pixels_after(a) - visible_pixels_before
```

If no moving action produces a positive gain, the oracle label is set to `Stay`. Therefore, the ground-truth score of `Stay` is 0, indicating that keeping the current viewpoint does not improve visibility but can avoid negative-gain movements.

### 3.3 Model Architecture

See2Move adopts a multimodal fusion architecture. The language instruction and RGB view are first encoded by a frozen Qwen3-VL model to obtain hidden states representing high-level semantic and visual context. The depth map is encoded by a lightweight convolutional network to provide distance, occlusion, and local geometric cues. The camera pose is encoded by an MLP to represent the spatial state of the current camera.

The three feature types are concatenated and passed through a gated fusion module to obtain a unified state representation. Finally, the gain-score head outputs the predicted gain for each candidate action:

```text
[g_hat(MoveAhead), ..., g_hat(Stay)]
```

This architecture allows the model to jointly use language intent, visual semantics, depth geometry, and camera pose to estimate how different camera actions affect target visibility.

### 3.4 Training Objective

The training objective consists of two parts. The first is a score regression loss, which directly fits the ground-truth gain score of each candidate action. The second is a ranking loss, which encourages the predicted score of the oracle-best action to be higher than those of other candidates. The overall loss is:

```text
L = lambda_score L_score + lambda_rank L_rank
```

In the current implementation, the auxiliary cross-entropy classification term is disabled, so the model mainly learns continuous gain scores rather than reducing the task to action-label classification. This design better matches the task objective because multiple actions may yield positive visibility gains, and the key problem is to compare their expected benefits.

### 3.5 Stay-Threshold Inference

Because the model outputs continuous gain scores, it may assign a slightly positive score to low-benefit actions, causing unnecessary movements. We therefore introduce a stay-threshold strategy:

```text
if max_a in moving_actions g_hat(a) <= tau:
    choose Stay
else:
    choose argmax_a g_hat(a)
```

where `tau` is an absolute gain threshold. This strategy does not require the ground-truth target mask at inference time, making it applicable when oracle visibility is unavailable. In experiments, different thresholds can be swept to analyze the trade-off among positive gain rate, negative gain rate, and mean predicted score.

### 3.6 Interface With Placement Models

See2Move can be used as an upstream module for placement models. Given a placement intent and current scene observation, See2Move first predicts the next camera action. If the current view is already sufficient, it outputs `Stay`; otherwise, it moves the camera toward a viewpoint with higher target visibility. Placement models such as SceneReVis can then use the updated RGB-D view, camera pose, and scene state for object placement or scene rearrangement reasoning.

This design decouples "seeing the target region clearly" from "generating the placement plan." See2Move selects a viewpoint that is more suitable for observing local spatial relations, while the placement model generates the final placement result from more reliable input.

## 4 Experiments and Results

### 4.1 Experimental Setup

We construct a visibility-driven camera motion prediction dataset in AI2-THOR. The dataset covers 120 indoor scenes, including kitchens, living rooms, bedrooms, and bathrooms, and contains 5,000 oracle records. Each record includes a language instruction, current RGB image, depth map, camera pose, target object information, candidate actions, and the visibility gain of each candidate action. The action space contains nine actions:

```text
MoveAhead, MoveBack, MoveLeft, MoveRight,
RotateLeft, RotateRight, LookUp, LookDown, Stay
```

The target object categories include CounterTop, Cabinet, Bed, Shelf, Chair, SideTable, DiningTable, Sofa, Sink, Toilet, Bathtub, and Desk. During data generation, AI2-THOR provides instance segmentation masks. We compute the visible area of the target object before and after each candidate action by summing the target mask pixels. The supervision score of an action is defined as the visible-pixel count after the action minus the visible-pixel count before the action.

The model uses frozen Qwen3-VL hidden states as language and visual-semantic features. The depth map is encoded by a lightweight CNN, and the camera pose is encoded by an MLP. The policy head outputs a predicted gain score for each candidate action. The training objective combines score regression loss and ranking loss. During training, a scene split is used for checkpoint selection. In this chapter, we use the same 5,000 records for unified offline evaluation across all compared models.

### 4.2 Evaluation Metrics

We evaluate the models using the following metrics:

1. **Accuracy**: the fraction of predicted actions that match the oracle label.
2. **Macro F1**: the macro-averaged F1 score over action classes.
3. **Top-2 / Top-3 Accuracy**: whether the oracle action appears in the top-2 or top-3 predicted actions.
4. **Mean Predicted Score**: the average ground-truth visibility gain of the action selected by the model.
5. **Positive Gain Rate**: the fraction of selected actions whose ground-truth gain is positive.
6. **Negative Gain Rate**: the fraction of selected actions whose ground-truth gain is negative.
7. **Safe Balanced Gain Score**: a combined metric over macro F1, positive gain rate, and nonnegative gain rate.
8. **Stay Predicted Count / Stay Recall**: how often the model predicts Stay and how well it recalls oracle Stay cases.

For placement-assistance tasks, accuracy is not the only objective. Since multiple actions may produce positive gains, a prediction can still be useful even if it does not exactly match the oracle label. Therefore, mean predicted score, positive gain rate, and negative gain rate more directly measure whether See2Move can provide a better observation for downstream placement models such as SceneReVis.

### 4.3 Gain-Score Training vs. Classification

An early version of the model formulated the task as 9-way action classification. On the full records, this classification model achieves 0.217 accuracy, 0.099 macro F1, 642.64 mean predicted score, and 0.555 positive gain rate. This indicates that directly predicting oracle labels does not sufficiently exploit the differences among candidate action gains and can lead to action-distribution collapse.

After switching to gain-score regression and ranking, the model directly predicts the visibility gain of each candidate action. Without using a Stay threshold, the gain-score model improves accuracy to 0.458, mean predicted score to 1773.59, and positive gain rate to 0.685. Compared with the classification model, the gain-score model improves the average visibility gain by approximately 2.76 times, suggesting that directly optimizing action utility is better aligned with the viewpoint adjustment objective.

| Model | Accuracy | Macro F1 | Mean Predicted Score | Positive Gain Rate | Top-2 Acc | Top-3 Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen classification | 0.217 | 0.099 | 642.64 | 0.555 | 0.414 | 0.565 |
| Qwen gain-score | 0.458 | 0.325 | 1773.59 | 0.685 | 0.655 | 0.763 |

This comparison shows that the main advantage of See2Move is not learning a fixed action label, but learning the relative utility of different camera actions. For downstream placement models, selecting an action that significantly improves target-region visibility is more important than strictly matching the oracle action label.

### 4.4 Stay-Threshold Analysis

Because the gain-score model outputs continuous scores, it may assign slightly positive scores to low-benefit actions, resulting in unnecessary camera movements. We therefore introduce a stay-threshold mechanism: if all non-Stay actions have predicted gains below the threshold, the model selects Stay.

We evaluate thresholds of 0, 100, 200, and 500. The results are shown below:

| Stay Threshold | Accuracy | Macro F1 | Mean Predicted Score | Positive Gain Rate | Negative Gain Rate | Safe Score | Stay Predicted Count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.458 | 0.326 | 1775.66 | 0.685 | 0.307 | 0.543 | 11 |
| 100 | 0.457 | 0.328 | 1773.96 | 0.683 | 0.304 | 0.543 | 38 |
| 200 | 0.456 | 0.332 | 1777.62 | 0.679 | 0.296 | 0.545 | 98 |
| 500 | 0.426 | 0.324 | 1731.92 | 0.611 | 0.229 | 0.528 | 781 |

Although threshold 500 substantially reduces the negative gain rate, it makes the policy overly conservative: the number of Stay predictions increases to 781, and both positive gain rate and accuracy drop. Threshold 200 maintains a high mean predicted score while reducing the negative gain rate from 0.307 to 0.296 and achieves the best safe balanced gain score among the tested settings. We therefore use stay-threshold = 200 in subsequent experiments.

### 4.5 Ablation Study

To analyze the contribution of each modality, we conduct ablation experiments under the same gain-score objective and the same stay-threshold = 200 setting. The compared models are:

1. **Full**: uses Qwen3-VL hidden states, depth map, and camera pose.
2. **No Depth**: removes the depth input.
3. **No Pose**: removes the camera pose input.
4. **No Qwen**: removes Qwen3-VL visual-language features and keeps only depth and pose.

The ablation results are:

| Model | Accuracy | Macro F1 | Mean Gain | Positive Gain | Negative Gain | Safe Score | Stay Pred | Stay Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.4560 | 0.3320 | 1777.62 | 0.6790 | 0.2960 | 0.5452 | 98 | 0.0678 |
| No Depth | 0.4356 | 0.3112 | 1659.26 | 0.6640 | 0.3120 | 0.5277 | 96 | 0.0565 |
| No Pose | 0.4398 | 0.3215 | 1718.64 | 0.6668 | 0.3124 | 0.5329 | 84 | 0.0395 |
| No Qwen | 0.1912 | 0.0625 | 776.45 | 0.5652 | 0.4284 | 0.3654 | 0 | 0.0000 |

The Full model achieves the best performance across all major metrics. Removing depth decreases the mean gain from 1777.62 to 1659.26 and increases the negative gain rate from 0.296 to 0.312, indicating that depth provides useful occlusion, distance, and local geometry cues. Removing camera pose decreases the mean gain to 1718.64 and increases the negative gain rate to 0.3124, showing that pose information also helps the model reason about the spatial effect of camera actions.

The most significant degradation occurs in the No Qwen setting. Without Qwen3-VL hidden states, accuracy drops to 0.1912, macro F1 drops to 0.0625, mean gain drops to 776.45, and negative gain rate increases to 0.4284. This shows that language and RGB semantic features are critical for identifying the target object, understanding the instruction, and selecting an effective viewpoint. Depth and pose alone provide geometric information, but they lack target semantics and are insufficient for deciding which object or region should guide the camera movement.

### 4.6 Discussion

The results support three main observations. First, directly predicting candidate action gain scores is more suitable for See2Move than action classification. In viewpoint adjustment, multiple actions may yield positive gains, and the oracle label is only the best among them. Continuous utility prediction better captures the relative quality of candidate actions. Second, Qwen3-VL semantic features, depth maps, and camera poses are complementary. Semantic features help identify the target and interpret the instruction; depth provides occlusion and geometric cues; pose helps model the spatial effect of camera motion. Third, the Stay threshold can reduce negative-gain movements, but an overly large threshold makes the model too conservative and suppresses useful camera movements.

From the perspective of assisting placement models such as SceneReVis, the value of See2Move lies in improving input observation quality. With stay-threshold = 200, the Full model achieves a mean ground-truth visibility gain of 1777.62 and a positive gain rate of 0.679, showing that the model can often select actions that make the target more visible. This suggests that downstream placement models can reason about target regions, supporting surfaces, and occlusion relations from clearer observations.

## 5 Limitations and Future Work

### 5.1 Limitations

The current experiments have several limitations. First, the evaluation is mainly based on offline single-step action prediction over AI2-THOR oracle records. We have not yet fully validated whether integrating See2Move with SceneReVis improves final placement quality. Second, Stay remains difficult to predict. Even with stay-threshold = 200, Stay recall is only 0.0678, indicating that the model still struggles to decide when not to move. Third, the gain score is generated using target object instance masks during training and offline evaluation. Although inference does not require ground-truth masks, future work could explore predicting current visibility or combining the model with a segmentation module. Finally, the current evaluation focuses on single-step camera actions; multi-step viewpoint adjustment and closed-loop integration with placement models remain important future directions.

### 5.2 Future Work: From Explicit Objects to Implicit Operation Regions

The current version of See2Move mainly handles explicit-target tasks, where the language instruction refers to a relatively clear target object, such as "look at the laptop," "move to see the chair," or "inspect the cabinet." In this setting, AI2-THOR can provide clear supervision through the target object's instance mask. The visibility gain can therefore be directly defined as the difference between the target object's visible pixels before and after a candidate action.

However, real TA and placement-model scenarios often involve implicit-target tasks. These tasks do not specify a single object instance and may not have a directly available ground-truth mask. Examples include "find the space under the table," "check whether there is room beside the bed," "inspect the inside of the cabinet," or "determine whether a side table can be placed next to the sofa." Such tasks focus on an operation region, a spatial relation, or an available support area rather than a concrete object. For placement models such as SceneReVis, these implicit targets are particularly important because placement decisions often depend on whether the target region is visible, unoccupied, and geometrically feasible.

Future work will therefore extend See2Move from explicit object visibility to implicit operation-region visibility. One possible direction is to use native simulation signals such as meshes, depth buffers, object transforms, and camera poses to construct region-level supervision, including spaces under tables, gaps between objects, areas near supporting surfaces, and container interiors. Another direction is to let the model infer the implicit target region from the language intent and learn the visibility or operability of that region under different camera views. With this extension, See2Move would not only help placement models observe concrete objects, but also help them actively search for spatial regions that are suitable for placement and scene rearrangement.

## References

[1] Kolve, E., Mottaghi, R., Han, W., et al. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv:1712.05474, 2017.

[2] Krantz, J., Wijmans, E., Majumdar, A., et al. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments. ECCV, 2020.

[3] Brohan, A., Brown, N., Carbajal, J., et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818, 2023.

[4] An, D., Wang, Y., Wang, R., et al. ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments. IEEE TPAMI, 2024.

[5] Wang, J., Chen, M., Karaev, N., et al. VGGT: Visual Geometry Grounded Transformer. arXiv:2503.11651, 2025.
