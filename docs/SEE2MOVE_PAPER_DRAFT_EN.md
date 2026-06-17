# See2Move: Visibility-Driven Camera Motion Prediction for Placement Models

## Abstract

Existing Vision-Language-Action (VLA) models and World Action Models (WAMs) mainly focus on real-world robot control, indoor navigation, or general embodied control, where the goal is often to directly generate manipulation or navigation actions from visual observations and language instructions. However, in Technical Artist (TA) and 3D scene-editing workflows, many operations do not start from direct object placement. Instead, the system often needs to first obtain a clearer and less occluded viewpoint. For placement or scene rearrangement models such as SceneReVis, an occluded camera view may limit the model's ability to understand the target object, supporting surface, or operation region.

We propose See2Move, a camera motion prediction framework that provides viewpoint assistance for placement models. Given a language intent, current RGB view, depth map, and camera pose, See2Move predicts the next camera action that makes the target object or operation region more visible. Unlike conventional Vision-Language Navigation (VLN), See2Move does not primarily optimize goal-reaching or path-following. Instead, it directly predicts the visibility gain of each candidate camera action and selects the action with the highest expected gain. We generate supervision automatically in AI2-THOR by using instance segmentation masks to compute the visible-pixel change of the target object before and after each candidate action. The model fuses frozen Qwen3-VL hidden states, depth features, and camera pose features to train a lightweight gain-score policy. A Stay action and a stay-threshold inference strategy are further introduced to reduce unnecessary or negative-gain camera movements. On 5,000 AI2-THOR oracle records, the Qwen3-VL gain-score policy with stay-threshold = 200 achieves a mean true visibility gain of 1777.62 and a positive gain rate of 0.679, outperforming a strong SigLIP baseline with 1479.99 mean gain and 0.6352 positive gain rate, as well as the model without vision-language features with 776.45 mean gain and 0.5652 positive gain rate.

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

### 1.4 Scope and Positioning

The goal of this work is not to replace existing placement models, scene rearrangement models, or general-purpose VLA models. Instead, See2Move targets a missing pre-perception step before placement reasoning: actively acquiring a more useful viewpoint. In TA workflows, the agent typically operates in an editable 3D scene rather than a completely unknown real-world environment. Therefore, the system can naturally access depth buffers, camera poses, instance IDs, object transforms, and other signals already maintained by the rendering or simulation environment. Under this setting, the model does not need to spend most of its capacity reconstructing 3D geometry from 2D images; it can directly learn whether a candidate camera action improves the visibility of the target region.

See2Move also differs from conventional VLN in scope. VLN emphasizes long-horizon path following and goal reaching, and agents are usually evaluated by navigation success or path efficiency. See2Move focuses on short-horizon, local viewpoint adjustment for downstream placement reasoning. The output can be a translation, rotation, camera pitch adjustment, or `Stay` when the current view is already sufficient. This design matches the interaction pattern of TA and scene-editing workflows: the system first helps the user or downstream placement model see the relevant region more clearly, and the placement model then predicts the actual object placement or scene rearrangement.

From an application perspective, See2Move can serve as an upstream observation module for local placement models such as SceneReVis. For tasks such as placing a chair under a table, checking whether a side table can fit beside a bed, or inspecting the inside of a cabinet, the placement model needs to reason about free space, support surfaces, and local occlusion. If the input viewpoint is occluded, placement quality may degrade. See2Move aims to reduce this pre-perception bottleneck by producing a more reliable RGB-D observation and camera state for the downstream model.

### 1.5 Paper Organization

Section 2 reviews related work in VLN, VLA/WAM, simulation environments, scene placement, vision foundation models, and active vision. Section 3 presents the task formulation, oracle data generation, model architecture, training objective, and Stay inference strategy. Section 4 reports the AI2-THOR experimental setup, metrics, main results, threshold analysis, and ablation study. Section 5 discusses limitations and the future transition from explicit object targets to implicit operation regions. Section 6 concludes the paper.

## 2 Related Work

### 2.1 Vision-Language Navigation

Vision-Language Navigation studies how an agent moves through an environment according to natural language instructions [1]. Early VLN tasks often rely on discrete navigation graphs over photo-realistic indoor environments such as Matterport3D [3], while Habitat provides a commonly used embodied simulation platform for navigation research [4]. VLN-CE extends this setting to continuous environments, requiring agents to execute low-level actions in more realistic physical spaces [2]. Later studies explore waypoint prediction, topological planning, and cross-modal Transformers to improve navigation in continuous environments [5,6,7,8].

These methods provide important foundations for language-conditioned visual motion decision-making. However, their primary objective is usually path following or goal reaching, and their metrics include success rate, SPL, and path length. In contrast, See2Move focuses on local viewpoint adjustment. The model does not need to complete long-horizon navigation; instead, it determines whether the next camera action can improve the visibility of the target object or operation region, thereby providing a better input viewpoint for downstream placement models.

### 2.2 Vision-Language-Action Models and World Action Models

VLA models attempt to unify vision, language, and action in an end-to-end framework. Representative works such as RT-1, RT-2, PaLM-E, OpenVLA, Octo, and pi-zero transfer large-scale vision-language or generalist policy models to robotic control, enabling models to output robot actions from images and language commands [13,15,16,18,19,20]. WAM-related studies further explore learning action-conditioned state transitions from videos, trajectories, or world models. These methods usually target robotic manipulation, mobile robot control, or general embodied control [14,17,21].

Although VLA/WAM methods could in principle be extended to camera control, few studies explicitly target pre-placement viewpoint adjustment in TA scenarios. For placement models such as SceneReVis, action prediction is not always equivalent to directly moving objects. In many cases, the system should first adjust the viewpoint to obtain more reliable local geometry and occlusion information. See2Move can be regarded as a concrete instantiation of VLA/WAM-style reasoning in a TA setting: the action space is camera motion and viewpoint adjustment rather than robot end-effector control, and the optimization target is the visibility gain of the next camera action rather than task completion alone.

### 2.3 3D Geometry, Simulation, and Placement Models

In real-world settings, models often need to estimate depth, camera pose, or 3D structure from RGB images. Visual geometry models have demonstrated the ability to predict camera parameters, depth maps, and point clouds from one or multiple images, providing important 3D cues when explicit geometry is unavailable. However, in DCC software, game engines, and simulation environments, much of this 3D information is directly maintained by the environment.

AI2-THOR is an interactive 3D indoor environment that provides RGB images, depth maps, instance segmentation, and agent poses [9]. Related embodied-AI benchmarks such as ALFRED, TEACh, and RoboTHOR also demonstrate the value of interactive simulators for grounded language and visual decision-making [10,11,12]. We use AI2-THOR instance segmentation masks to compute the visible-pixel count of the target object before and after each candidate action, and use the difference as the action gain score. Compared with reconstructing 3D geometry from 2D images, our method directly leverages native geometric and visibility signals in the simulator, reducing data generation cost and aligning the supervision more closely with the task objective.

In relation to placement models, See2Move does not replace the placement prediction ability of models such as SceneReVis. Instead, it complements their pre-perception stage. Placement models need to understand target regions, support relations, occlusion relations, and scene context. See2Move predicts camera actions that lead to clearer viewpoints, allowing downstream placement models to reason from more reliable observations.

### 2.4 Scene Synthesis, Object Placement, and Scene Rearrangement

Indoor scene synthesis and object placement focus on generating plausible 3D layouts from room structures, existing objects, and semantic relations. 3D-FRONT provides a large-scale indoor scene dataset with layouts and semantic annotations, serving as an important data foundation for indoor scene synthesis and placement tasks [22]. SceneFormer models indoor objects as sequences with positions and orientations using Transformers [23]. ATISS formulates indoor scene synthesis as autoregressive generation over unordered object sets and supports scene completion and partial rearrangement [24]. InstructScene further incorporates language instructions and semantic graph priors to improve the controllability of text-driven 3D indoor scene synthesis [25].

These works focus on how to generate or modify scene layouts, while See2Move focuses on how to obtain a better viewpoint before placement prediction. Therefore, See2Move is complementary to placement and scene rearrangement models: the placement model generates object positions and spatial relations, while See2Move selects camera views that better reveal target regions, supporting surfaces, and occlusion relations.

### 2.5 Vision Foundation Models, Language Grounding, and 3D Geometry Estimation

Recent vision foundation models and open-vocabulary perception models have substantially improved semantic and geometric understanding from images. CLIP learns transferable vision-language representations through large-scale image-text contrastive learning [26], while Qwen3-VL and related large vision-language models improve unified modeling of multi-resolution images, text, and video [27]. For open-vocabulary localization and segmentation, Segment Anything provides promptable general-purpose segmentation [28], and Grounding DINO supports open-set object detection based on category names or referring expressions [29]. For geometry, Depth Anything improves monocular depth generalization through large-scale unlabeled data [30], while DUSt3R, MASt3R, and VGGT explore estimating depth, camera pose, image matching, and 3D point clouds from images [31,32,33].

These methods are important when explicit 3D information is unavailable. In contrast, TA and simulation/DCC environments often already provide native 3D signals such as depth buffers, camera poses, meshes, object transforms, and instance IDs. The relative advantage of See2Move is that it does not need to first reconstruct 3D geometry from 2D images; instead, it directly leverages existing scene information to learn viewpoint adjustment.

### 2.6 Active Vision and Next-Best-View

Active vision argues that a perception system can actively control camera or sensor motion to acquire more useful information [34]. Next-Best-View (NBV) planning further studies how to select the next viewpoint to maximize 3D reconstruction coverage, target coverage, or information gain. Representative methods include shadowcasting-based viewpoint planning for spatial exploration [35] and shape-completion-driven NBV planning for occluded-object reconstruction [36].

See2Move shares a similar motivation with active vision and NBV: moving the viewpoint should produce a more useful observation. However, our task differs in two ways. First, See2Move is language-conditioned, and viewpoint selection is driven by user intent and placement needs. Second, it focuses on local visibility improvement for TA and placement models, rather than general reconstruction coverage. Thus, See2Move can be viewed as language-conditioned next-view prediction for pre-perception in placement workflows.

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

To reduce annotation cost, the entire data generation process is driven by the simulator. The generator first samples a room from the scene set, samples an agent position from the reachable positions returned by AI2-THOR, and randomizes the camera orientation and horizon. It then chooses an observable target object and constructs a natural language instruction from its category, such as "Move the camera to get a clearer view of the countertop." For each candidate action, the generator performs a temporary transition, measures target visibility in the resulting view, and restores the original camera state. Each record therefore contains the full gain distribution over candidate actions rather than only a single discrete label.

```text
Algorithm 1: AI2-THOR visibility-oracle generation
Input: scene set S, action set A, target record number N
Output: records with RGB, depth, pose, instruction, candidate scores

while number_of_records < N:
    sample scene s from S
    reset AI2-THOR controller to s
    sample reachable camera pose p
    render current observation o_t
    sample target object id from visible or candidate objects
    V_current = visible_pixels(target_id, o_t)
    for each action a in A \ {Stay}:
        execute a temporarily
        V_after(a) = visible_pixels(target_id, o_{t+1})
        score(a) = V_after(a) - V_current
        restore pose p
    score(Stay) = 0
    label = argmax_a score(a), or Stay if all moving actions are non-positive
    save record
```

This oracle has two advantages. First, it preserves continuous utilities for all candidate actions, enabling the policy to learn relative action quality instead of only a hard label. Second, it directly uses instance masks and depth buffers from the simulator, avoiding subjective manual annotation of the next camera action. Its current limitation is that supervision is still defined mainly around explicit target objects; extending it to implicit spatial regions remains future work.

### 3.3 Model Architecture

See2Move adopts a multimodal fusion architecture. The language instruction and RGB view are first encoded by a frozen Qwen3-VL model to obtain hidden states representing high-level semantic and visual context. The depth map is encoded by a lightweight convolutional network to provide distance, occlusion, and local geometric cues. The camera pose is encoded by an MLP to represent the spatial state of the current camera.

The three feature types are concatenated and passed through a gated fusion module to obtain a unified state representation. Finally, the gain-score head outputs the predicted gain for each candidate action:

```text
[g_hat(MoveAhead), ..., g_hat(Stay)]
```

This architecture allows the model to jointly use language intent, visual semantics, depth geometry, and camera pose to estimate how different camera actions affect target visibility.

#### 3.3.1 Qwen3-VL Feature Extraction

We use a frozen local Qwen3-VL model as the high-level vision-language encoder. For each record, the language instruction and the current RGB image are passed to Qwen3-VL with `output_hidden_states=True`. We take the last hidden state and apply attention-mask-weighted mean pooling over the sequence, which contains image tokens, text tokens, and special tokens. The resulting global multimodal feature is saved as a float16 tensor. During policy training, gradients are not propagated into Qwen3-VL, reducing memory usage and training time.

The pooled Qwen3-VL feature is projected by `LayerNorm -> Linear -> ReLU -> Dropout` into a 512-dimensional semantic branch. This design treats the VLM as a frozen perceptual prior rather than an end-to-end trainable component. The advantage is stable training under the current 5,000-record dataset size; the limitation is that the frozen feature is not specifically adapted to predicting how camera actions change future visibility.

#### 3.3.2 Depth and Pose Encoding

The depth map is obtained from the AI2-THOR depth frame. During preprocessing, NaN and infinite values are replaced, depth values are clipped to `[0, 5m]`, and the result is normalized to `[0,1]`. In the Qwen3-VL gain-score model, the depth input is not a single raw depth channel. Instead, we construct a three-channel depth stack:

```text
DepthStack = [depth, 1 - depth, depth_edge]
```

where `depth_edge` is approximated by horizontal and vertical depth differences, providing explicit cues about local geometric boundaries and occlusion changes. The three-channel depth map is resized to 128x128 and encoded by a lightweight residual CNN. The CNN consists of strided convolution, residual blocks, adaptive average pooling, and a linear projection, producing a 128-dimensional depth feature.

The camera pose is encoded as a 7-dimensional vector:

```text
[x, y, z, sin(yaw), cos(yaw), horizon / 90, is_standing]
```

Using `sin(yaw)` and `cos(yaw)` avoids the discontinuity around 0 and 360 degrees. This vector is projected to 64 dimensions by `Linear -> LayerNorm -> ReLU`. Pose features help the model reason about the current spatial state of the camera and the geometric effect of actions such as MoveAhead, RotateLeft, and LookDown.

#### 3.3.3 Fusion and Gain Head

The Qwen3-VL feature, depth feature, and pose feature are concatenated and passed through a gated fusion module. Let the concatenated vector be `z`. The gate produces `sigmoid(MLP(z))`, and the gated input is:

```text
z_fused = z * sigmoid(MLP(z))
```

The fused vector is then processed by a two-layer MLP with LayerNorm, ReLU, and Dropout to obtain a 512-dimensional representation. A final linear head outputs the predicted gain score for each of the nine actions. During training, modality dropout randomly zeros modality branches, reducing over-reliance on a single input type and improving robustness in ablation or deployment settings.

### 3.4 Training Objective

The training objective consists of two parts. The first is a score regression loss, which directly fits the ground-truth gain score of each candidate action. The second is a ranking loss, which encourages the predicted score of the oracle-best action to be higher than those of other candidates. The overall loss is:

```text
L = lambda_score L_score + lambda_rank L_rank
```

In the current implementation, the auxiliary cross-entropy classification term is disabled, so the model mainly learns continuous gain scores rather than reducing the task to action-label classification. This design better matches the task objective because multiple actions may yield positive visibility gains, and the key problem is to compare their expected benefits.

More concretely, the score regression loss uses the ground-truth gain score of each candidate action as supervision. Because raw pixel gains can reach thousands or more, scores are divided by `score_scale=1000` during training to stabilize gradient magnitudes. The ranking loss encourages the predicted score of the oracle-best action to exceed those of other candidates under a margin. The main model uses:

```text
L = 1.0 * L_score + 0.5 * L_rank + 0.0 * L_ce
```

where the cross-entropy term is explicitly disabled. We train with AdamW using an initial learning rate of `1e-3`, weight decay `5e-4`, batch size 64, and up to 25 epochs. A cosine learning-rate schedule is used with minimum learning rate `5e-5`. Gradients are clipped to a maximum norm of 1.0. Checkpoint selection uses validation `oracle_gain.mean_predicted_score`, and early stopping patience is set to 8.

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

Dataset statistics show that the 5,000 records cover 120 scenes and 17 target object categories. The action label distribution is naturally imbalanced: MoveAhead, RotateRight, RotateLeft, and LookDown appear more frequently, while MoveBack and Stay are relatively rare. This imbalance reflects the task itself. In random indoor viewpoints, moving forward, rotating, or looking down is more likely to reveal more of the target object, whereas moving backward or staying is optimal only under specific distance and occlusion conditions. Therefore, we report accuracy, macro F1, and gain-based metrics together, rather than relying on accuracy alone.

The experiments are run in WSL. Data and features are stored on the local HDD path `/mnt/f/see2move`, while the code is located at `/mnt/e/project/see2move`. Qwen3-VL hidden states are extracted offline before policy training, and training only loads the pooled feature tensor. This reduces memory requirements and allows the gain-score policy to be iterated quickly on a single-machine setup. Since Qwen3-VL feature extraction is relatively expensive, we separate feature extraction from policy training: the former is a one-time preprocessing step, while the latter is used for the main experiments, threshold sweep, and ablation study.

### 4.2 Evaluation Metrics

We evaluate the models using the following metrics:

1. **Accuracy**: the fraction of predicted actions that match the oracle label.
2. **Macro F1**: the macro-averaged F1 score over action classes.
3. **Top-2 / Top-3 Accuracy**: whether the oracle action appears in the top-2 or top-3 predicted actions.
4. **Mean Predicted Score**: the average ground-truth visibility gain of the action selected by the model.
5. **Positive Gain Rate**: the fraction of selected actions whose ground-truth gain is positive.
6. **Negative Gain Rate**: the fraction of selected actions whose ground-truth gain is negative.
7. **Relative Gain**: the relative visibility improvement of the selected action, defined as

```text
relative_gain(a) = score(a) / max(V_current, 1)
```

This metric reduces the influence of object size. Large objects naturally produce larger absolute pixel gains, while relative gain measures improvement with respect to the current visible area.
8. **Safe Balanced Gain Score**: an auxiliary metric combining macro F1, positive gain rate, and nonnegative gain rate:

```text
safe_score = 0.4 * macro_F1
           + 0.4 * positive_gain_rate
           + 0.2 * nonnegative_gain_rate
```

This metric is used only for threshold selection, balancing action-class quality, positive utility, and avoidance of negative-gain movements. The main conclusions are still based on macro F1, mean gain, relative gain, positive gain rate, and negative gain rate.
9. **Stay Predicted Count / Stay Recall**: how often the model predicts Stay and how well it recalls oracle Stay cases.

For placement-assistance tasks, accuracy is not the only objective. Since multiple actions may produce positive gains, a prediction can still be useful even if it does not exactly match the oracle label. Therefore, mean predicted score, positive gain rate, and negative gain rate more directly measure whether See2Move can provide a better observation for downstream placement models such as SceneReVis.

We also report `top-2` and `top-3` accuracy. These metrics measure whether the oracle-best action appears among the highest-scoring candidate actions. In deployment, top-k metrics are useful for two reasons. First, if the downstream system allows short-horizon search or physical reachability filtering, top-k predictions can provide fallback actions. Second, high top-k accuracy indicates that the model has learned a reasonable action ranking even when it does not select the exact oracle label. Compared with plain accuracy, top-k metrics better reflect the model's understanding of the candidate gain structure.

Top-2/Top-3 accuracy is not treated as the core metric in this 9-action setting, since random Top-3 already reaches about 33.3%. We use it only as an auxiliary signal about action ranking quality. For placement assistance, the more direct questions are whether the selected action improves target visibility, whether negative-gain actions are reduced, and whether the improvement is meaningful relative to the current visible area.

### 4.3 Gain-Score Training vs. Classification

An early version of the model formulated the task as 9-way action classification. On the full records, this classification model achieves 0.217 accuracy, 0.099 macro F1, 642.64 mean predicted score, and 0.555 positive gain rate. This indicates that directly predicting oracle labels does not sufficiently exploit the differences among candidate action gains and can lead to action-distribution collapse.

After switching to gain-score regression and ranking, the model directly predicts the visibility gain of each candidate action. Without using a Stay threshold, the gain-score model improves accuracy to 0.458, mean predicted score to 1773.59, and positive gain rate to 0.685. Compared with the classification model, the gain-score model improves the average visibility gain by approximately 2.76 times, suggesting that directly optimizing action utility is better aligned with the viewpoint adjustment objective.

| Model | Accuracy | Macro F1 | Mean Predicted Score | Mean Relative Gain | Positive Gain Rate | Top-2 Acc | Top-3 Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen classification | 0.217 | 0.099 | 642.64 | 5.457 | 0.555 | 0.414 | 0.565 |
| Qwen gain-score | 0.458 | 0.325 | 1773.59 | 2.451 | 0.685 | 0.655 | 0.763 |

This comparison shows that the main advantage of See2Move is not learning a fixed action label, but learning the relative utility of different camera actions. For downstream placement models, selecting an action that significantly improves target-region visibility is more important than strictly matching the oracle action label. Relative gain is sensitive to samples with very small current visible-pixel counts, so it should not be interpreted as a replacement for absolute mean gain, positive gain rate, and negative gain rate. Although the classification model has a higher relative gain, its absolute gain, macro F1, and top-k accuracy are substantially lower, indicating that it can produce local proportional improvements on low-visibility cases without learning a stable global action ranking.

The classification model performs poorly for three reasons. First, the oracle label is a hard label defined by the maximum visibility gain, but in many samples the second-best action may have a similar gain. Treating all non-best actions as equally wrong discards useful ranking information. Second, the action distribution is naturally imbalanced, and rare actions such as MoveBack and Stay are easy to ignore under a classification objective. Third, classification only asks which action is best; it does not directly penalize selecting a negative-gain action. The gain-score objective explicitly regresses the utility of each candidate action and uses ranking loss to constrain action ordering, making it better aligned with viewpoint assistance for placement models.

### 4.3.1 Non-Learned Baselines and Oracle Upper Bounds

In addition to the classification baseline, we include several training-free policies. `Random` samples from the executable candidate actions of each record. `Majority` always selects the most frequent oracle label. `Always-Stay` always keeps the current viewpoint. `Positive Oracle` selects Stay when all moving actions have non-positive gain and otherwise selects the moving action with the highest true gain. `Oracle` directly selects the candidate action with the highest true score and serves as a single-step upper bound under the current action set and visibility definition.

These baselines clarify the metric range and task difficulty. Random and Majority estimate distributional lower bounds, Always-Stay evaluates a fully conservative policy, and Positive Oracle / Oracle provide upper bounds given oracle visibility. All baselines are evaluated on the same records and with the same action vocabulary as the learned policies.

### 4.3.2 SigLIP Strong Baseline

We further evaluate a SigLIP-based learned baseline. This baseline uses a frozen `siglip-so400m-patch14-384` model to encode the current RGB image and the language instruction, producing an image embedding and a text embedding. To preserve image-text matching information, we concatenate four feature types:

```text
f_siglip = [f_image, f_text, f_image * f_text, |f_image - f_text|]
```

The SigLIP feature is then combined with the same depth encoder and camera-pose encoder used by the main model, and trained with the same gain-score policy. Thus, the SigLIP baseline and the Qwen3-VL full model use the same data, action space, depth CNN, pose MLP, fusion head, gain-score loss, scene split, and evaluation protocol. The only difference is the frozen vision-language feature extractor. This provides a fair learned baseline for testing whether Qwen3-VL features are stronger than a strong standard vision-language encoder for this task.

Under the unified stay-threshold = 200 setting, the SigLIP baseline achieves 0.419 accuracy, 0.2886 macro F1, 1479.99 mean predicted score, and 0.6352 positive gain rate. This is substantially better than the No Qwen setting, showing that a strong vision-language encoder is useful for this task. However, it remains below the Qwen3-VL full model, which reaches 1777.62 mean predicted score and 0.679 positive gain rate. This suggests that Qwen3-VL hidden states provide stronger target semantics and multimodal context for local viewpoint adjustment.

| Model | VLM Feature | Accuracy | Macro F1 | Mean Gain | Relative Gain | Positive Gain | Negative Gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Qwen | None | 0.1912 | 0.0625 | 776.45 | 5.629 | 0.5652 | 0.4284 |
| SigLIP Full | SigLIP-SO400M | 0.4190 | 0.2886 | 1479.99 | 2.788 | 0.6352 | 0.3256 |
| Qwen3-VL Full | Qwen3-VL | 0.4560 | 0.3320 | 1777.62 | 2.453 | 0.6790 | 0.2960 |

The SigLIP baseline is evaluated under the same stay-threshold setting as the Qwen3-VL policy. This comparison isolates the effect of the frozen vision-language encoder while keeping the downstream policy and supervision unchanged. Full VLN-CE or ETPNav systems are not directly comparable because they are designed for long-horizon language navigation rather than local visibility-gain prediction; a navigation-style comparison would require adapting them into local action scorers that treat candidate camera motions as waypoint choices.

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

It is important that the Stay threshold does not require oracle masks or true current visible pixels at inference time. It only uses predicted moving-action gain scores: if all moving actions are below the threshold, the system keeps the current view. Thus, the strategy can be used as a deployment-time safety filter. A low threshold still permits low-benefit movements, whereas a high threshold makes the policy overly conservative and suppresses useful movements. In our current data scale and score calibration, 200 is an empirical compromise rather than a universal constant. Changes in scene distribution, image resolution, or visibility definition would require recalibrating this threshold.

Stay prediction remains a major weakness. With stay-threshold = 200, the Full model reaches only 0.0678 Stay recall, meaning that most oracle Stay cases are still predicted as moving actions. Several factors contribute to this. First, Stay is relatively rare in the dataset, so the model is biased toward frequent moving actions. Second, Stay has a fixed ground-truth score of 0, while many moving actions have small positive or small negative scores, making it difficult to separate "do not move" from "low-benefit move." Third, the gain-score objective emphasizes selecting the maximum-utility action and does not explicitly optimize the decision of whether movement is necessary. Future versions should explore Stay resampling, Stay-aware losses, a two-stage policy that first predicts move-versus-stay, or an explicit movement cost.

The threshold sweep quantifies the trade-off among Stay precision/recall, negative gain rate, and mean gain.

### 4.5 Ablation Study

To analyze the contribution of each modality, we conduct ablation experiments under the same gain-score objective and the same stay-threshold = 200 setting. The compared models are:

1. **Full**: uses Qwen3-VL hidden states, depth map, and camera pose.
2. **No Depth**: removes the depth input.
3. **No Pose**: removes the camera pose input.
4. **No Qwen**: removes Qwen3-VL visual-language features and keeps only depth and pose.

The ablation results are:

| Model | Accuracy | Macro F1 | Mean Gain | Relative Gain | Positive Gain | Negative Gain | Stay Pred | Stay Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.4560 | 0.3320 | 1777.62 | 2.453 | 0.6790 | 0.2960 | 98 | 0.0678 |
| No Depth | 0.4356 | 0.3112 | 1659.26 | 6.226 | 0.6640 | 0.3120 | 96 | 0.0565 |
| No Pose | 0.4398 | 0.3215 | 1718.64 | 5.659 | 0.6668 | 0.3124 | 84 | 0.0395 |
| No Qwen | 0.1912 | 0.0625 | 776.45 | 5.629 | 0.5652 | 0.4284 | 0 | 0.0000 |

The Full model achieves the best performance across all major metrics. Removing depth decreases the mean gain from 1777.62 to 1659.26 and increases the negative gain rate from 0.296 to 0.312, indicating that depth provides useful occlusion, distance, and local geometry cues. Removing camera pose decreases the mean gain to 1718.64 and increases the negative gain rate to 0.3124, showing that pose information also helps the model reason about the spatial effect of camera actions.

The most significant degradation occurs in the No Qwen setting. Without Qwen3-VL hidden states, accuracy drops to 0.1912, macro F1 drops to 0.0625, mean gain drops to 776.45, and negative gain rate increases to 0.4284. This shows that language and RGB semantic features are critical for identifying the target object, understanding the instruction, and selecting an effective viewpoint. Depth and pose alone provide geometric information, but they lack target semantics and are insufficient for deciding which object or region should guide the camera movement.

The ablation results also clarify the roles of geometry and semantics. Depth and pose mainly help reduce bad movements and increase average gain, while Qwen3-VL features mainly tell the model which target is relevant. Removing depth or pose causes a moderate drop in accuracy, but the negative gain rate increases from 0.296 to about 0.312, suggesting that geometry helps avoid moving into worse views. Removing Qwen3-VL features is much more damaging: the model almost never predicts Stay, the mean gain is less than half of the Full model, and the model lacks the semantic grounding needed to connect the instruction with the relevant part of the scene.

In relative terms, Full improves mean gain from 776.45 to 1777.62 compared with No Qwen, corresponding to approximately 2.29x improvement. It reduces the negative gain rate from 0.4284 to 0.2960, a reduction of about 30.9%. Compared with No Depth, Full improves mean gain by about 7.1%; compared with No Pose, it improves mean gain by about 3.4%. These results suggest that Qwen3-VL semantic features are the dominant modality, while depth and pose provide additional geometric benefits. This matches the task intuition: without knowing which object is relevant, geometry alone cannot determine where the camera should move; once the target semantics are available, depth and pose help decide how to move to see it more clearly.

In addition to aggregate metrics, we report per-action precision, recall, and F1, with particular attention to minority actions such as Stay, MoveBack, and LookUp. The Full model with stay-threshold = 200 obtains the following per-action results:

| Action | Precision | Recall | F1 | Support | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoveAhead | 0.4407 | 0.6981 | 0.5403 | 1017 | 1611 |
| MoveBack | 0.0000 | 0.0000 | 0.0000 | 53 | 0 |
| MoveLeft | 0.3959 | 0.2144 | 0.2782 | 541 | 293 |
| MoveRight | 0.4361 | 0.1854 | 0.2602 | 534 | 227 |
| RotateLeft | 0.5064 | 0.4977 | 0.5020 | 872 | 857 |
| RotateRight | 0.5299 | 0.4615 | 0.4933 | 884 | 770 |
| LookUp | 0.3464 | 0.2732 | 0.3055 | 194 | 153 |
| LookDown | 0.4521 | 0.6154 | 0.5212 | 728 | 991 |
| Stay | 0.1224 | 0.0678 | 0.0873 | 177 | 98 |

The model performs best on frequent and relatively stable actions such as MoveAhead, RotateLeft, RotateRight, and LookDown. It performs poorly on MoveBack and Stay. MoveBack is never predicted in the current evaluation, suggesting that the model rarely identifies cases where moving backward improves visibility. Stay has 0.1224 precision and 0.0678 recall, indicating that the stay-threshold strategy only partially recovers no-move decisions. This per-action analysis shows that high mean gain does not imply balanced action modeling; minority-action failures remain a deployment risk in boundary cases.

### 4.6 Discussion

The results support three main observations. First, directly predicting candidate action gain scores is more suitable for See2Move than action classification. In viewpoint adjustment, multiple actions may yield positive gains, and the oracle label is only the best among them. Continuous utility prediction better captures the relative quality of candidate actions. Second, Qwen3-VL semantic features, depth maps, and camera poses are complementary. Semantic features help identify the target and interpret the instruction; depth provides occlusion and geometric cues; pose helps model the spatial effect of camera motion. Third, the Stay threshold can reduce negative-gain movements, but an overly large threshold makes the model too conservative and suppresses useful camera movements.

From the perspective of assisting placement models such as SceneReVis, the value of See2Move lies in improving input observation quality. With stay-threshold = 200, the Full model achieves a mean ground-truth visibility gain of 1777.62 and a positive gain rate of 0.679, showing that the model can often select actions that make the target more visible. This suggests that downstream placement models can reason about target regions, supporting surfaces, and occlusion relations from clearer observations.

### 4.7 Reproducibility and Error Sources

The reproducibility of our experiments depends on three components: the AI2-THOR data generation script, the Qwen3-VL feature extraction script, and the training/evaluation configuration files. Dataset records are stored as JSONL files. Each record explicitly contains the scene, instruction, target object, agent pose, RGB/depth file paths, candidate actions, and ground-truth candidate scores. Qwen3-VL features are stored in a `.pt` file and aligned with the JSONL records by order. Training settings are stored in YAML files, including the action set, data paths, model dimensions, objective type, learning rate, batch size, score scale, and early stopping configuration. Given the same dataset and feature file, the main experiments and ablations can be reproduced directly.

The current errors mainly come from four sources. First, random viewpoint sampling in AI2-THOR may produce very small or heavily occluded targets, making candidate gains sensitive to small camera changes. Second, visible-pixel gain only measures target mask area; it does not explicitly measure viewpoint quality, support-surface visibility, or spatial operability. Third, the frozen Qwen3-VL feature is not fine-tuned for predicting future visibility after camera actions, so it may lack action-conditioned dynamics. Fourth, single-step offline evaluation is not equivalent to multi-step closed-loop usage, where consecutive actions may introduce accumulated errors. These limitations motivate our future work.

### 4.8 Relative Advantages Over Existing Directions

Compared with conventional VLN, See2Move does not require long-horizon navigation or path-following metrics; it directly optimizes local target visibility for the next camera action. Compared with general VLA/WAM methods, See2Move does not attempt to learn a complete robot manipulation policy, but focuses on a narrower and practical pre-perception problem in TA and scene-editing workflows. Compared with methods that rely on external 3D reconstruction, See2Move directly uses depth buffers, camera poses, and instance IDs already available in simulation or DCC environments, reducing the cost of obtaining supervision. Compared with scene placement models, See2Move does not generate object positions; it improves the observation used by those models.

Therefore, the relative advantage of See2Move is fourfold: the task target is closer to TA placement workflows, the supervision can be generated automatically from simulation, the input modalities naturally integrate existing 3D signals, and the output action can be directly inserted before a placement model. The current results do not claim a complete TA agent; rather, they show that under explicit object targets, the model learns a more useful visibility-gain prediction strategy than a classification baseline.

### 4.9 Qualitative Case Studies

Viewpoint adjustment requires qualitative evidence to show where the model succeeds and fails. We analyze representative positive-gain successes, negative-gain failures, Stay misses, and Stay hits from the unified evaluation set. Each case contains the initial RGB image, a depth visualization, instruction, predicted action, oracle action, Top-3 predicted gains, and true candidate gains. Successful cases illustrate how the policy moves away from occlusion and improves target visibility, while failure and Stay-miss cases reveal the main failure modes of the current policy.

## 5 Limitations and Future Work

### 5.1 Limitations

The current experiments have several limitations. First, the evaluation is mainly based on offline single-step action prediction over AI2-THOR oracle records, so the effect of integrating See2Move with SceneReVis on final placement quality remains to be validated. Second, Stay remains difficult to predict. Even with stay-threshold = 200, Stay recall is only 0.0678, indicating that the model still struggles to decide when not to move. Third, the gain score is generated using target object instance masks during training and offline evaluation. Although inference does not require ground-truth masks, future work could explore predicting current visibility or combining the model with a segmentation module. Finally, the current evaluation focuses on single-step camera actions; multi-step viewpoint adjustment and closed-loop integration with placement models remain important future directions.

### 5.2 Future Work: From Explicit Objects to Implicit Operation Regions

The current version of See2Move mainly handles explicit-target tasks, where the language instruction refers to a relatively clear target object, such as "look at the laptop," "move to see the chair," or "inspect the cabinet." In this setting, AI2-THOR can provide clear supervision through the target object's instance mask. The visibility gain can therefore be directly defined as the difference between the target object's visible pixels before and after a candidate action.

However, real TA and placement-model scenarios often involve implicit-target tasks. These tasks do not specify a single object instance and may not have a directly available ground-truth mask. Examples include "find the space under the table," "check whether there is room beside the bed," "inspect the inside of the cabinet," or "determine whether a side table can be placed next to the sofa." Such tasks focus on an operation region, a spatial relation, or an available support area rather than a concrete object. For placement models such as SceneReVis, these implicit targets are particularly important because placement decisions often depend on whether the target region is visible, unoccupied, and geometrically feasible.

Future work will therefore extend See2Move from explicit object visibility to implicit operation-region visibility. One possible direction is to use native simulation signals such as meshes, depth buffers, object transforms, and camera poses to construct region-level supervision, including spaces under tables, gaps between objects, areas near supporting surfaces, and container interiors. Another direction is to let the model infer the implicit target region from the language intent and learn the visibility or operability of that region under different camera views. With this extension, See2Move would not only help placement models observe concrete objects, but also help them actively search for spatial regions that are suitable for placement and scene rearrangement.

## 6 Conclusion

We presented See2Move, a visibility-driven camera motion prediction framework for placement models. Unlike conventional VLN or general VLA/WAM systems, See2Move focuses on a pre-perception problem in TA and 3D scene-editing workflows: before placement or scene rearrangement, the system should actively choose a camera action that better reveals the target object or operation region. We use AI2-THOR to automatically generate oracle supervision by computing target visible-pixel changes from instance masks after candidate camera actions, and formulate the task as candidate gain-score prediction.

The model fuses frozen Qwen3-VL hidden states, depth-map features, and camera pose features through a gated fusion policy, producing a predicted gain score for each action. Experiments show that gain-score regression and ranking substantially outperform direct action classification: mean visibility gain improves from 642.64 to 1773.59, and positive gain rate improves from 0.555 to 0.685. With stay-threshold = 200, the Full model obtains 1777.62 mean gain, 0.679 positive gain rate, and 0.296 negative gain rate in unified offline evaluation. Ablations further show that Qwen3-VL semantic features, depth maps, and camera poses are complementary: Qwen3-VL is most important for target and instruction grounding, while depth and pose help reduce negative-gain movements and improve average utility.

Overall, See2Move demonstrates a practical route for TA-oriented placement assistance. Instead of relying entirely on external 3D reconstruction or end-to-end robot policies, it leverages native 3D signals already available in simulation and DCC environments to learn a lightweight, interpretable, and deployable viewpoint adjustment module. Future work will extend this framework to implicit operation regions, multi-step closed-loop viewpoint adjustment, and end-to-end evaluation with placement models such as SceneReVis.

## References

[1] Anderson, P., Wu, Q., Teney, D., et al. Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments. CVPR, 2018.

[2] Krantz, J., Wijmans, E., Majumdar, A., et al. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments. ECCV, 2020.

[3] Chang, A. X., Dai, A., Funkhouser, T., et al. Matterport3D: Learning from RGB-D Data in Indoor Environments. 3DV, 2017.

[4] Savva, M., Kadian, A., Maksymets, O., et al. Habitat: A Platform for Embodied AI Research. ICCV, 2019.

[5] Chen, S., Guhur, P.-L., Schmid, C., Laptev, I. History Aware Multimodal Transformer for Vision-and-Language Navigation. NeurIPS, 2021.

[6] Chen, S., Guhur, P.-L., Tapaswi, M., Schmid, C., Laptev, I. Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation. CVPR, 2022.

[7] An, D., Wang, Y., Wang, R., et al. ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments. IEEE TPAMI, 2024.

[8] Krantz, J., Gokaslan, A., Batra, D., Lee, S., Maksymets, O. Waypoint Models for Instruction-Guided Navigation in Continuous Environments. arXiv:2110.02207, 2021.

[9] Kolve, E., Mottaghi, R., Han, W., et al. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv:1712.05474, 2017.

[10] Shridhar, M., Thomason, J., Gordon, D., et al. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. CVPR, 2020.

[11] Padmakumar, A., Thomason, J., Shrivastava, A., et al. TEACh: Task-driven Embodied Agents that Chat. AAAI, 2022.

[12] Deitke, M., Han, W., Herrasti, A., et al. RoboTHOR: An Open Simulation-to-Real Embodied AI Platform. CVPR, 2020.

[13] Brohan, A., Chebotar, Y., Finn, C., et al. RT-1: Robotics Transformer for Real-World Control at Scale. RSS, 2023.

[14] Ahn, M., Brohan, A., Brown, N., et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. CoRL, 2022.

[15] Driess, D., Xia, F., Sajjadi, M. S. M., et al. PaLM-E: An Embodied Multimodal Language Model. ICML, 2023.

[16] Brohan, A., Brown, N., Carbajal, J., et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. CoRL, 2023.

[17] O'Neill, A., Rehman, A., Maddukuri, A., et al. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. ICRA, 2024.

[18] Kim, M. J., Pertsch, K., Karamcheti, S., et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246, 2024.

[19] Ghosh, D., Walke, H., Pertsch, K., et al. Octo: An Open-Source Generalist Robot Policy. RSS, 2024.

[20] Black, K., Brown, N., Driess, D., et al. pi0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164, 2024.

[21] Chi, C., Feng, S., Du, Y., et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS, 2023.

[22] Fu, H., Cai, B., Gao, L., et al. 3D-FRONT: 3D Furnished Rooms with Layouts and Semantics. ICCV, 2021.

[23] Wang, X., Yeshwanth, C., Nießner, M. SceneFormer: Indoor Scene Generation with Transformers. 3DV, 2021.

[24] Paschalidou, D., Kar, A., Shugrina, M., Kreis, K., Geiger, A., Fidler, S. ATISS: Autoregressive Transformers for Indoor Scene Synthesis. NeurIPS, 2021.

[25] Lin, C.-H., et al. InstructScene: Instruction-Driven 3D Indoor Scene Synthesis. arXiv, 2024.

[26] Radford, A., Kim, J. W., Hallacy, C., et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.

[27] Bai, S., Cai, Y., Chen, R., et al. Qwen3-VL Technical Report. arXiv:2511.21631, 2025.

[28] Kirillov, A., Mintun, E., Ravi, N., et al. Segment Anything. ICCV, 2023.

[29] Liu, S., Zeng, Z., Ren, T., et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV, 2024.

[30] Yang, L., Kang, B., Huang, Z., et al. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. CVPR, 2024.

[31] Wang, S., Leroy, V., Cabon, Y., et al. DUSt3R: Geometric 3D Vision Made Easy. CVPR, 2024.

[32] Leroy, V., Cabon, Y., Revaud, J. MASt3R: Grounding Image Matching in 3D. ECCV, 2024.

[33] Wang, J., Chen, M., Karaev, N., et al. VGGT: Visual Geometry Grounded Transformer. arXiv:2503.11651, 2025.

[34] Bajcsy, R. Active Perception. Proceedings of the IEEE, 1988.

[35] Batinovic, A., Petrovic, T., Bogdan, S. A Shadowcasting-Based Next-Best-View Planner for Autonomous 3D Exploration. IEEE Robotics and Automation Letters, 2022.

[36] Delmerico, J., Isler, S., Sabzevari, R., Scaramuzza, D. A Comparison of Volumetric Information Gain Metrics for Active 3D Object Reconstruction. Autonomous Robots, 2018.
