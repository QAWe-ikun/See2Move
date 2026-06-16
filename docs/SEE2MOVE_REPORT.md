# See2Move 技术报告

## 摘要

See2Move 研究指令条件下的主动视角调整问题：给定文本指令、当前相机 RGB 视图、深度图/Z-buffer、相机位姿和历史信息，预测下一步相机运动，使目标物体、目标区域或潜在操作区域在后续视角中更加可见。该任务面向交互式操作辅助场景，例如当用户希望在桌下、柜旁、遮挡区域或接触面附近完成摆放、抓取、检查时，系统需要主动判断相机应如何移动。

与长程 Vision-and-Language Navigation 不同，See2Move 当前聚焦短程视角微调：目标不是到达某个终点，而是在局部空间内改善可见性。本阶段基于 AI2-THOR 构建了自动 oracle 标注流程，实现了紧凑 RGB-D-text-pose baseline 和 Qwen3-VL 语义视觉模型，并完成初步消融实验。当前最佳模型为 `qwen_no_pose`，在 4,872 条、120 个场景的 scene-level split 数据上达到 top-1 accuracy 34.28%、mean predicted score 1081.48、positive gain rate 57.72%。该结果表明，Qwen3-VL 的 RGB+文本语义特征对视角收益预测有实际贡献，但当前数据规模、融合机制和错误分析仍不足以支撑顶会投稿级别结论。

## 1. 任务定义与边界

输入：

```text
instruction + RGB observation + depth/Z-buffer + camera pose + history
```

输出为离散相机动作：

```text
MoveAhead, MoveBack, MoveLeft, MoveRight,
RotateLeft, RotateRight, LookUp, LookDown, Stay
```

`Stay` 表示当前视角已经不应继续移动，或所有候选移动都会降低目标可见性。加入该动作后，模型不再被迫在无收益状态下执行相机运动。

当前阶段的监督目标是 object-level visibility improvement，即让指令对应目标物体在下一帧中暴露更多可见像素。项目最终目标更广：面向操作辅助时，应从 object-level 扩展到 region-level，例如支撑面、放置区域、抓取接触面、遮挡后方空间等。本报告中的结果应理解为 See2Move 的第一阶段单步主动视角预测，而不是完整操作规划系统。

### 与 VLN 的区别

VLN-CE 等视觉语言导航任务通常关注根据语言指令完成长程移动并到达目标位置。See2Move 关注的是局部视角质量：相机可能只需平移、旋转或俯仰一步，就能改善目标区域可见性。换句话说，VLN 的核心问题是“去哪里”，See2Move 的核心问题是“怎样看清”。

### 与 Next-Best-View / Active Perception 的区别

Next-Best-View 和 Active Object Detection 也研究主动移动相机获取更多信息。See2Move 的差异在于加入自然语言指令，使模型不仅需要判断几何可见性，还需要理解当前用户关心哪个物体或区域。当前实现仍较简单，主要使用目标物体可见像素作为 oracle；后续需要进一步引入操作区域、动作代价和多步 rollout，才能更接近完整 active perception 设置。

## 2. 相关工作定位

本项目与三类工作相关。

第一类是视觉语言导航。VLN、VLN-CE 和相关 waypoint prediction 方法关注从语言和视觉输入预测导航动作或路径。See2Move 借鉴其 RGB-D 环境和动作预测范式，但不以长程到达为目标，而是以局部可见性提升为目标。

第二类是主动感知与 Next-Best-View。该方向关注通过主动移动传感器减少不确定性、改善检测或重建质量。See2Move 与其共享“下一视角选择”的思想，但增加了 instruction conditioning，使模型需要根据文本目标选择应改善的区域。

第三类是具身问答与交互式感知。Embodied QA 和 interactive perception 要求 agent 主动移动以获得回答问题或操作所需的信息。See2Move 可被视为这类任务中的一个底层技能：在执行高层操作前，先预测一个更有利的观察视角。

因此，See2Move 的阶段性贡献不是提出新的通用导航算法，而是构建一个面向操作辅助的局部主动视角预测任务，并验证 RGB、深度、语言、相机状态和视觉语言模型特征在该任务中的作用。

## 3. 数据集构建

数据来自 AI2-THOR。生成流程如下：

1. 在 120 个 AI2-THOR 场景中轮转采样。
2. 随机选择可达位置、相机朝向和俯仰角。
3. 从当前视角中选择可见目标物体。
4. 保存 RGB 图像与深度图。
5. 枚举候选相机动作；新版动作空间包含 8 个运动动作和 1 个 `Stay` 动作。
6. 执行动作后重新统计目标物体可见像素数。
7. 选择可见像素提升最大的动作作为 oracle label。

每个候选动作的 oracle score 定义为：

```text
score(a) = visible_pixels_after(a) - visible_pixels_before
```

如果动作执行失败，score 记为 -1。当前数据集统计：

```text
records: 4872
scenes: 120
actions: 8 in completed reported runs, 9 after Stay relabeling
random-action baseline: 12.5% for 8-action runs, 11.1% for 9-action Stay runs
majority-label baseline: 20.9% before Stay relabeling
```

### 数据质量与局限

该自动标注流程可复现，但仍有明显限制：

- 样本量较小，4,872 条样本不足以充分训练多模态 fusion head。
- 动作类别不均衡，`MoveBack` 和 `LookUp` 样本较少，导致 balanced accuracy 明显低于 top-1 accuracy。
- oracle 只考虑目标物体可见像素提升，没有过滤“目标已充分可见”的样本，也没有建模移动代价和碰撞风险。
- score 使用绝对像素数，大物体天然可能获得更高 score；后续需要加入 relative gain。
- 当前没有按遮挡程度、物体大小、距离、场景类型做难度分层。

这些问题会直接影响结果解释。因此，当前数据集更适合作为中期验证集，而不是最终 benchmark。

## 4. 模型方法

### 4.1 Compact RGB-D-Text-Pose Baseline

该模型直接使用 RGB、深度图、文本和相机位姿：

- RGB 与 depth 拼接为 4 通道图像输入浅层 CNN。
- CNN 由 3 个卷积层组成，通道数为 32、64、128，随后使用 adaptive average pooling。
- 文本指令通过词表 embedding 编码，并对 token embedding 做 mask average pooling。
- pose 编码为 7 维向量：

```text
x, y, z, sin(yaw), cos(yaw), cameraHorizon / 90, isStanding
```

最终特征融合方式为直接拼接：

```text
f = concat(f_rgbd, f_text, f_pose)
logits = MLP(f)
```

该 baseline 的优点是简单、可解释、训练快；缺点是 RGB 编码能力弱，无法充分利用复杂视觉语义。

### 4.2 Qwen3-VL + Depth 模型

Qwen3-VL 作为冻结的 RGB+instruction 特征抽取器。输入格式为包含一张 RGB 图像和一段文本 prompt 的 chat template。实现中取最后一层 hidden states，并使用 attention mask 做 mean pooling，得到每条样本的 Qwen 特征。该特征离线保存，训练 action head 时不更新 Qwen3-VL 参数。

深度图不输入 Qwen3-VL，而是单独编码。深度预处理包括：

- 将 raw depth 裁剪到 `max_depth=5.0` 米。
- 归一化到 `[0, 1]`。
- 构造 depth、inverse depth、depth edge 三通道。
- 输入一个小型 residual depth CNN。

Qwen 模型当前融合方式仍是直接拼接：

```text
f = concat(f_qwen, f_depth, f_pose)
logits = MLP(f)
```

这解释了当前 full Qwen 不如若干消融变体的可能原因：融合头较浅，可能没有学会处理不同模态的尺度、噪声和冗余关系。该判断目前只是基于实验现象的解释，还需要 gated fusion、cross-attention、FiLM 或 modality dropout 等对照实验支撑。

## 5. 训练设置

所有实验采用 scene-level validation split，即训练和验证场景不重合。这样比 random split 更难，但更能测试泛化到新房间的能力。

主要配置：

```text
image size: 128 x 128
optimizer: AdamW
batch size: 16 for compact model, 64 for Qwen feature model
learning rate: 1e-3
weight decay: 1e-4 to 5e-4
early stopping: enabled
Qwen3-VL: frozen feature extractor
```

当前报告没有包含硬件型号、训练时间、loss curve 和混淆矩阵，这是后续正式论文版本需要补齐的实现细节。

## 6. 评价指标

分类指标：

- top-1 accuracy：预测动作是否等于 oracle 最优动作。
- top-2 / top-3 accuracy：oracle 动作是否在预测概率最高的前 2 或前 3 个动作中。
- balanced accuracy：每个动作类别 accuracy 的平均值，用于缓解类别不均衡影响。
- macro-F1：已在代码中加入，后续重跑实验会输出。

视角收益指标：

- mean predicted score：模型预测动作对应 oracle score 的平均值。
- positive gain rate：模型预测动作使目标可见像素增加的比例。
- mean best score：oracle 最优动作平均 score。
- mean predicted relative gain：已在代码中加入，定义为 `score / initial_visible_pixels`，后续重跑实验会输出。

mean predicted score 和 positive gain rate 比 top-1 更贴近任务目标，因为模型即使没有选择 oracle 最优动作，只要选择了能改善视角的动作，仍有实际价值。但现有指标仍不完整：score 未考虑动作代价，positive gain rate 约 57% 也意味着仍有约 43% 样本没有改善视角。正式系统需要进一步提升该指标，并加入多步 rollout 评估。

## 7. 实验结果

### 7.1 Compact Baseline 与消融

| 模型 | 输入 | Top-1 | Balanced | Top-2 | Top-3 | Positive gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full | RGB + depth + text + pose | 32.31 | 25.59 | 53.78 | 68.89 | 55.75 |
| no_depth | RGB + text + pose | 31.33 | 24.39 | 53.01 | 68.67 | 55.09 |
| no_text | RGB + depth + pose | 27.38 | 18.57 | 46.55 | 64.62 | 52.35 |
| no_pose | RGB + depth + text | 29.13 | 20.91 | 49.51 | 68.24 | 54.00 |
| no_rgb | depth + text + pose | 33.19 | 26.58 | 54.98 | 70.87 | 57.61 |
| rgb_only | RGB | 24.32 | 15.16 | 42.28 | 58.93 | 55.64 |

文本是紧凑模型中最关键的模态，去掉文本后 top-1 从 32.31% 降至 27.38%。pose 也有明显贡献。`no_rgb` 优于 full，说明当前浅层 CNN 对 raw RGB 利用不稳定；这一解释仍需更大 CNN backbone 或 ResNet 系列对照验证。

### 7.2 Qwen3-VL 结果与消融

| 模型 | 输入 | Top-1 | Balanced | Top-2 | Top-3 | Mean score | Positive gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen full | Qwen3-VL + depth + pose | 32.53 | 23.23 | 54.22 | 70.65 | 999.23 | 56.85 |
| Qwen gain-selected | Qwen3-VL + depth + pose | 29.46 | - | - | - | 913.65 | 57.17 |
| Qwen no_depth | Qwen3-VL + pose | 33.30 | 26.44 | 54.87 | 69.66 | 1008.06 | 57.72 |
| Qwen no_pose | Qwen3-VL + depth | 34.28 | 26.51 | 55.53 | 69.11 | 1081.48 | 57.72 |
| Qwen no_qwen | depth + pose | 28.70 | 21.16 | 48.19 | 65.28 | 870.07 | 54.98 |

`qwen_no_pose` 是当前最强模型，top-1 accuracy 达到 34.28%，mean predicted score 达到 1081.48。去掉 Qwen 特征后 top-1 降至 28.70%，说明 Qwen3-VL 的语义 RGB+文本特征确实有效。

但 Qwen full 低于 `qwen_no_depth` 和 `qwen_no_pose`，说明当前简单拼接融合不够稳健。该现象不能简单解释为 depth 或 pose 无用，更合理的判断是：小数据集上，多模态拼接可能引入冗余或噪声，fusion head 未能学到可靠的模态选择机制。

## 8. 错误分析与当前不足

当前模型的主要失败模式包括：

- 少数动作学习不足：`MoveBack` 和 `LookUp` 样本少，模型倾向预测高频动作。
- full fusion 不稳定：加入更多模态并不总是提升性能。
- positive gain rate 仍偏低：最优模型约 57.72%，说明仍有大量动作不能改善视角。
- 缺少可视化 case study：目前没有展示成功和失败样例，也没有混淆矩阵。
- 缺少难度分层：尚未按遮挡程度、目标大小、距离和场景类型分析模型表现。

对当前最佳 `qwen_no_pose` checkpoint 的混淆矩阵分析显示，macro-F1 为 40.51%。其中 `MoveBack` 的 support 为 87，但 predicted count 为 0，precision、recall 和 F1 均为 0；`LookUp` 的 F1 也只有 20.41%。相反，`MoveAhead` 和 `LookDown` 被明显过预测。这说明模型并非均匀掌握了动作空间，而是偏向高频且收益较稳定的动作。

闭环 rollout 进一步暴露了该问题：compact full 模型在 50 个 episode、3 步 rollout 中 mean total gain 为 -266.82，positive total gain rate 为 40.00%。这说明单步分类准确率不能直接代表闭环控制效果。为避免模型在无收益状态下被迫移动，当前主线已加入 `Stay` 输出头，默认数据生成配置会直接产生 9 动作标签。

这些不足意味着当前结果更适合作为阶段性 proof-of-concept，而非最终论文结论。

## 9. 投稿前改进计划

如果目标是 AAAI、ICCV 或 CoRL 等会议，最小可行补充包括：

1. 扩大数据规模到 20k-50k，并对少数动作做重采样或加权采样。
2. 过滤 trivial 样本，或按初始可见度将样本分为 easy、medium、hard。
3. 增加 relative gain、macro-F1、confusion matrix 和 per-class precision/recall。当前代码已支持这些指标；旧实验需要重新评估或重跑后才会产生对应字段。
4. 实现 greedy multi-step rollout，评估连续执行 3-5 步后的累计可见性收益。当前已提供 compact policy 的 AI2-THOR 在线 rollout 脚本。
5. 加入至少一个 strong baseline，例如 CLIP/Qwen frozen feature + LSTM/Transformer action head，或 ResNet-18/34/50 RGB-D baseline。
6. 设计更强的融合机制，如 gated fusion、FiLM、cross-modal attention 或 modality dropout，并与 concat fusion 对比。当前已提供 Qwen gated fusion + modality dropout 配置作为第一版对照。
7. 增加成功/失败 case visualization，展示 RGB、depth、instruction、预测动作、oracle 动作和实际 gain。
8. 将 object-level visibility 扩展到 operation-region visibility，更贴近摆放、抓取和检查任务。

## 10. 结论

本阶段完成了 See2Move 的基本闭环：AI2-THOR oracle 数据生成、RGB-D-text-pose baseline、Qwen3-VL 特征模型、主实验与消融分析。实验表明，指令语义、深度几何和视觉语言特征均对主动视角预测有贡献；其中 Qwen3-VL 特征显著优于浅层 raw RGB 特征。当前最佳结果来自 `qwen_no_pose`，top-1 accuracy 为 34.28%，mean predicted score 为 1081.48，positive gain rate 为 57.72%。

同时，结果也暴露出明确短板：数据规模较小、类别不均衡、oracle 标签仍粗糙、融合机制过浅、错误分析不足。因此，本报告应定位为项目中期技术报告。See2Move 的任务方向成立，但若要达到正式投稿标准，还需要在数据规模、融合方法、指标体系和对比实验上继续扩展。

## 11. 新增补充实验入口

按类指标与混淆矩阵：

```bash
bash scripts/analyze_ai2thor_predictions.sh \
  --checkpoint runs/ablations_4k8/qwen_no_pose/checkpoint_best.pt \
  --kind qwen \
  --output runs/analysis/qwen_no_pose_confusion.json

bash scripts/summarize_prediction_analysis.sh runs/analysis/qwen_no_pose_confusion.json
```

生成当前主线 10k Stay-aware 数据并训练：

```bash
bash scripts/generate_ai2thor_oracle.sh
bash scripts/train_ai2thor_policy.sh
bash scripts/train_ai2thor_qwen_policy.sh configs/train_ai2thor_qwen3vl.yaml
```

按难度分层的数据集与预测分析：

```bash
bash scripts/stratify_ai2thor_analysis.sh \
  --records /mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl \
  --output runs/analysis/dataset_strata.json

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

Compact policy 三步 rollout：

```bash
bash scripts/rollout_ai2thor_policy.sh \
  --checkpoint runs/ai2thor_policy_4k8_simple_e35/checkpoint_best.pt \
  --policy model \
  --episodes 50 \
  --steps 3 \
  --output runs/rollouts/simple_model_3step.json
```

可与 random 和 oracle rollout 对比：

```bash
bash scripts/rollout_ai2thor_policy.sh --policy random --episodes 50 --steps 3 --output runs/rollouts/random_3step.json
bash scripts/rollout_ai2thor_policy.sh --policy oracle --episodes 50 --steps 3 --output runs/rollouts/oracle_3step.json
```

Qwen gated fusion：

```bash
bash scripts/train_ai2thor_qwen_policy.sh configs/train_ai2thor_qwen3vl_gated.yaml
bash scripts/compare_ai2thor_runs.sh
```
