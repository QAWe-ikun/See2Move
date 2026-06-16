# See2Move：面向摆放模型的可见性驱动相机运动预测

## 摘要

现有 Vision-Language-Action（VLA）和 World Action Model（WAM）工作大多面向真实机器人操作、室内导航或通用具身控制，关注如何根据视觉和语言指令直接生成操作动作或导航动作。然而，在 Technical Artist（TA）和三维场景编辑工作流中，许多操作并不是从“直接摆放”开始，而是需要先获得一个更清晰、更少遮挡的观察视角。对于 SceneReVis 这类场景重排或物体摆放模型，若当前相机视角无法清楚观察目标物体、支撑面或操作区域，后续摆放预测可能受到遮挡和局部几何缺失的影响。

本文提出 See2Move，一个为摆放模型提供前置视角辅助的相机运动预测框架。给定文本意图、当前 RGB 视图、深度图和相机位姿，See2Move 预测下一步相机动作，使目标物体或操作区域在下一视角中更可见。与传统 Vision-Language Navigation（VLN）方法不同，See2Move 不以到达目标位置或路径跟随为主要目标，而是直接预测每个候选相机动作带来的可见性收益（visibility gain），并选择预期收益最大的动作。本文利用 AI2-THOR 仿真环境自动生成监督信号，通过 instance segmentation mask 计算目标物体在当前视角和候选动作后视角中的可见像素变化。模型融合冻结的 Qwen3-VL hidden state、深度图编码和相机位姿特征，训练一个轻量 gain-score policy，并通过 Stay 动作和 stay-threshold 策略减少无收益或负收益相机移动。

## 关键词

Technical Artist；场景摆放；SceneReVis；视角调整；相机运动预测；可见性提升；深度图；AI2-THOR

## 1 引言

### 1.1 研究背景

随着大规模视觉语言模型和具身智能技术的发展，智能体已经能够在一定程度上理解视觉场景、语言指令和动作之间的关系。VLA 模型尝试将视觉输入、文本指令和低层动作统一到一个端到端框架中，典型应用包括机器人抓取、移动操作和交互式任务执行。与此同时，VLN 方法研究智能体如何根据语言指令在室内环境中移动，并在连续环境中执行低层导航动作。

然而，TA 和三维场景编辑任务中的需求与传统机器人操作或导航任务并不完全相同。在 DCC 软件、游戏引擎或仿真环境中，操作对象通常是已经存在的三维场景。对于 SceneReVis 这类摆放或场景重排模型，核心目标是根据场景上下文生成合理的物体位置、朝向或重排方案。但在执行摆放预测之前，系统往往需要先获得足够清楚的观察视角，以理解目标物体、支撑面、遮挡关系和局部几何结构。

例如，当用户希望将椅子放到桌子下方时，当前相机视角可能被桌面、桌腿或其他家具遮挡。若摆放模型只能基于当前受遮挡视角进行推理，就可能无法准确判断桌下空间是否可用、目标区域是否被占用，以及物体之间的空间关系是否合理。因此，一个面向摆放模型的前置视角调整模块是有价值的：它不直接完成摆放，而是预测下一步相机移动，为后续摆放模型提供更清晰、更少遮挡的输入观测。

### 1.2 问题动机

在真实机器人场景中，模型通常需要从二维图像中估计三维结构，例如深度、相机位姿、点云或物体几何关系。近年来，视觉几何模型可以从图像中恢复相机参数、深度图和三维点云，为机器人和 embodied AI 提供重要几何先验。但在 TA 所处的三维创作环境中，大量几何信息本身就是场景的一部分。DCC 软件、游戏引擎和仿真平台通常可以直接提供 mesh、depth buffer、object transform、camera pose、instance id 和 object visibility 等信息。

这意味着面向 TA 和场景摆放的 agent 不必完全依赖外部三维重建模型，而可以直接利用创作环境中已有的几何信号。See2Move 正是基于这一设定：它利用当前 RGB、深度图和相机位姿，学习如何移动相机以提升目标区域的可见性。该模块可以作为 SceneReVis 等摆放模型的上游辅助组件，在摆放预测前主动寻找更合适的观察视角。

### 1.3 本文贡献

本文提出 See2Move，一个面向摆放模型的可见性驱动相机运动预测框架。本文的主要贡献如下：

1. 提出一个面向摆放模型前置感知的任务定义，将语言条件下的相机运动预测建模为目标可见性提升问题，而不是传统路径跟随或 waypoint 预测问题。

2. 构建基于 AI2-THOR 的 oracle 数据生成流程，利用仿真环境中的 instance mask、深度图和相机位姿自动获得候选动作后的目标可见像素变化，避免人工标注下一步相机动作。

3. 设计一个多模态 gain-score policy，融合 Qwen3-VL hidden state、深度图编码和相机位姿特征，直接预测每个候选相机动作的可见性收益。

4. 引入 Stay 动作和 stay-threshold 推理策略，使模型能够在预期移动收益不足时保持当前视角，从而减少无收益或负收益相机移动。

5. 将 See2Move 定位为 SceneReVis 等摆放/场景重排模型的前置辅助模块，为后续摆放预测提供更清晰、更少遮挡的观察输入。

## 2 相关工作

### 2.1 Vision-Language Navigation

Vision-Language Navigation 研究智能体如何根据自然语言指令在环境中移动。早期 VLN 任务多基于离散导航图，智能体在预定义节点之间移动。VLN-CE 将任务扩展到连续环境，使 agent 需要在更接近真实物理空间的场景中执行低层移动动作。后续工作进一步探索 waypoint 预测、拓扑规划和跨模态 Transformer 等方法，以提升连续环境中的导航能力。

此类方法为语言条件下的视觉运动决策提供了重要基础，但其主要目标通常是完成路径跟随或到达目标位置，评价指标包括 success rate、SPL、路径长度等。相比之下，See2Move 关注的是局部视角调整。模型不需要完成长程导航，而是判断下一步相机动作是否能提升目标物体或操作区域的可见性，从而为后续摆放模型提供更好的输入视角。

### 2.2 Vision-Language-Action Models 与 World Action Models

VLA 模型试图将视觉、语言和动作统一到一个端到端模型中。以 RT-2 为代表的工作将大规模视觉语言模型迁移到机器人控制，使模型能够根据图像和语言指令输出机器人动作。WAM 相关工作进一步探索从视频、轨迹或世界模型中学习动作条件下的状态变化。这些方法通常面向机器人 manipulation、移动机器人控制或通用 embodied control。

尽管 VLA/WAM 方法可以理论上扩展到相机控制，但现有研究较少直接面向 TA 场景中的摆放前视角调整任务。对于 SceneReVis 这类摆放模型而言，动作预测并不总是等价于直接移动物体；在很多情况下，系统需要先调整视角以获得更可靠的局部几何和遮挡信息。See2Move 可视为 VLA/WAM 思路在 TA 场景中的一个具体化：动作空间不是机械臂末端执行器控制，而是相机移动与视角调整；优化目标不是任务成功率本身，而是下一步动作带来的目标可见性收益。

### 2.3 三维几何感知、仿真环境与摆放模型

在真实世界设置中，模型通常需要从 RGB 图像中估计深度、相机位姿或三维结构。视觉几何模型展示了从一张或多张图像预测相机参数、深度图和点云的能力，为缺乏显式三维信息的场景提供了重要补充。然而，在 DCC 软件、游戏引擎和仿真环境中，许多三维信息已经由环境直接维护。

AI2-THOR 是一个可交互三维室内环境，能够提供 RGB 图像、深度图、实例分割和 agent pose 等信息。本文利用 AI2-THOR 的 instance segmentation mask 计算目标物体在当前视角和候选动作后视角中的可见像素数，并将二者差值作为动作 gain score。相比从二维图像中额外重建三维几何，本文方法直接使用仿真环境中的原生几何和可见性信息，因此数据生成成本更低，监督信号也更贴合任务目标。

与摆放模型的关系上，See2Move 不替代 SceneReVis 等模型的摆放预测能力，而是补足其前置感知环节。摆放模型通常需要理解目标区域、支撑关系、遮挡关系和场景上下文。See2Move 通过预测更合适的相机动作，使后续摆放模型能够在更清晰的视角下进行推理。

## 3 方法

### 3.1 任务定义

给定当前观测状态：

```text
o_t = {I_t, D_t, p_t, x}
```

其中，`I_t` 表示当前 RGB 图像，`D_t` 表示深度图，`p_t` 表示相机位姿，`x` 表示文本指令或摆放意图。模型需要从候选动作集合 `A` 中选择下一步相机动作：

```text
A = {MoveAhead, MoveBack, MoveLeft, MoveRight,
     RotateLeft, RotateRight, LookUp, LookDown, Stay}
```

本文不将任务简单建模为动作分类，而是预测每个候选动作带来的可见性收益：

```text
g(a) = V_after(a) - V_current
```

其中，`V_current` 是目标物体或目标区域在当前视角中的可见像素数，`V_after(a)` 是执行动作 `a` 后目标在新视角中的可见像素数。模型输出每个动作的预测收益 `g_hat(a)`，并选择预测收益最大的动作：

```text
a* = argmax_a g_hat(a)
```

在与 SceneReVis 等摆放模型结合时，See2Move 的输出动作可用于更新相机视角；更新后的 RGB-D 观测和相机位姿再输入摆放模型，从而提升后续摆放预测的可见性条件。

### 3.2 Oracle 数据生成

本文使用 AI2-THOR 自动生成训练数据。对于每个采样状态，首先随机选择场景、相机位置、相机朝向和目标物体。随后记录当前 RGB 图像、深度图、相机位姿和文本指令。对每个候选动作，环境临时执行该动作，计算目标物体在新视角中的 instance mask 面积，并将相机恢复到原始状态。

当前可见像素数由 AI2-THOR 的 instance segmentation mask 得到。环境返回的 `event.instance_masks` 包含每个物体实例对应的二值 mask。对于目标物体 `target_id`，mask 中像素值求和即为该目标在当前相机视图中的可见像素数：

```text
V = sum(mask(target_id))
```

候选动作的监督分数定义为：

```text
score(a) = visible_pixels_after(a) - visible_pixels_before
```

如果所有移动动作的收益均不为正，则 oracle label 设为 `Stay`。因此，`Stay` 的 GT score 为 0，表示保持当前视角不会提升可见性，但可以避免负收益移动。

### 3.3 模型结构

See2Move 采用多模态融合结构。文本指令与 RGB 视图首先通过冻结的 Qwen3-VL 模型提取 hidden state，用于表示高层语义和视觉上下文。深度图通过轻量卷积网络编码，用于提供距离、遮挡和局部几何线索。相机位姿通过 MLP 编码，用于表示当前相机在环境中的空间状态。

三类特征经过拼接后输入 gated fusion 模块，得到统一状态表示。最后，gain-score head 输出每个候选动作对应的预测收益：

```text
[g_hat(MoveAhead), ..., g_hat(Stay)]
```

该结构使模型能够同时利用语言意图、视觉语义、深度几何和相机姿态信息，判断不同相机动作对目标可见性的影响。

### 3.4 训练目标

模型训练目标包含两个部分。第一部分是 score regression loss，使模型直接拟合每个候选动作的真实 gain score。第二部分是 ranking loss，使 oracle 最优动作的预测分数高于其他候选动作。整体损失为：

```text
L = lambda_score L_score + lambda_rank L_rank
```

当前实现中，分类交叉熵辅助项被关闭，使模型主要学习连续 gain score，而不是将任务退化为动作标签分类。这一设计与任务目标更加一致，因为在视角调整中，多个动作可能都能带来正收益，关键在于比较它们的预期收益大小。

### 3.5 Stay 阈值推理

由于模型输出的是连续 gain score，推理时可能对某些低收益动作给出轻微正分，从而导致不必要移动。为此，本文引入 stay-threshold 策略：

```text
if max_a in moving_actions g_hat(a) <= tau:
    choose Stay
else:
    choose argmax_a g_hat(a)
```

其中，`tau` 是绝对 gain 阈值。该策略不依赖推理阶段的真实目标 mask，因此可用于没有 oracle visibility 的部署设置。实验中可通过扫描不同阈值，分析 positive gain rate、negative gain rate 和 mean predicted score 之间的权衡。

### 3.6 与摆放模型的接口

See2Move 可作为摆放模型的前置模块。给定用户摆放意图和当前场景观测，See2Move 首先预测下一步相机动作。如果模型判断当前视角已经足够好，则输出 `Stay`；否则移动相机以获得更高可见性的视角。随后，SceneReVis 等摆放模型可以使用更新后的 RGB-D 视图、相机位姿和场景状态进行物体摆放或场景重排推理。

这一设计将“看清楚目标区域”和“生成摆放方案”解耦：See2Move 负责选择更适合观察和判断空间关系的视角，摆放模型负责在更可靠的输入基础上生成具体摆放结果。

## 4 实验与结果

### 4.1 实验设置

本文在 AI2-THOR 环境中构建可见性驱动的相机运动预测数据集。数据集覆盖厨房、客厅、卧室和浴室等 120 个室内场景，共包含 5000 条 oracle 记录。每条记录包含文本指令、当前 RGB 图像、深度图、相机位姿、目标物体信息、候选动作集合以及每个候选动作对应的可见性收益。候选动作集合包含 9 类动作：

```text
MoveAhead, MoveBack, MoveLeft, MoveRight,
RotateLeft, RotateRight, LookUp, LookDown, Stay
```

目标物体类型覆盖 CounterTop、Cabinet、Bed、Shelf、Chair、SideTable、DiningTable、Sofa、Sink、Toilet、Bathtub、Desk 等多种室内物体。数据生成时，AI2-THOR 提供 instance segmentation mask，本文通过目标物体 mask 的像素数量计算当前视角和候选动作后视角中的目标可见面积。候选动作的监督分数定义为动作后的可见像素数减去当前可见像素数。

模型训练采用 Qwen3-VL 提取的冻结 hidden state 作为语言与视觉语义特征，深度图通过轻量 CNN 编码，相机位姿通过 MLP 编码。策略头输出每个候选动作的 predicted gain score。训练目标由 score regression loss 和 ranking loss 组成。训练过程中使用 scene split 选择最佳 checkpoint；本章中的离线评估使用统一的 5000 条 records 进行比较，以保证不同模型在同一批状态上评估。

### 4.2 评价指标

本文使用以下指标评估模型：

1. **Accuracy**：预测动作与 oracle label 一致的比例。
2. **Macro F1**：各动作类别 F1 的宏平均值，用于衡量动作预测是否均衡。
3. **Top-2 / Top-3 Accuracy**：oracle 动作是否出现在模型预测分数最高的前 2 或前 3 个动作中。
4. **Mean Predicted Score**：模型最终选择动作对应的真实可见性收益平均值。
5. **Positive Gain Rate**：模型选择动作后真实可见性收益大于 0 的比例。
6. **Negative Gain Rate**：模型选择动作后真实可见性收益小于 0 的比例。
7. **Safe Balanced Gain Score**：综合 macro F1、positive gain rate 和 nonnegative gain rate 的综合指标。
8. **Stay Predicted Count / Stay Recall**：模型选择 Stay 的次数及其对 oracle Stay 样本的召回率。

对于摆放模型辅助任务而言，accuracy 并不是唯一目标。由于多个动作可能都带来正收益，模型即使没有预测到 oracle label，也可能产生有用视角。因此，mean predicted score、positive gain rate 和 negative gain rate 更直接反映 See2Move 是否能为后续 SceneReVis 等摆放模型提供更好的观察视角。

### 4.3 Gain-Score 训练与分类训练对比

早期模型将任务建模为 9 类动作分类。该模型在全量记录上的 accuracy 为 0.217，macro F1 为 0.099，mean predicted score 为 642.64，positive gain rate 为 0.555。该结果表明，单纯预测 oracle label 难以充分利用候选动作之间的收益差异，并且容易出现动作分布塌缩。

改用 gain-score regression/ranking 后，模型直接预测每个候选动作带来的可见性收益。在不使用 Stay 阈值时，该模型的 accuracy 提升到 0.458，mean predicted score 提升到 1773.59，positive gain rate 提升到 0.685。与分类模型相比，gain-score 模型在平均可见性收益上提升约 2.76 倍，说明直接优化动作收益比动作分类更贴合本文的视角调整目标。

| Model | Accuracy | Macro F1 | Mean Predicted Score | Positive Gain Rate | Top-2 Acc | Top-3 Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen classification | 0.217 | 0.099 | 642.64 | 0.555 | 0.414 | 0.565 |
| Qwen gain-score | 0.458 | 0.325 | 1773.59 | 0.685 | 0.655 | 0.763 |

该对比说明，See2Move 的核心优势不在于学习某个固定动作标签，而在于学习不同动作对目标可见性的相对收益。对于后续摆放模型而言，选择一个能显著提升目标区域可见性的动作比严格匹配 oracle label 更重要。

### 4.4 Stay 阈值分析

由于 gain-score 模型输出连续分数，某些低收益动作可能被预测为轻微正收益，从而导致不必要的相机移动。为此，本文引入 stay-threshold 机制：当所有非 Stay 动作的预测收益均不超过阈值时，模型选择 Stay。

本文扫描了 0、100、200 和 500 四个阈值。结果如下：

| Stay Threshold | Accuracy | Macro F1 | Mean Predicted Score | Positive Gain Rate | Negative Gain Rate | Safe Score | Stay Predicted Count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.458 | 0.326 | 1775.66 | 0.685 | 0.307 | 0.543 | 11 |
| 100 | 0.457 | 0.328 | 1773.96 | 0.683 | 0.304 | 0.543 | 38 |
| 200 | 0.456 | 0.332 | 1777.62 | 0.679 | 0.296 | 0.545 | 98 |
| 500 | 0.426 | 0.324 | 1731.92 | 0.611 | 0.229 | 0.528 | 781 |

阈值 500 虽然显著降低了 negative gain rate，但模型过于保守，Stay 预测次数上升到 781，导致 positive gain rate 和 accuracy 明显下降。阈值 200 在保持较高 mean predicted score 的同时，将 negative gain rate 从 0.307 降低到 0.296，并取得最高 safe balanced gain score。因此，后续实验默认采用 stay-threshold = 200。

### 4.5 消融实验

为分析不同模态对模型性能的贡献，本文在相同 gain-score objective 和相同 stay-threshold = 200 设置下进行消融实验。对比模型包括：

1. **Full**：使用 Qwen3-VL hidden state、深度图和相机位姿。
2. **No Depth**：去掉深度图输入。
3. **No Pose**：去掉相机位姿输入。
4. **No Qwen**：去掉 Qwen3-VL 视觉语言特征，仅保留深度图和位姿。

消融结果如下：

| Model | Accuracy | Macro F1 | Mean Gain | Positive Gain | Negative Gain | Safe Score | Stay Pred | Stay Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.4560 | 0.3320 | 1777.62 | 0.6790 | 0.2960 | 0.5452 | 98 | 0.0678 |
| No Depth | 0.4356 | 0.3112 | 1659.26 | 0.6640 | 0.3120 | 0.5277 | 96 | 0.0565 |
| No Pose | 0.4398 | 0.3215 | 1718.64 | 0.6668 | 0.3124 | 0.5329 | 84 | 0.0395 |
| No Qwen | 0.1912 | 0.0625 | 776.45 | 0.5652 | 0.4284 | 0.3654 | 0 | 0.0000 |

从结果可以看出，Full 模型在所有主要指标上均取得最佳性能。去掉深度图后，mean gain 从 1777.62 降至 1659.26，negative gain rate 从 0.296 上升至 0.312，说明深度图提供了遮挡、距离和局部几何信息，有助于判断移动后是否能看到更多目标区域。去掉相机位姿后，mean gain 降至 1718.64，negative gain rate 上升至 0.3124，说明位姿信息对相机动作的空间效果建模也有帮助。

最显著的下降来自 No Qwen 设置。去掉 Qwen3-VL hidden state 后，accuracy 降至 0.1912，macro F1 降至 0.0625，mean gain 降至 776.45，negative gain rate 上升到 0.4284。这说明语言和 RGB 语义特征是识别目标对象、理解指令意图和选择有效视角的关键。仅依赖深度和位姿虽然能提供几何信息，但缺乏目标语义，因此难以判断应该围绕哪个对象或区域调整相机。

### 4.6 结果讨论

实验结果支持本文的三个核心判断。第一，直接预测候选动作的 gain score 比动作分类更适合 See2Move 的任务目标。对于视角调整任务，多个动作可能同时带来正收益，oracle label 只是其中收益最大的一个动作；因此，连续收益预测能更好地表达动作之间的优劣关系。第二，Qwen3-VL 语义特征、深度图和相机位姿具有互补作用。语义特征帮助模型定位目标和理解指令，深度图提供遮挡与几何信息，位姿帮助模型理解当前相机状态与动作后变化。第三，Stay 阈值可以在一定程度上降低负收益移动，但阈值过高会使模型过于保守，减少本应执行的有效相机移动。

从为 SceneReVis 等摆放模型提供辅助的角度看，See2Move 的价值主要体现在提升输入观测质量。Full 模型在 stay-threshold = 200 时获得 1777.62 的平均真实可见性收益，positive gain rate 达到 0.679，说明模型在多数情况下能够选择提升目标可见性的动作。这意味着后续摆放模型可以在更清晰的视角下推理目标区域、支撑面和遮挡关系。

## 5 局限性与未来工作

### 5.1 局限性

当前实验仍存在若干局限。首先，当前评估主要基于 AI2-THOR oracle records 的离线单步动作预测，尚未完整验证 See2Move 接入 SceneReVis 后对最终摆放质量的提升。其次，Stay 类别仍然较难预测，即使使用 stay-threshold = 200，Stay recall 仍只有 0.0678，说明模型对“不移动”的判断仍不充分。再次，当前 gain score 使用目标物体 instance mask 生成监督信号，这适合仿真训练和离线评估，但真实推理时并不依赖 GT mask；未来可进一步研究预测当前可见度或结合分割模块的方案。最后，当前实验以单步相机动作作为主要评估对象，多步视角调整和与摆放模型闭环结合仍是后续工作重点。

### 5.2 未来工作：从显式目标到隐式操作区域

当前 See2Move 主要处理显式目标任务，即文本指令中存在较明确的目标物体，例如“观察笔记本电脑”“移动到能看清椅子的位置”或“查看柜子”。在这类设置中，AI2-THOR 可以通过目标物体的 instance mask 提供明确的可见像素监督。因此，当前 visibility gain 可以直接定义为目标物体执行动作前后的可见像素差值。

然而，真实 TA 和摆放模型场景中经常存在隐式目标任务。此类任务没有明确的单一目标物体 instance，也不一定存在可直接使用的 GT mask。例如，“寻找桌子下面的空间”“检查床边是否有可放置区域”“观察柜子内部空间”或“判断沙发旁边是否能放边桌”关注的是一个操作区域、空间关系或可用支撑区域，而不是某个具体物体本身。对于 SceneReVis 这类摆放模型，这类隐式目标尤其重要，因为摆放决策往往取决于目标区域是否可见、是否空闲以及局部几何是否合理。

因此，后续工作将从显式目标物体可见性迁移到隐式操作区域可见性。一个可能方向是利用仿真环境中的 mesh、depth buffer、object transform 和 camera pose 自动构造区域级监督，例如桌面下方空间、物体之间的空隙、支撑面附近区域或容器内部区域。另一方向是让模型从文本意图中预测隐式目标区域，并学习该区域在不同视角下的可见性或可操作性。通过这一迁移，See2Move 将不仅能帮助摆放模型看清具体物体，还能帮助其主动寻找更适合执行摆放和场景重排的空间区域。

## 参考文献

[1] Kolve, E., Mottaghi, R., Han, W., et al. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv:1712.05474, 2017.

[2] Krantz, J., Wijmans, E., Majumdar, A., et al. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments. ECCV, 2020.

[3] Brohan, A., Brown, N., Carbajal, J., et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818, 2023.

[4] An, D., Wang, Y., Wang, R., et al. ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments. IEEE TPAMI, 2024.

[5] Wang, J., Chen, M., Karaev, N., et al. VGGT: Visual Geometry Grounded Transformer. arXiv:2503.11651, 2025.
