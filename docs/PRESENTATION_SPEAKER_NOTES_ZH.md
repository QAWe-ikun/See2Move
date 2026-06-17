# See2Move 项目汇报演讲稿

## Slide 1  封面

大家好，我汇报的项目是 See2Move，主题是多模态智能系统中的主动视角选择。我们的目标是为三维摆放和场景编辑任务提供一个前置模块：当当前视角被遮挡时，自动预测下一步相机应该怎么移动。

## Slide 2  背景与问题

很多摆放失败并不是模型完全不会放，而是当前视角看不清。例如要把椅子放到桌子下面，系统需要先找到能观察桌下空间的视角。See2Move 解决的就是这种“先看清，再操作”的问题。

## Slide 3  与已有工作的区别

传统 VLN 更关注导航到目标点，VLA/WAM 更关注机器人操作动作。我们的任务关注相机动作之后目标是否更可见，所以它更适合作为 Tech Artist agent 或摆放模型的视角调整模块。

## Slide 4  任务定义

任务输入是文本指令、RGB 图、深度图和相机位姿，输出是下一步相机动作，包括移动、旋转、俯仰和 Stay。核心目标不是单纯匹配动作标签，而是选择能提升可见性的动作。

## Slide 5  整体技术路线

整体流程是：在 AI2-THOR 中采样场景和目标，枚举候选动作得到 oracle gain，再提取 Qwen3-VL、深度和位姿特征，训练模型预测每个动作的 gain score。推理时选择预测收益最大的动作，如果收益不足则 Stay。

## Slide 6  AI2-THOR Oracle 自动标注

我们利用 AI2-THOR 自动标注数据：对同一个初始视角临时执行每个候选动作，统计目标 instance mask 的可见像素变化。动作收益就是执行后可见像素数减去执行前可见像素数，所有移动动作无正收益时标注为 Stay。

## Slide 7  数据集与动作空间

当前主实验使用 5000 条 Stay-aware oracle records，覆盖 120 个 AI2-THOR 房间场景。动作空间包含 8 个相机移动/旋转/俯仰动作，加上 Stay，一共 9 类。

## Slide 8  模型结构

模型融合三类信息：Qwen3-VL hidden state 提供文本和视觉语义，Depth CNN 提供遮挡和几何信息，Pose MLP 提供相机位置和朝向状态。最终输出 9 个动作的 predicted gain score。

## Slide 9  训练目标与 Stay 推理策略

相比直接做动作分类，我们改为预测每个动作的连续收益，并加入 ranking loss。这样更符合任务目标，因为多个动作可能都有正收益，模型需要学会排序而不是只学一个离散标签。

## Slide 10  实验设置与评价指标

实验采用 scene-level split，验证集房间和训练集房间不同。除了 accuracy 和 macro F1，我们更关注 mean predicted gain、positive gain rate 和 negative gain rate，因为它们直接反映相机动作是否真的改善了可见性。

## Slide 11  主实验结果

Qwen3-VL Full 取得最好结果：mean gain 达到 1777.62，positive gain rate 为 0.679，negative gain rate 降到 0.296。相比 No Qwen，说明语义特征对理解“要看清哪个目标”非常关键。

## Slide 12  消融实验

消融结果显示，去掉 Qwen 性能下降最明显；去掉深度图或位姿也会降低收益并增加坏动作。也就是说，语义、几何和相机状态三者结合效果最好。

## Slide 13  Stay 阈值与类别均衡

Stay 用来避免无收益移动，但它比较难预测。默认 stay-threshold 为 200 时，在保持较高 mean gain 的同时降低了 negative gain rate；类别均衡训练能提升 Stay recall，但会牺牲平均收益。

## Slide 14  定性案例分析

这里展示的是模型单步预测案例，每一行是一个样本，三列分别是 RGB、Depth 和结果图。它们用来直观看模型是否真的根据目标语义和遮挡关系选择了更好的视角。

## Slide 15  3-5 步 Greedy Rollout Demo

多步 greedy rollout 用来观察模型连续执行时的表现。结果显示前几步通常能提升可见性，但连续贪心也可能过度移动；我们也生成了 GT greedy demo，作为 oracle 上界进行对比。

## Slide 16  当前局限与下一步计划

目前局限主要是 Stay 预测、多步闭环稳定性，以及从显式目标到隐式空间任务的迁移。下一步计划接入 SceneReVis，让 See2Move 为摆放模型提供更好的观察视角，并进一步扩展到“寻找桌下可用空间”这类隐式任务。

