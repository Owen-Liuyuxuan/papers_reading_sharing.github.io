time: 20261118

# Arxiv Computer Vision Papers - 2026-11-18

## Executive Summary

## 今日概览
本次从181条完整分页元数据记录中得到180篇符合基础筛选条件的候选，按更新后的贡献与证据策略评估后选出10篇并完成PDF阅读。主题覆盖机器人感知、操作规划、人形动作跟踪、自动驾驶策略鲁棒性、农业导航、多模态定位，也纳入视频世界模型评估、视觉表征、少样本分割和扩散模型推理时控制。

核心阅读包括以分布式回报优势改进多动作人形跟踪的QTOP、用技能效果世界模型加速操作规划的Lazy-MOSAIC，以及以选择性传感器获取增强地点识别的方法。感知方向还有不确定性建模的占据预测与面向农业环境的长时程导航。更广义的发现论文中，SciWorldBench检验视频世界模型是否能理解科学实验；EBSG以文本和图像示例局部控制扩散采样，但标准配置测得约5.3倍基线延迟，且安全率指标和VLM评估存在边界。

阅读结果提示，论文中的实机或跨骨干实验应结合任务覆盖和比较范围解读；模拟结果、缩放规律和安全评估也各有适用边界。所有十篇入选论文均有逐页阅读证据、内部分析、公开介绍与独立卡片，报告已依据PDF证据整理。

---

## Table of Contents

1. [VoxSAOR: Semantic-Aware Occupancy Robustification via Aleatoric Uncertainty Modeling](#2611.20507v1)
2. [QTOP: Scaling Humanoid Motion Tracking with Quantile Critics](#2611.20791v1)
3. [FlowRow: Learning Navigation Behaviors in Agricultural Environments with Long-Horizon Goal-Conditioned Flow Matching Policies](#2611.21035v1)
4. [Lazy-MOSAIC: Accelerating Skill-Centric Manipulation Planning with Learned Skill-Effect World Models](#2611.21517v1)
5. [STA: A Split Temporal Adversary Method for Evaluating Robustness of Multimodal RL-based Driving Policies](#2611.20765v1)
6. [Selective Sensor Acquisition for Multimodal Place Recognition](#2611.20730v1)
7. [SciWorldBench: Do Video World Models Really Understand Scientific Experiments?](#2611.20899v1)
8. [State-Message Interaction Adapter for Cross-Domain Few-Shot Segmentation](#2611.21085v1)
9. [Learning a Task-Adaptive Low-Dimensional Semantic Space for Improved Visual Classification](#2611.20553v1)
10. [Example-Based Spatial Guidance for Training-Free Concept Erasure in Diffusion Models](#2611.20623v1)

---

## Papers

<a id='2611.20507v1'></a>
## [VoxSAOR: Semantic-Aware Occupancy Robustification via Aleatoric Uncertainty Modeling](https://arxiv.org/abs/2611.20507v1)

**Authors:** Mengfei Duan, Yuheng Zhang, Guoqiang Zhao, Yuanfan Zheng, Kailun Yang, Zhiyong Li

**Published:** 2026-11-16

**Categories:** cs.CV, cs.RO, eess.IV

**Abstract:**

Vision-based 3D occupancy prediction provides a dense semantic representation of scenes for autonomous driving and embodied perception, yet remains vulnerable to occlusion, depth ambiguity, and projection errors that induce spatially varying reliability in voxel features. Existing methods typically map voxel representations to semantic predictions without explicitly modeling such observation-induced feature variations. To address this limitation, we propose VoxSAOR, a semantic-aware occupancy robustification framework that models aleatoric uncertainty through feature-space proxies. It first quantifies Local Semantic Consistency (LSC) to derive a voxel-wise structural stability prior and uses cross-view semantic consensus to filter unreliable semantic correspondences. Within each semantic category, spatially matched stable and ambiguous voxels form stable-to-ambiguous feature transitions, which are modeled by a class-conditional diagonal Gaussian distribution, capturing both feature displacement and dispersion. Guided by these proxy distributions, Aleatoric Regime Ensemble Learning (AREL) trains independently parameterized prediction heads using feature perturbations at progressively increasing scales, with perturbation strength modulated by structural reliability. At inference, proxy modeling and sampling are disabled, and all heads process the same observed features; their output probabilities are then averaged to produce a deterministic consensus prediction without requiring sampling. Experiments across multiple datasets and occupancy backbones demonstrate the effectiveness of VoxSAOR in improving semantic occupancy prediction, long-tail recognition, and predictive calibration. The source code will be made publicly available at https://github.com/MengfeiD/VoxSAOR.

### 论文解读

#### 摘要翻译
VoxSAOR 针对视觉 3D 占据预测中的局部观测歧义，先从语义结构中估计哪些体素相对稳定、哪些更模糊，再用同类体素间的特征转移构建不确定性代理。训练时多个预测头在不同强度的代理扰动下学习；推理时关闭代理建模，以多个头对同一输入的概率平均得到确定性结果。作者报告它可接入不同占据预测骨干，并改善语义精度、长尾类别与置信度校准。

#### 方法动机分析
从图像恢复完整 3D 场景时，遮挡、深度不准和投影错误会令不同空间位置的证据可靠性不一，物体边界和细结构尤其如此。现有确定性预测器通常直接把体素特征映射到类别，未显式建模这种观测诱发的特征变化。论文的假设是，局部语义一致性可以作为结构稳定度线索；同类别且空间相近的稳定/模糊体素，其特征差可作为观测不确定性的经验代理。作者也说明该差异仍含上下文和模型误差。

#### 方法设计详解
输入图像经编码、视角变换和 3D 体素编码得到特征场。训练阶段的 SAUM 先计算 LSC：统计体素与其 3D 邻居标签的一致率，对面、棱、角邻居分别加权，归一化成稳定度分数。再用离线 OpenSeeD 生成的二维语义伪标签反投影至 3D，只保留二维与三维类别一致的候选。每类中按 LSC 选低分模糊点和高分稳定点，模糊点匹配同类别、空间最近的稳定点；两者特征差汇成通道均值与方差，并以类别条件的对角高斯分布表示。AREL 从该分布采样特征变化，建立多个独立占据头，各头扰动强度逐级增加，并按 LSC 调节幅度。所有头使用原占据标签监督。推理阶段停用伪标签、代理估计和随机采样，所有头接收同一原始特征，平均 softmax 概率得到结果。默认设置为 26 邻域、两条预测分支、基础扰动 α=0.01、每类四个模糊锚点和最低支持数 32。

#### 方法对比分析
相比在输出端估计熵或重复随机预测，VoxSAOR 的创新是先在体素特征空间构造稳定到模糊的变化代理，再用它训练多头。方法可接到多种现有骨干上，且推理不依赖采样；代价是训练需离线伪标签、额外多头参数。空间匹配不能完全剥离对象上下文和表示误差，因此代理并非真实校准的 aleatoric 分布。

#### 实验分析（精简版）
SemanticKITTI 隐藏测试中，VoxDet 的 IoU/mIoU 从 46.88/18.23 提升到 48.02/18.64；验证集 ECE/NLL 从 0.1626/0.8397 降至 0.1621/0.8298。Human360Occ 跨城设置下，MonoScene 的 mIoU 从 24.15 提升至 28.52。增益因骨干和类别而异，一些长尾类别下降；SGN 隐藏测试 mIoU 仅由 14.01 增至 14.37。隐藏测试结果使用作者复现的官方 VoxDet 代码；验证集则与不同占据骨干一起评估，适用范围仍是离线体素指标。

#### 实用指南
适合相机式 3D 占据、场景补全和体素级可靠性研究。复现需离线二维语义伪标签、对齐投影、体素邻域统计和同类空间匹配。作者采用 26 邻域、面/棱/角权重 1/0.1/0.3、两头、α=0.01；需依骨干重新校准锚点和扰动尺度。实际车辆部署仍需测延迟、显存和下游规划收益。

#### 总结
**核心思想：用局部结构差异训练抗歧义占据预测**

1. 计算体素局部一致性并筛选可靠语义点。
2. 匹配同类稳定/模糊特征，估计变化代理。
3. 用多强度扰动训练多个占据头。
4. 推理时平均同输入上的类别概率。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20507v1)
- [arXiv](https://arxiv.org/abs/2611.20507v1)

---

<a id='2611.20791v1'></a>
## [QTOP: Scaling Humanoid Motion Tracking with Quantile Critics](https://arxiv.org/abs/2611.20791v1)

**Authors:** Markus Peschl, Dana Kianfar, Daniel Dijkman

**Published:** 2026-11-16

**Categories:** cs.RO

**Abstract:**

Learning human motion tracking policies through reinforcement learning (RL) has emerged as a core capability that underpins a majority of recent progress in humanoid whole-body control. However, achieving generalization to unseen motions at high fidelity requires jointly scaling training to thousands of GPU and dataset hours. This work hypothesizes that motion tracking is inherently compute-bound because training on a diverse set of motions introduces learning interference, similar to multi-task RL. To tackle this, we propose Quantile Tracking Optimization (QTOP), which improves sample efficiency by introducing a distributional advantage estimate conditioned on state and reference pairs. QTOP outperforms on and off-policy RL baselines both in simulation and real-world tracking scenarios on multiple datasets by up to 10\%. Furthermore, we establish a scaling law for QTOP and find that, aside from scaling predictably, dataset size has minimal impact on performance for moderately sized GPU hour regimes.

### 论文解读
#### 摘要翻译
人形机器人可通过强化学习跟踪参考动作，但一个策略同时学习多种动作时，不同动作会造成不同状态访问和回报分布，标量价值函数可能难以给出一致有效的优势信号。QTOP为状态和参考条件预测回报分布，并依据实际回报相对该分布的位置更新策略。论文还分析训练计算量与动作数据量的缩放关系，并在Unitree G1上进行实机跟踪。
#### 方法动机分析
已有工作常通过增加动作数据与仿真算力改善跟踪，但两者通常一起增加，难以分辨哪个资源真正起作用。作者认为，跟踪更受优化计算限制，原因是不同参考动作之间的学习信号相互干扰。核心假设是，先在局部条件分布内判断一次回报的相对好坏，比直接使用跨动作的绝对回报差更能平衡策略更新。
#### 方法设计详解
QTOP保留PPO策略目标，替换评论家的训练信号。评论家不再只输出一个期望回报，而是对固定回报支持点输出概率分布；分布期望仍用于PPO需要标量价值的场景。训练样本得到实际折扣回报后，计算它在该状态条件回报分布中的累积分布概率，再将该分位点中心化为正负优势：高于条件中位数的回报给正优势，低于中位数则给负优势。这样回报尺度不同的动作可进行相对比较。训练开始先用标准GAE warmup，之后将标准化GAE与分布优势按权重混合，默认α=0.5，并使用Muon优化器。实验中的策略为四层MLP、约1150万参数；仿真训练预算为200 GPU小时，约等于八张A100并行训练一天。
#### 方法对比分析
与PPO标量评论家相比，QTOP保留相同策略优化形式，改变的是评论家表达及优势构造；与显式梯度手术相比，它不需要逐动作单独计算并修正梯度，保留PPO的小批量训练结构。它适用于参考动作异质、回报尺度差异大且可并行采样的跟踪问题，不能据此认为所有多任务干扰都被解决。
#### 实验分析（精简版）
在固定200 GPU小时预算下，QTOP在两个未见动作数据集上普遍优于PPO Adam/Muon及SAC、FastTD3；作者报告相对PPO约10%的跟踪误差改善。缩放律使用4、12、24小时动作集和最多约1300 GPU小时，拟合R²为0.981；测量范围内约700 GPU小时以内，增加计算通常比增加数据更有效。Unitree G1实测中，32种动作各重复三次，QTOP相对PPO将root-relative关键身体位置误差降低约8%。缩放结论不应外推到更大的数据规模。
#### 实用指南
复现需构造动作条件化回报分布，并保持GAE warmup、优势归一化和PPO目标一致；论文默认混合系数0.5、warmup 500次迭代，使用Muon。其评估控制50Hz，仿真采用8192并行环境，实机由外部工作站推理。将方法用于移动操作时，应检查长时接触、动作稀有性以及控制延迟下的回报分布校准。
#### 总结
核心思想：按动作条件分布评估回报
1. 评论家预测当前条件下的回报分布。
2. 把实际回报转成条件分位优势。
3. 与GAE混合后用PPO更新策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20791v1)
- [arXiv](https://arxiv.org/abs/2611.20791v1)

---

<a id='2611.21035v1'></a>
## [FlowRow: Learning Navigation Behaviors in Agricultural Environments with Long-Horizon Goal-Conditioned Flow Matching Policies](https://arxiv.org/abs/2611.21035v1)

**Authors:** Utkarsh Bajpai, Cédric Pradalier, Stéphanie Aravecchia

**Published:** 2026-11-16

**Categories:** cs.RO

**Abstract:**

Autonomous monitoring and intervention in semi-structured agricultural environments, such as orchards and vineyards, require mobile robots to navigate to locations across diverse field configurations. This entails performing navigation behaviors including in-row driving, row turning and skipping, and inter-row traversal. Recent learning-based approaches have enabled robust in-row navigation from onboard perception, but navigation across rows typically requires a priori knowledge of field geometry to specify and tune row-transition maneuvers, preventing arbitrary transitions between rows. We address this challenge by framing navigation in these environments as a goal-directed navigation task, where a long-horizon goal is expressed relative to the robot and a planner generates local paths based on the goal and the robot's perceptual observations. We present FlowRow, a flow-matching-based local planner which models the distribution of diverse agricultural navigation behaviors as $\mathrm{SE}(2)$ paths using the long-horizon goal and depth perception. To enable systematic evaluation, we introduce RowNavBench, a navigation benchmark for semi-structured agricultural environments developed in Isaac Sim, comprising 900 episodes across grape vineyards, lemon orchards, and apple orchards with varying row lengths, row and plant spacings, plant positions, and weed growth. FlowRow achieves 87% success, compared to 44% for the strongest generative behavior cloning baseline retrained on the same demonstrations. We will release our benchmark, data, and code upon acceptance.

### 论文解读
#### 摘要翻译
农业机器人要在作物行内行驶，也要在地头转弯、跳过若干行并跨到目标行。已有系统通常依赖已知田间几何或分别设计行间动作。FlowRow把任务写成目标条件导航：机器人只依据相对目标和深度观察生成局部路径。作者还建立了RowNavBench，并报告仿真性能及小规模零样本果园实测。
#### 方法动机分析
行间导航不是朝目标直线前进：目标即使空间距离近，也可能被作物行阻挡，必须先离开当前行、经过地头再进入另一行。固定几何动作要求已知行距和方向，容易限制到相邻行。FlowRow的假设是，局部深度与目标方向足以让策略在多步执行中自行组合行内行驶、转弯和跳行行为。
#### 方法设计详解
策略输入为机器人坐标系中的远距离目标，以及最近四帧深度图。目标距离裁剪至30米并与方位正余弦一起编码；深度经ResNet-18生成空间token，与目标token交给四层Transformer融合。条件flow-matching模块从噪声逐步生成12个SE(2)样条控制位姿，表达短距离局部路径。每步并行生成32条候选路径，再依据深度图构造的障碍间隙代价和终点目标距离选出较安全且有进展的一条，由纯追踪控制器跟踪；下一步重新观察并规划。为应对局部路径视野远短于目标距离，训练时多数样本的目标在当前演示路径片段之外，少数样本则让机器人学会近距离到达后停止。这样的双视野监督减少只朝目标贪心移动的倾向。
#### 方法对比分析
与只学行内跟踪或单独学习掉头相比，FlowRow统一处理多种农业导航行为；与使用地图和全局路径的规划器相比，它依靠局部深度和目标相对方位。它适合行结构可见但参数未知的环境，仍要求机器人能够估计自身位姿以转换目标坐标。
#### 实验分析（精简版）
RowNavBench含900个仿真回合，覆盖三类作物、15种场景和行内、转行、跨行任务。FlowRow总体成功率87%，同用178条农业演示训练的warm-start SanD-Planner为44%；在跨行任务中分别为82%和9%。仿真训练策略还在果园零样本运行8次，其中行内3次全成功、跨行5次中3次无干预完成。实机样本数有限，定位仍由RTK-GNSS提供，不能据此推断无定位系统下表现。
#### 实用指南
复现涉及AgriGen生成场景、178条专家轨迹、四帧深度历史、32条候选路径和每条12个SE(2)控制位姿。FlowRow在实机Husky A200与AGX Orin上以1.5Hz运行，使用ZED 2i深度相机。关键设计是双视野目标监督；固定短视野训练在长程跨行测试中仅有2%成功。FlowRow在实机推理频率为1.5 Hz。迁移到仓储或林下机器人时，需重新采集适配底盘的演示并检查位姿误差和深度遮挡。
#### 总结
核心思想：远目标引导局部路径生成
1. 编码目标方位和深度历史。
2. 生成多条SE(2)候选路线。
3. 按障碍间隙与目标进展筛选并滚动执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.21035v1)
- [arXiv](https://arxiv.org/abs/2611.21035v1)

---

<a id='2611.21517v1'></a>
## [Lazy-MOSAIC: Accelerating Skill-Centric Manipulation Planning with Learned Skill-Effect World Models](https://arxiv.org/abs/2611.21517v1)

**Authors:** Itamar Mishani, Mukund Krishnakumar, Oren Salzman, Maxim Likhachev

**Published:** 2026-11-16

**Categories:** cs.RO, cs.AI

**Abstract:**

Long-horizon manipulation tasks require a robot to perform a sequence of physical interactions with its environment, typically combining several different skills. Planning such behaviors efficiently is challenging: given a library of parameterized skills, solutions lie in a vast hybrid space of skill sequences and their parameters, and evaluating each candidate skill's feasibility and effect with a high-fidelity world model such as a physics simulator is computationally expensive. MOSAIC, a recent skill-centric approach, lets the skills guide a non-sequential search toward the regions of this space where they are likely to succeed; however, it pays the full cost of simulation for every parameterized skill it considers. In this work, we introduce Lazy-MOSAIC, a lazy planning framework that instead searches over hypothesized skill outcomes. A cheap learned skill-effect world model predicts, in a fraction of a millisecond, a skill's feasibility and the world state it produces. Lazy-MOSAIC then uses these predictions to focus expensive simulation on promising candidate skills and, when a hypothesis is invalidated, repairs only the affected part of the search, while ensuring that every skill in a returned plan is validated in simulation. We evaluate Lazy-MOSAIC on cluttered-shelf retrieval tasks of increasing difficulty and demonstrate it on a physical robot, showing that it plans substantially faster than baselines with fewer simulator queries, while matching or exceeding their success rate in every setting. Visit skill-mosaic.github.io.

### 论文解读
#### 摘要翻译
长时程操作要组合多种技能，但逐一用物理模拟评估候选动作代价很高。LAZY-MOSAIC先用廉价学习模型预测技能结果，再把较有希望的预测交给模拟器验证；预测失效时只修复依赖它的局部搜索图。返回的计划全部经过模拟认证。仿真任务中它减少模拟调用并加快规划，另展示了实机扳手检索。
#### 方法动机分析
MOSAIC以技能为单位搜索长程操作，但每个参数化技能都做高保真模拟，搜索范围一大就昂贵。单纯用学习世界模型替代模拟又会把错误状态写进完整计划。论文的关键假设是：学习模型足以筛选候选，最终安全性仍由模拟器把关；预测错了也能通过局部修复保留其他有效搜索结果。
#### 方法设计详解
系统维护两张图：认证图只存模拟实际执行过的技能轨迹，乐观图则存学习模型提出的成功概率、末态和不确定度。输入是机器人状态、对象特征与技能提案；DeepSets编码技能和本体状态，并汇集对象特征，三模型集成给出预测。算法在两图并集上搜索低代价路径，只模拟路径上第一个尚未认证的技能，并从已实现的前驱状态开始。若模拟结果与预测在对象位置、姿态和关节误差容限内相符，就把真实轨迹加入认证图；否则删除依赖错误状态的后续预测，保留已经实现的状态及其他无关假设，再重新规划。若候选路径反复失败，算法会认证当前较便宜的备选，也以非零概率随机认证候选，以避免搜索永远忽略某些技能。学习模型决定验证顺序，模拟结果决定计划内容。
#### 方法对比分析
与MOSAIC相比，主要变化是延迟而非省略模拟；与直接执行世界模型计划相比，它保留逐技能认证。相较只做学习排序的Ranked-MOSAIC，收益来自把评估延后到有希望路径上。该结构适合技能效果可离线学习、模拟调用昂贵且仍需可验证计划的半结构化环境。
#### 实验分析（精简版）
在ManiSkill3/SAPIEN的八种货架任务设置中，每种20个随机配置、600秒时限，LAZY-MOSAIC成功104/160次，MOSAIC为81/160；共有解设置里规划快2.1至6.6倍，模拟查询减少64%至69%。移除模拟认证的变体虽快，却仅成功33/160，说明加速建立在保留模拟验证上。主要量化证据来自仿真，实机只展示扳手取回。
#### 实用指南
实现时需分别记录预测状态与已实现状态，明确位置、姿态和关节容差，并确保模拟从实际前驱状态启动。论文的学习模型由3个宽度64网络组成，在约35000条提案上训练300 epochs；每次搜索批量评估128个提案，GPU评分约0.7毫秒。技能库及其成功率会限制可迁移范围。面向新工作区需重新验证离线模型，不能把仿真速度收益直接当作实机安全证据。
#### 总结
核心思想：预测筛选，模拟认证
1. 学习模型快速提出技能效果。
2. 搜索乐观路径，仅验证首个未认证技能。
3. 按真实结果局部修图，直至计划全认证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.21517v1)
- [arXiv](https://arxiv.org/abs/2611.21517v1)

---

<a id='2611.20765v1'></a>
## [STA: A Split Temporal Adversary Method for Evaluating Robustness of Multimodal RL-based Driving Policies](https://arxiv.org/abs/2611.20765v1)

**Authors:** Shenghao Xiong, Xiaolin Chang

**Published:** 2026-11-16

**Categories:** cs.CR, cs.LG, cs.RO

**Abstract:**

Multimodal reinforcement learning (MRL) offers a promising paradigm for autonomous driving by jointly leveraging complementary sensor observations to learn effective decision-making policies. However, its effectiveness critically depends on temporal alignment across modalities, making temporal inconsistency a fundamental vulnerability. Even subtle misalignment or latency can corrupt fused representations and ultimately lead to decision failures. Existing studies have explored adversarial attacks to induce cross-modal temporal inconsistencies, thereby enabling systematic evaluation of the temporal robustness of MRL policies. However, existing methods remain limited in generality, realism, and adaptivity. To address these limitations, we propose the Split Temporal Adversary (STA), which formulates multimodal temporal attacks as a unified decision process that jointly optimizes attack timing and modality selection under realistic policy-aware and budget-constrained settings. STA includes two mechanisms: a) a unified multimodal temporal perturbation mechanism that delays observations via historical states, removing the need for modality-specific noise; b) a split multimodal adversarial decision mechanism that jointly optimizes attack timing and modality selection through adaptive optimization while leveraging the victim policy's clean actions to identify sensitive decision patterns. We provide a theoretical analysis of the induced distribution shift and a modality selection principle. Experiments in the CARLA simulator show that STA more than doubles the collision rate of the strongest baseline against PPO while using 17.8\% fewer attack attempts, and that it generalizes to a VLM-based driving policy. These results reveal temporal consistency as a critical vulnerability, highlighting temporal robustness as essential for safe autonomous driving.

### 论文解读

#### 摘要翻译
论文提出 Split Temporal Adversary（STA），用多模态观测的时间错位攻击自动驾驶强化学习策略。攻击不修改图像或状态数值本身，而把选中模态替换成历史时刻的真实观测；攻击策略根据车辆状态、受害策略的干净动作和剩余预算，决定何时攻击、攻击哪种模态。作者在 CARLA 中发现该方法能以较少攻击次数提高碰撞率，并在一个视觉语言模型驾驶策略上观察到相似脆弱性。

#### 方法动机分析
摄像头、速度和航点等观测通常按时间戳同步，但同步器未必检查内容是否来自同一物理时刻。传感器延迟可让融合策略把“当前图像”与“过去状态”拼在一起，造成语义时间不一致。已有攻击常针对单一模态加噪，或固定随机选择攻击时刻，未充分利用受害策略反馈或有限攻击预算。STA 的假设是：历史观测仍然单独有效，但跨模态错时会扰乱依赖时间对齐的决策；通过自适应选择时机与模态，可以用有限干预暴露弱点。

#### 方法设计详解
输入包含 RGB 图像和数值状态（如速度、航点）。STA 用长度 n 的共享循环缓存保存完整多模态观测。攻击发生时，从缓存取 n 步前的指定模态，替换该模态当前值，其他模态保持不变；因此不需图像噪声或针对不同传感器设计扰动。攻击策略额外观察受害驾驶策略在干净观测上的动作和剩余预算。它输出两个决策：是否攻击、选择视觉还是数值模态。只有触发攻击且预算仍大于零时才执行替换并消耗一次预算。作者把两个决策分支分开优化，模态分支仅在实际触发攻击时更新，以应对碰撞奖励稀疏带来的信用分配困难。攻击策略使用 PPO 训练，受害策略固定；攻击策略训练步数为 800,000，延迟步数与缓存长度均为 10、攻击预算为 30。理论部分以分布变化、策略 Lipschitz 连续性和模态 Jacobian 解释延迟如何传导到动作变化；这依赖数学假设，不能直接视为真实系统敏感度测量。

#### 方法对比分析
相比图像/LiDAR 加噪，历史替换保留单模态内容的真实性，把干扰集中在模态间时间关系。相比随机或固定攻击时刻，STA 联合学习攻击触发和模态选择，并受预算约束。它测试的是策略对时间错位的鲁棒性，与空间扰动安全评测互补；不同受害策略下效果不一致。

#### 实验分析（精简版）
实验使用 CARLA 0.9.13、Town02 密集交通、固定晴天。PPO 策略上，STA 碰撞率为 35.6%，强基线 ADV 为 16.8%；每局攻击数 24.168 对 29.396，少 17.8%。结果是五个随机种子的均值与方差。单个 VLM 驾驶策略上，STA 为 62% 碰撞、26.98 次攻击，其他条件为 0–2%。但 DARRL 对 ADV 更脆弱，碰撞率 32.2%，高于 STA 的 18.6%。没有真实道路验证。

#### 实用指南
适合研究驾驶策略安全评估、传感同步和时延攻击的读者。复现需固定受害策略，明确延迟步数、缓存长度、攻击预算和碰撞判定窗口，并同时报告碰撞率与攻击次数。要迁移到真实车载系统，应加入传感器时钟、延迟分布和安全防护约束，并在多城镇、多天气和更多策略上评估；论文结论目前限于模拟器。

#### 总结
**核心思想：用真实旧观测打破模态同步**

1. 缓存完整多模态历史。
2. 自适应选择攻击时机与目标模态。
3. 用旧观测替换当前输入并消耗预算。
4. 观察碰撞率与攻击成本。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20765v1)
- [arXiv](https://arxiv.org/abs/2611.20765v1)

---

<a id='2611.20730v1'></a>
## [Selective Sensor Acquisition for Multimodal Place Recognition](https://arxiv.org/abs/2611.20730v1)

**Authors:** Margaux Edwards, Tobias Fischer, Michael Milford

**Published:** 2026-11-16

**Categories:** cs.RO, cs.CV

**Abstract:**

Multimodal Place Recognition (PR) can improve robustness, but most systems rigidly read and process every sensor for every query, even when the primary sensor is sufficient, with implications for energy consumption, second sensor update rate, and domain-specific considerations such as passive stealth. We introduce selective, controllable sensor acquisition, where a user specifies a desired second sensor usage rate and the system acquires the additional sensor only when the primary sensor's estimated performance is sufficiently poor. An online threshold adjustment achieves the requested usage rate by adapting to recent primary sensor performance, directing additional sensing towards queries where fusion is most likely to help without calibration specific to sensor modality, PR method, or environment. The approach requires only a PR method that produces a match score and applies to both single-frame and sequence-based PR. Our system, across multiple datasets, PR descriptors, and sensor combinations, recovers 55% of the available PR performance gain while using the second sensor on only 25% of queries, and 81% while using it on 50%. We further analyse alternative acquisition decision mechanisms and usage-rate control methods, and examine performance when the second sensor is stronger than, comparable to, or weaker than the primary sensor.

### 论文解读
#### 摘要翻译
多模态地点识别可以利用不同传感器的互补性，但许多系统每次都采集和处理全部传感器。本文提出选择性传感器采集：用户先指定希望启用第二传感器的比例，系统只在主传感器匹配看起来不可靠时才请求额外观测。在线阈值会根据近期主传感器表现动态调整，在多数据集和多种地点描述子上，以较低传感器使用率保留大部分融合收益。
#### 方法动机分析
传统融合通常在所有传感器已经被读取、编码和匹配后才决定如何加权，因此无法节省前端采集与处理成本。作者的核心想法是先评估当前主传感器结果，再决定是否付出第二传感器的代价。挑战在于不同数据集、传感器组合和描述子的分数尺度不同，固定阈值难以通用；核心假设是近期主传感器分数足以调整阈值，让额外观测集中到高风险查询。
#### 方法设计详解
处理流程是先用主传感器得到地点相似度分数，并从候选匹配估计当前结果可靠度。论文以空间不确定度SUE为例：取最高分的若干参考地点，依据其地图位置计算分布范围；候选地点越分散，主传感器越可能含糊。门控比较这一不确定度与阈值，低于阈值时只用主传感器，高于阈值才读取并处理第二传感器，再将两边分数融合得到地点预测。离线场景直接取整段分数的目标分位数；在线场景则用前400次查询的历史分布确定当前阈值，再按实际启用率和目标启用率的差异修正控制量。当前查询不进入其自身阈值历史，避免使用未来信息。默认前十个查询用来建立历史；请求率控制的是平均占用比例，不保证每个局部窗口精确命中。序列识别中已采集到的第二传感器分数还会在后续匹配窗口内继续参与。
#### 方法对比分析
与先采集再做多模态融合的方案相比，该方法把决策提前到第二传感器读取之前；与拒绝低置信匹配不同，它尝试用额外观测挽救查询。它兼容单帧与序列地点识别，依赖现有方法能够输出匹配分数。当前门控依据主传感器的不确定度，而不是直接预测第二传感器将带来多少收益。
#### 实验分析（精简版）
实验在三个数据集、20种传感器配对和十种地点描述子上进行。单帧匹配、第二传感器使用率10%时，平均Recall@1从主传感器基线79.16%提高到82.32%，随机采集为80.28%，Carson与KappaPlace-PT分别为80.39%和80.38%。在线请求10%、25%和50%使用率时，实际使用率保持在目标±1个百分点。以25%目标下的匹配召回为参照，方法平均处理时间29.40毫秒，低于随机采集的33.25毫秒；计算结果基于A100、batch size 1。
#### 实用指南
复现时需在同一数据集和识别器上比较固定占用率，报告召回率、真实启用率和计算成本。论文的在线控制使用400次历史窗口、步长0.005，前十次先积累样本。将其用于机器人或车辆时，可按电量、热成像或高成本雷达设定预算，但须检验传感器同步、突发失效和分布漂移；现有结果不能替代目标平台的能耗测量。
#### 总结
核心思想：不确定时才唤醒第二传感器
1. 用主传感器产生匹配与不确定度。
2. 按目标使用率动态设门控阈值。
3. 仅在低可信查询采集并融合第二传感器。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20730v1)
- [arXiv](https://arxiv.org/abs/2611.20730v1)

---

<a id='2611.20899v1'></a>
## [SciWorldBench: Do Video World Models Really Understand Scientific Experiments?](https://arxiv.org/abs/2611.20899v1)

**Authors:** Xue Yang, Haonan Cai, Xinyu Hang, Zhimeng Huang, Xuanyi Liu, Chuanmin Jia, Siwei Ma

**Published:** 2026-11-16

**Categories:** cs.CV

**Abstract:**

While the ability of video generation models to accurately depict scientific phenomena has received growing attention, determining whether these models genuinely understand the underlying principles remains a critical challenge. Existing benchmarks typically evaluate video samples in isolation, making it difficult to expose models that simply fit superficial visual priors. In scientific experiments, a small change in one condition produces a markedly different outcome even when the rest of the setup remains fixed. Capitalizing on this sensitivity, we introduce SciWorldBench, a benchmark comprising paired scientific experiments. Each pair varies a single condition in an otherwise fixed setup, yielding two mutually exclusive expected outcomes. Each generation prompt specifies the condition but omits the expected outcome. We evaluate ten text-and-image-to-video (TI2V) models using an independent vision-language model (VLM) that assesses each video without access to the condition text or the label identifying the correct outcome. A pair is scored as correct only when both videos depict their respective scientifically correct outcomes. Across all ten models, we find that the two videos in an experimental pair frequently depict the same outcome even though the conditions require different results. This recurring failure suggests that current video world models struggle to apply the underlying scientific principles reliably during generation. We will publicly release the benchmark dataset and open-source the evaluation code.

### 论文解读
#### 摘要翻译
论文提出 SciWorldBench，用成对科学实验检验视频生成模型是否会依据条件变化生成相应结果。每对实验只改变一项条件，两个结果应互斥；提示只给条件、不透露答案。十种模型常在两种条件下生成相同结果，说明逼真的单段视频并不等于掌握实验规律。
#### 方法动机分析
以往评测常单独判断一段视频是否合理，模型只要复现常见视觉先验就可能得高分，却未必理解为何改变材料、温度或操作会改变结果。作者的核心假设是：若模型真的利用条件，固定其余设置后，两个对照视频应分别呈现不同且正确的结果。成对设计因此把“看起来对”推进到“对条件变化有响应”。
#### 方法设计详解
基准先由模型产生候选实验，再经人工检查描述清晰度、科学有效性及结果能否在短视频中观察，最终留下222对、八类实验。每一对保持场景、步骤和镜头相同，只改变一个条件；一半共用首帧，让文字成为唯一线索，另一半则让首帧显示条件。视频提示不写预期结果，两侧分别生成。评估器均匀查看每段八秒视频的49帧，先检查目标是否可见、对象是否一致、操作是否完成，再判断两个候选结果各自是否出现。评估时不提供条件文字和答案标签，避免直接照抄。只有一对中的两段视频都通过前置检查，并分别显示各自正确结果，才算成对成功。
#### 方法对比分析
相较逐视频的物理合理性测试，SciWorldBench关注条件对结果的控制作用；相较只看运动是否像真的，它要求一组对照同时正确。共享首帧组还能区分模型是否真正响应文本条件。它适用于需要反事实敏感性的生成模型评测，但成对设计本身不能定位模型失败发生在推理还是视频合成环节。
#### 实验分析（精简版）
十种模型的成对准确率仅为1.4%至24.8%，单视频科学结果准确率最高为53.8%；模型在两种条件下重复同一结果的比例为55.7%至91.7%。把正确结果额外写入提示后，六种开源模型平均成对准确率由5.6%升至19.5%，仍有明显差距。实验支持“条件响应不稳定”，但评估依赖视觉语言模型判读，且样本限于人工筛选的短时实验。
#### 实用指南
若复现，应同时保留成对提示、首帧分组、前置可见性检查和逐视频盲评；不能只报告单视频准确率。论文列出所测模型并采用每段49帧的评估，但基准构造包含人工科学审核，复现时需要记录审核标准和评估器误差。迁移到机器人或自动驾驶时，可设计仅改变摩擦、负载或路面条件的输入对，检验预测是否随条件调整；这是应用设想，论文未直接验证。
#### 总结
核心思想：用成对结果检验条件响应
1. 固定实验设置，只改变一个条件。
2. 不给模型预期结果，分别生成两段视频。
3. 检查两段是否都满足前置条件且结果各自正确。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20899v1)
- [arXiv](https://arxiv.org/abs/2611.20899v1)

---

<a id='2611.21085v1'></a>
## [State-Message Interaction Adapter for Cross-Domain Few-Shot Segmentation](https://arxiv.org/abs/2611.21085v1)

**Authors:** Hang Xing, Yan Xia, Yongjiang Liu, Siqi Zhang, Xueming Ding

**Published:** 2026-11-16

**Categories:** cs.CV

**Abstract:**

Cross-domain few-shot segmentation adapts natural-image pretrained encoders to medical or industrial domains from a few pixel-annotated support masks. Within frozen DINOv3, intermediate per-head attention messages already carry target-background distinctions that baseline final-layer similarities and predictions do not consistently reflect. We propose the State-Message Interaction Adapter (SMIA) to learn how per-head messages should update each receiving token representation while keeping pretrained attention and FFN parameters frozen. Inserted right after per-head aggregation, SMIA conditions the transformation of each head's aggregated output (message) on each patch's pre-attention feature (state), allowing similar contextual messages to produce distinct or opposing updates depending on the receiving position. Because scarce support masks cover only limited state-message combinations, a shared low-rank bilinear map confines the learned interactions to low-dimensional state, message, and output subspaces. Added before each adapted block's frozen FFN, these updates yield adapted features that are refined by frozen FFNs and deeper blocks for linear readout. With only 0.51M trainable parameters and single-pass, support-free inference, SMIA achieves six-dataset mean mIoUs of 77.40% at 5 shots and 80.99% at 10 shots across 20 medical and industrial tasks, surpassing the strongest prior methods by +3.61 and +3.93 percentage points. In controlled 10-shot comparisons across three benchmarks on the same DINOv3 backbone, SMIA also outperforms parameter-matched AdaptFormer, LoRA, and ParaX with a better precision-recall balance.

### 论文解读
#### 摘要翻译
跨域少样本分割要用少数像素标注，把自然图像预训练模型迁移到医学影像或工业缺陷。作者观察到，冻结DINOv3的中间注意力消息可能已有目标与背景线索，但最终特征未必保留清楚边界。SMIA在每个注意力头聚合消息后，用接收位置自身的局部状态调节更新，同时保持预训练注意力和前馈层冻结。它在20项医学及工业任务上以少量可训练参数取得更高分割精度。
#### 方法动机分析
病灶边缘和细裂纹常与正常组织或材料纹理相似，单纯用末层特征相似度分类容易扩散到背景或漏掉目标。LoRA等参数高效方法通常做与token位置无关的低秩更新，不能针对“同一上下文消息落在不同patch上”作不同修正。SMIA的假设是，注意力消息提供上下文，而接收patch的局部状态能告诉模型如何将该上下文用于当前位置。
#### 方法设计详解
对每个patch，先取得注意力前的局部状态和各注意力头聚合的消息。两者分别映射到低秩子空间后做双线性交互：对角项配对状态与消息坐标，额外的非对角混合层表示跨维关系，再经输出投影形成残差更新。这样同样的上下文消息可以根据局部状态产生不同方向的更新。更新插在冻结注意力输出与冻结FFN之间，之后仍由预训练FFN及更深层继续处理。模型只训练适配分支和线性分割头；先用冻结特征拟合初始读出，再联合优化适配器和读出。DINOv3 ViT-S+/16配置使用12个block、rank 8，总可训练参数511,873，训练2000步Adam，学习率1e-4。测试时单次前向即可，不必把支持图像留在推理环节做检索。
#### 方法对比分析
与LoRA不同，SMIA不修改预训练注意力投影，而在每个head消息聚合后增加状态条件化更新；与冻结编码器后接大解码器相比，它逐层调整token表征，最终仅需线性读出。低秩结构控制少样本场景中的参数规模。该方法适合细粒度局部边界，但缺少显式全局部件关系建模。
#### 实验分析（精简版）
20项任务涵盖五个医学数据集和MVTec AD全部15类。六数据集总体mIoU在5-shot为77.40%，比此前最强方法高3.61个百分点；10-shot为80.99%，高3.93个百分点。受控同骨干10-shot比较中，SMIA均值79.48%，高于参数相近的ParaX 78.26%。RTX 5090上单图推理CVC为13.66毫秒、MVTec Bottle为23.50毫秒。Transistor等需要全局几何线索的异常仍有不足。
#### 实用指南
复现可从ViT-S+/16、12层适配、rank 8开始；需先拟合冻结特征线性头，再训练2000步并保持预训练注意力和FFN权重冻结。论文按5次嵌套支持抽样、统一查询集报告mIoU，并提供固定支持集的PEFT对照。部署到机器人视觉时，可将patch状态调节用于局部划痕、表面损伤或可见部件分割；实时性与跨相机域仍需在目标平台单独验证。
#### 总结
核心思想：用局部状态调制注意力消息
1. 提取patch状态与各头消息。
2. 低秩双线性交互生成位置相关更新。
3. 冻结骨干继续处理并线性读出掩码。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.21085v1)
- [arXiv](https://arxiv.org/abs/2611.21085v1)

---

<a id='2611.20553v1'></a>
## [Learning a Task-Adaptive Low-Dimensional Semantic Space for Improved Visual Classification](https://arxiv.org/abs/2611.20553v1)

**Authors:** Nikalal Kaldera, Wanqing Li, Chris Bunn, Philip O. Ogunbona

**Published:** 2026-11-16

**Categories:** cs.CV

**Abstract:**

Semantic visual classification commonly aligns visual embeddings with semantic spaces induced by pretrained text encoders. However, such spaces are typically high-dimensional, weakly aligned with visual feature geometry, and insufficiently discriminative for downstream classification tasks. We propose TaSS, a framework for learning a task-adaptive low-dimensional semantic space that preserves semantic structure while explicitly optimizing class separability. TaSS is constructed through a parameterized geometry transformation of pretrained text embeddings that enlarges inter-class margins via distance reshaping and improves intra-class compactness through prototype-preserving semantic constraints. The resulting semantic space maintains meaningful semantic relationships while producing a discriminative geometry tailored for classification. Frozen visual encoder features are subsequently aligned to TaSS using a lightweight MLP, enabling improved discriminative learning without fine-tuning pretrained backbones. We further provide theoretical analysis showing that TaSS preserves semantic consistency while enforcing enhanced inter-class separation and controlled intra-class variance. Extensive experiments on skeleton-based human action recognition, video classification, and image classification benchmarks demonstrate consistent improvements over existing semantic alignment approaches, with gains exceeding 3 percentage points on average and up to 8.5 points on challenging datasets.

### 论文解读

#### 摘要翻译
论文提出任务自适应语义空间 TaSS：把预训练文本编码器产生的类别语义向量，变换为维度合适、类别更易区分、同时尽量保留原语义关系的空间。冻结视觉骨干后，只需学习一个轻量映射，就能把图像、视频或人体动作特征对齐到该空间。作者在动作识别、视频分类和图像分类实验中报告了稳定收益。

#### 方法动机分析
文本编码器懂得“哪些概念接近”，但这种语言关系未必适合视觉分类。例如两种外观相似的鸟或动作方向相反的类别，在原文本空间里可能难以分开。传统降维只能解决维数问题，度量学习又可能牺牲类别间的语义结构。TaSS 的关键假设是：可以有控制地扩大易混类别的距离，再用类内原型保留同一类别不同描述所代表的变化。

#### 方法设计详解
流程分为语义空间构建和视觉特征对齐两段。第一步，为每个类别准备多条文字描述，并通过冻结文本编码器得到归一化向量。第二步，计算类别中心之间的角距离 δ，将其映射为 t=δmax(δ/δmax)^a；当 a 小于 1 时，原本较近的类别会获得更明显的相对扩张，但距离顺序仍保持。第三步，用映射网络拟合目标距离，并对每类描述作 K-means，形成多个语义原型；样本靠近其所属原型，原型也被约束在类别中心附近，以限制类内散布。第四步，冻结视觉编码器，把视觉特征经 MLP 映射到 TaSS。训练时从本类选择最相似原型为正例、其他类原型为负例，以温度参数控制的 InfoNCE 损失完成对齐。推理时取与各类别原型的最大余弦相似度作为分类分数。

#### 方法对比分析
相较直接对齐原文本空间，TaSS 显式重塑类间距离并保留多模态的类内结构；相较 SupCon、Circle Loss 等主要依靠视觉标签压紧同类、推远异类的度量学习，它还把文本类别语义关系作为优化目标。作者在 CUB-200 的 DINOv3 设置中比最强度量学习基线 Circle Loss 高 1.7 点。保序结论依赖映射误差小于目标距离最小间隔的一半，实验也单独评估语义距离排序，而非假设几何完全不变。

#### 实验分析（精简版）
论文覆盖骨骼动作、视频和图像分类。CUB-200 上，冻结 DINOv3 的 TaSS MLP 达到 88.4% Top-1，高于原文本空间 MLP 的 85.5%；最难的 30% 类别提升 8.5 个百分点。NTU120 的语义距离序关系违例率为 0.04%，CUB-200 为 0。实验在单张 RTX 4090 上进行。局限是固定类别集合、描述覆盖质量会影响构建效果，尚未验证开放词表或持续新增类别。

#### 实用指南
适合已有较强冻结视觉特征、类别集合相对明确且易混类别较多的分类任务。复现时需准备类别描述、选择文本编码器与每类原型数，并验证描述质量和类间隔超参数；论文中每类描述数在 50 至 450 间测试，300 条取得最佳结果。对机器人感知可尝试将类别语义与视觉特征对齐，但闭环安全和真实场景迁移仍需独立评测。

#### 总结
**核心思想：保留语义关系，拉开易混类别**

1. 为类别生成多样描述并编码。
2. 保序扩张近邻类别距离，同时约束类内多个原型。
3. 冻结视觉骨干，以 MLP 和对比损失对齐特征。
4. 通过原型相似度完成分类。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20553v1)
- [arXiv](https://arxiv.org/abs/2611.20553v1)

---

<a id='2611.20623v1'></a>
## [Example-Based Spatial Guidance for Training-Free Concept Erasure in Diffusion Models](https://arxiv.org/abs/2611.20623v1)

**Authors:** Younghwan Kil, Giung Nam, Joonhyeong Park, Jinwoo Shin, Juho Lee

**Published:** 2026-11-16

**Categories:** cs.CV

**Abstract:**

Text-to-image (T2I) diffusion models often synthesize policy-violating content, necessitating robust safeguards and rigorous evaluation frameworks. However, current approaches remain largely monolithic in both execution and evaluation, limited to coarse-grained categorization within broad unsafe concepts. Such approaches fail to account for the specific constituent factors of a violation, resulting in weak learning signals and allowing model collapse to bypass safety checks. To overcome these limitations, we propose Example-Based Spatial Guidance (EBSG), a training-free method that utilizes user-editable exemplar packs to provide granular spatial guidance for precise concept steering. Unlike existing methods that coarsely steer the entire image away from a broad concept, EBSG decomposes these categories into specific sub-concepts via exemplar text-image pairs, providing explicit, localized steering signals for more robust safety control. Furthermore, we introduce a vision language model (VLM)-based evaluation protocol that provides a fine-grained assessment, avoiding the pitfall of conventional binary evaluators that permit model collapse to bypass safety checks. Empirically, EBSG achieves state-of-the-art erasure performance across diverse safety datasets and concept categories, spanning four nudity benchmarks, seven I2P harmful-concept slices, and four MJA metaphor categories, while supporting multi-concept removal and transferring to a modern backbone, SD3.

### 论文解读
#### 摘要翻译
文本到图像扩散模型可能生成包含目标危险概念的内容。本文提出EBSG，一种不微调扩散模型、也不修改权重的推理时抑制方法。它用文本和图像示例构造可编辑的概念包，在采样过程中判断何时介入、定位目标子概念所在区域，再朝对应安全示例调整生成方向。作者在多种概念、数据集和扩散骨干上进行了评估，并讨论安全效果与推理开销的折中。
#### 方法动机分析
全图统一施加安全引导容易压制无关内容，也可能把画面推离原提示；单靠文字关键词则可能漏掉没有明确词语提示的视觉表达。作者的出发点是把危险概念拆成子类，给每类配置危险示例和安全锚点，再结合生成当前状态决定门控、定位和引导方向。这里的关键假设是，概念与提示在去噪过程中的对齐变化可用于判断介入时机，而不同子类的空间响应能够指向局部修正区域。动机回应了过度抑制和语义塌缩的挑战，但概念包覆盖不足和锚点偏移仍是实际瓶颈。
#### 方法设计详解
EBSG分为三个配合阶段。首先，概念级CAS门控比较当前输入方向和危险概念方向相对于无条件预测的夹角相似度，越过阈值后才激活；触发后保持开启，避免去噪中反复开关。其次，家族级空间探针结合文本关键词的交叉注意力和示例图像的视觉响应。两种响应阈值化、平滑后按像素winner-take-all分区，为不同子类建立互斥掩码。最后，每个子类将目标条件预测与安全锚点预测形成的方向差异，限制在自己的掩码内施加到采样更新上。多概念模式也会在概念间进行空间竞争。SD 1.4实验使用512×512图像、50步DDIM和CFG 7.5；图像探针使用CLIP ViT-L/14，空间网格为16×16或32×32，并需按概念校准阈值。SD 3与FLUX需要适配各自的注意力或特征读取接口。
#### 方法对比分析
与修改权重的概念编辑相比，EBSG保留原模型参数，控制发生在采样期间；与整图级负向引导相比，它根据家族掩码局部施加目标到锚点的调整；与只看文本的探针相比，它尝试利用图像示例覆盖隐式视觉线索。作者将贡献概括为门控、空间定位和条件引导的耦合，而非单独发明注意力定位。该设计适合需要按策略包更换概念、且能接受推理成本的图像生成场景；已有结果不足以说明它适用于所有骨干或低延迟部署。
#### 实验分析（精简版）
在UnlearnDiff、Ring-A-Bell、MMA-Diffusion和P4DN四个裸露基准上，EBSG的SR分别为97.2%、96.2%、84.2%和97.4%，FULL率分别为1.4%、2.5%、15.4%和2.6%。SR将SAFE与PARTIAL合并，不能当作严格无违规率。I2P七类单概念平均SR为73.6%，联合七概念为73.1%，而SAFREE联合设置为41.9%。消融中去掉CAS使平均SR降至63.8%；移除任一类探针也降低效果。SD 3的I2P平均SR为65.0%，FLUX为80.5%，但后者只与未防护模型比较。标准配置计算开销约为基线5.3倍；10步版本每图约2.79秒、SR约60.2%。评估器与328个人类标注多数标签的四分类精确一致率为62.2%，合并SAFE/PARTIAL后为82.6%。
#### 实用指南
复现时首先要建立目标与安全锚点示例包，并为具体骨干实现空间探针；论文给出SD 1.4的50步DDIM、CFG 7.5和16×16/32×32探针设置，但各概念阈值通过小规模保留集调节。部署前应分别测量安全率、提示保持、语义漂移、误报和端到端延迟，不能仅看SR。作者报告约5.3倍推理成本，低步数能省时却会改变效果。论文计划开放代码，正文尚未给出可验证仓库；跨到新骨干或新内容策略时，应重新校准锚点和空间特征接口。
#### 总结
核心思想：按子概念局部引导
1. 用CAS判断何时启动抑制。
2. 用文本与图像探针定位子概念区域。
3. 在局部掩码内朝安全锚点调整采样。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.20623v1)
- [arXiv](https://arxiv.org/abs/2611.20623v1)

---

