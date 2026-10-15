time: 20261015

# Arxiv Computer Vision Papers - 2026-10-15

## Executive Summary

## 今日研究概览

今天入选的论文集中在机器人操作与可部署感知：一类工作试图让策略在长程执行、接触变化或形态差异下保持可靠；另一类工作构建动作条件世界模型、力觉基准与无障碍目标数据，改善机器人从观测到行动的闭环能力。多篇论文把模拟数据、真实平台验证和明确的失败案例并列呈现，显示跨域泛化仍是核心难题。

在操作策略方面，RAIN根据预测的动作风险决定是否调用生成式修正，在真实任务中报告72.5%成功率，对比ACT的55%；Back on Track将进度监测与短时恢复规划结合，在十项LIBERO任务中把成功率从72.6%提升至81.5%，但世界模型也会出现对象预测错误。双臂擦拭工作把覆盖几何和MILP调度合并，报告相较顺序规划约9倍的规划速度；PolyMorph则展示单一控制器适配多种腿式形态的可能性。

数据与表征方向同样值得关注。SonoWorld用潜在动作预测探头运动后的超声序列，在留出视频上提升生成指标；NavAble发布面向盲人导航的无障碍目标数据及可扩展合成生成流程，合成数据改善了分割结果。ForceBench系统比较力觉策略，RoboSpan研究视觉与触觉的层级融合；Coordi-π和连续体驱动平台分别从双臂多阶段动作表示与器械执行机构设计推进操作能力。

建议优先通读 Back on Track 与 RAIN，理解长程失败恢复和按需计算；关注 SonoWorld 可了解动作条件世界模型如何处理医学影像动态；NavAble适合研究数据集与仿真数据生成的读者。ForceBench、RoboSpan、PolyMorph及Coordi-π则适合比较传感器融合、形态泛化和多阶段操作策略。以上结果均受论文任务、数据与硬件条件约束，尤其真实测试规模和装配类任务表现需要结合各自局限解读。

---

## Table of Contents

1. [Refine Actions If Needed: Adaptive Action Chunk Refinement for Robot Manipulation](#2610.19765v1)
2. [SonoWorld: Towards Interactive Pose-Controllable Ultrasound World Models](#2610.19327v1)
3. [NavAble: A Large-Scale Dataset and Synthetic Data Generation Pipeline for Blind Navigation](#2610.19330v1)
4. [Back on Track: Deployment-time Policy Improvement via World Model Planning](#2610.19579v1)
5. [Dual-arm Cooperation for Tabletop Wiping with Temporal Object Removal](#2610.19527v1)
6. [ForceBench: Benchmarking Force Integration for Robotic Manipulation](#2610.19604v1)
7. [PolyMorph: Learning Locomotion Policy for Cross-Embodiment Adaptation](#2610.19615v1)
8. [RoboSpan: Learning Modular Coordination Policies for High-DoF Whole-Body Mobile Manipulation with Teleoperation](#2610.19594v1)
9. [Design and Characterization of a Unified Continuum Robot Actuation Platform](#2610.19798v1)
10. [Coordi-pi: Coordination-Aware Dual-Hand Action Representation for Bimanual Dexterous Manipulation](#2610.19753v1)

---

## Papers

<a id='2610.19765v1'></a>
## [Refine Actions If Needed: Adaptive Action Chunk Refinement for Robot Manipulation](https://arxiv.org/abs/2610.19765v1)

**Authors:** Hanlin Liu, Zichen Zhou, Jiacheng Zhang, Liuyuan Na, Jin Jin, Guangyan Chen, Yufeng Yue, Gen Li, Fangqiang Ding

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Computationally lightweight manipulation policies predict executable actions with low inference latency, but prediction errors can limit task success. Iterative generative policies can achieve higher task success rates, but repeated network evaluations increase inference latency. Refining action proposals from computationally lightweight policies can combine these strengths, but the amount of refinement needed varies across states. We introduce RAIN (Refine Actions If Needed), a framework that predicts action errors at different refinement levels to decide whether and how much to refine each executable proposal, before performing any refinement. We train an ensemble to predict these errors by comparing action proposals and their refined versions with expert actions from offline demonstrations, without requiring explicit labels of refinement needs. Calibration balances uncertainty, computation, and temporal consistency. We evaluate RAIN on eight ManiSkill tasks and four real-robot tasks. In simulation, our policy achieves 66.6% success, approaching the best evaluated diffusion-policy success rate with 5.71 times lower policy latency. On the real robot, success increases from 55.0% with Action Chunking with Transformers (ACT) to 72.5%, with comparable average successful completion time. Matched controls show that dynamic routing maintains the strongest observed fixed-refinement success level with 26.6% fewer refiner evaluations. Project page: https://rain-policy.github.io/.

### 论文解读

#### 摘要翻译

让轻量动作策略只在预测到误差风险时调用昂贵的迭代修正。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：轻量策略推理快，却会在复杂接触和状态变化时产生错误动作；扩散式或迭代生成策略可修正动作，但每一步都多次调用网络。RAIN 的关键假设是，离线专家数据足以训练一个误差估计器，在真正执行修正前判断提案质量与所需修正等级。

#### 方法设计详解

输入当前观测和轻量策略生成的动作块。集成预测器对多个候选修正层级估计动作相对专家动作的误差；训练时比较原始提案及其逐级修正版，不需要人工标注“何时修正”。校准过程共同权衡不确定度、额外计算与跨时刻决策稳定性。控制器据此直接接受提案或选择修正深度，再执行动作块并滚动到下一观测。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

与固定深度迭代相比，按状态分配计算；与只用轻量策略相比，它在风险高时保留生成式修正能力。贡献在于把动作错误估计前置到修正之前，使预算可按需使用。

#### 实验分析（精简版）

在八个 ManiSkill 任务及四个真实机器人任务评估。模拟中报告 66.625% 成功率，动作推理延迟约 0.322 ms/action；真实任务中 RAIN 为 72.5%，对比 ACT 的 55%。真实执行总时长约 17.57 秒，对比 ACT 约 17.17 秒，说明成功率提高并未消除系统延迟差异。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现应先得到轻量基策略和离线专家动作，再构造提案/修正版误差监督，训练集成预测器并调校校准参数。部署时记录接受率、修正深度、端到端延迟和任务成功率；只有在模型推理与动作执行接口兼容时，才容易迁移到其他机械臂。 训练数据来自离线专家演示；部署报告单次动作约0.322 ms，因而需按目标控制周期核算额外修正开销。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：按需修正动作

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19765v1)
- [arXiv](https://arxiv.org/abs/2610.19765v1)

---

<a id='2610.19327v1'></a>
## [SonoWorld: Towards Interactive Pose-Controllable Ultrasound World Models](https://arxiv.org/abs/2610.19327v1)

**Authors:** Zefan Yang, Zheyuan Zhang, Boris Maihe, Yikang Liu, Hang Yu, Ankush Mukherjee, Xiao Chen, Pingkun Yan, Terrence Chen, Shanhui Sun

**Published:** 2026-10-14

**Categories:** cs.CV

**Abstract:**

Robotic ultrasound can standardize image acquisition and reduce operator workload and variability, but policy learning remains limited by the lack of realistic, controllable simulation environments. While controllable world models have advanced rapidly in natural-image domains, ultrasound world modeling remains challenging because each frame captures only a cross-section of unobserved 3D anatomy, causing small probe motions to induce large appearance changes. Existing approaches rely on robot-tracked poses and latent-space prediction, but do not directly generate the observations required for interactive control. We propose SonoWorld, a causal ultrasound world model that learns video dynamics from unlabeled clinical videos and grounds them to 6-DoF control using a small set of freehand clinical scans tracked with an EM sensor. SonoWorld is trained in three stages: pretraining a latent-action-conditioned video diffusion model on around 6.9 million unlabeled multi-organ ultrasound frames; grounding the learned video dynamics to metric probe poses with a small EM-tracked thyroid dataset; and applying distribution matching distillation to enable causal, long-horizon autoregressive generation with selectively caching eviction tokens at inference time. We evaluate controllability through pose-following fidelity, round-trip consistency, and axis-specific control under synthetic actions on held-out tracked scans, and further validate it with a user study. In model-based action discrimination, SonoWorld correctly ranks the target-directed probe motion over an equal-magnitude opposite motion in up to 74.8% of trials. To our knowledge, SonoWorld is the first causal, pose-controllable ultrasound world model learned primarily from freehand clinical videos, providing a data-driven interactive environment for robotic ultrasound data acquisition and policy evaluation.

### 论文解读

#### 摘要翻译

从手持超声视频学习可交互的超声世界模型与潜在动作。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：超声成像受探头姿态、接触和组织形变共同影响，单帧重建无法说明探头移动后会看到什么。直接从视频学习控制又缺少可复用动作表示。作者假设，跨视角视频中的视觉变化和潜在动作可共同约束一个能预测未来观测的模型。

#### 方法设计详解

系统从手持超声视频提取图像及时间相邻变化，编码成状态潜变量；训练潜在动作表示来概括探头运动引起的帧间变化，再以条件生成模型预测后续超声帧。模型学习输入图像和潜在动作到未来图像的映射，并以视频预测目标约束动态一致性。推理时可在潜空间提出动作，滚动预测结果，形成用于交互规划的世界模型。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

相比静态超声分割或单帧合成，SonoWorld关注动作条件下的时序变化；相较依赖精确运动标注的模型，它从手持视频中学习潜在动作。其创新重点是将可预测未来观测与动作表示结合，证据仍主要是模型预测及下游交互实验。

#### 实验分析（精简版）

论文在超声视频数据上评估预测与潜在动作学习，在500段器官均衡留出视频中，加入潜在动作使PSNR提高3.60 dB、SSIM提高0.083，相比不含潜在动作的模型；这衡量视频预测质量，不是诊断准确率。训练流程包含视频采样、潜变量编码和动作条件预测，需保持训练/测试病例隔离。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现重点是视频质量、时间间隔定义、潜在动作维度、预测损失及跨病例划分。将方法迁移到机器人视觉伺服时，可把动作条件未来预测用于候选轨迹比较；迁移至其他成像模态需重新检查动态因素是否由视频充分观测。 训练包含大规模未标注多器官视频的预训练、33个EM追踪扫描的后训练，并在器官均衡的500段留出片段评估；推理采用因果视频生成。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：用动作预测超声变化

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19327v1)
- [arXiv](https://arxiv.org/abs/2610.19327v1)

---

<a id='2610.19330v1'></a>
## [NavAble: A Large-Scale Dataset and Synthetic Data Generation Pipeline for Blind Navigation](https://arxiv.org/abs/2610.19330v1)

**Authors:** Hochul Hwang, Jahir Sadik Monon, Soowan Yang, Shiven Umeshbhai Patel, Anh Nhat Hong Nguyen, Kien Nguyen, Keshav Garg, Dylan Gage, Duretti Hordofaa, Anshu Anjna, Eshed Ohn-Bar, Donghyun Kim

**Published:** 2026-10-14

**Categories:** cs.CV, cs.RO

**Abstract:**

Reliable recognition of accessibility-critical objects (e.g., audible pedestrian signals, door-activation buttons, and handrails) is essential for assistive navigation technologies that support safe, independent mobility for blind and low-vision (BLV) users. Yet these categories are underrepresented in existing large-scale vision datasets, and the few datasets that include them suffer from limited class coverage, inconsistent annotations, and poor diversity. We introduce NavAble, a large-scale dataset spanning 11 accessibility-critical object classes, combining 41K curated open-source images with 8K newly collected, densely annotated real-world images. To extend scale and diversity, we develop a synthetic data generation pipeline that renders high-fidelity 3D assets across 37 environments under diverse styles and viewpoints, yielding 452K frames with rich ground-truth annotations from 565 assets. The pipeline minimizes manual effort by combining automated web image crawling, VLM-based filtering, SAM 3D-based asset generation, and convenient camera trajectory configuration, enabling large-scale data generation at a diversity unattainable through manual collection. Experiments show that fine-tuning existing segmentation models on the NavAble Dataset achieves reliable accessibility-object segmentation (>80% mIoU on our held-out test set). We also demonstrate the effectiveness of synthetic data augmentation, which yields up to +3.5 mIoU over real-world-only fine-tuning. Beyond benchmark evaluation, validation on egocentric data from two mobility-assistive robots demonstrates improved perception reliability in real-world navigation scenarios. Together, NavAble establishes a scalable foundation for BLV navigation perception, directly addressing the data scarcity and embodiment gaps. The dataset and pipeline are publicly available at https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV.

### 论文解读

#### 摘要翻译

把导航环境转化为适合视障者理解和行动的生成式辅助信息。 本文发布面向盲人导航的无障碍目标数据集及合成生成流程，并验证合成数据对目标分割的帮助。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：通用导航模型通常面向视觉显示或机器人路径规划，难以直接回答视障行人的局部通行问题。场景中的可行路径、障碍和语义地标需要结合起来表达。NavAble 的假设是生成式模型可以将环境观测转换成任务相关的路径与辅助描述，再由使用者或系统验证。

#### 方法设计详解

系统先获取环境图像及导航任务上下文，再通过生成模块形成可通行路径或辅助导航输出。论文给出从场景输入、路径生成到导航提示的整体流程，并在手杖与四足机器人等配置下检验。关键在于把视觉环境表征与可执行导航建议相连；提示和路径仍需服从障碍检测、定位及执行设备约束。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

与只报告场景描述的视觉问答相比，NavAble面向导航行动；与传统固定规则辅助相比，它试图生成适应环境的路径信息。贡献是将生成式场景理解置于可访问性任务中。它并没有消除定位误差、动态障碍或用户偏好带来的安全问题。

#### 实验分析（精简版）

实验比较多种导航配置，并在手杖和四足机器人条件下报告任务结果。在NAVABLE-Real留出集上，Mask2Former的mIoU从仅用真实数据的68.84%升至真实加合成数据的72.67%；这是分割指标。两种移动辅助机器人上的自我中心图像验证仅说明感知迁移，不是盲人用户导航试验。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现需记录场景、路线、设备、提示模板和成功判定，并分开报告生成正确性与实际安全通行。迁移到移动机器人可借用语义路径生成，但要接入独立避障和定位安全层；迁移到真实辅助设备需优先验证延迟和故障退化。 NAVABLE包含约36K整理图像、5.5K新采集实景图像及452K合成帧；合成部分由37种环境、11类无障碍目标构成。 训练集比较真实数据与合成数据组合，并在NAVABLE-Real留出集评估；系统发布的数据资源可用于复现。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：合成无障碍导航数据

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19330v1)
- [arXiv](https://arxiv.org/abs/2610.19330v1)

---

<a id='2610.19579v1'></a>
## [Back on Track: Deployment-time Policy Improvement via World Model Planning](https://arxiv.org/abs/2610.19579v1)

**Authors:** Colton Crosby, Hao Wang, Mac Schwager, Somil Bansal

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

State-of-the-art visuomotor policies are increasingly capable of executing complex tasks, yet can fail during deployment in ways they cannot predict or recover from on their own. Improving their performance and robustness typically requires collecting more data, retraining, or modifying the policy architecture, all of which are time- and resource-intensive in practice. We instead propose an offline-trained monitoring-and-intervention framework that improves a policy during deployment using only black-box query access. From a limited set of labeled on-policy rollouts, we learn a policy-conditioned liveness value that measures the nominal policy's progress toward task completion; we additionally train a compact latent world model using these rollouts together with play data. At deployment, the liveness value continuously monitors execution. When it detects a loss of progress, a short-horizon planner searches the world model for corrective actions using the same value as its planning objective, after which control returns to the nominal policy. We evaluate our method on the LIBERO-Goal task suite and on a Franka Panda manipulator, using the vision-language-action model $π_{0.5}$ and a diffusion policy as nominal policies, and demonstrate significant improvements in task success rates, while supporting real-time control. Additional results and videos are available on the project website (https://policy-improvement-wm.pages.dev/).

### 论文解读

#### 摘要翻译

在长时序任务失败后恢复，并把恢复经验纳入持续学习。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：长任务中早期偏差会累积，机器人即使掌握单个技能也可能偏离原任务。单纯重试不能保证回到正确状态，而每个新任务重新训练成本高。论文围绕恢复与持续适应提出机制：先判断当前进程如何偏离，再返回可继续完成任务的状态。

#### 方法设计详解

系统将长任务组织为由视觉状态和子任务构成的执行过程。执行中跟踪进度与成功信号；当状态不再符合预期，通过恢复策略重新定位到可继续的任务阶段，再调用技能继续执行。在线获得的成功经验用于更新技能或任务知识，使后续任务受益。价值函数式的进度/恢复决策降低了从失败状态盲目继续的风险。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

与固定顺序脚本相比，方法允许在失败后回到轨道；与仅训练单任务策略相比，强调跨任务持续学习。其核心贡献是将恢复决策纳入长期执行闭环。成功恢复依赖对任务进度的识别，模型判断错时也可能继续错误动作。

#### 实验分析（精简版）

在LIBERO十任务实验中，成功率从基线72.6%提升至81.5%；Franka真实机器人任务中从40%提升至60%。这些结果支持恢复与学习机制的效果，但不意味着长时任务的所有失败均可恢复。论文还报告世界模型相关失效情形，例如酒瓶对象预测错误，说明环境模型仍可能限制恢复。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现要明确任务阶段、成功判据、失败触发条件和恢复数据如何加入训练，并报告每种任务的失败类型。迁移到真实操作应加入人工接管和安全停止；可迁移的主要是任务进度跟踪与恢复闭环，而非未经验证的通用世界模型。 实验证据覆盖十个LIBERO任务与Franka真实任务；复现时应固定任务阶段、窗口和每个任务的成功判据。 训练集包括每任务200条策略rollout；作者还用RTX 4090训练liveness value和world model，并补充53分钟遥操作数据。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：让长任务失败后回轨

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19579v1)
- [arXiv](https://arxiv.org/abs/2610.19579v1)

---

<a id='2610.19527v1'></a>
## [Dual-arm Cooperation for Tabletop Wiping with Temporal Object Removal](https://arxiv.org/abs/2610.19527v1)

**Authors:** Pengqi Zhang, Longchang Yu, Daohan Li, Liang Yan, Hangxin Liu

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

We devise a dual-arm cooperative planning framework for wiping table surface in the presence of scattered objects. Unlike conventional surface-cleaning strategies that only cover free areas and detour around obstacles, our framework explicitly assigns one arm to rearrange objects while the other performs coverage wiping, thereby enabling access to previously obstructed regions. Specifically, we first construct a set of wiping and object-rearrangement action primitives. Based on these primitives, we introduce Geo-MILP, a geometry-aware task planner formulated as Mixed-Integer Linear Programming (MILP). By integrating Logic-Based Benders Decomposition (LBBD), Geo-MILP iteratively optimizes action sequencingand timing to resolve geometric conflicts, promoting efficient cooperation between wiping and object rearrangement. We then employ K-Order Markov Optimization (KOMO) to generate smooth, collision-free motions to ensure feasible execution. Simulation and physical experiments demonstrate that the proposed framework not only increases reachable coverage by enabling access to obstructed areas, but also significantly improves time efficiency compared to Task and Motion Planning (TAMP) baselines, reducing planning time by up to 9 times and execution makespan by approximately 40%.

### 论文解读

#### 摘要翻译

用几何约束与混合整数调度协调双臂擦拭任务。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：桌面擦拭同时涉及覆盖区域、工具姿态、双臂可达性和动作先后关系。只优化单臂轨迹或按固定顺序操作会浪费时间，也可能产生冲突。论文假设将覆盖几何与调度变量结合，可以快速求出实用的双臂任务安排。

#### 方法设计详解

首先把桌面脏污/目标区离散为可处理区域，并根据擦拭工具几何建立覆盖关系；然后生成两臂可行操作与区域间依赖，形成混合整数线性规划。优化器联合决定分配、动作顺序和并行时间安排，再将任务转换成机器人可执行轨迹。论文在仿真和真实机械臂系统中验证规划与执行。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

相比基线，相较顺序式PDDLStream规划，几何MILP把覆盖及调度结构显式化，减少搜索开销并允许双臂并行。它的优势是任务结构清楚且求解快，前提是目标离散、运动学与碰撞约束建模足够准确；求解所得调度不等于连续空间全局最优轨迹。

#### 实验分析（精简版）

在三物体场景中，规划约快9倍，makespan约低40%，比较对象为顺序式PDDLStream。作者还报告10个仿真实例及FRANKA和Kuavo平台演示。结果证明建模任务上的调度收益，无法直接外推到任意柔性物体、摩擦或动态障碍场景。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现应先检查区域离散分辨率、覆盖判据、可达/碰撞约束、MILP求解器时间和低层轨迹规划。迁移到清洁或仓储双臂任务时可复用任务分配与并行调度结构，但需要为物理接触加入力控与在线重规划。 规划输入为离散覆盖网格和双臂动作基元；先报告MILP求解时间，再单独计入连续轨迹生成与机器人执行时间。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：几何建模双臂擦拭

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19527v1)
- [arXiv](https://arxiv.org/abs/2610.19527v1)

---

<a id='2610.19604v1'></a>
## [ForceBench: Benchmarking Force Integration for Robotic Manipulation](https://arxiv.org/abs/2610.19604v1)

**Authors:** Xuan Han, Zinuo Jin, Tianlin Li, Yunfan Lou, Xin Liu, Yuchen Xue, Yuanpeng He, Boxian Yao, Shuhan Zhan, Ruihai Wu, Zhibo Pang

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Force feedback provides vision-language-action (VLA) policies with physical observations that can guide action selection and execution. However, differences in tasks, demonstrations, and control settings across existing systems hinder reliable comparisons of force integration designs. We introduce ForceBench, a simulation benchmark for evaluating six-axis wrist-force integration under a unified protocol. ForceBench combines eight tasks spanning physical-property inference, resistance-dependent interaction, and contact-rich assembly with 1,600 synchronized demonstrations and task-specific success criteria. We compare one no-force baseline and six force integration designs on a shared $π_{0.5}$ backbone. Prompt Fusion achieves the highest observed mean simulation success of 35.31%, compared with 23.25% for the baseline, and improves success rates across all three task categories. Predictive and temporal gains in simulation concentrate on physical-property inference, while both temporal designs underperform the baseline on assembly. Real-world evaluation of eight systems across five tasks supports the main qualitative trends: Prompt Fusion and Dual-Path Fusion achieve 63.00% mean success, compared with 40.00% for the baseline, but only Prompt Fusion improves every task. These findings support ForceBench's practical value for identifying the benefits and limitations of force integration across manipulation demands. The project page is available at https://forcebench-platform.github.io/.

### 论文解读

#### 摘要翻译

用统一基准比较机器人策略如何使用力觉完成操作。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：大量操作基准关注视觉或动作成功，难以区分策略是否真正利用接触力。力觉策略的任务、传感器与评价方式不统一，也使方法间结果难比较。ForceBench通过系统任务与传感器设置，检验力信息怎样影响策略表现。

#### 方法设计详解

基准构建具有接触需求的模拟操作任务，并配置可用的视觉、力觉输入和策略接口。被测策略以观测为输入预测动作，设置不使用力的基线、力提示融合及双路径等方法进行比较；同时通过真实机器人任务检验仿真结论。评测关注成功率和跨任务表现，强调策略融合力信号的方式，而非只看模型参数。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

与只统计视觉操作成功的基准相比，该工作将力觉作为核心变量并提供策略对照；不同融合方法可揭示简单提示、专用路径与无力输入的差异。基准的意义在于可比较性，不代表其任务覆盖了所有接触操作。

#### 实验分析（精简版）

模拟实验中Prompt Fusion平均成功率为35.31%，无力输入对照为23.25%；真实实验中Prompt与Dual-path均为63%，基线为40%，每个任务各20次试验。真实结果样本规模有限，适合视作基准中的对照证据，而非普遍性能保证。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现时严格遵守任务划分、传感器处理、成功阈值和试验次数；将力信号时序与视频同时保存，排查策略是否忽略触觉。迁移到装配或柔性操作时可沿用评价框架，但要加入峰值力和损伤风险指标。 策略训练使用每任务200条成功专家演示（共1,600条），物理仿真为60 Hz，策略观测/动作记录为15 Hz；对比需保持采样节奏一致。 训练集包含每任务200条成功专家演示，共1,600条；仿真60 Hz，策略观测与动作记录15 Hz。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：系统评测力觉操作

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19604v1)
- [arXiv](https://arxiv.org/abs/2610.19604v1)

---

<a id='2610.19615v1'></a>
## [PolyMorph: Learning Locomotion Policy for Cross-Embodiment Adaptation](https://arxiv.org/abs/2610.19615v1)

**Authors:** Hyunsik Oh, Seunghun Jeon, Donghoon Youm, Jungwoo Hur, Hyeongjun Kim, Jeong Hyun Lee, Moonkyu Jung, Jemin Hwangbo

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Reinforcement learning has enabled robust locomotion for legged robots, but learned policies are typically tied to a specific robot morphology and require additional engineering when the number of legs, actuators, or mechanical properties change. We present PolyMorph, a morphology-conditioned framework that learns a single locomotion policy for heterogeneous legged robots. PolyMorph converts each robot's URDF into joint-link modules and combines a recurrent encoder, graph-based Transformer, and per-joint decoder to generate actions across variable leg counts and joint configurations. To improve training stability, PolyMorph uses morphology-normalized rewards, adaptive robot sampling, and URDF-derived PD-gain initialization. We evaluate PolyMorph in simulation across bipedal, tripedal, and quadrupedal robots, including unseen morphology variants and platforms, and deploy the policy on real quadruped platforms operated in multiple configurations. The results show that a single PolyMorph policy generalizes across diverse robot morphologies without per-robot policy retraining.

### 论文解读

#### 摘要翻译

让一个控制器适应多种腿式机器人形态与尺度。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：为每种形态单独训练控制器耗费数据和工程资源，形态变化也会造成动作空间与动力学差异。作者尝试通过多形态训练和统一动作表示，让控制器在不同结构之间共享经验，并测试未见过的形态。

#### 方法设计详解

训练环境随机化机器人质量、尺度和腿部结构，构造形态条件化的控制输入。策略根据机器人状态与形态信息输出关节动作，经统一控制接口驱动不同构型。模拟训练覆盖多个体型与形态；训练完成后把同一控制器部署到四足、三足、双足以及未见三足设置，检验迁移和命令跟踪。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

与每种结构单独训练策略相比，PolyMorph使用共享控制器；其差异在于把形态变化纳入训练分布，并非把机器人视为完全相同。结果有助于研究形态泛化，但未见配置表现仍低于已知形态，不能保证任意拓扑直接适配。

#### 实验分析（精简版）

真实平台中四足和三足各20/20成功（100%），双足9/10，未见三足7/10。模拟训练覆盖约11至135 kg；结果支持一定形态迁移，但样本数小，极端尺度和未见动力学仍需专门评估。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现需核对仿真参数随机化范围、形态编码、观测/动作归一化及每种形态训练比例。迁移到新平台前先做低速台架测试并逐步提高命令幅度。可迁移思想是共享策略加条件信息，安全控制仍需按硬件单独校准。 训练在仿真中对形态、质量与体型随机化（约11至135 kg）；关节模块上限设为15，未见形态应单列测试。 模拟训练集覆盖约11至135 kg质量并随机化形态，关节模块上限为15；新硬件应先校准动作接口。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：共享多形态步态控制

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19615v1)
- [arXiv](https://arxiv.org/abs/2610.19615v1)

---

<a id='2610.19594v1'></a>
## [RoboSpan: Learning Modular Coordination Policies for High-DoF Whole-Body Mobile Manipulation with Teleoperation](https://arxiv.org/abs/2610.19594v1)

**Authors:** Keqiang Ren, Xuecheng Ye, Hanyu Liu, Zhihua Wang, Tao Jian, Yunsheng Ma, Jiayi Wen, Xinran Li, Zhigong Song

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Whole-body mobile manipulation suffers from two fundamental challenges: inherent motion heterogeneity across locomotion, arm, and hand modules, and progressive error accumulation over long-horizon multi-stage tasks. Existing teleoperation platforms also lack integrated dexterous tactile sensing to support coordinated whole-body decision-making. This paper presents RoboSpan, a high-DoF whole-body teleoperation and demonstration platform, and the Hierarchical Modular Framework (HMF) for learning modular coordination policies. HMF functionally decouples the 29-DoF action space into fine hand manipulation, arm pose adjustment, and mobile-base spatial configuration branches, and introduces the Coordinated Modular Embodied Transformer (COMET) with inter-modular cross-attention to enable dynamic information exchange and action coordination. Vision-language model driven hierarchical task decomposition structurally suppresses temporal error propagation, while a full-palm 2D tactile representation encodes distributed contact signals for cross-module tactile empowerment. Experiments on a physical 29-DoF robot demonstrate that the proposed method achieves over 20\% higher task success rate than mainstream baselines across a range of long-horizon whole-body tasks, markedly improving stability and robustness in contact-rich manipulation.

### 论文解读

#### 摘要翻译

用层级融合把视觉与触觉表征连接起来，支持跨模态操作。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：视觉擅长远距离识别和空间布局，触觉则在接触后提供局部形状、滑移与受力线索。若两者表征各自独立，跨模态任务容易产生信息断层。RoboSpan假设先建立可共享的模块表示，再在不同层级融合，能提升多模态操作的迁移表现。

#### 方法设计详解

模型分别编码视觉与触觉观测，并构造层级模块融合机制，在策略中交换和组合模态信息。高层表示用于任务及物体关系，细粒度的触觉/视觉特征补足接触状态，最后由策略生成操作动作。训练和评估覆盖视觉、触觉以及跨模块任务，比较融合结构与单模态或弱融合基线。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。作者给出的训练和评估场景为复现边界，超出该范围需要额外验证。

#### 方法对比分析

相比基线，相较只用视觉或触觉，方法利用各自互补信息；相较在输入端简单拼接，层级融合使模块间表示在不同抽象层级交换。实验收益支持融合设计在论文任务上的价值，但不等于所有任务都需要复杂多模态架构。

#### 实验分析（精简版）

结果中RoboSpan相对对照在主要操作指标上达到54.7对41.3；在触觉跨模块任务上为61.3。表格结果显示融合对部分任务有明显帮助，具体数值依赖任务集和训练配置，不能直接解释为真实世界成功率。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现应维持模态同步、触觉预处理和对照公平性，并报告缺失模态下的性能。迁移到装配、抓取可先判断任务中接触反馈是否不可替代，再逐步加入触觉编码器；不必默认所有机器人均需完整融合配置。 策略按手、臂和移动底盘三个模块编码不同传感器，并以跨策略上下文融合交换信息；部署时要保持触觉、视觉与关节状态同步。 训练集使用论文的多模块操作任务；推理时各模块各自编码，运行跨策略上下文融合，部署需同步触觉、视觉和关节状态。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：层级融合视触觉

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19594v1)
- [arXiv](https://arxiv.org/abs/2610.19594v1)

---

<a id='2610.19798v1'></a>
## [Design and Characterization of a Unified Continuum Robot Actuation Platform](https://arxiv.org/abs/2610.19798v1)

**Authors:** Yunti Xu, Aedan Mangan, Miheer Potdar, Alex Lu, Tania K. Morimoto

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Continuum robots enable dexterous access to confined spaces in applications such as minimally invasive surgery and industrial inspection. However, their actuation systems are often custom-designed for specific robot architectures and tasks, limiting reuse across publications, institutions, and applications. This paper presents a unified actuation platform that decouples actuation hardware and software from continuum robot design. The platform provides modular actuation across continuum robot classes, scales to a large number of degrees of freedom (DOFs), and integrates with conventional serial manipulators for hybrid architectures. We validate the platform on an open-sourced surgical robotic system and experimentally characterize the actuation modules, achieving sub-millimeter translational accuracy and less than $5^\circ$ rotational error over ranges of $30$~mm and $180^\circ$, respectively. We further demonstrate the platform in two surgical use cases: teleoperation of a continuum camera in a constrained environment and bimanual manipulation using $2.53$~mm concentric-tube-robot-based forceps alongside a conventional wristed instrument. All hardware designs and ROS-based software are open-sourced to facilitate reuse and further research in continuum and surgical robotics.

### 论文解读

#### 摘要翻译

以统一连续体驱动平台提升微创手术器械的可控性。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：微创器械需要在狭窄通道内完成平移、弯曲和旋转，传统驱动耦合与误差会限制精细操纵。论文将驱动结构和运动控制统一设计，目标是在较小器械空间内获得可预测的多自由度运动。

#### 方法设计详解

平台通过统一驱动机构产生连续体器械的平移与旋转/弯曲动作，结合传感或标定模型估计器械运动并闭环控制。系统将平台结构、运动学关系和控制接口纳入一体化设计，再通过台架精度实验和远程操作场景验证。关键是把多自由度输入映射为器械末端运动，同时评估累积定位误差和转角偏差。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。论文给出的评估场景构成复现边界，超出该范围需要额外验证。

#### 方法对比分析

相比彼此独立的驱动机构，统一驱动意在减少空间占用并协调多自由度；相比仅展示机构原型，论文报告运动精度和操作任务。它主要是系统与执行器贡献，性能不宜和纯算法指标直接对比。

#### 实验分析（精简版）

30 mm平移范围累计误差达到0.29±0.03 mm；180°旋转累计角误差约4.89±0.07°。论文还展示真实远程操作应用，包括力钳器械约2.53 mm的操作结果。台架精度不能直接代表临床效果，临床安全性、灭菌和组织交互需另行验证。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现应关注驱动几何、运动学标定、误差测量方式、负载与控制频率。迁移到柔性内窥、微操作平台可借鉴紧凑驱动与统一控制接口，但须针对不同器械直径、弯曲刚度和安全力限重新设计。 这是执行器系统而非策略学习方法；复现实验应固定器械装配、负载和标定，并分别测量平移与转角误差。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：统一驱动连续体器械

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19798v1)
- [arXiv](https://arxiv.org/abs/2610.19798v1)

---

<a id='2610.19753v1'></a>
## [Coordi-pi: Coordination-Aware Dual-Hand Action Representation for Bimanual Dexterous Manipulation](https://arxiv.org/abs/2610.19753v1)

**Authors:** Yiming Meng, Yanming Shao, Weijie Shen, Kunyang Lin, Jiayu Sun, Ping Luo, Yao Mu

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Vision-language-action (VLA) models have shown strong potential for general-purpose robotic manipulation, but extending them to bimanual dexterous manipulation introduces a high-dimensional and structured action space. Existing bimanual VLA policies commonly concatenate right- and left-side arm--hand actions at each control timestep, leaving cross-hand interactions implicit within a single action vector. We introduce Coordi-$π$, a structured action representation for bimanual dexterous manipulation. Coordi-$π$ represents the right- and left-side arm--hand actions as individual action tokens and organizes them into an interleaved sequence, explicitly exposing the two sides within the action sequence. To explicitly encode hand identity and temporal synchronization in this interleaved representation, a learnable hand-identity embedding distinguishes the two sides, while sequence--control-timestep factorized rotary positional encoding (RoPE) separates sequence position from control timestep, allowing actions at the same control timestep to share a temporal coordinate while retaining distinct sequence positions. Across all five bimanual dexterous manipulation tasks in DexJoCo, Coordi-$π$ achieves state-of-the-art performance under both \textit{rand-obj} and \textit{rand-full}, improving the average success rate over $π_{0.5}$ from 31.9\% to 51.3\% and from 18.3\% to 32.3\%, respectively. Coordi-$π$ also consistently outperforms $π_{0.5}$ across four real-world bimanual dexterous manipulation tasks.

### 论文解读

#### 摘要翻译

用协调视觉运动策略处理多阶段、跨部件操作。 本文针对机器人任务中的具体难点提出系统化方案，并在论文所设定的模拟或真实平台上进行验证。研究重点是方法如何从输入观测到达成任务输出，以及实验能支持哪些有限结论。

#### 方法动机分析

核心痛点与驱动力是：多阶段装配要求机器人不仅完成单步抓取，还要维持物体、工具和子任务间的协调。单阶段策略容易在阶段转换或对象位置变化时失败。Coordi-π把多阶段操作所需的视觉与动作协调作为策略学习目标，并通过随机化考察泛化。

#### 方法设计详解

方法将多阶段任务观测编码为视觉状态，由策略联合预测操作动作，并让阶段进度与对象关系影响后续动作。训练设置覆盖物体随机化和更完整场景随机化；策略通过示范或任务数据学习从当前视觉输入到连续操作输出的映射。评估分别对比不同随机化条件及真实机器人执行，以检验组合动作能力。 其输入、状态编码、策略或优化模块与输出动作之间形成闭环；实现时需要确保数据标注/传感器、动作接口及评估划分符合论文设定。论文给出的评估场景构成复现边界，超出该范围需要额外验证。

#### 方法对比分析

与逐阶段独立策略相比，协调策略意在跨阶段共享状态和动作上下文；与仅改变物体外观的训练相比，rand-full覆盖更广的场景变化。贡献是面向多阶段操作的协调学习，但长程误差仍会累积。

#### 实验分析（精简版）

五个DexJoCo随机物体任务成功率为51.3%，基线为31.9%；rand-full条件下为32.3%对18.3%。真实测试为75%对61.3%，但装配任务成功率仍只有7.3%，显示精细接触和长程阶段转换仍困难。 这些指标只适用于论文的任务定义与试验条件，不能替代不同硬件、操作者或真实部署环境中的独立测试。

#### 实用指南

复现需分开实现物体随机化与全场景随机化，固定任务成功标准并报告逐任务表现。迁移到装配或工具使用可借鉴统一策略，但应加入状态估计、接触反馈和安全恢复；总体平均分不应掩盖单个精细任务短板。 策略输出H步双臂动作块；训练评估应分别报告rand-obj与rand-full条件，避免只用总体均值掩盖装配失败。 训练集设置分别采用rand-obj和rand-full随机化；评估时分别报告每种设置与真实机器人表现。 阅读或复现时应同时记录成功判据、失败案例和计算/执行开销；论文中的结果不应脱离其比较基线与条件进行横向推断。

#### 总结

核心思想：协调双臂多阶段操作

1. 获取任务观测并识别当前状态。
2. 根据论文设计形成动作、计划或预测。
3. 执行并用任务反馈检验结果。
4. 在适用范围内比较基线，明确迁移限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.19753v1)
- [arXiv](https://arxiv.org/abs/2610.19753v1)

---

