time: 20261111

# Arxiv Computer Vision Papers - 2026-11-11

## Executive Summary

## 今日执行摘要

本期十篇论文横跨机器人、视觉几何、世界模型、三维渲染和目标检测。共同趋势是把评估从单一准确率扩展到系统行为与证据边界：HOLIDAYFACTORY 探索持续工业作业的仿真和任务级记录；WARC 研究世界模型在测试时通过交互学习新机制；WanderBench 将漫游视频质量拆成多维指标。三者分别把长期运行、在线适应和生成内容诊断作为研究对象，也都显示基准设计需要与真实任务范围相匹配。

在方法上，DeltaForge 以共享低秩子空间组合多个驾驶 VLA 专家，PAIR 将相机射线几何和机器人状态对齐用于跨视角操作，CSG-Nav 融合语义偏好与几何可达性。它们体现了从孤立模型转向多来源信息组合的方向。另有两篇工程取向工作：GRAM/MORE/ART 将多视图匹配直接纳入旋转与平移估计；3DGS 调度方法通过负载均衡缓解 GPU 渲染长尾。微小行人检测论文则提醒读者，置信度校准改善与检测精度提升并非同一目标。

建议优先完整阅读 WARC、PAIR 与 DeltaForge，以理解测试时机制学习、跨视角操作表征和多域专家融合；关注长期评测设计的读者可接着看 HOLIDAYFACTORY 与 WanderBench。阅读结果时应核对数据集、基线、指标定义和运行条件：例如 3DGS 工作优化的是 alpha blending 阶段，机器人仿真框架尚未提供持续策略排名，检测校准收益也伴随部分场景下 mAP 轻微下降。整体来看，本组论文的价值在于提出可检验的新系统设计和评估维度，许多方向仍需更广任务覆盖与真实部署验证。

---

## Table of Contents

1. [HolidayFactory 1.0: Simulating, Generating, and Evaluating Industrial Robot Tasks](#2611.11789v1)
2. [PAIR: Proprioception-Aligned Image-Ray Representations for Viewpoint-Robust Manipulation](#2611.11798v1)
3. [DeltaForge: A Post-hoc Mixture-of-Experts VLA for Multi-Domain Autonomous Driving](#2611.11856v1)
4. [How Many Exposures Does a World Model Need? Learning Unseen Game Mechanics at Test Time](#2611.12324v1)
5. [Data-Driven Precision Control of Soft Helicoid Robots](#2611.11709v1)
6. [CSG-Nav: Confidence-Aware Semantic-Geometric Fusion for Robust Exploration and Accurate Zero-Shot Object-Goal Navigation](#2611.11978v1)
7. [WanderBench: Can World Models Explore Dynamic Worlds from Video?](#2611.12557v1)
8. [Correctness Metric Matters: NWD-Based Calibration Loss for Tiny Person Detection in UAV Infrared Imagery](#2611.11735v1)
9. [Towards Rotation-Only Imaging Geometry: Multi-View Pose Estimation](#2611.11770v1)
10. [Mitigating SM-Level Load Imbalance in 3D Gaussian Splatting on GPUs](#2611.12319v1)

---

## Papers

<a id='2611.11789v1'></a>
## [HolidayFactory 1.0: Simulating, Generating, and Evaluating Industrial Robot Tasks](https://arxiv.org/abs/2611.11789v1)

**Authors:** Holiday Robotics,  :, Jeongmin Lee, Taehyeon Kim, Dohyun Jang, Minji Lee, Seonghun Oh, Dongwoo Park, Geonhyup Lee, Jaehyung Kim, Jaemin Yoon, Jinseop Bok, Jungwoo Kim, Minseok Yang, Philsik Chang, Soomin Lee

**Published:** 2026-11-09

**Categories:** cs.RO, cs.AI

**Abstract:**

Robots are typically evaluated in episodes that begin from a reset, while industrial work consists of repeated jobs whose outcomes affect subsequent operations. Evaluating such work at scale requires simulation of task-critical physical interactions, construction of industrial environments, and measurements of performance as scene state persists. We present HOLIDAYFACTORY 1.0, a framework for simulating, generating, and evaluating industrial robot work. Given a human-specified workplace and task, the framework defines a world pairing a scenario of scene variants and process logic with an evaluation defining a job, metrics, and a rubric. HOLIDAYFACTORY 1.0 comprises three components. HOLIDAY-SIM extends Newton's SDF-based hydroelastic contact with additional constitutive laws and a semi-implicit primal-dual solver, couples stateful actuator dynamics to articulated motion, and derives tactile and joint observations from simulated states. Agentic world construction generates simulation-ready assets and scene variants using geometric and interaction checks, and proposes task-conditioned evaluations for human confirmation and executable checks. 24/7-BENCH introduces seven industrial tasks in which jobs are evaluated without resetting the scene, measuring completed, failed, and unfinished jobs, throughput, latency, and motor energy. Controlled studies characterize the proposed contact, actuator, and tactile models and examine the effectiveness of verification-guided world construction. Experiments on scene and evaluation generation further reveal the limitations of model critics and the difficulty of ensuring that generated evaluations faithfully capture task requirements. We examine job-level evaluations with scripted trajectories (e.g., code-as-policy) but have not yet compared policies over persistent runs.

### 论文解读

#### 摘要翻译
机器人常在每次重置后执行单个任务回合，但工业生产由连续重复的工作组成，前序结果会影响后续操作。HOLIDAYFACTORY 1.0 尝试把物理仿真、工业场景生成和持续作业评测连成框架：它扩展基于稀疏 SDF 的水弹性接触仿真，加入有内部状态的驱动器与触觉/关节观测；通过带验证的生成过程构造资产、场景和评测；并以七个工业任务组成 24/7-BENCH，统计完成、失败、未完成工作、吞吐、延迟和电机能耗。作者也指出，生成评测是否忠实于任务要求、仿真是否能代表真实硬件仍是开放问题，且尚未在持续运行中比较机器人策略。

#### 方法动机分析
传统 episode 评测会重置场景，难以表示输送带持续来料、货架逐渐装满、失败后留下障碍等工业状态变化。单次成功率也不足以说明长时间运行中的吞吐、恢复和能耗。框架的核心假设是：把工作定义为可追踪的重复 job，并在不重置的运行中记录场景状态，可以更贴近工业作业评估；同时，生成的资产和场景必须经可执行约束检查，生成的评测也需单独验证。

#### 方法设计详解
系统将任务 world 表示为场景与评测的配对：场景含布局变体及传送带、事件、队列等流程逻辑；评测定义 job 生命周期、指标和 rubric。HOLIDAY-SIM 从稀疏 SDF 提取压力接触斑，以水弹性力、压缩阻尼和摩擦计算接触，并用半隐式 CDMM 求解接触与关节运动。电机电流、温度、传动和内部摩擦状态也耦合进动力学；接触斑驱动磁触觉读数。ERIS 将任务要求转成尺寸、清隙、功能、外观或关节约束，再路由至网格、参数化刚体或可动资产生成；碰撞体、关节及任务交互测试反馈用于修订。场景生成器构造任务计划与变体，评测提案经人确认后编译，并通过静态、重置和空闲检查。FRIDAY 按 60 Hz 控制、每帧四个物理子步运行。

#### 方法对比分析
与常见短 episode 仿真基准相比，24/7-BENCH 将评分单位从一次回合扩展为共享持续状态的工作过程，并报告 job 延迟、未完成工作和电机能耗。与点接触相比，水弹性模型保留分布式接触压力及扭转力矩；与一次性生成资产/场景相比，ERIS 和场景生成都把可测的验证失败反馈到修订环节。评测提案由人确认，但确认后的编译和基础检查可确定性执行。

#### 实验分析（精简版）
72 个资产的首轮验证中，34 个（47%）未通过至少一个检查；12 个首轮失败案例中，ERIS 依靠测量反馈修复 11 个，仅一轮修订，其余一个再一轮完成。独立重生成最终成功率为 84%，粗略通过/失败反馈为 81%。不过，场景生成接受率会随模型批评器显著变化；单 tote 评测实验中，澄清问题改善任务意图一致性，而检查反馈虽提高直接生成接受率，却没有纠正错误判定。32 条脚本轨迹测试通过，但策略尚未在七个持续任务上进行比较。接触、驱动和触觉模型均未与实机测量验证。

#### 实用指南
复现时需固定场景变体、流程逻辑、物理步长、job 事件及超时规则，并按 run 重复统计。论文报告 60 Hz 控制、每帧四个 1/240 秒物理子步；SDF、驱动器和触觉参数需按硬件校准。文中未明确提供可复现代码与模型清单。迁移到现有基准可先增加持续状态和失败类型记录，再用实机轨迹、重复策略运行和跨场景审计验证。

#### 总结
核心思想：把工业评测变成持续状态下的重复工作。
1. 人定义工作场景和任务要求。
2. 验证器辅助生成资产、布局和流程。
3. 仿真器在连续状态中运行机器人。
4. 评测器逐 job 记录结果、时间和能耗。
5. 用重复运行与实机证据检验评价是否可信。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11789v1)
- [arXiv](https://arxiv.org/abs/2611.11789v1)

---

<a id='2611.11798v1'></a>
## [PAIR: Proprioception-Aligned Image-Ray Representations for Viewpoint-Robust Manipulation](https://arxiv.org/abs/2611.11798v1)

**Authors:** Sung-Wook Lee, Xuhui Kang, Dylan Losey, Yen-Ling Kuo

**Published:** 2026-11-09

**Categories:** cs.RO

**Abstract:**

While imitation learning has made substantial progress toward generalizable robot manipulation through scaling, viewpoint robustness remains a major bottleneck. Visuomotor policies rely heavily on visual inputs, yet images provide an inherently ambiguous signal under viewpoint change: similar scenes may appear markedly different across viewpoints, while distinct states may share certain visual overlaps. A policy must therefore infer not only the task-relevant state, but also how it is visually projected under a particular camera configuration. This entanglement of viewpoint-dependent appearance and task-relevant state can make learning inefficient and limit generalization to unseen viewpoints. Recent work has begun to address this challenge through camera-ray conditioning and representation learning using geometric cues to induce invariance. However, providing camera information as additional context does not by itself ground visual representations in how the task scene is visually projected, while geometric alignment without camera context can introduce spurious correspondences by ignoring viewpoint-dependent projection. We propose PAIR, a contrastive learning framework that aligns joint image-ray representations using robot proprioception. PAIR encourages observations with similar robot states to remain close in representation space across viewpoints, providing a structured grounding signal for learning viewpoint-robust features. PAIR is a highly adaptable framework: it can be used to train Diffusion Policy (DP) or post-train vision-language-action (VLA) models. In simulated and real-world manipulation benchmarks with camera variation, PAIR consistently outperforms the behavior cloning baseline without ray conditioning by 12.7% and 26.7% points, respectively, demonstrating stronger generalization to unseen viewpoints. Project website: PAIRrobot.github.io.

### 论文解读
#### 摘要翻译
PAIR 研究机器人模仿学习在未见相机视角下的鲁棒性。方法把相机射线几何编码进图像特征，并利用机器人本体状态相似性进行对比学习，使不同视点下相似物理状态的表示靠近。作者在仿真与真实操作任务上，将该目标用于扩散策略和视觉语言动作模型。

#### 方法动机分析
图像是物理场景经过相机投影后的结果。同一末端状态可能因相机移动而外观不同；反过来，视觉相似也不保证本体状态相同。仅用本体状态构造正样本会忽略造成外观差异的相机几何。PAIR 的核心假设是把相机射线作为显式几何条件，再用本体状态监督跨视角表示，能缓解这类歧义。

#### 方法设计详解
对于每个视觉 patch，方法计算两种 Plücker 射线：相对末端执行器坐标系的射线，以及相对机器人基座坐标系的射线。前者随机械臂运动变化，后者提供固定参照。轻量卷积网络编码射线后，与 SigLIP 图像 token 特征拼接，再由恒等初始化线性层投回视觉维度并加入原位置编码。对比学习根据末端位置、姿态四元数和夹爪宽度计算两条示范的状态距离，并把距离的百分位转换为软正样本权重；几何状态越接近，图像射线表征被拉近的程度越大。总训练目标将该损失加到行为克隆损失上。它可训练 Diffusion Policy，也可后训练 SmolVLA；策略推理仍只看单个视角，多视图信号用于训练对齐。实机相机每轮移动，并通过 AprilTag 估计位姿。

#### 方法对比分析
相机射线条件方法提供视点信息，却不一定让表征与任务状态对齐；只按本体状态对齐又可能忽略投影差异。与只加入相机条件或只按本体状态对比的基线相比，PAIR 联合使用两者，创新点在于用双参考系射线消解状态对齐歧义。消融结果显示，末端系和基座系射线有互补作用；移除任一种都会令若干任务成功率下降。

#### 实验分析（精简版）
LIBERO 四类任务的平均成功率中，PAIR 加射线为 88.9%，射线条件行为克隆为 83.3%。实机三项任务报告平均子任务完成率 76.7%、最终成功率 57.8%；每个任务设置使用 30 次 rollout。训练策略按三种随机种子评估，并报告五个 checkpoint 的最大值，因此结果不代表单一固定训练步数的表现。

#### 实用指南
Diffusion Policy 训练 200k 步，SmolVLA 训练 160k 步，学习率均为 1×10⁻⁴；对比损失批量大小为 192。实机设置需提供相机外参和机器人本体状态，且需在采集和部署时保持可靠标定。迁移前应测量射线标定误差，并检查训练视角是否覆盖目标相机范围。

#### 总结
用射线和本体状态对齐视觉
1. 为图像 patch 计算两套相机射线。
2. 将射线特征注入视觉 token。
3. 按本体状态距离构造软正样本。
4. 与行为克隆联合训练策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11798v1)
- [arXiv](https://arxiv.org/abs/2611.11798v1)

---

<a id='2611.11856v1'></a>
## [DeltaForge: A Post-hoc Mixture-of-Experts VLA for Multi-Domain Autonomous Driving](https://arxiv.org/abs/2611.11856v1)

**Authors:** Bingzhang Wang, Mo Poorsartep, Yasaman Esfandiari

**Published:** 2026-11-09

**Categories:** cs.LG, cs.RO

**Abstract:**

Fleets of driving vision-language-action (VLA) models are developed one domain at a time: each operator finetunes a shared base model on private data, producing specialized experts that are costly to consolidate into a single model. Weight merging loses accuracy as heterogeneous domains accumulate, and dense mixture-of-experts routing is prohibitively expensive to combine them. We present DeltaForge, to our knowledge the first post-hoc mixture-of-experts VLA for driving, which consolidates N independently finetuned per-domain models into a single multi-domain model from their weights and a small proxy set alone. When different operators finetune the same base model, their weight updates end up sharing a lot of structure. We exploit this by decomposing all the updates together using singular value decomposition (SVD), splitting each one into a shared part (the same across all domains) and a tiny domain-specific part. The router then only needs to work with those tiny domain-specific pieces, not the full expert weights, which is what keeps the approach efficient. Then two cooperating clip-level blind routers, one over the VLM reasoning experts and one over the flow-matching dynamics experts, are trained from soft labels on the proxy dataset. Across six diverse domains spanning urban, off-road, and highway driving on cars, trucks, and all-terrain vehicles (ATVs), DeltaForge approaches the accuracy of the per-domain experts and surpasses every merging, routing, centralized, and federated baseline at close to single-model parameter and compute cost. Adding a new domain requires only projecting its weight update onto the shared basis with no end-to-end retraining needed. The resulting model also generalizes to domain combinations it was never explicitly trained on.

### 论文解读

#### 摘要翻译
自动驾驶团队往往分别在自有数据上微调驾驶视觉-语言-动作（VLA）模型，形成针对城市、越野、高速或不同车辆的专用模型。数据不便集中共享，而直接合并权重会损失精度，保留全部专家又增加推理成本。DeltaForge 从微调后的权重和少量代理样本出发，对多域权重更新做联合低秩分解，提取共享技能基与各域小型核心，再通过两个片段级路由器混合视觉语言推理和轨迹动力学专家。六个驾驶域实验显示，它的预测误差接近各域独立模型，并以接近单模型的计算开销支持新增域接入和部分未见域泛化。

#### 方法动机分析
现实车队的环境、地区、法规和车辆动力学不同，且原始驾驶数据可能因隐私限制无法汇集。集中训练虽然能看到全部数据，但统一模型会折中不同领域；普通模型合并也容易发生权重干扰；密集专家路由则需保存和计算完整模型。作者的假设是：从同一基础模型微调出的专家更新包含可共享的低秩结构，因此可以先抽出共用方向，再只路由各域在这些方向上的小型组合系数。

#### 方法设计详解
方法以 Alpamayo 1.5 为基础，其中 8B 视觉语言模型负责场景推理，2B flow-matching 动作专家负责生成轨迹。先对每个域计算微调差分 Δi=Wi−W0，对全部差分联合做秩 r=256 的 SVD，得到冻结的输入、输出共享基 U、V；各域权重再近似写成 W0+UCiVᵀ，其中 Ci=UᵀΔiV 是每域 r×r 核心。对注意力和 FFN 层路由核心，其它模块合并。第一路由器基于视觉特征和车辆运动摘要，在整段视频片段上选择推理专家；第二个路由器读取 VLM 隐状态及相同运动量，选择轨迹动力学专家，因此场景识别与车辆动力学专家可以来自不同领域。路由器不接收域标签，而是用各专家对代理样本的任务损失形成温度软化的 oracle 标签，再通过 KL 损失训练；随后冻结路由器，只修复 top-k 核心。训练先做 VLM、再做动作专家。每域使用 200 个代理片段，约占训练数据 1%；基础专家训练使用八张 H100，VLM 阶段学习率 1e-5、batch 32，动作阶段学习率 1e-4、batch 8，最多各 10 epoch。新域可先投影到冻结基，再用本域代理样本修复核心并刷新路由器。

#### 方法对比分析
与直接权重平均相比，DeltaForge 保留每域核心并由输入动态选择；与保存完整密集专家相比，共享低秩基显著缩小新增专家的参数量。路由发生在整个驾驶片段级，而非每个 token 或层级，因为同一轨迹共享场景和车辆动力学。两个路由器又分别建模视觉推理和动作动力学，允许将不同领域的技能组合。

#### 实验分析（精简版）
六个城市、越野及高速驾驶域上，top-1 DeltaForge 的平均 minADE@5秒为 1.0157 米，独立域专家为 0.9798 米，误差提高 0.0359 米；集中式和联邦微调分别为 1.1129 米和 1.1086 米。新增专家仅增加约 42M 参数；五域配置共 12.55B 参数，top-1 约 23.5 GFLOPs/token。未训练的 Torc 卡车域误差从零样本基座 8.9862 米降至 1.0627 米。结果来自离线数据集轨迹预测，论文未报告实车或闭环驾驶验证。

#### 实用指南
复现需从同一基础 VLA 保存各域完整微调权重，并保留每域约 200 个代表性代理片段；作者使用 r=256、路由温度 0.3，并分别训练 VLM 与动作专家路由。新增域仍需其权重及代理数据，且要修复核心、更新路由器；与已有专家无关的域不一定受益。应用到车队前，应验证摄像头、运动学、法规变化下的闭环安全表现。

#### 总结
核心思想：共享低秩技能基，按驾驶片段组合专家。
1. 各域独立微调同一基础 VLA。
2. 联合分解权重差分，提取共用基与域核心。
3. 用少量代理样本训练两级片段路由。
4. 通过核心混合预测驾驶轨迹，并投影接入新域。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11856v1)
- [arXiv](https://arxiv.org/abs/2611.11856v1)

---

<a id='2611.12324v1'></a>
## [How Many Exposures Does a World Model Need? Learning Unseen Game Mechanics at Test Time](https://arxiv.org/abs/2611.12324v1)

**Authors:** Ku Onoda, Haruto Kawano, Shota Takashiro, Kohei Hayashi

**Published:** 2026-11-09

**Categories:** cs.LG, cs.AI, cs.CV

**Abstract:**

World models enable agents to plan and learn from imagined experience, but in an unfamiliar environment they must learn its mechanics from limited interaction. A mechanic is a rule for how an action or an interaction changes part of the environment. Each occurrence of a mechanic that the model observes is one exposure. We ask how many exposures a world model needs to predict the effects of a new environment's mechanics. We address this question on ARC-AGI-3, a benchmark of interactive games defined by their programs. We introduce MechEval, which labels 69 recurring mechanics across 20 games from these programs. It scores each prediction on the cells the mechanic changes at each exposure. As a test bed, we build WARC (World model for ARC-AGI-3), a 35M-parameter vision transformer that is pretrained on other games and updates all of its parameters after each transition. On held-out games, WARC learns to predict four in five mechanics with at least 80% accuracy on the cells they change and keeps that accuracy to the end of the recordings. Half of them reach that accuracy within about 80 exposures, in a single play of the game, and without updates only one in seven does at all. In closed-loop play on the public games, WARC-Agent, which pairs WARC with a fixed graph explorer, triples the explorer's official efficiency score from 0.15 to 0.46, mostly through test-time training. On the hidden games of the Kaggle leaderboard, our entry scores 0.20.

### 论文解读

#### 摘要翻译
论文提出一个具体问题：预训练世界模型面对陌生游戏时，需要多少次交互才能学会规则？作者构建 MechEval，逐次追踪游戏机制触发后哪些画面单元格发生变化，并训练 WARC 在每次交互后更新参数。它在69种机制中使56种达到80%的变化单元格准确率，约一半机制约80次触发后达标。把模型用于固定探索器后，探索效率分数也高于不使用或冻结模型的版本。

#### 方法动机分析
常见世界模型依赖大量环境交互；但新游戏或新任务开始时，代理能获得的样本有限。只看通关率难以区分记忆与规则迁移，这是机制评测的挑战。本文因此把评价单位缩小到单个可重复触发的机制，例如按键移动、点击改变对象、计数条更新，并在测试录制过程中持续测量预测。

#### 方法设计详解
流程分三步：先在其他游戏录制上预训练，再对新游戏每个转移先预测，最后用近期和回放样本在线更新。训练设置上，WARC约3530万参数，在其他16或17个游戏的人类录制上预训练，每折使用六张RTX PRO 6000训练13至20小时；推理时看前三帧及动作。在线设置是每个有效转移先做预测，再以10条最近和20条随机旧转移形成缓冲样本，执行20个梯度步；动作旋转增强为31份。作者整理20个公开的确定性网格游戏，利用游戏程序和人类录制标注机制及其变化区域。评估模型每次看到前三帧与当前动作时，能否预测下一帧中与目标机制相关的变化单元格；每种机制分别统计准确率、达到80%所需触发次数和未达比例。WARC先在其他游戏的录制上预训练，进入未见游戏后先预测，再用刚发生的转移在线更新。更新缓冲区结合最近样本和较早样本，并使用旋转增强。作者还比较冻结模型、从头训练、循环状态模型、DreamerV3和转移检索等基线，并检查预训练、更新频率和更新参数范围的作用。

#### 方法对比分析
WARC的要点不是单靠更长的历史状态，而是把预训练得到的视觉动力学先验与测试时梯度更新结合。冻结模型衡量跨游戏先验本身的作用，检索基线衡量是否可以从过往相似局部转移直接复用，循环模型则检验隐状态是否能替代权重更新。结果显示，冻结模型总体变化单元格准确率为18.4%，在线更新的WARC为69.1%；从随机初始化开始为35.0%。每五个转移才更新一次会损害最初几次机制触发的学习表现。

#### 实验分析（精简版）
在69种机制中，WARC让56种达到80%准确率，机制准确率中位数所需触发次数为74；冻结模型仅10种达标。屏蔽目标机制的触发样本后，该机制上的在线改进接近消失，支持模型利用了自身机制样本。跨录制测试还显示，已学会的移动规律能迁移到新位置或方向，但面对新对象、新效果和被障碍阻挡的移动时更容易出错。闭环实验中，WARC-Agent在25个公开游戏的1000步评测上取得0.455的RHAE，高于图探索器的0.149和冻结模型组合的0.181；然而探索器只追求新颖性，并不理解目标。隐藏游戏公榜得分为0.20，实际时限也会显著压缩可执行步数。

#### 实用指南
复现需准备游戏代码、录制与GPU。研究仅覆盖确定性64×64游戏，机制标注依赖程序，录制未必覆盖所有状态。迁移到机器人或交互软件时，应先定义事件和影响区域，加入安全约束，并测试随机反馈、连续动作、受阻状态及高代价错误。论文尚未证明该方法适用于这些场景。

#### 总结
核心思想：预训练先验结合逐次在线更新。
1. 在其他游戏预训练视觉动力学。
2. 对新转移先预测，再更新模型。
3. 按机制评估学习速度并用于探索。证据限于特定网格游戏。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.12324v1)
- [arXiv](https://arxiv.org/abs/2611.12324v1)

---

<a id='2611.11709v1'></a>
## [Data-Driven Precision Control of Soft Helicoid Robots](https://arxiv.org/abs/2611.11709v1)

**Authors:** Roshan S. Kaundinya, Mark Leone, Roberto Sabella, Hugo Buurmeijer, Luis Pabon, John Irvin Alora, Marco Pavone, George Haller

**Published:** 2026-11-09

**Categories:** cs.RO, cs.LG, math.DS

**Abstract:**

A purely data-driven method grounded in the theory of adiabatic spectral submanifolds (aSSMs) was recently used to model and control nonlinear soft robotic simulators. Here, we extend the methodology to learn aSSM-reduced models from data collected from hardware soft robotic platforms. Specifically, we learn a five dimensional (5D) aSSM-reduced nonlinear ordinary differential equation (ODE) that accurately models the dynamics of a soft-helicoid trunk robot. We then use this ODE in a model predictive control scheme to demonstrate real-time precision control across various challenging closed-loop trajectory-tracking tasks, ranging from 2D to 3D trajectories with static and dynamic constraints. We find that our methodology enhances accuracy and precision across all control tasks compared to available data-driven linear methods and model-free reinforcement learning. While we focus here on a soft-helicoid trunk robot, our methodology relies solely on hardware data and is therefore applicable to a broader class of soft robotic systems.

### 论文解读
#### 摘要翻译
作者从软体机器人硬件数据中学习五维降阶动力学，并将其放入模型预测控制器。对于软螺旋躯干，方法在二维和三维轨迹跟踪及虚拟障碍约束任务中，比所比较的数据驱动线性模型和模型无关强化学习策略表现更准确。工作目标是避开繁重物理建模，同时保留足以支持实时控制的非线性结构。
#### 方法动机分析
软体机构的连续形变、材料迟滞和欠驱动特性，使有限元或Cosserat模型难以快速部署；线性数据模型虽然便宜，却无法表达大幅弯曲；硬件强化学习则需要长时间试错。论文利用一个结构性观察：软体机器人内部快模态会迅速衰减，而任务轨迹通常变化较慢。若这一时间尺度分离成立，系统主要运动可由低维吸引不变流形描述。作者据此把数据约化建立在aSSM理论上，而非任意压缩状态。
#### 方法设计详解
输入是运动捕捉得到的机器人末端和中段位置，以及六路腱驱动命令。第一步用随机阶跃命令采样多种静态形态和回归平衡的衰减轨迹；第二步将观测延迟嵌入，使缺失的速度信息可由历史位置补充；第三步以静态数据拟合控制输入到平衡观测的映射，并对每组衰减数据做奇异值分解，学习五维慢动态坐标；第四步拟合该坐标上的低阶动力学与观测重建，再用平滑随机命令校准输入变化带来的修正项。最终模型由六维控制输入和五维潜状态构成。作者采集30分钟阶跃数据和90分钟平滑随机数据；992个静态状态各配10条衰减轨迹，按90%/10%划分训练与测试。在线阶段，MPC优化跟踪误差、输入幅值与输入变化率，并满足状态和控制限制。求解使用GuSTO将非凸问题逐步近似为凸子问题，再用HPIPM求解；针对跟踪任务，MPC时域取M=10，障碍任务取M=20；控制器推理频率为50 Hz。每轮只执行最优控制序列前两步，然后根据新观测重新规划。
#### 方法对比分析
与DMDc等线性系统相比，aSSM保留了随控制输入变化的非线性平衡形态和慢模态；相较黑箱强化学习，它显式预测受控系统演化，并可将障碍约束放入规划器。与带记忆的MZ-Koopman方法相比，其模型维度更小，实时优化负担较低。关键区别是将“数据驱动”与有理论动机的流形结构结合，适合需要跨较大工作空间的软体闭环控制。
#### 实验分析（精简版）
在未见衰减数据上的归一化平均轨迹误差为9.6%。闭环测试汇总表中，aSSM的最大误差为5 cm、控制频率50 Hz；DMDc为18 cm，Koopman-pregain为38 cm，MZ-Koopman为17 cm，PPO为10 cm。三维轨迹图显示aSSM更贴近目标及3 cm不确定带，静态与移动虚拟障碍实验展示了约束避让能力。实验只在一个软螺旋躯干上进行；障碍是虚拟的，且较大球形约束下仍进入缓冲区，因此尚未证明真实接触环境中的安全性。
#### 实用指南
复现需要同步采集驱动输入与多个刚性标记点的位置，保留静态形态、衰减过程和慢变校准数据。输入需覆盖工作空间，延迟长度应足以嵌入主导动态；MPC还需按硬件频率和求解时延设置滚动时域。迁移到其他软体平台前，应检查内部模态是否明显快于目标运动，并重新学习平衡流形、慢坐标和校准项。论文未在正文明确给出代码仓库，因此复现主要依赖方法细节与补充材料。
#### 总结
核心思想：用慢流形模型驱动软体控制
1. 采集静态、衰减和慢变输入下的硬件观测。
2. 延迟嵌入并从衰减数据提取低维慢动态。
3. 拟合平衡映射、降阶动力学及输入变化校准项。
4. 将模型放进带约束的滚动MPC，按新观测持续更新。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11709v1)
- [arXiv](https://arxiv.org/abs/2611.11709v1)

---

<a id='2611.11978v1'></a>
## [CSG-Nav: Confidence-Aware Semantic-Geometric Fusion for Robust Exploration and Accurate Zero-Shot Object-Goal Navigation](https://arxiv.org/abs/2611.11978v1)

**Authors:** Shoubin Chen, Rui Xu, Zehao Wu, Kai Zhang, Tao Huang, Zhou Liu, Qingquan Li

**Published:** 2026-11-09

**Categories:** cs.RO

**Abstract:**

Zero-shot object-goal navigation requires autonomous agents to explore unseen environments and locate target objects without task-specific training, posing challenges in perceptual reliability under observation uncertainty and physically grounded goal termination. Although foundation-model-based approaches improve generalization, they remain vulnerable to unreliable long-range perception, hallucination-induced directional bias, and semantically plausible but geometrically incorrect stopping decisions. To address these limitations, we propose CSG-Nav, a zero-shot navigation framework centered on confidence-aware multimodal information fusion. CSG-Nav integrates three complementary fusion mechanisms. First, semantic-source fusion integrates open-vocabulary detection results, detection confidence, scene observations, and relational reasoning to produce more reliable direction-level semantic scores and suppress hallucination-induced directional bias. Second, spatial-confidence fusion weights multi-view semantic evidence using a Gaussian decay function, suppressing unreliable distant observations while accumulating spatially consistent exploration preferences. Third, semantic-geometric decision fusion combines semantic target detection with depth-based 3D localization to support physically grounded stopping decisions. Experiments on HM3Dv1, HM3Dv2, and MP3D demonstrate competitive zero-shot performance, achieving success rates of 58.9%, 71.6%, and 44.3%. Fusion-level ablations and real-world deployments in indoor and outdoor environments further verify the effectiveness and practicality of CSG-Nav for robust navigation under uncertain perception. Code is available at https://xr-ui.github.io/CSGNav.github.io/

### 论文解读

#### 摘要翻译
零样本物体导航要求机器人在陌生环境中寻找自然语言指定的目标，但基础模型可能误信远距离线索、受语义幻觉影响，或在目标可见但尚不可到达时提前停止。CSG-Nav提出三层置信度融合：结合开词汇检测置信度与视觉语言场景描述，按观测距离衰减空间证据，并在停止前融合目标语义和深度几何位置。HM3Dv1、HM3Dv2和MP3D仿真结果显示其成功率有竞争力；作者也在室内外移动机器人上进行了初步部署。

#### 方法动机分析
现有零样本导航系统常把 VLM 场景判断或 LLM 常识分数直接送入规划器。远处物体像素较少、深度更不稳定，幻觉线索可能把地图偏好带偏；即使识别到目标，也不代表机器人距离足够近或已到可交互位置。论文假设可用检测置信度、与机器人距离以及深度定位作为简单可靠性线索，分阶段过滤语义证据和停止决策。

#### 方法设计详解
每个导航步扫描水平面的六个候选方向，取得 RGB-D 观察。YOLO-World输出各方向的类别、检测框和置信度；VLM对多视角全景描述房间与通道；LLM把两类证据结合目标类别，给每个方向输出0–10相关分。空间融合将方向分投影到可达区域，并乘高斯距离权重 exp(−d²/(2σ²))，降低远处观察影响，再用历史偏好和置信状态做加权更新。机器人选择偏好最高的方向，生成子任务，并从深度图筛选局部可行移动动作。检测到目标后，系统选取置信度至少0.25的目标框，根据框中心深度计算全局三维坐标，并平均最近三帧位置。只有类别匹配且目标距离小于1.0米才发出停止命令，否则继续接近、微调和观察。

#### 方法对比分析
CSG-Nav的重点是决策链条中的多源证据管理，而不是学习一套新的端到端策略。它将物体级检测和场景级语义交给 LLM联合判断，用距离先验控制地图更新，再将深度距离加入停止条件。相比纯语义停止，这可减少看见目标便终止的风险；相比完整概率不确定性建模，距离衰减更简单、无需专门训练，但表达能力也有限。

#### 实验分析（精简版）
HM3Dv1、HM3Dv2和MP3D上的成功率分别为58.9%、71.6%和44.3%；HM3Dv2比 WMNav 高1.8个百分点，但SPL为29.5%，略高于WMNav的27.8%。在固定100个HM3Dv2案例的单次消融中，语义加YOLO几何停止为77% SR、28.99 SPL，语义单独停止为75%、27.62。实机每种目标运行五次，成功率在1/5至5/5之间。实机结果未与同平台基线配对，且作者指出高斯距离先验和地图初始化对结果敏感，仍缺少受控鲁棒性评估。

#### 实用指南
论文代码链接为 https://xr-ui.github.io/CSGNav.github.io/ 。复现需准备RGB-D相机、YOLO-World、Qwen3-VL-4B和Qwen3-4B，使用40步上限、0.5至1.7米候选动作，并针对相机和深度传感器调整1米停止阈值及距离衰减尺度。实机演示使用 Jetson AGX Orin 64GB。迁移至需要抓取或交互的机器人时，应将“成功到达”改成任务对应的可操作位姿，并测试遮挡、动态场景和视角覆盖。

#### 总结
核心思想：融合语义、距离置信与几何再决定停止。
1. 多方向采集 RGB-D 观察。
2. 联合检测与场景描述评估探索方向。
3. 按距离和可达性累积地图偏好。
4. 用目标三维位置确认到达后再停止。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11978v1)
- [arXiv](https://arxiv.org/abs/2611.11978v1)

---

<a id='2611.12557v1'></a>
## [WanderBench: Can World Models Explore Dynamic Worlds from Video?](https://arxiv.org/abs/2611.12557v1)

**Authors:** Yuhang Chen, Jiheng Li, Jiale Wang, Zai Wang, Guangyue Liu

**Published:** 2026-11-09

**Categories:** cs.CV, cs.AI

**Abstract:**

Video world models conditioned on an input video make it possible to explore real, recorded dynamic scenes, enabling re-shooting of footage, free-viewpoint replay, and real-to-sim environments for embodied agents. Such uses require the generated rollout to stay faithful to the recorded world, yet existing benchmarks start from a single image or a text prompt, and video-conditioned generation is evaluated mainly on perceptual quality and camera following. These evaluations do not directly assess whether exploration preserves subject identity, event progression, and scene consistency upon revisiting a location. We introduce WanderBench, a benchmark for exploring dynamic worlds from video. It comprises 745 single-shot source videos of real and AI-generated content in three difficulty tiers (3-25 s), paired with camera routes given as keyboard actions, and evaluates six dimensions with 24 human-validated sub-metrics. Benchmarking eight models, we find that as clips grow longer, models keep the static scene and visual quality but lose camera control and the source dynamics, and that none keeps revisited places consistent. We further propose WanderLand, a training-free framework that conditions each frame of a frozen model on the same-instant source and, on revisits, on its own earlier static content. WanderLand improves five models on every dimension, by up to 8.3 points in source-dynamics fidelity.

### 论文解读

#### 摘要翻译
本文提出 WanderBench，评估视频世界模型能否从一段动态源视频出发，按照用户给定的相机动作探索场景，同时保留原视频中的主体、事件和时间进程。基准包含745段视频、三种时长难度、六个评估维度和24项子指标。作者测试八个模型，发现视频越长，模型越难遵循相机路径并保持源事件同步，重访场景也不稳定。另提出无需训练的 WanderLand，通过按时间对齐源视频及模型先前生成的静态内容组织条件输入，在五个模型上提升各项指标。

#### 方法动机分析
现有视频生成评测通常关注画面质量，世界模型评测则更多看动作控制或一般一致性；但视频条件探索还面临一个挑战：模型可能生成清晰画面，却改变源视频里事件的发生时间，或在回到旧地点时把它重新画成不同状态。研究的动机是将“是否像一段好视频”和“是否忠实探索这段记录中的世界”分开度量，并同时检查动态主体与静态场景。

#### 方法设计详解
WanderBench收集745段3至25秒的单镜头视频，包括真实或渲染素材、既有数据集片段和AI生成内容，并按时长、运动幅度和视觉复杂度分成三档。用户通过键盘动作指定相机路线，包含随机探索和往返重访。评测包含视觉质量、相机控制、源世界保真、源动态保真、时空一致性、动态重访一致性六维24项指标，以同一时间点的源帧作为参照。WanderLand是免训练的推理期框架：第一步把目标相机与源视频同一时刻的相机对齐；第二步把同刻源图像投影到目标视角；第三步仅在重访且源图像覆盖不到时，用先前生成的静态内容填补空洞，并剔除随时间改变的内容，避免把运动主体冻结。它沿用各模型自身几何和渲染模块，不改模型参数。

#### 方法对比分析
与单看画质或相机跟随的基准相比，WanderBench将事件同步和动态重访作为独立维度，能区分静态场景保持良好、但动作时序已偏离源视频的情形。WanderLand也不是重新训练一个视频模型，而是改变输入证据的优先级：源视频同刻信息优先，过去生成的静态内容只用于补充遮挡区域，最后才由原模型生成。这种设计针对时序失真和重访不一致，优势是无需额外训练；适用范围则受到底层模型是否开放几何、渲染输入接口的限制。

#### 实验分析（精简版）
八个模型的长视频结果显示，视觉质量通常比相机控制和源动态保真更稳定。Hard档中全长度模型的整体分数为48.8至62.3，WorldForge v2相机控制为10.4，动态重访一致性最低为20.5。WanderLand在五个兼容模型的六个维度均有非负变化；InSpatio-World 1.5的源动态保真提高8.3分，五个模型的动态重访一致性提高2.4至4.8分。两个未纳入的系统本身也是训练免框架，且接口不支持重新组织其输入；因此效果尚未覆盖所有模型家族。

#### 实用指南
复现或迁移需要源视频、目标相机路径，以及模型自身的几何估计和点云渲染模块。使用时应验证源相机与目标帧的时间对齐、深度投影误差和运动主体过滤，尤其检查重访画面是否错误冻结动态内容。基准源包括网络视频和生成素材，应用前需确认数据授权及可获得性。该方法适合评估视频重放或合成环境中的视角探索；单镜头源视频不能直接代表带剪辑、交互因果或长时动态变化的场景。

#### 总结
核心思想：用源视频锚定视角探索
1. 按多维指标检查控制和事件保真。
2. 用同一时刻的源帧约束新视角。
3. 重访时以静态记忆补足未覆盖区域。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.12557v1)
- [arXiv](https://arxiv.org/abs/2611.12557v1)

---

<a id='2611.11735v1'></a>
## [Correctness Metric Matters: NWD-Based Calibration Loss for Tiny Person Detection in UAV Infrared Imagery](https://arxiv.org/abs/2611.11735v1)

**Authors:** Tian Zhang, Zhirong Su

**Published:** 2026-11-09

**Categories:** cs.CV

**Abstract:**

Detection calibration aligns predicted confidence with empirical correctness. Existing calibration losses inherit their correctness definition from box regression, specifically IoU against a hard threshold. We show that this target is geometrically ill-posed for tiny objects in UAV infrared imagery, where persons occupy less than 0.1\% of the image. The same pixel error is penalized more heavily as the target shrinks, so binary prediction calibration (BPC) actively suppresses confidences that should be high. On HIT-UAV, BPC with an IoU target degrades image-level localization-aware ECE (LaECE) from 0.0689 to 0.0761, 0.0649 to 0.0733, and 0.0653 to 0.0682 across Low, Mid, and High signal-to-clutter ratio (SCR) groups. We replace the IoU target with a continuous, scale-invariant normalized Wasserstein distance (NWD). The NWD target reduces LaECE\(_{\mathrm{NWD}}\) relative to BPC+IoU on all three groups (\(p<10^{-7}\)) with mAP neutral (0.9356 vs.\ 0.9347), and an independent center-offset anchor confirms that the gain is a genuine geometric correction. The improvement holds across a broad plateau of calibration weights. The NWD formulation requires no inference-time mapping layer. If a validation-fitted isotonic map is applied on top of it, calibration is lowered further, and the two corrections act on different components of the same error. The metric-level remedy has a measurable boundary. Target-background thermal pairing in the training distribution and detection-level aggregation under per-image detection-count heterogeneity lie outside it. Both are quantified independently of the NWD target.

### 论文解读

#### 摘要翻译
目标检测的置信度应反映预测框的实际正确程度。常见训练期校准方法以 IoU 阈值构造正确性标签，但在无人机红外图像中，行人框可能只有十几个像素；同样的几像素偏移会让小框的 IoU 急剧下降，导致校准损失把视觉上合理的检测压成低置信度。本文将校准目标改为连续的归一化 Wasserstein 距离（NWD），在 HIT-UAV 上改善不同信噪组的定位感知校准，同时基本保持检测精度。作者还分析了背景热纹理配对与按检测框聚合带来的额外误差。

#### 方法动机分析
若一个 4×12 像素行人框只偏移 (2,1) 像素，IoU 约为 0.30，常规阈值会把它当成错误；但偏移相对目标本身很小。问题不只是 IoU 不平滑，而是它的重叠比例天然依赖目标尺度。作者提出，训练校准所用的“正确性”度量应避免固定像素误差随目标变小而被夸大。核心假设是：在极小目标场景，用框中心和尺度构造的连续几何相似度，比硬 IoU 阈值更贴近检测框的实际定位质量。

#### 方法设计详解
方法保留原有检测器、IoU 匹配分配器和检测损失，只修改训练期置信度校准项。输入是模型预测框、匹配到的真值框及对应置信度；每个框被表示为二维高斯，中心对应框中心，协方差由宽高决定。随后计算预测与真值高斯之间的二阶 Wasserstein 距离，并用训练集目标框的平均几何尺度归一化，再通过指数映射得到 [0,1] 的 NWD 相似度。该连续分值作为匹配正样本的校准目标，未匹配预测仍以零为目标，校准损失则惩罚置信度与目标分数之差。NWD 只介入校准目标，不改变正负样本分配，也不替换框回归损失。实验采用 YOLOv8-s、640×640 输入；HIT-UAV 主对比训练 50 个 epoch，并使用 λ=50 作为校准权重。部署时校准项不进入前向网络，因此不增加推理层或参数。

#### 方法对比分析
IoU 以交并面积比较框，微小框对几个像素的移动高度敏感；NWD 比较高斯表示的中心与尺度差，在相同中心偏差下不因框尺寸缩小而塌缩。相较温度缩放或等距回归等后处理，NWD 从训练阶段改变了监督信号；相较把 NWD 用到匹配器中，本文只把它用作校准分值，避免影响检测器的正样本选择。其优势集中于微小目标，不代表所有尺度都更适用。

#### 实验分析（精简版）
HIT-UAV 测试集包含 579 张图像，主结果为三随机种子的 50 epoch 训练。NWD 校准的 mAP50 为 0.9356，IoU 校准为 0.9347；低、中、高信噪组的 NWD 定位感知校准误差分别从 0.0957、0.0993、0.0948 降至 0.0861、0.0914、0.0868。独立中心偏移指标也支持定位校准改善。将 NWD 同时用于匹配会使 mAP50 降至 0.9254。FLIR 街景红外实验中，NWD 的 mAP50 比 IoU 校准低 0.0101，说明收益受目标尺度分布限制。结果来自特定红外数据集与训练配置，不能直接证明其它传感器或部署场景也会获益。

#### 实用指南
实现时应复用原检测分配器，只在校准损失中以 NWD 替换 IoU 正确性目标；归一化尺度、校准权重和验证指标需按训练集目标尺寸重新检查。论文报告 HIT-UAV 采用 C=15.60 像素、λ=50，并在 λ∈{20,35,50,65,100,200} 上比较；大权重会使 IoU 一致性恶化。NWD 不增加推理计算，适合对运行时开销敏感的模型。迁移至自动驾驶或机器人视觉时，应先按目标大小分层，尤其验证中大型对象上的 mAP 变化，并同时报告独立定位指标。

#### 总结
核心思想：让校准度量符合微小目标的几何误差。
1. 检测器预测目标框并用原分配器匹配真值。
2. 将预测框和真值框映射为高斯表示。
3. 计算归一化 Wasserstein 相似度作为软正确性目标。
4. 训练置信度贴近该分值，推理网络保持原样。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11735v1)
- [arXiv](https://arxiv.org/abs/2611.11735v1)

---

<a id='2611.11770v1'></a>
## [Towards Rotation-Only Imaging Geometry: Multi-View Pose Estimation](https://arxiv.org/abs/2611.11770v1)

**Authors:** Xinrui Li, Qi Cai, Yuanxin Wu

**Published:** 2026-11-09

**Categories:** cs.CV

**Abstract:**

Global structure-from-motion (SfM) commonly refines relative poses on individual image pairs and then estimates global rotations by rotation averaging. Pairwise refinement does not explicitly combine feature-level constraints with multi-view consistency during rotation initialization. We introduce Global Rotations from Aggregated Matches (GRAM), which estimates global rotations directly from initial relative rotations and inlier matches by minimizing edge-wise rotation-only algebraic residuals. GRAM replaces two-view relative-pose optimization and rotation averaging with a single stage. We further extend rotation- only imaging geometry to multiple views and develop Multi-view Optimization with Rotation-only Reprojection Error (MORE) for nonlinear global rotation refinement. Analytical Rotation-guided Translation (ART) derives locally scaled triplet translations from Linear Global Translation (LiGT) constraints and recovers global translations using weighted least squares. Synthetic and real-world experiments demonstrate the accuracy, robustness, and efficiency of the proposed methods.

### 论文解读
#### 摘要翻译
传统全局 SfM 通常先优化图像对之间的相对位姿，再通过旋转平均估计全局旋转。本文提出 GRAM，直接用全局旋转和边级匹配约束初始化姿态；提出 MORE 扩展多视图旋转重投影优化；并提出 ART，从局部三视图几何解析恢复全局平移。作者在合成和真实场景上评估了准确性、鲁棒性与速度。

#### 方法动机分析
分阶段 SfM 会让两视图匹配细节与全局一致性脱节：两视图精化无法利用全局约束，旋转平均也不再查看原始匹配。作者假设旋转-only 几何可将特征级误差直接写入全局旋转优化；再利用多视图轨迹约束先解局部相对平移，最后统一求全局平移。

#### 方法设计详解
GRAM 首先根据相对旋转的一致性筛选可靠边，并将视图图分成连通子图；每个子图用加权最小生成树初始化相机旋转，再依据匹配特征构造边级残差联合精化。随后通过跨子图候选旋转聚类估计组件间对齐。MORE 使用三视图特征轨迹和 LiGT 几何推导局部平移，将其代入旋转-only 重投影关系，构造稳健损失并联合优化全局旋转。ART 对有效三元组构造解析的局部平移估计，再将多个局部解写成含相机平移与局部尺度的加权线性系统，用最小二乘求解全局平移，并通过手性约束确定符号。合成实验使用 10 台相机、1000 个共享三维点、8 个噪声等级，每级进行 500 次试验；真实数据包括 19 个有位姿真值的场景。

#### 方法对比分析
与“逐对精化—旋转平均—全局 BA”的传统流程相比，GRAM 在全局初始化时重访原始匹配约束，MORE 进一步联合多视图误差，ART 则避免非线性局部平移注册。它们主要改变全局位姿估计阶段的几何组织方式，不是完整 SfM 系统的所有组件替换。

#### 实验分析（精简版）
在 19 个真实场景中，GRAM 的旋转初始化误差均低于 GLOMAP-Avg；固定旋转下，ART 在 16/19 场景取得最低平移误差。报告的完整管线中位耗时为 0.809 秒，GLOMAP 为 16.781 秒。但两者计时均不包括共享五点 RANSAC；作者管线结束于 ART，而 GLOMAP 还包括三角化与全局 BA，因此速度比较并非完全相同的重建终点。

#### 实用指南
实现需要相机内参、两两匹配和初始相对位姿。作者使用 COLMAP/GLOMAP 产生相机模型、匹配与视图图，再以 PoseLib 五点 RANSAC 获取相对位姿；GRAM、MORE、ART 在单核 CPU 上运行。对机器人定位迁移时，需特别检查纯旋转、平面退化、错误匹配、尺度规范和后端 BA 的配合。

#### 总结
直接把匹配约束带入全局位姿
1. 筛选可靠边并初始化全局旋转。
2. 用多视轨迹优化旋转。
3. 从局部三元组解析平移。
4. 加权求解全局平移。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.11770v1)
- [arXiv](https://arxiv.org/abs/2611.11770v1)

---

<a id='2611.12319v1'></a>
## [Mitigating SM-Level Load Imbalance in 3D Gaussian Splatting on GPUs](https://arxiv.org/abs/2611.12319v1)

**Authors:** Junha Ryu, Hoi-Jun Yoo

**Published:** 2026-11-09

**Categories:** cs.DC, cs.CV, cs.GR, cs.LG

**Abstract:**

3D Gaussian Splatting (3DGS) enables photorealistic, real-time rendering on commodity GPUs and is increasingly adopted in generative world models and autonomous driving simulators. However, existing optimizations rely on algorithmic changes that often require retraining, or on specialized accelerators that lack flexibility across evolving 3DGS variants. In this paper, we identify a fundamental inefficiency in GPU-based 3DGS rendering: severe SM-level load imbalance caused by uneven workloads across screen tiles. This imbalance creates a long execution tail, leaving most SMs idle while a few process computationally intensive tiles, particularly in large-scale urban scenes. To address this, we present a software-defined GPU scheduling approach that mitigates the execution tail. Our approach uses a persistent kernel to schedule heavy tiles first and selectively chunks outlier tiles across multiple CTAs. A lightweight per-frame policy determines when and how much to chunk using tile statistics already available in the rendering pipeline. Our approach requires neither retraining nor specialized hardware and can be applied orthogonally to existing optimizations across diverse 3DGS variants. Averaged per scene, it accelerates alpha blending by up to 2.93x on a large-scale urban scene, and by up to 5.6x on its most imbalanced frames, while retaining the original alpha blending formulation.

### 论文解读
#### 摘要翻译
3D Gaussian Splatting 的 tile 光栅化会把不同工作量的 tile 固定分配给 CTA。少数繁重 tile 可能拖长执行尾部，使其他 SM 空闲。本文以 persistent kernel 优先调度重 tile，并选择性地把极重 tile 切成多个深度块并行处理，同时用在线代价模型决定每帧切分程度。方法不修改场景表示，也不需要重新训练。

#### 方法动机分析
常见优化侧重减少高斯数量或依赖专用硬件，但已有场景仍可能因 tile 负载差异而无法充分利用 GPU。作者针对这一调度痛点指出，tile 高斯数只是工作量代理，真正的执行尾部取决于最重 tile 相对 GPU 并行能力的大小。因此固定比例切分所有 tile 会在均衡场景中引入不必要开销。

#### 方法设计详解
方法流程从屏幕 tile 的高斯数统计开始：3DGS 将高斯投影到屏幕 tile 后按深度排序，再对每个像素做 alpha blending。本文先从排序列表得到每个 tile 的投影高斯数，以 persistent kernel 建立软件工作队列，并按 longest-processing-time-first（LPT）顺序让重 tile 先执行。对仍然过重的 tile，将深度顺序连续分成多个 chunk，各 CTA 计算局部颜色和透射率，最后按深度归并。独立 chunk 会失去前面片段的提前终止信息，因此作者加入饱和标志：若较早 chunk 已使所有像素透射率低于阈值，后续 chunk 可停止处理。chunk kernel 与普通 LPT tile 在不同 CUDA stream 并发运行。每帧轻量代价模型根据高斯数量、CTA 并行度和 chunk 额外代价，选择切分 tile 数和每 tile chunk 数；当预测节省不足时不切分。

#### 方法对比分析
LPT 只改变重 tile 开始时间；chunking 则把一个 tile 的工作分给多个 CTA，但会增加中间结果和归约开销。二者结合并由每帧策略选择，避免对均衡场景套用固定切分。算法保持原有 alpha blending 形式，适用于多种 tile-based 高斯渲染器。

#### 实验分析（精简版）
在 Hier-3DGS 城市场景，alpha blending 平均加速 2.93 倍，单帧最高 5.64 倍；驾驶场景平均 1.23 倍，标准 3DGS 基准平均 1.12 倍。均衡场景的收益较小，某些固定 chunk 设置会变慢。chunking-only 相对原渲染器在 24 个场景中的最大 PSNR 差为 0.0015 dB。实验加速针对 alpha blending 阶段，不能直接等同完整端到端帧率提升。

#### 实用指南
方法在 NVIDIA GB10（48 个 SM）和 RTX A6000（84 个 SM）上评估，并覆盖 3DGS、2DGS、3DGUT 及城市/驾驶场景。实现需要按深度分块、两条 CUDA stream、饱和标志与末端归约。部署时需按目标 GPU 测量每帧 tile 负载；不均衡程度和 SM 数量会改变收益，且需评估与同卡神经网络推理的竞争。

#### 总结
动态调度并切分重 tile
1. 统计每个 tile 的高斯数。
2. LPT 优先派发重 tile。
3. 并行切分长尾并传递饱和状态。
4. 按帧代价决定是否切分。
论文还显示收益与最重 tile 相对每个可并发 CTA 工作量的比例相关；例如 NuScenes 场景 005 达到 1.77 倍，而 LPT 单独调度仅 1.06 倍。作者报告计数排序低于 alpha blending kernel 时间的 1%，但并未把这一阶段的加速直接换算为完整仿真系统吞吐。部署评估应加入同步开销、帧间负载变化和其他 GPU 工作负载。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.12319v1)
- [arXiv](https://arxiv.org/abs/2611.12319v1)

---

