time: 20261019

# Arxiv Computer Vision Papers - 2026-10-19

## Executive Summary

## 今日论文概览

本日入选的十篇论文集中在机器人基础模型的效率、长期记忆、多机器人协作，以及复杂环境中的感知和评测。MambaAct 将计算优化放在 flow-matching VLA 的动作专家，报告在 LIBERO 和两项 UR5 任务上降低推理延迟；VLA-Fabric 则保留每个机械臂的完整本地策略，通过共享语义、同伴状态和动作意图协调固定的多臂团队。两者分别代表单体策略压缩和跨策略交互，适合关注 VLA 架构与部署效率的读者。

在移动与自动驾驶方向，SpatialBEV 将未来轮胎接触区域的地形特征引入越野车辆状态预测；Expand-and-Prune 在冻结规划器中增加意图与时间候选，再根据中间评分的不确定度调整候选预算。STRIDE 将驾驶视觉语言模型的空间、时间、风险与轨迹能力拆分评测，显示场景解释流畅度与精细安全判断并不等价。STRIDE 的模板审查和标签修订也提醒读者，基准分数要结合数据构造边界理解。

长期系统方面，InstAgent 用行为树连接异构机器人的技能组合、执行复盘和共享记忆；VLA 记忆综述从单回合上下文延伸到部署期持久记忆，指出完整自演化记忆尚未成熟。LonGaussian 面向静态长视频，以跨帧跟踪辅助合并高斯表示，在有限显存下扩展序列长度，但长序列闭环误差仍造成重影。草莓机械臂工作展示示教策略在 Orin 上的接近与对准，尚未完成实际切梗采摘。空间机器人综述整理跨域证据并提出仿真到地面类比再到空间任务的验证路线，而非新的实验结果。

建议优先精读 MambaAct、VLA-Fabric 和 SpatialBEV，分别理解动作专家压缩、策略间表示交互和地形条件动力学；若关注评测与系统可靠性，可接着读 STRIDE、InstAgent 及空间机器人综述。整体研究趋势是把模型能力与部署条件、状态记忆和可验证闭环一起评估，而不是只看单一离线分数。

---

## Table of Contents

1. [MambaAct: Efficient Mamba Action Experts for Flow-Matching Vision-Language-Action Models](#2610.22951v1)
2. [InstAgent: A Recursively Self-Improving Agent Operating System for Heterogeneous Robot Collaboration](#2610.23098v1)
3. [Structured Perception Representations for Off-road Dynamics Learning](#2610.22900v1)
4. [Expand-and-Prune: Test-Time Candidate-Space Adaptation for End-to-End Autonomous Driving](#2610.23262v1)
5. [LonGaussian: Tracking Foundation Priors for Long-Context Feedforward Gaussian Reconstruction](#2610.23250v1)
6. [Foundation Models for Autonomous Space Robotics: Architectures, Physical Adaptation, and Validation](#2610.23068v1)
7. [Pedicel-Targeted Strawberry Harvesting via Imitation Learning on a Low-Cost Teleoperated Robotic Arm](#2610.22765v1)
8. [VLA-Fabric: Learning Interaction for Composable Multi-Arm VLA Agents](#2610.23018v1)
9. [STRIDE: Evaluating Spatiotemporal Reasoning in Driving Edge Cases](#2610.22795v1)
10. [From Sliding Windows to Self-evolving Lifelong Memory: A Dual-perspective Survey of Memory Paradigms for Long-Horizon Robotics VLA Models](#2610.23319v1)

---

## Papers

<a id='2610.22951v1'></a>
## [MambaAct: Efficient Mamba Action Experts for Flow-Matching Vision-Language-Action Models](https://arxiv.org/abs/2610.22951v1)

**Authors:** Khuyen Pham, Tran Nguyen Le, Edoardo Fazzari, Cesare Stefanini

**Published:** 2026-10-16

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision-language-action (VLA) models have shown strong promise for generalist robotic manipulation, but their inference cost remains a major bottleneck for real-time control. This limitation is especially pronounced in flow-matching VLAs such as $π_{0.5}$, where the action expert is evaluated repeatedly during iterative denoising. We introduce MambaAct, an efficient Mamba action expert for flow-matching VLA policies. MambaAct retains the pretrained visual-language prefix encoder of $π_{0.5}$, while replacing its transformer-based action expert with a compact Mamba-based action expert. Within this expert, Mamba serves as the action mixer that models dependencies across the noisy continuous-action sequence. To preserve visual-language conditioning without repeated token-level cross-attention, MambaAct uses a lightweight Context Compressor to map Vision-language model (VLM) prefix tokens into a compact conditioning signal that gates and modulates each Mamba action block. MambaAct uses a $10\times$ smaller action expert and reduces flow-matching denoising latency by $64.8\%$ on LIBERO and $58.8\%$ in real-world deployment, while improving mean LIBERO success by $1.08$ percentage points across three independent fine-tuning seeds and maintaining gains on LIBERO-Plus and real-world benchmarks. Our code and model weights are publicly available at https://github.com/giakhuyendihoc/MambaAct.

### 论文解读
#### 摘要翻译
作者提出 MambaAct：保留视觉语言动作模型的视觉语言主干和 flow-matching 训练方式，把计算量较大的 Transformer 动作专家换成更轻的 Mamba 状态空间网络。目标是在不牺牲操作质量的情况下减少动作生成开销。
#### 方法动机分析
VLA 每次预测动作都要反复去噪，论文的延迟分解显示动作去噪是主要耗时环节。直接缩小整个模型可能损害感知和语言能力，因此作者把优化边界放在 action expert，假设紧凑的序列模型仍能处理动作 token 之间的时序依赖。这个假设需要在任务质量和延迟两方面同时验证。
#### 方法设计详解
输入包括腕部与第三人称图像、指令以及带噪动作块。首先，π0.5 视觉语言编码器生成前缀特征；视觉 token 被分段池化压缩，语言 token 保留，并由少量可学习 query 提取全局摘要。随后这些多模态特征被投影到动作专家维度，与动作 token、流时间嵌入组合。四个带门控的 Mamba block 逐层更新动作序列，FiLM 参数根据流时间调制状态；输出速度头估计去噪方向，仍用 flow-matching 损失训练。这样只重构动作专家，整体策略仍可沿用原 VLA 的输入、动作表示和训练目标。作者报告使用单张 H100、batch size 16、训练 30k 步。
#### 方法对比分析
与原 π0.5 Transformer action expert 相比，MambaAct 显著减少专家参数量，并通过视觉上下文压缩控制动作专家输入长度；与完全更换 VLA 骨干的方案相比，它保留原视觉语言能力，改动集中、便于在已有模型上微调。代价是增加了 token 压缩和汇总设计，效果依赖这些摘要仍保留对动作重要的视觉信息。
#### 实验分析（精简版）
在 LIBERO 三个种子上，平均成功率从 96.72% 提至 97.80%，去噪延迟降 64.8%、端到端延迟降 52.9%。UR5 两个实机任务成功率分别提升 8.7 与 5.8 个百分点，去噪延迟下降 58.8%。结果支持该替换在已测平台有效；实机任务只有两项，不能据此推断更广泛机器人或低功耗边缘设备上的收益。
#### 实用指南
适合动作去噪占时延较高、且有能力微调完整策略的 flow-matching VLA。复现时需对齐 π0.5 动作块、流时间采样、视觉 token 压缩策略和计时口径，并同时报告端到端延迟与成功率。论文报告单 H100 训练设定；尚不能把结果解释为 Raspberry Pi 或 Jetson 的部署验证。代码链接需以论文项目页当前状态核实。
结果只说明在所测数据和 UR5 配置下有效；不同控制频率、机械臂自由度与图像分辨率都可能改变延迟和成功率。
复现还应区分仿真去噪时延和真实控制环路耗时，因为相机前处理、通信和动作执行不会随 action expert 等比例减少。
#### 总结
核心思想：压缩动作专家而保留VLA主干
1. 编码多视角图像与指令，压缩视觉上下文。
2. 将带噪动作 token 与流时间送入轻量门控 Mamba。
3. 预测 flow-matching 速度并迭代去噪输出动作块。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.22951v1)
- [arXiv](https://arxiv.org/abs/2610.22951v1)

---

<a id='2610.23098v1'></a>
## [InstAgent: A Recursively Self-Improving Agent Operating System for Heterogeneous Robot Collaboration](https://arxiv.org/abs/2610.23098v1)

**Authors:** Haoyang Luo, Junhao Qin, Jingyan Cao, Yi Huang, Guangjie Yang, Xiaoyu Chen, Haotao Lu, Tianxiang Gui, Ye Shi, Jingya Wang

**Published:** 2026-10-16

**Categories:** cs.RO

**Abstract:**

Heterogeneous robot teams enable complex, long-horizon tasks by combining complementary sensing, mobility, and manipulation capabilities across multiple embodiments. Yet existing systems struggle to retain execution-time corrections as reusable experience, maintain task-relevant context memory across space and time for reliable handoffs, and explicitly represent the inter-robot dependencies required for structured collaboration. In order to address these challenges, we present InstAgent, a recursively self-improving (RSI) Agent Operating System for heterogeneous robot collaboration. InstAgent organizes team behaviors as agentic behavior trees (BTs) that explicitly model capability assignment, inter-robot dependencies and handoff checks. The RSI mechanism updates these collaboration programs from execution feedback and preserves successful revisions for reuse in future tasks, while shared multimodal spatio-temporal memory provides context for planning and handoff validation. Experiments on 20 real-world multi-robot tasks with three high-level models show consistent gains in task success, first-attempt completion, and execution efficiency.

### 论文解读
#### 摘要翻译
InstAgent 面向由不同型号机器人组成的团队，提出以行为树为共同任务表示的机器人 Agent OS。系统把技能检索、协作程序组合、执行反馈与共享视觉记忆连成闭环，让机器人能沿用经过验证的协作步骤，并在失败后修订。
#### 方法动机分析
异构机器人通常各自有不同控制接口和技能，单一联合策略难以复用，也难检查交接条件。只让语言模型即时分配任务，则容易遗漏先后依赖与失败恢复。作者假设将程序结构显式化为行为树，可以兼顾模块化、可审查和跨任务复用；执行反馈再用于局部修复，而不是每次从头规划。
#### 方法设计详解
系统接收自然语言目标、团队成员能力和现场观测。先从技能库选择各机器人已有的行为树片段；HeteroComposer 根据经验文档中的目标范围与适用条件筛选片段，再把它们绑定到具体机器人和协作角色。组合树用顺序、条件、并行等节点表达依赖，并加入结果检查与恢复分支。执行器运行程序并记录成功、失败交接或目标未达等信息。Collab-RSI 结合当前树、观测及相关经验定位故障步骤，调整子树、条件或恢复逻辑，经过结构检查后重新执行；验证通过的程序和上下文一起写回经验库。SMAR-T 将多机器人看到的图像、描述与位置等信息组织成可检索的共享空间记忆。实机使用 G1、Go2、ALOHA 三类平台。
#### 方法对比分析
与集中式联合策略相比，InstAgent 保留机器人各自技能，只在任务层组合程序，因此更容易更换单个组件。与纯语言规划相比，行为树明确表示前置条件和恢复分支；与固定模板相比，检索和复盘让程序可以按任务修改。系统仍依赖手工整理的技能库、能力描述及执行校验，不能假设生成的每个新程序天然正确。
#### 实验分析（精简版）
在20项任务、三种模型和480次运行中，完整系统相对 E0 的任务成功率分别提高15、22.5、25个百分点；记忆问答 RAVEN-QA 得分为93.14，高于 RAVEN 的84.57与 ReMEmbR 的64.49。实机覆盖固定机器人组合和预设任务，结果说明闭环组合与恢复有帮助，尚不能证明动态更换团队或开放环境中的长期泛化。
#### 实用指南
适合已有技能接口、需要可解释跨机器人编排的实验室。复现时应先定义每个技能的输入输出、前置条件、失败信号和停止条件，再评估检索错误如何影响行为树。论文报告的是特定20任务集与固定团队配置；部署到移动服务机器人需重新检查时序同步、共享地图坐标和安全停机逻辑。公开代码、模型的可用状态应以项目页为准。
#### 总结
核心思想：让协作程序可复用且可修复
1. 检索适用技能与历史协作树。
2. 组合带依赖和恢复条件的团队行为树。
3. 执行、诊断、修订并保存验证后的程序。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.23098v1)
- [arXiv](https://arxiv.org/abs/2610.23098v1)

---

<a id='2610.22900v1'></a>
## [Structured Perception Representations for Off-road Dynamics Learning](https://arxiv.org/abs/2610.22900v1)

**Authors:** Micah Nye, Samuel Triest, Matthew Sivaprakasam, Isaiah Adu, Sebastian Scherer, Wenshan Wang

**Published:** 2026-10-16

**Categories:** cs.RO

**Abstract:**

Autonomously navigating off-road terrain at high speed demands accurate control, which often relies on dynamics models that capture complex interactions between the robot and terrain. In off-road regimes, terrain interaction is a major component of the vehicle's dynamics. Prior work attempts to capture this effect by training learned models of the vehicle's dynamics, conditioned on its perception of the local environment. However, said conditioning is often overly simplistic, ignoring the spatial structure of the dynamics problem. Our approach, SpatialBEV, exploits this spatial structure. At each step, we query the bird's-eye-view (BEV) features around the predicted state, conditioning the next prediction on current geometric and semantic cues. We additionally present a benchmarking approach that mines a dataset's most difficult scenarios. On datasets collected across seasons, SpatialBEV reduces prediction error by 17.1%, compared to physics-based and learned baselines. We further evaluate on hardware on high-speed ($\geq$10 m/s) and geometrically challenging courses with a 25.3% reduction in maximum cross-track error. More details can be found at https://theairlab.org/spatial-bev-dynamics .

### 论文解读
#### 摘要翻译
SpatialBEV 研究越野车辆如何根据即将驶过的地形预测自身运动。作者将结构化鸟瞰图作为动力学模型的空间条件，使模型在预测每一步状态时查询预测位置附近的地形，而非只从过去动作和状态外推。
#### 方法动机分析
山坡、松软地面和冰雪会改变轮胎与地面的作用，车辆当前速度相同也可能产生不同轨迹。仅靠里程计历史的网络容易把未经历过的地形差异当成噪声。论文的核心假设是，将几何和视觉语义地图与状态序列结合，模型就能在驶入变化地形前修正未来状态预测。这个假设依赖地图与车辆坐标准确对齐。
#### 方法设计详解
输入包括车辆历史状态、动作以及 BEV 地图。状态和动作先进入 context RNN，形成包含运动历史的隐状态；predictor RNN 据此逐步生成未来状态。每次预测时，系统把轮胎轨迹点转换为 BEV 上的接触 footprint，在局部地图采样特征并送入空间查询模块。查询结果不仅参与下一状态预测，也参与对预测状态的校正，随后继续驱动下一预测步。几何表示保留道路高度、坡度等结构，视觉特征编码难以由几何直接描述的表面信息。论文预测未来5秒，控制更新频率为10Hz，并比较稀疏/稠密 BEV 与不同查询 footprint。
#### 方法对比分析
与不使用地图的状态模型相比，它显式观察未来接触区域；与整图 BEV 编码相比，足迹查询将计算集中于当前预测轨迹可能经过的地面。它适用于地形对车辆动力学影响明显、且地图可以可靠构建的场景。查询、插值和状态校正带来额外开销，地图错误也可能直接影响预测。
#### 实验分析（精简版）
数据含79.8k训练样本和75.3k留出样本，测试集约34%为冰雪条件。全测试集上 dense SpatialBEV 相比最佳基线预测误差降低17.1%。Yamaha Viking 的高速实车试验最大横向误差降低25.3%，但高速课程仅一次，地形课程四次，实机统计仍有限。
硬件课程结果还包括平均速度、最大速度、横向误差和人工干预次数；作者指出仅一趟高速试验不足以估计稳定性，且系统地图查询会增加在线计算量。
#### 实用指南
适合越野车辆、地面移动机器人和感知驱动动力学研究。复现重点是 BEV 与底盘坐标标定、接触 footprint 构造、地图查询范围以及离线与闭环指标的区分。部署需报告查询造成的额外延迟；作者指出全组件运行的推理耗时会随粒子数增加，长视野和高频控制之间需权衡。迁移到轮式机器人应重新采集轮胎、载荷和地面条件数据。
硬件结果以跟踪误差和干预记录为主，尚缺更大规模多季节闭环统计；仿真及离线改善只能作为进一步实车验证的依据。
#### 总结
核心思想：按未来足迹查询地形条件
1. 编码车辆状态动作历史。
2. 沿预测轨迹查询 BEV 接触区域。
3. 用地形特征逐步修正未来状态。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.22900v1)
- [arXiv](https://arxiv.org/abs/2610.22900v1)

---

<a id='2610.23262v1'></a>
## [Expand-and-Prune: Test-Time Candidate-Space Adaptation for End-to-End Autonomous Driving](https://arxiv.org/abs/2610.23262v1)

**Authors:** Fabian Schmidt, Karol Fedurko, Markus Enzweiler, Abhinav Valada

**Published:** 2026-10-16

**Categories:** cs.RO, cs.CV

**Abstract:**

Candidate-based end-to-end driving planners reason over a finite set of trajectory hypotheses. However, for sparse planners, a limited candidate set may provide insufficient support for scene-relevant maneuvers, while fixed downstream candidate budgets do not account for prediction uncertainty. To address these challenges, we introduce Expand-and-Prune, a test-time framework that jointly adapts candidate support and capacity without updating learned parameters. The framework expands the native candidate set using semantic driving intent and temporally propagated predictions and then uses uncertainty in intermediate trajectory scores to adaptively prune the resulting candidates. Across four heterogeneous candidate-based planners and four benchmarks, Expand-and-Prune consistently improves planning performance. It achieves state-of-the-art results on LangAuto, increasing the Driving Score from 68.2 to 75.9, while providing further improvement on Bench2Drive, Fail2Drive, and NAVSIMv2, with particularly strong improvements in out-of-distribution scenarios. Moreover, in its full instantiation, 72.5% of selected trajectories originate from the newly introduced intent-guided or temporal candidates, confirming that candidate-space expansion provides useful hypotheses beyond those generated natively by the planner. The code is available at https://github.com/iis-esslingen/Expand-and-Prune.

### 论文解读
#### 摘要翻译
Expand-and-Prune 是一个可接在已有候选轨迹规划器前后的测试时模块。它先用当前驾驶意图和前一时刻轨迹补充候选，再根据规划器中间评分的不确定度动态删减候选，让后续规划阶段在有限计算预算下看到更相关的轨迹。
#### 方法动机分析
候选式规划依赖一组轨迹假设。固定候选既可能漏掉与当前转向意图匹配的解，也可能在长路线中缺乏时间连续性；无差别扩大候选数则增加下游评估耗时。作者的核心假设是，规划器的动作置信度可指导候选扩展，而候选分数熵能估计当前场景下哪些轨迹需要保留更多计算。
#### 方法设计详解
系统从传感器、导航指令和上游骨干取得场景特征与动作意图。针对 LAD-Drive，离线将轨迹按六类动作聚类为 prototype banks，推理时根据动作概率抽取意图候选；同时缓存上一帧最终候选形成时间候选。新增与原始轨迹集合并后进入冻结规划器的第一阶段。若该阶段给出每条轨迹的评分，方法根据评分分布的熵决定保留数量：分布不确定时留更多候选，较确定时裁剪更多。剩余候选继续通过后续评分和精修阶段，最终输出轨迹。整个过程不微调上游或规划器参数。时间记忆只用上一帧，扩展实现主要针对 LAD-Drive。
#### 方法对比分析
与固定大小候选库相比，它补充当前动作语义和近期轨迹先验；与固定 Top-K 相比，它用评分熵按场景调整候选数。相比重新训练规划器，测试时处理便于接入冻结模型；但质量高度依赖候选原型覆盖、动作意图准确率和评分校准。
#### 实验分析（精简版）
LangAuto 上平均 Driving Score 从 LAD-Drive 的68.2提高到75.9；Fail2Drive 上 BridgeDrive 的 harmonic mean 从76.4升到80.5。LAD head 延迟由6.0ms增至8.2ms，端到端从45.0ms增至47.2ms。评估以 CARLA 仿真为主，NAVSIMv2 是记录数据评测，不能视为真实道路闭环部署。
论文的候选来源统计显示最终被选轨迹中72.5%来自新增候选，表明扩展改变了搜索支持；收益以平均分和有限仿真路线为主，未提供实车碰撞率或安全接管数据。
#### 实用指南
适合已有多阶段候选规划器、且中间阶段能输出候选评分的团队。复现时应报告候选来源占比、熵阈值、各阶段预算和端到端延迟，并检查上游置信度失准时是否会错误扩展。论文未展示真实车辆测试；迁移到实车需增加时序错位、突发障碍和闭环安全评估。代码链接状态应以项目仓库为准。
候选扩展与剪枝的收益需结合闭环成功率、失败类型和最坏推理耗时一起看，平均分数不能反映突发障碍下的安全边界。
#### 总结
核心思想：按规划不确定度分配候选预算
1. 从动作意图和上一帧轨迹扩展候选。
2. 冻结规划器打分并估计分布熵。
3. 熵控剪枝后继续精修并输出轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.23262v1)
- [arXiv](https://arxiv.org/abs/2610.23262v1)

---

<a id='2610.23250v1'></a>
## [LonGaussian: Tracking Foundation Priors for Long-Context Feedforward Gaussian Reconstruction](https://arxiv.org/abs/2610.23250v1)

**Authors:** Zhoufeng Xie, Guanjun Wu, Shuai Zhang, Xinggang Wang, Wenyu Liu

**Published:** 2026-10-16

**Categories:** cs.CV

**Abstract:**

Reconstructing high-quality, renderable 3D scenes from long videos is essential for robotic perception and embodied intelligence. Many feed-forward Gaussian reconstruction methods rapidly predict 3D scenes from a small set of images, but extending them to longer sequences presents two bottlenecks: jointly processing all views increases GPU memory requirements, while accumulating pixel-aligned predictions redundantly represents the same surfaces and increases storage and rendering costs. We present LonGaussian, a streaming feed-forward framework that reconstructs compact, globally renderable Gaussian scenes from long image sequences with known intrinsics and unknown camera poses. Built on a pretrained streaming 3D reconstruction backbone, LonGaussian decodes long-context geometric features into dense Gaussians. Within each chunk, multi-frame point tracks link repeated observations of the same surface and guide a learned merger to consolidate the corresponding Gaussian predictions. Rendered-depth consistency filtering further suppresses Gaussians from other chunks that incorrectly occlude locally reconstructed surfaces. On DL3DV and RealEstate10K benchmarks, LonGaussian improves novel-view rendering quality while using substantially fewer Gaussians than the evaluated feed-forward baselines.

### 论文解读
#### 摘要翻译
LonGaussian 面向超长单目视频重建，目标是用可控数量的 3D Gaussian 表示整个场景。它在冻结的长上下文几何重建骨干上增加块内高斯生成、跨帧对应引导的高斯合并和全局渲染监督，以减少重复表示并保持长序列下的重建质量。
#### 方法动机分析
逐帧预测的高斯若直接累加，会把同一表面反复建模，导致显存、渲染和优化成本不断增加；简单按固定距离切块又可能让块间表面错位。作者假设利用长上下文骨干估计的相机运动和跟踪对应关系，可以先在块内合并重复点，再把各块拼入共享场景坐标。
#### 方法设计详解
输入为相机内参已知、相机位姿未知的 RGB 序列。LoGeR 冻结骨干以重叠图像块处理长序列，并将局部特征、深度和位姿信息传给高斯生成头。AllTracker 为相邻帧提供密集轨迹对应；模型按对应关系合并同一表面的高斯，以减少冗余。随后进行跨块深度一致性筛选，并用远场天空壳处理天空区域。块级全局高斯进入同一坐标系，渲染到训练视角，与输入图像计算监督信号。核心是让跟踪对应不止用于估姿，也直接约束三维表示压缩。实验使用 DL3DV、RealEstate10K，输入分辨率约280×490，训练在24GB RTX 3090上进行。
#### 方法对比分析
与每帧独立高斯预测相比，LonGaussian强调跨帧合并；与逐块重建再后处理相比，它把重叠上下文、跟踪和场景级渲染放进一个连续管线。适合静态场景、内参已知且图像跟踪稳定的长序列。动态物体和大范围闭环误差会破坏对应关系，也会造成错误合并。
#### 实验分析（精简版）
DL3DV 的96帧设置中，PSNR 为18.88，高于 OF3GS 的15.31和适配 ReCoSplat 的15.74；平均高斯数3.54M，低于两者的10.45M与13.17M。单场景延长至3768帧时显存约20.05GB、耗时1445秒，但作者展示的长序列仍有全局闭环重影，不能把局部质量保持等同于全局一致。
结果采用已知内参、未知位姿的静态视频重建设置；PSNR/SSIM 与高斯数量衡量视图质量和压缩，但无法直接代表动态物体处理或在线地图更新能力。
#### 实用指南
适合长视频静态场景重建和显存受限的研究者。复现时需固定输入分辨率、块长、重叠帧数、相机内参和显存预算，并将OOM基线与可运行基线分开比较。论文使用 RTX 3090；长序列耗时仍高，不能据此推断在线机器人算力可用。代码释出状态需查项目页。
作者还展示了更长序列上的显存与时间扩展，但达到数千帧仍需数十分钟并保留明显闭环伪影，说明压缩没有消除全局漂移。该方法面向静态场景，不应直接用于动态视频理解。
#### 总结
核心思想：跟踪对应驱动跨块高斯去重
1. 长上下文骨干分块估计几何与相机位姿。
2. 用帧间轨迹对应生成并合并高斯。
3. 汇总到全局场景并以图像渲染监督。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.23250v1)
- [arXiv](https://arxiv.org/abs/2610.23250v1)

---

<a id='2610.23068v1'></a>
## [Foundation Models for Autonomous Space Robotics: Architectures, Physical Adaptation, and Validation](https://arxiv.org/abs/2610.23068v1)

**Authors:** Zizhe Wang, Jia Deng, Xin Lu, Zehao Jia, Yi Huang, Yao Zhu, Jianing Li

**Published:** 2026-10-16

**Categories:** cs.RO

**Abstract:**

Space robots must interpret tasks and act despite unfamiliar dynamics, difficult sensing conditions, and limited opportunities for human intervention. Foundation models offer reusable perceptual and behavioral knowledge, but their value depends on how that knowledge reaches physical execution. This critical review follows space-specific requirements through perception, prediction, planning, and action to deployment and validation, emphasizing orbital manipulation and proximity operations. Terrestrial comparisons establish benefits of particular pretraining stages and model interfaces; their transfer to space dynamics and constrained onboard operation remains insufficiently tested. Event cameras offer timely motion observations across strong illumination contrasts, while spiking neural networks provide possible routes to sparse local computation. We examine how sensor degradation, state estimation and computing choices affect feedback and recovery. Higher action throughput need not yield fresher feedback; lower processor power can also coexist with higher energy per inference. We organize validation as an iterative sim-to-ground-to-space pathway, connecting physical tests with telemetry-based diagnosis and closed-loop revalidation. A satellite-capture example links observations to approach, contact and recovery, while a corresponding protocol separates generalization from adaptation with limited target-domain interaction. These comparisons identify what models already contribute and the physical evidence needed to support autonomous space operation.

### 论文解读
#### 摘要翻译
本文是一篇关于自主空间机器人基础模型的综述。作者按感知、预测和动作整理视觉语言模型、VLA、事件传感与脉冲神经网络相关工作，并讨论地面模型怎样经过仿真、地面类比硬件和空间任务逐步验证。文章强调，空间环境会改变动力学、接触、通信、能耗与传感时序。
#### 方法动机分析
空间机器人挑战不止是模型是否能识别物体，还包括观测是否及时、接触是否稳定、推理能否持续运行。综述建议把这些条件放进复现记录和验证设计。本文不训练新模型，也没有统一推理设定；比较对象来自不同论文和硬件。许多机器人基础模型在地面数据和模拟器中表现良好，但太空任务存在微重力、辐射、狭窄能源预算、通信延迟和稀疏触觉反馈。直接将地面成功率外推为空间可用性，会混淆表示复用和目标环境适应。作者用跨域证据整理来提出核心问题：哪些能力可由预训练迁移，哪些必须通过目标物理环境标定或任务级验证。
#### 方法设计详解
这篇文章没有提出可训练的新网络或单一算法。作者先按 sensing、prediction、action 梳理基础模型使用方式，再区分仿真、地面硬件类比、在轨或行星任务的证据等级。随后对代表性工作检查初始化来源、是否更新目标模型、测试硬件、任务类型和比较条件，避免把同名模型在不同设置下直接排名。讨论涵盖视觉与事件传感、空间定位和交互预测、VLA 执行、神经形态计算等。最后提出 sim-to-ground-to-space 的验证路线：在仿真中覆盖轨迹与恢复，在平面气浮或重力卸载装置检查选定运动和接触，再逐步进入真实空间任务；每次模型或控制器更新都要用闭环测试重新验证。
#### 方法对比分析
相比按模型规模或任务名称编排的综述，本文更重视证据处于什么硬件环境、有没有目标域更新以及测量了什么指标。它贡献的是比较框架和研究议程，而不是新的性能算法。适合制定跨域验证计划；不能把表格中的异构实验数值当成统一基准。
#### 实验分析（精简版）
本文未进行新的系统实验，数字均来自所引用论文。例如一项 V-JEPA 工作报告 RTX4090 动作推理约16秒；另一项 SNN 研究报告每帧57.4mJ，对照 Jetson 7.63mJ，但吞吐和模型能力并不相同。引用的 Spike-driven VLA 报告 ANN/SNN 成功率96.3%/92.4%，并估计每 chunk 能耗179.7/20.9mJ。它们只能说明各自条件下的测量，不能视作本综述验证或横向排名。
#### 实用指南
后续迁移到长期机器人时应将记忆读写和策略更新分开记录，核对不同论文的硬件与任务边界。本文未训练新模型，也没有统一推理设定。适合空间机器人、在轨操作和具身模型评估团队。可把文章的分层验证思路转成记录表：任务阶段、硬件环境、传感器时延、接触动力学、算力与功耗、目标域更新和失败恢复。迁移到水下、极地等环境也可沿用“模拟到受控硬件再到目标域”的证据链，但每种环境仍需单独标定。该综述截至2026年10月14日的文献范围，后续工作可能补充其中空白。
#### 总结
核心思想：用逐层证据验证空间迁移
1. 明确感知、预测或动作任务与环境差异。
2. 通过模拟和地面类比硬件校准动力学。
3. 在目标空间任务逐步验证并持续闭环复测。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.23068v1)
- [arXiv](https://arxiv.org/abs/2610.23068v1)

---

<a id='2610.22765v1'></a>
## [Pedicel-Targeted Strawberry Harvesting via Imitation Learning on a Low-Cost Teleoperated Robotic Arm](https://arxiv.org/abs/2610.22765v1)

**Authors:** Nitin Rai, Won Suk Lee, Hongyoung Leon, Heping Zhu

**Published:** 2026-10-16

**Categories:** cs.RO

**Abstract:**

Precise pedicel-targeted grasping in strawberry harvesting remains a critical bottleneck due to the pedicel's small diameter (1.3-3 mm), color similarity with surrounding foliage, and precise force control required for clean abscission zone detachment. As a market-grade crop, strawberry aesthetic value makes bruise-free pedicel detachment essential over direct fruit grasping. Classical approaches struggle to unify vision models, motion planning, and contact manipulation into a robust framework, particularly in occluded canopy environments. Therefore, this work presents a visuomotor imitation learning pipeline for pedicel-targeted strawberry harvesting that unifies perception, motion planning, and contact manipulation from collecting human teleoperation demonstrations, training an Action Chunking with Transformers (ACT) policy with a ResNet-50 visual backbone, and deploying the learned policy autonomously on a Jetson AGX Orin edge platform at 15 Hz. The trained policy achieved consistent loss convergence from 2.84 to 0.091 over 100K gradient steps on 130 demonstrations. We have also established a reproducible open-source pipeline with publicly released dataset and model weights for the agricultural robotics community. Full dataset is available at, https://huggingface.co/datasets/nitindominicrai/strawberry_pedicel_grasp_teleoperation and trained policy at, https://huggingface.co/nitindominicrai/act_strawberry_pedicel. To replicate this work, all code, configurations, and deployment instructions are publicly available at, https://github.com/nitin-dominic/autonomous_strawberry_pedicel_harvesting.

### 论文解读
#### 摘要翻译
论文研究如何让低成本双臂机器人对准被叶片遮挡的草莓果梗。作者收集人类双臂遥操作示范，以 ACT 行为克隆训练视觉动作策略，并把策略部署到 Jetson AGX Orin。演示结果主要表明机器人能靠近、对准并接触目标。
#### 方法动机分析
草莓采摘需要对准毫米级果梗，叶片遮挡、颜色相近和果实易损使直接抓果不可行。传统脚本难以覆盖每种姿态，作者假设示教轨迹可以教会机器人稳定的接近和对准步骤。不过，采摘还要求专用末端执行器完成切断或扭转，因此对准成功不能直接等同于收获成功。
#### 方法设计详解
遥操作员用双臂引导从臂记录动作，系统同步保存腕部相机视频与六自由度关节状态，形成约130回合、25万帧数据。策略采用 ACT 行为克隆：腕部图像经 ResNet-50 提取特征，与关节状态组合后输入 Transformer，输出一段动作块而非单步动作。训练用位置误差和潜变量正则项优化动作预测。部署时，Orin 接收实时腕部图像并滚动执行预测动作块，机器人按接近、对准、夹爪闭合、回撤顺序完成演示。实验集中在学习曲线和部署关键帧，没有对真实果梗脱离进行成功率评估。
#### 方法对比分析
相比手工轨迹脚本，示教策略能从多回合姿态变化中学习连续动作；相比端到端高成本大模型，ACT 与 ResNet-50 更易在边缘设备运行。方法仍是单一工作台和固定示教数据上的行为克隆，缺少在线纠错和真正切梗工具。
#### 实验分析（精简版）
作者报告训练损失由2.84降至约0.091，使用单张 NVIDIA B200，训练步数在论文不同位置报告为100k、150k或160k；Orin 推理频率也出现15–17Hz与25Hz的不同报告。更关键的是没有专用末端执行器，论文没有展示果梗切断或采摘成功率，因此结果只证明接近和接触轨迹的复现。
论文没有报告果实脱离成功率，关键帧只显示夹爪到达目标附近并闭合回撤；因此损失下降和推理帧率仅能说明训练收敛及执行频率。
#### 实用指南
适合示教学习、精细视觉伺服和边缘机器人推理研究。复现应先核对论文各处不同的步数与频率口径，再记录视频帧率、动作块长度、相机位置和执行器几何。若目标是真实采摘，需要将切断动作和果实脱离纳入数据、末端执行器设计与成功判定，不能只以轨迹看起来接近为指标。
因此应把“学会接近”与“成功采下果实”分别定义，后续系统需报告成功脱离、碰伤率、误采率和跨植株条件的统计结果。
#### 总结
核心思想：示教学习果梗接近动作
1. 双臂遥操作采集图像与关节序列。
2. ACT 根据视觉和本体状态预测动作块。
3. Orin 执行接近、对准和接触轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.22765v1)
- [arXiv](https://arxiv.org/abs/2610.22765v1)

---

<a id='2610.23018v1'></a>
## [VLA-Fabric: Learning Interaction for Composable Multi-Arm VLA Agents](https://arxiv.org/abs/2610.23018v1)

**Authors:** Chenglong Zhang, Shuaijun Liu, Feiyang You, Shuyang Hao, Chengyu Wu, Ningxin Su

**Published:** 2026-10-16

**Categories:** cs.RO

**Abstract:**

Vision-language-action (VLA) models are rapidly expanding what individual robots can accomplish, yet many real-world manipulation settings require multiple arms to operate as a coordinated system. Existing approaches organize multi-arm coordination in different ways, including centralized joint policies, high-level task or role assignment, and locally executed policies that react to teammates. However, how complete action-producing VLAs should explicitly interact with one another during action generation remains largely unexplored. We introduce VLA-Fabric, which organizes one complete VLA per arm while preserving local perception and action ownership, and enables coordination through learned interaction across policy boundaries. We first systematically study where such interaction should occur, how peer information should be aggregated, and how the interaction should be learned, revealing complementary coordination functions spanning shared context, peer-specific latent information, and action-stage interaction. We then re-instantiate these functions in a structurally different VLA architecture, showing that the coordination principle is not tied to a particular tensor layout or backbone implementation. Finally, we compose teams of three and four arms and evaluate them in simulation and physical experiments. On all four physical tasks, Full interaction improves success over independently trained local policies by 16--40 percentage points. Together, these results suggest that explicit interaction among complete VLA policies offers a scalable path toward networked embodied intelligence.

### 论文解读
#### 摘要翻译
VLA-Fabric 让多臂团队中的每个机械臂继续运行完整的本地 VLA，同时学习如何交换少量协调信息。作者研究交互发生位置、信息聚合方式和策略适配，再把得到的交互原则迁移到不同 VLA 架构，并扩展到三臂、四臂仿真及真实多臂任务。
#### 方法动机分析
把所有机械臂合成一个联合策略，随着角色和数量增长会降低模块性；完全独立的策略又难处理交接、共持物体等耦合动作。作者提出本地策略应保留自己的感知和动作所有权，但通过明确的中间表示读取同伴状态。关键假设是交互结构和适配方式比单纯增加共享信息更重要，交互模块也必须适配底层 VLA 的计算边界。
#### 方法设计详解
方法设计详解：这套模块流程的关键是同伴表示在不同架构中进入不同层级，不能只把消息拼接到同一输入模板。在 SingleVLA 阶段，作者从双臂任务中拆出原跨臂依赖，系统搜索交互入口、聚合算子和微调策略。最终交互包括 shared context、peer-specific key/value 表示和 action intent：共享语义提供任务公共信息，peer 表示携带同伴状态，action intent 让当前机械臂动作条件化于同伴动作侧状态。每个本地 VLA 仍生成自己的动作。迁移到 π0.5 后，作者把公共与私有信息放进其多模态前缀融合边界，把动作意图放到迭代 Action Expert 中；没有改为一个统一全局动作头。之后将相同交互原则扩展到三臂和四臂团队，评测仿真与实体任务。实机每任务收集50条示范，每种策略配置运行25次。
#### 方法对比分析
与集中联合策略相比，VLA-Fabric 保留各臂策略边界；与只分配高层角色相比，它让同伴表示进入策略内部计算。它还展示了跨 Eagle2 SingleVLA 与 π0.5 的架构适配，但不是即插即用协议：不同骨干需要重新选择交互入口和张量含义。当前实验使用同构机器人和固定任务角色。
#### 实验分析（精简版）
方法设计详解：这套模块流程的关键是同伴表示在不同架构中进入不同层级，不能只把消息拼接到同一输入模板。实体任务中 Full 相比 Independent 的成功率：Basket3 为72%→88%，Relay3 为16%→56%，Relay4 为40%→72%，Frame4 为36%→60%。提升幅度16–40个百分点，较强交接和共同控制任务获益更多；长时高精度装配仍困难。证据上限是四个实体任务、每种设置25次试验和最多四臂团队。
#### 实用指南
适合多臂操作、VLA 协同和跨架构迁移研究。复现需保持任务示范、交互特征维度、动作时序和试验次数可比，并分别报告训练与推理时交换的信息。迁移到异构机械臂时要先校准共享语义和 peer 特征含义、通信延迟及角色变化；论文没有验证异构团队和动态成员加入。代码与项目链接见论文主页，实际可用状态应核验。
#### 总结
核心思想：本地策略协作但保留动作边界
1. 搜索共享、同伴和动作意图交互位置。
2. 按底层 VLA 架构重新实现表示交换。
3. 扩展团队规模并在实体任务验证协同收益。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.23018v1)
- [arXiv](https://arxiv.org/abs/2610.23018v1)

---

<a id='2610.22795v1'></a>
## [STRIDE: Evaluating Spatiotemporal Reasoning in Driving Edge Cases](https://arxiv.org/abs/2610.22795v1)

**Authors:** Lieqi Liu, Rui gao, Jia-Chen Gu, Wenbo Hu, Zhaobin Mo, Ahmadreza Moradipari, Nejib Ammar, Wei Wang, Nanyun Peng

**Published:** 2026-10-16

**Categories:** cs.CV, cs.AI

**Abstract:**

In autonomous driving, perceiving surrounding objects is not enough: failing to understand how they move and interact over time can lead to dangerous decisions. Evaluating vision-language models (VLMs) therefore requires testing whether they connect past spatial observations to future consequences. Yet existing benchmarks often assess edge cases through basic perception questions or test temporal understanding in simple scenarios, leaving complex spatiotemporal reasoning underexplored. We introduce STRIDE (SpatioTemporal Reasoning In Driving Edge Cases), an object-centric benchmark for long-tail, safety-critical driving scenarios. STRIDE covers six task families: spatial perception, spatial understanding, temporal memory, temporal extrapolation, trajectory prediction, and scene-context awareness. It pairs multiple-choice questions probing individual spatiotemporal judgments with related free-form visual question answering (VQA) requiring detailed scene explanations and justifications of driving decisions. We prioritize scenes with unusual object behaviors and complex traffic interactions, generate reference answers from structured dataset annotations through automated procedures, and use limited language-model assistance and human verification to ensure annotation quality. Across 12 proprietary, open-weight, and driving-specialized models, even the best performer, GPT-6-Astra, achieves only 43.4% average multiple-choice accuracy, versus an approximately 20% random-choice baseline. Further analysis reveals that strong performance in describing driving scenes does not necessarily translate to accurate safety-critical judgments, and that models struggle more with reconstructing object motion than recalling past positions. STRIDE thus exposes spatiotemporal driving limitations that aggregate VQA scores can conceal.

### 论文解读
#### 摘要翻译
STRIDE 是面向自动驾驶视觉语言模型的时空推理评测基准。它从道路长尾场景构建空间、时间、轨迹与场景理解任务，并结合多项选择题和开放式说明，观察模型能否把前序观察、当前物体关系和未来交通后果联系起来。
#### 方法动机分析
现有驾驶问答容易测到静态识别或泛化的驾驶常识，却未必检验跨帧追踪、风险判断和可行驶空间。作者假设把问题拆成对象级中间判断和场景级解释，可以区分“会描述”与“判断正确”。该基准的作用是诊断能力缺口，不是直接测试封闭道路驾驶安全。
#### 方法设计详解
基准来自 nuScenes 与 Waymo E2E，使用连续五帧、每帧组合周围摄像头的场景输入，并用查询帧框出目标对象。题目分为空间感知、空间理解、时间记忆、时间外推、轨迹预测与场景上下文六类；部分题目配套 MCQ 和开放回答。nuScenes 利用原有3D目标和轨迹标注生成位置、遮挡与运动标签；Waymo 对前视图像应用 SAM3 检测和启发式跟踪，推导物体运动与距离，再按规则派生风险、碰撞时间等答案。开放回答通过视觉语言裁判按0–10分评分，轨迹回答则解析为坐标并计算 ADE/FDE，未解析成功的输出记作错误。基准包含2,350题、1,452段片段。
#### 方法对比分析
相比普通驾驶 VQA，STRIDE 明确加入长尾交通、跨帧对象追踪、未来事件与轨迹坐标；相比单一总体分数，它把能力拆开并对照解释质量。它的创新在评测组织和诊断，而非提出更强驾驶策略。Waymo 距离是基于检测框底部中心与地面平面的估计，标签可靠性受检测、跟踪和几何假设影响。
#### 实验分析（精简版）
12个模型中 GPT-6-Astra 的选择题准确率最高：nuScenes 为45.5%，Waymo E2E 为41.2%，随机选择基线约19.8%和20.3%。nuScenes 上前一位置回忆可达72%，而最近运动模式只有18%。这些结果表明测试模型仍有明显弱项；不过作者审查时保留31类 MCQ 模板中的16类，并重写了250个开放问题中239个标签，成绩应结合剩余标注噪声理解。
#### 实用指南
适合驾驶 VLM 评测、数据集质量控制和安全关键多模态推理研究。使用时应同时报告 MCQ、开放回答评分、轨迹解析率和数据源；不要把裁判分数或描述流畅度直接等同于真实道路安全。论文未训练新模型，评测协议使用五帧视觉上下文；Waymo 距离估计采用平地近似，迁移到不同视角或道路坡度时需重新检查几何标签。
#### 总结
核心思想：拆开测量驾驶时空判断
1. 采集长尾场景多帧多视角输入。
2. 构造空间、时间、风险和轨迹问题。
3. 分别评估选择、解释与坐标预测。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.22795v1)
- [arXiv](https://arxiv.org/abs/2610.22795v1)

---

<a id='2610.23319v1'></a>
## [From Sliding Windows to Self-evolving Lifelong Memory: A Dual-perspective Survey of Memory Paradigms for Long-Horizon Robotics VLA Models](https://arxiv.org/abs/2610.23319v1)

**Authors:** Yong Wang, Boyu Shen, Xinwei Zhang, Yizheng Li, Keyuan Luo

**Published:** 2026-10-16

**Categories:** cs.RO

**Abstract:**

Long-horizon robotic tasks expose a fundamental limitation of current Vision-Language-Action (VLA) models because their restricted temporal context pre-vents effective accumulation and reuse of past experiences. As a result, memory has become a fundamental component for enabling embodied agents to perform long-horizon reasoning and adaptive decision-making. This survey presents a systematic taxonomy of VLA memory mechanisms based on memory scope and persistence, categorizing existing approaches into five representative paradigms and one emerging frontier. Based on this taxonomy, we compare representative VLA memory systems in terms of memory characteristics, architectural com-plexity, and open-source maturity, revealing a clear trade-off in which increasing architectural complexity is generally accompanied by decreasing open-source ma-turity. Finally, we discuss the key challenges in building full-lifecycle memory systems, including efficient memory management and lifelong knowledge evolu-tion. Overall, this survey provides a unified perspective on memory architectures for VLA agents and offers insights into the development of lifelong embodied in-telligence.

### 论文解读
#### 摘要翻译
这篇综述从记忆跨度和跨任务持久性两个角度梳理 VLA 机器人记忆，归纳六类方式：短窗口、隐式融合、显式检索、训练阶段记忆、部署期持久记忆，以及仍处于前沿的自演化记忆。作者进一步比较记忆表示、检索、更新和整合机制，并讨论长期运行需要解决的问题。
#### 方法动机分析
长期部署的挑战在于经验既要保留可复用信息，也要避免旧记录压制新任务。
长时机器人任务要求记住之前做过什么、失败过什么以及哪些技能可复用。仅增加输入窗口会提高上下文成本，外部记忆则带来检索噪声、更新和遗忘风险。作者希望用统一分类区分不同方案解决的是单回合时间跨度还是跨回合知识积累；核心判断是更持久的记忆需要更完整的生命周期管理，而不只是存下更多历史。
#### 方法设计详解
分类流程先确定记忆范围与持久性，再判断其表示、检索、更新和整合方式。
这是一篇文献综述，没有新模型架构。作者用 Scope 表示可调用经历从短窗口到生命周期的范围，用 Persistence 表示知识能否跨任务保留，再按记忆形式与工程机制整理代表工作。滑动窗口保留最近观察；隐式融合将历史压进潜状态；显式检索从外部情节库按相似度取回经验；训练期方法把经验整合进参数；持久记忆在部署阶段跨回合保存和检索；自演化记忆则希望自动完成表示、检索、更新和知识整合。作者还从速度、遗忘和开源成熟度比较系统，但说明这些评分是相对定性判断，不是统一设备上的绝对测量。
#### 方法对比分析
相较按 VLA 骨干或训练范式分类的综述，本文把记忆作为主轴，突出“能看到多久”和“能否跨回合保存”的差异。它的贡献是概念框架和文献对照，不是证明某一种记忆方案在统一任务上最好。滑窗轻量但忘得快，显式记忆可追溯却需要管理，参数记忆易部署但更新可能遗忘旧技能。
#### 实验分析（精简版）
本文未进行实验，也没有统一量化结果。作者基于代表论文作定性比较，并指出现有长期学习基准多测任务成功率、执行时间或连续训练后的遗忘，较少测记忆保持、检索有效性、存储占用和检索延迟。因而表中的速度、遗忘和开源成熟度分级应视为综述作者的相对判断，而不是可复现的性能排行。
#### 实用指南
本文未训练新模型，也没有统一推理设定；后续迁移时应将记忆读写和策略更新分开记录。
适合设计长期机器人记忆模块或建立评测基准的研究者。文章建议同时记录记忆保持率、检索准确率、存储消耗、推理延迟与新技能整合情况。它没有训练或推理设定；文献范围应结合论文截至2026年10月的参考工作理解，且部分系统开放程度可能随时间变化。将显式检索的可检查性与训练期压缩结合是可研究方向，但需记录更新来源和遗忘回滚。
#### 总结
核心思想：记忆要覆盖完整生命周期
1. 确定记忆范围和跨任务持久性。
2. 选择表示、检索与更新机制。
3. 用保持、检索、成本和遗忘指标评测。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.23319v1)
- [arXiv](https://arxiv.org/abs/2610.23319v1)

---

