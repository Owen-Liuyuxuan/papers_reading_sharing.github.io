time: 20260911

# Arxiv Computer Vision Papers - 2026-09-11

## Executive Summary

## 执行摘要

本期10篇论文集中体现了视觉感知、具身智能与机器人控制的协同演进。研究一方面继续提升三维形变配准、雷达深度估计和街景表示在视觉退化、复杂几何条件下的鲁棒性；另一方面将扩散策略、视觉—语言—动作模型与时序逻辑结合，推动机器人从离线模仿走向闭环、可验证的多步决策。

较具代表性的方向包括：BridgeMatch在匹配矩阵空间中构造条件传输桥，为三维形变配准提供新的生成式求解路径；IMLE-VLA以单步动作生成降低VLA策略推理延迟；Harness Robotic OS把四足机器人巡检抽象为统一的具身智能运行时；LTLDiff将有限线性时序逻辑约束注入多智能体操作的数据生成与策略学习。GRADE、ObstaDiff和Safety-aware Skill Adaptation则分别从恶劣视觉条件、障碍物表示和动态环境安全性改善部署可靠性。

值得关注的新趋势是“结构化中间表示+生成式控制”的组合：语义、深度、雷达或逻辑约束不再只是输入特征，而是参与策略生成、技能切换和失败恢复。同时，FARM探索从冻结世界模型的内部预测状态读取失败信号，ReactHuman则以物理约束基准评估具身多模态模型的类人反应能力，显示评测正在从静态问答转向可交互、可执行和安全相关的行为。

建议优先精读BridgeMatch、IMLE-VLA、Harness Robotic OS与LTLDiff，以理解表示学习、低延迟策略、系统运行时和形式化约束如何贯通机器人闭环；若关注真实部署，可进一步阅读GRADE、ObstaDiff、FARM和Safety-aware Skill Adaptation。

---

## Table of Contents

1. [BridgeMatch: Conditional Transport Bridges in Matching Matrix Space for 3D Deformable Registration](#2609.11472v1)
2. [IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies](#2609.10915v1)
3. [GRADE: Single-Frame Generative Radar Depth Estimation Under Visual Degradation](#2609.10756v1)
4. [Harness Robotic OS: A Unified Embodied-Agent Runtime for Closed-Loop Quadruped Inspection](#2609.11225v1)
5. [Safety-aware Skill Adaptation for Reinforcement Learning in Dynamic Environments](#2609.11433v1)
6. [ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs](#2609.10895v1)
7. [LTLDiff: Finite Linear Temporal Logic-Guided Data Generation and Diffusion Policies for Multi-agent Robotic Manipulation](#2609.11043v1)
8. [ObstaDiff: Generalizable Diffusion Policy Learning via Obstacle-aware Representations](#2609.10918v1)

---

## Papers

<a id='2609.11472v1'></a>
## [BridgeMatch: Conditional Transport Bridges in Matching Matrix Space for 3D Deformable Registration](https://arxiv.org/abs/2609.11472v1)

**Authors:** Qianliang Wu, Haobo Jiang, Guangwei Gao, Shuo Chen, Jin Xie, Jian Yang, Yaqing Ding

**Published:** 2026-09-10

**Categories:** cs.CV

**Abstract:**

Reliable non-rigid point cloud correspondences are important for deformable anatomical registration, embodied perception and manipulation, and dynamic 3D reconstruction. Coarse-to-fine methods reduce computational cost by selecting the top-\(K\) coarse regions. However, this pruning may remove weak but correct hypotheses and restrict fine matching to an incomplete search space. We present \paper, a two-stage generative solver that maintains the complete soft matching matrix at both coarse and high resolutions. Stage~I uses denoising diffusion to estimate a global matching matrix in the compact coarse-resolution space. We then lift this matrix to high resolution while preserving its hierarchy. The lifted matrix is rank-bounded and block-constant. Stage~II refines it through a conditional transport bridge. We implement the bridge with two types of dynamics: a deterministic endpoint-parameterized conditional Flow Matching (CFM) ODE and a stochastic Brownian-bridge SDE inspired by Schrödinger bridges. Both variants share the lifted source, a time-conditioned transformer, and a matching-matrix endpoint predictor. Experiments on 4DMatch and 4DLoMatch show that both variants produce more accurate correspondences than the compared methods and improve downstream registration, with larger gains in low-overlap cases. They also improve cross-dataset generalization on CAPE and DeepDeform without target-domain adaptation while using the same deformation solver.

### 论文解读
#### 摘要翻译
BridgeMatch面向三维可变形点云配准，同时估计点间对应关系与空间变形。它先在紧凑粗分辨率匹配矩阵中扩散搜索，再把结果提升至高分辨率，用条件传输桥精炼；第二阶段既可用确定性ODE，也可用随机布朗桥SDE。全程保留软匹配矩阵，避免Top-K剪枝丢失正确但低置信度的匹配。

#### 方法动机分析
低重叠、大形变和特征歧义时，粗阶段的Top-K并不可靠，一旦删掉正确匹配，精细局部搜索无法恢复。BridgeMatch假设粗层对应可借助点云层次结构提供桥源，而高分辨率真实对应是终点，从而把粗到细过程变成连续校正；代价是全矩阵搜索更耗算力。

#### 方法设计详解
推理阶段需固定采用ODE或SDE，并固定噪声尺度及显式Euler的积分步数或SDE采样步数，才能复现实验比较。
输入为两帧点云及多尺度特征。Stage I在粗矩阵上加高斯噪声，(X_k=\sqrt{\bar\alpha_k}Y+\sqrt{1-\bar\alpha_k}\epsilon)，时间条件Transformer预测干净矩阵。随后用源、目标父节点分配矩阵做提升：(X_0=P_s\hat YP_t^\top)，将粗匹配复制到对应的细项对，形成块恒定、低秩但覆盖完整空间的桥源。Stage II沿((1-t)X_0+tY^{hr}+\rho(t)\epsilon)演化；ODE令噪声为零，SDE令(
ho=\sigma_B\sqrt{t(1-t)})。每一步用Soft Procrustes估计刚性变换辅助特征对齐，再由高分辨率特征、当前矩阵和层次信息预测终点。ODE用显式Euler积分；SDE加入低秩噪声预测头并按布朗桥随机采样，最后输出细粒度软匹配。

#### 方法对比分析
与Diff-Reg等只在矩阵上扩散或仍依赖局部候选的方法相比，本文明确连接粗、细分辨率，并取消不可逆Top-K裁剪。ODE适合稳定、可重复的校正，SDE在歧义对应和低重叠场景提供探索能力；但全矩阵表示及特征骨干依赖限制了极大规模或资源受限部署。

#### 实验分析（精简版）
这些比较覆盖对应关系、下游配准和跨域泛化，说明收益并非单一指标偶然波动。
在4DMatch上，随机桥NFMR/IR为92.43/91.47，Diff-Reg为90.25/87.98；在更困难的4DLoMatch上为82.41/78.79，对比77.15/67.00，分别提升5.26和11.79个百分点。下游4DLoMatch-F中EPE由0.095降至0.089；零样本DeepDeform为0.0620，Diff-Reg为0.0748，降低17.1%。消融显示高分辨率精炼和随机桥尤其关键，但论文结果未证明其计算成本低于稀疏方法。

#### 实用指南
复现需准备多尺度特征、父节点分配矩阵和高分辨率对应监督，分别训练粗扩散与条件桥，并区分ODE/SDE评测。论文材料未确认代码或权重已公开，开源状态应以明确链接为准。迁移到新机器人或数据集时，应重建层次聚合、坐标及特征预处理，并通常重训桥模型；全矩阵内存和采样步数是部署重点。还应保持数据划分、特征骨干和后端配准器一致，避免把下游收益误归因于桥模型本身。

#### 总结
核心思想：全矩阵条件桥配准

1. 粗矩阵扩散，保留全局可能性。
2. 层次提升，构造细层桥源。
3. 特征对齐，预测高分辨率终点。
4. ODE确定性积分或SDE随机精炼，输出软对应。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.11472v1)
- [arXiv](https://arxiv.org/abs/2609.11472v1)

---

<a id='2609.10915v1'></a>
## [IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies](https://arxiv.org/abs/2609.10915v1)

**Authors:** Kian Hosseinkhani, Qinhe Peng, George Shramko, Mehran Aghabozorgi, Jianing Qian, Tristan Engst, Alireza Moazeni, Dinesh Jayaraman, Ke Li

**Published:** 2026-09-10

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision-language-action (VLA) policies leverage pretrained vision-language backbones to achieve strong cross-task generalization. A leading design couples this backbone with a dedicated continuous action head trained via diffusion or flow matching. However, such heads rely on iterative multi-step sampling, for example 10 Euler steps in $π_{0.5}$. This creates an inference bottleneck that produces stop-and-go movement in the robot and slower task completion. We introduce IMLE-VLA, which replaces the iterative action head with a single-step conditional generator trained via conditional Implicit Maximum Likelihood Estimation (cIMLE). The cIMLE objective promotes multimodal action coverage, avoiding the mode collapse of naive regression heads while eliminating multi-step sampling entirely. When IMLE-VLA is applied to $π_{0.5}$, it increases inference frequency 3.67x (55 Hz vs. 15 Hz), enabling up to 11x higher action throughput. On the 40-task LIBERO benchmark, IMLE-VLA achieves the highest average success rate (98.0%) among all baselines while leading in inference frequency. Under the test-time perturbations of LIBERO-plus, IMLE-VLA retains $π_{0.5}$'s robustness while other baselines degrade sharply, confirming that the cIMLE head preserves generalization. Real-world experiments on a Franka Emika Panda across four tasks demonstrate smoother motion (2.2x to 3.0x lower jerk) and faster task completion, with IMLE-VLA outperforming $π_{0.5}$ on every task and reducing average VLA inference time per episode by 3.9x to 6.6x. Videos and code are available at https://kianhk6.github.io/IMLE-VLA/

### 论文解读

#### 摘要翻译

IMLE-VLA 面向视觉-语言-动作策略提出快速单步动作生成器。它用条件隐式最大似然估计（cIMLE）训练动作头，在一次前向传播中生成完整动作块，减少扩散或流匹配多步采样造成的延迟，同时保留多模态动作能力。方法在 LIBERO、LIBERO-plus 和 Franka Panda 真实机器人任务上验证了速度、成功率与运动平滑性。

#### 方法动机分析

现有 VLA 常依赖强大的视觉语言骨干，但动作头需要迭代采样；例如 π0.5 使用 10 步 Euler 采样，推理频率约 15 Hz，机器人容易出现停顿。直接单步回归又会把多个合理动作平均化。作者的关键假设是：让不同噪声对应不同候选动作，并只强化最接近示范的候选，就能兼顾单步速度和模式覆盖。

#### 方法设计详解

输入为视觉观测、语言指令和高斯噪声。冻结的 VLM 输出条件表示，轻量生成器将它与噪声映射为动作块 A∈R^(C×D)，一次前向即可完成输出。训练时为每个示范采样 m 个噪声，生成 m 个候选；先按平方 L2 距离选择离真实动作最近的候选，再仅用该候选计算损失并更新。m=1 等价于普通回归，m>1 则允许不同噪声学习不同动作模式。实验使用约 3B 参数规模的 VLM 骨干；执行时界 H=30 是速度与准确率的折中。该设计把候选选择放在训练阶段，把实时控制阶段压缩为单次动作头计算。

#### 方法对比分析

它与扩散/流匹配方法的区别在于直接学习单步映射，而不是减少迭代次数；与确定性回归的区别在于保留噪声输入和最近候选分配，减轻模式崩塌。因而它特别适合需要高控制频率、动作分布多峰且要求连续运动的操纵任务。若主要延迟来自大型 VLM 骨干，动作头加速的收益会受到限制。

#### 实验分析（精简版）

在包含 40 个任务的 LIBERO 上，H=10 时 IMLE-VLA 平均成功率达到 98.0%。其动作生成频率为 55 Hz，相比 π0.5 的约 15 Hz 提升 3.67 倍；H=30 时动作吞吐量最高提升 11.0 倍。真实机器人四项任务均优于 π0.5，proprioceptive jerk 降低 2.2–3.0 倍，每集推理时间减少 3.9–6.6 倍，完成速度约快 2 倍。LIBERO-plus 中它保持了与 π0.5 相当的扰动鲁棒性。局限是仍受约 3B 骨干计算开销影响，真实任务覆盖范围也有限。

#### 实用指南

论文提供项目主页、视频和代码。复现需保持原 VLA 的视觉、语言、动作预处理，加载相同骨干并在动作头注入高斯噪声；建议从 m=2、H=30 起调，再联合观察成功率、吞吐量、端到端时间和 jerk。迁移到新机器人时要改动作维度、尺度化与控制接口，并用新示范重训动作头；若观测分布变化明显，可能还需微调骨干。

#### 总结

核心思想：cIMLE让单步生成覆盖多模态动作。

速记：
1. VLM编码观测与指令。
2. 多噪声单步生成候选动作。
3. 选择最近示范者并更新。
4. 一次前向输出动作块并滚动执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.10915v1)
- [arXiv](https://arxiv.org/abs/2609.10915v1)

---

<a id='2609.10756v1'></a>
## [GRADE: Single-Frame Generative Radar Depth Estimation Under Visual Degradation](https://arxiv.org/abs/2609.10756v1)

**Authors:** Bin Zhao, Patrick Chiou, Nakul Garg

**Published:** 2026-09-09

**Categories:** cs.CV, cs.RO

**Abstract:**

Dense 3D depth perception fails under smoke, fog, and darkness because optical sensors cannot penetrate airborne particulates. mmWave radar remains usable and measures range accurately under these conditions, but its small aperture limits angular resolution. We present GRADE, which grounds a pretrained generative prior in single-frame radar geometry to estimate high-fidelity metric depth. GRADE first maps raw 4D radar spectra to coarse metric depth. A latent diffusion backbone then recovers structural detail while conditioning every denoising step on this estimate. A pixel-space adapter uses residual camera cues when available and is trained across clear, smoke-degraded, and occluded inputs so the full output approaches the radar-conditioned path as visibility degrades. Trained and evaluated on ~95K frames across 12 buildings with real smoke, GRADE achieves an MAE of 0.303 m in clear scenes and 0.313 m under smoke, outperforming existing baselines. Code and datasets are available at https://phi-lab-rice.github.io/GRADE.

### 论文解读

#### 摘要翻译

GRADE 面向烟雾、黑暗等视觉退化环境，利用单帧毫米波雷达估计稠密深度。它先从 4D 雷达谱得到粗深度，再结合潜扩散模型的视觉生成先验恢复几何细节；相机可用时再提供 RGB 引导，失效时仍能退化为雷达模式。

#### 方法动机分析

相机和 LiDAR 在烟雾中容易失效，毫米波虽能穿透，却因角分辨率低而产生稀疏、模糊深度。现有多模态方法通常以相机为主导，相机一旦失效就会明显退化。GRADE 的核心假设与设计驱动力，是让雷达提供可信的测距和 Doppler 几何，让生成模型补全边界与表面连续性，同时用三维约束防止“看起来合理、空间位置错误”的幻觉。

#### 方法设计详解

输入为同步 4D 雷达谱，另可输入 RGB。第一阶段用 Transformer 编码器分块建模雷达空间关系，再由 CNN 解码器输出 128×256 粗深度图。第二阶段把粗深度作为条件送入潜扩散 U-Net，在潜空间迭代去噪，借助视觉先验恢复细节；推理采用 DDIM，通常只需 8 步。类似 ControlNet 的视觉适配器从 RGB 提取边缘并通过零初始化跳跃连接注入，因此相机失效时影响自然消失。训练包含 L1、LPIPS、SSIM、梯度损失、去噪损失和像素深度损失，并将预测深度投影回 3D，加入重建损失约束空间位置。训练扩散步数 T=1000，Adam 学习率 1e-4、权重衰减 1e-2，使用 FP16 混合精度。

#### 方法对比分析

Depth Anything V3 等相机模型细节丰富但怕烟雾；GRT 等雷达模型耐烟雾却受稀疏回波限制；CaFNet、RadarCam-Depth 等融合模型往往依赖 RGB。GRADE 的创新不是简单拼接传感器，而是“雷达粗深度条件化扩散细化”：先锁定物理尺度，再让生成先验补细节，并用可失效的 RGB 辅助和 3D 损失维持鲁棒性。Doppler 还帮助处理空间重叠目标。它适合室内救援和恶劣环境导航，但跨雷达、跨户外分布仍需重训。

#### 实验分析（精简版）

数据约 9.5 万帧，约 4 万帧来自 12 栋建筑的真实烟雾采集，并按建筑物无关方式测试。烟雾场景 MAE 为 0.313 m，优于 DA3 的 1.255 m 和 GRT 的 0.436 m。消融表明 Doppler 有助于区分重叠目标，3D 重建损失能减少空间幻觉，8 步采样兼顾质量与效率。局限是扩散推理仍有开销、逐帧结果缺少时间一致性，极端户外覆盖有限。

#### 实用指南

论文表示将开源数据、代码和权重。复现需使用 TI IWR1843BOOST 77 GHz 雷达与 ZED 2i 相机的同步标定，保留 Doppler，并遵循真实烟雾和建筑物无关划分。迁移时要替换雷达谱处理、标定和数据分布，并重新训练条件深度与扩散模块；论文未说明全部依赖版本及实时部署指标。

#### 总结

核心思想：雷达测距，扩散补细节

1. 4D 雷达提取测距与 Doppler。
2. Transformer-CNN 生成粗深度。
3. 潜扩散恢复稠密结构。
4. RGB 可选引导，失效自动退出。
5. 3D 损失约束空间真实性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.10756v1)
- [arXiv](https://arxiv.org/abs/2609.10756v1)

---

<a id='2609.11225v1'></a>
## [Harness Robotic OS: A Unified Embodied-Agent Runtime for Closed-Loop Quadruped Inspection](https://arxiv.org/abs/2609.11225v1)

**Authors:** Yaoyuan Yan, Zhiyou Heng, Haoxiang Jie, Gang Liu, Hongjie Yan, Wei Zhou

**Published:** 2026-09-10

**Categories:** cs.RO

**Abstract:**

Autonomous property inspection requires more than robust robot navigation: a deployable system must connect heterogeneous sensing, reusable autonomy capabilities, multimodal scene understanding, human interaction, and enterprise response within a traceable operational loop. Existing quadruped inspection systems commonly integrate these functions through task-specific interfaces, making contextual coordination, knowledge reuse, and controlled adaptation difficult. This paper presents \textit{Harness Robotic OS} (HROS), a unified embodied-agent runtime, and Argos, its realization for residential-community inspection. HROS organizes the system into robot runtime, embodied autonomy skills, cognitive agent runtime, and interaction and operations planes. A shared context connects physical state with agent reasoning; streaming ASR/TTS supports voice-based mission interaction; hierarchical working, episodic, and semantic memory preserves operational knowledge; and a safety-gated self-evolution loop converts execution traces into versioned candidate updates without permitting unconstrained online modification. The Argos prototype integrates a Vbot quadruped, Fast-LIO2 localization and mapping, Hobot-Stereo depth perception, PCT-Planner global planning, EGO-Planner local motion generation, and OpenClaw-orchestrated Qwen3-VL inspection analysis. Experiments in a residential property environment achieved 100\% waypoint reachability, outdoor localization error below 10~cm, local obstacle-response latency below 200~ms, representative hazard-detection rates of 85--95\%, and 99\% success in alarm delivery and structured-report generation. These results validate the deployed navigation and inspection closed loop, while HROS provides an extensible software foundation for memory-augmented, voice-aware, and continuously improvable embodied inspection agents.

### 论文解读

#### 摘要翻译

论文提出 Harness Robotic OS（HROS），面向居民社区巡检四足机器人，把感知、自主导航、多模态推理、人机交互和企业运营统一到一个可追踪的闭环中。系统包含工作、情景、语义三层记忆，并以安全门控的自演化机制将执行轨迹转化为可版本管理的更新；作者用 Vbot 四足机器人实现 Argos 原型。

#### 方法动机分析

传统巡检系统往往把导航、视觉分析、语音和工单分开，任务意图难以关联机器人状态与现场证据，历史轨迹和失败也难以复用。直接在线修改提示词、工具或任务图又存在不可评估、不可回滚的风险与挑战。HROS 的关键思路是记录完整执行链路，并让改进候选先经过离线回归和安全门，再进入现场。它主要针对社区安全与卫生巡检，季节植被、施工地图变化、开放世界危险和户外噪声仍是局限。

#### 方法设计详解

输入来自 16 线激光雷达、双目相机、IMU、GNSS、麦克风，以及语音指令、航点和巡检模板。Fast-LIO2 融合激光与惯性数据完成定位建图，Hobot-Stereo 生成稠密近场深度，帮助发现低矮或细小障碍；PCT-Planner 在三维点云上规划全局路径，EGO-Planner 再进行局部轨迹优化和反应式避障。OpenClaw 负责意图路由与任务图，Qwen3-VL 分析关键图像，三层记忆分别保存当前任务、带时间的轨迹决策失败，以及稳定的站点区域和规则。机器人使用 RDK S100P（6 核 ARM Cortex-A78AE、128 TOPS），通过 4G/5G 接入企业 API；流式 ASR/TTS 支持语音交互。推理部署在该边缘计算平台，障碍响应延迟小于 200 ms。任务结果和人工复核经钉钉、飞书进入反馈闭环。自演化流程会反思执行结果、归因错误、生成候选更新，并在离线回归通过安全门后部署。

#### 方法对比分析

它的本质区别不是提出新的单一视觉或规划算法，而是把成熟的定位、规划、视觉模型与智能体编排、分层记忆、企业工单和版本化安全更新组合成统一运行时。相比一次性检测脚本，HROS 更适合重复、可审计的物业巡检；换到其他机器人或站点时，地图、规则、任务模板和相关数据需要重新配置或训练。

#### 实验分析（精简版）

真实住宅社区测试显示：航点到达率 100%，室外定位误差小于 10 cm，障碍响应延迟小于 200 ms；一次完整覆盖任务不超过 60 min，续航超过 3 h。垃圾溢出和消防通道堵塞检测率均为 95%，积水 88%，设施损坏 85%，误报/漏报率低于 5%，告警投递与报告生成准确率均为 99%。这些结果支持其在既定巡检类别中的实用性，但缺少记忆和安全自演化模块的消融，也未充分验证跨季节、施工环境的泛化。

#### 实用指南

复现需准备 Vbot、RDK S100P 及上述传感器，并部署 Fast-LIO2、Hobot-Stereo、PCT-Planner、EGO-Planner、OpenClaw 和 Qwen3-VL。论文链接了 OpenClaw 仓库，但未明确 HROS/Argos 完整代码是否开源。实践中应先建图，再配置航点和规则，接入图像分析、人工复核及工单系统；迁移到新站点需重建地图与语义记忆，并用目标场景数据补充危险类别。提示、工具和任务图的改动必须经过回归测试与安全门。

#### 总结

核心思想：安全演化的具身巡检闭环

1. 多传感器建图定位，双目深度补足近场障碍。
2. 全局规划配合局部避障驱动四足巡航。
3. 智能体挑选关键画面并完成视觉分析。
4. 三层记忆沉淀经验，安全门审核后演化部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.11225v1)
- [arXiv](https://arxiv.org/abs/2609.11225v1)

---

<a id='2609.11433v1'></a>
## [Safety-aware Skill Adaptation for Reinforcement Learning in Dynamic Environments](https://arxiv.org/abs/2609.11433v1)

**Authors:** A K M Nadimul Haque, Sheila Sutjipto, Marc G. Carmichael, Teresa Vidal-Calleja

**Published:** 2026-09-10

**Categories:** cs.RO

**Abstract:**

Skill adaptation frameworks based on reinforcement learning often require restrictive assumptions to maintain stability, such as fixed observations or tightly controlled exploration schedules. In cluttered and dynamic environments, however, unrestricted exploration can lead to unsafe behaviour and unstable learning, particularly when task-relevant observations lie near obstacles or involve moving objects. In this work, we present Dist-GPRL, a distance-aware and safety-guided reinforcement learning framework for structured robot skill adaptation. Building upon Gaussian Process (GP)-based skill parameterisation, our framework sequentially adapts overlapping local windows of sparse trajectory via-points rather than modifying the complete skill at every policy step. Raw policy outputs are correlated through the GP covariance structure, producing temporally coherent trajectory updates while reducing the action-space and credit-assignment difficulties associated with global trajectory adaptation. Safety is incorporated through two complementary forms of guidance. A safe-subspace prior derived from the Hausdorff Approximation Planner (HAP) biases policy exploration toward feasible regions, while dynamically updated distance field clearance and gradient rewards provide local obstacle awareness. A trajectory-kinematics similarity regulariser further preserves the demonstrated velocity and acceleration characteristics during adaptation. We evaluate the framework on two dynamic object-manipulation tasks in simulation and transfer the learned policy to real-world robot execution. Experimental results demonstrate higher task success, lower collision frequency, and more stable learning than the baselines, while preserving the kinematic characteristics of the demonstrated skill.

### 论文解读
#### 摘要翻译
论文提出 Dist-GPRL，一种面向动态环境的距离感知安全技能自适应框架。它结合局部轨迹窗口、高斯过程协方差、Hausdorff 近似规划器（HAP）安全先验和动态欧几里得距离场（EDF）奖励，在障碍物移动、任务配置变化时调整机器人技能，并在动态推方块、动态金属条操作及 UR5e 真机上进行验证。

#### 方法动机分析
传统 GPRL 往往一次修改整条高维轨迹，探索空间大，奖励稀疏时难以判断哪段动作导致成功或碰撞；动态障碍也使“原轨迹无碰撞”的假设失效。本文假设专家技能的大体运动结构仍可保留，变化主要需要局部、连续的几何修正，因此把自适应、安全探索和运动学保持放进同一回路。

#### 方法设计详解
状态包含局部 via-points、EDF 间隙、末端到目标物/目标的距离向量和轨迹相位。SAC 只输出当前重叠窗口的修正，随后通过 GP 算子 ΔΓ=S_Wã 将离散动作变为时间相关、平滑的轨迹增量。HAP 预计算可行安全子空间，并以软最小距离损失抑制危险探索；EDF 实时产生靠近障碍物时快速下降的安全距离奖励，以及鼓励沿障碍梯度远离障碍的对齐奖励。速度、加速度剖面的余弦相似度则保持专家技能风格。训练采用 SAC，训练步逐步更新 EDF、选择窗口、修正轨迹、执行并用任务奖励和安全/相似度项更新策略。

#### 方法对比分析
相比全局 GPRL，Dist-GPRL 同时降低局部动作维度并改善信用分配；相比 ProMP-RRL，它把安全几何先验和距离梯度直接纳入探索；相比 Dist-GPRL-Global，它进一步检验了窗口化更新的价值。方法适合已有示范轨迹、可获得环境距离场的动态操作任务，但对噪声传感器和需要形式化安全证明的场景仍需扩展。

#### 实验分析（精简版）
在位置随机偏移最高 20 cm 的两类仿真任务中，Dist-GPRL 在动态推方块上的成功率/碰撞率为 89%/5%，动态金属条操作为 98%/1%；原始 GPRL 仅为 18%/68% 和 44%/54%。UR5e 真机金属条实验中，方法达到 90% 成功率、10% 碰撞率，最小间隙 0.027 m，而 GPRL 为 10%、70%、0.001 m。消融显示 HAP 加快收敛，GP 变换使曲线更平稳；但论文没有给出形式化安全保证。

#### 实用指南
复现需准备专家 via-points、GP 协方差算子、HAP 安全子空间和 EDF，并训练 SAC，同时实现任务、距离、梯度对齐、HAP 与运动学相似度项。论文所读内容未明确说明代码、模型或数据是否开源。迁移到新机器人需重建碰撞几何、运动学、EDF 和 HAP，并重新检查轨迹末端的 GP 外推与最小间隙。

#### 总结
核心思想：局部安全引导的平滑技能适应。
1. 观测 EDF 与目标状态。
2. SAC 修正局部窗口。
3. GP 平滑连接轨迹。
4. HAP 与距离梯度约束探索。
5. 滚动执行以适应动态障碍。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.11433v1)
- [arXiv](https://arxiv.org/abs/2609.11433v1)

---

<a id='2609.10895v1'></a>
## [ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs](https://arxiv.org/abs/2609.10895v1)

**Authors:** Yizhan Li, Jianxin You, Mengyang Xiong, Yinhuan Chen, Zicheng Zhao, Dekun Wu, Dongqing Zhang, Bang Liu

**Published:** 2026-09-09

**Categories:** cs.RO, cs.AI

**Abstract:**

Reacting to sudden physical hazards (catching a slipping plate, dodging a falling knife) is both a meaningful test of embodied intelligence and a hard requirement for deploying multimodal large language models (MLLMs) as the decision coreof household robots. Existing evaluations, however, probe intuitive physics passively through question answering over videos, or target deliberate, long-horizon tasks such as navigation and rearrangement; none measure whether a model can turn physical understanding into immediate, safety-critical action. We introduce ReactHuman, the first physics-grounded benchmark for human-like reactive decision-making, in which the evaluated MLLM acts as the brain of a simulated humanoid facing sudden household hazards; it spans 17 event families and over 1,000 bit-for-bit reproducible scenes with exact, annotation-free ground truth derived from 240 Hz rigid-body simulation, including adversarial objects whose appearance contradicts their physics (a foam anvil, a steel apple). We further design a five-metric suite that scores each reaction along three axes: reasonable, safe, and physically grounded. We physically execute every committed plan so that decisions have observable consequences. With this harness we evaluate seven representative MLLMs. Results show that reactive safety is far from solved: models mishandle roughly one hazard in three, act from fixed dispositions rather than the observed scene, trust appearance over motion, and miss interception points at meter scale even when the chosen action is correct; none of these failures shrink with model scale. ReactHuman thus offers both a fine-grained diagnosis and a scalable training signal toward physically grounded, safety-aware embodied agents. The benchmark can be found here: https://huggingface.co/datasets/Alan123/reacthuman-benchmark-scaled

### 论文解读

#### 摘要翻译

ReactHuman 是一个评估具身多模态大模型（MLLM）突发危险反应能力的基准，场景包括接住下落刀具、躲避倾倒货架等。它包含17个事件族、1000多个逐位可复现场景，并让模拟人形机器人在240 Hz刚体仿真中执行模型计划。

#### 方法动机分析

驱动力在于家庭机器人需要在极短时间内把“看懂物理”变成安全动作。传统视频问答偏被动，导航或整理任务又允许较长思考，难以测量模型是否选对动作、能否准确拦截，以及失败究竟来自识别还是控制；这正是现有评测的具体痛点。论文的核心假设是安全反应需要直觉物理和空间拦截能力，并关注外观偏见：模型可能把泡沫物体当重物，或把缓慢移动的危险当成无事发生。

#### 方法设计详解

基准采用 Freeze-and-Predict 协议：事件触发后，模型观看约0.6秒的视平线、近景和俯视同步视频，仿真在决策点冻结；模型输出包含 intent、confidence、walking_cmd、keyframes 的JSON计划，动作类别为 Catch、Dodge 或 No-Action。计划交给由预训练行走策略和手部关键帧跟踪组成的全身控制器，使Unitree G1在Genesis中执行。场景由语言规划器设定语义，再以整数seed确定速度、摩擦和光照等物理参数，最后以240 Hz求解。评测同时计算动作准确率、安全有效性、最终手部与撞击点距离、动作—意图对齐，以及手部距离随时间的变化，从而分离语义判断和空间执行误差。

#### 方法对比分析

ReactHuman不只核对答案，而是把模型计划放进可执行的人形体和物理世界。相比普通VQA，它能区分“选错接还是躲”“动作选对但伸手够不到”“语言意图与实际轨迹不一致”。17类突发事件和14种外观—运动冲突探针，专门检验直觉物理和安全反射。它适合评估具身决策，不等同于真实机器人认证，也不覆盖在线重规划。

#### 实验分析（精简版）

七个MLLM在306个场景上测试，动作标签包括Catch 141、Dodge 156、No-Action 9。平均动作准确率为54.0%，安全得分为80.8%，平均终点距离1.23 m。Gemma-3-27B安全得分88.2%，高于GPT-5.5的86.3%；Claude Opus 4.8终点距离最好，为1.07 m。Dodge最难，安全违规率35.9%，常见错误是原地冻结；正确选动作时也可能因只伸手、不迈步而错过目标，中位偏差0.48 m。模型还常按外观而非运动判断重量。

#### 实用指南

数据集发布在Hugging Face，场景由seed确定，可复现；模型通过OpenRouter进行zero-shot推理。复现需保持三视角、0.6秒观察窗、1280×720/60 fps渲染、Genesis 240 Hz求解和统一JSON格式，并同时记录五类指标。迁移到其他机器人要替换身体、可达性和控制器并重新校准物理真值；论文未说明训练代码或真实硬件方案。

#### 总结

核心思想：让反应接受物理执行检验

1. Seed生成突发危险场景。
2. 多视角短视频后冻结决策。
3. MLLM输出动作与关键帧计划。
4. 人形控制器恢复仿真执行。
5. 用安全、距离和意图指标定位失败。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.10895v1)
- [arXiv](https://arxiv.org/abs/2609.10895v1)

---

<a id='2609.11043v1'></a>
## [LTLDiff: Finite Linear Temporal Logic-Guided Data Generation and Diffusion Policies for Multi-agent Robotic Manipulation](https://arxiv.org/abs/2609.11043v1)

**Authors:** Chuhan Meng, Haiyan Yin

**Published:** 2026-09-10

**Categories:** cs.RO

**Abstract:**

Multi-agent robotic manipulation tasks require coordination among agents to satisfy task-level temporal, logical, and safety constraints. Recently, diffusion policies have been used to perform the task. However, they still suffer from desynchronization, incorrect action ordering, and coordination failures in tasks that require simultaneous or sequential multi-agent interaction. Therefore, LTLDiff is proposed as a framework that combines Finite Linear Temporal Logic (LTLf) specification learning for both the generation of demonstrations and learning via diffusion policies. Each task has a specific LTLf formula that is learned from a set of natural language instructions using a large-scale language model. To enable a fixed-dimensional vector embedding of the learned specification from the language model, LTLf uses an abstract syntax tree representation scheme. This embedding of logic serves as a condition for (i) logic-guided data collection and (ii) diffusion-based policy training, encouraging trajectories that are consistent with the desired ordering and coordination requirements. Experiments on multi-agent LTLDiff manipulation tasks demonstrate improved task success rates compared to the baseline. Together, these contributions demonstrate the effectiveness of LTLDiff for coordinated multi-agent manipulation.

### 论文解读
#### 摘要翻译
LTLDiff 用有限线性时序逻辑（LTL_f）指导多机器人操作的数据生成和扩散策略学习。自然语言任务先由大语言模型转成逻辑公式，再编码为固定维度向量；逻辑同时约束专家示范、作为策略条件，并在采样时指导联合轨迹。RoboFactory 上的实验显示，它能改善多机器人任务中的操作顺序、同步与协作，但超长任务仍然困难。

#### 方法动机分析
规则方法有时序和安全保证，却难适应新环境；行为克隆和强化学习灵活，却难显式保证“先做什么、何时协作”。多智能体扩散策略尤其容易失同步或违反跨机器人约束。作者的关键假设是，把可表达的任务约束注入示范分布、训练条件和推理采样，能给视觉动作学习提供稳定的时序归纳偏置。该假设主要适用于有限时域、公式可准确描述的任务。

#### 方法设计详解
输入是自然语言指令、目标条件、全局场景图像和各机器人的第一视角 RGB。Qwen 输出原子命题及 LTL_f 公式；公式解析成 AST 后，叶节点 one-hot 编码，内部节点把操作符 one-hot 与子节点表示的平均值拼接，根节点再零填充为 128 维向量。该向量一方面调节运动规划中的速度、加速度和物体位置，生成符合逻辑顺序的离线示范，另一方面条件化扩散策略。推理时，各机器人先局部去噪，再将联合轨迹交给 MLP 回归器 LTLMAG，预测提升逻辑满足度的梯度并修正动作。引导强度随时间变化：早期为 2ξ_i，后期为 0.1ξ_i，基础值为 0.1。

#### 方法对比分析
Base 是无逻辑条件的扩散策略，LTLR 是既有 LTLDOG 的多机器人适配。LTLDiff 的本质创新是将 LTL_f 贯穿示范生成、策略训练和联合采样，并在拼接后的多机器人轨迹上施加满足梯度，因此能直接处理协作时序，而非只逐个机器人纠偏。它适合有明确先后、同步或安全条件的视觉操作；未被公式描述的动力学和实时变化仍需反馈控制补足。

#### 实验分析（精简版）
RoboFactory 包含 1—4 个机器人、11 个任务，训练数据为 50、100 或 150 条专家示范。150 条示范时，Pick Meat 成功率由 Base 的 58% 和 LTLR 的 76% 提升到 LTLDiff 的 87%；Lift Barrier 为 58%、84% 和 96%；Camera Alignment 为 19%、55% 和 67%。消融中，Camera Alignment 使用 100 条示范时，仅逻辑数据生成（Gen）为 40%，训练阶段继续使用逻辑（Train）达到 67%。不过 Long Pipeline Delivery 三种方法均为 0%，显示长时域协作和开环误差累积仍是明显局限。

#### 实用指南
复现需 RoboFactory、任务 LTL_f 公式、Qwen 提示、视觉专家示范、128 维 AST 编码和 MLP 满足回归器；还要实现联合轨迹引导及上述 ξ 调度。论文提供任务公式和提示，但未给出明确代码仓库链接，开源状态不能确认。迁移到新任务要重写原子命题和公式，生成一致示范，并重训扩散策略和回归器；改变机器人数量还需调整联合输入。真实部署建议加入滚动时域重规划与安全控制层。

#### 总结
核心思想：让逻辑贯穿扩散策略

速记：
1. 语言转 LTL_f，再由 AST 编成条件向量。
2. 用逻辑生成协作示范，并进行联合扩散去噪与满足梯度引导。
3. 视觉执行；用闭环重规划补足长时域误差。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.11043v1)
- [arXiv](https://arxiv.org/abs/2609.11043v1)

---

<a id='2609.10918v1'></a>
## [ObstaDiff: Generalizable Diffusion Policy Learning via Obstacle-aware Representations](https://arxiv.org/abs/2609.10918v1)

**Authors:** Jiawen Wang, Kevin Yao, Khalid Jawed

**Published:** 2026-09-10

**Categories:** cs.RO, cs.LG

**Abstract:**

Imitation learning has achieved impressive results in robotic manipulation, yet most existing approaches assume clean backgrounds and lack explicit mechanisms for obstacle-aware motion generation. Extending such policies to cluttered, real-world scenes with unstructured obstacles remains a key generalization challenge. We present ObstaDiff, a decomposed diffusion-policy framework with a lightweight obstacle-aware visual encoder. ObstaDiff extracts a structured target-obstacle-background representation, enabling the downstream alignment policy to generate end-effector trajectories toward a target-centered bottleneck pose while reasoning about surrounding obstacles. We evaluate ObstaDiff on 61 real-robot greenhouse trials per method (366 executions in total). ObstaDiff achieves 75.41% average task success and 8.20% average obstacle collision rate, outperforming representative imitation-learning baselines and improving generalization in cluttered agricultural scenes.

### 论文解读

#### 摘要翻译

ObstaDiff 面向杂乱温室中的机器人模仿学习，提出障碍物感知表示，将视觉观察拆成目标、障碍物和背景（TOB），再配合扩散策略完成安全、可泛化的目标对准，并在瓶颈状态后回放交互动作。

#### 方法动机分析

叶片、茎秆和电线会形成狭窄通道，光照与目标外观也不断变化。普通 RGB 端到端编码器把“应接近的目标”和“应躲避的障碍”混在一起，导致泛化时碰撞。作者的假设是显式语义解耦能让策略直接学习目标—障碍几何关系；但它主要解决短程对准，不负责长程规划。

#### 方法设计详解

手眼 RGB-D 图像先经 YOLOv8s-Worldv2 检测、MobileSAM 分割，组成 4×240×320 的 TOB+深度输入。结构化编码器以语义和深度双分支处理，卷积后用 Spatial Softmax 提取空间关键点；64 维语义特征、32 维深度特征和 7 维末端位姿拼成 103 维条件向量。条件 Diffusion U-Net 接收连续两步观察，预测 16 步动作但执行前 8 步，滚动更新观察后继续控制。扩散训练用均方误差预测噪声，AdamW 学习率为 10^-4、batch size 32、训练 250 epochs。对准达到瓶颈后，利用图像误差阈值 0.2 和深度误差阈值 0.005，从 26 段交互轨迹中门控匹配并回放动作；这种切换避免模型在接触遮挡阶段继续依赖不稳定的连续避障预测。

#### 方法对比分析

贡献不只是添加深度，而是让显式 TOB 表示与结构化编码器协同工作。相比 ACT 或标准 Diffusion Policy 的混合视觉潜变量，ObstaDiff 将接近避障和接触交互分开；相比仅输入 TOB 的 DP(TOB)，它增加了匹配的空间编码机制。该方案适合有可靠目标/障碍分割和专家演示的短程操作。

#### 实验分析（精简版）

实验使用 7 自由度 Sawyer 和 RealSense D455，以紫色辣椒训练，在目标位姿、障碍布局和绿色辣椒外观上测试。平均成功率/碰撞率/耗时分别为：ObstaDiff 75.41%/8.20%/15.57 s，ACT(RGB) 45.90%/14.75%/20.77 s，DP(RGB) 49.18%/26.23%/18.41 s。DP(TOB) 碰撞率仍为 27.87%，说明语义输入若没有结构化编码器并不能带来安全性。局限是场景单一，并依赖检测、演示和回放库，跨场景结论仍需更多验证。

#### 实用指南

复现需准备 90 段对准演示和 26 段交互轨迹，并保持两步条件、16 步预测/8 步执行等设置；评测应同时统计成功、碰撞和耗时，并分开测试目标位姿、障碍布局与外观变化。论文未给出可确认的代码、模型或数据开源链接。迁移到新任务时要替换检测提示和分割器，重新采集演示；若没有稳定瓶颈状态或需要探索回放库之外的行为，还需重做切换机制。

#### 总结

核心思想：语义解耦让策略安全泛化

1. 分割目标与障碍，构造 TOB+深度观察。
2. 双分支编码器提取语义几何关键点。
3. 扩散策略短段控制，完成避障对准。
4. 门控匹配瓶颈状态，回放交互轨迹。

总体上，结构化表示同时改善了成功率与安全性，但实验规模仍不足以证明其对不同作物、传感器和更复杂障碍拓扑的普适性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.10918v1)
- [arXiv](https://arxiv.org/abs/2609.10918v1)

---

