time: 20260803

# Arxiv Computer Vision Papers - 2026-08-03

## Executive Summary

## 执行摘要

本报告汇编了 2026-07-31 日发表于 arXiv 的 10 篇计算机视觉与机器人领域论文，核心聚焦于**具身智能、机器人学习、视觉感知与操作**。整体来看，研究正从“单模型/单任务”走向“低成本仿真 + 闭环微调 + 多模态交互”的系统化路径。

### 1. 主要主题与趋势

- **世界模拟器与世界模型**：BWM 提出低成本高保真世界模拟器，WCM 将世界模型作为“评论家”用于视觉-语言-动作强化学习，两者共同指向**利用世界模型提升机器人学习效率**。
- **模仿学习与视觉表示**：RayViT 针对视点鲁棒性，Temporal Policy 利用历史信息生成动作，CLIFT 则通过闭环迭代微调将通用模型转化为专用人形机器人专家。
- **人形机器人与物理交互**：有研究人形搬运物体时的质量平衡与举升控制，也有可穿戴触觉传感器 TacPrint 用于人-机器人接触再现。
- **操作与规划**：TransGraspNet 解决透明实验器皿的抓取问题，Homotopy-Aware Corridor Generation 在没有参考路径的情况下生成走廊路径。
- **几何与无训练方法**：CorrelationFlow 提出无需训练的 LiDAR 场景流估计方法，体现了几何先验在降低数据依赖方面的价值。

### 2. 特别重要或创新的论文

- **CLIFT**：非侵入式闭环迭代微调，无需修改模型内部结构即可将通用机器人模型变为“人形专家”，对实际部署非常实用。
- **BWM**：低成本高保真世界模拟器，可能大幅降低机器人学习所需的真实数据采集成本。
- **WCM**：将世界模型与评论家结合，为 VLA 策略强化学习提供更强反馈信号，潜力大。
- **RayViT**：通过射线条件下的视觉表示增强视点鲁棒性，是模仿学习泛化的重要尝试。
- **TransGraspNet**：面向透明物体且兼顾物理/几何一致性，在科研与医疗实验室自动化场景中具有明确应用价值。

### 3. 新兴研究方向或技术

- **世界模型 + 批评家/价值网络的融合**，用于训练更高效的具身智能体。
- **非侵入式微调 + 在线闭环优化**，实现低资源、低成本的个性化机器人部署。
- **无训练、几何驱动的感知方法**，与基于学习的方案互补。
- **触觉传感与可穿戴设备**，增强机器人对物理接触的精细化表达能力。
- **透明材质物体的感知-操作一体化**，从传统抓取走向更复杂的材质与光照场景。

### 4. 建议全文阅读的论文

- 关注机器人学习基础设施：**BWM** 与 **WCM**。
- 关注人形机器人与部署：**CLIFT** 与 **Balancing of Humanoid with Object Mass**。
- 关注模仿学习与视点泛化：**RayViT** 与 **Temporal Policy**。
- 关注灵巧操作与触觉：**TransGraspNet** 与 **TacPrint**。
- 关注场景流估计与高效几何方法：**CorrelationFlow**。
- 另可阅读 **Homotopy-Aware Corridor Generation**，其对无参考路径的规划问题具有启发意义。

---

## Table of Contents

1. [BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning](#2607.29302v1)
2. [CLIFT: Turning Gemini Robotics On-Device into Humanoid Specialists via Non-Invasive Closed-Loop Iterative Fine-Tuning](#2607.29172v1)
3. [Balancing of Humanoid with Object Mass: Trade-off Analyses and Lifting Control](#2607.29625v1)
4. [RayViT: Ray-Conditioned Visual Representations for Viewpoint-Robust Imitation Learning](#2607.29622v1)
5. [WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning](#2607.29613v1)
6. [TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware](#2607.29567v1)
7. [Homotopy-Aware Corridor Generation without Predefined Reference Paths](#2607.29513v1)
8. [Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration](#2607.29482v1)
9. [CorrelationFlow: A Training-Free Geometric Approach for LiDAR Scene Flow Estimation](#2607.29237v1)
10. [TacPrint: A Wearable Fingertip Tactile Sensor for Human-to-Robot Contact Reproduction](#2607.29231v1)

---

## Papers

<a id='2607.29302v1'></a>
## [BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning](https://arxiv.org/abs/2607.29302v1)

**Authors:**  BWM Team

**Published:** 2026-07-31

**Categories:** cs.RO, cs.CV

**Abstract:**

Reliable robot learning requires a world simulator that can predict action consequences before execution on physical hardware, including risky and failure-prone outcomes. Existing physics simulators require substantial asset construction and calibration and still face a sim-to-real gap, while video generators often lack precise control over their responses to fine-grained robot actions. In this paper, we present the Boundless World Model (BWM), an open-source, low-cost, high-fidelity world simulator for robot manipulation. BWM is an action-conditioned world model that combines initial-environment guidance, dynamic visual history, and temporally aligned robot-action conditioning for stateful autoregressive prediction of future observations. We construct action-aligned training clips through trajectory replay, overlapping clip sampling, and initial-observation enhancement. BWM serves as a data engine that augments imitation-learning data with action-aligned rollouts, and as a policy evaluator for closed-loop assessment, risk anticipation, and policy ranking. Experiments on the WorldArena benchmark and physical robots demonstrate improved simulator fidelity and functional utility across the data-engine and policy-evaluator settings. BWM ranks first overall in the WorldArena Challenge across Track 1 and its two Track 2 applications. We release the BWM open-source ecosystem, including model checkpoints, training and inference code, and interfaces for data generation and policy evaluation.

**Analysis:**

以下是对《BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning》一文的深度分析：

### 1. 摘要翻译
可靠的机器人学习需要一个能在物理执行前预测动作后果的世界模拟器，包括预测风险和潜在故障。现有的物理模拟器建设和校准成本高昂且存在“仿真到现实”的鸿沟，而视频生成模型往往缺乏对精细机器人动作的精确控制。本文提出了Boundless World Model (BWM)，这是一个开源、低成本、高保真的机器人操作世界模拟器。BWM是一种动作条件世界模型，结合了初始环境引导、动态视觉历史和时序对齐的动作条件，用于状态化、自回归的未来观测预测。我们通过轨迹回放、重叠剪辑采样和初始观测增强构建了动作对齐的训练片段。BWM不仅作为通过动作对齐的Rollout增强模仿学习数据的数据引擎，还作为闭环评估工具，用于政策的评估、风险预判和排名。在WorldArena基准测试和物理机器人上的实验表明，BWM提升了模拟器的保真度和功能性。BWM在WorldArena挑战赛中总排名第一，我们开源了模型权重、训练推理代码及相关接口。

### 2. 方法动机分析
*   **驱动力**：旨在以低成本方式构建一个既能提供保真度仿真，又能直接服务于下游机器人任务（如数据增强和策略评估）的通用世界模型。
*   **痛点**：物理仿真器依赖繁琐的资产建模且仿真与现实存在Gap；通用视频生成模型（如Sora）缺乏对细粒度动作的精确控制能力，无法实现真正的交互式仿真。
*   **核心直觉**：通过“动作对齐”的训练范式，将机器人特定的动作序列作为条件注入到预训练的视频生成主干中，并通过初始帧和动态历史窗口维持场景的一致性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据构建**：利用已有的物理轨迹进行高分辨率回放（Trajectory Replay），通过重叠窗口切片（Overlapping Clip Sampling）并对初始观测进行增强，确保动作与观测的时间戳完全同步。
    2.  **模型架构**：采用基于DiT（Diffusion Transformer）的视频生成主干，引入专门的“动作接口”。
    3.  **动作注入**：通过“帧级”分支（Cross-Attention）注入精细动作信息，确保瞬时动作响应；通过“潜在级”分支（Latent-level）注入时序聚合的动作特征，通过AdaLN调整去噪过程。
    4.  **推理过程**：采用自回归预测，给定初始环境帧和动态历史窗口，生成未来观测片段。
*   **算法解释**：使用流匹配（Flow Matching）目标函数进行训练，通过对历史 latent 添加噪声 $\sigma_h$ 并保持初始帧 $z^0$ 清洁，强制模型学习在维持初始环境一致性的前提下，根据后续动作平滑演化视觉状态。

### 4. 方法对比分析
*   **本质区别**：BWM并非直接从零训练视频模型，而是将重点放在“动作接口”的设计上。它通过精细的轨迹对齐数据 pipeline 解决了通用视频生成模型的动作控制难题。
*   **创新贡献**：提出了“动作对齐的数据引擎”与“闭环策略评估”的双重范式，证明了即使是单一的视频生成模型，通过合理的动作注入设计，也能表现出优于专业物理模拟器的策略评估能力。

### 5. 实验分析（精简版）
*   **关键结论**：在WorldArena基准测试中，BWM的EWMScore（综合保真度得分）达到63.51，领先于所有现有主流世界模型；在物理机器人实验中，基于BWM增强的策略成功率达到71.00%，大幅优于基线方案。
*   **优势**：在低资源配置下实现高保真度；闭环评估与物理实验的相关性极高（Pearson $r=0.908$）。
*   **局限**：动作条件的注入虽然精细，但对超长程动作规划的复杂语义理解仍可能受限于视频生成主干的通用能力。

### 6. 实用指南
*   **开源地址**：[github.com/boundless-large-model/boundless-world-model](https://github.com/boundless-large-model/boundless-world-model)
*   **实现细节**：建议使用 $H=8$ 的历史帧长和 $K=72$ 的预测长度，且动作需预处理为 [p1, p99] 归一化区间。
*   **迁移建议**：可直接将此架构迁移至其他具备机器人交互视频数据的平台，核心只需更换动作注入的维度（$d_a$）和重采样逻辑。

### 7. 总结
*   **核心思想**：通过动作驱动的视频自回归生成，实现高保真的机器人交互仿真与策略评估。
*   **速记版pipeline**：
    1. 同步物理回放轨迹；
    2. 切分重叠训练样本；
    3. 将动作信息注入扩散模型；
    4. 自回归生成未来视帧；
    5. 用生成的轨迹训练或评估机器人策略。

**Key Findings:**

- In this paper, we present the Boundless World Model (BWM), an open-source, low-cost, high-fidelity world simulator for robot manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29302v1)
- [arXiv](https://arxiv.org/abs/2607.29302v1)

---

<a id='2607.29172v1'></a>
## [CLIFT: Turning Gemini Robotics On-Device into Humanoid Specialists via Non-Invasive Closed-Loop Iterative Fine-Tuning](https://arxiv.org/abs/2607.29172v1)

**Authors:** Yuxin Chen, Hari Srikanth, Nathan Jew, Menglin Wu, Pengcheng Wang, Junli Ren, Masayoshi Tomizuka, Peng Xu, Jinyu Xie, Thomas Tian

**Published:** 2026-07-31

**Categories:** cs.RO, cs.AI

**Abstract:**

While robot foundation models are growing increasingly capable, the strongest models are typically trained on proprietary data and remain closed-source, limiting downstream users' ability to adapt them to new tasks, embodiments, and deployment settings. Following the LLM community, an emerging access paradigm for closed-weight robot foundation models is the managed supervised fine-tuning (SFT) API, where users submit training data and receive a tuned policy without access to model weights, gradients, or training internals. While such APIs let downstream users leverage powerful proprietary foundation models, they restrict policy improvement to pure imitation, ruling out reinforcement learning and other closed-loop methods that rely on internal training signals. This limitation is particularly acute for agile, contact-rich humanoid manipulation, where the gap between policy outputs and deployed behavior is large due to novel states, action tracking dynamics, latency, and controller-specific failure modes. We study how effective this managed-API regime is for humanoid adaptation, and how closed-loop improvement can be realized within it to push policies toward task mastery. We conduct one of the first empirical studies of managed-API adaptation on a real humanoid, instantiated on Gemini Robotics On-Device (GROD). We find that direct SFT through the API substantially outperforms a leading open-weight VLA trained on the same demonstrations, yet still falls short of deployment-level mastery on agile, contact-rich tasks. To close this gap, we introduce CLIFT: Closed-Loop Iterative Fine-Tuning, which turns deployment-time reward feedback into API-compatible supervised data and enables closed-loop policy improvement without accessing weights, gradients, likelihoods, or losses-pushing GROD to near-perfect success after two flywheel cycles, all without "opening the model box."

**Analysis:**

### 1. 摘要翻译
随着机器人基础模型能力不断增强，最强大的模型通常由于依赖专有数据而保持闭源，限制了用户将其适配到新任务的能力。目前，针对闭源（closed-weight）模型的一种新兴范式是“受控监督微调（SFT）API”，但这种方式限制了策略改进只能通过模仿学习进行，排除了需要内部训练信号的强化学习方法。这在处理敏捷、接触丰富的类人操作任务时尤为受限，因为任务部署环境与人类演示环境存在巨大偏差。本文提出了 **CLIFT（闭环迭代微调）**，这是一种非侵入式的闭环自适应流水线，通过将部署时的奖励反馈转化为API兼容的监督训练数据，实现了在不触及模型权重、梯度或损失函数的情况下进行策略闭环改进，在两次飞行周期（flywheel cycles）内将Gemini Robotics On-Device (GROD) 机器人的任务成功率推向了近乎完美，无需“打开模型黑盒”。

### 2. 方法动机分析
*   **驱动力**：旨在解决在无法获取模型内部参数（闭源模型）的限制下，如何实现机器人策略在复杂、高接触任务上的自我进化。
*   **痛点**：传统的模仿学习（SFT）仅能复刻演示轨迹，难以应对部署环境下的状态偏移、延迟及动态变化。而传统的强化学习（PPO等）需要访问模型梯度或内部动作概率，在闭源API受限模式下不可用。
*   **核心假设**：强化反馈信号可以通过重新标注数据的方式“编码”进监督学习样本中，从而通过迭代式的SFT间接实现闭环优化。

### 3. 方法设计详解
*   **流程总结**：
    1.  **初始化**：基于人类远程操作演示数据进行首次SFT微调得到基础策略 $\pi_0$。
    2.  **闭环部署与评分**：在机器人上部署当前策略，收集轨迹 rollouts；使用预训练的“偏好校准稠密奖励模型”对轨迹进行分段评分。
    3.  **检索式优势标注**：对于每个动作块（chunk），从其他所有rollouts中检索视觉相似的状态，通过对比其未来的累计奖励（look-ahead window）来判断当前块的优劣，打上“positive”或“negative”标签。
    4.  **迭代微调**：将带有优势标签的数据加入训练集，通过受控SFT API重新训练得到下一代策略，不断循环。
*   **关键技术**：
    *   **偏好校准奖励模型**：结合VLM生成和人类偏好，先通过比较对齐候选奖励，再蒸馏为通用的稠密奖励网络，解决了稀疏标注与错误评估问题。
    *   **状态感知信用分配**：通过检索相似状态作为基准，使得“奖励”不仅是绝对值，更是相对优势，有效解决了“在困难任务中表现平庸可能是由于初始状态极差”的归因难题。

### 4. 方法对比分析
*   **本质区别**：不依赖任何梯度流或内部权重修改（非侵入式），而是将强化学习转化为带标签的监督微调问题（Data-driven RL）。
*   **创新贡献**：提出了一种通用的、与模型架构解耦的“数据级反馈注入”范式，能够将任何闭源API转化为闭环学习器。
*   **适用场景**：所有提供SFT API的闭源机器人基础模型，尤其适合接触类、高动态、复杂人形操控任务。

### 5. 实验分析（精简版）
*   **关键结果**：在三项接触丰富任务上，CLIFT 使 GROD 策略成功率从基础 SFT 的 50%-90% 提升至近 100%。即使对比拥有完整内部权限的开权重模型（$\pi_{0.5}$），经由 CLIFT 优化的 GROD 依然展现出更强的鲁棒性和涌现行为。
*   **主要优势**：极强的通用性，不依赖模型内部权限；能够从自身失败的轨迹中通过对比学习提取出纠正行为（如抓取失败后重试）。
*   **主要局限**：非常依赖真实机器人部署收集数据，这对硬件耗损和安全性有要求。

### 6. 实用指南
*   **开源情况**：官方代码（coming_soon），项目网站：`thomaschen98.github.io/clift`。
*   **实现细节**：建议使用DINOv3进行状态嵌入以保证检索精度；对比组检索建议限制在单次执行仅取一个Chunk，避免过拟合。
*   **迁移建议**：若想迁移到其他embodiment，重点在于构建一个针对该机器人动作空间（如whole-body controller）的偏好奖励模型。

### 7. 总结
*   **核心思想**：通过检索式相对优势标注，将物理环境反馈转化为监督数据进行闭环优化。
*   **速记版Pipeline**：
    1. 部署策略搜集轨迹；
    2. 奖励模型自动分段评分；
    3. 对比相似轨迹确定标签；
    4. 加入数据池进行SFT迭代。

**Key Findings:**

- While robot foundation models are growing increasingly capable, the strongest models are typically trained on proprietary data and remain closed-source, limiting downstream users' ability to adapt them to new tasks, embodiments, and deployment settings.
- This limitation is particularly acute for agile, contact-rich humanoid manipulation, where the gap between policy outputs and deployed behavior is large due to novel states, action tracking dynamics, latency, and controller-specific failure modes.
- We find that direct SFT through the API substantially outperforms a leading open-weight VLA trained on the same demonstrations, yet still falls short of deployment-level mastery on agile, contact-rich tasks.
- To close this gap, we introduce CLIFT: Closed-Loop Iterative Fine-Tuning, which turns deployment-time reward feedback into API-compatible supervised data and enables closed-loop policy improvement without accessing weights, gradients, likelihoods, or losses-pushing GROD to near-perfect success after two flywheel cycles, all without "opening the model box."

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29172v1)
- [arXiv](https://arxiv.org/abs/2607.29172v1)

---

<a id='2607.29625v1'></a>
## [Balancing of Humanoid with Object Mass: Trade-off Analyses and Lifting Control](https://arxiv.org/abs/2607.29625v1)

**Authors:** Hyunjong Song, William Z. Peng, Joo H. Kim

**Published:** 2026-07-31

**Categories:** cs.RO

**Abstract:**

The demand for humanoid loco-manipulation tasks with an object has recently increased, and most existing control approaches for stability in such tasks rely on heuristics or machine-learning techniques. This study rigorously analyzes and exploits the dynamic effects of the object mass on balance stability. By formulating the object mass parameters in the whole-body dynamics with distributed contact wrenches and centers of pressure at the stance contacts, their nonlinear effects on the system momenta and constraints are quantified. The dynamic models and constraints are incorporated into the construction of the balanced state basin/boundary (BSB), a partition of the center-of-mass state space for a biped system to maintain balance in its desired contacts. The implications of the BSB for prediction and control are highlighted using a humanoid robot and an analytically tractable reduced-order mechanism. The BSBs under different conditions of base of support, actuation capacity, and pose provide systematic analyses of the effects of object mass on the balancing capability of a system. In particular, the trade-off relationships between momentum regulation and limiting factors in balancing are characterized, introducing two key quantities of the object: the critical mass, at which the system's balancing capability is maximum, and the transition mass, which activates different limiting factors. In addition, sufficient conditions for imposing balanced states on a trajectory are established and implemented with BSBs as explicit threshold constraints in the whole-body trajectory optimization for stable object-lifting control of the humanoid, demonstrating the lift-and-hold and lift-and-release tasks with distinct mass properties in simulations and experiments.

**Analysis:**

### 1. 摘要翻译
类人机器人的 loco-manipulation（移动-操作）任务需求日益增长，但现有的稳定性控制多依赖启发式或机器学习方法。本研究从动力学角度严谨地分析并利用了物体质量对平衡稳定性的影响。通过将物体质量参数纳入包含分布式接触力和压力中心（CoP）的全身动力学模型，量化了物体质量对系统动量和约束的非线性效应。基于这些动力学模型和约束，我们构建了“平衡状态盆地/边界”（BSB），即双足系统在保持目标接触状态下的质心状态空间划分。研究引入了“临界质量”（系统平衡能力最大时的物体质量）和“转换质量”（激活不同限制因素的阈值）两个关键量。此外，本研究建立并实施了基于BSB显式约束的全身轨迹优化方法，在模拟和实验中成功实现了不同质量属性下的物体提升控制。

### 2. 方法动机分析
*   **驱动力**：在移动-操作任务中，物体质量对稳定性既有干扰也有潜在的辅助作用，目前缺乏对这种非线性关系的定量理解。
*   **现有痛点**：现有方法多基于缩减阶模型（如倒立摆），忽视了全身动力学及上肢动力学，导致其在复杂操纵任务中评价标准过于保守或不准确。
*   **核心假设**：物体质量可通过增加系统动量来提升平衡性能，但受到关节力矩限制和接触约束的制约，存在一个非线性的最优平衡区。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **全身动力学建模**：将物体质量作为参数引入包含浮动基座的关节空间动力学，并显式构建考虑接触扳手（Wrench）与CoP分布的约束模型。
    2.  **BSB 构建**：通过一系列非线性优化问题，计算给定姿态和质量下的质心速度最大允许范围，形成“平衡状态边界（BSB）”。
    3.  **轨迹优化**：将预计算的BSB转化为轨迹优化的显式约束，通过二次规划（SQP）求解 lift-and-hold 和 lift-and-release 任务。
*   **关键公式**：研究核心在于将动力学约束表述为：$\max \dot{r}_x(0)$ s.t. $\text{BSB}_{LB} \le \text{BSB}(v, \alpha) \le \text{BSB}_{UB}$。其中 $\alpha$ 作为时变参数，用于动态分配接触扳手。
*   **核心直觉**：当物体质量较小时，底座支撑区域（BoS）是限制因素；当物体质量增大到临界点后，关节力矩极限成为限制平衡能力的决定因素。

### 4. 方法对比分析
*   **本质区别**：从传统的基于参考点（如ZMP）的准静态判断，转变为基于“全身动力学状态盆地”的预测式动态判断。
*   **创新贡献**：首次揭示并定量化了“临界质量”与“转换质量”的概念，为机器人承重能力的上限评估提供了物理准则。
*   **适用场景**：适用于具有明确载荷、要求精确稳定性约束的类人机器人搬运任务。

### 5. 实验分析
*   **验证方法**：使用ROBOTIS-OP3类人机器人进行模拟和实验，对比有无BSB约束下的提升效果。
*   **关键结果**：在BSB约束下的机器人能够成功平衡，而仅依靠传统CoP边界的方法在动态操纵中极易失败。实验证明了当质心轨迹接近BSB边界时，系统充分利用了自身的动态极限。
*   **优势/局限**：优势是提供了鲁棒的稳定性保证；局限是计算开销较大，目前离线生成BSB，实时应用需依赖预存储或简化处理。

### 6. 实用指南
*   **开源建议**：文中引用了ROBOTIS-OP3开源库（ROBOTIS-GIT）。建议开发者在实现时使用SQP优化算法（如SNOPT），并对BSB进行插值表存储。
*   **迁移迁移**：BSB方法论具备普适性，可迁移至四足机器人或其他高冗余自由度系统，只需重新推导对应的全身动力学回归矩阵。
*   **注意项**：需注意接触面摩擦锥约束（Friction Cone）的线性化处理，避免计算不收敛。

### 7. 总结
*   **核心思想**：基于全身动力学构建状态盆地，量化物体质量对平衡稳定性的非线性制约。
*   **速记版Pipeline**：
    1. 建立包含物体载荷的全身动力学模型；
    2. 求解极限平衡状态，绘制BSB边界图；
    3. 将BSB边界作为硬约束植入轨迹规划；
    4. 实时轨迹跟踪并验证稳定性边界。

**Key Findings:**

- In particular, the trade-off relationships between momentum regulation and limiting factors in balancing are characterized, introducing two key quantities of the object: the critical mass, at which the system's balancing capability is maximum, and the transition mass, which activates different limiting factors.
- In addition, sufficient conditions for imposing balanced states on a trajectory are established and implemented with BSBs as explicit threshold constraints in the whole-body trajectory optimization for stable object-lifting control of the humanoid, demonstrating the lift-and-hold and lift-and-release tasks with distinct mass properties in simulations and experiments.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29625v1)
- [arXiv](https://arxiv.org/abs/2607.29625v1)

---

<a id='2607.29622v1'></a>
## [RayViT: Ray-Conditioned Visual Representations for Viewpoint-Robust Imitation Learning](https://arxiv.org/abs/2607.29622v1)

**Authors:** Qian Wang, Longrui Chen, Peiran Sun, Aleksandar Taranovic, Niklas Freymuth, Ge Li, Weiran Liao, C. F. Maximilian Nagy, Yucheng Tan, Tao Chen, Gerhard Neumann

**Published:** 2026-07-31

**Categories:** cs.RO, cs.CV

**Abstract:**

Visual imitation learning enables robots to acquire visuomotor skills directly from images, yet RGB observations lack explicit geometric cues, making learned policies brittle to camera perturbations. To address this, we propose \textbf{Ray-conditioned Vision Transformer Encoder (RayViT)}, a lightweight architecture that injects camera geometry into pretrained ViT backbones. RayViT represents camera geometry as a Plücker ray map, patchifies it into ray features, and uses gated cross-attention to produce a ray-conditioned class token. These ray features are added as dense positional embeddings, while the ray class token replaces the original ViT class token to provide a geometry-aware summary representation. We combine this approach with an auxiliary cosine similarity loss to consistently improve the performance and robustness for geometry-aware tokens. Experiments on sim- and real-robot tasks demonstrate that RayViT improves robustness by approximately 13 percentage points under camera perturbations in multi-task RoboCasa benchmark and by 1.78 average completed stages in real-world multi-task success rate compared to baselines.

**Analysis:**

### 1. 摘要翻译
视觉模仿学习使机器人能够直接从图像获取视觉运动技能，但RGB观测缺乏显式的几何线索，导致学习到的策略对相机扰动非常脆弱。为了解决这一问题，我们提出了“射线条件视觉Transformer编码器”（RayViT），这是一种将相机几何结构注入预训练ViT骨干网的轻量级架构。RayViT将相机几何表示为Plücker射线图，将其分块（patchify）为射线特征，并利用门控交叉注意力机制生成射线条件类别标记（class token）。这些射线特征被添加为稠密的位置嵌入，而射线类别标记替换了原有的ViT类别标记，从而提供了一种感知几何的摘要表示。我们结合辅助余弦相似度损失，持续提升了几何感知标记的性能和鲁棒性。在模拟和真实机器人任务上的实验表明，在RoboCasa基准测试中，RayViT在相机扰动下的鲁棒性提升了约13个百分点，在真实世界多任务成功率上，平均完成阶段数较基线提升了1.78。

---

### 2. 方法动机分析
*   **驱动力**：旨在构建对相机视角变化具有鲁棒性的机器人模仿学习策略，克服RGB图像缺乏显式几何信息的问题。
*   **现有方法痛点**：
    *   **数据依赖**：现有方法过度依赖海量视角的多样性数据增强，训练成本高。
    *   **架构兼容性**：基于点云/3D表示的方法难以与现代强大的纯视觉预训练模型（如DINOv3）融合。
    *   **特征解耦**：简单拼接几何特征往往会破坏预训练ViT骨干网中已有的丰富语义分布。
*   **研究假设**：通过将像素级的Plücker射线嵌入注入预训练ViT，并配合针对几何标记的交叉视角一致性损失，可以在不损失原有视觉先验的情况下，显式赋予策略对视角的几何感知力。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **射线图建模**：利用相机内参和外参，为每个像素生成唯一的6维Plücker射线向量（方向+矩）。
    2.  **射线特征注入**：将射线图平均池化至patch分辨率，通过MLP映射后注入ViT。
    3.  **双重注入机制**：
        *   **补丁级注入**：将射线特征作为附加位置嵌入，直接叠加在图像patch token上，为视觉特征提供3D坐标基础。
        *   **类别级注入**：通过两个门控交叉注意力（Gated Cross-attention）模块，将射线信息聚合到一个可学习的查询token中，替换原始的[CLS]标记，作为几何感知的全局场景描述。
    4.  **辅助损失函数**：引入余弦相似度损失（$L_{cos}$），强制不同视角下的几何条件token在中间层趋于一致。
*   **关键点**：门控机制设计至关重要，它能自适应调节几何信息的影响权重，避免在注入几何知识时“淹没”原有的预训练语义特征。

---

### 4. 方法对比分析
*   **本质区别**：与需要点云重建或多视角增强的方法不同，RayViT属于“特征级注入”，即在保留ViT语义先验的基础上，将几何信息作为一种“辅助引导信号”。
*   **创新贡献**：
    *   提出了一种轻量级的Ray-conditioned CLS token。
    *   采用非参数化的平均池化来保持Plücker坐标的几何语义。
    *   证明了几何一致性目标需配合几何条件token才能生效。

---

### 5. 实验分析
*   **验证方法**：在RoboCasa 16项模拟任务及4项真实机器人任务中进行评测，设置“标准视角”与“受扰视角”对比。
*   **关键结论**：在相机扰动下，RayViT相比RGB基线有显著的性能跌幅降低（下降幅度仅为2.5% vs 18.5%）。
*   **优势**：不依赖深度传感器，不依赖大规模数据扩增，对各类预训练骨干网（DINOv3, EUPE）具有良好的可移植性。
*   **局限**：目前的门控机制主要针对ViT，对于卷积架构的适应性尚待验证。

---

### 6. 实用指南
*   **实现细节**：
    *   **核心模块**：重点实现Gated Cross-attention层，确保其维度与ViT隐藏层匹配。
    *   **超参数**：$\lambda$（余弦损失权重）默认为0.01；建议将损失应用于第10层附近的深层特征。
    *   **预处理**：确保相机内参和外参计算出的Plücker射线坐标在训练和推理时严格对齐图像patch。
*   **迁移建议**：该模块可以作为一个独立的“即插即用”组件，挂载在任何基于ViT的任务（如物体跟踪、视频生成）之前。

---

### 7. 总结
*   **核心思想**：通过Plücker射线注入与门控注意力实现几何增强视角鲁棒表征。
*   **速记版pipeline**：
    1.  计算像素级射线图作为几何参考；
    2.  将射线图切片并映射至Transformer隐藏维度；
    3.  通过门控机制融合射线特征到CLS标记；
    4.  叠加射线嵌入至patch流；
    5.  用交叉视角余弦损失拉近不同视角下的表征。

**Key Findings:**

- To address this, we propose \textbf{Ray-conditioned Vision Transformer Encoder (RayViT)}, a lightweight architecture that injects camera geometry into pretrained ViT backbones.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29622v1)
- [arXiv](https://arxiv.org/abs/2607.29622v1)

---

<a id='2607.29613v1'></a>
## [WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning](https://arxiv.org/abs/2607.29613v1)

**Authors:** Senyu Fei, Xiaopeng Yu, Siyin Wang, Xianzhong Zhao, Jingjing Gong, Xipeng Qiu

**Published:** 2026-07-31

**Categories:** cs.RO, cs.CL, cs.CV

**Abstract:**

Reinforcement learning (RL) post-training of Vision-Language-Action (VLA) models has shown strong promise for robotic manipulation. Among RL methods, critic-based approaches rely on a value estimator that predominantly operates on single-frame observations or single-frame VLM backbone latents, which is a fundamental mismatch with the partially observable nature of robot control. A naive approach to incorporate observation history into the critic incurs exponential complexity with high-dimensional visual space, and still fails because pure scalar-return regression provides insufficient supervision for learning cross-temporal dynamics. We identify the root cause as a state approximation problem: without an explicit world modeling objective, the critic's representation cannot capture the temporal structure needed for accurate value estimation. To address this, we propose the World Critic Model (WCM), built on a lightweight LeJEPA architecture; WCM jointly predicts future latent state and estimates values, such that the critic's representation is explicitly trained to capture temporal dynamics rather than merely regress scalar returns. WCM integrates seamlessly into both on-policy and off-policy training pipelines and is compatible with state-of-the-art VLA backbones including Pi0, Pi0.5, and OpenVLA-OFT. Extensive experiments on 149 tasks across four benchmarks demonstrate that WCM consistently achieves state-of-the-art performance in both in-distribution and out-of-distribution settings, with particularly strong generalization gains. We further validate WCM on seven real-world manipulation tasks using OpenVLA-OFT and Pi0.5 with off-policy RL, confirming stable deployment across diverse settings.

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作 (VLA) 模型在机器人操控领域的强化学习 (RL) 后训练中展现出巨大潜力。然而，现有的基于评判者 (Critic-based) 的 RL 方法主要依赖于基于单帧观察的价值估计，这与机器人控制中部分可观测的特性存在根本性的不匹配。为了解决这一问题，我们提出了**世界评判者模型 (World Critic Model, WCM)**。WCM 采用轻量级的 LeJEPA 架构，通过联合预测未来潜空间状态和估计价值，使评判者的表示能够显式地捕获环境的时间动力学结构，而不仅仅是回归标量收益。WCM 可无缝集成到各种训练流水线中，并兼容主流的 VLA 主干网络。在 149 项任务上的广泛实验表明，WCM 在分布内 (IND) 和分布外 (OOD) 设置下均达到最优性能，且在现实世界中表现出强大的泛化能力和稳定的部署效果。

### 2. 方法动机分析
*   **驱动力**：解决机器人控制中由于 POMDP（部分可观测马尔可夫决策过程）特性，导致的“单帧价值估计无法捕获动态时间信息”这一根本矛盾。
*   **现有方法痛点**：单纯依靠历史帧堆叠会带来指数级复杂度，且简单的标量价值回归无法提供足够的监督信号来学习复杂的时序动力学。
*   **研究假设**：如果将评判者的表示学习任务从“标量预测”扩展为“未来状态预测（世界模型）+ 价值估计”，评判者将能提取出包含时序动态信息的、更优的状态表示，从而提升 RL 效果。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入处理**：编码过去 $K$ 帧观察，并通过 CLIP 引入语言指令。
    2.  **特征提取**：利用 causal Transformer（世界预测器）对语言条件下的视觉序列进行处理，得到隐表示 $h_t$。
    3.  **双头预测**：
        *   **价值头**：输出状态价值 $\hat{V}_t$。
        *   **动态头**：进行“动作条件下的潜空间动态预测”，即预测未来时刻的 latent state $\hat{z}_{t+1}$。
    4.  **联合优化**：损失函数由三部分组成：价值回归损失 $\mathcal{L}_{value}$、未来状态预测损失 $\mathcal{L}_{pred}$（teacher-forcing）、以及防止维度坍塌的 SIGReg 正则化项。
*   **模型结构**：采用了轻量级的 LeJEPA 架构，将世界预测（辅助任务）与价值评估（核心任务）耦合。通过 Gated FiLM 块处理动作的影响，使状态表示动态化。
*   **算法核心**：通过让 Critic “预测未来”，强迫其理解当前动作对状态演变的影响，这种结构化的监督比纯标量监督更能逼近 POMDP 的充要状态统计量。

### 4. 方法对比分析
*   **本质区别**：从“被动接收观察进行评判”转变为“主动预测环境演变来辅助评判”。
*   **创新贡献**：提出将世界模型目标直接注入 Critic 架构中，无需额外的世界模型训练阶段，实现端到端的特征学习。
*   **适用场景**：适用于所有基于 VLA 的机器人控制任务，尤其是需要长程时序理解的任务（如长任务规划、动态物体操作）。

### 5. 实验分析（精简版）
*   **验证方法**：在 ManiSkill, MetaWorld, CALVIN, LIBERO-Plus 四大 benchmark 上进行了大量测试。
*   **关键结果**：在 OpenVLA-OFT 上实现了 252% 的性能提升；在极低初始性能（0.78%）的情况下，通过 WCM 快速收敛至 98% 以上。
*   **优势**：极强的泛化能力，尤其在 OOD 设置下，解决了传统 Critic 过拟合导致性能崩溃的问题。
*   **局限**：对长观测历史的收益边际递减；在极端非理想环境下的计算开销（虽轻量但仍有额外参数）。

### 6. 实用指南
*   **开源情况**：已开源，代码库地址见论文 GitHub（`https://github.com/sylvestf/WCM`）。
*   **实现细节**：
    *   **超参数**：$\lambda$（预测损失权重）建议设在 [0.3, 0.5] 之间。
    *   **历史长度**：实验表明 $K=3$ 通常是性能最优的折中点。
    *   **数据利用**：在 off-policy 训练中，务必包含失败样本（利用负奖励惩罚），这对于构建稳健的 Critic 至关重要。
*   **迁移可能**：可直接替换现有 PPO 或 Flow-SDE 类框架中的 MLP-Critic。

### 7. 总结
*   **核心思想**：通过引入预测动力学的辅助目标，强制 Critic 显式编码时序动态状态。
*   **速记版pipeline**：
    1.  编码历史帧序列。
    2.  用 Transformer 融合语义信息。
    3.  同时预测未来状态和当前价值。
    4.  联合损失函数进行反向传播更新。

**Key Findings:**

- To address this, we propose the World Critic Model (WCM), built on a lightweight LeJEPA architecture; WCM jointly predicts future latent state and estimates values, such that the critic's representation is explicitly trained to capture temporal dynamics rather than merely regress scalar returns.
- WCM integrates seamlessly into both on-policy and off-policy training pipelines and is compatible with state-of-the-art VLA backbones including Pi0, Pi0.5, and OpenVLA-OFT.
- Extensive experiments on 149 tasks across four benchmarks demonstrate that WCM consistently achieves state-of-the-art performance in both in-distribution and out-of-distribution settings, with particularly strong generalization gains.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29613v1)
- [arXiv](https://arxiv.org/abs/2607.29613v1)

---

<a id='2607.29567v1'></a>
## [TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware](https://arxiv.org/abs/2607.29567v1)

**Authors:** Hailing Hu, Mingyi Zhu, Yiquan An, Yifei Tian, Tianyou Zuo, Lifeng Zhou

**Published:** 2026-07-31

**Categories:** cs.RO

**Abstract:**

Manipulating transparent laboratory glassware that contains liquid is inherently safety-critical: even small geometric errors can cause unstable grasps and hazardous spillage. Although recent progress has been made in transparent object perception and robotic grasping, most existing systems optimize detection, depth reconstruction, and grasp planning independently, which leads to cross-stage inconsistency imperfect boundaries induce depth bleeding, distorted surfaces corrupt normal estimation, and task agnostic grasp scoring yields tilted or off-center grasps that fail under dynamic motion. In this paper, we propose TransGraspNet, a geometry physics consistent framework that explicitly enforces consistency from perception to execution through three coupled principles: boundary consistency to produce structurally reliable object contours as downstream priors, surface consistency to preserve geometric fidelity and surface normal accuracy during depth reconstruction, and physics consistency to refine grasp selection with centroid alignment and wrench-space stability for upright and dynamically robust manipulation. We evaluate TransGraspNet on public benchmarks, a dedicated transparent glassware dataset, and a real robotic platform. The results show improved boundary quality and surface normal fidelity, and demonstrate strong task-level performance in cluttered transparent scenes. Most importantly, the proposed system achieves reliable real-world operation, including high grasp success rates in clutter and zero spillage during high speed liquid transport, highlighting the effectiveness of our method.

**Analysis:**

### 1. 摘要翻译
透明实验室器皿及其内部液体的操纵具有高度的安全性挑战：微小的几何误差即可能导致抓取不稳定甚至液体泄漏。尽管在透明物体感知和机械臂抓取领域已有研究进展，但现有的系统多采用解耦的级联流水线，导致跨阶段几何不一致：感知到的不完美边界会引起深度出血（depth bleeding），失真的表面会导致法线估计偏差，而与任务无关的抓取评分机制则常导致倾斜或偏离中心的抓取，无法适应动态抓取过程。本文提出了TransGraspNet，这是一个几何-物理一致性框架，通过三个核心原则显式地实施从感知到执行的一致性：1) 边界一致性：生成结构可靠的物体轮廓作为下游先验；2) 表面一致性：在深度重建中维持几何保真度和表面法线准确性；3) 物理一致性：通过质心对齐和扳手空间（wrench-space）稳定性分析优化抓取策略，实现动态稳健的操纵。在公开基准及实验室机器人平台上的实验表明，该系统在杂乱环境下实现了高成功率及高速液体运输中的零泄漏，验证了其在安全关键实验室自动化任务中的实用性。

### 2. 方法动机分析
*   **驱动力**：解决透明物体在复杂光学环境下的“视觉不可见”问题，以及传统流水线中感知与规划阶段脱节导致的物理执行不稳定问题。
*   **现有方法痛点**：感知（边界误差）、重建（深度溢出）与规划（局部几何特征未考虑任务物理约束）各模块独立优化，误差级联导致抓取动作在动态环境下失效。
*   **研究假设**：通过在整个处理链路中引入“几何-物理一致性”约束，可以抑制跨阶段的误差传播，从而实现针对特定任务（如实验室液体处理）的稳健抓取。

### 3. 方法设计详解
TransGraspNet的核心流程如下：
1.  **感知阶段 (Edge-Guided Boundary)**：引入E-CBAM模块，通过空间与通道注意力机制强化边界特征。利用轻量级边界分支（Edge Branch）与掩码分支融合，生成精确的轮廓，作为深度恢复的显式先验，有效抑制了背景干扰。
2.  **重建阶段 (Geometry-Aware Depth)**：提出Edge-Guided Attention Gate (EGAG) 调节多模态融合，并在MGR损失函数中引入深度与法线一致性约束，防止跨边界的深度信息污染。
3.  **抓取优化阶段 (Geometry-Physics Refinement)**：
    *   **几何项**：通过PCA计算中心点与主轴，强制执行径向、角向及质心对齐。
    *   **物理项**：评估反向接触条件（Antipodal）与扳手空间（Wrench-space）的稳健性，确保抓取能够抵抗重力及惯性扰动。
    *   **评分策略**：通过线性回归拟合成功率，对候选抓取动作进行重排序。

### 4. 方法对比分析
*   **本质区别**：与ClearGrasp等仅优化几何重建的方案不同，TransGraspNet将“抓取成功”这一物理任务作为目标函数，显式地将动力学稳定性约束注入优化过程。
*   **创新贡献**：设计了耦合边界感知、法线保持重建与扳手空间评分的闭环框架，特别适用于需要维持物体直立的实验室高精度抓取场景。
*   **适用场景**：高反光、透明、且对操作姿态有严格要求（如防止液体溢出）的实验室自动化任务。

### 5. 实验分析（精简版）
*   **关键结果**：在RobotSci-Glass数据集上，法线误差从15.2°显著降低至8.4°，真实机器人平台在杂乱环境下的抓取成功率达到86%，并实现了高速液体运输的“零泄漏”。
*   **主要优势**：极强的稳健性和抗干扰能力，对实验室常见器皿（烧杯、试管）具有普适性。
*   **主要局限**：评分机制依赖预定义的物理参数（如扳手空间），对于非常规形状的透明物体可能需要微调超参数。

### 6. 实用指南
*   **实现细节**：在训练深度重建模块时，务必使用“部分冻结策略（Partial Freezing Strategy）”，仅微调EGAG模块以保留预训练权重中的基础知识；抓取评分模块无须端到端训练，采用离线线性回归拟合即可，大大降低了计算开销。
*   **迁移建议**：该几何-物理一致性框架可直接迁移至其他需要精密操作的任务中，只需更改扳手空间分析的物理模型参数（如根据不同负载调整约束）。

### 7. 总结
*   **核心思想**：通过跨阶段一致性约束，将几何感知与物理稳定性分析融合至全链路抓取框架。
*   **速记版pipeline**：
    1.  用双流网络提取边缘增强的物体掩码。
    2.  利用感知到的轮廓引导多模态深度重建。
    3.  结合器皿质心与物理稳定性对抓取动作排序。
    4.  执行考虑动态惯性的闭环抓取任务。

**Key Findings:**

- In this paper, we propose TransGraspNet, a geometry physics consistent framework that explicitly enforces consistency from perception to execution through three coupled principles: boundary consistency to produce structurally reliable object contours as downstream priors, surface consistency to preserve geometric fidelity and surface normal accuracy during depth reconstruction, and physics consistency to refine grasp selection with centroid alignment and wrench-space stability for upright and dynamically robust manipulation.
- The results show improved boundary quality and surface normal fidelity, and demonstrate strong task-level performance in cluttered transparent scenes.
- Most importantly, the proposed system achieves reliable real-world operation, including high grasp success rates in clutter and zero spillage during high speed liquid transport, highlighting the effectiveness of our method.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29567v1)
- [arXiv](https://arxiv.org/abs/2607.29567v1)

---

<a id='2607.29513v1'></a>
## [Homotopy-Aware Corridor Generation without Predefined Reference Paths](https://arxiv.org/abs/2607.29513v1)

**Authors:** Haoze Dong, Minghan Li, Meng Guo, Zhongkui Li

**Published:** 2026-07-31

**Categories:** cs.RO

**Abstract:**

Generating safe corridors is essential for collision-free robotic motion planning, yet most existing methods rely on predefined reference paths, which bias corridor geometry and implicitly limit the homotopy classes that can be explored. We propose a reference-path-free corridor generation framework on graphs of convex sets (GCS) that constructs corridors directly as sequences of convex sets, allowing corridor structure to emerge from the free-space representation rather than from a guiding path. To reason about similarity among corridors, we extend visibility-based deformation from paths to convex-set sequences, enabling the fusion of topologically redundant corridors while preserving distinct alternatives. To overcome the limited adaptability of existing GCS methods based on static global decompositions, we further develop an adaptive multi-scale GCS, in which a sampling-based fine-scale graph supports localized updates and a visibility-based coarse-scale graph enables compact global exploration. The two levels maintain topological consistency, allowing incremental updates without full graph reconstruction under environmental uncertainty. Numerical experiments characterize GCS construction, corridor generation, homotopy-aware exploration, and local updates, showing efficient graph construction, stable trajectory-level performance, and shorter-duration homotopy-aware trajectories than existing baselines. Hardware experiments on ground and aerial robots, including deployment with onboard localization, further validate the framework under translated and previously unknown obstacles.

**Analysis:**

### 1. 摘要翻译
生成安全走廊对于无碰撞机器人运动规划至关重要，但大多数现有方法依赖于预定义的参考路径，这会限制走廊的几何形状并隐式地限制了可探索的同伦类。我们提出了一种基于凸集图（GCS）的无参考路径走廊生成框架，该框架直接将走廊构建为凸集序列，使走廊结构从自由空间表示中自然涌现，而非依赖于引导路径。为了推断走廊间的相似性，我们将基于可见性的变形（VD）从路径扩展到凸集序列，实现了在保留不同替代方案的同时融合拓扑冗余走廊。为克服现有基于静态全局分解的GCS方法的适应性局限，我们进一步开发了自适应多尺度GCS，其中采样驱动的细尺度图支持局部更新，而基于可见性的粗尺度图实现了紧凑的全局探索。这两个级别保持了拓扑一致性，允许在环境不确定性下进行增量更新，而无需进行全图重构。实验验证了该框架在地面和空中机器人上的性能，在处理移动和未知障碍物方面具有显著优势。

### 2. 方法动机分析
- **驱动力**：解决基于路径的走廊生成（如通过膨胀路径）所带来的“几何和拓扑先验偏见”，旨在实现从自由空间结构中直接涌现走廊，以获得更优的路径规划自由度。
- **痛点**：现有方法严重依赖预计算的参考路径，这限制了在复杂多拓扑环境中对潜在解空间的探索，且当路径局部异常时，走廊生成极为敏感。
- **研究假设**：通过在凸集图（GCS）上直接进行走廊搜索和基于可见性的走廊相似性度量，可以系统性地推断走廊层面的拓扑多样性。

### 3. 方法设计详解
- **多尺度GCS结构**：
  - **细尺度（F-GCS）**：捕获高保真度局部几何，作为局部更新的基础。通过采样无碰撞配置并扩展为凸集（轴对齐超立方体）。
  - **粗尺度（C-GCS）**：通过选择“互不可见根节点”并执行BFS聚合构建，形成对全局连通性的紧凑抽象，大幅降低搜索空间。
- **走廊生成逻辑**：
  - 不再寻找路径，而是直接在C-GCS上搜索凸集序列（SCS）。
  - **UVD（统一可见性变形）融合**：通过提出的定理2，对多条候选SCS进行相似性判定。若满足“单调嵌入”和“交集对齐”条件，则将它们融合，以去除拓扑冗余。
- **局部更新机制**：当感知到障碍物更新（$\Delta O$）时，仅在受影响区域（R）内局部重构F-GCS，并级联更新受影响的C-GCS节点，无需全图重构。

### 4. 方法对比分析
- **本质区别**：从“路径启发式走廊”转变为“基于空间结构的走廊直接发现”。
- **创新点**：
  1. 走廊层面的VD/UVD定义，为走廊聚类提供了数学基础。
  2. 耦合细/粗尺度的自适应GCS，兼顾了局部 fidelity 和全局搜索效率。
- **适用场景**：复杂动态环境、多拓扑场景、对计算实时性有要求的机器人导航任务。

### 5. 实验分析
- **关键结果**：在复杂多拓扑环境中，相比基线（VCC, R-IRIS），该方法在路径耗时上减少了12%–20%，且生成的图结构更加稀疏，搜索效率更高。
- **优势**：无需参考路径预设，对于移动障碍物的局部更新速度快（比重构快近20倍）。
- **局限**：在高维空间中，虽然有粗尺度图简化，但随着环境复杂度和维度增加，凸集扩展的计算开销仍需优化。

### 6. 实用指南
- **开源地址**：[https://github.com/HauserDong/path-free-gcs-corridors](https://github.com/HauserDong/path-free-gcs-corridors)
- **迁移建议**：本方法的核心在于“图的构建”与“凸集序列搜索”，可轻松迁移到任何支持凸集优化的轨迹规划任务中。关键点在于针对不同动力学模型（如四旋翼与地面机器人）调整凸集膨胀的超参数 $\varepsilon$。

### 7. 总结
- **核心思想**：直接在多尺度凸集图上搜索并融合空间走廊，而非路径驱动。
- **速记版Pipeline**：
  1. **构建多尺度图**：根据环境离线建立细/粗双层图结构。
  2. **图上搜走廊**：在粗图上搜索获取多个拓扑独立的凸集序列。
  3. **融合冗余走廊**：利用UVD准则合并相似的序列以精简候选集。
  4. **局部自适应更新**：发生障碍物变化时，仅局部修补图并更新受影响的走廊。

**Key Findings:**

- We propose a reference-path-free corridor generation framework on graphs of convex sets (GCS) that constructs corridors directly as sequences of convex sets, allowing corridor structure to emerge from the free-space representation rather than from a guiding path.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29513v1)
- [arXiv](https://arxiv.org/abs/2607.29513v1)

---

<a id='2607.29482v1'></a>
## [Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration](https://arxiv.org/abs/2607.29482v1)

**Authors:** Dylan Miller, Martin Jagersand

**Published:** 2026-07-31

**Categories:** cs.RO

**Abstract:**

By relying on independent couplings from uninformative Gaussian priors, standard diffusion and flow matching models are forced to learn complex, high-cost vector fields to reach the physical action space. Generative models excel at capturing multimodal behaviors for robotic Learning from Demonstration (LfD), but often suffer from high inference cost. This paper introduces Temporal Policy, a generative framework based on stochastic interpolants that formulates action generation as a temporally coupled transport problem. By initializing the generative flow at the robot's recent history, we explicitly couple past states to future action sequences. This data-dependent coupling reduces transport cost and produces straight vector fields. We validate Temporal Policy across visuomotor simulation benchmarks and on a physical Barrett WAM 2x 7DoF teleoperation platform. Our approach reduces transport costs by nearly an order of magnitude compared to noise-initialized baselines, achieving a 19.1 ms inference latency on a single NVIDIA RTX 4080. Crucially, these geometric and computational efficiencies are achieved while matching the success rates of state-of-the-art baselines. This simplified transport geometry bypasses the computational bottleneck of independent Gaussian priors, helping enable high-frequency, closed-loop control. The code is publicly available at https://github.com/dmiller12/TemporalPolicy.

**Analysis:**

### 1. 摘要翻译
生成式模型在机器人“学习从演示”（LfD）任务中表现出色，但常受限于高推理成本。现有的扩散模型（Diffusion）和流匹配（Flow Matching）模型依赖于从独立高斯先验中采样，被迫学习复杂的、高成本的向量场来映射到物理动作空间。本文提出了“时间策略”（Temporal Policy），这是一种基于随机插值（Stochastic Interpolants）的生成框架，将动作生成重构为一种时间耦合的传输问题。通过将生成流的初始点设置在机器人的近期历史状态，我们明确地将过去的状态与未来的动作序列耦合。这种数据依赖的耦合方式显著降低了传输成本，并产生了平直的向量场。实验表明，该方法在仿真基准和7自由度物理遥操作平台上，将传输成本降低了近一个数量级，在单张NVIDIA RTX 4080上实现了19.1ms的推理延迟，同时保持了最先进的成功率。

---

### 2. 方法动机分析
- **驱动力**：旨在解决生成式策略在机器人实时闭环控制中面临的“高延迟”与“计算资源瓶颈”问题。
- **现有方法痛点**：标准生成式模型（扩散模型/流匹配）将生成过程定义为从高熵、不相关的噪声分布（高斯分布）到结构化动作空间的映射。这种“无中生有”的初始方式导致了巨大的几何鸿沟，迫使网络学习高曲率的复杂向量场，从而需要大量的推理步骤。
- **研究假设**：通过利用机器人动作在时间上的强连续性，将“噪声”替换为“近期历史状态”，可以将生成任务简化为从历史状态到未来动作的短距离、低曲率传输问题。

---

### 3. 方法设计详解
- **核心流程**：
  1. **状态耦合**：将生成任务建模为从源分布（历史状态 $s_{t-H+1:t}$）到目标分布（未来动作序列 $s_{t-H+1+d:t+d}$）的映射。
  2. **随机插值**：定义概率路径 $x_\lambda = (1-\lambda)x_0 + \lambda x_1 + \epsilon(1-\lambda)w_\lambda$，其中 $x_0$ 为历史，$x_1$ 为动作，$w_\lambda$ 为维纳过程噪声。
  3. **漂移回归**：训练一个1D U-Net网络，以最小化漂移预测误差 $L(\theta) = \|b_\theta(x_\lambda, x_0, \lambda, c) - u_\lambda\|_2^2$，其中 $u_\lambda$ 是由插值导出的确定性漂移目标。
  4. **推理求解**：在推理时，利用漂移场通过ODE/SDE求解器从 $x_0$ 演化至 $x_1$。由于初始化就在目标附近，仅需极少步数（如10步）即可收敛。
- **模型结构**：包含一个ResNet-18观察编码器（提取特征）和一个1D U-Net（预测 drift），通过FiLM层将时间信息 $\lambda$ 和上下文信息 $c$ 注入模型。

---

### 4. 方法对比分析
- **本质区别**：将“噪声源”更换为“历史状态源”，从根本上将问题性质从“去噪生成”转变为“条件轨迹演化”。
- **创新贡献**：引入了数据依赖的耦合机制，不仅降低了推理步数（NFE），还通过分析法实现了对分数的精确恢复，使得单模型支持确定性ODE和随机SDE两种采样方式。
- **适用场景**：适用于对实时性要求极高的机器人操纵任务，尤其是涉及长序列动作生成的场景。

---

### 5. 实验分析
- **关键结果**：在Robomimic仿真基准上，保持了与Diffusion Policy相当的成功率，但参数量大幅削减（从255M降至17M），NFE从100降至10，推理延迟降至19.1ms。
- **优势**：极高的采样效率；不仅降低了延迟，还提升了数据利用率（小样本表现更好）。
- **局限**：对初始化质量敏感，如果历史观测中存在较大噪声，可能会偏离目标轨道；此外，对于某些特定任务（如Tool Hang）表现出对初始化精度的依赖。

---

### 6. 实用指南
- **开源情况**：已开源，GitHub地址：`dmiller12/TemporalPolicy`。
- **实现细节**：关键在于定义合理的“历史窗口”与“动作窗口”的重叠度（$d$）；训练时使用AdamW优化器和余弦调度。
- **迁移可能**：非常适合迁移到需要高频闭环控制的任务，如无人机轨迹生成、高速机械臂抓取等。迁移时需确保动作空间与状态空间的结构等价性。

---

### 7. 总结
- **核心思想**：以历史状态作为生成起点，将动作生成简化为短距离的轨迹平滑演化。
- **速记版pipeline**：
  1. 捕获当前机器人动作历史；
  2. 将历史作为生成模型的初始状态；
  3. 通过学习好的漂移场进行少步数值积分；
  4. 输出预测的未来动作序列。

**Key Findings:**

- Our approach reduces transport costs by nearly an order of magnitude compared to noise-initialized baselines, achieving a 19.1 ms inference latency on a single NVIDIA RTX 4080.
- Crucially, these geometric and computational efficiencies are achieved while matching the success rates of state-of-the-art baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29482v1)
- [arXiv](https://arxiv.org/abs/2607.29482v1)

---

<a id='2607.29237v1'></a>
## [CorrelationFlow: A Training-Free Geometric Approach for LiDAR Scene Flow Estimation](https://arxiv.org/abs/2607.29237v1)

**Authors:** Minh-Quan Dao, Yancong Lin, Julie Stephany Berrio Perez, Holger Caesar

**Published:** 2026-07-31

**Categories:** cs.CV, cs.RO

**Abstract:**

LiDAR scene flow estimation has settled into a monoculture: nearly all recent methods share the same feed-forward architecture and the same family of self-supervised losses, inheriting each other's assumptions, and each other's blind spots. When those assumptions fail, as they do for sparse, distant, or fast-moving objects, every method built on them fails together, and adding parameters or simulated training data does not fix what the formulation itself gets wrong. This paper takes the opposite path. We present CorrelationFlow, a training-free geometric framework that reduces scene flow to two textbook operations: connected-component labeling and correlation maximization on bird's-eye-view occupancy images. Objects are isolated as spatio-temporal connected components, their motions recovered as correlation peaks, and the resulting velocities propagated to all member points. However, this dense correlation evaluates every candidate displacement of every cluster and requires a window of past sweeps; therefore, we develop a sparse counterpart that operates on a single sweep pair by matching lightweight occupancy descriptors at boundary key points. Because nothing is trained, nothing is inherited: on the multi-domain test set of the Argoverse 2 2026 Scene Flow Challenge, spanning five datasets with heterogeneous sensors and platforms, CorrelationFlow ranked second among unsupervised methods and degrades most gracefully at long range, where the shared assumptions of learned methods break down. Our results suggest that a substantial share of the scene flow problem is solvable by classical computer vision, and that progress may require questioning the formulation, not scaling it.

**Analysis:**

这是一份针对论文《CorrelationFlow: A Training-Free Geometric Approach for LiDAR Scene Flow Estimation》的深度解析。

### 1. 摘要翻译
LiDAR场景流估计已陷入“单一种类”的困境：主流方法依赖相同的全监督学习架构与损失函数，导致在稀疏、远距离或高速运动场景下表现集体失效。本文提出了**CorrelationFlow**，这是一个完全不需要训练的几何框架，将场景流计算简化为两种经典视觉运算：鸟瞰图（BEV）中的连通分量标记与互相关最大化。我们通过空间-时间连通分量孤立物体，通过互相关峰值恢复其运动。此外，我们开发了一个基于关键点的变体，无需聚类即可实现匹配。在Argoverse 2 2026场景流挑战赛中，CorrelationFlow在无监督赛道排名第二，且在远距离表现出极高的稳健性，证明了经典计算机视觉在场景流问题上的巨大潜力。

### 2. 方法动机分析
*   **驱动力**：作者挑战了“必须依赖大规模标注数据进行深度学习”的范式，探索几何方法在场景流任务中的极限。
*   **现有痛点**：基于深度学习的方法（尤其是监督学习）对数据极其依赖、领域迁移能力弱，且难以处理远距离稀疏点云；测试时优化（Test-time optimization）方法则因计算量过大难以在实时任务中使用。
*   **核心假设**：动态物体在短时间内的点云外观近似不变；通过投影到BEV空间，3D位移可简化为2D平面的平移匹配，从而通过计算互相关（NCC）实现精确估计。

### 3. 方法设计详解
CorrelationFlow的核心处理流程如下：
1.  **预处理与投影**：对LiDAR扫描进行自车运动补偿，将其投影到BEV网格，转换为二值化的 occupancy map。
2.  **时空聚类（关键创新）**：通过将多帧点云聚合，利用形态学膨胀操作桥接因稀疏性断裂的物体部件，再使用经典的连通分量算法提取物体簇。这一步无需密度聚类算法（如DBSCAN），计算复杂度仅与图像分辨率相关，具有极高的效率和尺度鲁棒性。
3.  **运动估计（NCC最大化）**：
    *   **水平运动**：将目标物体在BEV投影下的二值图像视为模板，在下一帧图像中寻找互相关得分（NCC）最高的平移量，从而获得（$f_x, f_y$）。
    *   **垂直运动**：利用物体在BEV中估算的航向角进行旋转对齐，投影到侧视图（X-Z plane），利用相同逻辑求出垂直位移 $f_z$。
4.  **关键点变体**：为了规避聚类带来的误差传播，作者提出CorrelationFlow-Keypoints，直接在BEV边界提取关键点，利用局部patch特征计算NCC，通过比率测试（Lowe's ratio test）剔除错误匹配，最后通过连通分量传播流场。

### 4. 方法对比分析
*   **本质区别**：摒弃了点对点的回归训练，转而回归到经典的“特征匹配与几何对齐”问题。
*   **创新贡献**：提出了一种基于BEV连通分量的时空聚类方案，以及将场景流分解为解耦的2D/3D相关性最大化求解器，实现了无需任何训练参数的工业级精度。
*   **适用场景**：自动驾驶场景，尤其是传感器种类多变的 heterogeneous 多平台部署，以及对远距离物体检测精度要求较高的场景。

### 5. 实验分析
*   **关键结果**：在Argoverse 2数据集中，CorrelationFlow在长距离（35-70m）表现远优于所有基于深度学习的 baseline。
*   **核心优势**：极强的领域泛化性（无需训练）、远距离鲁棒性、推理速度快且可并行化。
*   **主要局限**：基于刚性运动假设，无法捕捉非刚性形变；若ego-motion补偿出现较大误差，会直接导致流场估计偏移。

### 6. 实用指南
*   **开源与复现**：该方法本质是几何运算，实现上主要依赖 `scipy.ndimage` 等基础库，核心在于 `connected components` 和 `NCC` 的实现。
*   **迁移与使用**：该逻辑可直接迁移到任何支持BEV投影的传感器任务中（如雷达、超声波点云）。建议在处理稀疏场景时引入关键点匹配策略。

### 7. 总结
*   **核心思想**：几何对齐优先于黑盒学习，回归经典视觉相关性求解。
*   **速记版Pipeline**：
    1. **去畸变**：补偿自车运动。
    2. **投影与聚合**：生成BEV二值图并进行形态学桥接。
    3. **连通分量分析**：快速识别运动物体簇。
    4. **互相关搜索**：通过NCC计算物体平移。
    5. **传播**：将物体级运动赋予所有点，生成稠密流场。

**Key Findings:**

- We present CorrelationFlow, a training-free geometric framework that reduces scene flow to two textbook operations: connected-component labeling and correlation maximization on bird's-eye-view occupancy images.
- However, this dense correlation evaluates every candidate displacement of every cluster and requires a window of past sweeps; therefore, we develop a sparse counterpart that operates on a single sweep pair by matching lightweight occupancy descriptors at boundary key points.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29237v1)
- [arXiv](https://arxiv.org/abs/2607.29237v1)

---

<a id='2607.29231v1'></a>
## [TacPrint: A Wearable Fingertip Tactile Sensor for Human-to-Robot Contact Reproduction](https://arxiv.org/abs/2607.29231v1)

**Authors:** Yongxi Liu, Chaofan Zhang, Xingyu Zhang, Xiangyin Bao, Boyue Zhang, Shaowei Cui, Shuo Wang

**Published:** 2026-07-31

**Categories:** cs.RO

**Abstract:**

Human-centric data collection is emerging as a significant paradigm for robot skill acquisition, but seamlessly integrating low-cost, scalable tactile sensing systems that capture fine-grained fingertip interactions without compromising natural operation remains a key challenge. This reduces the reliability of human-to-robot transfer in contact-rich tasks. In this work, we present TacPrint, a wearable fingertip tactile sensor, where protrusions on the inner surface of the silicone skin are aligned one-to-one with 24 capacitive taxels to enable localized capacitive responses. A real-to-sim-to-real pipeline estimates a 35 $\times$ 26 contact-depth map from 24-channel capacitive signals. Against simulation-generated labels, the model achieved a contact-region RMSE of 0.223 $\pm$ 0.161 mm, a weighted-centroid error of 1.213 $\pm$ 2.379 pixels, and an IoU of 0.829 $\pm$ 0.169. With measured capacitive inputs, the network-predicted depth evaluated at the guide-calibrated contact center showed a mean absolute error of 0.085 $\pm$ 0.057 mm across all 40 controlled trials, while the mean contact-position error was 0.250 $\pm$ 0.208 mm across the 37 trials whose reference contact regions were not truncated by the sensing boundary. In human-to-robot replay, tactile-guided compensation increased grasping and wiping success rates from 0% to 91.67% and 90%, respectively. In closed-loop grasping, dense-depth feedback achieved success rates of 87.5% over all tested positions and 85% under edge-contact conditions, compared with 67.5% and 45% for raw-taxel feedback.

**Analysis:**

### 1. 摘要翻译
以人为中心的数据采集是机器人技能习得的重要范式，但低成本、可扩展且能捕捉精细指尖交互的触觉传感系统的无缝集成仍是关键挑战。这降低了接触丰富任务中人机转移的可靠性。本文提出了TacPrint，一种可穿戴指尖触觉传感器，其硅胶皮肤内表面的突起与24个电容式感测点（taxels）一一对齐，实现了局部电容响应。本文采用实-仿-实（real-to-sim-to-real）流程，从24通道电容信号估计出35×26的接触深度图。实验表明，该传感器显著提高了人机操作复现的成功率，并在闭环抓取任务中表现优异。

### 2. 方法动机分析
*   **驱动力**：在人机数据采集（如示教）中，仅凭视觉和运动学难以精确获取接触状态，导致机器人难以复现接触丰富的精细操作任务。
*   **现有痛点**：现有的可穿戴传感器难以同时兼顾结构紧凑、低成本、对人类指尖的适配性以及对机器人操作任务提供高信息量的接触表示。
*   **核心直觉**：通过在指尖皮肤设计与电容传感器对齐的几何突起，可以将微小的接触变形转化为可学习的局部压力分布，进而通过学习模型映射为高分辨率的密度接触深度图。

### 3. 方法设计详解
*   **流程总结**：
    1.  **传感器制造**：使用Mold Star 20T硅胶制作包含24个半球形突起的皮肤，通过3D打印支持件固定，并使用Velcro绑带实现对人类手指的快速适配。
    2.  **触觉数据采集**：将传感器安装在XYZ线性导轨上进行受控压印，获取电容响应序列。
    3.  **标签生成（Simulation）**：利用TacFlex物理引擎后端（Isaac Gym/Flex），根据压印的物理几何参数复现接触过程，生成真值深度标签。
    4.  **学习 pipeline**：采用包含LSTM编码器（捕获时序上下文）和空间解码器（卷积神经网络）的端到端架构，将输入 $X_t \in \mathbb{R}^{T \times 24}$ 映射为 $Y_t \in \mathbb{R}^{35 \times 26}$ 的深度分布。
*   **模型结构**：LSTM单元负责处理传感器随时间变化的动态信号，通过投影层进入空间解码器，解码器通过 transposed-convolution 还原接触的空间拓扑特征。
*   **算法说明**：训练损失函数引入了 foreground-weighted 机制（公式6），通过设置权重 $w_{tij}$，强化模型对接触区域（foreground）的预测准确度，削弱背景干扰。

### 4. 方法对比分析
*   **本质区别**：TacPrint并非单纯的压力感测，而是通过“学习+仿真”的管线，将稀疏的传感器输入重构为稠密的几何空间接触图，实现了从“信号”到“物理理解”的飞跃。
*   **创新贡献**：提出了一种低成本（约50美元）、即插即用、且具备明确几何含义（深度图）的触觉表示方法。
*   **适用场景**：人机协同数据的采集、需要精确力反馈的机器人接触式装配、擦拭等闭环操作任务。

### 5. 实验分析
*   **验证方法**：通过受控的三角压头进行定量测试，并在抓取水果和擦拭白板任务中进行定性应用。
*   **关键结果**：在模拟标签下IoU达到0.829；在人机复现任务中，通过触觉引导补偿，抓取成功率从0%提升至91.67%。
*   **局限**：24个感测点的输入限制了对复杂接触面边界形状的重构能力；当前验证主要基于受控环境，未充分考虑极端复杂形变下的鲁棒性。

### 6. 实用指南
*   **开源与实现**：核心基于TacFlex框架（参考论文[23]），需重点配置Isaac Gym仿真环境来生成高质量标签。
*   **实现细节**：数据预处理阶段需对接触深度图进行全局缩放标准化（$D_{max}=3$ mm），训练时设置 $\lambda=0.5$ 以平衡平方误差和绝对误差。
*   **迁移建议**：该方法可迁移至任何具备阵列式触觉反馈的柔性皮肤传感器，只需替换结构参数并调整仿真模型。

### 7. 总结
*   **核心思想**：通过定制化柔性传感阵列与仿真学习，将指尖触觉转化为高精度接触深度图。
*   **速记pipeline**：
    1. 制作指尖传感器贴片；
    2. 仿真生成接触几何标签；
    3. 训练时序LSTM-CNN网络；
    4. 根据预测结果实时调整机器人末端轨迹。

**Key Findings:**

- In this work, we present TacPrint, a wearable fingertip tactile sensor, where protrusions on the inner surface of the silicone skin are aligned one-to-one with 24 capacitive taxels to enable localized capacitive responses.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.29231v1)
- [arXiv](https://arxiv.org/abs/2607.29231v1)

---

