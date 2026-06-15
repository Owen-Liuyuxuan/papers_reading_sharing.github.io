time: 20260615

# Arxiv Computer Vision Papers - 2026-06-15

## Executive Summary

以下是对2026年6月12日arXiv上10篇计算机视觉相关论文的执行摘要，旨在帮助研究人员快速把握领域内最新进展。

---

### 1. 主要主题与趋势概述

这组论文高度集中于**具身智能与机器人操作**，尤其强调**视觉-语言-动作（VLA）模型**、**扩散模型用于策略学习**，以及**因果与记忆增强的模仿学习**。具体趋势包括：

- **扩散模型从图像生成迁移到策略生成**：多篇论文（如Spatially Conditioned Diffusion Policy、HPSv3++）将扩散过程直接用于动作序列预测或奖励建模，提升了鲁棒性和精度。
- **语言引导的操控**：AERMANI-PLACE和Hy-Embodied-0.5-VLA利用自然语言指令实现物体放置或任务分解，反映了多模态融合的深入。
- **机器人与人类交互的演示数据高效收集**：EgoGuide、TRACE等关注从人类演示中高效学习，减少对机器人硬件的依赖，或处理观测延迟问题。
- **开源平台与标准化评估**：ORCA提供了灵巧操作研究的开源平台，StereoGeo贡献了端到端立体标定方法，体现了社区对可复现性与基础工具的重视。

### 2. 特别重要或创新的论文

- **Instruct-Particulate**：提出前馈式3D物体关节控制，通过运动学约束实现可泛化的关节操作，无需优化迭代，在速度与泛化性上显著优于现有方法。
- **HPSv3++**：将奖励模型扩展到扩散模型的全能力谱（从基础到高级生成），首次证明奖励模型可以像语言模型一样通过缩放数据与模型容量持续提升，对扩散模型的对齐训练有重大意义。
- **TRACE**：引入“轨迹路由因果记忆”机制，解决视觉模仿学习中观测延迟与因果混淆问题，在需要长期记忆的任务上表现突出，为因果建模在机器人学习中的应用提供了新思路。
- **Hy-Embodied-0.5-VLA**：将VLA模型（视觉-语言-动作）从仿真迁移到真实机器人，并构建了一套完整的“学习栈”，包括数据收集、策略训练与部署，是具身智能从研究到落地的典型范例。

### 3. 新兴研究方向与技术

- **扩散策略的条件化**：Spatially Conditioned Diffusion Policy探索以单目RGB图像作为条件，生成精确且鲁棒的操作轨迹，降低了多传感器依赖，或将推动低成本机器人部署。
- **空中操控与语言结合**：AERMANI-PLACE展示了无人机（空中机械臂）在自然语言指引下执行放置任务，拓展了具身智能的空间维度（由地面到空中）。
- **灵敏度塑造用于潜在建模**：Sensitivity Shaping提出一种在潜在空间中调控模型对输入扰动敏感度的方法，可提升生成模型或表征学习的稳定性与泛化能力，属于理论驱动的新方向。
- **端到端立体标定**：StereoGeo摒弃传统分步标定，用几何感知网络直接输出相机参数，简化了标定流程，对自动驾驶和AR/VR有实用价值。

### 4. 建议全文阅读的论文

- **Instruct-Particulate**：若关注3D物体关节操作或机器人操控基础能力，该文创新性高且方法简洁。
- **HPSv3++**：对扩散模型对齐、奖励工程或大规模生成模型感兴趣者必读，其缩放定律发现可能影响未来研究范式。
- **TRACE**：适合从事模仿学习、因果推理或长期任务规划的读者，其记忆机制设计具有启发性。
- **Hy-Embodied-0.5-VLA**：面向希望将VLA模型部署到真实机器人系统的工程师和研究者，文中提供的工程细节极具参考价值。
- **ORCA**：若从事灵巧手或开源机器人研究，该平台提供了完整的硬件与软件栈，值得深入了解。

---

## Table of Contents

1. [Instruct-Particulate: Scaling Feed-Forward 3D Object Articulation with Kinematic Control](#2606.14699v1)
2. [EgoGuide: Egocentric Guidance for Efficient Robot-Free Demonstration Collection and Learning](#2606.14665v1)
3. [HPSv3++: Scaling Reward Models Across the Full Spectrum of Diffusion Model Capabilities](#2606.14657v1)
4. [StereoGeo: an end-to-end stereo camera calibration method](#2606.14619v1)
5. [Sensitivity Shaping for Latent Modeling](#2606.14585v1)
6. [ORCA: A Platform for Open-Source Dexterity Research](#2606.14561v1)
7. [TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation](#2606.14551v1)
8. [Spatially Conditioned Diffusion Policy: Learning Precise and Robust Manipulation with a Single RGB Camera](#2606.14535v1)
9. [AERMANI-PLACE: Language Guided Object Placement with Aerial Manipulators](#2606.14531v1)
10. [Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack](#2606.14409v1)

---

## Papers

<a id='2606.14699v1'></a>
## [Instruct-Particulate: Scaling Feed-Forward 3D Object Articulation with Kinematic Control](https://arxiv.org/abs/2606.14699v1)

**Authors:** Ruining Li, Yuxin Yao, Matt Zhou, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi

**Published:** 2026-06-12

**Categories:** cs.CV, cs.GR, cs.RO

**Abstract:**

Reconstructing articulated 3D objects is important for animation, gaming, and robotic simulations. Recent neural networks can estimate the articulated structure of 3D objects, but their generalization remains limited by the scarcity of annotated data for this task. To address this gap, we introduce Instruct-Particulate, a model that takes a 3D mesh together with a target kinematic specification, including part descriptions, connectivity, joint types, and optional point prompts, and predicts the corresponding kinematic part segmentation and joint motion parameters. The kinematic specification disambiguates the task and allows the model to target annotations of different granularity, thereby making it possible to use more abundant heterogeneous training data. At test time, the kinematic specification can be obtained automatically from large-scale vision-language models, so the model can be applied to any input mesh. To train our model at scale, we construct a heterogeneous dataset of more than 150,000 articulated 3D objects, extending existing publicly available collections with data obtained by partially labelling other 3D models (monolithic or already decomposed into parts) with kinematic labels by means of vision-language models. Experiments show that our model generalizes better across categories and to AI-generated meshes, enabling articulated asset reconstruction from real-world images via image-to-3D models.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇题为 **《Instruct-Particulate: Scaling Feed-Forward 3D Object Articulation with Kinematic Control》** 的论文分析如下：

### 1. 核心贡献摘要
该论文提出了一种名为 **Instruct-Particulate** 的前馈模型，旨在解决3D物体关节化（articulation）重建中数据稀缺导致的泛化能力瓶颈问题。通过引入基于指令（Instruction-based）的运动学规范（Kinematic Specification），该模型能够根据输入的3D网格及部分语义提示，精准预测物体的零件分割及关节运动参数，从而实现了跨类别的大规模通用化重建。

### 2. 关键创新与方法论
*   **指令驱动的运动学控制（Kinematic Control as Guidance）：** 不同于传统的“黑盒”式端到端模型，Instruct-Particulate 将运动学规范（如零件描述、连接关系、关节类型）作为模型的输入。这种方法不仅解决了任务的歧义性，还允许模型在不同颗粒度的数据上进行训练，极大地增强了对异构数据的兼容性。
*   **跨模态驱动的数据自动标注：** 论文通过利用大型视觉-语言模型（VLM）对海量零散的3D模型进行半自动化标注，构建了一个包含15万个物体的异构数据集。这解决了该领域长期存在的“标注数据匮乏”痛点。
*   **前馈推理架构（Feed-Forward Architecture）：** 与传统需要耗时优化的方法相比，该模型支持前馈预测，推理速度极快，且能良好适应AI生成的高噪声或多样化网格，具备极强的工业落地潜力。

### 3. 对领域的潜在影响
*   **推动3D资产自动化管线：** 该研究为从图像（Image-to-3D）到可交互3D资产的自动化生成提供了一块核心拼图。它使得“静态模型到动态交互模型”的转换变得低门槛且可扩展。
*   **数据驱动范式的突破：** 通过VLM辅助的标注策略，该论文展示了如何利用非结构化数据规模化训练特定任务模型，为其他细分领域的3D几何理解研究提供了范例。
*   **提升交互式仿真真实性：** 在机器人模拟和游戏开发中，运动学属性的准确重建直接决定了物理仿真效果，该研究显著提升了这些下游任务的资产准备效率。

### 4. 相关受益领域与应用
*   **机器人学（Robotics）：** 机器人操控（Manipulation）任务需要对物体关节结构有精确理解，该模型可帮助机器人快速“理解”陌生对象的物理特性。
*   **增强/虚拟现实（AR/VR）：** 实现真实世界物体的数字化重构，并赋予其交互能力（如抽屉开关、门铰链旋转）。
*   **影视与游戏动画：** 大幅降低美术人员手工设置骨骼（Rigging）和关节约束的时间成本。
*   **生成式AI（Generative AI）：** 能够与目前的SOTA 3D生成模型无缝集成，增强生成资产的“可玩性”。

### 5. 可推断的局限性
*   **模型对VLM的依赖：** 模型性能在推理时高度依赖于VLM对输入网格的“运动学规范”提取能力，如果VLM对复杂或罕见物体的描述出现幻觉，下游的运动学预测可能会出错。
*   **复杂机械结构的边界：** 尽管能够处理常见类别，但在处理高度复杂的机械联动（如复杂的齿轮组合、非线性运动）时，可能依然存在泛化上限。
*   **语义与拓扑的对齐：** 在处理多层嵌套结构或极细碎零件时，运动学规范与网格拓扑之间的语义对齐依然是一个极具挑战性的难点，可能需要更精细的特征对齐技术。

**总结：** 这篇论文的精妙之处在于它没有试图设计一个更复杂的网络来“猜测”物体的运动，而是通过“指令性控制”这一巧妙的设计，将不可控的任务转换为了可控的查询任务。这是一种典型的以数据工程与规范化设计解决几何难题的成功案例，在当前3D视觉追求规模化与可控性的潮流下极具研究价值。

**Key Findings:**

- To address this gap, we introduce Instruct-Particulate, a model that takes a 3D mesh together with a target kinematic specification, including part descriptions, connectivity, joint types, and optional point prompts, and predicts the corresponding kinematic part segmentation and joint motion parameters.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14699v1)
- [arXiv](https://arxiv.org/abs/2606.14699v1)

---

<a id='2606.14665v1'></a>
## [EgoGuide: Egocentric Guidance for Efficient Robot-Free Demonstration Collection and Learning](https://arxiv.org/abs/2606.14665v1)

**Authors:** Yue Xu, Mingtao Nie, Tianle Li, Hong Li, Yibo Luo, Siyuan Huang, Yong-Lu Li

**Published:** 2026-06-12

**Categories:** cs.RO

**Abstract:**

Robot learning from real-world demonstrations is currently constrained by data scaling. Universal Manipulation Interface (UMI) provides an efficient robot-free data collection interface, yet current UMI-style pipelines often collect redundant demonstrations and lack global scene context. To improve data efficiency, we present EgoGuide, a collection interface that records synchronized wrist and head/egocentric observations and couples them with online visual-geometric data quality guidance. We also introduce a Gated Egocentric Residual Policy for robust learning from a viewpoint-varying egocentric camera, allowing head/egocentric context to correct ambiguous local observations while preserving stable wrist-view control. Real-world experiments show that EgoGuide reduces the required number of data episodes and improves data efficiency. The residual policy further improves robustness under visual occlusion. Project Page: https://silicx.github.io/EgoGuide

**Analysis:**

### 1. 摘要翻译
机器人学习目前受限于现实世界示范数据的规模。通用操作接口（UMI）提供了一种高效的非机器人数据收集接口，但现有的UMI式流程常收集冗余数据，且缺乏全局场景上下文。为了提升数据效率，我们提出了**EgoGuide**，这是一个记录同步腕部与头部/自我中心视角观测，并耦合在线视觉几何数据质量引导的收集界面。此外，我们引入了**门控自我中心残差策略（GERP）**，用于从视点变化的自我中心相机进行鲁棒学习，使头部/自我中心上下文能够校正模糊的局部观测，同时保持稳定的腕部视图控制。真实世界实验表明，EgoGuide减少了所需的示范数据量并提升了数据效率。残差策略进一步增强了在视觉遮挡下的鲁棒性。

### 2. 方法动机分析
*   **驱动力**：旨在解决非机器人示范收集（UMI）中“数据收集效率低”的问题，通过提升示范质量而非单纯增加数量，来实现更高效的机器人策略训练。
*   **现有方法痛点**：
    1.  **缺乏质量感知**：演示者不清楚当前示范的数据覆盖情况，导致大量冗余的“成功”示范。
    2.  **观测局限性**：单腕部相机视角在遮挡、长视距任务中无法捕捉完整环境信息。
    3.  **视点不对齐**：直接将头部视图加入策略训练往往因视角不稳定导致性能下降。
*   **研究假设**：通过在收集阶段提供在线视觉几何覆盖度反馈，并利用自我中心视角作为“门控残差”来补充腕部基线策略，可以大幅提升数据效率和鲁棒性。

### 3. 方法设计详解
*   **在线质量引导模块**：
    *   利用Meta Quest头显收集头部图像（$I^H$）和位姿（$T^H$），结合腕部图像（$I^W$）和位姿（$T^W$），计算视觉特征（DINOv2/CLIP）和几何位姿的相似度分数。
    *   通过AR反馈实时告知演示者当前状态在数据集中的“新颖度”，促使演示者调整初始位姿或场景布局，确保数据空间覆盖。
*   **门控自我中心残差策略（GERP）**：
    *   **基线（Base Policy）**：仅处理腕部相机和位姿，保证动作的稳定性。
    *   **残差分支（Residual Branch）**：将腕部相对位姿转换到头部坐标系（$T^{H \leftarrow W}$），与头部图像共同输入MLP，预测残差动作。
    *   **门控机制（Action Fusion）**：通过学习一个标量门控 $\alpha$，自动判断何时采纳自我中心视图的补充建议（如在遮挡严重时增加 $\alpha$），实现 $A_{final} = (1 - \alpha)A_b + \alpha A_r$。

### 4. 方法对比分析
*   **本质区别**：与现有主动感知方法（通常试图学习主动头部运动）不同，GERP将头部视图视为“增强上下文”而非动作输出，从而兼容固定相机设置。
*   **创新贡献**：引入了在线数据覆盖引导机制，实现了从“盲目采集”到“针对性补充”的范式转变；提出了无需主动控制头部的残差融合框架。
*   **适用场景**：适用于实验室或Crowdsourcing平台下的长视距、存在遮挡的机器人灵巧操作任务。

### 5. 实验分析
*   **验证方法**：在Pick Cube、Pepper Sorting等任务上对比了不同数据规模下的成功率（SR）和任务进度（TPS）。
*   **关键结论**：在Pepper Sorting任务中，EgoGuide仅需50%的数据量即可达到传统方法的性能水平。GERP框架在遮挡场景下比直接拼接视角的基线方法表现更稳健。
*   **优缺点**：有效解决了长尾分布覆盖问题；不足之处在于AR头显佩戴引起的疲劳及系统计算时延。

### 6. 实用指南
*   **开源情况**：已提供项目主页（https://silicx.github.io/EgoGuide/）。
*   **实现细节**：训练需分为两阶段，先冻结基线策略，再进行残差分支训练；利用课程学习（Curriculum Learning）从零权重逐渐增加残差权重 $\lambda_{act}$。
*   **迁移建议**：该架构可直接迁移至任何基于UMI的抓取任务，且其“覆盖度评分”逻辑可应用于其他需保证多样性的数据采集流程。

### 7. 总结
*   **核心思想**：通过实时反馈引导采集多样数据，利用残差门控融合多视角信息。
*   **速记版pipeline**：
    1.  AR引导：采集前实时展示数据覆盖度，调整位姿。
    2.  辅助视角：同步记录腕部与头部视角数据。
    3.  基线策略：训练稳定但局部的腕部控制。
    4.  残差融合：学习门控机制，按需引入自我中心上下文。

**Key Findings:**

- To improve data efficiency, we present EgoGuide, a collection interface that records synchronized wrist and head/egocentric observations and couples them with online visual-geometric data quality guidance.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14665v1)
- [arXiv](https://arxiv.org/abs/2606.14665v1)

---

<a id='2606.14657v1'></a>
## [HPSv3++: Scaling Reward Models Across the Full Spectrum of Diffusion Model Capabilities](https://arxiv.org/abs/2606.14657v1)

**Authors:** Yijun Liu, Jie Huang, Zeyue Xue, Yuming Li, Ruizhe He, Haoran Li, Shijia Ge, Siming Fu

**Published:** 2026-06-12

**Categories:** cs.CV

**Abstract:**

Reward models guide text-to-image (T2I) systems toward outputs aligned with human preferences. However, typical reward models such as HPSv3 are trained on pre-annotated data from earlier T2I models, without accounting for quality discriminative shifts arising from evolving model capabilities and reinforcement learning (RL) iterations, limiting their broader applicability. In this work, we propose HPSv3++, a reward model framework that elevates the HPSv3 model for varying T2I model capabilities and their RL iteration changes across the full capability-iteration spectrum. Specifically, we first introduce HPDv3++, a 212K dual-dimension preference dataset annotated for text fidelity and aesthetic quality using a recent high-capability (Qwen-Image) model with human supervision. We then propose a two-stage training framework. Stage 1 employs data-aware orthogonal gradient projection to incorporate diverse aesthetic perception from HPDv3++ while preserving the original effective human preference knowledge in HPSv3. Stage 2 further leverages unlabeled data from T2I models spanning different capability levels and RL iterations, and introduces a joint capability-iterations conditioned signal for the reward model together with a standard deviation-driven unsupervised guidance mechanism, strengthening reward model across the capability-iteration spectrum. HPSv3++ achieves state-of-the-art preference prediction, outperforming HPSv3 9.8% on HPDv3, 5.5% on GenAI-Bench, while achieving 79.1%/88.1% on our proposed HPDv3++. When used for T2I RL training, it consistently improves GenEval scores across diverse T2I models, demonstrating its wide-range capabilities. The code is available at https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus.

**Analysis:**

### 1. 摘要翻译
奖励模型（RM）引导文本到图像（T2I）系统生成符合人类偏好的输出。然而，诸如HPSv3之类的典型奖励模型通常在早期T2I模型的预标注数据上进行训练，未能考虑到随着模型能力演进和强化学习（RL）迭代而产生的质量判别漂移，限制了其广泛适用性。在本工作中，我们提出了HPSv3++，这是一个奖励模型框架，旨在提升HPSv3以适应不同T2I模型能力及其在全能力-迭代谱系中的RL迭代变化。具体而言，我们首先引入了HPDv3++，这是一个包含212K数据的双维度偏好数据集，并使用最新的高能力模型（Qwen-Image）结合人工监督进行标注，涵盖文本忠实度和审美质量。随后，我们提出了一个两阶段训练框架：阶段1采用数据感知正交梯度投影（Data-aware Orthogonal Gradient Descent）来整合HPDv3++的多样化审美感知，同时保留HPSv3原有的有效人类偏好知识。阶段2进一步利用来自跨不同能力级别和RL迭代的T2I模型的未标注数据，并引入联合“能力-迭代”条件信号，结合标准差驱动的无监督引导机制，从而在整个能力-迭代谱系中增强奖励模型。

### 2. 方法动机分析
- **驱动力**：解决奖励模型在面对前沿T2I模型和RL迭代过程中出现的“奖励失效”问题，实现全谱系（不同能力模型、不同训练阶段）的精准评估。
- **痛点**：
    - **分布偏移**：静态RM在预训练数据上的表现无法泛化到新一代高能力模型。
    - **动态判别力缺失**：RL训练过程中模型输出越来越相似，导致静态RM的判别力（Score Std）急剧下降，造成Reward Hacking。
- **研究假设**：模型输出的奖励得分标准差（Std）与模型对该分布的熟悉程度呈正相关。可以通过显式地将模型能力和RL进度作为条件输入，并结合无监督Std优化，提升对不同分布的判别力。

### 3. 方法设计详解
- **核心Pipeline**：
    - **阶段1：持续学习（OGD）**：通过正交梯度投影，将模型在HPDv3++上的梯度更新投影到HPSv3参考梯度的正交补空间，从而吸收新知识而不遗忘原有偏好。
    - **阶段2：条件自适应训练**：
        - **条件化模块**：利用Capability Encoder（从图像特征推断能力）和归一化RL步数作为条件，通过FiLM（Feature-wise Linear Modulation）注入Reward Head。
        - **标准差驱动的无监督损失**：引入$L_{std}$最大化组内得分标准差，缓解RL导致的同质化；引入$L_{adapt}$强制高能力/晚期模型组拥有更高的相对Std增益，增强判别力。
- **模型结构**：基于Qwen3-VL-8B主干，外接三层RankNet MLP奖励头，并额外增加一个用于隐式能力推理的Capability Encoder。

### 4. 方法对比分析
- **本质区别**：传统RM是无条件（Unconditional）的标量输出；HPSv3++是条件化（Conditioned）的动态奖励，能感知生成模型的具体能力等级和RL训练进度。
- **创新点**：提出了首个双维度（文本+审美）偏好数据集；结合了持续学习（OGD）和基于标准差的半监督自适应训练策略。
- **适用场景**：适用于任何依赖RLHF进行微调的T2I模型流程，尤其适合需要长期稳定训练且模型能力跨度大的场景。

### 5. 实验分析
- **关键结果**：在HPDv3基准上取得86.7%的准确率（提升9.8%），在GenAI-Bench上提升5.5%。
- **优势**：在RL训练中，相较于HPSv3，HPSv3++能保持单调递增的判别力（Std），避免了训练后期的Reward Hacking，生成质量更优。
- **局限**：对极端的超参数设置仍可能发生reward hacking；无法覆盖所有未来可能的生成模型变体。

### 6. 实用指南
- **开源情况**：代码已开源（github.com/PlantPotatoOnMoon/HPSv3-PlusPlus）。
- **实现关键**：
    - 确保RL过程中使用与训练一致的标准化迭代步数（0.0-1.0）。
    - 阶段2的Std优化非常依赖无监督组内数据的多样性，需构造多能力、多步数的Rollout数据。
- **迁移可能**：该框架中的“条件化奖励+标准差驱动优化”思想可直接迁移至T2V（视频生成）或多模态RLHF任务中，只要定义好相应的“能力”和“进度”特征即可。

### 7. 总结
- **核心思想**：通过能力-迭代条件注入和标准差自适应优化，实现奖励模型对生成动态的动态感知。
- **速记版pipeline**：
    1. 构建双维度偏好数据集。
    2. 使用正交梯度投影更新基础模型。
    3. 注入能力与迭代条件信号。
    4. 实施无监督标准差优化策略。

**Key Findings:**

- In this work, we propose HPSv3++, a reward model framework that elevates the HPSv3 model for varying T2I model capabilities and their RL iteration changes across the full capability-iteration spectrum.
- HPSv3++ achieves state-of-the-art preference prediction, outperforming HPSv3 9.8% on HPDv3, 5.5% on GenAI-Bench, while achieving 79.1%/88.1% on our proposed HPDv3++.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14657v1)
- [arXiv](https://arxiv.org/abs/2606.14657v1)

---

<a id='2606.14619v1'></a>
## [StereoGeo: an end-to-end stereo camera calibration method](https://arxiv.org/abs/2606.14619v1)

**Authors:** Imane Meddour, Andréa Macario Barros, Cédric Gouy-Pailler

**Published:** 2026-06-12

**Categories:** cs.CV

**Abstract:**

In this work, we propose StereoGeo, an end-to-end network-based approach for stereo camera calibration. Our method estimates the focal lengths and gravity directions of the left and right cameras, as well as the relative extrinsic transformation relating them. Existing methods often rely on calibration patterns in structured environments or address only a single camera configuration, being limited to either intrinsic or extrinsic estimation, and depending on a multi-view setups. StereoGeo extends the GeoCalib algorithm, integrating deep neural network feature extraction with a differentiable optimizer. Extensive experiments on real-world benchmarks demonstrate that StereoGeo achieves competitive performance for intrinsic calibration and provides accurate stereo extrinsic estimation, outperforming existing methods that are limited to monocular settings. The dataset used in this work is partially publicly available at https://github.com/meddourimane/StereoGeo-dataset.

**Analysis:**

## 1. 摘要翻译
本文提出了一种名为StereoGeo的端到端网络模型，用于立体相机校准。该方法可同时估计左右相机的焦距、重力方向以及它们之间的相对外参变换。现有方法通常依赖结构化环境中的校准图案，或仅限于单视图下的内参/外参估计，且往往依赖多视图设置。StereoGeo通过整合深度神经网络特征提取与可微优化器，扩展了GeoCalib算法。在真实世界基准测试上的实验表明，StereoGeo在内参校准方面表现出竞争力，并能提供精确的立体外参估计，优于现有的单目设置方法。

## 2. 方法动机分析
*   **驱动力**：实现一种无需校准图案、无需特征匹配、能够处理左右相机内参不一致（非对称）的鲁棒立体相机校准方案。
*   **现有方法痛点**：
    1.  传统方法（如Zhang氏法）极度依赖特定图案，且对图像质量敏感。
    2.  现有的立体学习方法（如UGCL）假设左右相机内参完全相同，无法处理不对称配置。
    3.  现有方案多集中于单视图内参估计，缺乏对立体外参的鲁棒恢复。
*   **核心直觉**：通过深度学习预测每像素的几何先验（Perspective Fields），利用可微的Levenberg-Marquardt (LM) 层在几何约束下对相机参数进行端到端的非线性优化。

## 3. 方法设计详解
*   **流程总结**：
    1.  **独立特征提取**：左右图像分别送入两个独立的SegNeXt编码器，保持分支模块化。
    2.  **几何先验预测**：预测每像素的“透视场（Perspective Fields）”，包括重力投射向量（up-vectors）和纬度（latitude），以及对应的置信度图。
    3.  **立体融合**：将左右分支的特征图拼接并进行全局池化，通过MLP回归出相对位姿（旋转和平移）。
    4.  **LM优化层**：利用预测出的几何先验作为残差约束，通过可微的LM算法迭代更新焦距、重力及外参。
*   **算法核心**：利用透视几何约束构建置信度加权的残差函数（公式8-9），通过LM算法求解非线性最小二乘问题，实现参数的梯度更新。

## 4. 方法对比分析
*   **本质区别**：将“几何优化过程”嵌入网络架构内部（Differentiable Optimization），而非将其视为后处理。
*   **创新贡献**：成功将GeoCalib扩展至非对称立体设置；通过联合训练和置信度权重，消除了对特征匹配和特定校准板的依赖。
*   **适用场景**：自动驾驶、机器人立体视觉、室内外非结构化环境下的相机自校准。

## 5. 实验分析
*   **关键结果**：在KITTI数据集上，虽然单视图内参误差与现有方法接近，但其外参估计的稳定性（标准差更低）显著优于特征匹配方法（ORB/SuperPoint）。
*   **主要优势**：鲁棒性强，无需特征匹配，能直接恢复度量尺度平移。
*   **主要局限**：目前尚未建模透镜畸变和主点偏移。

## 6. 实用指南
*   **开源情况**：数据集部分开源 (https://github.com/meddourimane/StereoGeo-dataset)。
*   **实现细节**：训练需注意学习率线性预热（warmup 4k steps）；LM优化层是核心，需确保雅可比矩阵的正确实现，以便梯度回传。
*   **迁移可能**：该架构可迁移至多传感器对齐任务，如将相机参数替换为激光雷达或IMU的内参，利用类似的几何约束进行联合优化。

## 7. 总结
*   **核心思想**：利用透视先验与可微LM优化，实现无标记立体自校准。
*   **速记版pipeline**：
    1. 用神经网络预测图像里的地平线和重力方向；
    2. 融合左右图像信息推算相对位置；
    3. 用数学算法（LM）微调焦距与姿态，直到画面几何逻辑最优；
    4. 对误差进行端到端反向传播训练。

**Key Findings:**

- In this work, we propose StereoGeo, an end-to-end network-based approach for stereo camera calibration.
- Our method estimates the focal lengths and gravity directions of the left and right cameras, as well as the relative extrinsic transformation relating them.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14619v1)
- [arXiv](https://arxiv.org/abs/2606.14619v1)

---

<a id='2606.14585v1'></a>
## [Sensitivity Shaping for Latent Modeling](https://arxiv.org/abs/2606.14585v1)

**Authors:** Hongzhan Yu, Chenghao Li, Ruipeng Zhang, Henrik Christensen, Sicun Gao

**Published:** 2026-06-12

**Categories:** cs.RO, cs.AI

**Abstract:**

Generative dynamics models enable planning in challenging robotic systems, but safe deployment requires reliably detecting policy-induced out-of-distribution (OOD) transitions. Existing methods typically treat the learned dynamics as fixed and attach post hoc support surrogates. We show that these surrogates can fail when the dynamics are locally insensitive to critical action choices: unsupported control actions may produce latent predictions that resemble demonstrated transitions, suppressing OOD signals despite large true predictive errors. To address this, we introduce support-conditioned control-sensitivity regularization, which promotes sensitive local response to control input changes in learned dynamics in high-support training regions. This preserves control-induced variation while limiting unstable extrapolation due to weak empirical support. Experiments in vision-based obstacle avoidance, manipulation, and real-robot navigation show improved OOD detection and safer closed-loop planning.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我针对这篇名为《Sensitivity Shaping for Latent Modeling》的论文进行了深入分析。以下是详细评估：

### 1. 主要贡献总结
该论文提出了一种新的正则化技术，用于优化生成式动力学模型（Generative Dynamics Models）在机器人决策中的安全性。核心贡献在于通过“支持条件控制灵敏度正则化”（support-conditioned control-sensitivity regularization），纠正了潜空间模型对关键控制动作响应不敏感的问题，从而解决了传统方法难以检测“看似合理实则危险”的分布外（OOD）动作的缺陷。

### 2. 关键创新与方法论
*   **痛点发现：** 现有方法通常在学习动力学模型后外挂 OOD 检测器，但在动态模型对特定动作缺乏灵敏度时（即模型对错误动作仍输出平滑/正确的预测），检测器会失效。
*   **方法论：** 引入了一种显式的正则化机制，旨在使学习到的动态模型在数据支撑度高的区域（high-support regions）对控制输入的变化保持高度敏感。
*   **物理意义：** 该方法强迫模型在“可信的动作空间”内建立强映射，通过“敏感度塑造”迫使模型在面临未见过的、错误的控制输入时产生剧烈的潜空间预测偏差，从而更容易被下游检测器捕获。

### 3. 对计算机视觉领域的潜在影响
对于计算机视觉（尤其是视觉驱动的机器人技术）而言，这项研究具有重要意义：
*   **增强视觉决策的鲁棒性：** 视觉潜空间（Latent space）学习通常存在“预测平滑性”问题，导致视觉模型容易对分布外场景产生“伪信心”。该方法将控制论与潜空间表示学习结合，为视觉模型训练增加了一层“安全约束”。
*   **闭环控制的安全性：** 视觉导航和操纵任务中，视觉偏差极易导致错误动作的产生。该方法直接在训练阶段强化模型对控制指令的敏感性，为视觉策略在真实世界的安全闭环控制提供了理论保障。

### 4. 相关领域与受益应用
*   **自动驾驶与移动机器人：** 在复杂的动态环境中，视觉模型需要准确感知哪些动作会导致碰撞，该方法能显著提升导航系统识别并拒斥危险动作的能力。
*   **机器人操作（Manipulation）：** 在处理透明、反光或形状复杂的物体时，视觉系统容易失效，该方法有助于在视觉感知不确定时实现更稳健的规划。
*   **Sim-to-Real 迁移：** 在从仿真转向真实环境时，该方法有助于识别哪些区域的视觉表现与动作行为是不匹配的，从而提高迁移效率。

### 5. 潜在局限性（基于摘要的推测）
*   **性能权衡（Performance Trade-off）：** 强迫模型保持高敏感度可能与模型的泛化性能存在博弈，过度敏感是否会导致模型对噪声（如视觉传感器噪声）过于脆弱，尚需进一步验证。
*   **正则化权重设定：** 该方法依赖于对“支持区域”的定义，如何动态且准确地量化“经验支持度”可能是一个挑战，参数调整不当可能导致模型训练不稳定。
*   **计算开销：** 在潜空间中引入额外的敏感度正则化梯度计算，是否会显著增加大规模视觉动态模型（如基于 Transformer 或扩散模型的动态预测）的训练耗时。

### 专家点评
这篇论文的巧妙之处在于它没有试图单纯改进 OOD 检测器（这是传统的后处理视角），而是从**产生数据的动态模型本身**入手，通过“敏感度塑造”从源头上改善潜空间特征的质量。对于从事视觉表示学习与具身智能（Embodied AI）的研究者来说，这是一个非常优雅的架构改进思路。

**Key Findings:**

- We show that these surrogates can fail when the dynamics are locally insensitive to critical action choices: unsupported control actions may produce latent predictions that resemble demonstrated transitions, suppressing OOD signals despite large true predictive errors.
- To address this, we introduce support-conditioned control-sensitivity regularization, which promotes sensitive local response to control input changes in learned dynamics in high-support training regions.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14585v1)
- [arXiv](https://arxiv.org/abs/2606.14585v1)

---

<a id='2606.14561v1'></a>
## [ORCA: A Platform for Open-Source Dexterity Research](https://arxiv.org/abs/2606.14561v1)

**Authors:** Francesco Capuano, Maximilian Eberlein, Fabrice Bourquin, Clemens Claudio Christoph

**Published:** 2026-06-12

**Categories:** cs.RO, cs.LG

**Abstract:**

Robotics manipulation research increasingly focuses on two-finger parallel grippers for their effectiveness, affordability, and ease of teleoperation. Grippers are nonetheless limited by their form factor, often requiring bimanual setups even for simple reorientation tasks. Anthropomorphic hands are a more natural platform for dexterous robot learning -- closer to the human hand, and capable of learning from human video -- yet they remain hard to use in learning research: even where open and accessible hand hardware exists, the software for control, simulation, teleoperation, and retargeting is scattered in one-off code bases, and largely disconnected from the robot-learning ecosystem. In this work, we introduce the \orca~learning stack, an open-source research stack for dexterity as a first-class robot learning domain. Our \orca~stack unifies low-level control, simulation, teleoperation from a range of consumer platforms, and hand retargeting, behind a single interface, and integrates natively with popular robot-learning frameworks such as \lerobot, so dexterous hand researchers can leverage the same data, training, and evaluation pipelines used for non-dexterous robot learning. We demonstrate a complete end-to-end workflow, collecting expert demonstrations of an in-hand reorientation task by teleoperation with a consumer-grade VR headset, training an autonomous policy with \lerobot, and evaluating the learned policy in a fully reproducible and observable setup. We open-source the entire stack as a shared, reproducible foundation for dexterous-manipulation research.

**Analysis:**

### 1. 摘要翻译
机器人操作研究日益关注双指平行夹爪，因其有效性、经济性和易于远程操作。然而，夹爪受限于其形态，即使简单的重新定向任务通常也需要双臂操作。拟人化手是一种更自然的灵巧机器人学习平台，更接近人手，且能够从人类视频中学习。然而，它们在学习研究中难以使用：即使存在开放且可访问的硬件，其控制、仿真、远程操作和重定向的软件分散在各种孤立的代码库中，且与机器人学习生态系统几乎脱节。本研究推出了 orca 学习栈，这是一个将灵巧操作作为一等公民的开源研究栈。orca 栈在单一接口下统一了低级控制、仿真、来自多种消费级平台的远程操作以及手部重定向，并与主流机器人学习框架（如 lerobot）原生集成。我们演示了一个完整的端到端工作流程：通过消费级 VR 头显进行远程操作收集专家演示，使用 lerobot 训练自主策略，并在一个完全可复现且可观察的设置中评估该策略。我们将整个栈开源，作为灵巧操作研究的共享且可复现的基础。

### 2. 方法动机分析
*   **驱动力**：作者旨在消除现有灵巧手研究中“硬件与软件割裂”的现状，通过提供标准化的软件栈，降低灵巧手从“科研玩具”转变为“通用学习平台”的门槛。
*   **现有方法痛点**：现有研究通常采用拼凑式开发（各家SDK、孤立的重定向脚本），缺乏统一的数据格式和接口，导致代码复用性极差、复现难度高，且无法直接复用现有的深度强化学习/模仿学习工具链。
*   **研究假设**：通过将灵巧手标准化，并将其与成熟的机器人学习框架（LeRobot）打通，可以利用相同的数据、训练和评估流水线实现跨任务的迁移学习，从而提升整体灵巧操作的研究效率。

### 3. 方法设计详解
orca 栈分为四个相互协作的包，核心逻辑如下：
*   **orca_core (基础控制)**：定义了统一的硬件交互 API，处理低级控制（位置/力矩）、校准、 tendon 紧固，通过“强类型”数据结构（`OrcaJointPosition`）屏蔽了不同底层电机配置的差异，提供一致的关节空间控制接口。
*   **orca_sim (仿真集成)**：扩展了 core 接口以支持 MuJoCo 仿真。核心创新在于实现了“虚实一致”：仿真环境与真实硬件共用相同的描述文件（URDF/MJCF）和控制接口，使得在仿真中训练的策略可以零修改迁移到硬件。
*   **orca_teleop (远程操作与重定向)**：这是该工作的关键。它定义了抽象的 Teleop Source（支持 Meta Quest, MediaPipe 等），通过重定向器（Retargeter）将人类手部特征点（MANO）实时映射到机器人的关节配置。
    *   **技术细节**：采用带 Huber loss 的优化算法进行重定向，既保证了小几何误差下的二次优化效果，又在 landmark 噪声较大时利用线性约束降低了对异常值的敏感度。
*   **orca_arm (平台集成)**：提供完整多自由度平台（如 OpenArm, Franka Panda）的仿真描述，支持单/双臂协同任务。

### 4. 方法对比分析
*   **本质区别**：与现有针对特定手/特定任务的固化方案不同，orca 采用“平台化”架构，不绑定特定硬件，而是定义了通用的接口规范。
*   **创新贡献**：成功将“灵巧手”这一高维动作空间对象，无缝映射到了“夹爪”机器人社区广泛使用的 LeRobot 训练流水线中。
*   **适用场景**：适用于实验室中低成本（3D 打印）灵巧手驱动的各种接触密集型操作任务（如物体翻转、精细堆叠）。

### 5. 实验分析（精简版）
*   **关键结果**：在“立方体原地旋转”任务中，利用远程操作收集的 10 条演示数据，训练出的 ACT 策略在测试中达到 9/10 的成功率。
*   **主要优势**：极高的灵活性和标准化程度，成功证明了即便利用低成本、小型化硬件也能实现闭环感知-动作策略的端到端学习。

### 6. 实用指南
*   **开源情况**：完全开源，见 [github.com/orcahand](https://github.com/orcahand)。
*   **迁移建议**：开发者只需关注实现一个满足 `orca_core` 接口的驱动后端，即可复用所有的仿真、teleop 和训练模块。对于重定向部分，只需映射好该硬件的 URDF 运动学参数即可。

### 7. 总结
*   **核心思想**：通过标准化软件栈，将灵巧操作整合进通用机器人学习生态。
*   **速记版 pipeline**：
    1. 通过 VR/摄像头捕捉人类动作；
    2. 使用重定向算法转化为手部关节动作；
    3. 在标准化接口下采集数据集；
    4. 接入 LeRobot 框架训练策略；
    5. 虚实一致性部署评估。

**Key Findings:**

- In this work, we introduce the \orca~learning stack, an open-source research stack for dexterity as a first-class robot learning domain.
- We demonstrate a complete end-to-end workflow, collecting expert demonstrations of an in-hand reorientation task by teleoperation with a consumer-grade VR headset, training an autonomous policy with \lerobot, and evaluating the learned policy in a fully reproducible and observable setup.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14561v1)
- [arXiv](https://arxiv.org/abs/2606.14561v1)

---

<a id='2606.14551v1'></a>
## [TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation](https://arxiv.org/abs/2606.14551v1)

**Authors:** Zihao Li, Ranpeng Qiu, Yincong Chen, Guoqiang Ren, Weiming Zhi

**Published:** 2026-06-12

**Categories:** cs.RO, cs.AI

**Abstract:**

Robots under autonomous operation may require decisions based on evidence that is no longer visible. We study \emph{delayed-evidence} tasks, where an early cue disappears before a later decision point, so visually similar observations can require different actions. In these settings, the current observation is not a sufficient state for control. We introduce TRAjectory-routed Causal Evidence (TRACE), a memory framework for visuomotor imitation policies. TRACE stores task-relevant visual and robot-state evidence, such as object identity, target choice, or route-dependent state, in a fixed-size latent memory that remains bounded over long episodes. Instead of indexing memory by raw time or manually provided task labels, TRACE uses \emph{path signatures}: compact, order-sensitive features of the executed robot-state trajectory. These signatures do not store the visual cue itself; rather, they provide trajectory-conditioned keys for writing and retrieving the evidence stored when the cue was visible. When the robot later reaches an ambiguous observation, the policy conditions on TRACE memory to recover the missing context and choose the correct branch. TRACE attaches through lightweight adapters to policies, without changing the policy backbone, action head, or imitation objective. Across real-world long-horizon manipulation tasks with visually ambiguous branch points, TRACE improves branch selection and task success over alternative baselines, including short-history and recurrent memory. Project page: https://jeong-zju.github.io/trace

**Analysis:**

### 1. 摘要翻译
机器人在自主作业时，往往需要基于不可见的证据做出决策。我们研究了“延迟证据”（delayed-evidence）任务：早期线索在随后的决策点前消失，导致视觉相似的观测可能需要执行不同的动作。在此类场景中，当前观测不足以构成决策所需的状态。我们提出了 **TRACE** (TRAjectory-routed Causal Memory)，这是一种用于视觉运动模仿策略的记忆框架。TRACE 将任务相关的视觉和机器人状态证据存储在固定大小的潜在记忆槽中，且在长任务片段中保持规模不变。TRACE 不使用原始时间戳或手动提供的标签来索引记忆，而是利用“路径签名”（path signatures）：一种紧凑且对顺序敏感的机器人轨迹特征。这些签名本身不存储视觉信息，而是提供轨迹条件化（trajectory-conditioned）的键，用于写入和读取线索可见时存入的证据。当机器人后续到达歧义观测点时，策略利用 TRACE 记忆恢复丢失的上下文并选择正确分支。TRACE 通过轻量级适配器挂载到现有策略上，无需更改策略主干、动作头或模仿目标。在具有视觉歧义分支点的真实长时操作任务中，TRACE 在分支选择和任务成功率方面均优于包括短时窗口和循环记忆在内的基线方法。项目主页：https://jeong-zju.github.io/trace

---

### 2. 方法动机分析
*   **驱动力**：解决长时任务中“延迟证据”导致的局部可观测性问题（即任务早期关键信息消失，导致决策点无法仅凭当前观测做出正确选择）。
*   **现有痛点**：
    *   **短时窗口**：关键信息超出窗口后失效。
    *   **长时窗口**：增加计算成本，且策略难以区分哪些历史片段是关键的。
    *   **循环/通用记忆**：关键线索容易被任务进度信号淹没、稀释或混淆。
*   **核心直觉**：任务历史的“几何轨迹”本身就是最好的索引。如果能将历史轨迹编码为确定性的路径签名，即可通过它实现对记忆存储与读取的精确路由。

---

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **路径编码**：实时获取机器人状态轨迹（piecewise-linear path），计算其深度为 $p$ 的路径签名 $\xi_t$ 和差分特征 $\delta_t$。
    2.  **地址生成**：通过 MLP 将轨迹特征映射为记忆索引键 $q_t$ 和路由状态 $\rho_t$。
    3.  **记忆写入**：将当前视觉观察 $e_t$ 与路由键 $q_t$ 结合。根据路由概率 $\omega_{t,k}$ 将信息写入 $K$ 个固定槽位中的特定位置。
    4.  **记忆读取**：依据当前的轨迹特征进行查询，通过 Attention 机制读取出关键上下文 $z_t^{\text{mem}}$。
    5.  **策略适配**：通过轻量级 Adapter 将读取的记忆信息注入原始策略（作为 Token 或全局条件向量），实现动作输出。
*   **关键点**：使用路径签名不仅是压缩历史，更赋予了模型对轨迹“形状”的识别能力，使其在不同历史路径到达相同状态时，能读出不同的记忆。

---

### 4. 方法对比分析
*   **本质区别**：传统记忆机制多依赖隐式的时间序列建模（如 RNN/Transformer），而 TRACE 使用显式的、基于轨迹几何属性的路由方案。
*   **创新贡献**：引入路径签名作为记忆索引，彻底解耦了“记忆存储内容”与“记忆访问逻辑”。
*   **适用场景**：长时序、涉及多种路径选择的机器人操作任务（如：先看到物体颜色，后续去拿该颜色的工具）。

---

### 5. 实验分析（精简版）
*   **验证方法**：在5个真实机器人任务（Tool, Book, Laundry, Cable, Medicine）中，将 TRACE 挂载于 Regression 和 Diffusion 策略上进行测试。
*   **关键结论**：在所有任务中，引入 TRACE 后，平均成功进度分别从 25.50 提升至 69.23（Regression）和从 25.00 提升至 59.53（Diffusion）。
*   **优势/局限**：优势是模块化程度高，不破坏原策略主干；局限是路径签名的维度随深度呈指数增长，长轨迹下对高维状态的处理需权衡计算成本。

---

### 6. 实用指南
*   **开源情况**：已开源，可参考项目主页实现。
*   **实现细节**：
    *   **签名深度**：推荐 $p=3$，在表示能力与计算量间达到最佳平衡。
    *   **辅助损失**：需要平衡损失（$L_{\text{bal}}$）、熵损失（$L_{\text{ent}}$）和读取一致性损失（$L_{\text{cons}}$）来防止槽位坍缩或信息丢失。
*   **迁移建议**：由于其插件式架构，可直接将其作为“记忆适配层”添加至任何基于视觉的机器人模仿策略中。

---

### 7. 总结
*   **核心思想**：利用轨迹路径签名路由记忆，实现确定性的长时因果信息保留。
*   **速记 Pipeline**：
    1.  计算机器人状态路径签名作为索引键。
    2.  基于键路由视觉信息到固定记忆槽。
    3.  根据当前轨迹形态从槽位中读取上下文。
    4.  通过适配器将上下文注入策略进行决策。

**Key Findings:**

- We introduce TRAjectory-routed Causal Evidence (TRACE), a memory framework for visuomotor imitation policies.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14551v1)
- [arXiv](https://arxiv.org/abs/2606.14551v1)

---

<a id='2606.14535v1'></a>
## [Spatially Conditioned Diffusion Policy: Learning Precise and Robust Manipulation with a Single RGB Camera](https://arxiv.org/abs/2606.14535v1)

**Authors:** Seoyoon Kim, Kanghyun Kim, Dongwoo Ko, Yeong Jin Heo, Min Jun Kim

**Published:** 2026-06-12

**Categories:** cs.RO

**Abstract:**

Recent visual imitation learning systems have widely adopted multi-camera setups with wrist-mounted cameras as the de facto standard. However, manipulation from a single global view remains challenging, as the policy should capture fine-grained interaction details and identify task-relevant regions without local wrist views. To address this challenge, we present Spatially Conditioned Diffusion Policy (SCDP), a diffusion-based visuomotor policy that achieves precise and robust manipulation in a single-camera setting. Our key idea is that end-effector trajectories can serve as visual attention anchors that reflect task-relevant regions. Building on this idea, SCDP consists of two key components: (i) a visual encoder that produces multi-scale feature maps to capture both broader context and fine-grained visual features, and (ii) a spatial conditioning module that samples point-wise features along intermediate end-effector trajectories in the diffusion loop. Extensive simulation experiments show that SCDP consistently outperforms strong single-view baselines and achieves performance comparable to multi-camera baselines. Real-world experiments further demonstrate precise manipulation and robustness to visual distractors, highlighting the potential of single-camera imitation learning.

**Analysis:**

## 1. 摘要翻译
现有的视觉模仿学习系统大多以配有腕式相机的多相机配置作为事实标准。然而，仅从单一全局视角进行操作仍面临挑战，因为策略需要在缺乏局部腕部视图的情况下，捕捉细粒度的交互细节并识别任务相关区域。为解决这一难题，我们提出了空间条件扩散策略（SCDP），这是一种基于扩散模型的视觉运动策略，能够在单相机设置下实现精确且稳健的操纵。我们的核心思想是，末端执行器的轨迹可以作为反映任务相关区域的视觉注意力锚点。SCDP包含两个关键组件：(i) 生成多尺度特征图的视觉编码器，以同时捕捉更广泛的上下文和细粒度的视觉特征；(ii) 在扩散循环中沿着中间末端执行器轨迹采样点状特征的空间条件模块。大规模仿真实验表明，SCDP在单视图基准测试中表现显著优于强基线模型，并达到与多相机基线相当的性能。现实世界的实验进一步证明了其精确的操作能力和对视觉干扰的鲁棒性，突显了单相机模仿学习的潜力。

## 2. 方法动机分析
- **驱动力**：旨在克服单目视觉下缺乏深度感知和细粒度信息的局限，实现低成本且高精度的机器人操控。
- **痛点**：现有方法将观察结果压缩为全局特征（容易丢失精细细节）或简单依赖大型预训练模型（通用性强但对特定任务操作不够精细）。
- **研究假设**：未来的末端执行器轨迹在图像空间中的投影，天然带有“任务关注点”的先验信息，可作为有效的视觉空间注意力机制。

## 3. 方法设计详解
SCDP的核心工作流包含以下三步：
1. **多尺度视觉编码**：利用ResNet-18提取不同层次的特征图（$C_1$至$C_5$），通过独立卷积对齐到维度$d$，形成多尺度特征集合$\mathcal{F}$，兼顾全局语境与局部细节。
2. **空间条件采样**：
   - **轨迹重建**：在扩散过程的每一步，根据当前Action序列重建未来一段时间（horizon $s$）的末端轨迹。
   - **坐标投影**：利用外参矩阵将3D轨迹投影到2D图像坐标。
   - **特征聚合**：在投影点处进行双线性插值采样，将采样结果在时间轴上平均池化，形成空间上下文向量$F$。
3. **条件扩散网络**：将$F$通过FiLM（Feature-wise Linear Modulation）层注入到Conditional U-Net中，引导去噪过程，使策略在生成动作时能够根据当前估计的轨迹“聚焦”于关键视觉区域。

## 4. 方法对比分析
- **本质区别**：传统方法要么全局编码，要么引入昂贵的外部感知模块（如跟踪器、分割器）；SCDP通过扩散模型产生的动作轨迹实现“自我寻路”，将策略的动作意图作为视觉注意力锚点，无需额外感知代价。
- **创新贡献**：提出了一种将“动作生成”与“视觉特征选择”深度耦合的机制，利用扩散模型迭代去噪的特性，让策略随着预测精度的提升逐渐锁定关键视觉区域。

## 5. 实验分析（精简版）
- **验证方法**：在Meta-World和DexArt基准测试中，与DP（Diffusion Policy）、DP3、SKIL、OTTER等模型对比。
- **关键结果**：在Hard任务上，SCDP不仅超越了所有单视图基线，其性能在某些场景下甚至媲美“单目+腕式相机”配置。
- **主要优势**：极佳的鲁棒性（对抗干扰）和数据效率（20条示范即可达到80%成功率）。
- **主要局限**：对“间接交互”（接触点远离末端执行器）的任务处理能力受限；在强遮挡环境下采样点位置可能偏差。

## 6. 实用指南
- **实现细节**：关键超参数为重建时域长度 $s=8$。使用双线性插值采样时，需确保相机内外参校准准确。
- **迁移建议**：该框架迁移到其他任务时，重点在于调整投影矩阵以适配新的工作空间，且由于SCDP完全基于RGB，对传感器设置要求极低，适合快速部署。

## 7. 总结
- **核心思想**：通过轨迹投影实现动作导向的视觉动态关注。
- **速记版pipeline**：
  1. 图像多尺度编码提取细节；
  2. 根据预估动作绘制轨迹点；
  3. 采样轨迹点的视觉信息作为特征；
  4. 将特征注入去噪网络引导动作更新。

**Key Findings:**

- To address this challenge, we present Spatially Conditioned Diffusion Policy (SCDP), a diffusion-based visuomotor policy that achieves precise and robust manipulation in a single-camera setting.
- Extensive simulation experiments show that SCDP consistently outperforms strong single-view baselines and achieves performance comparable to multi-camera baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14535v1)
- [arXiv](https://arxiv.org/abs/2606.14535v1)

---

<a id='2606.14531v1'></a>
## [AERMANI-PLACE: Language Guided Object Placement with Aerial Manipulators](https://arxiv.org/abs/2606.14531v1)

**Authors:** Sarthak Mishra, Ritama Sanyal, Rishabh Dev Yadav, Wei Pan, Spandan Roy

**Published:** 2026-06-12

**Categories:** cs.RO

**Abstract:**

Object placement is a fundamental component of aerial manipulation tasks, yet existing systems typically require the desired placement position to be specified explicitly in metric coordinates. Such interfaces are not intuitive and require users to reason about coordinate frames and scene geometry, making them difficult to use in practical deployments. In contrast, humans often communicate spatial goals through a combination of language and pointing gestures. Inspired by this observation, we present AERMANI-PLACE, a framework for language-guided object placement with aerial manipulators. Given a scene image and a natural language instruction, an image editing model generates a modified version of the scene containing a visual marker that indicates where the object should be placed. This marker is then grounded into the physical environment using depth observations to recover a metric place point, after which a placement trajectory is generated and executed by the aerial manipulator. We evaluate the proposed approach on a test set of 100 language-guided placement tasks and demonstrate successful execution on a real aerial manipulation platform. Experimental results show that the proposed method reliably infers placement locations from language instructions with an average success rate of 87\% on the test-set and transfers effectively to real-world aerial manipulation with an average success rate of 72\%.   Video: https://youtu.be/SgwwgLBsv0g

**Analysis:**

### 1. 摘要翻译
物体放置是空中操作任务中的关键一环，但现有系统通常要求以明确的度量坐标指定目标位置，这既不直观，又要求用户推理坐标系和场景几何，在实际应用中难以操作。人类通常通过语言和指点动作来传达空间目标。受此启发，本文提出了 AERMANI-PLACE，这是一个利用空中操纵器进行语言引导物体放置的框架。给定场景图像和自然语言指令，该框架使用图像编辑模型生成包含视觉标记的编辑图像，以指示物体放置位置。随后，该标记通过深度观测地面化（grounded）为三维度量点，进而生成并执行放置轨迹。我们在 100 个语言引导放置任务的测试集上评估了该方法，并演示了其在真实空中操作平台上的成功执行。实验结果表明，该方法能够可靠地从语言指令中推断放置位置，测试集平均成功率为 87%，在真实空中操作环境中的平均成功率为 72%。

### 2. 方法动机分析
- **驱动力**：旨在弥合人类自然语言指令与空中机器人高精度空间控制之间的“意图鸿沟”。
- **痛点**：传统的空中操纵方法高度依赖预先定义的精确坐标，对环境几何形状要求极高，且缺乏对复杂语义指令的理解，导致操作灵活性差且极其不直观。
- **核心直觉**：将“放置任务”重构为一种“视觉指向”问题。既然现有的图像编辑模型（Image-Editing Models）已具备强大的空间语义推理能力，那么通过让模型在图像中生成一个“标记点”作为语义锚点，即可避开复杂的直接坐标推理。

### 3. 方法设计详解
整个流水线分为四个阶段：
1. **语言引导的目标定位**：利用生成式图像编辑模型（如 Nano Banana Pro），输入场景图像和语言指令，模型输出一张标记了目标位置（如霓虹绿点）的新图像。采用了结构化提示（Consistency Constraints）来防止模型扭曲场景几何。
2. **三维空间恢复**：提取编辑图像中标记点的像素坐标 $(u, v)$，利用预先获取的深度图 $D(u)$ 和相机内参矩阵 $K$，通过背投影公式计算出摄像机坐标系下的位置，再通过已知的相机位姿转换至世界坐标系，得到初始放置点 $P_{init}$。
3. **物体几何对齐**：为了实现现实的放置，系统根据预先获得的物体掩膜（SAM3生成）重构物体点云 $C_O$，计算其底部支持点 $C_{base}$，并将物体底部平移至 $P_{init}$，生成初始放置配置 $X_{init}$。
4. **碰撞规避与执行**：针对深度噪声，执行局部 $xyz$ 轴几何精修。通过“触碰测试”（touch-down）策略，在垂直方向上调整直到接触表面。最后，由全局规划器规划一条从当前位姿到 $X_{place}$ 的路径，并采用受控垂直下降策略（mitigate downwash）完成放置。

### 4. 方法对比分析
- **本质区别**：与依赖 VLA（Vision-Language-Action）模型的重型训练范式不同，AERMANI-PLACE 是一种**无需训练（Training-free）**的零样本框架，直接调用现成的视觉生成模型作为空间推理引擎。
- **创新贡献**：提出了一种语义驱动的视觉锚点地标化管线，成功将“生成式推理”与“传统几何控制”解耦，在降低系统复杂度的同时提升了对复杂环境的泛化能力。

### 5. 实验分析（精简版）
- **关键结果**：在 100 个任务测试中，该方法实现了 87% 的成功率；在真实物理平台 25 次实验中，达到 72% 的成功率。
- **主要优势**：无需针对特定环境进行昂贵的训练，具备良好的零样本推理能力，对非结构化环境适应性强。
- **主要局限**：对生成模型的依赖性高，可能出现“物体幻觉”（在场景中生成额外的物体）或“marker漂浮”现象，需后续几何校正支撑。

### 6. 实用指南
- **开源/复现建议**：论文提供了视频演示，核心算法基于提示工程（Prompt Engineering）和标准的相机投影几何。实现时，建议选用高精度的深度相机（如 ZED2-i）以保证地标恢复的准确性。
- **迁移性**：该框架易于迁移，其核心思想“视觉标记 -> 投影 -> 轨迹规划”可直接移植到任何支持坐标系转换的机器人平台上，无需调整核心模型。

### 7. 总结
- **核心思想**：通过生成模型在视觉空间打点，实现语义到度量点的零样本转换。
- **速记版pipeline**：
    1. 给图给指令，让 AI 在图里点个点；
    2. 通过深度图，把图片里的点换算成 3D 空间位置；
    3. 结合物体的几何信息，调整物体至接触表面；
    4. 规划安全路径，下落并放下物体。

**Key Findings:**

- Inspired by this observation, we present AERMANI-PLACE, a framework for language-guided object placement with aerial manipulators.
- Experimental results show that the proposed method reliably infers placement locations from language instructions with an average success rate of 87\% on the test-set and transfers effectively to real-world aerial manipulation with an average success rate of 72\%.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14531v1)
- [arXiv](https://arxiv.org/abs/2606.14531v1)

---

<a id='2606.14409v1'></a>
## [Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack](https://arxiv.org/abs/2606.14409v1)

**Authors:** He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, Zhengyou Zhang

**Published:** 2026-06-12

**Categories:** cs.RO, cs.AI

**Abstract:**

In this report, we present Hy-Embodied-0.5-VLA, abbreviated as HyVLA-0.5, an end-to-end system that spans the full robot learning stack: data collection, model design, continued pre-training and supervised fine-tuning, RL post-training, and real-world deployment. Each component serves a distinct role in this stack.

**Analysis:**

这是一份针对《Hy-Embodied-0.5-VLA》的技术深入分析报告。

### 1. 摘要翻译
本文提出了 HyVLA-0.5，一个涵盖从数据采集到实机部署的全栈机器人学习系统。通过自定义的 UMI（通用操作接口）设备，我们采集了超 10,000 小时的亚毫米级精度人类演示数据。模型架构方面，采用基于 Mixture-of-Transformers (MoT) 的 Hy-Embodied-0.5 作为骨干，引入了条件流匹配动作专家和紧凑的记忆编码器，并采用 Delta-Chunk 动作表示以解耦本体运动学。为了提升部署表现，我们引入了基于 Proximalized Preference Optimization (PRO) 的 FlowPRO 离线强化学习算法，在无需显式奖励模型的情况下将失败案例转化为快速策略优化。最后，通过异步推理与基于 Bézier 曲线的路径平滑技术，实现了闭环高频控制。

### 2. 方法动机分析
- **驱动力**：通用机器人难以从单一模型中涌现，必须将数据、架构、强化学习和硬件部署视为统一的整体进行协同设计。
- **痛点**：
    1. 传统数据采集缺乏力反馈或动作标签不够精细；
    2. 通用 VLM 骨干未针对机器人控制进行优化；
    3. 现有离线 RL 依赖脆弱的奖励函数，且推理延迟往往导致部署不佳。
- **假设**：通过“端到端”的堆栈协同设计（从采集硬件到动作表示，再到部署推理），结合人类偏好驱动的离线 RL，能显著提升机器人长尾任务的成功率与部署鲁棒性。

### 3. 方法设计详解
- **数据层**：开发了带运动捕捉 cage 的 UMI 穿戴设备，直接在笛卡尔坐标系下记录 6-DoF 轨迹，避免了基于视觉的 SLAM 带来的位姿漂移。
- **建模层（核心）**：
    - **Delta-Chunk 动作表示**：预测末端执行器增量变化，解耦了策略学习与机器人的具体关节运动学，实现了跨形态（Embodiement-agnostic）的泛化。
    - **流匹配动作专家（Flow Matching）**：相比离散化的 Token 分类，该方法能生成连续、高频的动作流。
    - **紧凑记忆编码器**：利用时间-空间注意力机制，将多帧历史信息压缩至当前帧，保证推理时 Token 数量不变。
- **强化学习（FlowPRO）**：
    - 利用“干预-回滚”流程，由操作员纠正失败轨迹，生成（成功，失败）偏好对。
    - **RPRO Loss**：通过对比学习优化策略，加入对称近端正则化项（Proximal regularizer）防止奖励欺诈（Reward-hacking），从而在无奖励模型的情况下实现稳健的策略迭代。
- **部署层**：采用异步生产者-消费者架构，利用三次 Bézier 曲线平滑拼接预测的动作 Chunk，确保了 $C^1$ 连续性，解决了因推理延迟产生的控制抖动。

### 4. 方法对比分析
- **本质区别**：与采用单一动作 Token 离散化的方法不同，它强调“流匹配+增量动作表示”，将机器人控制看作流传输问题。
- **创新点**：
    1. **FlowPRO**：完全摆脱了对奖励模型（Reward Model）或价值函数（Value Function）的依赖；
    2. **部署闭环**：通过 Bézier 平滑处理异步推理的动作缝隙，是少数将工程部署作为首要考量的学术方案。

### 5. 实验分析
- **核心结果**：在 RoboTwin 2.0 基准测试中，HyVLA-0.5 在 Clean 和 Randomized 设置下分别达到 90.9% 和 90.1% 的成功率，远超主流基线。
- **结论**：证明了“高精度的 UMI 预训练数据 + 针对性强化学习”能有效应对高精度的长尾操作挑战。

### 6. 实用指南
- **开源情况**：已开源代码与模型。
- **关键细节**：
    - 训练需采用 `bfloat16` 混合精度；
    - **Delta-Chunk** 预测是关键，务必保证训练时状态与动作的归一化一致；
    - 部署时 $\gamma$ 超参数调节是控制平滑度的关键，应视硬件加速度限制调整。

### 7. 总结
- **核心思想**：构建协同化的机器人全栈学习系统，通过动作增量化与偏好对齐实现泛化。
- **速记版 pipeline**：
    1. **精准采集**：通过穿戴式 UMI 系统录制高精度人类操作；
    2. **骨干预训练**：在大规模数据上进行流匹配式动作建模；
    3. **偏好学习**：通过操作员回滚纠错，用对比损失优化策略；
    4. **异步平滑**：利用 Bézier 曲线缝合动作流，确保实机稳定控制。

**Key Findings:**

- In this report, we present Hy-Embodied-0.5-VLA, abbreviated as HyVLA-0.5, an end-to-end system that spans the full robot learning stack: data collection, model design, continued pre-training and supervised fine-tuning, RL post-training, and real-world deployment.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.14409v1)
- [arXiv](https://arxiv.org/abs/2606.14409v1)

---

