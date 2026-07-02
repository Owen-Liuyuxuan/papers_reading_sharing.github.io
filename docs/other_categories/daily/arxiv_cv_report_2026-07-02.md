time: 20260702

# Arxiv Computer Vision Papers - 2026-07-02

## Executive Summary

以下是根据2026年7月1日Arxiv计算机视觉论文生成的每日报告执行摘要：

---

### 1. 主要主题与趋势

本期论文高度聚焦于**机器人学与具身智能**，10篇中有7篇直接涉及机器人操作、规划或导航。具体趋势包括：
- **视觉-语言-动作模型（VLA）** 应用于复杂长期任务（如家具组装）。
- **3D高斯泼溅（3DGS）** 从静态场景重建扩展到动态安全滤波与自主飞行。
- **预测与规划** 中引入结构化潜在表示（4D、中间地图）以提升泛化性和效率。
- **触觉感知** 的预训练与迁移成为灵巧操作的新方向。
- **公平性** 在文本到图像生成中受到关注（交叉注意力引导）。
- **点云分解** 通过可变形几何基元（超二次曲面）实现更灵活的形状解析。

### 2. 特别重要或创新性强的论文

- **《FurnitureVLA》**：首个将VLA应用于**长周期双手协调装配**的框架，展示了大模型在结构化任务中的推理与执行能力。
- **《FastBridge》**：将3DGS用于**四旋翼安全滤波**，克服了模型与现实之间的差距，在高速飞行中实现实时安全约束。
- **《EquiSteer》**：提出**交叉注意力引导**方法，无需额外训练即可缓解文本到图像生成中的偏见，具有直接的社会价值。
- **《SuperFlex》**：引入**可变形超二次曲面**进行点云分解，比传统刚体基元更灵活，为场景理解与物体重建提供新工具。

### 3. 新兴研究方向与技术

- **基础模型服务系统**（ROSA）：针对机器人工厂的模型推理部署优化，标志着机器人基础模型从算法走向工程化。
- **失败感知与自我恢复**（FAR）：在测试时通过重试机制实现持续策略改进，提升机器人系统鲁棒性。
- **度量无关的轨迹预测**：摆脱对特定评估指标的依赖，推动预测模型在多变场景下的通用性。
- **以人为中心的触觉预训练**：将人类触觉演示迁移至机器人灵巧手，降低数据采集成本并提升操作泛化性。

### 4. 建议优先全文阅读的论文

| 论文 | 理由 |
|------|------|
| **FurnitureVLA** | VLA + 双手装配是当前机器人学习最前沿挑战，方法具启发性 |
| **FastBridge** | 3DGS与安全控制结合，对实时自主导航有重要实践意义 |
| **EquiSteer** | 公平性生成是AI伦理核心议题，方法简洁有效 |
| **SuperFlex** | 新几何表示对点云分割与物体建模有潜在变革 |
| **ROSA** | 系统级贡献推动基础模型在工业机器人中的落地 |

---

## Table of Contents

1. [FurnitureVLA: Learning Long-Horizon Bimanual Furniture Assembly with Vision-Language-Action Model](#2607.01212v1)
2. [FastBridge: Closing the Model-Based Realization Gap in Safety Filters on 3D Gaussian Splatting for Fast Quadrotor Flight](#2607.01200v1)
3. [Structured 4D Latent Predictive Model for Robot Planning](#2607.01166v1)
4. [EquiSteer: Cross-Attention Steering Towards a Fairer Text-Guided Image Generation](#2607.01147v1)
5. [SD-RouteFusion: Ego-Trajectory Prediction with SD-Map Route Conditioning](#2607.01139v1)
6. [Towards Metric-Agnostic Trajectory Forecasting](#2607.01133v1)
7. [FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement](#2607.01111v1)
8. [ROSA: A Robotics Foundation Model Serving System for Robot Factories](#2607.01088v1)
9. [Human-Centric Transferable Tactile Pre-Training for Dexterous Robotic Manipulation](#2607.01067v1)
10. [SuperFlex: Deformable Superquadrics for Point Cloud Decomposition](#2607.01015v1)

---

## Papers

<a id='2607.01212v1'></a>
## [FurnitureVLA: Learning Long-Horizon Bimanual Furniture Assembly with Vision-Language-Action Model](https://arxiv.org/abs/2607.01212v1)

**Authors:** Chenyang Ma, Yue Yang, Radu Corcodel, Siddarth Jain, Andrew Wu, Chiori Hori, Diego Romeres

**Published:** 2026-07-01

**Categories:** cs.RO, cs.AI

**Abstract:**

Current work on robot furniture assembly mostly focuses on toy-scale settings or single-arm manipulation. We introduce FurnitureVLA, the first systematic study of real-scale bimanual furniture assembly using Vision-Language-Action models (VLAs). We formalize the task, develop a scalable simulation pipeline for expert data generation and evaluation, and build a VR teleoperation system for single-operator bimanual control to collect high-quality real-world demonstrations. To address extreme long-horizon assembly with up to 7 subtasks and 1550 control steps, we propose a progress-enhanced VLA, finetuned on semantically grounded subtasks, that jointly predicts actions and a continuous progress signal, enabling automatic subtask transitions and reducing compounding errors during inference. We further study perception and control design factors that critically affect precision in real-scale assembly. FurnitureVLA improves average simulation success from 48% to 80% compared to baselines across three furniture types, with an additional 21% gain from our design factor study. We validate on a real Kinova Gen3 platform with only 16% drop on the hardest task.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对 **FurnitureVLA** 这篇论文的分析如下：

### 1. 核心贡献摘要
该论文提出了 **FurnitureVLA**，这是首个针对真实规模家具组装任务的视觉-语言-动作（VLA）模型框架。通过结合VR远程操控数据采集、仿真流水线以及具备“进度增强”能力的VLA模型，该研究成功攻克了超长跨度双臂协同操作的挑战，显著提升了机器人复杂任务的成功率与鲁棒性。

### 2. 关键创新与方法论
*   **进度感知机制 (Progress-enhanced VLA)：** 这是该论文的核心技术突破。模型不仅预测控制动作，还同步预测一个连续的进度信号（Progress Signal）。这种显式的语义接地（Semantic Grounding）机制能够辅助模型确定当前任务阶段，实现自动的子任务切换，从而有效缓解了长序列决策中常见的累积误差（Compounding Errors）问题。
*   **端到端的仿真与实操闭环：** 研究团队构建了从大规模仿真专家数据生成到VR远程操控真实数据采集的完整流水线，确保了模型能从高质量、多模态的演示中学习复杂的双臂协调策略。
*   **系统级工程设计研究：** 论文不仅关注算法，还深入研究了视觉感知与运动控制的工程设计因子（Design Factors），量化了这些因素对高精度装配任务的影响，为机器人系统集成提供了宝贵的实证参考。

### 3. 对该领域的潜在影响
*   **突破“玩具级”局限：** 此研究推动了具身智能（Embodied AI）从简单的捡拾操作向复杂、长程、真实规模的生产与生活任务跨越。
*   **VLA模型的进化：** 该论文展示了将视觉-语言模型从简单的图像描述扩展到复杂物理任务执行的能力，证明了引入进度感知等辅助任务可以大幅提升VLA在长序列任务中的泛化性。
*   **标准化范式：** 提供了一套针对长程双臂组装任务的评估基准和数据采集方法，为后续研究该领域复杂交互提供了参考框架。

### 4. 受益的相关领域与应用
*   **智能制造与装配：** 直接应用于家具自动化制造、精密零件组装等工业场景。
*   **服务机器人：** 助力家庭服务机器人处理更复杂的家务活，如整理杂乱房间、组装家居用品。
*   **人机协作 (HRC)：** 其中的双臂协同控制策略可以延伸到与人类共存的复杂协作任务中。
*   **多模态大模型研究：** 为视觉-语言模型如何进行长程物理决策提供了实验范例。

### 5. 可推测的局限性
*   **环境依赖性：** 尽管在仿真和特定实机上表现优秀，但在完全未知的、非结构化的真实家庭环境（光照变化、杂乱背景）中，模型的泛化能力仍需进一步验证。
*   **感知精度约束：** 尽管提出了设计因子研究，但在面对家具组装中常见的遮挡问题（Occlusion）和微小误差修正时，纯视觉方案是否能完全替代高精度的触觉反馈仍存疑。
*   **数据需求：** 尽管使用了仿真数据，但依赖高质量VR远程操控数据可能导致采集成本较高，如何在更少的人类演示下实现高效微调仍是挑战。

---

**专家视角的点评：**
这篇论文的趣味性在于它不仅是在“跑模型”，而是**直面了具身智能中最棘手的痛点之一：如何处理长达1550步的决策序列**。通过将“进度信号”作为中间约束引入VLA模型，论文巧妙地利用了多任务学习的思想解决了长序列中的状态漂移问题。对于计算机视觉研究者而言，这是一个极好的例子，展示了如何通过将高级语义信息（进度）注入低级动作控制，从而在物理世界中实现“视觉感知-语义理解-动作执行”的紧密耦合。

**Key Findings:**

- We introduce FurnitureVLA, the first systematic study of real-scale bimanual furniture assembly using Vision-Language-Action models (VLAs).
- To address extreme long-horizon assembly with up to 7 subtasks and 1550 control steps, we propose a progress-enhanced VLA, finetuned on semantically grounded subtasks, that jointly predicts actions and a continuous progress signal, enabling automatic subtask transitions and reducing compounding errors during inference.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01212v1)
- [arXiv](https://arxiv.org/abs/2607.01212v1)

---

<a id='2607.01200v1'></a>
## [FastBridge: Closing the Model-Based Realization Gap in Safety Filters on 3D Gaussian Splatting for Fast Quadrotor Flight](https://arxiv.org/abs/2607.01200v1)

**Authors:** Tscholl Dario, Nakka Yashwanth Kumar, Gunter Brian

**Published:** 2026-07-01

**Categories:** cs.RO

**Abstract:**

Fast quadrotor flight requires safe obstacle avoidance under tight onboard compute limits. While 3D Gaussian Splatting (3DGS) provides a continuous, geometry-aware scene representation for perception-driven navigation, existing 3DGS safety filters use reduced-order models such as single- and double-integrators that ignore actuator limits and assume commanded accelerations are realized instantaneously. Building on an analytic collision cone barrier for 3DGS, we introduce a nonlinear, actuator-aware safety filter enforced through the full quadrotor dynamics. We derive a high-relative-degree collision cone exponential CBF and a backup CBF that preserves QP feasibility under input constraints using a forward-simulated backup policy. Compared with a state-of-the-art 3DGS safety filter, our approach reduces trajectory jerk by 47% and runs 2.25 times faster. We validate the method in simulation and on hardware for real-time navigation in cluttered, perception-derived environments.

**Analysis:**

### 1. 摘要翻译
快速四旋翼飞行需要在严格的板载计算限制下实现安全避障。尽管 3D 高斯泼溅 (3DGS) 为感知驱动的导航提供了连续的几何感知场景表示，但现有的 3DGS 安全滤波器使用忽略执行器限制且假设指令加速度可瞬时实现的降阶模型（如单/双积分器）。本文基于 3DGS 的解析碰撞锥屏障函数，提出了一种通过全四旋翼动力学强制执行的非线性执行器感知安全滤波器。我们导出了高相对阶碰撞锥指数控制屏障函数 (ECBF) 和一种通过前向模拟备份策略在输入约束下保持二次规划 (QP) 可行性的备份 CBF。与最先进的 3DGS 安全滤波器相比，我们的方法减少了 47% 的轨迹加加速度（jerk），且运行速度提升了 2.25 倍。我们在仿真和真实硬件实验中验证了该方法在杂乱、感知导出环境中的实时导航性能。

### 2. 方法动机分析
*   **驱动力**：解决高性能四旋翼在高速飞行时，因“模型-现实鸿沟”（Model-based Realization Gap）导致的轨迹跟踪误差与安全失效问题。
*   **现有痛点**：现有方法多采用简化的双积分器模型，忽视了四旋翼复杂的非线性动力学和执行器饱和限制，导致安全滤波器输出的指令在实际飞行中无法被精确执行，使得屏障函数无法维持前向不变性。
*   **核心直觉**：通过将完整的非线性动力学纳入安全滤波器设计，并引入能够处理执行器限制的“备份策略”来确保安全性的形式化保证。

### 3. 方法设计详解
*   **流程总结**：
    1.  **场景建模**：将 3DGS 作为几何表示，提取 Gaussian Splats 的参数 $(\mu, \Sigma)$ 作为碰撞约束的几何基础。
    2.  **动力学线性化**：利用微分平坦特性，通过动态反馈线性化（Dynamic Feedback Linearization）将复杂动力学变换为可直接约束推力与扭矩的线性输入输出形式。
    3.  **碰撞锥 ECBF**：针对高相对阶约束，构造指数型 CBF (ECBF)，通过多阶导数约束确保在控制指令层面满足碰撞避免。
    4.  **备份 CBF 增强**：引入备份策略（饱和悬停调节器），在 QP 求解不可行时提供安全后备，将安全集定义为可控不变子集。
*   **关键算法**：碰撞锥屏障函数通过 ray-ellipsoid 射线-椭球相交问题建模，通过屏障函数的梯度下降确保机器人远离椭球区域。ECBF 引入了多阶微分，不仅考虑距离，还考虑了速度及其对安全的影响，实现前瞻性避障。

### 4. 方法对比分析
*   **本质区别**：与仅基于距离的避障方法不同，该方法直接在全动力学模型上定义屏障函数，处理了“加速度无法瞬时实现”导致的滞后。
*   **创新贡献**：
    1.  识别并量化了 3DGS 安全滤波器中的现实化鸿沟。
    2.  构造了适配 3DGS 椭球碰撞模型的解析高阶 ECBF。
    3.  提出了基于备份策略的 QP 可行性保证机制。
*   **适用场景**：高动态、狭窄、杂乱环境下的高速敏捷飞行。

### 5. 实验分析
*   **验证方法**：通过 395k 个高斯点组成的自定义场景，对比传统双积分器模型与本文提出的非线性 ECBF 模型。
*   **结论**：在高速工况下，传统模型的 realization error（现实化误差）剧增，而本文方法能保持零误差水平，且最大轨迹加加速度降低了 46%-47%。
*   **局限**：计算负荷随主动避障障碍物数量增加；硬件验证时的备份策略仍需精细调参以实现最优避障。

### 6. 实用指南
*   **开源建议**：该工作基于 Nerfstudio 和 Splatfacto，复现时需重点关注几何表示与动力学坐标系间的刚性变换（similarity transform）。
*   **迁移建议**：其核心框架（ECBF + 备份策略）可直接迁移至其他非线性受限系统，但需重新推导针对不同动力学模型的动态反馈线性化方程。
*   **注意事项**：确保屏障函数的梯度在奇异点附近（如 MRP 奇异点）保持平滑。

### 7. 总结
*   **核心思想**：通过全动力学 ECBF 弥补模型实现鸿沟，保障高速避障安全。
*   **速记版 Pipeline**：
    1. 提取 3DGS 场景几何信息。
    2. 执行动态反馈线性化处理系统方程。
    3. 构造考虑执行器限制的碰撞锥 ECBF。
    4. 嵌入备份策略确保 QP 求解器始终有解。

**Key Findings:**

- Building on an analytic collision cone barrier for 3DGS, we introduce a nonlinear, actuator-aware safety filter enforced through the full quadrotor dynamics.
- Compared with a state-of-the-art 3DGS safety filter, our approach reduces trajectory jerk by 47% and runs 2.25 times faster.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01200v1)
- [arXiv](https://arxiv.org/abs/2607.01200v1)

---

<a id='2607.01166v1'></a>
## [Structured 4D Latent Predictive Model for Robot Planning](https://arxiv.org/abs/2607.01166v1)

**Authors:** Zhiyi Li, Peilin Wu, Xiaoshen Han, Ruojin Cai, Yilun Du

**Published:** 2026-07-01

**Categories:** cs.RO, cs.CV

**Abstract:**

Video predictive models are emerging as a powerful paradigm in robotics, offering a promising path toward task generalization, long-horizon planning, and flexible decision-making. However, prevailing approaches often operate on 2D video sequences, inherently lacking the 3D geometric understanding necessary for precise spatial reasoning and physical consistency. We introduce a Structured 4D Latent Predictive Model, which predicts the evolution of a scene's 3D structure in a structured latent space conditioned on observations and textual instructions. Our representation encodes the scene holistically and can be decoded into diverse 3D formats, enabling a more complete and 3D consistent scene understanding. This structured 4D latent predictive model serves as a planner, generating future scenes that are translated into executable actions by a goal-conditioned inverse dynamics module. Experiments demonstrate that our model generates futures with strong visual quality, substantially better 3D consistency and multi-view coherence compared to state-of-the-art video-based planners. Consequently, our full planning pipeline achieves superior performance on complex manipulation tasks, exhibits robust generalization to novel visual conditions, and proves effective on real-world robotic platforms. Our website is available at https://structured-4d-model.github.io/.

**Analysis:**

### 1. 摘要翻译
视频预测模型在机器人领域展现出巨大潜力，但现有方法多基于2D视频序列，缺乏精确空间推理所需的3D几何理解。本文提出一种**结构化4D潜在预测模型**，通过在受观测和文本指令驱动的结构化潜在空间中，预测场景3D结构的演变。我们的表征对场景进行整体编码，并可解码为多种3D格式，从而实现更完整、3D一致的场景理解。该模型作为规划器，生成的未来场景由目标条件逆动力学模块转化为可执行动作。实验表明，相较于最先进的基于视频的规划器，本模型在视觉质量、3D一致性和多视图相干性上表现更优，在复杂操纵任务、跨视觉条件的泛化以及真实机器人平台上均取得了卓越的性能。

---

### 2. 方法动机分析
- **驱动力**：旨在克服2D视频预测模型在空间一致性上的天然缺陷，使机器人具备更强的环境物理认知能力。
- **痛点**：现有方法（如UniPi）在像素空间建模，受遮挡和视点变化影响大，缺乏多视图一致性；且直接预测RGB通常导致几何扭曲，无法胜任精细操作任务。
- **核心直觉**：通过将动态建模转移到结构化的3D潜在空间（而非像素空间），可以强制执行3D几何约束，从而获得跨视点、跨环境的高度稳健性。

---

### 3. 方法设计详解
**流程总结：**
1. **结构化编码**：利用多视图RGB-D图像，通过DINOv2特征嵌入，合并为基于稀疏体素网格（Sparse Voxel Grid）的3D潜在表征 $z_t$。
2. **4D动态预测**：采用两阶段预测流程。
   - **Single Dynamics Model (SD)**：使用条件流匹配（Flow Matching）预测下一时刻的粗略3D几何（体素位置）。
   - **Latent Generator (LG)**：预测各活动体素的细节特征（外观与几何信息）。
3. **解码与逆动力学规划**：将预测出的 $z_{t+1}$ 解码为点云，输入到目标条件逆动力学模块（ID），输出动作序列 $a_{1:H}$ 以驱动机器人。

**模型结构：**
- **SD与LG模块**：均采用Transformer架构，通过交叉注意力（Cross-Attention）注入文本指令和当前潜在状态。
- **逆动力学（ID）模块**：可选用学习型（基于点云的Diffusion Head）或学习-无损型（基于FPFH特征匹配+RANSAC+ICP的几何注册）。

---

### 4. 方法对比分析
- **本质区别**：从“像素级生成”转向“结构化3D潜在空间的流匹配”，本质上是物理世界模拟而非视觉概率建模。
- **创新点**：将场景建模为稀疏体素网格，实现了对场景的语义与几何双重表达，并引入流匹配以保证预测的稳定性。
- **适用场景**：对几何精度要求极高的精细操作任务，如物体装配、插入操作。

---

### 5. 实验分析
- **验证方法**：在ManiSkill3和RLBench基准上与视频生成方法（UniPi, TesserAct）及模仿学习（DP, DP3）对比。
- **关键结果**：在3D一致性指标（CD, depth error）上实现量级提升，在复杂操纵任务的零样本泛化能力上大幅领先。
- **主要优势**：多视图一致性强，抗干扰能力（光照、噪声、视点切换）优越。
- **主要局限**：对多视图传感器的校准有严格要求，且对于高精度的接触式任务，几何微小误差仍可能导致失败。

---

### 6. 实用指南
- **开源情况**：提供项目主页（https://structured-4d-model.github.io/）。
- **训练细节**：模型在4张NVIDIA H100上训练3天；使用条件流匹配 objective；Classifier-free guidance 概率为0.1。
- **迁移可能**：该架构具有良好的解耦性（动态预测器与逆动力学模块分离），可直接迁移至不同机器人实体，只需通过预训练的编码器（如TRELLIS）适配观测输入。

---

### 7. 总结
- **核心思想**：通过3D稀疏体素结构化表征，实现跨视点几何一致的4D动态预测。
- **速记版pipeline**：
  1. 多视图图像融合为3D稀疏体素表征。
  2. 利用流匹配在潜在空间预测未来几何状态。
  3. 将预测结果解码为点云。
  4. 计算几何注册或调用逆动力学模块执行动作。

**Key Findings:**

- We introduce a Structured 4D Latent Predictive Model, which predicts the evolution of a scene's 3D structure in a structured latent space conditioned on observations and textual instructions.
- Experiments demonstrate that our model generates futures with strong visual quality, substantially better 3D consistency and multi-view coherence compared to state-of-the-art video-based planners.
- Consequently, our full planning pipeline achieves superior performance on complex manipulation tasks, exhibits robust generalization to novel visual conditions, and proves effective on real-world robotic platforms.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01166v1)
- [arXiv](https://arxiv.org/abs/2607.01166v1)

---

<a id='2607.01147v1'></a>
## [EquiSteer: Cross-Attention Steering Towards a Fairer Text-Guided Image Generation](https://arxiv.org/abs/2607.01147v1)

**Authors:** Tatiana Gaintseva, Akshit Achara, Gregory Slabaugh, Jiankang Deng, Ismail Elezi

**Published:** 2026-07-01

**Categories:** cs.CV

**Abstract:**

Text-to-image diffusion models power everyday creative tasks, but they still reproduce the demographic biases in their training data. On common prompts such as ``a photo of a nurse,'' ``a photo of a CEO'', they skew their outputs toward one gender, driven by the statistics of training data rather than anything in the text. Existing debiasing methods show promise in narrow settings but require retraining, batch-level control, or prompt-specific tuning, limiting their scalability. We propose \emph{EquiSteer}, a training-free method that works per sample by steering cross-attention (CA) activations at inference time. For each target attribute, EquiSteer precomputes steering vectors from contrastive prompts. Then at generation time, a prompt-aware gate leaves attribute-specific prompts untouched, while for neutral ones it clears existing attribute signals from the CA activations and injects a target attribute. Across SD-1.5, SD-2.1, SDXL, and SANA, EquiSteer reduces the average parity gap by up to $87\%$, with minimal effect on image quality and text-image alignment. Code is available at \href{https://github.com/Atmyre/EquiSteer}{https://github.com/Atmyre/EquiSteer}.%

**Analysis:**

### 1. 摘要翻译
文本到图像扩散模型在日常创意任务中展现了强大能力，但仍会复现训练数据中的人口统计学偏差。在处理“护士”、“CEO”等常见提示词时，模型受训练数据分布影响，会显著偏向特定性别。现有去偏方法在窄域设置中表现尚可，但通常需要重新训练、进行批处理控制或针对特定提示词进行调优，限制了其可扩展性。我们提出了 EquiSteer，一种无需训练的方法，通过在推理阶段引导交叉注意力（CA）激活来实现样本级偏差修正。对于每个目标属性，EquiSteer 从对比提示词中预计算引导向量；在生成阶段，一个“提示词感知门控”机制在识别到特定属性提示时保持不变，而在处理中性提示词时，会清除原有的属性信号并注入目标属性。在 SD-1.5、SD-2.1、SDXL 和 SANA 上，EquiSteer 将平均公平性偏差（parity gap）降低了高达 87%，且几乎不影响图像质量和图文对齐。

### 2. 方法动机分析
- **驱动力**：旨在解决扩散模型在生成中表现出的顽固人口统计学偏差，提供一种无需额外训练或模型权值更新的通用去偏工具。
- **现有方法痛点**：
    - 微调类方法需模型权重访问权限，且跨架构迁移难。
    - 批处理引导方法（Guidance-based）无法针对单样本去偏，处理多属性时表现不佳。
    - 文本嵌入干预对提示词措辞极度敏感，缺乏空间控制能力。
- **核心直觉**：模型的 demographic 偏差直接编码在交叉注意力层的激活中，因此可以通过直接干预 CA 激活来实现语义空间的操作，而非仅仅修改输入提示词。

### 3. 方法设计详解
- **核心 Pipeline**：
    1. **预计算**：针对目标属性，通过正向（含目标）和反向（不含目标）提示词对，计算每个 cross-attention 层的“引导向量” ($s_{lt}^X$)。
    2. **提示词感知门控 (Gate)**：在推理过程中，利用 dot-product 计算 CA 激活与引导向量的响应强度 ($dp_{lt}$)。如果响应值超过阈值 ($thr^a$)，说明提示词已包含明确属性，则跳过干预，保持原始生成。
    3. **子空间正交化 (Orthogonalization)**：对于中性提示词，先将 CA 激活投影到目标属性子空间的补空间中，从而彻底清除原有的偏差信号。
    4. **自适应注入 (Re-weighting)**：根据预计算的属性表现强度，动态计算注入幅值 ($\alpha$)，在正交化后的激活中精准加入目标属性向量。
- **技术细节**：通过 Gram-Schmidt 构建正交基，并使用基于 CLIP/BLIP-VQA 的统计量来校准门控阈值，保证了操作的自动化与模型无关性。

### 4. 方法对比分析
- **本质区别**：EquiSteer 是一种**训练即时、基于推理期激活操作的干预方法**。它不仅是简单的偏移，而是通过“检测（门控）+ 清除（正交化）+ 补偿（自适应注入）”的完整闭环，解决了现有方法在处理多属性和特定词汇时的脆弱性。
- **创新贡献**：引入了显式的属性敏感门控机制，有效解决了“过度修正”和“混淆属性”问题。

### 5. 实验分析
- **验证方法**：在 SD-1.5, SD-2.1, SDXL, SANA 等多种架构上，对性别、种族、年龄、身体类型、眼镜等维度进行审计。
- **关键结果**：在保证图文对齐和图像质量的前提下，平均 parity gap 降低至接近均匀分布，且在复杂提示词（如长上下文、多主体）下表现出极强的泛化性。
- **优势**：训练免费、即插即用、且不改变模型原有的指令遵循能力。

### 6. 实用指南
- **开源情况**：代码已开源至 [https://github.com/Atmyre/EquiSteer](https://github.com/Atmyre/EquiSteer)。
- **迁移可能**：该方法本质是对 attention 层的高维向量空间进行几何投影操作，具有极高的架构无关性。只需更换不同模型的 attention hook，无需针对特定 backbone 进行特殊适配。

### 7. 总结
- **核心思想**：通过交叉注意力的即时正交化与语义注入，实现无损的公平性引导。
- **速记版 Pipeline**：
    1. **门控检测**：判断提示词是否已包含特定属性。
    2. **正交清除**：在中性提示词中移除所有预编码的属性偏差。
    3. **精准注入**：根据强度校准，向子空间注入目标属性向量。
    4. **规范化输出**：确保图像质量稳定。

**Key Findings:**

- We propose \emph{EquiSteer}, a training-free method that works per sample by steering cross-attention (CA) activations at inference time.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01147v1)
- [arXiv](https://arxiv.org/abs/2607.01147v1)

---

<a id='2607.01139v1'></a>
## [SD-RouteFusion: Ego-Trajectory Prediction with SD-Map Route Conditioning](https://arxiv.org/abs/2607.01139v1)

**Authors:** Sviatoslav Voloshyn, Bruno K. W. Martens, Wangxin Liu, Jakob Vinkås, Junsheng Fu

**Published:** 2026-07-01

**Categories:** cs.CV

**Abstract:**

This paper presents SD-RouteFusion, a deployable end-to-end ego-trajectory prediction method that fuses a front-facing camera, vehicle kinematics, and a navigation route derived from a Standard Definition (SD) map. Unlike approaches that rely on High Definition (HD) map geometry, SD-RouteFusion aligns the learning objective with scalable and production-ready SD-map route inputs, enabling route-aware prediction without requiring HD-map infrastructure. First, we demonstrate that SD-map route prior provides a powerful long-horizon semantic prior. Through a comprehensive study on a large-scale real-world dataset comprising 480k driving scenarios across 10 European countries and the U.S., we quantify the value of SD-route conditioning: incorporating SD-map routes yields a 10.5% ADE improvement over an image-and-kinematics baseline, while our full fusion strategy achieves a 16.9% ADE reduction given a prediction horizon of 8 seconds. The fusion strategy consists of a dual-hypothesis design paired with a gated classifier, to ensure robustness under route corruption and visual uncertainty. Finally, to support broader evaluation, we release an SD-route generation toolkit that enables SD-route-conditioned ego-trajectory prediction on all datasets containing ego pose and future trajectories. Together, SD-RouteFusion establishes a practical path toward robust, route-aware ego-trajectory prediction at scale.

**Analysis:**

### 1. 摘要翻译
本文提出了 SD-RouteFusion，一种可部署的端到端自我轨迹预测方法。该方法融合了前视摄像头输入、车辆运动学信息以及从标准定义（SD）地图中派生的导航路线。与依赖高清（HD）地图几何结构的方法不同，SD-RouteFusion 将学习目标与可扩展且生产就绪的 SD 地图路线输入对齐，从而在无需 HD 地图基础设施的情况下实现路线感知预测。研究表明，SD 地图路线先验提供了强大的长程语义引导。通过对涵盖欧洲 10 国及美国 48 万个驾驶场景的真实数据集进行广泛研究，我们量化了 SD 路线调节的价值：与仅基于图像和运动学的基线相比，引入 SD 地图路线可将平均位移误差（ADE）降低 10.5%，而我们的全融合策略在 8 秒预测周期下实现了 16.9% 的 ADE 降幅。该融合策略采用了双假设设计配合门控分类器，确保了在路线损坏和视觉不确定性情况下的稳健性。此外，我们发布了一个 SD 路线生成工具包，支持在任何包含自我位姿和轨迹的数据集上进行 SD 路线调节的轨迹预测。

### 2. 方法动机分析
*   **驱动力**：旨在构建一种不依赖成本高昂、难以维护的 HD 地图，同时又能实现鲁棒且高精度的自动驾驶车辆轨迹预测方案。
*   **痛点**：现有方法要么严重依赖 HD 地图（扩展性差），要么仅基于单目视觉（长程意图预测能力不足），且大多数端到端模型缺乏对输入信号（如地图噪声、传感器误差）的鲁棒筛选机制。
*   **研究假设**：SD 地图提供的路级语义信息虽缺乏车道级精度，但足以作为长程驾驶意图的“强先验”；通过双分支建模与门控机制，可以在确定性场景下利用路线先验，并在路线失效（如路况变化、定位漂移）时自动回退至视觉感知。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入处理**：图像通过 ResNet-18 + Lift-Splat-Shoot 转化为鸟瞰图（BEV）特征；运动学数据通过 GRU 编码；SD 地图生成的路线与运动学特征融合。
    2.  **双分支预测**：分别生成“图像主导”和“路线主导”两个轨迹假设（$T_i, T_r$）。
    3.  **交叉注意力机制**：利用镜像交叉注意力（Mirror Cross-Attention），让路线先验与局部视觉特征交互，实现语义增强。
    4.  **门控选择**：门控分类器依据当前特征和两种假设的差异，输出权重 logit，动态选择最优分支作为最终输出。
*   **模型核心**：引入了**双分支架构+后期门控（Late-stage gating）**。不同于传统早期融合（容易导致模型在路线错误时“过信任”），该设计确保了预测对路线噪声的容错性。
*   **公式意义**：$g = \text{MLP}([E_r, E_i, T_r - T_i])$，门控通过学习两种假设的冲突程度与环境上下文，动态切换预测逻辑。

### 4. 方法对比分析
*   **本质区别**：从“全量融合”转向“竞争性选择”，将路线信息作为可信度动态评估的辅助，而非硬性约束。
*   **创新贡献**：提出了一种无需 HD 地图的路线调节范式，并通过自监督方式训练门控逻辑，无需额外标注即可识别预测失效。
*   **适用场景**：适用于城市道路等复杂环境，特别是当高精度地图覆盖不全或因道路施工导致地图数据失效的生产级自动驾驶场景。

### 5. 实验分析
*   **验证方法**：在包含 48 万个驾驶场景的 ZOD 扩展集上进行测试，评估 ADE、FDE 和 Miss Rate。
*   **关键结论**：相比于不带路线信息的 baseline，SD-RouteFusion 在转弯场景下的 FDE 提升显著（28%），验证了 SD 路线在处理遮挡和长程意图时的有效性。
*   **主要优劣**：优势在于显著增强了长程轨迹的准确性与系统鲁棒性；局限性在于严重依赖 SD 地图数据的有效性，且在极端地理定位错误下仍有失败风险。

### 6. 实用指南
*   **开源情况**：已开源代码库与 SD 路线生成工具包（详见论文脚注）。
*   **实现细节**：建议使用 GNSS 数据辅助 SD 路径匹配；门控训练时使用温度系数 $\tau$ 调节软标签，以平滑切换边界。
*   **迁移可能**：该门控架构可直接迁移至需要多模态信息融合（如多传感器、多地图源）的预测与决策任务中，用于解决模态间不一致带来的性能劣化。

### 7. 总结
*   **核心思想**：通过双分支架构与动态门控机制，实现路线先验与实时感知的智能择优融合。
*   **速记版pipeline**：
    1. 提取图像与地图路线特征；
    2. 并行生成两组候选预测轨迹；
    3. 利用交互式注意力增强语义对齐；
    4. 门控模块依据可靠性自动选择最佳路径。

**Key Findings:**

- First, we demonstrate that SD-map route prior provides a powerful long-horizon semantic prior.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01139v1)
- [arXiv](https://arxiv.org/abs/2607.01139v1)

---

<a id='2607.01133v1'></a>
## [Towards Metric-Agnostic Trajectory Forecasting](https://arxiv.org/abs/2607.01133v1)

**Authors:** Markus Knoche, Daan de Geus, Bastian Leibe

**Published:** 2026-07-01

**Categories:** cs.CV, cs.RO

**Abstract:**

Accurate trajectory forecasting of surrounding traffic participants is a core capability for autonomous driving, enabling vehicles to anticipate behavior and plan safe maneuvers. We observe that current state-of-the-art forecasting models on Argoverse 2 and the Waymo Open Motion Dataset tailor their training objectives to the different benchmark metrics. Because these metrics encourage conflicting behavior, we propose a paradigm change for trajectory forecasting: training models with metric-agnostic probabilistic objectives and treating metric optimization as a downstream task applied to the predictive distribution. Concretely, we introduce Trajectory Distribution Evaluation (TraDiE) policies, metric-specific policies that map a predictive distribution to the set of $K$ trajectories and confidences required by trajectory forecasting metrics. We evaluate this framework by introducing DONUT-NLL, which adapts the training objective of the state-of-the-art trajectory forecasting model DONUT to directly optimize the predictive distribution. Using our policies, DONUT-NLL achieves state-of-the-art results on all metrics of the Waymo motion prediction benchmark.

**Analysis:**

这是一篇关于自动驾驶轨迹预测领域的重要研究，其核心思想在于将**轨迹预测模型训练**与**度量标准优化**解耦。

### 1. 摘要翻译
准确的交通参与者轨迹预测是自动驾驶的核心能力。我们观察到，当前 Argoverse 2 和 Waymo 开放运动数据集上的主流预测模型，其训练目标往往是针对特定基准度量标准定制的。由于这些度量标准往往鼓励相互冲突的行为，我们提出了一种轨迹预测的范式转换：**在训练时使用度量无关（metric-agnostic）的概率目标，并将度量优化作为应用于预测分布的下游任务。** 具体而言，我们引入了轨迹分布评估（TraDiE）策略，这是一种针对特定度量标准的策略，将预测的概率分布映射为度量所需的 $K$ 条轨迹和置信度。我们通过引入 DONUT-NLL 来评估此框架，该框架调整了 DONUT 模型的训练目标，以直接优化预测分布。使用我们的策略，DONUT-NLL 在 Waymo 运动预测基准的所有度量标准上均达到了最先进水平。

### 2. 方法动机分析
*   **驱动力**：消除预测模型对特定评测指标的“过拟合”。
*   **痛点**：当前模型为了刷高排行榜，将损失函数直接与特定评测指标（如 minFDE 或 soft mAP）耦合。由于不同指标要求不同（例如 minFDE 要求聚类准确，mAP 要求覆盖面广），导致模型行为扭曲，甚至需要为不同指标训练多个模型。
*   **核心假设**：如果模型能通过训练学习到一个高质量、且校准良好（well-calibrated）的**完整概率分布**，那么通过设计合理的后处理策略（即 TraDiE 策略），就可以针对任何指标获取最优解，无需针对特定指标进行冗余训练。

### 3. 方法设计详解
*   **解耦训练（DONUT-NLL）**：放弃 Winner-Takes-All (WTA) 等针对距离误差的训练损失，转向直接优化预测的**负对数似然（NLL）**。模型预测代理的未来轨迹分布，且支持不同分布族（拉普拉斯、广义高斯等）。
*   **TraDiE 策略（评估层）**：
    1.  **输入**：由预测模型生成的完整轨迹概率分布 $p_n(\mathbf{X}_n)$。
    2.  **采样**：从分布中进行蒙特卡洛采样（Monte Carlo sampling），得到一组预测点集。
    3.  **度量映射**：
        *   **minFDE 策略**：通过梯度下降寻找 K 个点，最小化期望的 minFDE。
        *   **soft mAP/Miss Rate 策略**：基于贪心启发式算法，迭代选择被最多评价窗口覆盖的点，同时避开已被覆盖的窗口，将预测分布转换为符合指标要求的 K 条轨迹及相应置信度。

### 4. 方法对比分析
*   **本质区别**：传统方法是在训练阶段“猜测”评估规则，而本文是在评估阶段“执行”评估规则。
*   **创新贡献**：提出了一种通用的、模型无关的后处理框架（TraDiE），证明了通过简单的 NLL 训练得到的分布具有更强的鲁棒性和泛化潜力。
*   **适用场景**：适用于所有输出概率分布或轨迹集合的预测模型，特别是在需要跨指标评测的场景中。

### 5. 实验分析
*   **结论**：在 Waymo 基准上，单模型在采用 TraDiE 策略后，能同时在 minFDE 和 (soft) mAP 上达到 SOTA。
*   **优势**：极大地提高了模型的可重用性，单一模型即可覆盖所有指标需求。
*   **局限**：推理阶段引入了额外的后处理开销（策略计算）；另外在处理极端多模态时，分布可能出现失准（如 QCNet 的实验所示）。

### 6. 实用指南
*   **开源**：代码位于 `vision.rwth-aachen.de/TraDiE-policies`。
*   **关键实现**：
    *   训练损失函数改用 `Negative Log-Likelihood`。
    *   在推断时，针对不同指标调用对应的 `TraDiE` 策略，而非直接输出原始预测。
*   **迁移建议**：该方法非常适合迁移到任何拥有独立评价指标集的多模态预测架构中，只需要将后处理部分替换为文中定义的采样策略即可。

### 7. 总结
*   **核心思想**：训练学分布，推断套策略，指标优化事后处理。
*   **速记版 Pipeline**：
    1.  训练时使用负对数似然（NLL）学习概率分布。
    2.  推断时从模型预测的概率分布中进行蒙特卡洛采样。
    3.  根据具体评估指标，应用特定优化策略筛选 K 条最佳轨迹。
    4.  计算最终指标完成评估。

**Key Findings:**

- We observe that current state-of-the-art forecasting models on Argoverse 2 and the Waymo Open Motion Dataset tailor their training objectives to the different benchmark metrics.
- Because these metrics encourage conflicting behavior, we propose a paradigm change for trajectory forecasting: training models with metric-agnostic probabilistic objectives and treating metric optimization as a downstream task applied to the predictive distribution.
- Concretely, we introduce Trajectory Distribution Evaluation (TraDiE) policies, metric-specific policies that map a predictive distribution to the set of $K$ trajectories and confidences required by trajectory forecasting metrics.
- We evaluate this framework by introducing DONUT-NLL, which adapts the training objective of the state-of-the-art trajectory forecasting model DONUT to directly optimize the predictive distribution.
- Using our policies, DONUT-NLL achieves state-of-the-art results on all metrics of the Waymo motion prediction benchmark.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01133v1)
- [arXiv](https://arxiv.org/abs/2607.01133v1)

---

<a id='2607.01111v1'></a>
## [FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement](https://arxiv.org/abs/2607.01111v1)

**Authors:** Haoran Hao, Shahram Najam Syed, Jeffrey Ichnowski, Jeff Schneider

**Published:** 2026-07-01

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Robot policies inevitably encounter failures when deployed in real environments. Naive retries often repeat the same mistakes, while many existing recovery methods rely on human intervention. In this paper, we propose Failure-Aware Retry (FAR), a framework that enables robots to learn from previous failures at test time, adapt their behavior accordingly, and eventually complete the task autonomously. FAR combines Failure-Contrastive Preference Adaptation, which constructs preference learning data from failures to steer the policy away from previously unsuccessful behaviors, with lightweight action perturbations during retries to encourage local exploration. We further incorporate successful recovery trajectories into a training loop for continual policy improvement. Experiments in both simulation and real-world manipulation tasks show that FAR substantially improves success rates and robustness, with average gains of 17.6% over the standard diffusion policy in simulation and 11.7% in the real world. In addition, FAR significantly improves data efficiency under both reset and timestep budgets during continual policy improvement by exploiting informative failure cases.

**Analysis:**

### 1. 摘要翻译
机器人策略在真实环境中部署时不可避免会遇到失败。简单的重试往往会重复同样的错误，而许多现有的恢复方法又过度依赖人工干预。在本文中，我们提出了“失败感知重试”（Failure-Aware Retry, FAR）框架，使机器人能够在测试时从之前的失败中学习，相应地调整行为，并最终自主完成任务。FAR结合了“失败对比偏好适应”（Failure-Contrastive Preference Adaptation, FCPA），通过构建失败案例的偏好学习数据，引导策略避开之前失败的行为；同时在重试期间引入轻量级动作扰动以鼓励局部探索。我们进一步将成功的恢复轨迹纳入持续策略改进的训练循环中。在仿真和真实世界的操作任务实验表明，FAR显著提高了成功率和鲁棒性，在仿真中较标准扩散策略平均提升了17.6%，在真实世界中提升了11.7%。此外，通过利用高信息量的失败案例，FAR显著提高了持续策略改进阶段的数据效率。

### 2. 方法动机分析
*   **驱动力**：旨在解决离线训练的策略在面对“分布外”（OOD）失败状态时，因缺乏针对性监督而导致持续失败的问题，并摆脱对人工修正的依赖。
*   **现有方法痛点**：
    *   **重试失效**：基础策略在相同状态下重试倾向于陷入局部最优，重复触发之前的错误。
    *   **数据利用不足**：传统在线学习未能充分利用失败案例来明确策略的“边界”。
    *   **人为成本**：依赖人机交互（如DAgger）修正数据代价高昂。
*   **研究假设**：通过对比失败动作与探索出的高价值动作，可以实时微调策略分布，使其从失败中“负向学习”，同时结合扰动探索，能更有效地实现自主恢复。

### 3. 方法设计详解
FAR分为**测试时适应（FCPA）**与**持续策略改进**两个阶段：
*   **失败归因（Failure Attribution）**：利用IQL（隐式Q学习）训练的保守Critic评估轨迹。对于失败轨迹，计算沿时间轴的价值差值（$\Delta V_t$），将价值下降最剧烈的时刻标记为失败原因，提取为**负样本**。
*   **对比偏好适应（FCPA）**：
    1.  **构造正样本**：在失败状态附近从策略采样一组动作，通过Critic评分筛选出距离适中且价值最高的动作作为**正样本**。
    2.  **优化**：定义偏好损失函数，通过最小化负样本的偏好误差和最大化正样本的偏好误差（借鉴DPO），在几秒钟内快速更新策略参数。
*   **轻量级扰动**：在重试执行时，注入经指数平滑处理的高斯噪声，在保留原始分布趋势的前提下，强制机器人跳出原策略的决定性失误，引导至新状态空间。
*   **持续策略改进**：将包含“恢复成功”后的轨迹纳入Replay Buffer，利用Critic对轨迹质量加权，通过“优势加权去噪目标”对策略进行离线-在线联合微调。

### 4. 方法对比分析
*   **本质区别**：FAR通过测试时的短步数梯度更新（梯度微调），而非仅仅是对行为进行过滤（如BGR的拒绝采样），这使得策略能从本质上改变对当前状态的响应逻辑。
*   **创新贡献**：首次将对比学习思想用于测试时失败恢复，并建立了一套“失败归因-快速自适应-持续改善”的闭环系统。
*   **适用场景**：适用于基于生成式策略（如Diffusion Policy）的机器人操作任务，特别是在长程、复杂且易发生分布偏移的场景。

### 5. 实验分析（精简版）
*   **验证方法**：在ManiSkill、RoboSuite、RoboMimic等9个仿真任务及3个真实xArm任务上进行对比实验。
*   **关键结论**：在保持相同算力预算下，FAR通过更具针对性的失败恢复，显著缩短了任务成功所需的平均重试次数和环境重置成本。
*   **主要优势**：不仅提升了单次任务成功率，还通过利用失败经验加速了策略在复杂场景下的收敛速度。
*   **主要局限**：测试时微调依赖于Critic的准确性；目前的失败检测仍基于简单的环境反馈，而非深层的意图理解。

### 6. 实用指南
*   **实现细节**：
    *   **Critic预训练**：Critic必须与扩散策略共享骨干编码器，确保特征空间对齐。
    *   **超参数**：$\rho$（失败归因分位数）设为10%，$K$（正样本数）设为8，测试时微调步数控制在5-10步。
    *   **扰动**：对于真实机器人，务必使用指数平滑（smoothing coefficient $\alpha=0.3$）来消除动作抖动。
*   **迁移建议**：该方法逻辑通用，可直接迁移至任何具有明确价值函数评分能力的离线训练策略框架。

### 7. 总结
*   **核心思想**：通过对比学习与在线梯度微调，将失败经验转化为策略的即时改进反馈。
*   **速记版pipeline**：
    1.  **找茬**：利用价值评估识别导致失败的动作片段。
    2.  **对比**：对比失败动作与高价值动作，构造正负样本对。
    3.  **微调**：测试时快速梯度更新，压制失败行为，强化成功路径。
    4.  **探索**：注入平滑扰动，辅助策略跳出失误陷阱。
    5.  **沉淀**：将高质量恢复轨迹存入缓冲区，循环持续优化。

**Key Findings:**

- In this paper, we propose Failure-Aware Retry (FAR), a framework that enables robots to learn from previous failures at test time, adapt their behavior accordingly, and eventually complete the task autonomously.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01111v1)
- [arXiv](https://arxiv.org/abs/2607.01111v1)

---

<a id='2607.01088v1'></a>
## [ROSA: A Robotics Foundation Model Serving System for Robot Factories](https://arxiv.org/abs/2607.01088v1)

**Authors:** Wenqi Jiang, Jason Clemons, Rowland O'Flaherty, Hugo Hadfield, Alperen Degirmenci, Shuran Song, Yashraj Narang, Christos Kozyrakis

**Published:** 2026-07-01

**Categories:** cs.RO, cs.DC

**Abstract:**

Robotics foundation models (RFMs) are making general-purpose robots increasingly practical for factory deployments. While RFM serving systems are central to this vision, existing systems are largely shaped by a single-robot, single-model assumption: inference is treated as an edge-computing problem handled by an on-robot or dedicated nearby GPU, and the serving objective is to minimize the latency of a single action model. In this paper, we propose ROSA, an RFM serving system for robot factories designed around three key principles. First, ROSA adopts shared GPU-pool serving, allowing a fleet of robots to access powerful server-class GPUs over the network in order to improve inference performance, battery duration, and GPU utilization. Second, ROSA provides a robotics-aware programming abstraction and system design that supports multi-model pipelines, per-task performance requirements, and failure handling. Third, ROSA uses factory-objective-driven scheduling to maximize SLO-qualified factory productivity rather than minimizing individual request latency. We implement ROSA on top of Ray Serve for distributed orchestration, with vLLM, PyTorch, and JAX as model-serving backends, and evaluate it on both real robots and synthetic large-scale workloads. The results show that ROSA improves factory productivity by up to 12.06x over conventional dedicated serving systems.

**Analysis:**

以下是对论文《ROSA: A Robotics Foundation Model Serving System for Robot Factories》的深度分析：

### 1. 摘要翻译
机器人基础模型（RFMs）使得通用机器人在工厂部署中越来越实用。然而，现有的推理服务系统主要局限于“单机器人、单模型”的假设，将推理视为一种边缘计算问题。本文提出了ROSA（Robotics Oriented Serving Architecture），这是一种专为机器人工厂设计的基础模型服务系统，旨在应对三个挑战：1）采用共享GPU池服务，允许机器人舰队访问服务器级硬件，以提升性能和利用率；2）提供机器人感知的编程抽象，支持多模型管道、任务级SLO和故障处理；3）采用工厂目标驱动的调度，旨在最大化SLO限定的工厂生产率，而非仅仅最小化单一请求延迟。实验结果显示，与传统的专用服务系统相比，ROSA将工厂生产率提升了高达12.06倍。

### 2. 方法动机分析
*   **驱动力**：作者观察到工厂环境对通用机器人（如人形机器人）的需求激增，而机器人基础模型（RFM）的高计算需求与机器人自身受限的SoC/边缘端能力之间存在巨大矛盾。
*   **现有痛点**：
    1.  **边端计算局限**：机器人机载芯片资源紧缺，无法运行大规模模型。
    2.  **独占部署低效**：专用服务器导致GPU在机器人执行动作时处于空闲状态，且缺乏机器人间的请求批处理（Batching）。
    3.  **单任务优化偏差**：现有系统仅以“最小化单一动作请求延迟”为目标，忽略了复杂的机器人多模型管道（如系统1动作、系统2推理、安全监测）的需求。
*   **研究假设**：通过服务器端的共享GPU集群，并基于工厂级的生产率目标进行协同调度，能够显著提升整体性能，且网络延迟在工厂稳定网络下是可以忽略的。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **任务声明**：用户通过YAML定义模型组件（S1/S2、监控、安全）、SLO和故障回退策略。
    2.  **调度决策**：ROSA调度器基于声明配置，综合考虑模型类型、约束条件和GPU资源，通过ILP（整数线性规划）计算放置策略、路由和批处理大小。
    3.  **在线执行**：机器人通过网络上传观测数据，网关路由请求至对应的GPU Worker；Worker执行推理后回传指令。
*   **核心模块**：
    *   **共享GPU池**：跨机器人批处理请求，提高利用率。
    *   **机器人感知抽象**：支持Pipeline编排，能够区分“目标耦合模型”（直接决定动作速度）和“义务模型”（周期性安全监测）。
    *   **调度器**：
        *   **Homogeneous Task**：二分搜索动作率 $f$，利用ILP优化配置。
        *   **Heterogeneous Task**：采用贪婪自适应前沿搜索，处理多任务类的复杂多维度调度，并使用“隔离打包+压缩”方法解决组合爆炸。

### 4. 方法对比分析
*   **本质区别**：从“单个设备视角”切换至“工厂集群视角”，将服务质量（QoS）从“低延迟”转化为“SLO合规条件下的高吞吐”。
*   **创新贡献**：提出工厂目标驱动（Weighted Action Throughput）的调度算法，结合了针对动作模型的批处理与多任务的资源均衡调度。
*   **适用场景**：大规模、多任务并行的工业机器人集群环境。

### 5. 实验分析
*   **验证方法**：使用Franka Panda机器人进行真实实验，并结合大规模合成负载验证扩展性。
*   **关键结果**：在32台虚拟机器人规模下，ROSA在复杂任务（如Assemble kit）上实现了相比传统专用模式12.06倍的吞吐提升；在异构任务下仍能保持高SLO合规率。
*   **主要优势**：打破了机器人硬件资源与计算能力的绑定，大幅提升了资源利用率。
*   **局限**：对网络稳定性有一定依赖（虽然论文论证了其对工业网络的可控性）；调度算法在极端动态变化下的实时重排负载仍有待优化。

### 6. 实用指南
*   **开源情况**：基于Ray Serve框架，配合OR-Tools进行ILP求解。
*   **实现细节**：关键在于模型Profiling，需要精确测量不同批大小下的延迟分布，这是调度策略有效的物理基础。
*   **迁移可能**：可直接迁移至任何依赖重型模型推理的边缘智能体集群（如自动驾驶车队）。

### 7. 总结
*   **核心思想**：通过集群化共享计算与工厂全局调度，实现机器人推理服务的效能最大化。
*   **速记版pipeline**：
    1. 定义任务需求（声明式）。
    2. 划分模型优先级与资源约束。
    3. 集群内均衡负载与请求批处理。
    4. 网络回传控制信号。
    5. 故障监测与自动化回退。

**Key Findings:**

- In this paper, we propose ROSA, an RFM serving system for robot factories designed around three key principles.
- The results show that ROSA improves factory productivity by up to 12.06x over conventional dedicated serving systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01088v1)
- [arXiv](https://arxiv.org/abs/2607.01088v1)

---

<a id='2607.01067v1'></a>
## [Human-Centric Transferable Tactile Pre-Training for Dexterous Robotic Manipulation](https://arxiv.org/abs/2607.01067v1)

**Authors:** Chi Zhang, Penglin Cai, Ziheng Xi, Haoqi Yuan, Hao Luo, Wanpeng Zhang, Sipeng Zheng, Chaoyi Xu, Zongqing Lu

**Published:** 2026-07-01

**Categories:** cs.RO, cs.CV

**Abstract:**

As an essential modality for dexterous and contact-rich tasks, tactile sensing provides precise force feedback that cannot be reliably inferred from vision. However, limited by hardware and data collection systems, existing datasets with tactility remain small in scale and narrow in contact coverage. Meanwhile, Vision-Language-Action (VLA) models with tactile modality are constrained on dynamics-agnostic post-training, which limits the performance ceiling on downstream tasks. In this paper, we present H-Tac, a large-scale tactile-action dataset with 160-hour egocentric human videos containing more than 300 tasks and 135k episodes. Building upon this, we propose Transferable Tactile Pre-Training (TTP), a system of tactile-based pre-training on human data for fine-grained robotic tasks. To bridge the gap between humans and robots, we use unified tactile and action spaces throughout the pre-training and post-training phases, preserving prior knowledge during human-to-robot transfer. By leveraging a tactile expert for future tactile prediction, our framework explicitly models the contact dynamics and precise physical interactions. Extensive experiments in simulation and on real robots demonstrate that our model achieves superior performance, exhibiting robust generalization and fine-grained manipulation capabilities. TTP paves the way for scalable tactile pre-training via human-to-robot transfer.

**Analysis:**

### 1. 摘要翻译
作为精细化和接触丰富任务的重要模态，触觉传感提供了无法通过视觉可靠推断的精确力反馈。然而，受限于硬件和数据收集系统，现有的触觉数据集规模较小且接触覆盖范围窄。同时，具有触觉模态的视觉-语言-动作（VLA）模型受限于动力学不可知的后训练，限制了下游任务的性能上限。在本文中，我们提出了 H-Tac，这是一个大规模触觉动作数据集，包含 160 小时的第一人称人类视频，涵盖 300 多个任务和 135k 个片段。在此基础上，我们提出了可迁移触觉预训练（TTP），这是一个在人类数据上进行触觉预训练的系统，用于细粒度机器人任务。为了弥合人机差距，我们在预训练和后训练阶段使用了统一的触觉和动作空间，在人机迁移过程中保留了先验知识。通过利用触觉专家进行未来触觉预测，我们的框架显式地建模了接触动力学和精确的物理交互。仿真和真实机器人上的广泛实验表明，我们的模型实现了卓越的性能，展现出强大的泛化能力和细粒度操纵能力。TTP 通过人机迁移为可扩展的触觉预训练铺平了道路。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决机器人难以获取大规模、高质量触觉数据的问题，利用易于获取的人类演示数据实现触觉模态的预训练。
*   **痛点**：现有VLA模型多为视觉导向，忽略触觉；且跨实体（embodiment）的触觉表示不统一，导致迁移困难；现有的触觉注入方式多为“后训练”阶段注入，导致预训练阶段缺失触觉先验。
*   **假设**：如果能在统一的触觉与动作空间中对人类演示数据进行预训练，并显式建模环境的接触动力学，则模型能够获得通用的“触觉先验”，显著提升机器人在接触密集型任务中的细粒度操作能力。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据构建**：收集包含视觉、触觉和动作的H-Tac大规模数据集，利用MANO手部模型将触觉数据投影到统一的UniTacHand空间。
    2.  **双专家预测架构**：在共享注意力机制基础上，设置“动作专家”（Action Expert）预测未来动作轨迹，同时引入“触觉专家”（Tactile Expert）预测未来触觉信号。
    3.  **流匹配预训练**：采用Flow Matching机制，将触觉预测视为生成任务，通过预测速度场来优化模型。
    4.  **触觉-动作流形保持门控（MPG）**：引入门控机制，根据动作和触觉流形的对齐程度动态调整特征上下文，确保在上下文不稳定时预测更稳健。
*   **关键算法**：通过sliced Wasserstein distance (SWD) 计算动作与触觉分布的Discrepancy，进而控制门控 $g = \exp(-D/\tau_g)$，仅当预测与 manifold 对齐时进行修正。

---

### 4. 方法对比分析
*   **本质区别**：与现有VLA模型相比，TTP在预训练阶段就将触觉模态深度耦合，而非仅作为下游微调的额外输入。
*   **创新贡献**：提出统一的“触觉与动作空间”解决跨实体迁移难题；引入“双专家预测”以显式建模触觉动力学；提出MPG机制缓解流匹配中的分布偏移。
*   **适用场景**：接触密集、需要精细力反馈的复杂操作任务（如拆卸、折叠、装配）。

---

### 5. 实验分析（精简版）
*   **关键结论**：在LIBERO、RoboCasa等基准测试中，TTP在各种Contact-rich任务中显著超越基线；在真实机器人实验中，实现了从人手到不同夹爪/灵巧手的零样本/少样本迁移。
*   **主要优势**：触觉先验不仅提升了操作精度，还提供了极强的跨实体泛化能力。
*   **主要局限**：对触觉数据同步和预处理要求较高；在极端视觉遮挡或环境变化下，依然依赖触觉反馈的准确性。

---

### 6. 实用指南
*   **开源情况**：项目主页：https://beingbeyond.github.io/TTP/。
*   **实现细节**：关键在于统一空间的映射（UniTacHand UV space），确保数据的一致性。MPG门控机制中的anchor构建是提高鲁棒性的关键。
*   **迁移可能**：可直接迁移至任何具有触觉反馈的机器人系统，只需适配不同实体的action/tactile维度即可。

---

### 7. 总结
*   **核心思想**：通过触觉-动作双专家预测，在预训练中植入通用的物理接触先验。
*   **速记版pipeline**：1.构建统一空间的人机触觉数据集；2.训练双专家模型实现触觉与动作的协同预测；3.引入动态门控机制增强预训练稳定性；4.将习得的先验迁移至机器人进行后训练。

**Key Findings:**

- In this paper, we present H-Tac, a large-scale tactile-action dataset with 160-hour egocentric human videos containing more than 300 tasks and 135k episodes.
- Building upon this, we propose Transferable Tactile Pre-Training (TTP), a system of tactile-based pre-training on human data for fine-grained robotic tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01067v1)
- [arXiv](https://arxiv.org/abs/2607.01067v1)

---

<a id='2607.01015v1'></a>
## [SuperFlex: Deformable Superquadrics for Point Cloud Decomposition](https://arxiv.org/abs/2607.01015v1)

**Authors:** Gabriel Tavernini, Elisabetta Fedele, Tiago Novello, Leonidas Guibas, Marc Pollefeys, Francis Engelmann

**Published:** 2026-07-01

**Categories:** cs.CV

**Abstract:**

Superquadrics have proven to provide a compact, geometrically meaningful representation for 3D objects. However, existing methods suffer from limited reconstruction accuracy, are restricted to rigid primitives, and lack robustness to partial point clouds. In this work, we present SuperFlex, an enhanced framework that expands the expressive power and applicability of superquadric decompositions. First, we introduce a novel loss formulation which significantly improves reconstruction accuracy. Second, we include bending and tapering deformations, enabling high-fidelity representation of curved and asymmetric geometries. Finally, we leverage these high-quality decompositions as supervision to train a model that is robust to partial real-world point clouds. Experiments demonstrate substantial improvements in reconstruction accuracy over both optimization- and learning-based baselines while maintaining a highly compact primitive representation.

**Analysis:**

这是一份针对论文《SuperFlex: Deformable Superquadrics for Point Cloud Decomposition》的深度分析报告。

### 1. 摘要翻译
超二次曲面（Superquadrics）为3D物体提供了紧凑且具有几何意义的表示。然而，现有方法在重构精度上受限，仅限于刚性基元，且对局部点云缺乏鲁棒性。本文提出了 **SuperFlex**，一个增强型框架，通过以下贡献扩展了超二次曲面分解的表达能力与适用性：首先，引入了一种显著提高重构精度的创新损失函数；其次，引入弯曲（bending）和锥化（tapering）变形，使模型能高保真地表示弯曲及非对称几何体；最后，利用这些高质量分解作为监督信号，训练了一个对现实世界局部点云具有鲁棒性的模型。实验表明，该方法在保持紧凑基元表示的同时，显著优于现有的优化和学习基准。

### 2. 方法动机分析
*   **驱动力**：旨在解决现有超二次曲面分解方法在几何表达能力不足（无法拟合复杂形状）以及对非完整点云（遮挡/噪声）缺乏鲁棒性的问题。
*   **现有痛点**：
    *   **表达限制**：传统超二次曲面仅能表示刚性、对称形状，对现实中弯曲、非对称物体拟合效果极差。
    *   **监督不足**：仅依赖Chamfer距离会导致基元间产生间隙，且难以捕捉精细几何细节。
    *   **数据完备性依赖**：现有模型多在理想的完整点云上训练，难以应对现实扫描场景中的遮挡问题。
*   **核心假设**：引入参数化变形（弯曲/锥化）并采用联合体积与表面损失（IoU + SDF），能够显著提升基元对复杂物体的拟合质量，并可作为高质量“伪真值”引导模型学习鲁棒的结构补全能力。

### 3. 方法设计详解
*   **流程总结**：
    1.  **Encoder-Decoder结构**：基于Transformer架构，通过Point-Voxel CNN提取点特征，通过Transformer Refinement精炼超二次曲面查询（Queries）。
    2.  **形变建模**：在基础的11个参数（t, R, s, ϵ）基础上，增加8个变形参数（2个锥化 τ，6个弯曲 β），构成19维参数空间。
    3.  **多目标损失函数**：结合可微IoU损失（针对整体形状）与SDF损失（针对局部表面细节），并加入重叠惩罚项。
    4.  **两阶段训练**：先在完整点云上实现高精度分解，再将其作为伪监督信号，通过Occlusion Augmentation训练出一个能处理局部输入、具备结构补全能力的鲁棒模型。
*   **关键公式意义**：
    *   $q(\mathbf{x}; \Theta) = 1$：引入变形算子 $D_{\tau,\beta}^{-1}$ 的反演，将变形后的点映射回 canonical 空间。
    *   $L_{\text{IoU}}$：衡量全局占位重合度，减少物体内部空洞。
    *   $L_{\text{SDF}}$：利用放射状距离（Radial Distance）通过可微SDF梯度驱动局部细化。

### 4. 方法对比分析
*   **本质区别**：从“刚性拟合”转向“可变形基元分解”，并利用重构质量作为监督信号，而非仅仅依赖点到点匹配。
*   **创新贡献**：
    *   提出了可微分的联合体积/表面损失，有效解决基元重叠和空洞问题。
    *   通过引入 bending/tapering 变形，极大地拓宽了超二次曲面对非凸、非对称零件的表达范围。
*   **适用场景**：适用于机器人场景理解、CAD物体逆向工程，以及室内外复杂场景的结构化分解。

### 5. 实验分析（精简版）
*   **关键结果**：在ShapeNet数据集上，SuperFlex 的 IoU 指标大幅领先于对比基准（SuperDec等）。且相比 Marching Primitives 等高复杂度方法，其基元数量更少且推理速度极快。
*   **主要优势**：极高的重构保真度，极紧凑的参数表示，对遮挡点云表现出极强的泛化能力。
*   **主要局限**：在处理极度复杂的非结构化场景时，基元数量固定（P=16）可能成为捕捉微小零件的瓶颈。

### 6. 实用指南
*   **开源情况**：项目主页：https://superflex3d.github.io
*   **实现细节**：
    *   **变形实现**：需注意形变算子（Tapering/Bending）的顺序（Inverse rigid -> Inverse bending -> Inverse tapering），计算梯度时需保证算子可微。
    *   **优化超参**：SDF 截断距离 $\delta=0.05$，SDF 权重 $\lambda_{\text{SDF}}=3.2$，训练需注意 IoU 温度参数 $\tau_{\text{IoU}}$ 的动态调整。
*   **迁移建议**：该方法中“以高精度分解作为伪监督”的思路极易迁移到其他神经基元建模任务（如球面或圆柱拟合）中。

### 7. 总结
*   **核心思想**：通过参数化形变扩展超二次曲面，并利用体积+表面损失联合优化。
*   **速记版pipeline**：
    1.  提取点特征并转化为超二次曲面基元。
    2.  应用弯曲与锥化形变算子进行几何表达。
    3.  结合体积IoU与表面SDF损失进行联合约束。
    4.  利用优化后的结果作为伪真值进行鲁棒性微调。

**Key Findings:**

- In this work, we present SuperFlex, an enhanced framework that expands the expressive power and applicability of superquadric decompositions.
- First, we introduce a novel loss formulation which significantly improves reconstruction accuracy.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.01015v1)
- [arXiv](https://arxiv.org/abs/2607.01015v1)

---

