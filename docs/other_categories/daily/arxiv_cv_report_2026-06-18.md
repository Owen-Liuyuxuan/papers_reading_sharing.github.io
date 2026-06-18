time: 20260618

# Arxiv Computer Vision Papers - 2026-06-18

## Executive Summary

# 每日报告执行摘要：计算机视觉前沿（2026-06-17）

## 一、主要主题与趋势

本期10篇论文集中反映了三个核心趋势：

1. **机器人灵巧操作与VLA模型的深度融合**：超过半数论文涉及灵巧操作（第2、3、5、8、10篇），其中视觉-语言-动作（VLA）模型成为主流框架，同时出现对VLA常识能力的关键性质疑（第6篇）。
2. **零样本与数据效率**：第2、3篇强调零样本迁移（sim-to-real或长时程操作），第5篇探索从日常人类视频低成本获取操作数据，第9、10篇则追求模型轻量化与高效推理。
3. **多模态与主动感知**：第4篇提出将感知视为推理过程，第7篇通过全景重投影提升3D场景理解，体现了感知范式的演进。

## 二、重要创新论文

- **第3篇**（J. Kim等）——零样本长时程灵巧操作，通过多视图3D接地VLM推理，首次在无需微调的情况下完成复杂序列操作，具有里程碑意义。
- **第9篇**（K. Duan等）——Moebius：仅0.2B参数实现10B级图像修复性能，提出极轻量级框架，为边缘部署开辟新路径。
- **第10篇**（Y. Zhang等）——可逆神经网络适配器用于一步流匹配，将生成式流匹配引入机器人动作规划，显著提升推理速度与稳定性。
- **第6篇**（N. Kachaev等）——首次系统评估VLA模型的基础常识与知识保留能力，揭示当前模型“看似智能实则缺乏世界知识”的隐患，对模型设计具有警醒作用。

## 三、新兴研究方向

1. **VLA常识与鲁棒性评估**：第6篇开创性地将评估视角从任务性能转向基础知识，预计将催生VLA模型的知识蒸馏与常识注入研究。
2. **从人类视频中被动学习操作**：第5篇利用日常视频而非实验室演示，降低数据获取成本，与第3篇的零样本推理形成互补。
3. **移动操作（Pedipulation）**：第8篇在轮式双足机器人上实现物体滑动操作，融合行走与操作，拓展了机器人形态学边界。
4. **主动感知即推理**：第4篇将主动感知形式化为一种推理过程，有望统一目标驱动感知与多模态理解。

## 四、建议精读论文

- **第3篇**（零样本长时程灵巧操作）：对于从事机器人操作、VLM推理的研究者，是必读的突破性工作。
- **第6篇**（VLA常识评估）：所有VLA相关从业者应重点阅读，以理解当前模型的根本局限。
- **第9篇**（轻量级图像修复）：对模型压缩、高效部署感兴趣的读者不容错过。
- **第10篇**（可逆适配器+流匹配）：对生成式模型与机器人规划交叉领域的研究者具有直接参考价值。

*注：第1、7篇分别涉及自动驾驶安全标签和3D场景理解，虽非前沿热点但技术扎实，建议根据具体需求选择性阅读。*

---

## Table of Contents

1. [Learning to Annotate Delayed and False AEB Events: A Practical System for Extreme Class Imbalance and Asymmetric Label Noise](#2606.19186v1)
2. [Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement](#2606.18953v1)
3. [Zero-Shot Long-Horizon Dexterous Manipulation via Multi-View 3D-Grounded VLM Reasoning](#2606.19340v1)
4. [Native Active Perception as Reasoning for Omni-Modal Understanding](#2606.19341v1)
5. [Do as I Do: Dexterous Manipulation Data from Everyday Human Videos](#2606.19333v1)
6. [Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models](#2606.19297v1)
7. [OneCanvas: 3D Scene Understanding via Panoramic Reprojection](#2606.19253v1)
8. [Mobile Pedipulation for Object Sliding via Hierarchical Control on a Wheeled Bipedal Robot](#2606.19233v1)
9. [Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance](#2606.19195v1)
10. [Invertible Neural Network Adapter for One-Step Flow Matching in Robot Manipulation](#2606.19194v1)

---

## Papers

<a id='2606.19186v1'></a>
## [Learning to Annotate Delayed and False AEB Events: A Practical System for Extreme Class Imbalance and Asymmetric Label Noise](https://arxiv.org/abs/2606.19186v1)

**Authors:** Mengxiang Hao, Xin Jiang, Xinghao Huang, Wenliang Su, Zhiteng Wang, Junjie Rao, Xiaotian Yang, Wei Liao, Chengyu Han, Gen Liang, Yulun Song, Zhitao Xu, Xianpeng Lang

**Published:** 2026-06-17

**Categories:** cs.RO, cs.LG

**Abstract:**

Autonomous Emergency Braking (AEB) optimization relies on accurately annotated real-world trigger events, particularly rare but critical delayed and false AEB triggers that expose system deficiencies. However, these minority samples comprise less than 5% of thousands of daily triggers, making manual annotation prohibitively expensive at scale. We present the first automated AEB annotation framework to address this problem. During development, we identified two fundamental challenges that severely impair delayed/false trigger annotation accuracy: (1) Extreme class imbalance where delayed/false triggers are overwhelmed by true triggers; (2) Asymmetric label noise where mislabeled majority samples (true triggers) suppress minority samples (delayed/false triggers) learning. To overcome these challenges, we propose two key innovations: (1) Specific data augmentation that synthesizes realistic samples by manipulating focal target attributes, transplanting ego-vehicle dynamics, and masking non-focal agents; (2) noise suppression using stable hardness estimation and probe-guided adaptive threshold to clean mislabeled true trigger samples. Crucially, we deploy our model as a practical annotation system with full-stack architecture, efficiently identifying critical delayed/false triggers from thousands of daily AEB events. Production results demonstrate 80% improvement in recall of delayed/false triggers and 50% reduction in manual workload. Beyond immediate gains, the system enables continuous self-improvement through accumulated high-quality annotations, establishing a necessary data foundation for on-vehicle AEB system optimization

**Analysis:**

这是一篇关于自动驾驶领域自动紧急制动（AEB）事件标注的实用性研究论文。以下是深度分析：

### 1. 摘要翻译
自动紧急制动（AEB）优化依赖于准确标注的真实触发事件，特别是极少数但至关重要的“延迟触发”和“误触发”事件。然而，这些少数类样本仅占每日成千上万个触发事件的不到5%，导致大规模人工标注成本高昂。本文提出了首个自动AEB标注框架。研究发现，AEB标注存在两大核心挑战：(1) **极端的类不平衡**，少数类被多数类淹没；(2) **非对称标签噪声**，多数类（正常触发）中混入约1%的噪声，误导模型学习方向。为此，我们提出了双重策略：(1) **基于AEB物理特性的数据增强**，通过操纵目标属性和移植车辆动力学来合成真实样本；(2) **基于探针的噪声抑制机制**，利用训练过程中的样本稳定性进行自适应噪声过滤。生产环境部署结果显示，该系统将少数类召回率提高了80%，人工标注负担降低了50%。

### 2. 方法动机分析
*   **驱动力**：工业界亟需大规模、高精度的AEB触发事件标注，以优化感知与控制算法，但人工标注由于样本极度稀缺且分布不均，已无法承载。
*   **现有方法痛点**：传统重采样（如SMOTE）在AEB这种具有复杂时空动态特性的多模态数据上容易产生分布偏移；现有的鲁棒学习方法多针对图像分类，未考虑到AEB数据中“少数类标注极准、多数类标注包含少量噪声”的非对称特性。
*   **研究假设**：通过引入领域物理知识进行合成增强，配合基于训练动态（Hardness）的噪声探测，可以有效修正模型对多数类的偏见及噪声干扰。

### 3. 方法设计详解
*   **流程总结**：
    1.  **统一特征嵌入**：将自车和周围障碍物特征转化为统一的$A \times T \times C$张量（$A$为障碍物数，$T$为时间， $C$为特征维度）。
    2.  **Transformer Backbone**：通过层级注意力机制提取时空上下文，经过全局池化得到分类向量。
    3.  **AEB靶向增强（Augmentation）**：
        *   **策略I**：修改真实目标轨迹，通过调整距离模拟“延迟”，通过移位或零掩码模拟“误触发”。
        *   **策略II（核心创新）**：建立属性银行，从真实难例中提取“自车-目标”向量对，直接替换正常样本，保持物理真实性。
    4.  **噪声抑制（Noise Suppression）**：监控模型对样本的预测置信度，计算指标 $H_i = |1 - p_i|$，并使用EMA（指数移动平均）平滑，利用“噪声探针”自动设定过滤阈值 $\tau = \text{mean}(H_{\text{noise}}) + \epsilon$。

### 4. 方法对比分析
*   **本质区别**：与通用重采样不同，本文方法利用了**物理约束（碰撞时间TTC、车辆动态）**进行合成，保证了生成样本的可信度；同时利用**非对称噪声特性**，精准打击误标注的多数类，而非泛泛去噪。
*   **创新贡献**：提出了“AEB专用数据合成+基于训练动力学的噪声抑制”双重机制。
*   **适用场景**：极端不平衡的工业时间序列分类任务，尤其是在标注本身存在非对称噪声的场景中效果显著。

### 5. 实验分析
*   **关键结果**：在DP@R90%和FP@R90%指标上分别达到60.1%和59.7%，大幅领先DeepEnsemble等基线。
*   **优势**：在保持高性能的同时，极大降低了95%的样本量需求。
*   **局限**：对前端感知系统具有强依赖（上游感知系统发生漂移时，模型需要定期重训练）。

### 6. 实用指南
*   **开源建议**：作者虽未直接提供代码库，但核心算法组件（Transformer Backbone、EMA Hardness、物理增强逻辑）描述清晰，可直接复现。
*   **实现要点**：
    *   **超参数**：EMA平滑系数 $\alpha$ 建议设置为0.05。
    *   **数据平衡**：合成少数类与真实少数类样本比例控制在1:1最为稳健。
    *   **迁移建议**：可直接迁移至其他自动驾驶安全功能（如ELK车道偏离预警、RAEB后方紧急制动）。

### 7. 总结
*   **核心思想**：利用物理仿真增强少数类，结合训练不确定性过滤噪声。
*   **速记版pipeline**：
    1.  利用AEB物理逻辑，合成真实感极强的少见事故样本。
    2.  利用模型训练时的犹豫程度（置信度），筛除标注错误的正常事件。
    3.  构建专家模型，通过迭代循环，实现数据的自动化清洗与标注。

**Key Findings:**

- We present the first automated AEB annotation framework to address this problem.
- To overcome these challenges, we propose two key innovations: (1) Specific data augmentation that synthesizes realistic samples by manipulating focal target attributes, transplanting ego-vehicle dynamics, and masking non-focal agents; (2) noise suppression using stable hardness estimation and probe-guided adaptive threshold to clean mislabeled true trigger samples.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19186v1)
- [arXiv](https://arxiv.org/abs/2606.19186v1)

---

<a id='2606.18953v1'></a>
## [Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement](https://arxiv.org/abs/2606.18953v1)

**Authors:** Kinam Kim, Namiko Saito, Heecheol Kim, Katsushi Ikeuchi, Jaegul Choo, Yasuyuki Matsushita

**Published:** 2026-06-17

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models can generalize across diverse manipulation tasks, but their imitation-learning-based policies remain brittle in precise physical interactions due to compounding execution errors; Can a reinforcement learning policy trained purely in simulation improve the robustness of real-world VLAs zero-shot? Residual RL, which learns a corrective policy on top of a frozen VLA, offers a natural framework, but existing approaches face a fundamental sim-to-real dilemma: privileged-state methods require lossy distillation for deployment; image-based methods suffer from the visual domain gap; and real-world RL is costly and unsafe. We propose an object-centric residual RL framework that refines VLA actions using object poses, enabling a compact observation space that transfers consistently between simulation and reality. To align the two domains, we additionally replay the same teleoperation demonstrations in simulation to train a sim counterpart of the real-world VLA. The residual RL policy is trained only in simulation with pose noise injection and dropout, and transfers zero-shot to the real robot. Across five manipulation tasks on a real Franka Research 3 (FR3) robot, our method improves the success rate from 42% to 76% zero-shot, and the improved rollouts can be further reused to retrain the base VLA for self-improvement without additional teleoperation. Project page: https://www.microsoft.com/en-us/research/articles/object-centric-residual-rl/

**Analysis:**

以下是对《Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement》一文的深度分析：

### 1. 摘要翻译
视觉语言动作（VLA）模型虽能泛化多种操作任务，但其基于模仿学习的策略在执行精确物理交互时，常因累积误差而显得脆弱。本文探讨能否通过仅在仿真中训练的强化学习（RL）策略，零样本（zero-shot）提升现实世界VLA的鲁棒性。我们提出一种“以物体为中心的残差RL框架”，它利用物体位姿作为紧凑的观测空间，实现了仿真与现实之间的一致性转移。通过在仿真中重放与现实一致的遥操作演示来对齐领域，并施加位姿噪声及Dropout训练，该残差策略能直接迁移至真实机器人。在五项真实FR3机器人任务中，该方法将平均成功率从42%提升至76%，且改进后的滚动数据可用于基准VLA的自我改进，无需额外遥操作。

### 2. 方法动机分析
*   **驱动力**：解决VLA因模仿学习导致的精度不足，同时避免传统Sim-to-Real（如蒸馏、真实机器人RL）带来的高昂成本或性能损失。
*   **痛点**： privileged-state（特权状态）方法需依赖真实环境无法获取的信息，而基于图像的Sim-to-Real方法存在不可忽视的视觉领域鸿沟。
*   **核心直觉**：通过建立“领域无关”的观测接口（物体位姿+本体状态+基准动作），跳过视觉层面的对齐，直接在低维空间进行动作残差修正。

### 3. 方法设计详解
*   **流程Pipeline**：
    1.  **配对训练**：在仿真与现实中分别训练两个VLA，利用相同的遥操作演示数据，确保两者动作分布对齐。
    2.  **残差训练**：仅在仿真中训练一个基于MLP的残差策略，其输入为 $s_t = [s_{obj}, s_{prop}, a_{base}]$。
    3.  **零样本部署**：在真实环境中将VLA输出的动作与残差策略输出的修正值相加（旋转部分通过四元数乘法复合），残差模块完全冻结，无需任何微调。
*   **关键机制**：
    *   **位姿噪声增强**：对输入的6-DoF物体位姿进行随机采样扰动，模拟现实Pose估计器的误差。
    *   **Confidence-Gated Dropout**：训练中以概率 $\rho_{drop}$ 丢弃位姿信息，强制策略学习仅依赖本体状态和VLA动作的Fallback逻辑，并在部署时根据置信度 $c_t$ 动态开启。

### 4. 方法对比分析
*   **本质区别**：不试图去弥合“视觉”鸿沟，而是通过选择“对现实和仿真均稳健且等价”的中间变量（物体位姿）作为控制基底。
*   **创新点**：提出了一个无需视觉对齐、无需蒸馏、无需真实环境探索的残差RL训练范式，极大降低了系统复杂度和工程开销。

### 5. 实验分析
*   **关键结论**：在五个操作任务上，平均成功率实现了从42%到76%的跨越。消融实验证明，位姿Dropout是抗探测失败的关键，而噪声注入提升了任务精度。
*   **局限性**：对Pose估计器的实时性与准确性依赖极强；无法解决超出基准VLA分布（Out-of-Distribution）的大偏差；难以处理极小物体或微米级精度任务。

### 6. 实用指南
*   **开源/实现**：项目主页已提供。实现时需注意FoundationPose + SAM2的异步运行，确保Pose估计器不拖累控制循环。
*   **迁移逻辑**：该方法高度模块化，理论上可叠加于任意冻结的VLA之上，只要能获取任务相关物体的6-DoF位姿。

### 7. 总结
*   **核心思想**：利用领域无关的物体位姿接口，实现仿真残差策略的零样本部署。
*   **速记版Pipeline**：
    1. 训练一个对齐仿真与现实的VLA底座；
    2. 仿真内训练残差策略，引入位姿噪声与Dropout；
    3. 现实部署时，Pose估计器置信度低则自动Fallback；
    4. 收集修正后的轨迹数据，递归提升底座VLA性能。

**Key Findings:**

- We propose an object-centric residual RL framework that refines VLA actions using object poses, enabling a compact observation space that transfers consistently between simulation and reality.
- Across five manipulation tasks on a real Franka Research 3 (FR3) robot, our method improves the success rate from 42% to 76% zero-shot, and the improved rollouts can be further reused to retrain the base VLA for self-improvement without additional teleoperation.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.18953v1)
- [arXiv](https://arxiv.org/abs/2606.18953v1)

---

<a id='2606.19340v1'></a>
## [Zero-Shot Long-Horizon Dexterous Manipulation via Multi-View 3D-Grounded VLM Reasoning](https://arxiv.org/abs/2606.19340v1)

**Authors:** Jisoo Kim, Sangwon Baik, Taeksoo Kim, Sungjoo Kim, Junyoung Lee, Mingi Choi, Hanbyul Joo

**Published:** 2026-06-17

**Categories:** cs.RO

**Abstract:**

We present a zero-shot framework for long-horizon dexterous manipulation that grounds language instructions into executable 3D task plans from calibrated multi-view RGB images. Rather than training an end-to-end policy, our system uses a vision-language model (VLM) to produce reference-frame task grounding and primitive-level 2D keypoints, then lifts them into 3D via multi-view fusion. This lifting combines triangulation of view-wise VLM groundings with reference-view ray voting, which searches along a semantic camera ray for geometrically consistent candidates across neighboring views. The resulting 3D keypoints support both pick-and-place and tool-use: for tool-use, we retrieve an object-centric atomic action corresponding to the inferred skill category and align its stored 6D tool trajectory to the scene; for dexterous execution, we expand the lifted grasp keypoint into a task-conditioned grasp affordance region and generate feasible grasp-motion pairs with an arm-hand motion generator. Real-world experiments show improved 3D grounding accuracy and execution reliability over single-view RGB-D grounding and fine-tuned VLA baselines. We further demonstrate long-horizon manipulation through closed-loop status verification and replan, enabling zero-shot execution on unseen objects and tool-use tasks in novel scenes.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文的分析如下：

### 1. 主要贡献总结
该论文提出了一种无需训练（Zero-Shot）的端到端长周期灵巧操作框架，能够将自然语言指令转化为基于多视角RGB图像的精确3D任务执行计划。其核心贡献在于通过多视角融合技术提升了VLM（视觉语言模型）在复杂环境下的3D语义定位精度，并结合闭环状态验证实现了对未知物体和工具任务的可靠操作。

### 2. 关键创新点与方法论
*   **多视角3D语义融合（Multi-view 3D-Grounded VLM Reasoning）：** 不同于传统的RGB-D直接深度感知，该方法利用VLM获取2D关键点，并通过“射线投票法（Ray Voting）”在多视角间进行几何一致性搜索，从而在没有深度传感器的情况下实现高精度的3D定位。
*   **分层任务执行架构：** 系统将复杂的长周期任务解耦为语义层（VLM任务规划）、几何层（3D关键点提升）和执行层（原子动作检索与6D轨迹对齐）。
*   **闭环重规划机制：** 引入了状态验证与重规划循环，赋予了系统在动态或未知环境中自我纠错的能力，显著提升了执行的鲁棒性。
*   **灵活的操作策略：** 针对“拾取放置”与“工具使用”采取了不同的策略——前者生成任务条件下的抓取力场（Affordance），后者通过6D轨迹对齐实现复杂操作。

### 3. 对领域的潜在影响
*   **跨越“模拟到现实”的鸿沟：** 该研究证明了在无需大量特定任务数据训练的情况下，利用预训练VLM的通用推理能力即可实现高难度的灵巧操作，这为机器人学习从“数据依赖”向“通用智能”转型提供了重要路径。
*   **突破RGB-D局限：** 通过算法手段从多视角RGB图像中“重建”高精度的3D几何信息，降低了对昂贵且易受干扰的深度传感器（如激光雷达或结构光）的依赖。
*   **提升长序列任务可行性：** 解决了目前VLA（视觉-语言-动作模型）在长周期任务中容易“漂移”的问题，为机器人处理多步骤、复杂交互任务提供了更稳定的工程范式。

### 4. 相关领域与应用前景
*   **家用服务机器人：** 能够处理厨房、收纳等多种未知物品的操作需求。
*   **工业精密装配：** 在需要工具交互及精细姿态对齐的工业场景中具有应用潜力。
*   **辅助医疗机器人：** 对未知医疗器械的操作与精准定位。
*   **人机协作（HRC）：** 能够根据口头指令理解并完成复杂的辅助工作。

### 5. 可推断的潜在局限性
*   **计算延迟与实时性挑战：** 由于涉及多视角图像推理、VLM多轮调用及闭环重规划，该系统在处理极高实时性要求（如动态避障）的任务时，可能存在推理延迟问题。
*   **VLM的幻觉风险：** 尽管引入了3D几何约束，但系统依然依赖VLM对语义的理解，若VLM在复杂场景中产生语义错误，可能导致下游3D定位的失效。
*   **场景依赖性：** 论文虽然强调了Zero-Shot，但依然需要多视角相机的标定与合理的空间放置，对于极度拥挤或遮挡严重的场景，多视角配准的难度会呈几何级增长。
*   **硬件协同要求：** 该方法假设存在一个成熟的“手臂-手运动生成器（Arm-Hand Motion Generator）”，其实际表现极大地依赖于底层动作生成器的控制能力。

**专家点评：** 这篇论文的趣味性在于它巧妙地绕过了目前VLA模型“黑盒”的局限性，通过显式的几何几何推理（Geometric Reasoning）将大模型的语义能力与机器人的空间物理执行力进行了有效的“中间件”连接，是当前具身智能（Embodied AI）领域非常具有前瞻性的解题思路。

**Key Findings:**

- We present a zero-shot framework for long-horizon dexterous manipulation that grounds language instructions into executable 3D task plans from calibrated multi-view RGB images.
- We further demonstrate long-horizon manipulation through closed-loop status verification and replan, enabling zero-shot execution on unseen objects and tool-use tasks in novel scenes.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19340v1)
- [arXiv](https://arxiv.org/abs/2606.19340v1)

---

<a id='2606.19341v1'></a>
## [Native Active Perception as Reasoning for Omni-Modal Understanding](https://arxiv.org/abs/2606.19341v1)

**Authors:** Zhenghao Xing, Ruiyang Xu, Yuxuan Wang, Jinzheng He, Ziyang Ma, Qize Yang, Yunfei Chu, Jin Xu, Junyang Lin, Chi-Wing Fu, Pheng-Ann Heng

**Published:** 2026-06-17

**Categories:** cs.CV, cs.CL, cs.SD

**Abstract:**

Passive models for long video understanding typically rely on a "watch-it-all" paradigm, processing frames uniformly regardless of query difficulty, causing computational cost to grow with video duration. Although interactive frameworks have emerged, they often rely on global pre-scanning, and their context cost still scales with video length. We propose OmniAgent, the first native omni-modal agent that formulates video understanding as a POMDP-based iterative Observation-Thought-Action cycle. OmniAgent executes on-demand actions to selectively distill audio-visual cues into a persistent textual memory, effectively decoupling reasoning complexity from raw video duration. To operationalize this, we introduce (1) Agentic Supervised Fine-Tuning to bootstrap native active perception via best-of-N trajectory synthesis with dual-stage quality control, and (2) Agentic Reinforcement Learning with TAURA (Turn-aware Adaptive Uncertainty Rescaled Advantage), which leverages turn-level entropy to steer credit assignment toward pivotal discovery turns. Crucially, OmniAgent exhibits positive test-time scaling, where performance improves as the number of reasoning turns increases, validating the efficacy of active perception. Empirical results across ten benchmarks (e.g., VideoMME, LVBench) demonstrate that OmniAgent achieves state-of-the-art performance among open-source models. Notably, on LVBench, our 7B agent outperforms the 10$\times$ larger Qwen2.5-VL-72B (50.5% vs. 47.3%).

**Analysis:**

### 1. 摘要翻译
被动式视频理解模型通常依赖“全景观看”模式，导致计算成本随视频时长线性增长。尽管交互式框架有所涌现，但它们往往依赖全局预扫描，上下文成本依然随视频时长增加。我们提出了 **OmniAgent**，这是首个将视频理解转化为基于 POMDP（部分可观测马尔可夫决策过程）的迭代“观测-思考-行动”循环的原生全模态代理。OmniAgent 通过按需执行行动，选择性地将音视频线索提取为持久文本记忆，从而将推理复杂度与原始视频时长解耦。为此，我们引入了：(1) **智能体监督微调（Agentic SFT）**，通过带双阶段质量控制的 $N$ 选优轨迹合成来引导原生主动感知；(2) **智能体强化学习（Agentic RL）**，结合 **TAURA**（Turn-aware Adaptive Uncertainty Rescaled Advantage，转向自适应不确定性重加权优势函数），利用转向级熵引导信用分配至关键决策时刻。实证结果表明，OmniAgent 在十个基准测试（如 VideoMME, LVBench）上达到了开源模型的最优水平。值得注意的是，在 LVBench 上，我们的 7B 模型以仅 1/10 的参数量，性能超越了 Qwen2.5-VL-72B（50.5% vs 47.3%）。

### 2. 方法动机分析
- **驱动力**：打破“长视频=高计算开销”的诅咒，使模型具备类似人类“主动观察”的能力。
- **痛点**：现有模型多为“被动观察者”，要么进行昂贵的全局密集计算，要么依赖不可靠的预提取模块，导致信息瓶颈和严重的冗余。
- **核心直觉**：视频理解应是基于任务目标（Query）驱动的、动态的、迭代的搜索过程，而非静态的处理过程。

### 3. 方法设计详解
- **Pipeline**：
    1. **初始化**：持久记忆 $M_0$ 包含 Query 和元数据。
    2. **OTA 循环**：在每一步 $k$：
        - **观测 ($O_k$)**：Agent 根据 $M_{k-1}$ 和环境反馈的原始感知 $E_{k-1}$，生成文本摘要存入持久记忆。
        - **思考 ($T_k$)**：分析现有记忆与 Query 需求，定位缺失信息。
        - **行动 ($A_k$)**：执行四类原子算子：`aframes`（帧采样）、`aaudio`（音频提取）、`aclip`（音视频片段捕获）或 `aanswer`（终结输出）。
    3. **记忆更新**：环境执行 $A_k$ 获得新感知 $E_k$，将原始 $E_{k-1}$ 从记忆中清洗，只保留精炼的文本摘要 $O_k$。
- **关键算法 TAURA**：解决了 GRPO 在多步推理中“优势同质化”问题。通过计算各回合（Turn）的 token 熵，将轨迹层级的奖励重塑为回合级加权优势，强制模型在 RL 阶段聚焦于产生高不确定性（即决策分叉点）的“思考”行为，而非 trivial 的琐碎动作。

### 4. 方法对比分析
- **本质区别**：与仅将 LLM 作为“控制器”调动外部工具的传统代理不同，OmniAgent 将感知与推理统一在一个原生全模态架构中，且具备可学习的“观察策略”。
- **优势**：实现了计算量与视频时长解耦，推理深度由任务难度而非视频时长决定。

### 5. 实验分析
- **关键结论**：在 LVBench 上，7B 的 OmniAgent 以显著减少（约 73%）的帧数输入，超越了 72B 参数的被动模型。
- **优势**：展示了“正向测试时缩放”（Test-time Scaling），即增加推理轮数能持续提升准确率。
- **局限**：序列化的迭代交互机制引入了单机处理的延迟，未来需探索并行探索机制。

### 6. 实用指南
- **开源/复现**：代码已开源至 GitHub (harryhsing/OmniAgent)。
- **微调重点**：Agentic SFT 是成功基础，需使用高质量的轨迹合成数据，特别是包含错误修正（Self-correction）的轨迹，训练模型如何从诊断信号中恢复。
- **迁移建议**：该架构可直接迁移至任何具备长序列输入、稀疏信息提取需求的任务（如长文档阅读、复杂音频流分析）。

### 7. 总结
- **核心思想**：将视频理解定义为 POMDP，通过主动采样与记忆压缩实现任务驱动的按需感知。
- **速记版pipeline**：
    1. 接收目标 Query；
    2. 迭代执行：观测视频/音频 -> 思考推理 -> 执行采样动作；
    3. 精炼感知内容存入持久化文本记忆，丢弃原始媒体流；
    4. 达到置信度阈值后，输出最终结果。

**Key Findings:**

- We propose OmniAgent, the first native omni-modal agent that formulates video understanding as a POMDP-based iterative Observation-Thought-Action cycle.
- To operationalize this, we introduce (1) Agentic Supervised Fine-Tuning to bootstrap native active perception via best-of-N trajectory synthesis with dual-stage quality control, and (2) Agentic Reinforcement Learning with TAURA (Turn-aware Adaptive Uncertainty Rescaled Advantage), which leverages turn-level entropy to steer credit assignment toward pivotal discovery turns.
- Empirical results across ten benchmarks (e.g., VideoMME, LVBench) demonstrate that OmniAgent achieves state-of-the-art performance among open-source models.
- Notably, on LVBench, our 7B agent outperforms the 10$\times$ larger Qwen2.5-VL-72B (50.5% vs.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19341v1)
- [arXiv](https://arxiv.org/abs/2606.19341v1)

---

<a id='2606.19333v1'></a>
## [Do as I Do: Dexterous Manipulation Data from Everyday Human Videos](https://arxiv.org/abs/2606.19333v1)

**Authors:** Bhawna Paliwal, Haritheja Etukuru, William Liang, Pieter Abbeel, Nur Muhammad Mahi Shafiullah, Jitendra Malik

**Published:** 2026-06-17

**Categories:** cs.RO, cs.CV

**Abstract:**

How can we scalably generate data for robotic manipulation, especially on human-like platforms such as dexterous multi-fingered hands? Learning from human videos has recently emerged as a likely answer to this question. However, difficulties in estimating hand-object interaction and crossing the human-to-robot embodiment gap have hindered the adoption of abundant monocular RGB-only human videos as the primary source of robot manipulation data. In this work, we present DO AS I DO, an algorithm to reconstruct and retarget monocular RGB human videos to multi-fingered dexterous robotic hands. DO AS I DO reconstructs hand-object interactions from various egocentric and exocentric in-the-wild video sources. The algorithm then retargets these hand-object interaction estimates into a sequence of actions executable in the real world, yielding robot-complete manipulation data from disparate human videos. Overall, DO AS I DO outperforms previous state of the art in estimating hand-object interactions and extracting dexterous manipulation trajectories from RGB videos, as we show in experiments on datasets with ground truths and on a dataset of video clips collected online. Our experiments enable us to propose an efficacy playbook for practitioners collecting human data for manipulation.

**Analysis:**

## 1. 摘要翻译
我们如何为机器人操作（特别是像灵巧多指手这样的人形平台）扩展性地生成数据？从人类视频中学习已成为解决该问题的一种可行方案。然而，手-物交互估计的困难以及人类与机器人之间的形态差异（embodiment gap），阻碍了单目RGB视频作为机器人操作数据主要来源的广泛应用。在本文中，我们提出了 **DO AS I DO**，这是一种将单目RGB人类视频重构并重定向（retarget）到多指灵巧机器人手上的算法。DO AS I DO 从各种第一人称和第三人称的“野外”视频源中重构手-物交互，并将这些交互估计值重定向为现实世界可执行的动作序列，从而从分散的人类视频中获取完整的机器人操作数据。实验表明，DO AS I DO 在估计手-物交互和从RGB视频提取灵巧操作轨迹方面优于现有最先进技术，且在标准基准数据集及在线采集的视频集上均表现出色。最后，我们基于实验提出了关于采集人类操作数据的高效策略建议。

## 2. 方法动机分析
*   **驱动力**：利用互联网上海量的人类观察视频作为机器人灵巧操作的“数据矿藏”，以解决机器人领域数据匮乏的问题。
*   **现有痛点**：
    *   **重建难**：现有的手-物交互重建方法在处理“野外”视频（含运动模糊、遮挡、多样化物体）时，容易丢失姿态追踪（drift/lose lock）。
    *   **重定向偏差**：传统运动学重定向忽略了物理交互（接触、力），导致仿真中出现穿模、滑移或 grasp 不稳定。
    *   **基准质量低**：直接利用互联网视频往往伴随噪声，缺乏高质量的真实交互对齐。
*   **研究假设**：手-物交互的形状与姿态在生成模型的潜在空间中共享信息；通过物理仿真中的采样优化，可以弥合噪声参考轨迹与可行机器人控制之间的鸿沟。

## 3. 方法设计详解
该算法分为重构（Reconstruction）和重定向（Retargeting）两部分：
*   **手-物重构（关键在于Guided Diffusion）**：
    *   **流程**：利用SAM 3D（生成模型）作为基石，不仅做单帧重构，还引入了时间维度的“Guided Diffusion”机制。
    *   **核心逻辑**：在推理时固定物体形状（anchor frame），将上一帧的姿态作为先验，通过ODE积分并辅以旋转速度引导（Adaptive Guidance），在pose空间进行采样，从而在不需重新训练的情况下实现鲁棒的视频追踪。
*   **重定向（Dynamics-aware Optimization）**：
    *   **流程**：将重构的噪声轨迹作为参考，通过MPPI（Model Predictive Path Integral）采样优化生成机器人的动作。
    *   **关键组件**：
        1.  **Warmup Steps**：在正式跟踪前，预置H个热身步，让机器人先调整到稳态，避免初始接触失败。
        2.  **Random Force Perturbation**：在仿真中施加随机力，确保控制器对轻微干扰鲁棒。
        3.  **Transition Reward**：显式惩罚“静止”与“手持有”之间的转换失败，解决噪声轨迹导致的逻辑断层。

## 4. 方法对比分析
*   **本质区别**：传统方法多依赖于昂贵的MoCap数据或实验室受控环境，而本文方法专注于从“不可控的野外RGB视频”提取数据，并使用物理仿真进行轨迹修正。
*   **创新贡献**：
    *   **架构创新**：将SAM 3D生成模型重新适配为视频追踪器，无需针对视频做重新训练。
    *   **算法增强**：针对重定向提出了Warmup及扰动策略，解决了噪声轨迹导致的轨迹不合理问题。
*   **适用场景**：适用于任意单目RGB视频（如YouTube、EG04D等）到多指灵巧手的轨迹迁移，尤其擅长处理包含复杂接触的操作任务。

## 5. 实验分析（精简版）
*   **关键结论**：在DexYCB和HOI4D上均达到了SOTA性能；在150个“野外”视频的人类评估中，追踪质量偏好率领先基准67% vs 18%。
*   **主要优势**：极强的野外适应性，显著降低了数据处理的“20倍惩罚”（通过精细筛选与重构）。
*   **主要局限**：对刚性物体假设较强，且无法感知环境约束（如桌子、障碍物）的全局信息。

## 6. 实用指南
*   **开源建议**：关注作者发布的[项目主页](https://do-as-i-do.com)，复现时核心是配置好MuJoCo Warp模拟器。
*   **关键超参数**：`num_samples=1024`，`horizon=3.0s`，以及用于控制引导强度的`alpha_p`（基于旋转速度动态调整）。
*   **迁移建议**：重定向部分的MPPI框架非常通用，可直接迁移至其他灵巧手模型（如Allegro, Shadow Hand）。

## 7. 总结
*   **核心思想**：通过生成先验重构野外视频，结合物理采样优化消除仿真误差。
*   **速记版Pipeline**：
    1.  视频预处理与遮挡分割（SAM 3D）。
    2.  利用引导式扩散（Guided Diffusion）实现鲁棒的对象姿态追踪。
    3.  通过物理模拟器进行采样优化，将参考轨迹映射为机器人动作。
    4.  通过热身与干扰机制确保轨迹在现实世界中稳健可执行。

**Key Findings:**

- In this work, we present DO AS I DO, an algorithm to reconstruct and retarget monocular RGB human videos to multi-fingered dexterous robotic hands.
- Overall, DO AS I DO outperforms previous state of the art in estimating hand-object interactions and extracting dexterous manipulation trajectories from RGB videos, as we show in experiments on datasets with ground truths and on a dataset of video clips collected online.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19333v1)
- [arXiv](https://arxiv.org/abs/2606.19333v1)

---

<a id='2606.19297v1'></a>
## [Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models](https://arxiv.org/abs/2606.19297v1)

**Authors:** Nikita Kachaev, Andrey Moskalenko, Matvey Skripkin, Nikita Kurlaev, Daria Pugacheva, Albina Burlova, Mikhail Kolosov, Denis Shepelev, Andrey Kuznetsov, Elena Tutubalina, Aleksandr I. Panov, Alexey K. Kovalev, Vlad Shakhuro

**Published:** 2026-06-17

**Categories:** cs.LG, cs.RO

**Abstract:**

Embodied Vision-Language-Action (VLA) models are typically obtained by fine-tuning powerful pretrained VLMs on robotics data, yet it is unclear how much commonsense and factual knowledge they retain after adaptation. Failures on knowledge-sensitive tasks are ambiguous, conflating missing knowledge with poor generalization of low-level control. We introduce Act2Answer, a lightweight protocol that adapts VLM knowledge benchmarks to VLA evaluation by requiring agents to answer through action. Each question becomes a short tabletop episode where the agent performs a single object-placement action to select among candidate answers, yielding an action-grounded success rate with reduced control confounds. We curate a test suite of such environments across diverse commonsense and world-knowledge categories and introduce layerwise intent probing to localize answer-relevant information across the VLM backbone and action head. In a large-scale study of 7 VLA models and 9 VLM baselines, we systematically rank models across categories, finding that VLAs show solid performance on simple concepts while exhibiting larger gaps on richer semantic categories relative to their source VLMs, that VQA co-training is associated with better knowledge retention, and that answer-relevant signals peak in middle VLA layers but attenuate in upper layers. Act2Answer is available at https://tttonyalpha.github.io/act2answer/.

**Analysis:**

### 1. 摘要翻译
具身视觉-语言-动作（VLA）模型通常通过在机器人数据上微调强大的预训练视觉语言模型（VLM）获得，但目前尚不清楚这些模型在适配后保留了多少常识和事实性知识。在知识敏感型任务上的失败往往很模糊，混淆了知识缺失与低级控制能力不足的问题。我们提出了 **Act2Answer**，一种轻量级协议，通过要求智能体执行动作来回答问题，从而将 VLM 知识基准适配到 VLA 评估中。每个问题被转化为一个简短的桌面操作任务，智能体通过执行单个对象放置动作在候选答案中进行选择，从而实现受控的、基于动作的任务成功率评估。我们在包含多样化常识和世界知识类别的测试套件上进行了大规模评估，并引入“层级意图探测”（layerwise intent probing）来定位 VLM 主干网络和动作头中与答案相关的信息。研究发现，VLA 在简单概念上表现稳健，但在更丰富的语义类别上与源 VLM 相比存在巨大差距；VQA 联合训练有助于知识保留；且答案相关信号在 VLA 中间层达到峰值，但在上层发生衰减。

---

### 2. 方法动机分析
- **驱动力**：VLA 模型常被宣传为具备广义的常识与世界认知，但在实际评估中，现有的操作任务（如 LIBERO、CALVIN）往往将“任务成功率”与“视觉推理能力”深度耦合。作者旨在拆解这种耦合，明确 VLA 性能下降是因为丧失了语义理解，还是由于复杂的机器人控制导致的。
- **现有方法痛点**：端到端的任务成功评估过于粗糙，难以诊断模型究竟是“不懂知识”还是“不会操作”。
- **研究假设**：如果将知识评估从复杂的长期任务简化为“基于动作的二元选择题”，就能剥离掉低级运动控制的干扰，直接测量 VLA 模型内部残留的语义知识。

---

### 3. 方法设计详解
- **流程总结**：
    1.  **任务适配**：利用 LLM 将现有的 VLM 知识基准问题改写为简单的“二元动作指令”（如“把立方体放在正确的选项上”）。
    2.  **环境配置**：构建桌面模拟场景，将候选答案以图像形式放置，智能体通过移动机械臂将物体置于目标区域。
    3.  **评价度量（Soft Success Rate）**：定义容差半径 $\epsilon$，计算物体最终位置落在目标区域的软成功率。
    4.  **层级意图探测**：在模型各层提取隐藏状态，训练线性探测器（Linear Probe）预测正确答案，分析知识在模型内部的传递与衰减路径。
- **模型结构**：该方案是一个黑盒评估框架，不修改 VLA 模型本身，仅将其视为一个接收指令和图像、输出动作轨迹的策略模型。
- **算法解释**：核心贡献在于通过“动作即回答”的范式，规避了文本生成的不确定性，并利用空间交换配置（Swap Left/Right）消除模型的空间位置偏差。

---

### 4. 方法对比分析
- **本质区别**：传统评估关注“能不能完成复杂长序列任务”，Act2Answer 关注“能不能利用语义知识做出正确的单一判断”。
- **创新点**：
    1.  **动作化评估协议**：将抽象推理映射为物理交互。
    2.  **层级诊断工具**：通过探测器定位知识从视觉主干到动作头的流失过程。
- **适用场景**：适用于任何基于 Transformer 的 VLA 模型知识保持能力的基准测试。

---

### 5. 实验分析
- **关键结果**：
    1.  **能力分层**：VLA 在“颜色、形状”等浅层感知任务上保留良好，但在“情感、生物、规范”等深层语义任务上性能骤降。
    2.  **瓶颈效应**：探测实验显示，VLM 主干层仍保留大量知识，但到了动作预测层，这些信号发生了衰减。
- **局限性**：尽管减少了控制复杂度，但依然无法彻底排除机器人运动控制的细微影响。

---

### 6. 实用指南
- **开源情况**：已发布在 `tttonyalpha.github.io/act2answer`。
- **实现建议**：进行此类研究时，务必进行“空间对称性校验”（即对调答案位置），以防模型通过记住空间偏好（如“永远选左边”）来欺骗评估指标。
- **迁移可能**：该框架可以轻松迁移到任何机器人模拟环境（如 PyBullet 或 MuJoCo），用于评估新型 VLA 架构的预训练知识保持力。

---

### 7. 总结
- **核心思想**：通过动作交互评估 VLA 模型对常识知识的真实保留与使用能力。
- **速记版pipeline**：
    1. 将知识问题转化为二选一动作任务；
    2. 机器人模拟环境下执行物体放置；
    3. 根据落点计算成功率；
    4. 训练层级分类器探测内部知识流动。

**Key Findings:**

- We introduce Act2Answer, a lightweight protocol that adapts VLM knowledge benchmarks to VLA evaluation by requiring agents to answer through action.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19297v1)
- [arXiv](https://arxiv.org/abs/2606.19297v1)

---

<a id='2606.19253v1'></a>
## [OneCanvas: 3D Scene Understanding via Panoramic Reprojection](https://arxiv.org/abs/2606.19253v1)

**Authors:** Bartłomiej Baranowski, Dave Zhenyu Chen, Matthias Nießner

**Published:** 2026-06-17

**Categories:** cs.CV, cs.AI, cs.LG, cs.RO

**Abstract:**

Existing approaches to 3D scene understanding in Vision-Language Models (VLMs) either rely on complex, model-specific geometry encoders or large training budgets in pursuit of spatial reasoning. Instead, OneCanvas aggregates patch features from all views onto a single equirectangular panoramic canvas. Namely, each patch is unprojected to a 3D world coordinate using its depth and camera pose, then placed on the canvas at the continuous longitude and latitude of that point as seen from the canvas origin, with no rasterization or aggregation across overlapping views. A 3D position embedding of the patch's metric coordinates is added to its feature, restoring the depth lost when collapsing the world position to an angular canvas coordinate. Patches from all frames thus share one spatial coordinate system with no fusion or major architectural modifications of the backbone. The pretrained VLM consumes this representation as if it were an ordinary image. Because the canvas can be centered on any pose of interest, the same representation directly supports situated reasoning from a specific viewpoint, a common requirement in robotics and embodied AI. Thanks to this representation, we can also introduce a spatial pretraining curriculum: by procedurally placing patch features of objects, drawn from real images, at chosen 3D world positions on an otherwise empty canvas, we generate on-the-fly supervision spanning a broad range of spatial reasoning tasks, with answer distributions controlled to reduce spatial reasoning shortcuts. OneCanvas achieves state-of-the-art accuracy on SQA3D and VSI-Bench, and generalizes to out-of-distribution data on SPBench, using an order of magnitude less training compute than the strongest competing methods.

**Analysis:**

### 1. 摘要翻译
现有视觉语言模型（VLM）在三维场景理解方面，要么依赖复杂的模型专用几何编码器，要么需要庞大的训练预算。为此，我们提出了OneCanvas，它将各视角下的补丁特征聚合到一个单一的等距柱状全景画布上。具体而言，每个补丁利用其深度和相机位姿被反投影到三维世界坐标系，随后放置在画布上对应视点的连续经纬度位置，且无需对重叠视图进行栅格化或聚合。我们通过为补丁添加三维位置嵌入（metric coordinates），恢复了坐标转换中丢失的深度信息。如此，各帧补丁共享同一空间坐标系，无需对骨干网络进行任何架构改造。预训练VLM将该表征视作普通图像进行处理。由于画布中心可自由调整，该方法直接支持特定视点的空间推理，满足了机器人和具身智能的需求。基于此，我们引入了一种空间预训练课程：通过将来自真实图像的物体补丁放置在空白画布的三维位置上，实现任务导向的几何监督。OneCanvas在SQA3D和VSI-Bench上达到了SOTA水平，在SPBench上展现了强大的零样本泛化能力，且训练算力需求较强竞争对手降低了一个数量级。

### 2. 方法动机分析
- **驱动力**：解决VLM在3D场景推理中由于缺乏显式几何感知，往往陷入“统计捷径”（Statistical Shortcuts，即盲目依赖场景类别先验而非几何分析）的问题，同时规避复杂架构融合带来的高算力成本。
- **现有方法痛点**：
    1. **架构冗余**：添加额外的点云编码器或融合模块，增加了训练负担且难以对齐。
    2. **局部视角**：现有方法通常处理帧序列，缺乏对整个场景的统一空间索引，难以进行跨视点的长距离空间推理。
- **研究假设**：通过一种统一的、带有 metric 位置信息的等距全景投影表征，能让冻结权重的预训练VLM“阅读”几何结构，而无需重新设计骨干网。

### 3. 方法设计详解
- **流程总结**：
    1. **特征提取与3D投影**：利用Qwen3-VL的Frozen视觉编码器提取补丁特征，根据相机位姿(T)和深度(D)将像素补丁(u, v)反投影到3D世界坐标点(p)。
    2. **全景映射与编码**：计算各点在特定画布中心(c)和朝向(R)下的经纬度(θ, φ)，构建等距柱状图表征。
    3. **Metric Position Embedding**：设计了一个两层MLP，将3D世界坐标、径向距离和单位方向向量映射为136维向量，加在补丁特征上。这强制VLM在注意力机制中不仅关注外观，还能获取度量空间信息。
    4. **空间预训练课程**：在空白画布上模拟放置物体，让模型学习物体距离、方位、遮挡等几何规则，随后在真实数据上通过LoRA微调适应下游任务。
- **算法精髓**：不再是将几何作为“辅助输入”或“特征拼接”，而是将其作为一种“投影方式”和“位置嵌入”融入到VLM的Native Attention输入序列中，实现了即插即用的几何增强。

### 4. 方法对比分析
- **本质区别**：不引入专门的3D编码器或显式几何分支，而是通过“画布重组”和“位置嵌入”将3D几何信息“编码”进视觉Transformer的序列输入中。
- **创新贡献**：
    1. **全景空间统一化**：将所有视图聚合到一个共享坐标系的“画布”上。
    2. **空间预训练课程**：定义了一套无需真实3D场景即可生成的几何监督信号，极大降低了对大规模3D标注数据的依赖。

### 5. 实验分析
- **验证方法**：在SQA3D, VSI-Bench, SPBench三个基准集上对比当前SOTA方法。
- **核心结论**：在保持较小训练算力的情况下，在路由规划（Route）、度量距离等强几何依赖任务中表现显著优于其他方法。
- **局限性**：高度依赖高质量的深度图和准确的相机位姿；在大规模或室外复杂场景下存在泛化瓶颈。

### 6. 实用指南
- **开源情况**：项目主页 `https://baranowskibrt.github.io/onecanvas/`。
- **关键细节**：
    - **位置编码**：采用了16个log-spaced频率对位置坐标进行傅里叶映射，这对提升模型对度量尺度的感知至关重要。
    - **训练策略**：必须分两阶段，第一阶段（预训练）专门用于克服“场景捷径”的偏见，第二阶段（适配）使用LoRA锁定核心知识，避免灾难性遗忘。
- **迁移建议**：该方法完全适用于任意支持类似RoPE或Patch-based位置嵌入的Vision-Language Backbone，仅需修改投影函数即可适配不同的输入相机模型。

### 7. 总结
- **核心思想**：通过全景投影重构空间几何，将3D理解转化为VLM的视觉注意力推理。
- **速记版Pipeline**：
    1. 提取多视角图片patch特征。
    2. 结合位姿和深度计算3D空间坐标。
    3. 将patch按经纬度映射并添加度量位置嵌入。
    4. 空间模拟课程预训练。
    5. 下游下游任务微调。

**Key Findings:**

- OneCanvas achieves state-of-the-art accuracy on SQA3D and VSI-Bench, and generalizes to out-of-distribution data on SPBench, using an order of magnitude less training compute than the strongest competing methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19253v1)
- [arXiv](https://arxiv.org/abs/2606.19253v1)

---

<a id='2606.19233v1'></a>
## [Mobile Pedipulation for Object Sliding via Hierarchical Control on a Wheeled Bipedal Robot](https://arxiv.org/abs/2606.19233v1)

**Authors:** Yue Qin, Yulun Zhuang, Zelin Shen, Yanran Ding

**Published:** 2026-06-17

**Categories:** cs.RO

**Abstract:**

In this letter, we present a hierarchical control framework that enables wheeled bipedal robots to perform planar object sliding tasks with their wheeled legs. The proposed approach formulates a nonlinear model predictive controller (NMPC) based on a reduced-order three rigid bodies (TRB) dynamical model that explicitly accounts for the hip roll degree of freedom and multiple wheel-environment contact modes, which is essential for lateral stepping and pedipulation tasks. Within this framework, the NMPC simultaneously regulates robot locomotion and interaction forces, allowing the robot to stably execute both rolling and object manipulation behaviors. A trajectory-optimization-based robot-object motion planner is developed to generate reference motions that incorporate stick-slip transitions in ground-object contact. Two representative pedipulation motions, namely scooting and lateral sliding, are validated through real-world hardware experiments, in which the robot successfully retrieves a 1 kg object from under a desk and slides a 4 kg object over a distance of 0.228 m via scooting.

**Analysis:**

### 1. 摘要翻译
本文提出了一种层次化控制框架，使轮式双足机器人能够利用其轮式腿执行平面物体滑动（pedipulation）任务。该方法基于简化的“三刚体”（TRB）动力学模型构建非线性模型预测控制器（NMPC），该模型显式地考虑了髋关节侧摆（hip roll）自由度和多种轮-地接触模式，这对侧向行走和操控任务至关重要。框架内整合了一个基于轨迹优化的机器人-物体运动规划器，用于生成包含接触面粘滑转换（stick-slip transition）的参考运动。通过真实硬件实验验证了两种代表性动作：滑行（scooting）和侧向滑动（lateral sliding），成功实现了从桌面下检索1kg物体及推动4kg物体滑动0.228米。

### 2. 方法动机分析
*   **驱动力**：现有的轮式双足机器人多专注于运动学，缺乏与环境及物体的动态交互能力。利用腿部（pedipulation）进行非抓取式操控（non-prehensile manipulation）能显著提升机器人处理特殊任务（如推开障碍物）的能力。
*   **现有痛点**：现有方法通常简化了髋部关节（仅保留俯仰轴，省略侧摆轴），且多假设“无滑移”接触，导致无法处理真实物理环境中复杂的侧向交互及打滑现象。
*   **研究假设**：通过引入TRB模型并显式建模物体动力学与库仑摩擦约束，可以在非线性MPC框架下，联合优化机器人的平衡与物体操控行为。

### 3. 方法设计详解
*   **流程总结**：
    1.  **运动规划层（Motion Planner）**：基于TRBO模型（TRB+物体模型），利用直接配置法（Direct Collocation）预生成目标机器人状态及物体轨迹。
    2.  **控制层（NMPC）**：以规划出的轨迹为参考，实时求解包含接触约束（如库仑摩擦、接触模式切换）的非线性优化问题，输出最优交互力。
    3.  **执行层（WBC）**：利用全动力学模型，通过QP求解器将NMPC产生的任务空间加速度和力指令转换为关节力矩。
*   **模型结构与算法**：
    *   **TRB 模型**：将机器人简化为“躯干+两轮”模型，保留髋部侧摆DoF。与以往刚体模型不同，它不强加无滑移约束，而是通过接触雅可比矩阵（Contact Jacobian）动态建模接触速度。
    *   **TRBO 扩展**：将点质量物体动力学整合入模型，利用相依性布尔变量（$s_k \in \{0,1\}$）切换粘滞与滑动状态的互补约束，实现物体运动预测。

### 4. 方法对比分析
*   **本质区别**：本文不是将操纵任务解耦，而是通过在MPC层面显式集成物体动力学和非滑移约束，实现了操控与平衡的协同优化。
*   **创新贡献**：首次在轮式双足机器人上实现了考虑粘滑转换的动态非抓取式操控；提出的TRB模型能完整覆盖髋部侧摆与多接触模态。
*   **适用场景**：适用于工业搬运、障碍物清除等需要机器人与地面物体进行长距离、精细力交互的场景。

### 5. 实验分析
*   **验证方法**：通过MuJoCo仿真进行参数遍历，在Tron1真实平台上进行侧向滑动与滑行任务实验。
*   **关键结果**：在模拟中证实了该框架能搬运高达机器人自重以上的物体（23kg），且在物体质量估计误差存在的情况下，TRBO-NMPC展现出远优于仅针对机器人本体建模的控制器的鲁棒性。
*   **主要优势**：对动态接触约束建模精确，侧向操作能力强。
*   **主要局限**：目前严重依赖外部动捕系统获取物体位置，尚未实现板载感知；物体模型简化为点质量，忽略了旋转及复杂接触几何。

### 6. 实用指南
*   **实现细节**：建议关注 CasADi + Fatrop 求解器的组合，这是实现高频（>100Hz）NMPC求解的关键。
*   **迁移可能**：TRB模型可直接迁移至具备类似构型（躯干+双轮）的其他平衡机器人平台；TRBO的互补约束方法可用于任何涉及“推”的任务。

### 7. 总结
*   **核心思想**：通过非线性MPC协同优化机器人的侧向动态交互与物体滑动控制。
*   **速记版pipeline**：
    1. 预演：通过模型规划出机器人与物体的动作轨迹。
    2. 预测：NMPC实时根据当前状态调整平衡与推力。
    3. 反馈：WBC精确控制关节力矩以落实执行。
    4. 修正：通过模型中嵌入的摩擦互补约束处理打滑。

**Key Findings:**

- In this letter, we present a hierarchical control framework that enables wheeled bipedal robots to perform planar object sliding tasks with their wheeled legs.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19233v1)
- [arXiv](https://arxiv.org/abs/2606.19233v1)

---

<a id='2606.19195v1'></a>
## [Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance](https://arxiv.org/abs/2606.19195v1)

**Authors:** Kangsheng Duan, Ziyang Xu, Wenyu Liu, Xiaohu Ruan, Xiaoxin Chen, Xinggang Wang

**Published:** 2026-06-17

**Categories:** cs.CV

**Abstract:**

While 10B-level industrial foundation models have pushed the boundaries of image inpainting, their prohibitive computational costs severely hinder practical deployment. Constructing a highly optimized task-specific specialist offers a promising solution; however, extreme structural compression inevitably triggers a severe representation bottleneck. To conquer this, we propose Moebius, a highly efficient lightweight inpainting framework. We systematically reconstruct the diffusion backbone by introducing the Local-$λ$ Mix Interaction ($LλMI$) block. Comprising Local-$λ$ and Interactive-$λ$ modules, it elegantly summarizes spatial contexts and global semantic priors into fixed-size linear matrices, preserving complex latent interactions while drastically shedding parameters. Furthermore, to unlock the full representational capacity of this highly compact architecture, we synergistically pair it with an adaptive multi-granularity distillation strategy. Operating strictly within the latent space to avoid expensive pixel-space decoding, this strategy dynamically balances multiple gradient-based losses to achieve high-fidelity alignment. Extensive experiments across natural and portrait benchmarks demonstrate that this optimal synergy enables Moebius to rival or even surpass the generation quality of the 10B-level industrial generalist FLUX.1-Fill-Dev. Remarkably, Moebius achieves this using less than 2\% of the parameters (0.22B vs. 11.9B) while delivering a $>15\times$ acceleration in total inference time, setting a new efficiency standard for high-fidelity inpainting. Project page at https://hustvl.github.io/Moebius.

**Analysis:**

### 1. 摘要翻译
尽管10B参数规模的工业级基础模型在图像修复领域取得了突破，但其极高的计算成本严重阻碍了实际部署。构建高度优化的特定领域专家模型是一个有前景的解决方案，但极端结构压缩往往会导致严重的表征瓶颈。为解决此问题，我们提出了Moebius，一个高效的轻量级图像修复框架。我们通过引入Local-λ Mix Interaction (LλMI) 块重构了扩散骨干网，该块由Local-λ和Interactive-λ模块组成，能将空间上下文和全局语义先验高效总结为固定大小的线性矩阵，在保留复杂潜在交互的同时大幅降低参数量。此外，为释放该紧凑架构的表征潜力，我们将其与自适应多粒度蒸馏策略协同使用。该策略严格在潜在空间内运行以避免昂贵的像素级解码，通过动态平衡多种梯度损失实现高保真度对齐。实验表明，Moebius仅使用不到2%的参数（0.22B vs. 11.9B），同时实现超过15倍的推理加速，在自然和人像修复基准测试中达到了甚至超越了工业级10B模型FLUX.1-Fill-Dev的质量水平。

---

### 2. 方法动机分析
*   **驱动力**：在保持10B参数级工业模型修复质量的前提下，通过极端压缩实现高效的边缘端部署。
*   **痛点**：轻量化过程中，简单的算子替换（如DWConv、线性注意力）会导致“表征瓶颈”，造成生成质量的大幅衰减；此外，现有轻量级注意力机制缺乏处理交叉注意力（cross-attention）的能力。
*   **研究假设**：通过精心设计的结构（LλMI）和多粒度知识蒸馏协同，可以将轻量级模型的表征能力恢复到与大型专家模型相当的水平。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入与编码**：掩码图像与掩码矩阵通过预训练VAE编码进入潜在空间。
    2.  **LλMI 模块**：替代传统Transformer块。通过Local-λ总结空间上下文，通过Interactive-λ处理全局语义先验（LCG），辅以Mix-FFN。
    3.  **多粒度蒸馏**：训练过程中，在潜在空间内将学生模型的特征与教师模型（PixelHacker）进行多粒度对齐（16x16, 64x64）。
    4.  **动态损失平衡**：基于梯度范数动态调整各损失项权重，解决多任务学习收敛难的问题。
*   **关键公式**：LλMI模块通过将键值对压缩为固定大小的矩阵（$\lambda$），实现了线性复杂度的注意力计算，从而避开了耗时的点积操作。
*   **算法解释**：Local-λ通过1x1卷积和位置信息卷积构建内容与位置映射；Interactive-λ通过轻量位置嵌入将全局语义先验投影至 latent 空间，实现了高效的语义注入。

---

### 4. 方法对比分析
*   **本质区别**：Moebius不是单纯的结构压缩，而是将架构重构（LλMI）与特定任务的知识蒸馏（Multi-granularity distillation）进行了深度协同。
*   **创新贡献**：提出了LλMI块解决了交叉注意力在轻量化架构中的缺失问题；提出了基于梯度平衡的蒸馏策略，有效解决了极端压缩下的性能衰减问题。

---

### 5. 实验分析
*   **验证方法**：在Places2, CelebA-HQ, FFHQ等数据集上，对比了包括FLUX.1, SD3.5等主流工业级模型。
*   **结果**：参数量仅为0.22B，推理速度提升15倍以上，在保持高质量生成的同时，有效避免了工业大模型常见的模糊和结构伪影。
*   **优势**：极高的计算效率与顶级的视觉还原能力平衡。
*   **局限**：在极小背景区域的微小几何细节还原上，相较1B规模教师模型仍有微小差距。

---

### 6. 实用指南
*   **开源情况**：项目主页 `https://hustvl.github.io/Moebius`。
*   **实现细节**：建议使用Muon优化器，对蒸馏过程中的loss权重进行动态平衡（基于梯度范数），这是模型收敛的关键。
*   **迁移可能**：LλMI模块设计的线性注意力机制具有通用性，可迁移至其他需要高分辨率处理或资源受限的扩散模型任务中。

---

### 7. 总结
*   **核心思想**：通过架构重构与动态多粒度蒸馏，实现小参数模型对大模型能力的“精准拷贝”。
*   **速记版pipeline**：
    1. 引入压缩算子LλMI替换重型注意力块。
    2. 严格在潜在空间内与大模型蒸馏对齐。
    3. 使用梯度动态缩放平衡多项损失。
    4. 对小模型进行针对性任务微调。

**Key Findings:**

- To conquer this, we propose Moebius, a highly efficient lightweight inpainting framework.
- 11.9B) while delivering a $>15\times$ acceleration in total inference time, setting a new efficiency standard for high-fidelity inpainting.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19195v1)
- [arXiv](https://arxiv.org/abs/2606.19195v1)

---

<a id='2606.19194v1'></a>
## [Invertible Neural Network Adapter for One-Step Flow Matching in Robot Manipulation](https://arxiv.org/abs/2606.19194v1)

**Authors:** Yu Zhang, Kangyi Ji, Yongxiang Zou, Rongtao Xu, Feng Zheng, Long Cheng

**Published:** 2026-06-17

**Categories:** cs.RO

**Abstract:**

This paper presents an invertible neural network adapter for general robotic manipulation, designed to generate precise high-dimensional actions conditioned on multimodal observations, including visual, linguistic, and proprioceptive inputs, through a one-step denoising process. Built upon a flow-matching formulation, the proposed adapter effectively constrains the action generation trajectory within an invertible latent space, thereby enabling efficient and high-quality dexterous action synthesis with only a single inference step. Compared with conventional iterative flow-matching policies, the proposed framework substantially reduces inference complexity while maintaining strong action prediction accuracy and stability. Extensive experiments are conducted across a diverse set of simulation benchmarks and real-world robotic platforms to evaluate the effectiveness of the proposed method. Across simulation benchmarks, the proposed adapter consistently demonstrates superior or near state-of-the-art performance on a wide range of manipulation tasks. Furthermore, real-world experiments reveal a significant improvement in inference efficiency for vision-language-action (VLA) models, reducing the average inference latency from 110 ms to 61 ms while maintaining strong task performance.

**Analysis:**

# 论文方法分析与总结

### 1. 摘要翻译
本文提出了一种用于通用机器人操作的可逆神经网络适配器。该适配器通过单步去噪过程，根据视觉、语言和本体感觉等多模态观测数据，生成精确的高维动作。该方法基于流匹配（Flow Matching）架构，将动作生成轨迹约束在可逆潜在空间内，从而在仅需单步推理的情况下，实现了高效且高质量的灵巧动作合成。与传统的迭代流匹配策略相比，该框架在保持动作预测精度和稳定性的同时，显著降低了推理复杂度。实验表明，该方法在多个仿真基准测试和真实世界机器人平台中表现优异，特别是在视觉-语言-动作（VLA）模型中，将平均推理延迟从110ms降至61ms。

### 2. 方法动机分析
*   **驱动力**：解决流匹配在机器人策略学习中“性能与推理速度”之间的矛盾，即在保持单步高效推理的同时，克服非线性轨迹导致的近似误差。
*   **现有痛点**：现有单步流匹配方法（如ManiFlow、Mean Flow）通常引入过多的辅助损失函数来拟合轨迹，导致训练不稳定；或者难以准确捕捉复杂的多指动作动态，导致在非训练分布下泛化能力下降。
*   **研究假设**：通过引入可逆神经网络（INN）将动作轨迹约束在特定的可逆潜在空间，可以防止信息在降维/单步近似过程中的不可逆损失，从而保持动态的一致性。

### 3. 方法设计详解
*   **流程总结**：
    1. **粗采样**：利用传统的流匹配模型，根据观测 $o$ 和噪声 $x_t$ 进行单步去噪，获得原始估计 $\hat{x}$。
    2. **正向映射**：将 $\hat{x}$ 输入可逆神经网络（INN）的编码器 $g(\cdot)$。
    3. **正则化**：对输出进行球面归一化处理（$\tilde{y} = \frac{y}{\|y\|_2}$），将其投影至紧凑的潜在空间。
    4. **逆向映射**：将潜在向量 $\tilde{y}$ 通过逆网络 $g^{-1}(\cdot)$ 重建为最终的动作输出 $x_{pre}$。
*   **模型结构**：核心是“流匹配头 + INN Adapter”。INN基于耦合层构建，具有三角形雅可比矩阵，行列式为1，确保了转换的高效性和数值稳定性。
*   **算法解释**：核心逻辑是“双重约束”。损失函数包含两部分：一是重建损失（$\|x_{pre} - x_1\|^2$），确保输出动作接近真实值；二是潜在空间一致性损失（$\|\tilde{y} - g(x_1)\|^2$），确保降维后的表示不仅平滑且与数据分布一致。

### 4. 方法对比分析
*   **本质区别**：它不是直接学习速度场，而是将流匹配的结果作为初始猜测，通过INN对其进行“几何纠偏”，类似于在生成动作上加了一个非线性的保结构滤波器。
*   **创新贡献**：提出INN Adapter作为通用模块，能够无缝嵌入现有主流VLA框架（如Pi0.5, Qwen-VL-based models），且不需要增加推理步数。
*   **适用场景**：高维度、高精度的灵巧操作（如堆叠、精细装配），特别适合对实时性要求高的实时控制系统。

### 5. 实验分析
*   **验证方法**：在RoboTwin仿真平台（2D/3D）和Libero任务中进行对比。
*   **关键结果**：在单步推理下，成功率优于现有的迭代流匹配方法；真实世界中，将Pi 0.5的推理延迟降低了约45%。
*   **主要优势**：极高的推理效率、优秀的几何结构保持力、训练收敛速度快。
*   **主要局限**：对长周期（Long-horizon）任务缺乏显式的记忆机制；INN本身增加了少量的训练期计算负担。

### 6. 实用指南
*   **开源情况**：部分代码基于ManiFlow codebase实现，参考资料中提及了Starvla开源框架。
*   **实现细节**：
    *   **3D场景**：建议将初始噪声设为全零（All-zero initialization），可省略时间步嵌入，显著简化模型。
    *   **超参数**：$\alpha$ 的设置用于平衡重建质量与流匹配拟合效果，建议从0.1开始调节。
*   **迁移可能**：该适配器与具体的策略网络解耦，可直接作为“动作精修层”插入任何预测连续动作的策略头之后。

### 7. 总结
*   **核心思想**：利用可逆变换在动作空间构建紧凑流形，实现单步高保真生成。
*   **速记版pipeline**：
    1. 流匹配模型做初步预测；
    2. 用可逆网络将动作压缩至潜在空间；
    3. 对潜在向量进行规范化约束；
    4. 逆映射回原始空间获取精确动作。

**Key Findings:**

- Across simulation benchmarks, the proposed adapter consistently demonstrates superior or near state-of-the-art performance on a wide range of manipulation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.19194v1)
- [arXiv](https://arxiv.org/abs/2606.19194v1)

---

