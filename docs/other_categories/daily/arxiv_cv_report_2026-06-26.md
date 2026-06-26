time: 20260626

# Arxiv Computer Vision Papers - 2026-06-26

## Executive Summary

# 每日Arxiv计算机视觉论文执行摘要（2026-06-25）

## 一、主要主题与趋势

本期10篇论文高度聚焦于**机器人学习与多模态基础模型**的交叉领域，呈现以下核心趋势：
- **持续学习与自我进化**：多篇工作探索模型在交互中持续改进，如基于生成回放的模仿学习（#4）、变分动力学策略更新（#6）以及自奖励驱动的多模态模型进化（#3、#5）。
- **世界模型的可信度**：明确关注世界模型的幻觉问题（#8），并提出可预测、可预防方案，标志着该方向从“构建”转向“可靠使用”。
- **多模态感知与行动联合**：从视觉-语言-动作联合预训练（#10）、振动触觉到动作映射（#7），到自监督多模态感知（#9），感知模态持续扩展。
- **高效决策与规划**：贝叶斯优化与局部成本图结合（#2），以及大规模行为克隆的开放基准（#1），强调数据与计算效率。

## 二、特别重要或创新的论文

1. **《Hallucination in World Models is Predictable and Preventable》**（#8）  
   首次系统论证世界模型中的幻觉可被预测并主动防止，为安全机器人部署提供理论基石，具有极高实际价值。

2. **《Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation》**（#3）  
   提出自一致性奖励驱动的统一框架，实现多模态理解与生成的闭环自我进化，无需外部标注，创新性突出。

3. **《Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models》**（#5）  
   在自进化多模态模型中对视觉token进行注意力重加权，显著提升细粒度视觉理解，对视觉-语言模型精准度提升有启发。

4. **《VibeAct: Vibration to Actions for Contact-Rich Reactive Robot Dexterity》**（#7）  
   利用低成本振动传感器实现接触丰富的灵巧操作，开辟触觉-动作映射新范式，硬件门槛低，实用前景广。

5. **《LA4VLA: Learning to Act without Seeing via Language-Action Pretraining》**（#10）  
   探索纯语言-动作预训练实现无视觉条件下的行动，挑战传统视觉主导的机器人学习，为低光照或视觉受限场景提供新思路。

## 三、新兴研究方向与技术

- **自我进化的多模态模型**：结合自一致性奖励和注意力机制，无需人工反馈即可持续优化，是迈向自主智能体的重要步骤。
- **世界模型的幻觉检测与预防**：从被动发现转向主动防范，将因果性推理引入世界模型校准。
- **触觉感知的轻量化表征**：振动信号作为低维度、低成本触觉替代，有望普及到灵巧操作中。
- **语言-动作直接映射**：绕过视觉模态，探索语言作为行动的唯一感官输入，可能改变多模态融合范式。
- **持续学习中的生成式重放**：利用世界模型生成过往经验，避免灾难性遗忘，成为机器人持续策略学习的主流框架。

## 四、建议优先全文阅读的论

- **#8**（世界模型幻觉）—— 安全性关键，所有从事机器人规划的研究者必读。
- **#3**（自进化多模态理解与生成）—— 方法论新潮，对多模态大模型方向有重要参考。
- **#7**（振动到动作）—— 硬件友好，可能推动低成本灵巧操作普及。
- **#5**（视觉token注意力）—— 提升视觉-语言模型性能的具体技巧，实用性强。
- **#10**（语言-动作预训练）—— 颠覆性思路，值得关注其理论边界与实验验证。

---

## Table of Contents

1. [Scalable Behavior Cloning with Open Data, Training, and Evaluation](#2606.27375v1)
2. [BOWConnect: Parallel Bayesian Optimization over Windows with Learned Local Cost Maps for Sample-Efficient Kinodynamic Motion Planning](#2606.27292v1)
3. [Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards](#2606.27376v1)
4. [World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays](#2606.27374v1)
5. [Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models](#2606.27373v1)
6. [Continual Robot Policy Learning via Variational Neural Dynamics](#2606.27353v1)
7. [VibeAct: Vibration to Actions for Contact-Rich Reactive Robot Dexterity](#2606.27344v1)
8. [Hallucination in World Models is Predictable and Preventable](#2606.27326v1)
9. [OctoSense: Self-Supervised Learning for Multimodal Robot Perception](#2606.27317v1)
10. [LA4VLA: Learning to Act without Seeing via Language-Action Pretraining](#2606.27295v1)

---

## Papers

<a id='2606.27375v1'></a>
## [Scalable Behavior Cloning with Open Data, Training, and Evaluation](https://arxiv.org/abs/2606.27375v1)

**Authors:** Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, Angjoo Kanazawa

**Published:** 2026-06-25

**Categories:** cs.RO

**Abstract:**

We introduce ABC, a fully open-source stack for manipulation with behavior cloning. At its core is ABC-130K: the largest open-source teleoperation dataset to date, featuring 3,500 hours of data spanning over 130K episodes across 195 diverse tasks. Furthermore, we open-source our accessible hardware setup, training infrastructure, and simulation pipeline. We also release 400 hours of sim-teleop data and provide a co-training recipe that produces correlated simulation and real-world evaluation, offering a reliable proxy for ablating model-design and training decisions before costly real-world evaluation. We explore various training recipes and compare common architectural choices for Diffusion Transformers (DiT) and Vision-Language-Action (VLA) models, grounding our findings in real-world evaluations. The resulting policies successfully execute dexterous tasks such as box folding and extracting credit cards from wallets. By providing a reproducible toolkit, we aim to place researchers on an equal footing, establishing the necessary foundation to learn the ABCs of Behavior Cloning together as a community.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇题为《Scalable Behavior Cloning with Open Data, Training, and Evaluation》的论文分析如下：

### 1. 论文核心贡献总结
该论文推出了名为“ABC”的完整开源机器人操纵行为克隆（BC）技术栈，其核心贡献在于发布了目前最大的开源遥操作数据集 ABC-130K（包含3,500小时、130K个任务片段）。此外，该研究还提供了涵盖硬件方案、训练架构、仿真流水线及真实世界评估的完整生态，旨在通过标准化的开源工具链降低机器人学习的研究门槛。

### 2. 关键创新与方法论
*   **规模化数据驱动：** 通过提供超大规模且多样化的真实世界数据集，解决了目前机器人学习中“数据匮乏”这一核心瓶颈。
*   **仿真与现实的协同（Co-training）：** 创新性地构建了仿真数据与现实数据的联动训练配方，提供了一个可靠的“代理评估环境”。研究者可以在仿真中对模型架构（如Diffusion Transformers vs. VLA）进行消融实验，减少昂贵的真实世界试错成本。
*   **全栈开源理念：** 不同于以往仅发布模型权重的做法，该论文开源了从“数据获取硬件”到“训练基础设施”的闭环，这是推动具身智能研究标准化、可复现性的重要范式转移。

### 3. 对领域的潜在影响
*   **打破“黑盒”现状：** 社区目前充斥着各种闭源的具身智能研究，ABC 栈提供了一个基准（Baseline），使得不同实验室的研究成果可以在同一框架下进行公平的对比。
*   **加速具身智能落地：** 通过展示如“折叠纸盒”、“从钱包取卡”等精细操纵任务，证明了在高质量大规模数据集下，行为克隆仍具有极强的拓展潜力，这将引导领域重新评估简单 BC 算法结合大规模数据的上限。

### 4. 相关领域与受益方向
*   **机器人操纵（Robotic Manipulation）：** 直接受益，特别是涉及灵巧手（Dexterous Manipulation）和复杂接触任务的研究。
*   **大模型与具身智能（VLA/DiT）：** 视觉-语言-动作模型的研究者可以利用此数据集验证不同的 Transformer 架构设计。
*   **仿真到现实（Sim-to-Real）迁移：** 该研究提供的仿真数据与现实数据的对齐方案，将为弥合模拟器与真实环境差距提供新思路。
*   **自动驾驶与移动机器人：** 数据处理流水线和端到端学习策略的工程经验可迁移至更广泛的自主导航和动态交互领域。

### 5. 可推断的局限性
*   **泛化能力的瓶颈：** 尽管拥有 130K 规模的片段，但在“开放环境（Open-world）”中，行为克隆依然面临泛化挑战。对于未见过的物体、复杂纹理或动态环境，BC 策略的鲁棒性仍有待观察。
*   **硬件依赖性：** 论文虽然开源了硬件配置，但对于缺乏特定高精度机械臂（如模仿收集数据的遥操作设备）的实验室，复现过程仍具有相当高的硬件集成成本。
*   **长程任务挑战：** 行为克隆模型通常擅长短程操作，对于涉及复杂逻辑推理、多步骤长程任务（Long-horizon tasks）的序列规划，该方法可能需要结合层次化规划或大语言模型（LLM）作为高层决策，而非仅依赖 BC。

**总结建议：**
这篇论文的趣味性在于它不仅是在做“算法竞赛”，而是在做“基础设施建设”。对于计算机视觉研究者而言，这是一个极佳的、具有高可信度的数据集和实验环境，它将促使我们从单纯追求模型性能，转向思考如何构建更具可重现性和工程透明度的机器人视觉系统。

**Key Findings:**

- We introduce ABC, a fully open-source stack for manipulation with behavior cloning.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27375v1)
- [arXiv](https://arxiv.org/abs/2606.27375v1)

---

<a id='2606.27292v1'></a>
## [BOWConnect: Parallel Bayesian Optimization over Windows with Learned Local Cost Maps for Sample-Efficient Kinodynamic Motion Planning](https://arxiv.org/abs/2606.27292v1)

**Authors:** Sourav Raxit, Abdullah Al Redwan Newaz, Jose Fuentes, Leonardo Bobadilla

**Published:** 2026-06-25

**Categories:** cs.RO

**Abstract:**

This paper presents BOWConnect, a bidirectional parallel kinodynamic motion planner that addresses three fundamental limitations of existing sampling-based methods: sample inefficiency in high-dimensional state spaces, unreliable cost heuristics under dynamic constraints, and poor performance in narrow passage environments. Unlike classical planners that rely on random control sampling and geometric distance heuristics, BOWConnect integrates Bayesian Optimization over Windows (BOW) as a learning-based steering function within a parallel tree-based exploration framework, enabling each worker to learn local cost maps and constraints to guide sampling toward dynamically feasible and collision-free controls. A bidirectional architecture simultaneously grows forward and backward trees from the start and goal regions in parallel threads, with a spatial hashing mechanism enabling fast connection queries and a boundary value problem solver generating kinodynamically consistent bridge trajectories. Extensive evaluations across ten benchmark environments demonstrate that BOWConnect achieves 100\% success while delivering the fastest or near-fastest planning time in complex scenarios, including narrow passages and non-convex spaces where state-of-the-art planners fail or degrade substantially. Real-world deployment on a ground vehicle and a quadrotor confirms real-time planning with no collisions. Videos of real-world and simulated experiments, high-resolution versions of the figures, and the open-source code are available at https://bow-connect.github.io/.

**Analysis:**

### 1. 摘要翻译
本文提出了BOWConnect，一种双向并行运动规划器，旨在解决现有采样规划方法在处理高维状态空间、动态约束下成本启发式函数不可靠以及在窄道环境中性能不佳等根本性局限。不同于依赖随机控制采样和几何距离启发式的传统规划器，BOWConnect将“基于窗口的贝叶斯优化（BOW）”作为一种学习型的转向函数，集成在并行树搜索框架中，使每个工作线程能够学习局部代价图和约束，引导采样趋向于动力学可行且无碰撞的控制序列。该架构能够并行生长双向树，利用空间哈希机制实现快速连接查询，并通过边值问题求解器生成一致的桥接轨迹。在十个基准测试中的评估表明，BOWConnect在复杂场景（包括窄道和非凸空间）中实现了100%的规划成功率，且规划时间最优或接近最优。实车（地面车辆和四轮无人机）部署验证了其无碰撞的实时规划能力。

### 2. 方法动机分析
*   **驱动力**：解决现有运动规划器在高维动力学约束下，无法在“探索效率”与“可行性保证”之间取得平衡的问题。
*   **痛点**：传统方法（如RRT*）依赖随机采样和简单的几何启发式，在窄道或严苛的非完整约束下效率极低且极易失败；现有的基于优化（BO）的方法通常是单向的、局部贪婪的，缺乏全局连接能力。
*   **核心直觉**：通过并行化手段和基于学习的局部转向器，将局部轨迹生成与全局树生长解耦，利用双向扩展提高对狭窄空间的渗透力。

### 3. 方法设计详解
*   **流程Pipeline**：
    1.  **并行初始化**：在起始点和目标点周围采样多个状态点，作为多线程树生长的根节点。
    2.  **局部搜索（BOW插件）**：每个线程独立运行基于贝叶斯优化的局部规划器，通过高斯过程（GP）建模回报和约束，根据当前空间特征自适应调整采样方向。
    3.  **双向生长**：向前树和向后树在多核环境下并发运行，通过空间哈希表（Spatial Hashing）实时缓存节点分布。
    4.  **连接发现与BVP求解**：主线程持续监控空间哈希表，一旦检测到两树邻近节点，执行多阶段验证（约束校验与BVP求解），确保双向轨迹动力学一致地桥接。
*   **关键机制**：
    *   **空间哈希**：将连续状态映射到离散网格，将原本$O(N)$的邻近查询降至$O(1)$。
    *   **边界值问题（BVP）求解器**：利用比例控制策略，根据 heading error 计算 yaw rate，确保桥接轨迹在连接两树时满足严格的非完整约束。
    *   **概率可行性函数**：利用$\text{P}_{\text{feas}}$惩罚不可行区域，引导算法在复杂障碍物中自动“避障”。

### 4. 方法对比分析
*   **本质区别**：从传统的“随机盲目采样”转变为“基于GP引导的概率采样”，并将局部轨迹生成与全局规划高度并行化。
*   **创新贡献**：提出了一种结合BO局部搜索与双向并行RRT的通用架构；引入了基于空间哈希的实时双向连接机制，显著提升了窄道环境下的收敛速度。
*   **适用场景**：高维非线性机器人系统（如汽车、四旋翼），特别是在存在狭窄障碍物、需要高精度动力学可行性的场景。

### 5. 实验分析
*   **验证方法**：在6个地面车辆环境和4个四旋翼环境下，与RRT、SST、EST、KPIECE等主流开源规划器进行基准比对。
*   **关键结论**：在所有环境下均保持100%成功率，且在复杂环境（如Intel布局）中规划速度比主流基线快1-2个数量级。
*   **优劣势**：优势在于在满足严格动力学约束下的实时性与鲁棒性；局限在于依赖高斯过程预处理，需一定的计算开销，且性能受限于线程数及并行通信延迟。

### 6. 实用指南
*   **开源地址**：https://bow-connect.github.io/
*   **实现细节**：
    *   **超参数**：重点调节 $r_{\text{goal}}$（采样半径）和 $T$（时间视界）。
    *   **迁移**：该框架是模型不可知的，若要迁移至新机器人，仅需修改动力学模型 $f(x, u)$ 的积分器以及代价函数中定义的 collision 检查逻辑。

### 7. 总结
*   **核心思想**：通过并行贝叶斯优化转向器，实现复杂约束下的动力学运动规划。
*   **速记Pipeline**：
    1. 多点并行触发两端生长。
    2. 贝叶斯优化指导局部路径采样。
    3. 空间哈希实时比对树间距离。
    4. 边界值计算桥接两端轨迹。

**Key Findings:**

- Extensive evaluations across ten benchmark environments demonstrate that BOWConnect achieves 100\% success while delivering the fastest or near-fastest planning time in complex scenarios, including narrow passages and non-convex spaces where state-of-the-art planners fail or degrade substantially.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27292v1)
- [arXiv](https://arxiv.org/abs/2606.27292v1)

---

<a id='2606.27376v1'></a>
## [Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards](https://arxiv.org/abs/2606.27376v1)

**Authors:** Ritesh Thawkar, Shravan Venkatraman, Omkar Thawakar, Abdelrahman Shaker, Fahad Khan, Hisham Cholakkal, Salman Khan, Rao Muhammad Anwer

**Published:** 2026-06-25

**Categories:** cs.CV

**Abstract:**

Most unified large multimodal models (LMMs) that support both visual understanding and image generation still rely on curated post-training supervision, such as human annotations, preference labels, or external reward models. We ask whether a unified LMM can improve both abilities autonomously using only unlabeled images. We propose a self-evolving training framework with three internal roles: a Proposer that generates visual questions, a Solver that answers and evaluates them, and a Generator that synthesizes images. Training uses only self-derived consistency signals, without human annotations, preference labels, or task-trained external reward/judge models. To stabilize learning, we introduce Solver Token Entropy (STE), a continuous difficulty signal based on token-level prediction uncertainty that remains useful even when sample-level consistency becomes unreliable. For image generation, we design a multi-scale internal evaluation scheme that combines question-answer fidelity scoring with cycle-consistent captioning. This creates a solver-mediated coupling, where better visual understanding enables more reliable generation assessment and stronger internal training signals. The framework preserves the same role decomposition, reward logic, and training schedule across diffusion-based BLIP3o, rectified-flow BAGEL, and autoregressive VARGPT-v1.1 architectures, requiring only each backbone's native prompting and generation interface. Across eight understanding metrics, our method consistently improves over the corresponding base models. On BAGEL, it achieves a $+3.5\%$ absolute gain on MMMU and improves GenEval image generation performance from $82\%$ to $85\%$. Code and models are publicly released.

**Analysis:**

### 1. 摘要翻译
现有的大型统一多模态模型（LMMs）在联合执行视觉理解与图像生成任务时，往往严重依赖训练后的精细标注、偏好标签或外部奖励模型。本研究提出一种能够仅利用无标签图像自主改进这两种能力的自演进训练框架。该框架由三个相互协作的内部角色构建：生成视觉问题的“提案者（Proposer）”、回答并评估问题的“求解者（Solver）”以及合成图像的“生成者（Generator）”。通过引入“求解者令牌熵（STE）”作为持续的难度度量信号，解决样本级自一致性退化问题；通过结合QA保真度与循环一致性标注的多尺度内部评估方案，解决理解与生成之间的弱耦合。实验表明，该方法在不依赖外部监督的情况下，在三大主流模型架构上均实现了持续的性能提升。

### 2. 方法动机分析
*   **驱动力**：消除对大规模人工标注和外部预训练奖励模型的依赖，通过模型内部角色之间的“自我博弈（Self-play）”实现理解与生成的联合协同进化。
*   **现有方法痛点**：
    *   **奖励退化（Reward Degeneracy）**：简单的采样自一致性（Self-consistency）容易陷入低熵解，即使在模型内部信心不足时也会收敛，导致训练信号微弱。
    *   **弱耦合（Weak Coupling）**：视觉理解与图像生成通常被视为独立的任务分支，二者缺乏有效的任务间反馈回路。
*   **研究假设**：通过将预训练模型分解为可协作的Proposer-Solver-Generator角色，并利用模型自身评估能力构建反馈闭环，可以在无监督环境下实现认知能力与生成质量的共同提升。

### 3. 方法设计详解
*   **核心角色**：三个冻结主干（Backbone）上的轻量级LoRA适配器：
    *   **Proposer**：生成高质量、具有挑战性的视觉问题。
    *   **Solver**：回答问题，并作为理解评估器（通过一致性熵）和生成评估器（通过QA保真度）。
    *   **Generator**：根据文本指令合成图像。
*   **核心技术细节**：
    *   **STE（Solver Token Entropy）**：计算Solver生成答案时前5个令牌的softmax熵最大值。相比样本级一致性，STE能更精确地定位模型在回答难点（颜色、计数、否定词等）时的不确定性，提供持续的课程学习信号。
    *   **多尺度生成评估**：生成器产生的图像由Solver通过“QA保真度（原始图像QA答案 vs 生成图像QA答案）”和“循环一致性（图像反向生成描述 vs 原文本）”进行双重验证，将理解任务直接转化为生成任务的反馈信号。
    *   **耦合逻辑**：理解任务（Proposer-Solver）的优化提升了Solver的质量，使得生成任务的评估信号更加可靠，形成了非对称的协同进化。

### 4. 方法对比分析
*   **本质区别**：不同于常规的RLHF或基于外部标注的SFT，该方法完全基于“自我博弈”，将视觉理解的评估能力直接转移为生成任务的奖励反馈。
*   **创新贡献**：首次实现了在统一模型架构（扩散、整流流、自回归）上的通用无监督自演进框架，通过STE彻底解决了自监督训练中的奖励信号退化问题。
*   **适用场景**：适用于具备理解与生成能力（U+G）的统一多模态模型，尤其是当特定领域标注数据匮乏时。

### 5. 实验分析（精简版）
*   **验证方法**：在三大主干（BLIP3o、BAGEL、VARGPT-v1.1）上进行基线对比实验，涵盖七项理解评估基准和GenEval生成评估。
*   **关键结果**：在MMMU任务上取得了平均+3%以上的提升，GenEval分数普遍提升3个百分点。
*   **优势**：零标注开销，极强的架构通用性，在复杂组合任务（如多目标计数、空间定位）上效果显著。
*   **局限**：对Solver的基础智能有一定要求，若模型原始理解能力极弱，自评估可能会引入噪声。

### 6. 实用指南
*   **开源**：代码与模型已在GitHub和HuggingFace发布（详见论文P1）。
*   **迁移建议**：本方法核心依赖LoRA适配器。迁移至新架构时，只需实现对应的模型Prompt接口和LoRA映射模块，无需修改核心训练算法逻辑。
*   **关键超参数**：U:G训练周期设定为3:2；STE窗口大小取128最为稳定；学习率保持在1e-6量级以防止破坏预训练权重。

### 7. 总结
*   **核心思想**：通过内部角色博弈实现无监督的认知与生成双向自我演进。
*   **Pipeline速记**：
    1. 生成挑战性视觉问题；
    2. Solver多视角回答并计算STE难度；
    3. 利用STE和自一致性反馈训练Proposer/Solver；
    4. Solver评估生成的图像质量；
    5. 根据评价反馈迭代生成器。

**Key Findings:**

- We propose a self-evolving training framework with three internal roles: a Proposer that generates visual questions, a Solver that answers and evaluates them, and a Generator that synthesizes images.
- To stabilize learning, we introduce Solver Token Entropy (STE), a continuous difficulty signal based on token-level prediction uncertainty that remains useful even when sample-level consistency becomes unreliable.
- Across eight understanding metrics, our method consistently improves over the corresponding base models.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27376v1)
- [arXiv](https://arxiv.org/abs/2606.27376v1)

---

<a id='2606.27374v1'></a>
## [World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays](https://arxiv.org/abs/2606.27374v1)

**Authors:** Manish Kumar Govind, Dominick Reilly, Smit Patel, Hieu Le, Srijan Das

**Published:** 2026-06-25

**Categories:** cs.RO, cs.CV

**Abstract:**

Going beyond predicting robot actions, World Action Models (WAMs) can also generate future visual observations. We build on this generative capability to propose Recurrent Generative Replay (REGEN), a continual imitation learning framework that synthesizes pseudo-replay trajectories, enabling a robot policy to rehearse previously learned tasks without storing their original human demonstrations. During continual adaptation, REGEN recursively queries the WAM to synthesize pseudo-replay trajectories conditioned only on prior task instructions and current-task observations. Experiments in both simulation and real-world manipulation settings show that REGEN reduces catastrophic forgetting by up to $50\%$ relative to sequential fine-tuning, while approaching the performance of privileged experience replay methods that require access to real replay data. Finally, we analyze the factors limiting generated replay, identifying long-horizon visual degradation and action-observation inconsistency as the primary bottlenecks. Our results establish WAMs as a promising foundation for continual robot learning without stored demonstrations.

**Analysis:**

这是一份关于论文《World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays》的深度分析报告。

---

### 1. 摘要翻译
除了预测机器人动作外，世界动作模型（WAMs）还能生成未来的视觉观测。我们基于这一生成能力，提出了递归生成重放（REGEN），这是一种持续模仿学习框架，它能合成伪重放轨迹，使机器人策略在无需存储原始人类演示的情况下，就能重演先前学习过的任务。在持续适应过程中，REGEN 递归地查询 WAM，仅根据先前的任务指令和当前任务的观测结果来合成伪重放轨迹。在模拟和现实环境中的操纵实验表明，相对于顺序微调，REGEN 将灾难性遗忘减少了多达 50%，同时逼近了需要真实重放数据的特权经验重放方法的性能。最后，我们分析了限制生成重放的因素，确定了长程视觉退化和动作-观测不一致是主要瓶颈。我们的结果确立了 WAM 作为无需存储演示的持续机器人学习的有力基础。

### 2. 方法动机分析
- **驱动力**：机器人策略在不断学习新任务时，无法保留旧任务的知识（灾难性遗忘）。传统的经验重放（Experience Replay）依赖于存储真实的、标注过的历史数据，这在存储空间受限或隐私要求高的现代机器人基础模型中变得不可行。
- **现有痛点**：现有方法严重依赖存储原始演示数据。随着机器人基础模型向大规模、私有化数据集发展，获取这些原始数据进行回放已不现实。
- **研究假设**：如果一个模型（WAM）能够同时预测动作和视觉状态（即具备“世界模型”能力），那么它自身就可以作为一个生成器，通过“想象”过去任务的场景和动作来替代真实数据的重放。

### 3. 方法设计详解
- **流程总结（Pipeline）**：
  1. **初始化**：给定旧任务的语言指令和当前任务的初始真实观测。
  2. **循环生成**：利用 WAM 策略，在已知任务指令下，将上一步生成的观测（或初始真实观测）作为输入，预测下一个动作块和未来的视觉观测。
  3. **终止控制**：使用目标奖励头（Goal-reward head）监控任务完成度，当预测奖励超过阈值时停止生成。
  4. **混合训练**：将生成的“伪演示”与当前任务的真实演示聚合，共同更新模型权重。
- **关键细节**：为了克服递归生成的误差累积，作者提出了“初始化阶段”利用真实数据启动，随后进入“循环生成阶段”的机制。此外，目标奖励头不仅是任务指标，更是过滤掉无效伪重放数据的质量保障。

### 4. 方法对比分析
- **本质区别**：与传统经验重放（存储真实数据）或正则化方法（限制权重更新）不同，REGEN 是一种**基于生成的隐含记忆机制**。
- **创新贡献**：首次证明了 WAM 可以自我重放，无需任何外部数据集即可实现多任务持续学习。
- **适用场景**：适用于各类基于动作-观测预测的机器人策略，尤其是在缺乏存储空间或难以获取历史演示的场景。

### 5. 实验分析
- **关键结论**：在 LIBERO  benchmark 上，REGEN 的灾难性遗忘指标（NBT）较顺序微调降低了 50% 以上，且在实际机器人系统（xArm7）上成功率从 50% 提升至 80%。
- **局限性**：生成长程轨迹时，由于递归导致的视觉 artifacts（退化）和动作预测不一致（动作物理上无法实现），依然存在性能瓶颈。

### 6. 实用指南
- **开源情况**：官方提供了代码仓库链接 (https://manishgovind.github.io/REGEN/)。
- **实现细节**：
  - **超参数**：生成伪轨迹时，建议限制生成的最大长度（Tmax），并使用奖励头进行 early-stopping 以避免生成质量过低。
  - **迁移建议**：若要迁移到新任务，关键在于保证预训练的 WAM 对观测的鲁棒性，否则递归反馈会导致视觉图像迅速模糊，从而破坏重放效果。

### 7. 总结
- **核心思想**：利用世界动作模型的生成属性进行“自我模拟式”经验重放，实现无数据存储的知识保留。
- **速记版pipeline**：
  1. 使用旧任务指令作为生成器引导；
  2. 将当前环境作为上下文，递归预测未来观测和动作；
  3. 过滤生成结果，剔除不符合任务目标的轨迹；
  4. 将这些伪数据与新数据混合进行微调。

**Key Findings:**

- Finally, we analyze the factors limiting generated replay, identifying long-horizon visual degradation and action-observation inconsistency as the primary bottlenecks.
- Our results establish WAMs as a promising foundation for continual robot learning without stored demonstrations.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27374v1)
- [arXiv](https://arxiv.org/abs/2606.27374v1)

---

<a id='2606.27373v1'></a>
## [Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models](https://arxiv.org/abs/2606.27373v1)

**Authors:** Shravan Venkatraman, Ritesh Thawkar, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Salman Khan, Fahad Khan

**Published:** 2026-06-25

**Categories:** cs.CV

**Abstract:**

Recently, self-evolving large multimodal models (LMMs) have received attention for improving visual reasoning in a purely unsupervised setting. However, multi-role self-play and self-consistency reward schemes in existing self-evolving LMMs optimize answer agreement without ensuring the decoder attends to visual content, relying instead on statistical language priors to produce self consistent outputs. This leads to a persistent failure mode we term visual under-conditioning, where the decoder relies on language priors rather than the image during generation, manifesting as insufficient attention to visual tokens. As a result, current self-evolving LMMs struggle on vision--language understanding tasks such as image captioning and visual question answering. To address this, we propose VISE (Visual Invariance Self-Evolution), a purely unsupervised self-evolving framework that directly regularizes the model's visual conditioning policy through two complementary invariance-based rewards: a geometric invariance reward that enforces spatial consistency under known transformations, and a semantic invariance reward that penalizes evidence-agnostic generation by requiring the model to recognize the absence of evidence when predicted regions are perturbed. VISE operates within a single model without specialist roles, external reward models, or annotations, and is trained on raw unlabeled images. Experiments on 18 benchmarks demonstrate the efficacy of our approach. Using Qwen3-VL-2B as the base model, VISE achieves gains of $+16.85$ CIDEr on COCO and $+19.66$ CIDEr on TextCaps, reduces object hallucination by $5.0$ Chair-I points, and generalizes across four model families and scales. Our code and models are available at https://mbzuai-oryx.github.io/VISE

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文《Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models》的分析如下：

### 1. 论文核心贡献总结
该论文揭示了现有自演进多模态大模型（LMMs）普遍存在的“视觉欠条件化”（Visual Under-conditioning）问题，即模型在缺乏监督的情况下倾向于依赖语言统计先验而非视觉特征。为此，作者提出了 **VISE**（Visual Invariance Self-Evolution）框架，通过引入几何和语义不变性约束，显著提升了模型在生成过程中对视觉Token的依赖度，从而在无标注数据下实现了性能的实质性提升。

### 2. 关键创新点与方法论
VISE 的核心在于**无需外部标注或辅助模型**的自监督正则化机制：
*   **几何不变性奖励（Geometric Invariance Reward）**：强制要求模型在对输入图像进行空间几何变换（如缩放、平移等）时，保持输出的一致性，从而迫使模型关注图像的空间结构。
*   **语义不变性奖励（Semantic Invariance Reward）**：通过扰动预测区域，要求模型识别出“证据缺失”的情况。如果模型在关键视觉信息被扰动时仍能做出自信预测，则会受到惩罚。这种机制有效遏制了模型脱离图像内容进行“幻觉式”生成的行为。
*   **单一模型架构**：该方法不依赖专家模型或复杂的奖励模型（Reward Models），仅通过原始无标签图像进行训练，体现了极高的计算效率和部署灵活性。

### 3. 潜在影响
*   **解决幻觉问题**：通过从机理上约束模型对视觉Token的关注，该研究为缓解大模型常见的“视觉幻觉”（Object Hallucination）提供了新的无监督路径。
*   **推动无监督学习**：VISE 证明了仅通过自演进机制（Self-Evolution）就能极大提升模型对视觉的理解能力，这对于数据匮乏或难以获取标注的特定垂直领域（如医疗影像、遥感）具有重要意义。
*   **模型评估范式**：该工作提出的“视觉欠条件化”评估视角，可能会引发行业内对LMMs训练目标的重新审视。

### 4. 相关应用领域
*   **医疗影像分析**：在标注成本极高且模型必须严谨依赖病灶图像的医疗场景中，VISE 的机制可以显著提高模型判读的准确性和可靠性。
*   **自动驾驶与机器人**：在动态环境感知中，模型需要实时且准确地处理视觉信息，防止因语言先验干扰而产生的误判。
*   **跨模态检索与生成**：在需要精细描述的领域（如电商智能生成、自动辅助描述），该方法能有效提升图文对齐的精度。

### 5. 可推断的局限性
*   **计算开销与训练复杂性**：尽管是在单一模型内运行，但额外的几何与语义不变性约束增加了训练过程中的前向/反向传播逻辑，可能增加每一步训练的时间成本。
*   **对复杂场景的鲁棒性**：虽然在 18 个基准测试中表现优异，但在极其复杂、噪声极大或语义模糊的图像（如艺术创作、极其抽象的图形）中，基于不变性的约束是否会产生过度的惩罚，导致模型变得过于保守，仍有待观察。
*   **对基座模型的依赖**：虽然文中提到其在四种模型家族上具有通用性，但对于极小参数量或编码能力较弱的视觉编码器（Visual Encoder），VISE 的效果是否会因视觉特征提取本身的瓶颈而受限，是一个值得后续探究的问题。

**总结：**
这篇论文的价值在于它抓住了“多模态模型如何真正理解图像”这一本质问题。它不仅是性能的提升，更是一种**将训练重心从“优化语言生成概率”转向“优化视觉-语言关联强度”的思维转变**，对于构建更具可靠性的多模态智能体具有重要启发。

**Key Findings:**

- To address this, we propose VISE (Visual Invariance Self-Evolution), a purely unsupervised self-evolving framework that directly regularizes the model's visual conditioning policy through two complementary invariance-based rewards: a geometric invariance reward that enforces spatial consistency under known transformations, and a semantic invariance reward that penalizes evidence-agnostic generation by requiring the model to recognize the absence of evidence when predicted regions are perturbed.
- Experiments on 18 benchmarks demonstrate the efficacy of our approach.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27373v1)
- [arXiv](https://arxiv.org/abs/2606.27373v1)

---

<a id='2606.27353v1'></a>
## [Continual Robot Policy Learning via Variational Neural Dynamics](https://arxiv.org/abs/2606.27353v1)

**Authors:** Jiaxu Xing, Zhiyuan Zhu, Yunfan Ren, Ismail Geles, Yifan Zhai, Rudolf Reiter, Davide Scaramuzza

**Published:** 2026-06-25

**Categories:** cs.RO

**Abstract:**

Robots deployed in the real world rarely operate under a single fixed dynamics model: wind changes, payloads vary, batteries drain, contacts shift, and hardware wears. Yet most learning-based controllers are trained once and deployed as if learning were complete. This prevents the robot from using deployment experience to further improve task performance. In this work, we propose a continual learning framework that uses real-world experience to improve robot policies under hidden and recurring dynamics. Our method learns a condition-aware dynamics model from real state-action trajectories by combining an analytical physics prior with a neural residual for unmodeled effects. A recurrent encoder infers the current hidden condition from recent interaction, and this estimate conditions both the residual model and the policy. Policy learning is performed via differentiable simulation using diverse learned dynamics sampled from the latent model. At deployment, these sampled conditions are replaced by conditions inferred online from recent real interaction, allowing the policy to recover recurring dynamics by recognition rather than residual re-fitting. Through extensive simulation studies and real-world experiments, we demonstrate that the framework improves policy performance under diverse unobserved disturbances. On real quadrotor trajectory tracking under changing wind, the policy recovers from recurring disturbances in roughly 1s, about 5x faster than online residual re-fitting. It also reduces large-disturbance hover and tracking errors by 65.7% and 53.3% over the state-of-the-art online adaptation approaches

**Analysis:**

### 1. 摘要翻译
现实世界的机器人常面临风力变化、负载变化、硬件磨损等非固定动态环境，但现有策略多采用“训练一次、永久部署”的范式，限制了机器人的持续进化能力。本文提出了一种连续学习框架，利用真实世界的交互经验在隐藏且循环往复的动态环境下改进策略。我们的方法通过组合分析物理先验与用于建模未建模效应的神经残差，学习一个条件感知的动力学模型。循环编码器从最近的交互历史中推断当前的隐藏状态，以此作为残差模型和策略的输入。在部署时，通过在线推理最近的交互历史来识别隐藏动态，从而实现对循环条件的快速恢复，而非传统的残差重新拟合。实验表明，该方法在真实四旋翼轨迹跟踪中，能在约1秒内从循环扰动中恢复，速度比在线残差微调快约5倍，且错误率显著降低。

---

### 2. 方法动机分析
*   **驱动力**：旨在克服机器人部署后动态环境（非平稳性）导致的性能衰退，赋予机器人“在部署中学习”的能力，使其能识别并利用已学过的经验。
*   **现有痛点**：
    *   **领域随机化**：训练一个保守的通用策略，性能折中，无法针对特定条件优化。
    *   **在线残差拟合**：需要重新计算，响应慢（如5秒），且属于“失忆性”适应，当环境再次切换时需要重新训练。
*   **研究假设**：通过变分动力学建模，将复杂的环境动态映射到一个结构化的隐空间中，使机器人能够通过简单的历史窗口“识别”环境，并利用该识别结果在策略中进行条件化调整。

---

### 3. 方法设计详解
*   **流程与模型结构**：
    1.  **残差动力学建模**：结合已知的物理模型（Analytical Physics Prior）和学习残差模型（Neural Residual）。模型公式为：$\hat{s}_{t+1} = f_{prior}(s_t, a_t) + D_\psi(s_t, a_t, z_t)$。
    2.  **隐空间推理**：利用GRU循环编码器 $E_\phi$ 提取上下文窗口 $H_{t-C:t}$ 中的信息，生成低维隐变量 $z_t$，表征当前隐藏动态。
    3.  **变分对齐**：使用最大均值差异 (MMD) 损失函数，将编码器输出分布 $q_\phi(z)$ 与标准正态分布 $p(z)$ 对齐，确保隐空间不仅可推断，且可采样。
    4.  **可微分策略训练**：将物理先验和训练好的残差模型整合进可微分模拟器。在训练时，从先验分布 $p(z)$ 采样 $z$，使策略学会适应多样的动态条件。
*   **关键公式解释**：
    *   $L_{vnd} = L_{dyn} + \lambda_{rec}L_{rec} + \lambda_{mmd}MMD(q_\phi, p)$：该损失函数同时保证了动力学的预测精度（$L_{dyn}$）、轨迹上下文的保留（$L_{rec}$）以及隐空间的正则化（$MMD$）。

---

### 4. 方法对比分析
*   **本质区别**：与现有方法不同，本方法将“识别”与“适应”分离。策略通过学习识别隐空间的分布来“理解”环境，一旦环境发生变化，仅需通过编码器推理出对应的隐变量，即可切换策略响应，无需耗时的模型微调。
*   **创新点**：
    1.  **结构化隐空间**：通过MMD实现了从先验分布采样，打破了隐变量仅能在在线推断中使用的限制，实现了有效的“变分动力学”模拟。
    2.  **物理先验结合**：在保留物理可解释性的同时，通过残差网络捕捉未知动力学。
*   **适用场景**：非平稳、循环出现的动态环境（如风向、载重、摩擦力变化）。

---

### 5. 实验分析（精简版）
*   **关键结论**：在四旋翼跟踪任务中，恢复时间从5秒压缩至1秒；大扰动下 Hover 错误率降低了65.7%。
*   **优势**：快速的闭环适应、具备跨部署的一致性、无需 privileged（特权）信息。
*   **局限**：对视觉/状态估计器的噪声敏感，若传感器数据不准，会误导动力学模型的学习。

---

### 6. 实用指南
*   **实现要点**：
    *   **MMD调度**：需要 annealed schedule（退火调度）来稳定隐空间的学习，防止过早崩溃。
    *   **数据预处理**：注意动作序列的归一化，使用相同的统计量维持训练与部署的一致性。
*   **迁移建议**：该框架高度模块化，只要任务满足马尔可夫决策过程 (MDP)，且存在可观测的历史上下文（轨迹窗），即可复用其“编码器+物理先验残差+可微分训练”的 pipeline。

---

### 7. 总结
*   **核心思想**：通过学习隐环境表征，实现对循环动态的快速识别与响应。
*   **速记版pipeline**：
    1.  采集交互数据缓存；
    2.  训练变分残差模型；
    3.  隐空间对齐（MMD）；
    4.  可微分模拟中采样隐变量训练策略；
    5.  部署时实时编码推理环境并适配策略。

**Key Findings:**

- In this work, we propose a continual learning framework that uses real-world experience to improve robot policies under hidden and recurring dynamics.
- Our method learns a condition-aware dynamics model from real state-action trajectories by combining an analytical physics prior with a neural residual for unmodeled effects.
- Through extensive simulation studies and real-world experiments, we demonstrate that the framework improves policy performance under diverse unobserved disturbances.
- It also reduces large-disturbance hover and tracking errors by 65.7% and 53.3% over the state-of-the-art online adaptation approaches

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27353v1)
- [arXiv](https://arxiv.org/abs/2606.27353v1)

---

<a id='2606.27344v1'></a>
## [VibeAct: Vibration to Actions for Contact-Rich Reactive Robot Dexterity](https://arxiv.org/abs/2606.27344v1)

**Authors:** Yuemin Mao, Uksang Yoo, Jean Oh, Jonathan Francis, Jeffrey Ichnowski

**Published:** 2026-06-25

**Categories:** cs.RO

**Abstract:**

Dexterous manipulation depends on contact events that are fast, local, and often visually occluded. Piezoelectric microphones offer a compact and high-bandwidth way to sense these interactions, but the resulting vibro-acoustic signals are difficult to simulate faithfully enough for end-to-end sim-to-real policy learning on dexterous robot hands. We propose VibeAct, a framework that bridges real vibrotactile sensing and simulation-based reinforcement learning through a shared physical representation of contact and slip. In the real world, we embed piezoelectric microphones into a dexterous robot hand and collect vibro-acoustic data through teleoperation, then replay the recordings in a calibrated digital clone to automatically label per-finger contact and slip. A tactile estimator learns to predict contact and slip from real microphone waveforms, while manipulation policies are trained in simulation on the same representation computed directly from simulated contacts. This decoupling lets policies exploit rapid tactile feedback without simulating raw audio. Across five contact-rich tasks spanning regrasping, in-hand reorientation, and insertion, VibeAct consistently outperforms a proprioception-and-point-cloud baseline in simulation, with the largest gains on tasks requiring sustained reactive control, where the continuous slip-magnitude channel proves the most informative observation. The learned policies transfer to a physical dexterous hand-arm platform, improving success rates on deployed tasks. Project videos and additional details are at https://vibeact.github.io/.

**Analysis:**

### 1. 摘要翻译
灵巧操作依赖于快速、局部且常被视觉遮挡的接触事件。压电麦克风为感知这些交互提供了一种紧凑且高带宽的途径，但产生的振动声学信号难以准确仿真，无法直接用于灵巧机器手从仿真到现实（sim-to-real）的策略学习。我们提出了 VIBEACT 框架，通过共享的接触和滑动物理表示，架起了真实振动触觉感知与基于仿真的强化学习之间的桥梁。在现实世界中，我们将压电麦克风嵌入灵巧手，通过遥操作收集振动数据，并在校准的数字克隆体中重放这些记录，以自动标记各手指的接触和滑动情况。触觉估计器学习从麦克风波形中预测接触和滑动，而操纵策略则在仿真中基于从接触动力学直接计算的相同表示进行训练。这种解耦使得策略可以在无需模拟原始音频的情况下利用快速触觉反馈。在涵盖重抓取、手内重定向和插入的五项接触丰富任务中，VIBEACT 在仿真中始终优于仅依赖本体感受和点云的基线，在需要持续反应式控制的任务中增益最为显著，其中连续滑动幅度通道提供了最丰富的信息。学习到的策略成功迁移到物理灵巧手-臂平台，提高了部署任务的成功率。

### 2. 方法动机分析
*   **驱动力**：解决灵巧操作中接触事件难以通过视觉捕捉的难题，利用压电麦克风的高频振动信号获取精细触觉反馈。
*   **痛点**：振动声学信号受材质、安装位置、环境噪声等多种复杂因素影响，在仿真中难以精确模拟（Sim-to-Real Gap），导致端到端学习策略无法直接迁移。
*   **核心直觉**：不直接将原始音频喂给策略，而是定义一个**物理驱动的中间表示（接触 onset、二值滑动、标量滑动幅度）**，将“感知问题”（预测表示）与“控制问题”（基于表示学习策略）解耦。

### 3. 方法设计详解
*   **pipeline**：
    1.  **数据采集与数字克隆**：通过遥操作采集机器人动作与麦克风波形。在 MuJoCo 模拟器中重放轨迹，由物理引擎通过接触求解器自动生成接触/滑动标注。
    2.  **触觉估计器（感知）**：每个手指配置独立的子网络。输入为麦克风的 log-mel 频谱图，通过卷积编码器提取特征，经预测头输出接触 onset（概率）、滑动发生（概率）和滑动幅度（标量）。
    3.  **策略学习（控制）**：策略网络接收点云、本体感受和上述预测的触觉表示。在仿真中直接使用物理引擎计算的触觉标注进行训练，实现无缝迁移。
*   **关键公式**：$z_t = \{b_t^i, m_t^i, e_t^i\}$。其中 $b_t^i$ 为滑动检测，$m_t^i$ 为滑动强度，$e_t^i$ 为接触爆发点。这种定义不仅反映了触觉特性，还保证了在仿真中的可计算性。

### 4. 方法对比分析
*   **本质区别**：传统方法要么依赖昂贵的触觉传感器，要么强行在仿真中模拟音频。VIBEACT 创造性地通过“自监督标注 + 物理中间层”绕过了音频模拟的复杂性。
*   **创新贡献**：提出了一种基于数字克隆的自动触觉标注管道，使低成本压电麦克风能成为鲁棒的灵巧操作感知模态。
*   **适用场景**：高接触频率、视觉遮挡严重的复杂操作任务。

### 5. 实验分析
*   **关键结论**：在所有测试任务（如攀爬、旋转、插孔）中，引入滑动幅度通道带来的性能增益最为关键。
*   **主要优势**：不仅大幅提升了仿真成功率，且在真实物理平台上的迁移效果显著，优于单纯的点云+本体感受基线。
*   **主要局限**：感知器受限于特定的硬件结构，若麦克风更换位置或安装材质改变，需重新校准；对物体的姿态估计精度依赖较高。

### 6. 实用指南
*   **开源**：项目官网 `vibeact.github.io`。
*   **训练细节**：预训练（固定物体数据集）+微调（移动物体数据集）策略至关重要，能有效填补领域间隙。
*   **迁移迁移**：核心在于将物理动力学的关键特征（如滑动状态）抽象化。若迁移至其他手，需确保新手指安装点的振动传递函数在合理范围内，且需重新在仿真中跑一遍数字克隆生成标注。

### 7. 总结
*   **核心思想**：通过触觉中间表示架起振动感知与仿真控制的桥梁。
*   **速记版pipeline**：
    1.  采集真实音频与轨迹。
    2.  仿真器重放生成自动标注。
    3.  训练触觉模型预测接触/滑动。
    4.  仿真中训练策略并部署到现实。

**Key Findings:**

- We propose VibeAct, a framework that bridges real vibrotactile sensing and simulation-based reinforcement learning through a shared physical representation of contact and slip.
- Across five contact-rich tasks spanning regrasping, in-hand reorientation, and insertion, VibeAct consistently outperforms a proprioception-and-point-cloud baseline in simulation, with the largest gains on tasks requiring sustained reactive control, where the continuous slip-magnitude channel proves the most informative observation.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27344v1)
- [arXiv](https://arxiv.org/abs/2606.27344v1)

---

<a id='2606.27326v1'></a>
## [Hallucination in World Models is Predictable and Preventable](https://arxiv.org/abs/2606.27326v1)

**Authors:** Nicklas Hansen, Xiaolong Wang

**Published:** 2026-06-25

**Categories:** cs.LG, cs.CV, cs.RO

**Abstract:**

Modern generative world models render increasingly realistic action-controllable futures, yet they frequently hallucinate: rollouts remain visually fluent while drifting from the ground-truth dynamics. We hypothesize that hallucination concentrates in low-coverage regions of the state-action space, where lightweight data-centric signals can both detect it and guide mitigation. To test this, we introduce MMBench2, a 427-hour, 210-task dataset for visual world modeling with ground-truth actions, rewards, and live simulators, and train a 350M-parameter world model on it. We identify three distinct hallucination modes: perceptual, action-marginalized, and scene-diverging -- each anchored to a different stage of the pipeline, and develop three signals that accurately predict where the model will fail. To close coverage gaps at training time, we develop a coverage-aware sampling technique; to close them online, our hallucination predictors serve as curiosity rewards for targeted data collection, yielding a data-efficient finetuning recipe that adapts the pretrained world model to entirely unseen environments with as few as 50 real environment trajectories. Overall, our findings reveal that hallucination in world models is inherently a data coverage issue, and that the same signals used to detect it can also be used for mitigation.   An interactive web version of our paper is available at https://www.nicklashansen.com/mmbench2

**Analysis:**

以下是对论文《Hallucination in World Models is Predictable and Preventable》的深度分析：

### 1. 摘要翻译
现代生成式世界模型能渲染出逼真的可控未来，但常出现“幻觉”：在保持视觉流畅的同时偏离真实动态。我们假设幻觉集中在状态-动作空间的低覆盖区域，通过轻量级数据中心信号可对其进行检测与治理。为此，我们引入了 MMBench2（427小时、210项任务的视觉世界建模数据集），并训练了一个350M参数的世界模型。我们识别了三种独特的幻觉模式（感知幻觉、动作边际化幻觉、场景分叉幻觉），并开发了三种对应的预测器来捕捉模型失效点。通过覆盖感知采样技术和基于预测器的好奇心导向数据收集，我们实现了高效的微调策略，仅需50条真实轨迹即可将预训练模型适应至全新的环境。研究表明，世界模型中的幻觉本质上是数据覆盖问题。

### 2. 方法动机分析
- **核心动机**：将世界模型的“幻觉”问题从单纯的架构缺陷重新定义为“数据覆盖率不足”的问题。
- **现有痛点**：现有研究多将其视为模型容量不够，试图通过扩大规模（更大backbone）解决，但忽略了模型在未见过的状态-动作空间中缺乏经验这一本质原因。
- **研究假设**：幻觉是可预测的（通过运行时信号）且可预防的（通过调整训练数据分布），无需重构模型架构。

### 3. 方法设计详解
- **pipeline核心逻辑**：
    1. **检测与建模**：定义了三种幻觉并开发对应的轻量级预测信号（无需额外标注或训练）。
    2. **缓解策略一（训练时）**：**覆盖感知采样（Coverage-aware training）**。重新平衡数据集，使采样在各任务间均匀分布，而非基于帧。
    3. **缓解策略二（在线时）**：**目标导向数据收集**。利用幻觉预测器作为好奇心奖励（Curiosity reward），引导模型在交互中主动探索容易导致幻觉的区域，收集新数据进行微调。
- **关键信号设计**：
    - **Tokenizer round-trip residual ($u_r$)**：检测感知幻觉。预测潜空间状态经过“解码-再编码”后的残差。
    - **Flow instability ($u_f$)**：检测动作边际化。度量动力学头在Euler积分过程中对动作输入的敏感度和稳定性。
    - **Inter-seed variance ($u_s$)**：检测场景分叉。度量不同噪声种子下多步预测的方差。

### 4. 方法对比分析
- **根本不同点**：从“架构驱动”转向“数据驱动”。传统方法通过加深网络解决，本文通过“检测->反馈->数据补全”的闭环，解决训练数据分布长尾带来的泛化问题。
- **创新点**：识别并分类了三种导致幻觉的具体阶段；提出了一套无需标签的运行时幻觉预测指标；证明了同一套信号既能发现幻觉，又能引导高效的数据增补。

### 5. 实验分析（精简版）
- **验证**：利用MMBench2在200个预训练任务和10个未见过的转移任务上进行测试。
- **关键结论**：三种预测器与rollout误差呈强负相关（$\rho \approx 0.80$）；采用本文提出的策略，仅用50条轨迹即可在未见环境获得接近专家水平的性能。
- **优劣势**：优势在于无需改变模型架构即可大幅提升鲁棒性；局限在于依然依赖大规模预训练基础，且计算成本随数据多样性增加。

### 6. 实用指南
- **开源情况**：已开源完整数据集、代码、预训练检查点及交互浏览器界面（[nicklashansen.com/mmbench2](https://nicklashansen.com/mmbench2)）。
- **迁移性**：该方法逻辑具有极强的通用性，其“覆盖感知采样”和“基于预测器的好奇心采集”可以轻松迁移到任何基于Latent Dynamics World Model的架构（如Dreamer系列）。
- **实现细节**：关键在于对场景动态性（dynamism）的归一化处理，即 $u^{norm} = u/m$，否则高动态场景会产生误报。

### 7. 总结
- **核心思想**：世界模型幻觉源于数据覆盖缺失，通过特征信号可预测并可自我修正。
- **速记版pipeline**：
  1. **构建多样化任务数据集**；
  2. **训练时调整任务分布，拉平覆盖率**；
  3. **运行时计算预测信号，实时监控幻觉**；
  4. **利用高幻觉信号区域，自动触发数据回采**。

**Key Findings:**

- To test this, we introduce MMBench2, a 427-hour, 210-task dataset for visual world modeling with ground-truth actions, rewards, and live simulators, and train a 350M-parameter world model on it.
- To close coverage gaps at training time, we develop a coverage-aware sampling technique; to close them online, our hallucination predictors serve as curiosity rewards for targeted data collection, yielding a data-efficient finetuning recipe that adapts the pretrained world model to entirely unseen environments with as few as 50 real environment trajectories.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27326v1)
- [arXiv](https://arxiv.org/abs/2606.27326v1)

---

<a id='2606.27317v1'></a>
## [OctoSense: Self-Supervised Learning for Multimodal Robot Perception](https://arxiv.org/abs/2606.27317v1)

**Authors:** Anthony Bisulco, Jeremy Wang, Kostas Daniilidis, Randall Balestriero, Pratik Chaudhari

**Published:** 2026-06-25

**Categories:** cs.CV, cs.RO

**Abstract:**

We present OctoSense, an open-source sensor platform with stereo RGB and event cameras, LiDAR, a thermal camera, an inertial measurement unit, RTK-corrected global positioning system, and proprioception (CAN bus data from a car, and joint angles for a quadruped robot). The eponymous OctoSense dataset contains 59 hours of time-synchronized driving data across different types of environments at different times of the day, including situations with highly degraded sensors. We demonstrate multi-modal self-supervised learning using such real-world robotics data, where sensors have different representations, frequencies, latencies and noise. Our approach, a "late-fusion" masked autoencoder, (i) uses modality-specific tokenizers to account for different spatiotemporal characteristics of these sensors, and (ii) caches modality-specific tokens at inference time to process new measurements as they come. This architecture (i) is fast (6.68 ms and 112 ms on NVIDIA 5090 and Orin NX respectively, to compute the representation), (ii) performs better than existing image-only foundation models on tasks such as estimation of optical flow, depth, semantic segmentation, and ego-motion (translation, rotation, and steering angle), and (iii) predicts robustly at nighttime or in situations where sensory data is degraded. See our project page for links to the dataset, code, and supplementary videos: https://abisulco.com/octosense/.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **OctoSense** 的论文分析如下：

### 1. 主要贡献总结
OctoSense 提出并开源了一个包含多种传感器（RGB、事件相机、LiDAR、热成像等）的高度时间同步机器人感知数据集，旨在解决真实世界复杂环境下多模态感知的难题。该研究通过一种新型的“后期融合掩码自动编码器”（Late-fusion Masked Autoencoder）架构，有效实现了传感器在不同采样频率、延迟和噪声环境下的自监督学习，并在自动驾驶及机器人状态估计任务中表现出优于单一模态基础模型的鲁棒性。

### 2. 核心创新与方法论
该工作的核心技术创新在于其**多模态处理的架构设计**：
*   **模态特异性 Tokenizer：** 针对不同传感器（如高频的事件相机与低频的热成像）的时空特性，设计了专门的嵌入与编码方式，解决了多模态数据输入空间不一致的问题。
*   **缓存机制与推理效率：** 通过在推理阶段缓存特定模态的 Token，系统能够实现高效的增量更新（在 5090 GPU 上仅需 6.68ms），这使得该模型能够处理流式传感器数据，极大地提升了在边缘计算设备（如 Orin NX）上的实时性能。
*   **自监督学习范式：** 利用掩码自动编码器（MAE）在海量非标注的多模态数据上进行预训练，使得模型学习到了跨模态的互补信息，从而在极端环境（如夜间、传感器损坏）下保持稳健。

### 3. 对该领域的潜在影响
*   **打破“视觉中心”局限：** 当前视觉基础模型大多依赖 RGB 图像，OctoSense 证明了通过融合 LiDAR、事件相机和热成像，可以显著提升视觉任务（深度、光流、分割）在复杂边缘场景下的表现。
*   **推动机器人“基础模型”研究：** 该论文为机器人领域提供了一种类似 NLP 中“大规模预训练+微调”的范式，有助于构建通用的具身智能感知底座。
*   **弥合学术界与工业界差距：** 该数据集覆盖了 59 小时的真实场景，特别是对“退化传感器”数据的关注，切中了工业界在自动驾驶部署中的痛点。

### 4. 受益的相关领域或应用
*   **自动驾驶：** 提高在雨雪、夜间、逆光等视觉传感器易失效环境下的感知安全性。
*   **四足机器人（Quadruped Robots）：** 由于文中特别提到包含 CAN 总线与关节角度数据，该模型非常适合用于复杂地形下的姿态稳定与路径规划。
*   **边缘计算与机器人控制：** 其低延迟、高吞吐的推理架构，直接推动了实时感知系统的开发。
*   **多模态融合感知研究：** 为后续研究如何处理异步、高噪声传感器的对齐问题提供了标准参考基准。

### 5. 可推断的潜在局限性
*   **泛化能力的边界：** 虽然论文强调了多环境，但 59 小时的数据集对于覆盖“长尾效应”（Rare events）可能仍然不足，在极其罕见的交通异常场景下模型表现有待验证。
*   **训练成本与复杂性：** 尽管推理速度快，但多模态融合模型的训练复杂度显著高于单一模态模型，且需要极为精确的时间同步硬件支持，这增加了复现和部署的门槛。
*   **动态场景的时序依赖：** 虽然采用了缓存机制，但在处理高度动态或快速运动场景时，模型在时域特征的融合上是否会因延迟补偿机制产生细微的偏差，仍需观察。

**专家总结：**
OctoSense 的趣味性在于它不仅是一个数据集，更是一套处理**感知不确定性**的工程实践。它成功地将“多模态”这一学术界常讨论的命题，通过高效的架构设计落地到了机器人实时推理中。这对于推动具身智能（Embodied AI）从实验室走向复杂真实世界具有重要的参考价值。

**Key Findings:**

- We present OctoSense, an open-source sensor platform with stereo RGB and event cameras, LiDAR, a thermal camera, an inertial measurement unit, RTK-corrected global positioning system, and proprioception (CAN bus data from a car, and joint angles for a quadruped robot).
- We demonstrate multi-modal self-supervised learning using such real-world robotics data, where sensors have different representations, frequencies, latencies and noise.
- Our approach, a "late-fusion" masked autoencoder, (i) uses modality-specific tokenizers to account for different spatiotemporal characteristics of these sensors, and (ii) caches modality-specific tokens at inference time to process new measurements as they come.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27317v1)
- [arXiv](https://arxiv.org/abs/2606.27317v1)

---

<a id='2606.27295v1'></a>
## [LA4VLA: Learning to Act without Seeing via Language-Action Pretraining](https://arxiv.org/abs/2606.27295v1)

**Authors:** Tao Lin, Yuxin Du, Yiran Mao, Zewei Ye, Yilei Zhong, Bing Cheng, Yiming Wang, Jiting Liu, Yang Tian, Junchi Yan, Feiran Wu, Zenan Meng, Hu Wei, Yuqian Fu, Gen Li, Bo Zhao

**Published:** 2026-06-25

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models are commonly pretrained on robot demonstrations by jointly mapping visual observations and language instructions to actions. However, dense visual-action supervision can dominate the comparatively sparse language-action signal. As a result, policies may rely on visual shortcuts rather than learn how language conditions action execution, making them sensitive to visual variations. To address this limitation, we propose LA4VLA, a language-action pretraining framework that enables policies to acquire language-conditioned action priors without visual observations. These priors capture reusable manipulation skills shared across tasks and scenes, reducing reliance on scene-specific visual cues. Specifically, LA4VLA decomposes expert demonstration trajectories into atomic action segments and pairs each segment with a corresponding low-level action description. This yields LA4-33K, a dataset of 33K Language-Action (LA) episodes derived entirely from existing demonstrations without additional robot data collection. We further develop LA4VLA-1B, a lightweight 1B-parameter VLA model, and investigate three paradigms for incorporating language-action supervision into VLA learning: LA-only pretraining, sequential LA-to-VLA pretraining, and mixed LA-VLA pretraining. Across simulation and real-world tasks, LA-pretrained policies consistently outperform matched VLA-pretrained counterparts, while combining LA and VLA supervision leads to further gains. In particular, mixed LA-VLA pretraining improves the average success rate of LA4VLA-1B over the no-pretraining baseline by up to 17.8 and 45.0 percentage points in simulation and real-world tasks, respectively. These results establish LA4VLA as an effective and complementary pretraining strategy for building stronger and more robust VLA policies.

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作（VLA）模型通常通过映射视觉观测和语言指令来预训练。然而，密集的视觉信号往往会掩盖稀疏的语言指令，导致策略产生“视觉捷径”而非学习语言条件下的动作执行。为此，我们提出了 **LA4VLA**，一种语言-动作预训练框架，使策略无需视觉观测即可获得语言条件动作先验。这些先验捕获了跨任务和场景的可重用操作技能。具体而言，LA4VLA 将专家轨迹分解为原子动作段，并将其与低级动作描述配对，构建了包含 33K 个语言-动作（LA）片段的数据集 LA4-33K。实验表明，LA 预训练策略在仿真和真实世界任务中均显著优于传统的 VLA 预训练，混合 LA-VLA 预训练在模拟和真实任务中分别实现了 17.8 和 45.0 个百分点的成功率提升。

### 2. 方法动机分析
*   **驱动力**：VLA 模型在预训练中严重依赖视觉输入，导致模型过度关注场景外观（视觉拟合），而非指令本身的语义（意图执行）。
*   **现有方法痛点**：数据层面存在极大的不对称（图像/动作序列密集，指令稀疏），导致语言信息在联合训练中被边缘化，模型难以在视觉干扰（光照、视角变化）下保持鲁棒性。
*   **研究假设**：通过“去视觉化”的预训练，强迫模型从纯语言指令和本体状态中推断动作，能促使模型构建更健壮的语言-动作映射先验。

### 3. 方法设计详解
LA4VLA 的核心流程如下：
1.  **轨迹分解**：利用关键帧检测（静止状态检测与夹爪开关状态）将长轨迹切分为原子段。
2.  **数据生成（VLM 标注）**：使用 Qwen-3-VL-Plus 将分解的轨迹片段转化为结构化的 JSON 数据，包含原语标签和视觉无关（vision-agnostic）的自然语言指令（例如：“向上提升”、“向右水平移动”）。
3.  **人工验证**：通过评分制（0-3分）清洗数据，过滤不一致的片段，确保指令与轨迹的语义对齐。
4.  **预训练范式**：
    *   **LA-only**：纯语言-动作预训练（无视觉）。
    *   **LA-VLA**：先执行 LA 预训练，再进行标准 VLA 联合预训练。
    *   **MixPT**：将 LA 和 VLA 数据混合同步训练。

### 4. 方法对比分析
*   **本质区别**：现有的主流方法多关注视觉特征的强化，而 LA4VLA 旨在**解耦视觉 grounding 与语言指令的意图 grounding**。
*   **创新贡献**：提出了一种无需额外机器人数据收集的轨迹重组流水线（LA4-33K），并证明了语言-动作先验在解决“视觉过拟合”问题上的核心作用。
*   **适用场景**：适用于需要抗干扰能力强、对复杂语义指令响应敏感的机器人操作任务。

### 5. 实验分析
*   **验证方法**：在 MetaWorld 和 LIBERO 仿真基准，以及真实场景 xArm6 压按钮、书本插入、饮料放置任务上进行测试。
*   **关键结果**：LA-VLA 预训练在 MetaWorld 上平均成功率提升 17.8%，真实世界任务成功率高达 83.3%。
*   **主要优势**：极强地增强了模型对视觉扰动（如 Gaussian 噪声）的鲁棒性，且表现出更清晰的指令-动作聚类特征。
*   **主要局限**：原子动作的预定义仍需一定的领域知识，且轨迹分割依赖于现有的专家数据质量。

### 6. 实用指南
*   **开源情况**：已在 GitHub (https://github.com/MINT-SJTU/LA4VLA) 开源。
*   **实现细节**：建议在分割时利用好机器人本体状态的 Savitzky-Golay 滤波，并保证 VLM 提示词中包含明确的动作类别约束。
*   **迁移可能**：该框架可以平滑迁移到任何基于 Transformer 的 VLA 架构（如 RT-1/RT-2 变体），只需替换预训练数据输入格式。

### 7. 总结
*   **核心思想**：通过去视觉化的动作片段预训练，解耦语言意图与视觉干扰。
*   **速记版 pipeline**：
    1.  从长轨迹中检测关键帧（静止点、夹爪开关）。
    2.  利用 VLM 将轨迹分段并生成自然语言指令。
    3.  清洗数据形成 vision-agnostic 原子动作数据集。
    4.  通过预训练任务强迫模型理解指令与动作的本质联系。

**Key Findings:**

- To address this limitation, we propose LA4VLA, a language-action pretraining framework that enables policies to acquire language-conditioned action priors without visual observations.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.27295v1)
- [arXiv](https://arxiv.org/abs/2606.27295v1)

---

