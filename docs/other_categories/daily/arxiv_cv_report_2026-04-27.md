time: 20260427

# Arxiv Computer Vision Papers - 2026-04-27

## Executive Summary

以下是为您准备的每日报告执行摘要，涵盖2026年4月24日发布的10篇计算机视觉Arxiv论文。

---

### 执行摘要：2026年4月24日 Arxiv 计算机视觉论文快报

**1. 主要主题与趋势**

本期论文呈现出三个核心趋势：
- **3D与场景理解的深化**：多篇工作致力于从更复杂、非理想化的输入（如遮挡、长尾分布、网络视频）中恢复或理解3D信息（论文#2, #3, #4），体现了从“实验室完美场景”向“野外真实世界”的坚定迈进。
- **感知-行动闭环与具身智能**：计算机视觉正从“看”向“做”延伸。论文#1（轨迹优化）和#5（机器人操控）将视觉信息作为决策与行动的驱动信号，表明视觉与机器人学、路径规划的交叉领域持续升温。
- **基础模型的适配与扩展**：CLIP等大型视觉-语言模型仍是核心工具，但研究重点已转向如何针对特定任务（如小样本动作识别、图像理解）进行高效、鲁棒的微调（论文#6, #10），并探索其在结构化场景理解（如场景图生成，论文#8）中的应用。

**2. 值得关注的创新性论文**

- **《EV-CLIP: Efficient Visual Prompt Adaptation for CLIP...》**：针对视觉挑战下（如遮挡、光照变化）的小样本动作识别任务，提出高效的视觉提示适配方法，具有显著的应用价值。其创新在于解决了一个实际痛点：在数据稀缺且环境复杂的场景中快速部署视觉能力。
- **《CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding》**：提出“组合式接地对比”方法，聚焦于多图像间细粒度交互理解。这超越了传统的单图或简单多图匹配，对需要跨图像推理的任务（如对比分析、故事线理解）有重要启发。
- **《GazeVLA: Learning Human Intention for Robotic Manipulation》**：将人类注视信息引入机器人操控的视觉-语言-动作模型，直接建模人类意图。这一跨模态的引入为机器人学习更自然、更符合预期的交互行为提供了新范式。

**3. 新兴研究方向与技术**

- **自我监督的3D学习**：论文#3展示了仅使用网络视频进行端到端的自我监督3D重建，这代表了一种摆脱昂贵3D标注、利用海量视频数据的极具潜力的方向。
- **长尾分布下的强健感知**：论文#2专门针对互联网照片的长尾重建问题，反映了社区对模型公平性和泛化能力，特别是在数据分布极不平衡的真实场景下应用的日益关注。
- **开放词汇与结构化预测**：论文#8提出的开放词汇场景图生成，通过关系网格补全，旨在模型能识别训练中未见过的物体和关系，是推动场景理解走向通用化的关键技术。

**4. 重点推荐阅读的论文**

- **若您专注于3D视觉与重建**：强烈推荐阅读 **《SS3D: End2End Self-Supervised 3D from Web Videos》** (论文#3) 和 **《Long-tail Internet photo reconstruction》** (论文#2)。它们代表了3D领域两个最前沿且极具实践意义的挑战。
- **若您关注视觉-语言模型的应用与效率**：推荐 **《EV-CLIP: Efficient Visual Prompt Adaptation...》** (论文#6) 和 **《CGC: Compositional Grounded Contrast...》** (论文#10)。前者讲效率，后者讲深度理解，都是基础模型落地时的关键问题。
- **若您的兴趣在于机器人学习与具身智能**： **《GazeVLA: Learning Human Intention for Robotic Manipulation》** (论文#5) 是必读，它提出了一个富有启发的技术路径。
- **若您从事自动驾驶或场景理解**： **《Cross-Stage Coherence in Hierarchical Driving VQA》** (论文#7) 针对层级化驾驶视觉问答中的跨阶段一致性这一具体但重要的问题，提供了明确的基线和方法。

**总结：** 本周的论文反映出计算机视觉领域正从“孤立地看”转向“连贯地理解并行动”。自我监督、长尾稳健性、以及通过视觉-语言模型驱动决策是三大关键引擎。对于忙碌的研究人员，优先关注上述推荐论文，将能高效把握当前领域最活跃的技术潮流。

---

## Table of Contents

1. [ATRS: Adaptive Trajectory Re-splitting via a Shared Neural Policy for Parallel Optimization](#2604.22715v1)
2. [Long-tail Internet photo reconstruction](#2604.22714v1)
3. [SS3D: End2End Self-Supervised 3D from Web Videos](#2604.22686v1)
4. [PASR: Pose-Aware 3D Shape Retrieval from Occluded Single Views](#2604.22658v1)
5. [GazeVLA: Learning Human Intention for Robotic Manipulation](#2604.22615v1)
6. [EV-CLIP: Efficient Visual Prompt Adaptation for CLIP in Few-shot Action Recognition under Visual Challenges](#2604.22595v1)
7. [Cross-Stage Coherence in Hierarchical Driving VQA: Explicit Baselines and Learned Gated Context Projectors](#2604.22560v1)
8. [ReLIC-SGG: Relation Lattice Completion for Open-Vocabulary Scene Graph Generation](#2604.22546v1)
9. [Railway Artificial Intelligence Learning Benchmark (RAIL-BENCH): A Benchmark Suite for Perception in the Railway Domain](#2604.22507v1)
10. [CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding](#2604.22498v1)

---

## Papers

<a id='2604.22715v1'></a>
## [ATRS: Adaptive Trajectory Re-splitting via a Shared Neural Policy for Parallel Optimization](https://arxiv.org/abs/2604.22715v1)

**Authors:** Jiajun Yu, Guodong Liu, Li Wang, Pengxiang Zhou, Wentao Liu, Yin He, Chao Xu, Fei Gao, Yanjun Cao

**Published:** 2026-04-24

**Categories:** cs.RO

**Abstract:**

Parallel trajectory optimization via the Alternating Direction Method of Multipliers (ADMM) has emerged as a scalable approach to long-horizon motion planning. However, existing frameworks typically decompose the problem into parallel subproblems based on a predefined fixed structure. Such structural rigidity often causes optimization stagnation in highly constrained regions, where a few lagging subproblems delay global convergence. A natural remedy is to adaptively re-split these stagnating segments online. Yet, deciding when, where, and how to split exceeds the capability of rule-based heuristics. To this end, we propose ATRS, a novel framework that embeds a shared Deep Reinforcement Learning policy into the parallel ADMM loop. We formulate this adaptive adjustment as a Multi-Agent Shared-Policy Markov Decision Process, where all trajectory segments act as homogeneous agents and share a unified neural policy network. This parameter-sharing architecture endows the system with size invariance, enabling it to handle dynamically changing segment counts during re-splitting and generalize to arbitrary trajectory lengths. Furthermore, our formulation inherently supports zero-shot generalization to unseen environments, as our network relies solely on the internal states of the numerical solver rather than on the geometric features of the environment. To ensure solver stability, a Confidence-Based Election mechanism selects only the most stagnating segment for re-splitting at each step. Extensive simulations demonstrate that ATRS accelerates convergence, reducing the number of iterations by up to 26.0% and the computation time by up to 19.1%. Real-world experiments further confirm its applicability to both large-scale offline global planning and real-time onboard replanning within 35 ms per cycle, with no sim-to-real degradation.

**Analysis:**

### 1. 摘要翻译
ATRS（基于共享神经策略的自适应轨迹重分裂并行优化框架）利用交替方向乘子法（ADMM）实现长周期运动规划。现有框架依赖固定的离散结构，导致高度受限区域出现优化停滞，严重拖累全局收敛。本文提出ATRS，将轨迹重分裂建模为多智能体共享策略马尔可夫决策过程（MASP-MDP），使轨迹段作为同质智能体共享一个策略网络。该架构具备尺寸不变性，无需重训练即可处理动态变化的轨迹段数。此外，通过仅依赖数值求解器内部状态，实现了跨环境零样本迁移。实验证明，ATRS较传统方法显著提升了收敛速度，并实现了毫秒级的机载实时 replanning。

### 2. 方法动机分析
*   **驱动力**：解决并行轨迹优化中，因各子问题计算难度不均衡导致的“木桶效应”（即个别受限段收敛慢导致整体迟滞）。
*   **现有痛点**：传统方法采用预设的固定离散结构，缺乏在线自适应调整能力；手动设计启发式规则难以建模复杂的优化动力学。
*   **核心直觉**：如果能让受限区域的子段“拆分”出更多局部自由度（即增加优化变量），就能平衡计算负载。决策过程应交由学习型策略完成，且该策略必须脱离特定几何特征，仅关注优化器内在数值状态。

### 3. 方法设计详解
*   **流程总结**：
    1.  **状态感知**：计算每个子段的局部状态（残差、对偶变量范数、能量密度、残差趋势、时长等）和全局状态（池化后的收敛度量）。
    2.  **决策生成**：通过共享策略网络（Actor）输出：(1) 拆分倾向（Gate），(2) 空间切分比，(3) 时间偏差，(4) 时长膨胀因子。
    3.  **电选机制**：采用置信度选举（Confidence-Based Election），仅选出最高“拆分倾向”且超过阈值的单一段进行拆分，确保求解器稳定性。
    4.  **结构重构**：插入中间路点，更新CADMM参数，进入下一轮并行求解。
*   **模型结构**：采用Actor-Critic结构。Actor利用共享编码器+多分支头（解耦拆分与时空调整）；Critic采用TD3的孪生网络，缓解Q值过估计。
*   **算法逻辑**：状态空间通过对数值量进行`log10`归一化处理，确保数值稳定性。奖励函数设计极其细致，包含收敛进展（Residual reduction）、时间惩罚、以及对不必要拆分的正则项（Action-Level Regularization）。

### 4. 方法对比分析
*   **本质区别**：从传统的“参数调优”转向“问题结构调整”。
*   **创新贡献**：提出尺寸不变的MASP-MDP框架，将优化结构调整策略与求解器解耦，实现了真正的结构自适应。
*   **适用场景**：大规模、非均匀障碍物分布的无人机路径规划，尤其适用于机载受限算力场景。

### 5. 实验分析
*   **验证方法**：在不同障碍物密度（稀疏、中等、稠密）及不同长度（短、中、长）的环境中进行对比。
*   **关键结论**：在复杂环境中，ATRS比固定结构求解器减少迭代次数最高达26%，墙上时间缩减最高19.1%。
*   **主要优势**：高鲁棒性（零样本泛化至不同环境）、低延迟（35ms replanning）、收敛效率显著。
*   **主要局限**：在极端超长轨迹中，过多切分可能增加线程调度开销，边际收益递减。

### 6. 实用指南
*   **开源建议**：作者基于LibTorch实现，具备C++原生部署能力，直接调用底层的ADMM求解器。
*   **实现细节**：关键在于`log10`归一化和`tanh`激活函数的合理应用。训练时需注意TD3的超参数（Table I），特别是在Reward中引入`decay factor`以平滑训练收敛过程。
*   **迁移可能**：可直接迁移至任何基于ADMM的非线性优化任务，只需根据特定领域调整状态空间的组成部分。

### 7. 总结
*   **核心思想**：通过智能学习机制动态重构并行优化器的任务结构。
*   **速记版Pipeline**：
    1. 监控求解器残差等内部指标。
    2. 策略网络判断是否拆分受限段。
    3. 选举最优段并执行局部拆分。
    4. 更新约束并并行迭代求解。

**Key Findings:**

- To this end, we propose ATRS, a novel framework that embeds a shared Deep Reinforcement Learning policy into the parallel ADMM loop.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22715v1)
- [arXiv](https://arxiv.org/abs/2604.22715v1)

---

<a id='2604.22714v1'></a>
## [Long-tail Internet photo reconstruction](https://arxiv.org/abs/2604.22714v1)

**Authors:** Yuan Li, Yuanbo Xiangli, Hadar Averbuch-Elor, Noah Snavely, Ruojin Cai

**Published:** 2026-04-24

**Categories:** cs.CV

**Abstract:**

Internet photo collections exhibit an extremely long-tailed distribution: a few famous landmarks are densely photographed and easily reconstructed in 3D, while most real-world sites are represented with sparse, noisy, uneven imagery beyond the capabilities of both classical and learned 3D methods. We believe that tackling this long-tail regime represents one of the next frontiers for 3D foundation models. Although reliable ground-truth 3D supervision from sparse scenes is challenging to acquire, we observe that it can be effectively simulated by sampling sparse subsets from well-reconstructed Internet landmarks. To this end, we introduce MegaDepth-X, a large dataset of 3D reconstructions with clean, dense depth, together with a strategy for sampling sets of training images that mimic camera distributions in long-tail scenes. Finetuning 3D foundation models with these components yields robust reconstructions under extreme sparsity, and also enables more reliable reconstruction in symmetric and repetitive scenes, while preserving generalization to standard, dense 3D benchmark datasets.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文的分析如下：

### 1. 主要贡献总结
该论文针对互联网照片集中常见的“长尾”现象（即大多数场景由于缺乏充足的图像覆盖而难以重建）提出了解决方案。作者构建了名为 **MegaDepth-X** 的大规模高质量数据集，并提出了一种模拟稀疏数据分布的采样策略，使 3D 基础模型能够在极端稀疏、噪声大且分布不均的真实世界场景中实现鲁棒的 3D 重建。

### 2. 核心创新与方法论
*   **长尾场景模拟策略**：论文的核心洞察在于“如何利用现有的高质量、密集重建场景来服务于稀疏长尾场景”。作者通过从已知的高质量互联网地标数据中进行采样，模拟出长尾分布特有的稀疏性，从而解决了长尾场景中缺乏高质量真值监督（Ground-truth）的困境。
*   **MegaDepth-X 数据集**：该数据集提供了高质量的深度信息和相机位姿，为 3D 基础模型提供了经过系统性优化的训练基准。
*   **训练范式**：通过在模拟的稀疏训练集上进行微调（Finetuning），赋予了 3D 基础模型在几何重建任务中的泛化能力，使其在处理对称、重复纹理等复杂场景时比传统方法表现更优。

### 3. 对领域的潜在影响
*   **推动 3D 基础模型向“长尾化”演进**：目前大多数 3D 重建方法依赖于稠密视图，该论文将视角转向了非受限、非理想的互联网数据，这标志着 3D 视觉从“实验室环境”向“野外（In-the-wild）”真实世界的范式转移。
*   **重新定义数据价值**：论文证明了通过策略性采样（而非仅仅依靠增加数据量）可以有效提升模型的泛化边界，为后续的研究提供了一种低成本获取高质量监督信息的范式。

### 4. 受益的领域与应用
*   **数字孪生（Digital Twins）与遗产保护**：对于大多数难以进行无人机扫描或大批量拍摄的偏远或小型地标，该技术能显著降低高质量 3D 模型生成的门槛。
*   **增强现实（AR）与地理空间定位**：AR 应用往往依赖于用户拍摄的零散图像，该技术可提高移动设备在非最优环境下的定位与建图（SLAM）精度。
*   **自动驾驶与机器人视觉**：在复杂的长尾地理环境下（如未标注的野外环境），提升视觉里程计和深度估计的鲁棒性。

### 5. 可推断的局限性
*   **模拟分布与真实分布的差异**：虽然采样策略能够模拟稀疏性，但真实世界的长尾场景（如光照剧变、拍摄器材极度混杂、严重遮挡）可能比从高质量地标采样得到的“模拟稀疏”更复杂，模型在处理真实极端条件下的鲁棒性仍需验证。
*   **对于完全缺失先验知识的场景的泛化能力**：该方法仍依赖于基础模型的预训练权重，如果目标场景的特征与训练集中的互联网地标（如地标性建筑）差异巨大，模型的重建效果可能会出现“幻觉”或几何偏差。
*   **计算成本与推理实时性**：虽然论文强调了重构质量，但并未提及该方法在资源受限设备上的实时性，这对于许多落地应用（如移动端实时扫描）至关重要。

**总结建议**：这篇论文在 3D 视觉领域极具启发性，它巧妙地将“数据匮乏的挑战”转化为“高质量数据的精细重采样问题”，为未来实现大范围的、低门槛的地球级 3D 建模铺平了道路。

**Key Findings:**

- To this end, we introduce MegaDepth-X, a large dataset of 3D reconstructions with clean, dense depth, together with a strategy for sampling sets of training images that mimic camera distributions in long-tail scenes.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22714v1)
- [arXiv](https://arxiv.org/abs/2604.22714v1)

---

<a id='2604.22686v1'></a>
## [SS3D: End2End Self-Supervised 3D from Web Videos](https://arxiv.org/abs/2604.22686v1)

**Authors:** Marwane Hariat, Gianni Franchi, David Filliat, Antoine Manzanera

**Published:** 2026-04-24

**Categories:** cs.CV

**Abstract:**

We present SS3D, a web-scale SfM-based self-supervision pretraining pipeline for feed-forward 3D estimation from monocular video. Our model jointly predicts depth, ego-motion, and intrinsics in a single forward pass and is trained/evaluated as a coherent end-to-end 3D estimator. To stabilize joint learning, we use an intrinsics-first two-stage schedule and a unified single-checkpoint evaluation protocol. Scaling SfM self-supervision to unconstrained web video is challenging due to weak multi-view observability and strong corpus heterogeneity; we address these with a multi-view signal proxy (MVS) used for filtering and curriculum sampling, and with expert training distilled into a single student. Pretraining on YouTube-8M (~100M frames after filtering) yields strong cross-domain zero-shot transfer and improved fine-tuning performance over prior self-supervised baselines. We release the pretrained checkpoint and code.

**Analysis:**

作为计算机视觉领域的专家，我对这篇题为《SS3D: End2End Self-Supervised 3D from Web Videos》的论文进行了如下分析：

### 1. 论文贡献摘要
SS3D 提出了一种针对单目视频进行 3D 估计的端到端自监督预训练框架，通过在大规模无约束网络视频（YouTube-8M）上进行训练，成功实现了深度、自运动（ego-motion）和相机内参的联合预测。该研究证明了利用 SfM（运动恢复结构）信号在大规模异构数据上进行自监督学习，可以显著提升模型在多种场景下的零样本迁移能力和微调性能。

### 2. 核心创新与方法论
*   **多视角信号代理（MVS Proxy）与筛选机制**：为了克服网络视频中多视角观测质量弱、数据异构性强的难题，作者设计了 MVS 代理信号用于自动过滤低质量数据并优化课程学习（Curriculum Sampling），确保预训练阶段的数据质量。
*   **两阶段训练策略（Intrinsics-First Schedule）**：针对深度、运动和内参联合学习的非稳定性，采用了“内参优先”的两阶段调度机制，有效解决了多任务学习中的收敛难题。
*   **知识蒸馏架构**：采用专家模型训练后再蒸馏至单一学生模型的方式，平衡了模型复杂度与推理效率，使模型能够在一个前向传播（Forward Pass）内完成完整的 3D 推理。

### 3. 对领域的潜在影响
*   **从“受控场景”走向“广义场景”**：传统 SfM 或自监督深度估计多依赖于自动驾驶（如 KITTI）等高质量、受控的视频数据集，SS3D 证明了通过适当的筛选和训练策略，无约束的互联网视频可以作为无穷无尽的 3D 预训练源。
*   **端到端架构的实用化**：将 SfM 的几何约束嵌入到深度学习的前向网络中，不仅摆脱了传统 SfM 对光流估计的高依赖，还极大地提升了模型在未见域的泛化能力（Zero-shot Transfer）。

### 4. 受益的相关领域与应用
*   **具身智能与机器人导航**：该技术能让机器人仅通过单目摄像头即可理解环境的 3D 结构与自身运动，无需昂贵的深度传感器。
*   **增强现实（AR/VR）**：快速提取视频中的 3D 场景几何信息，有助于 AR 渲染中的物体放置与遮挡处理。
*   **视频编辑与特效**：为视频后处理（如 2D 转 3D、背景替换、运动重定向）提供更精准的几何先验信息。
*   **自动驾驶仿真**：从海量互联网视频中自动构建 3D 环境模型，用于训练自动驾驶算法，降低对物理实测数据的依赖。

### 5. 可推断的局限性
*   **性能瓶颈**：尽管使用了 1 亿帧的数据，但在处理极端运动模糊、动态物体遮挡（Dynamic Occlusion）或缺乏纹理的场景时，单目视频的 SfM 估计本质上仍存在尺度模糊性（Scale Ambiguity），模型可能依然难以完全解决。
*   **算力成本**：虽然模型推理是高效的，但在大规模（100M 帧）网络视频上执行 SfM 预处理和蒸馏训练，其前期算力开销和数据工程成本非常高，普通研究机构可能难以复现。
*   **端到端的鲁棒性**：虽然“端到端”简化了流程，但一旦模型内部的几何一致性在复杂场景下失效，相较于传统的管道式（Pipeline）方法，其纠错和调试难度更大。

**总结：** SS3D 的重要性在于它验证了**“利用互联网规模的数据进行几何感知自监督预训练”**的可行性。这类似于视觉大模型的路线，即通过海量数据去“涌现”对 3D 世界的几何认知，而非依赖特定领域的精细标注。

**Key Findings:**

- We present SS3D, a web-scale SfM-based self-supervision pretraining pipeline for feed-forward 3D estimation from monocular video.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22686v1)
- [arXiv](https://arxiv.org/abs/2604.22686v1)

---

<a id='2604.22658v1'></a>
## [PASR: Pose-Aware 3D Shape Retrieval from Occluded Single Views](https://arxiv.org/abs/2604.22658v1)

**Authors:** Jiaxin Shi, Guofeng Zhang, Wufei Ma, Naifu Liang, Adam Kortylewski, Alan Vuile

**Published:** 2026-04-24

**Categories:** cs.CV

**Abstract:**

Single-view 3D shape retrieval is a fundamental yet challenging task that is increasingly important with the growth of available 3D data. Existing approaches largely fall into two categories: those using contrastive learning to map point cloud features into existing vision-language spaces and those that learn a common embedding space for 2D images and 3D shapes. However, these feed-forward, holistic alignments are often difficult to interpret, which in turn limits their robustness and generalization to real-world applications. To address this problem, we propose Pose-Aware 3D Shape Retrieval (PASR), a framework that formulates retrieval as a feature-level analysis-by-synthesis problem by distilling knowledge from a 2D foundation model (DINOv3) into a 3D encoder. By aligning pose-conditioned 3D projections with 2D feature maps, our method bridges the gap between real-world images and synthetic meshes. During inference, PASR performs a test-time optimization via analysis-by-synthesis, jointly searching for the shape and pose that best reconstruct the patch-level feature map of the input image. This synthesis-based optimization is inherently robust to partial occlusion and sensitive to fine-grained geometric details. PASR substantially outperforms existing methods on both clean and occluded 3D shape retrieval datasets by a wide margin. Additionally, PASR demonstrates strong multi-task capabilities, achieving robust shape retrieval, competitive pose estimation, and accurate category classification within a single framework.

**Analysis:**

以下是对《PASR: Pose-Aware 3D Shape Retrieval from Occluded Single Views》的深度分析：

### 1. 摘要翻译
单视图3D形状检索是一项基础且极具挑战性的任务。现有方法主要分为两类：基于对比学习将点云特征映射到视觉-语言空间的方法，以及学习2D图像与3D形状共享嵌入空间的方法。然而，这些前馈式的全局对齐方式往往缺乏可解释性，限制了其在真实场景中的鲁棒性与泛化能力。为此，我们提出了**姿态感知3D形状检索（PASR）**，将检索构建为特征级的“分析-综合”问题，通过将2D基础模型（DINOv3）的知识蒸馏至3D编码器，架起了真实世界图像与合成网格之间的桥梁。在推理阶段，PASR通过“分析-综合”进行测试时优化，联合搜索能够最好地重建输入图像特征图的形状与姿态。这种合成式优化对局部遮挡具有内在鲁棒性，且对精细几何细节高度敏感。实验表明，PASR在干净及遮挡数据集上均大幅领先现有方法，并在同一框架下实现了稳健的形状检索、姿态估计与分类。

### 2. 方法动机分析
- **驱动力**：解决现有检索方法（如CLIP-based对齐）在处理遮挡时的脆弱性，以及对真实世界数据泛化不足的问题。
- **现有方法痛点**：
    - 全局嵌入（Global Embedding）忽视了局部几何细节，面对遮挡时无法匹配关键部位。
    - 强制对齐合成数据与真实图像，在域差异（Domain Gap）面前表现不佳。
- **研究假设**：通过将3D形状的渲染特征直接与2D特征图（而非全局向量）在像素空间进行对齐，并利用测试时优化（Test-time Optimization）动态调整姿态，可以从根本上解决遮挡带来的匹配偏移。

### 3. 方法设计详解
- **流程总结**：
    1. **训练阶段（蒸馏）**：通过DINOv3提取图像特征图，将3D点云特征投影至2D空间，通过特征重建损失函数（Masked Cosine Similarity）将2D语义知识蒸馏进3D点云编码器。
    2. **推理阶段（两阶段检索）**：
        *   **初始搜索**：在预设的一组离散姿态下，快速比对检索前Top-K候选。
        *   **测试时优化**：利用AdamW对Top-K候选的欧拉角（Pose）进行迭代优化，使得3D投影特征与查询图像特征图的余弦相似度损失最小化。
- **算法核心**：公式(8)中的掩码余弦相似度损失。它引入了前景掩码 $M$，确保损失函数仅关注对象区域，从而剔除背景噪声对遮挡处理的干扰。
- **协同机制**：3D特征聚合器通过KNN多尺度特征融合，保证了特征既包含局部几何，又具备全局语义。

### 4. 方法对比分析
- **本质区别**：从“提取特征后直接对比向量”转变为“渲染后在图像特征图空间做重建误差优化”，体现了从判别式到生成/重建式的思维转变。
- **创新贡献**：提出了特征级分析-综合的检索范式，无需Ground-truth掩码即可实现对遮挡的鲁棒对齐。
- **适用场景**：复杂背景、部分遮挡、真实世界场景（Instance-level Retrieval）。

### 5. 实验分析
- **验证方法**：在Pix3D和Pascal3D上进行多层级（L0-L3）遮挡实验。
- **关键结果**：在Pascal3D L3级（最难）遮挡下，检索准确率较当前最优方案（CMIC）提升了8.29%。
- **优势与局限**：
    - **优势**：极强的遮挡鲁棒性，无需特定类别训练即可较好地泛化到新形状。
    - **局限**：测试时优化带来了额外的推理延迟（50次迭代）。

### 6. 实用指南
- **开源情况**：论文涉及DINOv3与PointNeXt，实现基于PyTorch3D。
- **实现细节**：初期搜索建议设置192个预设姿态。测试时优化仅针对姿态（elev, azim, $\theta$）进行50步更新，学习率设为0.01。
- **迁移可能**：该框架的“渲染-优化”思路可无缝迁移至物体6D姿态估计或基于实例的增强现实任务。

### 7. 总结
- **核心思想**：通过特征图级分析-综合与测试时姿态优化实现稳健检索。
- **速记版Pipeline**：
    1. 用2D大模型知识教3D编码器认识细节。
    2. 快速初筛出一批候选物体。
    3. 把候选物体转动到最佳角度重合图像。
    4. 选出重合度最高的那个物体。

**Key Findings:**

- To address this problem, we propose Pose-Aware 3D Shape Retrieval (PASR), a framework that formulates retrieval as a feature-level analysis-by-synthesis problem by distilling knowledge from a 2D foundation model (DINOv3) into a 3D encoder.
- By aligning pose-conditioned 3D projections with 2D feature maps, our method bridges the gap between real-world images and synthetic meshes.
- PASR substantially outperforms existing methods on both clean and occluded 3D shape retrieval datasets by a wide margin.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22658v1)
- [arXiv](https://arxiv.org/abs/2604.22658v1)

---

<a id='2604.22615v1'></a>
## [GazeVLA: Learning Human Intention for Robotic Manipulation](https://arxiv.org/abs/2604.22615v1)

**Authors:** Chengyang Li, Kaiyi Xiong, Yuan Xu, Lei Qian, Yizhou Wang, Wentao Zhu

**Published:** 2026-04-24

**Categories:** cs.RO

**Abstract:**

Embodied foundation models have achieved significant breakthroughs in robotic manipulation, yet they still depend heavily on large-scale robot demonstrations. Although recent works have explored leveraging human data to alleviate this dependency, effectively extracting transferable knowledge remains a significant challenge due to the inherent embodiment gap between human and robot. We argue that the intention underlying human actions can serve as a powerful intermediate representation for bridging this gap. In this paper, we introduce a novel framework that explicitly learns and transfers human intention to facilitate robotic manipulation. Specifically, we model intention through gaze, as it naturally precedes physical actions and serves as an observable proxy for human intent. Our model is first pretrained on a large-scale egocentric human dataset to capture human intention and its synergy with action, followed by finetuning on a small set of robot and human data. During inference, the model adopts a Chain-of-Thought reasoning paradigm, sequentially predicting intention before executing the action. Extensive evaluations in simulation and real-world settings, across long-horizon and fine-grained tasks, and under few-shot and robustness benchmarks, show that our method consistently outperforms strong baselines, generalizes better, and achieves state-of-the-art performance.

**Analysis:**

### 1. 摘要翻译
具身基础模型在机器人操作领域取得了显著突破，但仍严重依赖大规模机器人演示数据。虽然近期研究尝试利用人类数据来缓解这种依赖，但由于人类与机器人之间存在固有的“具身鸿沟（embodiment gap）”，如何有效提取可迁移的知识仍是一项重大挑战。我们认为，人类动作背后的“意图（intention）”可以作为弥合这一鸿沟的强大中间表示。为此，我们提出了一个新的框架，通过显式学习和迁移人类意图来促进机器人操作。具体而言，我们将意图建模为“注视（gaze）”，因为它自然地先于物理动作，并可作为人类意图的可观察代理。我们的模型首先在超大规模的以自我为中心（egocentric）的人类数据集上进行预训练，以捕捉人类意图及其与动作的协同关系，随后在少量机器人和人类数据上进行微调。在推理阶段，模型采用“思维链（Chain-of-Thought）”推理范式，在执行动作之前顺序预测意图。在模拟和真实环境中的广泛评估表明，我们的方法在长周期和细粒度任务、少样本以及鲁棒性基准测试中均持续优于强基线模型，具有更好的泛化能力，达到了最先进的性能。

### 2. 方法动机分析
*   **驱动力**：旨在解决机器人训练数据匮乏且昂贵的瓶颈，利用人类丰富的行为数据实现高效的知识迁移。
*   **现有方法痛点**：现有方法要么仅将人类数据作为通用视觉表示，要么在共享动作空间中对齐，忽视了动作背后的逻辑动因，导致跨具身迁移效果不佳。
*   **研究假设**：人类的注视行为（Gaze）是意图的直接外部表现，以意图为中介可以解构决策过程，从而建立起人类与机器人之间有效的协同桥梁。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据构建**：聚合13个以自我为中心的人类数据集，统一坐标系，包含RGB、注视点和手部姿态（150M+帧）。
    2.  **预训练阶段**：在人类数据集上训练VLIA模型，学习“视觉/语言指令 → 意图（注视坐标） → 动作序列”的映射关系。
    3.  **微调阶段**：在少量机器人+人类混合数据上进行联合训练，机器人数据虽无意图标注，但通过与人类协同训练实现隐式迁移。
    4.  **推理执行**：采用思维链范式：先输入视觉图像与语言，模型生成意图TOKEN（注视点），再根据意图条件生成后续的高频动作流。
*   **模型结构**：基于PaliGemma（SigLIP编码器 + Gemma-2B LLM）。**Action Expert**采用条件流匹配（Conditional Flow Matching）技术，确保生成高频、连续的动作轨迹。
*   **关键公式意义**：$L = \lambda_{action}L_{action} + \lambda_{intent}L_{intent}$。通过权衡意图预测与动作生成损失，迫使模型在执行动作前必须先“理解”目标所在区域。

### 4. 方法对比分析
*   **本质区别**：从传统的“视觉到动作”的直接映射，升级为“视觉→意图→动作”的因果推理，引入了可解释的注视信息作为行为锚点。
*   **创新贡献**：显式建模人类意图作为中间表示，极大地增强了对未见场景、物体位置和背景的泛化能力。
*   **适用场景**：适用于长周期任务（Long-horizon）和需要极高精度对齐的细粒度操作（Fine-grained）。

### 5. 实验分析
*   **验证方法**：在AV-ALOHA模拟器和真实机器人平台（ALOHA、Unitree G1）上进行大规模对比实验。
*   **关键结论**：在OOD（分布外）场景中，该方法展现出显著优势，成功率相对提升明显，特别是在存在背景干扰或照明变化时。
*   **主要局限**：目前尚未在预训练阶段引入机器人数据，且人类数据与机器人数据未在统一的潜空间中实现深层对齐。

### 6. 实用指南
*   **开源情况**：项目主页 https://gazevla.github.io/。
*   **实现细节**：建议使用Pupil Neon类设备采集同步注视数据；动作空间需预处理归一化；在预训练初期建议冻结VLM骨干网络，仅微调Action Expert以防止表示坍缩。
*   **迁移可能**：该框架的“思维链+意图中间表示”思路可直接迁移至任何具备视觉输入的具身智能体任务，如无人机飞行决策或复杂工业装配。

### 7. 总结
*   **核心思想**：利用注视行为作为人类意图的中介，实现可解释的跨具身知识迁移。
*   **速记版pipeline**：
    1. 聚合海量人类注视视频进行预训练；
    2. 通过语言指令预测注视点（意图）；
    3. 以注视点为条件生成动作序列；
    4. 在少量机器人数据上联合微调。

**Key Findings:**

- In this paper, we introduce a novel framework that explicitly learns and transfers human intention to facilitate robotic manipulation.
- Extensive evaluations in simulation and real-world settings, across long-horizon and fine-grained tasks, and under few-shot and robustness benchmarks, show that our method consistently outperforms strong baselines, generalizes better, and achieves state-of-the-art performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22615v1)
- [arXiv](https://arxiv.org/abs/2604.22615v1)

---

<a id='2604.22595v1'></a>
## [EV-CLIP: Efficient Visual Prompt Adaptation for CLIP in Few-shot Action Recognition under Visual Challenges](https://arxiv.org/abs/2604.22595v1)

**Authors:** Hyo Jin Jon, Longbin Jin, Eun Yi Kim

**Published:** 2026-04-24

**Categories:** cs.CV

**Abstract:**

CLIP has demonstrated strong generalization in visual domains through natural language supervision, even for video action recognition. However, most existing approaches that adapt CLIP for action recognition have primarily focused on temporal modeling, often overlooking spatial perception. In real-world scenarios, visual challenges such as low-light environments or egocentric viewpoints can severely impair spatial understanding, an essential precursor for effective temporal reasoning. To address this limitation, we propose Efficient Visual Prompting for CLIP (EV-CLIP), an efficient adaptation framework designed for few-shot video action recognition across diverse scenes and viewpoints. EV-CLIP introduces two visual prompts: mask prompts, which guide the model's attention to action-relevant regions by reweighting pixels, and context prompts, which perform lightweight temporal modeling by compressing frame-wise features into a compact representation. For a comprehensive evaluation, we curate five benchmark datasets and analyze domain shifts to quantify the influence of diverse visual and semantic factors on action recognition. Experimental results demonstrate that EV-CLIP outperforms existing parameter-efficient methods in overall performance. Moreover, its efficiency remains independent of the backbone scale, making it well-suited for deployment in real-world, resource-constrained scenarios. The code is available at https://github.com/AI-CV-Lab/EV-CLIP.

**Analysis:**

### 1. 摘要翻译
CLIP在视频动作识别等视觉领域展示了强大的泛化能力。然而，现有的CLIP适配方法主要关注时间建模，忽略了视觉挑战（如低光照、自视视角）下的空间感知能力，这对于有效的时间推理至关重要。为此，我们提出了EV-CLIP，一种面向少样本视频动作识别的高效视觉提示框架。EV-CLIP引入两种视觉提示：**掩码提示（mask prompts）**，通过像素重加权引导模型关注动作相关区域；**上下文提示（context prompts）**，通过将帧级特征压缩为紧凑表征进行轻量级时间建模。实验表明，EV-CLIP在保持计算高效且独立于主干网络规模的同时，超越了现有的参数高效微调方法，非常适用于资源受限的实际场景。

### 2. 方法动机分析
- **驱动力**：在现实场景（如监控、智能眼镜）中，视频常伴随低光照、 egocentric（自视）视角等视觉挑战，严重损害空间感知，进而导致下游的时间建模失效。
- **痛点**：现有方法往往为了实现时间建模而引入复杂结构或全参数微调，导致计算开销大且通用性差，难以在真实场景中部署。
- **研究假设**：通过轻量级的、与主干无关（backbone-agnostic）的视觉提示，可以在不改变预训练CLIP模型内部结构的前提下，显式增强模型的空间聚焦能力和时间上下文关联。

### 3. 方法设计详解
- **流程总结**：
  1. **输入准备**：采用预训练的Omnivore小模型作为提示生成器（VM），其参数在实验中被证明无需过大。
  2. **掩码提示（Mask Prompt）**：将输入特征传入Swin-Unet架构的解码器，生成像素级的重加权掩码（$p_m$）。该掩码通过MinMax缩放归一化到[0, 1]，乘入视频帧以抑制背景噪声，突出动作主体。
  3. **上下文提示（Context Prompt）**：通过池化技术将VM提取的特征压缩为紧凑向量，经过线性层投影为上下文提示（$p_c$），并在最终的视频表示中进行加权融合。
  4. **一致性损失（Consistency Loss）**：引入帧间相似度损失，促使CLIP在处理视觉差异较大的帧时，依然能学习到连贯、稳定的动作表征。
- **关键公式**：掩码通过MinMax缩放（公式11）防止权重过小导致信息丢失；视频表示计算（公式14）通过平均池化将上下文提示与帧特征融合，实现高效时间聚合。

### 4. 方法对比分析
- **本质区别**：与需要修改Transformer内部结构（如插入Adapter或提示符 tokens）的方法不同，EV-CLIP采用**“外部辅助式”**设计，即提示生成器独立于CLIP视觉编码器，不增加CLIP内部的计算深度。
- **创新贡献**：首次在少样本视频识别中明确提出“空间重加权”的视觉提示，解决了视觉挑战导致的特征模糊问题。
- **适用场景**：适用于资源受限的边缘设备，以及需要快速适应新任务且标注数据极少的场景。

### 5. 实验分析
- **核心结论**：在ARID（低光照）和EK100Verb（自视）等极具挑战的基准上，EV-CLIP显著超越了现有方法。
- **主要优势**：极高的计算效率（高吞吐量）、极少的训练参数（仅约6M左右）、优异的模块化适配性（兼容CNN和ViT）。
- **主要局限**：在高强度动作变化序列上，相较于深度集成的模型，其长时序建模能力略显不足。

### 6. 实用指南
- **开源情况**：代码已开源（https://github.com/AI-CV-Lab/EV-CLIP）。
- **实现细节**：建议使用Omnivore-small作为生成器，λ超参数对 consistency loss 影响较大，需针对特定数据集优化（文中范围0-10）。
- **迁移可能**：该框架的思想可直接迁移至图像修复、弱光视频增强等需要空间关注的任务中。

### 7. 总结
- **核心思想**：通过外部轻量提示显式增强空间聚焦与时间上下文，实现高效适配。
- **速记版pipeline**：
    1. 用小模型生成像素级遮罩图。
    2. 对视频原图进行像素增强（去噪与提亮）。
    3. 提取全局时间上下文向量。
    4. 将增强后的视频特征与上下文向量融合识别。

**Key Findings:**

- To address this limitation, we propose Efficient Visual Prompting for CLIP (EV-CLIP), an efficient adaptation framework designed for few-shot video action recognition across diverse scenes and viewpoints.
- Experimental results demonstrate that EV-CLIP outperforms existing parameter-efficient methods in overall performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22595v1)
- [arXiv](https://arxiv.org/abs/2604.22595v1)

---

<a id='2604.22560v1'></a>
## [Cross-Stage Coherence in Hierarchical Driving VQA: Explicit Baselines and Learned Gated Context Projectors](https://arxiv.org/abs/2604.22560v1)

**Authors:** Gautam Kumar Jain, Carsten Markgraf, Julian Stähler

**Published:** 2026-04-24

**Categories:** cs.CV, cs.AI

**Abstract:**

Graph Visual Question Answering (GVQA) for autonomous driving organizes reasoning into ordered stages, namely Perception, Prediction, and Planning, where planning decisions should remain consistent with the model's own perception. We present a comparative study of cross-stage context passing on DriveLM-nuScenes using two complementary mechanisms. The explicit variant evaluates three prompt-based conditioning strategies on a domain-adapted 4B VLM (Mini-InternVL2-4B-DA-DriveLM) without additional training, reducing NLI contradiction by up to 42.6% and establishing a strong zero-training baseline. The implicit variant introduces gated context projectors, which extract a hidden-state vector from one stage and inject a normalized, gated projection into the next stage's input embeddings. These projectors are jointly trained with stage-specific QLoRA adapters on a general-purpose 8B VLM (InternVL3-8B-Instruct) while updating only approximately 0.5% of parameters. The implicit variant achieves a statistically significant 34% reduction in planning-stage NLI contradiction (bootstrap 95% CIs, p < 0.05) and increases cross-stage entailment by 50%, evaluated with a multilingual NLI classifier to account for mixed-language outputs. Planning language quality also improves (CIDEr +30.3%), but lexical overlap and structural consistency degrade due to the absence of driving-domain pretraining. Since the two variants use different base models, we present them as complementary case studies: explicit context passing provides a strong training-free baseline for surface consistency, while implicit gated projection delivers significant planning-stage semantic gains, suggesting domain adaptation as a plausible next ingredient for full-spectrum improvement.

**Analysis:**

### 1. 摘要翻译
图视觉问答（GVQA）将自动驾驶推理划分为感知、预测和规划三个阶段。规划决策必须与模型自身的感知保持一致。本文提出了两种互补的交叉阶段上下文传递机制，应用于DriveLM-nuScenes数据集。显式变体在领域适配的4B VLM上测试了三种基于提示的条件策略，无需额外训练，将NLI（自然语言推理）矛盾减少了42.6%，建立了强大的零训练基线。隐式变体引入了门控上下文投影仪，从前一阶段提取隐藏状态向量，经归一化和门控后注入下一阶段的输入嵌入。该变体在通用8B VLM上通过阶段特定LoRA适配器进行联合训练，仅更新约0.5%的参数。隐式变体使规划阶段NLI矛盾显著减少34%，交叉阶段蕴含增加50%。尽管由于缺乏领域预训练导致词汇重叠和结构一致性下降，但规划语言质量（CIDEr +30.3%）得到显著提升。本文通过显式基线与隐式门控投影的对比，展示了领域适配是实现全谱性能提升的关键补充。

### 2. 方法动机分析
*   **驱动力**：解决多阶段自主驾驶推理中的“阶段间不一致”问题，确保规划动作不仅局限流畅，更要符合感知与预测的逻辑。
*   **痛点**：现有端到端模型虽然传递特征，但往往编码的是物理实体而非语义状态；链式推理（CoT）高度依赖自回归窗口，缺乏可训练的显式路由机制；独立适配器训练导致阶段间逻辑断裂。
*   **核心假设**：引入可训练的、模块化的隐式语义路由机制，可以将经过压缩的各阶段推理状态，作为“上下文胶水”跨阶段传递，从而提升全局逻辑一致性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **上下文提取**：在阶段 $s_k$ 完成后，提取最后一层隐藏状态中对应最后一个非视觉/非填充提示Token $h_k$ 作为语义压缩表示。
    2.  **门控投影**：使用 $W_k$ 投影并结合可学习的标量门 $g_k$ 进行变换：$\tilde{h}_k = \sigma(g_k) \cdot W_k \left( \frac{h_k}{\|h_k\|_2 + \epsilon} \right)$。$L_2$ 归一化是关键，防止隐藏状态的高范数引起生成崩溃。
    3.  **注入**：将投影向量 $\tilde{h}_k$ 直接累加到下一阶段 $s_{k+1}$ 的输入嵌入 $E_{k+1}$ 的对应位置 $\tau_{k+1}$。
*   **关键设计**：采用了“阶段特定LoRA + 门控投影”的架构。投影器初始化时门控接近关闭（$\sigma(g_k) \approx 0.029$），随着训练打开，形成从独立到上下文依赖的训练课程。

### 4. 方法对比分析
*   **本质区别**：不同于传统的特征对齐（Feature Alignment）或简单的文本拼接，本方法在隐藏空间进行“语义状态传递”，且不干扰原始VLM骨干架构。
*   **创新贡献**：提出了一种轻量级的门控投影模块，实现了参数效率极高（~0.5%）的模块化路由，并建立了基于NLI的交叉阶段一致性度量协议。
*   **适用场景**：适用于任何分阶段（Hierarchical）的视觉推理任务，特别是对逻辑一致性要求极高的自动驾驶安全领域。

### 5. 实验分析
*   **关键结论**：在保持通用VLM骨干冻结的情况下，隐式投影成功将规划阶段矛盾减少34%，证明了模块化路由的有效性。
*   **主要优势**：显著提升了跨阶段的语义一致性和规划语言质量（CIDEr +30.3%）。
*   **主要局限**：缺乏驾驶领域预训练导致Lexical Overlap（词汇重叠）性能受损，表明隐式路由虽优于语义逻辑，但表面层面的语言对齐仍需特定领域数据的支撑。

### 6. 实用指南
*   **迁移建议**：对于其他多步推理模型，可直接通过提取隐藏层状态（Hidden States）实现上下文路由。关键在于 $L_2$ 归一化和门控初始化的策略，以避免对预训练模型生成能力的破坏。
*   **注意事项**：必须保证注入位置（Token位置）在训练和推理时的一致性，防止发生分布偏移（Distribution Shift）。

### 7. 总结
*   **核心思想**：通过可训练门控投影，在不同推理阶段间路由压缩后的语义状态。
*   **速记版pipeline**：
    1. 提取当前阶段隐藏状态特征；
    2. 归一化并执行门控线性变换；
    3. 叠加到下一阶段的输入嵌入中；
    4. 阶段特定LoRA同步联合优化。

**Key Findings:**

- We present a comparative study of cross-stage context passing on DriveLM-nuScenes using two complementary mechanisms.
- Since the two variants use different base models, we present them as complementary case studies: explicit context passing provides a strong training-free baseline for surface consistency, while implicit gated projection delivers significant planning-stage semantic gains, suggesting domain adaptation as a plausible next ingredient for full-spectrum improvement.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22560v1)
- [arXiv](https://arxiv.org/abs/2604.22560v1)

---

<a id='2604.22546v1'></a>
## [ReLIC-SGG: Relation Lattice Completion for Open-Vocabulary Scene Graph Generation](https://arxiv.org/abs/2604.22546v1)

**Authors:** Amir Hosseini, Sara Farahani, Xinyi Li, Suiyang Guang

**Published:** 2026-04-24

**Categories:** cs.CV

**Abstract:**

Open-vocabulary scene graph generation (SGG) aims to describe visual scenes with flexible relation phrases beyond a fixed predicate set. Existing methods usually treat annotated triplets as positives and all unannotated object-pair relations as negatives. However, scene graph annotations are inherently incomplete: many valid relations are missing, and the same interaction can be described at different granularities, e.g., \textit{on}, \textit{standing on}, \textit{resting on}, and \textit{supported by}. This issue becomes more severe in open-vocabulary SGG due to the much larger relation space. We propose \textbf{ReLIC-SGG}, a relation-incompleteness-aware framework that treats unannotated relations as latent variables rather than definite negatives. ReLIC-SGG builds a semantic relation lattice to model similarity, entailment, and contradiction among open-vocabulary predicates, and uses it to infer missing positive relations from visual-language compatibility, graph context, and semantic consistency. A positive-unlabeled graph learning objective further reduces false-negative supervision, while lattice-guided decoding produces compact and semantically consistent scene graphs. Experiments on conventional, open-vocabulary, and panoptic SGG benchmarks show that ReLIC-SGG improves rare and unseen predicate recognition and better recovers missing relations.

**Analysis:**

这是一份关于论文 **ReLIC-SGG: Relation Lattice Completion for Open-Vocabulary Scene Graph Generation** 的深度方法分析报告。

---

### 1. 摘要翻译
开放词汇场景图生成（SGG）旨在描述超越固定谓词集的视觉场景。现有的方法通常将标注的三元组视为正样本，并将所有未标注的对象对关系视为负样本。然而，场景图标注本质上是不完整的：许多有效的关系缺失，且同一交互可以用不同的粒度描述（如 `on`，`standing on`，`supported by`）。在开放词汇 SGG 中，由于关系空间巨大，该问题更为严重。我们提出了 **ReLIC-SGG**，一个关系不完整性感知框架，将未标注的关系视为潜在变量而非绝对负样本。ReLIC-SGG 构建了一个语义关系格来建模谓词间的相似性、蕴含性和矛盾性，并利用它从视觉语言兼容性、图上下文和语义一致性中推断缺失的正样本。正样本-未标记（PU）图学习目标进一步减少了假阴性监督，而格引导的解码则产生了紧凑且语义一致的场景图。

---

### 2. 方法动机分析
*   **驱动力**：解决开放词汇 SGG 中“标注稀疏性”带来的虚假负样本惩罚问题，提升模型对细粒度、未标注但合理关系的识别能力。
*   **痛点**：传统方法将所有未标注关系视为负样本，在开放词汇场景下，由于语义关联（如 `standing on` 蕴含 `on`），很多标注缺失的关系被模型错误抑制。
*   **研究假设**：未标注关系不应直接视为负样本，而是可以通过其语义（通过关系格）和图上下文推断出的“潜在正样本”。

---

### 3. 方法设计详解
ReLIC-SGG 的 pipeline 主要包含四个核心步骤：
1.  **高召回率候选生成**：利用 Faster R-CNN 和 CLIP 文本编码器，基于视觉-语言兼容性计算 score，提取 Top-K 个候选关系对。
2.  **语义关系格构建**：构建一个包含 `Asim`（相似性）、`Aent`（蕴含性）、`Acon`（矛盾性）三类边的图结构（Lattice）。
    *   **核心逻辑**：通过公式 9 进行信任度传播（Confidence Propagation）。相似谓词相互支撑，细粒度谓词（如 `standing on`）支撑粗粒度谓词（如 `on`）。
    *   **矛盾抑制**：通过公式 10 抑制相互矛盾的预测，确保输出一致性。
3.  **潜在关系补全（Latent Relation Completion）**：使用公式 13 计算未标注关系为正样本的后验概率 $q_{ij}^r$。该概率综合了直接兼容性、格的一致性、图上下文（Graph Transformer 提取）和矛盾抑制项。
4.  **PU 图学习（Positive-Unlabeled Learning）**：
    *   将标注关系视为确定正样本（Fixed）。
    *   将未标注关系通过 $q_{ij}^r$ 作为软标签，并结合负可靠性估计 $\rho_{ij}^r$ 进行监督训练。
    *   引入图一致性 loss（公式 21）约束语义合理性。

---

### 4. 方法对比分析
*   **本质区别**：从传统的“分类任务”转变为“部分标注下的补全任务”。
*   **创新点**：引入**语义关系格**显式建模谓词间的结构化依赖（而非孤立分类），并结合 **PU 学习**机制，有效缓解了假阴性导致的性能偏见。
*   **适用场景**：所有标注不完整、谓词空间大且具有强语义关联的图推理任务。

---

### 5. 实验分析（精简版）
*   **结论**：在 VG150 和 PSG 数据集上，ReLIC-SGG 在 mR@K 指标上显著超越了现有基线。
*   **优势**：在 Open-Vocabulary 设置下，对“未见（Unseen）”谓词的提升远高于“已见”谓词，证明了其语义迁移能力。
*   **局限**：对候选集数量 $K$ 较敏感，过大的 $K$ 会增加冗余和计算代价。

---

### 6. 实用指南
*   **实现细节**：训练分为三阶段：(1) 暖机训练（Warm-up）；(2) 关系格构建与初始补全；(3) 联合优化。
*   **调参建议**：$\lambda_{sim}, \lambda_{ent}, \lambda_{con}$ 等超参数用于平衡语义传播强度，建议针对不同数据集的标注密度进行微调。
*   **迁移迁移**：核心的 Lattice 构建和 PU 学习模块可直接迁移到视频场景图（Video SGG）或 3D 场景图生成中，因为这些任务的标注同样存在严重的稀疏问题。

---

### 7. 总结
*   **核心思想**：通过语义关系格与 PU 学习，将缺失标签转化为推断出的潜在正样本。
*   **速记版 Pipeline**：
    1. 生成候选关系对；
    2. 构建语义关系格并传播置信度；
    3. 推断缺失关系作为软标签；
    4. 使用 PU 学习进行稳健优化。

**Key Findings:**

- We propose \textbf{ReLIC-SGG}, a relation-incompleteness-aware framework that treats unannotated relations as latent variables rather than definite negatives.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22546v1)
- [arXiv](https://arxiv.org/abs/2604.22546v1)

---

<a id='2604.22507v1'></a>
## [Railway Artificial Intelligence Learning Benchmark (RAIL-BENCH): A Benchmark Suite for Perception in the Railway Domain](https://arxiv.org/abs/2604.22507v1)

**Authors:** Annika Bätz, Pavel Klasek, Seo-Young Ham, Philipp Neumaier, Martin Köppel, Martin Lauer

**Published:** 2026-04-24

**Categories:** cs.CV

**Abstract:**

Automated train operation on existing railway infrastructure requires robust camera-based perception, yet the railway domain lacks public benchmark suites with standardized evaluation protocols that would enable reproducible comparison of approaches. We present RAIL-BENCH, the first perception benchmark suite for the railway domain. It comprises five challenges - rail track detection, object detection, vegetation segmentation, multi-object tracking, and monocular visual odometry - each tailored to the specific characteristics of railway environments. RAIL-BENCH provides curated training and test datasets drawn from diverse real-world scenarios, evaluation metrics, and public scoreboards (https://www.mrt.kit.edu/railbench). For the rail track detection challenge we introduce LineAP, a novel segment-based average precision metric that evaluates the geometric accuracy of polyline predictions independently of instance-level grouping, addressing key limitations of existing line detection metrics.

**Analysis:**

### 1. 摘要翻译
自动化列车运行依赖于强大的基于摄像头的感知系统，但铁路领域目前缺乏标准化的评估基准来支持算法的复现与对比。本文提出了 RAIL-BENCH，这是首个面向铁路领域的感知基准套件。它包含五个挑战任务——轨道检测、物体检测、植被分割、多目标跟踪和单目视觉里程计——并针对铁路环境进行了专门优化。RAIL-BENCH 提供了从多样化现实场景中提取的带标注数据集、评估指标及在线排行榜。针对轨道检测任务，本文提出了一种名为 LineAP 的新型基于片段的平均精度指标，它能够独立于实例分组对折线预测的几何精度进行评估，解决了现有线检测指标的关键局限。

### 2. 方法动机分析
- **驱动力**：铁路环境与汽车环境存在本质差异（如独特的轨道结构、特定的物体类别、复杂的几何约束），现有的汽车感知基准（如KITTI、nuScenes）无法满足铁路感知的需求。
- **现有痛点**：当前轨道检测指标（如TuSimple, CULane）存在局限：TuSimple依赖固定的垂直采样网格，对水平线效果差；CULane通过多边形IoU计算，无法精确反映线段的局部几何偏差。此外，缺乏统一的基准导致无法公正对比不同算法。
- **研究假设**：通过将折线分解为固定长度的片段进行匹配，可以更精确地度量局部几何准确性，并有效处理部分检测和过检测问题。

### 3. 方法设计详解
- **LineAP 核心流程**：
  1. **分段**：将预测折线和真值折线均划分为固定长度的短片段（最后一段可更短）。
  2. **匹配策略**：基于置信度得分降序排列，构建二分图。
  3. **度量标准**：使用“欧氏距离+方向差”作为双重匹配阈值。若中心点欧氏距离小于阈值且方向角差小于阈值，则视为匹配。
  4. **最优分配**：利用最小权重最大匹配算法进行二分图匹配，计算Precision-Recall曲线及AP。
- **创新点**：LineAP 允许“部分检测”贡献真阳性（TP），通过 penalize（惩罚）未匹配的预测片段来抑制过检测，通过置信度过滤实现对全曲线的综合评价。

### 4. 方法对比分析
- **本质区别**：传统指标多以整条线为单位，LineAP 转向基于“片段级别”的匹配，实现了几何误差与分组准确性的解耦。
- **创新贡献**：提出了一种不依赖特定几何假设（如垂直线）的通用评价框架，其思想可无缝迁移至车道线检测或复杂管线检测任务。

### 5. 实验分析
- **关键结论**：在 RAIL-BENCH 上，YOLinO 架构在 LineAP 指标上表现优于 PINet，证实了其针对线段检测设计的有效性；而 PINet 在 ChamferAP 上更强，说明两者在“线段局部特征”与“全局几何平滑性”上各有优劣。
- **优势**：该指标能更细致地反映模型在复杂铁路道岔处的性能。
- **局限**：分段处理增加了计算复杂度，对超参数（段长、距离/角度阈值）的敏感性尚需进一步探索。

### 6. 实用指南
- **开源情况**：已发布在 https://www.mrt.kit.edu/railbench。
- **实现细节**：在迁移此评价方法时，需注意：1. 相对距离阈值（如AP@0.1代表图像宽度的0.1%）是归一化处理的核心；2. 预处理阶段需定义好忽略区域（ignore regions）。
- **迁移建议**：该评价体系非常适合任何长条状结构的检测任务（如输电线、复杂道路标记），直接将折线分段处理即可应用。

### 7. 总结
- **核心思想**：通过片段级分解实现对折线几何精度的精细度量。
- **速记版 pipeline**：
  1. 将所有线段切割成短小片段。
  2. 匹配片段的坐标和角度信息。
  3. 通过匈牙利算法寻找最优配对。
  4. 计算匹配后的曲线精度。

**Key Findings:**

- We present RAIL-BENCH, the first perception benchmark suite for the railway domain.
- For the rail track detection challenge we introduce LineAP, a novel segment-based average precision metric that evaluates the geometric accuracy of polyline predictions independently of instance-level grouping, addressing key limitations of existing line detection metrics.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22507v1)
- [arXiv](https://arxiv.org/abs/2604.22507v1)

---

<a id='2604.22498v1'></a>
## [CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding](https://arxiv.org/abs/2604.22498v1)

**Authors:** Lihao Zheng, Zhenwei Shao, Yu Zhou, Yan Yang, Xintian Shen, Jiawei Chen, Hao Ma, Tao Wei

**Published:** 2026-04-24

**Categories:** cs.CV, cs.AI

**Abstract:**

Although Multimodal Large Language Models (MLLMs) have advanced rapidly, they still face notable challenges in fine-grained multi-image understanding, often exhibiting spatial hallucination, attention leakage, and failures in object constancy. In addition, existing approaches typically rely on expensive human annotations or large-scale chain-of-thought (CoT) data generation. We propose Compositional Grounded Contrast (abbr. CGC), a low-cost full framework for boosting fine-grained multi-image understanding of MLLMs. Built on existing single-image grounding annotations, CGC constructs compositional multi-image training instances through Inter-Image Contrast and Intra-Image Contrast, which introduce semantically decoupled distractor contexts for cross-image discrimination and correlated cross-view samples for object constancy, respectively. CGC further introduces a Rule-Based Spatial Reward within the GRPO framework to improve source-image attribution, spatial alignment, and structured output validity under a Think-before-Grounding paradigm. Experiments show that CGC achieves state-of-the-art results on fine-grained multi-image benchmarks, including MIG-Bench and VLM2-Bench. The learned multi-image understanding capability also transfers to broader multimodal understanding and reasoning tasks, yielding consistent gains over the Qwen3-VL-8B base model on MathVista (+2.90), MuirBench (+2.88), MMStar (+1.93), MMMU (+1.77), and BLINK (+1.69).

**Analysis:**

### 1. 摘要翻译
尽管多模态大语言模型（MLLMs）发展迅速，但在细粒度多图像理解方面仍面临挑战，常出现空间幻觉、注意力泄漏及对象恒常性缺失等问题。此外，现有方法多依赖昂贵的人工标注或大规模思维链（CoT）数据生成。本文提出“组合基础对比”（CGC），这是一个低成本的完整框架，旨在增强MLLM的细粒度多图像理解能力。CGC利用现有的单图像接地（Grounding）标注，通过“跨图像对比”和“图像内对比”构建组合式多图像训练实例，前者引入语义解耦的干扰上下文以进行跨图像区分，后者生成相关的跨视图样本以强化对象恒常性。CGC进一步在GRPO框架内引入基于规则的空间奖励，在“先思考后接地”（Think-before-Grounding）范式下提升源图像归因、空间对齐和结构化输出的有效性。实验表明，CGC显著提升了MLLM的多图像理解能力。

### 2. 方法动机分析
*   **驱动力**：细粒度多图像理解要求模型具备跨图像属性归因（Target Attribution）与精确定位能力，单纯的语义比较不足以解决这些问题。
*   **现有方法痛点**：
    1.  **注意力泄漏**：视觉特征在多图像序列间相互污染，导致跨图像干扰。
    2.  **空间幻觉**：缺乏坐标级监督，定位不准确。
    3.  **对象恒常性失效**：难以在不同视角或场景变化下追踪同一实体。
    4.  **成本与数据限制**：过分依赖昂贵的标注或不稳定的CoT生成。
*   **研究假设**：通过将细粒度多图像任务重构为“基于空间 groundings 的组合推理问题”，利用自动化合成的对比数据结合强化学习（RL），可以以低成本实现稳健的推理与归因。

### 3. 方法设计详解
*   **流程总结**：
    1.  **自动对比数据合成**：
        *   **跨图像对比（Inter-Image Contrast）**：将来自不同场景的图像组合，并打乱对应的引用关系，迫使模型学习显式的源图像归因，而非依赖位置偏差。
        *   **图像内对比（Intra-Image Contrast）**：对同一图像进行不同缩放比例（Focus View vs. Context View）的裁剪，并在随机坐标映射下建立对象恒常性约束。
    2.  **强化学习优化（GRPO）**：
        *   **Think-before-Grounding**：模型必须先输出推理轨迹（reasoning trace），再输出JSON格式的接地结果（img_idx, label, bbox_2d）。
        *   **规则化空间奖励（Rule-Based Spatial Reward）**：总奖励 $R = R_{miou} + R_{format}$。$R_{miou}$ 通过针对性的集合匹配（仅在同一image_idx内计算IoU）来强制要求正确的源图像归因。
*   **模型结构**：基于预训练的MLLM作为Actor，利用GRPO算法进行策略优化，无需修改模型架构。

### 4. 方法对比分析
*   **本质区别**：与现有模型直接进行端到端多图像微调不同，CGC将重点放在了“数据构建的结构化对比”与“以定位为核心的奖励信号”上。
*   **创新贡献**：提出了一套不需要人工标注的自动化数据构建流水线，并证明了低层空间接地（Grounding）是实现高层多图像推理的有效基础。
*   **适用场景**：适用于需要精确多图像定位、追踪、对比和科学图表分析的复杂推理任务。

### 5. 实验分析（精简版）
*   **验证方法**：在MIG-Bench和VLM2-Bench等专门的多图像基准上进行评测，并扩展至MathVista、MMMU等通用推理任务。
*   **关键结果**：在Qwen3-VL-8B上，CGC-8B在MIG-Bench上平均分从43.66提升至67.57，表现优于许多大尺寸模型。
*   **主要优势**：极强的鲁棒性，有效抑制幻觉；迁移能力强，无需任务特定微调。
*   **主要局限**：对极小目标或因强烈光影/视角剧变导致外观彻底改变的对象，仍存在定位挑战。

### 6. 实用指南
*   **开源情况**：论文明确表示框架设计为低成本、全自动化，建议参考作者开源的自动化合成代码和GRPO实现逻辑。
*   **迁移可能**：可直接迁移至任何具备Vision-Language能力的Base Model，特别推荐作为处理多图交互类任务的增强基座。
*   **关键技巧**：设置正确的KL惩罚项（0.01）以防灾难性遗忘；确保数据集中不同图像的语义干扰度适中。

### 7. 总结
*   **核心思想**：通过对比数据构建与空间奖励，强化多图归因与恒常性。
*   **速记版pipeline**：
    1.  将单图接地数据扩充合成对比多图数据。
    2.  模型先进行推理思考。
    3.  输出结构化定位预测。
    4.  通过规则奖励强化图像归因正确性。

**Key Findings:**

- We propose Compositional Grounded Contrast (abbr.
- Experiments show that CGC achieves state-of-the-art results on fine-grained multi-image benchmarks, including MIG-Bench and VLM2-Bench.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.22498v1)
- [arXiv](https://arxiv.org/abs/2604.22498v1)

---

