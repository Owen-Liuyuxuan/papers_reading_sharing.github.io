time: 20260313

# Arxiv Computer Vision Papers - 2026-03-13

## Executive Summary

### **Arxiv 计算机视觉领域日报执行摘要 (2026-03-12)**

**1. 核心主题与趋势观察**

今日的论文集清晰地反映了计算机视觉领域的几个强劲融合趋势：

*   **具身智能与机器人操作的深化**：超过三分之一的论文聚焦于此。研究重点正从单一的视觉感知或运动控制，转向**人形机器人（Humanoid）的全身协调操作（Loco-Manipulation）** 和**灵巧手（Dexterous Manipulation）** 的通用化、数据高效学习（如论文1, 2, 6）。目标是为机器人构建更通用、更接近人类能力的“基础模型”。
*   **视频理解的效率与实时性革命**：一个显著趋势是处理**连续视频流（Streaming Video）**，而非剪辑好的片段。多篇论文致力于开发能够**边看边思考（Think Simultaneously）**、进行**在线测试时训练（Test-Time Training）** 或利用**自回归凝视（Autoregressive Gazing）** 等高效注意力机制的模型（如论文4, 5, 8, 9），旨在实现低延迟、高能效的实时视频理解。
*   **多模态与组合推理的严谨化**：研究不仅追求多模态融合的“广度”，更开始强调推理的“深度”与“严谨性”。出现了通过**程序化验证（Programmatically Verified）** 来构建可靠评测基准的工作（论文3），这标志着该子领域向更可靠、可复现的方向发展。
*   **生成式技术的渗透与应用**：生成式AI技术继续作为强大工具，被应用于解决特定感知任务（如**视频深度估计**，论文10）和实现高度可控的**视频生成/定制**（论文7），展示了“生成即解决”的新范式。

**2. 重点与创新性论文亮点**

*   **最具野心的基础模型工作**：**论文1 ($Ψ_0$)** 提出面向“通用人形机器人全身操作”的开源基础模型。其定位具有高度战略意义，若实现所宣称的开放性与通用性，可能成为机器人学习领域的里程碑式基础设施。
*   **最具算法创新性的工作**：**论文9 (Attend Before Attention)** 提出了“自回归凝视”机制，挑战了传统Transformer在视频中全局计算注意力的范式。该方法有望极大提升长视频理解的效率，是底层架构层面的重要创新。
*   **最具工程与评测严谨性的工作**：**论文3 (MM-CondChain)** 通过程序化验证构建组合推理基准，直击当前多模态评测中数据泄露、评估不严的痛点。这项工作对于推动领域扎实进步具有关键价值。

**3. 新兴研究方向与技术**

*   **“流式”感知-决策范式**：以OmniStream（论文4）、Video Streaming Thinking（论文5）、Spatial-TTT（论文8）为代表，**在线、连续、非稳态的数据流处理**成为前沿焦点。与之配套的**测试时训练（TTT）**、**状态记忆**和**增量学习**技术重要性凸显。
*   **机器人学习的“基础模型”路径**：论文1和论文6共同指向一个方向：借鉴大语言模型的成功经验，构建**数据驱动、可扩展、能执行多种机器人任务**的预训练模型。这正在成为解决机器人泛化能力难题的主流技术路径。
*   **生成式先验用于感知任务**：如论文10 (DVD) 所示，利用**扩散模型等生成式先验**来提升传统视觉任务（深度估计、三维重建等）的鲁棒性和细节质量，是一个值得关注的技术交叉点。

**4. 全文阅读建议优先级**

*   **必读 (领域风向标)**：
    *   **论文1 ($Ψ_0$)**：了解机器人基础模型的最新进展与愿景。
    *   **论文9 (Attend Before Attention)**：学习视频理解在模型效率上的突破性思路。
*   **强烈推荐 (细分领域关键进展)**：
    *   **论文3 (MM-CondChain)**：关注多模态推理严谨化评测的方法。
    *   **论文4 (OmniStream) 或 论文5 (Video Streaming Thinking)**：任选其一，掌握流式视频理解的核心框架。
    *   **论文2 (Contact Coverage-Guided Exploration)**：对机器人强化学习感兴趣者必读，提供了高效的探索策略。
*   **值得浏览 (应用与技术创新)**：
    *   **论文7 (DreamVideo-Omni)**：了解多主体、运动可控视频生成的最新技巧。
    *   **论文10 (DVD)**：关注生成式模型如何革新经典视觉任务。

**总结**：今日论文显示，计算机视觉的核心驱动力正从“静态感知”加速迈向**动态交互（机器人）**、**连续理解（视频流）** 和**可靠推理（验证基准）**。研究的前沿在于构建能够实时处理复杂物理世界信息、并做出有效行动的智能系统。建议优先关注机器人基础模型与高效视频理解架构的相关工作。

---

## Table of Contents

1. [$Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation](#2603.12263v1)
2. [Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation](#2603.10971v1)
3. [MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](#2603.12266v1)
4. [OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams](#2603.12265v1)
5. [Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](#2603.12262v1)
6. [HumDex:Humanoid Dexterous Manipulation Made Easy](#2603.12260v1)
7. [DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](#2603.12257v1)
8. [Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](#2603.12255v1)
9. [Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing](#2603.12254v1)
10. [DVD: Deterministic Video Depth Estimation with Generative Priors](#2603.12250v1)

---

## Papers

<a id='2603.12263v1'></a>
## [$Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation](https://arxiv.org/abs/2603.12263v1)

**Authors:** Songlin Wei, Hongyi Jing, Boqian Li, Zhenyu Zhao, Jiageng Mao, Zhenhao Ni, Sicheng He, Jie Liu, Xiawei Liu, Kaidi Kang, Sheng Zang, Weiduo Yuan, Marco Pavone, Di Huang, Yue Wang

**Published:** 2026-03-12

**Categories:** cs.RO

**Abstract:**

We introduce $Ψ_0$ (Psi-Zero), an open foundation model to address challenging humanoid loco-manipulation tasks. While existing approaches often attempt to address this fundamental problem by co-training on large and diverse human and humanoid data, we argue that this strategy is suboptimal due to the fundamental kinematic and motion disparities between humans and humanoid robots. Therefore, data efficiency and model performance remain unsatisfactory despite the considerable data volume. To address this challenge, \ours\;decouples the learning process to maximize the utility of heterogeneous data sources. Specifically, we propose a staged training paradigm with different learning objectives: First, we autoregressively pre-train a VLM backbone on large-scale egocentric human videos to acquire generalizable visual-action representations. Then, we post-train a flow-based action expert on high-quality humanoid robot data to learn precise robot joint control. Our research further identifies a critical yet often overlooked data recipe: in contrast to approaches that scale with noisy Internet clips or heterogeneous cross-embodiment robot datasets, we demonstrate that pre-training on high-quality egocentric human manipulation data followed by post-training on domain-specific real-world humanoid trajectories yields superior performance. Extensive real-world experiments demonstrate that \ours\ achieves the best performance using only about 800 hours of human video data and 30 hours of real-world robot data, outperforming baselines pre-trained on more than 10$\times$ as much data by over 40\% in overall success rate across multiple tasks. We will open-source the entire ecosystem to the community, including a data processing and training pipeline, a humanoid foundation model, and a real-time action inference engine.

**Analysis:**

### 1. 摘要翻译
本文介绍了 $\Psi_0$ (Psi-Zero)，这是一个旨在解决高难度人形机器人移动-操作（loco-manipulation）任务的开放基础模型。现有方法通常尝试通过在庞大且多样的人类和机器人数据上进行联合训练来解决这一根本性问题，但我们认为该策略因人类与人形机器人之间存在基础性的运动学和动作差异，导致效果次优，且在数据利用效率上表现欠佳。为了克服这一挑战，$\Psi_0$ 将学习过程解耦，以最大化异构数据源的效用。具体而言，我们提出了一个具有不同学习目标的分阶段训练范式：首先，在人类第一人称视角（egocentric）大规模视频上自回归预训练视觉语言模型（VLM）骨干，以获取可泛化的视觉-动作表征；随后，在高质量人形机器人数据上对基于流（flow-based）的动作专家进行后训练，以学习精确的机器人关节控制。我们的研究识别出了一个关键但常被忽视的数据配方：与扩展大规模互联网片段或异构跨本体机器人数据集的方法相比，在高质量人类操作数据上进行预训练，随后在领域特定的人形机器人轨迹上进行后训练，能带来更优异的性能。广泛的现实世界实验表明，$\Psi_0$ 仅利用约 800 小时的人类视频数据和 30 小时的真实机器人数据，就取得了最佳性能，在多项任务中将整体成功率较基线提高了 40% 以上，且基线所使用的训练数据量是我们的 10 倍以上。我们将开源整个生态系统，包括数据处理和训练流水线、人形机器人基础模型以及实时动作推理引擎。

### 2. 方法动机分析
*   **驱动力**：在人形机器人上实现复杂的全身操作（loco-manipulation），但受限于高昂的机器人遥操作数据收集成本。
*   **现有痛点**：端到端的联合训练方法（将人类视频与机器人数据混杂）在处理“存在明显动作学差异”的异构数据时，模型容易产生次优的动作分布；且单纯扩展数据量（Scaling Law）效率极低。
*   **研究假设**：通过“解耦”学习范式，先利用廉价的视频数据学习任务语义和视觉先验，再利用高质量的机器人轨迹学习精细的关节级控制，能以极高的数据效率实现更好的泛化。

### 3. 方法设计详解
*   **流程总结**：
    1.  **VLM 预训练**：在 829 小时人类视频上进行自回归动作预测，输出任务空间的动作（Fast tokenizer处理为离散Token）。目的是获得任务语义和视觉表征。
    2.  **动作专家后训练**：冻结 VLM，在 30 小时机器人数据上训练一个基于流匹配（Flow Matching）的 MM-DiT（多模态扩散 Transformer）。直接预测关节空间的动作序列。
    3.  **微调与推理**：针对特定任务微调动作专家。推理时使用实时动作分块（RTC）处理推理延迟。
*   **核心结构**：采用三系统架构：(System 2) VLM作为视觉语言理解核心；(System 1) MM-DiT 作为动作专家；(System 0) 下层控制器，负责稳健的下肢轨迹跟踪。
*   **关键公式**：$L_{fm} = \mathbb{E} [\|v^{flow}_{\rho}(z_t, a^{\tau}_t, \tau) - (\epsilon - a_t)\|]$。这是一种基于流匹配的训练，不同于传统的基于扩散的方法，通过流（flow）将噪声分布变换为动作分布，提升了计算效率和动作生成质量。

### 4. 方法对比分析
*   **本质区别**：明确否定了“一刀切”的联合训练，提出“先通用表征，后专业运动控制”的解耦架构。
*   **创新点**：
    1.  **MM-DiT架构**：利用FiLM调制技术，实现视觉语言特征对动作空间的高效调制。
    2.  **训练时实时分块 (RTC)**：通过掩码动作Token强制模型训练时模拟推理延迟，解决了“停-思-行”导致的抖动问题。
*   **最佳场景**：高自由度、需要长程规划与精细操作的人形机器人任务。

### 5. 实验分析
*   **关键结论**：在仅使用 1/10 数据量的情况下，整体任务成功率比现有最先进的人形基础模型（如 GR00T）高出 40% 以上。
*   **优势**：极高的数据利用效率，动作输出极其平稳，全身协调性强。
*   **局限**：模型本身较重（2.5B参数），对于算力较小的边缘计算设备仍具挑战。

### 6. 实用指南
*   **开源情况**：已开源模型权重、训练流水线及代码（https://psi-lab.ai/Psi0）。
*   **关键点**：RTC的实现至关重要。训练时需通过 `uniform(0, dmax)` 随机掩码来模拟推理延迟，这比单纯增加模型训练规模更能解决实际落地时的抖动。
*   **迁移建议**：若想在不同机器人本体上应用，只需替换 System 0 的下层控制器，并微调 System 1 的输出层即可。

### 7. 总结
*   **核心思想**：通过解耦策略，利用大规模人类视频数据实现高效率全身控制。
*   **速记版pipeline**：
    1. 大规模视频预训练VLM获取语义；
    2. 基于流匹配在少量机器人数据上微调动作头；
    3. 训练时引入RTC模拟延迟，确保推理平稳；
    4. 对特定任务执行轻量级微调。

**Key Findings:**

- We introduce $Ψ_0$ (Psi-Zero), an open foundation model to address challenging humanoid loco-manipulation tasks.
- Specifically, we propose a staged training paradigm with different learning objectives: First, we autoregressively pre-train a VLM backbone on large-scale egocentric human videos to acquire generalizable visual-action representations.
- Our research further identifies a critical yet often overlooked data recipe: in contrast to approaches that scale with noisy Internet clips or heterogeneous cross-embodiment robot datasets, we demonstrate that pre-training on high-quality egocentric human manipulation data followed by post-training on domain-specific real-world humanoid trajectories yields superior performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12263v1)
- [arXiv](https://arxiv.org/abs/2603.12263v1)

---

<a id='2603.10971v1'></a>
## [Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation](https://arxiv.org/abs/2603.10971v1)

**Authors:** Zixuan Liu, Ruoyi Qiao, Chenrui Tie, Xuanwei Liu, Yunfan Lou, Chongkai Gao, Zhixuan Xu, Lin Shao

**Published:** 2026-03-11

**Categories:** cs.RO, cs.AI

**Abstract:**

Deep Reinforcement learning (DRL) has achieved remarkable success in domains with well-defined reward structures, such as Atari games and locomotion. In contrast, dexterous manipulation lacks general-purpose reward formulations and typically depends on task-specific, handcrafted priors to guide hand-object interactions. We propose Contact Coverage-Guided Exploration (CCGE), a general exploration method designed for general-purpose dexterous manipulation tasks. CCGE represents contact state as the intersection between object surface points and predefined hand keypoints, encouraging dexterous hands to discover diverse and novel contact patterns, namely which fingers contact which object regions. It maintains a contact counter conditioned on discretized object states obtained via learned hash codes, capturing how frequently each finger interacts with different object regions. This counter is leveraged in two complementary ways: (1) to assign a count-based contact coverage reward that promotes exploration of novel contact patterns, and (2) an energy-based reaching reward that guides the agent toward under-explored contact regions. We evaluate CCGE on a diverse set of dexterous manipulation tasks, including cluttered object singulation, constrained object retrieval, in-hand reorientation, and bimanual manipulation. Experimental results show that CCGE substantially improves training efficiency and success rates over existing exploration methods, and that the contact patterns learned with CCGE transfer robustly to real-world robotic systems. Project page is https://contact-coverage-guided-exploration.github.io.

**Analysis:**

### 1. 摘要翻译
深度强化学习（DRL）在定义明确的奖励结构领域（如Atari游戏、运动控制）取得了显著成功。然而，灵巧操作（dexterous manipulation）缺乏通用的奖励公式，通常依赖于手写任务先验。为此，我们提出了**接触覆盖引导探索（CCGE）**，这是一种专为通用灵巧操作任务设计的探索方法。CCGE将接触状态定义为物体表面点与预定义手部关键点之间的交集，鼓励灵巧手发现多样且新颖的接触模式（即手指与物体区域的配对）。它通过哈希编码离散化物体状态，维护一个条件化的接触计数器，用于记录手指与特定区域的交互频率。该计数器产生两种互补信号：(1) 基于计数的接触覆盖奖励，促进对新颖接触模式的探索；(2) 基于能量的到达奖励，引导代理向未充分探索的接触区域移动。我们在包括杂乱物体分离、受限物体检索、手内重定向和双臂操作在内的多种任务中评估了CCGE。实验结果表明，CCGE显著提升了训练效率和成功率，且学习到的接触策略能稳健迁移至现实世界。

---

### 2. 方法动机分析
*   **驱动力**：灵巧操作的核心是接触，但现有的DRL探索方法要么缺乏对“接触”这一物理特性的显式建模（仅关注状态/动态新颖性），要么使用极度不稳定的力反馈作为信号。CCGE旨在构建一种任务无关的、物理感知的通用探索奖励，替代繁琐的手写任务先验。
*   **现有方法痛点**：
    *   **通用探索不足**：追求泛化的状态空间访问常导致无关行为（如乱挥舞手臂）。
    *   **信号不稳定**：直接预测接触力极易受震荡影响，导致不稳定的优化方向。
    *   **任务配置干扰**：全局探索目标在不同任务阶段会相互冲突（跨状态干扰）。
*   **研究假设**：通过“空间+手指”的离散化接触状态建模，并辅以预接触（引导）与后接触（奖励）的双重机制，能有效解决灵巧操作中的探索稀疏与低效问题。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **状态离散化（Hashing）**：利用自编码器将物体状态压缩为D维二进制向量，通过SimHash映射为离散状态索引，解决不同任务配置下的跨状态干扰。
    2.  **接触匹配（ContactMatch）**：检测手指与物体点云的最小距离与接触力，过滤虚假接触，确定手指与物体表面区域的交互。
    3.  **接触计数器更新**：针对每个离散状态索引，维护一个 $S \times F \times K$ 的三维计数器，记录手指 $F$ 在状态 $S$ 下接触区域 $K$ 的频率。
    4.  **双重奖励计算**：
        *   **接触覆盖奖励（$R_{\text{contact}}$）**：对罕见的“手指-区域”接触给予高奖励，促使动作多样化。
        *   **能量到达奖励（$R_{\text{energy}}$）**：基于接触区域的覆盖计数，利用势能场（指数衰减）引导手部提前向未探索的区域移动。
*   **模型结构**：包含状态哈希自编码器、接触计数器、PPO策略网络。
*   **关键公式意义**：$g(c) = 1/\sqrt{c+1}$，随接触频率增加奖励递减，实现探索动机的自动调节。

---

### 4. 方法对比分析
*   **本质区别**：CCGE不再关注“我是否走过这个状态”，而是关注“我是否以**特定的方式**接触到了这个物体的特定部分”。
*   **创新贡献**：提出了一种将“接触拓扑结构”量化为探索奖励的新视角，并通过双层奖励（预接触到达+后接触覆盖）解决了稀疏接触问题。
*   **适用场景**：适用于所有高自由度、复杂接触、需要精细交互的机器人操控任务。

---

### 5. 实验分析
*   **验证方法**：在4种复杂仿真任务中与TR（仅任务奖励）、LHCC（状态哈希）、HaC（触觉好奇心）、RND-Dist（距离好奇心）进行对比。
*   **关键结果**：在“受限物体检索”等极端任务中，其他基线基本失效（0%成功率），而CCGE达到88%成功率，且所有任务下的样本效率显著提升2-3倍。
*   **局限**：模型依赖于物体点云的表示，对遮挡处理要求较高，且现阶段主要验证在固定任务环境下。

---

### 6. 实用指南
*   **实现细节**：
    *   $\lambda$（正则化权重）和 $H$（哈希长度）是核心超参数，需要平衡状态细粒度与样本效率。
    *   接触探测需结合距离阈值（$\delta_{\text{dist}}$）和力阈值（$\delta_{\text{force}}$），防止仿真噪声导致计数器爆表。
*   **迁移建议**：该方法逻辑上可直接迁移至其他高自由度机械手（如Allegro Hand），无需改动架构，只需微调 $g(c)$ 中的缩放系数。

---

### 7. 总结
*   **核心思想**：通过记录离散状态下的手指接触拓扑实现高效探索。
*   **速记版pipeline**：
    1.  哈希编码：将当前物体位置映射为特定类别的“状态 ID”；
    2.  接触检测：识别哪些手指碰到了物体的哪些表面区域；
    3.  计数更新：将接触信息记录到对应 ID 的计数器中；
    4.  动态奖励：奖励那些“从未被触发过的”手指-区域接触，并引导手部前往这些区域。

**Key Findings:**

- We propose Contact Coverage-Guided Exploration (CCGE), a general exploration method designed for general-purpose dexterous manipulation tasks.
- CCGE represents contact state as the intersection between object surface points and predefined hand keypoints, encouraging dexterous hands to discover diverse and novel contact patterns, namely which fingers contact which object regions.
- This counter is leveraged in two complementary ways: (1) to assign a count-based contact coverage reward that promotes exploration of novel contact patterns, and (2) an energy-based reaching reward that guides the agent toward under-explored contact regions.
- Experimental results show that CCGE substantially improves training efficiency and success rates over existing exploration methods, and that the contact patterns learned with CCGE transfer robustly to real-world robotic systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10971v1)
- [arXiv](https://arxiv.org/abs/2603.10971v1)

---

<a id='2603.12266v1'></a>
## [MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](https://arxiv.org/abs/2603.12266v1)

**Authors:** Haozhan Shen, Shilin Yan, Hongwei Xue, Shuaiqi Lu, Xiaojun Tang, Guannan Zhang, Tiancheng Zhao, Jianwei Yin

**Published:** 2026-03-12

**Categories:** cs.CV

**Abstract:**

Multimodal Large Language Models (MLLMs) are increasingly used to carry out visual workflows such as navigating GUIs, where the next step depends on verified visual compositional conditions (e.g., "if a permission dialog appears and the color of the interface is green, click Allow") and the process may branch or terminate early. Yet this capability remains under-evaluated: existing benchmarks focus on shallow-compositions or independent-constraints rather than deeply chained compositional conditionals. In this paper, we introduce MM-CondChain, a benchmark for visually grounded deep compositional reasoning. Each benchmark instance is organized as a multi-layer reasoning chain, where every layer contains a non-trivial compositional condition grounded in visual evidence and built from multiple objects, attributes, or relations. To answer correctly, an MLLM must perceive the image in detail, reason over multiple visual elements at each step, and follow the resulting execution path to the final outcome. To scalably construct such workflow-style data, we propose an agentic synthesis pipeline: a Planner orchestrates layer-by-layer generation of compositional conditions, while a Verifiable Programmatic Intermediate Representation (VPIR) ensures each layer's condition is mechanically verifiable. A Composer then assembles these verified layers into complete instructions. Using this pipeline, we construct benchmarks across three visual domains: natural images, data charts, and GUI trajectories. Experiments on a range of MLLMs show that even the strongest model attains only 53.33 Path F1, with sharp drops on hard negatives and as depth or predicate complexity grows, confirming that deep compositional reasoning remains a fundamental challenge.

**Analysis:**

### 1. 摘要翻译
多模态大语言模型（MLLMs）正越来越多地用于执行如导航图形用户界面（GUI）等视觉工作流，其中下一步操作依赖于经过验证的视觉组合条件（例如：“如果出现权限对话框且界面颜色为绿色，点击允许”），且过程可能发生分支或提前终止。然而，这种能力目前评估不足：现有基准主要关注浅层组合或独立约束，而非深度链式组合条件。本文提出了 MM-CondChain，一个用于视觉接地深度组合推理的基准。每个基准实例被组织为一个多层推理链，每一层都包含一个由视觉证据支撑的非平凡组合条件，并由多个对象、属性或关系构建。为了正确回答，MLLM 必须详细感知图像，在每一步对多个视觉元素进行推理，并遵循由此产生的执行路径直至最终结果。为了可扩展地构建此类工作流风格的数据，我们提出了一个代理合成流水线：规划器协调逐层生成组合条件，而可验证程序化中间表示（VPIR）确保每一层的条件是机械可验证的。随后，一个组合器将这些经过验证的层组装成完整的指令。利用此流水线，我们在自然图像、数据图表和 GUI 轨迹这三个视觉领域构建了基准。对一系列 MLLM 的实验表明，即使是最强的模型也仅达到 53.33 的路径 F1 分数，在困难负样本、推理深度或谓词复杂度增加时，性能出现急剧下降，这证实了深度组合推理仍然是一个根本性的挑战。

### 2. 方法动机分析
*   **驱动力**：旨在填补当前 MLLM 在处理长序列、多分支、高度依赖视觉确认的工作流推理能力的评估空白。
*   **现有方法痛点**：现有数据集多关注单层、简单的属性描述（如“对象是红色的吗？”），或互不依赖的指令集合，缺乏对“如果-那么”式链式逻辑和硬负样本（Hard Negatives）的系统性测试，导致模型往往通过猜测而非真正的视觉逻辑推理完成任务。
*   **研究假设**：通过将推理逻辑分解为可验证的程序化中间表示（VPIR），可以将推理的正确性与自然语言的生成解耦，从而构建逻辑严密、可机械验证的深度组合推理基准。

### 3. 方法设计详解
*   **流程总结（VPIR 合成流水线）**：
    1.  **关系策略与选择**：Planner 在每一步确定推理方向（深化 vs. 转换），提取相关的视觉事实 $F_t$。
    2.  **VPIR 生成**：将提取的事实转化为可执行的 Python 谓词（即 VPIR），通过 sandbox 环境确认其真值为 1（True Logic），并生成相应的假逻辑（False Logic）谓词，确保该步骤的“硬负样本”性质。
    3.  **逻辑渲染**：将 verified 后的 VPIR 谓词翻译为自然语言。
    4.  **组合与编译**：Composer 将生成的 $T$ 个层级组装，通过替换单一逻辑谓词构建“False-path”实例，确保两者的表面形式极度相似但执行路径不同。
*   **模型结构**：包含 Planner（负责流程控制）、Verifier（负责事实与逻辑验证）、Translator（负责语言渲染）和 Composer（负责最终实例编译）。
*   **算法解释**：核心在于 $p_t(F_t) = 1$ 与 $\tilde{p}_t(F_t) = 0$ 的机械约束，保证基准测试具有确定性的 ground truth，消除了“模型判别模型（LLM-as-judge）”带来的主观偏差。

### 4. 方法对比分析
*   **本质区别**：从传统的“结果导向评估”转变为“过程逻辑严密性评估”。
*   **创新贡献**：
    *   **VPIR**：首次引入可机械验证的中间逻辑表示，实现了从事实到语言的解耦，从源头上杜绝了数据逻辑冲突。
    *   **链式硬负样本**：通过“局部谓词替换”而非“全局指令修改”来制造极具挑战性的负样本，强迫模型进行精细化推理。
*   **适用场景**：适用于评估智能体（Agents）在 GUI 自动化、复杂图表分析和多步规划任务中的逻辑一致性与视觉感知精度。

### 5. 实验分析
*   **关键结论**：最先进的模型（Gemini-3-Pro）在深度组合推理任务上的表现仅略高于 50%，显著低于其在简单基准上的表现。
*   **优势**：具有极高的诊断价值，能够明确模型在“推理深度”和“逻辑复杂度”哪一方面失效。
*   **局限**：模型在 True-path 上表现远好于 False-path，揭示了模型普遍存在的“持续倾向性（Continue bias）”，即在复杂条件下倾向于假设所有条件均满足。

### 6. 实用指南
*   **开源情况**：已开源，访问 GitHub 仓库 `Accio-Lab/MM-CondChain` 获取数据与工具。
*   **实现细节**：Pipeline 依赖 Gemini-3-Pro 进行事实提取与渲染，需注意不同领域（自然、图表、GUI）的预处理适配器构建。
*   **迁移可能**：该框架的 VPIR 抽象可轻松迁移至任何包含“可提取事实”的任务场景（如视频理解、复杂文档解析）。

### 7. 总结
*   **核心思想**：通过程序化逻辑约束，构建可验证的深度视觉链式推理基准。
*   **速记版 pipeline**：
    1. 提取视觉事实；
    2. 生成可执行代码（逻辑真值验证）；
    3. 翻译为自然语言；
    4. 植入硬负样本；
    5. 组装多层链式指令。

**Key Findings:**

- In this paper, we introduce MM-CondChain, a benchmark for visually grounded deep compositional reasoning.
- To scalably construct such workflow-style data, we propose an agentic synthesis pipeline: a Planner orchestrates layer-by-layer generation of compositional conditions, while a Verifiable Programmatic Intermediate Representation (VPIR) ensures each layer's condition is mechanically verifiable.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12266v1)
- [arXiv](https://arxiv.org/abs/2603.12266v1)

---

<a id='2603.12265v1'></a>
## [OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams](https://arxiv.org/abs/2603.12265v1)

**Authors:** Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, Weidi Xie

**Published:** 2026-03-12

**Categories:** cs.CV

**Abstract:**

Modern visual agents require representations that are general, causal, and physically structured to operate in real-time streaming environments. However, current vision foundation models remain fragmented, specializing narrowly in image semantic perception, offline temporal modeling, or spatial geometry. This paper introduces OmniStream, a unified streaming visual backbone that effectively perceives, reconstructs, and acts from diverse visual inputs. By incorporating causal spatiotemporal attention and 3D rotary positional embeddings (3D-RoPE), our model supports efficient, frame-by-frame online processing of video streams via a persistent KV-cache. We pre-train OmniStream using a synergistic multi-task framework coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment on 29 datasets. Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training). Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

**Analysis:**

这是一份关于 **OmniStream** 论文的深度分析报告：

### 1. 摘要翻译
现代视觉智能体需具备通用、因果性和物理结构感知的实时流式处理能力。然而，现有视觉基础模型在语义感知、离线时序建模或空间几何领域往往碎片化。本文提出 **OmniStream**，一个统一的流式视觉骨干网络，能有效处理多样化视觉输入并进行感知、重建和行动。通过引入“因果时空注意力机制”和“3D旋转位置编码（3D-RoPE）”，该模型利用持久化 KV-cache 支持高效的帧级在线处理。我们在 29 个数据集上预训练了 OmniStream，耦合了静态/时序表征学习、流式几何重建和视觉-语言对齐。实验表明，即便在固定骨干网络参数的情况下，OmniStream 在图像/视频探测、流式几何重建、复杂推理及机器人操作任务上均表现出极强的竞争力。

### 2. 方法动机分析
*   **驱动力**：旨在打破视觉任务（感知、推理、动作）的碎片化，构建一个单一、通用且具备物理理解能力的流式视觉处理骨干。
*   **痛点**：现有主流视觉模型要么仅限于静态语义（如 DINO），要么依赖非因果、离线处理（如 VideoMAE），且缺乏对三维物理结构的深度集成，导致在实时流式场景（如机器人控制）中表现受限。
*   **研究假设**：通过在单一 ViT 骨干中显式注入因果时空依赖、3D 空间约束以及视觉语言多模态监督，能够学习到一种兼顾静态语义、动态运动及几何结构的通用表达。

### 3. 方法设计详解
*   **模型结构**：基于 DINOv3 架构，核心创新在于：
    1.  **因果时空注意力 (Causal Spatiotemporal Attention)**：强制因果掩码，确保当前帧仅能访问历史信息，并利用 KV-cache 消除重复计算。
    2.  **3D-RoPE**：将 2D 旋转位置编码扩展至 3D 空间，采用 (t, y, x) 的 2:3:3 分割策略，赋予模型理解“何时何地”的物理空间能力。
*   **训练目标 (协同优化)**：
    *   **静态/时序表征学习 ($\mathcal{L}_{ssl}$)**：通过 DINOv3 风格的蒸馏学习语义一致性。
    *   **流式几何重建 ($\mathcal{L}_{geo}$)**：通过深度头、射线头和相机头，显式预测深度图和位姿，强制骨干网理解 3D 结构。
    *   **视觉语言对齐 ($\mathcal{L}_{cap}$)**：通过轻量级 decoder，利用 OCR、字幕描述等任务进行语义 grounding。

### 4. 方法对比分析
*   **本质区别**：OmniStream 并非通过大模型接口统一输出，而是通过**表征层面的统一**，让骨干网络在推理阶段即具备空间与时序的物理理解力。
*   **创新贡献**：提出流式环境下的高效 KV-cache 机制与兼顾几何动态的 3D-RoPE，实现了在“冻结骨干”设置下的超强零样本迁移能力。

### 5. 实验分析
*   **验证方法**：在“冻结骨干”的严苛设定下，在 Perception、Reasoning、Action 三大类别下进行基准测试。
*   **核心结论**：在 SSv2 等动态数据集上显著优于 DINOv3，在机器人操纵（CALVIN/SIMPLER-ENV）任务中实现零样本迁移，无需微调即可超越许多专有模型。
*   **局限**：对超大规模数据集的算力依赖较高，目前主要在特定模拟环境中验证，真实世界复杂非结构化环境的稳健性有待后续扩容。

### 6. 实用指南
*   **开源情况**：代码已发布 (https://github.com/Go2Heart/OmniStream)。
*   **关键点**：注意其 3D-RoPE 的 (t, y, x) 通道切分方式；预训练阶段需使用梯度累积，并在多任务间平衡损失函数权重 ($\lambda_{ssl}=0.1, \lambda_{geo}=1, \lambda_{cap}=1$)。

### 7. 总结
*   **核心思想**：通过多任务协同训练与因果架构，赋予静态视觉骨干物理空间记忆能力。
*   **速记版 Pipeline**：
    1.  输入流式视频帧；
    2.  添加 3D-RoPE 编码时空信息；
    3.  通过因果注意力结合 KV-cache 缓存历史；
    4.  多任务头输出深度、位姿与语义令牌；
    5.  冻结骨干直接接入下游决策或推理模型。

**Key Findings:**

- Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training).
- Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12265v1)
- [arXiv](https://arxiv.org/abs/2603.12265v1)

---

<a id='2603.12262v1'></a>
## [Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](https://arxiv.org/abs/2603.12262v1)

**Authors:** Yiran Guan, Liang Yin, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Jian Luan, Yuliang Liu, Xiang Bai

**Published:** 2026-03-12

**Categories:** cs.CV

**Abstract:**

Online Video Large Language Models (VideoLLMs) play a critical role in supporting responsive, real-time interaction. Existing methods focus on streaming perception, lacking a synchronized logical reasoning stream. However, directly applying test-time scaling methods incurs unacceptable response latency. To address this trade-off, we propose Video Streaming Thinking (VST), a novel paradigm for streaming video understanding. It supports a thinking while watching mechanism, which activates reasoning over incoming video clips during streaming. This design improves timely comprehension and coherent cognition while preserving real-time responsiveness by amortizing LLM reasoning latency over video playback. Furthermore, we introduce a comprehensive post-training pipeline that integrates VST-SFT, which structurally adapts the offline VideoLLM to causal streaming reasoning, and VST-RL, which provides end-to-end improvement through self-exploration in a multi-turn video interaction environment. Additionally, we devise an automated training-data synthesis pipeline that uses video knowledge graphs to generate high-quality streaming QA pairs, with an entity-relation grounded streaming Chain-of-Thought to enforce multi-evidence reasoning and sustained attention to the video stream. Extensive evaluations show that VST-7B performs strongly on online benchmarks, e.g. 79.5% on StreamingBench and 59.3% on OVO-Bench. Meanwhile, VST remains competitive on offline long-form or reasoning benchmarks. Compared with Video-R1, VST responds 15.7 times faster and achieves +5.4% improvement on VideoHolmes, demonstrating higher efficiency and strong generalization across diverse video understanding tasks. Code, data, and models will be released at https://github.com/1ranGuan/VST.

**Analysis:**

以下是对论文《Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously》的深入分析：

### 1. 摘要翻译
在线视频大语言模型（VideoLLMs）在支持响应式、实时交互方面至关重要。现有方法专注于流式感知，但缺乏同步的逻辑推理流，而直接应用推理扩展方法会导致无法接受的响应延迟。为此，我们提出了“视频流式思维”（Video Streaming Thinking, VST），这是一种全新的流式视频理解范式。它支持“边看边想”机制，在流式传输过程中激活对传入视频片段的推理。该设计通过将LLM的推理延迟分摊到视频播放过程中，在保持实时响应的同时提升了理解力和认知连贯性。我们还引入了包括VST-SFT和VST-RL在内的全套后训练流水线，以及一种利用视频知识图谱生成高质量流式QA对的自动化数据合成方案。实验表明，VST-7B在StreamingBench和OVO-Bench上表现优异，且相比Video-R1响应速度提升15.7倍。

### 2. 方法动机分析
- **驱动力**：解决在线视频理解中“深度推理”与“实时响应”之间的天然冲突。
- **现有痛点**：传统流式VideoLLM仅处理视觉感知，若要在查询时加入思维链（CoT），推理延迟极高；若不推理，则缺乏复杂逻辑分析能力。
- **研究假设**：通过在视频流流入的“空窗期”进行预推理（即边看边想），将高昂的推理计算分摊到时间轴上，从而实现零延迟的最终回答。

### 3. 方法设计详解
- **流式思维机制**：将视频流划分为片段，每个片段通过LLM产生“流式思维（Streaming Thought）”。
  - **输入**：当前片段 + 历史文本记忆（Long-term Textual Memory）。
  - **过程**：LLM对当前片段进行 autoregressive 生成，将视觉语义压缩为文本记忆。
  - **记忆更新**：采用FIFO（先进先出）策略更新长文本记忆，保证有限上下文窗口下的信息连续性。
- **双阶段训练**：
  - **VST-SFT**：通过有监督微调，教会模型在固定视觉窗口和历史文本约束下进行因果推理，对齐“边看边想”的协议。
  - **VST-RL**：通过强化学习优化策略，利用最终答案的准确性作为奖励，鼓励模型生成更具逻辑性的中间思考。
- **自动化数据合成**：使用Gemini 3.0构建视频知识图谱，通过DFS（深度优先搜索）采样证据链，引导模型生成包含中间推理逻辑的训练数据。

### 4. 方法对比分析
- **本质区别**：从“被动响应”转变为“主动预测”，将推理工作量前置，而非查询后的集中式处理。
- **创新点**：引入流式因果关注机制（Streaming Attention Mask）和基于知识图谱的证据链采样，实现了高效的思维分摊。
- **适用场景**：实时视频分析、长视频实时问答、需要多步推理的在线交互场景。

### 5. 实验分析
- **关键结果**：VST-7B在StreamingBench上达到79.5%的准确率，且QA延迟极低（对比Video-R1的9.53s，VST仅需0.51s）。
- **优势**：显著降低了推理响应延迟，同时提升了长视频理解中的复杂推理能力。
- **局限**：模型在思维步数超过4步后性能趋于饱和；且额外的LLM推理会增加token消耗。

### 6. 实用指南
- **开源情况**：已发布代码、数据与模型（https://github.com/1ranGuan/VST）。
- **关键细节**：
  - **超参数**：训练集包含100K推理样本；推理时限定最大思维步数为4。
  - **实现迁移**：该范式可迁移至其他长文本/流式视频LLM中，关键在于构建“流式数据合成”流水线。
- **部署建议**：采用FSDP并行处理以加速预推理阶段，利用vLLM进行高效推理。

### 7. 总结
- **核心思想**：利用视频流入的等待间隔，通过分段式主动思维实现零延迟推理。
- **速记版Pipeline**：
  1. **实体抽取**：将视频流实时转化为知识图谱。
  2. **边看边想**：在每个视频片段到来时，生成概括性的思维文本。
  3. **记忆更新**：将思维存入FIFO结构的文本库。
  4. **最终查询**：基于累积的文本思维与当前视觉，直接输出最终答案。

**Key Findings:**

- To address this trade-off, we propose Video Streaming Thinking (VST), a novel paradigm for streaming video understanding.
- Furthermore, we introduce a comprehensive post-training pipeline that integrates VST-SFT, which structurally adapts the offline VideoLLM to causal streaming reasoning, and VST-RL, which provides end-to-end improvement through self-exploration in a multi-turn video interaction environment.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12262v1)
- [arXiv](https://arxiv.org/abs/2603.12262v1)

---

<a id='2603.12260v1'></a>
## [HumDex:Humanoid Dexterous Manipulation Made Easy](https://arxiv.org/abs/2603.12260v1)

**Authors:** Liang Heng, Yihe Tang, Jiajun Xu, Henghui Bao, Di Huang, Yue Wang

**Published:** 2026-03-12

**Categories:** cs.RO

**Abstract:**

This paper investigates humanoid whole-body dexterous manipulation, where the efficient collection of high-quality demonstration data remains a central bottleneck.   Existing teleoperation systems often suffer from limited portability, occlusion, or insufficient precision, which hinders their applicability to complex whole-body tasks. To address these challenges, we introduce HumDex, a portable teleoperation system designed for humanoid whole-body dexterous manipulation. Our system leverages IMU-based motion tracking to address the portability-precision trade-off, enabling accurate full-body tracking while remaining easy to deploy. For dexterous hand control, we further introduce a learning-based retargeting method that generates smooth and natural hand motions without manual parameter tuning. Beyond teleoperation, HumDex enables efficient collection of human motion data. Building on this capability, we propose a two-stage imitation learning framework that first pre-trains on diverse human motion data to learn generalizable priors, and then fine-tunes on robot data to bridge the embodiment gap for precise execution. We demonstrate that this approach significantly improves generalization to new configurations, objects, and backgrounds with minimal data acquisition costs. The entire system is fully reproducible and open-sourced at https://github.com/physical-superintelligence-lab/HumDex.

**Analysis:**

以下是针对《HumDex: Humanoid Dexterous Manipulation Made Easy》的方法论深度分析：

### 1. 摘要翻译
本文研究了人形机器人全身灵巧操作任务，其中高质量演示数据的采集是核心瓶颈。现有遥操作系统往往因便携性受限、遮挡问题或精度不足，难以胜任复杂的全身任务。为此，我们引入了HumDex，一个专为人形机器人全身灵巧操作设计的便携式遥操作系统。该系统利用基于IMU的运动跟踪，在便携性与精度之间实现了平衡，既保证了全身追踪的准确性，又易于部署。针对灵巧手控制，我们进一步提出了一种基于学习的重定向方法，无需人工参数调整即可生成平滑自然的动作。除了遥操作，HumDex还实现了人类运动数据的高效采集。基于此，我们提出了一种两阶段模仿学习框架：首先在多样化的人类运动数据上进行预训练以学习通用先验，随后在机器人数据上进行微调以弥补具体形态差异（embodiment gap）。实验证明，该方法以极低的采集成本显著提升了模型对新位姿、物体及环境的泛化能力。

### 2. 方法动机分析
*   **驱动力**：解决灵巧操作数据采集的“高成本、低质量、难部署”三难困境。
*   **痛点**：基于视觉的系统（VR/摄像机）在工具交互时存在严重遮挡，导致灵巧手动作捕捉失效；固定基础设施（Mocap/外骨骼）限制了场景多样性。
*   **核心假设**：利用低成本、便携的IMU采集人类全身数据，通过两阶段训练（先人类先验，后机器人微调），能有效弥补人机形态差异，同时利用人类数据的多样性增强机器人泛化性能。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据采集**：穿戴式IMU设备（如SlimeVR）获取人体全身数据，手套获取手指灵巧动作。
    2.  **重定向策略**：
        *   **全身**：采用基于骨架的优化方案（GMR），以骨盆为基准消除IMU漂移，保证全身运动连贯。
        *   **灵巧手**：使用一个轻量化MLP，将手套测得的5个指尖3D位置直接映射为20自由度关节角度。
    3.  **两阶段模仿学习**：
        *   **阶段一（预训练）**：在人类数据上训练ACT（Action Chunking Transformer），利用人类动作作为先验特征。
        *   **阶段二（微调）**：在少量机器人遥操作数据上微调，使策略适应机器人特有的动力学和动作空间。

### 4. 方法对比分析
*   **本质区别**：从传统的“依赖机器人硬件采集”转向“利用便携式设备采集人类先验+小规模机器人微调”。
*   **创新贡献**：
    *   **学习型手部重定向**：替代了复杂的基于逆运动学（IK）的优化过程，实现了恒定时间推理。
    *   **两阶段泛化框架**：直接解决了人机形态差距导致的模仿学习收敛困难问题。
*   **适用场景**：复杂全身、长序列、涉及遮挡（如工具使用）的灵巧操作任务。

### 5. 实验分析
*   **关键结论**：在Scan&Pack等极具挑战性的任务上，HumDex实现了90%的成功率，而传统视觉遥操作因遮挡完全失败；在泛化测试中，通过引入多样化人类数据，策略成功率提升近2倍。
*   **优势**：极高的数据采集效率（比基线快26%）和优异的物体、环境泛化能力。
*   **局限**：对极度精细的力反馈缺乏支持，且受限于机器人的负载极限。

### 6. 实用指南
*   **开源情况**：已开源，GitHub地址：`https://github.com/physical-superintelligence-lab/HumDex`
*   **实现要点**：
    *   **数据对齐**：不必强求人机动作一一对齐，通过序列化训练（先Pre-train，后Fine-tune）规避梯度冲突。
    *   **手部重定向**：确保训练数据包含足够多样化的手指姿态。
*   **迁移建议**：对于其他具有复杂自由度的机器人（如多指手或人形躯干），可直接复用该MLP重定向模块。

### 7. 总结
*   **核心思想**：通过便携IMU采集人类先验，通过两阶段模仿学习实现形态泛化。
*   **速记版pipeline**：
    1. 穿戴IMU获取人体运动轨迹；
    2. MLP模型将指尖位置映射为灵巧手指令；
    3. 利用人类数据训练通用决策模型；
    4. 少量机器人数据微调以适配硬件形态。

**Key Findings:**

- To address these challenges, we introduce HumDex, a portable teleoperation system designed for humanoid whole-body dexterous manipulation.
- Building on this capability, we propose a two-stage imitation learning framework that first pre-trains on diverse human motion data to learn generalizable priors, and then fine-tunes on robot data to bridge the embodiment gap for precise execution.
- We demonstrate that this approach significantly improves generalization to new configurations, objects, and backgrounds with minimal data acquisition costs.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12260v1)
- [arXiv](https://arxiv.org/abs/2603.12260v1)

---

<a id='2603.12257v1'></a>
## [DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](https://arxiv.org/abs/2603.12257v1)

**Authors:** Yujie Wei, Xinyu Liu, Shiwei Zhang, Hangjie Yuan, Jinbo Xing, Zhekai Chen, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Ruihang Chu, Yingya Zhang, Yike Guo, Xihui Liu, Hongming Shan

**Published:** 2026-03-12

**Categories:** cs.CV

**Abstract:**

While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control. In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm. In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements. To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances. In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone. This provides motion-aware identity rewards in the latent space, prioritizing identity preservation aligned with human preferences. Supported by our curated large-scale dataset and the comprehensive DreamOmni Bench for multi-subject and omni-motion control evaluation, DreamVideo-Omni demonstrates superior performance in generating high-quality videos with precise controllability.

**Analysis:**

### 1. 摘要翻译
尽管大规模扩散模型已彻底改变了视频合成，但实现对多主体身份和多粒度运动的精确控制仍是一个重大挑战。现有方法往往受限于运动粒度不足、控制歧义和身份退化，导致身份保持和运动控制性能欠佳。为此，我们提出了 **DreamVideo-Omni**，这是一个统一框架，通过渐进式两阶段训练范式，实现了多主体定制与全能运动（omni-motion）控制的和谐统一。第一阶段，我们整合了涵盖主体外观、全局运动、局部动态和相机运动的综合控制信号，引入了条件感知3D旋转位置嵌入（RoPE）以协调异构输入，并采用分层运动注入策略增强全局运动引导；同时，引入组和角色嵌入以显式锚定运动信号，解耦复杂场景。第二阶段，为缓解身份退化，我们设计了潜空间身份奖励反馈学习范式，基于预训练视频扩散模型训练了一个潜空间身份奖励模型，在潜空间内提供运动感知的身份奖励，从而优先实现与人类偏好对齐的身份保持。基于我们精心构建的大规模数据集及DreamOmni Bench评估基准，DreamVideo-Omni在生成高质量、高可控性视频方面表现出优越性能。

### 2. 方法动机分析
*   **驱动力**：在保持多主体身份的同时，实现对全局（物体移动）、局部（肢体动作）和相机运动的多粒度精确控制。
*   **现有方法痛点**：1) 运动控制粒度受限于单一信号（如仅边界框）；2) 多主体场景下，运动控制信号与特定主体之间缺乏绑定机制，导致控制歧义；3) 现有的身份保持与运动控制存在本质冲突，导致运动幅度较大时身份退化严重。
*   **研究假设**：通过显式绑定控制信号与对应主体，并引入感知对齐的人类偏好奖励，能在潜空间内高效解决身份保持与运动控制的权衡。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **第一阶段（SFT）**：构建统一 DiT 框架，输入参考图像（外观）、全局边界框（位置/尺度）、稀疏点轨迹（局部动态/相机）。通过 **条件感知 3D RoPE** 协调不同时序的输入，使用 **分层运动注入** 将边界框信息注入各 Transformer 层，利用 **组/角色嵌入** 显式绑定“主体-运动”映射。
    2.  **第二阶段（RLHF）**：引入 **潜空间身份奖励模型 (LIRM)**。直接基于视频扩散模型（VDM）的主干，利用交叉注意力机制对比生成视频特征与参考图像身份特征。
    3.  **奖励反馈学习 (LIReFL)**：在潜空间执行梯度反向传播。在推理过程中执行“单步梯度增强去噪”，根据 LIRM 的反馈调整潜变量，而非依赖昂贵的 VAE 解码。

*   **算法逻辑**：LIRM 使用交叉注意力层 $hattn = \text{Softmax}(\frac{QK^T}{\sqrt{d}})V$ 计算身份对齐度。LIReFL 通过 $L = L_{sft} + \lambda_2 L_{LIReFL}$ 损失函数组合，既利用 SFT 保证基础生成质量，又通过奖励反馈微调模型以对齐身份偏好。

### 4. 方法对比分析
*   **本质区别**：与现有将身份与运动分离或简单拼接的方法不同，本方法在架构上通过 **显式绑定（Group/Role Embeddings）** 解决了多主体冲突，在优化目标上通过 **潜空间奖励反馈（LIReFL）** 解决了生成质量与身份忠实度的博弈。
*   **创新贡献**：提出了一种不经过 VAE 解码、直接在潜空间进行 RLHF 的高效范式，并创新性地使用了基于 VDM 本身的奖励模型。

### 5. 实验分析
*   **关键结果**：在 DreamOmni Bench 上，在多主体定制与运动控制指标（mIoU, EPE）及身份忠实度（R-DINO, Face-S）上显著优于 DreamVideo-2 和其他主流方法。
*   **优势**：实现了对多个主体的精确、解耦控制，大幅降低了计算开销，具备强大的泛化性。
*   **局限**：在极端复杂的场景下，对于细微的物体变形，仍可能存在轻微的身份偏差，需进一步优化奖励模型的鲁棒性。

### 6. 实用指南
*   **开源情况**：已开源（见项目主页 https://dreamvideo-omni.github.io）。
*   **实现细节**：建议在第一阶段使用较大的 Lambda1（重加权损失权重）以强调物体内的学习。RLHF 阶段关键在于 $\lambda_2$ 的调节，实验发现 0.1 是最优解，过大易导致奖励攻击（Reward Hacking）。
*   **迁移建议**：本方法的“潜空间奖励模型 + 直接梯度传播”范式可直接迁移至任何基于 DiT 的生成任务，无需依赖外部重型 CLIP 类奖励模型。

### 7. 总结
*   **核心思想**：通过潜空间显式绑定与奖励学习，协同身份保持与多粒度动态控制。
*   **速记版 Pipeline**：
    1. 统一多信号（图像+框+轨迹）输入；
    2. 引入组/角色嵌入解耦多主体控制；
    3. 训练潜空间身份奖励模型；
    4. 绕过 VAE 进行潜空间奖励反馈微调。

**Key Findings:**

- In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm.
- To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance.
- Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12257v1)
- [arXiv](https://arxiv.org/abs/2603.12257v1)

---

<a id='2603.12255v1'></a>
## [Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](https://arxiv.org/abs/2603.12255v1)

**Authors:** Fangfu Liu, Diankun Wu, Jiawei Chi, Yimo Cai, Yi-Hsin Hung, Xumin Yu, Hao Li, Han Hu, Yongming Rao, Yueqi Duan

**Published:** 2026-03-12

**Categories:** cs.CV, cs.LG

**Abstract:**

Humans perceive and understand real-world spaces through a stream of visual observations. Therefore, the ability to streamingly maintain and update spatial evidence from potentially unbounded video streams is essential for spatial intelligence. The core challenge is not simply longer context windows but how spatial information is selected, organized, and retained over time. In this paper, we propose Spatial-TTT towards streaming visual-based spatial intelligence with test-time training (TTT), which adapts a subset of parameters (fast weights) to capture and organize spatial evidence over long-horizon scene videos. Specifically, we design a hybrid architecture and adopt large-chunk updates parallel with sliding-window attention for efficient spatial video processing. To further promote spatial awareness, we introduce a spatial-predictive mechanism applied to TTT layers with 3D spatiotemporal convolution, which encourages the model to capture geometric correspondence and temporal continuity across frames. Beyond architecture design, we construct a dataset with dense 3D spatial descriptions, which guides the model to update its fast weights to memorize and organize global 3D spatial signals in a structured manner. Extensive experiments demonstrate that Spatial-TTT improves long-horizon spatial understanding and achieves state-of-the-art performance on video spatial benchmarks. Project page: https://liuff19.github.io/Spatial-TTT.

**Analysis:**

### 1. 摘要翻译
人类通过视觉流感知并理解现实世界。因此，在潜在的无限视频流中持续维护和更新空间证据，对于空间智能至关重要。核心挑战不仅在于更长的上下文窗口，更在于如何随时间选择、组织和保留空间信息。本文提出了 **Spatial-TTT**，这是一种面向流式视觉空间智能的测试时训练（Test-Time Training, TTT）框架。它通过调整参数子集（快速权重）来捕获并组织长视野场景视频中的空间证据。具体而言，我们设计了一种混合架构，采用大块更新（large-chunk updates）并行滑动窗口注意力机制，实现高效的空间视频处理。为了进一步提升空间感知能力，我们在引入3D时空卷积的TTT层中加入空间预测机制，以鼓励模型捕获跨帧的几何对应关系和时间连续性。此外，我们构建了一个包含密集3D空间描述的数据集，用于引导模型以结构化方式更新其快速权重，以记忆和组织全局3D空间信号。大量实验表明，Spatial-TTT改善了长视野空间理解能力，并在多个视频空间基准测试中达到了SOTA水平。

### 2. 方法动机分析
- **驱动力**：现有的多模态大模型（MLLM）主要在2D图像-文本对上训练，缺乏3D几何先验，且在处理长视频时，单纯扩展上下文窗口导致计算开销呈二次方增长，且会丢失细粒度空间细节。
- **现有方法痛点**：传统Transformer难以应对无限视频流，且目前的空间感知方法大多局限于短视频片段，无法进行长期的空间记忆累积。
- **研究假设**：通过在测试阶段实时更新快速权重，可以将模型转变为一个紧凑的非线性记忆体，从而在不增加无限计算开销的前提下，实现长视野的空间信息记忆与推理。

### 3. 方法设计详解
- **核心Pipeline**：
    1. **混合架构（Hybrid Architecture）**：在Transformer层中，以3:1比例交替使用TTT层（负责记忆压缩）和标准自注意力锚层（负责维持语义推理能力）。
    2. **大块更新（Large Chunk Update）**：为了克服小块更新导致的硬件效率低和空间结构破坏问题，将长视频拆分为较大的数据块进行更新，并利用Muon更新规则提升梯度下降的效率。
    3. **空间预测机制（Spatial-Predictive Mechanism）**：在TTT分支引入轻量级深度可分离3D时空卷积，替代点对点投影，利用局部空间邻域信息，使快速权重学习空间结构映射而非孤立Token映射。
    4. **滑动窗口注意力（SWA）**：在TTT层内并行运行SWA以处理因果约束，保证Chunk内部的时空连续性。
- **算法精要**：TTT通过梯度下降更新快速权重 $W_t$，将当前的Key-Value对关联映射到权重矩阵中。空间预测机制通过3D卷积预处理Key和Value，增强了权重对于几何对应关系的建模能力。

### 4. 方法对比分析
- **本质区别**：不同于传统的固定参数In-context Learning，该方法通过在推理阶段对“快速权重”进行在线梯度更新，真正实现了模型的自适应记忆。
- **创新贡献**：
    1. 提出了TTT在大规模视觉任务中的混合架构设计；
    2. 引入了3D卷积引导的“空间预测机制”，将时空先验注入记忆更新过程；
    3. 构造了密集场景描述数据集，解决了空间任务稀疏监督的问题。

### 5. 实验分析
- **验证方法**：在VSI-Bench、MindCube、VSI-SUPER-Recall/Count等基准上，与主流的封闭/开源模型进行对比。
- **关键结论**：Spatial-TTT在2B规模下，不仅计算效率远高于Qwen3-VL-2B等模型，且在处理1024帧长视频时依然保持稳定，性能显著优于传统MLLM。
- **局限性**：尽管效率大幅提升，但模型仍依赖于底座模型的推理能力，且TTT过程的超参数（如学习率、Chunk大小）对于不同场景的适应性可能需要精细调参。

### 6. 实用指南
- **开源情况**：已提供项目主页（https://liuff19.github.io/Spatial-TTT）。
- **实现细节**：关键在于3D卷积的Dirac初始化（保证初始输出恒等），以及Muon更新规则的使用。训练分为两阶段：先通过密集场景描述学习记忆，再通过空间VQA进行微调。
- **迁移可能**：该框架可直接迁移到机器人导航、自动驾驶感知等需要实时空间记忆的端侧设备任务中。

### 7. 总结
- **核心思想**：通过测试时训练将时空记忆压缩进动态权重。
- **速记版pipeline**：
    1. 切分长视频流为大Chunk；
    2. 混合注意力层提取语义与空间特征；
    3. 执行3D卷积辅助的梯度更新（TTT）；
    4. 累积并更新空间记忆。

**Key Findings:**

- In this paper, we propose Spatial-TTT towards streaming visual-based spatial intelligence with test-time training (TTT), which adapts a subset of parameters (fast weights) to capture and organize spatial evidence over long-horizon scene videos.
- To further promote spatial awareness, we introduce a spatial-predictive mechanism applied to TTT layers with 3D spatiotemporal convolution, which encourages the model to capture geometric correspondence and temporal continuity across frames.
- Extensive experiments demonstrate that Spatial-TTT improves long-horizon spatial understanding and achieves state-of-the-art performance on video spatial benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12255v1)
- [arXiv](https://arxiv.org/abs/2603.12255v1)

---

<a id='2603.12254v1'></a>
## [Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing](https://arxiv.org/abs/2603.12254v1)

**Authors:** Baifeng Shi, Stephanie Fu, Long Lian, Hanrong Ye, David Eigen, Aaron Reite, Boyi Li, Jan Kautz, Song Han, David M. Chan, Pavlo Molchanov, Trevor Darrell, Hongxu Yin

**Published:** 2026-03-12

**Categories:** cs.CV

**Abstract:**

Multi-modal large language models (MLLMs) have advanced general-purpose video understanding but struggle with long, high-resolution videos -- they process every pixel equally in their vision transformers (ViTs) or LLMs despite significant spatiotemporal redundancy. We introduce AutoGaze, a lightweight module that removes redundant patches before processed by a ViT or an MLLM. Trained with next-token prediction and reinforcement learning, AutoGaze autoregressively selects a minimal set of multi-scale patches that can reconstruct the video within a user-specified error threshold, eliminating redundancy while preserving information. Empirically, AutoGaze reduces visual tokens by 4x-100x and accelerates ViTs and MLLMs by up to 19x, enabling scaling MLLMs to 1K-frame 4K-resolution videos and achieving superior results on video benchmarks (e.g., 67.0% on VideoMME). Furthermore, we introduce HLVid: the first high-resolution, long-form video QA benchmark with 5-minute 4K-resolution videos, where an MLLM scaled with AutoGaze improves over the baseline by 10.1% and outperforms the previous best MLLM by 4.5%. Project page: https://autogaze.github.io/.

**Analysis:**

## 论文方法分析：AutoGaze

### 1. 摘要翻译
多模态大语言模型（MLLMs）在通用视频理解方面取得了进展，但在处理长视频和高分辨率视频时面临困难。尽管存在显著的时空冗余，这些模型仍平等地处理视频Transformer（ViT）或LLM中的每一个像素。我们引入了**AutoGaze**，一个轻量级模块，用于在视频输入到ViT或MLLM之前剔除冗余补丁（patches）。AutoGaze通过下一词预测和强化学习进行训练，能够自回归地选择一组最小的多尺度补丁，在用户指定的误差阈值内重建视频，从而在保留信息的同时消除冗余。实验表明，AutoGaze将视觉token减少了4倍至100倍，并将ViT和MLLM加速高达19倍，使得MLLM能够扩展到1K帧、4K分辨率的视频，并在视频基准测试上取得了优异的表现。此外，我们引入了HLVid，这是首个包含5分钟4K分辨率视频的长视频问答基准，实验证明，使用AutoGaze扩展的MLLM较基线模型提升了10.1%，并超越了此前最佳模型。

### 2. 方法动机分析
- **驱动力**：打破长视频、高分辨率视频处理中的计算瓶颈，实现高效且具备长上下文的视频理解。
- **现有方法痛点**：现有方法要么平等处理所有像素导致效率低下；要么仅在LLM阶段裁剪token，而ViT仍需处理全量像素，造成计算瓶颈；或依赖于性能较差的启发式方法。
- **研究假设**：视频中存在显著的时空冗余，通过一个“前置”且可学习的轻量级注意力模块，能够根据当前帧与历史信息，自回归地精简并动态选择关键视觉区域。

### 3. 方法设计详解
- **流程总结**：
    1. **视频输入**：接收视频帧序列。
    2. **自回归凝视（Gazing）**：利用卷积编码器和Transformer解码器，自回归地逐帧选择补丁。词表为多尺度补丁索引（支持$32\times32$到$224\times224$等四种尺度）。
    3. **自动停止策略**：解码器添加一个“损失预测头”，实时预测重建当前帧所需的损失。当损失低于预设阈值时，停止当前帧的补丁选择。
    4. **下游处理**：将选定的多尺度补丁输入修改后的ViT（支持多尺度patch输入），再经过LLM进行理解。
- **模型结构**：包含一个卷积编码器（提取特征）和一个由LLaMA 3架构改造的4层Transformer解码器（总计3M参数），解码器负责输出patch索引和损失预测。
- **算法解释**：
    - **重建损失（$L$）**：结合像素级重建损失和感知损失（基于DINOv2/SigLIP2 embedding的距离），确保重建质量。
    - **强化学习（RL）**：通过GRPO算法，将“负重建损失”作为奖励，引导模型在给定阈值下精选补丁。

### 4. 方法对比分析
- **本质区别**：AutoGaze是“前置（Pre-ViT）”的动态Token削减器，直接减少ViT的输入负载，而非在模型中间层进行剪枝。
- **创新贡献**：提出了一种结合自回归预测和强化学习的轻量化补丁精简框架；实现了自适应多尺度选择与动态停止机制。
- **适用场景**：极高分辨率（4K）及超长时序（1K帧）的视频任务。

### 5. 实验分析（精简版）
- **验证方法**：在多个主流长视频基准（VideoMME, HLVid）上评估性能。
- **关键结果**：在保证性能基本不变的情况下，视觉Token压缩高达100倍，推理加速最高19倍。
- **优势**：极高压缩比，大幅提升处理长高分视频的能力。
- **局限**：对剧烈的摄像机平移（Panning）处理不理想，且缺乏对物理因果（如抛物线运动）的先验建模。

### 6. 实用指南
- **开源情况**：已开源（https://autogaze.github.io/）。
- **实现细节**：
    - **超参数**：推荐设置重建损失阈值 $\epsilon=0.7$。
    - **训练**：两阶段训练，先NTP预训练（250K视频），后RL微调。
- **迁移可能**：可作为即插即用模块集成到任何现有的ViT-based视觉模型中，无需重新训练下游ViT。

### 7. 总结
- **核心思想**：利用轻量模型预先“凝视”关键区域，实现极致的视频压缩与加速。
- **速记版pipeline**：
    1. 视频分块处理。
    2. 解码器自回归选择多尺度patch。
    3. 实时预测重建误差，误差达标即停止。
    4. 仅处理精简后的patch并送入下游模型。

**Key Findings:**

- We introduce AutoGaze, a lightweight module that removes redundant patches before processed by a ViT or an MLLM.
- Furthermore, we introduce HLVid: the first high-resolution, long-form video QA benchmark with 5-minute 4K-resolution videos, where an MLLM scaled with AutoGaze improves over the baseline by 10.1% and outperforms the previous best MLLM by 4.5%.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12254v1)
- [arXiv](https://arxiv.org/abs/2603.12254v1)

---

<a id='2603.12250v1'></a>
## [DVD: Deterministic Video Depth Estimation with Generative Priors](https://arxiv.org/abs/2603.12250v1)

**Authors:** Hongfei Zhang, Harold Haodong Chen, Chenfei Liao, Jing He, Zixin Zhang, Haodong Li, Yihao Liang, Kanghao Chen, Bin Ren, Xu Zheng, Shuai Yang, Kun Zhou, Yinchuan Li, Nicu Sebe, Ying-Cong Chen

**Published:** 2026-03-12

**Categories:** cs.CV

**Abstract:**

Existing video depth estimation faces a fundamental trade-off: generative models suffer from stochastic geometric hallucinations and scale drift, while discriminative models demand massive labeled datasets to resolve semantic ambiguities. To break this impasse, we present DVD, the first framework to deterministically adapt pre-trained video diffusion models into single-pass depth regressors. Specifically, DVD features three core designs: (i) repurposing the diffusion timestep as a structural anchor to balance global stability with high-frequency details; (ii) latent manifold rectification (LMR) to mitigate regression-induced over-smoothing, enforcing differential constraints to restore sharp boundaries and coherent motion; and (iii) global affine coherence, an inherent property bounding inter-window divergence, which enables seamless long-video inference without requiring complex temporal alignment. Extensive experiments demonstrate that DVD achieves state-of-the-art zero-shot performance across benchmarks. Furthermore, DVD successfully unlocks the profound geometric priors implicit in video foundation models using 163x less task-specific data than leading baselines. Notably, we fully release our pipeline, providing the whole training suite for SOTA video depth estimation to benefit the open-source community.

**Analysis:**

作为计算机视觉领域的专家，我对这篇题为《DVD: Deterministic Video Depth Estimation with Generative Priors》的论文分析如下：

### 1. 主要贡献总结
该论文提出了一种名为 **DVD (Deterministic Video Depth)** 的框架，首次实现了将预训练视频扩散模型（Video Diffusion Models）确定性地转化为单步深度回归器。该方法成功克服了生成式模型中的随机性幻觉与漂移问题，同时打破了判别式模型对海量标注数据的依赖，在实现零样本（Zero-shot）SOTA 性能的同时，显著提升了训练效率。

### 2. 关键创新与方法论
DVD 的核心技术创新在于如何将“生成式先验”转化为“回归任务”的确定性输出，主要通过以下三个支柱实现：
*   **扩散时间步（Diffusion Timestep）作为结构锚点：** 将扩散模型的时间步策略性地引入回归过程，作为平衡深度图“全局结构稳定性”与“局部高频细节”的调节器。
*   **潜在流形校正（Latent Manifold Rectification, LMR）：** 针对传统回归方法容易导致的图像模糊问题，LMR 通过引入微分约束，强制模型在流形空间中保持几何边界的锐利度与时间序列的运动一致性。
*   **全局仿射相干性（Global Affine Coherence）：** 这是一项内在的几何约束设计，用于界定时间窗口间的发散度，从而无需复杂的时间对齐后处理即可实现长视频的平滑推理。

### 3. 对领域的潜在影响
*   **范式转换：** 该研究标志着视频深度估计从“生成式随机预测”向“确定性回归估计”的范式演进，证明了无需通过去噪采样即可直接提取扩散模型中深层几何知识的可能性。
*   **数据效率：** 使用比领先基线少 163 倍的特定任务数据即可达到 SOTA 效果，这对资源受限的研究者极具价值，预示着“预训练模型知识提取”在几何视觉任务中的巨大潜力。
*   **开源贡献：** 完整发布训练套件不仅利于社区复现，也为后续研究在视频基础模型上进行下游任务适配提供了高标准参考。

### 4. 受益的相关领域与应用
*   **自动驾驶与机器人：** 需要高精度、高一致性且低延迟的场景深度重建，DVD 的单步回归特性非常契合实时感知需求。
*   **AR/VR 与 3D 内容生成：** 视频转 3D（Video-to-3D）任务中，该方法提供的几何一致性深度图能大幅提升后续网格重建（Meshing）的质量。
*   **电影与视觉特效（VFX）：** 为后期制作中的动态深度遮罩（Depth Matting）提供更稳定、边缘更清晰的解决方案。

### 5. 可推断的潜在局限性
*   **计算开销（推理侧）：** 尽管 DVD 是“单步”回归，但由于其底层是基于大型视频扩散模型，其推理时的显存占用和计算负载可能依然远高于轻量级的判别式模型（如基于 MobileNet 或轻量 Transformer 的架构）。
*   **对预训练模型性能的依赖：** DVD 的上限很大程度上受限于所使用的预训练视频扩散模型的先验质量；如果基础模型在某些极端环境或特殊视角下缺乏训练，DVD 的几何估计能力可能会受限。
*   **动态场景的鲁棒性：** 尽管采用了全局仿射相干性，但在高度非刚性变形或遮挡极其复杂的场景下，该确定性回归模型处理几何突变的能力是否优于基于轨迹跟踪的传统方法，仍有待实测验证。

**专家点评：**
这篇论文非常巧妙地处理了“生成模型用于判别任务”时的核心矛盾（确定性 vs 随机性）。它不仅仅是一个应用尝试，更是在**模型压缩与知识蒸馏**方向上的一次大胆跨界，将“扩散模型的生成能力”转化为“回归器的感知能力”，是当前 CV 领域最前沿的探索方向之一。

**Key Findings:**

- To break this impasse, we present DVD, the first framework to deterministically adapt pre-trained video diffusion models into single-pass depth regressors.
- Extensive experiments demonstrate that DVD achieves state-of-the-art zero-shot performance across benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12250v1)
- [arXiv](https://arxiv.org/abs/2603.12250v1)

---

