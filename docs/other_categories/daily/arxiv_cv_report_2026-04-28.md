time: 20260428

# Arxiv Computer Vision Papers - 2026-04-28

## Executive Summary

以下是为您准备的每日报告执行摘要，涵盖了2026年4月27日arXiv上10篇计算机视觉领域论文的主要发现和趋势。

---

### **每日报告执行摘要：2026年4月27日 Arxiv 计算机视觉论文**

**日期：** 2026年4月27日
**论文总数：** 10篇

#### **1. 整体主题与趋势概述**

本日论文集呈现出三大核心趋势：**（1）视觉-语言模型（VLM）的深度优化与对齐**，尤其是围绕文本到视频生成、多模态理解、360度场景理解及测试时自适应等方向；**（2）机器人学习中的具身智能**，强调从人类示范中学习先验知识以及在动态扰动下的鲁棒控制；**（3）扩散模型与基础模型的通用化应用**，包括将其用于通用分割、布局感知文本渲染等任务。此外，论文普遍关注**数据质量、模型效率（粗到细策略）以及奖励机制（过程奖励模型）** 对生成与理解能力的提升作用。

#### **2. 特别重要或创新的论文**

- **最具突破性：** **《Tuna-2: Pixel Embeddings Beat Vision Encoders...》** (Liu et al.)  
  该文挑战了当前多模态模型依赖独立视觉编码器的范式，提出直接用像素嵌入替代视觉编码器。这一创新简化了模型架构，并声称在多项理解和生成任务上超越了传统方法，可能重塑多模态模型的设计思路。

- **最具应用潜力：** **《World-R1: Reinforcing 3D Constraints for Text-to-Video Generation》** (Wang et al.)  
  通过强化学习将3D空间约束（如物体间几何关系、物理一致性）直接注入视频生成过程，有望解决当前视频生成中常见的“物理不合规”问题，是该领域向实用化迈进的关键一步。

#### **3. 新兴研究方向与技术**

- **过程奖励模型（Process Reward Models, PRM）在视觉-语言模型中的应用：** 《Improving Vision-language Models with Perception-centric Process Reward Models》 (Min et al.) 将原本用于数学推理的PRM引入视觉理解，通过细粒度、中间步骤的反馈来优化VLM，预示了“过程监督”将成为提升模型可靠性的重要工具。

- **粗到细（Coarse-to-Fine）策略在视觉-语言-动作（VLA）策略中的普及：** 《CF-VLA》 (Du et al.) 展示了如何通过分层生成（先规划宏观动作，再细化细节）来提升机器人操作效率，这反映了具身智能领域对计算效率与任务复杂性的平衡考量。

- **代理中心（Agent-Centric）强化学习：** 《Agent-Centric Visual RL under Dynamic Perturbations》 (Fang et al.) 强调在动态干扰下以智能体自身为锚点的表示学习，这与传统环境中心视角形成对比，可能对无人机、自动驾驶等受强干扰场景产生重要影响。

#### **4. 建议全文阅读的论文（按优先级排序）**

1.  **《Tuna-2》** — 对于任何从事多模态模型架构设计的研究者，这是必须阅读的，因为它提出了范式级变革。
2.  **《World-R1》** — 对文本到视频生成、游戏引擎或物理模拟感兴趣的研究者，此文提供了第一条可循的强化学习路径。
3.  **《Improving VLMs with Perception-centric Process Reward Models》** — 对于研究模型对齐、强化学习或视觉理解可靠性的学者，此文的方法论具有高度可迁移性。
4.  **《Learning Human-Intention Priors... for Robotic Manipulation》** — 机器人操作领域的研究者应重点关注，因为它展示了如何高效利用大规模非专家数据提取高维意图先验。

---

**总体评价：** 本日论文质量较高，尤其以《Tuna-2》和《World-R1》为代表，展现了从“架构拼凑”向“核心机制创新”的转变。视觉-语言模型的优化正从单纯增加数据量转向更精细的监督信号（过程奖励、几何约束），而机器人学习正更加务实地处理数据效率和动态环境适应问题。

---

## Table of Contents

1. [World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](#2604.24764v1)
2. [Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation](#2604.24763v1)
3. [Learning Human-Intention Priors from Large-Scale Human Demonstrations for Robotic Manipulation](#2604.24681v1)
4. [Agent-Centric Visual Reinforcement Learning under Dynamic Perturbations](#2604.24661v1)
5. [Probing CLIP's Comprehension of 360-Degree Textual and Visual Semantics](#2604.24642v1)
6. [CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies](#2604.24622v1)
7. [Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift](#2604.24602v1)
8. [Improving Vision-language Models with Perception-centric Process Reward Models](#2604.24583v1)
9. [Diffusion Model as a Generalist Segmentation Learner](#2604.24575v1)
10. [TextGround4M: A Prompt-Aligned Dataset for Layout-Aware Text Rendering](#2604.24459v1)

---

## Papers

<a id='2604.24764v1'></a>
## [World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](https://arxiv.org/abs/2604.24764v1)

**Authors:** Weijie Wang, Xiaoxuan He, Youping Gu, Yifan Yang, Zeyu Zhang, Yefei He, Yanbo Ding, Xirui Hu, Donny Y. Chen, Zhiyuan He, Yuqing Yang, Bohan Zhuang

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

Recent video foundation models demonstrate impressive visual synthesis but frequently suffer from geometric inconsistencies. While existing methods attempt to inject 3D priors via architectural modifications, they often incur high computational costs and limit scalability. We propose World-R1, a framework that aligns video generation with 3D constraints through reinforcement learning. To facilitate this alignment, we introduce a specialized pure text dataset tailored for world simulation. Utilizing Flow-GRPO, we optimize the model using feedback from pre-trained 3D foundation models and vision-language models to enforce structural coherence without altering the underlying architecture. We further employ a periodic decoupled training strategy to balance rigid geometric consistency with dynamic scene fluidity. Extensive evaluations reveal that our approach significantly enhances 3D consistency while preserving the original visual quality of the foundation model, effectively bridging the gap between video generation and scalable world simulation.

**Analysis:**

### 1. 摘要翻译
视频基础模型在视觉合成方面表现出色，但常受几何不一致性的困扰。现有方法尝试通过架构修改来注入3D先验，但往往导致高计算成本并限制了可扩展性。我们提出了World-R1，一个通过强化学习将视频生成与3D约束对齐的框架。为促进这种对齐，我们引入了一个专为世界模拟定制的纯文本数据集。利用Flow-GRPO，我们通过预训练3D基础模型和视觉-语言模型的反馈来优化模型，在不改变底层架构的情况下强制执行结构一致性。我们进一步采用周期性解耦训练策略，以平衡刚性几何一致性与动态场景的流动性。大量评估表明，我们的方法在保留基础模型原始视觉质量的同时，显著增强了3D一致性，有效填补了视频生成与可扩展世界模拟之间的鸿沟。

### 2. 方法动机分析
*   **驱动力**：现有的视频生成模型在生成长视频或复杂相机运动时，缺乏内嵌的3D几何感知，导致对象变形、消失或产生伪影。作者希望在不牺牲模型通用性和视觉质量的前提下，赋予视频模型几何一致的世界建模能力。
*   **现有方法痛点**：以往研究多依赖显式的3D表征注入或辅助控制模块，这不仅增加了推理成本，还限制了模型的泛化能力和生成的多样性。
*   **研究假设**：视频基础模型在预训练过程中已经隐含了丰富的3D几何知识，无需显式的结构修改，仅通过强化学习（RL）引导即可激发并对齐这些潜在能力。

### 3. 方法设计详解
World-R1的核心在于一个**无需架构修改的分析-by-合成闭环优化系统**：
1.  **隐式相机调节**：借鉴“Go-with-the-Flow”范式，将相机运动轨迹映射为2D光流场，并通过“离散噪声传输”机制，将轨迹先验注入到初始潜空间噪声中。这无需辅助网络即可实现精确的相机控制。
2.  **复合奖励系统**：
    *   **3D感知奖励($R_{3D}$)**：通过Depth Anything 3将视频提升至3D高斯溅射（3DGS）表示。
        *   $S_{meta}$：在与生成轨迹有较大偏移的元视角（meta-view）进行渲染，由Qwen3-VL评估结构合理性，惩罚伪影。
        *   $S_{recon}$：对比原始视频与3DGS重渲染视频的LPIPS差异，确保几何保真度。
        *   $S_{traj}$：评估生成动作与输入指令的对齐度。
    *   **通用生成奖励($R_{gen}$)**：使用HPSv3评估审美与视觉质量，防止过度约束导致画质下降。
3.  **周期性解耦训练**：为防止模型在追求几何刚性时丧失对非刚性动态（如流体、人物动作）的表现力，在常规RL优化过程中，每100步插入一个仅优化$R_{gen}$的微调阶段。

### 4. 方法对比分析
*   **本质区别**：与显式注入3D先验的方法不同，World-R1属于**后训练优化（Post-training optimization）**范式，模型参数在推理时无需额外计算负载。
*   **创新贡献**：成功将强化学习应用于视频模型的3D对齐，通过分析-by-合成的反馈机制，绕过了对大规模标注3D数据集的依赖。
*   **适用场景**：适合需要严苛几何一致性的场景（如自动驾驶模拟、物理世界建模），同时对生成的多样性和视觉美感有要求的任务。

### 5. 实验分析（精简版）
*   **验证方法**：在复杂场景下与基线模型（Wan 2.1等）对比，利用3DGS重建后的PSNR、SSIM和LPIPS指标进行量化评估。
*   **关键结果**：在3D一致性指标上，PSNR提升10.23dB，在通用视频评测（VBench）中表现优于基线。
*   **优势**：极高的几何一致性与严密的轨迹跟随，且推理速度保持不变。
*   **局限**：在线RL训练过程极其耗时，长程时序生成的复杂逻辑仍受限于基础模型的生成容量。

### 6. 实用指南
*   **开源情况**：项目主页为 https://aka.ms/world-r1。
*   **迁移可能**：该框架的复合奖励机制（基于VLM评价与几何重建）可直接迁移至其他以一致性为目标的生成任务（如3D资产生成、多视图一致性优化）。
*   **关键点**：周期性解耦训练是保持动态效果的关键超参数，需根据具体场景平衡$R_{3D}$与$R_{gen}$权重。

### 7. 总结
*   **核心思想**：利用RL强化引导，无需结构变动实现视频几何一致性。
*   **速记版pipeline**：
    1.  生成轨迹并注入噪声实现相机控制；
    2.  利用3DGS重建视频并渲染元视角；
    3.  通过多维度评价打分（VLM+几何+美学）；
    4.  周期性切换训练目标平衡几何与动态。

**Key Findings:**

- We propose World-R1, a framework that aligns video generation with 3D constraints through reinforcement learning.
- To facilitate this alignment, we introduce a specialized pure text dataset tailored for world simulation.
- Extensive evaluations reveal that our approach significantly enhances 3D consistency while preserving the original visual quality of the foundation model, effectively bridging the gap between video generation and scalable world simulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24764v1)
- [arXiv](https://arxiv.org/abs/2604.24764v1)

---

<a id='2604.24763v1'></a>
## [Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation](https://arxiv.org/abs/2604.24763v1)

**Authors:** Zhiheng Liu, Weiming Ren, Xiaoke Huang, Shoufa Chen, Tianhong Li, Mengzhao Chen, Yatai Ji, Sen He, Jonas Schult, Belinda Zeng, Tao Xiang, Wenhu Chen, Ping Luo, Luke Zettlemoyer, Yuren Cong

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

Unified multimodal models typically rely on pretrained vision encoders and use separate visual representations for understanding and generation, creating misalignment between the two tasks and preventing fully end-to-end optimization from raw pixels. We introduce Tuna-2, a native unified multimodal model that performs visual understanding and generation directly based on pixel embeddings. Tuna-2 drastically simplifies the model architecture by employing simple patch embedding layers to encode visual input, completely discarding the modular vision encoder designs such as the VAE or the representation encoder. Experiments show that Tuna-2 achieves state-of-the-art performance in multimodal benchmarks, demonstrating that unified pixel-space modelling can fully compete with latent-space approaches for high-quality image generation. Moreover, while the encoder-based variant converges faster in early pretraining, Tuna-2's encoder-free design achieves stronger multimodal understanding at scale, particularly on tasks requiring fine-grained visual perception. These results show that pretrained vision encoders are not necessary for multimodal modelling, and end-to-end pixel-space learning offers a scalable path toward stronger visual representations for both generation and perception.

**Analysis:**

### 1. 摘要翻译
统一的多模态模型通常依赖于预训练的视觉编码器，并为理解和生成任务使用独立的视觉表示，这导致了任务间的失调，并阻碍了从原始像素进行的端到端优化。我们引入了 Tuna-2，这是一个原生统一的多模态模型，直接基于像素嵌入进行视觉理解和生成。Tuna-2 通过采用简单的补丁嵌入层来编码视觉输入，彻底摒弃了诸如 VAE 或表示编码器等模块化视觉编码器设计，从而极大地简化了模型架构。实验表明，Tuna-2 在多模态基准测试中实现了最先进的性能，证明了统一的像素空间建模完全可以与高质量图像生成的潜在空间方法相媲美。此外，虽然基于编码器的变体在预训练早期收敛更快，但 Tuna-2 的无编码器设计在大规模训练下实现了更强的多模态理解能力，特别是在需要细粒度视觉感知的任务上。这些结果表明，预训练的视觉编码器对于多模态建模并非必要，端到端的像素空间学习为生成和感知提供了更强的视觉表示的可扩展路径。

### 2. 方法动机分析
- **驱动力**：旨在构建一个真正的端到端、原生统一的多模态模型（UMM），消除任务间的表示失调。
- **现有痛点**：现有 UMM 高度依赖预训练视觉编码器（如 CLIP、VAE），这些编码器引入了固定的归纳偏置（如固定分辨率、有限的底层视觉细节访问），且模块化设计限制了端到端优化的潜力。
- **研究假设**：预训练的视觉编码器对于构建强大的多模态模型并非必要；直接在像素空间进行端到端学习，配合恰当的架构设计和正则化，能获得更强的视觉表示。

### 3. 方法设计详解
- **流程总结**：
    1. **直接补丁嵌入**：弃用 VAE 和复杂的视觉表示编码器，利用简单的 Patch Embedding 层将原始图像直接转化为视觉 Token。
    2. **统一主干**：采用单一 Transformer 解码器架构，输入序列由视觉 Token 和文本 Token 混合构成。
    3. **生成头设计**：采用基于流匹配（Flow Matching）的生成策略，利用 $x$-预测和 $v$-损失目标函数，直接在像素空间进行去噪生成。
    4. **遮蔽视觉特征学习（正则化）**：训练时随机遮蔽一部分视觉补丁，并以可学习的掩码 Token 替代，迫使模型在局部信息缺失下进行推理，从而学习到更鲁棒的视觉表示。
- **关键公式**：
    - 去噪轨迹：$x_t = tx_1 + (1-t)x_0$（$x_1$ 为真实图，$x_0$ 为噪声）。
    - 学习目标：$L_{\text{flow}} = \mathbb{E}_{t,c,x_1,x_0} ||v_\theta - v||^2_2$，其中 $v_\theta = \frac{x_\theta - x_t}{1-t}$。
- **协同工作**：Patch Embedding 提供高分辨率原始像素信号，Transformer 解码器完成理解与生成任务的联合建模，Flow Matching 负责像素空间的重建，掩码机制提升了模型在冗余像素空间中的表征稳定性。

### 4. 方法对比分析
- **本质区别**：Tuna-2 移除了所有预训练视觉编码器，实现了真正的“无编码器（Encoder-free）”架构，完全依赖端到端的像素空间学习。
- **创新贡献**：提出了一种无需 VAE 和视觉表示编码器的原生 UMM 架构，并通过遮蔽学习方案解决了高维像素空间建模难的问题。
- **适用场景**：适用于高性能、通用化的视觉-语言理解与生成任务，尤其在需要细粒度感知（如 OCR、小物体检测）的场景表现卓越。

### 5. 实验分析（精简版）
- **验证方法**：在九个 VQA 理解基准、两个生成基准以及图像编辑任务上进行了测试。
- **关键结果**：在 7B 尺度下，Tuna-2 在多模态理解（尤其是细粒度感知）上优于带有表示编码器的变体（Tuna-R）及现有基线模型；生成性能与带编码器的模型持平。
- **优劣势**：优势在于细粒度感知能力更强，架构更简单，且无需额外的特征对齐预训练；局限在于大规模预训练早期收敛速度较慢。

### 6. 实用指南
- **开源情况**：请关注官方项目页面（https://tuna-ai.org/tuna-2）。
- **实现建议**：实验表明 7:3 的生成与理解数据采样比例是平衡两项任务的关键；掩码机制建议在预训练的最后 40% 阶段加入以增强鲁棒性。
- **迁移可能**：该架构可以直接扩展至多模态视频建模，其无编码器特性使其对不同输入分辨率具有天然的兼容性。

### 7. 总结
- **核心思想**：彻底去编码器化，利用单 Transformer 解码器在像素空间实现端到端的多模态统一建模。
- **速记版pipeline**：
    1. 输入图像切片并直接嵌入为视觉 Token；
    2. 将视觉 Token 与文本 Token 输入共享 Transformer；
    3. 训练时随机遮蔽图像片段以强化特征提取；
    4. 利用流匹配机制直接在像素空间进行理解与生成任务优化。

**Key Findings:**

- We introduce Tuna-2, a native unified multimodal model that performs visual understanding and generation directly based on pixel embeddings.
- Experiments show that Tuna-2 achieves state-of-the-art performance in multimodal benchmarks, demonstrating that unified pixel-space modelling can fully compete with latent-space approaches for high-quality image generation.
- These results show that pretrained vision encoders are not necessary for multimodal modelling, and end-to-end pixel-space learning offers a scalable path toward stronger visual representations for both generation and perception.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24763v1)
- [arXiv](https://arxiv.org/abs/2604.24763v1)

---

<a id='2604.24681v1'></a>
## [Learning Human-Intention Priors from Large-Scale Human Demonstrations for Robotic Manipulation](https://arxiv.org/abs/2604.24681v1)

**Authors:** Yifan Xie, YuAn Wang, Guangyu Chen, Jinkun Liu, Yu Sun, Wenbo Ding

**Published:** 2026-04-27

**Categories:** cs.RO

**Abstract:**

Human videos contain rich manipulation priors, but using them for robot learning remains difficult because raw observations entangle scene understanding, human motion, and embodiment-specific action. We introduce MoT-HRA, a hierarchical vision-language-action framework that learns human-intention priors from large-scale human demonstrations. We first curate HA-2.2M, a 2.2M-episode action-language dataset reconstructed from heterogeneous human videos through hand-centric filtering, spatial reconstruction, temporal segmentation, and language alignment. On top of this dataset, MoT-HRA factorizes manipulation into three coupled experts: a vision-language expert predicts an embodiment-agnostic 3D trajectory, an intention expert models MANO-style hand motion as a latent human-motion prior, and a fine expert maps the intention-aware representation to robot action chunks. A shared-attention trunk and read-only key-value transfer allow downstream control to use human priors while limiting interference with upstream representations. Experiments on hand motion generation, simulated manipulation, and real-world robot tasks show that MoT-HRA improves motion plausibility and robust control under distribution shift.

**Analysis:**

这是一份关于论文 **MoT-HRA** 的深度技术分析。

---

### 1. 摘要翻译
人类视频中包含丰富的操作先验，但由于原始观测中包含了复杂的场景理解、人类动作和体现特定（embodiment-specific）的动作，利用这些视频进行机器人学习依然困难。我们提出了 **MoT-HRA**，一种分层视觉-语言-动作（VLA）框架，用于从大规模人类演示中学习人类意图先验。首先，我们构建了 **HA-2.2M**，这是一个包含 220 万个剧集、通过手部中心过滤、空间重建、时间分割和语言对齐从异构人类视频中重构而成的动作-语言数据集。在此基础上，MoT-HRA 将操作分解为三个耦合专家：视觉-语言专家预测与体现无关（embodiment-agnostic）的 3D 轨迹；意图专家将 MANO 手部动作建模为潜在的人类运动先验；精细专家将意图感知表示映射为机器人动作块。共享注意力中枢和只读键值（key-value）传递机制允许下游控制器使用人类先验，同时限制对上游表示的干扰。实验证明，MoT-HRA 提升了运动合理性和分布偏移下的鲁棒控制能力。

---

### 2. 方法动机分析
*   **驱动力**：利用海量非机器人人类视频（如 YouTube）作为监督信号，弥补机器人演示数据匮乏且昂贵的问题。
*   **现有方法痛点**：直接将视频帧转化为动作标签会发生“表示崩塌”：将场景理解、手部运动和机器人特定的控制指令强行揉合进单一表示中，导致模型无法区分“人类想做什么”与“机器人如何做”，在环境变化时模型变得脆弱。
*   **研究假设**：通过分层结构——先空间规划（Where）、再意图建模（How）、最后机器人适配（Action），可以将人类视频中可迁移的物理知识从具体的机器人控制指令中解耦。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **HA-2.2M 数据构建**：通过视觉模型（Gemini+V-JEPA）过滤视频，利用 VitPose 和 HaMeR 重构 MANO 3D 手部姿态，结合 Depth Anything 3 重建 3D 空间，将原始视频转化为结构化的手部动作序列。
    2.  **MoT-HRA 架构**：采用“混合专家（MoE）”结构的 Transformer，包含三个专家：
        *   **视觉-语言专家**：预测 3D 空间航点（Waypoints），提供 embodiment-agnostic 的空间骨架。
        *   **意图专家**：使用条件流匹配（Conditional Flow Matching）技术，将 3D 轨迹和图像上下文转化为 MANO 手部动作的潜在表示（Latent Intention）。
        *   **精细专家**：接收意图表示，输出机器人特定的动作块（Action Chunks）。
*   **知识绝缘（Knowledge Insulation）**：这是核心设计。不同专家之间通过只读键值缓存通信，下游专家（精细专家）读取上游专家（意图专家）的隐藏状态，但其更新梯度**不会**回传影响上游先验，从而防止动作特定信息“污染”通用的意图先验。

---

### 4. 方法对比分析
*   **本质区别**：与现有直接预测动作的 VLA 不同，MoT-HRA 引入了**显式的手部运动学约束（MANO）作为中间意图流形**，而非仅仅依靠潜变量。
*   **创新贡献**：成功将“人类如何完成任务”的物理先验（以 MANO 表示）与“机器人如何操作”的执行策略（以动作块表示）实现了分层级联。
*   **适用场景**：适用于存在跨视角、跨 embodiment 的机器人学习任务，特别是在需要精准手部操作的场景。

---

### 5. 实验分析（精简版）
*   **关键结论**：在 Ego4D/OakInk 手部运动生成任务中，ADE 和 DTW 指标大幅优于对比方法；在 SimplerEnv 任务中，成功率从基线的 30%-40% 提升至 66.1%。
*   **主要优势**：运动轨迹平滑、 anatomical (人体结构) 合理性高，对分布偏移（如背景更换、物体位置改变）具有极强的鲁棒性。
*   **主要局限**：对超长视距、多物体复杂任务的泛化能力仍有待提升；高质量数据的自动 curation 存在噪声。

---

### 6. 实用指南
*   **实现细节**：
    *   **关键超参**： chunk horizon $H=15$；MANO 生成使用 Classifier-Free Guidance (CFG)，scale 为 6.0。
    *   **数据预处理**：必须确保 MANO 重建的 3D 坐标与 Depth Anything 提供的深度图实现 scale 对齐。
*   **迁移可能**：该架构的“空间-意图-精细”三阶段范式可直接迁移至需要处理多模态长序列的其它任务（如自动驾驶的路径预测与控制）。

---

### 7. 总结
*   **核心思想**：通过分层解耦与梯度绝缘，将人类视频转化为通用的意图物理先验。
*   **速记版 Pipeline**：
    1.  **数据清洗**：选出包含手部交互的短视频段。
    2.  **空间与意图重建**：从视频中提取 3D 路径及 MANO 手部姿态。
    3.  **专家分层预测**：先定空间方位，再定手部意图，最后映射机器人动作。
    4.  **梯度隔离训练**：确保动作学习不破坏通用的物理动作先验。

**Key Findings:**

- We introduce MoT-HRA, a hierarchical vision-language-action framework that learns human-intention priors from large-scale human demonstrations.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24681v1)
- [arXiv](https://arxiv.org/abs/2604.24681v1)

---

<a id='2604.24661v1'></a>
## [Agent-Centric Visual Reinforcement Learning under Dynamic Perturbations](https://arxiv.org/abs/2604.24661v1)

**Authors:** Zhengru Fang, Yu Guo, Fei Liu, Yuang Zhang, Yihang Tao, Senkang Hu, Wenbo Ding, Yuguang Fang

**Published:** 2026-04-27

**Categories:** cs.RO

**Abstract:**

Visual reinforcement learning aims to empower an agent to learn policies from visual observations, yet it remains vulnerable to dynamic visual perturbations, such as unpredictable shifts in corruption types. To systematically study this, we introduce the Visual Degraded Control Suite (VDCS), a benchmark extending DeepMind Control Suite with Markov-switching degradations to simulate non-stationary real-world perturbations. Experiments on VDCS reveal severe performance degradation in existing methods. We theoretically prove via information-theoretic analysis that this failure stems from reconstruction-based objectives inevitably entangling perturbation artifacts into latent representations. To mitigate this negative impact, we propose Agent-Centric Observations with Mixture-of-Experts (ACO-MoE) to robustify visual RL against perturbations. The proposed framework leverages unique agent-centric restoration experts, achieving restoration from corruptions and task-relevant foreground extraction, thereby decoupling perception from perturbation before being processed by the RL agent. Extensive experiments on VDCS show our ACO-MoE outperforms strong baselines, recovering 95.3% of clean performance under challenging Markov-switching corruptions. Moreover, it achieves SOTA results on DMControl Generalization with random-color and video-background perturbations, demonstrating a high level of robustness.

**Analysis:**

### 1. 摘要翻译
视觉强化学习旨在从视觉观察中学习策略，但极易受到动态视觉扰动（如不可预见的腐蚀类型转换）的影响。为系统地研究这一问题，我们引入了视觉退化控制套件（VDCS），这是一个通过马尔可夫切换退化来模拟非平稳现实世界扰动的基准。对VDCS的实验表明，现有方法性能严重下降。我们通过信息论分析从理论上证明，这一失效源于基于重构的目标函数不可避免地将扰动伪影纠缠到潜在表示中。为了减轻这种负面影响，我们提出了“代理中心观察与专家混合模型”（ACO-MoE）来强化视觉强化学习对扰动的鲁棒性。该框架利用独特的代理中心修复专家，实现从腐蚀中修复和任务相关前景的提取，从而在RL智能体处理观察之前实现感知与扰动的解耦。在VDCS上的大量实验表明，我们的ACO-MoE表现优于强基线，在具有挑战性的马尔可夫切换腐蚀下恢复了95.3%的干净性能。此外，它在具有随机颜色和视频背景扰动的DMControl Generalization上实现了SOTA结果，证明了极高的鲁棒性。

### 2. 方法动机分析
- **驱动力**：解决视觉RL在非平稳、动态变化环境中的脆弱性。
- **痛点**：现有方法（无论是数据增强还是重构驱动的世界模型）均无法在像素层面有效隔离扰动。重构目标（如DreamerV3）会将干扰信息强制编码进潜在空间，导致想象力受损（state collapse）。
- **研究假设**：通过将“感知（修复与分割）”与“决策（策略学习）”解耦，在进入RL模块前构建一个“代理中心（Agent-Centric）”的干净观察流，能彻底消除扰动对策略产生的干扰。

### 3. 方法设计详解
- **pipeline总结**：
  1. **输入处理**：接收含有物理扰动的原始RGB帧。
  2. **专家混合（MoE）路由**：通过一个轻量级路由器（基于特征全局平均池化）动态选择针对当前腐蚀类型的修复专家。
  3. **双流修复与分割**：
     - **RGB残差分支**：专家预测RGB残差，修复任务相关区域的纹理与信息。
     - **Mask分支**：预测前景掩码，分离出智能体和目标物体，同时抑制背景扰动。
  4. **合成（Composition）**：将修复后的前景与黑色背景合成，输出纯净的代理中心观察。
  5. **冻结预处理**：整个ACO-MoE在RL训练前预训练并冻结，确保下游RL模块面对的是分布一致的输入。
- **核心公式意义**：$x˜t = oˆt \odot mt + b \odot (1 − mt)$，这是一个显式的信息瓶颈，强制模型只关注经过验证的任务相关前景，忽略背景扰动。

### 4. 方法对比分析
- **本质区别**：传统方法试图学习“鲁棒表示（Robust Representation）”来对抗扰动，而ACO-MoE则是通过“像素级修复（Pixel-level Restoration）”在输入端直接移除扰动，属于预处理范式。
- **创新点**：将MoE结构引入视觉修复，实现对不同扰动（雨、雪、噪点等）的针对性处理；利用显式前景分割辅助信息解耦，理论上证明了该方法能够降低表征中的噪声互信息。
- **适用场景**：适用于环境视觉条件极差、且扰动类型随时间动态切换的机器人控制任务。

### 5. 实验分析
- **验证方法**：在VDCS（自定义动态扰动套件）和DMC-GB基准上进行测试，对比了多种模型自由和模型基础RL算法。
- **结论**：ACO-MoE在动态环境下相比最好的基线提升了78.5%，证明了其在非平稳环境下的优越性。
- **局限性**：依赖于离线预训练的前景标注，对于未见过的、非物理扰动（Out-of-distribution）的泛化能力可能有限。

### 6. 实用指南
- **开源情况**：作者提供了完整的VDCS基准，框架逻辑清晰，适合复现。
- **注意点**：需要提前采集带有前景掩码的对齐数据集，对各任务的前景分割容差需精细调参。
- **迁移性**：该框架是“即插即用（Plug-and-play）”的，完全不需要修改下游的RL训练逻辑，仅需将其作为一个冻结的CNN/U-Net层嵌入即可，兼容DreamerV3, TD-MPC2等多种架构。

### 7. 总结
- **核心思想**：通过分治的专家修复与前景遮盖，在输入端实现扰动与感知分离。
- **速记版pipeline**：
  1. 路由器动态识别当前扰动类型。
  2. 选定对应专家修复图像纹理。
  3. 通过掩码提取任务相关前景。
  4. 将前景移至黑色背景输出。
  5. 将清理后的观察送入决策网络。

**Key Findings:**

- To systematically study this, we introduce the Visual Degraded Control Suite (VDCS), a benchmark extending DeepMind Control Suite with Markov-switching degradations to simulate non-stationary real-world perturbations.
- To mitigate this negative impact, we propose Agent-Centric Observations with Mixture-of-Experts (ACO-MoE) to robustify visual RL against perturbations.
- Extensive experiments on VDCS show our ACO-MoE outperforms strong baselines, recovering 95.3% of clean performance under challenging Markov-switching corruptions.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24661v1)
- [arXiv](https://arxiv.org/abs/2604.24661v1)

---

<a id='2604.24642v1'></a>
## [Probing CLIP's Comprehension of 360-Degree Textual and Visual Semantics](https://arxiv.org/abs/2604.24642v1)

**Authors:** Hai Wang, Xiaochen Yang, Mingzhi Dong, Jing-Hao Xue

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

The dream of instantly creating rich 360-degree panoramic worlds from text is rapidly becoming a reality, yet a crucial gap exists in our ability to reliably evaluate their semantic alignment. Contrastive Language-Image Pre-training (CLIP) models, standard AI evaluators, predominantly trained on perspective image-text pairs, face an open question regarding their understanding of the unique characteristics of 360-degree panoramic image-text pairs. This paper addresses this gap by first introducing two concepts: \emph{360-degree textual semantics}, semantic information conveyed by explicit format identifiers, and \emph{360-degree visual semantics}, invariant semantics under horizontal circular shifts. To probe CLIP's comprehension of these semantics, we then propose novel evaluation methodologies using keyword manipulation and horizontal circular shifts of varying magnitudes. Rigorous statistical analyses across popular CLIP configurations reveal that: (1) CLIP models effectively leverage explicit textual identifiers, demonstrating an understanding of 360-degree textual semantics; and (2) CLIP models fail to robustly preserve semantic alignment under horizontal circular shifts, indicating limited comprehension of 360-degree visual semantics. To address this limitation, we propose a LoRA-based fine-tuning framework that explicitly instills invariance to circular shifts. Our fine-tuned models exhibit improved comprehension of 360-degree visual semantics, though with a slight degradation in original semantic evaluation performance, highlighting a fundamental trade-off in adapting CLIP to 360-degree panoramic images. Code is available at https://github.com/littlewhitesea/360Semantics.

**Analysis:**

### 1. 摘要翻译
即时生成丰富的360度全景图已成为现实，但其语义对齐的评估仍存在严重鸿沟。CLIP模型作为标准的AI评估器，主要在透视图像对上训练，对360度全景图像对的独特特性缺乏理解。本文引入了两个新概念：**360度文本语义**（由显式格式标识符传达的语义）和**360度视觉语义**（水平循环移位下的不变语义）。为了探测CLIP对这些语义的理解，我们提出了利用关键词操纵和水平循环移位的新评估方法。统计分析表明：(1) CLIP模型能有效利用文本标识符；(2) CLIP模型在水平循环移位下无法保持语义对齐，表明其视觉语义理解有限。为此，我们提出了一种基于LoRA的微调框架，显式注入循环移位不变性。实验证明，该框架提升了模型对360度视觉语义的理解，同时保留了基础语义能力，揭示了性能与不变性之间的权衡。

### 2. 方法动机分析
*   **驱动力**：填补360度全景生成模型在“评估”环节的缺失，即现有的CLIP评估指标未能考虑360度全景图像的“球形几何”特性。
*   **现有痛点**：CLIP在海量透视图像上预训练，不具备全景图像水平旋转不变的先验，导致模型对同一场景但在不同旋转角度下的得分不一致，从而影响评估的稳定性。
*   **研究假设**：通过显式引入循环移位不变性训练，可以使预训练的CLIP模型在保持原有的语义特征提取能力的同时，获得对全景场景的鲁棒性。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **语义探测**：通过关键词操纵（是否包含“360 panorama”等前缀）评估模型对“文本语义”的感知；通过计算图像水平循环移位前后CLIP得分的差异（基于稳定性边界$\beta$）评估“视觉语义”的感知。
    2.  **LoRA微调**：仅对图像编码器应用LoRA微调，利用定制的联合损失函数进行优化。
*   **损失函数**：$L_{FT} = \lambda \cdot L_{charb}(s^\Delta_\theta, s_\theta) + (1 - \lambda) \cdot L_{charb}(s^\Delta_\theta, s)$
    *   **Invariance term** ($L_{charb}(s^\Delta_\theta, s_\theta)$): 最小化微调模型对同一图像不同位移版本的评分差异，强制模型理解旋转不变性。
    *   **Regularization term** ($L_{charb}(s^\Delta_\theta, s)$): 约束模型保持与冻结的原始预训练模型一致的预测能力，防止语义遗忘。
    *   **$\lambda$平衡**：通过knee-point detection（膝点检测）自适应选择，在不变性提升与原始语义能力损失之间寻找平衡。

### 4. 方法对比分析
*   **本质区别**：不同于传统的全监督微调或数据增强，本文侧重于**“语义对齐的鲁棒性”**，不仅要求模型识别图像，更要求模型在拓扑变换下具备不变的评分逻辑。
*   **创新贡献**：定义了360度语义的两个维度；提出了基于Tukey箱线图的“稳定性边界”以量化模型的不稳定性；提供了基于LoRA的高效微调方案。

### 5. 实验分析
*   **验证方法**：使用Wilcoxon符号秩检验，对比frozen与fine-tuned模型在不同位移下的score分布。
*   **关键结论**：冻结的CLIP模型在循环移位下得分波动显著；微调后的模型在所有测试位移下，p-value均小于0.01，表明模型获得了显著的位移鲁棒性。
*   **权衡**：提升鲁棒性会轻微降低CLIP对某些复杂语义的识别精度。

### 6. 实用指南
*   **开源**：https://github.com/littlewhitesea/360Semantics
*   **注意点**：$\lambda$的设置至关重要，建议在实际应用中采用论文提到的knee-point方法动态寻找最佳超参数。
*   **迁移**：LoRA微调仅针对图像编码器，这种轻量化策略极易迁移至其他Vision-Language预训练模型（如SigLIP等）。

### 7. 总结
*   **核心思想**：通过LoRA轻量化微调注入全景几何不变性。
*   **速记版Pipeline**：
    1. 定义全景旋转不变性边界。
    2. 对CLIP图像编码器应用LoRA。
    3. 联合优化不变性损失与正则损失。
    4. 完成全景一致性对齐评估。

**Key Findings:**

- To probe CLIP's comprehension of these semantics, we then propose novel evaluation methodologies using keyword manipulation and horizontal circular shifts of varying magnitudes.
- To address this limitation, we propose a LoRA-based fine-tuning framework that explicitly instills invariance to circular shifts.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24642v1)
- [arXiv](https://arxiv.org/abs/2604.24642v1)

---

<a id='2604.24622v1'></a>
## [CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies](https://arxiv.org/abs/2604.24622v1)

**Authors:** Fan Du, Feng Yan, Jianxiong Wu, Xinrun Xu, Weiye Zhang, Weinong Wang, Yu Guo, Bin Qian, Zhihai He

**Published:** 2026-04-27

**Categories:** cs.CV, cs.AI

**Abstract:**

Flow-based vision-language-action (VLA) policies offer strong expressivity for action generation, but suffer from a fundamental inefficiency: multi-step inference is required to recover action structure from uninformative Gaussian noise, leading to a poor efficiency-quality trade-off under real-time constraints. We address this issue by rethinking the role of the starting point in generative action modeling. Instead of shortening the sampling trajectory, we propose CF-VLA, a coarse-to-fine two-stage formulation that restructures action generation into a coarse initialization step that constructs an action-aware starting point, followed by a single-step local refinement that corrects residual errors. Concretely, the coarse stage learns a conditional posterior over endpoint velocity to transform Gaussian noise into a structured initialization, while the fine stage performs a fixed-time refinement from this initialization. To stabilize training, we introduce a stepwise strategy that first learns a controlled coarse predictor and then performs joint optimization. Experiments on CALVIN and LIBERO show that our method establishes a strong efficiency-performance frontier under low-NFE (Number of Function Evaluations) regimes: it consistently outperforms existing NFE=2 methods, matches or surpasses the NFE=10 $π_{0.5}$ baseline on several metrics, reduces action sampling latency by 75.4\%, and achieves the best average real-robot success rate of 83.0\%, outperforming MIP by 19.5 points and $π_{0.5}$ by 4.0 points. These results suggest that structured, coarse-to-fine generation enables both strong performance and efficient inference. Our code is available at https://github.com/EmbodiedAI-RoboTron/CF-VLA.

**Analysis:**

### 1. 摘要翻译
基于流匹配（Flow Matching）的视觉-语言-动作（VLA）策略具有强大的表达能力，但在处理动作生成时存在根本性的效率低下：需要多步推理过程从无信息的噪声中恢复动作结构，导致其在实时约束下效率与质量的权衡较差。本文通过重新思考生成动作建模中“起始点”的作用，提出了CF-VLA。它将动作生成重构为两阶段过程：首先通过一个粗粒度初始化阶段构建一个动作感知的起始点，随后进行单步局部修正以纠正残差。实验表明，该方法在低NFE（函数评估次数）体制下建立了强大的效率-性能边界，在CALVIN和LIBERO上表现出色，并将动作采样延迟降低了75.4%。

---

### 2. 方法动机分析
- **核心动机**：流匹配方法在推理时，总是从无信息的纯高斯噪声开始，强制模型在整个轨迹中执行“全局传输”任务，造成了严重的计算冗余。
- **痛点**：现有方法将“寻找方向”与“微调动作”这两个本质不同的任务混合在同一多步求解过程中，导致在实时（低NFE）约束下，模型难以在有限步数内同时完成全局导航和细节刻画。
- **研究假设**：通过将“全局动作对齐（粗粒度）”与“局部动作修正（细粒度）”显式解耦，并利用学习到的动作先验构造高质量的初始起始点，可以极大减轻推理压力。

---

### 3. 方法设计详解
CF-VLA的核心是将生成过程解构为两个功能性阶段：
1. **粗粒度初始化（Coarse Initialization）**：
   - 模型学习一个关于终点速度的条件后验分布。
   - 输入观测后，将纯高斯噪声转换为一个偏向真实动作流形的“动作先验引导（AP-guided）”初始值 $\tilde{\epsilon}$。
2. **细粒度修正（Fine Refinement）**：
   - 仅执行单步局部更新。由于初始点已处于动作流形的邻域内，因此不再需要复杂的全局遍历，仅需利用一阶更新恢复目标动作。
3. **训练策略（Stepwise Training）**：
   - 引入两阶段训练：Phase I为“稳定性导向预热”，先学习控制粗糙预测；Phase II为“全联合优化”，通过KL散度对齐，使粗粒度输出与细粒度修正实现协同。

---

### 4. 方法对比分析
- **本质区别**：传统流匹配方法视所有采样步骤为同等地位的迭代，而CF-VLA将推理拆分为“全局位置校准”与“局部残差消除”，实现了非对称的计算分配。
- **创新点**：
    - 引入了**方差感知的终点分布建模**，使模型能够生成具有不确定性的动作分布，而非坍缩为确定性估计。
    - **两阶段异步训练策略**，解决了粗细阶段耦合导致的训练不稳定性。
- **适用场景**：实时性要求极高、计算资源有限且需要精确长程控制的机器人操作任务。

---

### 5. 实验分析
- **验证方法**：在LIBERO（模拟）、CALVIN（长程序列）及真实机器人任务（拾取、擦拭、倾倒等）中进行评测。
- **关键结论**：在LIBERO上，CF-VLA以2步（NFE=2）的超低计算量，达到了96.5%的成功率，超过了NFE=10的基线（95.7%），且采样延迟缩短了75.4%。
- **优势**：显著提升了动作生成的实时性与长程控制的稳定性。
- **局限**：对粗粒度初始化阶段的分布建模精度有较高依赖，且训练流程相比单阶段方法更为复杂。

---

### 6. 实用指南
- **开源地址**：[https://github.com/EmbodiedAI-RoboTron/CF-VLA](https://github.com/EmbodiedAI-RoboTron/CF-VLA)
- **实现细节**：
    - 需特别关注超参数 `noise_var`（共享噪声方差）和 `loss_logvar_weight`（Phase I中对数方差匹配系数）的调节。
    - 训练时务必区分Phase I和Phase II的任务语义。
- **迁移可能**：该框架是一个“即插即用”的模块，理论上可直接移植到任何基于流匹配的VLA模型（如RT-2/OpenVLA体系）中，只需更换动作头的预测逻辑，无需重构Backbone。

---

### 7. 总结
- **核心思想**：通过分层解耦（先粗对齐、后精细修正）实现高效动作生成。
- **速记版pipeline**：
    1. 输入观测，采样高斯噪声。
    2. 执行一步粗粒度预测，得到动作流形上的初始点。
    3.  rescaling状态与步长，进行单步局部细粒度修正。
    4. 输出最终执行动作。

**Key Findings:**

- Instead of shortening the sampling trajectory, we propose CF-VLA, a coarse-to-fine two-stage formulation that restructures action generation into a coarse initialization step that constructs an action-aware starting point, followed by a single-step local refinement that corrects residual errors.
- To stabilize training, we introduce a stepwise strategy that first learns a controlled coarse predictor and then performs joint optimization.
- Experiments on CALVIN and LIBERO show that our method establishes a strong efficiency-performance frontier under low-NFE (Number of Function Evaluations) regimes: it consistently outperforms existing NFE=2 methods, matches or surpasses the NFE=10 $π_{0.5}$ baseline on several metrics, reduces action sampling latency by 75.4\%, and achieves the best average real-robot success rate of 83.0\%, outperforming MIP by 19.5 points and $π_{0.5}$ by 4.0 points.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24622v1)
- [arXiv](https://arxiv.org/abs/2604.24622v1)

---

<a id='2604.24602v1'></a>
## [Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift](https://arxiv.org/abs/2604.24602v1)

**Authors:** Lixian Chen, Mingxuan Huang, Yanhui Chen, Junyi Lin, Yang Shi

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

Vision-language models transfer well in zero-shot settings, but at deployment the visual and textual branches often shift asymmetrically. Under this condition, entropy-based test-time adaptation can sharpen the fused posterior while increasing error, because an unreliable modality may still dominate fusion. We study this failure mode through a majorization view of multimodal posteriors and cast adaptation as a constrained de-mixing problem on the fused prediction. Based on this view, we propose MG-MTTA, which keeps the backbone frozen and updates only a lightweight gate or adapter. The objective combines fused-posterior entropy minimization with a reliability-aware gate prior built from anchor-based modality consistency and cross-modal conflict. Our analysis gives conditions under which entropy reduction preserves the correct ranking and a threshold that characterizes modality-dominance failure. On the ImageNet-based benchmark, MG-MTTA improves top-1 accuracy from 57.97 to 66.51 under semantics-preserving textual shift and from 21.68 to 26.27 under joint visual-textual shift, while remaining competitive in the visual-only benchmark. These results show that multimodal test-time adaptation should control modality reliability, not just prediction entropy.

**Analysis:**

### 1. 摘要翻译
视觉语言模型（VLMs）在零样本设置下表现出色，但部署时视觉和文本分支往往发生不对称偏移。在这种条件下，基于熵的测试时适应（TTA）可能会在增加错误的同时锐化融合后的后验概率，因为不可靠的模态可能仍主导融合过程。我们通过多模态后验概率的Majorization（主序）视角研究了这一失败模式，并将适应过程建模为融合预测上的约束解混问题。基于此，我们提出了MG-MTTA，它保持骨干网络冻结，仅更新轻量级门控或适配器。该目标函数将融合后验熵最小化与基于锚点模态一致性和跨模态冲突的可靠性感知门控先验相结合。我们的分析给出了熵减法能够保持正确排序的条件，并确定了一个刻画模态主导失败的阈值。实验表明，多模态测试时适应应控制模态可靠性，而不仅仅是预测熵。

### 2. 方法动机分析
*   **驱动力**：解决多模态模型在面对模态不对称偏移（如视觉受损、文本提示漂移）时，简单的熵最小化会导致模型过分自信地强化错误模态（即“sharpener of mistakes”）。
*   **现有痛点**：传统的熵最小化假设所有分支均可靠或一致，但在多模态场景中，这种假设不成立。错误模态的过高置信度会淹没正确模态，导致适应过程偏离目标。
*   **研究假设**：通过Majorization几何视角，将模态偏移建模为“双随机混合（Doubly Stochastic Mixing）”，适应过程本质上是一个“解混”过程，必须通过可靠性信号约束以确保Majorization方向正确。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **输入与前向传播**：输入偏置的视觉/文本数据，通过冻结的Backbone获得初始后验概率 $p_v, p_t$。
    2.  **可靠性估计**：通过维护各模态的“运行锚点（Running Anchors）”，计算当前分布与锚点的偏移（Majorization Proxy），并结合跨模态一致性（JS散度+排名冲突），估计模态可靠性。
    3.  **门控生成**：将可靠性信号转为logits并通过Softmax生成门控权重 $\alpha_\phi$，用于加权融合视觉与文本预测。
    4.  **适应优化**：在测试时仅更新轻量级门控参数，损失函数包含：熵最小化（提高锐度）、可靠性正则化（向更可靠的模态对齐）、多样性约束（防止坍塌）。
*   **关键公式解释**：
    *   **Majorization Proxy**：通过对数排序后的差异衡量分布偏离，确保模型在“正确”方向上进行平滑或锐化，而非盲目地向极值点收敛。
    *   **可靠性感知先验**：将动态估计的可靠性注入Gate，动态压制偏差严重的模态。

### 4. 方法对比分析
*   **本质区别**：从传统的单纯概率分布操作（熵最小化）转向了结构化的几何解混（Majorization-guided de-mixing）。
*   **创新贡献**：提出了一种理论框架来解释为何熵最小化在多模态下会失败，并提供了显式的“失败阈值”推导。
*   **适用场景**：适用于视觉-语言多模态模型的离线测试时适应（TTA），特别是在模态不对称偏移环境下。

### 5. 实验分析
*   **关键结果**：在ImageNet-C（视觉偏移）和语义保持的文本偏移实验中，MG-MTTA在保持视觉性能的基础上，显著优于单纯的熵最小化，大幅提升了对复杂多模态偏移的鲁棒性。
*   **优势**：理论完备，不仅提供了算法，还通过诊断性实验验证了其对“失败模式”的纠正能力。
*   **局限**：对轻量级门控组件的设计敏感，且依赖于在线更新的可靠性锚点，在极端低样本量下可能存在估计偏差。

### 6. 实用指南
*   **迁移建议**：该方法不依赖特定的Backbone，只需在融合层添加轻量门控，可以轻松迁移到任何双流或多流多模态融合模型中。
*   **实现要点**：
    *   **超参数**：温度 $\tau$ 和冲突惩罚权重 $\lambda_c$ 是调节可靠性敏感度的关键。
    *   **锚点更新**：需使用EMA（指数移动平均）保持锚点平滑，以适应测试过程的分布变化。

### 7. 总结
*   **核心思想**：通过可靠性加权的几何解混防止多模态模型产生错误的自信。
*   **速记版Pipeline**：
    1. 冻结模型权重，获取双流预测。
    2. 评估各流与锚点的偏离度与流间冲突。
    3. 更新融合权重以避开不可靠流。
    4. 联合优化熵与正则化目标。

**Key Findings:**

- Based on this view, we propose MG-MTTA, which keeps the backbone frozen and updates only a lightweight gate or adapter.
- These results show that multimodal test-time adaptation should control modality reliability, not just prediction entropy.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24602v1)
- [arXiv](https://arxiv.org/abs/2604.24602v1)

---

<a id='2604.24583v1'></a>
## [Improving Vision-language Models with Perception-centric Process Reward Models](https://arxiv.org/abs/2604.24583v1)

**Authors:** Yingqian Min, Kun Zhou, Yifan Li, Yuhuan Wu, Han Peng, Yifan Du, Wayne Xin Zhao, Min Yang, Ji-Rong Wen

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

Recent advancements in reinforcement learning with verifiable rewards (RLVR) have significantly improved the complex reasoning ability of vision-language models (VLMs). However, its outcome-level supervision is too coarse to diagnose and correct errors within the reasoning chain. To this end, we propose Perceval, a process reward model (PRM) that enables token-level error grounding, which can extract image-related claims from the response and compare them one by one with the visual evidence in the image, ultimately returning claims that contain perceptual errors. Perceval is trained with perception-intensive supervised training data. We then integrate Perceval into the RL training process to train the policy models. Specifically, compared to traditional GRPO, which applies sequence-level advantages, we apply token-level advantages by targeting penalties on hallucinated spans identified by Perceval, thus enabling fine-grained supervision signals. In addition to augmenting the training process, Perceval can also assist VLMs during the inference stage. Using Perceval, we can truncate the erroneous portions of the model's response, and then either have the model regenerate the response directly or induce the model to reflect on its previous output. This process can be repeated multiple times to achieve test-time scaling. Experiments show significant improvements on benchmarks from various domains across multiple reasoning VLMs trained with RL, highlighting the promise of perception-centric supervision as a general-purpose strategy. For test-time scaling, it also demonstrates consistent performance gains over other strategies, such as major voting. Our code and data will be publicly released at https://github.com/RUCAIBox/Perceval.

**Analysis:**

这是一份关于论文《Improving Vision-language Models with Perception-centric Process Reward Models》的深入分析。

### 1. 摘要翻译
强化学习结合可验证奖励（RLVR）显著提升了视觉语言模型（VLM）的复杂推理能力。然而，现有的结果级监督过于粗糙，难以识别并纠正推理链中的具体错误。为此，本文提出了**Perceval**，一种感知中心的过程奖励模型（PRM）。Perceval能够实现标记级（token-level）错误定位，即从回复中提取与图像相关的声明，并将其与视觉证据逐一对比，从而识别感知错误。Perceval使用感知密集型监督数据进行训练，并被集成到策略模型的强化学习训练过程中。相比于传统的GRPO，我们通过Perceval识别的幻觉片段施加标记级惩罚，实现了更细粒度的监督。此外，Perceval还能在推理阶段通过截断-重写机制实现测试时扩展（test-time scaling）。实验证明，该方法在多个视觉推理基准上均显著优于现有策略。

### 2. 方法动机分析
*   **驱动力**：解决VLM在多步视觉推理中因“幻觉”导致的错误积累问题，提升感知忠实度。
*   **现有痛点**：RLVR通常采用序列级奖励（即对整个推理链打分），这种稀疏奖励机制无法定位到底是哪一步感知或逻辑出错了，导致“信度分配”困难。
*   **研究假设**：通过引入过程监督，针对推理链中具体的感知性错误片段施加细粒度的负反馈，可以引导模型建立更精准的视觉 grounding 能力。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据构建**：利用强模型（如Gemini-2.5-Pro）生成感知强化数据集，包含图像、推理链及幻觉标记的结构化注解。
    2.  **Perceval训练**：采用SFT方式训练Perceval，使其输出结构化的验证信息（包括`<think>`部分的思考和`<answer>`部分的错误片段列表）。
    3.  **RL训练（token-level Advantage）**：在强化学习循环中，Perceval识别出幻觉span后，通过公式(3)动态调整GRPO的优势值。如果token属于幻觉片段，则大幅降低其奖励，从而在梯度更新时惩罚这些特定内容。
    4.  **测试时推断**：采用“截断-重写（Truncate-Regenerate）”策略。若Perceval发现错误，则删除错误部分，保留verified prefix让模型重新生成。
*   **核心算法**：改进的优势函数 $A'_{i,t} := A_i - \alpha \cdot m_{i,t} \cdot |A_i|$。其中 $m_{i,t}$ 是幻觉掩码，$\alpha$ 是控制惩罚强度的超参数（文中取0.1），实现了对序列中错误片段的精准降权。

### 4. 方法对比分析
*   **本质区别**：从“结果评价”升级为“过程溯源”。传统方法仅对结局负责，Perceval则对推理过程中的“感知对齐”负责。
*   **创新贡献**：提出了一种通用的感知中心过程奖励评估器，并设计了与GRPO深度集成的token级优势重分配方案。
*   **适用场景**：所有需要视觉精确定位、实体识别或空间关系推理的VLM任务。

### 5. 实验分析（精简版）
*   **结论**：在Visual Search任务上，3B模型显著提升了约4%的性能；同时在BLINK等基准上，Perceval有效减少了幻觉。
*   **主要优势**：展现了极强的泛化能力——即使只在感知任务上训练，模型在数学推理等任务上也获得了性能提升。
*   **局限性**：过度惩罚（如$\alpha > 0.1$）会导致非事实性但语法必需的词（如介词）被误伤，产生训练噪声。

### 6. 实用指南
*   **开源信息**：代码与数据已在 `https://github.com/RUCAIBox/Perceval` 发布。
*   **关键点**：超参数 $\alpha$ 是调优的核心，建议从0.1开始尝试。在推理时，迭代次数 $k$ 的增加能带来性能增益，但会引入延迟成本。
*   **迁移建议**：可将此PRM框架迁移到任何基于RL的视觉推理模型中，通过替换掉原来的标量奖励模型（Scalar RM）接口即可。

### 7. 总结
*   **核心思想**：通过token级惩罚幻觉，实现过程化监督与感知增强。
*   **速记版pipeline**：
    1. 训练一个能找错的PRM模型；
    2. 用PRM定位训练样本中的幻觉片段；
    3. 在RL训练中，给幻觉片段扣分；
    4. 推理时，若发现错，自动删掉并让模型重写。

**Key Findings:**

- To this end, we propose Perceval, a process reward model (PRM) that enables token-level error grounding, which can extract image-related claims from the response and compare them one by one with the visual evidence in the image, ultimately returning claims that contain perceptual errors.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24583v1)
- [arXiv](https://arxiv.org/abs/2604.24583v1)

---

<a id='2604.24575v1'></a>
## [Diffusion Model as a Generalist Segmentation Learner](https://arxiv.org/abs/2604.24575v1)

**Authors:** Haoxiao Wang, Antao Xiang, Haiyang Sun, Peilin Sun, Changhao Pan, Yifu Chen, Minjie Hong, Weijie Wang, Shuang Chen, Yue Chen, Zhou Zhao

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

Diffusion models are primarily trained for image synthesis, yet their denoising trajectories encode rich, spatially aligned visual priors. In this paper, we demonstrate that these priors can be utilized for text-conditioned semantic and open-vocabulary segmentation, and this approach can be generalized to various downstream tasks to make a general-purpose diffusion segmentation framework. Concretely, we introduce DiGSeg (Diffusion Models as a Generalist Segmentation Learner), which repurposes a pretrained diffusion model into a unified segmentation framework. Our approach encodes the input image and ground-truth mask into the latent space and concatenates them as conditioning signals for the diffusion U-Net. A parallel CLIP-aligned text pathway injects language features across multiple scales, enabling the model to align textual queries with evolving visual representations. This design transforms an off-the-shelf diffusion backbone into a universal interface that produces structured segmentation masks conditioned on both appearance and arbitrary text prompts. Extensive experiments demonstrate state-of-the-art performance on standard semantic segmentation benchmarks, as well as strong open-vocabulary generalization and cross-domain transfer to medical, remote sensing, and agricultural scenarios-without domain-specific architectural customization. These results indicate that modern diffusion backbones can serve as generalist segmentation learners rather than pure generators, narrowing the gap between visual generation and visual understanding.

**Analysis:**

作为计算机视觉领域的专家，我对这篇题为《Diffusion Model as a Generalist Segmentation Learner》的论文分析如下：

### 1. 主要贡献摘要
本文提出了 **DiGSeg** 框架，旨在打破扩散模型仅用于图像生成的刻板印象，将其转化为一个通用的分割学习器。该研究证明了利用预训练扩散模型的去噪轨迹作为空间对齐的视觉先验，能够实现高性能的语义分割和开放词汇分割，且在无需针对特定领域架构调整的情况下，即可在医学、遥感和农业等跨领域任务中表现优异。

### 2. 关键创新与方法论
*   **双模态条件注入：** 该方法巧妙地将图像与真值掩码（ground-truth mask）映射到潜在空间，并将其作为扩散模型 U-Net 的条件信号。这使得扩散模型在生成过程中能够从单纯的“无中生有”转变为基于输入语义的“结构化重构”。
*   **多尺度 CLIP 对齐路径：** 引入了一个并行的文本处理路径，将 CLIP 特征在多个尺度上注入到 U-Net 中。这种设计使得模型能够将用户的任意文本查询（Text Prompts）与扩散过程中不断演进的视觉表征进行精确对齐。
*   **通用接口设计：** 将扩散模型从“生成器”重塑为“通用的分割接口”，这种架构上的统一性避免了针对不同任务进行繁琐的架构修补。

### 3. 对领域的潜在影响
*   **范式迁移：** 该研究有力地推动了从“生成模型与判别模型分离”到“统一多模态生成式感知”的范式转变，证明了生成模型的底层表示对于视觉理解任务具有极高的利用价值。
*   **打破“数据壁垒”：** 由于其强大的跨领域泛化能力，DiGSeg 减少了在医疗或农业等标注数据稀缺领域对专门模型的依赖，为通用视觉人工智能（AGI）在特定垂直领域的落地提供了新思路。
*   **弥合生成与理解的鸿沟：** 这项工作强化了“视觉世界模型”的观点，即一个强大的扩散模型通过学习去噪，实际上已经获得了对物理世界空间布局和语义的深刻理解。

### 4. 受益的相关领域与应用
*   **医疗影像分析：** 无需针对不同器官或病灶进行特定模型训练，即可利用文本驱动进行精准病灶分割。
*   **遥感图像处理：** 在处理复杂地物分类和农作物监测时，能够利用预训练模型的泛化能力处理未见过的地形场景。
*   **交互式图像编辑与分析：** 此类框架天然适用于需要“分割-编辑-生成”闭环的复杂视觉交互任务。
*   **机器人视觉：** 为机器人提供了一种能够通过自然语言指令动态理解并分割环境对象的方法。

### 5. 可推断的局限性
*   **计算成本（Inference Latency）：** 扩散模型的去噪过程通常涉及多次迭代（Iterative Denoising），与传统的单次前向（Single-pass）分割网络（如 Mask2Former 或 SAM）相比，DiGSeg 在推理速度上可能面临显著挑战，难以满足实时性要求。
*   **对先验模型的依赖：** 模型的性能高度依赖于初始预训练扩散模型（如 Stable Diffusion）的质量，如果基础模型在特定领域表现不佳，该框架的性能上限也会受到约束。
*   **训练的复杂性：** 尽管不需要架构定制，但联合训练扩散模型与分割任务在超参数敏感度、显存占用以及对齐机制上可能存在较高的工程调试难度。

**总结：** 这篇论文的趣味性在于它不仅验证了扩散模型的“理解”能力，更通过极具洞察力的接口设计（Conditioning Mechanism），将生成模型的“去噪路径”成功转化为“分割特征映射”，这是连接生成式人工智能与机器感知领域的重要桥梁。

**Key Findings:**

- In this paper, we demonstrate that these priors can be utilized for text-conditioned semantic and open-vocabulary segmentation, and this approach can be generalized to various downstream tasks to make a general-purpose diffusion segmentation framework.
- Concretely, we introduce DiGSeg (Diffusion Models as a Generalist Segmentation Learner), which repurposes a pretrained diffusion model into a unified segmentation framework.
- Our approach encodes the input image and ground-truth mask into the latent space and concatenates them as conditioning signals for the diffusion U-Net.
- Extensive experiments demonstrate state-of-the-art performance on standard semantic segmentation benchmarks, as well as strong open-vocabulary generalization and cross-domain transfer to medical, remote sensing, and agricultural scenarios-without domain-specific architectural customization.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24575v1)
- [arXiv](https://arxiv.org/abs/2604.24575v1)

---

<a id='2604.24459v1'></a>
## [TextGround4M: A Prompt-Aligned Dataset for Layout-Aware Text Rendering](https://arxiv.org/abs/2604.24459v1)

**Authors:** Dongxing Mao, Yilin Wang, Linjie Li, Zhengyuan Yang, Alex Jinpeng Wang

**Published:** 2026-04-27

**Categories:** cs.CV

**Abstract:**

Despite recent advances in text-to-image generation, models still struggle to accurately render prompt-specified text with correct spatial layout -- especially in multi-span, structured settings. This challenge is driven not only by the lack of datasets that align prompts with the exact text and layout expected in the image, but also by the absence of effective metrics for evaluating layout quality. To address these issues, we introduce TextGround4M, a large-scale dataset of over 4 million prompt-image pairs, each annotated with span-level text grounded in the prompt and corresponding bounding boxes. This enables fine-grained supervision for layout-aware, prompt-grounded text rendering. Building on this, we propose a lightweight training strategy for autoregressive T2I models that appends layout-aware span tokens during training, without altering model architecture or inference behavior. We further construct a benchmark with stratified layout complexity to evaluate both open-source and proprietary models in a zero-shot setting. In addition, we introduce two layout-aware metrics to address the long-standing lack of spatial evaluation in text rendering. Our results show that models trained on TextGround4M outperform strong baselines in text fidelity, spatial accuracy, and prompt consistency, highlighting the importance of fine-grained layout supervision for grounded T2I generation.

**Analysis:**

### 1. 摘要翻译
尽管文本到图像（T2I）生成技术取得了巨大进步，但模型在处理多文本跨度（multi-span）的结构化布局时，仍难以准确呈现指定文本。这一难题不仅源于缺乏将提示词与图像中确切文本及布局对齐的数据集，还因为缺乏评估布局质量的有效指标。为此，我们推出了TextGround4M，这是一个包含超过400万个prompt-image对的大规模数据集，每个样本都标注了与提示词对应的跨度级文本及其边界框。这为布局感知、提示词驱动的文本渲染提供了细粒度监督。在此基础上，我们提出了一种针对自回归T2I模型的轻量级训练策略：在训练过程中附加布局感知跨度标记（layout-aware span tokens），且无需改变模型架构或推理行为。我们还构建了一个具有分层布局复杂度的基准测试，并引入了两种布局感知指标。结果表明，在TextGround4M上训练的模型在文本保真度、空间准确性和提示词一致性方面均优于强基线模型，突显了细粒度布局监督对于接地气（grounded）T2I生成的关键意义。

### 2. 方法动机分析
- **驱动力**：实现高质量、具有精确空间控制的文本渲染，满足海报设计、广告等对文本位置和内容高度敏感的场景需求。
- **现有方法痛点**：现有数据集（如AnyWord-3M）仅提供OCR标注，缺乏文本内容与提示词的语义绑定，导致模型在生成时常出现位置错乱、内容遗漏或拼写错误。
- **核心直觉**：通过在训练中显式地提供“文本内容+空间坐标”的监督信号，可以让自回归模型在不改动架构的情况下，“内化”文本与空间的对齐逻辑。

### 3. 方法设计详解
- **数据集构建（TextGround4M）**：
  1. **图像采集**：合并公共数据集并利用GPT-4o进行层次化查询生成，从CommonCrawl挖掘更多文本密集型图像。
  2. **标注对齐**：联合使用Qwen2.5-VL（生成细粒度描述）和PaddleOCR（提取边界框），通过多阶段匹配算法将语义跨度与坐标绑定。
  3. **质量过滤**：利用启发式规则剔除低质量样本，并通过VLM进行语义审计，确保标注可靠。
- **训练流程（Layout-posting）**：
  1. **Token构建**：将目标文本及其坐标信息转化为离散的“布局感知标记（Layout-aware tokens）”。
  2. **拼接策略**：将这些布局标记拼接在图像Token序列之后作为自回归目标。
  3. **训练目标**：定义了独立的Loss，即 $L = L_{img} + \alpha \cdot L_{text}$，实现视觉生成与文本渲染任务的解耦优化。
- **推理逻辑**：推理时仅输入Prompt，完全不依赖任何外部布局信息，保持了端到端的生成能力。

### 4. 方法对比分析
- **本质区别**：与需要额外布局控制（如Layout hints）的方法不同，本方法将布局监督转化为“隐式自回归预测”的一部分，训练后推理阶段无需任何额外负担。
- **创新贡献**：
  1. **TextGround4M数据集**：提供了大规模、高质量的prompt-grounded标注。
  2. **Layout-posting策略**：一种轻量级、无需架构改动的布局学习范式。
- **适用场景**：适用于任何基于自回归架构的T2I模型，特别是在需要精确布局控制的复杂文本合成场景。

### 5. 实验分析（精简版）
- **验证方法**：在Janus Pro 1B上进行微调，对比多种主流模型。
- **关键结果**：在TextGround-Bench（分层复杂度基准）上，该方法在Hard subset上展现出极高的鲁棒性，文本保真度提升显著。
- **优势**：训练时强监督，推理时零开销，模型架构通用。
- **局限**：目前尚未包含字号、字体风格等细粒度排版信息。

### 6. 实用指南
- **开源情况**：数据集已发布，推荐参考论文中的alignment算法细节。
- **实现细节**：建议设置 $\alpha=1$ 进行多目标加权；使用Post-Image顺序，这对于保持推理过程的一致性至关重要。
- **迁移可能**：该方案可直接扩展至视频生成领域（在空间坐标基础上增加时间维度预测），或用于其他需要精确语义定位的任务。

### 7. 总结
- **核心思想**：通过拼接坐标与内容标记，让模型在训练中习得文本的空间布局。
- **速记版pipeline**：
  1. **标注匹配**：将提示词与OCR检测到的文本框绑定。
  2. **序列构建**：将文本内容及其位置坐标转换为Token序列。
  3. **拼接训练**：把这些Token附加到图像数据后，进行自回归训练。
  4. **标准推理**：直接根据Prompt生成图像，不再输入任何布局辅助。

**Key Findings:**

- To address these issues, we introduce TextGround4M, a large-scale dataset of over 4 million prompt-image pairs, each annotated with span-level text grounded in the prompt and corresponding bounding boxes.
- Building on this, we propose a lightweight training strategy for autoregressive T2I models that appends layout-aware span tokens during training, without altering model architecture or inference behavior.
- In addition, we introduce two layout-aware metrics to address the long-standing lack of spatial evaluation in text rendering.
- Our results show that models trained on TextGround4M outperform strong baselines in text fidelity, spatial accuracy, and prompt consistency, highlighting the importance of fine-grained layout supervision for grounded T2I generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.24459v1)
- [arXiv](https://arxiv.org/abs/2604.24459v1)

---

