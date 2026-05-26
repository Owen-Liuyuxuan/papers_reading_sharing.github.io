time: 20260526

# Arxiv Computer Vision Papers - 2026-05-26

## Executive Summary

## 每日 Arxiv 计算机视觉论文执行摘要（2026-05-25）

**主要主题与趋势：**
本日论文集中反映出三个核心方向：**多模态大语言模型（VLM）的持续进化**（LLaVA-OneVision-2、ERNIE-Image、DRScaffold、MAGIC）、**3D重建与生成模型的效率突破**（Global SfM、Reinforcing Few-step Generators），以及**面向具体任务的模型微调与推理增强**（EXPO-FT、InstructSAM、Prism）。此外，基础组件如向量量化（Channel-wise VQ）也有新贡献。

**亮点与创新：**
- **LLaVA-OneVision-2**：作为新一代感知智能代表，可能整合了多模态对齐、推理与生成能力，是该领域最值得关注的演进。
- **InstructSAM**：将任意指令融入 SAM 分割框架，突破原有文本/点/框输入限制，有望推动通用视觉交互。
- **Global Structure-from-Motion**：将经典 SfM 与前馈网络结合，兼顾全局一致性与重建速度，对大规模场景重建有重要实践价值。

**新兴研究方向：**
- **强化学习微调与奖励分布匹配**（EXPO-FT、Reinforcing Few-step）正在将 RL 范式更系统地引入视觉-语言-动作模型及生成模型训练。
- **可复现基础设施**（Prism）与**数据集核心集选择**（MAGIC）表明领域正从“模型竞赛”转向“训练范式标准化”与“数据效率优化”。

**推荐精读论文：**
1. **LLaVA-OneVision-2** — 感知智能前沿，必读。
2. **InstructSAM** — 通用分割指令接口，创新性强。
3. **Global Structure-from-Motion** — 传统与现代融合案例，对3D CV研究者重要。
4. **DRScaffold** — 轻量 VLM 密集场景推理，适合资源受限场景应用。

---

## Table of Contents

1. [Global Structure-from-Motion Meets Feedforward Reconstruction](#2605.26103v1)
2. [EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models](#2605.25477v1)
3. [ERNIE-Image Technical Report](#2605.25347v1)
4. [Prism: A Plug-in Reproducible Infrastructure for Scalable Multimodal Continual Instruction Tuning](#2605.26110v1)
5. [Reinforcing Few-step Generators via Reward-Tilted Distribution Matching](#2605.26108v1)
6. [InstructSAM: Segment Any Instance with Any Instructions](#2605.26102v1)
7. [Channel-wise Vector Quantization](#2605.26089v1)
8. [DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language Models](#2605.26038v1)
9. [MAGIC: Multimodal Alignment & Grounding-aware Instruction Coreset for Vision-Language Models](#2605.26004v1)
10. [LLaVA-OneVision-2: Towards Next-Generation Perceptual Intelligence](#2605.25979v1)

---

## Papers

<a id='2605.26103v1'></a>
## [Global Structure-from-Motion Meets Feedforward Reconstruction](https://arxiv.org/abs/2605.26103v1)

**Authors:** Linfei Pan, Johannes Schönberge, Marc Pollefeys

**Published:** 2026-05-25

**Categories:** cs.CV

**Abstract:**

Structure-from-Motion -- the process of simultaneously estimating camera poses and 3D scene structure from a collection of images -- remains a central challenge in computer vision, with many open problems yet to be solved. Recent advances in feedforward 3D reconstruction have made significant strides in overcoming persistent failure cases of classical SfM methods, particularly in scenarios characterized by low texture, limited overlap, and symmetries. However, while feedforward approaches excel in these challenging conditions, they often face limitations regarding scalability, accuracy, or robustness, and typically fall short of classical methods in standard reconstruction settings. In this work, we systematically analyze these limitations and propose a new Structure-from-Motion pipeline by combining the respective strengths of classical and feedforward methods. Extensive experiments across multiple datasets show the benefits of our approach, achieving state-of-the-art results across a wide range of scenarios. We share our system as an open-source implementation at https://github.com/colmap/gluemap.

**Analysis:**

这是一篇关于Structure-from-Motion (SfM) 的高质量论文分析。

### 1. 摘要翻译
Structure-from-Motion (SfM)——即从图像集合中同时估计相机位姿和三维场景结构的过程——一直是计算机视觉中的一个核心挑战，且仍有许多悬而未决的问题。近年来，前馈（feedforward）式三维重建的进展在克服经典SfM方法的持续性失败案例方面取得了显著进步，特别是在低纹理、有限重叠和对称性特征明显的场景中。然而，尽管前馈方法在这些具有挑战性的条件下表现出色，但它们通常在可扩展性、准确性或鲁棒性方面面临局限，并且在标准重建设置中通常不如经典方法。在这项工作中，我们系统地分析了这些局限性，并提出了一种结合了经典方法和前馈方法各自优势的全新SfM流水线。跨多个数据集的广泛实验表明了我们方法的益处，在广泛的场景中实现了最先进的结果。我们将我们的系统作为开源实现分享在 https://github.com/colmap/gluemap。

### 2. 方法动机分析
*   **驱动力**：旨在融合前馈式模型在“困难场景”（如低纹理、稀疏重叠）下的局部推理能力，与经典SfM方法在“标准场景”下的鲁棒性、可扩展性及全局优化精度。
*   **痛点**：前馈方法通常受限于GPU内存（导致无法处理大规模数据集），且在标准数据下精度往往低于经典SfM；经典方法则在特征点稀疏或对称结构（如Doppelgangers）下极易导致重建失败或坍缩。
*   **研究假设**：通过将全局重建任务分解为局部“星型图”（Star Graphs）进行前馈推理，再利用全局运动平均（Global Motion Averaging）和增强型BA（Augmented BA）整合信息，能够兼顾鲁棒性与大规模扩展性。

### 3. 方法设计详解
*   **流水线**：
    1.  **图初始化**：利用SALAD进行大规模检索，结合Doppelgangers++过滤对称/非共视边，通过动态阈值构建鲁棒的局部星型图邻域。
    2.  **前馈局部推理**：以中心图像为基准，对每个星型图（中心及其邻居）进行独立的前馈重建（采用 $\pi^3$ 模型），获得局部相机位姿、深度图和关键点。
    3.  **全局运动平均**：整合所有局部星型图的结果，执行内在参数平均、旋转平均和相似度平均，计算全局相机位姿和尺度一致的深度图。
    4.  **增强型BA**：在传统BA中引入“虚拟轨道（Virtual Tracks）”，通过局部重建的深度图将采样像素反投影到相邻视角，形成虚拟对应关系，缓解传统BA因低纹理或弱重叠导致的欠约束问题。

### 4. 方法对比分析
*   **本质区别**：它不是简单的“模块替换”，而是将前馈模型作为“局部几何估计器”，嵌入到全局经典优化框架中。
*   **创新贡献**：提出“虚拟轨道（Virtual Tracks）”增强BA，有效解决了低重叠场景下的约束缺失；通过星型图分解实现了内存高效的大规模重建。
*   **适用场景**：极端挑战性场景（低重叠、低纹理、大规模）以及高精度标准场景。

### 5. 实验分析
*   **验证方法**：在ETH3D、IMC2021、CO3Dv2、SMERF和LaMAR等多个基准数据集上进行对比测试。
*   **关键结论**：在LaMAR等包含数万张图像的超大规模场景中，相比纯前馈方法（因OOM失败）或纯经典方法（因重叠不足坍缩），该方法均表现出最强的鲁棒性和精度。
*   **优势**：可扩展至万图规模，在极端场景下表现出极高的恢复成功率。
*   **局限**：依赖于前馈模型的局部重建质量，且暂不支持非针孔相机模型。

### 6. 实用指南
*   **开源地址**：https://github.com/colmap/gluemap
*   **实现细节**：关键参数如虚拟轨道采样比例（10%）和共视过滤阈值对于保持性能平衡至关重要。
*   **迁移可能**：其架构思想（前馈局部+全局优化）可直接迁移至其他SLAM或三维重建任务中，特别是那些依赖于深度学习算子但又受限于全局优化的一致性任务。

### 7. 总结
*   **核心思想**：前馈局部几何推理与经典全局优化的高效协同。
*   **速记版pipeline**：
    1. 过滤对称性，构建局部关联图；
    2. 分块进行神经网络推理；
    3. 融合局部信息，全局平滑位姿；
    4. 加入虚拟轨道，进行精细优化。

**Key Findings:**

- In this work, we systematically analyze these limitations and propose a new Structure-from-Motion pipeline by combining the respective strengths of classical and feedforward methods.
- Extensive experiments across multiple datasets show the benefits of our approach, achieving state-of-the-art results across a wide range of scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26103v1)
- [arXiv](https://arxiv.org/abs/2605.26103v1)

---

<a id='2605.25477v1'></a>
## [EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models](https://arxiv.org/abs/2605.25477v1)

**Authors:** Perry Dong, Kuo-Han Hung, Tian Gao, Dorsa Sadigh, Chelsea Finn

**Published:** 2026-05-25

**Categories:** cs.RO, cs.AI

**Abstract:**

The ability to efficiently and reliably learn new tasks has been a foundational challenge in robotics. Vision-Language-Action (VLA) models have demonstrated strong generalization across diverse manipulation tasks, yet pretrained policies consistently fall short of the reliability required for real-world deployment. Reinforcement learning (RL) fine-tuning offers a promising path to bridge this gap, but existing approaches either train from scratch without fully leveraging pretrained priors, or fine-tune VLAs without achieving the sample efficiency and success rates that practical deployment demands. We present EXPO-FT, a system for stable, sample-efficient RL finetuning of pretrained VLA policies that closes this gap. Our system solves a suite of challenging manipulation tasks, including routing string lights and inserting the plug to light it up, striking a pool ball into a pocket, and inserting a flower into a wine bottle, each requiring combinations of high precision, dynamic actions, and robustness to varied initial states. Our system achieves perfect task performance (30/30 successes) across all evaluated tasks within an average of 19.1 minutes of online robot data, outperforming both prior RL-from-scratch and VLA finetuning approaches. We release an open-source codebase with the aim of facilitating broader adoption of RL finetuning of VLA models in robotics.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我针对 **EXPO-FT** 这篇论文的分析如下：

### 1. 论文核心贡献总结
EXPO-FT 提出了一种针对视觉-语言-动作（VLA）模型的高效强化学习微调框架，旨在解决预训练模型在真实机器人任务中可靠性不足的问题。该系统通过极少的在线数据交互（平均仅需约 19 分钟），实现了对复杂操作任务的完美成功率，填补了通用预训练先验与特定任务部署精度需求之间的空白。

### 2. 关键创新与方法论
*   **利用预训练先验的深度整合**：与传统的从零开始训练（RL-from-scratch）或简单的微调不同，EXPO-FT 旨在最大化利用 VLA 模型已有的视觉感知与语义推理能力。
*   **高样本效率（Sample-Efficient）设计**：在强化学习中，样本效率通常是最大的瓶颈。该研究通过优化微调策略，能够在极短时间内实现从“泛化模型”到“特定专家模型”的跃迁，解决了机器人在线学习耗时过长、成本高昂的问题。
*   **动态与高精度任务的协同**：系统证明了其在需要高精度（插入）、动态性（击球）及稳健性（灯串布线）的复杂任务中的鲁棒性，这表明该方法在处理多模态序列建模时具备良好的稳定性。

### 3. 对领域的影响
*   **从“模型部署”向“实时迭代”的范式转变**：该研究证明了 VLA 模型并非部署后即固化，而是可以通过少量实时反馈迅速优化。这将极大地推动机器人从实验室环境走向复杂的家庭或工业现场。
*   **RL 与 VLA 的深度融合范式**：此前 RL 与 VLA 往往是两条路径，该论文确立了一种能够兼顾 VLA 泛化性和 RL 任务导向性的标准流程，为后续研究提供了可参照的技术框架。

### 4. 受益的相关领域与应用
*   **机器人操作（Manipulation）**：如家庭自动化、仓储物流中的精密装配等需要实时微调的任务。
*   **具身智能（Embodied AI）**：对于需要处理视觉不确定性并执行复杂动力学动作的智能体，该方法具有直接的参考价值。
*   **在线适应学习（Online Adaptive Learning）**：在环境发生变化（如光照变动、物体位置偏移）时，机器人如何快速调整其行为策略。

### 5. 可推断的局限性
*   **任务类型的扩展性**：虽然文中演示了字符串路由、击球等复杂任务，但未说明对于语义层面极其模糊或长时序复杂指令的泛化边界。
*   **计算开销与推理延迟**：尽管样本效率高，但在线微调过程中的计算资源消耗（如 GPU 的实时训练负载）未提及；如果模型过大，在边缘设备上进行实时微调可能面临算力瓶颈。
*   **环境变化的稳健性**：虽然强调了对初始状态的稳健性，但在面对完全未知的新物体或未见过的环境材质时，这种微调是否会引发“灾难性遗忘”（Catastrophic Forgetting）尚未可知。

---
**专家点评：**
这篇论文的趣味性在于它直接挑战了“VLA 模型是否需要大规模离线数据训练”这一现状。通过引入 RL 微调，它将 VLA 从“阅读者”变为了真正的“执行者”。对于 CV 领域而言，这意味着**视觉特征提取器不再仅仅是用于分类或检测，而是被转化为了一个动态的、可交互的控制闭环**，这是通向通用机器人技术的重要里程碑。

**Key Findings:**

- The ability to efficiently and reliably learn new tasks has been a foundational challenge in robotics.
- We present EXPO-FT, a system for stable, sample-efficient RL finetuning of pretrained VLA policies that closes this gap.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.25477v1)
- [arXiv](https://arxiv.org/abs/2605.25477v1)

---

<a id='2605.25347v1'></a>
## [ERNIE-Image Technical Report](https://arxiv.org/abs/2605.25347v1)

**Authors:** Jiaxiang Liu, Zhida Feng, Pengyu Zou, Zhenyu Qian, Tianrui Zhu, Jun Xia, Yuehu Dong, Yanzheng Lin, Honglin Xiong, Anqi Chen, Yunpeng Ding, Jinghui Duan, Lin Gao, Chao Han, Tiechao He, Jiakang Hu, Ranjun Hua, Xueming Jiang, Qingli Kong, Yuting Lei, Tianyu Li, Yunlin Liu, Changling Liu, Yaxin Liu, Yi Liu, Xuguang Liu, Xiaolong Ma, Yan Pan, Yiran Ren, Nan Sheng, Yu Sun, Siyang Sun, Yixiang Tu, Yang Wan, Huanai Wang, Siqi Wang, Yang Wu, Youzhi Yang, Xiaowen Yang, Jianwen Yang, Yehua Yang, Quanwen Zhang, Xinmin Zhang, Haoxin Zhang, Xiang Zhang, Jun Zhang, Qian Zhang, Qiao Zhao, Qi Zhou

**Published:** 2026-05-25

**Categories:** cs.CV, cs.LG

**Abstract:**

We introduce ERNIE-Image, an open-source text-to-image generation model built upon an 8B single-stream DiT architecture. ERNIE-Image aims to bridge the gap between current open-source models and leading closed-source systems through more effective mining of large-scale pre-training data and improved supervision quality throughout training. During pre-training, we adopt a bottom-up data construction pipeline that combines fine-grained image categorization, rich caption annotation, aesthetic assessment, and hierarchical sampling. This strategy reduces data noise while preserving long-tail concepts and detailed real-world knowledge, providing a stronger foundation for complex generation tasks. In the post-training stage, we use a top-down data construction pipeline for high-demand scenarios, diversify prompt annotations to better match real user inputs, and apply a stabilized DPO strategy to align the model with human aesthetic preferences. We further train ERNIE-Image-Turbo for efficient 8-NFE generation and propose MT-DMD to mitigate capability drift during distillation. To make the model easier to use in practical scenarios, we equip it with a lightweight Prompt Enhancer that expands concise user intents into structured visual descriptions. In addition, we develop ERNIE-Image-Aes, an industrial-grade aesthetic model, together with ERNIE-Image-Aes-1K, a human-annotated benchmark for realistic aesthetic evaluation. Extensive qualitative and quantitative experiments show that ERNIE-Image achieves leading performance among open-source models and approaches top-tier commercial models in instruction following, text rendering, and aesthetic quality. We release the trained models and aesthetic resources to facilitate further academic research and technical progress in the AIGC community.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对《ERNIE-Image Technical Report》的分析如下：

### 1. 论文主要贡献概述
ERNIE-Image 是一个基于 8B 参数、单流（Single-stream）DiT 架构的开源文本生成图像模型，旨在通过精细化的数据工程和强化对齐策略，缩小开源模型与闭源顶尖模型之间的差距。该研究通过引入系统化的数据构建流程（底层的预训练数据清洗与顶层的后训练对齐）以及蒸馏技术（MT-DMD），在保证模型生成效率（ERNIE-Image-Turbo）的同时，显著提升了指令跟随和美学表现。

### 2. 关键创新与方法论
*   **全链路数据工程（Data Pipeline Construction）：** 
    *   **预训练阶段（Bottom-up）：** 采用细粒度分类、语义描述增强、美学评估及分层采样，有效地去噪同时保留了长尾知识，解决了大规模数据中常见的质量与多样性矛盾。
    *   **后训练阶段（Top-down）：** 针对复杂场景进行提示词（Prompt）多元化增强，并结合**稳定化 DPO（Stabilized DPO）**策略，使模型输出更符合人类审美倾向。
*   **高效蒸馏与推理优化：** 提出 **MT-DMD（Mitigating Capability Drift in Distillation）**，专门用于解决知识蒸馏过程中常见的模型能力漂移问题，配合 Turbo 版本实现了 8-NFE（仅需 8 步推理）的高效生成。
*   **配套工具链：** 引入轻量级 **Prompt Enhancer**（语义扩展）和工业级美学评价模型 **ERNIE-Image-Aes**，配合 1K 人工标注美学基准，构建了一套完整的评估与提升闭环。

### 3. 对领域的潜在影响
*   **架构范式的验证：** 验证了 8B 规模的单流 DiT 在高质量数据工程加持下，能够触及当前最先进商业模型的性能边界，为开源社区提供了高性价比的基准参考。
*   **蒸馏范式的改良：** MT-DMD 的提出为解决“蒸馏模型往往不如教师模型”的性能损耗问题提供了新的技术路线。
*   **美学评估的标准化：** 其发布的工业级美学模型和基准（ERNIE-Image-Aes-1K），有望成为评估图像生成美学质量的新行业标准。

### 4. 受益的相关领域与应用
*   **内容创作与创意产业：** 高质量的美学输出和高效推理使其直接适用于数字艺术、海报设计和广告创作场景。
*   **交互式设计：** 内置的 Prompt Enhancer 极大地降低了用户的交互门槛，使普通用户能通过简单的意图描述获得结构化的视觉生成。
*   **自动化资产生成：** 其优秀的文本渲染和指令遵循能力，使其在游戏资产生成、UI 原型图设计等需要高度受控的任务中具备实用价值。

### 5. 潜在局限性（基于摘要分析）
*   **计算资源需求：** 尽管模型通过蒸馏实现了 8-NFE 高效推理，但 8B 的模型基底对边缘侧（Edge Device）推理仍有一定门槛，尚未触及移动端部署的极致轻量化。
*   **通用泛化限制：** 虽然强调了长尾知识的保留，但所有经过高度对齐和美学优化后的模型，通常在面临极具艺术实验性或非主流审美的“非典型”生成任务时，可能会出现某种程度的风格同质化（Model Collapse / Mode Covering）。
*   **依赖特定的 Prompt 增强：** 模型的高度表现严重依赖于配套的 Prompt Enhancer，这意味着如果用户直接输入未经优化的原始指令，模型原始的指令遵循能力与“增强后”的表现可能存在较大差距（即模型本身对原始指令的鲁棒性仍有待观察）。

---
**专家点评：**
ERNIE-Image 的核心价值在于它清晰地展示了**“数据工程优先（Data-Centric AI）”**的成功实践。在当前生成式模型架构日益趋同的背景下，通过这种精细化的数据清洗与后处理闭环来压榨模型潜力，是当前工业界提升 SOTA 性能最切实有效的途径。对于学术界而言，其开源的美学评价工具链比模型本身可能产生更深远的影响。

**Key Findings:**

- We introduce ERNIE-Image, an open-source text-to-image generation model built upon an 8B single-stream DiT architecture.
- In addition, we develop ERNIE-Image-Aes, an industrial-grade aesthetic model, together with ERNIE-Image-Aes-1K, a human-annotated benchmark for realistic aesthetic evaluation.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.25347v1)
- [arXiv](https://arxiv.org/abs/2605.25347v1)

---

<a id='2605.26110v1'></a>
## [Prism: A Plug-in Reproducible Infrastructure for Scalable Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2605.26110v1)

**Authors:** Jun-Tao Tang, Yu-Cheng Shi, Zhen-Hao Xie, Da-Wei Zhou

**Published:** 2026-05-25

**Categories:** cs.LG, cs.CL, cs.CV

**Abstract:**

Multimodal Large Language Models (MLLMs) achieve versatility by reformulating diverse tasks into a unified instruction-following framework via instruction tuning. However, real-world deployment requires continuous adaptation to emerging tasks, motivating Multimodal Continual Instruction Tuning (MCIT). Despite its growing importance, current MCIT research is hindered by severe engineering bottlenecks. Existing methods are typically implemented by directly modifying the base MLLM codebase, which imposes substantial implementation overhead and yields method-specific architectures that severely limit code reuse and fair comparison. To address this, we introduce Prism, a plug-in reproducible codebase specifically designed for scalable MCIT research. It separates algorithmic development from the backbone implementation via a lightweight plugin registration mechanism, enabling new strategies to be integrated as independent plugins without modifying the underlying MLLM codebase, thereby eliminating structural fragmentation and accelerating method development. Prism natively supports widely used large-scale training pipeline, thereby enabling reproducible and scalable MCIT experimentation. Code is available at https://github.com/LAMDA-CL/Prism.

**Analysis:**

### 1. 摘要翻译
多模态大模型（MLLMs）通过指令微调将多种任务转化为统一的指令遵循框架，从而实现了卓越的通用性。然而，实际部署需要模型在动态环境中对不断涌现的任务进行持续适应，这催生了多模态持续指令微调（MCIT）的研究。尽管其重要性日益凸显，但当前的MCIT研究受限于严重的技术工程瓶颈。现有方法通常直接修改基础MLLM的代码库，导致实现开销巨大，并产生了特定于方法的架构，严重限制了代码的复用与公平对比。为解决这一问题，我们推出了PRISM，一个专为可扩展MCIT研究设计的、插件式可复现代码库。它通过轻量级的插件注册机制将算法开发与骨干模型实现分离开来，使新策略能够以独立插件的形式集成，无需修改底层的MLLM代码库。这消除了结构化碎片化，加速了方法开发。PRISM原生支持广泛使用的大规模训练流水线，从而实现了可复现且可扩展的MCIT实验。代码地址：https://github.com/LAMDA-CL/Prism。

### 2. 方法动机分析
*   **驱动力**：旨在为多模态持续指令微调（MCIT）提供一个标准化的研发基础设施，解决目前研究中“代码重复造轮子”和“难以公平对比”的问题。
*   **现有痛点**：现有框架通过直接修改核心代码来实现持续学习算法，导致架构高度碎片化（每个方法都有完整代码副本），缺乏统一的训练流水线，难以兼容DeepSpeed等大规模并行训练技术。
*   **核心直觉**：通过“解耦”思想，将算法逻辑（方法）与模型架构（骨干）、训练配置（流水线）完全分离，实现“插拔式”的研究范式。

### 3. 方法设计详解
*   **架构解耦**：PRISM的核心在于其注册机制。它将整个系统拆分为四个独立模块：
    1.  **Backbone（骨干）**：处理底层LLM和多模态投影层，支持插件式替换。
    2.  **Method（方法）**：独立的算法插件（如Replay-LoRA, SAME等），包含具体的参数更新逻辑。
    3.  **Benchmark（基准）**：统一的数据加载与评估hooks。
    4.  **Infra（基础设施）**：处理分布式计算（DeepSpeed）、梯度检查点等。
*   **插件机制**：通过 `@CLMethodFactory.register()` 装饰器，任何新算法只需遵循统一的PEFT接口，存放在 `method/<name>/` 目录下即可自动被框架调用，无需改动主程序。
*   **统一评估**：内置了针对持续学习场景的核心指标（Last Accuracy, Average Accuracy, Forgetting Measure），确保不同方法在相同任务序列、相同预设下进行对比。

### 4. 方法对比分析
*   **本质区别**：从“一体化代码仓库”转向“插件化组件架构”。对比CoIN和MCITlib，PRISM是唯一同时做到 unified backbone design 和 large-scale experiment support 的 MCIT 框架。
*   **创新贡献**：提供了一个高度模块化的训练模板，将持续学习的复杂性（如灾难性遗忘的缓解、数据存储与采样）封装在独立的Wrappers中。
*   **适用场景**：适用于任何需要对比多种持续学习策略的多模态模型研究，特别适合在大规模数据集序列上进行长期性能压测。

### 5. 实验分析
*   **验证方法**：在UCIT（轻量级验证）和TriGap（长序列、高难度验证）两个基准上，利用LLaVA-v1.5-7B进行验证。
*   **关键结论**：结构化方法（如SAME, DISCO）在保留旧知识方面优于简单的LoRA微调；在长任务序列中，参数分配策略对最终模型性能影响巨大。
*   **优势**：极大地降低了实验配置复杂度，显著提升了跨论文方法复现的便捷性。
*   **局限**：目前尚未涵盖MCIT领域的所有前沿算法，仍需社区持续贡献。

### 6. 实用指南
*   **开源地址**：https://github.com/LAMDA-CL/Prism。
*   **实现建议**：
    *   **超参数**：严格遵循 `config/` 下的配置文件。注意对于MoE类方法，LoRA秩（rank）需要被专家数量整除。
    *   **迁移**：如需添加新方法，在 `method/` 下创建新目录，实现 `integration.py` 并向 `CLMethodFactory` 注册。
    *   **资源**：建议使用DeepSpeed Zero-2/3以适应大规模持续学习任务的内存开销。

### 7. 总结
*   **核心思想**：通过插件式解耦，将MCIT研究标准化、标准化、可复现。
*   **速记版Pipeline**：
    1. 配置基准任务序列（固定顺序）。
    2. 调用注册的算法插件注入PEFT模块。
    3. 运行统一训练流水线（支持分布式）。
    4. 自动计算持续学习评估指标。

**Key Findings:**

- To address this, we introduce Prism, a plug-in reproducible codebase specifically designed for scalable MCIT research.
- It separates algorithmic development from the backbone implementation via a lightweight plugin registration mechanism, enabling new strategies to be integrated as independent plugins without modifying the underlying MLLM codebase, thereby eliminating structural fragmentation and accelerating method development.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26110v1)
- [arXiv](https://arxiv.org/abs/2605.26110v1)

---

<a id='2605.26108v1'></a>
## [Reinforcing Few-step Generators via Reward-Tilted Distribution Matching](https://arxiv.org/abs/2605.26108v1)

**Authors:** Yushi Huang, Xiangxin Zhou, Ruoyu Wang, Chi Zhang, Jun Zhang, Tianyu Pang

**Published:** 2026-05-25

**Categories:** cs.CV

**Abstract:**

Recent advances in few-step diffusion distillation have enabled efficient image generation, yet aligning these models with human preferences remains challenging. We propose Reward-Tilted Distribution Matching Distillation (RTDMD), a two-stage framework that unifies distribution matching distillation with reward-guided reinforcement learning for few-step flow generators. We show that minimizing the KL divergence to a reward-tilted teacher distribution naturally decomposes into a distribution matching term and a reward maximization term. In the first stage, we introduce Ambient-Consistent Distribution Matching Distillation (AC-DMD), which performs subinterval-wise distribution matching and augments the fake score objective with a consistency regularizer to help the fake score model track the shifting generator distribution under limited updates. In the second stage, we jointly optimize both terms: for the reward maximization term, we derive a hybrid policy gradient that combines a GRPO-style estimator for the stochastic intermediate transitions with direct reward backpropagation through the deterministic final step, and further introduce step-subset GRPO (SubGRPO) to reduce variance. Experiments on SD3, SD3.5, and FLUX.2 demonstrate that RTDMD establishes new state-of-the-art results across preference, aesthetic, and compositional metrics with only 4 inference steps, outperforming previous few-step text-to-image generation methods. Code and models are available at https://github.com/Harahan/RTDMD.

**Analysis:**

## 1. 摘要翻译
近期扩散模型蒸馏技术的进步实现了高效图像生成，但将这些模型与人类偏好对齐仍面临挑战。我们提出了**奖励引导分布匹配蒸馏（RTDMD）**，这是一个将分布匹配蒸馏与奖励引导强化学习（RL）统一起来的两阶段框架，专门用于少步流生成器。我们证明，最小化向奖励倾斜的教师分布的KL散度，可以自然分解为分布匹配项和奖励最大化项。在第一阶段，我们引入**环境一致性分布匹配蒸馏（AC-DMD）**，执行子区间分布匹配，并增强伪分数目标（fake score objective）的一致性正则化，以帮助伪分数模型在有限更新下追踪动态变化的生成器分布。在第二阶段，我们联合优化上述两项：对于奖励最大化项，我们推导了一种**混合策略梯度**，它将用于随机中间转换的GRPO式估计器与通过确定性最终步骤的直接奖励反向传播相结合，并进一步引入**步子集GRPO（SubGRPO）**以降低方差。在SD3、SD3.5和FLUX.2上的实验表明，RTDMD仅需4步推理即可在偏好、审美和构图指标上达到新的最优结果。

## 2. 方法动机分析
- **驱动力**：在少步（Few-step）图像生成中，既要保持教师模型的生成质量（分布匹配），又要实现对齐人类偏好（奖励优化），且由于生成过程是多步采样的，需要一个统一的框架处理这两者的冲突。
- **现有痛点**：
    1. **冷启动困难**：生成器分布在训练过程中不断偏移，伪分数模型难以在有限计算下准确跟踪目标。
    2. **奖励优化不适配**：现有RL方法要么仅优化确定性最终步骤，要么仅优化中间随机步骤，未能兼顾混合采样的动力学特性。
- **研究假设**：通过向奖励倾斜的教师分布最小化KL散度，可以将蒸馏与RL统一在同一个目标函数下，通过分阶段优化可获得更稳定的收敛。

## 3. 方法设计详解
- **流程pipeline**：
    1. **冷启动阶段 (AC-DMD)**：在特定时间子区间$[t_k, 1]$进行训练，而非全区间$[0, 1]$。引入**一致性正则化**（Consistency Regularizer），强迫不同时间步对同一干净样本的预测保持一致，降低伪分数模型估计方差。
    2. **强化阶段 (RTDMD)**：联合优化分布匹配损失与奖励最大化损失。
- **关键技术**：
    *   **混合策略梯度 (Hybrid Policy Gradient)**：将奖励梯度分为两部分：一是中间随机步骤的梯度，利用**SubGRPO**估计；二是确定性最后一步的梯度，直接通过Gθ进行反向传播，利用了确定性操作的可微分性。
    *   **SubGRPO**：对$K-1$个采样步骤进行子集采样，仅对子集内的步骤引入独立噪声，其余共享噪声，显著降低了信噪比，实现了高效的信用分配。

## 4. 方法对比分析
- **本质区别**：传统方法将蒸馏与RL分开进行或简单拼接，RTDMD从KL散度分解出发，实现了两者的理论统一。
- **创新点**：
    1. **AC-DMD**：解决了少步生成在训练过程中伪分数模型无法同步漂移的问题。
    2. **Hybrid Policy Gradient**：针对流模型混合动力学特性，设计了兼顾随机性和确定性的梯度优化方案。

## 5. 实验分析
- **关键结论**：在SD3-M和FLUX.2 4B上，RTDMD在4步采样下超越了强基线模型，部分指标甚至超过了原版9B 50步模型。
- **优势**：极高的采样效率（4步）、卓越的审美与对齐能力。
- **局限**：目前主要针对文本生成图像，对视频生成或复杂编辑任务的temporal consistency尚需深入探索。

## 6. 实用指南
- **开源地址**：`https://github.com/Harahan/RTDMD`
- **实现细节**：
    *   核心超参数：$\eta=0.9$ (CPS采样控制)，$\beta=1.0$ (奖励倾向系数)，$M=2$ (SubGRPO步长子集大小)。
    *   训练策略：分两阶段训练，第一阶段冷启动利用AC-DMD保证基础分布匹配，第二阶段联合优化。
- **迁移性**：该混合策略梯度方案可直接适配任意具备“确定性最后一步+随机中间步”特征的流式生成模型。

## 7. 总结
- **核心思想**：通过KL散度分解，统一了分布匹配蒸馏与混合动力学的奖励强化学习。
- **速记pipeline**：
    1. 预训练教师模型蒸馏冷启动。
    2. 伪分数模型一致性正则化训练。
    3. 采样轨迹子集化奖励优化。
    4. 混合策略梯度更新生成器参数。

**Key Findings:**

- We propose Reward-Tilted Distribution Matching Distillation (RTDMD), a two-stage framework that unifies distribution matching distillation with reward-guided reinforcement learning for few-step flow generators.
- We show that minimizing the KL divergence to a reward-tilted teacher distribution naturally decomposes into a distribution matching term and a reward maximization term.
- In the first stage, we introduce Ambient-Consistent Distribution Matching Distillation (AC-DMD), which performs subinterval-wise distribution matching and augments the fake score objective with a consistency regularizer to help the fake score model track the shifting generator distribution under limited updates.
- Experiments on SD3, SD3.5, and FLUX.2 demonstrate that RTDMD establishes new state-of-the-art results across preference, aesthetic, and compositional metrics with only 4 inference steps, outperforming previous few-step text-to-image generation methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26108v1)
- [arXiv](https://arxiv.org/abs/2605.26108v1)

---

<a id='2605.26102v1'></a>
## [InstructSAM: Segment Any Instance with Any Instructions](https://arxiv.org/abs/2605.26102v1)

**Authors:** Yuqian Yuan, Wentong Li, Zhaocheng Li, Yutong Lin, Juncheng Li, Siliang Tang, Jun Xiao, Yueting Zhuang, Wenqiao Zhang

**Published:** 2026-05-25

**Categories:** cs.CV

**Abstract:**

In this paper, we introduce InstructSAM, a unified and streamlined framework designed for multi-instance segmentation under arbitrary instructions. We formulates instruction-driven instance segmentation as a set-structured query prediction problem and propose an explicit reasoning-to-instance query interface that elegantly bridges a vision-language model (VLM) and SAM3. Specifically, a bank of learnable instance queries is injected into the VLM and contextualized with instruction and visual information, enabling each query to serve as an instance-aware slot. A hybrid-attention mechanism further promotes interaction among these queries, visual tokens, and instruction tokens, improving instance enumeration and reducing duplicate predictions. The resulting LLM-conditioned queries are projected into SAM3's detector query space to drive accurate multi-instance segmentation in a single forward pass. This design equips SAM3 with high-level instruction understanding, compositional reasoning, and instance-level set prediction without modifying its core architecture. To support training and evaluation, we further construct Inst2Seg, a high-quality and large-scale instruction-based instance segmentation dataset and benchmark that couples free-form instructions with instance-level masks. Extensive experiments show that only 2B-scale InstructSAM achieves strong results across complex instruction-driven and phrase-level referring segmentation benchmarks, outperforming prior end-to-end methods and SAM3's agentic pipeline while enabling efficient single-pass multi-instance prediction.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **InstructSAM** 的论文进行了深入分析。以下是详细评估：

### 1. 主要贡献总结
InstructSAM 提出了一种统一且高效的框架，旨在实现基于任意指令的多实例分割（Multi-instance Segmentation）。通过将指令驱动的分割建模为集合预测问题，并设计了一种推理到实例的查询接口，该方法成功将视觉语言模型（VLM）与 SAM3 的分割能力深度结合，实现了“单次前向传播”即可完成复杂场景下的多实例定位与分割。

### 2. 核心创新与方法论
该论文的创新点在于构建了指令与分割之间的“语义桥梁”，具体表现为：
*   **推理到实例的查询接口 (Reasoning-to-Instance Query Interface)：** 论文将一组可学习的实例查询注入 VLM，使查询能够根据指令上下文动态调整，从而实现对目标的精准“聚焦”。
*   **混合注意力机制 (Hybrid-attention Mechanism)：** 通过增强查询向量、视觉 Token 和指令 Token 之间的交互，有效优化了多目标识别过程，减少了预测重复，提升了实例枚举的准确性。
*   **架构的轻量化与兼容性：** 该框架在不修改 SAM3 核心架构的前提下，通过特征投影将 LLM 生成的查询直接映射到 SAM3 的检测器空间，实现了“即插即用”式的高级指令理解能力。
*   **Inst2Seg 数据集：** 填补了大规模指令式实例分割训练数据的空白，为该领域设立了新的评估基准。

### 3. 对领域的潜在影响
*   **范式转变：** 将从“Prompting”（如点击、框选）转向“Instruction-driven”（如“分割图中所有的红色车辆”），显著提升了计算机视觉系统在复杂自然语言交互下的可用性。
*   **效率突破：** 相比于传统的基于 Agent 的反复调用 SAM 的方式，单次前向传播（Single-pass）的方法大幅降低了推理时延，使实时复杂场景分析成为可能。
*   **推理能力引入：** 该方法证明了 VLM 的逻辑推理能力可以被转化为像素级的语义分割能力，为具身智能（Embodied AI）提供了核心的感知支撑。

### 4. 受益的相关领域与应用
*   **具身智能 (Embodied AI)：** 机器人能够理解更复杂的自然语言指令（例如“把桌子上所有没洗过的盘子收起来”），并精确地对这些实例进行像素级分割。
*   **自动驾驶：** 对复杂交通指令（如“避开所有路边的违停车辆”）的实时语义理解。
*   **医疗影像分析：** 通过复杂的文本报告（而非简单的坐标点）自动提取多病灶区域。
*   **增强现实 (AR)：** 提升人机交互过程中对特定类别物体的即时识别与遮罩生成效率。

### 5. 可推断的潜在局限性
*   **对 VLM 的依赖性：** 若底层 VLM 的逻辑理解能力出现偏差，会直接导致分割结果出现严重的功能性错误。
*   **泛化挑战：** 尽管提出了 Inst2Seg 数据集，但指令的组合无穷无尽，模型对于罕见组合指令（Compositional Reasoning）的鲁棒性仍有待大规模实测。
*   **多实例冲突问题：** 虽然采用了混合注意力机制，但在目标极其密集或存在重叠遮挡的情况下，如何避免实例间的特征干扰（Feature interference）仍是潜在的瓶颈。
*   **算力门槛：** 尽管实现了单次传播，但包含 LLM 和 SAM3 的庞大参数量在嵌入式或资源受限的边缘设备上部署时，依然面临巨大的优化挑战。

---
**总结：** 
InstructSAM 的亮点在于其**“逻辑驱动分割”**的思路，它不再仅仅将 SAM 视为一个被动的分割引擎，而是赋予了它基于复杂语义推理的主动感知能力。如果该架构能实现高效的蒸馏或量化，它极有可能成为下一代交互式视觉感知的标配组件。

**Key Findings:**

- In this paper, we introduce InstructSAM, a unified and streamlined framework designed for multi-instance segmentation under arbitrary instructions.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26102v1)
- [arXiv](https://arxiv.org/abs/2605.26102v1)

---

<a id='2605.26089v1'></a>
## [Channel-wise Vector Quantization](https://arxiv.org/abs/2605.26089v1)

**Authors:** Wei Song, Tianhang Wang, Yitong Chen, Tong Zhang, Zuxuan Wu, Ming Li, Jiaqi Wang, Kaicheng Yu

**Published:** 2026-05-25

**Categories:** cs.CV, cs.AI

**Abstract:**

We present Channel-wise Vector Quantization (CVQ), a novel image tokenization paradigm that replaces patch-wise tokens with channel-wise tokens. Unlike conventional vector quantization, which assigns a discrete token to each patch feature vector, CVQ quantizes each channel of the feature map. This formulation represents an image as discrete levels of visual details, rather than as a grid of spatial patches. Based on CVQ, we introduce a new visual autoregressive framework with "next-channel prediction". Instead of rendering images patch by patch in raster order, our Channel-wise Autoregressive (CAR) model predicts image channels sequentially, producing progressively enriched visual details. Specifically, it first sketches global structure and then refines fine-grained attributes, akin to a human artist's workflow. Empirically, we show that: (1) CVQ achieves 100% codebook utilization with a 16K+ codebook size without any bells and whistles, and substantially improves reconstruction quality over conventional VQ; and (2) CAR attains a DPG score of 86.7 and a GenEval score of 0.79, demonstrating strong effectiveness for text-to-image generation.

**Analysis:**

以下是对论文《Channel-wise Vector Quantization》的深入分析：

### 1. 摘要翻译
我们提出了通道维度矢量量化（CVQ），这是一种新型图像标记化范式，用通道级标记替代了传统的块级（patch-wise）标记。与传统的矢量量化不同（传统方法将离散标记分配给每个块特征向量），CVQ量化特征图的每个通道。这种形式将图像表示为具有不同视觉细节水平的离散序列，而非空间块网格。基于CVQ，我们引入了一种名为“下一通道预测”（next-channel prediction）的视觉自回归框架。我们的通道自回归（CAR）模型不按栅格扫描顺序逐块渲染图像，而是顺序预测图像通道，从而生成逐步丰富视觉细节的内容。具体而言，它先勾勒整体结构，再细化细节，模拟了人类艺术家的创作流程。实验表明：(1) CVQ在无需特殊技巧的情况下，实现了16K+规模下100%的代码本利用率，且重构质量远超传统VQ；(2) CAR在文本生成图像任务中表现出色，在GenEval和DPG指标上达到了领先水平。

### 2. 方法动机分析
*   **痛点**：传统VQ存在两大核心瓶颈：
    1.  **代码本利用率不足（Codebook Collapse）**：由于图像块的冗余性，训练中大量代码本条目无法得到有效更新，导致严重信息丢失。
    2.  **空间顺序失配**：将2D图像块强行展平为1D序列进行自回归建模，破坏了图像原生的空间依赖，限制了推理效果。
*   **研究假设**：图像的语义信息并非均匀分布于空间块，而是隐含在特征通道中。通过“通道级”量化，可以将图像解耦为从全局结构到局部细节的序列，从而使自回归建模更符合从粗到细的创作逻辑。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **特征编码**：输入图像 $I$ 经过编码器得到特征图 $Z \in \mathbb{R}^{h \times w \times c}$。
    2.  **通道量化 (CVQ)**：不再对每个 $1 \times 1 \times c$ 的点进行量化，而是将每个通道 $Z^{(k)} \in \mathbb{R}^{h \times w \times 1}$ 看作一个向量，直接在通道代码本中搜索最近邻。
    3.  **自回归生成 (CAR)**：将图像视为 $c$ 个通道标记的序列。模型按顺序预测第 $1, 2, \dots, c$ 个通道。
    4.  **嵌套通道 Dropout**：为了让模型学习到层次化顺序，在训练时随机保留前 $c_{keep}$ 个通道，其余置零，强迫模型先输出全局结构。
*   **算法本质**：将复杂的2D空间生成问题，转化为基于通道信息的“下一通道预测”问题，直接利用代码本学习到的通道先验信息进行构建。

### 4. 方法对比分析
*   **本质区别**：传统VQ量化的是“空间位置”，CVQ量化的是“通道特征”。
*   **创新贡献**：彻底解决了VQ的代码本坍塌问题，实现了接近100%的利用率；同时为视觉生成提供了一种自然的空间层次化序列，规避了栅格扫描带来的空间依赖割裂。
*   **适用场景**：极高压缩率、高质量的图像生成任务，以及对训练稳定性要求较高的场景。

### 5. 实验分析
*   **关键结果**：在ImageNet-1K上，CVQ在1024 Token预算下，rFID仅为0.88（传统VQ通常大于1.0），代码本利用率达100%。在文本生成图像任务中，CAR模型在GenEval基准上大幅领先现有主流AR模型。
*   **优势**：训练极其稳健，无须复杂Trick；生成图像呈现出明显的从粗线条到精细纹理的层次性。
*   **局限**：目前主要针对固定分辨率特征图，虽然论文提出了变量分辨率扩展方案，但仍需额外的重采样模块。

### 6. 实用指南
*   **开源情况**：代码已开源（详见论文附带的Github链接）。
*   **实现细节**：
    *   **Nested Dropout**：这是实现“从粗到细”生成序列的关键，训练时必须加入概率 $\alpha$ 的随机通道截断。
    *   **架构一致性**：确保在比较实验中，CVQ与传统VQ的特征维度总和保持一致，以保证公平性。
*   **迁移可能**：可直接替换现有VQGAN架构的量化层；对于视频生成任务，可以类推将时间维度或帧间差异量化为通道级标记。

### 7. 总结
*   **核心思想**：将图像量化单位从空间块转向通道，实现语义结构的层次化建模。
*   **速记版 Pipeline**：
    1. 提取图像多通道特征图。
    2. 对每个通道进行离散编码（CVQ）。
    3. 加入嵌套Dropout强制通道排序。
    4. 顺序预测通道序列以还原图像。

**Key Findings:**

- We present Channel-wise Vector Quantization (CVQ), a novel image tokenization paradigm that replaces patch-wise tokens with channel-wise tokens.
- Based on CVQ, we introduce a new visual autoregressive framework with "next-channel prediction".
- Empirically, we show that: (1) CVQ achieves 100% codebook utilization with a 16K+ codebook size without any bells and whistles, and substantially improves reconstruction quality over conventional VQ; and (2) CAR attains a DPG score of 86.7 and a GenEval score of 0.79, demonstrating strong effectiveness for text-to-image generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26089v1)
- [arXiv](https://arxiv.org/abs/2605.26089v1)

---

<a id='2605.26038v1'></a>
## [DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language Models](https://arxiv.org/abs/2605.26038v1)

**Authors:** Xinrui Shi, Kai Liu, Ziqing Zhang, Jianze Li, Anqi Li, Yulun Zhang

**Published:** 2026-05-25

**Categories:** cs.CV, cs.AI

**Abstract:**

Lightweight vision-language models perform competitively on standard benchmarks yet fail systematically in dense-scene reasoning, where multiple objects, attributes, and relations must be jointly grounded and resolved through multi-step inference. Such capability is critical for real-world applications where models must reliably interpret cluttered environments. Yet existing training signals provide no explicit grounding between reasoning steps and the underlying visual entities and relations, leaving lightweight models free to generate fluent but visually unanchored reasoning chains. To address this gap, we first introduce DRBench, a benchmark of 14,573 questions across 2,943 images, organized into five task categories spanning three progressive reasoning layers. Building on DRBench, we propose DRScaffold, a supervised fine-tuning framework that decomposes the supervision target into four causally ordered stages, enforcing grounded reasoning without architectural modification. Experiments on three lightweight VLMs demonstrate substantial gains on DRBench while preserving or improving performance on general-purpose benchmarks. Notably, Qwen2.5-VL-3B trained with DRScaffold surpasses the frozen Qwen2.5-VL-32B on DRBench, demonstrating that structured supervision can substitute for a significant portion of model scale in dense-scene reasoning. Our code and models are available at https://github.com/irene-shi/DRScaffold .

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇关于 **DRScaffold** 的论文分析如下：

### 1. 主要贡献总结
该论文针对轻量级视觉语言模型（VLM）在处理复杂密集场景（Dense-Scene）推理时存在的“幻觉”和缺乏逻辑支撑问题，提出了 **DRBench** 基准测试集以及 **DRScaffold** 监督微调框架。通过将推理过程解耦为四个因果有序的阶段，该方法在不改变模型架构的前提下，显著提升了轻量级模型在复杂推理任务中的表现，甚至使 3B 参数模型在特定指标上超越了 32B 参数模型。

### 2. 关键创新与方法论
*   **DRBench 数据集设计**：引入了包含 14,573 个问题、覆盖三个递进推理层次的数据集，专门用于量化模型在复杂视觉环境下的推理能力，弥补了现有通用基准在此类细粒度任务上的不足。
*   **因果导向的微调策略 (DRScaffold)**：这是核心创新点。它将推理过程分解为四个阶段，强制要求模型在生成最终答案前，先完成对视觉实体（Entities）、属性（Attributes）及相互关系（Relations）的显式锚定（Grounding）。这种**“结构化监督”**取代了传统的端到端黑盒训练，确保了推理链条与视觉证据的强相关性。
*   **架构无关性**：该方法通过训练策略改进（而不是模型架构修改），使其具有高度的普适性，可以轻松迁移到各类轻量级 VLM 架构中。

### 3. 对领域的潜在影响
*   **以“训练质量”替代“模型规模”**：这是该研究最具启发性的结论——即通过结构化的监督信号，可以在一定程度上弥补模型参数量的不足。这对工业界极其重要，意味着在边缘设备上运行高性能推理模型成为可能。
*   **重新定义推理的可解释性**：DRScaffold 强制模型在生成最终结果前进行显式的中间推理，这不仅提升了准确性，还极大地增强了模型的透明度和可信度（即“所言即所见”）。

### 4. 受益的相关领域与应用
*   **自动驾驶与机器人技术**：在复杂且杂乱的交通/室内环境中，模型需要精准定位障碍物及其关系（如“谁在让行”、“谁处于视野盲区”），DRScaffold 的密集推理能力直接契合此类场景。
*   **医学图像分析**：在复杂的病理切片中，模型需要对多个病灶及其空间关联进行分步诊断，这种推理范式具有很高的应用价值。
*   **视觉辅助系统**：对于视觉受损人群，系统需要不仅能识别物体，还要解释复杂的场景逻辑，该研究能提升相关应用的理解深度。

### 5. 可推断的潜在局限性
*   **泛化能力的边际折损**：虽然论文提到在通用基准上保持了性能，但过度强化“因果推理逻辑”是否会降低模型在某些非结构化、开放式对话任务中的创造性或灵活性，仍需观察。
*   **推理延迟（Latency）**：由于框架强制要求分步推理（CoT），在实时性要求极高的场景中，相比于直接给出答案的“直觉型”模型，DRScaffold 可能产生更高的推理时延。
*   **数据依赖性**：该方法高度依赖于 DRBench 这种经过精细结构化标注的数据。如果面对领域外（OOD）的数据，模型是否依然具备类似的推理能力，取决于该监督框架对逻辑推理本身的泛化程度，而非仅仅是对特定任务的拟合。

**专家视角评价：**
这篇论文非常符合当前 AI 研究从“单纯堆砌参数”转向“精耕细作推理逻辑”的趋势。**“结构化监督”代替“模型参数”**的思路，为资源受限环境下的视觉模型部署开辟了一条清晰且高效的路径。

**Key Findings:**

- Building on DRBench, we propose DRScaffold, a supervised fine-tuning framework that decomposes the supervision target into four causally ordered stages, enforcing grounded reasoning without architectural modification.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26038v1)
- [arXiv](https://arxiv.org/abs/2605.26038v1)

---

<a id='2605.26004v1'></a>
## [MAGIC: Multimodal Alignment & Grounding-aware Instruction Coreset for Vision-Language Models](https://arxiv.org/abs/2605.26004v1)

**Authors:** Shristi Das Biswas, Kaushik Roy

**Published:** 2026-05-25

**Categories:** cs.CV, cs.CL

**Abstract:**

Instruction tuning of large vision-language models (LVLMs) increasingly depends on massive multimodal corpora, yet these datasets contain samples with substantial redundancy, low visual dependency, and highly imbalanced coverage of multimodal reasoning behaviors. As a result, uniform subsampling or naive score-based selection often yields suboptimal training subsets. We introduce MAGIC, a training-free, forward-only coreset selection method designed to construct compact yet behaviorally faithful subsets for multimodal instruction tuning. MAGIC is built on three intrinsic signals extracted from a pretrained VLM: Multimodal Gain, which measures the likelihood improvement obtained from visual input; Bridging Relevance, which captures the sharpness of answer-token grounding over visual tokens; and Skill-Neuron Signatures, which characterize the functional computation elicited by each sample via top-activated feed-forward neurons. MAGIC combines these signals in a three-stage pipeline: filtering low-gain examples, ranking candidates by a normalized quality objective, and performing bucket-wise budget allocation over discrete neuron signatures to preserve latent multimodal skill coverage. This formulation avoids backpropagation, auxiliary selector training, and expensive clustering in continuous activation spaces, while remaining efficient and easily deployable in existing VLMs. Across LLaVA-665K and Vision-Flan datasets, and transfer settings to large target models, LLaVA-1.5-7B and -13B, MAGIC consistently improves over strong baselines under matched 20% budgets: it achieves 100.3% relative performance to full finetuning on LLaVA-665K and 101.6% relative performance on Vision-Flan-186K, while yielding a 73.7% reduction in wall-clock run time.

**Analysis:**

### 1. 摘要翻译
大型视觉语言模型（LVLMs）的指令微调日益依赖大规模多模态语料，但这些数据集往往存在显著的冗余、视觉依赖度低以及多模态推理行为覆盖不平衡的问题。因此，均匀子采样或基于简单分数选择的方法往往会导致次优的训练子集。我们引入了 MAGIC，这是一种训练无关的、前向（Forward-only）的核集（Coreset）选择方法，旨在为多模态指令微调构建紧凑且行为忠实（Behaviorally faithful）的子集。MAGIC 基于从预训练 VLM 中提取的三个内在信号：多模态增益（Multimodal Gain，衡量视觉输入带来的似然改进）、桥接相关性（Bridging Relevance，捕捉答案标记在视觉标记上的定位锐度）以及技能神经元特征（Skill-Neuron Signatures，通过顶层激活的前馈神经元刻画样本诱导的功能计算）。MAGIC 通过三个阶段的流水线结合这些信号：过滤低增益样本、根据归一化质量目标对候选者进行排序，以及在离散神经元特征上进行桶式预算分配以保持潜在多模态技能的覆盖。该方案避免了反向传播、辅助选择器训练以及在连续激活空间中昂贵的聚类，同时在现有 VLM 中保持高效且易于部署。

---

### 2. 方法动机分析
*   **驱动力**：旨在以极小的计算成本（20%数据）构建一个具有代表性、高质量的多模态指令微调子集，以解决大规模语料数据冗余及覆盖偏差问题。
*   **现有方法痛点**：
    *   **计算昂贵**：需要梯度信息、反向传播或大规模计算开销。
    *   **效率低下**：在连续激活空间进行复杂聚类以寻求多样性，耗时且难扩展。
    *   **局部优化**：单分数指标倾向于只选出单一模态或狭窄任务风格的样本，导致下游多样性不足。
*   **研究假设**：通过提取预训练模型的“内在行为特征”（视觉工具利用率、视觉定位能力、特征激活模式），无需训练即可评估数据价值，从而构建分布均衡且高质量的核集。

---

### 3. 方法设计详解
**流程流水线（Pipeline）**：
1.  **特征提取（前向扫描）**：对训练集样本进行一次前向传播，提取三种指标：多模态增益（$g_i$）、桥接相关性（$b_i$）、技能神经元特征（$\phi_i$）。
2.  ** eligibility 过滤**：按 $g_i$ 过滤掉弱多模态样本（即视觉输入对预测贡献极小的样本）。
3.  **质量排序**：结合 $g_i$ 和 $b_i$ 计算综合质量分 $q_i$。
4.  **行为感知分组（核心创新）**：基于 $\phi_i$（即模型在前馈网络中激活的 top-k 神经元索引）将样本划分为不同的“技能桶”。
5.  **预算分配**：通过温度缩放（Temperature-scaled）的质量权重分配各桶预算，保证在每个行为模式下都有代表性采样。

**算法关键解释**：
*   **多模态增益**：本质是 $CE(\text{text}) - CE(\text{image+text})$，差值越大，证明视觉信号对答案预测的“信息增益”越高。
*   **技能神经元特征**：利用 FFN 层作为“知识记忆”的特性，将输入样本在每一层激活的特定神经元视为一个唯一的“行为指纹”，从而以离散方式实现多样性分布。

---

### 4. 方法对比分析
*   **本质区别**：与依赖外部监督或梯度下降的方法不同，MAGIC 纯粹依赖模型内部的“前向反馈信号”，属于纯训练无关（Training-free）方案。
*   **创新贡献**：将“技能神经元激活模式”直接作为一种可度量的行为多样性维度，替代了昂贵的嵌入空间（embedding space）聚类。
*   **适用场景**：适用于任何基于 Transformer 的 VLM，特别适合计算资源受限但需要进行大规模指令微调的场景。

---

### 5. 实验分析
*   **关键结果**：在 20% 子集下，MAGIC 在 LLaVA-665K 上达到 100.3% 的相对表现，在 Vision-Flan-186K 上达到 101.6%（超过全量微调）；端到端运行时减少约 74%。
*   **主要优势**：高通用性、极速训练（省去了辅助模型训练和特征空间聚类开销）、极佳的跨模型迁移能力。
*   **主要局限**：效果依赖于预训练 reference 模型本身的质量；对模型架构有一定要求（必须有显式的 FFN 激活分布）。

---

### 6. 实用指南
*   **实现细节**：
    *   **权重调节**：在本文实验中 $\alpha=\beta=0.5$ 效果最佳，平衡了质量与定位。
    *   **分桶配置**：$[1, 1, 2, 3]$ 的层级 neuron 保留方案是经验最优值，需根据不同规模的 VLM 适当微调。
*   **迁移建议**：该方法逻辑通用，对于任何 LLM 结构，只需找到能够代表样本特征的 FFN 层并提取 Top-K 索引即可迁移。

---

### 7. 总结
*   **核心思想**：通过模型前向计算，利用内在能力与激活模式选择行为多样性的数据子集。
*   **速记版 Pipeline**：
    1.  一次前向传播计算视觉贡献度。
    2.  提取顶层神经元激活分布作为行为指纹。
    3.  按指纹分桶，在每桶内取高质量样本。
    4.  汇聚样本构建最终核集。

**Key Findings:**

- We introduce MAGIC, a training-free, forward-only coreset selection method designed to construct compact yet behaviorally faithful subsets for multimodal instruction tuning.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.26004v1)
- [arXiv](https://arxiv.org/abs/2605.26004v1)

---

<a id='2605.25979v1'></a>
## [LLaVA-OneVision-2: Towards Next-Generation Perceptual Intelligence](https://arxiv.org/abs/2605.25979v1)

**Authors:** Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, Yifei Shen, Yuanhan Zhang, Kaichen Zhang, Wenkang Zhang, Zheng Cheng, Nansen Zhang, Chunsheng Wu, Chunjiang Ge, Zimin Ran, Dehua Song, Chunyuan Li, Shikun Feng, Ming Hu, Zhangquan Chen, Junbo Niu, Bo Li, Ziyong Feng, Ziwei Liu, Zongyuan Ge, Jiankang Deng

**Published:** 2026-05-25

**Categories:** cs.CV

**Abstract:**

We introduce LLaVA-OneVision-2 (LLaVA-OV-2), the most capable vision-language model in the LLaVA-OneVision series to date, achieving superior performance across a broad range of multimodal benchmarks. The model builds on a native OneVision-Encoder and incorporates Windowed Attention for efficient local computation while maintaining native resolution. Its key advance is codec-stream tokenization: it treats compressed video as a continuous bit-cost stream, where bit-cost dynamics determine adaptive temporal groups, and motion-residual cues select salient spatial evidence into compact visual canvases. This allocation concentrates a limited token budget on event-bearing content, enabling more stable long-video token compression than fixed groups of pictures. A shared 3D RoPE further places codec canvases, sampled frames, and images in a unified spatiotemporal coordinate system. Furthermore, we build the LLaVA-OV-2 data and training stack around large-scale open supervision: approximately 8M re-captioned video samples for pretraining, a 4M-sample spatial corpus for fine-tuning. We also introduce JumpScore, a temporal-localization benchmark targeting fine-grained grounding in high-frequency, densely repeated motion, a regime underrepresented by existing video evaluations. A standout capability of LLaVA-OV-2 is its unified perception across video understanding, temporal grounding, spatial grounding, and manipulation-trace reasoning. On JumpScore, LLaVA-OneVision-2-8B reaches 74.9 JumpScore mAP, surpassing Qwen3-VL-8B (30.1) by +44.8 points; under matched visual-token budgets on the same benchmark, codec-stream inputs improve temporal grounding over frame sampling by +9.7 points. Across standard benchmarks, LLaVA-OneVision-2-8B further outperforms Qwen3-VL-8B by +4.3 average points on video tasks, +5.3 on spatial tasks, and +15.6 average J&F on tracking tasks.

**Analysis:**

## 1. 摘要翻译

我们介绍了 LLaVA-OneVision-2 (LLaVA-OV-2)，这是目前 LLaVA-OneVision 系列中最强大的视觉语言模型，在多模态基准测试中展现出卓越的性能。该模型构建于原生的 OneVision-Encoder 之上，并结合窗口注意力机制，在维持原生分辨率的同时实现了高效的局部计算。其核心创新是**编解码流（codec-stream）标记化**：它将压缩视频视为连续的比特流，通过比特成本动态确定自适应时间分组，并利用运动残差线索将显著的空间证据选入紧凑的视觉画布。这种分配方式将有限的 Token 预算集中于事件发生的内容上，实现了比固定图片组（GOP）更稳定的长视频 Token 压缩。此外，我们构建了包含约 800 万视频样本的预训练数据和 400 万样本的空间微调语料，并推出了 JumpScore——一个针对高频重复动作中细粒度时间定位的基准测试。

## 2. 方法动机分析

*   **驱动力**：作者旨在解决长视频中 Token 预算浪费问题，使视觉观察不再盲目遵循固定的时间片，而是能够自动识别视频中的“重点”。
*   **痛点**：当前 LVLM 多采用“统一帧采样”或“混合分辨率”模式，这种方法将视频简化为一组离散帧，忽略了视频原本的预测性流信号（如 H.264/H.265 的 I/P 帧结构），导致无法捕捉关键的运动动力学和细粒度时空变化。
*   **研究假设**：视频压缩码流中蕴含的“比特成本”和“运动残差”是衡量时空显著性的天然信号；通过对这些信号进行建模，模型可以更智能地分配 Token 预算。

## 3. 方法设计详解

*   **流程总结**：
    1.  **码流分析**：提取视频编码（如 H.264/H.265）中的 P/B 帧数据，计算其在时间窗口内的比特成本 ($e_b$)，作为划分自适应 GOP 的依据。
    2.  **显著性评分**：基于运动向量（$M_t$）和亮度残差（$R_t$）生成像素级显著性图，进而聚合成 $2 \times 2$ 补丁块的评分。
    3.  **画布构建**：根据显著性评分，将最重要的块“打包”进 I-canvas（锚点）和 P-canvases（运动残差），构建成紧凑的视觉序列。
    4.  **统一推理**：将处理后的 canvases、 sampled frames 和 images 接入 OneVision-Encoder，利用 3D RoPE 和 group-visible 掩码（确保 GOP 内 Token 可见性）进行联合感知。
*   **核心逻辑**：通过比特成本动态调整时间切片（高变化处切片短，稳定处切片长），通过残差评分动态调整空间采样（只采最“值钱”的块）。

## 4. 方法对比分析

*   **本质区别**：从传统的“模型侧采样”（如 Token Merging/Dropout）转变为“输入侧预处理”，将视频编码器的内在属性（比特流动态）直接作为感知信号。
*   **创新贡献**：提出了 codec-stream 标记化范式，使模型不仅“看”到图像，还能“理解”编码流中的动态语义。
*   **适用场景**：极高长视频理解任务，尤其是需要细粒度时空定位（如 JumpScore 场景）的任务。

## 5. 实验分析

*   **关键结果**：在 JumpScore 上，LLaVA-OV-2-8B 达到 74.9 mAP，相比 Qwen3-VL-8B 有高达 +44.8 点的提升，证明其在处理高度相似循环运动时的优越性。
*   **优势**：在低预算下表现优异，能显著提升 temporal grounding 和空间推理能力。
*   **局限**：对编解码器质量有一定依赖；相对于均匀采样，增加了预处理开销。

## 6. 实用指南

*   **开源情况**：已开源代码、数据及模型。
*   **迁移建议**：该方法的核心逻辑（提取压缩码流特征）可迁移至任何基于 ViT 的视频 MLLM，只需在前处理端添加对应的编码比特流解析器，无需修改语言模型结构。

## 7. 总结

*   **核心思想**：利用视频编码压缩的动态比特信息，实现自适应的视觉感知分配。
*   **速记版 pipeline**：
    1. 解析视频编码比特流；
    2. 根据编码强度动态划分时段；
    3. 提取关键运动和残差块；
    4. 打包构建高效视觉画布；
    5. 送入统一编码器联合训练。

**Key Findings:**

- We introduce LLaVA-OneVision-2 (LLaVA-OV-2), the most capable vision-language model in the LLaVA-OneVision series to date, achieving superior performance across a broad range of multimodal benchmarks.
- Across standard benchmarks, LLaVA-OneVision-2-8B further outperforms Qwen3-VL-8B by +4.3 average points on video tasks, +5.3 on spatial tasks, and +15.6 average J&F on tracking tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.25979v1)
- [arXiv](https://arxiv.org/abs/2605.25979v1)

---

