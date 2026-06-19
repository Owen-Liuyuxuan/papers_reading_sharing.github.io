time: 20260619

# Arxiv Computer Vision Papers - 2026-06-19

## Executive Summary

## 每日技术摘要：2026年6月18日 Arxiv 计算机视觉论文

### 1. 主要主题与趋势

本期论文体现出三个核心趋势：**具身智能与第一人称视觉深度融合**、**零样本与生成式3D能力**、以及**鲁棒化视觉语言导航**。

- 多篇工作利用第一人称（egocentric）人类视频替代或增强机器人数据，用于具身预训练（HumanScale, UNIEGO）与动作建模（MemoryWAM），暗示人类日常视频正成为机器人学习的重要数据源。
- 3D生成领域向“零样本”和“视觉错觉”扩展（JanusMesh），突破传统几何重建框架。
- 视觉语言模型（VLM）在导航中的应用面临延迟挑战，出现了“慢大脑、快规划器”的解耦设计（Slow Brain）；同时注意力预测机制被引入主动感知（Fast Human Attention）。
- 数据集与仿真平台持续推动领域规范化：CalTennis提供了大规模多视角网球视频与3D姿态基准；TaCauchy与CoLI分别面向触觉与连续体机器人提供可复现仿真/硬件平台。

### 2. 特别重要或创新的论文

- **JanusMesh**：首次实现零样本3D视觉错觉生成，通过跨空间去噪统一了内容与视错觉效果，创意新颖且技术路线上具有启发性。
- **HumanScale**：通过实验证明第一人称人类视频在具身预训练中**优于真实机器人数据**，这一反直觉结论可能重塑具身AI的数据策略。
- **Slow Brain, Fast Planner**：针对VLM在实时导航中的延迟瓶颈，提出“慢大脑（VLM）异步推理、快规划器同步跟踪”架构，工程价值高。
- **CalTennis**：发布包含15个同步摄像头的网球比赛数据集，填补了复杂运动场景下多视角单目转3D姿态的标准基准空白。

### 3. 新兴研究方向与技术

- **以人为中心的具身预训练**：利用大规模第一人称视频（而非机器人遥操作数据）学习通用动作表征，降低机器人学习的数据获取成本。
- **视觉错觉生成**：将计算机图形学中的错觉现象与扩散模型结合，开辟了3D生成的新分支。
- **延迟鲁棒的VLM导航**：将高计算量视觉语言模型作为慢速决策器，与轻量跟踪器配合，解决端到端方案无法实时部署的痛点。
- **主动感知中的注意力预测**：从人类注视行为出发，指导自主导航系统下一步“看哪里”，提升感知效率。
- **可复现的软体/连续体机器人学习平台**：CoLI与TaCauchy分别从硬件（3D打印+同构遥操作）和仿真（有限元触觉模拟）两方面推动该方向标准化。

### 4. 建议全文阅读的论文

- **HumanScale** – 对任何从事具身智能、机器人预训练的研究者都值得精读，其结论可能改变数据路线的选择。
- **JanusMesh** – 若对3D生成、视觉认知或艺术计算感兴趣，本文提供了一种全新的任务范式。
- **CalTennis** – 需要多视角人体姿态估计基准或体育分析数据集的读者应重点关注，其多视角设置和挑战性动作极具价值。
- **Slow Brain, Fast Planner** – 对VLM部署于实时系统有需求的读者，本文的架构设计直接可借鉴。
- **MemoryWAM** – 关注世界模型与长期记忆结合的读者可参考其持久记忆机制，该思想对长时决策任务有普遍意义。

---

## Table of Contents

1. [JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising](#2606.20563v1)
2. [MemoryWAM: Efficient World Action Modeling with Persistent Memory](#2606.20562v1)
3. [UNIEGO: Proxies as Mediators for Unified Egocentric Video Representation Learning](#2606.20559v1)
4. [Generating Robot Hands from Human Demonstrations](#2606.20549v1)
5. [CalTennis: Large Multi-View Tennis Video Dataset and Benchmark of Monocular-to-3D Pose Estimation](#2606.20542v1)
6. [HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining](#2606.20521v1)
7. [Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation](#2606.20491v1)
8. [Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation](#2606.20458v1)
9. [TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation](#2606.20426v1)
10. [CoLI: A Reproducible Platform for Continuum Robot Learning via Monolithic 3D Printing and Isomorphic Teleoperation](#2606.20389v1)

---

## Papers

<a id='2606.20563v1'></a>
## [JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising](https://arxiv.org/abs/2606.20563v1)

**Authors:** Siang-Ling Zhang, Huai-Hsun Cheng, Tsung-Ju Yang, Yu-Lun Liu

**Published:** 2026-06-18

**Categories:** cs.CV

**Abstract:**

Creating 3D visual illusions, a single 3D mesh that reveals entirely different semantics from various viewing angles, is a fascinating but tough challenge. Existing optimization-based methods are slow and can produce oversaturated colors. In contrast, naive stitching approaches fail to produce geometrically coherent objects. This results in visible unnatural seams and semantic leaks. In this paper, we present a fast and training-free framework for generating text-driven 3D visual illusions. Our approach decouples the generation into two stages. First, we propose a cross-space dual-branch denoising process. This process dynamically decodes 3D latents into voxel space for CLIP-guided orientation alignment and Signed Distance Field (SDF) blending, which ensures seamless geometric fusion. Second, we introduce a view-conditioned texture synthesis module that projects and aggregates view-specific 2D diffusion priors onto the fused geometry. Extensive experiments demonstrate that our method generates highly realistic, dual-semantic 3D illusions in just 3-5 minutes. It significantly outperforms existing methods in geometric integrity, semantic recognizability, and efficiency. Project page: https://siang1105.github.io/JanusMesh.github.io/

**Analysis:**

### 1. 摘要翻译
创造3D视觉幻象（即从不同视角呈现完全不同语义的单一3D网格）是一项引人入胜但极具挑战的任务。现有的基于优化的方法不仅速度缓慢，还常产生过饱和的色彩；而简单的拼接方法则无法保证几何连续性，导致出现不自然的人工缝隙和语义泄露。本文提出了一种快速、无需训练的框架，用于生成文本驱动的3D视觉幻象。我们的方法将生成过程解耦为两个阶段：首先，提出了一种跨空间双分支去噪过程，将3D潜空间动态解码至体素空间进行CLIP引导的方位对齐和带符号距离场（SDF）融合，确保了几何结构的无缝融合；其次，引入了一种视角条件下的纹理合成模块，将视角特定的2D扩散先验投影并聚合到融合后的几何体上。实验表明，我们的方法仅需3-5分钟即可生成高度逼真的双语义3D幻象，在几何完整性、语义可识别性和效率方面显著优于现有方法。

### 2. 方法动机分析
- **驱动力**：旨在填补“多语义3D幻象生成”在速度与质量上的空白，实现零样本（Zero-shot）、快速的3D幻象合成。
- **现有方法痛点**：
    - **基于优化的方法（如SDS）**：收敛极慢（约40分钟），且容易导致颜色过饱和及局部形变。
    - **直接拼接方法**：在两个不同语义物体的交界面存在几何断层，产生明显的裂缝与非自然过渡，且容易发生语义互相渗透（泄露）。
- **研究假设**：通过在去噪过程中显式地在体素空间进行几何融合，并利用CLIP指导方位对齐，可以解决语义冲突并构建几何连续的中间过渡体。

### 3. 方法设计详解
- **流程总结**：
    1. **初始化**：输入两个文本提示，生成两个独立的单语义初始体素。
    2. **CLIP方位对齐**：通过渲染多视角图像，计算CLIP相似度，自动寻找最佳相对旋转角度，确保两个物体在融合时silhouette（轮廓）匹配。
    3. **Stage 1：双分支几何生成**：采用TRELLIS架构，在每个去噪步（denoising step）内，将潜变量解码至体素空间，计算各自的SDF，进行加权平均融合并二值化，再重新编码回潜空间，强制模型学习融合后的几何形状。
    4. **Stage 2：视角条件纹理合成**：针对融合后的非自然几何体，利用深度条件ControlNet进行纹理生成，根据观察视角实时投影不同语义的纹理，并利用cosine加权聚合。
- **关键公式**：$\text{SDF}_{\text{blend}} = \frac{\text{SDF}(v_1) + \text{SDF}(v_2)}{2}$。该公式本质上是在几何空间寻找中间态，通过二值化阈值保留语义边界。

### 4. 方法对比分析
- **本质区别**：从“后端后处理拼接”转向“前端去噪过程中的几何约束与融合”。
- **创新贡献**：
    - **跨空间融合**：在去噪步骤中引入体素级SDF融合，这是解决几何断层的关键。
    - **CLIP引导对齐**：解决了不同语义物体在空间位置上“对不上”的问题。
    - **视角条件纹理**：绕过了几何不完美带来的纹理塌陷。

### 5. 实验分析
- **关键结果**：GPT语义识别准确率达到84%，显著领先于拼接法（76%）和优化法（70%）；生成时间缩短至3-5分钟。
- **主要优势**：高效率（无需优化）、高语义清晰度、无缝几何过渡。
- **主要局限**：对部分难以对齐的复杂几何结构（如特定姿态的动物）仍可能出现融合伪影；对三物体融合，由于视角分配空间变窄，效果逊于双物体。

### 6. 实用指南
- **开源情况**：项目主页：https://siang1105.github.io/JanusMesh.github.io/。
- **实现细节**：关键超参数为SDF二值化阈值 $\tau=0.8$；空间控制指导步数 $t_0=10$；需预先生成单语义体素作为先验。
- **迁移可能**：双分支去噪与体素融合思路可直接迁移至任何基于Rectified Flow的3D生成模型中，用于构建组合式3D资产。

### 7. 总结
- **核心思想**：通过去噪过程中的SDF动态融合，在几何空间将两个语义形态无缝缝合。
- **速记版pipeline**：
    1. 使用CLIP对齐两个物体的方位；
    2. 在去噪循环中解码体素；
    3. 融合SDF并重新编码；
    4. 根据视角投影对应的纹理。

**Key Findings:**

- In this paper, we present a fast and training-free framework for generating text-driven 3D visual illusions.
- Our approach decouples the generation into two stages.
- First, we propose a cross-space dual-branch denoising process.
- Second, we introduce a view-conditioned texture synthesis module that projects and aggregates view-specific 2D diffusion priors onto the fused geometry.
- Extensive experiments demonstrate that our method generates highly realistic, dual-semantic 3D illusions in just 3-5 minutes.
- It significantly outperforms existing methods in geometric integrity, semantic recognizability, and efficiency.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20563v1)
- [arXiv](https://arxiv.org/abs/2606.20563v1)

---

<a id='2606.20562v1'></a>
## [MemoryWAM: Efficient World Action Modeling with Persistent Memory](https://arxiv.org/abs/2606.20562v1)

**Authors:** Sizhe Yang, Juncheng Mu, Tianming Wei, Chenhao Lu, Xiaofan Li, Linning Xu, Zhengrong Xue, Zhecheng Yuan, Dahua Lin, Jiangmiao Pang, Huazhe Xu

**Published:** 2026-06-18

**Categories:** cs.RO

**Abstract:**

Robust robotic manipulation in the real world requires not only an understanding of the current observation, but also memory and dynamics modeling. World action models (WAMs) possess these capabilities by jointly modeling visual foresight and actions conditioned on both current and historical observations, making them a promising paradigm for robotic manipulation. However, existing WAMs face a fundamental trade-off: methods with efficient inference typically condition only on a bounded window of recent observations and therefore struggle in non-Markovian environments, whereas methods that preserve long histories incur time and space costs that grow substantially with sequence length. To address this challenge, we introduce MemoryWAM, a world action model with efficient persistent memory. MemoryWAM uses a hybrid memory design that combines recent frames, event-boundary anchor frames, and compact gist tokens that summarize long-range history. A tailored attention mechanism enables retrieval of both detailed short-term context and compressed long-term context, supporting memory-dependent decision-making with reduced inference latency and GPU memory usage. Across long-horizon, memory-dependent manipulation tasks in both simulation and the real world, MemoryWAM outperforms strong vision-language-action (VLA) and WAM baselines while maintaining favorable computational efficiency.

**Analysis:**

### 1. 摘要翻译
现实世界中的鲁棒机器人操作不仅需要理解当前的观测，还需要进行记忆和动力学建模。世界动作模型（WAMs）通过联合建模视觉预测和基于当前及历史观测的动作来解决这一问题，使其成为机器人操作的一个有前景的范式。然而，现有的WAMs面临一个根本性的权衡：推理高效的方法通常仅基于近期观测的有限窗口，因此在非马尔可夫环境中表现不佳；而保留长历史的方法则会导致时间和空间成本随序列长度大幅增加。为了解决这一挑战，我们引入了MemoryWAM，一种具有高效持久记忆的世界动作模型。MemoryWAM采用了一种混合记忆设计，结合了近期帧、事件边界锚定帧和总结长程历史的紧凑型“要点（gist）”标记。一种定制的注意力机制使得模型能够同时检索详细的短期上下文和压缩的长期上下文，从而支持依赖记忆的决策制定，并减少了推理延迟和GPU内存占用。在仿真和现实世界的长程、依赖记忆的操作任务中，MemoryWAM在保持良好的计算效率的同时，优于强有力的视觉-语言-动作（VLA）和WAM基线。

### 2. 方法动机分析
- **驱动力**：在非马尔可夫机器人任务中，策略需要基于长期历史（如被遮挡的物体位置、初始状态）进行决策，但传统的全历史缓存会导致推理成本呈线性增长，变得不可用。
- **痛点**：现有方法在“推理效率”与“长程记忆保留”之间存在尖锐冲突，要么彻底丢弃历史（滑动窗口），要么全量存储导致内存爆炸（全历史缓存）。
- **研究假设**：人类记忆并非平等的记录，而是由短期工作记忆、关键事件锚点（如任务开始）和抽象的长期要点（Gist）组成的混合系统。可以通过类似机制压缩长程历史。

### 3. 方法设计详解
- **流程总结**：
  1. **输入处理**：将观测编码为视觉潜在表示（Latents）。
  2. **混合记忆缓存 (Hybrid Memory)**：
     - **短期记忆 (Short-term)**：维护一个最近N帧的滑动窗口，保留高保真细节。
     - **事件边界记忆 (Event-boundary)**：永久保留任务启动时的关键帧。
     - **要点记忆 (Gist Memory)**：利用压缩机制，将长程历史压缩为极少量的“要点标记”（每个帧仅8个Token）。
  3. **注意力机制**：动作模型通过特定的注意力掩码（Attention Mask），同时关注这三部分，实现对历史信息的选择性读取。
- **模型结构**：采用Mixture-of-Transformers (MoT) 架构，包含视频DiT（负责动力学建模，训练时使用）和动作DiT（负责动作推理，推理时只需利用KV Cache）。
- **算法解释**：核心公式 $C_{\text{gist}} = O(NM)$，通过压缩比 $d = L/M$（本文为15倍），将长程序列的存储代价从 $O(N)$ 降至 $O(N/d)$，在保持性能的同时极大降低了推理显存占用。

### 4. 方法对比分析
- **本质区别**：不采用简单的滑动窗口或全历史，而是基于认知心理学将历史记忆分层处理。
- **创新贡献**：提出了一种可学习的“要点标记（Gist Tokens）”压缩策略，并配合事件边界锚点，在机器人操作领域首次实现了大规模长程历史的高效压缩与精准检索。
- **适用场景**：高延迟敏感、长程、且依赖历史上下文的机器人操作任务（如长序列装配、多步任务）。

### 5. 实验分析（精简版）
- **验证方法**：在RMBench基准测试（包含9个复杂任务）及现实世界任务（Shell Game, Look and Press）中验证。
- **关键结果**：在RMBench上平均成功率达到83%，比之前的强基线LingBot-VA提升4.8个百分点，同时显存和推理延迟显著降低。
- **优势/局限**：优势在于在保证87%高性能的同时，显存占用远低于全注意力模型。局限是如果任务的长期依赖信息极度碎片化且无法被抽象，要点标记可能存在信息损失。

### 6. 实用指南
- **开源情况**：已发布项目主页：https://yangsizhe.github.io/MemoryWAM/。
- **实现细节**：
  - 核心参数：$M=8$ (要点Token数), $N_{\text{init}}=2$ (锚点帧), $N_{\text{recent}}=4$ (滑动窗口)。
  - 训练时需使用特定的Attention Mask来模拟推理时的KV Cache可见性。
- **迁移可能**：该架构可直接迁移到任何基于Transformer的长序列视频预测或动作规划任务中。

### 7. 总结
- **核心思想**：利用多层级记忆结构实现长程上下文的轻量化表征。
- **速记版pipeline**：
  1. 缓存近期高精帧；
  2. 固化任务起始锚点；
  3. 将旧历史压缩为少量Token；
  4. 动作模型联合注意力检索。

**Key Findings:**

- To address this challenge, we introduce MemoryWAM, a world action model with efficient persistent memory.
- Across long-horizon, memory-dependent manipulation tasks in both simulation and the real world, MemoryWAM outperforms strong vision-language-action (VLA) and WAM baselines while maintaining favorable computational efficiency.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20562v1)
- [arXiv](https://arxiv.org/abs/2606.20562v1)

---

<a id='2606.20559v1'></a>
## [UNIEGO: Proxies as Mediators for Unified Egocentric Video Representation Learning](https://arxiv.org/abs/2606.20559v1)

**Authors:** Wenhao Chi, Arkaprava Sinha, Dominick Reilly, Hieu Le, Srijan Das

**Published:** 2026-06-18

**Categories:** cs.CV, cs.LG

**Abstract:**

Egocentric video understanding is inherently limited by the narrow perspective of wearable cameras: a single viewpoint, a single modality, a single model cannot capture the full richness of human action. We argue that a truly expressive egocentric representation must subsume complementary knowledge across viewpoints, modalities, and foundation model representations, yet remain deployable from egocentric video alone. To this end, we introduce a hierarchical multi-teacher distillation framework that produces UNIEGO, a unified egocentric encoder trained with nine teachers spanning ego-exo viewpoints, RGB, depth, and skeleton modalities, and four foundation models. Rather than distilling directly from heterogeneous teachers whose incompatible architectures and feature geometries induce conflicting gradients, our framework interposes a layer of representation-specific Proxy models that translate diverse teacher knowledge into a homogeneous egocentric space. A second distillation stage, Selective Proxy Distillation (SPD), then adaptively selects, for each training sample, the subset of proxies that are both correct and confident, distilling exclusively from reliable supervision and suppressing erroneous signals. SPD is further stabilized by initializing UNIEGO as a learned convex combination of proxy parameters, placing the unified model in a well-conditioned region of the loss landscape before distillation begins. UNIEGO achieves state-of-the-art performance across three egocentric video understanding tasks - action recognition, video retrieval, and action segmentation on three challenging ego-exo benchmarks, outperforming naive multi-teacher distillation baselines and demonstrating that structured, proxy-mediated knowledge transfer yields richer and more discriminative egocentric representations.

**Analysis:**

### 1. 摘要翻译
本研究针对穿戴式相机视角窄、模态单一导致无法捕捉人类行为全貌的问题，提出了一种名为UNIEGO的统一 egocentric（自我中心）表示学习框架。该框架利用九个跨视点（ego-exo）、跨模态（RGB、深度、骨骼）及多种基础模型的教师模型，通过层次化知识蒸馏技术训练一个单一的编码器。为了解决异构教师模型间的架构冲突与“表示鸿沟”问题，我们引入了特定于表示的“代理（Proxy）”模型作为中介，将异构教师知识转换为统一的嵌入空间。此外，我们提出了“选择性代理蒸馏（SPD）”，通过动态选择最可靠的代理来引导学生模型的学习，并利用learned convex组合初始化模型。实验表明，UNIEGO在三项挑战性egocentric行为理解任务上达到了SOTA水平，证明了这种结构化代理介导的迁移优于传统的直接蒸馏。

---

### 2. 方法动机分析
- **驱动力**：利用海量非自我中心（Exocentric）及多模态数据，强化单一自我中心摄像头的感知能力，且要求推理阶段仅使用自我中心RGB视频。
- **痛点**：
  1. **表示鸿沟**：异构教师（如骨骼图神经网络 vs. RGB Transformer）的特征空间根本不兼容。
  2. **梯度冲突**：直接从多个异构教师蒸馏会导致优化目标不一致，引发严重的梯度冲突。
- **研究假设**：通过引入“代理层”将教师空间统一化，并基于样本实时可靠性进行选择性监督，可以有效消除冲突并获取更优的表征。

---

### 3. 方法设计详解
**UNIEGO Pipeline:**
1. **Level-I: 代理学习 (Proxy Learning)**：为每个教师模型 $T_r$ 独立训练一个同构的代理模型 $P_r$。无论教师原始结构如何，代理共享统一的编码器架构。通过余弦相似度损失和分类损失，将各类教师知识映射到统一的“代理空间”。
2. **Level-II: 代理合并 (Proxy Merging)**：在蒸馏前，通过 learned convex combination 将所有 $P_r$ 的参数初始化为 UNIEGO 学生模型的权重。这使得学生模型直接处于一个损失函数的平坦、良性区域。
3. **Level-II: 选择性代理蒸馏 (SPD)**：
    - **可靠性过滤**：对每个输入样本，计算所有代理的分类准确度，仅选择在该样本上预测正确且置信度高（Loss小）的 Top-k 个代理。
    - **混合蒸馏**：将所选代理的特征（通过 $D_{cos}$）和分类逻辑（通过 $D_{KL}$）注入学生模型，过滤错误信号，避免负迁移。

---

### 4. 方法对比分析
- **本质区别**：从“直接蒸馏教师”转变为“蒸馏经过空间对齐和筛选的代理”。
- **创新点**：
    - **代理中介**：将复杂的异构架构差异在Level-I即被“消解”。
    - **SPD策略**：引入实例级（instance-wise）的可靠性权重，而非全局固定权重。
    - **凸组合初始化**：从损失函数的可优化性出发，确保学生模型起步良好。

---

### 5. 实验分析
- **验证方法**：在EgoExo-Fitness、Assembly101、EgoExo4D三个数据集上进行动作识别、视频检索、动作分割。
- **关键结果**：在多个backbone（TimeSformer, UniFormer-S, ViFi-CLIP）上均显著提升，证明其模型不可知特性。
- **优势**：消除了多教师场景下的“梯度冲突”，在全身体能健身动作中表现尤其优异（利用了exocentric视角下的全身姿态信息）。
- **局限**：目前的代理选择依赖于Loss小的启发式准则，未来若能引入动态预测代理可靠性的神经网络，上限更高。

---

### 6. 实用指南
- **开源地址**：[github.com/Wenhao-Chi/UNIEGO](https://github.com/Wenhao-Chi/UNIEGO)
- **迁移建议**：若要将其迁移到其他任务，重点在于构建多样性的教师池（涵盖不同领域知识），并确保代理模型的设计能覆盖所需的特征维度。
- **超参数**：重点调优 $K$（选择代理数量）以及代理合并阶段的权重 $\alpha$。

---

### 7. 总结
- **核心思想**：通过中间代理对齐异构知识，并以实例选择策略实现精准、低冲突的知识蒸馏。
- **速记版pipeline**：
    1. 为每个教师训练一个统一结构的影子模型（代理）。
    2. 加权混合所有代理参数，初始化学生模型。
    3. 逐样本评估代理表现。
    4. 仅从表现最好的那几个代理中提取知识更新学生。

**Key Findings:**

- To this end, we introduce a hierarchical multi-teacher distillation framework that produces UNIEGO, a unified egocentric encoder trained with nine teachers spanning ego-exo viewpoints, RGB, depth, and skeleton modalities, and four foundation models.
- UNIEGO achieves state-of-the-art performance across three egocentric video understanding tasks - action recognition, video retrieval, and action segmentation on three challenging ego-exo benchmarks, outperforming naive multi-teacher distillation baselines and demonstrating that structured, proxy-mediated knowledge transfer yields richer and more discriminative egocentric representations.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20559v1)
- [arXiv](https://arxiv.org/abs/2606.20559v1)

---

<a id='2606.20549v1'></a>
## [Generating Robot Hands from Human Demonstrations](https://arxiv.org/abs/2606.20549v1)

**Authors:** Sha Yi, Nicklas Hansen, Xueqian Bai, Carmelo Sferrazza, Michael T. Tolley, Xiaolong Wang

**Published:** 2026-06-18

**Categories:** cs.RO

**Abstract:**

Robot learning has advanced rapidly in learning control, but learning the physical body of a robot remains much more difficult because jointly searching over design and control creates a very large combinatorial problem. Here, we present a data-driven framework for generating robot hands from human demonstrations. Instead of learning a complex controller together with each candidate design, we generate robot hand designs using the same simple control policy used after fabrication: matching fingertip positions through inverse kinematics. Using more than 4 million frames of human fingertip motion from everyday manipulation, our algorithm optimizes tree-structured robot hands to reproduce desired target motions. The framework produced both a 6-degree-of-freedom (DoF) general-purpose hand and lower-DoF task-specific hands with spatial four-bar mimic joints. To accelerate the search over designs, we trained a reinforcement-learning (RL) actor to propose good hand designs and joint angles, reducing search time from hours to minutes. We fabricated the mechanisms directly as one-piece articulated structures with print-in-place joints. In real-world experiments, the 6-DoF hand achieved highly accurate teleoperated fingertip tracking better than available commercial robot hands, whereas the specialized 3-DoF hands reproduced structured human and synthetic trajectories with reduced mechanical complexity. These results showed that large-scale human motion data can be used not only to train robot controllers but also as a reference for optimizing and generating the physical embodiment of robots.

**Analysis:**

### 1. 摘要翻译
机器人学习在控制层面已取得快速进展，但机器人物理实体的设计依然困难，因为联合搜索设计空间与控制空间是一个巨大的组合优化问题。本文提出了一种基于人类演示生成机器人手的驱动框架。我们不再为每个候选设计学习复杂的控制器，而是通过与制造后相同的简单逆运动学（IK）控制策略来生成机器人手设计。利用超过400万帧的人类指尖运动数据，我们的算法优化了树状结构的机器人手，以复现预期的目标动作。该框架既能生成6自由度（DoF）通用手，也能生成具有空间四连杆模仿关节的低自由度任务专用手。为了加速设计搜索，我们训练了一个强化学习（RL）执行器来提出优良的设计方案和关节角度，将搜索时间从数小时缩短至数分钟。我们将这些机构直接制造为具有一体化打印关节的结构。在真实实验中，6-DoF手实现了比商用机器人手更精确的指尖追踪，而专用3-DoF手在降低机械复杂性的同时复现了结构化动作。这些结果表明，大规模人类运动数据不仅可用于训练控制器，还可作为优化和生成机器人物理实体的参考。

### 2. 方法动机分析
- **驱动力**：旨在解决“机器人形态设计与控制策略协同优化”这一非凸、高组合复杂性的难题。
- **痛点**：传统协同设计（Co-design）往往要求为每个候选形态重新训练控制器，计算开销巨大；且形态变动会重塑可行运动空间，导致优化过程高度敏感。
- **研究假设**：如果机器人制成后仅采用简单的逆运动学（IK）策略进行控制，那么在设计阶段就应直接使用该IK控制器进行优化，从而实现“部署对齐（deployment-aligned）”的协同优化。

### 3. 方法设计详解
- **pipeline**：
  1. **运动表示**：输入为手部指尖轨迹，利用预训练的轨迹编码器将其映射为紧凑的潜在特征（Context Vector）。
  2. **Actor初始化**：RL执行器（MLP架构）基于潜在特征采样候选的硬件参数（如连杆长度、关节方向、Bennett连接参数）和关节初始角度。
  3. **梯度优化（Differentiable GD）**：通过可微的前向运动学模型，利用梯度下降（GD）在IK控制约束下，进一步微调硬件参数与关节路径，使指尖轨迹误差最小化。
  4. **制造转换**：优化后的运动学结构转化为CAD模型，通过3D打印实现“一体化打印（print-in-place）”的铰链结构。
- **核心算法**：引入了**Bennett四连杆机构**的软约束。通过半角公式（half-angle relation）模拟被动联动，将复杂的闭链约束松弛为可优化的残差参数（$r_j \ge 0$），从而在保持物理合理性的同时，避开了严苛的闭链硬约束带来的数值不稳定性。

### 4. 方法对比分析
- **本质区别**：本文不是在设计过程中“寻找适配的复杂策略”，而是通过“固定简单的IK策略”，反向推导最适配该策略的形态。
- **创新点**：将强化学习作为“初始化生成器”，极大缩小了后续梯度优化的搜索空间，解决了协同设计中“冷启动”和“局部最优”问题。

### 5. 实验分析（精简版）
- **关键结果**：生成的6-DoF手在指尖追踪误差上显著优于商业基线（如XHand和Inspire Hand）；3-DoF专用手通过被动机构实现了特定任务（如旋盖、插钥匙）的高精度追踪。
- **主要优势**：将数小时的搜索压缩至分钟级；能够通过专用形态有效利用“机械智能”降低控制自由度。
- **主要局限**：目前的优化仅基于指尖运动，未考虑更复杂的全身接触、摩擦力和负载承载能力。

### 6. 实用指南
- **开源情况**：已提供项目主页（https://yswhynot.github.io/generating-robot-hands/）。
- **实现建议**：在构建Bennett铰链时，必须采用软约束（残差参数化）而非硬约束，否则梯度在极小的可行域内难以更新。
- **迁移性**：该框架可轻松迁移至软体抓手设计或其他欠驱动机器人领域，核心在于定义好“机构约束”与“控制原语”的协同关系。

### 7. 总结
- **核心思想**：以简单的逆运动学为 priors，通过协同优化实现任务驱动的形态生成。
- **速记版pipeline**：
  1. 编码人类指尖运动轨迹；
  2. Actor建议初始硬件参数；
  3. 通过梯度优化精炼形态与控制；
  4. 3D打印实现结构。

**Key Findings:**

- Here, we present a data-driven framework for generating robot hands from human demonstrations.
- These results showed that large-scale human motion data can be used not only to train robot controllers but also as a reference for optimizing and generating the physical embodiment of robots.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20549v1)
- [arXiv](https://arxiv.org/abs/2606.20549v1)

---

<a id='2606.20542v1'></a>
## [CalTennis: Large Multi-View Tennis Video Dataset and Benchmark of Monocular-to-3D Pose Estimation](https://arxiv.org/abs/2606.20542v1)

**Authors:** Ilona Demler, Xinran Xie, Blake Werner, Anna Szczuka, Pietro Perona

**Published:** 2026-06-18

**Categories:** cs.CV

**Abstract:**

The Caltech Tennis Dataset (CalTennis) is a large-scale video benchmark for evaluating monocular-to-3D pose estimation in the wild. CalTennis comprises over 11 million frames (51 hours) of tennis practice and match play from 40 players, captured with 2-6 synchronized cameras at 60 Hz. It is 10 times larger than existing in-the-wild human motion video datasets and 3 times larger than existing MOCAP-ground-truthed datasets, and it is the first large-scale benchmark to provide synchronized multi-view recordings of expert athletic motion. The multi-view setup enables inexpensive, label-free evaluation of monocular-to-3D pose estimation algorithms. We describe a simple, standardized protocol that enables data collection without specialized equipment or expertise, along with fully automated video calibration and synchronization. Benchmarking state-of-the-art monocular-to-3D pose methods on CalTennis, we find that while 3D joint angle recovery is now quite accurate, all models struggle to estimate depth and foot contact consistently. We further propose two novel performance metrics, footwork and stability, as well as qualitatively study body shape inconsistency. These metrics expose previously underexplored failure modes and point to concrete opportunities for improvement in pose estimation and action analysis.

**Analysis:**

### 1. 摘要翻译
CalTennis是一个用于评估野外单目至3D姿态估计的大规模视频基准。它包含超过1100万帧（51小时）的网球训练和比赛画面，由40名选手参与，使用2-6个同步相机在60Hz下捕获。其规模是现有野外人体运动视频数据集的10倍，MOCAP真值数据集的3倍，且是首个提供专家级运动多视角同步记录的大规模基准。多视角设置实现了廉价、无需标签的评估。我们提出了一种简单的标准化协议和全自动视频校准与同步方案。对主流算法的基准测试表明，尽管3D关节角度恢复已较准确，但所有模型在深度和足部接触估计上仍表现不稳定。此外，我们提出了“步法（footwork）”和“稳定性（stability）”两个新指标，以揭示现有基准中未被充分探索的失效模式。

### 2. 方法动机分析
*   **驱动力**：旨在克服现有3D姿态估计基准对昂贵实验室环境（如MOCAP、传感器、扫描仪）的依赖，实现基于廉价硬件（手机、三脚架）在真实场景下的可靠评估。
*   **痛点**：现有数据集规模小、覆盖动作单一、且严重依赖昂贵的真值标注。此外，主流模型在处理体育等高动态、复杂深度环境时，难以保持多视角的一致性，导致下游生物力学分析出现严重误差。
*   **研究假设**：通过在真实场景下使用多视图同步采集，可以利用多视角间的一致性作为直接的、无需标签（label-free）的误差度量信号，从而更客观地评估模型在野外的表现。

### 3. 方法设计详解
*   **数据采集与校准**：使用iPhone及低成本三脚架，通过网球场标准化的几何线段交叉点（line intersections）进行自动相机标定。利用PnP算法最小化重投影误差，无需外部测绘设备。
*   **时空同步**：
    *   **空间融合**：将各相机视角的SMPL-X预测结果，根据标定的外参（$R, T$）通过刚性变换统一到共享的网球场坐标系中。
    *   **时间对齐**：由于设备记录的时间戳存在偏差（可达1000ms），作者定义了一个全局偏移量 $\Delta t$，通过网格搜索最小化多视角间的姿态重投影差，从而实现毫秒级的时间对齐。
*   **评估指标设计**：
    *   **一致性作为误差界**：将多视角的重构结果对比作为模型性能的“下界”，越一致则模型越可信。
    *   **足部与稳定性指标**：引入 `Eskate`（足部滑移，对比不同视图下的关节速度差异）和 `Estab`（稳定性，计算质心相对于支撑多边形的投影距离），用于量化动作的物理合理性。

### 4. 方法对比分析
*   **本质区别**：不试图构建复杂的MOCAP真值，而是利用网球场这一天然的几何空间作为约束，将“多视图一致性”直接转化为一种监督信号。
*   **创新贡献**：提出了一套低成本的采集与无需标签的评估框架；明确了当前单目姿态估计在绝对深度、足部接触和体型估计上的系统性偏差。
*   **适用场景**：体育分析、临床步态分析、以及任何具有结构化背景的复杂运动分析场景。

### 5. 实验分析（精简版）
*   **验证方法**：基准测试了五种SOTA模型（PromptHMR, WHAM, GVHMR, TRAM, GENMO）。
*   **关键结论**：没有单一模型在所有维度上最优。PromptHMR在标准度量下表现最好，但WHAM在足部接触一致性上更强。
*   **主要局限**：目前的实验主要基于网球场景，对环境的普适性验证尚未完全展开，且该评估仅限于“一致性”，而非绝对的“真值”。

### 6. 实用指南
*   **开源信息**：代码与数据集已在官网及Hugging Face公开。
*   **关键点**：需要确保拍摄场景有清晰的地面标记（球场白线）；在数据处理中，必须通过网格搜索进行时间对齐，否则模型误差会完全被时间错位掩盖。
*   **迁移**：该框架易于迁移到其他结构化运动（如篮球、足球），只需更换相应的标定基准几何对象即可。

### 7. 总结
*   **核心思想**：利用几何约束实现多视图一致性评估，揭示姿态估计物理合理性偏差。
*   **速记版pipeline**：
    1.  使用手机在球场四周多视角同步拍摄。
    2.  利用球场线段交叉点自动标定相机参数。
    3.  通过优化全局时间偏移量对齐视频序列。
    4.  对比不同视角的模型预测结果，计算稳定性与一致性指标。

**Key Findings:**

- Benchmarking state-of-the-art monocular-to-3D pose methods on CalTennis, we find that while 3D joint angle recovery is now quite accurate, all models struggle to estimate depth and foot contact consistently.
- We further propose two novel performance metrics, footwork and stability, as well as qualitatively study body shape inconsistency.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20542v1)
- [arXiv](https://arxiv.org/abs/2606.20542v1)

---

<a id='2606.20521v1'></a>
## [HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining](https://arxiv.org/abs/2606.20521v1)

**Authors:** Juncheng Ma, Jianxin Bi, Yufan Deng, Xuanran Zhai, Kewei Zhang, Ye Huang, Bo Liang, Shukai Gong, Jiankai Tu, Xiaotian Tang, Jiaxin Li, Kaiqi Chen, Duomin Wang, Yuqi Wang, Bingyi Kang, Eric Huang, Zhiyang Dou, Zhen Dong, Enze Xie, Wojciech Matusik, Tat-Seng Chua, Daquan Zhou

**Published:** 2026-06-18

**Categories:** cs.CV

**Abstract:**

Embodied foundation models are expected to benefit from data scaling like large language models, but face a much tighter data bottleneck. Teleoperated real-robot trajectories remain the dominant pretraining source due to their precise action supervision and embodiment alignment, yet their scalability is limited by high collection cost, acquisition difficulty, and low behavioral and environmental diversity. These limitations have sparked interest in egocentric human video as a scalable, substantially lower-cost, and more diverse alternative for embodied model pretraining. However, its effectiveness compared to teleoperated real-robot data remains underexplored. To address this question, we conduct a systematic study comparing egocentric human video and teleoperated real-robot trajectories as pretraining data sources for embodied foundation models, under fixed post-training and validation protocols. Surprisingly, we find that egocentric data, when processed through a carefully designed filtering and labeling pipeline, is not merely a viable substitute for model pretraining but can lead to superior performance. With the same amount of pretraining data, models pretrained on egocentric data achieve a 24% lower validation loss on real-robot action prediction, as well as 52.5% and 90% higher success rates on in-distribution and out-of-distribution real-robot task execution, respectively. This finding verifies a scalable paradigm for embodied foundation models: pretrain on egocentric human video to learn diverse world representations, then adapt with a small amount of labeled real-robot data for action-space alignment. We hope this study encourages broader exploration of egocentric data and offers guidance for data quality assessment before costly robot data collection.

**Analysis:**

### 1. 摘要翻译
具身基础模型有望像大语言模型一样受益于数据规模的扩大，但面临着严峻的数据瓶颈。目前的具身预训练数据主要依赖远程操作的真实机器人轨迹，虽然具备精确的动作监督和具身对齐，但面临采集成本高、获取困难以及环境与行为多样性不足的问题。对此，本文探讨了利用大规模、低成本且多样性更强的 egocentric（第一人称视角）人类视频进行具身模型预训练。研究发现，通过精心的过滤和标注流程，egocentric 数据不仅是机器人数据的可行替代，更能实现超越。在相同预训练数据量下，模型在真实机器人动作预测上的验证损失降低了 24%，在任务执行成功率上，其在分布内和分布外场景分别提升了 52.5% 和 90%。这验证了一个新的可扩展范式：利用 egocentric 视频学习多样化的世界表征，再通过少量标记的真实机器人数据进行动作空间对齐。

### 2. 方法动机分析
*   **驱动力**：打破“机器人数据稀缺”导致的具身模型 scaling 瓶颈。
*   **现有方法痛点**：机器人数据（Teleoperation Data）采集极为昂贵且缓慢（需物理硬件、人工操作、受控环境），导致样本多样性窄，模型难以泛化至复杂、未知的真实世界。
*   **研究假设**：egocentric 视频蕴含了海量的 contact-rich（接触丰富）交互信息与长期动作表征，只要解决了“动作空间对齐”这一鸿沟，其带来的世界表征收益将远超由于缺乏精确机器人动力学带来的损失。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **数据预处理与筛选**：从 HumanNet 数据集中抽取 5,000 小时的 egocentric 视频，重点选择多样性高的场景、物体和技能。
    2.  **伪标签生成**：利用手部姿态估计（hand-pose retargeting）技术，将人类视频中的手部动作转化为与机器人末端执行器（end-effector）动作兼容的伪标签。
    3.  **模型训练（预训练阶段）**：采用 Mixture-of-Transformers (MoT) 架构的自回归世界动作模型（WAM）。视频专家（video expert）通过 Wan 2.2 初始化，动作专家（action expert）通过插值初始化，共同学习视频动力学与动作推断。
    4.  **适配（后训练阶段）**：使用相同规模的真实机器人数据对模型进行微调，通过该步骤实现从人类动作空间到具体机器人动作空间的映射对齐。

### 4. 方法对比分析
*   **本质区别**：从传统的“完全依赖机器人数据”转变为“以 egocentric 视频做表示学习，机器人数据做微调对齐”的解耦架构。
*   **创新贡献**：首次系统性论证了在 matched-scale（匹配规模）下，egocentric 视频预训练在 OOD（分布外）泛化能力上显著优于机器人数据。
*   **适用场景**：适用于资源受限、需要大规模交互场景知识学习的具身智能任务。

### 5. 实验分析
*   **关键结论**：egocentric 数据在 unseen 任务（OOD）上的泛化表现远超真实机器人数据（ loss 低 20% 以上，成功率提升 90%）。
*   **优势**：极强的扩展性（Open-world coverage）和对未知场景的鲁棒性。
*   **局限**：动作标签为“伪标签”，存在领域鸿沟，且严重依赖后训练（Post-training）阶段的对齐质量。

### 6. 实用指南
*   **开源情况**：代码已开源至 [DAGroup-PKU/HumanNet](https://github.com/DAGroup-PKU/HumanNet/)。
*   **关键实现点**：视频数据的清洗与手部姿态 retargeting 的鲁棒性是成功的关键；预训练必须选择具有高 motion quality 和高 interaction diversity 的 subset。
*   **迁移可能**：可直接应用于目前流行的 VLA（视觉-语言-动作）架构中，替换掉现有的视觉 backbone 或增加额外的世界模型辅助训练任务。

### 7. 总结
*   **核心思想**：利用人类 egocentric 视频的广阔多样性弥补机器人数据的规模局限。
*   **速记版 Pipeline**：
    1. 收集海量第一人称人类活动视频。
    2. 通过手部姿态转换提取机器人兼容的伪动作标签。
    3. 训练世界动作模型学习视频与动作的关联。
    4. 引入少量真实机器人数据进行针对性对齐微调。

**Key Findings:**

- This finding verifies a scalable paradigm for embodied foundation models: pretrain on egocentric human video to learn diverse world representations, then adapt with a small amount of labeled real-robot data for action-space alignment.
- We hope this study encourages broader exploration of egocentric data and offers guidance for data quality assessment before costly robot data collection.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20521v1)
- [arXiv](https://arxiv.org/abs/2606.20521v1)

---

<a id='2606.20491v1'></a>
## [Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation](https://arxiv.org/abs/2606.20491v1)

**Authors:** Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis

**Published:** 2026-06-18

**Categories:** cs.RO, cs.CV

**Abstract:**

Human visual attention relies on structured scanpaths to efficiently process scenes, yet instilling this behavior into robot autonomy is in its infancy and hindered by the high,computational costs of existing predictive models. To address this, we introduce GazeLNN, a computationally lightweight,scanpath prediction model that leverages Liquid Neural Networks as its recurrent engine and employs MobileNetV3 for feature extraction. Operating auto-regressively, the architecture predicts sequential fixation heatmaps conditioned on the current visual stimulus and fixation history. Despite requiring only 0.61 GFLOPs, GazeLNN achieves state-of-the-art performance on the MIT Low Resolution dataset achieving 0.47 ScanMatch score. It outperforms existing recurrent baselines across diverse evaluation metrics, while reducing computational costs by 99.40% and accelerating inference by up to six times. To investigate the role of human attention modeling in robot autonomy and demonstrate the practical utility of this highly efficient architecture, we integrate GazeLNN into an active camera-robot control policy trained via Reinforcement Learning. This integration enables human-fixation-guided perception during autonomous navigation, validated through successful real-world deployments on an aerial robot.

**Analysis:**

以下是基于《Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation》的深度分析：

### 1. 摘要翻译
人类视觉注意依赖于结构化的扫描路径来高效处理场景，但将这种行为融入机器人自主系统仍处于起步阶段，且受到现有预测模型高昂计算成本的阻碍。为此，我们引入了 **GazeLNN**，这是一种计算轻量化的扫描路径预测模型。它利用液态神经网络（Liquid Neural Networks, LNNs）作为递归引擎，并采用 MobileNetV3 进行特征提取。该架构以自回归方式运行，根据当前视觉刺激和注视历史预测连续的注视热图。尽管仅需 0.61 GFLOPs，GazeLNN 仍在 MIT 低分辨率数据集上实现了 0.47 的 ScanMatch 分数，达到业内领先水平。此外，它将计算成本降低了 99.40%，推理速度提升了六倍。为了验证其在机器人自主性中的作用，我们将 GazeLNN 集成到由强化学习训练的主动相机-机器人控制策略中，并通过空中机器人的真实环境部署验证了其有效性。

### 2. 方法动机分析
*   **驱动力**：在资源受限的自主机器人平台上，实现人类高效的“主动视觉（Active Perception）”行为，通过预测人类注视点来动态引导相机，从而在不增加计算负载的前提下扩大感知范围。
*   **现有方法痛点**：现代扫描路径预测模型（如基于 Transformer 或重型 RNN 的方法）计算代价巨大，难以在嵌入式系统（如 Jetson Orin）上实现实时推理。
*   **研究假设**：人类的注视行为具有时空相关性和随机性，通过轻量级、具有输入依赖性动态的液态神经网络（LNN），可以在大幅降低参数量和计算量的同时，精确捕捉这种复杂的扫描路径动态。

### 3. 方法设计详解
*   **流程总结**：
    1.  **特征提取**：输入图像通过 MobileNetV3 骨干网，提取高维视觉特征。
    2.  **空间增强**：引入 CoordConv 层，将空间坐标（x, y）信息显式注入特征图，增强对空间位置的感知。
    3.  **循环推理（核心）**：使用闭式连续时间（CfC）液态神经网络作为递归模块。该模块接收：[当前特征 + 上一次注视热图 + 隐状态 + 时间间隔 $\Delta t$]。
    4.  **动态更新**：利用 CfC 的门控机制计算当前时刻的隐状态 $h_{i+1}$，通过公式 (1) 结合 sigmoid 门控和 tanh 激活，动态调整输入信息，实现对注视点序列的自回归预测。
    5.  **输出与反馈**：隐状态经过投影和上采样生成热图，热图的最大值点作为当前帧的预测注视点，并反馈作为下一帧的输入。
*   **算法意义**：公式 (1) 中的 $\Delta t$ 引入了连续时间动态，使模型能够根据注视点的持续时间灵活调整状态更新，这是其超越传统定长序列建模的关键。

### 4. 方法对比分析
*   **本质区别**：与现有模型依赖超大规模参数（如 VGG19+DeepLab）不同，GazeLNN 采用了极致轻量化的 CNN+CfC 组合，侧重于“时序动态的轻量建模”而非“特征的暴力堆叠”。
*   **创新贡献**：首次将 CfC 液态神经网络应用于人类注视预测，并成功嵌入到 RL 机器人控制闭环中，实现了从预测到控制的端到端轻量化。
*   **适用场景**：实时性要求高、算力资源受限的边缘机器人设备（如无人机、移动服务机器人）。

### 5. 实验分析（精简版）
*   **关键结果**：在保持最高预测精度的前提下，计算成本仅为现有最强方案（tSPM-Net）的 0.6%。在实机实验中，相比固定相机，该主动策略使积累的感知体素（Voxels）增加了近 50%。
*   **优势**：极高的推理速度（6.84ms/帧），非常适合嵌入式硬件；显著提升了机器人对环境周边区域的感知覆盖率。
*   **局限**：对极度复杂、视觉极度杂乱的场景，可能存在预测偏差，导致相机注视重心暂时漂移。

### 6. 实用指南
*   **开源建议**：关注作者团队在 GitHub 上关于 Aerial Gym 或 GazeLNN 的实现代码。
*   **实现细节**：
    *   **坐标注入**：CoordConv 是提升空间注意力的关键，务必在输入特征图处对齐。
    *   **训练策略**：必须使用 KL-DTW 损失函数来平衡扫描路径的时空对齐与序列特征，这是保证注视预测符合人类模式的核心。
    *   **迁移**：该模块可以轻易嵌入到任何现有的视觉导航框架（如 VSLAM 或基于 LiDAR 的导航系统）中作为“注意力辅助层”。

### 7. 总结
*   **核心思想**：利用液态神经网络实现轻量级、连续时间的注视扫描路径预测。
*   **速记版pipeline**：
    1.  用轻量网络提取图像特征；
    2.  注入空间坐标辅助理解方位；
    3.  通过液态神经单元计算路径序列；
    4.  输出注视热图并引导相机转向。

**Key Findings:**

- To address this, we introduce GazeLNN, a computationally lightweight,scanpath prediction model that leverages Liquid Neural Networks as its recurrent engine and employs MobileNetV3 for feature extraction.
- Despite requiring only 0.61 GFLOPs, GazeLNN achieves state-of-the-art performance on the MIT Low Resolution dataset achieving 0.47 ScanMatch score.
- It outperforms existing recurrent baselines across diverse evaluation metrics, while reducing computational costs by 99.40% and accelerating inference by up to six times.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20491v1)
- [arXiv](https://arxiv.org/abs/2606.20491v1)

---

<a id='2606.20458v1'></a>
## [Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation](https://arxiv.org/abs/2606.20458v1)

**Authors:** Zhenghao "Mark'' Peng, Honglin He, Quanyi Li, Yukai Ma, Bolei Zhou

**Published:** 2026-06-18

**Categories:** cs.RO

**Abstract:**

Learning-based planners for sidewalk navigation can generate diverse candidate trajectories in real time, yet their scoring functions often fail to select the best trajectory in challenging situations, outputting trajectories that make the mobile robot drive onto grass, toward pedestrians, or in the wrong direction, even when better candidates exist in the same set. We call this the trajectory scoring gap: in real-world sidewalk navigation, the gap between an anchor-based planner's top choice and the best possible candidate is substantial, likely due to limited high-level scene understanding capability of the planner. Rather than replacing the planner with an end-to-end Vision-Language-Action model, we propose a VLM-Planner interface that uses a VLM to select a candidate index from the planner's proposal set and then fuse it with the planner's initial output. However, VLMs take 1--3s per query and so cannot directly drive a 5--20Hz control loop. We contribute a training-free, latency-resilient trajectory-level fusion layer that turns a stale VLM selection into real-time planner scoring via geometric similarity with exponential decay. On $\sim$2,000 challenging real-world scenarios (e.g., junctions, pedestrian encounters), VLM selection achieves 30% ADE reduction versus the planner's best selection, while the planner remains competitive in routine situations. In simulation, Score Fusion maintains >80% success rate with delays up to 5s. We demonstrate the full system on a mobile robot navigating challenging campus sidewalks with varied network latency.

**Analysis:**

### 1. 摘要翻译
基于学习的规划器在人行道导航中虽能实时生成多样化候选轨迹，但其评分函数往往无法在复杂场景中选出最佳轨迹，导致机器人误入草地、冲向行人或走错方向。这种现象被称为“轨迹评分差距（trajectory scoring gap）”。为解决此问题，我们提出了一种 VLM-Planner 接口，利用视觉语言模型（VLM）从规划器的候选集中选择索引，并与规划器的初始输出进行融合。针对 VLM 1–3秒的查询延迟无法直接驱动 5–20Hz 控制回路的问题，我们贡献了一种无需训练、延迟韧性的轨迹级融合层。该层通过几何相似度与指数衰减策略，将过期的 VLM 选择转化为实时的规划器评分。在约2,000个真实挑战场景中，VLM 选择比规划器最佳选择降低了30%的平均位移误差（ADE），且在常规场景下保持竞争力。模拟实验表明，即使在延迟高达5秒的情况下，Score Fusion 仍能保持 >80% 的成功率。

### 2. 方法动机分析
- **驱动力**：解决“生成”与“选择”的脱节。现有的局部规划器擅长生成运动学上可行的轨迹，但缺乏对复杂环境的语义理解能力，难以在多个可行解中做出符合社交规范的最优选择。
- **现有方法痛点**：端到端 VLM 导航（VLA）推理速度太慢（1-2Hz），无法支撑高频控制（5-20Hz）；直接执行过期 VLM 轨迹会导致动态障碍物避障失败。
- **研究假设**：规划器负责提供“动力学可行性”，VLM 负责提供“语义指导”。通过融合 VLM 的语义倾向与规划器的实时评分，可以突破延迟瓶颈，实现实时、鲁棒的语义导航。

### 3. 方法设计详解
- **核心 Pipeline**：
  1. **高速循环（规划器）**：以 5–20Hz 生成 $K$ 条具备动力学可行性的候选轨迹 $\{\tau_1, \dots, \tau_K\}$，并附带内部评分 $S_1(\tau_i)$。
  2. **低速循环（VLM）**：以 1Hz 将当前摄像头帧（叠加轨迹编号）输入 VLM，VLM 输出最佳轨迹索引 $k^*$。
  3. **延迟融合（Fusion）**：这是本文的核心创新。系统并不直接执行 VLM 的过期轨迹，而是将该轨迹的“语义意图”作为一种 bias，作用于当前时间步的规划器评分上。
- **算法解释**：
  - **水平感知相似度**：计算当前候选轨迹 $\tau_i$ 与过期 VLM 选择 $\tau_{vlm}$ 的几何距离。仅对比剩余轨迹部分，忽略已执行的尾部。
  - **指数衰减（Staleness Decay）**：$w(\Delta t) = \exp(-\Delta t / \tau_{decay})$。随着 VLM 反馈的老化，其影响力呈指数级下降，保证了系统在没有最新指令时能平滑回退到规划器本身。
  - **融合策略**：
    - **Score Fusion**：直接在规划器分数上添加加权相似度项。
    - **Probability Fusion**：将规划器分布与 VLM 相似度分布按 $\alpha$ 进行混合，约束 VLM 的最大影响权重，使系统更稳健。

### 4. 方法对比分析
- **本质区别**：与端到端模型不同，本方法保留了高性能的传统局部规划器，仅在“决策层”引入 VLM，实现了语义推理与实时控制的解耦。
- **创新贡献**：提出了一种无需训练的轨迹级融合机制，通过几何相似度实现跨时间步的语义信息对齐，完美规避了 VLM 高延迟导致的控制抖动问题。
- **适用场景**：适用于城市人行道、园区等需要识别复杂场景（如行人交互、交通标志、地形边界）的轮式机器人平台。

### 5. 实验分析（精简版）
- **关键结论**：在“困难”场景中，VLM 选择相比规划器 Argmax 降低了 30% 的 ADE。
- **主要优势**：极强的延迟适应性（5秒延迟下仍有80%成功率），且支持“即插即用”，无需对特定下游任务进行 VLA 微调。
- **主要局限**：如果规划器的原始候选集中完全不包含正确路径，VLM 无法无中生有地创造新轨迹。

### 6. 实用指南
- **迁移建议**：本方法非常适合任何基于“候选轨迹采样”的局部规划框架（如 MPPI, DWA）。迁移时，只需将 VLM 的输出映射为轨迹索引，并计算几何相似度函数即可。
- **工程细节**：重点关注 `tau_decay` 和 `lambda` 两个超参数，实验证明它们具有较宽的稳定区间（Stable Plateau），无需复杂的调参。

### 7. 总结
- **核心思想**：通过语义引导的高速缓存融合，利用过期智能驱动实时导航。
- **速记版 Pipeline**：
  1. **采样**：规划器每帧生成多条可行轨迹。
  2. **推理**：VLM 异步选择最优轨迹索引。
  3. **匹配**：根据几何相似度，将陈旧建议转化为当前轨迹得分。
  4. **决策**：加权融合后选出当前最佳动作。

**Key Findings:**

- Rather than replacing the planner with an end-to-end Vision-Language-Action model, we propose a VLM-Planner interface that uses a VLM to select a candidate index from the planner's proposal set and then fuse it with the planner's initial output.
- We demonstrate the full system on a mobile robot navigating challenging campus sidewalks with varied network latency.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20458v1)
- [arXiv](https://arxiv.org/abs/2606.20458v1)

---

<a id='2606.20426v1'></a>
## [TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation](https://arxiv.org/abs/2606.20426v1)

**Authors:** Hengfei Zhao, Yifan Xie, Junhao Gong, Yue Sun, Kai Zhu, Weihua He, Shoujie Li, Haohuan Fu, Wenbo Ding

**Published:** 2026-06-18

**Categories:** cs.RO

**Abstract:**

Vision-based tactile sensors require high-fidelity simulation for reinforcement learning, yet existing approaches struggle to provide accurate mechanical stress fields within GPU-accelerated robotics platforms. We present TaCauchy, an extensible Finite Element Method (FEM) framework that integrates rigorous physics-based force computation into Isaac Sim. Built on the Unified Incremental Potential Contact (UIPC) solver, TaCauchy directly computes Cauchy stress tensors from hyperelastic constitutive laws and projects them onto contact surfaces to obtain traction forces and pressure distributions, providing mechanical ground truth from first principles rather than empirical estimation. Our framework features automatic mesh generation with geometry-aware adaptive refinement and a modular sensor interface enabling rapid integration of diverse sensors (GelSight Mini, DIGIT, 9DTact) with minimal configuration. Performance benchmarks demonstrate 33.40 FPS for single environments and 555 FPS aggregate throughput across 60 parallel environments, with stress extraction overhead under 1 ms. Physical validation experiments show strong agreement between simulated and real tactile responses across force ranges from 1.2556 N to 4.7332 N, achieving SSIM above 0.93, confirming the framework's capability to provide accurate, physically-grounded force supervision for downstream robotic manipulation tasks.

**Analysis:**

### 1. 摘要翻译
视觉触觉传感器需要高保真的仿真以支持强化学习，但现有方法在GPU加速的机器人平台中难以提供精确的机械应力场。我们提出了TaCauchy，这是一个可扩展的有限元方法（FEM）框架，将严格的物理力计算集成到Isaac Sim中。基于统一增量势接触（UIPC）求解器，TaCauchy通过超弹性本构律直接计算柯西应力张量，并将其投影到接触面上以获得牵引力和压力分布——从而提供基于第一性原理而非经验估计的力学真值。我们的框架具备几何感知自适应细化的自动网格生成功能，以及支持多种传感器（GelSight Mini, DIGIT, 9DTact）的模块化接口，且配置极简。性能基准测试显示，单环境运行速度为33.40 FPS，60个并行环境下的总吞吐量为555 FPS，应力提取开销低于1毫秒。物理验证实验表明，模拟与真实触觉响应在1.2556 N至4.7332 N的力范围内高度一致，SSIM值超过0.93，证实了该框架能够为下游机器人操作任务提供准确、物理上扎实的力监督。

### 2. 方法动机分析
*   **驱动力**：为大规模机器人强化学习提供高精度、物理一致的触觉反馈数据，解决“sim-to-real”鸿沟。
*   **现有方法痛点**：
    *   **仿真精度低**：仅依赖外观渲染（如TACTO、Taxim）无法获取力学数据。
    *   **计算效率瓶颈**：现有的FEM仿真（如TacIPC、DIFFTACTILE）难以无缝接入Isaac Sim等高吞吐量并行仿真生态。
    *   **缺乏物理严谨性**：简化接触模型或经验估计无法支撑高频、接触丰富的复杂操作任务。
*   **研究假设**：通过在Isaac Sim中集成基于UIPC的FEM求解器，并利用柯西应力张量直接从物理模型中导出表面牵引力，可以在保证计算性能的同时获得准确的力学真值。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **自动网格生成**：使用WildMeshing算法，根据传感器几何形状自动生成四面体网格。通过几何感知自适应细化（Adaptive Refinement），在接触区域设置较细的边缘长度（$l_c \approx 0.8 \text{ mm}$），在非接触区设置较粗的边缘（$l_g \approx 3.2 \text{ mm}$），以平衡计算开销与精度。
    2.  **物理仿真 (UIPC)**：采用Stable Neo-Hookean超弹性模型，该模型能精准描述硅胶类材料的非线性、不可压缩性及大变形特性。UIPC求解器处理接触，保证无穿透、无倒置。
    3.  **柯西应力提取**：从 deformation gradient $\mathbf{F}$ 计算 First Piola-Kirchhoff 应力 $\mathbf{P}$，再转换得到柯西应力张量 $\mathbf{\sigma} = \frac{1}{J} \mathbf{PF}^T$。
    4.  **表面力分解**：利用柯西应力定理（$\mathbf{t} = \mathbf{\sigma} \cdot \mathbf{n}$），将接触面上的应力张量投影为法向压力 $p_n$ 和切向牵引力 $\mathbf{t}_\tau$。
    5.  **光学渲染**：结合FEM计算出的变形面，通过物理约束的深度图生成 tactile images，确保力学仿真与视觉反馈的一致性。

### 4. 方法对比分析
*   **本质区别**：与仅做视觉拟合的方法不同，TaCauchy是“物理驱动的视觉感知”框架；与独立FEM仿真不同，它完全嵌入Isaac Sim并行加速生态。
*   **创新贡献**：首次实现了从连续介质力学应力张量到高保真触觉图像的一体化计算，确立了触觉感知的物理真值计算范式。
*   **适用场景**：涉及精密力控制、滑移检测、复杂接触任务的强化学习训练。

### 5. 实验分析（精简版）
*   **验证方法**：利用UR5机械臂配合力传感器进行受控挤压实验，对比真实传感器图像与仿真渲染图像。
*   **关键结果**：在各力级下，SSIM均值 > 0.93，PSNR均值约为22.13 dB，证实了仿真逼真度。
*   **主要优势**：兼具计算效率（<1 ms应力提取）与物理严谨性。
*   **主要局限**：目前难以实时更新或建模硅胶的随时间变化的磨损与迟滞特性。

### 6. 实用指南
*   **开源/实现**：基于Isaac Sim 5.1.0及UIPC库开发，建议关注论文提及的UIPC-FEM求解器文档。
*   **实现细节**：关键参数是自适应网格的边缘长度比例（$l_r$）和几何包络参数（$\epsilon_r$），需根据传感器材质模量进行精细调节。
*   **迁移建议**：该框架高度模块化，只需提供新的传感器CAD模型和标定参数，即可在TaCauchy框架下快速适配不同规格的传感器。

### 7. 总结
*   **核心思想**：基于第一性原理的FEM解算与并行仿真生态的深度融合。
*   **速记版pipeline**：1. 生成适配接触区域的优化网格；2. 执行超弹性物理力学仿真；3. 从应力张量中解析压力与摩擦力；4. 基于仿真变形渲染视觉图像。

**Key Findings:**

- We present TaCauchy, an extensible Finite Element Method (FEM) framework that integrates rigorous physics-based force computation into Isaac Sim.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20426v1)
- [arXiv](https://arxiv.org/abs/2606.20426v1)

---

<a id='2606.20389v1'></a>
## [CoLI: A Reproducible Platform for Continuum Robot Learning via Monolithic 3D Printing and Isomorphic Teleoperation](https://arxiv.org/abs/2606.20389v1)

**Authors:** Ziyuan Tang, Chenxi Xiao*

**Published:** 2026-06-18

**Categories:** cs.RO

**Abstract:**

Continuum robots offer strong potential for manipulation tasks due to their high degrees of freedom, compliant structures, and operational safety. However, their adoption in both research and practical applications has been hindered by reproducibility issues arising from complex fabrication and assembly processes, challenging kinematic modeling, and a lack of intuitive control interfaces. To address these challenges, we present a novel open-source continuum robot design. The platform features a simplified fabrication pipeline enabled by multi-material 3D printing, allowing the arm to be fabricated as a monolithic compliant structure with minimal assembly. Control is achieved through an isomorphic teleoperation interface that establishes a direct actuator-level mapping, eliminating the need for explicit kinematic modeling and providing a singularity-free mapping. Building on this hardware design, the platform further supports imitation-learning-based autonomous control. The proposed system is evaluated through hardware characterization and a set of manipulation tasks. Experimental results demonstrate that the platform provides a reproducible, learning-ready continuum robot system, accelerating algorithmic development and systematic benchmarking for the continuum robotics community.

**Analysis:**

### 1. 摘要翻译
连续体机器人因其高自由度、顺应性结构和操作安全性而在操作任务中展现出巨大潜力。然而，复杂制造和装配流程、极具挑战的运动学建模以及缺乏直观控制接口等可复现性问题，严重阻碍了其在研究和实际应用中的普及。为解决这些挑战，我们提出了一种新型开源连续体机器人设计。该平台采用多材料3D打印技术，实现了简化的制造流水线，使机械臂能够以单体顺应性结构制造，且仅需极少装配。控制方面，通过同构遥操作接口建立了直接的执行器级映射，无需显式运动学建模，并提供了无奇异点的映射。基于此硬件设计，该平台进一步支持基于模仿学习的自主控制。硬件特性和操作任务的实验结果证明，该平台提供了一个可复现、可学习的连续体机器人系统，为连续体机器人社区的算法开发和系统性基准测试提供了有力支持。

---

### 2. 方法动机分析
*   **驱动力**：旨在降低连续体机器人进入门槛，使其能够像刚性机器人一样轻松进行大规模数据采集和学习。
*   **痛点**：传统方法存在三个瓶颈：1.制造高度复杂（需多组件精密装配）；2.控制依赖于高难度的运动学建模；3.缺乏适配机器人学习框架（如LeRobot）的标准化开源软硬件平台。
*   **研究直觉**：通过“硬件同构（Mono-material printing）+ 控制同构（Direct-actuator mapping）”，将复杂的连续体机器人简化为类似于“高自由度关节机器人”的拓扑结构，从而绕过复杂的解析模型。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **单体制造**：使用Bambu Lab H2D多材料打印机，一体化打印硬质（PLA）间隔盘与柔性（TPU）骨架。
    2.  **同构映射**：构建完全一致的“领航者（Leader）- 追随者（Follower）”机器人对。领航者具备高背驱性（利用低减速比电机），直接被操作员变形。
    3.  **直接驱动**：将领航者电机编码器数据直接映射为追随者电机的PWM/位置指令，公式为：$q_i^f = (q_i^l - q_{i,0}^l) + q_{i,0}^f$。
    4.  **模仿学习**：利用上述接口采集高一致性Demonstration，直接喂入Action Chunking with Transformers (ACT) 模型进行策略训练。
*   **关键点**：抛弃了传统的“笛卡尔空间->运动学逆解->关节空间”路径，直接在“执行器空间”进行端到端数据传输，完全规避了连续体机器人因柔性导致的非线性运动学建模难题。

---

### 4. 方法对比分析
*   **本质区别**：从“模型驱动（Model-based）”转向“数据驱动（Data-driven）”，将连续体机器人视为一个高自由度的黑盒执行器。
*   **创新点**：
    1.  **硬件层**：基于单体多材料3D打印的制造方案，极大地提升了可复现性。
    2.  **控制层**：提出物理层面的同构映射，通过硬件结构的对称性消除了算法实现的复杂性。
*   **适用场景**：适用于实验室环境下的机器人操作学习、复杂环境探索任务及对连续体机器人感兴趣的快速迭代开发。

---

### 5. 实验分析
*   **核心结论**：系统在15小时运行中保持结构稳定，且能精准实现1kg负载下的动态操作。在自主抓取、推杆等任务中，模仿学习成功率达到83%以上。
*   **优势**：极低的软硬件部署门槛；无需运动学建模即可实现高精度控制；完美兼容LeRobot框架。
*   **局限**：受3D打印机体积限制，单体长度有限；高性能伺服电机成本较高。

---

### 6. 实用指南
*   **开源情况**：代码和硬件设计已发布在 [tangrobot.github.io/CoLI-website/](https://tangrobot.github.io/CoLI-website/)。
*   **迁移与实现**：
    *   **关键超参数**：打印时PLA 25%填充，TPU 100%填充是确保柔性顺应性的关键；校准时PWM设置（25%和10%）是保证预紧力的核心。
    *   **迁移建议**：若要迁移到其他形态的连续体机器人，只需保持Leader和Follower硬件拓扑结构的拓扑一致性，并重新校准零点偏移即可。

---

### 7. 总结
*   **核心思想**：通过硬件与控制的同构设计，用结构一致性消解连续体机器人的建模复杂性。
*   **速记版Pipeline**：
    1. **打印一体化骨架**：一次性产出核心柔性本体。
    2. **搭建对称控制台**：配置完全相同的领航与追随机器人。
    3. **物理同步映射**：直接利用领航者的运动数据驱动追随者，无需数学模型。
    4. **模仿学习训练**：将采集的动作序列接入ACT模型进行策略部署。

**Key Findings:**

- To address these challenges, we present a novel open-source continuum robot design.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.20389v1)
- [arXiv](https://arxiv.org/abs/2606.20389v1)

---

