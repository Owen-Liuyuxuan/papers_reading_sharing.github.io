time: 20260721

# Arxiv Computer Vision Papers - 2026-07-21

## Executive Summary

### 每日报告执行摘要：2026年7月20日 Arxiv 计算机视觉论文

#### 1. 主要主题与趋势
本期论文集中呈现两大趋势：**具身智能与机器人学习**（6篇）和 **4D/场景重建与理解**（2篇），其余涉及生成模型与全景分割。具身智能方向明显向**视觉-语言-动作（VLA）模型**与**长期任务规划**深化，强调从仿真到真实的迁移、接触力融合及多策略协调。4D重建方面则关注**从自我中心视频进行观察者-场景联合重建**，以及**通用场景压缩**。

#### 2. 显著或创新论文
- **“Three-Body Scattering for Generative Modeling”**：将物理学中的三体散射问题引入生成模型，理论视角新颖，可能开辟基于动力学的生成范式。
- **“ReViV”**：首次从单目自我中心视频同时重建观察者（例如头部/身体）与场景的4D动态，对增强现实和第一人称分析有重要意义。
- **“FM-VLA”**：将力/触觉记忆融入VLA模型，针对高接触操作任务，解决了纯视觉在精细操控中的不足。
- **“RoboHarness”**：提出内存驱动的异构机器人策略编排框架，支持长期任务规划，是解决多技能协调的关键进展。

#### 3. 新兴研究方向与技术
- **物理启发的生成模型**：三体散射为生成建模提供新数学工具，可能影响扩散模型或流模型的演变。
- **基于力/触觉的VLA记忆**：将接触力作为模态引入机器人学习，提升对变形、抓取等操作的可控性。
- **持久3D物体标记**：在闭环人形机器人VLA中引入可验证的3D token，实现长期定位与操作的一致性。
- **内存驱动的多策略编排**：异构机器人策略（如导航、抓取、放置）通过共享记忆协同完成长时任务，是迈向通用机器人的关键一步。
- **无配对域转换缩小Sim-to-Real差距**：结合反向动力学提取，无需配对数据即可迁移策略，降低真实数据采集成本。

#### 4. 建议精读的论文
- **ReViV**：对4D人体与场景重建领域具有标杆意义，适合关注AR/VR或第一人称理解的研究者。
- **Three-Body Scattering**：理论创新性强，可能引发生成模型新路线，适合对生成方法和数学物理结合感兴趣者。
- **FM-VLA**：接触操作是机器人实用化的关键瓶颈，该文提供可操作方案，适合具身智能和操作研究者。
- **RoboHarness**：长期规划与策略编排是当前机器人系统的核心挑战，该文提出系统级解决方案，适合机器人规划与控制领域。

---

## Table of Contents

1. [ReViV: Reconstructing the Viewer and the View in 4D from Monocular Egocentric Video](#2607.17790v1)
2. [Patch Policy: Efficient Embodied Control via Dense Visual Representations](#2607.18236v1)
3. [FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation](#2607.18231v1)
4. [Three-Body Scattering for Generative Modeling](#2607.18198v1)
5. [World Translation: Minimizing Sim-to-Real Gap with Backward Dynamics Extraction and Unpaired Domain Translation](#2607.18154v1)
6. [Plenoptic Condensation: A Novel Approach to Generalized Scene Reconstruction](#2607.18151v1)
7. [Occlusion-Aware Panoptic Segmentation with Joint Position Embedding and Occlusion-Level Attention](#2607.18112v1)
8. [RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning](#2607.18060v1)
9. [Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation](#2607.18016v1)
10. [RynnBrain 1.1: Towards More Capable and Generalizable Embodied Foundation Model](#2607.17977v1)

---

## Papers

<a id='2607.17790v1'></a>
## [ReViV: Reconstructing the Viewer and the View in 4D from Monocular Egocentric Video](https://arxiv.org/abs/2607.17790v1)

**Authors:** Xiaozhong Lyu, Gen Li, Zhiyin Qian, Xucong Zhang, Marc Pollefeys, Siyu Tang

**Published:** 2026-07-20

**Categories:** cs.CV, cs.AI

**Abstract:**

Egocentric devices, such as wearable front-facing cameras, provide a unique perspective for capturing the continuous interaction between a human viewer and the surrounding environment. A holistic and efficient multimodal model capable of reconstructing this 4D representation is therefore highly desirable. However, existing approaches often rely on auxiliary inputs such as pre-computed camera trajectories, treat scene perception and human ego-motion modeling as separate problems despite their strong interdependency, and suffer from slow inference time. To address these limitations, we present ReViV, the first unified framework for holistic egocentric 4D reconstruction that extracts both viewer and view dynamics from a single monocular RGB video. We formulate the task as learning the full joint probability distribution over multimodal signals, including RGB video, camera trajectory, gaze direction, full-body motion, hand motion, and depth. Powered by a Masked Generative Egocentric Transformer, ReViV operates within a single feed-forward architecture to simultaneously reconstruct the temporally consistent 4D reconstruction across the viewer and the view with fast inference speed. Extensive experiments on diverse benchmarks, including HoloAssist, HOT3D, ARCTIC, Aria Digital Twin, and TACO, demonstrate that ReViV achieves state-of-the-art accuracy and efficiency across holistic ego-body, hand, and gaze reconstruction, camera tracking, while maintaining highly competitive egocentric depth estimation without relying on heavy task-specific priors. Code and models are fully open-sourced: https://reviv4d.github.io/.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **ReViV** 这篇论文的分析如下：

### 1. 核心贡献摘要
ReViV 提出了首个统一的框架，仅通过单目 RGB 视频即可同时实现对“观察者（Viewer）”与“环境场景（View）”的 4D 协同重建。该模型打破了传统方法将相机轨迹、人体姿态、手部动作和深度估计视为独立任务的局限，通过单一的前馈架构实现了高效、时间一致的整体式（Holistic）自我中心场景解析。

### 2. 关键创新与方法论
*   **统一的概率建模：** 该研究的核心在于将该任务建模为多模态信号（RGB、相机轨迹、视线、全身动作、手部动作、深度）的“联合概率分布学习”。这种方式利用了各模态之间强烈的内在耦合性，而非简单地堆叠多个独立模型。
*   **Masked Generative Egocentric Transformer (MGET)：** 论文采用了一种基于掩码生成式的 Transformer 架构。这种设计允许模型在处理缺失信息或不确定性时具备更强的鲁棒性，同时通过单次前馈（Single feed-forward）极大优化了推理速度，解决了以往基于迭代优化方法（如非线性优化或长耗时多阶段处理）效率低下的问题。
*   **端到端协同重建：** 不同于依赖预计算轨迹或特定先验知识的方法，ReViV 证明了在一个统一模型中联合推断自我运动（Ego-motion）与环境动态的可行性。

### 3. 对领域的潜在影响
*   **从“任务解耦”到“范式融合”：** ReViV 代表了从“流水线式（Pipeline）”计算机视觉向“统一表征学习（Unified Representation Learning）”范式的转变。它证明了复杂的 4D 重建任务可以通过大规模 Transformer 架构实现端到端的涌现，这对于未来构建“通用具身智能”系统具有重要的参考意义。
*   **效率与实时性的突破：** 能够以极快速度进行 holistic 4D 重建，为实时 AR/VR 设备（如眼镜类穿戴设备）的落地提供了可能性。它消除了对昂贵算力或繁琐离线计算的依赖，直接提升了用户交互体验的上限。

### 4. 相关应用领域
*   **增强现实（AR）与虚拟现实（VR）：** 对于混合现实中的数字孪生构建、虚拟物体遮挡处理以及实时交互，该研究提供了即插即用的重建基础。
*   **具身智能与机器人学：** 对于配有前置摄像头的移动机器人，ReViV 能够帮助其理解自身运动与周围环境的动态关系，是实现自主导航与复杂物体操控的关键能力。
*   **辅助视觉（Assistive Tech）：** 在辅助视障人士或进行远程专家协作（如 HoloAssist 场景）时，能够实时重建视场内的人机动作具有深远应用价值。

### 5. 可推断的局限性
*   **长序列稳定性（Long-term drift）：** 尽管采用了生成式建模，但在超长视频序列中，单目 RGB 重建在尺度感知（Scale ambiguity）和漂移（Drift）问题上可能仍存在潜在风险。
*   **极端环境的泛化性：** 虽然在多项数据集上进行了验证，但作为生成式模型，在训练数据分布之外的极端光照、强遮挡或非典型视角的场景下，模型是否仍能保持高精度的物理一致性尚需观察。
*   **数据需求：** 这种强大的统一模型通常对训练数据的多样性和标注质量要求极高，构建此类数据集的成本是该技术进一步推广的潜在门槛。

**总结建议：** ReViV 是 4D 场景解析领域的一次重要跨越。对于关注“具身感知（Embodied Perception）”的研究者来说，该论文展示了如何利用生成式 Transformer 处理多模态时空约束，是构建未来多模态感知代理（Perceptual Agents）的重要前沿工作。

**Key Findings:**

- To address these limitations, we present ReViV, the first unified framework for holistic egocentric 4D reconstruction that extracts both viewer and view dynamics from a single monocular RGB video.
- Extensive experiments on diverse benchmarks, including HoloAssist, HOT3D, ARCTIC, Aria Digital Twin, and TACO, demonstrate that ReViV achieves state-of-the-art accuracy and efficiency across holistic ego-body, hand, and gaze reconstruction, camera tracking, while maintaining highly competitive egocentric depth estimation without relying on heavy task-specific priors.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.17790v1)
- [arXiv](https://arxiv.org/abs/2607.17790v1)

---

<a id='2607.18236v1'></a>
## [Patch Policy: Efficient Embodied Control via Dense Visual Representations](https://arxiv.org/abs/2607.18236v1)

**Authors:** Gaoyue Zhou, Zichen Jeff Cui, Ada Langford, Bowen Tan, Yann LeCun, Lerrel Pinto

**Published:** 2026-07-20

**Categories:** cs.RO, cs.LG

**Abstract:**

Pretrained dense visual features from Vision Transformers (ViTs) are powerful yet have been underutilized in robot learning. Modern robot policies either compress each observation into a single global token, or rely on visual backbones trained from scratch, sacrificing both fine-grained spatial detail and the benefits of large-scale visual pre-training. While there exist policies that do operate on dense patch features like large vision-language-action models (VLAs), they tend to be heavy and slow, inheriting the full cost of a billion-parameter vision-language model (VLM) backbone. We close this gap with Patch Policy, a minimal architectural extension that enables transformer-based policies to consume dense pre-trained patch tokens directly without the computational overhead of a full VLM. At its core is a block-causal attention mask that preserves the temporal causality of standard policies while letting the model attend over many patch tokens per observation, alongside other state information. Patch Policy is lightweight, fast, and highly effective. Across four simulated and three real-world environment suites, our method achieves a 40% relative improvement over policies using state-of-the-art global-pooled representations. Furthermore, it surpasses fine-tuned OpenVLA-OFT by 18% while using roughly 0.7% of the parameters. We believe Patch Policy provides a pipeline for the robotics community to readily leverage continuing progress in visual representation learning, without sacrificing the training efficiency or inference speed required for high-frequency, reactive control. Videos can be viewed at https://patch-policy.github.io

**Analysis:**

作为计算机视觉与机器人学习领域的专家，我对《Patch Policy: Efficient Embodied Control via Dense Visual Representations》一文的分析如下：

### 1. 核心贡献摘要
该论文提出了 **Patch Policy**，一种旨在解决机器人策略在“全局特征压缩”（丢失细节）与“大模型全量特征提取”（计算开销大）之间两难问题的架构。通过引入一种轻量级的架构扩展和特殊的注意力掩码机制，该方法使机器人策略能够直接利用预训练 ViT 的密集 Patch 特征，在显著降低参数规模和计算成本的同时，实现了优于大模型（如 OpenVLA）的控制表现。

### 2. 关键创新与方法论
*   **分块特征直接输入（Dense Patch Features）：** 不同于传统的 Global Pooling（将图像压缩为一个特征向量），该方法保留了 ViT 提供的空间精细信息，这对于机器人完成诸如精确抓取、避障等依赖视觉细节的任务至关重要。
*   **块因果注意力掩码（Block-Causal Attention Mask）：** 这是该方法的核心算法创新。通过设计特殊的注意力掩码，模型在处理多 Patch 输入和状态信息时，既能保持标准策略所需的“时序因果性”（Temporal Causality），又能有效处理高维度的视觉 Token，避免了传统 Transformer 在处理长序列时的效率瓶颈。
*   **架构极简化（Architectural Minimalism）：** 避开了全参数微调 VLM 的路径，仅通过极小的架构插件扩展，实现了与大模型相当甚至更好的感知精度，仅使用了约 0.7% 的参数量。

### 3. 对该领域的潜在影响
*   **打破“大即是美”的范式：** 证明了机器人控制并不一定需要依赖数十亿参数的 VLM。通过高效利用预训练特征，轻量级模型完全可以胜任高频响应的控制任务。
*   **提升实时性与可部署性：** 该研究为高频、实时机器人控制（High-frequency, reactive control）提供了切实可行的技术路径。对于工业机器人或移动平台，这种轻量级的高性能模型具有极高的商业转化潜力。
*   **视觉表示学习的红利：** 搭建了一座桥梁，使得机器人社区无需重新训练视觉主干，就能直接吸收计算机视觉领域在预训练模型（如 DINOv2 等）上的最新进展。

### 4. 相关领域或潜在应用
*   **高频精细操作（Fine-grained Manipulation）：** 如电子装配、手术机器人等，对视觉空间精度要求极高的场景。
*   **端侧机器人部署：** 在算力受限的嵌入式设备（如边缘计算板卡）上运行高性能视觉策略。
*   **具身智能体学习：** 对空间感知要求高的导航、避障类机器人，以及需要理解复杂环境语义的移动平台。

### 5. 可推断的局限性
*   **跨模态泛化能力：** 虽然该方法擅长利用视觉特征，但其对自然语言指令（Instruction-following）的理解是否达到类似 OpenVLA 的多模态对齐程度，仍有待考察（Patch Policy 侧重于高效视觉决策，而非复杂的视觉-语言推理）。
*   **Patch 数目的上限：** 虽然使用了块因果掩码，但在处理多视角、超高清视频输入时，随着 Patch 数目的进一步增加，Transformer 的注意力计算复杂度仍可能成为新的瓶颈。
*   **对预训练特征的依赖：** 该方法的性能上限在很大程度上取决于所选取的视觉预训练主干（Backbone）的质量，如果下游任务的领域分布与预训练数据集差异过大，其表现可能受限。

**总结：**
这篇论文的趣味性在于它提出了一种“四两拨千斤”的设计哲学。在当前大模型浪潮中，很多研究倾向于堆砌参数和计算资源，而 Patch Policy 回归了控制的本质——**高效、准确、实时**，并巧妙利用了 CV 领域现有的密集视觉特征红利，为具身智能的落地提供了极具价值的技术范式。

**Key Findings:**

- Across four simulated and three real-world environment suites, our method achieves a 40% relative improvement over policies using state-of-the-art global-pooled representations.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18236v1)
- [arXiv](https://arxiv.org/abs/2607.18236v1)

---

<a id='2607.18231v1'></a>
## [FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation](https://arxiv.org/abs/2607.18231v1)

**Authors:** Ruicheng Li, Qixiu Li, Ruichun Ma, Yu Deng, Lin Luo, Zhiying Du, Jianfeng Xiang, Huizhi Liang, Ruicheng Wang, Jiaolong Yang, Baining Guo

**Published:** 2026-07-20

**Categories:** cs.RO

**Abstract:**

Vision-language-action (VLA) models have achieved impressive generalization in robotic manipulation, and recent memory-augmented VLAs have relaxed the Markovian assumption by conditioning on past images or language summaries. Vision-based memory approaches address this by conditioning on sampled past image frames, but they are computationally expensive and fundamentally limited when temporal events are visually ambiguous, e.g., pushing a button multiple times with small movements. We propose FM-VLA, a VLA model with force-based memory, enabling temporal context reasoning for non-Markovian, contact-rich manipulation. We encode force histories into compact force memory tokens with a variational autoencoder (VAE) pretrained with force time series reconstruction. By projecting force latent representations and short state history as additional conditioning tokens to the action expert module, we enable VLAs to leverage accumulated contact event history to guide manipulation. We evaluate FM-VLA on three memory-dependent tasks, including finding a hidden block, pressing a button, and wiping a dish for a specific number of times. Our lightweight force memory achieves over 80% success rate with minimal inference overhead, significantly outperforming baseline approaches. Project page: https://qft-333.github.io/FM-VLA-Page/

**Analysis:**

这是一份对 **FM-VLA (Force-based Memory for Vision-Language-Action Models)** 的深度技术分析。

---

### 1. 摘要翻译
视觉语言动作（VLA）模型在机器人操作领域展现了卓越的泛化能力，但现有的内存增强型VLA模型主要依赖于过去采样的图像帧。这些方法计算成本高，且在视觉模糊（如多次按压按钮）的接触丰富任务中存在局限性。我们提出了FM-VLA，这是一种具有力觉内存的VLA模型，能够通过接触历史进行非马尔可夫推理。我们将力历史编码为紧凑的内存Token，这些Token是由通过力时间序列重构预训练的变分自编码器（VAE）生成的。通过将力潜在表示与短时状态历史作为额外的调节Token注入动作专家模块，FM-VLA使模型能够利用积累的接触事件历史来指导操作。在需要记忆的任务（如寻找隐藏方块、按压按钮、擦拭碗）中，该轻量级力内存显著提升了成功率（超过80%），且推理开销极小。

### 2. 方法动机分析
*   **驱动力**：解决VLA模型在“接触丰富（Contact-Rich）”且“非马尔可夫”任务中的长程推理问题，特别是当视觉输入由于遮挡或场景微小变化（如按钮按压）无法提供任务进度信息时。
*   **现有方法痛点**：基于视觉的内存方案（如MEM）通过存储大量图像增加推理延迟；且仅关注当前观测的现有力增强方法（如TA-VLA）缺乏对长程交互过程的记忆。
*   **研究假设**：力/力矩传感器直接捕获了交互动态的真实状态，通过VAE对这些高维、结构化的时序数据进行自监督压缩，可以提取出对动作决策至关重要的交互状态表示。

### 3. 方法设计详解
*   **流程总结**：
    1.  **Stage 1 - 力VAE预训练**：使用Perceiver-IO结构，将原始wrench信号（3轴力+3轴力矩）输入VAE，通过Masked-ELBO目标进行重构训练，得到通用的力觉潜在空间。
    2.  **Stage 2 - VLA集成**：冻结VAE编码器，将长程Wrench历史通过编码器转化为 $K=8$ 个内存Token。同时，提取最后1秒的关节状态（Short state history）进行线性投影，得到一个状态Token。
    3.  **动作生成**：将上述共 $K+1$ 个Token直接拼接在动作专家（Action Expert）的Token序列尾部，通过交叉注意力机制参与动作推理。
*   **关键公式与设计**：公式(1)定义了记忆表示 $h_t$，采用Context-wise Concatenation。通过随机噪声预填充（Randomized noise pre-padding）防止模型利用序列长度作为进度信息的捷径，强迫编码器关注力信号本身的动力学模式。

### 4. 方法对比分析
*   **本质区别**：FM-VLA将力觉视为“长期记忆”而非“实时反馈”，通过VAE重构任务转化为语义信息，而非仅仅是特征提取。
*   **创新贡献**：首次提出将 proprioceptive-wrench 序列作为轻量级长程记忆。相比视觉Token，力内存仅需极少量的Token（8个），显著降低了内存负担。
*   **适用场景**：所有涉及接触交互、需要计数或状态持续追踪的机器人操作任务。

### 5. 实验分析
*   **验证方法**：在Find a Block、Push Buttons、Wipe Dishes三个任务上进行对比，其中Buttons任务需精准记忆按压次数。
*   **关键结果**：在Buttons任务中，FM-VLA达到72.2%成功率，远超基线的11.1%。整体平均成功率83.3%。
*   **优势**：极低的推理开销（仅增加3ms左右）和强大的非马尔可夫推理能力。
*   **局限**：目前的VAE瓶颈固定为8个Token，对于极长跨度的任务，其表示能力可能受限。

### 6. 实用指南
*   **开源/实现**：项目主页为 https://qft-333.github.io/FM-VLA-Page/。
*   **实现细节**：
    *   **预处理**：一阶EMA滤波是关键，用于去除高频噪声。
    *   **注入方式**：务必在“noisy-action tokens”之后拼接，以保持原始RoPE位置编码的一致性。
    *   **超参数**：推荐 $K=8$；对于VAE，free-bits $\lambda$ 设为0.5 nats以防止后验崩溃。
*   **迁移建议**：若要迁移至其他平台，需保证力/力矩传感器的采样率一致（文中为100Hz下采样至30Hz），并针对特定机器人重新收集演示数据进行VAE预训练。

### 7. 总结
*   **核心思想**：通过VAE将长程触觉历史浓缩为轻量级Token，增强VLA的状态追踪能力。
*   **速记版Pipeline**：
    1. 收集交互历史的力传感数据。
    2. 用VAE将力信号压缩为固定数量的记忆Token。
    3. 提取近期关节状态转化为单个状态Token。
    4. 将这些记忆与状态Token拼接给VLA模型进行最终决策。

**Key Findings:**

- We propose FM-VLA, a VLA model with force-based memory, enabling temporal context reasoning for non-Markovian, contact-rich manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18231v1)
- [arXiv](https://arxiv.org/abs/2607.18231v1)

---

<a id='2607.18198v1'></a>
## [Three-Body Scattering for Generative Modeling](https://arxiv.org/abs/2607.18198v1)

**Authors:** Peng Sun, Zhenglin Cheng, Deyuan Liu, Jun Xie, Xinyi Shang, Tao Lin

**Published:** 2026-07-20

**Categories:** cs.LG, cs.CV

**Abstract:**

Modern generative models typically rely on an adversarial critic, a prescribed noise-to-data path, or an autoregressive factorization. Instead, we show that a proper distributional energy can induce sample-level motion and provide direct regression supervision for a one-step generator. Three-Body Scattering Modeling (TBSM) for generation turns the energy distance into a constant-size per-projectile interaction: each projectile is attracted toward one real source and repelled from one independently generated source. Conditioned on the projectile and its condition, its expectation equals the $2$-Wasserstein gradient-flow velocity of $\frac12D_E^2(P_θ,Q)$. A batch of $B$ frozen-target events yields $O(B)$ sample-level losses, each using one reference for its condition instead of the minibatch-wide all-pairs field used by methods such as Drifting Models. Tracking this conditional expectation online can reduce field noise. Using scattering in frozen image features, TBSM trains one-step generators on ImageNet-256, achieving FID${}=2.23$ with pixel-space PixelDiT-XL and FID${}=1.63$ with latent-space DiT-XL at NFE${}=1$. We provide a design map relating diffusion-related supervision, Drift-like dynamics, and GAN-like objectives. These results establish tracked scattering as a route to high-dimensional one-step generation. Code: https://github.com/sp12138/TBSM.

**Analysis:**

### 1. 摘要翻译
现代生成模型通常依赖对抗性判别器、预设的噪声到数据路径或自回归因式分解。本文证明了适当的分布能量可以诱导样本级运动，并为单步生成器提供直接的回归监督。三体散射建模（TBSM）通过将能量距离转化为恒定大小的每粒子相互作用来生成图像：每个粒子被吸引向一个真实源，并排斥一个独立生成的源。条件化后，其期望等于能量距离二范数的一半（$\frac{1}{2}D_E^2$）的2-Wasserstein梯度流速度。TBSM通过冻结目标事件的批量采样实现了$O(B)$样本级损失，且无需小批量全对字段。通过在线追踪条件期望，该方法降低了场噪声。TBSM在ImageNet-256上训练单步生成器，在NFE=1时，像素空间模型达到FID=2.23，潜空间模型达到FID=1.63。本文还提供了一个连接扩散监督、漂移类动力学和GAN类目标的通用设计映射，确立了跟踪散射作为高维单步生成的一种新路径。

### 2. 方法动机分析
*   **驱动力**：作者旨在解决高维空间下单步生成模型难以获得高效、高质量监督信号的问题，避免使用耗时的扩散路径迭代或复杂的对抗性训练。
*   **痛点**：现有方法（如蒸馏、漂移模型）往往依赖于教师模型查询、复杂的噪声调度或昂贵的全局批量计算。
*   **研究假设**：通过将生成过程建模为粒子在能量场中的受力运动（即三体相互作用），可以利用局部、恒定计算量的监督信号直接指导生成器一步到位，从而实现高效采样。

### 3. 方法设计详解
*   **流程总结**：
    1.  **采样三元组**：对每个生成粒子 $x_p$，采样一个真实源 $x_r$ 和一个独立生成的源 $x_s$。
    2.  **计算散射向量**：基于归一化的方向（bearing）计算 $v_{scat} = b_r - b_s$，其中 $b_r$ 指向真实源，$b_s$ 指向生成的源。
    3.  **在线追踪**：由于 $v_{scat}$ 存在采样噪声，引入一个追踪器（Tracker）网络来拟合该散射向量的条件期望 $v_{trk} \approx \mathbb{E}[v_{scat} | x_p]$。
    4.  **目标回归**：生成器向 $x_p + v_{trk}$ 这个“分离目标”（detached target）进行单步回归。
*   **关键公式**：$v_{scat} = \frac{x_r - x_p}{\|x_r - x_p\|} - \frac{x_s - x_p}{\|x_s - x_p\|}$。该公式本质上是2-Wasserstein梯度流的近似，通过分离目标消除了生成器梯度回传时的复杂依赖。

### 4. 方法对比分析
*   **本质区别**：TBSM将生成问题转化为“粒子动力学”而非“判别/去噪”问题，监督信号是针对每个粒子的局部相互作用，而非全局统计量。
*   **创新贡献**：引入“三体散射”视角和在线追踪机制，证明了能量距离梯度可以分解为独立的粒子更新，无需教师模型或噪声序列。
*   **适用场景**：适用于需要单步推理的高维图像生成任务，尤其是在计算资源受限且追求高质量生成的场景。

### 5. 实验分析
*   **关键结果**：在NFE=1的ImageNet-256基准上，TBSM达到了与复杂多步模型相当的FID性能（1.63 - 2.23）。
*   **优势**：训练过程去除了对抗损失和复杂的时间表，推理速度极大提升（NFE=1）。
*   **局限**：在高维复杂任务中，虽然单步效果优秀，但可能难以达到极深层迭代模型（如250+步扩散模型）的极致精细度；且需要较好的预训练模型初始化（Warm-start）。

### 6. 实用指南
*   **开源**：代码已开源（`https://github.com/sp12138/TBSM`）。
*   **关键细节**：
    *   **数值稳定性**：在计算 bearing 时需添加微小项 $\epsilon = 10^{-6}$ 以防止除零。
    *   **预训练权重**：建议利用现有的多步扩散模型Checkpoint进行初始化，而非随机训练。
    *   **配置**：推荐使用 `AdamW` 优化器，设置生成器学习率为 $10^{-5}$，追踪器学习率为 $10^{-3}$。
*   **迁移**：该方法逻辑通用，可直接迁移至任何能定义Representation-space的生成任务（如视频生成、音频生成），只需替换对应的Frozen Encoder。

### 7. 总结
*   **核心思想**：通过受限的三体粒子相互作用，直接驱动生成器一步生成分布。
*   **速记版pipeline**：
    1. 生成一个粒子；
    2. 对比真实数据与模型生成数据，计算受力方向；
    3. 用小神经网络平滑记录受力趋势；
    4. 调整生成器向平滑后的受力方向移动。

**Key Findings:**

- Instead, we show that a proper distributional energy can induce sample-level motion and provide direct regression supervision for a one-step generator.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18198v1)
- [arXiv](https://arxiv.org/abs/2607.18198v1)

---

<a id='2607.18154v1'></a>
## [World Translation: Minimizing Sim-to-Real Gap with Backward Dynamics Extraction and Unpaired Domain Translation](https://arxiv.org/abs/2607.18154v1)

**Authors:** Xinchen Yao, Leixin Chang, Hua Chen

**Published:** 2026-07-20

**Categories:** cs.RO

**Abstract:**

The gap between simulation and reality remains a fundamental challenge in deploying simulation-trained robotic policies in the real world. Real-to-sim methods narrow this gap from the real side, learning transition dynamics from real data to build a more realistic digital world. Learned dynamics models are their dominant instance. Such methods, however, face a partial observability problem: the same observation may branch to different transitions due to unobservable factors. Existing methods assume these factors can be recovered from observation history. However, this may fail whenever observation history is uninformative, such as a sudden contact event with no prior warning. To address this limitation, we propose \textit{World Translation}, which exploits a complementary strength of simulators and learned dynamics. Simulators are deterministic but physically imperfect, while learned models are accurate but underdetermined under partial observability. Rather than predicting transitions forward from history, we extract the unobservable dynamics information backward from an observed transition, then translate this feature across simulation and reality as an unpaired domain-translation problem that preserves dynamics content while transferring domain style. Experiments across humanoid, quadruped, and manipulator platforms show that our method achieves more accurate dynamics modeling than baselines, with the largest gains when unobservable factors cannot be recovered from observation history. Real-robot deployment on Go2 quadruped confirms improved policy transfer.

**Analysis:**

### 1. 摘要翻译
模拟与现实之间的鸿沟是部署模拟训练机器人策略的根本挑战。现有的“实转模”（Real-to-sim）方法试图通过学习真实数据中的转换动态来构建更真实的虚拟世界，但往往面临部分可观测性问题：由于存在不可观测因素，相同的观测结果可能导致不同的演变路径。现有方法假设这些因素可从观测历史中恢复，但这在接触等突发事件中往往失效。为此，我们提出了**世界翻译（World Translation）**框架，利用模拟器与学习模型的互补优势：模拟器物理完整但存在偏差，学习模型准确但部分可观测性下不确定。我们不预测历史演变，而是通过**逆向动力学提取**从观测到的转换中解析不可观测信息，并将其视为一个**非配对域翻译**问题，在跨域转换时保留动力学内容并迁移域风格。在人形、四足及机械臂平台的实验证明，该方法在不可观测因素严重时建模精度最高，且在Go2四足机器人上的部署显著提升了策略性能。

### 2. 方法动机分析
*   **驱动力**：解决机器人仿真到现实迁移（Sim-to-Real）中因“部分可观测性”导致的动态预测失效问题。
*   **痛点**：传统模型过度依赖历史观测。当面对如瞬时碰撞等“历史无关”事件时，预测会退化为平均值，导致策略鲁棒性下降。
*   **核心假设**：动力学的不确定性（即不可观测的隐藏变量 $h_t$）可以通过其产生的“结果”进行反向推断，且这种潜在特征可以在不同域之间进行风格迁移（Domain Translation）。

### 3. 方法设计详解
*   **逆向动力学提取 (Backward Dynamics Extraction)**：利用变分自编码器（VAE）处理 $(o_t, a_t, o_{t+1})$ 三元组。编码器通过未来状态 $o_{t+1}$ “逆向”推导出隐藏的特征 $z_t$。
*   **非配对域翻译 (Unpaired Domain Translation)**：引入 CycleGAN 框架。由于在仿真（S）和现实（R）中同一 $h_t$ 产生的现象不同，利用循环一致性（Cycle Consistency）学习映射 $G_{S \to R}$，将仿真中提取的动力学特征翻译为现实域的特征。
*   **模型协同**：
    *   **盲解码器（Blind Decoder）**：通过对抗训练防止编码器仅对观测值做简单映射（避开数据泄漏）。
    *   **领域分类器（Domain Classifier）**：强制 $z_t$ 编码具体的领域特性 $c$。
    *   **FiLM 层**：在解码器中使用特征级线性调制，确保动力学特征 $z_t$ 主导状态重构。

### 4. 方法对比分析
*   **本质区别**：传统方法是基于历史做“正向预测”，本项目是基于结果做“反向提取”，并引入了图像处理领域的“跨域翻译”思想来处理动力学特征。
*   **创新贡献**：将动力学建模从“预测”转化为“特征翻译”任务，有效解决了部分可观测场景下的不确定性归因问题。
*   **适用场景**：机器人与环境有复杂、不可测交互（如复杂地形、不可预见外力、载荷变化）的场景。

### 5. 实验分析（精简版）
*   **验证方法**：在Isaac Lab仿真环境下对比DirectPred、RawSim、RSSM等基线，并在Go2实体机上进行真实部署测试。
*   **关键结论**：在涉及高不确定性（如碰撞、外力）的任务中，世界翻译的建模精度远超基线，且在多步长预测下保持了极高的稳定性（约2%发散率）。
*   **主要优势**：不仅补足了缺失的动力学细节，还显著提升了下游策略的现实跟踪精度。
*   **主要局限**：对未建模的物理效应（如完全缺失的摩擦力维度）无法通过翻译“创造”出来，且训练需平衡VAE与CycleGAN的竞争目标，稳定性有一定挑战。

### 6. 实用指南
*   **实现要点**：必须使用离线采集的轨迹数据，且训练时禁用域随机化（Domain Randomization）以确保基准模拟环境的确定性。
*   **超参数**：重点调控盲解码器 penalty ($\lambda_b$) 与域分类器权重 ($\lambda_c$)，它们是解耦的关键。
*   **迁移建议**：该方法适用于任何存在“仿真模拟器”的机器人任务。通过替换数据域即可快速迁移，无需成对的仿真-现实样本。

### 7. 总结
*   **核心思想**：通过逆向推断与跨域翻译，将模拟动力学实时“翻译”为现实动力学。
*   **速记版pipeline**：
    1.  **逆向提取**：利用观测到的转换结果计算潜在隐藏状态。
    2.  **特征翻译**：通过对抗生成网络将仿真特征转换为现实域特征。
    3.  **状态覆盖**：在仿真运行中用翻译后的动力学重写下一步状态。
    4.  **策略微调**：基于更新后的“现实化”仿真轨迹对策略进行进一步优化。

**Key Findings:**

- To address this limitation, we propose \textit{World Translation}, which exploits a complementary strength of simulators and learned dynamics.
- Experiments across humanoid, quadruped, and manipulator platforms show that our method achieves more accurate dynamics modeling than baselines, with the largest gains when unobservable factors cannot be recovered from observation history.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18154v1)
- [arXiv](https://arxiv.org/abs/2607.18154v1)

---

<a id='2607.18151v1'></a>
## [Plenoptic Condensation: A Novel Approach to Generalized Scene Reconstruction](https://arxiv.org/abs/2607.18151v1)

**Authors:** Brevin Tilmon, Alex DeJournett, John Leffingwell, Scott Ackerson

**Published:** 2026-07-20

**Categories:** cs.CV

**Abstract:**

We present a novel Generalized Scene Reconstruction (GSR) approach called Plenoptic Condensation (PCon). PCon uses a multi-stage reconstruction pipeline, initially converting images into "soupy" scene elements with low (representational) power, then adaptively condensing the "soup" into "structured" elements of higher power capable of efficiently representing, for example, sharp edges and smooth reflective surfaces. PCon scene models called Reality Models (Relms) enable spatially varying representational power, which is essential for high-fidelity rendering, measurement, and scene understanding. We showcase several in-the-wild PCon reconstructions captured with consumer phone cameras and drones. In one case called "Damaged Fiat", PCon is benchmarked against two state-of-the-art (SOTA) GSR methods: NeRO and RT-Splatting. Referring to Figure 1 below, PCon reconstructs the car hood more than twice as accurately as the SOTA methods. But more importantly, the local damage profile error for PCon is 35 um (0.035 mm), whereas the two other SOTA methods are essentially unable to measure the damage at all. Our project website is available at https://quidient.github.io/pcon-2026.html.

**Analysis:**

### 1. 摘要翻译
我们提出了一种称为全光凝结（PCon）的新型广义场景重建（GSR）方法。PCon使用多阶段重建流水线，首先将图像转换为表示能力较低的“浓汤式”场景元素，然后通过自适应凝结将“浓汤”转化为表示能力更高的“结构化”元素，从而高效地表示尖锐边缘和光滑反射表面。PCon的场景模型（称为Reality Models™，简称Relms）支持空间变化的表示能力，这对高保真渲染、测量和场景理解至关重要。我们展示了多个使用消费级手机和无人机捕获的“野外”场景重建案例。在“受损菲亚特（Damaged Fiat）”基准测试中，PCon的重建精度是现有最优方法（NeRO和RT-Splatting）的两倍以上。更重要的是，PCon测得的局部损伤轮廓误差仅为35 µm，而现有最优方法几乎无法测量此类损伤。项目网站：https://quidient.github.io/pcon-2026.html。

### 2. 方法动机分析
- **驱动力**：解决现有GSR方法在处理高度反射表面和微小几何细节（如凹痕）时，难以同时满足高保真渲染（视觉）和精确度量（几何）的需求。
- **现有方法痛点**：NeRF/3DGS等现有方法倾向于将光照“烘焙”到外观中，导致严重的几何扭曲；现有的几何导向方法虽然能恢复表面，但在处理大范围复杂细节时容易产生过度平滑，且难以同时支持高质量重照明。
- **研究假设**：通过将“物质场（Matter Field）”与“光场（Light Field）”进行解耦，并引入自适应的“凝结”算子，可以根据局部复杂程度动态分配计算资源，从而在实现全局高保真重建的同时，精准捕捉微米级表面细节。

### 3. 方法设计详解
- **pipeline总结**：
  1. **输入阶段**：输入多视角图像及初始位姿（SfM生成）。
  2. **初级表示（Soupy）**：利用低表示能力的“浓汤”模型初步覆盖全局空间。
  3. **自适应凝结（Condensation）**：基于渲染残差（loss）驱动， coarse-to-fine 迭代。对高残差区域（如尖锐边缘、镜面高光点）进行“凝结”，从松散的初级元素升级为高阶、结构化的物质场基元（surfels/curvels）。
  4. **解耦建模**：将物质场（几何与材质）与光场（radiels，流经空间的光能分布）分离。
  5. **输出阶段**：直接读出一体化的Reality Model，支持直接导出包含PBR信息的网格。
- **核心算子**：凝结算子通过局部优化，将多视角下的辐照度观测压缩至单点物质描述（BSDF + 几何状态），从而实现物理意义上的解耦。

### 4. 方法对比分析
- **本质区别**：传统方法将视图相关外观“烘焙”进入表示，而PCon将光照（环境）与材质（物质）在模型底层即实现解耦。
- **创新贡献**：引入了“按需供能（Accuracy on demand）”的混合表示机制，在保证精度的同时优化了计算开销，是首个能同时满足视觉渲染与 metrology-grade（度量级）测量需求的方案。

### 5. 实验分析
- **关键结果**：在菲亚特引擎盖的损伤修复测试中，PCon将Chamfer距离指标提升至2.19mm（相比SOTA方法优势明显），且在微米级损伤识别上实现了从0到1的突破。
- **优势**：极高的几何重建精度（35 µm级误差）、支持重照明、原生支持PBR网格导出。
- **局限**：目前在大规模动态场景的表现（v2.0规划中）及复杂光照环境下的计算耗时相对较高。

### 6. 实用指南
- **开源/API**：该技术为Quidient私有引擎核心，通过API提供服务。
- **迁移建议**：该思路可以迁移到任何基于神经辐射场或高斯溅射的任务中，通过引入局部残差监测和分层物质更新，解决现有方法在处理镜面材质时出现的“伪影”和“平滑”问题。

### 7. 总结
- **核心思想**：通过分层凝结与光-物解耦，实现按需分配的高精度场景建模。
- **速记版pipeline**：
  1. 图像转化为初始低性能模型；
  2. 计算局部误差，标记待加强区域；
  3. 将残差区域升级为高保真结构化基元；
  4. 解耦物质与光照属性；
  5. 输出可直接渲染与测量的模型。

**Key Findings:**

- We present a novel Generalized Scene Reconstruction (GSR) approach called Plenoptic Condensation (PCon).
- We showcase several in-the-wild PCon reconstructions captured with consumer phone cameras and drones.
- In one case called "Damaged Fiat", PCon is benchmarked against two state-of-the-art (SOTA) GSR methods: NeRO and RT-Splatting.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18151v1)
- [arXiv](https://arxiv.org/abs/2607.18151v1)

---

<a id='2607.18112v1'></a>
## [Occlusion-Aware Panoptic Segmentation with Joint Position Embedding and Occlusion-Level Attention](https://arxiv.org/abs/2607.18112v1)

**Authors:** Wenbo Wei, Jun Wang, Shan Raza, Abhir Bhalerao

**Published:** 2026-07-20

**Categories:** cs.CV

**Abstract:**

Panoptic segmentation in complex scenes remains challenging because of occlusions, yet modern approaches often neglect occlusion modelling. In this paper, we propose \textbf{P}osition \textbf{E}mbedding \textbf{M}odulation with \textbf{O}cclusion-\textbf{L}evel \textbf{A}ttention (PEMOLA), a novel occlusion-aware module that can be seamlessly integrated into transformer-based panoptic segmentation. To obtain occlusion cues, we train an occlusion classifier on the COCO-OLAC dataset. The classifier derives the occlusion-level attention, which serves as spatial guidance, while the occlusion labels are encoded into a learnable embedding to produce channel-wise weights. Through joint modulation, PEMOLA elegantly introduces the occlusion priors into the position embedding, thereby improving the occlusion modelling. We further annotate the Cityscapes dataset with occlusion levels, termed Cityscapes Occlusion Labels for All Computer Vision Tasks (Cityscapes-OLAC), following the same labelling protocol as COCO-OLAC, to evaluate the cross-dataset generalisation ability of PEMOLA. Extensive experiments on COCO-OLAC and Cityscapes-OLAC demonstrate that PEMOLA consistently improves panoptic segmentation quality while introducing minimal computational overhead. These results highlight the importance of occlusion modelling, where incorporating occlusion-level attention helps deliver robust panoptic segmentation under occlusion. Code and dataset are available at https://github.com/wenbo-wei/PEMOLA.

**Analysis:**

### 1. 摘要翻译
复杂场景下的全景分割受遮挡问题挑战巨大，但现有方法常忽略遮挡建模。本文提出“位置嵌入调制与遮挡水平注意力机制”（PEMOLA），这是一个可无缝集成到基于Transformer的全景分割模型中的遮挡感知模块。为了获取遮挡线索，我们在COCO-OLAC数据集上训练了一个遮挡分类器。该分类器推导出“遮挡水平注意力”作为空间引导，同时将遮挡标签编码为可学习嵌入以产生通道级权重。通过联合调制，PEMOLA将遮挡先验优雅地引入到位置嵌入中，从而增强了遮挡建模能力。此外，我们标注了具有遮挡水平的Cityscapes数据集（Cityscapes-OLAC）以评估模型的跨数据集泛化能力。在COCO-OLAC和Cityscapes-OLAC上的大量实验表明，PEMOLA在几乎不增加计算开销的情况下，持续提升了全景分割质量，突显了遮挡建模对实现稳健全景分割的重要性。

### 2. 方法动机分析
*   **驱动力**：旨在解决Transformer类模型在处理遮挡时，因位置嵌入（Position Embedding）是“内容无关”的固定配置，无法动态适应复杂遮挡场景而导致边界模糊或实例缺失的问题。
*   **现有方法痛点**：现有主流方法多聚焦于可见特征的建模，而忽视了遮挡引起的空间语义变化。位置编码通常是静态的，缺乏对视觉上下文（尤其是遮挡程度与位置分布）的自适应调整能力。
*   **研究假设**：通过引入图像级别的遮挡分类信息，作为一种先验知识注入到网络的位置编码中，能够引导网络动态调整对遮挡区域的关注度，从而实现更稳健的实例分割。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **遮挡分类器**：利用预训练的Swin-L骨干网络（输入图像需掩盖非物体区域），输出遮挡水平（低、中、高）的预测。
    2.  **空间先验提取（Grad-CAM）**：通过Grad-CAM计算梯度，生成反映遮挡分布的空间权重图（Occlusion-Level Attention, $O_a$）。
    3.  **语义先验嵌入**：将遮挡类别标签映射为高维可学习向量（Occlusion Label Embedding, $O_l$）。
    4.  **调制操作**：利用公式 $E_{pos}^m = E_{pos} \odot (1 + O_a \otimes O_l)$ 将上述两种先验作用于原始位置嵌入，实现残差式动态调制。
*   **核心算法**：利用ReLU作用于 Grad-CAM 生成的加权特征图，提取仅包含正响应的遮挡关注区域。利用广播乘法（$O_a \otimes O_l$）实现空间分布与通道特征的深度融合。

### 4. 方法对比分析
*   **本质区别**：与传统静态位置编码不同，PEMOLA实现了“位置+遮挡语义”的双重动态调制。
*   **创新贡献**：提出了一种极轻量化的模块，在不改变原模型架构的前提下，通过残差调制方式将遮挡上下文注入位置编码，提升了模型对重叠实例的分辨力。
*   **适用场景**：自动驾驶、机器人视觉等遮挡频繁、物体密集的复杂场景。

### 5. 实验分析
*   **关键结果**：在COCO-OLAC数据集上，PEMOLA为Mask2Former和Mask DINO带来一致的性能提升（PQ指标提升约0.8~3.3不等）。在Cityscapes-OLAC上的迁移实验验证了其泛化能力。
*   **优势**：极低的计算开销（仅增加少量参数），易于集成，对遮挡水平较高的物体有显著改善。
*   **局限**：对遮挡分类器的依赖较高，如果预训练分类器在特定场景下失效，调制效果将大打折扣。

### 6. 实用指南
*   **开源地址**：https://github.com/wenbo-wei/PEMOLA
*   **实现细节**：建议仅在像素解码器（Pixel Decoder）处添加PEMOLA，因为该部分空间特征更丰富，与分类器的空间注意力图更对齐。训练时务必采用背景掩盖（Blackening）预处理以提升分类器专注力。
*   **迁移迁移**：该模块高度模块化，可直接复用到任何基于Transformer的分割框架（如DETR系列）。

### 7. 总结
*   **核心思想**：通过遮挡先验动态调制位置嵌入以增强空间感知能力。
*   **速记版pipeline**：
    1. 分类：判定遮挡等级。
    2. 定位：锁定遮挡空间区域。
    3. 编码：结合等级与位置语义。
    4. 调制：残差注入位置嵌入。

**Key Findings:**

- In this paper, we propose \textbf{P}osition \textbf{E}mbedding \textbf{M}odulation with \textbf{O}cclusion-\textbf{L}evel \textbf{A}ttention (PEMOLA), a novel occlusion-aware module that can be seamlessly integrated into transformer-based panoptic segmentation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18112v1)
- [arXiv](https://arxiv.org/abs/2607.18112v1)

---

<a id='2607.18060v1'></a>
## [RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning](https://arxiv.org/abs/2607.18060v1)

**Authors:** Jinbang Huang, Yuanzhao Hu, Zhiyuan Li, Ran Qi, Yixin Xiao, Zhanguang Zhang, Mark Coates, Tongtong Cao, Yingxue Zhang

**Published:** 2026-07-20

**Categories:** cs.RO

**Abstract:**

Long-horizon robotic tasks require diverse capabilities that no single policy can reliably provide. Heterogeneous policies offer complementary strengths, but orchestrating them requires reasoning over uncertain capability boundaries and cross-policy distribution mismatch, which are largely overlooked by existing planning methods built on homogeneous, predefined skills with fixed applicability. We propose RoboHarness, a unified framework that encapsulates independently developed robot control systems as reusable agentic skills. Although instantiated in this work with VLAs, RL policies, and task-and-motion planning (TAMP) systems, RoboHarness is designed as a general framework compatible with a broader range of robot policies, such as navigation policies, model predictive controllers, and world-action models. RoboHarness uses multi-modal execution memory and online evidence to characterize policy capability boundaries for capability-aware decomposition and routing. To stabilize policy handoffs, its Memory Bridge retrieves execution trajectories associated with the next policy, estimates its in-distribution state region, and guides the robot toward that region without joint policy retraining. Extensive experiments on three public benchmarks, 500 customized tasks, and 135 real-robot experiments demonstrate effective capability-aware routing and stable policy orchestration, yielding substantial improvements in zero-shot long-horizon planning and out-of-distribution robustness.

**Analysis:**

### 1. 摘要翻译
长跨度机器人任务需要多种能力，但没有任何单一策略能可靠地提供这些能力。异构策略具有互补优势，但对其进行编排需要处理不确定的能力边界及策略间分布不匹配问题，而现有基于具有固定适用性的同构预定义技能的规划方法对此往往忽视。我们提出了RoboHarness，一个统一框架，将独立开发的机器人控制系统封装为可重用的代理技能。尽管本工作在VLA、RL策略和任务与运动规划（TAMP）系统上进行了实例化，但RoboHarness是一个通用的框架，兼容更广泛的策略。RoboHarness利用多模态执行记忆和在线证据来刻画策略能力边界，以实现感知能力的任务分解和路由。为了稳定策略切换，其“记忆桥（Memory Bridge）”模块检索与下一策略相关的执行轨迹，估计其分布内状态区域，并在无需联合重新训练的情况下引导机器人进入该区域。在三个公共基准、500个定制任务和135次真实机器人实验上的广泛实验表明，该框架能实现有效的能力感知路由和稳定的策略编排，在零样本长跨度规划和分布外鲁棒性方面取得了显著提升。

### 2. 方法动机分析
*   **驱动力**：解决长跨度机器人任务中“没有单一策略能覆盖所有需求”的瓶颈，实现异构策略的协同。
*   **现有方法痛点**：现有工作多假设策略是同构的，忽略了异构策略在架构、输入输出和能力边界上的巨大差异。且策略切换时的分布不匹配（distribution mismatch）常导致级联失败。
*   **研究假设**：通过引入能力感知（Capability-aware）的路由机制和基于记忆的策略桥接（Memory-based bridging），可以将不同策略的互补优势组合，实现无联合训练的零样本长跨度规划。

### 3. 方法设计详解
RoboHarness由高层编码代理（Coding Agent）驱动，调用三个辅助模块：
*   **理解技能 (Understanding Skills)**：负责提取决策相关信息（如不确定性、语义上下文、状态策略兼容性）。核心是判断当前状态是否符合特定策略的“能力边界”。
*   **记忆技能 (Memory Skills)**：维护一个链接节点轨迹库（linked-node trajectories）。通过分层文本-视觉检索，获取最相关的历史执行案例，为后续规划提供经验证据。
*   **记忆桥 (Memory Bridge)**：关键组件。
    1.  **分布构建**：检索锚点（Anchor nodes），通过线性回归预测状态的局部进展得分。
    2.  **空间评分**：构建分布置信区（$R_{conf, t}$），确保机器人状态在下一策略的有效区域内。
    3.  **轨迹生成**：在满足 motion 约束的前提下，将当前状态引导至下一策略的“高置信度”输入区域。
*   **进化技能 (Evolution Skills)**：基于在线反馈进行策略适配、参数微调和元数据更新，形成闭环优化。

### 4. 方法对比分析
*   **本质区别**：不试图训练一个全能模型，而是通过“编排”现有的“专用模型”。引入了显式的“中间桥接（Bridging）”逻辑来解决跨策略传输的分布转移问题。
*   **创新贡献**：提出了将异构策略封装为代理技能的框架；利用记忆驱动的策略切换来规避分布不匹配。
*   **适用场景**：复杂、多步骤、需要结合语义推理（VLA）与精确物理控制（TAMP/RL）的长跨度机器人任务。

### 5. 实验分析
*   **关键结果**：在LIBERO-LoHo长跨度任务上表现优异（95.2% 成功率，远超单一策略）。
*   **主要优势**：极强的零样本泛化能力；对分布外扰动（如遮挡、噪声）表现出极强的鲁棒性。
*   **主要局限**：如果任务所需能力完全超出所有现有策略的覆盖范围，则无法解决；且依赖于历史记忆积累，冷启动阶段性能可能受限。

### 6. 实用指南
*   **开源情况**：基于已有的开源基准（LIBERO等）和预训练模型（$\pi_{0.5}$, OpenVLA-OFT, TAMP）。
*   **实现细节**：Memory Bridge 核心在于将“状态空间”转换为“进展分（Progress score）”空间，需谨慎选择回归模型（文中用SVM进行Pairwise Ranking）。
*   **迁移建议**：对于新任务，只需将新策略封装为“Policy Card”（包含输入输出接口、能力元数据），即可接入框架。

### 7. 总结
*   **核心思想**：通过记忆驱动的编排与中间状态桥接，实现异构策略的协同。
*   **速记版pipeline**：
    1. 评估当前任务与环境状态；
    2. 检索历史轨迹匹配最优策略；
    3. 利用记忆桥引导状态切换；
    4. 执行策略并收集反馈自我进化。

**Key Findings:**

- We propose RoboHarness, a unified framework that encapsulates independently developed robot control systems as reusable agentic skills.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18060v1)
- [arXiv](https://arxiv.org/abs/2607.18060v1)

---

<a id='2607.18016v1'></a>
## [Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation](https://arxiv.org/abs/2607.18016v1)

**Authors:** Peng Ren, Haoyang Ge, Jiang Zhao, Cong Huang, Yukun Shi, Pei Chi, Kai Chen

**Published:** 2026-07-20

**Categories:** cs.RO

**Abstract:**

Vision-language-action policies are a promising foundation for general robot control, but long-horizon humanoid loco-manipulation requires the robot to treat task objects as persistent physical entities across movement, contact, occlusion, and recovery. We study this problem as object-state divergence: the object state used to condition a whole-body action can differ from the state used to decide whether the action achieved the intended physical relation. We propose \emph{Persistent Object Tokenization} (POT), which maintains role-indexed 3D object records from RGB-D observations and converts them into object tokens for a whole-body action expert. Instantiated as \emph{POT-VLA}, the same object records condition action generation and support geometric predicate checks, yielding a closed-loop execution system in which object state is both actionable and verifiable. On a Unitree G1, POT-VLA improves a matched direct GR00T-N1.7 baseline from 39/80 to 71/80 successes over eight real-world task families. In an external Being-0-aligned reference, POT-VLA achieves 44/50 successes on aligned service tasks, compared with the 37/50 success reported by the Being-0 paper. The largest gains occur on tasks requiring maintained 3D relations, suggesting that persistent object-centered state is a useful abstraction for verifiable humanoid VLA execution.

**Analysis:**

这份报告针对论文《Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation》进行了深入分析。

---

### 1. 摘要翻译
视觉-语言-动作（VLA）策略为通用机器人控制提供了前景，但长程人形机器人全身操纵任务要求机器人在移动、接触、遮挡和故障恢复过程中将任务对象视为持续存在的物理实体。本文研究了“对象状态分歧”（object-state divergence）问题，即用于规划动作的对象状态与用于验证任务是否完成的状态存在不一致。为此，我们提出了“持久对象标记化”（Persistent Object Tokenization, POT）方法，该方法通过RGB-D观测维护角色索引化的3D对象记录，并将其转化为对象Token供全身动作专家使用。该系统（POT-VLA）实现了动作执行与物理验证的闭环。在Unitree G1机器人上的实验表明，POT-VLA将任务成功率从基线的39/80大幅提升至71/80，特别是在需要保持复杂3D空间关系的任务上表现突出。

### 2. 方法动机分析
*   **驱动力**：解决长程操纵任务中机器人因“缺乏对物体物理状态的持续感知与验证”而导致的执行失败。
*   **痛点**：当前VLA策略通常将物体信息隐式编码在视觉特征中，导致机器人无法在动作执行后准确判断物体状态是否如预期改变，出现“动作成功但目标未达成”的物理状态分歧。
*   **研究假设**：通过显式、持久的“对象状态”表征来闭合感知-动作-验证回路，可以显著提升人形机器人的任务成功率。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **持久化记忆构建 ($M_{\tau_i}^t$)**：根据任务指令，利用RGB-D数据和SAM3分割掩码，实时构建并维护一组角色索引（如Target, Destination, Support）的3D对象记录（含位置、外形、置信度等）。
    2.  **标记化（Tokenization）**：将上述记录映射为固定槽位的特征张量 $x_t^{obj}$，并投影嵌入到动作头的隐藏空间。
    3.  **条件动作预测**：动作专家（基于GR00T-N1.7）不仅输入视觉/语言特征，还通过自注意力机制融合这些“持久对象Token”，生成下一步动作。
    4.  **闭环验证与恢复**：动作执行后，利用新观测刷新记忆，通过几何谓词（如：物体是否被抓取、是否在容器内）进行检查，若未达成预期则触发重新观察或重试。
*   **模型结构**：核心在于在传统的VLA动作头中插入一个“对象Token注入”模块，该模块通过LayerNorm-Linear-GELU-Linear映射将物理空间状态转换为与动作头同维度的语义向量。

### 4. 方法对比分析
*   **本质区别**：从“端到端隐式映射”转变为“显式物理状态闭环”。主流方法通过深度特征预测动作，而POT-VLA强行要求动作由“经物理验证的显式对象坐标”条件化。
*   **创新贡献**：提出了一种通用的、非物理引擎依赖的闭环验证机制，利用角色索引槽位解决长程任务中的对象身份漂移问题。

### 5. 实验分析（精简版）
*   **核心结论**：在8个复杂的真实世界任务组中，成功率翻倍（39/80 vs 71/80）。
*   **优势**：在存在遮挡、移动、物体位置变动的情况下，具有极强的鲁棒性；支持结构化故障恢复（如识别出“未抓取”后触发重新抓取，而非盲目执行后续动作）。
*   **局限**：高度依赖RGB-D感知质量和相机标定精度；未能完全解决极其复杂的语义歧义（部分需要VLM辅助）。

### 6. 实用指南
*   **迁移方案**：该方法逻辑通用，可直接嵌入现有的Transformer基动作策略。
*   **实现要点**：
    *   **记忆维护**：需确保对象在不同时间步的ID一致性。
    *   **几何谓词设计**：需要定义清晰的物理评价标准（如 $p = \langle \kappa, \alpha, op, \nu, n \rangle$），将离散的几何度量转化为任务完成的判定依据。
    *   **训练策略**：必须在Fine-tuning阶段引入带有Token侧边信息的数据对，让动作头学习如何解析这些显式物理标记。

### 7. 总结
*   **核心思想**：通过维护角色索引的持久化3D对象记忆，实现物理可验证的闭环操纵。
*   **速记版pipeline**：
    1. 实时更新物体坐标记忆。
    2. 将物体空间位置转为智能Token。
    3. 输入动作模型指导下一步操作。
    4. 动作执行后核对物理状态，未达标则纠错重试。

**Key Findings:**

- We propose \emph{Persistent Object Tokenization} (POT), which maintains role-indexed 3D object records from RGB-D observations and converts them into object tokens for a whole-body action expert.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18016v1)
- [arXiv](https://arxiv.org/abs/2607.18016v1)

---

<a id='2607.17977v1'></a>
## [RynnBrain 1.1: Towards More Capable and Generalizable Embodied Foundation Model](https://arxiv.org/abs/2607.17977v1)

**Authors:** Kehan Li, Bohan Hou, Minghao Zhu, Tianyi Zhang, Zesen Cheng, Zhikai Wang, Sicong Leng, Xin Li, Xiao Lin, Biying Yao, Minghua Zeng, Jiangpin Liu, Ronghao Dang, Jiayan Guo, Siteng Huang, Haoyu Zhao, Heng Ping, Yaxi Zhao, Kexiang Wang, Tong Lu, Shengke Xue, Jiahao Tang, Yulei Wang, Zejing Wang, Jianwei Gao, Shijian Lu, Chengju Liu, Jianfei Yang, Mingxiu Chen, Deli Zhao

**Published:** 2026-07-20

**Categories:** cs.RO

**Abstract:**

We present RynnBrain 1.1, a family of embodied foundation models spanning 2B, 9B, and 122B-A10B scales. Trained with a unified spatio-temporal and physically grounded framework, RynnBrain 1.1 supports embodied perception, spatial reasoning, localization, and planning. Compared with RynnBrain 1.0, it further introduces contact-point prediction across the model family and native 3D grounding for the 2B and 9B models, yielding representations and outputs that are more directly aligned with robot manipulation. We also develop RynnBrain-VLA with a unified cross-embodiment action space and embodiment-specific masking, and deploy it on Unitree G1, Astribot-S1, and Tianji-Wuji. RynnBrain 1.1 achieves strong results on embodied cognition, localization, and 3D grounding, with the 122B-A10B model outperforming all evaluated proprietary and open-source models on VSI-Bench, MMSI, and RefSpatial-Bench. Real-robot experiments show that RynnBrain-initialized policies outperform Qwen-based and representative generalist VLAs, while joint multi-task and multi-embodiment training improves process scores and success rates over per-task training.

**Analysis:**

## 1. 摘要翻译
我们推出了 RynnBrain 1.1，这是一个涵盖 2B、9B 和 122B-A10B 参数规模的具身智能基础模型系列。通过统一的时空与物理接地框架进行训练，RynnBrain 1.1 支持具身感知、空间推理、定位与规划。与 RynnBrain 1.0 相比，该版本在模型全系列引入了接触点预测，并为 2B 和 9B 模型提供了原生 3D 接地能力，从而产生与机器人操作更直接对齐的表征与输出。我们还开发了 RynnBrain-VLA，采用了统一的跨具身动作空间与特定具身掩码，并将其部署在 Unitree G1、Astribot-S1 和 Tianji-Wuji 机器人上。RynnBrain 1.1 在具身认知、定位和 3D 接地方面取得了强劲成果，其中 122B-A10B 模型在 VSI-Bench、MMSI 和 RefSpatial-Bench 上超越了所有已知的专有与开源模型。真实机器人实验表明，RynnBrain 初始化的策略优于基于 Qwen 的策略及代表性通用 VLA，且联合多任务与多具身训练提升了过程评分与成功率。

## 2. 方法动机分析
- **驱动力**：旨在缩小基础模型与真实机器人操作之间的“具身鸿沟”，使视觉理解不仅停留在语义层面，更能转化为精确的物理世界交互。
- **现有方法痛点**：前作（RynnBrain 1.0）虽然实现了多模态理解，但其输出（如矩形框）与机器人实际的抓取操作（接触点、旋转角度）存在不对齐；且缺乏明确的 3D 空间理解能力。
- **研究假设**：通过显式的 3D 接地训练和接触点预测任务，可以增强模型对物体几何与空间姿态的感知；利用统一动作空间设计，能实现不同形态机器人（跨具身）之间的知识迁移与协同训练。

## 3. 方法设计详解
- **核心组件**：
    - **接触点预测 (Contact Point Prediction)**：放弃传统的矩形框表示，采用 $a = (p, \theta)$ 紧凑表示，其中 $p$ 为抓取中心坐标，$\theta$ 为夹爪平面旋转角。这剔除了与机器人无关的冗余几何信息，直接适配动作需求。
    - **原生 3D 接地 (Native 3D Grounding)**：输入自然语言与相机内参，模型直接回归相机坐标系下的 3D 边界框（中心点、尺寸、旋转），通过离散化转化为整数标记，纳入自回归框架。
    - **跨具身统一动作空间 (Unified Cross-Embodiment Action Space)**：定义了涵盖全机身的 81 维动作空间，将手臂、夹爪、末端执行器、头部等解耦为语义组。通过“特定具身掩码”技术，仅激活机器人物理上具备的维度。
- **流程总结**：
    1. **统一编码**：将图像、视频、语言和历史状态编码为统一序列。
    2. **自回归生成**：利用解码器统一预测文本回答、3D 空间坐标及抓取参数。
    3. **动作预测 (RynnBrain-VLA)**：采用流匹配 (Flow Matching) 框架，输入噪声动作序列，通过扩散过程进行去噪预测。
    4. **实时处理**：使用“实时分块 (Real-Time Chunking)”技术，在上一动作执行期间预测下一动作块，并根据剩余执行时间动态调整权重，确保连续性。

## 4. 方法对比分析
- **本质区别**：从传统的“视觉理解+策略生成”转变为“视觉-3D空间-动作”一体化的多任务联合学习。
- **创新贡献**：引入了显式的接触点预测作为抓取接口，通过掩码机制彻底解决了异构机器人动作空间不兼容的难题。

## 5. 实验分析（精简版）
- **关键结论**：在 122B 大参数规模下，模型在空间推理任务上超越了主流开源与闭源方案；多具身联合训练的策略比单任务训练效果更好。
- **优势**：极强的跨任务与跨具身迁移能力，在 Astribot 和 Unitree G1 等真实平台上表现出高成功率。
- **局限**：对长序列动作规划的计算开销较大；原生 3D 接地目前主要在 2B/9B 模型中充分验证。

## 6. 实用指南
- **开源情况**：已发布模型权重及相关代码，可参考其 GitHub 和 HuggingFace 仓库。
- **实现建议**：在进行跨具身迁移时，核心在于如何构建“语义对齐”的动作空间，建议对齐不同机器人的同类关节（如手臂位置），通过掩码进行遮盖。
- **训练细节**：学习率、批次大小和预热比例在文中已有详细表格，应针对不同规模模型调整其对应的训练配方。

## 7. 总结
- **核心思想**：通过显式 3D 接地与统一动作空间，构建具身智能的一体化认知-行动接口。
- **速记版 pipeline**：
    1. 多模态输入编码；
    2. 生成 3D 边界框与抓取接触点；
    3. 通过掩码激活特定机器人动作维度；
    4. 采用扩散式流匹配预测动作轨迹；
    5. 使用实时分块确保动作执行平滑。

**Key Findings:**

- We present RynnBrain 1.1, a family of embodied foundation models spanning 2B, 9B, and 122B-A10B scales.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.17977v1)
- [arXiv](https://arxiv.org/abs/2607.17977v1)

---

