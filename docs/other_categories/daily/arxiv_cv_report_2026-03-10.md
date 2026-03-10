time: 20260310

# Arxiv Computer Vision Papers - 2026-03-10

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要 (2026-03-09)**

**1. 核心主题与趋势概览**

今日的10篇论文清晰地反映了计算机视觉领域的三大融合与演进趋势：

*   **多模态与具身智能的深度融合**：超过一半的论文涉及视觉语言模型（VLM）、3D感知与机器人控制。研究重点已从单纯的感知转向**以任务为导向的、与物理世界交互的智能系统**（如 Exp-Force, Interactive World Simulator, Embedding Classical Balance Control）。
*   **生成模型的高效化与长序列化**：生成式AI继续向更高效、更高保真度和更长序列生成发展。**“HiAR”** 针对长视频生成提出分层去噪方法，而 **“Scale Space Diffusion”** 可能从尺度空间理论角度优化扩散模型，代表了生成技术底层创新的尝试。
*   **3D视觉的技术栈巩固与数据瓶颈突破**：3D Gaussian Splatting 作为新兴的3D表示方法，其高效实现（**ImprovedGS+**）和与基础模型结合解决长尾问题（**FOMO-3D**）成为热点。同时，研究致力于从多传感器（如毫米波雷达 **mmGAT**）或点云本身（**PCFEx**）提取更鲁棒的特征，以缓解3D数据稀缺的挑战。

**2. 重点与创新性论文亮点**

*   **最具系统创新潜力的论文**：**《Interactive World Simulator for Robot Policy Training and Evaluation》**。该工作直指机器人训练中的数据效率与安全性核心瓶颈，构建交互式仿真器，可能成为加速具身AI研究的平台级工具，价值显著。
*   **最具技术突破性的论文**：**《HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising》**。长视频生成是当前视频生成模型的圣杯，该方法通过“分层去噪”实现高效自回归生成，若验证有效，将是重要的技术进展。
*   **最具实用优化价值的论文**：**《ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting》**。3DGS的落地严重依赖性能，这篇论文通过底层工程优化直接提升该生态的可用性和研究迭代速度，对社区有即时贡献。

**3. 新兴研究方向识别**

*   **VLM的细粒度视觉控制**：**《FVG-PT》** 和 **《Exp-Force》** 展示了VLM不再局限于问答或描述，而是向**基于视觉感知的精细动作决策（如抓取力控制）和自适应提示学习**演进，标志着VLM在机器人学中的应用进入新阶段。
*   **经典控制理论与现代RL的融合**：**《Embedding Classical Balance Control Principles in Reinforcement Learning...》** 代表了一种理性回归：将已知的、可靠的经典控制理论先验嵌入数据驱动的RL中，以提高学习效率、稳定性和可解释性，这可能成为机器人控制的主流范式。
*   **非视觉传感器与视觉的互补**：**《mmGAT》** 利用毫米波雷达点云进行姿态估计，在黑暗、雾霾等视觉受限场景下具有不可替代性。**多模态感知从“视觉+语言”扩展到“视觉+物理信号（雷达、力、触觉等）”** 是一个明确趋势。

**4. 全文精读建议**

根据研究者的不同方向，优先推荐如下：

*   **所有研究者必读（技术风向标）**：
    *   **HiAR**：了解长视频生成的最前沿技术路径。
    *   **Interactive World Simulator**：关注具身智能研究基础设施的进展。
*   **机器人视觉与具身AI方向**：
    *   **Exp-Force**（VLM用于精细物理交互）。
    *   **Embedding Classical Balance Control Principles...**（混合控制方法典范）。
*   **3D视觉与生成模型方向**：
    *   **ImprovedGS+**（工程实践必备）。
    *   **FOMO-3D**（如何用2D基础模型解决3D长尾问题）。
*   **高效微调与多模态学习方向**：
    *   **FVG-PT**（前景引导的提示调优，具有创新性）。

**总结**：本日论文集表明，计算机视觉的研究前沿正强力推动 **“感知-生成-决策-控制”闭环** 的构建。研究重心从模型性能的绝对提升，转向**效率、可靠性、与物理世界的兼容性以及多模态信号的深度融合**。建议结合自身工作，重点关注生成模型的高效长序列生成、VLM的具身化应用以及3D视觉的工程与数据解决方案。

---

## Table of Contents

1. [Scale Space Diffusion](#2603.08709v1)
2. [FVG-PT: Adaptive Foreground View-Guided Prompt Tuning for Vision-Language Models](#2603.08708v1)
3. [HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising](#2603.08703v1)
4. [Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models](#2603.08668v1)
5. [ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting](#2603.08661v1)
6. [Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery](#2603.08619v1)
7. [FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection](#2603.08611v1)
8. [mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud](#2603.08551v1)
9. [Interactive World Simulator for Robot Policy Training and Evaluation](#2603.08546v1)
10. [PCFEx: Point Cloud Feature Extraction for Graph Neural Networks](#2603.08540v1)

---

## Papers

<a id='2603.08709v1'></a>
## [Scale Space Diffusion](https://arxiv.org/abs/2603.08709v1)

**Authors:** Soumik Mukhopadhyay, Prateksha Udhayanan, Abhinav Shrivastava

**Published:** 2026-03-09

**Categories:** cs.CV, cs.AI

**Abstract:**

Diffusion models degrade images through noise, and reversing this process reveals an information hierarchy across timesteps. Scale-space theory exhibits a similar hierarchy via low-pass filtering. We formalize this connection and show that highly noisy diffusion states contain no more information than small, downsampled images - raising the question of why they must be processed at full resolution. To address this, we fuse scale spaces into the diffusion process by formulating a family of diffusion models with generalized linear degradations and practical implementations. Using downsampling as the degradation yields our proposed Scale Space Diffusion. To support Scale Space Diffusion, we introduce Flexi-UNet, a UNet variant that performs resolution-preserving and resolution-increasing denoising using only the necessary parts of the network. We evaluate our framework on CelebA and ImageNet and analyze its scaling behavior across resolutions and network depths. Our project website ( https://prateksha.github.io/projects/scale-space-diffusion/ ) is available publicly.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《Scale Space Diffusion》的分析如下：

### 1. 核心贡献总结
该论文揭示了扩散模型（Diffusion Models）的去噪过程与经典计算机视觉中的“尺度空间理论”（Scale-space theory）之间的深刻数学联系。作者提出了一种名为“尺度空间扩散”（Scale Space Diffusion）的新框架，通过在扩散过程中引入降采样（Downsampling）作为退化算子，实现了在低分辨率状态下进行高效去噪，从而打破了传统扩散模型必须在全分辨率下处理所有时间步的计算瓶颈。

### 2. 关键创新与方法论
*   **理论映射**：论文将扩散模型中高噪声状态下的信息冗余与尺度空间中的低通滤波效应联系起来，证明了高噪声图像本质上等同于低分辨率图像，无需全分辨率处理。
*   **广义退化框架**：将扩散过程推广至包含线性退化（如降采样）的范式，使得模型可以在不同尺度上进行信息恢复。
*   **Flexi-UNet 架构**：这是该研究的工程核心。Flexi-UNet 能够根据当前尺度动态调整计算路径，支持分辨率保持（Resolution-preserving）和分辨率提升（Resolution-increasing）的去噪操作，避免了在低分辨率阶段浪费高分辨率特征图的计算资源。

### 3. 对领域的潜在影响
*   **计算效率的质变**：传统扩散模型（如 Stable Diffusion）在全分辨率下进行大量迭代，计算成本极高。该方法通过在扩散早期阶段利用低分辨率处理，有望显著降低推理和训练的计算开销（FLOPs）。
*   **理论视角的重构**：该研究为理解扩散模型的“信息演化”提供了一个物理意义明确的视角（即尺度空间演化），这可能引导未来研究者从信号处理的角度优化生成模型，而非单纯依赖堆叠算力。

### 4. 相关领域与应用价值
*   **高分辨率图像生成**：直接受益于该技术，能够以更低的显存占用生成超高清图像。
*   **实时生成任务**：对于需要低延迟的交互式生成应用（如实时视频生成、增强现实中的实时渲染），该方法提供了极具吸引力的加速方案。
*   **多尺度特征学习**：该方法在处理多尺度任务（如图像超分辨率、语义分割）时，能够更自然地融合不同层级的特征信息。

### 5. 可推断的局限性
*   **信息丢失风险**：虽然理论上高噪声状态下高频信息较少，但在极低分辨率下进行扩散可能导致模型在恢复细节时出现“伪影”或纹理丢失，特别是在需要极高保真度的场景中。
*   **架构复杂性**：Flexi-UNet 的动态计算路径可能在硬件部署（如 TensorRT 或特定 GPU 加速）时带来额外的调度开销，不如固定结构的 UNet 容易进行算子融合优化。
*   **泛化能力**：论文主要在 CelebA 和 ImageNet 上验证，对于复杂场景（如长文本提示词、复杂构图）的生成质量是否会因尺度切换而产生不连续性，仍需进一步观察。

**专家点评：**
这篇论文的趣味性在于它**“返璞归真”**——它没有盲目追求更大的模型参数或更复杂的注意力机制，而是通过重审扩散模型的数学本质（尺度空间），巧妙地解决了计算效率问题。这种将经典信号处理理论与现代生成式 AI 结合的思路，是当前计算机视觉领域非常值得关注的研究范式。

**Key Findings:**

- To support Scale Space Diffusion, we introduce Flexi-UNet, a UNet variant that performs resolution-preserving and resolution-increasing denoising using only the necessary parts of the network.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08709v1)
- [arXiv](https://arxiv.org/abs/2603.08709v1)

---

<a id='2603.08708v1'></a>
## [FVG-PT: Adaptive Foreground View-Guided Prompt Tuning for Vision-Language Models](https://arxiv.org/abs/2603.08708v1)

**Authors:** Haoyang Li, Liang Wang, Siyu Zhou, Jiacheng Sun, Jing Jiang, Chao Wang, Guodong Long, Yan Peng

**Published:** 2026-03-09

**Categories:** cs.CV

**Abstract:**

CLIP-based prompt tuning enables pretrained Vision-Language Models (VLMs) to efficiently adapt to downstream tasks. Although existing studies have made significant progress, they pay limited attention to changes in the internal attention representations of VLMs during the tuning process. In this paper, we attribute the failure modes of prompt tuning predictions to shifts in foreground attention of the visual encoder, and propose Foreground View-Guided Prompt Tuning (FVG-PT), an adaptive plug-and-play foreground attention guidance module, to alleviate the shifts. Concretely, FVG-PT introduces a learnable Foreground Reliability Gate to automatically enhance the foreground view quality, applies a Foreground Distillation Compensation module to guide visual attention toward the foreground, and further introduces a Prior Calibration module to mitigate generalization degradation caused by excessive focus on the foreground. Experiments on multiple backbone models and datasets show the effectiveness and compatibility of FVG-PT. Codes are available at: https://github.com/JREion/FVG-PT

**Analysis:**

### 1. 摘要翻译
基于CLIP的提示调优（Prompt Tuning）使预训练视觉-语言模型（VLMs）能够高效适应下游任务。尽管现有研究已取得显著进展，但它们对调优过程中VLM内部注意力表示的变化关注不足。本文将提示调优预测的失效模式归因于视觉编码器前景注意力的偏移，并提出了**前景视图引导提示调优（FVG-PT）**，这是一种自适应的即插即用前景注意力引导模块，旨在缓解这种偏移。具体而言，FVG-PT引入了可学习的“前景可靠性门（Foreground Reliability Gate）”来自动增强前景视图质量，应用“前景蒸馏补偿（Foreground Distillation Compensation）”模块引导视觉注意力聚焦于前景，并进一步引入“先验校准（Prior Calibration）”模块，以减轻因过度关注前景而导致的泛化能力下降。在多个骨干模型和数据集上的实验证明了FVG-PT的有效性和兼容性。

### 2. 方法动机分析
*   **驱动力**：作者发现提示调优过程中，模型注意力往往会从目标对象（前景）偏移到无关背景，导致语义对齐失效，这是导致预测错误的主要原因。
*   **现有方法痛点**：现有前景引导方法缺乏自适应机制，无法灵活平衡前景关注度与新类泛化能力（即Base-New Trade-off, BNT问题）。
*   **研究假设**：通过显式的前景监督引导注意力，并结合自适应的可靠性评估与解耦的先验校准，可以显著提升模型在基类上的准确率，同时保持对新类的泛化能力。

### 3. 方法设计详解
*   **流程总结**：
    1.  **前景获取**：利用预训练的SEEM分割模型提取输入图像的前景掩码，生成前景视图。
    2.  **前景可靠性门（FRG）**：通过MLP计算多个可靠性指标（分布熵、相似度、几何比例），输出一个标量权重 $r$，动态评估前景视图的可靠性。
    3.  **前景蒸馏补偿（FDC）**：在视觉和文本分支插入轻量级适配器（Adapter），利用 $r$ 引导模型将注意力从全图特征对齐转向前景特征对齐，通过蒸馏损失实现。
    4.  **先验校准（PC）**：在推理阶段将基类分支与新类分支解耦，利用“骨干可靠性门（BRG）”动态平衡微调后的模型与原始CLIP先验，解决BNT问题。
*   **算法解释**：FRG通过比较全图与前景视图的交叉熵损失，构建二值监督信号 $r^*$，训练MLP学习前景的“信任度”，从而实现自适应的注意力引导。

### 4. 方法对比分析
*   **本质区别**：不同于以往仅优化提示向量或简单引入视觉监督，FVG-PT引入了**自适应的可靠性评估**和**解耦的先验校准**，实现了对注意力偏移的动态修正。
*   **创新贡献**：提出了即插即用的前景引导框架，通过FRG和FDC模块实现了对注意力质量的精细控制，并通过PC模块有效缓解了提示调优中常见的BNT问题。
*   **适用场景**：适用于所有基于CLIP的提示调优架构，尤其在小样本学习和跨数据集迁移任务中表现优异。

### 5. 实验分析
*   **验证方法**：在11个数据集上，将FVG-PT接入4种主流提示调优骨干模型（CoOp, KgCoOp, PromptSRC, MMRL）进行基到新（Base-to-New）泛化实验。
*   **关键结果**：FVG-PT在所有骨干模型上均实现了性能提升，特别是在新类泛化能力上表现突出。
*   **优势与局限**：优势在于极高的参数效率（仅0.13M可训练参数）和良好的兼容性；局限在于依赖外部分割模型，且在缺乏文本分支的纯视觉提示方法中不适用。

### 6. 实用指南
*   **开源情况**：代码已开源：https://github.com/JREion/FVG-PT。
*   **实现细节**：建议将FDC适配器隐藏层维度设为64，可靠性门MLP隐藏层设为32，温度系数 $\tau_d=2.0$。在数据极少（如EuroSAT）时，适当增加训练轮数（如20 epoch）可避免过拟合。
*   **迁移可能**：该方法模块化程度高，可直接作为插件接入任何基于Transformer的视觉-语言提示调优模型。

### 7. 总结
*   **核心思想**：通过自适应前景引导与先验解耦，修正注意力偏移并提升泛化性能。
*   **速记版pipeline**：
    1. 提取前景掩码。
    2. 评估前景可靠性。
    3. 蒸馏引导注意力聚焦。
    4. 解耦基类与新类分支。

**Key Findings:**

- Experiments on multiple backbone models and datasets show the effectiveness and compatibility of FVG-PT.
- Codes are available at: https://github.com/JREion/FVG-PT

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08708v1)
- [arXiv](https://arxiv.org/abs/2603.08708v1)

---

<a id='2603.08703v1'></a>
## [HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising](https://arxiv.org/abs/2603.08703v1)

**Authors:** Kai Zou, Dian Zheng, Hongbo Liu, Tiankai Hang, Bin Liu, Nenghai Yu

**Published:** 2026-03-09

**Categories:** cs.CV

**Abstract:**

Autoregressive (AR) diffusion offers a promising framework for generating videos of theoretically infinite length. However, a major challenge is maintaining temporal continuity while preventing the progressive quality degradation caused by error accumulation. To ensure continuity, existing methods typically condition on highly denoised contexts; yet, this practice propagates prediction errors with high certainty, thereby exacerbating degradation. In this paper, we argue that a highly clean context is unnecessary. Drawing inspiration from bidirectional diffusion models, which denoise frames at a shared noise level while maintaining coherence, we propose that conditioning on context at the same noise level as the current block provides sufficient signal for temporal consistency while effectively mitigating error propagation. Building on this insight, we propose HiAR, a hierarchical denoising framework that reverses the conventional generation order: instead of completing each block sequentially, it performs causal generation across all blocks at every denoising step, so that each block is always conditioned on context at the same noise level. This hierarchy naturally admits pipelined parallel inference, yielding a 1.8 wall-clock speedup in our 4-step setting. We further observe that self-rollout distillation under this paradigm amplifies a low-motion shortcut inherent to the mode-seeking reverse-KL objective. To counteract this, we introduce a forward-KL regulariser in bidirectional-attention mode, which preserves motion diversity for causal inference without interfering with the distillation loss. On VBench (20s generation), HiAR achieves the best overall score and the lowest temporal drift among all compared methods.

**Analysis:**

以下是对论文《HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising》的深度分析：

### 1. 摘要翻译
自回归（AR）扩散模型为生成理论上无限长的视频提供了有前景的框架。然而，主要挑战在于如何在防止因误差累积导致的质量退化同时保持时间连续性。现有方法通常依赖于高度去噪的上下文，但这会以高确定性传播预测误差，从而加剧退化。本文认为，高度干净的上下文并非必要。受双向扩散模型（在共享噪声水平下对帧进行去噪并保持一致性）的启发，我们提出：在与当前块相同的噪声水平下对上下文进行条件化，既能提供足够的信号以保持时间一致性，又能有效缓解误差传播。基于此，我们提出了HiAR，一种分层去噪框架，它颠倒了传统的生成顺序：不再顺序完成每个块，而是在每个去噪步骤中跨所有块执行因果生成，使每个块始终在相同噪声水平的上下文下进行条件化。这种层级结构自然支持流水线并行推理，在4步设置下实现了约1.8倍的墙上时钟加速。我们进一步观察到，该范式下的自滚动蒸馏会放大模式寻求（mode-seeking）反向KL目标中固有的低运动捷径。为抵消这一点，我们引入了一种双向注意力模式下的前向KL正则化器，在不干扰蒸馏损失的情况下为因果推理保留了运动多样性。在VBench（20秒生成）上，HiAR在所有对比方法中实现了最佳总分和最低时间漂移。

### 2. 方法动机分析
*   **驱动力**：解决长视频生成中因“误差累积”导致的分布漂移（如过饱和、语义漂移）与推理效率低下问题。
*   **现有方法痛点**：传统AR方法强制要求前序块必须先被“完全去噪”（即噪声水平 $t_c=0$）作为上下文。这导致模型在极高置信度下传播了前序块的预测误差，形成恶性循环。
*   **研究假设**：上下文的噪声水平 $t_c$ 存在“偏差-信息”权衡。只要上下文携带的信息量不低于当前块在当前去噪步所需的信息量，即可保持时间连贯性，且较高的噪声水平能有效抑制误差传播。

### 3. 方法设计详解
*   **流程总结**：
    1.  **分层去噪（Hierarchical Denoising）**：打破“块-块”顺序，改为“步-步”顺序。在每一个去噪步 $j$，对所有块 $B_1 \dots B_N$ 进行并行处理。
    2.  **匹配噪声上下文**：将上下文的噪声水平设定为 $t_c = t_{j+1}$（当前去噪步的输出噪声水平），这是满足因果约束下的最噪声水平，能最大程度衰减偏差。
    3.  **流水线并行**：利用块在不同去噪步间的依赖关系（反对角线独立性），通过异步通信实现跨层级流水线并行。
*   **模型结构**：基于Wan2.1-1.3B作为骨干，引入因果注意力掩码。
*   **算法解释**：通过公式 $t_c^* = t_{j+1}$ 确定最优噪声水平，确保上下文既包含足够信息，又不会过度放大前序块的偏差。

### 4. 方法对比分析
*   **本质区别**：从“串行块生成”转变为“全局步生成”，将去噪过程从时间轴上的顺序依赖解耦为空间轴上的层级并行。
*   **创新贡献**：提出了匹配噪声水平的上下文条件化策略，并引入前向KL正则化器以解决蒸馏过程中的“低运动捷径”问题。
*   **适用场景**：需要长视频、实时流式输出且对质量稳定性要求极高的交互式世界模型。

### 5. 实验分析
*   **关键结果**：在VBench上，HiAR在总分（0.821）和质量（0.846）上均优于基线，且漂移指标（0.257）显著低于Self-Forcing（0.355）。
*   **主要优势**：在保持高质量的同时，通过流水线并行实现了1.8倍的推理加速，且极大地缓解了长视频生成中的颜色和语义漂移。
*   **主要局限**：对训练阶段的蒸馏策略依赖较强，需要精细调节前向KL正则化的权重 $\lambda$ 和步数 $K$。

### 6. 实用指南
*   **开源情况**：已开源，代码见 [https://jacky-hate.github.io/HiAR/](https://jacky-hate.github.io/HiAR/)。
*   **实现细节**：训练时需使用Wan2.1-14B作为教师模型进行DMD蒸馏；前向KL正则化仅在双向注意力模式下计算，且仅作用于前 $K=1$ 步。
*   **迁移可能**：该分层去噪范式可直接迁移至任何基于扩散的自回归视频生成模型，只需调整注意力掩码和去噪调度。

### 7. 总结
*   **核心思想**：通过匹配噪声水平的上下文条件化与分层去噪，实现长视频生成的稳定与高效。
*   **速记版pipeline**：
    1. 设定分层去噪调度，使所有块在同一噪声水平下同步更新。
    2. 采用匹配噪声的上下文，抑制前序块误差传播。
    3. 利用反对角线依赖性，通过流水线并行加速推理。
    4. 引入前向KL正则化，防止模型坍缩至低运动状态。

**Key Findings:**

- Drawing inspiration from bidirectional diffusion models, which denoise frames at a shared noise level while maintaining coherence, we propose that conditioning on context at the same noise level as the current block provides sufficient signal for temporal consistency while effectively mitigating error propagation.
- Building on this insight, we propose HiAR, a hierarchical denoising framework that reverses the conventional generation order: instead of completing each block sequentially, it performs causal generation across all blocks at every denoising step, so that each block is always conditioned on context at the same noise level.
- To counteract this, we introduce a forward-KL regulariser in bidirectional-attention mode, which preserves motion diversity for causal inference without interfering with the distillation loss.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08703v1)
- [arXiv](https://arxiv.org/abs/2603.08703v1)

---

<a id='2603.08668v1'></a>
## [Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models](https://arxiv.org/abs/2603.08668v1)

**Authors:** Siqi Shang, Minchao Huang, Bill Fan, Lillian Chin

**Published:** 2026-03-09

**Categories:** cs.RO

**Abstract:**

Accurate pre-contact grasp force selection is critical for safe and reliable robotic manipulation. Adaptive controllers regulate force after contact but still require a reasonable initial estimate. Starting a grasp with too little force requires reactive adjustment, while starting a grasp with too high a force risks damaging fragile objects. This trade-off is particularly challenging for compliant grippers, whose contact mechanics are difficult to model analytically. We propose Exp-Force, an experience-conditioned framework that predicts the minimum feasible grasping force from a single RGB image. The method retrieves a small set of relevant prior grasping experiences and conditions a vision-language model on these examples for in-context inference, without analytic contact models or manually designed heuristics. On 129 object instances, ExpForce achieves a best-case MAE of 0.43 N, reducing error by 72% over zero-shot inference. In real-world tests on 30 unseen objects, it improves appropriate force selection rate from 63% to 87%. These results demonstrate that Exp-Force enables reliable and generalizable pre-grasp force selection by leveraging prior interaction experiences. http://expforcesubmission.github.io/Exp-Force-Website/

**Analysis:**

### 1. 摘要翻译
准确的预接触抓取力选择对于安全可靠的机器人操作至关重要。自适应控制器虽能在接触后调节力，但仍需合理的初始估计。初始力过小会导致抓取失败，过大则可能损坏易碎物体。针对顺应性（compliant）夹爪接触力学难以建模的挑战，本文提出了 **Exp-Force**，这是一个基于单张RGB图像预测最小可行抓取力的经验条件框架。该方法检索少量相关的先验抓取经验，并利用视觉语言模型（VLM）进行上下文推理，无需解析接触模型或手动设计的启发式规则。在129个物体实例上，Exp-Force实现了0.43 N的平均绝对误差（MAE），较零样本推理降低了72%的误差。在30个未见物体的真实世界测试中，其合适力选择率从63%提升至87%。结果表明，Exp-Force通过利用先验交互经验，实现了可靠且可泛化的预抓取力选择。

---

### 2. 方法动机分析
*   **驱动力**：解决机器人抓取中“力选择”的冷启动问题，即在接触前如何根据物体外观给出合理的初始力估计。
*   **现有方法痛点**：
    *   **解析法**：依赖精确的物理参数（如摩擦系数、物体质量），在复杂物体上失效。
    *   **数据驱动法**：通常学习全局映射，在有限数据集上容易过拟合，且难以泛化到未见物体。
    *   **零样本VLM**：倾向于“幻觉”出物理参数并套用错误的解析公式（如库仑摩擦模型），导致严重过估计。
*   **研究假设**：人类通过视觉和语义相似性联想过往经验来选择抓取力；VLM具备类似的常识推理能力，若能通过上下文（In-context）提供相关经验，即可实现准确的力预测。

---

### 3. 方法设计详解
Exp-Force的核心是将“力预测”转化为“基于经验的上下文推理”问题，流程如下：
1.  **物体描述生成 (Descriptor VLM)**：输入目标物体图像 $I_o$，结合任务信息 $C$ 和指令 $\tau_{desc}$，生成描述物体物理属性（如材质、刚度、形状）的文本 $T_o$。
2.  **经验检索 (Experience Retrieval)**：利用多模态嵌入模型 $\phi$ 将 $(I_o, T_o)$ 映射到向量空间 $z_o$。在经验池 $\mathcal{E}$ 中计算余弦相似度，检索出最相似的 $k$ 个历史抓取案例 $\mathcal{E}_k(o)$。
3.  **经验条件推理 (Predictor VLM)**：将任务信息 $C$、检索到的案例 $\mathcal{E}_k(o)$、指令 $\tau_{pred}$ 和目标图像 $I_o$ 拼接成提示词，输入预测器 VLM，直接输出标量力 $F_o$。

*   **模型结构**：采用模块化设计，Descriptor 和 Predictor 可复用现有的前沿 VLM（如 Gemini-3.1-Pro），Embedding 模型使用 Qwen3-VL-Embedding-8B。
*   **算法逻辑**：通过 In-context Learning，模型不再是“物理引擎”，而是“语义插值器”，通过对比相似案例的力数据，隐式地捕捉了夹爪与物体间的复杂接触力学。

---

### 4. 方法对比分析
*   **本质区别**：从“绝对回归”（学习输入到力的映射）转向“局部比较推理”（利用相似案例进行类比）。
*   **创新贡献**：首次将检索增强生成（RAG）应用于机器人抓取力选择；证明了VLM可以通过上下文学习隐式地建模硬件的顺应性接触力学。
*   **适用场景**：适用于使用顺应性夹爪、物体种类多样且难以建立精确解析模型的机器人操作任务。

---

### 5. 实验分析
*   **验证方法**：在129个物体上进行5折交叉验证，并在30个未见物体上进行真实机器人抓取实验。
*   **关键结果**：在真实世界中，将合适力选择率从63%提升至87%，MAE降低85%。
*   **主要优势**：样本效率极高（$k=6$ 即可达到性能平台期），且无需针对特定夹爪进行繁琐的物理建模。
*   **主要局限**：依赖单张图像，无法直接感知物体质量（如空瓶与满瓶），仍需结合触觉反馈进行闭环修正。

---

### 6. 实用指南
*   **开源情况**：已发布代码与数据集（见论文链接）。
*   **实现细节**：
    *   **数据预处理**：需构建包含物体名称、质量、描述、图像及真值力的经验池。
    *   **超参数**：$k=6$ 至 $10$ 是性能与计算开销的最佳平衡点。
*   **迁移可能**：该框架可直接迁移至其他需要“基于经验的参数预测”任务，如机器人抓取位置预测、操作速度规划等。

---

### 7. 总结
*   **核心思想**：利用VLM的上下文学习能力，通过检索相似经验实现精准的抓取力预测。
*   **速记版pipeline**：
    1. 视觉描述：生成物体物理属性文本。
    2. 相似检索：在经验库中找最像的案例。
    3. 上下文推理：将案例喂给VLM预测力。
    4. 闭环执行：根据预测力进行抓取。

**Key Findings:**

- We propose Exp-Force, an experience-conditioned framework that predicts the minimum feasible grasping force from a single RGB image.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08668v1)
- [arXiv](https://arxiv.org/abs/2603.08668v1)

---

<a id='2603.08661v1'></a>
## [ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting](https://arxiv.org/abs/2603.08661v1)

**Authors:** Jordi Muñoz Vicente

**Published:** 2026-03-09

**Categories:** cs.CV

**Abstract:**

Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.

**Analysis:**

### 1. 摘要翻译
本文提出了 ImprovedGS+，一种在 LichtFeld-Studio 框架下原生实现的、高性能、低层级的 3D 高斯泼溅（3DGS）重构策略。通过将高层 Python 逻辑转换为优化的 C++/CUDA 内核，我们显著降低了主机-设备同步开销及训练延迟。本实现引入了长轴分裂（LAS）CUDA 内核、基于拉普拉斯算子的边缘重要性采样内核（含非极大值抑制 NMS），以及自适应指数尺度调度器。在 Mip-NeRF360 数据集上的实验表明，ImprovedGS+ 建立了新的帕累托最优前沿：在 1M 高斯预算下，训练时间缩短 26.8%，高斯数量减少 13.3%，且视觉质量更优；全量模型在参数复杂度降低 38.4% 的情况下，PSNR 提升了 1.28 dB。

### 2. 方法动机分析
- **驱动力**：在保持 3DGS 重构质量的同时，极致追求计算效率与训练吞吐量。
- **痛点**：现有基于 Python 的实现（如原始 ImprovedGS）依赖高层 PyTorch 调用，导致频繁的内存拷贝和主机-设备同步，且分裂策略不够精细，容易产生“ densification drift”（密度漂移）和冗余高斯。
- **研究假设**：通过将核心 densification（致密化）逻辑下沉至 CUDA 原生内核，并引入更精细的几何约束（边缘检测）和动态调度策略，可以实现更高效、更精准的场景表示。

### 3. 方法设计详解
- **流程总结**：
  1. **结构化重要性映射**：利用类 Canny 边缘检测逻辑（灰度化、高斯模糊、Sobel 梯度计算、NMS），在 GPU 上生成结构化掩码，确保高斯仅在几何边界处分裂。
  2. **长轴分裂（LAS）内核**：将分裂逻辑封装在单一 CUDA 内核中，直接在原始参数空间（对数空间）进行操作，避免了多次内存传递。
  3. **两阶段训练策略**：
     - **阶段 I（高动量扩张）**：提高初始尺度学习率，快速占据场景空间。
     - **阶段 II（精度细化）**：引入指数衰减调度器，稳定几何细节，防止过拟合。
- **算法解释**：LAS 策略通过识别高斯的最长轴，仅沿该轴进行分裂，并利用旋转矩阵的列向量直接计算全局位移，避免了昂贵的 3×3 矩阵乘法，显著降低了计算开销。

### 4. 方法对比分析
- **本质区别**：从“高层框架调用”转向“底层 CUDA 内核封装”，实现了计算与存储的深度融合。
- **创新贡献**：LAS 内核的单次内存传递设计、基于 NMS 的边缘约束、以及针对训练过程的指数尺度调度。
- **适用场景**：对训练速度和显存占用敏感的实时渲染任务，以及需要高保真度的大规模复杂场景。

### 5. 实验分析
- **验证方法**：在 Mip-NeRF360 数据集上，与 MCMC 和 ADC 基线进行对比。
- **关键结果**：在 1M 高斯预算下，训练时间缩短约 17 分钟，且 PSNR 优于 MCMC 基线。
- **优势**：极高的训练吞吐量，更少的参数量实现更高的 PSNR。
- **局限**：在训练初期，高学习率可能导致几何表示不稳定，需依赖后续的细化阶段。

### 6. 实用指南
- **开源情况**：已开源（https://github.com/jordizv/ImprovedGS-Plus）。
- **实现细节**：需注意 `position_lr` 的初始值设定（0.000128）及指数衰减调度器的参数（$\gamma=0.1$）。
- **迁移可能**：LAS 内核和边缘检测逻辑可直接迁移至其他基于 3DGS 的变体中，作为通用的致密化加速模块。

### 7. 总结
- **核心思想**：通过底层 CUDA 内核重构致密化逻辑，实现几何精准分裂与高效训练。
- **速记版pipeline**：
  1. **边缘提取**：利用 GPU 原生算子识别几何边界。
  2. **长轴分裂**：在单一内核中完成分裂与坐标变换。
  3. **动态调度**：分阶段调整学习率与尺度以稳定结构。
  4. **参数优化**：利用对数空间操作确保数值一致性。

**Key Findings:**

- In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework.
- By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency.
- Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction.
- Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08661v1)
- [arXiv](https://arxiv.org/abs/2603.08661v1)

---

<a id='2603.08619v1'></a>
## [Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery](https://arxiv.org/abs/2603.08619v1)

**Authors:** Nehar Poddar, Stephen McCrory, Luigi Penco, Geoffrey Clark, Hakki Erhan Svil, Robert Griffin

**Published:** 2026-03-09

**Categories:** cs.RO

**Abstract:**

Humanoid robots remain vulnerable to falls and unrecoverable failure states, limiting their practical utility in unstructured environments. While reinforcement learning has demonstrated stand-up behaviors, existing approaches treat recovery as a pure task-reward problem without an explicit representation of the balance state. We present a unified RL policy that addresses this limitation by embedding classical balance metrics: capture point, center-of-mass state, and centroidal momentum, as privileged critic inputs and shaping rewards directly around these quantities during training, while the actor relies solely on proprioception for zero-shot hardware transfer. Without reference trajectories or scripted contacts, a single policy spans the full recovery spectrum: ankle and hip strategies for small disturbances, corrective stepping under large pushes, and compliant falling with multi-contact stand-up using the hands, elbows, and knees. Trained on the Unitree H1-2 in Isaac Lab, the policy achieves a 93.4% recovery rate across randomized initial poses and unscripted fall configurations. An ablation study shows that removing the balance-informed structure causes stand-up learning to fail entirely, confirming that these metrics provide a meaningful learning signal rather than incidental structure. Sim-to-sim transfer to MuJoCo and preliminary hardware experiments further demonstrate cross-environment generalization. These results show that embedding interpretable balance structure into the learning framework substantially reduces time spent in failure states and broadens the envelope of autonomous recovery.

**Analysis:**

### 1. 摘要翻译
人形机器人极易跌倒且难以恢复，限制了其在非结构化环境中的实用性。虽然强化学习（RL）已展现出站立行为，但现有方法将恢复视为纯粹的任务奖励问题，缺乏对平衡状态的显式表示。我们提出了一种统一的RL策略，通过将经典的平衡指标（捕获点、质心状态、质心动量）作为特权评论家（Privileged Critic）输入并直接围绕这些量构建奖励函数，解决了这一局限。在训练中，执行器（Actor）仅依赖本体感知，实现了零样本硬件迁移。该策略无需参考轨迹或脚本化接触，即可覆盖完整的恢复谱系：小扰动的踝关节/髋关节策略、大推力的纠正步态，以及利用手、肘、膝的多接触站立。在Isaac Lab中基于Unitree H1-2训练，该策略在随机初始姿态和非脚本化跌倒配置下实现了93.4%的恢复率。消融研究表明，这种平衡感知结构对于发现站立行为至关重要。

### 2. 方法动机分析
*   **驱动力**：将经典控制理论中成熟的平衡分析（如捕获点理论）与现代RL的适应性相结合，赋予机器人对“可恢复性”的显式理解。
*   **现有方法痛点**：大多数RL方法将稳定性视为任务奖励的隐式结果，导致策略在面对复杂、非周期性的跌倒时，无法判断当前状态是否处于“不可恢复”边缘，泛化能力差。
*   **研究假设**：通过在训练阶段引入物理可解释的平衡指标作为特权信息，可以引导策略学习到更具鲁棒性和通用性的恢复行为，且无需在推理时依赖这些难以获取的量。

### 3. 方法设计详解
*   **流程总结**：
    1.  **特权训练（Asymmetric Actor-Critic）**：训练时，Critic接收完整状态（包括CoM、捕获点、动量等），Actor仅接收本体感知数据（关节位置、速度、重力投影等）。
    2.  **物理引导奖励设计**：将奖励分为三组：(I) 垂直恢复（高度跟踪、上升速度）；(II) 平衡与可捕获性（利用捕获点和CoM投影到支撑多边形的距离）；(III) 正则化与运动先验（安全约束、姿态对齐）。
    3.  **课程学习**：通过三个阶段（探索、难度扩展、约束退火）逐步引入扰动和硬件限制，确保策略能处理从简单站立到复杂多接触恢复的全过程。
*   **模型结构**：采用PPO算法，Actor和Critic均为多层感知机（MLP）。Critic利用特权信息进行价值估计，Actor在推理时仅依赖本体感知，实现零样本迁移。
*   **算法解释**：捕获点（Capture Point, $\xi$）是核心，它定义了机器人为了停止跌倒必须迈步的位置。通过将 $\xi$ 与支撑多边形（Support Hull）的距离作为奖励项，策略能自动学会何时该迈步、何时该用手支撑。

### 4. 方法对比分析
*   **本质区别**：不同于依赖参考轨迹（Reference-based）或脚本化接触（Scripted contact）的方法，本方法是完全基于物理指标的“无参考”学习。
*   **创新贡献**：首次将经典平衡指标显式嵌入到RL的Critic和Reward中，证明了这种结构化信号对于发现复杂多接触恢复行为是“必要而非可选”的。
*   **适用场景**：适用于需要高动态、多接触恢复的人形机器人，特别是在非结构化、不可预知的跌倒场景中。

### 5. 实验分析
*   **验证方法**：在Isaac Lab中进行大规模仿真训练，并在MuJoCo中进行跨模拟器验证，最后在Unitree H1-2实机上进行零样本部署。
*   **关键结果**：消融实验显示，移除平衡指标后，机器人完全无法站立（成功率从93.4%降至0%），证明了该设计的核心价值。
*   **优势与局限**：优势在于泛化性极强，无需手动设计动作序列；局限在于对非共面接触（如斜坡、台阶）的支撑多边形假设可能失效，需进一步扩展。

### 6. 实用指南
*   **实现细节**：
    *   **奖励设计**：关键在于使用高斯函数（Gaussian reward）提供平滑梯度，避免硬阈值切换带来的训练不稳定。
    *   **噪声注入**：在训练中对观测值添加噪声，对动力学参数（摩擦、质量、惯性）进行随机化，是实现零样本迁移的关键。
*   **迁移可能**：该方法可直接迁移至其他具有类似动力学特征的 legged robots（如四足机器人），只需调整支撑多边形的定义。

### 7. 总结
*   **核心思想**：利用经典平衡指标作为特权信号，引导RL学习鲁棒的恢复策略。
*   **速记版pipeline**：
    1. 训练时输入完整物理状态（特权信息）；
    2. 奖励函数显式惩罚偏离平衡点的行为；
    3. 课程学习逐步增加跌倒难度；
    4. 推理时仅使用本体感知实现零样本部署。

**Key Findings:**

- We present a unified RL policy that addresses this limitation by embedding classical balance metrics: capture point, center-of-mass state, and centroidal momentum, as privileged critic inputs and shaping rewards directly around these quantities during training, while the actor relies solely on proprioception for zero-shot hardware transfer.
- These results show that embedding interpretable balance structure into the learning framework substantially reduces time spent in failure states and broadens the envelope of autonomous recovery.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08619v1)
- [arXiv](https://arxiv.org/abs/2603.08619v1)

---

<a id='2603.08611v1'></a>
## [FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection](https://arxiv.org/abs/2603.08611v1)

**Authors:** Anqi Joyce Yang, James Tu, Nikita Dvornik, Enxu Li, Raquel Urtasun

**Published:** 2026-03-09

**Categories:** cs.CV, cs.RO

**Abstract:**

In order to navigate complex traffic environments, self-driving vehicles must recognize many semantic classes pertaining to vulnerable road users or traffic control devices. However, many safety-critical objects (e.g., construction worker) appear infrequently in nominal traffic conditions, leading to a severe shortage of training examples from driving data alone. Recent vision foundation models, which are trained on a large corpus of data, can serve as a good source of external prior knowledge to improve generalization. We propose FOMO-3D, the first multi-modal 3D detector to leverage vision foundation models for long-tailed 3D detection. Specifically, FOMO-3D exploits rich semantic and depth priors from OWLv2 and Metric3Dv2 within a two-stage detection paradigm that first generates proposals with a LiDAR-based branch and a novel camera-based branch, and refines them with attention especially to image features from OWL. Evaluations on real-world driving data show that using rich priors from vision foundation models with careful multi-modal fusion designs leads to large gains for long-tailed 3D detection. Project website is at https://waabi.ai/fomo3d/.

**Analysis:**

以下是对论文《FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection》的深度分析：

### 1. 摘要翻译
为了在复杂交通环境中导航，自动驾驶车辆必须识别多种语义类别。然而，许多关键安全对象（如施工人员）在常规交通中出现频率极低，导致仅依靠驾驶数据存在严重的训练样本短缺。近期在海量数据上训练的视觉基础模型可作为外部先验知识的良好来源，以提升泛化能力。我们提出了FOMO-3D，这是首个利用视觉基础模型进行长尾3D检测的多模态检测器。具体而言，FOMO-3D在两阶段检测范式中利用OWLv2和Metric3Dv2提供的丰富语义和深度先验，首先通过LiDAR分支和新型相机分支生成提议，随后通过专门针对OWL图像特征的注意力机制进行精炼。在真实世界驾驶数据上的评估表明，利用视觉基础模型的丰富先验并结合精心的多模态融合设计，能显著提升长尾3D检测性能。

### 2. 方法动机分析
*   **驱动力**：解决自动驾驶中长尾对象（如施工人员、碎片）因样本稀缺导致的检测性能差的问题。
*   **现有痛点**：传统方法（重采样、重加权）受限于原始数据分布；现有LiDAR-Camera融合方法多依赖LiDAR作为主模态，在稀疏、远距离或小目标上召回率低。
*   **核心直觉**：利用在海量互联网数据上预训练的视觉基础模型（OWLv2用于检测，Metric3Dv2用于深度）作为“外部大脑”，为3D检测器提供强大的语义和几何先验。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **Proposal Stage（提议阶段）**：
        *   **LiDAR分支**：基于CenterPoint生成3D提议。
        *   **相机分支（核心创新）**：将OWLv2的2D检测结果结合Metric3Dv2的深度图，通过“Frustum Lifting（视锥提升）”技术将2D检测转化为3D空间中的伪点云，进而生成3D提议。
    2.  **Refinement Stage（精炼阶段）**：
        *   采用基于Query的Transformer架构。将LiDAR和相机提议统一为Object Queries。
        *   通过多层注意力机制（Object-to-Object, LiDAR Cross-Attention, Camera Cross-Attention）融合多模态特征，最终输出3D边界框和类别。
*   **关键技术**：
    *   **Frustum-based Attention**：在相机提议阶段，通过在视锥内采样BEV特征，有效结合了2D语义与3D几何信息。
    *   **多模态融合**：不仅在输入层融合，更在Refinement阶段通过注意力机制实现特征级融合。

### 4. 方法对比分析
*   **本质区别**：FOMO-3D是首个将视觉基础模型（Foundation Models）引入闭集3D检测的方法，而非仅仅依赖特定领域的标注数据。
*   **创新贡献**：提出了一种将2D基础模型先验（语义+深度）转化为3D空间提议的有效框架，并设计了专门的视锥注意力机制。
*   **适用场景**：极度不平衡的真实驾驶场景，特别是需要高召回率的长尾目标检测。

### 5. 实验分析
*   **验证方法**：在nuScenes（城市）和自建Highway（高速）数据集上进行对比。
*   **关键结果**：在nuScenes上，Few类别的mAP从20.0提升至27.6，Many类别提升2.0 mAP，证明了对长尾目标的显著增益。
*   **优势**：显著提升了长尾目标召回率，对远距离小目标检测鲁棒性强。
*   **局限**：计算开销大（依赖重型基础模型），目前非实时，更适合离线自动标注。

### 6. 实用指南
*   **开源情况**：项目主页已公布（https://waabi.ai/fomo3d/）。
*   **实现细节**：需注意OWLv2的Prompt设计（如将“a person”作为“adult”的提示词）；训练时需缓存基础模型输出以节省计算资源。
*   **迁移可能**：该框架可迁移至任何需要结合2D语义先验的3D感知任务，如机器人抓取或室内场景理解。

### 7. 总结
*   **核心思想**：利用视觉基础模型先验，通过视锥融合提升长尾3D检测。
*   **速记版pipeline**：
    1. 提取2D语义与深度先验；
    2. 将2D检测提升为3D伪提议；
    3. 融合LiDAR与相机特征；
    4. 通过注意力机制精炼检测结果。

**Key Findings:**

- We propose FOMO-3D, the first multi-modal 3D detector to leverage vision foundation models for long-tailed 3D detection.
- Specifically, FOMO-3D exploits rich semantic and depth priors from OWLv2 and Metric3Dv2 within a two-stage detection paradigm that first generates proposals with a LiDAR-based branch and a novel camera-based branch, and refines them with attention especially to image features from OWL.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08611v1)
- [arXiv](https://arxiv.org/abs/2603.08611v1)

---

<a id='2603.08551v1'></a>
## [mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud](https://arxiv.org/abs/2603.08551v1)

**Authors:** Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki

**Published:** 2026-03-09

**Categories:** cs.CV, cs.IR

**Abstract:**

Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.

**Analysis:**

这是一份关于论文《mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud》的深度技术分析。

### 1. 摘要翻译
姿态估计与人体行为识别（HAR）是跨领域的核心技术。尽管基于图像的方法表现优异，但在隐私保护及低光照环境下存在局限。本文利用毫米波（mmWave）雷达技术，结合图神经网络（GNN）与注意力机制进行人体姿态估计。我们的目标是捕捉雷达点云的精细细节以提升性能。为此，我们提出了一种独特的特征提取技术，充分挖掘GNN在姿态估计中的潜力。模型mmGAT在两个公开基准数据集上表现卓越，在多数场景下刷新了人体姿态估计的SOTA水平，将平均每关节位置误差（MPJPE）降低了35.6%，PA-MPJPE降低了14.1%。

### 2. 方法动机分析
*   **驱动力**：雷达点云本质上是空间中非结构化的点集，传统的CNN方法通过体素化或投影将其转化为图像，导致了空间相干性信息的丢失。作者旨在利用图结构直接建模点与点之间的内在关联。
*   **现有方法痛点**：以往研究（如CNN方法）忽略了点云内部的“互特征”（Mutual Features），即点对之间的相对距离、方向、相对速度及相对强度，导致特征表达能力不足。
*   **研究假设**：通过将雷达点云建模为图，并显式地将点对间的互特征作为边特征（Edge Features）引入GAT，可以更有效地捕捉人体骨架结构信息。

### 3. 方法设计详解
*   **流程总结**：
    1.  **图构建**：将单帧雷达点云视为有向图 $G(V, E)$，其中 $V$ 为点集，$E$ 为通过K近邻（KNN）算法构建的边。
    2.  **特征准备**：节点特征包含坐标、速度、强度；边特征（互特征）包含欧氏距离、空间方向、相对速度、相对强度。
    3.  **边特征处理**：通过3层全连接网络（FCN）对边特征进行编码，提取高维边表示。
    4.  **图注意力计算**：利用GAT模块，将节点特征与编码后的边特征融合，通过注意力权重 $\alpha_{j,k}$ 聚合邻居信息，更新目标节点表示。
    5.  **池化与回归**：通过平均池化将节点级特征聚合成图级特征，最后经由5层全连接层组成的回归头输出关键点坐标。
*   **算法解释**：公式(3)中的注意力机制不仅考虑了节点特征 $\Theta(f_{p_j}), \Theta(f_{p_k})$，还显式引入了边特征 $\Theta_e(X_{p_j edge}[k])$，这使得模型在聚合邻居信息时，能够根据点对间的物理关系动态调整权重。

### 4. 方法对比分析
*   **本质区别**：与以往仅使用节点特征或通过CNN处理的方法不同，mmGAT将点对间的物理关系（互特征）作为核心输入，实现了对点云几何结构的深度建模。
*   **创新贡献**：提出了一套完整的“节点特征+互特征”融合框架，并证明了在雷达点云处理中，显式建模边特征比单纯依赖网络隐式学习更有效。
*   **适用场景**：适用于基于毫米波雷达的单人姿态估计任务，尤其在需要高精度空间结构感知的场景下表现优异。

### 5. 实验分析
*   **验证方法**：在MARS和mRI数据集上，对比了不同数据融合策略及是否使用互特征的性能差异。
*   **关键结果**：在mRI数据集上，引入互特征后，模型在所有场景下的MPJPE和PA-MPJPE指标均有显著提升。
*   **主要优势**：通过互特征捕捉了人体骨架的结构约束，有效降低了姿态估计误差。
*   **主要局限**：目前仅支持单人姿态估计，且对雷达覆盖范围之外的遮挡或无人的噪声点处理能力有限。

### 6. 实用指南
*   **实现细节**：
    *   **K值选择**：实验证明 $K=20$ 是内存效率与性能的最佳平衡点。
    *   **数据预处理**：采用了连续三帧数据融合以增加点云密度，这是性能提升的关键。
    *   **超参数**：使用Adam优化器，初始学习率0.001，配合0.995的衰减因子。
*   **迁移可能**：该框架可直接迁移至其他基于点云的感知任务，如手势识别、跌倒检测等，只需调整输入特征维度和回归头输出即可。

### 7. 总结
*   **核心思想**：利用图注意力网络显式建模雷达点云的点对互特征。
*   **速记版pipeline**：
    1. 将雷达点云构建为图结构；
    2. 计算并编码点对间的物理互特征；
    3. 使用GAT融合节点与边特征；
    4. 通过池化与全连接层回归人体关键点。

**Key Findings:**

- To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation.
- Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation.
- Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08551v1)
- [arXiv](https://arxiv.org/abs/2603.08551v1)

---

<a id='2603.08546v1'></a>
## [Interactive World Simulator for Robot Policy Training and Evaluation](https://arxiv.org/abs/2603.08546v1)

**Authors:** Yixuan Wang, Rhythm Syed, Fangyu Wu, Mengchao Zhang, Aykut Onol, Jose Barreiros, Hooshang Nayyeri, Tony Dear, Huan Zhang, Yunzhu Li

**Published:** 2026-03-09

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Action-conditioned video prediction models (often referred to as world models) have shown strong potential for robotics applications, but existing approaches are often slow and struggle to capture physically consistent interactions over long horizons, limiting their usefulness for scalable robot policy training and evaluation. We present Interactive World Simulator, a framework for building interactive world models from a moderate-sized robot interaction dataset. Our approach leverages consistency models for both image decoding and latent-space dynamics prediction, enabling fast and stable simulation of physical interactions. In our experiments, the learned world models produce interaction-consistent pixel-level predictions and support stable long-horizon interactions for more than 10 minutes at 15 FPS on a single RTX 4090 GPU. Our framework enables scalable demonstration collection solely within the world models to train state-of-the-art imitation policies. Through extensive real-world evaluation across diverse tasks involving rigid objects, deformable objects, object piles, and their interactions, we find that policies trained on world-model-generated data perform comparably to those trained on the same amount of real-world data. Additionally, we evaluate policies both within the world models and in the real world across diverse tasks, and observe a strong correlation between simulated and real-world performance. Together, these results establish the Interactive World Simulator as a stable and physically consistent surrogate for scalable robotic data generation and faithful, reproducible policy evaluation.

**Analysis:**

### 1. 摘要翻译
动作条件视频预测模型（通常称为世界模型）在机器人领域展现出巨大潜力，但现有方法往往速度缓慢，且难以在长时程内捕捉物理一致的交互，限制了其在可扩展机器人策略训练与评估中的应用。我们提出了“交互式世界模拟器”（Interactive World Simulator），这是一个利用中等规模机器人交互数据集构建交互式世界模型的框架。我们的方法利用一致性模型（Consistency Models）进行图像解码和潜在空间动力学预测，实现了物理交互的快速且稳定模拟。实验表明，学习到的世界模型能产生交互一致的像素级预测，并在单张RTX 4090 GPU上以15 FPS的速度支持超过10分钟的稳定长时程交互。该框架支持仅在世界模型内进行可扩展的演示收集，以训练最先进的模仿策略。通过对涉及刚体、可变形物体、物体堆叠及其交互的多种任务进行广泛的实机评估，我们发现，在世界模型生成数据上训练的策略表现与在等量真实数据上训练的策略相当。此外，我们在世界模型和真实世界中评估了策略，观察到模拟与真实世界性能之间存在强相关性。这些结果确立了交互式世界模拟器作为可扩展机器人数据生成和忠实、可重复策略评估的稳定且物理一致的替代方案。

### 2. 方法动机分析
*   **驱动力**：解决机器人策略训练中真实数据获取昂贵、难以扩展，以及策略评估过程耗时、难以进行“公平对比”的痛点。
*   **现有方法痛点**：现有模型要么因重型神经网络和多步扩散过程而计算昂贵（如Sora、Diffusion Forcing），要么因预测误差累积导致长时程rollout不稳定。
*   **研究假设**：通过在紧凑的潜在空间内使用一致性模型进行动力学建模，可以实现高效、稳定且物理一致的长时程视频预测，从而作为真实世界的有效替代。

### 3. 方法设计详解
*   **流程总结**：
    1.  **阶段一（自动编码器训练）**：训练一个CNN编码器将图像映射到紧凑的2D潜在空间，并使用一致性模型解码器进行高保真重建。通过在不同噪声尺度下训练，使解码器具备从噪声输入中恢复细节的能力。
    2.  **阶段二（动力学训练）**：冻结自动编码器，训练一个动作条件的一致性模型（$F_\psi$）在潜在空间内预测下一帧。该模型以历史潜在状态和动作序列为输入，通过去噪过程预测未来状态。
    3.  **推理（自回归预测）**：给定初始图像，模型自回归地生成未来潜在状态，并由解码器渲染为视频帧。为控制计算成本，采用固定长度的上下文窗口。
*   **模型结构**：$F_\psi$ 采用3D卷积块，结合FiLM调制（用于动作注入）和时空注意力机制（用于捕捉长时程依赖）。
*   **算法解释**：一致性模型的核心在于将多步去噪过程简化为单步或少步映射，这使得模型在保持生成质量的同时，大幅提升了推理速度，满足了15 FPS的实时交互需求。

### 4. 方法对比分析
*   **本质区别**：不同于传统的扩散模型（需要多次迭代去噪），该方法利用一致性模型在潜在空间直接建模动力学，实现了推理效率与长时程稳定性的平衡。
*   **创新贡献**：首次将一致性模型应用于机器人交互式视频预测，实现了在单卡RTX 4090上超过10分钟的稳定长时程模拟。
*   **适用场景**：适用于需要大量交互数据进行模仿学习，以及需要对机器人策略进行快速、可重复评估的实验室环境。

### 5. 实验分析
*   **验证方法**：在6个真实机器人任务和1个模拟任务上，对比Cosmos、UVA、Dreamer4等基线模型。
*   **关键结果**：在MSE、FID、PSNR等指标上全面超越基线；在策略训练实验中，使用100%模拟数据训练的策略性能与100%真实数据相当。
*   **主要优势**：推理速度快（15 FPS）、长时程稳定性高、模拟与真实世界性能强相关。
*   **主要局限**：在某些复杂任务中，模拟器与真实世界仍存在微小的Sim-to-Real Gap（表现为模拟器略微乐观）。

### 6. 实用指南
*   **开源情况**：项目主页已提供相关资源（https://www.yixuanwang.me/interactive_world_sim/）。
*   **实现细节**：训练分为两阶段，建议使用高性能GPU（如H200）进行训练，推理阶段对显存要求较低。
*   **迁移可能**：该框架高度通用，只需更换机器人交互数据集即可迁移至其他操作任务。

### 7. 总结
*   **核心思想**：利用一致性模型实现高效、稳定的交互式长时程视频预测。
*   **速记版pipeline**：
    1. 训练自动编码器实现图像到潜在空间的压缩。
    2. 训练动作条件一致性模型学习潜在空间动力学。
    3. 自回归推理生成未来视频帧。
    4. 在模拟器中进行策略训练与评估。

**Key Findings:**

- We present Interactive World Simulator, a framework for building interactive world models from a moderate-sized robot interaction dataset.
- Our approach leverages consistency models for both image decoding and latent-space dynamics prediction, enabling fast and stable simulation of physical interactions.
- Our framework enables scalable demonstration collection solely within the world models to train state-of-the-art imitation policies.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08546v1)
- [arXiv](https://arxiv.org/abs/2603.08546v1)

---

<a id='2603.08540v1'></a>
## [PCFEx: Point Cloud Feature Extraction for Graph Neural Networks](https://arxiv.org/abs/2603.08540v1)

**Authors:** Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki

**Published:** 2026-03-09

**Categories:** cs.CV, cs.IR

**Abstract:**

Graph neural networks (GNNs) have gained significant attention for their effectiveness across various domains. This study focuses on applying GNN to process 3D point cloud data for human pose estimation (HPE) and human activity recognition (HAR). We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph. Moreover, we introduce a GNN architecture designed to efficiently process these features. Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR. The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models. This work demonstrates the great potential of feature extraction incorporated with GNN modeling approach to enhance the precision of point cloud processing.

**Analysis:**

### 1. 摘要翻译
图神经网络（GNN）因其在多个领域的有效性而备受关注。本研究专注于应用GNN处理3D点云数据，以实现人体姿态估计（HPE）和人体活动识别（HAR）。我们提出了一种新颖的点云特征提取技术，通过将点云视为图，在点、边和图级别捕获有意义的信息。此外，我们引入了一种专门设计用于高效处理这些特征的GNN架构。我们的方法在四个最流行的公开毫米波雷达数据集（三个用于HPE，一个用于HAR）上进行了评估。结果显示，所有三个HPE基准测试的误差均显著降低，且在基于毫米波雷达的HAR中达到了98.8%的总体准确率，优于现有的最先进模型。这项工作展示了将特征提取与GNN建模方法相结合，以提高点云处理精度的巨大潜力。

### 2. 方法动机分析
*   **驱动力**：现有基于CNN或Transformer的雷达点云处理方法往往忽略了点云内部的结构信息，且容易受到噪声干扰。作者旨在通过图结构建模，显式地捕获点云的局部几何关系和全局统计特征。
*   **现有方法痛点**：
    1.  **空间扭曲**：将点云投影为图像（如CNN方法）会破坏点与点之间的空间邻近关系。
    2.  **噪声敏感**：现有方法缺乏对点云内部结构的鲁棒建模，导致对噪声点敏感，收敛困难。
    3.  **信息丢失**：许多方法仅使用原始坐标，忽略了统计特征（如分布、距离、方向等）对任务的辅助作用。
*   **研究假设**：通过显式提取点、边、帧三个层级的特征，并利用图注意力机制（GAT）进行融合，可以显著增强模型对复杂几何结构的理解能力，从而提升HPE和HAR的精度。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据预处理**：通过帧融合（Frame Fusion）增加密度，并使用网格化采样（Node Downsampling）降低计算复杂度。
    2.  **特征提取（Statbox）**：对每个点计算19维特征（原始5维 + 14维统计特征，如均值、标准差、分位数等）。
    3.  **图构建**：基于K-近邻（kNN）构建有向图，提取6维边特征（距离、角度、相对速度、相对强度）。
    4.  **并行分支**：同时处理点/边特征与帧级全局统计特征，通过残差路径增强信息流。
    5.  **GNN推理**：利用GAT层进行特征聚合，最后通过预测头输出结果。
*   **模型结构**：核心是**Statbox**模块，它通过10种统计算子提取局部和全局分布信息。GAT层负责根据学习到的注意力权重，动态决定邻居节点对目标节点的影响。
*   **算法解释**：公式(5)是核心，它将目标节点的原始特征与邻居节点的加权特征进行聚合，实现了对局部邻域信息的自适应融合。

### 4. 方法对比分析
*   **本质区别**：不同于将点云视为“图像”或“序列”，本方法将其视为“图”，并引入了多层级（点-边-帧）的显式特征工程。
*   **创新贡献**：
    1.  **Statbox模块**：首次将19维统计特征引入雷达点云处理。
    2.  **多级特征融合**：将全局帧特征作为残差路径，有效提升了模型对整体场景的感知。
    3.  **通用性**：该特征提取方法可作为插件，迁移至CNN、Transformer等其他架构中。

### 5. 实验分析
*   **验证方法**：在MARS、mRI、MMFi（HPE）和MMActivity（HAR）四个数据集上进行对比实验。
*   **关键结果**：在所有HPE基准测试中显著降低了MPJPE误差，在HAR任务中达到了98.8%的准确率。
*   **主要优势**：对噪声鲁棒，特征表示更丰富，在不同任务间具有良好的泛化性。
*   **主要局限**：特征提取过程增加了预处理时间，且在极度密集的点云上计算开销较大。

### 6. 实用指南
*   **开源情况**：论文基于PyTorch和PyTorch-Geometric实现。
*   **实现细节**：建议在处理密集点云时使用网格下采样（Cell size: 0.035m），K值设为20。
*   **迁移可能**：Statbox模块是高度解耦的，可直接作为预处理层插入任何点云处理网络，用于增强输入特征维度。

### 7. 总结
*   **核心思想**：通过多级图特征工程与注意力机制，实现对雷达点云的鲁棒建模。
*   **速记版pipeline**：
    1. 融合多帧数据并进行网格采样；
    2. 计算点、边及全局帧的统计特征；
    3. 构建kNN图并利用GAT进行特征聚合；
    4. 通过预测头输出姿态或活动类别。

**Key Findings:**

- We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph.
- Moreover, we introduce a GNN architecture designed to efficiently process these features.
- Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR.
- The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08540v1)
- [arXiv](https://arxiv.org/abs/2603.08540v1)

---

