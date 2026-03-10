time: 20260310

# Arxiv Computer Vision Papers - 2026-03-10

## Executive Summary

## **计算机视觉领域 arXiv 论文日报执行摘要 (2026-03-09)**

**1. 主要主题与趋势**

今日的10篇论文清晰地反映了当前计算机视觉研究的三大核心趋势：

*   **多模态与具身智能的深度融合**：超过一半的论文（2, 4, 5, 7, 10）聚焦于如何将视觉模型（VLMs、基础模型）与机器人操作、3D感知和物理世界模拟相结合。研究重点从“看”转向“理解并交互”，强调**视觉-语言-动作**的闭环。
*   **三维感知与生成的效率与泛化挑战**：多篇论文（3, 5, 6, 8, 9）致力于解决3D/4D数据（点云、雷达、视频）处理中的长尾分布、长序列生成和复杂场景理解问题。核心在于设计更**高效、可扩展的架构**（如层次化、图神经网络、占用表示）以处理真实世界的复杂性。
*   **基础模型的后训练与高效适配**：一个显著的技术趋势是利用预训练大模型，并通过**轻量级适配技术**（如提示调优、后训练）快速赋予其特定领域能力（如前景理解、操作策略），避免全参数微调的高成本。

**2. 重点与创新性论文**

*   **《HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising》**：提出分层去噪的自回归框架，直接针对**长视频生成**的算力与一致性瓶颈，是扩散模型在时序生成领域的重要效率创新。
*   **《OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras》**：将**全景占用网络**扩展到4D（3D+时间）跟踪，并适配环视鱼眼相机，为自动驾驶提供了更稠密、更实用的动态场景理解方案，工程与理论价值兼备。
*   **《AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models》**：通过**预测性隐世界模型**对视觉语言动作模型进行后训练，为实现可扩展的机器人操作提供了一条新路径，将模型预测控制与大型模型的知识相结合，思路新颖。

**3. 新兴研究方向与技术**

*   **“视觉-语言-动作”三联体的统一建模**：如`Exp-Force`和`AtomVLA`所示，研究正从静态的视觉-语言对齐，迈向动态的、以任务和物理经验为条件的决策生成。
*   **非视觉传感器与视觉的互补融合**：`mmGAT`利用毫米波雷达点云进行姿态估计，展示了在视觉受限场景下，多模态感知（尤其非光学传感器）的重要性。
*   **以具身评估为导向的世界模拟**：`Interactive World Simulator`的出现，标志着研究社区对**高保真、可交互仿真环境**的需求激增，以支持机器人策略的规模化训练与可靠评估。
*   **针对长尾与复杂分布的3D感知**：`FOMO-3D`直接利用视觉基础模型解决3D检测中的长尾问题，是将2D视觉先验注入3D任务的典型尝试。

**4. 推荐精读论文**

根据研究方向的普适性和技术影响力，建议优先阅读以下三篇：

1.  **《HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising》**：**推荐给所有关注生成式AI和视频理解的研究者**。其解决长视频生成效率的核心思路可能具有范式影响。
2.  **《OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras》**：**强烈推荐给自动驾驶、三维场景理解领域的研究者**。它代表了下一代环境感知的前沿方向，实用性强。
3.  **《AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models》**：**推荐给机器人学习、具身AI和模型高效适配方向的研究者**。它展示了如何通过世界模型桥接大模型与精细控制，是一个有潜力的技术框架。

**总结**：本日论文集表明，计算机视觉的研究前沿正快速向**具身化、三维化、高效化**演进。核心驱动力是如何让视觉智能在复杂、动态的物理世界中安全、可靠且高效地行动。建议结合自身研究方向，重点关注多模态交互、3D场景理解与生成以及基础模型高效适配的相关工作。

---

## Table of Contents

1. [Scale Space Diffusion](#2603.08709v1)
2. [FVG-PT: Adaptive Foreground View-Guided Prompt Tuning for Vision-Language Models](#2603.08708v1)
3. [HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising](#2603.08703v1)
4. [Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models](#2603.08668v1)
5. [FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection](#2603.08611v1)
6. [mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud](#2603.08551v1)
7. [Interactive World Simulator for Robot Policy Training and Evaluation](#2603.08546v1)
8. [PCFEx: Point Cloud Feature Extraction for Graph Neural Networks](#2603.08540v1)
9. [OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras](#2603.08521v1)
10. [AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models](#2603.08519v1)

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

作为计算机视觉和机器学习领域的专家，我对这篇论文《Scale Space Diffusion》的分析如下：

### 1. 核心贡献总结
该论文揭示了扩散模型（Diffusion Models）的去噪过程与经典尺度空间理论（Scale-space theory）之间的深层数学联系，指出高噪声扩散状态本质上等同于低分辨率图像。基于此，作者提出了一种“尺度空间扩散”（Scale Space Diffusion）框架，通过在扩散过程中引入下采样降质，实现了在不同分辨率尺度上的高效生成，并配套设计了能够自适应处理多分辨率的 Flexi-UNet 网络架构。

### 2. 关键创新与方法论
*   **理论桥梁构建**：论文将扩散模型的去噪过程重新诠释为尺度空间中的平滑过程，证明了高噪声状态下的高分辨率计算是冗余的。
*   **广义扩散框架**：将扩散过程推广至包含线性降质（如下采样）的范畴，使得模型可以在低分辨率空间进行大部分计算，仅在必要时恢复高频细节。
*   **Flexi-UNet 架构**：这是一种创新的 UNet 变体，能够根据当前尺度需求动态调整计算路径，实现“分辨率保持”或“分辨率提升”的去噪，避免了传统模型在全分辨率下进行所有计算的算力浪费。

### 3. 对领域的潜在影响
*   **计算效率的飞跃**：该研究直接挑战了扩散模型必须在全分辨率下进行长序列去噪的范式。通过在低分辨率空间处理大部分信息，该方法有望显著降低生成式 AI 的推理成本和显存占用。
*   **理论视角的更新**：它为理解扩散模型内部的“信息流”提供了一个基于经典信号处理的视角，有助于解释模型为何在不同时间步关注不同尺度的特征。
*   **架构设计的灵活性**：Flexi-UNet 的设计理念可能成为未来多尺度生成模型的基础组件，推动模型向更高效、更具伸缩性的方向发展。

### 4. 相关领域与应用受益
*   **高分辨率图像生成**：直接受益于该技术，能够以更低的计算代价生成超高清图像。
*   **实时视频生成**：视频生成对算力要求极高，尺度空间扩散可以大幅减少处理视频帧时的冗余计算。
*   **边缘设备部署**：由于降低了对全分辨率计算的依赖，该方法使得在资源受限的移动端或嵌入式设备上运行高质量扩散模型成为可能。
*   **图像超分辨率与修复**：该框架天然契合多尺度任务，在图像重建领域具有极高的应用潜力。

### 5. 可推断的局限性
*   **降质带来的信息损失**：虽然理论上高噪声状态下高分辨率信息冗余，但在实际训练中，如何精确平衡下采样带来的信息丢失与计算增益是一个挑战，可能会在极高频细节的恢复上产生伪影。
*   **训练复杂性**：引入多尺度降质后，训练过程可能需要更精细的超参数调节（如不同尺度下的噪声调度策略），以确保模型在不同分辨率切换时的平滑性。
*   **泛化能力**：Flexi-UNet 的架构设计是否能完美适配所有类型的扩散模型（如 Latent Diffusion 或 Consistency Models），以及在处理极端非线性降质时的表现，仍需进一步验证。

**专家点评：**
这篇论文的趣味性在于它**“返璞归真”**——利用经典的尺度空间理论来优化现代的深度生成模型。它不仅是一个工程上的优化，更是在尝试回答“扩散模型到底在学什么”这一核心问题。如果该方法能在大规模数据集上证明其在保持生成质量的同时显著降低计算开销，它极有可能成为下一代高效扩散模型架构的标准范式。

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
基于CLIP的提示调优（Prompt Tuning）使预训练视觉-语言模型（VLM）能够高效适应下游任务。尽管现有研究已取得显著进展，但它们对调优过程中VLM内部注意力表示的变化关注有限。本文将提示调优预测的失败模式归因于视觉编码器前景注意力的偏移，并提出了**前景视图引导提示调优（FVG-PT）**，这是一种自适应的即插即用前景注意力引导模块，旨在缓解这种偏移。具体而言，FVG-PT引入了一个可学习的“前景可靠性门（Foreground Reliability Gate）”来自动增强前景视图质量，应用“前景蒸馏补偿（Foreground Distillation Compensation）”模块引导视觉注意力聚焦于前景，并进一步引入“先验校准（Prior Calibration）”模块，以减轻因过度关注前景而导致的泛化能力下降。在多个骨干模型和数据集上的实验证明了FVG-PT的有效性和兼容性。

### 2. 方法动机分析
*   **驱动力**：作者观察到提示调优过程中，视觉编码器的注意力往往会从目标物体（前景）偏移到无关的背景中，导致语义对齐失效，这是导致模型预测失败的核心原因。
*   **现有方法痛点**：现有方法要么缺乏对前景质量的控制（可能引入噪声），要么过度关注前景导致“基类-新类权衡（BNT）”问题，即在提升基类性能的同时损害了对新类的泛化能力。
*   **研究假设**：通过自适应地引导视觉注意力聚焦于可靠的前景区域，并解耦基类与新类的优化路径，可以同时提升模型在特定任务上的准确性与对新任务的泛化能力。

### 3. 方法设计详解
FVG-PT是一个即插即用的框架，主要包含三个核心模块：
1.  **前景可靠性门（FRG）**：利用预训练的SEEM分割模型提取前景，并通过一个MLP评估前景的可靠性（输出信任分数 $r$）。该模块基于分布熵、相似度和几何比例三个指标，确保模型只在前景质量高时才进行强引导。
2.  **前景蒸馏补偿（FDC）**：在冻结的骨干模型后插入轻量级适配器（Adapter）。它根据信任分数 $r$，通过蒸馏损失引导模型将注意力从全图特征对齐转向前景特征对齐，从而修正注意力偏移。
3.  **先验校准（PC）**：为了解决BNT问题，PC模块在逻辑层（Logit level）将新类分支与前景增强的基类分支解耦。它通过“骨干可靠性门（BRG）”学习自适应权重，平衡微调后的模型与原始CLIP先验，从而保留通用知识，提升新类泛化。

### 4. 方法对比分析
*   **本质区别**：与以往仅关注提示向量设计或梯度约束的方法不同，FVG-PT通过显式的前景监督和自适应的质量评估，实现了对视觉编码器内部注意力分布的精细化控制。
*   **创新贡献**：提出了“自适应信任机制”，即模型能够根据前景的质量动态调整引导强度，而非盲目地进行前景对齐。
*   **适用场景**：适用于所有基于CLIP的提示调优架构，尤其在需要精细化视觉对齐的分类任务中表现优异。

### 5. 实验分析
*   **验证方法**：在11个数据集上，将FVG-PT挂载到CoOp、KgCoOp、PromptSRC、MMRL等4种主流提示调优骨干模型上进行基到新（Base-to-New）的泛化测试。
*   **关键结果**：在所有骨干模型上，FVG-PT均实现了基类准确率和新类泛化能力的双重提升，且在跨数据集迁移任务中表现稳健。
*   **主要优势**：参数高效（仅增加约0.13M参数），即插即用，且有效缓解了提示调优中的过拟合与泛化退化问题。
*   **主要局限**：对于缺乏文本分支的纯视觉提示调优方法（如VPT）不适用；在极小样本数据集上（如EuroSAT）需要调整训练轮数以避免欠拟合。

### 6. 实用指南
*   **开源情况**：代码已开源（https://github.com/JREion/FVG-PT）。
*   **实现细节**：建议将FDC适配器隐藏层维度设为64，可靠性门MLP隐藏层维度设为32，温度系数 $\tau_d=2.0$。
*   **迁移可能**：该方法模块化程度高，可直接迁移至任何基于Transformer的视觉-语言对齐模型中。

### 7. 总结
*   **核心思想**：通过自适应前景引导与分支解耦，修正注意力偏移并平衡泛化能力。
*   **速记版pipeline**：
    1. 提取图像前景掩码；
    2. 评估前景可靠性并计算信任分数；
    3. 插入适配器进行前景注意力蒸馏；
    4. 解耦新类分支并进行先验校准。

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
自回归（AR）扩散模型为生成无限长度视频提供了有前景的框架，但如何保持时间连续性并防止因误差累积导致的质量下降仍是核心挑战。现有方法通常依赖高度去噪的上下文，但这会以高确定性传播预测误差，从而加剧退化。本文提出，高度干净的上下文并非必要。受双向扩散模型启发，我们提出HiAR，一种层次化去噪框架。它颠倒了传统的生成顺序：不再顺序完成每个块，而是在每个去噪步骤中跨所有块执行因果生成，使每个块始终在相同噪声水平的上下文下进行条件化。这种层次结构天然支持流水线并行推理，在4步设置下实现了约1.8倍的加速。此外，我们引入了双向注意力模式下的前向KL正则化器，以防止自滚动蒸馏中的低运动捷径，从而在保持因果推理的同时保留运动多样性。在VBench上，HiAR实现了最佳的整体得分和最低的时间漂移。

### 2. 方法动机分析
- **驱动力**：解决自回归视频生成中“误差累积”与“推理效率”之间的矛盾。
- **现有痛点**：传统AR方法（如Self-Forcing）在生成新块时，强制要求前序块完全去噪（$t_c=0$）。这虽然锚定了时间一致性，但将前序块的预测误差以最大置信度传播，导致长视频中出现过饱和、语义漂移等现象。
- **研究假设**：上下文的噪声水平与当前块的去噪步噪声水平匹配（$t_c = t_{j+1}$）时，既能提供足够的信号保证时间一致性，又能有效抑制误差传播。

### 3. 方法设计详解
- **层次化去噪（Hierarchical Denoising）**：
  - **核心逻辑**：将“块优先（Block-first）”的生成顺序改为“步优先（Step-first）”。
  - **流程**：在第$j$个去噪步，模型同时处理所有块的第$j$步去噪。每个块$B_n$的上下文$c_{<n}$不再是完全去噪的，而是处于与当前块相同的噪声水平$t_{j+1}$。
  - **并行化**：由于第$j$步的块$B_n$仅依赖于第$j$步的$B_{<n}$和第$j-1$步的$B_n$，这使得网格上的反对角线元素可以并行计算，通过KV Cache交换信息，实现流水线并行。
- **前向KL正则化（Forward-KL Regularization）**：
  - **动机**：DMD（分布匹配蒸馏）本质是反向KL，倾向于模式坍缩（低运动捷径）。
  - **实现**：在双向注意力模式下，利用教师模型生成的高质量轨迹，计算学生模型单步Euler更新与教师轨迹之间的前向KL损失。
  - **解耦设计**：仅在双向注意力模式下计算该损失，且仅作用于前$K$步，避免干扰因果推理的训练。

### 4. 方法对比分析
- **本质区别**：从“串行块生成”转向“跨块同步去噪”，通过匹配噪声水平而非追求绝对干净的上下文来抑制误差。
- **创新贡献**：提出层次化去噪范式，在提升长视频稳定性的同时，通过流水线并行显著提升了推理速度（1.8×）。
- **适用场景**：长视频生成、流式视频输出、对时间一致性要求极高的交互式世界模型。

### 5. 实验分析
- **关键结果**：在VBench 20s生成任务中，HiAR在总分（0.821）和漂移指标（0.257）上均优于现有AR方法。
- **主要优势**：显著降低了长视频的时间漂移，解决了运动坍缩问题，且推理效率更高。
- **主要局限**：对训练阶段的计算资源要求较高（需要进行大规模的轨迹蒸馏）。

### 6. 实用指南
- **开源情况**：项目主页已提供代码（https://jacky-hate.github.io/HiAR/）。
- **实现细节**：
  - 关键超参数：$K=1$（仅在第一步施加正则化），$\lambda=0.1$。
  - 训练策略：使用Wan2.1-1.3B作为基础模型，采用5:1的生成器/判别器更新比例。
- **迁移可能**：该层次化去噪范式可直接迁移至其他基于扩散的自回归生成任务（如音频、长文本生成）。

### 7. 总结
- **核心思想**：通过匹配噪声水平的上下文进行层次化去噪，抑制误差累积。
- **速记版pipeline**：
  1. 准备多块视频噪声输入。
  2. 跨块同步执行去噪步，保持上下文噪声水平匹配。
  3. 利用KV Cache实现流水线并行加速。
  4. 引入前向KL正则化防止运动坍缩。

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
准确的预接触抓取力选择对于安全可靠的机器人操作至关重要。自适应控制器虽能在接触后调节力，但仍需合理的初始估计。初始力过小会导致抓取失败，过大则可能损坏易碎物体。对于难以通过解析建模的柔性夹爪，这一权衡尤为困难。我们提出了Exp-Force，一种基于经验条件化的框架，仅通过单张RGB图像即可预测最小可行抓取力。该方法检索一组相关的先验抓取经验，并将其作为上下文输入视觉语言模型（VLM）进行推理，无需解析接触模型或人工设计的启发式规则。在129个物体实例上，Exp-Force实现了0.43 N的最佳平均绝对误差（MAE），较零样本推理降低了72%的误差。在30个未见物体的真实世界测试中，其适当力选择率从63%提升至87%。结果表明，Exp-Force通过利用先验交互经验，实现了可靠且可泛化的预抓取力选择。

### 2. 方法动机分析
*   **驱动力**：解决柔性夹爪在接触前无法准确预估抓取力的问题，避免因力过大导致物体损坏或力过小导致滑脱。
*   **现有痛点**：传统解析方法（如库仑摩擦模型）依赖精确的物理参数（摩擦系数、物体质量等），在面对复杂柔性夹爪和多样化物体时极其脆弱；数据驱动的端到端学习方法则受限于训练数据的多样性，泛化能力差。
*   **研究假设**：人类通过视觉联想相似物体的过往交互经验来预估力，因此，利用VLM的常识推理能力，结合检索到的相似物体交互经验（RAG），可以隐式地捕捉复杂的接触物理特性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **对象描述生成**：利用描述VLM（Descriptor VLM）将目标物体图像 $I_o$ 转化为文本描述 $T_o$，包含材质、形状、刚度等物理属性。
    2.  **经验检索**：将 $I_o$ 和 $T_o$ 映射到共享多模态嵌入空间，计算余弦相似度，从经验池中检索出 $k$ 个最相似的先验交互案例（包含物体名、质量、描述、图像及真实力 $F^*$）。
    3.  **经验条件化推理**：将检索到的 $k$ 个案例作为上下文（Context），连同任务信息 $C$ 和目标图像 $I_o$ 一并输入预测VLM（Predictor VLM），直接输出预测力 $\hat{F}_o$。
*   **模型结构**：采用模块化设计，Descriptor VLM负责语义提取，Embedding Model负责特征对齐，Predictor VLM负责基于上下文的决策。
*   **算法解释**：核心在于将“绝对回归问题”转化为“基于相似案例的比较推理问题”。通过在Prompt中注入成功案例，VLM不再是“物理引擎”，而是“语义插值器”，从而绕过了对复杂接触力学的显式建模。

### 4. 方法对比分析
*   **本质区别**：放弃了显式物理建模和大规模端到端训练，转而利用大模型的上下文学习（In-context Learning）能力，通过检索相似经验实现“即插即用”的泛化。
*   **创新贡献**：首次将RAG引入机器人抓取力预测；证明了VLM可以通过少量示例隐式学习柔性夹爪的复杂接触物理。
*   **适用场景**：适用于需要高精度力控制、物体种类多样且难以建立精确物理模型的机器人抓取任务。

### 5. 实验分析
*   **验证方法**：在129个物体上进行5折交叉验证，并在真实机器人上测试30个未见物体。
*   **关键结果**：MAE误差降低72%，适当力选择率从63%提升至87%。
*   **优势**：样本效率极高（$k=6$即可达到最优），无需针对新物体重新训练。
*   **局限**：依赖VLM的推理能力，若检索到的案例与目标物体差异过大，预测可能失效；且大模型推理存在计算延迟。

### 6. 实用指南
*   **开源情况**：项目主页 http://expforcesubmission.github.io/Exp-Force-Website/。
*   **实现细节**：关键超参数为 $k$（检索示例数），实验表明 $k=6$ 到 $10$ 之间性能趋于稳定。数据预处理需确保经验池中包含多样化的物体属性。
*   **迁移可能**：该框架可直接迁移至其他需要“基于经验的参数预测”任务，如机器人操作中的速度规划、轨迹平滑度估计等。

### 7. 总结
*   **核心思想**：利用检索到的相似交互经验，引导VLM进行语义推理以预测抓取力。
*   **速记版pipeline**：
    1. 提取目标物体视觉语义描述；
    2. 在经验库中检索相似案例；
    3. 将案例作为上下文输入大模型；
    4. 大模型输出预测力。

**Key Findings:**

- We propose Exp-Force, an experience-conditioned framework that predicts the minimum feasible grasping force from a single RGB image.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08668v1)
- [arXiv](https://arxiv.org/abs/2603.08668v1)

---

<a id='2603.08611v1'></a>
## [FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection](https://arxiv.org/abs/2603.08611v1)

**Authors:** Anqi Joyce Yang, James Tu, Nikita Dvornik, Enxu Li, Raquel Urtasun

**Published:** 2026-03-09

**Categories:** cs.CV, cs.RO

**Abstract:**

In order to navigate complex traffic environments, self-driving vehicles must recognize many semantic classes pertaining to vulnerable road users or traffic control devices. However, many safety-critical objects (e.g., construction worker) appear infrequently in nominal traffic conditions, leading to a severe shortage of training examples from driving data alone. Recent vision foundation models, which are trained on a large corpus of data, can serve as a good source of external prior knowledge to improve generalization. We propose FOMO-3D, the first multi-modal 3D detector to leverage vision foundation models for long-tailed 3D detection. Specifically, FOMO-3D exploits rich semantic and depth priors from OWLv2 and Metric3Dv2 within a two-stage detection paradigm that first generates proposals with a LiDAR-based branch and a novel camera-based branch, and refines them with attention especially to image features from OWL. Evaluations on real-world driving data show that using rich priors from vision foundation models with careful multi-modal fusion designs leads to large gains for long-tailed 3D detection. Project website is at https://waabi.ai/fomo3d/.

**Analysis:**

这是一份关于论文 **FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection** 的深度分析报告。

### 1. 摘要翻译
为了在复杂的交通环境中导航，自动驾驶车辆必须识别多种语义类别，包括弱势道路使用者或交通控制设备。然而，许多安全关键型物体（如施工人员）在常规交通条件下出现频率极低，导致仅从驾驶数据中获取的训练样本严重不足。最近在海量数据上训练的视觉基础模型（Vision Foundation Models）可以作为外部先验知识的良好来源，以提高泛化能力。我们提出了 FOMO-3D，这是第一个利用视觉基础模型进行长尾 3D 检测的多模态检测器。具体而言，FOMO-3D 在两阶段检测范式中利用了来自 OWLv2 和 Metric3Dv2 的丰富语义和深度先验，首先通过 LiDAR 分支和新颖的基于相机的分支生成提案，并特别通过对 OWL 图像特征的注意力机制进行细化。在真实世界驾驶数据上的评估表明，利用视觉基础模型的丰富先验并结合精心的多模态融合设计，可以显著提升长尾 3D 检测性能。

### 2. 方法动机分析
- **驱动力**：解决自动驾驶中长尾物体（如施工人员、碎片）因数据稀缺导致的检测性能低下问题。
- **现有方法痛点**：传统方法（如重采样、损失重加权）受限于原始数据集的分布；现有多模态融合方法（如 MMF、MMLF）依赖 LiDAR 的高召回率，在处理稀疏、远距离或小物体时表现不佳。
- **研究假设**：利用在海量互联网数据上预训练的视觉基础模型（如 OWLv2 和 Metric3Dv2）作为外部先验，可以弥补自动驾驶数据集在长尾类别上的监督不足。

### 3. 方法设计详解
FOMO-3D 采用两阶段检测范式：
1.  **提案阶段（Proposal Stage）**：
    *   **LiDAR 分支**：基于 CenterPoint 架构，处理点云生成 3D 提案。
    *   **相机分支（核心创新）**：利用 OWLv2 进行 2D 检测，结合 Metric3Dv2 的深度估计，将 2D 检测结果“提升（Lifting）”至 3D 空间，生成相机提案。
    *   **融合**：通过“视锥（Frustum）”机制，将 2D 框内的像素点云化并编码为 BEV 特征，与 LiDAR BEV 特征拼接。
2.  **细化阶段（Refinement Stage）**：
    *   将提案转化为“对象查询（Object Queries）”。
    *   通过注意力机制（Attention-based Refinement）融合 LiDAR 特征、OWL 图像特征和对象间关系，进行迭代细化。
    *   **关键操作**：使用可变形注意力（Deformable Attention）高效采样特征，并利用 OWL 的语义特征进行分类增强。

### 4. 方法对比分析
- **本质区别**：FOMO-3D 是首个在闭集 3D 检测中系统性引入视觉基础模型先验的方法，而非仅仅依赖传统的传感器融合。
- **创新贡献**：提出了基于视锥的相机提案生成方法，以及一种能够同时利用 2D 语义先验和 3D 几何先验的注意力融合架构。
- **适用场景**：特别适用于长尾类别检测、远距离小物体检测以及对语义理解要求较高的复杂场景。

### 5. 实验分析
- **验证方法**：在 nuScenes（城市）和内部 Highway（高速）数据集上进行评估。
- **关键结果**：在 nuScenes 上，Few 类别的 mAP 从 20.0 提升至 27.6，Many 类别也有 2.0 的增益。
- **优势**：显著提升了对稀有类别的召回率，且在远距离场景下表现稳健。
- **局限**：由于使用了大型基础模型（OWL-Large, Metric3D-Giant），计算开销大，目前无法实现实时推理，更适合离线自动标注。

### 6. 实用指南
- **开源情况**：项目主页为 https://waabi.ai/fomo3d/。
- **实现细节**：需注意视锥采样参数（$N_x, N_y, N_z, \delta$）的调整；OWL 的提示词（Prompting）设计对性能影响较大。
- **迁移可能**：该架构可迁移至其他需要利用 2D 语义先验增强 3D 感知的任务（如机器人操作、室内 3D 重建）。

### 7. 总结
- **核心思想**：利用视觉基础模型先验，通过两阶段融合提升长尾 3D 检测能力。
- **速记版pipeline**：
    1. 预处理：利用基础模型提取 2D 语义和深度先验。
    2. 提案：LiDAR 和相机分支并行生成 3D 提案。
    3. 融合：通过视锥注意力机制整合多模态特征。
    4. 细化：利用 Transformer 结构对提案进行迭代优化。

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

### 1. 摘要翻译
姿态估计与人体行为识别（HAR）是跨领域的核心技术。尽管基于图像的方法表现优异，但它们在隐私保护及低光照环境下的性能存在局限。本文利用毫米波（mmWave）雷达技术，结合图神经网络（GNN）与注意力机制进行人体姿态估计。我们的目标是捕捉雷达点云的精细细节以提升性能。为此，我们提出了一种独特的特征提取技术，充分挖掘GNN在姿态估计中的潜力。模型mmGAT在两个公开基准数据集上表现卓越，在多数场景下刷新了人体姿态估计的SOTA结果，将平均每关节位置误差（MPJPE）降低了35.6%，PA-MPJPE降低了14.1%。

### 2. 方法动机分析
*   **驱动力**：雷达点云本质上是空间中非结构化的点集，传统的CNN方法通过体素化或投影将其转化为图像，导致了空间相干性信息的丢失。作者希望利用图结构直接建模点与点之间的内在关联。
*   **现有方法痛点**：以往研究（如CNN方法）忽略了点云中点对之间的“互特征”（如相对距离、相对速度、相对强度），且体素化方法在处理高密度点云时存在信息损失。
*   **研究假设**：雷达点云不仅包含点的自身属性（节点特征），还包含点对之间的几何与运动关系（边特征/互特征），通过GAT（图注意力网络）融合这两类特征，能更精准地表征人体骨架结构。

### 3. 方法设计详解
*   **流程总结**：
    1.  **图构建**：将一帧雷达点云视为图 $G(V, E)$，其中 $V$ 为点集，$E$ 为通过K近邻（KNN）算法构建的边。
    2.  **特征准备**：节点特征包含坐标、速度、强度；边特征（互特征）包含欧氏距离、空间方向向量、相对速度和相对强度。
    3.  **边特征处理**：通过3层全连接层（FCN）对边特征进行编码。
    4.  **图注意力计算**：利用GAT模块，结合节点特征与编码后的边特征，计算注意力权重 $\alpha_{j,k}$，实现邻居节点信息的加权聚合。
    5.  **池化与回归**：通过平均池化将节点级特征聚合为图级特征，最后经由5层全连接层组成的预测头输出关键点坐标。
*   **算法解释**：公式(3)中的注意力机制不仅考虑了节点 $j$ 和 $k$ 的特征，还显式引入了边特征 $\Theta_e(X_{pjedge})$，这使得模型在聚合邻居信息时，能根据点对间的相对物理关系动态调整权重。

### 4. 方法对比分析
*   **本质区别**：与传统CNN方法相比，mmGAT不依赖于将点云“图像化”，而是直接在图域处理点云，保留了原始的空间拓扑信息。
*   **创新贡献**：提出了显式的“互特征”提取方法，并将其成功集成到GAT框架中，证明了边特征对于雷达姿态估计的增益。
*   **适用场景**：适用于基于毫米波雷达的单人姿态估计，特别是在需要高精度空间定位的场景。

### 5. 实验分析
*   **验证方法**：在MARS和mRI数据集上，对比了不同数据融合策略及有无互特征的情况。
*   **关键结果**：引入互特征后，模型在所有场景下的MPJPE和PA-MPJPE均有显著下降；数据融合（多帧叠加）与GAT特征处理的结合产生了协同效应。
*   **优势**：显著提升了姿态估计精度，对旋转和缩放具有更好的鲁棒性。
*   **局限**：目前仅支持单人姿态估计，且对雷达监测范围外或存在背景噪声的场景处理能力有限。

### 6. 实用指南
*   **实现细节**：
    *   **K值选择**：实验中取 $K=20$ 是性能与内存的平衡点。
    *   **数据预处理**：必须进行空间轴排序和多帧数据融合（堆叠三帧）以提升密度。
    *   **损失函数**：使用MPJPE Loss，并结合Lambda学习率调度器（衰减因子0.995）。
*   **迁移可能**：该方法可直接迁移至其他基于点云的感知任务，如手势识别、跌倒检测，只需更换预测头（Prediction Head）即可。

### 7. 总结
*   **核心思想**：利用图注意力网络融合雷达点云的节点与互特征。
*   **速记版pipeline**：
    1. 构建点云图并计算点对间的几何/运动关系；
    2. 使用全连接层编码边特征；
    3. 通过图注意力机制聚合节点与边信息；
    4. 平均池化后回归输出人体关键点。

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
动作条件视频预测模型（通常称为世界模型）在机器人领域展现出巨大潜力，但现有方法往往速度缓慢，且难以在长时程内捕捉物理一致的交互，限制了其在可扩展机器人策略训练与评估中的应用。我们提出了“交互式世界模拟器”（Interactive World Simulator），这是一个利用中等规模机器人交互数据集构建交互式世界模型的框架。我们的方法对图像解码和潜空间动力学预测均采用一致性模型（Consistency Models），实现了物理交互的快速且稳定模拟。实验表明，学习到的世界模型能产生交互一致的像素级预测，并在单张RTX 4090 GPU上以15 FPS的速度支持超过10分钟的稳定长时程交互。该框架仅需在世界模型内收集演示数据，即可训练出最先进的模仿学习策略。通过在涉及刚体、可变形物体、物体堆叠及其交互的多种任务中进行广泛的实机评估，我们发现基于世界模型生成数据训练的策略，其性能与使用等量真实数据训练的策略相当。此外，我们在世界模型和真实世界中对策略进行了评估，观察到模拟与真实世界性能之间存在强相关性。这些结果确立了交互式世界模拟器作为一种稳定、物理一致的替代方案，可用于可扩展的机器人数据生成及忠实、可复现的策略评估。

### 2. 方法动机分析
*   **驱动力**：旨在解决机器人领域中“数据获取昂贵”与“策略评估难以复现”的两大痛点，通过构建一个高效、高保真的世界模型，实现模拟环境下的数据生成与策略验证。
*   **现有痛点**：现有模型要么因重型架构（如扩散模型）导致推理速度慢，无法实时交互；要么因长时程预测中累积误差导致物理一致性崩塌。
*   **研究假设**：通过将图像解码和潜空间动力学建模统一为一致性模型，可以兼顾生成质量与推理效率，从而支持长时程、稳定的交互式模拟。

### 3. 方法设计详解
*   **流程总结**：
    1.  **阶段一（自动编码器）**：训练一个CNN编码器将RGB图像映射到紧凑的2D潜空间，并使用一致性模型解码器实现高保真重建。
    2.  **阶段二（动力学预测）**：冻结编码器，训练一个动作条件的一致性动力学模型（$F_\psi$）。该模型以历史潜状态序列和动作序列为输入，预测下一帧的潜状态。
    3.  **推理阶段**：采用自回归方式，将预测的潜状态作为后续步骤的上下文，通过不断注入噪声并去噪，实现长时程视频预测。
*   **模型结构**：$F_\psi$ 采用3D卷积块，结合FiLM调制（用于动作注入）和时空注意力机制，以捕捉复杂的物理动态。
*   **算法解释**：一致性模型通过学习将高噪声分布映射到低噪声目标，相比传统扩散模型，它能以极少的去噪步数（甚至单步）生成高质量结果，极大提升了推理速度。

### 4. 方法对比分析
*   **本质区别**：不同于传统的自回归扩散模型（计算昂贵）或简单的RNN/Transformer（长时程易漂移），该方法利用一致性模型在潜空间进行高效动力学建模。
*   **创新贡献**：首次在机器人领域证明了基于一致性模型的世界模拟器可以实现10分钟以上的稳定长时程交互，且推理速度达到15 FPS。
*   **适用场景**：适用于需要大量交互数据进行模仿学习，或需要对机器人策略进行快速、可复现评估的场景。

### 5. 实验分析
*   **验证方法**：在MuJoCo模拟环境及6个真实机器人任务（如堆叠、抓取、推移）中进行对比。
*   **关键结果**：在所有指标（MSE, FID, PSNR等）上均优于Cosmos、UVA等基线；策略训练实验显示，100%模拟数据训练的策略性能与100%真实数据几乎持平。
*   **优势**：推理速度快、长时程稳定性高、模拟与真实世界性能相关性强。
*   **局限**：在极度复杂的非结构化动态场景下，仍可能存在轻微的物理偏差。

### 6. 实用指南
*   **开源情况**：项目主页已提供，建议关注官方GitHub仓库。
*   **实现细节**：训练分为两阶段，需注意一致性模型中噪声采样策略（$\sigma_t > \sigma_s$）的设置。推理时需维护一个固定长度的上下文窗口，以防止计算量随时间线性增长。
*   **迁移可能**：该框架高度通用，只需更换数据集即可迁移至其他机器人操作任务，且对硬件要求较低（单张RTX 4090即可）。

### 7. 总结
*   **核心思想**：利用一致性模型实现高效、稳定的长时程机器人交互模拟。
*   **速记版pipeline**：
    1. 训练自动编码器压缩图像；
    2. 训练一致性动力学模型预测未来动作；
    3. 自回归生成长时程视频；
    4. 在模拟器中收集数据并训练策略。

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

以下是对论文《PCFEx: Point Cloud Feature Extraction for Graph Neural Networks》的深度分析：

### 1. 摘要翻译
图神经网络（GNN）因其在多领域的有效性而备受关注。本研究专注于应用GNN处理3D点云数据，以实现人体姿态估计（HPE）和人体活动识别（HAR）。我们提出了一种新颖的点云特征提取技术，通过将点云视为图，在点、边和图级别捕获关键信息。此外，我们引入了一种专门用于高效处理这些特征的GNN架构。该方法在四个主流毫米波雷达数据集上进行了评估（三个用于HPE，一个用于HAR）。结果显示，所有HPE基准测试的误差均显著降低，且在基于毫米波雷达的HAR中达到了98.8%的准确率，优于现有的最先进模型。本工作展示了特征提取与GNN建模相结合在提升点云处理精度方面的巨大潜力。

### 2. 方法动机分析
*   **驱动力**：解决现有雷达点云处理中对点云内部结构（空间关系）利用不足的问题，提升模型对噪声的鲁棒性。
*   **痛点**：现有CNN方法常将点云投影为图像，导致空间关系扭曲；MLP方法处理点时缺乏局部空间关联；现有GNN方法对点云内部关系建模不足，易受噪声干扰。
*   **研究假设**：通过显式提取点、边、帧三个维度的统计特征（Statbox），并结合图注意力机制（GAT），能更有效地捕获点云的几何结构，从而提升任务精度。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据预处理**：通过帧融合（Frame Fusion）增加密度，并使用网格化采样（Node Downsampling）降低计算复杂度。
    2.  **多级特征提取**：
        *   **点特征（Node Features）**：利用Statbox（包含均值、标准差、偏度、分位数等10种统计算子）提取点相对于质心的几何统计特征，将维度从5D扩展至19D。
        *   **边特征（Edge Features）**：计算K近邻间的欧氏距离、角度、相对速度和相对强度，构建6维边特征。
        *   **帧特征（Frame Features）**：对整个点云进行全局统计，提取380维的帧级特征，作为全局上下文补充。
    3.  **GNN建模**：使用共享MLP处理点/边特征，通过GAT层进行特征聚合，最后通过全连接层进行预测。
*   **核心模块**：**Statbox**是核心创新，它通过统计算子将原始稀疏点云转化为包含丰富几何分布信息的特征向量，增强了模型对局部和全局结构的感知能力。

### 4. 方法对比分析
*   **本质区别**：不同于仅依赖原始坐标的传统GNN，PCFEx引入了显式的统计特征工程（Statbox），将点云的几何分布信息直接注入网络。
*   **创新贡献**：首次引入Statbox进行多级特征提取；提出了点、边、帧三级特征融合架构；引入了针对点云密度优化的网格采样策略。
*   **适用场景**：适用于毫米波雷达点云、3D物体分类等需要强几何结构感知的任务。

### 5. 实验分析
*   **验证方法**：在MARS、mRI、MMFi（HPE）及MMActivity（HAR）四个数据集上进行对比实验。
*   **关键结果**：在HPE任务中，MPJPE和PA-MPJPE误差显著降低；在HAR任务中，准确率达到98.8%。
*   **优势**：对噪声鲁棒，特征表示更具表达力，在不同规模数据集上均有提升。
*   **局限**：特征处理阶段增加了计算开销，导致预处理时间较长。

### 6. 实用指南
*   **开源情况**：基于PyTorch和PyTorch-Geometric实现。
*   **实现细节**：建议K近邻取$K=20$；对于密集点云，务必使用网格采样以平衡计算资源；Statbox的10种统计算子是提升性能的关键。
*   **迁移可能**：Statbox模块可作为“即插即用”插件，迁移至任何基于点云的CNN、Transformer或GNN架构中。

### 7. 总结
*   **核心思想**：通过多级统计特征工程增强点云的几何表达能力。
*   **速记版pipeline**：
    1. 融合多帧数据并网格化采样；
    2. 用统计算子提取点、边、帧特征；
    3. 利用图注意力机制聚合特征；
    4. 通过预测头输出最终结果。

**Key Findings:**

- We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph.
- Moreover, we introduce a GNN architecture designed to efficiently process these features.
- Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR.
- The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08540v1)
- [arXiv](https://arxiv.org/abs/2603.08540v1)

---

<a id='2603.08521v1'></a>
## [OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras](https://arxiv.org/abs/2603.08521v1)

**Authors:** Yongzhi Lin, Kai Luo, Yuanfan Zheng, Hao Shi, Mengfei Duan, Yang Liu, Kailun Yang

**Published:** 2026-03-09

**Categories:** cs.CV, cs.RO, eess.IV

**Abstract:**

Understanding dynamic 3D environments in a spatially continuous and temporally consistent manner is fundamental for robotics and autonomous driving. While recent advances in occupancy prediction provide a unified representation of scene geometry and semantics, progress in 4D panoptic occupancy tracking remains limited by the lack of benchmarks that support surround-view fisheye sensing, long temporal sequences, and instance-level voxel tracking. To address this gap, we present OccTrack360, a new benchmark for 4D panoptic occupancy tracking from surround-view fisheye cameras. OccTrack360 provides substantially longer and more diverse sequences (174~2234 frames) than prior benchmarks, together with principled voxel visibility annotations, including an all-direction occlusion mask and an MEI-based fisheye field-of-view mask. To establish a strong fisheye-oriented baseline, we further propose Focus on Sphere Occ (FoSOcc), a framework that addresses two core challenges in fisheye occupancy tracking: distorted spherical projection and inaccurate voxel-space localization. FoSOcc includes a Center Focusing Module (CFM) to enhance instance-aware spatial localization through supervised focus guidance, and a Spherical Lift Module (SLM) that extends perspective lifting to fisheye imaging under the Unified Projection Model. Extensive experiments on Occ3D-Waymo and OccTrack360 show that our method improves occupancy tracking quality with notable gains on geometrically regular categories, and establishes a strong baseline for future research on surround-view fisheye 4D occupancy tracking. The benchmark and source code will be made publicly available at https://github.com/YouthZest-Lin/OccTrack360.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **OccTrack360** 这篇论文的分析如下：

### 1. 论文核心贡献总结
该论文针对自动驾驶中“4D全景占用追踪（4D Panoptic Occupancy Tracking）”领域缺乏环视鱼眼相机基准的现状，提出了 **OccTrack360** 基准数据集，显著提升了序列长度与标注质量。同时，作者提出了 **FoSOcc** 框架，通过专门设计的模块解决了鱼眼相机特有的畸变投影与体素定位精度问题，为环视鱼眼感知的研究提供了强有力的基准与方法论。

### 2. 关键创新点与方法论
*   **数据集构建（OccTrack360）：** 填补了长序列（高达2234帧）和多视角鱼眼感知的空白，并引入了“全向遮挡掩码”和“基于MEI的视场掩码”，为处理遮挡和边界问题提供了精细的监督信号。
*   **FoSOcc 框架：**
    *   **Center Focusing Module (CFM)：** 通过监督式焦点引导，增强了模型对实例级空间定位的感知能力，解决了在复杂场景中实例边界模糊的问题。
    *   **Spherical Lift Module (SLM)：** 基于统一投影模型（Unified Projection Model），将传统的透视投影提升（Perspective Lifting）扩展至鱼眼成像，有效缓解了鱼眼镜头带来的严重几何畸变，实现了更准确的体素空间映射。

### 3. 对领域的潜在影响
*   **推动鱼眼感知研究：** 鱼眼相机因其大视场角（FoV）在自动驾驶中至关重要，但因畸变严重，长期以来在深度学习中难以处理。该论文将研究重心从透视相机转向更具挑战性的鱼眼相机，具有很高的学术价值。
*   **统一表征范式：** 推动了从“目标检测”向“全景占用预测”的范式转移，使得系统能够同时处理静态环境几何与动态实例追踪，这对于实现真正的端到端自动驾驶感知至关重要。

### 4. 相关领域与应用受益
*   **自动驾驶（尤其是泊车与低速场景）：** 鱼眼相机是环视泊车系统的核心，该研究直接提升了车辆在狭窄空间内的动态障碍物追踪能力。
*   **机器人导航：** 移动机器人在复杂室内环境中的避障与路径规划，依赖于对周围环境的连续几何理解，该方法可提供更鲁棒的语义地图构建。
*   **增强现实（AR）与虚拟现实（VR）：** 涉及大视场角全景重建的设备，可以借鉴其处理球形投影和畸变的方法。

### 5. 可推断的局限性
*   **计算开销：** 尽管 FoSOcc 提升了精度，但引入的 CFM 和 SLM 模块可能会增加推理延迟，对于实时性要求极高的车载嵌入式平台，其计算效率仍需进一步验证。
*   **泛化能力：** 论文主要基于特定数据集（如 Occ3D-Waymo），在极端天气（如雨雪、夜间）或传感器标定参数发生漂移时，该方法的鲁棒性可能面临挑战。
*   **长序列累积误差：** 虽然序列变长了，但 4D 追踪中常见的“ID切换”和“轨迹漂移”问题在长序列中依然是难点，论文是否能完全解决长时追踪的稳定性还有待观察。

**专家点评：**
这篇论文的趣味性在于它**直面了工业界最头疼的“鱼眼畸变”问题**，并将其与前沿的“4D占用预测”相结合。它不仅仅是一个数据集的发布，更是一套针对非线性投影成像系统的完整解决方案，对于希望在自动驾驶感知领域寻求差异化研究方向的团队具有极高的参考价值。

**Key Findings:**

- To address this gap, we present OccTrack360, a new benchmark for 4D panoptic occupancy tracking from surround-view fisheye cameras.
- Extensive experiments on Occ3D-Waymo and OccTrack360 show that our method improves occupancy tracking quality with notable gains on geometrically regular categories, and establishes a strong baseline for future research on surround-view fisheye 4D occupancy tracking.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08521v1)
- [arXiv](https://arxiv.org/abs/2603.08521v1)

---

<a id='2603.08519v1'></a>
## [AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models](https://arxiv.org/abs/2603.08519v1)

**Authors:** Xiaoquan Sun, Zetian Xu, Chen Cao, Zonghe Liu, Yihan Sun, Jingrui Pang, Ruijian Zhang, Zhen Yang, Kang Pang, Dingxin He, Mingqi Yuan, Jiayu Chen

**Published:** 2026-03-09

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models demonstrate remarkable potential for generalizable robotic manipulation. The execution of complex multi-step behaviors in VLA models can be improved by robust instruction grounding, a critical component for effective control. However, current paradigms predominantly rely on coarse, high-level task instructions during supervised fine-tuning. This instruction grounding gap leaves models without explicit intermediate guidance, leading to severe compounding errors in long-horizon tasks. Therefore, bridging this instruction gap and providing scalable post-training for VLA models is urgent. To tackle this problem, we propose \method, the first subtask-aware VLA framework integrated with a scalable offline post-training pipeline. Our framework leverages a large language model to decompose high-level demonstrations into fine-grained atomic subtasks. This approach utilizes a pretrained predictive world model to score candidate action chunks against subtask goals in the latent space, mitigating error accumulation while significantly improving long-horizon robustness. Furthermore, this approach enables highly efficient Group Relative Policy Optimization without the prohibitive expenses associated with online rollouts on physical robots. Extensive simulations validate that our AtomVLA maintains strong robustness under perturbations. When evaluated against fundamental baseline models, it achieves an average success rate of 97.0\% on the LIBERO benchmark and 48.0\% on the LIBERO-PRO benchmark. Finally, experiments conducted in the real world using the Galaxea R1 Lite platform confirm its broad applicability across diverse tasks, especially long-horizon tasks. All datasets, checkpoints, and code will be released to the public domain following the acceptance of this work for future research.

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作（VLA）模型在通用机器人操作方面展现出巨大潜力。然而，当前范式在监督微调（SFT）中主要依赖粗粒度的高层任务指令，导致模型缺乏明确的中间引导，在长程任务中极易产生累积误差。为解决这一指令对齐鸿沟，我们提出了AtomVLA，这是首个结合可扩展离线后训练流水线的子任务感知VLA框架。该框架利用大语言模型（LLM）将高层演示分解为细粒度原子子任务，并利用预训练的预测性世界模型在潜在空间中对候选动作块进行评分，从而在无需昂贵在线交互的情况下实现高效的组相对策略优化（GRPO）。实验表明，AtomVLA在LIBERO和LIBERO-PRO基准测试中分别达到97%和48%的成功率，并在真实世界长程任务中展现出极强的鲁棒性。

### 2. 方法动机分析
- **驱动力**：解决VLA模型在长程任务中因缺乏中间步骤引导而导致的“指令对齐鸿沟”及“累积误差”问题。
- **现有痛点**：当前模型多为端到端黑盒，缺乏对任务执行过程的显式规划；且在线强化学习（RL）在物理机器人上成本过高、风险极大。
- **研究假设**：通过LLM将复杂任务分解为原子子任务，并利用潜在空间的世界模型作为“虚拟裁判”进行离线策略优化，可以有效引导模型实现长程鲁棒控制。

### 3. 方法设计详解
- **流程总结**：
  1. **子任务分解（Stage I）**：利用GPT-4o将高层指令和演示视频分解为一系列带有起止帧的原子子任务（如“Pick up [object]”）。
  2. **SFT训练**：将原始指令与子任务指令拼接，训练Qwen3-VL backbone及流匹配（Flow-Matching）动作头。
  3. **离线后训练（Stage II）**：
     - **动作采样**：在给定状态下生成多个候选动作块。
     - **世界模型评分**：利用V-JEPA2作为冻结的视觉编码器，预测动作执行后的潜在状态，计算其与子目标（Subgoal）和最终目标（Final Goal）的距离。
     - **GRPO优化**：基于计算出的奖励，通过组相对策略优化（GRPO）更新动作头，并引入KL散度约束防止偏离SFT基准策略。
- **模型结构**：采用Qwen3-VL作为视觉语言骨干，结合交叉注意力扩散Transformer作为动作头，实现流匹配动作生成。
- **算法解释**：奖励函数 $r^{(k)}$ 综合了子目标距离、最终目标距离和模仿偏差项，确保策略既能完成阶段性任务，又不偏离专家演示的分布。

### 4. 方法对比分析
- **本质区别**：AtomVLA引入了显式的“子任务感知”机制，将长程任务转化为一系列可评估的短程目标，而非依赖单一的端到端映射。
- **创新贡献**：提出了一种基于潜在世界模型的离线RL后训练方案，成功绕过了在线交互的昂贵成本，同时缓解了生成式世界模型常见的像素级幻觉问题。
- **适用场景**：特别适用于需要多步操作、长程规划的复杂机器人操作任务。

### 5. 实验分析
- **验证方法**：在LIBERO和LIBERO-PRO模拟基准及Galaxea R1 Lite真实机器人上进行对比实验。
- **关键结果**：在LIBERO-Long任务上，引入子任务指令后成功率提升显著；后训练阶段在保持视觉表征稳定的前提下，平均成功率提升了4.0%。
- **优势**：极强的长程任务鲁棒性，尤其在处理形变物体（如折叠衣物）时表现优异。
- **局限**：目前依赖LLM预先生成的静态子任务边界，对高度动态、不可预测的环境适应性仍有提升空间。

### 6. 实用指南
- **开源情况**：作者承诺在论文被接收后开源代码、数据集及检查点。
- **实现细节**：关键超参数包括动作块大小（Chunk Size=4）、奖励权重（$\lambda_{sub}=0.3, \lambda_{goal}=0.4, \alpha=0.3$）。训练时需注意保持视觉编码器冻结，仅更新动作头。
- **迁移可能**：该框架的“LLM分解+潜在空间奖励评分”范式可直接迁移至其他机器人操作平台，只需更换对应的视觉编码器和动作空间定义。

### 7. 总结
- **核心思想**：通过LLM分解任务与世界模型离线评分，实现长程任务的鲁棒控制。
- **速记版pipeline**：
  1. LLM将任务拆解为原子步骤。
  2. 联合训练视觉语言模型与动作头。
  3. 利用世界模型预测动作后果。
  4. 通过离线RL优化动作策略。

**Key Findings:**

- To tackle this problem, we propose \method, the first subtask-aware VLA framework integrated with a scalable offline post-training pipeline.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.08519v1)
- [arXiv](https://arxiv.org/abs/2603.08519v1)

---

