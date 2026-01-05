time: 20260105

# Arxiv Computer Vision Papers - 2026-01-05

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份关于2026年1月2日Arxiv计算机视觉领域论文的简明执行摘要。

---

**执行摘要：2026年1月2日 Arxiv 计算机视觉论文精选**

**主要主题与趋势：**

本期Arxiv论文集展现了计算机视觉领域在**动态场景理解与重建、多模态融合、具身智能以及生成模型**等方面的显著进展。特别值得注意的是，**3D表示与生成**（如动态3D高斯）以及**利用大型语言模型（LLMs）增强视觉任务**成为突出趋势。此外，**零样本学习和通用性奖励模型**的探索也预示着模型能力的进一步提升。

**亮点与创新：**

*   **Pixel-to-4D: Camera-Controlled Image-to-Video Generation with Dynamic 3D Gaussians** 是一项突破性工作，它将静态图像转化为动态视频，并引入了**动态3D高斯**的概念，为高质量、可控的视频生成开辟了新途径。
*   **RoboReward: General-Purpose Vision-Language Reward Models for Robotics** 提出了**通用的视觉-语言奖励模型**，旨在解决机器人领域中奖励函数设计的挑战，有望加速机器人学习的通用性和效率。
*   **AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction** 在动态场景重建方面提出了**自适应Gabor表示**，为处理复杂动态场景提供了新的视角。

**新兴研究方向与技术：**

*   **动态3D表示与生成：** 以Pixel-to-4D为代表，动态3D高斯等技术正在成为生成逼真动态场景的关键。
*   **多模态LLMs在视觉任务中的应用：** 从工程考试评分到具身感知，LLMs正被广泛集成以提升视觉任务的理解和执行能力。
*   **具身智能与感知：** RoboReward和Modality Dominance-Aware Optimization等论文表明，对机器人和具身智能的感知能力（尤其是在多模态环境下）的研究正在深化。
*   **零样本与泛化能力：** GranAlign在视频检索中的零样本能力探索，以及RoboReward的通用性目标，都指向了模型泛化能力的提升。
*   **AI生成内容检测：** A Comprehensive Dataset for Human vs. AI Generated Image Detection 的出现，标志着对AI生成内容真实性验证的需求日益增长。

**建议阅读论文：**

为了快速了解当前研究热点和前沿技术，建议优先阅读以下论文：

1.  **Pixel-to-4D: Camera-Controlled Image-to-Video Generation with Dynamic 3D Gaussians** (对于视频生成和3D表示的最新进展)
2.  **RoboReward: General-Purpose Vision-Language Reward Models for Robotics** (对于具身智能和多模态奖励模型的研究)
3.  **AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction** (对于动态场景理解和重建的创新方法)
4.  **Unified Primitive Proxies for Structured Shape Completion** (对于3D形状理解和补全的结构化方法)

---

希望这份摘要能帮助您快速掌握近期Arxiv计算机视觉领域的最新动态。

---

## Table of Contents

1. [AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction](#2601.00796v1)
2. [Unified Primitive Proxies for Structured Shape Completion](#2601.00759v1)
3. [Grading Handwritten Engineering Exams with Multimodal Large Language Models](#2601.00730v1)
4. [Pixel-to-4D: Camera-Controlled Image-to-Video Generation with Dynamic 3D Gaussians](#2601.00678v1)
5. [RoboReward: General-Purpose Vision-Language Reward Models for Robotics](#2601.00675v1)
6. [Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation](#2601.00664v1)
7. [Modality Dominance-Aware Optimization for Embodied RGB-Infrared Perception](#2601.00598v1)
8. [GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](#2601.00584v1)
9. [AEGIS: Exploring the Limit of World Knowledge Capabilities for Unified Mulitmodal Models](#2601.00561v1)
10. [A Comprehensive Dataset for Human vs. AI Generated Image Detection](#2601.00553v1)

---

## Papers

<a id='2601.00796v1'></a>
## [AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction](https://arxiv.org/abs/2601.00796v1)

**Authors:** Jiewen Chan, Zhenjun Zhao, Yu-Lun Liu

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

Reconstructing dynamic 3D scenes from monocular videos requires simultaneously capturing high-frequency appearance details and temporally continuous motion. Existing methods using single Gaussian primitives are limited by their low-pass filtering nature, while standard Gabor functions introduce energy instability. Moreover, lack of temporal continuity constraints often leads to motion artifacts during interpolation. We propose AdaGaR, a unified framework addressing both frequency adaptivity and temporal continuity in explicit dynamic scene modeling. We introduce Adaptive Gabor Representation, extending Gaussians through learnable frequency weights and adaptive energy compensation to balance detail capture and stability. For temporal continuity, we employ Cubic Hermite Splines with Temporal Curvature Regularization to ensure smooth motion evolution. An Adaptive Initialization mechanism combining depth estimation, point tracking, and foreground masks establishes stable point cloud distributions in early training. Experiments on Tap-Vid DAVIS demonstrate state-of-the-art performance (PSNR 35.49, SSIM 0.9433, LPIPS 0.0723) and strong generalization across frame interpolation, depth consistency, video editing, and stereo view synthesis. Project page: https://jiewenchan.github.io/AdaGaR/

**Analysis:**

好的，这是对论文“AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction”的全面中文摘要：

**论文题目：** AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction

**作者：** Jiewen Chan, Zhenjun Zhao, Yu-Lun Liu

**摘要：**

**1. 研究问题/核心挑战：**
该论文旨在解决从单目视频中重建动态三维场景的核心挑战。这需要同时捕捉高频的外观细节和时间上连续的运动。现有方法在处理这两个方面时都存在局限性：
*   **高频细节捕捉不足：** 基于高斯原语的方法具有低通滤波的特性，难以保留精细的纹理细节。而标准 Gabor 函数虽然能增强高频细节，但容易引入能量不稳定性。
*   **时间连续性不足：** 缺乏严格的时间连续性约束，容易在插值过程中产生运动伪影，导致几何和运动的不连续性。

**2. 主要创新点/方法贡献：**
为了克服这些挑战，作者提出了 AdaGaR（Adaptive Gabor Representation for Dynamic Scene Reconstruction）框架，这是一个统一的解决方案，同时解决了频率适应性和时间连续性问题：

*   **自适应 Gabor 表示 (Adaptive Gabor Representation)：**
    *   **频率适应性：** 扩展了传统的高斯原语，引入了可学习的频率权重 (wi)，使其能够自适应地在低频（高斯）和高频（Gabor）之间切换。这使得模型能够根据场景需求捕捉不同频率的细节。
    *   **能量稳定性：** 通过引入自适应能量补偿机制，确保了 Gabor 表示的能量稳定性，避免了因频率调制带来的不稳定性。
    *   **平滑过渡：** 通过一个补偿项 `b`，使得模型能够平滑地从纯高斯（低频）过渡到 Gabor（高频）模式，并在不需要高频细节时自然退化为标准高斯。

*   **时间连续性约束：**
    *   **三次 Hermite 样条插值 (Cubic Hermite Splines)：** 用于插值动态原语的时间演化，确保了平滑的运动轨迹。
    *   **时间曲率正则化 (Temporal Curvature Regularization)：** 通过约束轨迹的二阶导数，进一步保证了运动的平滑性和几何连续性，有效避免了插值伪影。

*   **自适应初始化 (Adaptive Initialization)：**
    *   **多模态融合：** 结合了深度估计、点追踪和前景掩码等多种信息，在训练早期生成密集且时间上一致的点云分布，为后续的显式表示奠定稳固的几何基础。
    *   **自适应采样：** 根据场景的运动和深度分布动态调整采样密度，实现了更均衡的前景/背景覆盖，减少了早期闪烁。

**3. 主要结果与意义：**
*   **卓越的性能：** 在 Tap-Vid DAVIS 数据集上取得了最先进的性能，PSNR 达到 35.49 dB，SSIM 达到 0.9433，LPIPS 达到 0.0723。与第二名相比，PSNR 提高了 6.86 dB。
*   **强大的泛化能力：** AdaGaR 在帧插值、深度一致性、视频编辑和立体视图合成等下游应用中展现出强大的泛化能力。
*   **细节保留与运动平滑：** 实验结果表明，AdaGaR 能够有效地捕捉精细的纹理细节（如毛发、车窗边缘）并保持时间上的运动连续性，尤其是在具有挑战性的场景（如快速运动、遮挡和复杂形变）下。
*   **统一的框架：** 提供了一个紧凑、端到端的解决方案，能够同时建模显式动态表示中的时间和频率信息。

**4. 提及的局限性：**
*   **非线性运动的挑战：** 基于样条的运动建模在处理突变或高度非线性运动时可能存在对齐问题。
*   **高频区域的潜在振荡：** 自适应 Gabor 表示在高频区域可能由于能量约束而出现振荡。

**5. 未来研究方向：**
*   **自适应时间控制点：** 探索更灵活的自适应时间控制点，以更好地处理非线性运动。
*   **运动感知频率调制：** 开发运动感知的频率调制方法，以进一步优化高频细节的表示。

总而言之，AdaGaR 提出了一种新颖的自适应 Gabor 表示方法，并结合了三次 Hermite 样条和时间曲率正则化，成功地解决了单目视频动态场景重建中的高频细节捕捉和时间连续性两大难题。该方法在多个评估指标上取得了显著的领先，并展现了在多种下游任务中的强大应用潜力。

**Key Findings:**

- We propose AdaGaR, a unified framework addressing both frequency adaptivity and temporal continuity in explicit dynamic scene modeling.
- We introduce Adaptive Gabor Representation, extending Gaussians through learnable frequency weights and adaptive energy compensation to balance detail capture and stability.
- Experiments on Tap-Vid DAVIS demonstrate state-of-the-art performance (PSNR 35.49, SSIM 0.9433, LPIPS 0.0723) and strong generalization across frame interpolation, depth consistency, video editing, and stereo view synthesis.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00796v1)
- [arXiv](https://arxiv.org/abs/2601.00796v1)

---

<a id='2601.00759v1'></a>
## [Unified Primitive Proxies for Structured Shape Completion](https://arxiv.org/abs/2601.00759v1)

**Authors:** Zhaiyu Chen, Yuqing Wang, Xiao Xiang Zhu

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

Structured shape completion recovers missing geometry as primitives rather than as unstructured points, which enables primitive-based surface reconstruction. Instead of following the prevailing cascade, we rethink how primitives and points should interact, and find it more effective to decode primitives in a dedicated pathway that attends to shared shape features. Following this principle, we present UniCo, which in a single feed-forward pass predicts a set of primitives with complete geometry, semantics, and inlier membership. To drive this unified representation, we introduce primitive proxies, learnable queries that are contextualized to produce assembly-ready outputs. To ensure consistent optimization, our training strategy couples primitives and points with online target updates. Across synthetic and real-world benchmarks with four independent assembly solvers, UniCo consistently outperforms recent baselines, lowering Chamfer distance by up to 50% and improving normal consistency by up to 7%. These results establish an attractive recipe for structured 3D understanding from incomplete data. Project page: https://unico-completion.github.io.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Unified Primitive Proxies for Structured Shape Completion**

**1. 论文的主要贡献（2-3句话）：**

该论文提出了一种名为 UniCo 的新颖方法，用于结构化形状补全。与传统的级联方法不同，UniCo 在单一前馈通道中统一预测具有完整几何、语义和内点成员资格的原始体集合。其核心在于引入了“原始体代理”（primitive proxies），这些可学习的查询能够根据上下文生成可组装的输出，从而实现更有效的结构化形状理解。

**2. 关键创新或方法论：**

*   **统一的原始体预测通道：** UniCo 的核心创新在于打破了传统方法中点云和原始体之间分离处理的级联模式。它设计了一个专门的通道来解码原始体，该通道能够关注共享的形状特征，从而实现更紧密的点与原始体之间的交互。
*   **原始体代理（Primitive Proxies）：** 这是论文的另一项关键创新。原始体代理被设计为可学习的查询，它们能够被上下文信息“情境化”（contextualized），从而直接生成“组装就绪”（assembly-ready）的输出。这意味着代理不仅仅是占位符，而是能够根据输入数据动态地生成具有完整几何、语义和内点成员资格的原始体。
*   **耦合训练策略与在线目标更新：** 为了确保优化的一致性，论文采用了将原始体和点云耦合在一起的训练策略，并结合了在线目标更新机制。这有助于在训练过程中保持点云和预测的原始体之间的良好对齐和一致性。

**3. 对该领域的潜在影响：**

*   **提升结构化3D理解的效率和准确性：** UniCo 的统一预测范式有望显著提高结构化形状补全的效率，因为它避免了多阶段的级联处理。同时，通过更紧密的点与原始体交互以及原始体代理的引入，有望提升补全的准确性，尤其是在几何和语义方面。
*   **推动基于原始体的3D表面重建：** 论文明确指出，其方法能够生成用于原始体基表面重建的输出。这意味着 UniCo 可以为下游的3D重建任务提供更优质、更结构化的输入，从而推动该领域的发展。
*   **为不完整数据下的3D理解提供新范式：** 该研究为如何从不完整数据中理解3D形状提供了一个新的、更具吸引力的“配方”。它表明，通过精心设计的统一表示和学习机制，可以更有效地提取和利用结构化信息。
*   **潜在的通用性：** 论文在合成和真实世界基准上进行了测试，并与多种独立的组装求解器进行了比较，这表明 UniCo 的方法具有一定的通用性，能够适应不同的下游任务和评估标准。

**4. 可能受益的相关领域或应用：**

*   **3D模型检索与识别：** 结构化形状补全可以为不完整或损坏的3D模型提供更完整的表示，从而提高检索和识别的准确性。
*   **机器人感知与导航：** 机器人需要准确理解周围环境的3D结构，结构化形状补全可以帮助机器人从传感器数据中恢复出更完整的场景几何信息。
*   **虚拟现实/增强现实（VR/AR）：** 在 VR/AR 应用中，需要高质量的3D模型来构建沉浸式体验。UniCo 的方法可以用于生成更逼真、更完整的虚拟对象。
*   **计算机辅助设计（CAD）：** 在设计过程中，可能需要对不完整的模型进行补全和修复，UniCo 的方法可以为设计师提供更智能的工具。
*   **3D打印：** 结构化形状补全可以帮助修复3D模型中的缺陷，使其更适合3D打印。
*   **医学影像分析：** 在医学影像中，可能存在部分缺失的器官或结构，结构化形状补全可以帮助重建完整的解剖结构。

**5. 从摘要中可以推断出的局限性：**

*   **对原始体类型的依赖性：** 尽管摘要没有明确说明，但“原始体”（primitives）的定义和预设类型对方法的性能至关重要。如果输入的形状主要由摘要中未考虑的复杂或非标准原始体构成，方法的性能可能会受到影响。
*   **“组装就绪”的定义和挑战：** “组装就绪”是一个相对的概念。虽然论文声称原始体代理能够生成这样的输出，但其在实际组装过程中的鲁棒性和通用性仍需进一步验证。不同的组装求解器可能对输出的格式和精度有不同的要求。
*   **计算复杂度：** 虽然是单一前馈通道，但“情境化”原始体代理以及处理共享形状特征可能仍然需要一定的计算资源，尤其是在处理高分辨率或非常复杂的场景时。
*   **对训练数据的需求：** 任何机器学习方法都依赖于训练数据。UniCo 的性能将很大程度上取决于其训练数据的质量和多样性，特别是包含各种不完整形状和对应完整结构化表示的数据集。
*   **泛化到极端不完整情况的挑战：** 摘要提到“不完整数据”，但对于数据缺失程度非常严重的情况，即使是 UniCo 这种先进的方法，也可能面临挑战，其补全的准确性和可靠性可能会下降。

总而言之，这篇论文提出的 UniCo 方法在结构化形状补全领域具有重要的理论和实践意义。其核心创新在于统一的原始体预测和创新的原始体代理机制，有望在效率和准确性上取得显著突破，并为3D理解的多个下游应用带来积极影响。

**Key Findings:**

- Following this principle, we present UniCo, which in a single feed-forward pass predicts a set of primitives with complete geometry, semantics, and inlier membership.
- To drive this unified representation, we introduce primitive proxies, learnable queries that are contextualized to produce assembly-ready outputs.
- Across synthetic and real-world benchmarks with four independent assembly solvers, UniCo consistently outperforms recent baselines, lowering Chamfer distance by up to 50% and improving normal consistency by up to 7%.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00759v1)
- [arXiv](https://arxiv.org/abs/2601.00759v1)

---

<a id='2601.00730v1'></a>
## [Grading Handwritten Engineering Exams with Multimodal Large Language Models](https://arxiv.org/abs/2601.00730v1)

**Authors:** Janez Perš, Jon Muhovič, Andrej Košir, Boštjan Murovec

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

Handwritten STEM exams capture open-ended reasoning and diagrams, but manual grading is slow and difficult to scale. We present an end-to-end workflow for grading scanned handwritten engineering quizzes with multimodal large language models (LLMs) that preserves the standard exam process (A4 paper, unconstrained student handwriting). The lecturer provides only a handwritten reference solution (100%) and a short set of grading rules; the reference is converted into a text-only summary that conditions grading without exposing the reference scan. Reliability is achieved through a multi-stage design with a format/presence check to prevent grading blank answers, an ensemble of independent graders, supervisor aggregation, and rigid templates with deterministic validation to produce auditable, machine-parseable reports. We evaluate the frozen pipeline in a clean-room protocol on a held-out real course quiz in Slovenian, including hand-drawn circuit schematics. With state-of-the-art backends (GPT-5.2 and Gemini-3 Pro), the full pipeline achieves $\approx$8-point mean absolute difference to lecturer grades with low bias and an estimated manual-review trigger rate of $\approx$17% at $D_{\max}=40$. Ablations show that trivial prompting and removing the reference solution substantially degrade accuracy and introduce systematic over-grading, confirming that structured prompting and reference grounding are essential.

**Analysis:**

好的，这是一篇关于使用多模态大语言模型（LLMs）对工程考试进行自动评分的论文的全面中文摘要，重点关注其在计算机视觉领域的创新性和重要性。

**论文题目：** Grading Handwritten Engineering Exams with Multimodal Large Language Models (使用多模态大语言模型对工程考试进行评分)

**作者：** Janez Perš, Jon Muhovič, Andrej Košir, Boštjan Murovec

---

**全面摘要**

**1. 研究问题/核心挑战：**
手写工程考试（包含开放式推理和图表）在STEM教育中仍然普遍存在，但其手动评分过程耗时且难以规模化。传统方法要么要求学生将答案转换为机器可读格式，要么依赖于易出错的手写识别（OCR）技术，并且往往无法端到端地处理手写文本和图表。本研究旨在解决这一问题，提出一种能够直接处理扫描的手写工程考试（包括图表）的自动化评分流程，同时保留标准的考试形式（A4纸，无约束手写）。

**2. 关键创新/方法论贡献：**
该论文的核心贡献在于提出了一种**端到端的、多阶段的、多模态大语言模型（LLM）驱动的评分工作流程**，其创新性体现在以下几个方面：

*   **保留标准考试流程：** 流程设计旨在最小化对现有考试流程的干扰，学生只需提供标准的A4纸手写答案。
*   **参考条件化（Reference Conditioning）：** 讲师仅需提供一份手写参考答案（100%正确）和一套评分规则。参考答案被转换为文本摘要，用于指导评分，但原始参考图像本身不直接暴露给LLM，这在一定程度上解决了隐私问题。
*   **多阶段设计与鲁棒性：**
    *   **格式/存在性检查（Format/Presence Check）：** 在评分前，通过一个检查器来识别包含实际学生答案的任务，以防止对空白答案进行评分。
    *   **独立评分器集成（Ensemble of Independent Graders）：** 每个任务由多个独立的LLM调用进行评分，生成结构化的草稿。
    *   **监督者聚合（Supervisor Aggregation）：** 一个监督模型负责合并多个评分器的草稿，形成最终的考试级别输出，并强制执行模板合规性。
    *   **刚性模板与确定性验证（Rigid Templates & Deterministic Validation）：** 使用预定义的Markdown模板来规范LLM的输出格式，确保输出是机器可解析的，从而实现可审计性和一致性。
*   **零样本（Zero-shot）评估：** 该系统是不可训练的，完全在零样本设置下运行，避免了迭代式提示工程和评分标准的偏差。
*   **语言无关性：** 尽管实验在斯洛文尼亚语环境下进行，但该流程本身是语言无关的，只需翻译文本工件即可适应其他语言。

**3. 主要结果及其意义：**
在对一个真实的、包含电路图的斯洛文尼亚语工程考试（Class B）进行评估时，使用最先进的LLM后端（GPT-5.2和Gemini-3 Pro），该完整工作流程取得了以下成果：

*   **高准确性：** 平均绝对差（MAD）约为8个点，与讲师评分的偏差很小（低偏差）。
*   **可接受的手动复核率：** 在Dmax=40时，估计的手动复核触发率为约17%，表明系统在大多数情况下能够可靠评分，仅需少量人工干预。
*   **消融实验的重要性：** 消融实验（Ablation studies）表明，简单的提示（trivial prompting）和移除参考答案显著降低了准确性，并引入了系统性的过度评分（over-grading）。这有力地证实了结构化提示和参考条件化对于实现可靠评分至关重要。
*   **模型能力筛选：** 实验还对不同的LLM后端进行了基线性能评估，表明GPT-5.2、GPT-5.2-pro和Gemini-3 Pro在MAD和偏差方面表现最佳。

这些结果表明，通过精心设计的系统级工作流程，现代多模态LLM已经能够实现对包含手写文本和图表的工程考试进行接近人类评分者水平的自动评分，这对于解决STEM教育中评分的规模化和效率问题具有重要意义。

**4. 提及的局限性：**
*   **数据集限制：** 论文使用了作者自己收集的、私有的课程考试数据进行评估，因为公开可用的、支持端到端评估的手写考试评分数据集非常稀少。
*   **评估规模：** 评估仅基于一个“干净房间”（clean-room）协议下的一个实际课程测验，且仅有一个人类评分者作为地面真实（ground truth）。
*   **隐私问题：** 虽然参考答案被转换为文本摘要，但原始扫描件的隐私处理仍需谨慎。
*   **学生反馈：** 学生反馈显示，虽然大多数学生对AI评分持积极态度，但也有关于漏判答案、评分错误和考试难度变化的担忧。

**5. 潜在的未来研究方向：**
*   **扩展验证：** 在更大、更多样化的数据集上进行验证，并计划在隐私审查和机构批准后发布代码、提示和数据集，以支持标准化评估。
*   **更细粒度的评估：** 在实际部署中，可以将手动复核标准应用于更细粒度的层面，如单个问题或答案，以进一步精确定位和减少人工干预。
*   **多模态理解的深入：** 进一步探索LLM在理解和评估复杂图表（如电路图）方面的能力，以及如何更有效地将其与文本推理结合。
*   **用户体验优化：** 结合学生反馈，进一步优化评分反馈的质量和形式，解决学生担忧的问题。
*   **跨语言和跨学科应用：** 验证和扩展该框架在不同语言和不同STEM学科中的适用性。

**对计算机视觉领域的意义：**

这篇论文对计算机视觉领域具有重要意义，因为它展示了**多模态LLM在理解和处理包含视觉信息（手写文本和图表）的文档方面的强大能力**。它不仅将LLM的应用从纯文本领域扩展到了包含复杂视觉元素的场景，而且通过**“参考条件化”**和**“刚性模板”**等方法，有效地解决了LLM输出不确定性和不可靠性的问题，使其能够用于需要高精度和可审计性的实际应用场景，如教育评估。这为未来计算机视觉与自然语言处理的深度融合，以及在教育、文档分析等领域的创新应用提供了新的思路和实践范例。特别是，它证明了**视觉信息（图表）与文本推理的结合，对于理解和评估工程问题至关重要**，这正是多模态模型的核心优势所在。

**Key Findings:**

- We present an end-to-end workflow for grading scanned handwritten engineering quizzes with multimodal large language models (LLMs) that preserves the standard exam process (A4 paper, unconstrained student handwriting).
- With state-of-the-art backends (GPT-5.2 and Gemini-3 Pro), the full pipeline achieves $\approx$8-point mean absolute difference to lecturer grades with low bias and an estimated manual-review trigger rate of $\approx$17% at $D_{\max}=40$.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00730v1)
- [arXiv](https://arxiv.org/abs/2601.00730v1)

---

<a id='2601.00678v1'></a>
## [Pixel-to-4D: Camera-Controlled Image-to-Video Generation with Dynamic 3D Gaussians](https://arxiv.org/abs/2601.00678v1)

**Authors:** Melonie de Almeida, Daniela Ivanova, Tong Shi, John H. Williamson, Paul Henderson

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

Humans excel at forecasting the future dynamics of a scene given just a single image. Video generation models that can mimic this ability are an essential component for intelligent systems. Recent approaches have improved temporal coherence and 3D consistency in single-image-conditioned video generation. However, these methods often lack robust user controllability, such as modifying the camera path, limiting their applicability in real-world applications. Most existing camera-controlled image-to-video models struggle with accurately modeling camera motion, maintaining temporal consistency, and preserving geometric integrity. Leveraging explicit intermediate 3D representations offers a promising solution by enabling coherent video generation aligned with a given camera trajectory. Although these methods often use 3D point clouds to render scenes and introduce object motion in a later stage, this two-step process still falls short in achieving full temporal consistency, despite allowing precise control over camera movement. We propose a novel framework that constructs a 3D Gaussian scene representation and samples plausible object motion, given a single image in a single forward pass. This enables fast, camera-guided video generation without the need for iterative denoising to inject object motion into render frames. Extensive experiments on the KITTI, Waymo, RealEstate10K and DL3DV-10K datasets demonstrate that our method achieves state-of-the-art video quality and inference efficiency. The project page is available at https://melonienimasha.github.io/Pixel-to-4D-Website.

**Analysis:**

好的，这是一篇关于“Pixel-to-4D: Camera-Controlled Image-to-Video Generation with Dynamic 3D Gaussians”论文的中文摘要，其中包含了您要求的各个部分：

**论文题目：** Pixel-to-4D: Camera-Controlled Image-to-Video Generation with Dynamic 3D Gaussians

**作者：** Melonie de Almeida, Daniela Ivanova, Tong Shi, John H. Williamson, Paul Henderson

**摘要：**

**1. 研究问题/核心挑战：**
该论文主要解决了从单张图像生成具有精确相机控制和动态物体运动的视频这一核心挑战。现有方法在模仿人类预测场景未来动态的能力方面取得了进展，但普遍存在用户可控性不足（如相机路径修改困难）、相机运动建模不准确、时间一致性差以及几何完整性难以保证等问题。尤其是在需要精确相机控制的实际应用中，这些限制尤为突出。

**2. 主要创新点/方法贡献：**
作者提出了一个名为 **Pixel-to-4D** 的新颖框架，其核心创新在于：
*   **动态3D高斯表示 (Dynamic 3D Gaussian Representation)：** 引入了一种新的4D场景表示，该表示由像素对齐的静态和动态高斯参数组成，能够捕捉多层级的场景信息，并为每个高斯点赋予线速度、角速度及其加速度，从而实现动态场景的建模。
*   **单次前向传播的生成架构：** 设计了一个高效的前馈神经网络架构，能够从单张输入图像一次性生成上述4D高斯表示。该架构利用了预训练的DINOv2特征，并融合了静态和动态高斯参数的预测。
*   **无需迭代去噪的快速生成：** 与依赖迭代去噪来注入物体运动的方法不同，Pixel-to-4D直接在4D表示层面处理物体运动，从而实现了快速、相机引导的视频生成，无需复杂的后处理步骤。
*   **基于3D高斯喷绘 (3D Gaussian Splatting) 的渲染：** 利用了3D高斯喷绘技术，这是一种先进的3D重建方法，能够高效地渲染场景，并能通过自适应地调整高斯尺度来填补点云的空隙，从而生成高质量的视频帧。

**3. 主要结果与意义：**
*   **state-of-the-art 性能：** 在KITTI、Waymo Open、RealEstate10K和DL3DV-10K等多个大规模数据集上进行了广泛实验，结果表明Pixel-to-4D在PSNR、LPIPS、SSIM和FVD等指标上均优于现有的相机控制图像到视频生成模型，同时实现了更快的推理速度。
*   **卓越的相机控制和视觉质量：** 该方法能够生成具有高度时间一致性、精确相机运动跟踪以及逼真物体动态的视频，有效解决了现有方法的痛点。
*   **高效性：** 单次前向传播的生成方式大大降低了推理成本，使其更适用于实际应用。
*   **意义：** 该研究为实现更具可控性、真实感和效率的图像到视频生成提供了新的解决方案，尤其是在需要精确相机控制的自动驾驶、虚拟现实等领域具有重要应用潜力。

**4. 提及的局限性：**
*   **物体运动的不确定性：** 论文提到，给定单张图像，物体运动的参数本身就存在不确定性，因此需要从学习到的条件分布中进行采样，而不是直接回归。
*   **对预训练模型的依赖：** 尽管利用DINOv2等预训练模型带来了优势，但也意味着模型在一定程度上依赖于这些模型的泛化能力。
*   **对特定场景的适应性：** 虽然在多个数据集上进行了评估，但对于极端复杂或完全未见过的新场景，其性能仍可能受到影响。

**5. 潜在的未来研究方向：**
*   **更精细的物体运动建模：** 进一步探索更复杂的物体运动模式，例如非刚性形变或更精细的交互行为。
*   **更强的语义理解：** 结合更深层次的语义理解，以实现更智能的场景动态预测和内容生成。
*   **交互式编辑和控制：** 进一步增强用户交互能力，允许用户对生成的视频进行更细粒度的编辑和控制。
*   **跨模态生成：** 将该方法扩展到结合文本或其他模态进行更丰富的视频生成。
*   **实时生成：** 进一步优化模型以实现更高帧率的实时视频生成。

**Key Findings:**

- We propose a novel framework that constructs a 3D Gaussian scene representation and samples plausible object motion, given a single image in a single forward pass.
- Extensive experiments on the KITTI, Waymo, RealEstate10K and DL3DV-10K datasets demonstrate that our method achieves state-of-the-art video quality and inference efficiency.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00678v1)
- [arXiv](https://arxiv.org/abs/2601.00678v1)

---

<a id='2601.00675v1'></a>
## [RoboReward: General-Purpose Vision-Language Reward Models for Robotics](https://arxiv.org/abs/2601.00675v1)

**Authors:** Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, Chelsea Finn

**Published:** 2026-01-02

**Categories:** cs.RO

**Abstract:**

A well-designed reward is critical for effective reinforcement learning-based policy improvement. In real-world robotic domains, obtaining such rewards typically requires either labor-intensive human labeling or brittle, handcrafted objectives. Vision-language models (VLMs) have shown promise as automatic reward models, yet their effectiveness on real robot tasks is poorly understood. In this work, we aim to close this gap by introducing (1) \textbf{RoboReward}, a robotics reward dataset and benchmark built on large-scale real-robot corpora from Open X-Embodiment (OXE) and RoboArena, and (2) vision-language reward models trained on this dataset (RoboReward 4B/8B). Because OXE is success-heavy and lacks failure examples, we propose a \emph{negative examples data augmentation} pipeline that generates calibrated \emph{negatives} and \emph{near-misses} via counterfactual relabeling of successful episodes and temporal clipping to create partial-progress outcomes from the same videos. Using this framework, we produce an extensive training and evaluation dataset that spans diverse tasks and embodiments and enables systematic evaluation of whether state-of-the-art VLMs can reliably provide rewards for robotics. Our evaluation of leading open-weight and proprietary VLMs reveals that no model excels across all tasks, underscoring substantial room for improvement. We then train general-purpose 4B- and 8B-parameter models that outperform much larger VLMs in assigning rewards for short-horizon robotic tasks. Finally, we deploy the 8B-parameter reward VLM in real-robot reinforcement learning and find that it improves policy learning over Gemini Robotics-ER 1.5, a frontier physical reasoning VLM trained on robotics data, by a large margin, while substantially narrowing the gap to RL training with human-provided rewards.

**Analysis:**

好的，这是对论文“RoboReward: General-Purpose Vision-Language Reward Models for Robotics”的全面中文摘要：

**论文题目：** RoboReward: 通用视觉语言奖励模型用于机器人学

**作者：** Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, Chelsea Finn

**摘要：**

**1. 研究问题/核心挑战：**
在现实世界的机器人学中，强化学习（RL）策略的有效改进严重依赖于准确且信息丰富的奖励模型。然而，获取这些奖励通常需要耗费人力的人工标注或脆弱的手工设计奖励函数。尽管视觉语言模型（VLMs）在自动奖励建模方面展现出潜力，但它们在真实机器人任务上的有效性仍未得到充分理解。本研究旨在解决这一关键瓶颈，即如何为机器人学开发通用且可靠的奖励模型。

**2. 主要创新与方法贡献：**
*   **RoboReward 数据集与基准：** 作者引入了一个名为 RoboReward 的大规模机器人奖励数据集和基准。该数据集构建于 Open X-Embodiment (OXE) 和 RoboArena 的真实机器人语料库之上，涵盖了多样化的任务和机器人实体。
*   **负样本数据增强管道：** 针对现有数据集（如 OXE）主要包含成功案例而缺乏失败案例的问题，作者提出了一种创新的负样本数据增强管道。该管道通过**反事实重标注**（即在同一视频中，通过改变指令来生成部分成功或失败的场景）和**时间剪辑**（将成功视频截断以创建部分进展的片段）来生成校准过的负样本和近乎成功的样本。
*   **通用奖励模型训练：** 基于 RoboReward 数据集，作者训练了两个通用型视觉语言奖励模型：RoboReward 4B 和 RoboReward 8B。
*   **全面的 VLM 评估：** 作者构建了一个名为 RoboRewardBench 的标准化评估基准，用于评估 22 种（包括开源和闭源）先进 VLM 作为机器人奖励模型的能力。

**3. 主要研究结果与意义：**
*   **现有 VLM 的局限性：** 评估结果表明，当前的通用 VLM 在机器人控制的各个场景下都无法可靠地提供奖励，存在显著的泛化能力差距。
*   **RoboReward 模型的优越性：** 训练出的 RoboReward 4B 和 8B 模型在 RoboRewardBench 上表现出色，优于许多参数量更大的现有 VLM，尤其是在短时机器人任务的奖励分配方面。
*   **真实机器人 RL 的改进：** 将 RoboReward 8B 模型部署到真实机器人强化学习任务中，显著提高了策略学习的性能，大幅缩小了与人工奖励的差距，并远超了专门为机器人设计的 Gemini Robotics-ER 1.5 模型。
*   **奖励质量与 RL 性能的相关性：** 研究证实，奖励模型的准确性与下游 RL 策略的性能之间存在强烈的正相关关系，强调了高质量奖励模型的重要性。
*   **数据集的价值：** RoboReward 数据集和评估套件的发布，为机器人学领域通用奖励模型的开发和评估提供了宝贵的资源。

**4. 论文中提到的局限性：**
*   **泛化能力差距：** 尽管 RoboReward 模型表现优异，但评估显示，即使是先进的 VLM，在不同机器人实体、场景和视角下仍存在显著的泛化能力差距。
*   **奖励的细微差别：** 论文提到，即使是最好的模型，也可能在细微的空间和时间细节上出错（例如，未能准确判断物体是否真正放置到位），这可能导致奖励分配的误差。
*   **长时序任务的挑战：** 作者指出，将奖励建模扩展到更长时序、多阶段的任务将更具挑战性，因为信用分配和进度估计会变得更加困难。

**5. 未来研究方向：**
*   **长时序和多阶段任务：** 将奖励建模扩展到更复杂的、需要长期规划和信用分配的任务。
*   **更精细的奖励信号：** 探索更精细的奖励信号，以捕捉机器人任务中更微妙的成功和失败。
*   **提高泛化能力：** 进一步研究如何提高 VLM 在不同机器人实体、场景和视角下的泛化能力。
*   **实时奖励生成：** 探索更高效的奖励生成方法，使其能够集成到实时的 RL 训练循环中。

**总结：**
这篇论文在机器人学领域做出了重要贡献，通过引入 RoboReward 数据集、创新的数据增强方法以及高性能的通用视觉语言奖励模型，显著推动了机器人强化学习的发展。研究结果表明，高质量的奖励模型是实现高效机器人 RL 的关键，并为未来开发更强大、更通用的机器人智能指明了方向。

**Key Findings:**

- Because OXE is success-heavy and lacks failure examples, we propose a \emph{negative examples data augmentation} pipeline that generates calibrated \emph{negatives} and \emph{near-misses} via counterfactual relabeling of successful episodes and temporal clipping to create partial-progress outcomes from the same videos.
- Using this framework, we produce an extensive training and evaluation dataset that spans diverse tasks and embodiments and enables systematic evaluation of whether state-of-the-art VLMs can reliably provide rewards for robotics.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00675v1)
- [arXiv](https://arxiv.org/abs/2601.00675v1)

---

<a id='2601.00664v1'></a>
## [Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation](https://arxiv.org/abs/2601.00664v1)

**Authors:** Taekyung Ki, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon, Sung Ju Hwang

**Published:** 2026-01-02

**Categories:** cs.LG, cs.AI, cs.CV, cs.HC, cs.MM

**Abstract:**

Talking head generation creates lifelike avatars from static portraits for virtual communication and content creation. However, current models do not yet convey the feeling of truly interactive communication, often generating one-way responses that lack emotional engagement. We identify two key challenges toward truly interactive avatars: generating motion in real-time under causal constraints and learning expressive, vibrant reactions without additional labeled data. To address these challenges, we propose Avatar Forcing, a new framework for interactive head avatar generation that models real-time user-avatar interactions through diffusion forcing. This design allows the avatar to process real-time multimodal inputs, including the user's audio and motion, with low latency for instant reactions to both verbal and non-verbal cues such as speech, nods, and laughter. Furthermore, we introduce a direct preference optimization method that leverages synthetic losing samples constructed by dropping user conditions, enabling label-free learning of expressive interaction. Experimental results demonstrate that our framework enables real-time interaction with low latency (approximately 500ms), achieving 6.8X speedup compared to the baseline, and produces reactive and expressive avatar motion, which is preferred over 80% against the baseline.

**Analysis:**

好的，这是对论文“Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation”的全面中文摘要：

**论文题目：** Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation

**作者：** Taekyung Ki, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon, Sung Ju Hwang

**摘要**

**1. 研究问题/核心挑战：**
当前的面部头像生成技术虽然能够创建逼真的虚拟形象，但普遍缺乏真正交互式的沟通体验。现有的模型通常生成单向响应，情感参与度低，无法捕捉真实对话中用户与虚拟形象之间的动态互动。论文指出了实现真正交互式头像的两个关键挑战：
*   **实时性与因果约束下的运动生成：** 需要在低延迟下生成头像的运动，同时满足因果关系，即只能依赖过去和当前的信息，不能预知未来。
*   **无额外标注数据的表达性反应学习：** 学习生动、富有表现力的头像反应，而无需昂贵且难以获取的标注数据。

**2. 主要创新与方法贡献：**
为了解决上述挑战，作者提出了 **Avatar Forcing** 框架，其核心创新包括：

*   **基于扩散强制（Diffusion Forcing）的实时交互式头像生成：** 该框架利用因果扩散强制技术，在运动潜在空间中建模用户与头像之间的实时交互。这使得头像能够处理实时的多模态输入（用户音频和运动），实现低延迟的即时响应。
*   **双重运动编码器（Dual Motion Encoder）：** 该编码器能够融合用户的音频、运动以及头像自身的音频，生成一个统一的条件输入，以驱动头像的运动生成。
*   **因果扩散强制运动生成器（Causal DFoT Motion Generator）：** 采用基于块的因果注意力机制，并引入了“前瞻性因果掩码”（Blockwise Causal Look-ahead Mask），在保证因果性的同时，实现了更平滑的帧间过渡，有效缓解了运动抖动问题。
*   **基于偏好优化的表达性交互学习：** 引入了一种**直接偏好优化（Direct Preference Optimization, DPO）**方法，通过构造“劣势样本”（即仅基于头像音频生成的运动）来训练模型，从而在无需人工标注的情况下，学习更具表现力和响应性的头像运动。这种方法能够有效提升头像的交互性和丰富性。

**3. 主要结果与意义：**
*   **实时性：** Avatar Forcing 实现了约 500ms 的低延迟，相比基线模型提速 6.8 倍，能够进行真正的实时交互。
*   **表达性与响应性：** 实验结果表明，Avatar Forcing 生成的头像运动更加生动、富有表现力，并且对用户的非语言线索（如微笑、点头）反应更灵敏。
*   **用户偏好：** 人类偏好研究显示，80% 以上的参与者更倾向于 Avatar Forcing 生成的头像，认为其在自然度、响应性和整体交互质量方面表现更优。
*   **技术优势：** 在定量评估中，Avatar Forcing 在 Reactiveness（响应性）和 Motion Richness（运动丰富性）等关键指标上显著优于现有模型（如 INFP*）。

**4. 论文中提到的局限性：**
*   **身体姿态的限制：** 该系统主要关注头部运动，无法捕捉更丰富的身体姿态（如手势），这限制了更动态的沟通表现。
*   **显式控制的不足：** 在某些场景下，可能需要更精细的控制，例如直接引导眼球注视或强调情感变化，而当前模型在这方面能力有限。

**5. 未来研究方向：**
*   **整合更丰富的用户信号：** 探索整合眼球追踪或情感追踪等额外用户信号，以实现更精细的控制和更具表现力的交互。
*   **扩展到全身模型：** 将模型扩展到全身，以捕捉更丰富的肢体语言，实现更动态和全面的沟通。
*   **伦理考量与滥用防范：** 论文也提到了该技术可能被用于身份欺骗或制作深度伪造内容（deepfakes）的风险，并鼓励社区开发相关的检测模型。

**总结：**
Avatar Forcing 是一个在实时交互式头部头像生成领域的重要进展。它通过创新的扩散强制和偏好优化方法，成功解决了现有模型在实时性、表达性和交互性方面的不足。该框架能够生成高度响应和富有表现力的头像，为虚拟交流、内容创作等应用带来了更沉浸式的用户体验，并为未来更智能、更自然的虚拟人交互奠定了基础。

**Key Findings:**

- To address these challenges, we propose Avatar Forcing, a new framework for interactive head avatar generation that models real-time user-avatar interactions through diffusion forcing.
- Furthermore, we introduce a direct preference optimization method that leverages synthetic losing samples constructed by dropping user conditions, enabling label-free learning of expressive interaction.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00664v1)
- [arXiv](https://arxiv.org/abs/2601.00664v1)

---

<a id='2601.00598v1'></a>
## [Modality Dominance-Aware Optimization for Embodied RGB-Infrared Perception](https://arxiv.org/abs/2601.00598v1)

**Authors:** Xianhui Liu, Siqi Jiang, Yi Xie, Yuqing Lin, Siao Liu

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

RGB-Infrared (RGB-IR) multimodal perception is fundamental to embodied multimedia systems operating in complex physical environments. Although recent cross-modal fusion methods have advanced RGB-IR detection, the optimization dynamics caused by asymmetric modality characteristics remain underexplored. In practice, disparities in information density and feature quality introduce persistent optimization bias, leading training to overemphasize a dominant modality and hindering effective fusion. To quantify this phenomenon, we propose the Modality Dominance Index (MDI), which measures modality dominance by jointly modeling feature entropy and gradient contribution. Based on MDI, we develop a Modality Dominance-Aware Cross-modal Learning (MDACL) framework that regulates cross-modal optimization. MDACL incorporates Hierarchical Cross-modal Guidance (HCG) to enhance feature alignment and Adversarial Equilibrium Regularization (AER) to balance optimization dynamics during fusion. Extensive experiments on three RGB-IR benchmarks demonstrate that MDACL effectively mitigates optimization bias and achieves SOTA performance.

**Analysis:**

好的，这是一篇关于RGB-红外（RGB-IR）多模态感知中优化偏差问题的论文。以下是根据您提供的PDF内容撰写的中文摘要，包含您要求的各个部分：

**论文题目：** Modality Dominance-Aware Optimization for Embodied RGB-Infrared Perception

**摘要：**

**1. 研究问题：**
本文聚焦于RGB-红外（RGB-IR）多模态感知在具身智能系统中的应用。研究的核心问题在于，尽管现有的跨模态融合方法在RGB-IR检测方面取得了显著进展，但由于RGB和红外模态在信息密度和特征质量上的不对称性，训练过程中常常出现“优化偏差”（optimization bias）。这种偏差导致模型过度依赖某个占主导地位的模态，从而阻碍了有效的跨模态特征融合，影响了整体感知性能。

**2. 主要创新与方法贡献：**
为了解决上述问题，作者提出了以下关键创新和方法：

*   **模态支配指数（Modality Dominance Index, MDI）：** 提出了一种新颖的度量指标MDI，用于量化不同模态在训练过程中的支配程度。MDI通过联合建模模态特征的“信息丰富度”（Representational Diversity，通过特征熵衡量）和“任务响应敏感度”（Task-Response Sensitivity，通过对检测损失的影响衡量）来动态评估模态的支配性。
*   **模态支配感知跨模态学习框架（Modality Dominance-Aware Cross-modal Learning, MDACL）：** 基于MDI，作者构建了一个MDACL框架，旨在调节跨模态的优化过程。该框架包含两个核心组件：
    *   **分层跨模态引导（Hierarchical Cross-modal Guidance, HCG）：** HCG通过“低级特征映射与重投影”和“高级语义蒸馏”两个阶段，增强模态间的特征对齐。低级阶段侧重于结构对齐，高级阶段侧重于语义一致性，从而有效缓解模态间的特征级错位。
    *   **对抗性均衡正则化（Adversarial Equilibrium Regularization, AER）：** AER策略通过引入“最小逆权重”（Minimal Inverse Weight, MIW）方案，动态调整模态的融合权重，抑制占主导模态的过度影响，鼓励更均衡的学习过程，以达到一种“对抗性均衡”状态。

**3. 主要结果与意义：**
作者在LLVIP、M3FD和FLIR三个广泛使用的RGB-IR检测基准数据集上进行了大量实验。结果表明：

*   MDACL框架能够有效缓解RGB-IR检测中的优化偏差问题。
*   该方法在所有三个基准数据集上均取得了最先进（SOTA）的性能，并在mAP和mAP50指标上实现了显著提升，尤其是在LLVIP和M3FD数据集上表现突出。
*   消融实验证明了MDI、HCG和AER各组件的有效性及其协同作用。
*   定性分析显示，MDACL能够显著减少漏检和误检，尤其是在低光照和遮挡等复杂场景下，证明了其在提升鲁棒性和准确性方面的优势。
*   研究强调了在RGB-IR多模态感知中，显式地建模和调节优化动态的重要性，这超越了传统仅关注特征融合设计的方法。

**4. 局限性：**
论文中未明确提及具体的局限性，但从其研究方向来看，可能存在的潜在局限性包括：

*   **计算复杂度：** 引入MDI计算和HCG、AER策略可能会增加模型的训练和推理复杂度，尽管作者声称AER的MIW方案计算开销极小。
*   **超参数敏感性：** HCG中的权重α, β, γ以及MDI中的δ等超参数可能需要仔细调优，以达到最佳性能。
*   **泛化性：** 虽然在三个数据集上取得了良好结果，但其在其他更具挑战性的或不同类型的RGB-IR数据集上的泛化能力仍需进一步验证。

**5. 未来研究方向：**
论文在结论部分暗示了未来的研究方向：

*   **更先进的正则化范式：** 探索更高级的正则化方法来进一步增强动态优化均衡，以应对更复杂的模态不平衡问题。
*   **更广泛的应用：** 将所提出的优化感知方法扩展到其他多模态学习任务，而不仅仅局限于RGB-IR检测。
*   **更深入的理论分析：** 对模态支配和优化偏差之间的关系进行更深入的理论研究。

总而言之，这篇论文通过引入模态支配指数（MDI）和提出模态支配感知跨模态学习框架（MDACL），有效地解决了RGB-IR多模态感知中的关键挑战——优化偏差问题。其提出的方法不仅在多个基准数据集上取得了SOTA性能，而且为理解和解决多模态学习中的不对称性问题提供了新的视角和有效的解决方案。

**Key Findings:**

- To quantify this phenomenon, we propose the Modality Dominance Index (MDI), which measures modality dominance by jointly modeling feature entropy and gradient contribution.
- Based on MDI, we develop a Modality Dominance-Aware Cross-modal Learning (MDACL) framework that regulates cross-modal optimization.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00598v1)
- [arXiv](https://arxiv.org/abs/2601.00598v1)

---

<a id='2601.00584v1'></a>
## [GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](https://arxiv.org/abs/2601.00584v1)

**Authors:** Mingyu Jeon, Sunjae Yoon, Jonghee Kim, Junyeoung Kim

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

Zero-shot video moment retrieval (ZVMR) is the task of localizing a temporal moment within an untrimmed video using a natural language query without relying on task-specific training data. The primary challenge in this setting lies in the mismatch in semantic granularity between textual queries and visual content. Previous studies in ZVMR have attempted to achieve alignment by leveraging high-quality pre-trained knowledge that represents video and language in a joint space. However, these approaches failed to balance the semantic granularity between the pre-trained knowledge provided by each modality for a given scene. As a result, despite the high quality of each modality's representations, the mismatch in granularity led to inaccurate retrieval. In this paper, we propose a training-free framework, called Granularity-Aware Alignment (GranAlign), that bridges this gap between coarse and fine semantic representations. Our approach introduces two complementary techniques: granularity-based query rewriting to generate varied semantic granularities, and query-aware caption generation to embed query intent into video content. By pairing multi-level queries with both query-agnostic and query-aware captions, we effectively resolve semantic mismatches. As a result, our method sets a new state-of-the-art across all three major benchmarks (QVHighlights, Charades-STA, ActivityNet-Captions), with a notable 3.23% mAP@avg improvement on the challenging QVHighlights dataset.

**Analysis:**

好的，这是对论文“GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval”的全面中文摘要：

**论文题目：** GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval

**作者：** Mingyu Jeon, Sunjae Yoon, Jonghee Kim, Junyeoung Kim

**摘要：**

**1. 研究问题/核心挑战：**
本文旨在解决零样本视频时刻检索（Zero-Shot Video Moment Retrieval, ZVMR）任务中的核心挑战——**语义粒度不匹配（Granularity Mismatch）**。在ZVMR中，用户可能使用不同详细程度的自然语言查询来描述视频中的同一事件（例如，“一只可爱的狗” vs. “一只金毛幼犬正在散步”）。现有的方法虽然利用了高质量的预训练视觉语言模型（VLMs）和大型语言模型（LLMs），但往往未能有效平衡文本查询和视频内容之间粗粒度（高召回率）与细粒度（高精确率）的语义表示，导致检索结果不准确。

**2. 主要创新点/方法贡献：**
作者提出了一个名为 **Granularity-Aware Alignment (GranAlign)** 的训练免费框架，以弥合粗粒度和细粒度语义表示之间的差距。其核心创新在于：

*   **双路径粒度感知对齐：** GranAlign摒弃了单一的查询处理路径，而是采用双路径策略：
    *   **查询侧（Query Side）：** 利用LLM（如LLaMA-3）将原始查询重写为两个不同粒度的查询：**简化查询（Simplified Query, Qs）**，侧重于捕捉核心意图以实现高召回率；**详细查询（Detailed Query, Qd）**，保留精细的语义信息以实现高精确率。
    *   **视频侧（Video Side）：** 生成两种类型的视频描述：**查询无关的字幕（Query-Agnostic Caption, Cagn）**，提供通用的场景描述；**查询感知的字幕（Query-Aware Caption, Cawr）**，针对关键帧生成，更贴合查询意图。
*   **粒度感知配对与评分：** 通过将简化查询与查询无关字幕（Qs, Cagn）配对，以及将详细查询与查询感知字幕（Qd, Cawr）配对，GranAlign能够协同利用两种路径的优势。最终的时刻得分（Moment Score）是这两种配对的语义相似度得分的平均值，从而在召回率和精确率之间取得平衡。
*   **训练免费框架：** GranAlign不需要在特定任务上进行额外的训练，而是依赖于预训练模型的能力，使其易于部署和应用。

**3. 主要结果与意义：**
GranAlign在三个主要的ZVMR基准数据集（QVHighlights, Charades-STA, ActivityNet-Captions）上均取得了**最先进（State-of-the-Art, SOTA）**的性能。特别是在挑战性的QVHighlights数据集上，mAP@avg指标取得了**3.23%**的显著提升。实验结果表明，GranAlign有效解决了粒度不匹配问题，显著提高了检索的准确性和鲁棒性。该方法在不同类型的查询（Error, Simple, Detail, Else）上均表现出良好的泛化能力，并且对超参数设置不敏感。

**4. 论文中提到的局限性：**
论文中提到，GranAlign的局限性主要源于其核心组件（如查询感知字幕生成）的生成性质。具体来说：
*   **查询感知字幕的幻觉（Hallucination）：** 查询感知字幕有时可能生成视频中不存在的内容，或者过度模仿查询的语言结构，导致误导。
*   **对LLM和VLM的依赖：** 框架的性能在一定程度上依赖于所使用的LLM和VLM的质量和能力。

**5. 未来研究方向：**
基于上述局限性，论文提出了以下未来研究方向：
*   **事实核查机制：** 开发一种机制来验证查询感知字幕中生成的细节是否与视觉证据相符，以减少幻觉。
*   **语义验证：** 引入一个语义验证步骤，以确保LLM重写的查询能够准确反映原始查询的意图。
*   **更通用的粒度对齐策略：** 探索更通用的方法来处理不同粒度的表示，并将其应用于更广泛的多模态理解任务。

总而言之，GranAlign通过引入创新的双路径粒度感知对齐框架，成功解决了零样本视频时刻检索中的关键挑战，并在多个基准测试中取得了显著的性能提升，为未来的研究开辟了新的方向。

**Key Findings:**

- In this paper, we propose a training-free framework, called Granularity-Aware Alignment (GranAlign), that bridges this gap between coarse and fine semantic representations.
- Our approach introduces two complementary techniques: granularity-based query rewriting to generate varied semantic granularities, and query-aware caption generation to embed query intent into video content.
- As a result, our method sets a new state-of-the-art across all three major benchmarks (QVHighlights, Charades-STA, ActivityNet-Captions), with a notable 3.23% mAP@avg improvement on the challenging QVHighlights dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00584v1)
- [arXiv](https://arxiv.org/abs/2601.00584v1)

---

<a id='2601.00561v1'></a>
## [AEGIS: Exploring the Limit of World Knowledge Capabilities for Unified Mulitmodal Models](https://arxiv.org/abs/2601.00561v1)

**Authors:** Jintao Lin, Bowen Dong, Weikang Shi, Chenyang Lei, Suiyun Zhang, Rui Liu, Xihui Liu

**Published:** 2026-01-02

**Categories:** cs.CV

**Abstract:**

The capability of Unified Multimodal Models (UMMs) to apply world knowledge across diverse tasks remains a critical, unresolved challenge. Existing benchmarks fall short, offering only siloed, single-task evaluations with limited diagnostic power. To bridge this gap, we propose AEGIS (\emph{i.e.}, \textbf{A}ssessing \textbf{E}diting, \textbf{G}eneration, \textbf{I}nterpretation-Understanding for \textbf{S}uper-intelligence), a comprehensive multi-task benchmark covering visual understanding, generation, editing, and interleaved generation. AEGIS comprises 1,050 challenging, manually-annotated questions spanning 21 topics (including STEM, humanities, daily life, etc.) and 6 reasoning types. To concretely evaluate the performance of UMMs in world knowledge scope without ambiguous metrics, we further propose Deterministic Checklist-based Evaluation (DCE), a protocol that replaces ambiguous prompt-based scoring with atomic ``Y/N'' judgments, to enhance evaluation reliability. Our extensive experiments reveal that most UMMs exhibit severe world knowledge deficits and that performance degrades significantly with complex reasoning. Additionally, simple plug-in reasoning modules can partially mitigate these vulnerabilities, highlighting a promising direction for future research. These results highlight the importance of world-knowledge-based reasoning as a critical frontier for UMMs.

**Analysis:**

好的，这是对论文“AEGIS: Exploring the Limit of World Knowledge Capabilities for Unified Mulitmodal Models”的中文摘要，其中包含了您要求的关键信息：

**论文摘要：AEGIS：探索统一多模态模型世界知识能力的极限**

**1. 研究问题与动机**

统一多模态模型（UMMs）在跨越不同任务应用世界知识的能力方面仍面临严峻挑战。现有基准测试在评估UMMs的世界知识和推理能力时存在局限性，它们通常是单一任务导向的，诊断能力有限，且评估指标不够精确。这使得深入理解UMMs在处理复杂、现实世界场景时的弱点变得困难。

**2. 主要创新与方法贡献**

为了解决上述问题，作者提出了 **AEGIS**（Assessing Editing, Generation, Interpretation-Understanding for Super-intelligence）基准测试。AEGIS具有以下关键创新：

*   **全面的多任务基准：** AEGIS包含1050个精心设计、人工标注的问题，涵盖了视觉理解、生成、编辑和交错生成四种核心任务。
*   **广泛的世界知识覆盖：** 该基准测试跨越STEM、人文和社会生活三大领域，细分为21个主题，确保了对模型世界知识广度的全面考察。
*   **多样的推理类型：** AEGIS引入了6种不同的推理类型（空间、时间、因果、比较、类比、逻辑），以深入评估模型在复杂推理方面的能力。
*   **确定性清单式评估（DCE）：** 为了克服现有评分方法的模糊性和主观性，AEGIS提出了一种创新的DCE协议。该协议将复杂的评估任务分解为一系列原子化的“是/否”判断问题，通过LLM生成检查清单，从而提高了评估的客观性和可靠性。

**3. 主要结果与意义**

通过在AEGIS基准上进行的大量实验，作者发现：

*   **UMMs的世界知识缺陷显著：** 大多数UMMs在应用世界知识方面表现出严重的不足，尤其是在处理复杂推理时，性能会急剧下降。
*   **推理能力是关键瓶颈：** 模型在理解任务上表现相对较好，但在生成和编辑任务上性能有所下降，而在需要复杂推理的交错生成任务上则出现大幅下滑。这表明理解能力限制了其他任务的上限。
*   **简单推理模块的缓解作用：** 集成简单的插件式推理模块可以在一定程度上缓解模型在世界知识方面的不足，为未来的模型改进指明了方向。
*   **AEGIS的诊断价值：** AEGIS基准测试及其DCE评估协议能够提供更精细的诊断能力，帮助研究人员定位模型在不同任务和推理类型中的具体弱点。

这项研究强调了基于世界知识的推理对于提升UMMs能力的重要性，并为未来UMM的研究和开发提供了重要的方向和评估工具。

**4. 局限性**

论文中提到，在交错生成任务中，模型在“图像-文本一致性检查”方面的判断难度较高，这增加了评估的挑战性。此外，虽然AEGIS提供了详细的诊断信息，但模型在处理非常规或模糊指令时仍存在挑战。

**5. 未来研究方向**

*   **改进多模态推理框架：** 针对模型在处理复杂推理类型（如时间、因果推理）方面的弱点，需要设计更强大的多模态推理框架。
*   **提升视觉解码器能力：** 研究发现视觉解码器是限制UMMs世界知识能力的一个瓶颈，未来需要关注如何增强解码器对知识的编码能力或提高其对输入变化的鲁棒性。
*   **更精细的模块化分析：** 进一步研究不同模块（如LLM组件和视觉解码器）在世界知识应用中的具体作用和相互影响。
*   **探索更有效的知识整合方法：** 研究如何更有效地将外部知识（如通过网络搜索）与模型内部知识相结合，以提升模型在动态或实时信息处理方面的能力。

总而言之，AEGIS基准测试及其评估方法为衡量和提升统一多模态模型的世界知识和推理能力提供了一个重要的平台，并揭示了当前模型在该领域面临的关键挑战和未来研究的潜在方向。

**Key Findings:**

- To bridge this gap, we propose AEGIS (\emph{i.e.}, \textbf{A}ssessing \textbf{E}diting, \textbf{G}eneration, \textbf{I}nterpretation-Understanding for \textbf{S}uper-intelligence), a comprehensive multi-task benchmark covering visual understanding, generation, editing, and interleaved generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00561v1)
- [arXiv](https://arxiv.org/abs/2601.00561v1)

---

<a id='2601.00553v1'></a>
## [A Comprehensive Dataset for Human vs. AI Generated Image Detection](https://arxiv.org/abs/2601.00553v1)

**Authors:** Rajarshi Roy, Nasrin Imanpour, Ashhar Aziz, Shashwat Bajpai, Gurpreet Singh, Shwetangshu Biswas, Kapil Wanaskar, Parth Patwa, Subhankar Ghosh, Shreyas Dixit, Nilesh Ranjan Pal, Vipula Rawte, Ritvik Garimella, Gaytri Jena, Vasu Sharma, Vinija Jain, Aman Chadha, Aishwarya Naresh Reganti, Amitava Das

**Published:** 2026-01-02

**Categories:** cs.CV, cs.AI

**Abstract:**

Multimodal generative AI systems like Stable Diffusion, DALL-E, and MidJourney have fundamentally changed how synthetic images are created. These tools drive innovation but also enable the spread of misleading content, false information, and manipulated media. As generated images become harder to distinguish from photographs, detecting them has become an urgent priority. To combat this challenge, We release MS COCOAI, a novel dataset for AI generated image detection consisting of 96000 real and synthetic datapoints, built using the MS COCO dataset. To generate synthetic images, we use five generators: Stable Diffusion 3, Stable Diffusion 2.1, SDXL, DALL-E 3, and MidJourney v6. Based on the dataset, we propose two tasks: (1) classifying images as real or generated, and (2) identifying which model produced a given synthetic image. The dataset is available at https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将为您提供这篇论文的全面中文摘要。

**论文题目：** A Comprehensive Dataset for Human vs. AI Generated Image Detection

**作者：** Rajarshi Roy, Nasrin Imanpour, Ashhar Aziz, Shashwat Bajpai, Gurpreet Singh, Shwetangshu Biswas, Kapil Wanaskar, Parth Patwa, Subhankar Ghosh, Shreyas Dixit, Nilesh Ranjan Pal, Vipula Rawte, Ritvik Garimella, Gaytri Jena, Vasu Sharma, Vinija Jain, Aman Chadha, Aishwarya Naresh Reganti, Amitava Das

---

**论文全面中文摘要**

**1. 研究问题/核心挑战：**

随着Stable Diffusion、DALL-E和MidJourney等生成式AI技术的飞速发展，合成图像的生成能力得到了质的飞跃，使得区分真实图像与AI生成图像变得越来越困难。这带来了严重的社会风险，包括误导性信息、虚假新闻和操纵性媒体的传播，对公众信任和信息生态系统构成了威胁。因此，开发鲁棒的AI生成图像检测技术已成为一项紧迫的任务。

**2. 主要创新点/方法论贡献：**

*   **发布MS COCOAI数据集：** 该论文的核心贡献是发布了一个名为MS COCOAI的新型数据集，专门用于AI生成图像检测。该数据集包含96,000个真实和合成图像对，并构建在MS COCO数据集之上。
*   **多模型覆盖与语义对齐：** MS COCOAI数据集利用了五种主流的生成模型（Stable Diffusion 3, Stable Diffusion 2.1, SDXL, DALL-E 3, 和 MidJourney v6）来生成合成图像。关键创新在于，所有合成图像都是基于MS COCO数据集中真实图像的相同文本描述（caption）生成的，实现了“语义对齐”。这种对齐方式能够有效分离内容偏差与生成模型本身的伪影，为研究提供了更可控的环境。
*   **引入两种检测任务：** 基于该数据集，论文提出了两个核心任务：
    *   **任务A（二分类）：** 判断一张图像是真实图像还是AI生成图像。
    *   **任务B（多分类/模型识别）：** 识别出生成特定合成图像的具体AI模型。
*   **鲁棒性评估：** 数据集还包含了对生成图像进行的四种独立扰动（水平翻转、亮度降低、高斯噪声、JPEG压缩），旨在评估检测方法的鲁棒性。

**3. 主要结果与意义：**

*   **基线模型性能：** 论文使用ResNet-50模型作为基线，在频率域特征上进行了实验。
    *   在任务A（真实/AI二分类）上，基线模型取得了0.80144的得分，表明使用相对简单的方法进行二分类检测是可行的。
    *   在任务B（模型识别）上，基线模型仅取得了0.44913的得分，这显著低于任务A的得分。
*   **结果意义：**
    *   **检测难度差异：** 结果清晰地表明，模型识别（多分类）比简单的真实/AI二分类要困难得多，凸显了区分不同生成模型的技术挑战。
    *   **数据集价值：** MS COCOAI数据集为AI生成图像检测和模型归因研究提供了一个大规模、高质量且具有语义对齐特性的基准。它能够帮助研究人员更深入地理解生成模型的特性，并开发更先进的检测技术。
    *   **信息完整性：** 数据集包含了图像、文本描述、真实/AI标签、模型标签以及扰动版本，为多模态分析和鲁棒性研究提供了全面的支持。

**4. 提及的局限性：**

*   **模型识别的挑战：** 论文明确指出，模型识别任务的基线性能较低，表明当前的技术在区分不同生成模型方面仍有很大提升空间。
*   **现有数据集的不足：** 论文在“Related Work”部分回顾了现有数据集，并指出了它们的局限性，例如缺乏语义对齐、模型标签不精细、图像质量参差不齐等，从而突显了MS COCOAI的必要性。
*   **基线方法的局限：** 论文使用的频率域特征基线方法虽然有效，但可能无法捕捉所有细微的生成伪影，尤其是在面对更先进的生成模型时。

**5. 潜在的未来研究方向：**

*   **更精细的模型指纹技术：** 开发更先进的“指纹”技术，以更精确地识别不同生成模型的细微差异。
*   **跨模态学习：** 利用文本描述和图像之间的关联性，探索更有效的跨模态学习方法来提升检测性能。
*   **增强检测器的鲁棒性：** 进一步研究如何提高检测器对常见图像变换（如压缩、裁剪、噪声等）的鲁棒性。
*   **探索新的检测特征：** 除了频率域特征，还可以探索其他类型的特征，如语义特征、纹理特征、模型特定的伪影等。
*   **对抗性攻击与防御：** 研究针对AI生成图像检测器的对抗性攻击，并开发相应的防御策略。

**总结：**

这篇论文通过发布MS COCOAI数据集，为AI生成图像检测领域做出了重要贡献。该数据集的独特之处在于其大规模、多模型覆盖以及关键的语义对齐特性，为研究人员提供了一个强大的工具来评估和开发更先进的检测和模型归因技术。论文通过基线实验揭示了模型识别的巨大挑战，并为未来的研究指明了方向，强调了在日益复杂的AI生成内容环境中维护信息真实性的重要性。

**Key Findings:**

- To combat this challenge, We release MS COCOAI, a novel dataset for AI generated image detection consisting of 96000 real and synthetic datapoints, built using the MS COCO dataset.
- Based on the dataset, we propose two tasks: (1) classifying images as real or generated, and (2) identifying which model produced a given synthetic image.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.00553v1)
- [arXiv](https://arxiv.org/abs/2601.00553v1)

---

