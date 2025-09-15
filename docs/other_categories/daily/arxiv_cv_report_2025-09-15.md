time: 20250915

# Arxiv Computer Vision Papers - 2025-09-15

## Executive Summary

好的，这是一份针对您提供的 Arxiv 论文列表的每日报告执行摘要，旨在帮助忙碌的研究人员快速了解最新进展：

---

**Arxiv 计算机视觉每日报告执行摘要 (2025-09-12)**

**1. 主要主题和趋势概述：**

今天的论文涵盖了计算机视觉和机器学习领域的多个前沿方向，主要趋势包括：

*   **多模态融合与视觉语言模型 (VLM) 的深入探索：** 多篇论文关注 VLM 的理解、应用和优化，特别是视觉接地 (Visual Grounding) 和多语言视觉问答 (VQA)。
*   **高效与加速：** 针对视频处理、扩散模型和表示学习的效率提升是重要主题，旨在降低计算成本并提高实时性能。
*   **数据与基准：** 新数据集和基准的发布，特别是针对文本到图像生成中的伪影评估，凸显了高质量数据在模型开发中的关键作用。
*   **特定应用领域的进展：** 视频质量增强、事件相机应用、自动驾驶中的轨迹预测以及语义分割等领域均有创新。
*   **基础模型适应与优化：** 如何将大型基础模型（如 SAM）适应到特定任务，以及表示学习的稳定性问题也受到关注。

**2. 特别重要或创新的论文：**

*   **"Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching" (Zhixin Zheng et al.)：** 这篇论文在加速扩散模型方面展现了显著的创新。通过引入聚类驱动的特征缓存，它有望大幅降低扩散 Transformer 的计算成本，对于实时生成和资源受限环境下的应用具有重要意义。
*   **"MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation" (Jia Wang et al.)：** 随着文本到图像生成模型的普及，评估其生成质量，特别是细粒度伪影，变得至关重要。MagicMirror 提供了一个急需的大规模数据集和基准，将极大地推动该领域的研究和模型改进。
*   **"LayerLock: Non-collapsing Representation Learning with Progressive Freezing" (Goker Erdogan et al.)：** 在表示学习中，避免模型崩溃是一个核心挑战。LayerLock 提出的渐进式冻结策略为解决这一问题提供了一种新颖且可能更稳定的方法，对自监督学习和基础模型训练具有潜在影响。

**3. 新兴研究方向或技术：**

*   **事件相机与多模态融合：** "Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey" 强调了事件相机在传统视觉任务中的独特优势，预示着其与传统相机数据融合的更多应用。
*   **扩散模型加速与优化：** 除了上述的特征缓存，对扩散模型计算效率的持续关注将是未来研究的重点。
*   **多语言 VQA 与奖励优化：** "LaV-CoT" 提出的语言感知视觉 CoT 和多方面奖励优化，为处理真实世界多语言 VQA 带来了新的思路，强调了对语言和文化多样性的关注。
*   **基础模型（如 SAM）的轻量级适配：** "Multimodal SAM-adapter" 展示了如何高效地将大型基础模型应用于特定任务，这是一种重要的工程和研究方向。

**4. 建议阅读全文的论文：**

对于不同兴趣的研究人员，以下论文值得优先阅读：

*   **对于关注模型效率和生成式 AI 的研究人员：**
    *   **"Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching"** (Zhixin Zheng et al.)
    *   **"MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation"** (Jia Wang et al.)
*   **对于关注视觉语言模型和多模态理解的研究人员：**
    *   **"Towards Understanding Visual Grounding in Visual Language Models"** (Georgios Pantazopoulos, Eda B. Özyiğit)
    *   **"LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward Optimization for Real-World Multilingual VQA"** (Jing Huang et al.)
*   **对于关注基础模型、表示学习和通用视觉任务的研究人员：**
    *   **"LayerLock: Non-collapsing Representation Learning with Progressive Freezing"** (Goker Erdogan et al.)
    *   **"Multimodal SAM-adapter for Semantic Segmentation"** (Iacopo Curti et al.)
*   **对于关注自动驾驶和实时感知的研究人员：**
    *   **"BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals"** (Minsang Kong et al.)

---

这份摘要旨在提供一个高层次的概览，帮助您快速识别与您研究方向最相关的论文。建议根据您的具体兴趣，进一步深入阅读所推荐的论文。

---

## Table of Contents

1. [Compressed Video Quality Enhancement: Classifying and Benchmarking over Standards](#2509.10407v1)
2. [Towards Understanding Visual Grounding in Visual Language Models](#2509.10345v1)
3. [Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey](#2509.09971v1)
4. [Multimodal SAM-adapter for Semantic Segmentation](#2509.10408v1)
5. [Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching](#2509.10312v1)
6. [MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation](#2509.10260v1)
7. [LayerLock: Non-collapsing Representation Learning with Progressive Freezing](#2509.10156v1)
8. [VARCO-VISION-2.0 Technical Report](#2509.10105v1)
9. [BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals](#2509.10080v1)
10. [LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward Optimization for Real-World Multilingual VQA](#2509.10026v1)

---

## Papers

<a id='2509.10407v1'></a>
## [Compressed Video Quality Enhancement: Classifying and Benchmarking over Standards](https://arxiv.org/abs/2509.10407v1)

**Authors:** Xiem HoangVan, Dang BuiDinh, Sang NguyenQuang, Wen-Hsiao Peng

**Published:** 2025-09-12

**Categories:** cs.CV

**Abstract:**

Compressed video quality enhancement (CVQE) is crucial for improving user
experience with lossy video codecs like H.264/AVC, H.265/HEVC, and H.266/VVC.
While deep learning based CVQE has driven significant progress, existing
surveys still suffer from limitations: lack of systematic classification
linking methods to specific standards and artifacts, insufficient comparative
analysis of architectural paradigms across coding types, and underdeveloped
benchmarking practices. To address these gaps, this paper presents three key
contributions. First, it introduces a novel taxonomy classifying CVQE methods
across architectural paradigms, coding standards, and compressed-domain feature
utilization. Second, it proposes a unified benchmarking framework integrating
modern compression protocols and standard test sequences for fair
multi-criteria evaluation. Third, it provides a systematic analysis of the
critical trade-offs between reconstruction performance and computational
complexity observed in state-of-the-art methods and highlighting promising
directions for future research. This comprehensive review aims to establish a
foundation for consistent assessment and informed model selection in CVQE
research and deployment.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Xiem HoangVan等人撰写的论文“Compressed Video Quality Enhancement: Classifying and Benchmarking over Standards”的全面摘要。

---

### 论文摘要：压缩视频质量增强：基于标准的分类与基准测试

**1. 主要问题或研究问题：**
该论文旨在解决现有压缩视频质量增强（CVQE）领域研究中存在的局限性。尽管基于深度学习的CVQE方法取得了显著进展，但现有综述缺乏系统性分类，未能将方法与特定标准和伪影关联起来；对不同编码类型（帧内/帧间）的架构范式缺乏充分的比较分析，尤其是在新兴标准如H.266/VVC方面；以及基准测试实践不完善，导致性能评估不一致且模型选择缺乏依据。核心问题是如何建立一个统一、系统且全面的框架，以分类、评估和比较不同CVQE方法，从而促进该领域的研究和实际部署。

**2. 关键创新或方法论贡献：**
为了解决上述问题，该论文提出了三项关键贡献：

*   **新颖的分类法：** 论文引入了一个全新的分类体系，将CVQE方法根据其**架构范式**（如CNN、注意力机制、混合模型）、**视频编码标准**（H.264/AVC、H.265/HEVC、H.266/VVC）以及**压缩域特征利用**情况进行系统性分类。这为理解和组织现有方法提供了清晰的结构。
*   **统一的基准测试框架：** 论文提出了一个集成了现代压缩协议和标准测试序列的统一基准测试框架。该框架支持**多标准、多量化参数（QP）**下的**多准则评估**，包括客观质量指标（APSNR、ASSIM）和计算复杂度指标（参数量、FLOPs），确保了评估的公平性和可复现性。
*   **系统性分析与权衡：** 论文对最先进的CVQE方法在**重建性能**和**计算复杂度**之间的关键权衡进行了系统性分析。通过详细的数据分析，揭示了不同方法在质量提升和资源消耗之间的关系，并指出了未来研究的潜在方向。

**3. 主要结果及其意义：**
论文的分析揭示了以下主要结果和意义：

*   **HEVC是研究热点：** H.265/HEVC仍然是CVQE研究中最主要的编码标准，相关方法数量最多，且在帧内和帧间增强策略上均有显著进展。
*   **注意力机制和混合架构的优势：** 基于注意力机制的模型（如OVQE）在APSNR和ASSIM指标上表现出最高的性能，但通常伴随着较高的计算复杂度。混合架构（如CTVE、STFF）在性能和效率之间取得了更好的平衡，能够以较低的计算成本实现接近Transformer的重建质量。
*   **帧间方法优于帧内方法：** 帧间增强方法（利用多帧时空相关性）通常比帧内方法（仅处理单帧）表现出更优异的性能。
*   **VVC研究的初期阶段：** 针对H.266/VVC的CVQE研究仍处于早期阶段，现有方法多为对HEVC模型的直接适配，尚未充分利用VVC特有的编码工具和结构。尽管OVQE在VVC序列上表现最佳，但这表明VVC专属设计仍有待深入探索。
*   **压缩域信息利用的潜力：** 论文强调了利用压缩域信息（如运动矢量、残差信号、CTU结构）的潜力，这可以显著降低计算开销并保留关键的压缩感知线索，但如何有效利用这些特征仍是一个挑战。

**4. 论文中提及的局限性：**
论文本身指出了现有CVQE研究和基准测试的局限性，这也是其工作动机：

*   现有综述缺乏系统性分类，未能将方法与特定标准和伪影关联。
*   对不同编码类型（帧内/帧间）的架构范式缺乏充分的比较分析，尤其是在新兴标准如H.266/VVC方面。
*   基准测试实践不完善，导致性能评估不一致且模型选择缺乏依据。
*   针对VVC的CVQE研究仍处于早期阶段，现有方法多为对HEVC模型的直接适配，未能充分利用VVC特有的编码工具和结构。
*   在实时部署方面，高性能的注意力机制和帧间方法通常具有较高的计算需求，这限制了它们在资源受限场景下的应用。

**5. 潜在的未来研究方向：**
基于论文的分析和发现，未来的研究方向包括：

*   **VVC专属的CVQE设计：** 鉴于VVC的最新性和其独特的编码工具，开发专门针对VVC压缩伪影的深度学习模型，充分利用其编码结构和信息，是未来的重要方向。
*   **高效的混合架构：** 进一步探索和优化混合架构，以在重建性能和计算效率之间取得更好的平衡，使其更适用于实时和资源受限的应用。
*   **压缩域信息利用：** 深入研究如何更有效地利用压缩域信息（如运动矢量、残差、CTU结构等），以降低计算开销并提高增强性能，同时保持时空一致性。
*   **多任务学习和自适应方法：** 开发能够自适应不同压缩级别、视频内容和编码标准的CVQE模型，提高其泛化能力和鲁棒性。
*   **感知质量优化：** 除了传统的PSNR/SSIM等客观指标，未来研究应更多关注人类视觉系统感知的质量，开发更符合人眼感知的损失函数和评估指标。

---

这份摘要旨在全面概括论文的核心内容，突出其在CVQE领域分类、基准测试和性能分析方面的重要贡献。

**Key Findings:**

- To address these gaps, this paper presents three key
contributions.
- First, it introduces a novel taxonomy classifying CVQE methods
across architectural paradigms, coding standards, and compressed-domain feature
utilization.
- Third, it provides a systematic analysis of the
critical trade-offs between reconstruction performance and computational
complexity observed in state-of-the-art methods and highlighting promising
directions for future research.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10407v1)
- [arXiv](https://arxiv.org/abs/2509.10407v1)

---

<a id='2509.10345v1'></a>
## [Towards Understanding Visual Grounding in Visual Language Models](https://arxiv.org/abs/2509.10345v1)

**Authors:** Georgios Pantazopoulos, Eda B. Özyiğit

**Published:** 2025-09-12

**Categories:** cs.CV, cs.AI

**Abstract:**

Visual grounding refers to the ability of a model to identify a region within
some visual input that matches a textual description. Consequently, a model
equipped with visual grounding capabilities can target a wide range of
applications in various domains, including referring expression comprehension,
answering questions pertinent to fine-grained details in images or videos,
caption visual context by explicitly referring to entities, as well as low and
high-level control in simulated and real environments. In this survey paper, we
review representative works across the key areas of research on modern
general-purpose vision language models (VLMs). We first outline the importance
of grounding in VLMs, then delineate the core components of the contemporary
paradigm for developing grounded models, and examine their practical
applications, including benchmarks and evaluation metrics for grounded
multimodal generation. We also discuss the multifaceted interrelations among
visual grounding, multimodal chain-of-thought, and reasoning in VLMs. Finally,
we analyse the challenges inherent to visual grounding and suggest promising
directions for future research.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Georgios Pantazopoulos和Eda B. Özyiğit撰写的论文“Towards Understanding Visual Grounding in Vision-Language Models”的全面摘要。

---

### 论文摘要：理解视觉语言模型中的视觉基础

**1. 主要问题或研究问题：**
该论文旨在全面理解视觉基础（Visual Grounding）在现代视觉语言模型（VLMs）中的作用、发展、挑战和未来方向。视觉基础是指模型根据文本描述在视觉输入（图像或视频）中识别和定位特定区域的能力。作者认为，尽管视觉基础在多模态人工智能中至关重要，但其在VLMs中的多方面影响、架构选择、训练范式以及与多模态推理的复杂关系仍需深入探讨。

**2. 关键创新或方法论贡献：**
*   **全面综述视觉基础的演变：** 论文追溯了视觉基础从早期结合CNNs和RNNs的方法，到基于Transformer的多任务视觉语言预训练（VLP）模型，再到当前利用VLMs实现接地文本生成的发展历程。
*   **细致分析区域表示方法：** 探讨了对象中心（Object-centric）和像素级（Pixel-level）两种视觉参考表示方法，并进一步细分了像素级方法中的离散坐标和原始坐标表示，强调了它们对模型性能和空间精度的影响。
*   **深入剖析VLM架构组件：** 详细分析了视觉编码器（Vision Encoder）、多模态连接器（Multimodal Connector）和语言骨干（Language Backbone）等核心组件如何影响视觉基础能力。特别讨论了图像分辨率处理（位置编码插值、任意分辨率图像处理）和视觉-文本表示连接（线性/非线性映射、序列压缩方法如池化、卷积、交叉注意力、重采样器）等关键设计选择。
*   **训练流程与对齐机制：** 阐述了VLM多阶段训练流程，包括图像-文本对齐预训练和微调（SFT、强化学习）。强调了在训练中融入视觉基础目标的重要性，以及如何通过链式思考（Chain-of-Thought）和推理来增强多模态理解。
*   **广泛的评估领域和指标：** 总结了视觉基础在多种应用中的评估基准和数据集，包括指代表达理解（REC）、接地视觉问答（GVQA）、接地图像描述（GC）和GUI代理交互，并详细介绍了常用的评估指标（如IoU、Precision@F1、CLIPScore等）。

**3. 主要结果及其意义：**
该论文作为一篇综述，没有提出新的实验结果，而是对现有研究进行了系统性梳理和总结，其意义在于：
*   **强调视觉基础的核心地位：** 明确指出视觉基础不仅是实现细粒度多模态理解的关键，也是提高VLM输出可解释性和可靠性的重要手段，尤其是在减少幻觉方面。
*   **揭示架构选择的影响：** 强调了像素级表示（特别是原始坐标）和Transformer骨干在现代VLMs中的主导地位，并指出不同连接器设计（如MLP、Q-former、重采样器）对性能和效率的权衡。
*   **指导未来模型开发：** 通过对现有挑战和局限性的分析，为下一代接地VLM的设计和训练提供了清晰的指导，例如在预训练阶段引入接地目标、处理高分辨率图像、以及平衡语言模型和视觉编码器的质量。
*   **促进多模态推理发展：** 探讨了视觉基础与多模态链式思考和推理的互补关系，指出将接地信号整合到推理过程中可以提高模型的透明度和在复杂场景下的性能。

**4. 论文中提及的局限性：**
*   **数据质量和可复现性：** 许多现有数据集依赖伪标签、复杂管道或专有模型，导致数据质量次优，并引发可复现性、可访问性和数据污染方面的担忧。
*   **基准测试的生态有效性：** 现有REC基准测试（如RefCOCO系列）可能已饱和，且在对象和词汇变异性方面存在局限性，需要更具生态有效性的基准来评估模型的泛化能力。
*   **模型泛化能力：** 尽管微调可以提高基准分数，但其对模型泛化到真实世界场景的能力提供有限证据。
*   **Mamba骨干的局限性：** 尽管Mamba模型在序列建模方面表现出潜力，但在接地任务上，基于Transformer的骨干模型表现显著更好，表明Mamba在视觉基础任务上仍有待进一步探索。
*   **训练范式的复杂性：** VLM的开发是一个复杂的多阶段过程，需要精心平衡粗粒度和细粒度任务，并解决因分布偏移、任务干扰和参数冲突导致的遗忘问题。
*   **映射与压缩的权衡：** 图像级理解中的压缩可能适用于许多多模态任务，但对于需要细粒度理解的感知任务（如视觉基础），特征保留方法通常更有益。

**5. 潜在的未来研究方向：**
*   **预训练阶段的接地目标：** 迫切需要开发具有充分文档和透明数据创建流程的公共资源，以在VLM开发的所有阶段融入接地目标。
*   **新一代基准测试：** 开发更具生态有效性的基准，以更好地评估VLMs在真实世界场景中的泛化能力，并确保评估数据不被用于训练。
*   **GUI代理的进一步发展：** 持续开发模型、数据集和评估指标，以增强GUI代理的多模态接地能力，使其能更有效地与Web、操作系统和移动设备交互。
*   **验证架构设计选择：** 进一步研究VLM的设计空间，以验证不同架构选择（如交叉注意力与自注意力、映射与压缩）在视觉基础任务中的影响。
*   **多模态接地与推理的整合：** 探索将接地信号整合到链式思考推理轨迹中，以提高模型的解释性、事实准确性和在复杂场景下的性能。
*   **高分辨率图像处理：** 进一步优化高分辨率图像的处理策略，以在保持计算效率的同时，确保模型能够捕获细粒度的视觉信息。

---

这篇论文为理解视觉基础在VLMs中的关键作用提供了一个全面的框架，并为未来研究指明了方向，以克服现有挑战，推动多模态人工智能的进一步发展。

**Key Findings:**

- We first outline the importance
of grounding in VLMs, then delineate the core components of the contemporary
paradigm for developing grounded models, and examine their practical
applications, including benchmarks and evaluation metrics for grounded
multimodal generation.
- We also discuss the multifaceted interrelations among
visual grounding, multimodal chain-of-thought, and reasoning in VLMs. Finally,
we analyse the challenges inherent to visual grounding and suggest promising
directions for future research.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10345v1)
- [arXiv](https://arxiv.org/abs/2509.10345v1)

---

<a id='2509.09971v1'></a>
## [Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey](https://arxiv.org/abs/2509.09971v1)

**Authors:** Aupendu Kar, Vishnu Raj, Guan-Ming Su

**Published:** 2025-09-12

**Categories:** cs.CV

**Abstract:**

Event camera sensors are bio-inspired sensors which asynchronously capture
per-pixel brightness changes and output a stream of events encoding the
polarity, location and time of these changes. These systems are witnessing
rapid advancements as an emerging field, driven by their low latency, reduced
power consumption, and ultra-high capture rates. This survey explores the
evolution of fusing event-stream captured with traditional frame-based capture,
highlighting how this synergy significantly benefits various video restoration
and 3D reconstruction tasks. The paper systematically reviews major deep
learning contributions to image/video enhancement and restoration, focusing on
two dimensions: temporal enhancement (such as frame interpolation and motion
deblurring) and spatial enhancement (including super-resolution, low-light and
HDR enhancement, and artifact reduction). This paper also explores how the 3D
reconstruction domain evolves with the advancement of event driven fusion.
Diverse topics are covered, with in-depth discussions on recent works for
improving visual quality under challenging conditions. Additionally, the survey
compiles a comprehensive list of openly available datasets, enabling
reproducible research and benchmarking. By consolidating recent progress and
insights, this survey aims to inspire further research into leveraging event
camera systems, especially in combination with deep learning, for advanced
visual media restoration and enhancement.

**Analysis:**

好的，这是一篇关于Aupendu Kar, Vishnu Raj和Guan-Ming Su撰写的论文“Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey”的全面摘要，内容基于您提供的PDF全文：

**论文摘要：Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey**

**1. 主要问题或研究问题：**
该综述论文旨在探讨如何利用事件相机（一种异步捕捉像素亮度变化的生物启发式传感器）与传统基于帧的相机技术相结合，以解决传统视觉系统在极端光照、快速运动和带宽效率低下等挑战性条件下，在视频媒体恢复和3D重建任务中面临的固有局限性。具体而言，论文关注如何通过事件驱动的融合技术，显著提升图像/视频的质量（包括时间增强和空间增强）以及3D重建的鲁棒性和准确性。

**2. 关键创新或方法论贡献：**
该论文本身是一篇综述，其主要贡献在于系统性地梳理和总结了该领域内的关键创新和方法论：

*   **事件数据表示与处理：** 论文详细介绍了多种事件数据表示方法（如基于图像、基于体素、基于图、基于脉冲和基于学习的表示），以及事件数据增强技术（如特征提取、空间和时间上采样、去噪），以克服事件数据的稀疏性、噪声和低分辨率等挑战，使其能与深度学习模型有效融合。
*   **时间增强：** 综述了事件相机在视频重建、帧插值和运动去模糊方面的应用。它涵盖了从早期基于模型的方法到近期基于深度学习（CNN/RNN、Transformer、SNN、扩散模型）的混合方法，强调事件相机如何利用其微秒级时间分辨率和无运动模糊特性来恢复精确运动线索和填充时间空白。
*   **空间增强：** 论文探讨了事件相机如何通过与传统RGB帧融合，实现超分辨率、HDR增强、低光照增强、遮挡移除、雨水移除和焦点控制。这些方法利用事件相机捕捉高频细节和动态范围的优势，弥补了传统相机在这些方面的不足。
*   **3D重建：** 综述了事件相机在3D重建领域的进展，特别是与神经辐射场（NeRF）和3D高斯泼溅（3DGS）等光真实感3D重建方法的融合。事件相机能够处理运动模糊、低光照和姿态估计等挑战，为实时、高质量的场景建模开辟了新途径。
*   **数据集汇编：** 论文提供了一个全面的公开数据集列表，为可复现的研究和基准测试提供了资源。

**3. 主要结果及其意义：**
该综述的核心发现和意义在于：

*   **事件相机作为互补传感器的巨大潜力：** 事件相机凭借其低延迟、低功耗、超高捕捉率和高动态范围等独特优势，能够有效弥补传统基于帧相机在动态场景和极端光照条件下的不足。
*   **深度学习在事件融合中的核心作用：** 深度学习架构（如Transformer、SNN、扩散模型）的进步，使得处理异步、稀疏的事件数据流成为可能，并显著提升了视觉媒体恢复和3D重建的性能。
*   **多模态融合的协同效应：** 事件数据与RGB图像、深度信息等传统模态的融合，能够产生超越单一模态的协同效应，在各种挑战性条件下实现前所未有的视觉保真度。
*   **推动新应用领域：** 事件相机技术的发展，特别是与深度学习的结合，正在推动自动驾驶、计算摄影和实时增强现实等领域的进步。

**4. 论文中提到的局限性：**
尽管事件相机技术取得了显著进展，论文也指出了现有研究的一些局限性：

*   **事件数据稀疏性与噪声：** 在低亮度或慢速场景中，事件通常稀疏且嘈杂，这给事件处理带来了挑战。
*   **数据集多样性不足：** 现有事件相机数据集在场景多样性、物体类型、运动模式和背景方面存在局限性，且语义级标注不足，彩色事件数据集稀缺，多视角事件数据集也未得到充分代表。
*   **模型泛化能力：** 许多现有模型在部署到不同频率或不同域的数据时，泛化能力较差。
*   **计算成本：** 某些高级融合或重建方法（如SPADE-E2VID）可能会增加计算成本。
*   **校准与同步：** 在多相机设置中，事件相机与RGB相机之间的精确校准和同步仍然是一个挑战。

**5. 潜在的未来研究方向：**
论文提出了以下几个未来研究方向，以进一步推动该领域的发展：

*   **事件驱动的多模态融合：** 开发更鲁棒的融合框架，动态平衡RGB、深度和事件模态，以实现实时恢复。
*   **低资源和边缘部署：** 设计轻量级架构，优化事件处理在资源受限的移动和嵌入式平台上的性能。
*   **自监督和无监督学习：** 探索域适应、对比学习和生成模型，减少对标注数据的依赖，提高模型泛化能力。
*   **事件基校准和同步：** 研究使用事件流进行无校准多相机设置，以实现动态环境中的鲁棒同步和对齐。
*   **事件引导的生成模型：** 将事件数据与扩散模型结合，在极端条件下实现高质量的合成和恢复。
*   **事件基深度和焦点估计：** 推进基于事件的深度和焦点控制算法，应用于机器人、AR/VR和计算摄影。
*   **跨领域应用：** 将事件基恢复技术应用于医学成像、遥感和工业检测等领域。
*   **彩色事件相机：** 扩展彩色事件数据研究，以改进视觉媒体恢复和重建。
*   **基准测试和数据集扩展：** 创建多样化、带标注的多视角数据集，涵盖不同光照、运动和语义上下文，以支持可复现的研究。
*   **安全性和鲁棒性：** 解决事件基系统中的漏洞，包括对抗性攻击和后门威胁，确保关键应用中的安全部署。

这份综述为事件相机在视觉媒体恢复和3D重建领域的应用提供了一个全面的视角，并为未来的研究指明了方向。

**Key Findings:**

- The paper systematically reviews major deep
learning contributions to image/video enhancement and restoration, focusing on
two dimensions: temporal enhancement (such as frame interpolation and motion
deblurring) and spatial enhancement (including super-resolution, low-light and
HDR enhancement, and artifact reduction).

**Links:**

- [PDF](https://arxiv.org/pdf/2509.09971v1)
- [arXiv](https://arxiv.org/abs/2509.09971v1)

---

<a id='2509.10408v1'></a>
## [Multimodal SAM-adapter for Semantic Segmentation](https://arxiv.org/abs/2509.10408v1)

**Authors:** Iacopo Curti, Pierluigi Zama Ramirez, Alioscia Petrelli, Luigi Di Stefano

**Published:** 2025-09-12

**Categories:** cs.CV, cs.AI

**Abstract:**

Semantic segmentation, a key task in computer vision with broad applications
in autonomous driving, medical imaging, and robotics, has advanced
substantially with deep learning. Nevertheless, current approaches remain
vulnerable to challenging conditions such as poor lighting, occlusions, and
adverse weather. To address these limitations, multimodal methods that
integrate auxiliary sensor data (e.g., LiDAR, infrared) have recently emerged,
providing complementary information that enhances robustness. In this work, we
present MM SAM-adapter, a novel framework that extends the capabilities of the
Segment Anything Model (SAM) for multimodal semantic segmentation. The proposed
method employs an adapter network that injects fused multimodal features into
SAM's rich RGB features. This design enables the model to retain the strong
generalization ability of RGB features while selectively incorporating
auxiliary modalities only when they contribute additional cues. As a result, MM
SAM-adapter achieves a balanced and efficient use of multimodal information. We
evaluate our approach on three challenging benchmarks, DeLiVER, FMB, and MUSES,
where MM SAM-adapter delivers state-of-the-art performance. To further analyze
modality contributions, we partition DeLiVER and FMB into RGB-easy and RGB-hard
subsets. Results consistently demonstrate that our framework outperforms
competing methods in both favorable and adverse conditions, highlighting the
effectiveness of multimodal adaptation for robust scene understanding. The code
is available at the following link:
https://github.com/iacopo97/Multimodal-SAM-Adapter.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Iacopo Curti等人撰写的论文“Multimodal SAM-adapter for Semantic Segmentation”的全面摘要。

---

### 《Multimodal SAM-adapter for Semantic Segmentation》论文摘要

**1. 主要问题或研究问题：**
语义分割在自动驾驶、医学成像和机器人等领域至关重要，但现有方法在恶劣光照、遮挡和不利天气等挑战性条件下表现不佳。这些限制源于仅依赖RGB图像的固有局限性。为了提高鲁棒性，多模态方法（整合辅助传感器数据如LiDAR、红外等）已崭露头角，但如何有效地将这些多模态信息与强大的RGB基础模型（如Segment Anything Model, SAM）结合，同时保留其泛化能力并避免灾难性遗忘，是一个关键挑战。

**2. 关键创新或方法论贡献：**
本文提出了**MM SAM-adapter**，一个新颖的框架，旨在扩展Segment Anything Model (SAM)在多模态语义分割方面的能力。其核心创新和方法论贡献包括：

*   **SAM的适应性扩展：** MM SAM-adapter通过一个外部适配器网络，将融合后的多模态特征注入到SAM丰富的RGB特征中。这种设计使得模型能够保留SAM强大的RGB特征泛化能力，同时仅在辅助模态提供额外有用线索时选择性地整合它们，从而实现多模态信息的平衡和高效利用。
*   **非对称架构设计：** 论文提出了一种非对称的网络架构，其中SAM主干（处理RGB特征）拥有比多模态融合编码器和适配器模块（处理辅助模态）更多的参数。这种设计优先考虑SAM的RGB基础知识，同时利用多模态融合知识来处理挑战性场景，避免辅助模态在RGB信息充足时引入噪声。
*   **多模态融合编码器：** 采用了一个多模态融合编码器，它处理RGB图像和辅助模态的特征，生成多模态表示，然后输入到适配器模块。该编码器能够学习权衡不同模态的贡献，并在推理时动态选择相关模态。
*   **RGB-hard和RGB-easy数据集划分：** 为了更深入地分析模态贡献，作者将DeLiVER和FMB数据集划分为RGB-easy和RGB-hard子集，以评估方法在不同挑战程度下的性能。

**3. 主要结果及其意义：**
MM SAM-adapter在三个具有挑战性的基准数据集（DeLiVER、FMB和MUSES）上取得了**最先进的性能**。

*   **DeLiVER基准：** 在RGB-Depth、RGB-LiDAR和RGB-Event设置中，MM SAM-adapter均表现出色。特别是在RGB-hard场景下，当辅助模态（如LiDAR和Event）更具挑战性时，MM SAM-adapter的性能提升尤为显著，表明其能有效利用辅助信息。
*   **FMB基准：** 在RGB-Thermal设置中，MM SAM-adapter同样达到了最先进的性能，并在RGB-hard场景下展现出对辅助模态信息的卓越利用能力。
*   **MUSES基准：** 在RGB-LiDAR和RGB-Event设置中，MM SAM-adapter在整体以及各种昼夜和天气条件下均取得了最先进的性能，尤其在雾天和夜间等对RGB模态极具挑战性的场景中表现突出。
*   **消融研究：** 结果表明，侧调（side-tuning）SAM适配器优于标准微调和LoRA适应策略，能更好地保留SAM的先验知识。非对称架构设计被证实优于对称架构，强调了优先考虑RGB信息的重要性。此外，即使SAM主干完全冻结，多模态适配器和融合模块也能有效地引导SAM利用多模态信息。

这些结果一致证明了MM SAM-adapter在有利和不利条件下均优于竞争方法，突显了多模态适应性在鲁棒场景理解中的有效性。

**4. 论文中提及的局限性：**
论文中提到MM SAM-adapter的一个局限性是，由于路面融合模块的限制，它**目前仅支持两种输入模态**。

**5. 潜在的未来研究方向：**
*   **扩展对更多模态的支持：** 设计一个创新且高效的融合模块，以适应更复杂的多模态场景，支持超过两种输入模态。
*   **应用于其他任务：** 探索该框架在其他任务中的潜力，例如全景分割（panoptic segmentation），这为未来的研究提供了一个有前景的方向。

---

**Key Findings:**

- In this work, we
present MM SAM-adapter, a novel framework that extends the capabilities of the
Segment Anything Model (SAM) for multimodal semantic segmentation.
- We
evaluate our approach on three challenging benchmarks, DeLiVER, FMB, and MUSES,
where MM SAM-adapter delivers state-of-the-art performance.
- To further analyze
modality contributions, we partition DeLiVER and FMB into RGB-easy and RGB-hard
subsets.
- Results consistently demonstrate that our framework outperforms
competing methods in both favorable and adverse conditions, highlighting the
effectiveness of multimodal adaptation for robust scene understanding.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10408v1)
- [arXiv](https://arxiv.org/abs/2509.10408v1)

---

<a id='2509.10312v1'></a>
## [Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching](https://arxiv.org/abs/2509.10312v1)

**Authors:** Zhixin Zheng, Xinyu Wang, Chang Zou, Shaobo Wang, Linfeng Zhang

**Published:** 2025-09-12

**Categories:** cs.CV

**Abstract:**

Diffusion transformers have gained significant attention in recent years for
their ability to generate high-quality images and videos, yet still suffer from
a huge computational cost due to their iterative denoising process. Recently,
feature caching has been introduced to accelerate diffusion transformers by
caching the feature computation in previous timesteps and reusing it in the
following timesteps, which leverage the temporal similarity of diffusion models
while ignoring the similarity in the spatial dimension. In this paper, we
introduce Cluster-Driven Feature Caching (ClusCa) as an orthogonal and
complementary perspective for previous feature caching. Specifically, ClusCa
performs spatial clustering on tokens in each timestep, computes only one token
in each cluster and propagates their information to all the other tokens, which
is able to reduce the number of tokens by over 90%. Extensive experiments on
DiT, FLUX and HunyuanVideo demonstrate its effectiveness in both text-to-image
and text-to-video generation. Besides, it can be directly applied to any
diffusion transformer without requirements for training. For instance, ClusCa
achieves 4.96x acceleration on FLUX with an ImageReward of 99.49%, surpassing
the original model by 0.51%. The code is available at
https://github.com/Shenyi-Z/Cache4Diffusion.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zhixin Zheng等人撰写的论文“Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching”的全面摘要。

---

**论文摘要：Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching**

**1. 核心问题与研究动机**
扩散Transformer（Diffusion Transformers, DiTs）在生成高质量图像和视频方面表现出色，但其迭代去噪过程导致巨大的计算成本，严重限制了其实际应用。现有的特征缓存方法主要利用扩散模型在时间维度上的相似性来加速，但忽略了空间维度上的相似性。因此，论文旨在解决DiTs的计算效率瓶颈，特别是在高加速比下如何保持生成质量。

**2. 关键创新与方法贡献**
本文提出了**聚类驱动特征缓存（Cluster-Driven Feature Caching, ClusCa）**，作为现有时间维度特征缓存方法的正交补充。ClusCa的核心创新在于同时利用token的**时间相似性**和**空间相似性**来加速推理。具体贡献包括：

*   **空间相似性分析**：通过对DiTs中token空间冗余的系统性研究，论文发现同一聚类中的token在特征和运动模式上都表现出显著相似性，且聚类结构在时间步之间保持稳定。这为利用空间相似性进行加速提供了理论基础。
*   **时空特征重用范式**：ClusCa将所有时间步划分为缓存周期。在每个周期的第一个时间步进行全量计算，并利用K-Means算法对token进行空间聚类。在后续时间步中，ClusCa仅计算每个聚类中的一个代表性token（例如，通过随机选择），然后通过加权求和的方式将该代表性token的信息（空间重用）与前一时间步的缓存特征（时间重用）结合，传播给同一聚类中的所有其他token。这种方法能够将token数量减少90%以上，显著降低计算复杂度，同时通过双重重用策略缓解了缓存导致的误差累积。
*   **无需额外训练**：ClusCa是一个即插即用的方法，可以直接应用于任何扩散Transformer模型，无需进行额外的训练。

**3. 主要结果与意义**
论文在DiT、FLUX和HunyuanVideo等主流扩散架构上进行了广泛实验，验证了ClusCa在文本到图像和文本到视频生成任务中的有效性：

*   **显著加速与高质量生成**：ClusCa在FLUX模型上实现了4.96倍的加速，同时ImageReward评分达到99.49%，超越原始模型0.51%。在HunyuanVideo上，ClusCa实现了6.21倍加速，VBench评分达到79.60%。
*   **优越的效率-质量权衡**：与现有特征缓存方法（如FORA、ToCa、DuCa、TaylorSeer）相比，ClusCa在高加速比下表现出更低的FID和更高的生成质量，尤其是在极端加速条件下仍能保持性能。
*   **可视化验证**：PCA可视化结果显示，ClusCa生成的特征轨迹与未加速的原始DiT模型高度匹配，尤其是在去噪过程的后期，这定量验证了ClusCa在保持计算效率的同时成功最小化了误差累积。
*   **聚类与传播开销分析**：聚类操作的计算开销通常低于总计算成本的5%，传播机制虽然略有增加，但显著提升了生成质量。

**4. 论文局限性**
论文中未明确提及显著的局限性，但从方法描述和实验设置中可以推断出一些潜在的考虑：
*   **K值选择**：聚类数量K的设定需要平衡生成质量和计算开销。虽然论文指出K=16在DiT-XL/2上表现良好，但对于不同模型或任务，K值的最优选择可能需要进一步探索。
*   **传播比率γ的敏感性**：传播比率γ是一个关键超参数，影响空间重用和时间重用的平衡。过大或过小的γ值都可能导致性能下降，需要仔细调整。
*   **聚类算法的选择**：论文选择了K-Means算法，因其简单高效。但对于某些复杂的特征分布，其他更高级的聚类算法是否能带来进一步的性能提升值得探讨。
*   **通用性**：虽然论文强调ClusCa可以应用于任何扩散Transformer，但其在更广泛的模型架构、数据集和任务上的表现仍需进一步验证。

**5. 潜在未来研究方向**
*   **自适应聚类与传播**：探索更智能的自适应机制，根据不同的时间步、模型层或内容复杂性动态调整聚类数量K和传播比率γ，以进一步优化效率和质量。
*   **更复杂的空间相似性利用**：研究除了K-Means之外的其他聚类或图神经网络方法，以更精细地捕捉token之间的空间关系，从而实现更高效的特征重用。
*   **结合其他加速技术**：将ClusCa与步数优化（如DPM-Solver）、模型剪枝或量化等其他加速技术相结合，探索多维度加速策略的协同效应。
*   **理论分析与泛化性**：对ClusCa的误差累积机制进行更深入的理论分析，并研究其在不同扩散模型变体（如条件生成、多模态生成）上的泛化能力。
*   **硬件优化**：针对ClusCa的计算模式（聚类、选择、传播）进行特定的硬件加速优化，以进一步提升实际部署时的推理速度。

---

**Key Findings:**

- For instance, ClusCa
achieves 4.96x acceleration on FLUX with an ImageReward of 99.49%, surpassing
the original model by 0.51%.
- The code is available at
https://github.com/Shenyi-Z/Cache4Diffusion.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10312v1)
- [arXiv](https://arxiv.org/abs/2509.10312v1)

---

<a id='2509.10260v1'></a>
## [MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation](https://arxiv.org/abs/2509.10260v1)

**Authors:** Jia Wang, Jie Hu, Xiaoqi Ma, Hanghang Ma, Yanbing Zeng, Xiaoming Wei

**Published:** 2025-09-12

**Categories:** cs.CV

**Abstract:**

Text-to-image (T2I) generation has achieved remarkable progress in
instruction following and aesthetics. However, a persistent challenge is the
prevalence of physical artifacts, such as anatomical and structural flaws,
which severely degrade perceptual quality and limit application. Given the
diversity and complexity of these artifacts, a systematic and fine-grained
evaluation framework is required, which is lacking in current benchmarks. To
fill this gap, we introduce MagicMirror, a comprehensive framework for
artifacts assessment. We first establish a detailed taxonomy of generated image
artifacts. Guided by this taxonomy, we manually annotate MagicData340K, the
first human-annotated large-scale dataset of 340K generated images with
fine-grained artifact labels. Building on this dataset, we train MagicAssessor,
a Vision-Language Model (VLM) that provides detailed assessments and
corresponding labels. To overcome challenges like class imbalance and reward
hacking, we design a novel data sampling strategy and a multi-level reward
system for Group Relative Policy Optimization (GRPO). Finally, we leverage
MagicAssessor to construct MagicBench, an automated benchmark for evaluating
the image artifacts of current T2I models. Our evaluation with MagicBench
reveals that despite their widespread adoption, even top-tier models like
GPT-image-1 are consistently plagued by significant artifacts, highlighting
artifact reduction as a critical frontier for future T2I development. Project
page: https://wj-inf.github.io/MagicMirror-page/.

**Analysis:**

好的，以下是Jia Wang等人撰写的论文“MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation”的全面摘要：

**论文摘要：MagicMirror：用于文本到图像生成中细粒度伪影评估的大规模数据集和基准**

**1. 主要问题或研究问题：**
尽管文本到图像（T2I）生成在指令遵循和美学方面取得了显著进展，但物理伪影（如解剖和结构缺陷）的普遍存在严重降低了感知质量并限制了应用。现有基准缺乏系统且细粒度的评估框架来解决这些多样且复杂的伪影。

**2. 关键创新或方法论贡献：**
为了解决上述问题，作者引入了MagicMirror，一个用于伪影评估的综合框架，其主要贡献包括：
*   **详细的伪影分类法：** 首次建立了生成图像伪影的详细分类法，将伪影分为对象解剖、属性和交互三大类，并进一步细化为L2和L3级别标签。
*   **MagicData340K数据集：** 基于该分类法，手动标注了MagicData340K，这是首个包含34万张生成图像的、带有细粒度伪影标签的大规模人工标注数据集。
*   **MagicAssessor模型：** 训练了一个专门的视觉-语言模型（VLM），MagicAssessor，用于提供详细的伪影评估和相应标签。为了解决类别不平衡和奖励作弊等挑战，作者设计了一种新颖的数据采样策略和多级奖励系统，并将其应用于Group Relative Policy Optimization (GRPO)训练。
*   **MagicBench基准：** 利用MagicAssessor构建了MagicBench，这是一个用于评估当前T2I模型图像伪影的自动化基准。

**3. 主要结果及其意义：**
*   MagicBench的评估结果显示，即使是GPT-image-1等顶级T2I模型也普遍存在显著伪影，这表明伪影减少是未来T2I发展的关键前沿。
*   MagicAssessor在伪影检测任务上表现出色，在二元分类任务中实现了0.77的精确度和约0.7的F1分数，表明其作为奖励信号的强大潜力。
*   该模型在识别人类和动物解剖伪影方面表现突出，但在交互和对象形态问题上效果稍差。
*   与现有模型相比，MagicAssessor在所有主要评估指标上均显著优于竞争对手，填补了性能空白。

**4. 论文中提及的局限性：**
*   MagicAssessor在处理交互和对象形态问题时效果不佳。对于交互，模型难以区分元素重叠和低质量图像区域；对于对象形态，对象类型的多样性和庞大数量使得评估从根本上变得困难。
*   一些大型VLM模型（如Qwen-VL和InternVL系列）对伪影不敏感，召回率非常低，甚至更大版本表现更差，这可能与它们倾向于将图像视为正常有关。
*   GPT-40表现出不一致的性能，虽然能识别明显的解剖畸形，但对更细微或逻辑错误则会忽略，这可能与其提供通用描述的倾向有关。

**5. 潜在的未来研究方向：**
*   将MagicMirror框架直接整合到模型训练生命周期中，用于静态监控（在预训练阶段评估模型检查点）和主动优化（在后训练阶段作为奖励模型，通过RLHF或ReFL直接减少伪影生成）。
*   未来的模型开发需要在追求美学和减少伪影的关键需求之间取得平衡。

**Key Findings:**

- To
fill this gap, we introduce MagicMirror, a comprehensive framework for
artifacts assessment.
- To overcome challenges like class imbalance and reward
hacking, we design a novel data sampling strategy and a multi-level reward
system for Group Relative Policy Optimization (GRPO).

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10260v1)
- [arXiv](https://arxiv.org/abs/2509.10260v1)

---

<a id='2509.10156v1'></a>
## [LayerLock: Non-collapsing Representation Learning with Progressive Freezing](https://arxiv.org/abs/2509.10156v1)

**Authors:** Goker Erdogan, Nikhil Parthasarathy, Catalin Ionescu, Drew Hudson, Alexander Lerchner, Andrew Zisserman, Mehdi Sajjadi, Joao Carreira

**Published:** 2025-09-12

**Categories:** cs.CV

**Abstract:**

We introduce LayerLock, a simple yet effective approach for self-supervised
visual representation learning, that gradually transitions from pixel to latent
prediction through progressive layer freezing. First, we make the observation
that during training of video masked-autoencoding (MAE) models, ViT layers
converge in the order of their depth: shallower layers converge early, deeper
layers converge late. We then show that this observation can be exploited to
accelerate standard MAE by progressively freezing the model according to an
explicit schedule, throughout training. Furthermore, this same schedule can be
used in a simple and scalable approach to latent prediction that does not
suffer from "representation collapse". We apply our proposed approach,
LayerLock, to large models of up to 4B parameters with results surpassing those
of non-latent masked prediction on the 4DS perception suite.

**Analysis:**

以下是Goker Erdogan等人撰写的论文“LayerLock: Non-collapsing Representation Learning with Progressive Freezing”的摘要：

**1. 主要问题或研究问题**
该论文旨在解决自监督视觉表示学习中的效率和稳定性问题，特别是视频掩码自编码（MAE）模型在训练过程中层收敛顺序不一致以及潜在预测方法中常见的“表示崩溃”问题。研究人员观察到，在视频MAE模型训练中，ViT层按照深度顺序收敛：浅层先收敛，深层后收敛。如何利用这一观察结果来加速训练、提高表示学习的稳定性，并避免表示崩溃是核心问题。

**2. 关键创新或方法论贡献**
LayerLock提出了一个简单而有效的自监督视觉表示学习方法，其主要创新点包括：
*   **渐进式层冻结（Progressive Layer Freezing）**：利用ViT层按深度顺序收敛的观察，LayerLock在训练过程中根据预设的时间表逐步冻结模型中的浅层，从而加速标准MAE的训练。
*   **动态目标预测（Dynamically Evolving Prediction Target）**：LayerLock在训练过程中动态地将预测目标从浅层特征（如像素）过渡到更深层的中间潜在模型激活。这结合了像素预测的稳定性（避免崩溃）和潜在预测学习抽象语义特征的能力。
*   **避免表示崩溃**：通过渐进式冻结和动态目标预测，LayerLock提供了一种简单且可扩展的潜在预测方法，有效避免了传统潜在预测方法中常见的表示崩溃问题。
*   **新型3D旋转位置嵌入（3D Rotary Positional Embeddings）**：论文引入了一种新颖的3D旋转位置嵌入方法，显著提高了所有下游任务的性能。

**3. 主要结果及其意义**
*   **训练效率提升**：LayerLock通过渐进式冻结，在不损失性能的情况下，显著降低了MAE模型的总训练成本和峰值内存使用量。在1B训练样本上，FLOP效率提升高达19%。
*   **表示学习性能提升**：LayerLock在像素预测（如4DS MAE）和潜在预测（如V-JEPA）两种自监督方法上都取得了优于基线模型的性能。在动作分类（SSv2和Kinetics700）和深度估计（ScanNet）等语义和低级视觉任务上均有显著改进。
*   **稳定性与可扩展性**：LayerLock方法在训练大型视频模型（高达4B参数）时表现出高度稳定性，且不出现表示崩溃问题。
*   **3D旋转位置嵌入的有效性**：实验证明，新引入的3D旋转位置嵌入独立于LayerLock，能为基线模型和LayerLock模型带来性能提升，例如在SSv2分类任务中提升2.5%。

**4. 论文中提及的局限性**
*   **潜在损失的效率权衡**：在计算潜在损失时，使用少量（例如5%）的补丁虽然仍优于基线，但相比使用全部补丁，在深度估计性能上略有下降。这表明在效率和性能之间存在权衡，需要进一步探索。
*   **冻结调度参数敏感性**：冻结开始步数、冻结间隔和每次冻结的层数（layer jump）等参数对下游任务性能有显著影响，需要仔细调整。例如，过早冻结或一次冻结过多层会导致性能下降。

**5. 潜在的未来研究方向**
*   **扩展到更长视频、更高分辨率和更深模型**：渐进式冻结节省的计算和内存为将自监督学习扩展到更长视频、更大分辨率和更深的模型开辟了新途径。
*   **更精细的效率与性能权衡**：进一步探索潜在损失计算中补丁子选择的效率与性能之间的关系。
*   **优化冻结调度**：深入研究和优化冻结调度参数，以实现最佳性能和效率。
*   **多目标预测的探索**：尽管目前LayerLock的单目标预测方法表现良好，但未来可以继续探索在训练过程中同时预测多个目标（像素和中间层）的潜在益处。

**Key Findings:**

- We introduce LayerLock, a simple yet effective approach for self-supervised
visual representation learning, that gradually transitions from pixel to latent
prediction through progressive layer freezing.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10156v1)
- [arXiv](https://arxiv.org/abs/2509.10156v1)

---

<a id='2509.10105v1'></a>
## [VARCO-VISION-2.0 Technical Report](https://arxiv.org/abs/2509.10105v1)

**Authors:** Young-rok Cha, Jeongho Ju, SunYoung Park, Jong-Hyeon Lee, Younghyun Yu, Youngjune Kim

**Published:** 2025-09-12

**Categories:** cs.CV, cs.CL

**Abstract:**

We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model
(VLM) for Korean and English with improved capabilities compared to the
previous model VARCO-VISION-14B. The model supports multi-image understanding
for complex inputs such as documents, charts, and tables, and delivers
layoutaware OCR by predicting both textual content and its spatial location.
Trained with a four-stage curriculum with memory-efficient techniques, the
model achieves enhanced multimodal alignment, while preserving core language
abilities and improving safety via preference optimization. Extensive benchmark
evaluations demonstrate strong spatial grounding and competitive results for
both languages, with the 14B model achieving 8th place on the OpenCompass VLM
leaderboard among models of comparable scale. Alongside the 14B-scale model, we
release a 1.7B version optimized for on-device deployment. We believe these
models advance the development of bilingual VLMs and their practical
applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a
full-scale 14B model and a lightweight 1.7B model.

**Analysis:**

好的，这是一篇关于“VARCO-VISION-2.0 Technical Report”的全面摘要，由Young-rok Cha等人撰写：

**论文摘要：VARCO-VISION-2.0 技术报告**

**1. 解决的主要问题或研究问题**
该论文旨在解决现有视觉-语言模型（VLM）在处理多图像输入、韩语本地化任务以及在保持核心语言能力和安全性的同时提升多模态对齐方面的局限性。具体来说，它致力于开发一个开源、双语（韩语和英语）VLM，能够更准确、更忠实地理解图像和文本，并支持布局感知的光学字符识别（OCR）。

**2. 关键创新或方法论贡献**
*   **多图像理解支持：** VARCO-VISION-2.0 能够同时分析多张图像，从而实现更全面、上下文感知的决策，有效处理文档、图表和表格等复杂视觉内容。
*   **韩语语言专业化：** 模型对韩语的理解、上下文和文化有更深层次的把握，显著提高了韩语文本生成的自然度、流畅性和准确性。
*   **带文本定位的OCR：** 模型能够识别文本位置并提供边界框，这对于文档理解、标牌解释和结构化视觉数据特别有用。
*   **增强的安全性：** 通过偏好优化（preference optimization），模型改进了对有害或露骨内容的处理，确保了更安全可靠的交互。
*   **四阶段课程训练策略：** 模型采用四阶段课程训练，结合内存高效技术，实现了增强的多模态对齐，同时保留了核心语言能力。
*   **模型架构：** 基于LLaVA-OneVision架构，采用Qwen3作为LLM，SigLIP2（patch-16配置）作为视觉编码器，并通过两层MLP连接器将图像特征投影到LLM的嵌入空间。
*   **1.7B模型初始化策略：** 1.7B模型的视觉编码器权重从经过第三阶段训练的14B模型中初始化，促进了知识迁移并加速了收敛。
*   **模型合并策略（针对14B模型）：** 采用“合并-训练-合并”策略，通过合并多个第三阶段检查点来获得稳健的第四阶段初始化器，并在第四阶段完成后再次合并检查点以生成最终模型，从而减少检查点方差并聚合不同模式。

**3. 主要结果及其意义**
*   **强大的空间定位能力：** 广泛的基准评估表明，模型在空间定位方面表现出色，并在韩语和英语两种语言上都取得了有竞争力的结果。
*   **OpenCompass VLM排行榜表现：** 14B模型在OpenCompass VLM排行榜上，在同等规模的模型中排名第8。
*   **OCR性能：** VARCO-VISION-2.0在CORD、ICDAR2013和ICDAR2015等OCR基准测试中表现出显著优于流行开源OCR系统（如PaddleOCR和EasyOCR）的性能，并与商业系统CLOVA OCR具有竞争力。这表明其强大的视觉-文本对齐能力。
*   **文本专用任务表现：** 全尺寸和轻量级VARCO-VISION-2.0模型在各种文本专用任务（包括通用知识、指令遵循和多轮推理）上均取得持续高分，表明其语言能力在训练阶段得到了有效保持。
*   **轻量级版本：** 除了14B模型外，还发布了针对设备部署优化的1.7B轻量级版本，为实际应用提供了实用选择。

**4. 论文中提到的局限性**
*   **指令鲁棒性不足：** 模型的输出对表面格式变化（如空格、换行符）敏感，表明过度依赖固定提示模板和对多样化指令格式的接触不足。
*   **知识和文档理解不足：** 尽管在感知和空间推理方面表现强劲，但模型在知识密集型和文档中心任务上表现不佳，这归因于知识来源的有限纳入和训练期间缺乏稳健的布局感知监督。
*   **指代能力减弱：** 与之前的VARCO-VISION模型相比，模型在指代任务上的性能有所下降。这可能限制了模型在涉及精确对象选择或基于视觉上下文的指令遵循的实际应用中的性能。
*   **推理能力：** 即使明确指示遵循多步推理，模型也常常对其初始预测保持自信，很少改变输出，表明推理提示（如“让我们一步步思考”）并未显著提高准确性。
*   **文化理解：** 在韩语文化基准测试中，模型表现仍不如其他领先模型，这可能与韩语文化相关训练数据的相对有限性有关。

**5. 潜在的未来研究方向**
*   **小模型蒸馏：** 探索教师-学生训练方案，利用大模型作为教师，对小模型进行知识蒸馏。
*   **推理能力改进：** 通过强化学习激励推理行为，例如链式思考蒸馏和基于偏好的微调，以增强多步推理能力。
*   **高效上下文处理：** 采用YaRN等技术进行高效上下文扩展，并利用序列并行性减少内存开销，以支持高分辨率多图像输入和长时序视频理解。
*   **扩展到视频模态及其他：** 增强模型对长时序视频的理解能力，支持3D自由视角视频和可控相机轨迹，并开发整合音频和语音等模态的全模态模型。
*   **迈向具身多模态智能体：** 逐步开发能够感知和行动的具身智能体，包括与GUI交互和操纵屏幕环境。
*   **模型和数据规模扩展：** 扩大模型容量和数据量，采用高效架构设计，扩展高质量、多样化监督的多模态训练语料库，并通过改进奖励信号和可扩展算法优化偏好学习。

总而言之，VARCO-VISION-2.0代表了双语VLM领域的重大进展，特别是在韩语和英语的视觉-语言理解、OCR和多图像处理方面。尽管存在一些局限性，但其开源发布和强大的性能使其成为构建实用多模态系统的有前景的基础。

**Key Findings:**

- We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model
(VLM) for Korean and English with improved capabilities compared to the
previous model VARCO-VISION-14B.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10105v1)
- [arXiv](https://arxiv.org/abs/2509.10105v1)

---

<a id='2509.10080v1'></a>
## [BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals](https://arxiv.org/abs/2509.10080v1)

**Authors:** Minsang Kong, Myeongjun Kim, Sang Gu Kang, Sang Hun Lee

**Published:** 2025-09-12

**Categories:** cs.CV, I.2.9; I.4.8

**Abstract:**

In autonomous driving, trajectory prediction is essential for ensuring safe
and efficient navigation. To improve prediction accuracy, recent approaches
often rely on pre-built high-definition (HD) maps or real-time local map
construction modules to incorporate static environmental information. However,
pre-built HD maps are limited to specific regions and cannot adapt to transient
changes. In addition, local map construction modules, which recognize only
predefined elements, may fail to capture critical scene details or introduce
errors that degrade prediction performance. To overcome these limitations, we
propose Bird's-Eye View Trajectory Prediction (BEVTraj), a novel trajectory
prediction framework that operates directly in the bird's-eye view (BEV) space
utilizing real-time sensor data without relying on any pre-built maps. The
BEVTraj leverages deformable attention to efficiently extract relevant context
from dense BEV features. Furthermore, we introduce a Sparse Goal Candidate
Proposal (SGCP) module, which enables full end-to-end prediction without
requiring any post-processing steps. Extensive experiments demonstrate that the
BEVTraj achieves performance comparable to state-of-the-art HD map-based models
while offering greater flexibility by eliminating the dependency on pre-built
maps. The source code is available at https://github.com/Kongminsang/bevtraj.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Minsang Kong, Myeongjun Kim, Sang Gu Kang, Sang Hun Lee撰写的论文“BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals”的全面摘要。

---

**论文摘要：BEVTraj: 基于鸟瞰图、可变形注意力与稀疏目标提议的无地图端到端轨迹预测**

**1. 主要问题或研究问题：**
自动驾驶中的轨迹预测对于安全高效的导航至关重要。现有方法通常依赖预构建的高清（HD）地图或实时局部地图构建模块来整合静态环境信息。然而，预构建的HD地图覆盖范围有限且无法适应瞬态变化，而局部地图构建模块可能因仅识别预定义元素而遗漏关键场景细节或引入错误，从而降低预测性能。这篇论文旨在解决这些限制，提出一种不依赖预构建地图，直接利用实时传感器数据进行轨迹预测的方法。

**2. 关键创新或方法论贡献：**
BEVTraj框架引入了以下关键创新：
*   **无地图的鸟瞰图（BEV）空间操作：** BEVTraj是一种新颖的、目标驱动的轨迹预测框架，它直接利用从原始传感器数据构建的BEV表示进行操作，从而消除了对预构建HD地图的依赖。这使得系统能够适应瞬态变化，并减少信息损失。
*   **可变形注意力（Deformable Attention）：** BEVTraj利用可变形注意力机制，从密集的BEV特征中高效提取相关上下文信息。这种机制允许模型选择性地关注BEV特征图中的关键空间位置，从而在密集和冗余的BEV表示中提高效率和预测准确性。
*   **稀疏目标候选提议（Sparse Goal Candidate Proposal, SGCP）模块：** 引入SGCP模块以解决现有基于目标的方法对目标候选密度敏感的问题。SGCP模块能够生成稀疏的目标候选集，这些候选集基于目标智能体的动态状态和BEV特征图进行条件化，从而实现完全端到端的预测，无需非极大值抑制（NMS）等后处理步骤。
*   **迭代可变形解码器：** 该解码器通过迭代细化过程，基于BEV特征和场景上下文特征预测并改进目标智能体的多模态未来轨迹。

**3. 主要结果及其意义：**
*   **性能可与HD地图模型媲美：** 广泛的实验表明，BEVTraj在轨迹预测性能上达到了与最先进的基于HD地图的模型相当的水平，甚至在某些指标（如Miss Rate）上表现更优。这证明了无地图轨迹预测的可行性。
*   **增强的灵活性：** 通过消除对预构建地图的依赖，BEVTraj提供了更大的灵活性，能够适应未映射区域和动态道路条件。
*   **对复杂场景的鲁棒性：** BEVTraj在急转弯和遮挡交叉口等复杂条件下，能够生成合理且与车道对齐的未来轨迹，这得益于其直接从原始传感器数据中提取细粒度视觉线索的能力。
*   **与BEV范式兼容：** 该框架与采用BEV范式的现代自动驾驶系统完全兼容，便于集成到现有管道中。

**4. 论文中提及的局限性：**
*   **传感器数据范围限制：** 传感器数据固有的感知范围限制了BEVTraj的性能，尤其是在需要更广阔空间上下文的情况下。虽然可变形注意力有助于选择性地关注信息区域，但传感器范围的物理限制仍然存在。
*   **实时HD地图构建的准确性：** 论文虽然隔离了地图范围的影响，但实际部署中实时HD地图构建的准确性可变性仍需考虑，这可能在覆盖范围增加时恶化。

**5. 潜在的未来研究方向：**
*   **整合BEV特征的目标检测与跟踪：** 将BEVTraj扩展，纳入基于BEV特征的目标检测和跟踪模块，以在统一的表示空间中实现静态和动态场景信息的联合处理。
*   **增强BEV特征的时间建模能力：** 通过循环架构或跨扫描融合技术，进一步提高BEV特征的时间建模能力，以更好地捕捉长期运动模式。
*   **自适应BEV网格结构：** 研究自适应BEV网格结构，以缓解空间限制，实现更灵活和细粒度的轨迹预测。
*   **扩展到运动规划：** 将该框架自然地扩展到运动规划应用，作为连接感知和规划的统一解决方案。
*   **其他领域应用：** 将所开发的技术应用于更广泛的领域，如监控系统（车辆和人类活动监测）和机器人技术（人机交互和动态环境导航）。

---

**Key Findings:**

- To overcome these limitations, we
propose Bird's-Eye View Trajectory Prediction (BEVTraj), a novel trajectory
prediction framework that operates directly in the bird's-eye view (BEV) space
utilizing real-time sensor data without relying on any pre-built maps.
- Furthermore, we introduce a Sparse Goal Candidate
Proposal (SGCP) module, which enables full end-to-end prediction without
requiring any post-processing steps.
- Extensive experiments demonstrate that the
BEVTraj achieves performance comparable to state-of-the-art HD map-based models
while offering greater flexibility by eliminating the dependency on pre-built
maps.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10080v1)
- [arXiv](https://arxiv.org/abs/2509.10080v1)

---

<a id='2509.10026v1'></a>
## [LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward Optimization for Real-World Multilingual VQA](https://arxiv.org/abs/2509.10026v1)

**Authors:** Jing Huang, Zhiya Tan, Shutao Gong, Fanwei Zeng, Jianshu Li

**Published:** 2025-09-12

**Categories:** cs.CV

**Abstract:**

As large vision language models (VLMs) advance, their capabilities in
multilingual visual question answering (mVQA) have significantly improved.
Chain-of-thought (CoT) reasoning has been proven to enhance interpretability
and complex reasoning. However, most existing approaches rely primarily on
textual CoT and provide limited support for multilingual multimodal reasoning,
constraining their deployment in real-world applications. To address this gap,
we introduce \textbf{LaV-CoT}, the first Language-aware Visual CoT framework
with Multi-Aspect Reward Optimization. LaV-CoT incorporates an interpretable
multi-stage reasoning pipeline consisting of Text Summary with Bounding Box
(BBox), Language Identification, Spatial Object-level Captioning, and
Step-by-step Logical Reasoning. Following this reasoning pipeline, we design an
automated data curation method that generates multilingual CoT annotations
through iterative generation, correction, and refinement, enabling scalable and
high-quality training data. To improve reasoning and generalization, LaV-CoT
adopts a two-stage training paradigm combining Supervised Fine-Tuning (SFT)
with Language-aware Group Relative Policy Optimization (GRPO), guided by
verifiable multi-aspect rewards including language consistency, structural
accuracy, and semantic alignment. Extensive evaluations on public datasets
including MMMB, Multilingual MMBench, and MTVQA show that LaV-CoT achieves up
to \(\sim\)9.5\% accuracy improvements over open-source baselines of similar
size and even surpasses models with 2$\times$ larger scales by \(\sim\)2.6\%.
Moreover, LaV-CoT outperforms advanced proprietary models such as GPT-4o-0513
and Gemini-2.5-flash. We further conducted an online A/B test to validate our
method on real-world data, highlighting its effectiveness for industrial
deployment. Our code is available at this link:
\href{https://github.com/HJNVR/LaV-CoT}

**Analysis:**

好的，这是一篇关于“LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward Optimization for Real-World Multilingual VQA”论文的全面摘要，使用中文书写：

**论文摘要：LaV-CoT: 面向真实世界多语言VQA的语言感知视觉CoT与多方面奖励优化**

**1. 主要问题或研究问题：**
随着大型视觉语言模型（VLMs）在多语言视觉问答（mVQA）方面能力的提升，链式思考（CoT）推理被证明能增强可解释性和复杂推理能力。然而，现有方法主要依赖文本CoT，对多语言多模态推理的支持有限，这限制了它们在真实世界应用中的部署。具体挑战包括：(i) 语言不一致性，(ii) 视觉-文本错位（视觉内容与翻译文本的接地不足），以及(iii) 有限的多语言视觉推理能力，尤其是在需要复杂跨语言、多模态推理的任务中。

**2. 关键创新或方法论贡献：**
为解决上述问题，论文提出了**LaV-CoT**，这是首个结合多方面奖励优化的语言感知视觉CoT框架。其主要创新和贡献包括：

*   **可解释的多阶段推理管道：** LaV-CoT引入了一个由四个关键组件组成的多阶段推理管道：(1) 带有边界框（BBox）的文本摘要，(2) 语言识别，(3) 空间对象级图像描述，以及(4) 逐步逻辑推理。这种结构化管道明确地解耦了语言和视觉推理，实现了细粒度的跨模态对齐，并提高了多语言环境下的可解释性。
*   **自动化数据生成方法：** 针对高质量多语言推理数据构建的挑战，论文设计了一种自动数据生成方法，通过迭代生成、纠正和细化来创建多语言CoT标注。这确保了可扩展地生成结构化、可验证的高质量训练数据，同时捕捉语言保真度和多模态推理质量。
*   **两阶段训练范式与语言感知群组相对策略优化（GRPO）：** LaV-CoT采用两阶段训练范式，结合了监督微调（SFT）和语言感知群组相对策略优化（GRPO）。GRPO由可验证的多方面奖励指导，包括语言一致性奖励、文本段和对象计数奖励、最终答案编辑距离奖励和格式奖励，从而实现稳定的优化和跨语言、跨模态的鲁棒推理。

**3. 主要结果及其意义：**
广泛的实验评估证明了LaV-CoT的有效性：

*   在MMMB、Multilingual MMBench和MTVQA等公共数据集上，LaV-CoT（SFT + GRPO变体）相比同等规模的开源基线模型，准确率提高了约9.5%，甚至超越了规模大2倍的模型约2.6%。
*   LaV-CoT在特定多语言设置（如阿拉伯语、土耳其语、韩语等）上，表现优于GPT-4o-0513和Gemini-2.5-flash等先进专有模型。
*   在线A/B测试进一步验证了该方法在真实世界数据上的有效性，显示其在工业部署和商业应用中的潜力，显著提高了答案接受率和用户满意度。
*   GRPO训练的引入显著提升了LaV-CoT的性能，尤其是在跨语言推理方面。

**4. 论文中提及的局限性：**
尽管取得了可喜的成果，LaV-CoT仍存在一些局限性：

*   **对多语言输入质量的敏感性：** LaV-CoT的有效性取决于多模态输入的质量。当文档图像包含大量跨脚本的语言混合时，推理管道可能难以保持语言一致性和语义保真度。
*   **低资源语言的覆盖范围：** 当前训练主要依赖于侧重于中高资源语言的开源多语言VQA数据集。真正低资源或不常见语言的高质量数据集仍然稀缺，其收集和构建对于更广泛的包容性至关重要。
*   **快慢推理整合的探索有限：** LaV-CoT目前设计为慢速、多步骤推理管道以增强可解释性。虽然有效，但尚未整合快速思考策略或混合快慢推理机制，这将在未来工作中进一步探索以提高效率和适应性。

**5. 潜在的未来研究方向：**
未来工作计划将LaV-CoT扩展到更广泛的低资源语言和特定领域应用，并进一步探索先进的奖励建模，以增强多语言多模态推理系统的鲁棒性和包容性。

**Key Findings:**

- To address this gap,
we introduce \textbf{LaV-CoT}, the first Language-aware Visual CoT framework
with Multi-Aspect Reward Optimization.
- Moreover, LaV-CoT outperforms advanced proprietary models such as GPT-4o-0513
and Gemini-2.5-flash.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.10026v1)
- [arXiv](https://arxiv.org/abs/2509.10026v1)

---

