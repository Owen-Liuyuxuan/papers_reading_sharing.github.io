time: 20250919

# Arxiv Computer Vision Papers - 2025-09-19

## Executive Summary

## Arxiv 计算机视觉每日报告执行摘要 (2025-09-17)

**概述：**

今天的 Arxiv 计算机视觉论文主要围绕**多模态学习、具身智能（特别是机器人操作）、3D 视觉以及对模型鲁棒性和公平性的关注**。大型语言模型（LLMs）在视觉任务中的应用持续深化，尤其是在零样本学习和知识注入方面。

**主要主题和趋势：**

1.  **多模态LLMs的融合与应用：** 多个工作探索了多模态LLMs（MLLMs）在视频理解（零样本时空视频定位）、合成图像检测以及图像检索中的潜力，通过提示引导知识注入和思维链重排序等技术提升性能。
2.  **具身智能与机器人操作：** 机器人操作和感知是显著主题，包括利用多视图扩散策略进行鲁棒移动操作、通过3D几何关键点匹配增强2D对象识别，以及构建跨平台数据以扩展计算机使用代理。
3.  **3D 视觉与重建：** 3D 视觉领域有进展，如利用置信度感知扩散模型实现轻量级和准确的多视图立体，以及在跟踪、融合和预测中处理“视线外”轨迹。
4.  **模型鲁棒性与公平性：** 有论文关注模型在特定场景下的鲁棒性（如合成图像检测）和公平性（如消除人脸老化模型中的种族偏见）。
5.  **大规模模型与数据：** SAIL-VL2 和 ScaleCUA 等工作表明了构建和扩展大规模视觉语言模型及跨平台数据的持续努力。

**特别重要或创新的论文：**

*   **"Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding" (Zaiquan Yang et al.)：** 这篇论文展示了MLLMs在复杂视频理解任务（零样本时空视频定位）中的强大能力，预示了MLLMs在更高级别视频推理中的巨大潜力。
*   **"M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation" (Ju Dong et al.)：** 将多视图扩散模型与可操作性感知控制相结合，为机器人鲁棒移动操作提供了新颖且高效的解决方案，对具身智能领域具有重要意义。
*   **"Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model" (Fangjinhua Wang et al.)：** 引入置信度感知扩散模型来提升多视图立体的准确性和效率，为3D重建领域提供了一个有前景的轻量级方法。
*   **"DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection" (Zhuokang Shen et al.)：** 巧妙地利用提示引导的知识注入来增强MLLM在合成图像检测这一关键安全任务上的能力，展示了MLLM在特定领域应用中的灵活性。

**新兴研究方向或技术：**

*   **MLLMs在零样本和少样本视频理解中的深化应用：** 尤其是时空定位和复杂事件推理。
*   **扩散模型在3D视觉和机器人控制中的多功能性：** 不仅用于生成，还用于提高重建精度和策略学习的鲁棒性。
*   **具身智能中跨平台数据和通用代理的构建：** 旨在实现更广泛、更通用的机器人能力。
*   **通过提示工程和知识注入来定制和增强MLLMs：** 以解决特定下游任务，如检测虚假信息。
*   **对模型公平性和偏见的持续关注：** 尤其是在敏感应用如人脸识别中。

**建议阅读全文的论文：**

对于不同兴趣的研究人员，建议阅读以下论文：

*   **对多模态LLMs和视频理解感兴趣：** "Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding" (Zaiquan Yang et al.)
*   **对机器人操作和具身智能感兴趣：** "M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation" (Ju Dong et al.)
*   **对3D重建和扩散模型感兴趣：** "Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model" (Fangjinhua Wang et al.)
*   **对MLLMs在安全和检测任务中的应用感兴趣：** "DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection" (Zhuokang Shen et al.)
*   **对大规模模型和数据扩展感兴趣：** "SAIL-VL2 Technical Report" (Weijie Yin et al.) 和 "ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data" (Zhaoyang Liu et al.)

这份摘要旨在帮助您快速把握今日Arxiv计算机视觉领域的关键进展，为您的研究提供有价值的参考。

---

## Table of Contents

1. [SAIL-VL2 Technical Report](#2509.14033v1)
2. [ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data](#2509.15221v1)
3. [Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model](#2509.15220v1)
4. [Out-of-Sight Trajectories: Tracking, Fusion, and Prediction](#2509.15219v1)
5. [Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding](#2509.15178v1)
6. [A Race Bias Free Face Aging Model for Reliable Kinship Verification](#2509.15177v1)
7. [M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation](#2509.14980v1)
8. [RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching](#2509.14966v1)
9. [DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection](#2509.14957v1)
10. [Chain-of-Thought Re-ranking for Image Retrieval Tasks](#2509.14746v1)

---

## Papers

<a id='2509.14033v1'></a>
## [SAIL-VL2 Technical Report](https://arxiv.org/abs/2509.14033v1)

**Authors:** Weijie Yin, Yongjie Ye, Fangxun Shu, Yue Liao, Zijian Kang, Hongyuan Dong, Haiyang Yu, Dingkang Yang, Jiacong Wang, Han Wang, Wenzhuo Liu, Xiao Liang, Shuicheng Yan, Chao Feng

**Published:** 2025-09-17

**Categories:** cs.CV

**Abstract:**

We introduce SAIL-VL2, an open-suite vision-language foundation model (LVM)
for comprehensive multimodal understanding and reasoning. As the successor to
SAIL-VL, SAIL-VL2 achieves state-of-the-art performance at the 2B and 8B
parameter scales across diverse image and video benchmarks, demonstrating
strong capabilities from fine-grained perception to complex reasoning. Three
core innovations drive its effectiveness. First, a large-scale data curation
pipeline with scoring and filtering strategies enhances both quality and
distribution across captioning, OCR, QA, and video data, improving training
efficiency. Second, a progressive training framework begins with a powerful
pre-trained vision encoder (SAIL-ViT), advances through multimodal
pre-training, and culminates in a thinking-fusion SFT-RL hybrid paradigm that
systematically strengthens model capabilities. Third, architectural advances
extend beyond dense LLMs to efficient sparse Mixture-of-Experts (MoE) designs.
With these contributions, SAIL-VL2 demonstrates competitive performance across
106 datasets and achieves state-of-the-art results on challenging reasoning
benchmarks such as MMMU and MathVista. Furthermore, on the OpenCompass
leaderboard, SAIL-VL2-2B ranks first among officially released open-source
models under the 4B parameter scale, while serving as an efficient and
extensible foundation for the open-source multimodal community.

**Analysis:**

好的，这是一篇关于Weijie Yin等人在计算机视觉和机器学习领域撰写的论文“SAIL-VL2 Technical Report”的全面摘要。

**论文摘要：SAIL-VL2技术报告**

**1. 主要问题或研究问题：**
该论文旨在解决当前视觉-语言模型（LVM）在实现全面多模态理解和推理方面的挑战。尽管现有LVM在性能上有所提升，但仍面临计算效率、训练成本以及在细粒度感知和复杂推理任务中的局限性。SAIL-VL2的目标是开发一个高效、可扩展的开源LVM，能够实现最先进的性能，并推动多模态人工智能（AGI）的发展。

**2. 关键创新或方法论贡献：**
SAIL-VL2的有效性主要由以下三项核心创新驱动：

*   **大规模数据整理流水线：** 论文设计了一个全面的数据整理流水线，通过评分和过滤策略，提升了图像、视频、OCR和问答数据的质量和分布。这显著提高了训练效率，并确保模型能够从高质量、多样化的数据中学习。
*   **渐进式训练框架：** 训练过程分为三个阶段：
    *   首先，使用强大的预训练视觉编码器（SAIL-ViT）进行热身适应，将视觉输出粗粒度地适应到LLM域。
    *   其次，通过多模态预训练进行细粒度对齐，解锁视觉编码器和适配器，以实现更深层次的对齐。
    *   最后，通过思维融合（Thinking-Fusion）SFT-RL（监督微调-强化学习）混合范式，系统性地强化模型能力，使其能够进行复杂推理。
*   **高效的稀疏混合专家（MoE）架构：** SAIL-VL2超越了传统的密集LLM，采用了更高效的稀疏MoE设计。这使得模型能够在保持计算效率的同时，实现参数规模的扩展，并确保专家激活的平衡性和稳定性。

**3. 主要结果及其意义：**
SAIL-VL2在多个维度上展示了强大的性能：

*   **最先进的性能：** SAIL-VL2在2B和8B参数规模下，跨106个数据集实现了最先进的性能，涵盖了图像和视频基准测试，从细粒度感知到复杂推理都表现出强大的能力。
*   **卓越的推理能力：** 在MMMU和MathVista等挑战性推理基准测试中，SAIL-VL2取得了最先进的结果。特别是，SAIL-VL2-Thinking模型在OpenCompass多模态推理基准测试中取得了领先结果，甚至超越了Gemini-2.0-Flash等强大的闭源模型。
*   **高效与可扩展性：** SAIL-VL2-2B在OpenCompass排行榜上，在4B参数规模以下的开源模型中排名第一，证明了其作为高效且可扩展的多模态社区基础模型的潜力。
*   **细粒度视觉理解：** 在OCR、高分辨率文档布局分析和复杂图表解释等任务中，SAIL-VL2表现出高保真度的感知能力，实现了超越同等规模模型的详细视觉基础。

**4. 论文中提及的局限性：**
论文中没有明确提及SAIL-VL2的显著局限性。然而，在数据整理部分，作者提到合成数据可能会引入语言表达上的分布偏差，因为LLM倾向于产生同质化的措辞，但他们发现模型仍能从大规模合成数据中受益。此外，在RL阶段，对于无法确定性验证真实答案的任务，需要LVM作为判断模型来提供奖励信号，这可能引入一定程度的依赖性或潜在偏差。

**5. 潜在的未来研究方向：**
论文指出，未来将继续通过以下方式进一步增强SAIL-VL系列：

*   **更高效的架构：** 持续探索和开发更高效的模型架构。
*   **全面的预训练策略：** 进一步优化和完善预训练策略。
*   **改进的强化学习范式：** 持续改进强化学习方法，以实现模型能力的持续演进。

总而言之，SAIL-VL2代表了高效大型视觉-语言模型设计上的全面进步，通过在架构、训练策略和数据整理方面的创新，实现了效率和性能的持续提升，为开源多模态社区奠定了坚实的基础。

**Key Findings:**

- We introduce SAIL-VL2, an open-suite vision-language foundation model (LVM)
for comprehensive multimodal understanding and reasoning.
- As the successor to
SAIL-VL, SAIL-VL2 achieves state-of-the-art performance at the 2B and 8B
parameter scales across diverse image and video benchmarks, demonstrating
strong capabilities from fine-grained perception to complex reasoning.
- With these contributions, SAIL-VL2 demonstrates competitive performance across
106 datasets and achieves state-of-the-art results on challenging reasoning
benchmarks such as MMMU and MathVista.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.14033v1)
- [arXiv](https://arxiv.org/abs/2509.14033v1)

---

<a id='2509.15221v1'></a>
## [ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data](https://arxiv.org/abs/2509.15221v1)

**Authors:** Zhaoyang Liu, JingJing Xie, Zichen Ding, Zehao Li, Bowen Yang, Zhenyu Wu, Xuehui Wang, Qiushi Sun, Shi Liu, Weiyun Wang, Shenglong Ye, Qingyun Li, Zeyue Tian, Gen Luo, Xiangyu Yue, Biqing Qi, Kai Chen, Bowen Zhou, Yu Qiao, Qifeng Chen, Wenhai Wang

**Published:** 2025-09-18

**Categories:** cs.CV

**Abstract:**

Vision-Language Models (VLMs) have enabled computer use agents (CUAs) that
operate GUIs autonomously, showing great potential, yet progress is limited by
the lack of large-scale, open-source computer use data and foundation models.
In this work, we introduce ScaleCUA, a step toward scaling open-source CUAs. It
offers a large-scale dataset spanning 6 operating systems and 3 task domains,
built via a closed-loop pipeline uniting automated agents with human experts.
Trained on this scaled-up data, ScaleCUA can operate seamlessly across
platforms. Specifically, it delivers strong gains over baselines (+26.6 on
WebArena-Lite-v2, +10.7 on ScreenSpot-Pro) and sets new state-of-the-art
results (94.4% on MMBench-GUI L1-Hard, 60.6% on OSWorld-G, 47.4% on
WebArena-Lite-v2). These findings underscore the power of data-driven scaling
for general-purpose computer use agents. We will release data, models, and code
to advance future research: https://github.com/OpenGVLab/ScaleCUA.

**Analysis:**

以下是关于Zhaoyang Liu等人撰写的论文“ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data”的摘要：

**1. 主要问题或研究问题：**
该论文旨在解决当前计算机使用代理（CUAs）发展面临的核心挑战：缺乏大规模、开源的计算机使用数据和基础模型，这限制了视觉语言模型（VLMs）在自主操作图形用户界面（GUIs）方面的潜力。现有数据收集成本高昂、难以扩展且通用性差。

**2. 关键创新或方法论贡献：**
*   **大规模跨平台数据集：** 引入ScaleCUA数据集，涵盖Windows、macOS、Linux、Android、iOS和Web六大操作系统，以及GUI理解、GUI定位和任务完成三大任务领域。
*   **闭环数据收集管道：** 采用自动化代理与人类专家相结合的闭环管道，高效收集高质量的原始轨迹数据，包括屏幕截图和结构化元数据，并进行标注和增强。
*   **统一动作空间：** 设计了一个跨平台的统一动作空间，使得代理能够以标准化方式与异构环境交互，简化了数据标注和策略学习。
*   **ScaleCUA基础模型家族：** 基于Qwen2.5-VL训练了一系列ScaleCUA基础代理模型，支持三种推理范式：定位模式（Grounding Mode）、直接动作模式（Direct Action Mode）和推理动作模式（Reasoned Action Mode），以实现感知、推理和动作的统一。

**3. 主要结果及其意义：**
*   **卓越的性能提升：** ScaleCUA在多个GUI基准测试中取得了显著的性能提升，例如在WebArena-Lite-v2上提升+26.6，在ScreenSpot-Pro上提升+10.7。
*   **新的SOTA结果：** 在MMBench-GUI L1-Hard上达到94.4%，在OSWorld-G上达到60.6%，在WebArena-Lite-v2上达到47.4%，均创下新的最先进（SOTA）记录。
*   **数据驱动扩展的有效性：** 实验结果强调了数据驱动扩展对于通用跨平台CUAs的强大作用，证明了多样化训练语料库能显著增强视觉理解能力。
*   **推理模式的优势：** 推理动作模式（RAM）在所有基准测试中均优于直接动作模式（DAM），尤其在复杂多步骤环境中表现突出，表明显式推理有助于维持任务连贯性并减少错误传播。

**4. 论文中提及的局限性：**
*   **代理收集数据质量：** 自动化代理收集的数据质量仍落后于人类专家标注的数据，规则驱动的探索可能产生语义较弱的轨迹。
*   **高级代理机制：** 当前工作尚未整合反射、基于记忆的决策或分层规划等高级代理机制。
*   **记忆机制的局限性：** 当前记忆机制较为初级，将过往操作视为扁平历史，这限制了长周期推理能力。
*   **跨应用泛化：** 在长周期推理和跨应用泛化方面仍存在局限性。

**5. 潜在的未来研究方向：**
*   **自改进学习循环：** 有效结合自动化数据收集与迭代模型优化，形成一个自改进学习循环。
*   **高级代理机制集成：** 探索并整合强化学习、策略奖励模型（PRMs）、反射、记忆和分层规划等高级代理机制。
*   **轻量级记忆系统：** 开发能够捕获时间依赖关系的轻量级且有效的记忆系统。
*   **数据混合策略：** 针对通用多模态数据与GUI特定数据之间的冲突优化信号，需要更精细的数据混合策略。
*   **原生桌面环境评估：** 优先在原生桌面环境中执行基于Web的任务，可能需要开发新的基准测试。

**Key Findings:**

- In this work, we introduce ScaleCUA, a step toward scaling open-source CUAs. It
offers a large-scale dataset spanning 6 operating systems and 3 task domains,
built via a closed-loop pipeline uniting automated agents with human experts.
- Specifically, it delivers strong gains over baselines (+26.6 on
WebArena-Lite-v2, +10.7 on ScreenSpot-Pro) and sets new state-of-the-art
results (94.4% on MMBench-GUI L1-Hard, 60.6% on OSWorld-G, 47.4% on
WebArena-Lite-v2).

**Links:**

- [PDF](https://arxiv.org/pdf/2509.15221v1)
- [arXiv](https://arxiv.org/abs/2509.15221v1)

---

<a id='2509.15220v1'></a>
## [Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model](https://arxiv.org/abs/2509.15220v1)

**Authors:** Fangjinhua Wang, Qingshan Xu, Yew-Soon Ong, Marc Pollefeys

**Published:** 2025-09-18

**Categories:** cs.CV

**Abstract:**

To reconstruct the 3D geometry from calibrated images, learning-based
multi-view stereo (MVS) methods typically perform multi-view depth estimation
and then fuse depth maps into a mesh or point cloud. To improve the
computational efficiency, many methods initialize a coarse depth map and then
gradually refine it in higher resolutions. Recently, diffusion models achieve
great success in generation tasks. Starting from a random noise, diffusion
models gradually recover the sample with an iterative denoising process. In
this paper, we propose a novel MVS framework, which introduces diffusion models
in MVS. Specifically, we formulate depth refinement as a conditional diffusion
process. Considering the discriminative characteristic of depth estimation, we
design a condition encoder to guide the diffusion process. To improve
efficiency, we propose a novel diffusion network combining lightweight 2D U-Net
and convolutional GRU. Moreover, we propose a novel confidence-based sampling
strategy to adaptively sample depth hypotheses based on the confidence
estimated by diffusion model. Based on our novel MVS framework, we propose two
novel MVS methods, DiffMVS and CasDiffMVS. DiffMVS achieves competitive
performance with state-of-the-art efficiency in run-time and GPU memory.
CasDiffMVS achieves state-of-the-art performance on DTU, Tanks & Temples and
ETH3D. Code is available at: https://github.com/cvg/diffmvs.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Fangjinhua Wang等人撰写的论文“Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model”的全面摘要。

---

### 论文摘要：基于置信度感知的扩散模型实现轻量级高精度多视角立体匹配

**1. 解决的主要问题或研究问题：**
该论文旨在解决多视角立体匹配（MVS）中3D几何重建的效率和精度问题。传统的MVS方法通常通过多视角深度估计和深度图融合来重建3D几何。为了提高计算效率，许多方法会先初始化一个粗略的深度图，然后逐步在高分辨率下进行细化。然而，现有的学习型MVS方法在处理光照变化、非朗伯表面和低纹理区域等挑战性场景时仍面临困难，并且在效率和精度之间存在权衡。特别是，将扩散模型引入MVS面临着如何有效利用条件信息、如何进行高效采样以及如何保持计算效率等挑战。

**2. 关键创新或方法论贡献：**
该论文提出了一个新颖的MVS框架，将扩散模型引入MVS，并将其应用于深度细化任务。主要创新包括：

*   **条件扩散过程的深度细化：** 将深度细化任务重新定义为条件扩散过程，从随机噪声开始，通过迭代去噪逐步恢复深度图。
*   **条件编码器（Condition Encoder）：** 设计了一个条件编码器，用于融合匹配信息、图像上下文和深度上下文特征，以指导扩散过程。这使得模型能够感知局部相似性和长距离上下文信息，从而生成准确的深度预测。
*   **基于置信度的采样策略（Confidence-based Sampling Strategy）：** 引入了一种新颖的采样策略，根据扩散模型估计的置信度自适应地生成像素级的多个深度假设。这克服了传统方法固定采样范围的局限性，通过调整采样范围来捕获非局部一阶优化信息，从而提高去噪过程的效率和准确性。
*   **轻量级扩散网络（Lightweight Diffusion Network）：** 提出了一种结合轻量级2D U-Net和卷积GRU的新型扩散网络。通过在单个扩散时间步内进行多迭代细化，并利用GRU捕获历史信息，显著提高了计算效率，避免了使用大型或堆叠U-Net。
*   **两种新型MVS方法：** 基于该框架，提出了DiffMVS和CasDiffMVS。DiffMVS专注于实时应用，通过单阶段扩散模型进行深度细化；CasDiffMVS则通过两阶段级联扩散细化，旨在实现高精度重建。

**3. 主要结果及其意义：**
实验结果表明，该方法在多个基准测试上取得了显著的性能：

*   **DiffMVS：** 在运行时和GPU内存效率方面，DiffMVS达到了与现有技术相当的性能，同时在Tanks & Temples和ETH3D数据集上取得了有竞争力的重建性能。
*   **CasDiffMVS：** 在DTU、Tanks & Temples和ETH3D数据集上，CasDiffMVS实现了最先进的重建性能，同时保持了高效率。
*   **效率优势：** 相比于最先进的IterMVS方法，DiffMVS在GPU内存消耗上减少了9.13%，速度提升了69.49%。CasDiffMVS的效率与PatchmatchNet相当，但性能优于其他顶尖方法。
*   **泛化能力：** 该方法在Tanks & Temples和ETH3D等挑战性场景中表现出强大的零样本泛化能力，能够生成更完整、更准确的表面。
*   **消融研究：** 验证了扩散机制、条件编码器（包括成本体、深度上下文和图像上下文）以及基于置信度的采样策略的有效性。结果表明，这些组件对于提高重建精度和泛化能力至关重要。

这些结果的意义在于，该论文成功地将扩散模型引入MVS，并在效率和精度之间取得了新的平衡，为MVS领域提供了一个强大且轻量级的新基线。

**4. 论文中提及的局限性：**
论文中没有明确提及具体的局限性，但从方法设计和实验设置中可以推断出一些潜在的方面：

*   **超参数调优：** 扩散模型的噪声尺度、采样步数等超参数需要仔细调优，以平衡性能和效率。虽然论文提供了默认设置，但在不同场景下可能需要进一步优化。
*   **训练数据依赖：** 尽管在BlendedMVS上进行了微调以提高泛化能力，但作为学习型方法，其性能仍可能受到训练数据分布的影响。
*   **复杂场景的鲁棒性：** 尽管在挑战性场景中表现良好，但极端情况（如极度低纹理、强反光等）下的鲁棒性仍有待进一步探索。

**5. 潜在的未来研究方向：**
该论文为MVS领域的未来研究开辟了多个方向：

*   **更高效的扩散模型：** 进一步探索更轻量级、更快的扩散模型架构，以适应更多资源受限的设备。
*   **自适应噪声调度：** 研究更智能的噪声调度策略，使其能够根据场景特性或估计的深度不确定性自适应调整，进一步提高精度和鲁棒性。
*   **多模态融合：** 探索将扩散模型与其他传感器数据（如LiDAR、IMU等）融合，以进一步提升MVS在复杂环境下的性能。
*   **实时MVS应用：** 进一步优化DiffMVS，使其在实时MVS应用中发挥更大潜力，例如机器人导航、自动驾驶等。
*   **无监督或半监督学习：** 探索在MVS中利用扩散模型进行无监督或半监督学习，以减少对大量标注数据的依赖。

---

这篇论文通过将扩散模型与MVS任务相结合，并引入一系列创新性的设计，为3D几何重建领域带来了新的突破，特别是在效率和精度方面取得了令人印象深刻的成果。

**Key Findings:**

- In
this paper, we propose a novel MVS framework, which introduces diffusion models
in MVS.
- To improve
efficiency, we propose a novel diffusion network combining lightweight 2D U-Net
and convolutional GRU.
- Moreover, we propose a novel confidence-based sampling
strategy to adaptively sample depth hypotheses based on the confidence
estimated by diffusion model.
- Based on our novel MVS framework, we propose two
novel MVS methods, DiffMVS and CasDiffMVS.
- DiffMVS achieves competitive
performance with state-of-the-art efficiency in run-time and GPU memory.
- CasDiffMVS achieves state-of-the-art performance on DTU, Tanks & Temples and
ETH3D.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.15220v1)
- [arXiv](https://arxiv.org/abs/2509.15220v1)

---

<a id='2509.15219v1'></a>
## [Out-of-Sight Trajectories: Tracking, Fusion, and Prediction](https://arxiv.org/abs/2509.15219v1)

**Authors:** Haichao Zhang, Yi Xu, Yun Fu

**Published:** 2025-09-18

**Categories:** cs.CV, cs.LG, cs.MA, cs.MM, cs.RO, 68T45, 68U10, 68T07, 68T40, 93C85, 93E11, 62M20, 62M10, 68U05, 94A12, F.2.2; I.2.9; I.2.10; I.4.1; I.4.8; I.4.9; I.5.4; I.3.7

**Abstract:**

Trajectory prediction is a critical task in computer vision and autonomous
systems, playing a key role in autonomous driving, robotics, surveillance, and
virtual reality. Existing methods often rely on complete and noise-free
observational data, overlooking the challenges associated with out-of-sight
objects and the inherent noise in sensor data caused by limited camera
coverage, obstructions, and the absence of ground truth for denoised
trajectories. These limitations pose safety risks and hinder reliable
prediction in real-world scenarios. In this extended work, we present
advancements in Out-of-Sight Trajectory (OST), a novel task that predicts the
noise-free visual trajectories of out-of-sight objects using noisy sensor data.
Building on our previous research, we broaden the scope of Out-of-Sight
Trajectory Prediction (OOSTraj) to include pedestrians and vehicles, extending
its applicability to autonomous driving, robotics, surveillance, and virtual
reality. Our enhanced Vision-Positioning Denoising Module leverages camera
calibration to establish a vision-positioning mapping, addressing the lack of
visual references, while effectively denoising noisy sensor data in an
unsupervised manner. Through extensive evaluations on the Vi-Fi and JRDB
datasets, our approach achieves state-of-the-art performance in both trajectory
denoising and prediction, significantly surpassing previous baselines.
Additionally, we introduce comparisons with traditional denoising methods, such
as Kalman filtering, and adapt recent trajectory prediction models to our task,
providing a comprehensive benchmark. This work represents the first initiative
to integrate vision-positioning projection for denoising noisy sensor
trajectories of out-of-sight agents, paving the way for future advances. The
code and preprocessed datasets are available at github.com/Hai-chao-Zhang/OST

**Analysis:**

好的，这是一篇关于Haichao Zhang, Yi Xu, Yun Fu撰写的论文“Out-of-Sight Trajectories: Tracking, Fusion, and Prediction”的全面摘要，涵盖了您要求的所有要点：

**论文摘要：Out-of-Sight Trajectories: Tracking, Fusion, and Prediction**

**1. 主要问题或研究问题：**
该论文旨在解决计算机视觉和自主系统中轨迹预测面临的关键挑战，特别是当目标物体“不可见”（out-of-sight）时。现有方法通常依赖于完整、无噪声的观测数据，但现实世界中，由于摄像头覆盖范围有限、障碍物和缺乏去噪轨迹的真值，传感器数据往往存在噪声，且物体可能完全超出视野。这导致了安全风险，并阻碍了在自动驾驶、机器人、监控和虚拟现实等应用中进行可靠的轨迹预测。核心问题是如何利用有噪声的传感器数据，预测不可见物体的无噪声视觉轨迹。

**2. 关键创新或方法贡献：**
该论文提出了“Out-of-Sight Trajectory (OST)”这一新颖任务，并引入了一个创新的“Vision-Positioning Denoising Module (VPD)”框架来解决它。主要贡献包括：
*   **任务扩展：** 将OST任务的范围从行人扩展到车辆，使其适用于更广泛的场景，如自动驾驶、机器人、监控和虚拟现实。
*   **无监督去噪：** 针对有噪声的传感器数据，提出了一种无监督去噪方法。由于缺乏去噪轨迹的真值，传统监督学习方法不可行。该方法通过利用视觉定位投影来构建有效的去噪监督。
*   **视觉定位投影：** 引入了“Vision-Positioning Projection Module (VPP)”和“Mapping Parameters Estimator (MPE)”。VPP负责将去噪后的传感器轨迹（3D世界坐标）映射到2D摄像头坐标。MPE通过分析可见物体的视觉和传感器轨迹之间的相关性，动态预测摄像头矩阵嵌入（包括内参和外参），从而解决了缺乏直接视觉参考的问题。
*   **Transformer架构：** 传感器去噪编码器（SDE）、映射参数估计器（MPE）和不可见物体预测解码器（OPD）均采用基于Transformer的架构，以有效捕捉轨迹数据中的时序和上下文依赖性。
*   **模块化设计：** 整个框架设计为模块化，包括SDE（去噪）、MPE（映射参数估计）、VPP（视觉投影）和OPD（预测），每个模块协同工作，确保对有噪声和不完整传感器数据的全面处理。

**3. 主要结果及其意义：**
*   **最先进的性能：** 在Vi-Fi和JRDB数据集上的广泛评估表明，该方法在轨迹去噪和预测方面均达到了最先进的性能，显著超越了现有基线。
*   **去噪的重要性：** 实验结果强调了鲁棒去噪在提高不可见物体轨迹预测准确性方面的关键作用，去噪性能的提升直接转化为预测准确性的提高。
*   **VPD模块的有效性：** 消融研究证实了VPD模块中每个组件（SDE、MPE、VPP、OPD）的必要性，它们共同促进了模型实现最优性能。
*   **泛化能力：** 该模型在处理不同传感器噪声场景和利用精确视觉监督方面表现出鲁棒性，在两个数据集上均表现出一致的性能。
*   **开创性工作：** 这是首次尝试整合视觉定位投影来去噪不可见物体的有噪声传感器轨迹，为该领域的未来发展铺平了道路。

**4. 论文中提到的局限性：**
*   **摄像头校准的隐式约束：** 论文指出，在许多实际场景中，数据集不提供内参矩阵，且实际摄像头操作（如变焦、自动对焦）可能改变这些参数。虽然模型能够估计统一的嵌入来处理这些复杂场景，但当内参矩阵可用时，直接整合它们仍需考虑。
*   **不可见物体的感知距离：** 模型在处理距离非常远的物体（例如几英里外）时，性能可能会下降。目前模型在数据集范围内的距离上表现良好，但对于更远距离的性能仍需评估。极端距离的物体可能导致分布外（out-of-distribution）问题，影响性能。

**5. 潜在的未来研究方向：**
*   **处理极端噪声：** 进一步研究如何处理极端噪声情况，以提高模型在更具挑战性环境中的鲁棒性。
*   **实时实现：** 探索如何优化模型以实现实时轨迹预测，这对于自动驾驶等对延迟敏感的应用至关重要。
*   **更远的感知距离：** 评估和改进模型在处理更远距离不可见物体时的性能，以解决分布外数据问题。
*   **更复杂的环境：** 进一步研究在视觉观测不确定或输入信号固有噪声的复杂现实世界场景中，如何提高轨迹预测的准确性和可靠性。
*   **多模态融合的扩展：** 探索将更多模态（如雷达、激光雷达等）整合到框架中，以进一步增强去噪和预测能力。

总而言之，这篇论文通过引入OST任务和创新的VPD框架，为解决不可见物体的轨迹预测问题提供了开创性的解决方案。它通过无监督去噪和视觉定位投影，有效地处理了有噪声的传感器数据，并在多个关键应用领域取得了显著进展。

**Key Findings:**

- In this extended work, we present
advancements in Out-of-Sight Trajectory (OST), a novel task that predicts the
noise-free visual trajectories of out-of-sight objects using noisy sensor data.
- Through extensive evaluations on the Vi-Fi and JRDB
datasets, our approach achieves state-of-the-art performance in both trajectory
denoising and prediction, significantly surpassing previous baselines.
- Additionally, we introduce comparisons with traditional denoising methods, such
as Kalman filtering, and adapt recent trajectory prediction models to our task,
providing a comprehensive benchmark.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.15219v1)
- [arXiv](https://arxiv.org/abs/2509.15219v1)

---

<a id='2509.15178v1'></a>
## [Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding](https://arxiv.org/abs/2509.15178v1)

**Authors:** Zaiquan Yang, Yuhao Liu, Gerhard Hancke, Rynson W. H. Lau

**Published:** 2025-09-18

**Categories:** cs.CV

**Abstract:**

Spatio-temporal video grounding (STVG) aims at localizing the spatio-temporal
tube of a video, as specified by the input text query. In this paper, we
utilize multimodal large language models (MLLMs) to explore a zero-shot
solution in STVG. We reveal two key insights about MLLMs: (1) MLLMs tend to
dynamically assign special tokens, referred to as \textit{grounding tokens},
for grounding the text query; and (2) MLLMs often suffer from suboptimal
grounding due to the inability to fully integrate the cues in the text query
(\textit{e.g.}, attributes, actions) for inference. Based on these insights, we
propose a MLLM-based zero-shot framework for STVG, which includes novel
decomposed spatio-temporal highlighting (DSTH) and temporal-augmented
assembling (TAS) strategies to unleash the reasoning ability of MLLMs. The DSTH
strategy first decouples the original query into attribute and action
sub-queries for inquiring the existence of the target both spatially and
temporally. It then uses a novel logit-guided re-attention (LRA) module to
learn latent variables as spatial and temporal prompts, by regularizing token
predictions for each sub-query. These prompts highlight attribute and action
cues, respectively, directing the model's attention to reliable spatial and
temporal related visual regions. In addition, as the spatial grounding by the
attribute sub-query should be temporally consistent, we introduce the TAS
strategy to assemble the predictions using the original video frames and the
temporal-augmented frames as inputs to help improve temporal consistency. We
evaluate our method on various MLLMs, and show that it outperforms SOTA methods
on three common STVG benchmarks.
  The code will be available at https://github.com/zaiquanyang/LLaVA_Next_STVG.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zaiquan Yang, Yuhao Liu, Gerhard Hancke, Rynson W. H. Lau撰写的论文“Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding”的全面摘要。

---

### 论文摘要：Unleashing the Potential of Multimodal LLMs for Zero-Shot Spatio-Temporal Video Grounding

**1. 解决的主要问题或研究问题：**
该论文旨在解决**时空视频定位（Spatio-Temporal Video Grounding, STVG）**任务，即根据给定的文本查询，在视频中定位目标对象的时空管（spatio-temporal tube）。传统的STVG方法通常依赖于昂贵的帧级标注进行全监督训练，而本文则探索利用多模态大语言模型（MLLMs）实现**零样本（zero-shot）**STVG解决方案，以减轻标注负担并提高泛化能力。论文特别关注MLLMs在处理复杂视频查询时，由于未能充分整合文本查询中的属性和动作线索而导致的次优定位问题。

**2. 关键创新或方法贡献：**
作者基于对MLLMs的两个关键洞察（即MLLMs会动态分配“定位令牌”进行文本查询定位，但常因未能充分利用文本线索而导致定位次优），提出了一个新颖的零样本STVG框架，包含以下核心策略：

*   **分解时空高亮（Decomposed Spatio-Temporal Highlighting, DSTH）策略：**
    *   将原始文本查询分解为**属性子查询**和**动作子查询**，分别用于在空间和时间上查询目标的存在。
    *   引入**对数引导重注意力（Logit-guided Re-attention, LRA）模块**，通过正则化每个子查询的令牌预测，学习潜在变量作为空间和时间提示。这些提示能够分别高亮属性和动作线索，引导模型关注可靠的、与时空相关的视觉区域。
*   **时序增强组装（Temporal-Augmented Assembling, TAS）策略：**
    *   为了提高空间定位的时序一致性（特别是属性子查询的定位），TAS策略利用原始视频帧和时序增强帧（例如，反转帧顺序）作为输入，组装不同预测以改善时序一致性。
*   **定位令牌识别（Grounding Token Identification）：** 论文发现MLLMs会动态分配具有高视觉激活度的特殊令牌（称为“定位令牌”），这些令牌在定位文本相关区域方面表现出色，并利用这一发现构建了零样本STVG框架。

**3. 主要结果及其意义：**
该方法在多种MLLMs上进行了评估，并在三个常见的STVG基准测试（HCSTVG-v1、HCSTVG-v2和VidSTG）上取得了显著优于现有SOTA方法的性能。
*   例如，基于LLaVA-Next-Video-7B模型，该方法在vIoU@0.3和vIoU@0.5指标上分别比E3M高出4.2%和1.8%。
*   与更强大的LLaVA-OneVision-7B模型结合时，性能提升更为显著，分别达到12.1%和5.7%。
*   即使在动作线索较少的VidSTG数据集上，该框架也超越了之前的SOTA方法，展现了强大的泛化能力。
*   消融实验证明了DSTH策略（包括子查询分解和LRA模块）以及TAS策略的有效性，它们能够引导MLLMs更好地关注时空相关区域，并提高空间定位的鲁棒性。

这些结果表明，通过有效利用MLLMs的内在能力并引入创新的高亮和组装策略，可以显著提升零样本STVG的性能，甚至超越一些弱监督方法，并接近全监督方法的水平。

**4. 论文中提及的局限性：**
论文指出，该方法存在局限性。由于MLLMs的高计算消耗，它可能难以很好地处理长视频。

**5. 潜在的未来研究方向：**
未来的工作可以考虑在模型设计中整合**令牌剪枝（token pruning）**和**关键帧选择（key frame selection）**技术，以解决处理长视频时的计算效率问题。

---

**Key Findings:**

- Based on these insights, we
propose a MLLM-based zero-shot framework for STVG, which includes novel
decomposed spatio-temporal highlighting (DSTH) and temporal-augmented
assembling (TAS) strategies to unleash the reasoning ability of MLLMs. The DSTH
strategy first decouples the original query into attribute and action
sub-queries for inquiring the existence of the target both spatially and
temporally.
- It then uses a novel logit-guided re-attention (LRA) module to
learn latent variables as spatial and temporal prompts, by regularizing token
predictions for each sub-query.
- In addition, as the spatial grounding by the
attribute sub-query should be temporally consistent, we introduce the TAS
strategy to assemble the predictions using the original video frames and the
temporal-augmented frames as inputs to help improve temporal consistency.
- We
evaluate our method on various MLLMs, and show that it outperforms SOTA methods
on three common STVG benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.15178v1)
- [arXiv](https://arxiv.org/abs/2509.15178v1)

---

<a id='2509.15177v1'></a>
## [A Race Bias Free Face Aging Model for Reliable Kinship Verification](https://arxiv.org/abs/2509.15177v1)

**Authors:** Ali Nazari, Bardiya Kariminia, Mohsen Ebrahimi Moghaddam

**Published:** 2025-09-18

**Categories:** cs.CV

**Abstract:**

The age gap in kinship verification addresses the time difference between the
photos of the parent and the child. Moreover, their same-age photos are often
unavailable, and face aging models are racially biased, which impacts the
likeness of photos. Therefore, we propose a face aging GAN model, RA-GAN,
consisting of two new modules, RACEpSp and a feature mixer, to produce racially
unbiased images. The unbiased synthesized photos are used in kinship
verification to investigate the results of verifying same-age parent-child
images. The experiments demonstrate that our RA-GAN outperforms SAM-GAN on an
average of 13.14\% across all age groups, and CUSP-GAN in the 60+ age group by
9.1\% in terms of racial accuracy. Moreover, RA-GAN can preserve subjects'
identities better than SAM-GAN and CUSP-GAN across all age groups.
Additionally, we demonstrate that transforming parent and child images from the
KinFaceW-I and KinFaceW-II datasets to the same age can enhance the
verification accuracy across all age groups. The accuracy increases with our
RA-GAN for the kinship relationships of father-son and father-daughter,
mother-son, and mother-daughter, which are 5.22, 5.12, 1.63, and 0.41,
respectively, on KinFaceW-I. Additionally, the accuracy for the relationships
of father-daughter, father-son, and mother-son is 2.9, 0.39, and 1.6 on
KinFaceW-II, respectively. The code is available
at~\href{https://github.com/bardiya2254kariminia/An-Age-Transformation-whitout-racial-bias-for-Kinship-verification}{Github}

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Ali Nazari, Bardiya Kariminia, Mohsen Ebrahimi Moghaddam撰写的论文“A Race Bias Free Face Aging Model for Reliable Kinship Verification”的全面摘要。

---

**论文摘要：A Race Bias Free Face Aging Model for Reliable Kinship Verification**

**1. 主要问题或研究问题：**
该研究主要解决亲属关系验证（Kinship Verification, KV）中的两大挑战：
*   **年龄差距问题：** 父母和子女的照片通常存在显著的年龄差异，且难以获取同龄照片，这严重影响了亲属关系验证的准确性。
*   **现有面部老化模型的种族偏见：** 当前的面部老化模型在生成图像时存在种族偏见，这导致合成图像与原始图像的相似度降低，进而影响亲属关系验证的可靠性。

**2. 关键创新或方法论贡献：**
为了解决上述问题，作者提出了一种名为 **RA-GAN (RaceAgingGAN)** 的面部老化生成对抗网络模型，其核心创新包括：
*   **RACEpSp 模块：** 这是一个新颖的模块，旨在消除面部老化过程中产生的种族偏见，确保合成图像在种族上是无偏的。它利用预训练的ResNet34模型在Fairface数据集上进行训练，以更好地处理多样化的面部姿态和种族信息。
*   **特征混合器（Feature Mixer）：** 该模块用于融合年龄特征和种族特定面部特征，以找到最佳组合，从而生成在种族上无偏且身份保留的图像。
*   **新数据集的构建：** 作者从UTKFace数据集中收集并构建了一个新的、种族平衡的数据集，以解决现有数据集在种族属性上的不平衡问题，从而训练出更具泛化能力的模型。
*   **同龄图像的亲属关系验证：** 提出将父母和子女的图像转换为相同年龄，以提高亲属关系验证的准确性。

**3. 主要结果及其意义：**
实验结果显著地证明了RA-GAN的有效性：
*   **种族准确性提升：** RA-GAN在所有年龄组的种族准确性方面平均优于SAM-GAN 13.14%，在60岁以上年龄组中优于CUSP-GAN 9.1%。这表明RA-GAN能够生成种族无偏的合成图像。
*   **身份保留能力：** RA-GAN在所有年龄组中比SAM-GAN和CUSP-GAN更好地保留了受试者的身份，这对于亲属关系验证至关重要。
*   **亲属关系验证准确性提高：** 将KinFaceW-I和KinFaceW-II数据集中的父母和子女图像转换为相同年龄后，亲属关系验证的准确性显著提高。具体而言，在KinFaceW-I数据集上，父子、父女、母子和母女关系的准确性分别提高了5.22%、5.12%、1.63%和0.41%。在KinFaceW-II数据集上，父女、父子和母子关系的准确性分别提高了2.9%、0.39%和1.6%。
*   **年龄转换误差（MAE）降低：** RA-GAN在年龄转换方面表现优异，除了20岁年龄组外，在所有其他年龄组中，其目标年龄转换误差均低于CUSP模型。

这些结果的意义在于，RA-GAN不仅解决了面部老化模型中的种族偏见问题，还通过生成高质量、身份保留且种族无偏的同龄面部图像，显著提升了亲属关系验证的准确性和可靠性，尤其是在处理真实世界中不同来源和拍摄时间照片的挑战时。

**4. 论文中提及的局限性：**
论文中并未明确指出当前研究的局限性，但从其讨论和未来工作方向可以推断出一些隐含的方面：
*   **数据集的年龄分布不均：** 尽管作者构建了种族平衡的数据集，但仍指出如果要求数据集在种族和年龄上都均匀，图像数量会减少，这可能暗示在某些年龄段的数据量仍有待提高。
*   **模型泛化能力：** 尽管RA-GAN在种族和身份保留方面表现出色，但面部老化模型通常难以泛化到未充分代表的年龄组、种族或面部结构，这可能是一个持续的挑战。
*   **对全脸图像的依赖：** GAN模型通常需要全脸图像进行训练，而KinFaceW-I等数据集中的图像是裁剪过的，需要额外的镜像增强和自编码器转换步骤来生成全脸图像，这增加了处理的复杂性。

**5. 潜在的未来研究方向：**
论文的结论和未来工作部分指出了以下潜在研究方向：
*   **进一步消除种族偏见：** 作者强调其方法在融合种族特征以消除种族偏见方面具有高潜力，这暗示未来可以继续探索更先进的技术来彻底根除种族偏见，并生成与原始图像种族类别高度相似的图像。
*   **更广泛的年龄范围和多样性：** 尽管RA-GAN覆盖了20-80岁的年龄范围，但未来可以探索更广泛的年龄范围，并进一步提高模型在极端年龄（如儿童或老年）的性能。
*   **更复杂的亲属关系验证场景：** 论文主要关注父母-子女关系，未来可以扩展到其他更复杂的亲属关系（如祖父母-孙子女、兄弟姐妹等）。
*   **实时应用和效率：** 尽管深度学习模型在准确性上表现出色，但其计算资源需求和推理速度可能成为实时应用的瓶颈，未来可以研究更高效的模型架构和优化方法。

---

**Key Findings:**

- Therefore, we propose a face aging GAN model, RA-GAN,
consisting of two new modules, RACEpSp and a feature mixer, to produce racially
unbiased images.
- The experiments demonstrate that our RA-GAN outperforms SAM-GAN on an
average of 13.14\% across all age groups, and CUSP-GAN in the 60+ age group by
9.1\% in terms of racial accuracy.
- Additionally, we demonstrate that transforming parent and child images from the
KinFaceW-I and KinFaceW-II datasets to the same age can enhance the
verification accuracy across all age groups.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.15177v1)
- [arXiv](https://arxiv.org/abs/2509.15177v1)

---

<a id='2509.14980v1'></a>
## [M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation](https://arxiv.org/abs/2509.14980v1)

**Authors:** Ju Dong, Lei Zhang, Liding Zhang, Yao Ling, Yu Fu, Kaixin Bai, Zoltán-Csaba Márton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang

**Published:** 2025-09-18

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Mobile manipulation requires the coordinated control of a mobile base and a
robotic arm while simultaneously perceiving both global scene context and
fine-grained object details. Existing single-view approaches often fail in
unstructured environments due to limited fields of view, exploration, and
generalization abilities. Moreover, classical controllers, although stable,
struggle with efficiency and manipulability near singularities. To address
these challenges, we propose M4Diffuser, a hybrid framework that integrates a
Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP
(ReM-QP) controller for mobile manipulation. The diffusion policy leverages
proprioceptive states and complementary camera perspectives with both
close-range object details and global scene context to generate task-relevant
end-effector goals in the world frame. These high-level goals are then executed
by the ReM-QP controller, which eliminates slack variables for computational
efficiency and incorporates manipulability-aware preferences for robustness
near singularities. Comprehensive experiments in simulation and real-world
environments show that M4Diffuser achieves 7 to 56 percent higher success rates
and reduces collisions by 3 to 31 percent over baselines. Our approach
demonstrates robust performance for smooth whole-body coordination, and strong
generalization to unseen tasks, paving the way for reliable mobile manipulation
in unstructured environments. Details of the demo and supplemental material are
available on our project website https://sites.google.com/view/m4diffuser.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供对Ju Dong等人撰写的论文“M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation”的全面摘要。

---

### 论文摘要：M4Diffuser: 具有可操作性感知控制的多视角扩散策略，实现鲁棒的移动操作

**1. 解决的主要问题或研究问题：**
该论文旨在解决移动操作中的核心挑战，即如何在非结构化环境中实现移动基座和机械臂的协调控制，同时感知全局场景上下文和精细物体细节。现有方法存在以下局限性：
*   **单视角方法**：由于视野有限、探索能力和泛化能力不足，在非结构化环境中表现不佳。
*   **经典控制器**：虽然稳定，但在效率和接近奇异点时的可操作性方面存在困难，通常依赖于松弛变量，增加了计算开销并降低了轨迹平滑度。
*   **学习驱动方法**：虽然适应性和泛化能力强，但实际部署时稳定性不足，容易因视觉输入遮挡或超出视野而失败。

**2. 关键创新或方法论贡献：**
M4Diffuser 提出了一种混合框架，结合了多视角扩散策略和新型的“Reduced and Manipulability-aware QP (ReM-QP)”控制器，其主要创新点包括：

*   **多视角扩散Transformer策略 (Multi-View Diffusion Transformer Policy)**：
    *   该策略通过结合互补的视角和本体状态，同时捕捉局部物体细节和全局场景上下文。
    *   利用Transformer编码器和条件去噪扩散过程，生成世界坐标系中任务相关的末端执行器目标。
    *   多视角输入显著提高了鲁棒性和泛化能力，解决了单视角感知不足的问题。

*   **Reduced and Manipulability-aware QP (ReM-QP) 控制器**：
    *   在低层控制层面，ReM-QP 消除了传统QP公式中的松弛变量，从而提高了计算效率。
    *   引入了基于逆条件数（ICN）的可操作性偏好，以确保在接近奇异点时的稳定性和平滑性，提高了鲁棒性。
    *   通过标准不等式约束确保了安全性。

**3. 主要结果及其重要性：**
M4Diffuser 在仿真和真实世界环境中进行了广泛实验，结果表明：

*   **性能提升**：与基线方法相比，M4Diffuser 的成功率提高了7%到56%，碰撞率降低了3%到31%。
*   **鲁棒性与泛化能力**：该方法在非结构化环境中实现了平滑全身协调的鲁棒性能，并对未见过的任务表现出强大的泛化能力。
*   **与SOTA方法的比较**：M4Diffuser 显著优于传统规划方法和纯学习方法，平均成功率提高了28.4%，碰撞率降低了69%。与最先进的方法（如HoMeR）相比，成功率提高了10.0%，碰撞率减少了5.2%。
*   **ReM-QP的效率与平滑度**：ReM-QP 将任务执行时间缩短了28%，末端执行器加加速度（jerk）降低了35%，在效率和轨迹平滑度之间取得了良好平衡。

这些结果的重要性在于，M4Diffuser 为在非结构化环境中实现可靠的移动操作铺平了道路，解决了现有方法在鲁棒性、效率和泛化能力方面的关键限制。

**4. 论文中提到的局限性：**
论文中并未明确列出M4Diffuser自身的局限性，但通过对现有方法的讨论，可以推断出M4Diffuser旨在克服的挑战，这些挑战在一定程度上也可能构成其未来改进的方向：

*   **数据依赖性**：尽管模仿学习数据效率高，但训练仍然需要专家演示数据。
*   **复杂环境中的探索能力**：虽然多视角策略增强了感知，但机器人应对高度复杂、完全未知的环境时的探索能力仍有提升空间。
*   **计算资源**：虽然ReM-QP提高了计算效率，但整个混合框架在实时部署时仍可能面临计算资源的需求。

**5. 潜在的未来研究方向：**
论文指出，未来的工作将集中于：

*   **语言和多模态引导的移动操作 (Language- and Multimodal-guided Mobile Manipulation)**：将语言指令和更多模态信息整合到移动操作中，以实现更高级别的任务理解和执行。

---

这份摘要清晰地概述了论文的核心内容，突出了其在移动操作领域的技术贡献和潜在影响。

**Key Findings:**

- To address
these challenges, we propose M4Diffuser, a hybrid framework that integrates a
Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP
(ReM-QP) controller for mobile manipulation.
- Our approach
demonstrates robust performance for smooth whole-body coordination, and strong
generalization to unseen tasks, paving the way for reliable mobile manipulation
in unstructured environments.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.14980v1)
- [arXiv](https://arxiv.org/abs/2509.14980v1)

---

<a id='2509.14966v1'></a>
## [RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching](https://arxiv.org/abs/2509.14966v1)

**Authors:** Xingwu Zhang, Guanxuan Li, Zhuocheng Zhang, Zijun Long

**Published:** 2025-09-18

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

The rapidly growing number of product categories in large-scale e-commerce
makes accurate object identification for automated packing in warehouses
substantially more difficult. As the catalog grows, intra-class variability and
a long tail of rare or visually similar items increase, and when combined with
diverse packaging, cluttered containers, frequent occlusion, and large
viewpoint changes-these factors amplify discrepancies between query and
reference images, causing sharp performance drops for methods that rely solely
on 2D appearance features. Thus, we propose RoboEye, a two-stage identification
framework that dynamically augments 2D semantic features with domain-adapted 3D
reasoning and lightweight adapters to bridge training deployment gaps. In the
first stage, we train a large vision model to extract 2D features for
generating candidate rankings. A lightweight 3D-feature-awareness module then
estimates 3D feature quality and predicts whether 3D re-ranking is necessary,
preventing performance degradation and avoiding unnecessary computation. When
invoked, the second stage uses our robot 3D retrieval transformer, comprising a
3D feature extractor that produces geometry-aware dense features and a
keypoint-based matcher that computes keypoint-correspondence confidences
between query and reference images instead of conventional cosine-similarity
scoring. Experiments show that RoboEye improves Recall@1 by 7.1% over the prior
state of the art (RoboLLM). Moreover, RoboEye operates using only RGB images,
avoiding reliance on explicit 3D inputs and reducing deployment costs. The code
used in this paper is publicly available at:
https://github.com/longkukuhi/RoboEye.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Xingwu Zhang等人撰写的论文“RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching”的全面摘要。

---

### RoboEye: 通过选择性3D几何关键点匹配增强2D机器人物体识别的摘要

**1. 主要问题或研究问题：**
随着大规模电商产品目录的快速增长，仓库自动化包装中的物体识别变得越来越困难。现有方法主要依赖2D外观特征，但在面对类内变异性、长尾分布、多样化包装、杂乱容器、频繁遮挡和大幅度视角变化等挑战性场景时，其性能会急剧下降。这些因素放大了查询图像和参考图像之间的差异，使得仅基于2D特征的方法难以泛化。因此，核心研究问题是如何在不依赖显式3D输入（如点云或深度图）的情况下，利用3D几何线索来提高在复杂仓库条件下的物体识别鲁棒性，同时降低部署成本。

**2. 关键创新或方法论贡献：**
RoboEye提出了一个两阶段识别框架，动态地将2D语义特征与领域适应的3D推理和轻量级适配器相结合，以弥合训练与部署之间的差距。
*   **两阶段识别框架：**
    *   **第一阶段（2D检索）：** 使用大型视觉模型提取2D特征，生成初步的候选排名。
    *   **3D特征感知模块（3D-FAM）：** 一个轻量级的模块，用于评估3D特征的质量，并预测是否需要进行3D重排序。这避免了不必要的计算，并在3D线索嘈杂时防止性能下降。该模块通过MRR驱动的3D感知训练（M3AT）方案进行训练，该方案识别何时3D重排序能带来实际收益。
    *   **第二阶段（3D重排序）：** 当3D-FAM模块被激活时，使用机器人3D检索Transformer。
*   **机器人3D检索Transformer：**
    *   **3D特征提取器：** 采用VGGT（Visual Geometry Grounded Transformer）的聚合器组件，从多视图2D图像中推断3D几何信息，生成几何感知的密集特征。
    *   **关键点匹配器：** 替换了传统的余弦相似度评分，通过计算查询图像和参考图像之间的关键点对应置信度来提供更鲁棒的相似性度量。该匹配器基于稀疏关键点匹配，并经过重新设计以生成置信度分数作为相似性估计。
*   **基于适配器的领域适应策略：** 为了弥合VGGT预训练数据与仓库特定数据集之间的领域差距，RoboEye采用了一种基于适配器的训练策略，冻结3D特征提取器，仅对匹配器进行训练，并使用轻量级知识适配器进行增强，以实现高效的领域适应。

**3. 主要结果及其意义：**
*   **性能提升：** RoboEye在Amazon ARMBench数据集上，Recall@1指标比现有最先进方法（RoboLLM）提高了7.1%。在多视图设置下，其性能提升更为显著，例如在容器图库检索中，Recall@1从98.0%提高到99.4%（+1.4%），在全局图库检索中，Recall@1提高了7.1%。
*   **鲁棒性：** RoboEye在面对大规模目录、视角和姿态变化、遮挡和包装变化等挑战时，表现出卓越的鲁棒性。
*   **效率：** 3D特征感知模块在平衡效率和鲁棒性方面发挥了关键作用。它仅在必要时激活3D推理，保持了接近2D运行时的速度，同时保留了几何验证的优势。
*   **无需显式3D输入：** RoboEye仅使用RGB图像进行操作，避免了对显式3D输入（如点云或深度图）的依赖，从而降低了部署成本。
*   **模型规模与性能：** 实验表明，简单地增加2D特征提取器的模型规模不足以解决仓库环境中的挑战。RoboEye通过结合3D感知组件，以可比的参数规模实现了显著的性能提升。

**4. 论文中提到的局限性：**
*   **计算开销：** 尽管RoboEye通过3D-FAM模块减少了不必要的3D计算，但3D重排序本身仍然会带来一定的计算开销。在没有感知模块的情况下，无条件3D重排序会显著增加延迟。
*   **噪声敏感性：** 论文指出，在2D重排序阶段，如果包含过多低质量的候选对象，可能会引入噪声，稀释判别信号，并可能导致性能略微下降。
*   **硬件限制：** 论文提到，在与RoboLLM的对比中，由于硬件限制（较小的批次大小），其2D模型初始性能略低于RoboLLM，这表明对比学习通常受益于更大的批次。

**5. 潜在的未来研究方向：**
论文中没有明确提出未来的研究方向，但从其贡献和局限性来看，可以推断出以下潜在方向：
*   **更轻量级的3D推理：** 进一步优化3D特征提取和匹配机制，以在保持性能的同时进一步降低计算成本和推理延迟，使其更适用于资源受限的边缘设备。
*   **自适应关键点选择：** 探索更智能的关键点选择策略，以避免引入低质量的关键点，从而提高几何验证的准确性和稳定性。
*   **多模态融合的泛化：** 虽然RoboEye避免了显式3D输入，但可以研究如何在不增加部署复杂性的前提下，更有效地融合其他隐式模态信息（例如，通过物理模拟或语义知识），以进一步增强识别鲁棒性。
*   **零样本/少样本学习：** 鉴于电商产品目录的快速增长和长尾分布，探索如何将RoboEye框架扩展到零样本或少样本识别场景，以处理新产品或稀有产品。

---

这篇论文通过其创新的两阶段框架和选择性3D几何关键点匹配方法，为机器人物体识别领域带来了显著的进步，特别是在具有挑战性的仓库环境中。它有效地解决了传统2D方法在复杂场景下的局限性，同时避免了对昂贵3D传感器的依赖，为实际部署提供了高效且鲁棒的解决方案。

**Key Findings:**

- Thus, we propose RoboEye, a two-stage identification
framework that dynamically augments 2D semantic features with domain-adapted 3D
reasoning and lightweight adapters to bridge training deployment gaps.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.14966v1)
- [arXiv](https://arxiv.org/abs/2509.14966v1)

---

<a id='2509.14957v1'></a>
## [DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection](https://arxiv.org/abs/2509.14957v1)

**Authors:** Zhuokang Shen, Kaisen Zhang, Bohan Jia, Yuan Fang, Zhou Yu, Shaohui Lin

**Published:** 2025-09-18

**Categories:** cs.CV

**Abstract:**

With the increasing prevalence of synthetic images, evaluating image
authenticity and locating forgeries accurately while maintaining human
interpretability remains a challenging task. Existing detection models
primarily focus on simple authenticity classification, ultimately providing
only a forgery probability or binary judgment, which offers limited explanatory
insights into image authenticity. Moreover, while MLLM-based detection methods
can provide more interpretable results, they still lag behind expert models in
terms of pure authenticity classification accuracy. To address this, we propose
DF-LLaVA, a simple yet effective framework that unlocks the intrinsic
discrimination potential of MLLMs. Our approach first extracts latent knowledge
from MLLMs and then injects it into training via prompts. This framework allows
LLaVA to achieve outstanding detection accuracy exceeding expert models while
still maintaining the interpretability offered by MLLMs. Extensive experiments
confirm the superiority of our DF-LLaVA, achieving both high accuracy and
explainability in synthetic image detection. Code is available online at:
https://github.com/Eliot-Shen/DF-LLaVA.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zhuokang Shen等人撰写的论文“DF-LLaVA: Unlocking MLLM's potential for Synthetic Image Detection via Prompt-Guided Knowledge Injection”的全面摘要。

---

**论文摘要：DF-LLaVA: 通过提示引导的知识注入解锁MLLM在合成图像检测中的潜力**

**1. 主要问题或研究问题：**
随着生成模型（如扩散模型）的快速发展，合成图像日益普及，对图像真实性评估和伪造定位提出了严峻挑战。现有检测模型主要关注简单的真实性二分类，提供有限的解释性洞察。虽然基于多模态大语言模型（MLLM）的检测方法能提供更具解释性的结果，但在纯粹的真实性分类准确性方面仍落后于专家模型。因此，核心研究问题是如何开发一种既能实现高检测准确性，又能提供人类可解释的伪造证据的专家模型，以应对合成图像检测的挑战。

**2. 关键创新或方法论贡献：**
本文提出了DF-LLaVA，一个简单而有效的框架，旨在解锁MLLM在合成图像检测中的内在判别潜力。其关键创新和方法论贡献包括：
*   **知识提取与注入框架：** DF-LLaVA首先从MLLM的视觉编码器中提取潜在的判别知识（通过训练一个二分类器），然后通过提示（prompts）将其注入到训练过程中。这种方法允许LLaVA在保持MLLM解释性的同时，显著提高检测准确性。
*   **视觉编码器判别潜力揭示：** 论文揭示了MLLM的判别潜力主要存在于其视觉编码器中。通过在CLIP-ViT的[CLS] token上训练二分类器，并将其概率输出作为嵌入知识注入提示，DF-LLaVA能够有效利用这一潜力。
*   **多视角伪影解释：** DF-LLaVA能够从多个视角（如结构、失真和物理特征）识别合成图像模型产生的伪影，从而增强了对人类的解释性。
*   **基于LLaVA-v1.5的架构：** 框架构建在LLaVA-v1.5架构之上，包含视觉编码器（CLIP-ViT(L-14)）、视觉/语言投影器、线性头部和大型语言模型（Vicuna-v1.5-7B）。

**3. 主要结果及其意义：**
*   **卓越的检测准确性：** DF-LLaVA在FakeClue和LOKI数据集上实现了超越现有专家模型的检测准确性。与强大的开源模型Qwen2-VL-72B相比，DF-LLaVA在Acc和F1上平均提升了29.5%和40.1%。相对于先前的MLLM方法FakeVLM，DF-LLaVA在Acc和F1上平均提升了4.2%和3.9%。
*   **强大的解释性：** DF-LLaVA在CSS和ROUGE_L指标上表现出色，提供了比FakeVLM和通用MLLM更准确、可靠的伪影解释。
*   **泛化能力：** 在DMimage数据集上的实验结果表明，DF-LLaVA的性能与专家模型相当甚至超越，尤其在伪造图像检测方面表现最佳。
*   **消融实验验证：** 消融研究证实了提示引导知识注入（PGKI）框架的有效性，即使在LLaVA-FullFT设置下，PGKI也能带来进一步的性能提升，尤其在有限训练资源下，对提升LLaVA的判别能力至关重要。

**4. 论文中提及的局限性：**
*   **训练数据类别不平衡：** 论文提到，DF-LLaVA在真实类别上的表现可能略弱于专家模型，这可能是由于训练数据中类别不平衡造成的。
*   **低秩适配器（LoRA）的局限性：** 实验结果表明，LLaVA-LoRA在判别任务上的性能显著低于LLaVA-FullFT，这暗示低秩适配器可能不适合此类任务。
*   **FakeVLM的视觉编码器微调：** 论文指出，FakeVLM通过微调视觉编码器可能扰乱了其视觉表示，导致其性能略低于LLaVA-FullFT。

**5. 潜在的未来研究方向：**
*   **探索其他MLLM架构的判别潜力：** 未来研究可以进一步探索其他MLLM架构在合成图像检测中的判别潜力。
*   **增强判别能力的策略：** 进一步研究增强MLLM判别能力的策略，以应对不断演变的合成图像生成技术。

---

总而言之，DF-LLaVA通过创新的提示引导知识注入框架，成功地将MLLM的内在判别能力转化为卓越的合成图像检测性能，同时保持了高度的人类可解释性。这为计算机视觉领域在应对日益增长的合成内容挑战方面提供了一个有前景的方向。

**Key Findings:**

- To address this, we propose
DF-LLaVA, a simple yet effective framework that unlocks the intrinsic
discrimination potential of MLLMs. Our approach first extracts latent knowledge
from MLLMs and then injects it into training via prompts.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.14957v1)
- [arXiv](https://arxiv.org/abs/2509.14957v1)

---

<a id='2509.14746v1'></a>
## [Chain-of-Thought Re-ranking for Image Retrieval Tasks](https://arxiv.org/abs/2509.14746v1)

**Authors:** Shangrong Wu, Yanghong Zhou, Yang Chen, Feng Zhang, P. Y. Mok

**Published:** 2025-09-18

**Categories:** cs.CV, cs.IR

**Abstract:**

Image retrieval remains a fundamental yet challenging problem in computer
vision. While recent advances in Multimodal Large Language Models (MLLMs) have
demonstrated strong reasoning capabilities, existing methods typically employ
them only for evaluation, without involving them directly in the ranking
process. As a result, their rich multimodal reasoning abilities remain
underutilized, leading to suboptimal performance. In this paper, we propose a
novel Chain-of-Thought Re-Ranking (CoTRR) method to address this issue.
Specifically, we design a listwise ranking prompt that enables MLLM to directly
participate in re-ranking candidate images. This ranking process is grounded in
an image evaluation prompt, which assesses how well each candidate aligns with
users query. By allowing MLLM to perform listwise reasoning, our method
supports global comparison, consistent reasoning, and interpretable
decision-making - all of which are essential for accurate image retrieval. To
enable structured and fine-grained analysis, we further introduce a query
deconstruction prompt, which breaks down the original query into multiple
semantic components. Extensive experiments on five datasets demonstrate the
effectiveness of our CoTRR method, which achieves state-of-the-art performance
across three image retrieval tasks, including text-to-image retrieval (TIR),
composed image retrieval (CIR) and chat-based image retrieval (Chat-IR). Our
code is available at https://github.com/freshfish15/CoTRR .

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Shangrong Wu等人撰写的论文“Chain-of-Thought Re-ranking for Image Retrieval Tasks”的全面摘要。

---

### 论文《Chain-of-Thought Re-ranking for Image Retrieval Tasks》摘要

**1. 主要问题或研究问题：**
图像检索是计算机视觉领域的一个基础且具有挑战性的问题。尽管多模态大型语言模型（MLLMs）在推理能力方面取得了显著进展，但现有方法通常仅将MLLMs用于评估，而未直接参与图像的排序过程。这导致MLLMs丰富的多模态推理能力未被充分利用，从而影响了图像检索的性能。本文旨在解决这一问题，即如何有效利用MLLMs的推理能力，使其直接参与图像检索的重排序过程，以提升检索准确性。

**2. 关键创新或方法论贡献：**
本文提出了一种新颖的**思维链重排序（Chain-of-Thought Re-Ranking, CoTRR）**方法，以解决上述问题。CoTRR的核心创新和方法论贡献包括：

*   **MLLM直接参与重排序：** CoTRR设计了一个**列表式排序提示（listwise ranking prompt）**，使MLLM能够直接参与候选图像的重排序，而非仅仅进行评估。这使得MLLM能够进行全局比较和一致性推理。
*   **图像评估提示：** 排序过程基于一个**图像评估提示（image evaluation prompt）**，该提示评估每个候选图像与用户查询的匹配程度。这种评估方式支持可解释的决策制定。
*   **查询解构提示：** 为了实现结构化和细粒度的分析，CoTRR引入了一个**查询解构提示（query deconstruction prompt）**。该提示将原始查询分解为多个语义组件（例如，主要对象、活动、关键细节、环境、氛围），从而实现更有效和准确的匹配比较。
*   **统一框架：** CoTRR不依赖于特定的初始检索方法，使其能够轻松、无缝地应用于多种图像检索任务，包括文本到图像检索（TIR）、组合图像检索（CIR）和基于聊天的图像检索（Chat-IR）。

**3. 主要结果及其意义：**
CoTRR在五个数据集上进行了广泛实验，并在三种图像检索任务（TIR、CIR和Chat-IR）上取得了**最先进的性能**。

*   **CIR任务：** 在CIRR和CIRCO数据集上，CoTRR在R@1和mAP@k等指标上显著优于基线方法（如OSrCIR和ImageScope），尤其是在R@1指标上取得了显著提升。例如，使用ViT-B/32作为骨干网络时，CoTRR在CIRR数据集上的R@1和R@5分别比ImageScope提升了12.41%和6.53%。
*   **TIR任务：** 在Flickr30K和MSCOCO数据集上，CoTRR同样超越了原始CLIP和ImageScope，验证了其在文本到图像检索中的有效性。
*   **Chat-IR任务：** 在VisDial数据集上，CoTRR在多个对话轮次中持续优于OpenCLIP、PlugIR和ImageScope，表明其在交互式检索场景中的强大能力。
*   **消融研究：** 消融实验证实了CoTRR中列表式排序、查询解构和图像评估模块的互补性和有效性，每个组件都对性能提升有贡献。此外，CoTRR在不同MLLM（如Gemini 2.5 Pro、Qwen-VL-Max）上的表现稳健，其中Gemini 2.5 Pro表现最佳，这表明大型或指令对齐更好的MLLM能提供更强的基础和评估能力。

这些结果表明，CoTRR通过将MLLM直接整合到重排序流程中，并利用其强大的推理能力进行细致的图像评估和列表式比较，显著提升了图像检索的准确性和鲁棒性。

**4. 论文中提到的局限性：**
论文中未明确提及CoTRR方法的具体局限性。然而，从其方法论和实验设置来看，可能存在的隐性局限包括：

*   **计算成本：** 依赖大型MLLM进行思维链推理和列表式重排序可能带来较高的计算成本和延迟，尤其是在处理大规模候选集时。
*   **提示工程的敏感性：** CoTRR的性能可能对提示的设计（如查询解构和图像评估提示）敏感，需要精细的提示工程来优化。
*   **MLLM的通用性：** 尽管CoTRR在不同MLLM上表现稳健，但其性能仍受底层MLLM能力的限制。如果MLLM在特定领域或复杂推理任务上表现不佳，CoTRR的性能也可能受影响。
*   **可扩展性：** 尽管论文提到了处理top-K候选集，但对于非常大的检索结果集，将所有候选图像输入MLLM进行列表式排序可能面临可扩展性挑战。

**5. 潜在的未来研究方向：**
基于本论文的工作，未来研究可以探索以下方向：

*   **效率优化：** 探索更高效的MLLM推理策略或近似方法，以降低CoTRR的计算成本和延迟，使其适用于实时或大规模检索系统。
*   **自适应提示：** 研究如何动态生成或自适应调整查询解构和图像评估提示，以更好地适应不同查询类型、用户意图和图像内容。
*   **多模态反馈：** 探索除了文本评估之外，如何将其他模态（如视觉注意力图、用户交互行为）的反馈整合到MLLM的重排序过程中。
*   **更复杂的推理：** 将CoTRR扩展到更复杂的图像检索场景，例如多轮对话中涉及更深层次语义理解和上下文推理的任务。
*   **少样本/零样本学习：** 进一步探索CoTRR在少样本或零样本场景下的性能，以及如何通过少量示例或无需额外训练来提升其泛化能力。
*   **用户偏好学习：** 结合用户偏好学习机制，使CoTRR能够根据个体用户的历史行为和反馈进行个性化重排序。

---

**Key Findings:**

- In this paper, we propose a
novel Chain-of-Thought Re-Ranking (CoTRR) method to address this issue.
- By allowing MLLM to perform listwise reasoning, our method
supports global comparison, consistent reasoning, and interpretable
decision-making - all of which are essential for accurate image retrieval.
- Extensive experiments on five datasets demonstrate the
effectiveness of our CoTRR method, which achieves state-of-the-art performance
across three image retrieval tasks, including text-to-image retrieval (TIR),
composed image retrieval (CIR) and chat-based image retrieval (Chat-IR).

**Links:**

- [PDF](https://arxiv.org/pdf/2509.14746v1)
- [arXiv](https://arxiv.org/abs/2509.14746v1)

---

