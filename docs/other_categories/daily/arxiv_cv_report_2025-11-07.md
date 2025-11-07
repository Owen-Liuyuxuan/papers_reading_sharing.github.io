time: 20251107

# Arxiv Computer Vision Papers - 2025-11-07

## Executive Summary

好的，这是一份为忙碌的研究人员准备的 Arxiv 计算机视觉领域每日报告执行摘要，涵盖了 2025 年 11 月 6 日发布的十篇论文。

---

**Arxiv 计算机视觉每日报告执行摘要 (2025-11-06)**

**1. 主要主题与趋势概述：**

今天的论文集呈现出计算机视觉领域向**多模态、视频理解与生成、具身智能与机器人学习**的显著融合趋势。核心主题包括：

*   **视频作为核心模态：** 视频生成、视频理解（特别是空间和时空推理）以及视频在多模态推理中的应用成为焦点。
*   **具身智能与机器人学习：** 模拟到真实 (Sim-to-Real) 迁移、机器人策略学习、以及视觉-语言-动作 (VLA) 模型的开发是重要方向。
*   **统一与规模化：** 多个工作致力于构建统一的框架、模型或库，以处理不同任务或实现大规模学习。
*   **多模态推理与生成：** 结合视觉、语言甚至动作进行更复杂的推理和生成任务。

**2. 特别重要或创新的论文亮点：**

*   **"Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm" (Jingqi Tong et al.)**: 这篇论文将视频生成提升到多模态推理的范式高度，暗示了视频生成不仅仅是内容创作，更是理解和模拟复杂世界的一种强大工具。其提出的概念性框架可能对未来研究产生深远影响。
*   **"InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation" (Jinlai Liu et al.)**: 致力于统一时空自回归建模，这对于实现高质量、长序列的视频生成和理解至关重要，可能为未来的通用视觉生成模型奠定基础。
*   **"Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions" (Kaifeng Zhang et al.)**: 创新性地将高斯泼溅 (Gaussian Splatting) 技术引入机器人模拟，特别是在软体交互方面，有望显著提升 Sim-to-Real 迁移的真实性和效率，是具身智能领域的重要突破。

**3. 新兴研究方向或技术：**

*   **视频生成作为推理范式：** 不仅仅是生成逼真视频，更是利用生成过程来解决复杂的推理问题。
*   **高斯泼溅在机器人模拟中的应用：** 突破了传统模拟器的局限，尤其在处理复杂物理交互（如软体）方面展现出巨大潜力。
*   **跨具身 (Cross-Embodiment) 学习：** 如 "X-Diffusion" 所示，将人类演示的策略迁移到不同机器人形态，是提升机器人泛化能力的关键。
*   **空间超感知 (Spatial Supersensing) 在视频中的应用：** "Cambrian-S" 和 "SIMS-V" 强调了从视频中提取更深层次、更精细的空间理解，超越了传统的目标检测和跟踪。
*   **轻量级 VLA 模型：** "Evo-1" 致力于在保持语义对齐的同时实现模型轻量化，这对于边缘设备和实时应用至关重要。

**4. 建议完整阅读的论文：**

基于上述分析，以下论文建议优先完整阅读，以深入了解其方法和潜在影响：

1.  **"Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm" (Jingqi Tong et al.)**: 了解其提出的新范式和对未来研究的启发。
2.  **"InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation" (Jinlai Liu et al.)**: 深入理解其统一时空建模的方法，对视频生成和理解有重要意义。
3.  **"Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions" (Kaifeng Zhang et al.)**: 探索高斯泼溅在机器人模拟中的创新应用及其对具身智能的影响。
4.  **"NVIDIA Nemotron Nano V2 VL" (NVIDIA et al.)**: 作为大型科技公司发布的工作，通常代表了工业界的前沿进展和工程实践，值得关注其具体架构和性能。

这份摘要旨在帮助您快速把握当前计算机视觉领域的热点和前沿进展。

---

---

## Table of Contents

1. [Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm](#2511.04570v1)
2. [NVIDIA Nemotron Nano V2 VL](#2511.03929v1)
3. [Tracking and Understanding Object Transformations](#2511.04678v1)
4. [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](#2511.04675v1)
5. [X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations](#2511.04671v1)
6. [Cambrian-S: Towards Spatial Supersensing in Video](#2511.04670v1)
7. [SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding](#2511.04668v1)
8. [Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions](#2511.04665v1)
9. [Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment](#2511.04555v1)
10. [DORAEMON: A Unified Library for Visual Object Modeling and Representation Learning at Scale](#2511.04394v1)

---

## Papers

<a id='2511.04570v1'></a>
## [Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm](https://arxiv.org/abs/2511.04570v1)

**Authors:** Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, Xinchi Chen, Jun Zhao, Xuanjing Huang, Xipeng Qiu

**Published:** 2025-11-06

**Categories:** cs.CV, cs.CL

**Abstract:**

"Thinking with Text" and "Thinking with Images" paradigm significantly
improve the reasoning ability of large language models (LLMs) and Vision
Language Models (VLMs). However, these paradigms have inherent limitations. (1)
Images capture only single moments and fail to represent dynamic processes or
continuous changes, and (2) The separation of text and vision as distinct
modalities, hindering unified multimodal understanding and generation. To
overcome these limitations, we introduce "Thinking with Video", a new paradigm
that leverages video generation models, such as Sora-2, to bridge visual and
textual reasoning in a unified temporal framework. To support this exploration,
we developed the Video Thinking Benchmark (VideoThinkBench). VideoThinkBench
encompasses two task categories: (1) vision-centric tasks (e.g., Eyeballing
Puzzles), and (2) text-centric tasks (e.g., subsets of GSM8K, MMMU). Our
evaluation establishes Sora-2 as a capable reasoner. On vision-centric tasks,
Sora-2 is generally comparable to state-of-the-art (SOTA) VLMs, and even
surpasses VLMs on several tasks, such as Eyeballing Games. On text-centric
tasks, Sora-2 achieves 92% accuracy on MATH, and 75.53% accuracy on MMMU.
Furthermore, we systematically analyse the source of these abilities. We also
find that self-consistency and in-context learning can improve Sora-2's
performance. In summary, our findings demonstrate that the video generation
model is the potential unified multimodal understanding and generation model,
positions "thinking with video" as a unified multimodal reasoning paradigm.

**Analysis:**

好的，这是一篇关于“Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm”论文的中文摘要，内容涵盖了您要求的所有要点：

**论文摘要：Thinking with Video: 视频生成作为一种有前景的多模态推理范式**

**1. 主要问题或研究问题：**
当前大型语言模型（LLMs）和视觉语言模型（VLMs）在“Thinking with Text”和“Thinking with Images”范式下取得了显著的推理能力提升。然而，这些范式存在固有限制：图像只能捕捉单一瞬间，无法表示动态过程或连续变化；文本和视觉作为独立模态，阻碍了统一的多模态理解和生成。该研究旨在解决这些限制，探索视频生成模型在统一时间框架内桥接视觉和文本推理的潜力，并将其定位为一种统一的多模态推理范式。

**2. 关键创新或方法贡献：**
*   **引入“Thinking with Video”范式：** 提出利用视频生成模型（如Sora-2）进行动态推理，通过视频生成（如绘图和想象）实现类似人类的推理能力，从而克服静态图像和模态分离的限制。
*   **开发VideoThinkBench基准：** 构建了一个全面的基准，包含两类任务：
    *   **视觉中心任务：** 包括新创建的“Eyeballing Puzzles”和“Mazes”，以及改编自“Visual Puzzles”和“ARC-AGI-2”的任务，主要评估空间推理和归纳推理能力。
    *   **文本中心任务：** 包含改编自GSM8K、MMMU等现有基准的子集，评估数学推理和通用知识推理能力，并通过视频生成模型输出文本解决方案和语音答案。
*   **系统性评估与分析：** 对Sora-2在VideoThinkBench上的性能进行了全面评估，并与最先进的VLMs进行了比较。同时，系统分析了Sora-2推理能力的来源，并探讨了自洽性（self-consistency）和上下文学习（in-context learning）对其性能的影响。

**3. 主要结果及其意义：**
*   **Sora-2作为有能力的推理器：** 在视觉中心任务上，Sora-2的性能与最先进的VLMs相当，在某些任务（如“Eyeballing Games”）上甚至超越了它们，展现出强大的空间推理和归纳能力。
*   **文本中心任务表现出色：** 在文本中心任务上，Sora-2在MATH上达到92%的准确率，在MMMU上达到75.53%的准确率，表明视频生成模型能够将文本嵌入视频帧中，实现统一的多模态理解和生成。
*   **自洽性和上下文学习的有效性：** 研究发现自洽性和上下文学习可以显著提高Sora-2的性能，尤其是在可验证的视频生成推理任务中，这为视频生成推理任务中的测试时扩展提供了新方向。
*   **推理能力来源分析：** 尽管Sora-2在视频中生成连贯推理过程仍有困难，但其文本中心推理能力可能源于内部的提示重写机制。
*   **意义：** 论文的发现表明，视频生成模型不仅是一个通用的视觉推理模型，而且在统一多模态理解和生成方面具有巨大潜力，将“Thinking with Video”定位为一种统一的多模态推理范式。

**4. 论文中提及的局限性：**
*   **Sora-2的非开源性：** 主要评估集中在Sora-2的推理能力上，但由于Sora-2不是开源模型，限制了对其内部机制的深入分析。
*   **视频生成过程的连贯性：** Sora-2在生成连贯的推理过程视频方面仍存在挑战，即使最终答案正确，视频中的步骤也可能不完全正确或难以辨认。

**5. 潜在的未来研究方向：**
*   **纳入更多视频生成模型：** 未来的评估工作将包括更多视频生成模型，特别是开源模型，以便更深入地分析其内部机制。
*   **探索其他视频模型能力：** 视频模型还有其他值得探索的能力。
*   **通过强化学习增强推理能力：** 通过带有可验证奖励的强化学习（RLVR）来扩展VideoThinkBench中的可验证任务，以增强视频模型的“Thinking with Video”能力。
*   **统一多模态训练：** 探索将文本语料库转换为视频形式的训练数据（例如，逐帧生成单词以模拟白板手写），通过在此类文本生成任务上预训练视频生成模型，使其获得文本世界知识，最终实现大规模图像-文本数据训练下的统一多模态理解和生成。

**Key Findings:**

- To
overcome these limitations, we introduce "Thinking with Video", a new paradigm
that leverages video generation models, such as Sora-2, to bridge visual and
textual reasoning in a unified temporal framework.
- To support this exploration,
we developed the Video Thinking Benchmark (VideoThinkBench).
- On vision-centric tasks,
Sora-2 is generally comparable to state-of-the-art (SOTA) VLMs, and even
surpasses VLMs on several tasks, such as Eyeballing Games.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04570v1)
- [arXiv](https://arxiv.org/abs/2511.04570v1)

---

<a id='2511.03929v1'></a>
## [NVIDIA Nemotron Nano V2 VL](https://arxiv.org/abs/2511.03929v1)

**Authors:** NVIDIA, :, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Guo Chen, Karan Sapra, Zhiding Yu, Adi Renduchintala, Charles Wang, Peter Jin, Arushi Goel, Mike Ranzinger, Lukas Voegtle, Philipp Fischer, Timo Roman, Wei Ping, Boxin Wang, Zhuolin Yang, Nayeon Lee, Shaokun Zhang, Fuxiao Liu, Zhiqi Li, Di Zhang, Greg Heinrich, Hongxu, Yin, Song Han, Pavlo Molchanov, Parth Mannan, Yao Xu, Jane Polak Scowcroft, Tom Balough, Subhashree Radhakrishnan, Paris Zhang, Sean Cha, Ratnesh Kumar, Zaid Pervaiz Bhat, Jian Zhang, Darragh Hanley, Pritam Biswas, Jesse Oliver, Kevin Vasques, Roger Waleffe, Duncan Riach, Oluwatobi Olabiyi, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Pritam Gundecha, Khanh Nguyen, Alexandre Milesi, Eugene Khvedchenia, Ran Zilberstein, Ofri Masad, Natan Bagrov, Nave Assaf, Tomer Asida, Daniel Afrimi, Amit Zuker, Netanel Haber, Zhiyu Cheng, Jingyu, Xin, Di, Wu, Nik Spirin, Maryam Moosaei, Roman Ageev, Vanshil Atul Shah, Yuting Wu, Daniel Korzekwa, Unnikrishnan Kizhakkemadam Sreekumar, Wanli Jiang, Padmavathy Subramanian, Alejandra Rico, Sandip Bhaskar, Saeid Motiian, Kedi Wu, Annie Surla, Chia-Chih Chen, Hayden Wolff, Matthew Feinberg, Melissa Corpuz, Marek Wawrzos, Eileen Long, Aastha Jhunjhunwala, Paul Hendricks, Farzan Memarian, Benika Hall, Xin-Yu Wang, David Mosallanezhad, Soumye Singhal, Luis Vega, Katherine Cheung, Krzysztof Pawelec, Michael Evans, Katherine Luna, Jie Lou, Erick Galinkin, Akshay Hazare, Kaustubh Purandare, Ann Guan, Anna Warno, Chen Cui, Yoshi Suhara, Shibani Likhite, Seph Mard, Meredith Price, Laya Sleiman, Saori Kaji, Udi Karpas, Kari Briski, Joey Conway, Michael Lightstone, Jan Kautz, Mohammad Shoeybi, Mostofa Patwary, Jonathen Cohen, Oleksii Kuchaiev, Andrew Tao, Bryan Catanzaro

**Published:** 2025-11-06

**Categories:** cs.LG, cs.AI, cs.CV

**Abstract:**

We introduce Nemotron Nano V2 VL, the latest model of the Nemotron
vision-language series designed for strong real-world document understanding,
long video comprehension, and reasoning tasks. Nemotron Nano V2 VL delivers
significant improvements over our previous model,
Llama-3.1-Nemotron-Nano-VL-8B, across all vision and text domains through major
enhancements in model architecture, datasets, and training recipes. Nemotron
Nano V2 VL builds on Nemotron Nano V2, a hybrid Mamba-Transformer LLM, and
innovative token reduction techniques to achieve higher inference throughput in
long document and video scenarios. We are releasing model checkpoints in BF16,
FP8, and FP4 formats and sharing large parts of our datasets, recipes and
training code.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供NVIDIA团队撰写的论文“NVIDIA Nemotron Nano V2 VL”的全面摘要。

**论文摘要：NVIDIA Nemotron Nano V2 VL**

**1. 主要问题或研究问题：**
该论文旨在解决现有视觉-语言模型在处理复杂真实世界文档理解、长视频理解和推理任务方面的局限性。具体来说，它寻求开发一个高效、准确的视觉-语言模型，能够显著超越现有模型（如Llama-3.1-Nemotron-Nano-VL-8B）的性能，并在长上下文场景中实现更高的推理吞吐量。

**2. 关键创新或方法论贡献：**
*   **模型架构增强：** Nemotron Nano V2 VL建立在Nemotron Nano V2（一个混合Mamba-Transformer LLM）和RADIOv2.5视觉编码器之上，并采用了类似于Eagle 2和2.5的多模态融合架构。
*   **多阶段训练策略：** 论文采用多阶段训练方法，包括：
    *   **Stage 0 (预训练)：** 预热MLP连接器，建立语言和视觉域之间的跨模态对齐。
    *   **SFT Stage 1 (16K上下文长度)：** 解冻所有模型组件，在包含高质量推理数据、扩展OCR数据集和长上下文数据集的广泛多模态数据集上进行训练。
    *   **SFT Stage 2 (49K上下文扩展)：** 将模型上下文长度扩展到49,152个token，以增强多图像和视频理解能力。
    *   **SFT Stage 3 (49K文本恢复)：** 引入额外的SFT阶段，专门用于代码推理数据，以恢复在SFT Stage 1和2中观察到的文本推理能力下降。
    *   **SFT Stage 4 (300K上下文扩展)：** 进一步扩展模型上下文，并整合长上下文数据，以适应最长的样本。
*   **创新性Token减少技术：** 采用高效视频采样（EVS）技术，通过识别和修剪时间上静态的补丁来减少视觉token数量，从而在长视频场景中实现更高的推理吞吐量，同时保持准确性。
*   **推理优化：** 支持推理开启（reasoning-on）和推理关闭（reasoning-off）模式，前者允许更复杂的任务进行扩展推理。
*   **量化策略：** 采用Transformer Engine的延迟缩放FP8进行训练，并提供PTQ（训练后量化）和QAD（量化感知蒸馏）生成的BF16、FP8和FP4格式的模型检查点，以弥合训练和服务之间的差距。
*   **数据和工具共享：** 发布了模型权重、大部分训练数据集、训练配方和训练代码，以促进持续研究和开发。

**3. 主要结果及其意义：**
*   **领先的准确性：** Nemotron Nano V2 VL在OCRBench v2私有数据排行榜上取得了领先的准确性，并在推理、文档理解、长视频理解、视觉问答和STEM推理方面表现出色。
*   **显著性能提升：** 相较于前代模型Llama-3.1-Nemotron-Nano-VL-8B，Nemotron Nano V2 VL在所有基准测试中都表现出持续的改进。
*   **长上下文能力：** 模型上下文长度从16K扩展到128K（最终训练阶段达到311,296个token），显著提升了长视频和复杂推理任务的处理能力。
*   **推理吞吐量提升：** 混合Mamba-Transformer架构在长多页文档理解场景中提供了35%的吞吐量提升。高效视频采样（EVS）在视频理解用例中将吞吐量加速了2倍或更多，且对准确性影响最小。
*   **文本推理能力恢复：** 通过专门的SFT Stage 3，成功恢复了在早期多模态训练阶段下降的文本推理能力，如LiveCodeBench和RULER分数。
*   **推理预算控制：** 实验表明，调整推理预算可以提高推理开启模式的准确性，尤其是在处理分布外任务或需要简单推理的问题时。

**4. 论文中提到的局限性：**
*   **OCRBench-V2 (English) 性能差距：** 在图像处理消融实验中，尽管原生分辨率方法在大多数基准测试中表现良好，但在OCRBench-V2 (English)上仍存在性能下降，这表明在处理某些特定类型的图像时， tiling 算法可能存在不足。论文指出未来工作将进一步调查解决此差距的策略。
*   **推理开启模式的复杂性：** 尽管推理预算控制可以提高准确性，但推理开启模式的性能仍可能受到推理链过长、重复循环或过度冗余等因素的影响。

**5. 潜在的未来研究方向：**
*   **进一步优化图像处理策略：** 解决OCRBench-V2 (English)上存在的性能差距，可能通过改进tiling算法或探索其他图像处理方法。
*   **更精细的推理预算控制：** 探索更智能的推理预算分配机制，以在不同任务中最大化性能和效率。
*   **模型泛化能力：** 持续改进模型在更广泛的真实世界场景和分布外数据上的泛化能力。
*   **多模态融合机制：** 进一步研究和优化多模态融合架构，以实现更深层次的视觉和语言理解。
*   **量化技术的进步：** 探索新的量化技术，以在保持高准确性的同时进一步提高推理效率。

总而言之，NVIDIA Nemotron Nano V2 VL代表了视觉-语言模型领域的一个重要进展，通过创新的架构、训练策略和优化技术，显著提升了在文档、视频和推理任务中的性能，并为未来的研究奠定了坚实的基础。

**Key Findings:**

- We introduce Nemotron Nano V2 VL, the latest model of the Nemotron
vision-language series designed for strong real-world document understanding,
long video comprehension, and reasoning tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.03929v1)
- [arXiv](https://arxiv.org/abs/2511.03929v1)

---

<a id='2511.04678v1'></a>
## [Tracking and Understanding Object Transformations](https://arxiv.org/abs/2511.04678v1)

**Authors:** Yihong Sun, Xinyu Yang, Jennifer J. Sun, Bharath Hariharan

**Published:** 2025-11-06

**Categories:** cs.CV

**Abstract:**

Real-world objects frequently undergo state transformations. From an apple
being cut into pieces to a butterfly emerging from its cocoon, tracking through
these changes is important for understanding real-world objects and dynamics.
However, existing methods often lose track of the target object after
transformation, due to significant changes in object appearance. To address
this limitation, we introduce the task of Track Any State: tracking objects
through transformations while detecting and describing state changes,
accompanied by a new benchmark dataset, VOST-TAS. To tackle this problem, we
present TubeletGraph, a zero-shot system that recovers missing objects after
transformation and maps out how object states are evolving over time.
TubeletGraph first identifies potentially overlooked tracks, and determines
whether they should be integrated based on semantic and proximity priors. Then,
it reasons about the added tracks and generates a state graph describing each
observed transformation. TubeletGraph achieves state-of-the-art tracking
performance under transformations, while demonstrating deeper understanding of
object transformations and promising capabilities in temporal grounding and
semantic reasoning for complex object transformations. Code, additional
results, and the benchmark dataset are available at
https://tubelet-graph.github.io.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Yihong Sun等人撰写的论文“Tracking and Understanding Object Transformations”的全面摘要。

---

### 论文摘要：跟踪和理解物体变换

**1. 核心问题或研究问题**

该论文旨在解决现有物体跟踪方法在物体发生显著状态变换（如苹果被切开、毛毛虫变为蝴蝶）时失效的问题。传统的跟踪器主要依赖物体外观，当外观发生剧烈变化时，它们往往会丢失目标。因此，作者提出了“跟踪任何状态”（Track Any State）的新任务：在检测和描述状态变化的同时，跟踪经历变换的物体。

**2. 关键创新或方法贡献**

为了解决“跟踪任何状态”任务，作者提出了一个名为 **TubeletGraph** 的零样本系统，其主要创新点包括：

*   **时空分区（Spatiotemporal Partition）**：TubeletGraph 首先将视频划分为一系列“tubelets”（部分轨迹），这些tubelets由SAM2等现有模型生成，并在外观变化处进行划分。对于视频中未被跟踪的区域，系统会主动初始化新的轨迹，从而大大减少了搜索空间，提高了物体检索的可能性。
*   **语义和空间邻近性推理**：为了识别真正的变换后物体（而非无关物体），TubeletGraph 引入了两个先验：
    *   **空间邻近性（Spatial Proximity）**：利用SAM2预测的多个候选掩码，评估候选轨迹与原始提示物体轨迹在空间上的重叠程度，假设变换后的物体不会在短时间内剧烈改变位置。
    *   **语义一致性（Semantic Consistency）**：通过计算掩码CLIP特征的语义相似性，确保候选实体与提示物体之间存在语义对齐，假设物体身份和语义不会因变换而显著改变。
*   **状态图生成**：当检测到新的tubelets出现时，TubeletGraph 将其作为状态变换的标志。然后，它利用多模态大型语言模型（如GPT-4）来描述变换过程和产生的物体，并构建一个结构化的状态图，以自然语言形式表示物体随时间演变的状态。

**3. 主要结果及其意义**

*   **物体跟踪性能**：TubeletGraph 在VOST数据集上实现了最先进的跟踪性能，尤其是在物体变换场景下。与SAM2等基线模型相比，TubeletGraph 在召回率（R）方面有显著提升，表明其能有效恢复变换后丢失的物体。通过引入语义和空间邻近性约束，系统在保持高召回率的同时，也显著提高了精度（P）。
*   **状态图理解能力**：论文引入了VOST-TAS（Track Any State）新基准数据集，用于评估变换的检测和描述能力。TubeletGraph 在时间定位（Tp和TR）和语义准确性（Av和Ao）方面表现出色，能够准确识别变换发生的时间边界，并用自然语言描述动作动词和产生的物体。
*   **零样本泛化能力**：TubeletGraph 是一个零样本系统，无需针对特定变换域进行微调，展现了良好的泛化能力。

**4. 论文中提及的局限性**

*   **计算效率**：TubeletGraph 的主要效率瓶颈在于构建时空分区，即跟踪每个空间区域，这导致每帧平均需要7秒的计算时间，限制了其在实时应用中的使用。
*   **错误归因和诊断**：模块化设计可能导致系统性错误归因和诊断的潜在挑战。
*   **对非外观变化变换的检测**：对于不改变物体外观的变换，准确的跟踪反而会阻止变换检测，导致时间定位的召回率较低。
*   **遮挡事件**：在遮挡事件中，TubeletGraph 会添加额外的tubelets来恢复暂时丢失的目标物体，但由于VLM只观察可见帧，变换不会被描述为遮挡事件。
*   **与微调模型的互补性**：与在VOST上微调的SAM2.1(ft)结合时，TubeletGraph的改进不如预期显著，因为SAM2.1(ft)本身已经旨在最小化假阴性，从而减少了互补效益。

**5. 潜在的未来研究方向**

*   **提高计算效率**：优化时空分区构建过程，以实现更接近实时的性能。
*   **更全面的变换检测**：探索能够检测不改变物体外观的变换的方法。
*   **更鲁棒的错误处理**：改进系统对基础跟踪器产生的假阳性错误的鲁棒性。
*   **更精细的语义推理**：进一步提升VLM在复杂场景下对物体和动作的语义理解能力。
*   **多物体跟踪**：将时空分区适应于多物体跟踪，以分摊计算成本。

---

总而言之，这篇论文通过引入“跟踪任何状态”任务和TubeletGraph系统，在物体跟踪领域取得了重要进展。TubeletGraph通过创新的时空分区、语义和空间推理以及状态图生成，成功解决了物体在变换过程中跟踪丢失的问题，并能以自然语言描述这些变换。尽管存在计算效率等局限性，但该工作为机器人、视频编辑和场景建模等下游任务提供了更鲁棒、信息更丰富的跟踪系统，并为未来研究指明了方向。

**Key Findings:**

- To address
this limitation, we introduce the task of Track Any State: tracking objects
through transformations while detecting and describing state changes,
accompanied by a new benchmark dataset, VOST-TAS.
- TubeletGraph achieves state-of-the-art tracking
performance under transformations, while demonstrating deeper understanding of
object transformations and promising capabilities in temporal grounding and
semantic reasoning for complex object transformations.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04678v1)
- [arXiv](https://arxiv.org/abs/2511.04678v1)

---

<a id='2511.04675v1'></a>
## [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](https://arxiv.org/abs/2511.04675v1)

**Authors:** Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, Zehuan Yuan

**Published:** 2025-11-06

**Categories:** cs.CV

**Abstract:**

We introduce InfinityStar, a unified spacetime autoregressive framework for
high-resolution image and dynamic video synthesis. Building on the recent
success of autoregressive modeling in both vision and language, our purely
discrete approach jointly captures spatial and temporal dependencies within a
single architecture. This unified design naturally supports a variety of
generation tasks such as text-to-image, text-to-video, image-to-video, and long
interactive video synthesis via straightforward temporal autoregression.
Extensive experiments demonstrate that InfinityStar scores 83.74 on VBench,
outperforming all autoregressive models by large margins, even surpassing some
diffusion competitors like HunyuanVideo. Without extra optimizations, our model
generates a 5s, 720p video approximately 10x faster than leading
diffusion-based methods. To our knowledge, InfinityStar is the first discrete
autoregressive video generator capable of producing industrial level 720p
videos. We release all code and models to foster further research in efficient,
high-quality video generation.

**Analysis:**

以下是Jinlai Liu等人在“InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation”论文的全面摘要：

**1. 解决的主要问题或研究问题：**
该论文旨在解决高分辨率图像和动态视频合成中的挑战，特别是现有自回归模型在视觉保真度和推理效率方面的不足，以及扩散模型在生成速度和视频外推方面的局限性。研究目标是开发一个统一的生成框架，能够同时实现高质量视觉效果、高效率和良好的时间泛化能力。

**2. 关键创新或方法论贡献：**
*   **InfinityStar框架：** 提出了一种新颖的时空金字塔建模框架，统一了文本到图像、文本到视频、图像到视频以及长交互式视频合成等多种视觉生成任务。该框架将视频建模为图像金字塔和多剪辑金字塔，自然继承了文本到图像的能力，并解耦了视频中的静态外观和动态运动。
*   **改进的离散视频分词器：**
    *   **知识继承：** 通过继承预训练的连续视频分词器（VAE）的架构和权重，显著提高了离散重建质量并加速了收敛。
    *   **随机量化器深度（SQD）：** 在训练分词器时引入SQD，以缓解不同尺度间信息分布不平衡的问题，迫使分词器在早期尺度中存储更多信息，从而改善早期尺度的重建质量和VAR Transformer的优化。
*   **时空自回归Transformer的改进：**
    *   **语义尺度重复（SSR）：** 通过重复早期语义尺度（决定整体布局和前景对象位置的尺度）的token，增强了生成视频的结构一致性和运动动态，同时计算开销可忽略不计。
    *   **时空稀疏注意力（SSA）：** 针对长视频生成中的高计算成本问题，提出SSA，仅关注前一个剪辑的最后一个尺度，显著降低了训练和推理过程中的计算开销，并提高了性能。
    *   **时空旋转位置嵌入（Spacetime RoPE）：** 引入Spacetime RoPE，将原始旋转嵌入分解为尺度、时间、高度和宽度四个分量，以增强复杂位置信息的建模，并支持外推。
*   **长交互式视频生成扩展：** 采用滑动窗口方法和语义-细节条件（Semantic-Detail conditions），通过提取前一个剪辑的语义和细节特征，实现长视频的语义一致性和细节保持，支持多轮交互式视频生成。

**3. 主要结果及其意义：**
*   **性能领先：** InfinityStar在VBench基准测试中得分83.74，显著优于所有自回归模型，甚至超越了HunyuanVideo等一些领先的扩散模型。在GenEval和DPG基准测试中，InfinityStar-T2I也取得了最佳整体分数，超越了Infinity。
*   **推理效率高：** 在相同压缩率下，InfinityStar生成5秒720p视频的速度比领先的扩散模型快约10倍，比Wan-2.1快32倍，比Nova快6倍。SSA进一步将192p 161帧视频的生成速度提高了1.5倍。
*   **工业级视频生成：** InfinityStar是首个能够生成工业级720p视频的离散自回归视频生成器。
*   **统一生成能力：** 成功地将文本到图像、文本到视频、图像到视频和视频外推等任务统一在一个模型中，展示了卓越的灵活性和多功能性。
*   **人类偏好评估：** 在T2V和I2V任务中，InfinityStar-8B在所有评估指标上均优于HunyuanVideo-13B，尤其是在提示遵循和整体质量方面。

**4. 论文中提到的局限性：**
*   **图像质量与运动保真度的权衡：** 在高运动场景中，有时精细的视觉细节可能会受到影响。
*   **计算资源限制：** 由于计算资源有限，模型训练和参数规模尚未达到领先扩散模型的水平，这限制了性能的上限。
*   **推理管线未完全优化：** 仍有进一步改进的空间。
*   **长交互式视频生成中的累积误差：** 随着交互次数的增加，生成视频的质量可能会出现明显的下降。

**5. 潜在的未来研究方向：**
*   进一步优化推理管线，提高效率。
*   解决长交互式视频生成中的累积误差问题。
*   继续探索高效、高质量视频生成领域。
*   通过发布代码和模型，促进对高效、高质量视频生成的进一步研究。

这篇论文通过引入InfinityStar框架，在统一的自回归建模下，显著提升了高分辨率图像和视频合成的质量和效率，为计算机视觉领域带来了重要的进展。

**Key Findings:**

- We introduce InfinityStar, a unified spacetime autoregressive framework for
high-resolution image and dynamic video synthesis.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04675v1)
- [arXiv](https://arxiv.org/abs/2511.04675v1)

---

<a id='2511.04671v1'></a>
## [X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations](https://arxiv.org/abs/2511.04671v1)

**Authors:** Maximus A. Pace, Prithwish Dan, Chuanruo Ning, Atiksh Bhardwaj, Audrey Du, Edward W. Duan, Wei-Chiu Ma, Kushal Kedia

**Published:** 2025-11-06

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Human videos can be recorded quickly and at scale, making them an appealing
source of training data for robot learning. However, humans and robots differ
fundamentally in embodiment, resulting in mismatched action execution. Direct
kinematic retargeting of human hand motion can therefore produce actions that
are physically infeasible for robots. Despite these low-level differences,
human demonstrations provide valuable motion cues about how to manipulate and
interact with objects. Our key idea is to exploit the forward diffusion
process: as noise is added to actions, low-level execution differences fade
while high-level task guidance is preserved. We present X-Diffusion, a
principled framework for training diffusion policies that maximally leverages
human data without learning dynamically infeasible motions. X-Diffusion first
trains a classifier to predict whether a noisy action is executed by a human or
robot. Then, a human action is incorporated into policy training only after
adding sufficient noise such that the classifier cannot discern its embodiment.
Actions consistent with robot execution supervise fine-grained denoising at low
noise levels, while mismatched human actions provide only coarse guidance at
higher noise levels. Our experiments show that naive co-training under
execution mismatches degrades policy performance, while X-Diffusion
consistently improves it. Across five manipulation tasks, X-Diffusion achieves
a 16% higher average success rate than the best baseline. The project website
is available at https://portal-cornell.github.io/X-Diffusion/.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行详细分析。

---

### 论文摘要分析：X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations

**1. 论文主要贡献的简洁总结 (2-3 句话)**

这篇论文提出了一种名为 X-Diffusion 的新框架，旨在有效利用大规模人类演示视频来训练机器人策略，同时克服人类和机器人之间固有的本体差异（embodiment mismatch）。其核心思想是利用扩散过程的特性，在高噪声水平下从人类数据中提取高级任务指导，而在低噪声水平下则专注于机器人可执行的精细动作去噪，从而避免学习到机器人无法执行的动作。实验结果表明，X-Diffusion 显著提升了机器人操纵任务的成功率。

**2. 关键创新或方法论**

X-Diffusion 的关键创新在于其巧妙地利用了**扩散过程（diffusion process）**来解决跨本体数据利用的挑战。具体方法如下：

*   **本体分类器（Embodiment Classifier）:** 首先训练一个分类器，用于预测一个带噪声的动作是由人类还是机器人执行的。这个分类器是区分不同本体动作的关键。
*   **噪声引导的本体无关性（Noise-Guided Embodiment Agnosticism）:** 当将人类动作纳入策略训练时，只有在添加了足够的噪声，使得上述分类器无法辨别其本体（即无法区分是人类还是机器人）之后，才将其用于训练。这意味着在高噪声水平下，人类动作的低级执行细节被噪声模糊，只保留了高级的任务意图。
*   **分层去噪监督（Hierarchical Denoising Supervision）:**
    *   **高噪声水平：** 此时，人类动作（经过充分加噪后）提供粗粒度的指导，帮助策略理解任务的整体目标和操作流程。
    *   **低噪声水平：** 此时，策略的去噪过程由与机器人执行一致的动作进行监督，确保学习到的动作是机器人可物理执行的。

这种方法避免了直接的运动重定向（kinematic retargeting）可能导致的不可行动作问题，并允许策略在不同噪声水平下从不同来源的数据中学习不同粒度的信息。

**3. 对领域潜在影响**

X-Diffusion 对机器人学习和计算机视觉领域具有以下潜在影响：

*   **解锁大规模人类视频数据：** 极大地提高了利用现有和未来大规模人类视频数据训练机器人策略的效率和可行性。这可以加速机器人技能的学习，降低数据采集成本。
*   **弥合本体鸿沟：** 提供了一种原则性的方法来解决机器人学习中长期存在的本体差异问题，使得人类直观的演示能够更有效地转化为机器人可执行的策略。
*   **推动扩散模型在机器人领域的应用：** 进一步展示了扩散模型在生成和策略学习方面的强大潜力，特别是在处理异构数据源方面。
*   **促进通用机器人学习：** 通过更有效地利用人类经验，有助于开发出更通用、更灵活的机器人，能够适应各种未见过的任务和环境。

**4. 相关领域或应用**

这项研究可能惠及以下相关领域或应用：

*   **机器人操纵（Robotic Manipulation）：** 这是论文直接关注的领域，将显著提高机器人执行复杂操纵任务的能力。
*   **具身智能（Embodied AI）：** 为具身智能体从人类行为中学习提供了一种新的范式。
*   **人机协作（Human-Robot Collaboration）：** 机器人可以更好地理解和预测人类意图，从而实现更流畅的协作。
*   **虚拟现实/增强现实（VR/AR）中的机器人控制：** 允许用户通过自然的人类动作来控制虚拟或真实世界的机器人。
*   **自动驾驶（Autonomous Driving）：** 虽然摘要未直接提及，但从人类驾驶视频中学习高级驾驶策略，同时避免学习到人类特有的、车辆无法执行的微观操作，可能是一个潜在的扩展方向。
*   **运动生成与模仿学习（Motion Generation and Imitation Learning）：** 为从非匹配本体数据中生成逼真且可执行的运动提供了新的思路。

**5. 从摘要中推断出的局限性**

尽管 X-Diffusion 表现出色，但从摘要中仍可推断出一些潜在局限性：

*   **本体分类器的鲁棒性：** 训练一个能够准确区分带噪声动作本体的分类器可能具有挑战性，尤其是在噪声水平接近模糊界限时。分类器的性能直接影响了策略学习的质量。
*   **“足够噪声”的定义和调优：** 如何确定“足够的噪声”使得分类器无法辨别本体是一个关键的超参数，其选择可能需要精细的调优，并且可能因任务和本体差异程度而异。
*   **高级任务指导的粒度：** 在高噪声水平下，人类动作提供的“粗粒度指导”可能在某些需要非常精细协调的任务中不够具体。如果任务对初始姿态或轨迹的精确性要求极高，这种粗粒度指导可能不足。
*   **泛化能力：** 论文提到了在五项操纵任务上的成功，但其在更广泛、更复杂的任务（例如，需要长期规划、多阶段操作或与环境进行复杂交互的任务）上的泛化能力仍需进一步验证。
*   **计算成本：** 训练一个额外的本体分类器，并可能需要更复杂的扩散模型训练流程，可能会增加整体的计算成本和训练时间。
*   **数据需求：** 尽管旨在利用人类视频，但仍然需要一定量的机器人数据来监督低噪声水平下的精细去噪，这可能限制其在完全零样本机器人学习场景中的应用。

---

总而言之，X-Diffusion 提出了一种非常新颖且有前景的方法，通过巧妙地利用扩散模型来解决机器人学习中跨本体数据利用的根本性挑战。它有望显著加速机器人从人类演示中学习复杂技能的过程，是机器人学习领域的一个重要进展。

**Key Findings:**

- We present X-Diffusion, a
principled framework for training diffusion policies that maximally leverages
human data without learning dynamically infeasible motions.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04671v1)
- [arXiv](https://arxiv.org/abs/2511.04671v1)

---

<a id='2511.04670v1'></a>
## [Cambrian-S: Towards Spatial Supersensing in Video](https://arxiv.org/abs/2511.04670v1)

**Authors:** Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, Daohan Lu, Rob Fergus, Yann LeCun, Li Fei-Fei, Saining Xie

**Published:** 2025-11-06

**Categories:** cs.CV

**Abstract:**

We argue that progress in true multimodal intelligence calls for a shift from
reactive, task-driven systems and brute-force long context towards a broader
paradigm of supersensing. We frame spatial supersensing as four stages beyond
linguistic-only understanding: semantic perception (naming what is seen),
streaming event cognition (maintaining memory across continuous experiences),
implicit 3D spatial cognition (inferring the world behind pixels), and
predictive world modeling (creating internal models that filter and organize
information). Current benchmarks largely test only the early stages, offering
narrow coverage of spatial cognition and rarely challenging models in ways that
require true world modeling. To drive progress in spatial supersensing, we
present VSI-SUPER, a two-part benchmark: VSR (long-horizon visual spatial
recall) and VSC (continual visual spatial counting). These tasks require
arbitrarily long video inputs yet are resistant to brute-force context
expansion. We then test data scaling limits by curating VSI-590K and training
Cambrian-S, achieving +30% absolute improvement on VSI-Bench without
sacrificing general capabilities. Yet performance on VSI-SUPER remains limited,
indicating that scale alone is insufficient for spatial supersensing. We
propose predictive sensing as a path forward, presenting a proof-of-concept in
which a self-supervised next-latent-frame predictor leverages surprise
(prediction error) to drive memory and event segmentation. On VSI-SUPER, this
approach substantially outperforms leading proprietary baselines, showing that
spatial supersensing requires models that not only see but also anticipate,
select, and organize experience.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行深入分析。

---

**论文摘要分析：Cambrian-S: Towards Spatial Supersensing in Video**

**1. 论文主要贡献的简洁总结 (2-3 句话)**

这篇论文的核心贡献在于提出并倡导“空间超感知”（spatial supersensing）这一更广阔的范式，超越了当前以任务驱动和长上下文为主流的系统。为了推动这一范式的发展，作者引入了VSI-SUPER基准测试（包含VSR和VSC两部分），旨在挑战模型在任意长视频输入下的空间认知和世界建模能力，并展示了即使通过大规模数据训练的Cambrian-S模型，在这些任务上仍面临显著挑战，暗示仅靠规模不足以实现真正的超感知。论文进一步提出“预测感知”（predictive sensing）作为前进方向，通过一个概念验证展示了基于预测误差的自监督方法在VSI-SUPER上超越了现有基线。

**2. 关键创新或方法论**

*   **提出“空间超感知”范式：** 这是论文最核心的创新，将空间认知分解为四个阶段：语义感知、流式事件认知、隐式3D空间认知和预测世界建模，为未来多模态智能设定了更高层次的目标。
*   **VSI-SUPER基准测试：** 针对现有基准测试的局限性，设计了VSR（长时视觉空间回忆）和VSC（连续视觉空间计数）这两个新任务，它们对任意长视频输入具有鲁棒性，且难以通过暴力上下文扩展来解决，旨在真正考验模型的空间认知和世界建模能力。
*   **“预测感知”方法论：** 提出并初步验证了利用自监督的“下一潜在帧预测器”（next-latent-frame predictor）来驱动记忆和事件分割，通过“惊喜”（预测误差）来组织和过滤信息。这是一种从根本上不同于传统监督学习的方法，旨在模仿生物智能中对新奇事物的关注。
*   **Cambrian-S模型和VSI-590K数据集：** 虽然摘要指出规模本身不足，但构建大规模数据集VSI-590K并训练Cambrian-S模型，探索了数据规模的极限，并为后续研究提供了强大的基线和数据资源。

**3. 对领域潜在影响**

*   **重新定义智能评估标准：** “空间超感知”范式的提出，将促使研究者重新思考和设计更具挑战性的基准测试，超越当前以“识别”和“理解”为主的狭隘任务，转向更接近人类认知的“感知”、“记忆”、“推理”和“预测”。
*   **推动长视频理解和世界建模：** VSI-SUPER基准的引入将直接推动模型在处理极长视频序列、维持长期记忆、理解复杂事件流以及构建内部世界模型方面的研究。
*   **启发新的模型架构和学习范式：** “预测感知”和利用“惊喜”驱动学习的理念，可能会催生新的自监督学习方法、记忆机制和注意力机制，使模型能够更主动地选择和组织经验，而非被动地处理所有输入。
*   **促进多模态智能发展：** 论文明确指出这是“真正的多模态智能”的进步，空间超感知将是实现更通用、更具鲁棒性AI系统的关键组成部分。

**4. 相关领域或应用受益**

*   **具身智能与机器人学：** 机器人需要对环境进行持续的空间感知、记忆和预测，以进行导航、操作和人机交互。
*   **自动驾驶：** 车辆需要理解复杂的交通场景、预测其他车辆和行人的行为，并构建实时的3D世界模型。
*   **视频内容分析与生成：** 更深层次的视频理解将有助于更智能的视频检索、摘要、事件检测，甚至生成更连贯、更符合物理规律的视频内容。
*   **虚拟现实/增强现实：** 构建精确的3D世界模型和预测用户行为，对于沉浸式体验至关重要。
*   **通用人工智能（AGI）研究：** 空间超感知是迈向更接近人类智能的关键一步，因为它涉及对世界的内在理解和预测能力。

**5. 从摘要中可推断的局限性**

*   **“规模不足”的挑战：** 摘要明确指出“规模本身不足以实现空间超感知”，这意味着即使拥有大规模数据和强大的模型，当前的方法在VSI-SUPER上仍表现有限。这暗示了需要更根本的算法或架构创新，而不仅仅是扩大模型或数据。
*   **“预测感知”仍是概念验证：** 尽管“预测感知”方法在VSI-SUPER上表现出色，但摘要将其描述为“概念验证”（proof-of-concept），表明它可能仍处于早期阶段，需要进一步的理论和实证工作来验证其通用性和鲁棒性。
*   **任务复杂性与可解释性：** VSI-SUPER任务（长时视觉空间回忆和连续视觉空间计数）的复杂性可能使得模型内部决策过程难以解释，尤其是在涉及“隐式3D空间认知”和“预测世界建模”时。
*   **计算资源需求：** 处理“任意长视频输入”和训练大规模模型（如Cambrian-S）以及自监督预测器，无疑需要巨大的计算资源，这可能限制了小型研究团队的参与。
*   **“惊喜”机制的普适性：** 利用预测误差作为“惊喜”来驱动记忆和事件分割，虽然直观，但在不同类型的空间认知任务中，其有效性和最优实现方式可能需要进一步探索。

---

总而言之，这篇论文通过提出“空间超感知”这一宏大愿景和VSI-SUPER这一挑战性基准，为计算机视觉和机器学习领域设定了新的研究方向。它强调了超越当前任务驱动和长上下文方法的必要性，并初步探索了“预测感知”这一富有前景的路径，旨在推动模型从被动观察者转变为主动的、具有世界模型的感知者。这无疑是一项具有前瞻性和潜在颠覆性的工作。

**Key Findings:**

- On VSI-SUPER, this
approach substantially outperforms leading proprietary baselines, showing that
spatial supersensing requires models that not only see but also anticipate,
select, and organize experience.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04670v1)
- [arXiv](https://arxiv.org/abs/2511.04670v1)

---

<a id='2511.04668v1'></a>
## [SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding](https://arxiv.org/abs/2511.04668v1)

**Authors:** Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, Saining Xie

**Published:** 2025-11-06

**Categories:** cs.CV

**Abstract:**

Despite impressive high-level video comprehension, multimodal language models
struggle with spatial reasoning across time and space. While current spatial
training approaches rely on real-world video data, obtaining diverse footage
with precise spatial annotations remains a bottleneck. To alleviate this
bottleneck, we present SIMS-V -- a systematic data-generation framework that
leverages the privileged information of 3D simulators to create spatially-rich
video training data for multimodal language models. Using this framework, we
investigate which properties of simulated data drive effective real-world
transfer through systematic ablations of question types, mixes, and scales. We
identify a minimal set of three question categories (metric measurement,
perspective-dependent reasoning, and temporal tracking) that prove most
effective for developing transferable spatial intelligence, outperforming
comprehensive coverage despite using fewer question types. These insights
enable highly efficient training: our 7B-parameter video LLM fine-tuned on just
25K simulated examples outperforms the larger 72B baseline and achieves
competitive performance with proprietary models on rigorous real-world spatial
reasoning benchmarks. Our approach demonstrates robust generalization,
maintaining performance on general video understanding while showing
substantial improvements on embodied and real-world spatial tasks.

**Analysis:**

以下是Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, Saining Xie撰写的论文“SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding”的全面摘要：

**论文题目：** SIMS-V: Simulated Instruction-Tuning for Spatial Video Understanding

**作者：** Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, Saining Xie

**摘要：**

1.  **主要问题或研究问题：**
    尽管多模态语言模型（MLLMs）在高级视频理解方面表现出色，但在跨时间和空间的**空间推理**方面仍面临挑战。当前的空间训练方法依赖于真实世界的视频数据，但获取具有精确空间标注的多样化视频素材是一个瓶颈。因此，该研究旨在解决如何高效、有效地为多模态语言模型生成空间丰富的视频训练数据，以提升其在真实世界视频中的空间推理能力，并探究哪些模拟数据特性能够驱动有效的真实世界迁移。

2.  **关键创新或方法论贡献：**
    *   **SIMS-V框架：** 论文提出了SIMS-V，一个系统的**数据生成框架**。该框架利用3D模拟器中的特权信息，程序化地创建空间丰富的视频训练数据，用于多模态语言模型。这解决了真实世界视频数据中精确空间标注获取困难的瓶颈。
    *   **系统性消融研究：** SIMS-V框架允许对问题类型、数据混合和规模进行系统性消融研究，以探究模拟数据的哪些特性能够驱动有效的真实世界迁移。
    *   **识别核心问题类别：** 研究发现，仅三种问题类别（**度量测量、依赖视角的推理和时间跟踪**）的最小集合在开发可迁移的空间智能方面最有效，甚至优于全面的问题类型覆盖。

3.  **主要结果及其意义：**
    *   **高效训练：** 仅使用25K个模拟示例对一个7B参数的视频LLM进行微调，其性能优于更大的72B基线模型，并在严格的真实世界空间推理基准测试中与专有模型（如Gemini-1.5 Pro）达到竞争水平。
    *   **鲁棒泛化：** 该方法展示了强大的泛化能力，在保持通用视频理解性能的同时，在具身（embodied）和真实世界空间任务上取得了显著改进（例如，在VSI-Bench上LLaVA-Video提升8.4%，在OpenEQA上提升8.6%，在MMRealWorld上提升4.5%）。
    *   **数据效率：** 3Q Minimal混合（度量测量、视角和时空跟踪）在数据效率上优于VSI-Baseline混合，在更少的数据量下实现了更好的性能。

4.  **论文中提及的局限性：**
    *   **模型架构泛化：** 目前的研究主要集中在LLaVA系列视频语言模型上，未来工作应探索这些发现是否能泛化到其他近期架构。
    *   **灾难性遗忘：** 实验主要通过在SIMS数据上进行微调，而没有与通用指令数据混合，这可能导致一定程度的灾难性遗忘。
    *   **帧采样策略：** 视频语言模型在推理时会进行帧采样（例如，3分钟视频中1800帧只采样64帧），确保训练样本中的关键空间信息在所有可能的采样帧集中保持可见，这仍是一个挑战。

5.  **潜在的未来研究方向：**
    *   **优化数据混合策略：** 探索将SIMS-VSI数据与更广泛的多模态训练数据结合的最佳策略，以最大化空间增益同时保留通用能力。
    *   **模型-数据协同设计：** 协同设计模拟训练数据，使其与模型的特定处理特性（如帧采样策略）相匹配，以进一步提高学习效率。
    *   **更大规模训练：** 在更大规模的数据上进行训练，以进一步提升模型的性能。

总而言之，SIMS-V论文提出了一种新颖且高效的模拟数据生成框架，用于训练多模态语言模型以增强其空间推理能力。通过识别关键问题类型和利用模拟器的特权信息，该方法在数据效率和真实世界迁移方面取得了显著进展，为未来视频语言模型的发展奠定了基础。

**Key Findings:**

- To alleviate this
bottleneck, we present SIMS-V -- a systematic data-generation framework that
leverages the privileged information of 3D simulators to create spatially-rich
video training data for multimodal language models.
- These insights
enable highly efficient training: our 7B-parameter video LLM fine-tuned on just
25K simulated examples outperforms the larger 72B baseline and achieves
competitive performance with proprietary models on rigorous real-world spatial
reasoning benchmarks.
- Our approach demonstrates robust generalization,
maintaining performance on general video understanding while showing
substantial improvements on embodied and real-world spatial tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04668v1)
- [arXiv](https://arxiv.org/abs/2511.04668v1)

---

<a id='2511.04665v1'></a>
## [Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions](https://arxiv.org/abs/2511.04665v1)

**Authors:** Kaifeng Zhang, Shuo Sha, Hanxiao Jiang, Matthew Loper, Hyunjong Song, Guangyan Cai, Zhuo Xu, Xiaochen Hu, Changxi Zheng, Yunzhu Li

**Published:** 2025-11-06

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Robotic manipulation policies are advancing rapidly, but their direct
evaluation in the real world remains costly, time-consuming, and difficult to
reproduce, particularly for tasks involving deformable objects. Simulation
provides a scalable and systematic alternative, yet existing simulators often
fail to capture the coupled visual and physical complexity of soft-body
interactions. We present a real-to-sim policy evaluation framework that
constructs soft-body digital twins from real-world videos and renders robots,
objects, and environments with photorealistic fidelity using 3D Gaussian
Splatting. We validate our approach on representative deformable manipulation
tasks, including plush toy packing, rope routing, and T-block pushing,
demonstrating that simulated rollouts correlate strongly with real-world
execution performance and reveal key behavioral patterns of learned policies.
Our results suggest that combining physics-informed reconstruction with
high-quality rendering enables reproducible, scalable, and accurate evaluation
of robotic manipulation policies. Website: https://real2sim-eval.github.io/

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Kaifeng Zhang等人撰写的论文“Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions”的全面摘要。

---

### 论文摘要：Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions

**1. 论文主要问题或研究问题：**
机器人操作策略的评估在现实世界中成本高昂、耗时且难以复现，尤其是在涉及可变形物体的任务中。现有模拟器往往无法准确捕捉软体交互的视觉和物理复杂性，导致“从模拟到现实”（sim-to-real）的鸿沟。因此，论文旨在解决的核心问题是：**如何设计足够真实的模拟器，以可靠地评估机器人策略，并使其模拟结果与现实世界性能高度相关？**

**2. 关键创新或方法论贡献：**
为了弥合模拟与现实之间的差距，论文提出了一个新颖的“从现实到模拟”（real-to-sim）策略评估框架，其主要创新点包括：

*   **结合3D高斯泼溅（3DGS）进行高保真渲染：** 论文利用3DGS从现实世界视频中重建逼真的场景，并支持从任意视角进行渲染。为了进一步缩小视觉差距，该框架还引入了自动位置和颜色对齐以及物体变形处理。
*   **基于PhysTwin的软体数字孪生：** 针对可变形物体动力学难以准确模拟的问题，论文采用了PhysTwin框架。PhysTwin通过优化弹簧-质量系统参数，直接从物体交互视频中重建可变形物体，从而实现与现实世界动力学的高度匹配。
*   **统一的Gym风格模拟器接口：** 论文将上述渲染和动力学组件整合到一个统一的模拟器中，并通过Gym风格的接口暴露，使得训练好的策略能够高效评估。
*   **全面的评估框架：** 论文在毛绒玩具打包、绳索穿线和T型块推动等代表性可变形操作任务上，使用ACT、Diffusion Policy、SmolVLA和Pi-0等主流模仿学习算法对方法进行了验证。

**3. 主要结果及其重要性：**
论文的实验结果展示了该框架的显著优势：

*   **高模拟-现实相关性：** 模拟器中的策略成功率与现实世界中的成功率表现出强烈的相关性（Pearson相关系数r > 0.9），这表明该模拟器能够可靠地预测现实世界的性能。
*   **优于基线：** 与IsaacLab等现有模拟器相比，该方法在模拟-现实相关性方面表现出显著更强的性能，尤其是在处理软体动力学方面。
*   **渲染和动力学保真度的关键性：** 消融研究证实，高保真的渲染（通过颜色对齐）和准确的动力学（通过物理优化）对于实现高相关性至关重要。缺乏其中任何一项都会导致模拟结果与现实世界行为的偏差。
*   **策略学习动态的预测能力：** 模拟器能够捕捉策略在训练迭代过程中的性能趋势，这使其成为监控策略学习动态、选择检查点和估算现实世界性能的实用工具。
*   **可复现性和可扩展性：** 该框架通过构建软体数字孪生和高质量渲染，实现了可复现、可扩展且准确的机器人操作策略评估。

**4. 论文中提及的局限性：**
论文也提及了一些局限性：

*   **残余的视觉和动力学差距：** 尽管论文努力缩小差距，但模拟器与现实世界之间仍存在残余的视觉和动力学差异，这可能导致在某些情况下模拟成功率与现实成功率的绝对值不完全重叠。
*   **评估集规模的统计不确定性：** 在主论文的实验中，有限的评估集规模（每个任务16-27个episode）导致了统计不确定性，尽管通过扩大模拟评估规模可以显著缩小置信区间。
*   **简化接触或摩擦模型：** 在某些任务（如毛绒玩具打包）中，模拟器倾向于轻微高估成功率，这可能归因于简化的接触或摩擦模型。

**5. 潜在的未来研究方向：**
论文提出了以下未来研究方向：

*   **扩展到更大的任务和策略集：** 将模拟和评估扩展到更大的任务和策略集，以深入了解策略评估模拟器的关键设计考量。
*   **泛化到更多样化的环境：** 将real-to-sim框架泛化到更多样化的环境，以支持日益复杂的机器人操作任务。

---

总而言之，这篇论文为机器人操作策略的评估提供了一个强大的新范式，通过结合3D高斯泼溅和物理信息重建，显著提升了模拟器的真实感和预测能力，为机器人学习领域的可扩展、可复现研究奠定了基础。

**Key Findings:**

- We present a real-to-sim policy evaluation framework that
constructs soft-body digital twins from real-world videos and renders robots,
objects, and environments with photorealistic fidelity using 3D Gaussian
Splatting.
- We validate our approach on representative deformable manipulation
tasks, including plush toy packing, rope routing, and T-block pushing,
demonstrating that simulated rollouts correlate strongly with real-world
execution performance and reveal key behavioral patterns of learned policies.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04665v1)
- [arXiv](https://arxiv.org/abs/2511.04665v1)

---

<a id='2511.04555v1'></a>
## [Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment](https://arxiv.org/abs/2511.04555v1)

**Authors:** Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu, Yinxinyu Chen, Encheng Gu, Ziyan Liu, Hongyi Cai, Yanwen Zou, Lixing Zou, Zhaoye Zhou, Gen Li, Bo Zhao

**Published:** 2025-11-06

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have emerged as a powerful framework that
unifies perception, language, and control, enabling robots to perform diverse
tasks through multimodal understanding. However, current VLA models typically
contain massive parameters and rely heavily on large-scale robot data
pretraining, leading to high computational costs during training, as well as
limited deployability for real-time inference. Moreover, most training
paradigms often degrade the perceptual representations of the vision-language
backbone, resulting in overfitting and poor generalization to downstream tasks.
In this work, we present Evo-1, a lightweight VLA model that reduces
computation and improves deployment efficiency, while maintaining strong
performance without pretraining on robot data. Evo-1 builds on a native
multimodal Vision-Language model (VLM), incorporating a novel cross-modulated
diffusion transformer along with an optimized integration module, together
forming an effective architecture. We further introduce a two-stage training
paradigm that progressively aligns action with perception, preserving the
representations of the VLM. Notably, with only 0.77 billion parameters, Evo-1
achieves state-of-the-art results on the Meta-World and RoboTwin suite,
surpassing the previous best models by 12.4% and 6.9%, respectively, and also
attains a competitive result of 94.8% on LIBERO. In real-world evaluations,
Evo-1 attains a 78% success rate with high inference frequency and low memory
overhead, outperforming all baseline methods. We release code, data, and model
weights to facilitate future research on lightweight and efficient VLA models.

**Analysis:**

好的，这是一篇关于“Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment”论文的全面摘要，由Tao Lin等人撰写。

---

### Evo-1: 轻量级视觉-语言-动作模型与语义对齐保留

**1. 主要问题或研究问题：**
当前的视觉-语言-动作（VLA）模型在实现多模态理解和机器人控制方面表现出色，但面临几个关键挑战：
*   **高计算成本和部署效率低：** 大多数VLA模型参数量巨大（数十亿），需要大规模机器人数据预训练，导致训练成本高昂，实时推理部署受限。
*   **感知表示退化：** 现有的端到端训练范式通常会损害视觉-语言骨干网络的感知表示，导致过拟合和对下游任务的泛化能力差。
*   **对大规模机器人数据预训练的依赖：** 大多数模型严重依赖劳动密集且昂贵的大规模机器人数据集进行长时间训练。

本文旨在解决这些限制，开发一个轻量级、高效的VLA模型，该模型在不依赖大规模机器人数据预训练的情况下，仍能保持强大的性能和泛化能力，同时降低计算成本和提高部署效率。

**2. 关键创新或方法论贡献：**
Evo-1通过以下创新实现了其目标：
*   **轻量级和高效的架构：** Evo-1采用仅0.77亿参数的紧凑型VLA架构，显著减少了训练成本和推理资源消耗。它基于一个原生的多模态视觉-语言模型（VLM），并结合了：
    *   **新型交叉调制扩散Transformer：** 用于生成连续控制动作，实现高效的时间推理和一致的运动生成，同时保持模型紧凑性并提高推理频率。
    *   **优化的集成模块：** 用于将融合的视觉-语言表示与机器人本体感受信息对齐，确保多模态特征的无缝整合。
*   **语义保留的两阶段训练范式：** 提出了一种两阶段训练方法，以平衡VLM固有的多模态理解能力与对下游动作生成的有效适应性。
    *   **第一阶段：动作专家对齐（Action Expert Alignment）：** 冻结整个视觉-语言骨干网络，仅训练集成模块和动作专家，使其与多模态嵌入空间逐步对齐，避免反向传播的噪声梯度破坏预训练骨干网络。
    *   **第二阶段：全尺寸微调（Full-scale Fine-Tuning）：** 一旦集成和动作模块充分对齐，则解冻VLM骨干网络并进行全尺寸微调，实现整个架构的联合优化，从而深化集成并更好地适应各种操作任务。
    *   这种方法有效保留了VLM的语义空间，防止了感知表示的退化，从而增强了泛化能力。

**3. 主要结果及其意义：**
Evo-1在模拟和真实世界任务中均取得了显著成果：
*   **Meta-World基准测试：** 在Meta-World上实现了80.6%的平均成功率，超越了之前最佳模型12.4%，并在所有四个难度级别上持续优于所有基线。
*   **RoboTwin套件：** 在RoboTwin套件上实现了37.8%的平均成功率，超越了之前最佳模型6.9%，尤其在“Click AlarmClock”任务中表现出色，展示了精确的双臂协调能力。
*   **LIBERO基准测试：** 取得了94.8%的竞争性结果，在空间、物体、目标和长任务等所有任务类别中均保持强大性能，尤其在长任务中表现出高鲁棒性（92.3%）。
*   **真实世界评估：** 在四项代表性机器人任务中，Evo-1实现了78%的平均成功率，显著优于所有基线方法（SmolVLA 50%，OpenVLA-OFT 55%，πo 73%）。
*   **推理效率：** Evo-1在RTX 4090d GPU上实现了16.4 Hz的最高推理频率和2.3 GB的低内存开销，证明了其在消费级GPU上进行实时部署的效率。
*   **泛化能力：** 在真实世界泛化实验中，Evo-1在未见过的干扰物、背景颜色变化、目标位置变化和目标高度变化等所有干扰设置下，均持续优于SmolVLA，展现出卓越的泛化能力。

这些结果表明，Evo-1在不依赖大规模机器人数据预训练的情况下，实现了最先进的性能，同时显著降低了计算成本和提高了部署效率，为VLA模型的设计和训练树立了新标准。

**4. 论文中提及的局限性：**
论文中未明确提及Evo-1的特定局限性。然而，从其设计和评估范围来看，可能存在的隐性局限包括：
*   **任务复杂性：** 尽管Evo-1在Meta-World、LIBERO和RoboTwin等基准测试中表现出色，但这些任务的复杂性可能仍低于某些高度开放式或需要复杂长期规划的真实世界场景。
*   **数据量：** 尽管Evo-1不依赖大规模机器人数据预训练，但其训练仍需要一定量的任务特定演示数据（例如，Meta-World和RoboTwin每个任务50个演示，真实世界任务每个任务100个演示）。对于某些极度稀缺数据的任务，其性能可能仍有待验证。
*   **硬件依赖：** 尽管Evo-1可以在消费级GPU上高效运行，但其性能和效率可能仍受限于特定硬件配置。

**5. 潜在的未来研究方向：**
*   **进一步优化轻量级架构：** 探索更先进的模型压缩技术或更高效的Transformer变体，以进一步减少参数量和计算需求，同时保持或提升性能。
*   **零样本或少样本学习：** 尽管Evo-1在泛化方面表现出色，但可以进一步研究如何使其在零样本或极少样本设置下执行新任务，从而减少对任务特定演示数据的依赖。
*   **更复杂的任务和环境：** 将Evo-1扩展到更复杂、更开放的真实世界场景，包括需要更高级推理、长期规划或与人类进行更复杂交互的任务。
*   **多模态输入扩展：** 除了视觉和语言，可以探索整合其他模态信息（如触觉、听觉）以增强机器人的感知和理解能力。
*   **强化学习集成：** 结合强化学习方法，使Evo-1能够通过试错学习和自我改进，进一步提升其在未知环境中的适应性和鲁棒性。
*   **可解释性和安全性：** 深入研究VLA模型的可解释性，理解其决策过程，并探索如何确保机器人在复杂任务中的安全性和可靠性。

---

**Key Findings:**

- In this work, we present Evo-1, a lightweight VLA model that reduces
computation and improves deployment efficiency, while maintaining strong
performance without pretraining on robot data.
- Evo-1 builds on a native
multimodal Vision-Language model (VLM), incorporating a novel cross-modulated
diffusion transformer along with an optimized integration module, together
forming an effective architecture.
- Notably, with only 0.77 billion parameters, Evo-1
achieves state-of-the-art results on the Meta-World and RoboTwin suite,
surpassing the previous best models by 12.4% and 6.9%, respectively, and also
attains a competitive result of 94.8% on LIBERO.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04555v1)
- [arXiv](https://arxiv.org/abs/2511.04555v1)

---

<a id='2511.04394v1'></a>
## [DORAEMON: A Unified Library for Visual Object Modeling and Representation Learning at Scale](https://arxiv.org/abs/2511.04394v1)

**Authors:** Ke Du, Yimin Peng, Chao Gao, Fan Zhou, Siqiao Xue

**Published:** 2025-11-06

**Categories:** cs.CV

**Abstract:**

DORAEMON is an open-source PyTorch library that unifies visual object
modeling and representation learning across diverse scales. A single
YAML-driven workflow covers classification, retrieval and metric learning; more
than 1000 pretrained backbones are exposed through a timm-compatible interface,
together with modular losses, augmentations and distributed-training utilities.
Reproducible recipes match or exceed reference results on ImageNet-1K,
MS-Celeb-1M and Stanford online products, while one-command export to ONNX or
HuggingFace bridges research and deployment. By consolidating datasets, models,
and training techniques into one platform, DORAEMON offers a scalable
foundation for rapid experimentation in visual recognition and representation
learning, enabling efficient transfer of research advances to real-world
applications. The repository is available at https://github.com/wuji3/DORAEMON.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Ke Du等人的论文“DORAEMON: A Unified Library for Visual Object Modeling and Representation Learning at Scale”的全面摘要。

---

### DORAEMON: 一个用于大规模视觉对象建模和表示学习的统一库

**摘要**

这篇论文介绍了DORAEMON，一个开源的PyTorch库，旨在统一大规模视觉对象建模和表示学习。该库通过提供一个统一、可扩展的框架，解决了当前计算机视觉领域中碎片化的代码库、任务特定管道和不一致的训练实践所带来的挑战。

**1. 解决的主要问题或研究问题**

当前大规模视觉对象建模面临的主要问题是：
* **碎片化的代码库和任务特定管道：** 研究人员通常需要手动整合不同的数据集、模型架构和损失函数，这使得复现和实验变得复杂且耗时。
* **不一致的训练实践：** 缺乏统一的框架使得跨任务比较不同方法变得困难。
* **研究与部署之间的摩擦：** 研究原型向生产级视觉系统转移时存在显著障碍。

DORAEMON旨在通过提供一个统一的平台，实现视觉识别和表示学习的快速实验，并促进研究成果向实际应用的有效转化。

**2. 关键创新或方法论贡献**

DORAEMON的关键创新和方法论贡献包括：

*   **统一的YAML驱动工作流：** DORAEMON提供了一个单一的YAML驱动工作流，涵盖了图像分类、检索和度量学习等多种任务，极大地简化了实验配置和管理。
*   **全面的模型支持：** 集成了超过1000个预训练骨干网络（通过timm兼容接口），为图像分类、人脸识别和检索等任务提供了即插即用的工作流，支持跨广泛骨干网络的有效基准测试。
*   **灵活的训练管道：** 提供了模块化的损失函数、数据增强（如MixUp、CutOut、Copy-Paste等）、现代优化算法（如SGD、Adam、SAM）、学习率调度器以及分布式训练工具，实现了对训练过程的精细控制。
*   **可解释性和部署工具包：** 内置Grad-CAM可视化功能，便于定性分析和调试；支持一键导出到ONNX或HuggingFace，无缝连接研究与部署。
*   **模块化设计：** 采用共享编码器骨干网络和任务特定输出头（prediction heads）的模块化设计，实现了表示学习与任务特定预测的解耦，支持统一训练、联合优化和跨任务的可扩展适应。

**3. 主要结果及其意义**

*   **可复现性与性能：** DORAEMON提供的可复现配方在ImageNet-1K、MS-Celeb-1M和Stanford在线产品等基准数据集上，其结果达到或超越了现有参考结果，证明了其方法的有效性和鲁棒性。
*   **效率与可扩展性：** 通过整合数据集、模型和训练技术到一个平台，DORAEMON为视觉识别和表示学习的快速实验提供了可扩展的基础，显著提高了研究成果向实际应用的转化效率。
*   **降低研究门槛：** 模块化的API、丰富的数据增强套件和内置的可视化工具降低了计算机视觉研究的入门门槛，同时保持了对大规模生产工作负载的高度可扩展性。

**4. 论文中提到的局限性**

论文中没有明确指出DORAEMON当前的局限性。然而，从未来研究方向的讨论中可以推断出，当前版本可能主要关注传统的视觉任务，并且在与大型语言模型（LLMs）的深度集成方面仍有发展空间。

**5. 潜在的未来研究方向**

论文提出了两个主要的未来研究方向：

*   **集成强大的智能体：** 将DORAEMON不仅仅作为一个训练工具包，而是作为一个能够使用工具的智能体，集成数据库、时间戳视觉分析和预测决策能力，以支持更复杂的视觉工作流。
*   **多模态LLMs的先进训练范式：** 扩展对多模态大型语言模型的支持，包括：
    *   **持续多模态预训练：** 逐步摄取新的文本-图像/视频数据。
    *   **轻量级基于提示的跨模态微调：** 减少开发终身和任务特定多模态LLMs的开销。

---

总而言之，DORAEMON是一个重要的贡献，它通过提供一个统一、灵活且高效的框架，极大地简化了大规模视觉对象建模和表示学习的复杂性。它不仅为研究人员提供了强大的工具，也为工业界提供了快速部署和评估视觉任务的解决方案，并为未来与大型语言模型和多模态系统的集成奠定了基础。

**Key Findings:**

- By consolidating datasets, models,
and training techniques into one platform, DORAEMON offers a scalable
foundation for rapid experimentation in visual recognition and representation
learning, enabling efficient transfer of research advances to real-world
applications.
- The repository is available at https://github.com/wuji3/DORAEMON.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.04394v1)
- [arXiv](https://arxiv.org/abs/2511.04394v1)

---

