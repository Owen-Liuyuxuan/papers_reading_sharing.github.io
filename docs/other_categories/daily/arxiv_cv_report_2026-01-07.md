time: 20260107

# Arxiv Computer Vision Papers - 2026-01-07

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提炼一份简明的 Arxiv 计算机视觉领域近期论文的执行摘要。

---

**Arxiv 计算机视觉领域论文执行摘要 (2026-01-06)**

**1. 主要主题与趋势：**

本期 Arxiv 论文集中体现了以下几个关键主题和趋势：

*   **多模态融合与生成：** 跨越文本、图像、音频、3D 等多种模态的理解和生成是核心焦点。模型正朝着更通用、更强大的多模态能力发展，能够处理和生成复杂的内容。
*   **基础模型与通用性：** 基础模型（Foundation Models）的构建和优化是另一大趋势，旨在通过自监督或自生成的方式提升模型的通用性和自我改进能力，减少对大规模标注数据的依赖。
*   **3D 内容生成与应用：** 3D 内容的生成、表示和应用（如数字孪生、虚拟生物设计）正变得更加高效和逼真，特别是结合了神经隐式场和高斯溅射等新技术。
*   **效率与加速：** 在保持高性能的同时，提升模型的效率和推理速度也是重要的研究方向，例如通过特定架构或与大型语言模型（LLM）的结合来实现。
*   **类比生物系统：** 有研究开始探索将生物视觉系统的学习机制应用于人工神经网络，以期获得更高效、更鲁棒的学习能力。

**2. 重点关注的创新性论文：**

*   **"Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training"** 极具创新性，它展示了在**无需训练**的情况下，仅通过设计和组合即可生成逼真的 3D 幻想生物，这可能颠覆传统的 3D 内容创作流程。
*   **"InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields"** 在深度估计领域取得了显著进展，通过神经隐式场实现了任意分辨率和精细的深度估计，为 3D 重建和场景理解提供了更强大的工具。
*   **"UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision"** 提出了一个**自我改进**的多模态模型框架，通过自生成监督信号来提升模型的性能，是迈向更自主、更高效基础模型的重要一步。

**3. 新兴研究方向与技术：**

*   **无监督/自监督的 3D 内容生成：** "Muses" 论文预示着未来在无需大量标注数据的情况下生成复杂 3D 内容的可能性。
*   **神经隐式场在 3D 任务中的深度应用：** "InfiniDepth" 和 "A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting" 都展示了神经隐式场和 3D 高斯溅射在精细化 3D 表示和应用中的强大潜力。
*   **LLM 驱动的生成模型加速：** "DiffBench Meets DiffAgent" 探索了利用 LLM 来加速扩散模型代码生成，这可能成为提升生成模型开发效率的新途径。
*   **类比生物视觉系统的学习机制：** "Transformers self-organize like newborn visual systems when trained in prenatal worlds" 引入了从生物学角度理解和设计神经网络的新视角。

**4. 建议阅读全文的论文：**

考虑到其创新性和对未来研究方向的潜在影响，以下论文值得深入阅读：

*   **"Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training"**: 如果您对 3D 内容生成、无监督学习或创意 AI 感兴趣，这篇论文提供了全新的思路。
*   **"InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields"**: 对于任何从事 3D 视觉、SLAM、机器人感知或需要高精度深度信息的领域的研究者，这篇论文的技术细节至关重要。
*   **"UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision"**: 如果您关注多模态基础模型、自监督学习或模型的可持续发展，这篇论文提出的框架值得深入研究。
*   **"Transformers self-organize like newborn visual systems when trained in prenatal worlds"**: 这篇论文提供了对 Transformer 模型学习机制的独特见解，可能对理解和设计更高效的神经网络架构有启发。

---

这份摘要旨在为您提供一个快速了解最新研究动态的概览。希望它能帮助您高效地把握计算机视觉领域的最新进展。

---

## Table of Contents

1. [Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training](#2601.03256v1)
2. [InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields](#2601.03252v1)
3. [A Versatile Multimodal Agent for Multimedia Content Generation](#2601.03250v1)
4. [LTX-2: Efficient Joint Audio-Visual Foundation Model](#2601.03233v1)
5. [A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting](#2601.03200v1)
6. [UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision](#2601.03193v1)
7. [Multi-Modal Data-Enhanced Foundation Models for Prediction and Control in Wireless Networks: A Survey](#2601.03181v1)
8. [DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](#2601.03178v1)
9. [Unified Thinker: A General Reasoning Modular Core for Image Generation](#2601.03127v1)
10. [Transformers self-organize like newborn visual systems when trained in prenatal worlds](#2601.03117v1)

---

## Papers

<a id='2601.03256v1'></a>
## [Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training](https://arxiv.org/abs/2601.03256v1)

**Authors:** Hexiao Lu, Xiaokun Sun, Zeyu Cai, Hao Guo, Ying Tai, Jian Yang, Zhenyu Zhang

**Published:** 2026-01-06

**Categories:** cs.CV

**Abstract:**

We present Muses, the first training-free method for fantastic 3D creature generation in a feed-forward paradigm. Previous methods, which rely on part-aware optimization, manual assembly, or 2D image generation, often produce unrealistic or incoherent 3D assets due to the challenges of intricate part-level manipulation and limited out-of-domain generation. In contrast, Muses leverages the 3D skeleton, a fundamental representation of biological forms, to explicitly and rationally compose diverse elements. This skeletal foundation formalizes 3D content creation as a structure-aware pipeline of design, composition, and generation. Muses begins by constructing a creatively composed 3D skeleton with coherent layout and scale through graph-constrained reasoning. This skeleton then guides a voxel-based assembly process within a structured latent space, integrating regions from different objects. Finally, image-guided appearance modeling under skeletal conditions is applied to generate a style-consistent and harmonious texture for the assembled shape. Extensive experiments establish Muses' state-of-the-art performance in terms of visual fidelity and alignment with textual descriptions, and potential on flexible 3D object editing. Project page: https://luhexiao.github.io/Muses.github.io/.

**Analysis:**

好的，这是一篇关于“Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training”论文的全面中文摘要，其中包含了您要求的各个部分：

**论文题目：** Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training (Muses：在无需训练的情况下设计、组合、生成不存在的奇幻 3D 生物)

**作者：** Hexiao Lu, Xiaokun Sun, Zeyu Cai, Hao Guo, Ying Tai, Jian Yang, Zhenyu Zhang

---

### **论文摘要**

**1. 研究问题/研究目标：**

该论文旨在解决当前 3D 内容生成领域，特别是生成不存在的、具有高度创造性的奇幻生物（如融合了不同动物、机械、神话生物特征的生物）所面临的挑战。现有方法，如依赖部件优化、手动组装或 2D 图像生成的方法，往往难以生成逼真、结构连贯且符合复杂创意描述的 3D 模型。研究的核心问题是如何在**无需训练**的情况下，高效且可控地生成这些高度复杂的、非领域内的 3D 生物。

**2. 关键创新点/方法论贡献：**

Muses 提出了一个**首个无需训练的、前馈范式的奇幻 3D 生物生成框架**。其核心创新在于：

*   **以 3D 骨架为基础的生成范式：** Muses 将 3D 骨架视为生物形态的基本表示，将其作为设计、组合和生成过程的根本依据。这克服了传统方法在精细部件操作和跨领域生成方面的局限性。
*   **设计-组合-生成（Design-Compose-Generate）流程：**
    *   **骨架引导的概念设计（Skeleton-guided Concept Design）：** 利用图结构和大型语言模型（LLM）进行**图约束推理**，以生成具有合理布局和尺度的创意 3D 骨架。该方法能够将文本提示解析为概念，并生成相应的 3D 资产和骨架。
    *   **结构化潜在空间（SLAT）的内容组合（SLAT-based Content Composition）：** 利用骨架引导的**蒙皮权重（skinning weights）**，将骨架的各个部分映射到结构化潜在空间（SLAT）中的对应区域。然后，在 SLAT 中进行**基于体素的几何和纹理插值**，以融合不同对象的区域，确保组合后的形状在结构上与骨架对齐，并实现平滑连贯的几何和纹理。
    *   **风格一致的纹理生成（Style-consistent Texture Generation）：** 引入**几何不变的纹理编辑**方法，利用编辑后的图像和粗糙几何体，生成风格一致且和谐的纹理，从而提升最终 3D 生物的视觉保真度和多样性。
*   **训练无关性：** 整个流程是**训练无关的（training-free）**，这意味着它不需要在大量的 3D 数据集上进行预训练，从而能够处理更广泛的、非领域内的创意概念。

**3. 主要结果及其意义：**

*   **生成高质量、高保真的奇幻 3D 生物：** Muses 能够生成具有复杂结构、逼真几何和和谐纹理的奇幻 3D 生物，并且这些生物与文本描述高度一致。
*   **优于现有方法的性能：** 通过与 DreamBeast、GaussianDreamer、UNO+Trellis、Trellis-Text-to-3D、OmniPart 等多种先进方法的比较，Muses 在视觉保真度、文本对齐度以及处理复杂组合描述的能力上均取得了显著优势。
*   **灵活的 3D 对象编辑能力：** 该框架不仅能生成新生物，还展示了在几何和纹理方面的灵活编辑潜力，能够实现部件级别的 3D 编辑和风格迁移。
*   **意义：** Muses 的工作为生成高度创意和非领域内的 3D 内容提供了一种新颖且有效的解决方案，尤其是在无需大量 3D 训练数据的情况下。它展示了 3D 骨架作为一种基础表示在复杂 3D 内容创作中的重要作用。

**4. 论文中提到的局限性：**

*   **依赖于底层 3D 生成模型：** Muses 的生成质量在一定程度上依赖于其底层 3D 生成模型（如 Trellis）的能力。如果底层模型无法生成逼真的 3D 模型（例如，无法生成逼真的孔雀），那么 Muses 也难以从中提取有意义的骨架部分。
*   **骨架初始化问题：** 骨架的初始生成（例如使用 Puppeteer）有时可能不理想，导致无法执行设计阶段。
*   **无法处理非骨架化对象：** 该方法适用于具有骨架结构的生物，但无法处理无法用骨架形式化的抽象对象。

**5. 潜在的未来研究方向：**

*   **扩展到更通用的 3D 对象编辑工具：** 将 Muses 进一步发展为一个更灵活的 3D 对象编辑工具，以支持游戏、虚拟现实和动画等领域的交互式应用。
*   **改进底层 3D 生成和骨架建模方法：** 结合更强大的 3D 生成模型和骨架建模技术，以克服当前方法在处理复杂或不常见对象时的局限性。
*   **处理更广泛的非骨架化对象：** 探索将 Muses 的设计-组合-生成范式扩展到非骨架化对象的生成。

---

总而言之，Muses 论文提出了一种创新的、无需训练的 3D 生物生成框架，通过巧妙地利用 3D 骨架作为核心表示，实现了对复杂奇幻生物的有效设计、组合和生成，并在多个方面超越了现有技术，为未来 3D 内容创作开辟了新的可能性。

**Key Findings:**

- We present Muses, the first training-free method for fantastic 3D creature generation in a feed-forward paradigm.
- Extensive experiments establish Muses' state-of-the-art performance in terms of visual fidelity and alignment with textual descriptions, and potential on flexible 3D object editing.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03256v1)
- [arXiv](https://arxiv.org/abs/2601.03256v1)

---

<a id='2601.03252v1'></a>
## [InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields](https://arxiv.org/abs/2601.03252v1)

**Authors:** Hao Yu, Haotong Lin, Jiawei Wang, Jiaxin Li, Yida Wang, Xueyang Zhang, Yue Wang, Xiaowei Zhou, Ruizhen Hu, Sida Peng

**Published:** 2026-01-06

**Categories:** cs.CV

**Abstract:**

Existing depth estimation methods are fundamentally limited to predicting depth on discrete image grids. Such representations restrict their scalability to arbitrary output resolutions and hinder the geometric detail recovery. This paper introduces InfiniDepth, which represents depth as neural implicit fields. Through a simple yet effective local implicit decoder, we can query depth at continuous 2D coordinates, enabling arbitrary-resolution and fine-grained depth estimation. To better assess our method's capabilities, we curate a high-quality 4K synthetic benchmark from five different games, spanning diverse scenes with rich geometric and appearance details. Extensive experiments demonstrate that InfiniDepth achieves state-of-the-art performance on both synthetic and real-world benchmarks across relative and metric depth estimation tasks, particularly excelling in fine-detail regions. It also benefits the task of novel view synthesis under large viewpoint shifts, producing high-quality results with fewer holes and artifacts.

**Analysis:**

好的，这是对论文“InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields”的全面中文摘要：

**论文题目：** InfiniDepth: 基于神经隐式场实现任意分辨率和精细化深度估计

**作者：** Hao Yu, Haotong Lin, Jiawei Wang, Jiaxin Li, Yida Wang, Xueyang Zhang, Yue Wang, Xiaowei Zhou, Ruizhen Hu, Sida Peng

**摘要：**

**1. 研究问题/核心挑战：**
现有深度估计方法主要局限于在离散的图像网格上预测深度。这种表示方式限制了其扩展到任意输出分辨率的能力，并且难以恢复精细的几何细节。这导致在处理高分辨率场景或需要精确几何信息时效果不佳。

**2. 主要创新点/方法贡献：**
*   **神经隐式场深度表示：** 论文的核心贡献是提出了一种新的深度表示方法——**神经隐式场（Neural Implicit Fields）**。与传统的离散网格表示不同，InfiniDepth将深度建模为一个连续的函数，允许在任意2D坐标点上查询深度值。
*   **局部隐式解码器：** 引入了一个简单而有效的**局部隐式解码器**，该解码器能够聚合来自多尺度特征的金字塔的特征，并将其输入到一个轻量级的MLP中，从而在连续的2D坐标上预测深度。这种连续和局部化的预测范式使得模型能够自然地生成任意分辨率的深度图，并捕捉精细的几何细节。
*   **无限深度查询策略（Infinite Depth Query）：** 为了解决传统逐像素深度预测在视角变化较大时导致的3D点云密度不均问题，论文设计了一种**深度查询策略**。该策略通过自适应地分配子像素查询预算，使得可见表面上的3D点云分布更加均匀，从而显著提高了在大视角偏移下的新视图合成（NVS）质量，减少了孔洞和伪影。
*   **Synth4K数据集：** 为了更好地评估模型在任意分辨率和精细化细节方面的能力，作者**创建了一个高质量的4K合成数据集Synth4K**。该数据集包含来自五个不同游戏的场景，具有丰富的几何和外观细节，并提供了4K分辨率的地面真实深度图。

**3. 主要结果与意义：**
*   **SOTA性能：** 实验结果表明，InfiniDepth在Synth4K和真实世界数据集上的相对和度量深度估计任务中均取得了**最先进（State-of-the-Art）的性能**，尤其在精细细节区域表现出色。
*   **任意分辨率和精细化：** InfiniDepth能够生成任意分辨率的深度图，并且在恢复精细几何细节方面远超现有方法，这在图1(b)所示的精细化点云中得到了体现。
*   **提升新视图合成：** 结合无限深度查询策略，InfiniDepth在新视图合成任务中表现出显著优势，能够生成更完整、更稳定的新视图，尤其是在大视角偏移的情况下（如图1(c)和图8所示）。
*   **数据集贡献：** Synth4K数据集为高分辨率和精细化深度估计的研究提供了一个更具挑战性和更可靠的评估基准。

**4. 提及的局限性：**
*   **时间一致性：** 该方法主要针对单目深度估计，并且仅在单目深度数据上进行训练。当应用于视频时，它**不显式地强制执行时间一致性**，可能导致帧间出现闪烁现象。

**5. 潜在的未来研究方向：**
*   **多视角和时间一致性：** 将深度表示扩展到**多视角设置**，以提高时间一致性并减少视频中的闪烁。
*   **3D感知和重建：** 将InfiniDepth集成到更广泛的**3D感知和重建流水线**中。
*   **更广泛的应用：** 探索其在**更广泛的3D感知和重建应用**中的潜力。

**总结：**
InfiniDepth通过引入神经隐式场作为深度表示，成功解决了现有深度估计方法在分辨率扩展性和细节恢复方面的根本性限制。其提出的局部隐式解码器和无限深度查询策略，不仅在深度估计任务上取得了显著的性能提升，尤其是在精细细节方面，还极大地改善了在大视角偏移下的新视图合成质量。Synth4K数据集的创建也为该领域的研究提供了重要资源。该工作为实现更高精度、更灵活的深度估计和3D场景理解开辟了新的途径。

**Key Findings:**

- To better assess our method's capabilities, we curate a high-quality 4K synthetic benchmark from five different games, spanning diverse scenes with rich geometric and appearance details.
- Extensive experiments demonstrate that InfiniDepth achieves state-of-the-art performance on both synthetic and real-world benchmarks across relative and metric depth estimation tasks, particularly excelling in fine-detail regions.
- It also benefits the task of novel view synthesis under large viewpoint shifts, producing high-quality results with fewer holes and artifacts.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03252v1)
- [arXiv](https://arxiv.org/abs/2601.03252v1)

---

<a id='2601.03250v1'></a>
## [A Versatile Multimodal Agent for Multimedia Content Generation](https://arxiv.org/abs/2601.03250v1)

**Authors:** Daoan Zhang, Wenlin Yao, Xiaoyang Wang, Yebowen Hu, Jiebo Luo, Dong Yu

**Published:** 2026-01-06

**Categories:** cs.CV

**Abstract:**

With the advancement of AIGC (AI-generated content) technologies, an increasing number of generative models are revolutionizing fields such as video editing, music generation, and even film production. However, due to the limitations of current AIGC models, most models can only serve as individual components within specific application scenarios and are not capable of completing tasks end-to-end in real-world applications. In real-world applications, editing experts often work with a wide variety of images and video inputs, producing multimodal outputs -- a video typically includes audio, text, and other elements. This level of integration across multiple modalities is something current models are unable to achieve effectively. However, the rise of agent-based systems has made it possible to use AI tools to tackle complex content generation tasks. To deal with the complex scenarios, in this paper, we propose a MultiMedia-Agent designed to automate complex content creation. Our agent system includes a data generation pipeline, a tool library for content creation, and a set of metrics for evaluating preference alignment. Notably, we introduce the skill acquisition theory to model the training data curation and agent training. We designed a two-stage correlation strategy for plan optimization, including self-correlation and model preference correlation. Additionally, we utilized the generated plans to train the MultiMedia-Agent via a three stage approach including base/success plan finetune and preference optimization. The comparison results demonstrate that the our approaches are effective and the MultiMedia-Agent can generate better multimedia content compared to novel models.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将为您提供这篇论文的全面中文摘要，重点突出其在多模态内容生成领域的贡献。

**论文题目：** A Versatile Multimodal Agent for Multimedia Content Generation

**作者：** Daoan Zhang, Wenlin Yao, Xiaoyang Wang, Yebowen Hu, Jiebo Luo, Dong Yu

---

**论文摘要**

**1. 研究问题/核心挑战：**

当前AI生成内容（AIGC）技术虽然在视频编辑、音乐生成等领域取得了显著进展，但大多数模型仍局限于单一模态或作为特定应用场景的组件，无法有效处理现实世界中复杂、端到端的多模态内容生成任务。现实世界的媒体创作往往涉及多种模态（如图像、视频、音频、文本）的整合，而现有模型在有效整合这些模态以生成连贯、符合用户偏好的多模态内容方面存在不足。

**2. 主要创新点/方法论贡献：**

*   **MultiMedia-Agent 框架：** 论文提出了一种名为 MultiMedia-Agent 的系统，旨在自动化复杂的、端到端的多模态内容创作流程。
*   **技能获取理论驱动的训练：** 核心创新在于将“技能获取理论”（Skill Acquisition Theory）应用于多模态内容生成代理的训练。该理论将学习过程分为认知阶段、联想阶段和自主阶段，论文据此设计了一个三阶段的训练流程，使代理能够从零开始学习复杂计划的生成和执行。
*   **两阶段层级化计划策定策略：**
    *   **数据生成与工具库：** 构建了一个包含18种真实世界多模态任务的数据集，并设计了一个包含理解工具、生成/编辑工具和辅助工具的全面工具库。
    *   **自相关与模型偏好相关：** 引入了一个两阶段的计划优化策略。首先，利用GPT-4o进行“自相关”（self-correlation），通过自我反思和纠错来优化初始计划。接着，利用模型偏好相关（model preference correlation），通过基于人类偏好的评估指标来进一步精炼计划，确保生成的内容不仅功能上可行，而且在美学和情感上符合用户期望。
*   **多模态内容评估指标：** 设计了一系列针对图像、视频、音频和文本输出的评估指标，并引入了音频-视频对齐度量，以全面评估生成内容的质量和用户偏好对齐度。

**3. 主要研究结果与意义：**

*   **有效性验证：** 通过实验表明，MultiMedia-Agent 在多模态内容生成任务上表现出有效性，能够生成比现有模型更优质的多模态内容。
*   **技能习得的体现：** 三阶段的训练方法成功地使代理能够逐步掌握复杂计划的生成和执行，并在自主阶段实现了显著的性能提升，尤其是在满足人类偏好方面。
*   **对比优势：** 与现有的一些多模态工具代理和内容生成代理相比，MultiMedia-Agent 在多模态理解、计划能力和偏好对齐方面展现出更强的综合能力（如Table 1所示）。
*   **可视化结果：** 实验可视化结果（Figure 4）清晰地展示了代理在不同训练阶段生成内容的质量提升，从最初缺乏音频和特效，到后期能够整合字幕、特效和符合场景的音频，体现了方法的有效性。

**4. 提及的局限性：**

*   **工具选择方式：** 目前的工具选择主要依赖于提示（prompt-based），对于海量工具库而言，效率和准确性有待提升。
*   **单代理局限：** 在处理极其复杂的任务时，单代理系统可能不如多代理系统高效。

**5. 潜在的未来研究方向：**

*   **检索增强生成（RAG）用于工具选择：** 探索使用RAG技术来优化工具的选择过程，以应对大规模工具库。
*   **多代理系统：** 研究和开发多代理系统，以更有效地解决复杂的多模态内容生成任务。

**对计算机视觉领域的贡献：**

这篇论文为计算机视觉领域在**多模态内容生成**方面提供了重要的理论和实践贡献。它不仅提出了一种能够处理跨多种模态的复杂内容创作的通用代理框架，更重要的是，它成功地将**技能获取理论**这一认知科学的原理引入到AI代理的学习过程中，实现了代理从基础操作到复杂、偏好对齐的自主学习。这为构建更智能、更人性化的多模态内容生成系统开辟了新的途径，尤其是在**视频生成、图像编辑与音频整合**等视觉相关任务中，其方法论具有重要的参考价值和应用前景。论文提出的**层级化计划策定策略**和**偏好对齐机制**，也为提升生成内容的质量和用户满意度提供了有效的解决方案。

**Key Findings:**

- To deal with the complex scenarios, in this paper, we propose a MultiMedia-Agent designed to automate complex content creation.
- Notably, we introduce the skill acquisition theory to model the training data curation and agent training.
- The comparison results demonstrate that the our approaches are effective and the MultiMedia-Agent can generate better multimedia content compared to novel models.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03250v1)
- [arXiv](https://arxiv.org/abs/2601.03250v1)

---

<a id='2601.03233v1'></a>
## [LTX-2: Efficient Joint Audio-Visual Foundation Model](https://arxiv.org/abs/2601.03233v1)

**Authors:** Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, Zeev Farbman

**Published:** 2026-01-06

**Categories:** cs.CV

**Abstract:**

Recent text-to-video diffusion models can generate compelling video sequences, yet they remain silent -- missing the semantic, emotional, and atmospheric cues that audio provides. We introduce LTX-2, an open-source foundational model capable of generating high-quality, temporally synchronized audiovisual content in a unified manner. LTX-2 consists of an asymmetric dual-stream transformer with a 14B-parameter video stream and a 5B-parameter audio stream, coupled through bidirectional audio-video cross-attention layers with temporal positional embeddings and cross-modality AdaLN for shared timestep conditioning. This architecture enables efficient training and inference of a unified audiovisual model while allocating more capacity for video generation than audio generation. We employ a multilingual text encoder for broader prompt understanding and introduce a modality-aware classifier-free guidance (modality-CFG) mechanism for improved audiovisual alignment and controllability. Beyond generating speech, LTX-2 produces rich, coherent audio tracks that follow the characters, environment, style, and emotion of each scene -- complete with natural background and foley elements. In our evaluations, the model achieves state-of-the-art audiovisual quality and prompt adherence among open-source systems, while delivering results comparable to proprietary models at a fraction of their computational cost and inference time. All model weights and code are publicly released.

**Analysis:**

好的，这是对论文“LTX-2: Efficient Joint Audio-Visual Foundation Model”的全面中文摘要，重点关注其在计算机视觉和机器学习领域的重要性：

**论文题目：** LTX-2: Efficient Joint Audio-Visual Foundation Model

**作者：** Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, Zeev Farbman

---

**摘要：**

**1. 研究问题/核心挑战：**

当前文本到视频（Text-to-Video, T2V）扩散模型虽然能够生成引人入胜的视频序列，但普遍存在“沉默”的问题，即无法捕捉音频所提供的语义、情感和氛围线索。这使得生成的视频内容在沉浸感和实用性上大打折扣。同时，现有的文本到音频（Text-to-Audio, T2A）模型大多是任务特定的，缺乏统一、全面的音频生成能力。将两者结合的文本到音频+视频（Text-to-Audio+Video, T2AV）生成模型的研究仍处于早期阶段，并且现有方法往往采用解耦的流水线，未能充分建模音视频模态间的深层联合分布和双向依赖关系。

**2. 主要创新点/方法贡献：**

LTX-2 提出了一种新颖的、开源的、高效的联合音频-视频基础模型，旨在生成高质量、时间同步的视听内容。其核心创新点包括：

*   **非对称双流Transformer架构：** 模型采用一个包含140亿参数的视频流和一个50亿参数的音频流的非对称双流Transformer架构。这种设计将更多计算资源分配给视觉生成，同时保持音频流的高效性，以适应两种模态信息密度的差异。
*   **跨模态AdaLN和时间位置嵌入：** 通过双向音频-视频交叉注意力层、时间位置嵌入（视频使用3D RoPE，音频使用1D RoPE）以及跨模态AdaLN（Adaptive Layer Normalization）进行共享时间步长条件化，实现了紧密的时间对齐和模态间的有效信息交换。
*   **多语言文本编码器与“思考令牌”：** 采用一个强大的多语言文本编码器（Gemma3-12B）和一种改进的文本处理管道，引入了“思考令牌”（Thinking Tokens），以增强对复杂提示的理解能力，提升语音生成的韵律、口音和情感准确性。
*   **模态感知分类器自由引导（Modality-Aware CFG）：** 提出了一种新颖的Bimodal CFG机制，允许独立控制文本和跨模态引导的权重，从而显著改善了视听对齐和可控性。
*   **高效的音频VAE和潜在空间表示：** 设计了一个紧凑的音频VAE，生成高保真度的1D潜在空间，并支持立体声输入，优化了音频生成效率。
*   **多尺度、多切片推理：** 采用一种高效的推理策略，能够生成全高清（1080p）视听内容，同时显著降低了内存开销。

**3. 主要结果与意义：**

*   **状态级视听质量：** LTX-2 在评估中展现了最先进的视听质量和提示遵循度，优于现有的开源系统。
*   **与专有模型媲美：** 其生成结果在人类偏好研究中与领先的专有模型（如Veo 3和Sora 2）相当。
*   **极高的效率：** LTX-2 在计算成本和推理时间上远低于同类模型，例如比Wan 2.2-14B快约18倍，这使其成为一个非常实用的基础模型。
*   **开源与可访问性：** 模型权重和代码的公开，为社区提供了强大的T2AV生成基础，极大地推动了该领域的研究和应用。
*   **超越现有时间范围：** LTX-2 能够生成长达20秒的连续视频和同步立体声，超过了许多现有模型。

**4. 提及的局限性：**

*   **语言差异：** 模型在不同语言上的表现存在差异，对于训练数据中代表性不足的语言或方言，可能导致语音合成或视听对齐不够准确。
*   **多说话人场景：** 在多说话人场景下，模型可能无法始终准确地将语音内容分配给正确的角色，有时会混淆说话人。
*   **时间漂移：** 生成超过20秒的连续视听序列可能导致时间漂移、同步性下降或场景多样性减少。
*   **缺乏显式推理能力：** LTX-2 是一个生成模型，不具备显式的推理或世界建模能力，复杂的叙事连贯性、事实准确性或情境理解需要依赖外部大型语言模型。

**5. 潜在的未来研究方向：**

*   **偏见缓解与可解释性：** 探索方法来减轻模型中存在的偏见，并提高合成内容的真实性验证能力。
*   **跨语言和低资源语言支持：** 进一步提升模型在不同语言上的表现，特别是针对低资源语言。
*   **更长序列生成：** 克服长序列生成中的时间漂移问题，实现更长的叙事性内容生成。
*   **集成推理能力：** 将LTX-2与外部推理系统结合，以生成更具逻辑性和事实准确性的内容。
*   **更精细的控制：** 开发更精细的控制机制，允许用户在生成过程中对视听内容的各个方面进行更细致的调整。

**在计算机视觉和机器学习领域的重要性：**

LTX-2 的发布标志着文本到音频+视频（T2AV）生成领域的一个重要里程碑。它不仅提供了一个**高效、高质量且开源**的基础模型，解决了现有T2V模型缺乏音频的痛点，而且通过其创新的**非对称双流架构、跨模态交互机制和先进的文本理解能力**，显著提升了视听内容的同步性、连贯性和表现力。其**卓越的推理效率**使得大规模、高分辨率的视听内容生成成为可能，为内容创作、教育、辅助技术等领域开辟了新的可能性。LTX-2 的开源性质将极大地促进该领域的研究和应用，并为未来更复杂的**多模态生成模型**奠定坚实的基础。

**Key Findings:**

- We introduce LTX-2, an open-source foundational model capable of generating high-quality, temporally synchronized audiovisual content in a unified manner.
- In our evaluations, the model achieves state-of-the-art audiovisual quality and prompt adherence among open-source systems, while delivering results comparable to proprietary models at a fraction of their computational cost and inference time.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03233v1)
- [arXiv](https://arxiv.org/abs/2601.03233v1)

---

<a id='2601.03200v1'></a>
## [A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting](https://arxiv.org/abs/2601.03200v1)

**Authors:** Ziyang Sun, Lingfan Bao, Tianhu Peng, Jingcheng Sun, Chengxu Zhou

**Published:** 2026-01-06

**Categories:** cs.RO

**Abstract:**

Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer. However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry. We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs. Our system employs 3D Gaussian Splatting (3DGS) for fast, photorealistic reconstruction as a unified scene representation. We enhance 3DGS with visibility-aware semantic fusion for accurate 3D labelling and introduce an efficient, filter-based geometry conversion method to produce collision-ready models seamlessly integrated with a Unity-ROS2-MoveIt physics engine. In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials. These results demonstrate that 3DGS-based digital twins, enriched with semantic and geometric consistency, offer a fast, reliable, and scalable path from perception to manipulation in unstructured environments.

**Analysis:**

好的，这是一篇关于使用3D高斯溅射（3D Gaussian Splatting, 3DGS）构建高保真数字孪生的论文的全面中文摘要，重点关注其在机器人操作中的应用。

**论文题目：** A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting
**作者：** Ziyang Sun, Lingfan Bao, Tianhu Peng, Jingcheng Sun, Chengxu Zhou

---

**摘要**

**1. 研究问题/核心挑战：**
机器人操作的自主性发展需要高保真、可交互的数字孪生，以实现闭环运动规划和可靠的现实世界执行，这对于推进“仿真到现实”（sim-to-real）的迁移至关重要。然而，现有方法在重建速度慢、视觉保真度有限以及将照片级真实感模型转换为可用于规划的碰撞几何体方面存在显著瓶颈。

**2. 主要创新点/方法贡献：**
该论文提出了一种实用的端到端框架，能够从稀疏的RGB视图在几分钟内构建高质量的数字孪生。其核心贡献包括：

*   **基于3DGS的高保真、快速重建：** 利用3D高斯溅射（3DGS）作为统一的场景表示，实现了快速、照片级真实感的场景重建，解决了传统方法（如NeRFs）计算成本高、速度慢的问题。
*   **可见光感知语义融合（Visibility-Aware Semantic Fusion）：** 引入了一种视图依赖的语义聚合方法，结合遮挡感知置信度加权策略，将2D分割线索提升到3D空间，确保了准确的3D标注，即使在遮挡情况下也能保持语义一致性。
*   **高效的碰撞几何体生成：** 开发了一种基于过滤器的几何体转换方法，将原始的3DGS点云转换为精确、可用于规划的碰撞网格（collision meshes），无缝集成到Unity-ROS2-MoveIt物理引擎中。该过程包括多阶段的去噪和网格化处理，以去除伪影并生成水密网格。
*   **统一的“现实-仿真-现实”闭环流程：** 构建了一个完整的闭环流程，将现实世界的场景捕获、数字孪生生成、仿真中的运动规划与验证，以及最终在真实机器人上的执行整合在一起。

**3. 主要结果与意义：**
*   **重建效率与质量：** 该系统平均重建时间不到4分钟（229秒），比NeRFs方法快5倍，同时实现了更高的视觉保真度（PSNR 37.03 dB，SSIM 0.9821），优于NeRFs方法。
*   **语义理解：** 实现了0.87的2D分割mIoU和0.93的3D投影一致性，证明了其多视图融合方法能有效桥接2D感知和3D几何重建。
*   **几何体质量：** 通过消融实验证明，多阶段的去噪和网格化处理（启发式过滤+DBSCAN聚类）能够显著提高几何保真度，F1-Score达到近乎完美的0.9989，为机器人操作提供了可靠的碰撞几何体。
*   **现实世界验证：** 在Franka Emika机器人上进行的长期重排任务中，仿真验证成功率为100%，真实世界执行成功率为90%（9/10次），且成功执行的轨迹零碰撞，平均放置误差为0.83厘米。
*   **意义：** 该框架有效地缩小了“仿真到现实”的差距，为在非结构化环境中实现可靠的机器人操作提供了快速、可靠且可扩展的路径。它将神经渲染与传统运动规划框架的关键环节——可用于规划的碰撞几何体——有效结合。

**4. 提及的局限性：**
*   **静态场景假设：** 当前框架主要针对静态场景，并执行一次性重建。
*   **仅模型几何和外观：** 系统目前仅建模几何和外观，未包含物理属性的估计。
*   **预定义抓取：** 当前工作侧重于给定预定义抓取的运动规划，而非直接在高斯表示上进行抓取规划。

**5. 未来研究方向：**
*   **动态场景处理：** 集成动态3DGS变体或连续更新机制，使数字孪生能适应不断变化的环境。
*   **物理属性估计：** 整合在线物理属性估计方法，以支持更复杂的接触式交互。
*   **集成抓取规划：** 开发直接在3DGS表示上操作的鲁棒抓取规划模块。
*   **作为学习策略的赋能者：** 将数字孪生用作安全验证平台，或作为数据生成引擎，自主创建训练数据集，加速机器人学习模型（如视觉-语言-动作模型和强化学习）的开发。

---

**总结：**

这篇论文提出了一种创新的方法，利用3D高斯溅射（3DGS）技术，在极短的时间内从稀疏的RGB图像构建出高保真、语义丰富的数字孪生。该框架通过引入可见光感知语义融合和高效的几何体清理及网格化流程，成功解决了现有方法在重建速度、视觉保真度和可规划性方面的不足。实验结果表明，该方法能够生成高质量的碰撞几何体，并成功应用于真实的机器人操作任务，显著推进了“仿真到现实”的迁移能力。该工作为未来在复杂非结构化环境中实现更自主、更可靠的机器人操作奠定了坚实基础。

**Key Findings:**

- We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs.
- In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03200v1)
- [arXiv](https://arxiv.org/abs/2601.03200v1)

---

<a id='2601.03193v1'></a>
## [UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision](https://arxiv.org/abs/2601.03193v1)

**Authors:** Ruiyan Han, Zhen Fang, XinYu Sun, Yuchen Ma, Ziheng Wang, Yu Zeng, Zehui Chen, Lin Chen, Wenxuan Huang, Wei-Jie Xu, Yi Cao, Feng Zhao

**Published:** 2026-01-06

**Categories:** cs.CV, cs.AI

**Abstract:**

While Unified Multimodal Models (UMMs) have achieved remarkable success in cross-modal comprehension, a significant gap persists in their ability to leverage such internal knowledge for high-quality generation. We formalize this discrepancy as Conduction Aphasia, a phenomenon where models accurately interpret multimodal inputs but struggle to translate that understanding into faithful and controllable synthesis. To address this, we propose UniCorn, a simple yet elegant self-improvement framework that eliminates the need for external data or teacher supervision. By partitioning a single UMM into three collaborative roles: Proposer, Solver, and Judge, UniCorn generates high-quality interactions via self-play and employs cognitive pattern reconstruction to distill latent understanding into explicit generative signals. To validate the restoration of multimodal coherence, we introduce UniCycle, a cycle-consistency benchmark based on a Text to Image to Text reconstruction loop. Extensive experiments demonstrate that UniCorn achieves comprehensive and substantial improvements over the base model across six general image generation benchmarks. Notably, it achieves SOTA performance on TIIF(73.8), DPG(86.8), CompBench(88.5), and UniCycle while further delivering substantial gains of +5.0 on WISE and +6.5 on OneIG. These results highlight that our method significantly enhances T2I generation while maintaining robust comprehension, demonstrating the scalability of fully self-supervised refinement for unified multimodal intelligence.

**Analysis:**

作为一名计算机视觉和机器学习领域的专家，我对这篇题为“UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision”的论文进行了分析。以下是我的详细解读：

**1. 论文的主要贡献（2-3句话的简洁总结）**

这篇论文提出了一种名为 UniCorn 的自监督学习框架，旨在解决统一多模态模型（UMMs）在理解输入后难以生成高质量、可控输出的问题，即“传导失语症”。UniCorn 通过将模型分解为“提议者”、“解决者”和“评判者”三个角色，利用模型自身的交互和认知模式重构来实现自我改进，无需外部数据或教师监督，显著提升了文本到图像生成能力。

**2. 关键创新或方法论**

*   **“传导失语症”（Conduction Aphasia）的定义与解决：** 论文首次提出了“传导失语症”这一概念，形象地描述了 UMMs 在跨模态理解和生成之间存在的鸿沟。UniCorn 的核心在于解决这一问题。
*   **自生成监督（Self-Generated Supervision）：** 这是 UniCorn 最核心的创新点。它完全摆脱了对外部数据集或人工标注的依赖，通过模型自身的“自我玩耍”（self-play）和内部机制来产生训练信号。
*   **三角色协同框架（Proposer, Solver, Judge）：**
    *   **Proposer (提议者):** 负责提出生成任务或生成初始内容。
    *   **Solver (解决者):** 负责根据提议者的输入进行多模态生成（例如，根据文本生成图像）。
    *   **Judge (评判者):** 负责评估 Solver 的生成结果是否符合 Proposer 的意图或多模态的连贯性。这个评判者可以看作是模型内部的一个“鉴别器”或“评估器”。
*   **认知模式重构（Cognitive Pattern Reconstruction）：** 通过这种机制，模型将潜在的、隐式的多模态理解转化为显式的、可用于指导生成的信号。这可以理解为模型在学习如何“思考”并将其“思考过程”转化为生成指令。
*   **UniCycle 基准测试：** 为了量化和验证多模态连贯性的恢复，论文引入了一个新的循环一致性（cycle-consistency）基准测试。它通过“文本 -> 图像 -> 文本”的重建循环来评估模型的性能，这是一种非常直观且有力的评估方法。

**3. 对该领域的潜在影响**

*   **推动 UMMs 的生成能力：** UniCorn 的成功将极大地推动 UMMs 在生成任务上的表现，使其不再仅仅是强大的理解工具，更能成为高质量的内容创作者。
*   **降低数据依赖，加速模型发展：** 自监督学习是当前机器学习领域的重要趋势。UniCorn 的方法论如果被广泛接受和应用，将显著降低开发和训练 UMMs 的成本，加速相关领域的研究和应用落地。
*   **提升模型的可控性和忠实度：** 通过“传导失语症”的解决，模型生成的输出将更符合用户的意图，更忠实于输入信息，从而提高用户体验和模型的可信度。
*   **为其他多模态任务提供通用框架：** UniCorn 的自改进框架具有普适性，理论上可以应用于其他需要跨模态理解和生成相结合的任务，如视频生成、音频合成等。

**4. 可能受益于该研究的相关领域或应用**

*   **文本到图像生成（Text-to-Image Generation）：** 这是论文直接验证并取得显著成果的领域，如 DALL-E, Stable Diffusion 等模型的进一步优化。
*   **多模态对话系统：** 能够更好地理解用户意图并生成更具信息量和连贯性的回复。
*   **视觉问答（Visual Question Answering）与视觉推理：** 提升模型在理解图像和文本关系后，进行更深层次推理和生成答案的能力。
*   **内容创作与辅助设计：** 自动生成插画、设计草图、营销素材等。
*   **虚拟现实/增强现实（VR/AR）：** 实时生成符合场景描述的虚拟内容。
*   **教育与培训：** 生成教学材料、模拟场景等。

**5. 从摘要中可以推断出的局限性**

*   **“传导失语症”的普遍性与定义：** 虽然论文提出了“传导失语症”的概念，但其在所有 UMMs 中的普遍程度以及其具体表现形式可能需要更深入的研究来界定。
*   **三角色协同的复杂性与计算成本：** 将一个 UMM 分解为三个协同角色，虽然是自监督，但其训练过程可能比传统的监督学习模型更复杂，计算资源需求也可能更高。
*   **“认知模式重构”的内在机制：** 摘要中对“认知模式重构”的描述较为抽象，其具体的实现细节和理论基础可能需要进一步的论文内容来阐述。
*   **UniCycle 基准的局限性：** UniCycle 作为一种新的基准，虽然有意义，但其是否能完全捕捉到所有多模态连贯性的细微差别，以及其与现有基准的互补性，仍需进一步验证。
*   **泛化能力：** 论文展示了在六个通用图像生成基准上的提升，但其在更广泛、更复杂的多模态任务上的泛化能力仍需观察。
*   **“简单而优雅”的实现：** 尽管摘要称其“简单而优雅”，但实际实现过程中，如何有效地协调三个角色，如何设计有效的评判机制，可能仍然存在技术挑战。

**总结来说，这篇论文的亮点在于其创新的自监督学习框架 UniCorn，通过“传导失语症”的定义和三角色协同机制，成功解决了 UMMs 在生成方面的短板，并且完全摆脱了外部数据依赖。这对于推动多模态人工智能向更自主、更强大的方向发展具有重要意义。**

**Key Findings:**

- To address this, we propose UniCorn, a simple yet elegant self-improvement framework that eliminates the need for external data or teacher supervision.
- To validate the restoration of multimodal coherence, we introduce UniCycle, a cycle-consistency benchmark based on a Text to Image to Text reconstruction loop.
- These results highlight that our method significantly enhances T2I generation while maintaining robust comprehension, demonstrating the scalability of fully self-supervised refinement for unified multimodal intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03193v1)
- [arXiv](https://arxiv.org/abs/2601.03193v1)

---

<a id='2601.03181v1'></a>
## [Multi-Modal Data-Enhanced Foundation Models for Prediction and Control in Wireless Networks: A Survey](https://arxiv.org/abs/2601.03181v1)

**Authors:** Han Zhang, Mohammad Farzanullah, Mohammad Ghassemi, Akram Bin Sediq, Ali Afana, Melike Erol-Kantarci

**Published:** 2026-01-06

**Categories:** cs.NI, cs.AI, cs.CL, cs.CV

**Abstract:**

Foundation models (FMs) are recognized as a transformative breakthrough that has started to reshape the future of artificial intelligence (AI) across both academia and industry. The integration of FMs into wireless networks is expected to enable the development of general-purpose AI agents capable of handling diverse network management requests and highly complex wireless-related tasks involving multi-modal data. Inspired by these ideas, this work discusses the utilization of FMs, especially multi-modal FMs in wireless networks. We focus on two important types of tasks in wireless network management: prediction tasks and control tasks. In particular, we first discuss FMs-enabled multi-modal contextual information understanding in wireless networks. Then, we explain how FMs can be applied to prediction and control tasks, respectively. Following this, we introduce the development of wireless-specific FMs from two perspectives: available datasets for development and the methodologies used. Finally, we conclude with a discussion of the challenges and future directions for FM-enhanced wireless networks.

**Analysis:**

好的，这是对论文“Multi-Modal Data-Enhanced Foundation Models for Prediction and Control in Wireless Networks: A Survey”的全面中文摘要：

**论文摘要：多模态数据增强的无线网络预测与控制基础模型**

**1. 主要问题/研究问题：**

随着人工智能（AI）的快速发展，基础模型（Foundation Models, FMs）已成为一股变革性力量，有望重塑AI的未来。在无线网络领域，将FMs，特别是多模态FMs，集成进来，能够开发出能够处理多样化网络管理请求和复杂无线相关任务的通用AI代理。然而，目前的研究在如何利用FMs处理多模态数据以实现无线网络管理中的预测和控制任务方面，以及如何开发专门针对无线网络的FMs方面，仍存在不足。本文旨在全面探讨FMs在无线网络中的应用，重点关注多模态数据处理、预测和控制任务，并为开发无线网络特有的FMs提供指导。

**2. 关键创新/方法论贡献：**

*   **系统性综述：** 本文提供了对FMs在无线网络中应用的全面综述，重点关注多模态FMs。它详细介绍了FMs的演进、能力、典型模型以及部署方式，并阐述了如何有效利用它们处理上下文数据以实现网络管理。
*   **任务导向的分析：** 论文将无线网络管理任务分为两大类：预测任务和控制任务，并深入探讨了FMs在这些任务中的应用。
    *   **预测任务：** 涵盖了无线流量预测、信道状态信息（CSI）预测、无线链路故障预测和阻塞预测等。
    *   **控制任务：** 探讨了FMs如何指导强化学习（RL）代理的探索策略、提取观测的潜在表示，以及作为交互式代理在控制场景中发挥作用。
*   **无线网络特有FMs的开发：** 论文从两个关键视角介绍了开发无线网络特有FMs的方法：
    *   **数据集：** 总结了可用于FM开发、涵盖无线流量、射频（RF）信号和各种传感模态的现有数据集。
    *   **方法论：** 探讨了预训练（从头开始或利用现有模型）和微调FMs以适应无线网络的方法，包括自监督学习、跨模态学习和参数高效微调（PEFT）技术。
*   **多模态数据理解：** 论文重点介绍了FMs如何理解和利用无线网络中的多模态上下文信息，包括视觉数据、图信息、3D点云数据以及其他模态（如ISAC数据、RF数据和网络信息）。

**3. 主要结果及其意义：**

*   **通用AI代理的潜力：** FMs能够处理多样化的无线网络任务，并利用多模态数据提供更丰富的上下文理解，这预示着通用AI代理在无线网络管理中的巨大潜力。
*   **提升预测和控制能力：** FMs能够显著提高无线网络中预测任务（如流量预测、阻塞预测）和控制任务（如资源分配、波束管理）的准确性和效率。
*   **推动无线网络智能化：** 通过利用FMs，无线网络可以实现更智能、更自适应的网络管理，例如主动的资源调度、更优化的用户体验和更强的网络韧性。
*   **为未来研究奠定基础：** 本文为研究人员提供了关于FMs在无线网络中应用、开发和部署的全面视角，为未来的研究指明了方向。

**4. 论文中提到的局限性：**

*   **数据可用性与计算成本：** 开发无线网络特有FMs面临数据稀缺和标注成本高昂的挑战。同时，FMs的巨大规模带来了高昂的计算成本和部署难题。
*   **隐私问题：** 在联邦学习等协作式FM开发中，模型参数的传输可能暴露敏感信息，需要解决隐私保护问题。
*   **实时性与延迟：** FMs的计算量大，可能导致推理延迟，这对于对延迟敏感的无线网络应用是一个挑战。
*   **模型泛化能力：** 通用FMs可能缺乏对无线网络特有知识的深入理解，导致在特定任务上表现不佳，需要进行微调或领域适应。
*   **“幻觉”现象：** FMs有时会生成不准确或无意义的输出（“幻觉”），这会影响其鲁棒性和可靠性。

**5. 未来研究方向：**

*   **开发无线网络特有模态的FMs：** 需要开发能够有效处理无线网络特有数据模态（如CSI、RF信号）的FMs。
*   **增强预训练FMs的无线网络知识：** 通过微调、提示工程等技术，将无线网络特有知识注入预训练FMs，以提升其在无线通信任务中的性能。
*   **优化FMs的部署：** 研究如何在资源受限的无线网络边缘设备上高效部署大型FMs，包括模型压缩、量化、分布式推理等技术。
*   **解决延迟问题：** 探索低延迟的FM推理技术，如边缘计算、模型分区和硬件加速。
*   **提升模型鲁棒性与安全性：** 研究如何通过引入约束、数据过滤、领域知识等方法来提高FMs的鲁棒性，并解决模型中毒、对抗性攻击等安全问题。
*   **实现Agentic AI：** 将FMs作为智能代理，实现自主的决策和控制，以应对复杂多变的无线网络环境。
*   **克服多模态数据稀缺性：** 探索数据增强、合成数据生成、联邦学习等方法，以解决多模态无线数据稀缺的问题。

总而言之，这篇综述论文为理解和利用基础模型（FMs）在无线网络领域的潜力提供了全面的视角，特别强调了多模态数据在预测和控制任务中的关键作用，并为未来的研究和发展指明了方向。

**Key Findings:**

- Following this, we introduce the development of wireless-specific FMs from two perspectives: available datasets for development and the methodologies used.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03181v1)
- [arXiv](https://arxiv.org/abs/2601.03181v1)

---

<a id='2601.03178v1'></a>
## [DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](https://arxiv.org/abs/2601.03178v1)

**Authors:** Jiajun jiao, Haowei Zhu, Puyuan Yang, Jianghui Wang, Ji Liu, Ziqiong Liu, Dong Li, Yuejian Fang, Junhai Yong, Bin Wang, Emad Barsoum

**Published:** 2026-01-06

**Categories:** cs.CV

**Abstract:**

Diffusion models have achieved remarkable success in image and video generation. However, their inherently multiple step inference process imposes substantial computational overhead, hindering real-world deployment. Accelerating diffusion models is therefore essential, yet determining how to combine multiple model acceleration techniques remains a significant challenge. To address this issue, we introduce a framework driven by large language models (LLMs) for automated acceleration code generation and evaluation. First, we present DiffBench, a comprehensive benchmark that implements a three stage automated evaluation pipeline across diverse diffusion architectures, optimization combinations and deployment scenarios. Second, we propose DiffAgent, an agent that generates optimal acceleration strategies and codes for arbitrary diffusion models. DiffAgent employs a closed-loop workflow in which a planning component and a debugging component iteratively refine the output of a code generation component, while a genetic algorithm extracts performance feedback from the execution environment to guide subsequent code refinements. We provide a detailed explanation of the DiffBench construction and the design principles underlying DiffAgent. Extensive experiments show that DiffBench offers a thorough evaluation of generated codes and that DiffAgent significantly outperforms existing LLMs in producing effective diffusion acceleration strategies.

**Analysis:**

好的，这是对论文“DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation”的全面摘要，重点关注其在计算机视觉领域的贡献：

**论文摘要**

**1. 要解决的主要问题/研究问题：**

扩散模型在图像和视频生成方面取得了巨大成功，但其多步推理过程带来了显著的计算开销，阻碍了其在现实世界中的广泛应用。如何有效地结合多种模型加速技术以降低推理成本，同时保持生成质量，是一个巨大的挑战。现有方法需要专家干预和针对特定模型及部署场景的定制化工程，这在扩散模型架构日益多样化和部署环境日益复杂的情况下变得尤为困难。因此，论文旨在解决如何自动化生成高质量、优化的扩散模型加速代码的问题。

**2. 关键创新/方法贡献：**

该论文提出了一个端到端的、由大型语言模型（LLM）驱动的框架，用于自动化扩散模型加速代码的生成和评估。其核心贡献包括：

*   **DiffBench：** 一个全面的基准测试套件，用于评估 LLM 生成扩散模型加速代码的能力。DiffBench 包含 604 个任务，涵盖了多样化的扩散模型架构、优化组合和部署场景，并实现了一个三阶段的自动化评估流程（静态参数评估、绝对性能评估和相对性能分析），以确保对生成代码的严格评估。
*   **DiffAgent：** 一个基于 LLM 的智能体框架，用于生成任意扩散模型的最佳加速策略和代码。DiffAgent 采用闭环工作流程，集成了规划（Planning）、编码（Coding）和调试（Debugging）三个核心组件，并引入了一个遗传算法（Genetic Algorithm）选择器。该选择器利用执行环境提供的性能反馈，迭代地优化代码生成组件的输出，以满足用户指定的精度和效率目标。

**3. 主要结果及其意义：**

*   **DiffBench 的有效性：** 实验表明，DiffBench 提供了一个比现有基准更具挑战性的评估环境，能够有效区分不同 LLM 在扩散模型加速代码生成方面的能力。
*   **DiffAgent 的优越性：** 实验证明，DiffAgent 显著优于现有的 LLM，在生成有效的扩散模型加速策略方面取得了显著进步。与直接代码生成相比，DiffAgent 在不同难度级别的任务上，将 Claude Sonnet 4 的平均通过率从 54.30% 提升到 81.59%。
*   **模块化贡献：** 消融研究表明，DiffAgent 的三个核心模块（知识库、遗传算法和调试器）协同工作，共同实现了其卓越的性能。特别是遗传算法对于处理复杂性能约束的任务至关重要，知识库提供了广泛的优势，而调试器则增强了在高难度任务上的鲁棒性。
*   **自动化和效率提升：** DiffAgent 能够自动化地生成满足用户需求的高质量加速代码，显著减少了对人工干预的需求，并实现了可观的推理速度提升，同时保持了生成质量的损失在可接受范围内。

**4. 论文中提到的局限性：**

*   **计算资源需求：** 尽管 DiffAgent 旨在提高效率，但其迭代优化过程和 LLM 的调用仍然需要一定的计算资源。
*   **“硬”任务的挑战：** 对于一些极具挑战性的任务（“hard” samples），论文提到可能并不总是有有效的解决方案，这表明在某些极端情况下，LLM 的能力仍然受到限制。
*   **LLM 的固有局限性：** 尽管 DiffAgent 取得了显著进展，但 LLM 本身在理解复杂推理、处理细微的性能调优和避免级联错误方面仍可能存在局限性。

**5. 潜在的未来研究方向：**

*   **更广泛的加速技术集成：** 探索将更多新兴的扩散模型加速技术集成到 DiffAgent 框架中。
*   **更精细的性能调优：** 研究更高级的优化策略，以在保持极低质量损失的情况下实现更高的速度提升。
*   **跨模型和跨硬件的泛化能力：** 进一步提升 DiffAgent 在不同扩散模型架构和硬件平台上的泛化能力。
*   **用户交互和反馈机制：** 探索更直观的用户交互方式，以及更有效的反馈机制，以进一步优化代码生成过程。
*   **LLM 自身能力的提升：** 随着 LLM 技术的发展，其在代码理解、推理和优化方面的能力将不断增强，这将进一步推动 DiffAgent 的性能。

总而言之，这篇论文通过引入 DiffBench 和 DiffAgent，为解决扩散模型推理效率低下这一关键问题提供了创新的自动化解决方案。DiffBench 提供了一个标准化的评估平台，而 DiffAgent 则通过智能的 LLM 驱动的框架，实现了高质量的扩散模型加速代码生成，为该领域的研究和应用开辟了新的道路。

**Key Findings:**

- To address this issue, we introduce a framework driven by large language models (LLMs) for automated acceleration code generation and evaluation.
- First, we present DiffBench, a comprehensive benchmark that implements a three stage automated evaluation pipeline across diverse diffusion architectures, optimization combinations and deployment scenarios.
- Second, we propose DiffAgent, an agent that generates optimal acceleration strategies and codes for arbitrary diffusion models.
- Extensive experiments show that DiffBench offers a thorough evaluation of generated codes and that DiffAgent significantly outperforms existing LLMs in producing effective diffusion acceleration strategies.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03178v1)
- [arXiv](https://arxiv.org/abs/2601.03178v1)

---

<a id='2601.03127v1'></a>
## [Unified Thinker: A General Reasoning Modular Core for Image Generation](https://arxiv.org/abs/2601.03127v1)

**Authors:** Sashuai Zhou, Qiang Zhou, Jijin Hu, Hanqing Yang, Yue Cao, Junpeng Ma, Yinchao Ma, Jun Song, Tiezheng Ge, Cheng Yu, Bo Zheng, Zhou Zhao

**Published:** 2026-01-06

**Categories:** cs.CV, cs.AI

**Abstract:**

Despite impressive progress in high-fidelity image synthesis, generative models still struggle with logic-intensive instruction following, exposing a persistent reasoning--execution gap. Meanwhile, closed-source systems (e.g., Nano Banana) have demonstrated strong reasoning-driven image generation, highlighting a substantial gap to current open-source models. We argue that closing this gap requires not merely better visual generators, but executable reasoning: decomposing high-level intents into grounded, verifiable plans that directly steer the generative process. To this end, we propose Unified Thinker, a task-agnostic reasoning architecture for general image generation, designed as a unified planning core that can plug into diverse generators and workflows. Unified Thinker decouples a dedicated Thinker from the image Generator, enabling modular upgrades of reasoning without retraining the entire generative model. We further introduce a two-stage training paradigm: we first build a structured planning interface for the Thinker, then apply reinforcement learning to ground its policy in pixel-level feedback, encouraging plans that optimize visual correctness over textual plausibility. Extensive experiments on text-to-image generation and image editing show that Unified Thinker substantially improves image reasoning and generation quality.

**Analysis:**

好的，这是对论文“Unified Thinker: A General Reasoning Modular Core for Image Generation”的全面中文摘要：

**论文题目：** Unified Thinker: A General Reasoning Modular Core for Image Generation
**作者：** Sashuai Zhou, Qiang Zhou, Jijin Hu, Hanqing Yang, Yue Cao, Junpeng Ma, Yinchao Ma, Jun Song, Tiezheng Ge, Cheng Yu, Bo Zheng, Zhou Zhao

**摘要**

**1. 研究问题/核心挑战：**
尽管高保真图像合成技术取得了显著进展，但当前的生成模型在遵循复杂、逻辑密集型指令方面仍存在困难，表现出“推理-执行”的鸿沟。闭源系统（如 Nano Banana）在推理驱动的图像生成方面展现出强大能力，而开源模型在这方面仍有差距。论文认为，弥合这一差距不仅需要更好的视觉生成器，还需要可执行的推理能力，即能够将高层意图分解为可验证的、具体的计划，并直接指导生成过程。

**2. 主要创新与方法贡献：**
*   **Unified Thinker 框架：** 提出了一种名为 Unified Thinker 的任务无关推理架构，作为通用的规划核心，可插入各种图像生成器和工作流。该框架将一个独立的“Thinker”（思考者）模块与“Generator”（生成器）模块解耦。
*   **模块化设计：** Thinker 负责指令理解和规划，Generator 负责像素合成。这种解耦允许独立升级推理能力，而无需重新训练整个生成模型，增强了模块化和可迁移性。
*   **两阶段训练范式：**
    *   **结构化规划接口构建：** 首先，通过构建一个名为 HieraReason-40K 的数据集（包含指令与结构化、可执行计划的配对），训练 Thinker 生成所需的规划格式和基本逻辑分解。
    *   **执行导向的强化学习：** 接着，采用两阶段强化学习（RL）来弥合推理与执行之间的差距。第一阶段（Reasoning-Oriented RL）优化 Thinker 的规划能力，使其生成对 Generator 更有效的计划。第二阶段（Generation-Oriented RL）则在 Generator 中引入随机性，以优化其执行计划的保真度。这种方法将 Thinker 的策略直接建立在像素级反馈之上，鼓励生成优化视觉正确性的计划。

**3. 主要结果与意义：**
*   **显著的性能提升：** 在文本到图像生成和图像编辑任务上，Unified Thinker 显著提高了图像推理和生成质量，尤其在逻辑密集型指令遵循和约束满足方面表现出色。
*   **跨模型可迁移性：** 实验证明，解耦的 Thinker 模块能够学习可复用、可执行的推理模式，并能跨不同的生成器模型（如 Qwen-Image-Edit 和 BAGEL）进行迁移，验证了其通用性。
*   **缩小与闭源模型的差距：** 在 WiseBench 等基准测试中，Unified Thinker 显著缩小了与闭源模型（如 GPT-40）在推理能力上的差距。
*   **平衡多任务能力：** 尽管在某些低级编辑任务上可能存在轻微的性能权衡，但联合微调和两阶段 RL 训练能够有效缓解这种不匹配，并稳定多任务行为，最终在所有基准测试中取得一致的提升。

**4. 提及的局限性：**
*   **对中间表示和训练数据的依赖：** 方法的性能依赖于中间表示的质量、训练数据的覆盖范围以及 RL 中使用的奖励信号，这些都可能引入偏差并限制泛化能力。
*   **执行的挑战：** 尽管 Thinker 被设计为与生成器无关，但其可执行性并非完全独立于不同的生成器后端，尤其是在处理精细的几何变化、严格的局部性或精确的文本渲染等困难编辑任务时。
*   **推理延迟和计算成本：** 额外的规划阶段会增加推理延迟和计算成本，与直接提示单个生成器相比。

**5. 潜在的未来研究方向：**
论文并未明确列出未来研究方向，但基于其工作，可以推断以下潜在方向：
*   **更强大的中间表示：** 研究更丰富、更具表达力的中间表示，以更好地捕捉复杂视觉概念和逻辑关系。
*   **更精细的推理-执行对齐：** 探索更先进的对齐技术，以进一步缩小推理与像素级执行之间的差距，尤其是在处理更具挑战性的编辑任务时。
*   **更广泛的生成器支持：** 验证和扩展 Unified Thinker 在更多类型和架构的图像生成模型上的适用性。
*   **实时推理优化：** 探索减少推理延迟和计算成本的方法，以使其更适用于实时应用。
*   **多模态推理的泛化：** 将这种解耦的推理-生成框架扩展到其他多模态任务，如视频生成或3D内容创作。

**论文的独特性与重要性：**
Unified Thinker 的核心贡献在于其提出的 **解耦的“思考-执行”框架** 和 **两阶段的执行导向强化学习训练范式**。它不仅解决了当前生成模型在逻辑推理方面的短板，而且通过模块化设计实现了推理能力的 **高度可迁移性**，这对于构建更智能、更通用的图像生成系统具有重要意义。该方法通过将复杂的推理过程显式地结构化并与生成器紧密对齐，为实现更可靠、更可控的图像生成开辟了新的途径。

**Key Findings:**

- To this end, we propose Unified Thinker, a task-agnostic reasoning architecture for general image generation, designed as a unified planning core that can plug into diverse generators and workflows.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03127v1)
- [arXiv](https://arxiv.org/abs/2601.03127v1)

---

<a id='2601.03117v1'></a>
## [Transformers self-organize like newborn visual systems when trained in prenatal worlds](https://arxiv.org/abs/2601.03117v1)

**Authors:** Lalit Pandey, Samantha M. W. Wood, Justin N. Wood

**Published:** 2026-01-06

**Categories:** q-bio.NC, cs.AI, cs.CV

**Abstract:**

Do transformers learn like brains? A key challenge in addressing this question is that transformers and brains are trained on fundamentally different data. Brains are initially "trained" on prenatal sensory experiences (e.g., retinal waves), whereas transformers are typically trained on large datasets that are not biologically plausible. We reasoned that if transformers learn like brains, then they should develop the same structure as newborn brains when exposed to the same prenatal data. To test this prediction, we simulated prenatal visual input using a retinal wave generator. Then, using self-supervised temporal learning, we trained transformers to adapt to those retinal waves. During training, the transformers spontaneously developed the same structure as newborn visual systems: (1) early layers became sensitive to edges, (2) later layers became sensitive to shapes, and (3) the models developed larger receptive fields across layers. The organization of newborn visual systems emerges spontaneously when transformers adapt to a prenatal visual world. This developmental convergence suggests that brains and transformers learn in common ways and follow the same general fitting principles.

**Analysis:**

好的，这是对论文“Transformers self-organize like newborn visual systems when trained in prenatal worlds”的全面中文摘要，重点关注其研究问题、方法创新、主要结果、局限性和未来方向：

**论文题目：** Transformers self-organize like newborn visual systems when trained in prenatal worlds

**作者：** Lalit Pandey, Samantha M. W. Wood, Justin N. Wood

**摘要：**

这篇论文探讨了一个核心问题：**人工智能中的Transformer模型是否像大脑一样学习？** 论文指出，当前Transformer模型与大脑在学习数据上存在根本差异。大脑在出生前就通过产前感官体验（如视网膜波）进行“训练”，而Transformer模型通常使用不符合生物学原理的大型数据集进行训练。为了解决这一挑战，研究人员提出，如果Transformer模型真的像大脑一样学习，那么当它们暴露于与新生儿大脑相同的产前数据时，它们应该发展出相似的结构。

**关键创新与方法贡献：**

1.  **模拟产前视觉输入：** 论文的核心创新在于使用**视网膜波发生器**来模拟新生儿大脑在出生前的视觉输入。这是为了提供一个生物学上更具可信度的数据集，用于训练Transformer模型。
2.  **自监督时间学习：** 研究人员采用了**自监督时间学习**（self-supervised temporal learning）的训练目标。具体来说，他们使用了一种名为**ViT-CoT（Vision Transformer with Contrastive Learning through Time）**的模型，该模型通过将同一时间窗口内的图像嵌入拉近，同时将不同时间窗口的图像嵌入推开，来学习时空信息。
3.  **无先验硬编码：** 与传统的卷积神经网络（CNNs）不同，Transformer模型被设计为**通用学习者，不包含硬编码的领域特定先验**。这使得它们能够更清晰地揭示学习经验在塑造视觉系统结构中的作用。
4.  **多维度结构评估：** 论文通过三种关键指标来评估模型结构是否与新生儿视觉系统相似：
    *   **早期层的边缘敏感性（Edge Sensitivity）：** 模拟初级视觉皮层（V1）的特性。
    *   **后期层的形状敏感性（Shape Sensitivity）：** 模拟腹内侧颞叶皮层（inferior temporal cortex）的特性。
    *   **跨层感受野大小的增长（Larger Receptive Fields Across Layers）：** 模拟视觉系统层级化和视网膜拓扑结构的特征。

**主要结果及其意义：**

研究的主要结果表明，当Transformer模型（ViT-CoT）在模拟的产前视网膜波数据上进行训练时，它们**自发地发展出了与新生儿视觉系统相似的三种结构特征**：

1.  **早期层对边缘敏感：** 模型在早期层学会了检测定向边缘。
2.  **后期层对形状敏感：** 模型在后期层学会了区分和识别形状。
3.  **跨层感受野逐渐增大：** 模型展现出层级化的结构，感受野在深层网络中变得更大。

这些结果具有重要意义：

*   **支持选择论（Selectional Theories）：** 这一发现有力地支持了选择论的观点，即视觉系统的结构很大程度上是**通过适应环境的产前经验而形成的**，而无需基因的直接指令。
*   **揭示通用学习原则：** 这表明大脑和Transformer模型在学习方式上存在**共同的原则**，即通过**通用的时空拟合（space-time fitting）原理**来适应环境。
*   **为AI和神经科学提供新视角：** Transformer模型可以作为**产前和产后大脑发育的基线模型**，为理解视觉系统结构提供计算模型，并为AI工程提供新的思路。

**论文提及的局限性：**

1.  **仅测试了三个关键特征：** 研究仅关注了新生儿视觉系统的三个主要结构特征，未来需要探索其他特征是否也能自发形成。
2.  **学习目标的多样性：** 研究使用了特定的自监督时间学习目标，未来需要探索其他学习目标是否也能产生类似的组织结构。
3.  **产前经验的单一性：** 研究仅模拟了视觉领域的产前经验，而大脑的发育还受到听觉、本体感觉和触觉等多种感官输入的影响。

**潜在的未来研究方向：**

1.  **探索其他结构特征：** 研究其他新生儿视觉系统的特征，看它们是否也能在类似条件下自发形成。
2.  **评估其他学习目标：** 测试不同的无监督时间学习目标，以了解其对模型结构形成的影响。
3.  **模拟多模态产前经验：** 扩展研究范围，模拟听觉、本体感觉和触觉等其他领域的产前输入，以更全面地理解大脑发育。
4.  **弥合人机学习鸿沟：** 通过赋予Transformer模型产前发育阶段，使其更像大脑，从而缩小人类与机器之间的学习差距。

总而言之，这篇论文通过创新的实验设计，证明了Transformer模型在模拟的产前视觉环境中，能够自发地发展出与新生儿视觉系统相似的结构。这一发现不仅为理解大脑发育提供了新的计算视角，也为构建更具生物学合理性的AI模型指明了方向。

**Key Findings:**

- We reasoned that if transformers learn like brains, then they should develop the same structure as newborn brains when exposed to the same prenatal data.
- During training, the transformers spontaneously developed the same structure as newborn visual systems: (1) early layers became sensitive to edges, (2) later layers became sensitive to shapes, and (3) the models developed larger receptive fields across layers.
- The organization of newborn visual systems emerges spontaneously when transformers adapt to a prenatal visual world.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.03117v1)
- [arXiv](https://arxiv.org/abs/2601.03117v1)

---

