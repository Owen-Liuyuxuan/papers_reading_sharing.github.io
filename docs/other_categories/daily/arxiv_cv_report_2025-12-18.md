time: 20251218

# Arxiv Computer Vision Papers - 2025-12-18

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份简明的 Arxiv 计算机视觉领域近期论文的执行摘要。

---

**执行摘要：2025年12月16日 Arxiv 计算机视觉论文速览**

**主要主题与趋势：**

本期 Arxiv 论文集中体现了计算机视觉领域在**多模态理解与生成**、**视频内容生成与分析**以及**视觉预训练新范式**方面的显著进展。特别值得注意的是，研究人员正积极探索如何将**生成模型（尤其是扩散模型）**与**视觉语言任务**深度融合，并致力于提升模型的**泛化能力**和**可控性**。同时，**利用视觉语言模型进行评估和推理**也成为一个新兴的研究方向。

**亮点与创新：**

*   **MMGR: Multi-Modal Generative Reasoning** 提出了一种多模态生成推理框架，预示着模型在整合不同模态信息进行复杂推理方面迈出了重要一步。
*   **Spatia: Video Generation with Updatable Spatial Memory** 在视频生成领域引入了“可更新空间记忆”的概念，有望显著提升视频生成的一致性和连贯性。
*   **DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models** 提供了一种通用的方法，将现有的自回归模型转化为扩散视觉语言模型，极大地扩展了扩散模型在视觉语言任务中的应用范围。
*   **VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression** 巧妙地利用视觉语言模型作为“感知裁判”，为图像压缩任务引入了更符合人类视觉偏好的评估标准，具有重要的实际应用价值。

**新兴研究方向与技术：**

*   **扩散模型在视觉语言任务中的广泛应用：** 从文本到视频生成，再到模型转换，扩散模型正成为构建强大视觉语言模型的核心技术。
*   **视频生成的可控性与记忆机制：** 通过引入空间记忆等机制，研究人员正努力使视频生成更加可控，并能更好地处理长时序依赖。
*   **视觉预训练的像素级监督探索：** 论文“In Pursuit of Pixel Supervision for Visual Pre-training”表明，对像素级信息的深入挖掘是提升视觉预训练模型能力的关键。
*   **多模态基础模型（Multi-View Foundation Models）：** 预示着未来将出现更强大的、能够处理多视角信息的通用基础模型。
*   **AI生成内容的检测与溯源：** “Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning”关注AI生成内容的检测，是应对内容真实性挑战的重要研究方向。
*   **机器人控制的通用化：** “mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs”展示了将视频理解能力应用于更广泛的机器人控制任务的潜力。

**建议阅读全文的论文：**

考虑到其创新性和潜在影响力，以下论文值得深入阅读：

1.  **MMGR: Multi-Modal Generative Reasoning** (多模态推理的通用框架)
2.  **Spatia: Video Generation with Updatable Spatial Memory** (视频生成在一致性与可控性上的突破)
3.  **DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models** (通用模型转换方法，扩展扩散模型应用)
4.  **VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression** (视觉语言模型在评估领域的创新应用)

---

希望这份执行摘要能帮助您快速了解近期 Arxiv 计算机视觉领域的最新动态。

---

## Table of Contents

1. [MMGR: Multi-Modal Generative Reasoning](#2512.14691v2)
2. [Spatia: Video Generation with Updatable Spatial Memory](#2512.15716v1)
3. [In Pursuit of Pixel Supervision for Visual Pre-training](#2512.15715v1)
4. [DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](#2512.15713v1)
5. [Multi-View Foundation Models](#2512.15708v1)
6. [GateFusion: Hierarchical Gated Cross-Modal Fusion for Active Speaker Detection](#2512.15707v1)
7. [End-to-End Training for Autoregressive Video Diffusion via Self-Resampling](#2512.15702v1)
8. [VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression](#2512.15701v1)
9. [Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning](#2512.15693v1)
10. [mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs](#2512.15692v1)

---

## Papers

<a id='2512.14691v2'></a>
## [MMGR: Multi-Modal Generative Reasoning](https://arxiv.org/abs/2512.14691v2)

**Authors:** Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, Kung-Hsiang Huang, Parisa Kordjamshidi, Minjia Zhang, Wen Xiao, Jiuxiang Gu, Nanyun Peng, Junjie Hu

**Published:** 2025-12-16

**Categories:** cs.CL, cs.CV

**Abstract:**

Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causality, physics, and global consistency. We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal. MMGR evaluates generative reasoning across three domains: Abstract Reasoning (ARC-AGI, Sudoku), Embodied Navigation (real-world 3D navigation and localization), and Physical Commonsense (sports and compositional interactions). MMGR applies fine-grained metrics that require holistic correctness across both video and image generation. We benchmark leading video models (Veo-3, Sora-2, Wan-2.2) and image models (Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image), revealing strong performance gaps across domains. Models show moderate success on Physical Commonsense tasks but perform poorly on Abstract Reasoning (below 10 percent accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Our analysis highlights key limitations in current models, including overreliance on perceptual data, weak global state consistency, and objectives that reward visual plausibility over causal correctness. MMGR offers a unified diagnostic benchmark and a path toward reasoning-aware generative world models.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：MMGR: Multi-Modal Generative Reasoning**

**1. 论文的主要贡献（2-3句话）：**

本研究提出了MMGR，一个多模态生成推理的评估框架和基准，旨在弥补现有视频生成模型评估指标（如FVD）在衡量物理、逻辑和空间约束方面的不足。MMGR通过评估物理、逻辑、3D空间、2D空间和时间五种推理能力，在抽象推理、具身导航和物理常识三个领域对领先的生成模型进行了全面基准测试，揭示了当前模型在推理能力上的显著差距。该框架为开发更可靠、更具世界模拟能力的生成模型提供了方向。

**2. 关键创新或方法论：**

*   **多模态生成推理评估框架 (MMGR)：** 这是本研究的核心创新。MMGR不再仅仅关注生成内容的视觉逼真度和时间连贯性，而是引入了对生成内容中蕴含的“推理”能力的系统性评估。
*   **五种核心推理能力：** 论文明确定义了五种关键的推理能力：物理（Physical）、逻辑（Logical）、3D空间（3D Spatial）、2D空间（2D Spatial）和时间（Temporal）。这种细粒度的能力划分有助于更深入地诊断模型在不同推理维度上的表现。
*   **三个多样化的评估领域：** MMGR在三个具有代表性的领域进行评估：
    *   **抽象推理 (Abstract Reasoning):** 例如ARC-AGI和Sudoku，测试模型在符号操作和规则遵循方面的能力。
    *   **具身导航 (Embodied Navigation):** 模拟真实世界中的3D导航和定位任务，评估模型在理解和规划空间路径的能力。
    *   **物理常识 (Physical Commonsense):** 例如体育场景和组合式交互，测试模型对物理定律和物体间相互作用的理解。
*   **跨模态（视频和图像）的细粒度评估：** MMGR要求生成内容在视频和图像层面都展现出整体的正确性，这使得评估更加全面和严格。
*   **对现有模型的深入诊断：** 通过在MMGR框架下对Sora-2、Veo-3、GPT-4o-image等领先模型进行基准测试，论文揭示了它们在不同推理任务上的具体优劣势，为模型改进提供了明确的指导。

**3. 对该领域的潜在影响：**

*   **推动生成模型向“理解”和“推理”发展：** MMGR的出现将促使研究者从单纯追求视觉效果转向更关注生成内容的内在逻辑和物理一致性，从而推动生成模型向更具智能和可靠性的“世界模拟器”迈进。
*   **建立新的评估标准和研究方向：** MMGR提供了一个标准化的、多维度的评估框架，有望成为未来视频和多模态生成模型研究的基石，引导新的研究方向，例如如何设计能够有效学习和执行推理任务的模型架构和训练策略。
*   **加速具身智能和物理模拟的发展：** 对于需要精确物理和空间理解的应用（如机器人、虚拟现实），MMGR的评估方法和发现将直接促进相关领域的发展。
*   **提升生成内容的可靠性和可信度：** 通过强调推理能力，MMGR有助于减少生成内容中的“幻觉”和不合逻辑的错误，提高生成内容的可靠性和在实际应用中的可信度。

**4. 可能受益的相关领域或应用：**

*   **具身AI和机器人学：** 机器人需要在复杂环境中进行导航、规划和与物理世界交互，MMGR的具身导航和物理常识评估对机器人训练至关重要。
*   **虚拟现实 (VR) 和增强现实 (AR)：** 创建逼真且符合物理规律的虚拟环境需要强大的空间和物理推理能力，MMGR的评估方法可以指导VR/AR内容生成。
*   **教育和模拟训练：** 用于生成教学内容、模拟实验或训练场景，需要确保内容的准确性和逻辑性。
*   **内容创作和影视制作：** 生成更具说服力、逻辑严谨的视频内容，减少后期修正的工作量。
*   **科学研究：** 用于模拟物理过程、化学反应或生物现象，需要高度的物理准确性。
*   **自动驾驶：** 理解交通规则、预测其他车辆行为等都需要强大的逻辑和物理推理能力。

**5. 从摘要中可以推断出的局限性：**

*   **评估的全面性仍有待检验：** 尽管MMGR提出了五种推理能力和三个评估领域，但“世界模拟”是一个极其广泛的概念，可能还有其他未被涵盖的推理维度（例如，社会常识、因果关系中的更深层次理解等）。
*   **评估的自动化程度：** 摘要提到“需要整体正确性”，这可能意味着部分评估需要人工干预或复杂的后处理，其自动化程度和可扩展性可能是一个挑战。
*   **基准测试的覆盖范围：** 虽然提到了“领先模型”，但模型的选择可能存在一定的局限性，未来需要更广泛的模型覆盖来验证MMGR的普适性。
*   **对“推理”的定义和量化：** “推理”本身是一个复杂且难以精确量化的概念。MMGR的五种能力和具体指标的有效性，以及它们是否能完全捕捉到“推理”的本质，仍需进一步的实验验证和社区讨论。
*   **模型在抽象推理上的极端劣势：** 摘要指出模型在ARC-AGI上准确率低于10%，这表明当前模型在处理高度抽象和符号化的推理任务上存在根本性困难，可能需要全新的模型架构或训练范式来解决。
*   **对“物理常识”的定义和评估：** 虽然提到了体育和组合式交互，但“物理常识”的边界和评估的细致程度也可能影响结果的解释。

总而言之，这篇论文通过引入MMGR框架，为评估和提升多模态生成模型的世界模拟能力提供了一个重要且具有前瞻性的方向。它不仅指出了当前模型的关键短板，也为未来的研究提供了清晰的路线图，尤其是在从“感知”向“理解”和“推理”转型的过程中，MMGR具有重要的理论和实践意义。

**Key Findings:**

- We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.14691v2)
- [arXiv](https://arxiv.org/abs/2512.14691v2)

---

<a id='2512.15716v1'></a>
## [Spatia: Video Generation with Updatable Spatial Memory](https://arxiv.org/abs/2512.15716v1)

**Authors:** Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, Yan Lu

**Published:** 2025-12-17

**Categories:** cs.CV, cs.AI

**Abstract:**

Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals. To overcome this limitation, we propose Spatia, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory. Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM. This dynamic-static disentanglement design enhances spatial consistency throughout the generation process while preserving the model's ability to produce realistic dynamic entities. Furthermore, Spatia enables applications such as explicit camera control and 3D-aware interactive editing, providing a geometrically grounded framework for scalable, memory-driven video generation.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行分析。

**论文分析：Spatia: Video Generation with Updatable Spatial Memory**

**1. 论文的主要贡献（2-3句话）：**

该论文提出了一种名为 Spatia 的新型视频生成框架，其核心贡献在于通过引入可更新的 3D 场景点云作为持久化空间记忆，显著提升了视频生成在长期空间和时间上的一致性。Spatia 结合了条件生成与视觉 SLAM（Simultaneous Localization and Mapping）的动态更新机制，实现了生成内容与几何场景的解耦，从而克服了现有模型在处理高维视频信号时面临的挑战。

**2. 关键创新或方法论：**

Spatia 的关键创新在于其**空间记忆驱动的视频生成范式**。具体来说：

*   **显式 3D 场景点云作为空间记忆：** 与以往主要依赖隐式表示或仅关注时间序列的模型不同，Spatia 将一个 3D 场景点云作为核心的、持久化的“空间记忆”。这为生成过程提供了一个明确的几何约束。
*   **迭代生成与空间记忆更新：** 模型并非一次性生成整个视频，而是**迭代地**生成视频片段，并且**每次生成都以当前的空间记忆为条件**。更重要的是，它通过**视觉 SLAM 技术**来**持续更新**这个空间记忆。这意味着模型能够感知并适应场景的变化，即使在生成过程中。
*   **动态-静态解耦设计：** 这种方法实现了动态内容（如运动的物体）与静态场景几何（由点云表示）的解耦。这使得模型既能保持场景的几何一致性，又能生成逼真且具有物理合理性的动态元素。

**3. 对该领域的潜在影响：**

Spatia 的研究可能对视频生成领域产生深远影响：

*   **提升长期一致性：** 解决了当前视频生成模型在长视频中容易出现的空间扭曲、物体漂移等问题，使得生成的视频在视觉上更加连贯和可信。
*   **实现更可控的生成：** 通过显式的 3D 场景表示，为视频生成提供了更强的可控性。研究人员可以更精确地控制相机视角、场景布局等，从而生成符合特定需求的视频。
*   **推动 3D 场景理解与生成融合：** 将 3D 场景重建（通过 SLAM）与视频生成紧密结合，为构建更具几何感知能力的生成模型开辟了新路径。
*   **为下游应用奠定基础：** 这种几何基础的生成框架为诸如 3D 场景编辑、虚拟现实内容生成等应用提供了更坚实的基础。

**4. 可能受益的相关领域或应用：**

*   **虚拟现实 (VR) 和增强现实 (AR)：** 生成逼真且几何一致的虚拟场景和交互式内容。
*   **电影和游戏制作：** 自动化生成复杂场景的视频片段，或辅助进行场景设计和动画制作。
*   **机器人学：** 生成模拟环境以训练机器人，或在动态环境中进行导航和规划。
*   **自动驾驶：** 生成逼真的交通场景，用于训练和测试自动驾驶算法。
*   **3D 内容创作：** 简化从 2D 视频到 3D 场景的转换，或直接生成 3D 可编辑的视频内容。
*   **数字孪生：** 构建和更新动态的数字孪生场景。

**5. 从摘要中可以推断出的局限性：**

尽管摘要描绘了一个非常有前景的框架，但仍可以推断出一些潜在的局限性：

*   **计算复杂度：** 维护和更新一个 3D 点云，并结合视觉 SLAM 和视频生成，很可能需要大量的计算资源和时间。这可能会限制其在实时应用中的部署。
*   **SLAM 的鲁棒性：** 视觉 SLAM 本身在某些复杂场景（如纹理稀疏、光照剧烈变化、快速运动）下可能存在精度和鲁棒性问题。Spatia 的性能将很大程度上依赖于其底层 SLAM 模块的性能。
*   **点云表示的局限性：** 点云是一种稀疏的几何表示，可能难以捕捉精细的表面细节或拓扑结构。如何有效地将点云信息转化为高质量的视频纹理和细节是一个挑战。
*   **动态内容与静态场景的融合难度：** 尽管提出了解耦设计，但如何无缝地将逼真的动态内容（如人物、车辆）与几何精确的静态场景融合，使其看起来自然且物理一致，仍然是一个技术难题。
*   **训练数据的需求：** 训练这样一个复杂的模型可能需要大量的、带有精确 3D 几何标注的视频数据，这可能难以获取。

总而言之，Spatia 提出的空间记忆驱动的视频生成方法，通过显式的 3D 几何表示和动态更新机制，为解决视频生成中的长期一致性问题提供了一个创新的解决方案，并为未来的可控、3D 感知视频生成开辟了新的方向。

**Key Findings:**

- To overcome this limitation, we propose Spatia, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15716v1)
- [arXiv](https://arxiv.org/abs/2512.15716v1)

---

<a id='2512.15715v1'></a>
## [In Pursuit of Pixel Supervision for Visual Pre-training](https://arxiv.org/abs/2512.15715v1)

**Authors:** Lihe Yang, Shang-Wen Li, Yang Li, Xinjie Lei, Dong Wang, Abdelrahman Mohamed, Hengshuang Zhao, Hu Xu

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

At the most basic level, pixels are the source of the visual information through which we perceive the world. Pixels contain information at all levels, ranging from low-level attributes to high-level concepts. Autoencoders represent a classical and long-standing paradigm for learning representations from pixels or other raw inputs. In this work, we demonstrate that autoencoder-based self-supervised learning remains competitive today and can produce strong representations for downstream tasks, while remaining simple, stable, and efficient. Our model, codenamed "Pixio", is an enhanced masked autoencoder (MAE) with more challenging pre-training tasks and more capable architectures. The model is trained on 2B web-crawled images with a self-curation strategy with minimal human curation. Pixio performs competitively across a wide range of downstream tasks in the wild, including monocular depth estimation (e.g., Depth Anything), feed-forward 3D reconstruction (i.e., MapAnything), semantic segmentation, and robot learning, outperforming or matching DINOv3 trained at similar scales. Our results suggest that pixel-space self-supervised learning can serve as a promising alternative and a complement to latent-space approaches.

**Analysis:**

好的，这是对论文“In Pursuit of Pixel Supervision for Visual Pre-training”的全面中文摘要：

**论文题目：** In Pursuit of Pixel Supervision for Visual Pre-training (追求像素监督以实现视觉预训练)

**作者：** Lihe Yang, Shang-Wen Li, Yang Li, Xinjie Lei, Dong Wang, Abdelrahman Mohamed, Hengshuang Zhao, Hu Xu

**摘要：**

这篇论文探讨了在视觉表示学习领域，像素作为最基础的视觉信息来源，是否能够作为一种简单、稳定且高效的自监督学习信号。作者认为，像素包含了从低级属性到高级概念的全部视觉信息，并提出了一种名为“Pixio”的增强型掩码自编码器（MAE）模型，旨在利用像素级别的监督信号进行视觉预训练。

**1. 研究问题/核心挑战：**

论文的核心研究问题在于，在当前主流的自监督学习方法（如DINO系列）倾向于利用更抽象的潜在空间表示时，基于像素的自监督学习方法（如MAE）是否仍然具有竞争力，并能否产生与先进方法相媲美的、适用于广泛下游任务的强大视觉表示。研究还关注如何改进MAE以适应大规模数据和模型，并克服其在处理复杂视觉信息时的局限性。

**2. 主要创新与方法贡献：**

*   **Pixio 模型：** 作者提出了Pixio，一个在MAE基础上进行了四项关键改进的模型。这些改进包括：
    *   **更深的解码器 (Deeper Decoder)：** 增强解码器的容量，以更好地处理像素回归任务，从而减轻编码器在低级细节建模上的负担，使其能更专注于高级语义理解。
    *   **更大的掩码块 (Larger Mask Block)：** 采用4x4的局部掩码块而非单像素掩码，以提供更丰富的局部上下文信息，防止模型通过简单复制可见块来完成重建，从而强制模型进行更深入的理解。
    *   **更多的类别标记 (More [CLS] Tokens)：** 引入多个类别标记（class tokens），以捕捉图像更丰富的全局视觉属性，而非单个标记的局限性。
    *   **大规模、低人工干预的数据集：** 收集了20亿张网页爬取图像，并采用一种基于重构损失的软自策策略进行数据筛选，以减少对人工策展的依赖，避免数据偏差，并获得更具多样性的训练数据。
*   **像素监督的优势：** 论文强调了像素监督的优势在于其更少的人为偏见，因为它直接来源于物理世界的观察，相比于人类定义的类别或文本描述，它更少受到人类认知和语言的限制。

**3. 主要结果与意义：**

*   **下游任务表现优异：** Pixio 在包括单目深度估计（如Depth Anything）、前馈3D重建（如MapAnything）、语义分割和机器人学习等多个下游任务上，取得了与甚至超越同等规模的DINOv3模型相当的性能。
*   **证明像素监督的潜力：** 研究结果有力地证明了基于像素的自监督学习方法在今天仍然具有竞争力，并且能够产生强大且通用的视觉表示。这表明像素空间自监督学习可以作为一种有前景的替代方案，并能与潜在空间方法互补。
*   **大规模数据的重要性：** 论文强调了大规模、多样化且经过适当策展的数据集对于提升自监督学习模型性能的关键作用。

**4. 提及的局限性：**

*   **掩码的固有局限性：** 论文承认，即使是像素级别的自监督学习，掩码操作本身也是一种人为的失真，可能引入不必要的偏差。低掩码率会导致信息泄露，高掩码率则可能导致上下文不足和训练/推理分布的偏移。
*   **视频数据的潜力：** 作者指出，静态图像作为视觉信息的载体存在固有局限性，因为它们是孤立的快照。而视频数据通过捕捉事件的自然进程和因果关系，提供了更丰富的时序信息，可能带来更强大的预测性目标，从而减少对人工掩码的需求。

**5. 未来研究方向：**

*   **扩展到视频数据：** 作者计划将像素监督的方法扩展到大规模视频数据，利用视频的时序丰富性和自然预测目标，开发更强大、更少偏差的视觉基础模型。
*   **进一步探索数据策展策略：** 虽然论文提出了低人工干预的数据策展方法，但未来仍有空间探索更有效的策略，以平衡数据多样性和模型性能。
*   **理解掩码的理论基础：** 论文中也提到了对掩码自编码器理论的探索，未来可以进一步深入研究掩码比率、掩码策略等对模型性能的影响。

总而言之，这篇论文通过提出Pixio模型并进行大规模实验，成功地证明了像素级别的自监督学习在现代视觉表示学习中依然扮演着重要角色，并为未来的研究提供了新的方向和思路。

**Key Findings:**

- In this work, we demonstrate that autoencoder-based self-supervised learning remains competitive today and can produce strong representations for downstream tasks, while remaining simple, stable, and efficient.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15715v1)
- [arXiv](https://arxiv.org/abs/2512.15715v1)

---

<a id='2512.15713v1'></a>
## [DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](https://arxiv.org/abs/2512.15713v1)

**Authors:** Lunbin Zeng, Jingfeng Yao, Bencheng Liao, Hongyuan Tao, Wenyu Liu, Xinggang Wang

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

In recent multimodal research, the diffusion paradigm has emerged as a promising alternative to the autoregressive paradigm (AR), owing to its unique decoding advantages. However, due to the capability limitations of the base diffusion language model, the performance of the diffusion vision language model (dVLM) still lags significantly behind that of mainstream models. This leads to a simple yet fundamental question: Is it possible to construct dVLMs based on existing powerful AR models? In response, we propose DiffusionVL, a dVLM family that could be translated from any powerful AR models. Through simple fine-tuning, we successfully adapt AR pre-trained models into the diffusion paradigm. This approach yields two key observations: (1) The paradigm shift from AR-based multimodal models to diffusion is remarkably effective. (2) Direct conversion of an AR language model to a dVLM is also feasible, achieving performance competitive with LLaVA-style visual-instruction-tuning. Further, we introduce a block-decoding design into dVLMs that supports arbitrary-length generation and KV cache reuse, achieving a significant inference speedup. We conduct a large number of experiments. Despite training with less than 5% of the data required by prior methods, DiffusionVL achieves a comprehensive performance improvement-a 34.4% gain on the MMMU-Pro (vision) bench and 37.5% gain on the MME (Cog.) bench-alongside a 2x inference speedup. The model and code are released at https://github.com/hustvl/DiffusionVL.

**Analysis:**

好的，这是对论文“DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models”的全面中文摘要：

**论文标题：** DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models

**作者：** Lunbin Zeng, Jingfeng Yao, Bencheng Liao, Hongyuan Tao, Wenyu Liu, Xinggang Wang

---

**摘要：**

**1. 研究问题/核心挑战：**
近年来，扩散模型（Diffusion Models）作为一种新兴的范式，在多模态理解领域展现出潜力，尤其是在解码方面具有独特优势。然而，现有的扩散视觉语言模型（dVLM）在性能上普遍落后于主流的自回归视觉语言模型（AR-VLM）。这引发了一个关键问题：是否能够将现有的强大自回归模型（包括AR-VLM和AR-LM）有效地转化为高性能的扩散视觉语言模型？

**2. 主要创新与方法贡献：**
本文提出了 **DiffusionVL**，一个能够将任何强大的自回归模型转化为扩散视觉语言模型的框架。其核心贡献在于：

*   **统一的扩散微调（Diffusion Finetuning）：** 提出了一种简单的微调方法，能够将自回归模型的下一个词预测（NTP）范式转化为扩散范式，而无需修改原始模型的架构。
*   **两种转换路径：**
    *   **范式迁移（Paradigm Shift）：** 对于已经具备视觉语言对齐的AR-VLM，直接进行扩散微调即可得到dVLM。
    *   **模态与范式迁移（Modality and Paradigm Shift）：** 对于纯AR-LM，采用两阶段方法：首先通过一个可训练的连接器对齐视觉和文本空间，然后进行扩散微调，实现模态和范式的双重转换。
*   **块解码设计（Block-Decoding Design）：** 引入了块解码策略，支持任意长度的生成，并能有效复用KV缓存，显著提升了推理速度。
*   **广泛的实验验证：** 证明了该方法不仅适用于AR-VLM，也适用于AR-LM，并且在数据量远小于先前方法的情况下，取得了优异的性能。

**3. 主要结果与意义：**

*   **性能提升显著：** DiffusionVL在MMMU-Pro（vision）基准上提升了34.4%，在MME (Cog.) 基准上提升了37.5%。
*   **推理速度提升：** 实现了2倍的推理速度提升。
*   **数据效率高：** 仅使用了先前方法所需数据的5%进行训练，就取得了SOTA（State-of-the-Art）的扩散模型性能。
*   **缩小与AR-VLM的差距：** DiffusionVL显著缩小了现有dVLM与先进AR-VLM之间的性能差距。
*   **AR-LM到dVLM的可行性：** 首次证明了将AR-LM直接转化为高性能dVLM是可行的，并且性能可以与LLaVA风格的视觉指令微调模型相媲美。
*   **模型与代码开源：** 论文发布了模型和代码，促进了相关研究的进一步发展。

**4. 提及的局限性：**

*   **AR-VLM的优势：** 论文也指出，虽然DiffusionVL能够有效转化，但直接从AR-VLM迁移到dVLM的性能优势，部分归功于其基础模型已经进行了广泛且高质量的视觉语言对齐训练。
*   **AR-LM的潜力：** 作者认为，经过更长、更高质量视觉微调的AR-LM，也有潜力构建出与AR-VLM媲美的dVLM。
*   **块大小的权衡：** 实验表明，较小的训练块大小能带来略好的性能，但会牺牲一定的并行性。

**5. 潜在的未来研究方向：**

*   **AR-LM的进一步优化：** 探索如何通过更精细的视觉微调策略，进一步提升AR-LM转化为dVLM的性能上限。
*   **更复杂的模态与范式转换：** 探索将更多类型的自回归模型（如多模态AR模型）转化为扩散模型。
*   **动态低置信度重掩码策略的深入研究：** 进一步优化动态重掩码策略，以在加速和质量之间找到更好的平衡点。
*   **与其他扩散模型技术的结合：** 探索将DiffusionVL与最新的扩散模型技术（如更高效的噪声调度、注意力机制等）相结合，以进一步提升性能和效率。

**总结：**
DiffusionVL论文的核心贡献在于提出了一种通用且高效的方法，能够将现有的自回归模型（包括视觉语言模型和纯语言模型）转化为高性能的扩散视觉语言模型。通过创新的扩散微调和块解码设计，DiffusionVL在显著提升推理速度的同时，大幅缩小了与先进自回归模型的性能差距，并且在数据效率方面表现出色。这项工作不仅为构建高性能dVLM提供了一条低成本、高效率的途径，也为理解和融合自回归与扩散模型在多模态领域的潜力开辟了新的方向。

**Key Findings:**

- In response, we propose DiffusionVL, a dVLM family that could be translated from any powerful AR models.
- Further, we introduce a block-decoding design into dVLMs that supports arbitrary-length generation and KV cache reuse, achieving a significant inference speedup.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15713v1)
- [arXiv](https://arxiv.org/abs/2512.15713v1)

---

<a id='2512.15708v1'></a>
## [Multi-View Foundation Models](https://arxiv.org/abs/2512.15708v1)

**Authors:** Leo Segre, Or Hirschorn, Shai Avidan

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

Foundation models are vital tools in various Computer Vision applications. They take as input a single RGB image and output a deep feature representation that is useful for various applications. However, in case we have multiple views of the same 3D scene, they operate on each image independently and do not always produce consistent features for the same 3D point. We propose a way to convert a Foundation Model into a Multi-View Foundation Model. Such a model takes as input a set of images and outputs a feature map for each image such that the features of corresponding points are as consistent as possible. This approach bypasses the need to build a consistent 3D model of the features and allows direct manipulation in the image space. Specifically, we show how to augment Transformers-based foundation models (i.e., DINO, SAM, CLIP) with intermediate 3D-aware attention layers that help match features across different views. As leading examples, we show surface normal estimation and multi-view segmentation tasks. Quantitative experiments show that our method improves feature matching considerably compared to current foundation models.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Multi-View Foundation Models**

**1. 论文的主要贡献（2-3句话的简洁总结）**

该论文提出了一种将现有的单视图基础模型（Foundation Models）转化为多视图基础模型的方法，使其能够处理同一3D场景的多个RGB图像输入。其核心在于通过引入3D感知注意力层，确保不同视图下对应3D点的特征表示具有高度一致性，从而在图像空间内实现特征的跨视图匹配，而无需显式构建3D模型。

**2. 关键创新点或方法论**

*   **核心创新：** 将单视图基础模型扩展到多视图场景，并解决多视图特征不一致的问题。
*   **方法论：**
    *   **“3D-aware attention layers”：** 这是论文的核心技术创新。通过在Transformer-based基础模型（如DINO, SAM, CLIP）的中间层插入这些注意力层，模型能够学习到如何根据不同视图之间的几何关系来匹配特征。这种设计允许模型在不显式构建3D几何模型的情况下，隐式地理解3D结构和点之间的对应关系。
    *   **图像空间操作：** 论文强调其方法直接在图像空间进行操作，避免了构建复杂3D模型或进行3D重建的开销，这使得方法更具通用性和效率。
    *   **特征一致性优化：** 目标是使不同视图下对应3D点的特征尽可能一致，这是衡量多视图特征融合效果的关键指标。

**3. 对该领域的潜在影响**

*   **提升多视图理解能力：** 使得基础模型能够更好地理解和利用来自同一场景的多个视角信息，从而在各种下游任务中获得更鲁棒和准确的结果。
*   **降低多视图任务的门槛：** 通过直接增强现有成熟的基础模型，降低了开发高性能多视图应用的门槛，无需从头开始设计专门的多视图模型。
*   **推动3D感知研究：** 尽管避免了显式3D建模，但该方法通过隐式3D感知来提升特征匹配，这可能为未来更高效、更通用的3D感知方法提供新的思路。
*   **通用性：** 该方法可以应用于多种Transformer-based基础模型，具有良好的通用性，有望成为多视图计算机视觉领域的一个标准化增强技术。

**4. 可能受益的相关领域或应用**

*   **3D重建与场景理解：** 更一致的多视图特征可以显著提升多视图立体（MVS）和场景流（Scene Flow）等任务的准确性。
*   **机器人导航与感知：** 机器人通常依赖于多个传感器（如多个摄像头）获取信息，一致的多视图特征对于路径规划、障碍物检测和环境建图至关重要。
*   **增强现实（AR）与虚拟现实（VR）：** 在AR/VR应用中，准确地对齐和融合来自不同视角的真实世界信息是实现沉浸式体验的关键。
*   **自动驾驶：** 自动驾驶车辆需要处理来自多个摄像头和传感器的信息，以全面感知周围环境，一致的多视图特征有助于提高感知系统的鲁棒性。
*   **多视角图像检索与匹配：** 能够更准确地匹配不同视角下的相同物体或场景。
*   **物体识别与分割：** 在遮挡或视角变化较大的情况下，多视图信息可以提供更全面的物体信息，提高识别和分割的准确性。
*   **医学影像分析：** 在某些医学成像技术中，会获取同一病灶的多个视图，一致的特征表示有助于更精确的诊断和分析。

**5. 从摘要中可以推断出的局限性**

*   **对初始单视图基础模型的依赖：** 该方法的性能在很大程度上取决于底层单视图基础模型的质量和能力。如果基础模型本身存在缺陷，其多视图扩展也可能受到限制。
*   **“3D-aware attention layers”的实现细节：** 摘要中并未详细说明这些注意力层的具体结构和训练方式。其有效性可能取决于这些层的设计是否能够有效地捕捉跨视图的几何关系。
*   **计算开销的增加：** 引入额外的注意力层可能会增加模型的计算复杂度和推理时间，尽管论文声称避免了显式3D建模，但计算量的增加仍是潜在的考虑因素。
*   **对相机姿态的隐式或显式要求：** 虽然论文提到“bypasses the need to build a consistent 3D model”，但为了有效地匹配特征，模型可能仍然需要某种形式的相机姿态信息（即使是隐式的学习）来理解视图之间的相对关系。如果相机姿态信息不可用或不准确，效果可能会打折扣。
*   **“as consistent as possible”的度量：** 摘要中提到“as consistent as possible”，这表明特征一致性可能是一个优化目标，但达到完全一致可能仍然是一个挑战，其“一致性”的程度和评估标准需要进一步的实验验证。
*   **实验验证的范围：** 摘要中提到了“surface normal estimation and multi-view segmentation tasks”作为示例，但其在更广泛的多视图任务上的泛化能力仍需进一步验证。

总而言之，这篇论文提出了一种非常有前景的方法，通过引入“3D-aware attention layers”来增强现有基础模型的多视图理解能力，有望在多个计算机视觉领域带来显著的性能提升，尤其是在处理真实世界中常见的、具有多视角信息的场景时。其最大的亮点在于在不进行显式3D重建的前提下，实现了跨视图特征的有效对齐和一致性。

**Key Findings:**

- We propose a way to convert a Foundation Model into a Multi-View Foundation Model.
- Specifically, we show how to augment Transformers-based foundation models (i.e., DINO, SAM, CLIP) with intermediate 3D-aware attention layers that help match features across different views.
- As leading examples, we show surface normal estimation and multi-view segmentation tasks.
- Quantitative experiments show that our method improves feature matching considerably compared to current foundation models.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15708v1)
- [arXiv](https://arxiv.org/abs/2512.15708v1)

---

<a id='2512.15707v1'></a>
## [GateFusion: Hierarchical Gated Cross-Modal Fusion for Active Speaker Detection](https://arxiv.org/abs/2512.15707v1)

**Authors:** Yu Wang, Juhyung Ha, Frangil M. Ramirez, Yuchen Wang, David J. Crandall

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

Active Speaker Detection (ASD) aims to identify who is currently speaking in each frame of a video. Most state-of-the-art approaches rely on late fusion to combine visual and audio features, but late fusion often fails to capture fine-grained cross-modal interactions, which can be critical for robust performance in unconstrained scenarios. In this paper, we introduce GateFusion, a novel architecture that combines strong pretrained unimodal encoders with a Hierarchical Gated Fusion Decoder (HiGate). HiGate enables progressive, multi-depth fusion by adaptively injecting contextual features from one modality into the other at multiple layers of the Transformer backbone, guided by learnable, bimodally-conditioned gates. To further strengthen multimodal learning, we propose two auxiliary objectives: Masked Alignment Loss (MAL) to align unimodal outputs with multimodal predictions, and Over-Positive Penalty (OPP) to suppress spurious video-only activations. GateFusion establishes new state-of-the-art results on several challenging ASD benchmarks, achieving 77.8% mAP (+9.4%), 86.1% mAP (+2.9%), and 96.1% mAP (+0.5%) on Ego4D-ASD, UniTalk, and WASD benchmarks, respectively, and delivering competitive performance on AVA-ActiveSpeaker. Out-of-domain experiments demonstrate the generalization of our model, while comprehensive ablations show the complementary benefits of each component.

**Analysis:**

好的，这是对论文“GateFusion: Hierarchical Gated Cross-Modal Fusion for Active Speaker Detection”的全面中文摘要：

**论文题目：** GateFusion: Hierarchical Gated Cross-Modal Fusion for Active Speaker Detection (GateFusion：用于主动说话人检测的分层门控跨模态融合)

**作者：** Yu Wang, Juhyung Ha, Frangil M. Ramirez, Yuchen Wang, David J. Crandall

**摘要：**

**1. 研究问题/核心挑战：**
主动说话人检测（ASD）旨在识别视频中每一帧正在说话的人。现有的大多数先进方法依赖于“晚期融合”（late fusion），即将视觉和音频特征在最后阶段进行合并。然而，这种方法往往难以捕捉细粒度的跨模态交互，而这种交互对于在非约束场景下实现鲁棒的性能至关重要。当一种模态信息不足或模糊时（例如，说话人不可见，或多人声音相似），晚期融合的局限性尤为明显。

**2. 主要创新点/方法贡献：**
为了解决上述问题，本文提出了**GateFusion**，一种新颖的架构，其核心创新在于：

*   **分层门控融合解码器 (HiGate)：** GateFusion引入了一个名为HiGate的解码器，它能够实现渐进式的、多深度的跨模态融合。HiGate通过在Transformer骨干网络的多个层级，自适应地将一种模态的上下文特征注入到另一种模态中来实现融合。这种注入过程由可学习的、双模态条件控制的门控机制（gates）引导，从而实现细粒度的、上下文感知的跨模态交互。HiGate的设计是对称的，允许音频和视觉模态互相扮演主导或上下文角色，支持灵活的双向信息流。
*   **两个辅助训练目标：**
    *   **掩码对齐损失 (MAL)：** 旨在使单模态的预测结果与多模态的预测结果对齐，但仅限于有主动说话人的帧。这有助于增强单模态分支在处理模糊或有噪声情况下的可靠性，并鼓励它们学习跨模态依赖关系。
    *   **过正惩罚 (OPP)：** 专门用于抑制视频分支产生的过多的假阳性激活，尤其是在视觉信息模糊或退化的场景下。这有助于提高模型在视觉信息不确定时的鲁棒性。

**3. 主要结果与意义：**
GateFusion在多个具有挑战性的ASD基准测试中取得了最先进（state-of-the-art）的性能，具体表现为：
*   在Ego4D-ASD上达到 **77.8% mAP**（比之前最佳方法提高9.4%）。
*   在UniTalk上达到 **86.1% mAP**（比之前最佳方法提高2.9%）。
*   在WASD上达到 **96.1% mAP**（比之前最佳方法提高0.5%）。
*   在AVA-ActiveSpeaker上也取得了具有竞争力的性能。

此外，论文进行了跨领域（out-of-domain）实验，证明了GateFusion在不同数据集上的**泛化能力**。消融实验也充分展示了HiGate、MAL和OPP等组件的**互补效益**，共同提升了模型的性能和鲁棒性。这些结果表明，GateFusion在处理非约束和具有挑战性的音频-视觉场景方面具有显著优势。

**4. 提及的局限性：**
论文中并未明确列出具体的局限性，但从其研究方向和提出的方法来看，可以推断出一些潜在的方面：
*   **计算成本：** 尽管论文在效率分析中展示了GateFusion相比于某些基线模型（如LoCoNet）具有更高的效率，但分层融合和门控机制本身可能比简单的晚期融合方法有更高的计算开销。
*   **超参数敏感性：** 辅助损失的权重（λMAL和λOPP）需要仔细调整，以平衡它们与主分类损失的作用。
*   **对预训练模型的依赖：** GateFusion依赖于强大的预训练单模态编码器，其性能在一定程度上受限于这些编码器的能力。

**5. 潜在的未来研究方向：**
基于论文的研究内容，可以推测以下潜在的未来研究方向：
*   **更精细的门控机制：** 探索更复杂的门控机制，以更精细地控制跨模态信息的注入和交互。
*   **多说话人场景的扩展：** 虽然论文提到了处理多人场景的挑战，但进一步优化模型以在更复杂的、多人同时说话的场景下实现更精确的检测和区分。
*   **实时性优化：** 进一步探索模型压缩和加速技术，以满足更严格的实时性要求。
*   **跨模态信息互补的深入研究：** 探索如何更有效地利用一种模态的缺失信息来推断另一种模态，或者在模态信息质量差异较大的情况下进行更鲁棒的融合。
*   **应用于其他跨模态任务：** 将GateFusion的层级门控融合思想应用于其他需要细粒度跨模态交互的任务，如视频字幕生成、情感识别等。

总而言之，GateFusion通过引入创新的HiGate分层门控融合机制和有效的辅助训练目标，显著提升了主动说话人检测的性能和鲁棒性，尤其是在非约束和具有挑战性的场景下，为跨模态融合在计算机视觉领域的研究提供了新的思路和方法。

**Key Findings:**

- Most state-of-the-art approaches rely on late fusion to combine visual and audio features, but late fusion often fails to capture fine-grained cross-modal interactions, which can be critical for robust performance in unconstrained scenarios.
- In this paper, we introduce GateFusion, a novel architecture that combines strong pretrained unimodal encoders with a Hierarchical Gated Fusion Decoder (HiGate).
- To further strengthen multimodal learning, we propose two auxiliary objectives: Masked Alignment Loss (MAL) to align unimodal outputs with multimodal predictions, and Over-Positive Penalty (OPP) to suppress spurious video-only activations.
- GateFusion establishes new state-of-the-art results on several challenging ASD benchmarks, achieving 77.8% mAP (+9.4%), 86.1% mAP (+2.9%), and 96.1% mAP (+0.5%) on Ego4D-ASD, UniTalk, and WASD benchmarks, respectively, and delivering competitive performance on AVA-ActiveSpeaker.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15707v1)
- [arXiv](https://arxiv.org/abs/2512.15707v1)

---

<a id='2512.15702v1'></a>
## [End-to-End Training for Autoregressive Video Diffusion via Self-Resampling](https://arxiv.org/abs/2512.15702v1)

**Authors:** Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, Dahua Lin

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

Autoregressive video diffusion models hold promise for world simulation but are vulnerable to exposure bias arising from the train-test mismatch. While recent works address this via post-training, they typically rely on a bidirectional teacher model or online discriminator. To achieve an end-to-end solution, we introduce Resampling Forcing, a teacher-free framework that enables training autoregressive video models from scratch and at scale. Central to our approach is a self-resampling scheme that simulates inference-time model errors on history frames during training. Conditioned on these degraded histories, a sparse causal mask enforces temporal causality while enabling parallel training with frame-level diffusion loss. To facilitate efficient long-horizon generation, we further introduce history routing, a parameter-free mechanism that dynamically retrieves the top-k most relevant history frames for each query. Experiments demonstrate that our approach achieves performance comparable to distillation-based baselines while exhibiting superior temporal consistency on longer videos owing to native-length training.

**Analysis:**

好的，这是对论文“End-to-End Training for Autoregressive Video Diffusion via Self-Resampling”的全面中文摘要：

**论文题目：** End-to-End Training for Autoregressive Video Diffusion via Self-Resampling

**作者：** Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, Dahua Lin

**摘要：**

这篇论文提出了一种名为 **Resampling Forcing** 的新颖框架，旨在解决自回归视频扩散模型在训练过程中面临的 **曝光偏差（exposure bias）** 问题，并实现端到端的训练。

**1. 研究问题/核心挑战：**

自回归视频扩散模型在模拟世界和预测未来状态方面展现出巨大潜力，但其训练方式（教师强制，teacher forcing）与推理方式存在不匹配。在训练时，模型依赖于真实的、无误的历史帧进行预测；而在推理时，模型必须依赖于自身生成的、可能包含误差的历史帧。这种“训练-测试不匹配”会导致模型预测误差的累积，尤其是在生成长视频时，可能导致视频质量的灾难性下降（即视频崩溃）。现有方法通常采用训练后蒸馏或引入在线判别器来缓解此问题，但这些方法难以实现从头开始的大规模端到端训练，且可能引入额外的复杂性或泄露未来信息。

**2. 关键创新/方法贡献：**

*   **Resampling Forcing 框架：** 提出了一种教师无关（teacher-free）的端到端训练框架，允许自回归视频扩散模型从零开始进行大规模训练。
*   **自重采样（Self-Resampling）机制：** 这是该方法的核心。在训练过程中，该机制模拟了推理时模型可能产生的误差。具体而言，它通过在真实历史帧上引入一定程度的噪声（模拟模型预测误差），然后使用在线模型权重来完成剩余的去噪步骤，从而生成一个包含模型误差的“退化”历史帧。模型随后基于这些退化历史帧来预测下一帧。这种方式迫使模型学习在不完美的输入下保持鲁棒性，从而缓解误差累积。
*   **并行训练与因果掩码：** 结合退化历史帧和因果掩码（sparse causal mask），实现了并行训练，并利用逐帧扩散损失（frame-level diffusion loss）来保证时间因果性。
*   **历史路由（History Routing）机制：** 为了解决长视频生成中注意力机制复杂度随历史帧数增长而急剧增加的问题，论文引入了一个参数无关（parameter-free）的动态历史路由机制。该机制能够为每个查询（query）动态地检索最相关的 top-k 个历史帧进行注意力计算，从而将注意力复杂度维持在近乎恒定的水平，有效支持长时序生成。

**3. 主要结果与意义：**

*   **性能媲美蒸馏基线：** 实验结果表明，Resampling Forcing 在生成质量上与基于蒸馏的先进基线模型相当。
*   **长视频生成优势：** 由于采用了原生长视频训练（native-length training），该方法在生成长视频时表现出优越的时间一致性，显著优于那些通过截断或外插长视频的基线方法。
*   **严格的时间因果性：** 与蒸馏基线相比，该模型更严格地遵守了因果依赖关系。
*   **高效的长时序上下文：** 历史路由机制在保持可忽略的质量损失的情况下，实现了稀疏上下文，为长时序生成提供了可行的内存设计方案。
*   **意义：** 该工作为自回归视频扩散模型的端到端、可扩展训练提供了新的途径，并解决了长时序视频生成中的关键挑战，有望推动未来视频世界模型的进步。

**4. 论文中提到的局限性：**

*   **推理速度：** 作为一种扩散模型，其推理过程需要迭代去噪步骤，实时性可能需要后处理加速（如少步蒸馏）或改进采样器。
*   **训练效率：** 训练过程需要同时处理扩散样本和干净历史帧，这可能可以通过架构优化来进一步提高效率。

**5. 潜在的未来研究方向：**

*   **加速推理：** 探索更有效的后处理技术或采样方法来提高推理速度，以实现实时生成。
*   **训练优化：** 研究更优的架构设计，以进一步提高训练效率，可能通过更紧凑地处理扩散样本和干净历史帧。
*   **更长时序的探索：** 进一步探索和优化历史路由机制，以支持更长、更复杂的视频生成任务。
*   **多模态融合：** 将该框架扩展到更广泛的多模态视频生成任务中。

总而言之，这篇论文通过创新的自重采样和历史路由机制，成功地解决了自回归视频扩散模型在训练和长时序生成中的核心挑战，为构建更强大、更可靠的视频生成模型奠定了基础。

**Key Findings:**

- To achieve an end-to-end solution, we introduce Resampling Forcing, a teacher-free framework that enables training autoregressive video models from scratch and at scale.
- Central to our approach is a self-resampling scheme that simulates inference-time model errors on history frames during training.
- Experiments demonstrate that our approach achieves performance comparable to distillation-based baselines while exhibiting superior temporal consistency on longer videos owing to native-length training.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15702v1)
- [arXiv](https://arxiv.org/abs/2512.15702v1)

---

<a id='2512.15701v1'></a>
## [VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression](https://arxiv.org/abs/2512.15701v1)

**Authors:** Kyle Sargent, Ruiqi Gao, Philipp Henzler, Charles Herrmann, Aleksander Holynski, Li Fei-Fei, Jiajun Wu, Jason Zhang

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

Evaluations of image compression performance which include human preferences have generally found that naive distortion functions such as MSE are insufficiently aligned to human perception. In order to align compression models to human perception, prior work has employed differentiable perceptual losses consisting of neural networks calibrated on large-scale datasets of human psycho-visual judgments. We show that, surprisingly, state-of-the-art vision-language models (VLMs) can replicate binary human two-alternative forced choice (2AFC) judgments zero-shot when asked to reason about the differences between pairs of images. Motivated to exploit the powerful zero-shot visual reasoning capabilities of VLMs, we propose Vision-Language Models for Image Compression (VLIC), a diffusion-based image compression system designed to be post-trained with binary VLM judgments. VLIC leverages existing techniques for diffusion model post-training with preferences, rather than distilling the VLM judgments into a separate perceptual loss network. We show that calibrating this system on VLM judgments produces competitive or state-of-the-art performance on human-aligned visual compression depending on the dataset, according to perceptual metrics and large-scale user studies. We additionally conduct an extensive analysis of the VLM-based reward design and training procedure and share important insights. More visuals are available at https://kylesargent.github.io/vlic

**Analysis:**

好的，这是对论文“VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression”的全面中文摘要，重点突出其在计算机视觉领域的创新性和重要性：

**论文题目：** VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression

**作者：** Kyle Sargent, Ruiqi Gao, Philipp Henzler, Charles Herrmann, Aleksander Hołyński, Li Fei-Fei, Jiajun Wu, Jason Zhang

**摘要：**

**1. 研究问题/核心挑战：**
传统的图像压缩模型在评估图像质量时，往往依赖于均方误差（MSE）等失真度量，这些度量与人类的感知判断存在显著差距。为了使图像压缩模型更好地对齐人类感知，以往的研究通常依赖于使用大量人类心理视觉判断数据训练的可微分感知损失网络。然而，收集这些数据成本高昂，且训练出的感知模型可能存在泛化性问题。本研究的核心问题是如何在不依赖大量人工标注数据的情况下，构建一个能够准确评估图像感知质量并用于指导图像压缩模型训练的系统。

**2. 主要创新点/方法贡献：**
本文提出了一种名为 **VLIC (Vision-Language Models for Image Compression)** 的新颖图像压缩系统，其核心创新在于：

*   **利用视觉语言模型（VLMs）作为零样本感知裁判：** 作者发现，现成的先进VLMs（如Gemini 2.5-Flash）能够零样本（zero-shot）地准确复制人类在二选一强制选择（2AFC）任务中的视觉相似性判断。这意味着VLMs具备了强大的、无需额外训练的感知理解能力。
*   **基于VLM偏好进行扩散模型后训练：** VLIC系统将VLMs的判断能力直接应用于扩散模型（Diffusion Model）的后训练阶段。具体而言，利用VLM对同一潜在编码生成的两个不同重建图像进行排序，并将这种偏好信号通过 **Diffusion DPO (Direct Preference Optimization)** 技术来优化扩散模型，使其生成更符合人类感知的图像。这种方法避免了将VLM的判断蒸馏成一个独立的感知损失网络，而是直接利用了VLM的推理能力。
*   **VLM与LPIPS的集成：** 为了进一步提高奖励信号的鲁棒性和一致性，VLIC将VLM的偏好与传统的感知度量LPIPS（Learned Perceptual Image Patch Similarity）相结合。只有当两者达成一致时，才使用该偏好对进行训练，这有效减少了VLM可能产生的噪声和幻觉。
*   **改进的VLM奖励设计：** 为了提高VLM判断的可靠性，作者提出了一系列策略，包括：对同一对图像进行双向评估（反转顺序），对多个随机种子下的VLM判断进行自集成（self-ensembling），以及与LPIPS进行一致性检查。

**3. 主要结果与意义：**
VLIC在多个标准图像压缩数据集上取得了令人鼓舞的结果：

*   **竞争性或领先的性能：** VLIC在人类感知对齐的评估指标上，根据数据集的不同，达到了与现有最先进（state-of-the-art）的图像压缩方法相当甚至更好的性能。
*   **在MS-COCO数据集上的优势：** VLIC在包含大量人脸、文本等人类敏感特征的MS-COCO数据集上表现尤为出色，这表明其对人类感知细节的捕捉能力更强。
*   **验证了VLM作为感知裁判的潜力：** 研究结果有力地证明了VLMs作为零样本感知裁判的有效性，为未来图像压缩领域的研究提供了一种新的范式，即利用大型多模态模型来指导感知优化。
*   **对VLM后训练的深入分析：** 文章还提供了关于VLM奖励设计和训练程序的详细分析，为未来使用VLMs进行模型对齐提供了宝贵的实践指导。

**4. 提及的局限性：**
*   **扩散模型的固有延迟：** 与GANs等方法相比，扩散模型在推理时存在一定的延迟，尽管这一点在其他基于扩散的模型中也存在。
*   **VLM奖励计算成本：** 使用VLM作为奖励函数比使用小型感知网络计算成本更高。
*   **VLM的幻觉和不一致性：** 尽管采取了多种缓解措施，但VLMs在处理高度相似的图像时仍可能出现不一致或错误的判断（如图6所示）。
*   **对低比特率下PSNR的牺牲：** VLIC为了更好地对齐人类感知，可能会在像素级度量（如PSNR）上有所牺牲，这在固定比特率下是权衡的结果。

**5. 潜在的未来研究方向：**
*   **利用更强大的VLMs：** 随着VLMs的不断发展和改进，其零样本感知能力将进一步增强，有望为VLIC带来更优的性能。
*   **探索更广泛的VLM应用：** VLMs的感知能力不仅限于图像质量评估，还可以用于指导图像编辑、风格迁移等更复杂的任务。
*   **降低VLM推理成本：** 研究更高效的VLM推理方法或模型压缩技术，以降低VLIC的计算开销。
*   **更精细的奖励设计：** 进一步探索如何设计更鲁棒、更具区分度的VLM奖励信号，以应对各种复杂的图像内容和失真类型。
*   **多模态信息融合：** 探索将文本描述等其他模态信息与图像内容结合，以实现更全面的人类感知对齐。

**总结：**

这篇论文的核心贡献在于开创性地将大型视觉语言模型（VLMs）引入图像压缩领域，并成功地利用其强大的零样本感知理解能力，通过Diffusion DPO技术对扩散模型进行后训练，构建了VLIC系统。VLIC在不依赖大量人工标注数据的情况下，实现了与现有最先进方法相当甚至更优的人类感知对齐性能，尤其在包含人脸和文本等敏感特征的图像上表现突出。这项工作不仅为图像压缩领域提供了一种新颖且高效的训练范式，也为未来利用大型多模态模型指导计算机视觉任务的研究开辟了新的道路。

**Key Findings:**

- We show that, surprisingly, state-of-the-art vision-language models (VLMs) can replicate binary human two-alternative forced choice (2AFC) judgments zero-shot when asked to reason about the differences between pairs of images.
- Motivated to exploit the powerful zero-shot visual reasoning capabilities of VLMs, we propose Vision-Language Models for Image Compression (VLIC), a diffusion-based image compression system designed to be post-trained with binary VLM judgments.
- We show that calibrating this system on VLM judgments produces competitive or state-of-the-art performance on human-aligned visual compression depending on the dataset, according to perceptual metrics and large-scale user studies.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15701v1)
- [arXiv](https://arxiv.org/abs/2512.15701v1)

---

<a id='2512.15693v1'></a>
## [Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning](https://arxiv.org/abs/2512.15693v1)

**Authors:** Yifei Li, Wenzhao Zheng, Yanran Zhang, Runze Sun, Yu Zheng, Lei Chen, Jie Zhou, Jiwen Lu

**Published:** 2025-12-17

**Categories:** cs.CV

**Abstract:**

The misuse of AI-driven video generation technologies has raised serious social concerns, highlighting the urgent need for reliable AI-generated video detectors. However, most existing methods are limited to binary classification and lack the necessary explanations for human interpretation. In this paper, we present Skyra, a specialized multimodal large language model (MLLM) that identifies human-perceivable visual artifacts in AI-generated videos and leverages them as grounded evidence for both detection and explanation. To support this objective, we construct ViF-CoT-4K for Supervised Fine-Tuning (SFT), which represents the first large-scale AI-generated video artifact dataset with fine-grained human annotations. We then develop a two-stage training strategy that systematically enhances our model's spatio-temporal artifact perception, explanation capability, and detection accuracy. To comprehensively evaluate Skyra, we introduce ViF-Bench, a benchmark comprising 3K high-quality samples generated by over ten state-of-the-art video generators. Extensive experiments demonstrate that Skyra surpasses existing methods across multiple benchmarks, while our evaluation yields valuable insights for advancing explainable AI-generated video detection.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning**

**1. 论文的主要贡献（2-3句话）：**

本研究提出了Skyra，一个专门用于检测AI生成视频的多模态大型语言模型（MLLM）。Skyra的核心贡献在于其能够识别并利用人类可感知的视觉伪影作为检测和解释的依据，从而克服了现有方法仅限于二元分类且缺乏可解释性的局限。为此，论文构建了首个大规模AI生成视频伪影数据集ViF-CoT-4K，并提出了一种两阶段训练策略来提升模型的时空伪影感知、解释能力和检测准确性。

**2. 关键创新或方法论：**

*   **多模态大型语言模型（MLLM）的应用：** 将MLLM的能力引入AI生成视频检测领域，使其能够同时处理视觉信息和语言解释。
*   **基于“接地伪影推理”（Grounded Artifact Reasoning）的检测范式：** 这是本研究最核心的创新点。Skyra不是简单地进行二元分类，而是主动寻找并分析AI生成视频中存在的、人类肉眼可见的视觉伪影（如不自然的纹理、运动不连贯、物体变形等）。这些伪影被视为“接地证据”，用于支持检测结果，并能生成可解释的说明。
*   **首个大规模AI生成视频伪影数据集（ViF-CoT-4K）：** 专门为训练模型识别和理解伪影而构建，包含细粒度的人类标注，这对于监督微调（SFT）至关重要。
*   **两阶段训练策略：** 旨在系统性地提升模型在时空伪影感知、解释能力和最终检测准确性方面的表现。这可能意味着模型在不同阶段侧重于不同的学习目标。
*   **专门的评估基准（ViF-Bench）：** 包含来自十余个SOTA视频生成器的3K高质量样本，为全面评估Skyra的性能提供了标准。

**3. 对该领域的潜在影响：**

*   **提升AI生成视频检测的可靠性和可信度：** 通过提供可解释的检测结果，Skyra有望增强用户对检测结果的信任，并帮助区分真实视频和虚假视频。
*   **推动可解释AI（XAI）在多媒体安全领域的应用：** 本研究展示了如何将XAI的理念融入到AI生成内容检测中，为其他内容安全问题提供借鉴。
*   **为AI生成视频的伦理和法律监管提供技术支持：** 准确且可解释的检测工具对于打击虚假信息、保护个人隐私以及制定相关法规至关重要。
*   **促进AI视频生成技术的发展：** 通过揭示当前生成技术中存在的伪影，可以为研究人员提供改进生成模型的方向。

**4. 可能受益的相关领域或应用：**

*   **媒体内容审核与事实核查：** 社交媒体平台、新闻机构等可以利用Skyra来识别和标记AI生成的虚假视频内容。
*   **数字取证：** 在法律和调查领域，Skyra可以帮助识别视频证据的真实性。
*   **内容创作者和平台：** 帮助内容创作者理解其生成内容的潜在问题，并为平台提供内容安全工具。
*   **网络安全：** 防范利用AI生成视频进行的欺诈、诽谤或政治操纵。
*   **教育和研究：** 为AI生成内容检测和可解释AI的研究提供新的工具和数据集。

**5. 从摘要中可以推断出的局限性：**

*   **对“人类可感知”伪影的依赖：** 尽管这是其优势，但也意味着如果AI生成技术发展到能够完全消除人类肉眼难以察觉的伪影，Skyra的有效性可能会受到影响。
*   **数据集和基准的覆盖范围：** ViF-CoT-4K和ViF-Bench虽然规模较大，但可能仍无法覆盖所有现有的AI视频生成技术和可能出现的伪影类型。随着新生成模型的出现，数据集和基准可能需要不断更新。
*   **计算资源需求：** 作为基于MLLM的模型，Skyra的训练和部署可能需要大量的计算资源。
*   **解释的深度和准确性：** 虽然论文强调了解释能力，但解释的详细程度、准确性以及是否能被所有用户理解，仍需在实际应用中进一步验证。
*   **泛化能力：** 模型在未见过的新型AI生成视频上的泛化能力，以及对不同领域（如电影、纪录片、个人Vlog等）视频的适应性，是需要进一步考察的。

**总结：**

Skyra这篇论文在AI生成视频检测领域提出了一个非常有前景的新方向。通过将MLLM的能力与对视觉伪影的“接地推理”相结合，并辅以高质量的数据集和评估基准，该研究有望显著提升AI生成视频检测的准确性和可解释性。这对于应对日益严峻的AI生成内容滥用问题具有重要的理论和实践意义。其核心创新在于从“黑盒”分类转向“白盒”的伪影分析和解释，这正是当前AI安全和可信AI领域所急需的突破。

**Key Findings:**

- In this paper, we present Skyra, a specialized multimodal large language model (MLLM) that identifies human-perceivable visual artifacts in AI-generated videos and leverages them as grounded evidence for both detection and explanation.
- To comprehensively evaluate Skyra, we introduce ViF-Bench, a benchmark comprising 3K high-quality samples generated by over ten state-of-the-art video generators.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15693v1)
- [arXiv](https://arxiv.org/abs/2512.15693v1)

---

<a id='2512.15692v1'></a>
## [mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs](https://arxiv.org/abs/2512.15692v1)

**Authors:** Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, Elvis Nava

**Published:** 2025-12-17

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

Prevailing Vision-Language-Action Models (VLAs) for robotic manipulation are built upon vision-language backbones pretrained on large-scale, but disconnected static web data. As a result, despite improved semantic generalization, the policy must implicitly infer complex physical dynamics and temporal dependencies solely from robot trajectories. This reliance creates an unsustainable data burden, necessitating continuous, large-scale expert data collection to compensate for the lack of innate physical understanding. We contend that while vision-language pretraining effectively captures semantic priors, it remains blind to physical causality. A more effective paradigm leverages video to jointly capture semantics and visual dynamics during pretraining, thereby isolating the remaining task of low-level control. To this end, we introduce \model, a novel Video-Action Model (VAM) that pairs a pretrained Internet-scale video model with a flow matching-based action decoder conditioned on its latent representations. The decoder serves as an Inverse Dynamics Model (IDM), generating low-level robot actions from the latent representation of video-space action plans. Our extensive evaluation shows that our approach achieves state-of-the-art performance on simulated and real-world robotic manipulation tasks, improving sample efficiency by 10x and convergence speed by 2x compared to traditional VLA architectures.

**Analysis:**

好的，这是一篇关于“mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs”的全面中文摘要，重点突出了其新颖性和重要性：

**论文摘要：mimic-video：超越VLA的通用机器人控制视频-动作模型**

**1. 研究问题/核心挑战：**

当前主流的视觉-语言-动作（VLA）模型在机器人操控领域取得了显著进展，但它们主要依赖于在海量但静态的互联网图像-文本数据上预训练的视觉-语言模型（VLMs）。这种范式导致模型虽然在语义理解上表现良好，但必须从稀疏且昂贵的机器人轨迹数据中隐式地学习复杂的物理动力学和时间依赖关系。这带来了巨大的数据负担，需要持续收集大量专家演示数据来弥补模型对物理因果关系的先天理解不足。论文的核心问题在于：如何更有效地利用预训练模型来学习机器人控制，特别是如何解决当前VLA模型在物理理解和数据效率上的瓶颈。

**2. 关键创新/方法贡献：**

本文提出了一种名为 **mimic-video** 的新型 **视频-动作模型（VAM）**，其核心创新在于：

*   **利用视频预训练的物理动力学先验：** mimic-video 摒弃了仅依赖静态图像-文本的VLM，而是利用预训练的互联网规模视频模型。视频数据天然地包含了“事物如何发生”的动态信息，能够捕捉物体运动、形变和交互的物理过程。
*   **解耦规划与控制：** mimic-video 将视频模型的长时序规划能力（生成部分去噪的视频“视觉计划”）与一个轻量级的动作解码器（作为逆动力学模型，IDM）相结合。视频模型负责生成未来场景的潜在表示，而动作解码器则专注于将这些潜在表示转化为低维度的机器人动作。这种解耦使得动作解码器无需从头学习复杂的未来分布，而能专注于更简单的逆动力学问题。
*   **部分去噪策略：** 在推理阶段，mimic-video 采用一种“部分去噪”策略，即仅对视频模型生成的中间状态进行去噪，而不是完全去噪到最终的清晰视频。这种策略在实验中被证明能带来更好的策略性能，并显著加速推理速度，因为它只需要视频模型的一次前向传播。
*   **条件流匹配（CFM）框架：** 论文利用条件流匹配（CFM）框架来训练视频模型和动作解码器，这是一种用于学习数据分布的有效方法。

**3. 主要结果与意义：**

*   **状态-艺术性能：** mimic-video 在模拟和真实世界的机器人操控任务上取得了最先进的性能，包括精细的操纵任务。
*   **显著提升数据效率：** 与传统的VLA模型相比，mimic-video 将样本效率提高了 **10倍**。这意味着模型可以用更少的数据来达到相似或更好的性能。
*   **加速收敛速度：** 动作解码器的训练速度提高了 **2倍**，这得益于视频模型提供的更丰富、更具物理意义的先验信息。
*   **泛化能力：** mimic-video 在多种机器人实体（从单臂到双臂、模拟到真实世界）上展现了良好的泛化能力。
*   **对视频质量的洞察：** 研究发现，过高的视频保真度（完全去噪）反而可能导致性能下降，因为不完美的视频生成可能引入与训练数据分布不符的噪声，而适度的“噪声”可以作为一种有效的训练和测试时增强。

**4. 论文提及的局限性：**

*   **单视角视频模型：** 当前模型依赖于单视角视频模型，这限制了其在需要多视角理解的任务中的空间推理和遮挡鲁棒性。
*   **未实现统一的跨实体模型：** 论文尚未将VAM范式应用于训练一个统一的、大规模的、跨实体的模型，这被认为是解锁视频基础模型全部泛化能力的关键一步。
*   **真实世界实验任务有限：** 当前的真实世界实验仅限于一组特定的任务，未来需要扩展到更多样化的操纵行为。

**5. 未来研究方向：**

*   **探索多视角视频模型：** 集成多视角视频模型以增强空间推理和遮挡鲁棒性。
*   **构建统一的跨实体VAM模型：** 训练一个能够处理多种机器人实体和任务的通用VAM模型。
*   **扩展到更广泛的真实世界任务：** 将该方法应用于更广泛、更复杂的机器人操纵场景。
*   **进一步优化推理效率：** 探索更高效的视频生成和动作解码策略，以实现实时控制。

**总结：**

mimic-video 是一项重要的研究工作，它成功地将预训练视频模型的丰富物理动力学先验引入机器人控制领域，显著克服了现有VLA模型在数据效率和物理理解上的瓶颈。通过解耦规划与控制以及创新的部分去噪策略，mimic-video 在多种机器人任务上取得了优异的性能，并为未来更通用、更高效的机器人学习奠定了基础。这项工作强调了视频数据在机器人控制中的关键作用，并为如何有效利用这些数据提供了新的视角。

**Key Findings:**

- To this end, we introduce \model, a novel Video-Action Model (VAM) that pairs a pretrained Internet-scale video model with a flow matching-based action decoder conditioned on its latent representations.
- Our extensive evaluation shows that our approach achieves state-of-the-art performance on simulated and real-world robotic manipulation tasks, improving sample efficiency by 10x and convergence speed by 2x compared to traditional VLA architectures.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.15692v1)
- [arXiv](https://arxiv.org/abs/2512.15692v1)

---

