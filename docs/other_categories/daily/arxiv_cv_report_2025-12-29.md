time: 20251229

# Arxiv Computer Vision Papers - 2025-12-29

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份关于2025年12月26日 Arxiv 计算机视觉领域论文的简明执行摘要。

---

**执行摘要：2025年12月26日 Arxiv 计算机视觉论文精选**

**主要主题与趋势：**

本期论文集聚焦于**多模态理解与生成**、**基础模型（Foundation Models）的实用化与安全性**，以及**高效的交互式视觉智能体**。特别值得注意的是，研究人员正积极探索如何使大型视觉模型在真实世界场景中更具鲁棒性、可控性和交互性，同时关注其潜在的安全漏洞。

**亮点与创新：**

*   **真实世界中心的基础GUI智能体：** MAI-UI 和 iSHIFT 两篇论文（1, 9）共同展示了在构建能够理解和操作真实世界图形用户界面（GUI）的智能体方面的进展。MAI-UI 强调“真实世界中心”的设计理念，而 iSHIFT 则提出了轻量级、自适应感知的方法，预示着更实用、更高效的GUI自动化智能体即将到来。
*   **多模态交互与生成能力的提升：** Yume-1.5 (5) 和 StreamAvatar (6) 在文本控制的交互式内容生成方面取得了显著进展。Yume-1.5 实现了文本控制的交互式世界生成，而 StreamAvatar 则专注于实时交互式人类化身的流式扩散模型，这对于虚拟现实、元宇宙等应用具有重要意义。
*   **视觉语言模型（VLMs）的鲁棒性与可编辑性：** ProEdit (3) 和 Look Closer! (10) 均致力于提升VLMs的可控性和准确性。ProEdit 提出了一种基于反演的提示编辑方法，而 Look Closer! 则通过对抗性参数化编辑来缓解幻觉问题，这对于提高VLM的可靠性和用户体验至关重要。

**新兴研究方向与技术：**

*   **双向感知塑造（Bi-directional Perceptual Shaping）：** See Less, See Right (2) 提出的双向感知塑造方法，通过在多模态推理中显式地引导感知过程，为提升模型在复杂推理任务中的表现提供了新思路。
*   **长时域UAV导航的融合：** LongFly (8) 在长时域无人机（UAV）视觉与语言导航方面，通过时空上下文集成，解决了长距离、复杂环境下的导航挑战。
*   **基础模型的后门攻击研究：** Backdoor Attacks on Prompt-Driven Video Segmentation Foundation Models (7) 揭示了基础模型在提示驱动下的潜在安全风险，强调了模型安全性和鲁棒性研究的重要性。
*   **跟踪与检测的关联学习：** Learning Association via Track-Detection Matching (4) 提出了一种新的多目标跟踪方法，通过匹配跟踪与检测来学习关联，有望提升多目标跟踪的精度和效率。

**建议阅读全文的论文：**

考虑到其对当前研究热点和未来方向的潜在影响，以下论文值得深入阅读：

1.  **MAI-UI Technical Report: Real-World Centric Foundation GUI Agents (1)**：对于希望构建实际应用型智能体的研究者，理解其“真实世界中心”的设计理念至关重要。
2.  **See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning (2)**：该方法在多模态推理领域具有创新性，可能为提升模型理解能力提供新的技术路径。
3.  **ProEdit: Inversion-based Editing From Prompts Done Right (3)** 和 **Look Closer! An Adversarial Parametric Editing Framework for Hallucination Mitigation in VLMs (10)**：这两篇论文直接解决了当前VLM应用中的关键痛点——可控性和准确性，对于任何从事VLM研究或应用的人员都极具价值。
4.  **StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars (6)**：在元宇宙和虚拟交互领域具有前瞻性，对实时生成高质量虚拟形象的技术细节感兴趣的研究者不容错过。

---

希望这份摘要能帮助您快速了解近期 Arxiv 计算机视觉领域的最新进展。

---

## Table of Contents

1. [MAI-UI Technical Report: Real-World Centric Foundation GUI Agents](#2512.22047v1)
2. [See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning](#2512.22120v1)
3. [ProEdit: Inversion-based Editing From Prompts Done Right](#2512.22118v1)
4. [Learning Association via Track-Detection Matching for Multi-Object Tracking](#2512.22105v1)
5. [Yume-1.5: A Text-Controlled Interactive World Generation Model](#2512.22096v1)
6. [StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](#2512.22065v1)
7. [Backdoor Attacks on Prompt-Driven Video Segmentation Foundation Models](#2512.22046v1)
8. [LongFly: Long-Horizon UAV Vision-and-Language Navigation with Spatiotemporal Context Integration](#2512.22010v1)
9. [iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception](#2512.22009v1)
10. [Look Closer! An Adversarial Parametric Editing Framework for Hallucination Mitigation in VLMs](#2512.21999v1)

---

## Papers

<a id='2512.22047v1'></a>
## [MAI-UI Technical Report: Real-World Centric Foundation GUI Agents](https://arxiv.org/abs/2512.22047v1)

**Authors:** Hanzhang Zhou, Xu Zhang, Panrong Tong, Jianan Zhang, Liangyu Chen, Quyu Kong, Chenglin Cai, Chen Liu, Yue Wang, Jingren Zhou, Steven Hoi

**Published:** 2025-12-26

**Categories:** cs.CV

**Abstract:**

The development of GUI agents could revolutionize the next generation of human-computer interaction. Motivated by this vision, we present MAI-UI, a family of foundation GUI agents spanning the full spectrum of sizes, including 2B, 8B, 32B, and 235B-A22B variants. We identify four key challenges to realistic deployment: the lack of native agent-user interaction, the limits of UI-only operation, the absence of a practical deployment architecture, and brittleness in dynamic environments. MAI-UI addresses these issues with a unified methodology: a self-evolving data pipeline that expands the navigation data to include user interaction and MCP tool calls, a native device-cloud collaboration system routes execution by task state, and an online RL framework with advanced optimizations to scale parallel environments and context length. MAI-UI establishes new state-of-the-art across GUI grounding and mobile navigation. On grounding benchmarks, it reaches 73.5% on ScreenSpot-Pro, 91.3% on MMBench GUI L2, 70.9% on OSWorld-G, and 49.2% on UI-Vision, surpassing Gemini-3-Pro and Seed1.8 on ScreenSpot-Pro. On mobile GUI navigation, it sets a new SOTA of 76.7% on AndroidWorld, surpassing UI-Tars-2, Gemini-2.5-Pro and Seed1.8. On MobileWorld, MAI-UI obtains 41.7% success rate, significantly outperforming end-to-end GUI models and competitive with Gemini-3-Pro based agentic frameworks. Our online RL experiments show significant gains from scaling parallel environments from 32 to 512 (+5.2 points) and increasing environment step budget from 15 to 50 (+4.3 points). Finally, the native device-cloud collaboration system improves on-device performance by 33%, reduces cloud model calls by over 40%, and preserves user privacy.

**Analysis:**

作为一名计算机视觉和机器学习领域的专家，我对这篇论文摘要进行了深入分析，并为您提炼出以下关键信息：

**1. 论文的主要贡献（2-3句话）：**

该论文提出了 MAI-UI，一个涵盖多种规模（2B 至 235B）的**基础 GUI 智能体家族**，旨在解决现实世界中 GUI 智能体部署的四大挑战。MAI-UI 通过创新的自进化数据流水线、设备-云协同系统和在线强化学习框架，在 GUI 基础（grounding）和移动导航任务上均取得了**新的 SOTA 性能**，显著超越了现有领先模型。

**2. 关键创新或方法论：**

MAI-UI 的核心创新在于其**统一的方法论**，旨在克服现实世界部署的障碍：

*   **自进化数据流水线 (Self-evolving Data Pipeline):** 这是最关键的创新之一。它不仅仅是收集 UI 信息，而是**扩展了导航数据，主动融入了用户交互（user interaction）和多模态指令（MCP tool calls）**。这意味着智能体能够从真实的用户行为和更丰富的指令中学习，从而更贴近实际使用场景。
*   **设备-云协同系统 (Native Device-Cloud Collaboration System):** 这是一个**务实的部署架构**。它能够根据任务状态（task state）智能地路由执行，将计算任务分配到设备端或云端。这不仅提升了**设备端性能（+33%）**，**减少了云端调用（-40%）**，还**保护了用户隐私**。
*   **在线强化学习框架 (Online RL Framework):** 采用**先进的优化技术**来**扩展并行环境和上下文长度**。这使得智能体能够更有效地在动态环境中学习和适应，并处理更复杂的交互序列。

**3. 对该领域的潜在影响：**

*   **推动下一代人机交互 (HCI) 的发展:** MAI-UI 的成功部署将使 GUI 智能体真正成为下一代人机交互的革命性力量，使计算机能够更智能、更自然地理解和操作用户界面。
*   **为通用 GUI 智能体奠定基础:** MAI-UI 的“基础智能体”概念，以及其多尺寸模型家族，预示着未来可能出现能够处理各种 GUI 任务的通用型智能体。
*   **加速智能体在真实世界的应用:** 该研究直接解决了现实世界部署的挑战，如动态环境适应性和效率问题，为将 GUI 智能体推向实际应用铺平了道路。
*   **提升移动端智能体能力:** 在移动 GUI 导航上的 SOTA 表现，将极大地提升移动设备上的自动化和智能化体验。

**4. 可能受益的相关领域或应用：**

*   **自动化测试和质量保证:** 智能体可以模拟用户行为，进行更全面、更高效的 UI 测试。
*   **无障碍技术:** 帮助残障人士更轻松地与数字设备交互。
*   **个性化用户体验:** 根据用户习惯和偏好，智能体可以主动调整界面和操作流程。
*   **智能助手和虚拟代理:** 更强大的 GUI 操作能力将使智能助手能够执行更复杂的任务，如预订机票、填写表格等。
*   **教育和培训:** 智能体可以作为交互式教程，引导用户学习软件操作。
*   **机器人控制:** 将 GUI 操作能力与机器人本体控制相结合，实现更复杂的任务。
*   **软件开发和原型设计:** 辅助开发者进行 UI 设计和功能验证。

**5. 从摘要中可以推断出的局限性：**

*   **“基础智能体”的通用性仍需验证:** 虽然模型涵盖多种尺寸，但其在**跨领域、跨平台 GUI 的泛化能力**仍需在更广泛的场景下进行验证。摘要中提到的 SOTA 表现主要集中在特定的基准测试上。
*   **“自进化”的定义和机制:** 摘要提到了“自进化数据流水线”，但具体的进化机制、如何保证进化过程的稳定性和有效性，以及潜在的“漂移”问题，并未详细说明。
*   **“用户隐私”的实现细节:** 设备-云协同系统声称能保护用户隐私，但具体的隐私保护技术和策略（例如数据匿名化、差分隐私等）并未在摘要中披露。
*   **“动态环境”的定义和挑战:** 摘要提到了“动态环境”的挑战，但具体是哪些类型的动态变化（例如 UI 布局变化、网络延迟、用户输入中断等）以及 MAI-UI 如何应对这些变化，需要更深入的了解。
*   **计算资源和部署成本:** 尽管有设备-云协同，但 235B 的模型规模仍然巨大，其在实际部署中的**计算资源需求和成本**可能是一个挑战。
*   **“MCP tool calls”的含义和范围:** 摘要中提到了“MCP tool calls”，这可能指的是多模态指令（Multimodal Command Processing）或者某种特定的工具调用接口。其具体含义和支持的工具范围会影响智能体的能力。

**总结来说，** MAI-UI 的研究在 GUI 智能体领域具有里程碑式的意义。它不仅在多个关键任务上取得了显著的性能提升，更重要的是，它提出了一套**务实且创新的方法论**，直接解决了 GUI 智能体在现实世界部署的关键瓶颈。其设备-云协同和自进化数据流水线的设计，预示着更智能、更高效、更安全的下一代人机交互体验的到来。

**Key Findings:**

- Motivated by this vision, we present MAI-UI, a family of foundation GUI agents spanning the full spectrum of sizes, including 2B, 8B, 32B, and 235B-A22B variants.
- MAI-UI establishes new state-of-the-art across GUI grounding and mobile navigation.
- On mobile GUI navigation, it sets a new SOTA of 76.7% on AndroidWorld, surpassing UI-Tars-2, Gemini-2.5-Pro and Seed1.8. On MobileWorld, MAI-UI obtains 41.7% success rate, significantly outperforming end-to-end GUI models and competitive with Gemini-3-Pro based agentic frameworks.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22047v1)
- [arXiv](https://arxiv.org/abs/2512.22047v1)

---

<a id='2512.22120v1'></a>
## [See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning](https://arxiv.org/abs/2512.22120v1)

**Authors:** Shuoshuo Zhang, Yizhen Zhang, Jingjing Fu, Lei Song, Jiang Bian, Yujiu Yang, Rui Wang

**Published:** 2025-12-26

**Categories:** cs.CV

**Abstract:**

Large vision-language models (VLMs) often benefit from intermediate visual cues, either injected via external tools or generated as latent visual tokens during reasoning, but these mechanisms still overlook fine-grained visual evidence (e.g., polylines in charts), generalize poorly across domains, and incur high inference-time cost. In this paper, we propose Bi-directional Perceptual Shaping (BiPS), which transforms question-conditioned masked views into bidirectional where-to-look signals that shape perception during training. BiPS first applies a KL-consistency constraint between the original image and an evidence-preserving view that keeps only question-relevant regions, encouraging coarse but complete coverage of supporting pixels. It then applies a KL-separation constraint between the original and an evidence-ablated view where critical pixels are masked so the image no longer supports the original answer, discouraging text-only shortcuts (i.e., answering from text alone) and enforcing fine-grained visual reliance. Across eight benchmarks, BiPS boosts Qwen2.5-VL-7B by 8.2% on average and shows strong out-of-domain generalization to unseen datasets and image types.

**Analysis:**

好的，这是对论文“See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning”的全面中文摘要：

**论文题目：** See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning (少看多看：双向感知塑造用于多模态推理)

**作者：** Shuoshuo Zhang, Yizhen Zhang, Jingjing Fu, Lei Song, Jiang Bian, Yujiu Yang, Rui Wang

**摘要：**

**1. 研究问题/核心挑战：**

该论文旨在解决大型视觉语言模型（VLMs）在多模态推理任务中存在的感知瓶颈。现有方法通常依赖于中间视觉线索（如通过外部工具或生成潜在视觉标记），但这些方法存在以下不足：
*   **忽略精细视觉证据：** 难以捕捉图表中的细线（polylines）等精细视觉细节。
*   **泛化能力差：** 在不同领域和数据集上泛化能力不足。
*   **推理成本高：** 在推理时需要额外的计算步骤，增加了延迟。
*   **易产生文本捷径：** 模型可能过度依赖文本信息，而忽略了视觉证据。

**2. 关键创新/方法贡献：**

作者提出了**双向感知塑造（BiPS）**框架，一种创新的训练时方法，旨在将推理时的中间视觉线索转化为训练信号，从而塑造模型的内部感知策略。BiPS的核心在于利用两种互补的KL（Kullback-Leibler）散度约束：

*   **一致性约束（Consistency Constraint）：** 通过将原始图像与一个**证据保留视图（Evidence-Preserving View）**进行比较，该视图仅保留与问题相关的区域。此约束鼓励模型粗粒度但完整地覆盖支持性像素，确保模型关注到关键信息。
*   **分离约束（Separation Constraint）：** 通过将原始图像与一个**证据消融视图（Evidence-Ablated View）**进行比较，该视图掩盖了关键像素，使得图像无法支持原始答案。此约束旨在阻止模型依赖文本捷径，强制模型依赖精细的视觉信息进行推理。

BiPS采用**粗粒度到细粒度（Coarse-to-Fine）**的两阶段训练课程：
*   **阶段一（一致性阶段）：** 最小化Lcons，主要关注证据定位。
*   **阶段二（分离阶段）：** 最大化Lsep，引入证据消融视图，确保模型推理的视觉基础。

为了生成高质量的训练数据，论文还构建了一个**程序化数据构建流水线**，利用图表渲染代码（如ECD [47]）来精确生成证据保留和证据消融视图，避免了昂贵的人工标注。

**3. 主要结果与意义：**

*   **显著性能提升：** 在八个基准测试中，BiPS将Qwen2.5-VL-7B模型平均提升了8.2%。
*   **强大的跨领域泛化能力：** BiPS在未见过的数据集和图像类型上表现出强大的泛化能力，即使仅使用13K图表样本训练，也能在通用VQA任务上取得显著提升。
*   **数据效率高：** 相比于其他需要大量图表样本的专用模型，BiPS通过增强模型核心视觉感知能力，实现了更高效的数据利用。
*   **克服文本捷径：** 分离约束有效阻止了模型依赖文本信息，确保了推理过程的视觉基础。
*   **推理效率高：** BiPS在训练时引入了感知塑造，而无需在推理时生成额外的视觉线索，因此不会增加推理成本。
*   **案例研究证明：** 在图表理解和视觉计数等任务的案例研究中，BiPS能够生成更具视觉依据的答案，而基线模型则容易依赖统计线索或产生幻觉。

**4. 局限性：**

*   **对数据生成流水线依赖：** BiPS的有效性在一定程度上依赖于程序化数据生成流水线的质量和准确性。
*   **训练课程的敏感性：** 虽然论文提出了粗粒度到细粒度的训练课程，但实验表明，同时优化Lcons和Lsep（联合训练）或反转训练顺序可能会导致性能下降，表明训练课程的设计对最终性能有影响。
*   **超参数敏感性：** 实验表明，过大的KL约束系数（α和β）可能会导致性能下降，说明需要适度的约束来平衡模型性能。

**5. 未来研究方向：**

*   **更通用的数据生成：** 探索更通用的程序化数据生成方法，以覆盖更广泛的视觉推理任务和数据类型。
*   **自适应感知塑造：** 研究如何使感知塑造过程更加自适应，根据任务和数据的特性动态调整。
*   **与其他推理机制的结合：** 探索将BiPS与其他先进的多模态推理机制（如更复杂的注意力机制、知识图谱等）相结合，以进一步提升模型性能。
*   **可解释性增强：** 虽然BiPS提高了视觉依赖性，但进一步研究其内部机制的可解释性，理解模型如何利用塑造后的感知信号进行推理，将是有价值的方向。

**总结：**

该论文提出了一种名为BiPS的双向感知塑造框架，通过在训练阶段引入证据保留和证据消融视图，并利用KL散度约束来塑造模型的视觉感知能力。BiPS有效地解决了现有VLMs在精细视觉证据捕捉、领域泛化和推理成本方面的问题，显著提升了模型在图表理解和多模态推理任务上的性能。其程序化数据生成方法和粗粒度到细粒度的训练课程是关键创新，为提高VLMs的视觉基础推理能力提供了新的范式。

**Key Findings:**

- In this paper, we propose Bi-directional Perceptual Shaping (BiPS), which transforms question-conditioned masked views into bidirectional where-to-look signals that shape perception during training.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22120v1)
- [arXiv](https://arxiv.org/abs/2512.22120v1)

---

<a id='2512.22118v1'></a>
## [ProEdit: Inversion-based Editing From Prompts Done Right](https://arxiv.org/abs/2512.22118v1)

**Authors:** Zhi Ouyang, Dian Zheng, Xiao-Ming Wu, Jian-Jian Jiang, Kun-Yu Lin, Jingke Meng, Wei-Shi Zheng

**Published:** 2025-12-26

**Categories:** cs.CV

**Abstract:**

Inversion-based visual editing provides an effective and training-free way to edit an image or a video based on user instructions. Existing methods typically inject source image information during the sampling process to maintain editing consistency. However, this sampling strategy overly relies on source information, which negatively affects the edits in the target image (e.g., failing to change the subject's atributes like pose, number, or color as instructed). In this work, we propose ProEdit to address this issue both in the attention and the latent aspects. In the attention aspect, we introduce KV-mix, which mixes KV features of the source and the target in the edited region, mitigating the influence of the source image on the editing region while maintaining background consistency. In the latent aspect, we propose Latents-Shift, which perturbs the edited region of the source latent, eliminating the influence of the inverted latent on the sampling. Extensive experiments on several image and video editing benchmarks demonstrate that our method achieves SOTA performance. In addition, our design is plug-and-play, which can be seamlessly integrated into existing inversion and editing methods, such as RF-Solver, FireFlow and UniEdit.

**Analysis:**

好的，这是对论文“ProEdit: Inversion-based Editing From Prompts Done Right”的全面中文摘要，重点突出了其在计算机视觉领域的新颖性和重要性：

**论文题目：** ProEdit: Inversion-based Editing From Prompts Done Right (ProEdit：基于提示的正确反演式编辑)

**作者：** Zhi Ouyang, Dian Zheng, Xiao-Ming Wu, Jian-Jian Jiang, Kun-Yu Lin, Jingke Meng, Wei-Shi Zheng

**摘要**

**1. 研究问题/核心挑战：**

该论文主要解决了现有基于反演的视觉编辑方法在进行图像和视频编辑时遇到的一个关键问题：**过度注入源图像信息导致编辑失败**。现有方法为了保持编辑的一致性，会在采样过程中引入大量源图像的信息。然而，这种策略过度依赖源信息，严重影响了目标图像中主体属性（如姿态、数量、颜色）的准确修改，导致编辑结果不符合用户指令。

**2. 关键创新/方法贡献：**

为了解决上述问题，作者提出了名为 **ProEdit** 的新颖、无需训练的编辑方法，从**注意力（attention）**和**潜在空间（latent）**两个方面进行改进：

*   **注意力方面 (Attention Aspect) - KV-mix：**
    *   引入了 **KV-mix** 机制，通过混合源图像和目标图像在**编辑区域**的 KV 特征，来减轻源图像对编辑区域的影响，同时保持背景的一致性。
    *   对于**非编辑区域**，则完全注入源 KV 特征，以确保背景的稳定性。
    *   该机制可以应用于所有注意力操作，无需手动调整头、层或块。

*   **潜在空间方面 (Latent Aspect) - Latents-Shift：**
    *   提出了 **Latents-Shift** 模块，借鉴了风格迁移中的 AdaIN（Adaptive Instance Normalization）思想。
    *   通过在**编辑区域**的源潜在表示中注入随机噪声，来扰动其分布，从而消除反演得到的潜在表示对采样过程的影响。
    *   这有助于减少源图像属性的干扰，同时保持结构和背景的一致性。

**3. 主要结果与意义：**

*   **性能提升：** 大量实验表明，ProEdit 在多个图像和视频编辑基准测试中取得了**最先进 (SOTA) 的性能**。
*   **属性编辑能力增强：** 特别是在属性编辑方面，ProEdit 展现出前所未有的性能，有效解决了现有方法在此类任务上的不足。
*   **即插即用性 (Plug-and-play)：** ProEdit 的设计是即插即用的，可以**无缝集成**到现有的反演和编辑方法中，如 RF-Solver、FireFlow 和 UniEdit，极大地增强了现有方法的适用性。
*   **一致性与准确性兼顾：** ProEdit 能够同时实现**高背景一致性**和**准确的属性编辑**，解决了源图像信息注入与编辑目标之间的矛盾。
*   **通用性：** 该方法不仅适用于图像编辑，也成功应用于视频编辑任务，证明了其**通用性**。

**4. 提及的局限性：**

论文中并未明确列出局限性，但从方法描述和实验结果来看，其核心在于解决源信息注入问题。潜在的局限性可能在于：

*   **计算成本：** 虽然是训练-free 的，但引入的 KV-mix 和 Latents-Shift 模块可能会增加一定的计算开销。
*   **掩码提取的精度：** 论文提到使用注意力图提取掩码，虽然经过了扩散处理，但对于非常精细的编辑区域，掩码的精度可能仍是影响最终效果的因素。
*   **对特定模型的依赖（潜在）：** 虽然是即插即用的，但其效果可能在不同基础反演模型上表现略有差异。

**5. 潜在的未来研究方向：**

*   **更精细的掩码控制：** 探索更鲁棒和精细的编辑区域掩码提取方法，以应对更复杂的编辑场景。
*   **更高效的注意力融合：** 进一步优化 KV-mix 机制，探索更高效的注意力特征融合策略，以在保证性能的同时降低计算复杂度。
*   **更广泛的模型集成：** 将 ProEdit 集成到更多不同类型的生成模型（如 GANs）的反演编辑流程中，探索其通用性。
*   **交互式编辑的增强：** 结合大型语言模型（如论文中提到的 Qwen3-8B）进行指令引导编辑，可以进一步探索更自然、更用户友好的交互式编辑方式。
*   **视频编辑的鲁棒性提升：** 尽管在视频编辑方面表现出色，但对于更复杂的视频内容（如快速运动、遮挡等），仍有进一步提升时空一致性和编辑质量的空间。

**总结：**

ProEdit 是一项重要的研究成果，它成功地解决了基于反演的视觉编辑方法中长期存在的源图像信息过度注入问题。通过创新的 KV-mix 和 Latents-Shift 模块，ProEdit 在保持背景一致性的同时，显著提升了编辑的准确性和灵活性，尤其是在属性编辑方面。其即插即用的特性使其能够轻松地增强现有编辑方法，为未来更强大、更易用的视觉编辑工具奠定了基础。

**Key Findings:**

- In this work, we propose ProEdit to address this issue both in the attention and the latent aspects.
- In the attention aspect, we introduce KV-mix, which mixes KV features of the source and the target in the edited region, mitigating the influence of the source image on the editing region while maintaining background consistency.
- In the latent aspect, we propose Latents-Shift, which perturbs the edited region of the source latent, eliminating the influence of the inverted latent on the sampling.
- Extensive experiments on several image and video editing benchmarks demonstrate that our method achieves SOTA performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22118v1)
- [arXiv](https://arxiv.org/abs/2512.22118v1)

---

<a id='2512.22105v1'></a>
## [Learning Association via Track-Detection Matching for Multi-Object Tracking](https://arxiv.org/abs/2512.22105v1)

**Authors:** Momir Adžemović

**Published:** 2025-12-26

**Categories:** cs.CV

**Abstract:**

Multi-object tracking aims to maintain object identities over time by associating detections across video frames. Two dominant paradigms exist in literature: tracking-by-detection methods, which are computationally efficient but rely on handcrafted association heuristics, and end-to-end approaches, which learn association from data at the cost of higher computational complexity. We propose Track-Detection Link Prediction (TDLP), a tracking-by-detection method that performs per-frame association via link prediction between tracks and detections, i.e., by predicting the correct continuation of each track at every frame. TDLP is architecturally designed primarily for geometric features such as bounding boxes, while optionally incorporating additional cues, including pose and appearance. Unlike heuristic-based methods, TDLP learns association directly from data without handcrafted rules, while remaining modular and computationally efficient compared to end-to-end trackers. Extensive experiments on multiple benchmarks demonstrate that TDLP consistently surpasses state-of-the-art performance across both tracking-by-detection and end-to-end methods. Finally, we provide a detailed analysis comparing link prediction with metric learning-based association and show that link prediction is more effective, particularly when handling heterogeneous features such as detection bounding boxes. Our code is available at \href{https://github.com/Robotmurlock/TDLP}{https://github.com/Robotmurlock/TDLP}.

**Analysis:**

好的，这是对Momir Adžemović撰写的论文“Learning Association via Track-Detection Matching for Multi-Object Tracking”的全面中文摘要：

**论文题目：** Learning Association via Track-Detection Matching for Multi-Object Tracking

**作者：** Momir Adžemović

**摘要：**

**1. 主要问题/研究问题：**
本文旨在解决多目标跟踪（Multi-Object Tracking, MOT）中的核心挑战：如何在视频帧之间准确地维护和关联不同对象的身份。现有的方法主要分为两类：
*   **跟踪-检测（Tracking-by-Detection）方法：** 计算效率高，但依赖于手工设计的关联启发式规则，泛化能力受限。
*   **端到端（End-to-End）方法：** 能从数据中学习关联，但计算复杂度高，训练和推理速度慢。
这两种方法之间存在一个“差距”，即需要一种既能学习关联又能保持效率和模块化特性的方法。

**2. 关键创新/方法论贡献：**
作者提出了**Track-Detection Link Prediction (TDLP)**，一种新的跟踪-检测方法，它将数据关联问题转化为**逐帧的链接预测（Link Prediction）**问题。其核心创新在于：
*   **逐帧链接预测：** TDLP不依赖手工规则，而是通过预测每个检测框是否是现有轨迹的正确延续来学习关联。
*   **架构设计：** TDLP主要针对几何特征（如边界框），并可选择性地整合姿态和外观等其他线索。它采用Transformer编码器来处理时空信息和多模态特征。
*   **特征融合与交互：** 模型通过静态编码器、运动编码器、时间编码器和对象交互编码器来提取和融合多模态特征，并建模对象间的交互。
*   **计算效率与模块化：** 相较于端到端方法，TDLP在训练和推理时计算成本更低，同时保留了跟踪-检测方法的模块化优势。

**3. 主要结果及其意义：**
*   **性能卓越：** TDLP在多个具有挑战性的基准数据集（如DanceTrack, SportsMOT, BEE24）上取得了最先进的性能，超越了现有的跟踪-检测和端到端方法。
*   **轻量级变体优势：** 即使是仅使用边界框特征的轻量级TDLP-bbox变体，也优于所有依赖手工规则的跟踪器，甚至优于一些使用外观特征的方法。
*   **对度量学习的分析：** 文章深入分析了链接预测与度量学习（Metric Learning）在关联任务上的差异，并证明了在处理异构特征（尤其是边界框）时，链接预测更为有效。
*   **鲁棒性：** TDLP在处理非线性运动和遮挡等复杂场景时表现出更强的鲁棒性，尤其是在阈值测试中，能有效抑制假阳性检测。

**4. 论文中提到的局限性：**
*   **计算成本：** TDLP的主要局限性在于其计算成本，这源于二次方的跟踪-检测评分和Transformer编码器。

**5. 潜在的未来研究方向：**
*   **优化计算成本：** 未来工作将探索更长的时序窗口和架构优化，以进一步降低计算成本。
*   **更深入的分析：** 论文中提到对MOT17数据集的局限性（训练数据不足、无验证集）需要更深入的研究。

**总结：**
这篇论文提出了TDLP，一种创新的跟踪-检测方法，通过将数据关联视为逐帧的链接预测问题，成功地弥合了现有方法在效率和性能之间的差距。TDLP通过学习数据关联，避免了手工规则的限制，并能有效融合多模态特征，在多个基准测试中取得了显著的性能提升。其对链接预测与度量学习的深入分析，为理解和改进多目标跟踪中的关联机制提供了重要见解。尽管存在计算成本方面的挑战，TDLP为未来更高效、更鲁棒的多目标跟踪系统奠定了坚实的基础。

**Key Findings:**

- We propose Track-Detection Link Prediction (TDLP), a tracking-by-detection method that performs per-frame association via link prediction between tracks and detections, i.e., by predicting the correct continuation of each track at every frame.
- Extensive experiments on multiple benchmarks demonstrate that TDLP consistently surpasses state-of-the-art performance across both tracking-by-detection and end-to-end methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22105v1)
- [arXiv](https://arxiv.org/abs/2512.22105v1)

---

<a id='2512.22096v1'></a>
## [Yume-1.5: A Text-Controlled Interactive World Generation Model](https://arxiv.org/abs/2512.22096v1)

**Authors:** Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, Kaipeng Zhang

**Published:** 2025-12-26

**Categories:** cs.CV

**Abstract:**

Recent approaches have demonstrated the promise of using diffusion models to generate interactive and explorable worlds. However, most of these methods face critical challenges such as excessively large parameter sizes, reliance on lengthy inference steps, and rapidly growing historical context, which severely limit real-time performance and lack text-controlled generation capabilities. To address these challenges, we propose \method, a novel framework designed to generate realistic, interactive, and continuous worlds from a single image or text prompt. \method achieves this through a carefully designed framework that supports keyboard-based exploration of the generated worlds. The framework comprises three core components: (1) a long-video generation framework integrating unified context compression with linear attention; (2) a real-time streaming acceleration strategy powered by bidirectional attention distillation and an enhanced text embedding scheme; (3) a text-controlled method for generating world events. We have provided the codebase in the supplementary material.

**Analysis:**

好的，这是对论文“Yume1.5: A Text-Controlled Interactive World Generation Model”的全面中文摘要：

**Yume1.5: 一种文本控制的交互式世界生成模型**

**1. 研究问题/背景**

尽管基于扩散模型的交互式和可探索世界生成方法取得了显著进展，但现有方法普遍面临严峻挑战：参数量过大、推理过程耗时过长、历史上下文急剧增长导致实时性能受限，并且缺乏精细的文本控制生成能力。这些问题严重阻碍了生成逼真、交互性强且可实时探索的虚拟世界。

**2. 主要创新点/方法论贡献**

为了解决上述问题，论文提出了 **Yume1.5**，一个新颖的框架，能够从单个图像或文本提示生成逼真、交互式且连续的世界。Yume1.5 的核心在于其精心设计的框架，支持通过键盘进行交互式探索，并包含三个主要组成部分：

*   **长视频生成框架：** 集成了统一的上下文压缩与线性注意力机制，以高效处理长视频序列，克服了传统方法中上下文长度增加带来的计算瓶颈。
*   **实时流加速策略：** 采用双向注意力蒸馏和增强的文本嵌入方案，显著提升了生成速度和实时性，使得模型能够进行实时流式交互。
*   **文本控制的世界事件生成：** 引入了一种文本控制方法，能够生成世界中的动态事件，极大地增强了生成内容的丰富性和可控性。

具体技术贡献包括：
*   **联合时空通道建模 (TSCM)：** 提出了一种新的建模方法，用于高效的长视频生成，即使上下文长度增加，也能保持稳定的采样速度。
*   **加速方法：** 结合了 Self-Forcing 和 TSCM，加速了 Yume1.5 的推理过程，并有效减少了误差累积。
*   **数据处理与模型架构设计：** 通过精心设计的数据集（包括真实世界、合成和事件数据集）以及模型架构，Yume1.5 在世界生成和编辑方面取得了优越的性能。

**3. 主要结果及其意义**

Yume1.5 在多个方面取得了显著成果：

*   **卓越的可控性：** 在指令跟随（相机运动跟踪）方面得分达到 0.836，显著优于现有模型。
*   **高效的生成速度：** 在单块 A100 GPU 上，以 540p 分辨率实现了平均 12 fps 的生成速度。
*   **高质量的长视频生成：** 在长视频生成任务中，Yume1.5 能够保持更稳定的美学分数和图像质量，尤其是在处理更长的视频序列时。
*   **交互式探索能力：** 通过键盘控制，用户可以直观地探索生成的虚拟世界，实现了真正的交互性。
*   **文本控制事件生成：** 能够根据文本描述生成动态事件，增加了生成内容的趣味性和真实感。

这些结果表明，Yume1.5 在生成逼真、交互式且可控的虚拟世界方面取得了重大突破，为沉浸式体验和虚拟环境模拟开辟了新的可能性。

**4. 提及的局限性**

论文也指出了 Yume1.5 的一些局限性：

*   **生成伪影：** 在某些场景下，模型仍会产生一些生成伪影，例如车辆倒行或角色反向行走。
*   **高密度场景性能下降：** 在极高人群密度的场景下，生成性能会有所下降。
*   **模型规模限制：** 尽管通过 TSCM 缓解了部分问题，但 5B 参数模型的容量仍然有限，进一步提升分辨率会增加生成延迟。

**5. 潜在的未来研究方向**

论文展望了未来的研究方向：

*   **扩展到更复杂的交互：** 支持更复杂的虚拟世界交互和更广泛的应用场景，例如虚拟环境和模拟系统。
*   **探索更大模型架构：** 考虑使用 Mixture-of-Experts (MoE) 等架构来增加模型参数量，同时降低推理延迟，以应对更复杂的生成任务。
*   **提升生成质量：** 进一步研究以消除生成伪影，并提高在高密度场景下的性能。

总而言之，Yume1.5 是一个在文本控制的交互式世界生成领域具有重要意义的模型。它通过创新的长视频生成框架、实时加速策略和文本控制事件生成能力，有效解决了现有方法的关键挑战，并为未来更逼真、更具交互性的虚拟世界生成奠定了坚实基础。

**Key Findings:**

- To address these challenges, we propose \method, a novel framework designed to generate realistic, interactive, and continuous worlds from a single image or text prompt.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22096v1)
- [arXiv](https://arxiv.org/abs/2512.22096v1)

---

<a id='2512.22065v1'></a>
## [StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](https://arxiv.org/abs/2512.22065v1)

**Authors:** Zhiyao Sun, Ziqiao Peng, Yifeng Ma, Yi Chen, Zhengguang Zhou, Zixiang Zhou, Guozhen Zhang, Youliang Zhang, Yuan Zhou, Qinglin Lu, Yong-Jin Liu

**Published:** 2025-12-26

**Categories:** cs.CV, cs.AI, cs.HC

**Abstract:**

Real-time, streaming interactive avatars represent a critical yet challenging goal in digital human research. Although diffusion-based human avatar generation methods achieve remarkable success, their non-causal architecture and high computational costs make them unsuitable for streaming. Moreover, existing interactive approaches are typically limited to head-and-shoulder region, limiting their ability to produce gestures and body motions. To address these challenges, we propose a two-stage autoregressive adaptation and acceleration framework that applies autoregressive distillation and adversarial refinement to adapt a high-fidelity human video diffusion model for real-time, interactive streaming. To ensure long-term stability and consistency, we introduce three key components: a Reference Sink, a Reference-Anchored Positional Re-encoding (RAPR) strategy, and a Consistency-Aware Discriminator. Building on this framework, we develop a one-shot, interactive, human avatar model capable of generating both natural talking and listening behaviors with coherent gestures. Extensive experiments demonstrate that our method achieves state-of-the-art performance, surpassing existing approaches in generation quality, real-time efficiency, and interaction naturalness. Project page: https://streamavatar.github.io .

**Analysis:**

好的，这是对论文“StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars”的全面中文摘要：

**论文题目：** StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars

**作者：** Zhiyao Sun, Ziqiao Peng, Yifeng Ma, Yi Chen, Zhengguang Zhou, Zixiang Zhou, Guozhen Zhang, Youliang Zhang, Yuan Zhou, Qinglin Lu, Yong-Jin Liu

---

**摘要：**

**1. 研究问题/核心挑战：**
该论文旨在解决数字人类研究中的一个关键且具有挑战性的目标：实现**实时、流式交互式人类虚拟形象**。现有的基于扩散模型的虚拟形象生成方法虽然在生成质量上表现出色，但其**非因果（non-causal）架构和高计算成本**使其不适用于流式生成。此外，现有的交互式方法通常**局限于头部和肩部区域**，限制了其生成手势和身体动作的能力。因此，研究的核心问题是如何在保持高质量生成的同时，实现低延迟、流式、全身交互式虚拟形象的生成。

**2. 主要创新点/方法贡献：**
为了应对上述挑战，作者提出了一个名为 **StreamAvatar** 的框架，其核心创新在于一个**两阶段的自回归适应与加速框架**：

*   **阶段一：自回归蒸馏（Autoregressive Distillation）**
    *   **模型重构：** 将原有的双向扩散模型（Diffusion Transformer, DiT）重构为**块因果（block-causal）的DiT**，以支持自回归生成。
    *   **蒸馏过程：** 利用**分数身份蒸馏（Score Identity Distillation, SiD）**技术，将一个强大的、但速度较慢的双向教师模型（teacher model）的生成能力蒸馏到一个快速的、单向的（因果）学生模型（student model）中。这显著减少了推理步骤，将DiT的去噪过程加速了40倍。
    *   **长期稳定性与一致性组件：**
        *   **参考槽（Reference Sink）：** 强制模型持续关注参考帧，以防止在长视频生成中出现身份漂移。
        *   **参考锚定位置编码（Reference-Anchored Positional Re-encoding, RAPR）：** 解决训练-测试不匹配和注意力衰减问题，通过限制位置编码的最大距离来模拟长视频位置偏移，从而提高长序列生成的一致性。
        *   **一致性感知判别器（Consistency-Aware Discriminator）：** 在第二阶段用于提升生成质量和稳定性。

*   **阶段二：对抗性精炼（Adversarial Refinement）**
    *   为了解决蒸馏过程中可能出现的质量下降（如模糊、失真）和时间不一致问题，引入了**一致性感知判别器**。该判别器包含局部真实性分支和全局一致性分支，以评估生成帧的真实性和整体一致性。

*   **交互式能力增强：**
    *   **音频掩码（Audio Mask）：** 采用TalkNet生成的音频掩码来区分说话和倾听阶段，而非直接分离音频。这避免了音频特征的失真，并提供了精确的时间控制。
    *   **音频注意力模块：** 在Transformer块中引入了两个音频相关注意力模块：**音频注意力（Audio Attention）**用于驱动说话阶段的表情和动作，**交互音频注意力（Interact Audio Attention）**用于生成倾听阶段的自然反应。

**3. 主要结果与意义：**
*   **实时性与效率：** StreamAvatar实现了**实时流式生成**，RTF值远低于1，整体延迟仅为1.20秒，比现有方法快几个数量级。
*   **高质量生成：** 在FID、FVD、ASE、IQA等指标上，StreamAvatar在短视频和长视频生成上均取得了**最先进（state-of-the-art）或具有竞争力**的性能，尤其在生成质量、运动幅度方面表现优异。
*   **交互自然性：** 模型能够生成**自然、连贯的说话和倾听行为**，包括丰富的表情、手势和身体动作，并且在说话和倾听状态之间实现**平滑过渡**。
*   **长期一致性：** 通过引入Reference Sink和RAPR，模型在长视频生成中表现出**更好的身份保持和时间一致性**。
*   **全身生成：** 与许多仅限于头部和肩部的方法不同，StreamAvatar能够生成**全身**的虚拟形象。

**4. 提及的局限性：**
*   **有限的时间上下文：** 在某些区域，如果这些区域在长时间内被遮挡，模型可能**难以生成一致的内容**。
*   **VAE解码的计算瓶颈：** VAE解码占用了模型总处理时间的一半以上，未来可以探索**更高效的VAE解码**方法来进一步降低流式延迟。

**5. 潜在的未来研究方向：**
*   **长时记忆机制：** 引入长时记忆机制来解决因有限时间上下文导致的区域不一致问题。
*   **高效VAE解码：** 探索更优化的VAE解码策略，以进一步提升流式生成的速度。
*   **伦理考量与安全：** 作者强调了该技术可能被滥用的风险（如制造虚假身份、欺诈等），并承诺通过水印、明确披露合成内容等方式来缓解这些风险。未来的工作将致力于与社区合作，开发更先进的深度伪造检测工具，并建立媒体溯源标准。

**总结：**
StreamAvatar通过创新的两阶段蒸馏和精炼框架，以及用于提升长期稳定性和交互性的关键组件，成功地解决了现有扩散模型在实时流式、全身交互式虚拟形象生成方面的瓶颈。该方法在生成质量、效率和交互自然性方面均取得了显著的进步，为数字人类和虚拟交互领域带来了重要的贡献。

**Key Findings:**

- To address these challenges, we propose a two-stage autoregressive adaptation and acceleration framework that applies autoregressive distillation and adversarial refinement to adapt a high-fidelity human video diffusion model for real-time, interactive streaming.
- To ensure long-term stability and consistency, we introduce three key components: a Reference Sink, a Reference-Anchored Positional Re-encoding (RAPR) strategy, and a Consistency-Aware Discriminator.
- Building on this framework, we develop a one-shot, interactive, human avatar model capable of generating both natural talking and listening behaviors with coherent gestures.
- Extensive experiments demonstrate that our method achieves state-of-the-art performance, surpassing existing approaches in generation quality, real-time efficiency, and interaction naturalness.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22065v1)
- [arXiv](https://arxiv.org/abs/2512.22065v1)

---

<a id='2512.22046v1'></a>
## [Backdoor Attacks on Prompt-Driven Video Segmentation Foundation Models](https://arxiv.org/abs/2512.22046v1)

**Authors:** Zongmin Zhang, Zhen Sun, Yifan Liao, Wenhan Dong, Xinlei He, Xingshuo Han, Shengmin Xu, Xinyi Huang

**Published:** 2025-12-26

**Categories:** cs.CV, cs.CR

**Abstract:**

Prompt-driven Video Segmentation Foundation Models (VSFMs) such as SAM2 are increasingly deployed in applications like autonomous driving and digital pathology, raising concerns about backdoor threats. Surprisingly, we find that directly transferring classic backdoor attacks (e.g., BadNet) to VSFMs is almost ineffective, with ASR below 5\%. To understand this, we study encoder gradients and attention maps and observe that conventional training keeps gradients for clean and triggered samples largely aligned, while attention still focuses on the true object, preventing the encoder from learning a distinct trigger-related representation. To address this challenge, we propose BadVSFM, the first backdoor framework tailored to prompt-driven VSFMs. BadVSFM uses a two-stage strategy: (1) steer the image encoder so triggered frames map to a designated target embedding while clean frames remain aligned with a clean reference encoder; (2) train the mask decoder so that, across prompt types, triggered frame-prompt pairs produce a shared target mask, while clean outputs stay close to a reference decoder. Extensive experiments on two datasets and five VSFMs show that BadVSFM achieves strong, controllable backdoor effects under diverse triggers and prompts while preserving clean segmentation quality. Ablations over losses, stages, targets, trigger settings, and poisoning rates demonstrate robustness to reasonable hyperparameter changes and confirm the necessity of the two-stage design. Finally, gradient-conflict analysis and attention visualizations show that BadVSFM separates triggered and clean representations and shifts attention to trigger regions, while four representative defenses remain largely ineffective, revealing an underexplored vulnerability in current VSFMs.

**Analysis:**

好的，这是对论文“Backdoor Attacks on Prompt-Driven Video Segmentation Foundation Models”的全面中文摘要：

**论文题目：** Backdoor Attacks on Prompt-Driven Video Segmentation Foundation Models

**作者：** Zongmin Zhang, Zhen Sun, Yifan Liao, Wenhan Dong, Xinlei He, Xingshuo Han, Shengmin Xu, Xinyi Huang

**摘要：**

这篇论文**首次提出了针对提示驱动视频分割基础模型（VSFMs）的后门攻击框架 BadVSFM**，揭示了这些模型在安全方面存在的严重漏洞。

**1. 研究问题/核心挑战：**

随着像SAM2这样的提示驱动VSFM在自动驾驶和数字病理等领域的广泛应用，其鲁棒性问题日益受到关注，特别是后门攻击的威胁。然而，研究人员发现，将经典的后门攻击方法（如BadNet）直接应用于VSFM时，效果非常差，攻击成功率（ASR）低于5%。论文深入分析了这一失败现象，发现传统后门攻击训练下，图像编码器的干净样本和触发样本的梯度高度对齐，模型注意力仍然集中在真实目标上，未能学习到与触发器相关的独特表示。

**2. 主要创新与方法贡献：**

为了克服上述挑战，论文提出了**BadVSFM**，一个专门为提示驱动VSFM设计的两阶段后门攻击框架：

*   **阶段一：图像编码器对齐。** 该阶段仅微调图像编码器，使得触发帧的嵌入映射到一个预设的目标嵌入，同时保持干净帧的嵌入与一个干净参考模型对齐，以保留模型效用。
*   **阶段二：掩码解码器训练。** 该阶段仅更新掩码解码器，使得跨不同提示类型（点、框、掩码）的触发帧-提示对能够生成一个共享的目标掩码（例如，全零掩码），同时保持干净帧的分割行为与参考解码器一致。

该框架通过**显式地分离触发样本和干净样本的表示空间，并条件化解码器以生成目标掩码**，实现了有效的后门注入。

**3. 主要结果与意义：**

*   **强大的后门攻击效果：** 在DAVIS和LVOS两个数据集上，使用多种VSFM模型（包括SAM2、MedSAM2、SAM2-Long、BioSAM2和EdgeTAM）进行的广泛实验表明，BadVSFM能够实现**强大且可控的后门效果**，显著高于现有基线后门攻击方法，ASR提升高达90%以上。
*   **保持模型效用：** 尽管实现了高ASR，BadVSFM在保持模型在干净数据上的分割性能（mIoU和J&F指标）方面表现出色，仅有微小的下降，甚至在某些情况下有所提升。
*   **鲁棒性与必要性：** 通过对损失函数、训练阶段、攻击目标、触发器配置和中毒率的系统性消融研究，证明了BadVSFM对合理的超参数变化具有鲁棒性，并且**两阶段设计对于同时保证后门效果和干净性能至关重要**。
*   **解释性分析：** 梯度冲突分析和注意力图可视化表明，BadVSFM能够**将触发样本和干净样本的表示推向不同的方向，并将注意力转移到触发区域**，这解释了为何传统后门攻击在VSFM上失效。
*   **现有防御措施的无效性：** 四种代表性的防御方法（微调、剪枝、Spectral Signatures和STRIP）在对抗BadVSFM时**效果不佳**，凸显了当前VSFM面临的实际且未被充分探索的安全漏洞。

**4. 论文提及的局限性：**

*   **模型覆盖范围有限：** 实验主要集中在有限的VSFM模型上，未来研究需要扩展到更多模型架构，以验证攻击原理的普适性。
*   **数据集多样性有限：** 评估主要基于DAVIS和LVOS两个数据集，未来需要更多样化的数据集来评估模型的泛化能力。
*   **防御策略探索有限：** 仅评估了四种通用防御技术，揭示了现有通用防御对VSFM的不足，需要开发更具针对性的防御策略。

**5. 潜在的未来研究方向：**

*   **更广泛的模型和数据集评估：** 探索BadVSFM在更多不同类型和复杂场景下的VSFM模型上的表现。
*   **开发针对性的防御机制：** 设计专门用于VSFM的后门防御和检测方法，例如基于时空异常检测或触发器反演的方法。
*   **深入理解VSFM的脆弱性：** 进一步研究VSFM在复杂时空条件下的脆弱性，以及如何增强其对后门威胁的抵抗力。
*   **探索更隐蔽的攻击方式：** 研究如何设计更难被检测到的触发器和攻击策略。

**总结：**

这篇论文**首次系统地提出了针对提示驱动视频分割基础模型（VSFM）的后门攻击方法BadVSFM**，并深入分析了传统后门攻击失效的原因。研究结果表明，BadVSFM能够有效地在VSFM中植入后门，同时保持模型在干净数据上的性能，并且现有的通用防御方法难以有效抵御。这揭示了VSFM领域一个**被低估的安全风险**，并强调了开发**专门针对VSFM的后门攻击和防御策略**的紧迫性。

**Key Findings:**

- To address this challenge, we propose BadVSFM, the first backdoor framework tailored to prompt-driven VSFMs. BadVSFM uses a two-stage strategy: (1) steer the image encoder so triggered frames map to a designated target embedding while clean frames remain aligned with a clean reference encoder; (2) train the mask decoder so that, across prompt types, triggered frame-prompt pairs produce a shared target mask, while clean outputs stay close to a reference decoder.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22046v1)
- [arXiv](https://arxiv.org/abs/2512.22046v1)

---

<a id='2512.22010v1'></a>
## [LongFly: Long-Horizon UAV Vision-and-Language Navigation with Spatiotemporal Context Integration](https://arxiv.org/abs/2512.22010v1)

**Authors:** Wen Jiang, Li Wang, Kangyao Huang, Wei Fan, Jinyuan Liu, Shaoyu Liu, Hongwei Duan, Bin Xu, Xiangyang Ji

**Published:** 2025-12-26

**Categories:** cs.CV, cs.AI

**Abstract:**

Unmanned aerial vehicles (UAVs) are crucial tools for post-disaster search and rescue, facing challenges such as high information density, rapid changes in viewpoint, and dynamic structures, especially in long-horizon navigation. However, current UAV vision-and-language navigation(VLN) methods struggle to model long-horizon spatiotemporal context in complex environments, resulting in inaccurate semantic alignment and unstable path planning. To this end, we propose LongFly, a spatiotemporal context modeling framework for long-horizon UAV VLN. LongFly proposes a history-aware spatiotemporal modeling strategy that transforms fragmented and redundant historical data into structured, compact, and expressive representations. First, we propose the slot-based historical image compression module, which dynamically distills multi-view historical observations into fixed-length contextual representations. Then, the spatiotemporal trajectory encoding module is introduced to capture the temporal dynamics and spatial structure of UAV trajectories. Finally, to integrate existing spatiotemporal context with current observations, we design the prompt-guided multimodal integration module to support time-based reasoning and robust waypoint prediction. Experimental results demonstrate that LongFly outperforms state-of-the-art UAV VLN baselines by 7.89\% in success rate and 6.33\% in success weighted by path length, consistently across both seen and unseen environments.

**Analysis:**

好的，这是对论文“LongFly: Long-Horizon UAV Vision-and-Language Navigation with Spatiotemporal Context Integration”的全面中文摘要：

**论文题目：** LongFly: Long-Horizon UAV Vision-and-Language Navigation with Spatiotemporal Context Integration

**作者：** Wen Jiang, Li Wang, Kangyao Huang, Wei Fan, Jinyuan Liu, Shaoyu Liu, Hongwei Duan, Bin Xu, and Xiangyang Ji

**摘要：**

**1. 研究问题/核心挑战：**
本文旨在解决无人机（UAV）在执行长距离、复杂环境下的视觉与语言导航（VLN）任务时遇到的核心挑战。现有UAV VLN方法在处理长距离导航时，难以有效建模时空上下文信息，导致语义对齐不准确、路径规划不稳定，尤其是在信息密度高、视角变化快、结构动态复杂的场景下表现尤为明显。

**2. 主要创新与方法贡献：**
作者提出了**LongFly**，一个专门为长距离UAV VLN设计的时空上下文建模框架。其核心创新在于引入了**历史感知时空建模策略**，将零散且冗余的历史数据转化为结构化、紧凑且富有表现力的表示。具体而言，LongFly包含三个关键模块：

*   **基于槽的历史图像压缩（SHIC）模块：** 该模块动态地从多视角历史观测中提取关键信息，将其压缩成固定长度的上下文表示。这解决了直接存储高维历史特征带来的计算成本问题，并能捕捉持久的地标和空间布局。
*   **时空轨迹编码（STE）模块：** 该模块用于捕捉UAV轨迹的时间动态和空间结构。它将历史航点信息转化为轨迹令牌，以显式的运动先验来捕捉长距离路径的演变。
*   **提示引导的多模态融合（PGM）模块：** 该模块用于整合现有的时空上下文（历史视觉记忆和运动历史）与当前观测。它将多模态上下文组织成一个结构化的提示，并利用大型多模态语言模型（MLLM）进行时间推理和鲁棒的航点预测，从而实现指令对齐和长距离导航的一致性。

**3. 主要结果与意义：**
LongFly在OpenUAV数据集上进行了广泛的实验评估，并在**测试集（Unseen）**上取得了显著的性能提升。与最先进的UAV VLN基线方法相比，LongFly在**成功率（SR）上提升了7.89%**，在**路径长度加权的成功率（SPL）上提升了6.33%**。这些改进在**所有难度级别（Full、Easy、Hard）**以及**所有未见场景（Overall、Object、Map）**中都得到了一致的体现。

*   **意义重大：** 这些结果表明，LongFly提出的时空上下文建模策略对于提升UAV在复杂、长距离导航任务中的鲁棒性、准确性和效率至关重要。它有效地解决了现有方法在处理长距离依赖性时的不足，为实现更自主、更可靠的UAV导航提供了新的解决方案。尤其是在处理复杂布局、长距离依赖和语义模糊的场景下，LongFly展现出强大的优势。

**4. 提及的局限性：**
论文中提到，尽管LongFly取得了显著进展，但与人类的表现相比仍存在差距。此外，在**未见环境（Unseen Environments）**上的泛化能力仍有待提高，作者指出环境分布的偏移比物体类别的偏移更具挑战性，暗示了增加训练环境多样性的重要性。

**5. 潜在的未来研究方向：**
基于论文的发现和局限性，潜在的未来研究方向包括：

*   **增强对未见环境的泛化能力：** 通过引入更多样化的训练环境和数据增强技术来提高模型在全新场景下的适应性。
*   **进一步提升与人类表现的差距：** 探索更先进的建模技术或更精细的指令理解机制，以缩小与人类导航员的性能差距。
*   **探索更高效的上下文压缩与融合机制：** 尽管SHIC模块已有效压缩历史信息，但仍可研究更轻量级或更具信息保留能力的压缩方法。同时，探索更高效的多模态融合策略，以进一步降低计算复杂度。
*   **更广泛的应用场景探索：** 将LongFly框架扩展到其他需要长距离、复杂环境导航的机器人应用中，例如地面机器人或水下机器人。
*   **实时性与效率的进一步优化：** 对于实际的UAV应用，实时性至关重要。未来研究可以关注如何进一步优化模型的推理速度，使其能够满足实时导航的需求。

**Key Findings:**

- To this end, we propose LongFly, a spatiotemporal context modeling framework for long-horizon UAV VLN.
- First, we propose the slot-based historical image compression module, which dynamically distills multi-view historical observations into fixed-length contextual representations.
- Experimental results demonstrate that LongFly outperforms state-of-the-art UAV VLN baselines by 7.89\% in success rate and 6.33\% in success weighted by path length, consistently across both seen and unseen environments.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22010v1)
- [arXiv](https://arxiv.org/abs/2512.22010v1)

---

<a id='2512.22009v1'></a>
## [iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception](https://arxiv.org/abs/2512.22009v1)

**Authors:** Sarthak Mehrotra, Sairam V C Rebbapragada, Mani Hemanth Reddy Bonthu, Vineeth N Balasubramanian

**Published:** 2025-12-26

**Categories:** cs.CV

**Abstract:**

Multimodal Large Language Models (MLLMs) show strong potential for interpreting and interacting with complex, pixel-rich Graphical User Interface (GUI) environments. However, building agents that are both efficient for high-level tasks and precise for fine-grained interactions remains challenging. GUI agents must perform routine actions efficiently while also handling tasks that demand exact visual grounding, yet existing approaches struggle when accuracy depends on identifying specific interface elements. These MLLMs also remain large and cannot adapt their reasoning depth to the task at hand. In this work, we introduce iSHIFT: Implicit Slow-fast Hybrid Inference with Flexible Tokens, a lightweight agent that integrates latent thinking (implicit chain-of-thought) with a perception control module. iSHIFT enables an MLLM to switch between a slow mode, which leverages detailed visual grounding for high precision and a fast mode that uses global cues for efficiency. Special perception tokens guide attention to relevant screen regions, allowing the model to decide both how to reason and where to focus. Despite its compact 2.5B size, iSHIFT matches state-of-the-art performance on multiple benchmark datasets.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文标题：** iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception
**作者：** Sarthak Mehrotra, Sairam V C Rebbapragada, Mani Hemanth Reddy Bonthu, Vineeth N Balasubramanian
**分类：** cs.CV
**发表日期：** 2025-12-26

---

### 论文分析

**1. 论文的主要贡献（2-3句话的简洁总结）**

该论文提出了一种名为 iSHIFT 的轻量级 GUI 智能体，它通过结合隐式链式思考（latent thinking）和自适应感知控制模块，实现了在不同任务复杂度下推理速度和精度的权衡。iSHIFT 能够根据任务需求在“慢速模式”（高精度视觉细节）和“快速模式”（全局线索效率）之间切换，并通过特殊的感知 token 来引导模型关注屏幕的关键区域，从而在保持模型紧凑（2.5B 参数）的同时，达到了与现有最先进方法相当的性能。

**2. 关键创新或方法论**

iSHIFT 的核心创新在于其 **“隐式慢-快混合推理与灵活 token”（Implicit Slow-fast Hybrid Inference with Flexible Tokens）** 的设计理念，具体体现在以下几个方面：

*   **慢-快推理模式切换 (Slow-Fast Inference Modes):** 这是最核心的创新点。
    *   **慢速模式 (Slow Mode):** 强调 **详细的视觉接地 (detailed visual grounding)**。这意味着模型会深入分析界面元素的细节，例如文本内容、图标形状、布局关系等，以实现高精度操作，尤其适用于需要精确识别特定界面元素（如点击某个按钮、填写特定文本框）的任务。
    *   **快速模式 (Fast Mode):** 侧重于利用 **全局线索 (global cues)** 来提高效率。模型会从整体上理解界面布局和上下文信息，快速做出判断，适用于对精度要求不高但需要快速响应的任务，例如滚动页面、识别整体页面类型等。
    *   **自适应切换机制:** 模型能够根据任务的性质和当前状态，**动态地在两种模式之间切换**，从而在效率和精度之间找到最佳平衡点。

*   **感知控制模块与特殊感知 Token (Perception Control Module & Special Perception Tokens):**
    *   **引导注意力 (Guiding Attention):** iSHIFT 引入了 **特殊的感知 token**，这些 token 并非传统的视觉特征或语言 token，而是被设计用来 **显式地指导模型将注意力集中到屏幕上的相关区域**。这使得模型能够主动地“决定”应该关注界面的哪些部分，而不是被动地处理所有像素信息。
    *   **决策推理与视觉焦点结合:** 这些感知 token 不仅帮助模型进行视觉聚焦，还与模型的推理过程相结合，**共同决定“如何推理”和“在哪里聚焦”**。这是一种更主动、更具策略性的感知方式。

*   **隐式链式思考 (Implicit Chain-of-Thought / Latent Thinking):** 论文提到“latent thinking”，这暗示模型在内部进行一种类似链式思考的推理过程，但这种思考是**隐式的**，不需要显式地生成中间的推理步骤。这有助于模型在不增加模型复杂度的前提下，提升其逻辑推理能力，尤其是在处理多步操作或复杂决策时。

*   **轻量级设计 (Lightweight Design):** 尽管 iSHIFT 能够达到 SOTA 性能，但其参数量仅为 **2.5B**。这表明其设计在模型效率方面做了大量优化，使其更易于部署和运行，尤其是在资源受限的环境中。

**3. 对该领域的潜在影响**

iSHIFT 的研究对计算机视觉和多模态大模型领域具有重要的潜在影响：

*   **推动 GUI 智能体的发展:** GUI 智能体是实现人机交互自动化和增强用户体验的关键。iSHIFT 的方法为构建更高效、更智能的 GUI 智能体提供了新的思路，尤其是在处理复杂、动态的 GUI 环境方面。
*   **提升 MLLMs 的效率和适应性:** 现有的 MLLMs 通常参数量巨大，且推理速度较慢。iSHIFT 的轻量级设计和自适应推理模式，证明了在不牺牲性能的前提下，可以显著提升 MLLMs 的效率和对不同任务的适应性。
*   **开创主动感知的新范式:** 通过引入特殊的感知 token 来引导注意力，iSHIFT 提出了一种更主动、更具策略性的视觉感知方式，这可能启发未来在其他视觉任务中设计更智能的注意力机制。
*   **降低部署门槛:** 2.5B 的参数量使得 iSHIFT 更有可能被部署到边缘设备或对计算资源有严格要求的场景，从而加速 MLLMs 的实际应用。
*   **为“决策式感知”提供理论支持:** iSHIFT 的方法将“决策”与“感知”紧密结合，模型不仅感知，还根据任务目标主动决定感知什么，这为“决策式感知”或“目标驱动感知”的研究提供了实践案例。

**4. 可能受益于此研究的相关领域或应用**

*   **自动化软件测试 (Automated Software Testing):** 能够更高效、更精确地模拟用户操作，发现软件中的 bug。
*   **机器人导航与交互 (Robotics Navigation and Interaction):** 机器人可以通过理解和操作 GUI 来与软件系统交互，完成更复杂的任务。
*   **无障碍技术 (Accessibility Technologies):** 帮助视障人士或其他有特殊需求的用户更便捷地与数字界面交互。
*   **远程协助与支持 (Remote Assistance and Support):** 远程操作员可以更准确地指导用户完成软件操作。
*   **游戏 AI (Game AI):** 游戏中的 NPC 可以更智能地理解和操作游戏界面，提升游戏体验。
*   **智能助手与自动化工作流 (Intelligent Assistants and Automated Workflows):** 构建更强大的自动化工具，处理日常的电脑操作任务。
*   **教育与培训 (Education and Training):** 创建交互式教程，指导用户学习软件操作。

**5. 从摘要中可以推断出的局限性**

尽管摘要描绘了 iSHIFT 的诸多优点，但仍可以推断出一些潜在的局限性：

*   **“隐式”推理的解释性问题:** “Latent thinking”或隐式链式思考虽然提高了效率，但可能降低了模型决策过程的可解释性。当出现错误时，可能难以追溯具体原因。
*   **感知 Token 的设计与泛化性:** 特殊感知 token 的设计可能需要针对特定类型的 GUI 或任务进行优化。其泛化能力（能否轻松应用于完全不同类型的界面）有待验证。
*   **模式切换的鲁棒性:** 慢-快模式的切换机制虽然是优点，但其切换的准确性和鲁棒性（在模糊或不确定的情况下能否正确切换）是关键。如果切换不当，可能导致性能下降。
*   **对“高层任务”和“细粒度交互”的定义:** 摘要提到“high-level tasks”和“fine-grained interactions”，但具体界限和模型在不同复杂度任务上的表现差异，需要更详细的实验数据来支撑。
*   **“紧凑”的定义:** 2.5B 参数量在 MLLMs 中确实算紧凑，但与传统的 CV 模型相比仍可能较大。其在极度资源受限环境下的表现仍需考量。
*   **基准数据集的局限性:** 摘要提到“multiple benchmark datasets”，但这些数据集是否能完全代表真实世界 GUI 交互的复杂性和多样性，以及模型在未见过的新型 GUI 上的泛化能力，是需要进一步研究的。
*   **“状态-of-the-art”的定义:** 摘要声称“matches state-of-the-art performance”，这通常意味着在特定指标上达到或超越现有最佳水平。但“state-of-the-art”的定义本身可能随时间推移而变化，并且可能只在某些特定任务或数据集上成立。

总而言之，iSHIFT 是一项非常有前景的研究，它通过创新的慢-快推理模式和主动感知机制，有效地解决了 GUI 智能体在效率和精度之间的权衡问题，并实现了轻量化设计。这为未来更智能、更易于部署的 MLLMs 在实际应用中开辟了新的道路。

**Key Findings:**

- In this work, we introduce iSHIFT: Implicit Slow-fast Hybrid Inference with Flexible Tokens, a lightweight agent that integrates latent thinking (implicit chain-of-thought) with a perception control module.
- Despite its compact 2.5B size, iSHIFT matches state-of-the-art performance on multiple benchmark datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.22009v1)
- [arXiv](https://arxiv.org/abs/2512.22009v1)

---

<a id='2512.21999v1'></a>
## [Look Closer! An Adversarial Parametric Editing Framework for Hallucination Mitigation in VLMs](https://arxiv.org/abs/2512.21999v1)

**Authors:** Jiayu Hu, Beibei Li, Jiangwei Xia, Yanjun Qin, Bing Ji, Zhongshi He

**Published:** 2025-12-26

**Categories:** cs.CV, cs.LG

**Abstract:**

While Vision-Language Models (VLMs) have garnered increasing attention in the AI community due to their promising practical applications, they exhibit persistent hallucination issues, generating outputs misaligned with visual inputs. Recent studies attribute these hallucinations to VLMs' over-reliance on linguistic priors and insufficient visual feature integration, proposing heuristic decoding calibration strategies to mitigate them. However, the non-trainable nature of these strategies inherently limits their optimization potential. To this end, we propose an adversarial parametric editing framework for Hallucination mitigation in VLMs, which follows an \textbf{A}ctivate-\textbf{L}ocate-\textbf{E}dit \textbf{A}dversarially paradigm. Specifically, we first construct an activation dataset that comprises grounded responses (positive samples attentively anchored in visual features) and hallucinatory responses (negative samples reflecting LLM prior bias and internal knowledge artifacts). Next, we identify critical hallucination-prone parameter clusters by analyzing differential hidden states of response pairs. Then, these clusters are fine-tuned using prompts injected with adversarial tuned prefixes that are optimized to maximize visual neglect, thereby forcing the model to prioritize visual evidence over inherent parametric biases. Evaluations on both generative and discriminative VLM tasks demonstrate the significant effectiveness of ALEAHallu in alleviating hallucinations. Our code is available at https://github.com/hujiayu1223/ALEAHallu.

**Analysis:**

好的，这是一篇关于视觉语言模型（VLM）幻觉缓解的论文的全面中文摘要，重点关注其研究问题、方法、结果、局限性和未来方向。

**论文题目：** Look Closer! An Adversarial Parametric Editing Framework for Hallucination Mitigation in VLMs

**作者：** Jiayu Hu, Beibei Li, Jiangwei Xia, Yanjun Qin, Bing Ji, Zhongshi He

---

**全面摘要**

**1. 研究问题/研究动机：**

视觉语言模型（VLM）在多模态应用中展现出巨大潜力，但普遍存在“幻觉”问题，即生成的文本内容与输入的视觉信息不符。现有研究认为幻觉主要源于VLM过度依赖语言先验知识而忽视视觉特征整合。现有的缓解策略多为启发式解码校准，其不可训练的特性限制了优化潜力。因此，本文旨在提出一种**可优化且高效**的方法来解决VLM的幻觉问题。

**2. 关键创新/方法论贡献：**

本文提出了一种名为 **ALEAHallu**（Activate-Locate-Edit Adversarially）的对抗性参数编辑框架，用于缓解VLM的幻觉问题。其核心创新在于：

*   **激活-定位-编辑（Activate-Locate-Edit）的对抗性范式：**
    *   **激活数据集构建：** 创造一个包含“接地响应”（视觉特征锚定）和“幻觉响应”（语言先验偏见）的配对数据集。
    *   **编辑区域定位：** 通过分析正负响应对在模型隐藏状态下的差异，识别出与幻觉最相关的关键参数簇。
    *   **对抗性参数编辑：** 利用对抗性调优的前缀（Adversarial Prefix Tuning）来微调这些关键参数簇。该前缀被优化以最大化模型对视觉信息的“忽视”，从而迫使模型在标准提示下更优先考虑视觉证据，而非固有的参数偏见。
*   **对抗性前缀调优（Adversarial Prefix Tuning）：** 提出了一种自动优化对抗性前缀的方法，避免了手动设计的前缀的局限性，使其成为一个可学习的组件。
*   **参数高效性：** 该方法仅编辑少量关键参数簇，而非全局微调，从而在不增加额外推理开销的情况下实现幻觉缓解。

**3. 主要结果与意义：**

*   **显著的幻觉缓解效果：** 在图像描述生成和视觉问答（VQA）等生成性和判别性任务上，ALEAHallu均显著降低了幻觉率。
*   **提升视觉注意力：** 实验表明，ALEAHallu能够有效提升VLM对图像内容的关注度，使模型更倾向于依赖视觉信息。
*   **跨数据集泛化能力：** 在不同数据集上的实验证明了ALEAHallu的泛化能力，表明其编辑效果可以跨越不同数据集。
*   **高效性：** 由于仅编辑少量参数且不增加推理开销，ALEAHallu在实际应用中具有很高的效率。
*   **意义：** 本研究为解决VLM幻觉问题提供了一种新颖且有效的解决方案，有助于提升VLM在医疗、教育、媒体等高风险领域的可靠性和安全性。

**4. 提及的局限性：**

论文中并未明确提及ALEAHallu的局限性，但从其方法论和实验结果来看，可以推测：

*   **数据集依赖性：** 激活数据集的质量和规模可能影响编辑效果。
*   **过度缓解的风险：** 虽然论文强调了平衡，但过度缓解幻觉可能在某些创意领域限制模型的想象力。
*   **对特定模型架构的适应性：** 该方法主要针对Transformer架构的VLM，其在其他架构上的适用性有待验证。

**5. 潜在的未来研究方向：**

*   **课程学习（Curriculum Learning）：** 论文作者计划集成课程学习，以进一步探索知识编辑在幻觉抑制方面的性能上限。
*   **更广泛的应用领域：** 将该方法应用于更多下游任务和更复杂的VLM架构。
*   **平衡事实性与创造性：** 在缓解幻觉的同时，探索如何更好地平衡事实准确性与模型的创造性表达。
*   **可解释性增强：** 进一步研究ALEAHallu如何影响模型的内部机制，以提供更深入的可解释性。

---

总而言之，这篇论文提出了一种创新的对抗性参数编辑框架ALEAHallu，通过激活-定位-编辑的范式，有效地解决了VLM的幻觉问题，并在多个任务上取得了显著成果，为提升VLM的可靠性提供了重要贡献。

**Key Findings:**

- To this end, we propose an adversarial parametric editing framework for Hallucination mitigation in VLMs, which follows an \textbf{A}ctivate-\textbf{L}ocate-\textbf{E}dit \textbf{A}dversarially paradigm.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.21999v1)
- [arXiv](https://arxiv.org/abs/2512.21999v1)

---

