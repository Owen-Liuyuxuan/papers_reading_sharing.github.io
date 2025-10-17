time: 20251017

# Arxiv Computer Vision Papers - 2025-10-17

## Executive Summary

好的，这是一份针对2025年10月15日Arxiv计算机视觉论文的每日报告执行摘要，旨在帮助忙碌的研究人员快速了解领域最新进展。

---

**每日Arxiv计算机视觉论文执行摘要 (2025年10月15日)**

**概述与主要趋势：**

今日Arxiv论文展现了计算机视觉领域在**多模态理解与生成、3D场景理解与重建、以及具身智能与交互**方面的显著进展。核心趋势包括：

1.  **多模态融合与统一：** 视觉-语言模型继续向更深层次的融合迈进，旨在实现原生级的视觉-语言原语，并探索在数学推理、运动生成等复杂任务中的应用。
2.  **3D场景的动态与交互理解：** 研究重点从静态3D重建扩展到动态场景的理解（4D重建）、3D视觉定位以及与场景交互的生成。
3.  **可控生成与具身智能：** 论文强调了在视频、运动和手物交互等生成任务中实现精细化控制的重要性，并开始利用真实世界数据进行具身智能（如碰撞预测）的研究。
4.  **Prompt工程的深化：** Prompt-based方法在大型视觉模型中的应用日益广泛，表明其作为一种高效适应策略的重要性。

**特别显著或创新的论文：**

*   **"From Pixels to Words -- Towards Native Vision-Language Primitives at Scale" (Haiwen Diao et al.)**: 这篇论文可能代表了视觉-语言模型发展的一个重要方向，即超越简单的特征融合，探索更深层次的“原生”视觉-语言理解单元，有望为未来的多模态AI奠定基础。
*   **"C4D: 4D Made from 3D through Dual Correspondences" (Shizun Wang et al.)**: 提出了一种从3D数据构建4D（3D+时间）场景的新方法，这对于理解和建模动态世界具有开创性意义，可能在自动驾驶、机器人和虚拟现实等领域有广泛应用。
*   **"MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning" (Weikang Shi et al.)**: 将视觉链式思考（Visual Chain-of-Thought）引入多模态数学推理，这对于提升AI在复杂推理任务中的可解释性和准确性至关重要，是迈向更高级认知智能的一步。
*   **"OmniMotion: Multimodal Motion Generation with Continuous Masked Autoregression" (Zhe Li et al.)**: 提出了一种多模态运动生成框架，其连续掩码自回归机制可能在生成高质量、多样化运动方面表现出色，对动画、游戏和机器人领域有直接影响。

**新兴研究方向或技术：**

*   **原生视觉-语言原语 (Native Vision-Language Primitives):** 旨在构建更深层次、更统一的视觉-语言理解单元，而非简单的特征拼接。
*   **4D场景重建与理解 (4D Scene Reconstruction & Understanding):** 从静态3D向动态3D+时间维度扩展，以更好地捕捉真实世界的复杂性。
*   **视觉链式思考 (Visual Chain-of-Thought):** 将推理过程可视化，提高多模态模型在复杂任务（如数学推理）中的可解释性和性能。
*   **连续掩码自回归 (Continuous Masked Autoregression):** 一种在生成任务中实现高质量、多样化输出的有效机制，尤其适用于运动生成。
*   **3D场景Prompting (3D Scene Prompting):** 将Prompting技术从2D图像扩展到3D场景，以实现更精细的视频生成控制。
*   **具身智能中的真实世界数据应用 (Real-World Data for Embodied AI):** 利用行车记录仪等真实世界数据进行碰撞预测，推动具身智能在实际应用中的鲁棒性。

**建议阅读全文的论文：**

对于希望深入了解前沿进展的研究人员，建议优先阅读以下论文：

1.  **"From Pixels to Words -- Towards Native Vision-Language Primitives at Scale" (Haiwen Diao et al.)**: 了解多模态融合的未来方向。
2.  **"C4D: 4D Made from 3D through Dual Correspondences" (Shizun Wang et al.)**: 掌握4D场景重建的新范式。
3.  **"MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning" (Weikang Shi et al.)**: 探索多模态推理和可解释性的新方法。
4.  **"OmniMotion: Multimodal Motion Generation with Continuous Masked Autoregression" (Zhe Li et al.)**: 关注多模态生成和新型自回归机制。
5.  **"Prompt-based Adaptation in Large-scale Vision Models: A Survey" (Xi Xiao et al.)**: 对于希望了解Prompt工程在大型视觉模型中应用现状和未来趋势的研究人员，这篇综述是极佳的起点。

---

这份摘要旨在提供一个高层次的概览，帮助您快速识别与您研究兴趣最相关的论文。

---

## Table of Contents

1. [Prompt-based Adaptation in Large-scale Vision Models: A Survey](#2510.13219v1)
2. [From Pixels to Words -- Towards Native Vision-Language Primitives at Scale](#2510.14979v1)
3. [ChangingGrounding: 3D Visual Grounding in Changing Scenes](#2510.14965v1)
4. [C4D: 4D Made from 3D through Dual Correspondences](#2510.14960v1)
5. [MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning](#2510.14958v1)
6. [OmniMotion: Multimodal Motion Generation with Continuous Masked Autoregression](#2510.14954v1)
7. [3D Scene Prompting for Scene-Consistent Camera-Controllable Video Generation](#2510.14945v1)
8. [MaskCaptioner : Learning to Jointly Segment and Caption Object Trajectories in Videos](#2510.14904v1)
9. [BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data](#2510.14876v1)
10. [TOUCH: Text-guided Controllable Generation of Free-Form Hand-Object Interactions](#2510.14874v1)

---

## Papers

<a id='2510.13219v1'></a>
## [Prompt-based Adaptation in Large-scale Vision Models: A Survey](https://arxiv.org/abs/2510.13219v1)

**Authors:** Xi Xiao, Yunbei Zhang, Lin Zhao, Yiyang Liu, Xiaoying Liao, Zheda Mai, Xingjian Li, Xiao Wang, Hao Xu, Jihun Hamm, Xue Lin, Min Xu, Qifan Wang, Tianyang Wang, Cheng Han

**Published:** 2025-10-15

**Categories:** cs.CV

**Abstract:**

In computer vision, Visual Prompting (VP) and Visual Prompt Tuning (VPT) have
recently emerged as lightweight and effective alternatives to full fine-tuning
for adapting large-scale vision models within the ``pretrain-then-finetune''
paradigm. However, despite rapid progress, their conceptual boundaries remain
blurred, as VP and VPT are frequently used interchangeably in current research,
reflecting a lack of systematic distinction between these techniques and their
respective applications. In this survey, we revisit the designs of VP and VPT
from first principles, and conceptualize them within a unified framework termed
Prompt-based Adaptation (PA). We provide a taxonomy that categorizes existing
methods into learnable, generative, and non-learnable prompts, and further
organizes them by injection granularity -- pixel-level and token-level. Beyond
the core methodologies, we examine PA's integrations across diverse domains,
including medical imaging, 3D point clouds, and vision-language tasks, as well
as its role in test-time adaptation and trustworthy AI. We also summarize
current benchmarks and identify key challenges and future directions. To the
best of our knowledge, we are the first comprehensive survey dedicated to PA's
methodologies and applications in light of their distinct characteristics. Our
survey aims to provide a clear roadmap for researchers and practitioners in all
area to understand and explore the evolving landscape of PA-related research.

**Analysis:**

好的，这是一篇关于“Prompt-based Adaptation in Large-scale Vision Models: A Survey”的中文摘要，涵盖了您要求的所有要点：

**论文摘要：大型视觉模型中的基于提示的适应性：一项综述**

**1. 主要问题或研究问题：**
该论文旨在解决计算机视觉领域中，视觉提示（Visual Prompting, VP）和视觉提示微调（Visual Prompt Tuning, VPT）这两种轻量级模型适应方法之间概念边界模糊的问题。尽管它们在大型视觉模型的“预训练-微调”范式中取得了快速进展并被广泛应用，但当前研究中常将二者互换使用，缺乏系统性的区分，这阻碍了对它们各自技术特点和适用场景的深入理解。本综述的核心目标是提供一个统一的框架，清晰地界定和分类这些方法。

**2. 关键创新或方法论贡献：**
*   **统一框架与分类法：** 论文将VP和VPT从第一性原理出发，概念化为一个统一的框架，称之为“基于提示的适应性”（Prompt-based Adaptation, PA）。
*   **详细分类：** 提出了一种全面的分类法，将现有方法分为可学习（learnable）、生成式（generative）和不可学习（non-learnable）提示。
*   **注入粒度区分：** 进一步根据提示的注入粒度（像素级和Token级）对方法进行组织，这是区分VP和VPT的关键。VP主要在输入空间进行像素级修改，而VPT则在模型内部的Token序列中注入可学习的提示。
*   **应用领域整合：** 综述了PA在各种不同领域中的应用，包括医学影像、3D点云、视觉-语言任务，以及其在测试时间适应（Test-Time Adaptation）和可信赖AI（Trustworthy AI）中的作用。

**3. 主要结果及其意义：**
*   **PA的有效性：** 论文强调PA作为全量微调的轻量级且有效替代方案，在数据受限、资源受限和动态适应等多种场景下表现出显著效果。
*   **效率提升：** PA通过仅更新少量参数（VP可能不更新参数，VPT更新少量提示Token和头部），显著降低了计算和存储成本，尤其适用于商品硬件和延迟敏感的部署环境。
*   **鲁棒性与泛化能力：** 提示机制能够帮助模型更好地应对领域漂移、分布变化，并在少样本、零样本等数据受限场景下提高泛化能力。
*   **可信赖AI的贡献：** PA在提升模型鲁棒性、缓解偏见和确保隐私安全方面发挥作用，是构建可信赖AI系统的重要组成部分。

**4. 论文中提及的局限性：**
*   **训练开销与稳定性：** 尽管PA减少了参数效率，但总训练时长可能因超参数搜索和初始化不稳定性而增加。VP方法可能表现出不稳定性，对提示配置的微小扰动敏感。
*   **推理延迟：** 额外的提示组件（无论是VP的输入空间修改还是VPT的Token注入）可能导致推理延迟和额外的内存消耗。
*   **真实世界环境评估不足：** 当前PA方法的评估主要依赖标准化学术基准，这些基准可能无法准确反映真实世界场景的复杂性和分布变化。

**5. 潜在的未来研究方向：**
*   **训练效率与稳定性：** 进一步研究训练捷径、检测和纠正训练不稳定性的策略。
*   **混合方法：** 探索结合VP和VPT优势的混合方法，例如使用生成器提供初始空间提示（VP），同时使用可学习的条件感知Token（VPT）来引导模型内部特征提取。
*   **真实世界部署：** 优先开发能够应对复杂异构视觉上下文的鲁棒方法，以弥合学术研究与实际部署之间的差距。
*   **安全对齐：** 持续研究如何检测和纠正模型行为中的偏差、恶意内容生成等问题，确保PA方法符合人类价值观和目标。
*   **理论基础：** 深入探索PA如何诱导模型行为变化、视觉提示学习了什么以及PA方法在不同适应设置中的有效性等理论问题。

总而言之，这篇综述为研究人员和实践者提供了一个关于大型视觉模型中基于提示的适应性方法的清晰路线图，系统地梳理了其概念、方法、应用、挑战和未来方向，对于推动该领域的发展具有重要意义。

**Key Findings:**

- To the
best of our knowledge, we are the first comprehensive survey dedicated to PA's
methodologies and applications in light of their distinct characteristics.
- Our
survey aims to provide a clear roadmap for researchers and practitioners in all
area to understand and explore the evolving landscape of PA-related research.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13219v1)
- [arXiv](https://arxiv.org/abs/2510.13219v1)

---

<a id='2510.14979v1'></a>
## [From Pixels to Words -- Towards Native Vision-Language Primitives at Scale](https://arxiv.org/abs/2510.14979v1)

**Authors:** Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, Ziwei Liu

**Published:** 2025-10-16

**Categories:** cs.CV, cs.AI

**Abstract:**

The edifice of native Vision-Language Models (VLMs) has emerged as a rising
contender to typical modular VLMs, shaped by evolving model architectures and
training paradigms. Yet, two lingering clouds cast shadows over its widespread
exploration and promotion: (-) What fundamental constraints set native VLMs
apart from modular ones, and to what extent can these barriers be overcome? (-)
How to make research in native VLMs more accessible and democratized, thereby
accelerating progress in the field. In this paper, we clarify these challenges
and outline guiding principles for constructing native VLMs. Specifically, one
native VLM primitive should: (i) effectively align pixel and word
representations within a shared semantic space; (ii) seamlessly integrate the
strengths of formerly separate vision and language modules; (iii) inherently
embody various cross-modal properties that support unified vision-language
encoding, aligning, and reasoning. Hence, we launch NEO, a novel family of
native VLMs built from first principles, capable of rivaling top-tier modular
counterparts across diverse real-world scenarios. With only 390M image-text
examples, NEO efficiently develops visual perception from scratch while
mitigating vision-language conflicts inside a dense and monolithic model
crafted from our elaborate primitives. We position NEO as a cornerstone for
scalable and powerful native VLMs, paired with a rich set of reusable
components that foster a cost-effective and extensible ecosystem. Our code and
models are publicly available at: https://github.com/EvolvingLMMs-Lab/NEO.

**Analysis:**

这篇论文“From Pixels to Words -- Towards Native Vision-Language Primitives at Scale”由Haiwen Diao等人撰写，旨在解决原生视觉-语言模型（VLMs）在与传统模块化VLM竞争时所面临的挑战，并推动该领域的研究进展。

**1. 主要问题或研究问题：**
论文主要探讨了原生VLM在广泛探索和推广中面临的两个核心问题：
*   原生VLM与模块化VLM之间的根本区别是什么，以及如何克服这些障碍？
*   如何使原生VLM的研究更易于访问和民主化，从而加速该领域的进展？

**2. 关键创新或方法论贡献：**
论文提出了构建原生VLM的指导原则，并推出了一个名为**NEO**的新型原生VLM家族，其关键创新包括：
*   **原生VLM基元（Native VLM Primitive）：** NEO设计了一个统一的视觉-语言基元，能够在一个模块中同时整合跨模态的编码、对齐和推理。这包括：
    *   灵活的位置编码方案，有效泛化到动态空间结构。
    *   多头原生注意力（MHNA），共同处理视觉-文本连接。
    *   原生旋转位置嵌入（Native-RoPE），具有模态特定频率，兼容预训练LLM权重并吸收原始VE的交互模式。
*   **预缓冲区（Pre-Buffer）和后LLM（Post-LLM）架构：** 为了高效地扩展视觉训练并确保像素-词语对齐的一致性，NEO将骨干网络划分为预缓冲区和后LLM层。这种设计在预训练阶段使预训练LLM能够引导视觉学习，并在后续阶段建立连贯的相关性。在训练后期，这种划分会消失，形成一个统一的架构。
*   **端到端训练范式：** NEO通过简化的端到端训练，在仅3.9亿图像-文本示例上，从头开始高效地发展视觉感知能力，同时缓解密集、单片模型内部的视觉-语言冲突。

**3. 主要结果及其意义：**
*   **竞争性表现：** 尽管训练数据和计算资源相对有限，NEO在2B和8B规模上均展现出高度竞争性的性能，在多个基准测试中与顶级的模块化VLM（如Qwen2-VL、InternVL2.5等）相媲美，甚至超越了许多使用更多训练资源的原生VLM。
*   **高效的像素-词语对齐和推理：** NEO通过其统一的基元设计和训练策略，能够从头开始将视觉输入与文本特征对齐，并支持复杂的视觉推理。
*   **可重用组件和生态系统：** NEO提供了丰富的可重用组件，促进了成本效益和可扩展的生态系统，降低了原生VLM开发的门槛。
*   **对未来多模态系统的启示：** 论文表明，下一代多模态系统可以源于原生、统一且本质上多模态的架构。

**4. 论文中提及的局限性：**
*   **数据和计算资源限制：** NEO的性能仍受限于稀缺的训练数据和有限的计算资源，尤其是在知识密集型和OCR领域。
*   **无法完全从头训练：** 受限于当前的文本语料库和计算资源，NEO无法在不依赖现有LLM初始化的情况下完全从头训练一个完全原生的模型。这限制了缓解语言模态主导地位可能带来的潜在偏差的能力。
*   **在某些任务上的不足：** NEO在知识/OCR密集型任务（如MMMU、InfoVQA和TextVQA）上表现略逊一筹。

**5. 潜在的未来研究方向：**
*   **持续投入和扩展：** 持续投入大量资源，尤其是在预训练阶段，以充分释放NEO的性能潜力。
*   **开放关键组件：** 在中间开发阶段选择性地开放关键组件，以降低未来研究人员的后续训练成本，并吸引更多原生视觉-语言模型研究。
*   **探索全谱模型能力：** 扩展模型规模是推动实际应用的关键因素，NEO-2.2B的性能已接近同等容量的模块化VLM，表明0.6到80亿参数范围内的模型设计哲学已趋于成熟。
*   **架构和应用升级：** 将NEO视为一个视觉-语言智能的新范式，利用端到端训练和统一架构，消除手动施加的偏差和扩展复杂性。将NEO扩展到视频生成、长视频理解和具身AI等领域。
*   **去中心化多模态训练：** 探索从头开始进行多模态训练，以进一步揭示原生VLM架构的性能上限。

**Key Findings:**

- Hence, we launch NEO, a novel family of
native VLMs built from first principles, capable of rivaling top-tier modular
counterparts across diverse real-world scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14979v1)
- [arXiv](https://arxiv.org/abs/2510.14979v1)

---

<a id='2510.14965v1'></a>
## [ChangingGrounding: 3D Visual Grounding in Changing Scenes](https://arxiv.org/abs/2510.14965v1)

**Authors:** Miao Hu, Zhiwei Huang, Tai Wang, Jiangmiao Pang, Dahua Lin, Nanning Zheng, Runsen Xu

**Published:** 2025-10-16

**Categories:** cs.CV

**Abstract:**

Real-world robots localize objects from natural-language instructions while
scenes around them keep changing. Yet most of the existing 3D visual grounding
(3DVG) method still assumes a reconstructed and up-to-date point cloud, an
assumption that forces costly re-scans and hinders deployment. We argue that
3DVG should be formulated as an active, memory-driven problem, and we introduce
ChangingGrounding, the first benchmark that explicitly measures how well an
agent can exploit past observations, explore only where needed, and still
deliver precise 3D boxes in changing scenes. To set a strong reference point,
we also propose Mem-ChangingGrounder, a zero-shot method for this task that
marries cross-modal retrieval with lightweight multi-view fusion: it identifies
the object type implied by the query, retrieves relevant memories to guide
actions, then explores the target efficiently in the scene, falls back when
previous operations are invalid, performs multi-view scanning of the target,
and projects the fused evidence from multi-view scans to get accurate object
bounding boxes. We evaluate different baselines on ChangingGrounding, and our
Mem-ChangingGrounder achieves the highest localization accuracy while greatly
reducing exploration cost. We hope this benchmark and method catalyze a shift
toward practical, memory-centric 3DVG research for real-world applications.
Project page: https://hm123450.github.io/CGB/ .

**Analysis:**

以下是Miao Hu等人撰写的论文“ChangingGrounding: 3D Visual Grounding in Changing Scenes”的全面摘要：

**1. 主要问题或研究问题：**
该论文旨在解决现实世界机器人面临的一个核心挑战：如何在场景不断变化的环境中，根据自然语言指令准确有效地定位目标物体。现有的3D视觉定位（3DVG）方法大多假设场景是静态的，依赖于完整且最新的点云，这导致了昂贵的重复扫描和部署障碍。因此，研究问题是如何将3DVG重新定义为一个主动的、记忆驱动的问题，使机器人能够利用过去的观察、仅在需要时进行探索，并仍能在动态场景中提供精确的3D边界框。

**2. 关键创新或方法论贡献：**
*   **ChangingGrounding基准：** 论文引入了首个专门用于衡量机器人在变化场景中3DVG性能的基准。该基准明确评估了代理利用过去观察、高效探索和精确3D定位的能力。
*   **Mem-ChangingGrounder方法：** 提出了一种零样本（zero-shot）方法来解决此任务，该方法结合了跨模态检索和轻量级多视图融合。其核心流程包括：
    *   **查询分类：** 识别查询中隐含的物体类型。
    *   **记忆检索：** 检索相关记忆以指导行动。
    *   **高效探索：** 在场景中高效探索目标，并在先前操作无效时进行回退。
    *   **多视图扫描与融合：** 对目标进行多视图扫描，并将融合的证据投影以获得准确的物体边界框。
*   **行动策略（OSS和SRAS）：** 引入了全向场景扫描器（Omnidirectional Scene Scanner, OSS）和空间关系感知扫描器（Spatial Relation Aware Scanner, SRAS）两种行动策略，用于在未知场景中探索和定位目标物体。
*   **数据集构建：** 基于3RScan数据集构建了一个新的ChangingGrounding数据集，包含空间关系描述、RGB-D图像、相机姿态和网格文件，以模拟物体移动和生成新观察。

**3. 主要结果及其意义：**
*   **Mem-ChangingGrounder的优越性：** 在ChangingGrounding基准测试中，Mem-ChangingGrounder在低分辨率和高分辨率设置下均实现了最高的定位精度（Acc@0.25分别为29.2%和36.8%），同时显著降低了探索成本（动作成本Ca和运动成本Cm）。
*   **效率与准确性的平衡：** 实验结果表明，该方法在准确性和效率之间取得了卓越的平衡，通过在移动前咨询记忆并执行有针对性的短动作，避免了长时间的探索循环。
*   **基线比较：** 与“漫游定位”（Wandering Grounding）、“中心旋转定位”（Central Rotation Grounding）和“仅记忆定位”（Memory-Only Grounding）等基线方法相比，Mem-ChangingGrounder在准确性和成本方面均表现出显著优势。
*   **记忆和探索的重要性：** 消融研究证实了记忆策略和回退机制的有效性，以及多视图投影对提高定位准确性的贡献。

**4. 论文中提到的局限性：**
*   **CGB基准的局限性：** 当前数据集仅模拟目标及其周围环境的相对位置变化，未考虑光照变化、物体外观属性（颜色、材料、变形）或动态场景交互等关键因素。此外，缺乏“物体A在物体B前面”等绝对空间关系描述。
*   **MCG方法的局限性：**
    *   **VLM能力依赖：** MCG严重依赖底层视觉-语言模型（VLM）的能力，其性能受VLM能力和现实世界场景复杂性的影响。
    *   **渲染图像的噪声：** 渲染过程引入的噪声（如RGB图像中的伪影、深度图的不准确性）以及渲染图像与真实图像之间的固有差异，可能影响定位准确性。
    *   **2D模型引入的噪声：** MCG依赖2D物体检测器和分割网络，这些模型可能存在漏检、误报、边界框不精确和分割错误，这些缺陷会影响最终的定位准确性。

**5. 潜在的未来研究方向：**
*   **提高VLM的鲁棒性：** 开发更鲁棒的视觉-语言模型，以处理复杂的现实世界视觉信息并减少噪声影响。
*   **增强多模态集成：** 探索更好地集成多模态数据（视觉、语言和空间信息）以提高定位准确性的方法。
*   **扩展基准多样性：** 通过增加更多样化的场景（包括光照变化、物体外观和动态交互）来扩展CGB基准。
*   **减少渲染数据中的噪声：** 研究最小化渲染过程中引入的噪声，并弥合真实图像与渲染图像之间差距的方法。
*   **推进2D到3D投影技术：** 提高2D物体检测和分割模型的准确性和可靠性，以增强整体定位性能。

总而言之，这篇论文通过引入ChangingGrounding基准和Mem-ChangingGrounder方法，为动态场景中的3D视觉定位任务开辟了新方向。它强调了记忆和高效探索在实际机器人应用中的重要性，并为未来该领域的研究奠定了坚实基础。

**Key Findings:**

- We argue that
3DVG should be formulated as an active, memory-driven problem, and we introduce
ChangingGrounding, the first benchmark that explicitly measures how well an
agent can exploit past observations, explore only where needed, and still
deliver precise 3D boxes in changing scenes.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14965v1)
- [arXiv](https://arxiv.org/abs/2510.14965v1)

---

<a id='2510.14960v1'></a>
## [C4D: 4D Made from 3D through Dual Correspondences](https://arxiv.org/abs/2510.14960v1)

**Authors:** Shizun Wang, Zhenxiang Jiang, Xingyi Yang, Xinchao Wang

**Published:** 2025-10-16

**Categories:** cs.CV, cs.AI

**Abstract:**

Recovering 4D from monocular video, which jointly estimates dynamic geometry
and camera poses, is an inevitably challenging problem. While recent
pointmap-based 3D reconstruction methods (e.g., DUSt3R) have made great
progress in reconstructing static scenes, directly applying them to dynamic
scenes leads to inaccurate results. This discrepancy arises because moving
objects violate multi-view geometric constraints, disrupting the
reconstruction. To address this, we introduce C4D, a framework that leverages
temporal Correspondences to extend existing 3D reconstruction formulation to
4D. Specifically, apart from predicting pointmaps, C4D captures two types of
correspondences: short-term optical flow and long-term point tracking. We train
a dynamic-aware point tracker that provides additional mobility information,
facilitating the estimation of motion masks to separate moving elements from
the static background, thus offering more reliable guidance for dynamic scenes.
Furthermore, we introduce a set of dynamic scene optimization objectives to
recover per-frame 3D geometry and camera parameters. Simultaneously, the
correspondences lift 2D trajectories into smooth 3D trajectories, enabling
fully integrated 4D reconstruction. Experiments show that our framework
achieves complete 4D recovery and demonstrates strong performance across
multiple downstream tasks, including depth estimation, camera pose estimation,
and point tracking. Project Page: https://littlepure2333.github.io/C4D

**Analysis:**

好的，这是一篇关于Shizun Wang, Zhenxiang Jiang, Xingyi Yang, Xinchao Wang撰写的论文“C4D: 4D Made from 3D through Dual Correspondences”的全面摘要。

**论文摘要：C4D: 通过双重对应从3D构建4D**

这篇论文《C4D: 4D Made from 3D through Dual Correspondences》提出了一种新颖的框架C4D，旨在解决从单目视频中恢复动态场景4D表示的挑战性问题。4D重建不仅涉及估计动态场景的几何形状，还包括相机姿态和3D点跟踪。

**1. 主要问题或研究问题**
现有的基于点图（pointmap-based）的3D重建方法（如DUSt3R）在静态场景中取得了显著进展，但直接应用于动态场景时，由于移动物体违反了多视角几何约束，导致重建结果不准确。核心问题是如何在动态场景中实现准确、平滑且时间一致的4D重建，包括每帧3D几何、相机姿态和3D点轨迹。

**2. 关键创新或方法论贡献**
C4D框架通过引入“双重对应”（Dual Correspondences）将现有的3D重建扩展到4D，其主要创新包括：

*   **动态感知点跟踪器（DynPT）**：C4D训练了一个能够预测点在世界坐标系中是否动态的跟踪器。这超越了传统2D点跟踪器仅预测位置和遮挡的能力，为区分相机运动和物体自身运动提供了关键信息。
*   **对应引导的运动掩码估计**：利用短时光流（optical flow）和长时点跟踪（DynPT）两种对应，C4D能够生成可靠的运动掩码。这些掩码用于将动态元素从静态背景中分离出来，从而在静态区域进行更准确的相机参数估计和几何重建。
*   **对应辅助的动态场景优化目标**：C4D引入了一系列新的优化目标，包括：
    *   **相机运动对齐（CMA）**：确保估计的自我运动与静态区域的光流一致。
    *   **相机轨迹平滑度（CTS）**：通过惩罚连续帧之间相机旋转和平移的突然变化，强制相机运动平滑。
    *   **点轨迹平滑度（PTS）**：通过对稀疏3D点轨迹进行自适应加权的一维卷积平滑，然后通过线性混合位移（LBD）将其传播到所有点，确保3D点轨迹的时间平滑性。
*   **完全集成4D重建**：通过联合预测点图和上述双重对应，C4D将2D轨迹提升为平滑的3D轨迹，实现了每帧3D几何和相机参数的完全集成4D恢复。

**3. 主要结果及其意义**
实验结果表明，C4D框架在动态场景重建方面表现出色，并在多个下游任务中展示了强大的性能：

*   **深度估计**：C4D在Sintel、Bonn和KITTI数据集上实现了竞争性的深度估计性能，尤其在尺度不变对齐方面表现最佳。
*   **相机姿态估计**：C4D在Sintel、TUM-dynamics和ScanNet数据集上显著提高了相机姿态估计的准确性，甚至优于一些专门的视觉里程计方法。
*   **点跟踪**：尽管DynPT需要预测额外的“移动性”信息，但其在TAP-Vid和Kubric数据集上仍能与最先进的TAP方法保持竞争性性能，并准确预测点的动态状态。
*   **时间平滑性**：C4D通过PTS目标显著改善了视频深度和3D点轨迹的时间平滑性，有效减少了现有方法中常见的闪烁伪影。
*   **运动掩码准确性**：C4D生成的运动掩码比MonST3R等方法更准确和完整，尤其在复杂动态场景中。

这些结果证明了C4D在处理动态场景时的有效性和鲁棒性，为单目视频4D重建领域树立了新的基准。

**4. 论文中提到的局限性**
论文中没有明确提及C4D框架的显著局限性。然而，从方法论和实验设置中可以推断出一些潜在的限制：

*   **计算成本**：虽然论文提到通过稀疏场景图和滑动窗口策略来降低计算成本，但联合优化点图、光流、点跟踪和多个优化目标，可能仍然具有较高的计算复杂度，尤其是在处理超长视频时。
*   **合成数据依赖**：DynPT的训练依赖于Kubric等合成数据集，这些数据集提供了地面真实移动性标签。尽管合成数据有助于控制变量，但其在真实世界复杂场景中的泛化能力可能仍需进一步验证。
*   **模型权重初始化**：C4D利用了预训练的DUSt3R模型权重，这意味着其性能可能部分依赖于这些基础模型的质量和泛化能力。
*   **超参数敏感性**：优化过程中涉及多个损失权重（WGA, WCMA, WCTS, WPTS）和一些超参数（如平滑因子λ、核大小k），这些参数的选择可能对最终性能有影响。

**5. 潜在的未来研究方向**
基于C4D的贡献和潜在局限性，未来的研究方向可能包括：

*   **实时性能提升**：进一步优化C4D的计算效率，使其能够实现更接近实时的4D重建，这对于机器人和增强现实等应用至关重要。
*   **更强的泛化能力**：探索在更多样化、更具挑战性的真实世界动态场景数据集上训练和评估C4D，以提高其在未见场景中的泛化能力。
*   **自监督或弱监督学习**：减少对地面真实移动性标签的依赖，开发更强大的自监督或弱监督方法来训练动态感知点跟踪器和运动掩码估计。
*   **集成语义信息**：将语义分割或物体检测等高级语义信息集成到C4D框架中，以更好地理解场景中的动态元素，从而可能提高运动掩码和点轨迹的准确性。
*   **多模态输入**：探索结合其他传感器数据（如IMU、激光雷达）来增强4D重建的鲁棒性和准确性，尤其是在光照不足或纹理稀疏的挑战性环境中。
*   **交互式4D重建**：开发允许用户在4D重建过程中进行交互和修正的工具，以处理复杂或模糊的动态场景。

总而言之，C4D为单目视频4D重建提供了一个全面且高性能的解决方案，通过巧妙地结合短时和长时对应，有效解决了动态场景中的挑战，并为未来的研究奠定了坚实的基础。

**Key Findings:**

- To address this, we introduce C4D, a framework that leverages
temporal Correspondences to extend existing 3D reconstruction formulation to
4D.
- Furthermore, we introduce a set of dynamic scene optimization objectives to
recover per-frame 3D geometry and camera parameters.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14960v1)
- [arXiv](https://arxiv.org/abs/2510.14960v1)

---

<a id='2510.14958v1'></a>
## [MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning](https://arxiv.org/abs/2510.14958v1)

**Authors:** Weikang Shi, Aldrich Yu, Rongyao Fang, Houxing Ren, Ke Wang, Aojun Zhou, Changyao Tian, Xinyu Fu, Yuxuan Hu, Zimu Lu, Linjiang Huang, Si Liu, Rui Liu, Hongsheng Li

**Published:** 2025-10-16

**Categories:** cs.CV, cs.CL

**Abstract:**

While Large Language Models (LLMs) have excelled in textual reasoning, they
struggle with mathematical domains like geometry that intrinsically rely on
visual aids. Existing approaches to Visual Chain-of-Thought (VCoT) are often
limited by rigid external tools or fail to generate the high-fidelity,
strategically-timed diagrams necessary for complex problem-solving. To bridge
this gap, we introduce MathCanvas, a comprehensive framework designed to endow
unified Large Multimodal Models (LMMs) with intrinsic VCoT capabilities for
mathematics. Our approach consists of two phases. First, a Visual Manipulation
stage pre-trains the model on a novel 15.2M-pair corpus, comprising 10M
caption-to-diagram pairs (MathCanvas-Imagen) and 5.2M step-by-step editing
trajectories (MathCanvas-Edit), to master diagram generation and editing.
Second, a Strategic Visual-Aided Reasoning stage fine-tunes the model on
MathCanvas-Instruct, a new 219K-example dataset of interleaved visual-textual
reasoning paths, teaching it when and how to leverage visual aids. To
facilitate rigorous evaluation, we introduce MathCanvas-Bench, a challenging
benchmark with 3K problems that require models to produce interleaved
visual-textual solutions. Our model, BAGEL-Canvas, trained under this
framework, achieves an 86% relative improvement over strong LMM baselines on
MathCanvas-Bench, demonstrating excellent generalization to other public math
benchmarks. Our work provides a complete toolkit-framework, datasets, and
benchmark-to unlock complex, human-like visual-aided reasoning in LMMs. Project
Page: https://mathcanvas.github.io/

**Analysis:**

以下是Weikang Shi等人撰写的论文“MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning”的摘要：

**1. 主要问题或研究问题：**
大型语言模型（LLMs）在文本推理方面表现出色，但在几何学等数学领域中，由于其本质上依赖视觉辅助，LLMs表现不佳。现有的视觉思维链（VCoT）方法通常受限于僵硬的外部工具，或无法生成高保真度、策略性适时的图表，从而难以解决复杂的数学问题。该研究旨在弥合这一鸿沟，赋予统一大型多模态模型（LMMs）内在的VCoT能力，以进行数学推理。

**2. 关键创新或方法论贡献：**
该论文引入了MathCanvas，一个全面的框架，旨在赋予LMMs内在的VCoT能力。其方法分为两个阶段：
*   **视觉操作（Visual Manipulation）阶段：** 通过在一个包含15.2M对数据的新型语料库上预训练模型，掌握图表生成和编辑能力。该语料库包括10M的“标题-图表对”（MathCanvas-Imagen）和5.2M的“逐步编辑轨迹”（MathCanvas-Edit）。
*   **策略性视觉辅助推理（Strategic Visual-Aided Reasoning）阶段：** 在MathCanvas-Instruct数据集上对模型进行微调，该数据集是一个包含219K示例的交错视觉-文本推理路径数据集，旨在教授模型何时以及如何利用视觉辅助。
*   **MathCanvas-Bench基准测试：** 为了进行严格评估，引入了一个包含3K问题的挑战性基准测试，要求模型生成交错的视觉-文本解决方案。

**3. 主要结果及其意义：**
该研究训练的模型BAGEL-Canvas，在该框架下，在MathCanvas-Bench上比强大的LMM基线取得了86%的相对改进，并展示了对其他公共数学基准的优秀泛化能力。这表明MathCanvas成功地解锁了LMMs中复杂、类人视觉辅助推理的能力。

**4. 论文中提及的局限性：**
摘要中未明确提及论文的局限性，但从其强调“解锁复杂、类人视觉辅助推理”以及“对其他公共数学基准的优秀泛化能力”来看，可能暗示了现有模型在这些方面仍有提升空间，或者在某些特定数学领域（如微积分和向量）的改进相对较小（正文中提到微积分和向量领域的增益较小，可能超出当前视觉增强技术的范围）。

**5. 潜在的未来研究方向：**
该工作提供了一个完整的工具包、框架、数据集和基准测试，为LMMs中复杂、类人视觉辅助推理的未来研究奠定了坚实基础。未来的研究可以进一步探索如何优化模型在特定数学领域的表现，或扩展其在更广泛、更复杂的多模态推理任务中的应用。

**Key Findings:**

- To bridge
this gap, we introduce MathCanvas, a comprehensive framework designed to endow
unified Large Multimodal Models (LMMs) with intrinsic VCoT capabilities for
mathematics.
- Our approach consists of two phases.
- First, a Visual Manipulation
stage pre-trains the model on a novel 15.2M-pair corpus, comprising 10M
caption-to-diagram pairs (MathCanvas-Imagen) and 5.2M step-by-step editing
trajectories (MathCanvas-Edit), to master diagram generation and editing.
- Second, a Strategic Visual-Aided Reasoning stage fine-tunes the model on
MathCanvas-Instruct, a new 219K-example dataset of interleaved visual-textual
reasoning paths, teaching it when and how to leverage visual aids.
- To
facilitate rigorous evaluation, we introduce MathCanvas-Bench, a challenging
benchmark with 3K problems that require models to produce interleaved
visual-textual solutions.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14958v1)
- [arXiv](https://arxiv.org/abs/2510.14958v1)

---

<a id='2510.14954v1'></a>
## [OmniMotion: Multimodal Motion Generation with Continuous Masked Autoregression](https://arxiv.org/abs/2510.14954v1)

**Authors:** Zhe Li, Weihao Yuan, Weichao Shen, Siyu Zhu, Zilong Dong, Chang Xu

**Published:** 2025-10-16

**Categories:** cs.CV

**Abstract:**

Whole-body multi-modal human motion generation poses two primary challenges:
creating an effective motion generation mechanism and integrating various
modalities, such as text, speech, and music, into a cohesive framework. Unlike
previous methods that usually employ discrete masked modeling or autoregressive
modeling, we develop a continuous masked autoregressive motion transformer,
where a causal attention is performed considering the sequential nature within
the human motion. Within this transformer, we introduce a gated linear
attention and an RMSNorm module, which drive the transformer to pay attention
to the key actions and suppress the instability caused by either the abnormal
movements or the heterogeneous distributions within multi-modalities. To
further enhance both the motion generation and the multimodal generalization,
we employ the DiT structure to diffuse the conditions from the transformer
towards the targets. To fuse different modalities, AdaLN and cross-attention
are leveraged to inject the text, speech, and music signals. Experimental
results demonstrate that our framework outperforms previous methods across all
modalities, including text-to-motion, speech-to-gesture, and music-to-dance.
The code of our method will be made public.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zhe Li等人撰写的论文“OmniMotion: Multimodal Motion Generation with Continuous Masked Autoregression”的全面摘要。

---

### 论文摘要：OmniMotion: 多模态连续掩码自回归运动生成

**1. 主要问题或研究问题：**
该论文旨在解决全身多模态人体运动生成领域的两大挑战：一是如何构建一个高效的运动生成机制，二是如何将文本、语音和音乐等多种模态有效地整合到一个统一的框架中。现有的方法通常专注于单一模态或采用离散的掩码或自回归建模，这限制了其泛化能力和生成质量。

**2. 关键创新或方法论贡献：**
OmniMotion 提出了一个新颖的、统一的框架，用于从多种模态（文本、语音、音乐）生成全身人体运动。其核心创新包括：

*   **连续掩码自回归运动Transformer (Continuous Masked Autoregressive Motion Transformer)：** 与以往离散建模不同，该方法采用连续的掩码自回归Transformer。它通过因果注意力机制处理序列运动数据，以捕捉人体运动的顺序性，并预测被掩码的运动片段。
*   **门控线性注意力 (Gated Linear Attention) 和 RMSNorm 模块：** 在Transformer内部，引入了门控线性注意力机制作为自适应特征选择器，使模型能够关注关键动作（如手势切换、大幅度运动），同时抑制不相关或冗余动作（如静止运动）。RMSNorm模块则用于处理多模态输入中异构分布带来的不稳定性，并缓解异常运动（如突然跳跃）引起的梯度不稳定性。
*   **DiT (Diffusion Transformer) 结构的应用：** 为了进一步提升运动生成质量和多模态泛化能力，模型采用DiT结构，将Transformer生成的条件信息扩散到目标运动。在多模态学习阶段，DiT模块结构保持不变并被冻结，仅对掩码Transformer进行微调。
*   **多模态信号融合机制：** 利用 AdaLN (Adaptive Layer Normalization) 和交叉注意力机制，将文本、语音和音乐信号有效地注入到Transformer中，实现多模态条件的融合。

**3. 主要结果及其意义：**
实验结果表明，OmniMotion 框架在所有模态（包括文本到运动、语音到手势和音乐到舞蹈）上均优于现有方法。
*   **文本到运动生成：** 在HumanML3D数据集上，模型在R-Precision (Top-1, 2, 3) 上分别提升了19.3%、13.5%和11.7%，FID分数提升了75.2%，表明其生成的运动具有卓越的保真度和与文本描述的高度对齐。
*   **语音到手势生成：** 在BEAT2数据集上，模型在手部和身体运动生成方面表现出良好的质量和多样性，并能更好地与第一人称语音的节奏对齐。
*   **音乐到舞蹈生成：** 在FineDance数据集上，模型在生成手部动作和身体运动方面略优于现有方法。
*   **消融研究：** 验证了因果注意力、DiT、门控线性机制、RMSNorm和交叉注意力模块对模型性能的积极贡献，尤其是在处理复杂多模态上下文时。

这些结果突显了OmniMotion在统一框架下处理多种模态输入、生成高质量全身人体运动的强大能力，为计算机视觉领域的运动生成任务树立了新的基准。

**4. 论文中提及的局限性：**
论文指出，由于数据集的限制，运动生成模型的自然性和泛化能力仍然有限，尤其是在语音和音乐驱动的运动生成方面。这意味着模型可能在面对未见过或更复杂、更细致的语音/音乐输入时，其生成效果仍有提升空间。

**5. 潜在的未来研究方向：**
尽管论文未明确列出未来研究方向，但从其局限性可以推断出以下几个方向：
*   **扩展数据集规模和多样性：** 解决当前数据集限制，收集更大规模、更丰富多样（包含更多复杂动作、情感表达和风格）的多模态运动数据集，以进一步提升模型的自然性和泛化能力。
*   **更精细化的多模态对齐：** 探索更先进的机制，以实现语音、音乐与运动之间更精细、更准确的时间和语义对齐，尤其是在处理复杂节奏和情感表达时。
*   **实时生成和交互：** 优化模型效率，使其能够支持实时运动生成，并探索与用户进行交互式运动生成的方法。
*   **个性化和风格化运动生成：** 进一步研究如何根据用户的偏好或特定风格生成个性化和风格化的运动，例如通过学习不同舞者的风格或特定角色的动作特征。
*   **结合物理约束：** 引入物理约束或动力学模型，以生成更真实、更符合物理规律的运动，避免不自然的动作。

---

**Key Findings:**

- Unlike
previous methods that usually employ discrete masked modeling or autoregressive
modeling, we develop a continuous masked autoregressive motion transformer,
where a causal attention is performed considering the sequential nature within
the human motion.
- Within this transformer, we introduce a gated linear
attention and an RMSNorm module, which drive the transformer to pay attention
to the key actions and suppress the instability caused by either the abnormal
movements or the heterogeneous distributions within multi-modalities.
- Experimental
results demonstrate that our framework outperforms previous methods across all
modalities, including text-to-motion, speech-to-gesture, and music-to-dance.
- The code of our method will be made public.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14954v1)
- [arXiv](https://arxiv.org/abs/2510.14954v1)

---

<a id='2510.14945v1'></a>
## [3D Scene Prompting for Scene-Consistent Camera-Controllable Video Generation](https://arxiv.org/abs/2510.14945v1)

**Authors:** JoungBin Lee, Jaewoo Jung, Jisang Han, Takuya Narihira, Kazumi Fukuda, Junyoung Seo, Sunghwan Hong, Yuki Mitsufuji, Seungryong Kim

**Published:** 2025-10-16

**Categories:** cs.CV

**Abstract:**

We present 3DScenePrompt, a framework that generates the next video chunk
from arbitrary-length input while enabling precise camera control and
preserving scene consistency. Unlike methods conditioned on a single image or a
short clip, we employ dual spatio-temporal conditioning that reformulates
context-view referencing across the input video. Our approach conditions on
both temporally adjacent frames for motion continuity and spatially adjacent
content for scene consistency. However, when generating beyond temporal
boundaries, directly using spatially adjacent frames would incorrectly preserve
dynamic elements from the past. We address this by introducing a 3D scene
memory that represents exclusively the static geometry extracted from the
entire input video. To construct this memory, we leverage dynamic SLAM with our
newly introduced dynamic masking strategy that explicitly separates static
scene geometry from moving elements. The static scene representation can then
be projected to any target viewpoint, providing geometrically consistent warped
views that serve as strong 3D spatial prompts while allowing dynamic regions to
evolve naturally from temporal context. This enables our model to maintain
long-range spatial coherence and precise camera control without sacrificing
computational efficiency or motion realism. Extensive experiments demonstrate
that our framework significantly outperforms existing methods in scene
consistency, camera controllability, and generation quality. Project page :
https://cvlab-kaist.github.io/3DScenePrompt/

**Analysis:**

以下是论文“3D Scene Prompting for Scene-Consistent Camera-Controllable Video Generation”的全面摘要：

**1. 主要问题或研究问题：**
该论文旨在解决场景一致、相机可控的视频生成任务。现有方法在处理任意长度的输入视频时，难以在保持长期空间一致性、精确相机控制和生成质量的同时，生成与输入视频场景一致的后续视频片段。具体来说，当生成超出时间边界的视频时，直接使用过去的空间相邻帧会导致动态元素（如移动物体）被错误地保留，从而破坏场景的自然演变。

**2. 关键创新或方法贡献：**
*   **双重时空条件作用（Dual Spatio-Temporal Conditioning）：** 论文提出了一种新颖的双重时空条件作用策略，通过重新构建输入视频中的上下文视图引用，同时考虑时间相邻帧（用于运动连续性）和空间相邻内容（用于场景一致性）。
*   **3D 场景记忆（3D Scene Memory）：** 为了解决动态元素保留问题，论文引入了一个3D场景记忆，它专门表示从整个输入视频中提取的静态几何结构。这确保了空间条件作用只提供持久的静态场景结构，而动态内容可以从时间上下文中自然演变。
*   **动态掩蔽策略（Dynamic Masking Strategy）：** 为了构建3D场景记忆，论文利用动态SLAM，并引入了一种新的动态掩蔽策略，明确地将静态场景几何与移动元素分离。该策略通过像素级运动检测、反向跟踪聚合运动证据以及使用SAM2进行对象级掩蔽，确保只提取静态内容。
*   **几何一致的扭曲视图作为空间提示：** 静态场景表示可以投影到任何目标视点，生成几何一致的扭曲视图，作为强大的3D空间提示。这使得模型能够在不牺牲计算效率或运动真实感的情况下，保持长距离空间连贯性和精确相机控制。
*   **基于预训练视频生成器的架构：** 论文在强大的预训练视频生成器（如CogVideoX-I2V-5B）的基础上进行构建，通过重新设计内容引用方式，保留了其学习到的先验知识和训练效率。

**3. 主要结果及其意义：**
*   **卓越的场景一致性：** 3DScenePrompt在RealEstate10K和DynPose-100K数据集上，在PSNR、SSIM、LPIPS和MEt3R等所有指标上均显著优于现有方法（如DFoT）。尤其在MEt3R几何一致性误差上，下降了77%，表明其在多视图几何对齐方面的卓越性能。
*   **精确的相机可控性：** 论文方法在mRotErr、mTransErr和mCamMC等相机可控性指标上均优于MotionCtrl、CameraCtrl、FloVD和AC3D等基线方法，证明了其能够精确遵循给定的相机轨迹。
*   **高质量的视频生成：** 在FVD和VBench++（包括主体一致性、背景一致性、美学质量、成像质量、时间闪烁、运动平滑度和动态程度）等视频生成质量指标上，论文方法也取得了最佳性能。
*   **计算效率和运动真实感：** 通过选择性地检索最相关的帧（时空上），该框架实现了对任意长度视频的计算高效处理，同时保持了运动连续性和场景一致性。

**4. 论文中提及的局限性：**
*   论文中没有明确提及当前方法的具体局限性。然而，通过与现有方法的比较，可以推断出，在处理长序列时，现有方法（如DFoT）由于内存限制，难以保持长期空间连贯性。3DScenePrompt通过其3D场景记忆和动态掩蔽策略解决了这一问题。

**5. 潜在的未来研究方向：**
*   论文明确指出，该方法为长篇视频合成应用开辟了新的可能性，这些应用中保持空间一致性和精确相机控制至关重要。这暗示了未来研究可以进一步探索该框架在电影制作、虚拟现实和合成数据生成等领域的应用。
*   进一步优化动态掩蔽策略，以处理更复杂或更细微的动态场景，例如快速变化的动态或半透明物体。
*   探索将3D场景记忆与更高级的视频扩散模型或生成架构相结合，以进一步提升生成质量和效率。
*   研究如何将用户交互（例如，通过文本提示或草图）更深入地整合到3D场景记忆的构建和利用中，以实现更灵活的创作控制。

**Key Findings:**

- We present 3DScenePrompt, a framework that generates the next video chunk
from arbitrary-length input while enabling precise camera control and
preserving scene consistency.
- Our approach conditions on
both temporally adjacent frames for motion continuity and spatially adjacent
content for scene consistency.
- To construct this memory, we leverage dynamic SLAM with our
newly introduced dynamic masking strategy that explicitly separates static
scene geometry from moving elements.
- Extensive experiments demonstrate
that our framework significantly outperforms existing methods in scene
consistency, camera controllability, and generation quality.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14945v1)
- [arXiv](https://arxiv.org/abs/2510.14945v1)

---

<a id='2510.14904v1'></a>
## [MaskCaptioner : Learning to Jointly Segment and Caption Object Trajectories in Videos](https://arxiv.org/abs/2510.14904v1)

**Authors:** Gabriel Fiastre, Antoine Yang, Cordelia Schmid

**Published:** 2025-10-16

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Dense Video Object Captioning (DVOC) is the task of jointly detecting,
tracking, and captioning object trajectories in a video, requiring the ability
to understand spatio-temporal details and describe them in natural language.
Due to the complexity of the task and the high cost associated with manual
annotation, previous approaches resort to disjoint training strategies,
potentially leading to suboptimal performance. To circumvent this issue, we
propose to generate captions about spatio-temporally localized entities
leveraging a state-of-the-art VLM. By extending the LVIS and LV-VIS datasets
with our synthetic captions (LVISCap and LV-VISCap), we train MaskCaptioner, an
end-to-end model capable of jointly detecting, segmenting, tracking and
captioning object trajectories. Moreover, with pretraining on LVISCap and
LV-VISCap, MaskCaptioner achieves state-of-the-art DVOC results on three
existing benchmarks, VidSTG, VLN and BenSMOT. The datasets and code are
available at https://www.gabriel.fiastre.fr/maskcaptioner/.

**Analysis:**

好的，这是一篇关于“MaskCaptioner : Learning to Jointly Segment and Caption Object Trajectories in Videos”论文的全面摘要，内容将用中文呈现，并涵盖您要求的所有要点。

---

### MaskCaptioner: 学习联合分割和描述视频中物体轨迹

**1. 论文主要问题或研究问题**

该论文旨在解决**密集视频物体描述 (Dense Video Object Captioning, DVOC)** 任务中的核心挑战。DVOC要求模型能够联合检测、跟踪并用自然语言描述视频中所有物体的轨迹。这项任务的复杂性以及手动标注的高昂成本导致现有方法通常采用分离的训练策略，这可能导致次优性能。因此，论文的核心问题是如何实现端到端的DVOC训练，以克服对密集标注数据稀缺的依赖，并提高整体性能。

**2. 关键创新或方法论贡献**

*   **VLM驱动的合成数据生成：** 论文最主要的创新是提出了一种利用最先进的视觉语言模型（VLM，具体是Gemini 2.0 Flash）来生成物体级合成描述的方法。通过多模态提示策略，VLM能够为视频中空间局部化的物体生成详细的、以物体为中心的描述。
*   **扩展现有数据集：** 论文将LVIS（图像分割）和LV-VIS（视频实例分割）数据集扩展为LVISCap和LV-VISCap，首次为DVOC任务提供了包含（掩码、边界框、类别、描述）标注的统一训练集。这些合成数据极大地弥补了DVOC任务所需密集标注的不足。
*   **端到端MaskCaptioner模型：** 论文提出了MaskCaptioner，这是一个能够联合执行物体检测、分割、跟踪和描述的端到端模型。该架构基于Open-Vocabulary Video Instance Segmentation (OV-VIS) 模型OVFormer，并扩展了描述头，能够从视频片段级别的预测中聚合信息，生成轨迹级别的描述。
*   **统一训练策略：** 通过使用生成的LVISCap和LV-VISCap数据集，MaskCaptioner能够进行端到端训练，避免了以往方法中分离训练策略的次优性。

**3. 主要结果及其意义**

*   **最先进的DVOC性能：** MaskCaptioner在三个现有基准测试（VidSTG、VLN和BenSMOT）上取得了最先进的DVOC结果。这证明了其方法在联合检测、跟踪和描述物体轨迹方面的有效性。
*   **合成数据的重要性：** 实验结果表明，LVISCap和LV-VISCap等合成数据集极大地提升了MaskCaptioner的DVOC性能。CapA（描述准确性）与训练描述数量呈对数相关，表明生成更多数据可能带来进一步改进。
*   **分割任务的扩展：** 论文成功地将DVOC任务扩展到分割掩码，而不仅仅是边界框，这提供了更精细的物体定位和描述。
*   **鲁棒性：** MaskCaptioner对视觉骨干网络的选择表现出鲁棒性。
*   **时间聚合的优势：** 引入时间聚合模块显著提高了描述性能，通过整合来自多个视频片段的信息，丰富了对时间上扩展动作的描述。

**4. 论文中提及的局限性**

*   **定位和描述的改进空间：** 尽管MaskCaptioner表现出色，但在定位和描述方面仍有改进空间，特别是对于小物体。
*   **描述的通用性：** 自动生成的描述有时可能过于通用，或者在视频中混淆同一类的两个物体。
*   **复杂动作的限制：** 当前基准测试中可观察到的动作复杂性有限，模型在处理更复杂的物体交互和多动作片段时可能面临挑战。
*   **识别错误：** 在模糊上下文、模糊实例或稀有类别的情况下，MaskCaptioner可能无法正确识别所描述的物体，有时会导致错误的命名（例如，将“钳子”错误地识别为“刀”）。
*   **描述不一致：** 在类似情况下，MaskCaptioner生成的描述在指代同一物体时可能不一致。
*   **检测/分割错误：** 在复杂运动、外观变化或遮挡的情况下，MaskCaptioner有时会未能检测、分割或跟踪物体，导致缺少描述（例如，未能检测到苍鹭喙中的鱼）。

**5. 潜在的未来研究方向**

*   **更先进的自动描述技术：** 未来的工作可以探索不同的自动描述技术，例如基于Ref-SAV（Yuan et al., 2025）的方法，该方法分多步生成描述，以区分外观和运动描述。
*   **构建更复杂的基准测试：** 针对具有更复杂物体交互和多个动作片段的视频，构建新的基准测试，以推动DVOC任务的发展。
*   **提升小物体定位和描述：** 进一步研究如何提高模型对视频中小物体的定位和描述能力。
*   **解决描述通用性和一致性问题：** 探索方法以生成更具体、更一致的物体描述，避免混淆相似物体。

---

**Key Findings:**

- To circumvent this issue, we
propose to generate captions about spatio-temporally localized entities
leveraging a state-of-the-art VLM.
- Moreover, with pretraining on LVISCap and
LV-VISCap, MaskCaptioner achieves state-of-the-art DVOC results on three
existing benchmarks, VidSTG, VLN and BenSMOT.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14904v1)
- [arXiv](https://arxiv.org/abs/2510.14904v1)

---

<a id='2510.14876v1'></a>
## [BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data](https://arxiv.org/abs/2510.14876v1)

**Authors:** Roni Goldshmidt, Hamish Scott, Lorenzo Niccolini, Shizhan Zhu, Daniel Moura, Orly Zvitia

**Published:** 2025-10-16

**Categories:** cs.CV

**Abstract:**

Existing collision prediction methods often fail to distinguish between
ego-vehicle threats and random accidents not involving the ego vehicle, leading
to excessive false alerts in real-world deployment. We present BADAS, a family
of collision prediction models trained on Nexar's real-world dashcam collision
dataset -- the first benchmark designed explicitly for ego-centric evaluation.
We re-annotate major benchmarks to identify ego involvement, add consensus
alert-time labels, and synthesize negatives where needed, enabling fair AP/AUC
and temporal evaluation. BADAS uses a V-JEPA2 backbone trained end-to-end and
comes in two variants: BADAS-Open (trained on our 1.5k public videos) and
BADAS1.0 (trained on 40k proprietary videos). Across DAD, DADA-2000, DoTA, and
Nexar, BADAS achieves state-of-the-art AP/AUC and outperforms a
forward-collision ADAS baseline while producing more realistic time-to-accident
estimates. We release our BADAS-Open model weights and code, along with
re-annotations of all evaluation datasets to promote ego-centric collision
prediction research.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Roni Goldshmidt等人撰写的论文“BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data”的全面摘要。

---

### 论文摘要：BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data

**1. 主要问题或研究问题：**
现有的碰撞预测方法在实际部署中存在严重缺陷，即无法有效区分“自我车辆”（ego-vehicle）面临的威胁与不涉及自我车辆的随机事故。这导致了过多的虚假警报，降低了系统的实用性和用户接受度。论文旨在解决这一核心问题，即如何开发一种能够准确预测自我车辆相关碰撞，同时显著减少虚假警报的碰撞预测系统。

**2. 关键创新或方法论贡献：**
*   **以自我车辆为中心的碰撞预测范式：** 论文重新定义了碰撞预测任务，使其专注于自我车辆的安全性。通过对DAD、DoTA和DADA-2000等主要基准数据集进行重新标注，明确区分了自我车辆是否卷入事故，发现现有数据集中40-92%的事故与自我车辆无关。这些重新标注的数据集（包括共识警报时间标签和合成的负样本）已公开，以促进以自我车辆为中心的碰撞预测研究。
*   **V-JEPA2骨干网络的应用：** BADAS模型采用V-JEPA2（一种先进的视频基础模型）作为骨干网络，并进行端到端训练。V-JEPA2在理解时间动态和视觉模式方面表现出色，特别适合预测性任务。
*   **基于真实世界行车记录仪数据的训练：** BADAS模型在Nexar的真实世界行车记录仪碰撞数据集上进行训练，该数据集是第一个专门为自我车辆评估设计的基准。该数据集包含自我车辆卷入的碰撞和险情事件，以及通过紧急操作成功避免危险情况的近碰撞事件，提供了丰富的训练信号。
*   **标准化时间评估：** 论文通过10位标注员的共识，建立了警报时间的连贯定义，并为所有测试集提供了精确的时间标注，解决了现有数据集中定义不一致和主观性问题。
*   **数据增强和采样策略：** 采用数据增强和正样本2倍过采样策略，显著提高了模型的性能和对真实世界变化的鲁棒性。

**3. 主要结果及其意义：**
*   **最先进的性能：** BADAS模型在DAD、DADA-2000、DoTA和Nexar等所有主要基准测试中均取得了最先进的AP/AUC性能。BADAS1.0（基于40k专有视频训练）在DAD数据集上达到了0.94 AP，远超基线方法的0.06 AP。在Nexar数据集上，BADAS-Open（基于1.5k公开视频训练）的AP达到0.86，显著优于学术基线（0.48-0.53）和商用FCW系统（0.58）。
*   **更真实的事故发生时间估计：** BADAS模型生成的平均事故发生时间（mTTA）估计值更符合人类预测能力（3-5秒），而基线方法则报告了不切实际的9-10秒预测。
*   **鲁棒性和泛化能力：** BADAS模型在不同数据集上保持了一致的性能，表明其具有强大的泛化能力。
*   **数据规模效应：** 随着训练数据量的增加（从1.5k到40k视频），Nexar验证AP呈对数级改进，证实了利用大规模真实世界数据能够持续提升性能。
*   **定性分析：** BADAS-Open的预测分数在碰撞临近时急剧上升，且预测稳定、自信，而基线方法则表现出不规律的模式和过早或不一致的警报。

**4. 论文中提及的局限性：**
*   **长尾性能下降：** BADAS模型在少数类别（如行人、骑自行车者、摩托车和动物）上的召回率显著下降，表明模型在处理罕见、视觉多样且低频事件类型时存在泛化挑战。这是由于训练数据集中缺乏这些长尾事件所致。
*   **数据量限制：** 尽管BADAS1.0使用了40k专有视频，但论文指出数据规模的潜力尚未完全饱和，暗示仍有提升空间。

**5. 潜在的未来研究方向：**
*   **扩展数据集：** 进一步扩大数据集，以增强模型的泛化能力，尤其是在长尾类别方面。
*   **改进平均警报时间（mTTA）：** 优化模型以减少虚假警报，并提供更准确的警报时间。
*   **解决长尾类别问题：** 开发专门的策略来更好地评估和预测多样且罕见的驾驶场景中的长尾类别。
*   **多级分类：** 将碰撞预测模型从二元分类（碰撞/非碰撞）扩展到三级分类（正常、警告、警报），以实现更精细的风险评估和自适应决策。
*   **整合到ADAS和自动驾驶系统：** 进一步研究如何将BADAS的可靠、上下文感知碰撞预测能力整合到更广泛的ADAS和全自动驾驶系统中，以提高安全性。

---

总而言之，BADAS论文通过引入以自我车辆为中心的范式、利用V-JEPA2骨干网络和Nexar真实世界行车记录仪数据，显著提升了碰撞预测的性能和实用性。其关键贡献在于重新标注了现有数据集、标准化了评估协议，并展示了在减少虚假警报和提供更准确警报时间方面的卓越能力。尽管在长尾事件处理上仍有挑战，但BADAS为未来更安全、更智能的驾驶辅助系统和自动驾驶技术奠定了坚实基础。

**Key Findings:**

- We present BADAS, a family
of collision prediction models trained on Nexar's real-world dashcam collision
dataset -- the first benchmark designed explicitly for ego-centric evaluation.
- Across DAD, DADA-2000, DoTA, and
Nexar, BADAS achieves state-of-the-art AP/AUC and outperforms a
forward-collision ADAS baseline while producing more realistic time-to-accident
estimates.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14876v1)
- [arXiv](https://arxiv.org/abs/2510.14876v1)

---

<a id='2510.14874v1'></a>
## [TOUCH: Text-guided Controllable Generation of Free-Form Hand-Object Interactions](https://arxiv.org/abs/2510.14874v1)

**Authors:** Guangyi Han, Wei Zhai, Yuhang Yang, Yang Cao, Zheng-Jun Zha

**Published:** 2025-10-16

**Categories:** cs.CV

**Abstract:**

Hand-object interaction (HOI) is fundamental for humans to express intent.
Existing HOI generation research is predominantly confined to fixed grasping
patterns, where control is tied to physical priors such as force closure or
generic intent instructions, even when expressed through elaborate language.
Such an overly general conditioning imposes a strong inductive bias for stable
grasps, thus failing to capture the diversity of daily HOI. To address these
limitations, we introduce Free-Form HOI Generation, which aims to generate
controllable, diverse, and physically plausible HOI conditioned on fine-grained
intent, extending HOI from grasping to free-form interactions, like pushing,
poking, and rotating. To support this task, we construct WildO2, an in-the-wild
diverse 3D HOI dataset, which includes diverse HOI derived from internet
videos. Specifically, it contains 4.4k unique interactions across 92 intents
and 610 object categories, each with detailed semantic annotations. Building on
this dataset, we propose TOUCH, a three-stage framework centered on a
multi-level diffusion model that facilitates fine-grained semantic control to
generate versatile hand poses beyond grasping priors. This process leverages
explicit contact modeling for conditioning and is subsequently refined with
contact consistency and physical constraints to ensure realism. Comprehensive
experiments demonstrate our method's ability to generate controllable, diverse,
and physically plausible hand interactions representative of daily activities.
The project page is $\href{https://guangyid.github.io/hoi123touch}{here}$.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Guangyi Han等人撰写的论文“TOUCH: Text-guided Controllable Generation of Free-Form Hand-Object Interactions”的全面摘要。

---

### TOUCH: 文本引导的可控自由形式手物交互生成

**1. 主要问题或研究问题：**
现有的手物交互（HOI）生成研究主要局限于固定的抓取模式，其控制通常依赖于物理先验（如力闭合）或通用意图指令，即使通过复杂的语言表达，也未能捕捉到日常HOI的多样性。这种过度通用的条件设定导致模型偏向于生成稳定的抓取，从而限制了交互的多样性。该论文旨在解决这一限制，引入“自由形式HOI生成”任务，目标是生成可控、多样且物理上合理的手物交互，这些交互由细粒度意图条件化，并将HOI从抓取扩展到推、戳、旋转等自由形式交互。

**2. 关键创新或方法论贡献：**

*   **自由形式HOI生成任务的引入：** 论文首次提出了“自由形式HOI生成”任务，旨在打破以抓取为中心的限制，生成更广泛、更具语义表达力的日常交互，包括各种非抓取操作。
*   **WildO2数据集的构建：** 为支持新任务，论文构建了一个名为WildO2的野外多样化3D HOI数据集。该数据集包含从互联网视频中提取的4.4k独特交互，涵盖92种意图和610个对象类别，并附有详细的语义标注。数据集的构建通过一个半自动化流程实现，该流程利用O2HOI（Object-only to Hand-Object Interaction）帧配对策略，解决了野外视频中手部遮挡导致的对象重建难题。
*   **TOUCH三阶段框架：** 论文提出了一个名为TOUCH的三阶段框架，用于可控的自由形式HOI生成：
    *   **接触图预测（Contact Map Prediction）：** 利用两个独立的CVAE模型，根据文本和对象几何信息，预测手和对象表面的二值接触图，为交互位置和姿态提供强烈的空间先验。
    *   **多级条件扩散模型（Multi-Level Conditioned Diffusion）：** 采用基于Transformer的去噪扩散概率模型（DDPM），通过注意力机制融合语义和几何信息。在早期扩散阶段，粗粒度意图和全局对象几何信息引导整体姿态；在后期阶段，细粒度文本和局部接触特征进一步细化细节动作，实现细粒度语义控制。
    *   **物理约束细化（Physical Constraints Refinement）：** 引入自监督循环一致性损失和物理约束，对生成的姿态进行优化，以确保交互的真实性和物理可行性，解决全局姿态漂移问题。

**3. 主要结果及其意义：**
全面的实验结果表明，TOUCH方法在生成可控、多样且物理上合理的手物交互方面表现出色，这些交互能代表日常活动。与现有基线方法（如ContactGen和Text2HOI）相比，TOUCH在接触准确性（P-IoU, P-F1）、物理合理性（MPVPE, PD, PV）、多样性（熵, 聚类大小）和语义一致性（P-FID, VLM辅助评估, 感知分数）等所有评估指标上均优于基线。消融研究进一步验证了接触引导和粗到细文本控制设计的有效性。WildO2数据集的构建为该领域未来的研究提供了关键资源。

**4. 论文中提及的局限性：**
*   **静态HOI快照：** 当前框架主要关注静态HOI快照，这限制了其捕捉交互过程时间动态的能力。
*   **数据集规模：** 尽管WildO2数据集提供了快速扩展的潜力，但当前数据集的规模仍有增长空间。

**5. 潜在的未来研究方向：**
*   **动态序列扩展：** 未来工作计划将研究扩展到动态序列，利用大规模视频数据集和6-DoF对象姿态估计，从而建模完整的人-环境交互过程。
*   **更精细的物理模拟：** 进一步探索更精细的物理模拟，以提高交互的真实性和稳定性。
*   **更广泛的交互类型：** 持续扩展自由形式HOI的范围，涵盖更多复杂和多样的日常交互。

---

这篇论文通过引入自由形式HOI生成任务和构建WildO2数据集，显著推动了手物交互生成领域的发展。其提出的TOUCH框架，通过多级扩散模型和物理约束细化，实现了对细粒度意图的可控生成，为AR/VR、机器人和具身AI等应用提供了新的可能性。

**Key Findings:**

- To address these
limitations, we introduce Free-Form HOI Generation, which aims to generate
controllable, diverse, and physically plausible HOI conditioned on fine-grained
intent, extending HOI from grasping to free-form interactions, like pushing,
poking, and rotating.
- Building on
this dataset, we propose TOUCH, a three-stage framework centered on a
multi-level diffusion model that facilitates fine-grained semantic control to
generate versatile hand poses beyond grasping priors.
- Comprehensive
experiments demonstrate our method's ability to generate controllable, diverse,
and physically plausible hand interactions representative of daily activities.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.14874v1)
- [arXiv](https://arxiv.org/abs/2510.14874v1)

---

