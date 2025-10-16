time: 20251016

# Arxiv Computer Vision Papers - 2025-10-16

## Executive Summary

好的，这是一份针对2025年10月15日Arxiv计算机视觉领域论文的每日报告执行摘要，旨在帮助忙碌的研究人员快速掌握关键信息。

---

**每日Arxiv计算机视觉论文报告执行摘要 (2025年10月15日)**

**1. 主要主题和趋势概述：**

今天的论文集清晰地展示了计算机视觉领域向**多模态、通用性、大型模型适应与应用**的强劲发展趋势。核心主题包括：

*   **大型视觉-语言模型 (VLMs) 的泛化与应用：** 大量工作聚焦于VLMs在不同任务（如城市监控、机器人策略、多模态对话）中的能力拓展、评估和实际部署。
*   **多模态数据与基准：** 为了支持通用模型的发展，高质量、大规模的多模态数据集和统一基准的构建成为重要研究方向。
*   **通用智能体与元推理：** 出现了旨在构建能够处理多种模态、执行复杂推理任务的“通用验证器”或“元推理器”的尝试。
*   **视频理解与4D表示：** 视频内容的高效、细粒度表示方法（如轨迹场）是持续关注的焦点。
*   **模型效率与架构创新：** 尽管大型模型是主流，但也有工作关注如何通过图神经网络等架构创新提升效率。

**2. 特别重要或创新的论文亮点：**

*   **"Generative Universal Verifier as Multimodal Meta-Reasoner" (Xinchen Zhang et al.)：** 这篇论文提出了一个“生成式通用验证器”的概念，旨在作为多模态的元推理器。其目标是构建一个能够跨模态进行通用验证和推理的系统，这代表了迈向更高级通用人工智能的重要一步。
*   **"Trace Anything: Representing Any Video in 4D via Trajectory Fields" (Xinhang Liu et al.)：** 提出通过轨迹场来表示视频中的任何内容，这是一种新颖且强大的4D视频表示方法，有望极大地提升视频理解和编辑的精细度与灵活性。
*   **"NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching" (Run Luo et al.)：** 致力于构建“任意到任意”的全模态基础模型，并引入了离散流匹配技术。这代表了在统一不同模态输入和输出方面的前沿探索，具有巨大的潜力。

**3. 新兴研究方向或技术：**

*   **“全模态 (Omnimodal)”模型：** 不仅仅是多模态，而是追求能够处理和生成任意模态（文本、图像、音频、视频、动作等）之间转换的模型，如"NExT-OMNI"和"InteractiveOmni"。
*   **轨迹场 (Trajectory Fields) 进行视频表示：** 作为一种细粒度的4D视频表示方法，有望超越传统的帧级或稀疏点表示。
*   **离散流匹配 (Discrete Flow Matching)：** 在全模态生成模型中被提出，可能成为处理复杂多模态数据生成的新范式。
*   **Prompt-based Adaptation 的系统性研究：** 随着大型模型成为主流，如何高效、鲁棒地进行模型适应（尤其是基于Prompt）成为关键，"Prompt-based Adaptation in Large-scale Vision Models: A Survey"提供了全面的视角。

**4. 建议阅读全文的论文：**

对于不同兴趣的研究人员，建议阅读以下论文：

*   **对于关注通用AI和多模态推理的研究者：**
    *   **"Generative Universal Verifier as Multimodal Meta-Reasoner" (Xinchen Zhang et al.)**
    *   **"NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching" (Run Luo et al.)**
*   **对于关注大型模型应用和适应性的研究者：**
    *   **"Prompt-based Adaptation in Large-scale Vision Models: A Survey" (Xi Xiao et al.)** - 提供全面背景和未来方向。
    *   **"Towards General Urban Monitoring with Vision-Language Models: A Review, Evaluation, and a Research Agenda" (André Torneiro et al.)** - 结合实际应用场景，具有很强的参考价值。
    *   **"InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy" (Xinyi Chen et al.)** - 机器人领域的重要进展。
*   **对于关注视频理解和新表示方法的研究者：**
    *   **"Trace Anything: Representing Any Video in 4D via Trajectory Fields" (Xinhang Liu et al.)**
*   **对于关注多模态基准和数据构建的研究者：**
    *   **"Uni-MMMU: A Massive Multi-discipline Multimodal Unified Benchmark" (Kai Zou et al.)**
    *   **"Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs" (Yi Zhang et al.)**

---

这份摘要旨在提供一个高层次的概览，帮助您快速识别与您研究方向最相关的论文。

---

## Table of Contents

1. [Prompt-based Adaptation in Large-scale Vision Models: A Survey](#2510.13219v1)
2. [Towards General Urban Monitoring with Vision-Language Models: A Review, Evaluation, and a Research Agenda](#2510.12400v1)
3. [Generative Universal Verifier as Multimodal Meta-Reasoner](#2510.13804v1)
4. [Trace Anything: Representing Any Video in 4D via Trajectory Fields](#2510.13802v1)
5. [Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs](#2510.13795v1)
6. [InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy](#2510.13778v1)
7. [Uni-MMMU: A Massive Multi-discipline Multimodal Unified Benchmark](#2510.13759v1)
8. [InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue](#2510.13747v1)
9. [Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs](#2510.13740v1)
10. [NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching](#2510.13721v1)

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

**论文摘要：大型视觉模型中的基于提示的适应：一项综述**

**1. 主要问题或研究问题：**
该综述旨在解决计算机视觉领域中，视觉提示（Visual Prompting, VP）和视觉提示微调（Visual Prompt Tuning, VPT）这两种新兴的轻量级模型适应方法之间概念边界模糊的问题。尽管它们在“预训练-微调”范式中取得了快速进展，但当前研究中经常互换使用这些术语，缺乏对它们各自技术和应用的系统性区分。因此，核心问题是建立一个统一的框架，清晰地定义、分类并概述基于提示的适应（Prompt-based Adaptation, PA）方法及其在大型视觉模型中的应用。

**2. 关键创新或方法论贡献：**
* **统一框架与分类：** 论文将VP和VPT从第一性原理出发，概念化为一个统一的“基于提示的适应（PA）”框架。
* **详细分类法：** 提出了一种全面的分类法，将现有方法分为可学习、生成式和不可学习提示，并根据注入粒度（像素级和令牌级）进一步组织。
* **应用领域整合：** 深入探讨了PA在各种不同领域中的集成，包括医学影像、3D点云、视觉-语言任务，以及其在测试时间适应和可信赖AI中的作用。
* **首次全面综述：** 据作者所知，这是第一篇专门针对PA方法论及其应用，并结合其独特特征的全面综述。

**3. 主要结果及其意义：**
* **PA的有效性：** 综述表明，PA作为全量微调的轻量级且有效替代方案，在各种受限学习场景（如数据稀缺、数据分布非平稳、模型内部不可访问或计算资源有限）中展现出显著的有效性。
* **VP与VPT的区分：** VP通过修改输入空间来适应模型，而VPT则通过在模型内部注入可学习的提示令牌来调整模型行为。这种区分对于理解它们的效率特点和适用场景至关重要。
* **广泛的应用潜力：** PA在基础CV任务（如分割、图像恢复、压缩）和特定领域（如医疗、机器人、遥感）中都显示出强大的潜力，能够桥接预训练模型与下游任务。
* **可信赖AI的贡献：** PA在提升模型鲁棒性、缓解公平性与偏见、以及保障隐私与安全方面发挥着重要作用，为构建可信赖的AI系统提供了轻量级解决方案。

**4. 论文中提及的局限性：**
* **训练开销与稳定性：** 尽管PA提高了参数效率，但训练开销（如超参数搜索）和训练结果的不稳定性仍然是挑战。
* **推理延迟：** 额外的提示组件可能增加推理延迟和内存消耗。
* **真实世界环境评估不足：** 当前PA方法的评估主要依赖标准化基准数据集，未能充分反映真实世界场景的复杂性和分布漂移。
* **理论分析有限：** 相关的理论分析在当前研究社区中仍然有限，对PA如何引发模型行为变化、提示学习到什么以及在不同适应设置下的有效性等问题缺乏深入理解。

**5. 潜在的未来研究方向：**
* **解决训练开销与稳定性：** 需要进一步研究训练捷径、检测并纠正训练不稳定性的策略。
* **优化推理延迟：** 探索剪枝、知识蒸馏、量化和内存高效微调等技术以减少推理延迟。
* **加强真实世界评估：** 优先发展能够应对多样化、复杂和异构视觉上下文的鲁棒方法，以弥合与真实世界部署之间的差距。
* **深入理论分析：** 进一步探索PA如何诱导模型行为变化、视觉提示学习的内容以及PA方法在不同适应设置下的有效性。
* **安全对齐：** 将安全对齐（包括可解释性、治理和严格验证）融入PA方法的开发和部署中，以应对恶意行为、有害内容生成和偏见放大等问题。
* **混合方法：** 结合VP和VPT的优势，例如使用生成器提供初始空间提示，同时使用条件感知令牌进行内部调整，以应对环境退化。

这篇综述为研究人员和实践者提供了一个清晰的路线图，以理解和探索PA相关研究的不断发展。

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

<a id='2510.12400v1'></a>
## [Towards General Urban Monitoring with Vision-Language Models: A Review, Evaluation, and a Research Agenda](https://arxiv.org/abs/2510.12400v1)

**Authors:** André Torneiro, Diogo Monteiro, Paulo Novais, Pedro Rangel Henriques, Nuno F. Rodrigues

**Published:** 2025-10-14

**Categories:** cs.CV

**Abstract:**

Urban monitoring of public infrastructure (such as waste bins, road signs,
vegetation, sidewalks, and construction sites) poses significant challenges due
to the diversity of objects, environments, and contextual conditions involved.
Current state-of-the-art approaches typically rely on a combination of IoT
sensors and manual inspections, which are costly, difficult to scale, and often
misaligned with citizens' perception formed through direct visual observation.
This raises a critical question: Can machines now "see" like citizens and infer
informed opinions about the condition of urban infrastructure? Vision-Language
Models (VLMs), which integrate visual understanding with natural language
reasoning, have recently demonstrated impressive capabilities in processing
complex visual information, turning them into a promising technology to address
this challenge. This systematic review investigates the role of VLMs in urban
monitoring, with particular emphasis on zero-shot applications. Following the
PRISMA methodology, we analyzed 32 peer-reviewed studies published between 2021
and 2025 to address four core research questions: (1) What urban monitoring
tasks have been effectively addressed using VLMs? (2) Which VLM architectures
and frameworks are most commonly used and demonstrate superior performance? (3)
What datasets and resources support this emerging field? (4) How are VLM-based
applications evaluated, and what performance levels have been reported?

**Analysis:**

好的，这是一篇关于André Torneiro等人的论文“Towards General Urban Monitoring with Vision-Language Models: A Review, Evaluation, and a Research Agenda”的全面摘要：

**论文摘要：**

**1. 主要问题或研究问题：**
该论文旨在解决城市公共基础设施（如垃圾桶、路标、植被、人行道和建筑工地）监测所面临的重大挑战。传统的物联网传感器和人工检查方法成本高昂、难以扩展，且往往与市民的直观视觉感知不符。因此，核心研究问题是：机器能否像市民一样“看”，并对城市基础设施的状况形成知情判断？具体而言，论文通过系统综述，探讨了视觉-语言模型（VLMs）在城市监测中的作用，尤其侧重于零样本应用，并回答了以下四个核心问题：
1. VLMs有效解决了哪些城市监测任务？
2. 哪些VLM架构和框架最常用且表现优异？
3. 哪些数据集和资源支持这一新兴领域？
4. 基于VLM的应用如何评估，报告了哪些性能水平？

**2. 关键创新或方法论贡献：**
* **系统综述与分类法：** 论文采用PRISMA方法论，对2021年至2025年间发表的32篇同行评审研究进行了分析。在此基础上，提出了一套功能性VLM城市应用分类法，将现有研究划分为七个主要领域：目标检测与分割、城市规划与土地利用分类、导航与路径规划、交通分析与运输、城市场景理解与感知、地理定位与位置查找、以及城市监控与安全。
* **模型生态系统分析：** 论文将模型生态系统分为四类：纯视觉骨干网络、独立语言模型、多模态VLM和混合集成，并分析了它们在城市应用中的操作模式、集成策略和局限性。
* **数据集使用模式分析：** 论文详细分析了城市监测中数据集的使用情况，包括街景图像、合成数据、航空/俯视图以及专有数据集，揭示了现有数据集在泛化能力、可复现性和实际应用方面的结构性缺陷。

**3. 主要结果及其意义：**
* **VLM在城市监测中的潜力：** VLMs通过整合视觉理解和自然语言推理，在处理复杂视觉信息方面展现出强大能力，为解决城市监测挑战提供了有前景的技术。
* **性能多样性：** 在不同任务中，VLMs表现出不同的性能。例如，在目标检测中，SAM和Grounding DINO结合取得了高IoU分数；在城市规划中，UrbanCLIP在住宅区分类中F1分数达到0.82；在地理定位中，IM2City通过线性探测达到了85.9%的Top-1准确率。
* **模型和数据集趋势：** CLIP、Grounding DINO和GPT-3.5是最常用的模型，反映了对模块化、通用骨干网络的偏好。街景图像数据集（如Google Street View、Mapillary Vistas）占据主导地位，但合成数据集（如CARLA、SYNTHIA）在模拟稀有或危险条件方面也得到广泛应用。
* **领域适应与泛化挑战：** 尽管在有限监督下，mIoU和上下文感知准确性有所提高，但在跨域（如合成到真实场景）和跨城市泛化方面仍存在显著挑战。

**4. 论文中提及的局限性：**
* **评估标准不一致：** 性能报告差异大，缺乏统一的基准协议、标准化指标、置信区间和详细的错误分析，使得不同研究间的直接比较困难。
* **部署可行性不足：** 很少有研究评估模型的运行时、硬件兼容性、能耗或内存占用等部署相关因素，也未充分考虑对抗性攻击、遮挡或时间漂移的鲁棒性。
* **模态差距与上下文缺失：** 大多数城市VLM管道过度依赖静态图像-文本对，忽略了时间序列、深度图、地理定位和环境声音等丰富模态，限制了其在动态城市环境中的推理能力。
* **对资源密集型架构的过度依赖：** 许多最先进的VLM模型计算成本高昂，不适合实时、移动或嵌入式部署。
* **伦理盲点与法律疏忽：** 很少有研究将算法公平性、知情同意、数据来源和隐私保护等伦理维度整合到模型开发流程中。

**5. 潜在的未来研究方向：**
* **SLM-VLM混合架构：** 结合小型语言模型（SLMs）与模块化视觉编码器和解码器，实现高效的多模态推理，适用于边缘硬件部署。
* **统一的城市基准：** 开发集成多语言提示、多模态传感器流（图像、视频、音频、LiDAR）和文化多样性地理数据的评估套件，以确保可复现性、跨域可比性和鲁棒泛化。
* **以部署为中心的设计：** 将硬件限制、延迟要求、热预算和隐私考虑等部署约束嵌入模型开发周期。
* **嵌入式伦理与合规性：** 将文化鲁棒性检查、算法公平性评估、数据集同意追踪和偏见审计整合到核心基准测试和评估生命周期中。
* **可复现的开放生态系统：** 培养透明文化，通过版本化数据集、Docker化基线、公共排行榜和共享评估代码，促进社区驱动的倡议。

总而言之，这篇论文全面回顾了视觉-语言模型在城市监测领域的应用现状，揭示了其巨大潜力，同时也指出了当前研究在泛化能力、部署可行性和伦理考量方面的局限性，并提出了一个多维度的研究议程，旨在推动城市AI系统向更强大、可部署、包容、可解释和符合伦理的方向发展。

**Key Findings:**

- Current state-of-the-art approaches typically rely on a combination of IoT
sensors and manual inspections, which are costly, difficult to scale, and often
misaligned with citizens' perception formed through direct visual observation.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.12400v1)
- [arXiv](https://arxiv.org/abs/2510.12400v1)

---

<a id='2510.13804v1'></a>
## [Generative Universal Verifier as Multimodal Meta-Reasoner](https://arxiv.org/abs/2510.13804v1)

**Authors:** Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, Yujiu Yang

**Published:** 2025-10-15

**Categories:** cs.CV, cs.AI, cs.CL

**Abstract:**

We introduce Generative Universal Verifier, a novel concept and plugin
designed for next-generation multimodal reasoning in vision-language models and
unified multimodal models, providing the fundamental capability of reflection
and refinement on visual outcomes during the reasoning and generation process.
This work makes three main contributions: (1) We build ViVerBench, a
comprehensive benchmark spanning 16 categories of critical tasks for evaluating
visual outcomes in multimodal reasoning. Results show that existing VLMs
consistently underperform across these tasks, underscoring a substantial gap
from human-level capability in reliable visual verification. (2) We design two
automated pipelines to construct large-scale visual verification data and train
OmniVerifier-7B, the first omni-capable generative verifier trained for
universal visual verification and achieves notable gains on ViVerBench(+8.3).
Through training, we identify three atomic capabilities in visual verification
and demonstrate how they generalize and interact synergistically. (3) We
propose OmniVerifier-TTS, a sequential test-time scaling paradigm that
leverages the universal verifier to bridge image generation and editing within
unified models, enhancing the upper bound of generative ability through
iterative fine-grained optimization. Beyond generation, we extend universal
verifier to broader world-modeling interleaved reasoning scenarios.
Empirically, OmniVerifier-TTS achieves improvements on T2I-ReasonBench(+3.7),
and GenEval++(+4.3), outperforming existing parallel test-time scaling methods,
such as Best-of-N. By endowing multimodal reasoning with reliable visual
verification, OmniVerifier advances both reliable reflection during generation
and scalable test-time refinement, marking a step toward more trustworthy and
controllable next-generation reasoning systems.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Xinchen Zhang等人撰写的论文“Generative Universal Verifier as Multimodal Meta-Reasoner”的全面摘要。

---

**论文摘要：多模态元推理器——生成式通用验证器**

**1. 解决的主要问题或研究问题：**
该论文旨在解决下一代多模态推理中视觉结果验证的根本性挑战。现有的视觉-语言模型（VLMs）在多模态推理和生成过程中，对视觉结果进行可靠的反射和细化能力不足，导致其在验证视觉输出方面与人类水平存在显著差距。具体来说，论文探讨了如何系统地评估VLMs在视觉结果验证方面的表现，如何开发一个强大的生成式通用验证器，以及如何利用视觉验证来增强推理或生成能力。

**2. 关键创新或方法论贡献：**
*   **ViVerBench基准的构建：** 论文引入了一个全面且具有挑战性的基准ViVerBench，涵盖16类关键任务，用于评估多模态推理中的视觉结果。该基准通过人工标注精心构建，包含3,594个多样化且具有挑战性的验证问题，要求模型提供二元判断和详细解释。
*   **OmniVerifier-7B的训练与原子能力识别：** 论文设计了两种自动化数据构建流程，用于生成大规模视觉验证数据，并在此基础上训练了OmniVerifier-7B。这是首个为通用视觉验证而训练的全能生成式验证器。通过训练，论文识别出视觉验证的三种原子能力：显式对齐、关系验证和整合推理，并展示了它们如何协同泛化和交互。
*   **OmniVerifier-TTS的提出：** 论文提出了OmniVerifier-TTS，一种顺序测试时缩放范式。它利用通用验证器在统一模型中桥接图像生成和编辑，通过迭代细粒度优化来提升生成能力上限。该范式还扩展了通用验证器在更广泛的世界建模交错推理场景中的应用。

**3. 主要结果及其意义：**
*   **现有VLMs的不足：** ViVerBench上的实验结果表明，现有VLMs在视觉结果验证任务上表现不佳，与人类水平存在显著差距。具体表现为：在细粒度图像-提示对齐方面的弱点、世界知识表示不匹配以及视觉推理任务中批评能力不足。
*   **OmniVerifier-7B的卓越性能：** OmniVerifier-7B在ViVerBench上取得了显著的性能提升（+8.3），超越了GPT-4o，并在显式对齐和关系验证等任务上表现出显著改进。这表明通过针对原子能力进行强化学习训练，可以有效地构建更强大、更具泛化性的视觉验证器。
*   **OmniVerifier-TTS的生成增强：** OmniVerifier-TTS在T2I-ReasonBench（+3.7）和GenEval++（+4.3）上均实现了改进，优于现有的并行测试时缩放方法（如Best-of-N）。这证明了通过可靠的视觉验证，OmniVerifier能够实现生成过程中的可靠反射和可扩展的测试时细化，从而推动下一代推理系统更值得信赖和可控。

**4. 论文中提及的局限性：**
*   **任务泛化能力：** 某些任务（如迷宫）由于领域差距较大，泛化效果不佳，需要任务特定的数据进行优化。
*   **骨干模型的影响：** OmniVerifier-TTS的性能受其骨干模型的影响。目前，统一多模态模型对生成或编辑的图像分布敏感，在多步自细化过程中可能表现出异常行为（例如，GPT-Image-1在迭代编辑后倾向于生成偏黄的图像）。尽管这些伪影不影响验证性能，但它们是骨干模型本身的局限性。

**5. 潜在的未来研究方向：**
*   **增强通用验证器的泛化能力：** 未来的工作将探索训练和数据构建策略，以增强OmniVerifier的泛化能力，使其更接近真正的通用验证器。
*   **改进多模态后训练：** 进一步扩展通用验证器，并研究其在改进多模态后训练方面的潜力。
*   **解决骨干模型的风格一致性问题：** 鼓励进一步努力，在多步自细化下增强风格一致性。

---

这篇论文通过引入生成式通用验证器，为多模态推理和生成领域带来了重要的进展。它不仅提供了一个评估现有模型能力的全面基准，还提出了一种有效的方法来训练能够进行可靠视觉验证的模型，并将其应用于增强生成质量和推理能力，为构建更智能、更可控的AI系统奠定了基础。

**Key Findings:**

- We introduce Generative Universal Verifier, a novel concept and plugin
designed for next-generation multimodal reasoning in vision-language models and
unified multimodal models, providing the fundamental capability of reflection
and refinement on visual outcomes during the reasoning and generation process.
- This work makes three main contributions: (1) We build ViVerBench, a
comprehensive benchmark spanning 16 categories of critical tasks for evaluating
visual outcomes in multimodal reasoning.
- Results show that existing VLMs
consistently underperform across these tasks, underscoring a substantial gap
from human-level capability in reliable visual verification.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13804v1)
- [arXiv](https://arxiv.org/abs/2510.13804v1)

---

<a id='2510.13802v1'></a>
## [Trace Anything: Representing Any Video in 4D via Trajectory Fields](https://arxiv.org/abs/2510.13802v1)

**Authors:** Xinhang Liu, Yuxi Xiao, Donny Y. Chen, Jiashi Feng, Yu-Wing Tai, Chi-Keung Tang, Bingyi Kang

**Published:** 2025-10-15

**Categories:** cs.CV

**Abstract:**

Effective spatio-temporal representation is fundamental to modeling,
understanding, and predicting dynamics in videos. The atomic unit of a video,
the pixel, traces a continuous 3D trajectory over time, serving as the
primitive element of dynamics. Based on this principle, we propose representing
any video as a Trajectory Field: a dense mapping that assigns a continuous 3D
trajectory function of time to each pixel in every frame. With this
representation, we introduce Trace Anything, a neural network that predicts the
entire trajectory field in a single feed-forward pass. Specifically, for each
pixel in each frame, our model predicts a set of control points that
parameterizes a trajectory (i.e., a B-spline), yielding its 3D position at
arbitrary query time instants. We trained the Trace Anything model on
large-scale 4D data, including data from our new platform, and our experiments
demonstrate that: (i) Trace Anything achieves state-of-the-art performance on
our new benchmark for trajectory field estimation and performs competitively on
established point-tracking benchmarks; (ii) it offers significant efficiency
gains thanks to its one-pass paradigm, without requiring iterative optimization
or auxiliary estimators; and (iii) it exhibits emergent abilities, including
goal-conditioned manipulation, motion forecasting, and spatio-temporal fusion.
Project page: https://trace-anything.github.io/.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行深入分析。

---

**论文摘要分析：Trace Anything: Representing Any Video in 4D via Trajectory Fields**

**1. 论文主要贡献的简洁总结 (2-3 句话)**

这篇论文提出了一种新颖的视频表示方法——“轨迹场”（Trajectory Field），它将视频中的每个像素映射为一个连续的3D时间轨迹函数。基于此，作者引入了“Trace Anything”神经网络，该网络能够通过单次前向传播预测整个轨迹场，从而实现对视频中任何像素在任意时间点的3D位置的追踪和预测。

**2. 关键创新或方法论**

核心创新在于将视频的原子单元（像素）视为在时间上连续的3D轨迹，并提出了一种**轨迹场**的表示范式。传统方法通常关注离散的特征点追踪或光流估计，而轨迹场则提供了一种**稠密且连续的4D（3D空间+1D时间）表示**。

具体方法论是：
*   **轨迹场表示：** 将视频中的每个像素在每个帧中都关联一个连续的3D轨迹函数。
*   **参数化轨迹：** 使用一组控制点来参数化这些轨迹（例如，B样条），使得模型能够预测任意查询时间点的3D位置。
*   **单次前向传播预测：** “Trace Anything”神经网络通过一次前向传播直接预测所有像素的轨迹控制点，避免了迭代优化或依赖辅助估计器，显著提高了效率。
*   **大规模4D数据训练：** 模型在包括自建平台数据在内的大规模4D数据上进行训练，这暗示了对高质量、稠密轨迹标注数据的需求和获取能力。

**3. 对领域潜在影响**

*   **统一的视频表示：** 轨迹场提供了一种更基础、更统一的视频动态表示，可能成为未来视频理解和生成任务的基石。
*   **效率提升：** 单次前向传播的范式显著提高了轨迹预测的效率，使其更适用于实时应用或大规模视频处理。
*   **新能力涌现：** 摘要中提到的“目标条件操作”、“运动预测”和“时空融合”等新兴能力，表明这种稠密、连续的轨迹表示能够解锁更高级的视频理解和交互任务，超越了传统点追踪或光流的范畴。
*   **新的基准和研究方向：** 引入新的轨迹场估计基准，将推动该领域的研究进展，并可能激发更多基于轨迹场的新算法和应用。

**4. 相关领域或应用受益**

*   **视频理解与分析：** 更精确的运动理解、行为识别、事件检测。
*   **视频生成与编辑：** 运动风格迁移、视频插帧、物体移除与填充、视频内容创作。
*   **机器人学与自主驾驶：** 目标跟踪、运动预测、路径规划、场景理解。
*   **增强现实/虚拟现实 (AR/VR)：** 场景重建、运动追踪、虚拟物体与真实场景的融合。
*   **医学影像分析：** 器官运动追踪、细胞动力学分析。
*   **物理模拟与动画：** 更真实的物体运动模拟、角色动画。
*   **人机交互：** 手势识别、眼动追踪。

**5. 从摘要中可推断的局限性**

*   **数据依赖性：** 摘要强调了在“大规模4D数据”上进行训练，包括“新平台”的数据。这暗示了高质量、稠密、带有3D轨迹标注的数据集对于模型训练至关重要，获取和标注这类数据可能是一个巨大的挑战和成本。
*   **计算复杂度：** 尽管是单次前向传播，但为视频中“每个像素在每个帧”预测“一组控制点”来参数化轨迹，其计算量和内存消耗可能仍然非常大，尤其对于高分辨率、长时序视频。
*   **轨迹的平滑性和准确性：** B样条等参数化方法在表示复杂、非刚体或快速变化的运动时，其准确性和细节捕捉能力可能受到限制。模型预测的控制点数量和轨迹的阶数会影响其表达能力。
*   **泛化能力：** 模型在特定大规模4D数据上训练，其在与训练数据分布差异较大的真实世界复杂场景（如遮挡、光照变化、模糊、快速运动）下的泛化能力有待进一步验证。
*   **“新兴能力”的实现细节：** 摘要中提到的“目标条件操作”、“运动预测”等能力，其具体实现机制（例如，如何将目标条件融入轨迹预测）在摘要中并未详细说明，可能需要额外的模块或训练策略。

---

总的来说，这篇论文提出了一种非常具有前瞻性和潜力的视频表示方法，将视频理解推向了更深层次的4D时空连续性。其单次前向传播的效率优势和所展现出的新兴能力，使其在计算机视觉领域具有重要的研究价值和广泛的应用前景。

**Key Findings:**

- Based on this principle, we propose representing
any video as a Trajectory Field: a dense mapping that assigns a continuous 3D
trajectory function of time to each pixel in every frame.
- With this
representation, we introduce Trace Anything, a neural network that predicts the
entire trajectory field in a single feed-forward pass.
- We trained the Trace Anything model on
large-scale 4D data, including data from our new platform, and our experiments
demonstrate that: (i) Trace Anything achieves state-of-the-art performance on
our new benchmark for trajectory field estimation and performs competitively on
established point-tracking benchmarks; (ii) it offers significant efficiency
gains thanks to its one-pass paradigm, without requiring iterative optimization
or auxiliary estimators; and (iii) it exhibits emergent abilities, including
goal-conditioned manipulation, motion forecasting, and spatio-temporal fusion.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13802v1)
- [arXiv](https://arxiv.org/abs/2510.13802v1)

---

<a id='2510.13795v1'></a>
## [Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs](https://arxiv.org/abs/2510.13795v1)

**Authors:** Yi Zhang, Bolin Ni, Xin-Sheng Chen, Heng-Rui Zhang, Yongming Rao, Houwen Peng, Qinglin Lu, Han Hu, Meng-Hao Guo, Shi-Min Hu

**Published:** 2025-10-15

**Categories:** cs.CV, cs.AI

**Abstract:**

Fully open multimodal large language models (MLLMs) currently lag behind
proprietary counterparts, primarily due to a significant gap in data quality
for supervised fine-tuning (SFT). Existing open-source datasets are often
plagued by widespread noise and a critical deficit in complex reasoning data,
such as Chain-of-Thought (CoT), which hinders the development of advanced model
capabilities. Addressing these challenges, our work makes three primary
contributions. First, we introduce Honey-Data-15M, a new SFT dataset comprising
approximately 15 million QA pairs, processed through multiple cleaning
techniques and enhanced with a novel dual-level (short and long) CoT enrichment
strategy. Second, we introduce HoneyPipe, the data curation pipeline, and its
underlying framework DataStudio, providing the community with a transparent and
adaptable methodology for data curation that moves beyond static dataset
releases. Finally, to validate our dataset and pipeline, we train Bee-8B, an 8B
model on Honey-Data-15M. Experiments show that Bee-8B establishes a new
state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is
competitive with, and in some cases surpasses, recent semi-open models such as
InternVL3.5-8B. Our work delivers to the community a suite of foundational
resources, including: the Honey-Data-15M corpus; the full-stack suite
comprising HoneyPipe and DataStudio; training recipes; an evaluation harness;
and the model weights. This effort demonstrates that a principled focus on data
quality is a key pathway to developing fully open MLLMs that are highly
competitive with their semi-open counterparts.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供论文《Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs》的全面摘要。

**论文摘要：**

**1. 主要问题或研究问题：**
该论文旨在解决当前完全开源多模态大语言模型（MLLMs）在性能上落后于专有模型的核心问题。作者指出，这种差距主要源于监督微调（SFT）阶段数据质量的显著不足，具体表现为现有开源数据集普遍存在的噪声和复杂推理数据（如思维链CoT）的严重缺乏，这阻碍了模型高级能力的开发。

**2. 关键创新或方法论贡献：**
为了解决上述挑战，论文提出了三项主要贡献：
*   **Honey-Data-15M 数据集：** 引入了一个新的SFT数据集，包含约1500万个问答对。该数据集经过多重清洗技术处理，并通过新颖的双层（短CoT和长CoT）CoT富集策略进行增强，以提供不同深度的推理路径。
*   **HoneyPipe 数据策展流程与 DataStudio 框架：** 提出了一个透明且可适应的数据策展方法论，超越了静态数据集发布模式。HoneyPipe利用MLLMs自动化整个策展工作流，从清洗到富集，为开源社区提供了一种可扩展且经济高效的高质量数据构建方案。
*   **Bee-8B 模型：** 训练了一个8B参数模型Bee-8B，用于验证Honey-Data-15M数据集和HoneyPipe流程的有效性。

**3. 主要结果及其意义：**
*   **性能突破：** 实验结果表明，Bee-8B在完全开源MLLMs中建立了新的最先进（SOTA）性能，并且在某些情况下甚至超越了InternVL3.5-8B等近期半开源模型。
*   **数据策展的有效性：** 广泛的消融研究证实，数据策展过程（包括噪声过滤和CoT富集）对模型性能提升具有显著影响，尤其是在推理密集型基准测试上。这直接验证了高质量数据策略在弥合完全开源与半开源MLLMs性能差距方面的关键作用。
*   **资源发布：** 论文向社区提供了包括Honey-Data-15M语料库、包含HoneyPipe和DataStudio的全栈套件、训练方案、评估工具以及模型权重在内的基础资源。

**4. 论文中提及的局限性：**
论文中未明确提及当前工作的具体局限性，但其核心研究问题本身就暗示了现有开源MLLMs在数据质量和复杂推理能力上的不足。此外，论文强调了数据策展的成本效益和可扩展性，这可能间接说明了大规模人工标注的局限性。

**5. 潜在的未来研究方向：**
论文明确指出，未来的研究方向应继续关注数据质量，并通过透明、可复现的方法进行数据策展，而非仅仅追求数据量。这为开源社区开发具有高度竞争力的MLLMs指明了方向。此外，DataStudio框架的灵活性也为社区提供了不断迭代和改进数据策展方法的基础。

**Key Findings:**

- Addressing these challenges, our work makes three primary
contributions.
- First, we introduce Honey-Data-15M, a new SFT dataset comprising
approximately 15 million QA pairs, processed through multiple cleaning
techniques and enhanced with a novel dual-level (short and long) CoT enrichment
strategy.
- Second, we introduce HoneyPipe, the data curation pipeline, and its
underlying framework DataStudio, providing the community with a transparent and
adaptable methodology for data curation that moves beyond static dataset
releases.
- Experiments show that Bee-8B establishes a new
state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is
competitive with, and in some cases surpasses, recent semi-open models such as
InternVL3.5-8B.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13795v1)
- [arXiv](https://arxiv.org/abs/2510.13795v1)

---

<a id='2510.13778v1'></a>
## [InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy](https://arxiv.org/abs/2510.13778v1)

**Authors:** Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, Yang Tian, Bin Wang, Bolun Wang, Fangjing Wang, Hanqing Wang, Tai Wang, Ziqin Wang, Xueyuan Wei, Chao Wu, Shuai Yang, Jinhui Ye, Junqiu Yu, Jia Zeng, Jingjing Zhang, Jinyu Zhang, Shi Zhang, Feng Zheng, Bowen Zhou, Yangkun Zhu

**Published:** 2025-10-15

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

We introduce InternVLA-M1, a unified framework for spatial grounding and
robot control that advances instruction-following robots toward scalable,
general-purpose intelligence. Its core idea is spatially guided
vision-language-action training, where spatial grounding serves as the critical
link between instructions and robot actions. InternVLA-M1 employs a two-stage
pipeline: (i) spatial grounding pre-training on over 2.3M spatial reasoning
data to determine ``where to act'' by aligning instructions with visual,
embodiment-agnostic positions, and (ii) spatially guided action post-training
to decide ``how to act'' by generating embodiment-aware actions through
plug-and-play spatial prompting. This spatially guided training recipe yields
consistent gains: InternVLA-M1 outperforms its variant without spatial guidance
by +14.6% on SimplerEnv Google Robot, +17% on WidowX, and +4.3% on LIBERO
Franka, while demonstrating stronger spatial reasoning capability in box,
point, and trace prediction. To further scale instruction following, we built a
simulation engine to collect 244K generalizable pick-and-place episodes,
enabling a 6.2% average improvement across 200 tasks and 3K+ objects. In
real-world clustered pick-and-place, InternVLA-M1 improved by 7.3%, and with
synthetic co-training, achieved +20.6% on unseen objects and novel
configurations. Moreover, in long-horizon reasoning-intensive scenarios, it
surpassed existing works by over 10%. These results highlight spatially guided
training as a unifying principle for scalable and resilient generalist robots.
Code and models are available at
https://github.com/InternRobotics/InternVLA-M1.

**Analysis:**

以下是论文“InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy”的摘要：

**1. 主要问题或研究问题**
该论文旨在解决通用机器人领域的一个核心挑战：如何使机器人能够理解高级语言指令，并将其转化为在3D物理世界中精确、具身化的动作。现有方法在将文本抽象的指令与实际的、连续的机器人动作联系起来时存在根本性差距，尤其是在面对大规模、多样化任务时，难以实现可扩展的、通用的机器人智能。

**2. 关键创新或方法论贡献**
InternVLA-M1引入了一个统一的、空间引导的视觉-语言-动作（VLA）框架，其核心创新在于“空间引导的视觉-语言-动作训练”范式。该框架采用两阶段训练流程：
*   **空间接地预训练（Spatial Grounding Pre-training）：** 在超过230万的空间推理数据上进行预训练，以确定“在哪里行动”（where to act），通过将指令与视觉的、与具身无关的位置对齐。这建立了一个可迁移的空间先验知识。
*   **空间引导的动作后训练（Spatially Guided Action Post-training）：** 通过即插即用的空间提示（spatial prompting）生成具身感知的动作，以决定“如何行动”（how to act）。这种方法将空间先验知识转化为具体的运动控制。
*   **双系统架构：** InternVLA-M1采用双系统架构，包含一个VLM规划器（作为慢速但可靠的System 2推理器）和一个动作专家（作为快速的System 1控制器）。VLM规划器通过空间提示生成潜在规划令牌，指导动作专家生成控制信号。
*   **大规模合成数据引擎：** 构建了一个可扩展的模拟引擎，收集了24.4万个可泛化的抓取-放置（pick-and-place）任务片段，以进一步扩展指令遵循能力，并增强视觉多样性。

**3. 主要结果及其意义**
InternVLA-M1在多项基准测试和真实世界场景中展现出显著的性能提升和泛化能力：
*   **SimplerEnv基准：** 在SimplerEnv Google Robot上，性能比没有空间引导的变体提高了14.6%；在WidowX上提高了17%；在LIBERO Franka上提高了4.3%。同时，在盒子、点和轨迹预测方面展示了更强的空间推理能力。
*   **泛化抓取-放置任务：** 在200个任务和3000多个对象上，平均性能提高了6.2%。
*   **真实世界场景：** 在真实世界的聚类抓取-放置任务中，性能提高了7.3%；通过合成数据协同训练，在未见对象和新颖配置上实现了20.6%的提升。
*   **长时程推理任务：** 在长时程、推理密集型场景中，性能超越现有工作10%以上。
这些结果强调了空间引导训练作为实现可扩展和弹性通用机器人统一原则的有效性。

**4. 论文中提及的局限性**
论文中没有明确列出InternVLA-M1的局限性。然而，从其方法论和实验设置中可以推断出一些潜在的方面：
*   **数据依赖性：** 尽管使用了大规模合成数据，但模型的性能仍然依赖于训练数据的质量和多样性。对于某些高度复杂的、未充分表示的真实世界场景，可能仍存在泛化挑战。
*   **计算资源需求：** 模型的训练需要16块NVIDIA A100 GPU，这表明其训练过程对计算资源要求较高。
*   **推理速度：** 尽管通过FlashAttention和KV缓存进行了优化，VLM组件的推理速度约为10 FPS，对于某些需要极低延迟的实时交互任务，可能仍有提升空间。
*   **任务范围：** 尽管在抓取-放置、排序和长时程操作任务中表现出色，但其在更广泛、更开放的机器人任务（例如，需要复杂物理交互或精细操作的任务）中的表现仍需进一步探索。

**5. 潜在的未来研究方向**
论文中没有明确列出未来研究方向，但基于其贡献和潜在局限性，可以推断出以下方向：
*   **更广泛的任务泛化：** 探索InternVLA-M1在更多样化、更复杂的机器人任务中的应用，包括需要更精细运动控制、更复杂物理交互或更抽象推理的任务。
*   **效率优化：** 进一步优化模型的训练和推理效率，使其能够在更少的计算资源下运行，并实现更快的实时响应。
*   **数据效率：** 研究如何通过更少的数据（尤其是真实世界数据）实现相似的性能，例如通过更先进的自监督学习或领域适应技术。
*   **可解释性和鲁棒性：** 深入研究空间引导训练如何增强模型的可解释性和在未知环境中的鲁棒性，并探索如何进一步提升这些特性。
*   **多模态融合：** 探索除了视觉和语言之外，结合更多模态（如触觉、听觉）来增强空间接地和动作生成。
*   **持续学习和适应：** 研究如何使InternVLA-M1能够在新环境中持续学习和适应，而无需从头开始重新训练。

**Key Findings:**

- We introduce InternVLA-M1, a unified framework for spatial grounding and
robot control that advances instruction-following robots toward scalable,
general-purpose intelligence.
- This spatially guided training recipe yields
consistent gains: InternVLA-M1 outperforms its variant without spatial guidance
by +14.6% on SimplerEnv Google Robot, +17% on WidowX, and +4.3% on LIBERO
Franka, while demonstrating stronger spatial reasoning capability in box,
point, and trace prediction.
- In
real-world clustered pick-and-place, InternVLA-M1 improved by 7.3%, and with
synthetic co-training, achieved +20.6% on unseen objects and novel
configurations.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13778v1)
- [arXiv](https://arxiv.org/abs/2510.13778v1)

---

<a id='2510.13759v1'></a>
## [Uni-MMMU: A Massive Multi-discipline Multimodal Unified Benchmark](https://arxiv.org/abs/2510.13759v1)

**Authors:** Kai Zou, Ziqi Huang, Yuhao Dong, Shulin Tian, Dian Zheng, Hongbo Liu, Jingwen He, Bin Liu, Yu Qiao, Ziwei Liu

**Published:** 2025-10-15

**Categories:** cs.CV

**Abstract:**

Unified multimodal models aim to jointly enable visual understanding and
generation, yet current benchmarks rarely examine their true integration.
Existing evaluations either treat the two abilities in isolation or overlook
tasks that inherently couple them. To address this gap, we present Uni-MMMU, a
comprehensive and discipline-aware benchmark that systematically unfolds the
bidirectional synergy between generation and understanding across eight
reasoning-centric domains, including science, coding, mathematics, and puzzles.
Each task is bidirectionally coupled, demanding models to (i) leverage
conceptual understanding to guide precise visual synthesis, or (ii) utilize
generation as a cognitive scaffold for analytical reasoning. Uni-MMMU
incorporates verifiable intermediate reasoning steps, unique ground truths, and
a reproducible scoring protocol for both textual and visual outputs. Through
extensive evaluation of state-of-the-art unified, generation-only, and
understanding-only models, we reveal substantial performance disparities and
cross-modal dependencies, offering new insights into when and how these
abilities reinforce one another, and establishing a reliable foundation for
advancing unified models.

**Analysis:**

以下是论文“Uni-MMMU: A Massive Multi-discipline Multimodal Unified Benchmark”的全面摘要：

**1. 论文主要问题或研究问题**
当前统一多模态模型旨在同时实现视觉理解和生成，但现有基准测试未能充分评估这两种能力之间的真正整合。现有评估要么孤立地处理这两种能力，要么忽视了它们之间固有的耦合任务。论文旨在解决这一评估空白，系统地探究生成与理解之间的双向协同作用。

**2. 关键创新或方法论贡献**
*   **Uni-MMMU基准的提出：** 论文提出了Uni-MMMU，一个全面且跨学科的基准，系统地揭示了生成与理解在八个以推理为中心的领域（包括科学、编码、数学和谜题）中的双向协同作用。每个任务都是双向耦合的，要求模型利用概念理解指导精确的视觉合成，或利用生成作为分析推理的认知支架。
*   **双层评估协议：** Uni-MMMU整合了可验证的中间推理步骤、独特的真实值以及可重现的文本和视觉输出评分协议。这使得对最终结果和中间步骤进行双层评估成为可能，从而实现细粒度的错误归因。
*   **自动化和可重现的评估流程：** 评估流程采用程序化解析器、感知度量和LLM-as-a-Judge相结合的方式，确保了客观、一致和可解释的结果。

**3. 主要结果及其意义**
*   **性能差异和跨模态依赖性：** 对最先进的统一模型、仅生成模型和仅理解模型进行广泛评估后，揭示了显著的性能差异和跨模态依赖性。
*   **协同作用的有效性：** 结果表明，生成与理解之间的协同作用在具有严格逻辑依赖性的任务中最为有效。即使是不完美的中间生成结果，也能显著提高最终准确性。
*   **当前模型的局限性：** 分析揭示了当前统一模型存在明显不平衡，它们严重偏向理解能力，而生成能力是主要瓶颈。常见的失败点包括不精确的图像编辑、示意图的合成以及细粒度的空间推理。
*   **评估方法的有效性：** LLM-as-a-Judge评估组件的有效性通过与人类标注者和更强大的商业模型（Gemini-2.5-pro）的比较得到验证，Cohen's Kappa系数在0.6到0.8之间，表明高度一致性。

**4. 论文中提到的局限性**
*   **任务范围：** Uni-MMMU的任务主要集中在具有确定性和可验证解决方案的推理中心学科，这使得评估客观且可重现，但未能涵盖更广泛的真实世界场景，例如需要开放式创造力、主观判断或细微常识推理的任务。
*   **静态图像限制：** 当前基准完全基于静态图像，未来工作可以扩展到涉及视频或长期时间交互的任务。
*   **数据策展方法：** 迷宫导航、滑动拼图和代码渲染等任务采用程序化生成，这确保了唯一解决方案和客观解析，但可能导致数据缺乏真实世界图像的复杂性、噪声和视觉多样性。科学任务采用LLM驱动的流程和手动策展，可能引入生成模型或人类策展者的细微偏差。
*   **评估管道的局限性：** 某些任务依赖于“模型即评判者”框架，尽管已验证其与人类标注者的高度一致性，但这些评判模型并非万无一失，可能存在自身的偏见或知识空白，从而影响评估准确性。

**5. 潜在的未来研究方向**
*   **增强生成能力：** 解决当前统一模型在生成方面的瓶颈，特别是图像编辑、示意图合成和细粒度空间推理。
*   **更紧密的控制性：** 探索更紧密的控制机制（例如，程序或约束引导的生成）、跨编辑的更强空间/状态不变性，以及使可执行中间表示成为推理-生成循环中的一等公民的接口。
*   **扩展任务范围：** 将基准扩展到需要开放式创造力、主观判断或细微常识推理的真实世界场景。
*   **引入视频和时间交互：** 将评估扩展到涉及视频或长期时间交互的任务。
*   **改进评估方法：** 进一步完善“模型即评判者”框架，减少潜在偏差，提高评估准确性。

**Key Findings:**

- To address this gap, we present Uni-MMMU, a
comprehensive and discipline-aware benchmark that systematically unfolds the
bidirectional synergy between generation and understanding across eight
reasoning-centric domains, including science, coding, mathematics, and puzzles.
- Through
extensive evaluation of state-of-the-art unified, generation-only, and
understanding-only models, we reveal substantial performance disparities and
cross-modal dependencies, offering new insights into when and how these
abilities reinforce one another, and establishing a reliable foundation for
advancing unified models.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13759v1)
- [arXiv](https://arxiv.org/abs/2510.13759v1)

---

<a id='2510.13747v1'></a>
## [InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue](https://arxiv.org/abs/2510.13747v1)

**Authors:** Wenwen Tong, Hewei Guo, Dongchuan Ran, Jiangnan Chen, Jiefan Lu, Kaibin Wang, Keqiang Li, Xiaoxu Zhu, Jiakui Li, Kehan Li, Xueheng Li, Lumin Li, Chenxu Guo, Jiasheng Zhou, Jiandong Chen, Xianye Wu, Jiahao Wang, Silei Wu, Lei Chen, Hanming Deng, Yuxuan Song, Dinghao Zhou, Guiping Zhong, Ken Zheng, Shiyin Kang, Lewei Lu

**Published:** 2025-10-15

**Categories:** cs.CV

**Abstract:**

We introduce InteractiveOmni, a unified and open-source omni-modal large
language model for audio-visual multi-turn interaction, ranging from 4B to 8B
parameters, designed to lead the field of lightweight models by offering
comprehensive omni-modal understanding and speech generation capabilities. To
achieve this, we integrate the vision encoder, audio encoder, large language
model, and speech decoder into a unified model for understanding and generation
tasks. We design a multi-stage training strategy to ensure robust cross-modal
capabilities, including pre-training for omni-modal understanding, followed by
post-training with speech conversation and audio-visual interaction. To enable
human-like long-term conversational ability, we meticulously curate a
multi-turn training dataset that enhances the model's ability to handle complex
and multi-turn interactions. To effectively evaluate the multi-turn memory and
speech interaction capabilities, we construct the multi-modal multi-turn memory
benchmark and the multi-turn speech interaction benchmark. Experiments
demonstrate that InteractiveOmni significantly outperforms leading open-source
models and provides a more intelligent multi-turn audio-visual experience,
particularly in its long-term memory capabilities. Notably, InteractiveOmni-4B
is comparable to the much larger model like Qwen2.5-Omni-7B on general
benchmarks, and it can retain 97% of the performance of the InteractiveOmni-8B
while utilizing only 50% of the model size. Achieving state-of-the-art results
against similarly sized models across image, audio, video understanding, and
speech generation tasks, InteractiveOmni is an accessible, open-source
foundation for next-generation intelligent interactive systems.

**Analysis:**

以下是Wenwen Tong等人撰写的论文“InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue”的摘要：

**1. 主要问题或研究问题：**
当前的多模态大语言模型（MLLMs）在处理复杂、多轮的音频-视觉交互任务时，缺乏类人（human-like）的长期对话能力和无缝集成用户体验。它们主要关注单轮理解能力，未能提供端到端的全模态输入理解和语音响应生成能力。

**2. 关键创新或方法论贡献：**
*   **统一的全模态模型架构：** InteractiveOmni是一个统一的、开源的全模态大语言模型，能够同时接收图像、音频、文本和视频等全模态输入，并直接生成连贯的文本和语音流，实现真正的集成多轮交互。它集成了视觉编码器、音频编码器、大语言模型和语音解码器。
*   **多阶段训练策略：** 采用全模态预训练（用于跨模态理解）和后训练（用于语音对话和音频-视觉交互）的多阶段训练策略，以确保强大的跨模态能力。
*   **精心策划的多轮训练数据集：** 为了实现类人的长期对话能力，论文精心策划了一个多轮训练数据集，以增强模型处理复杂多轮交互的能力。
*   **新型多轮基准测试：** 构建了多模态多轮记忆基准（MMMB）和多轮语音交互基准（MSIB），以有效评估多轮记忆和语音交互能力，弥补现有基准的不足。

**3. 主要结果及其意义：**
*   **卓越的性能：** InteractiveOmni在图像、音频、视频理解和语音生成任务中，相对于同等规模的模型取得了最先进（state-of-the-art）的结果。
*   **轻量级模型的竞争力：** InteractiveOmni-4B在通用基准上与更大的Qwen2.5-Omni-7B模型相当，并且在模型尺寸仅为一半的情况下，仍能保持InteractiveOmni-8B 97%的性能。
*   **强大的长期记忆能力：** 实验证明InteractiveOmni显著优于领先的开源模型，提供了更智能的多轮音频-视觉体验，尤其在长期记忆能力方面表现突出。
*   **开源基础：** InteractiveOmni作为一个可访问的开源基础模型，为下一代智能交互系统奠定了基础。

**4. 论文中提及的局限性：**
论文摘要中并未明确提及InteractiveOmni模型的具体局限性。然而，从其研究背景和目标来看，当前MLLMs的普遍局限性（如多模态对齐的复杂性、端到端统一理解和生成框架的构建挑战、以及多轮对话中长期记忆和情感表达的不足）是InteractiveOmni旨在解决的问题。因此，可以推断这些是现有技术面临的挑战，而InteractiveOmni正努力克服它们。

**5. 潜在的未来研究方向：**
*   **提高模型效率：** 增强模型在实时交互中的效率。
*   **扩展理解复杂抽象跨模态关系的能力：** 进一步扩展模型理解更复杂、更抽象的跨模态关系的能力，为更真实、更类人的用户体验铺平道路。

**Key Findings:**

- We introduce InteractiveOmni, a unified and open-source omni-modal large
language model for audio-visual multi-turn interaction, ranging from 4B to 8B
parameters, designed to lead the field of lightweight models by offering
comprehensive omni-modal understanding and speech generation capabilities.
- Experiments
demonstrate that InteractiveOmni significantly outperforms leading open-source
models and provides a more intelligent multi-turn audio-visual experience,
particularly in its long-term memory capabilities.
- Achieving state-of-the-art results
against similarly sized models across image, audio, video understanding, and
speech generation tasks, InteractiveOmni is an accessible, open-source
foundation for next-generation intelligent interactive systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13747v1)
- [arXiv](https://arxiv.org/abs/2510.13747v1)

---

<a id='2510.13740v1'></a>
## [Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs](https://arxiv.org/abs/2510.13740v1)

**Authors:** Mustafa Munir, Alex Zhang, Radu Marculescu

**Published:** 2025-10-15

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Vision graph neural networks (ViG) have demonstrated promise in vision tasks
as a competitive alternative to conventional convolutional neural nets (CNN)
and transformers (ViTs); however, common graph construction methods, such as
k-nearest neighbor (KNN), can be expensive on larger images. While methods such
as Sparse Vision Graph Attention (SVGA) have shown promise, SVGA's fixed step
scale can lead to over-squashing and missing multiple connections to gain the
same information that could be gained from a long-range link. Through this
observation, we propose a new graph construction method, Logarithmic Scalable
Graph Construction (LSGC) to enhance performance by limiting the number of
long-range links. To this end, we propose LogViG, a novel hybrid CNN-GNN model
that utilizes LSGC. Furthermore, inspired by the successes of multi-scale and
high-resolution architectures, we introduce and apply a high-resolution branch
and fuse features between our high-resolution and low-resolution branches for a
multi-scale high-resolution Vision GNN network. Extensive experiments show that
LogViG beats existing ViG, CNN, and ViT architectures in terms of accuracy,
GMACs, and parameters on image classification and semantic segmentation tasks.
Our smallest model, Ti-LogViG, achieves an average top-1 accuracy on
ImageNet-1K of 79.9% with a standard deviation of 0.2%, 1.7% higher average
accuracy than Vision GNN with a 24.3% reduction in parameters and 35.3%
reduction in GMACs. Our work shows that leveraging long-range links in graph
construction for ViGs through our proposed LSGC can exceed the performance of
current state-of-the-art ViGs. Code is available at
https://github.com/mmunir127/LogViG-Official.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Mustafa Munir、Alex Zhang和Radu Marculescu撰写的论文《Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs》的全面摘要。

---

### 论文摘要：Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs

**1. 主要问题或研究问题：**
该论文旨在解决现有视觉图神经网络（ViG）在处理高分辨率图像时面临的计算效率和信息丢失问题。传统的图构建方法，如K近邻（KNN），在处理大型图像时计算成本高昂。而静态图构建方法，如稀疏视觉图注意力（SVGA），虽然简化了计算，但其固定的步长尺度会导致“过度压缩”（over-squashing），即在长距离连接中丢失重要信息，以及无法有效获取全局上下文。这限制了ViG在计算机视觉任务中的性能和可扩展性。

**2. 关键创新或方法论贡献：**
论文提出了以下关键创新：

*   **对数可伸缩图构建（Logarithmic Scalable Graph Construction, LSGC）：** 针对KNN和SVGA的局限性，LSGC提出了一种新的图构建方法。它利用图像尺寸的位深度（bit-depth）进行对数缩放，而非线性缩放。这使得图构建在处理高分辨率图像时能够生成更少的连接，从而减轻过度压缩并降低计算复杂度，同时通过优先考虑近距离连接来保持局部性，并通过长距离连接建立全局上下文。
*   **LogViG 混合CNN-GNN 模型：** 论文引入了一种新颖的混合CNN-GNN架构LogViG，它集成了LSGC。该模型在所有四个阶段都使用了卷积层和图层，以实现局部和全局处理。
*   **多尺度高分辨率架构：** LogViG借鉴了多尺度和高分辨率架构的成功经验，引入了一个高分辨率分支，并通过高分辨率快捷连接（High-Resolution Shortcut, HRS）在不同分辨率分支之间融合特征，以实现多尺度高分辨率视觉GNN网络。HRS通过两个3x3卷积（步长分别为2和1）将高分辨率特征注入模型的后期阶段，并通过双线性插值、逐点卷积和特征求和来融合低分辨率和高分辨率特征。
*   **网络深度与宽度优化：** 论文通过实验证明，更深更窄的网络架构（如Ti-LogViG）相比更宽更浅的网络能带来更好的性能提升。

**3. 主要结果及其重要性：**
论文通过在ImageNet-1K图像分类和ADE20K语义分割任务上的广泛实验，展示了LogViG的优越性能：

*   **图像分类性能：** 最小模型Ti-LogViG在ImageNet-1K上实现了79.9%的平均Top-1准确率（标准差±0.2%），比现有Vision GNN高出1.7%，同时参数减少了24.3%，GMACs减少了35.3%。S-LogViG在参数和GMACs相似的情况下，性能优于HRViT-b2、DeiT和PViG等模型。B-LogViG也显著优于EfficientFormer系列模型。
*   **语义分割性能：** S-LogViG在ADE20K上表现优于PoolFormer-S12、FastViT-SA12、EfficientFormer-L1和MobileViG-M，mIoU分别高出6.9、6.1、5.2和2.3。B-LogViG也优于FastViT-SA36和EfficientFormer-L3。
*   **效率提升：** LogViG在准确性、GMACs和参数方面均优于现有的ViG、CNN和ViT架构，证明了LSGC和混合架构在ViG设计上的显著进步。

**4. 论文中提到的局限性：**
论文中没有明确列出LogViG或LSGC的局限性。然而，可以从其设计和比较中推断出一些潜在的方面：

*   **计算复杂性：** 尽管LSGC旨在降低图构建的计算成本，但GNN固有的图操作仍然可能比纯CNN或ViT在某些场景下更复杂，尤其是在非常规硬件上。
*   **超参数敏感性：** LSGC中的扩展率K以及LogViG架构中的其他设计选择（如各阶段的通道维度和块数量）可能需要仔细调整以达到最佳性能。
*   **泛化能力：** 尽管在ImageNet-1K和ADE20K上表现出色，但LogViG在其他更广泛或更具挑战性的视觉任务上的泛化能力仍需进一步验证。

**5. 潜在的未来研究方向：**
论文没有明确提出未来的研究方向，但基于其贡献，可以推断出以下几点：

*   **LSGC的进一步优化：** 探索LSGC中对数缩放策略的更高级形式，或者结合自适应机制，使其能够根据图像内容或任务需求动态调整连接。
*   **LogViG架构的扩展：** 将LogViG应用于更广泛的计算机视觉任务，如目标检测、实例分割或视频理解，并探索其在这些任务中的性能。
*   **轻量化和部署：** 进一步优化LogViG模型，使其在资源受限的设备（如移动设备）上更高效地运行，可能通过剪枝、量化或知识蒸馏等技术。
*   **理论分析：** 对LSGC的图属性（如连通性、直径）进行更深入的理论分析，以更好地理解其在信息传播和全局上下文捕获方面的优势。
*   **与其他图构建方法的结合：** 探索将LSGC与其他先进的图构建或注意力机制相结合，以进一步提升性能或解决特定挑战。

---

**Key Findings:**

- Through this
observation, we propose a new graph construction method, Logarithmic Scalable
Graph Construction (LSGC) to enhance performance by limiting the number of
long-range links.
- To this end, we propose LogViG, a novel hybrid CNN-GNN model
that utilizes LSGC.
- Furthermore, inspired by the successes of multi-scale and
high-resolution architectures, we introduce and apply a high-resolution branch
and fuse features between our high-resolution and low-resolution branches for a
multi-scale high-resolution Vision GNN network.
- Our smallest model, Ti-LogViG, achieves an average top-1 accuracy on
ImageNet-1K of 79.9% with a standard deviation of 0.2%, 1.7% higher average
accuracy than Vision GNN with a 24.3% reduction in parameters and 35.3%
reduction in GMACs. Our work shows that leveraging long-range links in graph
construction for ViGs through our proposed LSGC can exceed the performance of
current state-of-the-art ViGs. Code is available at
https://github.com/mmunir127/LogViG-Official.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13740v1)
- [arXiv](https://arxiv.org/abs/2510.13740v1)

---

<a id='2510.13721v1'></a>
## [NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching](https://arxiv.org/abs/2510.13721v1)

**Authors:** Run Luo, Xiaobo Xia, Lu Wang, Longze Chen, Renke Shan, Jing Luo, Min Yang, Tat-Seng Chua

**Published:** 2025-10-15

**Categories:** cs.CL, cs.AI, cs.CV, cs.MM

**Abstract:**

Next-generation multimodal foundation models capable of any-to-any
cross-modal generation and multi-turn interaction will serve as core components
of artificial general intelligence systems, playing a pivotal role in
human-machine interaction. However, most existing multimodal models remain
constrained by autoregressive architectures, whose inherent limitations prevent
a balanced integration of understanding and generation capabilities. Although
hybrid and decoupling strategies have been explored to address these tasks
within unified frameworks separately, their redundant, non-integrated designs
limit their applicability to broader scenarios, such as cross-modal
retrieval.In this work, we introduce NExT-OMNI, an open-source omnimodal
foundation model that achieves unified modeling through discrete flow
paradigms. By leveraging metric-induced probability paths and kinetic optimal
velocities, NExT-OMNI natively supports any-to-any understanding and generation
with enhanced response efficiency, while enabling broader application scenarios
through concise unified representations rather than task-decoupled designs.
Trained on large-scale interleaved text, image, video, and audio data,
NExT-OMNI delivers competitive performance on multimodal generation and
understanding benchmarks, while outperforming prior unified models in
multi-turn multimodal interaction and cross-modal retrieval, highlighting its
architectural advantages as a next-generation multimodal foundation model. To
advance further research, we release training details, data protocols, and
open-source both the code and model checkpoints.

**Analysis:**

以下是Run Luo等人撰写的论文“NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching”的全面摘要：

**1. 主要问题或研究问题**

当前多模态基础模型在实现任意模态间的生成和多轮交互方面面临挑战。现有模型大多受限于自回归（AR）架构，这导致理解和生成能力难以平衡整合。尽管混合和解耦策略已被探索，但其冗余、非集成设计限制了它们在跨模态检索等更广泛场景中的应用。论文旨在解决这一问题，提出一个能够统一建模、实现任意模态理解和生成，并提升响应效率的多模态基础模型。

**2. 关键创新或方法论贡献**

*   **引入NEXT-OMNI模型：** 论文提出了NEXT-OMNI，一个基于离散流匹配（DFM）技术的开源全模态基础模型，首次实现了任意模态间的生成，并具有更快的推理速度。
*   **统一建模与离散流匹配范式：** NEXT-OMNI通过离散流匹配范式实现统一建模，利用度量诱导概率路径和动力学最优速度，原生支持任意模态的理解和生成，提高了响应效率。
*   **重建增强的统一表示：** 模型设计了带有中间特征融合的重建增强统一表示，这不仅实现了精确的跨模态检索，还支持多轮任意模态交互，避免了任务解耦的设计。
*   **动态长度生成策略与自适应缓存：** 为了提升理解任务的性能和加速推理，模型引入了动态长度生成策略和自适应缓存设计，显著提高了文本生成能力和推理速度。

**3. 主要结果及其意义**

*   **卓越的性能：** NEXT-OMNI在多模态生成和理解基准测试中表现出竞争性或卓越的性能，并在多轮多模态交互和跨模态检索方面超越了现有统一模型。例如，在OmniBench、WorldSense和AV-Odyssey等全模态理解基准测试中，NEXT-OMNI的平均性能比OpenOmni高出3.2个百分点。
*   **架构优势：** 实验结果突显了DFM架构作为下一代多模态基础模型的优势，尤其是在处理复杂的多模态交互和跨模态检索任务时。
*   **更快的推理速度：** 结合并行解码和自适应缓存机制，NEXT-OMNI的推理响应速度比AR架构提高了1.2倍。
*   **统一建模的潜力：** 结果验证了基于DFM的架构在统一多模态建模方面的巨大潜力，能够实现更广泛的应用场景。

**4. 论文中提及的局限性**

*   **资源限制：** 由于资源限制，模型仅在7B参数规模和2T token数据上进行训练和验证。因此，NEXT-OMNI的全部潜力尚未完全展现，尤其是在缺乏相应大型语言模型基础支持的情况下。

**5. 潜在的未来研究方向**

*   **扩展应用场景：** 未来研究计划将NEXT-OMNI扩展到更广泛的领域，例如视觉-语言-动作模型中的动作轨迹生成，以及物理AI理解中的视频生成，其中视觉生成可以辅助物理感知。
*   **模型规模化：** 进一步探索模型在更大规模数据和参数上的可扩展性，以充分发挥DFM架构的潜力。
*   **“世界大脑”愿景：** 论文展望统一多模态模型将成为与现实世界交互的“世界大脑”，通过持续的多模态数据交互不断增强其通用能力，最终实现通用人工智能（AGI）。

**Key Findings:**

- Although
hybrid and decoupling strategies have been explored to address these tasks
within unified frameworks separately, their redundant, non-integrated designs
limit their applicability to broader scenarios, such as cross-modal
retrieval.In this work, we introduce NExT-OMNI, an open-source omnimodal
foundation model that achieves unified modeling through discrete flow
paradigms.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.13721v1)
- [arXiv](https://arxiv.org/abs/2510.13721v1)

---

