time: 20251014

# Arxiv Computer Vision Papers - 2025-10-14

## Executive Summary

## Arxiv 计算机视觉每日报告执行摘要 (2025-10-13)

**概述：**

今天的 Arxiv 计算机视觉论文主要围绕**多模态学习、3D 内容生成与理解以及视频分析**展开。显著趋势包括利用大型语言模型 (LLMs) 增强视觉任务，以及在复杂场景下对 3D 人体、物体和视频进行更精细的建模和生成。

**主要主题和趋势：**

1.  **多模态与 LLM 融合：** 多篇论文探讨了将视觉与语言模型结合，以实现更强大的零样本学习、代理式多模态交互和全模态表示学习。这表明 LLMs 在计算机视觉领域的渗透日益加深，成为解决复杂推理和泛化问题的关键。
2.  **3D 内容生成与理解：** 3D 人体、场景和非刚性物体的生成与交互是另一个突出主题。从单目事件流生成新视角，到高保真全景图像生成，再到精确控制的 3D 人体创建和物理可信的 3D 人机交互，都显示出该领域在真实感和可控性方面的显著进步。
3.  **视频理解与分割：** 视频分析仍然是活跃的研究领域，特别是在复杂视频对象分割和实验视频理解与推理方面。这反映了对动态场景和时间序列数据更深层次理解的需求。
4.  **零样本与泛化能力：** 零样本学习作为一种重要的泛化范式，在多模态背景下得到了广泛关注，旨在使模型能够处理未见过的数据类别。

**特别重要或创新的论文：**

*   **"A Survey on Agentic Multimodal Large Language Models" (Huanjin Yao et al.)：** 这篇综述非常及时，因为它深入探讨了代理式多模态 LLMs，这是一个新兴且极具潜力的研究方向，预示着未来 AI 系统将具备更强的自主决策和多模态交互能力。
*   **"InfiniHuman: Infinite 3D Human Creation with Precise Control" (Yuxuan Xue et al.)：** 这篇论文在 3D 人体生成方面取得了显著突破，强调了“无限”和“精确控制”，这对于虚拟现实、游戏、电影制作以及数字人领域具有巨大应用价值。
*   **"PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image" (Pradyumna Yalandur Muralidhar et al.)：** 从单张图像推断物理可信的 3D 人机交互和接触是一个极具挑战性的任务，该工作在真实感和物理一致性方面迈出了重要一步，对机器人、人机交互和场景理解至关重要。
*   **"Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams" (Takuya Nakabayashi et al.)：** 利用单目事件流进行非刚性物体的新视角渲染，展示了事件相机在处理高速运动和低光照条件下的独特优势，为动态场景重建提供了新思路。

**新兴研究方向或技术：**

*   **代理式多模态 LLMs：** 将 LLMs 提升到具有代理能力，使其能够规划、执行和反思多模态任务，是未来 AI 发展的重要方向。
*   **事件相机在 3D 重建中的应用：** 事件流数据在处理高速非刚性物体和动态场景方面展现出巨大潜力。
*   **高保真、可控的 3D 内容生成：** 尤其是在 3D 人体和全景图像方面，强调了对生成内容细节和属性的精细控制。
*   **物理可信的 3D 交互建模：** 不仅仅是几何上的匹配，更要考虑物理规律和接触力学。

**建议阅读全文的论文：**

对于希望深入了解前沿进展的研究人员，建议优先阅读以下论文：

*   **"A Survey on Agentic Multimodal Large Language Models" (Huanjin Yao et al.)：** 了解该领域最新且最具潜力的方向。
*   **"InfiniHuman: Infinite 3D Human Creation with Precise Control" (Yuxuan Xue et al.)：** 如果您对 3D 人体生成和数字人技术感兴趣。
*   **"PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image" (Pradyumna Yalandur Muralidhar et al.)：** 如果您关注 3D 场景理解、人机交互或机器人领域。
*   **"Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams" (Takuya Nakabayashi et al.)：** 如果您对事件相机和动态 3D 重建感兴趣。
*   **"Scaling Language-Centric Omnimodal Representation Learning" (Chenghao Xiao et al.)：** 如果您关注多模态基础模型和通用表示学习。

这份摘要旨在帮助您快速把握今日 Arxiv 计算机视觉领域的关键发展，并为您的进一步研究提供方向。

---

## Table of Contents

1. [Compositional Zero-Shot Learning: A Survey](#2510.11106v1)
2. [LSVOS 2025 Challenge Report: Recent Advances in Complex Video Object Segmentation](#2510.11063v1)
3. [A Survey on Agentic Multimodal Large Language Models](#2510.10991v1)
4. [Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey](#2510.10671v1)
5. [Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams](#2510.11717v1)
6. [DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training](#2510.11712v1)
7. [Scaling Language-Centric Omnimodal Representation Learning](#2510.11693v1)
8. [InfiniHuman: Infinite 3D Human Creation with Precise Control](#2510.11650v1)
9. [PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image](#2510.11649v1)
10. [ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](#2510.11606v1)

---

## Papers

<a id='2510.11106v1'></a>
## [Compositional Zero-Shot Learning: A Survey](https://arxiv.org/abs/2510.11106v1)

**Authors:** Ans Munir, Faisal Z. Qureshi, Mohsen Ali, Muhammad Haris Khan

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

Compositional Zero-Shot Learning (CZSL) is a critical task in computer vision
that enables models to recognize unseen combinations of known attributes and
objects during inference, addressing the combinatorial challenge of requiring
training data for every possible composition. This is particularly challenging
because the visual appearance of primitives is highly contextual; for example,
``small'' cats appear visually distinct from ``older'' ones, and ``wet'' cars
differ significantly from ``wet'' cats. Effectively modeling this contextuality
and the inherent compositionality is crucial for robust compositional zero-shot
recognition. This paper presents, to our knowledge, the first comprehensive
survey specifically focused on Compositional Zero-Shot Learning. We
systematically review the state-of-the-art CZSL methods, introducing a taxonomy
grounded in disentanglement, with four families of approaches: no explicit
disentanglement, textual disentanglement, visual disentanglement, and
cross-modal disentanglement. We provide a detailed comparative analysis of
these methods, highlighting their core advantages and limitations in different
problem settings, such as closed-world and open-world CZSL. Finally, we
identify the most significant open challenges and outline promising future
research directions. This survey aims to serve as a foundational resource to
guide and inspire further advancements in this fascinating and important field.
Papers studied in this survey with their official code are available on our
github: https://github.com/ans92/Compositional-Zero-Shot-Learning

**Analysis:**

这篇由Ans Munir等人撰写的论文《Compositional Zero-Shot Learning: A Survey》全面概述了组合式零样本学习（CZSL）领域，该领域旨在使计算机视觉模型能够识别已知属性和对象的新颖组合，而无需针对每种组合进行显式训练。

**1. 主要问题或研究问题：**
CZSL的核心挑战在于，模型需要在推理时识别训练中未曾见过的属性与对象的组合。这尤其困难，因为基元（如属性和对象）的视觉外观具有高度语境依赖性（例如，“小”猫与“老”猫的视觉差异，以及“湿”车与“湿”猫的显著不同）。有效地建模这种语境性和固有的组合性对于鲁棒的组合式零样本识别至关重要。

**2. 关键创新或方法论贡献：**
该论文首次提出了一个全面的CZSL方法分类法，其核心思想是“解耦”（disentanglement）。分类法将现有方法分为四大类：
*   **无显式解耦（No Explicit Disentanglement）：** 将属性-对象组合视为单一单元，通过整体嵌入或直接融合机制进行建模。
*   **文本解耦（Textual Disentanglement）：** 在语言空间中分离基元的语义嵌入，通过独立概念表示实现系统组合。
*   **视觉解耦（Visual Disentanglement）：** 在视觉特征空间中分离属性和对象的视觉特征，将这些基元解耦为可组合的表示。
*   **跨模态（混合）解耦（Cross-Modal (Hybrid) Disentanglement）：** 同时在视觉和文本空间中解耦基元，并通过跨模态对齐整合互补信息。

在第二层，方法根据其建模属性和处理组合挑战的策略进一步分类，包括基于原型建模、合成嵌入、因果推理和约束驱动等。

**3. 主要结果及其意义：**
*   **骨干网络效应：** 早期使用ResNet编码器的方法性能较低，而基于CLIP编码器（从2023年开始）的最新方法在准确性上显著提高，表明视觉-语言预训练作为标准骨干网络的优势。
*   **解耦策略的有效性：**
    *   **无显式解耦** 方法通常处于性能谱的低端，凸显了将组合视为整体的局限性。
    *   **文本解耦** 方法比无解耦基线有所改进，但受限于仅依赖语言，无法捕捉视觉属性的变异性。
    *   **视觉解耦** 是最活跃且最具竞争力的方法类别，在闭环设置中表现出显著优势，甚至在某些情况下超越了混合方法。
    *   **跨模态解耦** 虽然起步较晚（主要在2024年出现），但在闭环设置中显示出最强的潜力，其编码器增强方法（如CAILA和Troika）在多样化数据集上已能与视觉基线匹敌。

*   **开放世界设置：** 在开放世界设置中，性能趋势与闭环设置有所不同。无解耦方法仍然是最弱的，但原始级别策略（如KG-SP）有时能与语境感知模型匹敌。文本解耦继续提供增益。视觉解耦仍然是最有效的方法，其中增强型（Retrieval）、原型中心型（CLUSPRO）和条件属性模型（CDS-CZSL）表现强劲。跨模态方法在开放世界评估中落后于顶级视觉方法，表明在扩展的标签空间下，仅视觉基础可能比联合视觉-文本对齐更可靠。

**4. 论文中提到的局限性：**
*   **无显式解耦：** 无法捕捉属性和对象的独特语义或其语境变异性，难以泛化到新颖组合。
*   **文本解耦：** 无法捕捉属性丰富的视觉变异性，忽略了图像中实际存在的纠缠。
*   **视觉解耦：** 强制严格分离可能过度简化自然依赖关系，丢弃有助于识别的语境线索。实现干净分离在实践中也很困难，特别是对于微妙或强语境依赖的属性。
*   **跨模态解耦：** 存在架构和计算复杂性，对齐两种模态通常需要额外的模块。一致的跨模态基础仍然难以实现，因为视觉基元在外观上差异很大，而文本基元相对稳定。

**5. 潜在的未来研究方向：**
*   **建模基元和语境性：** 进一步完善视觉解耦策略，并更加重视开发跨模态框架，以可扩展和鲁棒的方式捕捉语境性。
*   **扩展到开放世界评估：** 开发内在鲁棒于不可行组合的模型，缩小闭环和开放世界性能之间的差距，而无需显式可行性计算。
*   **泛化到未见基元：** 设计能够动态适应未见对象属性表示的模型，并探索跨模态解耦以利用语言模型的语义扩展能力。
*   **利用大型多模态模型（LMMs）：** 建立严格的评估协议以解决数据污染问题，并开发利用LMMs推理能力同时约束其学习真实组合结构而非表面相关性的适应策略。

总之，这篇综述为CZSL领域提供了一个结构化的分析框架，系统地回顾了现有方法，并指出了未来研究的关键方向，旨在推动该领域实现更鲁棒、可扩展和透明的组合式识别系统。

**Key Findings:**

- We
systematically review the state-of-the-art CZSL methods, introducing a taxonomy
grounded in disentanglement, with four families of approaches: no explicit
disentanglement, textual disentanglement, visual disentanglement, and
cross-modal disentanglement.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11106v1)
- [arXiv](https://arxiv.org/abs/2510.11106v1)

---

<a id='2510.11063v1'></a>
## [LSVOS 2025 Challenge Report: Recent Advances in Complex Video Object Segmentation](https://arxiv.org/abs/2510.11063v1)

**Authors:** Chang Liu, Henghui Ding, Kaining Ying, Lingyi Hong, Ning Xu, Linjie Yang, Yuchen Fan, Mingqi Gao, Jingkun Chen, Yunqi Miao, Gengshen Wu, Zhijin Qin, Jungong Han, Zhixiong Zhang, Shuangrui Ding, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Chang Soo Lim, Joonyoung Moon, Donghyeon Cho, Tingmin Li, Yixuan Li, Yang Yang, An Yan, Leilei Cao, Feng Lu, Ran Hong, Youhai Jiang, Fengjie Zhu, Yujie Xie, Hongyang Zhang, Zhihui Liu, Shihai Ruan, Quanzhu Niu, Dengxian Gong, Shihao Chen, Tao Zhang, Yikang Zhou, Haobo Yuan, Lu Qi, Xiangtai Li, Shunping Ji, Ran Hong, Feng Lu, Leilei Cao, An Yan, Alexey Nekrasov, Ali Athar, Daan de Geus, Alexander Hermans, Bastian Leibe

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

This report presents an overview of the 7th Large-scale Video Object
Segmentation (LSVOS) Challenge held in conjunction with ICCV 2025. Besides the
two traditional tracks of LSVOS that jointly target robustness in realistic
video scenarios: Classic VOS (VOS), and Referring VOS (RVOS), the 2025 edition
features a newly introduced track, Complex VOS (MOSEv2). Building upon prior
insights, MOSEv2 substantially increases difficulty, introducing more
challenging but realistic scenarios including denser small objects, frequent
disappear/reappear events, severe occlusions, adverse weather and lighting,
etc., pushing long-term consistency and generalization beyond curated
benchmarks. The challenge retains standard ${J}$, $F$, and ${J\&F}$ metrics for
VOS and RVOS, while MOSEv2 adopts ${J\&\dot{F}}$ as the primary ranking metric
to better evaluate objects across scales and disappearance cases. We summarize
datasets and protocols, highlight top-performing solutions, and distill
emerging trends, such as the growing role of LLM/MLLM components and
memory-aware propagation, aiming to chart future directions for resilient,
language-aware video segmentation in the wild.

**Analysis:**

好的，这是一篇关于“LSVOS 2025 Challenge Report: Recent Advances in Complex Video Object Segmentation”的摘要，由Chang Liu等撰写，内容基于您提供的PDF全文：

**论文摘要：LSVOS 2025 挑战赛报告：复杂视频目标分割的最新进展**

**1. 主要问题或研究问题：**
该报告旨在概述第七届大规模视频目标分割（LSVOS）挑战赛，该挑战赛与ICCV 2025同期举行。核心问题是推动视频目标分割（VOS）和指代视频目标分割（RVOS）在更复杂、更真实的视频场景中实现鲁棒性、长期一致性和泛化能力，超越现有策展基准的局限。特别是，新引入的复杂VOS (MOSEv2) 赛道旨在解决现有VOS方法在处理密集小物体、频繁出现/消失事件、严重遮挡、恶劣天气和光照等极端真实世界条件下的不足。

**2. 关键创新或方法论贡献：**
*   **引入MOSEv2数据集和赛道：** MOSEv2是MOSEv1的继任者，显著增加了难度，包含更多挑战性场景，如密集小物体、频繁出现/消失、严重遮挡、恶劣天气和光照等，旨在推动长期一致性和泛化能力。
*   **MOSEv2的新评估指标：** 为了更准确地评估跨对象尺度和消失情况下的性能，MOSEv2引入了新的指标，包括自适应边界精度（$\dot{F}$）和以$\text{J\&}\dot{F}$作为主要排名指标。
*   **顶尖解决方案的技术趋势：**
    *   **MOSEv2赛道：** 领先的解决方案（如DSS-Track的SeC框架）利用SAM-2（Segment Anything Model 2）和InternVL-2.5-4B等基础模型，通过增强的概念建模、更大的记忆尺寸（N=22）来捕获长期跨帧关系，并结合概念感知记忆（Concept-aware Memory）和场景自适应激活策略来处理复杂时空场景。
    *   **VOS赛道：** 顶尖方法（如NJUST-KMG）基于SAM2模型，通过置信度引导的多模型集成策略（结合SAM2Long、Cutie、LiVOS、XMem等）来增强鲁棒性，并采用多任务损失函数进行训练，以应对复杂场景。
    *   **RVOS赛道：** 领先方法（如SaSaSa2VA）结合多模态大语言模型（MLLM）和SAM2，通过关键帧压缩（KFC）和扩展[SEG] token数量来处理视频和文本指令，以捕获全局视频上下文和处理多样化的时间变化。还引入了视频语义匹配模块（VLC）以验证视频-文本对应关系。

**3. 主要结果及其意义：**
*   **MOSEv2的挑战性：** MOSEv2赛道的结果表明，即使是顶尖方法，其$\text{J\&}\dot{F}$分数也相对较低（第一名DSS-Track为39.89%），这突出显示了现代VOS方法在复杂真实场景中仍有显著提升空间。
*   **LLM/MLLM组件的日益重要性：** 报告强调，大语言模型（LLM）和多模态大语言模型（MLLM）已成为许多管道中的默认组件，尤其是在语言引导的视频任务中，这凸显了它们在视频理解方面的潜力。
*   **记忆感知传播的重要性：** 顶尖解决方案普遍采用了增强的记忆机制，无论是用于捕获长期跨帧关系，还是用于处理对象出现/消失，都强调了记忆管理在视频分割中的关键作用。

**4. 论文中提及的局限性：**
*   **MOSEv2的挑战性：** 尽管取得了进展，但MOSEv2赛道的结果表明，现有最先进系统在复杂、真实的场景中仍然面临挑战，导致性能显著下降。
*   **Sa2VA的局限性：** 在RVOS赛道中，Sa2VA模型在训练时仅采样5帧，并且使用单个[SEG] token来传递信息，这限制了MLLM捕获全局视频上下文的能力，并且难以适应对象位置、形状和外观的频繁变化。

**5. 潜在的未来研究方向：**
*   **更深层次的LLM/MLLM集成：** 报告预测，LLM/MLLM的更深层次集成将继续提升性能，尤其是在语言感知视频分割方面。
*   **解决最困难的失败模式：** 未来的研究将重点关注通过本次挑战赛结果和真实世界用例识别出的最困难的失败模式，以进一步推动视频目标分割及相关研究的前沿。

总而言之，LSVOS 2025挑战赛报告不仅展示了视频目标分割领域的最新进展，还通过引入MOSEv2数据集和新的评估指标，以及顶尖解决方案中涌现出的LLM/MLLM和记忆感知传播等趋势，为该领域的未来研究指明了方向。

**Key Findings:**

- Besides the
two traditional tracks of LSVOS that jointly target robustness in realistic
video scenarios: Classic VOS (VOS), and Referring VOS (RVOS), the 2025 edition
features a newly introduced track, Complex VOS (MOSEv2).

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11063v1)
- [arXiv](https://arxiv.org/abs/2510.11063v1)

---

<a id='2510.10991v1'></a>
## [A Survey on Agentic Multimodal Large Language Models](https://arxiv.org/abs/2510.10991v1)

**Authors:** Huanjin Yao, Ruifei Zhang, Jiaxing Huang, Jingyi Zhang, Yibo Wang, Bo Fang, Ruolin Zhu, Yongcheng Jing, Shunyu Liu, Guanbin Li, Dacheng Tao

**Published:** 2025-10-13

**Categories:** cs.CV, cs.AI, cs.CL

**Abstract:**

With the recent emergence of revolutionary autonomous agentic systems,
research community is witnessing a significant shift from traditional static,
passive, and domain-specific AI agents toward more dynamic, proactive, and
generalizable agentic AI. Motivated by the growing interest in agentic AI and
its potential trajectory toward AGI, we present a comprehensive survey on
Agentic Multimodal Large Language Models (Agentic MLLMs). In this survey, we
explore the emerging paradigm of agentic MLLMs, delineating their conceptual
foundations and distinguishing characteristics from conventional MLLM-based
agents. We establish a conceptual framework that organizes agentic MLLMs along
three fundamental dimensions: (i) Agentic internal intelligence functions as
the system's commander, enabling accurate long-horizon planning through
reasoning, reflection, and memory; (ii) Agentic external tool invocation,
whereby models proactively use various external tools to extend their
problem-solving capabilities beyond their intrinsic knowledge; and (iii)
Agentic environment interaction further situates models within virtual or
physical environments, allowing them to take actions, adapt strategies, and
sustain goal-directed behavior in dynamic real-world scenarios. To further
accelerate research in this area for the community, we compile open-source
training frameworks, training and evaluation datasets for developing agentic
MLLMs. Finally, we review the downstream applications of agentic MLLMs and
outline future research directions for this rapidly evolving field. To
continuously track developments in this rapidly evolving field, we will also
actively update a public repository at
https://github.com/HJYao00/Awesome-Agentic-MLLMs.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Huanjin Yao等人撰写的论文“A Survey on Agentic Multimodal Large Language Models”的全面摘要。

---

### 论文《A Survey on Agentic Multimodal Large Language Models》全面摘要

**1. 主要问题或研究问题：**
该论文旨在解决当前人工智能领域从传统静态、被动、领域专用型AI智能体向更动态、主动、通用型Agentic AI智能体转变的趋势。具体来说，它关注Agentic多模态大语言模型（Agentic MLLMs）这一新兴范式，并试图提供一个全面的综述，以阐明其概念基础、与传统MLLM智能体的区别、关键能力、训练与评估资源，以及未来的研究方向。核心问题是：如何系统地理解和推动Agentic MLLMs的发展，使其能够自主地进行长周期规划、主动使用工具并与动态环境交互，从而实现更接近通用人工智能（AGI）的目标。

**2. 关键创新或方法论贡献：**
该论文的主要贡献在于构建了一个全面的概念框架，将Agentic MLLMs的能力组织成三个基本维度：

*   **Agentic 内部智能（Agentic Internal Intelligence）：** 作为系统的“指挥官”，通过推理、反思和记忆实现准确的长周期规划。论文详细探讨了基于提示、SFT和RL的推理方法，以及显式和隐式反思机制，并区分了上下文记忆和外部记忆系统（包括启发式驱动和推理驱动）。
*   **Agentic 外部工具调用（Agentic External Tool Invocation）：** 模型主动利用各种外部工具（如信息搜索、代码执行、视觉处理）来扩展其超越内在知识的问题解决能力。论文分类讨论了Agentic搜索、Agentic编码和Agentic视觉处理（包括图像裁剪、图像操作和图像生成）等工具的使用。
*   **Agentic 环境交互（Agentic Environment Interaction）：** 模型能够与虚拟或物理环境进行交互，从而采取行动、调整策略并在动态真实世界场景中维持目标导向的行为。这包括虚拟GUI智能体（通过离线演示或在线交互学习）和物理具身AI（涉及具身感知、规划、导航和操作）。

此外，论文还：
*   **形式化了Agentic MLLMs与传统MLLM智能体的区别：** 强调了Agentic MLLMs的动态工作流、主动行动执行和跨领域泛化能力。
*   **系统梳理了训练与评估资源：** 整理了开源训练框架（包括Agentic CPT/SFT和RL框架）以及用于开发和评估Agentic MLLMs的训练和评估数据集。

**3. 主要结果及其意义：**
该论文本身是一篇综述，因此其“结果”体现在对现有研究的全面梳理和洞察上，而非实验结果。主要意义包括：

*   **提供统一的视角：** 首次全面地对Agentic MLLMs领域进行了系统性分类和总结，为研究人员提供了一个清晰、全面的领域概览和概念框架。
*   **揭示发展趋势：** 强调了Agentic MLLMs在自主决策、主动规划、工具使用和环境交互方面的优势，预示了AI智能体向更通用、更智能方向发展的趋势。
*   **促进社区发展：** 汇编了大量的开源资源（训练框架、数据集），为研究人员进入和加速该领域的研究提供了宝贵的起点。
*   **突出应用潜力：** 展示了Agentic MLLMs在深度研究、具身AI、医疗保健、GUI智能体、自动驾驶和推荐系统等广泛下游应用中的巨大潜力，表明其能够处理复杂的真实世界场景。

**4. 论文中提及的局限性：**
论文中明确指出了Agentic MLLMs当前面临的挑战和局限性：

*   **行动空间受限：** 现有模型的行动空间和可访问工具范围通常受限于单一类型，缺乏更广泛的外部工具和服务集成。
*   **效率问题：** 多轮推理和外部工具调用等迭代过程显著增加了计算和推理开销，导致训练和推理效率低下，难以满足实时应用和大规模部署的需求。
*   **长周期记忆的局限性：** 当前系统的记忆有效长度高度受限，难以在更长的时间跨度内维持连贯的知识，且多模态记忆管理方面的研究不足。
*   **训练与评估数据稀缺：** 专门为Agentic行为设计的训练数据集仍然稀缺，尤其是在多模态领域缺乏足够的探索。现有评估基准主要关注Agentic行为的特定方面，而记忆利用、跨工具调用协调等能力仍缺乏有效的评估数据集。
*   **安全问题：** 随着Agentic MLLMs在规划、工具调用和环境交互方面日益自主，确保其安全性（如避免意外后果、处理不正确或有害信息、防止行为不稳定）成为一个关键的研究优先事项。

**5. 潜在的未来研究方向：**
基于上述局限性，论文提出了以下未来研究方向：

*   **更丰富的行动空间：** 开发能够无缝集成更多外部工具和服务（如数据分析平台、仿真环境、多模态传感器、交互式API）的Agentic MLLMs。
*   **提高效率：** 专注于提升Agentic MLLMs的计算效率，加速训练和推理过程，同时不牺牲性能，以支持实时应用和大规模部署。
*   **长周期Agentic记忆：** 设计持久性记忆架构，使模型能够跨长时间跨度积累、组织和检索知识，并能够处理海量的多模态数据流，实现个性化、持续协作和自适应问题解决。
*   **Agentic训练与评估数据集：** 开发有效且高效的方法来合成高质量的多模态Agentic轨迹数据。同时，构建更全面的评估基准，以评估记忆利用、跨工具调用协调以及行动执行的准确性等关键Agentic能力。
*   **Agentic MLLMs的安全性：** 结合严格的基准测试、对抗性压力测试和规范框架的集成，确保Agentic MLLMs在更广泛的自主性方面保持可靠、可控并与人类意图对齐。

---

**Key Findings:**

- Motivated by the growing interest in agentic AI and
its potential trajectory toward AGI, we present a comprehensive survey on
Agentic Multimodal Large Language Models (Agentic MLLMs).

**Links:**

- [PDF](https://arxiv.org/pdf/2510.10991v1)
- [arXiv](https://arxiv.org/abs/2510.10991v1)

---

<a id='2510.10671v1'></a>
## [Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey](https://arxiv.org/abs/2510.10671v1)

**Authors:** Jinxuan Li, Chaolei Tan, Haoxuan Chen, Jianxin Ma, Jian-Fang Hu, Wei-Shi Zheng, Jianhuang Lai

**Published:** 2025-10-12

**Categories:** cs.CV, cs.AI

**Abstract:**

Image-Language Foundation Models (ILFM) have demonstrated remarkable success
in image-text understanding/generation tasks, providing transferable multimodal
representations that generalize across diverse downstream image-based tasks.
The advancement of video-text research has spurred growing interest in
extending image-based models to the video domain. This paradigm, known as
image-to-video transfer learning, succeeds in alleviating the substantial data
and computational requirements associated with training video-language
foundation models from scratch for video-text learning. This survey provides
the first comprehensive review of this emerging field, which begins by
summarizing the widely used ILFM and their capabilities. We then systematically
classify existing image-to-video transfer learning strategies into two
categories: frozen features and modified features, depending on whether the
original representations from ILFM are preserved or undergo modifications.
Building upon the task-specific nature of image-to-video transfer, this survey
methodically elaborates these strategies and details their applications across
a spectrum of video-text learning tasks, ranging from fine-grained (e.g.,
spatio-temporal video grounding) to coarse-grained (e.g., video question
answering). We further present a detailed experimental analysis to investigate
the efficacy of different image-to-video transfer learning paradigms on a range
of downstream video understanding tasks. Finally, we identify prevailing
challenges and highlight promising directions for future research. By offering
a comprehensive and structured overview, this survey aims to establish a
structured roadmap for advancing video-text learning based on existing ILFM,
and to inspire future research directions in this rapidly evolving domain.

**Analysis:**

好的，这是一篇关于Jinxuan Li等人撰写的论文“Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey”的全面摘要，内容基于您提供的摘要和PDF全文：

**论文全面摘要：Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey**

**1. 主要问题或研究问题 (Main Problem or Research Question Addressed)**

该论文主要解决的问题是：如何有效地将图像-语言基础模型（Image-Language Foundation Models, ILFM）的强大能力，迁移到视频理解任务中，以应对视频数据特有的时序维度挑战，并缓解从头训练视频-语言基础模型所需的大量数据和计算资源。具体来说，它旨在为基于现有ILFM的视频-文本学习建立一个结构化的路线图，并激发该领域未来的研究方向。

**2. 关键创新或方法论贡献 (Key Innovations or Methodological Contributions)**

该论文的主要贡献在于对图像到视频迁移学习这一新兴领域进行了首次全面且结构化的综述。其关键创新和方法论贡献包括：

*   **系统性分类迁移策略：** 论文将现有的图像到视频迁移学习策略系统地分为两大类：
    *   **冻结特征（Frozen Features）：** 保持ILFM的原始表示不变，通过知识蒸馏、后网络微调（Post-Network Tuning）和侧调（Side-Tuning）等方法进行迁移。
    *   **修改特征（Modified Features）：** 对ILFM的原始表示进行修改，包括全微调（Full Fine-Tuning）、部分微调（Partial Tuning）、使用额外模型微调（Fine-Tuning with Extra Models）、基于适配器微调（Fine-Tuning with Adapter）、LoRA微调（Fine-Tuning with LoRA）和提示微调（Prompt Tuning）。
*   **任务特定应用阐述：** 论文根据任务的粒度（从细粒度到粗粒度）详细阐述了这些迁移策略在各种视频-文本学习任务中的应用，例如：
    *   **细粒度任务：** 时空视频定位（Spatio-Temporal Video Grounding, STVG）、开放词汇多目标跟踪（Open Vocabulary Multi-Object Tracking, OV-MOT）、视频实例分割（Video Instance Segmentation, OV-VIS）等，这些任务需要精确的空间区域和时间间隔定位。
    *   **粗粒度任务：** 视频-文本检索（Video-Text Retrieval, VTR）、视频动作识别（Video Action Recognition, VAR）、视频问答（Video Question Answering, VideoQA）、视频字幕生成（Video Captioning）等，这些任务更侧重于对视频事件的整体理解。
*   **详细的实验分析：** 论文提供了详细的实验分析，通过比较不同图像到视频迁移学习范式在各种下游视频理解任务上的效果，验证了不同策略的有效性。例如，在TVG任务中，R2-tuning（侧调）在CLIP基础上表现最佳，而NumPro（额外模型微调）在LLaVA基础上表现突出。在VTR任务中，LoRA微调在CLIP基础上取得了最佳性能。在VAR任务中，适配器策略表现优于侧调。在VideoQA任务中，VideoDistill（知识蒸馏）在CLIP基础上表现优异，而BIMBA（后网络微调）在LLaVA基础上显著提升了性能。
*   **结构化路线图：** 通过提供全面且结构化的概述，论文旨在为基于现有ILFM的视频-文本学习建立一个结构化的路线图，并激发该领域未来的研究方向。

**3. 主要结果及其意义 (Main Results and Their Significance)**

论文的实验分析揭示了以下主要结果及其意义：

*   **迁移学习的有效性：** 图像到视频迁移学习范式在缓解从头训练视频-语言基础模型所需的大量数据和计算资源方面取得了成功。
*   **策略与任务的匹配性：** 没有单一的迁移策略在所有任务上都表现最佳。不同任务对模型能力有不同要求，因此需要选择最合适的迁移范式。例如，细粒度任务（如TVG）通常受益于能够精确建模时空关系的策略，而粗粒度任务（如VideoQA）可能更侧重于多模态知识理解的鲁棒性。
*   **基础模型的选择：** 不同的ILFM（如CLIP、MDETR、GroundingDINO、BLIP、LLaVA）具有不同的优势。例如，MDETR和GroundingDINO在细粒度视觉-文本对应方面表现更强，而CLIP和BLIP更擅长处理抽象或高级视觉-文本对应。LLaVA通过利用大型语言模型（LLM）弥合了这两种极端情况。
*   **LLM的潜力：** 将大型语言模型（LLM）作为基础模型进行迁移，在某些任务（如VideoQA）中显示出显著的性能提升，这表明LLM强大的语义推理能力对视频理解至关重要。
*   **参数效率的重要性：** 适配器、LoRA和提示微调等参数高效的微调方法在资源受限的环境下具有很高的实用性和可扩展性，同时也能保持ILFM的强大能力。

**4. 论文中提及的局限性 (Limitations Mentioned in the Paper)**

论文中提及的现有方法的局限性包括：

*   **冻结特征的局限性：** 冻结特征方法（如知识蒸馏、后网络微调、侧调）的性能可能受限于静态图像级特征固有的表示能力，以及它们与视频特定任务要求的不完美对齐，尤其是在泛化到新场景时。它们可能引入ILFM的错误知识，从而限制跨模态视频-文本理解的有效性。
*   **全微调的计算成本：** 全微调虽然简单有效，但需要大规模、定义良好且标注的数据，以及大量的计算资源，在实践中适用性较低。
*   **额外模型的复杂性：** 虽然额外模型微调可以有效注入视频特定的结构信息，但也会增加计算成本。
*   **LoRA和提示微调的局限性：** LoRA本身无法建模时序信息，需要依赖外部时序建模模块或网络。提示微调虽然计算成本低，但其对冻结骨干网络的依赖可能会限制细粒度时序推理能力。
*   **现有解决方案的碎片化：** 当前研究中，针对不同视频任务往往需要选择不同的基础模型和设计独特的微调策略，导致解决方案碎片化，缺乏统一性和通用性。
*   **时序维度建模的挑战：** 视频包含复杂的时空动态，包括运动、事件进程和因果交互，这使得从处理静态空间信息到有效建模这些复杂时空关系需要根本性的架构范式转变。

**5. 潜在的未来研究方向 (Potential Future Research Directions)**

论文提出了以下几个有前景的未来研究方向：

*   **统一迁移学习范式：** 开发一个统一的迁移学习框架，使单个ILFM能够同时有效地适应多个视频-语言任务，从而实现更通用、更高效的视频智能系统。这可能涉及探索基于提示的学习或设计参数高效的统一适配器/超网络。
*   **多基础模型协作：** 有效整合多个预训练的基础模型，每个模型专注于不同的模态或能力，以解决单个视频-语言任务。研究可以集中于融合机制、知识蒸馏和跨模型注意力，以利用互补优势并降低计算成本。
*   **高级双模态融合方法：** 进一步研究更动态、高效的融合技术，如跨模态Transformer、基于图的对齐或轻量级特征交互模块，以改善视觉和语言特征在时空维度上的对齐和融合，这对于细粒度理解和长篇视频推理至关重要。
*   **视频编辑和生成：** 扩展图像到视频迁移学习技术，以支持视频编辑和生成任务。这需要有效建模时序连贯性和动态运动，例如通过集成时序注意力、利用辅助模态（如深度图、光流）或开发新的扩散模型架构。
*   **开放词汇能力：** 进一步增强模型在开放词汇场景下的泛化能力，使其能够处理训练中未见过的对象类别和事件。

总而言之，这篇综述为图像到视频迁移学习领域提供了一个全面的技术蓝图，不仅总结了现有方法，还指出了未来的发展方向，对于推动视频-文本理解领域的发展具有重要意义。

**Key Findings:**

- Finally, we identify prevailing
challenges and highlight promising directions for future research.
- By offering
a comprehensive and structured overview, this survey aims to establish a
structured roadmap for advancing video-text learning based on existing ILFM,
and to inspire future research directions in this rapidly evolving domain.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.10671v1)
- [arXiv](https://arxiv.org/abs/2510.10671v1)

---

<a id='2510.11717v1'></a>
## [Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams](https://arxiv.org/abs/2510.11717v1)

**Authors:** Takuya Nakabayashi, Navami Kairanda, Hideo Saito, Vladislav Golyanik

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

Event cameras offer various advantages for novel view rendering compared to
synchronously operating RGB cameras, and efficient event-based techniques
supporting rigid scenes have been recently demonstrated in the literature. In
the case of non-rigid objects, however, existing approaches additionally
require sparse RGB inputs, which can be a substantial practical limitation; it
remains unknown if similar models could be learned from event streams only.
This paper sheds light on this challenging open question and introduces Ev4DGS,
i.e., the first approach for novel view rendering of non-rigidly deforming
objects in the explicit observation space (i.e., as RGB or greyscale images)
from monocular event streams. Our method regresses a deformable 3D Gaussian
Splatting representation through 1) a loss relating the outputs of the
estimated model with the 2D event observation space, and 2) a coarse 3D
deformation model trained from binary masks generated from events. We perform
experimental comparisons on existing synthetic and newly recorded real datasets
with non-rigid objects. The results demonstrate the validity of Ev4DGS and its
superior performance compared to multiple naive baselines that can be applied
in our setting. We will release our models and the datasets used in the
evaluation for research purposes; see the project webpage:
https://4dqv.mpi-inf.mpg.de/Ev4DGS/.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Takuya Nakabayashi, Navami Kairanda, Hideo Saito, Vladislav Golyanik撰写的论文“Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams”的全面摘要。

---

**论文摘要：Ev4DGS: 从单目事件流渲染非刚性物体的新视角**

这篇论文《Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams》首次提出了一种从单目事件流中渲染非刚性变形物体新视角的方法。传统上，事件相机在刚性场景的新视角合成方面已显示出优势，但对于非刚性物体，现有方法通常仍依赖于稀疏的RGB输入，这限制了其在纯事件流场景下的应用。Ev4DGS旨在解决这一挑战性问题，探索是否仅凭事件流就能学习到非刚性物体的模型。

**1. 主要问题或研究问题：**
该研究主要解决的核心问题是：如何仅使用单目事件流，实现对非刚性变形物体的高质量新视角渲染？现有方法在处理非刚性物体时，通常需要结合RGB图像，而纯事件流的解决方案仍是未知的。这篇论文旨在填补这一空白，并证明仅通过事件数据进行非刚性3D重建和新视角合成是可行的。

**2. 关键创新或方法论贡献：**
Ev4DGS引入了以下关键创新和方法论贡献：
*   **首个纯事件流非刚性新视角渲染方法：** Ev4DGS是第一个能够从单目事件流中渲染非刚性变形物体（以RGB或灰度图像形式）新视角的方法，无需任何RGB输入。
*   **两阶段训练策略：** 为了解决从单目观测中重建可变形物体的高度病态问题，Ev4DGS采用两阶段训练。
    *   **粗略变形模型（Coarse Stage）：** 第一阶段训练一个粗略的3D变形模型，该模型将非刚性物体形状表示为一组时间依赖的粗略点云，并能捕捉场景中的大尺度变形。它通过学习低秩基点云的线性组合来表示物体随时间演化的状态，并利用从事件生成的二值掩码进行监督。
    *   **精细3D高斯溅射表示（Fine Stage）：** 第二阶段在此粗略模型的基础上，利用4D高斯溅射（3DGS的泛化）来表示物体的精细外观，实现动态新视角合成。高斯参数通过粗略变形模型驱动，并共享时间信息。
*   **事件损失和轮廓损失：** 针对事件数据的特性，论文设计了特定的损失函数：
    *   **事件损失（Event Loss）：** 将估计模型的输出与2D事件观测空间关联起来，通过比较渲染图像的亮度变化与事件流中累积的亮度差异来优化3D高斯参数。
    *   **轮廓损失（Silhouette Loss）：** 利用从事件流生成的二值掩码，抑制背景中不必要的高斯点，从而提高图像质量。
*   **自监督学习：** 整个框架通过事件流和相机跟踪信息实现自监督学习，无需额外输入。
*   **新数据集：** 论文创建了新的合成和真实数据集，以评估Ev4DGS在非刚性物体上的性能，填补了现有数据集的不足。

**3. 主要结果及其意义：**
*   **优越的性能：** Ev4DGS在合成和真实数据集上均表现出卓越的性能，在PSNR指标上平均比现有基线方法（如3DGS和D3DGS）高出约10%，同时在SSIM上也具有竞争力。
*   **高质量新视角渲染：** 实验结果表明，Ev4DGS能够生成高质量、空间和时间连贯的新视角渲染，准确捕捉变形物体的外观和形状，而竞争方法往往会丢失物体内部的细节或出现模糊。
*   **纯事件流的可行性：** 论文成功证明了仅使用事件流进行非刚性3D重建和新视角合成的可行性，避免了中间帧重建带来的信息损失和误差。
*   **对掩码质量的敏感性：** 消融研究显示，模型性能对二值掩码的质量敏感。通过Snake算法直接从事件流生成掩码比通过E2VID+SAM从重建图像生成掩码效果更好，但仍不如使用真实（GT）二值掩码。

**4. 论文中提及的局限性：**
*   **外观伪影：** 尽管Ev4DGS取得了最高的准确性，但模型的一个局限性是渲染的高斯点可能导致外观上出现伪影（如单个或成组渲染高斯点）。
*   **对掩码质量的依赖：** 模型的性能在一定程度上依赖于二值掩码的质量。如果能直接从事件流生成更准确的二值掩码，性能有望进一步提升。
*   **优化难度：** 增加基向量K的数量虽然能增强模型的运动表达能力，但也会增加优化的难度。

**5. 潜在的未来研究方向：**
*   **改进掩码生成：** 探索更准确、更鲁棒的直接从事件流生成二值掩码的方法，以进一步提升Ev4DGS的性能。
*   **减少外观伪影：** 研究如何优化高斯溅射表示或渲染过程，以减少渲染图像中的外观伪影，提高视觉真实感。
*   **更复杂的变形：** 探索Ev4DGS在更复杂、更大范围的非刚性变形场景中的应用和改进。
*   **实时性能优化：** 尽管论文提到了全重建时间约为4小时，但未来可以研究如何进一步优化训练和渲染过程，以实现更接近实时的性能。
*   **多模态融合：** 虽然Ev4DGS专注于纯事件流，但未来也可以探索与少量RGB输入或其他传感器数据的更有效融合，以在特定应用中进一步提升性能。

---

总而言之，Ev4DGS是计算机视觉领域的一项重要进展，它为从单目事件流中进行非刚性物体的新视角渲染开辟了新途径，展示了事件相机在处理动态场景方面的巨大潜力。

**Key Findings:**

- Event cameras offer various advantages for novel view rendering compared to
synchronously operating RGB cameras, and efficient event-based techniques
supporting rigid scenes have been recently demonstrated in the literature.
- This paper sheds light on this challenging open question and introduces Ev4DGS,
i.e., the first approach for novel view rendering of non-rigidly deforming
objects in the explicit observation space (i.e., as RGB or greyscale images)
from monocular event streams.
- Our method regresses a deformable 3D Gaussian
Splatting representation through 1) a loss relating the outputs of the
estimated model with the 2D event observation space, and 2) a coarse 3D
deformation model trained from binary masks generated from events.
- We perform
experimental comparisons on existing synthetic and newly recorded real datasets
with non-rigid objects.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11717v1)
- [arXiv](https://arxiv.org/abs/2510.11717v1)

---

<a id='2510.11712v1'></a>
## [DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training](https://arxiv.org/abs/2510.11712v1)

**Authors:** Haoran Feng, Dizhe Zhang, Xiangtai Li, Bo Du, Lu Qi

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

In this work, we propose DiT360, a DiT-based framework that performs hybrid
training on perspective and panoramic data for panoramic image generation. For
the issues of maintaining geometric fidelity and photorealism in generation
quality, we attribute the main reason to the lack of large-scale, high-quality,
real-world panoramic data, where such a data-centric view differs from prior
methods that focus on model design. Basically, DiT360 has several key modules
for inter-domain transformation and intra-domain augmentation, applied at both
the pre-VAE image level and the post-VAE token level. At the image level, we
incorporate cross-domain knowledge through perspective image guidance and
panoramic refinement, which enhance perceptual quality while regularizing
diversity and photorealism. At the token level, hybrid supervision is applied
across multiple modules, which include circular padding for boundary
continuity, yaw loss for rotational robustness, and cube loss for distortion
awareness. Extensive experiments on text-to-panorama, inpainting, and
outpainting tasks demonstrate that our method achieves better boundary
consistency and image fidelity across eleven quantitative metrics. Our code is
available at https://github.com/Insta360-Research-Team/DiT360.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行详细分析。

---

**论文摘要分析：DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training**

**1. 论文主要贡献的简洁总结 (2-3 句话)**

DiT360 提出了一种基于 DiT 的全景图像生成框架，通过在透视数据和全景数据上进行混合训练来解决全景图像生成中的几何保真度和真实感问题。该方法的核心在于其数据中心视角，通过在图像和 token 级别引入跨域转换和域内增强模块，显著提升了生成质量、边界一致性和图像保真度。

**2. 关键创新或方法论**

该论文的关键创新在于其**混合训练范式和数据中心视角**，这与以往专注于模型设计的方法形成对比。具体方法论包括：

*   **混合训练 (Hybrid Training)：** 在透视数据和全景数据上同时进行训练，利用透视数据丰富的细节和真实感来弥补全景数据稀缺的不足。
*   **多层次跨域知识融合：**
    *   **图像级别 (Pre-VAE Image Level)：** 引入透视图像引导 (perspective image guidance) 和全景精炼 (panoramic refinement)，以增强感知质量，同时规范多样性和真实感。这表明模型在编码器之前就利用了透视图像的结构信息。
    *   **Token 级别 (Post-VAE Token Level)：** 在 VAE 编码后的潜在空间 (token level) 应用混合监督，包括：
        *   **循环填充 (Circular Padding)：** 确保全景图像左右边界的连续性，这是全景图特有的挑战。
        *   **偏航损失 (Yaw Loss)：** 增强旋转鲁棒性，使生成内容在不同视角下保持一致。
        *   **立方体损失 (Cube Loss)：** 提高对全景图像固有畸变的感知和处理能力，可能通过将全景图投影到立方体贴图来计算损失。
*   **数据中心视角：** 强调缺乏大规模、高质量、真实世界全景数据是生成质量问题的根本原因，并通过上述混合训练和多层次策略来有效利用现有数据。

**3. 对领域潜在影响**

DiT360 有望对计算机视觉领域产生以下潜在影响：

*   **提升全景图像生成质量标准：** 通过解决几何保真度和真实感的核心问题，DiT360 可能成为全景图像生成领域的新基线，推动更高质量的生成结果。
*   **启发新的数据利用策略：** 其混合训练和数据中心视角为处理特定领域数据稀缺问题提供了新的思路，尤其是在需要跨域知识迁移的生成任务中。
*   **推动全景内容创作和应用：** 更高质量的全景生成能力将直接赋能虚拟现实 (VR)、增强现实 (AR)、元宇宙、360° 视频制作、游戏环境生成等领域，降低内容创作门槛。
*   **为 DiT 架构的应用提供新方向：** 展示了 DiT 架构在处理复杂几何和拓扑结构数据（如全景图）方面的潜力，可能激发 DiT 在其他非标准图像格式生成任务中的应用。

**4. 相关领域或应用**

以下领域或应用将从这项研究中受益：

*   **虚拟现实 (VR) 和增强现实 (AR)：** 生成逼真的 360° 环境和纹理，用于沉浸式体验。
*   **元宇宙 (Metaverse)：** 快速创建和填充虚拟世界的全景背景和场景。
*   **360° 视频和图像编辑：** 文本到全景图生成、全景图修复 (inpainting) 和扩展 (outpainting) 将极大地简化 360° 媒体的后期制作。
*   **游戏开发：** 自动生成游戏场景的背景天空盒 (skybox) 或环境贴图。
*   **机器人和自动驾驶：** 生成多样化的全景环境数据用于训练和测试感知系统，尤其是在数据采集困难的场景。
*   **建筑和室内设计：** 快速生成不同设计方案的全景渲染图。

**5. 从摘要中可推断的局限性**

尽管摘要强调了显著的进步，但仍可推断出一些潜在局限性：

*   **计算资源需求：** 混合训练，尤其是在图像和 token 级别都进行复杂操作，可能需要大量的计算资源（GPU 内存和计算时间），这对于个人研究者或小型团队可能是一个挑战。
*   **数据依赖性：** 尽管强调了数据中心视角，但其性能仍可能高度依赖于所使用的透视数据和少量全景数据的质量和多样性。如果透视数据本身存在偏差或不足，可能会影响最终生成质量。
*   **泛化能力：** 摘要中未提及模型在处理极端复杂场景、高度动态内容或特定风格全景图时的泛化能力。例如，在生成具有复杂几何结构或精细纹理的全景图时，是否能始终保持高保真度。
*   **“缺乏大规模、高质量、真实世界全景数据”的根本问题：** 尽管 DiT360 旨在缓解这一问题，但它并未从根本上解决全景数据稀缺的挑战。如果未来有大量高质量全景数据可用，其混合训练策略可能需要调整或优化。
*   **特定畸变处理：** 尽管提到了“立方体损失”来处理畸变，但全景图的畸变是复杂的，尤其是在极点附近。摘要中未详细说明其对所有类型畸变（如极点拉伸）的处理效果。
*   **评估指标的局限性：** 摘要提到“十一个定量指标”，但这些指标是否能完全捕捉全景图像的视觉质量、几何准确性和用户体验，仍需在论文正文中详细探讨。例如，人类感知评估（user study）通常是不可或缺的。

---

总而言之，DiT360 提出了一种新颖且实用的全景图像生成方法，通过其独特的混合训练和多层次处理策略，有望在解决全景图生成中的核心挑战方面取得显著进展。其数据中心视角和对 DiT 架构的创新应用，使其成为计算机视觉领域一个值得关注的研究方向。

**Key Findings:**

- In this work, we propose DiT360, a DiT-based framework that performs hybrid
training on perspective and panoramic data for panoramic image generation.
- Extensive experiments on text-to-panorama, inpainting, and
outpainting tasks demonstrate that our method achieves better boundary
consistency and image fidelity across eleven quantitative metrics.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11712v1)
- [arXiv](https://arxiv.org/abs/2510.11712v1)

---

<a id='2510.11693v1'></a>
## [Scaling Language-Centric Omnimodal Representation Learning](https://arxiv.org/abs/2510.11693v1)

**Authors:** Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied, Yu Rong

**Published:** 2025-10-13

**Categories:** cs.CL, cs.AI, cs.CV

**Abstract:**

Recent multimodal embedding approaches leveraging multimodal large language
models (MLLMs) fine-tuned with contrastive learning (CL) have shown promising
results, yet the underlying reasons behind their superiority remain
underexplored. This work argues that a crucial advantage of MLLM-based
approaches stems from implicit cross-modal alignment achieved during generative
pretraining, where the language decoder learns to exploit multimodal signals
within a shared representation space for generating unimodal outputs. Through
analysis of anisotropy and kernel similarity structure, we empirically confirm
that latent alignment emerges within MLLM representations, allowing CL to serve
as a lightweight refinement stage. Leveraging this insight, we propose a
Language-Centric Omnimodal Embedding framework, termed LCO-Emb. Extensive
experiments across diverse backbones and benchmarks demonstrate its
effectiveness, achieving state-of-the-art performance across modalities.
Furthermore, we identify a Generation-Representation Scaling Law (GRSL),
showing that the representational capabilities gained through contrastive
refinement scales positively with the MLLM's generative capabilities. This
suggests that improving generative abilities evolves as an effective paradigm
for enhancing representation quality. We provide a theoretical explanation of
GRSL, which formally links the MLLM's generative quality to the upper bound on
its representation performance, and validate it on a challenging, low-resource
visual-document retrieval task, showing that continual generative pretraining
before CL can further enhance the potential of a model's embedding
capabilities. Codes, models, and resources are available at
https://github.com/LCO-Embedding/LCO-Embedding.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Chenghao Xiao等人撰写的论文“Scaling Language-Centric Omnimodal Representation Learning”的全面摘要。

---

### 论文摘要：Scaling Language-Centric Omnimodal Representation Learning

**1. 主要问题或研究问题**

该论文旨在解决一个核心问题：尽管基于多模态大语言模型（MLLM）并结合对比学习（CL）的嵌入方法在多模态任务中表现出色，但其优越性的深层原因尚未被充分探索。具体来说，研究者希望理解MLLM在生成式预训练过程中如何实现隐式跨模态对齐，以及这种对齐如何影响其表征能力，并为后续的对比学习提供基础。

**2. 关键创新或方法论贡献**

*   **MLLM中潜在跨模态对齐的发现与分析：** 论文通过对各模态（文本、图像、音频、视频）嵌入空间的各向异性（anisotropy）和核级相似性（kernel-level similarity）结构进行实证分析，发现MLLM在生成式预训练中已实现隐式跨模态对齐。文本模态的对比学习不仅提高了文本嵌入的可区分性，还泛化性地增强了非文本模态嵌入的可区分性，使其更具各向同性。
*   **提出语言中心全模态嵌入框架（LCO-EMB）：** 基于上述发现，论文提出了LCO-EMB框架。该框架利用语言中心配对数据进行高效的对比学习微调，将对比学习视为轻量级的精炼阶段，而非从头开始的对齐机制。LCO-EMB通过LoRA（Low-Rank Adaptation）技术对MLLM进行表征激活，旨在最小化扰动预训练的生成能力和潜在跨模态对齐。
*   **识别并理论解释生成-表征缩放定律（GRSL）：** 论文通过实验观察到，MLLM的表征能力（通过对比学习精炼后）与其生成能力呈正相关。研究者进一步提供了GRSL的理论解释，通过PAC-贝叶斯泛化界限，形式化地将MLLM的生成质量与其表征性能的上限联系起来，表明生成能力越强，表征潜力越大。
*   **引入SeaDoc基准任务：** 为了验证GRSL，论文引入了SeaDoc，这是一个具有挑战性的低资源视觉文档检索任务，用于评估MLLM在跨语言多模态文档理解方面的表征能力。

**3. 主要结果及其意义**

*   **LCO-EMB的卓越性能：** LCO-EMB在MIEB-Lite基准测试中，即使仅使用少量（约0.37M）训练对，也显著优于现有的多模态嵌入模型，并在多种模态任务中达到最先进的性能。这表明MLLM的内在跨模态对齐能力是其性能优势的关键。
*   **LoRA的有效性：** LoRA作为一种参数高效的微调方法，在保持模型生成能力和潜在跨模态对齐的同时，有效提升了表征能力，优于传统的CLIP风格对比学习和全量微调。
*   **GRSL的实证验证：** 实验结果一致表明，基线生成性能与对比学习后的表征性能之间存在正相关。这证实了GRSL，并提出通过提升MLLM的生成能力来增强其多模态表征质量是一种有效范式。
*   **SeaDoc任务的验证：** 在SeaDoc任务上，持续的生成式预训练（尤其是在高分辨率和结合通用图像标注数据的情况下）能进一步提升模型的嵌入能力，支持了GRSL的观点。

**4. 论文中提及的局限性**

*   **计算成本：** 论文指出，虽然可以联合训练生成损失和对比损失以同时保持模型知识和增强表征能力，但这种方法计算成本较高。
*   **LoRA超参数的优化：** 论文提到，LoRA的rank (r) 和 alpha (a) 超参数没有一个全局最优设置，其最佳值可能因模型大小而异，需要在引入新知识和修改预训练模型权重之间取得平衡。

**5. 潜在的未来研究方向**

*   **联合生成与对比学习：** 鉴于联合训练生成损失和对比损失的计算成本较高，未来的工作可以探索更高效的方法来实现这一目标，以进一步提升全模态表征学习。
*   **LoRA超参数的全面分析：** 对LoRA超参数进行更全面的实证分析和理论研究，以量化其与模型性能之间的关系，并找到更通用的优化策略。
*   **GRSL的进一步探索：** 深入研究GRSL，探索如何通过改进生成能力来系统性地提升表征质量，这可能涉及新的生成式预训练范式或模型架构。

---

这篇论文为理解MLLM在多模态嵌入领域的成功提供了深刻见解，并提出了一个新颖的框架和重要的缩放定律，为未来多模态表征学习的研究指明了方向。

**Key Findings:**

- Leveraging this insight, we propose a
Language-Centric Omnimodal Embedding framework, termed LCO-Emb.
- Extensive
experiments across diverse backbones and benchmarks demonstrate its
effectiveness, achieving state-of-the-art performance across modalities.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11693v1)
- [arXiv](https://arxiv.org/abs/2510.11693v1)

---

<a id='2510.11650v1'></a>
## [InfiniHuman: Infinite 3D Human Creation with Precise Control](https://arxiv.org/abs/2510.11650v1)

**Authors:** Yuxuan Xue, Xianghui Xie, Margaret Kostyrko, Gerard Pons-Moll

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

Generating realistic and controllable 3D human avatars is a long-standing
challenge, particularly when covering broad attribute ranges such as ethnicity,
age, clothing styles, and detailed body shapes. Capturing and annotating
large-scale human datasets for training generative models is prohibitively
expensive and limited in scale and diversity. The central question we address
in this paper is: Can existing foundation models be distilled to generate
theoretically unbounded, richly annotated 3D human data? We introduce
InfiniHuman, a framework that synergistically distills these models to produce
richly annotated human data at minimal cost and with theoretically unlimited
scalability. We propose InfiniHumanData, a fully automatic pipeline that
leverages vision-language and image generation models to create a large-scale
multi-modal dataset. User study shows our automatically generated identities
are undistinguishable from scan renderings. InfiniHumanData contains 111K
identities spanning unprecedented diversity. Each identity is annotated with
multi-granularity text descriptions, multi-view RGB images, detailed clothing
images, and SMPL body-shape parameters. Building on this dataset, we propose
InfiniHumanGen, a diffusion-based generative pipeline conditioned on text, body
shape, and clothing assets. InfiniHumanGen enables fast, realistic, and
precisely controllable avatar generation. Extensive experiments demonstrate
significant improvements over state-of-the-art methods in visual quality,
generation speed, and controllability. Our approach enables high-quality avatar
generation with fine-grained control at effectively unbounded scale through a
practical and affordable solution. We will publicly release the automatic data
generation pipeline, the comprehensive InfiniHumanData dataset, and the
InfiniHumanGen models at https://yuxuan-xue.com/infini-human.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行分析。

---

**论文摘要分析：InfiniHuman: Infinite 3D Human Creation with Precise Control**

**1. 论文主要贡献的简洁总结 (2-3 句话)**

这篇论文的核心贡献在于提出了一种名为 InfiniHuman 的框架，该框架通过蒸馏现有基础模型，以极低的成本和理论上无限的可扩展性生成了大规模、多样化且富含标注的 3D 人体数据（InfiniHumanData）。在此基础上，它还引入了一个扩散模型 InfiniHumanGen，能够实现快速、逼真且精确可控的 3D 人体化身生成，显著超越了现有技术。

**2. 关键创新或方法论方法**

该论文的关键创新在于其“数据蒸馏”和“数据驱动生成”的方法论：

*   **数据蒸馏与自动标注 (InfiniHumanData):** 核心思想是利用现有的视觉-语言模型和图像生成模型，以全自动的方式生成大规模、多模态的 3D 人体数据集。这解决了传统方法中数据采集和标注成本高昂、多样性受限的问题。通过这种方式，他们能够生成包含 111K 身份的数据集，涵盖了前所未有的多样性，并为每个身份提供了多粒度文本描述、多视角 RGB 图像、详细服装图像和 SMPL 身体形状参数。用户研究表明其自动生成的人体与扫描渲染图无异，这证明了数据质量。
*   **扩散模型驱动的精确控制生成 (InfiniHumanGen):** 在 InfiniHumanData 的基础上，他们开发了一个基于扩散的生成管道 InfiniHumanGen。这个模型能够以文本、身体形状和服装资产为条件，实现对 3D 人体化身的快速、逼真且精确的控制生成。这使得用户可以根据具体需求，通过多模态输入来定制化身。

**3. 对该领域的潜在影响**

InfiniHuman 对计算机视觉和图形学领域具有深远的潜在影响：

*   **打破数据瓶颈：** 解决了 3D 人体生成领域长期存在的数据稀缺和标注成本高昂的问题，为训练更强大的生成模型提供了“无限”的数据源。
*   **提升生成质量与控制力：** 显著提高了 3D 人体化身生成的视觉真实感、速度和精细控制能力，使得创建高度定制化的虚拟人变得更加可行。
*   **推动基础模型应用：** 展示了如何巧妙地利用和蒸馏现有的大型基础模型（如视觉-语言模型和图像生成模型）来解决特定领域的复杂问题，为其他领域的数据生成提供了新的范式。
*   **降低技术门槛：** 通过提供自动数据生成管道、数据集和模型，降低了研究人员和开发者进入 3D 人体生成领域的门槛。

**4. 可能受益于这项研究的相关领域或应用**

*   **虚拟现实 (VR) 和增强现实 (AR):** 创建高度逼真和可定制的虚拟化身，用于社交 VR、游戏、虚拟试穿等。
*   **电影、动画和游戏产业：** 快速生成大量多样化的角色模型，大幅缩短制作周期和成本。
*   **数字时尚和虚拟试穿：** 生成具有不同体型、肤色和服装风格的虚拟模特，用于服装设计、展示和在线购物体验。
*   **人体姿态估计和动作捕捉：** 生成多样化的人体数据可以作为训练数据，提高这些任务的鲁棒性。
*   **医疗和健康领域：** 生成不同体型和年龄段的人体模型，用于医学模拟、康复训练或人体工程学研究。
*   **人机交互 (HCI):** 创建更具表现力和个性化的虚拟助手或机器人形象。
*   **计算机图形学研究：** 为 3D 建模、渲染、动画和角色绑定等研究提供高质量的基准数据和生成工具。

**5. 从摘要中可以推断出的任何局限性**

尽管摘要展示了令人印象深刻的成果，但仍可推断出一些潜在的局限性：

*   **“理论上无限”与实际计算资源：** 尽管数据生成是“理论上无限”的，但实际生成和存储如此大规模的数据仍需要大量的计算资源和存储空间。
*   **基础模型的依赖性：** InfiniHuman 的性能在很大程度上依赖于所蒸馏的视觉-语言模型和图像生成模型的质量和能力。如果这些基础模型存在偏差或局限性，可能会传递到生成的数据和模型中。
*   **“ undistinguishable from scan renderings”的范围：** 用户研究表明自动生成的人体与扫描渲染图“无异”，但这通常是在特定条件下（例如，特定视角、光照、分辨率）进行的。在极端特写、复杂光照或高精度物理模拟等场景下，是否仍能保持这种“无异”的水平，需要进一步验证。
*   **SMPL 模型的局限性：** 摘要提到使用 SMPL 身体形状参数。SMPL 模型虽然广泛使用，但它是一个参数化模型，可能无法捕捉到所有细微的人体解剖学细节或非标准体型。
*   **服装资产的来源和多样性：** 摘要提到“detailed clothing images”和“clothing assets”。这些服装资产的来源、多样性和质量将直接影响生成结果的真实感和可控性。如果服装资产本身有限，那么生成的多样性也会受限。
*   **实时性或交互性：** 摘要强调了“fast”生成，但并未明确说明是否支持实时交互式生成，这对于某些应用（如 VR/AR）至关重要。

---

总的来说，InfiniHuman 是一项具有开创性的工作，它通过巧妙地利用现有 AI 基础模型，为 3D 人体生成领域带来了革命性的数据生成和模型训练范式，有望在多个应用领域产生深远影响。

**Key Findings:**

- We introduce
InfiniHuman, a framework that synergistically distills these models to produce
richly annotated human data at minimal cost and with theoretically unlimited
scalability.
- We propose InfiniHumanData, a fully automatic pipeline that
leverages vision-language and image generation models to create a large-scale
multi-modal dataset.
- Building on this dataset, we propose
InfiniHumanGen, a diffusion-based generative pipeline conditioned on text, body
shape, and clothing assets.
- Extensive experiments demonstrate
significant improvements over state-of-the-art methods in visual quality,
generation speed, and controllability.
- Our approach enables high-quality avatar
generation with fine-grained control at effectively unbounded scale through a
practical and affordable solution.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11650v1)
- [arXiv](https://arxiv.org/abs/2510.11650v1)

---

<a id='2510.11649v1'></a>
## [PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image](https://arxiv.org/abs/2510.11649v1)

**Authors:** Pradyumna Yalandur Muralidhar, Yuxuan Xue, Xianghui Xie, Margaret Kostyrko, Gerard Pons-Moll

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

Reconstructing metrically accurate humans and their surrounding scenes from a
single image is crucial for virtual reality, robotics, and comprehensive 3D
scene understanding. However, existing methods struggle with depth ambiguity,
occlusions, and physically inconsistent contacts. To address these challenges,
we introduce PhySIC, a framework for physically plausible Human-Scene
Interaction and Contact reconstruction. PhySIC recovers metrically consistent
SMPL-X human meshes, dense scene surfaces, and vertex-level contact maps within
a shared coordinate frame from a single RGB image. Starting from coarse
monocular depth and body estimates, PhySIC performs occlusion-aware inpainting,
fuses visible depth with unscaled geometry for a robust metric scaffold, and
synthesizes missing support surfaces like floors. A confidence-weighted
optimization refines body pose, camera parameters, and global scale by jointly
enforcing depth alignment, contact priors, interpenetration avoidance, and 2D
reprojection consistency. Explicit occlusion masking safeguards invisible
regions against implausible configurations. PhySIC is efficient, requiring only
9 seconds for joint human-scene optimization and under 27 seconds end-to-end.
It naturally handles multiple humans, enabling reconstruction of diverse
interactions. Empirically, PhySIC outperforms single-image baselines, reducing
mean per-vertex scene error from 641 mm to 227 mm, halving PA-MPJPE to 42 mm,
and improving contact F1 from 0.09 to 0.51. Qualitative results show realistic
foot-floor interactions, natural seating, and plausible reconstructions of
heavily occluded furniture. By converting a single image into a physically
plausible 3D human-scene pair, PhySIC advances scalable 3D scene understanding.
Our implementation is publicly available at https://yuxuan-xue.com/physic.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行分析。

---

**论文摘要分析：PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image**

**1. 论文主要贡献的简明总结 (2-3 句话)**

PhySIC 提出了一种从单张 RGB 图像重建物理上合理的三维人体-场景交互和接触的框架。它解决了现有方法在深度模糊、遮挡和物理不一致接触方面的挑战，通过联合优化人体网格、场景表面和顶点级接触图，实现了度量一致且物理可信的重建。该方法显著提高了重建精度，尤其是在场景几何、人体姿态和接触检测方面。

**2. 关键创新或方法论方法**

PhySIC 的关键创新在于其多阶段、联合优化的方法，特别强调了物理合理性：

*   **鲁棒的度量支架构建：** 从粗略的单目深度和人体估计开始，通过遮挡感知修复、可见深度与未缩放几何的融合，以及缺失支撑面（如地板）的合成，构建了一个稳健的度量支架。这有效地解决了单目深度估计的尺度模糊性问题。
*   **置信度加权的联合优化：** 引入了一个置信度加权的优化框架，同时精炼人体姿态、相机参数和全局尺度。这个优化器通过联合强制执行以下约束来确保物理合理性：
    *   **深度对齐：** 确保重建结果与原始深度估计一致。
    *   **接触先验：** 鼓励人体与场景之间发生合理的接触（例如，脚踩在地面上，人坐在椅子上）。
    *   **互穿避免：** 明确防止人体与场景之间发生不自然的穿透。
    *   **2D 重投影一致性：** 确保 3D 重建结果在图像平面上的投影与原始 2D 图像一致。
*   **显式遮挡掩码：** 使用显式遮挡掩码来保护不可见区域，防止其被优化到不合理的配置中，这对于处理复杂遮挡场景至关重要。
*   **顶点级接触图：** 不仅重建人体和场景，还输出精细的顶点级接触图，这对于理解交互细节非常有价值。

**3. 对该领域的潜在影响**

PhySIC 对计算机视觉领域具有显著的潜在影响：

*   **推动单目 3D 重建的边界：** 显著提高了从单张图像进行人体-场景联合 3D 重建的精度和物理合理性，尤其是在处理复杂交互和遮挡方面。
*   **更可靠的 3D 场景理解：** 提供了更准确、更可信的 3D 人体和场景表示，这对于需要理解场景中物体和代理之间关系的下游任务至关重要。
*   **为下游应用提供基础：** 其输出的度量一致的人体网格、场景表面和接触图，可以作为虚拟现实、机器人、动画和人机交互等领域更高级应用的基础。
*   **效率与可扩展性：** 相对高效的优化时间（9 秒用于联合优化，27 秒端到端）以及处理多人的能力，使其在实际应用中更具吸引力。

**4. 可能受益于这项研究的相关领域或应用**

*   **虚拟现实 (VR) 和增强现实 (AR)：** 创建更逼真、交互性更强的虚拟环境，例如，将真实世界的人体和场景无缝集成到虚拟世界中。
*   **机器人学：** 帮助机器人更好地理解其操作环境中的人类和物体，从而实现更安全、更自然的协作和导航。例如，机器人可以利用这些信息来预测人类的意图或避免碰撞。
*   **3D 内容创作和动画：** 自动化从 2D 图像生成 3D 场景和角色姿态的过程，极大地简化了动画师和 3D 艺术家的工作流程。
*   **人机交互 (HCI)：** 更好地理解用户在物理空间中的行为和意图，从而设计更直观、更响应式的人机界面。
*   **智能监控和安全：** 分析监控视频中的人类活动和交互，例如检测异常行为或理解人群动态。
*   **人体姿态估计和形状重建：** 为这些任务提供更强的场景上下文和物理约束，从而提高鲁棒性和准确性。

**5. 可以从摘要中推断出的任何局限性**

尽管 PhySIC 取得了显著进展，但摘要中仍可推断出一些潜在局限性：

*   **对初始估计的依赖：** 摘要提到“Starting from coarse monocular depth and body estimates”，这意味着该方法可能在一定程度上依赖于这些初始估计的质量。如果初始估计非常差，优化过程可能难以收敛到最优解。
*   **复杂场景的泛化能力：** 尽管它处理了遮挡和多人，但对于极端复杂、高度杂乱或包含大量非刚性物体的场景，其性能可能仍有待进一步验证。例如，对于非常规的物体形状或不常见的交互模式，接触先验的有效性可能受到限制。
*   **计算成本：** 尽管摘要称其“高效”，但 27 秒的端到端时间对于某些实时应用（如高帧率 VR/AR）可能仍然过长。
*   **纹理和材质重建：** 摘要主要关注几何形状和接触，并未提及对场景纹理、材质或光照的重建。这可能意味着输出的 3D 模型在视觉真实感方面仍需进一步处理。
*   **动态场景：** 摘要中没有明确说明其处理动态场景或视频序列的能力。从单张图像重建通常假定场景是静态的，或者至少在图像捕获瞬间是静态的。
*   **“物理合理性”的定义：** 尽管强调了物理合理性，但其具体实现（例如，接触先验和互穿避免的精确建模）可能仍是简化的。例如，它可能不考虑摩擦、弹性形变等更复杂的物理属性。

---

总而言之，PhySIC 是一项令人兴奋的研究，它通过引入一套新颖的联合优化策略和物理约束，显著提升了从单张图像进行 3D 人体-场景重建的质量和实用性。其对物理合理性的强调是其核心优势，有望在多个应用领域产生深远影响。

**Key Findings:**

- To address these challenges,
we introduce PhySIC, a framework for physically plausible Human-Scene
Interaction and Contact reconstruction.
- Empirically, PhySIC outperforms single-image baselines, reducing
mean per-vertex scene error from 641 mm to 227 mm, halving PA-MPJPE to 42 mm,
and improving contact F1 from 0.09 to 0.51.
- Qualitative results show realistic
foot-floor interactions, natural seating, and plausible reconstructions of
heavily occluded furniture.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11649v1)
- [arXiv](https://arxiv.org/abs/2510.11649v1)

---

<a id='2510.11606v1'></a>
## [ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](https://arxiv.org/abs/2510.11606v1)

**Authors:** Yicheng Xu, Yue Wu, Jiashuo Yu, Ziang Yan, Tianxiang Jiang, Yinan He, Qingsong Zhao, Kai Chen, Yu Qiao, Limin Wang, Manabu Okumura, Yi Wang

**Published:** 2025-10-13

**Categories:** cs.CV

**Abstract:**

Multimodal Large Language Models (MLLMs) hold promise for accelerating
scientific discovery by interpreting complex experimental procedures. However,
their true capabilities are poorly understood, as existing benchmarks neglect
the fine-grained and long-horizon nature of authentic laboratory work,
especially in wet-lab settings. To bridge this gap, we introduce ExpVid, the
first benchmark designed to systematically evaluate MLLMs on scientific
experiment videos. Curated from peer-reviewed video publications, ExpVid
features a new three-level task hierarchy that mirrors the scientific process:
(1) Fine-grained Perception of tools, materials, and actions; (2) Procedural
Understanding of step order and completeness; and (3) Scientific Reasoning that
connects the full experiment to its published conclusions. Our vision-centric
annotation pipeline, combining automated generation with multi-disciplinary
expert validation, ensures that tasks require visual grounding. We evaluate 19
leading MLLMs on ExpVid and find that while they excel at coarse-grained
recognition, they struggle with disambiguating fine details, tracking state
changes over time, and linking experimental procedures to scientific outcomes.
Our results reveal a notable performance gap between proprietary and
open-source models, particularly in high-order reasoning. ExpVid not only
provides a diagnostic tool but also charts a roadmap for developing MLLMs
capable of becoming trustworthy partners in scientific experimentation.

**Analysis:**

好的，这是一篇关于Yicheng Xu等人撰写的论文“ExpVid: A Benchmark for Experiment Video Understanding & Reasoning”的全面摘要：

**论文摘要：ExpVid：一个用于实验视频理解与推理的基准**

**1. 主要问题或研究问题：**
该论文旨在解决现有基准在评估多模态大型语言模型（MLLMs）理解科学实验视频方面的不足。现有基准未能充分捕捉真实实验室工作（特别是湿实验室环境）中细粒度、长时程的复杂性，导致对MLLMs在加速科学发现方面的真实能力理解不足。

**2. 关键创新或方法论贡献：**
*   **引入ExpVid基准：** 首次系统性地评估MLLMs在科学实验视频上的表现，特别关注湿实验室环境。
*   **三级任务层次结构：** ExpVid设计了一个模仿科学流程的三级任务层次结构：
    *   **一级：细粒度感知**（工具、材料和动作的识别）。
    *   **二级：程序理解**（步骤顺序和完整性的理解）。
    *   **三级：科学推理**（将整个实验与已发表的结论联系起来）。
*   **以视觉为中心的标注流程：** 结合自动化生成和多学科专家验证，确保任务需要视觉基础，避免仅依赖文本或背景知识。
*   **广泛的模型评估：** 在ExpVid上评估了19个主流MLLMs（包括开源和专有模型）。

**3. 主要结果及其意义：**
*   **MLLMs在粗粒度识别方面表现良好：** 模型在识别粗粒度信息方面表现出色。
*   **MLLMs在细粒度理解和高阶推理方面存在挑战：** 模型在以下方面表现不佳：
    *   区分细微细节。
    *   跟踪随时间变化的实验状态。
    *   将实验过程与科学结果联系起来。
*   **专有模型与开源模型之间的显著性能差距：** 尤其是在高阶推理任务中，专有模型的表现明显优于开源模型。
*   **诊断工具和路线图：** ExpVid不仅提供了一个诊断工具来识别MLLMs的弱点，也为开发能够成为科学实验中值得信赖的合作伙伴的MLLMs指明了方向。

**4. 论文中提及的局限性：**
*   **领域覆盖范围有限：** ExpVid目前主要关注湿实验室实验，尚未涵盖所有科学探究领域，例如物理学（涉及独特实验设备和抽象现象）或纯计算实验。
*   **推理任务的深度：** 三级推理任务评估的是结果，但未能深入阐明将实验与结论联系起来的底层推理过程（例如，思维链）。

**5. 潜在的未来研究方向：**
*   **扩展领域覆盖：** 将基准扩展到物理学、计算科学和大型工程测试等更多科学领域。
*   **深化推理评估：** 开发能够评估MLLMs在实验中更深层次推理过程（如因果链、假设检验）的机制。
*   **提升开源MLLMs的高阶推理能力：** 缩小开源模型与专有模型在复杂推理任务上的性能差距。
*   **优化模型对长时程视频数据的利用：** 解决模型在处理冗余输入时可能出现的饱和或干扰问题，以更好地利用扩展的时间上下文。

这篇论文通过引入ExpVid基准，为评估和推动MLLMs在理解真实科学实验视频方面的能力提供了重要贡献，为未来开发更智能、更可靠的科学AI助手奠定了基础。

**Key Findings:**

- To bridge this gap, we introduce ExpVid, the
first benchmark designed to systematically evaluate MLLMs on scientific
experiment videos.
- Curated from peer-reviewed video publications, ExpVid
features a new three-level task hierarchy that mirrors the scientific process:
(1) Fine-grained Perception of tools, materials, and actions; (2) Procedural
Understanding of step order and completeness; and (3) Scientific Reasoning that
connects the full experiment to its published conclusions.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.11606v1)
- [arXiv](https://arxiv.org/abs/2510.11606v1)

---

