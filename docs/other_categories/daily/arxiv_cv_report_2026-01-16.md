time: 20260116

# Arxiv Computer Vision Papers - 2026-01-16

## Executive Summary

好的，这是一份针对近期 Arxiv 计算机视觉论文的简明执行摘要，旨在帮助忙碌的研究人员快速了解该领域最重要的发展。

---

**执行摘要：近期 Arxiv 计算机视觉论文速览 (2026-01-14)**

**主要主题与趋势：**

本期 Arxiv 论文集聚焦于**大规模视觉语言模型 (VLMs)** 的发展与应用，以及在**动态环境下的视觉理解**和**多模态内容生成**方面的突破。自动驾驶、视频理解和内容编辑是贯穿其中的重要应用领域。

**亮点与创新：**

*   **大规模预训练模型：** "STEP3-VL-10B Technical Report" 和 "Molmo2" 预示着更大、更开放的视觉语言模型正在涌现，为下游任务提供了强大的基础。Molmo2 的开放权重和数据尤为值得关注。
*   **动态环境下的视觉合成与理解：** "WildRayZer" 在自监督学习方面取得了进展，能够处理动态环境下的视图合成，这对于机器人和增强现实至关重要。
*   **多模态内容生成与编辑：** "CoMoVi" 在3D人体运动与真实视频的联合生成方面展现了新的可能性，而 "Alterbute" 则专注于图像内在属性的编辑，为内容创作提供了更精细的控制。
*   **自动驾驶的泛化能力：** "See Less, Drive Better" 提出了一种利用基础模型和随机块选择来提升自动驾驶泛化能力的方法，是自动驾驶领域的重要进展。

**新兴研究方向与技术：**

*   **大规模基础模型的进一步演进：** 论文显示出向更大参数量、更广泛数据覆盖的 VLM 发展的趋势。
*   **动态场景下的自监督学习：** 在复杂、动态环境中进行有效学习是当前的研究热点。
*   **精细化内容编辑与生成：** 从属性编辑到3D内容生成，多模态生成技术正变得更加精细和多样化。
*   **对抗性攻击与防御：** "Adversarial Evasion Attacks on Computer Vision using SHAP Values" 揭示了利用可解释性技术进行对抗性攻击的新方法，也暗示了对模型鲁棒性研究的需求。
*   **长视频理解与推理：** "CURVE" 提出的基准测试表明，对长视频进行文化和多语言的推理是未来研究的重要方向。

**建议阅读论文：**

1.  **"STEP3-VL-10B Technical Report"**: 了解当前最先进的大规模 VLM 的技术细节和能力。
2.  **"Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding"**: 对于希望进行 VLM 研究或应用的研究人员来说，其开放的权重和数据具有极高的价值。
3.  **"See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection"**: 对自动驾驶领域的研究者而言，该论文提出的泛化能力提升方法值得深入研究。
4.  **"WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments"**: 对于在动态环境中进行视觉感知和合成的研究者，该论文提供了创新的自监督学习思路。

---

---

## Table of Contents

1. [STEP3-VL-10B Technical Report](#2601.09668v2)
2. [WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments](#2601.10716v1)
3. [Alterbute: Editing Intrinsic Attributes of Objects in Images](#2601.10714v1)
4. [From One-to-One to Many-to-Many: Dynamic Cross-Layer Injection for Deep Vision-Language Fusion](#2601.10710v1)
5. [See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection](#2601.10707v1)
6. [CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning](#2601.10649v1)
7. [CoMoVi: Co-Generation of 3D Human Motions and Realistic Videos](#2601.10632v1)
8. [Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding](#2601.10611v1)
9. [Action100M: A Large-scale Video Action Dataset](#2601.10592v1)
10. [Adversarial Evasion Attacks on Computer Vision using SHAP Values](#2601.10587v1)

---

## Papers

<a id='2601.09668v2'></a>
## [STEP3-VL-10B Technical Report](https://arxiv.org/abs/2601.09668v2)

**Authors:** Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, Jingcheng Hu, Kangheng Lin, Liang Zhao, Mitt Huang, Song Yuan, Wenwen Qu, Xiangfeng Wang, Yanlin Lai, Yingxiu Zhao, Yinmin Zhang, Yukang Shi, Yuyang Chen, Zejia Weng, Ziyang Meng, Ang Li, Aobo Kong, Bo Dong, Changyi Wan, David Wang, Di Qi, Dingming Li, En Yu, Guopeng Li, Haiquan Yin, Han Zhou, Hanshan Zhang, Haolong Yan, Hebin Zhou, Hongbo Peng, Jiaran Zhang, Jiashu Lv, Jiayi Fu, Jie Cheng, Jie Zhou, Jisheng Yin, Jingjing Xie, Jingwei Wu, Jun Zhang, Junfeng Liu, Kaijun Tan, Kaiwen Yan, Liangyu Chen, Lina Chen, Mingliang Li, Qian Zhao, Quan Sun, Shaoliang Pang, Shengjie Fan, Shijie Shang, Siyuan Zhang, Tianhao You, Wei Ji, Wuxun Xie, Xiaobo Yang, Xiaojie Hou, Xiaoran Jiao, Xiaoxiao Ren, Xiangwen Kong, Xin Huang, Xin Wu, Xing Chen, Xinran Wang, Xuelin Zhang, Yana Wei, Yang Li, Yanming Xu, Yeqing Shen, Yuang Peng, Yue Peng, Yu Zhou, Yusheng Li, Yuxiang Yang, Yuyang Zhang, Zhe Xie, Zhewei Huang, Zhenyi Lu, Zhimin Fan, Zihui Cheng, Daxin Jiang, Qi Han, Xiangyu Zhang, Yibo Zhu, Zheng Ge

**Published:** 2026-01-14

**Categories:** cs.CV

**Abstract:**

We present STEP3-VL-10B, a lightweight open-source foundation model designed to redefine the trade-off between compact efficiency and frontier-level multimodal intelligence. STEP3-VL-10B is realized through two strategic shifts: first, a unified, fully unfrozen pre-training strategy on 1.2T multimodal tokens that integrates a language-aligned Perception Encoder with a Qwen3-8B decoder to establish intrinsic vision-language synergy; and second, a scaled post-training pipeline featuring over 1k iterations of reinforcement learning. Crucially, we implement Parallel Coordinated Reasoning (PaCoRe) to scale test-time compute, allocating resources to scalable perceptual reasoning that explores and synthesizes diverse visual hypotheses. Consequently, despite its compact 10B footprint, STEP3-VL-10B rivals or surpasses models 10$\times$-20$\times$ larger (e.g., GLM-4.6V-106B, Qwen3-VL-235B) and top-tier proprietary flagships like Gemini 2.5 Pro and Seed-1.5-VL. Delivering best-in-class performance, it records 92.2% on MMBench and 80.11% on MMMU, while excelling in complex reasoning with 94.43% on AIME2025 and 75.95% on MathVision. We release the full model suite to provide the community with a powerful, efficient, and reproducible baseline.

**Analysis:**

好的，这是对论文“STEP3-VL-10B Technical Report”的全面中文摘要，重点关注其在计算机视觉和机器学习领域的贡献：

**论文名称：** STEP3-VL-10B Technical Report

**作者列表：** Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, Jingcheng Hu, Kangheng Lin, Liang Zhao, Mitt Huang, Song Yuan, Wenwen Qu, Xiangfeng Wang, Yanlin Lai, Yingxiu Zhao, Yinmin Zhang, Yukang Shi, Yuyang Chen, Zejia Weng, Ziyang Meng, Ang Li, Aobo Kong, Bo Dong, Changyi Wan, David Wang, Di Qi, Dingming Li, En Yu, Guopeng Li, Haiquan Yin, Han Zhou, Hanshan Zhang, Haolong Yan, Hebin Zhou, Hongbo Peng, Jiaran Zhang, Jiashu Lv, Jiayi Fu, Jie Cheng, Jie Zhou, Jisheng Yin, Jingjing Xie, Jingwei Wu, Jun Zhang, Junfeng Liu, Kaijun Tan, Kaiwen Yan, Liangyu Chen, Lina Chen, Mingliang Li, Qian Zhao, Quan Sun, Shaoliang Pang, Shengjie Fan, Shijie Shang, Siyuan Zhang, Tianhao You, Wei Ji, Wuxun Xie, Xiaobo Yang, Xiaojie Hou, Xiaoran Jiao, Xiaoxiao Ren, Xiangwen Kong, Xin Huang, Xin Wu, Xing Chen, Xinran Wang, Xuelin Zhang, Yana Wei, Yang Li, Yanming Xu, Yeqing Shen, Yuang Peng, Yue Peng, Yu Zhou, Yusheng Li, Yuxiang Yang, Yuyang Zhang, Zhe Xie, Zhewei Huang, Zhenyi Lu, Zhimin Fan, Zihui Cheng, Daxin Jiang, Qi Han, Xiangyu Zhang, Yibo Zhu, Zheng Ge

**摘要：**

**1. 主要问题与研究目标：**
该论文旨在解决当前多模态大语言模型（MLLMs）在效率与性能之间权衡的难题。一方面，大型专有模型（如Gemini-3-Pro）通过海量扩展实现了前沿的多模态智能，但其高昂的计算成本限制了实际部署；另一方面，传统轻量级模型（<10B参数）虽然高效，但在复杂的视觉感知和推理能力上表现受限。因此，研究的核心问题是如何构建一个紧凑（10B参数）但性能卓越的开源多模态基础模型，以重新定义效率与智能的边界。

**2. 关键创新与方法论贡献：**
STEP3-VL-10B模型的核心创新体现在两个方面：

*   **统一、全量微调的预训练策略：** 采用1.2T多模态token的单阶段、全量微调（fully unfrozen）预训练策略。该策略整合了一个语言对齐的感知编码器（Perception Encoder）和一个Qwen3-8B解码器，旨在建立内在的视觉-语言协同（vision-language synergy）。这种设计强调了在预训练阶段就实现跨模态的深度融合。
*   **规模化的后训练与并行推理：** 引入了经过大规模（超过1k次迭代）强化学习（RL）的后训练流程。关键的创新是实现了**并行协调推理（Parallel Coordinated Reasoning, PaCoRe）**。PaCoRe是一种测试时（test-time）计算扩展技术，它允许模型并行探索多种视觉假设，并通过合成（synthesize）这些假设来得出最终结论。这使得模型能够以更少的参数量模拟更复杂的推理过程，有效弥合了感知与推理之间的性能鸿沟。

**3. 主要结果与意义：**
STEP3-VL-10B在多个基准测试中取得了显著的成果：

*   **性能超越：** 尽管参数量仅为10B，STEP3-VL-10B在多项任务上能够媲美甚至超越参数量大10-20倍的开源模型（如GLM-4.6V-106B, Qwen3-VL-235B）以及顶级的专有模型（如Gemini 2.5 Pro, Seed-1.5-VL）。
*   **关键指标：** 在MM-Bench上达到92.2%的准确率，在MMMU上达到80.11%。在复杂的推理任务上表现尤为突出，AIME2025上达到94.43%，MathVision上达到75.95%。
*   **意义：** 这些结果表明，STEP3-VL-10B成功地打破了模型规模与性能之间的传统关联，证明了通过精巧的设计和优化的训练策略，可以在保持模型紧凑性的同时实现前沿的多模态智能。它为社区提供了一个强大、高效且可复现的开源基线模型，极大地推动了轻量级多模态模型的发展。

**4. 局限性：**
论文中并未明确列出STEP3-VL-10B的局限性。然而，从其对PaCoRe的强调以及对“计算密度”的讨论中可以推断，虽然模型在效率上表现出色，但其在某些极端复杂或需要极高精度的感知任务上，可能仍有进一步提升的空间。此外，论文也提到了“现实差距”（reality gap），暗示模型在与物理世界的交互和理解方面仍有待发展。

**5. 未来研究方向：**
论文展望了未来的研究方向，主要集中在以下几个方面：

*   **最大化Token效率与通用RL扩展：** 进一步优化计算资源的使用，使每一个计算单元都能最大化地贡献于智能密度。计划将更多计算资源投入到强化学习（RL）阶段，通过深度（顺序推理）和宽度（并行探索）的扩展，挖掘更高价值的感知和推理模式。
*   **优化推理密度：** 目标是整合并行探索的优势，消除冗余的“过度思考”，将复杂的推理路径压缩为高效的顺序性，最终实现“系统1”式的即时响应。
*   **弥合“现实差距”：** 提出了一种范式转变，从被动的数据消费转向主动的物理世界交互。这包括：
    *   **从语义到物理世界模型：** 将多智能体合成扩展到视频轨迹和传感器-动作序列，构建一个能够内化物理因果关系和时空动态的整体世界模型。
    *   **物理作为终极验证器：** 利用高保真模拟环境，让模型在遵循物理定律的约束下进行学习，实现基于可验证因果关系的推理，而非统计相关性。
    *   **具身思维链（Embodied Chain-of-Thought, E-CoT）：** 扩展推理上下文以显式建模时间动态和物理状态转换，训练模型预测动力学以实现长时规划。

总而言之，STEP3-VL-10B是一项重要的研究成果，它通过创新的预训练和后训练策略，特别是PaCoRe技术，成功地在模型效率和多模态智能之间取得了突破性的平衡，为轻量级、高性能的开源多模态模型树立了新的标杆。

**Key Findings:**

- We present STEP3-VL-10B, a lightweight open-source foundation model designed to redefine the trade-off between compact efficiency and frontier-level multimodal intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.09668v2)
- [arXiv](https://arxiv.org/abs/2601.09668v2)

---

<a id='2601.10716v1'></a>
## [WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments](https://arxiv.org/abs/2601.10716v1)

**Authors:** Xuweiyi Chen, Wentao Zhou, Zezhou Cheng

**Published:** 2026-01-15

**Categories:** cs.CV

**Abstract:**

We present WildRayZer, a self-supervised framework for novel view synthesis (NVS) in dynamic environments where both the camera and objects move. Dynamic content breaks the multi-view consistency that static NVS models rely on, leading to ghosting, hallucinated geometry, and unstable pose estimation. WildRayZer addresses this by performing an analysis-by-synthesis test: a camera-only static renderer explains rigid structure, and its residuals reveal transient regions. From these residuals, we construct pseudo motion masks, distill a motion estimator, and use it to mask input tokens and gate loss gradients so supervision focuses on cross-view background completion. To enable large-scale training and evaluation, we curate Dynamic RealEstate10K (D-RE10K), a real-world dataset of 15K casually captured dynamic sequences, and D-RE10K-iPhone, a paired transient and clean benchmark for sparse-view transient-aware NVS. Experiments show that WildRayZer consistently outperforms optimization-based and feed-forward baselines in both transient-region removal and full-frame NVS quality with a single feed-forward pass.

**Analysis:**

好的，这是对论文“WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments”的全面中文摘要，其中包含了您要求的各个方面：

**论文题目：** WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments

**作者：** Xuweiyi Chen, Wentao Zhou, Zezhou Cheng

**摘要：**

**1. 研究问题/核心挑战：**
该论文主要解决了在动态环境中进行新视角合成（Novel View Synthesis, NVS）的挑战。传统的NVS方法通常假设场景是静态的，并且依赖于精确的相机位姿。然而，在现实世界中，相机和场景中的物体都可能移动，这会破坏多视图一致性，导致生成图像出现重影、几何错误和不稳定的位姿估计。因此，研究的核心问题是如何在**没有显式3D监督（如相机位姿、深度图或动态掩码）**的情况下，从**稀疏、未校准的动态输入图像**中合成**无瞬态物体**的新视角。

**2. 关键创新与方法贡献：**
WildRayZer 提出了一种**自监督学习框架**，通过**分析-合成（analysis-by-synthesis）**的策略来解决上述挑战。其核心创新包括：

*   **动态场景的分析-合成方法：** 利用一个**仅包含静态渲染器**的模块来解释场景的**刚性结构**。渲染结果与真实图像之间的**残差**被用来**识别瞬态区域**。
*   **伪运动掩码（Pseudo Motion Masks）的构建：** 从渲染残差中，通过融合 DINOv3 特征和 SSIM 相似度，构建出**伪运动掩码**。这些掩码无需真实标注，是自监督学习的关键。
*   **运动估计器（Motion Estimator）的蒸馏：** 利用构建的伪运动掩码，蒸馏出一个**运动估计器**。该估计器能够预测像素级的运动概率。
*   **动态输入令牌（Tokens）的掩蔽：** 将运动估计器预测的动态区域用于**掩蔽输入图像的令牌**，从而**过滤掉瞬态内容**，确保场景编码器只关注**静态结构**。
*   **动态损失梯度门控：** 利用运动掩码来**门控损失梯度**，使得监督信号**聚焦于跨视图的背景补全**。
*   **动态数据集的构建：** 为了支持大规模训练和评估，作者**创建了两个新的数据集**：
    *   **Dynamic RealEstate10K (D-RE10K)：** 一个包含 15K 个真实世界、随意拍摄的动态室内序列的大规模数据集。
    *   **D-RE10K-iPhone：** 一个配对的、包含瞬态和干净视图的基准数据集，专门用于稀疏视图下的瞬态感知 NVS 评估。
*   **渐进式多阶段训练策略：** 采用一种**分阶段的训练流程**，首先训练运动估计器，然后训练掩码渲染器，最后进行联合微调，以确保稳定性和收敛性。
*   **Copy-Paste 数据增强：** 引入了**Copy-Paste 数据增强**技术，将 COCO 数据集中的物体粘贴到训练图像中，为运动估计器提供额外的监督信号，并提高其在**开放集（out-of-domain）**场景下的泛化能力。

**3. 主要结果与意义：**
实验结果表明，WildRayZer 在**稀疏视图下的动态场景新视角合成**和**运动分割**任务上均取得了**显著优于**现有方法的性能。

*   **新视角合成质量：** 在 D-RE10K 和 D-RE10K-iPhone 数据集上，WildRayZer 在**静态区域的保真度**和**整体图像质量**上都表现出色，能够有效地**去除瞬态物体**并**补全被遮挡的背景**。
*   **运动分割准确性：** 其自监督学习的运动掩码生成器在**运动边界的准确性**和**召回率**方面优于许多监督和自监督的基线方法。
*   **效率与鲁棒性：** WildRayZer 是一个**前馈（feed-forward）**模型，在**测试时无需相机位姿**，并且对**动态内容具有很强的鲁棒性**。
*   **数据集贡献：** 新数据集 D-RE10K 和 D-RE10K-iPhone 为动态场景 NVS 和运动分割研究提供了重要的资源。

**4. 提及的局限性：**
论文中提到了一些潜在的局限性：

*   **伪运动掩码的不足：** 伪运动掩码可能**无法实现实例级别的分割**，有时只能捕捉物体移动的部分，而忽略了静态部分，这可能影响渲染质量。
*   **欠分割（Under-segmentation）：** 在某些情况下，模型可能**欠分割**，即预测的运动掩码小于真实的移动区域。
*   **大比例瞬态物体：** 当瞬态物体**占据输入图像的很大一部分**时，运动掩码的质量可能会下降。
*   **对渲染质量的依赖：** 伪运动掩码的构建依赖于**渲染器的质量**，在训练早期，当渲染器质量不高时，可能导致噪声伪标签。

**5. 潜在的未来研究方向：**
虽然论文没有明确列出未来研究方向，但从其方法和局限性中可以推断出以下潜在方向：

*   **实例级动态物体分割：** 进一步提升运动掩码的**实例分割能力**，以更精确地分离和处理动态物体。
*   **更鲁棒的伪标签生成：** 研究更**鲁棒的伪标签生成策略**，减少对渲染质量的依赖，尤其是在训练早期。
*   **处理更复杂的动态场景：** 探索处理**更复杂动态场景**的方法，例如具有**快速运动、遮挡和变形**的物体。
*   **端到端实例级动态物体移除与重渲染：** 将动态物体的移除和背景的重渲染**更紧密地结合**，实现更自然的动态场景处理。
*   **更精细的动态物体表示：** 探索如何**表示和合成动态物体本身**，而不仅仅是移除它们。

总而言之，WildRayZer 在自监督动态场景新视角合成领域取得了重要进展，通过创新的分析-合成方法和伪标签生成策略，有效解决了动态内容带来的挑战，并为该领域的研究提供了新的数据集和方法论。

**Key Findings:**

- We present WildRayZer, a self-supervised framework for novel view synthesis (NVS) in dynamic environments where both the camera and objects move.
- Experiments show that WildRayZer consistently outperforms optimization-based and feed-forward baselines in both transient-region removal and full-frame NVS quality with a single feed-forward pass.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10716v1)
- [arXiv](https://arxiv.org/abs/2601.10716v1)

---

<a id='2601.10714v1'></a>
## [Alterbute: Editing Intrinsic Attributes of Objects in Images](https://arxiv.org/abs/2601.10714v1)

**Authors:** Tal Reiss, Daniel Winter, Matan Cohen, Alex Rav-Acha, Yael Pritch, Ariel Shamir, Yedid Hoshen

**Published:** 2026-01-15

**Categories:** cs.CV, cs.GR

**Abstract:**

We introduce Alterbute, a diffusion-based method for editing an object's intrinsic attributes in an image. We allow changing color, texture, material, and even the shape of an object, while preserving its perceived identity and scene context. Existing approaches either rely on unsupervised priors that often fail to preserve identity or use overly restrictive supervision that prevents meaningful intrinsic variations. Our method relies on: (i) a relaxed training objective that allows the model to change both intrinsic and extrinsic attributes conditioned on an identity reference image, a textual prompt describing the target intrinsic attributes, and a background image and object mask defining the extrinsic context. At inference, we restrict extrinsic changes by reusing the original background and object mask, thereby ensuring that only the desired intrinsic attributes are altered; (ii) Visual Named Entities (VNEs) - fine-grained visual identity categories (e.g., ''Porsche 911 Carrera'') that group objects sharing identity-defining features while allowing variation in intrinsic attributes. We use a vision-language model to automatically extract VNE labels and intrinsic attribute descriptions from a large public image dataset, enabling scalable, identity-preserving supervision. Alterbute outperforms existing methods on identity-preserving object intrinsic attribute editing.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Alterbute: Editing Intrinsic Attributes of Objects in Images**

**1. 主要贡献（2-3句话的简洁总结）**

本论文提出了一种名为 Alterbute 的新颖扩散模型方法，能够精确地编辑图像中物体的内在属性（如颜色、纹理、材质甚至形状），同时有效保持物体的身份识别和场景上下文。与现有方法不同，Alterbute 通过一种宽松的训练目标和创新的视觉命名实体（VNEs）概念，实现了更具可控性和身份保持性的内在属性编辑。

**2. 关键创新或方法论**

Alterbute 的核心创新在于其独特的方法论，主要体现在以下两点：

*   **宽松的训练目标与推理约束相结合：**
    *   **训练阶段：** 模型被训练来改变内在和外在属性，但其输入包含一个“身份参考图像”（identity reference image）、描述目标内在属性的“文本提示”（textual prompt），以及定义外在上下文的“背景图像和对象掩码”（background image and object mask）。这种设置允许模型学习如何解耦内在和外在属性的变化。
    *   **推理阶段：** 在实际应用中，通过“重用原始背景和对象掩码”来强制约束外在变化，从而确保只有用户指定的内在属性得到修改。这种“训练时灵活，推理时严格”的策略是实现身份保持的关键。
*   **视觉命名实体（Visual Named Entities - VNEs）：**
    *   VNEs 被定义为“细粒度的视觉身份类别”，例如“保时捷 911 Carrera”。它们能够将具有相似身份定义特征但内在属性可能不同的物体进行分组。
    *   通过使用“视觉-语言模型（vision-language model）”自动从大规模公共图像数据集中提取 VNE 标签和内在属性描述，实现了可扩展的、身份保持的监督学习。这解决了传统方法中监督信号不足或过于受限的问题。

**3. 对该领域的潜在影响**

Alterbute 的提出对计算机视觉领域的图像编辑和内容生成领域具有重要意义：

*   **提升图像编辑的精细度和可控性：** 能够精确控制物体的内在属性，而无需担心身份漂移或场景不一致，这将极大地提升图像编辑工具的实用性和用户体验。
*   **推动更逼真的内容生成：** 在虚拟现实、游戏开发、电影制作等领域，能够生成具有特定材质、颜色或形状变化的逼真物体，为内容创作提供强大的支持。
*   **促进对物体属性理解的研究：** VNEs 的概念和提取方法，以及模型对内在/外在属性的解耦能力，将有助于更深入地理解物体在视觉上的身份定义和属性变化规律。
*   **为下游任务提供更优质的训练数据：** 通过 Alterbute 生成的具有特定属性变化的图像，可以用于训练其他对物体属性敏感的模型，例如材质识别、风格迁移等。

**4. 可能受益的相关领域或应用**

*   **内容创作与设计：**
    *   **产品设计与可视化：** 快速修改产品模型（如汽车、家具）的颜色、材质、纹理，用于概念设计和营销展示。
    *   **时尚设计：** 尝试不同颜色、图案和材质的服装，用于虚拟试穿和设计预览。
    *   **室内设计：** 改变房间内家具、墙壁的材质和颜色，进行虚拟装修和效果预览。
*   **虚拟现实（VR）与增强现实（AR）：**
    *   在虚拟环境中动态改变物体的外观，增强用户交互的真实感和沉浸感。
    *   AR 应用中，叠加具有特定属性的虚拟物体到真实场景中。
*   **游戏开发：**
    *   为游戏角色或场景中的物体生成不同外观变体，丰富游戏内容。
    *   动态改变游戏内物品的材质和颜色，增加游戏的可玩性和个性化。
*   **数字艺术与摄影后期：**
    *   艺术家和摄影师可以更自由地对图像中的物体进行创意性编辑，实现独特的视觉效果。
*   **机器人与自动驾驶：**
    *   虽然不是直接应用，但对物体属性的精确理解和编辑能力，可能有助于训练更鲁棒的感知模型，例如在不同光照或材质条件下识别物体。

**5. 从摘要中可以推断出的局限性**

尽管 Alterbute 听起来非常强大，但从摘要中可以推断出一些潜在的局限性：

*   **对“身份”的定义和保持的挑战：** 尽管论文声称能保持身份，但“身份”的定义本身就具有主观性和模糊性。对于非常相似但身份不同的物体（例如同一品牌不同型号的汽车），模型能否始终如一地保持区分，仍需进一步验证。VNEs 的粒度也可能影响其泛化能力。
*   **对“内在属性”的定义和控制的边界：** 摘要提到了颜色、纹理、材质和形状。对于形状的编辑，其“内在性”的定义可能比颜色和纹理更复杂。例如，改变一个物体的形状是否会影响其功能性或结构完整性，这可能超出了“内在属性”的范畴，或者需要更精细的控制。
*   **对“场景上下文”的依赖和限制：** 模型依赖于背景图像和对象掩码来定义外在上下文。这意味着，如果原始图像的背景或对象掩码本身存在问题，或者用户想要将物体置于完全不同的场景中，Alterbute 的表现可能会受到影响。虽然推理时约束了外在变化，但训练时对这些信息的依赖可能意味着模型对复杂或不寻常的场景上下文的泛化能力有限。
*   **计算成本和效率：** 扩散模型通常计算成本较高。虽然摘要没有直接提及，但实际应用中的推理速度和所需的计算资源可能是需要考虑的因素。
*   **对文本提示的依赖：** 文本提示的质量直接影响编辑结果。模糊或不准确的文本描述可能会导致模型生成不符合预期的结果。
*   **VNEs 提取的准确性和覆盖范围：** 自动提取 VNEs 和属性描述的准确性，以及其覆盖的类别范围，将直接影响模型的训练数据质量和最终性能。如果某些重要的物体类别或属性没有被充分覆盖，模型在该方面的表现可能会受限。

总而言之，Alterbute 是一项令人兴奋的研究，它通过创新的方法论解决了图像编辑领域一个长期存在的挑战。其对身份保持和属性解耦的关注，以及 VNEs 的引入，为未来的图像编辑和内容生成技术开辟了新的可能性。然而，任何先进技术都伴随着其固有的挑战和局限性，这些都需要在实际应用和进一步的研究中加以探索和解决。

**Key Findings:**

- We introduce Alterbute, a diffusion-based method for editing an object's intrinsic attributes in an image.
- Our method relies on: (i) a relaxed training objective that allows the model to change both intrinsic and extrinsic attributes conditioned on an identity reference image, a textual prompt describing the target intrinsic attributes, and a background image and object mask defining the extrinsic context.
- Alterbute outperforms existing methods on identity-preserving object intrinsic attribute editing.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10714v1)
- [arXiv](https://arxiv.org/abs/2601.10714v1)

---

<a id='2601.10710v1'></a>
## [From One-to-One to Many-to-Many: Dynamic Cross-Layer Injection for Deep Vision-Language Fusion](https://arxiv.org/abs/2601.10710v1)

**Authors:** Cheng Chen, Yuyu Guo, Pengpeng Zeng, Jingkuan Song, Peng Di, Hang Yu, Lianli Gao

**Published:** 2026-01-15

**Categories:** cs.CV

**Abstract:**

Vision-Language Models (VLMs) create a severe visual feature bottleneck by using a crude, asymmetric connection that links only the output of the vision encoder to the input of the large language model (LLM). This static architecture fundamentally limits the ability of LLMs to achieve comprehensive alignment with hierarchical visual knowledge, compromising their capacity to accurately integrate local details with global semantics into coherent reasoning. To resolve this, we introduce Cross-Layer Injection (CLI), a novel and lightweight framework that forges a dynamic many-to-many bridge between the two modalities. CLI consists of two synergistic, parameter-efficient components: an Adaptive Multi-Projection (AMP) module that harmonizes features from diverse vision layers, and an Adaptive Gating Fusion (AGF) mechanism that empowers the LLM to selectively inject the most relevant visual information based on its real-time decoding context. We validate the effectiveness and versatility of CLI by integrating it into LLaVA-OneVision and LLaVA-1.5. Extensive experiments on 18 diverse benchmarks demonstrate significant performance improvements, establishing CLI as a scalable paradigm that unlocks deeper multimodal understanding by granting LLMs on-demand access to the full visual hierarchy.

**Analysis:**

好的，这是一篇关于改进视觉语言模型（VLM）中视觉信息融合方式的论文。以下是该论文的全面中文摘要，重点关注其核心贡献、方法、结果和意义：

**论文题目：** From One-to-One to Many-to-Many: Dynamic Cross-Layer Injection for Deep Vision-Language Fusion (一对一到多对多：用于深度视觉语言融合的动态跨层注入)

**作者：** Cheng Chen, Yuyu Guo, Pengpeng Zeng, Jingkuan Song, Peng Di, Hang Yu, Lianli Gao

---

**全面摘要**

**1. 主要问题/研究挑战：**

当前主流的视觉语言模型（VLM）在连接视觉编码器（如 Vision Transformer, ViT）和大型语言模型（LLM）时，存在一个**严重的视觉特征瓶颈**。这种连接方式是**粗糙且不对称**的，仅将视觉编码器的**最终层输出**连接到 LLM 的输入。这种静态的、单一的连接方式**严重限制了 LLM 整合分层视觉知识的能力**，使其难以准确地将局部视觉细节与全局语义信息融合，从而影响了其进行全面、细致推理的能力。论文通过图 1(a) 中的一个失败案例（将溜冰鞋误识别为旱冰鞋）来具体说明这一问题，强调了 LLM 需要访问早期视觉层提取的精细特征。

**2. 关键创新/方法贡献：**

为了解决上述问题，论文提出了一种新颖、轻量级的框架——**跨层注入（Cross-Layer Injection, CLI）**。CLI 的核心在于构建一个**动态的“多对多”桥梁**，连接视觉编码器的**多个层级**和 LLM 的**多个解码器层**。CLI 由两个协同工作的、参数高效的组件组成：

*   **自适应多投影（Adaptive Multi-Projection, AMP）模块：** 该模块利用 Low-Rank Adaptation (LoRA) 技术，能够**高效地对来自不同视觉层级的特征进行对齐和统一**，将它们映射到一个共享的语义空间。这解决了不同视觉层级特征分布差异大导致的对齐难题，同时避免了为每个层级训练独立投影器的计算开销。
*   **自适应门控融合（Adaptive Gating Fusion, AGF）机制：** 这是 CLI 的关键创新。AGF 作为一个**智能的、上下文感知的控制器**，允许 LLM 在其解码过程中的**每个注入点**，根据当前的解码上下文，**动态地查询并选择性地注入最相关的视觉信息**。它通过交叉注意力机制提取视觉特征和 LLM 隐藏状态的关键信息，然后使用一个门控控制器计算一个动态权重，来控制新视觉信息对 LLM 隐藏状态的更新程度。这种机制实现了“按需访问”视觉信息，确保了注入的信息是有效且不干扰 LLM 现有状态的。

CLI 的整体架构（图 2(b)）实现了视觉编码器和 LLM 解码器之间**动态的、多对多的信息流**，克服了传统 VLM 中信息瓶颈和静态连接的限制。

**3. 主要结果与意义：**

论文将 CLI 集成到两个主流的 VLM 架构——**LLaVA-OneVision 和 LLaVA-1.5** 中，并在**18 个多样化的基准测试**上进行了广泛的实验评估。结果表明：

*   **显著的性能提升：** CLI 在绝大多数基准测试上都取得了**显著的性能提升**，尤其是在需要细粒度视觉理解和复杂推理的任务上。例如，在 LLaVA-in-the-Wild 数据集上，CLI 带来了 +6.5% 的提升。
*   **架构无关性：** CLI 在 LLaVA-1.5 上的成功集成（Tab. 3, Tab. 8）证明了其**架构无关性**，能够与不同的 LLM 和视觉编码器协同工作，并带来一致的性能增益。
*   **克服现有方法的局限性：** 与 DeepStack（粗暴的“一对多”注入）和 SLI（僵化的“一对一”静态连接）等方法相比，CLI 展现出**更优越的性能和鲁棒性**。DeepStack 常常导致性能下降，而 SLI 则会严重损害复杂推理能力。
*   **实现更深层次的多模态理解：** CLI 使 LLM 能够**按需访问完整的视觉层级信息**，从而实现更深层次的多模态理解，能够进行更精细的视觉感知、更复杂的推理，并提高对模糊场景的鲁棒性，减少幻觉。
*   **状态的最先进（SOTA）表现：** CLI 显著提升了模型的性能，使其在一些关键任务上达到了**新的 SOTA 水平**。

**意义：** CLI 提供了一种**轻量级且可扩展的范式**，能够有效地解决当前 VLM 中视觉信息融合的瓶颈问题，为实现更强大的多模态理解能力开辟了新的途径。

**4. 提及的局限性：**

论文中**未明确提及**其方法存在的显著局限性。然而，从实验设置和结果分析中可以推断出一些潜在的考虑：

*   **计算成本：** 虽然 CLI 被设计为参数高效的，但“多对多”的动态注入策略相比于简单的“一对一”连接，可能会带来一定的**计算开销增加**，尤其是在高密度注入的情况下。
*   **训练数据依赖：** 论文在消融研究中展示了训练数据量对 CLI 性能的影响（图 6），表明 CLI 的优势在**低数据量情况下可能不那么明显**，需要一定量的指令调优数据才能充分发挥其潜力。
*   **超参数敏感性：** 尽管论文采用了 LoRA 等技术来提高参数效率，但 AMP 和 AGF 模块的**具体超参数设置**（如 LoRA 的 rank 和 alpha，以及 AGF 的查询向量等）可能对最终性能有一定影响，需要仔细调整。

**5. 潜在的未来研究方向：**

基于论文的研究，可以推断出以下潜在的未来研究方向：

*   **更精细的注入策略：** 探索更智能的注入密度和注入时机，以在性能和计算效率之间找到更好的平衡点。例如，研究如何根据任务类型动态调整注入密度。
*   **跨模态交互的深度探索：** CLI 已经实现了视觉到语言的动态注入，未来可以进一步探索**语言到视觉的动态交互**，实现更深层次的双向信息融合。
*   **更广泛的应用场景：** 将 CLI 应用于更复杂的视觉语言任务，如视频理解、多模态对话生成、视觉问答等，以验证其通用性和扩展性。
*   **模型可解释性增强：** 虽然论文通过可视化门控权重来解释 CLI 的工作机制，但可以进一步研究如何更深入地理解 CLI 如何影响 LLM 的内部表征和推理过程。
*   **与其他先进 VLM 架构的结合：** 将 CLI 与最新的 VLM 架构（如具有更复杂视觉编码器或 LLM 的模型）结合，探索其协同效应。

总而言之，这篇论文提出了一种创新的 CLI 框架，通过动态的“多对多”跨层注入机制，有效解决了当前 VLM 中视觉信息融合的瓶颈问题，显著提升了模型的视觉理解和推理能力，为构建更强大的多模态人工智能系统奠定了坚实的基础。

**Key Findings:**

- To resolve this, we introduce Cross-Layer Injection (CLI), a novel and lightweight framework that forges a dynamic many-to-many bridge between the two modalities.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10710v1)
- [arXiv](https://arxiv.org/abs/2601.10710v1)

---

<a id='2601.10707v1'></a>
## [See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection](https://arxiv.org/abs/2601.10707v1)

**Authors:** Amir Mallak, Erfan Aasi, Shiva Sreeram, Tsun-Hsuan Wang, Daniela Rus, Alaa Maalouf

**Published:** 2026-01-15

**Categories:** cs.CV, cs.LG, cs.RO

**Abstract:**

Recent advances in end-to-end autonomous driving show that policies trained on patch-aligned features extracted from foundation models generalize better to Out-of-Distribution (OOD). We hypothesize that due to the self-attention mechanism, each patch feature implicitly embeds/contains information from all other patches, represented in a different way and intensity, making these descriptors highly redundant. We quantify redundancy in such (BLIP2) features via PCA and cross-patch similarity: $90$% of variance is captured by $17/64$ principal components, and strong inter-token correlations are pervasive. Training on such overlapping information leads the policy to overfit spurious correlations, hurting OOD robustness. We present Stochastic-Patch-Selection (SPS), a simple yet effective approach for learning policies that are more robust, generalizable, and efficient. For every frame, SPS randomly masks a fraction of patch descriptors, not feeding them to the policy model, while preserving the spatial layout of the remaining patches. Thus, the policy is provided with different stochastic but complete views of the (same) scene: every random subset of patches acts like a different, yet still sensible, coherent projection of the world. The policy thus bases its decisions on features that are invariant to which specific tokens survive. Extensive experiments confirm that across all OOD scenarios, our method outperforms the state of the art (SOTA), achieving a $6.2$% average improvement and up to $20.4$% in closed-loop simulations, while being $2.4\times$ faster. We conduct ablations over masking rates and patch-feature reorganization, training and evaluating 9 systems, with 8 of them surpassing prior SOTA. Finally, we show that the same learned policy transfers to a physical, real-world car without any tuning.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：**

**Title:** See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection
**Authors:** Amir Mallak, Erfan Aasi, Shiva Sreeram, Tsun-Hsuan Wang, Daniela Rus, Alaa Maalouf
**Categories:** cs.CV, cs.LG, cs.RO
**Published Date:** 2026-01-15

**1. 论文的主要贡献（2-3句话的简洁总结）：**

本研究提出了一种名为“随机斑块选择”（Stochastic-Patch-Selection, SPS）的新颖方法，旨在提升端到端自动驾驶策略的泛化能力和鲁棒性。通过随机掩盖基础模型提取的斑块特征，SPS迫使策略模型学习对不同但完整的场景视图具有不变性的决策依据，从而有效缓解了因特征冗余导致的过拟合问题。实验证明，SPS在各种分布外（OOD）场景下显著优于现有技术，并能成功迁移至真实世界车辆。

**2. 关键创新或方法论：**

*   **核心创新：随机斑块选择 (Stochastic-Patch-Selection, SPS)。** 这是本论文最核心的贡献。其基本思想是，在将从基础模型（如BLIP2）提取的斑块特征输入到自动驾驶策略模型之前，随机地“丢弃”一部分斑块特征。这种随机掩盖并非随机删除，而是保留了剩余斑块的空间布局，使得策略模型在每次训练迭代中都能看到同一场景的不同“视角”或“子集”。
*   **理论基础：特征冗余分析。** 作者通过PCA和跨斑块相似性分析，量化了基础模型提取的斑块特征的高度冗余性。他们发现，即使只使用少量主成分（17/64）就能捕获大部分方差，并且不同斑块特征之间存在强烈的相关性。这种冗余被认为是导致策略模型过拟合 spurious correlations（虚假相关性）并损害OOD鲁棒性的根源。
*   **方法论细节：**
    *   **基础模型：** 使用了如BLIP2这样的基础模型来提取图像的斑块对齐特征。
    *   **随机掩盖：** 在特征层面进行随机掩盖，而不是在原始图像层面。
    *   **空间布局保留：** 尽管掩盖了部分斑块特征，但剩余斑块的空间关系被保留，这使得模型能够理解场景的整体结构。
    *   **目标：** 训练一个对特定斑块特征不敏感，而是依赖于更本质、更具不变性的场景理解的策略模型。

**3. 对该领域的潜在影响：**

*   **提升自动驾驶的OOD鲁棒性：** 这是最直接的影响。当前自动驾驶系统在面对未见过或罕见的场景时表现不佳，SPS提供了一种有效且简单的方法来解决这一关键问题。
*   **降低对大规模、多样化OOD数据的依赖：** 通过在训练过程中引入随机性来模拟OOD情况，可能减少对收集和标注大量OOD数据的需求，降低开发成本。
*   **提高模型效率：** 论文提到SPS可以使模型训练速度提高2.4倍，这对于需要快速迭代和部署的自动驾驶领域至关重要。
*   **推动基础模型在下游任务中的应用：** 本研究展示了如何有效地利用基础模型提取的丰富但冗余的特征，并提出了一种克服其潜在缺点的策略，这为其他依赖基础模型的下游任务提供了借鉴。
*   **促进对模型决策机制的理解：** SPS迫使模型学习更具泛化性的特征表示，有助于我们理解模型是如何在复杂和不确定的环境中做出决策的。

**4. 可能受益于此研究的相关领域或应用：**

*   **其他端到端自动驾驶系统：** 任何使用基础模型提取特征的端到端自动驾驶方法都可以尝试SPS。
*   **机器人感知与导航：** 在机器人领域，尤其是在复杂、动态且不可预测的环境中，鲁棒的感知和导航能力至关重要。
*   **计算机视觉中的泛化性研究：** 任何需要模型在分布外数据上表现良好的计算机视觉任务，例如图像识别、目标检测、语义分割等，都可以借鉴SPS的思想。
*   **多模态学习：** 基础模型通常是多模态的，SPS的思想可以推广到处理多模态数据时如何处理冗余信息以提高泛化性。
*   **对抗性训练的替代方案：** SPS提供了一种非对抗性的方法来提高模型的鲁棒性，这可能比传统的对抗性训练更易于实现和理解。

**5. 从摘要中可以推断出的局限性：**

*   **随机性的选择：** 虽然随机性是SPS的核心优势，但如何选择合适的掩盖比例（masking rate）可能是一个需要仔细调整的超参数。摘要中提到进行了“ablations over masking rates”，暗示了这一点。
*   **基础模型的依赖性：** SPS的效果很大程度上依赖于基础模型提取的特征的质量和冗余程度。如果基础模型提取的特征本身就非常稀疏且信息量不足，SPS的效果可能会打折扣。
*   **计算开销的权衡：** 虽然SPS提高了训练效率，但随机掩盖过程本身会增加一定的计算开销。在某些对计算资源极其敏感的场景下，可能需要权衡。
*   **特征重组的潜在影响：** 摘要中提到“patch-feature reorganization”，这暗示了除了随机掩盖，可能还有其他特征处理方式，这些方式的有效性和普适性仍需进一步验证。
*   **“See Less, Drive Better”的直观解释：** 虽然论文解释了“less”是指特征的“数量”减少，但从直观上理解“看得少反而开得好”可能需要更深入的解释，以避免误解。
*   **泛化到极端OOD场景：** 尽管在“all OOD scenarios”下表现优异，但对于极其罕见或完全未曾预料到的场景，其泛化能力仍可能存在上限。

**总结来说，这篇论文的亮点在于其对基础模型特征冗余问题的深刻洞察，并提出了一个简单而有效的解决方案——随机斑块选择。这种方法不仅在理论上解释了为什么会存在泛化性问题，更在实践中展示了显著的性能提升，尤其是在自动驾驶领域至关重要的OOD鲁棒性方面。其易于实现且能迁移到真实世界的特性，使其具有很高的研究价值和应用前景。**

**Key Findings:**

- We present Stochastic-Patch-Selection (SPS), a simple yet effective approach for learning policies that are more robust, generalizable, and efficient.
- Extensive experiments confirm that across all OOD scenarios, our method outperforms the state of the art (SOTA), achieving a $6.2$% average improvement and up to $20.4$% in closed-loop simulations, while being $2.4\times$ faster.
- Finally, we show that the same learned policy transfers to a physical, real-world car without any tuning.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10707v1)
- [arXiv](https://arxiv.org/abs/2601.10707v1)

---

<a id='2601.10649v1'></a>
## [CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning](https://arxiv.org/abs/2601.10649v1)

**Authors:** Darshan Singh, Arsha Nagrani, Kawshik Manikantan, Harman Singh, Dinesh Tewari, Tobias Weyand, Cordelia Schmid, Anelia Angelova, Shachi Dave

**Published:** 2026-01-15

**Categories:** cs.CV

**Abstract:**

Recent advancements in video models have shown tremendous progress, particularly in long video understanding. However, current benchmarks predominantly feature western-centric data and English as the dominant language, introducing significant biases in evaluation. To address this, we introduce CURVE (Cultural Understanding and Reasoning in Video Evaluation), a challenging benchmark for multicultural and multilingual video reasoning. CURVE comprises high-quality, entirely human-generated annotations from diverse, region-specific cultural videos across 18 global locales. Unlike prior work that relies on automatic translations, CURVE provides complex questions, answers, and multi-step reasoning steps, all crafted in native languages. Making progress on CURVE requires a deeply situated understanding of visual cultural context. Furthermore, we leverage CURVE's reasoning traces to construct evidence-based graphs and propose a novel iterative strategy using these graphs to identify fine-grained errors in reasoning. Our evaluations reveal that SoTA Video-LLMs struggle significantly, performing substantially below human-level accuracy, with errors primarily stemming from the visual perception of cultural elements. CURVE will be publicly available under https://github.com/google-deepmind/neptune?tab=readme-ov-file\#minerva-cultural

**Analysis:**

好的，这是对论文“CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning”的全面中文摘要：

**论文题目：** CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning (CURVE：一个用于文化和多语言长视频推理的基准)

**作者：** Darshan Singh, Arsha Nagrani, Kawshik Manikantan, Harman Singh, Dinesh Tewari, Tobias Weyand, Cordelia Schmid, Anelia Angelova, Shachi Dave

**1. 研究问题/核心挑战：**

当前视频理解模型在长视频理解方面取得了显著进展，但现有的评估基准存在严重的**西方中心主义和英语中心主义偏见**。这导致模型在评估时存在显著的偏差，限制了它们在多元文化和多语言环境下的有效性。现有的多语言数据集通常依赖于自动翻译，这可能引入错误并无法捕捉到细微的文化差异。因此，论文旨在解决的核心问题是：**如何构建一个能够公平、全面地评估模型在多元文化和多语言长视频推理能力上的基准，并揭示当前模型在这方面的局限性。**

**2. 主要创新点/方法论贡献：**

*   **CURVE 基准的构建：** 论文引入了 CURVE (Cultural Understanding and Reasoning in Video Evaluation) 基准，这是一个**大规模、多文化、多语言的长视频推理数据集**。
    *   **高质量、人工标注数据：** CURVE 包含来自全球 18 个不同地区和语言的视频，所有标注（问题、答案、多步推理过程）均由**本地专家**以**原生语言**手工创建，确保了文化真实性和语言准确性。
    *   **复杂的多步推理：** 问题设计要求模型具备深入的视觉文化理解、多模态推理能力以及对复杂时间关系的把握，需要至少三种技能才能解答。
    *   **详细的推理轨迹：** 每个问题都配有详细的人工推理轨迹，这为深入分析模型错误提供了基础。
*   **证据图 (Evidence Graph) 和迭代错误隔离 (Iterative Error Isolation)：**
    *   **证据图：** 利用 CURVE 的推理轨迹，论文构建了**有向无环图 (DAG)** 来形式化人类的推理过程，节点代表原子证据，边代表依赖关系。
    *   **迭代错误隔离：** 提出了一种新颖的**迭代策略**，通过在证据图上进行遍历，识别、标记和纠正模型在推理过程中的错误，从而实现对模型失败模式的**细粒度诊断**。这种方法能够揭示模型在感知和推理阶段的深层问题。

**3. 主要结果及其意义：**

*   **模型性能显著低于人类水平：** 论文的评估结果显示，即使是当前最先进的视频语言模型 (Video-LLMs)，在 CURVE 基准上的表现也**远低于人类基线**。例如，Gemini-2.5-Pro 的最高准确率仅为 45.07%，而人类基线为 95.22%。
*   **文化和语言偏见明显：** 模型在不同地区和语言上的表现存在**显著的文化差异**。模型在相对高资源语言（如韩语、英式英语）上表现较好，但在南印度语言（如泰卢固语、泰米尔语）等低资源地区则表现出严重的性能下降，这有力地证明了当前模型存在的文化和语言偏见。
*   **75% 的失败源于文化视觉感知：** 错误分析表明，**约 75% 的模型失败归因于文化视觉感知错误**（如时间定位、空间定位、属性识别等），而非纯粹的逻辑推理错误。这表明模型在理解和解释文化相关的视觉元素方面存在巨大挑战。
*   **音频信息的重要性：** 实验表明，结合音频信息可以显著提升模型在 CURVE 基准上的性能，尤其是在非英语地区，证明了**多模态理解的必要性**。
*   **推理预算的影响：** 增加模型的“思考预算”（即用于中间推理的 token 数量）可以带来性能提升，但收益会饱和，且整体性能仍远低于人类。

**4. 论文中提到的局限性：**

*   **LLM 在错误标记中的潜在偏见：** 论文的诊断流程依赖于 LLM（Gemini-2.5-Pro）进行错误分类和提示生成。虽然采取了多项措施来确保鲁棒性（如结构化任务、清晰的错误定义、提示工程和多数投票），但仍承认 LLM 本身可能存在固有的偏见。
*   **18 个地区并非详尽无遗：** 虽然 CURVE 覆盖了 18 个地区，但作者承认这并非详尽无遗，未来仍有扩展空间。

**5. 潜在的未来研究方向：**

*   **开发更鲁棒的 LLM 诊断工具：** 进一步研究和开发更可靠的、基于 LLM 的诊断工具，或者探索**人机协作的 LLM 错误验证**方法，以克服 LLM 潜在的偏见。
*   **扩展 CURVE 的覆盖范围：** 将 CURVE 扩展到更多的地区和语言，以提供更全面的文化和语言覆盖。
*   **开发更具可解释性的模型：** CURVE 提供的详细推理轨迹为开发更具可解释性、过程导向的模型提供了机会。
*   **解决文化视觉感知瓶颈：** 重点研究如何提升模型在理解和解释文化相关视觉元素方面的能力。
*   **探索更有效的多模态融合策略：** 进一步研究如何更有效地融合音频和视觉信息，以实现更全面的文化理解。

**总结：**

CURVE 基准的提出是视频理解领域的一项重要贡献，它**首次系统性地解决了现有评估基准在文化和语言多样性方面的不足**。通过提供高质量、多语言、人工标注的长视频推理数据，以及一套创新的错误分析方法，CURVE 揭示了当前最先进视频语言模型在理解多元文化内容方面的严峻挑战，特别是其在**文化视觉感知方面的显著短板**。这为未来开发更公平、更具全球适应性的多模态 AI 系统指明了明确的研究方向。

**Key Findings:**

- To address this, we introduce CURVE (Cultural Understanding and Reasoning in Video Evaluation), a challenging benchmark for multicultural and multilingual video reasoning.
- Furthermore, we leverage CURVE's reasoning traces to construct evidence-based graphs and propose a novel iterative strategy using these graphs to identify fine-grained errors in reasoning.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10649v1)
- [arXiv](https://arxiv.org/abs/2601.10649v1)

---

<a id='2601.10632v1'></a>
## [CoMoVi: Co-Generation of 3D Human Motions and Realistic Videos](https://arxiv.org/abs/2601.10632v1)

**Authors:** Chengfeng Zhao, Jiazhi Shu, Yubo Zhao, Tianyu Huang, Jiahao Lu, Zekai Gu, Chengwei Ren, Zhiyang Dou, Qing Shuai, Yuan Liu

**Published:** 2026-01-15

**Categories:** cs.CV

**Abstract:**

In this paper, we find that the generation of 3D human motions and 2D human videos is intrinsically coupled. 3D motions provide the structural prior for plausibility and consistency in videos, while pre-trained video models offer strong generalization capabilities for motions, which necessitate coupling their generation processes. Based on this, we present CoMoVi, a co-generative framework that couples two video diffusion models (VDMs) to generate 3D human motions and videos synchronously within a single diffusion denoising loop. To achieve this, we first propose an effective 2D human motion representation that can inherit the powerful prior of pre-trained VDMs. Then, we design a dual-branch diffusion model to couple human motion and video generation process with mutual feature interaction and 3D-2D cross attentions. Moreover, we curate CoMoVi Dataset, a large-scale real-world human video dataset with text and motion annotations, covering diverse and challenging human motions. Extensive experiments demonstrate the effectiveness of our method in both 3D human motion and video generation tasks.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：CoMoVi: Co-Generation of 3D Human Motions and Realistic Videos**

**1. 论文的主要贡献（2-3句话）：**

本论文提出了一种新颖的协同生成框架 CoMoVi，它能够同步生成逼真的 3D 人体运动和 2D 人体视频。该框架的核心在于认识到 3D 运动和 2D 视频生成过程的内在耦合性，并利用预训练视频扩散模型（VDMs）的强大能力来指导和增强运动生成。通过这种协同方式，CoMoVi 实现了更具结构性、合理性和一致性的运动与视频的联合生成。

**2. 关键创新或方法论：**

*   **协同生成框架（Co-Generative Framework）：** 这是最核心的创新。CoMoVi 并非独立生成 3D 运动和 2D 视频，而是将两者置于一个统一的扩散去噪循环中，实现同步生成。
*   **耦合两个视频扩散模型（VDMs）：** 论文明确指出使用了两个 VDMs，一个用于 3D 运动生成，一个用于 2D 视频生成，并通过特定的机制将它们耦合起来。
*   **有效的 2D 人体运动表示：** 为了能够利用预训练 VDMs 的强大先验知识，作者设计了一种新的 2D 人体运动表示方法。这使得预训练模型能够更好地理解和生成运动信息。
*   **双分支扩散模型（Dual-Branch Diffusion Model）：** 这是一个具体的实现方式，表明模型架构上存在两个并行的分支，分别处理运动和视频，但它们之间存在信息交互。
*   **互特征交互与 3D-2D 交叉注意力（Mutual Feature Interaction and 3D-2D Cross Attentions）：** 这是实现协同生成和信息耦合的关键技术。
    *   **互特征交互：** 表明两个分支在生成过程中会互相影响和学习对方的特征。
    *   **3D-2D 交叉注意力：** 这是实现跨模态信息融合的关键机制。3D 运动的结构信息可以通过交叉注意力机制注入到视频生成过程中，反之亦然，确保两者的一致性。
*   **CoMoVi 数据集：** 论文还提出了一个大规模、真实世界的人体视频数据集，包含文本和运动标注，这对于训练和评估此类模型至关重要，尤其是在覆盖多样化和挑战性运动方面。

**3. 对该领域的潜在影响：**

*   **提升人机交互和虚拟现实体验：** 更逼真、更自然的 3D 人体运动和视频生成是实现沉浸式虚拟现实、增强现实以及更具交互性的数字人体的基础。
*   **推动内容创作的自动化和智能化：** 能够根据文本描述或简单的输入生成复杂的 3D 运动和视频，将极大地降低内容创作的门槛，加速动画、游戏、电影等行业的生产流程。
*   **促进对人体运动理解的深入研究：** 协同生成模型迫使研究者更深入地理解 3D 运动和 2D 视觉表现之间的关系，从而推动对人体运动的建模和理解。
*   **为其他多模态生成任务提供范例：** CoMoVi 的协同生成思想和交叉注意力机制可以被借鉴到其他需要联合生成不同模态数据的任务中。
*   **加速扩散模型在复杂场景的应用：** 论文展示了如何将强大的预训练扩散模型应用于更复杂的、多模态的生成任务，并取得了显著效果。

**4. 可能受益的相关领域或应用：**

*   **虚拟现实 (VR) 和增强现实 (AR)：** 创建更逼真的虚拟角色和交互体验。
*   **游戏开发：** 自动化角色动画生成，提高开发效率。
*   **电影和动画制作：** 辅助或自动化角色动画的创建。
*   **数字人/虚拟主播：** 生成更自然、更具表现力的人体动作。
*   **体育分析和训练：** 捕捉和复现运动员的动作，用于分析和指导。
*   **机器人学：** 为机器人生成更自然的人体运动轨迹。
*   **行为识别和理解：** 生成特定行为的视频，用于训练和测试识别模型。
*   **医疗康复：** 生成康复训练动作的示范视频。

**5. 从摘要中可以推断出的局限性：**

*   **计算资源需求：** 扩散模型本身就计算密集，而 CoMoVi 耦合了两个 VDMs，并使用了交叉注意力，这很可能意味着极高的计算资源需求（GPU 内存和计算时间），尤其是在训练阶段。
*   **对预训练模型的依赖：** 方法的有效性在很大程度上依赖于预训练 VDMs 的质量和泛化能力。如果预训练模型在某些特定类型的运动或场景上表现不佳，可能会影响 CoMoVi 的整体性能。
*   **数据集的覆盖范围：** 尽管论文声称数据集“覆盖多样和挑战性的人体运动”，但任何数据集都可能存在未覆盖的极端情况或特定文化背景下的动作，这可能导致模型在这些区域表现受限。
*   **生成视频的真实感和一致性：** 虽然摘要声称生成“逼真”视频，但“逼真”是一个相对概念。在某些复杂场景下，例如精细的手部动作、面部表情的细微变化，或者与环境的物理交互，可能仍然存在挑战，需要进一步的评估。
*   **文本到运动/视频的映射精度：** 如果生成过程是基于文本输入的，那么文本描述的模糊性或复杂性可能会影响最终生成结果的准确性。摘要中提到了“文本和运动标注”，但未详细说明文本到运动的映射机制的鲁棒性。
*   **3D 运动的精度和细节：** 尽管协同生成，但 3D 运动本身的精度和细节（例如关节的自由度、骨骼的拓扑结构）仍然是生成质量的关键。摘要未详细说明 3D 运动表示的细节和精度。

总而言之，CoMoVi 是一项非常有前景的研究，它通过巧妙地耦合两个扩散模型并引入交叉注意力机制，解决了 3D 运动和 2D 视频生成之间的内在联系，有望在多个领域带来突破。然而，其计算成本和对预训练模型的依赖性是需要关注的方面。

**Key Findings:**

- Based on this, we present CoMoVi, a co-generative framework that couples two video diffusion models (VDMs) to generate 3D human motions and videos synchronously within a single diffusion denoising loop.
- Extensive experiments demonstrate the effectiveness of our method in both 3D human motion and video generation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10632v1)
- [arXiv](https://arxiv.org/abs/2601.10632v1)

---

<a id='2601.10611v1'></a>
## [Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding](https://arxiv.org/abs/2601.10611v1)

**Authors:** Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, Vincent Shao, Yue Yang, Weikai Huang, Ziqi Gao, Taira Anderson, Jianrui Zhang, Jitesh Jain, George Stoica, Winson Han, Ali Farhadi, Ranjay Krishna

**Published:** 2026-01-15

**Categories:** cs.CV, cs.AI

**Abstract:**

Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).

**Analysis:**

作为一名计算机视觉和机器学习领域的专家，我对这篇论文摘要进行了深入分析，并为您提炼出以下关键信息：

**1. 论文的主要贡献（2-3句话）：**

Molmo2 是一系列新的视觉-语言模型 (VLMs)，在开放权重和数据模型中达到了最先进的水平，并在点驱动的像素级视频和图像理解与定位任务上展现了卓越的新能力。其核心贡献在于构建了七个新的视频数据集和两个多图像数据集，这些数据集均独立于闭源 VLM 收集，为开放社区提供了改进 SOTA 模型所需的基础。

**2. 关键创新或方法论：**

*   **全新的、独立收集的数据集：** 这是 Molmo2 最突出的贡献。论文强调了这些数据集（包括详细视频字幕、自由形式视频问答、复杂查询对象跟踪和创新的视频点指数据集）的创建过程不依赖于闭源 VLM，这直接解决了当前开放模型依赖合成数据或数据不透明的问题。
*   **高效的训练配方：** 论文提出了一个高效的训练方案，包括：
    *   **高效打包和消息树编码：** 这可能是一种优化数据输入和模型处理效率的技术，尤其是在处理长视频或复杂信息时。
    *   **视觉标记的双向注意力：** 这意味着模型能够更有效地在视觉信息和文本信息之间建立联系，并且这种联系是双向的，可能提升了对视觉内容的理解深度。
    *   **新颖的标记权重策略：** 这表明模型在处理不同类型的视觉或文本标记时，会采用动态的权重分配机制，以突出关键信息，从而提升性能。

**3. 对该领域的潜在影响：**

*   **推动开放 VLM 的发展：** Molmo2 的出现极大地弥补了当前开放模型在数据和能力上的短板，为研究人员提供了一个强大的基准和可用的资源，有望加速开放 VLM 的研究和应用。
*   **提升视频理解和定位的精度：** 通过引入新的数据集和更优化的训练方法，Molmo2 在视频计数、字幕生成以及尤其是在视频定位（点指、跟踪）方面取得了显著的性能提升，甚至超越了一些专有模型，这标志着该领域向前迈进了一大步。
*   **降低研究门槛：** 开放权重和数据的发布，使得更多研究者和开发者能够接触到最先进的模型和训练方法，降低了研究和开发的门槛，促进了社区的创新。

**4. 可能受益的相关领域或应用：**

*   **视频内容检索和搜索：** 更精确的视频理解和定位能力将极大地提升视频内容的检索效率和准确性。
*   **视频编辑和内容生成：** 能够理解视频中的具体对象和动作，并进行像素级定位，为自动化视频编辑和更精细的内容生成提供了可能。
*   **机器人和自动驾驶：** 实时、精确的视频理解和对象跟踪是机器人导航、环境感知和自动驾驶的关键技术。
*   **辅助技术：** 例如，为视障人士提供更详细的视频描述，或者通过点指交互来控制设备。
*   **教育和培训：** 能够对视频内容进行精细的问答和定位，可以用于创建更具交互性的教育内容。
*   **内容审核和安全：** 更准确的视频理解有助于识别不当内容或进行安全监控。

**5. 从摘要中可以推断出的局限性：**

*   **模型规模与性能权衡：** 摘要中提到了“best-in-class 8B model”，这表明论文可能主要关注 80 亿参数的模型。虽然其性能优异，但对于需要更大模型以获得更高泛化能力或处理更复杂任务的场景，可能需要进一步的研究。
*   **长视频的性能：** 摘要提到“competitive on long-videos”，这暗示着在处理长视频方面，Molmo2 的性能可能不如其在短视频上的表现那样具有压倒性优势，可能仍有提升空间。
*   **数据集的覆盖范围：** 虽然论文发布了大量新数据集，但任何数据集都可能存在覆盖范围的局限性。例如，数据集的领域、多样性、标注的细致程度等，都可能影响模型在未见过的数据上的泛化能力。
*   **计算资源需求：** 训练和部署如此强大的 VLM 模型通常需要大量的计算资源，这可能会限制其在资源受限环境下的应用。
*   **“一些任务”的表述：** 在与 Gemini 3 Pro 的比较中，摘要使用了“surpasses proprietary models on some tasks”的表述。这暗示着 Molmo2 并非在所有任务上都超越了专有模型，可能在某些特定任务或评估指标上仍有差距。

总而言之，Molmo2 的发布是一项重要的研究成果，它通过提供高质量的开放数据和模型，以及创新的训练方法，显著推动了视频-语言模型领域的发展，尤其是在像素级视频理解和定位方面。这为未来的研究和应用打开了新的可能性。

**Key Findings:**

- As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models.
- We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks.
- Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance.
- Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos.
- On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10611v1)
- [arXiv](https://arxiv.org/abs/2601.10611v1)

---

<a id='2601.10592v1'></a>
## [Action100M: A Large-scale Video Action Dataset](https://arxiv.org/abs/2601.10592v1)

**Authors:** Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, Pascale Fung

**Published:** 2026-01-15

**Categories:** cs.CV

**Abstract:**

Inferring physical actions from visual observations is a fundamental capability for advancing machine intelligence in the physical world. Achieving this requires large-scale, open-vocabulary video action datasets that span broad domains. We introduce Action100M, a large-scale dataset constructed from 1.2M Internet instructional videos (14.6 years of duration), yielding O(100 million) temporally localized segments with open-vocabulary action supervision and rich captions. Action100M is generated by a fully automated pipeline that (i) performs hierarchical temporal segmentation using V-JEPA 2 embeddings, (ii) produces multi-level frame and segment captions organized as a Tree-of-Captions, and (iii) aggregates evidence with a reasoning model (GPT-OSS-120B) under a multi-round Self-Refine procedure to output structured annotations (brief/detailed action, actor, brief/detailed caption). Training VL-JEPA on Action100M demonstrates consistent data-scaling improvements and strong zero-shot performance across diverse action recognition benchmarks, establishing Action100M as a new foundation for scalable research in video understanding and world modeling.

**Analysis:**

好的，这是对论文“Action100M: A Large-scale Video Action Dataset”的全面中文摘要，重点突出了其在计算机视觉领域的新颖性和重要性：

**论文题目：** Action100M: A Large-scale Video Action Dataset

**作者：** Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, Pascale Fung

**摘要：**

**1. 研究问题/核心挑战：**
当前，理解物理世界的机器智能需要能够从视觉观察中推断出物理动作的能力。然而，现有的视频动作数据集在规模、领域覆盖范围和开放词汇能力方面存在显著不足。现有的数据集往往规模有限，且专注于狭窄的领域（如烹饪、玩具组装），这阻碍了开发能够理解广泛动作的通用模型。因此，构建一个大规模、开放词汇、跨领域的数据集是推进视频理解和世界建模的关键。

**2. 主要创新点/方法论贡献：**
论文的核心贡献是提出了 **Action100M**，一个前所未有的大规模视频动作数据集。其主要创新点体现在以下几个方面：

*   **大规模数据构建：** Action100M 从 120 万个互联网教学视频中构建，总时长达 14.6 年，生成了超过 1 亿个时间局部化的动作片段，并提供了开放词汇的动作监督和丰富的文本描述。
*   **全自动化流水线：** 数据集是通过一个完全自动化的流水线生成的，该流水线包含三个关键阶段：
    *   **分层时间分割：** 利用 V-JEPA 2 嵌入，将视频分解成具有时间连贯性的层级化片段，捕捉从精细动作到长流程步骤的多种抽象层次。
    *   **多层次字幕生成（Tree-of-Captions）：** 为每个片段生成多层次的帧和片段字幕，组织成一个“字幕树”（Tree-of-Captions）结构，同时捕获局部细节和全局上下文。
    *   **LLM 聚合与自我精炼：** 使用强大的推理模型 GPT-OSS-120B，通过多轮“自我精炼”（Self-Refine）过程，整合字幕树中的信息，生成结构化的标注，包括简要/详细动作描述、执行者以及简要/详细视频字幕。
*   **开放词汇和结构化标注：** 数据集提供了开放词汇的动作标签，这意味着模型可以理解未在训练集中明确出现的动作。同时，标注是结构化的，包含动作的多个层级描述，这对于理解复杂活动至关重要。
*   **语义重采样：** 为了解决动作频率的长尾分布问题，论文还提出了语义重采样策略，通过聚类动作描述并进行平衡采样，以提高训练效率和模型性能。

**3. 主要结果及其意义：**
论文通过在 Action100M 上训练 VL-JEPA 模型，展示了该数据集的有效性：

*   **显著的性能提升：** 在多个零样本动作识别和文本到视频检索基准测试中，使用 Action100M 训练的 VL-JEPA 模型取得了持续的数据缩放改进和强大的零样本性能。
*   **跨领域泛化能力：** Action100M 训练的模型在各种动作识别任务上表现出色，包括运动密集型任务（如 Something-something-v2）和步骤识别任务（如 COIN 和 CrossTask），证明了其跨领域泛化能力。
*   **奠定基础：** Action100M 被认为是视频理解和世界建模领域可扩展研究的新基础，为开发更强大的开放域动作识别器和具身智能体提供了关键的数据支撑。
*   **数据质量保证：** 通过 LLM 聚合和自我精炼机制，有效减少了字幕中的幻觉，提高了标注的准确性和一致性。

**4. 论文中提到的局限性：**
*   **数据源的固有局限：** 数据集主要来源于互联网教学视频，虽然覆盖广泛，但可能存在部分内容不完整、质量参差不齐的情况。
*   **动作频率的长尾分布：** 尽管进行了语义重采样，但大规模动作数据中固有的长尾分布仍然是一个挑战，某些动作的出现频率远高于其他动作。
*   **计算成本：** 构建如此大规模的数据集需要巨大的计算资源（130 万 V100 GPU 小时用于分割和字幕生成，0.3 百万 H100/H200 GPU 小时用于 LLM 聚合）。

**5. 潜在的未来研究方向：**
*   **更精细的动作理解：** Action100M 的丰富标注为研究更细粒度的动作识别、动作意图预测以及动作之间的因果关系提供了可能。
*   **具身智能和世界模型：** 数据集支持具身学习、可穿戴辅助应用以及物理世界建模，为开发能够与物理世界进行交互的智能体奠定了基础。
*   **长时序规划和推理：** Action100M 的层级化结构和跨领域覆盖，为研究长时序规划、多尺度活动理解以及需要推理复杂上下文的世界模型提供了理想的平台。
*   **数据效率和模型泛化：** 进一步探索如何更有效地利用 Action100M 进行模型训练，以及如何提升模型在未见过动作上的泛化能力。
*   **多模态融合：** 结合视频内容、文本描述以及可能的其他模态信息，进行更深入的视频理解和生成。

**总结：**
Action100M 是一个里程碑式的大规模视频动作数据集，它通过创新的全自动化流水线，生成了海量、开放词汇、结构化的动作标注。该数据集的发布极大地推动了视频理解和世界建模领域的研究进展，为开发更强大、更通用的视觉智能体提供了坚实的数据基础，并为未来的研究开辟了新的方向。

**Key Findings:**

- We introduce Action100M, a large-scale dataset constructed from 1.2M Internet instructional videos (14.6 years of duration), yielding O(100 million) temporally localized segments with open-vocabulary action supervision and rich captions.
- Training VL-JEPA on Action100M demonstrates consistent data-scaling improvements and strong zero-shot performance across diverse action recognition benchmarks, establishing Action100M as a new foundation for scalable research in video understanding and world modeling.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10592v1)
- [arXiv](https://arxiv.org/abs/2601.10592v1)

---

<a id='2601.10587v1'></a>
## [Adversarial Evasion Attacks on Computer Vision using SHAP Values](https://arxiv.org/abs/2601.10587v1)

**Authors:** Frank Mollard, Marcus Becker, Florian Roehrbein

**Published:** 2026-01-15

**Categories:** cs.CV, cs.AI

**Abstract:**

The paper introduces a white-box attack on computer vision models using SHAP values. It demonstrates how adversarial evasion attacks can compromise the performance of deep learning models by reducing output confidence or inducing misclassifications. Such attacks are particularly insidious as they can deceive the perception of an algorithm while eluding human perception due to their imperceptibility to the human eye. The proposed attack leverages SHAP values to quantify the significance of individual inputs to the output at the inference stage. A comparison is drawn between the SHAP attack and the well-known Fast Gradient Sign Method. We find evidence that SHAP attacks are more robust in generating misclassifications particularly in gradient hiding scenarios.

**Analysis:**

好的，这是对 Frank Mollard、Marcus Becker 和 Florian Röhrbein 的论文“Adversarial Evasion Attacks on Computer Vision using SHAP Values”的全面中文摘要，其中包含了您要求的五个部分：

**论文题目：** Adversarial Evasion Attacks on Computer Vision using SHAP Values
**作者：** Frank Mollard, Marcus Becker, Florian Röhrbein

**摘要**

**1. 研究问题/核心挑战：**

本文旨在解决深度学习模型在计算机视觉任务中面临的对抗性规避攻击问题。这类攻击通过对输入数据进行微小但难以察觉的扰动，来降低模型的性能，导致输出置信度下降或产生错误的分类（误分类）。这些攻击尤其危险，因为它们能够欺骗算法的感知，同时又逃避人眼的检测。研究的核心问题是如何更有效地生成这些对抗性样本，尤其是在模型对梯度信息进行隐藏（gradient hiding）的场景下，传统基于梯度的攻击方法可能效果不佳。

**2. 主要创新点/方法论贡献：**

该论文提出了一种新颖的**白盒对抗性规避攻击方法，该方法利用 SHAP (SHapley Additive exPlanations) 值**。与传统的基于梯度的方法（如 FGSM）不同，SHAP 值能够量化每个输入特征（在此论文中是图像的像素）对模型最终输出的贡献程度和重要性。论文的核心贡献在于：

*   **利用 SHAP 值进行攻击：** 将 SHAP 值作为一种解释模型决策的方法，转化为一种攻击工具。通过分析像素对模型预测的贡献，识别出对模型决策影响最大的像素，并对其进行有针对性的修改。
*   **更精细的攻击策略：** SHAP 攻击能够考虑像素的**影响幅度**，而不仅仅是梯度的符号（如 FGSM）。这意味着 SHAP 攻击可以更精细地操纵对模型输出有显著影响的像素，从而可能需要更少的像素修改就能达到误分类的目的，使得攻击更加隐蔽。
*   **对比分析：** 将提出的 SHAP 攻击与经典的 Fast Gradient Sign Method (FGSM) 进行了详细的比较，包括在不同数据集和模型架构下的性能评估。

**3. 主要结果及意义：**

研究的主要结果表明：

*   **SHAP 攻击的有效性：** SHAP 攻击在生成误分类方面比 FGSM 更为**鲁棒**，尤其是在梯度隐藏的场景下。实验结果（如表 1 和图 7、图 8）显示，在相同的攻击强度下，SHAP 攻击通常能达到更高的误分类率。
*   **攻击的隐蔽性：** SHAP 攻击通过关注像素的影响幅度，能够实现更精细的扰动，从而在保持攻击效果的同时，可能进一步增强攻击的隐蔽性。
*   **SHAP 值的双重价值：** 研究揭示了 SHAP 值不仅是理解模型决策的强大工具，还可以被用来**识别和利用模型的弱点**，从而进行对抗性攻击。这强调了深度学习模型在输入数据微小变化下的脆弱性，以及开发更具鲁棒性模型的必要性。

**4. 提及的局限性：**

论文中提到了一些局限性：

*   **计算复杂度高：** SHAP 值本身的计算就具有较高的复杂度，尤其是在处理高分辨率图像和大量类别时。这使得 SHAP 攻击在实际应用中可能需要**大量的计算资源**，限制了其在非常高分辨率的图像或视频上的应用。
*   **模型和数据访问要求：** SHAP 攻击属于白盒攻击，需要**访问模型本身**，并且需要**足够多的推理数据**来计算 SHAP 值。
*   **对模型和数据集的依赖性：** 虽然研究尝试通过使用不同的模型架构和数据集来排除依赖性，但 SHAP 值与模型和数据集的交互方式仍然是研究的一部分。

**5. 未来研究方向：**

基于上述研究，论文提出了以下未来研究方向：

*   **改进防御机制：** 重点应放在开发更有效的防御机制，以提高计算机视觉模型对这类对抗性攻击的**鲁棒性**。
*   **开发更泛化的模型：** 研究方向可以朝着开发更具泛化能力的模型发展，使得模型对**单个像素的依赖性降低**，从而更均衡地利用图像的整体信息进行分类。
*   **优化 SHAP 攻击的计算效率：** 探索更高效的 SHAP 值计算方法或采样策略，以降低其计算复杂度，使其在实际应用中更具可行性。
*   **探索 SHAP 在其他领域的应用：** 除了对抗性攻击，SHAP 值在模型解释和弱点分析方面的潜力也值得进一步挖掘。

总而言之，这篇论文通过引入基于 SHAP 值的对抗性攻击方法，为理解和应对深度学习模型在计算机视觉领域的脆弱性提供了新的视角和有力的工具。它不仅展示了 SHAP 值在攻击方面的潜力，也强调了开发更安全、更鲁棒的 AI 系统的紧迫性。

**Key Findings:**

- A comparison is drawn between the SHAP attack and the well-known Fast Gradient Sign Method.
- We find evidence that SHAP attacks are more robust in generating misclassifications particularly in gradient hiding scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2601.10587v1)
- [arXiv](https://arxiv.org/abs/2601.10587v1)

---

