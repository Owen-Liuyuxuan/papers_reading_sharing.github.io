time: 20251103

# Arxiv Computer Vision Papers - 2025-11-03

## Executive Summary

好的，这是一份为忙碌的研究人员准备的 Arxiv 计算机视觉领域每日报告执行摘要，涵盖了您提供的10篇论文：

---

**Arxiv 计算机视觉每日报告执行摘要 (2025-10-31)**

**概述与主要趋势：**

今日报告的论文展示了计算机视觉领域持续的多元化发展，尤其是在**多模态学习、大模型应用、鲁棒性提升和特定领域适应**方面。显著趋势包括：

1.  **大模型与多模态融合：** 多个工作探索了大型语言模型 (LLM) 和大型视觉模型 (如 SAM) 在图像编辑、检索和场景理解中的应用，强调了跨模态信息整合的重要性。
2.  **效率与鲁棒性：** 蒸馏技术用于提升模型效率，而针对遮挡、异构数据和复杂环境的鲁棒性研究依然是核心关注点。
3.  **特定领域应用：** 论文涵盖了水下场景理解、历史地图分割和复杂监控等具体应用，表明研究正深入解决现实世界挑战。
4.  **新型度量与表示：** 提出了新的距离度量和协作感知表示方法，以优化现有任务或实现新功能。

**特别显著或创新的论文：**

*   **"NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding" (Wei Xu et al.)**：这是一个非常具有前瞻性的工作。将大型多模态模型应用于水下场景，解决了传统视觉模型在复杂水下环境（如光照、能见度差）下的局限性，有望在海洋探索、水下机器人等领域产生重大影响。其特定领域的大模型构建思路值得关注。
*   **"Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model" (John Won et al.)**：这篇论文结合了扩散模型、世界模型和视觉-语言-动作模型，旨在构建更全面的具身智能体。其通过双流扩散和世界模型增强的架构，为实现更高级别的具身智能和决策提供了新的范式，代表了多模态具身智能的前沿探索。
*   **"Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing" (Yijia Wang et al.)**：这篇论文利用LLM来理解用户在图像编辑中的隐式意图，超越了简单的指令跟随，使得人机交互更加自然和智能。它展示了LLM在高级语义理解和推理方面赋能视觉任务的巨大潜力。

**新兴研究方向或技术：**

*   **世界模型与具身智能的结合：** "Dual-Stream Diffusion" 论文明确指出了将世界模型融入视觉-语言-动作模型的趋势，以实现更高级别的规划和决策。
*   **LLM驱动的意图理解与推理：** "Understanding the Implicit User Intention" 强调了LLM在解析复杂用户意图方面的能力，这将推动更智能、更人性化的交互式AI应用。
*   **异构协作感知：** "NegoCollab" 提出了异构协作感知中的表示协商，预示着多智能体系统在复杂、非同质环境下的协作将成为重要研究方向。
*   **特定领域大模型的构建与适应：** "NAUTILUS" 和 "MapSAM2" 都展示了将通用大模型（如 SAM）适应到特定领域（水下、历史地图）的有效性，这可能成为未来大模型落地应用的关键路径。

**建议阅读全文的论文：**

对于不同兴趣的研究人员，建议阅读以下论文：

*   **对于关注大模型和具身智能的：**
    *   **"Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model" (John Won et al.)**
    *   **"NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding" (Wei Xu et al.)**
*   **对于关注人机交互和智能编辑的：**
    *   **"Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing" (Yijia Wang et al.)**
*   **对于关注模型效率和鲁棒性的：**
    *   **"Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals" (Xiangyu Fan et al.)**
    *   **"Vision Transformer for Robust Occluded Person Reidentification in Complex Surveillance Scenes" (Bo Li et al.)**
*   **对于关注特定领域应用的：**
    *   **"MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series" (Xue Xia et al.)**

---

这份摘要旨在提供一个高层次的概览，帮助您快速识别与您研究兴趣最相关的论文。

---

## Table of Contents

1. [Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals](#2510.27684v1)
2. [Gaussian Combined Distance: A Generic Metric for Object Detection](#2510.27649v1)
3. [Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation](#2510.27632v1)
4. [MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series](#2510.27547v1)
5. [NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding](#2510.27481v1)
6. [RzenEmbed: Towards Comprehensive Multimodal Retrieval](#2510.27350v1)
7. [Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing](#2510.27335v1)
8. [Vision Transformer for Robust Occluded Person Reidentification in Complex Surveillance Scenes](#2510.27677v1)
9. [NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception](#2510.27647v1)
10. [Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](#2510.27607v1)

---

## Papers

<a id='2510.27684v1'></a>
## [Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals](https://arxiv.org/abs/2510.27684v1)

**Authors:** Xiangyu Fan, Zesong Qiu, Zhuguanyu Wu, Fanzhou Wang, Zhiqian Lin, Tianxiang Ren, Dahua Lin, Ruihao Gong, Lei Yang

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

Distribution Matching Distillation (DMD) distills score-based generative
models into efficient one-step generators, without requiring a one-to-one
correspondence with the sampling trajectories of their teachers. However,
limited model capacity causes one-step distilled models underperform on complex
generative tasks, e.g., synthesizing intricate object motions in text-to-video
generation. Directly extending DMD to multi-step distillation increases memory
usage and computational depth, leading to instability and reduced efficiency.
While prior works propose stochastic gradient truncation as a potential
solution, we observe that it substantially reduces the generation diversity of
multi-step distilled models, bringing it down to the level of their one-step
counterparts. To address these limitations, we propose Phased DMD, a multi-step
distillation framework that bridges the idea of phase-wise distillation with
Mixture-of-Experts (MoE), reducing learning difficulty while enhancing model
capacity. Phased DMD is built upon two key ideas: progressive distribution
matching and score matching within subintervals. First, our model divides the
SNR range into subintervals, progressively refining the model to higher SNR
levels, to better capture complex distributions. Next, to ensure the training
objective within each subinterval is accurate, we have conducted rigorous
mathematical derivations. We validate Phased DMD by distilling state-of-the-art
image and video generation models, including Qwen-Image (20B parameters) and
Wan2.2 (28B parameters). Experimental results demonstrate that Phased DMD
preserves output diversity better than DMD while retaining key generative
capabilities. We will release our code and models.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行详细分析。

---

### 论文摘要分析：Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals

**1. 论文主要贡献的简洁总结 (2-3 句话)**

这篇论文提出了 Phased DMD，一个多步蒸馏框架，旨在将基于分数的生成模型（score-based generative models）蒸馏成更高效、性能更好的多步生成器。它通过将信噪比（SNR）范围划分为子区间，并结合渐进式分布匹配和子区间内的分数匹配，解决了现有 DMD 方法在复杂任务上性能不足以及多步蒸馏带来的稳定性和多样性问题。Phased DMD 在保持生成多样性的同时，显著提升了蒸馏模型的生成能力。

**2. 关键创新或方法论**

Phased DMD 的核心创新在于其两项关键思想：

*   **渐进式分布匹配 (Progressive Distribution Matching)：** 模型将整个信噪比（SNR）范围划分为多个子区间。蒸馏过程是渐进的，模型在每个子区间内逐步学习并优化，从较低的 SNR 级别（更噪声）逐步精炼到较高的 SNR 级别（更清晰），从而更好地捕捉复杂的数据分布。这种分阶段的学习策略降低了单次学习的难度。
*   **子区间内的分数匹配 (Score Matching within Subintervals)：** 为了确保每个子区间内的训练目标是准确的，作者进行了严格的数学推导。这意味着在每个局部区间内，模型都精确地学习了对应数据分布的梯度（分数），从而保证了蒸馏的有效性和准确性。

此外，论文还提到了将**阶段性蒸馏 (phase-wise distillation)** 的思想与**专家混合模型 (Mixture-of-Experts, MoE)** 相结合，这进一步增强了模型容量并降低了学习难度，尽管摘要中没有详细说明 MoE 的具体应用方式，但暗示了其在处理复杂分布时的优势。

**3. 对该领域的潜在影响**

*   **提升生成模型效率与质量的平衡：** Phased DMD 提供了一种在保持生成质量（尤其是多样性）的同时，显著提高基于分数生成模型推理效率的新范式。这对于需要快速生成高质量内容的实际应用至关重要。
*   **推动多步蒸馏技术的发展：** 解决了传统多步蒸馏中常见的内存、计算深度、稳定性和多样性下降等问题，为未来多步蒸馏方法的研究开辟了新方向。
*   **赋能复杂生成任务：** 尤其在文本到视频生成等需要合成复杂对象运动的任务中，Phased DMD 能够使蒸馏模型更好地处理这些挑战，从而加速这些领域的发展。
*   **降低高性能模型部署成本：** 能够将大型、计算密集型的教师模型（如 Qwen-Image 20B, Wan2.2 28B）蒸馏成更小、更快的模型，有助于这些先进模型在资源受限环境中的部署和应用。

**4. 可能受益于这项研究的相关领域或应用**

*   **文本到图像/视频生成 (Text-to-Image/Video Generation)：** 摘要中明确提到了文本到视频生成中合成复杂对象运动的挑战，Phased DMD 在此领域具有直接应用价值。
*   **图像/视频编辑与合成 (Image/Video Editing and Synthesis)：** 任何需要高质量、高多样性图像或视频生成的任务，如风格迁移、超分辨率、图像修复等，都可以从更高效的生成器中受益。
*   **3D 内容生成 (3D Content Generation)：** 基于分数的生成模型在 3D 领域也取得了进展，Phased DMD 有望加速 3D 模型的生成效率。
*   **计算摄影 (Computational Photography)：** 快速生成高质量图像的能力可以应用于图像增强、去噪等领域。
*   **边缘设备上的生成模型部署 (Generative Model Deployment on Edge Devices)：** 蒸馏出的高效模型更适合在计算资源有限的设备上运行。

**5. 从摘要中可以推断出的任何局限性**

*   **MoE 的具体实现细节：** 摘要提到了 MoE，但没有详细说明其如何与阶段性蒸馏结合，以及它是否引入了额外的训练复杂性或推理开销（尽管通常 MoE 在推理时可以只激活部分专家）。
*   **数学推导的复杂性：** 摘要强调了“严格的数学推导”以确保子区间内训练目标的准确性。这可能意味着该方法的理论基础较为复杂，理解和实现可能需要较高的数学背景。
*   **子区间划分策略：** 摘要没有说明如何确定最佳的子区间数量和边界。这可能是一个需要仔细调优的超参数，对模型性能有重要影响。
*   **与教师模型的性能差距：** 尽管 Phased DMD 优于其他蒸馏方法，但摘要并未明确指出蒸馏后的模型与原始的、未蒸馏的教师模型在性能上（尤其是在极端复杂任务上）的差距。通常，蒸馏模型在某些方面仍会略逊于其教师模型。
*   **训练时间与资源：** 尽管目标是提高推理效率，但多步蒸馏本身，尤其是结合了 MoE 和分阶段训练，可能会增加训练时间和所需的计算资源。

---

总的来说，这篇论文提出了一种非常有前景的方法来解决基于分数生成模型蒸馏中的关键挑战，特别是在处理复杂生成任务和保持生成多样性方面。其分阶段和子区间分数匹配的思路是新颖且具有理论支撑的，有望在计算机视觉领域产生重要影响。

**Key Findings:**

- To address these limitations, we propose Phased DMD, a multi-step
distillation framework that bridges the idea of phase-wise distillation with
Mixture-of-Experts (MoE), reducing learning difficulty while enhancing model
capacity.
- We validate Phased DMD by distilling state-of-the-art
image and video generation models, including Qwen-Image (20B parameters) and
Wan2.2 (28B parameters).

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27684v1)
- [arXiv](https://arxiv.org/abs/2510.27684v1)

---

<a id='2510.27649v1'></a>
## [Gaussian Combined Distance: A Generic Metric for Object Detection](https://arxiv.org/abs/2510.27649v1)

**Authors:** Ziqian Guan, Xieyi Fu, Pengjun Huang, Hengyuan Zhang, Hubin Du, Yongtao Liu, Yinglin Wang, Qang Ma

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

In object detection, a well-defined similarity metric can significantly
enhance model performance. Currently, the IoU-based similarity metric is the
most commonly preferred choice for detectors. However, detectors using IoU as a
similarity metric often perform poorly when detecting small objects because of
their sensitivity to minor positional deviations. To address this issue, recent
studies have proposed the Wasserstein Distance as an alternative to IoU for
measuring the similarity of Gaussian-distributed bounding boxes. However, we
have observed that the Wasserstein Distance lacks scale invariance, which
negatively impacts the model's generalization capability. Additionally, when
used as a loss function, its independent optimization of the center attributes
leads to slow model convergence and unsatisfactory detection precision. To
address these challenges, we introduce the Gaussian Combined Distance (GCD).
Through analytical examination of GCD and its gradient, we demonstrate that GCD
not only possesses scale invariance but also facilitates joint optimization,
which enhances model localization performance. Extensive experiments on the
AI-TOD-v2 dataset for tiny object detection show that GCD, as a bounding box
regression loss function and label assignment metric, achieves state-of-the-art
performance across various detectors. We further validated the generalizability
of GCD on the MS-COCO-2017 and Visdrone-2019 datasets, where it outperforms the
Wasserstein Distance across diverse scales of datasets. Code is available at
https://github.com/MArKkwanGuan/mmdet-GCD.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Ziqian Guan等人撰写的论文“Gaussian Combined Distance: A Generic Metric for Object Detection”的全面摘要。

---

**论文摘要：Gaussian Combined Distance: A Generic Metric for Object Detection**

**1. 主要问题或研究问题：**
目标检测中，边界框相似度度量（如IoU）对于小目标检测性能不佳，主要原因在于其对微小位置偏差的敏感性，以及在无重叠时梯度消失的问题。尽管Wasserstein Distance（WD）被提出作为替代方案，但它缺乏尺度不变性，影响了模型的泛化能力。此外，WD作为损失函数时，其对中心属性的独立优化导致模型收敛缓慢和检测精度不足。因此，核心问题是开发一种既能有效处理小目标，又具备尺度不变性和联合优化特性的通用相似度度量。

**2. 关键创新或方法论贡献：**
本文提出了**高斯组合距离（Gaussian Combined Distance, GCD）**，旨在解决现有相似度度量的局限性。其主要创新点包括：
*   **尺度不变性：** GCD通过分析其梯度，被证明具有尺度不变性，这对于处理不同尺寸的目标至关重要，尤其是在小目标检测中。这解决了Wasserstein Distance缺乏尺度不变性的问题。
*   **联合优化特性：** GCD的梯度分析表明，它能够促进边界框中心、宽度和高度属性的联合优化。与WD独立优化中心不同，GCD通过动态调整梯度，对更精确对齐的边界框（特别是小目标）施加更大的梯度增益，从而提高定位精度和收敛速度。
*   **通用性：** GCD不仅作为边界框回归损失函数，还可作为标签分配度量，在多种检测器中表现出优异性能。
*   **非线性转换：** 为了将GCD转换为更精细和富有表现力的相似度度量，论文采用了非线性指数转换（`Mgcd = exp(-Dgc(Np, Nt))`），使其满足通用度量的所有标准（仿射不变性、对称性、可微分性、平滑边界处理）。

**3. 主要结果及其意义：**
*   **小目标检测的SOTA性能：** 在AI-TOD-v2微小目标检测数据集上的广泛实验表明，GCD作为边界框回归损失函数和标签分配度量，在各种检测器上均实现了最先进（SOTA）的性能。
*   **优于Wasserstein Distance：** GCD在AI-TOD-v2、MS-COCO-2017和Visdrone-2019等数据集上，在不同尺度的数据集上均优于Wasserstein Distance，尤其是在WD和NWD性能因尺度不变性问题下降的标准基准数据集上，GCD仍能保持与IoU相当的性能。
*   **提升模型定位性能：** GCD的联合优化特性显著提高了小目标检测的精度，解决了现有方法在小目标上定位不准确的问题。
*   **泛化能力强：** 实验验证了GCD在Visdrone-2019和MS-COCO-2017等通用数据集上的泛化能力，证明了其在不同尺度数据集上的鲁棒性。

**4. 论文中提及的局限性：**
论文中并未明确提及GCD本身的局限性。相反，它主要强调了现有方法（如IoU、Wasserstein Distance和NWD）的局限性，并展示了GCD如何克服这些问题。例如：
*   IoU在无重叠时梯度消失。
*   Wasserstein Distance缺乏尺度不变性，且独立优化中心属性导致收敛慢和精度不足。
*   NWD在通用数据集上的性能不一致。

**5. 潜在的未来研究方向：**
*   **旋转目标检测：** 论文指出，Kullback-Leibler Divergence（KLD）的联合优化特性已被证明能有效提升旋转目标检测性能。鉴于GCD具有相似的特性，作者推测GCD在旋转目标检测中，通过最少的配置调整，也能提供独特的优势。这暗示了将GCD应用于旋转边界框回归的潜力。
*   **更广泛的应用场景：** 作为一个“通用度量”，GCD未来可能被探索应用于其他需要精确相似度度量的计算机视觉任务，而不仅仅局限于目标检测。

---

总而言之，这篇论文通过引入高斯组合距离（GCD），成功地解决或显著缓解了现有边界框相似度度量在小目标检测和泛化能力方面的关键挑战。GCD的尺度不变性和联合优化特性使其成为目标检测领域一个有前景且强大的新工具。

**Key Findings:**

- To
address these challenges, we introduce the Gaussian Combined Distance (GCD).
- Through analytical examination of GCD and its gradient, we demonstrate that GCD
not only possesses scale invariance but also facilitates joint optimization,
which enhances model localization performance.
- Extensive experiments on the
AI-TOD-v2 dataset for tiny object detection show that GCD, as a bounding box
regression loss function and label assignment metric, achieves state-of-the-art
performance across various detectors.
- We further validated the generalizability
of GCD on the MS-COCO-2017 and Visdrone-2019 datasets, where it outperforms the
Wasserstein Distance across diverse scales of datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27649v1)
- [arXiv](https://arxiv.org/abs/2510.27649v1)

---

<a id='2510.27632v1'></a>
## [Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation](https://arxiv.org/abs/2510.27632v1)

**Authors:** Riccardo Brioschi, Aleksandr Alekseev, Emanuele Nevali, Berkay Döner, Omar El Malki, Blagoj Mitrevski, Leandro Kieliger, Mark Collier, Andrii Maksai, Jesse Berent, Claudiu Musat, Efi Kokiopoulou

**Published:** 2025-10-31

**Categories:** cs.CV, cs.AI

**Abstract:**

Graphic layout generation is a growing research area focusing on generating
aesthetically pleasing layouts ranging from poster designs to documents. While
recent research has explored ways to incorporate user constraints to guide the
layout generation, these constraints often require complex specifications which
reduce usability. We introduce an innovative approach exploiting user-provided
sketches as intuitive constraints and we demonstrate empirically the
effectiveness of this new guidance method, establishing the sketch-to-layout
problem as a promising research direction, which is currently under-explored.
To tackle the sketch-to-layout problem, we propose a multimodal
transformer-based solution using the sketch and the content assets as inputs to
produce high quality layouts. Since collecting sketch training data from human
annotators to train our model is very costly, we introduce a novel and
efficient method to synthetically generate training sketches at scale. We train
and evaluate our model on three publicly available datasets: PubLayNet,
DocLayNet and SlidesVQA, demonstrating that it outperforms state-of-the-art
constraint-based methods, while offering a more intuitive design experience. In
order to facilitate future sketch-to-layout research, we release O(200k)
synthetically-generated sketches for the public datasets above. The datasets
are available at https://github.com/google-deepmind/sketch_to_layout.

**Analysis:**

好的，这是一篇关于“Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation”论文的全面摘要：

**论文摘要：Sketch-to-Layout：草图引导的多模态布局生成**

**1. 主要问题或研究问题：**
该论文旨在解决图形布局生成领域的一个核心挑战：如何以更直观、更用户友好的方式引导布局生成，以创建美观且实用的多模态布局（包含图像和文本）。现有方法通常依赖于复杂且降低可用性的约束规范。因此，论文提出了“草图到布局”问题，即利用用户提供的草图作为直观约束来指导布局生成。

**2. 关键创新或方法论贡献：**
*   **草图作为直观引导方法：** 论文实证证明了草图作为一种新的布局生成引导方法的有效性，优于其他基于文本约束的方法，且所需时间更少。这确立了“草图到布局”作为一个有前景但尚未充分探索的研究方向。
*   **多模态Transformer解决方案：** 为解决草图到布局问题，作者提出了一种基于多模态Transformer的解决方案，该方案以草图和内容资产（图像和文本）作为输入，生成高质量的布局。
*   **大规模合成草图生成方法：** 鉴于人工标注草图训练数据成本高昂且耗时，论文引入了一种新颖高效的方法，可以大规模合成训练草图。该方法通过收集少量手绘图元，并根据布局元素的属性（如宽度、长宽比、字体大小等）将这些图元组合成完整的合成草图。
*   **内容排序分数（COS）：** 引入了一个新的度量标准COS，用于评估生成布局的内容感知能力，特别是其阅读顺序和叙事流畅性。

**3. 主要结果及其意义：**
*   **性能显著提升：** 该模型在PubLayNet、DocLayNet和SlidesVQA三个公开数据集上进行了训练和评估，结果显示其在最大IoU方面比最先进的基于约束的方法高出40%以上。
*   **合成草图的有效性：** 模型在合成草图和人工草图上的表现相当，这验证了合成草图作为VLM训练数据的可靠性。
*   **内容感知的重要性：** 实验结果强调了内容感知（即模型能够处理图像和文本内容）对于提高布局生成性能的重要性。
*   **部分草图的鲁棒性：** 即使在部分草图（覆盖率不同）的情况下，模型也能表现良好，且覆盖率越高，性能越好，这表明模型在信息不完整时仍具有创造潜力。
*   **数据发布：** 为了促进未来的草图到布局研究，论文发布了约20万张针对上述公开数据集合成生成的草图。

**4. 论文中提到的局限性：**
*   **多图像理解的挑战：** 在随机图像设置中，移除图像内容不一定会导致性能下降，这部分是由于数据集中大多数示例只包含一张图像。通过短时间微调来理解多张不相关图像对PaLIGemma模型来说仍然是一个难题。
*   **草图作为约束的非单调性：** 在某些数据集（如PubLayNet和DocLayNet）上，草图覆盖率的增加并不总是与性能的单调提升相关。

**5. 潜在的未来研究方向：**
*   **更复杂的草图图元：** 可以添加更复杂的草图图元，以进一步引导模型。
*   **跨领域和资产类型泛化：** 将该方法应用于生成各种领域和资产类型的草图。
*   **训练更大、更强大的模型：** 利用更大、更强大的模型来达到生产级别的性能。

总而言之，这篇论文通过引入草图作为直观约束和创新的合成草图生成方法，为多模态布局生成领域做出了重要贡献，并为未来的研究开辟了新的方向。

**Key Findings:**

- We introduce an innovative approach exploiting user-provided
sketches as intuitive constraints and we demonstrate empirically the
effectiveness of this new guidance method, establishing the sketch-to-layout
problem as a promising research direction, which is currently under-explored.
- To tackle the sketch-to-layout problem, we propose a multimodal
transformer-based solution using the sketch and the content assets as inputs to
produce high quality layouts.
- Since collecting sketch training data from human
annotators to train our model is very costly, we introduce a novel and
efficient method to synthetically generate training sketches at scale.
- We train
and evaluate our model on three publicly available datasets: PubLayNet,
DocLayNet and SlidesVQA, demonstrating that it outperforms state-of-the-art
constraint-based methods, while offering a more intuitive design experience.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27632v1)
- [arXiv](https://arxiv.org/abs/2510.27632v1)

---

<a id='2510.27547v1'></a>
## [MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series](https://arxiv.org/abs/2510.27547v1)

**Authors:** Xue Xia, Randall Balestriero, Tao Zhang, Yixin Zhou, Andrew Ding, Dev Saini, Lorenz Hurni

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

Historical maps are unique and valuable archives that document geographic
features across different time periods. However, automated analysis of
historical map images remains a significant challenge due to their wide
stylistic variability and the scarcity of annotated training data. Constructing
linked spatio-temporal datasets from historical map time series is even more
time-consuming and labor-intensive, as it requires synthesizing information
from multiple maps. Such datasets are essential for applications such as dating
buildings, analyzing the development of road networks and settlements, studying
environmental changes etc. We present MapSAM2, a unified framework for
automatically segmenting both historical map images and time series. Built on a
visual foundation model, MapSAM2 adapts to diverse segmentation tasks with
few-shot fine-tuning. Our key innovation is to treat both historical map images
and time series as videos. For images, we process a set of tiles as a video,
enabling the memory attention mechanism to incorporate contextual cues from
similar tiles, leading to improved geometric accuracy, particularly for areal
features. For time series, we introduce the annotated Siegfried Building Time
Series Dataset and, to reduce annotation costs, propose generating pseudo time
series from single-year maps by simulating common temporal transformations.
Experimental results show that MapSAM2 learns temporal associations effectively
and can accurately segment and link buildings in time series under limited
supervision or using pseudo videos. We will release both our dataset and code
to support future research.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Xue Xia等人撰写的论文“MapSAM2: Adapting SAM2 for Automatic Segmentation of Historical Map Images and Time Series”的全面摘要。

---

**论文摘要：MapSAM2：适应SAM2用于历史地图图像和时间序列的自动分割**

**1. 主要问题或研究问题：**
该论文旨在解决历史地图图像和时间序列的自动分析所面临的重大挑战。历史地图因其风格多样性和标注训练数据稀缺而难以进行自动化分析。更具挑战性的是，从历史地图时间序列构建链接的时空数据集（例如，用于建筑年代测定、道路网络发展分析、环境变化研究等）需要从多张地图中综合信息，这既耗时又劳动密集。因此，核心问题是如何开发一个统一的框架，能够有效地自动分割历史地图图像和时间序列，尤其是在数据稀缺和风格多变的环境下。

**2. 关键创新或方法贡献：**
MapSAM2是一个统一的框架，用于自动分割历史地图图像和时间序列，其关键创新和方法贡献如下：

*   **将历史地图图像和时间序列视为视频：** 这是MapSAM2的核心创新。对于时间序列数据，它直接将数据视为视频，利用SAM2强大的视频分割泛化能力。对于单个历史地图图像，MapSAM2将一组图像瓦片（tiles）作为一个伪视频序列进行处理，从而激活记忆注意力机制，使其能够从相似瓦片中整合上下文线索，提高几何精度，尤其对区域特征有效。
*   **基于视觉基础模型SAM2进行适应：** MapSAM2构建在SAM2（Segment Anything Model 2）这一视觉基础模型之上，并通过少量样本微调（few-shot fine-tuning）适应不同的分割任务。
*   **引入LoRA（Low-Rank Adaptation）进行高效微调：** 考虑到历史地图与自然图像之间的领域差距以及全量微调的计算成本和泛化能力下降问题，MapSAM2采用LoRA对图像编码器进行高效微调，冻结预训练权重，只更新低秩矩阵，从而在保持性能的同时降低计算开销。
*   **图像分割的无提示方法：** 对于历史地图图像的语义分割，MapSAM2无需外部提示，而是通过微调掩码解码器中固有的默认查询令牌来执行自动分割。
*   **时间序列的YOLO驱动提示：** 对于需要实例级分割和链接的时间序列数据，MapSAM2集成了YOLO检测器来自动生成边界框提示。
*   **伪时间序列生成以减少标注成本：** 针对视频格式训练数据稀缺且标注成本高昂的问题，MapSAM2提出通过模拟常见的时空变换（如对象位移、出现、消失和合并）从单年地图生成伪时间序列。这使得在仅有图像级标注的情况下，也能有效训练视频分割模型。
*   **发布Siegfried建筑时间序列数据集：** 论文引入并发布了标注的Siegfried建筑时间序列数据集，包含2000多个视频，每个视频包含四个历史时间戳的地图，以支持未来的研究。

**3. 主要结果及其意义：**
*   **图像分割性能：** MapSAM2在历史地图图像分割基准测试中（如铁路、葡萄园和建筑区块检测）表现优异，尤其在区域特征（如葡萄园和建筑区块）的分割上超越了现有最先进的方法，包括在全量和少量样本训练条件下。在葡萄园数据集上，MapSAM2的IoU达到77.3，略优于U-Net（77.0），并显著优于MapSAM。
*   **记忆注意力机制的有效性：** 消融研究表明，移除记忆注意力会导致性能显著下降。在10-shot训练设置下，记忆注意力将铁路的分割精度提高了16.1% IoU，葡萄园提高了14.3% IoU，这表明记忆注意力显著增强了MapSAM2利用上下文线索的能力。
*   **时间序列分割性能：** MapSAM2在有限监督（10-shot）下表现出强大的鲁棒性，在F1分数上分别比Mask2Former-VIS和Mask R-CNN+Link高出35.8%和15.7%。
*   **伪视频的有效性：** 伪时间序列数据集在全量伪数据集上实现了83.1的F1分数，在10-shot设置下实现了71.1的F1分数，与真实时间序列数据集上获得的结果相当，证明了伪视频在训练视频分割模型方面的实用性。
*   **提示质量的影响：** 提高YOLO检测器的训练数据量（从10-shot到全量）可以显著提升MapSAM2的性能，例如，使用全量数据训练的YOLO提示将MapSAM2的F1分数提高了12.8%，表明高质量提示对于低资源微调至关重要。

这些结果表明，MapSAM2能够有效地学习时间关联，并在有限监督或使用伪视频的情况下，准确地分割和链接时间序列中的建筑物，显著提高了自动化程度和准确性。

**4. 论文中提及的局限性：**
*   **线性特征分割性能：** MapSAM2在铁路等线性特征上的表现相对一般，在低数据量场景下（1%和10-shot）略低于MapSAM。这可能与注意力机制作为低通滤波器，更强调低频信息和全局上下文，而对铁路等狭窄、高频结构效果不佳有关。
*   **早期帧的缺失建筑：** 在实验设置中，提示仅在最新帧提供。因此，在早期帧中出现但在最新帧中未出现的建筑物会丢失其跟踪。

**5. 潜在的未来研究方向：**
*   **更自动和启发式地处理早期帧提示：** 探索更自动和启发式的方法来处理早期帧中缺失的建筑物，例如通过交互式或自动匹配YOLO检测到的边界框来扩展提示集，同时避免泄露链接信息。
*   **进一步优化线性特征分割：** 针对线性特征分割的不足，可以研究如何改进模型以更好地处理高频结构，例如通过引入更适合捕捉精细细节的模块或调整注意力机制。
*   **探索更长的伪视频序列：** 尽管论文认为两帧伪视频足以训练视频分割模型，但未来研究可以探索更长的伪视频序列，以模拟更复杂的时空变化，并解决可能出现的歧义。
*   **将MapSAM2应用于更多历史地图分析任务：** 将MapSAM2应用于更广泛的历史地图分析任务，如道路网络演变、土地利用变化等，以进一步验证其泛化能力和实用性。

---

**Key Findings:**

- We present MapSAM2, a unified framework for
automatically segmenting both historical map images and time series.
- For time series, we introduce the annotated Siegfried Building Time
Series Dataset and, to reduce annotation costs, propose generating pseudo time
series from single-year maps by simulating common temporal transformations.
- Experimental results show that MapSAM2 learns temporal associations effectively
and can accurately segment and link buildings in time series under limited
supervision or using pseudo videos.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27547v1)
- [arXiv](https://arxiv.org/abs/2510.27547v1)

---

<a id='2510.27481v1'></a>
## [NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding](https://arxiv.org/abs/2510.27481v1)

**Authors:** Wei Xu, Cheng Wang, Dingkang Liang, Zongchuang Zhao, Xingyu Jiang, Peng Zhang, Xiang Bai

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

Underwater exploration offers critical insights into our planet and attracts
increasing attention for its broader applications in resource exploration,
national security, etc. We study the underwater scene understanding methods,
which aim to achieve automated underwater exploration. The underwater scene
understanding task demands multi-task perceptions from multiple granularities.
However, the absence of large-scale underwater multi-task instruction-tuning
datasets hinders the progress of this research. To bridge this gap, we
construct NautData, a dataset containing 1.45 M image-text pairs supporting
eight underwater scene understanding tasks. It enables the development and
thorough evaluation of the underwater scene understanding models. Underwater
image degradation is a widely recognized challenge that interferes with
underwater tasks. To improve the robustness of underwater scene understanding,
we introduce physical priors derived from underwater imaging models and propose
a plug-and-play vision feature enhancement (VFE) module, which explicitly
restores clear underwater information. We integrate this module into renowned
baselines LLaVA-1.5 and Qwen2.5-VL and build our underwater LMM, NAUTILUS.
Experiments conducted on the NautData and public underwater datasets
demonstrate the effectiveness of the VFE module, consistently improving the
performance of both baselines on the majority of supported tasks, thus ensuring
the superiority of NAUTILUS in the underwater scene understanding area. Data
and models are available at https://github.com/H-EmbodVis/NAUTILUS.

**Analysis:**

以下是Wei Xu, Cheng Wang, Dingkang Liang, Zongchuang Zhao, Xingyu Jiang, Peng Zhang, Xiang Bai撰写的论文“NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding”的全面摘要：

**1. 主要问题或研究问题**
该论文旨在解决水下场景理解的挑战，特别是由于缺乏大规模、多任务指令微调数据集而阻碍了水下场景理解方法的发展。此外，水下图像退化（如光散射和吸收导致能见度差、对比度低和颜色失真）是影响水下任务性能的普遍问题。

**2. 关键创新或方法论贡献**
*   **NautData数据集的构建：** 论文构建了一个名为NautData的大规模水下指令遵循数据集，包含145万图像-文本对，支持八种不同的水下场景理解任务（粗粒度分类、细粒度分类、计数、视觉问答、检测、接地、区域描述和图像描述）。这弥补了现有数据集在多粒度、多任务标注方面的不足。
*   **视觉特征增强（VFE）模块：** 论文提出了一种即插即用的VFE模块，该模块利用水下成像模型（包括暗像素先验和深度信息）的物理先验，显式地恢复清晰的水下信息，以应对图像退化问题。VFE模块通过去除反向散射和恢复光吸收来增强视觉特征。
*   **NAUTILUS模型的开发：** 论文将VFE模块集成到LLaVA-1.5和Qwen2.5-VL等知名基线中，构建了NAUTILUS，这是一个能够实现图像、区域和对象级别水下场景理解的大型多模态模型（LMM）。

**3. 主要结果及其意义**
*   **VFE模块的有效性：** 实验证明，VFE模块能够持续提升LLaVA-1.5和Qwen2.5-VL在大多数支持任务上的性能，从而确保NAUTILUS在水下场景理解领域的优越性。
*   **NAUTILUS的卓越性能：** NAUTILUS在NautData测试集上，在细粒度分类、图像描述、接地和检测等任务上表现最佳，在计数任务上也优于其他LMM，展示了其在水下场景理解方面的强大能力。
*   **泛化能力和鲁棒性：** 在MarineInst20M数据集上的零样本接地评估表明，NAUTILUS在不同领域和模型之间具有良好的泛化能力。在低光、绿染和浑浊等退化条件下的评估显示，NAUTILUS表现出卓越的鲁棒性，在挑战性条件下仍能保持显著性能提升。
*   **多粒度理解：** NAUTILUS能够响应用户指令，并输出多粒度的水下场景理解结果，包括图像、区域和对象级别的信息，这对于全面的水下知识共享和人机交互具有重要意义。

**4. 论文中提及的局限性**
*   **开放词汇和少样本学习的挑战：** 论文指出，水下环境和物种的巨大多样性对现有数据集中的所有相关类别和场景的详尽表示提出了重大挑战。因此，水下场景理解算法需要具备开放词汇或少样本学习能力，以有效泛化到新颖和未见过的案例，而这在当前工作中尚未得到充分探索。

**5. 潜在的未来研究方向**
*   未来的研究可以探索如何赋予水下场景理解算法开放词汇和少样本学习能力，以更好地应对水下环境的巨大多样性和复杂性。
*   进一步优化多任务学习策略，以解决在某些任务（如计数任务的准确性）中可能出现的性能下降问题。
*   探索更先进的物理先验或学习方法，以进一步提升VFE模块在极端水下退化条件下的性能。

**Key Findings:**

- To improve the robustness of underwater scene understanding,
we introduce physical priors derived from underwater imaging models and propose
a plug-and-play vision feature enhancement (VFE) module, which explicitly
restores clear underwater information.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27481v1)
- [arXiv](https://arxiv.org/abs/2510.27481v1)

---

<a id='2510.27350v1'></a>
## [RzenEmbed: Towards Comprehensive Multimodal Retrieval](https://arxiv.org/abs/2510.27350v1)

**Authors:** Weijian Jian, Yajun Zhang, Dawei Liang, Chunyu Xie, Yixiao He, Dawei Leng, Yuhui Yin

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

The rapid advancement of Multimodal Large Language Models (MLLMs) has
extended CLIP-based frameworks to produce powerful, universal embeddings for
retrieval tasks. However, existing methods primarily focus on natural images,
offering limited support for other crucial visual modalities such as videos and
visual documents. To bridge this gap, we introduce RzenEmbed, a unified
framework to learn embeddings across a diverse set of modalities, including
text, images, videos, and visual documents. We employ a novel two-stage
training strategy to learn discriminative representations. The first stage
focuses on foundational text and multimodal retrieval. In the second stage, we
introduce an improved InfoNCE loss, incorporating two key enhancements.
Firstly, a hardness-weighted mechanism guides the model to prioritize
challenging samples by assigning them higher weights within each batch.
Secondly, we implement an approach to mitigate the impact of false negatives
and alleviate data noise. This strategy not only enhances the model's
discriminative power but also improves its instruction-following capabilities.
We further boost performance with learnable temperature parameter and model
souping. RzenEmbed sets a new state-of-the-art on the MMEB benchmark. It not
only achieves the best overall score but also outperforms all prior work on the
challenging video and visual document retrieval tasks. Our models are available
in https://huggingface.co/qihoo360/RzenEmbed.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Weijian Jian等人撰写的论文“RzenEmbed: Towards Comprehensive Multimodal Retrieval”的全面摘要。

---

### RzenEmbed: 面向综合多模态检索的全面摘要

**1. 主要问题或研究问题：**
当前的多模态大语言模型（MLLMs）和基于CLIP的框架在生成通用嵌入方面取得了显著进展，但它们主要关注自然图像，对视频和视觉文档等关键视觉模态的支持有限。这限制了它们在真实世界应用中作为通用检索系统的能力。因此，本文旨在解决如何构建一个统一的框架，以学习跨文本、图像、视频和视觉文档等多种模态的判别性嵌入，从而实现更全面的多模态检索。

**2. 关键创新或方法论贡献：**
RzenEmbed引入了一个新颖的两阶段训练策略和多项技术改进：
*   **统一框架和多模态支持：** RzenEmbed是一个统一的框架，能够学习跨文本、图像、视频和视觉文档的嵌入，弥补了现有方法在视频和视觉文档支持上的不足。它采用Qwen2-VL作为骨干模型，利用其动态分辨率、多模态旋转位置嵌入（M-RoPE）和强大的泛化能力来处理异构多模态数据。
*   **两阶段训练策略：**
    *   **第一阶段（多模态持续预训练）：** 专注于建立基础性的文本和多模态检索能力，通过使用多样化的单模态、跨模态和融合模态数据集（如MS-MARCO、LAION-2B、ShareGPT4V、MegaPairs）来对齐不同模态的表示。此阶段还通过CogVLM-19B对LAION-2B图像进行详细的重新标注，以增强模型对长文本的理解和细粒度语义对齐，并进行严格的数据清洗。
    *   **第二阶段（微调）：** 引入了多样化的指令格式数据，以全面提升模型处理专业场景和复杂任务的能力。特别地，通过合并图像分类数据集和增强视频数据（分段长视频、整合长篇视频）来增加任务难度和减少假阴性样本。
*   **改进的InfoNCE损失：**
    *   **硬度加权机制：** 通过为批次内具有挑战性的样本分配更高的权重，引导模型优先处理这些样本，从而增强模型的判别能力。
    *   **假阴性缓解：** 引入了一种策略来识别并排除训练批次中语义相似但被错误标记为负样本的实例，从而减轻假阴性对训练的负面影响，提高学习的稳定性和有效性。
*   **可学习温度参数：** 针对不同任务引入了任务特定的可学习温度参数（τt = exp(θt)），允许模型动态调整softmax概率分布的锐度，以适应不同任务的难度和样本分布。
*   **嵌入提示设计：** 采用系统提示和表示提示的组合，引导模型生成更适合判别性学习的表示，弥合生成式预训练和判别性微调之间的差距。
*   **模型融合（Model Souping）：** 针对LoRA适配器采用模型融合技术，将多个专业适配器合并为一个通用适配器，以捕获互补知识，提高性能并减少计算开销。

**3. 主要结果及其意义：**
*   **MMEB基准测试SOTA：** RzenEmbed在MMEB-V1和MMEB-V2基准测试上均取得了新的最先进（SOTA）结果。在2B和7B模型规模下，RzenEmbed均表现最佳，尤其在具有挑战性的视频和视觉文档检索任务上超越了所有现有工作。
*   **全面性能提升：** RzenEmbed不仅在整体得分上表现出色，还在MMEB-V1的Per Meta-Task和IND/OOD任务划分中实现了最佳性能，展示了其对各种任务和不同领域数据的卓越适应性和泛化能力。
*   **消融研究验证：** 消融研究证实了各项策略的有效性，包括合并分类数据集、可学习温度、系统提示和数据集重采样，每项都对模型性能有积极贡献。

**4. 论文中提到的局限性：**
*   **现有嵌入模型对视频和视觉文档支持有限：** 现有模型主要关注自然图像，在处理视频（时间段未对齐、噪声字幕）和视觉文档（结构模糊、布局敏感）时性能下降。
*   **标准对比学习的挑战：** InfoNCE损失在实践中存在局限性，如假阴性（语义相似样本被错误视为负样本）和易负样本的支配（模型将大部分学习能力分配给琐碎的区别，而忽略了信息量更大的硬负样本）。
*   **温度参数的固定性：** 传统的InfoNCE损失中温度参数通常是共享或固定的，无法适应不同任务的最佳尺度。
*   **文本提示设计不足：** 以前对生成一致且紧凑表示的系统性策略探索不足。

**5. 潜在的未来研究方向：**
论文中没有明确提出未来的研究方向，但从其解决的问题和贡献来看，可以推断出以下潜在方向：
*   **更复杂的视觉模态：** 进一步探索对更复杂和结构化的视觉模态（如3D数据、医学图像等）的支持。
*   **更高效的训练策略：** 优化多模态持续训练和微调策略，以处理更大规模、更多样化的数据，同时提高训练效率。
*   **自适应学习机制：** 探索更先进的自适应学习机制，例如更智能的硬负样本挖掘、动态调整损失权重等，以进一步提升模型在极端复杂场景下的性能。
*   **模型可解释性：** 深入研究RzenEmbed等MLLM在多模态检索任务中的决策过程，提高模型的可解释性和透明度。
*   **实际应用部署：** 将RzenEmbed应用于更广泛的实际场景，如AI代理、多模态搜索和推荐、检索增强生成（RAG）等，并解决实际部署中的挑战（如延迟、资源消耗）。

---

这份摘要旨在突出RzenEmbed在多模态检索领域的创新性和重要性，希望能满足您的要求。

**Key Findings:**

- To bridge this gap, we introduce RzenEmbed, a unified
framework to learn embeddings across a diverse set of modalities, including
text, images, videos, and visual documents.
- We employ a novel two-stage
training strategy to learn discriminative representations.
- RzenEmbed sets a new state-of-the-art on the MMEB benchmark.
- It not
only achieves the best overall score but also outperforms all prior work on the
challenging video and visual document retrieval tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27350v1)
- [arXiv](https://arxiv.org/abs/2510.27350v1)

---

<a id='2510.27335v1'></a>
## [Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing](https://arxiv.org/abs/2510.27335v1)

**Authors:** Yijia Wang, Yiqing Shen, Weiming Chen, Zhihai He

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

Existing image editing methods can handle simple editing instructions very
well. To deal with complex editing instructions, they often need to jointly
fine-tune the large language models (LLMs) and diffusion models (DMs), which
involves very high computational complexity and training cost. To address this
issue, we propose a new method, called \textbf{C}omplex \textbf{I}mage
\textbf{E}diting via \textbf{L}LM \textbf{R}easoning (CIELR), which converts a
complex user instruction into a set of simple and explicit editing actions,
eliminating the need for jointly fine-tuning the large language models and
diffusion models. Specifically, we first construct a structured semantic
representation of the input image using foundation models. Then, we introduce
an iterative update mechanism that can progressively refine this
representation, obtaining a fine-grained visual representation of the image
scene. This allows us to perform complex and flexible image editing tasks.
Extensive experiments on the SmartEdit Reasoning Scenario Set show that our
method surpasses the previous state-of-the-art by 9.955 dB in PSNR, indicating
its superior preservation of regions that should remain consistent. Due to the
limited number of samples of public datasets of complex image editing with
reasoning, we construct a benchmark named CIEBench, containing 86 image
samples, together with a metric specifically for reasoning-based image editing.
CIELR also outperforms previous methods on this benchmark. The code and dataset
are available at
\href{https://github.com/Jia-shao/Reasoning-Editing}{https://github.com/Jia-shao/Reasoning-Editing}.

**Analysis:**

好的，这是一篇关于Yijia Wang等人撰写的论文“Understanding the Implicit User Intention via Reasoning with Large Language Model for Image Editing”的全面摘要，内容基于您提供的PDF。

---

**论文摘要：通过大型语言模型推理理解隐式用户意图进行图像编辑**

**1. 主要问题或研究问题：**
现有图像编辑方法在处理简单指令时表现良好，但对于需要复杂推理和世界知识的复杂编辑指令（例如“将含维生素C最多的食物替换为橙子”）则力不从心。传统方法通常需要手动定义掩码，或者需要联合微调大型语言模型（LLMs）和扩散模型（DMs），这导致计算复杂度和训练成本极高。因此，该研究旨在解决如何在不进行昂贵联合微调的情况下，使图像编辑系统能够理解并执行复杂的、需要多步推理的隐式用户指令。

**2. 关键创新或方法论贡献：**
论文提出了一种名为**C**omplex **I**mage **E**diting via **L**LM **R**easoning (CIELR) 的新方法，其核心创新在于：
*   **解耦推理与编辑：** CIELR通过引入图像的结构化语义表示作为可解释的中间层，将推理过程与编辑执行过程解耦。这消除了对LLMs和DMs进行联合微调的需求，从而大大降低了计算复杂度和训练成本。
*   **结构化语义表示：** 首先利用基础模型（如SAM2、OWLv2、DepthAnything）构建图像的结构化语义表示，包含空间（分割掩码）、语义（对象标签）和深度信息。这种表示为LLMs提供了全面而抽象的视觉内容理解。
*   **迭代更新机制：** 引入了一个迭代更新机制，可以逐步完善结构化语义表示。当推理过程中识别出信息空白时，系统会动态更新语义表示，添加额外的空间或语义细节，直到获得足够的信息来完成推理。这使得CIELR能够处理需要多步推理的复杂查询。
*   **零样本操作：** 结构化语义表示的构建和推理模块均以零样本方式运行，实现了LLM-agnostic和DM-agnostic，能够无缝集成最先进的基础模型而无需昂贵再训练。
*   **新基准数据集和评估指标：** 针对推理型图像编辑任务，构建了名为CIEBench的新基准数据集（包含86个图像样本），并提出了专门的评估指标——图像差异检查分数（Image Difference Check Score, IDCS），用于评估语义正确性而非仅仅视觉相似性。

**3. 主要结果及其意义：**
*   **卓越的性能提升：** 在SmartEdit推理场景数据集上，CIELR方法在PSNR方面超越了现有最先进方法9.955 dB，表明其在保持未修改区域一致性方面表现出色。
*   **在CIEBench上的优越性：** 在新构建的CIEBench数据集上，CIELR在所有评估指标（包括PSNR、SSIM、LPIPS和IDCS）上均优于现有方法，尤其是在结合Inpaint Anything时表现最佳。IDCS得分的显著提高验证了其在处理复杂隐式查询方面的卓越推理能力。
*   **零样本能力：** CIELR在SmartEdit推理场景数据集上实现了区域保留指标的改进，且无需任何微调，而所有对比方法都经过了特定训练。这证明了CIELR在不依赖昂贵训练的情况下，能够实现高质量的区域特定编辑。
*   **对MagicBrush数据集的泛化性：** 即使在主要包含显式编辑指令的MagicBrush测试集上，CIELR在像素级准确性（L1、L2）和语义/特征级相似性（CLIP-I、DINO）指标上也优于所有竞争对手，包括经过MagicBrush训练的InstructPix2Pix。

**4. 论文中提及的局限性：**
论文中没有明确指出CIELR方法的具体局限性。然而，它提到了“由于公共数据集中带有推理的复杂图像编辑样本数量有限”，这促使他们构建了CIEBench数据集。这暗示了现有数据集在评估推理能力方面的不足，可能间接反映了复杂推理任务本身的挑战性。此外，虽然CIELR的完整模型会带来额外的计算成本（每张图像约6.5秒），但论文认为质量改进证明了这一适度处理时间的增加是合理的。

**5. 潜在的未来研究方向：**
论文指出，未来的研究可以探索将CIELR方法扩展到**视频推理编辑**和**交互式创意工作流**。这表明该框架的解耦和零样本特性使其具有进一步应用于更动态和交互式视觉任务的潜力。

---

**Key Findings:**

- To address this
issue, we propose a new method, called \textbf{C}omplex \textbf{I}mage
\textbf{E}diting via \textbf{L}LM \textbf{R}easoning (CIELR), which converts a
complex user instruction into a set of simple and explicit editing actions,
eliminating the need for jointly fine-tuning the large language models and
diffusion models.
- Then, we introduce
an iterative update mechanism that can progressively refine this
representation, obtaining a fine-grained visual representation of the image
scene.
- Extensive experiments on the SmartEdit Reasoning Scenario Set show that our
method surpasses the previous state-of-the-art by 9.955 dB in PSNR, indicating
its superior preservation of regions that should remain consistent.
- CIELR also outperforms previous methods on this benchmark.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27335v1)
- [arXiv](https://arxiv.org/abs/2510.27335v1)

---

<a id='2510.27677v1'></a>
## [Vision Transformer for Robust Occluded Person Reidentification in Complex Surveillance Scenes](https://arxiv.org/abs/2510.27677v1)

**Authors:** Bo Li, Duyuan Zheng, Xinyang Liu, Qingwen Li, Hong Li, Hongyan Cui, Ge Gao, Chen Liu

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

Person re-identification (ReID) in surveillance is challenged by occlusion,
viewpoint distortion, and poor image quality. Most existing methods rely on
complex modules or perform well only on clear frontal images. We propose Sh-ViT
(Shuffling Vision Transformer), a lightweight and robust model for occluded
person ReID. Built on ViT-Base, Sh-ViT introduces three components: First, a
Shuffle module in the final Transformer layer to break spatial correlations and
enhance robustness to occlusion and blur; Second, scenario-adapted augmentation
(geometric transforms, erasing, blur, and color adjustment) to simulate
surveillance conditions; Third, DeiT-based knowledge distillation to improve
learning with limited labels.To support real-world evaluation, we construct the
MyTT dataset, containing over 10,000 pedestrians and 30,000+ images from base
station inspections, with frequent equipment occlusion and camera variations.
Experiments show that Sh-ViT achieves 83.2% Rank-1 and 80.1% mAP on MyTT,
outperforming CNN and ViT baselines, and 94.6% Rank-1 and 87.5% mAP on
Market1501, surpassing state-of-the-art methods.In summary, Sh-ViT improves
robustness to occlusion and blur without external modules, offering a practical
solution for surveillance-based personnel monitoring.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Bo Li等人撰写的论文“Vision Transformer for Robust Occluded Person Reidentification in Complex Surveillance Scenes”的全面摘要。

---

### 论文摘要：Vision Transformer for Robust Occluded Person Reidentification in Complex Surveillance Scenes

**1. 主要问题或研究问题：**
该论文旨在解决复杂监控场景下行人重识别（Person Re-identification, ReID）面临的挑战。这些挑战主要包括：遮挡（如设备或人员遮挡）、视角畸变（如高角度或倾斜摄像头导致的几何畸变）以及图像质量不佳（如光照不足、模糊）。现有方法往往依赖复杂模块或仅在清晰正面图像上表现良好，难以应对真实世界监控场景的复杂性和鲁棒性需求。

**2. 关键创新或方法论贡献：**
为了解决上述问题，作者提出了 **Sh-ViT (Shuffling Vision Transformer)**，一个轻量级且鲁棒的遮挡行人ReID模型。Sh-ViT在ViT-Base骨干网络的基础上引入了三个关键创新：

*   **Shuffle 模块：** 在Transformer的最后一层嵌入Shuffle模块，通过随机重排特征令牌来打破空间相关性，并注入高斯噪声和零掩码。这种设计迫使模型学习空间不变的全局特征，从而增强对遮挡和模糊的鲁棒性，而无需显式遮挡检测或重建。
*   **场景自适应数据增强：** 针对监控场景的特点，设计了一套定制的数据增强策略。这包括几何变换（仿射/透视变换）以模拟视角畸变，随机擦除以模拟遮挡，高斯模糊以处理运动模糊和失焦，以及颜色调整以应对光照变化。
*   **基于 DeiT 的知识蒸馏：** 采用DeiT（Data-efficient Image Transformers）的知识蒸馏技术，以提高在有限标注数据下的学习效率和模型稳定性。

此外，为了支持真实世界场景的评估，作者还构建了一个新的数据集 **MyTT**，其中包含超过10,000名行人、30,000多张来自基站巡检场景的图像，这些图像具有频繁的设备遮挡和摄像头变化等特点，弥补了现有公共数据集的不足。

**3. 主要结果及其意义：**
实验结果表明Sh-ViT在多个数据集上均取得了优异性能：

*   **在MyTT数据集上：** Sh-ViT达到了83.2%的Rank-1准确率和80.1%的mAP，显著优于传统的CNN基线（如ResNet50和OSNet）以及ViT基线（TransReID）。这验证了Sh-ViT在处理遮挡和低质量图像方面的卓越能力，特别是在基站巡检等复杂监控场景中的实用性。
*   **在Market1501数据集上：** Sh-ViT取得了94.6%的Rank-1准确率和87.5%的mAP，超越了许多最新的SOTA方法（如DCReID和ACFL）。这表明Shuffle模块增强了模型的全局特征鲁棒性，使其能够适应不同的监控场景。
*   **在DukeMTMC-reID遮挡特定基准上：** Sh-ViT实现了89.6%的Rank-1和80.3%的mAP，再次超越了现有SOTA方法，证实了其在真实世界遮挡条件下的强大泛化能力。

消融研究进一步证实了Shuffle模块、场景自适应数据增强和SGD优化器的有效性。Shuffle模块通过打破局部相关性并强制Transformer学习空间不变特征，直接提升了性能，并显著稳定了mAP。场景自适应增强在数据稀缺时尤其有益，将Rank-1提高了8.2%。

**4. 论文中提及的局限性：**
尽管Sh-ViT表现出色，但论文也指出了一些局限性：

*   **增量收益：** 在某些指标上，Sh-ViT相对于TransReID的提升是增量的，表明仍有进一步改进的空间。
*   **失败案例：** 在重度设备遮挡（身体覆盖超过60%）或极端光照条件下，模型性能会显著下降。
*   **数据集范围：** MyTT数据集虽然多样，但主要关注基站环境；更广泛的遮挡基准评估可能提供更全面的评估。

**5. 潜在的未来研究方向：**
基于上述局限性，论文提出了以下未来研究方向：

*   **自适应令牌加权和光照增强：** 探索自适应令牌加权策略和更强的光照增强方法，以应对重度遮挡和极端光照。
*   **自适应Shuffle策略：** 研究根据遮挡模式动态调整置换强度的自适应Shuffle策略。
*   **时间建模：** 整合多帧序列中的时间线索，以增强视频ReID的跟踪稳定性，初步测试已显示出潜力。

---

这份摘要旨在全面概括论文的核心内容，突出其在解决复杂监控场景下行人重识别问题上的贡献和未来发展方向。

**Key Findings:**

- We propose Sh-ViT
(Shuffling Vision Transformer), a lightweight and robust model for occluded
person ReID.
- Experiments show that Sh-ViT achieves 83.2% Rank-1 and 80.1% mAP on MyTT,
outperforming CNN and ViT baselines, and 94.6% Rank-1 and 87.5% mAP on
Market1501, surpassing state-of-the-art methods.In summary, Sh-ViT improves
robustness to occlusion and blur without external modules, offering a practical
solution for surveillance-based personnel monitoring.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27677v1)
- [arXiv](https://arxiv.org/abs/2510.27677v1)

---

<a id='2510.27647v1'></a>
## [NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception](https://arxiv.org/abs/2510.27647v1)

**Authors:** Congzhang Shao, Quan Yuan, Guiyang Luo, Yue Hu, Danni Wang, Yilin Liu, Rui Pan, Bo Chen, Jinglin Li

**Published:** 2025-10-31

**Categories:** cs.CV

**Abstract:**

Collaborative perception improves task performance by expanding the
perception range through information sharing among agents. . Immutable
heterogeneity poses a significant challenge in collaborative perception, as
participating agents may employ different and fixed perception models. This
leads to domain gaps in the intermediate features shared among agents,
consequently degrading collaborative performance. Aligning the features of all
agents to a common representation can eliminate domain gaps with low training
cost. However, in existing methods, the common representation is designated as
the representation of a specific agent, making it difficult for agents with
significant domain discrepancies from this specific agent to achieve proper
alignment. This paper proposes NegoCollab, a heterogeneous collaboration method
based on the negotiated common representation. It introduces a negotiator
during training to derive the common representation from the local
representations of each modality's agent, effectively reducing the inherent
domain gap with the various local representations. In NegoCollab, the mutual
transformation of features between the local representation space and the
common representation space is achieved by a pair of sender and receiver. To
better align local representations to the common representation containing
multimodal information, we introduce structural alignment loss and pragmatic
alignment loss in addition to the distribution alignment loss to supervise the
training. This enables the knowledge in the common representation to be fully
distilled into the sender.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Congzhang Shao等人撰写的论文“NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception”的全面摘要。

---

**论文摘要：NegoCollab：一种异构协同感知中的通用表示协商方法**

**1. 主要问题或研究问题**
该论文旨在解决异构协同感知中的一个核心挑战：**不可变异构性（Immutable Heterogeneity）**。在协同感知中，不同智能体可能使用不同且固定的感知模型，导致它们共享的中间特征存在显著的**域间隙（Domain Gaps）**。这会严重影响协同性能。现有方法通常将通用表示指定为某个特定智能体的表示，这使得与该特定智能体存在较大域差异的智能体难以实现有效对齐。

**2. 关键创新或方法论贡献**
NegoCollab 提出了一种基于**协商式通用表示（Negotiated Common Representation）**的异构协同感知方法，其主要创新点包括：

*   **引入协商器（Negotiator）**：在训练阶段引入一个协商器，从每个模态智能体的局部表示中推导出通用表示。这有效地减少了通用表示与各种局部表示之间的固有域间隙，降低了对齐难度，并为异构协同感知中的通用表示提供了更灵活可靠的选择。
*   **发送器-接收器对（Sender-Receiver Pair）**：通过一对发送器和接收器，实现了局部表示空间与通用表示空间之间的特征相互转换，从而消除了域间隙。发送器将局部特征映射到通用表示空间进行共享，而接收器将接收到的特征投影回局部表示空间。
*   **多维对齐损失（Multi-dimensional Alignment Loss）**：为了更好地将局部表示与包含多模态信息的通用表示对齐，NegoCollab 在传统的**分布对齐损失（Distribution Alignment Loss）**之外，引入了**结构对齐损失（Structural Alignment Loss）**和**实用对齐损失（Pragmatic Alignment Loss）**来监督训练。这确保了通用表示中的知识能够充分地蒸馏到发送器中。
    *   **结构对齐损失**：确保场景组件之间的空间关系在不同表示之间保持一致。
    *   **实用对齐损失**：确保表示空间中前景信息的一致组织。

**3. 主要结果及其意义**
实验结果表明，NegoCollab 在基于通用表示的协同方法中显著优于现有方法。它在OPV2V-H、V2V4Real和DAIR-V2X等协同感知数据集上均取得了最先进的性能。

*   **性能提升**：NegoCollab 在所有测试条件下均表现出最佳性能，甚至在某些协同场景中优于一对一适应方法。
*   **域间隙减少**：通过协商器生成的通用表示与每个局部表示之间的域间隙显著减少，平均降低了约93.5%，这验证了协商器在减少域间隙方面的有效性。
*   **鲁棒性**：在定位误差（如高斯噪声）存在的情况下，NegoCollab 仍能保持卓越的性能，显示出其对噪声的鲁棒性。
*   **同构协同性能**：NegoCollab 在同构协同中也表现出色，甚至超越了原始的同构协同性能，这归因于多维对齐损失将多模态知识蒸馏到局部发送器中，增强了特征的表示能力。

**4. 论文中提及的局限性**
论文中提到NegoCollab的一个局限性是：
*   **通用表示的固定性**：一旦通用表示被协商确定，它就变得固定。将新智能体对齐到这个预先协商的通用表示不可避免地会导致信息损失。

**5. 潜在的未来研究方向**
针对上述局限性，论文提出了未来的研究方向：
*   **通用表示的泛化能力**：探索如何使通用表示更好地泛化到新智能体，以减少新智能体加入时可能发生的信息损失。

---

总而言之，NegoCollab通过引入一个创新的协商器和多维对齐损失，有效地解决了异构协同感知中的域间隙问题，实现了局部表示与通用表示之间的高效对齐。其卓越的性能和鲁棒性使其成为该领域的一个重要贡献，为未来异构协同感知系统的发展提供了新的思路。

**Key Findings:**

- To
better align local representations to the common representation containing
multimodal information, we introduce structural alignment loss and pragmatic
alignment loss in addition to the distribution alignment loss to supervise the
training.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27647v1)
- [arXiv](https://arxiv.org/abs/2510.27647v1)

---

<a id='2510.27607v1'></a>
## [Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](https://arxiv.org/abs/2510.27607v1)

**Authors:** John Won, Kyungmin Lee, Huiwon Jang, Dongyoung Kim, Jinwoo Shin

**Published:** 2025-10-31

**Categories:** cs.CV, cs.RO

**Abstract:**

Recently, augmenting Vision-Language-Action models (VLAs) with world modeling
has shown promise in improving robotic policy learning. However, it remains
challenging to jointly predict next-state observations and action sequences
because of the inherent difference between the two modalities. To address this,
we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework
that handles the modality conflict and enhances the performance of VLAs across
diverse tasks. Specifically, we propose a multimodal diffusion transformer
architecture that explicitly maintains separate modality streams while still
enabling cross-modal knowledge sharing. In addition, we introduce independent
noise perturbations for each modality and a decoupled flow-matching loss. This
design enables the model to learn the joint distribution in a bidirectional
manner while avoiding the need for a unified latent space. Based on the
decoupling of modalities during training, we also introduce a joint sampling
method that supports test-time scaling, where action and vision tokens evolve
asynchronously at different rates. Through experiments on simulated benchmarks
such as RoboCasa and GR-1, DUST achieves up to 6% gains over baseline methods,
while our test-time scaling approach provides an additional 2-5% boost. On
real-world tasks with the Franka Research 3, DUST improves success rates by
13%, confirming its effectiveness beyond simulation. Furthermore, pre-training
on action-free videos from BridgeV2 yields significant transfer gains on
RoboCasa, underscoring DUST's potential for large-scale VLA pretraining.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供John Won等人撰写的论文“Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model”的全面摘要。

---

### 论文摘要：Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model

**1. 主要问题或研究问题：**
该论文旨在解决增强型视觉-语言-动作（VLA）模型在机器人策略学习中面临的核心挑战。尽管世界建模（world modeling）已被证明能提升VLA性能，但同时预测下一状态观测（通常是高维、空间结构化的视觉信息）和动作序列（通常是低维、时间平滑的控制信号）仍然困难。这主要是因为这两种模态固有的统计特性差异，导致难以在统一的潜在空间中进行联合建模和双向知识传递。

**2. 关键创新或方法论贡献：**
为了解决上述问题，作者提出了**DUal-STream Diffusion (DUST)** 框架，其主要创新点包括：

*   **双流多模态扩散Transformer架构：** DUST采用一种独特的Transformer架构，明确为动作和视觉令牌维护独立的模态流（pathways）。尽管流是独立的，但它们通过共享的跨模态注意力层实现信息交互，从而在保持模态特异性的同时促进知识共享。这避免了对统一潜在空间的假设，并允许模型更好地处理两种模态的不同特性。
*   **解耦扩散训练算法：** DUST引入了一种解耦的训练方法，对每种模态（动作和未来视觉观测）应用独立的噪声扰动，并采用模态特定的流匹配损失。这种设计使得模型能够根据各自的统计结构学习动作和观测，同时捕捉它们之间的跨模态因果依赖关系。
*   **异步联合采样方法：** 在推理阶段，DUST提出了一种创新的联合采样策略，支持测试时缩放。该方法允许动作和视觉令牌以不同的速率演化（例如，视觉令牌可以进行更多去噪步骤以获得更高精度），从而在效率和准确性之间提供可调的权衡。

**3. 主要结果及其意义：**
DUST在多个基准测试中展现了显著的性能提升：

*   **模拟环境：** 在RoboCasa和GR-1等模拟基准测试中，DUST比基线方法（如GR00T-N1.5和FLARE）实现了高达6%的成功率提升。
*   **测试时缩放：** 异步联合采样方法额外带来了2-5%的性能提升，证明了其在推理效率和准确性之间进行权衡的有效性。
*   **真实世界任务：** 在使用Franka Research 3机械臂的真实世界抓取放置任务中，DUST的成功率提高了13%，验证了其在模拟环境之外的有效性。
*   **迁移学习：** 通过在BridgeV2的无动作视频上进行预训练，DUST在RoboCasa的下游微调任务中获得了显著的迁移增益。这表明DUST能够利用大规模被动视频数据进行高效的策略学习，突显了其在大规模VLA预训练方面的潜力。

这些结果共同证明了DUST在处理模态冲突、提升VLA性能以及在数据效率和泛化能力方面的强大能力。

**4. 论文中提及的局限性：**
论文中没有明确提及DUST框架的局限性。然而，从其设计和实验设置中可以推断出一些潜在的考虑：

*   **计算成本：** 尽管异步采样提供了测试时缩放，但多模态扩散Transformer架构，特别是当视觉令牌需要更多去噪步骤时，可能会增加推理时间，这在实时机器人应用中可能是一个考虑因素。
*   **超参数敏感性：** 损失权重超参数λ_wm的设置对性能有影响，论文指出在[0.5, 2.0]范围内性能稳定，但超出此范围会下降，这表明模型对该超参数可能存在一定的敏感性。
*   **VLM骨干的依赖：** DUST的性能部分依赖于预训练的VLM骨干（如Eagle-2和SIGLIP-2）提取的语义特征。这些VLM的局限性或偏差可能会间接影响DUST的性能。

**5. 潜在的未来研究方向：**
论文中没有明确列出未来研究方向，但从其贡献和讨论中可以推断出以下几点：

*   **更高效的异步采样策略：** 进一步探索和优化异步采样策略，以在推理速度和预测准确性之间找到更好的平衡，可能包括动态调整每种模态的去噪步数。
*   **扩展到更多模态：** DUST的双流架构原则可以扩展到集成更多模态，例如触觉反馈、声音信息等，以构建更全面的世界模型。
*   **更复杂的任务和泛化：** 探索DUST在更复杂、更开放的机器人任务中的应用，以及如何进一步提升其在未知环境和物体上的泛化能力。
*   **理论分析：** 对解耦噪声调度和双流架构在学习因果关系和联合分布方面的理论基础进行更深入的分析。
*   **大规模预训练的探索：** 进一步探索利用更大规模、更多样化的无动作视频数据集进行预训练，以期在下游机器人任务中实现更大的迁移增益和更强的泛化能力。

---

**Key Findings:**

- To address this,
we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework
that handles the modality conflict and enhances the performance of VLAs across
diverse tasks.
- Specifically, we propose a multimodal diffusion transformer
architecture that explicitly maintains separate modality streams while still
enabling cross-modal knowledge sharing.
- In addition, we introduce independent
noise perturbations for each modality and a decoupled flow-matching loss.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.27607v1)
- [arXiv](https://arxiv.org/abs/2510.27607v1)

---

