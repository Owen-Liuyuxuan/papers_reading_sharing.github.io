time: 20251222

# Arxiv Computer Vision Papers - 2025-12-22

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份关于2025年12月19日 Arxiv 计算机视觉领域论文的简明执行摘要。

---

**执行摘要：2025年12月19日 Arxiv 计算机视觉论文速览**

**日期：** 2025年12月19日

**主要主题与趋势：**

本期 Arxiv 论文涵盖了计算机视觉领域的多个前沿方向，其中尤为突出的是：

*   **生成模型与编辑的融合：** 多篇论文致力于提升文本到图像生成和编辑的质量与可控性，强调了表示编码器在语义理解和重建中的关键作用。
*   **多模态与跨模态学习：** 从摄像头生成雷达点云，以及利用视觉提示进行模型评估，都体现了多模态信息融合的重要性。
*   **模型鲁棒性与可解释性：** 对抗性鲁棒性、视觉模型在开放基础模型中的表现，以及无需训练即可实现模型自解释性的方法，是提升模型可靠性和透明度的重要研究方向。
*   **模拟到现实（Sim-to-Real）的挑战与解决方案：** 自动化任务和数据生成框架的出现，旨在弥合模拟环境与真实世界之间的差距。
*   **深度估计与场景理解的精进：** 通过自监督重光照等技术，进一步提升深度估计的精度和鲁棒性。

**亮点与创新：**

*   **“Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing”** 提出了一种新的表示编码器设计，显著提升了文本到图像生成和编辑的质量，是生成模型领域的重要进展。
*   **“Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting”** 引入了一种新颖的测试时深度精炼方法，利用自监督重光照技术，有望大幅提高现有深度估计模型的性能。
*   **“Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training”** 提出了一种无需额外训练即可使 Vision Transformers 具备自解释性的方法，为模型的可解释性研究开辟了新途径。
*   **“RadarGen: Automotive Radar Point Cloud Generation from Cameras”** 实现了从摄像头图像生成汽车雷达点云，为自动驾驶中的传感器融合和数据增强提供了新的解决方案。

**新兴研究方向与技术：**

*   **更精细的文本到图像控制：** 通过对表示编码器的优化和推理时空间对齐，实现更精确的图像生成和编辑。
*   **测试时模型自适应：** 利用自监督学习等技术，在测试阶段对模型进行优化，以适应特定场景或数据分布。
*   **无监督/自监督的可解释性方法：** 探索在不依赖标注数据的情况下，提升模型透明度和可信度的新技术。
*   **多模态数据生成与融合：** 跨越不同传感器模态（如视觉与雷达）的数据生成与融合，以应对更复杂的现实世界场景。
*   **自动化模拟环境与任务生成：** 为强化学习和机器人领域提供更高效、更具泛化性的训练数据和任务。

**建议阅读论文：**

考虑到其潜在影响和创新性，以下论文值得深入阅读：

1.  **“Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing”**：对于关注文本到图像生成和编辑的研究人员至关重要。
2.  **“Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting”**：对于深度估计和3D计算机视觉领域的研究人员具有重要价值。
3.  **“Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training”**：对于模型可解释性研究以及 Vision Transformer 的应用探索具有开创性意义。
4.  **“RadarGen: Automotive Radar Point Cloud Generation from Cameras”**：对于自动驾驶、传感器融合和多模态学习的研究人员具有直接的应用价值。

---

希望这份执行摘要能帮助您快速了解近期 Arxiv 计算机视觉领域的最新动态。

---

## Table of Contents

1. [Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing](#2512.17909v1)
2. [Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](#2512.17908v1)
3. [Dexterous World Models](#2512.17907v1)
4. [Adversarial Robustness of Vision in Open Foundation Models](#2512.17902v1)
5. [RadarGen: Automotive Radar Point Cloud Generation from Cameras](#2512.17897v1)
6. [Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training](#2512.17891v1)
7. [Visually Prompted Benchmarks Are Surprisingly Fragile](#2512.17875v1)
8. [InSPECT: Invariant Spectral Features Preservation of Diffusion Models](#2512.17873v1)
9. [AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning](#2512.17853v1)
10. [InfSplign: Inference-Time Spatial Alignment of Text-to-Image Diffusion Models](#2512.17851v1)

---

## Papers

<a id='2512.17909v1'></a>
## [Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing](https://arxiv.org/abs/2512.17909v1)

**Authors:** Shilong Zhang, He Zhang, Zhifei Zhang, Chongjian Ge, Shuchen Xue, Shaoteng Liu, Mengwei Ren, Soo Ye Kim, Yuqian Zhou, Qing Liu, Daniil Pakhomov, Kai Zhang, Zhe Lin, Ping Luo

**Published:** 2025-12-19

**Categories:** cs.CV

**Abstract:**

Modern Latent Diffusion Models (LDMs) typically operate in low-level Variational Autoencoder (VAE) latent spaces that are primarily optimized for pixel-level reconstruction. To unify vision generation and understanding, a burgeoning trend is to adopt high-dimensional features from representation encoders as generative latents. However, we empirically identify two fundamental obstacles in this paradigm: (1) the discriminative feature space lacks compact regularization, making diffusion models prone to off-manifold latents that lead to inaccurate object structures; and (2) the encoder's inherently weak pixel-level reconstruction hinders the generator from learning accurate fine-grained geometry and texture. In this paper, we propose a systematic framework to adapt understanding-oriented encoder features for generative tasks. We introduce a semantic-pixel reconstruction objective to regularize the latent space, enabling the compression of both semantic information and fine-grained details into a highly compact representation (96 channels with 16x16 spatial downsampling). This design ensures that the latent space remains semantically rich and achieves state-of-the-art image reconstruction, while remaining compact enough for accurate generation. Leveraging this representation, we design a unified Text-to-Image (T2I) and image editing model. Benchmarking against various feature spaces, we demonstrate that our approach achieves state-of-the-art reconstruction, faster convergence, and substantial performance gains in both T2I and editing tasks, validating that representation encoders can be effectively adapted into robust generative components.

**Analysis:**

好的，这是对论文“Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing”的全面中文摘要：

**论文题目：** Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing

**作者：** Shilong Zhang, He Zhang, Zhifei Zhang, Chongjian Ge, Shuchen Xue, Shaoteng Liu, Mengwei Ren, Soo Ye Kim, Yuqian Zhou, Qing Liu, Daniil Pakhomov, Kai Zhang, Zhe Lin, Ping Luo

**摘要：**

这篇论文旨在解决将强大的视觉理解编码器（representation encoders）应用于文本到图像生成和图像编辑任务时遇到的挑战。当前主流的生成模型（如 Latent Diffusion Models, LDMs）通常依赖于低维度的变分自编码器（VAE）的潜在空间，这些空间主要针对像素级重建进行优化。然而，直接使用高维度的、为理解任务设计的表示编码器的特征作为生成潜在空间存在两个主要问题：

1.  **缺乏紧凑的正则化：** 表示编码器的特征空间维度高但内在信息含量低，缺乏有效的正则化，导致扩散模型容易生成“离流形”（off-manifold）的潜在向量，从而产生结构不准确或扭曲的对象。
2.  **像素级重建能力弱：** 表示编码器本身通常不优化像素级重建，导致其生成的特征细节不足，难以让生成器学习到精确的几何形状和纹理。

为了克服这些障碍，论文提出了一种系统性的框架，将理解导向的编码器特征适配到生成任务中。

**核心创新与方法贡献：**

1.  **语义-像素重建目标：** 论文引入了一个创新的“语义-像素重建”（semantic-pixel reconstruction）目标。该目标旨在正则化潜在空间，将语义信息和精细的像素细节压缩到一个紧凑的表示中（例如，96通道，16x16空间下采样）。
2.  **像素-语义 VAE (PS-VAE)：** 论文设计了一个名为 PS-VAE 的模型。它首先通过一个语义 VAE（S-VAE）将高维度的、未正则化的表示编码器特征映射到一个紧凑的、KL 正则化的潜在空间。然后，通过解冻编码器并联合优化像素解码器和语义重建损失，进一步丰富该潜在空间，使其同时包含丰富的语义信息和高保真的像素细节。
3.  **统一的生成架构：** 基于 PS-VAE，论文设计了一个统一的文本到图像（T2I）和图像编辑模型。

**主要结果与意义：**

*   **卓越的重建性能：** PS-VAE 在图像重建任务上达到了最先进的性能，显著优于其他基于表示编码器的生成方法，并且在某些方面甚至超越了纯粹的 VAE 方法。
*   **更快的收敛速度与更优的生成性能：** 在文本到图像生成任务中，PS-VAE 表现出更快的收敛速度和更高的最终性能（GenEval 和 DPG-Bench 分数）。
*   **显著提升的图像编辑能力：** 在需要精确理解和执行指令的图像编辑任务中，PS-VAE 取得了大幅度的性能提升，显著优于 RAE 等基线模型。这表明其语义结构和高保真细节的结合对于理解复杂指令至关重要。
*   **统一的编码器潜力：** 论文证明了经过 PS-VAE 优化的表示编码器可以作为理解和生成任务的统一编码器，无需对大型语言模型（LLM）进行额外的微调，即可保持强大的理解能力。

**论文中提到的局限性：**

*   **模型容量与细节的权衡：** 论文提到，虽然 96 通道的 PS-VAE 提供了更好的重建质量，但在生成指标上略逊于 32 通道的 PS-VAE，这可能是因为模型容量有限，难以同时建模过多的精细细节。
*   **高维特征直接增强的失败：** 论文尝试直接在原始高维特征空间上进行像素重建，发现虽然重建质量有所提升，但生成性能却大幅下降，出现严重的结构伪影，表明直接处理高维特征空间存在固有的困难。
*   **SigLIP2 的重建性能：** 在使用 SigLIP2 作为编码器时，其重建性能相比 DINOv2 略有饱和，可能与其更高级别的抽象表示有关。

**潜在的未来研究方向：**

*   **更高分辨率的生成：** 论文提到，将 PS-VAE 应用于更高分辨率的生成将进一步提升其能力。
*   **LLM 微调的协同作用：** 论文推测，如果对 LLM 进行联合微调，可能会使 SigLIP2 编码器在理解任务上表现得更好。
*   **更深入的架构探索：** 论文中提到了一些架构上的探索，例如对称设计和直接在高维空间增强细节，这些方向仍有待进一步深入研究。
*   **模型容量的扩展：** 探索如何通过增加模型容量来更好地利用高通道数潜在空间中的丰富信息，以实现更高的生成性能上限。

总而言之，这篇论文成功地弥合了视觉理解编码器与生成任务之间的鸿沟，通过引入创新的语义-像素重建目标和 PS-VAE 模型，实现了在文本到图像生成和图像编辑任务上的显著进步，并为构建统一的视觉理解与生成模型提供了有力的支持。

**Key Findings:**

- In this paper, we propose a systematic framework to adapt understanding-oriented encoder features for generative tasks.
- We introduce a semantic-pixel reconstruction objective to regularize the latent space, enabling the compression of both semantic information and fine-grained details into a highly compact representation (96 channels with 16x16 spatial downsampling).
- This design ensures that the latent space remains semantically rich and achieves state-of-the-art image reconstruction, while remaining compact enough for accurate generation.
- Benchmarking against various feature spaces, we demonstrate that our approach achieves state-of-the-art reconstruction, faster convergence, and substantial performance gains in both T2I and editing tasks, validating that representation encoders can be effectively adapted into robust generative components.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17909v1)
- [arXiv](https://arxiv.org/abs/2512.17909v1)

---

<a id='2512.17908v1'></a>
## [Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](https://arxiv.org/abs/2512.17908v1)

**Authors:** Ananta R. Bhattarai, Helge Rhodin

**Published:** 2025-12-19

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Monocular depth estimation remains challenging as recent foundation models, such as Depth Anything V2 (DA-V2), struggle with real-world images that are far from the training distribution. We introduce Re-Depth Anything, a test-time self-supervision framework that bridges this domain gap by fusing DA-V2 with the powerful priors of large-scale 2D diffusion models. Our method performs label-free refinement directly on the input image by re-lighting predicted depth maps and augmenting the input. This re-synthesis method replaces classical photometric reconstruction by leveraging shape from shading (SfS) cues in a new, generative context with Score Distillation Sampling (SDS). To prevent optimization collapse, our framework employs a targeted optimization strategy: rather than optimizing depth directly or fine-tuning the full model, we freeze the encoder and only update intermediate embeddings while also fine-tuning the decoder. Across diverse benchmarks, Re-Depth Anything yields substantial gains in depth accuracy and realism over the DA-V2, showcasing new avenues for self-supervision by augmenting geometric reasoning.

**Analysis:**

好的，这是对论文“Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting”的全面中文摘要：

**论文题目：** Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting

**作者：** Ananta R. Bhattarai, Helge Rhodin

**摘要：**

**1. 研究问题/核心挑战：**
单目深度估计（Monocular Depth Estimation, MDE）在处理与训练数据分布差异较大的真实世界图像时，即使是先进的基础模型（如Depth Anything V2, DA-V2），也常常表现出不准确性。这种“域间隙”（domain gap）导致预测的深度图在细节、真实感和几何结构上存在不足。

**2. 主要创新点/方法论贡献：**
本文提出了一种名为 **Re-Depth Anything** 的新颖 **测试时自监督框架**，旨在弥合这一域间隙。其核心创新在于：

*   **融合2D扩散模型先验：** 将强大的2D扩散模型（如Stable Diffusion）的先验知识与DA-V2模型相结合，利用扩散模型对图像光照和纹理的丰富理解来指导深度图的精炼。
*   **基于重照明（Re-lighting）的自监督信号：** 引入了一种创新的 **重合成（re-synthesis）方法**，通过随机改变输入图像的光照条件来生成重照明的图像。这种方法取代了传统的、对渲染器要求极高的光度重建（photometric reconstruction）。
*   **利用形状阴影（SfS）和SDS损失：** 借鉴了形状阴影（Shape-from-Shading, SfS）的原理，在生成式上下文中使用 **分数蒸馏采样（Score Distillation Sampling, SDS）** 损失来评估重照明图像与原始图像的匹配度，从而实现无标签的深度图精炼。
*   **目标化优化策略：** 为防止优化崩溃和保留几何结构，作者提出了一种 **目标化优化策略**。具体来说，冻结了DA-V2的编码器，仅更新中间特征嵌入（embeddings）和解码器（DPT head）的权重。这种方法在保留了预训练模型强大的几何知识的同时，有效地调整了模型以适应特定输入图像。
*   **多运行平均（Ensembling）：** 为了应对SDS损失的随机性带来的结果方差，通过多次运行优化并对结果进行平均来稳定最终的深度图预测。

**3. 主要结果与意义：**
Re-Depth Anything 在多个基准数据集（CO3D, KITTI, ETH3D）上均取得了显著的性能提升，相较于DA-V2基线模型，在深度准确性和真实感方面均有substantial gains。

*   **定量评估：** 在多个评估指标上（如AbsRel, RMSE, log10, SI log, SqRel等）均显示出相对误差的显著降低，例如在KITTI数据集的SI log和RMSE log上降低了8.5%，在ETH3D数据集的AbsRel上降低了8.4%。
*   **定性评估：** 实验结果表明，Re-Depth Anything 能够有效地增强细节（如纹理、边缘），去除平坦区域的噪声，并纠正模型在处理特定物体（如论文中的老虎图像）时可能出现的偏差（如从狗的形状纠正为老虎的形状）。
*   **意义：** 该方法展示了利用2D生成模型作为先验，通过测试时自监督学习来提升现有深度估计模型泛化能力的新途径，尤其是在处理“in-the-wild”图像时。它证明了在不依赖额外标注数据的情况下，可以有效地弥合模型训练数据与实际应用场景之间的差距。

**4. 提及的局限性：**
论文中提到了一些局限性：

*   **小范围的幻觉边缘：** 在某些情况下，模型可能会产生小的幻觉边缘，例如卡车上的贴纸。
*   **过度平滑：** 在某些区域，模型可能会过度平滑细节，例如在黑暗区域的树木，或者在某些场景中（如图10和图11所示的红色方框区域）。
*   **天空区域的幻觉：** 在KITTI数据集的某些场景中，模型可能会在天空区域产生幻觉（如图12所示的红色方框区域）。
*   **对物体细节的提升更明显：** 在CO3D数据集中，模型对单个物体细节的提升更为显著，而在房间和街道场景中，最大的收益来自于去除初始DA-V2预测中的可疑细节，从而产生更逼真的重照明效果，但可能保留了实际细节。

**5. 未来研究方向：**
作者表示，未来工作将探索：

*   **替代性的重合成方法：** 寻找其他能够生成逼真光照效果的重合成技术。
*   **大规模模型微调：** 探索在更大规模的真实世界视频数据上对基础模型进行微调的可能性。

**总结：**
Re-Depth Anything 是一项重要的工作，它通过一种创新的测试时自监督重照明方法，有效地解决了现有单目深度估计模型在处理分布外图像时的局限性。该方法巧妙地利用了2D扩散模型的强大先验，并采用目标化优化策略，在不增加标注成本的情况下，显著提升了深度图的准确性和真实感，为提升现有深度估计模型的泛化能力开辟了新的方向。

**Key Findings:**

- We introduce Re-Depth Anything, a test-time self-supervision framework that bridges this domain gap by fusing DA-V2 with the powerful priors of large-scale 2D diffusion models.
- Our method performs label-free refinement directly on the input image by re-lighting predicted depth maps and augmenting the input.
- This re-synthesis method replaces classical photometric reconstruction by leveraging shape from shading (SfS) cues in a new, generative context with Score Distillation Sampling (SDS).
- Across diverse benchmarks, Re-Depth Anything yields substantial gains in depth accuracy and realism over the DA-V2, showcasing new avenues for self-supervision by augmenting geometric reasoning.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17908v1)
- [arXiv](https://arxiv.org/abs/2512.17908v1)

---

<a id='2512.17907v1'></a>
## [Dexterous World Models](https://arxiv.org/abs/2512.17907v1)

**Authors:** Byungjun Kim, Taeksoo Kim, Junyoung Lee, Hanbyul Joo

**Published:** 2025-12-19

**Categories:** cs.CV

**Abstract:**

Recent progress in 3D reconstruction has made it easy to create realistic digital twins from everyday environments. However, current digital twins remain largely static and are limited to navigation and view synthesis without embodied interactivity. To bridge this gap, we introduce Dexterous World Model (DWM), a scene-action-conditioned video diffusion framework that models how dexterous human actions induce dynamic changes in static 3D scenes.   Given a static 3D scene rendering and an egocentric hand motion sequence, DWM generates temporally coherent videos depicting plausible human-scene interactions. Our approach conditions video generation on (1) static scene renderings following a specified camera trajectory to ensure spatial consistency, and (2) egocentric hand mesh renderings that encode both geometry and motion cues to model action-conditioned dynamics directly. To train DWM, we construct a hybrid interaction video dataset. Synthetic egocentric interactions provide fully aligned supervision for joint locomotion and manipulation learning, while fixed-camera real-world videos contribute diverse and realistic object dynamics.   Experiments demonstrate that DWM enables realistic and physically plausible interactions, such as grasping, opening, and moving objects, while maintaining camera and scene consistency. This framework represents a first step toward video diffusion-based interactive digital twins and enables embodied simulation from egocentric actions.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Dexterous World Models**

**1. 论文的主要贡献（2-3句话）**

该论文提出了 Dexterous World Model (DWM)，一个创新的场景-动作条件视频扩散框架，能够模拟人类灵巧动作如何引起静态三维场景的动态变化。DWM 能够根据静态三维场景渲染和以自我为中心的（egocentric）手部运动序列，生成时间连贯且具有物理合理性的人体与场景交互视频。这标志着在构建具有交互能力的数字孪生方面迈出了重要一步，实现了从自我中心动作出发的具身模拟。

**2. 关键创新或方法论**

DWM 的核心创新在于其**场景-动作条件视频扩散模型**的设计，以及如何有效地将**空间一致性**和**动作动态性**融入视频生成过程。具体来说：

*   **双重条件生成：** DWM 巧妙地将视频生成条件化为两个关键部分：
    *   **静态场景渲染（遵循指定相机轨迹）：** 这确保了生成的视频在空间上保持一致性，即场景的几何结构和相机视角不会随意变化，从而模拟了真实世界中观察者在固定场景中的视角。
    *   **以自我为中心的手部网格渲染（编码几何和运动）：** 这是实现“灵巧交互”的关键。通过直接输入手部网格的几何形状和运动信息，模型能够直接学习和预测手部动作如何影响场景中的物体，从而捕捉到精细的操纵动态。
*   **混合交互视频数据集：** 为了训练如此复杂的模型，作者构建了一个创新的混合数据集。
    *   **合成的以自我为中心的交互：** 提供完全对齐的监督信号，用于联合学习身体运动（locomotion）和物体操纵（manipulation）。这使得模型能够学习到从手部动作到物体响应的精确映射。
    *   **固定摄像机的真实世界视频：** 引入了多样性和真实感，捕捉了现实世界中物体动力学的复杂性，弥补了纯合成数据的不足。
*   **视频扩散框架：** 利用了视频扩散模型强大的生成能力，能够生成高质量、时间连贯的视频序列，并且能够学习到复杂的概率分布，从而生成逼真的交互效果。

**3. 对该领域的潜在影响**

DWM 的研究对计算机视觉领域具有重要的潜在影响，主要体现在：

*   **推动具身智能（Embodied AI）的发展：** 该框架为构建能够理解和执行复杂交互的具身智能体提供了新的途径。通过模拟人类的灵巧动作，可以训练出更具适应性和交互能力的机器人或虚拟代理。
*   **提升数字孪生的交互性：** 现有的数字孪生多为静态模型，DWM 的出现使得数字孪生能够具备动态的、可交互的属性，极大地增强了其在模拟、训练、设计等领域的应用价值。
*   **促进人机交互（Human-Computer Interaction）的研究：** 通过生成逼真的人体与虚拟环境的交互视频，可以用于研究和评估新的交互方式，以及训练能够理解和响应人类意图的交互系统。
*   **为视频生成领域注入新的活力：** 将具身交互的复杂性引入视频生成，为视频生成模型的设计和训练提供了新的挑战和方向，有望催生更强大、更通用的视频生成模型。
*   **加速物理模拟与视觉的融合：** DWM 通过学习动作如何影响场景，实际上是在隐式地学习物理规律。这有助于弥合纯粹的视觉模型和需要物理理解的模拟之间的差距。

**4. 可能受益的相关领域或应用**

*   **机器人学：** 训练机器人进行精细的抓取、操作和组装任务，尤其是在复杂或未知环境中。
*   **虚拟现实（VR）/增强现实（AR）：** 创建更具沉浸感和交互性的虚拟环境，让用户能够以更自然的方式与虚拟物体互动。
*   **游戏开发：** 生成更逼真、更具动态性的游戏场景和角色交互。
*   **影视特效：** 自动化生成复杂的物理交互动画，降低制作成本。
*   **产品设计与原型制作：** 在虚拟环境中模拟产品的使用过程，评估设计可行性。
*   **教育与培训：** 创建交互式模拟环境，用于技能培训，例如外科手术模拟或设备操作培训。
*   **内容创作：** 允许用户通过简单的手部动作生成复杂的场景交互视频。

**5. 可从摘要推断的局限性**

尽管 DWM 取得了显著进展，但从摘要中仍可推断出一些潜在的局限性：

*   **对输入数据的依赖性：** 模型性能高度依赖于输入的静态场景渲染质量、相机轨迹的指定以及手部网格渲染的准确性。如果输入数据存在噪声或不准确，可能会影响生成视频的质量。
*   **计算成本：** 视频扩散模型通常计算成本较高，训练和推理可能需要大量的计算资源。
*   **泛化能力：** 虽然混合数据集旨在提高多样性，但模型在未见过的新颖场景、物体或动作上的泛化能力仍需进一步验证。摘要中提到“plausible human-scene interactions”，这表明模型可能更侧重于生成“看起来合理”的交互，而非严格的物理精确性。
*   **“灵巧”的定义和范围：** 摘要中提到了“dexterous human actions”，但“灵巧”的定义和模型能够处理的动作复杂度范围可能有限。例如，非常精细的、需要多指协同的复杂操作可能仍是挑战。
*   **全局场景理解的深度：** 模型主要关注手部动作与局部场景的交互，对于需要全局场景理解才能完成的复杂任务（例如，需要规划长距离移动和多步操作的任务）可能存在不足。
*   **真实世界复杂性的捕捉：** 尽管引入了真实世界视频，但要完全捕捉所有现实世界中物体交互的细微之处（如材质、摩擦力、形变等）仍然是一个巨大的挑战。

总而言之，Dexterous World Model (DWM) 是一项令人兴奋的研究，它通过将场景-动作条件视频扩散与以自我为中心的手部运动相结合，有效地解决了数字孪生缺乏交互性的问题。其创新的方法论和对具身智能及数字孪生领域的潜在影响使其成为计算机视觉领域值得关注的亮点。

**Key Findings:**

- To bridge this gap, we introduce Dexterous World Model (DWM), a scene-action-conditioned video diffusion framework that models how dexterous human actions induce dynamic changes in static 3D scenes.
- Our approach conditions video generation on (1) static scene renderings following a specified camera trajectory to ensure spatial consistency, and (2) egocentric hand mesh renderings that encode both geometry and motion cues to model action-conditioned dynamics directly.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17907v1)
- [arXiv](https://arxiv.org/abs/2512.17907v1)

---

<a id='2512.17902v1'></a>
## [Adversarial Robustness of Vision in Open Foundation Models](https://arxiv.org/abs/2512.17902v1)

**Authors:** Jonathon Fox, William J Buchanan, Pavlos Papadopoulos

**Published:** 2025-12-19

**Categories:** cs.CV, cs.AI, cs.CR

**Abstract:**

With the increase in deep learning, it becomes increasingly difficult to understand the model in which AI systems can identify objects. Thus, an adversary could aim to modify an image by adding unseen elements, which will confuse the AI in its recognition of an entity. This paper thus investigates the adversarial robustness of LLaVA-1.5-13B and Meta's Llama 3.2 Vision-8B-2. These are tested for untargeted PGD (Projected Gradient Descent) against the visual input modality, and empirically evaluated on the Visual Question Answering (VQA) v2 dataset subset. The results of these adversarial attacks are then quantified using the standard VQA accuracy metric. This evaluation is then compared with the accuracy degradation (accuracy drop) of LLaVA and Llama 3.2 Vision. A key finding is that Llama 3.2 Vision, despite a lower baseline accuracy in this setup, exhibited a smaller drop in performance under attack compared to LLaVA, particularly at higher perturbation levels. Overall, the findings confirm that the vision modality represents a viable attack vector for degrading the performance of contemporary open-weight VLMs, including Meta's Llama 3.2 Vision. Furthermore, they highlight that adversarial robustness does not necessarily correlate directly with standard benchmark performance and may be influenced by underlying architectural and training factors.

**Analysis:**

好的，这是对论文“Adversarial Robustness of Vision in Open Foundation Models”的全面中文摘要：

**论文题目：** Adversarial Robustness of Vision in Open Foundation Models (视觉在开放基础模型中的对抗鲁棒性)

**作者：** Jonathan Fox, William J Buchanan, Pavlos Papadopoulos

**摘要：**

**1. 主要问题/研究问题：**
随着深度学习模型在识别物体方面的能力日益增强，理解这些模型的内部工作机制变得愈发困难。攻击者可以通过在图像中添加不易察觉的元素来欺骗AI，从而干扰其识别能力。本文旨在研究当前主流的开放权重视觉语言模型（VLMs）在视觉输入受到对抗性攻击时的鲁棒性。具体来说，研究关注的是LLaVA-1.5-13B和Meta的Llama 3.2 Vision-8B-2这两个模型，它们在视觉输入模态上，使用无目标投影梯度下降（PGD）攻击方法进行测试，并在视觉问答（VQA）v2数据集子集上进行实证评估。

**2. 关键创新/方法论贡献：**
*   **实证评估：** 本文首次系统性地比较了两种重要的开放权重VLM（LLaVA和Llama 3.2 Vision）在视觉对抗攻击下的鲁棒性。
*   **对抗攻击方法：** 使用了无目标PGD攻击方法，该方法被认为是评估对抗鲁棒性的标准基准。
*   **评估指标：** 采用标准的VQA准确率作为评估指标，并通过比较模型在干净图像和对抗性图像上的准确率下降（accuracy drop）来量化鲁棒性。
*   **模型对比：** 深入分析了两种模型在不同架构（LLaVA的简单投影层 vs. Llama 3.2 Vision的交叉注意力适配器）和训练规模上的差异如何影响其对抗鲁棒性。

**3. 主要结果及其意义：**
*   **普遍脆弱性：** 研究发现，LLaVA和Llama 3.2 Vision模型都对视觉输入的对抗性攻击表现出明显的脆弱性。即使是细微的、近乎不可察觉的扰动（ε < 16/255），也会导致VQA准确率的下降。
*   **不同的鲁棒性表现：** 尽管LLaVA在干净数据集上表现出更高的基线准确率（87.4%），但在对抗攻击下，其准确率下降幅度更大（最高可达36.0%）。相比之下，Llama 3.2 Vision虽然基线准确率较低（42.8%），但在对抗攻击下表现出更小的准确率下降（最高10.4%），尤其是在高扰动水平下，其性能下降幅度相对较小。这表明Llama 3.2 Vision在一定程度上表现出更强的相对鲁棒性。
*   **架构与训练的影响：** 研究推测，Llama 3.2 Vision更复杂的交叉注意力适配器机制、更大的预训练数据集以及可能更先进的对齐过程，可能有助于其形成更稳定的内部表示，从而在对抗攻击下表现出更好的稳定性。
*   **重要发现：**
    *   视觉模态是降级当前开放权重VLM性能的可行攻击途径。
    *   对抗鲁棒性并不总是与标准的基准性能直接相关。
    *   模型架构和训练因素可能对对抗鲁棒性产生重要影响。

**4. 论文中提到的局限性：**
*   **数据集规模：** 由于计算资源的限制，实验仅在VQA v2数据集的500个样本子集上进行，这可能无法完全代表模型在完整数据集或其他数据集上的表现。
*   **攻击方法限制：** 研究仅使用了无目标PGD攻击和L∞范数约束。其他攻击算法（如CW攻击）或范数约束（L2, L∞）可能揭示不同的脆弱性。
*   **攻击目标：** 攻击是无目标的，旨在降低整体性能，而非诱导特定的错误输出。有针对性的攻击可能带来不同的挑战。
*   **任务范围：** 鲁棒性仅在VQA任务上进行了评估，在图像描述或复杂推理等其他多模态任务上的表现可能有所不同。
*   **超参数探索：** 计算成本限制了对PGD参数空间（迭代次数、步长）的详尽探索。

**5. 未来研究方向：**
*   **更全面的评估：** 在更大的数据集子集或完整的VQA v2数据集上重复实验，并扩展到其他多模态基准测试。
*   **多样化的攻击方法：** 探索更广泛的攻击算法，包括Carlini & Wagner (CW) 攻击、Image Hijacks等，以及有针对性的攻击。
*   **性能差异分析：** 深入探究Llama 3.2 Vision在VQA子集上基线性能较低的原因，可能涉及不同的提示配置或预处理步骤。
*   **架构与训练的深入分析：** 详细分析多模态适配器的设计、预训练数据规模、对齐技术（如RLHF）等因素如何量化地影响对抗鲁棒性。
*   **防御机制研究：** 开发和评估专门针对VLM的有效防御策略，以减轻视觉对抗攻击的风险。
*   **原生多模态 vs. 适配器方法：** 进一步研究原生多模态架构与适配器方法的区别，以及它们对模型脆弱性的影响。

**总结：**
这篇论文通过对LLaVA-1.5-13B和Llama 3.2 Vision-8B-2这两个重要的开放权重视觉语言模型进行实证评估，揭示了视觉模态在对抗性攻击下的普遍脆弱性。研究结果表明，对抗鲁棒性并非总是与基线性能成正比，并且模型架构和训练策略在其中扮演着关键角色。这项研究为理解和提升当前和未来多模态基础模型的安全性提供了重要的见解。

**Key Findings:**

- Overall, the findings confirm that the vision modality represents a viable attack vector for degrading the performance of contemporary open-weight VLMs, including Meta's Llama 3.2 Vision.
- Furthermore, they highlight that adversarial robustness does not necessarily correlate directly with standard benchmark performance and may be influenced by underlying architectural and training factors.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17902v1)
- [arXiv](https://arxiv.org/abs/2512.17902v1)

---

<a id='2512.17897v1'></a>
## [RadarGen: Automotive Radar Point Cloud Generation from Cameras](https://arxiv.org/abs/2512.17897v1)

**Authors:** Tomer Borreda, Fangqiang Ding, Sanja Fidler, Shengyu Huang, Or Litany

**Published:** 2025-12-19

**Categories:** cs.CV, cs.AI, cs.LG, cs.RO

**Abstract:**

We present RadarGen, a diffusion model for synthesizing realistic automotive radar point clouds from multi-view camera imagery. RadarGen adapts efficient image-latent diffusion to the radar domain by representing radar measurements in bird's-eye-view form that encodes spatial structure together with radar cross section (RCS) and Doppler attributes. A lightweight recovery step reconstructs point clouds from the generated maps. To better align generation with the visual scene, RadarGen incorporates BEV-aligned depth, semantic, and motion cues extracted from pretrained foundation models, which guide the stochastic generation process toward physically plausible radar patterns. Conditioning on images makes the approach broadly compatible, in principle, with existing visual datasets and simulation frameworks, offering a scalable direction for multimodal generative simulation. Evaluations on large-scale driving data show that RadarGen captures characteristic radar measurement distributions and reduces the gap to perception models trained on real data, marking a step toward unified generative simulation across sensing modalities.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：RadarGen: Automotive Radar Point Cloud Generation from Cameras**

**1. 论文的主要贡献 (2-3句话的简洁总结)**

本论文提出了 RadarGen，一个创新的扩散模型，能够从多视角摄像头图像生成逼真的汽车雷达点云。该模型通过将雷达测量值表示为鸟瞰图（BEV）形式，并结合视觉线索进行引导，实现了从视觉数据到雷达数据的跨模态生成，为多模态生成式仿真提供了一个可扩展的方向。

**2. 关键创新或方法论**

RadarGen 的核心创新在于其将**扩散模型**这一强大的生成技术成功应用于**雷达点云的生成**，并解决了跨模态生成中的关键挑战。具体来说，其方法论的关键点包括：

*   **雷达数据在BEV空间的表示：** 将雷达测量值（包括空间结构、雷达截面积 RCS 和多普勒信息）编码到鸟瞰图（BEV）表示中。这种表示方式能够有效地捕捉雷达数据的空间特性，并为扩散模型的处理奠定基础。
*   **高效的图像-潜在扩散模型适配：** 将高效的图像领域扩散模型（Image-Latent Diffusion）适配到雷达领域。这意味着模型在潜在空间中进行扩散过程，从而提高了生成效率和质量。
*   **轻量级的点云恢复：** 在生成BEV表示后，通过一个轻量级的恢复步骤将其转换回三维雷达点云。这保证了生成结果的可用性，可以直接用于下游任务。
*   **多模态线索的融合与引导：** 这是 RadarGen 最具吸引力的创新点之一。模型利用预训练的**基础模型（Foundation Models）**提取的**BEV对齐的深度、语义和运动信息**来指导扩散过程。这些视觉线索能够确保生成的雷达模式在物理上是合理的，并与真实的视觉场景更加匹配。这种显式的跨模态对齐是实现高质量生成和仿真一致性的关键。
*   **条件生成（Conditioning on Images）：** 模型以摄像头图像作为条件进行生成，这使得 RadarGen 能够**原则上兼容现有的视觉数据集和仿真框架**。这极大地降低了应用门槛，并为大规模多模态生成式仿真铺平了道路。

**3. 对该领域的潜在影响**

RadarGen 的研究对计算机视觉和自动驾驶领域具有重要的潜在影响：

*   **推动多模态生成式仿真：** 长期以来，为自动驾驶系统生成逼真的传感器数据一直是研究的热点。RadarGen 的工作为**统一的多模态生成式仿真**提供了一个可行的路径，能够同时生成摄像头图像和雷达点云，从而实现更真实、更全面的训练和测试环境。
*   **缓解真实数据稀缺问题：** 尤其是在特定场景或极端天气条件下，获取高质量的真实雷达数据可能非常困难。RadarGen 的生成能力可以**合成大量多样化的雷达数据**，用于扩充训练集，提高模型的鲁棒性。
*   **加速传感器融合研究：** 通过生成与视觉信息高度对齐的雷达数据，RadarGen 可以为研究**更先进的传感器融合算法**提供高质量的合成数据，从而加速相关研究的进展。
*   **降低数据采集和标注成本：** 相较于手动采集和标注真实传感器数据，生成式方法可以显著**降低数据相关的成本**。
*   **为雷达感知模型提供更优的训练数据：** 摘要中提到，RadarGen 生成的数据能够**减小与真实数据训练的感知模型的差距**，这意味着使用合成数据训练的模型在真实世界中的表现会更好。

**4. 可能受益的相关领域或应用**

*   **自动驾驶（Autonomous Driving）：** 这是最直接的应用领域。RadarGen 可以用于生成训练数据、测试场景、以及开发和验证雷达感知算法、传感器融合算法、以及端到端的自动驾驶系统。
*   **机器人技术（Robotics）：** 机器人也常常需要感知周围环境，雷达是一种重要的传感器。RadarGen 可以为机器人提供逼真的雷达感知数据，用于导航、避障、目标跟踪等任务的训练和仿真。
*   **计算机视觉（Computer Vision）：** 尽管主要关注雷达，但其跨模态生成能力也可能对其他需要从视觉生成其他模态数据的任务产生启发。
*   **虚拟现实/增强现实（VR/AR）：** 在构建逼真的虚拟环境时，模拟各种传感器数据是重要的组成部分。RadarGen 可以为VR/AR应用提供逼真的雷达模拟。
*   **遥感（Remote Sensing）：** 某些遥感应用也可能使用雷达技术，生成式方法可以用于模拟不同的地物和环境下的雷达回波。

**5. 可从摘要推断的局限性**

尽管 RadarGen 展现了巨大的潜力，但从摘要中可以推断出一些潜在的局限性：

*   **对基础模型的依赖：** 模型依赖于预训练的基础模型来提取视觉线索。如果基础模型的性能不佳或在特定场景下失效，可能会影响 RadarGen 的生成质量。
*   **“物理上可信”的定义：** 摘要中提到“物理上可信的雷达模式”。虽然这是目标，但“可信”的程度和如何量化其物理准确性可能是一个挑战。生成的雷达点云是否能完全捕捉到所有细微的物理现象（如多径效应、杂波等）仍需进一步验证。
*   **计算成本：** 扩散模型通常计算成本较高，尽管摘要提到了“高效的图像-潜在扩散”，但生成大规模、高分辨率的雷达点云可能仍然需要大量的计算资源。
*   **泛化能力：** 虽然模型兼容现有数据集，但其在**未见过**的极端场景、天气条件或传感器配置下的泛化能力仍需在实际评估中验证。
*   **雷达特有的复杂性：** 雷达数据具有其独特的物理特性，例如角度分辨率、距离分辨率、以及不同目标对雷达信号的散射特性。RadarGen 在多大程度上能够精确模拟这些复杂性有待进一步研究。
*   **“减小差距”的程度：** 摘要提到“减小了与感知模型训练在真实数据的差距”。这表明差距仍然存在，合成数据可能还不能完全替代真实数据，尤其是在对模型性能要求极高的场景下。

总而言之，RadarGen 是一项令人兴奋的研究，它将先进的生成模型与跨模态融合技术相结合，为解决自动驾驶领域中雷达数据生成和仿真的一大难题提供了新的思路和强大的工具。其对视觉线索的巧妙利用是实现逼真雷达点云生成的关键。

**Key Findings:**

- We present RadarGen, a diffusion model for synthesizing realistic automotive radar point clouds from multi-view camera imagery.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17897v1)
- [arXiv](https://arxiv.org/abs/2512.17897v1)

---

<a id='2512.17891v1'></a>
## [Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training](https://arxiv.org/abs/2512.17891v1)

**Authors:** Kristoffer Wickstrøm, Teresa Dorszewski, Siyan Chen, Michael Kampffmeyer, Elisabeth Wetzer, Robert Jenssen

**Published:** 2025-12-19

**Categories:** cs.CV

**Abstract:**

Current approaches for designing self-explainable models (SEMs) require complicated training procedures and specific architectures which makes them impractical. With the advance of general purpose foundation models based on Vision Transformers (ViTs), this impracticability becomes even more problematic. Therefore, new methods are necessary to provide transparency and reliability to ViT-based foundation models. In this work, we present a new method for turning any well-trained ViT-based model into a SEM without retraining, which we call Keypoint Counting Classifiers (KCCs). Recent works have shown that ViTs can automatically identify matching keypoints between images with high precision, and we build on these results to create an easily interpretable decision process that is inherently visualizable in the input. We perform an extensive evaluation which show that KCCs improve the human-machine communication compared to recent baselines. We believe that KCCs constitute an important step towards making ViT-based foundation models more transparent and reliable.

**Analysis:**

好的，这是对论文“Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training”的全面中文摘要：

**论文题目：** Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training (关键点计数分类器：将 Vision Transformers 转化为无需训练的自解释模型)

**作者：** Kristoffer Wickstrøm, Teresa Dorszewski, Siyan Chen, Michael Kampffmeyer, Elisabeth Wetzer, Robert Jenssen

---

**摘要**

**1. 研究问题/核心挑战：**

当前设计自解释模型（SEMs）的方法通常需要复杂的训练流程和特定的模型架构，这使得它们在实际应用中难以推广，尤其是在日益重要的 Vision Transformer (ViT) 基础模型领域。ViT 的通用性和强大的表征能力带来了便利，但其缺乏透明度和可解释性限制了其在安全关键领域的应用。现有的 SEMs 方法往往与 ViT 架构不兼容，或者需要额外的训练，这削弱了 ViT 的灵活性。此外，现有的 SEMs 可视化解释方法（如边界框和热力图）存在精度不足、信息量少、用户研究表现不佳等问题，阻碍了人机之间的有效沟通。因此，研究如何为 ViT 基础模型提供透明度和可靠性，并改进解释的可视化方式，是本文要解决的核心问题。

**2. 主要创新点/方法论贡献：**

本文提出了一种名为**关键点计数分类器（Keypoint Counting Classifiers, KCCs）**的新方法，旨在将任何已训练好的 ViT 模型转化为自解释模型（SEM），且**无需进行任何额外的训练**。KCCs 的核心创新在于：

*   **无需训练的 ViT 解释：** KCCs 利用 ViT 本身在识别图像语义部分（keypoints）方面的能力，通过比较查询图像与原型图像的关键点匹配情况来进行分类。整个过程不涉及对 ViT 特征提取器或分类头的任何修改或再训练，从而保留了 ViT 的灵活性。
*   **基于关键点的可视化解释：** KCCs 将解释可视化为匹配的关键点，这是一种新颖的可视化方法。这种方法受到教学材料和视觉学习中常用关键点表示的启发，旨在提供更直观、更易于理解的解释，避免了边界框和热力图的局限性。
*   **关键点识别与匹配：**
    *   **关键点识别：** KCCs 首先利用 ViT 的 token 能够捕捉语义部分信息的特性，结合前景分割和 SLIC 超像素分割等技术，将图像分割成语义上连贯的区域，并将每个区域的中心定义为一个关键点。
    *   **关键点匹配：** 利用互斥最近邻（Mutual NNs）的方法，比较查询图像和原型图像的关键点表示，找出相互匹配的关键点对。
*   **基于计数的分类：** 最终的分类决策是通过计算查询图像与属于某个类别的原型图像之间匹配到的关键点数量来完成的。
*   **利用 Vision-Language 模型增强解释：** 在特定情况下，当 ViT 具备视觉-语言能力时，KCCs 可以利用这些能力自动为关键点生成文本描述，从而进一步减少读者的解释偏差，提高解释的清晰度。

**3. 主要结果与意义：**

*   **用户研究结果：** 用户研究表明，KCCs 在解释质量和理解性方面显著优于现有的基线方法（PIP-Net 和 KMEx）。用户对 KCCs 的解释质量评分最高，并且更容易理解。在用户偏好方面，KCCs 与 KMEx 并列获得最高评分，表明 KCCs 在提供高质量解释的同时，也具有良好的用户友好性。更重要的是，KCCs 使得用户在纠正模型预测时更有信心，有助于缓解自动化偏差。
*   **定量评估结果：** 在 CUB200、CARS 和 PETS 等数据集上的定量评估显示，KCCs 在准确性和复杂度方面与许多需要额外训练的 SEMs 方法相比具有竞争力，甚至在某些情况下表现更优（例如，在 CUB200 数据集上，KCCs 的准确率高于未训练的 ProtoPNet）。这证明了无需训练的方法也能达到良好的性能。
*   **减少读者偏差的潜力：** 通过结合视觉-语言模型，KCCs 能够为关键点提供自动文本描述，这是一种减少解释偏差的有效途径，使得解释更加明确和客观。
*   **意义：** KCCs 为 ViT 基础模型提供了一种新颖、灵活且无需训练的自解释方法，显著提升了模型的可解释性和人机沟通效率。它为构建更透明、更可靠的深度学习模型开辟了新的方向。

**4. 提及的局限性：**

*   **关键点权重问题：** KCCs 目前对所有关键点赋予相同的权重，但实际上某些关键点对特定类别的区分度可能更高（例如，鸟类的喙部形状）。如何为关键点引入更具信息量的权重是一个潜在的研究方向。
*   **计算复杂度：** 当原型数量非常多时，计算关键点之间的相似性可能会变得内存密集。虽然论文提出了一种优化策略（仅考虑与最近的 J 个原型相关的 token），但计算成本仍需考虑。
*   **对 fine-grained 分类的挑战：** 在高度细粒度的分类任务中，监督信号仍然是必要的，这表明 KCCs 在这些场景下可能需要与监督学习方法结合。

**5. 潜在的未来研究方向：**

*   **引入关键点权重：** 研究如何为关键点引入自适应的权重，以提高分类性能和解释的准确性。
*   **识别类特定关键点：** 探索如何识别并利用类特定的关键点，以减少冗余并增强解释的特异性。
*   **更广泛的 ViT 模型和任务应用：** 将 KCCs 应用于更多不同类型的 ViT 模型和更广泛的计算机视觉任务。
*   **结合其他 XAI 技术：** 探索 KCCs 与其他可解释性技术（如注意力机制分析）的结合，以提供更全面的解释。
*   **进一步减少读者偏差：** 探索更多利用视觉-语言模型的方法，以实现更鲁棒和更少偏差的自动文本解释。

总而言之，这篇论文提出了一种创新的 KCCs 方法，成功地将强大的 ViT 模型转化为无需训练的自解释模型，并通过直观的关键点可视化和自动文本描述，显著提升了模型的可解释性和用户体验，为 XAI 领域带来了重要的贡献。

**Key Findings:**

- Therefore, new methods are necessary to provide transparency and reliability to ViT-based foundation models.
- In this work, we present a new method for turning any well-trained ViT-based model into a SEM without retraining, which we call Keypoint Counting Classifiers (KCCs).

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17891v1)
- [arXiv](https://arxiv.org/abs/2512.17891v1)

---

<a id='2512.17875v1'></a>
## [Visually Prompted Benchmarks Are Surprisingly Fragile](https://arxiv.org/abs/2512.17875v1)

**Authors:** Haiwen Feng, Long Lian, Lisa Dunlap, Jiahao Shu, XuDong Wang, Renhao Wang, Trevor Darrell, Alane Suhr, Angjoo Kanazawa

**Published:** 2025-12-19

**Categories:** cs.CV, cs.LG

**Abstract:**

A key challenge in evaluating VLMs is testing models' ability to analyze visual content independently from their textual priors. Recent benchmarks such as BLINK probe visual perception through visual prompting, where questions about visual content are paired with coordinates to which the question refers, with the coordinates explicitly marked in the image itself. While these benchmarks are an important part of VLM evaluation, we find that existing models are surprisingly fragile to seemingly irrelevant details of visual prompting: simply changing a visual marker from red to blue can completely change rankings among models on a leaderboard. By evaluating nine commonly-used open- and closed-source VLMs on two visually prompted tasks, we demonstrate how details in benchmark setup, including visual marker design and dataset size, have a significant influence on model performance and leaderboard rankings. These effects can even be exploited to lift weaker models above stronger ones; for instance, slightly increasing the size of the visual marker results in open-source InternVL3-8B ranking alongside or better than much larger proprietary models like Gemini 2.5 Pro. We further show that low-level inference choices that are often ignored in benchmarking, such as JPEG compression levels in API calls, can also cause model lineup changes. These details have substantially larger impacts on visually prompted benchmarks than on conventional semantic VLM evaluations. To mitigate this instability, we curate existing datasets to create VPBench, a larger visually prompted benchmark with 16 visual marker variants. VPBench and additional analysis tools are released at https://lisadunlap.github.io/vpbench/.

**Analysis:**

好的，这是对论文“Visually Prompted Benchmarks Are Surprisingly Fragile”的全面中文摘要：

**论文题目：** Visually Prompted Benchmarks Are Surprisingly Fragile (视觉提示基准测试出奇地脆弱)

**作者：** Haiwen Feng, Long Lian, Lisa Dunlap, Jiahao Shu, XuDong Wang, Renhao Wang, Trevor Darrell, Alane Suhr, Angjoo Kanazawa

**摘要：**

**1. 主要问题/研究问题：**

该论文的核心研究问题在于，当前用于评估视觉语言模型（VLMs）的视觉提示（visually prompted）基准测试，在多大程度上受到非语义性因素（如视觉标记的设计、数据集大小、甚至低级推理设置）的影响，以及这些因素如何导致模型性能和排行榜的**脆弱性**。研究旨在揭示这些基准测试在多大程度上衡量的是模型的真实视觉感知能力，而非其对基准测试设计细节的敏感度。

**2. 关键创新/方法贡献：**

*   **系统性地揭示视觉提示基准的脆弱性：** 作者通过对九个常用开源和闭源 VLM 进行实验，系统性地展示了视觉提示的细微变化（如标记颜色、大小、形状、位置）如何显著影响模型性能和排行榜。
*   **引入 VPBench 基准测试：** 为了解决现有基准测试的局限性，作者从现有大型数据集中（DA2K 和 SPair-71k）构建了一个更大、更具多样性的视觉提示基准测试——VPBench。VPBench 包含 16 种不同的视觉标记变体，旨在提供更稳定、更可靠的评估。
*   **量化不同因素的影响：** 论文量化了数据采样、视觉标记设计和低级推理设置（如 JPEG 压缩）对模型性能和排行榜的影响，并将其与传统语义基准测试进行了对比。
*   **提出缓解策略：** 作者提出了一系列建议，以提高视觉提示基准测试的稳健性，包括标准化和多样化视觉提示、使用一致的数据源和低级设置、报告不确定性和排行榜稳定性等。

**3. 主要结果及其意义：**

*   **视觉提示的细微变化可导致排行榜剧烈变动：** 研究发现，即使是简单的视觉标记颜色从红色变为蓝色，也可能完全改变模型在排行榜上的排名。这种敏感性远超传统语义基准测试。
*   **弱模型可被“操纵”以超越强模型：** 通过精心选择视觉标记（例如，将标记改为方形），较弱的模型（如 InternVL3-8B）可以超越更强大的专有模型（如 Gemini 2.5 Pro）。
*   **低级推理设置影响显著：** JPEG 压缩等通常被忽略的低级推理设置，也会对视觉提示基准测试产生显著影响，导致模型排名变化。
*   **VPBench 提供了更稳定的评估：** VPBench 基准测试通过增加数据量和提供多种标记变体，显著降低了评估的方差，使得模型性能差异更容易被区分，排行榜也更加稳定。
*   **意义：** 这些发现表明，当前视觉提示基准测试的有效性受到严重质疑，它们可能更多地反映了基准测试本身的“设计缺陷”而非模型的真实能力。这迫切需要改进评估方法，以确保对 VLM 视觉感知能力的公平和准确评估。

**4. 提及的局限性：**

*   **VPBench 的局限性：** 虽然 VPBench 旨在提高稳定性，但目前它主要集中在相对深度和语义对应任务上，可能无法完全代表所有类型的视觉提示任务。
*   **模型对特定提示的过拟合：** 研究暗示，一些模型可能对特定视觉提示（如 BLINK 默认的红色圆圈标记）存在过拟合现象，导致其在其他标记下性能下降。
*   **评估的复杂性：** 即使在 VPBench 中，要完全消除所有不确定性仍然具有挑战性，尤其是在模型性能接近的情况下。

**5. 潜在的未来研究方向：**

*   **开发更鲁棒的 VLM 评估框架：** 基于本研究的发现，未来需要开发更全面的评估框架，能够抵御非语义性因素的干扰，并提供更可靠的模型能力衡量。
*   **研究模型对视觉提示的内在敏感性：** 深入探究不同 VLM 模型为何对特定的视觉提示设计表现出不同的敏感性，这有助于理解模型的内部机制和潜在的偏见。
*   **探索更多类型的视觉提示任务：** 将本研究的发现推广到更广泛的视觉提示任务中，以验证这种脆弱性是否普遍存在。
*   **开发对抗性鲁棒性评估：** 进一步研究如何通过操纵视觉提示来“欺骗”模型，并开发相应的对抗性鲁棒性评估方法。
*   **标准化视觉提示设计：** 推动社区在设计视觉提示基准测试时，采用更统一和标准化的方法，以减少评估的不确定性。

总而言之，这篇论文通过揭示视觉提示基准测试的脆弱性，对当前 VLM 评估方法提出了严峻的挑战，并为未来开发更可靠、更具信息量的评估工具和方法奠定了基础。

**Key Findings:**

- By evaluating nine commonly-used open- and closed-source VLMs on two visually prompted tasks, we demonstrate how details in benchmark setup, including visual marker design and dataset size, have a significant influence on model performance and leaderboard rankings.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17875v1)
- [arXiv](https://arxiv.org/abs/2512.17875v1)

---

<a id='2512.17873v1'></a>
## [InSPECT: Invariant Spectral Features Preservation of Diffusion Models](https://arxiv.org/abs/2512.17873v1)

**Authors:** Baohua Yan, Qingyuan Liu, Jennifer Kava, Xuan Di

**Published:** 2025-12-19

**Categories:** cs.CV

**Abstract:**

Modern diffusion models (DMs) have achieved state-of-the-art image generation. However, the fundamental design choice of diffusing data all the way to white noise and then reconstructing it leads to an extremely difficult and computationally intractable prediction task. To overcome this limitation, we propose InSPECT (Invariant Spectral Feature-Preserving Diffusion Model), a novel diffusion model that keeps invariant spectral features during both the forward and backward processes. At the end of the forward process, the Fourier coefficients smoothly converge to a specified random noise, enabling features preservation while maintaining diversity and randomness. By preserving invariant features, InSPECT demonstrates enhanced visual diversity, faster convergence rate, and a smoother diffusion process. Experiments on CIFAR-10, Celeb-A, and LSUN demonstrate that InSPECT achieves on average a 39.23% reduction in FID and 45.80% improvement in IS against DDPM for 10K iterations under specified parameter settings, which demonstrates the significant advantages of preserving invariant features: achieving superior generation quality and diversity, while enhancing computational efficiency and enabling faster convergence rate. To the best of our knowledge, this is the first attempt to analyze and preserve invariant spectral features in diffusion models.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：InSPECT: Invariant Spectral Features Preservation of Diffusion Models**

**1. 论文的主要贡献（2-3句话的简洁总结）**

本论文提出了一种名为 InSPECT 的新型扩散模型，其核心贡献在于在扩散模型的正向和反向过程中保持了不变的谱特征。通过确保傅里叶系数平滑地收敛到指定的随机噪声，InSPECT 在保留图像特征的同时，实现了更高的视觉多样性、更快的收敛速度和更平滑的扩散过程。实验结果表明，InSPECT 在生成质量和多样性方面显著优于现有方法，并提高了计算效率。

**2. 关键创新点或方法论**

InSPECT 的关键创新在于其**不变谱特征保持（Invariant Spectral Features Preservation）**的理念。传统的扩散模型（如 DDPM）将数据完全扩散到白噪声，这使得从噪声中重建原始数据成为一个极其困难且计算量巨大的任务。InSPECT 提出的方法打破了这一范式，通过在扩散过程中主动地保持数据的谱特征（例如，通过傅里叶变换得到的频率成分）不变。

具体来说，其方法论的核心在于：

*   **谱特征的定义与保持：** 论文明确提出要保持“不变的谱特征”。虽然摘要中没有详细说明具体是哪些谱特征，但可以推断是指在数据扩散过程中，某些关键的频率成分（如低频信息代表的整体结构、轮廓等）能够被保留下来，而不是像传统方法那样完全被噪声淹没。
*   **傅里叶系数的平滑收敛：** 论文提到“傅里叶系数平滑地收敛到指定的随机噪声”。这意味着在正向扩散的末端，数据虽然变成了噪声，但其谱特征的“残余”或“模式”是以一种可控、平滑的方式融入到噪声中的，而不是完全随机地消失。反向过程则利用这些保留的谱特征来指导重建。
*   **正向与反向过程的统一：** 这种谱特征的保持贯穿于正向和反向两个过程，使得模型在学习和生成时都能够利用这些稳定的特征信息。

**3. 对该领域的潜在影响**

InSPECT 的提出可能对扩散模型领域产生深远影响，主要体现在以下几个方面：

*   **提升生成质量与多样性：** 通过保留关键的谱特征，模型能够更好地捕捉数据的本质结构和细节，从而生成更逼真、更多样化的图像。摘要中提到的 FID 和 IS 指标的显著提升（平均 FID 降低 39.23%，IS 提高 45.80%）直接证明了这一点。
*   **提高计算效率与收敛速度：** 传统扩散模型需要大量的迭代才能达到令人满意的生成效果。InSPECT 通过保持不变的谱特征，简化了预测任务的难度，从而实现了更快的收敛速度和更高的计算效率。这对于实际应用中部署大型扩散模型至关重要。
*   **新的研究方向：** 论文声称这是“首次尝试分析和保留不变谱特征在扩散模型中”。这为扩散模型的研究开辟了一个新的视角，鼓励研究人员探索其他类型的“不变特征”或“结构化噪声”在扩散模型中的应用。
*   **理论理解的深化：** 这种对谱特征的关注可能有助于更深入地理解扩散模型的内部工作机制，以及数据在扩散过程中的信息丢失与保留机制。

**4. 可能受益于该研究的相关领域或应用**

*   **图像生成与编辑：** 任何需要高质量、高多样性图像生成的应用，如艺术创作、虚拟现实内容生成、游戏资产制作等。
*   **图像超分辨率与修复：** 保留关键的谱特征有助于在低分辨率图像或损坏图像中恢复细节，从而提升超分辨率和修复的效果。
*   **数据增强：** InSPECT 的生成能力可以用于生成更多样化的训练数据，以提高其他下游任务模型的鲁棒性。
*   **医学影像分析：** 在医学影像领域，保留关键的解剖结构信息（可能与谱特征相关）对于诊断和分析至关重要。
*   **科学模拟与可视化：** 在需要生成复杂物理现象或科学数据的场景下，保留关键的模式和结构信息将非常有益。
*   **视频生成：** 将谱特征保持的思想扩展到视频领域，可能有助于生成更连贯、更真实的视频序列。

**5. 从摘要中可以推断出的局限性**

尽管摘要展示了显著的优势，但仍可以推断出一些潜在的局限性：

*   **谱特征的具体定义与选择：** 摘要中并未详细说明“不变谱特征”具体是指哪些频率成分，以及如何选择和量化这些特征。不同的谱特征选择可能会对模型性能产生不同影响。
*   **计算成本的权衡：** 虽然摘要声称提高了计算效率，但“保持不变谱特征”可能需要在正向和反向过程中引入额外的计算步骤（例如，进行傅里叶变换、特征提取和注入等），这可能在某些方面增加计算复杂度，需要仔细权衡。
*   **参数设置的敏感性：** 摘要中提到“在指定参数设置下”。这暗示了模型的性能可能对参数设置比较敏感，需要精细的调优才能达到最佳效果。
*   **泛化能力：** 摘要中列举了 CIFAR-10, Celeb-A, LSUN 等数据集，这些数据集在复杂度和领域上有所不同。论文需要进一步证明其方法在更广泛、更复杂的数据集上的泛化能力。
*   **理论证明的深度：** 摘要强调了“首次尝试分析”，但具体的理论分析和数学证明可能还需要在论文正文中详细阐述，以支持其方法的有效性。
*   **“平滑收敛”的实现细节：** 如何实现傅里叶系数的“平滑收敛”到随机噪声，以及这种平滑性如何被反向过程有效利用，是实现其性能的关键，其具体实现细节可能需要进一步研究。

总而言之，InSPECT 提出的不变谱特征保持理念为扩散模型的研究提供了一个令人兴奋的新方向，有望在生成质量、效率和多样性方面带来显著的突破。其对谱特征的关注，以及在正向和反向过程中保持这些特征的创新方法，使其成为计算机视觉领域一项值得关注的研究。

**Key Findings:**

- Modern diffusion models (DMs) have achieved state-of-the-art image generation.
- To overcome this limitation, we propose InSPECT (Invariant Spectral Feature-Preserving Diffusion Model), a novel diffusion model that keeps invariant spectral features during both the forward and backward processes.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17873v1)
- [arXiv](https://arxiv.org/abs/2512.17873v1)

---

<a id='2512.17853v1'></a>
## [AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning](https://arxiv.org/abs/2512.17853v1)

**Authors:** Ran Gong, Xiaohan Zhang, Jinghuan Shang, Maria Vittoria Minniti, Jigarkumar Patel, Valerio Pepe, Riedana Yan, Ahmet Gundogdu, Ivan Kapelyukh, Ali Abbas, Xiaoqiang Yan, Harsh Patel, Laura Herlant, Karl Schmeckpeper

**Published:** 2025-12-19

**Categories:** cs.RO, cs.AI

**Abstract:**

Generalist robot learning remains constrained by data: large-scale, diverse, and high-quality interaction data are expensive to collect in the real world. While simulation has become a promising way for scaling up data collection, the related tasks, including simulation task design, task-aware scene generation, expert demonstration synthesis, and sim-to-real transfer, still demand substantial human effort. We present AnyTask, an automated framework that pairs massively parallel GPU simulation with foundation models to design diverse manipulation tasks and synthesize robot data. We introduce three AnyTask agents for generating expert demonstrations aiming to solve as many tasks as possible: 1) ViPR, a novel task and motion planning agent with VLM-in-the-loop Parallel Refinement; 2) ViPR-Eureka, a reinforcement learning agent with generated dense rewards and LLM-guided contact sampling; 3) ViPR-RL, a hybrid planning and learning approach that jointly produces high-quality demonstrations with only sparse rewards. We train behavior cloning policies on generated data, validate them in simulation, and deploy them directly on real robot hardware. The policies generalize to novel object poses, achieving 44% average success across a suite of real-world pick-and-place, drawer opening, contact-rich pushing, and long-horizon manipulation tasks. Our project website is at https://anytask.rai-inst.com .

**Analysis:**

好的，这是对论文“AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning”的全面中文摘要：

**论文题目：** AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning

**作者：** Ran Gong, Xiaohan Zhang, Jinghuan Shang, Maria Vittoria Minniti, Jigarkumar Patel, Valerio Pepe, Riedana Yan, Ahmet Gundogdu, Ivan Kapelyukh, Ali Abbas, Xiaoqiang Yan, Harsh Patel, Laura Herlant, Karl Schmeckpeper

**摘要：**

**1. 研究问题/核心挑战：**
通用机器人学习面临数据瓶颈：大规模、多样化且高质量的真实世界交互数据收集成本高昂且耗时。尽管模拟环境是扩展数据收集的有力工具，但任务设计、场景生成、专家演示合成以及从模拟到现实的迁移等环节仍需大量人工干预，这限制了生成数据的多样性和规模。

**2. 主要创新点/方法论贡献：**
本文提出了 **AnyTask**，一个自动化框架，旨在解决上述挑战。其核心创新在于：

*   **端到端自动化数据生成流程：** AnyTask 集成了大规模并行 GPU 模拟与基础模型（如大型语言模型 VLM/LLM），实现了从高层任务指令到机器人数据合成的整个流程的自动化，显著减少了人工干预。
*   **智能对象数据库与任务生成：** 利用 VLM 自动为对象生成多视角、多部件的渲染以及详细的元数据，并基于此数据库，通过 LLM 自动生成多样化的机器人任务描述和场景配置。
*   **多样化的专家演示生成代理：** 引入了三种代理来自动生成高质量的专家演示：
    *   **VIPR：** 一个新颖的任务与运动规划 (TAMP) 代理，结合了 VLM 进行迭代式并行精炼。
    *   **VIPR-EUREKA：** 一个强化学习 (RL) 代理，利用 LLM 生成的密集奖励和 LLM 指导的接触采样。
    *   **VIPR-RL：** 一个混合规划与学习的代理，结合了 TAMP 和 RL 的优势，能够仅凭稀疏奖励生成高质量演示。
*   **大规模并行模拟与数据合成：** 利用大规模并行 GPU 模拟器（如 IsaacLab）高效生成大量模拟数据，并应用在线域随机化来增强场景和视觉观察的多样性。
*   **零样本模拟到现实迁移：** 训练的策略可以直接部署到物理机器人上，无需真实世界数据进行微调。

**3. 主要结果及其意义：**
*   **策略性能：** 在真实世界中，训练在 AnyTask 生成数据上的行为克隆策略在多种任务（包括拾取与放置、开门、接触式推挤和长时序操作）上实现了 **44% 的平均成功率**。这些策略能够泛化到新颖的物体姿态。
*   **数据生成效率：** AnyTask 框架能够高效地生成大规模、多样化的数据，显著降低了数据收集的成本和人力投入。
*   **代理能力：** 三种代理（VIPR, VIPR-EUREKA, VIPR-RL）在不同类型的任务上展现出互补的能力，共同提高了整体任务解决能力。
*   **模拟到现实的有效性：** 实验证明，仅使用合成数据训练的策略可以成功迁移到物理机器人上，验证了 AnyTask 框架在弥合模拟与现实差距方面的有效性。

**4. 论文中提到的局限性：**
*   **高精度或复杂物理推理任务的性能：** 在需要高精度或复杂物理推理的任务（如堆叠任意物体）上，代理的性能仍有待提高。
*   **RGB 输入的局限性：** 成功的模拟到现实迁移依赖于点云观测。将框架扩展到 RGB 输入的策略将降低真实世界部署的门槛。

**5. 潜在的未来研究方向：**
*   **扩展对象和机器人形态：** 增加更多样的对象和机器人形态，以提高框架的通用性。
*   **更复杂的长时序移动操作任务：** 将框架扩展到更复杂的、长时序的移动操作任务。
*   **RGB 输入的策略：** 研究使用 RGB 图像作为输入来训练策略，以降低对特定传感器（如点云）的依赖。
*   **提升高精度和复杂物理推理能力：** 进一步改进代理在复杂物理交互任务上的表现。

**对计算机视觉领域的意义：**
AnyTask 的工作对计算机视觉领域具有重要意义，因为它：

*   **推动了视觉-语言模型在机器人领域的应用：** 论文展示了如何有效地利用 VLM/LLM 来理解自然语言指令，生成任务描述，并为机器人提供丰富的上下文信息，这为视觉-语言模型在机器人领域的更广泛应用奠定了基础。
*   **促进了大规模合成数据集的生成：** 通过自动化任务设计和数据生成流程，AnyTask 为训练更强大、更通用的机器人策略提供了大规模、多样化的合成数据集，这对于克服真实世界数据收集的限制至关重要。
*   **提升了模拟到现实迁移的鲁棒性：** 论文通过引入域随机化和点云增强策略，有效提升了模拟训练策略在真实世界中的泛化能力，这对于计算机视觉在机器人感知和控制中的应用具有重要价值。
*   **为机器人学习提供了新的范式：** AnyTask 提供了一个端到端的框架，将 LLM 的语言理解能力与大规模模拟器的物理仿真能力相结合，为机器人学习提供了一种新的、更高效的范式。

**Key Findings:**

- We present AnyTask, an automated framework that pairs massively parallel GPU simulation with foundation models to design diverse manipulation tasks and synthesize robot data.
- We introduce three AnyTask agents for generating expert demonstrations aiming to solve as many tasks as possible: 1) ViPR, a novel task and motion planning agent with VLM-in-the-loop Parallel Refinement; 2) ViPR-Eureka, a reinforcement learning agent with generated dense rewards and LLM-guided contact sampling; 3) ViPR-RL, a hybrid planning and learning approach that jointly produces high-quality demonstrations with only sparse rewards.
- The policies generalize to novel object poses, achieving 44% average success across a suite of real-world pick-and-place, drawer opening, contact-rich pushing, and long-horizon manipulation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17853v1)
- [arXiv](https://arxiv.org/abs/2512.17853v1)

---

<a id='2512.17851v1'></a>
## [InfSplign: Inference-Time Spatial Alignment of Text-to-Image Diffusion Models](https://arxiv.org/abs/2512.17851v1)

**Authors:** Sarah Rastegar, Violeta Chatalbasheva, Sieger Falkena, Anuj Singh, Yanbo Wang, Tejas Gokhale, Hamid Palangi, Hadi Jamali-Rad

**Published:** 2025-12-19

**Categories:** cs.CV, cs.AI

**Abstract:**

Text-to-image (T2I) diffusion models generate high-quality images but often fail to capture the spatial relations specified in text prompts. This limitation can be traced to two factors: lack of fine-grained spatial supervision in training data and inability of text embeddings to encode spatial semantics. We introduce InfSplign, a training-free inference-time method that improves spatial alignment by adjusting the noise through a compound loss in every denoising step. Proposed loss leverages different levels of cross-attention maps extracted from the backbone decoder to enforce accurate object placement and a balanced object presence during sampling. The method is lightweight, plug-and-play, and compatible with any diffusion backbone. Our comprehensive evaluations on VISOR and T2I-CompBench show that InfSplign establishes a new state-of-the-art (to the best of our knowledge), achieving substantial performance gains over the strongest existing inference-time baselines and even outperforming the fine-tuning-based methods. Codebase is available at GitHub.

**Analysis:**

好的，这是对论文“InfSplign: Inference-Time Spatial Alignment of Text-to-Image Diffusion Models”的全面中文摘要：

**论文题目：** InfSplign: 推理时文本到图像扩散模型的空间对齐

**作者：** Sarah Rastegar, Violeta Chatalbasheva, Sieger Falkena, Anuj Singh, Yanbo Wang, Tejas Gokhale, Hamid Palangi, Hadi Jamali-Rad

**摘要：**

**1. 主要问题/研究问题：**
文本到图像（T2I）扩散模型虽然能够生成高质量、逼真的图像，但在准确捕捉文本提示中描述的空间关系方面存在显著不足。这种局限性主要源于两个方面：训练数据缺乏细粒度的空间监督，以及文本嵌入本身难以有效编码空间语义。这导致模型在生成图像时，物体的位置、相对关系（如“左边”、“右边”、“上面”、“下面”）经常出错，甚至完全忽略。

**2. 关键创新/方法贡献：**
为了解决上述问题，作者提出了 **InfSplign**，一种**无需训练、在推理时**进行空间对齐的方法。其核心贡献在于：

*   **推理时空间对齐损失：** InfSplign 在每个去噪步骤中，通过引入一个复合损失函数来调整噪声预测，从而引导生成过程朝着更符合空间语义的方向发展。
*   **利用多层级交叉注意力图：** 该方法巧妙地从扩散模型 U-Net 解码器的不同层级（粗粒度、中粒度）提取交叉注意力图。这些注意力图被视为物体空间信息的代理。
*   **复合损失函数：** 损失函数包含三个关键组成部分：
    *   **空间对齐损失 (Lspatial)：** 通过计算物体在注意力图中的质心（centroid）差异，来惩罚违反文本提示中指定空间关系的情况。
    *   **物体存在损失 (Lpresence)：** 通过最小化物体注意力图的方差，来确保物体在生成的图像中清晰可见且不会被忽略。
    *   **表示平衡损失 (Lbalance)：** 通过平衡不同物体注意力图的方差，来防止一个物体过度主导而另一个物体被抑制，确保所有物体都能得到充分表示。
*   **轻量级、即插即用：** InfSplign 是一种轻量级的方法，不需要对预训练的扩散模型进行任何修改或重新训练，可以轻松地集成到任何现有的扩散模型（如 Stable Diffusion）中。

**3. 主要结果及其意义：**
作者在 **VISOR** 和 **T2I-CompBench** 这两个广泛使用的 T2I 空间理解基准上进行了全面评估。结果表明：

*   **显著性能提升：** InfSplign 在这两个基准上均取得了**最先进（state-of-the-art）的性能**。
*   **超越现有方法：** 相较于最强的现有推理时基线方法，InfSplign 在空间对齐方面取得了显著的性能提升（例如，在 VISOR-4 上提升高达 24.81%）。
*   **媲美甚至超越微调方法：** 令人印象深刻的是，InfSplign 甚至在性能上**超越了那些需要昂贵微调的先进方法**（例如，在 VISOR-4 上提升高达 14.33%）。
*   **定性结果：** 定性实验结果（如图 4-12 所示）直观地展示了 InfSplign 在生成具有准确空间关系和物体组合的图像方面的强大能力，尤其是在处理不常见物体组合时表现出色。

**4. 提及的局限性：**
论文中提到了一些局限性：

*   **罕见物体组合的挑战：** 对于自然场景中很少共同出现的物体组合，基础扩散模型本身就难以生成，InfSplign 在这种情况下也难以完全纠正空间对齐，因为物体本身可能就无法在图像中出现。在这种情况下，物体准确率（object accuracy）成为瓶颈。
*   **对物体存在性的依赖：** InfSplign 的有效性在一定程度上依赖于物体能够被成功生成和检测。如果物体本身就无法出现，那么其空间对齐就无从谈起。

**5. 潜在的未来研究方向：**
作者指出了未来的研究方向：

*   **扩展到 Transformer 架构：** 作者目前正在将 InfSplign 扩展到 Transformer 架构，并已取得初步成果。
*   **更复杂的空间关系：** 探索处理更复杂的多维空间关系，例如三维空间中的相对位置。

**总结：**
InfSplign 是一项重要的研究成果，它通过一种创新性的、无需训练的推理时方法，显著提升了文本到图像扩散模型在理解和生成空间关系方面的能力。该方法利用交叉注意力图的丰富信息，通过精心设计的复合损失函数，在保证物体存在和平衡表示的同时，实现了精确的空间对齐。其优异的性能和即插即用的特性，使其成为 T2I 领域一个非常有价值的贡献，为实现更具可控性和准确性的图像生成开辟了新的道路。

**Key Findings:**

- We introduce InfSplign, a training-free inference-time method that improves spatial alignment by adjusting the noise through a compound loss in every denoising step.
- Our comprehensive evaluations on VISOR and T2I-CompBench show that InfSplign establishes a new state-of-the-art (to the best of our knowledge), achieving substantial performance gains over the strongest existing inference-time baselines and even outperforming the fine-tuning-based methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17851v1)
- [arXiv](https://arxiv.org/abs/2512.17851v1)

---

