time: 20251222

# Arxiv Computer Vision Papers - 2025-12-22

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份关于2025年12月19日 Arxiv 计算机视觉领域论文的简明执行摘要。

---

**执行摘要：2025年12月19日 Arxiv 计算机视觉论文精选**

**日期：** 2025年12月19日

**主要趋势与主题：**

本期 Arxiv 论文集聚焦于**多模态理解与生成**、**模型鲁棒性与可解释性**，以及**高效的视觉任务解决方案**。特别值得注意的是，**文本到图像生成与编辑**的进一步深化，以及**自监督学习**在各种视觉任务中的广泛应用，包括深度估计和模型对齐。此外，**模拟到真实（Sim-to-Real）的鸿沟**的弥合以及**模型在开放环境下的鲁棒性**也成为研究热点。

**亮点与创新：**

*   **文本到图像生成与编辑的精细化：** "Both Semantics and Reconstruction Matter" 和 "InfSplign" 论文共同展示了在文本到图像生成和编辑领域，对语义理解和几何重建的同等重视，以及在推理时进行空间对齐的技术，预示着更精确、更可控的图像生成。
*   **自监督深度估计的突破：** "Re-Depth Anything" 提出了一种在测试时通过自监督重照明来精炼深度估计的方法，显示了在无需额外标注数据的情况下提升深度感知能力的潜力。
*   **可解释性与鲁棒性的新方法：** "Keypoint Counting Classifiers" 提出了一种无需训练即可将 Vision Transformers 转化为自解释模型的方法，而 "Adversarial Robustness of Vision in Open Foundation Models" 则深入探讨了开放基础模型在对抗攻击下的鲁棒性问题，这对于模型的安全部署至关重要。
*   **多模态融合的创新应用：** "RadarGen" 实现了从摄像头数据生成汽车雷达点云，为自动驾驶中的多传感器融合提供了新的思路。

**新兴研究方向与技术：**

*   **多模态生成与编辑的深度融合：** 结合语义理解、几何重建和推理时对齐，以实现更精细的文本到图像生成和编辑。
*   **自监督学习在各种视觉任务中的泛化应用：** 从深度估计到模型对齐，自监督方法正成为减少对标注数据依赖的关键。
*   **开放世界基础模型的鲁棒性与安全性：** 关注模型在复杂、不可控环境下的表现，以及对抗攻击的防御策略。
*   **模拟到真实（Sim-to-Real）的自动化与泛化：** 开发自动化框架以生成多样化的任务和数据，加速机器人和自动驾驶等领域的策略学习。
*   **模型的可解释性与透明度：** 探索无需额外训练即可实现模型自解释的方法，增强模型的信任度。

**建议阅读论文：**

基于其潜在影响和创新性，以下论文值得深入阅读：

1.  **"Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing"**: 对于理解当前文本到图像生成和编辑技术的核心挑战以及未来的发展方向至关重要。
2.  **"Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting"**: 在自监督深度估计领域具有显著的创新性，可能对三维重建和场景理解产生广泛影响。
3.  **"Adversarial Robustness of Vision in Open Foundation Models"**: 对于关注模型安全性和可靠性的研究人员来说，这篇论文提供了对当前基础模型脆弱性的重要见解。
4.  **"Keypoint Counting Classifiers: Turning Vision Transformers into Self-Explainable Models Without Training"**: 提供了实现模型可解释性的新颖且高效的方法，对于理解和信任深度学习模型具有重要意义。

---

希望这份执行摘要能帮助您快速了解该领域的最新进展。

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

---

**摘要：**

这篇论文旨在解决将强大的视觉理解编码器（representation encoders）应用于文本到图像生成和图像编辑任务时遇到的挑战。当前主流的生成模型（如 Latent Diffusion Models, LDMs）通常依赖于低维度的变分自编码器（VAE）的潜在空间，这些空间主要针对像素级重建进行优化。然而，直接使用高维度的、为理解任务设计的表示编码器的特征作为生成潜在空间存在两个主要问题：

1.  **缺乏紧凑的正则化：** 表示编码器的特征空间维度高但内在信息量相对较低，缺乏有效的正则化，导致扩散模型容易生成“离流形”（off-manifold）的潜在向量，从而产生结构不准确或失真的对象。
2.  **像素级重建能力弱：** 表示编码器本身通常不优化像素级重建，其输出的特征丢失了精细的几何和纹理细节，这阻碍了生成器学习准确的细节。

为了克服这些障碍，论文提出了一种系统性的框架，将理解导向的编码器特征适配到生成任务中。

**核心创新与方法贡献：**

1.  **语义-像素重建目标：** 论文引入了一个创新的“语义-像素重建”（semantic-pixel reconstruction）目标。该目标首先通过一个语义自编码器（S-VAE）将高维度的、无约束的表示特征映射到一个紧凑的、经过 KL 散度正则化的潜在空间（例如，96通道，16x16空间分辨率）。这解决了离流形问题，并保留了丰富的语义信息。
2.  **联合优化与精细化：** 在此基础上，论文进一步解冻表示编码器，并联合优化一个像素级重建损失和一个语义重建损失。这使得表示编码器在捕获高层语义的同时，也能学习保留输入图像的精细几何和纹理细节。最终的模型被称为 **Pixel-Semantic VAE (PS-VAE)**。
3.  **统一的生成架构：** 利用 PS-VAE 产生的紧凑且语义丰富的潜在空间，论文设计了一个统一的文本到图像（T2I）和图像编辑模型。

**主要结果与意义：**

*   **卓越的重建性能：** PS-VAE 在图像重建任务上达到了最先进的性能，显著优于其他基于表示编码器的生成方法，并且在某些方面可以媲美甚至超越专门为重建设计的 VAE。
*   **更快的收敛速度与更强的生成能力：** 在文本到图像生成任务上，PS-VAE 展现出更快的收敛速度和更优越的最终性能，优于 RAE 等基线模型。
*   **显著提升的图像编辑能力：** 在需要精确理解指令和保留图像细节的图像编辑任务上，PS-VAE 取得了大幅度的性能提升，显著优于仅依赖像素重建或仅依赖语义的基线模型。这表明其结合了语义理解和细节保留的能力。
*   **统一的编码器潜力：** 论文证明了通过 PS-VAE 优化的表示编码器可以作为视觉理解和生成任务的统一编码器，为未来构建更通用的视觉模型提供了方向。

**论文中提到的局限性：**

*   **模型容量与细节权衡：** 论文提到，虽然 96 通道的 PS-VAE 提供了良好的重建质量，但在生成指标上略逊于 32 通道的版本，这可能是因为过多的通道容量在建模精细细节时可能消耗模型能力，并干扰语义学习。
*   **LLM 微调的潜力：** 在将 SigLIP2 作为统一编码器进行评估时，论文指出，在没有对 LLM 进行任何微调的情况下，PS-VAE 已经表现出色，但进一步的 LLM 微调可能带来更优越的性能。

**潜在的未来研究方向：**

*   **更高分辨率的生成：** 论文提到，将 PS-VAE 应用于更高分辨率的生成任务将进一步提升其能力。
*   **LLM 与统一编码器的联合训练：** 探索 LLM 与经过 PS-VAE 优化的统一编码器进行联合训练，以期获得超越当前基线模型的性能。
*   **更深入的架构探索：** 对编码器和解码器的架构进行更深入的研究，以优化计算效率和性能。
*   **更广泛的预训练模型适配：** 探索将 PS-VAE 框架应用于更多不同类型的预训练表示编码器。

**总结：**

这篇论文成功地解决了将表示编码器应用于生成任务的关键挑战，通过引入创新的语义-像素重建目标，实现了对潜在空间的有效正则化和精细细节的保留。其提出的 PS-VAE 模型在图像重建、文本到图像生成和图像编辑等多个任务上均取得了显著的性能提升，并展示了其作为统一视觉理解和生成编码器的巨大潜力。这项工作为构建更强大、更通用的视觉模型提供了重要的理论和实践基础。

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
单目深度估计（Monocular Depth Estimation, MDE）在处理与训练数据分布差异较大的真实世界图像时，即使是先进的基础模型（如 Depth Anything V2, DA-V2），也常常表现出不准确性。这种“域间隙”（domain gap）导致预测的深度图在细节、真实感等方面存在不足。

**2. 主要创新点/方法贡献：**
本文提出了一种名为 **Re-Depth Anything** 的新颖的 **测试时（test-time）自监督框架**，旨在弥合这一域间隙。其核心创新在于：

*   **融合2D扩散模型先验：** 将 DA-V2 的几何推理能力与大型2D扩散模型（如 Stable Diffusion）强大的图像生成先验相结合。
*   **基于重照明（Re-lighting）的自监督方法：** 引入了一种新颖的“重合成”（re-synthesis）方法，通过随机改变输入图像的光照条件来生成新的视角，从而替代了传统的基于光度重建（photometric reconstruction）的自监督方法。这种方法利用了**形状来自阴影（Shape from Shading, SfS）**的线索，并在**生成式上下文**中结合**得分蒸馏采样（Score Distillation Sampling, SDS）**损失来实现。
*   **目标化优化策略：** 为了防止优化崩溃并保留预训练模型的几何知识，该框架采用了**目标化优化策略**。具体来说，它**冻结了编码器**，仅更新**中间特征嵌入（intermediate embeddings）**和**解码器（decoder）的权重**。
*   **可微分重照明渲染器：** 开发了一个可微分的渲染器，能够将预测的深度图与输入图像联系起来，实现基于 SDS 损失的几何细化。

**3. 主要结果与意义：**
Re-Depth Anything 在多个基准数据集（CO3D, KITTI, ETH3D）上均取得了显著的性能提升，**在深度准确性和真实感方面均超越了 DA-V2 基线模型**。具体而言，该方法能够：

*   **增强细节：** 显著提升了精细几何细节的恢复，例如物体边缘、纹理等。
*   **去除噪声和伪影：** 有效地消除了 DA-V2 在平坦区域产生的噪声，并纠正了不准确的预测。
*   **修正偏差：** 能够修正因训练数据偏差导致的错误预测，例如将狗的形状修正为更像老虎的形状（如图1所示）。
*   **泛化能力强：** 即使在处理与训练数据分布差异较大的图像时，也能取得良好的效果，证明了其强大的泛化能力。

这项工作展示了利用2D生成模型进行自监督几何推理的新途径，为提升单目深度估计在复杂场景下的性能提供了新的解决方案。

**4. 提及的局限性：**
*   **细微伪影：** 在某些情况下，可能会观察到小的幻觉边缘（hallucinated edges），例如卡车上的贴纸。
*   **过度平滑：** 在某些区域，方法可能会过度平滑细节，例如在暗部区域的树木，或者在某些场景中出现轻微的过度平滑（如图10和图11所示）。
*   **天空区域幻觉：** 在 KITTI 数据集中，有时会在天空区域出现幻觉（如图12所示）。
*   **对相机模型和参数的敏感性：** 虽然作者进行了消融实验，但相机模型（如正交投影 vs. 透视投影）和参数（如 b 值）的选择仍然会影响最终结果。

**5. 未来研究方向：**
*   **探索替代的重合成方法：** 除了重照明，还可以探索其他方式来生成多样的训练信号。
*   **大规模模型微调：** 计划在更大规模的真实世界视频数据上探索微调基础模型。
*   **处理更复杂的场景：** 进一步研究如何处理更具挑战性的场景，例如包含复杂光照、反射和透明物体的场景。
*   **结合更精细的相机模型：** 在已知相机参数的情况下，探索更精细的相机模型以获得更准确的深度估计。

总而言之，Re-Depth Anything 是一项重要的研究工作，它通过创新的测试时自监督重照明方法，有效解决了现有单目深度估计模型在处理真实世界图像时的域间隙问题，显著提升了深度估计的准确性和真实感，并为未来的研究开辟了新的方向。

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

好的，作为计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Dexterous World Models**

**1. 论文的主要贡献（2-3句话）**

该论文提出了 Dexterous World Model (DWM)，一个创新的场景-动作条件视频扩散框架，能够模拟人类灵巧动作如何引起静态三维场景的动态变化。DWM 能够根据静态三维场景渲染和以自我为中心的（egocentric）手部运动序列，生成时间连贯且具有物理合理性的人体与场景交互视频。这标志着迈向基于视频扩散的交互式数字孪生和从以自我为中心动作进行具身仿真的重要一步。

**2. 关键创新或方法论**

DWM 的核心创新在于其**场景-动作条件视频扩散框架**，以及如何有效地将这些条件融入视频生成过程：

*   **条件化视频生成：** DWM 巧妙地将视频生成过程条件化在两个关键要素上：
    *   **静态场景渲染（Spatial Consistency）：** 通过指定相机轨迹的静态三维场景渲染，确保了生成视频在空间上的连贯性和一致性。这意味着生成的交互不会脱离预设的场景几何和视角。
    *   **以自我为中心的手部网格渲染（Action-Conditioned Dynamics）：** 这是该方法的一个重要亮点。通过输入编码了几何和运动信息的手部网格渲染，DWM 直接将动作的动态信息注入到视频生成中。这种方式比仅仅依赖文本描述或高级动作标签更精细，能够捕捉到更细微、更具“灵巧性”的手部操作。
*   **混合交互视频数据集：** 为了训练这样一个复杂的模型，作者构建了一个创新的混合数据集。
    *   **合成以自我为中心的交互：** 提供完全对齐的监督信号，用于联合学习身体运动（locomotion）和物体操作（manipulation）。这使得模型能够学习到精确的因果关系。
    *   **固定摄像机的真实世界视频：** 引入了多样性和真实感，捕捉了现实世界中物体动态的复杂性，弥补了纯合成数据的不足。

**3. 对该领域的潜在影响**

DWM 的研究对计算机视觉领域具有重要的潜在影响：

*   **推动具身智能（Embodied AI）的发展：** 该框架为构建更逼真、更具交互性的具身智能体提供了基础。能够从以自我为中心的视角模拟和预测人类的交互行为，是实现智能体在复杂环境中进行自主操作的关键。
*   **提升三维场景理解和生成能力：** DWM 不仅生成视频，更重要的是它在学习“场景-动作-动态变化”之间的因果关系。这有助于更深入地理解三维场景的物理属性以及人类如何与之交互。
*   **加速数字孪生（Digital Twins）的应用：** 当前的数字孪生多为静态，DWM 的工作是实现真正意义上的“交互式数字孪生”的关键一步。这将极大地扩展数字孪生的应用范围，例如在虚拟现实（VR）、增强现实（AR）中的沉浸式体验，以及在机器人训练、远程操作等领域。
*   **视频生成技术的进步：** 将扩散模型应用于更复杂的、条件化的视频生成任务，特别是涉及精细的物理交互和多模态条件（场景几何、手部动作），是视频生成领域的一个重要突破。

**4. 可能受益的相关领域或应用**

*   **虚拟现实（VR）/增强现实（AR）：** 创建更逼真、更具交互性的虚拟环境，用户可以更自然地与虚拟物体互动。
*   **机器人学：** 训练机器人进行精细操作，通过模拟人类的灵巧动作来学习和优化抓取、装配等任务。
*   **游戏开发：** 生成更真实的虚拟角色与环境的交互动画，提升游戏体验。
*   **影视制作：** 辅助生成复杂的交互场景动画，降低制作成本。
*   **人机交互（HCI）：** 设计更直观、更自然的交互方式，尤其是在需要精细操作的场景。
*   **物理仿真：** 为复杂的物理交互提供更逼真的视觉输出，用于研究和验证。

**5. 从摘要中可以推断出的局限性**

尽管摘要描绘了一个令人兴奋的框架，但仍可以推断出一些潜在的局限性：

*   **计算成本：** 扩散模型通常计算成本较高，尤其是在生成长序列或高分辨率视频时。训练和推理的效率可能是一个挑战。
*   **泛化能力：** 模型在多大程度上能够泛化到训练数据中未见过的场景、物体或动作类型，仍需进一步验证。例如，对于非常规的物体形状或极其复杂的、非人类标准的动作，模型表现如何？
*   **物理真实性的精确度：** 尽管摘要提到“物理上可行”，但“物理上精确”是另一个层面的要求。模型生成的交互是否能完全符合所有物理定律（如摩擦力、惯性、形变等）的精确模拟，可能存在一定差距。
*   **数据依赖性：** 模型的性能高度依赖于训练数据的质量和多样性。混合数据集的构建虽然有创新，但其覆盖范围和真实性仍是关键。
*   **“灵巧性”的定义和捕捉：** “灵巧”是一个相对概念。摘要中提到“灵巧的人类动作”，但模型如何精确地捕捉和复现人类手部动作的细微之处（如手指的精细配合、触觉反馈的模拟等）可能仍有待深入研究。
*   **场景复杂性：** 摘要提到“静态三维场景”，但对于包含大量动态元素、复杂光照或高度遮挡的场景，模型的表现可能受到影响。

总而言之，Dexterous World Models 是一项非常有前景的研究，它通过创新的条件化视频扩散方法，有效地连接了静态三维场景和动态的以自我为中心的人类交互，为具身智能和交互式数字孪生领域开辟了新的可能性。

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

**作者：** Jonathon Fox, William J Buchanan, Pavlos Papadopoulos

**摘要**

**1. 主要问题/研究问题：**
随着深度学习模型在识别物体方面的能力日益增强，理解这些模型的内部工作机制变得愈发困难。这为攻击者提供了机会，他们可以通过修改图像来引入难以察觉的元素，从而欺骗AI系统，使其无法正确识别物体。本文旨在研究当前流行的开放权重视觉语言模型（VLMs）在视觉输入受到对抗性攻击时的鲁棒性。具体来说，研究关注的是 LLaVA-1.5-13B 和 Meta 的 Llama 3.2 Vision-8B-2 这两个模型，它们在视觉输入模态上，使用无目标投影梯度下降（PGD）攻击下的表现。

**2. 关键创新/方法论贡献：**
*   **模型选择与对比：** 本研究选择了两个具有代表性的开放权重 VLM 进行对比分析：LLaVA-1.5-13B（一个已建立的 VLM 架构）和 Llama 3.2 Vision-8B-2（Meta 最新、采用适配器方法的模型）。这种选择允许研究者探究不同架构和训练规模对模型鲁棒性的影响。
*   **对抗攻击方法：** 采用了无目标投影梯度下降（PGD）攻击，这是一种广泛认可且强大的对抗攻击方法，用于评估模型的视觉输入鲁棒性。
*   **评估指标与数据集：** 使用了标准的视觉问答（VQA）准确率作为评估指标，并在 VQA v2 数据集的子集上进行了实证评估。通过比较模型在干净图像和对抗性图像上的准确率下降（accuracy drop）来量化其鲁棒性。
*   **细致的超参数控制：** PGD 攻击的超参数（如扰动预算 ε、步长 α 和迭代次数）被精心调整，以确保攻击强度与预算相匹配，从而进行公平的比较。

**3. 主要结果及其意义：**
*   **普遍的脆弱性：** 研究发现，两个模型都对视觉输入的对抗性攻击表现出明显的脆弱性。即使是细微的、近乎不可察觉的扰动（ε < 16/255），也会导致 VQA 准确率的下降。
*   **不同的鲁棒性表现：**
    *   LLaVA-1.5-13B 在干净数据集上表现出更高的基线准确率（87.4%），但在对抗性攻击下，其准确率随扰动增大而显著下降，最大下降幅度达到 36.0 个百分点。
    *   Llama 3.2 Vision-8B-2 的基线准确率较低（42.8%），但在对抗性攻击下，其准确率下降幅度相对较小，尤其是在高扰动水平下（最大下降 10.2 个百分点，相对于其在该运行中的 41.6% 基线）。这表明 Llama 3.2 Vision 在面对视觉对抗性攻击时，表现出更强的相对鲁棒性。
*   **架构与训练的影响：** 研究推测，Llama 3.2 Vision 更复杂的跨注意力适配器机制、更大的预训练数据集以及可能更先进的对齐过程，可能有助于其更稳定的内部表示，从而在对抗性攻击下表现出更好的鲁棒性。
*   **重要发现：**
    *   视觉模态是降级当前开放权重 VLM 性能的可行攻击向量。
    *   对抗鲁棒性并不一定直接与标准基准性能相关，它可能受到底层架构和训练因素的影响。

**4. 论文中提到的局限性：**
*   **数据集规模：** 由于计算资源的限制，实验仅在 VQA v2 数据集的 500 个样本子集上进行。在完整数据集或其他数据集上的结果可能有所不同。
*   **攻击类型限制：** 研究仅关注了无目标 PGD 攻击和 L∞ 范数约束。其他攻击算法（如 Carlini & Wagner 攻击）或范数约束（L2, L∞）可能会揭示不同的脆弱性。
*   **目标攻击缺失：** 攻击是无目标的，旨在降低整体性能，而非诱导特定的错误输出。有针对性的攻击可能带来不同的挑战。
*   **任务范围限制：** 鲁棒性仅在 VQA 任务上进行了评估。在图像描述或复杂推理等其他多模态任务上的性能下降可能有所不同。
*   **超参数探索有限：** 尽管努力使用了合适的超参数，但计算成本限制了对 PGD 参数空间（迭代次数、步长）的详尽探索。

**5. 未来研究方向：**
*   **更全面的评估：** 在更大的数据集子集或完整的 VQA v2 数据集上重复实验，并扩展到其他多模态基准测试，以获得更全面的鲁棒性评估。
*   **多样化的攻击方法：** 探索更广泛的攻击算法，包括 Carlini & Wagner (CW) 攻击、Image Hijacks 等，以及有针对性的攻击，以更全面地评估模型的安全性。
*   **性能差异的深入分析：** 深入探究 Llama 3.2 Vision 在 VQA 子集上基线性能较低的原因，可能涉及不同的提示配置或预处理步骤。
*   **架构与训练的细致分析：** 更深入地分析特定组件（如多模态适配器的设计）和训练阶段（预训练数据规模、对齐技术如 RLHF）如何量化影响对抗鲁棒性。
*   **防御策略的研究：** 开发和评估专门针对 VLM 的有效防御策略，以减轻视觉对抗性攻击带来的风险。
*   **原生多模态 vs. 适配器方法：** 进一步研究原生多模态架构与适配器方法的对比，例如 Meta 的 Llama 4 等新架构，以理解它们在鲁棒性方面的影响。

**总结：**
本文通过实证研究，揭示了当前开放权重视觉语言模型（LLaVA 和 Llama 3.2 Vision）在视觉输入受到对抗性攻击时的脆弱性。研究强调了视觉模态作为攻击向量的重要性，并指出对抗鲁棒性并非总是与标准性能指标直接相关，而是受到模型架构、训练规模和方法等多种因素的影响。研究结果为理解和提升这些强大模型的安全性提供了重要的见解，并为未来的研究指明了方向。

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

**1. 论文的主要贡献（2-3句话）**

本研究提出了RadarGen，一个创新的扩散模型，能够从多视角摄像头图像生成逼真的汽车雷达点云。该模型通过将雷达测量值表示为鸟瞰图（BEV）形式，并融合视觉线索，实现了从图像到雷达点云的跨模态生成，为多模态生成式仿真提供了一个可扩展的解决方案。

**2. 关键创新或方法论**

RadarGen的核心创新在于其**将高效的图像-潜在扩散模型（Image-Latent Diffusion）成功适配到雷达点云生成领域**。具体而言，其方法论的关键点包括：

*   **雷达数据表示的创新：** 将雷达测量值（包括空间结构、雷达截面积 RCS 和多普勒信息）编码到鸟瞰图（BEV）表示中。这种BEV表示能够有效地捕捉雷达数据的空间分布和物理属性。
*   **轻量级恢复步骤：** 在生成BEV图之后，采用一个轻量级的恢复步骤将其转换回三维雷达点云。这使得模型在生成逼真雷达数据的同时，保持了计算效率。
*   **多模态融合与引导：** 引入了从预训练基础模型中提取的BEV对齐的深度、语义和运动线索。这些视觉线索作为生成过程的引导，确保生成的雷达模式在物理上是合理的，并与视觉场景更加一致。
*   **条件生成能力：** 通过以摄像头图像作为条件进行生成，RadarGen具有广泛的兼容性，可以与现有的视觉数据集和仿真框架集成，为多模态生成式仿真开辟了新的途径。

**3. 对该领域的潜在影响**

RadarGen的潜在影响是深远的，主要体现在以下几个方面：

*   **推动多模态生成式仿真：** 长期以来，为自动驾驶系统生成逼真的传感器数据一直是研究的重点。RadarGen的出现，使得从易于获取的摄像头数据生成雷达点云成为可能，极大地降低了生成高质量、多模态仿真数据的门槛。这对于训练和评估自动驾驶感知算法，尤其是在数据稀缺或危险场景下，具有重要意义。
*   **提升雷达感知模型的鲁棒性：** 通过生成大量逼真的雷达数据，可以用于扩充训练数据集，从而提高雷达感知模型的泛化能力和鲁棒性，使其在各种真实世界条件下表现更佳。
*   **促进跨传感器数据融合研究：** RadarGen为研究不同传感器（如摄像头和雷达）之间的关联性提供了新的工具。通过生成同步的、相互一致的跨模态数据，可以更好地探索和优化多传感器融合算法。
*   **加速自动驾驶系统的开发和测试：** 逼真的仿真数据能够加速自动驾驶系统的开发周期，允许在虚拟环境中进行更广泛、更安全的测试，从而更快地迭代和改进算法。

**4. 可能受益的相关领域或应用**

*   **自动驾驶感知系统开发与测试：** 这是最直接的应用领域，用于生成训练和测试雷达目标检测、跟踪、分类等算法的数据。
*   **机器人导航与感知：** 机器人也常常依赖雷达进行环境感知和导航，RadarGen可以为机器人领域提供逼真的雷达数据仿真。
*   **计算机视觉中的生成模型研究：** 该研究将扩散模型成功应用于雷达点云这一非传统数据类型，为跨模态生成和扩散模型在不同领域的应用提供了新的思路。
*   **虚拟现实（VR）和增强现实（AR）：** 在构建逼真的虚拟环境时，能够模拟不同传感器的输出将有助于提升沉浸感和交互性。
*   **遥感和测绘：** 虽然论文聚焦于汽车雷达，但其核心技术可能可以推广到其他类型的雷达数据生成，例如用于遥感和测绘的雷达数据。

**5. 从摘要中可以推断出的局限性**

尽管摘要展示了RadarGen的强大能力，但仍可以推断出一些潜在的局限性：

*   **对预训练基础模型的依赖：** RadarGen依赖于预训练的视觉基础模型来提取深度、语义和运动线索。这些基础模型的性能和泛化能力将直接影响RadarGen的生成质量。如果基础模型在特定场景下表现不佳，可能会导致生成的雷达数据不准确。
*   **“逼真性”的定义和评估：** 摘要提到“逼真性”和“捕捉特征雷达测量分布”，但“逼真”的定义可能是一个主观或相对的概念。如何全面、客观地量化生成的雷达点云的逼真度，以及与真实雷达数据的差距，仍需要进一步的深入评估。
*   **计算资源需求：** 扩散模型通常计算量较大，尽管摘要提到了“高效的图像-潜在扩散”，但生成高质量的雷达点云可能仍然需要可观的计算资源，尤其是在大规模数据集上进行训练和推理时。
*   **对特定雷达类型的适配性：** 摘要主要关注“汽车雷达点云”。RadarGen是否能直接适用于其他类型的雷达（如气象雷达、SAR雷达等）或不同配置的汽车雷达，可能需要进一步的研究和调整。
*   **“轻量级恢复步骤”的性能边界：** 虽然恢复步骤是轻量级的，但其性能上限可能会影响最终点云的细节和精度。如果恢复步骤丢失了过多的信息，生成的点云可能不够精细。

总而言之，RadarGen是一项令人兴奋的研究，它巧妙地将先进的生成模型技术应用于跨模态传感器数据生成，为自动驾驶和相关领域的研究与开发带来了巨大的潜力。其核心在于将扩散模型的能力扩展到雷达领域，并利用视觉信息进行有效引导，从而实现从图像到雷达点云的高质量生成。

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

**全面摘要**

**1. 研究问题/核心挑战：**

当前设计自解释模型（Self-Explainable Models, SEMs）的方法通常需要复杂的训练流程和特定的模型架构，这使得它们在实践中难以应用，尤其是在通用性强的 Vision Transformers (ViTs) 基础模型日益普及的背景下，这一问题更为突出。现有的 SEMs 在灵活性和可视化解释方面存在局限性，例如它们通常依赖于卷积神经网络（CNN）架构，或者需要额外的训练来适应 ViTs，并且其可视化解释（如边界框和热力图）常常不够精确或信息量不足。因此，研究界迫切需要一种能够为 ViT 基础模型提供透明度和可靠性的新方法，同时保持其灵活性。

**2. 关键创新/方法论贡献：**

该论文提出了一种名为**关键点计数分类器（Keypoint Counting Classifiers, KCCs）**的新方法，旨在将任何已训练好的 ViT 基础模型转化为自解释模型，而**无需进行任何额外的训练**。KCCs 的核心创新在于：

*   **无需训练的转换：** KCCs 能够直接利用预训练的 ViT 模型，无需修改其特征提取器或训练额外的分类头，从而极大地提高了灵活性。
*   **基于关键点的解释：** KCCs 利用 ViT 的内部表示（tokens）来自动识别图像中的语义部分，并将这些部分视为“关键点”。通过比较查询图像与原型图像之间的关键点匹配，生成直观且可解释的决策过程。
*   **关键点匹配与计数：** 该方法通过计算查询图像和原型图像之间关键点的互为最近邻（Mutual Nearest Neighbors, MNNs）来识别匹配的语义区域。最终的分类决策是通过计数匹配到的关键点数量来完成的。
*   **可视化解释：** KCCs 生成的可视化解释是匹配的关键点，这是一种全新的可视化方式，旨在提高解释的直观性和用户理解。
*   **利用 Vision-Language 能力（可选）：** 在结合了具有视觉-语言能力的 ViTs 时，KCCs 可以自动为关键点生成文本描述，进一步减少读者的主观偏见。

KCCs 的实现过程分为三个主要部分：
    a. **图像关键点识别：** 利用 ViT 的 tokens 来识别图像中的语义部分，并将每个部分的中心点定义为关键点。
    b. **匹配关键点识别：** 使用互为最近邻（MNNs）的方法来寻找查询图像和原型图像之间匹配的关键点。
    c. **通过计数分类：** 根据匹配到的关键点数量来决定最终的分类结果。

**3. 主要结果与意义：**

*   **用户研究结果：** 用户研究表明，KCCs 在解释质量和理解性方面显著优于现有的基线方法（如 PIP-Net 和 KMEx）。用户对 KCCs 的解释感到更自信，并且在纠正模型错误预测时更有信心。KCCs 在用户偏好方面与 KMEx 并列第一。
*   **定量评估结果：** 在 CUB200、CARS 和 PETS 等数据集上的定量评估显示，KCCs 在准确性和复杂度方面与一些需要训练的 SEMs 相当，甚至在某些情况下表现更优，例如在 CUB200 数据集上，KCCs 的准确率高于 ProtoPNet，尽管 KCCs 未经训练。
*   **意义：** KCCs 提供了一种新颖的范式，能够将强大的 ViT 基础模型转化为易于理解和解释的自解释模型，而无需额外的训练成本。这对于提高模型的可信度、透明度和用户交互性具有重要意义，尤其是在对安全性要求高的领域。

**4. 提及的局限性：**

*   **关键点权重问题：** 目前 KCCs 中所有关键点被赋予相同的权重，但实际上某些关键点（如鸟类的喙部形状）可能对特定类别的识别更重要。如何有效地为关键点分配权重是一个潜在的研究方向。
*   **计算复杂度：** 当原型数量非常多时，计算相似度可能会变得内存密集。论文中提到通过仅考虑距离最近的几个原型来缓解这个问题。
*   **细粒度分类的挑战：** 论文提到，在细粒度分类任务中，尽管 KCCs 表现良好，但仍然需要监督信号来达到最佳性能。

**5. 潜在的未来研究方向：**

*   **加权关键点：** 探索如何为关键点引入权重，以更好地反映其在分类中的重要性。
*   **识别类特异性关键点：** 研究如何识别并利用类特异性的关键点，以进一步提高解释的精确度。
*   **结合更先进的 Vision-Language 模型：** 进一步探索如何利用最新的视觉-语言模型来增强 KCCs 的解释能力，例如生成更丰富、更具上下文的文本描述。
*   **应用于更多任务：** 将 KCCs 的方法扩展到其他计算机视觉任务，如目标检测、分割等。

总而言之，这篇论文提出了一种名为 KCCs 的创新方法，成功地将预训练的 ViT 模型转化为无需训练的自解释模型。通过利用关键点匹配和计数，KCCs 提供了直观、可解释的决策过程，并在用户研究和定量评估中展现出优越的性能和用户体验，为提高深度学习模型的透明度和可靠性开辟了新的途径。

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

**论文题目：** Visually Prompted Benchmarks Are Surprisingly Fragile

**作者：** Haiwen Feng, Long Lian, Lisa Dunlap, Jiahao Shu, XuDong Wang, Renhao Wang, Trevor Darrell, Alane Suhr, Angjoo Kanazawa

**摘要：**

**1. 主要问题/研究问题：**
这篇论文的核心问题在于，当前用于评估视觉语言模型（VLMs）的视觉提示（visually prompted）基准测试存在显著的脆弱性。这些基准测试通过在图像中标记特定区域并提出相关问题来评估模型的视觉感知能力，旨在独立于文本先验知识。然而，研究发现，即使是视觉提示中看似无关紧要的细节，例如标记的颜色、大小、形状或位置，以及数据集的大小和低级别的推理设置（如JPEG压缩），都可能极大地影响模型的性能和排行榜排名。这种脆弱性使得当前的评估结果可能更多地反映了基准测试的设计细节而非模型真正的感知能力。

**2. 关键创新/方法贡献：**
*   **系统性地揭示视觉提示的脆弱性：** 作者通过在多个常用的开源和闭源VLM上进行实验，系统地量化了视觉提示设计（如标记样式）和基准测试设置（如数据集大小、JPEG压缩）对模型性能和排名的影响。
*   **引入VPBench基准测试：** 为了解决现有基准测试的脆弱性问题，作者创建了一个名为VPBench的新基准测试。VPBench是一个扩展的视觉提示基准，包含16种不同的视觉标记变体，旨在提供更稳定和可靠的评估。
*   **提供分析工具和建议：** 作者不仅发布了VPBench数据集，还提供了相应的推理代码，支持不同的视觉标记和图像压缩设置，以帮助研究人员进行更稳健的评估。同时，论文也提出了标准化和多样化视觉提示、使用一致的低级别设置以及报告不确定性和排名稳定性等建议。

**3. 主要结果及其意义：**
*   **微小变化导致排名剧烈波动：** 研究发现，仅仅改变视觉标记的颜色（如从红色变为蓝色）就可能完全改变模型在排行榜上的排名。这种现象在视觉提示任务中尤为明显，其影响远大于传统语义评估任务。
*   **标记样式和数据集大小是关键因素：** 作者证明了视觉标记的设计（如大小、形状、颜色、文本位置）和数据集的大小对模型性能和排名有显著影响。甚至可以通过策略性地选择标记样式来“操纵”排行榜，使得较弱的模型（如InternVL3-8B）能够超越更强大的模型（如Gemini 2.5 Pro）。
*   **低级别推理设置也影响排名：** 即使是人类难以察觉的低级别推理设置，如JPEG压缩级别，也能导致模型排名的变化。
*   **意义：** 这些发现表明，当前许多VLM基准测试的评估结果可能受到非语义因素的干扰，而非模型真正的视觉理解能力。这削弱了对现有排行榜的信任，并强调了开发更稳健、更可靠的评估方法的重要性。VPBench的发布为解决这一问题提供了一个重要的工具。

**4. 提及的局限性：**
*   **VPBench的局限性：** 虽然VPBench旨在提高稳定性，但它目前主要集中在相对深度估计和语义对应任务上。
*   **模型对特定提示的过拟合：** 研究暗示，一些模型可能对特定视觉提示（如默认的红色圆圈）存在过拟合现象，导致在其他提示下性能急剧下降。
*   **评估的复杂性：** 即使在VPBench中，不同模型对不同标记的反应仍然存在个体差异，表明模型在视觉提示处理方面存在内在的偏见。

**5. 潜在的未来研究方向：**
*   **更广泛的视觉提示任务和数据集：** 将VPBench的理念扩展到更多类型的视觉提示任务，并构建更大规模、更多样化的数据集。
*   **鲁棒性评估的标准化：** 进一步研究和标准化用于评估VLM鲁棒性的方法，包括对各种图像扰动和提示变化的敏感性。
*   **模型内在偏见的分析：** 深入研究模型为何对特定的视觉提示表现出不同的反应，以及如何减轻这种对特定提示的偏见。
*   **开发更少依赖特定提示的评估方法：** 探索不依赖于显式视觉标记的评估范式，或者能够自动适应不同提示的评估方法。
*   **结合人类感知进行评估：** 进一步探索如何将人类的视觉感知能力与VLM的评估相结合，以获得更具参考价值的评估结果。

总而言之，这篇论文通过揭示视觉提示基准测试的脆弱性，为VLM评估领域带来了重要的警示和贡献。它不仅指出了当前评估方法中存在的关键问题，还通过引入VPBench等工具和提出改进建议，为未来更可靠、更有意义的VLM评估奠定了基础。

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

本论文提出了一种名为 InSPECT 的新型扩散模型，其核心贡献在于在扩散模型的正向和反向过程中保持了不变的谱特征。通过使傅里叶系数平滑地收敛到指定的随机噪声，InSPECT 在保留图像特征的同时，实现了更高的视觉多样性、更快的收敛速度和更平滑的生成过程。实验结果表明，InSPECT 在生成质量和多样性方面显著优于现有的 DDPM 模型，并提高了计算效率。

**2. 关键创新点或方法论**

InSPECT 的关键创新点在于其对**不变谱特征（Invariant Spectral Features）**的引入和保持。传统的扩散模型（如 DDPM）将数据完全扩散到白噪声，这使得从噪声中重建原始数据成为一个极其困难且计算量巨大的任务。InSPECT 通过以下方式克服了这一限制：

*   **谱特征的显式处理：** 论文的核心思想是识别并保留图像在频域（例如通过傅里叶变换）中的关键谱特征。这些特征可能代表了图像的结构、纹理等本质信息。
*   **前向和反向过程中的不变性：** InSPECT 确保这些谱特征在数据扩散（前向过程）和噪声重建（反向过程）的每一步都保持相对稳定或以可控的方式演变。
*   **平滑收敛到噪声：** 论文提到，傅里叶系数平滑地收敛到指定的随机噪声。这意味着模型并没有完全破坏所有信息，而是以一种保留关键谱结构的方式进行扩散，从而为反向重建提供了更有用的起点。
*   **“不变性”的定义：** 虽然摘要没有详细说明“不变性”的具体数学定义，但可以推断，这是一种在扩散过程中对某些频率分量或其组合的鲁棒性，使得它们在噪声扰动下依然能够被识别和重建。

**3. 对该领域的潜在影响**

InSPECT 的研究对扩散模型领域具有重要的潜在影响：

*   **提升生成质量和多样性：** 通过保留关键的谱特征，模型能够生成更逼真、细节更丰富且具有更高视觉多样性的图像，这对于图像生成任务至关重要。
*   **提高计算效率和收敛速度：** 摘要中提到的“更快的收敛速度”和“计算效率”是巨大的优势。这意味着在更少的迭代次数下就能达到更好的生成效果，从而降低了训练和推理成本，使得扩散模型在实际应用中更具可行性。
*   **新的理论视角：** 这是首次尝试分析和保留扩散模型中的不变谱特征，为理解和改进扩散模型提供了新的理论框架和研究方向。这可能会激发更多关于信息保留和特征不变性在生成模型中作用的研究。
*   **更鲁棒的模型：** 保持不变的谱特征可能意味着模型对输入噪声或扰动的鲁棒性更强，从而在更广泛的应用场景中表现更好。

**4. 可能受益于此研究的相关领域或应用**

*   **高分辨率图像生成：** 能够生成细节丰富、结构清晰的图像，对于需要高质量图像的应用至关重要，如艺术创作、设计、虚拟现实等。
*   **图像编辑和风格迁移：** 保留关键特征的能力可能有助于更精确地控制图像的编辑和风格迁移过程，避免破坏原始图像的核心内容。
*   **数据增强：** 生成多样化且逼真的合成数据，用于训练其他计算机视觉模型，尤其是在数据稀缺的领域。
*   **医学影像分析：** 生成高质量的医学影像，用于诊断、模拟或数据增强，同时保留重要的病理特征。
*   **视频生成：** 将谱特征保留的思想扩展到视频领域，有望生成更连贯、更具动态细节的视频。
*   **科学可视化：** 生成逼真的科学模拟结果或可视化数据。

**5. 从摘要中可以推断出的局限性**

尽管摘要充满了积极的成果，但仍可以推断出一些潜在的局限性：

*   **“不变性”的定义和实现细节：** 摘要并未详细说明如何精确地定义和实现“不变谱特征”。这可能是一个复杂的技术挑战，并且具体的实现方式可能对模型的性能产生重要影响。
*   **计算开销的权衡：** 虽然声称提高了计算效率，但引入谱特征分析和处理本身可能也会增加一定的计算开销。摘要中提到的“10K iterations under specified parameter settings”表明其效率提升是在特定条件下实现的，需要进一步研究其在不同参数和模型规模下的表现。
*   **对特定类型特征的偏好：** 谱特征可能更擅长捕捉全局结构和周期性信息，而对于高度局部化、非结构化的细节（如精细的纹理、随机噪声模式）的保留能力可能需要进一步验证。
*   **泛化性：** 摘要展示了在 CIFAR-10, Celeb-A, 和 LSUN 上的实验结果，这些数据集具有一定的代表性，但其在更复杂、更多样化的数据集上的泛化能力仍需评估。
*   **参数设置的敏感性：** “under specified parameter settings”暗示了模型的性能可能对参数选择较为敏感，找到最优参数组合可能需要大量的实验。
*   **理论分析的深度：** 摘要提到“这是第一份尝试分析和保留不变谱特征的论文”，这表明其理论分析可能仍处于初步阶段，未来需要更深入的数学证明和理论支撑。

总而言之，InSPECT 提出了一种新颖且有前景的方法来改进扩散模型，通过关注谱特征的不变性来解决现有模型的效率和生成质量问题。其潜在的计算效率提升和生成质量的显著改善，使其成为一个值得关注的研究方向。

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
通用机器人学习面临数据瓶颈：大规模、多样化且高质量的真实世界交互数据收集成本高昂且耗时。尽管仿真提供了扩展数据收集的潜力，但任务设计、场景生成、专家演示合成以及仿真到现实（sim-to-real）的迁移等环节仍需大量人工干预。

**2. 主要创新点/方法论贡献：**
本文提出了 **AnyTask**，一个自动化框架，旨在解决上述挑战。其核心创新在于：
*   **自动化任务和数据生成流水线：** AnyTask 集成了大规模并行 GPU 仿真与基础模型（如 VLM 和 LLM），能够自动设计多样化的操作任务、生成逼真的场景，并合成高质量的机器人演示数据。
*   **多样化的演示生成代理：** 引入了三种代理来自动生成专家演示，以最大化任务解决能力：
    *   **ViPR：** 一个新颖的任务与运动规划（TAMP）代理，结合了视觉语言模型（VLM）的迭代式并行精炼，以提高规划的鲁棒性。
    *   **ViPR-Eureka：** 一个强化学习（RL）代理，利用 LLM 生成的密集奖励和 LLM 指导的接触采样，以处理复杂接触任务。
    *   **ViPR-RL：** 一个混合规划与学习方法，结合了 TAMP 和 RL 的优势，能够仅凭稀疏奖励生成高质量演示。
*   **智能对象数据库和任务生成器：** 利用 VLM 和 LLM 构建了包含对象属性和语义信息的数据库，并能根据高层任务指令自动生成详细的任务描述和场景配置。
*   **仿真生成器与 API：** 能够将 LLM 生成的任务描述转化为可执行的仿真代码，并提供一套标准化的环境和机器人技能 API。
*   **大规模并行仿真：** 利用 GPU 加速的仿真环境，实现大规模、高效的数据收集。
*   **零样本仿真到现实迁移：** 训练的策略可以直接部署到物理机器人上，无需真实世界数据进行微调。

**3. 主要结果及其意义：**
*   AnyTask 框架能够生成多样化的任务和高质量的演示数据，显著减少了人工干预。
*   通过 AnyTask 生成的数据训练的策略在仿真环境中表现良好，并且能够泛化到新的物体姿态。
*   在真实世界机器人上进行了零样本仿真到现实迁移实验，在包括拾取-放置、开门、接触式推动和长时序操作等一系列任务中，取得了 **44% 的平均成功率**。
*   研究表明，AnyTask 生成的数据对于训练通用机器人策略至关重要，并且证明了仅使用合成数据实现有效的仿真到现实迁移是可行的。

**4. 论文中提到的局限性：**
*   尽管代理展现了广泛的能力，但在需要高精度或复杂物理推理的任务（如任意物体堆叠）上，其性能仍有待提高。
*   成功的仿真到现实迁移依赖于点云观测。将此扩展到 RGB 图像作为输入将降低对硬件的要求。

**5. 潜在的未来研究方向：**
*   将框架扩展到更多种类的物体和机器人形态。
*   将框架应用于更复杂的长时序移动操作任务。
*   探索使用 RGB 图像作为输入，以降低对传感器硬件的要求。
*   进一步提升代理在复杂物理交互任务上的性能。

**对计算机视觉领域的意义：**
这篇论文在计算机视觉领域具有重要意义，因为它展示了如何利用大型基础模型（VLM 和 LLM）与大规模并行仿真相结合，**自动化机器人学习中的数据生成过程**。这不仅解决了机器人领域长期存在的数据获取难题，还为开发更通用、更鲁棒的机器人策略提供了新的途径。特别是，它强调了：
*   **VLM/LLM 在理解和生成复杂任务指令、对象属性以及指导机器人行为方面的强大能力。**
*   **仿真在生成大规模、多样化、标注丰富的数据集方面的潜力，以及如何通过仿真来弥合与现实世界的差距。**
*   **点云作为一种有效的视觉输入，在实现零样本仿真到现实迁移中的作用。**

AnyTask 的方法论为未来机器人学习研究开辟了新的方向，尤其是在如何利用 AI 的进步来加速和简化机器人系统的开发和部署方面。

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
文本到图像（T2I）扩散模型在生成高质量图像方面取得了显著进展，但它们在准确捕捉文本提示中指定的物体空间关系方面存在固有缺陷。这种局限性主要源于两个因素：训练数据中缺乏细粒度的空间监督，以及文本嵌入本身难以有效编码空间语义。这导致模型在生成图像时，物体的位置、相对关系（如“左边”、“右边”、“上面”、“下面”）经常出现错误，甚至完全忽略了这些空间指令。

**2. 关键创新/方法贡献：**
为了解决上述问题，作者提出了 **InfSplign**，一种**无需训练、在推理时**对T2I扩散模型进行空间对齐的方法。其核心创新在于：

*   **推理时空间对齐损失：** InfSplign 在每次去噪步骤中引入一个复合损失函数，通过调整去噪过程中的噪声来引导生成过程。
*   **利用多层级交叉注意力图：** 该方法巧妙地从扩散模型 U-Net 解码器的不同层级（粗粒度、中粒度）提取交叉注意力图。这些注意力图被视为物体空间信息的代理。
*   **三个关键损失项：**
    *   **空间对齐损失 (Lspatial)：** 通过计算物体在注意力图中的质心（centroid）来估计物体的位置，并根据文本提示中的空间关系（如“左”、“右”、“上”、“下”）来惩罚偏离预期位置的行为。
    *   **物体存在损失 (Lpresence)：** 通过最小化物体注意力图的方差来确保物体在最终图像中清晰可见，防止物体被弱化或消失。
    *   **表示平衡损失 (Lbalance)：** 通过平衡不同物体在注意力图中的分散程度，防止一个物体过度主导而另一个物体被抑制，确保所有物体都能得到充分表示。
*   **轻量级、即插即用：** InfSplign 是一种轻量级的方法，不需要对预训练的扩散模型进行任何微调或重新训练，可以轻松地集成到任何现有的扩散模型骨干网络中。

**3. 主要结果及其意义：**
作者在两个广泛使用的空间理解基准测试集——VISOR 和 T2I-CompBench 上进行了全面的评估。结果表明：

*   **状态艺术（State-of-the-Art）性能：** InfSplign 在这两个基准测试上都取得了显著的性能提升，在空间对齐方面达到了新的最先进水平（据作者所知）。
*   **超越现有方法：** 相较于最强的现有推理时基线方法，InfSplign 实现了大幅度的性能提升（例如，在VISOR-4上提升高达24.81%）。更重要的是，它甚至超越了一些需要额外训练或微调的方法。
*   **鲁棒性：** InfSplign 在处理不同物体组合和空间关系时都表现出良好的鲁棒性，即使是那些不常见的物体组合也能生成更具空间一致性的图像。
*   **意义：** InfSplign 的成功表明，通过在推理时巧妙地利用模型内部的注意力机制，可以在不增加训练成本的情况下，显著提升T2I模型在空间理解方面的能力，这对于需要精确空间布局的应用（如机器人导航、增强现实）至关重要。

**4. 提及的局限性：**
*   **罕见物体组合的挑战：** 对于在自然场景中极少共同出现的物体组合，基础扩散模型本身就难以生成，InfSplign 在这种情况下也难以完全纠正空间对齐，因为物体的存在本身就受到限制。在这种情况下，物体准确率（object accuracy）成为瓶颈。
*   **依赖于注意力图质量：** 方法的有效性在一定程度上依赖于扩散模型骨干网络生成的高质量注意力图。

**5. 潜在的未来研究方向：**
*   **扩展到 Transformer 架构：** 作者正在积极探索将 InfSplign 扩展到 Transformer 架构，并已取得初步成果。
*   **更复杂的空间关系：** 研究将进一步探索更复杂的多维空间关系，例如三维空间中的相对位置。

总而言之，InfSplign 是一项重要的研究成果，它提供了一种高效且易于实现的推理时方法，显著改善了文本到图像扩散模型在理解和生成物体空间关系方面的能力，为实现更具可控性和准确性的图像生成开辟了新的途径。

**Key Findings:**

- We introduce InfSplign, a training-free inference-time method that improves spatial alignment by adjusting the noise through a compound loss in every denoising step.
- Our comprehensive evaluations on VISOR and T2I-CompBench show that InfSplign establishes a new state-of-the-art (to the best of our knowledge), achieving substantial performance gains over the strongest existing inference-time baselines and even outperforming the fine-tuning-based methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.17851v1)
- [arXiv](https://arxiv.org/abs/2512.17851v1)

---

