time: 20251001

# Arxiv Computer Vision Papers - 2025-10-01

## Executive Summary

好的，这是一份针对2025年9月30日Arxiv计算机视觉领域论文的每日报告执行摘要，旨在帮助忙碌的研究人员快速了解最新进展。

---

**每日Arxiv计算机视觉报告执行摘要 (2025年9月30日)**

**1. 主要主题和趋势概述：**

今天的论文集展示了计算机视觉领域持续的多元化和快速发展。主要趋势包括：

*   **多模态与扩散模型：** 扩散模型在图像生成和控制方面持续演进，并与多模态输入（如文本、姿态）结合，实现更精细的控制。
*   **3D视觉与重建：** 3D重建技术在效率、泛化性和特定场景（如城市规模SLAM、人体重建）的应用上取得了显著进展。
*   **基础模型与泛化能力：** 探索如何利用大型语言模型（LLM）的先验知识来增强视觉理解，以及如何提升模型在未知环境或数据上的泛化能力。
*   **特定应用领域：** 论文涵盖了从专业视频生成、视频对象分割到材料科学图像分析等多个具体应用场景，显示了CV技术在各行各业的渗透。
*   **效率与训练策略：** 出现了免训练（training-free）方法和测试时训练（test-time training）等策略，旨在提高模型部署的灵活性和效率。

**2. 特别重要或创新的论文亮点：**

*   **"Stitch: Training-Free Position Control in Multimodal Diffusion Transformers" (Jessica Bader et et al.)：** 这篇论文非常创新，提出了一个**免训练**的方法来控制多模态扩散模型中的物体位置，这对于生成式AI的实际应用具有重大意义，因为它大大降低了微调的成本和复杂性。
*   **"Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training" (Junlin Han et al.)：** 这篇论文深入探讨了LLM如何通过语言预训练获得视觉先验知识，并将其应用于视觉任务。这对于理解和构建更强大的**多模态基础模型**具有重要的理论和实践价值。
*   **"DA$^2$: Depth Anything in Any Direction" (Haodong Li et al.)：** 延续了“Depth Anything”的泛化能力，这篇工作可能进一步提升了**通用深度估计**的鲁棒性和应用范围，使其在各种复杂场景下都能提供可靠的深度信息。

**3. 新兴研究方向或技术：**

*   **免训练（Training-Free）方法：** "Stitch"的出现表明，在特定控制任务中，通过巧妙的设计而非大量数据训练，也能实现高性能，这可能成为未来模型部署和定制化的一个重要方向。
*   **LLM视觉先验的利用：** "Learning to See Before Seeing"强调了从语言模型中提取和利用视觉知识的潜力，预示着未来多模态模型将更紧密地融合语言和视觉信息。
*   **测试时训练（Test-Time Training, TTT）在3D重建中的应用：** "TTT3R"将TTT引入3D重建，这是一种提高模型对未知场景适应性的有效策略，有望在资源受限或需要快速适应新环境的场景中发挥作用。
*   **结构化视频生成与评估：** "Stable Cinemetrics"提出了专业视频生成的分类和评估框架，表明该领域正从单纯的生成质量转向更注重**内容结构和专业性**。

**4. 建议阅读全文的论文：**

为了更深入地了解这些进展，我建议您优先阅读以下论文：

*   **"Stitch: Training-Free Position Control in Multimodal Diffusion Transformers" (Jessica Bader et al.)：** 对于任何关注生成式AI和扩散模型控制的研究人员来说，这篇论文是必读的，其免训练的理念非常具有启发性。
*   **"Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training" (Junlin Han et al.)：** 如果您对多模态基础模型、LLM与视觉的结合感兴趣，这篇论文将提供深刻的见解。
*   **"DA$^2$: Depth Anything in Any Direction" (Haodong Li et al.)：** 对于从事3D视觉、机器人或自动驾驶领域的研究人员，了解通用深度估计的最新进展至关重要。
*   **"HART: Human Aligned Reconstruction Transformer" (Xiyi Chen et al.)：** 如果您专注于人体3D重建或虚拟人技术，这篇论文可能提供了新的SOTA方法。

---

这份摘要旨在为您提供一个快速概览，帮助您识别最相关的研究方向和论文，以便进一步深入阅读。

---

## Table of Contents

1. [Stitch: Training-Free Position Control in Multimodal Diffusion Transformers](#2509.26644v1)
2. [TTT3R: 3D Reconstruction as Test-Time Training](#2509.26645v1)
3. [Benchmarking Egocentric Visual-Inertial SLAM at City Scale](#2509.26639v1)
4. [Learning Generalizable Shape Completion with SIM(3) Equivariance](#2509.26631v1)
5. [Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training](#2509.26625v1)
6. [HART: Human Aligned Reconstruction Transformer](#2509.26621v1)
7. [DA$^2$: Depth Anything in Any Direction](#2509.26618v1)
8. [Video Object Segmentation-Aware Audio Generation](#2509.26604v1)
9. [Stable Cinemetrics : Structured Taxonomy and Evaluation for Professional Video Generation](#2509.26555v1)
10. [Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework](#2509.26548v1)

---

## Papers

<a id='2509.26644v1'></a>
## [Stitch: Training-Free Position Control in Multimodal Diffusion Transformers](https://arxiv.org/abs/2509.26644v1)

**Authors:** Jessica Bader, Mateusz Pach, Maria A. Bravo, Serge Belongie, Zeynep Akata

**Published:** 2025-09-30

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Text-to-Image (T2I) generation models have advanced rapidly in recent years,
but accurately capturing spatial relationships like "above" or "to the right
of" poses a persistent challenge. Earlier methods improved spatial relationship
following with external position control. However, as architectures evolved to
enhance image quality, these techniques became incompatible with modern models.
We propose Stitch, a training-free method for incorporating external position
control into Multi-Modal Diffusion Transformers (MMDiT) via
automatically-generated bounding boxes. Stitch produces images that are both
spatially accurate and visually appealing by generating individual objects
within designated bounding boxes and seamlessly stitching them together. We
find that targeted attention heads capture the information necessary to isolate
and cut out individual objects mid-generation, without needing to fully
complete the image. We evaluate Stitch on PosEval, our benchmark for
position-based T2I generation. Featuring five new tasks that extend the concept
of Position beyond the basic GenEval task, PosEval demonstrates that even top
models still have significant room for improvement in position-based
generation. Tested on Qwen-Image, FLUX, and SD3.5, Stitch consistently enhances
base models, even improving FLUX by 218% on GenEval's Position task and by 206%
on PosEval. Stitch achieves state-of-the-art results with Qwen-Image on
PosEval, improving over previous models by 54%, all accomplished while
integrating position control into leading models training-free. Code is
available at https://github.com/ExplainableML/Stitch.

**Analysis:**

好的，这是一篇关于Jessica Bader等人撰写的论文“Stitch: Training-Free Position Control in Multimodal Diffusion Transformers”的全面摘要：

**论文摘要：Stitch: 多模态扩散Transformer中的免训练位置控制**

**1. 主要问题或研究问题：**
尽管文本到图像（T2I）生成模型近年来取得了显著进展，但在准确捕捉“上方”、“右侧”等空间关系方面仍然面临持续挑战。早期的外部位置控制方法虽然能改善空间关系，但随着模型架构演进以提升图像质量，这些技术与现代模型变得不兼容。因此，核心问题是如何在不牺牲图像质量和生成速度的前提下，为最新的多模态扩散Transformer（MMDiT）模型提供精确且免训练的位置控制。

**2. 关键创新或方法论贡献：**
该论文提出了**Stitch**，一种免训练的方法，用于将外部位置控制集成到MMDiT架构中。其主要创新点包括：

*   **LLM驱动的边界框分解与区域绑定（Region Binding）：** Stitch利用大型语言模型（LLM）将完整的文本提示分解为多个子提示，每个子提示对应一个由LLM生成的边界框。在生成的前S步中，MMDiT模型在这些指定边界框内独立生成各个对象和背景，并通过注意力掩码约束（Region Binding）确保对象完全在各自的边界框内生成，并与周围上下文隔离。
*   **注意力引导的抠图（Cutout）：** 为了避免背景不匹配导致的可见接缝，Stitch在生成中期（S步之后）通过分析特定注意力头的最高注意力权重，从潜在空间中提取前景对象。这种方法无需外部分割模型，且能在图像未完全生成时进行。提取出的前景潜在tokens与背景潜在tokens合并形成复合潜在表示，用于后续的无约束生成。
*   **PosEval基准测试：** 论文引入了PosEval，一个扩展自GenEval的、针对基于位置的T2I生成能力的综合基准测试。PosEval包含五个新任务，旨在深入评估T2I模型的定位能力，超越了传统的“位置”概念，以揭示特定故障模式并应对更复杂的生成挑战。

**3. 主要结果及其重要性：**
Stitch在多个领先的MMDiT模型（如Qwen-Image、FLUX和SD3.5）上进行了评估，并取得了显著成果：

*   **显著提升位置准确性：** Stitch在PosEval基准测试中持续提升了基础模型的性能。例如，在GenEval的“位置”任务上，FLUX的性能提升了218%；在PosEval上，FLUX的性能提升了206%。
*   **实现最先进（SOTA）结果：** Stitch在Qwen-Image模型上实现了PosEval的SOTA结果，比之前模型提升了54%。
*   **保持图像质量和多样性：** Stitch在提升位置控制的同时，并未显著降低图像的视觉质量（通过美学分数评估）和多样性（通过DINOv2嵌入空间中的样本间距离评估）。
*   **免训练集成：** 所有这些改进都是通过免训练的方式实现的，使得Stitch能够快速且经济地升级现有T2I模型。
*   **PosEval揭示现有模型局限：** PosEval基准测试表明，即使是顶级的T2I模型在处理复杂的位置提示时仍有很大的改进空间，尤其是在多对象、相对关系和否定关系等任务上。

**4. 论文中提及的局限性：**
论文中没有明确指出Stitch方法的局限性，但提到了以下几点：

*   **现有T2I模型在复杂位置提示上的挣扎：** PosEval基准测试揭示，即使是SOTA模型在处理多对象、相对关系和否定关系等复杂位置任务时仍有显著不足。
*   **抠图掩码的窄边框：** 抠图掩码有时会在对象周围留下一个窄边框，这可能是因为低级细节尚未完全固化。尽管这种行为有助于捕获完整对象并在后续生成步骤中重建缺失部分，但它也暗示了抠图的精确度可能仍有提升空间。

**5. 潜在的未来研究方向：**
论文没有明确提出未来研究方向，但从其贡献和局限性中可以推断出以下潜在方向：

*   **优化抠图机制：** 进一步研究和优化抠图机制，以实现更精确的对象分割，减少窄边框现象，并探索在更早的生成阶段进行更精细的潜在空间操作。
*   **扩展PosEval基准：** 随着T2I模型能力的提升，可以进一步扩展PosEval，引入更多复杂、细致的位置任务，例如涉及三维空间关系、动态对象或更复杂的语言结构。
*   **探索更深层次的MMDiT集成：** 尽管Stitch是免训练的，但未来可以探索将位置控制机制更深层次地集成到MMDiT的训练过程中，以期实现更原生和鲁棒的性能。
*   **多模态输入融合：** 论文主要关注文本提示，未来可以探索如何将Stitch扩展到处理其他模态的输入（如草图、布局图），以实现更灵活和精细的位置控制。

总而言之，Stitch为多模态扩散Transformer中的位置控制提供了一个强大且免训练的解决方案，显著提升了T2I模型在复杂空间关系生成方面的能力，并通过引入PosEval基准测试，为该领域未来的研究提供了重要的评估工具。

**Key Findings:**

- We propose Stitch, a training-free method for incorporating external position
control into Multi-Modal Diffusion Transformers (MMDiT) via
automatically-generated bounding boxes.
- Featuring five new tasks that extend the concept
of Position beyond the basic GenEval task, PosEval demonstrates that even top
models still have significant room for improvement in position-based
generation.
- Stitch achieves state-of-the-art results with Qwen-Image on
PosEval, improving over previous models by 54%, all accomplished while
integrating position control into leading models training-free.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26644v1)
- [arXiv](https://arxiv.org/abs/2509.26644v1)

---

<a id='2509.26645v1'></a>
## [TTT3R: 3D Reconstruction as Test-Time Training](https://arxiv.org/abs/2509.26645v1)

**Authors:** Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen

**Published:** 2025-09-30

**Categories:** cs.CV

**Abstract:**

Modern Recurrent Neural Networks have become a competitive architecture for
3D reconstruction due to their linear-time complexity. However, their
performance degrades significantly when applied beyond the training context
length, revealing limited length generalization. In this work, we revisit the
3D reconstruction foundation models from a Test-Time Training perspective,
framing their designs as an online learning problem. Building on this
perspective, we leverage the alignment confidence between the memory state and
incoming observations to derive a closed-form learning rate for memory updates,
to balance between retaining historical information and adapting to new
observations. This training-free intervention, termed TTT3R, substantially
improves length generalization, achieving a $2\times$ improvement in global
pose estimation over baselines, while operating at 20 FPS with just 6 GB of GPU
memory to process thousands of images. Code available in
https://rover-xingyu.github.io/TTT3R

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen撰写的论文“TTT3R: 3D Reconstruction as Test-Time Training”的全面摘要。

---

### 论文摘要：TTT3R: 3D Reconstruction as Test-Time Training

**1. 主要问题或研究问题：**
该论文主要关注现代循环神经网络（RNNs）在3D重建任务中的一个核心限制：尽管RNNs具有线性时间复杂度，但在训练上下文长度之外应用时，其性能会显著下降，表现出有限的长度泛化能力。具体来说，现有的基于RNN的3D重建基础模型（如CUT3R）在处理长序列图像时，由于“遗忘问题”和状态过拟合，性能会严重退化，无法有效保留历史信息。

**2. 关键创新或方法论贡献：**
TTT3R的核心创新在于将3D重建基础模型的设计重新构想为**测试时训练（Test-Time Training, TTT）**问题，将其视为一个在线学习过程。在此视角下，论文提出了以下关键贡献：
*   **在线学习框架：** 将记忆状态（state）视为在测试时通过梯度下降学习的“快速权重”，而非训练阶段固定的“慢权重”，从而使其能够动态适应输入上下文。
*   **置信度引导的状态更新规则：** 论文利用记忆状态与传入观测值之间的对齐置信度，推导出一个**闭式（closed-form）学习率**来更新记忆。这个学习率能够平衡历史信息的保留和对新观测的适应，有效缓解了灾难性遗忘问题。具体而言，学习率$\beta_t$由状态查询$Q_{s_{t-1}}$和观测键$K_{x_t}$之间的对齐置信度（通过softmax加权）决定，从而实现每token自适应学习。
*   **训练无关的干预：** TTT3R是一种“训练无关、即插即用”的干预措施，无需对现有模型进行微调或添加额外参数，即可直接应用于下游任务，显著提升了长度泛化能力。

**3. 主要结果及其意义：**
TTT3R在多个3D重建基准测试中取得了显著成果：
*   **长度泛化能力大幅提升：** 相较于现有基线方法，TTT3R在全局姿态估计方面实现了**2倍**的性能提升，尤其在处理数千张图像的长序列时表现出强大的鲁化性。
*   **高效的资源利用：** 尽管处理数千张图像，TTT3R仍能以**20 FPS**的速度运行，且仅需**6 GB的GPU内存**，保持了与CUT3R基线相同的推理速度和内存效率。
*   **定性改进：** 在定性结果中，TTT3R实现了更准确的重建，有效缓解了CUT3R中出现的灾难性遗忘、相机姿态漂移、几何结构损坏和鬼影伪影等问题，并支持在线闭环。
*   **与在线方法的竞争力：** 在短序列评估中，TTT3R与最先进的在线重建模型（如CUT3R、Point3R）相比具有竞争力，在某些数据集上甚至表现最佳。

这些结果表明，TTT3R为解决RNNs在长序列3D重建中的泛化问题提供了一个有效且高效的解决方案，推动了在线3D重建技术的发展。

**4. 论文中提及的局限性：**
*   **未能完全解决状态遗忘：** 尽管TTT3R显著缓解了状态遗忘问题，但并未完全解决。
*   **与离线方法的差距：** TTT3R的重建精度尚未完全匹配强大的离线方法（如VGGT），因为全注意力机制虽然速度慢、内存需求高，但能保留完整的历史上下文。
*   **设计空间仍待探索：** 作为一种测试时回归方法，TTT3R的设计空间仍有待进一步探索。

**5. 潜在的未来研究方向：**
*   **开发更有效、稳定和可并行化的循环架构：** 论文指出，未来研究应继续探索更有效的循环架构，以进一步提高3D重建的精度和长度泛化能力。
*   **深入探索测试时回归的设计空间：** 鉴于测试时回归在关联记忆方面展现出的潜力，未来可以更深入地研究其设计空间，以期发现更优的解决方案。

---

**Key Findings:**

- Building on this
perspective, we leverage the alignment confidence between the memory state and
incoming observations to derive a closed-form learning rate for memory updates,
to balance between retaining historical information and adapting to new
observations.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26645v1)
- [arXiv](https://arxiv.org/abs/2509.26645v1)

---

<a id='2509.26639v1'></a>
## [Benchmarking Egocentric Visual-Inertial SLAM at City Scale](https://arxiv.org/abs/2509.26639v1)

**Authors:** Anusha Krishnan, Shaohui Liu, Paul-Edouard Sarlin, Oscar Gentilhomme, David Caruso, Maurizio Monge, Richard Newcombe, Jakob Engel, Marc Pollefeys

**Published:** 2025-09-30

**Categories:** cs.CV, cs.RO

**Abstract:**

Precise 6-DoF simultaneous localization and mapping (SLAM) from onboard
sensors is critical for wearable devices capturing egocentric data, which
exhibits specific challenges, such as a wider diversity of motions and
viewpoints, prevalent dynamic visual content, or long sessions affected by
time-varying sensor calibration. While recent progress on SLAM has been swift,
academic research is still driven by benchmarks that do not reflect these
challenges or do not offer sufficiently accurate ground truth poses. In this
paper, we introduce a new dataset and benchmark for visual-inertial SLAM with
egocentric, multi-modal data. We record hours and kilometers of trajectories
through a city center with glasses-like devices equipped with various sensors.
We leverage surveying tools to obtain control points as indirect pose
annotations that are metric, centimeter-accurate, and available at city scale.
This makes it possible to evaluate extreme trajectories that involve walking at
night or traveling in a vehicle. We show that state-of-the-art systems
developed by academia are not robust to these challenges and we identify
components that are responsible for this. In addition, we design tracks with
different levels of difficulty to ease in-depth analysis and evaluation of less
mature approaches. The dataset and benchmark are available at
https://www.lamaria.ethz.ch.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Anusha Krishnan等人撰写的论文“Benchmarking Egocentric Visual-Inertial SLAM at City Scale”的全面摘要。

---

**论文摘要：Benchmarking Egocentric Visual-Inertial SLAM at City Scale**

**1. 主要问题或研究问题：**
该论文旨在解决现有视觉惯性同步定位与建图（VI-SLAM）基准数据集在评估以自我为中心（egocentric）数据时存在的不足。具体来说，现有基准未能充分反映可穿戴设备捕获的自我中心数据的独特挑战，例如：运动和视角的多样性、动态视觉内容、长时间会话中时变传感器校准，以及缺乏足够精确的真值姿态。这导致学术界开发的SLAM系统在面对这些真实世界挑战时表现不佳。

**2. 关键创新或方法论贡献：**
*   **LaMAria数据集：** 引入了一个新的、大规模、以城市为中心的自我中心VI-SLAM数据集和基准。该数据集使用Project Aria眼镜式设备记录了数小时、数十公里的轨迹，包含多传感器（双灰度全局快门相机、RGB卷帘快门相机、双IMU、磁力计、气压计、温度计、GNSS接收器、WiFi和蓝牙收发器）数据。
*   **厘米级真值姿态：** 利用测量工具（包括GNSS-RTK和全站仪）获取稀疏控制点（CPs）作为间接姿态标注，这些标注具有度量、厘米级精度，并覆盖整个城市范围。通过将AprilTag标记附着在CPs上，实现了CP的自动检测。
*   **伪真值姿态生成：** 通过融合视觉、惯性和CP信息，并进行联合优化，生成了更密集的伪真值相机姿态，以支持细粒度评估和3D重建任务。
*   **挑战性场景覆盖：** 数据集涵盖了自我中心数据的独特挑战，包括极低光照、曝光变化、移动平台（如电车、缆车）、时变校准、动态场景（如行人、车辆）以及室内外过渡。
*   **分级难度轨道：** 设计了不同难度级别的实验轨道（Level I-IV），从受控平台运动到不受限制的头戴式运动，以便深入分析和评估不同成熟度的方法。

**3. 主要结果及其意义：**
*   **现有SOTA系统表现不佳：** 评估结果表明，学术界开发的现有最先进（SOTA）VI-SLAM系统在面对LaMAria数据集中的自我中心挑战时，鲁棒性不足，并且与Project Aria设备的专有SLAM API之间存在显著差距。
*   **多传感器融合的优势：** 依赖多相机和惯性传感器显著提升了VI-SLAM系统的性能。
*   **在线校准的重要性：** Project Aria的SLAM系统（包含在线校准）表现优于所有学术解决方案，尤其是在长时间序列中，焦距变化范围更大，凸显了在线校准对于可穿戴设备全天候使用的重要性。
*   **识别失败原因：** 论文识别了导致现有系统失败的关键组件，例如在快速复杂运动模式下容易丢失跟踪，以及在低光照和移动平台等挑战性条件下视觉信息不可靠。
*   **推动未来研究：** LaMAria数据集和基准的发布为多传感器SLAM在不受控自我中心记录下的发展提供了新的研究方向和评估工具。

**4. 论文中提及的局限性：**
*   **伪真值精度：** 尽管论文生成的密集伪真值姿态比现有数据集更准确，但其精度仍不如测量级的控制点，在某些情况下（如移动平台场景）其准确性保证有限。
*   **视觉信息不可靠：** 在某些最具挑战性的场景（如移动平台），视觉信息可能不可靠，导致视觉惯性系统难以初始化或跟踪失败。
*   **现有系统对在线校准支持不足：** 大多数评估的学术系统不支持在线内参校准，只实现了简单的相机模型，这限制了它们在时变校准场景下的性能。

**5. 潜在的未来研究方向：**
*   **时变校准的在线优化：** 开发能够适应可穿戴设备全天候使用特性的时变校准在线优化方法。
*   **回环检测和VI捆集调整：** 改进回环检测和视觉惯性捆集调整，以减少开环预测中的里程计漂移。
*   **鲁棒的异常值去除和跟踪丢失处理：** 针对移动平台等场景，开发更鲁棒的异常值去除策略和更好的跟踪丢失处理机制。
*   **基于机器学习的图像匹配和点跟踪：** 利用在大规模数据集上训练的机器学习模型，提升图像匹配和点跟踪的性能。

---

这篇论文通过引入一个具有高精度真值和多样化挑战场景的大规模自我中心数据集，为VI-SLAM领域的研究做出了重要贡献。它不仅揭示了现有SOTA系统在真实世界自我中心数据下的局限性，也为未来的研究指明了方向。

**Key Findings:**

- In this
paper, we introduce a new dataset and benchmark for visual-inertial SLAM with
egocentric, multi-modal data.
- We show that state-of-the-art systems
developed by academia are not robust to these challenges and we identify
components that are responsible for this.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26639v1)
- [arXiv](https://arxiv.org/abs/2509.26639v1)

---

<a id='2509.26631v1'></a>
## [Learning Generalizable Shape Completion with SIM(3) Equivariance](https://arxiv.org/abs/2509.26631v1)

**Authors:** Yuqing Wang, Zhaiyu Chen, Xiao Xiang Zhu

**Published:** 2025-09-30

**Categories:** cs.CV, cs.AI

**Abstract:**

3D shape completion methods typically assume scans are pre-aligned to a
canonical frame. This leaks pose and scale cues that networks may exploit to
memorize absolute positions rather than inferring intrinsic geometry. When such
alignment is absent in real data, performance collapses. We argue that robust
generalization demands architectural equivariance to the similarity group,
SIM(3), so the model remains agnostic to pose and scale. Following this
principle, we introduce the first SIM(3)-equivariant shape completion network,
whose modular layers successively canonicalize features, reason over
similarity-invariant geometry, and restore the original frame. Under a
de-biased evaluation protocol that removes the hidden cues, our model
outperforms both equivariant and augmentation baselines on the PCN benchmark.
It also sets new cross-domain records on real driving and indoor scans,
lowering minimal matching distance on KITTI by 17% and Chamfer distance $\ell1$
on OmniObject3D by 14%. Perhaps surprisingly, ours under the stricter protocol
still outperforms competitors under their biased settings. These results
establish full SIM(3) equivariance as an effective route to truly generalizable
shape completion. Project page: https://sime-completion.github.io.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Yuqing Wang, Zhaiyu Chen, Xiao Xiang Zhu撰写的论文“Learning Generalizable Shape Completion with SIM(3) Equivariance”的全面摘要。

---

### 论文摘要：Learning Generalizable Shape Completion with SIM(3) Equivariance

**1. 主要问题或研究问题：**
当前3D形状补全方法普遍存在一个核心问题：它们通常假设输入扫描已预先对齐到规范坐标系。这种做法导致网络倾向于记忆绝对位置和尺度信息，而非学习形状的内在几何特性。当在实际数据中缺乏这种预对齐时，模型的性能会急剧下降，导致泛化能力差。因此，论文旨在解决如何实现对任意姿态和尺度（即SIM(3)变换）具有鲁棒泛化能力的3D形状补全，从而避免对齐偏差。

**2. 关键创新或方法贡献：**
为了解决上述问题，论文提出了首个完全SIM(3)等变（SIM(3)-equivariant）的形状补全网络，其核心创新点在于：
*   **SIM(3)等变架构：** 论文主张鲁棒的泛化能力需要模型对相似变换群SIM(3)（包括旋转、平移和尺度）具有等变性。为此，作者设计了一个模块化的网络架构，其每一层都确保了SIM(3)等变性。
*   **模块化设计：** 网络由三个连续阶段组成：
    1.  **特征规范化（Canonicalization）：** 将特征转换为平移和尺度不变的形式，通过扩展层归一化（Layer Normalization）显式地去除全局平移和尺度信息，同时保持旋转等变性。
    2.  **相似性不变几何推理（Shape Reasoning）：** 在规范化特征空间中，通过SIM(3)不变的注意力机制进行形状推理，确保模型学习到内在几何。
    3.  **变换恢复（Transform Restoration）：** 引入一个轻量级的变换恢复路径，通过残差连接将姿态和尺度信息重新注入到特征中，以将补全后的形状恢复到原始传感器坐标系。
*   **去偏置评估协议：** 论文建立了一个严格的评估协议，移除了传统基准测试中存在的隐藏姿态和尺度线索，以公平地测试模型的真实泛化能力。

**3. 主要结果及其重要性：**
*   **超越基线：** 在去偏置评估协议下，该模型在PCN基准测试上显著优于现有的等变和数据增强基线方法。
*   **跨域泛化能力：** 在真实驾驶场景（KITTI）和室内场景（OmniObject3D）扫描数据上，模型取得了新的跨域泛化记录，将KITTI上的最小匹配距离（MMD）降低了17%，将OmniObject3D上的Chamfer距离$\ell1$降低了14%。
*   **鲁棒性：** 即使在更严格的协议下（即没有预对齐），该模型仍能超越在偏置设置下（即有预对齐）训练的竞争对手。这证明了完全SIM(3)等变性是实现真正可泛化形状补全的有效途径。
*   **对齐偏差的揭示：** 论文通过实验证实了现有方法中姿态和尺度偏差的存在，并强调了SIM(3)等变性对于“野外”泛化的重要性。

**4. 论文中提及的局限性：**
*   **姿态和尺度依赖特征的丢失：** 尽管模型对任意相似变换具有鲁棒性，但它也可能丢弃当物体始终出现在规范坐标系中时有用的线索。例如，没有可见腿的椅子背面可能被误认为是沙发。
*   **部分观测的对称性：** 框架中的等变性是针对单个部分扫描定义的。对于同一物体的不同部分观测，初始化变异性无法完全消除，因此跨视图对称性必须隐式学习。
*   **复杂场景和可动部件：** 目前的方法擅长补全静态形状，但尚未明确处理独立移动的子部件（如人体关节、机械臂或多物体场景）。
*   **计算开销：** 向量值特征和完全等变模块相比标量值层会带来三倍的计算开销，导致运行时延迟较高，可能限制实时或资源受限部署。

**5. 潜在的未来研究方向：**
*   将当前框架扩展到多物体和大规模场景建模，以处理更复杂的场景。
*   结合特定类别形状先验或允许多个局部变换，以更好地处理可动部件和复杂场景。
*   进一步优化计算效率，以降低等变模块带来的运行时开销，使其适用于实时应用。

---

这篇论文通过引入SIM(3)等变性，为3D形状补全领域带来了重要的突破，解决了长期存在的对齐偏差问题，并显著提升了模型在真实世界场景中的泛化能力。

**Key Findings:**

- Following this
principle, we introduce the first SIM(3)-equivariant shape completion network,
whose modular layers successively canonicalize features, reason over
similarity-invariant geometry, and restore the original frame.
- Under a
de-biased evaluation protocol that removes the hidden cues, our model
outperforms both equivariant and augmentation baselines on the PCN benchmark.
- It also sets new cross-domain records on real driving and indoor scans,
lowering minimal matching distance on KITTI by 17% and Chamfer distance $\ell1$
on OmniObject3D by 14%.
- Perhaps surprisingly, ours under the stricter protocol
still outperforms competitors under their biased settings.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26631v1)
- [arXiv](https://arxiv.org/abs/2509.26631v1)

---

<a id='2509.26625v1'></a>
## [Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training](https://arxiv.org/abs/2509.26625v1)

**Authors:** Junlin Han, Shengbang Tong, David Fan, Yufan Ren, Koustuv Sinha, Philip Torr, Filippos Kokkinos

**Published:** 2025-09-30

**Categories:** cs.LG, cs.AI, cs.CV, cs.MM

**Abstract:**

Large Language Models (LLMs), despite being trained on text alone,
surprisingly develop rich visual priors. These priors allow latent visual
capabilities to be unlocked for vision tasks with a relatively small amount of
multimodal data, and in some cases, to perform visual tasks without ever having
seen an image. Through systematic analysis, we reveal that visual priors-the
implicit, emergent knowledge about the visual world acquired during language
pre-training-are composed of separable perception and reasoning priors with
unique scaling trends and origins. We show that an LLM's latent visual
reasoning ability is predominantly developed by pre-training on
reasoning-centric data (e.g., code, math, academia) and scales progressively.
This reasoning prior acquired from language pre-training is transferable and
universally applicable to visual reasoning. In contrast, a perception prior
emerges more diffusely from broad corpora, and perception ability is more
sensitive to the vision encoder and visual instruction tuning data. In
parallel, text describing the visual world proves crucial, though its
performance impact saturates rapidly. Leveraging these insights, we propose a
data-centric recipe for pre-training vision-aware LLMs and verify it in 1T
token scale pre-training. Our findings are grounded in over 100 controlled
experiments consuming 500,000 GPU-hours, spanning the full MLLM construction
pipeline-from LLM pre-training to visual alignment and supervised multimodal
fine-tuning-across five model scales, a wide range of data categories and
mixtures, and multiple adaptation setups. Along with our main findings, we
propose and investigate several hypotheses, and introduce the Multi-Level
Existence Bench (MLE-Bench). Together, this work provides a new way of
deliberately cultivating visual priors from language pre-training, paving the
way for the next generation of multimodal LLMs.

**Analysis:**

以下是Junlin Han等人撰写的论文“Learning to See Before Seeing: Demystifying LLM Visual Priors from Language Pre-training”的全面摘要：

**1. 论文主要问题或研究问题**
该论文旨在系统地探究大型语言模型（LLMs）在仅通过文本训练后，如何出人意料地发展出丰富的视觉先验知识。具体来说，它试图揭示这些视觉先验的构成、来源、演变规律，以及如何利用这些先验来构建更强大的多模态LLMs（MLLMs）。

**2. 关键创新或方法论贡献**
*   **视觉先验的结构分解：** 论文首次系统地将LLM的视觉先验分解为可分离的“感知先验”和“推理先验”，并揭示了它们独特的扩展趋势和起源。
*   **数据中心化预训练配方：** 基于对视觉先验起源的深入分析，论文提出了一种数据中心化的预训练配方，用于有意识地培养视觉感知LLMs，并在1T token规模的预训练中得到了验证。
*   **多层次存在基准（MLE-Bench）：** 引入了一个新的基准测试，专门用于细粒度评估模型的感知能力，尤其是在不同尺寸对象（小、中、大）上的感知性能。
*   **盲视觉指令微调（Blind Visual Instruction Tuning）：** 提出了一种“盲视觉指令微调”技巧，作为提高视觉适应性的实用工具，并揭示了模型如何通过语言“捷径”解决视觉任务。
*   **系统性消融研究：** 通过超过100个受控实验（消耗500,000 GPU小时），涵盖LLM预训练到视觉对齐和监督多模态微调的整个MLLM构建流程，跨越五种模型规模、多种数据类别和混合比例以及多种适应设置，对视觉先验的起源进行了深入分析。

**3. 主要结果及其意义**
*   **视觉先验的构成与起源：** 论文发现LLM的潜在视觉推理能力主要通过推理中心数据（如代码、数学、学术论文）的预训练逐步发展和扩展。这种推理先验具有可迁移性和普遍适用性。相比之下，感知先验更广泛地从通用语料库中出现，并且感知能力对视觉编码器和视觉指令微调数据更敏感。描述视觉世界的文本虽然重要，但其性能影响迅速饱和。
*   **数据混合策略：** 实验表明，最大化MLLM VQA性能的最佳数据混合是严重偏向推理中心内容，但同时包含必要的视觉世界知识。在1T token规模的预训练中，平衡模型在语言能力保持竞争力的同时，在所有视觉任务上均优于语言偏好模型。
*   **感知先验的尺度依赖性：** MLE-Bench的评估结果显示，感知先验确实具有尺度依赖性，其优势在感知中小尺寸对象时最为显著，表明多样化的网络爬取数据对于获取感知先验至关重要。
*   **推理能力的跨模态通用性：** 论文通过定性分析和评估模型推理质量的实验证实，LLM从文本中获得的推理能力是模态无关的，可以有效地迁移到视觉问题解决中，表现出更强的逻辑性和更深的推理深度。
*   **语言数据结构与视觉对齐：** 结构化语言数据（如代码和数学）的比例增加，通常能提高LLM-视觉对齐分数，表明学习抽象结构有助于形成更一致的潜在空间，但这种趋势并非单调线性。

**4. 论文中提及的局限性**
*   **架构限制：** 研究主要集中在适配器风格的MLLM架构，其发现可能无法完全推广到其他方法，如离散视觉token化或端到端联合训练视觉和语言组件的模型。
*   **安全与伦理问题：** 论文未深入探讨这些学习到的视觉先验可能带来的安全和伦理影响，例如文本语料库中存在的社会偏见、刻板印象是否会通过视觉先验体现在下游MLLM的有害生成或分类行为中。
*   **模态限制：** 研究仅限于静态图像领域，未探索动态模态（如视频理解）的视觉先验。

**5. 潜在的未来研究方向**
*   **其他MLLM架构的视觉先验研究：** 探索离散视觉token化或端到端联合训练模型中视觉先验的形成和利用动态。
*   **视觉先验的安全与伦理审计：** 对LLM中学习到的视觉先验进行彻底的公平性和安全性审计，以识别和缓解潜在的偏见和有害行为。
*   **动态模态的视觉先验探索：** 研究不同文本源如何促进视频理解中时间推理、动作识别和因果关系等先验的形成。
*   **抽象结构与语义基础的相互作用：** 进一步探究抽象结构和语义基础在形成跨模态表示中的精确相互作用。
*   **更精细的视觉先验结构：** 进一步细化感知先验的内部结构，例如其在不同视觉属性（颜色、形状、纹理）上的表现。

**Key Findings:**

- We show that an LLM's latent visual
reasoning ability is predominantly developed by pre-training on
reasoning-centric data (e.g., code, math, academia) and scales progressively.
- Leveraging these insights, we propose a
data-centric recipe for pre-training vision-aware LLMs and verify it in 1T
token scale pre-training.
- Together, this work provides a new way of
deliberately cultivating visual priors from language pre-training, paving the
way for the next generation of multimodal LLMs.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26625v1)
- [arXiv](https://arxiv.org/abs/2509.26625v1)

---

<a id='2509.26621v1'></a>
## [HART: Human Aligned Reconstruction Transformer](https://arxiv.org/abs/2509.26621v1)

**Authors:** Xiyi Chen, Shaofei Wang, Marko Mihajlovic, Taewon Kang, Sergey Prokudin, Ming Lin

**Published:** 2025-09-30

**Categories:** cs.CV

**Abstract:**

We introduce HART, a unified framework for sparse-view human reconstruction.
Given a small set of uncalibrated RGB images of a person as input, it outputs a
watertight clothed mesh, the aligned SMPL-X body mesh, and a Gaussian-splat
representation for photorealistic novel-view rendering. Prior methods for
clothed human reconstruction either optimize parametric templates, which
overlook loose garments and human-object interactions, or train implicit
functions under simplified camera assumptions, limiting applicability in real
scenes. In contrast, HART predicts per-pixel 3D point maps, normals, and body
correspondences, and employs an occlusion-aware Poisson reconstruction to
recover complete geometry, even in self-occluded regions. These predictions
also align with a parametric SMPL-X body model, ensuring that reconstructed
geometry remains consistent with human structure while capturing loose clothing
and interactions. These human-aligned meshes initialize Gaussian splats to
further enable sparse-view rendering. While trained on only 2.3K synthetic
scans, HART achieves state-of-the-art results: Chamfer Distance improves by
18-23 percent for clothed-mesh reconstruction, PA-V2V drops by 6-27 percent for
SMPL-X estimation, LPIPS decreases by 15-27 percent for novel-view synthesis on
a wide range of datasets. These results suggest that feed-forward transformers
can serve as a scalable model for robust human reconstruction in real-world
settings. Code and models will be released.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Xiyi Chen等人撰写的论文“HART: Human Aligned Reconstruction Transformer”的全面摘要。

**论文摘要：HART: Human Aligned Reconstruction Transformer**

**1. 主要问题或研究问题：**
该论文旨在解决从少量未校准的RGB图像中进行稀疏视角人体重建的挑战。现有的方法在处理宽松衣物、人-物体交互以及在真实场景中应用时存在局限性，例如优化参数模板会忽略衣物细节，或在简化相机假设下训练隐式函数。核心问题是如何在存在自遮挡的情况下，从有限的输入图像中准确、完整地重建出穿着衣物的人体几何形状，并实现逼真的新视角渲染。

**2. 关键创新或方法论贡献：**
HART（Human Aligned Reconstruction Transformer）是一个统一的框架，其主要创新包括：
*   **统一框架：** HART能够同时重建出穿着衣物的水密网格、对齐的SMPL-X身体网格，以及用于逼真新视角渲染的高斯溅射（Gaussian-splat）表示。
*   **像素级3D属性预测：** HART预测每像素的3D点图、法线和身体对应关系，这比传统方法更精细，能够更好地捕捉细节和自遮挡区域。
*   **遮挡感知泊松重建（Occlusion-aware Poisson Reconstruction）：** 论文引入了一个3D U-Net来细化指示网格，以解决点图方法在处理自遮挡区域时的局限性，从而恢复完整且水密的几何形状。
*   **人体对齐的几何重建：** 预测结果与参数化的SMPL-X身体模型对齐，确保重建的几何形状与人体结构保持一致，同时捕捉宽松衣物和交互。
*   **几何引导的新视角合成：** 重建的人体对齐网格被用作高斯溅射的初始化和正则化，从而实现稀疏视角渲染。

**3. 主要结果及其意义：**
尽管仅在2.3K合成扫描数据上进行训练，HART仍取得了最先进的（state-of-the-art）结果：
*   **穿着衣物网格重建：** Chamfer Distance（倒角距离）提高了18-23%。
*   **SMPL-X估计：** PA-V2V（顶点到顶点误差）降低了6-27%。
*   **新视角合成：** LPIPS（感知距离）在各种数据集上降低了15-27%。
这些结果表明，前馈Transformer可以作为一种可扩展的模型，用于在真实世界环境中进行鲁棒的人体重建。

**4. 论文中提及的局限性：**
*   **细节恢复：** 重建结果在精细尺度细节（如手指、头发）方面仍有不足，这可能受限于指示网格的分辨率。
*   **稀疏视角和挑战性光照：** 在非常稀疏的视角（例如3个视角）或挑战性光照条件下，渲染质量会显著下降。

**5. 潜在的未来研究方向：**
*   **分层或多尺度架构：** 探索分层或多尺度架构以提高细节恢复能力。
*   **扩散先验：** 利用扩散先验来改进自遮挡区域的渲染。
*   **基于视频的训练：** 采用基于视频的训练方法，以增强时间一致性并实现可动画的重建。

总而言之，HART论文提出了一种新颖且统一的框架，通过结合Transformer的强大预测能力、遮挡感知几何重建和SMPL-X模型对齐，显著提升了稀疏视角人体重建的质量和鲁棒性。其在多个任务上的优异表现，特别是其在真实世界图像上的泛化能力，为未来的人体3D重建研究奠定了坚实的基础。

**Key Findings:**

- We introduce HART, a unified framework for sparse-view human reconstruction.
- Given a small set of uncalibrated RGB images of a person as input, it outputs a
watertight clothed mesh, the aligned SMPL-X body mesh, and a Gaussian-splat
representation for photorealistic novel-view rendering.
- While trained on only 2.3K synthetic
scans, HART achieves state-of-the-art results: Chamfer Distance improves by
18-23 percent for clothed-mesh reconstruction, PA-V2V drops by 6-27 percent for
SMPL-X estimation, LPIPS decreases by 15-27 percent for novel-view synthesis on
a wide range of datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26621v1)
- [arXiv](https://arxiv.org/abs/2509.26621v1)

---

<a id='2509.26618v1'></a>
## [DA$^2$: Depth Anything in Any Direction](https://arxiv.org/abs/2509.26618v1)

**Authors:** Haodong Li, Wangguangdong Zheng, Jing He, Yuhao Liu, Xin Lin, Xin Yang, Ying-Cong Chen, Chunchao Guo

**Published:** 2025-09-30

**Categories:** cs.CV

**Abstract:**

Panorama has a full FoV (360$^\circ\times$180$^\circ$), offering a more
complete visual description than perspective images. Thanks to this
characteristic, panoramic depth estimation is gaining increasing traction in 3D
vision. However, due to the scarcity of panoramic data, previous methods are
often restricted to in-domain settings, leading to poor zero-shot
generalization. Furthermore, due to the spherical distortions inherent in
panoramas, many approaches rely on perspective splitting (e.g., cubemaps),
which leads to suboptimal efficiency. To address these challenges, we propose
$\textbf{DA}$$^{\textbf{2}}$: $\textbf{D}$epth $\textbf{A}$nything in
$\textbf{A}$ny $\textbf{D}$irection, an accurate, zero-shot generalizable, and
fully end-to-end panoramic depth estimator. Specifically, for scaling up
panoramic data, we introduce a data curation engine for generating high-quality
panoramic depth data from perspective, and create $\sim$543K panoramic
RGB-depth pairs, bringing the total to $\sim$607K. To further mitigate the
spherical distortions, we present SphereViT, which explicitly leverages
spherical coordinates to enforce the spherical geometric consistency in
panoramic image features, yielding improved performance. A comprehensive
benchmark on multiple datasets clearly demonstrates DA$^{2}$'s SoTA
performance, with an average 38% improvement on AbsRel over the strongest
zero-shot baseline. Surprisingly, DA$^{2}$ even outperforms prior in-domain
methods, highlighting its superior zero-shot generalization. Moreover, as an
end-to-end solution, DA$^{2}$ exhibits much higher efficiency over fusion-based
approaches. Both the code and the curated panoramic data will be released.
Project page: https://depth-any-in-any-dir.github.io/.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Haodong Li等人撰写的论文“DA$^2$: Depth Anything in Any Direction”的全面摘要。

---

**论文摘要：DA$^2$: Depth Anything in Any Direction**

**1. 主要问题或研究问题**
该论文旨在解决全景深度估计领域面临的两个核心挑战：
1. **全景数据稀缺性导致泛化能力差：** 现有全景深度估计方法由于缺乏大规模高质量的全景数据，往往局限于特定领域，导致零样本泛化能力不足。
2. **球形畸变带来的效率和准确性问题：** 全景图像固有的球形畸变（尤其是在两极附近）使得许多方法依赖于透视分割（如立方体贴图），这不仅效率低下，也可能导致几何不一致。

**2. 关键创新或方法贡献**
为了解决上述问题，论文提出了DA$^2$（Depth Anything in Any Direction），一个准确、零样本泛化且完全端到端的全景深度估计器，其主要创新包括：

*   **全景数据策展引擎（Panoramic Data Curation Engine）：** 针对全景数据稀缺问题，DA$^2$引入了一个数据策展引擎，能够从现有的高质量透视深度数据中生成全景深度数据。通过透视到等距柱状投影（P2E）和全景图像外绘（使用FLUX-I2P模型），该引擎生成了约543K新的全景RGB-深度对，使总数据集达到约607K，极大地扩展了训练数据量和多样性。
*   **SphereViT架构：** 为缓解球形畸变的影响，DA$^2$提出了SphereViT作为其主要骨干网络。SphereViT通过显式利用全景图像的球形坐标（方位角和极角）来构建球形嵌入（Spherical Embedding）。这些嵌入通过交叉注意力机制与图像特征融合，从而在全景图像特征中强制执行球形几何一致性，生成畸变感知（distortion-aware）的表示，显著提高了性能。
*   **综合基准测试：** 论文构建了一个全面的基准测试，比较了零样本/域内、全景/透视方法，为全景深度估计领域提供了统一的评估框架。
*   **端到端高效解决方案：** DA$^2$作为一个完全端到端的解决方案，相比于基于融合的方法，展现出更高的效率。

**3. 主要结果及其意义**
DA$^2$在多个数据集上的综合基准测试中取得了最先进（SoTA）的性能，具体表现为：

*   **卓越的零样本泛化能力：** DA$^2$在最强的零样本基线上，AbsRel指标平均提高了38%。令人惊讶的是，DA$^2$甚至超越了先前的域内方法，这凸显了其卓越的零样本泛化能力。
*   **几何保真度显著提升：** SphereViT通过强制球形几何一致性，使得DA$^2$能够生成具有出色几何保真度的深度估计，重建的3D结构展现出清晰的几何细节，并在不同场景中表现出鲁棒性。
*   **数据规模化的重要性：** 实验结果（如缩放定律曲线）清晰地表明，随着从透视数据转换而来的全景深度数据量的增加，DA$^2$的性能稳步提升，验证了数据策展引擎的有效性。
*   **高效推理：** 作为端到端方法，DA$^2$在推理效率上远超基于融合的方法。

这些结果表明，通过大规模全景数据和显式建模球形几何，可以实现高质量和鲁棒的360°×180°几何估计，为沉浸式3D场景创建、AR/VR、机器人仿真和物理仿真等应用铺平了道路。

**4. 论文中提到的局限性**
尽管DA$^2$表现出色，论文也提到了其存在的局限性：

*   **分辨率限制：** 训练分辨率（1024x512）低于更高清晰度格式（如2K或4K），可能导致DA$^2$偶尔会遗漏精细细节。
*   **GT深度可用性：** 策展的透视数据在球形空间中仅提供部分可用的GT深度，这可能导致DA$^2$在预测中出现可见的接缝，尤其是在全景图像的左右边界处，这些边界理想情况下应无缝对齐。

**5. 潜在的未来研究方向**
虽然论文没有明确列出未来的研究方向，但从其局限性中可以推断出以下几点：

*   **更高分辨率的全景深度估计：** 探索如何将DA$^2$扩展到更高分辨率的输入，以捕捉更精细的几何细节。
*   **改进全景边界的无缝对齐：** 研究新的方法或损失函数，以更好地处理全景图像的左右边界，确保预测的深度图在这些区域无缝对齐。
*   **更完善的GT深度生成：** 进一步优化数据策展引擎，特别是外绘深度图的绝对精度，以减少对GT深度可用性的依赖。
*   **探索其他球形表示：** 除了等距柱状投影，可以探索其他球形表示（如立方体贴图的更高效集成或更复杂的网格表示），并将其与SphereViT结合，以进一步优化性能。
*   **多模态融合：** 结合其他模态（如LiDAR、惯性测量单元等）来进一步提高全景深度估计的精度和鲁棒性。

---

**Key Findings:**

- To address these challenges, we propose
$\textbf{DA}$$^{\textbf{2}}$: $\textbf{D}$epth $\textbf{A}$nything in
$\textbf{A}$ny $\textbf{D}$irection, an accurate, zero-shot generalizable, and
fully end-to-end panoramic depth estimator.
- Specifically, for scaling up
panoramic data, we introduce a data curation engine for generating high-quality
panoramic depth data from perspective, and create $\sim$543K panoramic
RGB-depth pairs, bringing the total to $\sim$607K.
- To further mitigate the
spherical distortions, we present SphereViT, which explicitly leverages
spherical coordinates to enforce the spherical geometric consistency in
panoramic image features, yielding improved performance.
- Surprisingly, DA$^{2}$ even outperforms prior in-domain
methods, highlighting its superior zero-shot generalization.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26618v1)
- [arXiv](https://arxiv.org/abs/2509.26618v1)

---

<a id='2509.26604v1'></a>
## [Video Object Segmentation-Aware Audio Generation](https://arxiv.org/abs/2509.26604v1)

**Authors:** Ilpo Viertola, Vladimir Iashin, Esa Rahtu

**Published:** 2025-09-30

**Categories:** cs.CV

**Abstract:**

Existing multimodal audio generation models often lack precise user control,
which limits their applicability in professional Foley workflows. In
particular, these models focus on the entire video and do not provide precise
methods for prioritizing a specific object within a scene, generating
unnecessary background sounds, or focusing on the wrong objects. To address
this gap, we introduce the novel task of video object segmentation-aware audio
generation, which explicitly conditions sound synthesis on object-level
segmentation maps. We present SAGANet, a new multimodal generative model that
enables controllable audio generation by leveraging visual segmentation masks
along with video and textual cues. Our model provides users with fine-grained
and visually localized control over audio generation. To support this task and
further research on segmentation-aware Foley, we propose Segmented Music Solos,
a benchmark dataset of musical instrument performance videos with segmentation
information. Our method demonstrates substantial improvements over current
state-of-the-art methods and sets a new standard for controllable,
high-fidelity Foley synthesis. Code, samples, and Segmented Music Solos are
available at https://saganet.notion.site

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Ilpo Viertola, Vladimir Iashin, Esa Rahtu撰写的论文“Video Object Segmentation-Aware Audio Generation”的全面摘要。

---

### 论文《Video Object Segmentation-Aware Audio Generation》摘要

**1. 主要问题或研究问题：**
现有多模态音频生成模型在专业拟音（Foley）工作流程中缺乏精确的用户控制。这些模型通常关注整个视频，无法提供精确的方法来优先处理场景中的特定对象、避免生成不必要的背景声音或错误地聚焦于非目标对象。这限制了它们在需要精细控制的音频合成场景中的应用。因此，该论文旨在解决如何实现对视频中特定对象声音的精细、视觉局部化控制，从而生成高质量、高保真的拟音。

**2. 关键创新或方法论贡献：**
为了解决上述问题，论文提出了以下关键创新和贡献：

*   **新型任务：视频对象分割感知音频生成（Video Object Segmentation-aware Audio Generation）**：该任务明确地将声音合成与对象级别的分割图进行条件化，从而实现对特定对象声音的生成。
*   **SAGANet 模型**：提出了一种新的多模态生成模型SAGANet，它通过利用视觉分割掩码、视频和文本线索，实现可控的音频生成。SAGANet在预训练的MMAudio模型基础上，引入了一个自监督控制模块，该模块融合了全局和局部视觉信息，并通过门控交叉注意力（Gated Cross-Attention）机制实现精细的集成。
*   **Focal Prompt 机制**：为了提供详细信息并结合全局上下文和分割信息，SAGANet引入了Focal Prompt，它包含原始未修改的视觉流及其掩码流，以及围绕感兴趣区域裁剪的视频流及其掩码流，从而提供全局概览和目标区域的详细视图。
*   **Localized Vision Backbone with Temporal Mask Embedding**：通过将视频和掩码流嵌入共享的时空表示中，并应用可学习的位置编码，SAGANet能够捕获精细的空间线索及其时间动态，这对于生成语义对齐的音频至关重要。
*   **Segmented Music Solos 数据集**：为了支持这项新任务和进一步研究分割感知拟音，论文构建了一个基准数据集，包含带有分割信息的乐器演奏视频。该数据集通过结合Solos、AVSBench和MUSIC21等数据集，并经过严格的视觉和听觉验证以及掩码生成流程（利用GroundedSAM2框架），确保了高质量的带声音对象分割图和高音频-视频对应关系。

**3. 主要结果及其意义：**
SAGANet在Segmented Music Solos数据集上的实验结果表明，它在所有评估指标上均显著优于基线模型MMAudio：

*   **分布匹配**：SAGANet在Fréchet Distance (FD) 和Kullback-Leibler Distance (KL) 等指标上表现更好，表明生成音频的分布更接近真实音频。
*   **音频质量**：Inception Score (IS) 评分更高，表明生成的音频具有更高的客观质量和多样性。
*   **语义对齐**：ImageBind (IB-score) 评分更高，证明模型能够更好地聚焦于正确的对象，实现更强的语义相似性。
*   **时间对齐**：DeSync（绝对偏移预测）指标显著降低，表明SAGANet在时间同步方面表现出色，解决了现有模型在复杂场景中难以对齐的问题。

这些结果的意义在于，SAGANet通过引入对象级分割控制，极大地提升了拟音合成的精确性和可控性，使其在多源复杂场景中能够有效聚焦于目标对象，即使在仅使用单源样本进行训练的情况下也能泛化到多源场景。LoRA微调进一步提升了模型的性能。

**4. 论文中提及的局限性：**
论文中提及的局限性主要包括：

*   **数据生成中的手动标注限制**：在测试数据中，目标对象的定位坐标是手动提供的，尽管这产生了更连贯的掩码，但有限的资源阻碍了在训练数据中手动标注所有目标对象。
*   **基线模型在复杂场景中的局限性**：基线MMAudio模型在复杂场景中，仅通过文本描述和视觉输入，难以提供足够强的指导来生成时间上和语义上对齐的音频，因为它会聚焦于场景中的其他乐器。

**5. 潜在的未来研究方向：**
论文为未来的研究方向奠定了基础，包括：

*   **更用户友好的拟音模型开发**：通过提供更精细、视觉局部化的控制，SAGANet为开发更易于使用和更强大的拟音模型铺平了道路，这些模型可以更好地集成到视频后期制作工作流程中。
*   **扩展到更广泛的场景和对象类型**：目前数据集主要关注乐器演奏，未来可以探索将分割感知音频生成扩展到更广泛的日常场景和对象类型。
*   **减少对手动标注的依赖**：进一步研究如何减少在数据生成过程中对目标对象手动标注的依赖，例如通过更先进的零样本或少样本分割方法。
*   **探索其他控制信号**：除了分割掩码、视频和文本，还可以探索其他模态或控制信号，以进一步增强音频生成的精细控制能力。

---

**Key Findings:**

- To address
this gap, we introduce the novel task of video object segmentation-aware audio
generation, which explicitly conditions sound synthesis on object-level
segmentation maps.
- We present SAGANet, a new multimodal generative model that
enables controllable audio generation by leveraging visual segmentation masks
along with video and textual cues.
- To support this task and
further research on segmentation-aware Foley, we propose Segmented Music Solos,
a benchmark dataset of musical instrument performance videos with segmentation
information.
- Our method demonstrates substantial improvements over current
state-of-the-art methods and sets a new standard for controllable,
high-fidelity Foley synthesis.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26604v1)
- [arXiv](https://arxiv.org/abs/2509.26604v1)

---

<a id='2509.26555v1'></a>
## [Stable Cinemetrics : Structured Taxonomy and Evaluation for Professional Video Generation](https://arxiv.org/abs/2509.26555v1)

**Authors:** Agneet Chatterjee, Rahim Entezari, Maksym Zhuravinskyi, Maksim Lapin, Reshinth Adithyan, Amit Raj, Chitta Baral, Yezhou Yang, Varun Jampani

**Published:** 2025-09-30

**Categories:** cs.CV

**Abstract:**

Recent advances in video generation have enabled high-fidelity video
synthesis from user provided prompts. However, existing models and benchmarks
fail to capture the complexity and requirements of professional video
generation. Towards that goal, we introduce Stable Cinemetrics, a structured
evaluation framework that formalizes filmmaking controls into four
disentangled, hierarchical taxonomies: Setup, Event, Lighting, and Camera.
Together, these taxonomies define 76 fine-grained control nodes grounded in
industry practices. Using these taxonomies, we construct a benchmark of prompts
aligned with professional use cases and develop an automated pipeline for
prompt categorization and question generation, enabling independent evaluation
of each control dimension. We conduct a large-scale human study spanning 10+
models and 20K videos, annotated by a pool of 80+ film professionals. Our
analysis, both coarse and fine-grained reveal that even the strongest current
models exhibit significant gaps, particularly in Events and Camera-related
controls. To enable scalable evaluation, we train an automatic evaluator, a
vision-language model aligned with expert annotations that outperforms existing
zero-shot baselines. SCINE is the first approach to situate professional video
generation within the landscape of video generative models, introducing
taxonomies centered around cinematic controls and supporting them with
structured evaluation pipelines and detailed analyses to guide future research.

**Analysis:**

以下是对Agneet Chatterjee等人撰写的论文“Stable Cinemetrics : Structured Taxonomy and Evaluation for Professional Video Generation”的全面摘要：

**1. 主要问题或研究问题：**
当前视频生成模型和基准未能捕捉专业视频生成所需的复杂性和要求。该研究旨在解决一个核心问题：“当前的视频生成模型是否已为专业用途做好准备？”换句话说，论文旨在弥合休闲探索性视频合成与支持专业级、可控电影输出的媒体之间的差距。

**2. 关键创新或方法论贡献：**
*   **Stable Cinemetrics (SCINE) 框架：** 引入了一个结构化的评估框架，将电影制作控制形式化为四个解耦的、分层的分类法：**设置 (Setup)、事件 (Event)、灯光 (Lighting) 和摄像机 (Camera)**。这些分类法共定义了76个基于行业实践的细粒度控制节点。
*   **专业对齐的基准提示：** 利用这些分类法构建了一个与专业用例对齐的提示基准，包括“故事驱动型”和“视觉阐述型”两种提示类型，以模拟实际的电影制作流程。
*   **自动化评估流程：** 开发了一个自动化流程，用于提示分类和问题生成，从而能够独立评估每个控制维度。
*   **大规模人工研究：** 对10多个模型和2万个视频进行了大规模人工研究，由80多位电影专业人士进行标注，确保了评估的高质量和专业性。
*   **自动评估器：** 训练了一个视觉-语言模型（VLM）作为自动评估器，该模型与专家标注对齐，性能优于现有的零样本基线，实现了可扩展的评估。

**3. 主要结果及其意义：**
*   **当前模型的显著差距：** 粗粒度和细粒度分析均显示，即使是最强大的当前模型也存在显著差距，尤其是在**事件 (Events)** 和**摄像机 (Camera)** 相关控制方面。
*   **不同控制维度的性能差异：** 模型在“设置 (Setup)”和“灯光 (Lighting)”方面表现相对较好，但在“事件 (Events)”和“摄像机 (Camera)”方面表现较弱。例如，模型在处理原子动作方面表现良好，但在因果和重叠事件方面表现不佳；在灯光方面，模型在自然光源（如阳光、频闪）方面表现较好，但在HMI、荧光灯和钨丝灯方面表现较差。
*   **VLM评估器的有效性：** 训练的VLM在与人类标注对齐方面表现出一致性，优于零样本基线，证明了其在专业视频生成评估中的可扩展性潜力。
*   **对未来研究的指导：** SCINE是第一个将专业视频生成置于视频生成模型领域的方法，通过引入以电影控制为中心的分类法、结构化评估流程和详细分析来指导未来的研究方向。

**4. 论文中提到的局限性：**
*   **分类法的范围：** 尽管分类法是与领域专家协商开发的，但其范围受限于合作者网络。电影制作术语和解释性细微差别因地区和文化而异，更广泛的专家多样性将有助于纳入全球电影控制。
*   **某些节点的抽象：** 某些分类法节点（例如色温、ISO）被抽象化以进行评估，因为标注者难以始终感知细粒度值。
*   **提示生成中的LLM偏见：** 提示生成依赖于LLM，其专有性质和潜在偏见可能会影响提示的语言和结构。
*   **VLM评估的计算和数据限制：** 零样本VLM评估受限于计算和数据资源，限制了实验的规模和范围。

**5. 潜在的未来研究方向：**
*   **扩展分类法：** 进一步扩展分类法，纳入更广泛的全球电影控制和更细粒度的控制值。
*   **VLM的应用：** 将VLM应用于分析视频数据集的电影多样性或作为视频字幕的结构。
*   **模型改进：** 解决当前模型在事件和摄像机控制方面的显著差距，特别是通过微调和定制化，使生成模型更接近实际生产需求。
*   **更深入的探索：** 鼓励在电影制作和视频生成模型交叉领域进行更深入的探索，促进艺术家和模型之间更紧密的合作。

总而言之，这篇论文通过引入Stable Cinemetrics框架，为专业视频生成提供了一个急需的结构化评估方法，揭示了当前模型在满足专业级电影制作要求方面的不足，并为未来的研究和模型开发指明了方向。

**Key Findings:**

- Towards that goal, we introduce Stable Cinemetrics, a structured
evaluation framework that formalizes filmmaking controls into four
disentangled, hierarchical taxonomies: Setup, Event, Lighting, and Camera.
- To enable scalable evaluation, we train an automatic evaluator, a
vision-language model aligned with expert annotations that outperforms existing
zero-shot baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26555v1)
- [arXiv](https://arxiv.org/abs/2509.26555v1)

---

<a id='2509.26548v1'></a>
## [Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework](https://arxiv.org/abs/2509.26548v1)

**Authors:** Jian Guo Pan, Lin Wang, Xia Cai

**Published:** 2025-09-30

**Categories:** cond-mat.mtrl-sci, cs.CV

**Abstract:**

Scanning Electron Microscopy (SEM) is indispensable for characterizing the
microstructure of thin films during perovskite solar cell fabrication. Accurate
identification and quantification of lead iodide and perovskite phases are
critical because residual lead iodide strongly influences crystallization
pathways and defect formation, while the morphology of perovskite grains
governs carrier transport and device stability. Yet current SEM image analysis
is still largely manual, limiting throughput and consistency. Here, we present
an automated deep learning-based framework for SEM image segmentation that
enables precise and efficient identification of lead iodide, perovskite and
defect domains across diverse morphologies. Built upon an improved YOLOv8x
architecture, our model named PerovSegNet incorporates two novel modules: (i)
Adaptive Shuffle Dilated Convolution Block, which enhances multi-scale and
fine-grained feature extraction through group convolutions and channel mixing;
and (ii) Separable Adaptive Downsampling module, which jointly preserves
fine-scale textures and large-scale structures for more robust boundary
recognition. Trained on an augmented dataset of 10,994 SEM images, PerovSegNet
achieves a mean Average Precision of 87.25% with 265.4 Giga Floating Point
Operations, outperforming the baseline YOLOv8x-seg by 4.08%, while reducing
model size and computational load by 24.43% and 25.22%, respectively. Beyond
segmentation, the framework provides quantitative grain-level metrics, such as
lead iodide/perovskite area and count, which can serve as reliable indicators
of crystallization efficiency and microstructural quality. These capabilities
establish PerovSegNet as a scalable tool for real-time process monitoring and
data-driven optimization of perovskite thin-film fabrication.The source code is
available at:https://github.com/wlyyj/PerovSegNet/tree/master.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Jian Guo Pan, Lin Wang, Xia Cai撰写的论文“Automated and Scalable SEM Image Analysis of Perovskite Solar Cell Materials via a Deep Segmentation Framework”的全面摘要。

**论文摘要：**

1.  **主要问题或研究问题：**
    钙钛矿太阳能电池（PSCs）薄膜的微观结构表征对于其性能和稳定性至关重要。扫描电子显微镜（SEM）是分析这些薄膜微观结构（如碘化铅、钙钛矿相和缺陷）的不可或缺的工具。然而，当前的SEM图像分析主要依赖手动操作，这限制了分析的吞吐量、一致性，并且难以准确识别细粒度特征和复杂背景。因此，论文旨在解决如何实现钙钛矿太阳能电池材料SEM图像的自动化、精确且可扩展的分割和量化分析，以克服手动分析的局限性。

2.  **关键创新或方法贡献：**
    为了解决上述问题，作者提出了一个名为 **PerovSegNet** 的自动化深度学习分割框架。该框架基于改进的YOLOv8x架构，并引入了两个新颖的模块：
    *   **(i) 自适应混洗膨胀卷积块 (Adaptive Shuffle Dilated Convolution Block, ASDCB)：** 该模块通过组卷积和通道混合增强多尺度和细粒度特征提取，从而提高网络区分晶界、小颗粒和缺陷区域的能力。
    *   **(ii) 可分离自适应下采样模块 (Separable Adaptive Downsampling module, SAD)：** 该模块通过结合深度可分离卷积和自适应池化机制，共同保留细尺度纹理和大尺度结构，以实现更鲁棒的边界识别，有效缓解传统下采样方法中常见的混叠和对小尺度纹理敏感度有限的问题。
    此外，为了克服带注释SEM数据稀缺的挑战，作者构建了一个包含10,994张增强SEM图像的 **PerovData 数据集**，并使用UMAP分析验证了特征的可分离性和形态多样性。

3.  **主要结果及其意义：**
    *   **卓越的分割性能：** PerovSegNet 在增强的PerovData数据集上实现了87.25%的平均精度（mAP@0.5），显著优于基线YOLOv8x-seg模型4.08%。
    *   **计算效率提升：** 模型尺寸和计算负载分别减少了24.43%和25.22%，同时保持了高性能，使其成为一个轻量级且高效的解决方案。
    *   **定性分析优势：** 相较于Mask R-CNN和Cascade InternImage-XL等基线模型，PerovSegNet能够更清晰地描绘边界，更可靠地分离密集堆积的晶粒，并显著减少漏检的碘化铅簇和缺陷区域。
    *   **定量微观结构指标：** 除了分割，该框架还提供了定量的晶粒级指标，如碘化铅/钙钛矿的面积和数量，这些指标可作为结晶效率和微观结构质量的可靠指示器。
    *   **与器件性能的关联：** 论文通过相关性分析（Pearson r）展示了图像衍生的微观结构描述符（如钙钛矿面积、缺陷密度）与器件光伏转换效率（PCE）之间的关系，为数据驱动的工艺优化提供了基础。
    *   **尺度不变性分析：** PerovSegNet在不同SEM分辨率下（从200 nm到2 µm）均表现出良好的分割能力，证明了其在不同实验条件下适应SEM数据的鲁棒性。

    这些结果表明，PerovSegNet是一个可扩展的工具，可用于实时过程监控和数据驱动的钙钛矿薄膜制造优化。

4.  **论文中提及的局限性：**
    *   **相关性非因果性：** 报告的相关性是观察性的，不暗示因果关系。其他因素（如成分或界面工程）可能同时影响形态和PCE。
    *   **误差传播：** 分割误差和类别不平衡（特别是缺陷类别）可能传播到派生指标中。
    *   **成像条件影响：** 成像条件（电压、放大倍数、校准）也会影响定量输出。
    *   **数据集规模和多样性：** 解决上述问题需要更大、更多样化的数据集，以及不确定性量化和跨放大倍数验证。

5.  **潜在的未来研究方向：**
    *   **扩展到其他材料系统：** 将该框架扩展到其他材料系统。
    *   **进一步加速推理：** 进一步加速推理，以实现原位（in situ）或在线（inline）应用。
    *   **数据驱动的工艺优化：** 利用PerovSegNet提供的定量微观结构指标，通过数据驱动的反馈循环，调整工艺参数以实现目标形态。
    *   **不确定性量化和跨放大倍数验证：** 进一步研究以提高模型的鲁棒性和泛化能力。

总而言之，PerovSegNet通过其创新的模块设计和对大规模数据集的训练，在钙钛矿太阳能电池材料的SEM图像分析方面取得了显著进展，为材料科学领域的自动化微观结构表征和优化提供了强大且高效的工具。

**Key Findings:**

- Here, we present
an automated deep learning-based framework for SEM image segmentation that
enables precise and efficient identification of lead iodide, perovskite and
defect domains across diverse morphologies.
- Built upon an improved YOLOv8x
architecture, our model named PerovSegNet incorporates two novel modules: (i)
Adaptive Shuffle Dilated Convolution Block, which enhances multi-scale and
fine-grained feature extraction through group convolutions and channel mixing;
and (ii) Separable Adaptive Downsampling module, which jointly preserves
fine-scale textures and large-scale structures for more robust boundary
recognition.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.26548v1)
- [arXiv](https://arxiv.org/abs/2509.26548v1)

---

