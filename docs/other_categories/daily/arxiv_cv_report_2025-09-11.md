time: 20250911

# Arxiv Computer Vision Papers - 2025-09-11

## Executive Summary

好的，这是一份针对2025年9月10日Arxiv计算机视觉领域论文的执行摘要，旨在帮助忙碌的研究人员快速了解最新进展。

---

**Arxiv 计算机视觉领域最新论文执行摘要 (2025年9月10日)**

**1. 主要主题和趋势概述：**

本次报告涵盖的10篇论文展现了计算机视觉领域几个关键且相互关联的趋势：

*   **多模态融合与大型模型（LLMs/LVLMs）的崛起：** 多篇论文探索了视觉与语言或其他模态的融合，特别是将大型视觉语言模型（LVLMs）应用于更复杂的感知任务，如水下目标检测和高效的视觉-语言理解。
*   **自动驾驶感知与地图构建的持续关注：** 自动驾驶仍然是研究热点，论文涉及基础模型在自动驾驶感知中的应用、高精地图的自更新机制以及拥挤场景下的检测优化。
*   **3D 视觉与重建的进步：** 3D 点云重建、单图像到3D对象生成以及6D姿态估计是重要的研究方向，强调了鲁棒性和生成能力。
*   **以人为中心的视觉理解与生成：** 视频生成领域开始关注以人为中心的生成，通过多模态条件控制实现更自然的视频内容。
*   **鲁棒性与效率的追求：** 无论是水下环境、拥挤场景还是3D重建，研究人员都在努力提升模型的鲁棒性、准确性和计算效率。

**2. 特别重要或创新的论文亮点：**

*   **"Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities" (Rajendramayavan Sathyam, Yueqi Li):** 这篇综述性论文非常及时和重要。它系统地梳理了基础模型在自动驾驶感知中的应用，为该领域的未来研究提供了清晰的路线图和挑战分析，对于理解宏观趋势至关重要。
*   **"BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion" (Sike Xiang, Shuang Chen, Amir Atapour-Abarghouei):** 这篇论文在多模态融合方面展现了创新性，通过蒸馏和Q门控交叉模态融合，解决了LVLMs在效率和理解能力上的平衡问题，对于资源受限的应用场景具有重要意义。
*   **"One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation" (Zheng Geng et al.):** 这篇论文结合了单图像到3D生成和生成式域随机化，以解决单次6D姿态估计的挑战。其创新点在于利用生成模型来增强训练数据的多样性和鲁棒性，为解决数据稀缺问题提供了新思路。
*   **"HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning" (Liyang Chen et al.):** 在生成式AI日益成熟的背景下，这篇论文专注于以人为中心的视频生成，通过多模态条件控制实现更精细、更可控的生成，预示着未来视频内容创作的新方向。

**3. 新兴研究方向或技术：**

*   **基础模型在特定领域（如自动驾驶、水下视觉）的深入应用与适应：** 不再仅仅是通用模型，而是如何将基础模型的能力有效迁移和优化到特定、复杂的应用场景。
*   **高效且鲁棒的多模态融合架构：** 如何在保持高性能的同时，降低大型多模态模型的计算成本和复杂性，例如通过蒸馏、门控机制等。
*   **自更新/自适应系统：** 如高精地图的自更新（ArgoTweak），以及未来可能出现的自适应感知系统，能够根据环境变化进行自我调整和优化。
*   **生成式AI在数据增强和鲁棒性提升中的应用：** 利用生成模型创建多样化的训练数据，以提高模型在真实世界复杂场景中的泛化能力和鲁棒性。
*   **以人为中心的生成与理解：** 随着AIGC的发展，对人类行为、姿态、交互的精确理解和生成将成为关键。

**4. 建议阅读全文的论文：**

为了全面了解这些领域的最新进展，我建议您优先阅读以下论文：

*   **"Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities" (Rajendramayavan Sathyam, Yueqi Li):** 对于任何从事自动驾驶或对基础模型应用感兴趣的研究人员，这篇综述是必读的，它提供了高层次的概览和未来方向。
*   **"BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion" (Sike Xiang, Shuang Chen, Amir Atapour-Abarghouei):** 如果您关注多模态学习的效率和架构创新，这篇论文提供了具体的技术细节和潜在的解决方案。
*   **"One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation" (Zheng Geng et al.):** 对于3D视觉、姿态估计以及如何利用生成式AI解决数据挑战感兴趣的研究人员，这篇论文提供了创新的方法。
*   **"ArgoTweak: Towards Self-Updating HD Maps through Structured Priors" (Lena Wild, Rafael Valencia, Patric Jensfelt):** 对于自动驾驶中的高精地图和基础设施维护感兴趣的研究人员，这篇论文提供了实用的工程解决方案和研究方向。

---

这份摘要希望能为您提供一个快速而全面的概览，帮助您高效地筛选和深入研究最相关的论文。

---

## Table of Contents

1. [Computational Imaging for Enhanced Computer Vision](#2509.08712v1)
2. [A Structured Review of Underwater Object Detection Challenges and Solutions: From Traditional to Large Vision Language Models](#2509.08490v1)
3. [Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities](#2509.08302v1)
4. [ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](#2509.08764v1)
5. [CrowdQuery: Density-Guided Query Module for Enhanced 2D and 3D Detection in Crowded Scenes](#2509.08738v1)
6. [BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion](#2509.08715v1)
7. [TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals](#2509.08699v1)
8. [HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning](#2509.08519v1)
9. [APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction](#2509.08104v1)
10. [One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation](#2509.07978v1)

---

## Papers

<a id='2509.08712v1'></a>
## [Computational Imaging for Enhanced Computer Vision](https://arxiv.org/abs/2509.08712v1)

**Authors:** Humera Shaikh, Kaur Jashanpreet

**Published:** 2025-09-10

**Categories:** cs.CV

**Abstract:**

This paper presents a comprehensive survey of computational imaging (CI)
techniques and their transformative impact on computer vision (CV)
applications. Conventional imaging methods often fail to deliver high-fidelity
visual data in challenging conditions, such as low light, motion blur, or high
dynamic range scenes, thereby limiting the performance of state-of-the-art CV
systems. Computational imaging techniques, including light field imaging, high
dynamic range (HDR) imaging, deblurring, high-speed imaging, and glare
mitigation, address these limitations by enhancing image acquisition and
reconstruc- tion processes. This survey systematically explores the synergies
between CI techniques and core CV tasks, including object detection, depth
estimation, optical flow, face recognition, and keypoint detection. By
analyzing the relationships between CI methods and their practical
contributions to CV applications, this work highlights emerging opportunities,
challenges, and future research directions. We emphasize the potential for
task-specific, adaptive imaging pipelines that improve robustness, accuracy,
and efficiency in real-world scenarios, such as autonomous navigation,
surveillance, augmented reality, and robotics.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Humera Shaikh和Kaur Jashanpreet撰写的论文“Computational Imaging for Enhanced Computer Vision”的全面摘要。

---

### 论文《Computational Imaging for Enhanced Computer Vision》摘要

**1. 主要问题或研究问题：**
该论文旨在解决传统成像方法在挑战性条件下（如低光照、运动模糊、高动态范围场景）无法提供高保真视觉数据，从而限制了最先进计算机视觉（CV）系统性能的问题。研究的核心在于探索计算成像（CI）技术如何通过增强图像采集和重建过程来克服这些限制，并提升核心CV任务的性能。

**2. 关键创新或方法论贡献：**
论文的主要贡献在于对计算成像技术及其对计算机视觉应用的变革性影响进行了全面而系统的综述。它详细阐述了以下关键CI技术：
*   **光场成像（Light Field Imaging）：** 通过捕捉光线的角度方向，提供多视角视图，显著增强深度估计、物体分割和遮挡处理，对3D重建和面部识别至关重要。
*   **高动态范围（HDR）成像：** 通过融合不同曝光水平的图像，捕捉亮区和暗区的细节，克服传统传感器的限制，提升高对比度环境下的物体检测和面部识别性能。
*   **图像去模糊（Image Deblurring）：** 通过建模和逆转运动模糊或散焦模糊，恢复图像清晰度，改善光学流估计、运动跟踪和物体检测。
*   **高速成像（High-Speed Imaging）：** 捕捉快速时间事件，提供精细的运动细节，对物体跟踪、光学流计算和运动分析至关重要。
*   **眩光缓解（Glare Mitigation）：** 通过偏振成像或多曝光方法抑制反射和强光源引起的伪影，确保在反射或不均匀光照环境下的CV系统准确性。

论文系统地探讨了这些CI技术与核心CV任务（包括物体检测、深度估计、光学流、人脸识别和关键点检测）之间的协同作用，分析了CI方法如何解决现有CV算法的局限性。

**3. 主要结果及其意义：**
论文通过分析CI方法与CV应用之间的关系，强调了CI技术对CV性能的显著提升：
*   **深度估计和关键点检测：** 光场成像通过提供空间和角度信息，显著提高了这些任务的准确性和鲁棒性，尤其是在遮挡和无纹理区域。
*   **物体检测和人脸识别：** HDR成像在高对比度环境下（如自动驾驶和监控）确保了细节可见性，而光场成像则通过多视角信息增强了人脸识别在不同姿态和遮挡下的鲁棒性。
*   **光学流和运动分析：** 高速成像和去模糊技术通过提供清晰的运动细节，显著提高了光学流估计和物体跟踪的准确性，尤其是在动态场景中。
*   **跨任务协同：** 论文指出，CI技术带来的改进并非孤立的，而是通过丰富的数据表示在不同CV任务之间产生协同效应，例如深度估计的改进可以间接提升物体检测和关键点匹配。

这些结果表明，CI技术为CV系统提供了更可靠、更丰富、更具信息量的输入，从而显著提高了其在复杂和挑战性现实场景中的准确性、鲁棒性和效率。

**4. 论文中提及的局限性：**
论文也指出了CI技术与CV集成过程中面临的一些挑战：
*   **计算复杂性：** 许多CI方法（如光场成像和多曝光HDR）涉及大量数据和高计算需求，尤其是在实时应用中。
*   **硬件与软件的权衡：** CI技术依赖专业光学设计和传感器架构，需要在物理复杂性、成本和可扩展性之间取得平衡。
*   **数据质量和泛化性：** 尽管CI提高了数据保真度，但在稀疏角度采样或快速运动等特定成像条件下，仍可能出现伪影或性能下降。
*   **互操作性：** 新兴CI硬件与现有CV框架之间的互操作性需要标准化管道和数据集。

**5. 潜在的未来研究方向：**
论文提出了几个未来研究方向和机会：
*   **任务特定、自适应成像管道：** 开发能够根据特定CV任务需求动态调整图像采集过程的CI系统，例如光场相机根据场景深度复杂性调整角度采样。
*   **深度学习与数据驱动方法：** 将深度学习与CI管道结合，利用神经网络学习原始传感器数据与任务特定输出之间的端到端映射，以实现更快、更准确、更灵活的成像解决方案。
*   **多模态成像系统集成：** 结合空间、角度、时间、光谱等多种成像方法，以获得更丰富的场景表示，从而在复杂环境中实现更鲁棒的CV性能。
*   **边缘计算和AI加速器：** 利用轻量级神经网络和高效硬件架构，在资源受限的边缘设备和移动平台上实现CI增强型CV系统的实时部署。

---

总而言之，这篇论文全面概述了计算成像技术如何通过提供更丰富、更准确的视觉数据来克服传统成像的局限性，从而显著增强了计算机视觉应用的性能。它不仅详细介绍了各种CI方法及其对核心CV任务的贡献，还强调了这些技术之间的协同作用，并指出了未来研究的挑战和机遇，为该领域的进一步发展奠定了基础。

**Key Findings:**

- Conventional imaging methods often fail to deliver high-fidelity
visual data in challenging conditions, such as low light, motion blur, or high
dynamic range scenes, thereby limiting the performance of state-of-the-art CV
systems.
- By
analyzing the relationships between CI methods and their practical
contributions to CV applications, this work highlights emerging opportunities,
challenges, and future research directions.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08712v1)
- [arXiv](https://arxiv.org/abs/2509.08712v1)

---

<a id='2509.08490v1'></a>
## [A Structured Review of Underwater Object Detection Challenges and Solutions: From Traditional to Large Vision Language Models](https://arxiv.org/abs/2509.08490v1)

**Authors:** Edwine Nabahirwa, Wei Song, Minghua Zhang, Yi Fang, Zhou Ni

**Published:** 2025-09-10

**Categories:** cs.CV, cs.AI

**Abstract:**

Underwater object detection (UOD) is vital to diverse marine applications,
including oceanographic research, underwater robotics, and marine conservation.
However, UOD faces numerous challenges that compromise its performance. Over
the years, various methods have been proposed to address these issues, but they
often fail to fully capture the complexities of underwater environments. This
review systematically categorizes UOD challenges into five key areas: Image
quality degradation, target-related issues, data-related challenges,
computational and processing constraints, and limitations in detection
methodologies. To address these challenges, we analyze the progression from
traditional image processing and object detection techniques to modern
approaches. Additionally, we explore the potential of large vision-language
models (LVLMs) in UOD, leveraging their multi-modal capabilities demonstrated
in other domains. We also present case studies, including synthetic dataset
generation using DALL-E 3 and fine-tuning Florence-2 LVLM for UOD. This review
identifies three key insights: (i) Current UOD methods are insufficient to
fully address challenges like image degradation and small object detection in
dynamic underwater environments. (ii) Synthetic data generation using LVLMs
shows potential for augmenting datasets but requires further refinement to
ensure realism and applicability. (iii) LVLMs hold significant promise for UOD,
but their real-time application remains under-explored, requiring further
research on optimization techniques.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Edwine Nabahirwa等人的论文“A Structured Review of Underwater Object Detection Challenges and Solutions: From Traditional to Large Vision Language Models”的全面摘要。

**论文摘要：水下目标检测的挑战与大型视觉语言模型的潜力**

**1. 主要问题或研究问题：**
该论文旨在系统性地回顾和分析水下目标检测（UOD）所面临的复杂挑战，并探讨从传统方法到现代深度学习技术（特别是大型视觉语言模型LVLMs）的解决方案。核心研究问题是：如何克服水下环境中图像质量退化、目标特性复杂、数据稀缺、计算资源受限以及现有检测方法局限性等问题，以实现鲁棒、准确且实时的UOD？

**2. 关键创新或方法论贡献：**
*   **系统性挑战分类：** 论文将UOD挑战系统地归纳为五大类：图像质量退化、目标相关问题、数据相关挑战、计算和处理限制以及检测方法局限性。这种结构化的分类为理解UOD的复杂性提供了清晰的框架。
*   **全面回顾解决方案：** 论文详细分析了从传统图像处理（如图像增强、恢复）到现代目标检测技术（如基于深度学习、Transformer和混合模型）的演进，以应对上述挑战。
*   **LVLMs在UOD中的潜力探索：** 论文首次深入探讨了大型视觉语言模型（LVLMs）在UOD领域的应用潜力，强调其多模态能力在解决水下环境复杂性方面的优势。
*   **案例研究：** 论文通过两个具体的案例研究验证了LVLMs的潜力：
    *   **基于DALL-E 3的合成数据生成：** 展示了如何利用LVLMs生成合成水下图像，并通过图像增强技术（如颜色调整、模糊）提高其真实感，以扩充数据集并改善模型性能。
    *   **Florence-2 LVLM的微调：** 探讨了使用LoRA（低秩适应）技术对Florence-2 LVLM进行微调以实现UOD，展示了其在小目标检测和定位方面的强大能力。

**3. 主要结果及其意义：**
*   **现有UOD方法的局限性：** 论文指出，当前的UOD方法在完全解决动态水下环境中的图像退化和小目标检测等挑战方面仍显不足。
*   **合成数据生成的潜力与挑战：** 案例研究表明，利用LVLMs（如DALL-E 3）生成合成数据在扩充数据集方面具有巨大潜力，能够提高模型在混合数据集上的性能（例如，YOLO11在组合数据集上的mAP@50从0.793提高到0.796，召回率从0.714提高到0.736）。然而，合成数据仍需进一步细化以确保真实感和适用性，尤其是在捕捉水下环境的自然复杂性（如浑浊度、光照变化、遮挡）方面。
*   **LVLMs在UOD中的前景与未探索领域：** Florence-2 LVLM的微调实验展示了其在水下小目标定位方面的强大能力。这表明LVLMs在UOD中具有显著潜力，但其在实际应用中仍面临挑战，如类名幻觉（模型生成拼写错误的类名）和灾难性遗忘（模型难以泛化到训练集之外的对象）。此外，LVLMs的实时应用及其优化技术仍有待深入研究。

**4. 论文中提及的局限性：**
*   **合成数据真实感不足：** DALL-E 3生成的合成图像虽然清晰，但缺乏真实水下场景中的自然缺陷，如浑浊度、光照变化和复杂遮挡。
*   **合成数据标注成本：** 生成的合成图像仍需要手动标注，这仍然是一个耗时且资源密集的过程。
*   **LVLMs的幻觉问题：** 微调后的Florence-2模型在生成类名时出现“幻觉”，即生成拼写错误或不准确的类名，严重影响了评估指标的可靠性。
*   **LVLMs的灾难性遗忘：** 微调后的模型难以保留其更广泛的预训练知识，导致在特定任务适应后对训练集之外的对象泛化能力下降。
*   **LVLMs实时应用未充分探索：** 尽管LVLMs潜力巨大，但其在资源受限的水下环境中的实时应用和优化仍是未充分探索的领域。

**5. 潜在的未来研究方向：**
*   **高效微调技术：** 进一步研究适配器微调和提示微调等参数高效技术，以更好地适应LVLMs在低光照、散射和颜色失真等复杂水下条件下的应用，同时最小化计算开销。
*   **更真实的合成数据生成：** 结合扩散模型与其他生成方法（如VAE、GAN），以提高合成水下图像的保真度，更好地捕捉复杂的水下特征和现象。同时，探索自动标注算法以简化标注流程。
*   **数据集标注自动化：** 利用Label-driven Automated Prompt Tuning (LAPT) 等框架，通过自动化提示工程和图像合成/检索方法，减少水下图像的手动标注工作。
*   **轻量级实时处理架构：** 针对AUV和实时监测系统，开发和优化轻量级LVLMs架构，例如采用模型剪枝和Transformer压缩技术，以实现实时检测。
*   **混合方法：** 探索结合多种技术（如图像增强、合成数据、LVLMs）的混合方法，以在复杂水下环境中实现更鲁棒的检测。

总而言之，这篇论文为水下目标检测领域提供了一个全面的视角，不仅清晰地阐述了现有挑战，还开创性地探讨了大型视觉语言模型在解决这些问题上的巨大潜力，并指明了未来研究的关键方向。

**Key Findings:**

- (ii) Synthetic data generation using LVLMs
shows potential for augmenting datasets but requires further refinement to
ensure realism and applicability.
- (iii) LVLMs hold significant promise for UOD,
but their real-time application remains under-explored, requiring further
research on optimization techniques.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08490v1)
- [arXiv](https://arxiv.org/abs/2509.08490v1)

---

<a id='2509.08302v1'></a>
## [Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities](https://arxiv.org/abs/2509.08302v1)

**Authors:** Rajendramayavan Sathyam, Yueqi Li

**Published:** 2025-09-10

**Categories:** cs.RO, cs.CV

**Abstract:**

Foundation models are revolutionizing autonomous driving perception,
transitioning the field from narrow, task-specific deep learning models to
versatile, general-purpose architectures trained on vast, diverse datasets.
This survey examines how these models address critical challenges in autonomous
perception, including limitations in generalization, scalability, and
robustness to distributional shifts. The survey introduces a novel taxonomy
structured around four essential capabilities for robust performance in dynamic
driving environments: generalized knowledge, spatial understanding,
multi-sensor robustness, and temporal reasoning. For each capability, the
survey elucidates its significance and comprehensively reviews cutting-edge
approaches. Diverging from traditional method-centric surveys, our unique
framework prioritizes conceptual design principles, providing a
capability-driven guide for model development and clearer insights into
foundational aspects. We conclude by discussing key challenges, particularly
those associated with the integration of these capabilities into real-time,
scalable systems, and broader deployment challenges related to computational
demands and ensuring model reliability against issues like hallucinations and
out-of-distribution failures. The survey also outlines crucial future research
directions to enable the safe and effective deployment of foundation models in
autonomous driving systems.

**Analysis:**

好的，这是一篇关于Rajendramayavan Sathyam和Yueqi Li撰写的论文“Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities”的全面摘要。

**论文摘要：自动驾驶感知基础模型：基于核心能力的综述**

**1. 主要问题或研究问题：**
该论文旨在解决自动驾驶感知领域面临的核心挑战，即传统窄范围、任务特定深度学习模型在泛化能力、可扩展性以及对分布偏移的鲁棒性方面的局限性。随着基础模型（Foundation Models）的兴起，论文探讨了这些通用型架构如何革新自动驾驶感知，并提出了一个新颖的分类框架来系统地理解和发展这些模型。

**2. 关键创新或方法论贡献：**
该论文的核心创新在于提出了一个以“核心能力”为导向的分类框架，而非传统的以方法或任务为中心的综述。这四项核心能力对于在动态驾驶环境中实现鲁棒性能至关重要：
*   **泛化知识（Generalized Knowledge）：** 模型应能适应广泛的驾驶场景，包括罕见或未曾出现的情况，并能以原则性方式推断结果和处理陌生代理。实现方法包括特征级蒸馏、伪标签监督和直接集成视觉基础模型（VFMs）、视觉语言模型（VLMs）和大型语言模型（LLMs）。
*   **空间理解（Spatial Understanding）：** 模型需对3D空间结构和关系有深刻理解，包括检测已知和未知物体，并推断其物理交互和未来轨迹。主要方法包括显式占用网络（Volumetric Models）、基于神经渲染的2D监督3D学习（如NeRF和3D Gaussian Splatting）以及3D掩码自编码器。
*   **多传感器鲁棒性（Multi-Sensor Robustness）：** 系统应在各种环境条件、传感器噪声和硬件退化下保持高性能。实现方法包括跨模态对比学习、跨模态知识蒸馏、多视角图像一致性、多模态掩码自编码器和多模态扩散模型。
*   **时间推理（Temporal Reasoning）：** 模型需捕捉时间依赖性并预测环境的未来状态，包括建模运动模式、识别被遮挡代理和推理物体持久性。实现方法包括时间一致性4D预测模型（如基于扩散模型）和时间对比学习。

**3. 主要结果及其意义：**
论文通过详细回顾每项能力下的前沿方法，展示了基础模型如何通过大规模、多样化数据集的自监督预训练，学习通用表示并捕获潜在世界知识，从而在自动驾驶感知中实现显著优势。这些模型能够：
*   **提高泛化能力：** 更好地适应长尾场景和未见情况。
*   **增强可扩展性：** 减少对昂贵手动标注数据的依赖。
*   **提升鲁棒性：** 在分布偏移、传感器退化和恶劣天气条件下保持性能。
*   **实现统一表示：** 促进感知任务间的无缝集成，提高对复杂驾驶环境解释的一致性。
*   **支持高级推理：** 结合语言模型实现更高级别的场景理解和规划。

**4. 论文中提到的局限性：**
尽管基础模型潜力巨大，论文也明确指出了当前面临的挑战：
*   **计算成本和延迟：** 大型基础模型的高计算需求和推理延迟与自动驾驶系统的实时性要求（毫秒级）存在冲突。
*   **系统集成复杂性：** 将这些能力整合到统一、实时、可扩展的系统中是一个非平凡的工程任务。
*   **领域鸿沟：** 基础模型在网络规模数据上预训练的通用知识与自动驾驶专用传感器数据（如LiDAR和雷达）的特定要求之间存在差异。
*   **幻觉风险：** 模型可能产生与现实不符的输出，带来严重的安全隐患。
*   **基准测试局限性：** 现有基准测试多关注通用场景，忽视了罕见或安全关键的极端情况，导致模型在实际部署中可靠性不足。
*   **可解释性不足：** 大型感知基础模型的“黑箱”性质阻碍了其在安全关键系统中的部署和监管信任。
*   **数据偏差：** 训练数据可能存在偏差，导致模型在特定场景（如恶劣天气、弱势道路使用者）中表现不佳。
*   **非确定性AI的监管挑战：** 传统汽车安全标准难以验证和认证具有新兴行为的非确定性AI系统。

**5. 潜在的未来研究方向：**
论文提出了以下关键未来研究方向，以实现基础模型在自动驾驶系统中的安全有效部署：
*   **核心能力的集成：** 开发能够无缝整合泛化知识、空间理解、多传感器鲁棒性和时间推理的统一、实时操作框架。
*   **实时延迟缓解：** 深入研究模型优化技术（如量化、剪枝、知识蒸馏）、专用硬件加速器和运行时策略（如多速率异步管道、随时推理）以满足严格的延迟要求。
*   **改进基准测试：** 转向基于场景的测试，结合真实世界数据和合成增强，明确测试鲁棒性，并从聚合精度转向部署相关弹性。
*   **解决数据偏差和确保公平性能：** 开发更具包容性和代表性的数据集，利用数据增强和合成数据，并研究算法公平性和偏差缓解技术。
*   **缓解模型幻觉和安全风险：** 开发不易产生幻觉的新型架构，创建全面的基准测试以评估模型鲁棒性，并实施实时监控系统。
*   **增强可解释性：** 推进可解释AI（XAI）技术，提供模型决策过程的清晰洞察，以建立监管信任并促进故障模式识别。
*   **适应监管挑战：** 发展鲁棒的验证和验证框架，以适应非确定性AI的监管要求。

总而言之，这篇综述为自动驾驶感知领域的基础模型研究提供了一个结构化且富有洞察力的路线图，强调了从能力角度出发进行模型开发和评估的重要性，并指明了未来研究的关键方向。

**Key Findings:**

- The survey introduces a novel taxonomy
structured around four essential capabilities for robust performance in dynamic
driving environments: generalized knowledge, spatial understanding,
multi-sensor robustness, and temporal reasoning.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08302v1)
- [arXiv](https://arxiv.org/abs/2509.08302v1)

---

<a id='2509.08764v1'></a>
## [ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](https://arxiv.org/abs/2509.08764v1)

**Authors:** Lena Wild, Rafael Valencia, Patric Jensfelt

**Published:** 2025-09-10

**Categories:** cs.CV

**Abstract:**

Reliable integration of prior information is crucial for self-verifying and
self-updating HD maps. However, no public dataset includes the required triplet
of prior maps, current maps, and sensor data. As a result, existing methods
must rely on synthetic priors, which create inconsistencies and lead to a
significant sim2real gap. To address this, we introduce ArgoTweak, the first
dataset to complete the triplet with realistic map priors. At its core,
ArgoTweak employs a bijective mapping framework, breaking down large-scale
modifications into fine-grained atomic changes at the map element level, thus
ensuring interpretability. This paradigm shift enables accurate change
detection and integration while preserving unchanged elements with high
fidelity. Experiments show that training models on ArgoTweak significantly
reduces the sim2real gap compared to synthetic priors. Extensive ablations
further highlight the impact of structured priors and detailed change
annotations. By establishing a benchmark for explainable, prior-aided HD
mapping, ArgoTweak advances scalable, self-improving mapping solutions. The
dataset, baselines, map modification toolbox, and further resources are
available at https://kth-rpl.github.io/ArgoTweak/.

**Analysis:**

好的，作为一名计算机视觉和机器学习专家，我将为您提供Lena Wild, Rafael Valencia, Patric Jensfelt撰写的论文“ArgoTweak: Towards Self-Updating HD Maps through Structured Priors”的全面摘要。

---

### 论文摘要：ArgoTweak: Towards Self-Updating HD Maps through Structured Priors

**1. 主要问题或研究问题：**
自动驾驶领域对高精地图（HD maps）的可靠自验证和自更新能力至关重要。然而，现有方法在整合先验信息时面临核心挑战：缺乏包含“先验地图、当前传感器数据和最新真值地图”三元组的公共数据集。这导致现有方法不得不依赖合成先验，从而引入不一致性并造成显著的“模拟到现实”（sim2real）差距，使得模型难以泛化到真实世界场景。此外，现有评估指标未能有效区分未变化区域的性能和新更新区域的性能，且合成扰动方法无法捕捉真实世界变化的结构化、语义相关性质。

**2. 关键创新或方法论贡献：**
*   **ArgoTweak数据集：** 论文引入了ArgoTweak，这是首个包含真实世界先验地图、当前传感器数据和最新真值地图三元组的手动标注数据集。它基于Argoverse 2 Map Change Dataset [12] 构建，并对真实世界变化进行了重新标注，使其符合现代HD地图标准，从而实现了先验整合方法的标准化训练和评估。
*   **双射变化映射框架：** ArgoTweak的核心是一个双射映射框架，它将大规模地图修改分解为地图元素级别的细粒度原子变化（如插入、删除、几何修改、标记更新等），确保了修改的可解释性。这种范式转变使得精确的变化检测和整合成为可能，同时以高保真度保留未变化的元素。
*   **可解释的先验辅助映射网络：** 论文提出了一种灵活的基线架构，能够以不同可解释性级别（无显式变化评估、二元变化检测或完整的可解释性模块）进行操作，并采用多头训练来预测更新后的地图元素及其变化状态。
*   **细粒度评估指标：** 引入了一个全面的双指标框架，包括粗粒度变化检测准确率（mACC）和细粒度地图生成平均精度（mAPC），能够分别评估未变化区域的稳定性以及对更新的响应能力，从而揭示现有指标的不足。

**3. 主要结果及其意义：**
*   **显著减少Sim2Real差距：** 实验表明，在ArgoTweak上训练的模型与使用合成先验的方法相比，显著减少了sim2real差距。对于mACCc的组合指标，ArgoTweak数据集将sim2real差距减少了十倍以上。
*   **结构化先验和详细标注的影响：** 广泛的消融实验进一步强调了结构化先验和详细变化标注对模型性能和可解释性的重要影响。结果表明，ArgoTweak训练的模型能够捕捉复杂的地图更新，而基于规则的先验模型倾向于过拟合车道标记变化，基于噪声的先验仅支持微小的几何校正。
*   **新基准的建立：** ArgoTweak通过提供数据集、基线模型、地图修改工具和评估协议，为可解释的、先验辅助的HD地图绘制建立了新基准，推动了可扩展、自改进的地图解决方案的发展。

**4. 论文中提到的局限性：**
*   **真实世界先验地图的稀缺性：** 真实世界中过时地图的稀缺性使得难以获取足够的真实先验数据进行训练，导致现有方法不得不依赖合成先验。
*   **现有评估指标的不足：** 传统的平均精度（mAP）等指标无法区分模型在未变化区域的性能和新更新区域的性能，也无法反映模型行为的深层差异，从而掩盖了稳定性与适应性之间的关键权衡。
*   **几何精度挑战：** 当前地图生成方法往往缺乏可靠区分细微道路形状修改与噪声所需的几何精度。
*   **计算效率：** 尽管模型未进行速度优化，但在单个NVIDIA A10G GPU上仍能以约4 FPS运行，这对于实时应用可能仍有提升空间。

**5. 潜在的未来研究方向：**
*   **统一地图生成、变化检测和地图更新：** 未来的方法应将这些过程统一起来，通过利用先验、强制一致性以及提供结构化、可解释的大规模修改，使HD地图能够演变为自改进、持续更新的道路表示。
*   **更复杂的原子变化处理：** 进一步探索和集成更多原子变化类别，以处理更复杂、多样化的地图修改场景。
*   **提升模型泛化能力：** 进一步研究如何提升模型在不同地理区域和环境条件下的泛化能力，减少sim2real差距。
*   **实时性能优化：** 优化模型架构和算法，以实现更快的推理速度，满足自动驾驶系统对实时地图更新的需求。
*   **结合更多传感器模态：** 探索如何更有效地整合更多传感器模态（如激光雷达、毫米波雷达等）来进一步提高地图更新的鲁棒性和准确性。

---

**Key Findings:**

- To address this, we introduce ArgoTweak, the first
dataset to complete the triplet with realistic map priors.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08764v1)
- [arXiv](https://arxiv.org/abs/2509.08764v1)

---

<a id='2509.08738v1'></a>
## [CrowdQuery: Density-Guided Query Module for Enhanced 2D and 3D Detection in Crowded Scenes](https://arxiv.org/abs/2509.08738v1)

**Authors:** Marius Dähling, Sebastian Krebs, J. Marius Zöllner

**Published:** 2025-09-10

**Categories:** cs.CV

**Abstract:**

This paper introduces a novel method for end-to-end crowd detection that
leverages object density information to enhance existing transformer-based
detectors. We present CrowdQuery (CQ), whose core component is our CQ module
that predicts and subsequently embeds an object density map. The embedded
density information is then systematically integrated into the decoder.
Existing density map definitions typically depend on head positions or
object-based spatial statistics. Our method extends these definitions to
include individual bounding box dimensions. By incorporating density
information into object queries, our method utilizes density-guided queries to
improve detection in crowded scenes. CQ is universally applicable to both 2D
and 3D detection without requiring additional data. Consequently, we are the
first to design a method that effectively bridges 2D and 3D detection in
crowded environments. We demonstrate the integration of CQ into both a general
2D and 3D transformer-based object detector, introducing the architectures CQ2D
and CQ3D. CQ is not limited to the specific transformer models we selected.
Experiments on the STCrowd dataset for both 2D and 3D domains show significant
performance improvements compared to the base models, outperforming most
state-of-the-art methods. When integrated into a state-of-the-art crowd
detector, CQ can further improve performance on the challenging CrowdHuman
dataset, demonstrating its generalizability. The code is released at
https://github.com/mdaehl/CrowdQuery.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Marius Dähling, Sebastian Krebs, J. Marius Zöllner撰写的论文“CrowdQuery: Density-Guided Query Module for Enhanced 2D and 3D Detection in Crowded Scenes”的全面摘要。

---

**论文摘要：CrowdQuery: Density-Guided Query Module for Enhanced 2D and 3D Detection in Crowded Scenes**

**1. 主要问题或研究问题：**
该论文旨在解决在拥挤场景中进行2D和3D目标检测（特别是行人检测）的挑战。现有的基于Transformer的检测器在处理密集人群时常遇到困难，因为它们最初均匀分布的查询需要适应图像中的局部密集区域。此外，尽管2D人群检测研究较多，但3D人群检测领域仍缺乏针对密集场景的适应性方法，导致2D和3D检测之间存在鸿沟。论文的核心问题是如何有效地利用目标密度信息来指导Transformer模型，从而提高在拥挤环境中的检测性能，并弥合2D和3D人群检测之间的差距。

**2. 关键创新或方法论贡献：**
该论文提出了一个名为**CrowdQuery (CQ)** 的新颖模块，其核心创新点包括：

*   **密度图定义扩展：** 传统的密度图通常基于头部位置或对象空间统计。CQ方法将这些定义扩展，纳入了单个边界框的尺寸（宽度和高度），从而生成更精细、更准确的密度表示，能够反映非方形物体形状和不对称性。
*   **密度引导的查询模块（CQ模块）：** CQ模块预测并嵌入一个对象密度图。该模块包含两个分支：一个分支通过多头自注意力增强密度特征并确保全局信息交换；另一个分支则学习基于预测密度图的合适密度图嵌入。
*   **密度信息系统集成：** 嵌入的密度信息通过交叉注意力机制系统地集成到Transformer解码器中，以密度引导查询的方式指导检测过程。这种集成方式在图像编码器信息之前处理密度信息，以预先准备查询。
*   **通用性和跨领域适用性：** CQ模块被设计为普遍适用于2D和3D检测任务，无需额外数据。论文首次提出了有效连接2D和3D拥挤环境中检测的方法，并展示了其在通用2D（CQ2D）和3D（CQ3D）Transformer检测器中的集成。
*   **端到端训练：** CQ模块的集成不改变整体训练流程，仅影响损失函数，通过像素级L1损失将目标密度图与预测密度图进行比较，保持了Transformer模型的端到端学习特性。

**3. 主要结果及其意义：**
论文通过在STCrowd数据集上进行2D和3D检测实验，以及在CrowdHuman数据集上进行泛化性测试，取得了显著成果：

*   **2D检测性能提升：** 在STCrowd数据集上，CQ2D在AP指标上超越了大多数现有SOTA方法，包括GigaHumanDet，AP达到91.4%，比基线Deformable DETR提高了1.8个点，MR-2降低了6.3个点。
*   **3D检测性能提升：** 在STCrowd数据集上，CQ3D在所有报告的指标上均优于基线MonoDETR，mAP提高了4.2个点，AR0、AR1、AR2分别提高了3.2、3.4和3.6个点。即使使用更简单的ResNet-50骨干网络，CQ3D的mAP也比其他测试方法至少高出3.2个点。
*   **与SOTA人群检测器集成：** 当CQ模块集成到SOTA人群检测器DDQ [24]中时（称为CQ2D++），在CrowdHuman数据集上，AP进一步提高到94.0%，MR-2降低到39.0%，超越了所有其他方法，包括UniHCP和Iter-Def-DETR。这证明了CQ的泛化能力和与现有模型的良好兼容性。
*   **消融研究：** 密度引导的Transformer、密度图嵌入的映射方式、嵌入bin的数量以及密度图的缩放因子等组件的有效性得到了验证。结果表明，均匀嵌入和较高分辨率的bin计数效果最佳。
*   **计算效率：** CQ3D引入了16.0%的运行时增加，其中三分之二归因于CQ模块中的自注意力。参数数量仅增加了12.9%，表明该方法具有较高的计算效率。

**4. 论文中提及的局限性：**
*   **3D数据集的缺乏：** 论文指出，当前缺乏适用于拥挤场景的单目3D数据集，这限制了3D人群检测的进一步研究和评估。
*   **MR-2性能稳定性：** 在2D检测中，查询式方法（包括CQ2D）在训练过程中表现出较不稳定的MR-2性能，并且最终性能通常不如基于边界框的方法。
*   **特定Transformer模型的选择：** 尽管CQ具有通用性，但论文主要在Deformable DETR和MonoDETR上进行了验证，其在其他Transformer模型上的具体表现仍需进一步探索。

**5. 潜在的未来研究方向：**
*   **扩展到更多网络和数据集：** 未来工作将探索CQ方法在更多不同Transformer网络和数据集上的泛化能力。
*   **集成到LiDAR-based检测器：** 将CQ模块扩展到基于LiDAR的检测器，以进一步提升其在自动驾驶等领域的应用潜力。
*   **统一2D和3D人群检测：** 论文希望启发研究人员将2D和3D人群检测视为一个更统一的课题，以开发更全面的解决方案。

---

总而言之，这篇论文通过引入CrowdQuery模块，巧妙地将精细化的对象密度信息集成到Transformer检测器中，显著提升了在拥挤场景中2D和3D目标检测的性能。其方法论的通用性和在多个数据集上的优异表现，使其成为人群检测领域的一项重要贡献。

**Key Findings:**

- This paper introduces a novel method for end-to-end crowd detection that
leverages object density information to enhance existing transformer-based
detectors.
- We present CrowdQuery (CQ), whose core component is our CQ module
that predicts and subsequently embeds an object density map.
- Our method extends these definitions to
include individual bounding box dimensions.
- By incorporating density
information into object queries, our method utilizes density-guided queries to
improve detection in crowded scenes.
- We demonstrate the integration of CQ into both a general
2D and 3D transformer-based object detector, introducing the architectures CQ2D
and CQ3D.
- Experiments on the STCrowd dataset for both 2D and 3D domains show significant
performance improvements compared to the base models, outperforming most
state-of-the-art methods.
- When integrated into a state-of-the-art crowd
detector, CQ can further improve performance on the challenging CrowdHuman
dataset, demonstrating its generalizability.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08738v1)
- [arXiv](https://arxiv.org/abs/2509.08738v1)

---

<a id='2509.08715v1'></a>
## [BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion](https://arxiv.org/abs/2509.08715v1)

**Authors:** Sike Xiang, Shuang Chen, Amir Atapour-Abarghouei

**Published:** 2025-09-10

**Categories:** cs.CV

**Abstract:**

As multimodal large language models (MLLMs) advance, their large-scale
architectures pose challenges for deployment in resource-constrained
environments. In the age of large models, where energy efficiency,
computational scalability and environmental sustainability are paramount, the
development of lightweight and high-performance models is critical for
real-world applications. As such, we propose a lightweight MLLM framework for
end-to-end visual question answering. Our proposed approach centres on
BreezeCLIP, a compact yet powerful vision-language encoder optimised for
efficient multimodal understanding. With only 1.2 billion parameters overall,
our model significantly reduces computational cost while achieving performance
comparable to standard-size MLLMs. Experiments conducted on multiple datasets
further validate its effectiveness in balancing accuracy and efficiency. The
modular and extensible design enables generalisation to broader multimodal
tasks. The proposed lightweight vision-language framework is denoted as BcQLM
(BreezeCLIP-enhanced Q-Gated Multimodal Language Model). It offers a promising
path toward deployable MLLMs under practical hardware constraints. The source
code is available at https://github.com/thico0224/BcQLM.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Sike Xiang, Shuang Chen, Amir Atapour-Abarghouei撰写的论文“BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion”的全面摘要。

---

### BcQLM: 高效视觉-语言理解与蒸馏Q门控跨模态融合

**1. 主要问题或研究问题：**
随着多模态大型语言模型（MLLMs）的快速发展，其庞大的架构在资源受限环境中部署面临巨大挑战。在对能源效率、计算可扩展性和环境可持续性要求日益提高的背景下，开发轻量级、高性能的模型对于实际应用至关重要。本文旨在解决如何在保持高性能的同时，显著降低MLLMs的计算成本和参数量，从而实现高效的视觉问答（VQA）和其他多模态任务。

**2. 关键创新或方法论贡献：**
该论文提出了一个名为**BcQLM (BreezeCLIP-enhanced Q-Gated Multimodal Language Model)** 的轻量级MLLM框架，其核心创新包括：

*   **BreezeCLIP编码器：** 这是一个紧凑但强大的视觉-语言编码器，旨在显著降低计算成本，同时实现高效的多模态表示。它通过将原始CLIP中的BERT和ViT骨干替换为受倒置瓶颈设计启发的紧凑型Transformer模块，从而实现参数量的大幅减少。BreezeCLIP采用双重训练策略：对比学习（确保视觉和文本特征在共享嵌入空间中良好对齐）和知识蒸馏（从CLIP教师模型转移高级语义对齐）。
*   **Q门控跨模态融合模块（Q-GCAM）：** 该模块动态调整视觉和文本特征的贡献，以适应输入问题，从而实现细粒度、问题感知的模态间交互。这使得模型能够根据问题的语义进行更精确的推理。
*   **轻量级LLaMA-3.2-1B解码器：** 采用一个参数量较小的LLaMA模型作为解码器，用于生成答案，进一步降低了计算开销。

**3. 主要结果及其意义：**
*   **参数效率：** BcQLM模型总参数量仅为1.2亿，与现有标准尺寸的MLLMs相比，参数量显著减少（仅为最先进方法的10%）。
*   **性能表现：** 在GQA、VQAv2和VizWiz等多个VQA基准数据集上，BcQLM取得了与标准尺寸MLLMs相当的性能。例如，在224x224分辨率下，BcQLM在GQA上达到60.8%，VQAv2上达到71.0%，VizWiz上达到49.5%，甚至在某些情况下超越了使用更高分辨率的现有方法。当输入分辨率提高到336x336时，性能进一步提升，在GQA上达到62.4%，VQAv2上达到78.7%，VizWiz上达到56.1%。
*   **效率分析：** 在NVIDIA RTX 4070 Ti上进行的推理效率测试表明，BcQLM的运行速度是Qwen2.5-VL-3B和Gemma3-4B的两倍，同时内存使用量降低约30%，FLOPs数量相当或更少，突显了其在边缘部署场景中的卓越效率。
*   **特征判别能力：** 实验表明，BreezeCLIP在多模态嵌入空间中实现了更强的对齐，能够清晰地分离正负样本对，显著提高了特征判别能力。

这些结果证明了BcQLM在平衡准确性和效率方面的有效性，为在实际硬件约束下部署MLLMs提供了一条有前景的途径。

**4. 论文中提及的局限性：**
*   **数据集依赖：** 模型依赖于公开可用的视觉-语言数据集，这些数据集可能无法完全捕捉真实世界多模态场景的复杂性和分布。数据集多样性和质量的限制可能影响模型的泛化能力，尤其是在需要细粒度或专业推理的领域。
*   **解码器限制：** 系统中使用的解码器（LLaMA-3.2-1B）仅在文本上进行预训练，并在融合训练期间保持冻结。尽管这种设计提高了效率和可控性，但可能限制了模型在生成过程中自适应整合视觉语义的能力。
*   **任务范围：** 目前的方法主要在VQA基准上进行了验证。

**5. 潜在的未来研究方向：**
*   **扩展到更动态的模态：** 未来的工作应探索将BcQLM扩展到更动态的模态，包括视频、音频和实时交互式设置。
*   **增强泛化能力：** 解决数据集多样性和质量的限制，以提高模型在更广泛、更复杂的真实世界场景中的泛化能力。
*   **自适应视觉语义整合：** 探索在生成过程中更有效地自适应整合视觉语义的方法，可能通过对解码器进行更灵活的微调或引入新的融合机制。

---

这份摘要突出了BcQLM在解决MLLMs部署挑战方面的创新性，通过轻量级编码器和智能融合模块，在保持高性能的同时显著提升了效率，为资源受限环境下的多模态AI应用开辟了新途径。

**Key Findings:**

- As such, we propose a lightweight MLLM framework for
end-to-end visual question answering.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08715v1)
- [arXiv](https://arxiv.org/abs/2509.08715v1)

---

<a id='2509.08699v1'></a>
## [TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals](https://arxiv.org/abs/2509.08699v1)

**Authors:** Stefan Podgorski, Sourav Garg, Mehdi Hosseinzadeh, Lachlan Mares, Feras Dayoub, Ian Reid

**Published:** 2025-09-10

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG, cs.SY, eess.SY

**Abstract:**

Visual navigation in robotics traditionally relies on globally-consistent 3D
maps or learned controllers, which can be computationally expensive and
difficult to generalize across diverse environments. In this work, we present a
novel RGB-only, object-level topometric navigation pipeline that enables
zero-shot, long-horizon robot navigation without requiring 3D maps or
pre-trained controllers. Our approach integrates global topological path
planning with local metric trajectory control, allowing the robot to navigate
towards object-level sub-goals while avoiding obstacles. We address key
limitations of previous methods by continuously predicting local trajectory
using monocular depth and traversability estimation, and incorporating an
auto-switching mechanism that falls back to a baseline controller when
necessary. The system operates using foundational models, ensuring open-set
applicability without the need for domain-specific fine-tuning. We demonstrate
the effectiveness of our method in both simulated environments and real-world
tests, highlighting its robustness and deployability. Our approach outperforms
existing state-of-the-art methods, offering a more adaptable and effective
solution for visual navigation in open-set environments. The source code is
made publicly available: https://github.com/podgorki/TANGO.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Stefan Podgorski等人撰写的论文“TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals”的全面摘要。

---

### 论文摘要：TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals

**1. 主要问题或研究问题：**
传统的机器人视觉导航方法通常依赖于全局一致的3D地图或预训练控制器，这些方法计算成本高昂，且难以泛化到多样化的环境。这限制了机器人在开放集（open-set）环境中进行零样本（zero-shot）、长距离导航的能力。论文旨在解决如何在不依赖3D地图或预训练控制器的情况下，实现机器人对物体级拓扑目标的鲁棒、可泛化导航，同时有效避开障碍物。

**2. 关键创新或方法论贡献：**
TANGO提出了一种新颖的、仅基于RGB图像的、物体级拓扑度量导航（topometric navigation）管道，其核心创新点包括：

*   **拓扑全局路径规划与局部度量轨迹控制的独特融合：** 论文将全局拓扑路径规划与局部度量轨迹控制相结合，使机器人能够向物体级子目标导航，同时避开障碍物。这通过计算BEV（Bird's Eye View）可通行性地图实现，该地图结合了单目深度估计和可通行性语义估计。
*   **连续预测局部轨迹：** 利用单目深度和可通行性估计，系统能够连续预测局部轨迹，克服了以往方法在处理动态环境和障碍物时的局限性。
*   **自动切换控制机制：** 当度量可通行性预测不可靠或不可用时（例如机器人离墙太近或被障碍物阻挡），系统会自动切换回基于拓扑的基线控制器（RoboHop），确保在挑战性场景下仍能有效导航。
*   **基于基础模型的操作：** 系统利用基础模型（如SAM、CLIP、Depth Anything）进行感知任务，如图像分割、可通行性估计和深度估计，从而实现开放集适用性，无需特定领域的微调。

**3. 主要结果及其意义：**
论文在模拟环境（Habitat-Matterport 3D Dataset）和真实世界测试中验证了TANGO方法的有效性。

*   **性能超越现有SOTA方法：** TANGO在不同轨迹长度（“easy”、“hard”、“full”）下，在导航成功率方面显著优于现有的学习型控制器（PixNav）和不具备可通行性意识的零样本控制器（RoboHop）。
*   **开放集泛化能力：** 实验证明了TANGO在“已见但未访问过”目标（seen-but-unvisited goals）导航方面的能力，这强调了物体级拓扑地图的重要性，超越了简单的“教导-重复”（teach-and-repeat）范式。
*   **模块化和鲁棒性：** 系统的模块化设计和对基础模型的依赖，使其在面对地图显著变化（如障碍物）时仍能有效避障，展现了其鲁棒性和可部署性。

**4. 论文中提及的局限性：**
论文也坦诚地指出了当前管道的一些局限性，这些局限性可能导致导航失败：

*   **感知错误：** 当前视图段与参考段地图之间的不正确匹配可能导致不准确的子目标。
*   **规划不足：** 地图图中纯拓扑的边缺乏几何信息，无法在当前图像中区分不同子目标的相关性。
*   **可通行性估计误差：** 基于文本和分割的可通行性估计虽然方便，但容易出错，这导致了需要回退控制器的机制。

**5. 潜在的未来研究方向：**
尽管论文没有明确列出未来研究方向，但从其局限性中可以推断出以下潜在方向：

*   **改进感知模型：** 进一步提升分割和匹配方法的准确性，以减少不正确的子目标分配。
*   **增强拓扑图的几何感知：** 探索将几何信息更紧密地集成到拓扑图结构中，以提高规划的精确性。
*   **更鲁棒的可通行性估计：** 开发更精确、更少依赖文本提示的可通行性估计方法，减少对回退机制的依赖。
*   **动态环境适应性：** 进一步研究如何在高度动态的环境中，即使地图发生显著变化，也能保持导航的鲁棒性。
*   **多模态融合：** 探索更复杂的多模态信息融合，以提高机器人对环境的理解和导航能力。

---

总而言之，TANGO为开放集环境中的视觉导航提供了一个新颖且高效的解决方案，通过将拓扑全局规划与局部度量控制相结合，并利用基础模型实现零样本泛化，显著提升了机器人在复杂场景下的导航能力。

**Key Findings:**

- In this work, we present a
novel RGB-only, object-level topometric navigation pipeline that enables
zero-shot, long-horizon robot navigation without requiring 3D maps or
pre-trained controllers.
- Our approach integrates global topological path
planning with local metric trajectory control, allowing the robot to navigate
towards object-level sub-goals while avoiding obstacles.
- We demonstrate
the effectiveness of our method in both simulated environments and real-world
tests, highlighting its robustness and deployability.
- Our approach outperforms
existing state-of-the-art methods, offering a more adaptable and effective
solution for visual navigation in open-set environments.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08699v1)
- [arXiv](https://arxiv.org/abs/2509.08699v1)

---

<a id='2509.08519v1'></a>
## [HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning](https://arxiv.org/abs/2509.08519v1)

**Authors:** Liyang Chen, Tianxiang Ma, Jiawei Liu, Bingchuan Li, Zhuowei Chen, Lijie Liu, Xu He, Gen Li, Qian He, Zhiyong Wu

**Published:** 2025-09-10

**Categories:** cs.CV, cs.MM

**Abstract:**

Human-Centric Video Generation (HCVG) methods seek to synthesize human videos
from multimodal inputs, including text, image, and audio. Existing methods
struggle to effectively coordinate these heterogeneous modalities due to two
challenges: the scarcity of training data with paired triplet conditions and
the difficulty of collaborating the sub-tasks of subject preservation and
audio-visual sync with multimodal inputs. In this work, we present HuMo, a
unified HCVG framework for collaborative multimodal control. For the first
challenge, we construct a high-quality dataset with diverse and paired text,
reference images, and audio. For the second challenge, we propose a two-stage
progressive multimodal training paradigm with task-specific strategies. For the
subject preservation task, to maintain the prompt following and visual
generation abilities of the foundation model, we adopt the minimal-invasive
image injection strategy. For the audio-visual sync task, besides the commonly
adopted audio cross-attention layer, we propose a focus-by-predicting strategy
that implicitly guides the model to associate audio with facial regions. For
joint learning of controllabilities across multimodal inputs, building on
previously acquired capabilities, we progressively incorporate the audio-visual
sync task. During inference, for flexible and fine-grained multimodal control,
we design a time-adaptive Classifier-Free Guidance strategy that dynamically
adjusts guidance weights across denoising steps. Extensive experimental results
demonstrate that HuMo surpasses specialized state-of-the-art methods in
sub-tasks, establishing a unified framework for collaborative
multimodal-conditioned HCVG. Project Page:
https://phantom-video.github.io/HuMo.

**Analysis:**

好的，这是一篇关于Liyang Chen等人撰写的论文“HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning”的摘要，内容基于您提供的PDF和摘要：

**论文摘要：HuMo：通过协同多模态条件实现以人为中心的视频生成**

这篇论文《HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning》由Liyang Chen及其团队撰写，旨在解决以人为中心的视频生成（HCVG）领域中的核心挑战。HCVG的目标是从文本、图像和音频等多模态输入中合成人类视频，但现有方法在有效协调这些异构模态方面面临两大难题：一是缺乏带有配对三元组条件（文本、参考图像、音频）的高质量训练数据；二是难以在多模态输入下协同完成主体保留和音视频同步等子任务。

**1. 主要问题或研究问题：**
该研究旨在解决现有HCVG方法在处理多模态输入（文本、图像、音频）时，难以有效协调不同模态以实现高质量、可控的视频生成，特别是如何克服数据稀缺性和多模态子任务（如主体保留和音视频同步）之间的协同困难。

**2. 关键创新或方法论贡献：**
为了应对上述挑战，HuMo提出了一个统一的HCVG框架，其关键创新和方法论贡献包括：
*   **高质量多模态数据集构建：** 针对数据稀缺问题，HuMo构建了一个包含多样化、配对文本、参考图像和音频的高质量数据集。这通过一个两阶段的数据处理流程实现，首先从大规模文本-视频样本中检索具有相同语义但不同视觉属性的参考图像，然后进一步过滤带有同步音频轨道的视频样本，并进行语音增强和唇部对齐估计。
*   **渐进式多模态训练范式：** 针对多模态协同困难，HuMo提出了一种两阶段渐进式多模态训练范式，并采用任务特定策略：
    *   **主体保留任务：** 采用“最小侵入式图像注入策略”，将参考图像的VAE潜在表示与噪声视频潜在表示沿时间维度拼接，并限制参数更新在DiT的自注意力层，以在不损害基础模型文本遵循和视觉生成能力的前提下，实现主体一致性。
    *   **音视频同步任务：** 除了常用的音频交叉注意力层外，HuMo引入了“聚焦预测策略”（focus-by-predicting strategy）。通过引入一个面部区域预测器Fmask，并使用二进制交叉熵损失进行监督，隐式引导模型将音频与面部区域关联起来，从而增强音视频同步。为了确保主体保留能力不被削弱，音视频同步任务是渐进式地整合到训练中的。
*   **时间自适应分类器自由引导（CFG）策略：** 在推理阶段，HuMo设计了一种时间自适应CFG策略，动态调整去噪步骤中的引导权重。早期步骤侧重于文本/图像主导的语义结构和空间布局，后期步骤则强调音频和图像控制，以实现灵活、细粒度和协同的多模态控制。

**3. 主要结果及其意义：**
广泛的实验结果表明，HuMo在主体保留和音视频同步等子任务上超越了专门的现有（SOTA）方法。这证明了HuMo作为一个统一框架，能够实现协同多模态条件下的HCVG，并能生成高质量、主体一致且音视频同步的视频。论文还通过在1.7B和17B参数模型上的验证，展示了其有效性和可扩展性。定性结果（如图5和图6所示）进一步支持了HuMo在文本遵循、主体保留和音视频同步方面的卓越性能。

**4. 论文中提及的局限性：**
论文中没有明确列出当前方法的局限性，但从其“伦理考量”部分可以推断出潜在的社会和技术挑战。例如，生成逼真人类视频的能力可能被滥用，如制作深度伪造或未经同意的内容。此外，对生成内容的细粒度控制也要求负责任的使用指南，以防止操纵或错误信息。这些虽然不是技术上的局限，但代表了该技术在实际应用中需要面对的重要挑战。

**5. 潜在的未来研究方向：**
论文中没有明确提出未来的研究方向，但从其贡献和伦理考量可以推断：
*   **增强鲁棒性和泛化性：** 进一步提升模型在更复杂、多样化场景和未见数据上的鲁棒性和泛化能力。
*   **更精细的控制粒度：** 探索更细粒度的多模态控制，例如对特定面部表情、身体姿态或服装细节的精确控制。
*   **伦理和安全机制：** 针对潜在的滥用风险，开发和集成更强大的伦理和安全机制，包括水印、检测工具和负责任的使用策略，以确保技术被用于积极目的。
*   **扩展到更广泛的应用：** 将HCVG技术应用于更广泛的领域，如虚拟现实、游戏、教育和个性化内容创作。

总而言之，HuMo为以人为中心的视频生成领域提供了一个全面且创新的解决方案，通过其数据处理、训练范式和推理策略，有效地解决了多模态协调的难题，为该领域未来的发展奠定了坚实基础。

**Key Findings:**

- In this work, we present HuMo, a
unified HCVG framework for collaborative multimodal control.
- For the second challenge, we propose a two-stage
progressive multimodal training paradigm with task-specific strategies.
- For the audio-visual sync task, besides the commonly
adopted audio cross-attention layer, we propose a focus-by-predicting strategy
that implicitly guides the model to associate audio with facial regions.
- Extensive experimental results
demonstrate that HuMo surpasses specialized state-of-the-art methods in
sub-tasks, establishing a unified framework for collaborative
multimodal-conditioned HCVG.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08519v1)
- [arXiv](https://arxiv.org/abs/2509.08519v1)

---

<a id='2509.08104v1'></a>
## [APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction](https://arxiv.org/abs/2509.08104v1)

**Authors:** Sasan Sharifipour, Constantino Álvarez Casado, Mohammad Sabokrou, Miguel Bordallo López

**Published:** 2025-09-09

**Categories:** cs.CV, cs.AI

**Abstract:**

Training deep learning models for point cloud prediction tasks such as shape
completion and generation depends critically on loss functions that measure
discrepancies between predicted and ground-truth point sets. Commonly used
functions such as Chamfer Distance (CD), HyperCD, and InfoCD rely on
nearest-neighbor assignments, which often induce many-to-one correspondences,
leading to point congestion in dense regions and poor coverage in sparse
regions. These losses also involve non-differentiable operations due to index
selection, which may affect gradient-based optimization. Earth Mover Distance
(EMD) enforces one-to-one correspondences and captures structural similarity
more effectively, but its cubic computational complexity limits its practical
use. We propose the Adaptive Probabilistic Matching Loss (APML), a fully
differentiable approximation of one-to-one matching that leverages Sinkhorn
iterations on a temperature-scaled similarity matrix derived from pairwise
distances. We analytically compute the temperature to guarantee a minimum
assignment probability, eliminating manual tuning. APML achieves near-quadratic
runtime, comparable to Chamfer-based losses, and avoids non-differentiable
operations. When integrated into state-of-the-art architectures (PoinTr, PCN,
FoldingNet) on ShapeNet benchmarks and on a spatiotemporal Transformer (CSI2PC)
that generates 3D human point clouds from WiFi CSI measurements, APM loss
yields faster convergence, superior spatial distribution, especially in
low-density regions, and improved or on-par quantitative performance without
additional hyperparameter search. The code is available at:
https://github.com/apm-loss/apml.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Sasan Sharifipour, Constantino Álvarez Casado, Mohammad Sabokrou, Miguel Bordallo López撰写的论文“APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction”的全面摘要。

---

**论文摘要：APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction**

**1. 主要问题或研究问题：**
该论文旨在解决3D点云预测任务（如形状补全和生成）中现有损失函数的局限性。当前常用的损失函数，如Chamfer Distance (CD)及其变体（HyperCD, InfoCD），依赖于最近邻分配，这会导致多对一的对应关系，造成点云在密集区域拥堵、稀疏区域覆盖不足，并且由于索引选择涉及不可微分操作，影响梯度优化。而Earth Mover Distance (EMD)虽然能有效捕捉结构相似性并强制一对一对应，但其立方级的计算复杂度使其在大规模深度学习中不切实际。因此，核心问题是开发一种既能提供EMD的几何监督优势，又能避免其高昂计算成本和不可微分问题的损失函数。

**2. 关键创新或方法贡献：**
论文提出了**自适应概率匹配损失 (Adaptive Probabilistic Matching Loss, APML)**，其主要创新点包括：
*   **近似一对一匹配与Sinkhorn迭代：** APML通过在基于成对距离的温度缩放相似性矩阵上应用Sinkhorn迭代，提供了一种完全可微分的一对一匹配近似。这使得模型能够建立软性、概率性的对应关系，从而更好地捕捉点云的结构相似性。
*   **数据驱动的自适应温度计算：** APML引入了一种新颖的、数据驱动的、分析推导的温度调度机制。该机制根据点云的局部几何上下文自动计算温度，以保证每个点至少有一个最小的分配概率，从而消除了手动调整正则化参数的需要，提高了训练的稳定性和泛化能力。
*   **近二次时间复杂度：** APML的计算复杂度接近二次（O(NM(d+L))），与Chamfer-based损失相当，远低于EMD的立方复杂度。同时，它避免了不可微分操作，确保了平滑的梯度流。
*   **双向一致性与Sinkhorn归一化：** APML通过双向（预测到真值和真值到预测）的软分配矩阵平均化来确保一致性，并进一步通过Sinkhorn-Knopp算法进行迭代归一化，以生成近似双随机的传输计划。

**3. 主要结果及其意义：**
*   **性能提升：** APML在ShapeNet基准测试（PCN, FoldingNet, PoinTr）和基于WiFi-CSI测量生成3D人体点云的时空Transformer (CSI2PC) 等最先进架构上进行集成和评估。结果显示，APML在EMD指标上显著优于现有损失函数（通常降低15-81%），同时保持或略微提升F1分数。这表明APML能生成几何上更忠实、结构更连贯的点云重建。
*   **更快的收敛速度：** APML在训练过程中表现出更快的收敛速度，在更少的epoch内达到更高的性能，从而降低了有效的计算预算。
*   **更好的空间分布和稀疏区域覆盖：** 相比Chamfer-based损失，APML在低密度区域能更好地保留结构，减少点云聚拢，并能跨不同输入模态进行泛化。
*   **无需额外超参数搜索：** APML的自适应温度机制消除了手动调整正则化参数的需要，使其成为一个易于使用的“即插即用”替代品。

**4. 论文中提及的局限性：**
*   **超参数Pmin：** 尽管APML消除了Sinkhorn正则化参数ε，但引入了一个新的超参数——软分配阈值Pmin。论文中Pmin被固定为0.8，并未进行调优，其敏感性分析是未来的工作。
*   **内存二次扩展：** APML所需的内存仍然随点云数量呈二次方增长。虽然经验性研究表明传输矩阵在Sinkhorn归一化前大部分是稀疏的（超过90%的条目接近零），但目前仍以密集形式存储。这限制了更大批量大小的使用。
*   **未充分利用稀疏性：** 当前实现未充分利用传输矩阵的稀疏性，也未使用FP16混合精度，这导致内存占用较高。
*   **评估范围：** 评估主要集中在两个合成补全数据集和一个真实生成数据集，尚未在ScanNet或KITTI等真实扫描数据集上进行补全评估，也未超越轮廓生成任务。

**5. 潜在的未来研究方向：**
*   **Pmin的替代方案：** 探索Pmin的可学习或基于调度的方法。
*   **内存优化：** 开发低秩或分片Sinkhorn变体以减少内存使用，并实现完全优化的CUDA内核以利用传输矩阵的稀疏性，从而将有效成本推向O(N log N)。
*   **扩展评估范围：** 将APML的评估扩展到噪声、真实世界扫描和非欧几里得领域（如表面或图），以进一步验证其鲁棒性。
*   **实际应用：** 将APML应用于机器人、AR和数字孪生等需要高感知结构保真度的实际部署场景，并可能促进边缘设备的轻量级、节能模型开发。

---

**Key Findings:**

- We propose the Adaptive Probabilistic Matching Loss (APML), a fully
differentiable approximation of one-to-one matching that leverages Sinkhorn
iterations on a temperature-scaled similarity matrix derived from pairwise
distances.
- When integrated into state-of-the-art architectures (PoinTr, PCN,
FoldingNet) on ShapeNet benchmarks and on a spatiotemporal Transformer (CSI2PC)
that generates 3D human point clouds from WiFi CSI measurements, APM loss
yields faster convergence, superior spatial distribution, especially in
low-density regions, and improved or on-par quantitative performance without
additional hyperparameter search.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.08104v1)
- [arXiv](https://arxiv.org/abs/2509.08104v1)

---

<a id='2509.07978v1'></a>
## [One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation](https://arxiv.org/abs/2509.07978v1)

**Authors:** Zheng Geng, Nan Wang, Shaocong Xu, Chongjie Ye, Bohan Li, Zhaoxi Chen, Sida Peng, Hao Zhao

**Published:** 2025-09-09

**Categories:** cs.CV

**Abstract:**

Estimating the 6D pose of arbitrary unseen objects from a single reference
image is critical for robotics operating in the long-tail of real-world
instances. However, this setting is notoriously challenging: 3D models are
rarely available, single-view reconstructions lack metric scale, and domain
gaps between generated models and real-world images undermine robustness. We
propose OnePoseViaGen, a pipeline that tackles these challenges through two key
components. First, a coarse-to-fine alignment module jointly refines scale and
pose by combining multi-view feature matching with render-and-compare
refinement. Second, a text-guided generative domain randomization strategy
diversifies textures, enabling effective fine-tuning of pose estimators with
synthetic data. Together, these steps allow high-fidelity single-view 3D
generation to support reliable one-shot 6D pose estimation. On challenging
benchmarks (YCBInEOAT, Toyota-Light, LM-O), OnePoseViaGen achieves
state-of-the-art performance far surpassing prior approaches. We further
demonstrate robust dexterous grasping with a real robot hand, validating the
practicality of our method in real-world manipulation. Project page:
https://gzwsama.github.io/OnePoseviaGen.github.io/

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zheng Geng等人撰写的论文“One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation”的全面摘要。

---

**论文摘要：One View, Many Worlds: Single-Image to 3D Object Meets Generative Domain Randomization for One-Shot 6D Pose Estimation**

**1. 主要问题或研究问题**
该论文旨在解决机器人技术中一个核心且极具挑战性的问题：如何仅凭一张参考图像，对任意未见过的物体进行一次性（one-shot）6D姿态估计（即确定其在三维空间中的精确位置和方向）。这一任务在现实世界中尤其困难，因为3D模型通常不可用，单视图重建缺乏度量尺度，且生成模型与真实图像之间存在领域鸿沟，这些因素都严重影响了姿态估计的鲁棒性。

**2. 关键创新或方法论贡献**
作者提出了一个名为 **OnePoseViaGen** 的端到端管道，通过两个关键组件克服了上述挑战：

*   **生成式一次性6D姿态管道：** OnePoseViaGen是首个将单视图3D生成整合到训练和推理中，用于一次性6D姿态和尺度估计的管道。它证明了生成式建模可以直接促进姿态估计。
*   **粗到精的度量对齐模块：** 该模块通过结合多视图特征匹配和渲染-比较（render-and-compare）细化，共同优化了物体的尺度和姿态。首先，从单张RGB-D锚点图像生成一个缺乏真实世界尺度和姿态的纹理3D模型。然后，通过粗略对齐（使用SuperGlue进行2D-3D特征匹配和PnP求解）获得初始姿态和尺度估计。接着，通过迭代的渲染-比较细化（利用FoundationPose网络）进一步精细化姿态和尺度，从而实现从单张图像中准确恢复度量尺度。
*   **文本引导的生成式领域随机化策略：** 为了弥合生成模型与真实世界图像之间的领域鸿沟，该方法引入了一种文本驱动的增强策略。它利用文本提示（例如，通过VLM生成详细描述）引导3D生成模型（如Trellis）创建结构一致但纹理多样的物体变体。这些变体在随机化的光照、背景和遮挡条件下进行渲染，生成大规模合成数据集，用于姿态估计器的有效微调，从而显著提高鲁棒性。

**3. 主要结果及其意义**
OnePoseViaGen在多个具有挑战性的基准测试（YCBInEOAT、Toyota-Light、LM-O）上取得了显著的SOTA（State-of-the-Art）性能，远超现有方法。例如，在YCBInEOAT数据集上，OnePoseViaGen在ADD指标上取得了81.3的平均分数，而基线方法（如Oryon、LoFTR、Gedi）在该一次性设置下表现不佳。在LM-O和TOYL等数据集上，该方法在AR、MSSD、MSPD和VSD等所有评估标准上均显示出持续改进。

此外，论文通过在真实机器人（配备XHAND1灵巧手）上进行鲁棒的灵巧抓取任务，验证了该方法在实际世界操作中的实用性。这表明OnePoseViaGen不仅在理论上有效，而且在实际应用中也具有很高的可靠性。

**4. 论文中提及的局限性**
尽管OnePoseViaGen取得了令人鼓舞的成果，但它在处理**可变形或关节式物体**时仍面临挑战。在这种情况下，物体形状的变化可能导致6D姿态估计不准确。

**5. 潜在的未来研究方向**
未来的工作将侧重于将**测试时训练（test-time training）**整合到推理管道中，以实现对可变形物体几何形状的持续细化和准确姿态估计。这将充分利用生成模型在6D姿态估计任务中的灵活性和泛化能力。

---

总而言之，OnePoseViaGen通过其创新的粗到精对齐模块和文本引导的生成式领域随机化策略，为一次性6D姿态估计提供了一个强大且实用的解决方案，显著推动了机器人操作和计算机视觉领域的发展。

**Key Findings:**

- On challenging
benchmarks (YCBInEOAT, Toyota-Light, LM-O), OnePoseViaGen achieves
state-of-the-art performance far surpassing prior approaches.
- We further
demonstrate robust dexterous grasping with a real robot hand, validating the
practicality of our method in real-world manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.07978v1)
- [arXiv](https://arxiv.org/abs/2509.07978v1)

---

