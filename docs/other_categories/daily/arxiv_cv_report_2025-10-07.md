time: 20251007

# Arxiv Computer Vision Papers - 2025-10-07

## Executive Summary

好的，这是一份针对2025年10月6日Arxiv计算机视觉领域论文的每日报告执行摘要，旨在帮助忙碌的研究人员快速掌握关键信息。

---

**每日Arxiv计算机视觉论文报告执行摘要 (2025年10月06日)**

**1. 主要主题与趋势概述：**

今天的论文集展现了计算机视觉领域几个活跃且相互关联的趋势：

*   **多模态大模型 (LMM) 的深入探索与应用：** 多篇论文聚焦于视频与文本等多模态数据的融合，特别是大型多模态模型在视频理解、生成和推理方面的应用，显示出LMM在复杂场景理解中的巨大潜力。
*   **3D 场景理解与表示：** 3D 数据处理、场景图预测、占用预测以及与几何相关的匹配仍然是重要研究方向，强调了从2D图像向更丰富3D世界理解的演进。
*   **生成模型与内容创作：** 视频生成、人体与相机运动生成、以及精细化的人脸表情生成等内容创作方向持续发展，生成模型在控制性、真实感和一致性方面不断提升。
*   **特定应用与数据集：** 针对水下环境分类、体育赛事分析等特定应用场景的数据集构建和方法开发，体现了CV技术在实际问题解决中的落地。
*   **基础技术优化：** 散列（Hashing）在近似最近邻搜索（ANN）中的应用，以及几何匹配等基础算法的改进，为上层应用提供了更高效或更鲁棒的支撑。

**2. 显著或创新性论文亮点：**

*   **"Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models" (Yunlong Tang et al.)：** 这篇论文深入探讨了视频推理中大型多模态模型的后训练策略，可能揭示了如何有效提升LMM在复杂视频理解任务上的性能，对于LMM的实际应用具有指导意义。
*   **"Pulp Motion: Framing-aware multimodal camera and human motion generation" (Robin Courant et al.)：** 创新性地结合了相机和人体运动生成，并考虑了“构图感知”，这对于电影制作、虚拟现实和动画等领域的内容创作具有突破性，能生成更自然、更具叙事感的动态场景。
*   **"Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction" (Chi Yan, Dan Xu)：** 结合了Transformer和高斯表示，并引入了各向异性采样，以实现开放词汇的占用预测。这在3D场景理解中是一个重要进步，尤其是在处理未知物体和复杂几何结构时。

**3. 新兴研究方向或技术：**

*   **LMM在视频领域的精细化应用：** 不再仅仅是简单的视频-文本对应，而是深入到视频推理、事件理解等更高级的认知任务。
*   **多模态生成中的协同控制：** 如“Pulp Motion”所示，同时控制相机和人体运动，实现更协调、更具表现力的生成。
*   **3D 场景理解的开放词汇能力：** 能够识别和预测未见过物体的占用信息，是迈向通用3D理解的关键一步。
*   **结合几何先验的深度学习：** “SegMASt3R”和“Object-Centric Representation Learning”都强调了将几何信息融入深度学习模型的重要性，以提高鲁棒性和解释性。

**4. 建议阅读全文的论文：**

对于不同兴趣的研究人员，建议阅读以下论文：

*   **对于关注多模态大模型和视频理解的研究人员：**
    *   **"Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models" (Yunlong Tang et al.)**：了解LMM在视频推理中的最新进展和训练策略。
    *   **"Bridging Text and Video Generation: A Survey" (Nilay Kumar et al.)**：全面了解文本到视频生成领域的现状和挑战。
*   **对于关注生成模型和内容创作的研究人员：**
    *   **"Pulp Motion: Framing-aware multimodal camera and human motion generation" (Robin Courant et al.)**：探索创新的多模态运动生成技术。
    *   **"ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion" (Foivos Paraperas Papantoniou, Stefanos Zafeiriou)**：了解高保真人脸表情生成。
*   **对于关注3D视觉和场景理解的研究人员：**
    *   **"Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction" (Chi Yan, Dan Xu)**：深入了解开放词汇3D占用预测。
    *   **"Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction" (KunHo Heo et al.)**：探索如何通过以对象为中心的表示改进3D场景图预测。
*   **对于关注特定应用或基础算法的研究人员：**
    *   **"BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping" (Hayat Rajani et al.)**：如果对水下机器人或环境监测感兴趣。
    *   **"Learning-Based Hashing for ANN Search: Foundations and Early Advances" (Sean Moran)**：如果对高效检索和基础算法优化感兴趣。

---

这份摘要希望能帮助您快速把握今日Arxiv计算机视觉领域的关键动态。

---

## Table of Contents

1. [Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models](#2510.05034v1)
2. [Bridging Text and Video Generation: A Survey](#2510.04999v1)
3. [Learning-Based Hashing for ANN Search: Foundations and Early Advances](#2510.04127v1)
4. [Pulp Motion: Framing-aware multimodal camera and human motion generation](#2510.05097v1)
5. [SegMASt3R: Geometry Grounded Segment Matching](#2510.05051v1)
6. [BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping](#2510.04876v1)
7. [Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction](#2510.04759v1)
8. [ExposureEngine: Oriented Logo Detection and Sponsor Visibility Analytics in Sports Broadcasts](#2510.04739v1)
9. [Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction](#2510.04714v1)
10. [ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion](#2510.04706v1)

---

## Papers

<a id='2510.05034v1'></a>
## [Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models](https://arxiv.org/abs/2510.05034v1)

**Authors:** Yunlong Tang, Jing Bi, Pinxin Liu, Zhenyu Pan, Zhangyun Tan, Qianxiang Shen, Jiani Liu, Hang Hua, Junjia Guo, Yunzhong Xiao, Chao Huang, Zhiyuan Wang, Susan Liang, Xinyi Liu, Yizhi Song, Yuhe Nie, Jia-Xing Zhong, Bozheng Li, Daiqing Qi, Ziyun Zeng, Ali Vosoughi, Luchuan Song, Zeliang Zhang, Daiki Shimada, Han Liu, Jiebo Luo, Chenliang Xu

**Published:** 2025-10-06

**Categories:** cs.CV

**Abstract:**

Video understanding represents the most challenging frontier in computer
vision, requiring models to reason about complex spatiotemporal relationships,
long-term dependencies, and multimodal evidence. The recent emergence of
Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders
with powerful decoder-based language models, has demonstrated remarkable
capabilities in video understanding tasks. However, the critical phase that
transforms these models from basic perception systems into sophisticated
reasoning engines, post-training, remains fragmented across the literature.
This survey provides the first comprehensive examination of post-training
methodologies for Video-LMMs, encompassing three fundamental pillars:
supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL)
from verifiable objectives, and test-time scaling (TTS) through enhanced
inference computation. We present a structured taxonomy that clarifies the
roles, interconnections, and video-specific adaptations of these techniques,
addressing unique challenges such as temporal localization, spatiotemporal
grounding, long video efficiency, and multimodal evidence integration. Through
systematic analysis of representative methods, we synthesize key design
principles, insights, and evaluation protocols while identifying critical open
challenges in reward design, scalability, and cost-performance optimization. We
further curate essential benchmarks, datasets, and metrics to facilitate
rigorous assessment of post-training effectiveness. This survey aims to provide
researchers and practitioners with a unified framework for advancing Video-LMM
capabilities. Additional resources and updates are maintained at:
https://github.com/yunlong10/Awesome-Video-LMM-Post-Training

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Yunlong Tang等人撰写的论文“Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models”的全面摘要。

**论文摘要：Video-LMM 后训练：深入探索大型多模态模型中的视频推理**

**1. 主要问题或研究问题：**
该论文旨在解决视频理解领域的核心挑战，即如何将新兴的视频大型多模态模型（Video-LMMs）从基本的感知系统提升为复杂的推理引擎。具体而言，研究问题集中在Video-LMMs的“后训练”阶段，这一阶段对于模型理解复杂的时空关系、长期依赖和多模态证据至关重要，但目前文献中对这一阶段的系统性研究尚不完善。

**2. 关键创新或方法论贡献：**
该论文的核心贡献在于首次对Video-LMMs的后训练方法进行了全面的系统性审查和分类。它将后训练方法分为三大支柱：
*   **带思维链的监督微调（SFT with Chain-of-Thought, CoT-SFT）：** 强调通过模仿推理模式来引导模型，并作为强化学习（RL）的冷启动阶段，提供结构化的推理格式和稳定的初始化。
*   **基于可验证目标的强化学习（RL from Verifiable Objectives）：** 探讨了如何利用可验证的输出（如答案正确性、时空定位精度）来优化模型，避免对人工偏好数据的依赖，并引入了GRPO（Group Relative Policy Optimization）等R1风格的RL算法。
*   **通过增强推理计算进行测试时缩放（Test-Time Scaling, TTS）：** 涵盖了推理阶段的计算分配，以提高可靠性，包括思维链提示、自洽性解码、基于置信度的迭代推理、自改进循环、蒙特卡洛树搜索（MCTS）以及工具增强推理。

论文还提出了一个结构化的分类法，阐明了这些技术在解决视频特有挑战（如时间定位、时空定位、长视频效率和多模态证据整合）中的作用、相互联系和视频特定适应性。

**3. 主要结果及其意义：**
该调查通过对代表性方法的系统分析，综合了关键设计原则、见解和评估协议。主要发现和意义包括：
*   **SFT作为基础：** CoT-SFT能够有效地将结构化推理行为注入Video-LMMs，并为后续的RL训练提供稳定的起点。
*   **RL的有效性：** 基于可验证奖励的RL（特别是GRPO）在提升Video-LMMs的推理能力方面表现出显著效果，尤其是在处理复杂时空任务和长视频理解方面。RL方法被证明是数据高效的，少量高质量数据即可媲美大规模监督微调的性能。
*   **TTS的可靠性提升：** TTS策略通过在推理时分配额外计算资源，显著提高了Video-LMMs的可靠性、推理深度和路径多样性，有助于减少幻觉并提高答案准确性。
*   **统一框架：** 论文提供了一个统一的框架，将SFT、RL和TTS视为模型优化的不可或缺的组成部分，这对于推动Video-LMMs能力的发展具有重要意义。

**4. 论文中提及的局限性：**
论文在讨论开放挑战时提及了当前方法的局限性：
*   **奖励设计挑战：** 现有奖励设计在处理复杂、可组合的奖励（如实体链接、排序、对象-动作绑定）时仍面临挑战，需要更精细的进程奖励模型（PRMs）来提供密集的信用分配。
*   **可扩展性问题：** 尽管RL数据高效，但在长视频上扩展RL仍然面临预算限制，需要更高效的帧选择和缓存机制。
*   **成本-性能优化：** 帧优化和压缩框架仍然成本高昂，需要未来的工作使其在数据和计算上更高效。
*   **数据稀缺和偏差：** 高质量、可验证的CoT数据构建成本高昂，且存在模板和单模型偏差。RL训练中也存在评估偏差和长度偏差，可能导致模型出现“奉承”或“字幕泄露”等问题。
*   **探索能力不足：** RL的探索能力仍需提升，超越教师模型所能提供的策略，需要多样性驱动的目标和自博弈机制。

**5. 潜在的未来研究方向：**
论文指出了以下几个未来研究方向：
*   **结构化接口和接地CoT：** 规范推理格式，将推理步骤与证据（时间戳、帧ID、区域）绑定，以提高忠实度并简化验证器设计。
*   **大规模验证器在环CoT合成：** 自动化草稿-细化-审计流程，从ASR/OCR/镜头元数据开始，通过轻量级检查器进行细化和过滤，以减少幻觉。
*   **三模态监督和字幕控制：** 将SFT扩展到对语音、事件和视觉证据的对齐，并始终报告带/不带转录的结果，以避免ASR快捷方式。
*   **幻觉感知指令微调：** 结合反事实和缺失案例，训练模型进行校准的弃权和验证行为，减少过度肯定。
*   **多语言、OCR和叙事结构：** 扩展SFT以处理多语言、退化文本和长跨度叙事推理。
*   **可组合、可验证的奖励：** 开发更精细的奖励机制，能够处理复杂的时空语义检查，并控制PRMs的成本和偏差。
*   **超越教师的探索：** 开发多样性驱动的目标和自博弈机制，使RL能够发现超越教师能力的策略。
*   **置信度感知、验证器引导的TTS：** 将停止规则与不确定性结合，并在需要时深化推理或密集化视图，实现随时可用的准确性。
*   **工具增强推理和蒸馏：** 将工具调用（检索、跟踪、ASR对齐）与推理相结合，并通过后验蒸馏将这些益处转移到基础模型中。
*   **带记忆的流式代理：** 开发能够决定何时观看、观看什么，并维护任务感知工作记忆的代理规划器，以处理长视频或流式视频。
*   **标准化报告和泄漏控制：** 报告查看预算、推理长度、路径计数、延迟/吞吐量和字幕使用情况，并进行“奉承”和“判断偏差”诊断。
*   **受限视图下的计算-准确性权衡：** 协同调整帧选择和压缩与推理质量，以在仅处理少量帧时保持系统性能。

总而言之，这篇论文为Video-LMMs的后训练提供了一个全面的路线图，不仅总结了现有技术，还指明了未来研究的关键方向，对于推动视频理解领域的发展具有重要的指导意义。

**Key Findings:**

- We present a structured taxonomy that clarifies the
roles, interconnections, and video-specific adaptations of these techniques,
addressing unique challenges such as temporal localization, spatiotemporal
grounding, long video efficiency, and multimodal evidence integration.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.05034v1)
- [arXiv](https://arxiv.org/abs/2510.05034v1)

---

<a id='2510.04999v1'></a>
## [Bridging Text and Video Generation: A Survey](https://arxiv.org/abs/2510.04999v1)

**Authors:** Nilay Kumar, Priyansh Bhandari, G. Maragatham

**Published:** 2025-10-06

**Categories:** cs.GR, cs.AI, cs.CV

**Abstract:**

Text-to-video (T2V) generation technology holds potential to transform
multiple domains such as education, marketing, entertainment, and assistive
technologies for individuals with visual or reading comprehension challenges,
by creating coherent visual content from natural language prompts. From its
inception, the field has advanced from adversarial models to diffusion-based
models, yielding higher-fidelity, temporally consistent outputs. Yet challenges
persist, such as alignment, long-range coherence, and computational efficiency.
Addressing this evolving landscape, we present a comprehensive survey of
text-to-video generative models, tracing their development from early GANs and
VAEs to hybrid Diffusion-Transformer (DiT) architectures, detailing how these
models work, what limitations they addressed in their predecessors, and why
shifts toward new architectural paradigms were necessary to overcome challenges
in quality, coherence, and control. We provide a systematic account of the
datasets, which the surveyed text-to-video models were trained and evaluated
on, and, to support reproducibility and assess the accessibility of training
such models, we detail their training configurations, including their hardware
specifications, GPU counts, batch sizes, learning rates, optimizers, epochs,
and other key hyperparameters. Further, we outline the evaluation metrics
commonly used for evaluating such models and present their performance across
standard benchmarks, while also discussing the limitations of these metrics and
the emerging shift toward more holistic, perception-aligned evaluation
strategies. Finally, drawing from our analysis, we outline the current open
challenges and propose a few promising future directions, laying out a
perspective for future researchers to explore and build upon in advancing T2V
research and applications.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Nilay Kumar, Priyansh Bhandari, G. Maragatham撰写的论文“Bridging Text and Video Generation: A Survey”的全面摘要。

---

### 论文摘要：文本到视频生成技术综述

**1. 主要问题或研究问题：**
该论文旨在全面综述文本到视频（Text-to-Video, T2V）生成领域，解决如何从自然语言提示中创建连贯的视觉内容这一核心问题。尽管T2V技术在教育、营销、娱乐和辅助技术等多个领域具有巨大潜力，但该领域仍面临诸多挑战，包括生成视频的语义对齐、长期连贯性、视觉质量以及计算效率。本综述旨在系统地追踪该领域的发展，识别关键模型、数据集、评估方法、现有局限性以及未来的研究方向。

**2. 关键创新或方法论贡献：**
该论文本身是一篇综述，其主要贡献在于对T2V领域进行了系统性、结构化的梳理和分析，而非提出新的模型。其方法论贡献体现在：
*   **发展路径梳理：** 详细追溯了T2V模型从早期的生成对抗网络（GANs）和变分自编码器（VAEs）到当前混合扩散-Transformer（DiT）架构的演变过程。解释了这些模型的工作原理、它们如何解决前代模型的局限性，以及为何需要转向新的架构范式以克服质量、连贯性和控制方面的挑战。
*   **核心技术回顾：** 详细介绍了T2V模型所依赖的基础性架构，包括U-Net、VAEs、GANs和去噪扩散概率模型（DDPMs），以及Transformer机制。这为理解T2V模型的内部机制提供了必要的背景知识。
*   **数据集和训练配置的系统化：** 提供了T2V模型训练和评估所用数据集的详细清单，并为了支持可复现性和评估模型训练的可访问性，详细列出了硬件规格、GPU数量、批次大小、学习率、优化器、训练周期等关键超参数。
*   **评估指标和基准的全面分析：** 概述了T2V模型常用的定量评估指标（如IS、FID、FVD、CLIP-SIM、KVD）及其在标准基准上的表现。同时，讨论了这些指标的局限性，并强调了向更全面、感知对齐的评估策略（如VBench）转变的必要性。

**3. 主要结果及其意义：**
*   **技术演进的清晰图景：** 综述展示了T2V技术从早期对抗模型到基于扩散模型的显著进步，这些进步带来了更高保真度、更具时间连贯性的输出。这表明扩散模型已成为当前T2V生成的主流范式。
*   **对现有模型优缺点的深入理解：** 论文详细分析了不同架构（GANs、VAEs、扩散模型）在解决视频生成固有问题（如时间一致性、视觉质量和文本-视频对齐）方面的优势和局限性。
*   **标准化评估的必要性：** 强调了现有定量指标在捕捉人类感知质量方面的不足，并介绍了VBench等新兴基准如何通过多维度评估和人类偏好标注来提供更细致、更全面的模型性能评估。这对于推动领域发展和确保模型与人类期望对齐至关重要。
*   **可复现性和可访问性的促进：** 通过详细列出训练配置，为研究人员提供了宝贵的实践指导，有助于评估实现和改进现有方法的可能性，并识别潜在的瓶颈和改进机会。

**4. 论文中提到的局限性：**
*   **计算效率：** 现有模型在处理长视频序列时计算成本高昂，且训练时间长。
*   **数据集限制：** 现有T2V数据集规模有限、质量不足，且存在版权限制，这阻碍了模型泛化能力和生成高质量视频的能力。
*   **输出质量和连贯性：** 尽管有所进步，但生成的视频仍可能存在时间或空间上的不连贯性，物体可能不自然地移动、出现或消失，物理交互不真实，以及难以捕捉复杂场景和多交互元素。
*   **多样性不足：** 生成内容的多样性有限，部分原因在于训练数据本身缺乏多样性。
*   **分辨率限制：** 维持长序列高分辨率输出仍然是一个挑战。
*   **评估指标的局限性：** 传统定量指标无法全面捕捉人类感知到的视频质量，如身份保持、运动流畅性和时间稳定性。

**5. 潜在的未来研究方向：**
*   **数据集丰富：**
    *   利用游戏引擎（如Unity、Unreal Engine）合成大规模、高分辨率、多样化且无版权限制的数据集。
    *   开发广义的提示框架，通过结构化提示（包括主体、属性、动作、背景、风格等）自动化视频生成，以提高数据集的质量和数量。
*   **模型架构和优化：**
    *   开发新的模型架构或算法，以更有效地处理视频数据，提高计算效率和可扩展性。
    *   增强模型的**时间建模能力**，以生成更长、更连贯的视频。
    *   整合先进的**注意力机制**，利用多模态数据，并修改损失函数，以更有效地关注连贯性和真实感。
    *   改进**物理约束建模**，使生成的视频更具物理合理性。
    *   通过更多样化的数据集和先进的**数据增强技术**，提高生成内容的多样性。
*   **应用和影响：**
    *   探索T2V技术在教育、营销、娱乐、辅助技术、文化遗产保护、法律取证、合成数据生成、游戏和虚拟现实等领域的更广泛应用。

---

这篇综述为T2V领域的研究人员提供了一个宝贵的资源，不仅总结了现有技术，还指明了未来的发展方向，有助于推动该领域在质量、效率和应用方面的进一步突破。

**Key Findings:**

- Addressing this evolving landscape, we present a comprehensive survey of
text-to-video generative models, tracing their development from early GANs and
VAEs to hybrid Diffusion-Transformer (DiT) architectures, detailing how these
models work, what limitations they addressed in their predecessors, and why
shifts toward new architectural paradigms were necessary to overcome challenges
in quality, coherence, and control.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04999v1)
- [arXiv](https://arxiv.org/abs/2510.04999v1)

---

<a id='2510.04127v1'></a>
## [Learning-Based Hashing for ANN Search: Foundations and Early Advances](https://arxiv.org/abs/2510.04127v1)

**Authors:** Sean Moran

**Published:** 2025-10-05

**Categories:** cs.IR, cs.AI, cs.CV, cs.LG

**Abstract:**

Approximate Nearest Neighbour (ANN) search is a fundamental problem in
information retrieval, underpinning large-scale applications in computer
vision, natural language processing, and cross-modal search. Hashing-based
methods provide an efficient solution by mapping high-dimensional data into
compact binary codes that enable fast similarity computations in Hamming space.
Over the past two decades, a substantial body of work has explored learning to
hash, where projection and quantisation functions are optimised from data
rather than chosen at random.
  This article offers a foundational survey of early learning-based hashing
methods, with an emphasis on the core ideas that shaped the field. We review
supervised, unsupervised, and semi-supervised approaches, highlighting how
projection functions are designed to generate meaningful embeddings and how
quantisation strategies convert these embeddings into binary codes. We also
examine extensions to multi-bit and multi-threshold models, as well as early
advances in cross-modal retrieval.
  Rather than providing an exhaustive account of the most recent methods, our
goal is to introduce the conceptual foundations of learning-based hashing for
ANN search. By situating these early models in their historical context, we aim
to equip readers with a structured understanding of the principles, trade-offs,
and open challenges that continue to inform current research in this area.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Sean Moran撰写的论文“Learning-Based Hashing for ANN Search: Foundations and Early Advances”的全面摘要。

---

**论文摘要：Learning-Based Hashing for ANN Search: Foundations and Early Advances**

这篇论文由Sean Moran撰写，旨在对早期基于学习的哈希方法进行基础性综述，这些方法主要用于近似最近邻（ANN）搜索。ANN搜索是信息检索中的一个核心问题，广泛应用于计算机视觉、自然语言处理和跨模态搜索等大规模应用。

**1. 主要问题或研究问题：**
论文主要探讨了如何通过将高维数据映射到紧凑的二进制代码（哈希码）来高效地解决大规模数据集中的ANN搜索问题，从而在汉明空间中实现快速相似性计算。核心研究问题在于，传统的哈希方法（如局部敏感哈希LSH）通常是数据无关的，其投影和量化函数是随机选择的，这导致了信息损失和次优的检索性能。因此，论文旨在回顾和分析那些通过数据优化投影和量化函数来提高哈希码质量和检索效率的早期“学习哈希”方法。

**2. 关键创新或方法论贡献：**
论文回顾了早期学习哈希方法在以下几个方面的创新：

*   **数据驱动的投影函数学习：** 强调了通过数据优化投影函数（即哈希超平面）的重要性，而非随机选择。这包括无监督方法（如PCAH、SH、ITQ、AGH）和有监督方法（如ITQ+CCA、KSH、BRE、STH），它们旨在学习能够更好地保留原始空间相似性的超平面。
*   **多位和多阈值量化策略：** 探讨了超越简单符号函数（单阈值）的量化方法，以减少信息损失。这包括分层量化（HQ）、双位量化（DBQ）和曼哈顿哈希量化（MHQ），它们通过引入多个阈值和更复杂的编码方案来提高哈希码的判别力。
*   **跨模态检索的早期进展：** 介绍了将学习哈希扩展到跨不同数据模态（如图像和文本）的检索任务，例如跨视图哈希（CVH）、协同正则化哈希（CRH）等。
*   **评估范式和指标：** 论文详细介绍了评估哈希方法性能的标准范式（汉明排序和哈希表桶评估）以及常用指标（AUPRC、mAP），并讨论了数据集划分策略以确保评估的稳健性。

**3. 主要结果及其意义：**
论文通过对早期模型的历史背景进行梳理，揭示了以下重要见解：

*   **数据感知二值化优于静态规则：** 优化量化阈值通常能产生比单一静态阈值更具判别力的代码。
*   **信息量不均的投影：** 非均匀分配阈值（即对信息量更大的投影进行更精细的量化）能提高检索效率。
*   **学习投影优于随机投影：** 引入监督信息来指导哈希超平面的放置，通常能提高效率。
*   **跨模态哈希的可行性和价值：** 跨模态投影学习能够实现最先进的检索性能。
*   **流水线耦合的重要性：** 联合学习投影和量化比孤立处理每个步骤能带来改进。

这些发现为后续学习哈希领域的研究奠定了基础，强调了数据依赖性方法在提高ANN搜索效率和准确性方面的潜力。

**4. 论文中提到的局限性：**
论文也指出了早期方法的一些局限性：

*   **量化模型缺乏监督信息：** 早期多阈值量化模型（如HQ、DBQ、MHQ）通常以无监督方式学习，未能充分利用标签信息。
*   **阈值分配的均匀性：** 阈值通常均匀分配在投影维度上，忽略了判别力的变化。
*   **计算成本高昂：** 有监督投影函数（如ITQ+CCA、KSH、BRE、STH）通常依赖于计算成本高昂的特征值分解或核方法，限制了可扩展性。
*   **缺乏统一框架：** 早期工作未能将投影超平面学习与多量化阈值学习结合在一个统一的框架中。
*   **评估方法的不一致性：** 不同的研究在地面真值定义（类标签与度量e-ball）、数据集划分协议（可能导致过拟合）和报告指标（mAP与AUPRC）方面存在不一致，影响了结果的可比性。

**5. 潜在的未来研究方向：**
基于上述局限性和早期研究的经验，论文提出了几个有前景的未来研究方向：

*   **人类对齐：** 验证度量改进（AUPRC/mAP）是否与用户满意度相关。
*   **在线/流式哈希：** 适应监督投影学习以处理非静态数据流。
*   **跨语言检索：** 将跨模态哈希扩展到多语言文档集合，不依赖翻译。
*   **位间依赖性：** 探索跨维度和跨超平面的依赖性阈值和去相关。
*   **端到端目标：** 在单一训练准则下联合优化投影和量化。

总而言之，这篇论文为学习哈希领域提供了一个宝贵的历史视角，总结了早期研究的成就、挑战和未来方向，为理解当前深度学习时代哈希方法的发展提供了坚实的基础。

**Key Findings:**

- Rather than providing an exhaustive account of the most recent methods, our
goal is to introduce the conceptual foundations of learning-based hashing for
ANN search.
- By situating these early models in their historical context, we aim
to equip readers with a structured understanding of the principles, trade-offs,
and open challenges that continue to inform current research in this area.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04127v1)
- [arXiv](https://arxiv.org/abs/2510.04127v1)

---

<a id='2510.05097v1'></a>
## [Pulp Motion: Framing-aware multimodal camera and human motion generation](https://arxiv.org/abs/2510.05097v1)

**Authors:** Robin Courant, Xi Wang, David Loiseaux, Marc Christie, Vicky Kalogeiton

**Published:** 2025-10-06

**Categories:** cs.GR, cs.CV

**Abstract:**

Treating human motion and camera trajectory generation separately overlooks a
core principle of cinematography: the tight interplay between actor performance
and camera work in the screen space. In this paper, we are the first to cast
this task as a text-conditioned joint generation, aiming to maintain consistent
on-screen framing while producing two heterogeneous, yet intrinsically linked,
modalities: human motion and camera trajectories. We propose a simple,
model-agnostic framework that enforces multimodal coherence via an auxiliary
modality: the on-screen framing induced by projecting human joints onto the
camera. This on-screen framing provides a natural and effective bridge between
modalities, promoting consistency and leading to more precise joint
distribution. We first design a joint autoencoder that learns a shared latent
space, together with a lightweight linear transform from the human and camera
latents to a framing latent. We then introduce auxiliary sampling, which
exploits this linear transform to steer generation toward a coherent framing
modality. To support this task, we also introduce the PulpMotion dataset, a
human-motion and camera-trajectory dataset with rich captions, and high-quality
human motions. Extensive experiments across DiT- and MAR-based architectures
show the generality and effectiveness of our method in generating on-frame
coherent human-camera motions, while also achieving gains on textual alignment
for both modalities. Our qualitative results yield more cinematographically
meaningful framings setting the new state of the art for this task. Code,
models and data are available in our
\href{https://www.lix.polytechnique.fr/vista/projects/2025_pulpmotion_courant/}{project
page}.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Robin Courant等人撰写的论文“Pulp Motion: Framing-aware multimodal camera and human motion generation”的全面摘要。

---

### 论文摘要：Pulp Motion: Framing-aware Multimodal Camera and Human Motion Generation

**1. 主要问题或研究问题：**
该论文旨在解决现有方法将人体运动和摄像机轨迹生成视为独立任务的局限性。这种分离忽略了电影摄影的核心原则——演员表演和摄像机工作在屏幕空间中紧密交织。因此，核心研究问题是如何实现文本条件下的联合生成，以在生成人体运动和摄像机轨迹这两种异构但内在关联的模态时，保持一致的屏幕构图（即“构图感知”），并确保多模态连贯性。

**2. 关键创新或方法论贡献：**
该论文提出了一个简单、模型无关的框架，通过引入一个辅助模态——**屏幕构图（on-screen framing）**来强制多模态连贯性。屏幕构图是通过将人体关节投影到摄像机上而产生的，它为不同模态之间提供了一个自然有效的桥梁。具体创新点包括：

*   **联合自编码器与共享潜在空间：** 设计了一个联合自编码器，学习一个共享的潜在空间来表示人体运动和摄像机轨迹。
*   **轻量级线性变换：** 引入了一个轻量级的线性变换，将人体和摄像机潜在表示映射到一个构图潜在表示。这个变换直接在潜在空间中捕捉了生成模态与辅助模态之间的关系。
*   **辅助采样机制：** 提出了一种辅助采样技术，利用上述线性变换在推理过程中引导生成，使其趋向于连贯的构图模态。这种方法在训练期间无需显式地将辅助模态纳入条件，降低了训练成本并提高了通用性。
*   **PulpMotion数据集：** 为了支持这项任务，论文还引入了PulpMotion数据集，这是一个包含丰富文本描述和高质量人体运动的人体运动和摄像机轨迹数据集，显著扩展了现有数据集的规模和模态覆盖范围。

**3. 主要结果及其意义：**
通过在DiT和MAR两种不同架构上进行广泛实验，论文展示了该方法的通用性和有效性：

*   **显著提升多模态连贯性：** 辅助采样显著改善了生成运动和轨迹之间的连贯性，降低了构图误差（FDframing）和出画率（Out-rate），同时保持了强大的单模态生成性能。
*   **提高文本对齐质量：** 该方法在人体运动和摄像机轨迹的文本对齐方面也取得了提升（TMR-Score和CLaTr-Score），表明生成内容与文本描述更加一致。
*   **电影摄影意义上的构图：** 定性结果表明，该方法生成的构图在电影摄影上更具意义，将该任务的现有技术水平推向了新的高度。

**4. 论文中提及的局限性：**
论文中没有明确提及当前方法的具体局限性，但可以从其未来工作方向中推断出一些潜在的限制：

*   **构图粒度：** 目前的构图可能仍是相对宏观的，尚未实现更精细的构图控制。

**5. 潜在的未来研究方向：**
论文提出了以下未来研究方向：

*   **扩展辅助模态方法到其他领域：** 将辅助模态方法推广到其他多模态生成任务中。
*   **实现更精细的构图控制：** 探索如何实现更精细的构图控制，例如针对特定身体部位的构图。

---

总而言之，这篇论文通过引入“屏幕构图”作为辅助模态，开创性地将人体运动和摄像机轨迹的生成任务统一为文本条件下的联合生成问题。其模型无关的框架和新颖的辅助采样机制，结合新发布的PulpMotion数据集，显著提升了生成内容的连贯性和文本对齐质量，为电影摄影领域带来了更具意义的构图，为多模态生成领域树立了新的基准。

**Key Findings:**

- We propose a simple,
model-agnostic framework that enforces multimodal coherence via an auxiliary
modality: the on-screen framing induced by projecting human joints onto the
camera.
- Extensive experiments across DiT- and MAR-based architectures
show the generality and effectiveness of our method in generating on-frame
coherent human-camera motions, while also achieving gains on textual alignment
for both modalities.
- Our qualitative results yield more cinematographically
meaningful framings setting the new state of the art for this task.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.05097v1)
- [arXiv](https://arxiv.org/abs/2510.05097v1)

---

<a id='2510.05051v1'></a>
## [SegMASt3R: Geometry Grounded Segment Matching](https://arxiv.org/abs/2510.05051v1)

**Authors:** Rohit Jayanti, Swayam Agrawal, Vansh Garg, Siddharth Tourani, Muhammad Haris Khan, Sourav Garg, Madhava Krishna

**Published:** 2025-10-06

**Categories:** cs.CV

**Abstract:**

Segment matching is an important intermediate task in computer vision that
establishes correspondences between semantically or geometrically coherent
regions across images. Unlike keypoint matching, which focuses on localized
features, segment matching captures structured regions, offering greater
robustness to occlusions, lighting variations, and viewpoint changes. In this
paper, we leverage the spatial understanding of 3D foundation models to tackle
wide-baseline segment matching, a challenging setting involving extreme
viewpoint shifts. We propose an architecture that uses the inductive bias of
these 3D foundation models to match segments across image pairs with up to 180
degree view-point change. Extensive experiments show that our approach
outperforms state-of-the-art methods, including the SAM2 video propagator and
local feature matching methods, by upto 30% on the AUPRC metric, on ScanNet++
and Replica datasets. We further demonstrate benefits of the proposed model on
relevant downstream tasks, including 3D instance segmentation and image-goal
navigation. Project Page: https://segmast3r.github.io/

**Analysis:**

SegMASt3R: Geometry Grounded Segment Matching 论文摘要

**1. 主要问题或研究问题：**
该论文旨在解决计算机视觉领域中“宽基线图像对的语义或几何连贯区域（即图像片段）匹配”这一具有挑战性的问题。与传统的关键点匹配不同，片段匹配需要建立图像间结构化区域的对应关系，这在存在极端视角变化（高达180度）、遮挡、光照变化和视角变化的情况下尤其困难。现有方法，包括依赖2D监督的局部特征匹配和视频传播器，在这种宽基线条件下表现不佳。

**2. 关键创新或方法论贡献：**
*   **利用3D基础模型（3DFM）的归纳偏置：** 论文的核心创新在于利用预训练的3D基础模型MASt3R的强大空间理解能力来解决宽基线片段匹配问题。MASt3R在各种3D视觉数据集上进行训练，能够捕捉场景的深度、形状和姿态等空间和结构属性，其几何感知表示对于处理极端视角变化至关重要。
*   **SegMASt3R架构：** 论文提出了一种名为SegMASt3R的架构，它在MASt3R骨干网络的基础上，引入了一个轻量级的“片段特征头”（Segment-Feature Head）。该头部将MASt3R解码器输出的补丁级特征转换为片段级描述符。
*   **可微分最优传输层：** 片段描述符通过一个可微分的最优传输层进行匹配，以建立图像间的片段对应关系，并通过行向argmax操作生成最终匹配结果。
*   **可学习的“垃圾桶”（Dustbin）：** 为了处理宽基线设置下可能存在的无匹配片段，模型引入了一个可学习的“垃圾桶”行和列到亲和矩阵中，以吸收非匹配项，从而提高匹配的鲁棒性。
*   **端到端训练：** 整个匹配层与上游片段编码器和下游任务损失一起进行端到端训练，采用SuperGlue的交叉熵损失函数。

**3. 主要结果及其意义：**
*   **显著优于现有技术：** SegMASt3R在ScanNet++和Replica数据集上，在AUPRC（精确度-召回率曲线下面积）指标上，比包括SAM2视频传播器和局部特征匹配方法在内的最先进方法高出高达30%。这表明其在处理宽基线图像对方面的卓越性能。
*   **在下游任务中的实用性：** 论文进一步证明了SegMASt3R在3D实例映射和目标相对导航等相关下游任务中的实际效用，并且在这些任务中也优于竞争对手。
*   **强大的泛化能力：** 模型在未见过的室内数据集Replica上表现出良好的泛化能力。即使在具有挑战性的室外MapFree数据集上，通过简单的可学习“垃圾桶”参数校准，模型也能显著缩小与领域转移相关的性能差距，展示了其学习到的几何特征的强大适应性。
*   **对噪声分割掩码的鲁棒性：** 即使在FastSAM生成的噪声分割掩码条件下，SegMASt3R也能保持显著的性能优势，这证实了其学习到的几何先验在不一致和噪声输入下的鲁棒性。

**4. 论文中提到的局限性：**
*   **未在主论文中明确讨论：** 论文在正文中并未设置专门的“局限性”章节，但根据NeurIPS清单的回答，作者表示局限性在补充材料中有所提及。
*   **计算资源：** 论文提到在ScanNet++上训练模型需要22小时，单次推理（批处理大小为1）需要0.579秒，这可能暗示了模型在某些应用场景下的计算成本。
*   **代码和数据发布：** 论文表示将在接受后发布代码和训练图像对，这意味着在论文提交时，代码和数据尚未公开。

**5. 潜在的未来研究方向：**
论文并未明确提出未来的研究方向，但从其贡献和局限性中可以推断出一些潜在方向：
*   **进一步提升宽基线匹配性能：** 尽管SegMASt3R已取得显著进展，但仍有提升空间，尤其是在极端视角变化和感知实例混叠等复杂场景下。
*   **更广泛的泛化能力：** 探索如何进一步提升模型在更多样化、更具挑战性的室外场景或跨领域数据集上的泛化能力，可能涉及更复杂的领域适应技术。
*   **计算效率优化：** 针对模型在推理和训练时的计算成本进行优化，使其更适用于资源受限的实时应用。
*   **结合其他3D基础模型：** 探索将SegMASt3R的框架应用于其他新兴的3D基础模型，以评估其性能和适用性。
*   **更复杂的下游任务：** 将SegMASt3R应用于更广泛的机器人感知和导航任务，例如更复杂的场景理解、人机交互等。

**Key Findings:**

- We propose an architecture that uses the inductive bias of
these 3D foundation models to match segments across image pairs with up to 180
degree view-point change.
- Extensive experiments show that our approach
outperforms state-of-the-art methods, including the SAM2 video propagator and
local feature matching methods, by upto 30% on the AUPRC metric, on ScanNet++
and Replica datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.05051v1)
- [arXiv](https://arxiv.org/abs/2510.05051v1)

---

<a id='2510.04876v1'></a>
## [BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping](https://arxiv.org/abs/2510.04876v1)

**Authors:** Hayat Rajani, Valerio Franchi, Borja Martinez-Clavel Valles, Raimon Ramos, Rafael Garcia, Nuno Gracias

**Published:** 2025-10-06

**Categories:** cs.CV, cs.LG, I.2.6; I.4.6; I.5.1; I.5.4

**Abstract:**

Benthic habitat mapping is fundamental for understanding marine ecosystems,
guiding conservation efforts, and supporting sustainable resource management.
Yet, the scarcity of large, annotated datasets limits the development and
benchmarking of machine learning models in this domain. This paper introduces a
thorough multi-modal dataset, comprising about a million side-scan sonar (SSS)
tiles collected along the coast of Catalonia (Spain), complemented by
bathymetric maps and a set of co-registered optical images from targeted
surveys using an autonomous underwater vehicle (AUV). Approximately \num{36000}
of the SSS tiles have been manually annotated with segmentation masks to enable
supervised fine-tuning of classification models. All the raw sensor data,
together with mosaics, are also released to support further exploration and
algorithm development. To address challenges in multi-sensor data fusion for
AUVs, we spatially associate optical images with corresponding SSS tiles,
facilitating self-supervised, cross-modal representation learning. Accompanying
open-source preprocessing and annotation tools are provided to enhance
accessibility and encourage research. This resource aims to establish a
standardized benchmark for underwater habitat mapping, promoting advancements
in autonomous seafloor classification and multi-sensor integration.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将对这篇论文摘要进行分析。

---

**论文摘要分析：BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping**

**1. 论文主要贡献的简明总结 (2-3 句话)**

这篇论文的核心贡献是引入了一个名为 BenthiCat 的大规模多模态数据集，用于海底栖息地测绘。该数据集包含近百万张侧扫声纳 (SSS) 图像，辅以水深图和来自自主水下航行器 (AUV) 的共配准光学图像，其中约 36,000 张 SSS 图像已手动标注分割掩码。通过发布原始传感器数据、镶嵌图、预处理和标注工具，该工作旨在为水下栖息地测绘提供一个标准化基准，并促进多传感器融合和自主海底分类的研究。

**2. 关键创新或方法论方法**

该论文的关键创新在于构建并发布了一个**大规模、多模态、且部分标注的“声学-光学”数据集**，专门针对海底栖息地测绘。其方法论亮点包括：

*   **多模态数据集成：** 将侧扫声纳 (SSS) 图像（提供广域覆盖和底质信息）与光学图像（提供高分辨率纹理和细节）以及水深图结合，克服了单一传感器数据的局限性。
*   **大规模标注：** 提供了约 36,000 张 SSS 图像的手动分割掩码标注，这对于监督学习模型的微调至关重要，尤其是在数据稀缺的海洋领域。
*   **空间关联与跨模态学习：** 强调了将光学图像与相应的 SSS 瓦片进行空间关联，以促进自监督、跨模态表征学习，这对于解决 AUV 多传感器数据融合的挑战至关重要。
*   **工具链支持：** 随数据集一同发布了开源的预处理和标注工具，极大地降低了研究人员的使用门槛，鼓励了社区参与和算法开发。

**3. 对领域潜在影响**

该研究对计算机视觉和海洋科学领域具有显著的潜在影响：

*   **推动海底分类模型发展：** BenthiCat 数据集将成为开发和基准测试新的机器学习模型（特别是深度学习模型）的宝贵资源，用于海底底质分类、栖息地识别和异常检测。
*   **促进多传感器融合研究：** 其多模态特性将激励研究人员探索更先进的传感器融合技术，以结合声学和光学数据的优势，提高水下环境感知的鲁棒性和准确性。
*   **加速自主水下航行器 (AUV) 能力：** 更好的海底测绘和分类能力将直接提升 AUV 的自主导航、目标识别和科学考察能力，使其能更有效地执行任务。
*   **建立标准化基准：** 作为第一个大规模的声学-光学海底数据集，它有望成为该领域的标准化基准，促进不同算法和方法的公平比较，加速研究进展。
*   **支持海洋生态保护：** 准确的栖息地测绘是海洋保护、资源管理和环境监测的基础，该数据集将直接支持这些应用。

**4. 可能受益的相关领域或应用**

*   **海洋生物学与生态学：** 用于研究底栖生物分布、栖息地健康评估、生物多样性监测。
*   **海洋地质学：** 用于海底地貌分析、底质类型识别、沉积物研究。
*   **水下机器人与自主系统：** 提升 AUV 的环境感知、路径规划、目标识别和自主决策能力。
*   **海洋工程：** 用于海底管线、电缆铺设路径规划、水下基础设施检查。
*   **渔业管理：** 评估渔业资源、识别鱼类栖息地。
*   **军事与安全：** 水下目标检测、水雷识别、海底地形分析。
*   **计算机视觉与机器学习：** 特别是多模态学习、自监督学习、语义分割、目标检测和传感器融合算法的开发与测试。

**5. 从摘要中可推断出的局限性**

*   **标注范围：** 尽管有 36,000 张 SSS 瓦片进行了手动分割标注，但相对于近百万张的总量，这仍然是相对较小的一部分。这意味着在某些应用中，可能需要进一步的半监督或无监督学习方法来利用未标注数据。
*   **光学数据覆盖：** 摘要提到光学图像来自“目标调查 (targeted surveys)”，这可能意味着光学数据的覆盖范围不如 SSS 数据广泛，且可能存在空间上的稀疏性或不连续性。这会影响光学数据在广域测绘中的直接应用，但对于局部高精度分析和跨模态学习仍有巨大价值。
*   **地理局限性：** 数据集收集于“加泰罗尼亚海岸 (coast of Catalonia, Spain)”，这意味着其地理范围有限。海底环境具有高度多样性，该数据集的泛化能力可能需要进一步验证，以适应全球不同区域的底质类型和声学/光学特征。
*   **标注粒度/类别：** 摘要未详细说明分割掩码的标注粒度（例如，是粗略的底质类型还是精细的生物群落）以及具体包含的类别数量。这会影响模型能够识别的细节程度。
*   **数据质量与噪声：** 侧扫声纳数据容易受到水体条件、传感器参数和海底地形复杂性的影响而产生噪声和伪影。摘要未提及数据预处理中对这些挑战的处理程度。
*   **时间维度：** 摘要未提及数据收集的时间跨度。如果数据是在不同时间点收集的，可能会存在环境变化（如季节性生物群落变化）带来的挑战。

---

总而言之，BenthiCat 数据集是计算机视觉和海洋科学交叉领域的一个重要里程碑。它通过提供一个前所未有的大规模、多模态数据集，有望显著加速海底栖息地测绘和自主水下机器人技术的发展，为海洋生态研究和保护提供强有力的工具。

**Key Findings:**

- Accompanying
open-source preprocessing and annotation tools are provided to enhance
accessibility and encourage research.
- This resource aims to establish a
standardized benchmark for underwater habitat mapping, promoting advancements
in autonomous seafloor classification and multi-sensor integration.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04876v1)
- [arXiv](https://arxiv.org/abs/2510.04876v1)

---

<a id='2510.04759v1'></a>
## [Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction](https://arxiv.org/abs/2510.04759v1)

**Authors:** Chi Yan, Dan Xu

**Published:** 2025-10-06

**Categories:** cs.CV, cs.AI

**Abstract:**

The 3D occupancy prediction task has witnessed remarkable progress in recent
years, playing a crucial role in vision-based autonomous driving systems. While
traditional methods are limited to fixed semantic categories, recent approaches
have moved towards predicting text-aligned features to enable open-vocabulary
text queries in real-world scenes. However, there exists a trade-off in
text-aligned scene modeling: sparse Gaussian representation struggles to
capture small objects in the scene, while dense representation incurs
significant computational overhead. To address these limitations, we present
PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables
open-vocabulary 3D occupancy prediction. Our framework employs progressive
online densification, a feed-forward strategy that gradually enhances the 3D
Gaussian representation to capture fine-grained scene details. By iteratively
enhancing the representation, the framework achieves increasingly precise and
detailed scene understanding. Another key contribution is the introduction of
an anisotropy-aware sampling strategy with spatio-temporal fusion, which
adaptively assigns receptive fields to Gaussians at different scales and
stages, enabling more effective feature aggregation and richer scene
information capture. Through extensive evaluations, we demonstrate that PG-Occ
achieves state-of-the-art performance with a relative 14.3% mIoU improvement
over the previous best performing method. Code and pretrained models will be
released upon publication on our project page:
https://yanchi-3dv.github.io/PG-Occ

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Chi Yan和Dan Xu撰写的论文《Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction》的全面摘要。

---

### 论文摘要：Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction

**1. 主要问题或研究问题：**
该论文旨在解决3D占用预测任务中的一个核心挑战：如何在自动驾驶系统中实现开放词汇的3D占用预测，同时克服现有方法在处理场景细节和计算效率方面的局限性。具体来说，现有方法要么使用固定语义类别，限制了对未知物体的感知；要么采用稀疏高斯表示，难以捕捉场景中的小物体；或者采用密集表示，导致巨大的计算开销。

**2. 关键创新或方法贡献：**
为了解决上述问题，作者提出了PG-Occ框架，其主要创新点包括：

*   **渐进式高斯Transformer框架（Progressive Gaussian Transformer Framework）：** PG-Occ通过一种创新的渐进式在线稠密化（Progressive Online Densification, POD）策略，迭代地增强3D高斯表示。它从粗略的基础高斯开始建模全局场景结构，然后根据感知误差逐步细化未充分感知的区域，从而捕捉精细的场景细节。这种前馈策略避免了梯度反向传播的计算开销，提高了效率。
*   **各向异性感知采样策略（Anisotropy-aware Sampling Strategy, AFS）：** 针对高斯表示的各向异性特性，PG-Occ引入了一种自适应采样方法。该方法根据每个高斯的空间分布调整其感受野，并将其投影到具有不同感受野的特征平面上，从而实现更有效的时空特征聚合和更丰富的场景信息捕获。
*   **非对称自注意力机制（Asymmetric Self-Attention, ASA）：** 为了在渐进式建模中保持训练稳定性，ASA确保新添加的高斯不会干扰已优化的高斯，同时允许新高斯利用现有信息进行自我完善。

**3. 主要结果及其意义：**
PG-Occ在Occ3D-nuScenes数据集上取得了最先进的性能，相对于之前表现最佳的方法，mIoU相对提升了14.3%。在nuScenes检索数据集上，PG-Occ也显著优于现有基于视觉的方法。这些结果表明：

*   **卓越的感知准确性：** PG-Occ能够更准确、更连贯地预测3D占用，捕捉更精细的结构细节，并生成更厚、更真实的表面。
*   **开放词汇能力：** 该框架能够根据任意文本查询进行零样本语义3D占用预测，有效桥接语言理解和空间感知之间的鸿沟。
*   **高效性：** 尽管模型复杂度增加，但通过渐进式稠密化和前馈策略，PG-Occ在训练时间和推理速度上仍具有竞争力，实现了效率与准确性的平衡。
*   **几何精度：** 在深度估计方面，PG-Occ也表现出色，甚至超越了原始监督标签的精度，这得益于多视角深度一致性和特征连贯性带来的几何约束。

**4. 论文中提及的局限性：**
论文中也坦诚地指出了PG-Occ的几个局限性：

*   **稀疏视角下的高斯尺度约束：** 在驾驶场景中，由于视角稀疏，约束高斯在深度上的尺度具有挑战性，可能导致“弹出”伪影（popping artifacts）。
*   **内存和计算成本：** 随着建模过程中高斯数量的增加，内存和计算成本也会随之增长，可能影响实时性能。
*   **小物体性能：** 尽管在检测中型物体方面表现出色，但由于粗糙的体素分辨率（0.4米），PG-Occ在小物体上的性能略低。
*   **遮挡问题：** 在缺乏视觉观测的区域，自监督方法在进行准确预测时可能面临挑战。

**5. 潜在的未来研究方向：**
为了解决上述局限性，作者提出了未来的研究方向：

*   探索**4D高斯方法**，以更好地处理时序信息和动态场景。
*   研究**多视角约束**，以进一步提高在稀疏视角下的高斯尺度约束和整体场景表示的准确性。

---

**Key Findings:**

- To address these limitations, we present
PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables
open-vocabulary 3D occupancy prediction.
- Another key contribution is the introduction of
an anisotropy-aware sampling strategy with spatio-temporal fusion, which
adaptively assigns receptive fields to Gaussians at different scales and
stages, enabling more effective feature aggregation and richer scene
information capture.
- Through extensive evaluations, we demonstrate that PG-Occ
achieves state-of-the-art performance with a relative 14.3% mIoU improvement
over the previous best performing method.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04759v1)
- [arXiv](https://arxiv.org/abs/2510.04759v1)

---

<a id='2510.04739v1'></a>
## [ExposureEngine: Oriented Logo Detection and Sponsor Visibility Analytics in Sports Broadcasts](https://arxiv.org/abs/2510.04739v1)

**Authors:** Mehdi Houshmand Sarkhoosh, Frøy Øye, Henrik Nestor Sørlie, Nam Hoang Vu, Dag Johansen, Cise Midoglu, Tomas Kupka, Pål Halvorsen

**Published:** 2025-10-06

**Categories:** cs.CV, cs.MM

**Abstract:**

Quantifying sponsor visibility in sports broadcasts is a critical marketing
task traditionally hindered by manual, subjective, and unscalable analysis
methods. While automated systems offer an alternative, their reliance on
axis-aligned Horizontal Bounding Box (HBB) leads to inaccurate exposuremetrics
when logos appear rotated or skewed due to dynamic camera angles and
perspective distortions. This paper introduces ExposureEngine, an end-to-end
system designed for accurate, rotation-aware sponsor visibility analytics in
sports broadcasts, demonstrated in a soccer case study. Our approach predicts
Oriented Bounding Box (OBB) to provide a geometrically precise fit to each logo
regardless of the orientation on-screen. To train and evaluate our detector, we
developed a new dataset comprising 1,103 frames from Swedish elite soccer,
featuring 670 unique sponsor logos annotated with OBBs. Our model achieves a
mean Average Precision (mAP@0.5) of 0.859, with a precision of 0.96 and recall
of 0.87, demonstrating robust performance in localizing logos under diverse
broadcast conditions. The system integrates these detections into an analytical
pipeline that calculates precise visibility metrics, such as exposure duration
and on-screen coverage. Furthermore, we incorporate a language-driven agentic
layer, enabling users to generate reports, summaries, and media content through
natural language queries. The complete system, including the dataset and the
analytics dashboard, provides a comprehensive solution for auditable and
interpretable sponsor measurement in sports media. An overview of the
ExposureEngine is available online: https://youtu.be/tRw6OBISuW4 .

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您总结Mehdi Houshmand Sarkhoosh等人撰写的论文“ExposureEngine: Oriented Logo Detection and Sponsor Visibility Analytics in Sports Broadcasts”。

---

### ExposureEngine: 体育赛事广播中定向标志检测与赞助商可见性分析

**论文摘要**

这篇论文介绍了 **ExposureEngine**，一个端到端系统，旨在解决体育赛事广播中赞助商可见性分析的传统挑战。该系统通过引入定向边界框（OBB）检测和语言驱动的分析层，显著提高了赞助商曝光度量衡的准确性和可解释性。

**1. 主要问题或研究问题**

传统上，体育赛事广播中的赞助商可见性量化是一个关键的营销任务，但受限于手动、主观且难以扩展的分析方法。现有的自动化系统通常依赖于轴对齐水平边界框（HBB），当标志因动态摄像机角度和透视畸变而旋转或倾斜时，HBB会导致不准确的曝光度量。HBB会包含非标志背景区域，从而高估标志尺寸和屏幕突出度。因此，核心问题是如何实现精确、旋转感知的标志检测，并将其转化为可审计和可解释的赞助商可见性分析。

**2. 关键创新或方法论贡献**

*   **定向边界框（OBB）检测模型：** ExposureEngine的核心创新在于采用基于YOLOv11的OBB检测模型。与HBB不同，OBB能够为每个标志提供几何上精确的拟合，无论其在屏幕上的方向如何，从而避免了背景区域的过度包含，提高了尺寸和位置估计的准确性。
*   **新数据集的创建：** 为了训练和评估其检测器，作者构建了一个包含1,103帧瑞典精英足球比赛的新数据集，其中包含670个独特的赞助商标志，并使用OBB进行标注。这是首个提供基于OBB的足球广播赞助商标志标注的开放数据集。
*   **质量感知损失函数（Varifocal Loss, VFL）：** 为了解决长尾类别分布和类别不平衡问题，模型采用了VFL作为分类损失，它能有效降低简单负样本的权重，并根据定位质量提高正样本的权重，从而更好地校准类别置信度与边界框精度。
*   **端到端分析管道：** 系统将OBB检测结果集成到一个分析管道中，用于计算精确的可见性指标，如曝光时长和屏幕覆盖率。
*   **语言驱动的智能体层：** 引入了一个基于大型语言模型（LLM）的智能体层，使用户能够通过自然语言查询生成报告、摘要和媒体内容，支持排名、统计查询和内容生成等操作。

**3. 主要结果及其意义**

*   **高检测性能：** OBB检测模型在测试集上实现了0.859的平均精度（mAP@0.5），精确度为0.96，召回率为0.87。这表明在各种广播条件下，模型在定位标志方面表现出强大的鲁棒性。
*   **OBB的几何精度优势：** 与HBB相比，OBB能够更紧密地包围标志区域，最大限度地减少冗余背景。紧密度比率（TR）分析表明，OBB在标志旋转时能提供更紧密的包围，从而实现更准确的屏幕面积和位置估计。
*   **系统效率：** GPU加速使得系统能够以接近实时（19.98 FPS）的速度运行，适用于实时仪表板和自动化亮点生成。
*   **可审计和可解释性：** 完整的系统，包括数据集和分析仪表板，为体育媒体中赞助商测量提供了一个全面、可审计和可解释的解决方案。

**4. 论文中提到的局限性**

*   **数据稀疏性：** 模型的准确性最终受限于可用数据。瑞典足球语料库的长尾类别分布导致每类别AP的方差增加，并限制了不常出现赞助商的召回率。VFL虽然有所帮助，但对于稀有类别，其表示能力仍然是瓶颈。
*   **泛化能力：** 鲁棒的泛化能力需要跨不同联赛、制作风格和场馆进行测试。
*   **时间连贯性：** 帧级检测结果容易出现漂移和闪烁，需要OBB感知的跟踪机制来形成稳定的对象轨迹，以平滑瞬态错误并防止重复计数。

**5. 潜在的未来研究方向**

*   **多管齐下的数据策略：** 扩展跨更多赛季和联赛的覆盖范围，引入分层标签，采用类别感知采样和有针对性的数据增强（如复制-粘贴），以及利用未标注广播的半监督学习，以解决稀有类别问题。
*   **从存在量化到价值评估：** 未来的工作应从量化存在转向评估价值。开发一个事件加权系统，利用比赛元数据（如进球、射门、黄牌）为关键时刻出现的标志赋予更高的价值。
*   **区域兴趣（ROI）分析：** 结合ROI分析，特别是针对社交媒体的垂直ROI，以衡量标志在不同平台分发渠道中的实用性。
*   **智能体层的演进：** 将智能体层从简单的内容打包器发展为战略工具，能够根据上下文和格式感知指标自动识别和呈现高价值片段。
*   **验证和审计：** 对聚合指标的最终偏差和方差进行严格的审计，以对抗人工标注的真实数据。

---

总而言之，ExposureEngine通过引入OBB检测和智能体驱动的分析，为体育赛事广播中的赞助商可见性分析提供了一个创新且全面的解决方案。它不仅提高了检测的几何精度，还通过智能体层增强了用户交互和数据解释能力，为体育营销和媒体分析领域带来了显著进步。

**Key Findings:**

- Our approach predicts
Oriented Bounding Box (OBB) to provide a geometrically precise fit to each logo
regardless of the orientation on-screen.
- To train and evaluate our detector, we
developed a new dataset comprising 1,103 frames from Swedish elite soccer,
featuring 670 unique sponsor logos annotated with OBBs. Our model achieves a
mean Average Precision (mAP@0.5) of 0.859, with a precision of 0.96 and recall
of 0.87, demonstrating robust performance in localizing logos under diverse
broadcast conditions.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04739v1)
- [arXiv](https://arxiv.org/abs/2510.04739v1)

---

<a id='2510.04714v1'></a>
## [Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction](https://arxiv.org/abs/2510.04714v1)

**Authors:** KunHo Heo, GiHyun Kim, SuYeon Kim, MyeongAh Cho

**Published:** 2025-10-06

**Categories:** cs.CV

**Abstract:**

3D Semantic Scene Graph Prediction aims to detect objects and their semantic
relationships in 3D scenes, and has emerged as a crucial technology for
robotics and AR/VR applications. While previous research has addressed dataset
limitations and explored various approaches including Open-Vocabulary settings,
they frequently fail to optimize the representational capacity of object and
relationship features, showing excessive reliance on Graph Neural Networks
despite insufficient discriminative capability. In this work, we demonstrate
through extensive analysis that the quality of object features plays a critical
role in determining overall scene graph accuracy. To address this challenge, we
design a highly discriminative object feature encoder and employ a contrastive
pretraining strategy that decouples object representation learning from the
scene graph prediction. This design not only enhances object classification
accuracy but also yields direct improvements in relationship prediction.
Notably, when plugging in our pretrained encoder into existing frameworks, we
observe substantial performance improvements across all evaluation metrics.
Additionally, whereas existing approaches have not fully exploited the
integration of relationship information, we effectively combine both geometric
and semantic features to achieve superior relationship prediction.
Comprehensive experiments on the 3DSSG dataset demonstrate that our approach
significantly outperforms previous state-of-the-art methods. Our code is
publicly available at https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供KunHo Heo, GiHyun Kim, SuYeon Kim, MyeongAh Cho撰写的论文“Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction”的全面摘要。

---

### 论文摘要：Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction

**1. 主要问题或研究问题：**
该论文旨在解决3D语义场景图预测（3DSSG）中的核心挑战。现有方法在优化对象和关系特征的表征能力方面存在不足，过度依赖图神经网络（GNN）但判别能力不足，导致对象分类不准确，进而影响关系预测的准确性。具体而言，研究发现对象特征的质量对整体场景图的准确性起着关键作用，且现有方法未能充分整合关系信息。

**2. 关键创新或方法论贡献：**
为了解决上述问题，作者提出了以下关键创新：
*   **判别性对象特征编码器（Discriminative Object Feature Encoder）：** 论文提出并预训练了一个高度判别性的对象特征编码器，将对象表征学习与场景图预测解耦。该编码器通过对比预训练策略，利用3D对象实例及其2D图像视图和文本描述之间的对应关系，增强语义表达能力，同时保持几何不变性。
*   **关系特征编码器（Relationship Feature Encoder）：** 该编码器有效结合了几何和语义特征，以实现更优越的关系预测。它通过引入局部空间增强（Local Spatial Enhancement, LSE）模块来平衡高维对象嵌入和相对简单的几何描述符之间的信息不平衡，并设计了一个双向边缘门控（Bidirectional Edge Gated, BEG）机制的GNN，以明确建模主体-客体不对称性。
*   **全局空间增强（Global Spatial Enhancement, GSE）：** 该机制通过整合全局几何位置信息，将对象关系情境化，以捕捉全局空间依赖性，进一步提升关系预测的准确性。

**3. 主要结果及其意义：**
*   **显著提升对象分类和关系预测：** 论文通过广泛分析表明，对象特征的质量对整体场景图准确性至关重要。所提出的判别性对象特征编码器不仅提高了对象分类准确性，还直接改善了关系预测。
*   **超越现有SOTA方法：** 将预训练的编码器集成到现有框架中时，在所有评估指标上都观察到显著的性能提升。在3DSSG数据集上的综合实验表明，该方法显著优于先前的最先进方法。
*   **有效整合多模态信息：** 论文强调了视觉、文本和几何信号的联合利用，能够产生更清晰的对象后验，从而推动下游场景图指标的提升。

**4. 论文中提及的局限性：**
*   **缺乏3D对象检测能力：** 本研究严格遵循3DSSG的评估协议，因此不包含3D对象检测能力。这意味着该方法不能直接应用于需要执行3D对象检测的真实世界场景。
*   **不支持增量图更新：** 该方法需要整个场景可用才能生成场景图，不支持增量图更新，这限制了其在实际场景部署中的应用。
*   **闭集词汇设置：** 本研究主要关注闭集词汇设置，以验证其核心假设，这与开放词汇设置存在差异。

**5. 潜在的未来研究方向：**
*   **集成3D对象检测：** 未来的工作目标是开发一个集成框架，将3D对象检测与增量场景图生成模块相结合，以实现更实用的场景图生成算法。
*   **利用现有高性能对象表征方法：** 进一步研究如何利用现有高性能对象表征方法，以显著提升整体场景图生成性能。
*   **探索开放词汇设置：** 将当前框架扩展到开放词汇设置，通过解耦细粒度对象和谓词类别，并对齐嵌入空间，以适应开放词汇3D场景图预测。

---

这份摘要突出了论文的核心贡献，即通过增强对象特征的判别能力和有效整合多模态关系信息来改进3D场景图预测，并指出了其在实际应用中的局限性及未来的研究方向。

**Key Findings:**

- In this work, we demonstrate
through extensive analysis that the quality of object features plays a critical
role in determining overall scene graph accuracy.
- Comprehensive experiments on the 3DSSG dataset demonstrate that our approach
significantly outperforms previous state-of-the-art methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04714v1)
- [arXiv](https://arxiv.org/abs/2510.04714v1)

---

<a id='2510.04706v1'></a>
## [ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion](https://arxiv.org/abs/2510.04706v1)

**Authors:** Foivos Paraperas Papantoniou, Stefanos Zafeiriou

**Published:** 2025-10-06

**Categories:** cs.CV

**Abstract:**

Human-centric generative models designed for AI-driven storytelling must
bring together two core capabilities: identity consistency and precise control
over human performance. While recent diffusion-based approaches have made
significant progress in maintaining facial identity, achieving fine-grained
expression control without compromising identity remains challenging. In this
work, we present a diffusion-based framework that faithfully reimagines any
subject under any particular facial expression. Building on an ID-consistent
face foundation model, we adopt a compositional design featuring an expression
cross-attention module guided by FLAME blendshape parameters for explicit
control. Trained on a diverse mixture of image and video data rich in
expressive variation, our adapter generalizes beyond basic emotions to subtle
micro-expressions and expressive transitions, overlooked by prior works. In
addition, a pluggable Reference Adapter enables expression editing in real
images by transferring the appearance from a reference frame during synthesis.
Extensive quantitative and qualitative evaluations show that our model
outperforms existing methods in tailored and identity-consistent expression
generation. Code and models can be found at
https://github.com/foivospar/Arc2Face.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Foivos Paraperas Papantoniou和Stefanos Zafeiriou撰写的论文“ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion”的全面摘要。

---

### 论文摘要：ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion

这篇论文提出了一种基于扩散模型的新颖框架，旨在实现对人脸表情的精细、身份一致的生成和编辑。

**1. 主要问题或研究问题：**
人脸生成模型在AI驱动的故事叙述中需要具备两个核心能力：身份一致性（identity consistency）和对人类表演的精确控制（precise control over human performance）。尽管近期基于扩散的方法在保持面部身份方面取得了显著进展，但在不损害身份一致性的前提下实现对精细表情的控制仍然是一个挑战。现有方法往往难以捕捉表情的全部精度和细微之处，尤其是在处理极端或不对称表情时，并且可能引入身份失真。

**2. 关键创新或方法论贡献：**
该论文的核心贡献在于其提出的扩散模型框架，它能够忠实地重塑任何主体在任何特定面部表情下的形象。具体创新包括：

*   **基于ID一致性基础模型的构建：** 该方法建立在Arc2Face [33] 这一ID一致性人脸基础模型之上，该模型能够生成具有高度身份相似性的多样化、逼真的人脸图像。
*   **表情交叉注意力模块（Expression Cross-Attention Module）：** 引入了一个组合式设计，通过FLAME blendshape参数指导的表情交叉注意力模块，实现了对表情的显式控制。FLAME 3D人脸模型 [21] 提供的参数化表示，实现了对表情的连续、高维控制，并将表情控制问题从图像空间映射到与主体无关的3D模型参数空间。
*   **多样化数据训练：** 模型在包含丰富表情变化的图像和视频数据混合集上进行训练，使其能够泛化到基本情绪之外的微妙微表情和表情过渡，这是以往工作所忽视的。
*   **可插拔参考适配器（Pluggable Reference Adapter）：** 引入了一个参考适配器，通过在合成过程中从参考帧转移外观，实现了在真实图像中进行表情编辑，同时不改变主体的外观或背景。该适配器通过Reference UNet提取空间对齐的特征，并与主UNet的自注意力层融合，同时使用LoRA层进行微调以解决参考表情与目标表情不一致可能带来的冲突。

**3. 主要结果及其意义：**
广泛的定量和定性评估表明，该模型在定制化和身份一致的表情生成方面优于现有方法。
*   **表情保真度：** 在表情一致性方面显著优于基于AU（Action Units）和渲染的替代方法，能够准确地传递表情，包括极端和不对称表情。
*   **图像质量和身份保持：** 实现了较低的FID分数，表明更高的视觉质量，并且始终保持与输入主体的高度身份相似性。
*   **用户研究：** 用户研究结果显示，该方法在表情准确性方面获得了72%的投票，证明了其强大的表情保真度。
*   **参考驱动编辑：** 在参考驱动的表情生成任务中，也取得了优于现有方法的表现，能够更忠实地转移表情，同时更好地保留身份和视觉一致性。

这些结果突出了其精确、参数化表情表示的有效性，以及在ID一致性人脸生成背景下实现精细表情控制的能力。

**4. 论文中提及的局限性：**
*   **参数化表示的语义可解释性：** 所采用的参数化表示（FLAME blendshape）缺乏语义可解释性，导致表情操作通常依赖于从参考图像中提取参数，这可能限制某些应用。
*   **对3D重建方法的依赖：** 表情再现的准确性固有地依赖于用于提取blendshape参数的3D重建方法的质量，尽管该方法是SOTA，但仍可能存在缺陷并引入偶尔的错误。
*   **参考适配器的一致性问题：** 在某些情况下，使用参考适配器进行表情编辑可能不一致。当参考表情与目标表情差异较大时，模型可能过度依赖源图像，导致背景或姿态与参考图像略有偏差。虽然可以通过调整LoRA层的缩放因子来缓解，但这需要在姿态和背景一致性与表情保真度之间进行权衡。

**5. 潜在的未来研究方向：**
论文中没有明确提出未来的研究方向，但从其局限性和贡献中可以推断出一些潜在方向：
*   **提升语义可控性：** 探索将FLAME blendshape参数与更具语义可解释性的表示（如自然语言描述或高级情感类别）相结合，以实现更直观的表情控制。
*   **改进3D重建的鲁棒性：** 进一步研究和集成更鲁棒、更精确的3D人脸重建方法，以减少表情提取中的错误，从而提高整体表情生成的准确性。
*   **增强参考适配器的一致性：** 探索更先进的机制来解决参考适配器在处理复杂表情差异时可能出现的“复制粘贴”行为，例如通过更智能的特征融合或自适应权重调整，以在保持背景和姿态一致性的同时，确保表情的准确转移。
*   **社会影响与伦理考量：** 论文强调了可控人脸生成技术可能被滥用的伦理问题。未来的研究可以专注于开发检测合成内容的对策，并确保技术用于积极的领域，如可访问性和创意故事叙述。

---

**Key Findings:**

- In this
work, we present a diffusion-based framework that faithfully reimagines any
subject under any particular facial expression.
- Extensive quantitative and qualitative evaluations show that our model
outperforms existing methods in tailored and identity-consistent expression
generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2510.04706v1)
- [arXiv](https://arxiv.org/abs/2510.04706v1)

---

