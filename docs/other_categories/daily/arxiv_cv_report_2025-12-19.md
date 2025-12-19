time: 20251219

# Arxiv Computer Vision Papers - 2025-12-19

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份关于2025年12月18日 Arxiv 计算机视觉领域论文的简明执行摘要。这份摘要旨在帮助忙碌的研究人员快速了解该领域最重要的发展。

---

**执行摘要：2025年12月18日 Arxiv 计算机视觉论文精选**

**主要主题与趋势：**

本期 Arxiv 论文集呈现出几个显著的趋势：

*   **多模态融合的深化与泛化：** 多个研究聚焦于整合文本、图像、视频甚至动作信息，以构建更强大、更通用的模型。这包括评估多模态奖励模型、开发用于自动驾驶的视觉-语言-动作模型，以及利用参考图像、轨迹和文本进行事件生成。
*   **基础模型的构建与应用：** 论文中出现了构建通用基础模型的努力，例如用于全景深度估计的“Depth Any Panoramas”，以及旨在提升视觉学习能力的“Next-Embedding Prediction”。
*   **模型的可解释性、审计与鲁棒性：** 研究开始关注模型的内部机制和能力差异，例如通过“Differences That Matter”来审计模型并发现和纠正能力差距。
*   **视觉内容生成与编辑的进步：** 论文展示了在图像和视频生成与编辑方面的创新，如“The World is Your Canvas”和“EasyV2V”。
*   **自适应与工具使用：** “AdaTooler-V”展示了模型在处理图像和视频时自适应工具使用的能力，预示着模型将更加灵活和高效。

**特别值得关注的论文：**

*   **"Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image"**：这篇论文在多模态奖励模型评估方面迈出了重要一步，对于衡量和改进跨文本和图像的通用奖励模型至关重要。
*   **"The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text"**：该研究在事件生成方面展现了令人兴奋的潜力，通过多模态输入实现更具创造性和可控性的内容生成。
*   **"Next-Embedding Prediction Makes Strong Vision Learners"**：这项工作提出了一种简单但有效的方法来提升视觉学习器的性能，可能对未来的视觉模型预训练产生广泛影响。

**新兴研究方向与技术：**

*   **全能奖励模型（Omni Reward Models）：** 评估和开发能够处理多种模态输入的奖励模型是当前的热点。
*   **视觉-语言-动作（VLA）模型在特定领域的应用：** 尤其是在自动驾驶等复杂场景中，VLA模型的整合和发展是关键。
*   **基于参考的生成模型：** 利用参考图像、轨迹和文本来指导内容生成，实现更精细化的控制。
*   **自适应工具使用：** 模型能够根据任务和数据动态选择和使用工具，是提升模型泛化能力和效率的重要方向。
*   **模型能力审计与对齐：** 深入理解模型的内在能力，并识别和弥合能力差距，对于构建可靠和负责任的AI至关重要。

**建议阅读全文的论文：**

考虑到其对多模态评估、基础模型构建以及内容生成领域的潜在影响，以下论文值得深入阅读：

1.  **"Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image"** (对于理解多模态模型评估的最新进展至关重要)
2.  **"The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text"** (对于对生成模型和多模态内容创作感兴趣的研究人员)
3.  **"Next-Embedding Prediction Makes Strong Vision Learners"** (对于希望提升视觉模型性能和理解预训练技术的研究人员)
4.  **"Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation"** (对于3D视觉和基础模型研究人员)

---

希望这份摘要能帮助您快速掌握近期 Arxiv 计算机视觉领域的最新动态。

---

## Table of Contents

1. [Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image](#2512.16899v1)
2. [Kling-Omni Technical Report](#2512.16776v1)
3. [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](#2512.16760v1)
4. [The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text](#2512.16924v1)
5. [Next-Embedding Prediction Makes Strong Vision Learners](#2512.16922v1)
6. [EasyV2V: A High-quality Instruction-based Video Editing Framework](#2512.16920v1)
7. [DVGT: Driving Visual Geometry Transformer](#2512.16919v1)
8. [Differences That Matter: Auditing Models for Capability Gap Discovery and Rectification](#2512.16921v1)
9. [AdaTooler-V: Adaptive Tool-Use for Images and Videos](#2512.16918v1)
10. [Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation](#2512.16913v1)

---

## Papers

<a id='2512.16899v1'></a>
## [Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image](https://arxiv.org/abs/2512.16899v1)

**Authors:** Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, Marjan Ghazvininejad

**Published:** 2025-12-18

**Categories:** cs.CL, cs.CV

**Abstract:**

Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning ("thinking-with-images"), providing 1,000 expert-annotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66-75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：**

**Title:** Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image
**Authors:** Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, Marjan Ghazvininejad
**Categories:** cs.CL, cs.CV
**Published Date:** 2025-12-18

**Abstract:**
Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning ("thinking-with-images"), providing 1,000 expert-annotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66-75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.

---

**中文分析：**

**1. 论文的主要贡献（2-3句话）：**
本论文提出了 Multimodal RewardBench 2 (MMRB2)，这是首个针对处理交织文本和图像的“全能型”（omni）模型而设计的、全面的多模态奖励模型（RM）评估基准。MMRB2 涵盖了文本到图像生成、图像编辑、交织生成和多模态推理等四个关键任务，通过高质量、专家标注的偏好数据，为评估和改进多模态奖励模型提供了重要的资源和研究方向。

**2. 关键创新或方法论：**
*   **首个全面的多模态奖励模型基准 (MMRB2)：** 论文的核心创新在于构建了一个专门针对处理交织文本和图像的多模态奖励模型的基准。这填补了现有研究中对这类模型评估的空白，因为以往的奖励模型研究主要集中在纯文本领域。
*   **任务多样性与深度：** MMRB2 不仅包含基础的生成任务（文本到图像、图像编辑、交织生成），还引入了更具挑战性的“图像思维”（thinking-with-images）多模态推理任务，这要求模型能够理解和整合视觉信息进行逻辑推理。
*   **高质量、专家标注的数据集：** 论文强调了其数据集的质量，包括：
    *   **实用且具挑战性的提示 (prompts)：** 确保评估的真实性和难度。
    *   **来自 SOTA 模型和代理的响应：** 提供了当前最先进模型生成的输出，用于比较。
    *   **强人类专家共识的偏好对：** 通过“集成过滤策略”（ensemble filtering strategy）精心筛选，确保了标注数据的可靠性和一致性，这对于训练有效的奖励模型至关重要。
*   **评估方法的多样性：** 论文不仅评估了传统的基于人类偏好的奖励模型，还考察了“多模态 LLM-as-a-judge”等新兴的自动化评估方法，并对比了不同模型（包括 Gemini 3 Pro, GPT-5, Gemini 2.5 Pro, GPT-4o, Qwen3-VL-32B, Gemini 2.5 Flash）在 MMRB2 上的表现。

**3. 对该领域的潜在影响：**
*   **推动多模态奖励模型的发展：** MMRB2 的发布将极大地促进多模态奖励模型的研究和开发。研究人员将有了一个标准化的平台来评估和比较他们的模型，从而加速算法的迭代和优化。
*   **提升多模态生成和理解能力：** 通过更有效的奖励模型，可以训练出在多模态任务上表现更优的 LLM，从而提升模型在图像生成、编辑、多模态对话和推理等方面的能力。
*   **为模型评估提供新视角：** “LLM-as-a-judge”在多模态领域的应用和评估，为自动化模型评估提供了新的思路和实践，有助于降低人工评估的成本和时间。
*   **促进跨模态对齐和理解：** 评估奖励模型在处理交织文本和图像时的表现，间接推动了模型对视觉和语言信息之间深层对齐和理解的研究。

**4. 可能受益的相关领域或应用：**
*   **多模态对话系统：** 能够理解和生成包含图像的对话，例如智能助手、虚拟客服等。
*   **内容创作工具：** 辅助用户进行图像生成、编辑，以及根据文本描述创作包含图像的叙事内容。
*   **教育和培训：** 开发能够理解和解释图文并茂内容的智能教育平台。
*   **辅助技术：** 为视障人士提供更丰富的图像描述和多模态信息交互。
*   **机器人和自动驾驶：** 提升机器人对复杂环境（包含图像和指令）的理解和决策能力。
*   **多模态搜索和推荐：** 结合图像和文本信息进行更精准的搜索和个性化推荐。

**5. 从摘要中可以推断出的局限性：**
*   **评估的局限性：** 尽管 MMRB2 旨在全面，但任何基准都可能无法完全覆盖所有潜在的多模态场景和挑战。摘要中提到“21个源任务”，这表明其覆盖范围是有限的。
*   **“LLM-as-a-judge”的局限性：** 尽管论文评估了 LLM 作为评判者，但 LLM 本身也可能存在偏见、幻觉或对某些细微差别的理解不足，这可能影响评估的准确性。摘要中也指出了 LLM 评判者与人类专家之间仍有差距（>90% vs 66-80%）。
*   **数据标注的成本和主观性：** 尽管强调了专家共识，但偏好标注本身仍然可能存在一定程度的主观性，尤其是在评估创造性或开放式生成任务时。
*   **模型性能的相对性：** 摘要中给出的模型性能是相对于 MMRB2 基准而言的，并不代表模型在所有实际应用中的绝对表现。
*   **未来研究方向的提示：** 摘要中提到“key areas to improve the reward models going forward”，这暗示了当前奖励模型在某些方面仍存在不足，需要进一步的研究和改进。

**对计算机视觉领域的趣味性或重要性：**

这篇论文对于计算机视觉领域具有重要的意义，主要体现在以下几个方面：

*   **视觉与语言的深度融合：** MMRB2 关注的是“交织”的文本和图像，这意味着模型需要处理的不仅仅是独立的图像或文本，而是它们之间相互嵌入、相互依赖的复杂关系。这直接推动了计算机视觉模型在理解和生成具有上下文关联的视觉内容方面的能力。
*   **从“理解”到“生成”的桥梁：** 论文涵盖了文本到图像生成、图像编辑和交织生成等任务，这表明其研究成果直接服务于生成式视觉模型的发展。通过更优的奖励模型，可以指导生成模型产生更符合人类偏好、更具创造性或更符合指令的图像。
*   **“图像思维”的挑战：** “thinking-with-images”这一任务尤其令人兴奋。它要求模型不仅能识别图像中的物体，还能基于图像内容进行推理、规划或解决问题。这需要模型具备更高级的视觉理解能力，能够从像素层面提取抽象概念，并将其与逻辑推理结合。这对于需要视觉感知和决策的机器人、自动驾驶等领域至关重要。
*   **评估方法的创新：** 引入“LLM-as-a-judge”来评估多模态模型，为计算机视觉领域提供了一种新的、可扩展的评估范式。这可以加速对视觉模型（尤其是生成模型）的评估过程，并可能发现人类评估者容易忽略的细微问题。
*   **推动多模态数据集的建设：** MMRB2 的成功将激励更多高质量、多模态数据集的创建，这些数据集将成为未来多模态研究的重要基石。

总而言之，这篇论文通过构建一个创新的、高质量的基准，直接解决了当前多模态模型（尤其是处理交织文本和图像的模型）在训练和评估中的关键瓶颈，为计算机视觉领域在理解、生成和推理方面的发展提供了重要的推动力。

**Key Findings:**

- We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation.
- MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16899v1)
- [arXiv](https://arxiv.org/abs/2512.16899v1)

---

<a id='2512.16776v1'></a>
## [Kling-Omni Technical Report](https://arxiv.org/abs/2512.16776v1)

**Authors:**  Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, Xiao Hu, Xiaohua Hu, Boyuan Jiang, Fangyuan Kong, Hang Li, Jie Li, Qingyu Li, Shen Li, Xiaohan Li, Yan Li, Jiajun Liang, Borui Liao, Yiqiao Liao, Weihong Lin, Quande Liu, Xiaokun Liu, Yilun Liu, Yuliang Liu, Shun Lu, Hangyu Mao, Yunyao Mao, Haodong Ouyang, Wenyu Qin, Wanqi Shi, Xiaoyu Shi, Lianghao Su, Haozhi Sun, Peiqin Sun, Pengfei Wan, Chao Wang, Chenyu Wang, Meng Wang, Qiulin Wang, Runqi Wang, Xintao Wang, Xuebo Wang, Zekun Wang, Min Wei, Tiancheng Wen, Guohao Wu, Xiaoshi Wu, Zhenhua Wu, Da Xie, Yingtong Xiong, Yulong Xu, Sile Yang, Zikang Yang, Weicai Ye, Ziyang Yuan, Shenglong Zhang, Shuaiyu Zhang, Yuanxing Zhang, Yufan Zhang, Wenzheng Zhao, Ruiliang Zhou, Yan Zhou, Guosheng Zhu, Yongjie Zhu

**Published:** 2025-12-18

**Categories:** cs.CV

**Abstract:**

We present Kling-Omni, a generalist generative framework designed to synthesize high-fidelity videos directly from multimodal visual language inputs. Adopting an end-to-end perspective, Kling-Omni bridges the functional separation among diverse video generation, editing, and intelligent reasoning tasks, integrating them into a holistic system. Unlike disjointed pipeline approaches, Kling-Omni supports a diverse range of user inputs, including text instructions, reference images, and video contexts, processing them into a unified multimodal representation to deliver cinematic-quality and highly-intelligent video content creation. To support these capabilities, we constructed a comprehensive data system that serves as the foundation for multimodal video creation. The framework is further empowered by efficient large-scale pre-training strategies and infrastructure optimizations for inference. Comprehensive evaluations reveal that Kling-Omni demonstrates exceptional capabilities in in-context generation, reasoning-based editing, and multimodal instruction following. Moving beyond a content creation tool, we believe Kling-Omni is a pivotal advancement toward multimodal world simulators capable of perceiving, reasoning, generating and interacting with the dynamic and complex worlds.

**Analysis:**

好的，这是对论文“Kling-Omni Technical Report”的全面摘要，重点关注其在计算机视觉和机器学习领域的贡献：

**论文题目：** Kling-Omni Technical Report

**作者：** Kling Team, Kuaishou Technology (Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, Xiao Hu, Xiaohua Hu, Boyuan Jiang, Fangyuan Kong, Hang Li, Jie Li, Qingyu Li, Shen Li, Xiaohan Li, Yan Li, Jiajun Liang, Borui Liao, Yiqiao Liao, Weihong Lin, Quande Liu, Xiaokun Liu, Yilun Liu, Yuliang Liu, Shun Lu, Hangyu Mao, Yunyao Mao, Haodong Ouyang, Wenyu Qin, Wanqi Shi, Xiaoyu Shi, Lianghao Su, Haozhi Sun, Peiqin Sun, Pengfei Wan, Chao Wang, Chenyu Wang, Meng Wang, Qiulin Wang, Runqi Wang, Xintao Wang, Xuebo Wang, Zekun Wang, Min Wei, Tiancheng Wen, Guohao Wu, Xiaoshi Wu, Zhenhua Wu, Da Xie, Yingtong Xiong, Yulong Xu, Sile Yang, Zikang Yang, Weicai Ye, Ziyang Yuan, Shenglong Zhang, Shuaiyu Zhang, Yuanxing Zhang, Yufan Zhang, Wenzheng Zhao, Ruiliang Zhou, Yan Zhou, Guosheng Zhu, Yongjie Zhu)

**摘要：**

**1. 主要问题或研究问题：**
当前视频生成领域存在功能分离、任务碎片化的问题，例如视频生成、编辑和智能推理被割裂开来。现有的模型在处理多样化的多模态输入（文本、图像、视频）并生成高质量、智能化的视频内容方面存在挑战。特别是，自然语言提示难以捕捉视觉细节和复杂的用户意图，导致模型难以实现精细的控制和深度的理解。此外，现有模型在语义推理和理解场景的物理逻辑方面能力有限，更像是被动的生成器而非智能的代理。

**2. 关键创新或方法论贡献：**
Kling-Omni 提出了一个**通用生成框架**，旨在解决上述问题，实现视频生成、编辑和智能推理任务的统一。其核心创新包括：

*   **多模态视觉语言 (MVL) 作为新的交互范式：** Kling-Omni 引入了 MVL 作为一种新的交互方式，将自然语言作为语义骨架，并结合多模态描述，构建统一的输入表示。这增强了模型对文本和视觉信号的理解与控制能力，使其能够更深入地理解和推断用户意图。
*   **端到端统一的框架：** Kling-Omni 将视频生成、编辑和推理整合到一个**整体系统**中，打破了传统上分离的流水线方法。它能够直接从 MVL 输入合成高保真度的视频。
*   **Prompt Enhancer (PE) 模块：** 该模块利用多模态大语言模型 (MLLM) 来理解复杂的用户输入，并将其映射到与模型训练数据分布一致的表示。PE 能够推断创作者的具体意图，并进行提示重构，从而提高生成质量，尤其是在身份保持、空间一致性和颜色保真度方面。
*   **多模态视觉语言 (MVL) 信号的整合：** Kling-Omni 能够处理文本指令、参考图像和视频上下文等多种输入，并将它们统一处理，生成电影级质量的视频内容。
*   **多阶段训练策略：** 框架采用了从指令预训练、监督微调到强化学习 (RL) 的多阶段训练策略，并结合了 DPO (Direct Preference Optimization) 来优化模型输出以符合人类审美偏好。
*   **高效的训练和推理优化：** 论文详细介绍了其在数据处理、模型加速（如两阶段蒸馏）、训练优化（如多模态数据管道和负载均衡、微批次弹性 ulysses 并行）以及推理优化（如模型并行、张量并行、计算-通信重叠、混合量化和缓存机制）方面的技术细节，以支持大规模训练和高效推理。
*   **全面的数据系统：** 构建了一个包含大规模真实世界数据采集和任务导向的合成数据构建的综合数据系统，并设计了三层数据处理流程（基础过滤、时间质量评估、视频-文本和图像-视频对齐）来确保数据质量。

**3. 主要结果及其意义：**
Kling-Omni 在各项评估中展现出卓越的能力，包括：

*   **卓越的多模态指令遵循能力：** 模型能够准确理解并执行复杂的文本、图像和视频组合指令。
*   **强大的推理和编辑能力：** 在推理驱动的编辑（如添加、移除、替换、背景替换、风格化）和多模态参考（如图像/元素库参考、新视角生成、运动迁移）方面表现出色。
*   **与 SOTA 模型相比的优越性：** 通过与 Veo 3.1 和 Runway-Aleph 等领先模型的对比评估，Kling-Omni 在动态质量、视觉质量、提示遵循和身份一致性等多个维度上均取得了显著的优势，尤其是在图像参考和视频编辑任务上。
*   **推动多模态世界模拟器的发展：** Kling-Omni 被认为是迈向能够感知、推理、生成和与动态复杂世界交互的多模态世界模拟器的重要一步。

**4. 论文中提到的局限性：**
论文中并未明确列出局限性，但从其对未来研究方向的展望中可以推测，一些高级功能（如“Features described in this section are not yet supported in the online version.”）可能尚未完全实现或集成到在线版本中。此外，虽然模型在推理方面取得了进展，但“更高级的交互式和推理增强型生成任务”仍有进一步探索的空间。

**5. 潜在的未来研究方向：**
论文展望了 Kling-Omni 在构建多模态世界模拟器方面的潜力，这暗示了未来的研究方向可能包括：

*   **更强的感知、推理和交互能力：** 进一步提升模型在理解和模拟复杂现实世界动态方面的能力。
*   **更精细的控制和创造力：** 探索更广泛的用户交互方式，例如通过视觉信号（如绘图、标注）进行更精细的视频控制，以及增强模型的自主创造力。
*   **更广泛的应用场景：** 将 Kling-Omni 应用于更广泛的领域，如电影制作、游戏开发、虚拟现实等。
*   **实时性和交互性：** 进一步优化模型的推理速度，以支持更实时的交互式视频生成和编辑。
*   **伦理和社会影响：** 随着模型能力的增强，对生成内容的真实性、偏见和滥用等伦理问题进行深入研究和应对。

**总结：**
Kling-Omni 技术报告介绍了一个开创性的通用视频生成框架，通过引入多模态视觉语言 (MVL) 范式，将视频生成、编辑和智能推理任务统一在一个端到端的系统中。该框架通过创新的 Prompt Enhancer 模块、多阶段训练策略以及高效的优化技术，实现了对复杂多模态输入的深刻理解和高保真度视频的生成。其在各项评估中展现出的优越性能，以及在推理和交互式编辑方面的强大能力，标志着其在迈向更智能、更通用的视频内容创作和多模态世界模拟器方面迈出了坚实的一步。这篇论文为未来视频生成和多模态人工智能的研究提供了重要的理论和技术基础。

**Key Findings:**

- We present Kling-Omni, a generalist generative framework designed to synthesize high-fidelity videos directly from multimodal visual language inputs.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16776v1)
- [arXiv](https://arxiv.org/abs/2512.16776v1)

---

<a id='2512.16760v1'></a>
## [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](https://arxiv.org/abs/2512.16760v1)

**Authors:** Tianshuai Hu, Xiaolu Liu, Song Wang, Yiyao Zhu, Ao Liang, Lingdong Kong, Guoyang Zhao, Zeying Gong, Jun Cen, Zhiyu Huang, Xiaoshuai Hao, Linfeng Li, Hang Song, Xiangtai Li, Jun Ma, Shaojie Shen, Jianke Zhu, Dacheng Tao, Ziwei Liu, Junwei Liang

**Published:** 2025-12-18

**Categories:** cs.RO

**Abstract:**

Autonomous driving has long relied on modular "Perception-Decision-Action" pipelines, where hand-crafted interfaces and rule-based components often break down in complex or long-tailed scenarios. Their cascaded design further propagates perception errors, degrading downstream planning and control. Vision-Action (VA) models address some limitations by learning direct mappings from visual inputs to actions, but they remain opaque, sensitive to distribution shifts, and lack structured reasoning or instruction-following capabilities. Recent progress in Large Language Models (LLMs) and multimodal learning has motivated the emergence of Vision-Language-Action (VLA) frameworks, which integrate perception with language-grounded decision making. By unifying visual understanding, linguistic reasoning, and actionable outputs, VLAs offer a pathway toward more interpretable, generalizable, and human-aligned driving policies. This work provides a structured characterization of the emerging VLA landscape for autonomous driving. We trace the evolution from early VA approaches to modern VLA frameworks and organize existing methods into two principal paradigms: End-to-End VLA, which integrates perception, reasoning, and planning within a single model, and Dual-System VLA, which separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners). Within these paradigms, we further distinguish subclasses such as textual vs. numerical action generators and explicit vs. implicit guidance mechanisms. We also summarize representative datasets and benchmarks for evaluating VLA-based driving systems and highlight key challenges and open directions, including robustness, interpretability, and instruction fidelity. Overall, this work aims to establish a coherent foundation for advancing human-compatible autonomous driving systems.

**Analysis:**

好的，这是对论文“Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future”的全面中文摘要：

**论文题目：** Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future

**作者：** Tianshuai Hu, Xiaolu Liu, Song Wang, Yiyao Zhu, Ao Liang, Lingdong Kong, Guoyang Zhao, Zeying Gong, Jun Cen, Zhiyu Huang, Xiaoshuai Hao, Linfeng Li, Hang Song, Xiangtai Li, Jun Ma, Shaojie Shen, Jianke Zhu, Dacheng Tao, Ziwei Liu, Junwei Liang

**摘要：**

这篇论文全面回顾了自动驾驶领域中从传统的“感知-决策-行动”（Perception-Decision-Action, PDA）模块化方法，到视觉-动作（Vision-Action, VA）模型，再到当前新兴的视觉-语言-动作（Vision-Language-Action, VLA）框架的演进历程。论文旨在为 VLA 模型在自动驾驶领域的快速发展提供一个结构化的视角，梳理其概念基础、架构趋势以及未来的研究方向。

**1. 主要问题或研究问题：**

传统的 PDA 自动驾驶系统在复杂、长尾场景下表现不佳，其模块化设计易导致误差累积。VA 模型虽然能直接从视觉输入映射到动作，但缺乏可解释性、泛化能力弱且难以进行结构化推理。因此，研究如何构建更智能、可解释、泛化能力强且能理解人类指令的自动驾驶系统是核心问题。

**2. 关键创新或方法论贡献：**

*   **结构化梳理与分类：** 论文首次对自动驾驶领域的 VLA 模型进行了系统性的梳理和分类。
    *   **VLA 架构分类：** 将 VLA 模型分为两大范式：
        *   **端到端 VLA (End-to-End VLA)：** 将感知、推理和规划集成在单个模型中。
        *   **双系统 VLA (Dual-System VLA)：** 将 VLM 的慢速推理与专用驾驶模块的快速执行分离。
    *   **子类划分：** 在两大范式下，进一步区分了文本式与数值式动作生成器，以及显式与隐式引导机制。
*   **演进历程追踪：** 详细追溯了从早期 VA 模型到现代 VLA 框架的发展脉络，阐述了 VLA 模型出现的动机。
*   **数据集与基准总结：** 系统性地总结了用于评估 VLA 驾驶系统的代表性数据集和基准，为模型评估提供了参考。
*   **挑战与未来方向展望：** 深入分析了 VLA 模型在实际部署中面临的关键挑战，并提出了未来研究方向。

**3. 主要结果及其意义：**

*   **VLA 模型的优势：** VLA 模型通过整合视觉理解、语言推理和可执行动作，提供了一种更具可解释性、泛化性和人类对齐性的自动驾驶策略。它们能够处理更复杂的场景，并理解人类的高级指令。
*   **架构范式的重要性：** 论文提出的端到端 VLA 和双系统 VLA 的分类，为理解不同 VLA 模型的设计理念和权衡提供了清晰的框架。
*   **推动领域发展：** 通过对现有工作的全面梳理和分类，论文为该领域的研究人员提供了一个结构化的路线图，有助于加速 VLA 模型在自动驾驶领域的进步。

**4. 论文中提到的局限性：**

*   **实时处理与延迟：** VLA 模型继承了大型视觉语言模型（VLM）的计算密集性，高分辨率、高帧率的输入会产生大量的视觉 token，多视图融合会加剧内存和延迟问题。实现自动驾驶所需的亚 50ms 推理仍然是一个挑战。
*   **领域特定基础模型缺失：** 通用 VLM 虽然提供了强大的先验知识，但并未针对自动驾驶的特定感知、物理或多传感器融合进行优化。自动驾驶需要精确的空间推理、遵守交通规则以及理解罕见的高风险场景，这些能力通用模型尚未完全捕捉。
*   **数据成本高昂：** VLA 模型依赖于多样化的高质量多模态数据集，但收集这些数据集成本高昂。合成环境可以提供帮助，但模拟到现实的差距（如噪声特性、光照和行为差异）仍然存在。
*   **可解释性与幻觉：** VLA 模型生成的自然语言解释（通过 CoT 等方式）可能是人为产物，而非真实因果推理的忠实反映。语言幻觉（hallucination）风险增加，模型可能用看似合理的叙述来解释错误的决策。确保感知、动作和解释之间的一致性是一个开放性挑战。
*   **长时序连贯性：** 当前基于 Transformer 的 VLA 模型受限于有限的上下文窗口和短时序条件，难以在长时序内保持情境感知和多阶段交互的连贯性，尤其是在多智能体或动态交通场景下，可能导致决策不一致。
*   **安全与可靠性：** 尽管 VLA 模型有望提高可解释性，但其推理失败、指令遵循错误或跨模态不一致等风险仍需深入研究和评估。

**5. 潜在的未来研究方向：**

*   **统一的视觉-语言-世界模型：** 将 VLA 与预测性世界模型相结合，模拟未来场景演变，实现主动规划和更可靠的决策。
*   **更丰富的多模态融合：** 整合更多传感器（如 LiDAR、雷达、事件相机、高精地图）的早期、紧密融合，并利用语言增强语义理解，同时确保 3D 几何的精确性。
*   **社会化和知识驱动的驾驶：** 培养模型更深层次的常识推理能力，理解意图、惯例和因果关系，并利用外部知识库支持社会化和预见性的驾驶行为。
*   **持续学习与在线学习：** 使模型能够适应不断变化的道路基础设施和区域驾驶习惯，实现安全、增量的日常驾驶学习，避免灾难性遗忘，并解决长尾泛化问题。
*   **标准化评估与安全保障：** 开发更全面的基准，评估多步指令执行、模糊语言处理和抗幻觉能力，并探索形式化验证工具以提供安全保证。
*   **以人为本的交互与个性化：** 实现更丰富的车内交互，允许用户指定驾驶偏好，并使模型能够适应不同用户的驾驶风格，同时平衡个性化与安全性。

总而言之，这篇论文为理解和推进自动驾驶领域的 VLA 模型研究提供了一个全面且深入的视角，指出了当前的研究现状、关键挑战以及未来的发展方向。

**Key Findings:**

- We also summarize representative datasets and benchmarks for evaluating VLA-based driving systems and highlight key challenges and open directions, including robustness, interpretability, and instruction fidelity.
- Overall, this work aims to establish a coherent foundation for advancing human-compatible autonomous driving systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16760v1)
- [arXiv](https://arxiv.org/abs/2512.16760v1)

---

<a id='2512.16924v1'></a>
## [The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text](https://arxiv.org/abs/2512.16924v1)

**Authors:** Hanlin Wang, Hao Ouyang, Qiuyu Wang, Yue Yu, Yihao Meng, Wen Wang, Ka Leong Cheng, Shuailei Ma, Qingyan Bai, Yixuan Li, Cheng Chen, Yanhong Zeng, Xing Zhu, Yujun Shen, Qifeng Chen

**Published:** 2025-12-18

**Categories:** cs.CV

**Abstract:**

We present WorldCanvas, a framework for promptable world events that enables rich, user-directed simulation by combining text, trajectories, and reference images. Unlike text-only approaches and existing trajectory-controlled image-to-video methods, our multimodal approach combines trajectories -- encoding motion, timing, and visibility -- with natural language for semantic intent and reference images for visual grounding of object identity, enabling the generation of coherent, controllable events that include multi-agent interactions, object entry/exit, reference-guided appearance and counterintuitive events. The resulting videos demonstrate not only temporal coherence but also emergent consistency, preserving object identity and scene despite temporary disappearance. By supporting expressive world events generation, WorldCanvas advances world models from passive predictors to interactive, user-shaped simulators. Our project page is available at: https://worldcanvas.github.io/.

**Analysis:**

好的，这是对论文“The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text”的全面中文摘要：

**论文题目：** The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text

**作者：** Hanlin Wang, Hao Ouyang, Qiuyu Wang, Yue Yu, Yihao Meng, Wen Wang, Ka Leong Cheng, Shuailei Ma, Qingyan Bai, Yixuan Li, Cheng Chen, Yanhong Zeng, Xing Zhu, Yujun Shen, Qifeng Chen

---

**摘要：**

**1. 研究问题/核心挑战：**
当前的世界模型（world models）在生成可控、语义丰富的事件方面存在局限性。现有的“可提示事件”（promptable events）方法主要依赖文本提示，这在处理复杂的时空动态、多主体交互以及精确的对象身份控制时显得不足。文本本身难以完全捕捉事件的“何时”、“何地”、“谁”以及“什么”等关键信息，尤其是在需要精细控制对象运动、出现/消失以及外观时。

**2. 主要创新点/方法贡献：**
本文提出了 **WorldCanvas**，一个创新的框架，用于生成可提示的世界事件。其核心贡献在于引入了一种**多模态提示范式**，该范式结合了三种互补的模态：

*   **轨迹（Trajectories）：** 编码了事件的“何时”和“何地”，通过点序列定义了对象的运动路径、速度（点间距）和可见性（是否出现/消失）。
*   **参考图像（Reference Images）：** 提供了“谁”的视觉基础，用于精确地指定对象的外观和身份，实现参考图像引导的生成。
*   **文本（Text）：** 描述了事件的“什么”，提供了高层语义意图、交互和因果关系。

WorldCanvas 的关键技术创新包括：
*   **数据策展流水线：** 构建了一个包含轨迹-视频-文本三元组的数据集，其中文本（动作描述）与轨迹紧密对齐，并提取了用于视觉基础的参考图像。
*   **轨迹注入（Trajectory Injection）：** 将轨迹信息通过高斯热力图和点 VAE 映射等方式注入到生成模型中，使模型能够遵循用户指定的运动路径。
*   **空间感知加权交叉注意力（Spatial-Aware Weighted Cross-Attention）：** 提出了一种新的注意力机制，用于在多主体场景中将文本描述与对应的轨迹进行精确对齐，解决了主体相似或动态出现时文本-轨迹匹配的挑战。
*   **直观的用户界面：** 设计了一个用户友好的接口，允许用户方便地输入轨迹、参考图像和文本，实现对事件的精细化控制。

**3. 主要结果与意义：**
WorldCanvas 在生成可控、连贯且语义丰富的世界事件方面取得了显著成果。
*   **生成能力：** 能够生成包含多主体交互、对象进入/退出场景、参考图像引导的外观以及反直觉事件的视频。
*   **一致性：** 生成的视频不仅在时间上连贯，而且在对象身份和场景方面表现出涌现式的一致性，即使对象暂时消失后重新出现也能保持其身份和外观。
*   **控制精度：** 通过多模态提示，用户可以精确控制事件的“何时”、“何地”、“谁”和“什么”，实现了比纯文本方法更精细的控制。
*   **意义：** WorldCanvas 将世界模型从被动的预测者转变为**交互式的、用户塑造的模拟器**，为构建更高级、更具创造性的世界模型奠定了基础。其在物理合理性、因果推理和未来预测方面的能力也得到了初步验证。

**4. 提及的局限性：**
论文中提到，尽管 WorldCanvas 在许多方面表现出色，但在处理**极端复杂的空间变换或逻辑推理**时，有时仍会失败。例如，在相机进行大幅度旋转时，可能会出现模糊和不一致；在涉及复杂逻辑推理（如相机移开后物体状态的持续变化）的场景中，模型可能无法完全捕捉预期的物理行为。

**5. 未来研究方向：**
论文指出了未来研究的几个方向：
*   **复杂运动下的持续性：** 如何在剧烈和复杂的运动中保持场景和对象的一致性。
*   **视线外内容的逻辑推理：** 如何让模型在内容暂时不可见时，仍能进行正确的逻辑推理，并预测其后续状态。
*   **更高级的世界模型：** WorldCanvas 作为一种生成可控、语义丰富事件的框架，可以作为构建更先进、能够进行连贯、持久场景模拟的世界模型的基石。

**总体而言，** WorldCanvas 是一项重要的研究成果，它通过创新的多模态提示范式，极大地提升了生成可控、语义丰富的世界事件的能力，为实现真正交互式的、用户驱动的世界模拟开辟了新的道路。

**Key Findings:**

- We present WorldCanvas, a framework for promptable world events that enables rich, user-directed simulation by combining text, trajectories, and reference images.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16924v1)
- [arXiv](https://arxiv.org/abs/2512.16924v1)

---

<a id='2512.16922v1'></a>
## [Next-Embedding Prediction Makes Strong Vision Learners](https://arxiv.org/abs/2512.16922v1)

**Authors:** Sihan Xu, Ziqiao Ma, Wenhao Chai, Xuweiyi Chen, Weiyang Jin, Joyce Chai, Saining Xie, Stella X. Yu

**Published:** 2025-12-18

**Categories:** cs.CV

**Abstract:**

Inspired by the success of generative pretraining in natural language, we ask whether the same principles can yield strong self-supervised visual learners. Instead of training models to output features for downstream use, we train them to generate embeddings to perform predictive tasks directly. This work explores such a shift from learning representations to learning models. Specifically, models learn to predict future patch embeddings conditioned on past ones, using causal masking and stop gradient, which we refer to as Next-Embedding Predictive Autoregression (NEPA). We demonstrate that a simple Transformer pretrained on ImageNet-1k with next embedding prediction as its sole learning objective is effective - no pixel reconstruction, discrete tokens, contrastive loss, or task-specific heads. This formulation retains architectural simplicity and scalability, without requiring additional design complexity. NEPA achieves strong results across tasks, attaining 83.8% and 85.3% top-1 accuracy on ImageNet-1K with ViT-B and ViT-L backbones after fine-tuning, and transferring effectively to semantic segmentation on ADE20K. We believe generative pretraining from embeddings provides a simple, scalable, and potentially modality-agnostic alternative to visual self-supervised learning.

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Next-Embedding Prediction Makes Strong Vision Learners**

**1. 论文的主要贡献 (2-3句话的简洁总结):**

本研究提出了一种名为“Next-Embedding Predictive Autoregression (NEPA)”的新型自监督学习范式，将生成式预训练的成功经验迁移至计算机视觉领域。与传统的学习表示（representation learning）不同，NEPA直接训练模型生成嵌入（embeddings）以执行预测任务，具体而言，模型学习根据过去的图像块嵌入预测未来的图像块嵌入。这种方法在不依赖像素重建、离散化token、对比损失或特定任务头部的情况下，仅通过预测未来嵌入，就能够训练出强大的视觉学习器，并在ImageNet等基准测试中取得了优异的性能。

**2. 关键创新或方法论:**

*   **核心创新：从学习表示到学习模型 (Learning Representations to Learning Models):** 这是本研究最核心的理念转变。传统的自监督学习（如SimCLR, MoCo, DINO等）侧重于学习能够捕捉图像语义信息的“表示”，这些表示随后用于下游任务。而NEPA则直接将模型训练目标设定为“生成嵌入以执行预测任务”，即模型本身就是一个预测器，其输出的嵌入具有直接的可预测性。
*   **Next-Embedding Predictive Autoregression (NEPA):**
    *   **预测未来嵌入 (Predicting Future Embeddings):** 模型被训练来预测序列中下一个图像块的嵌入，给定前面一系列图像块的嵌入。这类似于自然语言处理中的自回归模型（如GPT系列），但作用于视觉嵌入序列。
    *   **因果掩码 (Causal Masking):** 在预测时，模型只能访问过去的嵌入信息，而不能看到未来的信息，这确保了预测的“因果性”。
    *   **停止梯度 (Stop Gradient):** 摘要中提到“stop gradient”，这通常用于在训练过程中阻止梯度流向某些部分，以避免信息泄露或优化问题。在NEPA中，这可能意味着在计算损失时，目标嵌入（即未来的真实嵌入）的梯度不会反向传播到生成这些目标嵌入的模型部分，或者用于防止模型过度拟合于生成目标嵌入本身。
    *   **仅使用嵌入进行预测:** NEPA不依赖于像素级别的重建（如MAE），也不依赖于将图像块量化为离散token（如VQ-VAE, BEiT），更不使用对比学习的损失函数。这极大地简化了模型设计和训练过程。

**3. 对该领域的潜在影响:**

*   **简化自监督学习范式:** NEPA提供了一种极其简洁的自监督学习框架，消除了许多现有方法中常见的复杂组件（如负样本对、大批量数据、多头结构等）。这使得模型更容易实现和扩展。
*   **提升模型泛化能力:** 通过直接学习预测嵌入的能力，模型可能能够学习到更具鲁棒性和泛化性的视觉特征，从而在各种下游任务中表现出色。
*   **推动生成式预训练在视觉领域的应用:** 成功将自然语言处理中生成式预训练的强大能力引入计算机视觉，为未来视觉模型的设计提供了新的思路和方向。
*   **探索新的模型架构和训练目标:** NEPA的成功可能会激发研究人员探索更多基于生成式预测的自监督学习方法，以及更轻量级、更高效的视觉模型。
*   **潜在的模态无关性:** 摘要中提到“potentially modality-agnostic”，暗示这种基于嵌入预测的生成式预训练方法可能不仅限于视觉，也可能适用于其他模态（如音频、文本），甚至多模态融合。

**4. 可能受益于此研究的相关领域或应用:**

*   **基础视觉模型预训练:** NEPA可以作为构建强大基础视觉模型（Foundation Models）的新方法，这些模型可以为各种下游任务提供强大的起点。
*   **图像识别与分类:** 如摘要所示，在ImageNet等数据集上微调后表现优异。
*   **语义分割、目标检测等密集预测任务:** 摘要提到在ADE20K上的有效迁移，表明其在像素级理解任务上的潜力。
*   **视频理解:** 将NEPA扩展到视频序列，预测未来帧的嵌入，可能对视频预测、动作识别等任务有益。
*   **3D视觉:** 预测点云或体素的嵌入，可能有助于3D场景理解和生成。
*   **多模态学习:** 如果其模态无关性得到证实，将极大地促进跨模态理解和生成。
*   **低资源场景下的模型训练:** 简洁的训练范式和对复杂组件的依赖减少，可能使其在数据量有限的情况下更具优势。

**5. 从摘要中可以推断出的局限性:**

*   **“未来”的定义和粒度:** 摘要中提到“future patch embeddings”。“未来”具体指多远的未来？“patch embeddings”的粒度如何？这些细节会影响模型的学习内容和性能。例如，预测非常近的未来嵌入可能更侧重于局部纹理，而预测更远的未来嵌入则可能需要更强的全局语义理解。
*   **“停止梯度”的具体实现和影响:** 摘要中仅提及“stop gradient”，但其具体实现方式（例如，是应用于目标嵌入还是其他部分）以及它对训练动态和最终性能的影响，需要进一步的实验验证。
*   **计算效率和内存需求:** 虽然摘要强调了“scalability”，但对于非常大的模型和高分辨率图像，预测大量嵌入的计算量和内存需求可能仍然是一个挑战。
*   **对特定任务的适应性:** 尽管摘要声称迁移效果好，但对于某些与“预测未来嵌入”目标差异较大的下游任务，其适应性仍需进一步评估。例如，一些需要精细局部特征的任务，可能需要额外的微调策略。
*   **与现有方法的直接比较:** 摘要提供了在ImageNet上的性能数据，但并未直接与当前最先进的自监督学习方法（如MAE, DINOv2等）进行详细的横向比较，例如在训练时间和计算资源消耗上的对比。
*   **理论解释的深度:** 摘要主要侧重于方法和实验结果，对于“为什么”这种方法能够如此有效，其背后的理论解释可能还需要更深入的研究。

总而言之，这篇论文提出的NEPA方法是一个非常令人兴奋的进展，它通过一种新颖的生成式预测范式，极大地简化了自监督视觉学习的流程，并取得了令人印象深刻的性能。其潜在的模态无关性和普适性，预示着它可能成为未来视觉模型预训练的重要方向。

**Key Findings:**

- We demonstrate that a simple Transformer pretrained on ImageNet-1k with next embedding prediction as its sole learning objective is effective - no pixel reconstruction, discrete tokens, contrastive loss, or task-specific heads.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16922v1)
- [arXiv](https://arxiv.org/abs/2512.16922v1)

---

<a id='2512.16920v1'></a>
## [EasyV2V: A High-quality Instruction-based Video Editing Framework](https://arxiv.org/abs/2512.16920v1)

**Authors:** Jinjie Mai, Chaoyang Wang, Guocheng Gordon Qian, Willi Menapace, Sergey Tulyakov, Bernard Ghanem, Peter Wonka, Ashkan Mirzaei

**Published:** 2025-12-18

**Categories:** cs.CV, cs.AI

**Abstract:**

While image editing has advanced rapidly, video editing remains less explored, facing challenges in consistency, control, and generalization. We study the design space of data, architecture, and control, and introduce \emph{EasyV2V}, a simple and effective framework for instruction-based video editing. On the data side, we compose existing experts with fast inverses to build diverse video pairs, lift image edit pairs into videos via single-frame supervision and pseudo pairs with shared affine motion, mine dense-captioned clips for video pairs, and add transition supervision to teach how edits unfold. On the model side, we observe that pretrained text-to-video models possess editing capability, motivating a simplified design. Simple sequence concatenation for conditioning with light LoRA fine-tuning suffices to train a strong model. For control, we unify spatiotemporal control via a single mask mechanism and support optional reference images. Overall, EasyV2V works with flexible inputs, e.g., video+text, video+mask+text, video+mask+reference+text, and achieves state-of-the-art video editing results, surpassing concurrent and commercial systems. Project page: https://snap-research.github.io/easyv2v/

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：EasyV2V: A High-quality Instruction-based Video Editing Framework**

**1. 论文的主要贡献 (2-3句话总结)**

EasyV2V 提出了一种新颖且高效的框架，用于实现基于文本指令的视频编辑。该框架通过创新的数据构建策略、简化的模型设计以及统一的时空控制机制，显著提升了视频编辑的质量、一致性和可控性，并在多项评估中超越了现有技术和商业系统。

**2. 关键创新或方法论**

EasyV2V 的核心创新在于其对视频编辑设计空间的系统性研究和整合，具体体现在以下几个方面：

*   **数据构建策略 (Data Design):**
    *   **利用现有专家模型及其快速逆向操作 (Composing existing experts with fast inverses):** 这是一种巧妙的数据增强方法，通过组合已有的图像编辑模型（如风格迁移、对象替换等）及其逆向操作，生成大量高质量的视频编辑对。这解决了直接获取大规模、多样化的视频编辑数据困难的问题。
    *   **单帧监督和伪对 (Single-frame supervision and pseudo pairs):** 将图像编辑对提升到视频编辑，通过单帧的监督信息，并利用共享的仿射运动来生成伪视频对，这是一种高效利用现有图像编辑能力的方法。
    *   **挖掘带密集字幕的视频片段 (Mining dense-captioned clips):** 从现有数据集中寻找带有详细描述的视频片段，用于构建视频编辑对，增加了数据的丰富性和指令的准确性。
    *   **添加过渡监督 (Transition supervision):** 专门训练模型理解编辑是如何在时间上展开的，这对于生成平滑自然的视频编辑至关重要，解决了视频编辑中常见的突兀感问题。

*   **模型设计 (Architecture Design):**
    *   **利用预训练文本到视频模型的编辑能力 (Pretrained text-to-video models possess editing capability):** 作者发现，现有的文本到视频生成模型本身就蕴含了强大的编辑潜力，无需从头设计复杂的模型。
    *   **简化的条件输入和轻量级微调 (Simple sequence concatenation for conditioning with light LoRA fine-tuning):** 采用简单的序列拼接方式来整合文本指令，并利用 LoRA (Low-Rank Adaptation) 等轻量级微调技术，即可训练出高性能的模型。这大大降低了训练成本和模型复杂度，使得模型更易于部署和使用。

*   **控制机制 (Control Mechanism):**
    *   **统一的时空掩码机制 (Unify spatiotemporal control via a single mask mechanism):** 提出了一种灵活的掩码机制，能够同时控制编辑的空间区域和时间范围，实现了精细化的编辑控制。
    *   **支持可选的参考图像 (Optional reference images):** 允许用户提供参考图像，以指导编辑的风格或内容，进一步增强了编辑的灵活性和用户的主动性。

**3. 对该领域的潜在影响**

EasyV2V 的研究对视频编辑领域具有重要的潜在影响：

*   **降低视频编辑的门槛:** 通过简化的模型和灵活的输入方式，使得非专业用户也能更容易地进行高质量的视频编辑，推动了视频创作的民主化。
*   **推动指令驱动的视频内容生成:** 这种基于文本指令的编辑方式是未来视频内容生成的重要方向，EasyV2V 的成功将激励更多研究者探索更强大的指令理解和执行能力。
*   **促进视频编辑技术的商业化应用:** 其在性能上的突破和易用性将加速相关技术在短视频平台、内容创作工具、电影后期制作等领域的商业化落地。
*   **为未来视频生成模型的设计提供新思路:** 其对预训练模型编辑能力的挖掘以及轻量级微调的有效性，为后续视频生成模型的研发提供了宝贵的经验和方向。

**4. 可能受益的相关领域或应用**

*   **内容创作与社交媒体:** 极大地赋能短视频创作者，使其能够快速、高效地对视频内容进行个性化编辑和风格化处理。
*   **电影与电视后期制作:** 为专业后期制作人员提供更强大、更灵活的工具，加速特效制作和画面调整流程。
*   **虚拟现实 (VR) 和增强现实 (AR) 内容生成:** 能够更便捷地编辑和生成沉浸式视频内容。
*   **教育与培训:** 制作更具吸引力和互动性的教学视频。
*   **个性化视频广告:** 快速生成符合特定用户需求的广告视频。
*   **数字人与虚拟形象:** 为虚拟形象的动画和表情编辑提供更精细的控制。

**5. 从摘要中可以推断出的局限性**

尽管摘要中强调了 EasyV2V 的优势，但仍可以推断出一些潜在的局限性：

*   **对数据质量的依赖:** 尽管作者提出了多种数据构建策略，但最终编辑效果仍可能受到原始视频质量、指令清晰度以及生成数据质量的影响。
*   **复杂场景下的挑战:** 对于包含大量动态物体、复杂光照变化或快速运动的场景，模型的编辑效果和一致性可能仍会面临挑战。
*   **指令的理解深度:** 虽然支持文本指令，但对于非常抽象、模糊或需要高度语义理解的编辑指令，模型可能仍难以完美执行。
*   **计算资源需求:** 尽管模型设计简化，但视频处理本身通常需要较高的计算资源，尤其是在处理高分辨率或长视频时。
*   **对特定编辑类型的泛化能力:** 摘要提到“state-of-the-art video editing results”，但具体在哪些类型的编辑任务上表现优异，以及对全新、未见过编辑类型的泛化能力，需要进一步的实验验证。例如，对于需要深度语义理解的“让视频中的猫变成狗”这类任务，其效果如何仍是未知数。
*   **“Fast inverses” 的局限性:** 尽管利用了“fast inverses”，但这些逆向操作的质量和效率可能并非完美，可能会引入伪影或限制编辑的自由度。

总而言之，EasyV2V 是一项令人兴奋的研究，它通过系统性的方法解决了视频编辑中的关键挑战，并有望对该领域产生深远影响。其创新的数据构建和简化的模型设计是其成功的关键。

**Key Findings:**

- Overall, EasyV2V works with flexible inputs, e.g., video+text, video+mask+text, video+mask+reference+text, and achieves state-of-the-art video editing results, surpassing concurrent and commercial systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16920v1)
- [arXiv](https://arxiv.org/abs/2512.16920v1)

---

<a id='2512.16919v1'></a>
## [DVGT: Driving Visual Geometry Transformer](https://arxiv.org/abs/2512.16919v1)

**Authors:** Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, Jiwen Lu

**Published:** 2025-12-18

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Perceiving and reconstructing 3D scene geometry from visual inputs is crucial for autonomous driving. However, there still lacks a driving-targeted dense geometry perception model that can adapt to different scenarios and camera configurations. To bridge this gap, we propose a Driving Visual Geometry Transformer (DVGT), which reconstructs a global dense 3D point map from a sequence of unposed multi-view visual inputs. We first extract visual features for each image using a DINO backbone, and employ alternating intra-view local attention, cross-view spatial attention, and cross-frame temporal attention to infer geometric relations across images. We then use multiple heads to decode a global point map in the ego coordinate of the first frame and the ego poses for each frame. Unlike conventional methods that rely on precise camera parameters, DVGT is free of explicit 3D geometric priors, enabling flexible processing of arbitrary camera configurations. DVGT directly predicts metric-scaled geometry from image sequences, eliminating the need for post-alignment with external sensors. Trained on a large mixture of driving datasets including nuScenes, OpenScene, Waymo, KITTI, and DDAD, DVGT significantly outperforms existing models on various scenarios. Code is available at https://github.com/wzzheng/DVGT.

**Analysis:**

好的，这是一篇关于“DVGT: Driving Visual Geometry Transformer”论文的全面中文摘要，其中包含了您要求的各个部分：

**论文题目：** DVGT: Driving Visual Geometry Transformer

**作者：** Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, Jiwen Lu

---

**全面摘要**

**1. 研究问题/核心挑战：**

自动驾驶的核心任务之一是准确感知和重建三维场景几何。然而，现有方法在以下方面存在不足：
*   **缺乏针对自动驾驶的通用模型：** 大多数模型难以适应不同的驾驶场景和多变的相机配置。
*   **对精确相机参数的依赖：** 传统方法通常需要精确的相机内外参，这限制了其灵活性和可扩展性。
*   **几何信息不完整或精度不足：** 一些方法只能预测2.5D几何或存在量化误差，难以实现精细的场景理解。
*   **缺乏端到端的度量尺度几何预测：** 许多模型需要额外的后处理（如与LiDAR对齐）才能获得度量尺度的几何信息。

**2. 主要创新点/方法贡献：**

本文提出了**Driving Visual Geometry Transformer (DVGT)**，一个专为自动驾驶设计的、端到端的视觉几何Transformer模型，旨在解决上述挑战。其核心创新点包括：

*   **全局密集3D点图重建：** DVGT能够从一系列无序的多视图图像中直接预测一个全局、密集且度量尺度的3D点图，提供连续且高保真的场景几何表示。
*   **3D先验自由（Prior-Free）设计：** 模型完全摆脱了对显式3D几何先验（如精确相机参数）的依赖，通过纯粹的数据驱动方式学习几何信息，使其能够灵活适应各种相机配置。
*   **高效的空间-时间注意力机制：** DVGT采用了一种分解式的注意力机制，包括**视图内局部注意力（Intra-View Local Attention）**、**跨视图空间注意力（Cross-View Spatial Attention）**和**跨帧时间注意力（Cross-Frame Temporal Attention）**。这种设计在保持有效几何信息融合的同时，显著提高了计算效率和推理速度，克服了传统全局注意力机制的计算瓶颈。
*   **统一的Ego-centric坐标系：** 模型将3D点图预测在**参考帧的Ego-centric坐标系**下，并将预测的位姿也转换为相对于参考帧的Ego位姿。这种设计使得几何表示对相机焦距、位姿和视图数量具有不变性，是实现通用自动驾驶感知模型的关键。
*   **多任务联合预测：** DVGT不仅预测3D点图，还**联合预测每帧的Ego位姿**，使其成为一个全面的视觉几何模型。
*   **大规模、多样化的训练数据集构建：** 为了训练一个具有强大泛化能力的模型，作者构建了一个包含nuScenes、OpenScene、Waymo、KITTI和DDAD等多个公开数据集的混合数据集，并开发了一个鲁棒的流程来生成高质量的密集3D点图伪标签。

**3. 主要结果与意义：**

*   **卓越的3D几何重建性能：** DVGT在多个公开数据集上，包括KITTI、nuScenes、Waymo和OpenScene，均显著优于现有的通用视觉几何模型和专门的驾驶几何模型。在度量尺度3D点图重建和射线深度估计方面，DVGT取得了最先进的性能。
*   **准确的Ego位姿估计：** DVGT在Ego位姿估计任务上也表现出色，在OpenScene和DDAD数据集上取得了领先结果，并在nuScenes和Waymo上与现有模型相当，证明了其作为综合视觉几何模型的有效性。
*   **强大的泛化能力：** 由于其3D先验自由设计和在多样化数据集上的训练，DVGT展现出对不同相机配置和驾驶场景的强大适应性，克服了传统方法的局限性。
*   **端到端、无需后处理：** DVGT能够直接从图像序列预测度量尺度的几何信息，无需依赖外部传感器（如LiDAR）进行后对齐，大大简化了部署流程。
*   **高效性：** 分解式注意力机制使得模型在保持高性能的同时，实现了更快的推理速度。

**4. 提及的局限性：**

*   **Waymo数据集性能相对较低：** 作者提到在Waymo数据集上的性能不如其他数据集，这归因于训练时对该数据集的采样权重设置，可能未能充分体现其数据量和多样性。
*   **KITTI数据集位姿估计略低：** 在KITTI数据集上，DVGT的位姿估计性能略低于其他数据集，作者认为这可能与KITTI数据集的双目相机设置限制了3D和Ego运动的理解有关。
*   **尺度缩放的挑战：** 在处理远距离场景时，直接回归大数值的3D坐标可能导致训练不稳定。虽然通过线性缩放（10x）解决了这个问题，但过大的线性缩放（100x）或非线性缩放（arcsinh）可能导致精度下降或几何结构失真。

**5. 潜在的未来研究方向：**

*   **优化数据集采样策略：** 针对Waymo等数据量大但可能存在采样不均的数据集，进一步优化训练时的采样权重，以提升在这些数据集上的性能。
*   **探索更精细的尺度处理：** 研究更鲁棒的尺度处理方法，以应对自动驾驶场景中极端动态范围的几何信息，同时避免精度损失。
*   **进一步提升Ego位姿估计精度：** 针对特定场景（如KITTI）或更具挑战性的位姿估计任务，探索更先进的位姿估计模块或训练策略。
*   **扩展到更广泛的下游任务：** 将DVGT预测的密集、度量尺度的3D几何信息应用于更广泛的自动驾驶下游任务，如路径规划、目标检测和跟踪等，并评估其带来的提升。
*   **实时性优化：** 尽管DVGT已经实现了高效性，但对于对实时性要求极高的自动驾驶系统，可以进一步探索模型压缩、量化或更高效的网络结构来提升推理速度。

**总结：**

DVGT通过引入创新的空间-时间注意力机制和3D先验自由设计，成功构建了一个能够从无序多视图图像中端到端预测度量尺度全局3D点图和Ego位姿的Transformer模型。该模型在自动驾驶场景下展现出卓越的几何重建和位姿估计能力，并克服了传统方法对相机参数的依赖，为构建更鲁棒、更通用的自动驾驶感知系统奠定了坚实基础。

**Key Findings:**

- To bridge this gap, we propose a Driving Visual Geometry Transformer (DVGT), which reconstructs a global dense 3D point map from a sequence of unposed multi-view visual inputs.
- Trained on a large mixture of driving datasets including nuScenes, OpenScene, Waymo, KITTI, and DDAD, DVGT significantly outperforms existing models on various scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16919v1)
- [arXiv](https://arxiv.org/abs/2512.16919v1)

---

<a id='2512.16921v1'></a>
## [Differences That Matter: Auditing Models for Capability Gap Discovery and Rectification](https://arxiv.org/abs/2512.16921v1)

**Authors:** Qihao Liu, Chengzhi Mao, Yaojie Liu, Alan Yuille, Wen-Sheng Chu

**Published:** 2025-12-18

**Categories:** cs.CV, cs.AI

**Abstract:**

Conventional evaluation methods for multimodal LLMs (MLLMs) lack interpretability and are often insufficient to fully disclose significant capability gaps across models. To address this, we introduce AuditDM, an automated framework that actively discovers and rectifies MLLM failure modes by auditing their divergence. AuditDM fine-tunes an MLLM as an auditor via reinforcement learning to generate challenging questions and counterfactual images that maximize disagreement among target models. Once trained, the auditor uncovers diverse, interpretable exemplars that reveal model weaknesses and serve as annotation-free data for rectification. When applied to SoTA models like Gemma-3 and PaliGemma-2, AuditDM discovers more than 20 distinct failure types. Fine-tuning on these discoveries consistently improves all models across 16 benchmarks, and enables a 3B model to surpass its 28B counterpart. Our results suggest that as data scaling hits diminishing returns, targeted model auditing offers an effective path to model diagnosis and improvement.

**Analysis:**

好的，这是对论文“Differences That Matter: Auditing Models for Capability Gap Discovery and Rectification”的全面中文摘要：

**论文题目：** Differences That Matter: Auditing Models for Capability Gap Discovery and Rectification
**作者：** Qihao Liu, Chengzhi Mao, Yaojie Liu, Alan Yuille, Wen-Sheng Chu

**摘要：**

这篇论文提出了一种名为 **AuditDM** 的自动化框架，旨在解决当前多模态大语言模型（MLLMs）评估方法在解释性不足和难以充分揭示模型能力差距方面的问题。AuditDM 通过“审计”模型间的差异来主动发现和纠正 MLLMs 的失效模式。

**1. 研究问题/核心挑战：**

*   **现有评估方法的局限性：** 传统的 MLLMs 评估方法（如基准测试）通常是封闭集、固定知识范围的，难以揭示模型在长尾、细微能力上的差异，并且缺乏解释性，使得理解模型为何失败变得困难。
*   **模型能力差距的诊断与修复：** 如何系统性地发现 MLLMs 的细微能力差距和失效模式，并利用这些发现来改进模型，是实际应用中的关键挑战。

**2. 关键创新/方法贡献：**

*   **模型审计（Model Auditing）范式：** 论文引入了“模型审计”这一新范式，旨在系统性地发现模型能力差距并诊断其弱点。
*   **AuditDM 框架：**
    *   **审计模型（Auditor Model）：** 论文提出训练一个 MLLM 作为“审计模型”，通过强化学习（GRPO）进行微调。
    *   **生成挑战性样本：** 该审计模型能够生成具有挑战性的问题-图像对（包括反事实图像），这些样本旨在最大化目标模型与参考模型（或模型集合）之间的响应差异，从而暴露目标模型的弱点。
    *   **无标注数据生成：** AuditDM 生成的失效示例是无标注的，可以直接用于模型的纠正和改进。
    *   **解释性与诊断性：** AuditDM 生成的失效模式示例具有多样性和可解释性，能够提供对模型弱点的深入理解。
*   **两种纠正策略：**
    *   **增强标注数据：** 将审计生成的样本添加到原始训练数据中。
    *   **自举无标注数据：** 利用审计模型生成伪标签数据，进行迭代式训练和改进。

**3. 主要结果与意义：**

*   **高效的失效模式发现：** AuditDM 在发现模型弱点方面比基线方法（仅依赖提示工程）效率高得多，成功率显著提升。
*   **揭示细微能力差距：** 在 PaliGemma2 模型上，AuditDM 发现了超过 20 种不同的失效类型，并揭示了 28B 模型在某些任务上（如幻觉规避、计数、颜色识别）反而不如 3B 模型。
*   **显著的模型性能提升：**
    *   通过 AuditDM 生成的数据进行微调，在 16 个基准测试上，所有模型都获得了持续的性能提升。
    *   一个 3B 模型在经过 AuditDM 改进后，甚至超越了其 28B 的对应模型。
    *   在 Gemma3-4B 模型上，AuditDM 在多个基准测试上带来了显著的性能提升，缩小了与更大模型的差距，并在某些任务上超越了 12B 模型。
*   **意义：** 论文表明，随着数据规模化效应递减，有针对性的模型审计是诊断和改进 MLLMs 的有效途径，为模型持续学习和改进提供了新的方向。

**4. 论文中提到的局限性：**

*   **图像生成限制：** AuditDM 在生成用于接地（grounding）和分割（segmentation）任务的探针问题时，需要具有密集标注的图像。此外，对于文本/图表导向的 OCR 任务，合成具有密集文本和复杂图表的图像存在困难。
*   **计算复杂度：** 整个 AuditDM 流程（包括审计模型训练、数据生成和目标模型微调）需要大量的计算资源，例如生成 Gemma3-4B 的数据集需要数天时间。

**5. 潜在的未来研究方向：**

*   **改进图像生成：** 通过自举伪标签和使用更强的视觉标注器，以及开发文本/图表专用生成器来克服图像生成方面的限制。
*   **优化计算效率：** 探索更高效的数据生成和训练策略，以降低计算成本。
*   **更广泛的应用：** 将 AuditDM 应用于更多类型的多模态模型和任务，探索其在不同场景下的有效性。
*   **更强的审计模型：** 研究如何构建更强大的审计模型，以发现更复杂、更隐蔽的失效模式。

**总结：**

AuditDM 框架通过引入“模型审计”这一创新范式，利用强化学习训练的 MLLM 作为审计者，能够高效、系统地发现和解释 MLLMs 的能力差距和失效模式。其生成的无标注、针对性训练数据能够显著提升模型的性能，并为模型提供更可靠的评估。这项工作为理解和改进日益复杂的多模态大模型提供了一条有前景的路径。

**Key Findings:**

- To address this, we introduce AuditDM, an automated framework that actively discovers and rectifies MLLM failure modes by auditing their divergence.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16921v1)
- [arXiv](https://arxiv.org/abs/2512.16921v1)

---

<a id='2512.16918v1'></a>
## [AdaTooler-V: Adaptive Tool-Use for Images and Videos](https://arxiv.org/abs/2512.16918v1)

**Authors:** Chaoyang Wang, Kaituo Feng, Dongyang Chen, Zhongyu Wang, Zhixun Li, Sicheng Gao, Meng Meng, Xu Zhou, Manyuan Zhang, Yuzhang Shang, Xiangyu Yue

**Published:** 2025-12-18

**Categories:** cs.CV

**Abstract:**

Recent advances have shown that multimodal large language models (MLLMs) benefit from multimodal interleaved chain-of-thought (CoT) with vision tool interactions. However, existing open-source models often exhibit blind tool-use reasoning patterns, invoking vision tools even when they are unnecessary, which significantly increases inference overhead and degrades model performance. To this end, we propose AdaTooler-V, an MLLM that performs adaptive tool-use by determining whether a visual problem truly requires tools. First, we introduce AT-GRPO, a reinforcement learning algorithm that adaptively adjusts reward scales based on the Tool Benefit Score of each sample, encouraging the model to invoke tools only when they provide genuine improvements. Moreover, we construct two datasets to support training: AdaTooler-V-CoT-100k for SFT cold start and AdaTooler-V-300k for RL with verifiable rewards across single-image, multi-image, and video data. Experiments across twelve benchmarks demonstrate the strong reasoning capability of AdaTooler-V, outperforming existing methods in diverse visual reasoning tasks. Notably, AdaTooler-V-7B achieves an accuracy of 89.8\% on the high-resolution benchmark V*, surpassing the commercial proprietary model GPT-4o and Gemini 1.5 Pro. All code, models, and data are released.

**Analysis:**

好的，这是对论文“AdaTooler-V: Adaptive Tool-Use for Images and Videos”的全面中文摘要：

**论文题目：** AdaTooler-V: Adaptive Tool-Use for Images and Videos

**作者：** Chaoyang Wang, Kaituo Feng, Dongyang Chen, Zhongyu Wang, Zhixun Li, Sicheng Gao, Meng Meng, Xu Zhou, Manyuan Zhang, Yuzhang Shang, Xiangyu Yue

**摘要：**

**1. 研究问题/核心挑战：**
该论文主要解决了当前多模态大语言模型（MLLMs）在处理图像和视频任务时，存在“盲目使用工具”的问题。现有模型倾向于不加区分地调用视觉工具，即使在问题本身不需要工具的情况下，这不仅增加了推理开销，还可能降低模型性能。核心研究问题是如何让 MLLMs 能够自适应地决定何时真正需要使用视觉工具，从而在保证性能的同时优化推理效率。

**2. 主要创新点/方法贡献：**
*   **AdaTooler-V 模型：** 提出了一种名为 AdaTooler-V 的 MLLM，其核心能力是实现自适应的工具使用。模型能够判断视觉问题是否真正需要工具，并据此选择文本链式思考（CoT）或多模态交错 CoT（包含视觉工具）。
*   **AT-GRPO 算法：** 引入了一种名为 AT-GRPO（Adaptive Tool-use GRPO）的强化学习算法。该算法通过计算每个样本的“工具效益得分”（Tool Benefit Score, AS），动态调整奖励函数。只有当工具使用能带来实际性能提升时，模型才会被奖励，否则会受到惩罚，从而鼓励模型仅在必要时使用工具。
*   **新数据集：** 构建了两个大规模数据集：AdaTooler-V-CoT-100k 用于监督微调（SFT）的冷启动，以及 AdaTooler-V-300k 用于强化学习（RL）训练。这些数据集涵盖了单图像、多图像和视频等多种模态，以及数学、逻辑推理、空间理解等多样化的视觉推理任务。
*   **两阶段训练框架：** 采用 SFT 和 RLVR（Reinforcement Learning with Verifiable Rewards）的两阶段训练策略。SFT 阶段通过多轮工具交互轨迹建立丰富的推理模式和行为先验，RLVR 阶段则利用 AT-GRPO 算法进一步优化模型的推理策略，使其能够自主学习更有效的工具使用方式。

**3. 主要结果与意义：**
*   **性能提升：** 在十二个基准测试上的实验表明，AdaTooler-V 在各种视觉推理任务中展现出强大的推理能力，显著优于现有方法。
*   **SOTA 表现：** AdaTooler-V-7B 模型在 V\* 高分辨率基准测试上取得了 89.8% 的准确率，超越了商业模型 GPT-4o 和 Gemini 1.5 Pro。
*   **效率优化：** 通过自适应工具使用，模型能够减少不必要的工具调用，从而降低推理开销，同时保持甚至提升性能。
*   **跨领域泛化：** AdaTooler-V 在 MME、MathVista、InfoVQA 等通用推理基准上也表现出色，显示出良好的跨领域泛化能力。
*   **对视觉理解的重要性：** 实验结果（如 Tab. 5）验证了视觉工具使用对于准确的多模态理解至关重要，尤其是在需要精细视觉细节或空间对应关系的任务中。

**4. 提及的局限性：**
*   **工具效益评估的单一参考模型：** 当前的工具效益（AS）评估依赖于单一参考模型，这可能导致对工具是否真正有益的评估存在偏差。
*   **奖励设计侧重于可验证任务：** 奖励设计主要针对多项选择和数值问答等可验证任务，对于开放式生成任务的适应性较差。
*   **数据集来源：** AdaTooler-V-300k 数据集主要来自公开基准，对真实世界中的长尾案例、噪声条件和跨领域场景的覆盖有限。

**5. 未来研究方向：**
*   **更鲁棒的工具效益评估：** 开发学习型效益估计器或利用模型集成来获得更准确的 AS 预测。
*   **支持开放式生成任务：** 引入学习型奖励模型、多模态判别器或对比学习信号，以更好地支持开放式生成任务。
*   **扩展数据集：** 增加真实世界样本、合成困难案例或采用领域自适应技术，以增强模型的泛化能力。

**总结：**
AdaTooler-V 论文提出了一种创新的 MLLM，通过引入自适应工具使用机制，解决了现有模型盲目调用视觉工具的问题。其核心贡献在于 AT-GRPO 算法和配套的数据集，使得模型能够智能地决定何时使用工具，从而在提高推理效率的同时，在多模态视觉推理任务上取得了显著的性能提升，甚至超越了顶尖的商业模型。该研究为开发更高效、更智能的多模态模型提供了重要方向。

**Key Findings:**

- To this end, we propose AdaTooler-V, an MLLM that performs adaptive tool-use by determining whether a visual problem truly requires tools.
- First, we introduce AT-GRPO, a reinforcement learning algorithm that adaptively adjusts reward scales based on the Tool Benefit Score of each sample, encouraging the model to invoke tools only when they provide genuine improvements.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16918v1)
- [arXiv](https://arxiv.org/abs/2512.16918v1)

---

<a id='2512.16913v1'></a>
## [Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation](https://arxiv.org/abs/2512.16913v1)

**Authors:** Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, Lu Qi

**Published:** 2025-12-18

**Categories:** cs.CV

**Abstract:**

In this work, we present a panoramic metric depth foundation model that generalizes across diverse scene distances. We explore a data-in-the-loop paradigm from the view of both data construction and framework design. We collect a large-scale dataset by combining public datasets, high-quality synthetic data from our UE5 simulator and text-to-image models, and real panoramic images from the web. To reduce domain gaps between indoor/outdoor and synthetic/real data, we introduce a three-stage pseudo-label curation pipeline to generate reliable ground truth for unlabeled images. For the model, we adopt DINOv3-Large as the backbone for its strong pre-trained generalization, and introduce a plug-and-play range mask head, sharpness-centric optimization, and geometry-centric optimization to improve robustness to varying distances and enforce geometric consistency across views. Experiments on multiple benchmarks (e.g., Stanford2D3D, Matterport3D, and Deep360) demonstrate strong performance and zero-shot generalization, with particularly robust and stable metric predictions in diverse real-world scenes. The project page can be found at: \href{https://insta360-research-team.github.io/DAP_website/} {https://insta360-research-team.github.io/DAP\_website/}

**Analysis:**

好的，这是对论文“Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation”的全面中文摘要：

**论文题目：** Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation (全景深度任意化：一个全景深度估计的基础模型)

**作者：** Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, Lu Qi

**摘要：**

这篇论文提出了一种名为 **DAP (Depth Any Panoramas)** 的全景深度估计基础模型，旨在实现跨越不同场景距离的泛化能力。研究的核心在于解决全景深度估计领域中数据规模不足、领域鸿沟（室内/室外、合成/真实）以及模型泛化能力有限等关键挑战。

**1. 研究问题/核心挑战：**

*   **全景深度估计的泛化性不足：** 现有的全景深度估计方法，无论是基于相对/尺度不变性还是统一的度量深度方法，在泛化到多样化的真实世界场景（尤其是室外）时都存在困难。
*   **数据规模和质量的限制：** 收集和标注大规模、高质量的全景深度数据成本高昂，是制约模型性能的关键因素。
*   **领域鸿沟：** 合成数据与真实数据、室内场景与室外场景之间存在显著的领域差异，影响模型的鲁棒性。

**2. 主要创新点/方法论贡献：**

*   **大规模数据引擎：** 论文构建了一个包含超过200万个全景图像的大规模数据集，整合了公共数据集（如Structured3D）、高质量的UE5模拟器生成数据（DAP-2M-Labeled）以及从网络收集的真实全景图像（DAP-2M-Unlabeled）。
*   **三阶段伪标签精炼流水线：** 为了有效利用海量无标签数据并弥合领域鸿沟，论文设计了一个创新的三阶段伪标签生成和精炼流程：
    *   **阶段1：场景不变性标注器训练 (Scene-Invariant Labeler Training)：** 在高质量的合成数据上训练一个标注器，使其能够跨越室内外场景学习到物理上一致的深度线索，为后续的伪标签生成提供良好初始化。
    *   **阶段2：真实性不变性标注器训练 (Realism-Invariant Labeler Training)：** 利用一个PatchGAN判别器来评估深度预测的质量，并选择高置信度的伪标签样本。然后，在包含合成数据和精炼后的伪标签数据的扩展数据集上训练一个真实性不变性标注器，使其能够适应真实世界的视觉变化。
    *   **阶段3：DAP模型训练：** 在所有标记数据和精炼后的伪标签数据上进行最终的DAP模型训练，实现大规模半监督学习。
*   **模型设计：**
    *   **DINOv3-Large骨干网络：** 利用强大的预训练视觉模型DINOv3-Large作为特征提取器，以获得优越的泛化能力。
    *   **即插即用范围掩码头 (Plug-and-play Range Mask Head)：** 引入一个能够根据不同距离阈值（10m, 20m, 50m, 100m）生成有效深度区域掩码的模块，以适应不同尺度的场景，并提高预测的鲁棒性和稳定性。
    *   **多损失优化：** 结合多种损失函数来提升模型性能，包括：
        *   **尺度不变性损失 (LSILog)：** 标准的尺度不变性损失。
        *   **密集保真度损失 (LDF)：** 通过将深度图分解为12个透视视图来增强局部细节和结构一致性。
        *   **梯度锐化损失 (Lgrad)：** 聚焦于边缘区域，以保留物体边界的清晰度。
        *   **法线损失 (Lnormal)：** 增强几何一致性。
        *   **点云损失 (Lpts)：** 进一步保证几何一致性。
        *   **掩码损失 (Lmask)：** 用于训练范围掩码头。
    *   **失真图 (Distortion Map)：** 用于补偿全景图像（ERP）投影中的像素几何畸变，确保梯度贡献在整个球形域内均衡。

**3. 主要结果与意义：**

*   **卓越的零样本泛化能力：** 在Stanford2D3D、Matterport3D和Deep360等多个室内外基准测试中，DAP模型在零样本设置下（无需在测试集上进行微调）取得了最先进的性能，显著优于现有方法。
*   **鲁棒且度量一致的深度预测：** DAP模型能够生成锐利的对象边界、平滑的全局几何，并在远距离和天空区域表现出优越的鲁棒性，尤其是在复杂的真实世界场景中。
*   **强大的尺度感知能力：** 模型能够准确地预测场景的绝对尺度，并且在不同尺度下保持一致性。
*   **有效弥合领域鸿沟：** 三阶段伪标签精炼流水线成功地减少了合成与真实、室内与室外数据之间的领域差异。
*   **基础模型潜力：** DAP模型作为一个基础模型，为全景深度估计领域提供了一个统一且强大的框架，为未来的研究奠定了基础。

**4. 局限性：**

*   论文中未明确提及明显的局限性，但可以推测，尽管模型在多种场景下表现出色，但在极端复杂或非常规的场景下，其性能仍可能受到影响。
*   虽然模型在零样本设置下表现优异，但对于特定领域的数据，通过微调可能仍能进一步提升性能。

**5. 潜在的未来研究方向：**

*   **更广泛的场景覆盖：** 探索更多样化的真实世界场景，包括极端天气、低光照、动态场景等，以进一步提升模型的鲁棒性。
*   **实时性提升：** 针对需要实时应用的场景，研究如何优化模型结构和推理速度。
*   **多模态融合：** 结合其他传感器信息（如RGB-D、LiDAR）或更丰富的语义信息，以进一步提高深度估计的准确性和鲁棒性。
*   **可解释性研究：** 深入分析模型在处理不同场景和几何结构时的决策过程，以增强其可解释性。
*   **动态场景下的全景深度估计：** 扩展到处理包含运动物体的动态全景场景。

总而言之，这篇论文通过构建大规模数据集和创新的伪标签精炼策略，并结合先进的模型设计，成功地开发了一个强大的全景深度估计基础模型DAP。该模型在泛化能力、度量准确性和鲁棒性方面取得了显著的突破，为全景深度估计领域的研究和应用开辟了新的可能性。

**Key Findings:**

- In this work, we present a panoramic metric depth foundation model that generalizes across diverse scene distances.
- To reduce domain gaps between indoor/outdoor and synthetic/real data, we introduce a three-stage pseudo-label curation pipeline to generate reliable ground truth for unlabeled images.

**Links:**

- [PDF](https://arxiv.org/pdf/2512.16913v1)
- [arXiv](https://arxiv.org/abs/2512.16913v1)

---

